#!/usr/bin/env python3
"""Build one hash-bound AQ4 P2 fidelity metrics input.

The active producer is a separate Rust diagnostic binary.  This adapter only
reads validated BF16/source and active-AQ4 full-vector artifacts and streams
one row at a time; it never starts a model or touches the production service.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
MAX_ROWS = 24
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_ROW_BYTES = 64 * 1024
HIDDEN_SIZE = 4096
VOCAB_SIZE = 248320
TOP_K = 10
F32_BYTES = 4
METRICS_SCHEMA = "ullm.aq4_p2_fidelity_calibration_metrics.v1"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROTOCOL = _load("aq4_fidelity_protocol", "generate-aq4-p2-fidelity-holdout.py")
VALIDATE_SPLIT = _load("aq4_fidelity_split_validator", "validate-aq4-p2-fidelity-holdout.py")
FULL_COMPARE = _load("aq4_fidelity_full_compare", "compare-qwen35-aq4-p2-calibration.py")
FULL_VALIDATE = _load("aq4_fidelity_full_validate", "validate-qwen35-aq4-p2-full-calibration.py")


class CaptureError(ValueError):
    pass


def _sha(path: Path, label: str, limit: int | None = None) -> str:
    if path.is_symlink() or not path.is_file():
        raise CaptureError(f"{label} must be a regular file")
    if limit is not None and path.stat().st_size > limit:
        raise CaptureError(f"{label} exceeds bounded size")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    if os.path.lexists(path):
        raise CaptureError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    with temporary.open("xb") as stream:
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        if len(encoded) > MAX_OUTPUT_BYTES:
            raise CaptureError("metrics output exceeds bounded size")
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise CaptureError(f"{label} is non-finite")
    return value


def _chunks(fd: int, offset: int, elements: int, chunk: int) -> Iterator[list[float]]:
    remaining = elements
    cursor = offset
    while remaining:
        count = min(remaining, chunk)
        raw = os.pread(fd, count * F32_BYTES, cursor)
        if len(raw) != count * F32_BYTES:
            raise CaptureError("vector sidecar ended before row boundary")
        yield list(struct.unpack(f"<{count}f", raw))
        cursor += len(raw)
        remaining -= count


def _stream_stats(reference: Iterator[list[float]], candidate: Iterator[list[float]], elements: int) -> dict[str, float | int]:
    ref_sq = candidate_sq = dot = delta_sq = 0.0
    max_abs = 0.0
    ref_seen = candidate_seen = 0
    for left, right in zip(reference, candidate):
        if len(left) != len(right):
            raise CaptureError("reference/candidate vector chunk differs")
        for ref, actual in zip(left, right):
            if not math.isfinite(ref) or not math.isfinite(actual):
                raise CaptureError("full-vector comparison encountered a non-finite value")
            ref = float(ref)
            actual = float(actual)
            delta = actual - ref
            ref_sq += ref * ref
            candidate_sq += actual * actual
            dot += ref * actual
            delta_sq += delta * delta
            max_abs = max(max_abs, abs(delta))
            ref_seen += 1
            candidate_seen += 1
    if ref_seen != elements or candidate_seen != elements:
        raise CaptureError("full-vector comparison element count differs")
    try:
        next(reference)
        raise CaptureError("reference vector stream contains surplus elements")
    except StopIteration:
        pass
    try:
        next(candidate)
        raise CaptureError("candidate vector stream contains surplus elements")
    except StopIteration:
        pass
    ref_norm = math.sqrt(ref_sq)
    candidate_norm = math.sqrt(candidate_sq)
    denom = max(ref_norm, 1e-30)
    cosine = dot / max(ref_norm * candidate_norm, 1e-30)
    return {
        "reference_norm_sq": _finite(ref_sq, "reference_norm_sq"),
        "candidate_norm_sq": _finite(candidate_sq, "candidate_norm_sq"),
        "dot": _finite(dot, "dot"),
        "delta_norm_sq": _finite(delta_sq, "delta_norm_sq"),
        "relative_l2": _finite(math.sqrt(delta_sq) / denom, "relative_l2"),
        "cosine": _finite(cosine, "cosine"),
        "max_abs": _finite(max_abs, "max_abs"),
        "elements": elements,
    }


def _load_split(split_root: Path, expected_split_sha: str, expected_policy_sha: str, expected_cases_sha: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str, str, str]:
    try:
        split_info = VALIDATE_SPLIT.validate(split_root)
    except Exception as error:
        raise CaptureError(f"split validation failed: {error}") from error
    manifest = json.loads((split_root / "split-manifest.json").read_text(encoding="utf-8"))
    policy = json.loads((split_root / "policy.json").read_text(encoding="utf-8"))
    cases_path = split_root / "calibration-cases.jsonl"
    rows = []
    seen: set[str] = set()
    raw = cases_path.read_bytes()
    if len(raw) > MAX_OUTPUT_BYTES:
        raise CaptureError("calibration-cases.jsonl exceeds bounded size")
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line or len(line) > MAX_ROW_BYTES:
            raise CaptureError(f"calibration row {line_number} is empty or oversized")
        try:
            row = json.loads(line, object_pairs_hook=PROTOCOL.pairs, parse_constant=PROTOCOL.no_constants)
        except (UnicodeError, json.JSONDecodeError, PROTOCOL.ProtocolError) as error:
            raise CaptureError(f"calibration row {line_number} is invalid: {error}") from error
        if not isinstance(row, dict) or row.get("case_id") in seen:
            raise CaptureError("calibration rows contain duplicate case_id")
        seen.add(row.get("case_id"))
        if row.get("subset") != "calibration" or row.get("step") != 0 or row.get("row_count") != 1 or row.get("cached_prefix_tokens") != 0 or row.get("generated_tokens") != 0 or row.get("prompt_tokens") != row.get("context_tokens"):
            raise CaptureError(f"calibration row {row.get('case_id')} violates full-context step-zero contract")
        rows.append(row)
    if len(rows) != MAX_ROWS:
        raise CaptureError(f"calibration rows must contain exactly {MAX_ROWS} rows")
    if split_info["split_manifest_sha256"] != expected_split_sha or split_info["policy_sha256"] != expected_policy_sha or _sha(cases_path, "calibration cases") != expected_cases_sha:
        raise CaptureError("split/policy/calibration SHA does not match the pinned execution contract")
    return split_info, manifest, rows, _sha(split_root / "split-manifest.json", "split manifest"), _sha(split_root / "policy.json", "policy"), _sha(cases_path, "calibration cases")


def _artifact(root: Path, expected_kind: str) -> dict[str, Any]:
    try:
        artifact = FULL_COMPARE.load_artifact(root)
    except Exception as error:
        raise CaptureError(f"{expected_kind} calibration artifact failed validation: {error}") from error
    kind = artifact["manifest"].get("oracle_kind")
    if kind != expected_kind:
        raise CaptureError(f"{expected_kind} artifact kind differs: {kind}")
    if artifact["nonfinite_rows"]:
        raise CaptureError(f"{expected_kind} artifact contains non-finite rows")
    manifest = artifact["manifest"]
    cases = manifest.get("cases", {})
    runtime = manifest.get("runtime", {})
    run = runtime.get("run", {}) if isinstance(runtime, dict) else {}
    row_count = len(artifact["rows"])
    if cases.get("row_count") != row_count or run.get("row_count") != row_count:
        raise CaptureError(f"{expected_kind} nested row_count binding differs")
    return artifact


def _row_key(row: dict[str, Any]) -> tuple[str, int]:
    return row.get("case_id"), row.get("step")


def capture(split_root: Path, source_root: Path, active_root: Path, output: Path, *, expected_split_sha: str, expected_policy_sha: str, expected_cases_sha: str, expected_served_sha: str, expected_package_sha: str, expected_worker_sha: str, expected_guard_sha: str, expected_device_architecture: str, expected_quantized_revision: str) -> dict[str, Any]:
    _split_info, split_manifest, split_rows, split_sha, policy_sha, cases_sha = _load_split(split_root, expected_split_sha, expected_policy_sha, expected_cases_sha)
    source = _artifact(source_root, "independent_source_full")
    active = _artifact(active_root, "aq4_target")
    parent = active["manifest"].get("parent_sampled_oracle", {})
    if parent.get("schema_version") == FULL_VALIDATE.SCHEMA:
        parent_path = Path(parent.get("path", ""))
        source_manifest_path = (source_root / "manifest.json").resolve()
        if parent_path.resolve() != source_manifest_path or _sha(parent_path, "active direct source parent manifest") != source["manifest_sha256"]:
            raise CaptureError("active direct source calibration parent binding differs")
    active_runtime = active["manifest"].get("runtime", {}).get("runtime", {})
    expected_active = {"served_model_manifest_sha256": expected_served_sha, "package_manifest_sha256": expected_package_sha, "worker_binary_sha256": expected_worker_sha, "guard_sha256": expected_guard_sha, "device": {"architecture": expected_device_architecture}}
    if active_runtime.get("served_model_manifest_sha256") != expected_active["served_model_manifest_sha256"] or active_runtime.get("package_manifest_sha256") != expected_active["package_manifest_sha256"] or active_runtime.get("worker_binary_sha256") != expected_active["worker_binary_sha256"] or active_runtime.get("guard_sha256") != expected_active["guard_sha256"] or active_runtime.get("device", {}).get("architecture") != expected_device_architecture or active_runtime.get("quantized_artifact_revision") != expected_quantized_revision:
        raise CaptureError("active artifact identity differs from the pinned execution contract")
    if active["manifest"]["identity"].get("model_id") != source["manifest"]["identity"].get("model_id") or active["manifest"]["identity"].get("model_revision") != source["manifest"]["identity"].get("model_revision") or active["manifest"]["identity"].get("tokenizer", {}).get("aggregate_sha256") != source["manifest"]["identity"].get("tokenizer", {}).get("aggregate_sha256") or active_runtime.get("upstream_model_revision") != source["manifest"]["identity"].get("model_revision") or active_runtime.get("tokenizer_aggregate_sha256") != source["manifest"]["identity"].get("tokenizer", {}).get("aggregate_sha256"):
        raise CaptureError("source/active upstream revision binding differs")
    if active_runtime.get("quantized_artifact_revision") == source["manifest"]["identity"].get("model_revision"):
        raise CaptureError("upstream and quantized artifact revisions must remain distinct")
    expected = {(row["case_id"], 0): row for row in split_rows}
    if set(source["rows"]) != set(expected) or set(active["rows"]) != set(expected):
        raise CaptureError("source/active artifacts must contain exactly the 24 calibration rows")
    source_manifest_sha = source["manifest_sha256"]
    active_manifest_sha = active["manifest_sha256"]
    rows: list[dict[str, Any]] = []
    with FULL_VALIDATE.stable_fd(source["hidden"], "source hidden") as (source_hidden, _), FULL_VALIDATE.stable_fd(source["logits"], "source logits") as (source_logits, _), FULL_VALIDATE.stable_fd(active["hidden"], "active hidden") as (active_hidden, _), FULL_VALIDATE.stable_fd(active["logits"], "active logits") as (active_logits, _):
        for key in sorted(expected):
            split_row = expected[key]
            left = source["rows"][key]
            right = active["rows"][key]
            # Runtime rows use the canonical context hash (JSON token array plus a
            # trailing newline), which is the split's context_token_ids_sha256.
            if left["input_token_ids_sha256"] != right["input_token_ids_sha256"] or left["input_token_ids_sha256"] != split_row["context_token_ids_sha256"]:
                raise CaptureError(f"input identity differs for {key}")
            if left["greedy_token_id"] is None or right["greedy_token_id"] is None:
                raise CaptureError(f"greedy token is unavailable for {key}")
            hidden = _stream_stats(_chunks(source_hidden, left["hidden"]["offset_bytes"], HIDDEN_SIZE, source["chunk_elements"]), _chunks(active_hidden, right["hidden"]["offset_bytes"], HIDDEN_SIZE, active["chunk_elements"]), HIDDEN_SIZE)
            logits = _stream_stats(_chunks(source_logits, left["logits"]["offset_bytes"], VOCAB_SIZE, source["chunk_elements"]), _chunks(active_logits, right["logits"]["offset_bytes"], VOCAB_SIZE, active["chunk_elements"]), VOCAB_SIZE)
            source_top = [item["token_id"] for item in left["topk"]]
            active_top = [item["token_id"] for item in right["topk"]]
            rows.append({
                "case_id": split_row["case_id"], "case_sha256": split_row["case_sha256"], "fixture_sha256": split_row["fixture_sha256"],
                "prompt_token_ids_sha256": split_row["prompt_token_ids_sha256"], "context_token_ids_sha256": split_row["context_token_ids_sha256"],
                "prompt_tokens": split_row["prompt_tokens"], "context_tokens": split_row["context_tokens"], "baseline_mode": split_row["baseline_mode"],
                "prefill_requested_m": split_row["prefill_requested_m"], "resolved_m": split_row["resolved_m"], "step": 0, "row_count": 1,
                "greedy": {"source": left["greedy_token_id"], "active": right["greedy_token_id"], "exact": left["greedy_token_id"] == right["greedy_token_id"]},
                "ordered_top10": {"source": source_top, "active": active_top, "exact": source_top == active_top, "overlap": len(set(source_top) & set(active_top)) / TOP_K},
                "metrics": {
                    "token_agreement_rate": float(left["greedy_token_id"] == right["greedy_token_id"]),
                    "topk_overlap_rate_k10": len(set(source_top) & set(active_top)) / TOP_K,
                    "logits_cosine": logits["cosine"], "logits_relative_l2": logits["relative_l2"],
                    "hidden_cosine": hidden["cosine"], "hidden_relative_l2": hidden["relative_l2"], "hidden_max_abs": hidden["max_abs"],
                    "bf16_top1_retained_in_aq4_top10_rate": float(left["greedy_token_id"] in active_top),
                },
                "raw": {"hidden": hidden, "logits": logits, "source_top1_retained_in_active_top10": left["greedy_token_id"] in active_top},
            })
    result = {
        "schema_version": METRICS_SCHEMA, "status": "ready_for_freeze", "subset": "calibration", "row_count": len(rows),
        "split_manifest_sha256": split_sha, "policy_sha256": policy_sha, "calibration_cases_sha256": cases_sha,
        "source_manifest_sha256": source_manifest_sha, "active_manifest_sha256": active_manifest_sha,
        "identity": {"split_manifest_sha256": split_sha, "policy_sha256": policy_sha, "calibration_cases_sha256": cases_sha, "source_manifest_sha256": source_manifest_sha, "active_manifest_sha256": active_manifest_sha, "source_identity": source["manifest"]["identity"], "active_identity": active["manifest"]["identity"]},
        "rows": rows,
    }
    _atomic_json(output, result)
    return {"status": "ok", "row_count": len(rows), "output": str(output), "split_manifest_sha256": split_sha, "policy_sha256": policy_sha, "source_manifest_sha256": source_manifest_sha, "active_manifest_sha256": active_manifest_sha}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--active", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split-manifest-sha256", required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--expected-calibration-cases-sha256", required=True)
    parser.add_argument("--expected-served-model-manifest-sha256", required=True)
    parser.add_argument("--expected-package-manifest-sha256", required=True)
    parser.add_argument("--expected-worker-binary-sha256", required=True)
    parser.add_argument("--expected-guard-sha256", required=True)
    parser.add_argument("--expected-device-architecture", required=True)
    parser.add_argument("--expected-quantized-artifact-revision", required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(capture(args.split_root, args.source, args.active, args.output, expected_split_sha=args.expected_split_manifest_sha256, expected_policy_sha=args.expected_policy_sha256, expected_cases_sha=args.expected_calibration_cases_sha256, expected_served_sha=args.expected_served_model_manifest_sha256, expected_package_sha=args.expected_package_manifest_sha256, expected_worker_sha=args.expected_worker_binary_sha256, expected_guard_sha=args.expected_guard_sha256, expected_device_architecture=args.expected_device_architecture, expected_quantized_revision=args.expected_quantized_artifact_revision), ensure_ascii=True, sort_keys=True))
        return 0
    except (CaptureError, OSError, ValueError) as error:
        print(f"Qwen3.5 AQ4 fidelity capture failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
