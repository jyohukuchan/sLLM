#!/usr/bin/env python3
"""Prepare and freeze the AQ4 P2 fidelity calibration/holdout contract.

This tool only reads the expanded manifest and prompt fixtures.  It never starts a model or
collects measurements.  ``split`` creates a deterministic, disjoint 24/24 split of the
48-case production-server representative profile; ``freeze`` consumes calibration metrics and
derives the pre-registered envelope for one later holdout evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

MAX_JSON = 64 * 1024 * 1024
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
SPLIT_SCHEMA = "ullm.aq4_p2_fidelity_split.v1"
POLICY_SCHEMA = "ullm.aq4_p2_fidelity_policy.v1"
METRICS_SCHEMA = "ullm.aq4_p2_fidelity_calibration_metrics.v1"
RECEIPT_SCHEMA = "ullm.aq4_p2_fidelity_freeze_receipt.v1"
FIXTURE_SCHEMA = "ullm.aq4_p2_case_fixture.v1"
INDEX_SCHEMA = "ullm.aq4_p2_fixture_index.v1"
EXPANDED_SCHEMA = "ullm.aq4_production_p2_expanded.v2"
ATTEMPT2_CASE_IDS = {"fixture-prompt-0", "fixture-prompt-1"}
ATTEMPT2_CONTEXT_HASHES = {
    "42ea52c2bbdb5aaf75c70220eb41d4b018e85f2eeda3d4e40a8ac130d2e96215",
    "6af1601e1b18925e5a59b8446901a0afd7e3e3f915646c56ee77b5e77b4d6249",
    "3bca9e0d24c5f880460202136254ac1e81afc6696682f351cceb8df1f2e79e6",
}
STRATA_FIELDS = ("prompt_tokens", "baseline_mode")
METRICS = {
    "token_agreement_rate": {"role": "promotion", "direction": "higher", "aggregation": "wilson_lower_one_sided", "confidence_level": 0.95, "margin": None, "relative_margin": None, "absolute_floor": None, "absolute_ceiling": 1.0, "sample_minimum": 24},
    "topk_overlap_rate_k10": {"role": "promotion", "direction": "higher", "aggregation": "mean", "margin": 0.01, "relative_margin": 0.01, "absolute_floor": 0.1, "absolute_ceiling": 1.0, "sample_minimum": 24},
    "logits_cosine": {"role": "promotion", "direction": "higher", "aggregation": "mean", "margin": 0.01, "relative_margin": 0.01, "absolute_floor": 0.0, "absolute_ceiling": 1.0, "sample_minimum": 24},
    "logits_relative_l2": {"role": "promotion", "direction": "lower", "aggregation": "mean", "margin": 0.05, "relative_margin": 0.05, "absolute_floor": 0.0, "absolute_ceiling": 1.0, "pathological_rejection_ceiling": 1.0, "sample_minimum": 24},
    "hidden_cosine": {"role": "promotion", "direction": "higher", "aggregation": "mean", "margin": 0.01, "relative_margin": 0.01, "absolute_floor": 0.0, "absolute_ceiling": 1.0, "sample_minimum": 24},
    "hidden_relative_l2": {"role": "promotion", "direction": "lower", "aggregation": "mean", "margin": 0.05, "relative_margin": 0.05, "absolute_floor": 0.0, "absolute_ceiling": 1.0, "pathological_rejection_ceiling": 1.0, "sample_minimum": 24},
    "hidden_max_abs": {"role": "diagnostic_only", "direction": "diagnostic", "aggregation": "max", "confidence_level": None, "margin": None, "relative_margin": None, "absolute_floor": None, "absolute_ceiling": None, "sample_minimum": 24},
    "bf16_top1_retained_in_aq4_top10_rate": {"role": "promotion", "direction": "higher", "aggregation": "wilson_lower_one_sided", "confidence_level": 0.95, "margin": None, "relative_margin": None, "absolute_floor": None, "absolute_ceiling": 1.0, "sample_minimum": 24},
}
BOUNDED_UNIT_METRICS = {"token_agreement_rate", "topk_overlap_rate_k10", "logits_cosine", "hidden_cosine", "bf16_top1_retained_in_aq4_top10_rate"}
BINARY_RATE_METRICS = {"token_agreement_rate", "bf16_top1_retained_in_aq4_top10_rate"}
WILSON_Z_95_ONE_SIDED = 1.6448536269514722


class ProtocolError(ValueError):
    pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in items:
        if key in out:
            raise ProtocolError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def no_constants(value: str) -> Any:
    raise ProtocolError(f"non-finite JSON value: {value}")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path, label: str, limit: int = MAX_JSON) -> str:
    reject_path(path, label)
    if path.stat().st_size > limit:
        raise ProtocolError(f"{label} exceeds bounded size")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def reject_path(path: Path, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ProtocolError(f"{label} has symlink component: {current}")
    if path.is_symlink() or not path.is_file():
        raise ProtocolError(f"{label} must be a regular file")


def load(path: Path, label: str, limit: int = MAX_JSON) -> tuple[dict[str, Any], bytes]:
    reject_path(path, label)
    if path.stat().st_size > limit:
        raise ProtocolError(f"{label} exceeds bounded size")
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=no_constants)
    except (UnicodeError, json.JSONDecodeError, ProtocolError) as error:
        raise ProtocolError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} root must be an object")
    return value, raw


def case_hash(case: dict[str, Any]) -> str:
    value = json.loads(json.dumps(case))
    value["case_sha256"] = None
    return sha_bytes(canonical(value))


def context_hash(ids: list[int]) -> str:
    # The Rust differential trace canonical context includes a trailing newline.
    return sha_bytes(canonical(ids) + b"\n")


def safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256 or ID_RE.fullmatch(value) is None:
        raise ProtocolError(f"invalid {label}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    if os.path.lexists(path):
        raise ProtocolError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    with temporary.open("xb", buffering=0) as stream:
        stream.write(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    if os.path.lexists(path):
        raise ProtocolError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def selected_cases(expanded: dict[str, Any], index: dict[str, Any], index_path: Path) -> list[dict[str, Any]]:
    if expanded.get("schema_version") != EXPANDED_SCHEMA or index.get("schema_version") != INDEX_SCHEMA:
        raise ProtocolError("expanded/index schema differs")
    cases = expanded.get("cases")
    entries = index.get("cases")
    if not isinstance(cases, list) or not isinstance(entries, list) or index.get("case_count") != len(entries):
        raise ProtocolError("expanded/index case lists are invalid")
    selected = [case for case in cases if isinstance(case, dict) and case.get("stage_id") == "representative" and case.get("scope") == "production_server" and case.get("phase") == "cold_prefill" and case.get("device", {}).get("device_id") == "r9700-rdna4" and case.get("control_id") == "aq4_0_target"]
    if len(selected) != 48:
        raise ProtocolError(f"production target profile must contain 48 cases, got {len(selected)}")
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ProtocolError("fixture index entry is not an object")
        case_id = safe_id(entry.get("case_id"), "fixture case_id")
        if case_id in by_id:
            raise ProtocolError(f"duplicate fixture case_id: {case_id}")
        by_id[case_id] = entry
    result: list[dict[str, Any]] = []
    for case in selected:
        case_id = safe_id(case.get("case_id"), "case_id")
        if case.get("case_sha256") != case_hash(case):
            raise ProtocolError(f"expanded case identity differs: {case_id}")
        entry = by_id.get(case_id)
        if not isinstance(entry, dict) or any(entry.get(field) != case.get(field) for field in ("case_sha256", "prompt_tokens", "context_tokens", "generated_tokens")):
            raise ProtocolError(f"fixture index does not bind selected case: {case_id}")
        fixture_path = Path(entry.get("fixture_path", ""))
        if not fixture_path.is_absolute():
            fixture_path = index_path.parent.resolve() / fixture_path
        else:
            fixture_path = fixture_path.resolve()
        if sha_file(fixture_path, f"fixture {case_id}") != entry.get("fixture_sha256"):
            raise ProtocolError(f"fixture hash differs: {case_id}")
        fixture, _ = load(fixture_path, f"fixture {case_id}")
        if fixture.get("schema_version") != FIXTURE_SCHEMA or not isinstance(fixture.get("cases"), list) or len(fixture["cases"]) != 1:
            raise ProtocolError(f"fixture schema differs: {case_id}")
        item = fixture["cases"][0]
        ids = item.get("prompt_token_ids")
        if item.get("case_id") != case_id or not isinstance(ids, list) or len(ids) != case["prompt_tokens"] or any(type(token) is not int or token < 0 for token in ids):
            raise ProtocolError(f"fixture token contract differs: {case_id}")
        expected_prompt_hash = sha_bytes(canonical(ids))
        if entry.get("prompt_token_ids_sha256") != expected_prompt_hash:
            raise ProtocolError(f"fixture prompt hash differs: {case_id}")
        if case.get("cached_prefix_tokens") != 0 or case.get("context_tokens") != case.get("prompt_tokens") or case.get("generated_tokens") != 0:
            raise ProtocolError(f"selected case is not full-context step-zero: {case_id}")
        context_sha = context_hash(ids)
        if case_id in ATTEMPT2_CASE_IDS or context_sha in ATTEMPT2_CONTEXT_HASHES:
            raise ProtocolError(f"attempt2 fixture is excluded: {case_id}")
        result.append({"case": case, "entry": entry, "fixture_path": str(fixture_path), "prompt_token_ids_sha256": expected_prompt_hash, "context_token_ids_sha256": context_sha})
    return sorted(result, key=lambda item: item["case"]["case_id"])


def row(item: dict[str, Any], subset: str) -> dict[str, Any]:
    case = item["case"]
    entry = item["entry"]
    return {
        "case_id": case["case_id"], "case_sha256": case["case_sha256"], "fixture_sha256": entry["fixture_sha256"],
        "fixture_path": item["fixture_path"], "prompt_token_ids_sha256": item["prompt_token_ids_sha256"], "context_token_ids_sha256": item["context_token_ids_sha256"],
        "prompt_tokens": case["prompt_tokens"], "cached_prefix_tokens": 0, "context_tokens": case["context_tokens"], "generated_tokens": 0,
        "baseline_mode": case["baseline_mode"], "prefill_requested_m": case["prefill_requested_m"], "resolved_m": case["resolved_m"], "step": 0, "row_count": 1, "subset": subset,
    }


def policy() -> dict[str, Any]:
    metric_policy: dict[str, Any] = {}
    for name, spec in METRICS.items():
        item = {**spec, "observed_domain": "[0,1]" if name in BOUNDED_UNIT_METRICS else "[0,+inf)"}
        if name in BINARY_RATE_METRICS:
            item["formula"] = "wilson_lower_one_sided(successes=sum(exact 1.0 rows), n=24, confidence_level=0.95); no mean-minus-margin"
        elif spec["role"] == "diagnostic_only":
            item["formula"] = "diagnostic_max only; no promotion bound and no absolute ceiling"
        else:
            item["formula"] = "higher: max(absolute_floor, mean-max(absolute_margin, relative_margin*abs(mean))); lower: min(absolute_ceiling, mean+max(absolute_margin, relative_margin*abs(mean)))"
        metric_policy[name] = item
    return {
        "schema_version": POLICY_SCHEMA, "status": "formula_frozen_unbound", "promotion_eligible": False,
        "attempt2_threshold_source_forbidden": True, "observed_attempt2_values_forbidden": True, "attempt2_artifact_ids_forbidden": sorted(ATTEMPT2_CASE_IDS), "attempt2_context_hashes_forbidden": sorted(ATTEMPT2_CONTEXT_HASHES),
        "calibration_subset_only_for_active_bf16_envelope": True, "holdout_evaluation_allowed_once": True, "candidate_active_behavioral_gate": {"mode": "exact", "mismatch_action": "no-go"},
        "split": {"strata_fields": list(STRATA_FIELDS), "stratum_size": 6, "calibration_per_stratum": 3, "holdout_per_stratum": 3, "total_cases": 48, "calibration_cases": 24, "holdout_cases": 24},
        "metrics": metric_policy,
        "quality_task": {"kind": "binary_retention_rate", "score": "bf16_top1_retained_in_aq4_top10", "calculation": "1 when BF16 source greedy token is present in the AQ4 top-10 set, else 0", "task_fixture_identity_required": True, "natural_language_suite": "optional_separate_artifact", "no_regression_against_calibration_bound": True},
        "relative_l2_rejection": {"ceiling": 1.0, "action": "reject any observed relative-L2 > 1 before aggregation", "reason": "relative-L2 above 100 percent is a predeclared pathological-drift rejection; this structural check is distinct from raw hidden max-abs, which has no natural scale"},
        "non_vacuity": {"reason": "No Qwen3.5 AQ4 acceptance artifact supplies an absolute token-exact floor; Wilson lower bounds provide finite-sample non-vacuity for binary rates, continuous metrics use fixed absolute+relative margins, and raw hidden max-abs is diagnostic-only because its scale is not dimensionless."},
        "forbidden_threshold_sources": ["attempt2 differential-trace observed values", "attempt2 rows/payload/VRAM/power/producer summaries"],
    }


def digest_sort_key(case_sha: str) -> str:
    return sha_bytes(b"ullm.aq4_p2_fidelity_split.v1\0" + case_sha.encode("ascii"))


def split(args: argparse.Namespace) -> None:
    expanded, expanded_raw = load(args.expanded, "expanded")
    index, index_raw = load(args.fixture_index, "fixture index")
    items = selected_cases(expanded, index, args.fixture_index)
    strata: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for item in items:
        key = (item["case"]["prompt_tokens"], item["case"]["baseline_mode"])
        strata.setdefault(key, []).append(item)
    if set(strata) != {(p, mode) for p in (1011, 1024, 1339, 2048) for mode in ("all_m1", "cold_batched")} or any(len(v) != 6 for v in strata.values()):
        raise ProtocolError("selected cases do not form the expected 8 strata of 6")
    calibration: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    strata_receipt: list[dict[str, Any]] = []
    for key in sorted(strata):
        ordered = sorted(strata[key], key=lambda item: digest_sort_key(item["case"]["case_sha256"]))
        calibration.extend(ordered[:3]); holdout.extend(ordered[3:])
        strata_receipt.append({"prompt_tokens": key[0], "baseline_mode": key[1], "case_count": 6, "calibration_case_ids": [x["case"]["case_id"] for x in ordered[:3]], "holdout_case_ids": [x["case"]["case_id"] for x in ordered[3:]]})
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise ProtocolError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    calibration_rows = [row(item, "calibration") for item in sorted(calibration, key=lambda x: x["case"]["case_id"])]
    holdout_rows = [row(item, "holdout") for item in sorted(holdout, key=lambda x: x["case"]["case_id"])]
    atomic_text(output / "calibration-cases.jsonl", "".join(json.dumps(x, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n" for x in calibration_rows))
    atomic_text(output / "holdout-cases.jsonl", "".join(json.dumps(x, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n" for x in holdout_rows))
    policy_value = policy()
    atomic_json(output / "policy.json", policy_value)
    manifest = {"schema_version": SPLIT_SCHEMA, "status": "ready_for_calibration", "expanded_manifest_sha256": sha_bytes(expanded_raw), "fixture_index_sha256": sha_bytes(index_raw), "selected_case_count": 48, "calibration_case_count": 24, "holdout_case_count": 24, "split_domain": "ullm.aq4_p2_fidelity_split.v1\\0 + case_sha256", "algorithm": "sha256(domain || case_sha256), lexicographic within each prompt_tokens/baseline_mode stratum; first 3 calibration, last 3 holdout", "attempt2_exclusions": {"case_ids": sorted(ATTEMPT2_CASE_IDS), "context_token_ids_sha256": sorted(ATTEMPT2_CONTEXT_HASHES)}, "strata": strata_receipt, "calibration_sha256": sha_file(output / "calibration-cases.jsonl", "calibration cases"), "holdout_sha256": sha_file(output / "holdout-cases.jsonl", "holdout cases"), "policy_sha256": sha_file(output / "policy.json", "policy")}
    atomic_json(output / "split-manifest.json", manifest)
    sums = "".join(f"{sha_file(output / name, name)}  {name}\n" for name in ("calibration-cases.jsonl", "holdout-cases.jsonl", "policy.json", "split-manifest.json"))
    atomic_text(output / "SHA256SUMS", sums)
    print(json.dumps({"status": "ok", "selected": 48, "calibration": 24, "holdout": 24, "output": str(output)}, sort_keys=True))


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    reject_path(path, label)
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line, object_pairs_hook=pairs, parse_constant=no_constants)
        except (UnicodeError, json.JSONDecodeError, ProtocolError) as error:
            raise ProtocolError(f"invalid {label} line {number}: {error}") from error
        if not isinstance(value, dict):
            raise ProtocolError(f"{label} line {number} is not an object")
        rows.append(value)
    return rows


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ProtocolError(f"{label} must be finite")
    return float(value)


def wilson_lower_one_sided(successes: int, samples: int, z: float = WILSON_Z_95_ONE_SIDED) -> float:
    if not 0 <= successes <= samples or samples <= 0:
        raise ProtocolError("Wilson input count is invalid")
    p = successes / samples
    z2 = z * z
    denominator = 1.0 + z2 / samples
    center = p + z2 / (2.0 * samples)
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * samples)) / samples)
    return max(0.0, (center - radius) / denominator)


def reject_attempt2_reference(value: Any) -> None:
    """Calibration metrics must not carry an attempt2 observation as a provenance/source."""
    if isinstance(value, dict):
        for key, child in value.items():
            if "attempt2" in str(key).lower():
                raise ProtocolError("attempt2 reference is forbidden in calibration metrics")
            reject_attempt2_reference(child)
    elif isinstance(value, list):
        for child in value:
            reject_attempt2_reference(child)
    elif isinstance(value, str) and "attempt2" in value.lower():
        raise ProtocolError("attempt2 reference is forbidden in calibration metrics")


def freeze(args: argparse.Namespace) -> None:
    root = args.split_root
    manifest, manifest_raw = load(root / "split-manifest.json", "split manifest")
    policy_value, policy_raw = load(root / "policy.json", "policy")
    metrics, metrics_raw = load(args.metrics, "calibration metrics")
    if manifest.get("schema_version") != SPLIT_SCHEMA or policy_value.get("schema_version") != POLICY_SCHEMA or metrics.get("schema_version") != METRICS_SCHEMA:
        raise ProtocolError("freeze schemas differ")
    if metrics.get("split_manifest_sha256") != sha_bytes(manifest_raw) or metrics.get("subset") != "calibration":
        raise ProtocolError("metrics do not bind this split/calibration subset")
    reject_attempt2_reference(metrics)
    expected = {x["case_id"]: x for x in read_jsonl(root / "calibration-cases.jsonl", "calibration cases")}
    rows = metrics.get("rows")
    if not isinstance(rows, list) or len(rows) != 24 or len({x.get("case_id") for x in rows if isinstance(x, dict)}) != 24:
        raise ProtocolError("calibration metrics must contain exactly 24 unique rows")
    aggregate: dict[str, list[float]] = {name: [] for name in METRICS}
    for item in rows:
        if not isinstance(item, dict) or item.get("case_id") not in expected:
            raise ProtocolError("calibration row case identity differs")
        expected_row = expected[item["case_id"]]
        for field in ("case_sha256", "fixture_sha256", "prompt_token_ids_sha256", "context_token_ids_sha256", "prompt_tokens", "context_tokens", "baseline_mode", "prefill_requested_m", "resolved_m", "step", "row_count"):
            if item.get(field) != expected_row.get(field):
                raise ProtocolError(f"calibration row {item.get('case_id')} identity differs: {field}")
        values = item.get("metrics")
        if not isinstance(values, dict):
            raise ProtocolError("calibration row metrics are missing")
        for name in METRICS:
            number = finite_number(values.get(name), f"{item['case_id']}.{name}")
            if number < 0 or (name in BOUNDED_UNIT_METRICS and number > 1) or (name in {"logits_relative_l2", "hidden_relative_l2"} and number > 1):
                raise ProtocolError(f"{item['case_id']}.{name} is outside its frozen domain")
            if name in BINARY_RATE_METRICS and number not in (0.0, 1.0):
                raise ProtocolError(f"{item['case_id']}.{name} must be binary 0 or 1")
            aggregate[name].append(number)
    derived: dict[str, Any] = {}
    for name, spec in METRICS.items():
        if len(aggregate[name]) < spec["sample_minimum"]:
            raise ProtocolError(f"sample minimum not met: {name}")
        mean = sum(aggregate[name]) / len(aggregate[name])
        if name in BINARY_RATE_METRICS:
            successes = sum(value == 1.0 for value in aggregate[name])
            bound = wilson_lower_one_sided(successes, len(aggregate[name]))
            derived[name] = {"calibration_mean": mean, "successes": successes, "confidence_level": spec["confidence_level"], "wilson_z": WILSON_Z_95_ONE_SIDED, "bound": bound, "direction": spec["direction"], "sample_count": len(aggregate[name])}
        elif spec["role"] == "diagnostic_only":
            derived[name] = {"diagnostic_max": max(aggregate[name]), "bound": None, "direction": spec["direction"], "sample_count": len(aggregate[name])}
        else:
            margin = max(float(spec["margin"]), float(spec["relative_margin"]) * abs(mean))
            if spec["direction"] == "higher":
                bound = mean - margin
                if spec["absolute_floor"] is not None:
                    bound = max(spec["absolute_floor"], bound)
            else:
                bound = mean + margin
                if spec["absolute_floor"] is not None:
                    bound = max(spec["absolute_floor"], bound)
            bound = min(spec["absolute_ceiling"], bound)
            derived[name] = {"calibration_mean": mean, "absolute_margin": spec["margin"], "relative_margin": spec["relative_margin"], "effective_margin": margin, "bound": bound, "direction": spec["direction"], "sample_count": len(aggregate[name])}
    receipt = {"schema_version": RECEIPT_SCHEMA, "status": "frozen_calibration_envelope", "split_manifest_sha256": sha_bytes(manifest_raw), "policy_sha256": sha_bytes(policy_raw), "metrics_sha256": sha_bytes(metrics_raw), "calibration_case_count": 24, "holdout_status": "not_started", "holdout_evaluations_remaining": 1, "derived_bounds": derived, "candidate_active_behavioral_gate": "exact; any context/token/greedy/top-k/KV/state/scheduler/reset mismatch is no-go", "attempt2_threshold_source_forbidden": True}
    atomic_json(args.output, receipt)
    print(json.dumps({"status": "ok", "receipt": str(args.output), "holdout_status": "not_started"}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split_parser = commands.add_parser("split")
    split_parser.add_argument("--expanded", type=Path, required=True)
    split_parser.add_argument("--fixture-index", type=Path, required=True)
    split_parser.add_argument("--output", type=Path, required=True)
    split_parser.set_defaults(function=split)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--split-root", type=Path, required=True)
    freeze_parser.add_argument("--metrics", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    freeze_parser.set_defaults(function=freeze)
    args = parser.parse_args(argv)
    try:
        args.function(args)
        return 0
    except (ProtocolError, OSError, ValueError) as error:
        print(f"AQ4 P2 fidelity protocol failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
