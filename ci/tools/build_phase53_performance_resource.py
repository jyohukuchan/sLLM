#!/usr/bin/env python3
"""Build Phase 53 performance/resource evidence from 14 direct reports.

The caller supplies one FP16 and one block16 ``engine-performance-direct-v2``
report for every frozen case plus the external HBM/GTT monitor observation.
This tool validates and digest-binds those raw inputs; it never runs a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
DIRECT_SCHEMA = ROOT / "ci/schema/engine-performance-direct-v2.schema.json"
HBM_SCHEMA = ROOT / "ci/schema/phase53-external-hbm-observation-v1.schema.json"
OUTPUT_SCHEMA = ROOT / "ci/schema/phase53-performance-resource-evidence-v2.schema.json"
MODEL_FINGERPRINT = "sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae"
FULL_ATTENTION_LAYERS = 8
KV_HEADS = 4
KV_PLANES = 2

CASE_SPECS: tuple[tuple[str, str, int, int, int, int, int, bool], ...] = (
    ("short-odd", "normal", 17, 17, 3, 10, 34, False),
    ("32-32", "normal", 32, 32, 3, 10, 64, False),
    ("prefill-long", "normal", 1024, 128, 3, 10, 1152, False),
    ("decode-long", "normal", 32, 256, 3, 10, 288, False),
    ("long-10001", "normal", 10_001, 2, 3, 10, 10_003, False),
    ("long-100000", "long-running", 100_000, 2, 1, 3, 131_072, False),
    ("decode-20000", "long-running", 32, 20_000, 1, 3, 131_072, True),
)
CASE_IDS = tuple(spec[0] for spec in CASE_SPECS)
TARGET_ARCH = {
    "gfx942:sramecc+:xnack-": "gfx942",
    "gfx1201": "gfx1201",
    "gfx1030": "gfx1030",
}
TARGET_ENCODING = {
    "gfx942:sramecc+:xnack-": "kv-fp8-e4-block16",
    "gfx1201": "kv-fp8-e4-block16",
    "gfx1030": "kv-fp8-e5-block16",
}
TARGET_SELECTION = {
    "gfx942:sramecc+:xnack-": ("E4M3-FNUZ", "kv-fp8-e4-block16-v2"),
    "gfx1201": ("E4M3-OCP", "kv-fp8-e4-block16-v2"),
    "gfx1030": ("E5M2-software", "kv-fp8-e5-block16-v2"),
}


class ContractError(ValueError):
    """Raw evidence is incomplete or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {path}: {error}") from error
    require(isinstance(value, dict), f"{path}: JSON root is not an object")
    return value, digest_bytes(raw)


def validator(path: Path) -> Draft202012Validator:
    document, _ = read_json(path)
    Draft202012Validator.check_schema(document)
    return Draft202012Validator(document)


def validate_schema(value: dict[str, Any], schema: Draft202012Validator, label: str) -> None:
    errors = sorted(schema.iter_errors(value), key=lambda error: tuple(str(item) for item in error.path))
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.path) or "<root>"
        raise ContractError(f"{label}: schema violation at {location}: {first.message}")


def finite_positive(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label}: not numeric")
    result = float(value)
    require(math.isfinite(result) and result > 0.0, f"{label}: not finite and positive")
    return result


def parse_case_paths(values: list[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in values:
        case_id, separator, raw_path = item.partition("=")
        require(separator == "=" and case_id in CASE_IDS and raw_path, f"{option}: expected CASE=PATH")
        require(case_id not in result, f"{option}: duplicate {case_id}")
        result[case_id] = Path(raw_path)
    require(tuple(result) == CASE_IDS, f"{option}: cases must be supplied once in frozen order {CASE_IDS}")
    return result


def expected_selection(target: str, encoding: str) -> dict[str, Any]:
    if encoding == "fp16":
        return {
            "requested": "fp16", "resolved": "fp16", "selection_source": "process-explicit",
            "physical_variant": None, "descriptor_id": None, "policy_version": 1,
        }
    physical, descriptor = TARGET_SELECTION[target]
    return {
        "requested": encoding, "resolved": encoding, "selection_source": "process-explicit",
        "physical_variant": physical, "descriptor_id": descriptor, "policy_version": 1,
    }


def validate_direct(
    report: dict[str, Any], report_digest: str, *, target: str, case_spec: tuple[str, str, int, int, int, int, int, bool], encoding: str,
    schema: Draft202012Validator,
) -> dict[str, Any]:
    case_id, _case_class, input_count, output_count, warmups, measured, context, ignore_eos = case_spec
    label = f"{case_id}/{encoding}"
    validate_schema(report, schema, label)
    arch = TARGET_ARCH[target]
    row = report["row"]
    identities = report["identities"]
    config = report["config"]
    require(report.get("state") == "PASS" and report.get("lane") == "direct", f"{label}: direct report is not PASS")
    require(identities.get("engine") == "sllm" and identities.get("backend") == "hip" and identities.get("target") == arch, f"{label}: engine/target differs")
    require(identities.get("model", {}).get("model_size") == "4B", f"{label}: model size differs")
    require(identities.get("model", {}).get("lock_fingerprint") == MODEL_FINGERPRINT, f"{label}: model lock differs")
    require(identities.get("binding", {}).get("model_fingerprint") == MODEL_FINGERPRINT, f"{label}: binding fingerprint differs")
    require(row.get("case_id") == case_id and row.get("input_token_count") == input_count and row.get("requested_output_tokens") == output_count, f"{label}: row identity differs")
    require(len(row.get("input_token_ids", [])) == input_count, f"{label}: row input token count differs")
    expected_config = {
        "input_token_count": input_count, "max_new_tokens": output_count, "warmups": warmups,
        "measured": measured, "effective_context_length": context, "ignore_eos": ignore_eos,
        "kv_cache_encoding": encoding,
    }
    for key, expected in expected_config.items():
        require(config.get(key) == expected, f"{label}: config {key} differs")
    require(config.get("context_length") == context, f"{label}: explicit context length differs")
    require(len(config.get("input_token_ids", [])) == input_count and config.get("input_token_ids") == row.get("input_token_ids"), f"{label}: config tokens differ")
    selection = config.get("kv_cache_selection")
    require(isinstance(selection, dict), f"{label}: target-aware KV selection is absent")
    expected = expected_selection(target, encoding)
    for key, expected_value in expected.items():
        require(selection.get(key) == expected_value, f"{label}: KV selection {key} differs")
    require(isinstance(selection.get("reason"), str) and selection["reason"], f"{label}: KV selection reason is absent")
    audit = report["audit"]
    require(audit.get("selected_backend") == "hip" and audit.get("target") == arch and audit.get("fallback_used") is False and audit.get("all_dispatches_hip") is True, f"{label}: aggregate dispatch is not HIP-only/no-fallback")
    require(report["cleanup"].get("all_requests_dropped") is True and report["cleanup"].get("retryable_cleanup") == 0 and report["cleanup"].get("durable_quarantine") == 0, f"{label}: request cleanup differs")
    require(report["session_cleanup"] == {"retryable_cleanup": 0, "durable_quarantine": 0}, f"{label}: session cleanup differs")
    after_drop = report["memory"]["after_model_drop"]
    require(after_drop.get("current_bytes") == 0 and after_drop.get("poisoned") is False, f"{label}: model drop did not empty the runtime allocator")

    samples = report["warmups"]["samples"] + report["measured"]["samples"]
    require(len(samples) == warmups + measured, f"{label}: sample count differs")
    for index, sample in enumerate(samples):
        sample_audit = sample["audit"]
        cleanup = sample["cleanup"]
        require(sample_audit.get("selected_backend") == "hip" and sample_audit.get("target") == arch and sample_audit.get("fallback_used") is False and sample_audit.get("all_dispatches_hip") is True and sample_audit.get("model_fingerprint") == MODEL_FINGERPRINT, f"{label}: sample {index} is not HIP-only/no-fallback")
        require(cleanup.get("request_dropped") is True and cleanup.get("allocator_cleanup_validated") is True and cleanup.get("retryable_cleanup") == 0 and cleanup.get("durable_quarantine") == 0, f"{label}: sample {index} cleanup differs")
        require(sample["memory"]["after_cleanup"].get("poisoned") is False, f"{label}: sample {index} allocator is poisoned")

    measured_samples = report["measured"]["samples"]
    e2e_values: list[float] = []
    throughput_values: list[float] = []
    physical_values: list[int] = []
    for index, sample in enumerate(measured_samples):
        generated = sample["tokens"].get("generated_token_ids")
        require(isinstance(generated, list) and generated, f"{label}: measured sample {index} generated zero tokens")
        e2e = finite_positive(sample["derived"].get("e2e_ns"), f"{label}: measured sample {index} e2e")
        e2e_values.append(e2e)
        throughput_values.append((input_count + len(generated)) * 1_000_000_000.0 / e2e)
        kv = sample["memory"].get("kv")
        require(isinstance(kv, dict), f"{label}: measured sample {index} lacks physical KV evidence")
        physical = kv.get("committed_kv_bytes")
        require(isinstance(physical, int) and not isinstance(physical, bool) and physical > 0, f"{label}: measured sample {index} physical KV bytes are zero")
        require(isinstance(kv.get("layers"), list) and len(kv["layers"]) == kv.get("kv_layer_count") == FULL_ATTENTION_LAYERS, f"{label}: measured sample {index} KV layer inventory differs")
        physical_values.append(physical)
    bytes_per_plane = 512 if encoding == "fp16" else 272
    logical = context * FULL_ATTENTION_LAYERS * KV_HEADS * KV_PLANES * bytes_per_plane
    return {
        "digest": report_digest,
        "row_id": row["row_id"],
        "median_e2e_ns": float(statistics.median(e2e_values)),
        "tokens_per_second": float(statistics.median(throughput_values)),
        "logical_kv_bytes": logical,
        "physical_kv_bytes": max(physical_values),
    }


def validate_hbm_run(run: dict[str, Any], expected_digest: str, label: str) -> int:
    require(run.get("direct_report_sha256") == expected_digest, f"{label}: direct report digest differs")
    require(run["completed_ns"] > run["started_ns"], f"{label}: monitor interval is not positive")
    require(run["peak_hbm_bytes"] >= run["baseline_hbm_bytes"] and run["peak_gtt_bytes"] >= run["baseline_gtt_bytes"], f"{label}: peak is below baseline")
    require(run["settled_hbm_bytes"] == run["baseline_hbm_bytes"] and run["settled_gtt_bytes"] == run["baseline_gtt_bytes"], f"{label}: HBM/GTT did not settle")
    require(run.get("settled") is True and run.get("process_group_gone") is True and run.get("monitor_samples", 0) > 0, f"{label}: external monitor cleanup differs")
    return run["peak_hbm_bytes"] - run["baseline_hbm_bytes"]


def build(
    *, policy_path: Path, binary_path: Path, target: str, fp16_paths: dict[str, Path], candidate_paths: dict[str, Path], hbm_path: Path,
) -> dict[str, Any]:
    require(target in TARGET_ARCH, f"unknown exact target {target!r}")
    policy, policy_digest = read_json(policy_path)
    require(policy.get("schema_version") == "kv-cache-default-v2", "policy schema version differs")
    binary_digest = digest_bytes(binary_path.read_bytes())
    hbm, hbm_digest = read_json(hbm_path)
    validate_schema(hbm, validator(HBM_SCHEMA), "HBM observation")
    require(hbm.get("target") == target and hbm.get("binary_sha256") == binary_digest, "HBM target/binary identity differs")
    direct_validator = validator(DIRECT_SCHEMA)
    encoding = TARGET_ENCODING[target]
    hbm_cases = hbm["cases"]
    rows: list[dict[str, Any]] = []
    previous_completed: int | None = None
    candidate_physical: list[int] = []
    for row_number, (spec, observation) in enumerate(zip(CASE_SPECS, hbm_cases), 1):
        case_id, case_class, input_count, output_count, warmups, measured, context, ignore_eos = spec
        require(observation.get("case_id") == case_id and observation.get("execution_order") == ["fp16", "block16"], f"{case_id}: HBM case order differs")
        fp16_report, fp16_digest = read_json(fp16_paths[case_id])
        candidate_report, candidate_digest = read_json(candidate_paths[case_id])
        fp16 = validate_direct(fp16_report, fp16_digest, target=target, case_spec=spec, encoding="fp16", schema=direct_validator)
        candidate = validate_direct(candidate_report, candidate_digest, target=target, case_spec=spec, encoding=encoding, schema=direct_validator)
        fp16_monitor = observation["fp16"]
        candidate_monitor = observation["candidate"]
        fp16_delta = validate_hbm_run(fp16_monitor, fp16_digest, f"{case_id}/fp16")
        candidate_delta = validate_hbm_run(candidate_monitor, candidate_digest, f"{case_id}/candidate")
        require(fp16_monitor["completed_ns"] <= candidate_monitor["started_ns"], f"{case_id}: FP16 and candidate were not serial")
        if previous_completed is not None:
            require(previous_completed <= fp16_monitor["started_ns"], f"{case_id}: case overlapped the preceding case")
        previous_completed = candidate_monitor["completed_ns"]
        candidate_physical.append(candidate["physical_kv_bytes"])

        def output_run(data: dict[str, Any], hbm_delta: int) -> dict[str, Any]:
            return {
                "direct_report_sha256": data["digest"], "row_id": data["row_id"],
                "median_e2e_ns": data["median_e2e_ns"], "tokens_per_second": data["tokens_per_second"],
                "logical_kv_bytes": data["logical_kv_bytes"], "physical_kv_bytes": data["physical_kv_bytes"],
                "hbm_peak_delta_bytes": hbm_delta, "generated": True, "hip_only": True,
                "fallback_used": False, "cleanup_empty": True, "hbm_gtt_settled": True,
            }

        rows.append({
            "row": row_number, "case_id": case_id, "class": case_class,
            "input_token_count": input_count, "requested_output_tokens": output_count,
            "protocol": {"warmups": warmups, "measured": measured, "context_length": context, "ignore_eos": ignore_eos},
            "execution_order": ["fp16", "block16"],
            "fp16": output_run(fp16, fp16_delta), "candidate": output_run(candidate, candidate_delta),
            "candidate_to_fp16_throughput_ratio": candidate["tokens_per_second"] / fp16["tokens_per_second"],
        })
    result = {
        "$schema": "https://sllm.dev/schema/phase53-performance-resource-evidence-v2.schema.json",
        "schema_version": "sllm-phase53-performance-resource-evidence-v2", "state": "PASS",
        "producer": "ci/tools/build_phase53_performance_resource.py", "target": target, "encoding": encoding,
        "descriptor_id": TARGET_SELECTION[target][1], "scale_recipe": "standard-mx-floor-power-v1",
        "policy_sha256": policy_digest, "binary_sha256": binary_digest, "hbm_observation_sha256": hbm_digest,
        "model_lock_fingerprint": MODEL_FINGERPRINT, "selected_count": len(rows), "rows": rows,
        "memory": {"fp16_bytes_per_token_head_plane": 512, "candidate_bytes_per_token_head_plane": 272, "logical_reduction_fraction": 0.46875, "candidate_physical_kv_bytes_max": max(candidate_physical), "physical_measured": True},
        "decisions": {"performance": "pass", "resource": "pass", "memory": "pass"},
        "fallback_used": False, "cleanup": {"retryable": 0, "durable": 0, "settled": True},
    }
    validate_schema(result, validator(OUTPUT_SCHEMA), "Phase 53 output")
    return result


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--fp16", action="append", default=[], metavar="CASE=PATH")
    parser.add_argument("--candidate", action="append", default=[], metavar="CASE=PATH")
    parser.add_argument("--hbm-observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(
            policy_path=args.policy, binary_path=args.binary, target=args.target,
            fp16_paths=parse_case_paths(args.fp16, "--fp16"),
            candidate_paths=parse_case_paths(args.candidate, "--candidate"),
            hbm_path=args.hbm_observation,
        )
        atomic_write(args.output, result)
    except (ContractError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
