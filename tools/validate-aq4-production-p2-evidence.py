#!/usr/bin/env python3
"""Independently validate a complete AQ4 P2 scheduled result matrix."""

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

HASH_RE = re.compile(r"^[0-9a-f]{64}$")
STATUSES = {"ok", "failed", "oom", "unsupported", "skipped"}
POWER_FIELDS = ("expected_power_limit_watts", "allowed_power_tolerance_watts", "maximum_temperature_c", "minimum_vram_headroom_bytes")
CORRECTNESS_FIELDS = ("max_hidden_relative_l2", "max_hidden_max_abs", "max_logits_relative_l2", "max_logits_max_abs", "minimum_top_k_overlap")


class EvidenceError(ValueError): pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in items:
        if key in value: raise EvidenceError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024: raise EvidenceError(f"{label} must be a bounded regular file")
    try: value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(EvidenceError(f"non-finite JSON number: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error: raise EvidenceError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict): raise EvidenceError(f"{label} root must be an object")
    return value


def canonical(value: Any) -> bytes: return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
def sha_bytes(value: bytes) -> str: return hashlib.sha256(value).hexdigest()


def sha_file(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file(): raise EvidenceError(f"{label} must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def contained(root: Path, path: Path, label: str) -> Path:
    root = root.resolve(strict=True); resolved = path.resolve(strict=True)
    if resolved != root and root not in resolved.parents: raise EvidenceError(f"{label} escapes run root")
    return resolved


def case_hash(case: dict[str, Any]) -> str:
    value = json.loads(json.dumps(case)); value["case_sha256"] = None
    return sha_bytes(canonical(value))


def identity_hash(identity: dict[str, Any]) -> str:
    value = json.loads(json.dumps(identity)); value["identity_sha256"] = None
    return sha_bytes(canonical(value))


def policy_hash(policy: dict[str, Any]) -> str:
    value = json.loads(json.dumps(policy)); value.setdefault("hash_binding", {})["policy_sha256"] = None
    return sha_bytes(canonical(value))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values); rank = (len(ordered) - 1) * q; low = math.floor(rank); high = math.ceil(rank)
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def trace_failures(trace: dict[str, Any], case: dict[str, Any], prefix: str) -> list[str]:
    failures = []
    if trace.get("schema_version") != "ullm.production_execution_trace.v1" or trace.get("status") != "ok": failures.append(f"trace_schema_status:{prefix}")
    if trace.get("scope") != case.get("scope"): failures.append(f"trace_scope:{prefix}")
    independent = trace.get("verification", {}).get("independent_validation", {})
    if independent.get("status") != "valid" or HASH_RE.fullmatch(str(independent.get("report_sha256", ""))) is None: failures.append(f"trace_independent:{prefix}")
    reset = trace.get("state_commit", {}).get("reset", {})
    if reset.get("attempted") is not True or reset.get("complete") is not True or reset.get("failed") is not False: failures.append(f"trace_reset:{prefix}")
    fallback = trace.get("fallback", {})
    if any(fallback.get(field) != 0 for field in ("unexpected_fallback_count", "fail_closed_count", "unsupported_count")): failures.append(f"trace_fallback:{prefix}")
    memory = trace.get("memory", {})
    if memory.get("oom") is not None or memory.get("observer", {}).get("complete") is not True or not isinstance(memory.get("observed_headroom_bytes"), int) or memory.get("observed_headroom_bytes", 0) <= 0: failures.append(f"trace_memory:{prefix}")
    phases = trace.get("phases", [])
    if case.get("phase") != "decode":
        matching = [phase for phase in phases if phase.get("kind") == case.get("phase")]
        if not matching or any(phase.get("actual_token_batch_width") != case.get("resolved_m") for phase in matching): failures.append(f"trace_phase_width:{prefix}")
    return failures


def validate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.run_root.resolve(strict=True)
    for path, label in ((args.expanded, "expanded"), (args.identity, "identity"), (args.policy, "policy"), (args.source_oracle, "source oracle")): contained(root, path, label)
    expanded = load(args.expanded, "expanded"); identity = load(args.identity, "identity"); policy = load(args.policy, "policy"); source = load(args.source_oracle, "source oracle")
    failures: list[str] = []
    if expanded.get("schema_version") != "ullm.aq4_production_p2_expanded.v2": failures.append("expanded_schema")
    cases = expanded.get("cases") if isinstance(expanded.get("cases"), list) else []
    if len(cases) != expanded.get("case_count") or len(cases) != expanded.get("expected_case_count", {}).get("total"): failures.append("expanded_case_count")
    if expanded.get("canonical_case_sha256") != sha_bytes(canonical(cases)): failures.append("expanded_case_aggregate_hash")
    case_by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or case.get("case_sha256") != case_hash(case): failures.append("expanded_case_self_hash"); continue
        if case.get("case_id") in case_by_id: failures.append(f"expanded_duplicate:{case.get('case_id')}")
        case_by_id[case.get("case_id")] = case
    expanded_sha = sha_file(args.expanded, "expanded")
    if identity.get("schema_version") != "ullm.aq4_production_p2_identity.v2" or identity.get("status") != "bound" or identity.get("identity_sha256") != identity_hash(identity): failures.append("identity_self_hash")
    if identity.get("expanded_manifest_sha256") != expanded_sha or identity.get("hash_binding", {}).get("bound_case_manifest_sha256") != expanded_sha or identity.get("canonical_case_sha256") != expanded.get("canonical_case_sha256"): failures.append("identity_expanded_binding")
    if policy.get("status") != "bound" or policy.get("hash_binding", {}).get("policy_sha256") != policy_hash(policy): failures.append("policy_self_hash")
    contract = policy.get("binding_contract", {})
    for field in contract.get("required_hash_fields", []):
        if HASH_RE.fullmatch(str(policy.get("hash_binding", {}).get(field, ""))) is None: failures.append(f"policy_hash_field:{field}")
    for field in POWER_FIELDS:
        value = policy.get("power_condition", {}).get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0: failures.append(f"policy_power_field:{field}")
    for field in CORRECTNESS_FIELDS:
        value = policy.get("correctness_thresholds", {}).get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0: failures.append(f"policy_correctness_field:{field}")
    if identity.get("policy_sha256") != policy.get("hash_binding", {}).get("policy_sha256"): failures.append("identity_policy_binding")
    source_sha = sha_file(args.source_oracle, "source oracle")
    if source.get("schema_version") != "ullm.qwen35_aq4_source_oracle.v1" or source.get("oracle_kind") != "independent_source" or identity.get("hash_binding", {}).get("source_oracle_sha256") != source_sha: failures.append("source_oracle_binding")
    result_paths = list(args.result)
    if args.result_dir:
        directory = contained(root, args.result_dir, "result directory")
        if not directory.is_dir(): raise EvidenceError("result-dir must be a directory")
        result_paths.extend(sorted(directory.glob("*.json")))
    if len({str(path.resolve()) for path in result_paths}) != len(result_paths): failures.append("duplicate_result_path")
    results: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in result_paths:
        try:
            contained(root, path, "result"); result = load(path, "result")
        except (EvidenceError, OSError) as error:
            failures.append(f"result_unavailable:{path.name}:{error}"); continue
        case_id = result.get("case_id")
        if case_id in results: failures.append(f"duplicate_result_case:{case_id}"); continue
        if case_id not in case_by_id: failures.append(f"extra_result_case:{case_id}"); continue
        results[case_id] = (result, path)
    missing = sorted(set(case_by_id) - set(results))
    if missing: failures.append(f"partial_matrix:{len(missing)}")
    if len(results) != len(case_by_id): failures.append("result_count")
    for case_id, (result, result_path) in results.items():
        case = case_by_id[case_id]; prefix = case_id
        if result.get("schema_version") != "ullm.prefill_validation.v1" or result.get("case_sha256") != case.get("case_sha256"): failures.append(f"result_schema_case_hash:{prefix}")
        if result.get("scope") != case.get("scope") or result.get("control_id") != case.get("control_id") or result.get("identity", {}).get("format_id") != case.get("format_id"): failures.append(f"result_case_identity:{prefix}")
        for field in ("phase", "baseline_mode", "prompt_tokens", "cached_prefix_tokens", "context_tokens", "decode_start_tokens", "prefill_requested_m", "resolved_m", "decode_request_count", "generated_tokens"):
            if result.get("workload", {}).get(field) != case.get(field): failures.append(f"result_workload_{field}:{prefix}")
        try:
            case_path = contained(root, Path(result.get("case", {}).get("path", "")), "case link")
            if sha_file(case_path, "case link") != result.get("case", {}).get("sha256") or load(case_path, "case link") != case: failures.append(f"case_link:{prefix}")
        except (EvidenceError, OSError): failures.append(f"case_link:{prefix}")
        raw_link = result.get("evidence", {}).get("raw_result", {})
        try:
            raw_path = contained(root, Path(raw_link.get("path", "")), "raw link"); raw_sha = sha_file(raw_path, "raw")
            raw = load(raw_path, "raw")
            if raw_sha != raw_link.get("sha256") or raw.get("case_id") != case_id or raw.get("case_sha256") != case.get("case_sha256") or raw.get("status") != result.get("status") or raw.get("immutable_status") is not (raw.get("status") != "ok"): failures.append(f"raw_binding_status:{prefix}")
            if raw.get("declared_execution", {}).get("executable_sha256") != identity.get("hash_binding", {}).get("worker_binary_sha256") or raw.get("declared_execution", {}).get("package_content_sha256") != identity.get("hash_binding", {}).get("package_content_sha256") or raw.get("declared_execution", {}).get("argv_values_recorded") is not False: failures.append(f"raw_execution_identity:{prefix}")
        except (EvidenceError, OSError): failures.append(f"raw_link:{prefix}"); raw = {}; raw_sha = ""
        if result.get("status") not in STATUSES: failures.append(f"result_status:{prefix}")
        if result.get("promotion", {}).get("eligible") is not False: failures.append(f"producer_promotion_claim:{prefix}")
        if case.get("control_id") != "aq4_0_target" and result.get("promotion", {}).get("eligible") is not False: failures.append(f"control_promotion:{prefix}")
        source_link = result.get("oracles", {}).get("source_oracle", {})
        if source_link.get("sha256") != source_sha or source_link.get("independent") is not True: failures.append(f"source_link:{prefix}")
        source_validation_link = result.get("evidence", {}).get("source_oracle_validation", {})
        try:
            source_validation_path = contained(root, Path(source_validation_link.get("path", "")), "source oracle validation")
            source_validation = load(source_validation_path, "source oracle validation")
            if sha_file(source_validation_path, "source oracle validation") != source_validation_link.get("sha256") or source_validation.get("schema_version") != "ullm.qwen35_aq4_p2_oracle_validator.v1" or source_validation.get("status") != "valid" or source_validation.get("oracle_kind") != "independent_source" or source_validation.get("manifest_sha256") != source_sha:
                failures.append(f"source_oracle_validation:{prefix}")
        except (EvidenceError, OSError): failures.append(f"source_oracle_validation:{prefix}")
        path_link = result.get("oracles", {}).get("path_oracle"); expected_path_sha = None
        if case.get("mode") in {"cold_batched", "cached_prefix_chunked"}:
            oracle_case_id = case.get("path_oracle_case_id"); oracle_entry = results.get(oracle_case_id)
            if not isinstance(path_link, dict) or oracle_entry is None or path_link.get("result_sha256") != sha_file(oracle_entry[1], "path result") or Path(path_link.get("result_path", "")).resolve() != oracle_entry[1].resolve(): failures.append(f"path_oracle_link:{prefix}")
            else:
                expected_path_sha = sha_file(oracle_entry[1], "path result")
                oracle_result = oracle_entry[0]
                for field in ("phase", "cached_prefix_tokens", "prompt_tokens", "prefill_requested_m"):
                    if oracle_result.get("workload", {}).get(field) != case.get(field): failures.append(f"path_oracle_state_{field}:{prefix}")
                if oracle_result.get("workload", {}).get("baseline_mode") != "all_m1" or oracle_result.get("scope") != case.get("scope") or oracle_result.get("control_id") != case.get("control_id"): failures.append(f"path_oracle_identity:{prefix}")
        elif path_link is not None: failures.append(f"unexpected_path_oracle:{prefix}")
        trace_link = result.get("evidence", {}).get("execution_trace"); expected_trace_sha = None
        if trace_link is not None:
            try:
                trace_path = contained(root, Path(trace_link.get("path", "")), "trace"); trace = load(trace_path, "trace")
                expected_trace_sha = sha_file(trace_path, "trace")
                if expected_trace_sha != trace_link.get("sha256"): failures.append(f"trace_hash:{prefix}")
                failures.extend(trace_failures(trace, case, prefix))
            except (EvidenceError, OSError): failures.append(f"trace_link:{prefix}")
        elif case.get("scope") == "production_server" and result.get("status") == "ok": failures.append(f"production_trace_missing:{prefix}")
        independent_link = result.get("evidence", {}).get("independent_validation", {})
        try:
            independent_path = contained(root, Path(independent_link.get("path", "")), "independent validation"); independent = load(independent_path, "independent validation")
            if sha_file(independent_path, "independent validation") != independent_link.get("sha256") or independent.get("schema_version") != "ullm.aq4_p2_independent_validation.v1" or independent.get("status") != "valid" or independent.get("validator_independent") is not True or independent.get("case_id") != case_id or independent.get("case_sha256") != case.get("case_sha256") or independent.get("raw_sha256") != raw_sha or independent.get("source_oracle_sha256") != source_sha or independent.get("path_oracle_result_sha256") != expected_path_sha or independent.get("trace_sha256") != expected_trace_sha or result.get("correctness") != independent.get("correctness"): failures.append(f"independent_validation:{prefix}")
            correctness = independent.get("correctness", {}); thresholds = policy.get("correctness_thresholds", {})
            required_correctness = ("finite", "shape_contract_passed", "source_oracle_passed", "greedy_tokens_exact", "kv_state_cache_passed", "scheduler_progress_passed", "chunk_equivalence_passed", "cancel_reset_passed", "publish_failure_reset_passed")
            if any(correctness.get(field) is not True for field in required_correctness) or (case.get("mode") != "all_m1" and correctness.get("path_oracle_passed") is not True): failures.append(f"correctness_boolean:{prefix}")
            hidden = correctness.get("final_hidden", {}); logits = correctness.get("logits", {})
            if hidden.get("relative_l2", math.inf) > thresholds.get("max_hidden_relative_l2", -1) or hidden.get("max_abs", math.inf) > thresholds.get("max_hidden_max_abs", -1) or logits.get("relative_l2", math.inf) > thresholds.get("max_logits_relative_l2", -1) or logits.get("max_abs", math.inf) > thresholds.get("max_logits_max_abs", -1) or logits.get("top_k_overlap", -1) < thresholds.get("minimum_top_k_overlap", math.inf): failures.append(f"correctness_threshold:{prefix}")
        except (EvidenceError, OSError, TypeError): failures.append(f"independent_validation:{prefix}")
        measurement_link = result.get("evidence", {}).get("measurement", {})
        try:
            measurement_path = contained(root, Path(measurement_link.get("path", "")), "measurement"); measurement = load(measurement_path, "measurement")
            if sha_file(measurement_path, "measurement") != measurement_link.get("sha256") or len(measurement.get("warmup_runs", [])) != 2 or len(measurement.get("measured_runs", [])) != 10: failures.append(f"measurement_schedule:{prefix}")
            else:
                rows = measurement["measured_runs"]; prompt = case.get("prompt_tokens", 0); generated = case.get("generated_tokens", 0)
                tps = [prompt * 1000.0 / row["prefill_ms"] if prompt else 0.0 for row in rows]; decode_tps = [generated * 1000.0 / row["decode_ms"] if generated else 0.0 for row in rows]
                expected_performance = {
                    "warmup_runs": 2, "measured_runs": 10,
                    "prefill_tokens_per_second_p50": percentile(tps, .5), "prefill_tokens_per_second_p95": percentile(tps, .95),
                    "ttft_ms_p50": percentile([row["ttft_ms"] for row in rows], .5), "ttft_ms_p95": percentile([row["ttft_ms"] for row in rows], .95),
                    "decode_tokens_per_second_p50": percentile(decode_tps, .5), "inter_token_latency_ms_p95": percentile([row["inter_token_latency_ms"] for row in rows], .95),
                    "end_to_end_ms_p50": percentile([row["end_to_end_ms"] for row in rows], .5), "end_to_end_ms_p95": percentile([row["end_to_end_ms"] for row in rows], .95),
                    "vram_peak_bytes": int(max(row["vram_peak_bytes"] for row in rows)), "workspace_peak_bytes": int(max(row["workspace_peak_bytes"] for row in rows)),
                    "actual_token_batch_width_p50": percentile([row["actual_token_batch_width"] for row in rows], .5), "actual_request_batch_width_p50": percentile([row["actual_request_batch_width"] for row in rows], .5),
                }
                performance = result.get("performance", {})
                for field, wanted in expected_performance.items():
                    actual = performance.get(field)
                    if isinstance(wanted, float):
                        if not isinstance(actual, (int, float)) or not math.isclose(actual, wanted, rel_tol=1e-12): failures.append(f"performance_recompute_{field}:{prefix}")
                    elif actual != wanted: failures.append(f"performance_recompute_{field}:{prefix}")
        except (EvidenceError, OSError, KeyError, ZeroDivisionError, TypeError): failures.append(f"measurement_link:{prefix}")
        state_link = result.get("evidence", {}).get("state", {})
        try:
            state_path = contained(root, Path(state_link.get("path", "")), "state"); state = load(state_path, "state")
            checks = state.get("checks", {})
            required_checks = ("finite_outputs", "shape_contract_passed", "kv_state_cache_passed", "scheduler_progress_passed", "chunk_equivalence_passed", "cancel_reset_passed", "publish_failure_reset_passed")
            reset = state.get("reset", {})
            if sha_file(state_path, "state") != state_link.get("sha256") or state.get("case_id") != case_id or state.get("status") != "valid" or any(checks.get(field) is not True for field in required_checks) or reset.get("attempted") is not True or reset.get("complete") is not True or reset.get("failed") is not False or state.get("memory", {}).get("oom") is not None or state.get("memory", {}).get("headroom_bytes", 0) <= 0 or any(state.get("fallback", {}).get(field) != 0 for field in ("unexpected_count", "fail_closed_count", "unsupported_count")): failures.append(f"state_reset_fallback_memory:{prefix}")
        except (EvidenceError, OSError): failures.append(f"state_link:{prefix}")
        try:
            baseline_path = contained(root, Path(result.get("regression", {}).get("baseline_result_path", "")), "baseline")
            baseline = load(baseline_path, "baseline"); baseline_sha = sha_file(baseline_path, "baseline")
            regression = result.get("regression", {}); performance = result.get("performance", {})
            expected_p50 = (performance["prefill_tokens_per_second_p50"] / baseline["prefill_tokens_per_second_p50"] - 1) * 100
            expected_p95 = (performance["prefill_tokens_per_second_p95"] / baseline["prefill_tokens_per_second_p95"] - 1) * 100
            if baseline_sha != identity.get("hash_binding", {}).get("baseline_result_sha256") or regression.get("baseline_result_sha256") != baseline_sha or not math.isclose(regression.get("prefill_p50_change_percent"), expected_p50, rel_tol=1e-12) or not math.isclose(regression.get("prefill_p95_change_percent"), expected_p95, rel_tol=1e-12) or regression.get("new_oom") is not (result.get("status") == "oom" and baseline.get("oom") is not True): failures.append(f"regression_recompute:{prefix}")
        except (EvidenceError, OSError, KeyError, ZeroDivisionError, TypeError): failures.append(f"regression_link:{prefix}")
    promotion = False
    return {"schema_version": "ullm.aq4_production_p2_evidence_validator.v2", "status": "valid" if not failures else "invalid", "failure_codes": sorted(set(failures)), "scheduled_case_count": len(case_by_id), "result_count": len(results), "complete_matrix": not missing and len(results) == len(case_by_id), "promotion_eligible": promotion, "production_live_execution": False, "review_note": "CPU/synthetic validation never promotes; GPU/live execution was not performed"}


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink(): raise EvidenceError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("xb") as target: target.write((json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode()); target.flush(); os.fsync(target.fileno())
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-root", type=Path, required=True); parser.add_argument("--expanded", type=Path, required=True); parser.add_argument("--identity", type=Path, required=True); parser.add_argument("--policy", type=Path, required=True); parser.add_argument("--source-oracle", type=Path, required=True); parser.add_argument("--result", type=Path, action="append", default=[]); parser.add_argument("--result-dir", type=Path); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.result and not args.result_dir: parser.error("at least one --result or --result-dir is required")
    try:
        root = args.run_root.resolve(strict=True); output = args.output.resolve(strict=False)
        if output != root and root not in output.parents: raise EvidenceError("output escapes run root")
        report = validate(args); atomic_write(args.output, report); print(json.dumps({"status": report["status"], "failure_count": len(report["failure_codes"])}, sort_keys=True)); return 0 if report["status"] == "valid" else 1
    except (EvidenceError, OSError, ValueError) as error:
        print(f"P2 evidence validation failed closed: {error}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
