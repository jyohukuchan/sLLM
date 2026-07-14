#!/usr/bin/env python3
"""Build one normative ``ullm.prefill_validation.v1`` P2 result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

OK_STATUSES = {"ok", "failed", "oom", "unsupported", "skipped"}
STRICT_TRACE_VALIDATOR = Path(__file__).with_name("validate-production-execution-trace.py")


class ResultError(ValueError): pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result: raise ResultError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024 * 1024: raise ResultError(f"{label} must be a bounded regular file")
    try: value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(ResultError(f"non-finite JSON number: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error: raise ResultError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict): raise ResultError(f"{label} root must be an object")
    return value


def sha_file(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file(): raise ResultError(f"{label} must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes: return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
def sha_bytes(value: bytes) -> str: return hashlib.sha256(value).hexdigest()


def contained(root: Path, path: Path, label: str, *, existing: bool = True) -> Path:
    root = root.resolve(strict=True); resolved = path.resolve(strict=existing)
    if resolved != root and root not in resolved.parents: raise ResultError(f"{label} escapes run root")
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


def percentile(values: list[float], quantile: float) -> float:
    if not values: raise ResultError("percentile input is empty")
    ordered = sorted(values); rank = (len(ordered) - 1) * quantile; lower = math.floor(rank); upper = math.ceil(rank)
    return ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def numeric(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < minimum: raise ResultError(f"{label} must be finite and >= {minimum}")
    return float(value)


def validate_measurements(value: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != "ullm.aq4_p2_measurements.v1" or value.get("case_id") != case["case_id"]: raise ResultError("measurement identity differs")
    warmup = value.get("warmup_runs"); measured = value.get("measured_runs")
    if not isinstance(warmup, list) or len(warmup) != 2 or not isinstance(measured, list) or len(measured) != 10: raise ResultError("measurement schedule must be 2 warmup + 10 measured")
    fields = ("prefill_ms", "ttft_ms", "decode_ms", "inter_token_latency_ms", "end_to_end_ms", "vram_peak_bytes", "workspace_peak_bytes", "actual_token_batch_width", "actual_request_batch_width")
    for group, name in ((warmup, "warmup"), (measured, "measured")):
        for index, row in enumerate(group):
            if not isinstance(row, dict) or set(row) != set(fields): raise ResultError(f"{name}[{index}] fields differ")
            for field in fields: numeric(row[field], f"{name}[{index}].{field}", minimum=1 if "width" in field else 0)
    prompt = case.get("prompt_tokens", 0); generated = case.get("generated_tokens", 0)
    prefill_tps = [prompt * 1000.0 / numeric(row["prefill_ms"], "prefill_ms", minimum=1e-12) if prompt else 0.0 for row in measured]
    decode_tps = [generated * 1000.0 / numeric(row["decode_ms"], "decode_ms", minimum=1e-12) if generated else 0.0 for row in measured]
    return {
        "warmup_runs": 2, "measured_runs": 10, "percentile_method": "linear_interpolation_rank_(n-1)*p",
        "prefill_tokens_per_second_p50": percentile(prefill_tps, .50), "prefill_tokens_per_second_p95": percentile(prefill_tps, .95),
        "ttft_ms_p50": percentile([float(row["ttft_ms"]) for row in measured], .50), "ttft_ms_p95": percentile([float(row["ttft_ms"]) for row in measured], .95),
        "decode_tokens_per_second_p50": percentile(decode_tps, .50), "inter_token_latency_ms_p95": percentile([float(row["inter_token_latency_ms"]) for row in measured], .95),
        "end_to_end_ms_p50": percentile([float(row["end_to_end_ms"]) for row in measured], .50), "end_to_end_ms_p95": percentile([float(row["end_to_end_ms"]) for row in measured], .95),
        "vram_peak_bytes": int(max(row["vram_peak_bytes"] for row in measured)), "workspace_peak_bytes": int(max(row["workspace_peak_bytes"] for row in measured)),
        "actual_token_batch_width_p50": percentile([float(row["actual_token_batch_width"]) for row in measured], .50), "actual_request_batch_width_p50": percentile([float(row["actual_request_batch_width"]) for row in measured], .50),
    }


def validate_state(value: dict[str, Any], case_id: str) -> None:
    if value.get("schema_version") != "ullm.aq4_p2_state_evidence.v1" or value.get("case_id") != case_id or value.get("status") != "valid": raise ResultError("state evidence identity/status differs")
    for field in ("finite_outputs", "shape_contract_passed", "kv_state_cache_passed", "scheduler_progress_passed", "chunk_equivalence_passed", "cancel_reset_passed", "publish_failure_reset_passed"):
        if value.get("checks", {}).get(field) is not True: raise ResultError(f"state check failed: {field}")
    reset = value.get("reset", {})
    if reset.get("attempted") is not True or reset.get("complete") is not True or reset.get("failed") is not False: raise ResultError("state reset is incomplete")
    fallback = value.get("fallback", {})
    if any(fallback.get(field) != 0 for field in ("unexpected_count", "fail_closed_count", "unsupported_count")) or fallback.get("reasons") != []: raise ResultError("state evidence contains fallback")
    memory = value.get("memory", {})
    if memory.get("oom") is not None or numeric(memory.get("headroom_bytes"), "state memory headroom") <= 0: raise ResultError("state memory evidence is unsafe")


def validate_trace(trace: dict[str, Any], case: dict[str, Any]) -> None:
    if trace.get("schema_version") != "ullm.production_execution_trace.v1" or trace.get("status") != "ok" or trace.get("scope") != case.get("scope"): raise ResultError("trace schema/status/scope differs")
    independent = trace.get("verification", {}).get("independent_validation", {})
    if independent.get("status") != "valid" or not isinstance(independent.get("report_sha256"), str): raise ResultError("trace independent validation is absent")
    reset = trace.get("state_commit", {}).get("reset", {})
    if reset.get("attempted") is not True or reset.get("complete") is not True or reset.get("failed") is not False: raise ResultError("trace reset is incomplete")
    fallback = trace.get("fallback", {})
    if any(fallback.get(field) != 0 for field in ("unexpected_fallback_count", "fail_closed_count", "unsupported_count")): raise ResultError("trace fallback is unsafe")
    memory = trace.get("memory", {})
    if memory.get("oom") is not None or memory.get("observer", {}).get("complete") is not True or numeric(memory.get("observed_headroom_bytes"), "trace headroom") <= 0: raise ResultError("trace memory evidence is unsafe")


def trace_terminal_sha(trace: dict[str, Any]) -> str:
    phases = trace.get("phases", [])
    terminal = {
        "request_summary": trace.get("request_summary"),
        "phase_terminal": [{"phase_id": item.get("phase_id"), "kind": item.get("kind"), "context_tokens_before": item.get("context_tokens_before"), "context_tokens_after": item.get("context_tokens_after"), "input_token_count": item.get("input_token_count"), "output_token_count": item.get("output_token_count")} for item in phases],
        "state_commit": trace.get("state_commit"), "fallback": trace.get("fallback"), "memory": trace.get("memory"),
    }
    return sha_bytes(canonical(terminal))


def bound_trace_identity(identity: dict[str, Any]) -> dict[str, Any]:
    artifacts = identity.get("artifacts", {}); manifest_path = Path(artifacts.get("served_model_manifest", ""))
    manifest = load(manifest_path, "served model manifest")
    public = manifest.get("public", {}); fmt = manifest.get("format", {}); worker = manifest.get("worker", {}); product = manifest.get("product", {})
    product_root = (manifest_path.parent / product.get("root", "")).resolve(strict=True)
    package_path = (product_root / product.get("package", {}).get("manifest_path", "")).resolve(strict=True)
    receipt_path = (manifest_path.parent / manifest.get("promotion", {}).get("receipt", "")).resolve(strict=True)
    worker_path = (manifest_path.parent / worker.get("binary", "")).resolve(strict=True)
    package_sha = sha_file(package_path, "trace package manifest")
    product_value = {"id": public.get("id"), "revision": public.get("revision"), "root": str(product_root), "package_manifest_sha256": package_sha}
    artifact_manifest = manifest.get("artifact", {}).get("manifest")
    artifact_path = (manifest_path.parent / artifact_manifest).resolve(strict=True) if artifact_manifest else None
    expected = {
        "model": {"id": public.get("id"), "revision": public.get("revision"), "format_id": fmt.get("format_id"), "implementation_id": fmt.get("implementation_id")},
        "served_model_manifest_sha256": sha_file(manifest_path, "served model manifest"),
        "worker": {"protocol": worker.get("protocol"), "binary_sha256": sha_file(worker_path, "trace worker")},
        "artifact": {"manifest_sha256": sha_file(artifact_path, "artifact manifest") if artifact_path else None, "content_sha256": manifest.get("artifact", {}).get("content_sha256")},
        "package": {"manifest_sha256": package_sha},
        "product": {"id": public.get("id"), "revision": public.get("revision"), "identity_sha256": sha_bytes(canonical(product_value)), "promotion_receipt_sha256": sha_file(receipt_path, "promotion receipt")},
    }
    return expected


def validate_trace_association(trace: dict[str, Any], case: dict[str, Any], raw: dict[str, Any], identity: dict[str, Any], measurement: dict[str, Any], trace_path: Path, trace_sha: str) -> None:
    summary = trace.get("request_summary", {})
    if case.get("fixture_id") != case.get("case_id"):
        raise ResultError("case fixture_id must equal case_id")
    contract_fields = ("fixture_id", "scope", "phase", "mode", "prompt_tokens", "cached_prefix_tokens", "context_tokens", "decode_start_tokens", "generated_tokens", "prefill_requested_m", "resolved_m", "decode_request_count", "sampling", "control_id", "format_id", "implementation_id", "device")
    if raw.get("case_contract") != {key: case.get(key) for key in contract_fields}:
        raise ResultError("raw case contract differs")
    expected_summary = {
        "fixture_id": case.get("fixture_id"), "request_count": case.get("request_count", case.get("decode_request_count")),
        "prompt_token_count": case.get("prompt_tokens"), "cached_prefix_token_count": case.get("cached_prefix_tokens"),
        "context_tokens_at_decode_start": case.get("decode_start_tokens"), "generated_token_count": case.get("generated_tokens"),
    }
    if any(wanted is None or summary.get(field) != wanted for field, wanted in expected_summary.items()):
        raise ResultError("trace request/case association differs")
    phases = trace.get("phases", [])
    matching = [item for item in phases if item.get("kind") == case.get("phase")]
    if len(matching) != 1: raise ResultError("trace phase/case association differs")
    phase = matching[0]
    expected_mode = {"all_m1": "cold", "cold_batched": "cold", "cached_prefix_chunked": "cached_prefix"}.get(case.get("mode"))
    if case.get("phase") != "decode" and phase.get("prefill_mode") != expected_mode: raise ResultError("trace prefill mode differs")
    expected_phase = {
        "input_token_count": case.get("prompt_tokens") if case.get("phase") != "decode" else case.get("generated_tokens"),
        "cached_prefix_token_count": case.get("cached_prefix_tokens"),
        "context_tokens_before": case.get("cached_prefix_tokens") if case.get("phase") != "decode" else case.get("decode_start_tokens"),
        "context_tokens_after": case.get("context_tokens") if case.get("phase") != "decode" else case.get("decode_start_tokens", 0) + case.get("generated_tokens", 0),
        "actual_token_batch_width": case.get("resolved_m") if case.get("phase") != "decode" else 1,
        "actual_request_batch_width": case.get("decode_request_count", case.get("request_count")),
        "request_count": case.get("decode_request_count", case.get("request_count")),
    }
    if any(wanted is None or phase.get(field) != wanted for field, wanted in expected_phase.items()): raise ResultError("trace phase shape/width/context differs")
    if case.get("phase") != "decode" and (case.get("prefill_requested_m") is None or phase.get("chunk_width_tokens") != case.get("resolved_m")): raise ResultError("trace requested/resolved width differs")
    executor = trace.get("executor", {}); device = case.get("device", {}); trace_device = executor.get("device", {})
    expected_device = {"backend": device.get("backend"), "name": device.get("name"), "architecture": device.get("architecture"), "runtime_device_index": device.get("runtime_device_index")}
    actual_device = {"backend": executor.get("backend"), "name": trace_device.get("name"), "architecture": trace_device.get("architecture"), "runtime_device_index": trace_device.get("runtime_device_index")}
    if any(value is None for value in expected_device.values()) or actual_device != expected_device: raise ResultError("trace device association differs")
    trace_identity = trace.get("identity", {}); model = identity.get("model_identity", {})
    if trace_identity != bound_trace_identity(identity): raise ResultError("trace manifest/product identity differs")
    if case.get("format_id") != trace_identity.get("model", {}).get("format_id") or case.get("implementation_id") != trace_identity.get("model", {}).get("implementation_id"): raise ResultError("trace case format/implementation differs")
    if trace_identity.get("model") != {key: model.get(key) for key in ("id", "revision", "format_id", "implementation_id")}:
        raise ResultError("trace model identity differs")
    hashes = identity.get("hash_binding", {})
    for trace_field, identity_field in (("served_model_manifest_sha256", "served_model_manifest_sha256"),):
        if trace_identity.get(trace_field) != hashes.get(identity_field): raise ResultError("trace served identity differs")
    if trace_identity.get("worker", {}).get("binary_sha256") != hashes.get("worker_binary_sha256") or trace_identity.get("package", {}).get("manifest_sha256") != hashes.get("package_manifest_sha256"):
        raise ResultError("trace worker/package identity differs")
    artifact = trace_identity.get("artifact", {})
    if artifact.get("manifest_sha256") is not None and artifact.get("manifest_sha256") != hashes.get("artifact_manifest_sha256"): raise ResultError("trace artifact identity differs")
    if artifact.get("content_sha256") is not None and artifact.get("content_sha256") != hashes.get("artifact_content_sha256"): raise ResultError("trace artifact content differs")
    if trace.get("sampling") is not None and trace.get("sampling") != case.get("sampling"): raise ResultError("trace sampling differs")
    if trace.get("control") is not None and trace.get("control") != case.get("control"): raise ResultError("trace control differs")
    raw_link = raw.get("links", {}).get("trace")
    expected_link = {"path": str(trace_path.resolve()), "sha256": trace_sha, "trace_id": trace.get("trace_id")}
    if raw_link != expected_link: raise ResultError("raw trace path/hash/id association differs")
    aggregation = measurement.get("trace_aggregation", {})
    expected_times = {item.get("phase_id"): item.get("wall_time_ms") for item in phases}
    if aggregation.get("schema_version") != "ullm.aq4_p2_trace_aggregation.v1" or aggregation.get("case_id") != case.get("case_id") or aggregation.get("trace_id") != trace.get("trace_id") or aggregation.get("trace_sha256") != trace_sha or aggregation.get("sample_count") != len(measurement.get("measured_runs", [])) or aggregation.get("phase_wall_time_ms") != expected_times or aggregation.get("terminal_audit_sha256") != trace_terminal_sha(trace):
        raise ResultError("trace measurement aggregation differs")


def validate_trace_bundle(args: argparse.Namespace, root: Path, case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {
        "trace": args.trace, "manifest": args.trace_manifest,
        "executor record": args.trace_executor_record, "binding": args.trace_binding,
        "detached report": args.trace_report,
    }
    if any(path is None for path in required.values()):
        raise ResultError("trace requires manifest, executor record, binding, and detached report")
    for label, path in required.items():
        assert path is not None
        contained(root, path, f"trace {label}")
    for source in args.trace_source:
        contained(root, source, "trace source")
    command = [
        sys.executable, str(STRICT_TRACE_VALIDATOR),
        "--trace", str(args.trace), "--manifest", str(args.trace_manifest),
        "--executor-record", str(args.trace_executor_record),
        "--binding", str(args.trace_binding), "--report", str(args.trace_report),
    ]
    for source in args.trace_source:
        command.extend(("--source-trace", str(source)))
    try:
        completed = subprocess.run(command, cwd=STRICT_TRACE_VALIDATOR.parent.parent, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ResultError(f"strict trace validator failed to run: {error}") from error
    if completed.returncode != 0:
        raise ResultError(f"strict trace validation failed: {completed.stderr.strip()[:512]}")
    try:
        strict_report = json.loads(completed.stdout, object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(ResultError(f"non-finite strict report number: {item}")))
    except (json.JSONDecodeError, ResultError) as error:
        raise ResultError(f"strict trace validator returned invalid JSON: {error}") from error
    trace = load(args.trace, "trace")
    detached = load(args.trace_report, "detached trace report")
    validate_trace(trace, case)
    expected_promotion = case.get("scope") == "production_server" and trace.get("status") == "ok"
    if strict_report.get("schema_version") != "ullm.production_execution_trace_validator.v1" or strict_report.get("status") != "valid" or strict_report.get("trace_sha256") != sha_file(args.trace, "trace") or strict_report.get("executor_record_sha256") != sha_file(args.trace_executor_record, "trace executor record") or strict_report.get("scope") != case.get("scope") or strict_report.get("promotion_eligible") is not expected_promotion:
        raise ResultError("strict trace validator report fields differ")
    if detached.get("schema_version") != "ullm.production_execution_trace_validator.v1" or detached.get("status") != "valid" or detached.get("scope") != case.get("scope") or detached.get("executor_record_sha256") != sha_file(args.trace_executor_record, "trace executor record") or detached.get("promotion_eligible") is not expected_promotion:
        raise ResultError("detached trace validator report fields differ")
    if trace.get("verification", {}).get("independent_validation", {}).get("report_sha256") != sha_file(args.trace_report, "detached trace report"):
        raise ResultError("trace detached report hash differs")
    links = {
        "manifest": {"path": str(args.trace_manifest.resolve()), "sha256": sha_file(args.trace_manifest, "trace manifest")},
        "executor_record": {"path": str(args.trace_executor_record.resolve()), "sha256": sha_file(args.trace_executor_record, "trace executor record")},
        "binding": {"path": str(args.trace_binding.resolve()), "sha256": sha_file(args.trace_binding, "trace binding")},
        "detached_report": {"path": str(args.trace_report.resolve()), "sha256": sha_file(args.trace_report, "detached trace report"), "report": detached},
        "source_traces": [{"path": str(path.resolve()), "sha256": sha_file(path, "trace source")} for path in args.trace_source],
        "strict_validation": strict_report,
    }
    return trace, links


def validate_source_oracle(value: dict[str, Any], validation: dict[str, Any], source_sha: str) -> None:
    if value.get("schema_version") != "ullm.qwen35_aq4_source_oracle.v1" or value.get("oracle_kind") != "independent_source" or value.get("status") not in {"available", "fixture"}: raise ResultError("source oracle is not an independent source artifact")
    if validation.get("schema_version") != "ullm.qwen35_aq4_p2_oracle_validator.v1" or validation.get("status") != "valid" or validation.get("oracle_kind") != "independent_source" or validation.get("manifest_sha256") != source_sha: raise ResultError("source oracle independent validation artifact differs")


def validate_independent(value: dict[str, Any], case: dict[str, Any], raw_sha: str, source_sha: str, path_sha: str | None, trace_sha: str | None, policy: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != "ullm.aq4_p2_independent_validation.v1" or value.get("status") != "valid" or value.get("validator_independent") is not True or value.get("case_id") != case["case_id"] or value.get("case_sha256") != case["case_sha256"]: raise ResultError("independent validation identity differs")
    expected = {"raw_sha256": raw_sha, "source_oracle_sha256": source_sha, "path_oracle_result_sha256": path_sha, "trace_sha256": trace_sha}
    if any(value.get(field) != wanted for field, wanted in expected.items()): raise ResultError("independent validation artifact hash binding differs")
    correctness = value.get("correctness", {})
    for field in ("finite", "shape_contract_passed", "path_oracle_passed", "source_oracle_passed", "greedy_tokens_exact", "kv_state_cache_passed", "scheduler_progress_passed", "chunk_equivalence_passed", "cancel_reset_passed", "publish_failure_reset_passed"):
        required = case.get("mode") != "all_m1" if field == "path_oracle_passed" else True
        if required and correctness.get(field) is not True: raise ResultError(f"independent correctness failed: {field}")
    thresholds = policy.get("correctness_thresholds", {})
    hidden = correctness.get("final_hidden", {}); logits = correctness.get("logits", {})
    if numeric(hidden.get("relative_l2"), "hidden relative_l2") > thresholds.get("max_hidden_relative_l2") or numeric(hidden.get("max_abs"), "hidden max_abs") > thresholds.get("max_hidden_max_abs"): raise ResultError("hidden correctness threshold failed")
    if numeric(logits.get("relative_l2"), "logits relative_l2") > thresholds.get("max_logits_relative_l2") or numeric(logits.get("max_abs"), "logits max_abs") > thresholds.get("max_logits_max_abs") or logits.get("top_k_overlap", -1) < thresholds.get("minimum_top_k_overlap"): raise ResultError("logit correctness threshold failed")
    return correctness


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = args.run_root.resolve(strict=True)
    paths = [(args.case, "case"), (args.expanded, "expanded"), (args.raw, "raw"), (args.identity, "identity"), (args.policy, "policy"), (args.source_oracle, "source oracle"), (args.source_oracle_validation, "source oracle validation"), (args.independent_validation, "independent validation")]
    if args.path_oracle_result: paths.append((args.path_oracle_result, "path oracle result"))
    if args.trace: paths.append((args.trace, "trace"))
    elif any((args.trace_manifest, args.trace_executor_record, args.trace_binding, args.trace_report, args.trace_source)):
        raise ResultError("trace validation artifacts cannot be supplied without a trace")
    for path, label in paths: contained(root, path, label)
    contained(root, args.output, "output", existing=False)
    case = load(args.case, "case"); expanded = load(args.expanded, "expanded"); raw = load(args.raw, "raw"); identity = load(args.identity, "identity"); policy = load(args.policy, "policy")
    source = load(args.source_oracle, "source oracle"); source_validation = load(args.source_oracle_validation, "source oracle validation"); independent = load(args.independent_validation, "independent validation")
    if expanded.get("schema_version") != "ullm.aq4_production_p2_expanded.v2" or case.get("case_sha256") != case_hash(case) or len([item for item in expanded.get("cases", []) if item == case]) != 1: raise ResultError("case/expanded binding differs")
    if raw.get("schema_version") != "ullm.aq4_production_p2_raw_result.v2" or raw.get("case_id") != case.get("case_id") or raw.get("case_sha256") != case.get("case_sha256"): raise ResultError("raw case binding differs")
    status = raw.get("status")
    if status not in OK_STATUSES or raw.get("immutable_status") is not (status != "ok"): raise ResultError("raw immutable status differs")
    if raw.get("links", {}).get("expanded", {}).get("sha256") != sha_file(args.expanded, "expanded") or identity.get("expanded_manifest_sha256") != sha_file(args.expanded, "expanded"): raise ResultError("raw/identity expanded binding differs")
    if identity.get("schema_version") != "ullm.aq4_production_p2_identity.v2" or identity.get("status") != "bound" or identity.get("identity_sha256") != identity_hash(identity) or raw.get("links", {}).get("identity", {}).get("sha256") != sha_file(args.identity, "identity"): raise ResultError("identity self-binding differs")
    if policy.get("status") != "bound" or policy.get("hash_binding", {}).get("policy_sha256") != policy_hash(policy) or identity.get("policy_sha256") != policy.get("hash_binding", {}).get("policy_sha256") or raw.get("links", {}).get("policy", {}).get("sha256") != sha_file(args.policy, "policy"): raise ResultError("bound policy differs")
    source_sha = sha_file(args.source_oracle, "source oracle"); validate_source_oracle(source, source_validation, source_sha)
    if identity.get("hash_binding", {}).get("source_oracle_sha256") != source_sha: raise ResultError("source oracle identity differs")
    measurement_path = Path(raw.get("links", {}).get("measurement", {}).get("path", "")); state_path = Path(raw.get("links", {}).get("state", {}).get("path", ""))
    contained(root, measurement_path, "measurement"); contained(root, state_path, "state")
    if sha_file(measurement_path, "measurement") != raw["links"]["measurement"]["sha256"] or sha_file(state_path, "state") != raw["links"]["state"]["sha256"]: raise ResultError("raw evidence hash differs")
    measurement = load(measurement_path, "measurement"); performance = validate_measurements(measurement, case); state = load(state_path, "state"); validate_state(state, case["case_id"])
    path_sha = None; path_link = None
    if case.get("mode") in {"cold_batched", "cached_prefix_chunked"}:
        if args.path_oracle_result is None: raise ResultError("optimized case requires a path oracle result")
        path_result = load(args.path_oracle_result, "path oracle result"); path_sha = sha_file(args.path_oracle_result, "path oracle result")
        if path_result.get("schema_version") != "ullm.prefill_validation.v1" or path_result.get("case_id") != case.get("path_oracle_case_id") or path_result.get("status") != "ok" or path_result.get("workload", {}).get("baseline_mode") != "all_m1": raise ResultError("path oracle result identity/status differs")
        if path_result.get("identity", {}).get("sha256") != sha_file(args.identity, "identity") or path_result.get("oracles", {}).get("source_oracle", {}).get("sha256") != source_sha or path_result.get("oracles", {}).get("threshold_policy", {}).get("self_sha256") != policy.get("hash_binding", {}).get("policy_sha256"): raise ResultError("path oracle artifact/source/policy identity differs")
        for field in ("phase", "cached_prefix_tokens", "prompt_tokens", "prefill_requested_m", "scope", "control_id"):
            left = path_result.get("workload", {}).get(field) if field in {"phase", "cached_prefix_tokens", "prompt_tokens", "prefill_requested_m"} else path_result.get(field)
            if left != case.get(field): raise ResultError(f"path oracle same-state field differs: {field}")
        path_link = {"mode": "all_m1", "result_path": str(args.path_oracle_result.resolve()), "result_sha256": path_sha}
    elif args.path_oracle_result is not None: raise ResultError("all-M1/decode case must not attach a path oracle result")
    trace_sha = None; trace_link = None
    if args.trace:
        trace, trace_validation = validate_trace_bundle(args, root, case); trace_sha = sha_file(args.trace, "trace")
        validate_trace_association(trace, case, raw, identity, measurement, args.trace, trace_sha)
        trace_link = {"schema_version": trace["schema_version"], "trace_id": trace.get("trace_id"), "path": str(args.trace.resolve()), "sha256": trace_sha, "scope": trace.get("scope"), "validation": trace_validation}
    if case.get("scope") == "production_server" and status == "ok" and trace_link is None: raise ResultError("production-server ok result requires a trace")
    raw_sha = sha_file(args.raw, "raw"); correctness = validate_independent(independent, case, raw_sha, source_sha, path_sha, trace_sha, policy)
    baseline_path = Path(identity.get("artifacts", {}).get("baseline_result", "")); contained(root, baseline_path, "baseline")
    baseline = load(baseline_path, "baseline"); baseline_sha = sha_file(baseline_path, "baseline")
    if baseline_sha != identity.get("hash_binding", {}).get("baseline_result_sha256"): raise ResultError("baseline identity differs")
    baseline_p50 = numeric(baseline.get("prefill_tokens_per_second_p50"), "baseline p50", minimum=1e-12); baseline_p95 = numeric(baseline.get("prefill_tokens_per_second_p95"), "baseline p95", minimum=1e-12)
    p50_change = (performance["prefill_tokens_per_second_p50"] / baseline_p50 - 1) * 100
    p95_change = (performance["prefill_tokens_per_second_p95"] / baseline_p95 - 1) * 100
    prefill_policy = policy.get("performance_thresholds", {}).get("prefill", {})
    new_oom = status == "oom" and baseline.get("oom") is not True
    regression = {"baseline_result_path": str(baseline_path), "baseline_result_sha256": baseline_sha, "prefill_p50_change_percent": p50_change, "prefill_p95_change_percent": p95_change, "new_oom": new_oom, "passed": status == "ok" and not new_oom and p50_change >= -100 * prefill_policy.get("p50_regression_stop_fraction", 0) and p95_change >= -100 * prefill_policy.get("p95_regression_stop_fraction", 0)}
    mode = raw.get("mode")
    reasons = ["producer_non_authoritative"]
    if mode == "cpu_synthetic": reasons.append("cpu_synthetic_never_promotes")
    if case.get("scope") != "production_server": reasons.append(f"scope_{case.get('scope')}_not_production_server")
    result = {
        "schema_version": "ullm.prefill_validation.v1", "run_id": root.name, "case_id": case["case_id"], "case_sha256": case["case_sha256"], "status": status, "scope": case["scope"], "control_id": case["control_id"],
        "case": {"path": str(args.case.resolve()), "sha256": sha_file(args.case, "case")},
        "identity": {"path": str(args.identity.resolve()), "sha256": sha_file(args.identity, "identity"), "binding_sha256": identity.get("identity_sha256"), "format_id": case["format_id"], "build_git_commit": identity.get("build_git_commit")},
        "workload": {key: case.get(key) for key in ("phase", "baseline_mode", "prompt_tokens", "cached_prefix_tokens", "context_tokens", "decode_start_tokens", "prefill_requested_m", "resolved_m", "decode_request_count", "generated_tokens")},
        "evidence": {"raw_result": {"path": str(args.raw.resolve()), "sha256": raw_sha, "status": status}, "measurement": raw["links"]["measurement"], "state": raw["links"]["state"], "execution_trace": trace_link, "independent_validation": {"path": str(args.independent_validation.resolve()), "sha256": sha_file(args.independent_validation, "independent validation")}, "source_oracle_validation": {"path": str(args.source_oracle_validation.resolve()), "sha256": sha_file(args.source_oracle_validation, "source oracle validation")}},
        "oracles": {"path_oracle": path_link, "source_oracle": {"path": str(args.source_oracle.resolve()), "sha256": source_sha, "independent": True}, "threshold_policy": {"policy_id": policy.get("policy_id"), "path": str(args.policy.resolve()), "sha256": sha_file(args.policy, "policy"), "self_sha256": policy.get("hash_binding", {}).get("policy_sha256")}},
        "correctness": correctness, "performance": performance, "regression": regression,
        "promotion": {"eligible": False, "reason_codes": reasons, "required_next_scope": "independent_complete_matrix_validator"},
        "error": raw.get("failure_reason"), "notes": [],
    }
    return result


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink(): raise ResultError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("xb") as target: target.write((json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode()); target.flush(); os.fsync(target.fileno())
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run_root", "case", "expanded", "raw", "identity", "policy", "source_oracle", "source_oracle_validation", "independent_validation", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--path-oracle-result", type=Path); parser.add_argument("--trace", type=Path)
    parser.add_argument("--trace-manifest", type=Path); parser.add_argument("--trace-executor-record", type=Path); parser.add_argument("--trace-binding", type=Path); parser.add_argument("--trace-report", type=Path); parser.add_argument("--trace-source", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    try:
        result = build(args); atomic_write(args.output, result); print(json.dumps({"status": "ok", "promotion_eligible": False}, sort_keys=True)); return 0
    except (ResultError, OSError, ValueError) as error:
        print(f"P2 validation result failed: {error}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
