#!/usr/bin/env python3
"""Build one normative ``ullm.prefill_validation.v1`` P2 result."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

OK_STATUSES = {"ok", "failed", "oom", "unsupported", "skipped"}
STRICT_TRACE_VALIDATOR = Path(__file__).with_name("validate-production-execution-trace.py")
FIXED_EXPANDER = Path(__file__).with_name("expand-aq4-production-p2.py")
STANDARD_MANIFEST = Path(__file__).parent.parent / "benchmarks/workloads/aq4-production-opt-p2-case-manifest-v0.1.json"
CALIBRATION_BINDING_SCHEMA = "ullm.aq4_p2_calibration_evidence.v1"
CALIBRATION_COMPARISON_SCHEMA = "ullm.qwen35_aq4_calibration_comparison.v1"
CALIBRATION_METRICS = ("max_hidden_relative_l2", "max_hidden_max_abs", "max_logits_relative_l2", "max_logits_max_abs", "minimum_top_k_overlap")
SHA256_CHARS = frozenset("0123456789abcdef")


class ResultError(ValueError): pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result: raise ResultError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode, info.st_nlink


def _open_regular(path: Path, label: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try: descriptor = os.open(path, flags)
    except OSError as error: raise ResultError(f"{label} is unavailable: {error}") from error
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(descriptor); raise ResultError(f"{label} must be a single-link regular file")
    return descriptor


def _read_stable(path: Path, label: str, maximum: int) -> bytes:
    descriptor = _open_regular(path, label)
    try:
        before = os.fstat(descriptor)
        if before.st_size > maximum: raise ResultError(f"{label} exceeds {maximum} bytes")
        chunks: list[bytes] = []; remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk: break
            chunks.append(chunk); remaining -= len(chunk)
        if remaining == 0: raise ResultError(f"{label} exceeds {maximum} bytes")
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after): raise ResultError(f"{label} changed while being read")
        return b"".join(chunks)
    finally: os.close(descriptor)


def load(path: Path, label: str) -> dict[str, Any]:
    try: value = json.loads(_read_stable(path, label, 32 * 1024 * 1024).decode("utf-8"), object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(ResultError(f"non-finite JSON number: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error: raise ResultError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict): raise ResultError(f"{label} root must be an object")
    return value


def sha_file(path: Path, label: str) -> str:
    descriptor = _open_regular(path, label); digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        while chunk := os.read(descriptor, 1024 * 1024): digest.update(chunk)
        if _identity(before) != _identity(os.fstat(descriptor)): raise ResultError(f"{label} changed while being hashed")
        return digest.hexdigest()
    finally: os.close(descriptor)


def canonical(value: Any) -> bytes: return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
def sha_bytes(value: bytes) -> str: return hashlib.sha256(value).hexdigest()


def _reject_symlink_components(path: Path, label: str, *, allow_missing_leaf: bool = False) -> None:
    absolute = Path(os.path.abspath(path)); components = [Path(absolute.anchor)]
    components.extend(Path(absolute.anchor, *absolute.parts[1:index]) for index in range(1, len(absolute.parts) + 1))
    for index, component in enumerate(components):
        try: info = component.lstat()
        except FileNotFoundError:
            if allow_missing_leaf and index == len(components) - 1: return
            raise ResultError(f"{label} path component is missing: {component}")
        if stat.S_ISLNK(info.st_mode): raise ResultError(f"{label} path component is a symlink: {component}")


def contained(root: Path, path: Path, label: str, *, existing: bool = True) -> Path:
    lexical_root = Path(os.path.abspath(root)); lexical = Path(os.path.abspath(path))
    if lexical != lexical_root and lexical_root not in lexical.parents: raise ResultError(f"{label} escapes run root")
    _reject_symlink_components(lexical_root, "run root")
    _reject_symlink_components(lexical, label, allow_missing_leaf=not existing)
    resolved_root = lexical_root.resolve(strict=True); resolved = lexical.resolve(strict=existing)
    if resolved != resolved_root and resolved_root not in resolved.parents: raise ResultError(f"{label} escapes run root")
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


def validate_complete_expansion(expanded: dict[str, Any], identity: dict[str, Any], root: Path) -> None:
    manifest_path = Path(identity.get("artifacts", {}).get("planning_manifest", ""))
    resolved = manifest_path.resolve(strict=True)
    if resolved != STANDARD_MANIFEST.resolve() and resolved != root.resolve() and root.resolve() not in resolved.parents: raise ResultError("planning manifest escapes run root")
    manifest = load(resolved, "planning manifest")
    if sha_file(resolved, "planning manifest") != identity.get("planning_manifest_sha256"): raise ResultError("planning manifest identity differs")
    spec = importlib.util.spec_from_file_location("aq4_p2_builder_expander", FIXED_EXPANDER)
    if spec is None or spec.loader is None: raise ResultError("fixed expander is unavailable")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    expected = module.expand(json.loads(json.dumps(manifest)), sha_file(resolved, "planning manifest"))
    if expanded != expected: raise ResultError("expanded matrix differs from complete planning expansion")


def percentile(values: list[float], quantile: float) -> float:
    if not values: raise ResultError("percentile input is empty")
    ordered = sorted(values); rank = (len(ordered) - 1) * quantile; lower = math.floor(rank); upper = math.ceil(rank)
    return ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def numeric(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < minimum: raise ResultError(f"{label} must be finite and >= {minimum}")
    return float(value)


def exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        actual = set(value) if isinstance(value, dict) else set()
        raise ResultError(f"{label} fields differ: missing={sorted(fields - actual)} extra={sorted(actual - fields)}")
    return value


def ensure_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in SHA256_CHARS for char in value): raise ResultError(f"{label} must be a lowercase SHA-256")
    return value


def calibration_identity(identity: dict[str, Any], source_sha: str, policy: dict[str, Any]) -> dict[str, Any]:
    hashes = identity.get("hash_binding", {})
    return {
        "model": identity.get("model_identity"),
        "source_oracle_sha256": source_sha,
        "package_content_sha256": hashes.get("package_content_sha256"),
        "package_manifest_sha256": hashes.get("package_manifest_sha256"),
        "worker_binary_sha256": hashes.get("worker_binary_sha256"),
        "policy_sha256": policy.get("hash_binding", {}).get("policy_sha256"),
    }


def calibration_step_count(case: dict[str, Any]) -> int:
    return int(case.get("generated_tokens", 0)) if case.get("phase") == "decode" else 1


def calibration_thresholds(policy: dict[str, Any]) -> dict[str, float]:
    values = policy.get("correctness_thresholds")
    if not isinstance(values, dict): raise ResultError("calibration correctness thresholds are absent")
    result: dict[str, float] = {}
    for field in CALIBRATION_METRICS:
        value = numeric(values.get(field), f"policy {field}")
        if field == "minimum_top_k_overlap" and (not float(value).is_integer() or value > 10): raise ResultError("policy minimum_top_k_overlap must be an integer in 0..10")
        result[field] = value
    return result


def validate_calibration_comparison(value: dict[str, Any], compare_kind: str, thresholds: dict[str, float]) -> dict[str, Any]:
    exact(value, {"schema_version", "status", "promotion_eligible", "created_utc", "compare_kind", "reference", "candidate", "vector_contract", "rows", "summary", "observed_values_only"}, "calibration comparison")
    if value["schema_version"] != CALIBRATION_COMPARISON_SCHEMA or value["status"] != "valid" or value["promotion_eligible"] is not False or value["observed_values_only"] is not True or value["compare_kind"] != compare_kind or not isinstance(value["created_utc"], str): raise ResultError("calibration comparison schema/status/kind differs")
    reference = exact(value["reference"], {"path", "manifest_sha256", "schema_version", "oracle_kind"}, "calibration reference")
    candidate = exact(value["candidate"], {"path", "manifest_sha256", "schema_version", "oracle_kind"}, "calibration candidate")
    for side, item in (("reference", reference), ("candidate", candidate)):
        if not isinstance(item["path"], str) or not item["path"] or not isinstance(item["schema_version"], str): raise ResultError(f"calibration {side} identity differs")
        ensure_sha(item["manifest_sha256"], f"calibration {side} manifest")
    expected_kinds = {"source_gate": ("independent_source_full", "aq4_target"), "path_gate": ("aq4_target", "aq4_optimized")}
    if (reference["oracle_kind"], candidate["oracle_kind"]) != expected_kinds[compare_kind]: raise ResultError("calibration reference/candidate roles differ")
    contract = exact(value["vector_contract"], {"hidden_shape", "logits_shape", "dtype", "endianness", "metric_denominator", "top_k"}, "calibration vector contract")
    if contract != {"hidden_shape": [4096], "logits_shape": [248320], "dtype": "f32", "endianness": "little", "metric_denominator": "max(reference_l2,1e-30)", "top_k": 10}: raise ResultError("calibration vector contract differs")
    rows = exact(value["rows"], {"file", "record_count", "sha256"}, "calibration rows")
    if rows["file"] != "rows.jsonl" or not isinstance(rows["record_count"], int) or isinstance(rows["record_count"], bool) or rows["record_count"] <= 0: raise ResultError("calibration rows contract differs")
    ensure_sha(rows["sha256"], "calibration rows")
    summary = exact(value["summary"], {"row_count", "nonfinite_rows", "greedy_mismatch_rows", *CALIBRATION_METRICS}, "calibration summary")
    if summary["row_count"] != rows["record_count"] or summary["nonfinite_rows"] != 0 or summary["greedy_mismatch_rows"] != 0: raise ResultError("calibration row status/count differs")
    metrics = {field: numeric(summary[field], f"calibration {field}") for field in CALIBRATION_METRICS}
    if not metrics["minimum_top_k_overlap"].is_integer() or metrics["minimum_top_k_overlap"] > 10: raise ResultError("calibration top-k overlap differs")
    if metrics["max_hidden_relative_l2"] > thresholds["max_hidden_relative_l2"] or metrics["max_hidden_max_abs"] > thresholds["max_hidden_max_abs"] or metrics["max_logits_relative_l2"] > thresholds["max_logits_relative_l2"] or metrics["max_logits_max_abs"] > thresholds["max_logits_max_abs"] or metrics["minimum_top_k_overlap"] < thresholds["minimum_top_k_overlap"]: raise ResultError("calibration comparison exceeds pre-bound correctness policy")
    return {"reference": reference, "candidate": candidate, "metrics": {"row_count": summary["row_count"], **metrics}}


def calibration_artifact_manifest(root: Path, side: dict[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
    artifact_root = contained(root, Path(side["path"]), f"{label} root")
    if not artifact_root.is_dir(): raise ResultError(f"{label} root must be a directory")
    manifest_path = contained(root, artifact_root / "manifest.json", f"{label} manifest")
    if sha_file(manifest_path, f"{label} manifest") != side["manifest_sha256"]: raise ResultError(f"{label} manifest hash differs")
    return manifest_path, load(manifest_path, f"{label} manifest")


def validate_source_calibration_reference(root: Path, side: dict[str, Any], source_path: Path, source: dict[str, Any], source_sha: str) -> dict[str, Any]:
    manifest_path, manifest = calibration_artifact_manifest(root, side, "independent source calibration")
    if manifest.get("schema_version") != "ullm.qwen35_aq4_source_calibration.v1" or manifest.get("oracle_kind") != "independent_source_full" or manifest.get("status") != "available": raise ResultError("independent source calibration manifest differs")
    parent = exact(manifest.get("parent_sampled_oracle"), {"path", "manifest_sha256", "schema_version"}, "source calibration parent")
    sampled_path = contained(root, Path(parent["path"]), "sampled source parent")
    if sampled_path != source_path.resolve() or parent["manifest_sha256"] != source_sha or parent["schema_version"] != source.get("schema_version") or sha_file(sampled_path, "sampled source parent") != source_sha: raise ResultError("source calibration sampled-v2 parent hash chain differs")
    full_identity = manifest.get("identity", {}); sampled_identity = source.get("identity", {})
    if not isinstance(full_identity, dict) or not isinstance(sampled_identity, dict): raise ResultError("source calibration parent identity is absent")
    for field in ("model_id", "model_revision"):
        if full_identity.get(field) != sampled_identity.get(field): raise ResultError(f"source calibration parent identity differs: {field}")
    for field in ("source_checkpoint", "tokenizer"):
        if full_identity.get(field, {}).get("aggregate_sha256") != sampled_identity.get(field, {}).get("aggregate_sha256"): raise ResultError(f"source calibration parent identity differs: {field}")
    return {"path": str(manifest_path), "sha256": side["manifest_sha256"], "identity": full_identity}


def validate_target_calibration_manifest(root: Path, side: dict[str, Any], expected_kind: str, case: dict[str, Any], identity: dict[str, Any], source_manifest_sha: str | None, source_manifest_path: str | None = None) -> dict[str, Any]:
    manifest_path, manifest = calibration_artifact_manifest(root, side, f"{expected_kind} target calibration")
    if manifest.get("schema_version") != "ullm.qwen35_aq4_target_calibration.v1" or manifest.get("oracle_kind") != expected_kind or manifest.get("status") != "available" or manifest.get("capture_complete") is not True or manifest.get("promotion_eligible") is not False: raise ResultError(f"{expected_kind} target calibration manifest differs")
    target_identity = manifest.get("identity", {}); hashes = identity.get("hash_binding", {}); model = identity.get("model_identity", {})
    expected_identity = {"format_id": model.get("format_id"), "implementation_id": model.get("implementation_id"), "package_content_sha256": hashes.get("package_content_sha256"), "package_manifest_sha256": hashes.get("package_manifest_sha256"), "worker_binary_sha256": hashes.get("worker_binary_sha256")}
    if not isinstance(target_identity, dict) or any(target_identity.get(field) != wanted for field, wanted in expected_identity.items()): raise ResultError(f"{expected_kind} target calibration identity differs")
    binding = manifest.get("binding", {})
    if binding.get("case_id") != case.get("case_id") or binding.get("case_sha256") != case.get("case_sha256") or binding.get("requested_m") != case.get("prefill_requested_m") or binding.get("resolved_m") != case.get("resolved_m"): raise ResultError(f"{expected_kind} target calibration case/prompt binding differs")
    device = binding.get("device", {}); expected_device = case.get("device", {})
    if not isinstance(device, dict) or device.get("requested_index") != expected_device.get("runtime_device_index") or any(device.get(field) != expected_device.get(field) for field in ("device_id", "backend", "name", "architecture")): raise ResultError(f"{expected_kind} target calibration device binding differs")
    source_link = binding.get("source", {}).get("manifest", {})
    if not isinstance(source_link, dict): raise ResultError(f"{expected_kind} target source link is absent")
    linked_source_sha = ensure_sha(source_link.get("sha256"), f"{expected_kind} target source manifest")
    linked_source_path = contained(root, Path(source_link.get("path", "")), f"{expected_kind} target source manifest")
    if sha_file(linked_source_path, f"{expected_kind} target source manifest") != linked_source_sha: raise ResultError(f"{expected_kind} target source manifest hash chain differs")
    if source_manifest_sha is not None and linked_source_sha != source_manifest_sha: raise ResultError(f"{expected_kind} target source manifest differs")
    if source_manifest_path is not None and linked_source_path != Path(source_manifest_path).resolve(): raise ResultError(f"{expected_kind} target source manifest path differs")
    return {"path": str(manifest_path), "sha256": side["manifest_sha256"], "source_manifest_path": str(linked_source_path), "source_manifest_sha256": linked_source_sha}


def validate_calibration_evidence(path: Path, compare_kind: str, root: Path, case: dict[str, Any], expanded: dict[str, Any], identity: dict[str, Any], policy: dict[str, Any], source_path: Path, source: dict[str, Any], source_sha: str, path_oracle: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = contained(root, path, f"{compare_kind} calibration evidence")
    value = load(resolved, f"{compare_kind} calibration evidence")
    fields = {"schema_version", "status", "compare_kind", "case", "canonical_case_sha256", "step_count", "identity", "comparison"}
    if compare_kind == "path_gate": fields |= {"path_oracle_case_id", "path_oracle_result_sha256", "path_oracle_calibration_manifest_sha256"}
    exact(value, fields, f"{compare_kind} calibration evidence")
    if value["schema_version"] != CALIBRATION_BINDING_SCHEMA or value["status"] != "valid" or value["compare_kind"] != compare_kind: raise ResultError(f"{compare_kind} calibration binding schema/status differs")
    if value["case"] != case or value["canonical_case_sha256"] != expanded.get("canonical_case_sha256") or value["step_count"] != calibration_step_count(case): raise ResultError(f"{compare_kind} calibration case/prompt/step/case-set binding differs")
    bound_identity = exact(value["identity"], {"model", "source_oracle_sha256", "package_content_sha256", "package_manifest_sha256", "worker_binary_sha256", "policy_sha256"}, f"{compare_kind} calibration identity")
    if bound_identity != calibration_identity(identity, source_sha, policy): raise ResultError(f"{compare_kind} calibration model/source/package/worker/device/policy identity differs")
    # Device identity is nested in the exact case object; keep it explicit in
    # the diagnostic boundary because case swaps between devices must fail.
    if value["case"].get("device") != case.get("device"): raise ResultError(f"{compare_kind} calibration device identity differs")
    comparison_link = exact(value["comparison"], {"path", "sha256"}, f"{compare_kind} calibration comparison link")
    comparison_path = contained(root, Path(comparison_link["path"]), f"{compare_kind} calibration comparison")
    comparison_sha = sha_file(comparison_path, f"{compare_kind} calibration comparison")
    if comparison_link["sha256"] != comparison_sha: raise ResultError(f"{compare_kind} calibration comparison hash differs")
    compared = validate_calibration_comparison(load(comparison_path, f"{compare_kind} calibration comparison"), compare_kind, calibration_thresholds(policy))
    metrics = compared["metrics"]
    if metrics["row_count"] != value["step_count"]: raise ResultError(f"{compare_kind} calibration step coverage differs")
    if compare_kind == "source_gate":
        source_manifest = validate_source_calibration_reference(root, compared["reference"], source_path, source, source_sha)
        target_manifest = validate_target_calibration_manifest(root, compared["candidate"], "aq4_target", case, identity, source_manifest["sha256"], source_manifest["path"])
        path_binding = None
    else:
        if path_oracle is None: raise ResultError("path calibration requires an all-M1 result chain")
        expected_path = {"path_oracle_case_id": path_oracle["case"]["case_id"], "path_oracle_result_sha256": path_oracle["result_sha256"], "path_oracle_calibration_manifest_sha256": path_oracle["calibration_manifest_sha256"]}
        if any(value[field] != wanted for field, wanted in expected_path.items()): raise ResultError("path calibration all-M1 result/manifest binding differs")
        reference_manifest = validate_target_calibration_manifest(root, compared["reference"], "aq4_target", path_oracle["case"], identity, path_oracle["source_manifest_sha256"], path_oracle["source_manifest_path"])
        if reference_manifest["sha256"] != path_oracle["calibration_manifest_sha256"] or Path(compared["reference"]["path"]).resolve() != Path(path_oracle["calibration_root_path"]).resolve(): raise ResultError("all-M1 target manifest/path to path comparison hash chain differs")
        target_manifest = validate_target_calibration_manifest(root, compared["candidate"], "aq4_optimized", case, identity, reference_manifest["source_manifest_sha256"], reference_manifest["source_manifest_path"])
        source_manifest = {"path": reference_manifest["source_manifest_path"], "sha256": reference_manifest["source_manifest_sha256"]}
        path_binding = expected_path
    result = {"path": str(resolved), "sha256": sha_file(resolved, f"{compare_kind} calibration evidence"), "comparison": {"path": str(comparison_path), "sha256": comparison_sha}, "manifests": {"reference_path": compared["reference"]["path"], "reference_sha256": compared["reference"]["manifest_sha256"], "candidate_path": compared["candidate"]["path"], "candidate_sha256": compared["candidate"]["manifest_sha256"], "source_path": source_manifest["path"], "source_sha256": source_manifest["sha256"]}, "metrics": metrics}
    if path_binding is not None: result["path_oracle"] = path_binding
    return result


def path_oracle_calibration_chain(path_result: dict[str, Any], path_result_sha: str, root: Path, expanded: dict[str, Any], identity: dict[str, Any], policy: dict[str, Any], source_path: Path, source: dict[str, Any], source_sha: str) -> dict[str, Any]:
    oracle_case_id = path_result.get("case_id")
    matches = [item for item in expanded.get("cases", []) if isinstance(item, dict) and item.get("case_id") == oracle_case_id]
    if len(matches) != 1: raise ResultError("path oracle result case is absent from expanded matrix")
    oracle_case = matches[0]
    calibration = exact(path_result.get("calibration"), {"source_gate", "path_gate"}, "path oracle result calibration")
    if calibration["path_gate"] is not None: raise ResultError("all-M1 path oracle result must not contain a path gate")
    source_link = exact(calibration["source_gate"], {"path", "sha256", "comparison", "manifests", "metrics"}, "path oracle source calibration link")
    rebuilt = validate_calibration_evidence(Path(source_link["path"]), "source_gate", root, oracle_case, expanded, identity, policy, source_path, source, source_sha)
    if source_link != rebuilt: raise ResultError("path oracle source calibration link differs")
    return {"case": oracle_case, "result_sha256": path_result_sha, "calibration_root_path": rebuilt["manifests"]["candidate_path"], "calibration_manifest_sha256": rebuilt["manifests"]["candidate_sha256"], "source_manifest_path": rebuilt["manifests"]["source_path"], "source_manifest_sha256": rebuilt["manifests"]["source_sha256"]}


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
    contract_fields = ("fixture_id", "scope", "phase", "mode", "prompt_tokens", "cached_prefix_tokens", "context_tokens", "decode_start_tokens", "generated_tokens", "prefill_requested_m", "resolved_m", "request_count", "decode_request_count", "sampling", "control", "control_id", "format_id", "implementation_id", "device")
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
    if not matching or (case.get("phase") != "decode" and len(matching) != 1): raise ResultError("trace phase/case association differs")
    phase = matching[0]
    expected_mode = {"all_m1": "cold", "cold_batched": "cold", "cached_prefix_chunked": "cached_prefix"}.get(case.get("mode"))
    if case.get("phase") != "decode" and phase.get("prefill_mode") != expected_mode: raise ResultError("trace prefill mode differs")
    expected_phase = {
        "input_token_count": case.get("prompt_tokens"),
        "cached_prefix_token_count": case.get("cached_prefix_tokens"),
        "context_tokens_before": case.get("cached_prefix_tokens") if case.get("phase") != "decode" else case.get("decode_start_tokens"),
        "context_tokens_after": case.get("context_tokens") if case.get("phase") != "decode" else case.get("decode_start_tokens", 0) + case.get("generated_tokens", 0),
        "actual_token_batch_width": case.get("resolved_m") if case.get("phase") != "decode" else 1,
        "actual_request_batch_width": case.get("request_count"),
        "request_count": case.get("request_count"),
    }
    if case.get("phase") == "decode":
        if sum(item.get("input_token_count", -1) for item in matching) != case.get("generated_tokens") or sum(item.get("output_token_count", -1) for item in matching) != case.get("generated_tokens") or matching[0].get("context_tokens_before") != case.get("decode_start_tokens") or matching[-1].get("context_tokens_after") != case.get("decode_start_tokens", 0) + case.get("generated_tokens", 0) or any(item.get("actual_token_batch_width") != 1 or item.get("actual_request_batch_width") != case.get("request_count") or item.get("request_count") != case.get("request_count") for item in matching):
            raise ResultError("trace decode phase aggregation differs")
    elif any(wanted is None or phase.get(field) != wanted for field, wanted in expected_phase.items()): raise ResultError("trace phase shape/width/context differs")
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
    if trace.get("sampling") != case.get("sampling"): raise ResultError("trace sampling differs")
    if trace.get("control") != case.get("control"): raise ResultError("trace control differs")
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
    root = contained(args.run_root, args.run_root, "run root")
    paths = [(args.case, "case"), (args.expanded, "expanded"), (args.raw, "raw"), (args.identity, "identity"), (args.policy, "policy"), (args.source_oracle, "source oracle"), (args.source_oracle_validation, "source oracle validation"), (args.source_calibration_evidence, "source calibration evidence"), (args.independent_validation, "independent validation")]
    if args.path_oracle_result: paths.append((args.path_oracle_result, "path oracle result"))
    if args.path_calibration_evidence: paths.append((args.path_calibration_evidence, "path calibration evidence"))
    if args.trace: paths.append((args.trace, "trace"))
    elif any((args.trace_manifest, args.trace_executor_record, args.trace_binding, args.trace_report, args.trace_source)):
        raise ResultError("trace validation artifacts cannot be supplied without a trace")
    for path, label in paths: contained(root, path, label)
    contained(root, args.output, "output", existing=False)
    case = load(args.case, "case"); expanded = load(args.expanded, "expanded"); raw = load(args.raw, "raw"); identity = load(args.identity, "identity"); policy = load(args.policy, "policy")
    source = load(args.source_oracle, "source oracle"); source_validation = load(args.source_oracle_validation, "source oracle validation"); independent = load(args.independent_validation, "independent validation")
    validate_complete_expansion(expanded, identity, root)
    if expanded.get("schema_version") != "ullm.aq4_production_p2_expanded.v2" or case.get("case_sha256") != case_hash(case) or len([item for item in expanded.get("cases", []) if item == case]) != 1: raise ResultError("case/expanded binding differs")
    if raw.get("schema_version") != "ullm.aq4_production_p2_raw_result.v2" or raw.get("case_id") != case.get("case_id") or raw.get("case_sha256") != case.get("case_sha256"): raise ResultError("raw case binding differs")
    status = raw.get("status")
    if status not in OK_STATUSES or raw.get("immutable_status") is not (status != "ok"): raise ResultError("raw immutable status differs")
    if raw.get("links", {}).get("expanded", {}).get("sha256") != sha_file(args.expanded, "expanded") or identity.get("expanded_manifest_sha256") != sha_file(args.expanded, "expanded"): raise ResultError("raw/identity expanded binding differs")
    if identity.get("schema_version") != "ullm.aq4_production_p2_identity.v2" or identity.get("status") != "bound" or identity.get("identity_sha256") != identity_hash(identity) or raw.get("links", {}).get("identity", {}).get("sha256") != sha_file(args.identity, "identity"): raise ResultError("identity self-binding differs")
    if policy.get("status") != "bound" or policy.get("hash_binding", {}).get("policy_sha256") != policy_hash(policy) or identity.get("policy_sha256") != policy.get("hash_binding", {}).get("policy_sha256") or raw.get("links", {}).get("policy", {}).get("sha256") != sha_file(args.policy, "policy"): raise ResultError("bound policy differs")
    source_sha = sha_file(args.source_oracle, "source oracle"); validate_source_oracle(source, source_validation, source_sha)
    if identity.get("hash_binding", {}).get("source_oracle_sha256") != source_sha: raise ResultError("source oracle identity differs")
    source_calibration = validate_calibration_evidence(args.source_calibration_evidence, "source_gate", root, case, expanded, identity, policy, args.source_oracle, source, source_sha)
    measurement_path = Path(raw.get("links", {}).get("measurement", {}).get("path", "")); state_path = Path(raw.get("links", {}).get("state", {}).get("path", ""))
    contained(root, measurement_path, "measurement"); contained(root, state_path, "state")
    if sha_file(measurement_path, "measurement") != raw["links"]["measurement"]["sha256"] or sha_file(state_path, "state") != raw["links"]["state"]["sha256"]: raise ResultError("raw evidence hash differs")
    measurement = load(measurement_path, "measurement"); performance = validate_measurements(measurement, case); state = load(state_path, "state"); validate_state(state, case["case_id"])
    path_sha = None; path_link = None; path_calibration = None
    if case.get("mode") in {"cold_batched", "cached_prefix_chunked"}:
        if args.path_oracle_result is None: raise ResultError("optimized case requires a path oracle result")
        if args.path_calibration_evidence is None: raise ResultError("optimized case requires a same-artifact all-M1 path calibration")
        path_result = load(args.path_oracle_result, "path oracle result"); path_sha = sha_file(args.path_oracle_result, "path oracle result")
        if path_result.get("schema_version") != "ullm.prefill_validation.v1" or path_result.get("case_id") != case.get("path_oracle_case_id") or path_result.get("status") != "ok" or path_result.get("workload", {}).get("baseline_mode") != "all_m1": raise ResultError("path oracle result identity/status differs")
        if path_result.get("identity", {}).get("sha256") != sha_file(args.identity, "identity") or path_result.get("oracles", {}).get("source_oracle", {}).get("sha256") != source_sha or path_result.get("oracles", {}).get("threshold_policy", {}).get("self_sha256") != policy.get("hash_binding", {}).get("policy_sha256"): raise ResultError("path oracle artifact/source/policy identity differs")
        for field in ("phase", "cached_prefix_tokens", "prompt_tokens", "prefill_requested_m", "scope", "control_id"):
            left = path_result.get("workload", {}).get(field) if field in {"phase", "cached_prefix_tokens", "prompt_tokens", "prefill_requested_m"} else path_result.get(field)
            if left != case.get(field): raise ResultError(f"path oracle same-state field differs: {field}")
        path_link = {"mode": "all_m1", "result_path": str(args.path_oracle_result.resolve()), "result_sha256": path_sha}
        path_chain = path_oracle_calibration_chain(path_result, path_sha, root, expanded, identity, policy, args.source_oracle, source, source_sha)
        if path_chain["case"]["case_id"] != case.get("path_oracle_case_id"): raise ResultError("path oracle calibration case differs")
        path_calibration = validate_calibration_evidence(args.path_calibration_evidence, "path_gate", root, case, expanded, identity, policy, args.source_oracle, source, source_sha, path_oracle=path_chain)
        if path_calibration["comparison"]["sha256"] == source_calibration["comparison"]["sha256"]: raise ResultError("source/path calibration comparisons must be separate artifacts")
    elif args.path_oracle_result is not None or args.path_calibration_evidence is not None: raise ResultError("all-M1/decode case must not attach path evidence")
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
        "calibration": {"source_gate": source_calibration, "path_gate": path_calibration},
        "correctness": correctness, "performance": performance, "regression": regression,
        "promotion": {"eligible": False, "reason_codes": reasons, "required_next_scope": "independent_complete_matrix_validator"},
        "error": raw.get("failure_reason"), "notes": [],
    }
    return result


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    if os.path.lexists(path): raise ResultError(f"refusing to overwrite {path}")
    _reject_symlink_components(path.parent, "output parent")
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    try:
        with temporary.open("xb") as target: target.write((json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode()); target.flush(); os.fsync(target.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try: os.fsync(directory)
        finally: os.close(directory)
    except FileExistsError as error: raise ResultError(f"refusing to overwrite {path}") from error
    finally:
        try: temporary.unlink()
        except FileNotFoundError: pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run_root", "case", "expanded", "raw", "identity", "policy", "source_oracle", "source_oracle_validation", "source_calibration_evidence", "independent_validation", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--path-oracle-result", type=Path); parser.add_argument("--path-calibration-evidence", type=Path); parser.add_argument("--trace", type=Path)
    parser.add_argument("--trace-manifest", type=Path); parser.add_argument("--trace-executor-record", type=Path); parser.add_argument("--trace-binding", type=Path); parser.add_argument("--trace-report", type=Path); parser.add_argument("--trace-source", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    try:
        result = build(args); atomic_write(args.output, result); print(json.dumps({"status": "ok", "promotion_eligible": False}, sort_keys=True)); return 0
    except (ResultError, OSError, ValueError) as error:
        print(f"P2 validation result failed: {error}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
