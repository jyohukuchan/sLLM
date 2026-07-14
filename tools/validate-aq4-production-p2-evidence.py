#!/usr/bin/env python3
"""Independently validate a complete AQ4 P2 scheduled result matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

HASH_RE = re.compile(r"^[0-9a-f]{64}$")
STATUSES = {"ok", "failed", "oom", "unsupported", "skipped"}
POWER_FIELDS = ("expected_power_limit_watts", "allowed_power_tolerance_watts", "maximum_temperature_c", "minimum_vram_headroom_bytes")
CORRECTNESS_FIELDS = ("max_hidden_relative_l2", "max_hidden_max_abs", "max_logits_relative_l2", "max_logits_max_abs", "minimum_top_k_overlap")
STRICT_TRACE_VALIDATOR = Path(__file__).with_name("validate-production-execution-trace.py")
P2_RESULT_BUILDER = Path(__file__).with_name("build-aq4-prefill-validation-result.py")
FIXED_EXPANDER = Path(__file__).with_name("expand-aq4-production-p2.py")
STANDARD_MANIFEST = Path(__file__).parent.parent / "benchmarks/workloads/aq4-production-opt-p2-case-manifest-v0.1.json"


class EvidenceError(ValueError): pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in items:
        if key in value: raise EvidenceError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode, info.st_nlink


def _open_regular(path: Path, label: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try: descriptor = os.open(path, flags)
    except OSError as error: raise EvidenceError(f"{label} is unavailable: {error}") from error
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(descriptor); raise EvidenceError(f"{label} must be a single-link regular file")
    return descriptor


def _read_stable(path: Path, label: str, maximum: int) -> bytes:
    descriptor = _open_regular(path, label)
    try:
        before = os.fstat(descriptor)
        if before.st_size > maximum: raise EvidenceError(f"{label} exceeds {maximum} bytes")
        chunks: list[bytes] = []; remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk: break
            chunks.append(chunk); remaining -= len(chunk)
        if remaining == 0: raise EvidenceError(f"{label} exceeds {maximum} bytes")
        if _identity(before) != _identity(os.fstat(descriptor)): raise EvidenceError(f"{label} changed while being read")
        return b"".join(chunks)
    finally: os.close(descriptor)


def load(path: Path, label: str) -> dict[str, Any]:
    try: value = json.loads(_read_stable(path, label, 64 * 1024 * 1024).decode("utf-8"), object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(EvidenceError(f"non-finite JSON number: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error: raise EvidenceError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict): raise EvidenceError(f"{label} root must be an object")
    return value


def canonical(value: Any) -> bytes: return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
def sha_bytes(value: bytes) -> str: return hashlib.sha256(value).hexdigest()


def sha_file(path: Path, label: str) -> str:
    descriptor = _open_regular(path, label); digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        while chunk := os.read(descriptor, 1024 * 1024): digest.update(chunk)
        if _identity(before) != _identity(os.fstat(descriptor)): raise EvidenceError(f"{label} changed while being hashed")
        return digest.hexdigest()
    finally: os.close(descriptor)


def _reject_symlink_components(path: Path, label: str, *, allow_missing_leaf: bool = False) -> None:
    absolute = Path(os.path.abspath(path)); components = [Path(absolute.anchor)]
    components.extend(Path(absolute.anchor, *absolute.parts[1:index]) for index in range(1, len(absolute.parts) + 1))
    for index, component in enumerate(components):
        try: info = component.lstat()
        except FileNotFoundError:
            if allow_missing_leaf and index == len(components) - 1: return
            raise EvidenceError(f"{label} path component is missing: {component}")
        if stat.S_ISLNK(info.st_mode): raise EvidenceError(f"{label} path component is a symlink: {component}")


def contained(root: Path, path: Path, label: str, *, existing: bool = True) -> Path:
    lexical_root = Path(os.path.abspath(root)); lexical = Path(os.path.abspath(path))
    if lexical != lexical_root and lexical_root not in lexical.parents: raise EvidenceError(f"{label} escapes run root")
    _reject_symlink_components(lexical_root, "run root")
    _reject_symlink_components(lexical, label, allow_missing_leaf=not existing)
    resolved_root = lexical_root.resolve(strict=True); resolved = lexical.resolve(strict=existing)
    if resolved != resolved_root and resolved_root not in resolved.parents: raise EvidenceError(f"{label} escapes run root")
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


def complete_expansion_failure(expanded: dict[str, Any], identity: dict[str, Any], root: Path) -> str | None:
    try:
        manifest_path = Path(identity.get("artifacts", {}).get("planning_manifest", "")).resolve(strict=True)
        if manifest_path != STANDARD_MANIFEST.resolve() and manifest_path != root.resolve() and root.resolve() not in manifest_path.parents: return "planning_manifest_path"
        manifest = load(manifest_path, "planning manifest")
        if sha_file(manifest_path, "planning manifest") != identity.get("planning_manifest_sha256"): return "planning_manifest_identity"
        spec = importlib.util.spec_from_file_location("aq4_p2_validator_expander", FIXED_EXPANDER)
        if spec is None or spec.loader is None: return "fixed_expander"
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        expected = module.expand(json.loads(json.dumps(manifest)), sha_file(manifest_path, "planning manifest"))
        return None if expanded == expected else "expanded_not_complete_planning_set"
    except Exception: return "complete_expansion_unavailable"


def result_builder_module() -> Any:
    spec = importlib.util.spec_from_file_location("aq4_p2_calibration_binding", P2_RESULT_BUILDER)
    if spec is None or spec.loader is None: raise EvidenceError("P2 result builder is unavailable")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


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


def strict_trace_failures(root: Path, trace_path: Path, trace: dict[str, Any], case: dict[str, Any], validation: Any, prefix: str) -> list[str]:
    failure = f"trace_strict_validation:{prefix}"
    if not isinstance(validation, dict): return [failure]
    required = ("manifest", "executor_record", "binding", "detached_report")
    resolved: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    try:
        for role in required:
            link = validation.get(role)
            if not isinstance(link, dict): return [failure]
            path = contained(root, Path(link.get("path", "")), f"trace {role}")
            if sha_file(path, f"trace {role}") != link.get("sha256"): return [failure]
            resolved[role] = (path, load(path, f"trace {role}"), link)
        sources = validation.get("source_traces")
        if not isinstance(sources, list): return [failure]
        source_paths: list[Path] = []
        for link in sources:
            if not isinstance(link, dict): return [failure]
            path = contained(root, Path(link.get("path", "")), "trace source")
            if sha_file(path, "trace source") != link.get("sha256"): return [failure]
            source_paths.append(path)
        command = [
            sys.executable, str(STRICT_TRACE_VALIDATOR), "--trace", str(trace_path),
            "--manifest", str(resolved["manifest"][0]), "--executor-record", str(resolved["executor_record"][0]),
            "--binding", str(resolved["binding"][0]), "--report", str(resolved["detached_report"][0]),
        ]
        for source in source_paths: command.extend(("--source-trace", str(source)))
        completed = subprocess.run(command, cwd=STRICT_TRACE_VALIDATOR.parent.parent, capture_output=True, text=True, timeout=60, check=False)
        if completed.returncode != 0: return [failure]
        strict_report = json.loads(completed.stdout, object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(EvidenceError(f"non-finite strict report number: {item}")))
        detached = resolved["detached_report"][1]
        expected_promotion = case.get("scope") == "production_server" and trace.get("status") == "ok"
        if validation.get("strict_validation") != strict_report or resolved["detached_report"][2].get("report") != detached: return [failure]
        if strict_report.get("schema_version") != "ullm.production_execution_trace_validator.v1" or strict_report.get("status") != "valid" or strict_report.get("trace_sha256") != sha_file(trace_path, "trace") or strict_report.get("executor_record_sha256") != sha_file(resolved["executor_record"][0], "trace executor record") or strict_report.get("scope") != case.get("scope") or strict_report.get("promotion_eligible") is not expected_promotion: return [failure]
        if detached.get("schema_version") != "ullm.production_execution_trace_validator.v1" or detached.get("status") != "valid" or detached.get("scope") != case.get("scope") or detached.get("executor_record_sha256") != sha_file(resolved["executor_record"][0], "trace executor record") or detached.get("promotion_eligible") is not expected_promotion: return [failure]
        if trace.get("verification", {}).get("independent_validation", {}).get("report_sha256") != sha_file(resolved["detached_report"][0], "detached trace report"): return [failure]
        return []
    except (EvidenceError, OSError, ValueError, subprocess.TimeoutExpired):
        return [failure]


def trace_association_failures(root: Path, trace_path: Path, trace: dict[str, Any], case: dict[str, Any], raw: dict[str, Any], identity: dict[str, Any], prefix: str) -> list[str]:
    failure = f"trace_case_association:{prefix}"
    try:
        measurement_link = raw.get("links", {}).get("measurement", {})
        measurement_path = contained(root, Path(measurement_link.get("path", "")), "trace measurement")
        if sha_file(measurement_path, "trace measurement") != measurement_link.get("sha256"): return [failure]
        measurement = load(measurement_path, "trace measurement")
        spec = importlib.util.spec_from_file_location("aq4_p2_result_builder_association", P2_RESULT_BUILDER)
        if spec is None or spec.loader is None: return [failure]
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.validate_trace_association(trace, case, raw, identity, measurement, trace_path, sha_file(trace_path, "trace"))
        return []
    except Exception:
        return [failure]


def validate(args: argparse.Namespace) -> dict[str, Any]:
    root = contained(args.run_root, args.run_root, "run root")
    for path, label in ((args.expanded, "expanded"), (args.identity, "identity"), (args.policy, "policy"), (args.source_oracle, "source oracle")): contained(root, path, label)
    expanded = load(args.expanded, "expanded"); identity = load(args.identity, "identity"); policy = load(args.policy, "policy"); source = load(args.source_oracle, "source oracle")
    failures: list[str] = []
    complete_failure = complete_expansion_failure(expanded, identity, root)
    if complete_failure: failures.append(complete_failure)
    if identity.get("evidence_class") != "production_candidate" or identity.get("promotion_eligible") is not False: failures.append("non_production_identity_class")
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
    try: calibration_module = result_builder_module()
    except EvidenceError:
        calibration_module = None; failures.append("calibration_validator_unavailable")
    used_calibration_comparisons: dict[str, dict[str, str]] = {"source_gate": {}, "path_gate": {}}
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
        try:
            result_identity_path = contained(root, Path(result.get("identity", {}).get("path", "")), "result identity")
            if result_identity_path != args.identity.resolve() or result.get("identity", {}).get("sha256") != sha_file(args.identity, "identity") or result.get("identity", {}).get("binding_sha256") != identity.get("identity_sha256"): failures.append(f"result_identity_link:{prefix}")
        except (EvidenceError, OSError): failures.append(f"result_identity_link:{prefix}")
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
        calibration = result.get("calibration")
        source_calibration = None; path_calibration = None
        if not isinstance(calibration, dict) or set(calibration) != {"source_gate", "path_gate"}:
            failures.append(f"calibration_fields:{prefix}")
        elif calibration_module is not None:
            try:
                source_link = calibration["source_gate"]
                if not isinstance(source_link, dict) or set(source_link) != {"path", "sha256", "comparison", "metrics"}: raise EvidenceError("source calibration link fields differ")
                source_calibration = calibration_module.validate_calibration_evidence(Path(source_link["path"]), "source_gate", root, case, expanded, identity, policy, source_sha)
                if source_link != source_calibration: raise EvidenceError("source calibration result link differs")
                source_comparison_sha = source_calibration["comparison"]["sha256"]
                prior = used_calibration_comparisons["source_gate"].get(source_comparison_sha)
                if prior is not None and prior != case_id: failures.append(f"calibration_source_reuse:{prefix}:{prior}")
                used_calibration_comparisons["source_gate"][source_comparison_sha] = case_id
            except Exception:
                failures.append(f"calibration_source_gate:{prefix}")
            optimized = case.get("mode") in {"cold_batched", "cached_prefix_chunked"}
            if optimized:
                try:
                    path_link_value = calibration["path_gate"]
                    if not isinstance(path_link_value, dict) or set(path_link_value) != {"path", "sha256", "comparison", "metrics"}: raise EvidenceError("path calibration link fields differ")
                    path_calibration = calibration_module.validate_calibration_evidence(Path(path_link_value["path"]), "path_gate", root, case, expanded, identity, policy, source_sha)
                    if path_link_value != path_calibration: raise EvidenceError("path calibration result link differs")
                    path_comparison_sha = path_calibration["comparison"]["sha256"]
                    if source_calibration is not None and source_calibration["comparison"]["sha256"] == path_comparison_sha: raise EvidenceError("source/path calibration comparison is reused")
                    prior = used_calibration_comparisons["path_gate"].get(path_comparison_sha)
                    if prior is not None and prior != case_id: failures.append(f"calibration_path_reuse:{prefix}:{prior}")
                    used_calibration_comparisons["path_gate"][path_comparison_sha] = case_id
                except Exception:
                    failures.append(f"calibration_path_gate:{prefix}")
            elif calibration["path_gate"] is not None:
                failures.append(f"calibration_unexpected_path_gate:{prefix}")
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
                if oracle_result.get("identity", {}).get("sha256") != sha_file(args.identity, "identity") or oracle_result.get("oracles", {}).get("source_oracle", {}).get("sha256") != source_sha or oracle_result.get("oracles", {}).get("threshold_policy", {}).get("self_sha256") != policy.get("hash_binding", {}).get("policy_sha256"): failures.append(f"path_oracle_artifact_identity:{prefix}")
        elif path_link is not None: failures.append(f"unexpected_path_oracle:{prefix}")
        trace_link = result.get("evidence", {}).get("execution_trace"); expected_trace_sha = None
        if trace_link is not None:
            try:
                trace_path = contained(root, Path(trace_link.get("path", "")), "trace"); trace = load(trace_path, "trace")
                expected_trace_sha = sha_file(trace_path, "trace")
                expected_raw_trace = {"path": str(trace_path), "sha256": expected_trace_sha, "trace_id": trace.get("trace_id")}
                if expected_trace_sha != trace_link.get("sha256") or trace_link.get("trace_id") != trace.get("trace_id") or raw.get("links", {}).get("trace") != expected_raw_trace: failures.append(f"trace_hash:{prefix}")
                failures.extend(trace_failures(trace, case, prefix))
                failures.extend(trace_association_failures(root, trace_path, trace, case, raw, identity, prefix))
                failures.extend(strict_trace_failures(root, trace_path, trace, case, trace_link.get("validation"), prefix))
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
            new_oom = result.get("status") == "oom" and baseline.get("oom") is not True
            prefill_policy = policy.get("performance_thresholds", {}).get("prefill", {})
            expected_passed = result.get("status") == "ok" and not new_oom and expected_p50 >= -100 * prefill_policy.get("p50_regression_stop_fraction", 0) and expected_p95 >= -100 * prefill_policy.get("p95_regression_stop_fraction", 0)
            if baseline_sha != identity.get("hash_binding", {}).get("baseline_result_sha256") or regression.get("baseline_result_sha256") != baseline_sha or not math.isclose(regression.get("prefill_p50_change_percent"), expected_p50, rel_tol=1e-12) or not math.isclose(regression.get("prefill_p95_change_percent"), expected_p95, rel_tol=1e-12) or regression.get("new_oom") is not new_oom or regression.get("passed") is not expected_passed: failures.append(f"regression_recompute:{prefix}")
        except (EvidenceError, OSError, KeyError, ZeroDivisionError, TypeError): failures.append(f"regression_link:{prefix}")
    promotion = False
    return {"schema_version": "ullm.aq4_production_p2_evidence_validator.v2", "status": "valid" if not failures else "invalid", "failure_codes": sorted(set(failures)), "scheduled_case_count": len(case_by_id), "result_count": len(results), "complete_matrix": not missing and len(results) == len(case_by_id), "promotion_eligible": promotion, "production_live_execution": False, "review_note": "CPU/synthetic validation never promotes; GPU/live execution was not performed"}


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    if os.path.lexists(path): raise EvidenceError(f"refusing to overwrite {path}")
    _reject_symlink_components(path.parent, "output parent")
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    try:
        with temporary.open("xb") as target: target.write((json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode()); target.flush(); os.fsync(target.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try: os.fsync(directory)
        finally: os.close(directory)
    except FileExistsError as error: raise EvidenceError(f"refusing to overwrite {path}") from error
    finally:
        try: temporary.unlink()
        except FileNotFoundError: pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-root", type=Path, required=True); parser.add_argument("--expanded", type=Path, required=True); parser.add_argument("--identity", type=Path, required=True); parser.add_argument("--policy", type=Path, required=True); parser.add_argument("--source-oracle", type=Path, required=True); parser.add_argument("--result", type=Path, action="append", default=[]); parser.add_argument("--result-dir", type=Path); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.result and not args.result_dir: parser.error("at least one --result or --result-dir is required")
    try:
        root = contained(args.run_root, args.run_root, "run root"); output = contained(root, args.output, "output", existing=False)
        report = validate(args); atomic_write(args.output, report); print(json.dumps({"status": report["status"], "failure_count": len(report["failure_codes"])}, sort_keys=True)); return 0 if report["status"] == "valid" else 1
    except (EvidenceError, OSError, ValueError) as error:
        print(f"P2 evidence validation failed closed: {error}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
