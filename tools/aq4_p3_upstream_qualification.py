#!/usr/bin/env python3
"""Build and validate the typed upstream P2 qualification for AQ4 P3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any


SCHEMA = "ullm.aq4_p3_upstream_p2_qualification.v1"
P2_REJECTION_SCHEMA = "ullm.qwen35_aq4_sq8_fidelity_calibration_rejection_evidence.v1"
P2_RECEIPT_NAME = "calibration-no-go-evidence.json"
P2_SUMS_NAME = "SHA256SUMS"
REASON = "upstream_p2_calibration_rejected_no_go"
GO_REASON = "upstream_p2_holdout_passed"
P2_PLAN_SCHEMA = "ullm.qwen35_aq4_sq8_fidelity_plan.v1"
P2_METRICS_SCHEMA = "ullm.qwen35_aq4_sq8_fidelity_metrics.v1"
P2_FREEZE_SCHEMA = "ullm.qwen35_aq4_sq8_fidelity_freeze_receipt.v1"
P2_PREFLIGHT_SCHEMA = "ullm.qwen35_aq4_sq8_fidelity_holdout_preflight.v1"
P2_LEDGER_SCHEMA = "ullm.qwen35_aq4_sq8_fidelity_attempt_ledger.v1"
P2_HOLDOUT_SCHEMA = "ullm.qwen35_aq4_sq8_fidelity_holdout_receipt.v1"
P2_ATTEMPT_SCHEMA = "ullm.qwen35_aq4_sq8_fidelity_holdout_attempt.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_BYTES = 16 * 1024 * 1024


class QualificationError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path, label: str) -> str:
    digest = hashlib.sha256()
    size = 0
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size <= 0 or before.st_size > MAX_BYTES:
            raise QualificationError(f"{label} file identity differs")
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            if size > MAX_BYTES:
                raise QualificationError(f"{label} exceeds the bounded size")
            digest.update(chunk)
        after = os.fstat(descriptor)
        current = path.lstat()
        identity = lambda info: (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if identity(before) != identity(after) or identity(before) != identity(current):
            raise QualificationError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        missing = sorted(fields - set(value)) if isinstance(value, dict) else sorted(fields)
        unknown = sorted(set(value) - fields) if isinstance(value, dict) else []
        raise QualificationError(f"{label} fields differ: missing={missing}, unknown={unknown}")
    return value


def digest(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise QualificationError(f"{label} must be a lowercase SHA-256")
    return value


def finite(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < minimum:
        raise QualificationError(f"{label} must be a finite number >= {minimum}")
    return float(value)


def integer(value: Any, expected: int, label: str) -> int:
    if type(value) is not int or value != expected:
        raise QualificationError(f"{label} must be the integer {expected}")
    return value


def parse(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_absolute() or path != path.resolve() or path.is_symlink() or not path.is_file():
        raise QualificationError(f"{label} path must be canonical and regular")
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_BYTES:
        raise QualificationError(f"{label} size differs")
    sha_file(path, label)
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise QualificationError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda token: (_ for _ in ()).throw(QualificationError(f"non-finite JSON in {label}: {token}")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise QualificationError(f"{label} must be an object")
    return value, raw


def file_ref(path: Path, label: str) -> dict[str, str]:
    resolved = path.resolve()
    if path != resolved:
        raise QualificationError(f"{label} path must be canonical")
    return {"path": str(resolved), "sha256": sha_file(resolved, label)}


def validate_ref(value: Any, label: str) -> Path:
    ref = exact(value, {"path", "sha256"}, f"{label} reference")
    expected = digest(ref["sha256"], f"{label} SHA-256")
    path = Path(ref["path"])
    if not path.is_absolute() or path != path.resolve() or path.is_symlink() or not path.is_file() or sha_file(path, label) != expected:
        raise QualificationError(f"{label} reference differs")
    return path


def self_hash(value: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))
    clone["qualification_sha256"] = None
    return sha_bytes(canonical(clone))


def validate_p2_rejection_package(root: Path) -> dict[str, Any]:
    if not root.is_absolute() or root != root.resolve() or root.is_symlink() or not root.is_dir() or stat.S_IMODE(root.stat().st_mode) != 0o555:
        raise QualificationError("P2 rejection package directory identity differs")
    children = list(root.iterdir())
    if {item.name for item in children} != {P2_RECEIPT_NAME, P2_SUMS_NAME} or any(item.is_symlink() or not item.is_file() or item.stat().st_nlink != 1 or stat.S_IMODE(item.stat().st_mode) != 0o444 for item in children):
        raise QualificationError("P2 rejection package inventory differs")
    receipt_path = root / P2_RECEIPT_NAME
    receipt_sha = sha_file(receipt_path, "P2 calibration rejection evidence")
    sums_sha = sha_file(root / P2_SUMS_NAME, "P2 rejection SHA256SUMS")
    if (root / P2_SUMS_NAME).read_text(encoding="ascii") != f"{receipt_sha}  {P2_RECEIPT_NAME}\n":
        raise QualificationError("P2 rejection SHA256SUMS differs")
    receipt, _ = parse(receipt_path, "P2 calibration rejection evidence")
    exact(receipt, {"schema_version", "status", "reason", "policy", "observed", "state", "bindings", "capture", "lineage"}, "P2 calibration rejection evidence")
    if receipt["schema_version"] != P2_REJECTION_SCHEMA or receipt["status"] != "calibration_rejected_no_go" or receipt["reason"] != "relative_l2_pathological_rejection_before_aggregation":
        raise QualificationError("P2 calibration rejection status differs")
    state = exact(receipt["state"], {"metrics_published", "freeze_published", "holdout_status", "holdout_evaluations_remaining", "retry_permitted", "holdout_executed"}, "P2 terminal state")
    expected_state = {"metrics_published": False, "freeze_published": False, "holdout_status": "not_started", "holdout_evaluations_remaining": 1, "retry_permitted": False, "holdout_executed": False}
    if state != expected_state:
        raise QualificationError("P2 terminal state differs")
    policy = exact(receipt["policy"], {"path", "sha256", "ceiling", "action"}, "P2 policy")
    policy_path = Path(policy["path"])
    if policy["ceiling"] != 1.0 or policy["action"] != "reject any observed relative-L2 > 1 before aggregation" or sha_file(policy_path, "P2 policy") != digest(policy["sha256"], "P2 policy SHA-256"):
        raise QualificationError("P2 rejection policy differs")
    observed = exact(receipt["observed"], {"row_count", "nonfinite_rows", "logits_relative_l2", "hidden_relative_l2", "greedy_mismatch_rows", "minimum_top_k_overlap"}, "P2 observations")
    if type(observed["row_count"]) is not int or observed["row_count"] != 24 or type(observed["nonfinite_rows"]) is not int or observed["nonfinite_rows"] != 0 or type(observed["greedy_mismatch_rows"]) is not int or not 0 <= observed["greedy_mismatch_rows"] <= 24 or type(observed["minimum_top_k_overlap"]) is not int or not 0 <= observed["minimum_top_k_overlap"] <= 10:
        raise QualificationError("P2 observation counts differ")
    for name in ("logits_relative_l2", "hidden_relative_l2"):
        item = exact(observed[name], {"count_above_ceiling", "minimum", "mean", "maximum"}, f"P2 {name}")
        if type(item["count_above_ceiling"]) is not int or not 0 <= item["count_above_ceiling"] <= 24:
            raise QualificationError(f"P2 {name} rejection count differs")
        minimum, mean, maximum = (finite(item[field], f"P2 {name}.{field}") for field in ("minimum", "mean", "maximum"))
        if not minimum <= mean <= maximum:
            raise QualificationError(f"P2 {name} ordering differs")
    if observed["logits_relative_l2"]["count_above_ceiling"] + observed["hidden_relative_l2"]["count_above_ceiling"] <= 0:
        raise QualificationError("P2 observations do not trigger rejection")
    bindings = exact(receipt["bindings"], {"plan", "actual_receipt", "source_artifact", "target_artifact", "comparison"}, "P2 bindings")
    plan_path = validate_ref(bindings["plan"], "P2 plan")
    actual_path = validate_ref(bindings["actual_receipt"], "P2 actual receipt")
    for name in ("source_artifact", "target_artifact", "comparison"):
        binding = exact(bindings[name], {"root", "manifest_sha256", "sha256sums_sha256"}, f"P2 {name}")
        artifact_root = Path(binding["root"])
        if not artifact_root.is_absolute() or artifact_root != artifact_root.resolve() or not artifact_root.is_dir() or artifact_root.is_symlink() or sha_file(artifact_root / "manifest.json", f"P2 {name} manifest") != digest(binding["manifest_sha256"], f"P2 {name} manifest SHA-256") or sha_file(artifact_root / P2_SUMS_NAME, f"P2 {name} SHA256SUMS") != digest(binding["sha256sums_sha256"], f"P2 {name} SHA256SUMS SHA-256"):
            raise QualificationError(f"P2 {name} binding differs")
    return {
        "package": {"root": str(root), "receipt_sha256": receipt_sha, "sha256sums_sha256": sums_sha},
        "plan": file_ref(plan_path, "P2 plan"),
        "actual_receipt": file_ref(actual_path, "P2 actual receipt"),
        "policy": file_ref(policy_path, "P2 policy"),
        "observed": observed,
        "holdout": {"status": "not_started", "evaluations_remaining": 1, "executed": False},
    }


def build_rejection(root: Path) -> dict[str, Any]:
    evidence = validate_p2_rejection_package(root)
    value = {
        "schema_version": SCHEMA,
        "status": "rejected_no_go",
        "qualification_sha256": None,
        "promotion_eligible": False,
        "reason": REASON,
        "p2": evidence,
    }
    value["qualification_sha256"] = self_hash(value)
    return value


IDENTITY_FIELDS = {
    "sq8_receipt_path", "sq8_receipt_sha256", "receipt_state", "request_id", "source_commit",
    "source_tree_sha256", "source_archive_sha256", "source_v32_path", "source_v32_sha256",
    "served_model", "worker", "package", "overlay_content_sha256", "overlay_tensor_set_sha256",
    "token_ids_sha256", "telemetry_binding", "maintenance_evidence", "executor_record",
    "prepared_receipt", "split_manifest_sha256", "calibration_cases_path",
    "calibration_cases_sha256", "holdout_cases_path", "holdout_cases_sha256", "policy_path",
    "policy_sha256",
}


def strict_equal(left: Any, right: Any) -> bool:
    """JSON equality which does not allow bool to impersonate int."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(strict_equal(a, b) for a, b in zip(left, right))
    return left == right


def checked_document(path: Path, label: str, fields: set[str]) -> tuple[dict[str, Any], bytes]:
    value, raw = parse(path, label)
    exact(value, fields, label)
    return value, raw


def checked_identity(value: Any, label: str) -> dict[str, Any]:
    identity = exact(value, IDENTITY_FIELDS, label)
    if identity["receipt_state"] != "actual_verified" or type(identity["request_id"]) is not str or not identity["request_id"]:
        raise QualificationError(f"{label} is not actual_verified")
    for field in (
        "sq8_receipt_sha256", "source_tree_sha256", "source_archive_sha256", "source_v32_sha256",
        "overlay_content_sha256", "overlay_tensor_set_sha256", "split_manifest_sha256",
        "calibration_cases_sha256", "holdout_cases_sha256", "policy_sha256",
    ):
        digest(identity[field], f"{label}.{field}")
    for field, sha_field in (
        ("sq8_receipt_path", "sq8_receipt_sha256"), ("source_v32_path", "source_v32_sha256"),
        ("calibration_cases_path", "calibration_cases_sha256"),
        ("holdout_cases_path", "holdout_cases_sha256"), ("policy_path", "policy_sha256"),
    ):
        path = Path(identity[field])
        if not path.is_absolute() or path != path.resolve() or sha_file(path, f"P2 identity {field}") != identity[sha_field]:
            raise QualificationError(f"P2 identity {field} binding differs")
    return identity


def validate_p2_success(
    plan_path: Path,
    calibration_metrics_path: Path,
    freeze_path: Path,
    preflight_path: Path,
    ledger_path: Path,
    holdout_metrics_path: Path,
    holdout_path: Path,
) -> dict[str, Any]:
    """Validate the official P2 success-chain schemas and all inter-document bindings."""
    plan_fields = {"schema_version", "status", "preflight_only", "actual_verified_required", "identity", "policy", "calibration", "holdout", "resource_contract", "holdout_state"}
    plan, plan_raw = checked_document(plan_path, "P2 fidelity plan", plan_fields)
    if plan["schema_version"] != P2_PLAN_SCHEMA or plan["status"] != "ready_for_calibration" or plan["preflight_only"] is not False or plan["actual_verified_required"] is not True:
        raise QualificationError("P2 fidelity plan is not an actual-verified calibration plan")
    identity = checked_identity(plan["identity"], "P2 plan identity")
    holdout_state = exact(plan["holdout_state"], {"status", "evaluations_remaining", "retry_permitted"}, "P2 plan holdout state")
    if holdout_state["status"] != "not_started" or holdout_state["retry_permitted"] is not False:
        raise QualificationError("P2 plan holdout state differs")
    integer(holdout_state["evaluations_remaining"], 1, "P2 plan evaluations remaining")
    for name, identity_path, identity_sha in (
        ("calibration", "calibration_cases_path", "calibration_cases_sha256"),
        ("holdout", "holdout_cases_path", "holdout_cases_sha256"),
    ):
        subset = exact(plan[name], {"path", "sha256", "row_count"}, f"P2 plan {name}")
        integer(subset["row_count"], 24, f"P2 plan {name} row count")
        if subset["path"] != identity[identity_path] or subset["sha256"] != identity[identity_sha]:
            raise QualificationError(f"P2 plan {name} binding differs")
    resource = exact(plan["resource_contract"], {"jobs", "case_concurrency", "one_model_load", "chunk_elements", "bounded_vectors", "bounded_disk", "max_rows", "max_case_file_bytes", "vram_headroom_required", "vram_headroom_bytes_min", "vram_observed_headroom_bytes"}, "P2 resource contract")
    integer(resource["jobs"], 1, "P2 jobs")
    integer(resource["case_concurrency"], 1, "P2 case concurrency")
    integer(resource["chunk_elements"], 65_536, "P2 chunk elements")
    integer(resource["max_rows"], 24, "P2 max rows")
    integer(resource["max_case_file_bytes"], 64 * 1024 * 1024, "P2 max case file bytes")
    if any(resource[field] is not True for field in ("one_model_load", "bounded_vectors", "bounded_disk", "vram_headroom_required")):
        raise QualificationError("P2 resource safety contract differs")
    if type(resource["vram_headroom_bytes_min"]) is not int or type(resource["vram_observed_headroom_bytes"]) is not int or resource["vram_headroom_bytes_min"] < 1 or resource["vram_observed_headroom_bytes"] < resource["vram_headroom_bytes_min"]:
        raise QualificationError("P2 VRAM headroom contract differs")

    metric_fields = {"schema_version", "identity", "subset", "evidence", "rows"}
    calibration_metrics, calibration_raw = checked_document(calibration_metrics_path, "P2 calibration metrics", metric_fields)
    holdout_metrics, holdout_metrics_raw = checked_document(holdout_metrics_path, "P2 holdout metrics", metric_fields)
    for value, subset in ((calibration_metrics, "calibration"), (holdout_metrics, "holdout")):
        if value["schema_version"] != P2_METRICS_SCHEMA or value["subset"] != subset or not strict_equal(value["identity"], identity) or not isinstance(value["rows"], list) or len(value["rows"]) != 24:
            raise QualificationError(f"P2 {subset} metrics schema/identity differs")

    freeze_fields = {"schema_version", "status", "identity", "plan_path", "plan_sha256", "metrics_path", "metrics_sha256", "calibration_case_count", "derived_bounds", "holdout_status", "holdout_evaluations_remaining", "retry_permitted", "relative_l2_rejection_ceiling", "attempt_boundary"}
    freeze, freeze_raw = checked_document(freeze_path, "P2 freeze receipt", freeze_fields)
    boundary = exact(freeze["attempt_boundary"], {"remaining_before", "remaining_after", "failure_consumes_attempt"}, "P2 freeze attempt boundary")
    if freeze["schema_version"] != P2_FREEZE_SCHEMA or freeze["status"] != "frozen_calibration_envelope" or not strict_equal(freeze["identity"], identity) or freeze["plan_path"] != str(plan_path) or freeze["plan_sha256"] != sha_bytes(plan_raw) or freeze["metrics_path"] != str(calibration_metrics_path) or freeze["metrics_sha256"] != sha_bytes(calibration_raw) or freeze["holdout_status"] != "not_started" or freeze["retry_permitted"] is not False or freeze["relative_l2_rejection_ceiling"] != 1.0 or boundary != {"remaining_before": 1, "remaining_after": 0, "failure_consumes_attempt": True}:
        raise QualificationError("P2 freeze receipt binding/state differs")
    integer(freeze["calibration_case_count"], 24, "P2 freeze calibration count")
    integer(freeze["holdout_evaluations_remaining"], 1, "P2 freeze evaluations remaining")
    if not isinstance(freeze["derived_bounds"], dict) or not freeze["derived_bounds"]:
        raise QualificationError("P2 freeze derived bounds are absent")

    preflight_fields = {"schema_version", "status", "freeze_receipt_sha256", "freeze_receipt_path", "plan_path", "plan_sha256", "identity", "holdout_cases_sha256", "holdout_case_count", "evaluations_remaining", "retry_permitted", "attempt_boundary"}
    preflight, preflight_raw = checked_document(preflight_path, "P2 holdout preflight", preflight_fields)
    preflight_boundary = exact(preflight["attempt_boundary"], {"remaining_before", "remaining_after", "failure_consumes_attempt"}, "P2 preflight attempt boundary")
    if preflight["schema_version"] != P2_PREFLIGHT_SCHEMA or preflight["status"] != "ready_for_one_shot_holdout" or preflight["freeze_receipt_path"] != str(freeze_path) or preflight["freeze_receipt_sha256"] != sha_bytes(freeze_raw) or preflight["plan_path"] != str(plan_path) or preflight["plan_sha256"] != sha_bytes(plan_raw) or not strict_equal(preflight["identity"], identity) or preflight["holdout_cases_sha256"] != identity["holdout_cases_sha256"] or preflight["retry_permitted"] is not False or preflight_boundary != {"remaining_before": 1, "remaining_after": 0, "failure_consumes_attempt": True}:
        raise QualificationError("P2 holdout preflight binding/state differs")
    integer(preflight["holdout_case_count"], 24, "P2 preflight holdout count")
    integer(preflight["evaluations_remaining"], 1, "P2 preflight evaluations remaining")

    ledger_fields = {"schema_version", "status", "attempt_id", "preflight_sha256", "identity", "remaining_before", "remaining_after", "retry_permitted"}
    ledger, ledger_raw = checked_document(ledger_path, "P2 holdout ledger", ledger_fields)
    expected_attempt = sha_bytes(b"ullm.qwen35-aq4-sq8-holdout-attempt-v1\0" + sha_bytes(preflight_raw).encode())
    if ledger["schema_version"] != P2_LEDGER_SCHEMA or ledger["status"] != "consumed" or ledger["attempt_id"] != expected_attempt or ledger["preflight_sha256"] != sha_bytes(preflight_raw) or not strict_equal(ledger["identity"], identity) or ledger["retry_permitted"] is not False:
        raise QualificationError("P2 holdout ledger binding/state differs")
    integer(ledger["remaining_before"], 1, "P2 ledger remaining before")
    integer(ledger["remaining_after"], 0, "P2 ledger remaining after")

    holdout_fields = {"schema_version", "attempt_schema", "status", "attempt_id", "preflight_sha256", "ledger_sha256", "metrics_sha256", "identity", "derived_metrics", "gate_checks", "retry_permitted", "evaluations_remaining"}
    holdout, holdout_raw = checked_document(holdout_path, "P2 holdout receipt", holdout_fields)
    checks = holdout["gate_checks"]
    if holdout["schema_version"] != P2_HOLDOUT_SCHEMA or holdout["attempt_schema"] != P2_ATTEMPT_SCHEMA or holdout["status"] != "passed" or holdout["attempt_id"] != expected_attempt or holdout["preflight_sha256"] != sha_bytes(preflight_raw) or holdout["ledger_sha256"] != sha_bytes(ledger_raw) or holdout["metrics_sha256"] != sha_bytes(holdout_metrics_raw) or not strict_equal(holdout["identity"], identity) or holdout["retry_permitted"] is not False or not isinstance(checks, dict) or set(checks) != set(freeze["derived_bounds"]) or not checks or any(value is not True for value in checks.values()) or not isinstance(holdout["derived_metrics"], dict) or set(holdout["derived_metrics"]) != set(checks):
        raise QualificationError("P2 holdout success receipt binding/state differs")
    integer(holdout["evaluations_remaining"], 0, "P2 holdout evaluations remaining")
    return {
        "plan": file_ref(plan_path, "P2 plan"),
        "actual_receipt": file_ref(Path(identity["sq8_receipt_path"]), "P2 actual receipt"),
        "policy": file_ref(Path(identity["policy_path"]), "P2 policy"),
        "calibration_metrics": file_ref(calibration_metrics_path, "P2 calibration metrics"),
        "freeze_receipt": file_ref(freeze_path, "P2 freeze receipt"),
        "preflight": file_ref(preflight_path, "P2 preflight"),
        "ledger": file_ref(ledger_path, "P2 ledger"),
        "holdout_metrics": file_ref(holdout_metrics_path, "P2 holdout metrics"),
        "holdout_receipt": file_ref(holdout_path, "P2 holdout receipt"),
        "holdout": {"status": "passed", "evaluations_remaining": 0, "executed": True},
    }


def build_qualified(paths: list[Path]) -> dict[str, Any]:
    evidence = validate_p2_success(*paths)
    value = {"schema_version": SCHEMA, "status": "qualified_go", "qualification_sha256": None, "promotion_eligible": True, "reason": GO_REASON, "p2": evidence}
    value["qualification_sha256"] = self_hash(value)
    return value


def validate(value: dict[str, Any]) -> dict[str, Any]:
    exact(value, {"schema_version", "status", "qualification_sha256", "promotion_eligible", "reason", "p2"}, "P3 upstream qualification")
    if value["schema_version"] != SCHEMA or digest(value["qualification_sha256"], "qualification SHA-256") != self_hash(value):
        raise QualificationError("P3 upstream qualification status/hash differs")
    if value["status"] == "rejected_no_go":
        if value["promotion_eligible"] is not False or value["reason"] != REASON:
            raise QualificationError("P3 rejected qualification state differs")
        p2 = exact(value["p2"], {"package", "plan", "actual_receipt", "policy", "observed", "holdout"}, "P3 rejected qualification P2")
        package = exact(p2["package"], {"root", "receipt_sha256", "sha256sums_sha256"}, "P2 rejection package reference")
        derived = validate_p2_rejection_package(Path(package["root"]))
        if not strict_equal(p2, derived):
            raise QualificationError("P3 qualification differs from P2 rejection evidence")
        return {"status": "valid_rejected_no_go", "promotion_eligible": False, "reason": REASON, "qualification_sha256": value["qualification_sha256"]}
    if value["status"] == "qualified_go":
        if value["promotion_eligible"] is not True or value["reason"] != GO_REASON:
            raise QualificationError("P3 qualified qualification state differs")
        p2 = exact(value["p2"], {"plan", "actual_receipt", "policy", "calibration_metrics", "freeze_receipt", "preflight", "ledger", "holdout_metrics", "holdout_receipt", "holdout"}, "P3 qualified qualification P2")
        paths = [validate_ref(p2[name], f"P2 {name}") for name in ("plan", "calibration_metrics", "freeze_receipt", "preflight", "ledger", "holdout_metrics", "holdout_receipt")]
        derived = validate_p2_success(*paths)
        if not strict_equal(p2, derived):
            raise QualificationError("P3 qualification differs from P2 success evidence")
        return {"status": "valid_qualified_go", "promotion_eligible": True, "reason": GO_REASON, "qualification_sha256": value["qualification_sha256"]}
    raise QualificationError("P3 upstream qualification variant is unknown")


def load(path: Path) -> tuple[dict[str, Any], str]:
    value, raw = parse(path, "P3 upstream qualification")
    validate(value)
    return value, sha_bytes(raw)


def publish(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise QualificationError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode("ascii") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise QualificationError(f"refusing to overwrite output: {path}") from error
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-rejection")
    build.add_argument("--p2-no-go-evidence", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    qualified = sub.add_parser("build-qualified")
    for option in ("plan", "calibration-metrics", "freeze", "preflight", "ledger", "holdout-metrics", "holdout"):
        qualified.add_argument(f"--{option}", type=Path, required=True)
    qualified.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("validate")
    check.add_argument("--qualification", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build-rejection":
            value = build_rejection(args.p2_no_go_evidence.resolve())
            validate(value)
            publish(args.output, value)
            result = validate(value)
        elif args.command == "build-qualified":
            paths = [getattr(args, name).resolve() for name in ("plan", "calibration_metrics", "freeze", "preflight", "ledger", "holdout_metrics", "holdout")]
            value = build_qualified(paths)
            validate(value)
            publish(args.output, value)
            result = validate(value)
        else:
            value, file_sha = load(args.qualification.resolve())
            result = {**validate(value), "file_sha256": file_sha}
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, QualificationError, ValueError, KeyError) as error:
        print(f"AQ4 P3 upstream qualification failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
