from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("aq4_p3_upstream_qualification", ROOT / "tools/aq4_p3_upstream_qualification.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: object) -> bytes:
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode("ascii") + b"\n"
    path.write_bytes(raw)
    return raw


def write_bound(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def success_chain(root: Path, *, request_id: str = "synthetic-go") -> list[Path]:
    root.mkdir()
    actual = root / "actual.json"
    source_v32 = root / "source-v32.json"
    policy = root / "policy.json"
    calibration_cases = root / "calibration.jsonl"
    holdout_cases = root / "holdout.jsonl"
    actual_sha = write_bound(actual, b'{"status":"actual_verified"}\n')
    source_v32_sha = write_bound(source_v32, b'{"version":32}\n')
    policy_sha = write_bound(policy, b'{"schema_version":"ullm.aq4_p2_fidelity_policy.v1"}\n')
    calibration_sha = write_bound(calibration_cases, b"synthetic-calibration-cases\n")
    holdout_cases_sha = write_bound(holdout_cases, b"synthetic-holdout-cases\n")
    hex64 = "1" * 64
    identity = {
        "sq8_receipt_path": str(actual), "sq8_receipt_sha256": actual_sha,
        "receipt_state": "actual_verified", "request_id": request_id, "source_commit": "abc",
        "source_tree_sha256": hex64, "source_archive_sha256": "2" * 64,
        "source_v32_path": str(source_v32), "source_v32_sha256": source_v32_sha,
        "served_model": {}, "worker": {}, "package": {}, "overlay_content_sha256": "3" * 64,
        "overlay_tensor_set_sha256": "4" * 64, "token_ids_sha256": None,
        "telemetry_binding": {}, "maintenance_evidence": {}, "executor_record": {},
        "prepared_receipt": {}, "split_manifest_sha256": "5" * 64,
        "calibration_cases_path": str(calibration_cases), "calibration_cases_sha256": calibration_sha,
        "holdout_cases_path": str(holdout_cases), "holdout_cases_sha256": holdout_cases_sha,
        "policy_path": str(policy), "policy_sha256": policy_sha,
    }
    plan_path = root / "plan.json"
    plan = {
        "schema_version": MODULE.P2_PLAN_SCHEMA, "status": "ready_for_calibration",
        "preflight_only": False, "actual_verified_required": True, "identity": identity,
        "policy": {"schema_version": "ullm.aq4_p2_fidelity_policy.v1"},
        "calibration": {"path": str(calibration_cases), "sha256": calibration_sha, "row_count": 24},
        "holdout": {"path": str(holdout_cases), "sha256": holdout_cases_sha, "row_count": 24},
        "resource_contract": {"jobs": 1, "case_concurrency": 1, "one_model_load": True,
            "chunk_elements": 65_536, "bounded_vectors": True, "bounded_disk": True,
            "max_rows": 24, "max_case_file_bytes": 64 * 1024 * 1024,
            "vram_headroom_required": True, "vram_headroom_bytes_min": 1,
            "vram_observed_headroom_bytes": 1},
        "holdout_state": {"status": "not_started", "evaluations_remaining": 1, "retry_permitted": False},
    }
    plan_raw = write_json(plan_path, plan)
    metrics_fields = {"schema_version": MODULE.P2_METRICS_SCHEMA, "identity": identity,
                      "evidence": {}, "rows": [{} for _ in range(24)]}
    calibration_metrics_path = root / "calibration-metrics.json"
    calibration_raw = write_json(calibration_metrics_path, {**metrics_fields, "subset": "calibration"})
    bounds = {"token_agreement_rate": {"bound": 0.5}}
    freeze_path = root / "freeze.json"
    freeze = {
        "schema_version": MODULE.P2_FREEZE_SCHEMA, "status": "frozen_calibration_envelope",
        "identity": identity, "plan_path": str(plan_path), "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "metrics_path": str(calibration_metrics_path), "metrics_sha256": hashlib.sha256(calibration_raw).hexdigest(),
        "calibration_case_count": 24, "derived_bounds": bounds, "holdout_status": "not_started",
        "holdout_evaluations_remaining": 1, "retry_permitted": False,
        "relative_l2_rejection_ceiling": 1.0,
        "attempt_boundary": {"remaining_before": 1, "remaining_after": 0, "failure_consumes_attempt": True},
    }
    freeze_raw = write_json(freeze_path, freeze)
    preflight_path = root / "preflight.json"
    preflight = {
        "schema_version": MODULE.P2_PREFLIGHT_SCHEMA, "status": "ready_for_one_shot_holdout",
        "freeze_receipt_sha256": hashlib.sha256(freeze_raw).hexdigest(), "freeze_receipt_path": str(freeze_path),
        "plan_path": str(plan_path), "plan_sha256": hashlib.sha256(plan_raw).hexdigest(), "identity": identity,
        "holdout_cases_sha256": holdout_cases_sha, "holdout_case_count": 24,
        "evaluations_remaining": 1, "retry_permitted": False,
        "attempt_boundary": {"remaining_before": 1, "remaining_after": 0, "failure_consumes_attempt": True},
    }
    preflight_raw = write_json(preflight_path, preflight)
    preflight_sha = hashlib.sha256(preflight_raw).hexdigest()
    attempt_id = hashlib.sha256(b"ullm.qwen35-aq4-sq8-holdout-attempt-v1\0" + preflight_sha.encode()).hexdigest()
    ledger_path = root / "ledger.json"
    ledger_raw = write_json(ledger_path, {
        "schema_version": MODULE.P2_LEDGER_SCHEMA, "status": "consumed", "attempt_id": attempt_id,
        "preflight_sha256": preflight_sha, "identity": identity, "remaining_before": 1,
        "remaining_after": 0, "retry_permitted": False,
    })
    holdout_metrics_path = root / "holdout-metrics.json"
    holdout_metrics_raw = write_json(holdout_metrics_path, {**metrics_fields, "subset": "holdout"})
    holdout_path = root / "holdout.json"
    write_json(holdout_path, {
        "schema_version": MODULE.P2_HOLDOUT_SCHEMA, "attempt_schema": MODULE.P2_ATTEMPT_SCHEMA,
        "status": "passed", "attempt_id": attempt_id, "preflight_sha256": preflight_sha,
        "ledger_sha256": hashlib.sha256(ledger_raw).hexdigest(),
        "metrics_sha256": hashlib.sha256(holdout_metrics_raw).hexdigest(), "identity": identity,
        "derived_metrics": {"token_agreement_rate": {"bound": 0.6}},
        "gate_checks": {"token_agreement_rate": True}, "retry_permitted": False,
        "evaluations_remaining": 0,
    })
    return [plan_path, calibration_metrics_path, freeze_path, preflight_path, ledger_path, holdout_metrics_path, holdout_path]


def rejection_package(root: Path) -> Path:
    root.mkdir()
    evidence_root = root / "evidence"
    evidence_root.mkdir()
    plan = root / "plan.json"
    actual = root / "actual.json"
    policy = root / "policy.json"
    write_json(plan, {"status": "ready_for_calibration"})
    write_json(actual, {"status": "actual_verified"})
    write_json(policy, {"ceiling": 1.0})
    bindings: dict[str, object] = {
        "plan": MODULE.file_ref(plan, "plan"),
        "actual_receipt": MODULE.file_ref(actual, "actual"),
    }
    for name in ("source_artifact", "target_artifact", "comparison"):
        artifact = root / name
        artifact.mkdir()
        write_json(artifact / "manifest.json", {"name": name})
        (artifact / "SHA256SUMS").write_text("synthetic\n", encoding="ascii")
        bindings[name] = {
            "root": str(artifact),
            "manifest_sha256": hashlib.sha256((artifact / "manifest.json").read_bytes()).hexdigest(),
            "sha256sums_sha256": hashlib.sha256((artifact / "SHA256SUMS").read_bytes()).hexdigest(),
        }
    receipt = {
        "schema_version": MODULE.P2_REJECTION_SCHEMA,
        "status": "calibration_rejected_no_go",
        "reason": "relative_l2_pathological_rejection_before_aggregation",
        "policy": {"path": str(policy), "sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
                   "ceiling": 1.0, "action": "reject any observed relative-L2 > 1 before aggregation"},
        "observed": {
            "row_count": 24, "nonfinite_rows": 0,
            "logits_relative_l2": {"count_above_ceiling": 1, "minimum": 0.9, "mean": 1.0, "maximum": 1.1},
            "hidden_relative_l2": {"count_above_ceiling": 0, "minimum": 0.1, "mean": 0.2, "maximum": 0.3},
            "greedy_mismatch_rows": 1, "minimum_top_k_overlap": 9,
        },
        "state": {"metrics_published": False, "freeze_published": False, "holdout_status": "not_started",
                  "holdout_evaluations_remaining": 1, "retry_permitted": False, "holdout_executed": False},
        "bindings": bindings, "capture": {}, "lineage": {},
    }
    receipt_path = evidence_root / MODULE.P2_RECEIPT_NAME
    write_json(receipt_path, receipt)
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    sums = evidence_root / MODULE.P2_SUMS_NAME
    sums.write_text(f"{receipt_sha}  {MODULE.P2_RECEIPT_NAME}\n", encoding="ascii")
    receipt_path.chmod(0o444)
    sums.chmod(0o444)
    evidence_root.chmod(0o555)
    return evidence_root


def test_synthetic_official_success_chain_is_the_only_go_fixture(tmp_path: Path) -> None:
    paths = success_chain(tmp_path / "go")
    value = MODULE.build_qualified(paths)
    result = MODULE.validate(value)
    assert result["status"] == "valid_qualified_go"
    assert result["promotion_eligible"] is True


def test_synthetic_rejection_package_is_never_promotable(tmp_path: Path) -> None:
    value = MODULE.build_rejection(rejection_package(tmp_path / "rejected"))
    result = MODULE.validate(value)
    assert result["status"] == "valid_rejected_no_go"
    assert result["promotion_eligible"] is False


@pytest.mark.parametrize("mutation", ["unknown", "missing", "bool_as_int", "hash", "cross_variant"])
def test_qualified_union_rejects_schema_and_binding_mutations(tmp_path: Path, mutation: str) -> None:
    paths = success_chain(tmp_path / "go")
    value = MODULE.build_qualified(paths)
    changed = copy.deepcopy(value)
    if mutation == "unknown":
        changed["unknown"] = True
    elif mutation == "missing":
        del changed["reason"]
    elif mutation == "bool_as_int":
        plan = json.loads(paths[0].read_text())
        plan["resource_contract"]["jobs"] = True
        write_json(paths[0], plan)
    elif mutation == "hash":
        changed["p2"]["holdout_receipt"]["sha256"] = "0" * 64
    else:
        changed["status"] = "rejected_no_go"
        changed["promotion_eligible"] = False
        changed["reason"] = MODULE.REASON
    if mutation in {"unknown", "missing", "hash", "cross_variant"}:
        changed["qualification_sha256"] = MODULE.self_hash(changed)
    with pytest.raises(MODULE.QualificationError):
        MODULE.validate(changed)


def test_qualified_union_rejects_receipt_swap(tmp_path: Path) -> None:
    first = success_chain(tmp_path / "first", request_id="first")
    second = success_chain(tmp_path / "second", request_id="second")
    value = MODULE.build_qualified(first)
    swapped = copy.deepcopy(value)
    swapped["p2"]["holdout_receipt"] = MODULE.file_ref(second[-1], "swapped receipt")
    swapped["qualification_sha256"] = MODULE.self_hash(swapped)
    with pytest.raises(MODULE.QualificationError):
        MODULE.validate(swapped)


def test_rejected_variant_cannot_be_relabelled_as_go(tmp_path: Path) -> None:
    paths = success_chain(tmp_path / "go")
    qualified = MODULE.build_qualified(paths)
    qualified["p2"] = {"package": {}, "plan": {}, "actual_receipt": {}, "policy": {}, "observed": {}, "holdout": {}}
    qualified["qualification_sha256"] = MODULE.self_hash(qualified)
    with pytest.raises(MODULE.QualificationError):
        MODULE.validate(qualified)
