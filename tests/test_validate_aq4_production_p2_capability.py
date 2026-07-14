from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("aq4_p2_validator", ROOT / "tools/validate-aq4-production-p2-evidence.py")
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def _write(path: Path, value: dict) -> dict:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _case(control_id: str = "reference_source_oracle") -> dict:
    return {"case_id": "p2-smoke-capability", "case_sha256": "a" * 64, "control_id": control_id, "format_id": "REFERENCE", "scope": "full_model", "device": {"backend": "cpu"}}


def _bundle(tmp_path: Path, control_id: str = "reference_source_oracle") -> tuple[dict, dict, dict, dict]:
    served_path = tmp_path / "served.json"
    served = {"schema_version": "ullm.served_model.v2", "worker": {"identity": {"device": "gfx1201"}}}
    served_link = _write(served_path, served)
    identity = {"artifacts": {"served_model_manifest": str(served_path)}, "hash_binding": {"served_model_manifest_sha256": served_link["sha256"], "worker_binary_sha256": "b" * 64}}
    case = _case(control_id)
    reason = "cpu_reference_unsupported_by_hip_resident_worker"
    capability = {"schema_version": VALIDATOR.CAPABILITY_SCHEMA, "status": "unsupported", "reason_code": reason, "case_id": case["case_id"], "case_sha256": case["case_sha256"], "served_worker": {"backend": "hip", "architecture": "gfx1201", "binary_sha256": "b" * 64, "manifest_sha256": served_link["sha256"]}, "model_loads": 0, "gpu_processes": 0, "evidence_class": "immutable_capability_record"}
    raw = {"status": "unsupported", "immutable_status": True, "failure_reason": reason, "execution": {"status": "unsupported", "returncode": None}}
    result = {"status": "unsupported", "capability": capability, "oracles": {}}
    if control_id == "reference_source_oracle":
        manifest_path = tmp_path / "source-manifest.json"
        payload_path = tmp_path / "source-payload.jsonl"
        validator_path = tmp_path / "source-validator.json"
        manifest_link = _write(manifest_path, {"schema_version": VALIDATOR.SOURCE_ORACLE_SCHEMA, "oracle_kind": "independent_source", "status": "fixture", "cases": [{"case_id": case["case_id"]}]})
        payload_link = _write(payload_path, {"rows": []})
        validator_link = _write(validator_path, {"schema_version": VALIDATOR.SOURCE_ORACLE_VALIDATOR_SCHEMA, "status": "valid", "oracle_kind": "independent_source", "manifest_sha256": manifest_link["sha256"]})
        result["oracles"]["source_oracle"] = {"manifest": manifest_link, "payload": payload_link, "validator": validator_link}
    else:
        identity_path = tmp_path / "sq8-identity.json"
        evidence_path = tmp_path / "sq8-evidence.json"
        result["oracles"]["cross_format_control"] = {"schema_version": VALIDATOR.SQ8_CANONICAL_SCHEMA, "identity": _write(identity_path, {"schema_version": VALIDATOR.SQ8_CANONICAL_SCHEMA, "source_correct": True}), "evidence": _write(evidence_path, {"schema_version": VALIDATOR.SQ8_CANONICAL_SCHEMA, "source_correct": True}), "source_correct": True}
    return case, result, raw, identity


def test_cpu_unsupported_capability_is_accepted_without_measurement(tmp_path: Path) -> None:
    case, result, raw, identity = _bundle(tmp_path)
    assert VALIDATOR.validate_capability_result(tmp_path, case, result, raw, identity) == []
    result["capability"]["model_loads"] = 1
    assert any(code.startswith("capability_zero_execution") for code in VALIDATOR.validate_capability_result(tmp_path, case, result, raw, identity))


def test_capability_reason_and_reference_oracle_mutations_fail_closed(tmp_path: Path) -> None:
    case, result, raw, identity = _bundle(tmp_path)
    result["capability"]["reason_code"] = "ok"
    assert any(code.startswith("capability_reason") for code in VALIDATOR.validate_capability_result(tmp_path, case, result, raw, identity))
    case, result, raw, identity = _bundle(tmp_path, "reference_source_oracle")
    del result["oracles"]["source_oracle"]["validator"]
    assert any(code.startswith("capability_source_oracle") for code in VALIDATOR.validate_capability_result(tmp_path, case, result, raw, identity))


def test_sq8_cross_format_requires_canonical_v02_identity_and_evidence(tmp_path: Path) -> None:
    case, result, raw, identity = _bundle(tmp_path, "sq8_0_cross_format")
    assert VALIDATOR.validate_capability_result(tmp_path, case, result, raw, identity) == []
    result["oracles"]["cross_format_control"]["schema_version"] = "legacy"
    assert any(code.startswith("capability_sq8_control") for code in VALIDATOR.validate_capability_result(tmp_path, case, result, raw, identity))
