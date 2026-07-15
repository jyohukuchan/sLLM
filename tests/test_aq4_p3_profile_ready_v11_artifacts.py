from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/run-aq4-p2-resident-smoke-maintenance.py"
SPEC = importlib.util.spec_from_file_location("aq4_profile_ready_v11_artifacts", SCRIPT)
assert SPEC and SPEC.loader
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


def verify_sealed(root: Path) -> dict[str, str]:
    metadata = root.lstat()
    assert stat.S_ISDIR(metadata.st_mode) and not root.is_symlink()
    assert stat.S_IMODE(metadata.st_mode) == 0o555
    declared: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        declared[name] = digest
    assert set(declared) == {item.name for item in root.iterdir()} - {"SHA256SUMS"}
    for path in root.iterdir():
        child = path.lstat()
        assert stat.S_ISREG(child.st_mode) and child.st_nlink == 1
        assert stat.S_IMODE(child.st_mode) == 0o444
        if path.name != "SHA256SUMS":
            assert hashlib.sha256(path.read_bytes()).hexdigest() == declared[path.name]
    return declared


def test_profile_ready_v11_is_sealed_and_pins_final_authorities() -> None:
    verify_sealed(HARNESS.PROFILE_READY_ROOT)
    ready = HARNESS.load_ready_artifact(HARNESS.PROFILE_READY_PATH)
    trust = json.loads(HARNESS.PROFILE_HARNESS_TRUST_PATH.read_text())
    qa = json.loads(HARNESS.PROFILE_ATTESTATION_PATH.read_text())
    assert trust["commit"] == "7e6486b4055e72584fcd2dfd9a6251048d683906"
    assert trust["git_blob"] == "a1a33bbb6249e6605ae73f2b3626e29777476b2d"
    assert trust["sha256"] == "e81ec8f6f93a32881293403abef8e4ee2338d43862972d416efb432c3715e0ac"
    assert ready["status"] == "ready_for_one_case" and ready["actual_eligible"] is True
    assert ready["execution_mode"] == "profile_diagnostic"
    assert ready["authorization"]["run_id"] == "p2-r9700-resident-one-case-smoke-profile-diagnostic-v8"
    assert ready["trust"]["launcher"]["commit"] == "b81066dbf86857afbeb0dc7d41493fdef680266d"
    assert ready["profile_diagnostic"]["capture_tool"]["commit"] == "a098ca53c1c3e5c16ec02a08013c55b82f18301c"
    helpers = {
        item["role"]: item
        for item in ready["launcher_binding"]["execution_contract"]["target_manifest"]["capture_helpers"]
    }
    assert helpers["selection_raw_producer"]["sha256"] == "d0360a494f30c2bbac7ca1d043385dd6de9384fa2d81ab99881e54afeaaed934"
    assert qa["automated_tests"]["aggregate"] == {
        "collected": 623,
        "deselected": 0,
        "distinct_test_file_count": 12,
        "failed": 0,
        "passed": 623,
    }
    assert hashlib.sha256((HARNESS.PROFILE_READY_ROOT / "ready-binding.json").read_bytes()).hexdigest() == "ef23daf6b8166abc98fa0a72a0eeeae86ab24b5b1747ff0018c4240398ba0c18"
    assert hashlib.sha256((HARNESS.PROFILE_READY_ROOT / "SHA256SUMS").read_bytes()).hexdigest() == "7bb6a891969ef73a3024aec370c8e38a245bb95e21711e0f1b6068cdfabf9217"
    execute_binding = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-execute-binding-v8/SHA256SUMS"
    assert hashlib.sha256(execute_binding.read_bytes()).hexdigest() == "cb369d7eeab3aae6ab8370f956f8df564a1b3f8ac9ef6415c6d307d2c5ab7ca6"


def test_profile_ready_v11_dry_run_is_process_zero_and_sealed() -> None:
    verify_sealed(HARNESS.PROFILE_DRY_RUN_EVIDENCE)
    evidence = json.loads((HARNESS.PROFILE_DRY_RUN_EVIDENCE / "launcher-evidence.json").read_text())
    assert evidence["status"] == "passed" and evidence["mode"] == "dry-run"
    assert evidence["execution_mode"] == "profile_diagnostic" and evidence["actual_eligible"] is True
    assert evidence["process_counts"] and set(evidence["process_counts"].values()) == {0}
    assert evidence["service_touched"] is evidence["gpu_command_executed"] is evidence["model_load_executed"] is False
    assert evidence["profile_diagnostic"]["capture_executed"] is False
    assert evidence["ready_binding_sha256"] == hashlib.sha256(HARNESS.PROFILE_READY_PATH.read_bytes()).hexdigest()
    assert hashlib.sha256((HARNESS.PROFILE_DRY_RUN_EVIDENCE / "launcher-evidence.json").read_bytes()).hexdigest() == "b5a863514207ed7055689a9b26e839254ec5805c67cfb626352904121a0dcd2a"
    assert hashlib.sha256((HARNESS.PROFILE_DRY_RUN_EVIDENCE / "SHA256SUMS").read_bytes()).hexdigest() == "5f09fe28a036e0fe476e3c9d2fd1003dd52f775bb77711347feb10647002841b"
