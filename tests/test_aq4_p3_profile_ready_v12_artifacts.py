from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/run-aq4-p2-resident-smoke-maintenance.py"
SPEC = importlib.util.spec_from_file_location("aq4_profile_ready_v12_artifacts", SCRIPT)
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


def test_profile_ready_v12_is_sealed_and_pins_final_authorities() -> None:
    verify_sealed(HARNESS.PROFILE_READY_ROOT)
    ready = HARNESS.load_ready_artifact(HARNESS.PROFILE_READY_PATH)
    trust = json.loads(HARNESS.PROFILE_HARNESS_TRUST_PATH.read_text())
    qa = json.loads(HARNESS.PROFILE_ATTESTATION_PATH.read_text())
    assert trust["commit"] == "62e8fe91e073575c4776603786f9909f2b8001cd"
    assert trust["git_blob"] == "bffd039b86fdbb5d3cff7402e30f8b12f7ab2e1b"
    assert trust["sha256"] == "3295e56fba8b5139ffca55cc3d742d83a916aa8cb1cf53ded4f7f41fb268892d"
    assert ready["status"] == "ready_for_one_case" and ready["actual_eligible"] is True
    assert ready["execution_mode"] == "profile_diagnostic"
    assert ready["authorization"]["run_id"] == "p2-r9700-resident-one-case-smoke-profile-diagnostic-v9"
    assert ready["trust"]["launcher"]["commit"] == "7f961f8de75ccbb1080fcd35a5b274584d4e00f3"
    assert ready["profile_diagnostic"]["capture_tool"]["commit"] == "1aed601a7e4102c99550b09384ef45fe57d43287"
    helpers = {
        item["role"]: item
        for item in ready["launcher_binding"]["execution_contract"]["target_manifest"]["capture_helpers"]
    }
    assert helpers["selection_raw_producer"]["sha256"] == "a589c3e644d36132fb6054afdb15b27543d8e8181e3c737dcbd071d7c52e3d20"
    assert helpers["profile_family_classifier"]["sha256"] == "f8d32c340231e329f004d9e16192c02378f1fd58b8ab713e8efbbd3029b052d6"
    assert qa["automated_tests"]["aggregate"] == {
        "collected": 639,
        "deselected": 0,
        "distinct_test_file_count": 12,
        "failed": 0,
        "passed": 639,
    }
    assert hashlib.sha256((HARNESS.PROFILE_READY_ROOT / "ready-binding.json").read_bytes()).hexdigest() == "4c1fcee0c980e341e5346066a4a59bd7c8ace9eab562e18189b7050ceaf52890"
    assert hashlib.sha256((HARNESS.PROFILE_READY_ROOT / "SHA256SUMS").read_bytes()).hexdigest() == "c81139e9361b1a8ee740c3d0cb3202f333c5ccd88a4f766a9edd756a54fba575"
    execute_binding = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-execute-binding-v9/SHA256SUMS"
    assert hashlib.sha256(execute_binding.read_bytes()).hexdigest() == "b5764d5bdbcfd2fcd56e602c88081796913429fe6137711e20c249bc524a1532"


def test_profile_ready_v12_dry_run_is_process_zero_and_sealed() -> None:
    verify_sealed(HARNESS.PROFILE_DRY_RUN_EVIDENCE)
    evidence = json.loads((HARNESS.PROFILE_DRY_RUN_EVIDENCE / "launcher-evidence.json").read_text())
    assert evidence["status"] == "passed" and evidence["mode"] == "dry-run"
    assert evidence["execution_mode"] == "profile_diagnostic" and evidence["actual_eligible"] is True
    assert evidence["process_counts"] and set(evidence["process_counts"].values()) == {0}
    assert evidence["service_touched"] is evidence["gpu_command_executed"] is evidence["model_load_executed"] is False
    assert evidence["profile_diagnostic"]["capture_executed"] is False
    assert evidence["ready_binding_sha256"] == hashlib.sha256(HARNESS.PROFILE_READY_PATH.read_bytes()).hexdigest()
    assert hashlib.sha256((HARNESS.PROFILE_DRY_RUN_EVIDENCE / "launcher-evidence.json").read_bytes()).hexdigest() == "20834ed11e58b1d440c015d4bc38f4ab2fc6321dac6b2e86cdb44949d809a70e"
    assert hashlib.sha256((HARNESS.PROFILE_DRY_RUN_EVIDENCE / "SHA256SUMS").read_bytes()).hexdigest() == "87ebe1c6db8211bb3fa8818a5216d19e63237a27e42f1ee5513418febf75ebf3"
