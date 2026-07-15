from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2"
READY_ROOT = BASE / "resident-one-case-smoke-profile-ready-v7"
DRY_ROOT = BASE / "resident-one-case-smoke-profile-ready-dry-run-v7"


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


def test_profile_ready_v7_is_sealed_and_pins_final_authorities() -> None:
    verify_sealed(READY_ROOT)
    ready = json.loads((READY_ROOT / "ready-binding.json").read_text())
    trust = json.loads((READY_ROOT / "harness-trust.json").read_text())
    assert trust["commit"] == "3fc2b8cd6f6910fbebd3ff4728855d55bf2cbbd2"
    assert trust["git_blob"] == "9b5566a0f6d1381732342c0ee26f9778c54f852b"
    assert trust["sha256"] == "6a964e0dc93c889a31e28e89ccbc25ba5e0db095aad3d7c2ca427230c36428b0"
    assert ready["status"] == "ready_for_one_case" and ready["actual_eligible"] is True
    assert ready["execution_mode"] == "profile_diagnostic"
    assert ready["authorization"]["run_id"] == "p2-r9700-resident-one-case-smoke-profile-diagnostic-v7"
    binding = ready["launcher_binding"]
    assert binding["runner_output"].endswith("resident-one-case-smoke-profile-execute-v7")
    assert binding["evidence_output"].endswith("resident-one-case-smoke-profile-execute-evidence-v7")
    roctx = ready["profile_diagnostic"]["roctx"]["roctx_library"]
    assert roctx["invocation_path"] == roctx["resolved_path"] == "/opt/rocm-7.2.1/lib/librocprofiler-sdk-roctx.so.1.1.0"
    assert roctx["sha256"] == "1a5831a3817eac29f63d1442dc348ba31b417202b7ce15f3aed9c09a8f4773c9"
    capture = ready["profile_diagnostic"]["capture_tool"]
    assert capture["commit"] == "e86cf512183574340ddfc6564477395766262092"
    assert capture["git_blob"] == "124f5e89834fda2ace8a2d8c42e362ec1adce29c"
    assert capture["sha256"] == "ab3d77d4bc77c43c82ac9ee1d993a029266119ca3365f1a285ab03cca9bcf00a"


def test_profile_ready_v7_dry_run_is_process_zero_and_sealed() -> None:
    verify_sealed(DRY_ROOT)
    evidence = json.loads((DRY_ROOT / "launcher-evidence.json").read_text())
    assert evidence["status"] == "passed" and evidence["mode"] == "dry-run"
    assert evidence["execution_mode"] == "profile_diagnostic" and evidence["actual_eligible"] is True
    assert evidence["process_counts"] and set(evidence["process_counts"].values()) == {0}
    assert evidence["service_touched"] is evidence["gpu_command_executed"] is evidence["model_load_executed"] is False
    assert evidence["ready_binding_sha256"] == hashlib.sha256((READY_ROOT / "ready-binding.json").read_bytes()).hexdigest()
