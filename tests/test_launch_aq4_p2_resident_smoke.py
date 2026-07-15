from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/launch-aq4-p2-resident-smoke.py"
SPEC = importlib.util.spec_from_file_location("aq4_p2_immutable_launcher", SCRIPT)
assert SPEC and SPEC.loader
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def validator_success() -> bytes:
    return (json.dumps({"status": "prepared_not_executed", "promotion": False, "run_id": "p2-r9700-resident-one-case-smoke-binding-v7"}, sort_keys=True) + "\n").encode()


def test_real_dry_run_orders_validator_before_runner_and_writes_immutable_evidence(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    completed = subprocess.run([sys.executable, str(SCRIPT), "--mode", "dry-run", "--evidence-output", str(output)], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    evidence = json.loads((output / "launcher-evidence.json").read_text())
    assert evidence["status"] == "passed"
    assert evidence["sequence"] == ["validator", "runner"]
    assert evidence["process_counts"] == {"launcher_validator": 1, "runner": 1, "runner_internal_validator": 1, "fake_driver": 1}
    assert evidence["result"] == {"kind": "dry_run_plan", "sha256": LAUNCHER.BINDING_PLAN_SHA, "B_plan_match": True}
    assert evidence["safety"] == {"gpu_command_executed": False, "model_load_executed": False, "service_touched": False, "service_stopped": False}
    assert evidence["self"]["sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert set(output.iterdir()) == {output / name for name in ("launcher-evidence.json", "runner-plan.json", "runner.stdout.bin", "runner.stderr.bin", "validator.stdout.bin", "validator.stderr.bin", "SHA256SUMS")}
    for path in output.iterdir():
        assert path.stat().st_nlink == 1
        assert path.stat().st_mode & 0o777 == 0o444
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_binding_manifest_rejects_unknown_duplicate_and_rebound_values() -> None:
    raw = LAUNCHER.BINDING_MANIFEST.read_bytes()
    value = json.loads(raw)
    value["unknown"] = True
    with pytest.raises(LAUNCHER.LauncherError, match="exact schema"):
        LAUNCHER.validate_binding_manifest(LAUNCHER.pretty(value))

    duplicate = raw.replace(b'  "status": "prepared_not_executed"', b'  "status": "prepared_not_executed",\n  "status": "prepared_not_executed"', 1)
    with pytest.raises(LAUNCHER.LauncherError, match="duplicate JSON key"):
        LAUNCHER.validate_binding_manifest(duplicate)

    rebound = json.loads(raw)
    rebound["trust_roots"]["runner"]["sha256"] = "0" * 64
    with pytest.raises(LAUNCHER.LauncherError, match="runner trust root"):
        LAUNCHER.validate_binding_manifest(LAUNCHER.pretty(rebound))

    for field, replacement in (
        ("source_tree", "0" * 40),
        ("source_sha256", "0" * 64),
        ("binary_build_id_sha1", "0" * 40),
    ):
        rebound = json.loads(raw)
        rebound["trust_roots"]["resident_driver"][field] = replacement
        with pytest.raises(LAUNCHER.LauncherError, match="resident trust root"):
            LAUNCHER.validate_binding_manifest(LAUNCHER.pretty(rebound))
    rebound = json.loads(raw)
    rebound["trust_roots"]["resident_driver"]["build"]["cargo_build_jobs"] = 2
    with pytest.raises(LAUNCHER.LauncherError, match="resident trust root"):
        LAUNCHER.validate_binding_manifest(LAUNCHER.pretty(rebound))

    rebound = json.loads(raw)
    rebound["binding_root_contract"]["mode"] = "0755"
    with pytest.raises(LAUNCHER.LauncherError, match="root contract"):
        LAUNCHER.validate_binding_manifest(LAUNCHER.pretty(rebound))


def test_binding_root_is_sealed_read_only() -> None:
    metadata = LAUNCHER.BINDING_ROOT.lstat()
    assert metadata.st_mode & 0o777 == LAUNCHER.BINDING_ROOT_MODE == 0o555
    assert metadata.st_nlink == 2
    for path in LAUNCHER.BINDING_ROOT.iterdir():
        member = path.lstat()
        assert member.st_mode & 0o777 == 0o444
        assert member.st_nlink == 1


def test_rejects_symlinked_ancestor_and_existing_output(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(LAUNCHER.LauncherError, match="symlink component"):
        LAUNCHER.reject_symlink_components(alias / "evidence", "test", allow_missing_leaf=True)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(LAUNCHER.LauncherError, match="already exists"):
        LAUNCHER.launch("dry-run", existing)


def test_validator_failure_records_evidence_and_starts_zero_runners(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fail_validator(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=b"rejected\n")

    code, evidence = LAUNCHER.launch("dry-run", tmp_path / "validator-failure", run=fail_validator)
    assert code == 1
    assert len(calls) == 1
    assert evidence["process_counts"] == {"launcher_validator": 1, "runner": 0, "runner_internal_validator": 0, "fake_driver": 0}
    assert evidence["failure"]["stage"] == "validator"
    assert evidence["failure"]["runner_started"] is False
    assert (tmp_path / "validator-failure/launcher-evidence.json").is_file()


def test_runner_failure_is_single_attempt_and_preserves_failure_evidence(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def validator_then_runner_failure(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            return subprocess.CompletedProcess(argv, 0, stdout=validator_success(), stderr=b"")
        return subprocess.CompletedProcess(argv, 7, stdout=b"partial", stderr=b"runner failed\n")

    code, evidence = LAUNCHER.launch("dry-run", tmp_path / "runner-failure", run=validator_then_runner_failure)
    assert code == 1
    assert len(calls) == 2
    assert evidence["sequence"] == ["validator", "runner"]
    assert evidence["process_counts"]["launcher_validator"] == 1
    assert evidence["process_counts"]["runner"] == 1
    assert evidence["failure"] == {"stage": "runner", "reason": "trusted runner subprocess failed", "runner_started": True}
    assert (tmp_path / "runner-failure/runner.stderr.bin").read_bytes() == b"runner failed\n"


def test_snapshot_detects_late_replacement(tmp_path: Path) -> None:
    path = tmp_path / "member"
    path.write_bytes(b"trusted")
    snapshot = LAUNCHER.Snapshot()
    snapshot.file(path, hashlib.sha256(b"trusted").hexdigest(), "member")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"trusted")
    path.unlink()
    os.link(replacement, path)
    with pytest.raises(LAUNCHER.LauncherError, match="late replacement"):
        snapshot.verify()


def test_execute_mode_is_disabled_without_starting_processes(tmp_path: Path) -> None:
    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("no subprocess may start")

    code, evidence = LAUNCHER.launch("execute", tmp_path / "execute-rejected", run=forbidden)
    assert code == 1
    assert calls == 0
    assert evidence["failure"]["stage"] == "constants"
    assert evidence["process_counts"] == {"launcher_validator": 0, "runner": 0, "runner_internal_validator": 0, "fake_driver": 0}
    assert evidence["safety"]["gpu_command_executed"] is False
