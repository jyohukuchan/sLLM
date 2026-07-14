from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/launch-aq4-p2-resident-smoke.py"
SPEC = importlib.util.spec_from_file_location("aq4_p2_execute_launcher", SCRIPT)
assert SPEC and SPEC.loader
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def _validator_success() -> bytes:
    return b'{"promotion": false, "run_id": "p2-r9700-resident-one-case-smoke-binding-v4", "status": "prepared_not_executed"}\n'


def _ready_binding(tmp_path: Path) -> tuple[dict, Path, Path, str]:
    evidence = tmp_path / "execute-evidence"
    result = tmp_path / "execute-result"
    run_id = "execute-test-run"
    value = json.loads(json.dumps(LAUNCHER.execute_binding_document()))
    value.update(status="ready_for_explicit_execute", actual_eligible=True, blocked_reasons=[], evidence_output=str(evidence), runner_output=str(result), run_id=run_id)
    value["live_preflight"] = {"required": True, "path": str(evidence / "live-preflight.json"), "sha256": None, "replaces_synthetic_preflight": True}
    return value, evidence, result, run_id


def _gates() -> dict:
    commands = LAUNCHER.expected_live_probe_contracts()
    return {
        "passed": True,
        "environment": LAUNCHER.EXECUTE_ENV,
        "services": [
            {"unit": "ullm-openai.service", "active_state": "inactive", "sub_state": "dead", "main_pid": 0},
            {"unit": "llama-qwen35-udq4.service", "active_state": "inactive", "sub_state": "dead", "main_pid": 0},
        ],
        "old_worker_pids": [],
        "runtime_mapping": {"runtime_device_index": 1, "visible_token": "1", "amd_smi_index": 2, "bdf": LAUNCHER.GPU_BDF, "uuid": LAUNCHER.GPU_UUID, "kfd_id": LAUNCHER.KFD_ID, "node_id": 2},
        "amd_smi_owners": [], "kfd_owners": [],
        "lock": {"path": str(LAUNCHER.LOCK_PATH), "free": True, "device": 66306, "inode": 123},
        "vram": {"total_bytes": 32_624_000_000, "used_bytes": 0, "free_bytes": 32_624_000_000, "headroom_bytes": 32_624_000_000},
        "probes": [
            {"label": label, "argv": argv, "exit_code": exit_code, "stdout_sha256": "0" * 64, "stderr_sha256": "0" * 64, "captured_unix_ns": index}
            for index, (label, (argv, exit_code)) in enumerate(commands.items())
        ],
    }


def test_execute_bound_generates_live_sidecar_and_exact_runner_argv(tmp_path: Path) -> None:
    binding, evidence_path, result_path, run_id = _ready_binding(tmp_path)
    calls: list[list[str]] = []
    restores: list[bool] = []

    def validator(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, _validator_success(), b"")

    def runner(command: list[str], environment: dict[str, str], on_started):
        assert environment == LAUNCHER.EXECUTE_ENV
        assert command == LAUNCHER.execute_runner_argv(binding)
        assert command[-8:] == [
            "--driver-command", str(LAUNCHER.RESIDENT_DRIVER), "--served-model-manifest", str(LAUNCHER.SERVED_MANIFEST),
            "--device-index", "1", "--build-git-commit", LAUNCHER.RESIDENT_COMMIT,
        ]
        result_path.mkdir()
        (result_path / "case.raw.json").write_text("{}\n")
        (result_path / "resident-batch.summary.json").write_text("{}\n")
        on_started()
        return {"completed": subprocess.CompletedProcess(command, 0, b"", b""), "keepalives": [{"label": "sudo-keepalive-1", "argv": [str(LAUNCHER.SUDO), "-n", "-v"], "exit_code": 0}], "keepalive_failed": False, "gpu_command_executed": True, "model_load_executed": True}

    def restore() -> dict:
        restores.append(True)
        return {"required": False, "service_stop_performed": False, "state_preserved": True}

    code, evidence = LAUNCHER.execute_bound(binding, evidence_path, result_path, run_id, run=validator, gate_provider=_gates, restore_provider=restore, runner_executor=runner)
    assert code == 0
    assert len(calls) == 1 and restores == [True]
    assert evidence["status"] == "passed"
    assert evidence["sequence"] == ["validator", "pre-exec-gates", "runner"]
    assert evidence["process_counts"]["runner"] == 1
    assert evidence["safety"]["gpu_command_executed"] is True
    assert evidence["safety"]["model_load_executed"] is True
    assert evidence["sudo_keepalive"]["failed"] is False
    live = evidence_path / "live-preflight.json"
    assert live.stat().st_mode & 0o777 == 0o444
    value = json.loads(live.read_text())
    assert value["prepared_preflight"]["role"] == "synthetic_bundle_contract_only"
    assert value["runtime_mapping"]["amd_smi_index"] == 2
    assert value["compute_owners"] == {"amd_smi": [], "kfd": []}
    assert value["environment"] == LAUNCHER.EXECUTE_ENV
    assert evidence["result"]["files"] == {"case.raw.json": LAUNCHER.sha_bytes(b"{}\n"), "resident-batch.summary.json": LAUNCHER.sha_bytes(b"{}\n")}


def test_keepalive_failure_interrupts_runner_and_finally_restores(tmp_path: Path) -> None:
    binding, evidence_path, result_path, run_id = _ready_binding(tmp_path)
    restored = False

    def validator(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, _validator_success(), b"")

    def failed_runner(command: list[str], environment: dict[str, str], on_started):
        on_started()
        return {"completed": subprocess.CompletedProcess(command, -2, b"partial", b""), "keepalives": [{"label": "sudo-keepalive-1", "argv": [str(LAUNCHER.SUDO), "-n", "-v"], "exit_code": 1}], "keepalive_failed": True, "gpu_command_executed": "unknown", "model_load_executed": "unknown"}

    def restore() -> dict:
        nonlocal restored
        restored = True
        return {"required": True, "service_stop_performed": False, "state_preserved": True, "priority": "restore_before_reporting"}

    code, evidence = LAUNCHER.execute_bound(binding, evidence_path, result_path, run_id, run=validator, gate_provider=_gates, restore_provider=restore, runner_executor=failed_runner)
    assert code == 1 and restored is True
    assert evidence["failure"]["stage"] == "runner"
    assert evidence["failure"]["runner_started"] is True
    assert "keepalive failed" in evidence["failure"]["reason"]
    assert evidence["restore"]["state_preserved"] is True
    assert evidence["safety"]["gpu_command_executed"] == "unknown"
    assert evidence["safety"]["model_load_executed"] == "unknown"


def test_fake_runner_start_failure_keeps_gpu_and_model_flags_false(tmp_path: Path) -> None:
    binding, evidence_path, result_path, run_id = _ready_binding(tmp_path)

    def validator(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, _validator_success(), b"")

    def start_failure(command: list[str], environment: dict[str, str], on_started):
        raise OSError("synthetic spawn failure")

    code, evidence = LAUNCHER.execute_bound(binding, evidence_path, result_path, run_id, run=validator, gate_provider=_gates, runner_executor=start_failure)
    assert code == 1
    assert evidence["process_counts"]["runner"] == 0
    assert evidence["failure"]["runner_started"] is False
    assert evidence["safety"]["gpu_command_executed"] is False
    assert evidence["safety"]["model_load_executed"] is False
    assert evidence["safety"]["execution_state_source"] == "runner_not_started"


def test_fake_runner_midway_failure_records_proven_gpu_and_model_activity(tmp_path: Path) -> None:
    binding, evidence_path, result_path, run_id = _ready_binding(tmp_path)

    def validator(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, _validator_success(), b"")

    def midway_failure(command: list[str], environment: dict[str, str], on_started):
        on_started()
        return {"completed": subprocess.CompletedProcess(command, 9, b"partial", b""), "keepalives": [], "keepalive_failed": False, "gpu_command_executed": True, "model_load_executed": True}

    code, evidence = LAUNCHER.execute_bound(binding, evidence_path, result_path, run_id, run=validator, gate_provider=_gates, runner_executor=midway_failure)
    assert code == 1
    assert evidence["failure"]["runner_started"] is True
    assert evidence["safety"]["gpu_command_executed"] is True
    assert evidence["safety"]["model_load_executed"] is True
    assert "runner-after" in evidence["trust_verifications"]


@pytest.mark.parametrize(
    ("swap_point", "runner_started", "activity"),
    [("validator-before", False, False), ("runner-before", False, False), ("runner-after", True, True), ("finalize-before", True, True)],
)
def test_execute_snapshot_rejects_stage_specific_toctou_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, swap_point: str, runner_started: bool, activity: bool) -> None:
    binding, evidence_path, result_path, run_id = _ready_binding(tmp_path)
    watched = tmp_path / "watched-tool"
    watched.write_bytes(b"trusted")
    replacement = tmp_path / "replacement-tool"
    replacement.write_bytes(b"trusted")
    original_validate = LAUNCHER.validate_execute_constants

    def validate_with_watched(snapshot, self_sha):
        value = original_validate(snapshot, self_sha)
        snapshot.file(watched, LAUNCHER.sha_bytes(b"trusted"), "TOCTOU watched tool")
        return value

    swapped = False

    def hook(point: str):
        nonlocal swapped
        if point == swap_point and not swapped:
            swapped = True
            os.replace(replacement, watched)

    def validator(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, _validator_success(), b"")

    def successful_runner(command: list[str], environment: dict[str, str], on_started):
        on_started()
        result_path.mkdir()
        (result_path / "resident-batch.summary.json").write_text("{}\n")
        return {"completed": subprocess.CompletedProcess(command, 0, b"", b""), "keepalives": [], "keepalive_failed": False, "gpu_command_executed": True, "model_load_executed": True}

    monkeypatch.setattr(LAUNCHER, "validate_execute_constants", validate_with_watched)
    code, evidence = LAUNCHER.execute_bound(binding, evidence_path, result_path, run_id, run=validator, gate_provider=_gates, runner_executor=successful_runner, verification_hook=hook)
    assert code == 1 and swapped is True
    assert evidence["process_counts"]["runner"] == int(runner_started)
    assert evidence["safety"]["gpu_command_executed"] is activity
    assert evidence["safety"]["model_load_executed"] is activity
    assert "replacement" in evidence["failure"]["reason"]


def test_real_runner_wrapper_uses_fake_sudo_keepalive_and_interrupts_on_failure() -> None:
    sudo_calls = 0

    def fake_sudo(argv, **kwargs):
        nonlocal sudo_calls
        sudo_calls += 1
        return subprocess.CompletedProcess(argv, 1, b"", b"")

    started = time.monotonic()
    outcome = LAUNCHER.run_runner_with_sudo_keepalive(
        [sys.executable, "-c", "import time; time.sleep(10)"], dict(os.environ), sudo_run=fake_sudo, interval=0.02,
    )
    completed, records, failed = outcome["completed"], outcome["keepalives"], outcome["keepalive_failed"]
    assert time.monotonic() - started < 2
    assert sudo_calls == 1 and len(records) == 1 and failed is True
    assert completed.returncode != 0
    assert records[0]["argv"] == [str(LAUNCHER.SUDO), "-n", "-v"]
    assert outcome["gpu_command_executed"] == "unknown"
    assert outcome["model_load_executed"] == "unknown"


def test_real_runner_wrapper_keeps_running_with_fake_valid_sudo() -> None:
    def fake_sudo(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    outcome = LAUNCHER.run_runner_with_sudo_keepalive(
        [sys.executable, "-c", "import time; time.sleep(.08)"], dict(os.environ), sudo_run=fake_sudo, interval=0.02,
    )
    completed, records, failed = outcome["completed"], outcome["keepalives"], outcome["keepalive_failed"]
    assert completed.returncode == 0 and failed is False
    assert len(records) >= 1
    assert all(record["argv"] == [str(LAUNCHER.SUDO), "-n", "-v"] for record in records)
    assert outcome["gpu_command_executed"] is True
    assert outcome["model_load_executed"] is True


def test_execute_rejects_output_reuse_before_starting_processes(tmp_path: Path) -> None:
    binding, evidence_path, result_path, run_id = _ready_binding(tmp_path)
    evidence_path.mkdir()
    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("no process may start")

    with pytest.raises(LAUNCHER.LauncherError, match="already exists"):
        LAUNCHER.execute_bound(binding, evidence_path, result_path, run_id, run=forbidden, gate_provider=_gates)
    assert calls == 0


def test_execute_binding_remains_ineligible_until_live_sidecar_and_qa() -> None:
    value = LAUNCHER.execute_binding_document()
    assert value["actual_eligible"] is False
    assert value["live_preflight"]["sha256"] is None
    assert value["tools"]["sudo"]["prevalidate_argv"] == [str(LAUNCHER.SUDO), "-n", "-v"]
    assert value["blocked_reasons"] == ["live preflight sidecar is absent", "independent execute-launcher QA is pending"]


def test_execute_binding_parent_chain_creation_rejects_symlink(tmp_path: Path) -> None:
    nested = tmp_path / "new" / "deep" / "parent"
    LAUNCHER.ensure_directory_chain(nested, "test parent")
    assert nested.is_dir()
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(LAUNCHER.LauncherError, match="symlink component"):
        LAUNCHER.ensure_directory_chain(alias / "child", "test parent")


def _gate_router(*, duplicate_bdf: bool = False, active_service: bool = False):
    target = {"gpu": 2, "bdf": LAUNCHER.GPU_BDF, "uuid": LAUNCHER.GPU_UUID, "kfd_id": LAUNCHER.KFD_ID, "node_id": 2, "partition_id": 0}
    other = {"gpu": 0, "bdf": "0000:01:00.0", "uuid": "other", "kfd_id": 1, "node_id": 0, "partition_id": 0}
    if duplicate_bdf:
        other["bdf"] = LAUNCHER.GPU_BDF

    def run(argv, **kwargs):
        if argv == [str(LAUNCHER.SUDO), "-n", "-v"]:
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        if argv[:2] == [str(LAUNCHER.SYSTEMCTL), "show"]:
            stdout = b"ActiveState=active\nSubState=running\nMainPID=99\n" if active_service else b"ActiveState=inactive\nSubState=dead\nMainPID=0\n"
            return subprocess.CompletedProcess(argv, 0, stdout, b"")
        if argv[0] == str(LAUNCHER.PGREP):
            return subprocess.CompletedProcess(argv, 1, b"", b"")
        if argv[0] == str(LAUNCHER.AMD_SMI) and argv[1] == "list":
            return subprocess.CompletedProcess(argv, 0, json.dumps([other, target]).encode(), b"")
        if argv[0] == str(LAUNCHER.ROCMINFO):
            stdout = b"Name:                    gfx1201\nUuid:                    GPU-a8e9ddefa2d60f55\nMarketing Name:          AMD Radeon Graphics\n"
            return subprocess.CompletedProcess(argv, 0, stdout, b"")
        if argv[0] == str(LAUNCHER.AMD_SMI) and argv[1] == "process":
            return subprocess.CompletedProcess(argv, 0, b'[{"gpu": 2, "process_list": []}]', b"")
        if argv[0] == str(LAUNCHER.AMD_SMI) and argv[1] == "static":
            return subprocess.CompletedProcess(argv, 0, b'{"gpu_data": [{"gpu": 2, "vram": {"size": {"value": 32624, "unit": "MB"}}}]}', b"")
        raise AssertionError(argv)

    return run


def test_collect_execute_gates_uses_order_independent_unique_gpu_mapping_and_no_owners() -> None:
    lock = {"path": str(LAUNCHER.LOCK_PATH), "free": True, "device": 66306, "inode": 123}
    gates = LAUNCHER.collect_execute_gates(run=_gate_router(), environment=dict(LAUNCHER.EXECUTE_ENV), kfd_owner_provider=lambda: [], lock_provider=lambda: lock)
    assert gates["passed"] is True
    assert gates["runtime_mapping"] == {"runtime_device_index": 1, "visible_token": "1", "amd_smi_index": 2, "bdf": LAUNCHER.GPU_BDF, "uuid": LAUNCHER.GPU_UUID, "kfd_id": LAUNCHER.KFD_ID, "node_id": 2}
    assert gates["amd_smi_owners"] == gates["kfd_owners"] == []
    assert gates["probes"][0]["argv"] == [str(LAUNCHER.SUDO), "-n", "-v"]


def test_collect_execute_gates_rejects_active_service_duplicate_mapping_and_kfd_owner() -> None:
    with pytest.raises(LAUNCHER.LauncherError, match="service is not inactive"):
        LAUNCHER.collect_execute_gates(run=_gate_router(active_service=True), environment=dict(LAUNCHER.EXECUTE_ENV), kfd_owner_provider=lambda: [], lock_provider=lambda: {})
    with pytest.raises(LAUNCHER.LauncherError, match="unique identity"):
        LAUNCHER.collect_execute_gates(run=_gate_router(duplicate_bdf=True), environment=dict(LAUNCHER.EXECUTE_ENV), kfd_owner_provider=lambda: [], lock_provider=lambda: {})
    with pytest.raises(LAUNCHER.LauncherError, match="KFD compute owners"):
        LAUNCHER.collect_execute_gates(run=_gate_router(), environment=dict(LAUNCHER.EXECUTE_ENV), kfd_owner_provider=lambda: [123], lock_provider=lambda: {})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda gates: gates["runtime_mapping"].update(unknown=1),
        lambda gates: gates["runtime_mapping"].update(node_id=3),
        lambda gates: gates["lock"].update(unknown=1),
        lambda gates: gates["lock"].update(inode=-1),
        lambda gates: gates["vram"].update(total_bytes=1, free_bytes=1, headroom_bytes=1),
        lambda gates: gates["vram"].update(used_bytes=1),
        lambda gates: gates["probes"][0].update(exit_code=1),
        lambda gates: gates["probes"][0].update(stdout_sha256="A" * 64),
        lambda gates: gates["probes"][1].update(label="sudo-n", argv=[str(LAUNCHER.SUDO), "-n", "-v"]),
        lambda gates: gates["probes"][0].update(unknown=1),
    ],
)
def test_launcher_rejects_qa_nested_schema_negatives_before_writing_sidecar(tmp_path: Path, mutate) -> None:
    binding, evidence_path, _, _ = _ready_binding(tmp_path)
    evidence_path.mkdir()
    gates = _gates()
    mutate(gates)
    with pytest.raises(LAUNCHER.LauncherError):
        LAUNCHER.make_live_preflight(binding, gates, evidence_path)
    assert not (evidence_path / "live-preflight.json").exists()
