from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/run-aq4-p2-resident-smoke-maintenance.py"
SPEC = importlib.util.spec_from_file_location("aq4_p2_maintenance", SCRIPT)
assert SPEC and SPEC.loader
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


class FakeRuntime:
    def __init__(self, *, fail: str | None = None, launcher_mode: str = "success") -> None:
        self.active = True
        self.epoch = 0
        self.fail = fail
        self.launcher_mode = launcher_mode
        self.calls: list[list[str]] = []

    @property
    def gateway_pid(self) -> int:
        return 1001 + self.epoch * 1000

    @property
    def worker_pid(self) -> int:
        return 1002 + self.epoch * 1000

    def run(self, argv, **kwargs):
        self.calls.append(argv)
        if argv == [str(HARNESS.LAUNCHER.SUDO), "-n", "-v"]:
            label = "sudo-pre" if sum(call == argv for call in self.calls) == 1 else "sudo-stop" if sum(call == argv for call in self.calls) == 2 else "sudo-restore"
            return subprocess.CompletedProcess(argv, 1 if self.fail == label else 0, b"", b"")
        if argv[:2] == [str(HARNESS.LAUNCHER.SYSTEMCTL), "show"]:
            if not self.active:
                return subprocess.CompletedProcess(argv, 0, b"ActiveState=inactive\nSubState=dead\nMainPID=0\nNRestarts=0\nControlGroup=/system.slice/ullm-openai.service\n", b"")
            raw = f"ActiveState=active\nSubState=running\nMainPID={self.gateway_pid}\nNRestarts=0\nControlGroup=/system.slice/ullm-openai.service\n".encode()
            return subprocess.CompletedProcess(argv, 0, raw, b"")
        if argv[0] == str(HARNESS.LAUNCHER.PGREP):
            return subprocess.CompletedProcess(argv, 0 if self.active else 1, f"{self.worker_pid}\n".encode() if self.active else b"", b"")
        if argv[:2] == [str(HARNESS.LAUNCHER.AMD_SMI), "list"]:
            value = [{"gpu": 2, "bdf": HARNESS.LAUNCHER.GPU_BDF, "uuid": HARNESS.LAUNCHER.GPU_UUID, "kfd_id": HARNESS.LAUNCHER.KFD_ID, "node_id": 2, "partition_id": 0}]
            return subprocess.CompletedProcess(argv, 0, json.dumps(value).encode(), b"")
        if argv[-2:] == ["stop", HARNESS.SERVICE]:
            if self.fail == "stop":
                return subprocess.CompletedProcess(argv, 1, b"", b"")
            self.active = False
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        if argv[-2:] == ["start", HARNESS.SERVICE]:
            if self.fail == "start":
                return subprocess.CompletedProcess(argv, 1, b"", b"")
            self.epoch += 1; self.active = True
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        raise AssertionError(argv)

    def http(self, url: str) -> dict:
        if self.fail == "health" and self.epoch > 0:
            return {"url": url, "status": 503, "body": b"not ready"}
        body = HARNESS.GATEWAY_READY_BODY if url == HARNESS.GATEWAY_READY_URL else HARNESS.OPENWEBUI_HEALTH_BODY
        return {"url": url, "status": 200, "body": body}

    def stopped(self) -> dict:
        if self.fail == "stopped-gate":
            raise HARNESS.HarnessError("synthetic stopped gate failure")
        return {
            "passed": True,
            "services": [{"unit": "ullm-openai.service", "active_state": "inactive", "sub_state": "dead", "main_pid": 0}, {"unit": "llama-qwen35-udq4.service", "active_state": "inactive", "sub_state": "dead", "main_pid": 0}],
            "old_worker_pids": [], "amd_smi_owners": [], "kfd_owners": [], "lock": {"free": True},
        }

    def owners(self, run, worker_pid: int) -> dict:
        if not self.active or worker_pid != self.worker_pid:
            raise HARNESS.HarnessError("fake owner differs")
        return {"amd_smi": [worker_pid], "kfd": [worker_pid]}

    def launch(self, binding: dict) -> tuple[int, dict]:
        if self.launcher_mode == "raise":
            raise OSError("synthetic launcher startup failure")
        if self.launcher_mode == "fail":
            return 1, {"status": "failed", "safety": {"gpu_command_executed": "unknown", "model_load_executed": "unknown"}, "failure": {"reason": "synthetic"}}
        return 0, {"status": "passed", "safety": {"gpu_command_executed": True, "model_load_executed": True}, "failure": None}

    def dependencies(self) -> HARNESS.Dependencies:
        return HARNESS.Dependencies(self.run, self.http, self.stopped, lambda: self.active, self.owners, lambda root: HARNESS.PACKAGE_CONTENT_SHA, self.launch, lambda seconds: None)


def ready(tmp_path: Path) -> dict:
    identity = {"path": str(SCRIPT), "commit": "1" * 40, "tree": "2" * 40, "git_blob": "3" * 40, "sha256": "4" * 64}
    value = HARNESS.ready_document(identity)
    value["launcher_binding"]["runner_output"] = str(tmp_path / "runner")
    value["launcher_binding"]["evidence_output"] = str(tmp_path / "launcher-evidence")
    value["launcher_binding"]["live_preflight"]["path"] = str(tmp_path / "launcher-evidence/live-preflight.json")
    return value


def test_ready_document_fixes_one_case_live_policy_and_trust() -> None:
    value = HARNESS.ready_document({"path": str(SCRIPT), "commit": "1" * 40, "tree": "2" * 40, "git_blob": "3" * 40, "sha256": "4" * 64})
    assert value["status"] == "ready_for_one_case" and value["actual_eligible"] is True
    assert value["promotion_eligible"] is False
    assert value["authorization"] == {"run_id": HARNESS.RUN_ID, "one_case_only": True, "maximum_invocations": 1, "output_no_reuse": True, "external_service_stop_required": True}
    assert value["live_preflight_policy"]["pre_execution_sha256"] is None
    assert value["live_preflight_policy"]["final_evidence_binding"]["path_and_sha256_required"] is True
    assert value["trust"]["launcher"]["commit"] == HARNESS.LAUNCHER_COMMIT
    assert value["trust"]["runner"]["commit"] == HARNESS.RUNNER_COMMIT
    assert value["trust"]["runner"]["cli_ancestor_commit"] == HARNESS.RUNNER_CLI_ANCESTOR


def test_successful_fake_maintenance_stops_launches_and_restores(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), tmp_path / "maintenance", runtime.dependencies())
    assert code == 0 and evidence["status"] == "passed"
    assert evidence["sequence"] == ["sudo-prevalidate", "pre-stop-snapshot", "durable-marker", "service-stopped", "stopped-gates", "launcher", "service-start", "service-restored"]
    assert evidence["process_counts"] == {"sudo": 3, "systemctl_stop": 1, "launcher": 1, "systemctl_start": 1}
    assert evidence["safety"] == {"service_touched": True, "service_stopped": True, "gpu_command_executed": True, "model_load_executed": True}
    assert evidence["restore"]["passed"] is True
    assert evidence["restore"]["post_start"]["service"]["main_pid"] != evidence["pre_stop"]["service"]["main_pid"]
    assert (tmp_path / "maintenance/maintenance-marker.json").stat().st_mode & 0o777 == 0o444
    assert evidence["secret_material_recorded"] is False


@pytest.mark.parametrize("launcher_mode", ("raise", "fail"))
def test_launcher_start_or_partial_failure_always_restores(tmp_path: Path, launcher_mode: str) -> None:
    runtime = FakeRuntime(launcher_mode=launcher_mode)
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), tmp_path / f"maintenance-{launcher_mode}", runtime.dependencies())
    assert code == 1 and evidence["status"] == "failed"
    assert evidence["failure"]["stage"] == "launcher"
    assert evidence["restore"]["attempted"] is True and evidence["restore"]["passed"] is True
    assert runtime.active is True and runtime.epoch == 1


@pytest.mark.parametrize(
    ("failure", "stop_count", "launcher_count", "restore_attempted"),
    [("sudo-pre", 0, 0, False), ("sudo-stop", 0, 0, False), ("stop", 1, 0, True), ("stopped-gate", 1, 0, True), ("sudo-restore", 1, 1, True), ("start", 1, 1, True), ("health", 1, 1, True)],
)
def test_each_fake_maintenance_gate_fails_closed(tmp_path: Path, failure: str, stop_count: int, launcher_count: int, restore_attempted: bool) -> None:
    runtime = FakeRuntime(fail=failure)
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), tmp_path / f"maintenance-{failure}", runtime.dependencies())
    assert code == 1 and evidence["status"] == "failed"
    assert evidence["process_counts"]["systemctl_stop"] == stop_count
    assert evidence["process_counts"]["launcher"] == launcher_count
    assert evidence["restore"]["attempted"] is restore_attempted


def test_dry_run_writes_process_zero_evidence_without_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value = ready(tmp_path)
    monkeypatch.setattr(HARNESS, "READY_PATH", tmp_path / "ready-binding.json")
    HARNESS.READY_PATH.write_text("{}\n")
    code, evidence = HARNESS.dry_run_ready(value, tmp_path / "dry")
    assert code == 0 and evidence["status"] == "passed"
    assert evidence["process_counts"] == {"sudo": 0, "systemctl_stop": 0, "launcher": 0, "systemctl_start": 0}
    assert evidence["service_touched"] is False and evidence["gpu_command_executed"] is False


def test_output_reuse_rejected_before_any_fake_command(tmp_path: Path) -> None:
    runtime = FakeRuntime(); output = tmp_path / "existing"; output.mkdir()
    with pytest.raises(HARNESS.HarnessError, match="already exists"):
        HARNESS.execute_maintenance(ready(tmp_path), output, runtime.dependencies())
    assert runtime.calls == []
