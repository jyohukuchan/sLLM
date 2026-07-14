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
        self.profile_captured = False
        self.trust_stages: list[str] = []

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

    def profile_capture(self, contract: dict) -> dict:
        self.profile_captured = True
        if self.fail == "capture-start":
            raise OSError("synthetic capture startup failure")
        timed_out = self.fail == "capture-timeout"
        remaining = [999999] if self.fail == "capture-child" else []
        launcher_failed = self.fail == "capture-launcher"
        command = contract["command"]
        return {
            "completed": subprocess.CompletedProcess(command, 1 if timed_out or remaining or launcher_failed else 0, b"", b""),
            "started": True,
            "timed_out": timed_out,
            "cleanup_passed": not remaining,
            "children_remaining": remaining,
            "rocprof_started": True,
            "launcher_started": True,
            "launcher_status": "failed" if timed_out or remaining or launcher_failed else "passed",
            "gpu_command_executed": "unknown" if timed_out or launcher_failed else True,
            "model_load_executed": "unknown" if timed_out or launcher_failed else True,
        }

    def profile_trust(self, contract: dict, stage: str) -> dict:
        self.trust_stages.append(stage)
        if self.fail == f"trust-{stage}":
            raise HARNESS.HarnessError(f"synthetic trust failure: {stage}")
        return {"stage": stage, "passed": True}

    def dependencies(self) -> HARNESS.Dependencies:
        return HARNESS.Dependencies(self.run, self.http, self.stopped, lambda: self.active, self.owners, lambda root: HARNESS.PACKAGE_CONTENT_SHA, self.launch, self.profile_capture, self.profile_trust, lambda seconds: None)


def ready(tmp_path: Path) -> dict:
    identity = {"path": str(SCRIPT), "commit": "1" * 40, "tree": "2" * 40, "git_blob": "3" * 40, "sha256": "4" * 64}
    value = HARNESS.ready_document(identity)
    value["launcher_binding"]["runner_output"] = str(tmp_path / "runner")
    value["launcher_binding"]["evidence_output"] = str(tmp_path / "launcher-evidence")
    value["launcher_binding"]["live_preflight"]["path"] = str(tmp_path / "launcher-evidence/live-preflight.json")
    return value


def profile_ready(tmp_path: Path) -> dict:
    identity = {"path": str(SCRIPT), "commit": "1" * 40, "tree": "2" * 40, "git_blob": "3" * 40, "sha256": "4" * 64}
    value = HARNESS.ready_document(identity, profile_diagnostic=True)
    value["launcher_binding"]["runner_output"] = str(tmp_path / "profile-runner")
    value["launcher_binding"]["evidence_output"] = str(tmp_path / "profile-launcher-evidence")
    value["launcher_binding"]["live_preflight"]["path"] = str(tmp_path / "profile-launcher-evidence/live-preflight.json")
    return value


def test_ready_document_fixes_one_case_live_policy_and_trust() -> None:
    value = HARNESS.ready_document({"path": str(SCRIPT), "commit": "1" * 40, "tree": "2" * 40, "git_blob": "3" * 40, "sha256": "4" * 64})
    assert value["status"] == "ready_for_one_case" and value["actual_eligible"] is True
    assert value["promotion_eligible"] is False
    assert value["execution_mode"] == "one_case"
    assert value["authorization"] == {"run_id": HARNESS.RUN_ID, "one_case_only": True, "maximum_invocations": 1, "output_no_reuse": True, "external_service_stop_required": True, "rocprof_wrapper_required": False}
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
    assert evidence["process_counts"] == {"sudo": 3, "systemctl_stop": 1, "launcher": 1, "systemctl_start": 1, "capture_tool": 0, "rocprof": 0}
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
    assert evidence["process_counts"] == {"sudo": 0, "systemctl_stop": 0, "launcher": 0, "systemctl_start": 0, "rocprof": 0, "capture_tool": 0}
    assert evidence["service_touched"] is False and evidence["gpu_command_executed"] is False


def test_output_reuse_rejected_before_any_fake_command(tmp_path: Path) -> None:
    runtime = FakeRuntime(); output = tmp_path / "existing"; output.mkdir()
    with pytest.raises(HARNESS.HarnessError, match="already exists"):
        HARNESS.execute_maintenance(ready(tmp_path), output, runtime.dependencies())
    assert runtime.calls == []


def test_profile_ready_exact_capture_wrapper_and_identity_binding() -> None:
    value = HARNESS.ready_document({"path": str(SCRIPT), "commit": "1" * 40, "tree": "2" * 40, "git_blob": "3" * 40, "sha256": "4" * 64}, profile_diagnostic=True)
    profile = value["profile_diagnostic"]
    command = profile["command"]
    assert value["execution_mode"] == "profile_diagnostic"
    assert value["measurement_eligible"] is False and value["promotion_eligible"] is False
    assert value["authorization"]["run_id"] == HARNESS.LAUNCHER.PROFILE_RUN_ID
    assert value["authorization"]["maximum_invocations"] == 1
    assert value["authorization"]["rocprof_wrapper_required"] is True
    assert command == HARNESS.profile_capture_command()
    assert "--runner-command" not in command
    assert command[command.index("--profiler-path") + 1] == str(HARNESS.PROFILE_PROFILER)
    assert command[command.index("--profiler-sha256") + 1] == HARNESS.PROFILE_PROFILER_SHA
    assert command[command.index("--target-command-manifest") + 1] == str(HARNESS.PROFILE_TARGET_COMMAND_MANIFEST)
    assert command[command.index("--target-command-manifest-sha256") + 1] == HARNESS.sha_bytes(HARNESS.pretty(HARNESS.profile_target_command_manifest()))
    assert profile["command_sha256"] == HARNESS.sha_bytes(HARNESS.canonical(command))
    assert profile["output"]["must_not_exist_before_capture"] is True
    assert profile["resident_evidence"]["run_id"] == HARNESS.LAUNCHER.PROFILE_RUN_ID
    assert profile["resident_evidence"]["resident_session_id_source"] == "resident_raw.resident.session_id"
    assert profile["resident_evidence"]["case_id"] == HARNESS.LAUNCHER.CASE_ID
    assert profile["roctx"]["roctx_library"]["resolved_path"] == str(HARNESS.LAUNCHER.ROCTX_LIBRARY_RESOLVED)
    assert profile["target_launcher"]["command"] == HARNESS.profile_launcher_command()
    assert profile["target_launcher"]["binding_sha256"] == HARNESS.sha_bytes(HARNESS.canonical(HARNESS.LAUNCHER.ready_profile_execute_binding()))
    target = HARNESS.profile_target_command_manifest()
    assert target["argv"] == HARNESS.profile_launcher_command()
    assert {item["argument_index"] for item in target["input_files"]} == {0, 1}
    assert {item["argument_index"] for item in target["output_paths"]} == {5, 7}
    assert profile["target_launcher"]["manifest"] == {
        "path": str(HARNESS.PROFILE_TARGET_COMMAND_MANIFEST),
        "sha256": HARNESS.sha_bytes(HARNESS.pretty(target)),
        "manifest_sha256": target["manifest_sha256"],
    }


def test_profile_fake_maintenance_captures_child_then_restores(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    code, evidence = HARNESS.execute_maintenance(profile_ready(tmp_path), tmp_path / "profile-maintenance", runtime.dependencies())
    assert code == 0 and evidence["status"] == "passed"
    assert runtime.profile_captured is True
    assert runtime.trust_stages == ["before-start", "capture-before", "capture-after", "finalize-before"]
    assert evidence["execution_mode"] == "profile_diagnostic"
    assert evidence["sequence"] == ["sudo-prevalidate", "pre-stop-snapshot", "durable-marker", "service-stopped", "stopped-gates", "profile-capture", "service-start", "service-restored"]
    assert evidence["process_counts"]["capture_tool"] == evidence["process_counts"]["rocprof"] == evidence["process_counts"]["launcher"] == 1
    assert evidence["restore"]["passed"] is True and runtime.active is True


@pytest.mark.parametrize("failure", ("capture-start", "capture-timeout", "capture-child", "capture-launcher"))
def test_profile_capture_failure_always_restores_outer_service(tmp_path: Path, failure: str) -> None:
    runtime = FakeRuntime(fail=failure)
    code, evidence = HARNESS.execute_maintenance(profile_ready(tmp_path), tmp_path / f"profile-{failure}", runtime.dependencies())
    assert code == 1 and evidence["status"] == "failed"
    assert runtime.profile_captured is True
    assert evidence["restore"]["attempted"] is True and evidence["restore"]["passed"] is True
    assert evidence["restore"]["post_start"]["service"]["main_pid"] != evidence["pre_stop"]["service"]["main_pid"]
    assert runtime.active is True and runtime.epoch == 1
    assert runtime.trust_stages == ["before-start", "capture-before", "capture-after", "finalize-before"]


@pytest.mark.parametrize("target", ("capture", "profiler", "launcher", "manifest"))
def test_profile_trust_rejects_same_path_hash_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    capture = tmp_path / "capture.py"; capture.write_bytes(b"capture-trusted")
    profiler = tmp_path / "rocprofv3"; profiler.write_bytes(b"profiler-trusted"); profiler.chmod(0o755)
    launcher = tmp_path / "launcher.py"; launcher.write_bytes(b"launcher-trusted")
    monkeypatch.setattr(HARNESS, "PROFILE_CAPTURE_TOOL", capture)
    monkeypatch.setattr(HARNESS, "PROFILE_CAPTURE_SHA", HARNESS.sha_bytes(capture.read_bytes()))
    monkeypatch.setattr(HARNESS, "PROFILE_PROFILER", profiler)
    monkeypatch.setattr(HARNESS, "PROFILE_PROFILER_SHA", HARNESS.sha_bytes(profiler.read_bytes()))
    monkeypatch.setattr(HARNESS, "LAUNCHER_PATH", launcher)
    monkeypatch.setattr(HARNESS, "LAUNCHER_SHA", HARNESS.sha_bytes(launcher.read_bytes()))
    manifest = tmp_path / "target-command-manifest.json"
    monkeypatch.setattr(HARNESS, "PROFILE_TARGET_COMMAND_MANIFEST", manifest)
    manifest.write_bytes(HARNESS.pretty(HARNESS.profile_target_command_manifest()))
    contract = HARNESS.ready_document({"path": str(SCRIPT), "commit": "1" * 40, "tree": "2" * 40, "git_blob": "3" * 40, "sha256": "4" * 64}, profile_diagnostic=True)["profile_diagnostic"]
    guard = HARNESS.ProfileTrustGuard()
    assert guard(contract, "before-start")["passed"] is True
    watched = {"capture": capture, "profiler": profiler, "launcher": launcher, "manifest": manifest}[target]
    replacement = tmp_path / f"{target}-replacement"; replacement.write_bytes(b"swapped-bytes")
    if target == "profiler":
        replacement.chmod(0o755)
    replacement.replace(watched)
    with pytest.raises(HARNESS.LAUNCHER.LauncherError, match="replacement"):
        guard(contract, "capture-before")


def test_profile_trust_rechecks_exact_command_at_each_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture = tmp_path / "capture.py"; capture.write_bytes(b"capture-trusted")
    profiler = tmp_path / "rocprofv3"; profiler.write_bytes(b"profiler-trusted"); profiler.chmod(0o755)
    launcher = tmp_path / "launcher.py"; launcher.write_bytes(b"launcher-trusted")
    monkeypatch.setattr(HARNESS, "PROFILE_CAPTURE_TOOL", capture)
    monkeypatch.setattr(HARNESS, "PROFILE_CAPTURE_SHA", HARNESS.sha_bytes(capture.read_bytes()))
    monkeypatch.setattr(HARNESS, "PROFILE_PROFILER", profiler)
    monkeypatch.setattr(HARNESS, "PROFILE_PROFILER_SHA", HARNESS.sha_bytes(profiler.read_bytes()))
    monkeypatch.setattr(HARNESS, "LAUNCHER_PATH", launcher)
    monkeypatch.setattr(HARNESS, "LAUNCHER_SHA", HARNESS.sha_bytes(launcher.read_bytes()))
    manifest = tmp_path / "target-command-manifest.json"
    monkeypatch.setattr(HARNESS, "PROFILE_TARGET_COMMAND_MANIFEST", manifest)
    manifest.write_bytes(HARNESS.pretty(HARNESS.profile_target_command_manifest()))
    contract = HARNESS.ready_document({"path": str(SCRIPT), "commit": "1" * 40, "tree": "2" * 40, "git_blob": "3" * 40, "sha256": "4" * 64}, profile_diagnostic=True)["profile_diagnostic"]
    guard = HARNESS.ProfileTrustGuard()
    assert guard(contract, "before-start")["passed"] is True
    contract["command"] = [*contract["command"], "--unexpected"]
    with pytest.raises(HARNESS.HarnessError, match="command manifest differs"):
        guard(contract, "capture-before")


@pytest.mark.parametrize("failure", ("trust-capture-after", "trust-finalize-before"))
def test_profile_late_trust_failure_occurs_with_outer_restore_done(tmp_path: Path, failure: str) -> None:
    runtime = FakeRuntime(fail=failure)
    code, evidence = HARNESS.execute_maintenance(profile_ready(tmp_path), tmp_path / failure, runtime.dependencies())
    assert code == 1 and evidence["status"] == "failed"
    assert evidence["restore"]["passed"] is True and evidence["restore"]["post_start"] is not None
    assert runtime.active is True and runtime.epoch == 1


def test_profile_capture_output_reuse_rejected_before_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(); existing = tmp_path / "existing-profile"; existing.mkdir()
    monkeypatch.setattr(HARNESS, "PROFILE_OUTPUT_DIRECTORY", existing)
    with pytest.raises(HARNESS.HarnessError, match="profile capture output already exists"):
        HARNESS.execute_maintenance(profile_ready(tmp_path), tmp_path / "maintenance", runtime.dependencies())
    assert runtime.calls == [] and runtime.profile_captured is False


def test_base_and_profile_mode_cannot_be_cross_invoked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    monkeypatch.setattr(HARNESS, "load_ready_artifact", lambda path: ready(tmp_path))
    code = HARNESS.main(["--profile-diagnostic", "--ready-artifact", str(HARNESS.PROFILE_READY_PATH), "--mode", "execute", "--confirm-one-case", "--evidence-output", str(tmp_path / "cross")], dependencies=runtime.dependencies())
    assert code == 1 and runtime.calls == []


def test_canonical_ready_artifact_readback_and_harness_pin() -> None:
    value = HARNESS.load_ready_artifact()
    assert value["status"] == "ready_for_one_case" and value["actual_eligible"] is True
    assert value["promotion_eligible"] is False
    harness = value["trust"]["harness"]
    assert harness["sha256"] == HARNESS.sha_bytes(SCRIPT.read_bytes())
    committed = subprocess.run(["git", "show", f'{harness["commit"]}:tools/run-aq4-p2-resident-smoke-maintenance.py'], cwd=ROOT, check=True, stdout=subprocess.PIPE)
    assert committed.stdout == SCRIPT.read_bytes()
    assert value["qa_attestation_sha256"] == HARNESS.sha_bytes(HARNESS.pretty(HARNESS.QA_ATTESTATION))


def test_canonical_dry_run_cli_has_zero_actual_processes(tmp_path: Path) -> None:
    output = tmp_path / "ready-dry-run"
    code = HARNESS.main(["--mode", "dry-run", "--evidence-output", str(output)])
    assert code == 0
    evidence = json.loads((output / "launcher-evidence.json").read_text())
    assert evidence["process_counts"] == {"launcher": 0, "sudo": 0, "systemctl_start": 0, "systemctl_stop": 0, "rocprof": 0, "capture_tool": 0}
    assert evidence["service_touched"] is False


def test_canonical_profile_ready_readback_and_dry_run_process_zero(tmp_path: Path) -> None:
    value = HARNESS.load_ready_artifact(HARNESS.PROFILE_READY_PATH)
    assert value["execution_mode"] == "profile_diagnostic"
    assert value["actual_eligible"] is True and value["measurement_eligible"] is False and value["promotion_eligible"] is False
    assert value["profile_diagnostic"]["command"] == HARNESS.profile_capture_command()
    output = tmp_path / "profile-ready-dry-run"
    code = HARNESS.main(["--mode", "dry-run", "--profile-diagnostic", "--ready-artifact", str(HARNESS.PROFILE_READY_PATH), "--evidence-output", str(output)])
    assert code == 0
    evidence = json.loads((output / "launcher-evidence.json").read_text())
    assert evidence["process_counts"] == {"launcher": 0, "sudo": 0, "systemctl_start": 0, "systemctl_stop": 0, "rocprof": 0, "capture_tool": 0}
    assert evidence["profile_diagnostic"]["capture_executed"] is False


@pytest.mark.parametrize("failure", ("sudo-pre", "sudo-stop", "stop", "stopped-gate", "sudo-restore", "start", "health"))
def test_actual_cli_with_each_fake_gate_fails_closed(tmp_path: Path, failure: str) -> None:
    runtime = FakeRuntime(fail=failure)
    output = tmp_path / f"cli-{failure}"
    code = HARNESS.main(["--mode", "execute", "--confirm-one-case", "--evidence-output", str(output)], dependencies=runtime.dependencies())
    assert code == 1
    evidence = json.loads((output / "launcher-evidence.json").read_text())
    assert evidence["status"] == "failed"
    assert evidence["secret_material_recorded"] is False


def test_actual_cli_requires_explicit_one_case_confirmation_without_commands(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    code = HARNESS.main(["--mode", "execute", "--evidence-output", str(tmp_path / "no-confirm")], dependencies=runtime.dependencies())
    assert code == 1 and runtime.calls == []
