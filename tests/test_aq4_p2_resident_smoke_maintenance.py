from __future__ import annotations

import importlib.util
import json
import os
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
        self.now = 0.0
        self.stopped_observation_count = 0

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
        if self.fail == "host-route" and url == HARNESS.GATEWAY_READY_URL:
            raise TimeoutError("synthetic host route mismatch")
        body = HARNESS.GATEWAY_READY_BODY if url == HARNESS.GATEWAY_READY_URL else HARNESS.OPENWEBUI_HEALTH_BODY
        return {"url": url, "status": 200, "body": body}

    def container_health(self, run) -> dict:
        if self.fail == "health" and self.epoch > 0:
            raise HARNESS.HarnessError("synthetic container health failure")
        container_curl = 5 if self.fail == "curl-count-off-by-one" else 6
        return {
            "transport": "docker-exec-container-network-namespace",
            "secret_material_recorded": False,
            "container": {"id": HARNESS.OPENWEBUI_CONTAINER_ID},
            "endpoints": {"gateway_models": {"model_id": HARNESS.GATEWAY_MODEL_ID}},
            "commands": [],
            "process_counts": {
                "docker": 9,
                "docker_exec": 6,
                "container_curl": container_curl,
                "container_curl_total": 6,
                "container_curl_version": 1,
                "container_curl_endpoint": 5,
            },
        }

    def stopped(self, old_worker_pid: int, old_service_pid: int, run) -> dict:
        attempt = self.stopped_observation_count
        self.stopped_observation_count += 1
        amd_smi_owners: list[int] = []
        kfd_owners: list[int] = []
        lock_free = True
        lock_holders: list[int] = []
        if self.fail == "delayed-release" and attempt == 0:
            amd_smi_owners = [old_worker_pid]
        elif self.fail == "never-release":
            amd_smi_owners = [old_worker_pid]
        elif self.fail in {"stopped-gate", "foreign-owner"}:
            amd_smi_owners = [999999]
        elif self.fail == "owner-reappearance" and attempt > 0:
            amd_smi_owners = [old_worker_pid]
        elif self.fail == "lock-delay" and attempt == 0:
            lock_free = False
            lock_holders = [old_service_pid]
        elif self.fail == "foreign-lock":
            lock_free = False
            lock_holders = [999998]
        elif self.fail == "kfd-delay" and attempt == 0:
            kfd_owners = [old_worker_pid]
        pids = sorted(set(amd_smi_owners) | set(kfd_owners) | set(lock_holders) | {old_worker_pid, old_service_pid})
        probe_raw = HARNESS.canonical({"attempt": attempt, "amd": amd_smi_owners, "kfd": kfd_owners})
        return {
            "captured_unix_ns": 1_000_000_000 + attempt,
            "services": [{"unit": "ullm-openai.service", "active_state": "inactive", "sub_state": "dead", "main_pid": 0}, {"unit": "llama-qwen35-udq4.service", "active_state": "inactive", "sub_state": "dead", "main_pid": 0}],
            "worker_pids": [],
            "amd_smi_owners": amd_smi_owners,
            "kfd_owners": kfd_owners,
            "lock": {"path": str(HARNESS.LAUNCHER.LOCK_PATH), "free": lock_free, "device": 1, "inode": 2, "holder_pids": lock_holders, "source_sha256": HARNESS.sha_bytes(HARNESS.canonical(lock_holders)), "source_bytes": len(lock_holders)},
            "vram": {"total_bytes": 32_000_000_000, "used_bytes": 0 if not amd_smi_owners and not kfd_owners else None, "free_bytes": 32_000_000_000 if not amd_smi_owners and not kfd_owners else None, "headroom_bytes": 32_000_000_000 if not amd_smi_owners and not kfd_owners else None},
            "proc_cmdlines": [{"pid": pid, "readable": True, "bytes": 0, "sha256": HARNESS.sha_bytes(b""), "argv0_basename": "redacted", "matches_expected_worker": pid == old_worker_pid, "raw_recorded": False} for pid in pids],
            "probes": [{"label": "fake-stopped-observation", "argv": ["fake-stopped-observation", str(attempt)], "exit_code": 0, "stdout_sha256": HARNESS.sha_bytes(probe_raw), "stderr_sha256": HARNESS.sha_bytes(b""), "captured_unix_ns": 1_000_000_000 + attempt}],
            "virtual_sources": {"kfd_owners": {"raw_sha256": HARNESS.sha_bytes(HARNESS.canonical(kfd_owners)), "parsed_pids": kfd_owners}, "lock_holders": {"raw_sha256": HARNESS.sha_bytes(HARNESS.canonical(lock_holders)), "raw_bytes": len(lock_holders), "parsed_pids": lock_holders}},
            "secret_material_recorded": False,
        }

    def sleep(self, seconds: float) -> None:
        self.now += seconds

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
        return HARNESS.Dependencies(self.run, self.http, self.container_health, self.stopped, lambda: self.active, self.owners, lambda root: HARNESS.PACKAGE_CONTENT_SHA, self.launch, self.profile_capture, self.profile_trust, self.sleep, lambda: self.now)


class FakeDockerRuntime:
    def __init__(self, *, fail: str | None = None) -> None:
        self.fail = fail
        self.commands: list[list[str]] = []
        self.inspect_count = 0
        self.secret_inputs: list[bytes] = []
        self.curl_version = b"curl 7.88.1 fake-build\n"

    def __call__(self, argv, **kwargs):
        self.commands.append(argv)
        if argv[1:4] == ["version", "--format", "{{json .Client}}"]:
            value = {"Version": HARNESS.DOCKER_CLIENT_VERSION, "ApiVersion": HARNESS.DOCKER_CLIENT_API_VERSION, "Os": "linux", "Arch": "amd64"}
            return subprocess.CompletedProcess(argv, 0, json.dumps(value).encode() + b"\n", b"")
        if argv[1:4] == ["inspect", "--type", "container"]:
            self.inspect_count += 1
            if self.fail == "container-absent":
                return subprocess.CompletedProcess(argv, 1, b"", b"absent")
            container_id = "f" * 64 if self.fail == "container-replaced" and self.inspect_count > 1 else HARNESS.OPENWEBUI_CONTAINER_ID
            image_id = "sha256:" + "d" * 64 if self.fail == "image-swap" else HARNESS.OPENWEBUI_IMAGE_ID
            health = "unhealthy" if self.fail == "health-state" else "healthy"
            network_id = "e" * 64 if self.fail == "network-swap" else HARNESS.OPENWEBUI_NETWORK_ID
            fields = [container_id, image_id, "/open-webui", "running", "true", health, network_id, HARNESS.OPENWEBUI_CONTAINER_IP, HARNESS.OPENWEBUI_GATEWAY_IP]
            return subprocess.CompletedProcess(argv, 0, ("|".join(fields) + "\n").encode(), b"")
        if argv[1] != "exec":
            raise AssertionError(argv)
        command_index = 4 if argv[2] == "-i" else 3
        assert argv[command_index - 1] == HARNESS.OPENWEBUI_CONTAINER_ID
        command = argv[command_index:]
        if command == [HARNESS.CONTAINER_CURL, "--version"]:
            return subprocess.CompletedProcess(argv, 0, self.curl_version, b"")
        if command == ["/usr/bin/sha256sum", HARNESS.CONTAINER_CURL]:
            return subprocess.CompletedProcess(argv, 0, f"{HARNESS.CONTAINER_CURL_SHA}  {HARNESS.CONTAINER_CURL}\n".encode(), b"")
        url = argv[-1]
        if url == HARNESS.GATEWAY_HEALTH_URL:
            body = HARNESS.GATEWAY_HEALTH_BODY
        elif url == HARNESS.GATEWAY_READY_URL:
            if self.fail == "curl-failure":
                return subprocess.CompletedProcess(argv, 1, b"", b"synthetic curl failure")
            body = HARNESS.GATEWAY_READY_BODY
        elif url == HARNESS.OPENWEBUI_CONTAINER_HEALTH_URL:
            body = HARNESS.OPENWEBUI_HEALTH_BODY
        elif url == HARNESS.GATEWAY_MODELS_URL:
            header = kwargs.get("input")
            assert isinstance(header, bytes)
            self.secret_inputs.append(header)
            if self.fail == "secret-echo":
                token = header.removeprefix(b"Authorization: Bearer ").rstrip(b"\r\n")
                return subprocess.CompletedProcess(argv, 0, token + b"\n__ULLM_HTTP_STATUS__200\n", b"")
            model_id = "wrong-model" if self.fail == "model-mismatch" else HARNESS.GATEWAY_MODEL_ID
            body = HARNESS.canonical({"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "ullm"}]})
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, 0, body + b"\n__ULLM_HTTP_STATUS__200\n", b"")


def container_guard_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fail: str | None = None):
    docker = tmp_path / "docker"
    docker.write_bytes(b"fake-docker-binary")
    docker.chmod(0o755)
    key = tmp_path / "openai-api-key"
    secret = b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    key.write_bytes(secret + b"\n")
    key.chmod(0o640)
    runtime = FakeDockerRuntime(fail=fail)
    monkeypatch.setattr(HARNESS, "DOCKER", docker)
    monkeypatch.setattr(HARNESS, "DOCKER_SHA", HARNESS.sha_bytes(docker.read_bytes()))
    monkeypatch.setattr(HARNESS, "API_KEY_FILE", key)
    monkeypatch.setattr(HARNESS, "API_KEY_UID", os.geteuid())
    monkeypatch.setattr(HARNESS, "API_KEY_GID", os.getegid())
    monkeypatch.setattr(HARNESS, "CONTAINER_CURL_VERSION_SHA", HARNESS.sha_bytes(runtime.curl_version))
    return HARNESS.ContainerHealthGuard(), runtime, secret


def container_health_expected_argv() -> list[list[str]]:
    inspect = [str(HARNESS.DOCKER), "inspect", "--type", "container", "--format", HARNESS.DOCKER_INSPECT_FORMAT, HARNESS.OPENWEBUI_CONTAINER_NAME]
    container_id = HARNESS.OPENWEBUI_CONTAINER_ID
    return [
        [str(HARNESS.DOCKER), "version", "--format", "{{json .Client}}"],
        inspect,
        [str(HARNESS.DOCKER), "exec", container_id, HARNESS.CONTAINER_CURL, "--version"],
        [str(HARNESS.DOCKER), "exec", container_id, "/usr/bin/sha256sum", HARNESS.CONTAINER_CURL],
        HARNESS._curl_command(container_id, HARNESS.GATEWAY_HEALTH_URL, authenticated=False),
        HARNESS._curl_command(container_id, HARNESS.GATEWAY_READY_URL, authenticated=False),
        HARNESS._curl_command(container_id, HARNESS.OPENWEBUI_CONTAINER_HEALTH_URL, authenticated=False),
        HARNESS._curl_command(container_id, HARNESS.GATEWAY_MODELS_URL, authenticated=True),
        inspect,
    ]


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
    assert value["maintenance"]["stopped_gate_poll"] == {
        "timeout_seconds": 30.0,
        "initial_interval_seconds": 0.25,
        "maximum_interval_seconds": 1.0,
        "required_consecutive_stable": 2,
        "sudo_keepalive_seconds": 10.0,
        "transitional_amd_kfd_owner": "pre_stop_worker_pid_only",
        "transitional_lock_holder": "pre_stop_service_main_pid_only",
        "foreign_new_or_reappeared_owner": "immediate_fail_closed",
        "evidence": "atomic_immutable_per_poll_secret_safe_digests_and_parsed_pids",
    }


def test_successful_fake_maintenance_stops_launches_and_restores(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), tmp_path / "maintenance", runtime.dependencies())
    assert code == 0 and evidence["status"] == "passed"
    assert evidence["sequence"] == ["sudo-prevalidate", "pre-stop-snapshot", "durable-marker", "service-stopped", "stopped-gates", "launcher", "service-start", "service-restored"]
    assert evidence["process_counts"] == {"sudo": 3, "sudo_keepalive": 0, "systemctl_stop": 1, "launcher": 1, "systemctl_start": 1, "capture_tool": 0, "rocprof": 0, "docker": 18, "docker_exec": 12, "container_curl": 12, "container_curl_total": 12, "container_curl_version": 2, "container_curl_endpoint": 10, "stopped_gate_polls": 2, "stopped_gate_probe_commands": 2}
    assert evidence["safety"] == {"service_touched": True, "service_stopped": True, "gpu_command_executed": True, "model_load_executed": True}
    assert evidence["restore"]["passed"] is True
    assert evidence["restore"]["post_start"]["service"]["main_pid"] != evidence["pre_stop"]["service"]["main_pid"]
    assert (tmp_path / "maintenance/maintenance-marker.json").stat().st_mode & 0o777 == 0o444
    assert evidence["secret_material_recorded"] is False


def _poll_documents(output: Path, evidence: dict) -> list[dict]:
    files = evidence["stopped_gate_poll"]["poll_files"]
    documents = []
    for item in files:
        path = output / item["name"]
        raw = path.read_bytes()
        assert HARNESS.sha_bytes(raw) == item["sha256"]
        assert path.stat().st_mode & 0o777 == 0o444
        documents.append(json.loads(raw))
    return documents


@pytest.mark.parametrize("failure", ("delayed-release", "lock-delay", "kfd-delay"))
def test_stopped_gate_delayed_pre_stop_resource_release_reaches_stable2(tmp_path: Path, failure: str) -> None:
    runtime = FakeRuntime(fail=failure)
    output = tmp_path / failure
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), output, runtime.dependencies())
    assert code == 0 and evidence["status"] == "passed"
    assert evidence["stopped_gate_poll"]["passed"] is True
    documents = _poll_documents(output, evidence)
    assert [item["decision"] for item in documents] == ["pending", "stable", "stable"]
    assert [item["consecutive_stable"] for item in documents] == [0, 1, 2]
    assert evidence["process_counts"]["launcher"] == 1


def test_stopped_gate_never_release_times_out_with_all_poll_evidence(tmp_path: Path) -> None:
    runtime = FakeRuntime(fail="never-release")
    output = tmp_path / "never-release"
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), output, runtime.dependencies())
    assert code == 1 and evidence["status"] == "failed"
    assert evidence["stopped_gate_poll"]["failure"]["kind"] == "timeout"
    documents = _poll_documents(output, evidence)
    assert len(documents) == evidence["stopped_gate_poll"]["poll_count"]
    assert documents[-1]["decision"] == "pending"
    assert evidence["process_counts"]["launcher"] == 0
    assert evidence["process_counts"]["sudo_keepalive"] >= 2
    assert any(command["label"].startswith("sudo-stopped-poll-keepalive-") for command in evidence["commands"])
    assert evidence["restore"]["passed"] is True


@pytest.mark.parametrize(("failure", "source"), (("foreign-owner", "amd_smi_owners"), ("foreign-lock", "lock")))
def test_stopped_gate_foreign_owner_fails_immediately(tmp_path: Path, failure: str, source: str) -> None:
    runtime = FakeRuntime(fail=failure)
    output = tmp_path / failure
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), output, runtime.dependencies())
    assert code == 1 and evidence["stopped_gate_poll"]["failure"]["kind"] == "terminal"
    documents = _poll_documents(output, evidence)
    assert len(documents) == 1
    assert documents[0]["source_classification"][source] in {"foreign_or_new", "foreign_or_unknown_holder"}
    assert evidence["process_counts"]["launcher"] == 0


def test_stopped_gate_owner_reappearance_after_zero_fails_immediately(tmp_path: Path) -> None:
    runtime = FakeRuntime(fail="owner-reappearance")
    output = tmp_path / "owner-reappearance"
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), output, runtime.dependencies())
    assert code == 1
    documents = _poll_documents(output, evidence)
    assert [item["decision"] for item in documents] == ["stable", "terminal_failure"]
    assert documents[-1]["source_classification"]["amd_smi_owners"] == "reappeared"
    assert evidence["process_counts"]["launcher"] == 0


def test_stopped_gate_requires_two_consecutive_stable_observations(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    output = tmp_path / "stable2"
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), output, runtime.dependencies())
    assert code == 0
    documents = _poll_documents(output, evidence)
    assert [item["decision"] for item in documents] == ["stable", "stable"]
    assert [item["consecutive_stable"] for item in documents] == [1, 2]


def test_default_stopped_observer_records_probe_sha_pids_vram_and_redacted_cmdline(monkeypatch: pytest.MonkeyPatch) -> None:
    old_worker_pid = 111
    old_service_pid = 222
    monkeypatch.setattr(HARNESS.LAUNCHER, "validate_amd_smi_tool", lambda: None)
    expected_sha = {HARNESS.LAUNCHER.SYSTEMCTL: HARNESS.LAUNCHER.SYSTEMCTL_SHA, HARNESS.LAUNCHER.PGREP: HARNESS.LAUNCHER.PGREP_SHA}
    monkeypatch.setattr(HARNESS.LAUNCHER, "sha_file", lambda path, label: (expected_sha[path], ()))
    monkeypatch.setattr(HARNESS.LAUNCHER, "_kfd_owners", lambda: [old_worker_pid])
    monkeypatch.setattr(HARNESS, "_poll_lock_observation", lambda: {"path": str(HARNESS.LAUNCHER.LOCK_PATH), "free": False, "device": 1, "inode": 2, "holder_pids": [old_service_pid], "source_sha256": "a" * 64, "source_bytes": 12})
    monkeypatch.setattr(HARNESS, "_safe_proc_cmdline", lambda pid: {"pid": pid, "readable": True, "bytes": 9, "sha256": "b" * 64, "argv0_basename": "worker", "matches_expected_worker": pid == old_worker_pid, "raw_recorded": False})

    def run(argv, **kwargs):
        if argv[:2] == [str(HARNESS.LAUNCHER.SYSTEMCTL), "show"]:
            return subprocess.CompletedProcess(argv, 0, b"ActiveState=inactive\nSubState=dead\nMainPID=0\n", b"")
        if argv[0] == str(HARNESS.LAUNCHER.PGREP):
            return subprocess.CompletedProcess(argv, 1, b"", b"")
        if argv[1:3] == ["process", "--gpu"]:
            value = [{"gpu": HARNESS.LAUNCHER.AMD_SMI_INDEX, "process_list": [{"process_info": {"pid": old_worker_pid}}]}]
            return subprocess.CompletedProcess(argv, 0, json.dumps(value).encode(), b"")
        if argv[1:3] == ["static", "--gpu"]:
            value = {"gpu_data": [{"gpu": HARNESS.LAUNCHER.AMD_SMI_INDEX, "vram": {"size": {"value": 32624, "unit": "MB"}}}]}
            return subprocess.CompletedProcess(argv, 0, json.dumps(value).encode(), b"")
        raise AssertionError(argv)

    value = HARNESS.StoppedGateObserver()(old_worker_pid, old_service_pid, run)
    assert value["amd_smi_owners"] == value["kfd_owners"] == [old_worker_pid]
    assert value["lock"]["holder_pids"] == [old_service_pid]
    assert value["vram"] == {"total_bytes": 32_624_000_000, "used_bytes": None, "free_bytes": None, "headroom_bytes": None}
    assert len(value["probes"]) == 5
    assert all(len(item["stdout_sha256"]) == 64 and len(item["stderr_sha256"]) == 64 for item in value["probes"])
    assert {item["pid"] for item in value["proc_cmdlines"]} == {old_worker_pid, old_service_pid}
    assert all(item["raw_recorded"] is False for item in value["proc_cmdlines"])


def test_host_route_mismatch_is_diagnostic_and_container_route_remains_formal(tmp_path: Path) -> None:
    runtime = FakeRuntime(fail="host-route")
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), tmp_path / "host-route", runtime.dependencies())
    assert code == 0 and evidence["status"] == "passed"
    for snapshot in (evidence["pre_stop"], evidence["restore"]["post_start"]):
        assert snapshot["health"]["formal"]["transport"] == "docker-exec-container-network-namespace"
        diagnostic = snapshot["health"]["host_route_diagnostics"]
        assert diagnostic["formal_gate"] is False
        assert diagnostic["probes"]["gateway_readyz"]["reachable"] is False
        assert diagnostic["probes"]["gateway_readyz"]["matches_formal_response"] is False


def test_container_health_rejects_legacy_curl_count_off_by_one(tmp_path: Path) -> None:
    runtime = FakeRuntime(fail="curl-count-off-by-one")
    with pytest.raises(HARNESS.HarnessError, match="container namespace health contract differs"):
        HARNESS.capture_running(runtime.dependencies())
    assert not any(call[0][-2:] == ["stop", HARNESS.SERVICE] for call in runtime.calls)


def test_container_health_guard_pins_identity_uses_id_exec_and_keeps_secret_off_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    guard, runtime, secret = container_guard_fixture(tmp_path, monkeypatch)
    before = guard(runtime)
    after = guard(runtime)
    assert before["container"] == after["container"]
    assert before["endpoints"]["gateway_models"]["model_id"] == HARNESS.GATEWAY_MODEL_ID
    assert before["process_counts"] == {
        "docker": 9,
        "docker_exec": 6,
        "container_curl": 6,
        "container_curl_total": 6,
        "container_curl_version": 1,
        "container_curl_endpoint": 5,
    }
    assert HARNESS._container_process_counts(before["commands"]) == before["process_counts"]
    assert before["secret_material_recorded"] is False
    expected_header = b"Authorization: Bearer " + secret + b"\n"
    assert runtime.secret_inputs == [expected_header, expected_header]
    expected_argv = container_health_expected_argv()
    assert runtime.commands == [*expected_argv, *expected_argv]
    serialized = HARNESS.canonical([before, after, runtime.commands])
    assert secret not in serialized and expected_header.rstrip(b"\n") not in serialized
    exec_commands = [command for command in runtime.commands if command[1] == "exec"]
    assert exec_commands and all(HARNESS.OPENWEBUI_CONTAINER_NAME not in command for command in exec_commands)
    assert all(HARNESS.OPENWEBUI_CONTAINER_ID in command for command in exec_commands)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("container-absent", "container/image/network identity differs"),
        ("container-replaced", "container/image/network identity differs"),
        ("image-swap", "container/image/network identity differs"),
        ("health-state", "container/image/network identity differs"),
        ("network-swap", "container/image/network identity differs"),
        ("curl-failure", "container health endpoint differs"),
        ("model-mismatch", "gateway model identity differs"),
        ("secret-echo", "secret material escaped"),
    ],
)
def test_container_health_guard_fails_closed_without_secret_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, message: str) -> None:
    guard, runtime, secret = container_guard_fixture(tmp_path, monkeypatch, fail=failure)
    with pytest.raises(HARNESS.HarnessError, match=message) as captured:
        guard(runtime)
    assert secret not in str(captured.value).encode()
    assert all(secret not in "\0".join(command).encode() for command in runtime.commands)


def test_container_health_guard_rejects_same_path_docker_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    guard, runtime, _ = container_guard_fixture(tmp_path, monkeypatch)
    assert guard(runtime)["secret_material_recorded"] is False
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replacement-docker")
    replacement.chmod(0o755)
    replacement.replace(HARNESS.DOCKER)
    with pytest.raises(HARNESS.LAUNCHER.LauncherError, match="replacement"):
        guard(runtime)


def test_container_health_guard_rejects_api_key_replacement_across_snapshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    guard, runtime, secret = container_guard_fixture(tmp_path, monkeypatch)
    assert guard(runtime)["secret_material_recorded"] is False
    replacement = tmp_path / "replacement-key"
    replacement.write_bytes(secret + b"\n")
    replacement.chmod(0o640)
    replacement.replace(HARNESS.API_KEY_FILE)
    with pytest.raises(HARNESS.HarnessError, match="API key changed across maintenance boundary") as captured:
        guard(runtime)
    assert secret not in str(captured.value).encode()


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
    if failure == "health":
        assert evidence["restore"]["passed"] is False
        assert evidence["restore"]["post_start"] is None
        assert runtime.active is True and runtime.epoch == 1


def test_dry_run_writes_process_zero_evidence_without_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value = ready(tmp_path)
    monkeypatch.setattr(HARNESS, "READY_PATH", tmp_path / "ready-binding.json")
    HARNESS.READY_PATH.write_text("{}\n")
    code, evidence = HARNESS.dry_run_ready(value, tmp_path / "dry")
    assert code == 0 and evidence["status"] == "passed"
    assert evidence["process_counts"] == {"sudo": 0, "sudo_keepalive": 0, "systemctl_stop": 0, "launcher": 0, "systemctl_start": 0, "rocprof": 0, "capture_tool": 0, "docker": 0, "docker_exec": 0, "container_curl": 0, "container_curl_total": 0, "container_curl_version": 0, "container_curl_endpoint": 0, "stopped_gate_polls": 0, "stopped_gate_probe_commands": 0}
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
    assert evidence["process_counts"] == {"launcher": 0, "sudo": 0, "sudo_keepalive": 0, "systemctl_start": 0, "systemctl_stop": 0, "rocprof": 0, "capture_tool": 0, "docker": 0, "docker_exec": 0, "container_curl": 0, "container_curl_total": 0, "container_curl_version": 0, "container_curl_endpoint": 0, "stopped_gate_polls": 0, "stopped_gate_probe_commands": 0}
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
    assert evidence["process_counts"] == {"launcher": 0, "sudo": 0, "sudo_keepalive": 0, "systemctl_start": 0, "systemctl_stop": 0, "rocprof": 0, "capture_tool": 0, "docker": 0, "docker_exec": 0, "container_curl": 0, "container_curl_total": 0, "container_curl_version": 0, "container_curl_endpoint": 0, "stopped_gate_polls": 0, "stopped_gate_probe_commands": 0}
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
