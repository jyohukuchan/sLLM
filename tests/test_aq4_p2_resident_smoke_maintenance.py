from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
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
    def __init__(
        self,
        *,
        fail: str | None = None,
        launcher_mode: str = "success",
        restore_health_failures: int = 0,
        hash_seconds: float = 0.0,
        metadata_mutation: str | None = None,
        pre_metadata_mutation: str | None = None,
        pre_nrestarts: int = 0,
        post_nrestarts: int = 0,
    ) -> None:
        self.active = True
        self.epoch = 0
        self.fail = fail
        self.launcher_mode = launcher_mode
        self.calls: list[list[str]] = []
        self.profile_captured = False
        self.trust_stages: list[str] = []
        self.now = 0.0
        self.stopped_observation_count = 0
        self.restore_health_failures = restore_health_failures
        self.restore_health_calls = 0
        self.hash_seconds = hash_seconds
        self.full_hash_count = 0
        self.metadata_scan_count = 0
        self.metadata_mutation = metadata_mutation
        self.pre_metadata_mutation = pre_metadata_mutation
        self.pre_nrestarts = pre_nrestarts
        self.post_nrestarts = post_nrestarts
        self.events: list[str] = []
        self.runner_started_count = 0

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
            nrestarts = self.pre_nrestarts if self.epoch == 0 else self.post_nrestarts
            raw = f"ActiveState=active\nSubState=running\nMainPID={self.gateway_pid}\nNRestarts={nrestarts}\nControlGroup=/system.slice/ullm-openai.service\n".encode()
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
        if self.epoch > 0:
            self.restore_health_calls += 1
        if self.fail == "restore-deadline-crossing" and self.epoch > 0:
            self.now += HARNESS.RESTORE_TIMEOUT_SECONDS + 1.0
        if self.fail == "health" and self.epoch > 0:
            raise HARNESS.HarnessError("synthetic container health failure")
        if self.epoch > 0 and self.restore_health_calls <= self.restore_health_failures:
            raise HARNESS.HarnessError("synthetic transient container health failure")
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

    def stopped(self, old_worker_pid: int, old_service_pid: int, run, control: HARNESS.StoppedPollControl) -> dict:
        attempt = self.stopped_observation_count
        self.stopped_observation_count += 1
        if self.fail == "slow-second-stable" and attempt == 1:
            self.now += HARNESS.STOP_POLL_TIMEOUT_SECONDS
        if self.fail == "deadline-crossing-probe":
            def blocked_probe(argv, **kwargs):
                self.now += HARNESS.STOP_POLL_TIMEOUT_SECONDS
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output=b"partial-out", stderr=b"partial-error")
            control.command(blocked_probe, ["fake-deadline-crossing-probe"], "fake-deadline-crossing-probe")
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

    def launch(self, binding: dict, *, profile_runner_executor=None) -> tuple[int, dict]:
        self.events.append("launcher")
        if self.launcher_mode == "raise":
            raise OSError("synthetic launcher startup failure")
        if self.launcher_mode == "fail":
            return 1, {"status": "failed", "safety": {"gpu_command_executed": "unknown", "model_load_executed": "unknown"}, "failure": {"reason": "synthetic", "runner_started": False}}
        if profile_runner_executor is not None:
            self.events.extend(["validator", "gates"])
            target = {
                "path": str(Path(binding["evidence_output"]) / "profile-runner-target-command-manifest.json"),
                "sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "identity": [1, 2, 3, 4, 1, 5, 6],
            }

            def mark_runner_started() -> None:
                self.runner_started_count += 1
                self.events.append("rocprof")

            outcome = profile_runner_executor(
                ["/fake/runner", "profile"],
                dict(HARNESS.LAUNCHER.EXECUTE_ENV),
                mark_runner_started,
                target,
            )
            completed = outcome["completed"]
            capture = outcome["profile_capture"]
            passed = completed.returncode == 0 and not completed.stderr and capture["status"] == "complete_diagnostic"
            return (0 if passed else 1), {
                "status": "passed" if passed else "failed",
                "sequence": ["validator", "pre-exec-gates", "profile-runner-target", "runner", "runner-complete"],
                "profile_runner_target": target,
                "profile_capture": capture,
                "profile_diagnostics": outcome["profile_diagnostics"],
                "safety": {
                    "gpu_command_executed": outcome["gpu_command_executed"],
                    "model_load_executed": outcome["model_load_executed"],
                },
                "failure": None if passed else {"reason": "synthetic profile runner failure", "runner_started": self.runner_started_count == 1},
            }
        return 0, {"status": "passed", "safety": {"gpu_command_executed": True, "model_load_executed": True}, "failure": None}

    def package_hash(self, root: Path) -> str:
        self.full_hash_count += 1
        self.now += self.hash_seconds
        return HARNESS.PACKAGE_CONTENT_SHA

    def package_metadata(self, root: Path) -> HARNESS.PackageTreeSnapshot:
        self.metadata_scan_count += 1
        entries = [
            HARNESS.PackageTreeEntry(".", 1, 10, 0o40755, 3, 4096, 100, 100),
            HARNESS.PackageTreeEntry("manifest.json", 1, 11, 0o100644, 1, 1024, 100, 100),
            HARNESS.PackageTreeEntry("weights", 1, 12, 0o40755, 2, 4096, 100, 100),
            HARNESS.PackageTreeEntry("weights/shard.gguf", 1, 13, 0o100644, 1, 7_200_000_000, 100, 100),
        ]
        mutation = self.metadata_mutation if self.epoch > 0 else self.pre_metadata_mutation
        if mutation is not None:
            index = 3
            if mutation == "added":
                entries.append(HARNESS.PackageTreeEntry("weights/new.gguf", 1, 14, 0o100644, 1, 1, 101, 101))
            elif mutation == "empty-directory":
                entries.append(HARNESS.PackageTreeEntry("unexpected-empty", 1, 15, 0o40755, 2, 4096, 101, 101))
            elif mutation == "special":
                entries.append(HARNESS.PackageTreeEntry("unexpected-fifo", 1, 16, 0o010644, 1, 0, 101, 101))
            elif mutation == "removed":
                entries.pop(index)
            elif mutation == "replaced":
                entries[index] = HARNESS.PackageTreeEntry("weights/shard.gguf", 1, 99, 0o100644, 1, 7_200_000_000, 100, 101)
            elif mutation == "content":
                entries[index] = HARNESS.PackageTreeEntry("weights/shard.gguf", 1, 13, 0o100644, 1, 7_200_000_001, 101, 101)
            elif mutation == "symlink":
                entries[index] = HARNESS.PackageTreeEntry("weights/shard.gguf", 1, 99, 0o120777, 1, 17, 101, 101)
            elif mutation == "directory":
                entries[2] = HARNESS.PackageTreeEntry("weights", 1, 12, 0o40750, 2, 4096, 101, 101)
            else:
                raise AssertionError(mutation)
        ordered = tuple(sorted(entries, key=lambda item: item.relative_path))
        digest = HARNESS.hashlib.sha256()
        for item in ordered:
            digest.update(HARNESS.canonical(item.evidence()))
            digest.update(b"\n")
        return HARNESS.PackageTreeSnapshot(
            entries=ordered,
            identity_sha256=digest.hexdigest(),
            entry_count=len(ordered),
            file_count=sum(HARNESS.stat.S_ISREG(item.mode) for item in ordered),
            directory_count=sum(HARNESS.stat.S_ISDIR(item.mode) for item in ordered),
            symlink_count=sum(HARNESS.stat.S_ISLNK(item.mode) for item in ordered),
            special_count=sum(not (HARNESS.stat.S_ISREG(item.mode) or HARNESS.stat.S_ISDIR(item.mode) or HARNESS.stat.S_ISLNK(item.mode)) for item in ordered),
            bytes=sum(item.size for item in ordered if HARNESS.stat.S_ISREG(item.mode)),
        )

    def profile_capture(self, request: dict) -> dict:
        self.profile_captured = True
        self.events.append("capture")
        if self.fail == "capture-start":
            raise OSError("synthetic capture startup failure")
        timed_out = self.fail == "capture-timeout"
        remaining = [999999] if self.fail == "capture-child" else []
        launcher_failed = self.fail == "capture-launcher"
        request["mark_runner_started"]()
        command = request["runner_argv"]
        complete = not timed_out and not remaining and not launcher_failed
        if complete:
            self.events.append("runner")
        return {
            "completed": subprocess.CompletedProcess(command, 0 if complete else 1, b"", b""),
            "keepalives": [],
            "keepalive_failed": False,
            "gpu_command_executed": True if complete else "unknown",
            "model_load_executed": True if complete else "unknown",
            "profile_capture": {
                "status": "complete_diagnostic" if complete else "failed",
                "runner_profiled": True,
                "validator_profiled": False,
                "gates_profiled": False,
                "capture_tool_invocations": 1,
                "rocprof_invocations": 1,
                "target_manifest_sha256": request["target_binding"]["sha256"],
                "target_manifest_semantic_sha256": request["target_binding"]["manifest_sha256"],
                "target_argv_sha256": HARNESS.sha_bytes(HARNESS.canonical(command)),
                "environment_sha256": HARNESS.sha_bytes(HARNESS.canonical(request["environment"])),
                "capture_stdout_sha256": HARNESS.sha_bytes(b""),
                "capture_stderr_sha256": HARNESS.sha_bytes(b""),
                "rocprof_started": True,
                "runner_start_known": complete,
                "runner_started": complete,
                "runner_completed": complete,
                "timed_out": timed_out,
                "cleanup_passed": not remaining,
                "children_state_known": True,
                "children_remaining": remaining,
            },
            "profile_diagnostics": {
                "schema_version": "ullm.aq4_p3_profile_executor_diagnostics.v1",
                "runner_finished": complete,
                "capture_artifact": None,
                "failure_evidence": None,
                "validation_error": None,
                "executor_exception": None,
            },
        }

    def profile_trust(self, contract: dict, stage: str) -> dict:
        self.trust_stages.append(stage)
        if self.fail == f"trust-{stage}":
            raise HARNESS.HarnessError(f"synthetic trust failure: {stage}")
        return {"stage": stage, "passed": True}

    def dependencies(self) -> HARNESS.Dependencies:
        return HARNESS.Dependencies(
            self.run,
            self.http,
            self.container_health,
            self.stopped,
            lambda: self.active,
            self.owners,
            self.package_hash,
            self.launch,
            self.profile_capture,
            self.profile_trust,
            self.sleep,
            lambda: int(self.now * 1_000_000_000),
            self.package_metadata,
        )


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
    fake_tree = FakeRuntime().package_metadata(HARNESS.PACKAGE_ROOT)
    value["trust"]["production"]["expected_package_integrity_identity_sha256"] = HARNESS.package_integrity_identity(HARNESS.PACKAGE_CONTENT_SHA, fake_tree)
    value["launcher_binding"]["runner_output"] = str(tmp_path / "runner")
    value["launcher_binding"]["evidence_output"] = str(tmp_path / "launcher-evidence")
    value["launcher_binding"]["live_preflight"]["path"] = str(tmp_path / "launcher-evidence/live-preflight.json")
    return value


def profile_ready(tmp_path: Path) -> dict:
    identity = {"path": str(SCRIPT), "commit": "1" * 40, "tree": "2" * 40, "git_blob": "3" * 40, "sha256": "4" * 64}
    value = HARNESS.ready_document(identity, profile_diagnostic=True)
    fake_tree = FakeRuntime().package_metadata(HARNESS.PACKAGE_ROOT)
    value["trust"]["production"]["expected_package_integrity_identity_sha256"] = HARNESS.package_integrity_identity(HARNESS.PACKAGE_CONTENT_SHA, fake_tree)
    value["launcher_binding"]["runner_output"] = str(tmp_path / "profile-runner")
    value["launcher_binding"]["evidence_output"] = str(tmp_path / "profile-launcher-evidence")
    value["launcher_binding"]["live_preflight"]["path"] = str(tmp_path / "profile-launcher-evidence/live-preflight.json")
    value["profile_diagnostic"]["output"]["directory"] = str(tmp_path / "profile-output")
    value["profile_diagnostic"]["output"]["artifact"] = str(tmp_path / "profile-output/capture-artifact.json")
    value["profile_diagnostic"]["resident_evidence"]["identity"] = str(tmp_path / "resident-evidence/identity.json")
    value["profile_diagnostic"]["resident_evidence"]["summary"] = str(tmp_path / "resident-evidence/resident-batch.summary.json")
    value["profile_diagnostic"]["resident_evidence"]["raw"] = str(tmp_path / "resident-evidence/case.raw.json")
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
        "maximum_probe_timeout_seconds": 2.0,
        "deadline_semantics": "fixed_absolute_monotonic_ns_checked_before_and_after_each_observation_and_probe",
        "transitional_amd_kfd_owner": "pre_stop_worker_pid_only",
        "transitional_lock_holder": "pre_stop_service_main_pid_only",
        "foreign_new_or_reappeared_owner": "immediate_fail_closed",
        "evidence": "atomic_immutable_per_poll_secret_safe_digests_and_parsed_pids",
        "kfd_scan": "bounded_enoent_rescan_and_fatal_non_enoent_source_diagnostics",
        "trusted_lock_substrate": {
            "directory": "/run/ullm",
            "owner": "homelab1",
            "directory_mode": "0750",
            "lock_mode": "0600",
            "create": "pinned_sudo_install_then_nonroot_o_excl_o_nofollow",
            "identity": "same_device_inode_from_stopped_poll_through_runner_and_cleanup",
            "cleanup": "same_inode_unlink_then_pinned_sudo_rmdir_before_unconditional_service_restore",
        },
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
    assert runtime.full_hash_count == evidence["package_integrity"]["full_hash_count"] == 1
    assert evidence["package_integrity"]["full_content"] == {
        "stage": "pre-stop-full-content-hash",
        "passed": True,
        "sha256": HARNESS.PACKAGE_CONTENT_SHA,
        "duration_ns": 0,
        "file_count": 2,
        "bytes": 7_200_001_024,
    }
    assert evidence["package_integrity"]["tree_identity"]["stage"] == "pre-stop-tree-metadata"
    assert evidence["restore"]["poll_count"] == 1
    assert evidence["restore"]["deadline_monotonic_ns"] - evidence["restore"]["started_monotonic_ns"] == 120_000_000_000
    assert evidence["restore"]["final_metadata_recheck"]["passed"] is True
    assert evidence["restore"]["post_start"]["service_epoch"] == {
        "restart_kind": "explicit_systemctl_stop_start",
        "main_pid_changed": True,
        "worker_pid_changed": True,
        "nrestarts_before": 0,
        "nrestarts_after": 0,
        "nrestarts_semantics": "explicit_stop_start_resets_automatic_restart_counter_to_zero",
        "nrestarts_reset_to_zero": True,
        "control_group_unchanged": True,
    }
    assert (tmp_path / "maintenance/maintenance-marker.json").stat().st_mode & 0o777 == 0o444
    assert evidence["secret_material_recorded"] is False


def test_profile_ready_v7_through_v11_are_sealed_historical_readback() -> None:
    base = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1"
    roots = [
        base / f"p2/resident-one-case-smoke-profile-{kind}-v{version}"
        for version in range(7, 12)
        for kind in ("ready", "ready-dry-run")
    ]
    for root in roots:
        completed = subprocess.run(
            ["sha256sum", "-c", "SHA256SUMS"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert completed.returncode == 0, (root, completed.stdout, completed.stderr)
    ready_root = roots[0]
    dry_root = roots[1]
    ready = json.loads((ready_root / "ready-binding.json").read_text())
    trust = json.loads((ready_root / "harness-trust.json").read_text())
    attestation = json.loads((ready_root / "qa-attestation.json").read_text())
    dry = json.loads((dry_root / "launcher-evidence.json").read_text())
    assert ready["status"] == "ready_for_one_case"
    assert ready["actual_eligible"] is True
    assert ready["execution_mode"] == "profile_diagnostic"
    assert ready["launcher_binding"]["runner_output"].endswith(
        "/resident-one-case-smoke-profile-execute-v7"
    )
    assert ready["launcher_binding"]["evidence_output"].endswith(
        "/resident-one-case-smoke-profile-execute-evidence-v7"
    )
    assert ready["profile_diagnostic"]["output"]["directory"].endswith(
        "/aq4-p3-diagnostic-rocprof-capture-v7"
    )
    assert ready["launcher_binding"]["run_id"] == "p2-r9700-resident-one-case-smoke-profile-diagnostic-v7"
    assert trust["commit"] == "3fc2b8cd6f6910fbebd3ff4728855d55bf2cbbd2"
    assert trust["ready_binding_sha256"] == HARNESS.sha_bytes((ready_root / "ready-binding.json").read_bytes())
    assert ready["qa_attestation_sha256"] == HARNESS.sha_bytes(HARNESS.pretty(attestation))
    assert dry["status"] == "passed" and dry["mode"] == "dry-run"
    assert all(count == 0 for count in dry["process_counts"].values())
    assert dry["service_touched"] is False and dry["gpu_command_executed"] is False


def test_actual_v12_exact_35_file_seal_and_capture_parser_authority() -> None:
    seal = HARNESS.actual_v12_seal()
    assert seal["commit"] == HARNESS.ACTUAL_V12_COMMIT == "44617f7fd46c39f71f04502b248739cc116fe095"
    assert seal["tree"] == HARNESS.ACTUAL_V12_TREE == "813c4ffc88fb58cf8764b91d3c80cea9ef351f0f"
    assert seal["member_count"] == len(seal["members"]) == 35
    assert seal["members_sha256"] == HARNESS.sha_bytes(HARNESS.canonical(seal["members"]))
    assert len(seal["root_sums_sha256"]) == len(HARNESS.ACTUAL_V12_ROOTS) == 6
    assert HARNESS.PROFILE_CAPTURE_COMMIT == "eb00cbd83b90d6fd8d519f6662ddea16d5f4438c"
    assert HARNESS.PROFILE_CAPTURE_TREE == "545511060d95a02d69f4164d35bb56d89c22ea59"
    assert HARNESS.PROFILE_CAPTURE_GIT_BLOB == "91f243ff5dcc0c36c63e471ac7c4581c74535a2f"
    assert HARNESS.PROFILE_CAPTURE_SHA == "e326fb5c9f5ff04290fe0c37cfd25ad7e1e37bd7f76b5d7a62002465b9965df4"
    capture_raw = HARNESS.PROFILE_CAPTURE_TOOL.read_bytes()
    capture_module = HARNESS._load_profile_capture_module(capture_raw)
    derivation = HARNESS.derive_actual_v12_generic_memcpy(capture_module)
    assert {key: derivation[key] for key in HARNESS.GENERIC_MEMCPY_EXPECTED_COVERAGE} == HARNESS.GENERIC_MEMCPY_EXPECTED_COVERAGE
    assert derivation["raw_trace_sha256"] == HARNESS.GENERIC_MEMCPY_RAW_SHA256


def test_generic_memcpy_exact_one_direction_join() -> None:
    value = HARNESS.derive_generic_memcpy_rows(
        [
            {"Function": "hipMemcpyAsync", "Correlation_Id": "1"},
            {"Function": "hipMemcpyAsync", "Correlation_Id": "2"},
        ],
        [{"Correlation_Id": "1", "Direction": "MEMORY_COPY_HOST_TO_DEVICE"}],
        [{"Correlation_Id": "2", "Kernel_Name": "__amd_rocclr_copyBuffer"}],
    )
    assert value["direction_counts"] == {"H2D": 1, "D2H": 0, "D2D": 1}
    assert value["memory_exact_one"] == value["kernel_copy_buffer_exact_one"] == 1
    assert value["missing"] == value["duplicate"] == value["overlap"] == 0


@pytest.mark.parametrize("failure", ("missing", "duplicate", "other_kernel", "overlap", "duplicate_hip"))
def test_generic_memcpy_direction_join_rejects_non_exact_coverage(failure: str) -> None:
    hip = [{"Function": "hipMemcpyAsync", "Correlation_Id": "1"}]
    memory: list[dict[str, str]] = []
    kernel: list[dict[str, str]] = []
    if failure == "duplicate":
        memory = [
            {"Correlation_Id": "1", "Direction": "MEMORY_COPY_HOST_TO_DEVICE"},
            {"Correlation_Id": "1", "Direction": "MEMORY_COPY_HOST_TO_DEVICE"},
        ]
    elif failure == "other_kernel":
        kernel = [{"Correlation_Id": "1", "Kernel_Name": "some_other_kernel"}]
    elif failure == "overlap":
        memory = [{"Correlation_Id": "1", "Direction": "MEMORY_COPY_DEVICE_TO_HOST"}]
        kernel = [{"Correlation_Id": "1", "Kernel_Name": "__amd_rocclr_copyBuffer"}]
    elif failure == "duplicate_hip":
        hip.append(dict(hip[0]))
        memory = [{"Correlation_Id": "1", "Direction": "MEMORY_COPY_HOST_TO_DEVICE"}]
    with pytest.raises(HARNESS.HarnessError, match="exact-one direction coverage differs"):
        HARNESS.derive_generic_memcpy_rows(hip, memory, kernel)


def test_offline_reassembly_generator_is_process_zero_and_self_validating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = tmp_path / "capture-v10"
    evidence = tmp_path / "maintenance-evidence-v11"
    monkeypatch.setattr(
        HARNESS,
        "_git_source_identity",
        lambda: {
            "path": str(SCRIPT),
            "commit": "1" * 40,
            "tree": "2" * 40,
            "git_blob": "3" * 40,
            "sha256": HARNESS.sha_bytes(SCRIPT.read_bytes()),
        },
    )
    before = HARNESS.actual_v12_seal()
    value = HARNESS.prepare_profile_offline_reassembly(capture, evidence)
    assert value == HARNESS.validate_profile_offline_reassembly(capture, evidence)
    assert value["status"] == "offline_reassembled_sealed"
    assert value["source_actual_seal"] == before == HARNESS.actual_v12_seal()
    assert value["raw_inputs"]["count"] == len(HARNESS.PROFILE_CAPTURE_RAW_NAMES) == 7
    assert value["output"]["capture_artifact"]["status"] == "complete_diagnostic"
    assert value["execution"] == {
        "offline_assemble_calls": 1,
        "workload_processes": 0,
        "rocprof_processes": 0,
        "gpu_commands": 0,
        "service_operations": 0,
        "operator_invocations": 0,
        "actual_invocations": 0,
        "model_loads": 0,
    }
    assert capture.lstat().st_mode & 0o777 == evidence.lstat().st_mode & 0o777 == 0o555
    assert HARNESS.HISTORICAL_PROFILE_READY_V15_ROOT.is_dir()
    assert HARNESS.ACTUAL_V12_ROOTS[4].is_dir()
    assert value == HARNESS.validate_profile_offline_reassembly(capture, evidence)

    evidence_path = evidence / "offline-reassembly.json"
    sums_path = evidence / "SHA256SUMS"
    os.chmod(evidence, 0o700)
    os.chmod(evidence_path, 0o600)
    os.chmod(sums_path, 0o600)
    tampered = json.loads(evidence_path.read_text())
    tampered["generic_memcpy_derivation"]["direction_counts"]["D2D"] += 1
    tampered["evidence_sha256"] = HARNESS._semantic_self_hash(tampered, "evidence_sha256")
    tampered_raw = HARNESS.pretty(tampered)
    evidence_path.write_bytes(tampered_raw)
    sums_path.write_bytes(
        f"{HARNESS.sha_bytes(tampered_raw)}  offline-reassembly.json\n".encode("ascii")
    )
    os.chmod(evidence_path, 0o444)
    os.chmod(sums_path, 0o444)
    os.chmod(evidence, 0o555)
    with pytest.raises(HARNESS.HarnessError, match="source/output seal differs"):
        HARNESS.validate_profile_offline_reassembly(capture, evidence)

    os.chmod(evidence, 0o700)
    os.chmod(evidence_path, 0o600)
    os.chmod(sums_path, 0o600)
    authority_tampered = json.loads(json.dumps(value))
    authority_tampered["generator"]["commit"] = "0" * 40
    authority_tampered["evidence_sha256"] = HARNESS._semantic_self_hash(
        authority_tampered,
        "evidence_sha256",
    )
    authority_raw = HARNESS.pretty(authority_tampered)
    evidence_path.write_bytes(authority_raw)
    sums_path.write_bytes(
        f"{HARNESS.sha_bytes(authority_raw)}  offline-reassembly.json\n".encode("ascii")
    )
    os.chmod(evidence_path, 0o444)
    os.chmod(sums_path, 0o444)
    os.chmod(evidence, 0o555)
    with pytest.raises(HARNESS.HarnessError, match="source/output seal differs"):
        HARNESS.validate_profile_offline_reassembly(capture, evidence)


def test_profile_v13_is_invalid_preoperator_and_future_outputs_are_poststate_independent() -> None:
    base = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1"
    assert HARNESS.PROFILE_READY_ROOT == base / "p2/resident-one-case-smoke-profile-ready-v16"
    assert HARNESS.HISTORICAL_PROFILE_READY_V15_ROOT == base / "p2/resident-one-case-smoke-profile-ready-v15"
    assert HARNESS.PROFILE_MAINTENANCE_EVIDENCE == base / "p2/resident-one-case-smoke-profile-maintenance-evidence-v11"
    assert HARNESS.PROFILE_OFFLINE_REASSEMBLY_EVIDENCE == base / "p2/resident-one-case-smoke-profile-maintenance-offline-reassembly-evidence-v11"
    assert HARNESS.PROFILE_DRY_RUN_EVIDENCE == base / "p2/resident-one-case-smoke-profile-ready-dry-run-v16"
    assert HARNESS.PROFILE_OUTPUT_DIRECTORY == base / "p3/aq4-p3-diagnostic-rocprof-capture-v10"
    assert HARNESS.PROFILE_ARTIFACT == HARNESS.PROFILE_OUTPUT_DIRECTORY / "capture-artifact.json"
    assert HARNESS.PROFILE_OFFLINE_REASSEMBLY_OUTPUT_DIRECTORY == base / "p3/aq4-p3-diagnostic-rocprof-capture-offline-reassembly-v11"
    assert not HARNESS.PROFILE_OUTPUT_DIRECTORY.exists()
    assert not HARNESS.PROFILE_OUTPUT_DIRECTORY.is_symlink()
    assert not HARNESS.PROFILE_MAINTENANCE_EVIDENCE.exists()
    assert not HARNESS.PROFILE_MAINTENANCE_EVIDENCE.is_symlink()
    invalid_preoperator_v13_roots = (
        base / "p2/resident-one-case-smoke-profile-ready-v13",
        base / "p2/resident-one-case-smoke-profile-ready-dry-run-v13",
    )
    actual_v11_roots = (
        base / "p2/resident-one-case-smoke-profile-maintenance-evidence-v9",
        base / "p2/resident-one-case-smoke-profile-operator-result-v11",
        base / "p2/resident-one-case-smoke-profile-actual-audit-v11",
    )
    historical_v9_roots = (
        base / "p2/resident-one-case-smoke-profile-maintenance-evidence-v8",
        base / "p2/resident-one-case-smoke-profile-execute-evidence-v8",
        base / "p2/resident-one-case-smoke-profile-execute-v8",
        base / "p3/aq4-p3-diagnostic-rocprof-capture-v8",
        base / "p2/resident-one-case-smoke-profile-operator-result-v9",
        base / "p2/resident-one-case-smoke-profile-actual-audit-v9",
    )
    historical_v8_roots = (
        base / "p2/resident-one-case-smoke-profile-maintenance-evidence-v7",
        base / "p2/resident-one-case-smoke-profile-execute-evidence-v7",
        base / "p2/resident-one-case-smoke-profile-execute-v7",
        base / "p3/aq4-p3-diagnostic-rocprof-capture-v7",
        base / "p2/resident-one-case-smoke-profile-operator-result-v8",
        base / "p2/resident-one-case-smoke-profile-actual-audit-v8",
    )
    historical_v6_roots = (
        base / "p2/resident-one-case-smoke-profile-ready-v6",
        base / "p2/resident-one-case-smoke-profile-ready-dry-run-v6",
        base / "p2/resident-one-case-smoke-profile-execute-v6",
        base / "p2/resident-one-case-smoke-profile-execute-evidence-v6",
        base / "p2/resident-one-case-smoke-profile-maintenance-evidence-v6",
        base / "p3/aq4-p3-diagnostic-rocprof-capture-v6",
        base / "p2/resident-one-case-smoke-profile-operator-result-v7",
        base / "p2/resident-one-case-smoke-profile-actual-audit-v7",
    )
    historical_v5_roots = (
        base / "p2/resident-one-case-smoke-profile-execute-evidence-v5",
        base / "p2/resident-one-case-smoke-profile-maintenance-evidence-v5",
        base / "p3/aq4-p3-diagnostic-rocprof-capture-v5",
        base / "p2/resident-one-case-smoke-profile-operator-result-v6",
        base / "p2/resident-one-case-smoke-profile-actual-audit-v6",
    )
    assert not (base / "p2/resident-one-case-smoke-profile-execute-v5").exists()
    for root in invalid_preoperator_v13_roots + actual_v11_roots + historical_v9_roots + historical_v8_roots + historical_v6_roots + historical_v5_roots:
        completed = subprocess.run(
            ["sha256sum", "-c", "SHA256SUMS"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert completed.returncode == 0, (root, completed.stdout, completed.stderr)
    invalid_preoperator_v13_commit = "5f67d7edf9ea6285b6b5c01445b3dadbca65d562"
    committed_tree = subprocess.run(
        ["git", "rev-parse", f"{invalid_preoperator_v13_commit}^{{tree}}"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert committed_tree.returncode == 0 and not committed_tree.stderr
    assert committed_tree.stdout.strip() == "6c01686cfa456ce17b34646627682b3afe8d59d1"
    for root in invalid_preoperator_v13_roots:
        for path in root.iterdir():
            relative = str(path.relative_to(ROOT))
            committed = subprocess.run(
                ["git", "rev-parse", f"{invalid_preoperator_v13_commit}:{relative}"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            observed = subprocess.run(
                ["git", "hash-object", str(path)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            assert committed.returncode == observed.returncode == 0
            assert not committed.stderr and not observed.stderr
            assert committed.stdout.strip() == observed.stdout.strip()
    assert HARNESS.sha_bytes((invalid_preoperator_v13_roots[0] / "ready-binding.json").read_bytes()) == "d919d4addbda6338e7869ac185eeb47634e1da9d76793b5127357b638f31ec22"
    assert HARNESS.sha_bytes((invalid_preoperator_v13_roots[0] / "SHA256SUMS").read_bytes()) == "2ad6093cae677b897a868918bfb68b98ae299016c150166b2c65ab15641a4f74"
    assert HARNESS.sha_bytes((invalid_preoperator_v13_roots[1] / "launcher-evidence.json").read_bytes()) == "09012bb0a8e2c3f879718e560798fa5475473986729d205b07f9d1b29fc1cf92"
    assert HARNESS.sha_bytes((invalid_preoperator_v13_roots[1] / "SHA256SUMS").read_bytes()) == "44d6e4bd039b98c20915b29096888ea1e2e7c95356c23620a6ab55aa16c20de1"
    ready_v13 = json.loads((invalid_preoperator_v13_roots[0] / "ready-binding.json").read_text())
    trust_v13 = json.loads((invalid_preoperator_v13_roots[0] / "harness-trust.json").read_text())
    qa_v13 = json.loads((invalid_preoperator_v13_roots[0] / "qa-attestation.json").read_text())
    dry_v13 = json.loads((invalid_preoperator_v13_roots[1] / "launcher-evidence.json").read_text())
    assert ready_v13["status"] == "ready_for_one_case"
    assert ready_v13["actual_eligible"] is True
    assert ready_v13["execution_mode"] == "profile_diagnostic"
    assert ready_v13["authorization"]["maximum_invocations"] == 1
    assert ready_v13["launcher_binding"]["runner_output"] == str(
        base / "p2/resident-one-case-smoke-profile-execute-v9"
    )
    assert ready_v13["launcher_binding"]["evidence_output"] == str(
        base / "p2/resident-one-case-smoke-profile-execute-evidence-v9"
    )
    assert ready_v13["profile_diagnostic"]["output"]["directory"] == str(
        base / "p3/aq4-p3-diagnostic-rocprof-capture-v9"
    )
    assert ready_v13["trust"]["harness"] == {
        "commit": "576ab7d30f04742f4d48a200beb2e905b6ff83a9",
        "git_blob": "e177fc8e95a051c3d9370b7cec0729ab4c89dc2d",
        "path": str(SCRIPT),
        "sha256": "6c5a49e82ea4f00163bce9d7edbfaf511ed3a78e3bade98b194234ee9cbb8187",
        "tree": "f00a9380a901f63fde70fd6a647c334ba3250f1e",
    }
    assert trust_v13["commit"] == ready_v13["trust"]["harness"]["commit"]
    assert trust_v13["ready_binding_sha256"] == HARNESS.sha_bytes((invalid_preoperator_v13_roots[0] / "ready-binding.json").read_bytes())
    assert qa_v13["automated_tests"]["aggregate"] == {
        "collected": 639,
        "deselected": 0,
        "distinct_test_file_count": 12,
        "failed": 0,
        "passed": 639,
    }
    assert dry_v13["status"] == "passed" and dry_v13["mode"] == "dry-run"
    assert dry_v13["ready_binding_sha256"] == trust_v13["ready_binding_sha256"]
    assert set(dry_v13["process_counts"].values()) == {0}
    assert dry_v13["service_touched"] is False
    assert dry_v13["gpu_command_executed"] is False
    assert dry_v13["model_load_executed"] is False
    assert dry_v13["profile_diagnostic"]["capture_executed"] is False
    actual_v11_commit = "854e5a348bd3c0f442f2371a0d3619308bce3b95"
    for root in actual_v11_roots:
        for path in (root / "SHA256SUMS", *(item for item in root.iterdir() if item.name != "SHA256SUMS")):
            relative = str(path.relative_to(ROOT))
            committed = subprocess.run(
                ["git", "rev-parse", f"{actual_v11_commit}:{relative}"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            observed = subprocess.run(
                ["git", "hash-object", str(path)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            assert committed.returncode == observed.returncode == 0
            assert not committed.stderr and not observed.stderr
            assert committed.stdout.strip() == observed.stdout.strip()
    actual_v11_maintenance = json.loads((actual_v11_roots[0] / "launcher-evidence.json").read_text())
    actual_v11_result = json.loads((actual_v11_roots[1] / "operator-result.json").read_text())
    actual_v11_audit = json.loads((actual_v11_roots[2] / "actual-audit.json").read_text())
    assert actual_v11_maintenance["status"] == "failed"
    assert actual_v11_maintenance["failure"] == {
        "launcher_started": False,
        "reason": "restored worker does not uniquely own target GPU",
        "stage": "pre-stop-snapshot",
    }
    assert actual_v11_maintenance["process_counts"]["systemctl_stop"] == 0
    assert actual_v11_maintenance["process_counts"]["systemctl_start"] == 0
    assert all(actual_v11_maintenance["process_counts"][name] == 0 for name in ("launcher", "capture_tool", "rocprof"))
    assert actual_v11_maintenance["safety"]["service_touched"] is False
    assert actual_v11_result["returncode"] == 1
    assert actual_v11_result["invocation_count"] == actual_v11_result["maximum_invocations"] == 1
    assert actual_v11_result["retry_performed"] is False
    assert actual_v11_audit["status"] == "failed_immutable_evidence_preserved_restore_passed"
    assert actual_v11_audit["restore_classification"] == "pre_stop_untouched_same_epoch"
    assert actual_v11_audit["profile_artifacts"]["status"] == "failure_evidence_only"
    assert all(actual_v11_audit["evidence"][name] is None for name in ("execute", "runtime", "capture"))
    assert json.loads((historical_v9_roots[0] / "launcher-evidence.json").read_text())["status"] == "failed"
    assert json.loads((historical_v9_roots[1] / "launcher-evidence.json").read_text())["status"] == "failed"
    assert json.loads((historical_v9_roots[3] / "capture-failure.json").read_text())["status"] == "failed"
    assert json.loads((historical_v9_roots[4] / "operator-result.json").read_text())["status"] == "failed"
    assert json.loads((historical_v9_roots[5] / "actual-audit.json").read_text())["status"] == "failed_immutable_evidence_preserved_restore_passed"
    assert json.loads((historical_v8_roots[0] / "launcher-evidence.json").read_text())["status"] == "failed"
    assert json.loads((historical_v8_roots[1] / "launcher-evidence.json").read_text())["status"] == "failed"
    assert json.loads((historical_v8_roots[3] / "capture-failure.json").read_text())["status"] == "failed"
    assert json.loads((historical_v8_roots[4] / "operator-result.json").read_text())["status"] == "failed"
    assert json.loads((historical_v8_roots[5] / "actual-audit.json").read_text())["status"] == "failed_immutable_evidence_preserved_restore_passed"
    assert json.loads((historical_v6_roots[3] / "launcher-evidence.json").read_text())["status"] == "failed"
    assert json.loads((historical_v6_roots[4] / "launcher-evidence.json").read_text())["status"] == "failed"
    assert json.loads((historical_v6_roots[5] / "capture-failure.json").read_text())["schema_version"] == HARNESS.PROFILE_CAPTURE_FAILURE_SCHEMA
    assert json.loads((historical_v6_roots[6] / "operator-result.json").read_text())["status"] == "failed"
    assert json.loads((historical_v6_roots[7] / "actual-audit.json").read_text())["status"] == "failed_immutable_evidence_preserved_restore_passed"
    assert json.loads((historical_v5_roots[0] / "launcher-evidence.json").read_text())["status"] == "failed"
    assert json.loads((historical_v5_roots[1] / "launcher-evidence.json").read_text())["status"] == "failed"
    assert json.loads((historical_v5_roots[2] / "capture-failure.json").read_text())["schema_version"] == HARNESS.PROFILE_CAPTURE_FAILURE_SCHEMA
    assert json.loads((historical_v5_roots[3] / "operator-result.json").read_text())["status"] == "failed"
    assert json.loads((historical_v5_roots[4] / "actual-audit.json").read_text())["status"] == "failed"
    assert (
        base
        / "p2/resident-one-case-smoke-profile-execute-v3/resident-batch.failure.json"
    ).is_file()
    assert (
        base / "p3/aq4-p3-diagnostic-rocprof-capture-v3/capture-failure.json"
    ).is_file()
    historical_output = base / "p3/aq4-p3-diagnostic-rocprof-capture-v4"
    assert historical_output.is_dir()
    failure_path = historical_output / HARNESS.PROFILE_CAPTURE_FAILURE_NAME
    target_path = (
        base
        / "p2/resident-one-case-smoke-profile-execute-evidence-v4/runner-target-command-manifest.json"
    )
    target_raw = target_path.read_bytes()
    target = json.loads(target_raw)
    target_binding = {"path": str(target_path), "sha256": HARNESS.sha_bytes(target_raw)}
    contract = HARNESS.ready_document(
        {
            "path": str(SCRIPT), "commit": "1" * 40, "tree": "2" * 40,
            "git_blob": "3" * 40, "sha256": "4" * 64,
        },
        profile_diagnostic=True,
    )["profile_diagnostic"]
    contract["output"] = {
        "directory": str(historical_output),
        "name": HARNESS.PROFILE_OUTPUT_NAME,
        "artifact": str(historical_output / "capture-artifact.json"),
        "must_not_exist_before_capture": True,
    }
    expected_command = HARNESS._expected_profile_command(target["argv"], contract)
    with pytest.raises(HARNESS.HarnessError, match="semantic binding differs"):
        HARNESS._validate_profile_failure_evidence(
            failure_path,
            historical_output,
            target_binding,
            expected_command,
        )
    historical = HARNESS._read_historical_profile_failure_evidence(
        failure_path,
        historical_output,
        target_binding,
        expected_command,
    )
    assert historical["schema_version"] == HARNESS.HISTORICAL_PROFILE_CAPTURE_FAILURE_SCHEMA
    assert historical["historical_readback"] is True
    assert historical["ready_candidate_audit"] is None
    assert historical["process_group_cleanup_complete"] is True
    assert historical["children_state_known"] is True
    assert historical["children_remaining"] == []
    assert historical["sha256"] == "58619cb05c13cac5fed392d587c7d9878a53bba6ed02ace15e1c37d5969e99c5"
    value = HARNESS.ready_document({"path": str(SCRIPT), "commit": "1" * 40, "tree": "2" * 40, "git_blob": "3" * 40, "sha256": "4" * 64})
    assert value["trust"]["production"]["expected_package_integrity_identity_sha256"] == HARNESS.PACKAGE_INTEGRITY_IDENTITY_SHA


def test_offline_reassembly_poststate_does_not_change_canonical_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {
        "path": str(SCRIPT),
        "commit": "1" * 40,
        "tree": "2" * 40,
        "git_blob": "3" * 40,
        "sha256": "4" * 64,
    }
    before = HARNESS.ready_document(identity, profile_diagnostic=True)
    offline = tmp_path / "aq4-p3-diagnostic-rocprof-capture-offline-reassembly-v11"
    monkeypatch.setattr(HARNESS, "PROFILE_OFFLINE_REASSEMBLY_OUTPUT_DIRECTORY", offline)
    offline.mkdir()
    (offline / "capture-artifact.json").write_text("{}\n", encoding="ascii")
    after = HARNESS.ready_document(identity, profile_diagnostic=True)
    assert after == before
    assert after["actual_eligible"] is True
    assert after["profile_diagnostic"]["output"] == {
        "directory": str(HARNESS.PROFILE_OUTPUT_DIRECTORY),
        "name": HARNESS.PROFILE_OUTPUT_NAME,
        "artifact": str(HARNESS.PROFILE_ARTIFACT),
        "must_not_exist_before_capture": True,
    }
    assert str(offline) not in json.dumps(after, sort_keys=True)


def test_historical_ready_v15_loader_is_final_state_independent() -> None:
    trust = json.loads(HARNESS.HISTORICAL_PROFILE_HARNESS_TRUST_V15_PATH.read_text())
    assert subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip() != trust["commit"]
    assert HARNESS.ACTUAL_V12_ROOTS[4].is_dir()
    value = HARNESS.load_ready_artifact(HARNESS.HISTORICAL_PROFILE_READY_V15_PATH)
    assert value == json.loads(HARNESS.HISTORICAL_PROFILE_READY_V15_PATH.read_text())
    assert value["actual_eligible"] is True


@pytest.mark.parametrize("tamper", ("trust", "qa", "source"))
def test_historical_ready_v15_rejects_resealed_authority_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    root = tmp_path / "profile-ready-v15"
    shutil.copytree(HARNESS.HISTORICAL_PROFILE_READY_V15_ROOT, root)
    os.chmod(root, 0o700)
    for path in root.iterdir():
        os.chmod(path, 0o600)
    ready_path = root / "ready-binding.json"
    trust_path = root / "harness-trust.json"
    attestation_path = root / "qa-attestation.json"
    ready = json.loads(ready_path.read_text())
    trust = json.loads(trust_path.read_text())
    attestation = json.loads(attestation_path.read_text())
    if tamper == "trust":
        trust["tree"] = "0" * 40
    elif tamper == "source":
        trust["sha256"] = "0" * 64
    else:
        attestation["manual_checks"]["boundary_count"] += 1
        attestation_raw = HARNESS.pretty(attestation)
        ready["qa_attestation_sha256"] = HARNESS.sha_bytes(attestation_raw)
        ready_raw = HARNESS.pretty(ready)
        trust["ready_binding_sha256"] = HARNESS.sha_bytes(ready_raw)
        ready_path.write_bytes(ready_raw)
        attestation_path.write_bytes(attestation_raw)
    trust_path.write_bytes(HARNESS.pretty(trust))
    members = [
        ("harness-trust.json", trust_path.read_bytes()),
        ("qa-attestation.json", attestation_path.read_bytes()),
        ("ready-binding.json", ready_path.read_bytes()),
    ]
    (root / "SHA256SUMS").write_bytes(
        "".join(
            f"{HARNESS.sha_bytes(raw)}  {name}\n"
            for name, raw in members
        ).encode("ascii")
    )
    for path in root.iterdir():
        os.chmod(path, 0o444)
    os.chmod(root, 0o555)
    monkeypatch.setattr(HARNESS, "HISTORICAL_PROFILE_READY_V15_ROOT", root)
    monkeypatch.setattr(HARNESS, "HISTORICAL_PROFILE_READY_V15_PATH", ready_path)
    monkeypatch.setattr(HARNESS, "HISTORICAL_PROFILE_HARNESS_TRUST_V15_PATH", trust_path)
    monkeypatch.setattr(HARNESS, "HISTORICAL_PROFILE_ATTESTATION_V15_PATH", attestation_path)
    with pytest.raises(HARNESS.HarnessError, match="ready (historical|artifact)"):
        HARNESS.load_ready_artifact(ready_path)


def test_offline_source_authority_uses_path_last_change_not_current_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit = "1" * 40
    source_tree = "2" * 40
    source_blob = "3" * 40
    calls: list[list[str]] = []

    def git_stdout(argv: list[str], _label: str) -> str:
        calls.append(argv)
        if argv[:4] == ["log", "-1", "--format=%H", "--"]:
            return source_commit
        if argv == ["rev-parse", f"{source_commit}^{{tree}}"]:
            return source_tree
        if argv[0:2] == ["rev-parse", f"{source_commit}:tools/run-aq4-p2-resident-smoke-maintenance.py"]:
            return source_blob
        if argv[0] == "hash-object":
            return source_blob
        raise AssertionError(f"unexpected Git lookup: {argv}")

    monkeypatch.setattr(HARNESS, "_git_stdout", git_stdout)
    monkeypatch.setattr(HARNESS, "_git_bytes", lambda _argv, _label: SCRIPT.read_bytes())
    identity = HARNESS._git_source_identity()
    assert identity == {
        "path": str(SCRIPT),
        "commit": source_commit,
        "tree": source_tree,
        "git_blob": source_blob,
        "sha256": HARNESS.sha_bytes(SCRIPT.read_bytes()),
    }
    assert all("HEAD" not in argument for argv in calls for argument in argv)

    monkeypatch.setattr(HARNESS, "_git_bytes", lambda _argv, _label: b"tampered-source")
    with pytest.raises(HARNESS.HarnessError, match="differs from last-change Git authority"):
        HARNESS._git_source_identity()


def test_qa_manifest_strictly_resolves_all_source_commit_path_blobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    HARNESS.validate_qa_test_manifest()
    original = HARNESS.QA_ATTESTATION
    missing = json.loads(json.dumps(original))
    selection = next(
        suite for suite in missing["automated_tests"]["suites"]
        if suite["name"] == "selection_raw_producer"
    )["files"][0]
    selection["source_commit"] = "0" * 40
    monkeypatch.setattr(HARNESS, "QA_ATTESTATION", missing)
    with pytest.raises(HARNESS.HarnessError, match="source commit/path is unavailable"):
        HARNESS.validate_qa_test_manifest()

    mismatched = json.loads(json.dumps(original))
    selection = next(
        suite for suite in mismatched["automated_tests"]["suites"]
        if suite["name"] == "selection_raw_producer"
    )["files"][0]
    selection["git_blob"] = "0" * 40
    monkeypatch.setattr(HARNESS, "QA_ATTESTATION", mismatched)
    with pytest.raises(HARNESS.HarnessError, match="source Git blob differs"):
        HARNESS.validate_qa_test_manifest()


def test_explicit_restore_resets_nonzero_nrestarts_to_zero(tmp_path: Path) -> None:
    runtime = FakeRuntime(pre_nrestarts=1, post_nrestarts=0)
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), tmp_path / "nrestarts-reset", runtime.dependencies())
    assert code == 0 and evidence["restore"]["passed"] is True
    epoch = evidence["restore"]["post_start"]["service_epoch"]
    assert epoch["nrestarts_before"] == 1
    assert epoch["nrestarts_after"] == 0
    assert epoch["nrestarts_reset_to_zero"] is True
    assert runtime.full_hash_count == 1


def test_explicit_restore_rejects_nonzero_post_start_nrestarts(tmp_path: Path) -> None:
    runtime = FakeRuntime(pre_nrestarts=1, post_nrestarts=1)
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), tmp_path / "nrestarts-not-reset", runtime.dependencies())
    assert code == 1 and evidence["restore"]["passed"] is False
    assert evidence["restore"]["poll_count"] == 120
    assert "NRestarts semantics differ" in evidence["restore"]["last_failure"]["reason"]
    assert evidence["restore"]["duration_ns"] == 120_000_000_000
    assert runtime.full_hash_count == 1


def test_package_tree_snapshot_includes_root_directories_files_and_symlinks(tmp_path: Path) -> None:
    package = tmp_path / "package"
    weights = package / "weights"
    weights.mkdir(parents=True)
    shard = weights / "shard.gguf"
    shard.write_bytes(b"abc")
    (package / "shard-link").symlink_to(shard)
    before = HARNESS.package_tree_snapshot(package)
    assert {item.relative_path for item in before.entries} == {".", "weights", "weights/shard.gguf", "shard-link"}
    assert before.entry_count == 4
    assert before.file_count == 1
    assert before.directory_count == 2
    assert before.symlink_count == 1
    assert before.bytes == 3
    shard.write_bytes(b"abcd")
    after = HARNESS.package_tree_snapshot(package)
    difference = HARNESS._package_tree_difference(before, after)
    assert difference["kind"] == "metadata_changed"
    assert difference["relative_path"] == "weights/shard.gguf"
    assert {"size", "mtime_ns", "ctime_ns"} & set(difference["changed_fields"])


def test_pre_stop_23_second_full_hash_does_not_consume_restore_deadline(tmp_path: Path) -> None:
    runtime = FakeRuntime(hash_seconds=23.0)
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), tmp_path / "slow-hash", runtime.dependencies())
    assert code == 0 and evidence["status"] == "passed"
    assert runtime.full_hash_count == evidence["package_integrity"]["full_hash_count"] == 1
    assert evidence["package_integrity"]["full_content"]["duration_ns"] == 23_000_000_000
    assert evidence["restore"]["started_monotonic_ns"] >= 23_000_000_000
    assert evidence["restore"]["duration_ns"] == 0


@pytest.mark.parametrize("readiness_failures", (1, 2, 7))
def test_restore_retries_lightweight_dynamic_probe_until_success(tmp_path: Path, readiness_failures: int) -> None:
    runtime = FakeRuntime(restore_health_failures=readiness_failures)
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), tmp_path / f"restore-{readiness_failures}", runtime.dependencies())
    assert code == 0 and evidence["restore"]["passed"] is True
    assert evidence["restore"]["poll_count"] == readiness_failures + 1
    assert evidence["restore"]["last_failure"]["reason"] == "synthetic transient container health failure"
    assert runtime.full_hash_count == 1
    assert evidence["restore"]["final_metadata_recheck"]["passed"] is True


def test_restore_permanent_dynamic_failure_uses_one_absolute_deadline(tmp_path: Path) -> None:
    runtime = FakeRuntime(fail="health")
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), tmp_path / "restore-permanent", runtime.dependencies())
    assert code == 1 and evidence["restore"]["passed"] is False
    assert evidence["restore"]["poll_count"] == 120
    assert evidence["restore"]["duration_ns"] == 120_000_000_000
    assert max(item["probe_timeout_seconds"] for item in evidence["restore"]["polls"]) == 10.0
    assert evidence["restore"]["polls"][-1]["probe_timeout_seconds"] == 1.0
    assert evidence["restore"]["last_failure"]["reason"] == "synthetic container health failure"
    assert evidence["restore"]["final_metadata_recheck"] is None
    assert runtime.full_hash_count == 1


def test_restore_rejects_probe_that_crosses_absolute_deadline(tmp_path: Path) -> None:
    runtime = FakeRuntime(fail="restore-deadline-crossing")
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), tmp_path / "restore-crossing", runtime.dependencies())
    assert code == 1 and evidence["restore"]["passed"] is False
    assert evidence["restore"]["poll_count"] == 1
    assert evidence["restore"]["polls"][0]["probe_timeout_seconds"] == HARNESS.RESTORE_PROBE_TIMEOUT_SECONDS
    assert "crossed absolute deadline" in evidence["restore"]["last_failure"]["reason"]
    assert runtime.full_hash_count == 1


@pytest.mark.parametrize("mutation", ("added", "removed", "replaced", "content", "symlink", "directory"))
def test_restore_fails_closed_on_package_tree_metadata_mutation(tmp_path: Path, mutation: str) -> None:
    runtime = FakeRuntime(metadata_mutation=mutation)
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), tmp_path / f"metadata-{mutation}", runtime.dependencies())
    assert code == 1 and evidence["restore"]["passed"] is False
    recheck = evidence["restore"]["final_metadata_recheck"]
    assert recheck["passed"] is False
    assert recheck["identity_sha256"] != recheck["expected_identity_sha256"]
    assert recheck["difference"]["kind"] in {"added", "removed", "metadata_changed"}
    assert runtime.full_hash_count == 1


@pytest.mark.parametrize("mutation", ("added", "empty-directory", "special", "symlink", "directory"))
def test_pre_stop_rejects_stable_package_tree_divergence_from_trusted_identity(tmp_path: Path, mutation: str) -> None:
    runtime = FakeRuntime(pre_metadata_mutation=mutation)
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), tmp_path / f"pre-metadata-{mutation}", runtime.dependencies())
    assert code == 1 and evidence["failure"]["stage"] == "pre-stop-package-integrity"
    assert evidence["package_integrity"]["integrity_identity"]["passed"] is False
    assert evidence["package_integrity"]["error"] == "production package trusted integrity identity differs"
    assert evidence["process_counts"]["systemctl_stop"] == 0
    assert evidence["restore"]["attempted"] is False
    assert runtime.full_hash_count == 1


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
    assert documents[-1]["decision"] == "deadline_timeout"
    assert documents[-1]["observation_completed_monotonic_ns"] >= documents[-1]["absolute_deadline_monotonic_ns"]
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


def test_stopped_gate_slow_second_stable_crossing_deadline_fails_closed_and_restores(tmp_path: Path) -> None:
    runtime = FakeRuntime(fail="slow-second-stable")
    output = tmp_path / "slow-second-stable"
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), output, runtime.dependencies())
    assert code == 1 and evidence["status"] == "failed"
    assert evidence["stopped_gate_poll"]["failure"]["kind"] == "timeout"
    documents = _poll_documents(output, evidence)
    assert [item["decision"] for item in documents] == ["stable", "deadline_timeout"]
    assert documents[-1]["consecutive_stable"] == 0
    assert documents[-1]["observation_completed_monotonic_ns"] >= documents[-1]["absolute_deadline_monotonic_ns"]
    assert evidence["process_counts"]["launcher"] == 0
    assert evidence["restore"]["passed"] is True


def test_stopped_gate_deadline_crossing_probe_saves_timeout_evidence_and_restores(tmp_path: Path) -> None:
    runtime = FakeRuntime(fail="deadline-crossing-probe")
    output = tmp_path / "deadline-crossing-probe"
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), output, runtime.dependencies())
    assert code == 1 and evidence["stopped_gate_poll"]["failure"]["kind"] == "timeout"
    document = _poll_documents(output, evidence)[-1]
    assert document["decision"] == "deadline_timeout"
    assert document["deadline_checkpoints"]
    timeout_probe = document["observation"]["partial_probes"][-1]
    assert timeout_probe["label"] == "fake-deadline-crossing-probe"
    assert timeout_probe["exit_code"] == "timeout"
    assert 0 < timeout_probe["timeout_seconds"] <= HARNESS.STOP_POLL_PROBE_TIMEOUT_SECONDS
    assert len(timeout_probe["stdout_sha256"]) == len(timeout_probe["stderr_sha256"]) == 64
    assert evidence["process_counts"]["launcher"] == 0
    assert evidence["restore"]["passed"] is True


def test_stopped_gate_malformed_amd_schema_saves_secret_free_shape_and_raw_sha(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    raw = b'[{"gpu":2,"process_list":[{"process_info":"N/A"}]}]'

    def malformed_observation(old_worker_pid, old_service_pid, run, control):
        completed, _ = control.command(
            lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, raw, b""),
            ["fake-malformed-amd-process"],
            "stopped-poll-amd-process",
        )
        HARNESS.LAUNCHER.parse_amd_process_owners(completed.stdout)
        raise AssertionError("strict parser unexpectedly accepted malformed sentinel")

    dependencies = replace(runtime.dependencies(), stopped_observation=malformed_observation)
    output = tmp_path / "malformed-amd-process"
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), output, dependencies)
    assert code == 1 and evidence["status"] == "failed"
    document = _poll_documents(output, evidence)[-1]
    assert document["decision"] == "terminal_failure"
    assert document["source_classification"] == {"observer": "amd_process_schema"}
    diagnostic = document["observation"]["parse_diagnostic"]
    probe = document["observation"]["partial_probes"][-1]
    assert diagnostic["reason_code"] == "sentinel_mixed_or_unknown"
    assert diagnostic["top_level_type"] == "list"
    assert diagnostic["root_keys"] == ["gpu", "process_list"]
    assert diagnostic["raw_sha256"] == probe["stdout_sha256"] == HARNESS.sha_bytes(raw)
    assert raw not in HARNESS.pretty(document)
    assert evidence["process_counts"]["launcher"] == 0
    assert evidence["restore"]["passed"] is True


def test_stopped_poll_runs_due_keepalive_after_blocked_probe() -> None:
    now = 0
    keepalives: list[tuple[int, float, int]] = []

    def monotonic_ns() -> int:
        return now

    def keepalive(attempt: int, timeout: float) -> None:
        keepalives.append((attempt, timeout, now))

    control = HARNESS.StoppedPollControl(deadline_ns=30_000_000_000, monotonic_ns=monotonic_ns, keepalive=keepalive, first_keepalive_ns=1_000_000_000)
    control.begin_attempt(7)

    def blocked_probe(argv, **kwargs):
        nonlocal now
        assert kwargs["timeout"] <= HARNESS.STOP_POLL_PROBE_TIMEOUT_SECONDS
        now += 1_500_000_000
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    control.command(blocked_probe, ["fake-blocked-probe"], "fake-blocked-probe")
    assert keepalives == [(7, HARNESS.STOP_POLL_PROBE_TIMEOUT_SECONDS, 1_500_000_000)]
    assert control.checkpoints[-1]["keepalive"] == "passed"


def test_stopped_poll_caps_probe_timeout_to_remaining_deadline() -> None:
    now = 750_000_000
    observed_timeouts: list[float] = []
    control = HARNESS.StoppedPollControl(deadline_ns=1_000_000_000, monotonic_ns=lambda: now, keepalive=lambda attempt, timeout: None, first_keepalive_ns=2_000_000_000)
    control.begin_attempt(0)

    def probe(argv, **kwargs):
        observed_timeouts.append(kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    control.command(probe, ["fake-near-deadline-probe"], "fake-near-deadline-probe")
    assert observed_timeouts == [0.25]
    assert control.probes[-1]["timeout_seconds"] == 0.25


def test_default_stopped_observer_records_probe_sha_pids_vram_and_redacted_cmdline(monkeypatch: pytest.MonkeyPatch) -> None:
    old_worker_pid = 111
    old_service_pid = 222
    monkeypatch.setattr(HARNESS.LAUNCHER, "validate_amd_smi_tool", lambda: None)
    expected_sha = {HARNESS.LAUNCHER.SYSTEMCTL: HARNESS.LAUNCHER.SYSTEMCTL_SHA, HARNESS.LAUNCHER.PGREP: HARNESS.LAUNCHER.PGREP_SHA}
    monkeypatch.setattr(HARNESS.LAUNCHER, "sha_file", lambda path, label: (expected_sha[path], ()))
    kfd_source = {
        "schema_version": "ullm.aq4_p2_kfd_owner_snapshot.v1",
        "classification": "stable",
        "owners": [old_worker_pid],
        "secret_material_recorded": False,
    }
    monkeypatch.setattr(HARNESS.LAUNCHER, "_kfd_owner_snapshot", lambda **kwargs: kfd_source)
    monkeypatch.setattr(HARNESS, "_poll_lock_observation", lambda: {"path": str(HARNESS.LAUNCHER.LOCK_PATH), "free": False, "device": 1, "inode": 2, "holder_pids": [old_service_pid], "source_sha256": "a" * 64, "source_bytes": 12})
    monkeypatch.setattr(HARNESS, "_safe_proc_cmdline", lambda pid: {"pid": pid, "readable": True, "bytes": 9, "sha256": "b" * 64, "argv0_basename": "worker", "matches_expected_worker": pid == old_worker_pid, "raw_recorded": False})

    def run(argv, **kwargs):
        if argv[:2] == [str(HARNESS.LAUNCHER.SYSTEMCTL), "show"]:
            return subprocess.CompletedProcess(argv, 0, b"ActiveState=inactive\nSubState=dead\nMainPID=0\n", b"")
        if argv[0] == str(HARNESS.LAUNCHER.PGREP):
            return subprocess.CompletedProcess(argv, 1, b"", b"")
        if argv[1:3] == ["process", "--gpu"]:
            value = [{"gpu": HARNESS.LAUNCHER.AMD_SMI_INDEX, "process_list": [{"process_info": {
                "name": str(HARNESS.WORKER),
                "pid": old_worker_pid,
                "mem_usage": {"value": 7_351_832_576, "unit": "B"},
                "cu_occupancy": "N/A",
                "evicted_time": {"value": 682, "unit": "ms"},
            }}]}]
            return subprocess.CompletedProcess(argv, 0, json.dumps(value).encode(), b"")
        if argv[1:3] == ["static", "--gpu"]:
            value = {"gpu_data": [{"gpu": HARNESS.LAUNCHER.AMD_SMI_INDEX, "vram": {"size": {"value": 32624, "unit": "MB"}}}]}
            return subprocess.CompletedProcess(argv, 0, json.dumps(value).encode(), b"")
        raise AssertionError(argv)

    control = HARNESS.StoppedPollControl(deadline_ns=30_000_000_000, monotonic_ns=lambda: 0, keepalive=lambda attempt, timeout: None, first_keepalive_ns=10_000_000_000)
    control.begin_attempt(0)
    value = HARNESS.StoppedGateObserver()(old_worker_pid, old_service_pid, run, control)
    assert value["amd_smi_owners"] == value["kfd_owners"] == [old_worker_pid]
    assert value["lock"]["holder_pids"] == [old_service_pid]
    assert value["vram"] == {"total_bytes": 32_624_000_000, "used_bytes": None, "free_bytes": None, "headroom_bytes": None}
    assert len(value["probes"]) == 5
    assert all(len(item["stdout_sha256"]) == 64 and len(item["stderr_sha256"]) == 64 for item in value["probes"])
    assert value["virtual_sources"]["amd_smi_owners"]["reason_code"] == "accepted_owner_records"
    assert len(value["virtual_sources"]["amd_smi_owners"]["raw_sha256"]) == 64
    assert value["virtual_sources"]["kfd_owners"] == kfd_source
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
    assert runtime.full_hash_count == 0
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
    assert runtime.full_hash_count == 1
    if launcher_mode == "fail":
        assert evidence["launcher"]["failure"]["runner_started"] is False
        assert evidence["launcher"]["runner_finished"] is False
        assert evidence["launcher"]["runner_not_started"] is True


def test_launcher_rc1_before_runner_start_allows_known_safe_substrate_cleanup(tmp_path: Path) -> None:
    runtime = FakeRuntime(launcher_mode="fail")
    substrate = HARNESS.LockSubstrate(tmp_path / "lock-dir", tmp_path / "lock-dir/device.lock", (7, 8), (1, 2), {"passed": True})
    cleanup_calls: list[dict] = []

    def stopped(old_worker_pid, old_service_pid, run, control):
        value = runtime.stopped(old_worker_pid, old_service_pid, run, control)
        value["lock"].update(
            {
                "source": "trusted_substrate",
                "substrate": {"directory": {"device": 7, "inode": 8}, "lock": {"device": 1, "inode": 2}},
            }
        )
        return value

    def cleanup(value, run, *, runner_finished, runner_children):
        cleanup_calls.append({"runner_finished": runner_finished, "runner_children": runner_children})
        return {"passed": True, "secret_material_recorded": False}

    dependencies = replace(
        runtime.dependencies(),
        stopped_observation=stopped,
        lock_substrate_prepare=lambda run: substrate,
        lock_substrate_cleanup=cleanup,
    )
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), tmp_path / "launcher-never-started", dependencies)
    assert code == 1 and evidence["restore"]["passed"] is True
    assert cleanup_calls == [{"runner_finished": False, "runner_children": []}]
    assert evidence["lock_substrate_cleanup"]["passed"] is True


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
    code, evidence = HARNESS.dry_run_ready(value, tmp_path / "dry", HARNESS.READY_PATH)
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
    assert value["execution_mode"] == "profile_diagnostic"
    assert value["measurement_eligible"] is False and value["promotion_eligible"] is False
    assert value["authorization"]["run_id"] == HARNESS.LAUNCHER.PROFILE_RUN_ID
    assert value["authorization"]["maximum_invocations"] == 1
    assert value["authorization"]["rocprof_wrapper_required"] is True
    assert profile["execution_boundary"] == {
        "order": ["maintenance", "launcher", "validator", "gates", "capture", "rocprof", "runner"],
        "runner_profiled": True,
        "validator_profiled": False,
        "gates_profiled": False,
    }
    assert profile["target_runner"] == {
        "generated_by": "launcher_after_live_preflight",
        "file_name": HARNESS.LAUNCHER.PROFILE_RUNNER_TARGET_MANIFEST_NAME,
        "fresh_per_execution": True,
        "environment": "exact_execute_environment",
        "maximum_invocations": 1,
    }
    assert profile["output"]["must_not_exist_before_capture"] is True
    assert profile["resident_evidence"]["run_id"] == HARNESS.LAUNCHER.PROFILE_RUN_ID
    assert profile["resident_evidence"]["resident_session_id_source"] == "resident_raw.resident.session_id"
    assert profile["resident_evidence"]["case_id"] == HARNESS.LAUNCHER.CASE_ID
    assert profile["roctx"]["roctx_library"]["resolved_path"] == str(HARNESS.LAUNCHER.ROCTX_LIBRARY_RESOLVED)


def test_profile_capture_command_binds_fresh_launcher_runner_target(tmp_path: Path) -> None:
    profile = profile_ready(tmp_path)["profile_diagnostic"]
    target = {"path": str(tmp_path / "runner-target-command-manifest.json"), "sha256": "a" * 64}
    command = HARNESS.profile_capture_command(target, profile)
    assert command[command.index("--target-command-manifest") + 1] == target["path"]
    assert command[command.index("--target-command-manifest-sha256") + 1] == target["sha256"]
    assert command[command.index("--profile-output-directory") + 1] == profile["output"]["directory"]
    assert command[command.index("--artifact") + 1] == profile["output"]["artifact"]
    assert "--runner-command" not in command


def valid_profile_capture_artifact(request: dict, output: Path) -> dict:
    contract = request["contract"]
    target = request["target_binding"]
    command = HARNESS._expected_profile_command(request["runner_argv"], contract)
    identity_sha = "d" * 64

    def ref(path: Path, digit: str = "e") -> dict:
        del digit
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(f"fixture:{path.name}\n".encode())
        return {"path": str(path), "sha256": HARNESS.sha_bytes(path.read_bytes())}

    capabilities = ref(output / "capture-capabilities.json", "f")
    helpers = [
        {"role": role, "path": str(path), "identity": list(HARNESS.LAUNCHER.file_identity(path.lstat())), "sha256": sha}
        for role, path, sha in (
            ("selection_raw_producer", HARNESS.LAUNCHER.PROFILE_PRODUCER_HELPER, HARNESS.LAUNCHER.PROFILE_PRODUCER_HELPER_SHA),
            ("candidate_selector", HARNESS.LAUNCHER.PROFILE_SELECTOR_HELPER, HARNESS.LAUNCHER.PROFILE_SELECTOR_HELPER_SHA),
            ("profile_family_classifier", HARNESS.LAUNCHER.PROFILE_FAMILY_HELPER, HARNESS.LAUNCHER.PROFILE_FAMILY_HELPER_SHA),
        )
    ]
    runs = []
    memory_traces = []
    for index in range(2, 12):
        runs.append({
            "schema_version": "ullm.aq4_p3_rocprof_run_binding.v1",
            "case_id": contract["resident_evidence"]["case_id"],
            "case_sha256": HARNESS.LAUNCHER.CASE_SHA,
            "identity_sha256": identity_sha,
            "resident_run_index": index,
            "measurement_eligible": False,
            "clock_domain": "rocprofv3_monotonic_ns",
            "kernel_trace_complete": True,
            "hip_api_trace_complete": True,
            "capture_capabilities": capabilities,
            "kernel_trace": ref(output / "measured-runs" / f"run-{index:02d}_kernel_trace.csv"),
            "hip_api_trace": ref(output / "measured-runs" / f"run-{index:02d}_hip_api_trace.csv"),
        })
        memory_traces.append(ref(output / "measured-runs" / f"run-{index:02d}_memory_copy_trace.csv"))
    artifact = {
        "schema_version": HARNESS.PROFILE_CAPTURE_SCHEMA,
        "status": "complete_diagnostic",
        "measurement_eligible": False,
        "promotion_eligible": False,
        "artifact_sha256": None,
        "binding": {
            "run_id": contract["resident_evidence"]["run_id"],
            "resident_session_id": "fixture-session",
            "case_id": contract["resident_evidence"]["case_id"],
            "case_sha256": HARNESS.LAUNCHER.CASE_SHA,
            "identity_sha256": identity_sha,
            "device": {"runtime_device_index": 1, "device_id": "r9700-rdna4", "backend": "hip", "name": "AMD Radeon Graphics", "architecture": "gfx1201"},
            "identity": ref(Path(contract["resident_evidence"]["identity"])),
            "resident_summary": ref(Path(contract["resident_evidence"]["summary"])),
            "resident_raw": ref(Path(contract["resident_evidence"]["raw"])),
        },
        "profiler": {
            "tool": "rocprofv3",
            "invocation_path": str(HARNESS.PROFILE_PROFILER),
            "resolved_path": str(HARNESS.PROFILE_PROFILER),
            "executable_sha256": HARNESS.PROFILE_PROFILER_SHA,
            "resolved_identity": list(HARNESS.LAUNCHER.file_identity(HARNESS.PROFILE_PROFILER.lstat())),
            "symlink_chain": [],
            "version": "fixture",
            "rocm_version": None,
            "version_output_sha256": "a" * 64,
            "target_command_manifest": {"path": target["path"], "sha256": target["sha256"]},
            "target_environment": {"sha256": HARNESS.sha_bytes(HARNESS.canonical(request["environment"])), "keys": sorted(request["environment"]), "exact_base_environment": True, "secret_material_recorded": False},
            "capture_helpers": helpers,
            "command": command,
            "command_sha256": HARNESS.sha_bytes(HARNESS.canonical(command)),
            "subprocess_profile_runs": 1,
        },
        "source_traces": {kind: ref(output / f"fixture_{kind}_trace.csv") for kind in ("kernel", "hip_api", "memory_copy", "marker")},
        "capture_capabilities": capabilities,
        "marker_contract": {"schema_version": "ullm.aq4_p2.run.v1", "clock_domain": "rocprofv3_monotonic_ns", "range_count": 12, "warmup_indices": [0, 1], "measured_indices": list(range(2, 12)), "warmup_excluded": True},
        "producer_profile_runs": runs,
        "memory_copy_traces": memory_traces,
        "eligibility_blockers": ["rocprof instrumentation overhead forbids performance promotion", "one-case diagnostic evidence does not satisfy seven-prompt promotion coverage"],
    }
    artifact["artifact_sha256"] = HARNESS._semantic_self_hash(artifact, "artifact_sha256")
    return artifact


def ready_candidate_capture_absent(stderr_raw: bytes) -> dict:
    value = {
        "schema_version": HARNESS.READY_CANDIDATE_CAPTURE_SCHEMA,
        "self_sha256": None,
        "status": "absent",
        "reason_code": "ready_candidate_marker_absent",
        "source_stream": "rocprof.stderr",
        "source_stream_sha256": HARNESS.sha_bytes(stderr_raw),
        "marker_count": 0,
        "marker_sha256": None,
        "audit_sha256": None,
        "audit": None,
    }
    value["self_sha256"] = HARNESS._semantic_self_hash(value, "self_sha256")
    return value


def ready_candidate_failed_audit() -> dict:
    candidate = {
        "event": "not_ready",
        "schema_version": "ullm.aq4_p2_resident_driver.v2",
        "model_loads": 1,
        "resident_session_id": "fixture-session",
        "driver_identity": {"binary_sha256": "a" * 64},
        "served_model_binding": {"schema_version": "ullm.aq4_p2_served_model_binding.v2"},
    }

    def json_type(value):
        if value is None: return "null"
        if isinstance(value, bool): return "boolean"
        if isinstance(value, int): return "integer"
        if isinstance(value, float): return "number"
        if isinstance(value, str): return "string"
        if isinstance(value, list): return "array"
        return "object"

    def key_types(value):
        keys = sorted(value)
        return keys, {key: json_type(value[key]) for key in keys}

    def scalar(key):
        value = candidate[key]
        return {
            "present": True,
            "json_type": json_type(value),
            "value": value,
            "string_length": len(value) if isinstance(value, str) else None,
            "canonical_sha256": HARNESS.sha_bytes(HARNESS.canonical(value)),
        }

    def nested(key):
        value = candidate[key]
        keys, types = key_types(value)
        return {
            "present": True,
            "json_type": "object",
            "canonical_sha256": HARNESS.sha_bytes(HARNESS.canonical(value)),
            "keys": keys,
            "key_types": types,
        }

    top_keys, top_types = key_types(candidate)
    candidate_raw = HARNESS.canonical(candidate) + b"\n"
    audit = {
        "schema_version": HARNESS.READY_CANDIDATE_AUDIT_SCHEMA,
        "audit_sha256": None,
        "raw": {
            "byte_count": len(candidate_raw),
            "raw_sha256": HARNESS.sha_bytes(candidate_raw),
        },
        "top_level": {
            "key_count": len(candidate),
            "keys": top_keys,
            "key_types": top_types,
        },
        "safe_scalars": {
            "event": scalar("event"),
            "schema_version": scalar("schema_version"),
            "model_loads": scalar("model_loads"),
        },
        "resident_session_id": {
            "present": True,
            "json_type": "string",
            "string_length": len(candidate["resident_session_id"]),
            "canonical_sha256": HARNESS.sha_bytes(
                HARNESS.canonical(candidate["resident_session_id"])
            ),
        },
        "nested": {
            "driver_identity": nested("driver_identity"),
            "served_model_binding": nested("served_model_binding"),
        },
        "validation": {
            "status": "failed",
            "reason_code": "ready_candidate_event_differs",
            "predicates": {
                "field_set_exact": True,
                "event_is_ready": False,
                "schema_version_exact": True,
                "model_loads_is_integer": True,
                "model_loads_is_one": True,
                "resident_session_id_is_string": True,
                "resident_session_id_nonempty": True,
            },
        },
    }
    audit["audit_sha256"] = HARNESS._semantic_self_hash(audit, "audit_sha256")
    return audit


def ready_candidate_capture_valid(audit: dict | None = None, *, marker_payload: bytes | None = None):
    audit = ready_candidate_failed_audit() if audit is None else audit
    payload = HARNESS.canonical(audit) if marker_payload is None else marker_payload
    marker = HARNESS.READY_CANDIDATE_MARKER_PREFIX + payload + b"\n"
    value = {
        "schema_version": HARNESS.READY_CANDIDATE_CAPTURE_SCHEMA,
        "self_sha256": None,
        "status": "valid",
        "reason_code": "ready_candidate_marker_bound",
        "source_stream": "rocprof.stderr",
        "source_stream_sha256": HARNESS.sha_bytes(marker),
        "marker_count": 1,
        "marker_sha256": HARNESS.sha_bytes(marker),
        "audit_sha256": audit["audit_sha256"],
        "audit": audit,
    }
    value["self_sha256"] = HARNESS._semantic_self_hash(value, "self_sha256")
    return marker, value


def ready_candidate_capture_invalid(stderr_raw: bytes, reason: str, marker_count: int, marker_sha256=None):
    value = {
        "schema_version": HARNESS.READY_CANDIDATE_CAPTURE_SCHEMA,
        "self_sha256": None,
        "status": "invalid",
        "reason_code": reason,
        "source_stream": "rocprof.stderr",
        "source_stream_sha256": HARNESS.sha_bytes(stderr_raw),
        "marker_count": marker_count,
        "marker_sha256": marker_sha256,
        "audit_sha256": None,
        "audit": None,
    }
    value["self_sha256"] = HARNESS._semantic_self_hash(value, "self_sha256")
    return value


def test_ready_candidate_capture_accepts_valid_canonical_and_absent_marker() -> None:
    valid_stream, valid = ready_candidate_capture_valid()
    observed = HARNESS._validate_ready_candidate_capture(
        valid,
        stderr_raw=valid_stream,
        stderr_sha256=HARNESS.sha_bytes(valid_stream),
    )
    assert observed["status"] == "valid"
    assert observed["audit"]["validation"]["reason_code"] == "ready_candidate_event_differs"

    absent_stream = b"AQ4 P2 resident batch failed: unrelated failure\n"
    absent = ready_candidate_capture_absent(absent_stream)
    observed = HARNESS._validate_ready_candidate_capture(
        absent,
        stderr_raw=absent_stream,
        stderr_sha256=HARNESS.sha_bytes(absent_stream),
    )
    assert observed["status"] == "absent" and observed["marker_count"] == 0


@pytest.mark.parametrize("kind", ("malformed", "oversize", "multiple"))
def test_ready_candidate_capture_accepts_bounded_invalid_diagnostics(kind: str) -> None:
    if kind == "malformed":
        stream = HARNESS.READY_CANDIDATE_MARKER_PREFIX + b"{malformed}\n"
        envelope = ready_candidate_capture_invalid(
            stream,
            "ready_candidate_marker_payload_invalid",
            1,
            HARNESS.sha_bytes(stream),
        )
    elif kind == "oversize":
        stream = (
            HARNESS.READY_CANDIDATE_MARKER_PREFIX
            + b"x" * HARNESS.MAX_READY_CANDIDATE_MARKER_BYTES
            + b"\n"
        )
        envelope = ready_candidate_capture_invalid(
            stream, "ready_candidate_marker_oversize", 0
        )
    else:
        marker = HARNESS.READY_CANDIDATE_MARKER_PREFIX + b"{malformed}\n"
        stream = marker + marker
        envelope = ready_candidate_capture_invalid(
            stream, "ready_candidate_marker_count_differs", 2
        )
    observed = HARNESS._validate_ready_candidate_capture(
        envelope,
        stderr_raw=stream,
        stderr_sha256=HARNESS.sha_bytes(stream),
    )
    assert observed["status"] == "invalid"
    assert observed["audit"] is None


@pytest.mark.parametrize(
    "mutation",
    (
        "envelope-extra", "envelope-self-hash", "stream-hash", "audit-extra",
        "audit-self-hash", "predicate", "reason", "noncanonical", "secret-key",
        "secret-value", "raw-path", "fd-path", "nested-hash",
    ),
)
def test_ready_candidate_capture_rejects_unknown_mismatch_secret_path_and_fd(
    mutation: str,
) -> None:
    audit = ready_candidate_failed_audit()
    marker_payload = None
    if mutation == "audit-extra":
        audit["unknown"] = True
    elif mutation == "audit-self-hash":
        audit["audit_sha256"] = "f" * 64
    elif mutation == "predicate":
        audit["validation"]["predicates"]["schema_version_exact"] = False
        audit["audit_sha256"] = HARNESS._semantic_self_hash(audit, "audit_sha256")
    elif mutation == "reason":
        audit["validation"]["reason_code"] = "ready_candidate_schema_differs"
        audit["audit_sha256"] = HARNESS._semantic_self_hash(audit, "audit_sha256")
    elif mutation == "secret-key":
        audit["top_level"]["key_count"] = 7
        audit["top_level"]["keys"].append("password")
        audit["top_level"]["keys"].sort()
        audit["top_level"]["key_types"]["password"] = "string"
        audit["validation"]["predicates"]["field_set_exact"] = False
        audit["validation"]["reason_code"] = "ready_candidate_field_set_differs"
        audit["audit_sha256"] = HARNESS._semantic_self_hash(audit, "audit_sha256")
    elif mutation in {"secret-value", "raw-path", "fd-path"}:
        value = {
            "secret-value": "BearerSecret",
            "raw-path": "/tmp/private-model",
            "fd-path": "/proc/self/fd/91",
        }[mutation]
        scalar = audit["safe_scalars"]["event"]
        scalar.update({
            "value": value,
            "string_length": len(value),
            "canonical_sha256": HARNESS.sha_bytes(HARNESS.canonical(value)),
        })
        audit["audit_sha256"] = HARNESS._semantic_self_hash(audit, "audit_sha256")
    elif mutation == "nested-hash":
        audit["nested"]["driver_identity"]["canonical_sha256"] = "invalid"
        audit["audit_sha256"] = HARNESS._semantic_self_hash(audit, "audit_sha256")
    elif mutation == "noncanonical":
        marker_payload = json.dumps(audit, sort_keys=False, separators=(", ", ": ")).encode()
    stream, envelope = ready_candidate_capture_valid(audit, marker_payload=marker_payload)
    if mutation == "envelope-extra":
        envelope["unknown"] = True
    elif mutation == "envelope-self-hash":
        envelope["self_sha256"] = "e" * 64
    elif mutation == "stream-hash":
        envelope["source_stream_sha256"] = "d" * 64
        envelope["self_sha256"] = HARNESS._semantic_self_hash(envelope, "self_sha256")
    with pytest.raises(HARNESS.HarnessError):
        HARNESS._validate_ready_candidate_capture(
            envelope,
            stderr_raw=stream,
            stderr_sha256=HARNESS.sha_bytes(stream),
        )


def default_profile_adapter_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str):
    bridge_name = f"profile_capture_test_bridge_{mode.replace('-', '_')}"
    bridge = HARNESS.types.ModuleType(bridge_name)
    trusted_raw = f"import sys\ndef main(argv, *, on_rocprof_started=None, on_runner_completed=None):\n    return sys.modules[{bridge_name!r}].invoke(argv, on_rocprof_started, on_runner_completed)\n".encode()
    trusted_path = tmp_path / "trusted-capture.py"
    trusted_path.write_bytes(trusted_raw)
    monkeypatch.setattr(HARNESS, "PROFILE_CAPTURE_TOOL", trusted_path)
    monkeypatch.setattr(HARNESS, "PROFILE_CAPTURE_SHA", HARNESS.sha_bytes(trusted_raw))
    profile = profile_ready(tmp_path)["profile_diagnostic"]
    output = Path(profile["output"]["directory"])
    artifact_path = Path(profile["output"]["artifact"])
    target = {
        "path": str(tmp_path / "runner-target-command-manifest.json"),
        "sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "identity": [1, 2, 3, 4, 1, 5, 6],
    }
    starts: list[str] = []

    def invoke(argv, on_rocprof_started, on_runner_completed):
        if mode == "exception-before":
            raise RuntimeError("synthetic pre-spawn executor exception")
        output.mkdir(mode=0o700)
        target_reference = {
            "path": argv[argv.index("--target-command-manifest") + 1],
            "sha256": argv[argv.index("--target-command-manifest-sha256") + 1],
        }
        if mode != "before-start-failure":
            on_rocprof_started()
        if mode == "exception-after":
            raise RuntimeError("synthetic post-spawn executor exception")
        (output / "rocprof.stdout").write_bytes(b"")
        (output / "rocprof.stderr").write_bytes(b"")
        if mode in {
            "success", "tampered-artifact", "missing-artifact", "artifact-mode",
            "artifact-binding-field", "artifact-profiler-field", "artifact-command",
            "artifact-ref-hash", "artifact-ref-outside", "artifact-ref-dotdot",
            "artifact-ref-symlink", "artifact-ref-hardlink", "artifact-ref-mode",
        }:
            on_runner_completed()
            if mode != "missing-artifact":
                artifact = valid_profile_capture_artifact(request, output)
                if mode == "tampered-artifact":
                    artifact["status"] = "tampered"
                elif mode == "artifact-binding-field":
                    artifact["binding"]["unknown"] = True
                    artifact["artifact_sha256"] = HARNESS._semantic_self_hash(artifact, "artifact_sha256")
                elif mode == "artifact-profiler-field":
                    artifact["profiler"]["unknown"] = True
                    artifact["artifact_sha256"] = HARNESS._semantic_self_hash(artifact, "artifact_sha256")
                elif mode == "artifact-command":
                    artifact["profiler"]["command"][-1] = "different-runner"
                    artifact["profiler"]["command_sha256"] = HARNESS.sha_bytes(HARNESS.canonical(artifact["profiler"]["command"]))
                    artifact["artifact_sha256"] = HARNESS._semantic_self_hash(artifact, "artifact_sha256")
                elif mode == "artifact-ref-hash":
                    artifact["source_traces"]["kernel"]["sha256"] = "9" * 64
                    artifact["artifact_sha256"] = HARNESS._semantic_self_hash(artifact, "artifact_sha256")
                elif mode == "artifact-ref-outside":
                    outside = output.parent / "outside-kernel.csv"
                    outside.write_bytes(b"outside\n")
                    artifact["source_traces"]["kernel"] = {"path": str(outside), "sha256": HARNESS.sha_bytes(outside.read_bytes())}
                    artifact["artifact_sha256"] = HARNESS._semantic_self_hash(artifact, "artifact_sha256")
                elif mode == "artifact-ref-dotdot":
                    (output / "nested").mkdir()
                    original = Path(artifact["source_traces"]["kernel"]["path"])
                    artifact["source_traces"]["kernel"]["path"] = str(output / "nested" / ".." / original.name)
                    artifact["artifact_sha256"] = HARNESS._semantic_self_hash(artifact, "artifact_sha256")
                elif mode == "artifact-ref-symlink":
                    original = Path(artifact["source_traces"]["kernel"]["path"])
                    link = output / "linked-kernel.csv"
                    link.symlink_to(original)
                    artifact["source_traces"]["kernel"]["path"] = str(link)
                    artifact["artifact_sha256"] = HARNESS._semantic_self_hash(artifact, "artifact_sha256")
                elif mode == "artifact-ref-hardlink":
                    original = Path(artifact["source_traces"]["kernel"]["path"])
                    os.link(original, output / "hardlinked-kernel.csv")
                    artifact["artifact_sha256"] = HARNESS._semantic_self_hash(artifact, "artifact_sha256")
                elif mode == "artifact-ref-mode":
                    Path(artifact["source_traces"]["kernel"]["path"]).chmod(0o755)
                    artifact["artifact_sha256"] = HARNESS._semantic_self_hash(artifact, "artifact_sha256")
                artifact_path.write_bytes(HARNESS.pretty(artifact))
                artifact_path.chmod(0o644 if mode == "artifact-mode" else 0o444)
            return 0
        reason = "rocprof diagnostic capture timed out" if mode == "timeout" else "synthetic capture failure"
        cleanup = mode not in {"cleanup-failed", "children-nonempty"}
        children_known = mode != "cleanup-failed"
        children_remaining = [4242] if mode == "children-nonempty" else []
        if mode == "failure-ready-invalid":
            (output / "rocprof.stderr").write_bytes(
                HARNESS.READY_CANDIDATE_MARKER_PREFIX + b"{malformed}\n"
            )
        streams = {}
        for name in ("rocprof.stdout", "rocprof.stderr"):
            raw = (output / name).read_bytes()
            streams[name] = {"bytes": len(raw), "sha256": HARNESS.sha_bytes(raw)}
        failure = {
            "schema_version": HARNESS.PROFILE_CAPTURE_FAILURE_SCHEMA,
            "status": "failed",
            "measurement_eligible": False,
            "promotion_eligible": False,
            "failure_sha256": None,
            "reason": reason,
            "rocprof_child_new_session": True,
            "outer_harness_signalled": False,
            "process_group_cleanup_complete": cleanup,
            "children_state_known": children_known,
            "children_remaining": children_remaining,
            "command_sha256": HARNESS.sha_bytes(HARNESS.canonical(HARNESS._expected_profile_command(request["runner_argv"], profile))),
            "effective_command_sha256": HARNESS.sha_bytes(HARNESS.canonical(["/proc/self/fd/91", *HARNESS._expected_profile_command(request["runner_argv"], profile)[1:]])),
            "context": {"profiler": {"tool": "rocprofv3", "invocation_path": str(HARNESS.PROFILE_PROFILER), "resolved_path": str(HARNESS.PROFILE_PROFILER), "executable_sha256": HARNESS.PROFILE_PROFILER_SHA, "resolved_identity": list(HARNESS.LAUNCHER.file_identity(HARNESS.PROFILE_PROFILER.lstat())), "symlink_chain": []}, "target_command_manifest": target_reference},
            "streams": streams,
            "ready_candidate_audit": (
                ready_candidate_capture_invalid(
                    (output / "rocprof.stderr").read_bytes(),
                    "ready_candidate_marker_payload_invalid",
                    1,
                    HARNESS.sha_bytes((output / "rocprof.stderr").read_bytes()),
                )
                if mode == "failure-ready-invalid"
                else ready_candidate_capture_absent((output / "rocprof.stderr").read_bytes())
            ),
        }
        failure["failure_sha256"] = HARNESS._semantic_self_hash(failure, "failure_sha256")
        if mode == "tampered-failure":
            failure["reason"] = "tampered after self hash"
        elif mode == "failure-profiler":
            failure["context"]["profiler"]["executable_sha256"] = "9" * 64
            failure["failure_sha256"] = HARNESS._semantic_self_hash(failure, "failure_sha256")
        elif mode == "failure-command":
            failure["command_sha256"] = "9" * 64
            failure["failure_sha256"] = HARNESS._semantic_self_hash(failure, "failure_sha256")
        elif mode == "failure-effective-command":
            failure["effective_command_sha256"] = failure["command_sha256"]
            failure["failure_sha256"] = HARNESS._semantic_self_hash(failure, "failure_sha256")
        elif mode == "failure-extra":
            failure["unknown"] = True
            failure["failure_sha256"] = HARNESS._semantic_self_hash(failure, "failure_sha256")
        failure_path = output / HARNESS.PROFILE_CAPTURE_FAILURE_NAME
        failure_path.write_bytes(HARNESS.pretty(failure))
        failure_path.chmod(0o644 if mode == "failure-mode" else 0o444)
        return 1

    bridge.invoke = invoke
    monkeypatch.setitem(sys.modules, bridge_name, bridge)
    request = {
        "contract": profile,
        "runner_argv": ["/fake/runner", "profile"],
        "environment": dict(HARNESS.LAUNCHER.EXECUTE_ENV),
        "mark_runner_started": lambda: starts.append("started"),
        "target_binding": target,
    }
    return request, trusted_raw, starts, trusted_path


def test_default_profile_adapter_validates_success_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, trusted_raw, starts, _ = default_profile_adapter_fixture(tmp_path, monkeypatch, "success")
    outcome = HARNESS.run_profile_capture(request, trusted_capture_raw=trusted_raw)
    capture = outcome["profile_capture"]
    assert outcome["completed"].returncode == 0, outcome["profile_diagnostics"]["validation_error"]
    assert starts == ["started"]
    assert capture["status"] == "complete_diagnostic"
    assert outcome["profile_diagnostics"]["capture_artifact"]["mode"] == 0o444
    assert capture["cleanup_passed"] is True and capture["children_remaining"] == []
    assert capture["rocprof_started"] is True and capture["runner_start_known"] is True
    assert capture["runner_started"] is True and capture["runner_completed"] is True
    assert outcome["profile_diagnostics"]["runner_finished"] is True


def test_invalid_ready_candidate_diagnostic_does_not_fabricate_cleanup_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, trusted_raw, _, _ = default_profile_adapter_fixture(
        tmp_path, monkeypatch, "failure-ready-invalid"
    )
    outcome = HARNESS.run_profile_capture(request, trusted_capture_raw=trusted_raw)
    capture = outcome["profile_capture"]
    assert outcome["completed"].returncode == 1
    assert capture["status"] == "failed"
    assert capture["children_state_known"] is True
    assert capture["children_remaining"] == []
    assert capture["cleanup_passed"] is True
    failure = json.loads(
        (
            Path(request["contract"]["output"]["directory"])
            / HARNESS.PROFILE_CAPTURE_FAILURE_NAME
        ).read_text()
    )
    assert failure["ready_candidate_audit"]["status"] == "invalid"
    assert failure["ready_candidate_audit"]["audit"] is None


def test_default_profile_adapter_rejects_reference_change_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, trusted_raw, _, _ = default_profile_adapter_fixture(tmp_path, monkeypatch, "success")
    original = HARNESS._verified_profile_file
    changed = False

    def verify_then_change(path: Path, expected_sha256: str):
        nonlocal changed
        identity = original(path, expected_sha256)
        if identity is not None and path.name == "fixture_kernel_trace.csv" and not changed:
            changed = True
            path.write_bytes(b"changed-after-reference-hash\n")
        return identity

    monkeypatch.setattr(HARNESS, "_verified_profile_file", verify_then_change)
    outcome = HARNESS.run_profile_capture(request, trusted_capture_raw=trusted_raw)
    assert changed is True
    assert outcome["completed"].returncode == 1
    assert "referenced file identity changed" in outcome["profile_diagnostics"]["validation_error"]


@pytest.mark.parametrize("capture_mode", ("success", "failure-clean", "failure-unknown", "exception"))
def test_default_adapter_invokes_actual_capture_main_lifecycle_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capture_mode: str
) -> None:
    request, _, starts, _ = default_profile_adapter_fixture(tmp_path, monkeypatch, "success")
    actual_path = ROOT / "tools/capture-aq4-p3-diagnostic-profile.py"
    actual_raw = actual_path.read_bytes()
    monkeypatch.setattr(HARNESS, "PROFILE_CAPTURE_TOOL", actual_path)
    monkeypatch.setattr(HARNESS, "PROFILE_CAPTURE_SHA", HARNESS.sha_bytes(actual_raw))
    module = HARNESS._load_profile_capture_module(actual_raw)
    lifecycle: list[str] = []

    class FakeProfiler:
        descriptor = 91
        invocation = HARNESS.PROFILE_PROFILER
        def verify(self): return None
        def evidence(self):
            return {
                "tool": "rocprofv3", "invocation_path": str(self.invocation),
                "resolved_path": str(self.invocation), "executable_sha256": HARNESS.PROFILE_PROFILER_SHA,
                "resolved_identity": list(HARNESS.LAUNCHER.file_identity(HARNESS.PROFILE_PROFILER.lstat())), "symlink_chain": [],
            }
        def close(self): return None

    class FakeSnapshot:
        path = Path(request["target_binding"]["path"])
        sha256 = request["target_binding"]["sha256"]
        def verify(self): return None

    class FakeFdMap:
        descriptor = 92
        sha256 = "b" * 64
        value = {
            "map_sha256": "c" * 64,
            "bindings": [],
        }
        def verify(self): return None
        def close(self): return None

    profiler = FakeProfiler()
    snapshot = FakeSnapshot()
    runner_output = tmp_path / "profile-runner-output.json"
    target_value = {
        "argv": request["runner_argv"],
        "environment": request["environment"],
        "output_paths": [{"argument_index": 0, "path": str(runner_output)}],
        "closure_contract": {
            "code_execution_closure": "pinned_fd",
            "control_input_closure": "pinned_fd",
            "device_lock_closure": "pinned_fd",
            "data_integrity": "trusted_pre_post_guarded",
        },
    }
    def open_profiler(path, sha):
        if capture_mode == "exception":
            raise RuntimeError("synthetic actual-main executor exception")
        return profiler

    monkeypatch.setattr(module.PinnedProfiler, "open", staticmethod(open_profiler))
    monkeypatch.setattr(module.PinnedFdMap, "create", staticmethod(lambda value, snapshots: FakeFdMap()))
    monkeypatch.setattr(module, "pinned_profiler_version", lambda value: {**profiler.evidence(), "version": "fixture", "rocm_version": None, "version_output_sha256": "a" * 64})
    monkeypatch.setattr(module, "load_target_command_manifest", lambda path, sha, allow_existing_outputs=False: (target_value, [snapshot]))
    monkeypatch.setattr(module, "pinned_target_argv", lambda value, snapshots: (request["runner_argv"], ()))
    monkeypatch.setattr(module, "capture_helper_contract", lambda: [])
    monkeypatch.setattr(module, "verify_capture_helpers", lambda: None)
    monkeypatch.setattr(
        module,
        "profiler_command",
        lambda value, directory, name, command: [
            f"/proc/self/fd/{profiler.descriptor}",
            *HARNESS._expected_profile_command(request["runner_argv"], request["contract"])[1:],
        ],
    )

    def run_profile(command, output_directory, timeout, **kwargs):
        lifecycle.append("rocprof")
        kwargs["on_rocprof_started"]()
        output_directory.mkdir(mode=0o700)
        (output_directory / "rocprof.stdout").write_bytes(b"")
        (output_directory / "rocprof.stderr").write_bytes(b"")
        runner_output.write_text("{}\n")
        if capture_mode != "success":
            reason = "synthetic actual-main failure" if capture_mode == "failure-clean" else "synthetic actual-main cleanup failed"
            module.write_failure_evidence(
                output_directory,
                reason,
                HARNESS._expected_profile_command(request["runner_argv"], request["contract"]),
                kwargs["failure_context"],
                effective_command=command,
            )
            raise module.CaptureError(reason)

    def assemble(**kwargs):
        lifecycle.append("assemble")
        artifact = valid_profile_capture_artifact(request, Path(request["contract"]["output"]["directory"]))
        kwargs["artifact_path"].write_bytes(HARNESS.pretty(artifact))
        kwargs["artifact_path"].chmod(0o444)
        return artifact

    monkeypatch.setattr(module, "run_profile", run_profile)
    monkeypatch.setattr(module, "discover", lambda output: {})
    monkeypatch.setattr(module, "assemble", assemble)
    monkeypatch.setattr(module, "close_target_snapshots", lambda snapshots: None)
    monkeypatch.setattr(HARNESS, "_load_profile_capture_module", lambda raw: module)
    outcome = HARNESS.run_profile_capture(request, trusted_capture_raw=actual_raw)
    expected_lifecycle = ["rocprof", "assemble"] if capture_mode == "success" else ([] if capture_mode == "exception" else ["rocprof"])
    assert lifecycle == expected_lifecycle
    assert starts == ([] if capture_mode == "exception" else ["started"])
    assert outcome["completed"].returncode == (0 if capture_mode == "success" else 1)
    assert outcome["profile_capture"]["rocprof_started"] is (capture_mode != "exception")
    assert outcome["profile_capture"]["runner_completed"] is (capture_mode == "success")
    assert outcome["profile_capture"]["children_state_known"] is (capture_mode != "failure-unknown")
    assert outcome["profile_capture"]["cleanup_passed"] is (capture_mode != "failure-unknown")
    assert outcome["profile_diagnostics"]["executor_exception"] == ("RuntimeError" if capture_mode == "exception" else None)


@pytest.mark.parametrize("mode", (
    "missing-artifact", "tampered-artifact", "artifact-mode", "artifact-binding-field",
    "artifact-profiler-field", "artifact-command", "artifact-ref-hash", "artifact-ref-outside",
    "artifact-ref-dotdot", "artifact-ref-symlink", "artifact-ref-hardlink", "artifact-ref-mode",
    "tampered-failure", "failure-mode",
    "failure-profiler", "failure-command", "failure-effective-command", "failure-extra", "failure",
    "cleanup-failed", "children-nonempty",
    "timeout", "before-start-failure", "exception-before", "exception-after",
))
def test_default_profile_adapter_propagates_failure_safety(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    request, trusted_raw, starts, _ = default_profile_adapter_fixture(tmp_path, monkeypatch, mode)
    outcome = HARNESS.run_profile_capture(request, trusted_capture_raw=trusted_raw)
    capture = outcome["profile_capture"]
    assert outcome["completed"].returncode == 1
    assert capture["status"] == "failed"
    runner_completed = mode in {
        "missing-artifact", "tampered-artifact", "artifact-mode", "artifact-binding-field",
        "artifact-profiler-field", "artifact-command", "artifact-ref-hash", "artifact-ref-outside",
        "artifact-ref-dotdot", "artifact-ref-symlink", "artifact-ref-hardlink", "artifact-ref-mode",
    }
    children_state_known = runner_completed or mode in {"failure", "children-nonempty", "timeout", "before-start-failure", "exception-before"}
    expected_children = [4242] if mode == "children-nonempty" else []
    assert capture["runner_profiled"] is True
    assert capture["runner_completed"] is runner_completed
    assert outcome["profile_diagnostics"]["runner_finished"] is runner_completed
    assert capture["children_state_known"] is children_state_known
    assert capture["cleanup_passed"] is (children_state_known and expected_children == [])
    assert capture["children_remaining"] == expected_children
    if mode == "timeout":
        assert capture["timed_out"] is True
        assert capture["cleanup_passed"] is True and capture["children_remaining"] == []
    if mode == "before-start-failure":
        assert starts == []
        assert capture["rocprof_started"] is False and capture["runner_started"] is False
    if mode.startswith("exception-"):
        assert outcome["profile_diagnostics"]["executor_exception"] == "RuntimeError"


def test_profile_capture_uses_guarded_bytes_across_swap_restore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, trusted_raw, starts, trusted_path = default_profile_adapter_fixture(tmp_path, monkeypatch, "success")
    profiler = tmp_path / "profiler"; profiler.write_bytes(b"profiler"); profiler.chmod(0o755)
    python = tmp_path / "python"; python.write_bytes(b"python"); python.chmod(0o755)
    launcher = tmp_path / "launcher"; launcher.write_bytes(b"launcher")
    monkeypatch.setattr(HARNESS, "PROFILE_PROFILER", profiler)
    monkeypatch.setattr(HARNESS, "PROFILE_PROFILER_SHA", HARNESS.sha_bytes(profiler.read_bytes()))
    monkeypatch.setattr(HARNESS.LAUNCHER, "PYTHON", python)
    monkeypatch.setattr(HARNESS.LAUNCHER, "PYTHON_SHA", HARNESS.sha_bytes(python.read_bytes()))
    monkeypatch.setattr(HARNESS, "LAUNCHER_PATH", launcher)
    monkeypatch.setattr(HARNESS, "LAUNCHER_SHA", HARNESS.sha_bytes(launcher.read_bytes()))
    guard = HARNESS.ProfileTrustGuard()
    assert guard(request["contract"], "before-start")["passed"] is True
    marker = tmp_path / "malicious-executed"
    malicious_path = tmp_path / "malicious-capture.py"
    malicious_path.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n")
    monkeypatch.setattr(HARNESS, "PROFILE_CAPTURE_TOOL", malicious_path)
    outcome = HARNESS.run_profile_capture(request, trusted_capture_raw=guard.capture_tool_raw)
    monkeypatch.setattr(HARNESS, "PROFILE_CAPTURE_TOOL", trusted_path)
    assert guard(request["contract"], "capture-after")["passed"] is True
    assert outcome["completed"].returncode == 0 and starts == ["started"]
    assert not marker.exists()


@pytest.mark.parametrize(
    ("mode", "expected_children", "cleanup_allowed"),
    (("failure", [], True), ("cleanup-failed", [-1], False), ("children-nonempty", [4242], False)),
)
def test_default_profile_cleanup_state_controls_lock_substrate_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_children: list[int],
    cleanup_allowed: bool,
) -> None:
    request, trusted_raw, _, _ = default_profile_adapter_fixture(tmp_path, monkeypatch, mode)
    runtime = FakeRuntime()
    substrate = HARNESS.LockSubstrate(tmp_path / "lock-dir", tmp_path / "lock-dir/device.lock", (7, 8), (1, 2), {"passed": True})
    cleanup_calls: list[dict] = []

    def stopped(old_worker_pid, old_service_pid, run, control):
        value = runtime.stopped(old_worker_pid, old_service_pid, run, control)
        value["lock"].update({"source": "trusted_substrate", "substrate": {"directory": {"device": 7, "inode": 8}, "lock": {"device": 1, "inode": 2}}})
        return value

    def cleanup(value, run, *, runner_finished, runner_children):
        cleanup_calls.append({"runner_finished": runner_finished, "runner_children": runner_children})
        if not cleanup_allowed:
            raise HARNESS.HarnessError("synthetic lock retained for unsafe capture child state")
        return {"passed": True, "secret_material_recorded": False}

    dependencies = replace(
        runtime.dependencies(),
        stopped_observation=stopped,
        profile_capture=lambda actual_request: HARNESS.run_profile_capture(actual_request, trusted_capture_raw=trusted_raw),
        lock_substrate_prepare=lambda run: substrate,
        lock_substrate_cleanup=cleanup,
    )
    value = profile_ready(tmp_path)
    assert value["profile_diagnostic"] == request["contract"]
    code, evidence = HARNESS.execute_maintenance(value, tmp_path / "maintenance", dependencies)
    assert code == 1 and evidence["restore"]["passed"] is True
    assert cleanup_calls == [{"runner_finished": cleanup_allowed, "runner_children": expected_children}]
    assert evidence["lock_substrate_cleanup"]["passed"] is cleanup_allowed
    if not cleanup_allowed:
        assert "lock retained" in evidence["lock_substrate_cleanup"]["error"]
    assert evidence["capture"]["cleanup_passed"] is cleanup_allowed
    assert evidence["capture"]["children_remaining"] == ([] if mode != "children-nonempty" else [4242])
    assert evidence["capture"]["diagnostics"]["runner_finished"] is False


@pytest.mark.parametrize("raw_mode", ("schema-rejected", "malicious-cleanup-safe"))
def test_unvalidated_launcher_profile_outcome_never_authorizes_lock_unlink_and_still_restores(
    tmp_path: Path,
    raw_mode: str,
) -> None:
    runtime = FakeRuntime()
    substrate = HARNESS.LockSubstrate(
        tmp_path / "lock-dir",
        tmp_path / "lock-dir/device.lock",
        (7, 8),
        (1, 2),
        {"passed": True},
    )
    cleanup_calls: list[dict] = []

    def stopped(old_worker_pid, old_service_pid, run, control):
        value = runtime.stopped(old_worker_pid, old_service_pid, run, control)
        value["lock"].update({
            "source": "trusted_substrate",
            "substrate": {
                "directory": {"device": 7, "inode": 8},
                "lock": {"device": 1, "inode": 2},
            },
        })
        return value

    def rejecting_launcher(binding, *, profile_runner_executor):
        del binding
        target = {
            "path": str(tmp_path / "target-command-manifest.json"),
            "sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "identity": [1, 2, 3, 4, 1, 5, 6],
        }
        outcome = profile_runner_executor(
            ["/fake/runner", "profile"],
            dict(HARNESS.LAUNCHER.EXECUTE_ENV),
            lambda: None,
            target,
        )
        raw_capture = outcome["profile_capture"]
        raw_capture["unknown_schema_field"] = True
        if raw_mode == "malicious-cleanup-safe":
            raw_capture["cleanup_passed"] = True
            raw_capture["children_state_known"] = True
            raw_capture["children_remaining"] = []
            outcome["profile_diagnostics"]["runner_finished"] = True
        else:
            raw_capture["cleanup_passed"] = False
            raw_capture["children_state_known"] = False
            raw_capture["children_remaining"] = [4242]
            outcome["profile_diagnostics"]["runner_finished"] = False
        return 1, {
            "status": "failed",
            "safety": {
                "gpu_command_executed": "unknown",
                "model_load_executed": "unknown",
            },
            "failure": {
                "reason": "profile capture summary schema differs",
                "runner_started": True,
            },
            # The launcher rejected the callback result and deliberately did
            # not persist profile_capture/profile_diagnostics as evidence.
        }

    def cleanup(value, run, *, runner_finished, runner_children):
        cleanup_calls.append({
            "runner_finished": runner_finished,
            "runner_children": runner_children,
        })
        raise AssertionError("unvalidated profile state must not reach lock cleanup")

    dependencies = replace(
        runtime.dependencies(),
        stopped_observation=stopped,
        launcher_execute=rejecting_launcher,
        lock_substrate_prepare=lambda run: substrate,
        lock_substrate_cleanup=cleanup,
    )
    code, evidence = HARNESS.execute_maintenance(
        profile_ready(tmp_path),
        tmp_path / f"unvalidated-{raw_mode}",
        dependencies,
    )

    assert code == 1 and evidence["status"] == "failed"
    assert cleanup_calls == []
    assert evidence["launcher"]["profile_lifecycle_evidence_validated"] is False
    for key in (
        "runner_finished",
        "runner_not_started",
        "runner_started",
        "children_state_known",
        "children_remaining",
        "cleanup_passed",
    ):
        assert evidence["launcher"][key] == HARNESS.UNKNOWN_LIFECYCLE_STATE
    assert evidence["lock_substrate_cleanup"] == {
        "passed": False,
        "attempted": False,
        "reason": "trusted lock substrate retained because launcher profile lifecycle evidence is unverified",
        "runner_finished": HARNESS.UNKNOWN_LIFECYCLE_STATE,
        "runner_children": HARNESS.UNKNOWN_LIFECYCLE_STATE,
        "secret_material_recorded": False,
    }
    assert evidence["capture"]["authority"] == "diagnostic_only_unvalidated"
    assert evidence["capture"]["raw_profile_capture"]["cleanup_passed"] is (raw_mode == "malicious-cleanup-safe")
    assert evidence["capture"]["raw_profile_capture"]["children_remaining"] == ([] if raw_mode == "malicious-cleanup-safe" else [4242])
    for key in (
        "rocprof_started",
        "runner_start_known",
        "runner_started",
        "runner_completed",
        "runner_finished",
        "timed_out",
        "children_state_known",
        "children_remaining",
        "cleanup_passed",
    ):
        assert evidence["capture"][key] == HARNESS.UNKNOWN_LIFECYCLE_STATE
    assert evidence["restore"]["attempted"] is True
    assert evidence["restore"]["passed"] is True
    assert evidence["restore"]["lock_substrate_cleanup_passed"] is False
    assert runtime.active is True and runtime.epoch == 1


def test_profile_fake_maintenance_captures_child_then_restores(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    code, evidence = HARNESS.execute_maintenance(profile_ready(tmp_path), tmp_path / "profile-maintenance", runtime.dependencies())
    assert code == 0 and evidence["status"] == "passed"
    assert runtime.profile_captured is True
    assert runtime.trust_stages == ["before-start", "capture-before", "capture-after", "finalize-before"]
    assert evidence["execution_mode"] == "profile_diagnostic"
    assert evidence["sequence"] == ["sudo-prevalidate", "pre-stop-snapshot", "durable-marker", "service-stopped", "stopped-gates", "launcher", "profile-capture", "service-start", "service-restored"]
    assert runtime.events == ["launcher", "validator", "gates", "capture", "rocprof", "runner"]
    assert evidence["launcher"]["profile_runner_target"]["sha256"] == evidence["capture"]["target_manifest_sha256"]
    assert evidence["capture"]["runner_profiled"] is True
    assert evidence["capture"]["validator_profiled"] is False
    assert evidence["capture"]["gates_profiled"] is False
    assert evidence["process_counts"]["capture_tool"] == evidence["process_counts"]["rocprof"] == evidence["process_counts"]["launcher"] == 1
    assert evidence["restore"]["passed"] is True and runtime.active is True


def test_default_launcher_adapter_forwards_dedicated_profile_runner_executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict = {}

    def execute_bound(binding, evidence_output, runner_output, run_id, **kwargs):
        observed.update(
            {
                "binding": binding,
                "evidence_output": evidence_output,
                "runner_output": runner_output,
                "run_id": run_id,
                **kwargs,
            }
        )
        return 0, {"status": "passed"}

    monkeypatch.setattr(HARNESS.LAUNCHER, "execute_bound", execute_bound)
    executor = lambda runner_argv, base_env, on_started, target: {}
    binding = {
        "evidence_output": str(tmp_path / "launcher-evidence"),
        "runner_output": str(tmp_path / "runner-output"),
        "run_id": HARNESS.LAUNCHER.PROFILE_RUN_ID,
    }
    code, evidence = HARNESS._default_launcher_execute(binding, profile_runner_executor=executor)
    assert code == 0 and evidence["status"] == "passed"
    assert observed["profile_runner_executor"] is executor
    assert observed["trusted_launcher_sha"] == HARNESS.LAUNCHER_SHA
    assert callable(observed["gate_provider"])


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


@pytest.mark.parametrize("target", ("capture", "profiler", "launcher"))
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
    contract = HARNESS.ready_document({"path": str(SCRIPT), "commit": "1" * 40, "tree": "2" * 40, "git_blob": "3" * 40, "sha256": "4" * 64}, profile_diagnostic=True)["profile_diagnostic"]
    guard = HARNESS.ProfileTrustGuard()
    assert guard(contract, "before-start")["passed"] is True
    watched = {"capture": capture, "profiler": profiler, "launcher": launcher}[target]
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
    contract = HARNESS.ready_document({"path": str(SCRIPT), "commit": "1" * 40, "tree": "2" * 40, "git_blob": "3" * 40, "sha256": "4" * 64}, profile_diagnostic=True)["profile_diagnostic"]
    guard = HARNESS.ProfileTrustGuard()
    assert guard(contract, "before-start")["passed"] is True
    contract["execution_boundary"]["order"] = ["capture", "launcher", "runner"]
    with pytest.raises(HARNESS.HarnessError, match="execution boundary differs"):
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
    value = profile_ready(tmp_path)
    value["profile_diagnostic"]["output"]["directory"] = str(existing)
    with pytest.raises(HARNESS.HarnessError, match="profile capture output already exists"):
        HARNESS.execute_maintenance(value, tmp_path / "maintenance", runtime.dependencies())
    assert runtime.calls == [] and runtime.profile_captured is False


def test_base_and_profile_mode_cannot_be_cross_invoked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    monkeypatch.setattr(HARNESS, "load_ready_artifact", lambda path: ready(tmp_path))
    code = HARNESS.main(["--profile-diagnostic", "--ready-artifact", str(HARNESS.PROFILE_READY_PATH), "--mode", "execute", "--confirm-one-case", "--evidence-output", str(tmp_path / "cross")], dependencies=runtime.dependencies())
    assert code == 1 and runtime.calls == []


def test_immutable_ready_v1_artifact_is_stale_after_consumer_pin_update() -> None:
    historical_path = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-ready-v1/ready-binding.json"
    historical = json.loads(historical_path.read_text())
    assert historical["actual_eligible"] is True
    assert historical["qa_attestation_sha256"] != HARNESS.sha_bytes(
        HARNESS.pretty(HARNESS.QA_ATTESTATION)
    )


def test_stale_canonical_ready_dry_run_fails_before_processes(tmp_path: Path) -> None:
    output = tmp_path / "ready-dry-run"
    historical_path = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-ready-v1/ready-binding.json"
    code = HARNESS.main(["--mode", "dry-run", "--ready-artifact", str(historical_path), "--evidence-output", str(output)])
    assert code == 1
    assert not output.exists()


def test_stale_profile_ready_pin_is_rejected_before_dry_run_processes(tmp_path: Path) -> None:
    historical_path = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-ready-v1/ready-binding.json"
    historical = json.loads(historical_path.read_text())
    assert historical["profile_diagnostic"]["capture_tool"]["sha256"] != HARNESS.PROFILE_CAPTURE_SHA
    output = tmp_path / "profile-ready-dry-run"
    code = HARNESS.main(["--mode", "dry-run", "--profile-diagnostic", "--ready-artifact", str(historical_path), "--evidence-output", str(output)])
    assert code == 1
    assert not output.exists()


@pytest.mark.parametrize("failure", ("sudo-pre", "sudo-stop", "stop", "stopped-gate", "sudo-restore", "start", "health"))
def test_actual_cli_with_each_fake_gate_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    runtime = FakeRuntime(fail=failure)
    monkeypatch.setattr(HARNESS, "load_ready_artifact", lambda path: ready(tmp_path))
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


def _substrate_fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, install_failure: bool = False):
    lock = tmp_path / "run" / "ullm" / "r9700.lock"
    lock.parent.parent.mkdir()
    install = tmp_path / "install"
    install.write_bytes(b"trusted-install")
    install.chmod(0o755)
    rmdir = tmp_path / "rmdir"
    rmdir.write_bytes(b"trusted-rmdir")
    rmdir.chmod(0o755)
    monkeypatch.setattr(HARNESS.LAUNCHER, "LOCK_PATH", lock)
    monkeypatch.setattr(HARNESS, "INSTALL", install)
    monkeypatch.setattr(HARNESS, "INSTALL_SHA", HARNESS.sha_bytes(install.read_bytes()))
    monkeypatch.setattr(HARNESS, "RMDIR", rmdir)
    monkeypatch.setattr(HARNESS, "RMDIR_SHA", HARNESS.sha_bytes(rmdir.read_bytes()))
    calls: list[list[str]] = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[2] == str(install):
            if install_failure:
                return subprocess.CompletedProcess(argv, 1, b"", b"install failed")
            lock.parent.mkdir(mode=HARNESS.LOCK_SUBSTRATE_MODE)
            os.chmod(lock.parent, HARNESS.LOCK_SUBSTRATE_MODE)
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        if argv[2] == str(rmdir):
            lock.parent.rmdir()
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        raise AssertionError(argv)

    return lock, calls, run


def test_lock_substrate_prepare_records_pinned_install_and_exact_identities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock, calls, run = _substrate_fake(tmp_path, monkeypatch)
    substrate = HARNESS.prepare_lock_substrate(run)
    assert substrate.directory == lock.parent
    assert substrate.lock == lock
    assert substrate.evidence["pre"]["directory"]["present"] is False
    assert substrate.evidence["post"]["directory"]["mode"] == HARNESS.LOCK_SUBSTRATE_MODE
    assert substrate.evidence["post"]["lock"]["mode"] == HARNESS.LOCK_SUBSTRATE_LOCK_MODE
    assert substrate.evidence["post"]["lock"]["uid"] == os.getuid()
    command = substrate.evidence["commands"][0]
    assert command["argv"] == calls[0]
    assert command["executable"] == str(HARNESS.INSTALL)
    assert command["executable_sha256"] == HARNESS.INSTALL_SHA
    assert command["argv_sha256"] == HARNESS.sha_bytes(HARNESS.canonical(calls[0]))


@pytest.mark.parametrize("initial", ("directory", "file", "symlink"))
def test_lock_substrate_prepare_rejects_foreign_present_race(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, initial: str) -> None:
    lock, _, run = _substrate_fake(tmp_path, monkeypatch)
    if initial == "directory":
        lock.parent.mkdir()
    elif initial == "file":
        lock.parent.write_bytes(b"foreign")
    else:
        target = tmp_path / "foreign-target"
        target.mkdir()
        lock.parent.symlink_to(target, target_is_directory=True)
    with pytest.raises((HARNESS.HarnessError, HARNESS.LAUNCHER.LauncherError), match="absent|symlink"):
        HARNESS.prepare_lock_substrate(run)


def test_lock_substrate_prepare_rejects_install_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls, run = _substrate_fake(tmp_path, monkeypatch, install_failure=True)
    with pytest.raises(HARNESS.HarnessError, match="command failed"):
        HARNESS.prepare_lock_substrate(run)
    assert len(calls) == 1


def test_stopped_poll_classifies_lock_inode_swap_as_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock, _, run = _substrate_fake(tmp_path, monkeypatch)
    substrate = HARNESS.prepare_lock_substrate(run)
    lock.rename(tmp_path / "poll-original.lock")
    lock.write_bytes(b"replacement")
    lock.chmod(HARNESS.LOCK_SUBSTRATE_LOCK_MODE)
    monkeypatch.setattr(HARNESS, "_lock_holder_pids_for_stat", lambda metadata: ([], b""))
    observation = HARNESS._poll_lock_observation(substrate)
    assert observation["source"] == "replacement"
    assert observation["free"] is False
    decision, reason, classifications = HARNESS._stopped_observation_decision(
        {"captured_unix_ns": 1, "services": [], "worker_pids": [], "amd_smi_owners": [], "kfd_owners": [], "lock": observation, "vram": {}, "proc_cmdlines": [], "probes": [], "virtual_sources": {}, "secret_material_recorded": False},
        1,
        2,
        {"worker_pids": True, "amd_smi_owners": True, "kfd_owners": True, "lock": False},
    )
    assert decision == "terminal_failure" and "replacement" in (reason or "") and classifications["lock"] == "replacement"


def test_lock_substrate_cleanup_requires_same_inode_and_handles_runner_remove(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock, calls, run = _substrate_fake(tmp_path, monkeypatch)
    substrate = HARNESS.prepare_lock_substrate(run)
    lock.unlink()
    monkeypatch.setattr(HARNESS, "_lock_holder_pids_for_identity", lambda device, inode: ([], b""))
    cleanup = HARNESS.cleanup_lock_substrate(substrate, run, runner_finished=True)
    assert cleanup["passed"] is True and cleanup["source"] == "runner_removed"
    assert not lock.parent.exists()
    assert calls[-1][2] == str(HARNESS.RMDIR)


def test_lock_substrate_cleanup_unlinks_lock_kept_by_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock, _, run = _substrate_fake(tmp_path, monkeypatch)
    substrate = HARNESS.prepare_lock_substrate(run)
    lock.write_bytes(b'{"runner":"completed"}\n')
    monkeypatch.setattr(HARNESS, "_lock_holder_pids_for_stat", lambda metadata: ([], b""))
    cleanup = HARNESS.cleanup_lock_substrate(substrate, run, runner_finished=True)
    assert cleanup["passed"] is True and cleanup["source"] == "maintenance_unlinked"
    assert not lock.parent.exists()


def test_lock_substrate_cleanup_rejects_inode_swap_and_keeps_service_recovery_independent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock, _, run = _substrate_fake(tmp_path, monkeypatch)
    substrate = HARNESS.prepare_lock_substrate(run)
    lock.rename(tmp_path / "cleanup-original.lock")
    lock.write_bytes(b"replacement")
    lock.chmod(HARNESS.LOCK_SUBSTRATE_LOCK_MODE)
    with pytest.raises(HARNESS.HarnessError, match="replacement"):
        HARNESS.cleanup_lock_substrate(substrate, run, runner_finished=True)


def test_lock_substrate_cleanup_failure_still_attempts_service_start_and_records_recovery(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    prepared = {"value": False}
    cleaned = {"value": False}
    substrate_box: dict[str, HARNESS.LockSubstrate] = {}

    def prepare(run):
        prepared["value"] = True
        substrate = HARNESS.LockSubstrate(
            tmp_path / "run" / "ullm",
            tmp_path / "run" / "ullm" / "r9700.lock",
            (1, 2, 0o40750, 2, os.getuid(), os.getgid()),
            (1, 3, 0o100600, 1, os.getuid(), os.getgid(), 1),
            {"schema_version": "test", "secret_material_recorded": False},
        )
        substrate_box["value"] = substrate
        return substrate

    def cleanup(*args, **kwargs):
        cleaned["value"] = True
        raise HARNESS.HarnessError("synthetic cleanup failure")

    def stopped(old_worker_pid, old_service_pid, run, control):
        observation = runtime.stopped(old_worker_pid, old_service_pid, run, control)
        substrate = substrate_box["value"]
        observation["lock"]["source"] = "trusted_substrate"
        observation["lock"]["substrate"] = {
            "directory": {"device": substrate.directory_identity[0], "inode": substrate.directory_identity[1]},
            "lock": {"device": substrate.lock_identity[0], "inode": substrate.lock_identity[1]},
        }
        return observation

    dependencies = replace(runtime.dependencies(), stopped_observation=stopped, lock_substrate_prepare=prepare, lock_substrate_cleanup=cleanup)
    code, evidence = HARNESS.execute_maintenance(ready(tmp_path), tmp_path / "cleanup-failure", dependencies)
    assert code == 1 and evidence["status"] == "failed"
    assert prepared["value"] is True and cleaned["value"] is True
    assert evidence["failure"]["stage"] == "lock-substrate-cleanup"
    assert evidence["restore"]["attempted"] is True
    assert evidence["restore"]["passed"] is True
    assert runtime.active is True and runtime.epoch == 1
