#!/usr/bin/env python3
"""Single-use maintenance harness around the immutable AQ4 P2 smoke launcher."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "tools/launch-aq4-p2-resident-smoke.py"
SPEC = importlib.util.spec_from_file_location("aq4_p2_pinned_execute_launcher", LAUNCHER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("immutable launcher import failed")
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)

LAUNCHER_COMMIT = "bdb06083ca3646c8f934fea10dac691a6efd4626"
LAUNCHER_TREE = "7d7f9f30e90118ade4d0d66f058447b6b7777367"
LAUNCHER_GIT_BLOB = "5e7a3feebe82de13eeefbb3b99f95ad14244953b"
LAUNCHER_SHA = "ddd84b5b85dc303f381f44048b03eeb1e542bcf8173dfe506aeb0e7ebd235ee5"
RUNNER_COMMIT = "e93a2c162eb059cb2db883953d331f7a158d3a16"
RUNNER_SHA = "0d68f7141ea531e2200251597d601f9060b21b723faae2c8f96ae586c8cbeccc"
RUNNER_CLI_ANCESTOR = "ee341c019d873f7c250adbb81414d58b5285a454"
VALIDATOR_COMMIT = "82635456825503c535ce0b662e72a7a233d18c40"
B_COMMIT = "7e59baee0c1ac93a350da58a4292a84fbfde9f1c"
RESIDENT_COMMIT = "319d6187b29e877536aa5dfe80c02bde0c77ed7a"
READY_ROOT = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-ready-v1"
READY_PATH = READY_ROOT / "ready-binding.json"
HARNESS_TRUST_PATH = READY_ROOT / "harness-trust.json"
ATTESTATION_PATH = READY_ROOT / "qa-attestation.json"
MAINTENANCE_EVIDENCE = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-maintenance-evidence-v1"
DRY_RUN_EVIDENCE = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-ready-dry-run-v1"
PROFILE_READY_ROOT = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-ready-v1"
PROFILE_READY_PATH = PROFILE_READY_ROOT / "ready-binding.json"
PROFILE_HARNESS_TRUST_PATH = PROFILE_READY_ROOT / "harness-trust.json"
PROFILE_ATTESTATION_PATH = PROFILE_READY_ROOT / "qa-attestation.json"
PROFILE_MAINTENANCE_EVIDENCE = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-maintenance-evidence-v1"
PROFILE_DRY_RUN_EVIDENCE = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-ready-dry-run-v1"
PROFILE_CAPTURE_TOOL = ROOT / "tools/capture-aq4-p3-diagnostic-profile.py"
PROFILE_CAPTURE_COMMIT = "489183abba581332544d0d004338a2cee08a0d89"
PROFILE_CAPTURE_SHA = "be62caa7eee810cd6b33033eab15418b803ac1cee6559153e0cf7af446fa21f7"
PROFILE_PROFILER = Path("/opt/rocm-7.2.1/bin/rocprofv3")
PROFILE_PROFILER_SHA = "13060810d6b80653631b14f0f5e33ea160c2b79a6a3a4c6850142010b48b8ec8"
PROFILE_OUTPUT_DIRECTORY = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p3/aq4-p3-diagnostic-rocprof-capture-v1"
PROFILE_OUTPUT_NAME = "aq4-p3-diagnostic"
PROFILE_ARTIFACT = PROFILE_OUTPUT_DIRECTORY / "capture-artifact.json"
PROFILE_TIMEOUT_SECONDS = 1800
SERVICE = "ullm-openai.service"
WORKER = ROOT / "target/reasoning-v2/release/ullm-aq4-worker"
WORKER_SHA = "177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d"
PACKAGE_ROOT = Path("/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package")
PACKAGE_MANIFEST = PACKAGE_ROOT / "manifest.json"
PACKAGE_MANIFEST_SHA = "a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad"
PACKAGE_CONTENT_SHA = "a24774432d3f0b7f175dc761ef9a53df1fed901dd02f825e8542b17181f004b1"
GATEWAY_READY_URL = "http://172.20.0.1:8000/readyz"
OPENWEBUI_HEALTH_URL = "http://127.0.0.1:3000/health"
GATEWAY_READY_BODY = b'{"status":"ready"}'
OPENWEBUI_HEALTH_BODY = b'{"status":true}'
RUN_ID = LAUNCHER.EXECUTE_RUN_ID
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class HarnessError(ValueError):
    pass


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pretty(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def tree_hash(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise HarnessError("package root is invalid")
    paths: list[Path] = []
    for item in root.rglob("*"):
        if item.is_symlink():
            raise HarnessError("package contains a symlink")
        if item.is_file():
            paths.append(item)
    if not paths:
        raise HarnessError("package is empty")
    digest = hashlib.sha256()
    for item in sorted(paths, key=lambda value: value.relative_to(root).as_posix()):
        relative = item.relative_to(root).as_posix()
        file_digest = LAUNCHER.sha_file(item, f"package/{relative}")[0]
        digest.update(relative.encode()); digest.update(b"\0"); digest.update(bytes.fromhex(file_digest)); digest.update(b"\n")
    return digest.hexdigest()


def hash_regular_with_nlink(path: Path, label: str, expected_nlink: int) -> str:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != expected_nlink:
        raise HarnessError(f"{label} file identity differs")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mode, value.st_nlink, value.st_mtime_ns, value.st_ctime_ns)
    if identity(before) != identity(after) or identity(after) != identity(path.lstat()):
        raise HarnessError(f"{label} changed while hashing")
    return digest.hexdigest()


def _command(run: Callable[..., subprocess.CompletedProcess[bytes]], argv: list[str], label: str) -> tuple[subprocess.CompletedProcess[bytes], dict[str, Any]]:
    completed = run(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30)
    record = {"label": label, "argv": argv, "exit_code": completed.returncode, "stdout_sha256": sha_bytes(completed.stdout), "stderr_sha256": sha_bytes(completed.stderr), "captured_unix_ns": time.time_ns()}
    return completed, record


def _sudo_valid(run: Callable[..., subprocess.CompletedProcess[bytes]], label: str) -> dict[str, Any]:
    completed, record = _command(run, [str(LAUNCHER.SUDO), "-n", "-v"], label)
    if completed.returncode != 0 or completed.stdout or completed.stderr:
        raise HarnessError("sudo credential cache is not valid")
    return record


def _service_snapshot(run: Callable[..., subprocess.CompletedProcess[bytes]]) -> tuple[dict[str, Any], dict[str, Any]]:
    argv = [str(LAUNCHER.SYSTEMCTL), "show", SERVICE, "--property=ActiveState", "--property=SubState", "--property=MainPID", "--property=NRestarts", "--property=ControlGroup", "--no-pager"]
    completed, record = _command(run, argv, "service-running")
    try:
        values = dict(line.split("=", 1) for line in completed.stdout.decode().splitlines())
        main_pid = int(values["MainPID"]); restarts = int(values["NRestarts"])
    except (UnicodeError, ValueError, KeyError) as error:
        raise HarnessError("running service snapshot schema differs") from error
    if completed.returncode != 0 or completed.stderr or set(values) != {"ActiveState", "SubState", "MainPID", "NRestarts", "ControlGroup"} or values["ActiveState"] != "active" or values["SubState"] != "running" or main_pid <= 0 or restarts < 0 or values["ControlGroup"] != "/system.slice/ullm-openai.service":
        raise HarnessError("service is not healthy and active")
    return {"unit": SERVICE, "active_state": "active", "sub_state": "running", "main_pid": main_pid, "nrestarts": restarts, "control_group": values["ControlGroup"]}, record


def _worker_pid(run: Callable[..., subprocess.CompletedProcess[bytes]]) -> tuple[int, dict[str, Any]]:
    argv = [str(LAUNCHER.PGREP), "-f", "-x", f"{WORKER}.*"]
    completed, record = _command(run, argv, "worker-running")
    try:
        pids = [int(item) for item in completed.stdout.decode().splitlines() if item]
    except (UnicodeError, ValueError) as error:
        raise HarnessError("worker PID output differs") from error
    if completed.returncode != 0 or completed.stderr or len(pids) != 1 or pids[0] <= 0:
        raise HarnessError("running worker is not unique")
    return pids[0], record


def _gpu_identity(run: Callable[..., subprocess.CompletedProcess[bytes]]) -> tuple[dict[str, Any], dict[str, Any]]:
    completed, record = _command(run, [str(LAUNCHER.AMD_SMI), "list", "--json"], "gpu-identity")
    try:
        values = json.loads(completed.stdout)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise HarnessError("GPU identity JSON differs") from error
    matches = [item for item in values if isinstance(item, dict) and item.get("gpu") == 2 and item.get("bdf") == LAUNCHER.GPU_BDF and item.get("uuid") == LAUNCHER.GPU_UUID and item.get("kfd_id") == LAUNCHER.KFD_ID and item.get("node_id") == 2]
    if completed.returncode != 0 or completed.stderr or not isinstance(values, list) or len(matches) != 1:
        raise HarnessError("target GPU identity differs")
    return {"amd_smi_index": 2, "bdf": LAUNCHER.GPU_BDF, "uuid": LAUNCHER.GPU_UUID, "kfd_id": LAUNCHER.KFD_ID, "node_id": 2}, record


def default_http_probe(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        body = response.read(65537)
        status = response.status
    if len(body) > 65536:
        raise HarnessError("health response exceeds bound")
    return {"url": url, "status": status, "body": body}


def default_lock_busy() -> bool:
    descriptor = os.open(LAUNCHER.LOCK_PATH, os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def default_owner_probe(run: Callable[..., subprocess.CompletedProcess[bytes]], worker_pid: int) -> dict[str, Any]:
    completed, _ = _command(run, [str(LAUNCHER.AMD_SMI), "process", "--gpu", "2", "--general", "--json"], "gpu-owner")
    try:
        value = json.loads(completed.stdout)
        process_list = value[0]["process_list"]
        amd_pids = sorted(item["process_info"]["pid"] for item in process_list)
    except (UnicodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise HarnessError("GPU owner schema differs") from error
    kfd_pids = LAUNCHER._kfd_owners()
    if completed.returncode != 0 or completed.stderr or amd_pids != [worker_pid] or kfd_pids != [worker_pid]:
        raise HarnessError("restored worker does not uniquely own target GPU")
    return {"amd_smi": amd_pids, "kfd": kfd_pids}


@dataclass(frozen=True)
class Dependencies:
    run: Callable[..., subprocess.CompletedProcess[bytes]]
    http_probe: Callable[[str], dict[str, Any]]
    stopped_gates: Callable[[], dict[str, Any]]
    lock_busy: Callable[[], bool]
    owner_probe: Callable[[Callable[..., subprocess.CompletedProcess[bytes]], int], dict[str, Any]]
    package_hash: Callable[[Path], str]
    launcher_execute: Callable[[dict[str, Any]], tuple[int, dict[str, Any]]]
    profile_wrapper: Callable[[dict[str, Any]], dict[str, Any]]
    sleep: Callable[[float], None]


def _http_health(dependencies: Dependencies, url: str, expected: bytes) -> dict[str, Any]:
    value = dependencies.http_probe(url)
    if not isinstance(value, dict) or set(value) != {"url", "status", "body"} or value.get("url") != url or value.get("status") != 200 or value.get("body") != expected:
        raise HarnessError(f"health endpoint differs: {url}")
    return {"url": url, "status": 200, "body_sha256": sha_bytes(expected), "body_bytes": len(expected)}


def capture_running(dependencies: Dependencies, previous: dict[str, Any] | None = None) -> dict[str, Any]:
    service, service_record = _service_snapshot(dependencies.run)
    worker_pid, worker_record = _worker_pid(dependencies.run)
    gpu, gpu_record = _gpu_identity(dependencies.run)
    manifest_sha = LAUNCHER.sha_file(LAUNCHER.SERVED_MANIFEST, "active manifest")[0]
    worker_sha = hash_regular_with_nlink(WORKER, "active worker", 2)
    package_manifest_sha = LAUNCHER.sha_file(PACKAGE_MANIFEST, "package manifest")[0]
    package_content_sha = dependencies.package_hash(PACKAGE_ROOT)
    if manifest_sha != LAUNCHER.SERVED_SHA or worker_sha != WORKER_SHA or package_manifest_sha != PACKAGE_MANIFEST_SHA or package_content_sha != PACKAGE_CONTENT_SHA:
        raise HarnessError("production manifest/worker/package hash differs")
    if not dependencies.lock_busy():
        raise HarnessError("production service does not hold device lock")
    owners = dependencies.owner_probe(dependencies.run, worker_pid)
    gateway = _http_health(dependencies, GATEWAY_READY_URL, GATEWAY_READY_BODY)
    openwebui = _http_health(dependencies, OPENWEBUI_HEALTH_URL, OPENWEBUI_HEALTH_BODY)
    if previous is not None and (service["main_pid"] == previous["service"]["main_pid"] or worker_pid == previous["worker"]["pid"] or service["nrestarts"] != previous["service"]["nrestarts"] or service["control_group"] != previous["service"]["control_group"]):
        raise HarnessError("restored service epoch/NRestarts differs")
    return {
        "service": service, "worker": {"path": str(WORKER), "pid": worker_pid, "sha256": worker_sha}, "gpu": gpu,
        "owners": owners, "lock": {"path": str(LAUNCHER.LOCK_PATH), "busy": True},
        "hashes": {"served_manifest_sha256": manifest_sha, "worker_sha256": worker_sha, "package_manifest_sha256": package_manifest_sha, "package_content_sha256": package_content_sha},
        "health": {"gateway": gateway, "openwebui": openwebui}, "commands": [service_record, worker_record, gpu_record],
    }


def _default_launcher_execute(binding: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    gate_provider = lambda: LAUNCHER.collect_execute_gates(environment=dict(LAUNCHER.EXECUTE_ENV))
    return LAUNCHER.execute_bound(binding, Path(binding["evidence_output"]), Path(binding["runner_output"]), binding["run_id"], trusted_launcher_sha=LAUNCHER_SHA, gate_provider=gate_provider)


def validate_profile_wrapper(contract: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "ROCPROF_OUTPUT_PATH": str(PROFILE_OUTPUT_DIRECTORY),
        "ROCPROF_OUTPUT_FILE_NAME": PROFILE_OUTPUT_NAME,
        "ROCPROF_KERNEL_TRACE": "1",
        "ROCPROF_HIP_RUNTIME_API_TRACE": "1",
        "ROCPROF_MEMORY_COPY_TRACE": "1",
        "ROCPROF_MARKER_API_TRACE": "1",
    }
    observed = {name: os.environ.get(name) for name in expected}
    if contract.get("command") != profile_capture_command() or observed != expected:
        raise HarnessError("profile diagnostic must run through the exact rocprof wrapper")
    formats = os.environ.get("ROCPROF_OUTPUT_FORMAT", "").split(",")
    if "csv" not in formats or os.environ.get("ROCPROFILER_LIBRARY_CTOR") not in {"1", "true", "True"}:
        raise HarnessError("rocprof instrumentation environment differs")
    try:
        maps = Path("/proc/self/maps").read_text(errors="strict")
    except (OSError, UnicodeError) as error:
        raise HarnessError("rocprof process map is unavailable") from error
    required = ("librocprofiler-sdk-tool.so", "librocprofiler-sdk-roctx.so")
    if any(name not in maps for name in required):
        raise HarnessError("rocprof instrumentation libraries are not mapped")
    return {
        "validated": True,
        "profiler": {"path": str(PROFILE_PROFILER), "sha256": PROFILE_PROFILER_SHA},
        "environment_sha256": sha_bytes(canonical(observed)),
        "mapped_libraries": list(required),
    }


def default_dependencies() -> Dependencies:
    stopped_gates = lambda: LAUNCHER.collect_execute_gates(environment=dict(LAUNCHER.EXECUTE_ENV))
    return Dependencies(subprocess.run, default_http_probe, stopped_gates, default_lock_busy, default_owner_probe, tree_hash, _default_launcher_execute, validate_profile_wrapper, time.sleep)


def profile_harness_command() -> list[str]:
    return [
        str(LAUNCHER.PYTHON),
        str(Path(__file__).resolve()),
        "--mode",
        "execute",
        "--profile-diagnostic",
        "--ready-artifact",
        str(PROFILE_READY_PATH),
        "--evidence-output",
        str(PROFILE_MAINTENANCE_EVIDENCE),
        "--confirm-one-case",
    ]


def profile_capture_command() -> list[str]:
    return [
        str(LAUNCHER.PYTHON),
        str(PROFILE_CAPTURE_TOOL),
        "capture",
        "--profiler",
        str(PROFILE_PROFILER),
        "--profile-output-directory",
        str(PROFILE_OUTPUT_DIRECTORY),
        "--profile-output-name",
        PROFILE_OUTPUT_NAME,
        "--identity",
        str(LAUNCHER.INPUT_ROOT / "identity.json"),
        "--resident-summary",
        str(LAUNCHER.PROFILE_RUN_OUTPUT / "resident-batch.summary.json"),
        "--resident-raw",
        str(LAUNCHER.PROFILE_RUN_OUTPUT / f"{LAUNCHER.CASE_ID}.raw.json"),
        "--artifact",
        str(PROFILE_ARTIFACT),
        "--timeout",
        str(float(PROFILE_TIMEOUT_SECONDS)),
        "--runner-command",
        *profile_harness_command(),
    ]


def ready_launcher_binding(profile_diagnostic: bool = False) -> dict[str, Any]:
    value = copy.deepcopy(LAUNCHER.profile_execute_binding_document() if profile_diagnostic else LAUNCHER.execute_binding_document())
    value["status"] = "ready_for_explicit_execute"
    value["actual_eligible"] = True
    value["blocked_reasons"] = []
    live_path = LAUNCHER.PROFILE_LIVE_PREFLIGHT_PATH if profile_diagnostic else LAUNCHER.LIVE_PREFLIGHT_PATH
    value["live_preflight"] = {"required": True, "path": str(live_path), "sha256": None, "replaces_synthetic_preflight": True}
    LAUNCHER.validate_execute_binding(value, permit_test_live_preflight=True)
    return value


QA_ATTESTATION = {
    "schema_version": "ullm.aq4_p2_resident_execute_qa_attestation.v1", "status": "passed", "actual_executed": False,
    "test_count": 120, "runner_strict_negative_count": 18,
    "coverage": ["safety-success-start-failure-partial", "validator-runner-finalize-toctou", "identity-and-hash-bindings", "blocked-artifact-process-count-zero"],
    "launcher": {"commit": LAUNCHER_COMMIT, "sha256": LAUNCHER_SHA},
}


def ready_document(harness_identity: dict[str, str], *, profile_diagnostic: bool = False) -> dict[str, Any]:
    run_id = LAUNCHER.PROFILE_RUN_ID if profile_diagnostic else RUN_ID
    value = {
        "schema_version": "ullm.aq4_p2_resident_smoke_ready_binding.v1", "status": "ready_for_one_case", "actual_eligible": True, "promotion_eligible": False,
        "execution_mode": "profile_diagnostic" if profile_diagnostic else "one_case",
        "measurement_eligible": False if profile_diagnostic else None,
        "authorization": {"run_id": run_id, "one_case_only": True, "maximum_invocations": 1, "output_no_reuse": True, "external_service_stop_required": True, "rocprof_wrapper_required": profile_diagnostic},
        "launcher_binding": ready_launcher_binding(profile_diagnostic),
        "live_preflight_policy": {
            "pre_execution_sha256": None, "reason": "generated only after external service stop and all launcher live gates pass",
            "final_evidence_binding": {"path_and_sha256_required": True, "immutable_mode": "0444"},
            "schema_version": "ullm.aq4_p2_resident_live_preflight.v1", "run_id": run_id,
            "runtime_mapping": {"runtime_device_index": 1, "visible_token": "1", "amd_smi_index": 2, "bdf": LAUNCHER.GPU_BDF, "uuid": LAUNCHER.GPU_UUID, "kfd_id": LAUNCHER.KFD_ID, "node_id": 2},
            "vram": {"minimum_total_bytes": 30_000_000_000, "used_bytes": 0, "free_equals_total": True, "headroom_equals_total": True},
            "gates": ["sudo-n-v", "services-inactive", "worker-absent", "amd-owner-zero", "kfd-owner-zero", "lock-free", "exact-environment", "exact-probe-contract"],
        },
        "maintenance": {"service": SERVICE, "marker_required_before_stop": True, "restore_in_outer_finally": True, "same_pty_sudo_cache_required": True, "sudo_keepalive_seconds": 30, "secret_storage_forbidden": True},
        "trust": {
            "launcher": {"commit": LAUNCHER_COMMIT, "tree": LAUNCHER_TREE, "git_blob": LAUNCHER_GIT_BLOB, "sha256": LAUNCHER_SHA},
            "harness": harness_identity,
            "runner": {"commit": RUNNER_COMMIT, "sha256": RUNNER_SHA, "cli_ancestor_commit": RUNNER_CLI_ANCESTOR},
            "validator": {"commit": VALIDATOR_COMMIT, "sha256": LAUNCHER.VALIDATOR_SHA},
            "B": {"commit": B_COMMIT, "manifest_sha256": LAUNCHER.BINDING_MANIFEST_SHA},
            "resident": {"commit": RESIDENT_COMMIT, "sha256": LAUNCHER.RESIDENT_SHA},
            "production": {"manifest_sha256": LAUNCHER.SERVED_SHA, "worker_sha256": WORKER_SHA, "package_manifest_sha256": PACKAGE_MANIFEST_SHA, "package_content_sha256": PACKAGE_CONTENT_SHA},
        },
        "qa_attestation_sha256": sha_bytes(pretty(QA_ATTESTATION)),
    }
    if not profile_diagnostic:
        value.pop("measurement_eligible")
    if profile_diagnostic:
        command = profile_capture_command()
        value["profile_diagnostic"] = {
            "schema_version": "ullm.aq4_p3_diagnostic_rocprof_ready.v1",
            "status": "ready_for_one_profile_diagnostic",
            "measurement_eligible": False,
            "promotion_eligible": False,
            "maximum_invocations": 1,
            "output_no_reuse": True,
            "capture_tool": {"path": str(PROFILE_CAPTURE_TOOL), "commit": PROFILE_CAPTURE_COMMIT, "sha256": PROFILE_CAPTURE_SHA},
            "profiler": {"path": str(PROFILE_PROFILER), "resolved_path": str(PROFILE_PROFILER), "sha256": PROFILE_PROFILER_SHA},
            "command": command,
            "command_sha256": sha_bytes(canonical(command)),
            "output": {"directory": str(PROFILE_OUTPUT_DIRECTORY), "name": PROFILE_OUTPUT_NAME, "artifact": str(PROFILE_ARTIFACT), "must_not_exist_before_capture": True},
            "resident_evidence": {
                "identity": str(LAUNCHER.INPUT_ROOT / "identity.json"),
                "summary": str(LAUNCHER.PROFILE_RUN_OUTPUT / "resident-batch.summary.json"),
                "raw": str(LAUNCHER.PROFILE_RUN_OUTPUT / f"{LAUNCHER.CASE_ID}.raw.json"),
                "run_id": run_id,
                "resident_session_id_source": "resident_raw.resident.session_id",
                "case_id": LAUNCHER.CASE_ID,
                "case_sha256": LAUNCHER.CASE_SHA,
            },
            "roctx": value["launcher_binding"]["profile_diagnostic"],
        }
    return value


def _git_identity() -> dict[str, str]:
    path = Path(__file__).resolve(); relative = path.relative_to(ROOT); raw = path.read_bytes()
    values = []
    for revision in ("HEAD", "HEAD^{tree}", f"HEAD:{relative}"):
        completed = subprocess.run(["git", "rev-parse", revision], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if completed.returncode != 0 or completed.stderr:
            raise HarnessError("harness Git identity lookup failed")
        values.append(completed.stdout.decode("ascii").strip())
    committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if committed.returncode != 0 or committed.stderr or committed.stdout != raw:
        raise HarnessError("harness is not the exact committed HEAD blob")
    return {"path": str(path), "commit": values[0], "tree": values[1], "git_blob": values[2], "sha256": sha_bytes(raw)}


def prepare_ready_artifact(*, profile_diagnostic: bool = False) -> dict[str, Any]:
    root = PROFILE_READY_ROOT if profile_diagnostic else READY_ROOT
    LAUNCHER.ensure_directory_chain(root.parent, "ready artifact parent")
    if root.exists() or root.is_symlink():
        raise HarnessError("ready artifact already exists")
    root.mkdir(mode=0o700)
    harness_identity = _git_identity()
    value = ready_document(harness_identity, profile_diagnostic=profile_diagnostic)
    ready_raw = pretty(value); attestation_raw = pretty(QA_ATTESTATION)
    trust = {"schema_version": "ullm.aq4_p2_resident_maintenance_harness_trust.v1", "status": "ready_for_one_case", "execution_mode": value["execution_mode"], "actual_eligible": True, **harness_identity, "ready_binding_sha256": sha_bytes(ready_raw)}
    trust_raw = pretty(trust)
    for name, raw in (("ready-binding.json", ready_raw), ("qa-attestation.json", attestation_raw), ("harness-trust.json", trust_raw)):
        LAUNCHER.atomic_write(root, name, raw)
    sums = "".join(f"{sha_bytes(raw)}  {name}\n" for name, raw in (("harness-trust.json", trust_raw), ("qa-attestation.json", attestation_raw), ("ready-binding.json", ready_raw))).encode("ascii")
    LAUNCHER.atomic_write(root, "SHA256SUMS", sums)
    os.chmod(root, 0o555)
    return value


def load_ready_artifact(path: Path = READY_PATH) -> dict[str, Any]:
    if path == READY_PATH:
        root, trust_path, attestation_path, profile_diagnostic = READY_ROOT, HARNESS_TRUST_PATH, ATTESTATION_PATH, False
    elif path == PROFILE_READY_PATH:
        root, trust_path, attestation_path, profile_diagnostic = PROFILE_READY_ROOT, PROFILE_HARNESS_TRUST_PATH, PROFILE_ATTESTATION_PATH, True
    else:
        raise HarnessError("ready artifact path differs")
    if {item.name for item in root.iterdir()} != {"ready-binding.json", "qa-attestation.json", "harness-trust.json", "SHA256SUMS"}:
        raise HarnessError("ready artifact path/coverage differs")
    ready_raw, _ = LAUNCHER.read_regular(path, "ready binding"); value = LAUNCHER.parse_json(ready_raw, "ready binding")
    trust_raw, _ = LAUNCHER.read_regular(trust_path, "harness trust"); trust = LAUNCHER.parse_json(trust_raw, "harness trust")
    attestation_raw, _ = LAUNCHER.read_regular(attestation_path, "QA attestation"); attestation = LAUNCHER.parse_json(attestation_raw, "QA attestation")
    expected = ready_document({key: trust[key] for key in ("path", "commit", "tree", "git_blob", "sha256")}, profile_diagnostic=profile_diagnostic)
    if value != expected or attestation != QA_ATTESTATION or trust.get("schema_version") != "ullm.aq4_p2_resident_maintenance_harness_trust.v1" or trust.get("status") != "ready_for_one_case" or trust.get("execution_mode") != value["execution_mode"] or trust.get("actual_eligible") is not True or trust.get("ready_binding_sha256") != sha_bytes(ready_raw):
        raise HarnessError("ready artifact semantic binding differs")
    self_sha = LAUNCHER.sha_file(Path(__file__).resolve(), "maintenance harness self")[0]
    if trust.get("path") != str(Path(__file__).resolve()) or trust.get("sha256") != self_sha:
        raise HarnessError("maintenance harness self differs")
    expected_sums = "".join(f"{digest}  {name}\n" for name, digest in (("harness-trust.json", sha_bytes(trust_raw)), ("qa-attestation.json", sha_bytes(attestation_raw)), ("ready-binding.json", sha_bytes(ready_raw)))).encode("ascii")
    sums_raw, _ = LAUNCHER.read_regular(root / "SHA256SUMS", "ready sums")
    if sums_raw != expected_sums:
        raise HarnessError("ready artifact SHA256SUMS differs")
    return value


def _finalize(output: Path, evidence: dict[str, Any]) -> None:
    LAUNCHER.finalize_output(output, evidence)


def dry_run_ready(value: dict[str, Any], output: Path, ready_path: Path = READY_PATH) -> tuple[int, dict[str, Any]]:
    LAUNCHER.reject_symlink_components(output, "ready dry-run output", allow_missing_leaf=True)
    if output.exists() or output.is_symlink():
        raise HarnessError("ready dry-run output already exists")
    output.mkdir(mode=0o700)
    evidence = {"schema_version": "ullm.aq4_p2_resident_maintenance.v1", "status": "passed", "mode": "dry-run", "execution_mode": value["execution_mode"], "actual_eligible": value["actual_eligible"], "promotion_eligible": False, "run_id": value["authorization"]["run_id"], "process_counts": {"sudo": 0, "systemctl_stop": 0, "launcher": 0, "systemctl_start": 0, "rocprof": 0, "capture_tool": 0}, "service_touched": False, "gpu_command_executed": False, "model_load_executed": False, "ready_binding_sha256": LAUNCHER.sha_file(ready_path, "ready binding")[0]}
    if value["execution_mode"] == "profile_diagnostic":
        evidence["profile_diagnostic"] = {"command": value["profile_diagnostic"]["command"], "command_sha256": value["profile_diagnostic"]["command_sha256"], "wrapper_executed": False, "measurement_eligible": False, "promotion_eligible": False}
    _finalize(output, evidence)
    return 0, evidence


def execute_maintenance(value: dict[str, Any], output: Path, dependencies: Dependencies) -> tuple[int, dict[str, Any]]:
    profile_diagnostic = value.get("execution_mode") == "profile_diagnostic"
    run_id = value.get("authorization", {}).get("run_id")
    if value.get("status") != "ready_for_one_case" or value.get("actual_eligible") is not True or value.get("promotion_eligible") is not False or value.get("authorization", {}).get("maximum_invocations") != 1 or value.get("execution_mode") not in {"one_case", "profile_diagnostic"} or not isinstance(run_id, str):
        raise HarnessError("ready one-case authorization differs")
    if profile_diagnostic:
        if value.get("measurement_eligible") is not False or value.get("authorization", {}).get("rocprof_wrapper_required") is not True or not isinstance(value.get("profile_diagnostic"), dict):
            raise HarnessError("profile diagnostic authorization differs")
        wrapper = dependencies.profile_wrapper(value["profile_diagnostic"])
        if not isinstance(wrapper, dict) or wrapper.get("validated") is not True:
            raise HarnessError("profile diagnostic rocprof wrapper differs")
    elif value.get("authorization", {}).get("rocprof_wrapper_required") is not False or "profile_diagnostic" in value:
        raise HarnessError("normal one-case profile contract differs")
    else:
        wrapper = None
    for path, label in ((output, "maintenance evidence"), (Path(value["launcher_binding"]["runner_output"]), "runner output"), (Path(value["launcher_binding"]["evidence_output"]), "launcher evidence")):
        LAUNCHER.reject_symlink_components(path, label, allow_missing_leaf=True)
        if path.exists() or path.is_symlink():
            raise HarnessError(f"{label} already exists")
    output.mkdir(mode=0o700)
    evidence: dict[str, Any] = {"schema_version": "ullm.aq4_p2_resident_maintenance.v1", "status": "failed", "mode": "execute", "execution_mode": value["execution_mode"], "run_id": run_id, "promotion_eligible": False, "profile_wrapper": wrapper, "sequence": [], "commands": [], "pre_stop": None, "stopped_gates": None, "launcher": None, "restore": None, "failure": None, "process_counts": {"sudo": 0, "systemctl_stop": 0, "launcher": 0, "systemctl_start": 0}, "safety": {"service_touched": False, "service_stopped": False, "gpu_command_executed": False, "model_load_executed": False}, "secret_material_recorded": False}
    stop_attempted = False; pre: dict[str, Any] | None = None; code = 1; stage = "sudo-prevalidate"
    try:
        record = _sudo_valid(dependencies.run, "sudo-prevalidate"); evidence["commands"].append(record); evidence["process_counts"]["sudo"] += 1; evidence["sequence"].append("sudo-prevalidate")
        stage = "pre-stop-snapshot"; pre = capture_running(dependencies); evidence["pre_stop"] = pre; evidence["sequence"].append("pre-stop-snapshot")
        marker = {"schema_version": "ullm.aq4_p2_resident_maintenance_marker.v1", "run_id": run_id, "restore_required": True, "service": SERVICE, "pre_stop_sha256": sha_bytes(canonical(pre)), "created_unix_ns": time.time_ns()}
        LAUNCHER.atomic_write(output, "maintenance-marker.json", pretty(marker)); evidence["marker"] = {"path": str(output / "maintenance-marker.json"), "sha256": sha_bytes(pretty(marker))}; evidence["sequence"].append("durable-marker")
        stage = "service-stop"; evidence["commands"].append(_sudo_valid(dependencies.run, "sudo-before-stop")); evidence["process_counts"]["sudo"] += 1
        stop_attempted = True; evidence["safety"]["service_touched"] = True
        stopped, record = _command(dependencies.run, [str(LAUNCHER.SUDO), "-n", str(LAUNCHER.SYSTEMCTL), "stop", SERVICE], "service-stop"); evidence["commands"].append(record); evidence["process_counts"]["systemctl_stop"] = 1
        if stopped.returncode != 0 or stopped.stdout or stopped.stderr:
            raise HarnessError("service stop failed")
        evidence["safety"]["service_stopped"] = True; evidence["sequence"].append("service-stopped")
        stage = "stopped-gates"; gates = dependencies.stopped_gates(); evidence["stopped_gates"] = gates
        if not isinstance(gates, dict) or gates.get("passed") is not True or gates.get("services") != [{"unit": "ullm-openai.service", "active_state": "inactive", "sub_state": "dead", "main_pid": 0}, {"unit": "llama-qwen35-udq4.service", "active_state": "inactive", "sub_state": "dead", "main_pid": 0}] or gates.get("old_worker_pids") != [] or gates.get("amd_smi_owners") != [] or gates.get("kfd_owners") != [] or gates.get("lock", {}).get("free") is not True:
            raise HarnessError("stopped live gates differ")
        evidence["sequence"].append("stopped-gates")
        stage = "launcher"; evidence["process_counts"]["launcher"] = 1; evidence["sequence"].append("launcher")
        evidence["safety"]["gpu_command_executed"] = "unknown"; evidence["safety"]["model_load_executed"] = "unknown"
        launcher_code, launcher_evidence = dependencies.launcher_execute(value["launcher_binding"]); evidence["launcher"] = {"code": launcher_code, "status": launcher_evidence.get("status"), "safety": launcher_evidence.get("safety"), "failure": launcher_evidence.get("failure")}
        evidence["safety"]["gpu_command_executed"] = launcher_evidence.get("safety", {}).get("gpu_command_executed", "unknown")
        evidence["safety"]["model_load_executed"] = launcher_evidence.get("safety", {}).get("model_load_executed", "unknown")
        if launcher_code != 0 or launcher_evidence.get("status") != "passed":
            raise HarnessError("immutable launcher failed")
        code = 0
    except (HarnessError, LAUNCHER.LauncherError, OSError, ValueError, subprocess.SubprocessError) as error:
        evidence["failure"] = {"stage": stage, "reason": str(error), "launcher_started": evidence["process_counts"]["launcher"] == 1}
        code = 1
    finally:
        if stop_attempted:
            restore_error: str | None = None; post: dict[str, Any] | None = None
            try:
                evidence["commands"].append(_sudo_valid(dependencies.run, "sudo-before-restore")); evidence["process_counts"]["sudo"] += 1
                started, record = _command(dependencies.run, [str(LAUNCHER.SUDO), "-n", str(LAUNCHER.SYSTEMCTL), "start", SERVICE], "service-start"); evidence["commands"].append(record); evidence["process_counts"]["systemctl_start"] = 1; evidence["sequence"].append("service-start")
                if started.returncode != 0 or started.stdout or started.stderr:
                    raise HarnessError("service start failed")
                if pre is None:
                    raise HarnessError("pre-stop snapshot is absent during restore")
                last_error: Exception | None = None
                for _ in range(120):
                    try:
                        expected_previous = pre if evidence["safety"]["service_stopped"] else None
                        post = capture_running(dependencies, expected_previous); last_error = None; break
                    except (HarnessError, OSError, ValueError, subprocess.SubprocessError) as error:
                        last_error = error; dependencies.sleep(1.0)
                if last_error is not None or post is None:
                    raise HarnessError(f"service recovery validation failed: {last_error}")
                evidence["sequence"].append("service-restored")
            except (HarnessError, OSError, ValueError, subprocess.SubprocessError) as error:
                restore_error = str(error); code = 1
            evidence["restore"] = {"attempted": True, "passed": restore_error is None, "error": restore_error, "post_start": post}
        else:
            evidence["restore"] = {"attempted": False, "passed": True, "error": None, "post_start": None}
        evidence["status"] = "passed" if code == 0 and evidence["restore"]["passed"] else "failed"
        _finalize(output, evidence)
    return code, evidence


def main(argv: list[str] | None = None, *, dependencies: Dependencies | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-ready-artifact", action="store_true")
    parser.add_argument("--prepare-profile-ready-artifact", action="store_true")
    parser.add_argument("--mode", choices=("dry-run", "execute"), default="dry-run")
    parser.add_argument("--profile-diagnostic", action="store_true")
    parser.add_argument("--ready-artifact", type=Path, default=READY_PATH)
    parser.add_argument("--evidence-output", type=Path)
    parser.add_argument("--confirm-one-case", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.prepare_ready_artifact and args.prepare_profile_ready_artifact:
            raise HarnessError("ready artifact preparation modes are mutually exclusive")
        if args.prepare_ready_artifact or args.prepare_profile_ready_artifact:
            profile = args.prepare_profile_ready_artifact
            value = prepare_ready_artifact(profile_diagnostic=profile)
            artifact = PROFILE_READY_PATH if profile else READY_PATH
            print(json.dumps({"status": value["status"], "execution_mode": value["execution_mode"], "actual_eligible": value["actual_eligible"], "artifact": str(artifact)}, sort_keys=True)); return 0
        value = load_ready_artifact(args.ready_artifact)
        if args.profile_diagnostic != (value.get("execution_mode") == "profile_diagnostic"):
            raise HarnessError("profile diagnostic flag/artifact mode differs")
        if args.evidence_output is None:
            raise HarnessError("--evidence-output is required")
        if args.mode == "dry-run":
            code, evidence = dry_run_ready(value, args.evidence_output, args.ready_artifact)
        else:
            if not args.confirm_one_case:
                raise HarnessError("execute requires --confirm-one-case")
            code, evidence = execute_maintenance(value, args.evidence_output, dependencies or default_dependencies())
        print(json.dumps({"status": evidence["status"], "mode": evidence["mode"], "evidence": str(args.evidence_output / "launcher-evidence.json")}, sort_keys=True)); return code
    except (HarnessError, LAUNCHER.LauncherError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"AQ4 P2 maintenance harness failed: {error}", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
