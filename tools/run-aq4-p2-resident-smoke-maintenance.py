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
import pwd
import re
import signal
import stat
import subprocess
import sys
import tempfile
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

LAUNCHER_COMMIT = "73438ede07e0bc48d300aeb9986742532fbe730b"
LAUNCHER_TREE = "94b7033f2d5522374e552f5d5cd011e605d4b2df"
LAUNCHER_GIT_BLOB = "ec7faee66de04b515c6bdd9a8c72d5e84e00348b"
LAUNCHER_SHA = "56634a890aa18b3e0c8d4ac3b83800a7e1a6dea0a09156dbd1dfbc722388af77"
RUNNER_COMMIT = "084d2e71114857da77e4196061d18a1dfefd53e8"
RUNNER_SHA = "a3ba3e099a931682ffc441e268e56f77aeb5d95220e15fc9efce61aa13962f3b"
RUNNER_CLI_ANCESTOR = "ee341c019d873f7c250adbb81414d58b5285a454"
VALIDATOR_COMMIT = "3b7a8e4603ae79002bd5307ceee877f9dd2d8bfd"
B_COMMIT = "2c24b7670b52610f6b1db33633139023778b18e9"
RESIDENT_COMMIT = "084d2e71114857da77e4196061d18a1dfefd53e8"
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
PROFILE_TARGET_COMMAND_MANIFEST = PROFILE_READY_ROOT / "target-command-manifest.json"
PROFILE_MAINTENANCE_EVIDENCE = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-maintenance-evidence-v1"
PROFILE_DRY_RUN_EVIDENCE = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-ready-dry-run-v1"
PROFILE_CAPTURE_TOOL = ROOT / "tools/capture-aq4-p3-diagnostic-profile.py"
PROFILE_CAPTURE_COMMIT = "b4d515f9908136fa773f957775beab79edc3065d"
PROFILE_CAPTURE_TREE = "228bbbd0d05b8055640bd47dd3ed95952e504eef"
PROFILE_CAPTURE_GIT_BLOB = "5197f7a2607da2ec281ab8a013ce1476178bf1b1"
PROFILE_CAPTURE_SHA = "605a68d308bf4336fc96d23d0ba9f819799ef24b169e3f49ae6a377638ab6cf8"
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
GATEWAY_HEALTH_URL = "http://172.20.0.1:8000/healthz"
GATEWAY_MODELS_URL = "http://172.20.0.1:8000/v1/models"
OPENWEBUI_HEALTH_URL = "http://127.0.0.1:3000/health"
OPENWEBUI_CONTAINER_HEALTH_URL = "http://127.0.0.1:8080/health"
GATEWAY_READY_BODY = b'{"status":"ready"}'
GATEWAY_HEALTH_BODY = b'{"status":"ok"}'
OPENWEBUI_HEALTH_BODY = b'{"status":true}'
GATEWAY_MODEL_ID = "ullm-qwen3.5-9b-aq4"
DOCKER = Path("/usr/bin/docker")
DOCKER_SHA = "f8470ebe5d201284a9fb4e7e59326f116e5b764a8d0d9a47097c26d52257d446"
DOCKER_CLIENT_VERSION = "29.6.0"
DOCKER_CLIENT_API_VERSION = "1.55"
OPENWEBUI_CONTAINER_NAME = "open-webui"
OPENWEBUI_CONTAINER_ID = "41d5759a47c417ad4774cb4f19647393baa1a39b3987534a377776beb7d4977a"
OPENWEBUI_IMAGE_ID = "sha256:ef5ae4fbc06abb662eeefe87e584ea7c69e55838f5f08f637057b9108048b409"
OPENWEBUI_NETWORK_NAME = "open-webui-network"
OPENWEBUI_NETWORK_ID = "79bb7cfca31cb5d76978cbbb229c946662c137b93ea647b5ae6c205af9126dc8"
OPENWEBUI_CONTAINER_IP = "172.20.0.2"
OPENWEBUI_GATEWAY_IP = "172.20.0.1"
CONTAINER_CURL = "/usr/bin/curl"
CONTAINER_CURL_SHA = "58824afa640f512e07e895c0f2d1e1fe2c3fd1456acab689b6e4c1c75cef2593"
CONTAINER_CURL_VERSION = "7.88.1"
CONTAINER_CURL_VERSION_SHA = "cc1470d1e66681b5b01a8df907c86e2947cb30622d2419b9d7d10aea1ddcf7b8"
API_KEY_FILE = Path("/etc/ullm/openai-api-key")
API_KEY_UID = 0
API_KEY_GID = 1000
API_KEY_MODE = 0o640
HTTP_STATUS_MARKER = b"\n__ULLM_HTTP_STATUS__"
DOCKER_INSPECT_FORMAT = '{{.Id}}|{{.Image}}|{{.Name}}|{{.State.Status}}|{{.State.Running}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}|{{with index .NetworkSettings.Networks "open-webui-network"}}{{.NetworkID}}|{{.IPAddress}}|{{.Gateway}}{{end}}'
STOP_POLL_TIMEOUT_SECONDS = 30.0
STOP_POLL_INITIAL_INTERVAL_SECONDS = 0.25
STOP_POLL_MAX_INTERVAL_SECONDS = 1.0
STOP_POLL_STABLE_OBSERVATIONS = 2
STOP_POLL_SUDO_KEEPALIVE_SECONDS = 10.0
STOP_POLL_PROBE_TIMEOUT_SECONDS = 2.0
RUN_ID = LAUNCHER.EXECUTE_RUN_ID
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
LOCK_SUBSTRATE_OWNER = "homelab1"
LOCK_SUBSTRATE_DIRECTORY = Path("/run/ullm")
LOCK_SUBSTRATE_MODE = 0o750
LOCK_SUBSTRATE_LOCK_MODE = 0o600
INSTALL = Path("/usr/bin/install")
INSTALL_SHA = "0e328ae109217200da3207ece12514b867d44fb90b444958b4d64b6007736f33"
RMDIR = Path("/usr/bin/rmdir")
RMDIR_SHA = "2450cf2f4eaad71378cfdc6ac6da5cd6b40a6aa772766ae9040d32bf2ee45193"


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


def _command(run: Callable[..., subprocess.CompletedProcess[bytes]], argv: list[str], label: str, *, timeout: float = 30.0) -> tuple[subprocess.CompletedProcess[bytes], dict[str, Any]]:
    completed = run(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)
    record = {"label": label, "argv": argv, "exit_code": completed.returncode, "stdout_sha256": sha_bytes(completed.stdout), "stderr_sha256": sha_bytes(completed.stderr), "captured_unix_ns": time.time_ns()}
    return completed, record


def _sudo_valid(run: Callable[..., subprocess.CompletedProcess[bytes]], label: str, *, timeout: float = 30.0) -> dict[str, Any]:
    completed, record = _command(run, [str(LAUNCHER.SUDO), "-n", "-v"], label, timeout=timeout)
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


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _lock_substrate_directory() -> Path:
    """Return the fixed directory containing the production device lock.

    Deriving this from the launcher lock path keeps test fakes and an explicitly
    bound launcher path on the same substrate while production remains exactly
    ``/run/ullm``.
    """

    directory = LAUNCHER.LOCK_PATH.parent
    if directory != LOCK_SUBSTRATE_DIRECTORY:
        # A test may bind a temporary lock path, but a real run must never move
        # the trusted substrate away from the production location.
        return directory
    return LOCK_SUBSTRATE_DIRECTORY


def _lock_owner_ids() -> tuple[int, int]:
    try:
        owner = pwd.getpwnam(LOCK_SUBSTRATE_OWNER)
    except KeyError as error:
        raise HarnessError("trusted lock substrate owner is unavailable") from error
    if owner.pw_uid == 0 or owner.pw_gid == 0:
        raise HarnessError("trusted lock substrate owner must be nonroot")
    return owner.pw_uid, owner.pw_gid


def _substrate_identity(metadata: os.stat_result, *, directory: bool = False) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
    )


def _substrate_metadata(metadata: os.stat_result) -> dict[str, Any]:
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
        "nlink": metadata.st_nlink,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "regular": stat.S_ISREG(metadata.st_mode),
        "directory": stat.S_ISDIR(metadata.st_mode),
    }


@dataclass(frozen=True)
class LockSubstrate:
    directory: Path
    lock: Path
    directory_identity: tuple[int, ...]
    lock_identity: tuple[int, ...]
    evidence: dict[str, Any]


def _pinned_tool_sha(path: Path, expected: str, label: str) -> str:
    LAUNCHER.reject_symlink_components(path, label)
    observed, _ = LAUNCHER.sha_file(path, label)
    if observed != expected:
        raise HarnessError(f"{label} SHA differs")
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or not os.access(path, os.X_OK):
        raise HarnessError(f"{label} executable identity differs")
    return observed


def _substrate_dir_state(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return {"path": str(path), "present": False, "symlink": False, "metadata": None, "empty": None}
    if stat.S_ISLNK(metadata.st_mode):
        raise HarnessError("trusted lock substrate directory is a symlink")
    empty = stat.S_ISDIR(metadata.st_mode) and not any(path.iterdir())
    return {"path": str(path), "present": True, "symlink": False, "metadata": _substrate_metadata(metadata), "empty": empty}


def _verify_substrate_directory(
    path: Path,
    owner_uid: int,
    owner_gid: int,
    *,
    expected_identity: tuple[int, ...] | None = None,
    empty: bool,
) -> os.stat_result:
    LAUNCHER.reject_symlink_components(path, "trusted lock substrate directory")
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != LOCK_SUBSTRATE_MODE
        or metadata.st_nlink != 2
        or metadata.st_uid != owner_uid
        or metadata.st_gid != owner_gid
        or (expected_identity is not None and _substrate_identity(metadata, directory=True) != expected_identity)
    ):
        raise HarnessError("trusted lock substrate directory identity differs")
    if empty and any(path.iterdir()):
        raise HarnessError("trusted lock substrate directory is not empty")
    return metadata


def _verify_substrate_lock(
    path: Path,
    owner_uid: int,
    owner_gid: int,
    *,
    expected_identity: tuple[int, ...] | None = None,
) -> os.stat_result:
    LAUNCHER.reject_symlink_components(path, "trusted lock substrate lock")
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != LOCK_SUBSTRATE_LOCK_MODE
        or metadata.st_nlink != 1
        or metadata.st_uid != owner_uid
        or metadata.st_gid != owner_gid
        or metadata.st_uid == 0
        or (expected_identity is not None and _substrate_identity(metadata) != expected_identity)
    ):
        raise HarnessError("trusted lock substrate lock identity differs")
    return metadata


def _command_with_pinned_executable(
    run: Callable[..., subprocess.CompletedProcess[bytes]],
    argv: list[str],
    label: str,
    executable: Path,
    executable_sha: str,
) -> dict[str, Any]:
    observed_sha = _pinned_tool_sha(executable, executable_sha, label)
    completed, record = _command(run, argv, label)
    record["executable"] = str(executable)
    record["executable_sha256"] = observed_sha
    record["sha256"] = observed_sha
    record["argv_sha256"] = sha_bytes(canonical(argv))
    if completed.returncode != 0 or completed.stdout or completed.stderr:
        raise HarnessError(f"{label} command failed")
    return record


def prepare_lock_substrate(
    run: Callable[..., subprocess.CompletedProcess[bytes]],
) -> LockSubstrate:
    """Create the non-root lock substrate after the production service stops."""

    directory = _lock_substrate_directory()
    lock = LAUNCHER.LOCK_PATH
    owner_uid, owner_gid = _lock_owner_ids()
    LAUNCHER.reject_symlink_components(directory, "trusted lock substrate directory", allow_missing_leaf=True)
    pre_directory = _substrate_dir_state(directory)
    if pre_directory["present"]:
        raise HarnessError("trusted lock substrate directory must be absent before install")
    try:
        pre_lock = lock.lstat()
    except FileNotFoundError:
        pre_lock = None
    if pre_lock is not None:
        raise HarnessError("trusted lock substrate lock must be absent before install")
    install_argv = [str(LAUNCHER.SUDO), "-n", str(INSTALL), "-d", "-o", LOCK_SUBSTRATE_OWNER, "-g", LOCK_SUBSTRATE_OWNER, "-m", f"{LOCK_SUBSTRATE_MODE:04o}", str(directory)]
    install_record = _command_with_pinned_executable(run, install_argv, "lock-substrate-install", INSTALL, INSTALL_SHA)
    directory_metadata = _verify_substrate_directory(directory, owner_uid, owner_gid, empty=True)
    directory_descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        if _substrate_identity(os.fstat(directory_descriptor), directory=True) != _substrate_identity(directory_metadata, directory=True):
            raise HarnessError("trusted lock substrate directory changed before lock creation")
        descriptor = os.open(
            lock.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            LOCK_SUBSTRATE_LOCK_MODE,
            dir_fd=directory_descriptor,
        )
    except FileExistsError as error:
        raise HarnessError("trusted lock substrate lock appeared during creation") from error
    finally:
        try:
            if descriptor >= 0:
                try:
                    os.fchmod(descriptor, LOCK_SUBSTRATE_LOCK_MODE)
                finally:
                    os.close(descriptor)
        finally:
            os.close(directory_descriptor)
    lock_metadata = _verify_substrate_lock(lock, owner_uid, owner_gid)
    # A second directory check closes the install/create race and records the
    # exact directory and lock identities used by all subsequent gates.
    directory_metadata = _verify_substrate_directory(directory, owner_uid, owner_gid, empty=False)
    evidence = {
        "schema_version": "ullm.aq4_p2_trusted_lock_substrate.v1",
        "owner": {"name": LOCK_SUBSTRATE_OWNER, "uid": owner_uid, "gid": owner_gid, "nonroot": True},
        "directory": str(directory),
        "lock": str(lock),
        "pre": {"directory": pre_directory, "lock": {"path": str(lock), "present": False}},
        "post": {
            "directory": _substrate_metadata(directory_metadata) | {"empty": False},
            "lock": _substrate_metadata(lock_metadata),
        },
        "commands": [install_record],
        "identity": {
            "directory": {"device": directory_metadata.st_dev, "inode": directory_metadata.st_ino},
            "lock": {"device": lock_metadata.st_dev, "inode": lock_metadata.st_ino},
        },
        "secret_material_recorded": False,
    }
    return LockSubstrate(directory, lock, _substrate_identity(directory_metadata, directory=True), _substrate_identity(lock_metadata), evidence)


def _api_key_snapshot() -> tuple[bytes, tuple[int, ...]]:
    descriptor = -1
    try:
        descriptor = os.open(
            API_KEY_FILE,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        entry_before = API_KEY_FILE.lstat()
        if (
            _file_identity(before) != _file_identity(entry_before)
            or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != API_KEY_MODE
            or before.st_nlink != 1
            or before.st_uid != API_KEY_UID
            or before.st_gid != API_KEY_GID
            or not 17 <= before.st_size <= 4096
        ):
            raise HarnessError("gateway API key file identity differs")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(4097 - total, 1024)):
            chunks.append(chunk)
            total += len(chunk)
            if total > 4096:
                raise HarnessError("gateway API key exceeds bound")
        after = os.fstat(descriptor)
        entry_after = API_KEY_FILE.lstat()
        if _file_identity(before) != _file_identity(after) or _file_identity(before) != _file_identity(entry_after):
            raise HarnessError("gateway API key changed while reading")
        raw = b"".join(chunks)
        if raw.endswith(b"\r\n"):
            secret = raw[:-2]
        elif raw.endswith(b"\n"):
            secret = raw[:-1]
        else:
            secret = raw
        if not 16 <= len(secret) <= 4094 or any(item in secret for item in (b"\r", b"\n", b"\x00")):
            raise HarnessError("gateway API key is not one bounded line")
        return secret, _file_identity(before)
    except HarnessError:
        raise
    except OSError as error:
        raise HarnessError("gateway API key snapshot failed") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _container_command(
    run: Callable[..., subprocess.CompletedProcess[bytes]],
    argv: list[str],
    label: str,
    *,
    stdin_secret: bytes | None = None,
) -> tuple[subprocess.CompletedProcess[bytes], dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "cwd": ROOT,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "check": False,
        "timeout": 10,
    }
    if stdin_secret is None:
        kwargs["stdin"] = subprocess.DEVNULL
    else:
        kwargs["input"] = stdin_secret
    completed = run(argv, **kwargs)
    if len(completed.stdout) > 65536 or len(completed.stderr) > 65536:
        raise HarnessError("container health command output exceeds bound")
    command_raw = "\0".join(argv).encode("utf-8")
    if stdin_secret is not None:
        candidates = [stdin_secret]
        prefix = b"Authorization: Bearer "
        if stdin_secret.startswith(prefix):
            candidates.append(stdin_secret[len(prefix):].rstrip(b"\r\n"))
        if any(candidate and (candidate in command_raw or candidate in completed.stdout or candidate in completed.stderr) for candidate in candidates):
            raise HarnessError("secret material escaped container health probe")
    record = {
        "label": label,
        "argv": argv,
        "exit_code": completed.returncode,
        "stdout_sha256": sha_bytes(completed.stdout),
        "stderr_sha256": sha_bytes(completed.stderr),
        "stdin": "authorization_header_redacted" if stdin_secret is not None else "devnull",
        "captured_unix_ns": time.time_ns(),
    }
    return completed, record


def _curl_command(container_id: str, url: str, *, authenticated: bool) -> list[str]:
    command = [str(DOCKER), "exec"]
    if authenticated:
        command.append("-i")
    command.extend(
        [
            container_id,
            CONTAINER_CURL,
            "--silent",
            "--show-error",
            "--fail-with-body",
            "--max-time",
            "5",
            "--request",
            "GET",
            "--header",
            "Accept: application/json",
        ]
    )
    if authenticated:
        command.extend(["--header", "@-"])
    command.extend(["--write-out", "\n__ULLM_HTTP_STATUS__%{http_code}\n", "--url", url])
    return command


def _parse_curl_response(
    completed: subprocess.CompletedProcess[bytes],
    url: str,
    expected: bytes | None,
    *,
    authenticated: bool,
) -> dict[str, Any]:
    body, marker, status_raw = completed.stdout.rpartition(HTTP_STATUS_MARKER)
    if (
        completed.returncode != 0
        or completed.stderr
        or marker != HTTP_STATUS_MARKER
        or status_raw.strip() != b"200"
        or len(body) > 65536
        or (expected is not None and body != expected)
    ):
        raise HarnessError(f"container health endpoint differs: {url}")
    return {
        "url": url,
        "status": 200,
        "body_sha256": sha_bytes(body),
        "body_bytes": len(body),
        "authenticated": authenticated,
    }


def _container_process_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    if any(
        not isinstance(record, dict)
        or not isinstance(record.get("argv"), list)
        or any(not isinstance(item, str) for item in record["argv"])
        for record in records
    ):
        raise HarnessError("container health command count input differs")
    docker_exec = sum(record["argv"][1:2] == ["exec"] for record in records)
    container_curl_total = sum(
        record["argv"][1:2] == ["exec"] and CONTAINER_CURL in record["argv"]
        for record in records
    )
    container_curl_version = sum(
        record["argv"][1:2] == ["exec"]
        and record["argv"][-2:] == [CONTAINER_CURL, "--version"]
        for record in records
    )
    container_curl_endpoint = container_curl_total - container_curl_version
    return {
        "docker": len(records),
        "docker_exec": docker_exec,
        # Compatibility total: this intentionally preserves the historical
        # argv-membership definition, including `sha256sum /usr/bin/curl`.
        "container_curl": container_curl_total,
        "container_curl_total": container_curl_total,
        "container_curl_version": container_curl_version,
        "container_curl_endpoint": container_curl_endpoint,
    }


class ContainerHealthGuard:
    def __init__(self) -> None:
        self.snapshot = LAUNCHER.Snapshot()
        self.initialized = False
        self.api_key_identity: tuple[int, ...] | None = None
        self.api_key: bytes | None = None

    def _inspect(self, run: Callable[..., subprocess.CompletedProcess[bytes]], records: list[dict[str, Any]]) -> dict[str, Any]:
        command = [str(DOCKER), "inspect", "--type", "container", "--format", DOCKER_INSPECT_FORMAT, OPENWEBUI_CONTAINER_NAME]
        completed, record = _container_command(run, command, "openwebui-container-inspect")
        records.append(record)
        try:
            fields = completed.stdout.decode("ascii").strip().split("|")
        except UnicodeError as error:
            raise HarnessError("OpenWebUI container identity is not ASCII") from error
        expected = [
            OPENWEBUI_CONTAINER_ID,
            OPENWEBUI_IMAGE_ID,
            f"/{OPENWEBUI_CONTAINER_NAME}",
            "running",
            "true",
            "healthy",
            OPENWEBUI_NETWORK_ID,
            OPENWEBUI_CONTAINER_IP,
            OPENWEBUI_GATEWAY_IP,
        ]
        if completed.returncode != 0 or completed.stderr or fields != expected:
            raise HarnessError("OpenWebUI container/image/network identity differs")
        return {
            "id": fields[0],
            "image_id": fields[1],
            "name": OPENWEBUI_CONTAINER_NAME,
            "status": fields[3],
            "running": True,
            "health": fields[5],
            "network": {"name": OPENWEBUI_NETWORK_NAME, "id": fields[6], "ip": fields[7], "gateway": fields[8]},
        }

    def __call__(self, run: Callable[..., subprocess.CompletedProcess[bytes]]) -> dict[str, Any]:
        if not self.initialized:
            self.snapshot.file(DOCKER, DOCKER_SHA, "Docker executable")
            self.initialized = True
        self.snapshot.verify()
        secret, secret_identity = _api_key_snapshot()
        if self.api_key is None:
            self.api_key = secret
            self.api_key_identity = secret_identity
        elif secret != self.api_key or secret_identity != self.api_key_identity:
            raise HarnessError("gateway API key changed across maintenance boundary")
        records: list[dict[str, Any]] = []
        version_command = [str(DOCKER), "version", "--format", "{{json .Client}}"]
        version, record = _container_command(run, version_command, "docker-client-version")
        records.append(record)
        try:
            version_value = LAUNCHER.parse_json(version.stdout, "Docker client version")
        except LAUNCHER.LauncherError as error:
            raise HarnessError("Docker client version JSON differs") from error
        if (
            version.returncode != 0
            or version.stderr
            or not isinstance(version_value, dict)
            or version_value.get("Version") != DOCKER_CLIENT_VERSION
            or version_value.get("ApiVersion") != DOCKER_CLIENT_API_VERSION
            or version_value.get("Os") != "linux"
            or version_value.get("Arch") != "amd64"
        ):
            raise HarnessError("Docker client version identity differs")
        container = self._inspect(run, records)
        container_id = container["id"]
        curl_version_command = [str(DOCKER), "exec", container_id, CONTAINER_CURL, "--version"]
        curl_version, record = _container_command(run, curl_version_command, "container-curl-version")
        records.append(record)
        try:
            curl_line = curl_version.stdout.decode("ascii").splitlines()[0]
        except (UnicodeError, IndexError) as error:
            raise HarnessError("container curl version output differs") from error
        if (
            curl_version.returncode != 0
            or curl_version.stderr
            or sha_bytes(curl_version.stdout) != CONTAINER_CURL_VERSION_SHA
            or not curl_line.startswith(f"curl {CONTAINER_CURL_VERSION} ")
        ):
            raise HarnessError("container curl version identity differs")
        curl_sha_command = [str(DOCKER), "exec", container_id, "/usr/bin/sha256sum", CONTAINER_CURL]
        curl_sha, record = _container_command(run, curl_sha_command, "container-curl-sha256")
        records.append(record)
        if curl_sha.returncode != 0 or curl_sha.stderr or curl_sha.stdout != f"{CONTAINER_CURL_SHA}  {CONTAINER_CURL}\n".encode("ascii"):
            raise HarnessError("container curl SHA-256 differs")
        endpoints: dict[str, Any] = {}
        for name, url, expected in (
            ("gateway_healthz", GATEWAY_HEALTH_URL, GATEWAY_HEALTH_BODY),
            ("gateway_readyz", GATEWAY_READY_URL, GATEWAY_READY_BODY),
            ("openwebui_health", OPENWEBUI_CONTAINER_HEALTH_URL, OPENWEBUI_HEALTH_BODY),
        ):
            command = _curl_command(container_id, url, authenticated=False)
            completed, record = _container_command(run, command, name)
            records.append(record)
            endpoints[name] = _parse_curl_response(completed, url, expected, authenticated=False)
        models_command = _curl_command(container_id, GATEWAY_MODELS_URL, authenticated=True)
        authorization = b"Authorization: Bearer " + secret + b"\n"
        models, record = _container_command(run, models_command, "gateway-models", stdin_secret=authorization)
        records.append(record)
        endpoints["gateway_models"] = _parse_curl_response(models, GATEWAY_MODELS_URL, None, authenticated=True)
        try:
            models_value = LAUNCHER.parse_json(models.stdout.rpartition(HTTP_STATUS_MARKER)[0], "gateway model list")
        except LAUNCHER.LauncherError as error:
            raise HarnessError("gateway model list JSON differs") from error
        expected_models = {"object": "list", "data": [{"id": GATEWAY_MODEL_ID, "object": "model", "owned_by": "ullm"}]}
        if models_value != expected_models:
            raise HarnessError("gateway model identity differs")
        endpoints["gateway_models"]["model_id"] = GATEWAY_MODEL_ID
        if self._inspect(run, records) != container:
            raise HarnessError("OpenWebUI container identity changed during health probe")
        docker_metadata = DOCKER.lstat()
        self.snapshot.verify()
        result = {
            "transport": "docker-exec-container-network-namespace",
            "docker": {
                "path": str(DOCKER),
                "sha256": DOCKER_SHA,
                "device": docker_metadata.st_dev,
                "inode": docker_metadata.st_ino,
                "size": docker_metadata.st_size,
                "mode": stat.S_IMODE(docker_metadata.st_mode),
                "client_version": DOCKER_CLIENT_VERSION,
                "client_api_version": DOCKER_CLIENT_API_VERSION,
            },
            "container": container,
            "curl": {"path": CONTAINER_CURL, "sha256": CONTAINER_CURL_SHA, "version": CONTAINER_CURL_VERSION},
            "endpoints": endpoints,
            "commands": records,
            "process_counts": _container_process_counts(records),
            "secret_material_recorded": False,
        }
        serialized = canonical(result)
        if secret in serialized or authorization.rstrip(b"\n") in serialized:
            raise HarnessError("secret material escaped container health evidence")
        return result


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


def _lock_holder_pids_for_identity(device: int, inode: int) -> tuple[list[int], bytes]:
    try:
        locks_raw = Path("/proc/locks").read_bytes()
    except OSError as error:
        raise HarnessError("trusted lock substrate holder source failed") from error
    if len(locks_raw) > 4 * 1024 * 1024:
        raise HarnessError("trusted lock substrate holder source exceeds bound")
    lock_id = f"{os.major(device):02x}:{os.minor(device):02x}:{inode}"
    holder_pids: set[int] = set()
    try:
        for line in locks_raw.decode("ascii").splitlines():
            fields = line.split()
            if len(fields) >= 6 and fields[5].lower() == lock_id.lower():
                holder_pids.add(int(fields[4]))
    except (UnicodeError, ValueError) as error:
        raise HarnessError("trusted lock substrate holder source schema differs") from error
    return sorted(holder_pids), locks_raw


def _lock_holder_pids_for_stat(metadata: os.stat_result) -> tuple[list[int], bytes]:
    return _lock_holder_pids_for_identity(metadata.st_dev, metadata.st_ino)


def cleanup_lock_substrate(
    substrate: LockSubstrate,
    run: Callable[..., subprocess.CompletedProcess[bytes]],
    *,
    runner_finished: bool,
    runner_children: list[int] | None = None,
) -> dict[str, Any]:
    """Remove only the exact substrate created by :func:`prepare_lock_substrate`."""

    owner_uid, owner_gid = _lock_owner_ids()
    children = sorted(set(runner_children or []))
    if any(type(pid) is not int or pid <= 0 for pid in children):
        raise HarnessError("trusted lock substrate runner child schema differs")
    if children:
        raise HarnessError("trusted lock substrate runner child is still alive")
    directory = substrate.directory
    lock = substrate.lock
    pre = {"directory": _substrate_dir_state(directory), "lock": None}
    try:
        lock_metadata = lock.lstat()
    except FileNotFoundError:
        lock_metadata = None
    if lock_metadata is None:
        pre["lock"] = {"path": str(lock), "present": False}
        if not runner_finished:
            raise HarnessError("trusted lock substrate lock disappeared before runner completion")
        holder_pids, locks_raw = _lock_holder_pids_for_identity(substrate.lock_identity[0], substrate.lock_identity[1])
        if holder_pids:
            raise HarnessError("trusted lock substrate removed lock still has a holder")
        lock_source = "runner_removed"
    else:
        pre["lock"] = _substrate_metadata(lock_metadata)
        if _substrate_identity(lock_metadata) != substrate.lock_identity:
            raise HarnessError("trusted lock substrate lock replacement detected during cleanup")
        _verify_substrate_lock(lock, owner_uid, owner_gid, expected_identity=substrate.lock_identity)
        holder_pids, locks_raw = _lock_holder_pids_for_stat(lock_metadata)
        descriptor = os.open(lock, os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                after_lock = lock.lstat()
            except FileNotFoundError:
                raise HarnessError("trusted lock substrate lock disappeared during cleanup")
            if _substrate_identity(after_lock) != substrate.lock_identity:
                raise HarnessError("trusted lock substrate lock replacement detected during cleanup")
            if holder_pids:
                raise HarnessError("trusted lock substrate lock holder remains")
            directory_descriptor = os.open(
                directory,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                if _substrate_identity(os.fstat(directory_descriptor), directory=True) != substrate.directory_identity:
                    raise HarnessError("trusted lock substrate directory replacement detected during cleanup")
                os.unlink(lock.name, dir_fd=directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except BlockingIOError as error:
            raise HarnessError("trusted lock substrate lock still has a holder") from error
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        lock_source = "maintenance_unlinked"
        locks_raw = b""
    if holder_pids:
        raise HarnessError("trusted lock substrate lock holder remains")
    directory_metadata = _verify_substrate_directory(directory, owner_uid, owner_gid, expected_identity=substrate.directory_identity, empty=True)
    rmdir_argv = [str(LAUNCHER.SUDO), "-n", str(RMDIR), str(directory)]
    rmdir_record = _command_with_pinned_executable(run, rmdir_argv, "lock-substrate-rmdir", RMDIR, RMDIR_SHA)
    if directory.exists() or directory.is_symlink():
        raise HarnessError("trusted lock substrate directory remained after rmdir")
    return {
        "schema_version": "ullm.aq4_p2_trusted_lock_substrate_cleanup.v1",
        "passed": True,
        "directory": str(directory),
        "lock": str(lock),
        "source": lock_source,
        "runner_finished": runner_finished,
        "runner_children": children,
        "holder_pids": holder_pids,
        "pre": pre,
        "post": {"directory": {"path": str(directory), "present": False}, "lock": {"path": str(lock), "present": False}},
        "identity": {
            "directory": {"device": directory_metadata.st_dev, "inode": directory_metadata.st_ino},
            "lock": {"device": substrate.lock_identity[0], "inode": substrate.lock_identity[1]},
        },
        "commands": [rmdir_record],
        "source_sha256": sha_bytes(locks_raw),
        "source_bytes": len(locks_raw),
        "secret_material_recorded": False,
    }


def default_owner_probe(run: Callable[..., subprocess.CompletedProcess[bytes]], worker_pid: int) -> dict[str, Any]:
    completed, record = _command(run, [str(LAUNCHER.AMD_SMI), "process", "--gpu", "2", "--general", "--json"], "gpu-owner")
    if completed.returncode != 0 or completed.stderr:
        raise HarnessError("GPU owner probe failed")
    parsed = LAUNCHER.parse_amd_process_owners(completed.stdout)
    amd_pids = parsed["owners"]
    kfd_source = LAUNCHER._kfd_owner_snapshot(allowed_owners={worker_pid})
    kfd_pids = kfd_source["owners"]
    if amd_pids != [worker_pid] or kfd_pids != [worker_pid]:
        raise HarnessError("restored worker does not uniquely own target GPU")
    return {"amd_smi": amd_pids, "amd_smi_process": parsed["diagnostic"], "amd_smi_probe": record, "kfd": kfd_pids, "kfd_source": kfd_source}


def _poll_service_value(completed: subprocess.CompletedProcess[bytes], unit: str) -> dict[str, Any]:
    try:
        values = dict(line.split("=", 1) for line in completed.stdout.decode().splitlines())
        main_pid = int(values["MainPID"])
    except (UnicodeError, ValueError, KeyError) as error:
        raise HarnessError(f"stopped poll service schema differs: {unit}") from error
    if (
        completed.returncode != 0
        or completed.stderr
        or set(values) != {"ActiveState", "SubState", "MainPID"}
        or main_pid < 0
    ):
        raise HarnessError(f"stopped poll service probe differs: {unit}")
    return {
        "unit": unit,
        "active_state": values["ActiveState"],
        "sub_state": values["SubState"],
        "main_pid": main_pid,
    }


def _poll_pid_lines(completed: subprocess.CompletedProcess[bytes], label: str) -> list[int]:
    if completed.returncode == 1 and not completed.stdout and not completed.stderr:
        return []
    try:
        values = sorted({int(item) for item in completed.stdout.decode().splitlines() if item})
    except (UnicodeError, ValueError) as error:
        raise HarnessError(f"{label} PID schema differs") from error
    if completed.returncode != 0 or completed.stderr or any(value <= 0 for value in values):
        raise HarnessError(f"{label} PID probe differs")
    return values


def _safe_proc_cmdline(pid: int) -> dict[str, Any]:
    path = Path(f"/proc/{pid}/cmdline")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            raw = os.read(descriptor, 65537)
        finally:
            os.close(descriptor)
        if len(raw) > 65536:
            raise OSError("cmdline exceeds bound")
        argv0 = raw.split(b"\0", 1)[0]
        try:
            argv0_name = Path(argv0.decode("utf-8", "strict")).name if argv0 else None
        except UnicodeError:
            argv0_name = None
        return {
            "pid": pid,
            "readable": True,
            "bytes": len(raw),
            "sha256": sha_bytes(raw),
            "argv0_basename": argv0_name,
            "matches_expected_worker": raw.startswith(str(WORKER).encode()),
            "raw_recorded": False,
        }
    except OSError as error:
        return {
            "pid": pid,
            "readable": False,
            "bytes": None,
            "sha256": None,
            "argv0_basename": None,
            "matches_expected_worker": None,
            "raw_recorded": False,
            "error_type": type(error).__name__,
        }


def _poll_lock_observation(substrate: LockSubstrate | None = None) -> dict[str, Any]:
    directory = substrate.directory if substrate is not None else _lock_substrate_directory()
    LAUNCHER.reject_symlink_components(directory, "stopped poll lock substrate directory")
    try:
        directory_metadata = directory.lstat()
        metadata = LAUNCHER.LOCK_PATH.lstat()
    except FileNotFoundError:
        if substrate is not None:
            return {
                "path": str(LAUNCHER.LOCK_PATH),
                "free": False,
                "device": -1,
                "inode": -1,
                "holder_pids": [],
                "source": "absent",
                "source_sha256": sha_bytes(b"absent"),
                "source_bytes": 6,
            }
        raise HarnessError("stopped poll device lock is absent")
    source = None
    if substrate is not None:
        if _substrate_identity(directory_metadata, directory=True) != substrate.directory_identity or _substrate_identity(metadata) != substrate.lock_identity:
            source = "replacement"
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o600:
        if substrate is not None:
            return {
                "path": str(LAUNCHER.LOCK_PATH),
                "free": False,
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "holder_pids": [],
                "source": source or "invalid",
                "source_sha256": sha_bytes(canonical(_substrate_metadata(metadata))),
                "source_bytes": len(canonical(_substrate_metadata(metadata))),
            }
        raise HarnessError("stopped poll device lock identity differs")
    if source is not None:
        # Never flock or parse holder records for an untrusted replacement.
        return {
            "path": str(LAUNCHER.LOCK_PATH),
            "free": False,
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "holder_pids": [],
            "source": source,
            "source_sha256": sha_bytes(canonical(_substrate_metadata(metadata))),
            "source_bytes": len(canonical(_substrate_metadata(metadata))),
        }
    identity = LAUNCHER.file_identity(metadata)
    descriptor = os.open(LAUNCHER.LOCK_PATH, os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    free = False
    try:
        opened = os.fstat(descriptor)
        if opened.st_dev != metadata.st_dev or opened.st_ino != metadata.st_ino:
            raise HarnessError("stopped poll device lock changed while opening")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            free = True
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except BlockingIOError:
            free = False
    finally:
        os.close(descriptor)
    if LAUNCHER.file_identity(LAUNCHER.LOCK_PATH.lstat()) != identity:
        raise HarnessError("stopped poll device lock changed")
    holder_values, locks_raw = _lock_holder_pids_for_stat(metadata)
    holder_pids = set(holder_values)
    if free and holder_pids:
        raise HarnessError("stopped poll free lock has a holder")
    result = {
        "path": str(LAUNCHER.LOCK_PATH),
        "free": free,
        "device": identity[0],
        "inode": identity[1],
        "holder_pids": holder_values,
        "source_sha256": sha_bytes(locks_raw),
        "source_bytes": len(locks_raw),
    }
    if substrate is not None:
        result["source"] = "trusted_substrate"
        result["substrate"] = {
            "directory": {"device": directory_metadata.st_dev, "inode": directory_metadata.st_ino},
            "lock": {"device": metadata.st_dev, "inode": metadata.st_ino},
        }
    return result


class StoppedPollDeadline(HarnessError):
    pass


class StoppedPollProbeTimeout(HarnessError):
    pass


class StoppedPollKeepaliveFailure(HarnessError):
    pass


class StoppedPollControl:
    def __init__(
        self,
        *,
        deadline_ns: int,
        monotonic_ns: Callable[[], int],
        keepalive: Callable[[int, float], None],
        first_keepalive_ns: int,
        lock_substrate: LockSubstrate | None = None,
    ) -> None:
        self.deadline_ns = deadline_ns
        self.monotonic_ns = monotonic_ns
        self.keepalive = keepalive
        self.next_keepalive_ns = first_keepalive_ns
        self.lock_substrate = lock_substrate
        self.attempt = 0
        self.checkpoints: list[dict[str, Any]] = []
        self.probes: list[dict[str, Any]] = []

    def begin_attempt(self, attempt: int) -> None:
        self.attempt = attempt
        self.checkpoints = []
        self.probes = []

    def checkpoint(self, label: str) -> int:
        now_ns = self.monotonic_ns()
        if now_ns >= self.deadline_ns:
            self.checkpoints.append({"label": label, "monotonic_ns": now_ns, "remaining_ns": max(0, self.deadline_ns - now_ns), "keepalive": "not_run_deadline"})
            raise StoppedPollDeadline("stopped gate absolute deadline reached")
        keepalive_state = "not_due"
        if now_ns >= self.next_keepalive_ns:
            timeout = min(STOP_POLL_PROBE_TIMEOUT_SECONDS, (self.deadline_ns - now_ns) / 1_000_000_000)
            try:
                self.keepalive(self.attempt, timeout)
            except (HarnessError, OSError, subprocess.SubprocessError) as error:
                failed_ns = self.monotonic_ns()
                self.checkpoints.append({"label": label, "monotonic_ns": failed_ns, "remaining_ns": max(0, self.deadline_ns - failed_ns), "keepalive": "failed", "timeout_seconds": timeout})
                raise StoppedPollKeepaliveFailure("stopped gate sudo keepalive failed") from error
            keepalive_state = "passed"
            now_ns = self.monotonic_ns()
            while self.next_keepalive_ns <= now_ns:
                self.next_keepalive_ns += int(STOP_POLL_SUDO_KEEPALIVE_SECONDS * 1_000_000_000)
            if now_ns >= self.deadline_ns:
                self.checkpoints.append({"label": label, "monotonic_ns": now_ns, "remaining_ns": 0, "keepalive": keepalive_state})
                raise StoppedPollDeadline("stopped gate deadline reached after sudo keepalive")
        self.checkpoints.append({"label": label, "monotonic_ns": now_ns, "remaining_ns": self.deadline_ns - now_ns, "keepalive": keepalive_state})
        return now_ns

    def remaining_timeout(self, label: str) -> float:
        now_ns = self.checkpoint(f"{label}:timeout")
        remaining_seconds = (self.deadline_ns - now_ns) / 1_000_000_000
        timeout = min(STOP_POLL_PROBE_TIMEOUT_SECONDS, remaining_seconds)
        if timeout <= 0:
            raise StoppedPollDeadline("stopped gate probe has no deadline remaining")
        return timeout

    def command(
        self,
        run: Callable[..., subprocess.CompletedProcess[bytes]],
        argv: list[str],
        label: str,
    ) -> tuple[subprocess.CompletedProcess[bytes], dict[str, Any]]:
        self.checkpoint(f"{label}:before")
        timeout = self.remaining_timeout(label)
        try:
            completed, record = _command(run, argv, label, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout if isinstance(error.stdout, bytes) else b""
            stderr = error.stderr if isinstance(error.stderr, bytes) else b""
            self.probes.append({
                "label": label,
                "argv": argv,
                "exit_code": "timeout",
                "stdout_sha256": sha_bytes(stdout),
                "stderr_sha256": sha_bytes(stderr),
                "captured_unix_ns": time.time_ns(),
                "timeout_seconds": timeout,
            })
            try:
                self.checkpoint(f"{label}:after-timeout")
            except StoppedPollDeadline:
                raise StoppedPollDeadline("stopped gate probe crossed absolute deadline") from error
            raise StoppedPollProbeTimeout("stopped gate probe exceeded bounded timeout") from error
        record["timeout_seconds"] = timeout
        self.probes.append(record)
        self.checkpoint(f"{label}:after")
        return completed, record

    def bounded_call(self, label: str, function: Callable[[], Any]) -> Any:
        self.checkpoint(f"{label}:before")
        try:
            value = function()
        except Exception:
            self.checkpoint(f"{label}:after-error")
            raise
        self.checkpoint(f"{label}:after")
        return value


class StoppedGateObserver:
    def __call__(self, old_worker_pid: int, old_service_pid: int, run: Callable[..., subprocess.CompletedProcess[bytes]], control: StoppedPollControl) -> dict[str, Any]:
        control.checkpoint("observer:validation-before")
        LAUNCHER.validate_amd_smi_tool()
        for path, digest, label in (
            (LAUNCHER.SYSTEMCTL, LAUNCHER.SYSTEMCTL_SHA, "systemctl"),
            (LAUNCHER.PGREP, LAUNCHER.PGREP_SHA, "pgrep"),
        ):
            if LAUNCHER.sha_file(path, f"stopped poll {label}")[0] != digest:
                raise HarnessError(f"stopped poll {label} SHA differs")
        control.checkpoint("observer:validation-after")
        probes: list[dict[str, Any]] = []
        services: list[dict[str, Any]] = []
        for unit in LAUNCHER.SERVICE_UNITS:
            command = [str(LAUNCHER.SYSTEMCTL), "show", unit, "--property=ActiveState", "--property=SubState", "--property=MainPID", "--no-pager"]
            completed, record = control.command(run, command, f"stopped-poll-service-{unit}")
            probes.append(record)
            services.append(_poll_service_value(completed, unit))
        worker_command = [str(LAUNCHER.PGREP), "-f", "-x", f"{WORKER}.*"]
        worker, record = control.command(run, worker_command, "stopped-poll-worker")
        probes.append(record)
        worker_pids = _poll_pid_lines(worker, "stopped poll worker")
        process_command = [str(LAUNCHER.AMD_SMI), "process", "--gpu", str(LAUNCHER.AMD_SMI_INDEX), "--general", "--json"]
        processes, record = control.command(run, process_command, "stopped-poll-amd-process")
        probes.append(record)
        if processes.returncode != 0 or processes.stderr:
            raise HarnessError("stopped poll AMD process probe failed")
        parsed_processes = LAUNCHER.parse_amd_process_owners(processes.stdout)
        amd_smi_owners = parsed_processes["owners"]
        static_command = [str(LAUNCHER.AMD_SMI), "static", "--gpu", str(LAUNCHER.AMD_SMI_INDEX), "--vram", "--json"]
        static, record = control.command(run, static_command, "stopped-poll-amd-vram")
        probes.append(record)
        try:
            static_value = LAUNCHER.parse_json(static.stdout, "stopped poll AMD VRAM")
            gpu_data = static_value["gpu_data"]
            vram_item = gpu_data[0]
            size = vram_item["vram"]["size"]
            total_bytes = int(size["value"]) * 1_000_000
        except (LAUNCHER.LauncherError, KeyError, IndexError, TypeError, ValueError) as error:
            raise HarnessError("stopped poll AMD VRAM schema differs") from error
        if static.returncode != 0 or static.stderr or len(gpu_data) != 1 or vram_item.get("gpu") != LAUNCHER.AMD_SMI_INDEX or size.get("unit") != "MB" or total_bytes <= 0:
            raise HarnessError("stopped poll AMD VRAM probe differs")
        kfd_source = control.bounded_call(
            "stopped-poll-kfd",
            lambda: LAUNCHER._kfd_owner_snapshot(allowed_owners={old_worker_pid}),
        )
        kfd_owners = kfd_source["owners"]
        lock = control.bounded_call(
            "stopped-poll-lock",
            (lambda: _poll_lock_observation(control.lock_substrate)) if control.lock_substrate is not None else _poll_lock_observation,
        )
        observed_pids = sorted(set(worker_pids) | set(amd_smi_owners) | set(kfd_owners) | set(lock["holder_pids"]) | {old_worker_pid, old_service_pid})
        proc_cmdlines = [control.bounded_call(f"stopped-poll-proc-{pid}", lambda pid=pid: _safe_proc_cmdline(pid)) for pid in observed_pids]
        return {
            "captured_unix_ns": time.time_ns(),
            "services": services,
            "worker_pids": worker_pids,
            "amd_smi_owners": amd_smi_owners,
            "kfd_owners": kfd_owners,
            "lock": lock,
            "vram": {
                "total_bytes": total_bytes,
                "used_bytes": 0 if not amd_smi_owners and not kfd_owners else None,
                "free_bytes": total_bytes if not amd_smi_owners and not kfd_owners else None,
                "headroom_bytes": total_bytes if not amd_smi_owners and not kfd_owners else None,
            },
            "proc_cmdlines": proc_cmdlines,
            "probes": probes,
            "virtual_sources": {
                "amd_smi_owners": parsed_processes["diagnostic"],
                "kfd_owners": kfd_source,
                "lock_holders": {"raw_sha256": lock["source_sha256"], "raw_bytes": lock["source_bytes"], "parsed_pids": lock["holder_pids"]},
            },
            "secret_material_recorded": False,
        }


def _stopped_observation_decision(
    observation: dict[str, Any],
    old_worker_pid: int,
    old_service_pid: int,
    seen_zero: dict[str, bool],
    substrate: LockSubstrate | None = None,
) -> tuple[str, str | None, dict[str, str]]:
    expected_services = [
        {"unit": unit, "active_state": "inactive", "sub_state": "dead", "main_pid": 0}
        for unit in LAUNCHER.SERVICE_UNITS
    ]
    required = {"captured_unix_ns", "services", "worker_pids", "amd_smi_owners", "kfd_owners", "lock", "vram", "proc_cmdlines", "probes", "virtual_sources", "secret_material_recorded"}
    classifications: dict[str, str] = {}
    if not isinstance(observation, dict) or set(observation) != required or observation.get("secret_material_recorded") is not False:
        return "terminal_failure", "stopped observation schema differs", classifications
    if any(not isinstance(observation.get(name), list) or any(type(pid) is not int or pid <= 0 for pid in observation[name]) for name in ("worker_pids", "amd_smi_owners", "kfd_owners")):
        return "terminal_failure", "stopped observation PID schema differs", classifications
    services_stable = observation.get("services") == expected_services
    classifications["services"] = "stable" if services_stable else "pending"
    for name in ("worker_pids", "amd_smi_owners", "kfd_owners"):
        values = sorted(set(observation[name]))
        foreign = [pid for pid in values if pid != old_worker_pid]
        if foreign:
            classifications[name] = "foreign_or_new"
            return "terminal_failure", f"foreign or new owner observed in {name}", classifications
        if values and seen_zero[name]:
            classifications[name] = "reappeared"
            return "terminal_failure", f"old owner reappeared in {name}", classifications
        if not values:
            seen_zero[name] = True
            classifications[name] = "stable"
        else:
            classifications[name] = "draining_pre_stop_worker"
    lock = observation.get("lock")
    lock_keys = {"path", "free", "device", "inode", "holder_pids", "source_sha256", "source_bytes"}
    if not isinstance(lock, dict) or not lock_keys.issubset(lock) or lock.get("path") != str(LAUNCHER.LOCK_PATH) or type(lock.get("free")) is not bool or not isinstance(lock.get("holder_pids"), list):
        return "terminal_failure", "stopped observation lock schema differs", classifications
    if lock.get("source") in {"replacement", "absent", "invalid"}:
        classifications["lock"] = str(lock["source"])
        return "terminal_failure", "trusted lock substrate replacement or disappearance observed", classifications
    if "source" in lock and lock.get("source") != "trusted_substrate":
        classifications["lock"] = "untrusted_source"
        return "terminal_failure", "stopped observation lock source is untrusted", classifications
    if substrate is not None:
        observed_substrate = lock.get("substrate")
        expected_substrate = {
            "directory": {"device": substrate.directory_identity[0], "inode": substrate.directory_identity[1]},
            "lock": {"device": substrate.lock_identity[0], "inode": substrate.lock_identity[1]},
        }
        if lock.get("source") != "trusted_substrate" or observed_substrate != expected_substrate:
            classifications["lock"] = "identity_mismatch"
            return "terminal_failure", "stopped observation lock substrate identity differs", classifications
    foreign_lock = [pid for pid in lock["holder_pids"] if pid != old_service_pid]
    if foreign_lock or (lock["free"] is False and not lock["holder_pids"]):
        classifications["lock"] = "foreign_or_unknown_holder"
        return "terminal_failure", "foreign or unknown lock holder observed", classifications
    if not lock["free"] and seen_zero["lock"]:
        classifications["lock"] = "reappeared"
        return "terminal_failure", "pre-stop lock holder reappeared", classifications
    if lock["free"]:
        seen_zero["lock"] = True
        classifications["lock"] = "stable"
    else:
        classifications["lock"] = "draining_pre_stop_service"
    vram = observation.get("vram")
    if not isinstance(vram, dict) or set(vram) != {"total_bytes", "used_bytes", "free_bytes", "headroom_bytes"} or type(vram.get("total_bytes")) is not int or vram["total_bytes"] <= 0:
        return "terminal_failure", "stopped observation VRAM schema differs", classifications
    stable = (
        services_stable
        and not observation["worker_pids"]
        and not observation["amd_smi_owners"]
        and not observation["kfd_owners"]
        and lock["free"] is True
        and vram["used_bytes"] == 0
        and vram["free_bytes"] == vram["total_bytes"]
        and vram["headroom_bytes"] == vram["total_bytes"]
    )
    classifications["vram"] = "stable" if vram["used_bytes"] == 0 and vram["free_bytes"] == vram["total_bytes"] and vram["headroom_bytes"] == vram["total_bytes"] else "pending"
    return ("stable", None, classifications) if stable else ("pending", None, classifications)


def poll_stopped_gates(
    output: Path,
    old_worker_pid: int,
    old_service_pid: int,
    dependencies: "Dependencies",
    keepalive: Callable[[int, float], None],
    lock_substrate: LockSubstrate | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    started_ns = dependencies.monotonic_ns()
    deadline_ns = started_ns + int(STOP_POLL_TIMEOUT_SECONDS * 1_000_000_000)
    control = StoppedPollControl(
        deadline_ns=deadline_ns,
        monotonic_ns=dependencies.monotonic_ns,
        keepalive=keepalive,
        first_keepalive_ns=started_ns + int(STOP_POLL_SUDO_KEEPALIVE_SECONDS * 1_000_000_000),
        lock_substrate=lock_substrate,
    )
    stable_count = 0
    seen_zero = {"worker_pids": False, "amd_smi_owners": False, "kfd_owners": False, "lock": False}
    files: list[dict[str, Any]] = []
    attempt = 0
    final_observation: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    while True:
        control.begin_attempt(attempt)
        observation_started_ns = dependencies.monotonic_ns()
        elapsed = max(0.0, (observation_started_ns - started_ns) / 1_000_000_000)
        try:
            control.checkpoint("observation:start")
            observation = dependencies.stopped_observation(old_worker_pid, old_service_pid, dependencies.run, control)
            observation_completed_ns = control.checkpoint("observation:complete")
            decision, reason, classifications = _stopped_observation_decision(observation, old_worker_pid, old_service_pid, seen_zero, lock_substrate)
        except StoppedPollDeadline as error:
            observation = {
                "captured_unix_ns": time.time_ns(),
                "error_type": type(error).__name__,
                "partial_probes": list(control.probes),
                "secret_material_recorded": False,
            }
            observation_completed_ns = dependencies.monotonic_ns()
            decision, reason, classifications = "deadline_timeout", "stopped gate absolute deadline reached", {"observer": "deadline_timeout"}
        except StoppedPollProbeTimeout as error:
            observation = {
                "captured_unix_ns": time.time_ns(),
                "error_type": type(error).__name__,
                "partial_probes": list(control.probes),
                "secret_material_recorded": False,
            }
            observation_completed_ns = dependencies.monotonic_ns()
            decision, reason, classifications = "probe_timeout", "stopped gate bounded probe timed out", {"observer": "probe_timeout"}
        except StoppedPollKeepaliveFailure as error:
            observation = {
                "captured_unix_ns": time.time_ns(),
                "error_type": type(error).__name__,
                "partial_probes": list(control.probes),
                "secret_material_recorded": False,
            }
            observation_completed_ns = dependencies.monotonic_ns()
            decision, reason, classifications = "keepalive_failure", "stopped gate sudo keepalive failed", {"observer": "sudo_keepalive"}
        except LAUNCHER.AmdProcessSchemaError as error:
            observation = {
                "captured_unix_ns": time.time_ns(),
                "error_type": type(error).__name__,
                "parse_diagnostic": error.diagnostic,
                "partial_probes": list(control.probes),
                "secret_material_recorded": False,
            }
            observation_completed_ns = dependencies.monotonic_ns()
            decision, reason, classifications = "terminal_failure", f"stopped AMD process schema rejected: {error.diagnostic['reason_code']}", {"observer": "amd_process_schema"}
        except LAUNCHER.KfdOwnerScanError as error:
            observation = {
                "captured_unix_ns": time.time_ns(),
                "error_type": type(error).__name__,
                "parse_diagnostic": error.diagnostic,
                "partial_probes": list(control.probes),
                "secret_material_recorded": False,
            }
            observation_completed_ns = dependencies.monotonic_ns()
            decision, reason, classifications = "terminal_failure", f"stopped KFD owner scan rejected: {error.diagnostic['reason_code']}", {"observer": "kfd_owner_scan"}
        except (HarnessError, LAUNCHER.LauncherError, OSError, ValueError, subprocess.SubprocessError) as error:
            observation = {
                "captured_unix_ns": time.time_ns(),
                "error_type": type(error).__name__,
                "partial_probes": list(control.probes),
                "secret_material_recorded": False,
            }
            observation_completed_ns = dependencies.monotonic_ns()
            decision, reason, classifications = "terminal_failure", "stopped observation failed", {"observer": "error"}
        stable_count = stable_count + 1 if decision == "stable" else 0
        next_interval = min(STOP_POLL_INITIAL_INTERVAL_SECONDS * (2 ** attempt), STOP_POLL_MAX_INTERVAL_SECONDS)
        poll_document = {
            "schema_version": "ullm.aq4_p2_stopped_gate_poll_observation.v1",
            "attempt": attempt,
            "captured_unix_ns": observation["captured_unix_ns"],
            "elapsed_seconds": elapsed,
            "observation_started_monotonic_ns": observation_started_ns,
            "observation_completed_monotonic_ns": observation_completed_ns,
            "absolute_deadline_monotonic_ns": deadline_ns,
            "decision": decision,
            "reason": reason,
            "source_classification": classifications,
            "consecutive_stable": stable_count,
            "required_consecutive_stable": STOP_POLL_STABLE_OBSERVATIONS,
            "next_interval_seconds": None if decision in {"terminal_failure", "deadline_timeout", "probe_timeout", "keepalive_failure"} or stable_count >= STOP_POLL_STABLE_OBSERVATIONS else next_interval,
            "deadline_checkpoints": list(control.checkpoints),
            "observation": observation,
            "secret_material_recorded": False,
        }
        raw = pretty(poll_document)
        name = f"stopped-gate-poll-{attempt:04d}.json"
        LAUNCHER.atomic_write(output, name, raw)
        files.append({"name": name, "sha256": sha_bytes(raw), "decision": decision, "consecutive_stable": stable_count})
        final_observation = observation if set(observation) == {"captured_unix_ns", "services", "worker_pids", "amd_smi_owners", "kfd_owners", "lock", "vram", "proc_cmdlines", "probes", "virtual_sources", "secret_material_recorded"} else None
        if decision == "deadline_timeout":
            failure = {"kind": "timeout", "reason": reason, "attempt": attempt}
            break
        if decision == "probe_timeout":
            failure = {"kind": "probe_timeout", "reason": reason, "attempt": attempt}
            break
        if decision == "keepalive_failure":
            failure = {"kind": "sudo_keepalive", "reason": reason, "attempt": attempt}
            break
        if decision == "terminal_failure":
            failure = {"kind": "terminal", "reason": reason, "attempt": attempt}
            break
        if stable_count >= STOP_POLL_STABLE_OBSERVATIONS:
            break
        now_ns = dependencies.monotonic_ns()
        remaining_seconds = max(0.0, (deadline_ns - now_ns) / 1_000_000_000)
        if remaining_seconds <= 0:
            attempt += 1
            continue
        dependencies.sleep(min(next_interval, remaining_seconds))
        attempt += 1
    poll_evidence = {
        "schema_version": "ullm.aq4_p2_stopped_gate_poll.v1",
        "passed": failure is None and stable_count >= STOP_POLL_STABLE_OBSERVATIONS,
        "policy": {
            "timeout_seconds": STOP_POLL_TIMEOUT_SECONDS,
            "initial_interval_seconds": STOP_POLL_INITIAL_INTERVAL_SECONDS,
            "maximum_interval_seconds": STOP_POLL_MAX_INTERVAL_SECONDS,
            "required_consecutive_stable": STOP_POLL_STABLE_OBSERVATIONS,
            "sudo_keepalive_seconds": STOP_POLL_SUDO_KEEPALIVE_SECONDS,
            "maximum_probe_timeout_seconds": STOP_POLL_PROBE_TIMEOUT_SECONDS,
            "absolute_deadline_monotonic_ns": deadline_ns,
            "only_pre_stop_worker_pid_may_drain": old_worker_pid,
            "only_pre_stop_service_pid_may_hold_lock": old_service_pid,
        },
        "poll_files": files,
        "poll_count": len(files),
        "probe_command_count": 0,
        "failure": failure,
        "secret_material_recorded": False,
    }
    # Count from the immutable poll files' source observations without retaining raw output.
    for item in files:
        document = LAUNCHER.parse_json((output / item["name"]).read_bytes(), "stopped poll evidence")
        source = document["observation"]
        poll_evidence["probe_command_count"] += len(source.get("probes", source.get("partial_probes", [])))
    if not poll_evidence["passed"] or final_observation is None:
        return None, poll_evidence
    gates = {
        "passed": True,
        "environment": dict(LAUNCHER.EXECUTE_ENV),
        "services": final_observation["services"],
        "old_worker_pids": final_observation["worker_pids"],
        "runtime_mapping": {"runtime_device_index": 1, "visible_token": "1", "amd_smi_index": 2, "bdf": LAUNCHER.GPU_BDF, "uuid": LAUNCHER.GPU_UUID, "kfd_id": LAUNCHER.KFD_ID, "node_id": 2},
        "amd_smi_owners": final_observation["amd_smi_owners"],
        "kfd_owners": final_observation["kfd_owners"],
        "lock": {
            **{key: final_observation["lock"][key] for key in ("path", "free", "device", "inode")},
            **({key: final_observation["lock"][key] for key in ("source", "substrate") if key in final_observation["lock"]}),
        },
        "vram": final_observation["vram"],
        "probes": final_observation["probes"],
    }
    return gates, poll_evidence


@dataclass(frozen=True)
class Dependencies:
    run: Callable[..., subprocess.CompletedProcess[bytes]]
    http_probe: Callable[[str], dict[str, Any]]
    container_health: Callable[[Callable[..., subprocess.CompletedProcess[bytes]]], dict[str, Any]]
    stopped_observation: Callable[[int, int, Callable[..., subprocess.CompletedProcess[bytes]], StoppedPollControl], dict[str, Any]]
    lock_busy: Callable[[], bool]
    owner_probe: Callable[[Callable[..., subprocess.CompletedProcess[bytes]], int], dict[str, Any]]
    package_hash: Callable[[Path], str]
    launcher_execute: Callable[[dict[str, Any]], tuple[int, dict[str, Any]]]
    profile_capture: Callable[[dict[str, Any]], dict[str, Any]]
    profile_trust: Callable[[dict[str, Any], str], dict[str, Any]]
    sleep: Callable[[float], None]
    monotonic_ns: Callable[[], int]
    lock_substrate_prepare: Callable[[Callable[..., subprocess.CompletedProcess[bytes]]], LockSubstrate] | None = None
    lock_substrate_cleanup: Callable[..., dict[str, Any]] | None = None


def _host_route_diagnostics(dependencies: Dependencies) -> dict[str, Any]:
    result: dict[str, Any] = {"formal_gate": False, "probes": {}}
    for name, url, expected in (
        ("gateway_readyz", GATEWAY_READY_URL, GATEWAY_READY_BODY),
        ("openwebui_health", OPENWEBUI_HEALTH_URL, OPENWEBUI_HEALTH_BODY),
    ):
        try:
            value = dependencies.http_probe(url)
            valid_shape = isinstance(value, dict) and set(value) == {"url", "status", "body"} and value.get("url") == url and isinstance(value.get("status"), int) and isinstance(value.get("body"), bytes)
            body = value.get("body", b"") if valid_shape else b""
            result["probes"][name] = {
                "url": url,
                "reachable": valid_shape,
                "status": value.get("status") if valid_shape else None,
                "body_sha256": sha_bytes(body) if valid_shape and len(body) <= 65536 else None,
                "body_bytes": len(body) if valid_shape and len(body) <= 65536 else None,
                "matches_formal_response": valid_shape and value.get("status") == 200 and body == expected,
                "error_type": None,
            }
        except Exception as error:
            result["probes"][name] = {
                "url": url,
                "reachable": False,
                "status": None,
                "body_sha256": None,
                "body_bytes": None,
                "matches_formal_response": False,
                "error_type": type(error).__name__,
            }
    return result


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
    container_health = dependencies.container_health(dependencies.run)
    if (
        not isinstance(container_health, dict)
        or container_health.get("secret_material_recorded") is not False
        or container_health.get("process_counts")
        != {
            "docker": 9,
            "docker_exec": 6,
            "container_curl": 6,
            "container_curl_total": 6,
            "container_curl_version": 1,
            "container_curl_endpoint": 5,
        }
    ):
        raise HarnessError("container namespace health contract differs")
    host_diagnostics = _host_route_diagnostics(dependencies)
    if previous is not None and (service["main_pid"] == previous["service"]["main_pid"] or worker_pid == previous["worker"]["pid"] or service["nrestarts"] != previous["service"]["nrestarts"] or service["control_group"] != previous["service"]["control_group"]):
        raise HarnessError("restored service epoch/NRestarts differs")
    return {
        "service": service, "worker": {"path": str(WORKER), "pid": worker_pid, "sha256": worker_sha}, "gpu": gpu,
        "owners": owners, "lock": {"path": str(LAUNCHER.LOCK_PATH), "busy": True},
        "hashes": {"served_manifest_sha256": manifest_sha, "worker_sha256": worker_sha, "package_manifest_sha256": package_manifest_sha, "package_content_sha256": package_content_sha},
        "health": {"formal": container_health, "host_route_diagnostics": host_diagnostics}, "commands": [service_record, worker_record, gpu_record],
    }


def _default_launcher_execute(binding: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    gate_provider = lambda: LAUNCHER.collect_execute_gates(environment=dict(LAUNCHER.EXECUTE_ENV))
    return LAUNCHER.execute_bound(binding, Path(binding["evidence_output"]), Path(binding["runner_output"]), binding["run_id"], trusted_launcher_sha=LAUNCHER_SHA, gate_provider=gate_provider)


class ProfileTrustGuard:
    def __init__(self) -> None:
        self.snapshot = LAUNCHER.Snapshot()
        self.initialized = False

    def __call__(self, contract: dict[str, Any], stage: str) -> dict[str, Any]:
        allowed = {"before-start", "capture-before", "capture-after", "finalize-before"}
        if stage not in allowed:
            raise HarnessError("profile trust stage differs")
        manifest = profile_target_command_manifest()
        manifest_ref = {
            "path": str(PROFILE_TARGET_COMMAND_MANIFEST),
            "sha256": sha_bytes(pretty(manifest)),
            "manifest_sha256": manifest["manifest_sha256"],
        }
        if contract.get("command") != profile_capture_command() or contract.get("target_launcher", {}).get("command") != profile_launcher_command():
            raise HarnessError("profile capture/launcher command manifest differs")
        if contract.get("target_launcher", {}).get("manifest") != manifest_ref:
            raise HarnessError("profile target command manifest binding differs")
        if not self.initialized:
            if stage != "before-start":
                raise HarnessError("profile trust was not initialized before capture")
            self.snapshot.file(PROFILE_CAPTURE_TOOL, PROFILE_CAPTURE_SHA, "profile capture tool")
            self.snapshot.file(PROFILE_PROFILER, PROFILE_PROFILER_SHA, "profile profiler")
            self.snapshot.file(LAUNCHER.PYTHON, LAUNCHER.PYTHON_SHA, "profile target Python")
            self.snapshot.file(LAUNCHER_PATH, LAUNCHER_SHA, "profile target launcher")
            self.snapshot.file(
                PROFILE_TARGET_COMMAND_MANIFEST,
                manifest_ref["sha256"],
                "profile target command manifest",
            )
            self.initialized = True
        self.snapshot.verify()
        return {
            "stage": stage,
            "passed": True,
            "capture_tool_sha256": PROFILE_CAPTURE_SHA,
            "profiler_sha256": PROFILE_PROFILER_SHA,
            "python_sha256": LAUNCHER.PYTHON_SHA,
            "launcher_sha256": LAUNCHER_SHA,
            "target_manifest_sha256": contract["target_launcher"]["manifest"]["sha256"],
            "capture_command_sha256": contract["command_sha256"],
            "launcher_command_sha256": contract["target_launcher"]["command_sha256"],
        }


def _descendant_pids(root_pid: int) -> list[int]:
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            parents[int(entry.name)] = int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
    descendants: list[int] = []
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        children = [pid for pid, ppid in parents.items() if ppid == parent]
        descendants.extend(children); pending.extend(children)
    return sorted(set(descendants))


def _signal_profile_tree(root_pid: int, descendants: list[int], value: signal.Signals) -> None:
    groups: set[int] = set()
    for pid in [*descendants, root_pid]:
        try:
            groups.add(os.getpgid(pid))
        except ProcessLookupError:
            continue
    own_group = os.getpgrp()
    for group in groups:
        if group == own_group:
            continue
        try:
            os.killpg(group, value)
        except ProcessLookupError:
            pass


def run_profile_capture(contract: dict[str, Any]) -> dict[str, Any]:
    command = contract.get("command")
    if command != profile_capture_command():
        raise HarnessError("profile capture command differs")
    timed_out = False; descendants: list[int] = []
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout_file, stderr=stderr_file, shell=False, start_new_session=True)
        try:
            return_code = process.wait(timeout=PROFILE_TIMEOUT_SECONDS + 30)
        except subprocess.TimeoutExpired:
            timed_out = True; descendants = _descendant_pids(process.pid)
            for sig, seconds in ((signal.SIGINT, 1.0), (signal.SIGTERM, 1.0), (signal.SIGKILL, 5.0)):
                _signal_profile_tree(process.pid, descendants, sig)
                try:
                    return_code = process.wait(timeout=seconds)
                    break
                except subprocess.TimeoutExpired:
                    descendants = sorted(set(descendants) | set(_descendant_pids(process.pid)))
            else:
                process.kill(); return_code = process.wait()
        stdout_file.seek(0); stdout = stdout_file.read(LAUNCHER.MAX_BYTES + 1)
        stderr_file.seek(0); stderr = stderr_file.read(LAUNCHER.MAX_BYTES + 1)
    if len(stdout) > LAUNCHER.MAX_BYTES or len(stderr) > LAUNCHER.MAX_BYTES:
        raise HarnessError("profile capture output exceeds evidence bound")
    remaining = [pid for pid in descendants if Path(f"/proc/{pid}").exists()]
    completed = subprocess.CompletedProcess(command, return_code, stdout, stderr)
    launcher_evidence = None
    launcher_path = LAUNCHER.PROFILE_EVIDENCE_OUTPUT / "launcher-evidence.json"
    if launcher_path.is_file() and not launcher_path.is_symlink():
        launcher_evidence = LAUNCHER.parse_json(LAUNCHER.read_regular(launcher_path, "profile launcher evidence")[0], "profile launcher evidence")
    safety = launcher_evidence.get("safety", {}) if isinstance(launcher_evidence, dict) else {}
    return {
        "completed": completed,
        "started": True,
        "timed_out": timed_out,
        "cleanup_passed": not remaining,
        "children_remaining": remaining,
        "rocprof_started": PROFILE_OUTPUT_DIRECTORY.exists(),
        "launcher_started": isinstance(launcher_evidence, dict),
        "launcher_status": launcher_evidence.get("status") if isinstance(launcher_evidence, dict) else None,
        "gpu_command_executed": safety.get("gpu_command_executed", "unknown") if timed_out or return_code != 0 else True,
        "model_load_executed": safety.get("model_load_executed", "unknown") if timed_out or return_code != 0 else True,
    }


def default_dependencies() -> Dependencies:
    trust = ProfileTrustGuard()
    container_health = ContainerHealthGuard()
    stopped_observation = StoppedGateObserver()
    return Dependencies(
        subprocess.run,
        default_http_probe,
        container_health,
        stopped_observation,
        default_lock_busy,
        default_owner_probe,
        tree_hash,
        _default_launcher_execute,
        run_profile_capture,
        trust,
        time.sleep,
        time.monotonic_ns,
        prepare_lock_substrate,
        cleanup_lock_substrate,
    )


def profile_launcher_command() -> list[str]:
    return [
        str(LAUNCHER.PYTHON),
        str(LAUNCHER_PATH),
        "--mode",
        "profile-execute",
        "--evidence-output",
        str(LAUNCHER.PROFILE_EVIDENCE_OUTPUT),
        "--runner-output",
        str(LAUNCHER.PROFILE_RUN_OUTPUT),
        "--run-id",
        LAUNCHER.PROFILE_RUN_ID,
        "--trusted-launcher-sha",
        LAUNCHER_SHA,
    ]


def profile_target_command_manifest() -> dict[str, Any]:
    command = profile_launcher_command()
    value: dict[str, Any] = {
        "schema_version": "ullm.aq4_p3_profile_target_command.v1",
        "status": "bound",
        "manifest_sha256": None,
        "argv": command,
        "input_files": [
            {
                "argument_index": 0,
                "path": command[0],
                "sha256": LAUNCHER.PYTHON_SHA,
                "executable": True,
            },
            {
                "argument_index": 1,
                "path": command[1],
                "sha256": LAUNCHER_SHA,
                "executable": False,
            },
        ],
        "output_paths": [
            {"argument_index": 5, "path": command[5]},
            {"argument_index": 7, "path": command[7]},
        ],
    }
    value["manifest_sha256"] = sha_bytes(canonical(value))
    return value


def profile_capture_command() -> list[str]:
    return [
        str(LAUNCHER.PYTHON),
        str(PROFILE_CAPTURE_TOOL),
        "capture",
        "--profiler-path",
        str(PROFILE_PROFILER),
        "--profiler-sha256",
        PROFILE_PROFILER_SHA,
        "--target-command-manifest",
        str(PROFILE_TARGET_COMMAND_MANIFEST),
        "--target-command-manifest-sha256",
        sha_bytes(pretty(profile_target_command_manifest())),
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
    "schema_version": "ullm.aq4_p2_resident_execute_qa_attestation.v2", "status": "passed", "actual_executed": False,
    "automated_tests": {
        "schema_version": "ullm.aq4_p2_exact_test_file_manifest.v1",
        "aggregate": {"distinct_test_file_count": 11, "collected": 342, "passed": 342, "failed": 0, "deselected": 0},
        "suites": [
            {
                "name": "resident_trust_chain",
                "command": ["python3", "-m", "pytest", "-q", "tests/test_prepare_aq4_p2_resident_smoke_bundle.py", "tests/test_run_aq4_p2_resident_batch.py", "tests/test_run_aq4_p2_resident_live_preflight.py", "tests/test_launch_aq4_p2_resident_smoke.py", "tests/test_launch_aq4_p2_resident_smoke_execute.py", "tests/test_aq4_p2_resident_smoke_maintenance.py"],
                "collected": 252, "passed": 252, "failed": 0, "deselected": 0,
                "files": [
                    {"path": "tests/test_prepare_aq4_p2_resident_smoke_bundle.py", "source_commit": "2c24b7670b52610f6b1db33633139023778b18e9", "git_blob": "34c25be2a019d52564e4eb2449cb68622b2336f3", "collected": 50, "passed": 50},
                    {"path": "tests/test_run_aq4_p2_resident_batch.py", "source_commit": "4005c80e542d7e37cf79e62d2fd053f99c0353c1", "git_blob": "61778f9935e2d8bcd469cdf1a8a74495543c1aee", "collected": 37, "passed": 37},
                    {"path": "tests/test_run_aq4_p2_resident_live_preflight.py", "source_commit": "774f6ddc10791db8795b37f41c5245d0edfebe42", "git_blob": "9ca9a777e96c34d308a3ce64354acf62e05f7d2d", "collected": 24, "passed": 24},
                    {"path": "tests/test_launch_aq4_p2_resident_smoke.py", "source_commit": "2ff2e7c4172a2edee49dfce67b07009364a2f958", "git_blob": "6229512f6ee12d21fd9aa42ea85f01380a379546", "collected": 7, "passed": 7},
                    {"path": "tests/test_launch_aq4_p2_resident_smoke_execute.py", "source_commit": "3642999bd89bac462330a15da1e114f92204f8b5", "git_blob": "00ebb0291738b373c1bdc3e848da82cc54100291", "collected": 58, "passed": 58},
                    {"path": "tests/test_aq4_p2_resident_smoke_maintenance.py", "source_commit": "9a3de26914fee595466644bc9f47f276ae7337c0", "git_blob": "c6f55e522435c037ee607d3bc661a2954371bd61", "collected": 76, "passed": 76},
                ],
            },
            {
                "name": "resident_roctx_ranges",
                "command": ["python3", "-m", "pytest", "-q", "tests/test_aq4_p2_resident_roctx_ranges.py"],
                "collected": 5, "passed": 5, "failed": 0, "deselected": 0,
                "files": [{"path": "tests/test_aq4_p2_resident_roctx_ranges.py", "source_commit": "62eadada3082b0c72eb1b467177ffe0c9445f26d", "git_blob": "a6ee8886ffdf58ea668b1b4c49452fa47637f7d9", "collected": 5, "passed": 5}],
            },
            {
                "name": "diagnostic_capture",
                "command": ["python3", "-m", "pytest", "-q", "tests/test_capture_aq4_p3_diagnostic_profile.py"],
                "collected": 11, "passed": 11, "failed": 0, "deselected": 0,
                "files": [{"path": "tests/test_capture_aq4_p3_diagnostic_profile.py", "source_commit": "b4d515f9908136fa773f957775beab79edc3065d", "git_blob": "86b95305159a831151c12472e3366a937e38e0fd", "collected": 11, "passed": 11}],
            },
            {
                "name": "selection_raw_producer",
                "command": ["python3", "-m", "pytest", "-q", "tests/test_build_aq4_p3_selection_raw.py"],
                "collected": 21, "passed": 21, "failed": 0, "deselected": 0,
                "files": [{"path": "tests/test_build_aq4_p3_selection_raw.py", "source_commit": "78ba33c982c994df47c8ff4541df85b8d7da4a63", "git_blob": "1227c0a4fd60b9730e2e9e8b9f663fbc5867914f", "collected": 21, "passed": 21}],
            },
            {
                "name": "profile_family_exclusion",
                "command": ["python3", "-m", "pytest", "-q", "tests/test_profile_aq4_p2_family_exclusive.py"],
                "collected": 27, "passed": 27, "failed": 0, "deselected": 0,
                "files": [{"path": "tests/test_profile_aq4_p2_family_exclusive.py", "source_commit": "0a630c705b7594a016edb42e32a29e4647da9d10", "git_blob": "30827a375a2dacd329e3023fd02b0631d8526b67", "collected": 27, "passed": 27}],
            },
            {
                "name": "candidate_selector",
                "command": ["python3", "-m", "pytest", "-q", "tests/test_select_aq4_p3_candidate.py"],
                "collected": 26, "passed": 26, "failed": 0, "deselected": 0,
                "files": [{"path": "tests/test_select_aq4_p3_candidate.py", "source_commit": "01055a1960e55ab98990a6ee109e4a778f5bfd67", "git_blob": "9665da8fd81cd875b522c80f42bea6777642caf7", "collected": 26, "passed": 26}],
            },
        ],
    },
    "manual_checks": {"boundary_count": 15, "status": "passed"},
    "strict_negative_contract_count": 38,
    "coverage": ["safety-success-start-failure-partial", "validator-runner-finalize-toctou", "identity-and-hash-bindings", "worker-exact-two-hardlink-set-pre-open-post-and-rehash", "bounded-driver-stdout-and-streamed-stderr-failure-evidence", "driver-process-group-descendant-cleanup-and-secret-redaction", "strict-amd-process-active-owner-and-zero-sentinel-schema", "secret-free-amd-process-rejection-shape-and-raw-sha", "bounded-kfd-enoent-rescan-and-fatal-source-diagnostics", "trusted-runtime-lock-substrate-lifecycle-and-same-inode-runner-binding", "absolute-deadline-stable2-stopped-gate-poll-and-foreign-owner-rejection", "remaining-capped-probe-timeouts-and-between-probe-sudo-keepalive", "immutable-streamed-stop-poll-evidence", "container-namespace-health-and-authenticated-model-binding", "secret-free-stdin-header-transport", "base-and-profile-dry-run-process-count-zero", "rocprof-pinned-fd-and-target-manifest", "roctx-run-session-case-and-library-binding"],
    "launcher": {"commit": LAUNCHER_COMMIT, "sha256": LAUNCHER_SHA},
    "runner": {"commit": RUNNER_COMMIT, "sha256": RUNNER_SHA},
    "capture_tool": {"commit": PROFILE_CAPTURE_COMMIT, "sha256": PROFILE_CAPTURE_SHA},
}


def validate_qa_test_manifest() -> None:
    automated = QA_ATTESTATION.get("automated_tests")
    if not isinstance(automated, dict) or set(automated) != {"schema_version", "aggregate", "suites"} or automated.get("schema_version") != "ullm.aq4_p2_exact_test_file_manifest.v1":
        raise HarnessError("QA exact test manifest schema differs")
    aggregate = automated.get("aggregate")
    suites = automated.get("suites")
    if not isinstance(aggregate, dict) or set(aggregate) != {"distinct_test_file_count", "collected", "passed", "failed", "deselected"} or not isinstance(suites, list) or not suites:
        raise HarnessError("QA exact test manifest aggregate differs")
    observed_paths: set[str] = set()
    collected = passed = failed = deselected = 0
    for suite in suites:
        exact_suite = {"name", "command", "collected", "passed", "failed", "deselected", "files"}
        if not isinstance(suite, dict) or set(suite) != exact_suite or not isinstance(suite.get("name"), str) or not suite["name"] or not isinstance(suite.get("files"), list) or not suite["files"]:
            raise HarnessError("QA exact test suite schema differs")
        suite_paths: list[str] = []
        suite_collected = suite_passed = 0
        for item in suite["files"]:
            if not isinstance(item, dict) or set(item) != {"path", "source_commit", "git_blob", "collected", "passed"}:
                raise HarnessError("QA exact test file schema differs")
            path = item.get("path")
            if not isinstance(path, str) or not path.startswith("tests/test_") or Path(path).is_absolute() or ".." in Path(path).parts or path in observed_paths:
                raise HarnessError("QA exact test file coverage differs")
            if not isinstance(item.get("source_commit"), str) or re.fullmatch(r"[0-9a-f]{40}", item["source_commit"]) is None or not isinstance(item.get("git_blob"), str) or re.fullmatch(r"[0-9a-f]{40}", item["git_blob"]) is None:
                raise HarnessError("QA exact test file Git identity differs")
            if type(item.get("collected")) is not int or type(item.get("passed")) is not int or item["collected"] <= 0 or item["passed"] != item["collected"]:
                raise HarnessError("QA exact test file counts differ")
            observed_paths.add(path)
            suite_paths.append(path)
            suite_collected += item["collected"]
            suite_passed += item["passed"]
        if suite.get("command") != ["python3", "-m", "pytest", "-q", *suite_paths] or suite.get("collected") != suite_collected or suite.get("passed") != suite_passed or suite.get("failed") != 0 or suite.get("deselected") != 0:
            raise HarnessError("QA exact test suite command/counts differ")
        collected += suite_collected
        passed += suite_passed
        failed += suite["failed"]
        deselected += suite["deselected"]
    expected = {"distinct_test_file_count": len(observed_paths), "collected": collected, "passed": passed, "failed": failed, "deselected": deselected}
    if aggregate != expected:
        raise HarnessError("QA exact test manifest sum differs")


def ready_document(harness_identity: dict[str, str], *, profile_diagnostic: bool = False) -> dict[str, Any]:
    validate_qa_test_manifest()
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
        "maintenance": {
            "service": SERVICE,
            "marker_required_before_stop": True,
            "restore_in_outer_finally": True,
            "same_pty_sudo_cache_required": True,
            "sudo_keepalive_seconds": 30,
            "secret_storage_forbidden": True,
            "formal_health_route": "docker-exec-openwebui-container-network-namespace",
            "host_direct_health_is_diagnostic_only": True,
            "authenticated_models_header_transport": "docker-exec-stdin-header-file",
            "stopped_gate_poll": {
                "timeout_seconds": STOP_POLL_TIMEOUT_SECONDS,
                "initial_interval_seconds": STOP_POLL_INITIAL_INTERVAL_SECONDS,
                "maximum_interval_seconds": STOP_POLL_MAX_INTERVAL_SECONDS,
                "required_consecutive_stable": STOP_POLL_STABLE_OBSERVATIONS,
                "sudo_keepalive_seconds": STOP_POLL_SUDO_KEEPALIVE_SECONDS,
                "maximum_probe_timeout_seconds": STOP_POLL_PROBE_TIMEOUT_SECONDS,
                "deadline_semantics": "fixed_absolute_monotonic_ns_checked_before_and_after_each_observation_and_probe",
                "transitional_amd_kfd_owner": "pre_stop_worker_pid_only",
                "transitional_lock_holder": "pre_stop_service_main_pid_only",
            "foreign_new_or_reappeared_owner": "immediate_fail_closed",
            "evidence": "atomic_immutable_per_poll_secret_safe_digests_and_parsed_pids",
            "kfd_scan": "bounded_enoent_rescan_and_fatal_non_enoent_source_diagnostics",
            "trusted_lock_substrate": {
                "directory": str(LOCK_SUBSTRATE_DIRECTORY),
                "owner": LOCK_SUBSTRATE_OWNER,
                "directory_mode": f"{LOCK_SUBSTRATE_MODE:04o}",
                "lock_mode": f"{LOCK_SUBSTRATE_LOCK_MODE:04o}",
                "create": "pinned_sudo_install_then_nonroot_o_excl_o_nofollow",
                "identity": "same_device_inode_from_stopped_poll_through_runner_and_cleanup",
                "cleanup": "same_inode_unlink_then_pinned_sudo_rmdir_before_unconditional_service_restore",
            },
        },
        },
        "trust": {
            "launcher": {"commit": LAUNCHER_COMMIT, "tree": LAUNCHER_TREE, "git_blob": LAUNCHER_GIT_BLOB, "sha256": LAUNCHER_SHA},
            "harness": harness_identity,
            "runner": {"commit": RUNNER_COMMIT, "sha256": RUNNER_SHA, "cli_ancestor_commit": RUNNER_CLI_ANCESTOR},
            "validator": {"commit": VALIDATOR_COMMIT, "sha256": LAUNCHER.VALIDATOR_SHA},
            "B": {"commit": B_COMMIT, "manifest_sha256": LAUNCHER.BINDING_MANIFEST_SHA},
            "resident": {"commit": RESIDENT_COMMIT, "sha256": LAUNCHER.RESIDENT_SHA},
            "production": {"manifest_sha256": LAUNCHER.SERVED_SHA, "worker_sha256": WORKER_SHA, "package_manifest_sha256": PACKAGE_MANIFEST_SHA, "package_content_sha256": PACKAGE_CONTENT_SHA},
            "container_health": {
                "docker": {"path": str(DOCKER), "sha256": DOCKER_SHA, "client_version": DOCKER_CLIENT_VERSION, "client_api_version": DOCKER_CLIENT_API_VERSION},
                "container": {"name": OPENWEBUI_CONTAINER_NAME, "id": OPENWEBUI_CONTAINER_ID, "image_id": OPENWEBUI_IMAGE_ID, "status": "running", "health": "healthy"},
                "network": {"name": OPENWEBUI_NETWORK_NAME, "id": OPENWEBUI_NETWORK_ID, "container_ip": OPENWEBUI_CONTAINER_IP, "gateway": OPENWEBUI_GATEWAY_IP},
                "curl": {"path": CONTAINER_CURL, "sha256": CONTAINER_CURL_SHA, "version": CONTAINER_CURL_VERSION, "version_output_sha256": CONTAINER_CURL_VERSION_SHA},
                "endpoints": {"healthz": GATEWAY_HEALTH_URL, "readyz": GATEWAY_READY_URL, "models": GATEWAY_MODELS_URL, "openwebui_health": OPENWEBUI_CONTAINER_HEALTH_URL, "model_id": GATEWAY_MODEL_ID},
            },
        },
        "qa_attestation_sha256": sha_bytes(pretty(QA_ATTESTATION)),
    }
    if not profile_diagnostic:
        value.pop("measurement_eligible")
    if profile_diagnostic:
        command = profile_capture_command()
        launcher_command = profile_launcher_command()
        launcher_binding = LAUNCHER.ready_profile_execute_binding()
        target_manifest = profile_target_command_manifest()
        value["profile_diagnostic"] = {
            "schema_version": "ullm.aq4_p3_diagnostic_rocprof_ready.v1",
            "status": "ready_for_one_profile_diagnostic",
            "measurement_eligible": False,
            "promotion_eligible": False,
            "maximum_invocations": 1,
            "output_no_reuse": True,
            "capture_tool": {
                "path": str(PROFILE_CAPTURE_TOOL),
                "commit": PROFILE_CAPTURE_COMMIT,
                "tree": PROFILE_CAPTURE_TREE,
                "git_blob": PROFILE_CAPTURE_GIT_BLOB,
                "sha256": PROFILE_CAPTURE_SHA,
            },
            "profiler": {"path": str(PROFILE_PROFILER), "resolved_path": str(PROFILE_PROFILER), "sha256": PROFILE_PROFILER_SHA},
            "command": command,
            "command_sha256": sha_bytes(canonical(command)),
            "target_launcher": {
                "schema_version": "ullm.aq4_p2_profile_target_launcher.v1",
                "path": str(LAUNCHER_PATH),
                "commit": LAUNCHER_COMMIT,
                "sha256": LAUNCHER_SHA,
                "command": launcher_command,
                "command_sha256": sha_bytes(canonical(launcher_command)),
                "binding_sha256": sha_bytes(canonical(launcher_binding)),
                "manifest": {
                    "path": str(PROFILE_TARGET_COMMAND_MANIFEST),
                    "sha256": sha_bytes(pretty(target_manifest)),
                    "manifest_sha256": target_manifest["manifest_sha256"],
                },
                "run_id": LAUNCHER.PROFILE_RUN_ID,
                "evidence_output": str(LAUNCHER.PROFILE_EVIDENCE_OUTPUT),
                "runner_output": str(LAUNCHER.PROFILE_RUN_OUTPUT),
            },
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
    files = [("ready-binding.json", ready_raw), ("qa-attestation.json", attestation_raw), ("harness-trust.json", trust_raw)]
    if profile_diagnostic:
        files.append(("target-command-manifest.json", pretty(profile_target_command_manifest())))
    for name, raw in files:
        LAUNCHER.atomic_write(root, name, raw)
    sums = "".join(f"{sha_bytes(raw)}  {name}\n" for name, raw in sorted(files)).encode("ascii")
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
    expected_names = {"ready-binding.json", "qa-attestation.json", "harness-trust.json", "SHA256SUMS"}
    if profile_diagnostic:
        expected_names.add("target-command-manifest.json")
    if {item.name for item in root.iterdir()} != expected_names:
        raise HarnessError("ready artifact path/coverage differs")
    ready_raw, _ = LAUNCHER.read_regular(path, "ready binding"); value = LAUNCHER.parse_json(ready_raw, "ready binding")
    trust_raw, _ = LAUNCHER.read_regular(trust_path, "harness trust"); trust = LAUNCHER.parse_json(trust_raw, "harness trust")
    attestation_raw, _ = LAUNCHER.read_regular(attestation_path, "QA attestation"); attestation = LAUNCHER.parse_json(attestation_raw, "QA attestation")
    target_raw = None
    if profile_diagnostic:
        target_raw, _ = LAUNCHER.read_regular(PROFILE_TARGET_COMMAND_MANIFEST, "profile target command manifest")
        target = LAUNCHER.parse_json(target_raw, "profile target command manifest")
        if target != profile_target_command_manifest():
            raise HarnessError("profile target command manifest differs")
    expected = ready_document({key: trust[key] for key in ("path", "commit", "tree", "git_blob", "sha256")}, profile_diagnostic=profile_diagnostic)
    if value != expected or attestation != QA_ATTESTATION or trust.get("schema_version") != "ullm.aq4_p2_resident_maintenance_harness_trust.v1" or trust.get("status") != "ready_for_one_case" or trust.get("execution_mode") != value["execution_mode"] or trust.get("actual_eligible") is not True or trust.get("ready_binding_sha256") != sha_bytes(ready_raw):
        raise HarnessError("ready artifact semantic binding differs")
    self_sha = LAUNCHER.sha_file(Path(__file__).resolve(), "maintenance harness self")[0]
    if trust.get("path") != str(Path(__file__).resolve()) or trust.get("sha256") != self_sha:
        raise HarnessError("maintenance harness self differs")
    sum_inputs = [("harness-trust.json", trust_raw), ("qa-attestation.json", attestation_raw), ("ready-binding.json", ready_raw)]
    if target_raw is not None:
        sum_inputs.append(("target-command-manifest.json", target_raw))
    expected_sums = "".join(f"{sha_bytes(raw)}  {name}\n" for name, raw in sorted(sum_inputs)).encode("ascii")
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
    evidence = {"schema_version": "ullm.aq4_p2_resident_maintenance.v1", "status": "passed", "mode": "dry-run", "execution_mode": value["execution_mode"], "actual_eligible": value["actual_eligible"], "promotion_eligible": False, "run_id": value["authorization"]["run_id"], "process_counts": {"sudo": 0, "sudo_keepalive": 0, "systemctl_stop": 0, "launcher": 0, "systemctl_start": 0, "rocprof": 0, "capture_tool": 0, "docker": 0, "docker_exec": 0, "container_curl": 0, "container_curl_total": 0, "container_curl_version": 0, "container_curl_endpoint": 0, "stopped_gate_polls": 0, "stopped_gate_probe_commands": 0}, "service_touched": False, "gpu_command_executed": False, "model_load_executed": False, "ready_binding_sha256": LAUNCHER.sha_file(ready_path, "ready binding")[0]}
    if value["execution_mode"] == "profile_diagnostic":
        evidence["profile_diagnostic"] = {"command": value["profile_diagnostic"]["command"], "command_sha256": value["profile_diagnostic"]["command_sha256"], "capture_executed": False, "measurement_eligible": False, "promotion_eligible": False}
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
    elif value.get("authorization", {}).get("rocprof_wrapper_required") is not False or "profile_diagnostic" in value:
        raise HarnessError("normal one-case profile contract differs")
    for path, label in ((output, "maintenance evidence"), (Path(value["launcher_binding"]["runner_output"]), "runner output"), (Path(value["launcher_binding"]["evidence_output"]), "launcher evidence")):
        LAUNCHER.reject_symlink_components(path, label, allow_missing_leaf=True)
        if path.exists() or path.is_symlink():
            raise HarnessError(f"{label} already exists")
    if profile_diagnostic:
        LAUNCHER.ensure_directory_chain(
            PROFILE_OUTPUT_DIRECTORY.parent,
            "profile capture output parent",
        )
        LAUNCHER.reject_symlink_components(
            PROFILE_OUTPUT_DIRECTORY,
            "profile capture output",
            allow_missing_leaf=True,
        )
        if PROFILE_OUTPUT_DIRECTORY.exists() or PROFILE_OUTPUT_DIRECTORY.is_symlink():
            raise HarnessError("profile capture output already exists")
        trust_records = [dependencies.profile_trust(value["profile_diagnostic"], "before-start")]
    else:
        trust_records = []
    output.mkdir(mode=0o700)
    evidence: dict[str, Any] = {"schema_version": "ullm.aq4_p2_resident_maintenance.v1", "status": "failed", "mode": "execute", "execution_mode": value["execution_mode"], "run_id": run_id, "promotion_eligible": False, "profile_trust": trust_records, "capture": None, "sequence": [], "commands": [], "pre_stop": None, "stopped_gates": None, "stopped_gate_poll": None, "lock_substrate": None, "lock_substrate_cleanup": None, "launcher": None, "restore": None, "failure": None, "process_counts": {"sudo": 0, "sudo_keepalive": 0, "systemctl_stop": 0, "launcher": 0, "systemctl_start": 0, "capture_tool": 0, "rocprof": 0, "docker": 0, "docker_exec": 0, "container_curl": 0, "container_curl_total": 0, "container_curl_version": 0, "container_curl_endpoint": 0, "stopped_gate_polls": 0, "stopped_gate_probe_commands": 0}, "safety": {"service_touched": False, "service_stopped": False, "gpu_command_executed": False, "model_load_executed": False}, "secret_material_recorded": False}
    stop_attempted = False; capture_attempted = False; pre: dict[str, Any] | None = None; substrate: LockSubstrate | None = None; runner_finished = False; runner_evidence: dict[str, Any] | None = None; code = 1; stage = "sudo-prevalidate"
    try:
        record = _sudo_valid(dependencies.run, "sudo-prevalidate"); evidence["commands"].append(record); evidence["process_counts"]["sudo"] += 1; evidence["sequence"].append("sudo-prevalidate")
        stage = "pre-stop-snapshot"; pre = capture_running(dependencies); evidence["pre_stop"] = pre; evidence["sequence"].append("pre-stop-snapshot")
        for name, count in pre["health"]["formal"]["process_counts"].items():
            evidence["process_counts"][name] += count
        marker = {"schema_version": "ullm.aq4_p2_resident_maintenance_marker.v1", "run_id": run_id, "restore_required": True, "service": SERVICE, "pre_stop_sha256": sha_bytes(canonical(pre)), "created_unix_ns": time.time_ns()}
        LAUNCHER.atomic_write(output, "maintenance-marker.json", pretty(marker)); evidence["marker"] = {"path": str(output / "maintenance-marker.json"), "sha256": sha_bytes(pretty(marker))}; evidence["sequence"].append("durable-marker")
        stage = "service-stop"; evidence["commands"].append(_sudo_valid(dependencies.run, "sudo-before-stop")); evidence["process_counts"]["sudo"] += 1
        stop_attempted = True; evidence["safety"]["service_touched"] = True
        stopped, record = _command(dependencies.run, [str(LAUNCHER.SUDO), "-n", str(LAUNCHER.SYSTEMCTL), "stop", SERVICE], "service-stop"); evidence["commands"].append(record); evidence["process_counts"]["systemctl_stop"] = 1
        if stopped.returncode != 0 or stopped.stdout or stopped.stderr:
            raise HarnessError("service stop failed")
        evidence["safety"]["service_stopped"] = True; evidence["sequence"].append("service-stopped")
        stage = "lock-substrate"
        if dependencies.lock_substrate_prepare is not None:
            substrate = dependencies.lock_substrate_prepare(dependencies.run)
            if not isinstance(substrate, LockSubstrate):
                raise HarnessError("trusted lock substrate preparation contract differs")
            evidence["lock_substrate"] = substrate.evidence
            evidence["process_counts"]["lock_substrate_install"] = 1
        stage = "stopped-gates"
        def stopped_poll_keepalive(attempt: int, timeout: float) -> None:
            record = _sudo_valid(dependencies.run, f"sudo-stopped-poll-keepalive-{attempt}", timeout=timeout)
            evidence["commands"].append(record)
            evidence["process_counts"]["sudo"] += 1
            evidence["process_counts"]["sudo_keepalive"] += 1
        gates, poll_evidence = poll_stopped_gates(output, pre["worker"]["pid"], pre["service"]["main_pid"], dependencies, stopped_poll_keepalive, substrate)
        evidence["stopped_gate_poll"] = poll_evidence
        evidence["process_counts"]["stopped_gate_polls"] = poll_evidence["poll_count"]
        evidence["process_counts"]["stopped_gate_probe_commands"] = poll_evidence["probe_command_count"]
        evidence["stopped_gates"] = gates
        if not isinstance(gates, dict) or gates.get("passed") is not True or gates.get("services") != [{"unit": "ullm-openai.service", "active_state": "inactive", "sub_state": "dead", "main_pid": 0}, {"unit": "llama-qwen35-udq4.service", "active_state": "inactive", "sub_state": "dead", "main_pid": 0}] or gates.get("old_worker_pids") != [] or gates.get("amd_smi_owners") != [] or gates.get("kfd_owners") != [] or gates.get("lock", {}).get("free") is not True:
            raise HarnessError("stopped live gates did not reach stable2")
        evidence["sequence"].append("stopped-gates")
        evidence["safety"]["gpu_command_executed"] = "unknown"; evidence["safety"]["model_load_executed"] = "unknown"
        if profile_diagnostic:
            stage = "profile-capture-before"
            evidence["profile_trust"].append(dependencies.profile_trust(value["profile_diagnostic"], "capture-before"))
            stage = "profile-capture"; capture_attempted = True; evidence["process_counts"]["capture_tool"] = 1; evidence["sequence"].append("profile-capture")
            try:
                outcome = dependencies.profile_capture(value["profile_diagnostic"])
                runner_finished = True
                runner_evidence = outcome if isinstance(outcome, dict) else None
            finally:
                evidence["profile_trust"].append(dependencies.profile_trust(value["profile_diagnostic"], "capture-after"))
            required = {"completed", "started", "timed_out", "cleanup_passed", "children_remaining", "rocprof_started", "launcher_started", "launcher_status", "gpu_command_executed", "model_load_executed"}
            if not isinstance(outcome, dict) or set(outcome) != required or not isinstance(outcome.get("completed"), subprocess.CompletedProcess) or outcome.get("started") is not True or type(outcome.get("timed_out")) is not bool or type(outcome.get("cleanup_passed")) is not bool or not isinstance(outcome.get("children_remaining"), list) or type(outcome.get("rocprof_started")) is not bool or type(outcome.get("launcher_started")) is not bool:
                raise HarnessError("profile capture outcome contract differs")
            completed = outcome["completed"]
            evidence["process_counts"]["rocprof"] = int(outcome["rocprof_started"]); evidence["process_counts"]["launcher"] = int(outcome["launcher_started"])
            evidence["safety"]["gpu_command_executed"] = outcome["gpu_command_executed"]; evidence["safety"]["model_load_executed"] = outcome["model_load_executed"]
            evidence["capture"] = {"command": completed.args, "exit_code": completed.returncode, "stdout_sha256": sha_bytes(completed.stdout), "stderr_sha256": sha_bytes(completed.stderr), "timed_out": outcome["timed_out"], "cleanup_passed": outcome["cleanup_passed"], "children_remaining": outcome["children_remaining"], "launcher_status": outcome["launcher_status"]}
            if outcome["timed_out"] or not outcome["cleanup_passed"] or outcome["children_remaining"] or completed.returncode != 0 or completed.stderr or outcome["rocprof_started"] is not True or outcome["launcher_started"] is not True or outcome["launcher_status"] != "passed":
                raise HarnessError("profile capture/launcher failed or left a child process")
        else:
            stage = "launcher"; evidence["process_counts"]["launcher"] = 1; evidence["sequence"].append("launcher")
            try:
                launcher_code, launcher_evidence = dependencies.launcher_execute(value["launcher_binding"])
                runner_finished = True
                runner_evidence = launcher_evidence if isinstance(launcher_evidence, dict) else None
            except Exception:
                runner_finished = False
                raise
            evidence["launcher"] = {"code": launcher_code, "status": launcher_evidence.get("status"), "safety": launcher_evidence.get("safety"), "failure": launcher_evidence.get("failure"), "children_remaining": launcher_evidence.get("children_remaining", [])}
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
            cleanup_error: str | None = None
            if substrate is not None:
                try:
                    children = []
                    if isinstance(runner_evidence, dict):
                        for key in ("children_remaining", "child_pids", "children"):
                            value_children = runner_evidence.get(key)
                            if isinstance(value_children, list):
                                children.extend(value_children)
                    if not runner_finished and (capture_attempted or evidence["process_counts"]["launcher"]):
                        # A failed/aborted runner has no trustworthy child
                        # inventory.  Keep the service stopped only long
                        # enough to record cleanup failure; never unlink a
                        # substrate while an unknown child may still hold it.
                        children = [-1]
                    cleanup = (
                        dependencies.lock_substrate_cleanup(
                            substrate,
                            dependencies.run,
                            runner_finished=runner_finished,
                            runner_children=children,
                        )
                        if dependencies.lock_substrate_cleanup is not None
                        else cleanup_lock_substrate(
                            substrate,
                            dependencies.run,
                            runner_finished=runner_finished,
                            runner_children=children,
                        )
                    )
                    if not isinstance(cleanup, dict) or cleanup.get("passed") is not True or cleanup.get("secret_material_recorded") is not False:
                        raise HarnessError("trusted lock substrate cleanup contract differs")
                    evidence["lock_substrate_cleanup"] = cleanup
                    evidence["process_counts"]["lock_substrate_rmdir"] = 1
                except Exception as error:
                    cleanup_error = str(error)
                    evidence["lock_substrate_cleanup"] = {"passed": False, "error": cleanup_error, "secret_material_recorded": False}
                    evidence["failure"] = {"stage": "lock-substrate-cleanup", "reason": cleanup_error, "launcher_started": evidence["process_counts"]["launcher"] == 1}
                    code = 1
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
                for name, count in post["health"]["formal"]["process_counts"].items():
                    evidence["process_counts"][name] += count
                evidence["sequence"].append("service-restored")
            except (HarnessError, OSError, ValueError, subprocess.SubprocessError) as error:
                restore_error = str(error); code = 1
            evidence["restore"] = {"attempted": True, "passed": restore_error is None, "error": restore_error, "post_start": post, "lock_substrate_cleanup_passed": cleanup_error is None}
        else:
            evidence["restore"] = {"attempted": False, "passed": True, "error": None, "post_start": None}
        if profile_diagnostic:
            try:
                evidence["profile_trust"].append(dependencies.profile_trust(value["profile_diagnostic"], "finalize-before"))
            except (HarnessError, LAUNCHER.LauncherError, OSError, ValueError) as error:
                evidence["failure"] = {"stage": "profile-finalize-trust", "reason": str(error), "launcher_started": evidence["process_counts"]["launcher"] == 1}
                code = 1
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
