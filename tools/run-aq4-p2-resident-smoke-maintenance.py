#!/usr/bin/env python3
"""Single-use maintenance harness around the immutable AQ4 P2 smoke launcher."""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
import pwd
import re
import stat
import subprocess
import sys
import time
import types
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

LAUNCHER_COMMIT = "288b165c707413aac01753b8254ea98fe843308f"
LAUNCHER_TREE = "479268bfbf63f72f4103164bfddda31d41e1ecb6"
LAUNCHER_GIT_BLOB = "9bcda3dad770a7103b018f58544a2bbbb3cf41d9"
LAUNCHER_SHA = "52908fc4790fbc83e0a95decaad64a5ab1427b7a3c4367f21ef7662927f2bbfc"
RUNNER_COMMIT = "81ceebb13518f590b5dbf439cd00b35e508c1c3f"
RUNNER_SHA = "5d4cf385a83961f8aedc37d36c3e4625d783ec7ddd6b17de4f93648516d42354"
RUNNER_CLI_ANCESTOR = "ee341c019d873f7c250adbb81414d58b5285a454"
VALIDATOR_COMMIT = "a44074278d4bbd5e243153ab8c5be272489e23a2"
B_COMMIT = "bad728000405a711dec4faf10d4a60393bf9d7e8"
RESIDENT_COMMIT = "81ceebb13518f590b5dbf439cd00b35e508c1c3f"
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
PROFILE_MAINTENANCE_EVIDENCE = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-maintenance-evidence-v4"
PROFILE_DRY_RUN_EVIDENCE = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-ready-dry-run-v4"
PROFILE_CAPTURE_TOOL = ROOT / "tools/capture-aq4-p3-diagnostic-profile.py"
PROFILE_CAPTURE_COMMIT = "0e8bf9f47583d10cf4daf1092aef5a0e388aa496"
PROFILE_CAPTURE_TREE = "c15f588e88bcba577c1a36a9b9a2bc4d720f1d10"
PROFILE_CAPTURE_GIT_BLOB = "df5af6b86fff19b977c1bfffac243842d533bb5e"
PROFILE_CAPTURE_SHA = "b66ef14ebaaa9b2828dbe17e93aeed13595284e361776e7c67dc197e318f01af"
PROFILE_PROFILER = Path("/opt/rocm-7.2.1/bin/rocprofv3")
PROFILE_PROFILER_SHA = "13060810d6b80653631b14f0f5e33ea160c2b79a6a3a4c6850142010b48b8ec8"
PROFILE_OUTPUT_DIRECTORY = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p3/aq4-p3-diagnostic-rocprof-capture-v4"
PROFILE_OUTPUT_NAME = "aq4-p3-diagnostic"
PROFILE_ARTIFACT = PROFILE_OUTPUT_DIRECTORY / "capture-artifact.json"
PROFILE_TIMEOUT_SECONDS = 1800
PROFILE_CAPTURE_SCHEMA = "ullm.aq4_p3_diagnostic_rocprof_capture.v1"
PROFILE_CAPTURE_FAILURE_SCHEMA = "ullm.aq4_p3_diagnostic_rocprof_failure.v2"
HISTORICAL_PROFILE_CAPTURE_FAILURE_SCHEMA = "ullm.aq4_p3_diagnostic_rocprof_failure.v1"
PROFILE_CAPTURE_FAILURE_NAME = "capture-failure.json"
READY_CANDIDATE_AUDIT_SCHEMA = "ullm.aq4_p2_ready_candidate_audit.v1"
READY_CANDIDATE_CAPTURE_SCHEMA = "ullm.aq4_p3_ready_candidate_capture.v1"
READY_CANDIDATE_MARKER_PREFIX = b"ULLM_AQ4_READY_CANDIDATE_AUDIT_V1 "
MAX_READY_CANDIDATE_MARKER_BYTES = 16 * 1024
MAX_READY_CANDIDATE_RAW_BYTES = 1024 * 1024
READY_CANDIDATE_JSON_TYPES = {
    "absent", "null", "boolean", "integer", "number", "string", "array", "object",
}
READY_CANDIDATE_PREDICATES = {
    "field_set_exact",
    "event_is_ready",
    "schema_version_exact",
    "model_loads_is_integer",
    "model_loads_is_one",
    "resident_session_id_is_string",
    "resident_session_id_nonempty",
}
SERVICE = "ullm-openai.service"
WORKER = ROOT / "target/reasoning-v2/release/ullm-aq4-worker"
WORKER_SHA = "177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d"
PACKAGE_ROOT = Path("/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package")
PACKAGE_MANIFEST = PACKAGE_ROOT / "manifest.json"
PACKAGE_MANIFEST_SHA = "a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad"
PACKAGE_CONTENT_SHA = "a24774432d3f0b7f175dc761ef9a53df1fed901dd02f825e8542b17181f004b1"
PACKAGE_INTEGRITY_IDENTITY_SHA = "c0382f0cabec53f07e45c7be5f1d1618c18fe0c16de98a0901b97e217ca5e267"
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
RESTORE_TIMEOUT_SECONDS = 120.0
RESTORE_POLL_INTERVAL_SECONDS = 1.0
RESTORE_PROBE_TIMEOUT_SECONDS = 10.0
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
UNKNOWN_LIFECYCLE_STATE = "unknown"


class HarnessError(ValueError):
    pass


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pretty(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


@dataclass(frozen=True)
class PackageTreeEntry:
    relative_path: str
    device: int
    inode: int
    mode: int
    nlink: int
    size: int
    mtime_ns: int
    ctime_ns: int

    def evidence(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "nlink": self.nlink,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


@dataclass(frozen=True)
class PackageTreeSnapshot:
    entries: tuple[PackageTreeEntry, ...]
    identity_sha256: str
    entry_count: int
    file_count: int
    directory_count: int
    symlink_count: int
    special_count: int
    bytes: int

    def evidence(self, stage: str) -> dict[str, Any]:
        return {
            "stage": stage,
            "identity_sha256": self.identity_sha256,
            "entry_count": self.entry_count,
            "file_count": self.file_count,
            "directory_count": self.directory_count,
            "symlink_count": self.symlink_count,
            "special_count": self.special_count,
            "bytes": self.bytes,
            "identity_fields": [
                "relative_path",
                "device",
                "inode",
                "mode",
                "nlink",
                "size",
                "mtime_ns",
                "ctime_ns",
            ],
        }


def package_tree_snapshot(root: Path) -> PackageTreeSnapshot:
    """Capture a no-follow identity snapshot for every package tree entry."""

    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise HarnessError("package root metadata is unavailable") from error
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise HarnessError("package root is invalid")

    def entry(relative_path: str, metadata: os.stat_result) -> PackageTreeEntry:
        return PackageTreeEntry(
            relative_path=relative_path,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mode=metadata.st_mode,
            nlink=metadata.st_nlink,
            size=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
            ctime_ns=metadata.st_ctime_ns,
        )

    entries = [entry(".", root_metadata)]
    pending = [(root, "")]
    while pending:
        directory, prefix = pending.pop()
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise HarnessError("package tree enumeration failed") from error
        for child in children:
            relative = f"{prefix}/{child.name}" if prefix else child.name
            try:
                metadata = child.stat(follow_symlinks=False)
            except OSError as error:
                raise HarnessError("package tree entry metadata is unavailable") from error
            entries.append(entry(relative, metadata))
            if stat.S_ISDIR(metadata.st_mode):
                pending.append((Path(child.path), relative))
    ordered = tuple(sorted(entries, key=lambda item: item.relative_path))
    digest = hashlib.sha256()
    for item in ordered:
        digest.update(canonical(item.evidence()))
        digest.update(b"\n")
    return PackageTreeSnapshot(
        entries=ordered,
        identity_sha256=digest.hexdigest(),
        entry_count=len(ordered),
        file_count=sum(stat.S_ISREG(item.mode) for item in ordered),
        directory_count=sum(stat.S_ISDIR(item.mode) for item in ordered),
        symlink_count=sum(stat.S_ISLNK(item.mode) for item in ordered),
        special_count=sum(
            not (stat.S_ISREG(item.mode) or stat.S_ISDIR(item.mode) or stat.S_ISLNK(item.mode))
            for item in ordered
        ),
        bytes=sum(item.size for item in ordered if stat.S_ISREG(item.mode)),
    )


def package_integrity_identity(content_sha256: str, tree: PackageTreeSnapshot) -> str:
    return sha_bytes(
        canonical(
            {
                "schema_version": "ullm.aq4_package_integrity_identity.v1",
                "full_content_sha256": content_sha256,
                "tree_metadata_identity_sha256": tree.identity_sha256,
                "entry_count": tree.entry_count,
                "file_count": tree.file_count,
                "directory_count": tree.directory_count,
                "symlink_count": tree.symlink_count,
                "special_count": tree.special_count,
                "bytes": tree.bytes,
            }
        )
    )


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


def _service_snapshot(run: Callable[..., subprocess.CompletedProcess[bytes]], *, timeout: float = 30.0) -> tuple[dict[str, Any], dict[str, Any]]:
    argv = [str(LAUNCHER.SYSTEMCTL), "show", SERVICE, "--property=ActiveState", "--property=SubState", "--property=MainPID", "--property=NRestarts", "--property=ControlGroup", "--no-pager"]
    completed, record = _command(run, argv, "service-running", timeout=timeout)
    try:
        values = dict(line.split("=", 1) for line in completed.stdout.decode().splitlines())
        main_pid = int(values["MainPID"]); restarts = int(values["NRestarts"])
    except (UnicodeError, ValueError, KeyError) as error:
        raise HarnessError("running service snapshot schema differs") from error
    if completed.returncode != 0 or completed.stderr or set(values) != {"ActiveState", "SubState", "MainPID", "NRestarts", "ControlGroup"} or values["ActiveState"] != "active" or values["SubState"] != "running" or main_pid <= 0 or restarts < 0 or values["ControlGroup"] != "/system.slice/ullm-openai.service":
        raise HarnessError("service is not healthy and active")
    return {"unit": SERVICE, "active_state": "active", "sub_state": "running", "main_pid": main_pid, "nrestarts": restarts, "control_group": values["ControlGroup"]}, record


def _worker_pid(run: Callable[..., subprocess.CompletedProcess[bytes]], *, timeout: float = 30.0) -> tuple[int, dict[str, Any]]:
    argv = [str(LAUNCHER.PGREP), "-f", "-x", f"{WORKER}.*"]
    completed, record = _command(run, argv, "worker-running", timeout=timeout)
    try:
        pids = [int(item) for item in completed.stdout.decode().splitlines() if item]
    except (UnicodeError, ValueError) as error:
        raise HarnessError("worker PID output differs") from error
    if completed.returncode != 0 or completed.stderr or len(pids) != 1 or pids[0] <= 0:
        raise HarnessError("running worker is not unique")
    return pids[0], record


def _gpu_identity(run: Callable[..., subprocess.CompletedProcess[bytes]], *, timeout: float = 30.0) -> tuple[dict[str, Any], dict[str, Any]]:
    completed, record = _command(run, [str(LAUNCHER.AMD_SMI), "list", "--json"], "gpu-identity", timeout=timeout)
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
    launcher_execute: Callable[..., tuple[int, dict[str, Any]]]
    profile_capture: Callable[[dict[str, Any]], dict[str, Any]]
    profile_trust: Callable[[dict[str, Any], str], dict[str, Any]]
    sleep: Callable[[float], None]
    monotonic_ns: Callable[[], int]
    package_metadata: Callable[[Path], PackageTreeSnapshot] = package_tree_snapshot
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


def _bounded_run(
    run: Callable[..., subprocess.CompletedProcess[bytes]],
    maximum_timeout_seconds: float,
) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    if maximum_timeout_seconds <= 0:
        raise HarnessError("dynamic readiness probe timeout is exhausted")

    def bounded(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        requested = kwargs.get("timeout", maximum_timeout_seconds)
        if not isinstance(requested, (int, float)) or requested <= 0:
            requested = maximum_timeout_seconds
        kwargs["timeout"] = min(float(requested), maximum_timeout_seconds)
        return run(argv, **kwargs)

    return bounded


def _package_tree_difference(expected: PackageTreeSnapshot, actual: PackageTreeSnapshot) -> dict[str, Any] | None:
    if expected.entries == actual.entries:
        return None
    expected_by_path = {item.relative_path: item for item in expected.entries}
    actual_by_path = {item.relative_path: item for item in actual.entries}
    added = sorted(set(actual_by_path) - set(expected_by_path))
    removed = sorted(set(expected_by_path) - set(actual_by_path))
    changed = sorted(path for path in set(expected_by_path) & set(actual_by_path) if expected_by_path[path] != actual_by_path[path])
    kind = "added" if added else "removed" if removed else "metadata_changed"
    paths = added or removed or changed
    result: dict[str, Any] = {"kind": kind, "relative_path": paths[0] if paths else None}
    if changed:
        path = changed[0]
        before = expected_by_path[path]
        after = actual_by_path[path]
        result["changed_fields"] = [
            field
            for field in ("device", "inode", "mode", "nlink", "size", "mtime_ns", "ctime_ns")
            if getattr(before, field) != getattr(after, field)
        ]
    return result


def capture_package_integrity(
    dependencies: Dependencies,
    evidence: dict[str, Any],
    expected_integrity_identity_sha256: str,
) -> PackageTreeSnapshot:
    """Run the one expensive content hash and bind it to stable tree metadata."""

    evidence.update(
        {
            "stage": "pre-stop",
            "full_hash_count": 0,
            "full_content": None,
            "tree_identity": None,
            "error": None,
        }
    )
    before = dependencies.package_metadata(PACKAGE_ROOT)
    started_ns = dependencies.monotonic_ns()
    evidence["full_hash_count"] = 1
    try:
        content_sha256 = dependencies.package_hash(PACKAGE_ROOT)
    except Exception as error:
        evidence["full_content"] = {
            "stage": "pre-stop-full-content-hash",
            "passed": False,
            "duration_ns": max(0, dependencies.monotonic_ns() - started_ns),
        }
        evidence["error"] = f"{type(error).__name__}: {error}"
        raise
    finished_ns = dependencies.monotonic_ns()
    after = dependencies.package_metadata(PACKAGE_ROOT)
    difference = _package_tree_difference(before, after)
    evidence["full_content"] = {
        "stage": "pre-stop-full-content-hash",
        "passed": content_sha256 == PACKAGE_CONTENT_SHA and difference is None,
        "sha256": content_sha256,
        "duration_ns": max(0, finished_ns - started_ns),
        "file_count": after.file_count,
        "bytes": after.bytes,
    }
    evidence["tree_identity"] = {
        **after.evidence("pre-stop-tree-metadata"),
        "stable_across_full_hash": difference is None,
        "difference": difference,
    }
    observed_integrity_identity = package_integrity_identity(content_sha256, after)
    evidence["integrity_identity"] = {
        "schema_version": "ullm.aq4_package_integrity_identity.v1",
        "expected_sha256": expected_integrity_identity_sha256,
        "observed_sha256": observed_integrity_identity,
        "passed": observed_integrity_identity == expected_integrity_identity_sha256,
    }
    if content_sha256 != PACKAGE_CONTENT_SHA:
        evidence["error"] = "production package full content hash differs"
        raise HarnessError(evidence["error"])
    if difference is not None:
        evidence["error"] = "production package tree metadata changed during full content hash"
        raise HarnessError(evidence["error"])
    if SHA_RE.fullmatch(expected_integrity_identity_sha256) is None or observed_integrity_identity != expected_integrity_identity_sha256:
        evidence["error"] = "production package trusted integrity identity differs"
        raise HarnessError(evidence["error"])
    return after


def capture_running(
    dependencies: Dependencies,
    previous: dict[str, Any] | None = None,
    *,
    probe_timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Capture only lightweight dynamic production readiness."""

    bounded_run = _bounded_run(dependencies.run, probe_timeout_seconds)
    service, service_record = _service_snapshot(bounded_run, timeout=probe_timeout_seconds)
    worker_pid, worker_record = _worker_pid(bounded_run, timeout=probe_timeout_seconds)
    gpu, gpu_record = _gpu_identity(bounded_run, timeout=probe_timeout_seconds)
    manifest_sha = LAUNCHER.sha_file(LAUNCHER.SERVED_MANIFEST, "active manifest")[0]
    worker_sha = hash_regular_with_nlink(WORKER, "active worker", 1)
    package_manifest_sha = LAUNCHER.sha_file(PACKAGE_MANIFEST, "package manifest")[0]
    if manifest_sha != LAUNCHER.SERVED_SHA or worker_sha != WORKER_SHA or package_manifest_sha != PACKAGE_MANIFEST_SHA:
        raise HarnessError("production manifest/worker/package hash differs")
    if not dependencies.lock_busy():
        raise HarnessError("production service does not hold device lock")
    owners = dependencies.owner_probe(bounded_run, worker_pid)
    container_health = dependencies.container_health(bounded_run)
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
    service_epoch = None
    if previous is not None:
        service_epoch = {
            "restart_kind": "explicit_systemctl_stop_start",
            "main_pid_changed": service["main_pid"] != previous["service"]["main_pid"],
            "worker_pid_changed": worker_pid != previous["worker"]["pid"],
            "nrestarts_before": previous["service"]["nrestarts"],
            "nrestarts_after": service["nrestarts"],
            "nrestarts_semantics": "explicit_stop_start_resets_automatic_restart_counter_to_zero",
            "nrestarts_reset_to_zero": service["nrestarts"] == 0,
            "control_group_unchanged": service["control_group"] == previous["service"]["control_group"],
        }
        if not all(
            service_epoch[key]
            for key in ("main_pid_changed", "worker_pid_changed", "nrestarts_reset_to_zero", "control_group_unchanged")
        ):
            raise HarnessError("restored explicit service epoch/NRestarts semantics differ")
    return {
        "service": service, "worker": {"path": str(WORKER), "pid": worker_pid, "sha256": worker_sha}, "gpu": gpu,
        "owners": owners, "lock": {"path": str(LAUNCHER.LOCK_PATH), "busy": True},
        "hashes": {"served_manifest_sha256": manifest_sha, "worker_sha256": worker_sha, "package_manifest_sha256": package_manifest_sha},
        "service_epoch": service_epoch,
        "health": {"formal": container_health, "host_route_diagnostics": host_diagnostics}, "commands": [service_record, worker_record, gpu_record],
    }


def _default_launcher_execute(
    binding: dict[str, Any],
    *,
    profile_runner_executor: Callable[..., dict[str, Any]] | None = None,
) -> tuple[int, dict[str, Any]]:
    gate_provider = lambda: LAUNCHER.collect_execute_gates(environment=dict(LAUNCHER.EXECUTE_ENV))
    return LAUNCHER.execute_bound(
        binding,
        Path(binding["evidence_output"]),
        Path(binding["runner_output"]),
        binding["run_id"],
        trusted_launcher_sha=LAUNCHER_SHA,
        gate_provider=gate_provider,
        profile_runner_executor=profile_runner_executor,
    )


class ProfileTrustGuard:
    def __init__(self) -> None:
        self.snapshot = LAUNCHER.Snapshot()
        self.initialized = False
        self.capture_tool_raw: bytes | None = None

    def __call__(self, contract: dict[str, Any], stage: str) -> dict[str, Any]:
        allowed = {"before-start", "capture-before", "capture-after", "finalize-before"}
        if stage not in allowed:
            raise HarnessError("profile trust stage differs")
        if contract.get("execution_boundary") != {
            "order": ["maintenance", "launcher", "validator", "gates", "capture", "rocprof", "runner"],
            "runner_profiled": True,
            "validator_profiled": False,
            "gates_profiled": False,
        } or contract.get("target_runner") != {
            "generated_by": "launcher_after_live_preflight",
            "file_name": LAUNCHER.PROFILE_RUNNER_TARGET_MANIFEST_NAME,
            "fresh_per_execution": True,
            "environment": "exact_execute_environment",
            "maximum_invocations": 1,
        }:
            raise HarnessError("profile capture/runner execution boundary differs")
        if not self.initialized:
            if stage != "before-start":
                raise HarnessError("profile trust was not initialized before capture")
            self.capture_tool_raw = self.snapshot.file(PROFILE_CAPTURE_TOOL, PROFILE_CAPTURE_SHA, "profile capture tool")
            self.snapshot.file(PROFILE_PROFILER, PROFILE_PROFILER_SHA, "profile profiler")
            self.snapshot.file(LAUNCHER.PYTHON, LAUNCHER.PYTHON_SHA, "profile target Python")
            self.snapshot.file(LAUNCHER_PATH, LAUNCHER_SHA, "profile target launcher")
            self.initialized = True
        self.snapshot.verify()
        return {
            "stage": stage,
            "passed": True,
            "capture_tool_sha256": PROFILE_CAPTURE_SHA,
            "profiler_sha256": PROFILE_PROFILER_SHA,
            "python_sha256": LAUNCHER.PYTHON_SHA,
            "launcher_sha256": LAUNCHER_SHA,
            "target_manifest_source": "fresh_launcher_generated_after_live_preflight",
            "execution_boundary_sha256": sha_bytes(canonical(contract["execution_boundary"])),
        }


def _semantic_self_hash(value: dict[str, Any], field: str) -> str:
    clone = copy.deepcopy(value)
    clone[field] = None
    return sha_bytes(canonical(clone))


def _sha256_string(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _ready_exact(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise HarnessError(f"{label} fields differ")
    return value


def _ready_json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise HarnessError("ready candidate value has an unsupported JSON type")


def _ready_string_is_secret_or_location(value: str) -> bool:
    lowered = value.lower()
    secret = re.search(
        r"authorization|bearer|api[-_]?key|token|secret|password|credential",
        value,
        re.IGNORECASE,
    )
    return (
        secret is not None
        or value.startswith(("/", "./", "../", "file:"))
        or "\\" in value
        or "/proc/" in lowered
        or "fd:" in lowered
    )


def _ready_summary_key_valid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", value) is not None:
        return not _ready_string_is_secret_or_location(value)
    return re.fullmatch(r"(?:sha256|omitted-sha256):[0-9a-f]{64}", value) is not None


def _validate_ready_key_types(
    keys: Any,
    key_types: Any,
    label: str,
    *,
    key_count: int | None = None,
) -> dict[str, str]:
    if (
        not isinstance(keys, list)
        or len(keys) > 17
        or keys != sorted(keys)
        or len(keys) != len(set(keys))
        or any(not _ready_summary_key_valid(key) for key in keys)
        or not isinstance(key_types, dict)
        or set(key_types) != set(keys)
        or any(kind not in READY_CANDIDATE_JSON_TYPES | {"omitted"} for kind in key_types.values())
    ):
        raise HarnessError(f"{label} key/type summary differs")
    omitted = [key for key in keys if key.startswith("omitted-sha256:")]
    if len(omitted) > 1 or (omitted and key_types[omitted[0]] != "omitted"):
        raise HarnessError(f"{label} omitted-key summary differs")
    if any(kind == "omitted" for key, kind in key_types.items() if key not in omitted):
        raise HarnessError(f"{label} omitted-key type differs")
    if key_count is not None:
        if key_count <= 16:
            if omitted or key_count != len(keys):
                raise HarnessError(f"{label} key count differs")
        elif len(keys) != 17 or len(omitted) != 1:
            raise HarnessError(f"{label} truncated key count differs")
    return key_types


def _validate_ready_safe_scalar(value: Any, label: str) -> dict[str, Any]:
    item = _ready_exact(
        value,
        {"present", "json_type", "value", "string_length", "canonical_sha256"},
        label,
    )
    present = item["present"]
    kind = item["json_type"]
    safe_value = item["value"]
    string_length = item["string_length"]
    digest = item["canonical_sha256"]
    if type(present) is not bool or kind not in READY_CANDIDATE_JSON_TYPES:
        raise HarnessError(f"{label} presence/type differs")
    if present != (kind != "absent") or present != (digest is not None):
        raise HarnessError(f"{label} presence/hash differs")
    if digest is not None and not _sha256_string(digest):
        raise HarnessError(f"{label} hash differs")
    if kind == "string":
        if type(string_length) is not int or not 0 <= string_length <= MAX_READY_CANDIDATE_RAW_BYTES:
            raise HarnessError(f"{label} string length differs")
        if safe_value is not None and (
            not isinstance(safe_value, str)
            or len(safe_value) != string_length
            or len(safe_value) > 128
            or re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", safe_value) is None
            or _ready_string_is_secret_or_location(safe_value)
        ):
            raise HarnessError(f"{label} safe string differs")
        if safe_value is None and (
            1 <= string_length <= 128
            or (
                string_length == 0
                and sha_bytes(canonical("")) != digest
            )
        ):
            raise HarnessError(f"{label} withheld bounded string differs")
    elif string_length is not None:
        raise HarnessError(f"{label} non-string length differs")
    if kind in {"absent", "array", "object", "string"} and kind != "string" and safe_value is not None:
        raise HarnessError(f"{label} unsafe value was retained")
    if kind == "null" and safe_value is not None:
        raise HarnessError(f"{label} null value differs")
    if kind == "boolean" and type(safe_value) is not bool:
        raise HarnessError(f"{label} boolean value differs")
    if kind == "integer" and type(safe_value) is not int:
        raise HarnessError(f"{label} integer value differs")
    if kind == "number" and (type(safe_value) is not float or not math.isfinite(safe_value)):
        raise HarnessError(f"{label} number value differs")
    if present and (kind != "string" or safe_value is not None):
        if sha_bytes(canonical(safe_value)) != digest:
            raise HarnessError(f"{label} canonical hash differs")
    return item


def _validate_ready_candidate_audit(value: Any) -> dict[str, Any]:
    audit = _ready_exact(
        value,
        {
            "schema_version", "audit_sha256", "raw", "top_level", "safe_scalars",
            "resident_session_id", "nested", "validation",
        },
        "ready candidate audit",
    )
    if (
        audit["schema_version"] != READY_CANDIDATE_AUDIT_SCHEMA
        or not _sha256_string(audit["audit_sha256"])
        or audit["audit_sha256"] != _semantic_self_hash(audit, "audit_sha256")
        or len(canonical(audit))
        > MAX_READY_CANDIDATE_MARKER_BYTES - len(READY_CANDIDATE_MARKER_PREFIX) - 1
    ):
        raise HarnessError("ready candidate audit identity/size differs")
    raw = _ready_exact(audit["raw"], {"byte_count", "raw_sha256"}, "ready candidate raw")
    if (
        type(raw["byte_count"]) is not int
        or not 0 < raw["byte_count"] <= MAX_READY_CANDIDATE_RAW_BYTES
        or not _sha256_string(raw["raw_sha256"])
    ):
        raise HarnessError("ready candidate raw summary differs")
    top = _ready_exact(
        audit["top_level"], {"key_count", "keys", "key_types"}, "ready candidate top level"
    )
    if type(top["key_count"]) is not int or not 0 <= top["key_count"] <= MAX_READY_CANDIDATE_RAW_BYTES:
        raise HarnessError("ready candidate top-level key count differs")
    top_types = _validate_ready_key_types(
        top["keys"], top["key_types"], "ready candidate top level", key_count=top["key_count"]
    )
    scalars = _ready_exact(
        audit["safe_scalars"], {"event", "schema_version", "model_loads"},
        "ready candidate safe scalars",
    )
    scalar_values = {
        name: _validate_ready_safe_scalar(scalars[name], f"ready candidate {name}")
        for name in ("event", "schema_version", "model_loads")
    }
    for name, item in scalar_values.items():
        if name in top_types:
            if item["present"] is not True or item["json_type"] != top_types[name]:
                raise HarnessError(f"ready candidate {name} top-level binding differs")
        elif item["present"] is not False:
            raise HarnessError(f"ready candidate {name} presence differs")
    session = _ready_exact(
        audit["resident_session_id"],
        {"present", "json_type", "string_length", "canonical_sha256"},
        "ready candidate session ID",
    )
    if (
        type(session["present"]) is not bool
        or session["json_type"] not in READY_CANDIDATE_JSON_TYPES
        or session["present"] != (session["json_type"] != "absent")
        or session["present"] != (session["canonical_sha256"] is not None)
        or (session["canonical_sha256"] is not None and not _sha256_string(session["canonical_sha256"]))
        or (
            session["json_type"] == "string"
            and (
                type(session["string_length"]) is not int
                or not 0 <= session["string_length"] <= MAX_READY_CANDIDATE_RAW_BYTES
            )
        )
        or (session["json_type"] != "string" and session["string_length"] is not None)
    ):
        raise HarnessError("ready candidate session ID summary differs")
    if "resident_session_id" in top_types:
        if session["present"] is not True or session["json_type"] != top_types["resident_session_id"]:
            raise HarnessError("ready candidate session ID top-level binding differs")
    elif session["present"] is not False:
        raise HarnessError("ready candidate session ID presence differs")
    nested = _ready_exact(
        audit["nested"], {"driver_identity", "served_model_binding"},
        "ready candidate nested summaries",
    )
    for name in ("driver_identity", "served_model_binding"):
        item = _ready_exact(
            nested[name], {"present", "json_type", "canonical_sha256", "keys", "key_types"},
            f"ready candidate {name}",
        )
        if (
            type(item["present"]) is not bool
            or item["json_type"] not in READY_CANDIDATE_JSON_TYPES
            or item["present"] != (item["json_type"] != "absent")
            or item["present"] != (item["canonical_sha256"] is not None)
            or (item["canonical_sha256"] is not None and not _sha256_string(item["canonical_sha256"]))
        ):
            raise HarnessError(f"ready candidate {name} summary differs")
        nested_types = _validate_ready_key_types(
            item["keys"], item["key_types"], f"ready candidate {name}"
        )
        if item["json_type"] != "object" and nested_types:
            raise HarnessError(f"ready candidate {name} non-object keys differ")
        if name in top_types and (
            item["present"] is not True or item["json_type"] != top_types[name]
        ):
            raise HarnessError(f"ready candidate {name} top-level binding differs")
        if name not in top_types:
            if item["present"] is not False:
                raise HarnessError(f"ready candidate {name} presence differs")
    validation = _ready_exact(
        audit["validation"], {"status", "reason_code", "predicates"},
        "ready candidate validation",
    )
    predicates = validation["predicates"]
    if (
        validation["status"] != "failed"
        or not isinstance(validation["reason_code"], str)
        or not isinstance(predicates, dict)
        or set(predicates) != READY_CANDIDATE_PREDICATES
        or any(type(result) is not bool for result in predicates.values())
    ):
        raise HarnessError("ready candidate validation summary differs")
    expected_fields = {
        "event", "schema_version", "model_loads", "resident_session_id",
        "driver_identity", "served_model_binding",
    }
    recomputed = {
        "field_set_exact": top["key_count"] == 6 and top["keys"] == sorted(expected_fields),
        "event_is_ready": scalar_values["event"]["json_type"] == "string" and scalar_values["event"]["value"] == "ready",
        "schema_version_exact": scalar_values["schema_version"]["json_type"] == "string" and scalar_values["schema_version"]["value"] == "ullm.aq4_p2_resident_driver.v2",
        "model_loads_is_integer": scalar_values["model_loads"]["json_type"] == "integer",
        "model_loads_is_one": scalar_values["model_loads"]["json_type"] == "integer" and scalar_values["model_loads"]["value"] == 1,
        "resident_session_id_is_string": session["json_type"] == "string",
        "resident_session_id_nonempty": session["json_type"] == "string" and bool(session["string_length"]),
    }
    if predicates != recomputed:
        raise HarnessError("ready candidate predicates differ from summaries")
    ordered_failures = (
        ("field_set_exact", "ready_candidate_field_set_differs"),
        ("event_is_ready", "ready_candidate_event_differs"),
        ("schema_version_exact", "ready_candidate_schema_differs"),
        ("model_loads_is_integer", "ready_candidate_model_loads_type_differs"),
        ("model_loads_is_one", "ready_candidate_model_loads_value_differs"),
        ("resident_session_id_is_string", "ready_candidate_session_id_type_differs"),
        ("resident_session_id_nonempty", "ready_candidate_session_id_empty"),
    )
    expected_reason = next((reason for predicate, reason in ordered_failures if not predicates[predicate]), None)
    downstream_reasons = {
        "ready_candidate_driver_identity_invalid",
        "ready_candidate_binary_sha256_differs",
        "ready_candidate_served_model_binding_invalid",
    }
    if (
        (expected_reason is not None and validation["reason_code"] != expected_reason)
        or (expected_reason is None and validation["reason_code"] not in downstream_reasons)
    ):
        raise HarnessError("ready candidate reason/predicate binding differs")
    return audit


def _ready_stream_markers(raw: bytes) -> tuple[list[bytes], bool]:
    markers: list[bytes] = []
    oversize = False
    source = io.BytesIO(raw)
    while True:
        line = source.readline(MAX_READY_CANDIDATE_MARKER_BYTES + 2)
        if not line:
            break
        starts_marker = line.startswith(READY_CANDIDATE_MARKER_PREFIX)
        if len(line) > MAX_READY_CANDIDATE_MARKER_BYTES and not line.endswith(b"\n"):
            while line and not line.endswith(b"\n"):
                line = source.readline(MAX_READY_CANDIDATE_MARKER_BYTES + 2)
            oversize = oversize or starts_marker
            continue
        if starts_marker:
            if len(line.removesuffix(b"\n")) > MAX_READY_CANDIDATE_MARKER_BYTES:
                oversize = True
            else:
                markers.append(line)
    return markers, oversize


def _validate_ready_candidate_capture(
    value: Any,
    *,
    stderr_raw: bytes,
    stderr_sha256: str,
) -> dict[str, Any]:
    envelope = _ready_exact(
        value,
        {
            "schema_version", "self_sha256", "status", "reason_code", "source_stream",
            "source_stream_sha256", "marker_count", "marker_sha256", "audit_sha256", "audit",
        },
        "ready candidate capture",
    )
    if (
        envelope["schema_version"] != READY_CANDIDATE_CAPTURE_SCHEMA
        or not _sha256_string(envelope["self_sha256"])
        or envelope["self_sha256"] != _semantic_self_hash(envelope, "self_sha256")
        or len(canonical(envelope)) > 2 * MAX_READY_CANDIDATE_MARKER_BYTES
        or envelope["source_stream"] != "rocprof.stderr"
        or envelope["source_stream_sha256"] != stderr_sha256
        or type(envelope["marker_count"]) is not int
        or envelope["marker_count"] < 0
        or (envelope["marker_sha256"] is not None and not _sha256_string(envelope["marker_sha256"]))
        or (envelope["audit_sha256"] is not None and not _sha256_string(envelope["audit_sha256"]))
    ):
        raise HarnessError("ready candidate capture identity/type/size differs")
    markers, oversize = _ready_stream_markers(stderr_raw)
    status = envelope["status"]
    reason = envelope["reason_code"]
    if status == "valid":
        if (
            reason != "ready_candidate_marker_bound"
            or oversize
            or len(markers) != 1
            or envelope["marker_count"] != 1
            or envelope["marker_sha256"] != sha_bytes(markers[0])
            or envelope["audit"] is None
        ):
            raise HarnessError("valid ready candidate marker binding differs")
        audit = _validate_ready_candidate_audit(envelope["audit"])
        payload = markers[0][len(READY_CANDIDATE_MARKER_PREFIX):]
        if not payload.endswith(b"\n") or payload.endswith(b"\r\n"):
            raise HarnessError("valid ready candidate marker termination differs")
        if (
            payload[:-1] != canonical(audit)
            or envelope["audit_sha256"] != audit["audit_sha256"]
        ):
            raise HarnessError("valid ready candidate marker/audit canonical binding differs")
    elif status == "absent":
        if (
            reason != "ready_candidate_marker_absent"
            or oversize
            or markers
            or envelope["marker_count"] != 0
            or envelope["marker_sha256"] is not None
            or envelope["audit_sha256"] is not None
            or envelope["audit"] is not None
        ):
            raise HarnessError("absent ready candidate marker binding differs")
    elif status == "invalid":
        expected_reason: str | None = None
        expected_count = len(markers)
        expected_marker_sha: str | None = None
        if oversize:
            expected_reason = "ready_candidate_marker_oversize"
            expected_count = len(markers)
        elif len(markers) != 1:
            expected_reason = "ready_candidate_marker_count_differs"
        elif not markers[0].endswith(b"\n") or markers[0].endswith(b"\r\n"):
            expected_reason = "ready_candidate_marker_termination_differs"
            expected_marker_sha = sha_bytes(markers[0])
        else:
            expected_reason = "ready_candidate_marker_payload_invalid"
            expected_marker_sha = sha_bytes(markers[0])
        if (
            reason != expected_reason
            or envelope["marker_count"] != expected_count
            or envelope["marker_sha256"] != expected_marker_sha
            or envelope["audit_sha256"] is not None
            or envelope["audit"] is not None
        ):
            raise HarnessError("invalid ready candidate marker diagnostic differs")
    else:
        raise HarnessError("ready candidate capture status differs")
    return envelope


def _read_profile_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw, _ = LAUNCHER.read_regular(path, label)
    if len(raw) > LAUNCHER.MAX_BYTES:
        raise HarnessError(f"{label} exceeds evidence bound")
    return LAUNCHER.parse_json(raw, label), raw


def _path_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return path != directory


def _verified_profile_file(path: Path, expected_sha256: str) -> tuple[int, ...] | None:
    maximum = 128 * 1024 * 1024
    try:
        if not path.is_absolute() or ".." in path.parts or path.resolve(strict=True) != path:
            return None
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum
            or stat.S_IMODE(before.st_mode) not in {0o400, 0o440, 0o444, 0o600, 0o640, 0o644, 0o660, 0o664}
        ):
            return None
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            if LAUNCHER.file_identity(opened) != LAUNCHER.file_identity(before):
                return None
            digest = hashlib.sha256()
            size = 0
            while chunk := os.read(descriptor, 1024 * 1024):
                size += len(chunk)
                if size > maximum:
                    return None
                digest.update(chunk)
        finally:
            os.close(descriptor)
        after = path.lstat()
        if (
            size != before.st_size
            or digest.hexdigest() != expected_sha256
            or LAUNCHER.file_identity(after) != LAUNCHER.file_identity(before)
        ):
            return None
        return LAUNCHER.file_identity(after)
    except OSError:
        return None


def _profile_ref(
    value: Any,
    *,
    expected_path: str | None = None,
    expected_root: Path | None = None,
    verified: dict[str, tuple[str, tuple[int, ...]]] | None = None,
) -> bool:
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "sha256"}
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
        or SHA_RE.fullmatch(value["sha256"]) is None
    ):
        return False
    path = Path(value["path"])
    if (
        not path.is_absolute()
        or ".." in path.parts
        or (expected_path is not None and value["path"] != expected_path)
        or (expected_root is not None and not _path_within(path, expected_root))
    ):
        return False
    if verified is None:
        return True
    previous = verified.get(value["path"])
    if previous is not None:
        return previous[0] == value["sha256"]
    identity = _verified_profile_file(path, value["sha256"])
    if identity is None:
        return False
    verified[value["path"]] = (value["sha256"], identity)
    return True


def _profile_refs_unchanged(verified: dict[str, tuple[str, tuple[int, ...]]]) -> bool:
    try:
        return all(
            Path(path).resolve(strict=True) == Path(path)
            and LAUNCHER.file_identity(Path(path).lstat()) == identity
            for path, (_sha256, identity) in verified.items()
        )
    except OSError:
        return False


def _profile_identity(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 7 and all(type(item) is int for item in value)


def _expected_profile_command(runner_argv: list[str], contract: dict[str, Any]) -> list[str]:
    return [
        str(PROFILE_PROFILER),
        "--log-level", "error", "--kernel-trace", "--hip-runtime-trace",
        "--memory-copy-trace", "--marker-trace", "--output-format", "csv",
        "--output-directory", contract["output"]["directory"],
        "--output-file", contract["output"]["name"], "--", *runner_argv,
    ]


def _profile_helpers_valid(value: Any) -> bool:
    expected = (
        ("selection_raw_producer", LAUNCHER.PROFILE_PRODUCER_HELPER, LAUNCHER.PROFILE_PRODUCER_HELPER_SHA),
        ("candidate_selector", LAUNCHER.PROFILE_SELECTOR_HELPER, LAUNCHER.PROFILE_SELECTOR_HELPER_SHA),
        ("profile_family_classifier", LAUNCHER.PROFILE_FAMILY_HELPER, LAUNCHER.PROFILE_FAMILY_HELPER_SHA),
    )
    return isinstance(value, list) and len(value) == len(expected) and all(
        isinstance(item, dict)
        and set(item) == {"role", "path", "identity", "sha256"}
        and item.get("role") == role
        and item.get("path") == str(path)
        and item.get("sha256") == expected_sha
        and item.get("identity") == list(LAUNCHER.file_identity(path.lstat()))
        and _verified_profile_file(path, expected_sha) is not None
        for item, (role, path, expected_sha) in zip(value, expected)
    )


def _profile_profiler_binding_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "tool", "invocation_path", "resolved_path", "executable_sha256",
        "resolved_identity", "symlink_chain",
    }:
        return False
    chain = value.get("symlink_chain")
    return (
        value.get("tool") == "rocprofv3"
        and value.get("invocation_path") == str(PROFILE_PROFILER)
        and value.get("resolved_path") == str(PROFILE_PROFILER)
        and value.get("executable_sha256") == PROFILE_PROFILER_SHA
        and value.get("resolved_identity") == list(LAUNCHER.file_identity(PROFILE_PROFILER.lstat()))
        and chain == []
    )


def _validate_profile_success_artifact(
    artifact_path: Path,
    target_binding: dict[str, Any],
    contract: dict[str, Any],
    runner_argv: list[str],
    environment: dict[str, str],
) -> dict[str, Any]:
    value, raw = _read_profile_json(artifact_path, "profile capture success artifact")
    metadata = artifact_path.lstat()
    expected_keys = {
        "schema_version", "status", "measurement_eligible", "promotion_eligible",
        "artifact_sha256", "binding", "profiler", "source_traces",
        "capture_capabilities", "marker_contract", "producer_profile_runs",
        "memory_copy_traces", "eligibility_blockers",
    }
    profiler = value.get("profiler")
    binding = value.get("binding")
    output_directory = Path(contract["output"]["directory"])
    expected_command = _expected_profile_command(runner_argv, contract)
    profiler_binding = {
        key: profiler.get(key) for key in (
            "tool", "invocation_path", "resolved_path", "executable_sha256",
            "resolved_identity", "symlink_chain",
        )
    } if isinstance(profiler, dict) else None
    source_traces = value.get("source_traces")
    marker_contract = value.get("marker_contract")
    runs = value.get("producer_profile_runs")
    memory_traces = value.get("memory_copy_traces")
    eligibility_blockers = value.get("eligibility_blockers")
    device = binding.get("device") if isinstance(binding, dict) else None
    verified_refs: dict[str, tuple[str, tuple[int, ...]]] = {}
    if (
        stat.S_IMODE(metadata.st_mode) != 0o444
        or set(value) != expected_keys
        or value.get("schema_version") != PROFILE_CAPTURE_SCHEMA
        or value.get("status") != "complete_diagnostic"
        or value.get("measurement_eligible") is not False
        or value.get("promotion_eligible") is not False
        or not isinstance(value.get("artifact_sha256"), str)
        or value["artifact_sha256"] != _semantic_self_hash(value, "artifact_sha256")
        or not isinstance(profiler, dict)
        or set(profiler) != {
            "tool", "invocation_path", "resolved_path", "executable_sha256",
            "resolved_identity", "symlink_chain", "version", "rocm_version",
            "version_output_sha256", "target_command_manifest", "target_environment",
            "capture_helpers", "command", "command_sha256", "subprocess_profile_runs",
        }
        or not _profile_profiler_binding_valid(profiler_binding)
        or not isinstance(profiler.get("version"), str)
        or not profiler["version"]
        or (profiler.get("rocm_version") is not None and (not isinstance(profiler["rocm_version"], str) or not profiler["rocm_version"]))
        or not isinstance(profiler.get("version_output_sha256"), str)
        or SHA_RE.fullmatch(profiler["version_output_sha256"]) is None
        or profiler.get("target_command_manifest") != {"path": target_binding["path"], "sha256": target_binding["sha256"]}
        or profiler.get("target_environment") != {
            "sha256": sha_bytes(canonical(environment)),
            "keys": sorted(environment),
            "exact_base_environment": True,
            "secret_material_recorded": False,
        }
        or not _profile_helpers_valid(profiler.get("capture_helpers"))
        or profiler.get("command") != expected_command
        or profiler.get("command_sha256") != sha_bytes(canonical(expected_command))
        or profiler.get("subprocess_profile_runs") != 1
        or not isinstance(binding, dict)
        or set(binding) != {
            "run_id", "resident_session_id", "case_id", "case_sha256",
            "identity_sha256", "device", "identity", "resident_summary", "resident_raw",
        }
        or binding.get("run_id") != contract["resident_evidence"]["run_id"]
        or not isinstance(binding.get("resident_session_id"), str)
        or not binding["resident_session_id"]
        or binding.get("case_id") != contract["resident_evidence"]["case_id"]
        or binding.get("case_sha256") != LAUNCHER.CASE_SHA
        or not isinstance(binding.get("identity_sha256"), str)
        or SHA_RE.fullmatch(binding["identity_sha256"]) is None
        or device != {
            "runtime_device_index": 1, "device_id": "r9700-rdna4", "backend": "hip",
            "name": "AMD Radeon Graphics", "architecture": "gfx1201",
        }
        or not _profile_ref(binding.get("identity"), expected_path=contract["resident_evidence"]["identity"], verified=verified_refs)
        or not _profile_ref(binding.get("resident_summary"), expected_path=contract["resident_evidence"]["summary"], verified=verified_refs)
        or not _profile_ref(binding.get("resident_raw"), expected_path=contract["resident_evidence"]["raw"], verified=verified_refs)
        or not isinstance(source_traces, dict)
        or set(source_traces) != {"kernel", "hip_api", "memory_copy", "marker"}
        or any(not _profile_ref(item, expected_root=output_directory, verified=verified_refs) for item in source_traces.values())
        or len({item["path"] for item in source_traces.values()}) != 4
        or len({item["sha256"] for item in source_traces.values()}) != 4
        or not _profile_ref(value.get("capture_capabilities"), expected_path=str(output_directory / "capture-capabilities.json"), verified=verified_refs)
        or marker_contract != {
            "schema_version": "ullm.aq4_p2.run.v1",
            "clock_domain": "rocprofv3_monotonic_ns",
            "range_count": 12,
            "warmup_indices": [0, 1],
            "measured_indices": list(range(2, 12)),
            "warmup_excluded": True,
        }
        or not isinstance(runs, list)
        or len(runs) != 10
        or not isinstance(memory_traces, list)
        or len(memory_traces) != 10
        or any(
            not _profile_ref(
                item,
                expected_path=str(output_directory / "measured-runs" / f"run-{index:02d}_memory_copy_trace.csv"),
                verified=verified_refs,
            )
            for index, item in enumerate(memory_traces, start=2)
        )
        or eligibility_blockers != [
            "rocprof instrumentation overhead forbids performance promotion",
            "one-case diagnostic evidence does not satisfy seven-prompt promotion coverage",
        ]
    ):
        raise HarnessError("profile capture success artifact semantic binding differs")
    assert isinstance(runs, list)
    for index, run in enumerate(runs, start=2):
        if (
            not isinstance(run, dict)
            or set(run) != {
                "schema_version", "case_id", "case_sha256", "identity_sha256",
                "resident_run_index", "measurement_eligible", "clock_domain",
                "kernel_trace_complete", "hip_api_trace_complete", "capture_capabilities",
                "kernel_trace", "hip_api_trace",
            }
            or run.get("schema_version") != "ullm.aq4_p3_rocprof_run_binding.v1"
            or run.get("case_id") != contract["resident_evidence"]["case_id"]
            or run.get("case_sha256") != LAUNCHER.CASE_SHA
            or run.get("identity_sha256") != binding["identity_sha256"]
            or run.get("resident_run_index") != index
            or run.get("measurement_eligible") is not False
            or run.get("clock_domain") != "rocprofv3_monotonic_ns"
            or run.get("kernel_trace_complete") is not True
            or run.get("hip_api_trace_complete") is not True
            or run.get("capture_capabilities") != value["capture_capabilities"]
            or not _profile_ref(
                run.get("kernel_trace"),
                expected_path=str(output_directory / "measured-runs" / f"run-{index:02d}_kernel_trace.csv"),
                verified=verified_refs,
            )
            or not _profile_ref(
                run.get("hip_api_trace"),
                expected_path=str(output_directory / "measured-runs" / f"run-{index:02d}_hip_api_trace.csv"),
                verified=verified_refs,
            )
        ):
            raise HarnessError("profile capture producer run semantic binding differs")
    if not _profile_refs_unchanged(verified_refs):
        raise HarnessError("profile capture referenced file identity changed during validation")
    return {
        "path": str(artifact_path),
        "sha256": sha_bytes(raw),
        "artifact_sha256": value["artifact_sha256"],
        "schema_version": value["schema_version"],
        "status": value["status"],
    }


def _validate_profile_failure_evidence(
    failure_path: Path,
    output_directory: Path,
    target_binding: dict[str, Any],
    expected_command: list[str],
    *,
    allow_historical_v1: bool = False,
) -> dict[str, Any]:
    value, raw = _read_profile_json(failure_path, "profile capture failure evidence")
    metadata = failure_path.lstat()
    context = value.get("context")
    streams = value.get("streams")
    reason = value.get("reason")
    common_keys = {
        "schema_version", "status", "measurement_eligible", "promotion_eligible",
        "failure_sha256", "reason", "rocprof_child_new_session",
        "outer_harness_signalled", "process_group_cleanup_complete",
        "children_state_known", "children_remaining", "command_sha256",
        "effective_command_sha256", "context", "streams",
    }
    v2_keys = common_keys | {"ready_candidate_audit"}
    legacy_v1_keys = common_keys - {
        "children_state_known", "children_remaining", "effective_command_sha256",
    }
    schema = value.get("schema_version")
    historical_legacy_shape = False
    if schema == PROFILE_CAPTURE_FAILURE_SCHEMA:
        expected_keys = v2_keys
    elif allow_historical_v1 and schema == HISTORICAL_PROFILE_CAPTURE_FAILURE_SCHEMA:
        if set(value) == common_keys:
            expected_keys = common_keys
        elif set(value) == legacy_v1_keys:
            expected_keys = legacy_v1_keys
            historical_legacy_shape = True
        else:
            expected_keys = set()
    else:
        expected_keys = set()
    children_state_known = value.get("children_state_known")
    children_remaining = value.get("children_remaining")
    cleanup_complete = value.get("process_group_cleanup_complete")
    logical_command_sha = sha_bytes(canonical(expected_command))
    if (
        stat.S_IMODE(metadata.st_mode) != 0o444
        or set(value) != expected_keys
        or value.get("status") != "failed"
        or value.get("measurement_eligible") is not False
        or value.get("promotion_eligible") is not False
        or not isinstance(value.get("failure_sha256"), str)
        or value["failure_sha256"] != _semantic_self_hash(value, "failure_sha256")
        or not isinstance(reason, str)
        or not reason
        or len(reason.encode()) > 65536
        or value.get("rocprof_child_new_session") is not True
        or value.get("outer_harness_signalled") is not False
        or type(cleanup_complete) is not bool
        or not isinstance(value.get("command_sha256"), str)
        or value["command_sha256"] != logical_command_sha
        or not isinstance(context, dict)
        or set(context) != {"profiler", "target_command_manifest"}
        or not _profile_profiler_binding_valid(context.get("profiler"))
        or context.get("target_command_manifest")
        != {"path": target_binding["path"], "sha256": target_binding["sha256"]}
        or not isinstance(streams, dict)
        or set(streams) != {"rocprof.stdout", "rocprof.stderr"}
    ):
        raise HarnessError("profile capture failure evidence semantic binding differs")
    if not historical_legacy_shape and (
        type(children_state_known) is not bool
        or not isinstance(children_remaining, list)
        or any(type(pid) is not int or pid <= 0 for pid in children_remaining)
        or children_remaining != sorted(set(children_remaining))
        or cleanup_complete is not (children_state_known and children_remaining == [])
        or (not children_state_known and children_remaining != [])
        or not isinstance(value.get("effective_command_sha256"), str)
        or SHA_RE.fullmatch(value["effective_command_sha256"]) is None
        or value["effective_command_sha256"] == logical_command_sha
    ):
        raise HarnessError("profile capture failure lifecycle binding differs")
    stderr_raw: bytes | None = None
    for name, reference in streams.items():
        stream_path = output_directory / name
        stream_raw, _ = LAUNCHER.read_regular(stream_path, f"profile capture failure {name}")
        if (
            not isinstance(reference, dict)
            or set(reference) != {"bytes", "sha256"}
            or reference.get("bytes") != len(stream_raw)
            or reference.get("sha256") != sha_bytes(stream_raw)
        ):
            raise HarnessError("profile capture failure stream binding differs")
        if name == "rocprof.stderr":
            stderr_raw = stream_raw
    ready_candidate_audit = None
    if schema == PROFILE_CAPTURE_FAILURE_SCHEMA:
        if stderr_raw is None:
            raise HarnessError("profile capture failure READY source stream is unavailable")
        ready_candidate_audit = _validate_ready_candidate_capture(
            value["ready_candidate_audit"],
            stderr_raw=stderr_raw,
            stderr_sha256=streams["rocprof.stderr"]["sha256"],
        )
    return {
        "path": str(failure_path),
        "sha256": sha_bytes(raw),
        "failure_sha256": value["failure_sha256"],
        "schema_version": value["schema_version"],
        "status": value["status"],
        "reason": reason,
        "timed_out": "timed out" in reason.lower(),
        "process_group_cleanup_complete": cleanup_complete,
        "children_state_known": UNKNOWN_LIFECYCLE_STATE if historical_legacy_shape else children_state_known,
        "children_remaining": UNKNOWN_LIFECYCLE_STATE if historical_legacy_shape else children_remaining,
        "ready_candidate_audit": ready_candidate_audit,
        "historical_readback": schema == HISTORICAL_PROFILE_CAPTURE_FAILURE_SCHEMA,
        "streams": {
            name: {"path": str(output_directory / name), **reference}
            for name, reference in sorted(streams.items())
        },
    }


def _read_historical_profile_failure_evidence(
    failure_path: Path,
    output_directory: Path,
    target_binding: dict[str, Any],
    expected_command: list[str],
) -> dict[str, Any]:
    return _validate_profile_failure_evidence(
        failure_path,
        output_directory,
        target_binding,
        expected_command,
        allow_historical_v1=True,
    )


def _load_profile_capture_module(trusted_capture_raw: bytes) -> types.ModuleType:
    module_name = "aq4_p3_profile_capture_executor"
    capture_module = types.ModuleType(module_name)
    capture_module.__file__ = str(PROFILE_CAPTURE_TOOL)
    capture_module.__package__ = ""
    sys.modules[module_name] = capture_module
    try:
        code = compile(trusted_capture_raw, str(PROFILE_CAPTURE_TOOL), "exec", dont_inherit=True)
        exec(code, capture_module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return capture_module


def run_profile_capture(
    request: dict[str, Any],
    *,
    trusted_capture_raw: bytes | None = None,
) -> dict[str, Any]:
    required = {"contract", "runner_argv", "environment", "mark_runner_started", "target_binding"}
    if not isinstance(request, dict) or set(request) != required:
        raise HarnessError("profile runner executor request differs")
    contract = request["contract"]
    runner_argv = request["runner_argv"]
    environment = request["environment"]
    target_binding = request["target_binding"]
    mark_runner_started = request["mark_runner_started"]
    output = contract.get("output") if isinstance(contract, dict) else None
    if (
        not isinstance(contract, dict)
        or not isinstance(runner_argv, list)
        or not runner_argv
        or any(not isinstance(item, str) or not item for item in runner_argv)
        or environment != LAUNCHER.EXECUTE_ENV
        or not callable(mark_runner_started)
        or not isinstance(target_binding, dict)
        or set(target_binding) != {"path", "sha256", "manifest_sha256", "identity"}
        or not isinstance(target_binding.get("path"), str)
        or not Path(target_binding["path"]).is_absolute()
        or ".." in Path(target_binding["path"]).parts
        or not isinstance(target_binding.get("sha256"), str)
        or SHA_RE.fullmatch(target_binding["sha256"]) is None
        or not isinstance(target_binding.get("manifest_sha256"), str)
        or SHA_RE.fullmatch(target_binding["manifest_sha256"]) is None
        or not _profile_identity(target_binding.get("identity"))
        or not isinstance(output, dict)
        or set(output) != {"directory", "name", "artifact", "must_not_exist_before_capture"}
        or not isinstance(output.get("directory"), str)
        or not Path(output["directory"]).is_absolute()
        or ".." in Path(output["directory"]).parts
        or not isinstance(output.get("name"), str)
        or not output["name"]
        or output.get("artifact") != str(Path(output["directory"]) / "capture-artifact.json")
        or output.get("must_not_exist_before_capture") is not True
    ):
        raise HarnessError("profile runner executor binding differs")
    command = profile_capture_command(target_binding, contract)
    if trusted_capture_raw is None:
        trusted_capture_raw, _ = LAUNCHER.read_regular(PROFILE_CAPTURE_TOOL, "profile capture executor")
    if sha_bytes(trusted_capture_raw) != PROFILE_CAPTURE_SHA:
        raise HarnessError("profile capture executor trusted bytes differ")
    capture_module = _load_profile_capture_module(trusted_capture_raw)
    module_name = capture_module.__name__
    stdout_text = io.StringIO()
    stderr_text = io.StringIO()
    rocprof_started = False
    runner_completed = False

    def observed_rocprof_start() -> None:
        nonlocal rocprof_started
        if rocprof_started:
            raise HarnessError("profile capture reported rocprof start more than once")
        mark_runner_started()
        rocprof_started = True

    def observed_runner_completed() -> None:
        nonlocal runner_completed
        if not rocprof_started or runner_completed:
            raise HarnessError("profile capture runner completion order differs")
        runner_completed = True

    capture_exception: Exception | None = None
    try:
        with contextlib.redirect_stdout(stdout_text), contextlib.redirect_stderr(stderr_text):
            return_code = capture_module.main(
                command[2:],
                on_rocprof_started=observed_rocprof_start,
                on_runner_completed=observed_runner_completed,
            )
    except Exception as error:
        capture_exception = error
        return_code = 1
        stderr_text.write(f"profile capture executor raised {type(error).__name__}\n")
    finally:
        sys.modules.pop(module_name, None)
    capture_stdout = stdout_text.getvalue().encode()
    capture_stderr = stderr_text.getvalue().encode()
    if len(capture_stdout) > LAUNCHER.MAX_BYTES or len(capture_stderr) > LAUNCHER.MAX_BYTES:
        raise HarnessError("profile capture output exceeds evidence bound")

    def runner_stream(name: str, fallback: bytes) -> bytes:
        path = Path(contract["output"]["directory"]) / name
        if not path.exists():
            return fallback
        raw, _ = LAUNCHER.read_regular(path, f"profiled runner {name}")
        if len(raw) > LAUNCHER.MAX_BYTES:
            raise HarnessError("profiled runner output exceeds evidence bound")
        return raw

    runner_stdout = runner_stream("rocprof.stdout", capture_stdout)
    runner_stderr = runner_stream("rocprof.stderr", capture_stderr)
    output_directory = Path(contract["output"]["directory"])
    artifact_path = Path(contract["output"]["artifact"])
    failure_path = output_directory / PROFILE_CAPTURE_FAILURE_NAME
    if artifact_path.parent != output_directory or artifact_path.name != "capture-artifact.json":
        raise HarnessError("profile capture artifact output binding differs")

    artifact_evidence: dict[str, Any] | None = None
    failure_evidence: dict[str, Any] | None = None
    validation_error: str | None = None
    cleanup_passed = False
    children_remaining: list[int] = []
    timed_out = False
    complete = False
    try:
        if type(return_code) is not int:
            raise HarnessError("profile capture executor return code differs")
        if return_code == 0:
            if failure_path.exists() or failure_path.is_symlink():
                raise HarnessError("profile capture success conflicts with failure evidence")
            artifact_evidence = _validate_profile_success_artifact(
                artifact_path, target_binding, contract, runner_argv, environment
            )
            if not rocprof_started or not runner_completed:
                raise HarnessError("profile capture artifact exists without lifecycle callbacks")
            if runner_stderr:
                raise HarnessError("profiled runner stderr is not empty")
            complete = True
            cleanup_passed = True
            children_remaining = []
        else:
            if artifact_path.exists() or artifact_path.is_symlink():
                raise HarnessError("profile capture failure conflicts with success artifact")
            failure_evidence = _validate_profile_failure_evidence(
                failure_path,
                output_directory,
                target_binding,
                _expected_profile_command(runner_argv, contract),
            )
            cleanup_passed = failure_evidence["process_group_cleanup_complete"]
            children_remaining = failure_evidence["children_remaining"]
            timed_out = failure_evidence["timed_out"]
    except (HarnessError, LAUNCHER.LauncherError, OSError, ValueError) as error:
        validation_error = f"{type(error).__name__}: {error}"
        return_code = 1
        cleanup_passed = False
        children_remaining = []
        timed_out = False
    if complete or runner_completed:
        children_state_known = True
        cleanup_passed = True
        children_remaining = []
    elif failure_evidence is not None:
        children_state_known = failure_evidence["children_state_known"]
        children_remaining = failure_evidence["children_remaining"]
        cleanup_passed = failure_evidence["process_group_cleanup_complete"]
    elif not rocprof_started:
        children_state_known = True
        cleanup_passed = True
        children_remaining = []
    else:
        children_state_known = False
        cleanup_passed = False
        children_remaining = []
    completed = subprocess.CompletedProcess(runner_argv, 0 if complete else 1, runner_stdout, runner_stderr)
    runner_start_known = not rocprof_started or runner_completed
    runner_started = runner_completed
    runner_finished = runner_completed
    profile_diagnostics = {
        "schema_version": "ullm.aq4_p3_profile_executor_diagnostics.v1",
        "runner_finished": runner_finished,
        "capture_artifact": None if artifact_evidence is None else {
            "path": artifact_evidence["path"],
            "sha256": artifact_evidence["sha256"],
            "mode": stat.S_IMODE(Path(artifact_evidence["path"]).lstat().st_mode),
        },
        "failure_evidence": None if failure_evidence is None else {
            "path": failure_evidence["path"],
            "sha256": failure_evidence["sha256"],
            "mode": stat.S_IMODE(Path(failure_evidence["path"]).lstat().st_mode),
            "reason": failure_evidence["reason"],
        },
        "validation_error": validation_error,
        "executor_exception": type(capture_exception).__name__ if capture_exception is not None else None,
    }
    return {
        "completed": completed,
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
            "rocprof_invocations": int(rocprof_started),
            "target_manifest_sha256": target_binding.get("sha256"),
            "target_manifest_semantic_sha256": target_binding.get("manifest_sha256"),
            "target_argv_sha256": sha_bytes(canonical(runner_argv)),
            "environment_sha256": sha_bytes(canonical(environment)),
            "capture_stdout_sha256": sha_bytes(capture_stdout),
            "capture_stderr_sha256": sha_bytes(capture_stderr),
            "rocprof_started": rocprof_started,
            "runner_start_known": runner_start_known,
            "runner_started": runner_started,
            "runner_completed": runner_completed,
            "timed_out": timed_out,
            "cleanup_passed": cleanup_passed,
            "children_state_known": children_state_known,
            "children_remaining": children_remaining,
        },
        "profile_diagnostics": profile_diagnostics,
    }


def default_dependencies() -> Dependencies:
    trust = ProfileTrustGuard()
    container_health = ContainerHealthGuard()
    stopped_observation = StoppedGateObserver()
    return Dependencies(
        run=subprocess.run,
        http_probe=default_http_probe,
        container_health=container_health,
        stopped_observation=stopped_observation,
        lock_busy=default_lock_busy,
        owner_probe=default_owner_probe,
        package_hash=tree_hash,
        launcher_execute=_default_launcher_execute,
        profile_capture=lambda request: run_profile_capture(request, trusted_capture_raw=trust.capture_tool_raw),
        profile_trust=trust,
        sleep=time.sleep,
        monotonic_ns=time.monotonic_ns,
        package_metadata=package_tree_snapshot,
        lock_substrate_prepare=prepare_lock_substrate,
        lock_substrate_cleanup=cleanup_lock_substrate,
    )


def profile_capture_command(target_binding: dict[str, Any], contract: dict[str, Any] | None = None) -> list[str]:
    target_path = target_binding.get("path")
    target_sha256 = target_binding.get("sha256")
    if not isinstance(target_path, str) or not Path(target_path).is_absolute() or not isinstance(target_sha256, str) or SHA_RE.fullmatch(target_sha256) is None:
        raise HarnessError("profile runner target binding differs")
    output = contract.get("output") if isinstance(contract, dict) else None
    output_directory = Path(output["directory"]) if isinstance(output, dict) and isinstance(output.get("directory"), str) else PROFILE_OUTPUT_DIRECTORY
    output_name = output["name"] if isinstance(output, dict) and isinstance(output.get("name"), str) else PROFILE_OUTPUT_NAME
    artifact = Path(output["artifact"]) if isinstance(output, dict) and isinstance(output.get("artifact"), str) else PROFILE_ARTIFACT
    return [
        str(LAUNCHER.PYTHON),
        str(PROFILE_CAPTURE_TOOL),
        "capture",
        "--profiler-path",
        str(PROFILE_PROFILER),
        "--profiler-sha256",
        PROFILE_PROFILER_SHA,
        "--target-command-manifest",
        target_path,
        "--target-command-manifest-sha256",
        target_sha256,
        "--profile-output-directory",
        str(output_directory),
        "--profile-output-name",
        output_name,
        "--identity",
        str(LAUNCHER.INPUT_ROOT / "identity.json"),
        "--resident-summary",
        str(LAUNCHER.PROFILE_RUN_OUTPUT / "resident-batch.summary.json"),
        "--resident-raw",
        str(LAUNCHER.PROFILE_RUN_OUTPUT / f"{LAUNCHER.CASE_ID}.raw.json"),
        "--artifact",
        str(artifact),
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
        "aggregate": {"distinct_test_file_count": 12, "collected": 481, "passed": 481, "failed": 0, "deselected": 0},
        "suites": [
            {
                "name": "resident_trust_chain",
                "command": ["python3", "-m", "pytest", "-q", "tests/test_prepare_aq4_p2_resident_smoke_bundle.py", "tests/test_run_aq4_p2_resident_batch.py", "tests/test_run_aq4_p2_resident_live_preflight.py", "tests/test_launch_aq4_p2_resident_smoke.py", "tests/test_launch_aq4_p2_resident_smoke_execute.py", "tests/test_aq4_p2_resident_smoke_maintenance.py"],
                "collected": 346, "passed": 346, "failed": 0, "deselected": 0,
                "files": [
                    {"path": "tests/test_prepare_aq4_p2_resident_smoke_bundle.py", "source_commit": "bad728000405a711dec4faf10d4a60393bf9d7e8", "git_blob": "b09941f39318c0e4a6e1324445c35c94231ce1ec", "collected": 63, "passed": 63},
                    {"path": "tests/test_run_aq4_p2_resident_batch.py", "source_commit": "1e65fd5c99845c7a64e707df5bf140ca6d62ff82", "git_blob": "3acef1634f3dee47d860b639c6b9f66d5fd0662d", "collected": 44, "passed": 44},
                    {"path": "tests/test_run_aq4_p2_resident_live_preflight.py", "source_commit": "e993016f4a62b9970423223db8702f77ee834b12", "git_blob": "7f70bb62b8c46ff68e8597663b6054568b676d9f", "collected": 27, "passed": 27},
                    {"path": "tests/test_launch_aq4_p2_resident_smoke.py", "source_commit": "288b165c707413aac01753b8254ea98fe843308f", "git_blob": "1eb197e2cb357c8af264275de33090382619ef21", "collected": 7, "passed": 7},
                    {"path": "tests/test_launch_aq4_p2_resident_smoke_execute.py", "source_commit": "a0a61219be28be1e3765c076d4a23513f6bd6221", "git_blob": "7e79ac50b1f69e49128c90652cf3623db2ecfd78", "collected": 69, "passed": 69},
                    {"path": "tests/test_aq4_p2_resident_smoke_maintenance.py", "source_commit": "288b165c707413aac01753b8254ea98fe843308f", "git_blob": "c3ae74131fb464520c6f00a0c69ac4aec5d21690", "collected": 136, "passed": 136},
                ],
            },
            {
                "name": "resident_driver_unit",
                "command": ["cargo", "test", "-p", "ullm-engine", "--bin", "ullm-aq4-p2-resident-driver", "--no-default-features"],
                "collected": 22, "passed": 22, "failed": 0, "deselected": 0,
                "files": [{"path": "crates/ullm-engine/src/bin/ullm-aq4-p2-resident-driver.rs", "source_commit": "81ceebb13518f590b5dbf439cd00b35e508c1c3f", "git_blob": "7e37119cc8b66dc0e0f7abcf49b896fcdad8315f", "collected": 22, "passed": 22}],
            },
            {
                "name": "resident_roctx_ranges",
                "command": ["python3", "-m", "pytest", "-q", "tests/test_aq4_p2_resident_roctx_ranges.py"],
                "collected": 5, "passed": 5, "failed": 0, "deselected": 0,
                "files": [{"path": "tests/test_aq4_p2_resident_roctx_ranges.py", "source_commit": "62eadada3082b0c72eb1b467177ffe0c9445f26d", "git_blob": "a6ee8886ffdf58ea668b1b4c49452fa47637f7d9", "collected": 5, "passed": 5}],
            },
            {
                "name": "diagnostic_capture",
                "command": ["env", "ULLM_TEST_AQ4_P2_RESIDENT_DRIVER=/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v1/resident-driver", "python3", "-m", "pytest", "-q", "tests/test_capture_aq4_p3_diagnostic_profile.py"],
                "collected": 29, "passed": 29, "failed": 0, "deselected": 0,
                "files": [{"path": "tests/test_capture_aq4_p3_diagnostic_profile.py", "source_commit": "1f5b12803759e6596021dfd8c5e1455f2635f586", "git_blob": "4dfa4e419098e3bf2dfc658eb2f93e1be6fa8008", "collected": 29, "passed": 29}],
            },
            {
                "name": "selection_raw_producer",
                "command": ["python3", "-m", "pytest", "-q", "tests/test_build_aq4_p3_selection_raw.py"],
                "collected": 26, "passed": 26, "failed": 0, "deselected": 0,
                "files": [{"path": "tests/test_build_aq4_p3_selection_raw.py", "source_commit": "c743007f97486e7c7e070f4258ce4e98f0665aad", "git_blob": "8167859108c68fa27c67fe21c3d772e4899e384a", "collected": 26, "passed": 26}],
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
    "strict_negative_contract_count": 43,
    "coverage": ["safety-success-start-failure-partial", "validator-runner-finalize-toctou", "identity-and-hash-bindings", "source-family-and-runtime-gfx-vocabulary-separation", "runtime-device-five-field-exact-binding", "worker-fixture-driven-single-or-two-link-exact-topology-pre-open-post-and-rehash", "bounded-driver-stdout-and-streamed-stderr-failure-evidence", "driver-process-group-descendant-cleanup-and-secret-redaction", "strict-amd-process-active-owner-and-zero-sentinel-schema", "secret-free-amd-process-rejection-shape-and-raw-sha", "bounded-kfd-enoent-rescan-and-fatal-source-diagnostics", "trusted-runtime-lock-substrate-lifecycle-and-same-inode-runner-binding", "absolute-deadline-stable2-stopped-gate-poll-and-foreign-owner-rejection", "remaining-capped-probe-timeouts-and-between-probe-sudo-keepalive", "immutable-streamed-stop-poll-evidence", "container-namespace-health-and-authenticated-model-binding", "secret-free-stdin-header-transport", "base-and-profile-dry-run-process-count-zero", "rocprof-pinned-fd-and-target-manifest", "roctx-run-session-case-and-library-binding"],
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
            if (
                not isinstance(path, str)
                or (not path.startswith("tests/test_") and path != "crates/ullm-engine/src/bin/ullm-aq4-p2-resident-driver.rs")
                or Path(path).is_absolute()
                or ".." in Path(path).parts
                or path in observed_paths
            ):
                raise HarnessError("QA exact test file coverage differs")
            if not isinstance(item.get("source_commit"), str) or re.fullmatch(r"[0-9a-f]{40}", item["source_commit"]) is None or not isinstance(item.get("git_blob"), str) or re.fullmatch(r"[0-9a-f]{40}", item["git_blob"]) is None:
                raise HarnessError("QA exact test file Git identity differs")
            committed = subprocess.run(
                ["git", "rev-parse", f'{item["source_commit"]}:{path}'],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if committed.returncode != 0 or committed.stderr:
                raise HarnessError("QA exact test source commit/path is unavailable")
            if committed.stdout.decode("ascii").strip() != item["git_blob"]:
                raise HarnessError("QA exact test source Git blob differs")
            if type(item.get("collected")) is not int or type(item.get("passed")) is not int or item["collected"] <= 0 or item["passed"] != item["collected"]:
                raise HarnessError("QA exact test file counts differ")
            observed_paths.add(path)
            suite_paths.append(path)
            suite_collected += item["collected"]
            suite_passed += item["passed"]
        if suite.get("name") == "resident_driver_unit" and suite_paths == ["crates/ullm-engine/src/bin/ullm-aq4-p2-resident-driver.rs"]:
            expected_command = ["cargo", "test", "-p", "ullm-engine", "--bin", "ullm-aq4-p2-resident-driver", "--no-default-features"]
        elif suite.get("name") == "diagnostic_capture" and suite_paths == ["tests/test_capture_aq4_p3_diagnostic_profile.py"]:
            expected_command = ["env", f"ULLM_TEST_AQ4_P2_RESIDENT_DRIVER={LAUNCHER.RESIDENT_DRIVER}", "python3", "-m", "pytest", "-q", *suite_paths]
        else:
            expected_command = ["python3", "-m", "pytest", "-q", *suite_paths]
        if suite.get("command") != expected_command or suite.get("collected") != suite_collected or suite.get("passed") != suite_passed or suite.get("failed") != 0 or suite.get("deselected") != 0:
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
            "package_integrity": {
                "full_content_hash_stage": "pre_stop_once",
                "full_hash_count": 1,
                "tree_metadata_identity_fields": ["relative_path", "device", "inode", "mode", "nlink", "size", "mtime_ns", "ctime_ns"],
                "includes": ["regular_file", "directory", "symlink", "special_file"],
                "post_readiness": "full_tree_metadata_reenumeration_exact_identity",
            },
            "restore_poll": {
                "timeout_seconds": RESTORE_TIMEOUT_SECONDS,
                "poll_interval_seconds": RESTORE_POLL_INTERVAL_SECONDS,
                "maximum_probe_timeout_seconds": RESTORE_PROBE_TIMEOUT_SECONDS,
                "deadline_semantics": "fixed_absolute_monotonic_ns_checked_before_and_after_each_probe_and_final_metadata_recheck",
                "dynamic_probe_only": True,
            },
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
            "production": {"manifest_sha256": LAUNCHER.SERVED_SHA, "worker_sha256": WORKER_SHA, "package_manifest_sha256": PACKAGE_MANIFEST_SHA, "package_content_sha256": PACKAGE_CONTENT_SHA, "expected_package_integrity_identity_sha256": PACKAGE_INTEGRITY_IDENTITY_SHA},
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
            "execution_boundary": {
                "order": ["maintenance", "launcher", "validator", "gates", "capture", "rocprof", "runner"],
                "runner_profiled": True,
                "validator_profiled": False,
                "gates_profiled": False,
            },
            "target_runner": {
                "generated_by": "launcher_after_live_preflight",
                "file_name": LAUNCHER.PROFILE_RUNNER_TARGET_MANIFEST_NAME,
                "fresh_per_execution": True,
                "environment": "exact_execute_environment",
                "maximum_invocations": 1,
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
    if {item.name for item in root.iterdir()} != expected_names:
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
    sum_inputs = [("harness-trust.json", trust_raw), ("qa-attestation.json", attestation_raw), ("ready-binding.json", ready_raw)]
    expected_sums = "".join(f"{sha_bytes(raw)}  {name}\n" for name, raw in sorted(sum_inputs)).encode("ascii")
    sums_raw, _ = LAUNCHER.read_regular(root / "SHA256SUMS", "ready sums")
    if sums_raw != expected_sums:
        raise HarnessError("ready artifact SHA256SUMS differs")
    return value


def _finalize(output: Path, evidence: dict[str, Any]) -> None:
    LAUNCHER.finalize_output(output, evidence)


def poll_restore_readiness(
    dependencies: Dependencies,
    previous: dict[str, Any],
    package_before: PackageTreeSnapshot,
    restore: dict[str, Any],
) -> dict[str, Any]:
    """Poll dynamic readiness against one fixed deadline, then recheck tree identity."""

    started_ns = restore["started_monotonic_ns"]
    deadline_ns = restore["deadline_monotonic_ns"]
    attempts: list[dict[str, Any]] = []
    post: dict[str, Any] | None = None
    last_failure: dict[str, Any] | None = None
    restore.update(
        {
            "timeout_seconds": RESTORE_TIMEOUT_SECONDS,
            "probe_timeout_seconds": RESTORE_PROBE_TIMEOUT_SECONDS,
            "poll_interval_seconds": RESTORE_POLL_INTERVAL_SECONDS,
            "deadline_semantics": "fixed_absolute_monotonic_ns_checked_before_and_after_each_probe",
            "polls": attempts,
            "poll_count": 0,
            "last_failure": None,
            "final_metadata_recheck": None,
        }
    )
    while True:
        before_ns = dependencies.monotonic_ns()
        remaining_ns = deadline_ns - before_ns
        if remaining_ns <= 0:
            break
        probe_timeout = min(RESTORE_PROBE_TIMEOUT_SECONDS, remaining_ns / 1_000_000_000)
        attempt: dict[str, Any] = {
            "attempt": len(attempts),
            "started_monotonic_ns": before_ns,
            "remaining_before_ns": remaining_ns,
            "probe_timeout_seconds": probe_timeout,
            "passed": False,
            "failure": None,
        }
        try:
            candidate = capture_running(
                dependencies,
                previous,
                probe_timeout_seconds=probe_timeout,
            )
            after_ns = dependencies.monotonic_ns()
            if after_ns > deadline_ns:
                raise HarnessError("service recovery probe crossed absolute deadline")
            attempt["passed"] = True
            post = candidate
        except (HarnessError, OSError, ValueError, subprocess.SubprocessError) as error:
            after_ns = dependencies.monotonic_ns()
            last_failure = {"type": type(error).__name__, "reason": str(error)}
            attempt["failure"] = last_failure
        attempt["finished_monotonic_ns"] = after_ns
        attempt["duration_ns"] = max(0, after_ns - before_ns)
        attempts.append(attempt)
        if post is not None:
            break
        remaining_ns = deadline_ns - after_ns
        if remaining_ns <= 0:
            break
        dependencies.sleep(min(RESTORE_POLL_INTERVAL_SECONDS, remaining_ns / 1_000_000_000))

    restore["poll_count"] = len(attempts)
    restore["last_failure"] = last_failure
    restore["completed_monotonic_ns"] = dependencies.monotonic_ns()
    restore["duration_ns"] = max(0, restore["completed_monotonic_ns"] - started_ns)
    if post is None:
        reason = last_failure["reason"] if last_failure is not None else "absolute deadline expired"
        raise HarnessError(f"service recovery validation failed: {reason}")

    if dependencies.monotonic_ns() > deadline_ns:
        raise HarnessError("service recovery readiness crossed absolute deadline")
    final_tree = dependencies.package_metadata(PACKAGE_ROOT)
    metadata_finished_ns = dependencies.monotonic_ns()
    difference = _package_tree_difference(package_before, final_tree)
    restore["final_metadata_recheck"] = {
        **final_tree.evidence("post-readiness-final-tree-metadata"),
        "expected_identity_sha256": package_before.identity_sha256,
        "passed": difference is None,
        "difference": difference,
        "finished_monotonic_ns": metadata_finished_ns,
        "within_absolute_deadline": metadata_finished_ns <= deadline_ns,
    }
    restore["completed_monotonic_ns"] = metadata_finished_ns
    restore["duration_ns"] = max(0, metadata_finished_ns - started_ns)
    if metadata_finished_ns > deadline_ns:
        raise HarnessError("production package final metadata recheck crossed absolute deadline")
    if difference is not None:
        raise HarnessError("production package tree metadata identity changed across maintenance")
    return post


def dry_run_ready(value: dict[str, Any], output: Path, ready_path: Path = READY_PATH) -> tuple[int, dict[str, Any]]:
    LAUNCHER.reject_symlink_components(output, "ready dry-run output", allow_missing_leaf=True)
    if output.exists() or output.is_symlink():
        raise HarnessError("ready dry-run output already exists")
    output.mkdir(mode=0o700)
    evidence = {"schema_version": "ullm.aq4_p2_resident_maintenance.v1", "status": "passed", "mode": "dry-run", "execution_mode": value["execution_mode"], "actual_eligible": value["actual_eligible"], "promotion_eligible": False, "run_id": value["authorization"]["run_id"], "process_counts": {"sudo": 0, "sudo_keepalive": 0, "systemctl_stop": 0, "launcher": 0, "systemctl_start": 0, "rocprof": 0, "capture_tool": 0, "docker": 0, "docker_exec": 0, "container_curl": 0, "container_curl_total": 0, "container_curl_version": 0, "container_curl_endpoint": 0, "stopped_gate_polls": 0, "stopped_gate_probe_commands": 0}, "service_touched": False, "gpu_command_executed": False, "model_load_executed": False, "ready_binding_sha256": LAUNCHER.sha_file(ready_path, "ready binding")[0]}
    if value["execution_mode"] == "profile_diagnostic":
        evidence["profile_diagnostic"] = {"execution_boundary": value["profile_diagnostic"]["execution_boundary"], "target_runner": value["profile_diagnostic"]["target_runner"], "capture_executed": False, "measurement_eligible": False, "promotion_eligible": False}
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
    expected_package_integrity = value.get("trust", {}).get("production", {}).get("expected_package_integrity_identity_sha256")
    if not isinstance(expected_package_integrity, str) or SHA_RE.fullmatch(expected_package_integrity) is None:
        raise HarnessError("ready package trusted integrity identity differs")
    for path, label in ((output, "maintenance evidence"), (Path(value["launcher_binding"]["runner_output"]), "runner output"), (Path(value["launcher_binding"]["evidence_output"]), "launcher evidence")):
        LAUNCHER.reject_symlink_components(path, label, allow_missing_leaf=True)
        if path.exists() or path.is_symlink():
            raise HarnessError(f"{label} already exists")
    if profile_diagnostic:
        profile_output_directory = Path(value["profile_diagnostic"]["output"]["directory"])
        LAUNCHER.ensure_directory_chain(
            profile_output_directory.parent,
            "profile capture output parent",
        )
        LAUNCHER.reject_symlink_components(
            profile_output_directory,
            "profile capture output",
            allow_missing_leaf=True,
        )
        if profile_output_directory.exists() or profile_output_directory.is_symlink():
            raise HarnessError("profile capture output already exists")
        trust_records = [dependencies.profile_trust(value["profile_diagnostic"], "before-start")]
    else:
        trust_records = []
    output.mkdir(mode=0o700)
    evidence: dict[str, Any] = {"schema_version": "ullm.aq4_p2_resident_maintenance.v1", "status": "failed", "mode": "execute", "execution_mode": value["execution_mode"], "run_id": run_id, "promotion_eligible": False, "profile_trust": trust_records, "capture": None, "sequence": [], "commands": [], "package_integrity": {}, "pre_stop": None, "stopped_gates": None, "stopped_gate_poll": None, "lock_substrate": None, "lock_substrate_cleanup": None, "launcher": None, "restore": None, "failure": None, "process_counts": {"sudo": 0, "sudo_keepalive": 0, "systemctl_stop": 0, "launcher": 0, "systemctl_start": 0, "capture_tool": 0, "rocprof": 0, "docker": 0, "docker_exec": 0, "container_curl": 0, "container_curl_total": 0, "container_curl_version": 0, "container_curl_endpoint": 0, "stopped_gate_polls": 0, "stopped_gate_probe_commands": 0}, "safety": {"service_touched": False, "service_stopped": False, "gpu_command_executed": False, "model_load_executed": False}, "secret_material_recorded": False}
    stop_attempted = False; capture_attempted = False; profile_lifecycle_validated = False; pre: dict[str, Any] | None = None; package_before: PackageTreeSnapshot | None = None; substrate: LockSubstrate | None = None; runner_finished = False; runner_not_started = False; runner_evidence: dict[str, Any] | None = None; code = 1; stage = "sudo-prevalidate"
    try:
        record = _sudo_valid(dependencies.run, "sudo-prevalidate"); evidence["commands"].append(record); evidence["process_counts"]["sudo"] += 1; evidence["sequence"].append("sudo-prevalidate")
        stage = "pre-stop-package-integrity"; package_before = capture_package_integrity(dependencies, evidence["package_integrity"], expected_package_integrity)
        stage = "pre-stop-snapshot"; pre = capture_running(dependencies); pre["package_integrity"] = copy.deepcopy(evidence["package_integrity"]); evidence["pre_stop"] = pre; evidence["sequence"].append("pre-stop-snapshot")
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
        stage = "launcher"; evidence["process_counts"]["launcher"] = 1; evidence["sequence"].append("launcher")
        profile_outcome: dict[str, Any] | None = None

        def profile_runner_executor(
            runner_argv: list[str],
            environment: dict[str, str],
            mark_runner_started: Callable[[], None],
            target_binding: dict[str, Any],
        ) -> dict[str, Any]:
            nonlocal capture_attempted, profile_outcome, stage
            stage = "profile-capture-before"
            evidence["profile_trust"].append(dependencies.profile_trust(value["profile_diagnostic"], "capture-before"))
            stage = "profile-capture"; capture_attempted = True; evidence["process_counts"]["capture_tool"] = 1; evidence["sequence"].append("profile-capture")
            try:
                profile_outcome = dependencies.profile_capture(
                    {
                        "contract": value["profile_diagnostic"],
                        "runner_argv": runner_argv,
                        "environment": environment,
                        "mark_runner_started": mark_runner_started,
                        "target_binding": target_binding,
                    }
                )
                if not isinstance(profile_outcome, dict):
                    raise HarnessError("profile capture executor outcome differs")
                capture = profile_outcome.get("profile_capture")
                if isinstance(capture, dict):
                    evidence["process_counts"]["rocprof"] = capture.get("rocprof_invocations", 0)
                return profile_outcome
            finally:
                evidence["profile_trust"].append(dependencies.profile_trust(value["profile_diagnostic"], "capture-after"))

        try:
            launcher_code, launcher_evidence = dependencies.launcher_execute(
                value["launcher_binding"],
                **({"profile_runner_executor": profile_runner_executor} if profile_diagnostic else {}),
            )
            launcher_failure = launcher_evidence.get("failure")
            runner_not_started = isinstance(launcher_failure, dict) and launcher_failure.get("runner_started") is False
            runner_finished = launcher_code == 0 or not runner_not_started
        except Exception:
            runner_finished = False
            raise
        capture_summary = launcher_evidence.get("profile_capture") if profile_diagnostic else None
        profile_diagnostics = launcher_evidence.get("profile_diagnostics") if profile_diagnostic else None
        profile_lifecycle_validated = (
            profile_diagnostic
            and isinstance(capture_summary, dict)
            and isinstance(profile_diagnostics, dict)
        )
        unverified_profile_outcome = profile_diagnostic and capture_attempted and not profile_lifecycle_validated
        if profile_lifecycle_validated:
            runner_finished = profile_diagnostics.get("runner_finished") is True
            runner_not_started = (
                capture_summary.get("rocprof_started") is False
                and capture_summary.get("runner_started") is False
            )
        elif unverified_profile_outcome:
            # The callback result is visible to maintenance before the launcher
            # validates it.  It is diagnostic input only: lifecycle and cleanup
            # authority comes exclusively from the launcher evidence.
            runner_finished = False
            runner_not_started = False
        capture_children: list[int] | str
        if profile_lifecycle_validated:
            capture_children = capture_summary.get("children_remaining", [])
        elif unverified_profile_outcome:
            capture_children = UNKNOWN_LIFECYCLE_STATE
        else:
            capture_children = []
        launcher_children = launcher_evidence.get("children_remaining", [])
        recorded_children = (
            UNKNOWN_LIFECYCLE_STATE
            if unverified_profile_outcome
            else capture_children or launcher_children
        )
        evidence["launcher"] = {
            "code": launcher_code,
            "status": launcher_evidence.get("status"),
            "safety": launcher_evidence.get("safety"),
            "failure": launcher_evidence.get("failure"),
            "children_remaining": recorded_children,
            "children_state_known": UNKNOWN_LIFECYCLE_STATE if unverified_profile_outcome else capture_summary.get("children_state_known") if isinstance(capture_summary, dict) else None,
            "cleanup_passed": UNKNOWN_LIFECYCLE_STATE if unverified_profile_outcome else capture_summary.get("cleanup_passed") if isinstance(capture_summary, dict) else None,
            "runner_started": UNKNOWN_LIFECYCLE_STATE if unverified_profile_outcome else capture_summary.get("runner_started") if isinstance(capture_summary, dict) else None,
            "runner_finished": UNKNOWN_LIFECYCLE_STATE if unverified_profile_outcome else runner_finished,
            "runner_not_started": UNKNOWN_LIFECYCLE_STATE if unverified_profile_outcome else runner_not_started,
            "profile_lifecycle_evidence_validated": profile_lifecycle_validated if profile_diagnostic else None,
            "sequence": launcher_evidence.get("sequence"),
            "profile_runner_target": launcher_evidence.get("profile_runner_target"),
            "profile_capture": capture_summary,
            "profile_diagnostics": profile_diagnostics,
        }
        runner_evidence = evidence["launcher"]
        evidence["safety"]["gpu_command_executed"] = launcher_evidence.get("safety", {}).get("gpu_command_executed", "unknown")
        evidence["safety"]["model_load_executed"] = launcher_evidence.get("safety", {}).get("model_load_executed", "unknown")
        if profile_diagnostic and isinstance(profile_outcome, dict):
            completed = profile_outcome.get("completed")
            if isinstance(completed, subprocess.CompletedProcess):
                evidence["capture"] = {
                    "command": completed.args,
                    "exit_code": completed.returncode,
                    "stdout_sha256": sha_bytes(completed.stdout),
                    "stderr_sha256": sha_bytes(completed.stderr),
                    **(capture_summary if isinstance(capture_summary, dict) else {}),
                    "diagnostics": profile_diagnostics,
                    "launcher_evidence_validated": profile_lifecycle_validated,
                }
                if unverified_profile_outcome:
                    evidence["capture"].update({
                        "authority": "diagnostic_only_unvalidated",
                        "rocprof_started": UNKNOWN_LIFECYCLE_STATE,
                        "runner_start_known": UNKNOWN_LIFECYCLE_STATE,
                        "runner_started": UNKNOWN_LIFECYCLE_STATE,
                        "runner_completed": UNKNOWN_LIFECYCLE_STATE,
                        "runner_finished": UNKNOWN_LIFECYCLE_STATE,
                        "timed_out": UNKNOWN_LIFECYCLE_STATE,
                        "children_state_known": UNKNOWN_LIFECYCLE_STATE,
                        "children_remaining": UNKNOWN_LIFECYCLE_STATE,
                        "cleanup_passed": UNKNOWN_LIFECYCLE_STATE,
                        "raw_profile_capture": profile_outcome.get("profile_capture"),
                        "raw_profile_diagnostics": profile_outcome.get("profile_diagnostics"),
                    })
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
                if profile_diagnostic and capture_attempted and not profile_lifecycle_validated:
                    cleanup_error = "trusted lock substrate retained because launcher profile lifecycle evidence is unverified"
                    evidence["lock_substrate_cleanup"] = {
                        "passed": False,
                        "attempted": False,
                        "reason": cleanup_error,
                        "runner_finished": UNKNOWN_LIFECYCLE_STATE,
                        "runner_children": UNKNOWN_LIFECYCLE_STATE,
                        "secret_material_recorded": False,
                    }
                    evidence["failure"] = {
                        "stage": "lock-substrate-cleanup",
                        "reason": cleanup_error,
                        "launcher_started": evidence["process_counts"]["launcher"] == 1,
                    }
                    code = 1
                else:
                    try:
                        children = []
                        if isinstance(runner_evidence, dict):
                            for key in ("children_remaining", "child_pids", "children"):
                                value_children = runner_evidence.get(key)
                                if isinstance(value_children, list):
                                    children.extend(value_children)
                        profile_cleanup_safe = (
                            not capture_attempted
                            or (
                                isinstance(runner_evidence, dict)
                                and runner_evidence.get("cleanup_passed") is True
                                and runner_evidence.get("children_state_known") is True
                                and children == []
                            )
                        ) if profile_diagnostic else runner_finished or runner_not_started
                        if not profile_cleanup_safe and not children:
                            # A failed/aborted runner has no trustworthy child
                            # inventory.  Keep the service stopped only long
                            # enough to record cleanup failure; never unlink a
                            # substrate while an unknown child may still hold it.
                            children = [-1]
                        cleanup_runner_finished = runner_finished or (profile_diagnostic and profile_cleanup_safe)
                        cleanup = (
                            dependencies.lock_substrate_cleanup(
                                substrate,
                                dependencies.run,
                                runner_finished=cleanup_runner_finished,
                                runner_children=children,
                            )
                            if dependencies.lock_substrate_cleanup is not None
                            else cleanup_lock_substrate(
                                substrate,
                                dependencies.run,
                                runner_finished=cleanup_runner_finished,
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
            restore_started_ns = dependencies.monotonic_ns()
            restore_state: dict[str, Any] = {
                "attempted": True,
                "passed": False,
                "error": None,
                "post_start": None,
                "lock_substrate_cleanup_passed": cleanup_error is None,
                "started_monotonic_ns": restore_started_ns,
                "deadline_monotonic_ns": restore_started_ns + int(RESTORE_TIMEOUT_SECONDS * 1_000_000_000),
            }
            try:
                evidence["commands"].append(_sudo_valid(dependencies.run, "sudo-before-restore")); evidence["process_counts"]["sudo"] += 1
                started, record = _command(dependencies.run, [str(LAUNCHER.SUDO), "-n", str(LAUNCHER.SYSTEMCTL), "start", SERVICE], "service-start"); evidence["commands"].append(record); evidence["process_counts"]["systemctl_start"] = 1; evidence["sequence"].append("service-start")
                if started.returncode != 0 or started.stdout or started.stderr:
                    raise HarnessError("service start failed")
                if pre is None or package_before is None:
                    raise HarnessError("pre-stop snapshot/package identity is absent during restore")
                expected_previous = pre if evidence["safety"]["service_stopped"] else None
                if expected_previous is None:
                    raise HarnessError("explicitly stopped service epoch expectation is absent")
                post = poll_restore_readiness(dependencies, expected_previous, package_before, restore_state)
                for name, count in post["health"]["formal"]["process_counts"].items():
                    evidence["process_counts"][name] += count
                evidence["sequence"].append("service-restored")
            except (HarnessError, OSError, ValueError, subprocess.SubprocessError) as error:
                restore_error = str(error); code = 1
            restore_state.update({"passed": restore_error is None, "error": restore_error, "post_start": post, "lock_substrate_cleanup_passed": cleanup_error is None})
            restore_state.setdefault("completed_monotonic_ns", dependencies.monotonic_ns())
            restore_state.setdefault("duration_ns", max(0, restore_state["completed_monotonic_ns"] - restore_started_ns))
            evidence["restore"] = restore_state
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
