#!/usr/bin/env python3
"""Collect phase-1 SQ8 OpenWebUI release evidence without claiming release.

The collector owns the single-request schedule, official raw writers, journal
correlation, resource sampling, and bundle hashing.  Browser, cancellation, and
latency producers can add records through the restart hook in a later revision;
phase 1 remains explicitly incomplete and never writes release-validation.json.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import collections
import dataclasses
import errno
import hashlib
import json
import math
import os
import re
import select
import signal
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Iterable, Protocol, Sequence


CONFIG_SCHEMA = "ullm.sq8.openwebui_release.collector.config.v1"
SESSION_SCHEMA = "ullm.sq8.openwebui_release.raw.v1"
RESOURCE_SCHEMA = "ullm.sq8.release_measurement.raw.v1"
LIFECYCLE_SCHEMA = "ullm.gateway.lifecycle.v1"
MATRIX_SCHEMA = "ullm.sq8.openwebui_release.matrix.v1"
HTTP_COMMAND_SCHEMA = "ullm.sq8.openwebui_http_client.command.v1"
HTTP_EVENT_SCHEMA = "ullm.sq8.openwebui_http_client.event.v1"
HOOK_SCHEMA = "ullm.sq8.openwebui_release.hook.v1"

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_HTTP_BODY_BYTES = 4 * 1024 * 1024
MAX_PHASE_ARTIFACT_BYTES = 128 * 1024 * 1024
MAX_DIAGNOSTIC_BYTES = 64 * 1024
MAX_HOOK_RECORDS = 64
MAX_HOOK_RECORD_BYTES = 256 * 1024
COPY_CHUNK_BYTES = 1024 * 1024
COMMAND_TIMEOUT_SECONDS = 30.0
HTTP_REQUEST_TIMEOUT_SECONDS = 240.0
RELEASE_TIMEOUT_NS = 5_000_000_000
RESTART_TIMEOUT_NS = 600_000_000_000
NEGATIVE_QUIET_NS = 250_000_000
JOURNAL_POLL_NS = 50_000_000
IDLE_SETTLE_NS = 5_000_000_000
SAMPLE_INTERVAL_NS = 1_000_000_000
KFD_SNAPSHOT_TIMEOUT_NS = 1_000_000_000
KFD_RETRY_SLEEP_SECONDS = 0.005

GPU_INDEX = 2
GPU_BDF = "0000:47:00.0"
GPU_UUID = "a8ff7551-0000-1000-80e9-ddefa2d60f55"
KFD_GPU_ID = 51_545
SERVICE_UNIT = "ullm-openai.service"
DOCKER_BIN = "/usr/bin/docker"
AMD_SMI_BIN = "/opt/rocm/bin/amd-smi"
HTTP_NETWORK_NAME = "open-webui-network"
HTTP_NETWORK_SUBNET = "172.20.0.0/16"
HTTP_NETWORK_GATEWAY = "172.20.0.1"
HTTP_BASE_URL = "http://172.20.0.1:8000"
HTTP_READY_URL = HTTP_BASE_URL + "/ready"
HTTP_CLIENT_UID = 1000
HTTP_CLIENT_GID = 1000
HTTP_CLIENT_SOURCE_RELATIVE = "tools/sq8-openwebui-http-client.py"
HTTP_CLIENT_INPUT_PATH = HTTP_CLIENT_SOURCE_RELATIVE
RESOURCE_FIXTURE_INPUT_PATH = "collector/resource-chat-fixture.json"
CONTAINER_CLIENT_PATH = "/run/ullm/sq8-openwebui-http-client.py"
CONTAINER_API_KEY_PATH = "/run/secrets/ullm-api-key"
CONTEXT_OVERFLOW_CONTENT = {
    "context_overflow_1": "one" + (" overflow" * 5000),
    "context_overflow_2": "two" + (" overflow" * 5000),
}

FIXTURE_IDS = (
    "exact-p0032",
    "exact-p0128",
    "exact-p0512",
    "exact-p2048",
    "exact-p3584",
)
CANCEL_PHASES = (
    "after_started_before_progress",
    "prefill_after_128",
    "prefill_after_2048",
    "decode_after_first_content",
    "openwebui_stop_after_visible_content",
)
SCHEDULE = {
    "openwebui_chats": 20,
    "cancel_phases": list(CANCEL_PHASES),
    "normal_warmups": 10,
    "normal_requests": 100,
    "sampled_normal_indices": list(range(5, 101, 5)),
    "restart_warmups": 10,
    "restart_requests": 20,
    "ttft_fixture_ids": list(FIXTURE_IDS),
    "latency_warmups_per_case": 2,
    "latency_measured_per_case": 10,
    "decode_warmups": 2,
    "decode_measured": 10,
    "idle_settle_ms": 5000,
    "samples_per_point": 5,
    "sample_interval_ms": 1000,
}
RESOURCE_SCHEDULE = {
    "normal_warmups": 10,
    "normal_requests": 100,
    "restart_warmups": 10,
    "restart_requests": 20,
    "idle_settle_ms": 5000,
    "samples_per_point": 5,
    "sample_interval_ms": 1000,
}
THRESHOLDS = {
    "ttft_seconds_maximum": {
        "exact-p0032": {"p50": 2.5, "p95": 3},
        "exact-p0128": {"p50": 4, "p95": 5},
        "exact-p0512": {"p50": 10, "p95": 12},
        "exact-p2048": {"p50": 30, "p95": 35},
        "exact-p3584": {"p50": 50, "p95": 60},
    },
    "decode_p50_tokens_per_second_minimum": 15,
    "decode_p95_inter_content_seconds_maximum": 0.1,
    "cancel_release_max_ns": RELEASE_TIMEOUT_NS,
    "final_delta_max_bytes": 67_108_864,
    "theil_sen_max_bytes_per_request": 262_144,
}

COMMANDS = {
    "systemd_version": "systemctl --version",
    "service_identity": (
        "systemctl show ullm-openai.service --property=ControlGroup "
        "--property=MainPID --no-pager"
    ),
    "cgroup_type": "stat -fc %T /sys/fs/cgroup",
    "host_memory": "cat /sys/fs/cgroup${ControlGroup}/memory.current",
    "proc_stat": "cat /proc/${PID}/stat",
    "proc_status": "cat /proc/${PID}/status",
    "proc_exe": "readlink /proc/${PID}/exe",
    "proc_fds": "find -P /proc/${PID}/fd -mindepth 1 -maxdepth 1 -printf '%f\\n'",
    "proc_children": "cat /proc/${PID}/task/${PID}/children",
    "amd_smi_version": "amd-smi version",
    "amd_smi_list": "amd-smi list --json",
    "amd_smi_process": "amd-smi process --gpu 2 --general --json",
    "amd_smi_metric": "amd-smi metric --gpu 2 --json",
    "kfd_proc_probe": "test -d /sys/class/kfd/kfd/proc",
    "kfd_processes": (
        "find -P /sys/class/kfd/kfd/proc -mindepth 1 -maxdepth 1 -printf '%f\\n'"
    ),
    "kfd_vram": "cat /sys/class/kfd/kfd/proc/${PID}/vram_51545",
}

EXPECTED_ROLES = {
    "environment.json": "environment",
    "model-identity.json": "model_identity",
    "raw-session-results.jsonl": "session_raw",
    "soak-resources.raw.jsonl": "resource_raw",
    "service-journal.raw.jsonl": "service_journal_raw",
    "amd-smi-metric-normal-before.json": "gpu_metric_raw",
    "amd-smi-metric-normal-after.json": "gpu_metric_raw",
    "amd-smi-metric-restart-before.json": "gpu_metric_raw",
    "amd-smi-metric-restart-after.json": "gpu_metric_raw",
    "sampling-results.json": "derived_view",
    "cancel-results.json": "derived_view",
    "prefill-latency-results.json": "derived_view",
    "api-contract-results.json": "derived_view",
    "openwebui-smoke.json": "derived_view",
    "soak-results.json": "derived_view",
    "browser/openwebui-stop-before.png": "browser_screenshot",
    "browser/post-header-failure.png": "browser_screenshot",
}
PHASE_ARTIFACT_PATHS = {
    "environment.json",
    "model-identity.json",
    "sampling-results.json",
    "cancel-results.json",
    "prefill-latency-results.json",
    "api-contract-results.json",
    "openwebui-smoke.json",
    "soak-results.json",
    "browser/openwebui-stop-before.png",
    "browser/post-header-failure.png",
    "summary.md",
}
BUNDLE_FILES = set(EXPECTED_ROLES) | {
    "release-matrix.json",
    "summary.md",
    "SHA256SUMS",
}

RESOURCE_HEADER_FIELDS = {
    "schema_version",
    "record_type",
    "service_unit",
    "commands",
    "tools",
    "probes",
    "schedule",
}
LIFECYCLE_FIELDS = {
    "request_admitted": {
        "request_id",
        "completion_id",
        "stream",
        "prompt_tokens",
        "max_completion_tokens",
    },
    "request_started": {
        "request_id",
        "completion_id",
        "stream",
        "prompt_tokens",
        "admit_to_start_ns",
    },
    "request_progress": {
        "request_id",
        "completion_id",
        "phase",
        "processed_prompt_tokens",
        "prompt_tokens",
    },
    "request_first_token": {
        "request_id",
        "completion_id",
        "stream",
        "completion_tokens",
    },
    "request_cancel_requested": {
        "request_id",
        "completion_id",
        "stream",
        "reason",
        "admit_to_cancel_ns",
    },
    "request_released": {
        "request_id",
        "completion_id",
        "stream",
        "outcome",
        "cancel_reason",
        "prompt_tokens",
        "completion_tokens",
        "reset_complete",
        "admit_to_start_ns",
        "start_to_release_ns",
        "admit_to_release_ns",
    },
    "worker_fatal": {
        "request_id",
        "completion_id",
        "reason",
        "admit_to_fatal_ns",
    },
}
BROWSER_ACTION_FIELDS = {
    "browser_case",
    "action_index",
    "action",
    "selector",
    "input_sha256",
    "started_monotonic_ns",
    "completed_monotonic_ns",
    "result",
    "screenshot_file",
    "screenshot_sha256",
}
FAULT_INJECTION_FIELDS = {
    "injection",
    "target_pid",
    "target_starttime_ticks",
    "signal",
    "command",
    "started_monotonic_ns",
    "completed_monotonic_ns",
}

GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


class CollectorError(RuntimeError):
    """A fail-closed collection error with no credential-bearing detail."""


def fail(message: str) -> None:
    raise CollectorError(message)


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict:
        fail(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        fail(
            f"{label} fields differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail(f"{label} must be an integer >= {minimum}")
    return value


def nonempty_string(value: Any, label: str, *, maximum: int = 4096) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        fail(f"{label} must be a bounded non-empty string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        fail(f"{label} is not strict UTF-8")
    return value


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> None:
    fail(f"JSON contains non-finite numeric constant {value}")


def parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        fail("JSON contains a non-finite number")
    return parsed


def strict_json_bytes(raw: bytes, label: str, *, maximum: int = MAX_JSON_BYTES) -> Any:
    if not raw or len(raw) > maximum:
        fail(f"{label} has an invalid size")
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_float=parse_finite_float,
            parse_constant=reject_json_constant,
        )
    except CollectorError:
        raise
    except (UnicodeError, ValueError, RecursionError):
        fail(f"{label} is not strict UTF-8 JSON")


def strict_json_object(
    raw: bytes, label: str, *, maximum: int = MAX_JSON_BYTES
) -> dict[str, Any]:
    value = strict_json_bytes(raw, label, maximum=maximum)
    if type(value) is not dict:
        fail(f"{label} root must be an object")
    return value


def compact_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        fail("internal evidence record is not strict JSON")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(COPY_CHUNK_BYTES):
                digest.update(chunk)
    except OSError:
        fail("failed to hash an evidence file")
    return digest.hexdigest()


def regular_file(path: Path, label: str, *, maximum: int | None = None) -> Path:
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError:
            fail(f"failed to stat {label}")
        if stat.S_ISLNK(metadata.st_mode):
            fail(f"{label} contains a symlink path component")
    try:
        metadata = absolute.stat()
    except OSError:
        fail(f"failed to stat {label}")
    if not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} must be a regular file")
    if maximum is not None and (metadata.st_size <= 0 or metadata.st_size > maximum):
        fail(f"{label} has an invalid size")
    return absolute


def safe_relative(value: Any, label: str) -> str:
    text = nonempty_string(value, label)
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or "\\" in text
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        fail(f"{label} is not a safe relative path")
    return text


def git_commit(value: Any, label: str) -> str:
    text = nonempty_string(value, label, maximum=40)
    if GIT_COMMIT_RE.fullmatch(text) is None:
        fail(f"{label} is not a lowercase Git commit")
    return text


def sha256_value(value: Any, label: str) -> str:
    text = nonempty_string(value, label, maximum=64)
    if SHA256_RE.fullmatch(text) is None:
        fail(f"{label} is not a lowercase SHA-256")
    return text


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class SecretGuard:
    def __init__(self, value: bytes):
        if len(value) < 16 or len(value) > 4096 or b"\x00" in value:
            fail("API credential file has an invalid secret length")
        self._value = value

    @classmethod
    def from_file(cls, path: Path) -> "SecretGuard":
        guard, _ = cls.snapshot_from_file(path)
        return guard

    @classmethod
    def snapshot_from_file(cls, path: Path) -> tuple["SecretGuard", bytes]:
        _, raw, _ = read_regular_snapshot(
            path,
            "API credential file",
            maximum=4097,
            require_private=True,
        )
        if raw.endswith(b"\n"):
            raw = raw[:-1]
        if b"\r" in raw or b"\n" in raw:
            fail(
                "API credential file must contain one LF-terminated or unterminated line"
            )
        return cls(raw), raw

    def reject(self, raw: bytes, label: str) -> None:
        if self._value in raw:
            fail(f"{label} contains the API credential")

    def scanner(self, label: str) -> "StreamingSecretScanner":
        return StreamingSecretScanner(self._value, label)

    def scan_file(self, path: Path, label: str) -> None:
        overlap = max(0, len(self._value) - 1)
        tail = b""
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(COPY_CHUNK_BYTES):
                    combined = tail + chunk
                    self.reject(combined, label)
                    tail = combined[-overlap:] if overlap else b""
        except OSError:
            fail(f"failed to scan {label}")


class StreamingSecretScanner:
    def __init__(self, secret: bytes, label: str):
        self.secret = secret
        self.label = label
        self.tail = b""

    def feed(self, chunk: bytes) -> None:
        combined = self.tail + chunk
        if self.secret in combined:
            fail(f"{self.label} contains the API credential")
        overlap = len(self.secret) - 1
        self.tail = combined[-overlap:] if overlap else b""


def read_regular_snapshot(
    path: Path,
    label: str,
    *,
    maximum: int,
    require_private: bool = False,
) -> tuple[Path, bytes, os.stat_result]:
    """Read one stable regular-file snapshot through a no-follow descriptor."""

    absolute = path if path.is_absolute() else Path.cwd() / path
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(absolute, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail(f"{label} must be a regular file")
        if before.st_size <= 0 or before.st_size > maximum:
            fail(f"{label} has an invalid size")
        if require_private and stat.S_IMODE(before.st_mode) & 0o077:
            fail(f"{label} permissions must not grant group or other access")
        raw = read_fd_bounded(descriptor, maximum, label)
        after = os.fstat(descriptor)
        if (
            stable_fd_identity(before) != stable_fd_identity(after)
            or len(raw) != before.st_size
        ):
            fail(f"{label} changed while it was read")
        return absolute, raw, before
    except CollectorError:
        raise
    except OSError:
        fail(f"failed to read {label} without following links")
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                fail(f"failed to close {label}")


@dataclasses.dataclass
class RuntimeSnapshots:
    directory: Path
    client_path: Path
    credential_path: Path
    closed: bool = False

    @classmethod
    def create(
        cls,
        client_raw: bytes,
        credential_raw: bytes,
        *,
        parent: Path | None = None,
    ) -> "RuntimeSnapshots":
        if os.geteuid() != HTTP_CLIENT_UID or os.getegid() != HTTP_CLIENT_GID:
            fail("HTTP client snapshots must be created by the fixed production user")
        if parent is None:
            parent = Path(f"/run/user/{os.geteuid()}")
        try:
            metadata = parent.lstat()
        except OSError:
            fail("secure runtime snapshot parent is unavailable")
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            fail("secure runtime snapshot parent ownership or mode is unsafe")
        directory: Path | None = None
        try:
            directory = Path(tempfile.mkdtemp(prefix="ullm-p8f-http-", dir=parent))
            os.chmod(directory, 0o700)
            client_path = directory / "sq8-openwebui-http-client.py"
            credential_path = directory / "openai-api-key"
            cls._write_private(client_path, client_raw, 0o500)
            cls._write_private(credential_path, credential_raw, 0o600)
        except (CollectorError, OSError) as error:
            if directory is not None:
                try:
                    shutil.rmtree(directory)
                except OSError:
                    fail("failed to clean an incomplete secure runtime snapshot")
            if isinstance(error, CollectorError):
                raise
            fail("failed to create secure runtime snapshots")
        return cls(directory, client_path, credential_path)

    @staticmethod
    def _write_private(path: Path, raw: bytes, mode: int) -> None:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            mode,
        )
        try:
            write_all(descriptor, raw, f"runtime snapshot {path.name}")
            os.fsync(descriptor)
            os.fchmod(descriptor, mode)
        finally:
            os.close(descriptor)

    def unlink_credential(self) -> None:
        try:
            self.credential_path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            fail("failed to remove the host credential snapshot after client startup")

    def close(self) -> None:
        if self.closed:
            return
        try:
            shutil.rmtree(self.directory)
        except FileNotFoundError:
            pass
        except OSError:
            fail("failed to remove secure runtime snapshots")
        self.closed = True


def stable_fd_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


@dataclasses.dataclass(frozen=True)
class FileSeal:
    identity: tuple[int, int, int, int, int]
    sha256: str

    @property
    def size(self) -> int:
        return self.identity[2]


def inspect_sealed_file(
    path: Path,
    label: str,
    guard: SecretGuard,
    *,
    expected: FileSeal | None = None,
) -> FileSeal:
    descriptor = -1
    digest = hashlib.sha256()
    scanner = guard.scanner(label)
    total = 0
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail(f"{label} must be a regular file")
        while True:
            chunk = os.read(descriptor, COPY_CHUNK_BYTES)
            if not chunk:
                break
            scanner.feed(chunk)
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        identity = stable_fd_identity(before)
        if identity != stable_fd_identity(after) or total != before.st_size:
            fail(f"{label} changed while it was sealed")
        seal = FileSeal(identity, digest.hexdigest())
        if expected is not None and seal != expected:
            fail(f"{label} differs from its staged identity or content")
        return seal
    except CollectorError:
        raise
    except OSError:
        fail(f"failed to seal {label} without following links")
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                fail(f"failed to close {label}")


def write_all(descriptor: int, raw: bytes, label: str) -> None:
    offset = 0
    while offset < len(raw):
        try:
            written = os.write(descriptor, raw[offset:])
        except OSError:
            fail(f"failed to write {label}")
        if written <= 0:
            fail(f"short write while writing {label}")
        offset += written


def read_fd_bounded(descriptor: int, maximum: int, label: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            chunk = os.read(descriptor, min(COPY_CHUNK_BYTES, maximum + 1 - total))
        except OSError:
            fail(f"failed to read {label}")
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum:
            fail(f"{label} exceeds its size limit")


def inspect_input_file(item: "InputFile", guard: SecretGuard) -> tuple[int, str]:
    label = f"input file {item.path}"
    if item.snapshot is not None:
        guard.reject(item.snapshot, label)
        return len(item.snapshot), sha256_bytes(item.snapshot)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(item.source_file, flags)
    except OSError:
        fail(f"failed to open {label}")
    digest = hashlib.sha256()
    scanner = guard.scanner(label)
    total = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail(f"{label} is not a regular file")
        if item.expected_device is not None and (
            before.st_dev != item.expected_device
            or before.st_ino != item.expected_inode
        ):
            fail(f"{label} identity changed after configuration load")
        while True:
            chunk = os.read(descriptor, COPY_CHUNK_BYTES)
            if not chunk:
                break
            scanner.feed(chunk)
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        if (
            stable_fd_identity(before) != stable_fd_identity(after)
            or total != before.st_size
        ):
            fail(f"{label} changed during inspection")
    except OSError:
        fail(f"failed to inspect {label}")
    finally:
        try:
            os.close(descriptor)
        except OSError:
            fail(f"failed to close {label}")
    return total, digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class ObserverDatagram:
    raw_payload: bytes
    event: dict[str, Any]
    received_monotonic_ns: int
    sender_pid: int
    sender_uid: int
    sender_gid: int

    @property
    def mirror_delay_ns(self) -> int:
        return self.received_monotonic_ns - self.event["observed_monotonic_ns"]


class LifecycleObserver:
    """Receive the low-latency mirror; journal evidence remains authoritative."""

    MAX_DATAGRAM_BYTES = 64 * 1024

    def __init__(
        self,
        path: Path,
        guard: SecretGuard,
        *,
        expected_uid: int,
        expected_gid: int | None = None,
    ):
        self.path = path
        self.guard = guard
        self.expected_uid = expected_uid
        self.expected_gid = expected_gid
        self.socket: socket.socket | None = None

    def open(self) -> None:
        if self.socket is not None:
            fail("lifecycle observer is already open")
        parent = self.path.parent
        try:
            parent_metadata = parent.lstat()
        except OSError:
            fail("lifecycle observer parent directory is unavailable")
        if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(
            parent_metadata.st_mode
        ):
            fail("lifecycle observer parent is not a regular directory")
        if parent_metadata.st_uid != os.geteuid() or parent_metadata.st_mode & 0o022:
            fail("lifecycle observer parent ownership or mode is unsafe")
        if self.path.exists() or self.path.is_symlink():
            fail("lifecycle observer socket path already exists")
        observer = socket.socket(
            socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC
        )
        try:
            observer.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            observer.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            observer.bind(os.fspath(self.path))
            os.chmod(self.path, 0o600)
            metadata = self.path.lstat()
            if (
                not stat.S_ISSOCK(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                fail("lifecycle observer socket identity or mode differs")
        except (CollectorError, OSError):
            observer.close()
            try:
                self.path.unlink()
            except OSError:
                pass
            raise
        self.socket = observer

    def close(self) -> None:
        observer = self.socket
        self.socket = None
        if observer is not None:
            observer.close()
        try:
            metadata = self.path.lstat()
        except FileNotFoundError:
            return
        except OSError:
            fail("failed to stat lifecycle observer socket during cleanup")
        if not stat.S_ISSOCK(metadata.st_mode) or metadata.st_uid != os.geteuid():
            fail("refusing to unlink a replaced lifecycle observer path")
        try:
            self.path.unlink()
        except OSError:
            fail("failed to remove lifecycle observer socket")

    def receive(
        self, deadline_ns: int, *, expected_sender_pid: int
    ) -> ObserverDatagram:
        observer = self.socket
        if observer is None:
            fail("lifecycle observer is not open")
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            fail("lifecycle observer deadline expired")
        ready, _, _ = select.select([observer], [], [], remaining_ns / 1_000_000_000)
        if not ready:
            fail("lifecycle observer datagram timed out")
        try:
            payload, ancillary, flags, _ = observer.recvmsg(
                self.MAX_DATAGRAM_BYTES,
                socket.CMSG_SPACE(struct.calcsize("3i")),
            )
            received = time.monotonic_ns()
        except OSError:
            fail("failed to receive a lifecycle observer datagram")
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            fail("lifecycle observer datagram or credentials were truncated")
        credentials = []
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS:
                if len(data) < struct.calcsize("3i"):
                    fail("lifecycle observer credentials are truncated")
                credentials.append(struct.unpack("3i", data[: struct.calcsize("3i")]))
        if len(credentials) != 1:
            fail("lifecycle observer lacks exactly one sender credential")
        sender_pid, sender_uid, sender_gid = credentials[0]
        if sender_pid != expected_sender_pid or sender_uid != self.expected_uid:
            fail("lifecycle observer sender PID or UID differs")
        if self.expected_gid is not None and sender_gid != self.expected_gid:
            fail("lifecycle observer sender GID differs")
        if not payload or len(payload) >= self.MAX_DATAGRAM_BYTES:
            fail("lifecycle observer payload has an invalid size")
        self.guard.reject(payload, "lifecycle observer payload")
        try:
            payload.decode("ascii", errors="strict")
        except UnicodeError:
            fail("lifecycle observer payload is not ASCII")
        event = decode_lifecycle_payload(payload, "lifecycle observer payload")
        delay = received - event["observed_monotonic_ns"]
        if delay < 0:
            fail("lifecycle observer event timestamp is after receipt")
        return ObserverDatagram(
            raw_payload=payload,
            event=event,
            received_monotonic_ns=received,
            sender_pid=sender_pid,
            sender_uid=sender_uid,
            sender_gid=sender_gid,
        )

    def require_empty(self) -> None:
        observer = self.socket
        if observer is None:
            fail("lifecycle observer is not open")
        ready, _, _ = select.select([observer], [], [], 0)
        if ready:
            fail("lifecycle observer contains a stale or unconsumed datagram")


class ObserverJournalCorrelator:
    def __init__(self):
        self.pending: collections.deque[ObserverDatagram] = collections.deque()

    def observe(self, datagram: ObserverDatagram) -> None:
        if self.pending:
            previous = self.pending[-1].event["observed_monotonic_ns"]
            if datagram.event["observed_monotonic_ns"] < previous:
                fail("lifecycle observer event timestamps regressed")
        self.pending.append(datagram)

    def correlate_journal_message(
        self, message: str, event: dict[str, Any]
    ) -> ObserverDatagram:
        if not self.pending:
            fail("journal lifecycle event lacks its observer datagram")
        datagram = self.pending.popleft()
        raw = lifecycle_payload_from_message(message)
        if raw != datagram.raw_payload or event != datagram.event:
            fail("observer datagram and authoritative journal MESSAGE differ")
        return datagram

    def require_complete(self) -> None:
        if self.pending:
            fail("lifecycle observer datagram lacks authoritative journal evidence")


@dataclasses.dataclass(frozen=True)
class InputFile:
    path: str
    source_file: Path
    snapshot: bytes | None = None
    expected_device: int | None = None
    expected_inode: int | None = None


@dataclasses.dataclass(frozen=True)
class NegativeCase:
    after_request: int
    name: str
    body: bytes
    expected_status: int


@dataclasses.dataclass(frozen=True)
class CollectorConfig:
    run_id: str
    identities: dict[str, Any]
    input_files: tuple[InputFile, ...]
    phase_artifacts: dict[str, Path]
    target: str
    resource_body_template: dict[str, Any]
    negative_cases: tuple[NegativeCase, ...]
    restart_command: tuple[str, ...]
    ready_url: str
    amd_smi: str
    phase_artifact_identities: dict[str, tuple[int, int]] | None = None


def canonical_base64(value: Any, label: str) -> bytes:
    text = nonempty_string(value, label, maximum=((MAX_HTTP_BODY_BYTES + 2) // 3) * 4)
    try:
        raw = base64.b64decode(text, validate=True)
    except (ValueError, binascii.Error):
        fail(f"{label} is not canonical base64")
    if (
        not raw
        or len(raw) > MAX_HTTP_BODY_BYTES
        or base64.b64encode(raw).decode("ascii") != text
    ):
        fail(f"{label} is not canonical bounded base64")
    return raw


def require_malformed_json(raw: bytes, label: str) -> None:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError:
        fail(f"{label} must be strict UTF-8")
    try:
        json.loads(text, parse_constant=reject_json_constant)
    except (CollectorError, json.JSONDecodeError, ValueError, RecursionError):
        return
    fail(f"{label} must be malformed JSON")


def validate_overflow_request_body(
    raw: bytes,
    template: dict[str, Any],
    case_name: str,
    label: str,
) -> None:
    value = strict_json_object(raw, label, maximum=MAX_HTTP_BODY_BYTES)
    exact_keys(
        value,
        {
            "model",
            "messages",
            "stream",
            "stream_options",
            "max_tokens",
            "temperature",
            "top_p",
            "seed",
        },
        label,
    )
    stream_options = exact_keys(
        value["stream_options"], {"include_usage"}, f"{label}.stream_options"
    )
    if (
        value["model"] != template["model"]
        or value["messages"]
        != [
            {
                "role": "user",
                "content": CONTEXT_OVERFLOW_CONTENT[case_name],
            }
        ]
        or value["stream"] is not True
        or stream_options["include_usage"] is not True
        or type(value["max_tokens"]) is not int
        or value["max_tokens"] != 2
        or type(value["temperature"]) is not int
        or value["temperature"] != 0
        or type(value["top_p"]) is not int
        or value["top_p"] != 1
        or type(value["seed"]) is not int
        or value["seed"] != 0
    ):
        fail(f"{label} differs from the supported context-overflow request shape")


def parse_command_array(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value or len(value) > 128:
        fail(f"{label} must be a non-empty bounded array")
    result = tuple(nonempty_string(item, f"{label}[]", maximum=4096) for item in value)
    if any("\x00" in item for item in result):
        fail(f"{label} contains NUL")
    return result


def build_http_client_command(
    config: CollectorConfig,
    snapshots: RuntimeSnapshots,
) -> tuple[str, ...]:
    for path in (snapshots.client_path, snapshots.credential_path):
        if not path.is_absolute() or any(
            character in os.fspath(path) for character in ",\n\r"
        ):
            fail("runtime snapshot path cannot be represented as a Docker bind mount")
    image_id = config.identities["openwebui"]["derived_image_id"]
    return (
        DOCKER_BIN,
        "run",
        "--rm",
        "--interactive",
        "--pull=never",
        f"--network={HTTP_NETWORK_NAME}",
        f"--user={HTTP_CLIENT_UID}:{HTTP_CLIENT_GID}",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        "--memory=256m",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16777216",
        (
            "--mount=type=bind,src="
            f"{snapshots.client_path},dst={CONTAINER_CLIENT_PATH},readonly"
        ),
        (
            "--mount=type=bind,src="
            f"{snapshots.credential_path},dst={CONTAINER_API_KEY_PATH},readonly"
        ),
        "--entrypoint=python3",
        image_id,
        CONTAINER_CLIENT_PATH,
        "--base-url",
        HTTP_BASE_URL,
        "--api-key-file",
        CONTAINER_API_KEY_PATH,
    )


def load_config(
    path: Path,
    *,
    http_client_snapshot: bytes | None = None,
) -> CollectorConfig:
    config_path, config_raw, _ = read_regular_snapshot(
        path,
        "collector config",
        maximum=MAX_JSON_BYTES,
    )
    document = strict_json_object(config_raw, "collector config")
    exact_keys(
        document,
        {
            "schema_version",
            "run_id",
            "identities",
            "input_files",
            "phase_artifacts",
            "http",
            "restart_command",
            "ready_url",
            "amd_smi",
        },
        "collector config",
    )
    if document["schema_version"] != CONFIG_SCHEMA:
        fail("collector config schema_version differs")
    run_id = nonempty_string(document["run_id"], "collector config.run_id", maximum=128)
    if RUN_ID_RE.fullmatch(run_id) is None:
        fail("collector config.run_id has invalid syntax")

    identities = exact_keys(
        document["identities"],
        {
            "openwebui",
            "docker_network_id",
            "gateway_source_sha256",
            "worker_source_sha256",
            "worker_binary_sha256",
        },
        "collector config.identities",
    )
    openwebui = exact_keys(
        identities["openwebui"],
        {
            "version",
            "source_revision",
            "base_image_digest",
            "base_image_id",
            "derived_image_id",
            "Dockerfile_sha256",
            "patch_sha256",
            "patched_middleware_sha256",
        },
        "collector config.identities.openwebui",
    )
    for key in ("version", "source_revision"):
        nonempty_string(openwebui[key], f"collector config.identities.openwebui.{key}")
    for key in ("base_image_digest", "base_image_id", "derived_image_id"):
        value = nonempty_string(
            openwebui[key], f"collector config.identities.openwebui.{key}", maximum=71
        )
        if not value.startswith("sha256:") or SHA256_RE.fullmatch(value[7:]) is None:
            fail(
                f"collector config.identities.openwebui.{key} is not a content image identity"
            )
    for key in ("Dockerfile_sha256", "patch_sha256", "patched_middleware_sha256"):
        sha256_value(openwebui[key], f"collector config.identities.openwebui.{key}")
    network_id = nonempty_string(
        identities["docker_network_id"],
        "collector config.identities.docker_network_id",
        maximum=64,
    )
    if SHA256_RE.fullmatch(network_id) is None:
        fail("collector config.identities.docker_network_id is not a 64-hex network ID")
    for key in (
        "gateway_source_sha256",
        "worker_source_sha256",
        "worker_binary_sha256",
    ):
        sha256_value(identities[key], f"collector config.identities.{key}")

    input_value = document["input_files"]
    if type(input_value) is not list or not input_value:
        fail("collector config.input_files must be a non-empty array")
    inputs: list[InputFile] = []
    for index, item in enumerate(input_value):
        exact_keys(
            item, {"path", "source_file"}, f"collector config.input_files[{index}]"
        )
        source = regular_file(
            Path(
                nonempty_string(
                    item["source_file"],
                    f"collector config.input_files[{index}].source_file",
                )
            ),
            f"collector input {index}",
        )
        metadata = source.stat()
        inputs.append(
            InputFile(
                path=safe_relative(
                    item["path"], f"collector config.input_files[{index}].path"
                ),
                source_file=source,
                expected_device=metadata.st_dev,
                expected_inode=metadata.st_ino,
            )
        )
    paths = [item.path for item in inputs]
    if paths != sorted(set(paths), key=lambda item: item.encode("utf-8")):
        fail("collector config.input_files must be bytewise ascending and unique")
    artifact_value = exact_keys(
        document["phase_artifacts"],
        PHASE_ARTIFACT_PATHS,
        "collector config.phase_artifacts",
    )
    phase_artifacts: dict[str, Path] = {}
    phase_artifact_identities: dict[str, tuple[int, int]] = {}
    for relative, source in artifact_value.items():
        artifact = regular_file(
            Path(
                nonempty_string(
                    source,
                    f"collector config.phase_artifacts.{relative}",
                )
            ),
            f"phase artifact {relative}",
            maximum=MAX_PHASE_ARTIFACT_BYTES,
        )
        metadata = artifact.stat()
        phase_artifacts[relative] = artifact
        phase_artifact_identities[relative] = (metadata.st_dev, metadata.st_ino)

    http_value = exact_keys(
        document["http"],
        {"target", "resource_body_template", "negative_cases"},
        "collector config.http",
    )
    target = nonempty_string(http_value["target"], "collector config.http.target")
    try:
        target.encode("ascii", errors="strict")
    except UnicodeError:
        fail("collector config.http.target must be ASCII")
    if not target.startswith("/") or target.startswith("//") or "#" in target:
        fail("collector config.http.target is not an origin-form target")
    template = exact_keys(
        http_value["resource_body_template"],
        {"model", "messages"},
        "collector config.http.resource_body_template",
    )
    nonempty_string(
        template["model"], "collector config.http.resource_body_template.model"
    )
    if type(template["messages"]) is not list or not template["messages"]:
        fail("collector config HTTP messages must be a non-empty array")
    for index, message in enumerate(template["messages"]):
        exact_keys(
            message,
            {"role", "content"},
            f"collector config HTTP messages[{index}]",
        )
        if message["role"] not in {"system", "user", "assistant"}:
            fail(f"collector config HTTP messages[{index}].role differs")
        nonempty_string(
            message["content"], f"collector config HTTP messages[{index}].content"
        )
    if len(compact_json(template)) > MAX_HTTP_BODY_BYTES:
        fail("collector config HTTP template is too large")

    collector_implementation, collector_raw, _ = read_regular_snapshot(
        Path(__file__),
        "collector implementation",
        maximum=MAX_JSON_BYTES,
    )
    if http_client_snapshot is None:
        http_client_implementation, http_client_snapshot, _ = read_regular_snapshot(
            Path(__file__).with_name("sq8-openwebui-http-client.py"),
            "HTTP client implementation",
            maximum=MAX_JSON_BYTES,
        )
    else:
        http_client_implementation = Path(__file__).with_name(
            "sq8-openwebui-http-client.py"
        )
        if not http_client_snapshot or len(http_client_snapshot) > MAX_JSON_BYTES:
            fail("HTTP client implementation snapshot has an invalid size")
    fixture_raw = compact_json(
        {"model": template["model"], "messages": template["messages"]}
    )
    automatic_inputs = {
        "collector/config.json": InputFile(
            path="collector/config.json",
            source_file=config_path,
            snapshot=config_raw,
        ),
        RESOURCE_FIXTURE_INPUT_PATH: InputFile(
            path=RESOURCE_FIXTURE_INPUT_PATH,
            source_file=config_path,
            snapshot=fixture_raw,
        ),
        "tools/collect-sq8-openwebui-release.py": InputFile(
            path="tools/collect-sq8-openwebui-release.py",
            source_file=collector_implementation,
            snapshot=collector_raw,
        ),
        HTTP_CLIENT_INPUT_PATH: InputFile(
            path=HTTP_CLIENT_INPUT_PATH,
            source_file=http_client_implementation,
            snapshot=http_client_snapshot,
        ),
    }
    if set(paths) & set(automatic_inputs):
        fail("collector config.input_files uses a producer-reserved path")
    inputs.extend(automatic_inputs.values())
    inputs.sort(key=lambda item: item.path.encode("utf-8"))

    negative_value = http_value["negative_cases"]
    if type(negative_value) is not list or len(negative_value) != 3:
        fail("collector config.http.negative_cases must contain exactly three cases")
    negatives: list[NegativeCase] = []
    for index, item in enumerate(negative_value):
        exact_keys(
            item,
            {"after_request", "name", "body_base64", "expected_status"},
            f"collector config.http.negative_cases[{index}]",
        )
        negatives.append(
            NegativeCase(
                after_request=integer(
                    item["after_request"], "negative after_request", minimum=1
                ),
                name=nonempty_string(item["name"], "negative name", maximum=64),
                body=canonical_base64(item["body_base64"], "negative body_base64"),
                expected_status=integer(
                    item["expected_status"], "negative expected_status", minimum=100
                ),
            )
        )
    if [(item.after_request, item.name) for item in negatives] != [
        (25, "context_overflow_1"),
        (50, "malformed_json"),
        (75, "context_overflow_2"),
    ] or any(item.expected_status != 400 for item in negatives):
        fail("collector config negative request schedule differs")
    for item in negatives:
        if item.name == "malformed_json":
            require_malformed_json(item.body, "malformed JSON negative request")
        else:
            validate_overflow_request_body(
                item.body,
                template,
                item.name,
                f"context-overflow negative request {item.name}",
            )

    ready_url = nonempty_string(document["ready_url"], "collector config.ready_url")
    parsed_ready = urllib.parse.urlsplit(ready_url)
    if (
        parsed_ready.scheme != "http"
        or not parsed_ready.hostname
        or parsed_ready.fragment
        or parsed_ready.query
    ):
        fail(
            "collector config.ready_url must be a plain HTTP URL without query or fragment"
        )
    if ready_url != HTTP_READY_URL:
        fail("collector config.ready_url differs from the fixed bridge readiness URL")
    amd_smi = nonempty_string(document["amd_smi"], "collector config.amd_smi")
    if amd_smi != AMD_SMI_BIN:
        fail("collector config.amd_smi differs from the fixed executable")
    return CollectorConfig(
        run_id=run_id,
        identities=identities,
        input_files=tuple(inputs),
        phase_artifacts=phase_artifacts,
        target=target,
        resource_body_template=template,
        negative_cases=tuple(negatives),
        restart_command=parse_command_array(
            document["restart_command"], "restart command"
        ),
        ready_url=ready_url,
        amd_smi=amd_smi,
        phase_artifact_identities=phase_artifact_identities,
    )


class AtomicFile:
    def __init__(self, final_path: Path, *, binary: bool = True):
        self.final_path = final_path
        self.incomplete_path = final_path.with_name(final_path.name + ".incomplete")
        if (
            final_path.exists()
            or final_path.is_symlink()
            or self.incomplete_path.exists()
            or self.incomplete_path.is_symlink()
        ):
            fail(f"output already exists for {final_path.name}")
        mode = "xb" if binary else "x"
        self.handle = self.incomplete_path.open(mode)
        self.closed = False
        self.synced = False

    def write(self, raw: bytes) -> None:
        if self.closed:
            fail("attempted to write a closed evidence file")
        self.handle.write(raw)
        self.synced = False

    def sync(self) -> None:
        if self.closed:
            fail("attempted to sync a closed evidence file")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.synced = True

    def commit(self) -> None:
        if self.closed:
            fail("attempted to commit a closed evidence file")
        if not self.synced:
            self.sync()
        self.handle.close()
        self.closed = True
        os.replace(self.incomplete_path, self.final_path)
        directory_fd = os.open(self.final_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def abort_close(self) -> None:
        if not self.closed:
            self.handle.flush()
            self.handle.close()
            self.closed = True


class AtomicJsonlWriter:
    def __init__(self, path: Path, guard: SecretGuard):
        self.file = AtomicFile(path)
        self.guard = guard
        self.line_count = 0

    def write_value(self, value: dict[str, Any]) -> None:
        raw = compact_json(value)
        self.guard.reject(raw, self.file.final_path.name)
        self.file.write(raw + b"\n")
        self.line_count += 1

    def write_raw_line(self, raw: bytes, label: str) -> None:
        if not raw or raw.endswith(b"\r") or b"\n" in raw or len(raw) > MAX_JSON_BYTES:
            fail(f"{label} is not one bounded LF-free JSON line")
        self.guard.reject(raw, label)
        strict_json_object(raw, label)
        self.file.write(raw + b"\n")
        self.line_count += 1

    def commit(self) -> None:
        if self.line_count == 0:
            fail(f"{self.file.final_path.name} is empty")
        self.file.commit()

    def abort_close(self) -> None:
        self.file.abort_close()


class SessionWriter:
    def __init__(self, path: Path, guard: SecretGuard):
        self.writer = AtomicJsonlWriter(path, guard)
        self.counts: collections.Counter[str] = collections.Counter()
        self.sequence = 0

    def append(
        self, record_type: str, phase: str, case_id: str | None, **fields: Any
    ) -> None:
        value = {
            "schema_version": SESSION_SCHEMA,
            "record_type": record_type,
            "sequence": self.sequence,
            "phase": phase,
            "case_id": case_id,
            **fields,
        }
        reject_forbidden_passed(value, "session record")
        self.writer.write_value(value)
        self.sequence += 1
        self.counts[record_type] += 1


def reject_forbidden_passed(value: Any, label: str) -> None:
    if type(value) is dict:
        if "passed" in value:
            fail(f"{label} contains forbidden producer field 'passed'")
        for child in value.values():
            reject_forbidden_passed(child, label)
    elif type(value) is list:
        for child in value:
            reject_forbidden_passed(child, label)


def validate_hook_fields(record_type: str, fields: Any) -> dict[str, Any]:
    if record_type == "browser_action":
        value = exact_keys(fields, BROWSER_ACTION_FIELDS, "restart hook browser_action")
        nonempty_string(value["browser_case"], "restart hook browser_case")
        integer(value["action_index"], "restart hook action_index")
        if value["action"] not in {
            "navigate",
            "select_model",
            "submit_chat",
            "wait_visible",
            "click_stop",
            "wait_failed",
            "wait_ready",
        }:
            fail("restart hook browser action differs")
        for key in ("selector",):
            if value[key] is not None:
                nonempty_string(value[key], f"restart hook {key}")
        if value["input_sha256"] is not None:
            sha256_value(value["input_sha256"], "restart hook input_sha256")
        started = integer(value["started_monotonic_ns"], "restart hook browser start")
        completed = integer(
            value["completed_monotonic_ns"], "restart hook browser completion"
        )
        if completed < started:
            fail("restart hook browser action timestamps regress")
        result = exact_keys(
            value["result"],
            {"visible", "enabled", "text_utf8_bytes", "text_sha256"},
            "restart hook browser result",
        )
        for key in ("visible", "enabled"):
            if result[key] is not None and type(result[key]) is not bool:
                fail(f"restart hook browser result {key} is not boolean or null")
        if result["text_utf8_bytes"] is None:
            if result["text_sha256"] is not None:
                fail("restart hook browser text fields are not null together")
        else:
            integer(result["text_utf8_bytes"], "restart hook browser text bytes")
            sha256_value(result["text_sha256"], "restart hook browser text SHA-256")
        screenshot = value["screenshot_file"]
        screenshot_sha = value["screenshot_sha256"]
        if screenshot is None:
            if screenshot_sha is not None:
                fail("restart hook screenshot fields are not null together")
        else:
            if screenshot not in {
                "browser/openwebui-stop-before.png",
                "browser/post-header-failure.png",
            }:
                fail("restart hook screenshot path differs")
            sha256_value(screenshot_sha, "restart hook screenshot SHA-256")
        return value
    if record_type == "fault_injection":
        value = exact_keys(
            fields, FAULT_INJECTION_FIELDS, "restart hook fault_injection"
        )
        if (
            value["injection"] != "post_header_worker_kill"
            or value["signal"] != "SIGKILL"
        ):
            fail("restart hook fault injection identity differs")
        integer(value["target_pid"], "restart hook target PID", minimum=1)
        integer(
            value["target_starttime_ticks"],
            "restart hook target starttime",
            minimum=1,
        )
        nonempty_string(value["command"], "restart hook fault command")
        started = integer(value["started_monotonic_ns"], "restart hook fault start")
        completed = integer(
            value["completed_monotonic_ns"], "restart hook fault completion"
        )
        if completed < started:
            fail("restart hook fault timestamps regress")
        return value
    fail("restart hook record type is unsupported")


@dataclasses.dataclass(frozen=True)
class ProcessIdentity:
    control_group: str
    gateway_pid: int
    gateway_starttime_ticks: int
    worker_pid: int
    worker_starttime_ticks: int
    n_restarts: int


@dataclasses.dataclass(frozen=True)
class LifecycleProbe:
    observed_monotonic_ns: int
    service_active: bool
    ready_http_status: int
    identity: ProcessIdentity


@dataclasses.dataclass(frozen=True)
class MetricCapture:
    raw: bytes
    captured_monotonic_ns: int


@dataclasses.dataclass(frozen=True)
class ResourceCapture:
    sample_monotonic_ns: int
    systemd: dict[str, Any]
    host: dict[str, Any]
    gateway: dict[str, Any]
    worker: dict[str, Any]
    gpu: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class HttpPlan:
    phase: str
    case_id: str
    request_index: int
    request_key: str
    target: str
    body: bytes
    expected_status: int
    expect_release: bool
    expected_error_code: str | None = None


@dataclasses.dataclass(frozen=True)
class HttpObservation:
    status: int
    completion_id: str | None
    outcome: str


class Runtime(Protocol):
    def now_ns(self) -> int: ...

    def wait_until(self, deadline_ns: int) -> None: ...

    def start(self) -> None: ...

    def close(self) -> None: ...

    def boot_id(self) -> str: ...

    def lifecycle_probe(self) -> LifecycleProbe: ...

    def run_http(
        self, plan: HttpPlan, emit: Callable[[str, dict[str, Any]], None]
    ) -> HttpObservation: ...

    def poll_journal(self) -> list[bytes]: ...

    def wait_for_journal(self, deadline_ns: int) -> None: ...

    def capture_metric(self, segment: str, boundary: str) -> MetricCapture: ...

    def capture_resource(self) -> ResourceCapture: ...

    def restart_hook(self) -> Iterable[dict[str, Any]]: ...

    def git_identity(self) -> tuple[str, str]: ...

    def resource_header(self) -> dict[str, Any]: ...


@dataclasses.dataclass
class JournalState:
    boot_id: str
    raw_writer: AtomicJsonlWriter
    session: SessionWriter
    last_cursor: str | None = None
    last_monotonic_usec: int = -1
    last_lifecycle_ns: int = -1
    cursors: set[str] = dataclasses.field(default_factory=set)
    observer_correlator: ObserverJournalCorrelator | None = None

    def consume(
        self,
        raw_lines: Iterable[bytes],
        phase: str,
        case_id: str,
        *,
        expected_gateway_pids: frozenset[int],
        gateway_pid_deadlines_ns: dict[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        if not expected_gateway_pids or any(pid <= 0 for pid in expected_gateway_pids):
            fail("journal consumption lacks a valid gateway PID epoch")
        lifecycle: list[dict[str, Any]] = []
        for raw in raw_lines:
            record = strict_json_object(raw, "journal record")
            for field in (
                "__CURSOR",
                "__MONOTONIC_TIMESTAMP",
                "_BOOT_ID",
                "_PID",
                "_SYSTEMD_UNIT",
                "PRIORITY",
                "MESSAGE",
            ):
                if field not in record:
                    fail(f"journal record lacks {field}")
            cursor = nonempty_string(record["__CURSOR"], "journal cursor")
            if cursor in self.cursors:
                fail("journal cursor is duplicated")
            self.cursors.add(cursor)
            monotonic_text = nonempty_string(
                record["__MONOTONIC_TIMESTAMP"], "journal monotonic"
            )
            pid_text = nonempty_string(record["_PID"], "journal PID")
            if not monotonic_text.isdecimal() or not pid_text.isdecimal():
                fail("journal numeric field is invalid")
            monotonic_usec = int(monotonic_text)
            if monotonic_usec < self.last_monotonic_usec:
                fail("journal monotonic timestamps regressed")
            if (
                record["_BOOT_ID"] != self.boot_id
                or record["_SYSTEMD_UNIT"] != SERVICE_UNIT
            ):
                fail("journal boot or systemd unit identity differs")
            message = record["MESSAGE"]
            if type(message) is not str:
                fail("journal MESSAGE is not a string")
            self.raw_writer.write_raw_line(raw, "service journal record")
            self.last_cursor = cursor
            self.last_monotonic_usec = monotonic_usec
            event = decode_lifecycle_message(message)
            if event is not None:
                if int(pid_text) not in expected_gateway_pids:
                    fail(
                        "gateway lifecycle journal PID differs from the active gateway"
                    )
                observed = event["observed_monotonic_ns"]
                deadline = (
                    None
                    if gateway_pid_deadlines_ns is None
                    else gateway_pid_deadlines_ns.get(int(pid_text))
                )
                if deadline is not None and observed > deadline:
                    fail("gateway lifecycle event exceeds its process-identity epoch")
                if observed < self.last_lifecycle_ns:
                    fail("gateway lifecycle observed timestamps regressed")
                self.last_lifecycle_ns = observed
                if self.observer_correlator is not None:
                    self.observer_correlator.correlate_journal_message(message, event)
                self.session.append(
                    "gateway_event",
                    phase,
                    case_id,
                    journal_cursor=cursor,
                    journal_monotonic_usec=monotonic_usec,
                    journal_pid=int(pid_text),
                    message=message,
                    message_sha256=sha256_bytes(message.encode("utf-8")),
                    event=event,
                )
                lifecycle.append(event)
        return lifecycle


def decode_lifecycle_message(message: str) -> dict[str, Any] | None:
    raw = message.encode("utf-8", errors="strict")
    if raw.startswith(b"{"):
        payload = raw
    elif raw.startswith(b"INFO:     {"):
        payload = raw[len(b"INFO:     ") :]
    else:
        return None
    value = strict_json_object(payload, "gateway lifecycle MESSAGE")
    if value.get("schema_version") != LIFECYCLE_SCHEMA:
        return None
    return validate_lifecycle_value(value)


def lifecycle_payload_from_message(message: str) -> bytes:
    raw = message.encode("utf-8", errors="strict")
    if raw.startswith(b"{"):
        return raw
    if raw.startswith(b"INFO:     {"):
        return raw[len(b"INFO:     ") :]
    fail("journal MESSAGE is not an exact lifecycle JSON payload")


def decode_lifecycle_payload(payload: bytes, label: str) -> dict[str, Any]:
    value = strict_json_object(payload, label)
    if value.get("schema_version") != LIFECYCLE_SCHEMA:
        fail(f"{label} schema_version differs")
    return validate_lifecycle_value(value)


def validate_lifecycle_value(value: dict[str, Any]) -> dict[str, Any]:
    event_name = value.get("event")
    if event_name not in LIFECYCLE_FIELDS:
        fail("gateway lifecycle event name is unknown")
    exact_keys(
        value,
        {"schema_version", "event", "observed_monotonic_ns"}
        | LIFECYCLE_FIELDS[event_name],
        "gateway lifecycle event",
    )
    integer(value["observed_monotonic_ns"], "gateway observed_monotonic_ns")
    request_id = value.get("request_id")
    completion_id = value.get("completion_id")
    if event_name == "worker_fatal" and request_id is None:
        if completion_id is not None or value["admit_to_fatal_ns"] is not None:
            fail("idle worker_fatal nullable fields differ")
    else:
        nonempty_string(request_id, "gateway request_id")
        nonempty_string(completion_id, "gateway completion_id")
    if event_name == "request_admitted":
        if type(value["stream"]) is not bool or value["stream"] is not True:
            fail("resource admission must be streaming")
        integer(value["prompt_tokens"], "gateway prompt_tokens", minimum=1)
        integer(
            value["max_completion_tokens"], "gateway max_completion_tokens", minimum=1
        )
    elif event_name == "request_started":
        if type(value["stream"]) is not bool:
            fail("gateway stream flag is not boolean")
        integer(value["prompt_tokens"], "gateway prompt_tokens", minimum=1)
        integer(value["admit_to_start_ns"], "gateway admit_to_start_ns")
    elif event_name == "request_progress":
        nonempty_string(value["phase"], "gateway progress phase")
        processed = integer(
            value["processed_prompt_tokens"], "processed prompt tokens", minimum=1
        )
        prompt = integer(value["prompt_tokens"], "prompt tokens", minimum=1)
        if processed > prompt:
            fail("gateway processed prompt tokens exceed prompt tokens")
    elif event_name == "request_first_token":
        if type(value["stream"]) is not bool or value["completion_tokens"] != 1:
            fail("gateway first-token event differs")
    elif event_name == "request_cancel_requested":
        if type(value["stream"]) is not bool:
            fail("gateway cancel stream flag differs")
        nonempty_string(value["reason"], "gateway cancel reason")
        integer(value["admit_to_cancel_ns"], "gateway admit_to_cancel_ns")
    elif event_name == "request_released":
        if type(value["stream"]) is not bool or value["outcome"] not in {
            "stop",
            "length",
            "cancelled",
        }:
            fail("gateway release stream or outcome differs")
        if value["outcome"] == "cancelled":
            nonempty_string(value["cancel_reason"], "gateway cancel_reason")
        elif value["cancel_reason"] is not None:
            fail("non-cancelled gateway release has cancel_reason")
        integer(value["prompt_tokens"], "gateway release prompt_tokens", minimum=1)
        integer(value["completion_tokens"], "gateway release completion_tokens")
        if value["reset_complete"] is not True:
            fail("gateway release lacks reset_complete=true")
        admit = integer(value["admit_to_start_ns"], "gateway release admit_to_start_ns")
        duration = integer(value["start_to_release_ns"], "gateway start_to_release_ns")
        total = integer(value["admit_to_release_ns"], "gateway admit_to_release_ns")
        if total != admit + duration:
            fail("gateway release duration arithmetic differs")
    elif event_name == "worker_fatal":
        nonempty_string(value["reason"], "gateway worker fatal reason")
        if request_id is not None:
            integer(value["admit_to_fatal_ns"], "gateway admit_to_fatal_ns")
    return value


class Phase1Collector:
    def __init__(
        self,
        config: CollectorConfig,
        output_dir: Path,
        repo_root: Path,
        expected_commit: str,
        expected_worker_sha256: str,
        guard: SecretGuard,
        runtime: Runtime,
    ):
        self.config = config
        self.output_dir = output_dir
        self.repo_root = repo_root
        self.expected_commit = git_commit(expected_commit, "expected commit")
        self.expected_worker_sha256 = sha256_value(
            expected_worker_sha256, "expected worker binary SHA-256"
        )
        self.guard = guard
        self.runtime = runtime
        self.session: SessionWriter | None = None
        self.resource: AtomicJsonlWriter | None = None
        self.journal: JournalState | None = None
        self.normal_identity: ProcessIdentity | None = None
        self.restart_identity: ProcessIdentity | None = None
        self.staged_artifacts: dict[str, FileSeal] = {}
        self.output_seals: dict[str, FileSeal] = {}
        self._completed = False

    def run(self) -> dict[str, Any]:
        self._prepare_output()
        try:
            self.runtime.start()
            self._start_raw_files()
            self._write_header()
            self._run_segment("normal", warmups=10, measured=100)
            self._run_restart_boundary()
            self._run_segment("restart", warmups=10, measured=20)
            self.runtime.close()
            self._finalize()
            self._completed = True
            return {
                "schema_version": "ullm.sq8.openwebui_release.collection.phase1.v1",
                "release_status": "incomplete",
                "phase1_collected": True,
                "run_id": self.config.run_id,
                "output_dir": str(self.output_dir),
            }
        finally:
            try:
                self.runtime.close()
            finally:
                if not self._completed:
                    self._abort_raw_files()

    def _prepare_output(self) -> None:
        candidate = (
            self.output_dir
            if self.output_dir.is_absolute()
            else Path.cwd() / self.output_dir
        )
        if ".." in candidate.parts:
            fail("output directory must not contain parent traversal")
        current = Path(candidate.anchor)
        for part in candidate.parts[1:-1]:
            current /= part
            if current.exists() and current.is_symlink():
                fail("output directory parent contains a symlink")
        if candidate.exists() or candidate.is_symlink():
            fail("output directory must be fresh")
        candidate.mkdir(parents=True, mode=0o750)
        (candidate / "browser").mkdir(mode=0o750)
        self.output_dir = candidate
        for relative in ("environment.json", "model-identity.json"):
            self._stage_phase_artifact(relative)

    def _stage_phase_artifact(self, relative: str) -> None:
        source = self.config.phase_artifacts[relative]
        destination = self.output_dir / PurePosixPath(relative)
        if destination.exists() or destination.is_symlink():
            fail(f"phase artifact destination already exists: {relative}")
        temporary = destination.with_name(destination.name + ".incomplete")
        source_fd = -1
        temporary_fd = -1
        try:
            source_fd = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            temporary_fd = os.open(
                temporary,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
            )
            source_before = os.fstat(source_fd)
            if not stat.S_ISREG(source_before.st_mode):
                fail(f"phase artifact {relative} is not a regular file")
            expected = (
                None
                if self.config.phase_artifact_identities is None
                else self.config.phase_artifact_identities.get(relative)
            )
            if (
                expected is not None
                and (source_before.st_dev, source_before.st_ino) != expected
            ):
                fail(f"phase artifact {relative} identity changed after config load")
            scanner = self.guard.scanner(f"phase artifact {relative}")
            digest = hashlib.sha256()
            total = 0
            while True:
                chunk = os.read(source_fd, COPY_CHUNK_BYTES)
                if not chunk:
                    break
                scanner.feed(chunk)
                digest.update(chunk)
                total += len(chunk)
                if total > MAX_PHASE_ARTIFACT_BYTES:
                    fail(f"phase artifact {relative} exceeds its size limit")
                write_all(temporary_fd, chunk, f"phase artifact {relative}")
            source_after = os.fstat(source_fd)
            if (
                stable_fd_identity(source_before) != stable_fd_identity(source_after)
                or total != source_before.st_size
            ):
                fail(f"phase artifact {relative} changed during staging")
            os.fsync(temporary_fd)
            os.lseek(temporary_fd, 0, os.SEEK_SET)
            if relative.endswith((".json", ".md")):
                raw = read_fd_bounded(
                    temporary_fd,
                    MAX_JSON_BYTES,
                    f"phase artifact {relative}",
                )
                if len(raw) != total:
                    fail(f"phase artifact {relative} exceeds its document limit")
                if relative.endswith(".json"):
                    strict_json_bytes(raw, f"phase artifact {relative}")
                else:
                    try:
                        raw.decode("utf-8", errors="strict")
                    except UnicodeError:
                        fail(f"phase artifact {relative} is not strict UTF-8")
            elif relative.endswith(".png"):
                if os.read(temporary_fd, 8) != b"\x89PNG\r\n\x1a\n":
                    fail(f"phase artifact {relative} is not a PNG")
            os.fchmod(temporary_fd, 0o640)
            os.replace(temporary, destination)
            destination_metadata = os.fstat(temporary_fd)
            destination_seal = FileSeal(
                stable_fd_identity(destination_metadata), digest.hexdigest()
            )
            if destination_seal.size != total:
                fail(f"phase artifact {relative} staged size differs")
            directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self.staged_artifacts[relative] = destination_seal
        except CollectorError:
            raise
        except OSError:
            fail(f"failed to stage phase artifact {relative}")
        finally:
            for descriptor in (temporary_fd, source_fd):
                if descriptor >= 0:
                    try:
                        os.close(descriptor)
                    except OSError:
                        fail(f"failed to close phase artifact {relative}")

    def _start_raw_files(self) -> None:
        self.session = SessionWriter(
            self.output_dir / "raw-session-results.jsonl", self.guard
        )
        self.resource = AtomicJsonlWriter(
            self.output_dir / "soak-resources.raw.jsonl", self.guard
        )
        self.journal = JournalState(
            boot_id=self.runtime.boot_id(),
            raw_writer=AtomicJsonlWriter(
                self.output_dir / "service-journal.raw.jsonl", self.guard
            ),
            session=self.session,
        )

    def _write_header(self) -> None:
        assert self.session is not None
        if (
            self.config.identities["worker_binary_sha256"]
            != self.expected_worker_sha256
        ):
            fail("collector config worker SHA differs from trusted CLI anchor")
        commit, _ = self.runtime.git_identity()
        if commit != self.expected_commit:
            fail("initial Git commit differs from trusted CLI anchor")
        input_records: list[dict[str, Any]] = []
        for item in self.config.input_files:
            byte_count, digest = inspect_input_file(item, self.guard)
            input_records.append(
                {
                    "path": item.path,
                    "bytes": byte_count,
                    "sha256": digest,
                }
            )
        identities = {
            "environment_file": "environment.json",
            "environment_sha256": self.staged_artifacts["environment.json"].sha256,
            "model_identity_file": "model-identity.json",
            "model_identity_sha256": self.staged_artifacts[
                "model-identity.json"
            ].sha256,
            **self.config.identities,
        }
        self.session.append(
            "header",
            "preflight",
            None,
            run_id=self.config.run_id,
            started_utc=utc_now(),
            clock="python.time.monotonic_ns",
            boot_id=self.runtime.boot_id(),
            identities=identities,
            input_files=input_records,
            schedule=SCHEDULE,
            thresholds=THRESHOLDS,
        )
        header = self.runtime.resource_header()
        exact_keys(header, RESOURCE_HEADER_FIELDS, "runtime resource header")
        reject_forbidden_passed(header, "runtime resource header")
        assert self.resource is not None
        self.resource.write_value(header)

    def _run_segment(self, segment: str, *, warmups: int, measured: int) -> None:
        phase = "resource_normal" if segment == "normal" else "resource_restart"
        probe = self.runtime.lifecycle_probe()
        self._append_probe(phase, f"{segment}-segment-start", probe)
        if not probe.service_active or probe.ready_http_status != 200:
            fail(f"{segment} segment service is not ready")
        if segment == "normal":
            self.normal_identity = probe.identity
        else:
            self.restart_identity = probe.identity
            self._validate_restart_identity()

        self._capture_metric(segment, "before")
        last_release: dict[str, Any] | None = None
        for index in range(1, warmups + 1):
            case_id = f"{segment}-warmup-{index:02d}"
            last_release = self._run_positive_request(
                segment,
                phase,
                case_id,
                request_index=index,
                measured=False,
            )
        if last_release is None:
            fail("resource segment lacks warmup releases")
        baseline_start = max(
            self.runtime.now_ns(), last_release["observed_monotonic_ns"]
        )
        self._sample_point(segment, None, None, baseline_start)

        negatives = {item.after_request: item for item in self.config.negative_cases}
        for index in range(1, measured + 1):
            case_id = f"{segment}-measured-{index:03d}"
            release = self._run_positive_request(
                segment,
                phase,
                case_id,
                request_index=index,
                measured=True,
            )
            settle_start = max(self.runtime.now_ns(), release["observed_monotonic_ns"])
            self._sample_point(segment, index, release, settle_start)
            if segment == "normal" and index in negatives:
                self._run_negative_request(phase, negatives[index])
        self._capture_metric(segment, "after")

    def _resource_body(self, segment: str, index: int, measured: bool) -> bytes:
        body = {
            "model": self.config.resource_body_template["model"],
            "messages": self.config.resource_body_template["messages"],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 2,
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
        }
        if (
            segment == "normal"
            and measured
            and index in SCHEDULE["sampled_normal_indices"]
        ):
            body["temperature"] = 0.6
            body["top_p"] = 0.95
            body["seed"] = index
        raw = compact_json(body)
        if len(raw) > MAX_HTTP_BODY_BYTES:
            fail("resource request body exceeds the HTTP client limit")
        self.guard.reject(raw, "resource request body")
        return raw

    def _run_positive_request(
        self,
        segment: str,
        phase: str,
        case_id: str,
        *,
        request_index: int,
        measured: bool,
    ) -> dict[str, Any]:
        key = f"p8f-{case_id}"
        plan = HttpPlan(
            phase=phase,
            case_id=case_id,
            request_index=request_index,
            request_key=key,
            target=self.config.target,
            body=self._resource_body(segment, request_index, measured),
            expected_status=200,
            expect_release=True,
        )
        observation = self.runtime.run_http(
            plan, lambda event, fields: self._append_http(plan, event, fields)
        )
        if (
            observation.status != 200
            or observation.outcome != "eof"
            or observation.completion_id is None
        ):
            fail("resource HTTP request did not complete as a successful SSE response")
        release = self._wait_for_release(plan, observation.completion_id)
        if (
            release["outcome"] != "length"
            or release["completion_tokens"] != 2
            or release["reset_complete"] is not True
        ):
            fail("resource request release differs from length/two/reset-complete")
        return release

    def _run_negative_request(self, phase: str, negative: NegativeCase) -> None:
        case_id = f"negative-after-{negative.after_request:03d}-{negative.name}"
        plan = HttpPlan(
            phase=phase,
            case_id=case_id,
            request_index=negative.after_request,
            request_key=f"p8f-{case_id}",
            target=self.config.target,
            body=negative.body,
            expected_status=negative.expected_status,
            expect_release=False,
            expected_error_code=(
                "invalid_request_error"
                if negative.name == "malformed_json"
                else "context_length_exceeded"
            ),
        )
        self.guard.reject(plan.body, "negative request body")
        observation = self.runtime.run_http(
            plan, lambda event, fields: self._append_http(plan, event, fields)
        )
        if (
            observation.status != negative.expected_status
            or observation.outcome != "eof"
        ):
            fail("negative request HTTP outcome differs")
        deadline = self.runtime.now_ns() + NEGATIVE_QUIET_NS
        while True:
            events = self._consume_journal(phase, case_id)
            if events:
                fail("negative request produced a gateway lifecycle admission/event")
            if self.runtime.now_ns() >= deadline:
                break
            self.runtime.wait_for_journal(
                min(deadline, self.runtime.now_ns() + JOURNAL_POLL_NS)
            )

    def _append_http(self, plan: HttpPlan, event: str, fields: dict[str, Any]) -> None:
        assert self.session is not None
        record_fields = dict(fields)
        if event == "http_request":
            record_fields = {"request_index": plan.request_index, **record_fields}
        self.session.append(event, plan.phase, plan.case_id, **record_fields)

    def _wait_for_release(self, plan: HttpPlan, completion_id: str) -> dict[str, Any]:
        deadline = self.runtime.now_ns() + RELEASE_TIMEOUT_NS
        admitted_request_id: str | None = None
        started = False
        first_token = False
        last_progress = 0
        while True:
            release_result: dict[str, Any] | None = None
            for event in self._consume_journal(plan.phase, plan.case_id):
                if release_result is not None:
                    fail("gateway lifecycle event appears after resource release")
                name = event["event"]
                event_completion = event.get("completion_id")
                if event_completion != completion_id:
                    fail(
                        "journal lifecycle completion ID differs from the active HTTP request"
                    )
                if name == "request_admitted":
                    if (
                        admitted_request_id is not None
                        or event["max_completion_tokens"] != 2
                    ):
                        fail(
                            "resource request admission is duplicated or has the wrong limit"
                        )
                    admitted_request_id = event["request_id"]
                elif (
                    admitted_request_id is None
                    or event["request_id"] != admitted_request_id
                ):
                    fail("resource lifecycle event precedes or differs from admission")
                if name == "request_started":
                    if started:
                        fail("resource request has duplicate start events")
                    started = True
                elif name == "request_progress":
                    if (
                        not started
                        or first_token
                        or event["processed_prompt_tokens"] <= last_progress
                    ):
                        fail("resource progress event order differs")
                    last_progress = event["processed_prompt_tokens"]
                elif name == "request_first_token":
                    if not started or first_token:
                        fail("resource first-token event order differs")
                    first_token = True
                elif name in {"request_cancel_requested", "worker_fatal"}:
                    fail("resource request was cancelled or fatally terminated")
                elif name == "request_released":
                    if (
                        not started
                        or not first_token
                        or event["reset_complete"] is not True
                    ):
                        fail(
                            "resource release precedes start or lacks reset acknowledgement"
                        )
                    if release_result is not None:
                        fail("resource request has duplicate release events")
                    release_result = event
            if release_result is not None:
                return release_result
            if self.runtime.now_ns() >= deadline:
                fail(
                    "timed out waiting for journal request_released(reset_complete=true)"
                )
            self.runtime.wait_for_journal(
                min(deadline, self.runtime.now_ns() + JOURNAL_POLL_NS)
            )

    def _consume_journal(self, phase: str, case_id: str) -> list[dict[str, Any]]:
        assert self.journal is not None
        if phase in {
            "preflight",
            "api_contract",
            "openwebui",
            "cancellation",
            "resource_normal",
            "post_header_failure",
        }:
            identity = self.normal_identity
        elif phase in {"resource_restart", "latency", "final"}:
            identity = self.restart_identity
        else:
            fail("journal phase has no gateway identity mapping")
        if identity is None:
            fail("journal phase lacks an active gateway identity")
        return self.journal.consume(
            self.runtime.poll_journal(),
            phase,
            case_id,
            expected_gateway_pids=frozenset({identity.gateway_pid}),
        )

    def _consume_restart_boundary_journal(
        self, case_id: str, normal_epoch_end_ns: int
    ) -> list[dict[str, Any]]:
        assert self.journal is not None
        if self.normal_identity is None or self.restart_identity is None:
            fail("restart-boundary journal lacks both gateway identities")
        return self.journal.consume(
            self.runtime.poll_journal(),
            "post_header_failure",
            case_id,
            expected_gateway_pids=frozenset(
                {
                    self.normal_identity.gateway_pid,
                    self.restart_identity.gateway_pid,
                }
            ),
            gateway_pid_deadlines_ns={
                self.normal_identity.gateway_pid: normal_epoch_end_ns
            },
        )

    def _sample_point(
        self,
        segment: str,
        request_index: int | None,
        release: dict[str, Any] | None,
        settle_start: int,
    ) -> None:
        assert self.resource is not None
        prior_sample: int | None = None
        expected_identity = (
            self.normal_identity if segment == "normal" else self.restart_identity
        )
        if expected_identity is None:
            fail("resource segment identity is unavailable")
        for sample_index in range(5):
            deadline = (
                settle_start + IDLE_SETTLE_NS
                if sample_index == 0
                else int(prior_sample) + SAMPLE_INTERVAL_NS
            )
            self.runtime.wait_until(deadline)
            capture = self.runtime.capture_resource()
            self._validate_capture_identity(capture, expected_identity)
            if capture.sample_monotonic_ns < deadline:
                fail("resource sample was captured before its scheduled boundary")
            record = {
                "schema_version": RESOURCE_SCHEMA,
                "record_type": "resource_sample",
                "segment": segment,
                "phase": "baseline" if release is None else "post_release",
                "request_index": request_index,
                "request_id": None if release is None else release["request_id"],
                "release_outcome": None if release is None else release["outcome"],
                "release_observed_monotonic_ns": None
                if release is None
                else release["observed_monotonic_ns"],
                "reset_complete": None
                if release is None
                else release["reset_complete"],
                "idle_settle_started_monotonic_ns": settle_start,
                "sample_index": sample_index,
                "sample_monotonic_ns": capture.sample_monotonic_ns,
                "systemd": capture.systemd,
                "host": capture.host,
                "gateway": capture.gateway,
                "worker": capture.worker,
                "gpu": capture.gpu,
            }
            reject_forbidden_passed(record, "resource sample")
            self.resource.write_value(record)
            prior_sample = capture.sample_monotonic_ns

    def _capture_metric(self, segment: str, boundary: str) -> None:
        capture = self.runtime.capture_metric(segment, boundary)
        strict_json_bytes(capture.raw, f"amd-smi metric {segment} {boundary}")
        self.guard.reject(capture.raw, "AMD SMI metric output")
        relative = f"amd-smi-metric-{segment}-{boundary}.json"
        atomic = AtomicFile(self.output_dir / relative)
        atomic.write(capture.raw)
        atomic.commit()
        assert self.resource is not None
        self.resource.write_value(
            {
                "schema_version": RESOURCE_SCHEMA,
                "record_type": "gpu_metric",
                "segment": segment,
                "boundary": boundary,
                "captured_monotonic_ns": capture.captured_monotonic_ns,
                "gpu_index": GPU_INDEX,
                "raw_output_file": relative,
                "raw_output_sha256": sha256_bytes(capture.raw),
            }
        )

    def _append_probe(self, phase: str, probe_name: str, probe: LifecycleProbe) -> None:
        assert self.session is not None
        identity = probe.identity
        self.session.append(
            "lifecycle_probe",
            phase,
            probe_name,
            probe=probe_name,
            observed_monotonic_ns=probe.observed_monotonic_ns,
            service_active=probe.service_active,
            ready_http_status=probe.ready_http_status,
            control_group=identity.control_group,
            gateway_pid=identity.gateway_pid,
            gateway_starttime_ticks=identity.gateway_starttime_ticks,
            worker_pid=identity.worker_pid,
            worker_starttime_ticks=identity.worker_starttime_ticks,
            n_restarts=identity.n_restarts,
        )

    def _run_restart_boundary(self) -> None:
        hook_records = self.runtime.restart_hook()
        for record in hook_records:
            exact_keys(record, {"schema_version", "record"}, "restart hook output")
            if record["schema_version"] != HOOK_SCHEMA:
                fail("restart hook schema_version differs")
            payload = exact_keys(
                record["record"],
                {"record_type", "phase", "case_id", "fields"},
                "restart hook record",
            )
            if payload["record_type"] not in {"browser_action", "fault_injection"}:
                fail("restart hook may emit only browser_action or fault_injection")
            if payload["phase"] != "post_header_failure":
                fail("restart hook record phase differs")
            case_id = nonempty_string(payload["case_id"], "restart hook case_id")
            fields = validate_hook_fields(payload["record_type"], payload["fields"])
            if payload["record_type"] == "fault_injection":
                if self.normal_identity is None or (
                    fields["target_pid"],
                    fields["target_starttime_ticks"],
                ) != (
                    self.normal_identity.worker_pid,
                    self.normal_identity.worker_starttime_ticks,
                ):
                    fail(
                        "restart hook fault target differs from the live worker identity"
                    )
            assert self.session is not None
            self.session.append(
                payload["record_type"], payload["phase"], case_id, **fields
            )
        deadline = self.runtime.now_ns() + RESTART_TIMEOUT_NS
        while True:
            probe = self.runtime.lifecycle_probe()
            if probe.service_active and probe.ready_http_status == 200:
                self.restart_identity = probe.identity
                self._append_probe(
                    "post_header_failure", "post-header-restart-ready", probe
                )
                self._validate_restart_identity()
                break
            if self.runtime.now_ns() >= deadline:
                fail("planned restart did not recover readiness before the deadline")
            self.runtime.wait_until(
                min(deadline, self.runtime.now_ns() + JOURNAL_POLL_NS)
            )
        self._consume_restart_boundary_journal(
            "post-header-failure", probe.observed_monotonic_ns
        )

    def _validate_restart_identity(self) -> None:
        if self.normal_identity is None or self.restart_identity is None:
            return
        normal = self.normal_identity
        restart = self.restart_identity
        if normal.control_group != restart.control_group:
            fail("systemd ControlGroup changed across the planned restart")
        if (normal.gateway_pid, normal.gateway_starttime_ticks) == (
            restart.gateway_pid,
            restart.gateway_starttime_ticks,
        ) or (normal.worker_pid, normal.worker_starttime_ticks) == (
            restart.worker_pid,
            restart.worker_starttime_ticks,
        ):
            fail(
                "gateway and worker identities must both change across the planned restart"
            )
        if restart.n_restarts != normal.n_restarts + 1:
            fail("systemd restart count did not increase exactly once")

    @staticmethod
    def _validate_capture_identity(
        capture: ResourceCapture, expected: ProcessIdentity
    ) -> None:
        systemd_value = capture.systemd
        if (
            systemd_value.get("control_group_before") != expected.control_group
            or systemd_value.get("control_group_after") != expected.control_group
            or systemd_value.get("main_pid_before") != expected.gateway_pid
            or systemd_value.get("main_pid_after") != expected.gateway_pid
        ):
            fail("resource sample systemd identity changed")
        gateway = capture.gateway
        worker = capture.worker
        if (
            gateway.get("pid") != expected.gateway_pid
            or gateway.get("starttime_ticks_before") != expected.gateway_starttime_ticks
            or gateway.get("starttime_ticks_after") != expected.gateway_starttime_ticks
            or worker.get("pid") != expected.worker_pid
            or worker.get("starttime_ticks_before") != expected.worker_starttime_ticks
            or worker.get("starttime_ticks_after") != expected.worker_starttime_ticks
        ):
            fail("resource sample process identity changed")

    def _finalize(self) -> None:
        assert (
            self.session is not None
            and self.resource is not None
            and self.journal is not None
        )
        for relative in sorted(
            PHASE_ARTIFACT_PATHS - {"environment.json", "model-identity.json"},
            key=lambda item: item.encode("utf-8"),
        ):
            self._stage_phase_artifact(relative)
        final_commit, final_status = self.runtime.git_identity()
        if final_commit != self.expected_commit:
            fail("Git commit changed during collection")
        self.guard.reject(final_status.encode("utf-8"), "final Git status")

        extra = self._consume_journal("final", "final-journal-drain")
        if extra:
            fail("unexpected request lifecycle appeared during final journal drain")
        final_probe = self.runtime.lifecycle_probe()
        if (
            not final_probe.service_active
            or final_probe.ready_http_status != 200
            or self.restart_identity is None
            or final_probe.identity != self.restart_identity
        ):
            fail("final service readiness or restart-segment identity differs")
        self._append_probe("final", "final-service-ready", final_probe)
        if self._consume_journal("final", "final-journal-drain-after-probe"):
            fail(
                "unexpected request lifecycle appeared after the final readiness probe"
            )
        if self.journal.last_cursor is None:
            fail("service journal contains no bounded run record")
        counts = dict(self.session.counts)
        counts["run_end"] = counts.get("run_end", 0) + 1
        self.session.append(
            "run_end",
            "final",
            None,
            completed_utc=utc_now(),
            completed_monotonic_ns=self.runtime.now_ns(),
            final_git_commit=final_commit,
            final_git_status_raw=final_status,
            final_git_status_sha256=sha256_bytes(final_status.encode("utf-8")),
            record_counts=counts,
            final_journal_cursor=self.journal.last_cursor,
        )
        self.resource.commit()
        self.journal.raw_writer.commit()
        self.session.writer.commit()
        self._write_matrix()
        checksum = self._prepare_sha256sums()
        published = False
        try:
            self._verify_post_seal_service_state()
            checksum.commit()
            published = True
        finally:
            if not published:
                checksum.abort_close()

    def _verify_post_seal_service_state(self) -> None:
        if self.runtime.poll_journal():
            fail("service journal changed during post-seal final drain")
        probe = self.runtime.lifecycle_probe()
        if (
            not probe.service_active
            or probe.ready_http_status != 200
            or self.restart_identity is None
            or probe.identity != self.restart_identity
        ):
            fail("post-seal service readiness or identity differs")
        if self.runtime.poll_journal():
            fail("service journal changed after the post-seal final probe")

    def _write_matrix(self) -> None:
        entries: list[dict[str, Any]] = []
        for relative in sorted(EXPECTED_ROLES, key=lambda item: item.encode("utf-8")):
            seal = inspect_sealed_file(
                self.output_dir / PurePosixPath(relative),
                f"matrix input {relative}",
                self.guard,
                expected=self.staged_artifacts.get(relative),
            )
            self.output_seals[relative] = seal
            entries.append(
                {
                    "role": EXPECTED_ROLES[relative],
                    "path": relative,
                    "bytes": seal.size,
                    "sha256": seal.sha256,
                }
            )
        value = {
            "schema_version": MATRIX_SCHEMA,
            "run_id": self.config.run_id,
            "files": entries,
            "schedule": SCHEDULE,
            "thresholds": THRESHOLDS,
        }
        self.output_seals["summary.md"] = inspect_sealed_file(
            self.output_dir / "summary.md",
            "summary.md",
            self.guard,
            expected=self.staged_artifacts["summary.md"],
        )
        self._write_atomic_document("release-matrix.json", compact_json(value))
        self.output_seals["release-matrix.json"] = inspect_sealed_file(
            self.output_dir / "release-matrix.json",
            "release matrix",
            self.guard,
        )

    def _prepare_sha256sums(self) -> AtomicFile:
        lines = []
        for relative in sorted(
            BUNDLE_FILES - {"SHA256SUMS"}, key=lambda item: item.encode("utf-8")
        ):
            expected = self.output_seals.get(relative)
            if expected is None:
                fail(f"checksum input {relative} lacks its prior seal")
            seal = inspect_sealed_file(
                self.output_dir / PurePosixPath(relative),
                f"checksum input {relative}",
                self.guard,
                expected=expected,
            )
            lines.append(f"{seal.sha256}  {relative}\n")
        checksum = AtomicFile(self.output_dir / "SHA256SUMS")
        checksum.write("".join(lines).encode("ascii"))
        checksum.sync()
        return checksum

    def _write_atomic_document(self, relative: str, raw: bytes) -> None:
        self.guard.reject(raw, relative)
        atomic = AtomicFile(self.output_dir / relative)
        atomic.write(raw)
        atomic.commit()

    def _abort_raw_files(self) -> None:
        if self.resource is not None:
            self.resource.abort_close()
        if self.journal is not None:
            self.journal.raw_writer.abort_close()
        if self.session is not None:
            self.session.writer.abort_close()


def terminate_process_group(
    process: subprocess.Popen[Any], timeout_seconds: float = 2.0
) -> None:
    if process.pid <= 0:
        fail("child process has an invalid process-group identity")
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + timeout_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        fail("child process group remained after SIGKILL")


def run_bounded_command(
    arguments: Sequence[str],
    label: str,
    *,
    cwd: Path | None = None,
    timeout_seconds: float = COMMAND_TIMEOUT_SECONDS,
    maximum_stdout: int = MAX_JSON_BYTES,
) -> bytes:
    if not arguments:
        fail(f"{label} command is empty")
    with (
        tempfile.TemporaryFile() as stdout_file,
        tempfile.TemporaryFile() as stderr_file,
    ):
        try:
            process = subprocess.Popen(
                list(arguments),
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                terminate_process_group(process)
                fail(f"{label} timed out")
            stdout_file.seek(0)
            raw = stdout_file.read(maximum_stdout + 1)
            stderr_file.seek(0)
            stderr_file.read(MAX_DIAGNOSTIC_BYTES)
        except OSError:
            fail(f"failed to execute {label}")
    if len(raw) > maximum_stdout:
        fail(f"{label} stdout exceeds its size limit")
    if code != 0:
        fail(f"{label} exited {code}; stderr is intentionally not retained")
    return raw


def stream_bounded_jsonl_command(
    arguments: Sequence[str],
    label: str,
    *,
    cwd: Path | None = None,
    timeout_seconds: float = COMMAND_TIMEOUT_SECONDS,
    maximum_records: int,
) -> Iterable[bytes]:
    if not arguments or maximum_records <= 0:
        fail(f"{label} command or record limit is invalid")
    process: subprocess.Popen[bytes] | None = None
    with tempfile.TemporaryFile() as stderr_file:
        try:
            process = subprocess.Popen(
                list(arguments),
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                bufsize=0,
                start_new_session=True,
            )
            if process.stdout is None:
                fail(f"{label} stdout pipe is unavailable")
            descriptor = process.stdout.fileno()
            deadline_ns = time.monotonic_ns() + int(timeout_seconds * 1_000_000_000)
            buffer = bytearray()
            record_count = 0
            while True:
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    terminate_process_group(process)
                    fail(f"{label} timed out")
                ready, _, _ = select.select(
                    [descriptor], [], [], remaining_ns / 1_000_000_000
                )
                if not ready:
                    terminate_process_group(process)
                    fail(f"{label} timed out")
                chunk = os.read(descriptor, 64 * 1024)
                if chunk:
                    buffer.extend(chunk)
                    while True:
                        newline = buffer.find(b"\n")
                        if newline < 0:
                            break
                        raw = bytes(buffer[:newline])
                        del buffer[: newline + 1]
                        if (
                            not raw
                            or raw.endswith(b"\r")
                            or len(raw) > MAX_HOOK_RECORD_BYTES
                        ):
                            fail(f"{label} contains an invalid bounded line")
                        record_count += 1
                        if record_count > maximum_records:
                            fail(f"{label} exceeds its record-count limit")
                        yield raw
                    if len(buffer) > MAX_HOOK_RECORD_BYTES:
                        fail(f"{label} contains an oversized unterminated line")
                    continue
                if buffer:
                    fail(f"{label} is not LF-terminated JSONL")
                try:
                    exit_code = process.wait(
                        timeout=max(0.001, (deadline_ns - time.monotonic_ns()) / 1e9)
                    )
                except subprocess.TimeoutExpired:
                    terminate_process_group(process)
                    fail(f"{label} timed out after closing stdout")
                if exit_code != 0:
                    fail(
                        f"{label} exited {exit_code}; stderr is intentionally not retained"
                    )
                return
        except CollectorError:
            raise
        except OSError:
            fail(f"failed to execute or stream {label}")
        finally:
            if process is not None:
                if process.poll() is None:
                    terminate_process_group(process)
                if process.stdout is not None:
                    try:
                        process.stdout.close()
                    except OSError:
                        pass


class BoundedLineReader:
    def __init__(self, descriptor: int):
        self.descriptor = descriptor
        self.buffer = bytearray()

    def read(self, deadline_ns: int, label: str) -> bytes:
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                raw = bytes(self.buffer[:newline])
                del self.buffer[: newline + 1]
                if raw.endswith(b"\r"):
                    fail(f"{label} uses CRLF")
                return raw
            if len(self.buffer) > MAX_JSON_BYTES:
                fail(f"{label} exceeds its size limit")
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                fail(f"{label} timed out")
            ready, _, _ = select.select(
                [self.descriptor], [], [], remaining_ns / 1_000_000_000
            )
            if not ready:
                fail(f"{label} timed out")
            try:
                chunk = os.read(self.descriptor, 64 * 1024)
            except OSError:
                fail(f"failed to read {label}")
            if not chunk:
                fail(f"{label} ended before an LF-terminated record")
            self.buffer.extend(chunk)


class HttpClientProcess:
    def __init__(self, command: Sequence[str], guard: SecretGuard):
        self.command = tuple(command)
        self.guard = guard
        self.process: subprocess.Popen[bytes] | None = None
        self.reader: BoundedLineReader | None = None
        self.stderr_file: BinaryIO | None = None
        self.last_response_end_ns = -1

    def start(self) -> None:
        if self.process is not None:
            fail("HTTP client process is already started")
        self.stderr_file = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(
                list(self.command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr_file,
                bufsize=0,
                start_new_session=True,
            )
        except OSError:
            fail("failed to start the HTTP evidence client")
        if self.process.stdin is None or self.process.stdout is None:
            fail("HTTP evidence client pipes are unavailable")
        self.reader = BoundedLineReader(self.process.stdout.fileno())
        event = self._read_event(time.monotonic_ns() + 30_000_000_000)
        exact_keys(
            event,
            {"schema_version", "event", "observed_monotonic_ns"},
            "HTTP client ready event",
        )
        if event["schema_version"] != HTTP_EVENT_SCHEMA or event["event"] != "ready":
            fail("HTTP client did not emit the required ready event")
        integer(event["observed_monotonic_ns"], "HTTP client ready timestamp")

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        shutdown_error: CollectorError | None = None
        if process.poll() is not None:
            shutdown_error = CollectorError(
                "HTTP client exited before the shutdown handshake"
            )
        else:
            try:
                self._write_command(
                    {"schema_version": HTTP_COMMAND_SCHEMA, "command": "shutdown"}
                )
                event = self._read_event(time.monotonic_ns() + 5_000_000_000)
                exact_keys(
                    event,
                    {"schema_version", "event", "observed_monotonic_ns"},
                    "HTTP client shutdown event",
                )
                if (
                    event["schema_version"] != HTTP_EVENT_SCHEMA
                    or event["event"] != "shutdown_complete"
                ):
                    fail("HTTP client shutdown acknowledgement differs")
                integer(
                    event["observed_monotonic_ns"], "HTTP client shutdown timestamp"
                )
                exit_code = process.wait(timeout=5.0)
                if exit_code != 0:
                    fail(
                        "HTTP client exited nonzero after its shutdown acknowledgement"
                    )
                if self.reader is None or process.stdout is None:
                    fail("HTTP client shutdown stream state is unavailable")
                if self.reader.buffer or os.read(process.stdout.fileno(), 1):
                    fail("HTTP client emitted data after its shutdown acknowledgement")
            except CollectorError as error:
                shutdown_error = error
                terminate_process_group(process)
            except (OSError, subprocess.TimeoutExpired):
                shutdown_error = CollectorError("HTTP client shutdown failed")
                terminate_process_group(process)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        if self.stderr_file is not None:
            self.stderr_file.close()
        self.process = None
        if shutdown_error is not None:
            raise shutdown_error

    def request(
        self,
        plan: HttpPlan,
        emit: Callable[[str, dict[str, Any]], None],
    ) -> HttpObservation:
        self.guard.reject(plan.body, "HTTP request body")
        command = {
            "schema_version": HTTP_COMMAND_SCHEMA,
            "command": "request",
            "request_key": plan.request_key,
            "method": "POST",
            "target": plan.target,
            "body_base64": base64.b64encode(plan.body).decode("ascii"),
            "authorization_mode": "valid_bearer",
            "close_on_first_nonempty_sse_content": False,
        }
        self._write_command(command)
        deadline = time.monotonic_ns() + int(
            HTTP_REQUEST_TIMEOUT_SECONDS * 1_000_000_000
        )
        saw_request = False
        saw_start = False
        status: int | None = None
        next_chunk = 0
        response_body = bytearray()
        response_digest = hashlib.sha256()
        outcome: str | None = None
        last_observed_ns = -1
        while outcome is None:
            event = self._read_event(deadline)
            if event.get("schema_version") != HTTP_EVENT_SCHEMA:
                fail("HTTP client event schema_version differs")
            event_name = event.get("event")
            fields = {
                key: value
                for key, value in event.items()
                if key not in {"schema_version", "event"}
            }
            if event_name == "http_request":
                if saw_request:
                    fail("HTTP client duplicated its request event")
                connect, sent = self._validate_http_request_event(fields, plan)
                if connect < self.last_response_end_ns:
                    fail("HTTP request begins before the prior response ended")
                last_observed_ns = sent
                saw_request = True
            elif event_name == "http_response_start":
                if not saw_request or saw_start:
                    fail("HTTP response start order differs")
                exact_keys(
                    fields,
                    {"request_key", "status", "headers", "observed_monotonic_ns"},
                    "HTTP response start",
                )
                if fields["request_key"] != plan.request_key:
                    fail("HTTP response start request key differs")
                status = integer(fields["status"], "HTTP status", minimum=100)
                if status > 599:
                    fail("HTTP status exceeds 599")
                if type(fields["headers"]) is not list:
                    fail("HTTP response headers are not an array")
                for pair in fields["headers"]:
                    if (
                        type(pair) is not list
                        or len(pair) != 2
                        or any(type(item) is not str for item in pair)
                    ):
                        fail("HTTP response header is not a two-string array")
                content_types = [
                    value.lower()
                    for name, value in fields["headers"]
                    if name.lower() == "content-type"
                ]
                expected_media = (
                    "text/event-stream" if status == 200 else "application/json"
                )
                media_types = [
                    value.split(";", 1)[0].strip() for value in content_types
                ]
                if len(media_types) != 1 or media_types[0] != expected_media:
                    fail("HTTP response Content-Type differs")
                observed = integer(
                    fields["observed_monotonic_ns"], "HTTP response start timestamp"
                )
                if observed < last_observed_ns:
                    fail(
                        "HTTP response start timestamp precedes the request send boundary"
                    )
                last_observed_ns = observed
                saw_start = True
            elif event_name == "http_body_chunk":
                if not saw_start:
                    fail("HTTP body chunk precedes response start")
                exact_keys(
                    fields,
                    {
                        "request_key",
                        "chunk_index",
                        "body_base64",
                        "body_sha256",
                        "body_bytes",
                        "observed_monotonic_ns",
                    },
                    "HTTP body chunk",
                )
                if (
                    fields["request_key"] != plan.request_key
                    or fields["chunk_index"] != next_chunk
                ):
                    fail("HTTP body chunk correlation differs")
                chunk = decode_bound_bytes(fields, "HTTP body chunk")
                if len(response_body) + len(chunk) > MAX_JSON_BYTES:
                    fail("complete HTTP response exceeds its evidence limit")
                self.guard.reject(chunk, "HTTP response body")
                response_body.extend(chunk)
                response_digest.update(chunk)
                observed = integer(
                    fields["observed_monotonic_ns"], "HTTP body chunk timestamp"
                )
                if observed < last_observed_ns:
                    fail("HTTP body chunk timestamps regressed")
                last_observed_ns = observed
                next_chunk += 1
            elif event_name == "http_response_end":
                if not saw_start:
                    fail("HTTP response end precedes response start")
                exact_keys(
                    fields,
                    {
                        "request_key",
                        "outcome",
                        "error",
                        "body_bytes",
                        "body_sha256",
                        "observed_monotonic_ns",
                    },
                    "HTTP response end",
                )
                if fields["request_key"] != plan.request_key:
                    fail("HTTP response end request key differs")
                outcome = fields["outcome"]
                if outcome not in {"eof", "client_closed", "timeout", "error"}:
                    fail("HTTP response end outcome differs")
                if (outcome in {"eof", "client_closed"}) != (fields["error"] is None):
                    fail("HTTP response end error field differs from outcome")
                if fields["error"] is not None:
                    nonempty_string(
                        fields["error"], "HTTP response diagnostic", maximum=1024
                    )
                if (
                    fields["body_bytes"] != len(response_body)
                    or fields["body_sha256"] != response_digest.hexdigest()
                ):
                    fail("HTTP response end body aggregate differs")
                observed = integer(
                    fields["observed_monotonic_ns"], "HTTP response end timestamp"
                )
                if observed < last_observed_ns:
                    fail("HTTP response end timestamp precedes response evidence")
                last_observed_ns = observed
                self.last_response_end_ns = observed
            elif event_name == "command_error":
                fail("HTTP client reported a command error")
            else:
                fail("HTTP client emitted an unexpected event")
            if event_name in {
                "http_request",
                "http_response_start",
                "http_body_chunk",
                "http_response_end",
            }:
                emit(event_name, fields)
        if status is None:
            fail("HTTP response lacks a status")
        if status != plan.expected_status:
            fail("HTTP response status differs from the request plan")
        completion_id = None
        if status == 200:
            completion_id = validate_resource_sse(bytes(response_body))
        else:
            if plan.expected_error_code is None:
                fail("negative HTTP response lacks its expected semantic error class")
            validate_error_response(
                bytes(response_body),
                plan.expected_error_code,
                "negative HTTP error body",
            )
        return HttpObservation(
            status=status, completion_id=completion_id, outcome=outcome
        )

    def _write_command(self, value: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None or process.poll() is not None:
            fail("HTTP client is not running")
        raw = compact_json(value)
        self.guard.reject(raw, "HTTP client command")
        try:
            process.stdin.write(raw + b"\n")
            process.stdin.flush()
        except OSError:
            fail("failed to write an HTTP client command")

    def _read_event(self, deadline_ns: int) -> dict[str, Any]:
        if self.reader is None:
            fail("HTTP client line reader is unavailable")
        raw = self.reader.read(deadline_ns, "HTTP client event")
        self.guard.reject(raw, "HTTP client event")
        return strict_json_object(raw, "HTTP client event")

    @staticmethod
    def _validate_http_request_event(
        fields: dict[str, Any], plan: HttpPlan
    ) -> tuple[int, int]:
        exact_keys(
            fields,
            {
                "request_key",
                "method",
                "target",
                "headers",
                "body_base64",
                "body_sha256",
                "body_bytes",
                "connect_completed_monotonic_ns",
                "write_started_monotonic_ns",
                "last_body_byte_sent_monotonic_ns",
            },
            "HTTP request event",
        )
        if (
            fields["request_key"] != plan.request_key
            or fields["method"] != "POST"
            or fields["target"] != plan.target
        ):
            fail("HTTP request event identity differs")
        headers = exact_keys(
            fields["headers"],
            {"content_type", "content_length", "authorization_mode"},
            "HTTP request headers",
        )
        if headers != {
            "content_type": "application/json",
            "content_length": len(plan.body),
            "authorization_mode": "valid_bearer",
        }:
            fail("HTTP request evidence headers differ")
        raw = decode_bound_bytes(fields, "HTTP request event")
        if raw != plan.body:
            fail("HTTP request evidence body differs from the plan")
        connect = integer(
            fields["connect_completed_monotonic_ns"], "HTTP connect timestamp"
        )
        started = integer(fields["write_started_monotonic_ns"], "HTTP write timestamp")
        sent = integer(
            fields["last_body_byte_sent_monotonic_ns"], "HTTP last-body timestamp"
        )
        if not connect <= started <= sent:
            fail("HTTP request timestamp order differs")
        return connect, sent


def decode_bound_bytes(fields: dict[str, Any], label: str) -> bytes:
    encoded = fields.get("body_base64")
    if type(encoded) is not str:
        fail(f"{label} body_base64 is not a string")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        fail(f"{label} body_base64 is invalid")
    if base64.b64encode(raw).decode("ascii") != encoded:
        fail(f"{label} body_base64 is not canonical")
    if fields.get("body_bytes") != len(raw) or fields.get(
        "body_sha256"
    ) != sha256_bytes(raw):
        fail(f"{label} byte count or hash differs")
    return raw


def validate_resource_sse(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError:
        fail("resource SSE is not strict UTF-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    data_events: list[str] = []
    data_lines: list[str] = []
    for line in normalized.split("\n"):
        if line == "":
            if data_lines:
                data_events.append("\n".join(data_lines))
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            data_lines.append(value)
    if data_lines:
        data_events.append("\n".join(data_lines))
    if not data_events or data_events[-1] != "[DONE]":
        fail("resource SSE lacks a terminal [DONE] event")
    completion_ids: set[str] = set()
    content_count = 0
    usage_count: int | None = None
    for index, payload in enumerate(data_events[:-1]):
        value = strict_json_object(
            payload.encode("utf-8"), f"resource SSE data {index}"
        )
        if "id" in value:
            completion_ids.add(
                nonempty_string(value["id"], "resource SSE completion id")
            )
        choices = value.get("choices")
        if type(choices) is list and choices:
            first = choices[0]
            if type(first) is dict and type(first.get("delta")) is dict:
                content = first["delta"].get("content")
                if type(content) is str and content:
                    content_count += 1
        usage = value.get("usage")
        if type(usage) is dict and "completion_tokens" in usage:
            count = integer(
                usage["completion_tokens"], "resource SSE usage completion_tokens"
            )
            if usage_count is not None:
                fail("resource SSE duplicates usage")
            usage_count = count
    if len(completion_ids) != 1 or content_count < 1 or usage_count != 2:
        fail("resource SSE completion identity/content/usage differs")
    return next(iter(completion_ids))


def validate_error_response(raw: bytes, expected_code: str, label: str) -> None:
    value = strict_json_object(raw, label, maximum=MAX_HTTP_BODY_BYTES)
    exact_keys(value, {"error"}, label)
    error = exact_keys(
        value["error"],
        {"message", "type", "param", "code"},
        f"{label}.error",
    )
    nonempty_string(error["message"], f"{label}.error.message")
    if error["type"] != "invalid_request_error" or error["code"] != expected_code:
        fail(f"{label} semantic error class differs")
    if expected_code == "context_length_exceeded":
        if error["param"] != "messages":
            fail(f"{label} context overflow param differs")
    elif error["param"] is not None:
        nonempty_string(error["param"], f"{label}.error.param")


class JournalSource:
    def __init__(self, boot_id: str):
        self.boot_id = boot_id
        self.cursor: str | None = None
        self.reader: Any | None = None

    def start(self) -> None:
        try:
            from systemd import journal

            reader = journal.Reader()
            reader.add_match(_SYSTEMD_UNIT=SERVICE_UNIT)
            reader.this_boot()
            reader.seek_tail()
            entry = reader.get_previous()
        except (ImportError, OSError, ValueError):
            fail("failed to initialize the direct sd-journal reader")
        if not entry:
            fail("service journal has no initial cursor")
        cursor = nonempty_string(entry.get("__CURSOR"), "initial journal cursor")
        try:
            reader.seek_cursor(cursor)
            positioned = reader.get_next()
        except (OSError, ValueError):
            fail("failed to position the direct sd-journal reader")
        if not positioned or positioned.get("__CURSOR") != cursor:
            fail("direct sd-journal cursor positioning differs")
        self.cursor = cursor
        self.reader = reader

    def poll(self) -> list[bytes]:
        if self.cursor is None or self.reader is None:
            fail("journal source is not initialized")
        lines: list[bytes] = []
        total_bytes = 0
        while True:
            try:
                entry = self.reader.get_next()
            except (OSError, ValueError):
                fail("direct sd-journal read failed")
            if not entry:
                break
            record = sd_journal_json_record(entry, self.boot_id)
            line = compact_json(record)
            total_bytes += len(line) + 1
            if total_bytes > MAX_JSON_BYTES:
                fail("direct sd-journal poll exceeds its bounded output size")
            lines.append(line)
            self.cursor = record["__CURSOR"]
        return lines

    def wait_until(self, deadline_ns: int) -> None:
        if self.reader is None:
            fail("journal source is not initialized")
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            return
        timeout_usec = max(1, (remaining_ns + 999) // 1000)
        try:
            self.reader.wait(timeout_usec)
        except (OSError, ValueError):
            fail("direct sd-journal wait failed")


def sd_journal_json_record(entry: dict[str, Any], boot_id: str) -> dict[str, str]:
    for field in (
        "__CURSOR",
        "__MONOTONIC_TIMESTAMP",
        "_BOOT_ID",
        "_PID",
        "_SYSTEMD_UNIT",
        "PRIORITY",
        "MESSAGE",
    ):
        if field not in entry:
            fail(f"direct sd-journal entry lacks {field}")
    monotonic = entry["__MONOTONIC_TIMESTAMP"]
    try:
        delta = monotonic.timestamp
        monotonic_usec = (
            delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds
        )
    except (AttributeError, TypeError, OverflowError):
        fail("direct sd-journal monotonic timestamp is invalid")
    boot_value = entry["_BOOT_ID"]
    if isinstance(boot_value, uuid.UUID):
        boot_text = boot_value.hex
    else:
        boot_text = str(boot_value).replace("-", "")
    if boot_text != boot_id:
        fail("direct sd-journal boot ID differs")
    message = entry["MESSAGE"]
    if type(message) is not str:
        fail("direct sd-journal MESSAGE is not text")
    return {
        "__CURSOR": nonempty_string(entry["__CURSOR"], "direct journal cursor"),
        "__MONOTONIC_TIMESTAMP": str(monotonic_usec),
        "_BOOT_ID": boot_text,
        "_PID": str(integer(entry["_PID"], "direct journal PID", minimum=1)),
        "_SYSTEMD_UNIT": nonempty_string(entry["_SYSTEMD_UNIT"], "direct journal unit"),
        "PRIORITY": str(integer(entry["PRIORITY"], "direct journal priority")),
        "MESSAGE": message,
    }


def parse_proc_stat(raw: str, expected_pid: int) -> tuple[int, int]:
    prefix = f"{expected_pid} ("
    if not raw.startswith(prefix):
        fail("/proc stat PID prefix differs")
    candidates = list(re.finditer(r"\) ([A-Za-z]) ", raw))
    for match in reversed(candidates):
        fields = [match.group(1), *raw[match.end() :].strip().split()]
        if len(fields) < 20:
            continue
        try:
            ppid = int(fields[1], 10)
            starttime = int(fields[19], 10)
        except ValueError:
            continue
        if ppid >= 1 and starttime >= 1:
            return ppid, starttime
    fail("/proc stat lacks a valid process identity")


def read_bounded_file(path: Path, label: str, maximum: int = 1024 * 1024) -> bytes:
    try:
        with path.open("rb") as handle:
            raw = handle.read(maximum + 1)
    except OSError:
        fail(f"failed to read {label}")
    if len(raw) > maximum:
        fail(f"{label} exceeds its size limit")
    return raw


def decimal_file(path: Path, label: str) -> int:
    raw = read_bounded_file(path, label, maximum=128)
    try:
        text = raw.decode("ascii", errors="strict").strip()
    except UnicodeError:
        fail(f"{label} is not ASCII")
    if not text.isdecimal():
        fail(f"{label} is not an unsigned decimal integer")
    return int(text)


def control_group_parts(value: str) -> tuple[str, ...]:
    text = nonempty_string(value, "systemd ControlGroup")
    pure = PurePosixPath(text)
    if (
        not pure.is_absolute()
        or text.startswith("//")
        or any(part in {"", ".", ".."} for part in pure.parts[1:])
        or len(pure.parts) < 2
    ):
        fail("systemd ControlGroup is not a safe absolute cgroup path")
    return tuple(pure.parts[1:])


def read_cgroup_memory_current(cgroup_root: Path, control_group: str) -> int:
    parts = control_group_parts(control_group)
    root_fd = -1
    current_fd = -1
    memory_fd = -1
    try:
        root_fd = os.open(
            cgroup_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        current_fd = root_fd
        for part in parts:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=current_fd,
            )
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd
        memory_fd = os.open(
            "memory.current",
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=current_fd,
        )
        metadata = os.fstat(memory_fd)
        if not stat.S_ISREG(metadata.st_mode):
            fail("service cgroup memory.current is not a regular file")
        raw = os.read(memory_fd, 129)
        if len(raw) > 128:
            fail("service cgroup memory.current exceeds its size limit")
    except CollectorError:
        raise
    except OSError:
        fail("failed to open service cgroup memory.current safely")
    finally:
        closed: set[int] = set()
        for descriptor in (memory_fd, current_fd, root_fd):
            if descriptor >= 0 and descriptor not in closed:
                try:
                    os.close(descriptor)
                except OSError:
                    fail("failed to close a cgroup descriptor")
                closed.add(descriptor)
    try:
        text = raw.decode("ascii", errors="strict").strip()
    except UnicodeError:
        fail("service cgroup memory.current is not ASCII")
    if not text.isdecimal():
        fail("service cgroup memory.current is not an unsigned decimal integer")
    return int(text)


def process_identity(proc_root: Path, pid: int) -> tuple[int, int]:
    raw = read_bounded_file(proc_root / str(pid) / "stat", f"/proc/{pid}/stat")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError:
        fail("/proc stat is not UTF-8")
    return parse_proc_stat(text, pid)


def hash_live_process_executable(
    proc_root: Path,
    pid: int,
    expected_starttime_ticks: int,
) -> str:
    _, start_before = process_identity(proc_root, pid)
    if start_before != expected_starttime_ticks:
        fail("live executable process starttime differs before hashing")
    path = proc_root / str(pid) / "exe"
    descriptor = -1
    verify_descriptor = -1
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail("live process executable is not a regular file")
        while chunk := os.read(descriptor, COPY_CHUNK_BYTES):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if stable_fd_identity(before) != stable_fd_identity(after):
            fail("live process executable changed while hashing")
        verify_descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        verify = os.fstat(verify_descriptor)
        if (verify.st_dev, verify.st_ino) != (before.st_dev, before.st_ino):
            fail("live process executable changed after hashing")
    except CollectorError:
        raise
    except OSError:
        fail("failed to hash the live process executable")
    finally:
        for current in (verify_descriptor, descriptor):
            if current >= 0:
                try:
                    os.close(current)
                except OSError:
                    fail("failed to close the live process executable")
    _, start_after = process_identity(proc_root, pid)
    if start_after != expected_starttime_ticks:
        fail("live executable process starttime differs after hashing")
    return digest.hexdigest()


def process_record(proc_root: Path, pid: int) -> dict[str, Any]:
    ppid, start_before = process_identity(proc_root, pid)
    status_raw = read_bounded_file(
        proc_root / str(pid) / "status", f"/proc/{pid}/status"
    )
    try:
        status = status_raw.decode("utf-8", errors="strict")
    except UnicodeError:
        fail("/proc status is not UTF-8")
    vmrss: int | None = None
    threads: int | None = None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            match = re.fullmatch(r"VmRSS:\s+([0-9]+) kB", line)
            if match is None or vmrss is not None:
                fail("/proc status VmRSS is malformed or duplicated")
            vmrss = int(match.group(1))
        elif line.startswith("Threads:"):
            match = re.fullmatch(r"Threads:\s+([0-9]+)", line)
            if match is None or threads is not None:
                fail("/proc status Threads is malformed or duplicated")
            threads = int(match.group(1))
    if vmrss is None or threads is None or threads < 1:
        fail("/proc status lacks VmRSS or Threads")
    try:
        exe = os.readlink(proc_root / str(pid) / "exe")
        fd_names = os.listdir(proc_root / str(pid) / "fd")
    except OSError:
        fail("failed to capture process executable or descriptors")
    if not exe.startswith("/") or exe.endswith(" (deleted)"):
        fail("process executable identity is invalid")
    child_raw = read_bounded_file(
        proc_root / str(pid) / "task" / str(pid) / "children",
        f"/proc/{pid}/children",
    )
    try:
        child_text = child_raw.decode("ascii", errors="strict").strip()
        children = [] if not child_text else [int(item) for item in child_text.split()]
    except (UnicodeError, ValueError):
        fail("process children list is invalid")
    if children != sorted(set(children)) or any(item <= 0 for item in children):
        fail("process children list is not ascending and unique")
    ppid_after, start_after = process_identity(proc_root, pid)
    if ppid_after != ppid or start_after != start_before:
        fail("process identity changed during capture")
    return {
        "pid": pid,
        "ppid": ppid,
        "exe": exe,
        "starttime_ticks_before": start_before,
        "starttime_ticks_after": start_after,
        "vmrss_kb": vmrss,
        "vmrss_bytes": vmrss * 1024,
        "threads": threads,
        "fd_count": len(fd_names),
        "children": children,
    }


class KfdSnapshotUnstable(RuntimeError):
    pass


def enumerate_kfd_pids(root_fd: int) -> tuple[str, ...]:
    try:
        names = os.listdir(root_fd)
    except OSError:
        fail("failed to enumerate KFD process entries")
    values: list[tuple[int, str]] = []
    for name in names:
        if not name.isascii() or not name.isdecimal():
            continue
        pid = int(name)
        if pid <= 0 or name != str(pid):
            fail("KFD process PID is not canonical positive decimal")
        values.append((pid, name))
    values.sort()
    return tuple(name for _, name in values)


def open_kfd_pid_dirs(
    root_fd: int,
    names: tuple[str, ...],
    expected_worker_pid: int,
) -> dict[str, tuple[int, int, int]]:
    opened: dict[str, tuple[int, int, int]] = {}
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        for name in names:
            pid = int(name)
            try:
                descriptor = os.open(name, flags, dir_fd=root_fd)
            except OSError as error:
                if error.errno == errno.ENOENT and pid != expected_worker_pid:
                    raise KfdSnapshotUnstable from error
                if error.errno == errno.ENOENT:
                    fail("required worker disappeared from the KFD process set")
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    fail("KFD process entry is not a real directory")
                fail("failed to open a KFD process directory")
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(descriptor)
                fail("KFD process entry is not a directory")
            opened[name] = (descriptor, metadata.st_dev, metadata.st_ino)
        return opened
    except BaseException:
        close_kfd_dirs(opened)
        raise


def close_kfd_dirs(opened: dict[str, tuple[int, int, int]]) -> None:
    while opened:
        _, (descriptor, _, _) = opened.popitem()
        try:
            os.close(descriptor)
        except OSError:
            fail("failed to close a KFD process directory")


def read_kfd_vram_fd(process_fd: int, pid: int, expected_worker_pid: int) -> int:
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                f"vram_{KFD_GPU_ID}",
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=process_fd,
            )
        except OSError as error:
            if error.errno == errno.ENOENT and pid != expected_worker_pid:
                raise KfdSnapshotUnstable from error
            if error.errno == errno.ENOENT:
                fail("required worker KFD VRAM counter is missing")
            if error.errno == errno.ELOOP:
                fail("KFD VRAM counter must not be a symlink")
            fail("failed to open a KFD VRAM counter")
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            fail("KFD VRAM counter is not a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(4096, 4097 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > 4096:
                fail("KFD VRAM counter exceeds its size limit")
        raw = b"".join(chunks)
    except OSError:
        fail("failed to inspect a KFD VRAM counter")
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                fail("failed to close a KFD VRAM counter")
    try:
        text = raw.decode("ascii", errors="strict").strip()
    except UnicodeError:
        fail("KFD VRAM counter is not ASCII")
    if not text.isdecimal():
        fail("KFD VRAM counter is not an unsigned decimal integer")
    return int(text)


def capture_stable_kfd_vram(kfd_root: Path, expected_worker_pid: int) -> dict[int, int]:
    integer(expected_worker_pid, "expected KFD worker PID", minimum=1)
    try:
        root_fd = os.open(
            kfd_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError:
        fail("failed to open the KFD process root")
    deadline = time.monotonic_ns() + KFD_SNAPSHOT_TIMEOUT_NS
    try:
        while True:
            before: dict[str, tuple[int, int, int]] = {}
            after: dict[str, tuple[int, int, int]] = {}
            try:
                names_before = enumerate_kfd_pids(root_fd)
                if str(expected_worker_pid) not in names_before:
                    fail("required worker is absent from the KFD process set")
                before = open_kfd_pid_dirs(root_fd, names_before, expected_worker_pid)
                values = {
                    int(name): read_kfd_vram_fd(item[0], int(name), expected_worker_pid)
                    for name, item in before.items()
                }
                unrelated = sorted(
                    pid
                    for pid, value in values.items()
                    if pid != expected_worker_pid and value > 0
                )
                if unrelated:
                    fail("an unrelated process owns positive R9700 VRAM")
                names_after = enumerate_kfd_pids(root_fd)
                if str(expected_worker_pid) not in names_after:
                    fail("required worker left the KFD process set")
                after = open_kfd_pid_dirs(root_fd, names_after, expected_worker_pid)
                if names_before != names_after:
                    raise KfdSnapshotUnstable
                for name in names_before:
                    if before[name][1:] != after[name][1:]:
                        fail("KFD process directory identity changed during capture")
                return values
            except KfdSnapshotUnstable:
                if time.monotonic_ns() >= deadline:
                    fail("stable KFD process snapshot exceeded one second")
            finally:
                close_kfd_dirs(after)
                close_kfd_dirs(before)
            time.sleep(KFD_RETRY_SLEEP_SECONDS)
    finally:
        try:
            os.close(root_fd)
        except OSError:
            fail("failed to close the KFD process root")


def parse_key_value_lines(raw: bytes, label: str) -> dict[str, str]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError:
        fail(f"{label} is not UTF-8")
    result: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            fail(f"{label} contains a malformed or duplicate field")
        result[key] = value
    return result


class SystemRuntime:
    def __init__(
        self,
        config: CollectorConfig,
        repo_root: Path,
        guard: SecretGuard,
        snapshots: RuntimeSnapshots,
    ):
        self.config = config
        self.repo_root = repo_root
        self.guard = guard
        self.snapshots = snapshots
        self.proc_root = Path("/proc")
        self.kfd_root = Path("/sys/class/kfd/kfd/proc")
        self._boot_id = ""
        self.http = HttpClientProcess(
            build_http_client_command(config, snapshots), guard
        )
        self.journal: JournalSource | None = None

    def now_ns(self) -> int:
        return time.monotonic_ns()

    def wait_until(self, deadline_ns: int) -> None:
        while True:
            remaining = deadline_ns - time.monotonic_ns()
            if remaining <= 0:
                return
            time.sleep(remaining / 1_000_000_000)

    def start(self) -> None:
        self._validate_docker_identity()
        raw = read_bounded_file(
            Path("/proc/sys/kernel/random/boot_id"), "boot ID", maximum=128
        )
        try:
            self._boot_id = (
                raw.decode("ascii", errors="strict").strip().replace("-", "")
            )
        except UnicodeError:
            fail("boot ID is not ASCII")
        nonempty_string(self._boot_id, "boot ID", maximum=128)
        self.journal = JournalSource(self._boot_id)
        self.journal.start()
        self.http.start()
        self.snapshots.unlink_credential()

    def close(self) -> None:
        self.http.close()

    def boot_id(self) -> str:
        if not self._boot_id:
            fail("runtime boot ID is unavailable")
        current = read_bounded_file(
            Path("/proc/sys/kernel/random/boot_id"), "boot ID", maximum=128
        )
        try:
            value = current.decode("ascii", errors="strict").strip().replace("-", "")
        except UnicodeError:
            fail("boot ID is not ASCII")
        if value != self._boot_id:
            fail("boot ID changed during collection")
        return self._boot_id

    def lifecycle_probe(self) -> LifecycleProbe:
        state = self._systemd_state()
        active = state.get("ActiveState") == "active"
        control_group = state.get("ControlGroup", "")
        gateway_pid = int(state.get("MainPID", "0") or "0")
        n_restarts = int(state.get("NRestarts", "0") or "0")
        gateway_start = 0
        worker_pid = 0
        worker_start = 0
        if active:
            control_group_parts(control_group)
        if active and gateway_pid > 0:
            _, gateway_start = process_identity(self.proc_root, gateway_pid)
            gateway = process_record(self.proc_root, gateway_pid)
            candidates = []
            for child in gateway["children"]:
                try:
                    exe = os.readlink(self.proc_root / str(child) / "exe")
                except OSError:
                    continue
                if Path(exe).name == "ullm-sq8-worker":
                    candidates.append(child)
            if len(candidates) == 1:
                worker_pid = candidates[0]
                _, worker_start = process_identity(self.proc_root, worker_pid)
        ready_status = self._ready_status()
        if ready_status == 200 and (
            not active
            or not control_group.startswith("/")
            or gateway_pid <= 0
            or gateway_start <= 0
            or worker_pid <= 0
            or worker_start <= 0
        ):
            fail("ready service lacks a complete gateway/worker identity")
        if ready_status == 200:
            live_worker_sha = hash_live_process_executable(
                self.proc_root,
                worker_pid,
                worker_start,
            )
            if live_worker_sha != self.config.identities["worker_binary_sha256"]:
                fail("live worker executable differs from the trusted binary SHA-256")
        return LifecycleProbe(
            observed_monotonic_ns=time.monotonic_ns(),
            service_active=active,
            ready_http_status=ready_status,
            identity=ProcessIdentity(
                control_group=control_group,
                gateway_pid=gateway_pid,
                gateway_starttime_ticks=gateway_start,
                worker_pid=worker_pid,
                worker_starttime_ticks=worker_start,
                n_restarts=n_restarts,
            ),
        )

    def run_http(
        self, plan: HttpPlan, emit: Callable[[str, dict[str, Any]], None]
    ) -> HttpObservation:
        return self.http.request(plan, emit)

    def poll_journal(self) -> list[bytes]:
        if self.journal is None:
            fail("journal source is unavailable")
        return self.journal.poll()

    def wait_for_journal(self, deadline_ns: int) -> None:
        if self.journal is None:
            fail("journal source is unavailable")
        self.journal.wait_until(deadline_ns)

    def capture_metric(self, segment: str, boundary: str) -> MetricCapture:
        raw = run_bounded_command(
            [self.config.amd_smi, "metric", "--gpu", str(GPU_INDEX), "--json"],
            "amd-smi metric",
        )
        value = strict_json_bytes(raw, "amd-smi metric output")
        if type(value) not in {dict, list}:
            fail("amd-smi metric output root is not an object or array")
        return MetricCapture(raw=raw, captured_monotonic_ns=time.monotonic_ns())

    def capture_resource(self) -> ResourceCapture:
        sample_time = time.monotonic_ns()
        before = self._systemd_state()
        control_group = nonempty_string(
            before.get("ControlGroup"), "systemd ControlGroup"
        )
        gateway_pid = int(before.get("MainPID", "0"))
        if before.get("ActiveState") != "active" or gateway_pid <= 0:
            fail("service is not active during resource capture")
        gateway = process_record(self.proc_root, gateway_pid)
        candidates = []
        for child in gateway["children"]:
            try:
                exe = os.readlink(self.proc_root / str(child) / "exe")
            except OSError:
                continue
            if Path(exe).name == "ullm-sq8-worker":
                candidates.append(child)
        if len(candidates) != 1:
            fail("gateway does not have exactly one SQ8 worker child")
        worker_pid = candidates[0]
        worker = process_record(self.proc_root, worker_pid)
        process_raw = run_bounded_command(
            [
                self.config.amd_smi,
                "process",
                "--gpu",
                str(GPU_INDEX),
                "--general",
                "--json",
            ],
            "amd-smi process",
        )
        vram, process_count = parse_amd_process(process_raw, worker_pid)
        kfd_vram = capture_stable_kfd_vram(self.kfd_root, worker_pid)
        own_vram = kfd_vram.get(worker_pid)
        unrelated = sorted(
            pid for pid, value in kfd_vram.items() if pid != worker_pid and value > 0
        )
        if own_vram != vram or unrelated:
            fail("AMD SMI and isolated KFD VRAM ownership differ")
        memory = read_cgroup_memory_current(Path("/sys/fs/cgroup"), control_group)
        after = self._systemd_state()
        if (
            after.get("ControlGroup") != control_group
            or int(after.get("MainPID", "0")) != gateway_pid
        ):
            fail("systemd identity changed during resource capture")
        _, gateway_final = process_identity(self.proc_root, gateway_pid)
        _, worker_final = process_identity(self.proc_root, worker_pid)
        if (
            gateway_final != gateway["starttime_ticks_before"]
            or worker_final != worker["starttime_ticks_before"]
        ):
            fail("gateway or worker identity changed during resource capture")
        return ResourceCapture(
            sample_monotonic_ns=sample_time,
            systemd={
                "control_group_before": control_group,
                "control_group_after": control_group,
                "main_pid_before": gateway_pid,
                "main_pid_after": gateway_pid,
            },
            host={"memory_current_bytes": memory},
            gateway=gateway,
            worker=worker,
            gpu={
                "index": GPU_INDEX,
                "bdf": GPU_BDF,
                "uuid": GPU_UUID,
                "kfd_gpu_id": KFD_GPU_ID,
                "process_record_count": process_count,
                "worker_pid": worker_pid,
                "mem_usage": {"value": vram, "unit": "B"},
                "kfd_vram_bytes": own_vram,
                "unrelated_process_pids": unrelated,
            },
        )

    def restart_hook(self) -> Iterable[dict[str, Any]]:
        for line in stream_bounded_jsonl_command(
            self.config.restart_command,
            "planned post-header restart hook",
            cwd=self.repo_root,
            timeout_seconds=RESTART_TIMEOUT_NS / 1_000_000_000,
            maximum_records=MAX_HOOK_RECORDS,
        ):
            self.guard.reject(line, "restart hook output")
            yield strict_json_object(line, "restart hook record")

    def git_identity(self) -> tuple[str, str]:
        commit_raw = run_bounded_command(
            ["git", "rev-parse", "HEAD"],
            "Git commit",
            cwd=self.repo_root,
            maximum_stdout=128,
        )
        status_raw = run_bounded_command(
            ["git", "status", "--porcelain=v1"],
            "Git porcelain status",
            cwd=self.repo_root,
            maximum_stdout=4 * 1024 * 1024,
        )
        try:
            commit = commit_raw.decode("ascii", errors="strict").strip()
            status = status_raw.decode("utf-8", errors="strict")
        except UnicodeError:
            fail("Git identity output is not strict text")
        git_commit(commit, "Git commit")
        return commit, status

    def resource_header(self) -> dict[str, Any]:
        systemd_raw = run_bounded_command(["systemctl", "--version"], "systemd version")
        amd_version_raw = run_bounded_command(
            [self.config.amd_smi, "version"], "amd-smi version"
        )
        cgroup_raw = run_bounded_command(
            ["stat", "-fc", "%T", "/sys/fs/cgroup"], "cgroup fs type"
        )
        amd_list_raw = run_bounded_command(
            [self.config.amd_smi, "list", "--json"], "amd-smi list"
        )
        try:
            systemd_line = systemd_raw.decode("utf-8", errors="strict").splitlines()[0]
            amd_version = amd_version_raw.decode("utf-8", errors="strict").strip()
            cgroup = cgroup_raw.decode("ascii", errors="strict").strip()
        except (UnicodeError, IndexError):
            fail("resource tool identity output is invalid")
        if not systemd_line.startswith("systemd 255 ") or cgroup != "cgroup2fs":
            fail("systemd or cgroup identity differs")
        expected_versions = {
            "amd_smi_tool": "26.2.2+e1a6bc5663",
            "amd_smi_library": "26.2.2",
            "rocm": "7.2.1",
        }
        if any(value not in amd_version for value in expected_versions.values()):
            fail("AMD SMI or ROCm version differs")
        parse_amd_list(amd_list_raw)
        if not self.kfd_root.is_dir():
            fail("KFD process directory is unavailable")
        return {
            "schema_version": RESOURCE_SCHEMA,
            "record_type": "header",
            "service_unit": SERVICE_UNIT,
            "commands": COMMANDS,
            "tools": {
                "systemd_major": 255,
                "systemd_version_line": systemd_line,
                **expected_versions,
                "amd_smi_version_output": amd_version,
            },
            "probes": {
                "cgroup_fs_type": "cgroup2fs",
                "kfd_proc_present": True,
                "gpu_index": GPU_INDEX,
                "gpu_bdf": GPU_BDF,
                "gpu_uuid": GPU_UUID,
                "kfd_gpu_id": KFD_GPU_ID,
            },
            "schedule": RESOURCE_SCHEDULE,
        }

    def _systemd_state(self) -> dict[str, str]:
        raw = run_bounded_command(
            [
                "systemctl",
                "show",
                SERVICE_UNIT,
                "--property=ActiveState",
                "--property=ControlGroup",
                "--property=MainPID",
                "--property=NRestarts",
                "--no-pager",
            ],
            "systemd service identity",
        )
        fields = parse_key_value_lines(raw, "systemd service identity")
        if set(fields) != {"ActiveState", "ControlGroup", "MainPID", "NRestarts"}:
            fail("systemd service identity fields differ")
        if not fields["MainPID"].isdecimal() or not fields["NRestarts"].isdecimal():
            fail("systemd PID or restart count is invalid")
        return fields

    def _ready_status(self) -> int:
        script = (
            "import sys,urllib.request;"
            "r=urllib.request.urlopen(sys.argv[1],timeout=2);"
            "print(r.status)"
        )
        try:
            raw = run_bounded_command(
                [
                    DOCKER_BIN,
                    "run",
                    "--rm",
                    "--pull=never",
                    f"--network={HTTP_NETWORK_NAME}",
                    "--read-only",
                    "--cap-drop=ALL",
                    "--security-opt=no-new-privileges",
                    "--pids-limit=32",
                    "--memory=128m",
                    "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=8388608",
                    "--entrypoint=python3",
                    self.config.identities["openwebui"]["derived_image_id"],
                    "-c",
                    script,
                    HTTP_READY_URL,
                ],
                "Docker-network readiness probe",
                timeout_seconds=10.0,
                maximum_stdout=32,
            )
            text = raw.decode("ascii", errors="strict").strip()
            return int(text) if text.isdecimal() else 0
        except (CollectorError, UnicodeError):
            return 0

    def _validate_docker_identity(self) -> None:
        network_raw = run_bounded_command(
            [DOCKER_BIN, "network", "inspect", HTTP_NETWORK_NAME],
            "Docker network identity",
        )
        network_value = strict_json_bytes(network_raw, "Docker network identity")
        if (
            type(network_value) is not list
            or len(network_value) != 1
            or type(network_value[0]) is not dict
        ):
            fail("Docker network identity output differs")
        network = network_value[0]
        if network.get("Id") != self.config.identities["docker_network_id"]:
            fail("live Docker network ID differs from the recorded identity")
        ipam = network.get("IPAM")
        configurations = ipam.get("Config") if type(ipam) is dict else None
        if (
            type(configurations) is not list
            or len(configurations) != 1
            or type(configurations[0]) is not dict
        ):
            fail("Docker network IPAM identity differs")
        if (
            configurations[0].get("Subnet") != HTTP_NETWORK_SUBNET
            or configurations[0].get("Gateway") != HTTP_NETWORK_GATEWAY
        ):
            fail("Docker network subnet or gateway differs")
        bridge = Path("/sys/class/net") / f"br-{network['Id'][:12]}"
        if not bridge.is_dir():
            fail("Docker network bridge interface is unavailable")

        image_id = self.config.identities["openwebui"]["derived_image_id"]
        image_raw = run_bounded_command(
            [DOCKER_BIN, "image", "inspect", image_id],
            "HTTP client image identity",
        )
        image_value = strict_json_bytes(image_raw, "HTTP client image identity")
        if (
            type(image_value) is not list
            or len(image_value) != 1
            or type(image_value[0]) is not dict
            or image_value[0].get("Id") != image_id
        ):
            fail("HTTP client image content identity differs")


def parse_amd_process(raw: bytes, worker_pid: int) -> tuple[int, int]:
    value = strict_json_bytes(raw, "amd-smi process output")
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        fail("amd-smi process output must contain one GPU object")
    gpu = value[0]
    if gpu.get("gpu") != GPU_INDEX:
        fail("amd-smi process GPU index differs")
    processes = gpu.get("process_list")
    if (
        type(processes) is not list
        or len(processes) != 1
        or type(processes[0]) is not dict
    ):
        fail("amd-smi process must contain exactly one process")
    info = processes[0].get("process_info")
    if type(info) is not dict or info.get("pid") != worker_pid:
        fail("amd-smi process worker PID differs")
    memory = info.get("mem_usage")
    if type(memory) is not dict or memory.get("unit") != "B":
        fail("amd-smi process memory unit differs")
    return integer(memory.get("value"), "amd-smi process VRAM", minimum=1), len(
        processes
    )


def parse_amd_list(raw: bytes) -> None:
    value = strict_json_bytes(raw, "amd-smi list output")
    if type(value) is not list:
        fail("amd-smi list root must be an array")
    matches = [
        item
        for item in value
        if type(item) is dict
        and item.get("gpu") == GPU_INDEX
        and item.get("bdf") == GPU_BDF
        and item.get("uuid") == GPU_UUID
        and item.get("kfd_id") == KFD_GPU_ID
    ]
    if len(matches) != 1:
        fail("amd-smi list physical GPU identity differs")


def regular_directory(path: Path, label: str) -> Path:
    absolute = path if path.is_absolute() else Path.cwd() / path
    try:
        metadata = absolute.lstat()
    except OSError:
        fail(f"failed to stat {label}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        fail(f"{label} must be a regular directory")
    return absolute.resolve(strict=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-worker-binary-sha256", required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snapshots: RuntimeSnapshots | None = None
    output: bytes | None = None
    exit_code = 0
    try:
        repo_root = regular_directory(args.repo_root, "repository root")
        client_path = repo_root / HTTP_CLIENT_SOURCE_RELATIVE
        _, client_raw, _ = read_regular_snapshot(
            client_path,
            "HTTP client implementation",
            maximum=MAX_JSON_BYTES,
        )
        config = load_config(args.config, http_client_snapshot=client_raw)
        guard, credential_raw = SecretGuard.snapshot_from_file(args.api_key_file)
        for item in config.input_files:
            if item.snapshot is not None:
                guard.reject(item.snapshot, f"input snapshot {item.path}")
        snapshots = RuntimeSnapshots.create(client_raw, credential_raw)
        runtime = SystemRuntime(config, repo_root, guard, snapshots)
        collector = Phase1Collector(
            config,
            args.output_dir,
            repo_root,
            args.expected_commit,
            args.expected_worker_binary_sha256,
            guard,
            runtime,
        )
        result = collector.run()
        output = compact_json(result) + b"\n"
    except CollectorError as error:
        print(f"collection failed: {error}", file=sys.stderr)
        exit_code = 1
    finally:
        if snapshots is not None:
            try:
                snapshots.close()
            except CollectorError as error:
                print(f"collection cleanup failed: {error}", file=sys.stderr)
                exit_code = 1
    if exit_code == 0:
        assert output is not None
        sys.stdout.buffer.write(output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
