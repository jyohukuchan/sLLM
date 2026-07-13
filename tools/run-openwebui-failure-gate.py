#!/usr/bin/env python3
"""Inject one worker failure and prove OpenWebUI failure/recovery semantics."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pwd
import re
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn


GATE_SCHEMA = "ullm.openwebui.failure_gate.v1"
BROWSER_SCHEMA = "ullm.openwebui.failure_smoke.v1"
CONTROL_SCHEMA = "ullm.openwebui.failure_control.v1"
LIFECYCLE_SCHEMA = "ullm.gateway.lifecycle.v1"
BROWSER_CASE = "post_header_worker_failure"
MODEL_ID = os.environ.get("ULLM_MODEL_ID", "ullm-qwen3-14b-sq8")
MODEL_LABEL = os.environ.get("ULLM_MODEL_NAME", "uLLM Qwen3 14B SQ8")
BROWSER_SCRIPT_CONTAINER_PATH = "/usr/src/app/ullm-browser-failure-smoke.cjs"
BROWSER_CONTAINER_OUTPUT_DIR_NAME = "browser-output"
KILL_CONTROL_CONTAINER_PATH = "/run/control/worker-killed"
RECOVERY_CONTROL_CONTAINER_PATH = "/run/control/gateway-recovered"
SCREENSHOT_NAME = "post-header-failure.png"
BROWSER_SUMMARY_NAME = "openwebui-failure-summary.json"
MAX_JSON_LINE_BYTES = 1024 * 1024
MAX_JOURNAL_LINES = 4096
MAX_JOURNAL_BYTES = 64 * 1024 * 1024
MAX_BROWSER_LINES = 4
MAX_BROWSER_SOCKET_EVENTS = 2048
MAX_STDERR_BYTES = 4 * 1024 * 1024
COPY_CHUNK_BYTES = 64 * 1024
PROCESS_GRACE_SECONDS = 2.0
DEFAULT_TIMEOUT_SECONDS = 360
CONTENT_IMAGE_RE = re.compile(
    r"(?:(?:[A-Za-z0-9][A-Za-z0-9._/:+-]*)@)?sha256:([0-9a-f]{64})\Z"
)
SERVICE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@:-]{0,127}\.service\Z")
NETWORK_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
REASONING_RELEASE_FIELDS = {"reasoning_tokens", "forced_end_tokens"}

FAILURE_PROMPT = " ".join(
    (
        "Begin with FAIL_STREAM_MARKER.",
        "Then write the integers from 1 through 1000, one per line.",
        "Do not summarize and do not stop early.",
    )
)
RECOVERY_MARKER = "FAILURE_RECOVERY_OK"
RECOVERY_PROMPT = (
    "For this new turn, reply with exactly FAILURE_RECOVERY_OK and nothing else."
)
FINAL_ACTIONS = (
    "navigate",
    "select_model",
    "submit_chat",
    "wait_visible",
    "wait_failed",
    "wait_ready",
    "submit_chat",
    "wait_visible",
    "wait_ready",
)

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


class FailureGateError(RuntimeError):
    """Fail-closed error whose message contains no external values."""


class TransientServiceState(RuntimeError):
    """Expected while systemd is replacing the failed process."""


def fail(message: str) -> NoReturn:
    raise FailureGateError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compact_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError):
        fail("failed to encode bounded JSON")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("JSON object contains a duplicate key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    fail("JSON contains a non-finite constant")


def strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    if not raw or len(raw) > MAX_JSON_LINE_BYTES:
        fail(f"{label} size is invalid")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError):
        fail(f"{label} is not strict UTF-8 JSON")
    if not isinstance(value, dict):
        fail(f"{label} root is not an object")
    return value


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        fail(f"{label} fields differ")


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        fail(f"{label} is not a bounded integer")
    return value


def decimal_timestamp(value: Any, label: str) -> int:
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdecimal()
        or len(value) > 32
    ):
        fail(f"{label} is not a decimal timestamp")
    return int(value, 10)


def nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        fail(f"{label} is not a nonempty string")
    return value


def sha256_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} is not a SHA-256 value")
    return value


def read_regular_exact(path: Path, label: str, maximum: int) -> bytes:
    try:
        before = path.lstat()
    except OSError:
        fail(f"{label} is unavailable")
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        fail(f"{label} is not a regular non-symlink file")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                fail(f"{label} identity changed while opening")
            raw = handle.read(maximum + 1)
            after = os.fstat(handle.fileno())
    except OSError:
        fail(f"failed to read {label}")
    if len(raw) > maximum:
        fail(f"{label} exceeds its size bound")

    def identity(item: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if identity(opened) != identity(after) or opened.st_size != len(raw):
        fail(f"{label} changed while reading")
    return raw


def write_private_snapshot(path: Path, raw: bytes, label: str) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o400,
        )
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            handle.write(raw)
            os.fsync(handle.fileno())
    except OSError:
        fail(f"failed to create {label} snapshot")


class SecretGuard:
    def __init__(self, values: list[bytes]):
        self.values = tuple(value for value in values if len(value) >= 4)

    def extend(self, values: list[str]) -> SecretGuard:
        return SecretGuard([*self.values, *(value.encode("utf-8") for value in values)])

    def reject(self, raw: bytes, label: str) -> None:
        for value in self.values:
            if value in raw:
                fail(
                    f"{label} contains forbidden cleartext "
                    f"(value_sha256={hashlib.sha256(value).hexdigest()})"
                )

    def scan_file(self, path: Path, label: str) -> None:
        overlap = max((len(value) for value in self.values), default=1) - 1
        tail = b""
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(COPY_CHUNK_BYTES):
                    combined = tail + chunk
                    self.reject(combined, label)
                    tail = combined[-overlap:] if overlap else b""
        except OSError:
            fail(f"failed to scan {label}")


class AtomicLineWriter:
    def __init__(self, final_path: Path, *, maximum_bytes: int):
        self.final_path = final_path
        self.incomplete_path = final_path.with_name(final_path.name + ".incomplete")
        self.maximum_bytes = maximum_bytes
        self.bytes_written = 0
        self.lines_written = 0
        self.digest = hashlib.sha256()
        try:
            descriptor = os.open(
                self.incomplete_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600,
            )
            self.handle = os.fdopen(descriptor, "wb", buffering=0)
        except OSError:
            fail("failed to create an incomplete raw artifact")
        self.closed = False

    def write_line(self, raw: bytes) -> None:
        if self.closed or not raw or b"\n" in raw or b"\r" in raw:
            fail("raw artifact line is invalid")
        framed = raw + b"\n"
        if self.bytes_written + len(framed) > self.maximum_bytes:
            fail("raw artifact exceeds its byte bound")
        try:
            self.handle.write(framed)
        except OSError:
            fail("failed to stream a raw artifact")
        self.digest.update(framed)
        self.bytes_written += len(framed)
        self.lines_written += 1

    def commit(self) -> None:
        if self.closed:
            fail("raw artifact writer is already closed")
        try:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
            self.closed = True
            os.rename(self.incomplete_path, self.final_path)
        except OSError:
            fail("failed to commit a raw artifact")

    def abort(self) -> None:
        if self.closed:
            return
        try:
            self.handle.close()
        except OSError:
            pass
        self.closed = True

    @property
    def sha256(self) -> str:
        return self.digest.hexdigest()


class AtomicRunDirectory:
    def __init__(self, final_path: Path):
        self.final_path = final_path
        try:
            parent = final_path.parent.resolve(strict=True)
            parent_metadata = parent.lstat()
            existing = final_path.lstat()
        except FileNotFoundError:
            existing = None
        except OSError:
            fail("output parent or destination is unavailable")
        if not stat.S_ISDIR(parent_metadata.st_mode) or stat.S_ISLNK(
            parent_metadata.st_mode
        ):
            fail("output parent is not a real directory")
        if existing is not None:
            fail("output directory already exists")
        suffix = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
        self.stage = parent / f".{final_path.name}.incomplete-{suffix}"
        try:
            self.stage.mkdir(mode=0o700)
            (self.stage / "browser").mkdir(mode=0o700)
            (self.stage / "control").mkdir(mode=0o700)
            (self.stage / "runtime").mkdir(mode=0o700)
        except OSError:
            fail("failed to create the atomic output staging directory")
        self.published = False

    def publish(self) -> None:
        if self.published:
            fail("output directory is already published")
        try:
            descriptor = os.open(
                self.stage, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.rename(self.stage, self.final_path)
            parent_descriptor = os.open(
                self.final_path.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
            )
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        except OSError:
            fail("failed to publish the atomic output directory")
        self.published = True

    def abort(self) -> None:
        if not self.published:
            shutil.rmtree(self.stage, ignore_errors=True)


def process_group_exists(group: int) -> bool:
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def terminate_process_group(process: subprocess.Popen[Any]) -> None:
    group = process.pid
    if process_group_exists(group):
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + PROCESS_GRACE_SECONDS
    while time.monotonic() < deadline and process_group_exists(group):
        process.poll()
        time.sleep(0.02)
    if process_group_exists(group):
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + PROCESS_GRACE_SECONDS
    while time.monotonic() < deadline and process_group_exists(group):
        process.poll()
        time.sleep(0.02)
    try:
        process.wait(timeout=max(0.01, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        fail("child process leader did not become waitable")
    if process_group_exists(group):
        fail("child process group survived SIGKILL")


def run_bounded_command(
    arguments: list[str],
    label: str,
    *,
    timeout_seconds: float = 15.0,
    maximum_output: int = MAX_JSON_LINE_BYTES,
) -> bytes:
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as diagnostic:
        try:
            process = subprocess.Popen(
                arguments,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=diagnostic,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                terminate_process_group(process)
                fail(f"{label} timed out")
            output.seek(0)
            raw = output.read(maximum_output + 1)
        except OSError:
            fail(f"failed to execute {label}")
    if code != 0:
        fail(f"{label} failed")
    if len(raw) > maximum_output:
        fail(f"{label} output exceeds its bound")
    return raw


@dataclasses.dataclass(frozen=True)
class ServiceIdentity:
    unit: str
    main_pid: int
    user: str
    uid: int
    gid: int
    restarts: int


def query_service_identity(
    systemctl: str,
    unit: str,
    runner: Callable[..., bytes] = run_bounded_command,
) -> ServiceIdentity:
    if SERVICE_RE.fullmatch(unit) is None:
        fail("service unit syntax is invalid")
    raw = runner(
        [
            systemctl,
            "show",
            unit,
            "--property=MainPID",
            "--property=User",
            "--property=ActiveState",
            "--property=SubState",
            "--property=NRestarts",
            "--no-pager",
        ],
        "systemd service identity",
    )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError:
        fail("systemd service identity is not UTF-8")
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            fail("systemd service identity line is malformed")
        key, value = line.split("=", 1)
        if key in values:
            fail("systemd service identity field is duplicated")
        values[key] = value
    if set(values) != {"MainPID", "User", "ActiveState", "SubState", "NRestarts"}:
        fail("systemd service identity fields differ")
    if values["ActiveState"] != "active" or values["SubState"] != "running":
        raise TransientServiceState("service is not running")
    if not values["MainPID"].isdecimal() or int(values["MainPID"]) <= 0:
        raise TransientServiceState("service has no MainPID")
    if not values["NRestarts"].isdecimal():
        fail("systemd restart count is invalid")
    try:
        account = pwd.getpwnam(nonempty_string(values["User"], "service user"))
    except KeyError:
        fail("gateway service user does not exist")
    return ServiceIdentity(
        unit=unit,
        main_pid=int(values["MainPID"]),
        user=account.pw_name,
        uid=account.pw_uid,
        gid=account.pw_gid,
        restarts=int(values["NRestarts"]),
    )


@dataclasses.dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    parent_pid: int
    starttime_ticks: int
    uid: int
    executable_sha256: str


def _read_proc_file(path: Path, maximum: int, label: str) -> bytes:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        )
        with os.fdopen(descriptor, "rb") as handle:
            raw = handle.read(maximum + 1)
    except (FileNotFoundError, ProcessLookupError):
        raise TransientServiceState(f"{label} disappeared") from None
    except OSError:
        raise TransientServiceState(f"{label} is unavailable") from None
    if len(raw) > maximum:
        fail(f"{label} exceeds its size bound")
    return raw


def read_process_identity(pid: int) -> ProcessIdentity:
    if pid <= 0:
        fail("process PID is invalid")
    proc = Path("/proc") / str(pid)
    raw = _read_proc_file(proc / "stat", 16 * 1024, "process stat")
    try:
        text = raw.decode("ascii", errors="strict")
    except UnicodeError:
        fail("process stat is not ASCII")

    def parse_stat(value: str) -> tuple[int, int]:
        closing = value.rfind(") ")
        if closing < 1:
            fail("process stat framing differs")
        fields = value[closing + 2 :].split()
        if len(fields) < 20 or not fields[1].isdecimal() or not fields[19].isdecimal():
            fail("process stat fields differ")
        return int(fields[1]), int(fields[19])

    parent_pid, starttime_ticks = parse_stat(text)
    try:
        metadata = proc.stat()
        executable = os.readlink(proc / "exe").encode("utf-8", errors="strict")
    except (FileNotFoundError, ProcessLookupError):
        raise TransientServiceState(
            "process disappeared during identity query"
        ) from None
    except (OSError, UnicodeError):
        raise TransientServiceState("process executable is unavailable") from None
    after = _read_proc_file(proc / "stat", 16 * 1024, "process stat")
    try:
        after_text = after.decode("ascii", errors="strict")
    except UnicodeError:
        fail("process stat is not ASCII")
    if parse_stat(after_text) != (parent_pid, starttime_ticks):
        raise TransientServiceState("process identity changed during query")
    return ProcessIdentity(
        pid=pid,
        parent_pid=parent_pid,
        starttime_ticks=starttime_ticks,
        uid=metadata.st_uid,
        executable_sha256=sha256_bytes(executable),
    )


def query_worker_identity(service: ServiceIdentity) -> ProcessIdentity:
    gateway_before = read_process_identity(service.main_pid)
    if gateway_before.uid != service.uid:
        fail("gateway process UID differs from systemd service user")
    children_path = (
        Path("/proc")
        / str(service.main_pid)
        / "task"
        / str(service.main_pid)
        / "children"
    )
    raw = _read_proc_file(children_path, 4096, "gateway child list")
    try:
        text = raw.decode("ascii", errors="strict").strip()
    except UnicodeError:
        fail("gateway child list is not ASCII")
    children = text.split()
    if len(children) != 1 or not children[0].isdecimal() or int(children[0]) <= 0:
        raise TransientServiceState("gateway does not have exactly one worker")
    worker = read_process_identity(int(children[0]))
    gateway_after = read_process_identity(service.main_pid)
    if gateway_after != gateway_before:
        raise TransientServiceState("gateway identity changed during worker query")
    if worker.parent_pid != service.main_pid or worker.uid != service.uid:
        fail("worker parent or UID differs")
    return worker


def same_process(left: ProcessIdentity, right: ProcessIdentity) -> bool:
    return (
        left.pid == right.pid
        and left.parent_pid == right.parent_pid
        and left.starttime_ticks == right.starttime_ticks
        and left.uid == right.uid
        and left.executable_sha256 == right.executable_sha256
    )


def open_worker_pidfd(worker: ProcessIdentity) -> int:
    try:
        descriptor = os.pidfd_open(worker.pid, 0)
    except (AttributeError, OSError):
        fail("failed to open the worker pidfd")
    poller = select.poll()
    poller.register(descriptor, select.POLLIN)
    if poller.poll(0):
        os.close(descriptor)
        raise TransientServiceState("worker exited before pidfd pinning")
    return descriptor


def inject_worker_kill(
    pidfd: int,
    expected_service: ServiceIdentity,
    expected_worker: ProcessIdentity,
    *,
    systemctl: str,
) -> tuple[int, int]:
    current_service = query_service_identity(systemctl, expected_service.unit)
    current_worker = query_worker_identity(current_service)
    if current_service != expected_service or not same_process(
        current_worker, expected_worker
    ):
        fail("service or worker identity changed before fault injection")
    poller = select.poll()
    poller.register(pidfd, select.POLLIN)
    if poller.poll(0):
        fail("worker pidfd was already readable before fault injection")
    started = time.monotonic_ns()
    try:
        signal.pidfd_send_signal(pidfd, signal.SIGKILL)
    except (AttributeError, OSError):
        fail("pidfd worker fault injection failed")
    completed = time.monotonic_ns()
    if not poller.poll(5_000):
        fail("worker did not exit within the fatal deadline")
    return started, completed


def validate_lifecycle_payload(raw: bytes) -> dict[str, Any]:
    try:
        raw.decode("ascii", errors="strict")
    except UnicodeError:
        fail("lifecycle payload is not canonical ASCII")
    value = strict_json_object(raw, "lifecycle payload")
    event = value.get("event")
    if (
        value.get("schema_version") != LIFECYCLE_SCHEMA
        or not isinstance(event, str)
        or event not in LIFECYCLE_FIELDS
    ):
        fail("lifecycle schema or event differs")
    expected_fields = (
        {"schema_version", "event", "observed_monotonic_ns"}
        | LIFECYCLE_FIELDS[event]
    )
    if event == "request_released":
        actual_fields = set(value)
        if actual_fields != expected_fields and actual_fields != (
            expected_fields | REASONING_RELEASE_FIELDS
        ):
            fail("lifecycle event fields differ")
    else:
        exact_keys(value, expected_fields, "lifecycle event")
    integer(value["observed_monotonic_ns"], "lifecycle timestamp")
    request_id = value["request_id"]
    completion_id = value["completion_id"]
    if event == "worker_fatal" and request_id is None and completion_id is None:
        if value["admit_to_fatal_ns"] is not None:
            fail("idle worker fatal carries an admission duration")
    else:
        nonempty_string(request_id, "lifecycle request ID")
        nonempty_string(completion_id, "lifecycle completion ID")
    if (
        event
        in {
            "request_admitted",
            "request_started",
            "request_first_token",
            "request_cancel_requested",
            "request_released",
        }
        and value["stream"] is not True
    ):
        fail("failure-gate lifecycle request is not streaming")
    if event == "request_admitted":
        integer(value["prompt_tokens"], "admitted prompt tokens", minimum=1)
        integer(
            value["max_completion_tokens"],
            "admitted completion tokens",
            minimum=1,
        )
    elif event == "request_started":
        integer(value["prompt_tokens"], "started prompt tokens", minimum=1)
        integer(value["admit_to_start_ns"], "admit-to-start")
    elif event == "request_progress":
        if value["phase"] != "prefill":
            fail("gateway progress phase differs")
        processed = integer(
            value["processed_prompt_tokens"], "processed prompt tokens", minimum=1
        )
        prompt = integer(value["prompt_tokens"], "progress prompt tokens", minimum=1)
        if processed > prompt:
            fail("gateway progress exceeds prompt tokens")
    elif event == "request_first_token":
        if value["completion_tokens"] != 1:
            fail("gateway first-token count differs")
    elif event == "request_cancel_requested":
        nonempty_string(value["reason"], "gateway cancel reason")
        integer(value["admit_to_cancel_ns"], "admit-to-cancel")
    elif event == "request_released":
        if value["outcome"] not in {"stop", "length", "cancelled"}:
            fail("gateway release outcome differs")
        if value["reset_complete"] is not True:
            fail("gateway release reset is incomplete")
        integer(value["prompt_tokens"], "release prompt tokens", minimum=1)
        integer(value["completion_tokens"], "release completion tokens")
        for name in ("admit_to_start_ns", "start_to_release_ns", "admit_to_release_ns"):
            integer(value[name], name)
        has_reasoning_fields = "reasoning_tokens" in value
        if has_reasoning_fields != ("forced_end_tokens" in value):
            fail("reasoning release accounting fields are incomplete")
        if has_reasoning_fields:
            reasoning_tokens = integer(
                value["reasoning_tokens"], "reasoning release tokens"
            )
            forced_end_tokens = integer(
                value["forced_end_tokens"], "forced-end release tokens"
            )
            if reasoning_tokens + forced_end_tokens > value["completion_tokens"]:
                fail("reasoning release accounting exceeds completion tokens")
        if value["admit_to_release_ns"] != (
            value["admit_to_start_ns"] + value["start_to_release_ns"]
        ):
            fail("gateway release duration identity differs")
        if value["outcome"] == "cancelled":
            nonempty_string(value["cancel_reason"], "gateway release cancel reason")
        elif value["cancel_reason"] is not None:
            fail("normal gateway release carries a cancel reason")
    elif event == "worker_fatal":
        nonempty_string(value["reason"], "worker fatal reason")
        if request_id is not None:
            integer(value["admit_to_fatal_ns"], "admit-to-fatal")
    if compact_json(value) != raw:
        fail("lifecycle payload is not canonical gateway JSON")
    return value


def lifecycle_payload_from_journal_message(message: str) -> bytes | None:
    try:
        raw = message.encode("utf-8", errors="strict")
    except UnicodeError:
        fail("journal MESSAGE is not UTF-8")
    if raw.startswith(b"{"):
        candidate = raw
    elif raw.startswith(b"INFO:     {"):
        candidate = raw[len(b"INFO:     ") :]
    else:
        return None
    value = strict_json_object(candidate, "journal lifecycle MESSAGE")
    if value.get("schema_version") != LIFECYCLE_SCHEMA:
        return None
    validate_lifecycle_payload(candidate)
    return candidate


@dataclasses.dataclass(frozen=True)
class JournalLifecycle:
    cursor: str
    journal_monotonic_usec: int
    journal_pid: int
    raw: bytes
    event: dict[str, Any]


def validate_journal_record(
    payload: bytes,
    *,
    service: str,
    boot_id: str,
    cursors: set[str],
    lifecycle_payloads: set[bytes],
) -> tuple[str, JournalLifecycle | None]:
    record = strict_json_object(payload, "journal record")
    required = {
        "__CURSOR",
        "__MONOTONIC_TIMESTAMP",
        "_BOOT_ID",
        "_PID",
        "_SYSTEMD_UNIT",
        "PRIORITY",
        "MESSAGE",
    }
    if not required.issubset(record):
        fail("journal record lacks required fields")
    cursor = nonempty_string(record["__CURSOR"], "journal cursor")
    if cursor in cursors:
        fail("journal cursor is duplicated")
    if record["_BOOT_ID"] != boot_id:
        fail("journal boot identity differs")
    monotonic = str(record["__MONOTONIC_TIMESTAMP"])
    pid = str(record["_PID"])
    priority = record["PRIORITY"]
    if not monotonic.isdecimal() or not pid.isdecimal() or int(pid) <= 0:
        fail("journal numeric identity is invalid")
    if (
        not isinstance(priority, str)
        or not priority.isascii()
        or not priority.isdecimal()
        or not 0 <= int(priority, 10) <= 7
    ):
        fail("journal PRIORITY is invalid")
    message = record["MESSAGE"]
    if not isinstance(message, str):
        fail("journal MESSAGE is not text")
    service_record = record["_SYSTEMD_UNIT"] == service
    manager_record = (
        pid == "1"
        and record["_SYSTEMD_UNIT"] == "init.scope"
        and record.get("UNIT") == service
        and record.get("SYSLOG_IDENTIFIER") == "systemd"
    )
    if not service_record and not manager_record:
        fail("journal service identity differs")
    lifecycle_raw = (
        lifecycle_payload_from_journal_message(message) if service_record else None
    )
    lifecycle = None
    if lifecycle_raw is not None:
        if lifecycle_raw in lifecycle_payloads:
            fail("journal lifecycle payload is duplicated")
        lifecycle = JournalLifecycle(
            cursor=cursor,
            journal_monotonic_usec=int(monotonic),
            journal_pid=int(pid),
            raw=lifecycle_raw,
            event=validate_lifecycle_payload(lifecycle_raw),
        )
    return cursor, lifecycle


class JournalFollower:
    def __init__(
        self,
        process: subprocess.Popen[bytes],
        writer: AtomicLineWriter,
        *,
        service: str,
        boot_id: str,
    ):
        self.process = process
        self.writer = writer
        self.service = service
        self.boot_id = boot_id
        self.records: list[bytes] = []
        self.lifecycle: list[JournalLifecycle] = []
        self.cursors: set[str] = set()
        self.lifecycle_payloads: set[bytes] = set()
        self.error: BaseException | None = None
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.stderr_digest = hashlib.sha256()
        self.stderr_bytes = 0
        if process.stdout is None or process.stderr is None:
            fail("journal follower pipes are unavailable")
        self.stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)

    def start(self) -> None:
        self.stdout_thread.start()
        self.stderr_thread.start()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            while True:
                raw = self.process.stdout.readline(MAX_JSON_LINE_BYTES + 2)
                if raw == b"":
                    if self.stop_event.is_set():
                        return
                    fail("journal follower ended unexpectedly")
                if not raw.endswith(b"\n") or len(raw) > MAX_JSON_LINE_BYTES + 1:
                    fail("journal line framing or size is invalid")
                payload = raw[:-1]
                cursor, lifecycle = validate_journal_record(
                    payload,
                    service=self.service,
                    boot_id=self.boot_id,
                    cursors=self.cursors,
                    lifecycle_payloads=self.lifecycle_payloads,
                )
                with self.condition:
                    if len(self.records) >= MAX_JOURNAL_LINES:
                        fail("journal line count exceeds its bound")
                    self.writer.write_line(payload)
                    self.cursors.add(cursor)
                    self.records.append(payload)
                    if lifecycle is not None:
                        self.lifecycle_payloads.add(lifecycle.raw)
                        self.lifecycle.append(lifecycle)
                    self.condition.notify_all()
        except BaseException as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        try:
            while chunk := self.process.stderr.read(COPY_CHUNK_BYTES):
                self.stderr_bytes += len(chunk)
                if self.stderr_bytes > MAX_STDERR_BYTES:
                    fail("journal follower stderr exceeds its bound")
                self.stderr_digest.update(chunk)
        except BaseException as error:
            with self.condition:
                if self.error is None:
                    self.error = error
                self.condition.notify_all()

    def wait_for(
        self,
        predicate: Callable[[list[JournalLifecycle]], bool],
        deadline_ns: int,
        label: str,
    ) -> list[JournalLifecycle]:
        with self.condition:
            while True:
                if self.error is not None:
                    raise self.error
                snapshot = list(self.lifecycle)
                if predicate(snapshot):
                    return snapshot
                remaining = deadline_ns - time.monotonic_ns()
                if remaining <= 0:
                    fail(f"{label} timed out")
                self.condition.wait(min(0.1, remaining / 1_000_000_000))

    def snapshot(self) -> list[JournalLifecycle]:
        with self.condition:
            if self.error is not None:
                raise self.error
            return list(self.lifecycle)

    def stop(self) -> None:
        self.stop_event.set()
        terminate_process_group(self.process)
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()
        self.stdout_thread.join(timeout=2.0)
        self.stderr_thread.join(timeout=2.0)
        if self.stdout_thread.is_alive() or self.stderr_thread.is_alive():
            fail("journal follower drains did not terminate")
        if self.error is not None:
            raise self.error


def read_boot_id() -> str:
    raw = _read_proc_file(Path("/proc/sys/kernel/random/boot_id"), 128, "boot ID")
    try:
        value = raw.decode("ascii", errors="strict").strip().replace("-", "")
    except UnicodeError:
        fail("boot ID is not ASCII")
    if re.fullmatch(r"[0-9a-f]{32}", value) is None:
        fail("boot ID syntax differs")
    return value


def initial_journal_cursor(journalctl: str, service: str) -> str:
    raw = run_bounded_command(
        [
            journalctl,
            f"--unit={service}",
            "--boot",
            "--lines=0",
            "--show-cursor",
            "--no-pager",
        ],
        "initial journal cursor",
    )
    try:
        lines = raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeError:
        fail("initial journal cursor output is not UTF-8")
    cursors = [
        line[len("-- cursor: ") :] for line in lines if line.startswith("-- cursor: ")
    ]
    if len(cursors) != 1 or not cursors[0]:
        fail("initial journal cursor is missing or duplicated")
    return cursors[0]


def spawn_journal_follower(
    journalctl: str, service: str, cursor: str
) -> subprocess.Popen[bytes]:
    try:
        return subprocess.Popen(
            [
                journalctl,
                f"--unit={service}",
                "--boot",
                f"--after-cursor={cursor}",
                "--follow",
                "--output=json",
                "--no-pager",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=True,
        )
    except OSError:
        fail("failed to start the journal follower")


class BrowserProcess:
    def __init__(self, process: subprocess.Popen[bytes], writer: AtomicLineWriter):
        self.process = process
        self.writer = writer
        self.lines: list[tuple[bytes, dict[str, Any]]] = []
        self.stderr_digest = hashlib.sha256()
        self.stderr_bytes = 0
        self.error: BaseException | None = None
        self.condition = threading.Condition()
        if process.stdout is None or process.stderr is None:
            fail("browser process pipes are unavailable")
        self.stdout_thread = threading.Thread(target=self._stdout, daemon=True)
        self.stderr_thread = threading.Thread(target=self._stderr, daemon=True)

    def start(self) -> None:
        self.stdout_thread.start()
        self.stderr_thread.start()

    def _stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            while True:
                raw = self.process.stdout.readline(MAX_JSON_LINE_BYTES + 2)
                if raw == b"":
                    return
                if not raw.endswith(b"\n") or len(raw) > MAX_JSON_LINE_BYTES + 1:
                    fail("browser stdout line framing or size is invalid")
                payload = raw[:-1]
                value = strict_json_object(payload, "browser stdout")
                with self.condition:
                    if len(self.lines) >= MAX_BROWSER_LINES:
                        fail("browser stdout line count exceeds its bound")
                    self.writer.write_line(payload)
                    self.lines.append((payload, value))
                    self.condition.notify_all()
        except BaseException as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()

    def _stderr(self) -> None:
        assert self.process.stderr is not None
        try:
            while chunk := self.process.stderr.read(COPY_CHUNK_BYTES):
                self.stderr_bytes += len(chunk)
                if self.stderr_bytes > MAX_STDERR_BYTES:
                    fail("browser stderr exceeds its bound")
                self.stderr_digest.update(chunk)
        except BaseException as error:
            with self.condition:
                if self.error is None:
                    self.error = error
                self.condition.notify_all()

    def wait_record(
        self, record_type: str, deadline_ns: int
    ) -> tuple[bytes, dict[str, Any]]:
        with self.condition:
            while True:
                if self.error is not None:
                    raise self.error
                matches = [
                    item
                    for item in self.lines
                    if item[1].get("record_type") == record_type
                ]
                if len(matches) > 1:
                    fail("browser stdout record type is duplicated")
                if matches:
                    return matches[0]
                if self.process.poll() is not None:
                    fail("browser exited before its required summary")
                remaining = deadline_ns - time.monotonic_ns()
                if remaining <= 0:
                    fail("browser summary timed out")
                self.condition.wait(min(0.1, remaining / 1_000_000_000))

    def wait_exit(self, deadline_ns: int) -> int:
        remaining = deadline_ns - time.monotonic_ns()
        if remaining <= 0:
            fail("browser process deadline expired")
        try:
            code = self.process.wait(timeout=remaining / 1_000_000_000)
        except subprocess.TimeoutExpired:
            fail("browser process timed out")
        self.stdout_thread.join(timeout=2.0)
        self.stderr_thread.join(timeout=2.0)
        if self.stdout_thread.is_alive() or self.stderr_thread.is_alive():
            fail("browser pipe drains did not terminate")
        if self.error is not None:
            raise self.error
        return code


def normalized_content_image(raw: str) -> tuple[str, str]:
    match = CONTENT_IMAGE_RE.fullmatch(raw)
    if match is None:
        fail("container image must be an immutable SHA-256 content identity")
    digest = f"sha256:{match.group(1)}"
    return raw, digest


def normalized_openwebui_url(raw: str) -> str:
    try:
        value = urllib.parse.urlsplit(raw)
    except ValueError:
        fail("OpenWebUI URL is invalid")
    if (
        value.scheme not in {"http", "https"}
        or not value.hostname
        or value.username is not None
        or value.password is not None
        or value.path not in {"", "/"}
        or value.query
        or value.fragment
    ):
        fail("OpenWebUI URL shape differs")
    port = f":{value.port}" if value.port is not None else ""
    return f"{value.scheme}://{value.hostname}{port}/"


def normalized_ready_url(raw: str) -> str:
    try:
        value = urllib.parse.urlsplit(raw)
    except ValueError:
        fail("gateway readiness URL is invalid")
    if (
        value.scheme != "http"
        or value.hostname != "172.20.0.1"
        or value.port != 8000
        or value.path != "/readyz"
        or value.username is not None
        or value.password is not None
        or value.query
        or value.fragment
    ):
        fail("gateway readiness URL differs from the deployment boundary")
    return raw


def _safe_mount_path(path: Path, label: str) -> str:
    value = os.fspath(path.resolve(strict=True))
    if "," in value or "\0" in value or "\n" in value or "\r" in value:
        fail(f"{label} path cannot be represented as a Docker mount")
    return value


def validate_browser_output_layout(root: Path) -> tuple[Path, Path]:
    try:
        metadata = root.lstat()
        entries = {entry.name: entry for entry in root.iterdir()}
    except OSError:
        fail("browser container output is unavailable")
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or set(entries) != {SCREENSHOT_NAME, BROWSER_SUMMARY_NAME}
    ):
        fail("browser container output layout or identity differs")
    return entries[SCREENSHOT_NAME], entries[BROWSER_SUMMARY_NAME]


def snapshot_validated_browser_artifacts(
    source_root: Path,
    destination_root: Path,
    *,
    expected_summary: bytes,
    expected_screenshot_sha256: str,
) -> tuple[bytes, bytes]:
    screenshot_source, summary_source = validate_browser_output_layout(source_root)
    screenshot = read_regular_exact(
        screenshot_source, "browser failed-state screenshot", 64 * 1024 * 1024
    )
    summary = read_regular_exact(
        summary_source, "browser failure summary artifact", MAX_JSON_LINE_BYTES + 1
    )
    if (
        sha256_bytes(screenshot) != expected_screenshot_sha256
        or summary != expected_summary + b"\n"
    ):
        fail("browser artifacts changed after final validation")
    write_private_snapshot(
        destination_root / SCREENSHOT_NAME,
        screenshot,
        "browser failed-state screenshot",
    )
    write_private_snapshot(
        destination_root / BROWSER_SUMMARY_NAME,
        summary,
        "browser failure summary",
    )
    return screenshot, summary


def build_browser_command(
    *,
    docker: str,
    image: str,
    name: str,
    script: Path,
    token_file: Path,
    browser_output: Path,
    control_dir: Path,
    openwebui_url: str,
    uid: int,
    gid: int,
    control_timeout_ms: int,
) -> list[str]:
    image, _digest = normalized_content_image(image)
    mounts = (
        f"type=bind,src={_safe_mount_path(script, 'browser script')},dst={BROWSER_SCRIPT_CONTAINER_PATH},readonly",
        f"type=bind,src={_safe_mount_path(token_file, 'token file')},dst=/run/secrets/openwebui-token,readonly",
        f"type=bind,src={_safe_mount_path(browser_output, 'browser output')},dst=/output",
        f"type=bind,src={_safe_mount_path(control_dir, 'control directory')},dst=/run/control,readonly",
    )
    command = [
        docker,
        "run",
        "--rm",
        "--network=host",
        f"--name={name}",
        f"--user={uid}:{gid}",
        "--pids-limit=256",
        "--security-opt=no-new-privileges",
    ]
    for mount in mounts:
        command.extend(("--mount", mount))
    command.extend(
        (
            "--env",
            f"OPENWEBUI_URL={openwebui_url}",
            "--env",
            "OPENWEBUI_TOKEN_FILE=/run/secrets/openwebui-token",
            "--env",
            f"ULLM_MODEL_ID={MODEL_ID}",
            "--env",
            f"ULLM_MODEL_NAME={MODEL_LABEL}",
            "--env",
            "OPENWEBUI_FAILURE_SCREENSHOT=/output/post-header-failure.png",
            "--env",
            "OPENWEBUI_FAILURE_SUMMARY=/output/openwebui-failure-summary.json",
            "--env",
            f"OPENWEBUI_WORKER_KILLED_FILE={KILL_CONTROL_CONTAINER_PATH}",
            "--env",
            f"OPENWEBUI_GATEWAY_RECOVERED_FILE={RECOVERY_CONTROL_CONTAINER_PATH}",
            "--env",
            f"OPENWEBUI_FAILURE_CONTROL_TIMEOUT_MS={control_timeout_ms}",
            image,
            "node",
            BROWSER_SCRIPT_CONTAINER_PATH,
        )
    )
    return command


def spawn_browser(command: list[str]) -> subprocess.Popen[bytes]:
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=True,
        )
    except OSError:
        fail("failed to start the transient browser container")


def build_ready_probe_command(
    *,
    docker: str,
    image: str,
    network: str,
    ready_url: str,
    timeout_seconds: int,
    uid: int,
    gid: int,
    name: str | None = None,
) -> list[str]:
    image, _digest = normalized_content_image(image)
    if NETWORK_RE.fullmatch(network) is None:
        fail("Docker network name syntax is invalid")
    normalized_ready_url(ready_url)
    if timeout_seconds < 1 or timeout_seconds > 600:
        fail("readiness probe timeout is outside its bound")
    command = [
        docker,
        "run",
        "--rm",
        f"--network={network}",
        f"--user={uid}:{gid}",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=32",
        "--memory=128m",
    ]
    if name is not None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) is None:
            fail("readiness probe container name syntax is invalid")
        command.append(f"--name={name}")
    command.extend(
        (
            image,
            "--fail",
            "--silent",
            "--show-error",
            "--output",
            "/dev/null",
            "--write-out",
            '{"ready":true,"status":%{http_code}}\\n',
            "--retry",
            "999999",
            "--retry-delay",
            "1",
            "--retry-all-errors",
            "--retry-max-time",
            str(timeout_seconds),
            "--max-time",
            "3",
            ready_url,
        )
    )
    return command


def run_ready_probe(command: list[str], timeout_seconds: int) -> tuple[int, int]:
    started = time.monotonic_ns()
    raw = run_bounded_command(
        command,
        "Docker-network gateway readiness probe",
        timeout_seconds=timeout_seconds + 10,
        maximum_output=4096,
    )
    completed = time.monotonic_ns()
    if raw != b'{"ready":true,"status":200}\n':
        fail("readiness probe output differs")
    return started, completed


@dataclasses.dataclass(frozen=True)
class DockerNetworkIdentity:
    network_id: str
    subnet: str
    gateway: str


def query_docker_network(
    docker: str,
    network: str,
    runner: Callable[..., bytes] = run_bounded_command,
) -> DockerNetworkIdentity:
    if NETWORK_RE.fullmatch(network) is None:
        fail("Docker network name syntax is invalid")
    raw = runner(
        [
            docker,
            "network",
            "inspect",
            "--format={{.Id}}|{{(index .IPAM.Config 0).Subnet}}|{{(index .IPAM.Config 0).Gateway}}",
            network,
        ],
        "Docker network identity",
    )
    try:
        text = raw.decode("ascii", errors="strict").strip()
    except UnicodeError:
        fail("Docker network identity is not ASCII")
    fields = text.split("|")
    if (
        len(fields) != 3
        or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None
        or fields[1] != "172.20.0.0/16"
        or fields[2] != "172.20.0.1"
    ):
        fail("Docker network identity differs from the deployment boundary")
    return DockerNetworkIdentity(*fields)


def control_content(stage: str, nonce: str) -> bytes:
    if stage not in {"worker_killed", "gateway_recovered"}:
        fail("failure control stage is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        fail("failure control nonce is invalid")
    return f"{CONTROL_SCHEMA}:{stage}:{nonce}\n".encode("ascii")


def create_control_file(path: Path, stage: str, nonce: str) -> tuple[int, str]:
    raw = control_content(stage, nonce)
    created = time.monotonic_ns()
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o400,
        )
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            handle.write(raw)
            os.fsync(handle.fileno())
    except OSError:
        fail("failed to publish browser control file")
    return created, sha256_bytes(raw)


def write_atomic_json(path: Path, value: dict[str, Any]) -> bytes:
    raw = compact_json(value) + b"\n"
    incomplete = path.with_name(path.name + ".incomplete")
    try:
        descriptor = os.open(
            incomplete,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            handle.write(raw)
            os.fsync(handle.fileno())
        os.rename(incomplete, path)
    except OSError:
        fail("failed to publish an atomic JSON artifact")
    return raw


def wait_recovered_service(
    *,
    systemctl: str,
    service: str,
    initial_service: ServiceIdentity,
    initial_worker: ProcessIdentity,
    deadline_ns: int,
) -> tuple[ServiceIdentity, ProcessIdentity]:
    while time.monotonic_ns() < deadline_ns:
        try:
            candidate = query_service_identity(systemctl, service)
            worker = query_worker_identity(candidate)
        except TransientServiceState:
            time.sleep(0.1)
            continue
        if candidate.restarts > initial_service.restarts + 1:
            fail("gateway restarted more than once")
        if candidate.restarts == initial_service.restarts + 1:
            if candidate.main_pid == initial_service.main_pid:
                fail("restarted gateway retained its MainPID")
            if (
                worker.pid == initial_worker.pid
                or worker.starttime_ticks == initial_worker.starttime_ticks
                or same_process(worker, initial_worker)
            ):
                fail("restarted service retained its worker identity")
            return candidate, worker
        if candidate != initial_service or not same_process(worker, initial_worker):
            fail("service identity changed without the planned restart count")
        time.sleep(0.1)
    fail("systemd service recovery timed out")


def _identity_hashes(value: dict[str, Any], prefix: str, label: str) -> None:
    exact_keys(
        value,
        {f"{prefix}_utf8_bytes", f"{prefix}_sha256"},
        label,
    )
    integer(value[f"{prefix}_utf8_bytes"], f"{label} bytes", minimum=1)
    sha256_value(value[f"{prefix}_sha256"], f"{label} digest")


def _validate_result(value: Any, *, action: str) -> None:
    if not isinstance(value, dict):
        fail("browser action result is not an object")
    exact_keys(
        value,
        {"visible", "enabled", "text_utf8_bytes", "text_sha256"},
        "browser action result",
    )
    expected_enabled = True if action in {"submit_chat", "wait_ready"} else None
    if value["visible"] is not True or value["enabled"] is not expected_enabled:
        fail("browser action visibility or enabled state differs")
    if action in {"wait_visible", "wait_failed"} or (
        action == "wait_ready" and value["text_utf8_bytes"] is not None
    ):
        integer(value["text_utf8_bytes"], "browser action text bytes", minimum=1)
        sha256_value(value["text_sha256"], "browser action text digest")
    elif value["text_utf8_bytes"] is not None or value["text_sha256"] is not None:
        fail("browser action unexpectedly carries text evidence")


def validate_browser_actions(
    actions: Any,
    *,
    expected_count: int,
) -> tuple[int, int]:
    if not isinstance(actions, list) or len(actions) != expected_count:
        fail("browser action count differs")
    expected_selectors = (
        None,
        "body",
        "#chat-input",
        ".chat-assistant",
        ".chat-assistant",
        "#chat-input",
        "#chat-input",
        ".chat-assistant",
        "#chat-input",
    )
    prior_completed = -1
    first_started = -1
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            fail("browser action is not an object")
        exact_keys(
            action,
            {
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
            },
            "browser action",
        )
        expected = FINAL_ACTIONS[index]
        action_index = integer(action["action_index"], "browser action index")
        if (
            action["browser_case"] != BROWSER_CASE
            or action_index != index
            or action["action"] != expected
            or action["selector"] != expected_selectors[index]
        ):
            fail("browser action identity or selector differs")
        expected_input = None
        if index == 1:
            expected_input = sha256_bytes(MODEL_ID.encode("utf-8"))
        elif index == 2:
            expected_input = sha256_bytes(FAILURE_PROMPT.encode("utf-8"))
        elif index == 6:
            expected_input = sha256_bytes(RECOVERY_PROMPT.encode("utf-8"))
        elif index == 0:
            sha256_value(action["input_sha256"], "navigation input digest")
            expected_input = action["input_sha256"]
        if action["input_sha256"] != expected_input:
            fail("browser action input digest differs")
        started = decimal_timestamp(
            action["started_monotonic_ns"], "browser action start"
        )
        completed = decimal_timestamp(
            action["completed_monotonic_ns"], "browser action completion"
        )
        if completed < started or started < prior_completed:
            fail("browser actions overlap or regress")
        if index == 0:
            first_started = started
        prior_completed = completed
        _validate_result(action["result"], action=expected)
        screenshot_expected = index == 4
        if screenshot_expected:
            if (
                action["screenshot_file"] != f"browser/{SCREENSHOT_NAME}"
                or SHA256_RE.fullmatch(
                    nonempty_string(
                        action["screenshot_sha256"], "browser screenshot digest"
                    )
                )
                is None
            ):
                fail("failed-state screenshot action differs")
        elif (
            action["screenshot_file"] is not None
            or action["screenshot_sha256"] is not None
        ):
            fail("unexpected browser action screenshot evidence")
    return first_started, prior_completed


def validate_socket_events(
    events: Any,
    *,
    allow_recovery: bool,
    require_failure: bool,
) -> dict[str, int]:
    if (
        not isinstance(events, list)
        or not events
        or len(events) > MAX_BROWSER_SOCKET_EVENTS
    ):
        fail("browser socket event evidence is empty or outside its bound")
    fields = {
        "sequence",
        "observed_monotonic_ns",
        "correlation_target",
        "type",
        "done",
        "has_error",
        "content_utf8_bytes",
        "content_sha256",
    }
    allowed_types = {
        "chat:active",
        "chat:completion",
        "chat:outlet",
        "chat:tasks:cancel",
    }
    prior = -1
    target_content = 0
    target_error = 0
    target_cancel = 0
    target_done = 0
    recovery_content = 0
    recovery_error = 0
    recovery_cancel = 0
    recovery_done = 0
    first_target_content_ns = -1
    first_target_error_ns = -1
    first_target_cancel_ns = -1
    first_recovery_content_ns = -1
    first_recovery_done_ns = -1
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            fail("browser socket event is not an object")
        exact_keys(event, fields, "browser socket event")
        timestamp = decimal_timestamp(
            event["observed_monotonic_ns"], "browser socket event timestamp"
        )
        target = event["correlation_target"]
        if (
            integer(event["sequence"], "browser socket event sequence") != index
            or timestamp < prior
            or target not in {"failure_target", "recovery_target"}
            or (target == "recovery_target" and not allow_recovery)
            or event["type"] not in allowed_types
            or not isinstance(event["done"], bool)
            or not isinstance(event["has_error"], bool)
        ):
            fail("browser socket event identity or ordering differs")
        prior = timestamp
        content_bytes = integer(
            event["content_utf8_bytes"], "browser socket content bytes"
        )
        if content_bytes == 0:
            if event["content_sha256"] is not None:
                fail("empty browser socket content carries a digest")
        else:
            sha256_value(event["content_sha256"], "browser socket content digest")
        if event["type"] in {"chat:active", "chat:outlet"} and (
            event["done"] or event["has_error"] or content_bytes != 0
        ):
            fail("browser state event carries content or terminal state")
        if event["done"] and event["type"] != "chat:completion":
            fail("browser non-completion event is terminal")
        if event["has_error"] and event["type"] != "chat:completion":
            fail("browser non-completion event carries a provider error")
        is_content = event["type"] == "chat:completion" and content_bytes > 0
        is_error = event["type"] == "chat:completion" and event["has_error"]
        is_cancel = event["type"] == "chat:tasks:cancel"
        is_done = event["type"] == "chat:completion" and event["done"]
        if target == "failure_target":
            if first_target_error_ns >= 0 and event["type"] == "chat:completion":
                fail("browser target completion follows provider error")
            if is_content:
                target_content += 1
                if first_target_content_ns < 0:
                    first_target_content_ns = timestamp
                if first_target_error_ns >= 0:
                    fail("browser target content follows provider error")
            if is_error:
                target_error += 1
                if first_target_error_ns < 0:
                    first_target_error_ns = timestamp
            if is_cancel:
                target_cancel += 1
                if first_target_cancel_ns < 0:
                    first_target_cancel_ns = timestamp
            if is_done:
                target_done += 1
        else:
            if first_recovery_done_ns >= 0 and event["type"] == "chat:completion":
                fail("browser recovery completion follows normal completion")
            if is_content:
                if first_recovery_done_ns >= 0:
                    fail("browser recovery content follows normal completion")
                recovery_content += 1
                if first_recovery_content_ns < 0:
                    first_recovery_content_ns = timestamp
            if is_error:
                recovery_error += 1
            if is_cancel:
                recovery_cancel += 1
            if is_done:
                recovery_done += 1
                if first_recovery_done_ns < 0:
                    first_recovery_done_ns = timestamp
    if target_content < 1 or target_done != 0:
        fail("failure target content or done count differs")
    if require_failure:
        if (
            target_error != 1
            or target_cancel != 1
            or first_target_cancel_ns < first_target_error_ns
        ):
            fail("failure target provider-error or cancellation count differs")
    elif target_error != 0 or target_cancel != 0:
        fail("failure target terminated before fault injection")
    if allow_recovery:
        if (
            recovery_content < 1
            or recovery_error != 0
            or recovery_cancel != 0
            or recovery_done != 1
            or first_recovery_done_ns < first_recovery_content_ns
        ):
            fail("browser recovery terminal state differs")
    elif recovery_content or recovery_error or recovery_cancel or recovery_done:
        fail("browser recovery events appeared before recovery")
    return {
        "target_content_count": target_content,
        "target_error_count": target_error,
        "target_cancel_count": target_cancel,
        "target_done_count": target_done,
        "first_target_content_ns": first_target_content_ns,
        "first_target_error_ns": first_target_error_ns,
        "first_target_cancel_ns": first_target_cancel_ns,
        "recovery_content_count": recovery_content,
        "recovery_done_count": recovery_done,
        "first_recovery_content_ns": first_recovery_content_ns,
        "first_recovery_done_ns": first_recovery_done_ns,
    }


def _validate_clear_control(
    value: Any,
    *,
    stage: str,
    expected_path: str,
    expected_timeout_ms: int,
) -> str:
    if not isinstance(value, dict):
        fail("browser clear control evidence is not an object")
    exact_keys(
        value,
        {
            "control_schema",
            "control_stage",
            "control_file",
            "nonce",
            "content_utf8_bytes",
            "content_sha256",
            "timeout_ms",
        },
        "browser clear control evidence",
    )
    nonce = nonempty_string(value["nonce"], "browser control nonce")
    expected = control_content(stage, nonce)
    content_bytes = integer(
        value["content_utf8_bytes"], "browser control content bytes", minimum=1
    )
    timeout_ms = integer(value["timeout_ms"], "browser control timeout", minimum=1)
    if (
        value["control_schema"] != CONTROL_SCHEMA
        or value["control_stage"] != stage
        or value["control_file"] != expected_path
        or content_bytes != len(expected)
        or value["content_sha256"] != sha256_bytes(expected)
        or timeout_ms != expected_timeout_ms
    ):
        fail("browser clear control evidence differs")
    return nonce


def _validate_redacted_control(
    value: Any,
    *,
    stage: str,
    expected_path: str,
    nonce: str,
) -> tuple[int, int]:
    if not isinstance(value, dict):
        fail("browser redacted control evidence is not an object")
    exact_keys(
        value,
        {
            "control_schema",
            "control_stage",
            "control_file_utf8_bytes",
            "control_file_sha256",
            "nonce_sha256",
            "content_utf8_bytes",
            "content_sha256",
            "requested_monotonic_ns",
            "observed_monotonic_ns",
        },
        "browser redacted control evidence",
    )
    expected = control_content(stage, nonce)
    control_file_bytes = integer(
        value["control_file_utf8_bytes"], "browser control file bytes", minimum=1
    )
    content_bytes = integer(
        value["content_utf8_bytes"], "browser control content bytes", minimum=1
    )
    requested = decimal_timestamp(
        value["requested_monotonic_ns"], "browser control request"
    )
    observed = decimal_timestamp(
        value["observed_monotonic_ns"], "browser control observation"
    )
    if (
        value["control_schema"] != CONTROL_SCHEMA
        or value["control_stage"] != stage
        or control_file_bytes != len(expected_path.encode("utf-8"))
        or value["control_file_sha256"] != sha256_bytes(expected_path.encode("utf-8"))
        or value["nonce_sha256"] != sha256_bytes(nonce.encode("ascii"))
        or content_bytes != len(expected)
        or value["content_sha256"] != sha256_bytes(expected)
        or observed < requested
    ):
        fail("browser redacted control evidence differs")
    return requested, observed


def _validate_target_identity(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} is not an object")
    exact_keys(
        value,
        {
            "chat_id_utf8_bytes",
            "chat_id_sha256",
            "message_id_utf8_bytes",
            "message_id_sha256",
        },
        label,
    )
    integer(value["chat_id_utf8_bytes"], f"{label} chat bytes", minimum=1)
    integer(value["message_id_utf8_bytes"], f"{label} message bytes", minimum=1)
    sha256_value(value["chat_id_sha256"], f"{label} chat digest")
    sha256_value(value["message_id_sha256"], f"{label} message digest")
    return value


@dataclasses.dataclass(frozen=True)
class BrowserInterim:
    target: dict[str, Any]
    kill_nonce: str
    recovery_nonce: str | None
    visible_completed_ns: int
    error_observed_ns: int | None
    cancel_observed_ns: int | None
    kill_wait_requested_ns: int
    kill_control_observed_ns: int | None
    recovery_wait_requested_ns: int | None
    first_target_content_ns: int
    action_prefix: tuple[bytes, ...]
    socket_prefix: tuple[bytes, ...]


def _canonical_record_prefix(values: list[Any], label: str) -> tuple[bytes, ...]:
    if not isinstance(values, list):
        fail(f"{label} is not a list")
    return tuple(compact_json(value) for value in values)


def _require_record_prefix(
    observed: tuple[bytes, ...], expected: tuple[bytes, ...], label: str
) -> None:
    if len(observed) < len(expected) or observed[: len(expected)] != expected:
        fail(f"{label} changed between browser evidence records")


def validate_kill_interim(
    value: dict[str, Any],
    raw: bytes,
    guard: SecretGuard,
    *,
    expected_timeout_ms: int,
) -> BrowserInterim:
    exact_keys(
        value,
        {
            "schema_version",
            "record_type",
            "browser_case",
            "observed_monotonic_ns",
            "browser_actions",
            "socket_correlation",
            "socket_events",
            "page_error_count",
            "worker_killed_control",
        },
        "browser worker-kill interim",
    )
    if (
        value["schema_version"] != BROWSER_SCHEMA
        or value["record_type"] != "openwebui_failure_worker_kill_wait"
        or value["browser_case"] != BROWSER_CASE
    ):
        fail("browser worker-kill interim identity differs")
    if integer(value["page_error_count"], "browser page-error count") != 0:
        fail("browser worker-kill interim page-error count differs")
    _first_action, last_action = validate_browser_actions(
        value["browser_actions"], expected_count=4
    )
    observed = decimal_timestamp(
        value["observed_monotonic_ns"], "browser worker-kill interim timestamp"
    )
    if observed < last_action:
        fail("browser worker-kill interim precedes its actions")
    socket = validate_socket_events(
        value["socket_events"], allow_recovery=False, require_failure=False
    )
    correlation = value["socket_correlation"]
    if not isinstance(correlation, dict):
        fail("browser worker-kill correlation is not an object")
    exact_keys(
        correlation,
        {
            "target",
            "submit_completed_monotonic_ns",
            "visible_completed_monotonic_ns",
            "pre_fault_done_count",
            "pre_fault_error_count",
            "pre_fault_cancel_count",
        },
        "browser worker-kill correlation",
    )
    target = _validate_target_identity(correlation["target"], "failure target")
    submit = decimal_timestamp(
        correlation["submit_completed_monotonic_ns"], "failure submit completion"
    )
    visible = decimal_timestamp(
        correlation["visible_completed_monotonic_ns"], "failure visible completion"
    )
    actions = value["browser_actions"]
    submit_started = decimal_timestamp(
        actions[2]["started_monotonic_ns"], "failure submit action start"
    )
    pre_fault_done = integer(
        correlation["pre_fault_done_count"], "browser pre-fault done count"
    )
    pre_fault_error = integer(
        correlation["pre_fault_error_count"], "browser pre-fault error count"
    )
    pre_fault_cancel = integer(
        correlation["pre_fault_cancel_count"], "browser pre-fault cancel count"
    )
    if (
        submit
        != decimal_timestamp(
            actions[2]["completed_monotonic_ns"], "failure submit action completion"
        )
        or visible
        != decimal_timestamp(
            actions[3]["completed_monotonic_ns"], "failure visible action completion"
        )
        or socket["first_target_content_ns"] > visible
        or socket["first_target_content_ns"] < submit_started
        or pre_fault_done != 0
        or pre_fault_error != 0
        or pre_fault_cancel != 0
    ):
        fail("browser worker-kill action or socket correlation differs")
    nonce = _validate_clear_control(
        value["worker_killed_control"],
        stage="worker_killed",
        expected_path=KILL_CONTROL_CONTAINER_PATH,
        expected_timeout_ms=expected_timeout_ms,
    )
    guard.reject(raw, "browser worker-kill interim")
    return BrowserInterim(
        target=target,
        kill_nonce=nonce,
        recovery_nonce=None,
        visible_completed_ns=visible,
        error_observed_ns=None,
        cancel_observed_ns=None,
        kill_wait_requested_ns=observed,
        kill_control_observed_ns=None,
        recovery_wait_requested_ns=None,
        first_target_content_ns=socket["first_target_content_ns"],
        action_prefix=_canonical_record_prefix(
            value["browser_actions"], "browser action prefix"
        ),
        socket_prefix=_canonical_record_prefix(
            value["socket_events"], "browser socket prefix"
        ),
    )


def validate_recovery_interim(
    value: dict[str, Any],
    raw: bytes,
    guard: SecretGuard,
    *,
    kill: BrowserInterim,
    expected_timeout_ms: int,
    screenshot_path: Path,
) -> BrowserInterim:
    exact_keys(
        value,
        {
            "schema_version",
            "record_type",
            "browser_case",
            "observed_monotonic_ns",
            "browser_actions",
            "socket_correlation",
            "socket_events",
            "page_error_count",
            "worker_killed_control",
            "gateway_recovered_control",
        },
        "browser recovery interim",
    )
    if (
        value["schema_version"] != BROWSER_SCHEMA
        or value["record_type"] != "openwebui_failure_gateway_recovery_wait"
        or value["browser_case"] != BROWSER_CASE
    ):
        fail("browser recovery interim identity differs")
    if integer(value["page_error_count"], "browser page-error count") != 0:
        fail("browser recovery interim page-error count differs")
    _first_action, last_action = validate_browser_actions(
        value["browser_actions"], expected_count=5
    )
    observed = decimal_timestamp(
        value["observed_monotonic_ns"], "browser recovery interim timestamp"
    )
    if observed < last_action:
        fail("browser recovery interim precedes its actions")
    socket = validate_socket_events(
        value["socket_events"], allow_recovery=False, require_failure=True
    )
    action_prefix = _canonical_record_prefix(
        value["browser_actions"], "browser action prefix"
    )
    socket_prefix = _canonical_record_prefix(
        value["socket_events"], "browser socket prefix"
    )
    _require_record_prefix(action_prefix, kill.action_prefix, "browser action prefix")
    _require_record_prefix(socket_prefix, kill.socket_prefix, "browser socket prefix")
    correlation = value["socket_correlation"]
    if not isinstance(correlation, dict):
        fail("browser failure correlation is not an object")
    exact_keys(
        correlation,
        {
            "target",
            "error_first_observed_monotonic_ns",
            "cancel_first_observed_monotonic_ns",
            "error_event_count",
            "cancel_event_count",
            "done_after_fault_count",
            "content_after_error_count",
        },
        "browser failure correlation",
    )
    target = _validate_target_identity(correlation["target"], "failure target")
    if target != kill.target:
        fail("browser failure target changed between interims")
    error_ns = decimal_timestamp(
        correlation["error_first_observed_monotonic_ns"], "browser provider error"
    )
    cancel_ns = decimal_timestamp(
        correlation["cancel_first_observed_monotonic_ns"], "browser task cancellation"
    )
    error_count = integer(
        correlation["error_event_count"], "browser provider-error count"
    )
    cancel_count = integer(
        correlation["cancel_event_count"], "browser cancellation count"
    )
    done_count = integer(
        correlation["done_after_fault_count"], "browser post-fault done count"
    )
    content_after_error_count = integer(
        correlation["content_after_error_count"],
        "browser content-after-error count",
    )
    if (
        error_ns != socket["first_target_error_ns"]
        or cancel_ns != socket["first_target_cancel_ns"]
        or error_count != 1
        or cancel_count != 1
        or done_count != 0
        or content_after_error_count != 0
    ):
        fail("browser failure terminal correlation differs")
    kill_requested, kill_observed = _validate_redacted_control(
        value["worker_killed_control"],
        stage="worker_killed",
        expected_path=KILL_CONTROL_CONTAINER_PATH,
        nonce=kill.kill_nonce,
    )
    if kill_requested != kill.kill_wait_requested_ns:
        fail("browser worker-killed control request changed between interims")
    recovery_nonce = _validate_clear_control(
        value["gateway_recovered_control"],
        stage="gateway_recovered",
        expected_path=RECOVERY_CONTROL_CONTAINER_PATH,
        expected_timeout_ms=expected_timeout_ms,
    )
    screenshot = read_regular_exact(
        screenshot_path, "browser failed-state screenshot", 64 * 1024 * 1024
    )
    if not screenshot.startswith(b"\x89PNG\r\n\x1a\n"):
        fail("browser failed-state screenshot is not a PNG")
    screenshot_action = value["browser_actions"][4]
    if screenshot_action["screenshot_sha256"] != sha256_bytes(screenshot):
        fail("browser failed-state screenshot hash differs")
    guard.reject(raw, "browser recovery interim")
    return BrowserInterim(
        target=target,
        kill_nonce=kill.kill_nonce,
        recovery_nonce=recovery_nonce,
        visible_completed_ns=kill.visible_completed_ns,
        error_observed_ns=error_ns,
        cancel_observed_ns=cancel_ns,
        kill_wait_requested_ns=kill.kill_wait_requested_ns,
        kill_control_observed_ns=kill_observed,
        recovery_wait_requested_ns=observed,
        first_target_content_ns=kill.first_target_content_ns,
        action_prefix=action_prefix,
        socket_prefix=socket_prefix,
    )


@dataclasses.dataclass(frozen=True)
class BrowserFinal:
    target: dict[str, Any]
    recovery: dict[str, Any]
    visible_completed_ns: int
    error_observed_ns: int
    cancel_observed_ns: int
    recovery_submit_started_ns: int
    recovery_done_ns: int
    first_target_content_ns: int
    first_recovery_content_ns: int
    screenshot_sha256: str
    action_count: int
    socket_event_count: int
    recovery_control_observed_ns: int


def validate_final_browser(
    value: dict[str, Any],
    raw: bytes,
    summary_path: Path,
    screenshot_path: Path,
    guard: SecretGuard,
    *,
    recovery_interim: BrowserInterim,
) -> BrowserFinal:
    exact_keys(
        value,
        {
            "schema_version",
            "record_type",
            "browser_case",
            "observed_monotonic_ns",
            "browser_actions",
            "socket_correlation",
            "page_error_count",
            "page_errors",
            "socket_events",
            "controls",
            "screenshot",
        },
        "browser failure final summary",
    )
    if (
        value["schema_version"] != BROWSER_SCHEMA
        or value["record_type"] != "openwebui_failure_smoke"
        or value["browser_case"] != BROWSER_CASE
        or value["page_errors"] != []
    ):
        fail("browser failure final identity differs")
    if integer(value["page_error_count"], "browser page-error count") != 0:
        fail("browser failure final page-error count differs")
    _first_action, last_action = validate_browser_actions(
        value["browser_actions"], expected_count=len(FINAL_ACTIONS)
    )
    if (
        decimal_timestamp(value["observed_monotonic_ns"], "browser final timestamp")
        < last_action
    ):
        fail("browser final summary precedes its actions")
    socket = validate_socket_events(
        value["socket_events"], allow_recovery=True, require_failure=True
    )
    action_prefix = _canonical_record_prefix(
        value["browser_actions"], "browser action prefix"
    )
    socket_prefix = _canonical_record_prefix(
        value["socket_events"], "browser socket prefix"
    )
    _require_record_prefix(
        action_prefix, recovery_interim.action_prefix, "browser action prefix"
    )
    _require_record_prefix(
        socket_prefix, recovery_interim.socket_prefix, "browser socket prefix"
    )
    correlation = value["socket_correlation"]
    if not isinstance(correlation, dict):
        fail("browser final correlation is not an object")
    exact_keys(
        correlation,
        {
            "target",
            "error_first_observed_monotonic_ns",
            "cancel_first_observed_monotonic_ns",
            "error_event_count",
            "cancel_event_count",
            "done_after_fault_count",
            "content_after_error_count",
            "recovery",
        },
        "browser final correlation",
    )
    target = _validate_target_identity(correlation["target"], "failure target")
    if target != recovery_interim.target:
        fail("browser final failure target changed")
    error_ns = decimal_timestamp(
        correlation["error_first_observed_monotonic_ns"], "browser final error"
    )
    cancel_ns = decimal_timestamp(
        correlation["cancel_first_observed_monotonic_ns"], "browser final cancellation"
    )
    error_count = integer(
        correlation["error_event_count"], "browser final provider-error count"
    )
    cancel_count = integer(
        correlation["cancel_event_count"], "browser final cancellation count"
    )
    done_after_fault = integer(
        correlation["done_after_fault_count"], "browser final post-fault done count"
    )
    content_after_error = integer(
        correlation["content_after_error_count"],
        "browser final content-after-error count",
    )
    if (
        error_ns != recovery_interim.error_observed_ns
        or cancel_ns != recovery_interim.cancel_observed_ns
        or error_ns != socket["first_target_error_ns"]
        or cancel_ns != socket["first_target_cancel_ns"]
        or error_count != 1
        or cancel_count != 1
        or done_after_fault != 0
        or content_after_error != 0
    ):
        fail("browser final failure correlation differs")
    recovery = correlation["recovery"]
    if not isinstance(recovery, dict):
        fail("browser recovery correlation is not an object")
    exact_keys(
        recovery,
        {
            "chat_id_utf8_bytes",
            "chat_id_sha256",
            "message_id_utf8_bytes",
            "message_id_sha256",
            "submit_completed_monotonic_ns",
            "done_observed_monotonic_ns",
            "done_event_count",
            "cancel_event_count",
            "error_event_count",
        },
        "browser recovery correlation",
    )
    recovery_identity = {
        key: recovery[key]
        for key in (
            "chat_id_utf8_bytes",
            "chat_id_sha256",
            "message_id_utf8_bytes",
            "message_id_sha256",
        )
    }
    _validate_target_identity(recovery_identity, "recovery target")
    if (
        recovery_identity["chat_id_sha256"] != target["chat_id_sha256"]
        or recovery_identity["chat_id_utf8_bytes"] != target["chat_id_utf8_bytes"]
        or recovery_identity["message_id_sha256"] == target["message_id_sha256"]
    ):
        fail("browser recovery is not a new message in the same chat")
    actions = value["browser_actions"]
    recovery_submit_started = decimal_timestamp(
        actions[6]["started_monotonic_ns"], "browser recovery submit start"
    )
    recovery_submit_completed = decimal_timestamp(
        recovery["submit_completed_monotonic_ns"], "browser recovery submit completion"
    )
    recovery_done = decimal_timestamp(
        recovery["done_observed_monotonic_ns"], "browser recovery done"
    )
    recovery_done_count = integer(
        recovery["done_event_count"], "browser recovery done count"
    )
    recovery_cancel_count = integer(
        recovery["cancel_event_count"], "browser recovery cancel count"
    )
    recovery_error_count = integer(
        recovery["error_event_count"], "browser recovery error count"
    )
    if (
        recovery_submit_completed
        != decimal_timestamp(
            actions[6]["completed_monotonic_ns"], "browser recovery action completion"
        )
        or recovery_done != socket["first_recovery_done_ns"]
        or socket["first_recovery_content_ns"] < recovery_submit_started
        or recovery_done_count != 1
        or recovery_cancel_count != 0
        or recovery_error_count != 0
    ):
        fail("browser recovery action or terminal correlation differs")
    controls = value["controls"]
    if not isinstance(controls, dict):
        fail("browser final controls are not an object")
    exact_keys(controls, {"worker_killed", "gateway_recovered"}, "browser controls")
    kill_requested, kill_observed = _validate_redacted_control(
        controls["worker_killed"],
        stage="worker_killed",
        expected_path=KILL_CONTROL_CONTAINER_PATH,
        nonce=recovery_interim.kill_nonce,
    )
    if recovery_interim.recovery_nonce is None:
        fail("browser recovery control nonce is absent")
    recovery_requested, recovery_observed = _validate_redacted_control(
        controls["gateway_recovered"],
        stage="gateway_recovered",
        expected_path=RECOVERY_CONTROL_CONTAINER_PATH,
        nonce=recovery_interim.recovery_nonce,
    )
    if kill_requested != recovery_interim.kill_wait_requested_ns:
        fail("browser worker-killed control request changed")
    if (
        recovery_requested != recovery_interim.recovery_wait_requested_ns
        or kill_observed != recovery_interim.kill_control_observed_ns
        or recovery_submit_started < recovery_observed
    ):
        fail("browser control timing changed between records")
    screenshot = value["screenshot"]
    if not isinstance(screenshot, dict):
        fail("browser screenshot evidence is not an object")
    exact_keys(
        screenshot,
        {"screenshot_file", "screenshot_bytes", "screenshot_sha256"},
        "browser screenshot evidence",
    )
    screenshot_raw = read_regular_exact(
        screenshot_path, "browser failed-state screenshot", 64 * 1024 * 1024
    )
    screenshot_digest = sha256_bytes(screenshot_raw)
    screenshot_bytes = integer(
        screenshot["screenshot_bytes"], "browser screenshot bytes", minimum=1
    )
    if (
        screenshot["screenshot_file"] != f"browser/{SCREENSHOT_NAME}"
        or screenshot_bytes != len(screenshot_raw)
        or screenshot["screenshot_sha256"] != screenshot_digest
        or actions[4]["screenshot_sha256"] != screenshot_digest
    ):
        fail("browser final screenshot evidence differs")
    summary_raw = read_regular_exact(
        summary_path, "browser failure summary artifact", MAX_JSON_LINE_BYTES + 1
    )
    if summary_raw != raw + b"\n":
        fail("browser stdout and summary artifact differ")
    guard.reject(raw, "browser final summary")
    return BrowserFinal(
        target=target,
        recovery=recovery_identity,
        visible_completed_ns=recovery_interim.visible_completed_ns,
        error_observed_ns=error_ns,
        cancel_observed_ns=cancel_ns,
        recovery_submit_started_ns=recovery_submit_started,
        recovery_done_ns=recovery_done,
        first_target_content_ns=socket["first_target_content_ns"],
        first_recovery_content_ns=socket["first_recovery_content_ns"],
        screenshot_sha256=screenshot_digest,
        action_count=len(FINAL_ACTIONS),
        socket_event_count=len(value["socket_events"]),
        recovery_control_observed_ns=recovery_observed,
    )


@dataclasses.dataclass(frozen=True)
class LifecycleEvidence:
    target_request_id: str
    target_completion_id: str
    recovery_request_id: str
    recovery_completion_id: str
    worker_fatal_ns: int
    recovery_admitted_ns: int
    recovery_released_ns: int
    lifecycle_count: int


def _singular_event(events: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [event for event in events if event["event"] == name]
    if len(matches) != 1:
        fail("gateway lifecycle singular event count differs")
    return matches[0]


def validate_failure_lifecycle(
    records: list[JournalLifecycle],
    *,
    initial_gateway_pid: int,
    recovered_gateway_pid: int,
    fault_started_ns: int,
    fault_completed_ns: int,
    browser: BrowserFinal,
) -> LifecycleEvidence:
    if not records or len(records) > 64:
        fail("gateway failure lifecycle count is outside its bound")
    prior_event_ns = -1
    prior_journal_usec = -1
    old: list[dict[str, Any]] = []
    recovered: list[dict[str, Any]] = []
    for record in records:
        event_ns = integer(
            record.event["observed_monotonic_ns"], "gateway lifecycle timestamp"
        )
        if (
            event_ns < prior_event_ns
            or record.journal_monotonic_usec < prior_journal_usec
            or event_ns > record.journal_monotonic_usec * 1000 + 999
        ):
            fail("gateway lifecycle or journal timestamps regress")
        prior_event_ns = event_ns
        prior_journal_usec = record.journal_monotonic_usec
        if record.journal_pid == initial_gateway_pid:
            if recovered:
                fail("initial gateway emitted lifecycle after recovered gateway")
            old.append(record.event)
        elif record.journal_pid == recovered_gateway_pid:
            recovered.append(record.event)
        else:
            fail("gateway lifecycle journal PID is outside the planned epochs")

    if not old or not recovered:
        fail("gateway lifecycle lacks an initial or recovered epoch")
    old_pairs = {
        (event["request_id"], event["completion_id"])
        for event in old
        if event["request_id"] is not None
    }
    recovered_pairs = {
        (event["request_id"], event["completion_id"])
        for event in recovered
        if event["request_id"] is not None
    }
    if len(old_pairs) != 1 or len(recovered_pairs) != 1:
        fail("gateway failure lifecycle request correlation differs")
    old_pair = next(iter(old_pairs))
    recovered_pair = next(iter(recovered_pairs))
    if old_pair == recovered_pair:
        fail("gateway reused request identity after restart")
    if any(
        (event["request_id"], event["completion_id"]) != old_pair for event in old
    ) or any(
        (event["request_id"], event["completion_id"]) != recovered_pair
        for event in recovered
    ):
        fail("idle or foreign lifecycle event occurred during the failure gate")

    old_names = [event["event"] for event in old]
    if (
        old_names[:2] != ["request_admitted", "request_started"]
        or old_names[-2:] != ["request_first_token", "worker_fatal"]
        or not old_names[2:-2]
        or any(name != "request_progress" for name in old_names[2:-2])
    ):
        fail("failure target lifecycle sequence differs")
    admitted = _singular_event(old, "request_admitted")
    started = _singular_event(old, "request_started")
    progress_events = [event for event in old if event["event"] == "request_progress"]
    first_token = _singular_event(old, "request_first_token")
    fatal = _singular_event(old, "worker_fatal")
    if (
        admitted["prompt_tokens"] != started["prompt_tokens"]
        or any(
            event["prompt_tokens"] != admitted["prompt_tokens"]
            for event in progress_events
        )
        or progress_events[-1]["processed_prompt_tokens"] != admitted["prompt_tokens"]
        or any(
            later["processed_prompt_tokens"] <= earlier["processed_prompt_tokens"]
            for earlier, later in zip(
                progress_events, progress_events[1:], strict=False
            )
        )
        or fatal["reason"] != "unexpected worker stdout EOF"
        or fatal["admit_to_fatal_ns"] < started["admit_to_start_ns"]
        or not (
            admitted["observed_monotonic_ns"]
            <= started["observed_monotonic_ns"]
            <= progress_events[0]["observed_monotonic_ns"]
            <= progress_events[-1]["observed_monotonic_ns"]
            <= first_token["observed_monotonic_ns"]
            <= browser.first_target_content_ns
            <= fatal["observed_monotonic_ns"]
        )
    ):
        fail("failure target lifecycle fields or ordering differ")
    if (
        browser.visible_completed_ns > fault_started_ns
        or first_token["observed_monotonic_ns"] > fault_started_ns
        or fatal["observed_monotonic_ns"] < fault_started_ns
        or fatal["observed_monotonic_ns"] > fault_completed_ns + 5_000_000_000
        or browser.error_observed_ns < fatal["observed_monotonic_ns"]
        or browser.cancel_observed_ns < browser.error_observed_ns
    ):
        fail("fault injection, lifecycle, and browser failure timing differs")

    recovered_names = [event["event"] for event in recovered]
    if (
        recovered_names[:2] != ["request_admitted", "request_started"]
        or recovered_names[-2:] != ["request_first_token", "request_released"]
        or not recovered_names[2:-2]
        or any(name != "request_progress" for name in recovered_names[2:-2])
    ):
        fail("recovery lifecycle sequence differs")
    recovery_admitted = _singular_event(recovered, "request_admitted")
    recovery_started = _singular_event(recovered, "request_started")
    recovery_progress_events = [
        event for event in recovered if event["event"] == "request_progress"
    ]
    recovery_first = _singular_event(recovered, "request_first_token")
    recovery_released = _singular_event(recovered, "request_released")
    if (
        recovery_admitted["prompt_tokens"] != recovery_started["prompt_tokens"]
        or any(
            event["prompt_tokens"] != recovery_admitted["prompt_tokens"]
            for event in recovery_progress_events
        )
        or recovery_progress_events[-1]["processed_prompt_tokens"]
        != recovery_admitted["prompt_tokens"]
        or any(
            later["processed_prompt_tokens"] <= earlier["processed_prompt_tokens"]
            for earlier, later in zip(
                recovery_progress_events,
                recovery_progress_events[1:],
                strict=False,
            )
        )
        or recovery_released["prompt_tokens"] != recovery_admitted["prompt_tokens"]
        or recovery_released["completion_tokens"] < 1
        or recovery_released["completion_tokens"]
        > recovery_admitted["max_completion_tokens"]
        or recovery_released["admit_to_start_ns"]
        != recovery_started["admit_to_start_ns"]
        or recovery_released["outcome"] != "stop"
        or recovery_released["cancel_reason"] is not None
        or recovery_released["reset_complete"] is not True
        or not (
            recovery_admitted["observed_monotonic_ns"]
            <= recovery_started["observed_monotonic_ns"]
            <= recovery_progress_events[0]["observed_monotonic_ns"]
            <= recovery_progress_events[-1]["observed_monotonic_ns"]
            <= recovery_first["observed_monotonic_ns"]
            <= recovery_released["observed_monotonic_ns"]
            <= browser.recovery_done_ns
        )
        or recovery_first["observed_monotonic_ns"] > browser.first_recovery_content_ns
        or browser.first_recovery_content_ns > browser.recovery_done_ns
        or recovery_admitted["observed_monotonic_ns"]
        < browser.recovery_submit_started_ns
    ):
        fail("recovery lifecycle fields or browser ordering differ")
    return LifecycleEvidence(
        target_request_id=old_pair[0],
        target_completion_id=old_pair[1],
        recovery_request_id=recovered_pair[0],
        recovery_completion_id=recovered_pair[1],
        worker_fatal_ns=fatal["observed_monotonic_ns"],
        recovery_admitted_ns=recovery_admitted["observed_monotonic_ns"],
        recovery_released_ns=recovery_released["observed_monotonic_ns"],
        lifecycle_count=len(records),
    )


def lifecycle_has_active_fatal(
    records: list[JournalLifecycle], initial_gateway_pid: int
) -> bool:
    return any(
        record.journal_pid == initial_gateway_pid
        and record.event["event"] == "worker_fatal"
        and record.event["request_id"] is not None
        for record in records
    )


def lifecycle_has_recovery_release(
    records: list[JournalLifecycle], recovered_gateway_pid: int
) -> bool:
    return any(
        record.journal_pid == recovered_gateway_pid
        and record.event["event"] == "request_released"
        for record in records
    )


def validate_prefault_lifecycle(
    records: list[JournalLifecycle], initial_gateway_pid: int
) -> None:
    if not records or len(records) > 16:
        fail("pre-fault gateway lifecycle count is outside its bound")
    if any(record.journal_pid != initial_gateway_pid for record in records):
        fail("foreign gateway PID appeared before fault injection")
    prior_event_ns = -1
    prior_journal_usec = -1
    for record in records:
        event_ns = integer(
            record.event["observed_monotonic_ns"], "pre-fault lifecycle timestamp"
        )
        if (
            event_ns < prior_event_ns
            or record.journal_monotonic_usec < prior_journal_usec
            or event_ns > record.journal_monotonic_usec * 1000 + 999
        ):
            fail("pre-fault lifecycle or journal timestamps regress")
        prior_event_ns = event_ns
        prior_journal_usec = record.journal_monotonic_usec
    events = [record.event for record in records]
    pairs = {(event["request_id"], event["completion_id"]) for event in events}
    if len(pairs) != 1 or (None, None) in pairs:
        fail("pre-fault gateway lifecycle request correlation differs")
    names = [event["event"] for event in events]
    if (
        names[:2] != ["request_admitted", "request_started"]
        or names[-1] != "request_first_token"
        or not names[2:-1]
        or any(name != "request_progress" for name in names[2:-1])
    ):
        fail("pre-fault gateway lifecycle sequence differs")
    admitted = _singular_event(events, "request_admitted")
    started = _singular_event(events, "request_started")
    progress = [event for event in events if event["event"] == "request_progress"]
    if (
        started["prompt_tokens"] != admitted["prompt_tokens"]
        or any(
            event["prompt_tokens"] != admitted["prompt_tokens"] for event in progress
        )
        or progress[-1]["processed_prompt_tokens"] != admitted["prompt_tokens"]
        or any(
            later["processed_prompt_tokens"] <= earlier["processed_prompt_tokens"]
            for earlier, later in zip(progress, progress[1:], strict=False)
        )
    ):
        fail("pre-fault gateway lifecycle fields differ")


def fsync_bundle_tree(root: Path) -> None:
    directories: list[Path] = []
    for current, child_directories, files in os.walk(root, topdown=False):
        directory = Path(current)
        directories.append(directory)
        for name in files:
            path = directory / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                fail("failure gate bundle contains a non-regular file")
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for name in child_directories:
            path = directory / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                fail("failure gate bundle contains a non-directory")
    for directory in directories:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def best_effort_remove_container(docker: str, name: str) -> None:
    try:
        subprocess.run(
            [docker, "rm", "--force", name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
            start_new_session=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


GATE_SOURCE_RAW = read_regular_exact(
    Path(__file__).resolve(), "failure gate source", 4 * 1024 * 1024
)


def execute(args: argparse.Namespace) -> None:
    output = AtomicRunDirectory(args.output_dir)
    journal: JournalFollower | None = None
    journal_process: subprocess.Popen[bytes] | None = None
    browser: BrowserProcess | None = None
    browser_process: subprocess.Popen[bytes] | None = None
    journal_writer: AtomicLineWriter | None = None
    browser_writer: AtomicLineWriter | None = None
    pidfd: int | None = None
    browser_name = f"ullm-failure-browser-{os.getpid()}-{os.urandom(8).hex()}"
    initial_probe_name = f"ullm-failure-ready-a-{os.getpid()}-{os.urandom(6).hex()}"
    recovery_probe_name = f"ullm-failure-ready-b-{os.getpid()}-{os.urandom(6).hex()}"
    deadline_ns = time.monotonic_ns() + args.timeout_seconds * 1_000_000_000
    try:
        script_raw = read_regular_exact(
            args.browser_script, "browser failure script", 2 * 1024 * 1024
        )
        runner_raw = GATE_SOURCE_RAW
        token = read_regular_exact(args.token_file, "OpenWebUI token file", 65_536)
        script = output.stage / "runtime" / "browser-failure-smoke.cjs"
        token_file = output.stage / "runtime" / "openwebui-token"
        browser_container_output = (
            output.stage / "runtime" / BROWSER_CONTAINER_OUTPUT_DIR_NAME
        )
        try:
            browser_container_output.mkdir(mode=0o700)
        except OSError:
            fail("failed to create isolated browser output staging")
        write_private_snapshot(script, script_raw, "browser failure script")
        write_private_snapshot(token_file, token, "OpenWebUI token")
        try:
            token_text = token.decode("utf-8", errors="strict")
        except UnicodeError:
            fail("OpenWebUI token is not UTF-8")
        if token_text.endswith("\n"):
            token_text = token_text[:-1]
        if (
            not token_text
            or token_text.strip() != token_text
            or any(character in token_text for character in "\r\n\0")
        ):
            fail("OpenWebUI token is not one strict line")
        openwebui_url = normalized_openwebui_url(args.openwebui_url)
        ready_url = normalized_ready_url(args.ready_url)
        browser_image, browser_digest = normalized_content_image(args.browser_image)
        probe_image, probe_digest = normalized_content_image(args.probe_image)
        network_identity = query_docker_network(args.docker, args.network)
        guard = SecretGuard(
            [
                token_text.encode("utf-8"),
                openwebui_url.encode("utf-8"),
                ready_url.encode("utf-8"),
                FAILURE_PROMPT.encode("utf-8"),
                RECOVERY_PROMPT.encode("utf-8"),
                RECOVERY_MARKER.encode("utf-8"),
            ]
        )

        try:
            initial_service = query_service_identity(args.systemctl, args.service)
            initial_worker = query_worker_identity(initial_service)
        except TransientServiceState:
            fail("gateway service is not initially stable")
        if os.geteuid() != initial_service.uid:
            fail("failure gate must run as the gateway service user")
        pidfd = open_worker_pidfd(initial_worker)
        try:
            pinned_service = query_service_identity(args.systemctl, args.service)
            pinned_worker = query_worker_identity(pinned_service)
        except TransientServiceState:
            fail("gateway changed while pinning the worker")
        if pinned_service != initial_service or not same_process(
            pinned_worker, initial_worker
        ):
            fail("gateway changed while pinning the worker")
        boot_id = read_boot_id()

        initial_probe_command = build_ready_probe_command(
            docker=args.docker,
            image=probe_image,
            network=args.network,
            ready_url=ready_url,
            timeout_seconds=min(args.recovery_probe_timeout_seconds, 30),
            uid=os.geteuid(),
            gid=os.getegid(),
            name=initial_probe_name,
        )
        initial_ready_started, initial_ready_completed = run_ready_probe(
            initial_probe_command,
            min(args.recovery_probe_timeout_seconds, 30),
        )
        cursor = initial_journal_cursor(args.journalctl, args.service)
        journal_writer = AtomicLineWriter(
            output.stage / "service-journal.raw.jsonl",
            maximum_bytes=MAX_JOURNAL_BYTES,
        )
        browser_writer = AtomicLineWriter(
            output.stage / "browser" / "browser-stdout.jsonl",
            maximum_bytes=16 * 1024 * 1024,
        )
        journal_process = spawn_journal_follower(args.journalctl, args.service, cursor)
        journal = JournalFollower(
            journal_process,
            journal_writer,
            service=args.service,
            boot_id=boot_id,
        )
        journal.start()

        browser_command = build_browser_command(
            docker=args.docker,
            image=browser_image,
            name=browser_name,
            script=script,
            token_file=token_file,
            browser_output=browser_container_output,
            control_dir=output.stage / "control",
            openwebui_url=openwebui_url,
            uid=os.geteuid(),
            gid=os.getegid(),
            control_timeout_ms=args.control_timeout_ms,
        )
        browser_process = spawn_browser(browser_command)
        browser = BrowserProcess(browser_process, browser_writer)
        browser.start()
        kill_raw, kill_value = browser.wait_record(
            "openwebui_failure_worker_kill_wait", deadline_ns
        )
        kill_interim = validate_kill_interim(
            kill_value,
            kill_raw,
            guard,
            expected_timeout_ms=args.control_timeout_ms,
        )
        journal.wait_for(
            lambda records: any(
                record.journal_pid == initial_service.main_pid
                and record.event["event"] == "request_first_token"
                for record in records
            ),
            deadline_ns,
            "pre-fault lifecycle first token",
        )
        validate_prefault_lifecycle(journal.snapshot(), initial_service.main_pid)
        if pidfd is None:
            fail("worker pidfd is unavailable")
        fault_started, fault_completed = inject_worker_kill(
            pidfd,
            initial_service,
            initial_worker,
            systemctl=args.systemctl,
        )
        os.close(pidfd)
        pidfd = None
        if (
            fault_started < kill_interim.kill_wait_requested_ns
            or fault_started < kill_interim.visible_completed_ns
        ):
            fail("worker fault injection preceded browser-visible content")
        kill_control_created, kill_control_sha = create_control_file(
            output.stage / "control" / "worker-killed",
            "worker_killed",
            kill_interim.kill_nonce,
        )
        if kill_control_created < fault_completed:
            fail("worker-killed control preceded fault completion")

        recovery_raw, recovery_value = browser.wait_record(
            "openwebui_failure_gateway_recovery_wait", deadline_ns
        )
        recovery_interim = validate_recovery_interim(
            recovery_value,
            recovery_raw,
            guard,
            kill=kill_interim,
            expected_timeout_ms=args.control_timeout_ms,
            screenshot_path=browser_container_output / SCREENSHOT_NAME,
        )
        if (
            recovery_interim.kill_control_observed_ns is None
            or kill_control_created > recovery_interim.kill_control_observed_ns
            or recovery_interim.error_observed_ns is None
            or recovery_interim.error_observed_ns < fault_started
        ):
            fail("worker-killed control or browser failure timing differs")
        journal.wait_for(
            lambda records: lifecycle_has_active_fatal(
                records, initial_service.main_pid
            ),
            deadline_ns,
            "active worker fatal lifecycle",
        )

        recovered_service, recovered_worker = wait_recovered_service(
            systemctl=args.systemctl,
            service=args.service,
            initial_service=initial_service,
            initial_worker=initial_worker,
            deadline_ns=deadline_ns,
        )
        if recovered_service.restarts != initial_service.restarts + 1:
            fail("systemd restart count did not increase exactly once")
        recovery_probe_command = build_ready_probe_command(
            docker=args.docker,
            image=probe_image,
            network=args.network,
            ready_url=ready_url,
            timeout_seconds=args.recovery_probe_timeout_seconds,
            uid=os.geteuid(),
            gid=os.getegid(),
            name=recovery_probe_name,
        )
        recovery_ready_started, recovery_ready_completed = run_ready_probe(
            recovery_probe_command, args.recovery_probe_timeout_seconds
        )
        try:
            ready_service = query_service_identity(args.systemctl, args.service)
            ready_worker = query_worker_identity(ready_service)
        except TransientServiceState:
            fail("gateway changed at the readiness boundary")
        if ready_service != recovered_service or not same_process(
            ready_worker, recovered_worker
        ):
            fail("gateway identity changed during readiness validation")
        if recovery_interim.recovery_nonce is None:
            fail("browser recovery control nonce is absent")
        recovery_control_created, recovery_control_sha = create_control_file(
            output.stage / "control" / "gateway-recovered",
            "gateway_recovered",
            recovery_interim.recovery_nonce,
        )
        if recovery_control_created < recovery_ready_completed:
            fail("gateway-recovered control preceded readiness completion")

        final_raw, final_value = browser.wait_record(
            "openwebui_failure_smoke", deadline_ns
        )
        code = browser.wait_exit(deadline_ns)
        if code != 0 or len(browser.lines) != 3:
            fail("browser failure process exit or stdout count differs")
        browser_final = validate_final_browser(
            final_value,
            final_raw,
            browser_container_output / BROWSER_SUMMARY_NAME,
            browser_container_output / SCREENSHOT_NAME,
            guard,
            recovery_interim=recovery_interim,
        )
        if recovery_control_created > browser_final.recovery_control_observed_ns:
            fail("browser observed gateway recovery before host publication")
        journal.wait_for(
            lambda records: lifecycle_has_recovery_release(
                records, recovered_service.main_pid
            ),
            deadline_ns,
            "recovery request release lifecycle",
        )
        time.sleep(0.5)
        try:
            final_service = query_service_identity(args.systemctl, args.service)
            final_worker = query_worker_identity(final_service)
        except TransientServiceState:
            fail("gateway is not stable after browser recovery")
        if final_service != recovered_service or not same_process(
            final_worker, recovered_worker
        ):
            fail("gateway identity changed after the planned recovery")
        if read_boot_id() != boot_id:
            fail("host boot identity changed during the failure gate")

        journal.stop()
        lifecycle_records = journal.snapshot()
        journal_records = len(journal.records)
        journal_cursors = len(journal.cursors)
        journal_stderr_bytes = journal.stderr_bytes
        journal_stderr_sha = journal.stderr_digest.hexdigest()
        journal = None
        lifecycle = validate_failure_lifecycle(
            lifecycle_records,
            initial_gateway_pid=initial_service.main_pid,
            recovered_gateway_pid=recovered_service.main_pid,
            fault_started_ns=fault_started,
            fault_completed_ns=fault_completed,
            browser=browser_final,
        )

        browser_writer.commit()
        journal_writer.commit()
        browser_writer = None
        journal_writer = None
        snapshot_validated_browser_artifacts(
            browser_container_output,
            output.stage / "browser",
            expected_summary=final_raw,
            expected_screenshot_sha256=browser_final.screenshot_sha256,
        )
        fault_artifact = {
            "schema_version": GATE_SCHEMA,
            "record_type": "fault_injection",
            "injection": "post_header_worker_kill",
            "target_pid": initial_worker.pid,
            "target_starttime_ticks": initial_worker.starttime_ticks,
            "target_parent_pid": initial_worker.parent_pid,
            "signal": "SIGKILL",
            "command": "signal.pidfd_send_signal",
            "started_monotonic_ns": fault_started,
            "completed_monotonic_ns": fault_completed,
        }
        fault_raw = write_atomic_json(
            output.stage / "fault-injection.json", fault_artifact
        )
        readiness_artifact = {
            "schema_version": GATE_SCHEMA,
            "record_type": "readiness_evidence",
            "network_id": network_identity.network_id,
            "subnet": network_identity.subnet,
            "gateway": network_identity.gateway,
            "initial": {
                "started_monotonic_ns": initial_ready_started,
                "completed_monotonic_ns": initial_ready_completed,
                "status": 200,
            },
            "recovered": {
                "started_monotonic_ns": recovery_ready_started,
                "completed_monotonic_ns": recovery_ready_completed,
                "status": 200,
            },
        }
        readiness_raw = write_atomic_json(
            output.stage / "readiness-evidence.json", readiness_artifact
        )

        try:
            (output.stage / "control" / "worker-killed").unlink()
            (output.stage / "control" / "gateway-recovered").unlink()
            (output.stage / "control").rmdir()
            (browser_container_output / SCREENSHOT_NAME).unlink()
            (browser_container_output / BROWSER_SUMMARY_NAME).unlink()
            browser_container_output.rmdir()
            script.unlink()
            token_file.unlink()
            (output.stage / "runtime").rmdir()
        except OSError:
            fail("failed to remove private failure-gate staging")

        dynamic_guard = guard.extend(
            [
                lifecycle.target_request_id,
                lifecycle.target_completion_id,
                lifecycle.recovery_request_id,
                lifecycle.recovery_completion_id,
            ]
        )
        summary = {
            "schema_version": GATE_SCHEMA,
            "passed": True,
            "service": {
                "unit_sha256": sha256_bytes(args.service.encode("utf-8")),
                "initial_gateway_pid": initial_service.main_pid,
                "recovered_gateway_pid": recovered_service.main_pid,
                "initial_worker_pid": initial_worker.pid,
                "recovered_worker_pid": recovered_worker.pid,
                "initial_worker_starttime_ticks": initial_worker.starttime_ticks,
                "recovered_worker_starttime_ticks": recovered_worker.starttime_ticks,
                "initial_restart_count": initial_service.restarts,
                "recovered_restart_count": recovered_service.restarts,
                "restart_delta": 1,
                "boot_id_sha256": sha256_bytes(boot_id.encode("ascii")),
            },
            "browser": {
                "image_reference_sha256": sha256_bytes(browser_image.encode("utf-8")),
                "image_content_digest": browser_digest,
                "script_sha256": sha256_bytes(script_raw),
                "action_count": browser_final.action_count,
                "socket_event_count": browser_final.socket_event_count,
                "screenshot_sha256": browser_final.screenshot_sha256,
                "stdout_lines": 3,
                "stdout_bytes": (output.stage / "browser" / "browser-stdout.jsonl")
                .stat()
                .st_size,
                "stdout_sha256": sha256_bytes(
                    read_regular_exact(
                        output.stage / "browser" / "browser-stdout.jsonl",
                        "browser stdout artifact",
                        16 * 1024 * 1024,
                    )
                ),
                "stderr_bytes": browser.stderr_bytes,
                "stderr_sha256": browser.stderr_digest.hexdigest(),
            },
            "fault": {
                "target_request_sha256": sha256_bytes(
                    lifecycle.target_request_id.encode("utf-8")
                ),
                "target_completion_sha256": sha256_bytes(
                    lifecycle.target_completion_id.encode("utf-8")
                ),
                "worker_fatal_monotonic_ns": lifecycle.worker_fatal_ns,
                "signal_to_fatal_ns": lifecycle.worker_fatal_ns - fault_started,
                "fault_artifact_sha256": sha256_bytes(fault_raw),
                "kill_control_sha256": kill_control_sha,
            },
            "recovery": {
                "request_sha256": sha256_bytes(
                    lifecycle.recovery_request_id.encode("utf-8")
                ),
                "completion_sha256": sha256_bytes(
                    lifecycle.recovery_completion_id.encode("utf-8")
                ),
                "admitted_monotonic_ns": lifecycle.recovery_admitted_ns,
                "released_monotonic_ns": lifecycle.recovery_released_ns,
                "outcome": "stop",
                "reset_complete": True,
                "readiness_artifact_sha256": sha256_bytes(readiness_raw),
                "recovery_control_sha256": recovery_control_sha,
            },
            "gateway_journal": {
                "lifecycle_count": lifecycle.lifecycle_count,
                "record_count": journal_records,
                "cursor_count": journal_cursors,
                "raw_sha256": sha256_bytes(
                    read_regular_exact(
                        output.stage / "service-journal.raw.jsonl",
                        "service journal artifact",
                        MAX_JOURNAL_BYTES,
                    )
                ),
                "stderr_bytes": journal_stderr_bytes,
                "stderr_sha256": journal_stderr_sha,
            },
            "probe": {
                "image_reference_sha256": sha256_bytes(probe_image.encode("utf-8")),
                "image_content_digest": probe_digest,
                "network_id_sha256": sha256_bytes(
                    network_identity.network_id.encode("ascii")
                ),
            },
            "gate_source_sha256": sha256_bytes(runner_raw),
        }
        summary_raw = compact_json(summary)
        dynamic_guard.reject(summary_raw, "failure gate summary")
        write_atomic_json(output.stage / "summary.json", summary)
        for path in (
            output.stage / "browser" / "browser-stdout.jsonl",
            output.stage / "browser" / BROWSER_SUMMARY_NAME,
            output.stage / "service-journal.raw.jsonl",
            output.stage / "fault-injection.json",
            output.stage / "readiness-evidence.json",
            output.stage / "summary.json",
        ):
            guard.scan_file(path, f"failure gate text artifact {path.name}")
        fsync_bundle_tree(output.stage)
        output.publish()
    finally:
        if pidfd is not None:
            try:
                os.close(pidfd)
            except OSError:
                pass
        if browser_process is not None and browser_process.poll() is None:
            try:
                terminate_process_group(browser_process)
            except BaseException:
                pass
        best_effort_remove_container(args.docker, browser_name)
        best_effort_remove_container(args.docker, initial_probe_name)
        best_effort_remove_container(args.docker, recovery_probe_name)
        if journal is not None:
            try:
                journal.stop()
            except BaseException:
                pass
        elif journal_process is not None and journal_process.poll() is None:
            try:
                terminate_process_group(journal_process)
            except BaseException:
                pass
        if browser_writer is not None:
            browser_writer.abort()
        if journal_writer is not None:
            journal_writer.abort()
        output.abort()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the formal OpenWebUI post-header worker-failure gate."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument(
        "--browser-script",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "deploy"
        / "openwebui"
        / "browser-failure-smoke.cjs",
    )
    parser.add_argument("--browser-image", required=True)
    parser.add_argument("--probe-image", required=True)
    parser.add_argument("--openwebui-url", default="http://192.168.0.66:3000/")
    parser.add_argument("--ready-url", default="http://172.20.0.1:8000/readyz")
    parser.add_argument("--network", default="open-webui-network")
    parser.add_argument("--service", default="ullm-openai.service")
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--systemctl", default="systemctl")
    parser.add_argument("--journalctl", default="journalctl")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--control-timeout-ms", type=int, default=180_000)
    parser.add_argument("--recovery-probe-timeout-seconds", type=int, default=180)
    args = parser.parse_args(argv)
    if args.timeout_seconds < 60 or args.timeout_seconds > 1800:
        parser.error("--timeout-seconds must be between 60 and 1800")
    if args.control_timeout_ms < 10_000 or args.control_timeout_ms > 600_000:
        parser.error("--control-timeout-ms must be between 10000 and 600000")
    if (
        args.recovery_probe_timeout_seconds < 10
        or args.recovery_probe_timeout_seconds > 600
    ):
        parser.error("--recovery-probe-timeout-seconds must be between 10 and 600")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        execute(args)
    except FailureGateError as error:
        print(f"OpenWebUI failure gate failed: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("OpenWebUI failure gate failed", file=sys.stderr)
        return 1
    print("OpenWebUI failure gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
