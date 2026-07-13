#!/usr/bin/env python3
"""Close the real OpenWebUI Stop path against gateway lifecycle evidence."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pwd
import re
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, NoReturn


GATE_SCHEMA = "ullm.openwebui.stop_gate.v1"
BROWSER_SCHEMA = "ullm.openwebui.stop_smoke.v1"
CONTROL_SCHEMA = "ullm.openwebui.stop_gateway_release_control.v1"
LIFECYCLE_SCHEMA = "ullm.gateway.lifecycle.v1"
BROWSER_CASE = "openwebui_stop_after_visible_content"
MODEL_ID = os.environ.get("ULLM_MODEL_ID", "ullm-qwen3-14b-sq8")
OBSERVER_SOCKET = Path("/run/ullm/lifecycle-observer.sock")
CONTROL_CONTAINER_PATH = "/run/control/gateway-released"
BROWSER_SCRIPT_CONTAINER_PATH = "/usr/src/app/ullm-browser-stop-smoke.cjs"
SCREENSHOT_NAME = "openwebui-stop-before.png"
BROWSER_SUMMARY_NAME = "openwebui-stop-summary.json"
MAX_JSON_LINE_BYTES = 1024 * 1024
MAX_DATAGRAM_BYTES = 64 * 1024
MAX_OBSERVER_EVENTS = 256
MAX_JOURNAL_LINES = 4096
MAX_JOURNAL_BYTES = 64 * 1024 * 1024
MAX_BROWSER_LINES = 4
MAX_BROWSER_STDERR_BYTES = 4 * 1024 * 1024
COPY_CHUNK_BYTES = 64 * 1024
PROCESS_GRACE_SECONDS = 2.0
DEFAULT_TIMEOUT_SECONDS = 300
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
CONTENT_IMAGE_RE = re.compile(
    r"(?:(?:[A-Za-z0-9][A-Za-z0-9._/:+-]*)@)?sha256:([0-9a-f]{64})\Z"
)
SERVICE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@:-]{0,127}\.service\Z")

STOP_PROMPT = " ".join(
    (
        "Begin with STOP_STREAM_MARKER.",
        "Then write the integers from 1 through 1000, one per line.",
        "Do not summarize and do not stop early.",
    )
)
RECOVERY_MARKER = "STOP_RECOVERY_OK"
RECOVERY_PROMPT = (
    "For this new turn, reply with exactly STOP_RECOVERY_OK and nothing else."
)
FINAL_ACTIONS = (
    "navigate",
    "select_model",
    "submit_chat",
    "wait_visible",
    "click_stop",
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


class StopGateError(RuntimeError):
    """A fail-closed error whose messages never include external values."""


def fail(message: str) -> NoReturn:
    raise StopGateError(message)


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


def regular_file(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError:
        fail(f"{label} is unavailable")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} is not a regular non-symlink file")
    return path.resolve(strict=True)


def read_bounded_file(path: Path, label: str, maximum: int) -> bytes:
    path = regular_file(path, label)
    try:
        with path.open("rb") as handle:
            raw = handle.read(maximum + 1)
    except OSError:
        fail(f"failed to read {label}")
    if len(raw) > maximum:
        fail(f"{label} exceeds its size bound")
    return raw


class SecretGuard:
    def __init__(self, values: list[bytes]):
        self.values = tuple(value for value in values if len(value) >= 4)

    def extend(self, values: list[str]) -> "SecretGuard":
        return SecretGuard([*self.values, *(value.encode("utf-8") for value in values)])

    def reject(self, raw: bytes, label: str) -> None:
        if any(value in raw for value in self.values):
            fail(f"{label} contains forbidden cleartext")

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
        parent = final_path.parent.resolve(strict=True)
        try:
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
            parent_fd = os.open(
                self.final_path.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
            )
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
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
    while time.monotonic() < deadline:
        process.poll()
        if not process_group_exists(group):
            break
        time.sleep(0.02)
    process.poll()
    if process_group_exists(group):
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + PROCESS_GRACE_SECONDS
    while time.monotonic() < deadline:
        process.poll()
        if not process_group_exists(group):
            break
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
        fail("gateway service is not active and running")
    if not values["MainPID"].isdecimal() or int(values["MainPID"]) <= 0:
        fail("gateway service MainPID is invalid")
    if not values["NRestarts"].isdecimal():
        fail("gateway service restart count is invalid")
    user = nonempty_string(values["User"], "gateway service user")
    try:
        account = pwd.getpwnam(user)
    except KeyError:
        fail("gateway service user does not exist")
    return ServiceIdentity(
        unit=unit,
        main_pid=int(values["MainPID"]),
        user=user,
        uid=account.pw_uid,
        gid=account.pw_gid,
        restarts=int(values["NRestarts"]),
    )


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
    exact_keys(
        value,
        {"schema_version", "event", "observed_monotonic_ns"} | LIFECYCLE_FIELDS[event],
        "lifecycle event",
    )
    integer(value["observed_monotonic_ns"], "lifecycle timestamp")
    if event == "worker_fatal":
        fail("gateway worker_fatal occurred during the Stop gate")
    nonempty_string(value["request_id"], "lifecycle request ID")
    nonempty_string(value["completion_id"], "lifecycle completion ID")
    if event in {
        "request_admitted",
        "request_started",
        "request_first_token",
        "request_cancel_requested",
        "request_released",
    }:
        if value["stream"] is not True:
            fail("Stop gate lifecycle request is not streaming")
    if event == "request_admitted":
        integer(value["prompt_tokens"], "admitted prompt tokens", minimum=1)
        integer(value["max_completion_tokens"], "admitted completion tokens", minimum=1)
    elif event == "request_started":
        integer(value["prompt_tokens"], "started prompt tokens", minimum=1)
        integer(value["admit_to_start_ns"], "admit-to-start")
    elif event == "request_progress":
        if value["phase"] != "prefill":
            fail("gateway progress phase differs")
        processed = integer(
            value["processed_prompt_tokens"], "processed tokens", minimum=1
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
        if value["outcome"] == "cancelled":
            nonempty_string(value["cancel_reason"], "gateway release cancel reason")
        elif value["cancel_reason"] is not None:
            fail("normal gateway release carries a cancel reason")
    if compact_json(value) != raw:
        fail("lifecycle payload is not canonical gateway JSON")
    return value


@dataclasses.dataclass
class LifecycleTrace:
    request_id: str
    completion_id: str
    events: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    started: bool = False
    first_token: bool = False
    cancelled: bool = False
    released: bool = False
    last_progress: int = 0

    def event(self, name: str) -> dict[str, Any]:
        matches = [item for item in self.events if item["event"] == name]
        if len(matches) != 1:
            fail("gateway trace singular event count differs")
        return matches[0]


class LifecycleMachine:
    def __init__(self) -> None:
        self.traces: list[LifecycleTrace] = []
        self.active: LifecycleTrace | None = None
        self.max_active = 0
        self.last_timestamp = -1
        self.seen_pairs: set[tuple[str, str]] = set()

    def consume(self, event: dict[str, Any]) -> None:
        observed = event["observed_monotonic_ns"]
        if observed < self.last_timestamp:
            fail("gateway lifecycle timestamps regressed")
        self.last_timestamp = observed
        name = event["event"]
        pair = (event["request_id"], event["completion_id"])
        if name == "request_admitted":
            if self.active is not None or pair in self.seen_pairs:
                fail("gateway admitted an overlapping or duplicate request")
            trace = LifecycleTrace(*pair)
            trace.events.append(event)
            self.traces.append(trace)
            self.active = trace
            self.seen_pairs.add(pair)
            self.max_active = max(self.max_active, 1)
            return
        active = self.active
        if active is None or pair != (active.request_id, active.completion_id):
            fail("gateway lifecycle correlation differs from the active request")
        if name == "request_started":
            if active.started:
                fail("gateway request_started is duplicated")
            active.started = True
        elif name == "request_progress":
            processed = event["processed_prompt_tokens"]
            if (
                not active.started
                or active.first_token
                or active.cancelled
                or processed <= active.last_progress
            ):
                fail("gateway progress ordering differs")
            active.last_progress = processed
        elif name == "request_first_token":
            if not active.started or active.first_token or active.cancelled:
                fail("gateway first-token ordering differs")
            active.first_token = True
        elif name == "request_cancel_requested":
            if not active.started or active.cancelled:
                fail("gateway cancel ordering or count differs")
            active.cancelled = True
        elif name == "request_released":
            if not active.started or active.released:
                fail("gateway release ordering or count differs")
            if (event["outcome"] == "cancelled") != active.cancelled:
                fail("gateway cancel and release outcomes differ")
            active.released = True
            self.active = None
        else:
            fail("gateway lifecycle event is not valid in a request trace")
        active.events.append(event)


@dataclasses.dataclass(frozen=True)
class ObserverRecord:
    raw: bytes
    event: dict[str, Any]
    received_ns: int
    sender_pid: int
    sender_uid: int
    sender_gid: int


class LifecycleObserver:
    def __init__(
        self,
        path: Path,
        expected_pid: int,
        expected_uid: int,
        writer: AtomicLineWriter,
    ):
        self.path = path
        self.expected_pid = expected_pid
        self.expected_uid = expected_uid
        self.writer = writer
        self.socket: socket.socket | None = None
        self.identity: tuple[int, int] | None = None
        self.records: list[ObserverRecord] = []
        self.payloads: set[bytes] = set()
        self.machine = LifecycleMachine()
        self.error: BaseException | None = None
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.thread: threading.Thread | None = None

    def open(self) -> None:
        try:
            parent = self.path.parent.lstat()
        except OSError:
            fail("lifecycle observer parent is unavailable")
        if (
            stat.S_ISLNK(parent.st_mode)
            or not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.geteuid()
            or parent.st_mode & 0o022
        ):
            fail("lifecycle observer parent identity or mode is unsafe")
        try:
            self.path.lstat()
        except FileNotFoundError:
            pass
        except OSError:
            fail("failed to inspect lifecycle observer path")
        else:
            fail("lifecycle observer path already exists")
        kind = socket.SOCK_DGRAM | getattr(socket, "SOCK_CLOEXEC", 0)
        observer = socket.socket(socket.AF_UNIX, kind)
        try:
            observer.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            observer.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            observer.settimeout(0.1)
            observer.bind(os.fspath(self.path))
            os.chmod(self.path, 0o600)
            metadata = self.path.lstat()
            if (
                not stat.S_ISSOCK(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                fail("lifecycle observer socket identity or mode differs")
        except BaseException:
            observer.close()
            try:
                self.path.unlink()
            except OSError:
                pass
            raise
        self.socket = observer
        self.identity = (metadata.st_dev, metadata.st_ino)
        self.thread = threading.Thread(
            target=self._run,
            name="openwebui-stop-lifecycle-observer",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        assert self.socket is not None
        try:
            while True:
                try:
                    payload, ancillary, flags, _ = self.socket.recvmsg(
                        MAX_DATAGRAM_BYTES,
                        socket.CMSG_SPACE(struct.calcsize("3i")),
                    )
                except TimeoutError:
                    if self.stop_event.is_set():
                        return
                    continue
                except OSError:
                    if self.stop_event.is_set():
                        return
                    raise
                received = time.monotonic_ns()
                if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                    fail("lifecycle observer datagram or credentials were truncated")
                credentials = [
                    struct.unpack("3i", data[: struct.calcsize("3i")])
                    for level, kind, data in ancillary
                    if level == socket.SOL_SOCKET
                    and kind == socket.SCM_CREDENTIALS
                    and len(data) >= struct.calcsize("3i")
                ]
                if len(credentials) != 1:
                    fail("lifecycle observer lacks exactly one sender credential")
                sender_pid, sender_uid, sender_gid = credentials[0]
                if sender_pid != self.expected_pid or sender_uid != self.expected_uid:
                    fail("lifecycle observer sender PID or UID differs")
                if not payload or len(payload) >= MAX_DATAGRAM_BYTES:
                    fail("lifecycle observer payload size is invalid")
                event = validate_lifecycle_payload(payload)
                if event["observed_monotonic_ns"] > received:
                    fail("lifecycle event timestamp follows receipt")
                with self.condition:
                    if payload in self.payloads:
                        fail("lifecycle observer payload is duplicated")
                    if len(self.records) >= MAX_OBSERVER_EVENTS:
                        fail("lifecycle observer event count exceeds its bound")
                    self.machine.consume(event)
                    self.writer.write_line(payload)
                    self.payloads.add(payload)
                    self.records.append(
                        ObserverRecord(
                            payload,
                            event,
                            received,
                            sender_pid,
                            sender_uid,
                            sender_gid,
                        )
                    )
                    self.condition.notify_all()
        except BaseException as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()

    def snapshot(self) -> list[ObserverRecord]:
        with self.condition:
            if self.error is not None:
                raise self.error
            return list(self.records)

    def wait_for(
        self, predicate: Callable[[LifecycleMachine], bool], deadline_ns: int
    ) -> None:
        with self.condition:
            while True:
                if self.error is not None:
                    raise self.error
                if predicate(self.machine):
                    return
                remaining = deadline_ns - time.monotonic_ns()
                if remaining <= 0:
                    fail("lifecycle observer condition timed out")
                self.condition.wait(min(0.1, remaining / 1_000_000_000))

    def close(self) -> None:
        self.stop_event.set()
        drain_timed_out = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            drain_timed_out = self.thread.is_alive()
        if self.socket is not None:
            self.socket.close()
        if drain_timed_out and self.thread is not None:
            self.thread.join(timeout=0.2)
        pending_error = self.error
        try:
            metadata = self.path.lstat()
        except FileNotFoundError:
            metadata = None
        except OSError:
            fail("failed to inspect lifecycle observer during cleanup")
        if metadata is not None:
            if self.identity != (metadata.st_dev, metadata.st_ino) or not stat.S_ISSOCK(
                metadata.st_mode
            ):
                fail("lifecycle observer path was replaced")
            try:
                self.path.unlink()
            except OSError:
                fail("failed to remove lifecycle observer socket")
        if drain_timed_out:
            fail("lifecycle observer drain did not terminate")
        if pending_error is not None:
            raise pending_error


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


def validate_journal_record(
    payload: bytes,
    *,
    service: str,
    main_pid: int,
    boot_id: str,
    cursors: set[str],
    lifecycle_payloads: set[bytes],
) -> tuple[str, bytes | None]:
    record = strict_json_object(payload, "journal record")
    required = {
        "__CURSOR",
        "__MONOTONIC_TIMESTAMP",
        "_BOOT_ID",
        "_PID",
        "_SYSTEMD_UNIT",
        "MESSAGE",
    }
    if not required.issubset(record):
        fail("journal record lacks required fields")
    cursor = nonempty_string(record["__CURSOR"], "journal cursor")
    if cursor in cursors:
        fail("journal cursor is duplicated")
    if record["_BOOT_ID"] != boot_id or record["_SYSTEMD_UNIT"] != service:
        fail("journal boot or service identity differs")
    if (
        not str(record["__MONOTONIC_TIMESTAMP"]).isdecimal()
        or not str(record["_PID"]).isdecimal()
    ):
        fail("journal numeric identity is invalid")
    message = record["MESSAGE"]
    if not isinstance(message, str):
        fail("journal MESSAGE is not text")
    lifecycle = lifecycle_payload_from_journal_message(message)
    if lifecycle is not None:
        if int(record["_PID"]) != main_pid:
            fail("journal lifecycle PID differs from gateway MainPID")
        if lifecycle in lifecycle_payloads:
            fail("journal lifecycle payload is duplicated")
    return cursor, lifecycle


def require_correlated_prefix(observer: list[bytes], journal: list[bytes]) -> None:
    for observer_payload, journal_payload in zip(observer, journal):
        if observer_payload != journal_payload:
            fail("observer and journal lifecycle payload bytes differ")


class JournalFollower:
    def __init__(
        self,
        process: subprocess.Popen[bytes],
        writer: AtomicLineWriter,
        *,
        service: str,
        main_pid: int,
        boot_id: str,
    ):
        self.process = process
        self.writer = writer
        self.service = service
        self.main_pid = main_pid
        self.boot_id = boot_id
        self.records: list[bytes] = []
        self.lifecycle: list[bytes] = []
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
                    main_pid=self.main_pid,
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
                        self.lifecycle_payloads.add(lifecycle)
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
                if self.stderr_bytes > MAX_BROWSER_STDERR_BYTES:
                    fail("journal follower stderr exceeds its bound")
                self.stderr_digest.update(chunk)
        except BaseException as error:
            with self.condition:
                if self.error is None:
                    self.error = error
                self.condition.notify_all()

    def wait_correlated(self, observer: LifecycleObserver, deadline_ns: int) -> None:
        self.wait_correlated_records(observer.snapshot, deadline_ns)

    def wait_correlated_records(
        self,
        snapshot: Callable[[], list[ObserverRecord]],
        deadline_ns: int,
    ) -> None:
        while True:
            observed = snapshot()
            with self.condition:
                if self.error is not None:
                    raise self.error
                journal = list(self.lifecycle)
                require_correlated_prefix(
                    [record.raw for record in observed],
                    journal,
                )
                if len(journal) == len(observed):
                    return
                if len(journal) > len(observed):
                    # Logging precedes the best-effort datagram send; give the socket drain time.
                    pass
                remaining = deadline_ns - time.monotonic_ns()
                if remaining <= 0:
                    fail("observer-to-journal correlation timed out")
                self.condition.wait(min(0.1, remaining / 1_000_000_000))

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
    raw = read_bounded_file(Path("/proc/sys/kernel/random/boot_id"), "boot ID", 128)
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
    journalctl: str,
    service: str,
    cursor: str,
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
                if self.stderr_bytes > MAX_BROWSER_STDERR_BYTES:
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


def _safe_mount_path(path: Path, label: str) -> str:
    value = os.fspath(path.resolve(strict=True))
    if "," in value or "\0" in value or "\n" in value or "\r" in value:
        fail(f"{label} path cannot be represented as a Docker mount")
    return value


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
    gateway_wait_ms: int,
) -> list[str]:
    image, _content_digest = normalized_browser_image(image)
    mounts = (
        f"type=bind,src={_safe_mount_path(script, 'browser script')},dst={BROWSER_SCRIPT_CONTAINER_PATH},readonly",
        f"type=bind,src={_safe_mount_path(token_file, 'token file')},dst=/run/secrets/openwebui-token,readonly",
        f"type=bind,src={_safe_mount_path(browser_output, 'browser output')},dst=/output",
        f"type=bind,src={_safe_mount_path(control_dir, 'control directory')},dst=/run/control",
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
            "OPENWEBUI_STOP_SCREENSHOT=/output/openwebui-stop-before.png",
            "--env",
            "OPENWEBUI_STOP_SUMMARY=/output/openwebui-stop-summary.json",
            "--env",
            f"OPENWEBUI_GATEWAY_RELEASE_CONTROL_FILE={CONTROL_CONTAINER_PATH}",
            "--env",
            f"OPENWEBUI_GATEWAY_RELEASE_TIMEOUT_MS={gateway_wait_ms}",
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


def validate_browser_action_sequence(actions: Any, expected: tuple[str, ...]) -> None:
    if not isinstance(actions, list) or len(actions) != len(expected):
        fail("browser action count differs")
    prior_completed = -1
    fields = {
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
    for index, (action, expected_name) in enumerate(
        zip(actions, expected, strict=True)
    ):
        if not isinstance(action, dict):
            fail("browser action is not an object")
        exact_keys(action, fields, "browser action")
        if (
            action["browser_case"] != BROWSER_CASE
            or action["action_index"] != index
            or action["action"] != expected_name
        ):
            fail("browser action ordering differs")
        started = decimal_timestamp(
            action["started_monotonic_ns"], "browser action start"
        )
        completed = decimal_timestamp(
            action["completed_monotonic_ns"], "browser action completion"
        )
        if completed < started or started < prior_completed:
            fail("browser action timestamps overlap or regress")
        prior_completed = completed
        if (
            action["input_sha256"] is not None
            and SHA256_RE.fullmatch(action["input_sha256"]) is None
        ):
            fail("browser action input digest is invalid")
        if (
            action["screenshot_sha256"] is not None
            and SHA256_RE.fullmatch(action["screenshot_sha256"]) is None
        ):
            fail("browser action screenshot digest is invalid")
        result = action["result"]
        if not isinstance(result, dict):
            fail("browser action result is not an object")
        exact_keys(
            result,
            {"visible", "enabled", "text_utf8_bytes", "text_sha256"},
            "browser action result",
        )
        expected_enabled = (
            True
            if expected_name in {"submit_chat", "click_stop", "wait_ready"}
            else None
        )
        text_expected = expected_name in {"wait_visible", "click_stop", "wait_ready"}
        if result["visible"] is not True or result["enabled"] is not expected_enabled:
            fail("browser action visibility or enabled state differs")
        if text_expected:
            integer(result["text_utf8_bytes"], "browser action text bytes", minimum=1)
            if (
                SHA256_RE.fullmatch(
                    nonempty_string(result["text_sha256"], "browser action text digest")
                )
                is None
            ):
                fail("browser action text digest is invalid")
        elif result["text_utf8_bytes"] is not None or result["text_sha256"] is not None:
            fail("browser action unexpectedly carries text evidence")
        if expected_name == "click_stop":
            if (
                action["screenshot_file"] != "browser/openwebui-stop-before.png"
                or action["screenshot_sha256"] is None
            ):
                fail("Stop click screenshot action differs")
        elif (
            action["screenshot_file"] is not None
            or action["screenshot_sha256"] is not None
        ):
            fail("non-click browser action carries screenshot evidence")
    if actions[1]["input_sha256"] != sha256_bytes(MODEL_ID.encode("utf-8")):
        fail("browser model selection digest differs")
    if actions[2]["input_sha256"] != sha256_bytes(STOP_PROMPT.encode("utf-8")):
        fail("browser Stop prompt digest differs")
    if len(actions) == len(FINAL_ACTIONS) and actions[6][
        "input_sha256"
    ] != sha256_bytes(RECOVERY_PROMPT.encode("utf-8")):
        fail("browser recovery prompt digest differs")


def _validate_identity_hashes(value: dict[str, Any], prefix: str) -> None:
    if not isinstance(value, dict):
        fail("browser correlation identity is not an object")
    if (
        integer(value[f"{prefix}_utf8_bytes"], f"{prefix} bytes", minimum=1) < 1
        or SHA256_RE.fullmatch(
            nonempty_string(value[f"{prefix}_sha256"], f"{prefix} digest")
        )
        is None
    ):
        fail("browser correlation identity digest differs")


def validate_socket_events(events: Any, *, final: bool) -> dict[str, Any]:
    if not isinstance(events, list) or not events:
        fail("browser socket event evidence is empty or malformed")
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
    recovery_done = 0
    recovery_cancel = 0
    recovery_content = 0
    target_cancel = 0
    target_content = 0
    target_done = 0
    target_cancel_timestamps: list[int] = []
    prior_timestamp = -1
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            fail("browser socket event is not an object")
        exact_keys(event, fields, "browser socket event")
        if event["sequence"] != index:
            fail("browser socket event sequence differs")
        timestamp = decimal_timestamp(
            event["observed_monotonic_ns"], "browser socket event timestamp"
        )
        if timestamp < prior_timestamp:
            fail("browser socket event timestamps regress")
        prior_timestamp = timestamp
        target = event["correlation_target"]
        if target not in {"cancel_target", "recovery_target"}:
            fail("browser socket event correlation target differs")
        if not final and target != "cancel_target":
            fail("browser interim contains a recovery socket event")
        if not isinstance(event["done"], bool) or not isinstance(
            event["has_error"], bool
        ):
            fail("browser socket event boolean fields differ")
        content_bytes = integer(
            event["content_utf8_bytes"], "browser socket content bytes"
        )
        if content_bytes == 0:
            if event["content_sha256"] is not None:
                fail("empty browser socket content carries a digest")
        elif (
            SHA256_RE.fullmatch(
                nonempty_string(
                    event["content_sha256"], "browser socket content digest"
                )
            )
            is None
        ):
            fail("browser socket content digest differs")
        if event["has_error"]:
            fail("browser socket event contains an error")
        if event["type"] not in {
            "chat:active",
            "chat:completion",
            "chat:outlet",
            "chat:tasks:cancel",
        }:
            fail("browser socket event type differs")
        if event["type"] in {"chat:active", "chat:outlet"} and (
            event["done"] or content_bytes != 0
        ):
            fail("browser state event carries terminal or content state")
        if event["done"] and event["type"] != "chat:completion":
            fail("browser socket non-completion event is terminal")
        if target == "cancel_target" and event["type"] == "chat:tasks:cancel":
            target_cancel += 1
            target_cancel_timestamps.append(timestamp)
        if target == "cancel_target" and event["type"] == "chat:completion":
            target_content += int(content_bytes > 0)
            target_done += int(event["done"])
        if target == "recovery_target" and event["type"] == "chat:tasks:cancel":
            recovery_cancel += 1
        if target == "recovery_target" and event["type"] == "chat:completion":
            recovery_content += int(content_bytes > 0)
            recovery_done += int(event["done"])
    if target_cancel not in {1, 2} or target_content < 1 or target_done != 0:
        fail("browser target socket cancellation count differs")
    if final and (recovery_done != 1 or recovery_cancel != 0 or recovery_content < 1):
        fail("browser recovery socket terminal events differ")
    return {
        "target_cancel_count": target_cancel,
        "target_cancel_first_ns": min(target_cancel_timestamps),
        "recovery_done_count": recovery_done,
    }


def validate_interim(
    value: dict[str, Any],
    guard: SecretGuard,
    *,
    expected_timeout_ms: int | None = None,
) -> str:
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
            "gateway_release_control",
        },
        "browser interim",
    )
    if (
        value["schema_version"] != BROWSER_SCHEMA
        or value["record_type"] != "openwebui_stop_gateway_release_wait"
        or value["browser_case"] != BROWSER_CASE
        or value["page_error_count"] != 0
    ):
        fail("browser interim identity differs")
    decimal_timestamp(value["observed_monotonic_ns"], "browser interim timestamp")
    validate_browser_action_sequence(value["browser_actions"], FINAL_ACTIONS[:6])
    socket_evidence = validate_socket_events(value["socket_events"], final=False)
    correlation = value["socket_correlation"]
    if not isinstance(correlation, dict):
        fail("browser interim correlation is not an object")
    exact_keys(
        correlation,
        {
            "target",
            "click_completed_monotonic_ns",
            "cancel_first_observed_monotonic_ns",
            "cancel_event_count",
            "done_after_click_count",
            "content_after_cancel_count",
        },
        "browser interim correlation",
    )
    if not isinstance(correlation["target"], dict):
        fail("browser interim target identity is not an object")
    exact_keys(
        correlation["target"],
        {
            "chat_id_utf8_bytes",
            "chat_id_sha256",
            "message_id_utf8_bytes",
            "message_id_sha256",
        },
        "browser interim target identity",
    )
    _validate_identity_hashes(correlation["target"], "chat_id")
    _validate_identity_hashes(correlation["target"], "message_id")
    click_completed = decimal_timestamp(
        correlation["click_completed_monotonic_ns"], "Stop click completion"
    )
    cancel_first = decimal_timestamp(
        correlation["cancel_first_observed_monotonic_ns"], "socket cancel"
    )
    if (
        click_completed
        != decimal_timestamp(
            value["browser_actions"][4]["completed_monotonic_ns"],
            "Stop action completion",
        )
        or cancel_first < click_completed
        or cancel_first != socket_evidence["target_cancel_first_ns"]
        or correlation["cancel_event_count"] != socket_evidence["target_cancel_count"]
        or correlation["done_after_click_count"] != 0
        or correlation["content_after_cancel_count"] != 0
    ):
        fail("browser interim socket invariants differ")
    control = value["gateway_release_control"]
    if not isinstance(control, dict):
        fail("browser gateway release control is not an object")
    exact_keys(
        control,
        {
            "control_schema",
            "control_file",
            "nonce",
            "content_utf8_bytes",
            "content_sha256",
            "timeout_ms",
        },
        "browser gateway release control",
    )
    nonce = nonempty_string(control["nonce"], "gateway release nonce")
    content = f"{CONTROL_SCHEMA}:{nonce}\n".encode("ascii", errors="strict")
    if (
        control["control_schema"] != CONTROL_SCHEMA
        or control["control_file"] != CONTROL_CONTAINER_PATH
        or SHA256_RE.fullmatch(nonce) is None
        or control["content_utf8_bytes"] != len(content)
        or control["content_sha256"] != sha256_bytes(content)
        or (
            expected_timeout_ms is not None
            and control["timeout_ms"] != expected_timeout_ms
        )
    ):
        fail("browser gateway release control content differs")
    integer(control["timeout_ms"], "gateway release control timeout", minimum=1)
    if decimal_timestamp(
        value["observed_monotonic_ns"], "browser interim timestamp"
    ) < decimal_timestamp(
        value["browser_actions"][-1]["completed_monotonic_ns"],
        "browser interim final action",
    ):
        fail("browser interim precedes its action evidence")
    guard.reject(compact_json(value), "browser interim")
    return nonce


def create_control_file(path: Path, nonce: str) -> tuple[int, str]:
    content = f"{CONTROL_SCHEMA}:{nonce}\n".encode("ascii", errors="strict")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError:
        fail("failed to create the exclusive gateway release control")
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
        ):
            fail("gateway release control identity or mode differs")
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                fail("gateway release control write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return time.monotonic_ns(), sha256_bytes(content)


def read_regular_exact(path: Path, label: str, maximum: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        fail(f"failed to open {label}")
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
            fail(f"{label} type or size differs")
        raw = b""
        while len(raw) <= maximum:
            chunk = os.read(descriptor, min(COPY_CHUNK_BYTES, maximum + 1 - len(raw)))
            if not chunk:
                break
            raw += chunk
    finally:
        os.close(descriptor)
    if len(raw) > maximum:
        fail(f"{label} exceeds its size bound")
    return raw


def write_private_snapshot(path: Path, raw: bytes, label: str) -> None:
    if not raw:
        fail(f"{label} snapshot is empty")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o400)
    except OSError:
        fail(f"failed to create the {label} snapshot")
    try:
        os.fchmod(descriptor, 0o400)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                fail(f"{label} snapshot write made no progress")
            offset += written
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or metadata.st_nlink != 1
            or metadata.st_size != len(raw)
        ):
            fail(f"{label} snapshot identity or mode differs")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def validate_final_browser(
    value: dict[str, Any],
    interim: dict[str, Any],
    raw_stdout: bytes,
    summary_path: Path,
    screenshot_path: Path,
    guard: SecretGuard,
) -> dict[str, Any]:
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
            "gateway_release_control",
            "screenshot",
        },
        "browser final summary",
    )
    if (
        value["schema_version"] != BROWSER_SCHEMA
        or value["record_type"] != "openwebui_stop_smoke"
        or value["browser_case"] != BROWSER_CASE
        or value["page_error_count"] != 0
        or value["page_errors"] != []
    ):
        fail("browser final identity or page-error state differs")
    final_observed = decimal_timestamp(
        value["observed_monotonic_ns"], "browser final timestamp"
    )
    validate_browser_action_sequence(value["browser_actions"], FINAL_ACTIONS)
    socket_evidence = validate_socket_events(value["socket_events"], final=True)
    if value["browser_actions"][:6] != interim["browser_actions"]:
        fail("browser interim actions differ from final actions")
    interim_events = interim["socket_events"]
    if value["socket_events"][: len(interim_events)] != interim_events:
        fail("browser interim socket events differ from final events")
    correlation = value["socket_correlation"]
    if not isinstance(correlation, dict) or not isinstance(
        correlation.get("recovery"), dict
    ):
        fail("browser final correlation is malformed")
    exact_keys(
        correlation,
        {
            "target",
            "click_started_monotonic_ns",
            "click_completed_monotonic_ns",
            "cancel_first_observed_monotonic_ns",
            "cancel_event_count",
            "done_after_click_count",
            "content_after_cancel_count",
            "recovery",
        },
        "browser final correlation",
    )
    target = correlation["target"]
    recovery = correlation["recovery"]
    if not isinstance(target, dict) or not isinstance(recovery, dict):
        fail("browser final target identities are malformed")
    exact_keys(
        target,
        {
            "chat_id_utf8_bytes",
            "chat_id_sha256",
            "message_id_utf8_bytes",
            "message_id_sha256",
        },
        "browser final target identity",
    )
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
        },
        "browser final recovery identity",
    )
    _validate_identity_hashes(target, "chat_id")
    _validate_identity_hashes(target, "message_id")
    _validate_identity_hashes(recovery, "chat_id")
    _validate_identity_hashes(recovery, "message_id")
    click_started = decimal_timestamp(
        correlation["click_started_monotonic_ns"], "browser Stop click start"
    )
    click_completed = decimal_timestamp(
        correlation["click_completed_monotonic_ns"], "browser Stop click completion"
    )
    cancel_first = decimal_timestamp(
        correlation["cancel_first_observed_monotonic_ns"], "browser Stop socket cancel"
    )
    recovery_submit = decimal_timestamp(
        recovery["submit_completed_monotonic_ns"], "browser recovery submit"
    )
    recovery_done = decimal_timestamp(
        recovery["done_observed_monotonic_ns"], "browser recovery done"
    )
    if (
        target != interim["socket_correlation"]["target"]
        or target["chat_id_sha256"] != recovery["chat_id_sha256"]
        or target["chat_id_utf8_bytes"] != recovery["chat_id_utf8_bytes"]
        or target["message_id_sha256"] == recovery["message_id_sha256"]
        or click_started
        != decimal_timestamp(
            value["browser_actions"][4]["started_monotonic_ns"],
            "Stop action start",
        )
        or click_completed
        != decimal_timestamp(
            value["browser_actions"][4]["completed_monotonic_ns"],
            "Stop action completion",
        )
        or click_completed
        != decimal_timestamp(
            interim["socket_correlation"]["click_completed_monotonic_ns"],
            "interim Stop click completion",
        )
        or cancel_first < click_completed
        or cancel_first != socket_evidence["target_cancel_first_ns"]
        or cancel_first
        != decimal_timestamp(
            interim["socket_correlation"]["cancel_first_observed_monotonic_ns"],
            "interim socket cancel",
        )
        or correlation["cancel_event_count"] != socket_evidence["target_cancel_count"]
        or correlation["cancel_event_count"]
        != interim["socket_correlation"]["cancel_event_count"]
        or recovery_submit
        != decimal_timestamp(
            value["browser_actions"][6]["completed_monotonic_ns"],
            "recovery submit action completion",
        )
        or recovery_done < recovery_submit
        or recovery_done
        > decimal_timestamp(
            value["browser_actions"][8]["completed_monotonic_ns"],
            "recovery ready action completion",
        )
        or recovery["done_event_count"] != 1
        or recovery["cancel_event_count"] != 0
        or correlation["done_after_click_count"] != 0
        or correlation["content_after_cancel_count"] != 0
    ):
        fail("browser same-chat recovery invariants differ")
    control = value["gateway_release_control"]
    if not isinstance(control, dict):
        fail("browser final control evidence is malformed")
    exact_keys(
        control,
        {
            "control_schema",
            "control_file_utf8_bytes",
            "control_file_sha256",
            "nonce_sha256",
            "content_utf8_bytes",
            "content_sha256",
            "requested_monotonic_ns",
            "observed_monotonic_ns",
        },
        "browser final control evidence",
    )
    _validate_identity_hashes(control, "control_file")
    nonce = interim["gateway_release_control"]["nonce"]
    control_requested = decimal_timestamp(
        control["requested_monotonic_ns"], "browser control request"
    )
    control_observed = decimal_timestamp(
        control["observed_monotonic_ns"], "browser control observation"
    )
    if (
        control["control_schema"] != CONTROL_SCHEMA
        or control["control_file_utf8_bytes"]
        != len(CONTROL_CONTAINER_PATH.encode("utf-8"))
        or control["control_file_sha256"]
        != sha256_bytes(CONTROL_CONTAINER_PATH.encode("utf-8"))
        or control["nonce_sha256"] != sha256_bytes(nonce.encode("ascii"))
        or control["content_utf8_bytes"]
        != interim["gateway_release_control"]["content_utf8_bytes"]
        or control["content_sha256"]
        != interim["gateway_release_control"]["content_sha256"]
        or control_requested
        != decimal_timestamp(
            interim["observed_monotonic_ns"], "browser interim timestamp"
        )
        or control_observed < control_requested
    ):
        fail("browser final control evidence differs from interim")
    screenshot = value["screenshot"]
    if not isinstance(screenshot, dict):
        fail("browser screenshot evidence is malformed")
    exact_keys(
        screenshot,
        {"screenshot_file", "screenshot_bytes", "screenshot_sha256"},
        "browser screenshot evidence",
    )
    screenshot_raw = read_regular_exact(
        screenshot_path, "browser screenshot", 64 * 1024 * 1024
    )
    if (
        not screenshot_raw
        or screenshot["screenshot_file"] != "browser/openwebui-stop-before.png"
        or screenshot["screenshot_bytes"] != len(screenshot_raw)
        or screenshot["screenshot_sha256"] != sha256_bytes(screenshot_raw)
        or value["browser_actions"][4]["screenshot_sha256"]
        != screenshot["screenshot_sha256"]
    ):
        fail("browser screenshot evidence differs")
    if final_observed < decimal_timestamp(
        value["browser_actions"][-1]["completed_monotonic_ns"],
        "browser final action completion",
    ):
        fail("browser final summary precedes its action evidence")
    summary_raw = read_regular_exact(
        summary_path, "browser summary file", MAX_JSON_LINE_BYTES + 1
    )
    if summary_raw != raw_stdout + b"\n":
        fail("browser stdout and summary file differ")
    guard.reject(raw_stdout, "browser final summary")
    return {
        "action_count": len(value["browser_actions"]),
        "socket_event_count": len(value["socket_events"]),
        "screenshot_bytes": len(screenshot_raw),
        "screenshot_sha256": sha256_bytes(screenshot_raw),
        "browser_summary_sha256": sha256_bytes(summary_raw),
    }


def validate_gateway_traces(
    machine: LifecycleMachine,
    *,
    click_completed_ns: int,
    control_created_ns: int | None,
    final: bool,
) -> None:
    expected = 2 if final else 1
    if (
        len(machine.traces) != expected
        or machine.active is not None
        or machine.max_active != 1
    ):
        fail("gateway request count, activity, or concurrency differs")
    target = machine.traces[0]
    cancel = target.event("request_cancel_requested")
    release = target.event("request_released")
    if (
        cancel["reason"] != "client_disconnect"
        or cancel["observed_monotonic_ns"] < click_completed_ns
        or release["outcome"] != "cancelled"
        or release["cancel_reason"] != "client_disconnect"
        or release["observed_monotonic_ns"] < cancel["observed_monotonic_ns"]
    ):
        fail("gateway Stop cancellation trace differs")
    if (
        control_created_ns is not None
        and release["observed_monotonic_ns"] > control_created_ns
    ):
        fail("gateway release control preceded the cancelled release")
    if final:
        recovery = machine.traces[1]
        admitted = recovery.event("request_admitted")
        recovered = recovery.event("request_released")
        if (
            (recovery.request_id, recovery.completion_id)
            == (target.request_id, target.completion_id)
            or admitted["observed_monotonic_ns"] < release["observed_monotonic_ns"]
            or control_created_ns is None
            or admitted["observed_monotonic_ns"] < control_created_ns
            or recovered["outcome"] not in {"stop", "length"}
            or recovered["cancel_reason"] is not None
        ):
            fail("gateway recovery trace ordering or outcome differs")


def normalized_url(raw: str) -> str:
    try:
        value = urllib.parse.urlsplit(raw)
    except ValueError:
        fail("OpenWebUI URL is invalid")
    if (
        value.scheme not in {"http", "https"}
        or not value.netloc
        or value.username is not None
        or value.password is not None
        or value.path not in {"", "/"}
        or value.query
        or value.fragment
    ):
        fail("OpenWebUI URL must be a credential-free HTTP origin")
    return urllib.parse.urlunsplit((value.scheme, value.netloc, "", "", ""))


def normalized_browser_image(raw: str) -> tuple[str, str]:
    if not isinstance(raw, str):
        fail("browser image reference is not text")
    match = CONTENT_IMAGE_RE.fullmatch(raw)
    if match is None:
        fail("browser image must be an immutable Docker content digest")
    return raw, f"sha256:{match.group(1)}"


def write_atomic_json(path: Path, value: dict[str, Any], guard: SecretGuard) -> None:
    raw = compact_json(value) + b"\n"
    guard.reject(raw, "gate summary")
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
        fail("failed to publish the gate summary")


def execute(args: argparse.Namespace) -> None:
    output = AtomicRunDirectory(args.output_dir)
    observer: LifecycleObserver | None = None
    journal: JournalFollower | None = None
    browser: BrowserProcess | None = None
    browser_process: subprocess.Popen[bytes] | None = None
    journal_process: subprocess.Popen[bytes] | None = None
    observer_writer: AtomicLineWriter | None = None
    journal_writer: AtomicLineWriter | None = None
    browser_writer: AtomicLineWriter | None = None
    container_name = f"ullm-stop-gate-{os.getpid()}-{os.urandom(8).hex()}"
    deadline_ns = time.monotonic_ns() + args.timeout_seconds * 1_000_000_000
    try:
        script_raw = read_regular_exact(
            args.browser_script,
            "browser script",
            2 * 1024 * 1024,
        )
        token = read_regular_exact(
            args.token_file,
            "OpenWebUI token file",
            65_536,
        )
        script = output.stage / "runtime" / "browser-stop-smoke.cjs"
        token_file = output.stage / "runtime" / "openwebui-token"
        write_private_snapshot(script, script_raw, "browser script")
        write_private_snapshot(token_file, token, "OpenWebUI token")
        try:
            token_text = token.decode("utf-8", errors="strict")
        except UnicodeError:
            fail("OpenWebUI token is not UTF-8")
        if token_text.endswith("\n"):
            token_text = token_text[:-1]
        if not token_text or any(character in token_text for character in "\r\n\0"):
            fail("OpenWebUI token is not one strict line")
        url = normalized_url(args.openwebui_url)
        browser_image, browser_content_digest = normalized_browser_image(
            args.browser_image
        )
        base_guard = SecretGuard(
            [
                token_text.encode("utf-8"),
                url.encode("utf-8"),
                STOP_PROMPT.encode("utf-8"),
                RECOVERY_PROMPT.encode("utf-8"),
                RECOVERY_MARKER.encode("utf-8"),
            ]
        )
        initial_identity = query_service_identity(args.systemctl, args.service)
        if os.geteuid() != initial_identity.uid:
            fail("Stop gate must run as the gateway service user")
        boot_id = read_boot_id()
        cursor = initial_journal_cursor(args.journalctl, args.service)

        observer_writer = AtomicLineWriter(
            output.stage / "observer.raw.jsonl", maximum_bytes=16 * 1024 * 1024
        )
        journal_writer = AtomicLineWriter(
            output.stage / "service-journal.raw.jsonl",
            maximum_bytes=MAX_JOURNAL_BYTES,
        )
        browser_writer = AtomicLineWriter(
            output.stage / "browser" / "browser-stdout.jsonl",
            maximum_bytes=4 * 1024 * 1024,
        )
        observer = LifecycleObserver(
            args.observer_socket,
            initial_identity.main_pid,
            initial_identity.uid,
            observer_writer,
        )
        observer.open()
        journal_process = spawn_journal_follower(args.journalctl, args.service, cursor)
        journal = JournalFollower(
            journal_process,
            journal_writer,
            service=args.service,
            main_pid=initial_identity.main_pid,
            boot_id=boot_id,
        )
        journal.start()

        command = build_browser_command(
            docker=args.docker,
            image=browser_image,
            name=container_name,
            script=script,
            token_file=token_file,
            browser_output=output.stage / "browser",
            control_dir=output.stage / "control",
            openwebui_url=url,
            uid=os.geteuid(),
            gid=os.getegid(),
            gateway_wait_ms=args.gateway_wait_ms,
        )
        browser_process = spawn_browser(command)
        browser = BrowserProcess(browser_process, browser_writer)
        browser.start()

        interim_raw, interim = browser.wait_record(
            "openwebui_stop_gateway_release_wait", deadline_ns
        )
        nonce = validate_interim(
            interim,
            base_guard,
            expected_timeout_ms=args.gateway_wait_ms,
        )
        click_completed = decimal_timestamp(
            interim["socket_correlation"]["click_completed_monotonic_ns"],
            "Stop click completion",
        )
        observer.wait_for(
            lambda machine: len(machine.traces) == 1
            and machine.active is None
            and machine.traces[0].released,
            deadline_ns,
        )
        validate_gateway_traces(
            observer.machine,
            click_completed_ns=click_completed,
            control_created_ns=None,
            final=False,
        )
        journal.wait_correlated(observer, deadline_ns)
        control_created, control_sha = create_control_file(
            output.stage / "control" / "gateway-released", nonce
        )
        validate_gateway_traces(
            observer.machine,
            click_completed_ns=click_completed,
            control_created_ns=control_created,
            final=False,
        )

        final_raw, final_browser = browser.wait_record(
            "openwebui_stop_smoke", deadline_ns
        )
        code = browser.wait_exit(deadline_ns)
        if code != 0 or len(browser.lines) != 2:
            fail("browser process exit or stdout record count differs")
        browser_evidence = validate_final_browser(
            final_browser,
            interim,
            final_raw,
            output.stage / "browser" / BROWSER_SUMMARY_NAME,
            output.stage / "browser" / SCREENSHOT_NAME,
            base_guard,
        )
        observer.wait_for(
            lambda machine: len(machine.traces) == 2
            and machine.active is None
            and machine.traces[1].released,
            deadline_ns,
        )
        validate_gateway_traces(
            observer.machine,
            click_completed_ns=click_completed,
            control_created_ns=control_created,
            final=True,
        )
        journal.wait_correlated(observer, deadline_ns)
        time.sleep(0.2)
        journal.wait_correlated(observer, deadline_ns)
        if len(journal.lifecycle) != len(observer.records):
            fail("observer and journal lifecycle counts differ")

        base_guard.reject(interim_raw, "browser interim stdout")
        base_guard.reject(final_raw, "browser final stdout")

        # Stop the datagram source first, then let journal catch up to that exact set.
        observer.close()
        machine = observer.machine
        observer_records = list(observer.records)
        observer = None
        validate_gateway_traces(
            machine,
            click_completed_ns=click_completed,
            control_created_ns=control_created,
            final=True,
        )
        final_identity = query_service_identity(args.systemctl, args.service)
        if final_identity != initial_identity:
            fail("gateway service identity changed during the Stop gate")
        dynamic_ids = [
            value
            for trace in machine.traces
            for value in (trace.request_id, trace.completion_id)
        ]
        summary_guard = base_guard.extend(dynamic_ids)
        journal.wait_correlated_records(lambda: observer_records, deadline_ns)
        time.sleep(0.2)
        if len(journal.lifecycle) != len(observer_records):
            fail("observer and journal lifecycle counts differ after observer close")
        journal_records = len(journal.records)
        journal_cursors = len(journal.cursors)
        journal_stderr_bytes = journal.stderr_bytes
        journal_stderr_sha256 = journal.stderr_digest.hexdigest()
        journal.stop()
        journal = None
        browser_writer.commit()
        observer_writer.commit()
        journal_writer.commit()
        try:
            (output.stage / "control" / "gateway-released").unlink()
            (output.stage / "control").rmdir()
            script.unlink()
            token_file.unlink()
            (output.stage / "runtime").rmdir()
        except OSError:
            fail("failed to remove private runtime staging")

        summary = {
            "schema_version": GATE_SCHEMA,
            "passed": True,
            "service": {
                "unit_sha256": sha256_bytes(args.service.encode("utf-8")),
                "main_pid_sha256": sha256_bytes(
                    str(initial_identity.main_pid).encode("ascii")
                ),
                "user_uid_sha256": sha256_bytes(
                    str(initial_identity.uid).encode("ascii")
                ),
                "restart_count": initial_identity.restarts,
            },
            "browser": {
                "image_sha256": sha256_bytes(browser_image.encode("utf-8")),
                "image_content_digest": browser_content_digest,
                "script_sha256": sha256_bytes(script_raw),
                **browser_evidence,
                "stdout_lines": browser_writer.lines_written,
                "stdout_sha256": browser_writer.sha256,
                "stderr_bytes": browser.stderr_bytes,
                "stderr_sha256": browser.stderr_digest.hexdigest(),
            },
            "gateway": {
                "request_count": len(machine.traces),
                "maximum_active_requests": machine.max_active,
                "cancel_reason": "client_disconnect",
                "target_outcome": "cancelled",
                "recovery_outcome": machine.traces[1].event("request_released")[
                    "outcome"
                ],
                "target_request_sha256": sha256_bytes(
                    machine.traces[0].request_id.encode("utf-8")
                ),
                "target_completion_sha256": sha256_bytes(
                    machine.traces[0].completion_id.encode("utf-8")
                ),
                "recovery_request_sha256": sha256_bytes(
                    machine.traces[1].request_id.encode("utf-8")
                ),
                "recovery_completion_sha256": sha256_bytes(
                    machine.traces[1].completion_id.encode("utf-8")
                ),
                "control_content_sha256": control_sha,
            },
            "artifacts": {
                "observer": {
                    "file": "observer.raw.jsonl",
                    "bytes": observer_writer.bytes_written,
                    "records": observer_writer.lines_written,
                    "sha256": observer_writer.sha256,
                },
                "journal": {
                    "file": "service-journal.raw.jsonl",
                    "bytes": journal_writer.bytes_written,
                    "records": journal_records,
                    "sha256": journal_writer.sha256,
                    "unique_cursors": journal_cursors,
                    "stderr_bytes": journal_stderr_bytes,
                    "stderr_sha256": journal_stderr_sha256,
                },
            },
        }
        summary_guard.reject(compact_json(summary), "gate summary")
        write_atomic_json(output.stage / "summary.json", summary, summary_guard)
        for path in (
            output.stage / "observer.raw.jsonl",
            output.stage / "service-journal.raw.jsonl",
        ):
            base_guard.scan_file(path, "credential-safe raw artifact")
        for path in (
            output.stage / "summary.json",
            output.stage / "browser" / "browser-stdout.jsonl",
            output.stage / "browser" / BROWSER_SUMMARY_NAME,
        ):
            summary_guard.scan_file(path, "redacted pass artifact")
        output.publish()
    except BaseException:
        if browser_process is not None:
            try:
                run_bounded_command(
                    [args.docker, "rm", "--force", container_name],
                    "browser container cleanup",
                    timeout_seconds=15.0,
                )
            except StopGateError:
                pass
            if browser_process.poll() is None:
                try:
                    terminate_process_group(browser_process)
                except StopGateError:
                    pass
        if journal is not None:
            try:
                journal.stop()
            except StopGateError:
                pass
        elif journal_process is not None and journal_process.poll() is None:
            try:
                terminate_process_group(journal_process)
            except StopGateError:
                pass
        if observer is not None:
            try:
                observer.close()
            except StopGateError:
                pass
        for writer in (browser_writer, observer_writer, journal_writer):
            if writer is not None:
                writer.abort()
        output.abort()
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument(
        "--browser-image",
        required=True,
        help="immutable sha256:<digest> or name@sha256:<digest>",
    )
    parser.add_argument("--openwebui-url", required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument(
        "--browser-script",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "deploy"
        / "openwebui"
        / "browser-stop-smoke.cjs",
    )
    parser.add_argument("--observer-socket", type=Path, default=OBSERVER_SOCKET)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--systemctl", default="systemctl")
    parser.add_argument("--journalctl", default="journalctl")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        choices=range(60, 601),
        metavar="[60-600]",
    )
    parser.add_argument(
        "--gateway-wait-ms",
        type=int,
        default=30_000,
        choices=range(1_000, 60_001),
        metavar="[1000-60000]",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        execute(parse_args(argv))
        return 0
    except KeyboardInterrupt:
        print("OpenWebUI Stop gate interrupted", file=sys.stderr)
        return 130
    except Exception:
        print("OpenWebUI Stop gate failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
