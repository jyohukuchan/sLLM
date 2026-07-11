#!/usr/bin/env python3
"""Run the four direct SQ8 client-disconnect cancellation gates.

The lifecycle observer is the low-latency trigger.  The service journal remains
the authoritative record and every observer payload must later match one
journal MESSAGE byte-for-byte.  The actual HTTP client runs in the fixed
OpenWebUI Docker network and is the only process that handles the API key.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import dataclasses
import errno
import hashlib
import json
import os
import pwd
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, NoReturn, Sequence, cast


MAX_SUPPORT_BYTES = 8 * 1024 * 1024
MAX_GATE_FILE_BYTES = 8 * 1024 * 1024
MAX_RAW_BYTES = 64 * 1024 * 1024
MAX_RAW_LINES = 4096
MAX_HTTP_RESPONSE_BYTES = 4 * 1024 * 1024
RELEASE_DEADLINE_NS = 5_000_000_000
REQUEST_TIMEOUT_NS = 300_000_000_000
QUIET_DRAIN_NS = 250_000_000
GATE_SCHEMA = "ullm.sq8.direct_cancel_gate.v1"
HTTP_COMMAND_SCHEMA = "ullm.sq8.openwebui_http_client.command.v1"
HTTP_EVENT_SCHEMA = "ullm.sq8.openwebui_http_client.event.v1"
MODEL_ID = "ullm-qwen3-14b-sq8"
SERVICE_UNIT = "ullm-openai.service"
OBSERVER_SOCKET = Path("/run/ullm/lifecycle-observer.sock")
HTTP_TARGET = "/v1/chat/completions"
HTTP_NETWORK_NAME = "open-webui-network"
HTTP_READY_URL = "http://172.20.0.1:8000/readyz"
HTTP_CLIENT_SHA256 = "a64642a0f31bcdd92cf02883e195ee270b9752ee6117908b789cc66187053285"
HTTP_NETWORK_ID_RE = re.compile(r"[0-9a-f]{64}\Z")
CONTENT_IMAGE_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
FIXTURE_SCHEMA = "ullm.sq8.chat_template_fixture.v1"
FIXTURE_IDENTITIES = {
    "exact-p0032": (
        32,
        "c660c7fb3c25d2a3e25693e2beb2abc10295a06935772d17d23cedab04f24c07",
    ),
    "exact-p3584": (
        3584,
        "e3cd6c722302f73d688492b73a182298f34cc0a1498def209c262e5e9aa92912",
    ),
}


class GateError(RuntimeError):
    """A fail-closed diagnostic that never embeds external values."""


def fail(message: str) -> NoReturn:
    raise GateError(message)


def _single_fd_snapshot(
    path: Path, label: str, maximum: int
) -> tuple[bytes, tuple[int, ...]]:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            fail(f"{label} is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if (
            identity
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or len(raw) != before.st_size
        ):
            fail(f"{label} changed while it was read")
        return raw, identity
    except GateError:
        raise
    except OSError:
        fail(f"failed to read {label} without following links")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _load_collector_support() -> tuple[types.ModuleType, bytes, tuple[int, ...]]:
    path = Path(__file__).with_name("collect-sq8-openwebui-release.py")
    raw, identity = _single_fd_snapshot(path, "collector support", MAX_SUPPORT_BYTES)
    name = "_ullm_sq8_cancel_collector_support"
    module = types.ModuleType(name)
    module.__file__ = os.fspath(path)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        code = compile(raw, os.fspath(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, raw, identity


COL, COLLECTOR_SUPPORT_RAW, COLLECTOR_SUPPORT_IDENTITY = _load_collector_support()


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
        fail("failed to serialize bounded canonical JSON")


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing a raced destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        operation = libc.renameat2
    except AttributeError:
        fail("renameat2 is unavailable for exclusive output publication")
    operation.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    operation.restype = ctypes.c_int
    result = operation(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            fail("output destination appeared before exclusive publication")
        raise OSError(error, "renameat2 failed")


@dataclasses.dataclass(frozen=True)
class PhaseSpec:
    phase: str
    fixture_id: str
    prompt_tokens: int
    trigger_progress: int | None
    allow_content: bool
    auto_close: bool


PHASE_ORDER = (
    "after_started_before_progress",
    "prefill_after_128",
    "prefill_after_2048",
    "decode_after_first_content",
)
PHASE_SPECS = {
    "after_started_before_progress": PhaseSpec(
        "after_started_before_progress", "exact-p3584", 3584, None, False, False
    ),
    "prefill_after_128": PhaseSpec(
        "prefill_after_128", "exact-p3584", 3584, 128, False, False
    ),
    "prefill_after_2048": PhaseSpec(
        "prefill_after_2048", "exact-p3584", 3584, 2048, False, False
    ),
    "decode_after_first_content": PhaseSpec(
        "decode_after_first_content", "exact-p0032", 32, None, True, True
    ),
}


def validate_phase_order(phases: Sequence[str]) -> None:
    if tuple(phases) != PHASE_ORDER:
        fail("direct cancellation phase order differs")


def _validate_event_identity(
    event: dict[str, Any], request_id: str, completion_id: str
) -> None:
    if (
        event.get("request_id") != request_id
        or event.get("completion_id") != completion_id
    ):
        fail("lifecycle request or completion identity differs")


class CancellationTraceValidator:
    """Validate one cancelled target independently of journal transport."""

    def __init__(
        self, spec: PhaseSpec | str, expected_prompt_tokens: int | None = None
    ):
        if isinstance(spec, str):
            try:
                spec = PHASE_SPECS[spec]
            except KeyError:
                fail("unknown direct cancellation phase")
        self.spec = spec
        self.expected_prompt_tokens = (
            spec.prompt_tokens
            if expected_prompt_tokens is None
            else expected_prompt_tokens
        )
        if self.expected_prompt_tokens != spec.prompt_tokens:
            fail("cancellation fixture prompt length differs")
        self.events: list[dict[str, Any]] = []
        self.request_id: str | None = None
        self.completion_id: str | None = None
        self.started = False
        self.progress: list[int] = []
        self.first_token = False
        self.cancel: dict[str, Any] | None = None
        self.release: dict[str, Any] | None = None
        self.close_ns: int | None = None
        self.content: list[tuple[int, int]] = []
        self.last_timestamp = -1

    def consume(self, event: dict[str, Any]) -> None:
        value = COL.validate_lifecycle_value(dict(event))
        observed = value["observed_monotonic_ns"]
        if observed < self.last_timestamp:
            fail("cancellation lifecycle timestamps regressed")
        self.last_timestamp = observed
        name = value["event"]
        if name == "worker_fatal":
            fail("worker_fatal occurred during direct cancellation")
        if self.release is not None:
            fail("lifecycle event follows cancellation release")
        if name == "request_admitted":
            if self.request_id is not None or self.events:
                fail("cancellation admission is duplicated or out of order")
            if (
                value["prompt_tokens"] != self.expected_prompt_tokens
                or value["max_completion_tokens"] != 512
                or value["stream"] is not True
            ):
                fail("cancellation admission request shape differs")
            self.request_id = value["request_id"]
            self.completion_id = value["completion_id"]
        else:
            if self.request_id is None or self.completion_id is None:
                fail("cancellation lifecycle event precedes admission")
            _validate_event_identity(value, self.request_id, self.completion_id)
            if value.get("stream", True) is not True:
                fail("cancellation lifecycle stream flag differs")
        if name == "request_started":
            if self.started or len(self.events) != 1:
                fail("cancellation start is duplicated or out of order")
            if value["prompt_tokens"] != self.expected_prompt_tokens:
                fail("cancellation started prompt length differs")
            self.started = True
        elif name == "request_progress":
            if not self.started or self.first_token or self.cancel is not None:
                fail("cancellation progress ordering differs")
            processed = value["processed_prompt_tokens"]
            expected_next = min(
                128 if not self.progress else self.progress[-1] + 128,
                self.expected_prompt_tokens,
            )
            if value["phase"] != "prefill" or processed != expected_next:
                fail("cancellation progress sequence differs")
            if value["prompt_tokens"] != self.expected_prompt_tokens:
                fail("cancellation progress prompt length differs")
            if self.spec.phase == "after_started_before_progress":
                fail("after-start cancellation overshot into progress")
            if (
                self.spec.trigger_progress is not None
                and processed > self.spec.trigger_progress
            ):
                fail("prefill cancellation overshot its trigger boundary")
            self.progress.append(processed)
        elif name == "request_first_token":
            if not self.started or self.first_token or self.cancel is not None:
                fail("cancellation first-token ordering differs")
            if not self.spec.allow_content:
                fail("prefill cancellation overshot into first token")
            if self.progress != [self.expected_prompt_tokens]:
                fail("decode first token preceded its complete prompt progress")
            self.first_token = True
        elif name == "request_cancel_requested":
            if not self.started or self.cancel is not None:
                fail("cancellation request ordering or count differs")
            if value["reason"] != "client_disconnect":
                fail("cancellation reason differs from client_disconnect")
            if not self._trigger_reached():
                fail("cancellation occurred before its frozen trigger")
            if not self.spec.auto_close and self.close_ns is None:
                fail("gateway cancellation preceded the explicit client close")
            if self.close_ns is not None and observed < self.close_ns:
                fail("gateway cancellation timestamp precedes client close")
            self.cancel = value
        elif name == "request_released":
            if self.cancel is None or self.release is not None:
                fail("cancellation release ordering or count differs")
            if (
                value["outcome"] != "cancelled"
                or value["cancel_reason"] != "client_disconnect"
                or value["reset_complete"] is not True
                or value["prompt_tokens"] != self.expected_prompt_tokens
                or value["admit_to_release_ns"]
                != value["admit_to_start_ns"] + value["start_to_release_ns"]
            ):
                fail("cancellation release result differs")
            if self.spec.allow_content:
                if value["completion_tokens"] < 1:
                    fail("decode cancellation lacks its first completion token")
            elif value["completion_tokens"] != 0:
                fail("prefill cancellation produced completion tokens")
            absolute_delay = observed - self.cancel["observed_monotonic_ns"]
            stored_delay = (
                value["admit_to_release_ns"] - self.cancel["admit_to_cancel_ns"]
            )
            if (
                absolute_delay < 0
                or absolute_delay > RELEASE_DEADLINE_NS
                or stored_delay < 0
                or stored_delay > RELEASE_DEADLINE_NS
            ):
                fail("cancellation release exceeded the five-second deadline")
            self.release = value
        elif name != "request_admitted":
            fail("unsupported lifecycle event in cancellation trace")
        self.events.append(value)

    def _trigger_reached(self) -> bool:
        if self.spec.phase == "after_started_before_progress":
            return self.started and not self.progress and not self.first_token
        if self.spec.trigger_progress is not None:
            return (
                bool(self.progress) and self.progress[-1] == self.spec.trigger_progress
            )
        return self.first_token

    def trigger_reached(self) -> bool:
        return self._trigger_reached()

    def observe_content(self, timestamp_ns: int, chunk_index: int = 0) -> None:
        if type(timestamp_ns) is not int or timestamp_ns < 0:
            fail("HTTP content timestamp is invalid")
        if type(chunk_index) is not int or chunk_index < 0:
            fail("HTTP content chunk index is invalid")
        if self.content and chunk_index < self.content[-1][1]:
            fail("HTTP content chunk indices regressed")
        self.content.append((timestamp_ns, chunk_index))

    def mark_close(self, timestamp_ns: int) -> None:
        if (
            type(timestamp_ns) is not int
            or timestamp_ns < 0
            or self.close_ns is not None
        ):
            fail("client close boundary is invalid or duplicated")
        if not self._trigger_reached() and not self.spec.auto_close:
            fail("explicit client close preceded its lifecycle trigger")
        if not self.spec.auto_close and timestamp_ns < self._trigger_timestamp():
            fail("explicit client close timestamp precedes its trigger")
        self.close_ns = timestamp_ns

    def finalize(self) -> dict[str, Any]:
        if self.cancel is None or self.release is None or self.close_ns is None:
            fail("cancellation trace is incomplete")
        if not self.spec.allow_content:
            if self.content:
                fail("prefill cancellation exposed non-empty response content")
        else:
            if not self.content:
                fail("decode cancellation lacks client-visible non-empty content")
            first_timestamp = self.content[0][0]
            if self.close_ns < first_timestamp:
                fail("decode client close preceded its first content")
            cancel_timestamp = self.cancel["observed_monotonic_ns"]
            if first_timestamp > cancel_timestamp:
                fail("decode cancellation preceded client-visible content")
            if any(timestamp > cancel_timestamp for timestamp, _ in self.content):
                fail("response content appeared after cancellation")
            trigger_chunk = self.content[0][1]
            if any(chunk_index != trigger_chunk for _, chunk_index in self.content):
                fail("response content appeared in a chunk after the decode trigger")
        return {
            "phase": self.spec.phase,
            "request_id": self.request_id,
            "completion_id": self.completion_id,
            "trigger_observed_monotonic_ns": self._trigger_timestamp(),
            "client_close_monotonic_ns": self.close_ns,
            "cancel_observed_monotonic_ns": self.cancel["observed_monotonic_ns"],
            "release_observed_monotonic_ns": self.release["observed_monotonic_ns"],
            "progress": list(self.progress),
            "completion_tokens": self.release["completion_tokens"],
        }

    def _trigger_timestamp(self) -> int:
        if self.spec.phase == "after_started_before_progress":
            return int(
                next(
                    event["observed_monotonic_ns"]
                    for event in self.events
                    if event["event"] == "request_started"
                )
            )
        if self.spec.trigger_progress is not None:
            return int(
                next(
                    event["observed_monotonic_ns"]
                    for event in self.events
                    if event["event"] == "request_progress"
                    and event["processed_prompt_tokens"] == self.spec.trigger_progress
                )
            )
        return self.content[0][0]


class RecoveryTraceValidator:
    def __init__(self, expected_prompt_tokens: int = 32):
        self.expected_prompt_tokens = expected_prompt_tokens
        self.events: list[dict[str, Any]] = []
        self.request_id: str | None = None
        self.completion_id: str | None = None
        self.started = False
        self.first_token = False
        self.last_progress = 0
        self.release: dict[str, Any] | None = None
        self.last_timestamp = -1

    def consume(self, event: dict[str, Any]) -> None:
        value = COL.validate_lifecycle_value(dict(event))
        observed = value["observed_monotonic_ns"]
        if observed < self.last_timestamp:
            fail("recovery lifecycle timestamps regressed")
        self.last_timestamp = observed
        name = value["event"]
        if name == "worker_fatal" or self.release is not None:
            fail("fatal or trailing lifecycle event occurred during recovery")
        if name == "request_admitted":
            if self.request_id is not None or self.events:
                fail("recovery admission is duplicated or out of order")
            if (
                value["prompt_tokens"] != self.expected_prompt_tokens
                or value["max_completion_tokens"] != 2
                or value["stream"] is not True
            ):
                fail("recovery admission request shape differs")
            self.request_id = value["request_id"]
            self.completion_id = value["completion_id"]
        else:
            if self.request_id is None or self.completion_id is None:
                fail("recovery lifecycle event precedes admission")
            _validate_event_identity(value, self.request_id, self.completion_id)
        if name == "request_started":
            if self.started or len(self.events) != 1:
                fail("recovery start is duplicated or out of order")
            if value["prompt_tokens"] != self.expected_prompt_tokens:
                fail("recovery started prompt length differs")
            self.started = True
        elif name == "request_progress":
            processed = value["processed_prompt_tokens"]
            if (
                not self.started
                or self.first_token
                or processed <= self.last_progress
                or value["prompt_tokens"] != self.expected_prompt_tokens
            ):
                fail("recovery progress ordering differs")
            self.last_progress = processed
        elif name == "request_first_token":
            if (
                not self.started
                or self.first_token
                or self.last_progress != self.expected_prompt_tokens
            ):
                fail("recovery first-token ordering differs")
            self.first_token = True
        elif name == "request_cancel_requested":
            fail("recovery request was cancelled")
        elif name == "request_released":
            if (
                not self.started
                or not self.first_token
                or value["outcome"] != "length"
                or value["cancel_reason"] is not None
                or value["completion_tokens"] != 2
                or value["prompt_tokens"] != self.expected_prompt_tokens
                or value["reset_complete"] is not True
                or value["admit_to_release_ns"]
                != value["admit_to_start_ns"] + value["start_to_release_ns"]
            ):
                fail("recovery release differs from length/two/reset-complete")
            self.release = value
        elif name not in {
            "request_admitted",
            "request_started",
            "request_progress",
            "request_first_token",
        }:
            fail("unsupported lifecycle event in recovery trace")
        self.events.append(value)

    def finalize(self) -> dict[str, Any]:
        if self.release is None:
            fail("recovery lifecycle trace is incomplete")
        return {
            "request_id": self.request_id,
            "completion_id": self.completion_id,
            "release_observed_monotonic_ns": self.release["observed_monotonic_ns"],
        }


class DirectCancelRunValidator:
    """Enforce target/recovery alternation and a maximum active count of one."""

    def __init__(self) -> None:
        self.schedule = tuple(
            item
            for phase in PHASE_ORDER
            for item in ((phase, "target"), (phase, "recovery"))
        )
        self.index = 0
        self.active: CancellationTraceValidator | RecoveryTraceValidator | None = None
        self.active_phase: str | None = None
        self.completed: list[dict[str, Any]] = []
        self.max_active = 0
        self.last_release_ns = -1

    def _begin(self, phase: str, role: str) -> None:
        if self.active is not None or self.index >= len(self.schedule):
            fail("request began while another was active or after the frozen schedule")
        if self.schedule[self.index] != (phase, role):
            fail("direct cancellation request schedule differs")
        self.active_phase = phase
        self.active = (
            CancellationTraceValidator(PHASE_SPECS[phase])
            if role == "target"
            else RecoveryTraceValidator()
        )
        self.max_active = max(self.max_active, 1)

    def begin_target(self, phase: str) -> None:
        self._begin(phase, "target")

    def begin_recovery(self, phase: str) -> None:
        if self.last_release_ns < 0:
            fail("recovery began before a cancellation release")
        self._begin(phase, "recovery")

    def consume(self, event: dict[str, Any]) -> None:
        if self.active is None:
            fail("lifecycle event occurred without an expected active request")
        if event.get("event") == "request_admitted" and (
            event["observed_monotonic_ns"] <= self.last_release_ns
        ):
            fail("request admission did not follow the prior release")
        self.active.consume(event)

    def complete_active(self) -> dict[str, Any]:
        if self.active is None or self.active_phase is None:
            fail("no active request can be completed")
        result = self.active.finalize()
        release_ns = result["release_observed_monotonic_ns"]
        if release_ns <= self.last_release_ns:
            fail("request releases are not strictly ordered")
        phase, role = self.schedule[self.index]
        record = {"phase": phase, "role": role, **result}
        self.completed.append(record)
        self.last_release_ns = release_ns
        self.index += 1
        self.active = None
        self.active_phase = None
        return record

    def finalize(self) -> list[dict[str, Any]]:
        if self.active is not None or self.index != len(self.schedule):
            fail("direct cancellation run does not contain exactly eight requests")
        if self.max_active != 1 or len(self.completed) != 8:
            fail("direct cancellation active or request count differs")
        return list(self.completed)


@dataclasses.dataclass(frozen=True)
class CorrelatedObserverRecord:
    raw_payload: bytes
    event: dict[str, Any]
    received_monotonic_ns: int
    sender_pid: int
    sender_uid: int
    sender_gid: int


def correlate_records(
    observer_records: Sequence[Any],
    journal_records: Sequence[bytes | dict[str, Any]],
    expected_pid: int,
) -> list[dict[str, Any]]:
    lifecycle_journal: list[tuple[dict[str, Any], bytes]] = []
    cursors: set[str] = set()
    for raw_record in journal_records:
        record = (
            COL.strict_json_object(raw_record, "journal correlation record")
            if isinstance(raw_record, bytes)
            else dict(raw_record)
        )
        cursor = COL.nonempty_string(record.get("__CURSOR"), "journal cursor")
        if cursor in cursors:
            fail("journal correlation cursor is duplicated")
        cursors.add(cursor)
        event = COL.decode_lifecycle_message(record.get("MESSAGE"))
        if event is None:
            continue
        if record.get("_PID") != str(expected_pid):
            fail("journal lifecycle PID differs from gateway MainPID")
        lifecycle_journal.append(
            (record, COL.lifecycle_payload_from_message(record["MESSAGE"]))
        )
    if len(lifecycle_journal) != len(observer_records):
        fail("observer and authoritative journal lifecycle counts differ")
    result: list[dict[str, Any]] = []
    for index, (observer, journal_pair) in enumerate(
        zip(observer_records, lifecycle_journal)
    ):
        record, journal_payload = journal_pair
        raw_payload = getattr(observer, "raw_payload", None)
        event = getattr(observer, "event", None)
        sender_pid = getattr(observer, "sender_pid", None)
        if raw_payload != journal_payload or event != COL.decode_lifecycle_payload(
            journal_payload, "correlated lifecycle"
        ):
            fail("observer and authoritative journal lifecycle bytes differ")
        if sender_pid != expected_pid:
            fail("observer lifecycle PID differs from gateway MainPID")
        result.append(
            {
                "sequence": index,
                "cursor": record["__CURSOR"],
                "journal_monotonic_usec": record["__MONOTONIC_TIMESTAMP"],
                "journal_pid": record["_PID"],
                "observer_received_monotonic_ns": observer.received_monotonic_ns,
                "observer_sender_pid": observer.sender_pid,
                "observer_sender_uid": observer.sender_uid,
                "observer_sender_gid": observer.sender_gid,
                "payload_sha256": hashlib.sha256(raw_payload).hexdigest(),
                "payload_bytes": len(raw_payload),
            }
        )
    return result


class AtomicRunDirectory:
    def __init__(self, final_path: Path):
        requested = final_path if final_path.is_absolute() else Path.cwd() / final_path
        try:
            if requested.name in {"", ".", ".."}:
                fail("output directory name is invalid")
            parent = requested.parent.resolve(strict=True)
            self.final_path = parent / requested.name
            if self.final_path.exists() or self.final_path.is_symlink():
                fail("output directory already exists")
            metadata = parent.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                fail("output parent is not a real directory")
            suffix = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
            self.stage = parent / f".{self.final_path.name}.incomplete-{suffix}"
            self.stage.mkdir(mode=0o700)
        except GateError:
            raise
        except OSError:
            fail("failed to create the atomic output staging directory")
        self.published = False

    def publish(self) -> None:
        if self.published:
            fail("output directory is already published")
        renamed = False
        try:
            descriptor = os.open(
                self.stage, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            rename_noreplace(self.stage, self.final_path)
            renamed = True
            parent = os.open(
                self.final_path.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
            )
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
            self.published = True
        except OSError:
            if renamed:
                shutil.rmtree(self.final_path, ignore_errors=True)
            fail("failed to publish the atomic output directory")

    def abort(self) -> None:
        if not self.published:
            shutil.rmtree(self.stage, ignore_errors=True)


class RawWriter:
    def __init__(self, path: Path, guard: Any, maximum_bytes: int = MAX_RAW_BYTES):
        self.path = path
        descriptor = -1
        try:
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
            )
            self.handle = os.fdopen(descriptor, "wb", buffering=0)
            descriptor = -1
        except OSError:
            if descriptor >= 0:
                os.close(descriptor)
            fail("failed to create a raw evidence artifact")
        self.guard = guard
        self.maximum_bytes = maximum_bytes
        self.bytes_written = 0
        self.lines_written = 0
        self.digest = hashlib.sha256()
        self.closed = False

    def write(self, raw: bytes, label: str) -> None:
        if self.closed or not raw or b"\n" in raw or b"\r" in raw:
            fail("raw evidence line framing differs")
        self.guard.reject(raw, label)
        framed = raw + b"\n"
        if (
            self.lines_written >= MAX_RAW_LINES
            or self.bytes_written + len(framed) > self.maximum_bytes
        ):
            fail("raw evidence exceeds its streaming bound")
        try:
            self.handle.write(framed)
        except OSError:
            fail("failed to stream a raw evidence artifact")
        self.digest.update(framed)
        self.bytes_written += len(framed)
        self.lines_written += 1

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
        except OSError:
            fail("failed to seal a raw evidence artifact")
        self.closed = True

    def abort(self) -> None:
        if not self.closed:
            self.handle.close()
            self.closed = True


def build_request_command(
    request_key: str, body: bytes, auto_close: bool
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", request_key):
        fail("HTTP request key syntax differs")
    if not body or len(body) > MAX_HTTP_RESPONSE_BYTES or type(auto_close) is not bool:
        fail("HTTP request command body or close mode is invalid")
    return {
        "schema_version": HTTP_COMMAND_SCHEMA,
        "command": "request",
        "request_key": request_key,
        "method": "POST",
        "target": HTTP_TARGET,
        "body_base64": base64.b64encode(body).decode("ascii"),
        "authorization_mode": "valid_bearer",
        "close_on_first_nonempty_sse_content": auto_close,
    }


def build_close_command(request_key: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", request_key):
        fail("HTTP close request key syntax differs")
    return {
        "schema_version": HTTP_COMMAND_SCHEMA,
        "command": "close",
        "request_key": request_key,
    }


@dataclasses.dataclass(frozen=True)
class HttpPlan:
    request_key: str
    phase: str
    role: str
    body: bytes
    auto_close: bool


@dataclasses.dataclass(frozen=True)
class HttpChunk:
    index: int
    raw: bytes
    observed_monotonic_ns: int


@dataclasses.dataclass(frozen=True)
class SseItem:
    raw_data: bytes
    value: dict[str, Any] | None
    done: bool
    chunk_index: int
    observed_monotonic_ns: int


@dataclasses.dataclass(frozen=True)
class HttpResult:
    status: int | None
    outcome: str
    response_body: bytes
    chunks: tuple[HttpChunk, ...]
    items: tuple[SseItem, ...]
    response_end_monotonic_ns: int


class SseParser:
    """Reconstruct SSE objects and their final required raw chunk."""

    def __init__(self) -> None:
        self.line = bytearray()
        self.data_lines: list[bytes] = []
        self.items: list[SseItem] = []
        self.previous_cr = False
        self.event_bytes = 0

    def feed(self, chunk: HttpChunk) -> None:
        for byte in chunk.raw:
            if self.previous_cr:
                self.previous_cr = False
                if byte == 0x0A:
                    continue
            if byte == 0x0D:
                self._finish_line(chunk)
                self.previous_cr = True
            elif byte == 0x0A:
                self._finish_line(chunk)
            else:
                self.line.append(byte)
                if len(self.line) > 1024 * 1024:
                    fail("SSE line exceeds its bounded size")

    def finish(
        self, final_chunk: HttpChunk | None, *, allow_incomplete: bool
    ) -> tuple[SseItem, ...]:
        if self.previous_cr:
            self.previous_cr = False
        if self.line:
            if final_chunk is None:
                fail("SSE response lacks a final chunk")
            self._finish_line(final_chunk)
        if self.data_lines:
            if allow_incomplete:
                self.data_lines.clear()
                self.event_bytes = 0
            elif final_chunk is not None:
                self._dispatch(final_chunk)
        return tuple(self.items)

    def _finish_line(self, chunk: HttpChunk) -> None:
        line = bytes(self.line)
        self.line.clear()
        if not line:
            self._dispatch(chunk)
            return
        if line.startswith(b":"):
            return
        field, separator, value = line.partition(b":")
        if separator and value.startswith(b" "):
            value = value[1:]
        if field == b"data":
            self.event_bytes += len(value) + (1 if self.data_lines else 0)
            if self.event_bytes > 2 * 1024 * 1024:
                fail("SSE event exceeds its bounded size")
            self.data_lines.append(value)

    def _dispatch(self, chunk: HttpChunk) -> None:
        if not self.data_lines:
            self.event_bytes = 0
            return
        raw = b"\n".join(self.data_lines)
        self.data_lines.clear()
        self.event_bytes = 0
        if raw == b"[DONE]":
            item = SseItem(raw, None, True, chunk.index, chunk.observed_monotonic_ns)
        else:
            value = COL.strict_json_object(raw, "SSE data object")
            item = SseItem(raw, value, False, chunk.index, chunk.observed_monotonic_ns)
        self.items.append(item)


def nonempty_content_items(items: Sequence[SseItem]) -> list[SseItem]:
    result: list[SseItem] = []
    for item in items:
        if item.value is None:
            continue
        choices = item.value.get("choices")
        if type(choices) is not list or not choices:
            continue
        first = choices[0]
        delta = first.get("delta") if type(first) is dict else None
        content = delta.get("content") if type(delta) is dict else None
        if type(content) is str and content:
            result.append(item)
    return result


class EvidenceHttpClient:
    """Strict asynchronous controller for the committed HTTP evidence client."""

    def __init__(self, command: Sequence[str], guard: Any, writer: RawWriter):
        self.command = tuple(command)
        self.guard = guard
        self.writer = writer
        self.process: subprocess.Popen[bytes] | None = None
        self.reader: Any | None = None
        self.stderr: Any | None = None
        self.active: HttpPlan | None = None
        self.request_count = 0
        self.last_response_end_ns = -1

    def start(self) -> None:
        if self.process is not None:
            fail("HTTP evidence client is already started")
        self.stderr = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(
                list(self.command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr,
                bufsize=0,
                start_new_session=True,
            )
        except OSError:
            fail("failed to start the HTTP evidence client")
        if self.process.stdin is None or self.process.stdout is None:
            fail("HTTP evidence client pipes are unavailable")
        self.reader = COL.BoundedLineReader(self.process.stdout.fileno())
        event = self._read_event(time.monotonic_ns() + 30_000_000_000)
        COL.exact_keys(
            event,
            {"schema_version", "event", "observed_monotonic_ns"},
            "HTTP ready event",
        )
        if event["schema_version"] != HTTP_EVENT_SCHEMA or event["event"] != "ready":
            fail("HTTP evidence client ready event differs")
        COL.integer(event["observed_monotonic_ns"], "HTTP ready timestamp")

    def begin(self, plan: HttpPlan) -> None:
        if self.active is not None:
            fail("HTTP request began while another request was active")
        self.guard.reject(plan.body, "HTTP request body")
        self._write_command(
            build_request_command(plan.request_key, plan.body, plan.auto_close)
        )
        self.active = plan
        self.request_count += 1

    def request_close(self, request_key: str) -> int:
        if self.active is None or self.active.request_key != request_key:
            fail("HTTP close does not match the active request")
        boundary = time.monotonic_ns()
        self._write_command(build_close_command(request_key))
        return boundary

    def finish(self, deadline_ns: int) -> HttpResult:
        plan = self.active
        if plan is None:
            fail("HTTP response collection lacks an active request")
        saw_request = False
        saw_start = False
        status: int | None = None
        next_chunk = 0
        chunks: list[HttpChunk] = []
        digest = hashlib.sha256()
        total = 0
        outcome: str | None = None
        last_timestamp = -1
        end_timestamp = -1
        while outcome is None:
            event = self._read_event(deadline_ns)
            if event.get("schema_version") != HTTP_EVENT_SCHEMA:
                fail("HTTP event schema_version differs")
            name = event.get("event")
            fields = {
                key: value
                for key, value in event.items()
                if key not in {"schema_version", "event"}
            }
            if name == "http_request":
                if saw_request:
                    fail("HTTP request evidence is duplicated")
                support_plan = COL.HttpPlan(
                    phase="cancellation",
                    case_id=plan.phase,
                    request_index=self.request_count,
                    request_key=plan.request_key,
                    target=HTTP_TARGET,
                    body=plan.body,
                    expected_status=200,
                    expect_release=True,
                )
                connect, sent = COL.HttpClientProcess._validate_http_request_event(
                    fields, support_plan
                )
                if connect < self.last_response_end_ns:
                    fail("HTTP request begins before the prior response ended")
                last_timestamp = sent
                saw_request = True
            elif name == "http_response_start":
                if not saw_request or saw_start:
                    fail("HTTP response start ordering differs")
                COL.exact_keys(
                    fields,
                    {"request_key", "status", "headers", "observed_monotonic_ns"},
                    "HTTP response start",
                )
                if fields["request_key"] != plan.request_key:
                    fail("HTTP response start request key differs")
                status = COL.integer(fields["status"], "HTTP status", minimum=100)
                if status != 200 or type(fields["headers"]) is not list:
                    fail("direct cancellation HTTP response status or headers differ")
                for pair in fields["headers"]:
                    if (
                        type(pair) is not list
                        or len(pair) != 2
                        or any(type(item) is not str for item in pair)
                    ):
                        fail("direct cancellation HTTP response header differs")
                media = [
                    value.split(";", 1)[0].strip().lower()
                    for name_, value in fields["headers"]
                    if type(name_) is str
                    and type(value) is str
                    and name_.lower() == "content-type"
                ]
                if media != ["text/event-stream"]:
                    fail("direct cancellation HTTP Content-Type differs")
                observed = COL.integer(
                    fields["observed_monotonic_ns"], "HTTP response start timestamp"
                )
                if observed < last_timestamp:
                    fail("HTTP response start precedes request send")
                last_timestamp = observed
                saw_start = True
            elif name == "http_body_chunk":
                if not saw_start:
                    fail("HTTP body chunk precedes response start")
                COL.exact_keys(
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
                raw = COL.decode_bound_bytes(fields, "HTTP body chunk")
                total += len(raw)
                if total > MAX_HTTP_RESPONSE_BYTES:
                    fail("HTTP response exceeds its bounded size")
                self.guard.reject(raw, "HTTP response body")
                observed = COL.integer(
                    fields["observed_monotonic_ns"], "HTTP chunk timestamp"
                )
                if observed < last_timestamp:
                    fail("HTTP response timestamps regressed")
                chunks.append(HttpChunk(next_chunk, raw, observed))
                digest.update(raw)
                next_chunk += 1
                last_timestamp = observed
            elif name == "http_response_end":
                if not saw_request:
                    fail("HTTP response end precedes request evidence")
                COL.exact_keys(
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
                    fail("HTTP response outcome differs")
                if (outcome in {"eof", "client_closed"}) != (fields["error"] is None):
                    fail("HTTP response error field differs from outcome")
                if (
                    fields["body_bytes"] != total
                    or fields["body_sha256"] != digest.hexdigest()
                ):
                    fail("HTTP response end aggregate differs")
                end_timestamp = COL.integer(
                    fields["observed_monotonic_ns"], "HTTP response end timestamp"
                )
                if end_timestamp < last_timestamp:
                    fail("HTTP response end timestamp regressed")
                self.last_response_end_ns = end_timestamp
            elif name == "command_error":
                fail("HTTP evidence client rejected a gate command")
            else:
                fail("HTTP evidence client emitted an unexpected event")
        body = b"".join(chunk.raw for chunk in chunks)
        parser = SseParser()
        for chunk in chunks:
            parser.feed(chunk)
        items = parser.finish(
            chunks[-1] if chunks else None, allow_incomplete=outcome == "client_closed"
        )
        self.active = None
        return HttpResult(status, outcome, body, tuple(chunks), items, end_timestamp)

    def close(self, *, require_eight: bool = True) -> None:
        process = self.process
        if process is None:
            return
        pending: BaseException | None = None
        try:
            if self.active is not None:
                fail("HTTP evidence client shutdown has an active request")
            if require_eight and self.request_count != 8:
                fail("HTTP evidence client request count differs from eight")
            self._write_command(
                {"schema_version": HTTP_COMMAND_SCHEMA, "command": "shutdown"}
            )
            event = self._read_event(time.monotonic_ns() + 5_000_000_000)
            COL.exact_keys(
                event,
                {"schema_version", "event", "observed_monotonic_ns"},
                "HTTP shutdown event",
            )
            if (
                event["schema_version"] != HTTP_EVENT_SCHEMA
                or event["event"] != "shutdown_complete"
            ):
                fail("HTTP shutdown acknowledgement differs")
            if process.wait(timeout=5.0) != 0:
                fail("HTTP evidence client exited nonzero")
            if self.reader is None or process.stdout is None:
                fail("HTTP shutdown stream state is unavailable")
            if self.reader.buffer or os.read(process.stdout.fileno(), 1):
                fail("HTTP evidence client emitted trailing stdout")
        except BaseException as error:
            pending = error
            if process.poll() is None:
                COL.terminate_process_group(process)
        finally:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            self._check_stderr()
            self.process = None
        if pending is not None:
            raise pending

    def abort(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            COL.terminate_process_group(process)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self._check_stderr(require_empty=False)
        self.process = None

    def _check_stderr(self, *, require_empty: bool = True) -> None:
        if self.stderr is None:
            return
        self.stderr.seek(0)
        raw = self.stderr.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            fail("HTTP evidence client stderr exceeds its bound")
        self.guard.reject(raw, "HTTP evidence client stderr")
        self.stderr.close()
        self.stderr = None
        if require_empty and raw:
            fail("HTTP evidence client emitted stderr")

    def _write_command(self, value: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None or process.poll() is not None:
            fail("HTTP evidence client is not running")
        raw = compact_json(value)
        self.guard.reject(raw, "HTTP evidence client command")
        try:
            process.stdin.write(raw + b"\n")
            process.stdin.flush()
        except OSError:
            fail("failed to write an HTTP evidence client command")

    def _read_event(self, deadline_ns: int) -> dict[str, Any]:
        if self.reader is None:
            fail("HTTP evidence client reader is unavailable")
        raw = self.reader.read(deadline_ns, "HTTP evidence client event")
        self.guard.reject(raw, "HTTP evidence client event")
        self.writer.write(raw, "raw HTTP evidence")
        return cast(
            dict[str, Any],
            COL.strict_json_object(raw, "HTTP evidence client event"),
        )


@dataclasses.dataclass(frozen=True)
class ServiceIdentity:
    unit: str
    user: str
    uid: int
    gid: int
    control_group: str
    gateway_pid: int
    gateway_starttime_ticks: int
    worker_pid: int
    worker_starttime_ticks: int
    n_restarts: int
    boot_id: str


def read_boot_id() -> str:
    descriptor = -1
    try:
        descriptor = os.open(
            "/proc/sys/kernel/random/boot_id",
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            fail("boot ID source is not a regular proc file")
        raw = os.read(descriptor, 129)
        if not raw or len(raw) > 128:
            fail("boot ID size is invalid")
    except GateError:
        raise
    except OSError:
        fail("failed to read the boot ID")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        value = raw.decode("ascii", errors="strict").strip().replace("-", "")
    except UnicodeError:
        fail("boot ID is not ASCII")
    if re.fullmatch(r"[0-9a-f]{32}", value) is None:
        fail("boot ID syntax differs")
    return value


def capture_service_identity() -> ServiceIdentity:
    raw = COL.run_bounded_command(
        [
            "systemctl",
            "show",
            SERVICE_UNIT,
            "--property=ActiveState",
            "--property=SubState",
            "--property=ControlGroup",
            "--property=MainPID",
            "--property=NRestarts",
            "--property=User",
            "--no-pager",
        ],
        "direct cancellation systemd identity",
        maximum_stdout=4096,
    )
    fields = COL.parse_key_value_lines(raw, "direct cancellation systemd identity")
    if set(fields) != {
        "ActiveState",
        "SubState",
        "ControlGroup",
        "MainPID",
        "NRestarts",
        "User",
    }:
        fail("direct cancellation systemd identity fields differ")
    if fields["ActiveState"] != "active" or fields["SubState"] != "running":
        fail("SQ8 gateway service is not active and running")
    if not fields["MainPID"].isdecimal() or not fields["NRestarts"].isdecimal():
        fail("SQ8 gateway PID or restart count is invalid")
    gateway_pid = int(fields["MainPID"])
    if gateway_pid <= 0:
        fail("SQ8 gateway MainPID is invalid")
    try:
        account = pwd.getpwnam(COL.nonempty_string(fields["User"], "service user"))
    except KeyError:
        fail("SQ8 gateway service user does not exist")
    control_group = COL.nonempty_string(fields["ControlGroup"], "service cgroup")
    COL.control_group_parts(control_group)
    _, gateway_start = COL.process_identity(Path("/proc"), gateway_pid)
    gateway = COL.process_record(Path("/proc"), gateway_pid)
    workers: list[int] = []
    for child in gateway["children"]:
        try:
            executable = os.readlink(Path("/proc") / str(child) / "exe")
        except OSError:
            continue
        if Path(executable).name == "ullm-sq8-worker":
            workers.append(child)
    if len(workers) != 1:
        fail("SQ8 gateway does not have exactly one worker child")
    worker_pid = workers[0]
    _, worker_start = COL.process_identity(Path("/proc"), worker_pid)
    return ServiceIdentity(
        unit=SERVICE_UNIT,
        user=fields["User"],
        uid=account.pw_uid,
        gid=account.pw_gid,
        control_group=control_group,
        gateway_pid=gateway_pid,
        gateway_starttime_ticks=gateway_start,
        worker_pid=worker_pid,
        worker_starttime_ticks=worker_start,
        n_restarts=int(fields["NRestarts"]),
        boot_id=read_boot_id(),
    )


def require_service_identity(expected: ServiceIdentity) -> None:
    if capture_service_identity() != expected:
        fail("gateway, worker, service, user, restart, or boot identity changed")


def validate_docker_identity(image_id: str, network_id: str) -> None:
    if CONTENT_IMAGE_RE.fullmatch(image_id) is None:
        fail("HTTP client image must be an exact content identity")
    if HTTP_NETWORK_ID_RE.fullmatch(network_id) is None:
        fail("Docker network ID syntax differs")
    network_raw = COL.run_bounded_command(
        [COL.DOCKER_BIN, "network", "inspect", HTTP_NETWORK_NAME],
        "direct cancellation Docker network identity",
    )
    network_value = COL.strict_json_bytes(network_raw, "Docker network identity")
    if (
        type(network_value) is not list
        or len(network_value) != 1
        or type(network_value[0]) is not dict
    ):
        fail("Docker network inspection shape differs")
    network = network_value[0]
    ipam = network.get("IPAM")
    configurations = ipam.get("Config") if type(ipam) is dict else None
    if (
        network.get("Id") != network_id
        or type(configurations) is not list
        or len(configurations) != 1
        or configurations[0].get("Subnet") != COL.HTTP_NETWORK_SUBNET
        or configurations[0].get("Gateway") != COL.HTTP_NETWORK_GATEWAY
    ):
        fail("Docker network content, subnet, or gateway identity differs")
    image_raw = COL.run_bounded_command(
        [COL.DOCKER_BIN, "image", "inspect", image_id],
        "direct cancellation HTTP client image identity",
    )
    image_value = COL.strict_json_bytes(image_raw, "HTTP client image identity")
    if (
        type(image_value) is not list
        or len(image_value) != 1
        or type(image_value[0]) is not dict
        or image_value[0].get("Id") != image_id
    ):
        fail("HTTP client image content identity differs")


def require_ready(image_id: str) -> None:
    script = (
        "import sys,urllib.request;"
        "r=urllib.request.urlopen(sys.argv[1],timeout=2);"
        "print(r.status)"
    )
    raw = COL.run_bounded_command(
        [
            COL.DOCKER_BIN,
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
            image_id,
            "-c",
            script,
            HTTP_READY_URL,
        ],
        "direct cancellation Docker-network readiness probe",
        timeout_seconds=10.0,
        maximum_stdout=32,
    )
    try:
        status = raw.decode("ascii", errors="strict").strip()
    except UnicodeError:
        fail("SQ8 gateway readiness response is not ASCII")
    if status != "200":
        fail("SQ8 gateway readiness differs from HTTP 200")


@dataclasses.dataclass(frozen=True)
class FixtureSnapshot:
    fixture_id: str
    prompt_tokens: int
    messages: list[dict[str, str]]
    raw: bytes
    sha256: str
    identity: tuple[int, ...]


def load_fixture(path: Path, fixture_id: str) -> FixtureSnapshot:
    expected_prompt, expected_sha = FIXTURE_IDENTITIES[fixture_id]
    raw, identity = _single_fd_snapshot(
        path, f"{fixture_id} fixture", MAX_GATE_FILE_BYTES
    )
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        fail("fixed chat-template fixture SHA-256 differs")
    value = COL.strict_json_object(raw, f"{fixture_id} fixture")
    COL.exact_keys(
        value,
        {
            "schema_version",
            "fixture_id",
            "kind",
            "messages",
            "construction",
            "template_options",
            "expected",
        },
        f"{fixture_id} fixture",
    )
    expected = value["expected"]
    options = value["template_options"]
    if (
        value["schema_version"] != FIXTURE_SCHEMA
        or value["fixture_id"] != fixture_id
        or value["kind"] != "exact_length"
        or type(expected) is not dict
        or expected.get("prompt_tokens") != expected_prompt
        or options != {"add_generation_prompt": True, "enable_thinking": False}
        or type(value["messages"]) is not list
        or len(value["messages"]) != 1
    ):
        fail("fixed chat-template fixture contract differs")
    message = value["messages"][0]
    if (
        type(message) is not dict
        or set(message) != {"role", "content"}
        or message["role"] != "user"
        or type(message["content"]) is not str
        or not message["content"]
    ):
        fail("fixed chat-template fixture message differs")
    return FixtureSnapshot(
        fixture_id,
        expected_prompt,
        [{"role": message["role"], "content": message["content"]}],
        raw,
        expected_sha,
        identity,
    )


def request_body(fixture: FixtureSnapshot, max_tokens: int) -> bytes:
    if max_tokens not in {2, 512}:
        fail("direct cancellation max_tokens differs")
    raw = compact_json(
        {
            "model": MODEL_ID,
            "messages": fixture.messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
        }
    )
    if len(raw) > COL.MAX_HTTP_BODY_BYTES:
        fail("direct cancellation request body exceeds its bound")
    return raw


class JournalCapture:
    def __init__(
        self,
        boot_id: str,
        expected_pid: int,
        writer: RawWriter,
        guard: Any,
    ):
        self.boot_id = boot_id
        self.expected_pid = expected_pid
        self.writer = writer
        self.guard = guard
        self.source = COL.JournalSource(boot_id)
        self.records: list[bytes] = []
        self.cursors: set[str] = set()

    def start(self) -> None:
        self.source.start()

    def poll(self) -> None:
        for raw in self.source.poll():
            self.guard.reject(raw, "service journal evidence")
            record = COL.strict_json_object(raw, "service journal record")
            if set(record) != {
                "__CURSOR",
                "__MONOTONIC_TIMESTAMP",
                "_BOOT_ID",
                "_PID",
                "_SYSTEMD_UNIT",
                "PRIORITY",
                "MESSAGE",
            }:
                fail("service journal record fields differ")
            cursor = COL.nonempty_string(record["__CURSOR"], "service journal cursor")
            if cursor in self.cursors:
                fail("service journal cursor is duplicated")
            self.cursors.add(cursor)
            if (
                record["_BOOT_ID"] != self.boot_id
                or record["_SYSTEMD_UNIT"] != SERVICE_UNIT
            ):
                fail("service journal boot or unit identity differs")
            event = COL.decode_lifecycle_message(record["MESSAGE"])
            if event is not None and record["_PID"] != str(self.expected_pid):
                fail("service journal lifecycle PID differs")
            self.writer.write(raw, "raw service journal evidence")
            self.records.append(raw)

    def wait_correlated(
        self, observer_records: Sequence[Any], deadline_ns: int
    ) -> list[dict[str, Any]]:
        while True:
            self.poll()
            lifecycle_count = 0
            for raw in self.records:
                record = COL.strict_json_object(
                    raw, "service journal correlation record"
                )
                if COL.decode_lifecycle_message(record["MESSAGE"]) is not None:
                    lifecycle_count += 1
            if lifecycle_count > len(observer_records):
                fail("authoritative journal has an unobserved lifecycle event")
            if lifecycle_count == len(observer_records):
                return correlate_records(
                    observer_records, self.records, self.expected_pid
                )
            if time.monotonic_ns() >= deadline_ns:
                fail("observer-to-journal byte correlation timed out")
            self.source.wait_until(min(deadline_ns, time.monotonic_ns() + 50_000_000))


class DirectCancelGate:
    def __init__(
        self,
        identity: ServiceIdentity,
        image_id: str,
        network_id: str,
        fixtures: dict[str, FixtureSnapshot],
        observer: Any,
        observer_writer: RawWriter,
        correlation_writer: RawWriter,
        journal: JournalCapture,
        http: EvidenceHttpClient,
    ):
        self.identity = identity
        self.image_id = image_id
        self.network_id = network_id
        self.fixtures = fixtures
        self.observer = observer
        self.observer_writer = observer_writer
        self.correlation_writer = correlation_writer
        self.journal = journal
        self.http = http
        self.validator = DirectCancelRunValidator()
        self.observer_records: list[Any] = []
        self.correlation_count = 0

    def run(self) -> list[dict[str, Any]]:
        validate_phase_order(PHASE_ORDER)
        for phase in PHASE_ORDER:
            self._run_target(PHASE_SPECS[phase])
            self._run_recovery(phase)
        results = self.validator.finalize()
        time.sleep(QUIET_DRAIN_NS / 1_000_000_000)
        self.observer.require_empty()
        self._correlate(time.monotonic_ns() + RELEASE_DEADLINE_NS)
        require_service_identity(self.identity)
        require_ready(self.image_id)
        validate_docker_identity(self.image_id, self.network_id)
        return results

    def _run_target(self, spec: PhaseSpec) -> None:
        require_service_identity(self.identity)
        self.validator.begin_target(spec.phase)
        fixture = self.fixtures[spec.fixture_id]
        plan = HttpPlan(
            request_key=f"direct-{spec.phase}-target",
            phase=spec.phase,
            role="target",
            body=request_body(fixture, 512),
            auto_close=spec.auto_close,
        )
        self.http.begin(plan)
        active = self.validator.active
        assert isinstance(active, CancellationTraceValidator)
        request_deadline = time.monotonic_ns() + REQUEST_TIMEOUT_NS
        if not spec.auto_close:
            while not active.trigger_reached():
                self._receive_one(request_deadline)
            close_ns = self.http.request_close(plan.request_key)
            active.mark_close(close_ns)
        self._wait_for_release(request_deadline)
        result = self.http.finish(request_deadline)
        self._validate_target_http(active, result)
        self.validator.complete_active()
        self._correlate(time.monotonic_ns() + RELEASE_DEADLINE_NS)
        require_service_identity(self.identity)

    def _run_recovery(self, phase: str) -> None:
        require_service_identity(self.identity)
        self.validator.begin_recovery(phase)
        fixture = self.fixtures["exact-p0032"]
        plan = HttpPlan(
            request_key=f"direct-{phase}-recovery",
            phase=phase,
            role="recovery",
            body=request_body(fixture, 2),
            auto_close=False,
        )
        self.http.begin(plan)
        request_deadline = time.monotonic_ns() + REQUEST_TIMEOUT_NS
        self._wait_for_release(request_deadline)
        result = self.http.finish(request_deadline)
        active = self.validator.active
        assert isinstance(active, RecoveryTraceValidator)
        self._validate_recovery_http(active, result)
        self.validator.complete_active()
        self._correlate(time.monotonic_ns() + RELEASE_DEADLINE_NS)
        require_service_identity(self.identity)
        require_ready(self.image_id)
        validate_docker_identity(self.image_id, self.network_id)

    def _receive_one(self, deadline_ns: int) -> None:
        datagram = self.observer.receive(
            deadline_ns, expected_sender_pid=self.identity.gateway_pid
        )
        if (
            datagram.sender_uid != self.identity.uid
            or datagram.sender_gid != self.identity.gid
        ):
            fail("lifecycle observer sender UID or GID differs from the service")
        self.observer_writer.write(
            datagram.raw_payload, "raw lifecycle observer payload"
        )
        self.observer_records.append(datagram)
        self.validator.consume(datagram.event)
        self.journal.poll()

    def _wait_for_release(self, request_deadline_ns: int) -> None:
        while True:
            active = self.validator.active
            if isinstance(active, CancellationTraceValidator):
                if active.release is not None:
                    return
                deadline = request_deadline_ns
                if active.cancel is not None:
                    deadline = min(
                        deadline,
                        active.cancel["observed_monotonic_ns"] + RELEASE_DEADLINE_NS,
                    )
            elif isinstance(active, RecoveryTraceValidator):
                if active.release is not None:
                    return
                deadline = request_deadline_ns
            else:
                fail("lifecycle release wait lacks an active trace")
            self._receive_one(deadline)

    def _validate_target_http(
        self, active: CancellationTraceValidator, result: HttpResult
    ) -> None:
        if result.outcome != "client_closed" or result.status not in {None, 200}:
            fail("cancel target HTTP outcome differs from deliberate client close")
        contents = nonempty_content_items(result.items)
        completion_ids = {
            item.value["id"]
            for item in result.items
            if item.value is not None and type(item.value.get("id")) is str
        }
        if completion_ids and completion_ids != {active.completion_id}:
            fail("cancel target HTTP and lifecycle completion IDs differ")
        for item in contents:
            active.observe_content(item.observed_monotonic_ns, item.chunk_index)
        if active.spec.auto_close:
            if not contents:
                fail("decode target lacks its first non-empty SSE content")
            if any(
                item.value is None or item.value.get("id") != active.completion_id
                for item in contents
            ):
                fail("decode content lacks its exact lifecycle completion ID")
            trigger_chunk = contents[0].chunk_index
            if not result.chunks or result.chunks[-1].index != trigger_chunk:
                fail("decode HTTP client read another chunk after its content trigger")
            active.mark_close(contents[0].observed_monotonic_ns)
        active.finalize()

    @staticmethod
    def _validate_recovery_http(
        active: RecoveryTraceValidator, result: HttpResult
    ) -> None:
        if result.status != 200 or result.outcome != "eof":
            fail("recovery HTTP request did not complete at EOF")
        completion_id = COL.validate_resource_sse(result.response_body)
        if completion_id != active.completion_id:
            fail("recovery HTTP and lifecycle completion IDs differ")
        done = [item for item in result.items if item.done]
        if len(done) != 1 or result.items[-1] is not done[0]:
            fail("recovery SSE [DONE] count or ordering differs")
        final_choices = []
        usage_counts = []
        for item in result.items:
            if item.value is None:
                continue
            choices = item.value.get("choices")
            if type(choices) is list and choices and type(choices[0]) is dict:
                finish_reason = choices[0].get("finish_reason")
                if finish_reason is not None:
                    final_choices.append(finish_reason)
            usage = item.value.get("usage")
            if type(usage) is dict and "completion_tokens" in usage:
                usage_counts.append(usage["completion_tokens"])
        if final_choices != ["length"] or usage_counts != [2]:
            fail("recovery SSE final choice or usage differs")

    def _correlate(self, deadline_ns: int) -> None:
        correlations = self.journal.wait_correlated(self.observer_records, deadline_ns)
        if len(correlations) < self.correlation_count:
            fail("observer-to-journal correlation count regressed")
        for value in correlations[self.correlation_count :]:
            self.correlation_writer.write(
                compact_json({"schema_version": GATE_SCHEMA, **value}),
                "observer journal correlation evidence",
            )
        self.correlation_count = len(correlations)


def write_json_file(path: Path, value: dict[str, Any], guard: Any) -> None:
    raw = compact_json(value) + b"\n"
    guard.reject(raw, path.name)
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                fail("atomic JSON output write was short")
            offset += written
        os.fsync(descriptor)
    except GateError:
        raise
    except OSError:
        fail("failed to write an atomic JSON output")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def verify_snapshot(
    path: Path, label: str, raw: bytes, identity: tuple[int, ...]
) -> None:
    current, current_identity = _single_fd_snapshot(path, label, max(len(raw), 1))
    if current_identity != identity or current != raw:
        fail("a snapshotted gate input changed during the run")


def verify_raw_writer(writer: RawWriter) -> None:
    raw, _identity = _single_fd_snapshot(
        writer.path, f"sealed {writer.path.name}", writer.maximum_bytes
    )
    if (
        len(raw) != writer.bytes_written
        or raw.count(b"\n") != writer.lines_written
        or hashlib.sha256(raw).digest() != writer.digest.digest()
    ):
        fail("a sealed raw evidence artifact changed before publication")


def verify_json_file(path: Path, expected: dict[str, Any]) -> None:
    raw, _identity = _single_fd_snapshot(path, f"sealed {path.name}", MAX_RAW_BYTES)
    if raw != compact_json(expected) + b"\n":
        fail("a sealed JSON evidence artifact changed before publication")


@dataclasses.dataclass(frozen=True)
class Arguments:
    output_dir: Path
    api_key_file: Path
    http_image_id: str
    docker_network_id: str


def execute(args: Arguments) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    fixture_root = repo_root / "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures"
    output = AtomicRunDirectory(args.output_dir)
    writers: list[RawWriter] = []
    snapshots: Any | None = None
    observer: Any | None = None
    http: EvidenceHttpClient | None = None
    published = False
    try:
        gate_raw, gate_identity = _single_fd_snapshot(
            Path(__file__),
            "direct cancellation gate implementation",
            MAX_GATE_FILE_BYTES,
        )
        client_path = Path(__file__).with_name("sq8-openwebui-http-client.py")
        client_raw, client_identity = _single_fd_snapshot(
            client_path, "HTTP evidence client implementation", MAX_GATE_FILE_BYTES
        )
        if hashlib.sha256(client_raw).hexdigest() != HTTP_CLIENT_SHA256:
            fail("HTTP evidence client implementation SHA-256 differs")
        fixtures = {
            fixture_id: load_fixture(fixture_root / f"{fixture_id}.json", fixture_id)
            for fixture_id in FIXTURE_IDENTITIES
        }
        guard, credential_raw = COL.SecretGuard.snapshot_from_file(args.api_key_file)
        for label, raw in (
            ("gate implementation", gate_raw),
            ("collector support", COLLECTOR_SUPPORT_RAW),
            ("HTTP client implementation", client_raw),
            *((f"{item.fixture_id} fixture", item.raw) for item in fixtures.values()),
        ):
            guard.reject(raw, label)
        validate_docker_identity(args.http_image_id, args.docker_network_id)
        identity = capture_service_identity()
        if identity.uid != COL.HTTP_CLIENT_UID or identity.gid != COL.HTTP_CLIENT_GID:
            fail("service identity differs from the fixed HTTP client UID or GID")
        require_ready(args.http_image_id)

        snapshots = COL.RuntimeSnapshots.create(client_raw, credential_raw)
        config = types.SimpleNamespace(
            identities={"openwebui": {"derived_image_id": args.http_image_id}}
        )
        http_command = COL.build_http_client_command(config, snapshots)
        http_writer = RawWriter(output.stage / "http-client.raw.jsonl", guard)
        observer_writer = RawWriter(output.stage / "observer.raw.jsonl", guard)
        journal_writer = RawWriter(output.stage / "service-journal.raw.jsonl", guard)
        correlation_writer = RawWriter(
            output.stage / "observer-journal-correlation.raw.jsonl", guard
        )
        writers.extend(
            [http_writer, observer_writer, journal_writer, correlation_writer]
        )
        journal = JournalCapture(
            identity.boot_id, identity.gateway_pid, journal_writer, guard
        )
        observer = COL.LifecycleObserver(
            OBSERVER_SOCKET,
            guard,
            expected_uid=identity.uid,
            expected_gid=identity.gid,
        )
        observer.open()
        journal.start()
        http = EvidenceHttpClient(http_command, guard, http_writer)
        http.start()
        snapshots.unlink_credential()

        gate = DirectCancelGate(
            identity,
            args.http_image_id,
            args.docker_network_id,
            fixtures,
            observer,
            observer_writer,
            correlation_writer,
            journal,
            http,
        )
        results = gate.run()
        http.close()
        time.sleep(QUIET_DRAIN_NS / 1_000_000_000)
        observer.require_empty()
        journal.poll()
        final_correlations = journal.wait_correlated(
            gate.observer_records, time.monotonic_ns() + RELEASE_DEADLINE_NS
        )
        if (
            len(gate.observer_records) != gate.correlation_count
            or len(final_correlations) != gate.correlation_count
        ):
            fail("final observer-to-journal correlation is incomplete")
        observer.require_empty()
        observer.close()
        observer = None
        time.sleep(QUIET_DRAIN_NS / 1_000_000_000)
        journal.poll()
        post_close_correlations = correlate_records(
            gate.observer_records, journal.records, identity.gateway_pid
        )
        if len(post_close_correlations) != gate.correlation_count:
            fail("post-observer-close journal correlation differs")
        require_service_identity(identity)
        require_ready(args.http_image_id)

        for writer in writers:
            writer.close()
        input_manifest = {
            "schema_version": GATE_SCHEMA,
            "record_type": "input_manifest",
            "inputs": [
                {
                    "path": "tools/run-sq8-direct-cancel-gate.py",
                    "bytes": len(gate_raw),
                    "sha256": hashlib.sha256(gate_raw).hexdigest(),
                },
                {
                    "path": "tools/collect-sq8-openwebui-release.py",
                    "bytes": len(COLLECTOR_SUPPORT_RAW),
                    "sha256": hashlib.sha256(COLLECTOR_SUPPORT_RAW).hexdigest(),
                },
                {
                    "path": "tools/sq8-openwebui-http-client.py",
                    "bytes": len(client_raw),
                    "sha256": hashlib.sha256(client_raw).hexdigest(),
                },
                *[
                    {
                        "path": f"tests/fixtures/sq8-serving-v0.1/chat-template/fixtures/{item.fixture_id}.json",
                        "bytes": len(item.raw),
                        "sha256": item.sha256,
                    }
                    for item in fixtures.values()
                ],
            ],
            "request_bodies": [
                {
                    "fixture_id": fixture_id,
                    "max_tokens": max_tokens,
                    "bytes": len(request_body(fixtures[fixture_id], max_tokens)),
                    "sha256": hashlib.sha256(
                        request_body(fixtures[fixture_id], max_tokens)
                    ).hexdigest(),
                }
                for fixture_id, max_tokens in (
                    ("exact-p3584", 512),
                    ("exact-p0032", 512),
                    ("exact-p0032", 2),
                )
            ],
        }
        write_json_file(output.stage / "input-manifest.json", input_manifest, guard)
        summary = {
            "schema_version": GATE_SCHEMA,
            "record_type": "summary",
            "phase_order": list(PHASE_ORDER),
            "request_count": len(results),
            "max_active": gate.validator.max_active,
            "service_identity": dataclasses.asdict(identity),
            "http_image_id": args.http_image_id,
            "docker_network_name": HTTP_NETWORK_NAME,
            "docker_network_id": args.docker_network_id,
            "observer_socket": os.fspath(OBSERVER_SOCKET),
            "observer_event_count": len(gate.observer_records),
            "journal_correlation_count": gate.correlation_count,
            "requests": results,
            "artifacts": {
                writer.path.name: {
                    "bytes": writer.bytes_written,
                    "lines": writer.lines_written,
                    "sha256": writer.digest.hexdigest(),
                }
                for writer in writers
            },
        }
        write_json_file(output.stage / "summary.json", summary, guard)

        verify_snapshot(
            Path(__file__),
            "direct cancellation gate implementation",
            gate_raw,
            gate_identity,
        )
        verify_snapshot(
            client_path,
            "HTTP evidence client implementation",
            client_raw,
            client_identity,
        )
        support_path = Path(__file__).with_name("collect-sq8-openwebui-release.py")
        verify_snapshot(
            support_path,
            "collector support",
            COLLECTOR_SUPPORT_RAW,
            COLLECTOR_SUPPORT_IDENTITY,
        )
        for item in fixtures.values():
            verify_snapshot(
                fixture_root / f"{item.fixture_id}.json",
                f"{item.fixture_id} fixture",
                item.raw,
                item.identity,
            )
        for path in output.stage.iterdir():
            guard.scan_file(path, f"staged output {path.name}")
        for writer in writers:
            verify_raw_writer(writer)
        verify_json_file(output.stage / "input-manifest.json", input_manifest)
        verify_json_file(output.stage / "summary.json", summary)
        snapshots.close()
        snapshots = None
        output.publish()
        published = True
        return {
            "schema_version": GATE_SCHEMA,
            "output_dir": os.fspath(output.final_path),
            "request_count": len(results),
        }
    finally:
        if not published:
            if http is not None:
                try:
                    http.abort()
                except BaseException:
                    pass
            if observer is not None:
                try:
                    observer.close()
                except BaseException:
                    pass
            for writer in writers:
                try:
                    writer.abort()
                except BaseException:
                    pass
            output.abort()
        if snapshots is not None:
            snapshots.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--http-image-id", required=True)
    parser.add_argument("--docker-network-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    arguments = Arguments(
        output_dir=namespace.output_dir,
        api_key_file=namespace.api_key_file,
        http_image_id=namespace.http_image_id,
        docker_network_id=namespace.docker_network_id,
    )
    try:
        result = execute(arguments)
    except (GateError, COL.CollectorError) as error:
        print(f"SQ8 direct cancellation gate: {error}", file=sys.stderr)
        return 2
    print(compact_json(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
