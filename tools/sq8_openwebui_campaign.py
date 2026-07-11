#!/usr/bin/env python3
"""Continuous sd-journal capture and lifecycle claiming for an SQ8 campaign."""

from __future__ import annotations

import copy
import dataclasses
import datetime
import enum
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable, NoReturn, Protocol, cast


SERVICE_UNIT = "ullm-openai.service"
LIFECYCLE_SCHEMA = "ullm.gateway.lifecycle.v1"
MAX_JOURNAL_LINE_BYTES = 1 << 20
DEFAULT_MAX_JOURNAL_ROWS = 16_384
DEFAULT_MAX_PENDING_EVENTS = 2_048
DEFAULT_MAX_PENDING_BYTES = 16 << 20
SOURCE_WAIT_USEC = 50_000
MAX_SOURCE_WAIT_USEC = 2_147_483_647_000
CHECKPOINT_EMPTY_POLLS = 2

REQUIRED_JOURNAL_FIELDS = (
    "__CURSOR",
    "__MONOTONIC_TIMESTAMP",
    "_BOOT_ID",
    "_PID",
    "_SYSTEMD_UNIT",
    "PRIORITY",
    "MESSAGE",
)

PHASE_ORDER = (
    "preflight",
    "api_contract",
    "openwebui",
    "cancellation",
    "resource_normal",
    "post_header_failure",
    "resource_restart",
    "latency",
    "final",
)
NORMAL_PHASES = frozenset(PHASE_ORDER[:5])
RESTART_PHASES = frozenset(PHASE_ORDER[6:])

LIFECYCLE_FIELDS: dict[str, frozenset[str]] = {
    "request_admitted": frozenset(
        {
            "request_id",
            "completion_id",
            "stream",
            "prompt_tokens",
            "max_completion_tokens",
        }
    ),
    "request_started": frozenset(
        {
            "request_id",
            "completion_id",
            "stream",
            "prompt_tokens",
            "admit_to_start_ns",
        }
    ),
    "request_progress": frozenset(
        {
            "request_id",
            "completion_id",
            "phase",
            "processed_prompt_tokens",
            "prompt_tokens",
        }
    ),
    "request_first_token": frozenset(
        {"request_id", "completion_id", "stream", "completion_tokens"}
    ),
    "request_cancel_requested": frozenset(
        {
            "request_id",
            "completion_id",
            "stream",
            "reason",
            "admit_to_cancel_ns",
        }
    ),
    "request_released": frozenset(
        {
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
        }
    ),
    "worker_fatal": frozenset(
        {"request_id", "completion_id", "reason", "admit_to_fatal_ns"}
    ),
}

BOOT_ID_RE = re.compile(r"[0-9a-f]{32}\Z")


class CampaignJournalError(RuntimeError):
    """A fail-closed campaign journal error without evidence contents."""


class JournalSourceGap(CampaignJournalError):
    """The journal source was invalidated, so continuity is no longer provable."""


def fail(message: str) -> NoReturn:
    raise CampaignJournalError(message)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("journal JSON contains a duplicate key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    fail("journal JSON contains a non-finite number")


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        fail("journal JSON contains a non-finite number")
    return parsed


def _strict_object(raw: bytes, label: str) -> dict[str, Any]:
    if (
        not raw
        or len(raw) > MAX_JOURNAL_LINE_BYTES
        or b"\n" in raw
        or raw.endswith(b"\r")
    ):
        fail(f"{label} is not one bounded LF-free JSON object")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=_parse_float,
            parse_constant=_reject_constant,
        )
    except CampaignJournalError:
        raise
    except (UnicodeError, ValueError, RecursionError):
        fail(f"{label} is not strict UTF-8 JSON")
    if type(value) is not dict:
        fail(f"{label} root is not an object")
    return cast(dict[str, Any], value)


def _bounded_text(value: Any, label: str, *, maximum: int = 65_536) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        fail(f"{label} is not bounded non-empty text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        fail(f"{label} is not strict UTF-8")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail(f"{label} is not an integer >= {minimum}")
    return value


def _decimal_field(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not str or not value.isdecimal():
        fail(f"{label} is not a decimal string")
    parsed = int(value, 10)
    if parsed < minimum:
        fail(f"{label} is below {minimum}")
    return parsed


def _compact_json(value: dict[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        fail("journal record cannot be serialized as strict JSON")


def _canonical_required_bytes(record: dict[str, Any]) -> bytes:
    return _compact_json({field: record[field] for field in REQUIRED_JOURNAL_FIELDS})


def _validate_lifecycle(value: dict[str, Any]) -> dict[str, Any]:
    event_name = value.get("event")
    if type(event_name) is not str or event_name not in LIFECYCLE_FIELDS:
        fail("gateway lifecycle event name is unknown")
    expected = {
        "schema_version",
        "event",
        "observed_monotonic_ns",
    } | set(LIFECYCLE_FIELDS[event_name])
    if set(value) != expected:
        fail("gateway lifecycle fields differ")
    if value["schema_version"] != LIFECYCLE_SCHEMA:
        fail("gateway lifecycle schema differs")
    _integer(value["observed_monotonic_ns"], "gateway observed_monotonic_ns")
    request_id = value.get("request_id")
    completion_id = value.get("completion_id")
    if event_name == "worker_fatal" and request_id is None:
        if completion_id is not None or value["admit_to_fatal_ns"] is not None:
            fail("idle worker_fatal nullable fields differ")
    else:
        _bounded_text(request_id, "gateway request_id", maximum=4096)
        _bounded_text(completion_id, "gateway completion_id", maximum=4096)

    if event_name == "request_admitted":
        if value["stream"] is not True:
            fail("gateway admission is not streaming")
        _integer(value["prompt_tokens"], "gateway prompt_tokens", minimum=1)
        _integer(
            value["max_completion_tokens"],
            "gateway max_completion_tokens",
            minimum=1,
        )
    elif event_name == "request_started":
        if type(value["stream"]) is not bool:
            fail("gateway start stream flag is not boolean")
        _integer(value["prompt_tokens"], "gateway prompt_tokens", minimum=1)
        _integer(value["admit_to_start_ns"], "gateway admit_to_start_ns")
    elif event_name == "request_progress":
        _bounded_text(value["phase"], "gateway progress phase", maximum=4096)
        processed = _integer(
            value["processed_prompt_tokens"],
            "gateway processed_prompt_tokens",
            minimum=1,
        )
        prompt = _integer(value["prompt_tokens"], "gateway prompt_tokens", minimum=1)
        if processed > prompt:
            fail("gateway processed prompt tokens exceed prompt tokens")
    elif event_name == "request_first_token":
        if type(value["stream"]) is not bool or value["completion_tokens"] != 1:
            fail("gateway first-token fields differ")
    elif event_name == "request_cancel_requested":
        if type(value["stream"]) is not bool:
            fail("gateway cancel stream flag is not boolean")
        _bounded_text(value["reason"], "gateway cancel reason", maximum=4096)
        _integer(value["admit_to_cancel_ns"], "gateway admit_to_cancel_ns")
    elif event_name == "request_released":
        if type(value["stream"]) is not bool or value["outcome"] not in {
            "stop",
            "length",
            "cancelled",
        }:
            fail("gateway release stream or outcome differs")
        if value["outcome"] == "cancelled":
            _bounded_text(value["cancel_reason"], "gateway cancel_reason", maximum=4096)
        elif value["cancel_reason"] is not None:
            fail("non-cancelled gateway release has a cancel_reason")
        _integer(value["prompt_tokens"], "gateway release prompt_tokens", minimum=1)
        _integer(value["completion_tokens"], "gateway release completion_tokens")
        if value["reset_complete"] is not True:
            fail("gateway release lacks reset_complete=true")
        admit = _integer(
            value["admit_to_start_ns"], "gateway release admit_to_start_ns"
        )
        duration = _integer(
            value["start_to_release_ns"], "gateway release start_to_release_ns"
        )
        total = _integer(
            value["admit_to_release_ns"], "gateway release admit_to_release_ns"
        )
        if total != admit + duration:
            fail("gateway release duration arithmetic differs")
    elif event_name == "worker_fatal":
        _bounded_text(value["reason"], "gateway worker fatal reason", maximum=4096)
        if request_id is not None:
            _integer(value["admit_to_fatal_ns"], "gateway admit_to_fatal_ns")
    return value


def decode_lifecycle_message(message: str) -> dict[str, Any] | None:
    """Decode only either exact lifecycle MESSAGE framing allowed by the spec."""

    try:
        raw = message.encode("utf-8", errors="strict")
    except UnicodeError:
        fail("journal MESSAGE is not strict UTF-8")
    if raw.startswith(b"{"):
        payload = raw
    elif raw.startswith(b"INFO:     {"):
        payload = raw[len(b"INFO:     ") :]
    else:
        return None
    value = _strict_object(payload, "gateway lifecycle MESSAGE")
    if value.get("schema_version") != LIFECYCLE_SCHEMA:
        return None
    return _validate_lifecycle(value)


@dataclasses.dataclass(frozen=True)
class PidEpoch:
    gateway_pid: int
    worker_pid: int

    def __post_init__(self) -> None:
        if type(self.gateway_pid) is not int or self.gateway_pid < 1:
            fail("epoch gateway PID is invalid")
        if type(self.worker_pid) is not int or self.worker_pid < 1:
            fail("epoch worker PID is invalid")


@dataclasses.dataclass(frozen=True)
class BundleLifecycleClaim:
    raw: bytes
    phase: str
    case_id: str


@dataclasses.dataclass(frozen=True)
class ClaimedGatewayEvent:
    phase: str
    case_id: str
    fields: dict[str, Any]

    def session_hook_record(self) -> dict[str, Any]:
        return {
            "record_type": "gateway_event",
            "phase": self.phase,
            "case_id": self.case_id,
            "fields": copy.deepcopy(self.fields),
        }


class JournalSource(Protocol):
    """A source used exclusively by the capture's reader thread."""

    def open_after(self, unit: str, boot_id: str) -> str: ...

    def read_next(self, timeout_usec: int) -> bytes | None: ...

    def close(self) -> None: ...


def _journal_monotonic_usec(value: Any) -> int:
    if type(value) is str and value.isdecimal():
        return int(value, 10)
    if type(value) is int and value >= 0:
        return value
    try:
        delta = value.timestamp
        if type(delta) is datetime.timedelta:
            return (
                delta.days * 86_400_000_000
                + delta.seconds * 1_000_000
                + delta.microseconds
            )
    except (AttributeError, TypeError, OverflowError):
        pass
    fail("direct sd-journal monotonic timestamp is invalid")


def _journal_boot_id(value: Any) -> str:
    if isinstance(value, uuid.UUID):
        return value.hex
    return str(value).replace("-", "")


def _source_entry_bytes(entry: dict[str, Any], boot_id: str) -> bytes:
    for field in REQUIRED_JOURNAL_FIELDS:
        if field not in entry:
            fail(f"direct sd-journal entry lacks {field}")
    message = entry["MESSAGE"]
    if type(message) is not str:
        fail("direct sd-journal MESSAGE is not text")
    boot_value = _journal_boot_id(entry["_BOOT_ID"])
    if boot_value != boot_id:
        fail("direct sd-journal boot ID differs")
    record = {
        "__CURSOR": str(entry["__CURSOR"]),
        "__MONOTONIC_TIMESTAMP": str(
            _journal_monotonic_usec(entry["__MONOTONIC_TIMESTAMP"])
        ),
        "_BOOT_ID": boot_value,
        "_PID": str(entry["_PID"]),
        "_SYSTEMD_UNIT": str(entry["_SYSTEMD_UNIT"]),
        "PRIORITY": str(entry["PRIORITY"]),
        "MESSAGE": message,
    }
    return _compact_json(record)


class SystemdJournalSource:
    """Production source backed by ``systemd.journal.Reader``."""

    def __init__(self) -> None:
        self._reader: Any | None = None
        self._journal: Any | None = None
        self._boot_id: str | None = None
        self._last_cursor: str | None = None

    def open_after(self, unit: str, boot_id: str) -> str:
        if self._reader is not None:
            fail("direct sd-journal source is already open")
        try:
            from systemd import journal  # type: ignore[import-untyped]

            reader = journal.Reader()
            reader.add_match(_SYSTEMD_UNIT=unit)
            reader.this_boot()
            reader.seek_tail()
            anchor = reader.get_previous()
        except (ImportError, OSError, ValueError):
            fail("failed to initialize the direct sd-journal reader")
        if not anchor:
            fail("service journal has no campaign start cursor")
        anchor_raw = _source_entry_bytes(anchor, boot_id)
        anchor_record = _strict_object(anchor_raw, "campaign journal anchor")
        if anchor_record["_SYSTEMD_UNIT"] != unit:
            fail("campaign journal anchor unit differs")
        cursor = _bounded_text(anchor_record["__CURSOR"], "campaign start cursor")
        try:
            reader.seek_cursor(cursor)
            positioned = reader.get_next()
        except (OSError, ValueError):
            fail("failed to position the direct sd-journal reader")
        if not positioned or str(positioned.get("__CURSOR")) != cursor:
            fail("direct sd-journal start cursor positioning differs")
        self._reader = reader
        self._journal = journal
        self._boot_id = boot_id
        self._last_cursor = cursor
        return cursor

    def _entry_bytes(self, entry: dict[str, Any], boot_id: str) -> bytes:
        raw = _source_entry_bytes(entry, boot_id)
        record = _strict_object(raw, "direct sd-journal entry")
        self._last_cursor = _bounded_text(
            record["__CURSOR"], "direct sd-journal entry cursor"
        )
        return raw

    def _recover_after_invalidate(self, reader: Any, boot_id: str) -> bytes | None:
        cursor = self._last_cursor
        if cursor is None:
            raise JournalSourceGap("direct sd-journal continuity cursor is absent")
        try:
            reader.seek_cursor(cursor)
            positioned = reader.get_next()
            if (
                not positioned
                or type(positioned.get("__CURSOR")) is not str
                or positioned["__CURSOR"] != cursor
            ):
                raise JournalSourceGap(
                    "direct sd-journal continuity cursor is unavailable"
                )
            entry = reader.get_next()
        except JournalSourceGap:
            raise
        except (OSError, ValueError) as error:
            raise JournalSourceGap(
                "direct sd-journal continuity could not be verified"
            ) from error
        if entry:
            return self._entry_bytes(entry, boot_id)
        return None

    def read_next(self, timeout_usec: int) -> bytes | None:
        if (
            type(timeout_usec) is not int
            or timeout_usec < 1
            or timeout_usec > MAX_SOURCE_WAIT_USEC
        ):
            fail("direct sd-journal timeout is not bounded positive microseconds")
        timeout_msec = max(1, (timeout_usec + 999) // 1000)
        reader = self._reader
        journal = self._journal
        boot_id = self._boot_id
        if reader is None or journal is None or boot_id is None:
            fail("direct sd-journal source is not open")
        reader = cast(Any, reader)
        journal = cast(Any, journal)
        try:
            entry = reader.get_next()
            if entry:
                return self._entry_bytes(entry, boot_id)
            result = reader.wait(timeout_msec)
        except (OSError, ValueError):
            fail("direct sd-journal read failed")
        if result == journal.INVALIDATE:
            return self._recover_after_invalidate(reader, boot_id)
        if result not in {journal.NOP, journal.APPEND}:
            fail("direct sd-journal wait result is invalid")
        if result == journal.APPEND:
            try:
                entry = reader.get_next()
            except (OSError, ValueError):
                fail("direct sd-journal read after append failed")
            if entry:
                return self._entry_bytes(entry, boot_id)
        return None

    def close(self) -> None:
        self._reader = None
        self._journal = None
        self._boot_id = None
        self._last_cursor = None


class _CaptureState(enum.Enum):
    NEW = "new"
    RUNNING = "running"
    SEALED = "sealed"
    ABORTED = "aborted"


@dataclasses.dataclass(frozen=True)
class _JournalRecord:
    cursor: str
    monotonic_usec: int
    pid: int
    message: str
    canonical_required: bytes


@dataclasses.dataclass(frozen=True)
class _PendingLifecycle:
    record: _JournalRecord
    event: dict[str, Any]
    epoch: str

    @property
    def retained_bytes(self) -> int:
        return len(self.record.canonical_required) + len(_compact_json(self.event))


class CampaignJournalCapture:
    """Capture a full campaign journal and claim lifecycle rows exactly once."""

    def __init__(
        self,
        final_path: Path,
        boot_id: str,
        normal_epoch: PidEpoch,
        *,
        scan_raw: Callable[[bytes, str], None],
        source: JournalSource | None = None,
        max_journal_rows: int = DEFAULT_MAX_JOURNAL_ROWS,
        max_pending_events: int = DEFAULT_MAX_PENDING_EVENTS,
        max_pending_bytes: int = DEFAULT_MAX_PENDING_BYTES,
    ):
        if BOOT_ID_RE.fullmatch(boot_id) is None:
            fail("campaign boot ID syntax differs")
        if final_path.name.endswith(".incomplete"):
            fail("campaign final journal path is already incomplete")
        if any(
            type(value) is not int or value < 1
            for value in (max_journal_rows, max_pending_events, max_pending_bytes)
        ):
            fail("campaign journal bounds are invalid")
        if not callable(scan_raw):
            fail("campaign journal evidence scanner is not callable")
        self.final_path = final_path
        self.incomplete_path = final_path.with_name(final_path.name + ".incomplete")
        self.boot_id = boot_id
        self.normal_epoch = normal_epoch
        self.restart_epoch: PidEpoch | None = None
        self.scan_raw = scan_raw
        self.source = source if source is not None else SystemdJournalSource()
        self.max_journal_rows = max_journal_rows
        self.max_pending_events = max_pending_events
        self.max_pending_bytes = max_pending_bytes

        self._state = _CaptureState.NEW
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._error: BaseException | None = None
        self._initialized = False
        self._start_cursor: str | None = None
        self._last_cursor: str | None = None
        self._last_monotonic_usec = -1
        self._last_lifecycle_ns = -1
        self._seen_cursors: set[str] = set()
        self._claimed_cursors: set[str] = set()
        self._row_count = 0
        self._pending: deque[_PendingLifecycle] = deque()
        self._pending_bytes = 0
        self._empty_polls = 0
        self._last_claimed_phase_rank = -1
        self._checkpoint_phase_rank = -1
        self._restart_armed = False
        self._restart_discovered_pid: int | None = None
        self._switched_to_restart = False
        self._sealing = False
        self._fatal_count = 0
        self._owns_final_path = False

    @property
    def start_cursor(self) -> str:
        with self._condition:
            if self._start_cursor is None:
                fail("campaign journal capture has no start cursor")
            return self._start_cursor

    @property
    def last_cursor(self) -> str:
        with self._condition:
            if self._last_cursor is None:
                fail("campaign journal capture has no captured cursor")
            return self._last_cursor

    @property
    def discovered_restart_gateway_pid(self) -> int | None:
        with self._condition:
            return self._restart_discovered_pid

    def start(self) -> str:
        if self._state is not _CaptureState.NEW:
            fail("campaign journal capture cannot be started twice")
        if not self.final_path.parent.is_dir() or self.final_path.parent.is_symlink():
            fail("campaign journal parent is not a real directory")
        if self.final_path.exists() or self.final_path.is_symlink():
            fail("campaign final journal path already exists")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            self._fd = os.open(self.incomplete_path, flags, 0o600)
        except OSError:
            fail("failed to create campaign journal incomplete file")
        self._state = _CaptureState.RUNNING
        self._thread = threading.Thread(
            target=self._reader_main,
            name="sq8-campaign-journal",
            daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic_ns() + 5_000_000_000
        try:
            with self._condition:
                while not self._initialized and self._error is None:
                    remaining = deadline - time.monotonic_ns()
                    if remaining <= 0:
                        fail("campaign journal reader initialization timed out")
                    self._condition.wait(remaining / 1_000_000_000)
                self._raise_thread_error_locked()
                assert self._start_cursor is not None
                return self._start_cursor
        except BaseException:
            self.abort()
            raise

    def arm_restart_transition(self) -> None:
        with self._condition:
            self._require_running_locked()
            self._raise_thread_error_locked()
            if self._restart_armed or self.restart_epoch is not None:
                fail("campaign restart transition is already armed")
            if (
                self._checkpoint_phase_rank != PHASE_ORDER.index("resource_normal")
                or self._pending
            ):
                fail("campaign restart transition precedes the normal checkpoint")
            self._restart_armed = True

    def confirm_restart_epoch(self, epoch: PidEpoch) -> None:
        with self._condition:
            self._require_running_locked()
            self._raise_thread_error_locked()
            if not self._restart_armed or self.restart_epoch is not None:
                fail("campaign restart epoch confirmation is out of order")
            if (
                epoch.gateway_pid == self.normal_epoch.gateway_pid
                or epoch.worker_pid == self.normal_epoch.worker_pid
            ):
                fail("campaign restart PID identities did not both change")
            if (
                self._restart_discovered_pid is not None
                and self._restart_discovered_pid != epoch.gateway_pid
            ):
                fail("discovered restart gateway PID differs from the probe")
            self.restart_epoch = epoch

    def claim_bundle_records(
        self,
        claims: Iterable[BundleLifecycleClaim],
        deadline_ns: int,
    ) -> list[ClaimedGatewayEvent]:
        result: list[ClaimedGatewayEvent] = []
        for claim in claims:
            expected_record, expected_event = self._bundle_claim_value(claim)
            with self._condition:
                self._wait_for_pending_locked(deadline_ns)
                pending = self._pending[0]
                if pending.record.cursor != expected_record.cursor:
                    fail("bundle lifecycle cursor is not the next campaign cursor")
                if (
                    pending.record.canonical_required
                    != expected_record.canonical_required
                    or pending.event != expected_event
                ):
                    fail("bundle lifecycle raw bytes differ from campaign journal")
                result.append(
                    self._claim_pending_locked(pending, claim.phase, claim.case_id)
                )
        return result

    def claim_completion_trace(
        self,
        completion_id: str,
        phase: str,
        case_id: str,
        deadline_ns: int,
    ) -> list[ClaimedGatewayEvent]:
        _bounded_text(completion_id, "resource completion_id", maximum=4096)
        self._validate_phase_case(phase, case_id)
        with self._condition:
            while True:
                self._raise_thread_error_locked()
                self._require_running_locked()
                if self._pending:
                    first = self._pending[0]
                    if first.event.get("completion_id") != completion_id:
                        fail("an earlier lifecycle trace remains unclaimed")
                    trace: list[_PendingLifecycle] = []
                    request_id = first.event.get("request_id")
                    terminal_index: int | None = None
                    for index, pending in enumerate(self._pending):
                        if (
                            pending.event.get("completion_id") != completion_id
                            or pending.event.get("request_id") != request_id
                        ):
                            fail(
                                "resource lifecycle trace is interleaved or incomplete"
                            )
                        trace.append(pending)
                        if pending.event["event"] in {
                            "request_released",
                            "worker_fatal",
                        }:
                            terminal_index = index
                            break
                    if terminal_index is not None:
                        if trace[0].event["event"] != "request_admitted":
                            fail(
                                "resource lifecycle trace does not begin with admission"
                            )
                        return [
                            self._claim_pending_locked(pending, phase, case_id)
                            for pending in trace
                        ]
                remaining = deadline_ns - time.monotonic_ns()
                if remaining <= 0:
                    fail("resource lifecycle trace claim timed out")
                self._condition.wait(remaining / 1_000_000_000)

    def wait_quiet(self, deadline_ns: int) -> str:
        """Wait through a negative-request quiet window without advancing phase."""

        with self._condition:
            self._require_running_locked()
            target_empty_polls: int | None = None
            while True:
                self._raise_thread_error_locked()
                if self._pending:
                    fail("campaign quiet window contains an unclaimed lifecycle event")
                remaining = deadline_ns - time.monotonic_ns()
                if remaining <= 0 and target_empty_polls is None:
                    target_empty_polls = self._empty_polls + CHECKPOINT_EMPTY_POLLS
                if (
                    target_empty_polls is not None
                    and self._empty_polls >= target_empty_polls
                ):
                    cursor = self._last_cursor or self._start_cursor
                    if cursor is None:
                        fail("campaign quiet window has no journal boundary cursor")
                    return cursor
                wait_seconds = (
                    remaining / 1_000_000_000
                    if remaining > 0
                    else SOURCE_WAIT_USEC / 1_000_000
                )
                self._condition.wait(wait_seconds)

    def checkpoint(self, phase: str, deadline_ns: int) -> str:
        if phase not in PHASE_ORDER:
            fail("campaign checkpoint phase is invalid")
        phase_rank = PHASE_ORDER.index(phase)
        with self._condition:
            self._require_running_locked()
            self._raise_thread_error_locked()
            if phase_rank < self._checkpoint_phase_rank:
                fail("campaign checkpoint phase regressed")
            target_empty_polls = self._empty_polls + CHECKPOINT_EMPTY_POLLS
            while self._empty_polls < target_empty_polls:
                self._raise_thread_error_locked()
                remaining = deadline_ns - time.monotonic_ns()
                if remaining <= 0:
                    fail("campaign journal checkpoint timed out")
                self._condition.wait(remaining / 1_000_000_000)
            self._raise_thread_error_locked()
            if self._pending:
                fail("campaign phase checkpoint has unclaimed lifecycle events")
            if self._last_claimed_phase_rank > phase_rank:
                fail("campaign checkpoint precedes a claimed lifecycle phase")
            self._checkpoint_phase_rank = phase_rank
            cursor = self._last_cursor or self._start_cursor
            if cursor is None:
                fail("campaign checkpoint has no journal boundary cursor")
            return cursor

    def seal(self, expected_final_cursor: str, deadline_ns: int) -> str:
        _bounded_text(expected_final_cursor, "expected final journal cursor")
        with self._condition:
            self._require_running_locked()
            self._raise_thread_error_locked()
            if self._checkpoint_phase_rank != PHASE_ORDER.index("final"):
                fail("campaign journal requires a final phase checkpoint")
            if self._pending:
                fail("campaign journal cannot seal with unclaimed lifecycle events")
            if (
                not self._restart_armed
                or self.restart_epoch is None
                or not self._switched_to_restart
            ):
                fail("campaign journal does not contain one confirmed PID epoch switch")
            if self._fatal_count != 1:
                fail("campaign journal does not contain exactly one worker_fatal")
            if self._last_cursor is None:
                fail("campaign journal cannot seal without a final cursor")
            final_cursor = self._last_cursor
            if final_cursor != expected_final_cursor:
                fail("campaign journal advanced after the run_end final cursor")
            target_empty_polls = self._empty_polls + 1
            self._sealing = True
            self._condition.notify_all()
            while self._empty_polls < target_empty_polls and self._error is None:
                remaining = deadline_ns - time.monotonic_ns()
                if remaining <= 0:
                    self._sealing = False
                    fail("campaign journal final seal timed out")
                self._condition.wait(remaining / 1_000_000_000)
            self._raise_thread_error_locked()
            self._stop.set()
        self._join_reader(deadline_ns)
        try:
            assert self._fd is not None
            os.fsync(self._fd)
            os.close(self._fd)
            self._fd = None
            os.link(self.incomplete_path, self.final_path, follow_symlinks=False)
            self._owns_final_path = True
            directory_fd = os.open(self.final_path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self.incomplete_path.unlink()
            directory_fd = os.open(self.final_path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            self.abort()
            fail("failed to publish the campaign journal")
        self._state = _CaptureState.SEALED
        return final_cursor

    def abort(self) -> None:
        if self._state in {_CaptureState.ABORTED, _CaptureState.SEALED}:
            return
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        try:
            self.incomplete_path.unlink(missing_ok=True)
        except OSError:
            pass
        if self._owns_final_path:
            try:
                self.final_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._owns_final_path = False
        self._state = _CaptureState.ABORTED

    def _reader_main(self) -> None:
        try:
            start_cursor = self.source.open_after(SERVICE_UNIT, self.boot_id)
            _bounded_text(start_cursor, "campaign start cursor")
            with self._condition:
                self._start_cursor = start_cursor
                self._initialized = True
                self._condition.notify_all()
            while not self._stop.is_set():
                raw = self.source.read_next(SOURCE_WAIT_USEC)
                with self._condition:
                    if raw is None:
                        self._empty_polls += 1
                        self._condition.notify_all()
                        continue
                    if self._sealing:
                        fail("a journal row appeared after final seal began")
                    self._consume_raw_locked(raw)
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error = error
                self._initialized = True
                self._condition.notify_all()
        finally:
            try:
                self.source.close()
            except BaseException as error:
                with self._condition:
                    if self._error is None:
                        self._error = error
                    self._condition.notify_all()

    def _consume_raw_locked(self, raw: bytes) -> None:
        record, event = self._parse_journal_record(raw, "campaign journal record")
        if record.cursor == self._start_cursor:
            fail("campaign journal did not advance past its start cursor")
        if record.cursor in self._seen_cursors:
            fail("campaign journal cursor is duplicated")
        if record.monotonic_usec < self._last_monotonic_usec:
            fail("campaign journal monotonic timestamps regressed")
        self._row_count += 1
        if self._row_count > self.max_journal_rows:
            fail("campaign journal row count exceeds its bound")
        try:
            self.scan_raw(raw, "service journal evidence")
        except CampaignJournalError:
            raise
        except BaseException as error:
            raise CampaignJournalError(
                "campaign journal evidence scan failed"
            ) from error
        self._write_all(raw + b"\n")
        self._seen_cursors.add(record.cursor)
        self._last_cursor = record.cursor
        self._last_monotonic_usec = record.monotonic_usec
        if event is None:
            return
        observed_ns = cast(int, event["observed_monotonic_ns"])
        if observed_ns < self._last_lifecycle_ns:
            fail("campaign lifecycle observed timestamps regressed")
        if record.monotonic_usec < observed_ns // 1000:
            fail("campaign journal timestamp precedes lifecycle observation")
        epoch = self._classify_lifecycle_pid_locked(record.pid)
        if event["event"] == "worker_fatal":
            self._fatal_count += 1
            if self._fatal_count > 1 or epoch != "normal":
                fail("worker_fatal is outside the sole normal-to-restart transition")
        pending = _PendingLifecycle(record, event, epoch)
        retained = pending.retained_bytes
        if (
            len(self._pending) >= self.max_pending_events
            or self._pending_bytes + retained > self.max_pending_bytes
        ):
            fail("campaign lifecycle pending queue overflowed")
        self._pending.append(pending)
        self._pending_bytes += retained
        self._last_lifecycle_ns = observed_ns

    def _parse_journal_record(
        self, raw: bytes, label: str
    ) -> tuple[_JournalRecord, dict[str, Any] | None]:
        value = _strict_object(raw, label)
        for field in REQUIRED_JOURNAL_FIELDS:
            if field not in value:
                fail(f"{label} lacks {field}")
        cursor = _bounded_text(value["__CURSOR"], f"{label} cursor", maximum=65_536)
        monotonic_usec = _decimal_field(
            value["__MONOTONIC_TIMESTAMP"], f"{label} monotonic"
        )
        pid = _decimal_field(value["_PID"], f"{label} PID", minimum=1)
        if value["_BOOT_ID"] != self.boot_id:
            fail(f"{label} boot ID differs")
        if value["_SYSTEMD_UNIT"] != SERVICE_UNIT:
            fail(f"{label} systemd unit differs")
        priority = _decimal_field(value["PRIORITY"], f"{label} priority")
        if priority > 7:
            fail(f"{label} priority is outside syslog range")
        message = value["MESSAGE"]
        if type(message) is not str:
            fail(f"{label} MESSAGE is not text")
        try:
            message.encode("utf-8", errors="strict")
        except UnicodeError:
            fail(f"{label} MESSAGE is not strict UTF-8")
        event = decode_lifecycle_message(message)
        return (
            _JournalRecord(
                cursor=cursor,
                monotonic_usec=monotonic_usec,
                pid=pid,
                message=message,
                canonical_required=_canonical_required_bytes(value),
            ),
            event,
        )

    def _classify_lifecycle_pid_locked(self, pid: int) -> str:
        if not self._switched_to_restart:
            if pid == self.normal_epoch.gateway_pid:
                return "normal"
            confirmed_pid = (
                None if self.restart_epoch is None else self.restart_epoch.gateway_pid
            )
            if not self._restart_armed or (
                confirmed_pid is not None and pid != confirmed_pid
            ):
                fail("campaign lifecycle PID is outside the active normal epoch")
            if (
                self._restart_discovered_pid is not None
                and self._restart_discovered_pid != pid
            ):
                fail("campaign observed more than one candidate restart PID")
            self._restart_discovered_pid = pid
            self._switched_to_restart = True
            return "restart"
        restart_pid = (
            self._restart_discovered_pid
            if self.restart_epoch is None
            else self.restart_epoch.gateway_pid
        )
        if pid != restart_pid:
            fail("campaign lifecycle PID changed after the sole restart switch")
        return "restart"

    def _bundle_claim_value(
        self, claim: BundleLifecycleClaim
    ) -> tuple[_JournalRecord, dict[str, Any]]:
        self._validate_phase_case(claim.phase, claim.case_id)
        record, event = self._parse_journal_record(claim.raw, "bundle journal record")
        if event is None:
            fail("bundle claim is not a gateway lifecycle record")
        return record, event

    def _claim_pending_locked(
        self, pending: _PendingLifecycle, phase: str, case_id: str
    ) -> ClaimedGatewayEvent:
        self._validate_phase_case(phase, case_id)
        phase_rank = PHASE_ORDER.index(phase)
        if phase_rank < self._last_claimed_phase_rank:
            fail("claimed campaign lifecycle phase regressed")
        if phase_rank <= self._checkpoint_phase_rank:
            fail("claimed lifecycle belongs to an already checkpointed phase")
        if pending.epoch == "normal" and phase not in NORMAL_PHASES | {
            "post_header_failure"
        }:
            fail("normal PID lifecycle was claimed in a restart phase")
        if pending.epoch == "restart" and phase not in RESTART_PHASES | {
            "post_header_failure"
        }:
            fail("restart PID lifecycle was claimed in a normal phase")
        if pending.event["event"] == "worker_fatal" and (
            phase != "post_header_failure" or pending.epoch != "normal"
        ):
            fail("worker_fatal claim phase differs")
        actual = self._pending.popleft()
        assert actual is pending
        self._pending_bytes -= pending.retained_bytes
        if pending.record.cursor in self._claimed_cursors:
            fail("campaign lifecycle cursor was claimed twice")
        self._claimed_cursors.add(pending.record.cursor)
        self._last_claimed_phase_rank = phase_rank
        fields = {
            "journal_cursor": pending.record.cursor,
            "journal_monotonic_usec": pending.record.monotonic_usec,
            "journal_pid": pending.record.pid,
            "message": pending.record.message,
            "message_sha256": hashlib.sha256(
                pending.record.message.encode("utf-8")
            ).hexdigest(),
            "event": copy.deepcopy(pending.event),
        }
        return ClaimedGatewayEvent(phase, case_id, fields)

    def _wait_for_pending_locked(self, deadline_ns: int) -> None:
        while not self._pending:
            self._raise_thread_error_locked()
            self._require_running_locked()
            remaining = deadline_ns - time.monotonic_ns()
            if remaining <= 0:
                fail("bundle lifecycle claim timed out")
            self._condition.wait(remaining / 1_000_000_000)

    def _validate_phase_case(self, phase: str, case_id: str) -> None:
        if phase not in PHASE_ORDER:
            fail("campaign lifecycle claim phase is invalid")
        _bounded_text(case_id, "campaign lifecycle case_id", maximum=4096)

    def _write_all(self, raw: bytes) -> None:
        if self._fd is None:
            fail("campaign journal output is not open")
        fd = self._fd
        view = memoryview(raw)
        while view:
            try:
                written = os.write(fd, view)
            except OSError:
                fail("failed to stream the campaign journal")
            if written <= 0:
                fail("campaign journal write made no progress")
            view = view[written:]

    def _require_running_locked(self) -> None:
        if self._state is not _CaptureState.RUNNING:
            fail("campaign journal capture is not running")

    def _raise_thread_error_locked(self) -> None:
        if self._error is None:
            return
        if isinstance(self._error, CampaignJournalError):
            raise self._error
        raise CampaignJournalError("campaign journal reader failed") from self._error

    def _join_reader(self, deadline_ns: int) -> None:
        thread = self._thread
        if thread is None:
            fail("campaign journal reader thread is absent")
        remaining = max(0, deadline_ns - time.monotonic_ns()) / 1_000_000_000
        thread.join(timeout=remaining)
        if thread.is_alive():
            self.abort()
            fail("campaign journal reader did not terminate")
        with self._condition:
            self._raise_thread_error_locked()


__all__ = [
    "BundleLifecycleClaim",
    "CampaignJournalCapture",
    "CampaignJournalError",
    "ClaimedGatewayEvent",
    "JournalSource",
    "JournalSourceGap",
    "PidEpoch",
    "SERVICE_UNIT",
    "SystemdJournalSource",
    "decode_lifecycle_message",
]
