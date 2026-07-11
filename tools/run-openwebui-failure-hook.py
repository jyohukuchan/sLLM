#!/usr/bin/env python3
"""Run the formal failure gate and emit its collector restart-hook records."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NoReturn


HOOK_SCHEMA = "ullm.sq8.openwebui_release.hook.v1"
GATE_SCHEMA = "ullm.openwebui.failure_gate.v1"
BROWSER_SCHEMA = "ullm.openwebui.failure_smoke.v1"
BROWSER_CASE = "post_header_worker_failure"
PHASE = "post_header_failure"
SUCCESS_STDOUT = b"OpenWebUI failure gate passed\n"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

MAX_JSON_BYTES = 1024 * 1024
MAX_BROWSER_STDOUT_BYTES = 4 * 1024 * 1024
MAX_JOURNAL_BYTES = 64 * 1024 * 1024
MAX_SCREENSHOT_BYTES = 64 * 1024 * 1024
MAX_GATE_SOURCE_BYTES = 4 * 1024 * 1024
MAX_BROWSER_SCRIPT_BYTES = 4 * 1024 * 1024
MAX_SUBPROCESS_STDOUT_BYTES = 256
MAX_SUBPROCESS_STDERR_BYTES = 64 * 1024
READ_CHUNK_BYTES = 64 * 1024

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_ACTIONS = (
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
EXPECTED_SELECTORS = (
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

ROOT_LAYOUT = {
    "browser",
    "fault-injection.json",
    "readiness-evidence.json",
    "service-journal.raw.jsonl",
    "summary.json",
}
BROWSER_LAYOUT = {
    "browser-stdout.jsonl",
    "openwebui-failure-summary.json",
    "post-header-failure.png",
}
ROOT_FILE_MODES = {
    "fault-injection.json": 0o600,
    "readiness-evidence.json": 0o600,
    "service-journal.raw.jsonl": 0o600,
    "summary.json": 0o600,
}
BROWSER_FILE_MODES = {
    "browser-stdout.jsonl": 0o600,
    "openwebui-failure-summary.json": 0o400,
    "post-header-failure.png": 0o400,
}


class FailureHookError(RuntimeError):
    """A fail-closed adapter error whose message never needs to be published."""


def fail(message: str) -> NoReturn:
    raise FailureHookError(message)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("JSON object contains a duplicate key")
        result[key] = value
    return result


def _reject_nonfinite(_value: str) -> NoReturn:
    fail("JSON contains a non-finite number")


def strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    if not raw or len(raw) > MAX_JSON_BYTES:
        fail(f"{label} size differs")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError):
        fail(f"{label} is not strict UTF-8 JSON")
    if not isinstance(value, dict):
        fail(f"{label} is not a JSON object")
    return value


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
        fail("hook JSON encoding failed")


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        fail(f"{label} fields differ")
    return value


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
    converted = int(value, 10)
    if str(converted) != value:
        fail(f"{label} is not canonical decimal")
    return converted


def nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        fail(f"{label} is not a nonempty string")
    return value


def sha256_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} is not a SHA-256 value")
    return value


def boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        fail(f"{label} is not boolean")
    return value


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _source_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def snapshot_source(path: Path, label: str, *, maximum: int) -> SourceSnapshot:
    descriptor = -1
    digest = hashlib.sha256()
    total = 0
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            fail(f"{label} is not a regular non-symlink file")
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if _source_identity(opened) != _source_identity(before):
            fail(f"{label} identity changed while opening")
        while True:
            chunk = os.read(descriptor, READ_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                fail(f"{label} exceeds its size bound")
            digest.update(chunk)
        after = os.fstat(descriptor)
    except OSError:
        fail(f"failed to snapshot {label}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity = _source_identity(opened)
    if identity != _source_identity(after) or opened.st_size != total:
        fail(f"{label} changed while reading")
    return SourceSnapshot(path, identity, total, digest.hexdigest())


def require_unchanged_source(
    expected: SourceSnapshot, label: str, *, maximum: int
) -> None:
    observed = snapshot_source(expected.path, label, maximum=maximum)
    if (
        observed.identity != expected.identity
        or observed.size != expected.size
        or observed.sha256 != expected.sha256
    ):
        fail(f"{label} changed during failure gate execution")


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@dataclasses.dataclass(frozen=True)
class StreamEvidence:
    size: int
    sha256: str
    line_count: int
    prefix: bytes


@dataclasses.dataclass(frozen=True)
class SourceSnapshot:
    path: Path
    identity: tuple[int, int, int, int, int, int, int]
    size: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class BundleBindings:
    gate_source_sha256: str
    browser_script_sha256: str
    browser_image_reference_sha256: str
    probe_image_reference_sha256: str
    service_unit_sha256: str


class BundleSnapshot:
    """Read one already-published bundle through stable directory descriptors."""

    def __init__(self, path: Path):
        self.path = path
        self.root_fd = -1
        self.browser_fd = -1
        self.root_stat: os.stat_result | None = None
        self.browser_stat: os.stat_result | None = None
        self.file_identities: dict[
            tuple[bool, str], tuple[int, int, int, int, int, int, int, int]
        ] = {}

    def __enter__(self) -> BundleSnapshot:
        try:
            before = self.path.lstat()
            flags = (
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0)
            )
            self.root_fd = os.open(self.path, flags)
            opened = os.fstat(self.root_fd)
            if _stat_identity(before) != _stat_identity(opened):
                fail("failure bundle identity changed while opening")
            self._validate_directory(opened, 0o700, "failure bundle")
            if set(os.listdir(self.root_fd)) != ROOT_LAYOUT:
                fail("failure bundle layout differs")
            self.browser_fd = os.open("browser", flags, dir_fd=self.root_fd)
            browser = os.fstat(self.browser_fd)
            self._validate_directory(browser, 0o700, "failure browser directory")
            if set(os.listdir(self.browser_fd)) != BROWSER_LAYOUT:
                fail("failure browser layout differs")
            self.root_stat = opened
            self.browser_stat = browser
        except OSError:
            self.close()
            fail("failure bundle is unavailable")
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self.browser_fd >= 0:
            os.close(self.browser_fd)
            self.browser_fd = -1
        if self.root_fd >= 0:
            os.close(self.root_fd)
            self.root_fd = -1

    @staticmethod
    def _validate_directory(value: os.stat_result, mode: int, label: str) -> None:
        if (
            not stat.S_ISDIR(value.st_mode)
            or stat.S_IMODE(value.st_mode) != mode
            or value.st_uid != os.geteuid()
        ):
            fail(f"{label} ownership or mode differs")

    def _directory_fd(self, browser: bool) -> int:
        descriptor = self.browser_fd if browser else self.root_fd
        if descriptor < 0:
            fail("failure bundle snapshot is closed")
        return descriptor

    def read_file(
        self,
        name: str,
        *,
        browser: bool,
        maximum: int,
    ) -> bytes:
        directory_fd = self._directory_fd(browser)
        expected_mode = BROWSER_FILE_MODES[name] if browser else ROOT_FILE_MODES[name]
        descriptor = -1
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            opened = os.fstat(descriptor)
            self._validate_file(opened, expected_mode, name)
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(READ_CHUNK_BYTES, maximum + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > maximum:
                    fail(f"{name} exceeds its size bound")
            after = os.fstat(descriptor)
        except OSError:
            fail(f"failed to read {name}")
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        raw = b"".join(chunks)
        if _stat_identity(opened) != _stat_identity(after) or opened.st_size != len(
            raw
        ):
            fail(f"{name} changed while reading")
        self.file_identities[(browser, name)] = self._file_identity(after)
        return raw

    def stream_file(
        self,
        name: str,
        *,
        browser: bool,
        maximum: int,
    ) -> StreamEvidence:
        directory_fd = self._directory_fd(browser)
        expected_mode = BROWSER_FILE_MODES[name] if browser else ROOT_FILE_MODES[name]
        descriptor = -1
        digest = hashlib.sha256()
        total = 0
        line_count = 0
        prefix = bytearray()
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            opened = os.fstat(descriptor)
            self._validate_file(opened, expected_mode, name)
            while True:
                chunk = os.read(descriptor, READ_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum:
                    fail(f"{name} exceeds its size bound")
                digest.update(chunk)
                line_count += chunk.count(b"\n")
                if len(prefix) < 16:
                    prefix.extend(chunk[: 16 - len(prefix)])
            after = os.fstat(descriptor)
        except OSError:
            fail(f"failed to read {name}")
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if _stat_identity(opened) != _stat_identity(after) or opened.st_size != total:
            fail(f"{name} changed while reading")
        self.file_identities[(browser, name)] = self._file_identity(after)
        return StreamEvidence(total, digest.hexdigest(), line_count, bytes(prefix))

    @staticmethod
    def _file_identity(
        value: os.stat_result,
    ) -> tuple[int, int, int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    @staticmethod
    def _validate_file(value: os.stat_result, mode: int, label: str) -> None:
        if (
            not stat.S_ISREG(value.st_mode)
            or stat.S_IMODE(value.st_mode) != mode
            or value.st_uid != os.geteuid()
            or value.st_nlink != 1
        ):
            fail(f"{label} ownership, mode, or link count differs")

    def seal(self) -> None:
        if self.root_stat is None or self.browser_stat is None:
            fail("failure bundle snapshot was not opened")
        try:
            root_after = os.fstat(self.root_fd)
            browser_after = os.fstat(self.browser_fd)
            path_after = self.path.lstat()
            root_entries = set(os.listdir(self.root_fd))
            browser_entries = set(os.listdir(self.browser_fd))
        except OSError:
            fail("failure bundle changed before sealing")
        if (
            _stat_identity(root_after) != _stat_identity(self.root_stat)
            or _stat_identity(path_after) != _stat_identity(self.root_stat)
            or _stat_identity(browser_after) != _stat_identity(self.browser_stat)
            or root_entries != ROOT_LAYOUT
            or browser_entries != BROWSER_LAYOUT
        ):
            fail("failure bundle changed before sealing")
        expected_files = {
            *((False, name) for name in ROOT_FILE_MODES),
            *((True, name) for name in BROWSER_FILE_MODES),
        }
        if set(self.file_identities) != expected_files:
            fail("failure bundle was not read completely")
        for browser, name in sorted(expected_files):
            descriptor = -1
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=self._directory_fd(browser),
                )
                observed = os.fstat(descriptor)
            except OSError:
                fail("failure bundle file changed before sealing")
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            if self._file_identity(observed) != self.file_identities[(browser, name)]:
                fail("failure bundle file changed before sealing")


def _validate_result(value: Any, action: str) -> dict[str, Any]:
    result = exact_keys(
        value,
        {"visible", "enabled", "text_utf8_bytes", "text_sha256"},
        "browser action result",
    )
    if result["visible"] is not True:
        fail("browser action is not visible")
    expected_enabled = True if action in {"submit_chat", "wait_ready"} else None
    if result["enabled"] is not expected_enabled:
        fail("browser action enabled state differs")
    carries_text = action in {"wait_visible", "wait_failed"}
    if action == "wait_ready" and result["text_utf8_bytes"] is not None:
        carries_text = True
    if carries_text:
        integer(result["text_utf8_bytes"], "browser action text bytes", minimum=1)
        sha256_value(result["text_sha256"], "browser action text digest")
    elif result["text_utf8_bytes"] is not None or result["text_sha256"] is not None:
        fail("browser action has unexpected text evidence")
    return result


def validate_actions(value: Any, expected_count: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != expected_count:
        fail("browser action count differs")
    validated: list[dict[str, Any]] = []
    prior_completed = -1
    for index, item in enumerate(value):
        action = exact_keys(
            item,
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
        name = EXPECTED_ACTIONS[index]
        if (
            action["browser_case"] != BROWSER_CASE
            or integer(action["action_index"], "browser action index") != index
            or action["action"] != name
            or action["selector"] != EXPECTED_SELECTORS[index]
        ):
            fail("browser action identity differs")
        if index in {0, 1, 2, 6}:
            sha256_value(action["input_sha256"], "browser action input digest")
        elif action["input_sha256"] is not None:
            fail("browser action has unexpected input digest")
        started = decimal_timestamp(action["started_monotonic_ns"], "action start")
        completed = decimal_timestamp(
            action["completed_monotonic_ns"], "action completion"
        )
        if started < prior_completed or completed < started:
            fail("browser action timestamps overlap or regress")
        prior_completed = completed
        _validate_result(action["result"], name)
        if index == 4:
            if action["screenshot_file"] != "browser/post-header-failure.png":
                fail("failure screenshot path differs")
            sha256_value(action["screenshot_sha256"], "failure screenshot digest")
        elif (
            action["screenshot_file"] is not None
            or action["screenshot_sha256"] is not None
        ):
            fail("browser action has unexpected screenshot evidence")
        validated.append(action)
    return validated


def _validate_target(value: Any, label: str) -> dict[str, Any]:
    target = exact_keys(
        value,
        {
            "chat_id_utf8_bytes",
            "chat_id_sha256",
            "message_id_utf8_bytes",
            "message_id_sha256",
        },
        label,
    )
    integer(target["chat_id_utf8_bytes"], f"{label} chat bytes", minimum=1)
    integer(target["message_id_utf8_bytes"], f"{label} message bytes", minimum=1)
    sha256_value(target["chat_id_sha256"], f"{label} chat digest")
    sha256_value(target["message_id_sha256"], f"{label} message digest")
    return target


def _validate_socket_events(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value or len(value) > 512:
        fail("browser socket event count differs")
    prior = -1
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        event = exact_keys(
            item,
            {
                "sequence",
                "observed_monotonic_ns",
                "correlation_target",
                "type",
                "done",
                "has_error",
                "content_utf8_bytes",
                "content_sha256",
            },
            "browser socket event",
        )
        if integer(event["sequence"], "socket sequence") != index:
            fail("browser socket sequence differs")
        observed = decimal_timestamp(event["observed_monotonic_ns"], "socket time")
        if observed < prior:
            fail("browser socket timestamps regress")
        prior = observed
        if event["correlation_target"] not in {"failure_target", "recovery_target"}:
            fail("browser socket correlation target differs")
        nonempty_string(event["type"], "browser socket event type")
        boolean(event["done"], "browser socket done")
        boolean(event["has_error"], "browser socket error")
        content_bytes = integer(event["content_utf8_bytes"], "socket content bytes")
        if content_bytes == 0:
            if event["content_sha256"] is not None:
                fail("empty socket content has a digest")
        else:
            sha256_value(event["content_sha256"], "socket content digest")
        validated.append(event)
    return validated


def _validate_redacted_control(value: Any, stage: str) -> None:
    control = exact_keys(
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
        "browser control",
    )
    if (
        control["control_schema"] != "ullm.openwebui.failure_control.v1"
        or control["control_stage"] != stage
    ):
        fail("browser control identity differs")
    for key in ("control_file_utf8_bytes", "content_utf8_bytes"):
        integer(control[key], f"browser control {key}", minimum=1)
    for key in ("control_file_sha256", "nonce_sha256", "content_sha256"):
        sha256_value(control[key], f"browser control {key}")
    requested = decimal_timestamp(
        control["requested_monotonic_ns"], "browser control request"
    )
    observed = decimal_timestamp(
        control["observed_monotonic_ns"], "browser control observation"
    )
    if observed < requested:
        fail("browser control timestamps regress")


def _validate_clear_control(value: Any, stage: str) -> str:
    control = exact_keys(
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
        "clear browser control",
    )
    expected_path = {
        "worker_killed": "/run/control/worker-killed",
        "gateway_recovered": "/run/control/gateway-recovered",
    }[stage]
    nonce = nonempty_string(control["nonce"], "browser control nonce")
    if SHA256_RE.fullmatch(nonce) is None:
        fail("browser control nonce differs")
    expected_raw = f"ullm.openwebui.failure_control.v1:{stage}:{nonce}\n".encode(
        "ascii"
    )
    if (
        control["control_schema"] != "ullm.openwebui.failure_control.v1"
        or control["control_stage"] != stage
        or control["control_file"] != expected_path
        or integer(control["content_utf8_bytes"], "browser control bytes", minimum=1)
        != len(expected_raw)
        or control["content_sha256"] != sha256_bytes(expected_raw)
        or integer(control["timeout_ms"], "browser control timeout", minimum=1)
        > 600_000
    ):
        fail("clear browser control evidence differs")
    return nonce


def _bind_redacted_control(value: Any, stage: str, nonce: str) -> None:
    _validate_redacted_control(value, stage)
    control = value
    expected_path = {
        "worker_killed": "/run/control/worker-killed",
        "gateway_recovered": "/run/control/gateway-recovered",
    }[stage]
    expected_raw = f"ullm.openwebui.failure_control.v1:{stage}:{nonce}\n".encode(
        "ascii"
    )
    if (
        control["control_file_utf8_bytes"] != len(expected_path.encode("utf-8"))
        or control["control_file_sha256"] != sha256_bytes(expected_path.encode("utf-8"))
        or control["nonce_sha256"] != sha256_bytes(nonce.encode("ascii"))
        or control["content_utf8_bytes"] != len(expected_raw)
        or control["content_sha256"] != sha256_bytes(expected_raw)
    ):
        fail("redacted browser control does not bind its clear control")


def validate_final_browser(value: dict[str, Any]) -> list[dict[str, Any]]:
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
        "browser final summary",
    )
    if (
        value["schema_version"] != BROWSER_SCHEMA
        or value["record_type"] != "openwebui_failure_smoke"
        or value["browser_case"] != BROWSER_CASE
        or integer(value["page_error_count"], "browser page errors") != 0
        or value["page_errors"] != []
    ):
        fail("browser final identity differs")
    actions = validate_actions(value["browser_actions"], len(EXPECTED_ACTIONS))
    observed = decimal_timestamp(value["observed_monotonic_ns"], "browser summary time")
    if observed < decimal_timestamp(
        actions[-1]["completed_monotonic_ns"], "last action completion"
    ):
        fail("browser summary precedes its actions")
    events = _validate_socket_events(value["socket_events"])
    correlation = exact_keys(
        value["socket_correlation"],
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
    target = _validate_target(correlation["target"], "failure target")
    error_ns = decimal_timestamp(
        correlation["error_first_observed_monotonic_ns"], "provider error time"
    )
    cancel_ns = decimal_timestamp(
        correlation["cancel_first_observed_monotonic_ns"], "cancellation time"
    )
    if (
        error_ns > cancel_ns
        or integer(correlation["error_event_count"], "provider error count") != 1
        or integer(correlation["cancel_event_count"], "cancellation count") != 1
        or integer(correlation["done_after_fault_count"], "post-fault done count") != 0
        or integer(correlation["content_after_error_count"], "post-error content count")
        != 0
    ):
        fail("browser failure correlation differs")
    failure_errors = [
        decimal_timestamp(event["observed_monotonic_ns"], "failure error event")
        for event in events
        if event["correlation_target"] == "failure_target" and event["has_error"]
    ]
    failure_cancels = [
        decimal_timestamp(event["observed_monotonic_ns"], "failure cancel event")
        for event in events
        if event["correlation_target"] == "failure_target"
        and event["type"] == "chat:tasks:cancel"
    ]
    if failure_errors != [error_ns] or failure_cancels != [cancel_ns]:
        fail("browser failure socket correlation differs")
    recovery = exact_keys(
        correlation["recovery"],
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
    recovery_target = _validate_target(
        {key: recovery[key] for key in target}, "recovery target"
    )
    if (
        recovery_target["chat_id_sha256"] != target["chat_id_sha256"]
        or recovery_target["chat_id_utf8_bytes"] != target["chat_id_utf8_bytes"]
        or recovery_target["message_id_sha256"] == target["message_id_sha256"]
        or integer(recovery["done_event_count"], "recovery done count") != 1
        or integer(recovery["cancel_event_count"], "recovery cancellation count") != 0
        or integer(recovery["error_event_count"], "recovery error count") != 0
    ):
        fail("browser recovery correlation differs")
    recovery_submit = decimal_timestamp(
        recovery["submit_completed_monotonic_ns"], "recovery submit time"
    )
    recovery_done = decimal_timestamp(
        recovery["done_observed_monotonic_ns"], "recovery done time"
    )
    if recovery_done < recovery_submit:
        fail("browser recovery timestamps regress")
    recovery_done_events = [
        decimal_timestamp(event["observed_monotonic_ns"], "recovery done event")
        for event in events
        if event["correlation_target"] == "recovery_target" and event["done"]
    ]
    if recovery_done_events != [recovery_done]:
        fail("browser recovery socket correlation differs")
    controls = exact_keys(
        value["controls"], {"worker_killed", "gateway_recovered"}, "browser controls"
    )
    _validate_redacted_control(controls["worker_killed"], "worker_killed")
    _validate_redacted_control(controls["gateway_recovered"], "gateway_recovered")
    screenshot = exact_keys(
        value["screenshot"],
        {"screenshot_file", "screenshot_bytes", "screenshot_sha256"},
        "browser screenshot",
    )
    if screenshot["screenshot_file"] != "browser/post-header-failure.png":
        fail("browser screenshot path differs")
    integer(screenshot["screenshot_bytes"], "browser screenshot bytes", minimum=1)
    sha256_value(screenshot["screenshot_sha256"], "browser screenshot digest")
    if screenshot["screenshot_sha256"] != actions[4]["screenshot_sha256"]:
        fail("browser screenshot action digest differs")
    return actions


def _validate_interim(
    value: dict[str, Any],
    *,
    record_type: str,
    action_count: int,
    final_actions: list[dict[str, Any]],
    final_events: list[dict[str, Any]],
    final_target: dict[str, Any],
    final_controls: dict[str, Any],
) -> str:
    expected_keys = {
        "schema_version",
        "record_type",
        "browser_case",
        "observed_monotonic_ns",
        "browser_actions",
        "socket_correlation",
        "page_error_count",
        "socket_events",
        "worker_killed_control",
    }
    if record_type == "openwebui_failure_gateway_recovery_wait":
        expected_keys.add("gateway_recovered_control")
    exact_keys(value, expected_keys, "browser interim")
    if (
        value["schema_version"] != BROWSER_SCHEMA
        or value["record_type"] != record_type
        or value["browser_case"] != BROWSER_CASE
        or integer(value["page_error_count"], "interim page errors") != 0
    ):
        fail("browser interim identity differs")
    decimal_timestamp(value["observed_monotonic_ns"], "browser interim time")
    actions = validate_actions(value["browser_actions"], action_count)
    if actions != final_actions[:action_count]:
        fail("browser interim action prefix differs")
    events = _validate_socket_events(value["socket_events"])
    if events != final_events[: len(events)]:
        fail("browser interim socket prefix differs")
    correlation = value["socket_correlation"]
    if record_type == "openwebui_failure_worker_kill_wait":
        correlation = exact_keys(
            correlation,
            {
                "target",
                "submit_completed_monotonic_ns",
                "visible_completed_monotonic_ns",
                "pre_fault_done_count",
                "pre_fault_error_count",
                "pre_fault_cancel_count",
            },
            "pre-fault browser correlation",
        )
        target = _validate_target(correlation["target"], "interim target")
        if (
            target != final_target
            or decimal_timestamp(
                correlation["submit_completed_monotonic_ns"],
                "failure submit completion",
            )
            != decimal_timestamp(actions[2]["completed_monotonic_ns"], "submit action")
            or decimal_timestamp(
                correlation["visible_completed_monotonic_ns"], "visible completion"
            )
            != decimal_timestamp(actions[3]["completed_monotonic_ns"], "visible action")
            or any(
                integer(correlation[key], f"pre-fault {key}") != 0
                for key in (
                    "pre_fault_done_count",
                    "pre_fault_error_count",
                    "pre_fault_cancel_count",
                )
            )
        ):
            fail("pre-fault browser correlation differs")
        return _validate_clear_control(value["worker_killed_control"], "worker_killed")
    correlation = exact_keys(
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
        "failed browser correlation",
    )
    target = _validate_target(correlation["target"], "interim target")
    if (
        target != final_target
        or integer(correlation["error_event_count"], "interim error count") != 1
        or integer(correlation["cancel_event_count"], "interim cancel count") != 1
        or integer(correlation["done_after_fault_count"], "interim done count") != 0
        or integer(correlation["content_after_error_count"], "interim content count")
        != 0
    ):
        fail("failed browser correlation differs")
    decimal_timestamp(correlation["error_first_observed_monotonic_ns"], "interim error")
    decimal_timestamp(
        correlation["cancel_first_observed_monotonic_ns"], "interim cancellation"
    )
    if value["worker_killed_control"] != final_controls["worker_killed"]:
        fail("worker-killed control changed after failure")
    _validate_redacted_control(value["worker_killed_control"], "worker_killed")
    return _validate_clear_control(
        value["gateway_recovered_control"], "gateway_recovered"
    )


def validate_fault(value: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    exact_keys(
        value,
        {
            "schema_version",
            "record_type",
            "injection",
            "target_pid",
            "target_starttime_ticks",
            "target_parent_pid",
            "signal",
            "command",
            "started_monotonic_ns",
            "completed_monotonic_ns",
        },
        "fault artifact",
    )
    if (
        value["schema_version"] != GATE_SCHEMA
        or value["record_type"] != "fault_injection"
        or value["injection"] != "post_header_worker_kill"
        or value["signal"] != "SIGKILL"
        or value["command"] != "signal.pidfd_send_signal"
    ):
        fail("fault artifact identity differs")
    target_pid = integer(value["target_pid"], "fault target PID", minimum=1)
    target_starttime = integer(
        value["target_starttime_ticks"], "fault target starttime", minimum=1
    )
    target_parent = integer(
        value["target_parent_pid"], "fault target parent PID", minimum=1
    )
    started = integer(value["started_monotonic_ns"], "fault start", minimum=1)
    completed = integer(value["completed_monotonic_ns"], "fault completion", minimum=1)
    service = summary["service"]
    if (
        completed < started
        or target_pid != service["initial_worker_pid"]
        or target_starttime != service["initial_worker_starttime_ticks"]
        or target_parent != service["initial_gateway_pid"]
    ):
        fail("fault artifact process identity or timing differs")
    return value


def validate_readiness(value: dict[str, Any], summary: dict[str, Any]) -> None:
    exact_keys(
        value,
        {
            "schema_version",
            "record_type",
            "network_id",
            "subnet",
            "gateway",
            "initial",
            "recovered",
        },
        "readiness artifact",
    )
    if (
        value["schema_version"] != GATE_SCHEMA
        or value["record_type"] != "readiness_evidence"
    ):
        fail("readiness artifact identity differs")
    network_id = nonempty_string(value["network_id"], "readiness network ID")
    nonempty_string(value["subnet"], "readiness subnet")
    nonempty_string(value["gateway"], "readiness gateway")
    prior = -1
    for phase in ("initial", "recovered"):
        sample = exact_keys(
            value[phase],
            {"started_monotonic_ns", "completed_monotonic_ns", "status"},
            f"{phase} readiness",
        )
        started = integer(sample["started_monotonic_ns"], f"{phase} readiness start")
        completed = integer(
            sample["completed_monotonic_ns"], f"{phase} readiness completion"
        )
        status = integer(sample["status"], f"{phase} readiness status")
        if started < prior or completed < started or status != 200:
            fail("readiness artifact timing or status differs")
        prior = completed
    if summary["probe"]["network_id_sha256"] != sha256_bytes(
        network_id.encode("utf-8")
    ):
        fail("readiness network hash differs")


def validate_summary(value: dict[str, Any], bindings: BundleBindings) -> None:
    exact_keys(
        value,
        {
            "schema_version",
            "passed",
            "service",
            "browser",
            "fault",
            "recovery",
            "gateway_journal",
            "probe",
            "gate_source_sha256",
        },
        "failure gate summary",
    )
    if value["schema_version"] != GATE_SCHEMA or value["passed"] is not True:
        fail("failure gate summary did not pass")
    service = exact_keys(
        value["service"],
        {
            "unit_sha256",
            "initial_gateway_pid",
            "recovered_gateway_pid",
            "initial_worker_pid",
            "recovered_worker_pid",
            "initial_worker_starttime_ticks",
            "recovered_worker_starttime_ticks",
            "initial_restart_count",
            "recovered_restart_count",
            "restart_delta",
            "boot_id_sha256",
        },
        "failure service summary",
    )
    for key in ("unit_sha256", "boot_id_sha256"):
        sha256_value(service[key], f"service {key}")
    if service["unit_sha256"] != bindings.service_unit_sha256:
        fail("failure service unit differs from the requested unit")
    for key in (
        "initial_gateway_pid",
        "recovered_gateway_pid",
        "initial_worker_pid",
        "recovered_worker_pid",
        "initial_worker_starttime_ticks",
        "recovered_worker_starttime_ticks",
    ):
        integer(service[key], f"service {key}", minimum=1)
    initial_restarts = integer(service["initial_restart_count"], "initial restarts")
    recovered_restarts = integer(
        service["recovered_restart_count"], "recovered restarts"
    )
    restart_delta = integer(service["restart_delta"], "service restart delta")
    if (
        service["initial_gateway_pid"] == service["recovered_gateway_pid"]
        or service["initial_worker_pid"] == service["recovered_worker_pid"]
        or initial_restarts + 1 != recovered_restarts
        or restart_delta != 1
    ):
        fail("failure service restart identity differs")
    browser = exact_keys(
        value["browser"],
        {
            "image_reference_sha256",
            "image_content_digest",
            "script_sha256",
            "action_count",
            "socket_event_count",
            "screenshot_sha256",
            "stdout_lines",
            "stdout_bytes",
            "stdout_sha256",
            "stderr_bytes",
            "stderr_sha256",
        },
        "failure browser summary",
    )
    for key in (
        "image_reference_sha256",
        "script_sha256",
        "screenshot_sha256",
        "stdout_sha256",
        "stderr_sha256",
    ):
        sha256_value(browser[key], f"browser {key}")
    if (
        browser["image_reference_sha256"] != bindings.browser_image_reference_sha256
        or browser["script_sha256"] != bindings.browser_script_sha256
    ):
        fail("failure browser source or image reference differs")
    if not isinstance(browser["image_content_digest"], str) or not browser[
        "image_content_digest"
    ].startswith("sha256:"):
        fail("browser image content digest differs")
    sha256_value(browser["image_content_digest"][7:], "browser image content digest")
    action_count = integer(browser["action_count"], "browser action count")
    socket_count = integer(
        browser["socket_event_count"], "socket event count", minimum=1
    )
    stdout_lines = integer(browser["stdout_lines"], "browser stdout lines")
    stdout_bytes = integer(browser["stdout_bytes"], "browser stdout bytes", minimum=1)
    stderr_bytes = integer(browser["stderr_bytes"], "browser stderr bytes")
    if (
        action_count != len(EXPECTED_ACTIONS)
        or socket_count > 512
        or stdout_lines != 3
        or stdout_bytes < 1
        or stderr_bytes != 0
        or browser["stderr_sha256"] != EMPTY_SHA256
    ):
        fail("failure browser summary counts differ")
    fault = exact_keys(
        value["fault"],
        {
            "target_request_sha256",
            "target_completion_sha256",
            "worker_fatal_monotonic_ns",
            "signal_to_fatal_ns",
            "fault_artifact_sha256",
            "kill_control_sha256",
        },
        "failure fault summary",
    )
    for key in (
        "target_request_sha256",
        "target_completion_sha256",
        "fault_artifact_sha256",
        "kill_control_sha256",
    ):
        sha256_value(fault[key], f"fault {key}")
    integer(fault["worker_fatal_monotonic_ns"], "worker fatal time", minimum=1)
    integer(fault["signal_to_fatal_ns"], "signal-to-fatal duration")
    recovery = exact_keys(
        value["recovery"],
        {
            "request_sha256",
            "completion_sha256",
            "admitted_monotonic_ns",
            "released_monotonic_ns",
            "outcome",
            "reset_complete",
            "readiness_artifact_sha256",
            "recovery_control_sha256",
        },
        "failure recovery summary",
    )
    for key in (
        "request_sha256",
        "completion_sha256",
        "readiness_artifact_sha256",
        "recovery_control_sha256",
    ):
        sha256_value(recovery[key], f"recovery {key}")
    admitted = integer(recovery["admitted_monotonic_ns"], "recovery admission")
    released = integer(recovery["released_monotonic_ns"], "recovery release")
    if (
        released < admitted
        or recovery["outcome"] != "stop"
        or recovery["reset_complete"] is not True
    ):
        fail("failure recovery result differs")
    journal = exact_keys(
        value["gateway_journal"],
        {
            "lifecycle_count",
            "record_count",
            "cursor_count",
            "raw_sha256",
            "stderr_bytes",
            "stderr_sha256",
        },
        "failure journal summary",
    )
    lifecycle_count = integer(journal["lifecycle_count"], "lifecycle count", minimum=1)
    record_count = integer(journal["record_count"], "journal record count", minimum=1)
    cursor_count = integer(journal["cursor_count"], "journal cursor count", minimum=1)
    sha256_value(journal["raw_sha256"], "journal digest")
    sha256_value(journal["stderr_sha256"], "journal stderr digest")
    journal_stderr_bytes = integer(journal["stderr_bytes"], "journal stderr bytes")
    if (
        lifecycle_count > record_count
        or cursor_count != record_count
        or journal_stderr_bytes != 0
        or journal["stderr_sha256"] != EMPTY_SHA256
    ):
        fail("failure journal summary counts differ")
    probe = exact_keys(
        value["probe"],
        {"image_reference_sha256", "image_content_digest", "network_id_sha256"},
        "failure probe summary",
    )
    sha256_value(probe["image_reference_sha256"], "probe image reference digest")
    sha256_value(probe["network_id_sha256"], "probe network digest")
    if not isinstance(probe["image_content_digest"], str) or not probe[
        "image_content_digest"
    ].startswith("sha256:"):
        fail("probe image content digest differs")
    sha256_value(probe["image_content_digest"][7:], "probe image content digest")
    sha256_value(value["gate_source_sha256"], "failure gate source digest")
    if (
        probe["image_reference_sha256"] != bindings.probe_image_reference_sha256
        or value["gate_source_sha256"] != bindings.gate_source_sha256
    ):
        fail("failure gate source or probe image reference differs")


def hook_record(record_type: str, fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": HOOK_SCHEMA,
        "record": {
            "record_type": record_type,
            "phase": PHASE,
            "case_id": BROWSER_CASE,
            "fields": fields,
        },
    }


def action_hook_fields(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "browser_case": action["browser_case"],
        "action_index": action["action_index"],
        "action": action["action"],
        "selector": action["selector"],
        "input_sha256": action["input_sha256"],
        "started_monotonic_ns": decimal_timestamp(
            action["started_monotonic_ns"], "hook action start"
        ),
        "completed_monotonic_ns": decimal_timestamp(
            action["completed_monotonic_ns"], "hook action completion"
        ),
        "result": {
            "visible": action["result"]["visible"],
            "enabled": action["result"]["enabled"],
            "text_utf8_bytes": action["result"]["text_utf8_bytes"],
            "text_sha256": action["result"]["text_sha256"],
        },
        "screenshot_file": action["screenshot_file"],
        "screenshot_sha256": action["screenshot_sha256"],
    }


def fault_hook_fields(fault: dict[str, Any]) -> dict[str, Any]:
    return {
        "injection": fault["injection"],
        "target_pid": fault["target_pid"],
        "target_starttime_ticks": fault["target_starttime_ticks"],
        "signal": fault["signal"],
        "command": fault["command"],
        "started_monotonic_ns": fault["started_monotonic_ns"],
        "completed_monotonic_ns": fault["completed_monotonic_ns"],
    }


def validate_bundle(path: Path, bindings: BundleBindings) -> list[dict[str, Any]]:
    with BundleSnapshot(path) as bundle:
        summary_raw = bundle.read_file(
            "summary.json", browser=False, maximum=MAX_JSON_BYTES
        )
        summary = strict_json_object(summary_raw, "failure gate summary")
        if summary_raw != compact_json(summary) + b"\n":
            fail("failure gate summary is not canonical JSON")
        validate_summary(summary, bindings)

        fault_raw = bundle.read_file(
            "fault-injection.json", browser=False, maximum=MAX_JSON_BYTES
        )
        fault = strict_json_object(fault_raw, "fault artifact")
        if fault_raw != compact_json(fault) + b"\n":
            fail("fault artifact is not canonical JSON")
        if sha256_bytes(fault_raw) != summary["fault"]["fault_artifact_sha256"]:
            fail("fault artifact hash differs")
        validate_fault(fault, summary)

        readiness_raw = bundle.read_file(
            "readiness-evidence.json", browser=False, maximum=MAX_JSON_BYTES
        )
        readiness = strict_json_object(readiness_raw, "readiness artifact")
        if readiness_raw != compact_json(readiness) + b"\n":
            fail("readiness artifact is not canonical JSON")
        if (
            sha256_bytes(readiness_raw)
            != summary["recovery"]["readiness_artifact_sha256"]
        ):
            fail("readiness artifact hash differs")
        validate_readiness(readiness, summary)

        journal = bundle.stream_file(
            "service-journal.raw.jsonl",
            browser=False,
            maximum=MAX_JOURNAL_BYTES,
        )
        if (
            journal.size == 0
            or journal.line_count != summary["gateway_journal"]["record_count"]
            or journal.sha256 != summary["gateway_journal"]["raw_sha256"]
        ):
            fail("service journal evidence differs")

        screenshot = bundle.stream_file(
            "post-header-failure.png",
            browser=True,
            maximum=MAX_SCREENSHOT_BYTES,
        )
        if not screenshot.prefix.startswith(b"\x89PNG\r\n\x1a\n"):
            fail("failure screenshot is not a PNG")

        browser_summary_raw = bundle.read_file(
            "openwebui-failure-summary.json", browser=True, maximum=MAX_JSON_BYTES
        )
        if not browser_summary_raw.endswith(b"\n") or browser_summary_raw == b"\n":
            fail("browser summary framing differs")
        browser_summary = strict_json_object(
            browser_summary_raw[:-1], "browser final summary"
        )
        actions = validate_final_browser(browser_summary)
        screenshot_value = browser_summary["screenshot"]
        if (
            screenshot.size != screenshot_value["screenshot_bytes"]
            or screenshot.sha256 != screenshot_value["screenshot_sha256"]
            or screenshot.sha256 != summary["browser"]["screenshot_sha256"]
        ):
            fail("failure screenshot evidence differs")

        browser_stdout = bundle.read_file(
            "browser-stdout.jsonl", browser=True, maximum=MAX_BROWSER_STDOUT_BYTES
        )
        lines = browser_stdout.splitlines(keepends=True)
        if len(lines) != 3 or any(not line.endswith(b"\n") for line in lines):
            fail("browser stdout framing differs")
        stdout_records = [
            strict_json_object(line[:-1], "browser stdout record") for line in lines
        ]
        final_events = browser_summary["socket_events"]
        final_target = browser_summary["socket_correlation"]["target"]
        final_controls = browser_summary["controls"]
        worker_nonce = _validate_interim(
            stdout_records[0],
            record_type="openwebui_failure_worker_kill_wait",
            action_count=4,
            final_actions=actions,
            final_events=final_events,
            final_target=final_target,
            final_controls=final_controls,
        )
        recovery_nonce = _validate_interim(
            stdout_records[1],
            record_type="openwebui_failure_gateway_recovery_wait",
            action_count=5,
            final_actions=actions,
            final_events=final_events,
            final_target=final_target,
            final_controls=final_controls,
        )
        _bind_redacted_control(
            final_controls["worker_killed"], "worker_killed", worker_nonce
        )
        _bind_redacted_control(
            final_controls["gateway_recovered"], "gateway_recovered", recovery_nonce
        )
        if lines[2] != browser_summary_raw:
            fail("browser stdout final record differs from its artifact")
        if (
            len(browser_stdout) != summary["browser"]["stdout_bytes"]
            or sha256_bytes(browser_stdout) != summary["browser"]["stdout_sha256"]
            or len(browser_summary["socket_events"])
            != summary["browser"]["socket_event_count"]
        ):
            fail("browser stdout summary evidence differs")

        fault_started = fault["started_monotonic_ns"]
        fault_completed = fault["completed_monotonic_ns"]
        visible_completed = decimal_timestamp(
            actions[3]["completed_monotonic_ns"], "visible action completion"
        )
        failed_completed = decimal_timestamp(
            actions[4]["completed_monotonic_ns"], "failed action completion"
        )
        worker_fatal = summary["fault"]["worker_fatal_monotonic_ns"]
        if (
            fault_started < visible_completed
            or fault_completed < fault_started
            or worker_fatal < fault_started
            or worker_fatal > failed_completed
            or summary["fault"]["signal_to_fatal_ns"] != worker_fatal - fault_started
        ):
            fail("fault and browser timeline differs")
        bundle.seal()

    records = [
        hook_record("browser_action", action_hook_fields(item)) for item in actions[:4]
    ]
    records.append(hook_record("fault_injection", fault_hook_fields(fault)))
    records.extend(
        hook_record("browser_action", action_hook_fields(item)) for item in actions[4:]
    )
    return records


@dataclasses.dataclass(frozen=True)
class BoundedCommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_bounded_command(
    command: list[str], *, cwd: Path, timeout: float
) -> BoundedCommandResult:
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError:
        fail("failed to start failure gate")
    if process.stdout is None or process.stderr is None:
        _terminate_process(process)
        fail("failure gate pipes are unavailable")
    output = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {
        "stdout": MAX_SUBPROCESS_STDOUT_BYTES,
        "stderr": MAX_SUBPROCESS_STDERR_BYTES,
    }
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    deadline = time.monotonic() + timeout
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                fail("failure gate timed out")
            ready = selector.select(min(remaining, 1.0))
            if not ready and process.poll() is not None:
                ready = [
                    (key, selectors.EVENT_READ) for key in selector.get_map().values()
                ]
            for key, _events in ready:
                try:
                    file_object = key.fileobj
                    descriptor = (
                        file_object
                        if isinstance(file_object, int)
                        else file_object.fileno()
                    )
                    chunk = os.read(descriptor, READ_CHUNK_BYTES)
                except OSError:
                    fail("failed to read failure gate output")
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                stream = key.data
                if stream not in output:
                    fail("failure gate output stream differs")
                output[stream].extend(chunk)
                if len(output[stream]) > limits[stream]:
                    fail("failure gate output exceeded its bound")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            fail("failure gate timed out")
        returncode = process.wait(timeout=remaining)
    except BaseException:
        if process.poll() is None:
            _terminate_process(process)
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    return BoundedCommandResult(
        returncode, bytes(output["stdout"]), bytes(output["stderr"])
    )


def build_failure_gate_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        os.fspath(args.failure_gate),
        "--output-dir",
        os.fspath(args.failure_bundle),
        "--token-file",
        os.fspath(args.token_file),
        "--browser-script",
        os.fspath(args.browser_script),
        "--browser-image",
        args.browser_image,
        "--probe-image",
        args.probe_image,
        "--openwebui-url",
        args.openwebui_url,
        "--ready-url",
        args.ready_url,
        "--network",
        args.network,
        "--service",
        args.service,
        "--docker",
        args.docker,
        "--systemctl",
        args.systemctl,
        "--journalctl",
        args.journalctl,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--control-timeout-ms",
        str(args.control_timeout_ms),
        "--recovery-probe-timeout-seconds",
        str(args.recovery_probe_timeout_seconds),
    ]


def execute(args: argparse.Namespace) -> bytes:
    gate_source = snapshot_source(
        args.failure_gate, "failure gate source", maximum=MAX_GATE_SOURCE_BYTES
    )
    browser_source = snapshot_source(
        args.browser_script,
        "failure browser source",
        maximum=MAX_BROWSER_SCRIPT_BYTES,
    )
    bindings = BundleBindings(
        gate_source_sha256=gate_source.sha256,
        browser_script_sha256=browser_source.sha256,
        browser_image_reference_sha256=sha256_bytes(args.browser_image.encode("utf-8")),
        probe_image_reference_sha256=sha256_bytes(args.probe_image.encode("utf-8")),
        service_unit_sha256=sha256_bytes(args.service.encode("utf-8")),
    )
    try:
        args.failure_bundle.lstat()
    except FileNotFoundError:
        pass
    except OSError:
        fail("failure bundle destination is unavailable")
    else:
        fail("failure bundle destination already exists")
    command = build_failure_gate_command(args)
    result = run_bounded_command(
        command,
        cwd=args.failure_gate.resolve(strict=True).parents[1],
        timeout=float(args.timeout_seconds + 30),
    )
    if (
        result.returncode != 0
        or result.stdout != SUCCESS_STDOUT
        or result.stderr != b""
    ):
        fail("failure gate subprocess result differs")
    require_unchanged_source(
        gate_source, "failure gate source", maximum=MAX_GATE_SOURCE_BYTES
    )
    require_unchanged_source(
        browser_source, "failure browser source", maximum=MAX_BROWSER_SCRIPT_BYTES
    )
    records = validate_bundle(args.failure_bundle, bindings)
    require_unchanged_source(
        gate_source, "failure gate source", maximum=MAX_GATE_SOURCE_BYTES
    )
    require_unchanged_source(
        browser_source, "failure browser source", maximum=MAX_BROWSER_SCRIPT_BYTES
    )
    if len(records) != 10:
        fail("failure hook record count differs")
    raw = b"".join(compact_json(record) + b"\n" for record in records)
    if len(raw) > 256 * 1024:
        fail("failure hook output exceeds its bound")
    return raw


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the formal failure gate and adapt its atomic bundle to hook JSONL."
    )
    parser.add_argument("--failure-gate", type=Path, required=True)
    parser.add_argument("--failure-bundle", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--browser-script", type=Path, required=True)
    parser.add_argument("--browser-image", required=True)
    parser.add_argument("--probe-image", required=True)
    parser.add_argument("--openwebui-url", required=True)
    parser.add_argument("--ready-url", required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--docker", required=True)
    parser.add_argument("--systemctl", required=True)
    parser.add_argument("--journalctl", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--control-timeout-ms", type=int, required=True)
    parser.add_argument("--recovery-probe-timeout-seconds", type=int, required=True)
    args = parser.parse_args(argv)
    for name in ("failure_gate", "failure_bundle", "token_file", "browser_script"):
        if not getattr(args, name).is_absolute():
            parser.error(f"--{name.replace('_', '-')} must be an absolute path")
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
        raw = execute(parse_args(argv))
    except FailureHookError:
        print("OpenWebUI failure hook failed", file=sys.stderr)
        return 1
    except Exception:
        print("OpenWebUI failure hook failed", file=sys.stderr)
        return 1
    try:
        sys.stdout.buffer.write(raw)
        sys.stdout.buffer.flush()
    except OSError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
