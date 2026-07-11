#!/usr/bin/env python3
"""Run the exact 20-chat soak, optionally after one fixed OpenWebUI smoke."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import stat
import subprocess
import sys
import threading
import time
import types
import urllib.parse
from pathlib import Path
from typing import Any, Callable, NoReturn, Protocol, cast


SUPPORT_TOOL_PATH = Path(__file__).with_name("run-openwebui-stop-gate.py")
MAX_SUPPORT_TOOL_BYTES = 4 * 1024 * 1024
MAX_GATE_SOURCE_BYTES = 2 * 1024 * 1024
SNAPSHOT_CHUNK_BYTES = 64 * 1024


def stable_regular_snapshot(path: Path, label: str, maximum: int) -> bytes:
    descriptor = -1
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 1
            or before.st_size > maximum
        ):
            raise RuntimeError(f"{label} is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(SNAPSHOT_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_uid,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_uid,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or len(raw) != before.st_size:
            raise RuntimeError(f"{label} changed while it was read")
        return raw
    except RuntimeError:
        raise
    except OSError as error:
        raise RuntimeError(f"failed to read {label} without following links") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


GATE_SOURCE_RAW = stable_regular_snapshot(
    Path(__file__).resolve(), "OpenWebUI soak gate source", MAX_GATE_SOURCE_BYTES
)


def _load_support() -> tuple[Any, bytes]:
    raw = stable_regular_snapshot(
        SUPPORT_TOOL_PATH, "OpenWebUI soak gate support", MAX_SUPPORT_TOOL_BYTES
    )
    name = "_ullm_openwebui_stop_gate_support"
    module = types.ModuleType(name)
    module.__file__ = os.fspath(SUPPORT_TOOL_PATH)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        code = compile(raw, os.fspath(SUPPORT_TOOL_PATH), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, raw


SUPPORT, SUPPORT_SOURCE_RAW = _load_support()

GATE_SCHEMA = "ullm.openwebui.browser_soak_gate.v1"
BROWSER_SCHEMA = "ullm.openwebui.browser_soak.v1"
COMBINED_GATE_SCHEMA = "ullm.openwebui.browser_smoke_soak_gate.v1"
COMBINED_BROWSER_SCHEMA = "ullm.openwebui.browser_smoke_soak.v1"
LIFECYCLE_SCHEMA = SUPPORT.LIFECYCLE_SCHEMA
RUN_CASE = "openwebui_20_chat_soak"
COMBINED_RUN_CASE = "openwebui_smoke_and_20_chat_soak"
COMBINED_SUMMARY_RECORD_TYPE = "openwebui_smoke_soak_summary"
COMBINED_MODE = "smoke_then_soak20"
CASE_PREFIX = "openwebui_soak_chat_"
SMOKE_CASE = "openwebui_smoke"
SMOKE_MARKER = "OPENWEBUI_SMOKE_OK"
CHAT_COUNT = 20
MODEL_ID = "ullm-qwen3-14b-sq8"
MODEL_LABEL = "uLLM Qwen3 14B SQ8"
OBSERVER_SOCKET = SUPPORT.OBSERVER_SOCKET
BROWSER_SCRIPT_CONTAINER_PATH = "/usr/src/app/ullm-browser-soak.cjs"
BROWSER_SUMMARY_NAME = "openwebui-soak-summary.json"
BROWSER_CONTAINER_OUTPUT_DIR_NAME = "browser-output"
MAX_BROWSER_LINES = CHAT_COUNT + 1
MAX_COMBINED_BROWSER_LINES = CHAT_COUNT + 2
MAX_BROWSER_STDERR_BYTES = 4 * 1024 * 1024
MAX_BROWSER_RAW_BYTES = 32 * 1024 * 1024
MAX_BROWSER_SCRIPT_BYTES = 2 * 1024 * 1024
MAX_BROWSER_SOCKET_EVENTS = 128
DEFAULT_TIMEOUT_SECONDS = 1800
PROCESS_GRACE_SECONDS = SUPPORT.PROCESS_GRACE_SECONDS
COPY_CHUNK_BYTES = SUPPORT.COPY_CHUNK_BYTES
SHA256_RE = SUPPORT.SHA256_RE
CONTENT_IMAGE_RE = SUPPORT.CONTENT_IMAGE_RE
FINAL_ACTIONS = (
    "navigate",
    "select_model",
    "submit_chat",
    "wait_visible",
    "wait_ready",
)
ACTION_SELECTORS = (None, "body", "#chat-input", ".chat-assistant", "#chat-input")

SoakGateError = SUPPORT.StopGateError
SecretGuard = SUPPORT.SecretGuard
AtomicLineWriter = SUPPORT.AtomicLineWriter
AtomicRunDirectory = SUPPORT.AtomicRunDirectory
ServiceIdentity = SUPPORT.ServiceIdentity
ObserverRecord = SUPPORT.ObserverRecord
JournalFollower = SUPPORT.JournalFollower

compact_json = SUPPORT.compact_json
strict_json_object = SUPPORT.strict_json_object
exact_keys = SUPPORT.exact_keys
integer = SUPPORT.integer
decimal_timestamp = SUPPORT.decimal_timestamp
nonempty_string = SUPPORT.nonempty_string
sha256_bytes = SUPPORT.sha256_bytes
read_bounded_file = SUPPORT.read_bounded_file
write_private_snapshot = SUPPORT.write_private_snapshot
write_atomic_json = SUPPORT.write_atomic_json
query_service_identity = SUPPORT.query_service_identity
read_boot_id = SUPPORT.read_boot_id
initial_journal_cursor = SUPPORT.initial_journal_cursor
spawn_journal_follower = SUPPORT.spawn_journal_follower
run_bounded_command = SUPPORT.run_bounded_command
terminate_process_group = SUPPORT.terminate_process_group
normalized_url = SUPPORT.normalized_url
normalized_browser_image = SUPPORT.normalized_browser_image
validate_lifecycle_payload = SUPPORT.validate_lifecycle_payload
_support_validate_journal_record = SUPPORT.validate_journal_record
require_correlated_prefix = SUPPORT.require_correlated_prefix


class LineWriterProtocol(Protocol):
    bytes_written: int
    lines_written: int
    sha256: str

    def write_line(self, raw: bytes) -> None: ...

    def commit(self) -> None: ...

    def abort(self) -> None: ...


class SecretGuardProtocol(Protocol):
    def reject(self, raw: bytes, label: str) -> None: ...

    def scan_file(self, path: Path, label: str) -> None: ...

    def extend(self, values: list[str]) -> SecretGuardProtocol: ...


class ObserverRecordProtocol(Protocol):
    raw: bytes


class JournalFollowerProtocol(Protocol):
    records: list[bytes]
    lifecycle: list[bytes]
    cursors: set[str]
    stderr_bytes: int
    stderr_digest: Any

    def start(self) -> None: ...

    def wait_correlated(self, observer: Any, deadline_ns: int) -> None: ...

    def wait_correlated_records(
        self,
        snapshot: Callable[[], list[Any]],
        deadline_ns: int,
    ) -> None: ...

    def stop(self) -> None: ...


def fail(message: str) -> NoReturn:
    raise SoakGateError(message)


def validate_journal_record(
    payload: bytes,
    *,
    service: str,
    main_pid: int,
    boot_id: str,
    cursors: set[str],
    lifecycle_payloads: set[bytes],
) -> tuple[str, bytes | None]:
    record = strict_json_object(payload, "browser soak journal record")
    priority = record.get("PRIORITY")
    if (
        not isinstance(priority, str)
        or not priority.isascii()
        or not priority.isdecimal()
        or not 0 <= int(priority, 10) <= 7
    ):
        fail("browser soak journal PRIORITY is missing or invalid")
    return cast(
        tuple[str, bytes | None],
        _support_validate_journal_record(
            payload,
            service=service,
            main_pid=main_pid,
            boot_id=boot_id,
            cursors=cursors,
            lifecycle_payloads=lifecycle_payloads,
        ),
    )


# JournalFollower resolves this name in the dynamically loaded support module.
SUPPORT.validate_journal_record = validate_journal_record


def case_indices(*, include_smoke: bool = False) -> tuple[int, ...]:
    if not isinstance(include_smoke, bool):
        fail("browser soak mode flag is not boolean")
    prefix = (0,) if include_smoke else ()
    return prefix + tuple(range(1, CHAT_COUNT + 1))


def browser_case(case_index: int, *, include_smoke: bool = False) -> str:
    if include_smoke and case_index == 0:
        return SMOKE_CASE
    if not 1 <= case_index <= CHAT_COUNT:
        fail("browser soak case index is outside the frozen schedule")
    return f"{CASE_PREFIX}{case_index:02d}"


def case_marker(case_index: int, *, include_smoke: bool = False) -> str:
    browser_case(case_index, include_smoke=include_smoke)
    if include_smoke and case_index == 0:
        return SMOKE_MARKER
    return f"OPENWEBUI_SOAK_OK_{case_index:02d}"


def case_prompt(case_index: int, *, include_smoke: bool = False) -> str:
    marker = case_marker(case_index, include_smoke=include_smoke)
    return f"Reply with exactly {marker} and nothing else."


def case_record_type(case_index: int, *, include_smoke: bool = False) -> str:
    browser_case(case_index, include_smoke=include_smoke)
    if include_smoke and case_index == 0:
        return "openwebui_smoke_chat"
    return "openwebui_soak_chat"


def schedule_evidence(*, include_smoke: bool = False) -> list[dict[str, Any]]:
    return [
        {
            "position": position,
            "case_index": case_index,
            "case_kind": "smoke" if case_index == 0 else "soak",
            "browser_case": browser_case(case_index, include_smoke=include_smoke),
        }
        for position, case_index in enumerate(case_indices(include_smoke=include_smoke))
    ]


def navigation_url(base_url: str) -> str:
    query = urllib.parse.urlencode((("temporary-chat", "true"), ("models", MODEL_ID)))
    return f"{base_url}/?{query}"


@dataclasses.dataclass
class SoakTrace:
    request_id: str
    completion_id: str
    prompt_tokens: int
    max_completion_tokens: int
    events: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    started: bool = False
    first_token: bool = False
    released: bool = False
    last_progress: int = 0
    admit_to_start_ns: int | None = None

    def event(self, name: str) -> dict[str, Any]:
        matches = [item for item in self.events if item["event"] == name]
        if len(matches) != 1:
            fail("gateway soak trace singular event count differs")
        return matches[0]


class SoakLifecycleMachine:
    def __init__(self, *, expected_count: int = CHAT_COUNT) -> None:
        if expected_count not in {CHAT_COUNT, CHAT_COUNT + 1}:
            fail("gateway soak expected request count is unsupported")
        self.expected_count = expected_count
        self.traces: list[SoakTrace] = []
        self.active: SoakTrace | None = None
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
                fail("gateway admitted an overlapping or duplicate soak request")
            if len(self.traces) >= self.expected_count:
                fail("gateway admitted an extra soak request")
            trace = SoakTrace(
                *pair, event["prompt_tokens"], event["max_completion_tokens"]
            )
            trace.events.append(event)
            self.traces.append(trace)
            self.active = trace
            self.seen_pairs.add(pair)
            self.max_active = max(self.max_active, 1)
            return

        active = self.active
        if active is None or pair != (active.request_id, active.completion_id):
            fail("gateway lifecycle correlation differs from the active soak request")
        if name == "request_started":
            if active.started:
                fail("gateway soak request_started is duplicated")
            if event["prompt_tokens"] != active.prompt_tokens:
                fail("gateway soak prompt-token identity differs")
            active.started = True
            active.admit_to_start_ns = event["admit_to_start_ns"]
        elif name == "request_progress":
            processed = event["processed_prompt_tokens"]
            if (
                not active.started
                or active.first_token
                or processed <= active.last_progress
            ):
                fail("gateway soak progress ordering differs")
            if event["prompt_tokens"] != active.prompt_tokens:
                fail("gateway soak prompt-token identity differs")
            active.last_progress = processed
        elif name == "request_first_token":
            if not active.started or active.first_token:
                fail("gateway soak first-token ordering differs")
            active.first_token = True
        elif name == "request_cancel_requested":
            fail("gateway soak request was cancelled")
        elif name == "request_released":
            if not active.started or not active.first_token or active.released:
                fail("gateway soak release ordering or count differs")
            if (
                event["outcome"] != "stop"
                or event["cancel_reason"] is not None
                or event["reset_complete"] is not True
                or event["completion_tokens"] < 1
                or event["completion_tokens"] > active.max_completion_tokens
            ):
                fail("gateway soak release is not a reset-complete stop")
            if event["prompt_tokens"] != active.prompt_tokens:
                fail("gateway soak prompt-token identity differs")
            if (
                event["admit_to_start_ns"] != active.admit_to_start_ns
                or event["admit_to_release_ns"]
                != event["admit_to_start_ns"] + event["start_to_release_ns"]
            ):
                fail("gateway soak release duration identity differs")
            active.released = True
            self.active = None
        else:
            fail("gateway lifecycle event is not valid in a soak trace")
        active.events.append(event)


class LifecycleObserver(SUPPORT.LifecycleObserver):  # type: ignore[name-defined,misc]
    def __init__(
        self,
        path: Path,
        expected_pid: int,
        expected_uid: int,
        writer: LineWriterProtocol,
        *,
        expected_count: int = CHAT_COUNT,
    ) -> None:
        super().__init__(path, expected_pid, expected_uid, writer)
        self.machine = SoakLifecycleMachine(expected_count=expected_count)


class BrowserProcess:
    def __init__(
        self,
        process: subprocess.Popen[bytes],
        writer: LineWriterProtocol,
        *,
        maximum_lines: int = MAX_BROWSER_LINES,
    ):
        if maximum_lines not in {MAX_BROWSER_LINES, MAX_COMBINED_BROWSER_LINES}:
            fail("browser stdout line limit is unsupported")
        self.process = process
        self.writer = writer
        self.maximum_lines = maximum_lines
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
                raw = self.process.stdout.readline(SUPPORT.MAX_JSON_LINE_BYTES + 2)
                if raw == b"":
                    return
                if (
                    not raw.endswith(b"\n")
                    or len(raw) > SUPPORT.MAX_JSON_LINE_BYTES + 1
                ):
                    fail("browser stdout line framing or size is invalid")
                payload = raw[:-1]
                value = strict_json_object(payload, "browser stdout")
                with self.condition:
                    if len(self.lines) >= self.maximum_lines:
                        fail("browser stdout line count exceeds the soak schedule")
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

    def wait_exit(self, deadline_ns: int) -> int:
        remaining = deadline_ns - time.monotonic_ns()
        if remaining <= 0:
            fail("browser soak process deadline expired")
        try:
            code = self.process.wait(timeout=remaining / 1_000_000_000)
        except subprocess.TimeoutExpired:
            fail("browser soak process timed out")
        self.stdout_thread.join(timeout=2.0)
        self.stderr_thread.join(timeout=2.0)
        if self.stdout_thread.is_alive() or self.stderr_thread.is_alive():
            fail("browser soak pipe drains did not terminate")
        if self.error is not None:
            raise self.error
        return code


def _safe_mount_path(path: Path, label: str) -> str:
    value = os.fspath(path.resolve(strict=True))
    if any(character in value for character in ",\0\n\r"):
        fail(f"{label} path cannot be represented as a Docker mount")
    return value


def validate_container_output_directory(root: Path) -> Path:
    try:
        metadata = root.lstat()
        entries = list(root.iterdir())
    except OSError:
        fail("browser soak container output is unavailable")
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or len(entries) != 1
        or entries[0].name != BROWSER_SUMMARY_NAME
    ):
        fail("browser soak container output layout or identity differs")
    return entries[0]


def snapshot_validated_browser_summary(
    source: Path, destination: Path, expected_stdout: bytes
) -> bytes:
    raw = stable_regular_snapshot(
        source,
        "browser soak summary artifact",
        SUPPORT.MAX_JSON_LINE_BYTES + 1,
    )
    if raw != expected_stdout + b"\n":
        fail("browser soak summary changed after stdout validation")
    write_private_snapshot(
        destination,
        raw,
        "browser soak summary artifact",
    )
    return raw


def fsync_bundle_tree(root: Path) -> None:
    directories: list[Path] = []
    try:
        for current, names, files in os.walk(root, topdown=True, followlinks=False):
            current_path = Path(current)
            metadata = current_path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                fail("browser soak bundle contains an unsafe directory")
            directories.append(current_path)
            names.sort()
            files.sort()
            for name in names:
                child = current_path / name
                child_metadata = child.lstat()
                if stat.S_ISLNK(child_metadata.st_mode) or not stat.S_ISDIR(
                    child_metadata.st_mode
                ):
                    fail("browser soak bundle contains an unsafe child directory")
            for name in files:
                file = current_path / name
                file_metadata = file.lstat()
                if stat.S_ISLNK(file_metadata.st_mode) or not stat.S_ISREG(
                    file_metadata.st_mode
                ):
                    fail("browser soak bundle contains an unsafe artifact")
                flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(file, flags)
                try:
                    opened = os.fstat(descriptor)
                    if (opened.st_dev, opened.st_ino) != (
                        file_metadata.st_dev,
                        file_metadata.st_ino,
                    ):
                        fail("browser soak artifact identity changed during fsync")
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        for directory in reversed(directories):
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(directory, flags)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except SoakGateError:
        raise
    except OSError:
        fail("failed to fsync the browser soak bundle")


def build_browser_command(
    *,
    docker: str,
    image: str,
    name: str,
    script: Path,
    token_file: Path,
    browser_output: Path,
    openwebui_url: str,
    uid: int,
    gid: int,
    include_smoke: bool = False,
) -> list[str]:
    image, _content_digest = normalized_browser_image(image)
    mounts = (
        f"type=bind,src={_safe_mount_path(script, 'browser script')},dst={BROWSER_SCRIPT_CONTAINER_PATH},readonly",
        f"type=bind,src={_safe_mount_path(token_file, 'token file')},dst=/run/secrets/openwebui-token,readonly",
        f"type=bind,src={_safe_mount_path(browser_output, 'browser output')},dst=/output",
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
            "OPENWEBUI_SOAK_SUMMARY=/output/openwebui-soak-summary.json",
        )
    )
    if include_smoke:
        command.extend(("--env", f"OPENWEBUI_SOAK_MODE={COMBINED_MODE}"))
    command.extend((image, "node", BROWSER_SCRIPT_CONTAINER_PATH))
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
        fail("failed to start the transient browser soak container")


def _validate_identity_hashes(value: dict[str, Any], prefix: str) -> None:
    if (
        integer(value[f"{prefix}_utf8_bytes"], f"{prefix} bytes", minimum=1) < 1
        or SHA256_RE.fullmatch(
            nonempty_string(value[f"{prefix}_sha256"], f"{prefix} digest")
        )
        is None
    ):
        fail("browser soak correlation identity digest differs")


def validate_browser_action_sequence(
    actions: Any,
    *,
    case_index: int,
    base_url: str,
    include_smoke: bool = False,
) -> tuple[int, int]:
    if not isinstance(actions, list) or len(actions) != len(FINAL_ACTIONS):
        fail("browser soak action count differs")
    case_id = browser_case(case_index, include_smoke=include_smoke)
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
    expected_inputs = (
        sha256_bytes(navigation_url(base_url).encode("utf-8")),
        sha256_bytes(MODEL_ID.encode("utf-8")),
        sha256_bytes(
            case_prompt(case_index, include_smoke=include_smoke).encode("utf-8")
        ),
        None,
        None,
    )
    prior_completed = -1
    first_started = -1
    for index, (action, expected_name, expected_selector, expected_input) in enumerate(
        zip(
            actions,
            FINAL_ACTIONS,
            ACTION_SELECTORS,
            expected_inputs,
            strict=True,
        )
    ):
        if not isinstance(action, dict):
            fail("browser soak action is not an object")
        exact_keys(action, fields, "browser soak action")
        action_index = integer(
            action["action_index"], "browser soak action index", minimum=0
        )
        if (
            action["browser_case"] != case_id
            or action_index != index
            or action["action"] != expected_name
            or action["selector"] != expected_selector
            or action["input_sha256"] != expected_input
            or action["screenshot_file"] is not None
            or action["screenshot_sha256"] is not None
        ):
            fail("browser soak action identity or ordering differs")
        started = decimal_timestamp(
            action["started_monotonic_ns"], "browser soak action start"
        )
        completed = decimal_timestamp(
            action["completed_monotonic_ns"], "browser soak action completion"
        )
        if completed < started or started < prior_completed:
            fail("browser soak action timestamps overlap or regress")
        if index == 0:
            first_started = started
        prior_completed = completed
        result = action["result"]
        if not isinstance(result, dict):
            fail("browser soak action result is not an object")
        exact_keys(
            result,
            {"visible", "enabled", "text_utf8_bytes", "text_sha256"},
            "browser soak action result",
        )
        expected_enabled = (
            True if expected_name in {"submit_chat", "wait_ready"} else None
        )
        if result["visible"] is not True or result["enabled"] is not expected_enabled:
            fail("browser soak action visibility or enabled state differs")
        if expected_name in {"wait_visible", "wait_ready"}:
            integer(result["text_utf8_bytes"], "browser soak text bytes", minimum=1)
            if (
                SHA256_RE.fullmatch(
                    nonempty_string(result["text_sha256"], "browser soak text digest")
                )
                is None
            ):
                fail("browser soak action text digest differs")
        elif result["text_utf8_bytes"] is not None or result["text_sha256"] is not None:
            fail("browser soak action unexpectedly carries text evidence")
    return first_started, prior_completed


def validate_socket_events(events: Any) -> dict[str, int]:
    if (
        not isinstance(events, list)
        or not events
        or len(events) > MAX_BROWSER_SOCKET_EVENTS
    ):
        fail("browser soak socket event evidence is empty or outside its bound")
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
    allowed = {
        "chat:active",
        "chat:completion",
        "chat:outlet",
        "chat:tasks:cancel",
    }
    content_timestamps: list[int] = []
    done_timestamps: list[int] = []
    completion_indices: list[int] = []
    done_indices: list[int] = []
    cancel_count = 0
    provider_error_count = 0
    prior_timestamp = -1
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            fail("browser soak socket event is not an object")
        exact_keys(event, fields, "browser soak socket event")
        timestamp = decimal_timestamp(
            event["observed_monotonic_ns"], "browser soak socket event timestamp"
        )
        if (
            integer(event["sequence"], "browser soak socket sequence") != index
            or event["correlation_target"] != "chat_target"
            or timestamp < prior_timestamp
            or event["type"] not in allowed
            or not isinstance(event["done"], bool)
            or not isinstance(event["has_error"], bool)
        ):
            fail("browser soak socket event identity or ordering differs")
        prior_timestamp = timestamp
        content_bytes = integer(
            event["content_utf8_bytes"], "browser soak socket content bytes"
        )
        if content_bytes == 0:
            if event["content_sha256"] is not None:
                fail("empty browser soak socket content carries a digest")
        elif (
            SHA256_RE.fullmatch(
                nonempty_string(
                    event["content_sha256"], "browser soak socket content digest"
                )
            )
            is None
        ):
            fail("browser soak socket content digest differs")
        if event["type"] in {"chat:active", "chat:outlet"} and (
            event["done"] or content_bytes != 0
        ):
            fail("browser soak state event carries terminal or content state")
        if event["done"] and event["type"] != "chat:completion":
            fail("browser soak non-completion event is terminal")
        if event["type"] == "chat:completion" and content_bytes > 0:
            content_timestamps.append(timestamp)
        if event["type"] == "chat:completion":
            completion_indices.append(index)
            if event["done"]:
                done_timestamps.append(timestamp)
                done_indices.append(index)
        cancel_count += int(event["type"] == "chat:tasks:cancel")
        provider_error_count += int(event["has_error"])
    if (
        not content_timestamps
        or len(done_timestamps) != 1
        or cancel_count != 0
        or provider_error_count != 0
        or done_timestamps[0] < content_timestamps[0]
        or completion_indices[-1] != done_indices[0]
    ):
        fail(
            "browser soak content, done, cancellation, or provider-error count differs"
        )
    return {
        "first_content_ns": content_timestamps[0],
        "done_ns": done_timestamps[0],
        "done_count": len(done_timestamps),
        "cancel_count": cancel_count,
        "provider_error_count": provider_error_count,
    }


def validate_browser_case(
    value: dict[str, Any],
    raw: bytes,
    guard: SecretGuardProtocol,
    *,
    case_index: int,
    base_url: str,
    include_smoke: bool = False,
) -> dict[str, Any]:
    exact_keys(
        value,
        {
            "schema_version",
            "record_type",
            "browser_case",
            "case_index",
            "observed_monotonic_ns",
            "browser_actions",
            "socket_correlation",
            "socket_events",
            "visible_marker",
            "page_error_count",
            "page_errors",
            "page_state",
        },
        "browser soak case",
    )
    if (
        value["schema_version"]
        != (COMBINED_BROWSER_SCHEMA if include_smoke else BROWSER_SCHEMA)
        or value["record_type"]
        != case_record_type(case_index, include_smoke=include_smoke)
        or value["browser_case"]
        != browser_case(case_index, include_smoke=include_smoke)
        or value["page_errors"] != []
    ):
        fail("browser soak case identity or page-error state differs")
    if (
        integer(
            value["case_index"],
            "browser soak case index",
            minimum=0 if include_smoke else 1,
        )
        != case_index
        or integer(value["page_error_count"], "browser soak page-error count") != 0
    ):
        fail("browser soak case index or page-error count differs")
    first_action, last_action = validate_browser_action_sequence(
        value["browser_actions"],
        case_index=case_index,
        base_url=base_url,
        include_smoke=include_smoke,
    )
    observed = decimal_timestamp(
        value["observed_monotonic_ns"], "browser soak case timestamp"
    )
    if observed < last_action:
        fail("browser soak case precedes its action evidence")

    marker = value["visible_marker"]
    if not isinstance(marker, dict):
        fail("browser soak marker evidence is not an object")
    exact_keys(
        marker,
        {"expected_marker_utf8_bytes", "expected_marker_sha256", "observed"},
        "browser soak marker evidence",
    )
    expected_marker = case_marker(case_index, include_smoke=include_smoke).encode(
        "utf-8"
    )
    if (
        integer(
            marker["expected_marker_utf8_bytes"],
            "browser soak expected marker bytes",
            minimum=1,
        )
        != len(expected_marker)
        or marker["expected_marker_sha256"] != sha256_bytes(expected_marker)
        or marker["observed"] is not True
    ):
        fail("browser soak expected marker evidence differs")

    page_state = value["page_state"]
    if not isinstance(page_state, dict):
        fail("browser soak page state is not an object")
    exact_keys(
        page_state,
        {
            "page_index",
            "temporary_chat",
            "created",
            "closed",
            "open_pages_after_close",
        },
        "browser soak page state",
    )
    if (
        integer(
            page_state["page_index"],
            "browser soak page index",
            minimum=0 if include_smoke else 1,
        )
        != case_index
        or page_state["temporary_chat"] is not True
        or page_state["created"] is not True
        or page_state["closed"] is not True
        or integer(
            page_state["open_pages_after_close"],
            "browser soak open pages after close",
        )
        != 0
    ):
        fail("browser soak temporary page separation differs")

    socket = validate_socket_events(value["socket_events"])
    correlation = value["socket_correlation"]
    if not isinstance(correlation, dict) or not isinstance(
        correlation.get("target"), dict
    ):
        fail("browser soak socket correlation is malformed")
    exact_keys(
        correlation,
        {
            "target",
            "submit_started_monotonic_ns",
            "submit_completed_monotonic_ns",
            "first_content_observed_monotonic_ns",
            "done_observed_monotonic_ns",
            "done_event_count",
            "cancellation_event_count",
            "provider_error_count",
        },
        "browser soak socket correlation",
    )
    target = correlation["target"]
    exact_keys(
        target,
        {
            "chat_id_utf8_bytes",
            "chat_id_sha256",
            "message_id_utf8_bytes",
            "message_id_sha256",
        },
        "browser soak target identity",
    )
    _validate_identity_hashes(target, "chat_id")
    _validate_identity_hashes(target, "message_id")
    submit_started = decimal_timestamp(
        correlation["submit_started_monotonic_ns"], "browser soak submit start"
    )
    submit_completed = decimal_timestamp(
        correlation["submit_completed_monotonic_ns"], "browser soak submit completion"
    )
    first_content = decimal_timestamp(
        correlation["first_content_observed_monotonic_ns"],
        "browser soak first content",
    )
    done = decimal_timestamp(
        correlation["done_observed_monotonic_ns"], "browser soak done"
    )
    done_event_count = integer(
        correlation["done_event_count"], "browser soak done event count"
    )
    cancellation_event_count = integer(
        correlation["cancellation_event_count"],
        "browser soak cancellation event count",
    )
    provider_error_count = integer(
        correlation["provider_error_count"], "browser soak provider error count"
    )
    actions = value["browser_actions"]
    if (
        submit_started
        != decimal_timestamp(actions[2]["started_monotonic_ns"], "submit action start")
        or submit_completed
        != decimal_timestamp(
            actions[2]["completed_monotonic_ns"], "submit action completion"
        )
        or first_content != socket["first_content_ns"]
        or done != socket["done_ns"]
        or first_content < submit_started
        or first_content
        > decimal_timestamp(
            actions[3]["completed_monotonic_ns"], "visible action completion"
        )
        or done < first_content
        or done
        > decimal_timestamp(
            actions[4]["completed_monotonic_ns"], "ready action completion"
        )
        or done_event_count != socket["done_count"]
        or cancellation_event_count != socket["cancel_count"]
        or provider_error_count != socket["provider_error_count"]
    ):
        fail("browser soak socket-to-action correlation differs")
    guard.reject(raw, "browser soak case stdout")
    return {
        "case_index": case_index,
        "browser_case": browser_case(case_index, include_smoke=include_smoke),
        "browser_case_sha256": sha256_bytes(
            browser_case(case_index, include_smoke=include_smoke).encode("utf-8")
        ),
        "record_sha256": sha256_bytes(raw),
        "first_action_ns": first_action,
        "last_action_ns": last_action,
        "submit_started_ns": submit_started,
        "first_content_ns": first_content,
        "done_ns": done,
        "chat_id_sha256": target["chat_id_sha256"],
        "message_id_sha256": target["message_id_sha256"],
        "socket_event_count": len(value["socket_events"]),
    }


def validate_browser_summary(
    value: dict[str, Any],
    raw: bytes,
    summary_path: Path,
    guard: SecretGuardProtocol,
    cases: list[dict[str, Any]],
    *,
    include_smoke: bool = False,
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "record_type",
        "browser_case",
        "observed_monotonic_ns",
        "chat_count",
        "action_count",
        "socket_event_count",
        "browser_process_count",
        "browser_context_count",
        "browser_context_closed_count",
        "page_count_created",
        "page_count_closed",
        "maximum_open_pages",
        "page_error_count",
        "cancellation_event_count",
        "provider_error_count",
        "case_record_sha256",
    }
    if include_smoke:
        expected_fields.update({"mode", "schedule"})
    exact_keys(value, expected_fields, "browser soak summary")
    expected_count = CHAT_COUNT + int(include_smoke)
    if len(cases) != expected_count:
        fail("browser soak summary case count differs")
    expected_case_hashes = [item["record_sha256"] for item in cases]
    expected_socket_events = sum(item["socket_event_count"] for item in cases)
    integer_fields = {
        "chat_count": expected_count,
        "action_count": expected_count * len(FINAL_ACTIONS),
        "socket_event_count": expected_socket_events,
        "browser_process_count": 1,
        "browser_context_count": 1,
        "browser_context_closed_count": 1,
        "page_count_created": expected_count,
        "page_count_closed": expected_count,
        "maximum_open_pages": 1,
        "page_error_count": 0,
        "cancellation_event_count": 0,
        "provider_error_count": 0,
    }
    for field, expected in integer_fields.items():
        if integer(value[field], f"browser soak summary {field}") != expected:
            fail("browser soak summary counts or bounds differ")
    if include_smoke and (
        value["mode"] != COMBINED_MODE
        or value["schedule"] != schedule_evidence(include_smoke=True)
    ):
        fail("browser soak summary mode or schedule differs")
    if (
        value["schema_version"]
        != (COMBINED_BROWSER_SCHEMA if include_smoke else BROWSER_SCHEMA)
        or value["record_type"]
        != (COMBINED_SUMMARY_RECORD_TYPE if include_smoke else "openwebui_soak_summary")
        or value["browser_case"] != (COMBINED_RUN_CASE if include_smoke else RUN_CASE)
        or value["case_record_sha256"] != expected_case_hashes
    ):
        fail("browser soak summary counts, bounds, or case hashes differ")
    observed = decimal_timestamp(
        value["observed_monotonic_ns"], "browser soak summary timestamp"
    )
    if observed < cases[-1]["last_action_ns"]:
        fail("browser soak summary precedes its cases")
    summary_raw = stable_regular_snapshot(
        summary_path, "browser soak summary file", SUPPORT.MAX_JSON_LINE_BYTES + 1
    )
    if summary_raw != raw + b"\n":
        fail("browser soak stdout and summary file differ")
    guard.reject(raw, "browser soak summary stdout")
    result = {
        "chat_count": expected_count,
        "action_count": expected_count * len(FINAL_ACTIONS),
        "socket_event_count": expected_socket_events,
        "browser_summary_bytes": len(summary_raw),
        "browser_summary_sha256": sha256_bytes(summary_raw),
    }
    if include_smoke:
        result["mode"] = COMBINED_MODE
        result["schedule"] = schedule_evidence(include_smoke=True)
    return result


def validate_browser_stdout(
    lines: list[tuple[bytes, dict[str, Any]]],
    summary_path: Path,
    guard: SecretGuardProtocol,
    *,
    base_url: str,
    include_smoke: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indices = case_indices(include_smoke=include_smoke)
    if len(lines) != len(indices) + 1:
        fail("browser stdout record count differs from the frozen soak schedule")
    cases: list[dict[str, Any]] = []
    seen_chat_ids: set[str] = set()
    seen_message_ids: set[str] = set()
    prior_completed = -1
    for case_index, (raw, value) in zip(indices, lines[:-1], strict=True):
        evidence = validate_browser_case(
            value,
            raw,
            guard,
            case_index=case_index,
            base_url=base_url,
            include_smoke=include_smoke,
        )
        chat_id = evidence["chat_id_sha256"]
        message_id = evidence["message_id_sha256"]
        if chat_id in seen_chat_ids or message_id in seen_message_ids:
            fail("browser soak chat/message correlation is duplicated")
        if evidence["first_action_ns"] < prior_completed:
            fail("browser soak cases overlap or regress")
        seen_chat_ids.add(chat_id)
        seen_message_ids.add(message_id)
        prior_completed = evidence["last_action_ns"]
        cases.append(evidence)
    summary_raw, summary_value = lines[-1]
    summary = validate_browser_summary(
        summary_value,
        summary_raw,
        summary_path,
        guard,
        cases,
        include_smoke=include_smoke,
    )
    return cases, summary


def validate_gateway_traces(
    machine: SoakLifecycleMachine,
    browser_cases: list[dict[str, Any]],
    *,
    include_smoke: bool = False,
) -> list[dict[str, Any]]:
    indices = case_indices(include_smoke=include_smoke)
    expected_count = len(indices)
    if (
        machine.expected_count != expected_count
        or len(machine.traces) != expected_count
        or len(browser_cases) != expected_count
        or machine.active is not None
        or machine.max_active != 1
    ):
        fail("gateway soak request count, activity, or concurrency differs")
    correlations: list[dict[str, Any]] = []
    prior_release = -1
    for case_index, trace, browser in zip(
        indices, machine.traces, browser_cases, strict=True
    ):
        admitted = trace.event("request_admitted")
        started = trace.event("request_started")
        first_token = trace.event("request_first_token")
        released = trace.event("request_released")
        admitted_ns = admitted["observed_monotonic_ns"]
        released_ns = released["observed_monotonic_ns"]
        if (
            admitted_ns < prior_release
            or started["observed_monotonic_ns"] < admitted_ns
            or first_token["observed_monotonic_ns"] < started["observed_monotonic_ns"]
            or released_ns < first_token["observed_monotonic_ns"]
            or admitted_ns < browser["submit_started_ns"]
            or first_token["observed_monotonic_ns"] > browser["first_content_ns"]
            or released_ns > browser["done_ns"]
            or released["outcome"] != "stop"
            or released["cancel_reason"] is not None
            or released["reset_complete"] is not True
        ):
            fail("gateway soak release or browser ordering differs")
        prior_release = released_ns
        correlation = {
            "case_index": case_index,
            "browser_case_sha256": browser["browser_case_sha256"],
            "chat_id_sha256": browser["chat_id_sha256"],
            "message_id_sha256": browser["message_id_sha256"],
            "request_id_sha256": sha256_bytes(trace.request_id.encode("utf-8")),
            "completion_id_sha256": sha256_bytes(trace.completion_id.encode("utf-8")),
            "admitted_monotonic_ns": str(admitted_ns),
            "released_monotonic_ns": str(released_ns),
            "outcome": "stop",
            "reset_complete": True,
        }
        if include_smoke:
            correlation["browser_case"] = browser["browser_case"]
        correlations.append(correlation)
    return correlations


def stop_and_validate_journal(
    journal: JournalFollowerProtocol,
    observer_records: list[ObserverRecordProtocol],
) -> dict[str, Any]:
    journal.stop()
    observed = [record.raw for record in observer_records]
    captured = list(journal.lifecycle)
    require_correlated_prefix(observed, captured)
    if len(captured) != len(observed):
        fail("observer and journal lifecycle counts differ at final seal")
    return {
        "records": len(journal.records),
        "unique_cursors": len(journal.cursors),
        "lifecycle_records": len(captured),
        "stderr_bytes": journal.stderr_bytes,
        "stderr_sha256": journal.stderr_digest.hexdigest(),
    }


def execute(args: argparse.Namespace) -> None:
    include_smoke = getattr(args, "include_smoke", False)
    if not isinstance(include_smoke, bool):
        fail("browser soak CLI mode flag is not boolean")
    indices = case_indices(include_smoke=include_smoke)
    expected_count = len(indices)
    output = AtomicRunDirectory(args.output_dir)
    observer: LifecycleObserver | None = None
    journal: JournalFollowerProtocol | None = None
    browser: BrowserProcess | None = None
    browser_process: subprocess.Popen[bytes] | None = None
    journal_process: subprocess.Popen[bytes] | None = None
    observer_writer: LineWriterProtocol | None = None
    journal_writer: LineWriterProtocol | None = None
    browser_writer: LineWriterProtocol | None = None
    container_name = f"ullm-browser-soak-{os.getpid()}-{os.urandom(8).hex()}"
    deadline_ns = time.monotonic_ns() + args.timeout_seconds * 1_000_000_000
    try:
        script_raw = stable_regular_snapshot(
            args.browser_script, "browser soak script", MAX_BROWSER_SCRIPT_BYTES
        )
        runner_raw = GATE_SOURCE_RAW
        support_raw = SUPPORT_SOURCE_RAW
        token = stable_regular_snapshot(args.token_file, "OpenWebUI token file", 65_536)
        script = output.stage / "runtime" / "browser-soak.cjs"
        token_file = output.stage / "runtime" / "openwebui-token"
        browser_container_output = (
            output.stage / "runtime" / BROWSER_CONTAINER_OUTPUT_DIR_NAME
        )
        try:
            browser_container_output.mkdir(mode=0o700)
        except OSError:
            fail("failed to create isolated browser soak output staging")
        write_private_snapshot(script, script_raw, "browser soak script")
        write_private_snapshot(token_file, token, "OpenWebUI token")
        try:
            token_text = token.decode("utf-8", errors="strict")
        except UnicodeError:
            fail("OpenWebUI token is not UTF-8")
        if token_text.endswith("\n"):
            token_text = token_text[:-1]
        if (
            not token_text
            or len(token_text.encode("utf-8")) < 8
            or token_text.strip() != token_text
            or any(character in token_text for character in "\r\n\0")
        ):
            fail("OpenWebUI token is not one strict line")
        url = normalized_url(args.openwebui_url)
        browser_image, browser_content_digest = normalized_browser_image(
            args.browser_image
        )
        sensitive = [
            token_text.encode("utf-8"),
            url.encode("utf-8"),
            MODEL_ID.encode("utf-8"),
            MODEL_LABEL.encode("utf-8"),
            *(
                case_marker(index, include_smoke=include_smoke).encode("utf-8")
                for index in indices
            ),
            *(
                case_prompt(index, include_smoke=include_smoke).encode("utf-8")
                for index in indices
            ),
        ]
        if "@" in browser_image:
            sensitive.append(browser_image.encode("utf-8"))
        base_guard = SecretGuard(sensitive)

        initial_identity = query_service_identity(args.systemctl, args.service)
        if os.geteuid() != initial_identity.uid:
            fail("browser soak gate must run as the gateway service user")
        boot_id = read_boot_id()
        cursor = initial_journal_cursor(args.journalctl, args.service)

        observer_writer = AtomicLineWriter(
            output.stage / "observer.raw.jsonl", maximum_bytes=16 * 1024 * 1024
        )
        journal_writer = AtomicLineWriter(
            output.stage / "service-journal.raw.jsonl",
            maximum_bytes=SUPPORT.MAX_JOURNAL_BYTES,
        )
        browser_writer = AtomicLineWriter(
            output.stage / "browser" / "browser-stdout.jsonl",
            maximum_bytes=MAX_BROWSER_RAW_BYTES,
        )
        observer = LifecycleObserver(
            args.observer_socket,
            initial_identity.main_pid,
            initial_identity.uid,
            observer_writer,
            expected_count=expected_count,
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
            browser_output=browser_container_output,
            openwebui_url=url,
            uid=os.geteuid(),
            gid=os.getegid(),
            include_smoke=include_smoke,
        )
        browser_process = spawn_browser(command)
        browser = BrowserProcess(
            browser_process,
            browser_writer,
            maximum_lines=(
                MAX_COMBINED_BROWSER_LINES if include_smoke else MAX_BROWSER_LINES
            ),
        )
        browser.start()
        code = browser.wait_exit(deadline_ns)
        if code != 0:
            fail("browser soak process failed")
        browser_cases, browser_evidence = validate_browser_stdout(
            browser.lines,
            validate_container_output_directory(browser_container_output),
            base_guard,
            base_url=url,
            include_smoke=include_smoke,
        )

        observer.wait_for(
            lambda machine: (
                len(machine.traces) == expected_count
                and machine.active is None
                and machine.traces[-1].released
            ),
            deadline_ns,
        )
        correlations = validate_gateway_traces(
            observer.machine, browser_cases, include_smoke=include_smoke
        )
        journal.wait_correlated(observer, deadline_ns)
        time.sleep(0.2)
        journal.wait_correlated(observer, deadline_ns)
        if len(journal.lifecycle) != len(observer.records):
            fail("observer and journal lifecycle counts differ")

        observer.close()
        machine = observer.machine
        observer_records = list(observer.records)
        observer = None
        correlations = validate_gateway_traces(
            machine, browser_cases, include_smoke=include_smoke
        )
        final_identity = query_service_identity(args.systemctl, args.service)
        final_boot_id = read_boot_id()
        if final_identity != initial_identity or final_boot_id != boot_id:
            fail("gateway service or boot identity changed during the browser soak")
        dynamic_ids = [
            value
            for trace in machine.traces
            for value in (trace.request_id, trace.completion_id)
        ]
        summary_guard = base_guard.extend(dynamic_ids)
        journal.wait_correlated_records(lambda: observer_records, deadline_ns)
        time.sleep(0.2)
        sealed_journal = journal
        journal = None
        journal_evidence = stop_and_validate_journal(sealed_journal, observer_records)

        browser_writer.commit()
        observer_writer.commit()
        journal_writer.commit()
        browser_summary_source = validate_container_output_directory(
            browser_container_output
        )
        browser_summary_raw = snapshot_validated_browser_summary(
            browser_summary_source,
            output.stage / "browser" / BROWSER_SUMMARY_NAME,
            browser.lines[-1][0],
        )
        try:
            browser_summary_source.unlink()
            browser_container_output.rmdir()
            script.unlink()
            token_file.unlink()
            (output.stage / "runtime").rmdir()
            (output.stage / "control").rmdir()
        except OSError:
            fail("failed to remove private browser soak runtime staging")

        summary: dict[str, Any] = {
            "schema_version": (COMBINED_GATE_SCHEMA if include_smoke else GATE_SCHEMA),
            "passed": True,
            "service": {
                "unit_sha256": sha256_bytes(args.service.encode("utf-8")),
                "main_pid_sha256": sha256_bytes(
                    str(initial_identity.main_pid).encode("ascii")
                ),
                "user_uid_sha256": sha256_bytes(
                    str(initial_identity.uid).encode("ascii")
                ),
                "user_gid_sha256": sha256_bytes(
                    str(initial_identity.gid).encode("ascii")
                ),
                "boot_id_sha256": sha256_bytes(boot_id.encode("ascii")),
                "restart_count": initial_identity.restarts,
                "identity_invariant": True,
            },
            "browser": {
                "image_reference_sha256": sha256_bytes(browser_image.encode("utf-8")),
                "image_content_digest": browser_content_digest,
                "script_sha256": sha256_bytes(script_raw),
                "gate_source_sha256": sha256_bytes(runner_raw),
                "support_source_sha256": sha256_bytes(support_raw),
                **browser_evidence,
                "stdout_lines": browser_writer.lines_written,
                "stdout_bytes": browser_writer.bytes_written,
                "stdout_sha256": browser_writer.sha256,
                "stderr_bytes": browser.stderr_bytes,
                "stderr_sha256": browser.stderr_digest.hexdigest(),
            },
            "gateway": {
                "request_count": len(machine.traces),
                "maximum_active_requests": machine.max_active,
                "stop_release_count": len(machine.traces),
                "reset_complete_count": len(machine.traces),
                "every_admission_after_previous_release": True,
                "correlations": correlations,
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
                    "records": journal_evidence["records"],
                    "sha256": journal_writer.sha256,
                    "unique_cursors": journal_evidence["unique_cursors"],
                    "lifecycle_records": journal_evidence["lifecycle_records"],
                    "stderr_bytes": journal_evidence["stderr_bytes"],
                    "stderr_sha256": journal_evidence["stderr_sha256"],
                },
                "browser_stdout": {
                    "file": "browser/browser-stdout.jsonl",
                    "bytes": browser_writer.bytes_written,
                    "records": browser_writer.lines_written,
                    "sha256": browser_writer.sha256,
                },
                "browser_summary": {
                    "file": f"browser/{BROWSER_SUMMARY_NAME}",
                    "bytes": len(browser_summary_raw),
                    "sha256": sha256_bytes(browser_summary_raw),
                },
            },
        }
        if include_smoke:
            summary["mode"] = COMBINED_MODE
            summary["schedule"] = schedule_evidence(include_smoke=True)
        summary_guard.reject(compact_json(summary), "browser soak gate summary")
        write_atomic_json(output.stage / "summary.json", summary, summary_guard)
        for path in (
            output.stage / "observer.raw.jsonl",
            output.stage / "service-journal.raw.jsonl",
        ):
            base_guard.scan_file(path, "browser soak credential-safe raw artifact")
        for path in (
            output.stage / "summary.json",
            output.stage / "browser" / "browser-stdout.jsonl",
            output.stage / "browser" / BROWSER_SUMMARY_NAME,
        ):
            summary_guard.scan_file(path, "browser soak redacted pass artifact")
        fsync_bundle_tree(output.stage)
        output.publish()
    except BaseException:
        if browser_process is not None:
            try:
                run_bounded_command(
                    [args.docker, "rm", "--force", container_name],
                    "browser soak container cleanup",
                    timeout_seconds=15.0,
                )
            except SoakGateError:
                pass
            if browser_process.poll() is None:
                try:
                    terminate_process_group(browser_process)
                except SoakGateError:
                    pass
        if journal is not None:
            try:
                journal.stop()
            except SoakGateError:
                pass
        elif journal_process is not None and journal_process.poll() is None:
            try:
                terminate_process_group(journal_process)
            except SoakGateError:
                pass
        if observer is not None:
            try:
                observer.close()
            except SoakGateError:
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
        / "browser-soak.cjs",
    )
    parser.add_argument("--observer-socket", type=Path, default=OBSERVER_SOCKET)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--systemctl", default="systemctl")
    parser.add_argument("--journalctl", default="journalctl")
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="run one fixed OpenWebUI smoke before the exact 20-chat soak",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        choices=range(300, 3601),
        metavar="[300-3600]",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        execute(parse_args(argv))
        return 0
    except KeyboardInterrupt:
        print("OpenWebUI browser soak gate interrupted", file=sys.stderr)
        return 130
    except Exception:
        print("OpenWebUI browser soak gate failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
