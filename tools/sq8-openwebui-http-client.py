#!/usr/bin/env python3
"""Stream raw HTTP evidence from the OpenWebUI Docker network.

The process accepts one strict JSON command per stdin line and emits one compact
JSON event per stdout line.  It deliberately interprets SSE only far enough to
implement the optional close-after-first-content trigger.  Raw body chunks are
the authoritative evidence.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import dataclasses
import hashlib
import http.client
import json
import math
import os
import re
import socket
import stat
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, BinaryIO


COMMAND_SCHEMA = "ullm.sq8.openwebui_http_client.command.v1"
EVENT_SCHEMA = "ullm.sq8.openwebui_http_client.event.v1"
MAX_COMMAND_LINE_BYTES = 8 * 1024 * 1024
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_API_KEY_FILE_BYTES = 4096
MAX_SSE_LINE_BYTES = 1024 * 1024
MAX_SSE_EVENT_BYTES = 2 * 1024 * 1024
DEFAULT_READ_CHUNK_BYTES = 64 * 1024
MAX_READ_CHUNK_BYTES = 1024 * 1024
REQUEST_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
ALLOWED_METHODS = frozenset({"GET", "POST"})
AUTHORIZATION_MODES = frozenset({"valid_bearer", "invalid_bearer", "missing"})


class ClientError(RuntimeError):
    """A bounded, credential-free protocol or transport diagnostic."""


def fail(message: str) -> None:
    raise ClientError(message)


def exact_keys(value: dict[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        fail("command fields differ from the required schema")


def duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("duplicate JSON key")
        result[key] = value
    return result


def reject_json_constant(_value: str) -> None:
    fail("non-finite JSON number")


def parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        fail("non-finite JSON number")
    return parsed


def strict_json_bytes(raw: bytes, label: str, *, maximum: int) -> Any:
    if len(raw) > maximum:
        fail(f"{label} exceeds its size limit")
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=duplicate_rejecting_object,
            parse_float=parse_finite_float,
            parse_constant=reject_json_constant,
        )
    except ClientError:
        raise
    except (UnicodeError, ValueError, RecursionError):
        fail(f"{label} is not strict UTF-8 JSON")


def strict_json_object(raw: bytes, label: str, *, maximum: int) -> dict[str, Any]:
    value = strict_json_bytes(raw, label, maximum=maximum)
    if not isinstance(value, dict):
        fail(f"{label} root must be an object")
    return value


def nonempty_string(value: Any, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        fail(f"{label} must be a bounded non-empty string")
    return value


def strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        fail(f"{label} must be a boolean")
    return value


def canonical_base64(value: Any) -> tuple[str, bytes]:
    if not isinstance(value, str) or len(value) > ((MAX_BODY_BYTES + 2) // 3) * 4:
        fail("body_base64 must be a bounded string")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        fail("body_base64 is invalid")
    if len(raw) > MAX_BODY_BYTES:
        fail("decoded request body exceeds its size limit")
    canonical = base64.b64encode(raw).decode("ascii")
    if value != canonical:
        fail("body_base64 is not canonical")
    return value, raw


@dataclasses.dataclass(frozen=True)
class RequestCommand:
    request_key: str
    method: str
    target: str
    body_base64: str
    body: bytes
    authorization_mode: str
    close_on_first_nonempty_sse_content: bool


@dataclasses.dataclass(frozen=True)
class CloseCommand:
    request_key: str


@dataclasses.dataclass(frozen=True)
class ShutdownCommand:
    pass


Command = RequestCommand | CloseCommand | ShutdownCommand


def parse_command_line(raw: bytes) -> Command:
    value = strict_json_object(raw, "command line", maximum=MAX_COMMAND_LINE_BYTES)
    if value.get("schema_version") != COMMAND_SCHEMA:
        fail("command schema_version mismatch")
    command = value.get("command")
    if command == "request":
        exact_keys(
            value,
            {
                "schema_version",
                "command",
                "request_key",
                "method",
                "target",
                "body_base64",
                "authorization_mode",
                "close_on_first_nonempty_sse_content",
            },
        )
        request_key = nonempty_string(value["request_key"], "request_key", maximum=128)
        if REQUEST_KEY_RE.fullmatch(request_key) is None:
            fail("request_key has invalid syntax")
        method = nonempty_string(value["method"], "method", maximum=16)
        if method not in ALLOWED_METHODS:
            fail("method is unsupported")
        target = nonempty_string(value["target"], "target", maximum=4096)
        try:
            target_bytes = target.encode("ascii", errors="strict")
        except UnicodeError:
            fail("target must be ASCII")
        if (
            not target.startswith("/")
            or target.startswith("//")
            or b"#" in target_bytes
            or any(byte <= 0x20 or byte == 0x7F for byte in target_bytes)
        ):
            fail("target is not an origin-form HTTP target")
        body_base64, body = canonical_base64(value["body_base64"])
        authorization_mode = nonempty_string(
            value["authorization_mode"], "authorization_mode", maximum=32
        )
        if authorization_mode not in AUTHORIZATION_MODES:
            fail("authorization_mode is unsupported")
        close_on_content = strict_bool(
            value["close_on_first_nonempty_sse_content"],
            "close_on_first_nonempty_sse_content",
        )
        return RequestCommand(
            request_key=request_key,
            method=method,
            target=target,
            body_base64=body_base64,
            body=body,
            authorization_mode=authorization_mode,
            close_on_first_nonempty_sse_content=close_on_content,
        )
    if command == "close":
        exact_keys(value, {"schema_version", "command", "request_key"})
        request_key = nonempty_string(value["request_key"], "request_key", maximum=128)
        if REQUEST_KEY_RE.fullmatch(request_key) is None:
            fail("request_key has invalid syntax")
        return CloseCommand(request_key=request_key)
    if command == "shutdown":
        exact_keys(value, {"schema_version", "command"})
        return ShutdownCommand()
    fail("command is unsupported")


def read_bounded_line(stream: BinaryIO) -> bytes | None:
    raw = stream.readline(MAX_COMMAND_LINE_BYTES + 2)
    if raw == b"":
        return None
    if (
        len(raw) > MAX_COMMAND_LINE_BYTES + 1
        or not raw.endswith(b"\n")
        or raw.endswith(b"\r\n")
    ):
        fail("command line exceeds its size limit or lacks LF termination")
    return raw[:-1]


def read_api_key(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        fail("this platform does not support no-follow credential reads")
    flags |= nofollow
    try:
        descriptor = os.open(path, flags)
    except OSError:
        fail("failed to open API key file without following links")
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            fail("API key path must be a regular file")
        if metadata.st_size < 1 or metadata.st_size > MAX_API_KEY_FILE_BYTES:
            fail("API key file size is out of range")
        chunks: list[bytes] = []
        remaining = MAX_API_KEY_FILE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) != metadata.st_size:
            fail("API key file changed while it was read")
    finally:
        os.close(descriptor)
    if raw.endswith(b"\n"):
        raw = raw[:-1]
    if not raw or b"\n" in raw or b"\r" in raw:
        fail("API key file must contain exactly one non-empty line")
    if any(byte < 0x21 or byte > 0x7E for byte in raw):
        fail("API key contains an invalid HTTP header byte")
    return raw


@dataclasses.dataclass(frozen=True)
class Endpoint:
    host: str
    port: int


def parse_base_url(value: str) -> Endpoint:
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError:
        fail("base URL is invalid")
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        fail("base URL must be an HTTP origin without credentials or a path")
    return Endpoint(parsed.hostname, port or 80)


class CredentialGuard:
    def __init__(self, api_key: bytes):
        self.api_key = api_key

    def reject_bytes(self, value: bytes, label: str) -> None:
        if self.api_key in value:
            fail(f"{label} contains credential material")

    def authorization_value(self, mode: str) -> str | None:
        if mode == "missing":
            return None
        if mode == "invalid_bearer":
            return "Bearer ullm-intentionally-invalid"
        return "Bearer " + self.api_key.decode("ascii")


class EventSink:
    def __init__(self, stream: BinaryIO, credential: CredentialGuard):
        self._stream = stream
        self._credential = credential
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> None:
        value = {"schema_version": EVENT_SCHEMA, "event": event, **fields}
        try:
            raw = json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("ascii")
        except (TypeError, ValueError):
            fail("internal event serialization failed")
        self._credential.reject_bytes(raw, "event")
        with self._lock:
            self._stream.write(raw + b"\n")
            self._stream.flush()


class StrictSseContentTrigger:
    """Incrementally detect one non-empty OpenAI delta.content string."""

    def __init__(self):
        self._line = bytearray()
        self._data_lines: list[bytes] = []
        self._event_bytes = 0
        self._previous_was_cr = False

    def feed(self, raw: bytes) -> bool:
        for byte in raw:
            if self._previous_was_cr:
                self._previous_was_cr = False
                if byte == 0x0A:
                    continue
            if byte == 0x0D:
                if self._finish_line():
                    return True
                self._previous_was_cr = True
            elif byte == 0x0A:
                if self._finish_line():
                    return True
            else:
                self._line.append(byte)
                if len(self._line) > MAX_SSE_LINE_BYTES:
                    fail("SSE line exceeds its size limit")
        return False

    def _finish_line(self) -> bool:
        line = bytes(self._line)
        self._line.clear()
        if not line:
            return self._dispatch()
        if line.startswith(b":"):
            return False
        field, separator, value = line.partition(b":")
        if separator and value.startswith(b" "):
            value = value[1:]
        if field == b"data":
            self._event_bytes += len(value)
            if self._data_lines:
                self._event_bytes += 1
            if self._event_bytes > MAX_SSE_EVENT_BYTES:
                fail("SSE event exceeds its size limit")
            self._data_lines.append(value)
        return False

    def _dispatch(self) -> bool:
        if not self._data_lines:
            self._event_bytes = 0
            return False
        data = b"\n".join(self._data_lines)
        self._data_lines.clear()
        self._event_bytes = 0
        if data == b"[DONE]":
            return False
        value = strict_json_object(data, "SSE data", maximum=MAX_SSE_EVENT_BYTES)
        choices = value.get("choices")
        if choices is None:
            return False
        if not isinstance(choices, list):
            fail("SSE choices must be an array")
        if not choices:
            return False
        first = choices[0]
        if not isinstance(first, dict):
            fail("SSE first choice must be an object")
        delta = first.get("delta")
        if delta is None:
            return False
        if not isinstance(delta, dict):
            fail("SSE delta must be an object")
        content = delta.get("content")
        if content is None:
            return False
        if not isinstance(content, str):
            fail("SSE delta.content must be a string or null")
        return bool(content)


class ActiveRequest:
    def __init__(self, command: RequestCommand):
        self.command = command
        self.thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._socket: socket.socket | None = None
        self._close_requested = False

    def bind_socket(self, transport: socket.socket) -> None:
        with self._lock:
            self._socket = transport
            close_requested = self._close_requested
        if close_requested:
            self._shutdown(transport)

    def request_close(self) -> None:
        with self._lock:
            self._close_requested = True
            transport = self._socket
        if transport is not None:
            self._shutdown(transport)

    def mark_close_without_shutdown(self) -> None:
        with self._lock:
            self._close_requested = True

    def close_requested(self) -> bool:
        with self._lock:
            return self._close_requested

    @staticmethod
    def _shutdown(transport: socket.socket) -> None:
        try:
            transport.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            transport.close()
        except OSError:
            pass


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    endpoint: Endpoint
    socket_timeout_seconds: float
    read_chunk_bytes: int


class RequestController:
    def __init__(
        self,
        config: RuntimeConfig,
        credential: CredentialGuard,
        sink: EventSink,
    ):
        self._config = config
        self._credential = credential
        self._sink = sink
        self._condition = threading.Condition()
        self._active: ActiveRequest | None = None

    def has_active(self) -> bool:
        with self._condition:
            return self._active is not None

    def start(self, command: RequestCommand) -> None:
        self._credential.reject_bytes(
            command.request_key.encode("ascii"), "request_key"
        )
        self._credential.reject_bytes(command.target.encode("ascii"), "target")
        self._credential.reject_bytes(command.body, "request body")
        with self._condition:
            if self._active is not None:
                fail("a request is already active; waiting requests are forbidden")
            active = ActiveRequest(command)
            thread = threading.Thread(
                target=self._run_request,
                args=(active,),
                name="sq8-http-request",
                daemon=True,
            )
            active.thread = thread
            self._active = active
            thread.start()

    def close(self, command: CloseCommand) -> None:
        with self._condition:
            active = self._active
            if active is None:
                fail("close requires one active request")
            if active.command.request_key != command.request_key:
                fail("close request_key does not match the active request")
            active.request_close()

    def abort(self) -> None:
        with self._condition:
            active = self._active
        if active is not None:
            active.request_close()

    def wait_inactive(self, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._active is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def _finish(self, active: ActiveRequest, **end_fields: Any) -> None:
        # Keep end publication and active-slot release atomic to a new command.
        with self._condition:
            if self._active is not active:
                return
            self._sink.emit("http_response_end", **end_fields)
            self._active = None
            self._condition.notify_all()

    def _run_request(self, active: ActiveRequest) -> None:
        command = active.command
        body_digest = hashlib.sha256()
        body_bytes = 0
        outcome = "error"
        error: str | None = "HTTP request failed"
        connection: http.client.HTTPConnection | None = None
        try:
            connection = http.client.HTTPConnection(
                self._config.endpoint.host,
                self._config.endpoint.port,
                timeout=self._config.socket_timeout_seconds,
            )
            connection.connect()
            connect_completed = time.monotonic_ns()
            if connection.sock is None:
                fail("HTTP connection has no socket")
            active.bind_socket(connection.sock)
            if active.close_requested():
                outcome, error = "client_closed", None
                return
            connection.putrequest(
                command.method,
                command.target,
                skip_accept_encoding=True,
            )
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(len(command.body)))
            connection.putheader("Accept", "text/event-stream, application/json")
            authorization = self._credential.authorization_value(
                command.authorization_mode
            )
            if authorization is not None:
                connection.putheader("Authorization", authorization)
            write_started = time.monotonic_ns()
            connection.endheaders(command.body)
            last_body_byte_sent = time.monotonic_ns()
            self._sink.emit(
                "http_request",
                request_key=command.request_key,
                method=command.method,
                target=command.target,
                headers={
                    "content_type": "application/json",
                    "content_length": len(command.body),
                    "authorization_mode": command.authorization_mode,
                },
                body_base64=command.body_base64,
                body_sha256=hashlib.sha256(command.body).hexdigest(),
                body_bytes=len(command.body),
                connect_completed_monotonic_ns=connect_completed,
                write_started_monotonic_ns=write_started,
                last_body_byte_sent_monotonic_ns=last_body_byte_sent,
            )
            response = connection.getresponse()
            observed_start = time.monotonic_ns()
            response_headers = [[name, value] for name, value in response.getheaders()]
            for name, value in response_headers:
                self._credential.reject_bytes(
                    name.encode("latin-1", errors="replace"), "response header"
                )
                self._credential.reject_bytes(
                    value.encode("latin-1", errors="replace"), "response header"
                )
            self._sink.emit(
                "http_response_start",
                request_key=command.request_key,
                status=response.status,
                headers=response_headers,
                observed_monotonic_ns=observed_start,
            )
            trigger = (
                StrictSseContentTrigger()
                if command.close_on_first_nonempty_sse_content
                else None
            )
            chunk_index = 0
            while True:
                chunk = response.read1(self._config.read_chunk_bytes)
                observed_chunk = time.monotonic_ns()
                if not chunk:
                    if active.close_requested():
                        outcome, error = "client_closed", None
                    else:
                        outcome, error = "eof", None
                    break
                self._credential.reject_bytes(chunk, "response body")
                body_digest.update(chunk)
                body_bytes += len(chunk)
                self._sink.emit(
                    "http_body_chunk",
                    request_key=command.request_key,
                    chunk_index=chunk_index,
                    body_base64=base64.b64encode(chunk).decode("ascii"),
                    body_sha256=hashlib.sha256(chunk).hexdigest(),
                    body_bytes=len(chunk),
                    observed_monotonic_ns=observed_chunk,
                )
                chunk_index += 1
                if trigger is not None and trigger.feed(chunk):
                    active.mark_close_without_shutdown()
                    outcome, error = "client_closed", None
                    break
        except (socket.timeout, TimeoutError):
            if active.close_requested():
                outcome, error = "client_closed", None
            else:
                outcome, error = "timeout", "socket timeout"
        except ClientError as exception:
            outcome, error = "error", str(exception)
        except (OSError, http.client.HTTPException) as exception:
            if active.close_requested():
                outcome, error = "client_closed", None
            else:
                outcome, error = "error", type(exception).__name__
        except Exception as exception:  # pragma: no cover - final fail-closed guard
            outcome, error = "error", type(exception).__name__
        finally:
            if connection is not None:
                connection.close()
            self._finish(
                active,
                request_key=command.request_key,
                outcome=outcome,
                error=error,
                body_bytes=body_bytes,
                body_sha256=body_digest.hexdigest(),
                observed_monotonic_ns=time.monotonic_ns(),
            )


def positive_finite_timeout(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0 or parsed > 3600:
        raise argparse.ArgumentTypeError("must be finite and in (0, 3600]")
    return parsed


def bounded_chunk_size(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1 or parsed > MAX_READ_CHUNK_BYTES:
        raise argparse.ArgumentTypeError(f"must be in [1, {MAX_READ_CHUNK_BYTES}]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument(
        "--socket-timeout-seconds",
        type=positive_finite_timeout,
        default=180.0,
    )
    parser.add_argument(
        "--read-chunk-bytes",
        type=bounded_chunk_size,
        default=DEFAULT_READ_CHUNK_BYTES,
    )
    return parser


def run(
    config: RuntimeConfig,
    api_key: bytes,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
) -> int:
    credential = CredentialGuard(api_key)
    sink = EventSink(output_stream, credential)
    controller = RequestController(config, credential, sink)
    try:
        sink.emit("ready", observed_monotonic_ns=time.monotonic_ns())
        while True:
            raw = read_bounded_line(input_stream)
            if raw is None:
                if controller.has_active():
                    fail("stdin closed while a request was active")
                return 0
            command = parse_command_line(raw)
            if isinstance(command, RequestCommand):
                controller.start(command)
            elif isinstance(command, CloseCommand):
                controller.close(command)
            else:
                if controller.has_active():
                    fail("shutdown requires an idle client")
                sink.emit(
                    "shutdown_complete", observed_monotonic_ns=time.monotonic_ns()
                )
                return 0
    except ClientError as exception:
        controller.abort()
        controller.wait_inactive(2.0)
        try:
            sink.emit("command_error", error=str(exception))
        except (ClientError, BrokenPipeError):
            pass
        return 2
    except (BrokenPipeError, OSError):
        controller.abort()
        controller.wait_inactive(2.0)
        return 2


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        endpoint = parse_base_url(arguments.base_url)
        api_key = read_api_key(arguments.api_key_file)
    except ClientError as exception:
        print(f"sq8 HTTP client: {exception}", file=sys.stderr)
        return 2
    config = RuntimeConfig(
        endpoint=endpoint,
        socket_timeout_seconds=arguments.socket_timeout_seconds,
        read_chunk_bytes=arguments.read_chunk_bytes,
    )
    return run(config, api_key, sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
