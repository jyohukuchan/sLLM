#!/usr/bin/env python3
"""Run the non-GPU OpenAI API contract preflight from the OpenWebUI network."""

from __future__ import annotations

import argparse
import base64
import binascii
import dataclasses
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, NoReturn, Protocol, Sequence, cast


MAX_SUPPORT_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_HTTP_EVENTS = 128
REQUEST_TIMEOUT_NS = 10_000_000_000
QUIET_DRAIN_NS = 250_000_000
POST_POLL_DRAIN_NS = 50_000_000
GATE_SCHEMA = "ullm.sq8.api_contract_gate.v1"
HTTP_COMMAND_SCHEMA = "ullm.sq8.openwebui_http_client.command.v1"
HTTP_EVENT_SCHEMA = "ullm.sq8.openwebui_http_client.event.v1"
MODEL_ID = "ullm-qwen3-14b-sq8"
HTTP_CLIENT_SHA256 = "a64642a0f31bcdd92cf02883e195ee270b9752ee6117908b789cc66187053285"
OBSERVER_SOCKET = Path("/run/ullm/lifecycle-observer.sock")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
REQUEST_KEY_RE = re.compile(r"api-contract-[0-9]{2}-[a-z0-9-]+\Z")
GATEWAY_SOURCE_RELATIVES = (
    "services/openai-gateway/src/ullm_openai_gateway/app.py",
    "services/openai-gateway/src/ullm_openai_gateway/errors.py",
    "services/openai-gateway/src/ullm_openai_gateway/schemas.py",
)
INVALID_KEY_MESSAGE = "The supplied API key is invalid."
QUERY_MESSAGE = "Query parameters are not supported."
INVALID_JSON_MESSAGE = "The request body is not valid JSON."
UNSUPPORTED_MESSAGE = "The requested parameter is not supported."
MODEL_NOT_FOUND_MESSAGE = "The requested model does not exist."


class GateError(RuntimeError):
    """A fail-closed API contract error without credential-bearing detail."""


def fail(message: str) -> NoReturn:
    raise GateError(message)


def _snapshot(path: Path, label: str, maximum: int) -> tuple[bytes, tuple[int, ...]]:
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
        fail(f"failed to snapshot {label}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _load_direct_support() -> tuple[Any, bytes, tuple[int, ...]]:
    path = Path(__file__).with_name("run-sq8-direct-cancel-gate.py")
    raw, identity = _snapshot(path, "direct cancellation support", MAX_SUPPORT_BYTES)
    name = "_ullm_sq8_api_contract_direct_support"
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


DIRECT, DIRECT_SUPPORT_RAW, DIRECT_SUPPORT_IDENTITY = _load_direct_support()


def compact_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        fail("failed to serialize canonical JSON")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("JSON evidence contains a duplicate key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    fail("JSON evidence contains a non-finite number")


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        fail("JSON evidence contains a non-finite number")
    return parsed


def strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    if not raw or len(raw) > MAX_RESPONSE_BYTES:
        fail(f"{label} has an invalid size")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=_parse_float,
            parse_constant=_reject_constant,
        )
    except GateError:
        raise
    except (UnicodeError, ValueError, RecursionError):
        fail(f"{label} is not strict UTF-8 JSON")
    if type(value) is not dict:
        fail(f"{label} root is not an object")
    return cast(dict[str, Any], value)


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        fail(f"{label} fields differ")
    return cast(dict[str, Any], value)


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail(f"{label} is not an integer >= {minimum}")
    return value


def text(value: Any, label: str, *, maximum: int = 4096) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        fail(f"{label} is not bounded non-empty text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        fail(f"{label} is not strict UTF-8")
    return value


def sha256_value(value: Any, label: str) -> str:
    candidate = text(value, label, maximum=64)
    if SHA256_RE.fullmatch(candidate) is None:
        fail(f"{label} is not lowercase SHA-256")
    return candidate


CANONICAL_BODY = compact_json(
    {
        "messages": [{"content": "API contract preflight", "role": "user"}],
        "model": MODEL_ID,
    }
)
MALFORMED_BODY = b'{"broken":'
DUPLICATE_KEY_BODY = (
    b'{"model":"'
    + MODEL_ID.encode("ascii")
    + b'","model":"'
    + MODEL_ID.encode("ascii")
    + b'","messages":[{"role":"user","content":"API contract preflight"}]}'
)
UNSUPPORTED_N_BODY = compact_json(
    {
        "messages": [{"content": "API contract preflight", "role": "user"}],
        "model": MODEL_ID,
        "n": 2,
    }
)
MISSING_MODEL_BODY = compact_json(
    {
        "messages": [{"content": "API contract preflight", "role": "user"}],
        "model": "missing",
    }
)


@dataclasses.dataclass(frozen=True)
class ApiCase:
    case_id: str
    method: str
    target: str
    body: bytes
    authorization_mode: str
    expected_status: int
    expected_code: str | None
    expected_param: str | None
    expected_message: str | None
    expect_models: bool = False


FROZEN_SCHEDULE = (
    ApiCase(
        "models-valid",
        "GET",
        "/v1/models",
        b"",
        "valid_bearer",
        200,
        None,
        None,
        None,
        True,
    ),
    ApiCase(
        "models-missing-auth",
        "GET",
        "/v1/models",
        b"",
        "missing",
        401,
        "invalid_api_key",
        None,
        INVALID_KEY_MESSAGE,
    ),
    ApiCase(
        "models-invalid-auth",
        "GET",
        "/v1/models",
        b"",
        "invalid_bearer",
        401,
        "invalid_api_key",
        None,
        INVALID_KEY_MESSAGE,
    ),
    ApiCase(
        "models-query",
        "GET",
        "/v1/models?x=1",
        b"",
        "valid_bearer",
        400,
        "invalid_request_error",
        None,
        QUERY_MESSAGE,
    ),
    ApiCase(
        "chat-malformed-missing-auth",
        "POST",
        "/v1/chat/completions",
        MALFORMED_BODY,
        "missing",
        401,
        "invalid_api_key",
        None,
        INVALID_KEY_MESSAGE,
    ),
    ApiCase(
        "chat-invalid-auth",
        "POST",
        "/v1/chat/completions",
        CANONICAL_BODY,
        "invalid_bearer",
        401,
        "invalid_api_key",
        None,
        INVALID_KEY_MESSAGE,
    ),
    ApiCase(
        "chat-malformed-valid-auth",
        "POST",
        "/v1/chat/completions",
        MALFORMED_BODY,
        "valid_bearer",
        400,
        "invalid_request_error",
        None,
        INVALID_JSON_MESSAGE,
    ),
    ApiCase(
        "chat-duplicate-key",
        "POST",
        "/v1/chat/completions",
        DUPLICATE_KEY_BODY,
        "valid_bearer",
        400,
        "invalid_request_error",
        None,
        INVALID_JSON_MESSAGE,
    ),
    ApiCase(
        "chat-unsupported-n",
        "POST",
        "/v1/chat/completions",
        UNSUPPORTED_N_BODY,
        "valid_bearer",
        400,
        "unsupported_parameter",
        "n",
        UNSUPPORTED_MESSAGE,
    ),
    ApiCase(
        "chat-missing-model",
        "POST",
        "/v1/chat/completions",
        MISSING_MODEL_BODY,
        "valid_bearer",
        404,
        "model_not_found",
        "model",
        MODEL_NOT_FOUND_MESSAGE,
    ),
)


def validate_schedule(schedule: Sequence[ApiCase]) -> tuple[ApiCase, ...]:
    frozen = tuple(schedule)
    if frozen != FROZEN_SCHEDULE or len({case.case_id for case in frozen}) != 10:
        fail("API contract case schedule differs")
    for index, case in enumerate(frozen, start=1):
        request_key = f"api-contract-{index:02d}-{case.case_id}"
        if REQUEST_KEY_RE.fullmatch(request_key) is None:
            fail("API contract request key syntax differs")
    return frozen


@dataclasses.dataclass(frozen=True)
class HttpChunk:
    index: int
    raw: bytes
    observed_monotonic_ns: int


@dataclasses.dataclass(frozen=True)
class HttpObservation:
    request_key: str
    method: str
    target: str
    authorization_mode: str
    request_body: bytes
    connect_completed_monotonic_ns: int
    write_started_monotonic_ns: int
    last_body_byte_sent_monotonic_ns: int
    status: int
    response_headers: tuple[tuple[str, str], ...]
    response_started_monotonic_ns: int
    chunks: tuple[HttpChunk, ...]
    outcome: str
    response_body: bytes
    response_end_monotonic_ns: int


def _decode_base64(value: Any, expected_size: int, label: str) -> bytes:
    if type(value) is not str:
        fail(f"{label} base64 is not text")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        fail(f"{label} base64 is invalid")
    if len(raw) != expected_size or base64.b64encode(raw).decode("ascii") != value:
        fail(f"{label} base64 size or canonical form differs")
    return raw


def parse_http_events(
    case: ApiCase,
    request_key: str,
    events: Sequence[dict[str, Any]],
    *,
    previous_response_end_ns: int = -1,
) -> HttpObservation:
    if not events or len(events) > MAX_HTTP_EVENTS:
        fail("HTTP event sequence size differs")
    saw_request = False
    saw_start = False
    saw_end = False
    connect_ns = -1
    write_ns = -1
    sent_ns = -1
    status = -1
    response_start_ns = -1
    response_end_ns = -1
    headers: tuple[tuple[str, str], ...] = ()
    chunks: list[HttpChunk] = []
    digest = hashlib.sha256()
    total = 0
    last_timestamp = -1
    outcome = ""
    request_body = b""

    for position, event in enumerate(events):
        if event.get("schema_version") != HTTP_EVENT_SCHEMA:
            fail("HTTP event schema differs")
        name = event.get("event")
        if name == "http_request":
            value = exact_keys(
                event,
                {
                    "schema_version",
                    "event",
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
            if position != 0 or saw_request or value["request_key"] != request_key:
                fail("HTTP request event order or key differs")
            if value["method"] != case.method or value["target"] != case.target:
                fail("HTTP request method or target differs")
            request_headers = exact_keys(
                value["headers"],
                {"content_type", "content_length", "authorization_mode"},
                "HTTP request headers",
            )
            body_bytes = integer(value["body_bytes"], "HTTP request body bytes")
            if (
                request_headers["content_type"] != "application/json"
                or request_headers["content_length"] != len(case.body)
                or request_headers["authorization_mode"] != case.authorization_mode
                or body_bytes != len(case.body)
            ):
                fail("HTTP request headers or body length differ")
            request_body = _decode_base64(
                value["body_base64"], body_bytes, "HTTP request body"
            )
            if (
                request_body != case.body
                or sha256_value(value["body_sha256"], "HTTP request body SHA-256")
                != hashlib.sha256(case.body).hexdigest()
            ):
                fail("HTTP request raw body or hash differs")
            connect_ns = integer(
                value["connect_completed_monotonic_ns"], "HTTP connect timestamp"
            )
            write_ns = integer(
                value["write_started_monotonic_ns"], "HTTP write timestamp"
            )
            sent_ns = integer(
                value["last_body_byte_sent_monotonic_ns"], "HTTP send timestamp"
            )
            if not (previous_response_end_ns <= connect_ns <= write_ns <= sent_ns):
                fail("HTTP request timestamps overlap or regress")
            last_timestamp = sent_ns
            saw_request = True
        elif name == "http_response_start":
            value = exact_keys(
                event,
                {
                    "schema_version",
                    "event",
                    "request_key",
                    "status",
                    "headers",
                    "observed_monotonic_ns",
                },
                "HTTP response start event",
            )
            if not saw_request or saw_start or value["request_key"] != request_key:
                fail("HTTP response start order or key differs")
            status = integer(value["status"], "HTTP response status", minimum=100)
            if status > 599 or type(value["headers"]) is not list:
                fail("HTTP response status or header array differs")
            parsed_headers: list[tuple[str, str]] = []
            for pair in value["headers"]:
                if (
                    type(pair) is not list
                    or len(pair) != 2
                    or type(pair[0]) is not str
                    or type(pair[1]) is not str
                ):
                    fail("HTTP response raw header pair differs")
                parsed_headers.append((pair[0], pair[1]))
            headers = tuple(parsed_headers)
            response_start_ns = integer(
                value["observed_monotonic_ns"], "HTTP response start timestamp"
            )
            if response_start_ns < last_timestamp:
                fail("HTTP response starts before the request send boundary")
            last_timestamp = response_start_ns
            saw_start = True
        elif name == "http_body_chunk":
            value = exact_keys(
                event,
                {
                    "schema_version",
                    "event",
                    "request_key",
                    "chunk_index",
                    "body_base64",
                    "body_sha256",
                    "body_bytes",
                    "observed_monotonic_ns",
                },
                "HTTP body chunk event",
            )
            chunk_index = integer(value["chunk_index"], "HTTP chunk index")
            chunk_bytes = integer(value["body_bytes"], "HTTP chunk bytes", minimum=1)
            if (
                not saw_start
                or saw_end
                or value["request_key"] != request_key
                or chunk_index != len(chunks)
            ):
                fail("HTTP body chunk order or correlation differs")
            raw = _decode_base64(value["body_base64"], chunk_bytes, "HTTP body chunk")
            if (
                sha256_value(value["body_sha256"], "HTTP chunk SHA-256")
                != hashlib.sha256(raw).hexdigest()
            ):
                fail("HTTP body chunk hash differs")
            observed_ns = integer(
                value["observed_monotonic_ns"], "HTTP chunk timestamp"
            )
            if observed_ns < last_timestamp:
                fail("HTTP chunk timestamp regressed")
            total += len(raw)
            if total > MAX_RESPONSE_BYTES:
                fail("HTTP response body exceeds its bound")
            digest.update(raw)
            chunks.append(HttpChunk(chunk_index, raw, observed_ns))
            last_timestamp = observed_ns
        elif name == "http_response_end":
            value = exact_keys(
                event,
                {
                    "schema_version",
                    "event",
                    "request_key",
                    "outcome",
                    "error",
                    "body_bytes",
                    "body_sha256",
                    "observed_monotonic_ns",
                },
                "HTTP response end event",
            )
            if (
                not saw_request
                or saw_end
                or position != len(events) - 1
                or value["request_key"] != request_key
            ):
                fail("HTTP response end order or correlation differs")
            outcome = value["outcome"]
            if outcome != "eof" or value["error"] is not None:
                fail("API contract HTTP response did not terminate at EOF")
            if (
                value["body_bytes"] != total
                or sha256_value(value["body_sha256"], "HTTP response body SHA-256")
                != digest.hexdigest()
            ):
                fail("HTTP response aggregate bytes or hash differ")
            response_end_ns = integer(
                value["observed_monotonic_ns"], "HTTP response end timestamp"
            )
            if response_end_ns < last_timestamp:
                fail("HTTP response end timestamp regressed")
            saw_end = True
        else:
            fail("HTTP evidence client emitted an unexpected event")
    if not saw_request or not saw_start or not saw_end or not chunks:
        fail("HTTP event sequence is incomplete")
    body = b"".join(chunk.raw for chunk in chunks)
    return HttpObservation(
        request_key=request_key,
        method=case.method,
        target=case.target,
        authorization_mode=case.authorization_mode,
        request_body=request_body,
        connect_completed_monotonic_ns=connect_ns,
        write_started_monotonic_ns=write_ns,
        last_body_byte_sent_monotonic_ns=sent_ns,
        status=status,
        response_headers=headers,
        response_started_monotonic_ns=response_start_ns,
        chunks=tuple(chunks),
        outcome=outcome,
        response_body=body,
        response_end_monotonic_ns=response_end_ns,
    )


def _response_header_values(headers: Sequence[tuple[str, str]], name: str) -> list[str]:
    return [value for key, value in headers if key.lower() == name.lower()]


def validate_case_observation(
    case: ApiCase, observation: HttpObservation, case_index: int
) -> dict[str, Any]:
    expected_key = f"api-contract-{case_index:02d}-{case.case_id}"
    if (
        observation.request_key != expected_key
        or observation.method != case.method
        or observation.target != case.target
        or observation.authorization_mode != case.authorization_mode
        or observation.request_body != case.body
        or observation.status != case.expected_status
        or observation.outcome != "eof"
    ):
        fail("API contract HTTP observation differs from its case")
    if not (
        observation.connect_completed_monotonic_ns
        <= observation.write_started_monotonic_ns
        <= observation.last_body_byte_sent_monotonic_ns
        <= observation.response_started_monotonic_ns
        <= observation.response_end_monotonic_ns
    ):
        fail("API contract observation timestamps regress")
    if observation.response_body != b"".join(chunk.raw for chunk in observation.chunks):
        fail("API contract response chunks differ from the aggregate body")

    content_types = _response_header_values(
        observation.response_headers, "Content-Type"
    )
    if content_types != ["application/json"]:
        fail("API contract response Content-Type differs")
    authenticate = _response_header_values(
        observation.response_headers, "WWW-Authenticate"
    )
    expected_authenticate = ["Bearer"] if case.expected_status == 401 else []
    if authenticate != expected_authenticate:
        fail("API contract WWW-Authenticate header differs")
    content_lengths = _response_header_values(
        observation.response_headers, "Content-Length"
    )
    if content_lengths != [str(len(observation.response_body))]:
        fail("API contract response Content-Length differs")
    if _response_header_values(observation.response_headers, "Retry-After"):
        fail("non-busy API contract response contains Retry-After")
    if _response_header_values(observation.response_headers, "Transfer-Encoding"):
        fail("API contract response unexpectedly uses Transfer-Encoding")

    value = strict_json_object(observation.response_body, "API contract response body")
    error_summary: dict[str, Any] | None = None
    if case.expect_models:
        expected_models = {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "ullm"}],
        }
        if value != expected_models:
            fail("API contract model list differs")
    else:
        envelope = exact_keys(value, {"error"}, "API error envelope")
        error = exact_keys(
            envelope["error"], {"message", "type", "param", "code"}, "API error"
        )
        message = text(error["message"], "API error message", maximum=4096)
        if (
            error["type"] != "invalid_request_error"
            or error["code"] != case.expected_code
            or error["param"] != case.expected_param
            or message != case.expected_message
        ):
            fail("API error message, type, code, or param differs")
        message_raw = message.encode("utf-8")
        error_summary = {
            "type": error["type"],
            "code": error["code"],
            "param": error["param"],
            "message_utf8_bytes": len(message_raw),
            "message_sha256": hashlib.sha256(message_raw).hexdigest(),
        }
    return {
        "case_index": case_index,
        "case_id": case.case_id,
        "method": case.method,
        "target": case.target,
        "authorization_mode": case.authorization_mode,
        "request_body_bytes": len(case.body),
        "request_body_sha256": hashlib.sha256(case.body).hexdigest(),
        "connect_completed_monotonic_ns": observation.connect_completed_monotonic_ns,
        "write_started_monotonic_ns": observation.write_started_monotonic_ns,
        "last_body_byte_sent_monotonic_ns": observation.last_body_byte_sent_monotonic_ns,
        "status": observation.status,
        "response_started_monotonic_ns": observation.response_started_monotonic_ns,
        "response_end_monotonic_ns": observation.response_end_monotonic_ns,
        "content_type": content_types[0],
        "content_length": int(content_lengths[0], 10),
        "www_authenticate": authenticate,
        "response_body_bytes": len(observation.response_body),
        "response_body_sha256": hashlib.sha256(observation.response_body).hexdigest(),
        "error": error_summary,
    }


class GateRuntime(Protocol):
    def require_identity(self) -> None: ...

    def request(self, case: ApiCase, request_key: str) -> HttpObservation: ...

    def quiet(self, label: str) -> None: ...


class ApiContractRunner:
    def __init__(self, runtime: GateRuntime):
        self.runtime = runtime
        self.active = 0
        self.max_active = 0
        self.last_response_end_ns = -1

    def run(
        self, schedule: Sequence[ApiCase] = FROZEN_SCHEDULE
    ) -> list[dict[str, Any]]:
        cases = validate_schedule(schedule)
        results: list[dict[str, Any]] = []
        self.runtime.require_identity()
        for case_index, case in enumerate(cases, start=1):
            if self.active != 0:
                fail("API contract request overlap was detected")
            request_key = f"api-contract-{case_index:02d}-{case.case_id}"
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                observation = self.runtime.request(case, request_key)
            finally:
                self.active -= 1
            if observation.connect_completed_monotonic_ns < self.last_response_end_ns:
                fail("API contract request began before the prior response ended")
            results.append(validate_case_observation(case, observation, case_index))
            self.last_response_end_ns = observation.response_end_monotonic_ns
            self.runtime.quiet(case.case_id)
            self.runtime.require_identity()
        if self.max_active != 1 or len(results) != 10:
            fail("API contract active or request count differs")
        return results


def build_request_command(case: ApiCase, request_key: str) -> dict[str, Any]:
    if REQUEST_KEY_RE.fullmatch(request_key) is None:
        fail("API contract request key differs")
    return {
        "schema_version": HTTP_COMMAND_SCHEMA,
        "command": "request",
        "request_key": request_key,
        "method": case.method,
        "target": case.target,
        "body_base64": base64.b64encode(case.body).decode("ascii"),
        "authorization_mode": case.authorization_mode,
        "close_on_first_nonempty_sse_content": False,
    }


class ApiEvidenceHttpClient:
    def __init__(self, command: Sequence[str], guard: Any, writer: Any):
        self.command = tuple(command)
        self.guard = guard
        self.writer = writer
        self.process: subprocess.Popen[bytes] | None = None
        self.reader: Any | None = None
        self.stderr: Any | None = None
        self.active = False
        self.request_count = 0
        self.max_active = 0
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
        self.reader = DIRECT.COL.BoundedLineReader(self.process.stdout.fileno())
        event = self._read_event(time.monotonic_ns() + 30_000_000_000)
        exact_keys(
            event,
            {"schema_version", "event", "observed_monotonic_ns"},
            "HTTP ready event",
        )
        if event["schema_version"] != HTTP_EVENT_SCHEMA or event["event"] != "ready":
            fail("HTTP evidence client ready event differs")
        integer(event["observed_monotonic_ns"], "HTTP ready timestamp")

    def run_case(self, case: ApiCase, request_key: str) -> HttpObservation:
        if self.active:
            fail("HTTP evidence client already has an active request")
        self.active = True
        self.max_active = max(self.max_active, 1)
        self.request_count += 1
        try:
            self.guard.reject(case.body, "API contract HTTP request body")
            self._write_command(build_request_command(case, request_key))
            events: list[dict[str, Any]] = []
            deadline_ns = time.monotonic_ns() + REQUEST_TIMEOUT_NS
            while len(events) < MAX_HTTP_EVENTS:
                event = self._read_event(deadline_ns)
                events.append(event)
                if event.get("event") == "http_response_end":
                    break
            else:
                fail("HTTP evidence response event count exceeds its bound")
            result = parse_http_events(
                case,
                request_key,
                events,
                previous_response_end_ns=self.last_response_end_ns,
            )
            self.last_response_end_ns = result.response_end_monotonic_ns
            return result
        finally:
            self.active = False

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        pending: BaseException | None = None
        try:
            if self.active or self.request_count != 10 or self.max_active != 1:
                fail("HTTP evidence client shutdown schedule differs")
            self._write_command(
                {"schema_version": HTTP_COMMAND_SCHEMA, "command": "shutdown"}
            )
            event = self._read_event(time.monotonic_ns() + 5_000_000_000)
            exact_keys(
                event,
                {"schema_version", "event", "observed_monotonic_ns"},
                "HTTP shutdown event",
            )
            if (
                event["schema_version"] != HTTP_EVENT_SCHEMA
                or event["event"] != "shutdown_complete"
            ):
                fail("HTTP shutdown acknowledgement differs")
            integer(event["observed_monotonic_ns"], "HTTP shutdown timestamp")
            if process.wait(timeout=5.0) != 0:
                fail("HTTP evidence client exited nonzero")
            if self.reader is None or process.stdout is None:
                fail("HTTP shutdown stream state is unavailable")
            if self.reader.buffer or os.read(process.stdout.fileno(), 1):
                fail("HTTP evidence client emitted trailing stdout")
        except BaseException as error:
            pending = error
            if process.poll() is None:
                DIRECT.COL.terminate_process_group(process)
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
            DIRECT.COL.terminate_process_group(process)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self._check_stderr(require_empty=False)
        self.process = None

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
        return cast(dict[str, Any], DIRECT.COL.strict_json_object(raw, "HTTP event"))

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


class ProductionRuntime:
    def __init__(
        self,
        identity: Any,
        http: ApiEvidenceHttpClient,
        observer: Any,
        journal: Any,
        quiet_writer: Any,
    ):
        self.identity = identity
        self.http = http
        self.observer = observer
        self.journal = journal
        self.quiet_writer = quiet_writer
        self.journal_index = 0
        self.quiet_sequence = 0

    def require_identity(self) -> None:
        DIRECT.require_service_identity(self.identity)

    def request(self, case: ApiCase, request_key: str) -> HttpObservation:
        return self.http.run_case(case, request_key)

    def quiet(self, label: str) -> None:
        time.sleep(QUIET_DRAIN_NS / 1_000_000_000)
        self.observer.require_empty()
        added = self._poll_journal_lifecycle()
        time.sleep(POST_POLL_DRAIN_NS / 1_000_000_000)
        self.observer.require_empty()
        added += self._poll_journal_lifecycle()
        self._record_quiet(label, added, True)

    def post_observer_close(self, label: str) -> None:
        time.sleep(QUIET_DRAIN_NS / 1_000_000_000)
        added = self._poll_journal_lifecycle()
        self._record_quiet(label, added, False)

    def _poll_journal_lifecycle(self) -> int:
        self.journal.poll()
        records = self.journal.records[self.journal_index :]
        self.journal_index = len(self.journal.records)
        for raw in records:
            value = DIRECT.COL.strict_json_object(raw, "API contract service journal")
            event = DIRECT.COL.decode_lifecycle_message(value["MESSAGE"])
            if event is not None:
                fail("non-GPU API contract produced a worker lifecycle admission")
        return len(records)

    def _record_quiet(self, label: str, added: int, observer_open: bool) -> None:
        cursor = self.journal.source.cursor
        self.quiet_writer.write(
            compact_json(
                {
                    "schema_version": GATE_SCHEMA,
                    "record_type": "lifecycle_quiet_check",
                    "sequence": self.quiet_sequence,
                    "label": label,
                    "checked_monotonic_ns": time.monotonic_ns(),
                    "observer_open": observer_open,
                    "observer_event_count": 0,
                    "new_journal_record_count": added,
                    "journal_record_count": self.journal_index,
                    "journal_cursor": cursor,
                }
            ),
            "lifecycle quiet evidence",
        )
        self.quiet_sequence += 1


def _read_regular(path: Path, label: str, maximum: int) -> bytes:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
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
        if len(raw) != metadata.st_size:
            fail(f"{label} changed while it was read")
        return raw
    except GateError:
        raise
    except OSError:
        fail(f"failed to read {label}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def write_json_file(path: Path, value: dict[str, Any], guard: Any) -> None:
    write_file(path, compact_json(value) + b"\n", guard, path.name)


def write_file(path: Path, raw: bytes, guard: Any, label: str) -> None:
    guard.reject(raw, label)
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
                fail("atomic evidence write was short")
            offset += written
        os.fsync(descriptor)
    except GateError:
        raise
    except OSError:
        fail("failed to write an evidence artifact")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def verify_snapshot(
    path: Path, label: str, raw: bytes, identity: tuple[int, ...]
) -> None:
    current, current_identity = _snapshot(path, label, max(1, len(raw)))
    if current != raw or current_identity != identity:
        fail("a snapshotted API contract input changed")


def verify_writer(writer: Any) -> None:
    raw = _read_regular(writer.path, f"sealed {writer.path.name}", writer.maximum_bytes)
    if (
        len(raw) != writer.bytes_written
        or raw.count(b"\n") != writer.lines_written
        or hashlib.sha256(raw).hexdigest() != writer.digest.hexdigest()
    ):
        fail("a sealed raw API contract artifact changed")


def write_sha256sums(stage: Path, artifact_names: Sequence[str], guard: Any) -> bytes:
    names = sorted(artifact_names, key=lambda item: item.encode("utf-8"))
    if len(names) != len(set(names)) or "SHA256SUMS" in names:
        fail("checksum input path set differs")
    lines: list[str] = []
    for name in names:
        if "/" in name or name in {"", ".", ".."}:
            fail("checksum artifact name syntax differs")
        raw = _read_regular(
            stage / name, f"checksum input {name}", DIRECT.MAX_RAW_BYTES
        )
        guard.reject(raw, f"checksum input {name}")
        lines.append(f"{hashlib.sha256(raw).hexdigest()}  {name}\n")
    document = "".join(lines).encode("ascii")
    write_file(stage / "SHA256SUMS", document, guard, "SHA256SUMS")
    return document


def verify_sha256sums(
    stage: Path, artifact_names: Sequence[str], expected_document: bytes
) -> None:
    actual = _read_regular(
        stage / "SHA256SUMS", "sealed SHA256SUMS", DIRECT.MAX_RAW_BYTES
    )
    if actual != expected_document:
        fail("SHA256SUMS changed before publication")
    lines = actual.decode("ascii", errors="strict").splitlines()
    names = sorted(artifact_names, key=lambda item: item.encode("utf-8"))
    if len(lines) != len(names):
        fail("SHA256SUMS entry count differs")
    for line, name in zip(lines, names):
        expected_digest = hashlib.sha256(
            _read_regular(
                stage / name, f"verified checksum input {name}", DIRECT.MAX_RAW_BYTES
            )
        ).hexdigest()
        if line != f"{expected_digest}  {name}":
            fail("SHA256SUMS artifact digest or ordering differs")


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
    )


@dataclasses.dataclass(frozen=True)
class LockedArtifact:
    name: str
    descriptor: int
    identity: tuple[int, ...]
    sha256: str


class LockedStage:
    """Pin every final artifact inode across the exclusive directory rename."""

    def __init__(self, stage: Path, artifact_names: Sequence[str], guard: Any):
        self.stage = stage
        self.names = tuple(
            sorted(artifact_names, key=lambda item: item.encode("utf-8"))
        )
        self.guard = guard
        self.directory_descriptor = -1
        self.directory_identity: tuple[int, ...] | None = None
        self.artifacts: list[LockedArtifact] = []

    def open(self) -> None:
        if self.directory_descriptor >= 0:
            fail("locked output stage is already open")
        try:
            os.chmod(self.stage, 0o500)
            self.directory_descriptor = os.open(
                self.stage,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            directory_metadata = os.fstat(self.directory_descriptor)
            if (
                not stat.S_ISDIR(directory_metadata.st_mode)
                or stat.S_IMODE(directory_metadata.st_mode) != 0o500
                or tuple(sorted(os.listdir(self.directory_descriptor))) != self.names
            ):
                fail("locked output stage directory differs")
            self.directory_identity = _directory_identity(directory_metadata)
            for name in self.names:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=self.directory_descriptor,
                )
                try:
                    before = os.fstat(descriptor)
                    if (
                        not stat.S_ISREG(before.st_mode)
                        or stat.S_IMODE(before.st_mode) != 0o600
                        or before.st_nlink != 1
                        or before.st_uid != os.geteuid()
                        or before.st_size > DIRECT.MAX_RAW_BYTES
                    ):
                        fail("locked output artifact identity or mode differs")
                    digest = self._digest_descriptor(descriptor, name)
                    after = os.fstat(descriptor)
                    if _stat_identity(before) != _stat_identity(after):
                        fail("locked output artifact changed during hashing")
                    self.artifacts.append(
                        LockedArtifact(name, descriptor, _stat_identity(before), digest)
                    )
                    descriptor = -1
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
            os.fsync(self.directory_descriptor)
        except GateError:
            self.close()
            raise
        except OSError:
            self.close()
            fail("failed to lock the final output stage")
        except BaseException:
            self.close()
            raise

    def verify_published(self, final_path: Path) -> None:
        if self.directory_descriptor < 0 or self.directory_identity is None:
            fail("locked output stage is not open")
        final_directory = -1
        try:
            if (
                _directory_identity(os.fstat(self.directory_descriptor))
                != self.directory_identity
            ):
                fail("locked output directory changed across publication")
            for artifact in self.artifacts:
                if _stat_identity(os.fstat(artifact.descriptor)) != artifact.identity:
                    fail("locked output artifact changed across publication")
            final_directory = os.open(
                final_path,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            if (
                _directory_identity(os.fstat(final_directory))
                != self.directory_identity
                or tuple(sorted(os.listdir(final_directory))) != self.names
            ):
                fail("published output directory differs from its locked stage")
            for artifact in self.artifacts:
                descriptor = os.open(
                    artifact.name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=final_directory,
                )
                try:
                    if (
                        _stat_identity(os.fstat(descriptor)) != artifact.identity
                        or self._digest_descriptor(descriptor, artifact.name)
                        != artifact.sha256
                    ):
                        fail("published output artifact differs from its locked inode")
                finally:
                    os.close(descriptor)
        except GateError:
            raise
        except OSError:
            fail("failed to verify the published output through directory FDs")
        finally:
            if final_directory >= 0:
                os.close(final_directory)

    def close(self) -> None:
        for artifact in self.artifacts:
            try:
                os.close(artifact.descriptor)
            except OSError:
                pass
        self.artifacts.clear()
        if self.directory_descriptor >= 0:
            try:
                os.close(self.directory_descriptor)
            except OSError:
                pass
            self.directory_descriptor = -1
        if self.stage.exists():
            try:
                os.chmod(self.stage, 0o700)
            except OSError:
                pass

    def _digest_descriptor(self, descriptor: int, name: str) -> str:
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            scanner = self.guard.scanner(f"locked output artifact {name}")
            while chunk := os.read(descriptor, 64 * 1024):
                scanner.feed(chunk)
                digest.update(chunk)
            os.lseek(descriptor, 0, os.SEEK_SET)
            return digest.hexdigest()
        except OSError:
            fail("failed to hash a locked output artifact")


def publish_locked(output: Any, lock: LockedStage) -> None:
    renamed = False
    try:
        DIRECT.rename_noreplace(output.stage, output.final_path)
        renamed = True
        parent = os.open(
            output.final_path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        lock.verify_published(output.final_path)
        output.published = True
    except BaseException:
        if renamed:
            try:
                metadata = output.final_path.lstat()
                if (
                    lock.directory_identity is not None
                    and _directory_identity(metadata) == lock.directory_identity
                ):
                    os.chmod(output.final_path, 0o700)
                    DIRECT.shutil.rmtree(output.final_path)
                    parent = os.open(
                        output.final_path.parent,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    )
                    try:
                        os.fsync(parent)
                    finally:
                        os.close(parent)
            except OSError:
                pass
        raise


def build_input_manifest(
    gate_raw: bytes,
    client_raw: bytes,
    gateway_sources: dict[str, bytes],
) -> dict[str, Any]:
    if set(gateway_sources) != set(GATEWAY_SOURCE_RELATIVES):
        fail("gateway source manifest input set differs")
    inputs = [
        ("tools/run-sq8-api-contract-gate.py", gate_raw),
        ("tools/run-sq8-direct-cancel-gate.py", DIRECT_SUPPORT_RAW),
        ("tools/collect-sq8-openwebui-release.py", DIRECT.COLLECTOR_SUPPORT_RAW),
        ("tools/sq8-openwebui-http-client.py", client_raw),
        *((path, gateway_sources[path]) for path in GATEWAY_SOURCE_RELATIVES),
    ]
    return {
        "schema_version": GATE_SCHEMA,
        "record_type": "input_manifest",
        "inputs": [
            {
                "path": path,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            for path, raw in sorted(inputs, key=lambda item: item[0].encode("utf-8"))
        ],
        "request_bodies": [
            {
                "case_index": index,
                "case_id": case.case_id,
                "bytes": len(case.body),
                "sha256": hashlib.sha256(case.body).hexdigest(),
            }
            for index, case in enumerate(FROZEN_SCHEDULE, start=1)
        ],
    }


@dataclasses.dataclass(frozen=True)
class Arguments:
    output_dir: Path
    api_key_file: Path
    http_image_id: str
    docker_network_id: str


def execute(args: Arguments) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    gate_path = Path(__file__).resolve()
    direct_path = gate_path.with_name("run-sq8-direct-cancel-gate.py")
    collector_path = gate_path.with_name("collect-sq8-openwebui-release.py")
    client_path = gate_path.with_name("sq8-openwebui-http-client.py")
    gate_raw, gate_identity = _snapshot(
        gate_path, "API contract gate", MAX_SUPPORT_BYTES
    )
    client_raw, client_identity = _snapshot(
        client_path, "HTTP evidence client", MAX_SUPPORT_BYTES
    )
    gateway_sources: dict[str, bytes] = {}
    gateway_source_identities: dict[str, tuple[int, ...]] = {}
    for relative in GATEWAY_SOURCE_RELATIVES:
        raw, identity = _snapshot(
            repo_root / relative, f"gateway source {relative}", MAX_SUPPORT_BYTES
        )
        gateway_sources[relative] = raw
        gateway_source_identities[relative] = identity
    if hashlib.sha256(client_raw).hexdigest() != HTTP_CLIENT_SHA256:
        fail("HTTP evidence client implementation SHA-256 differs")

    output = DIRECT.AtomicRunDirectory(args.output_dir)
    writers: list[Any] = []
    snapshots: Any | None = None
    observer: Any | None = None
    http: ApiEvidenceHttpClient | None = None
    stage_lock: LockedStage | None = None
    published = False
    try:
        guard, credential_raw = DIRECT.COL.SecretGuard.snapshot_from_file(
            args.api_key_file
        )
        for label, raw in (
            ("API contract gate", gate_raw),
            ("direct cancellation support", DIRECT_SUPPORT_RAW),
            ("collector support", DIRECT.COLLECTOR_SUPPORT_RAW),
            ("HTTP evidence client", client_raw),
            *((f"gateway source {path}", raw) for path, raw in gateway_sources.items()),
        ):
            guard.reject(raw, label)

        DIRECT.validate_docker_identity(args.http_image_id, args.docker_network_id)
        identity = DIRECT.capture_service_identity()
        if (
            identity.uid != DIRECT.COL.HTTP_CLIENT_UID
            or identity.gid != DIRECT.COL.HTTP_CLIENT_GID
        ):
            fail("service identity differs from the fixed HTTP client user")
        DIRECT.require_ready(args.http_image_id)

        snapshots = DIRECT.COL.RuntimeSnapshots.create(client_raw, credential_raw)
        config = types.SimpleNamespace(
            identities={"openwebui": {"derived_image_id": args.http_image_id}}
        )
        http_command = DIRECT.COL.build_http_client_command(config, snapshots)
        http_writer = DIRECT.RawWriter(output.stage / "http-client.raw.jsonl", guard)
        journal_writer = DIRECT.RawWriter(
            output.stage / "service-journal.raw.jsonl", guard
        )
        quiet_writer = DIRECT.RawWriter(
            output.stage / "lifecycle-quiet.raw.jsonl", guard
        )
        writers.extend((http_writer, journal_writer, quiet_writer))

        journal = DIRECT.JournalCapture(
            identity.boot_id, identity.gateway_pid, journal_writer, guard
        )
        observer = DIRECT.COL.LifecycleObserver(
            OBSERVER_SOCKET,
            guard,
            expected_uid=identity.uid,
            expected_gid=identity.gid,
        )
        observer.open()
        journal.start()
        http = ApiEvidenceHttpClient(http_command, guard, http_writer)
        http.start()
        snapshots.unlink_credential()

        runtime = ProductionRuntime(identity, http, observer, journal, quiet_writer)
        runner = ApiContractRunner(runtime)
        results = runner.run()
        http.close()
        http = None
        runtime.quiet("http-client-shutdown")
        observer.require_empty()
        observer.close()
        observer = None
        runtime.post_observer_close("post-observer-close")

        DIRECT.require_service_identity(identity)
        DIRECT.require_ready(args.http_image_id)
        DIRECT.validate_docker_identity(args.http_image_id, args.docker_network_id)
        runtime.post_observer_close("final-readiness-and-identity")
        for writer in writers:
            writer.close()

        input_manifest = build_input_manifest(gate_raw, client_raw, gateway_sources)
        summary = {
            "schema_version": GATE_SCHEMA,
            "record_type": "summary",
            "model_id": MODEL_ID,
            "request_count": len(results),
            "max_active": runner.max_active,
            "service_identity": dataclasses.asdict(identity),
            "http_image_id": args.http_image_id,
            "docker_network_name": DIRECT.HTTP_NETWORK_NAME,
            "docker_network_id": args.docker_network_id,
            "observer_socket": os.fspath(OBSERVER_SOCKET),
            "observer_event_count": 0,
            "lifecycle_event_count": 0,
            "quiet_check_count": runtime.quiet_sequence,
            "cases": results,
            "artifacts": {
                writer.path.name: {
                    "bytes": writer.bytes_written,
                    "lines": writer.lines_written,
                    "sha256": writer.digest.hexdigest(),
                }
                for writer in writers
            },
        }
        write_json_file(output.stage / "input-manifest.json", input_manifest, guard)
        write_json_file(output.stage / "summary.json", summary, guard)

        verify_snapshot(gate_path, "API contract gate", gate_raw, gate_identity)
        verify_snapshot(
            direct_path,
            "direct cancellation support",
            DIRECT_SUPPORT_RAW,
            DIRECT_SUPPORT_IDENTITY,
        )
        verify_snapshot(
            collector_path,
            "collector support",
            DIRECT.COLLECTOR_SUPPORT_RAW,
            DIRECT.COLLECTOR_SUPPORT_IDENTITY,
        )
        verify_snapshot(
            client_path,
            "HTTP evidence client",
            client_raw,
            client_identity,
        )
        for relative in GATEWAY_SOURCE_RELATIVES:
            verify_snapshot(
                repo_root / relative,
                f"gateway source {relative}",
                gateway_sources[relative],
                gateway_source_identities[relative],
            )
        for writer in writers:
            verify_writer(writer)
        if (
            _read_regular(
                output.stage / "input-manifest.json",
                "sealed input manifest",
                DIRECT.MAX_RAW_BYTES,
            )
            != compact_json(input_manifest) + b"\n"
        ):
            fail("input manifest changed before publication")
        if (
            _read_regular(
                output.stage / "summary.json", "sealed summary", DIRECT.MAX_RAW_BYTES
            )
            != compact_json(summary) + b"\n"
        ):
            fail("summary changed before publication")

        artifact_names = (
            "http-client.raw.jsonl",
            "input-manifest.json",
            "lifecycle-quiet.raw.jsonl",
            "service-journal.raw.jsonl",
            "summary.json",
        )
        if {path.name for path in output.stage.iterdir()} != set(artifact_names):
            fail("API contract staged artifact set differs before checksums")
        checksum_document = write_sha256sums(output.stage, artifact_names, guard)
        verify_sha256sums(output.stage, artifact_names, checksum_document)
        if {path.name for path in output.stage.iterdir()} != set(artifact_names) | {
            "SHA256SUMS"
        }:
            fail("API contract final staged artifact set differs")
        for path in output.stage.iterdir():
            guard.scan_file(path, f"staged API contract output {path.name}")

        verify_snapshot(gate_path, "API contract gate", gate_raw, gate_identity)
        verify_snapshot(
            direct_path,
            "direct cancellation support",
            DIRECT_SUPPORT_RAW,
            DIRECT_SUPPORT_IDENTITY,
        )
        verify_snapshot(
            collector_path,
            "collector support",
            DIRECT.COLLECTOR_SUPPORT_RAW,
            DIRECT.COLLECTOR_SUPPORT_IDENTITY,
        )
        verify_snapshot(
            client_path,
            "HTTP evidence client",
            client_raw,
            client_identity,
        )
        for relative in GATEWAY_SOURCE_RELATIVES:
            verify_snapshot(
                repo_root / relative,
                f"gateway source {relative}",
                gateway_sources[relative],
                gateway_source_identities[relative],
            )

        snapshots.close()
        snapshots = None
        stage_lock = LockedStage(output.stage, (*artifact_names, "SHA256SUMS"), guard)
        stage_lock.open()
        publish_locked(output, stage_lock)
        published = True
        return {
            "schema_version": GATE_SCHEMA,
            "output_dir": os.fspath(output.final_path),
            "request_count": len(results),
        }
    finally:
        if stage_lock is not None:
            stage_lock.close()
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
    except Exception:
        print("SQ8 API contract gate failed", file=sys.stderr)
        return 2
    print(compact_json(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
