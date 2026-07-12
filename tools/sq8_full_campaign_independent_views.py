#!/usr/bin/env python3
"""Independently reconstruct the non-metric full-campaign derived views.

This module intentionally does not import the producer or release validator.  It
accepts the validator's already structural ``SessionData`` result and rebuilds
the redacted views from its bounded raw projections.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import stat
import struct
import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast


API_RESULTS_SCHEMA = "ullm.sq8.openwebui_release.api_contract_results.v1"
CANCEL_RESULTS_SCHEMA = "ullm.sq8.openwebui_release.cancel_results.v1"
OPENWEBUI_SMOKE_SCHEMA = "ullm.sq8.openwebui_release.openwebui_smoke.v1"
SAMPLING_RESULTS_SCHEMA = "ullm.sq8.openwebui_release.sampling_results.v1"
SOAK_RESULTS_SCHEMA = "ullm.sq8.openwebui_release.soak_results.v1"

MODEL_ID = "ullm-qwen3-14b-sq8"
DIRECT_CANCEL_PHASES = (
    "after_started_before_progress",
    "prefill_after_128",
    "prefill_after_2048",
    "decode_after_first_content",
)
CANCEL_PHASES = DIRECT_CANCEL_PHASES + ("openwebui_stop_after_visible_content",)
SAMPLED_NORMAL_INDICES = tuple(range(5, 101, 5))

STOP_PROMPT = " ".join(
    (
        "Begin with STOP_STREAM_MARKER.",
        "Then write the integers from 1 through 1000, one per line.",
        "Do not summarize and do not stop early.",
    )
)
STOP_RECOVERY_PROMPT = (
    "For this new turn, reply with exactly STOP_RECOVERY_OK and nothing else."
)
FAILURE_PROMPT = " ".join(
    (
        "Begin with FAIL_STREAM_MARKER.",
        "Then write the integers from 1 through 1000, one per line.",
        "Do not summarize and do not stop early.",
    )
)
FAILURE_RECOVERY_PROMPT = (
    "For this new turn, reply with exactly FAILURE_RECOVERY_OK and nothing else."
)
FAULT_COMMAND = "signal.pidfd_send_signal"

STOP_SCREENSHOT = "browser/openwebui-stop-before.png"
FAILURE_SCREENSHOT = "browser/post-header-failure.png"
MAX_PNG_BYTES = 128 << 20
MAX_PNG_CHUNK_BYTES = 64 << 20
MAX_PNG_DECODED_BYTES = 512 << 20
COPY_CHUNK_BYTES = 1 << 20
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
STOP_SELECTOR = (
    '#message-input-container button:has(svg[viewBox="0 0 24 24"] '
    'path[d="M2.25 12c0-5.385 4.365-9.75 9.75-9.75s9.75 4.365 '
    "9.75 9.75-4.365 9.75-9.75 9.75S2.25 17.385 2.25 12zm6-2.438c0-.724.588-1.312 "
    "1.313-1.312h4.874c.725 0 1.313.588 1.313 1.313v4.874c0 .725-.588 "
    '1.313-1.313 1.313H9.564a1.312 1.312 0 01-1.313-1.313V9.564z"])'
)


class IndependentViewError(ValueError):
    """The retained raw evidence cannot reconstruct one exact derived view."""


def fail(message: str) -> NoReturn:
    raise IndependentViewError(message)


class SessionLike(Protocol):
    full_campaign_order: Any
    api_contract: Any
    http_results: Sequence[Any]
    http_requests: Mapping[str, Mapping[str, Any]]
    browser_actions: Sequence[Any]
    api_journal_observations: Sequence[Any]
    lifecycle_quiet_checks: Sequence[Any]
    fault_injection: Any
    traces: Mapping[str, Any]
    releases_by_phase: Mapping[str, Sequence[Mapping[str, Any]]]
    probes: Mapping[str, Mapping[str, Any]]


@dataclasses.dataclass(frozen=True)
class ScreenshotEvidence:
    file: str
    bytes: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class IndependentFrontViews:
    api_contract_results: dict[str, Any]
    sampling_results: dict[str, Any]
    cancel_results: dict[str, Any]
    openwebui_smoke: dict[str, Any]
    browser_soak_cases: list[dict[str, Any]]
    canonical_bytes: dict[str, bytes]


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail(f"{label} is not an integer in range")
    return value


def _text(value: Any, label: str, *, maximum: int = 512) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > maximum:
        fail(f"{label} is not one bounded non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} is not a lowercase SHA-256 digest")
    return value


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _attr(value: Any, name: str, label: str) -> Any:
    try:
        return getattr(value, name)
    except AttributeError:
        fail(f"{label} lacks {name}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail(f"{label} is not a mapping")
    return cast(Mapping[str, Any], value)


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        fail(f"{label} is not a sequence")
    return value


def _reject_passed(value: Any, label: str) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    visited = 0
    while pending:
        item, depth = pending.pop()
        visited += 1
        if depth > 128 or visited > 100_000:
            fail(f"{label} exceeds its structural bound")
        if isinstance(item, Mapping):
            if "passed" in item:
                fail(f"{label} contains forbidden passed data")
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            pending.extend((child, depth + 1) for child in item)


def canonical_json_bytes(
    value: Mapping[str, Any], *, forbidden_values: tuple[bytes, ...] = ()
) -> bytes:
    """Encode a redacted view as canonical ASCII JSON plus one LF."""

    if type(value) is not dict:
        fail("canonical view root is not an exact object")
    _reject_passed(value, "canonical view")
    for secret in forbidden_values:
        if type(secret) is not bytes or not secret:
            fail("canonical forbidden value is invalid")
    pending: list[tuple[Any, int]] = [(value, 0)]
    visited = 0
    while pending:
        item, depth = pending.pop()
        visited += 1
        if depth > 128 or visited > 100_000:
            fail("canonical view exceeds the semantic secret-scan bound")
        if isinstance(item, Mapping):
            for key, child in item.items():
                if type(key) is str:
                    try:
                        key_raw = key.encode("utf-8", errors="strict")
                    except UnicodeError:
                        fail("canonical view contains a non-UTF-8 object key")
                    if any(secret in key_raw for secret in forbidden_values):
                        fail("canonical view contains forbidden semantic cleartext")
                pending.append((child, depth + 1))
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            try:
                item_raw = item.encode("utf-8", errors="strict")
            except UnicodeError:
                fail("canonical view contains a non-UTF-8 string")
            if any(secret in item_raw for secret in forbidden_values):
                fail("canonical view contains forbidden semantic cleartext")
        elif type(item) in {bytes, bytearray} and any(
            secret in bytes(item) for secret in forbidden_values
        ):
            fail("canonical view contains forbidden semantic cleartext")
    try:
        raw = (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise IndependentViewError("canonical view cannot be encoded") from error
    if any(secret in raw for secret in forbidden_values):
        fail("canonical view contains forbidden cleartext")
    return raw


def _validate_session_ready(session: SessionLike) -> None:
    order = _attr(session, "full_campaign_order", "session")
    api = _attr(session, "api_contract", "session")
    if order is None or api is None:
        fail("session lacks complete full-campaign or API validation")
    phases = tuple(_sequence(_attr(order, "phases", "campaign order"), "phases"))
    if phases != (
        "preflight",
        "api_contract",
        "openwebui",
        "cancellation",
        "resource_normal",
        "post_header_failure",
        "resource_restart",
        "latency",
        "final",
    ):
        fail("session full-campaign phase order differs")
    if (
        tuple(
            _sequence(
                _attr(order, "cancellation_phases", "campaign order"),
                "cancellation phases",
            )
        )
        != CANCEL_PHASES
    ):
        fail("session cancellation classification differs")
    if (
        _integer(
            _attr(order, "openwebui_successful_requests", "campaign order"),
            "OpenWebUI successful request count",
        )
        != 21
    ):
        fail("session OpenWebUI cardinality differs")
    _validate_browser_action_set(session)


def _http_map(session: SessionLike) -> dict[tuple[str, str], Any]:
    result: dict[tuple[str, str], Any] = {}
    for item in _sequence(_attr(session, "http_results", "session"), "HTTP results"):
        key = (
            _text(_attr(item, "phase", "HTTP result"), "HTTP phase"),
            _text(_attr(item, "case_id", "HTTP result"), "HTTP case ID"),
        )
        if key in result:
            fail("HTTP phase/case identity is duplicated")
        result[key] = item
    return result


def _trace_map(session: SessionLike) -> dict[tuple[str, str], Any]:
    result: dict[tuple[str, str], Any] = {}
    reconstructed_releases: dict[str, list[Mapping[str, Any]]] = {}
    traces = _mapping(_attr(session, "traces", "session"), "session traces")
    for request_id, trace in traces.items():
        _text(request_id, "trace request ID", maximum=1024)
        key = (
            _text(_attr(trace, "phase", "trace"), "trace phase"),
            _text(_attr(trace, "case_id", "trace"), "trace case ID"),
        )
        if key in result:
            fail("lifecycle phase/case identity is duplicated")
        completion_id = _text(
            _attr(trace, "completion_id", "trace"),
            "trace completion ID",
            maximum=1024,
        )
        events = _sequence(_attr(trace, "events", "trace"), "trace events")
        if not events:
            fail("lifecycle trace is empty")
        for event in events:
            item = _mapping(event, "lifecycle event")
            if (
                item.get("request_id") != request_id
                or item.get("completion_id") != completion_id
            ):
                fail("lifecycle request/completion identity differs")
            if item.get("event") == "request_released":
                reconstructed_releases.setdefault(key[0], []).append(item)
        result[key] = trace
    release_index = _mapping(
        _attr(session, "releases_by_phase", "session"),
        "session releases by phase",
    )
    normalized_releases = {
        phase: [
            _mapping(item, "indexed lifecycle release")
            for item in _sequence(values, "indexed lifecycle releases")
        ]
        for phase, values in release_index.items()
    }
    if normalized_releases != reconstructed_releases:
        fail("lifecycle trace releases differ from the phase release index")
    return result


def _events(trace: Any) -> list[Mapping[str, Any]]:
    return [
        _mapping(item, "lifecycle event")
        for item in _sequence(_attr(trace, "events", "trace"), "trace events")
    ]


def _single_event(trace: Any, name: str) -> Mapping[str, Any]:
    found = [event for event in _events(trace) if event.get("event") == name]
    if len(found) != 1:
        fail(f"lifecycle {name} cardinality differs")
    return found[0]


def _completion_digest(trace: Any) -> str:
    completion_id = _text(
        _attr(trace, "completion_id", "trace"),
        "trace completion ID",
        maximum=1024,
    )
    return _hash_text(completion_id)


def _validate_sse_identity(http: Any, trace: Any, *, require_id: bool) -> None:
    sse = _attr(http, "sse", "HTTP result")
    if sse is None:
        fail("HTTP result lacks compact SSE evidence")
    items = _sequence(_attr(sse, "items", "SSE metadata"), "SSE items")
    observed: set[str] = set()
    for item in items:
        digest = _attr(item, "completion_id_sha256", "SSE item")
        byte_count = _attr(item, "completion_id_utf8_bytes", "SSE item")
        if digest is None:
            if byte_count is not None:
                fail("SSE completion ID byte/hash fields differ")
            continue
        _integer(byte_count, "SSE completion ID bytes", minimum=1)
        observed.add(_sha256(digest, "SSE completion ID SHA-256"))
    if require_id and not observed:
        fail("successful SSE response lacks a completion ID")
    if observed and observed != {_completion_digest(trace)}:
        fail("SSE and lifecycle completion IDs differ")


def _validate_success_sse(http: Any, trace: Any, *, outcome: str) -> None:
    if (
        _integer(_attr(http, "status", "HTTP result"), "HTTP status", 100) != 200
        or _attr(http, "outcome", "HTTP result") != "eof"
    ):
        fail("successful SSE HTTP status or outcome differs")
    _validate_sse_identity(http, trace, require_id=True)
    sse = _attr(http, "sse", "HTTP result")
    items = _sequence(_attr(sse, "items", "SSE metadata"), "SSE items")
    done = [item for item in items if _attr(item, "done", "SSE item") is True]
    if len(done) != 1 or items[-1] is not done[0]:
        fail("successful SSE [DONE] placement differs")
    finishes = [
        _attr(item, "finish_reason", "SSE item")
        for item in items
        if _attr(item, "finish_reason", "SSE item") is not None
    ]
    usage = [item for item in items if _attr(item, "usage_present", "SSE item") is True]
    if finishes != [outcome] or len(usage) != 1:
        fail("successful SSE finish or usage cardinality differs")
    if (
        _attr(usage[0], "usage_is_object", "SSE item") is not True
        or _integer(
            _attr(usage[0], "completion_tokens", "SSE item"),
            "SSE completion tokens",
        )
        != 2
    ):
        fail("successful SSE completion usage differs")


_API_SPECS = (
    (
        "models-valid",
        "GET",
        "/v1/models",
        b"",
        "valid_bearer",
        200,
        None,
        None,
        None,
    ),
    (
        "models-missing-auth",
        "GET",
        "/v1/models",
        b"",
        "missing",
        401,
        "invalid_api_key",
        None,
        "The supplied API key is invalid.",
    ),
    (
        "models-invalid-auth",
        "GET",
        "/v1/models",
        b"",
        "invalid_bearer",
        401,
        "invalid_api_key",
        None,
        "The supplied API key is invalid.",
    ),
    (
        "models-query",
        "GET",
        "/v1/models?x=1",
        b"",
        "valid_bearer",
        400,
        "invalid_request_error",
        None,
        "Query parameters are not supported.",
    ),
    (
        "chat-malformed-missing-auth",
        "POST",
        "/v1/chat/completions",
        b'{"broken":',
        "missing",
        401,
        "invalid_api_key",
        None,
        "The supplied API key is invalid.",
    ),
    (
        "chat-invalid-auth",
        "POST",
        "/v1/chat/completions",
        b'{"messages":[{"content":"API contract preflight","role":"user"}],"model":"ullm-qwen3-14b-sq8"}',
        "invalid_bearer",
        401,
        "invalid_api_key",
        None,
        "The supplied API key is invalid.",
    ),
    (
        "chat-malformed-valid-auth",
        "POST",
        "/v1/chat/completions",
        b'{"broken":',
        "valid_bearer",
        400,
        "invalid_request_error",
        None,
        "The request body is not valid JSON.",
    ),
    (
        "chat-duplicate-key",
        "POST",
        "/v1/chat/completions",
        b'{"model":"ullm-qwen3-14b-sq8","model":"ullm-qwen3-14b-sq8","messages":[{"role":"user","content":"API contract preflight"}]}',
        "valid_bearer",
        400,
        "invalid_request_error",
        None,
        "The request body is not valid JSON.",
    ),
    (
        "chat-unsupported-n",
        "POST",
        "/v1/chat/completions",
        b'{"messages":[{"content":"API contract preflight","role":"user"}],"model":"ullm-qwen3-14b-sq8","n":2}',
        "valid_bearer",
        400,
        "unsupported_parameter",
        "n",
        "The requested parameter is not supported.",
    ),
    (
        "chat-missing-model",
        "POST",
        "/v1/chat/completions",
        b'{"messages":[{"content":"API contract preflight","role":"user"}],"model":"missing"}',
        "valid_bearer",
        404,
        "model_not_found",
        "model",
        "The requested model does not exist.",
    ),
)


def _validate_api_quiet_checks(
    session: SessionLike, http: Mapping[tuple[str, str], Any]
) -> int:
    labels = [spec[0] for spec in _API_SPECS] + [
        "http-client-shutdown",
        "post-observer-close",
        "final-readiness-and-identity",
    ]
    checks = _sequence(
        _attr(session, "lifecycle_quiet_checks", "session"),
        "API lifecycle quiet checks",
    )
    if len(checks) != len(labels):
        fail("API lifecycle quiet-check cardinality differs")
    observations = _sequence(
        _attr(session, "api_journal_observations", "session"),
        "API journal observations",
    )
    expected_gateway_pid = _integer(
        _attr(
            _attr(session, "full_campaign_order", "session"),
            "normal_gateway_pid",
            "campaign order",
        ),
        "normal gateway PID",
        minimum=1,
    )
    observed_cursors: set[str] = set()
    prior_journal_monotonic = -1
    normalized_observations: list[tuple[str, int]] = []
    for index, observation in enumerate(observations):
        phase = _text(
            _attr(observation, "phase", "API journal observation"),
            "API journal observation phase",
        )
        case_id = _text(
            _attr(observation, "case_id", "API journal observation"),
            "API journal observation case",
        )
        observation_index = _integer(
            _attr(observation, "observation_index", "API journal observation"),
            "API journal observation index",
        )
        cursor = _text(
            _attr(observation, "journal_cursor", "API journal observation"),
            "API journal observation cursor",
            maximum=1024,
        )
        monotonic_usec = _integer(
            _attr(
                observation,
                "journal_monotonic_usec",
                "API journal observation",
            ),
            "API journal monotonic timestamp",
        )
        journal_pid = _integer(
            _attr(observation, "journal_pid", "API journal observation"),
            "API journal PID",
            minimum=1,
        )
        if (
            phase != "api_contract"
            or case_id != f"api-journal-{index + 1:02d}"
            or observation_index != index
            or cursor in observed_cursors
            or monotonic_usec < prior_journal_monotonic
            or journal_pid != expected_gateway_pid
        ):
            fail(
                "API journal observation identity, order, cursor, time, or PID differs"
            )
        _integer(
            _attr(observation, "message_utf8_bytes", "API journal observation"),
            "API journal message bytes",
        )
        _sha256(
            _attr(observation, "message_sha256", "API journal observation"),
            "API journal message SHA-256",
        )
        observed_cursors.add(cursor)
        prior_journal_monotonic = monotonic_usec
        normalized_observations.append((cursor, monotonic_usec))
    api_http = {
        case_id: value
        for (phase, case_id), value in http.items()
        if phase == "api_contract"
    }
    if set(api_http) != set(labels[: len(_API_SPECS)]):
        fail("API quiet checks lack their complete HTTP result set")
    final_response_end = max(
        _integer(
            _attr(value, "response_end_monotonic_ns", "API HTTP result"),
            "API response end",
        )
        for value in api_http.values()
    )
    prior_checked = -1
    prior_journal_count = 0
    prior_cursor: str | None = None
    for sequence, (check, expected_label) in enumerate(
        zip(checks, labels, strict=True)
    ):
        phase = _attr(check, "phase", "API quiet check")
        case_id = _attr(check, "case_id", "API quiet check")
        label = _attr(check, "label", "API quiet check")
        quiet_sequence = _integer(
            _attr(check, "quiet_sequence", "API quiet check"),
            "API quiet sequence",
        )
        checked = _integer(
            _attr(check, "checked_monotonic_ns", "API quiet check"),
            "API quiet timestamp",
        )
        observer_open = _attr(check, "observer_open", "API quiet check")
        observer_count = _integer(
            _attr(check, "observer_event_count", "API quiet check"),
            "API quiet observer count",
        )
        new_journal_count = _integer(
            _attr(check, "new_journal_record_count", "API quiet check"),
            "API quiet new journal count",
        )
        journal_count = _integer(
            _attr(check, "journal_record_count", "API quiet check"),
            "API quiet journal count",
        )
        cursor = _text(
            _attr(check, "journal_cursor", "API quiet check"),
            "API quiet journal cursor",
            maximum=1024,
        )
        if normalized_observations:
            count_invalid = (
                journal_count <= 0
                or journal_count > len(normalized_observations)
                or journal_count < prior_journal_count
                or new_journal_count != journal_count - prior_journal_count
            )
        else:
            count_invalid = (
                journal_count != 0
                or new_journal_count != 0
                or (prior_cursor is not None and cursor != prior_cursor)
            )
        if (
            phase != "api_contract"
            or case_id != expected_label
            or label != expected_label
            or quiet_sequence != sequence
            or observer_open is not (sequence <= 10)
            or observer_count != 0
            or checked < prior_checked
            or count_invalid
        ):
            fail("API quiet-check identity, order, observer, or count differs")
        boundary = (
            _integer(
                _attr(
                    api_http[expected_label],
                    "response_end_monotonic_ns",
                    "API HTTP result",
                ),
                "API response end",
            )
            if sequence < len(_API_SPECS)
            else final_response_end
        )
        if checked < boundary:
            fail("API quiet check precedes its completed HTTP boundary")
        if normalized_observations:
            bound_cursor, bound_monotonic_usec = normalized_observations[
                journal_count - 1
            ]
            if cursor != bound_cursor or checked < bound_monotonic_usec * 1000:
                fail("API quiet check differs from its journal observation boundary")
        if prior_cursor is not None and (
            (new_journal_count == 0 and cursor != prior_cursor)
            or (new_journal_count > 0 and cursor == prior_cursor)
        ):
            fail("API quiet-check cursor does not follow its journal count delta")
        prior_checked = checked
        prior_journal_count = journal_count
        prior_cursor = cursor
    if prior_journal_count != len(normalized_observations):
        fail("API quiet checks do not cover all journal observations")
    return len(checks)


def reconstruct_api_contract(session: SessionLike) -> dict[str, Any]:
    """Rebuild api-contract-results.json from compact HTTP/API validation data."""

    _validate_session_ready(session)
    result = _attr(session, "api_contract", "session")
    raw_cases = _sequence(_attr(result, "cases", "API result"), "API cases")
    case_ids = tuple(_sequence(_attr(result, "case_ids", "API result"), "API case IDs"))
    statuses = tuple(_sequence(_attr(result, "statuses", "API result"), "API statuses"))
    request_keys = tuple(
        _sequence(_attr(result, "request_keys", "API result"), "API request keys")
    )
    if len(raw_cases) != 10 or len(request_keys) != 10:
        fail("API contract cardinality differs")
    http = _http_map(session)
    requests = _mapping(
        _attr(session, "http_requests", "session"), "session HTTP requests"
    )
    projected: list[dict[str, Any]] = []
    for index, (raw, spec) in enumerate(zip(raw_cases, _API_SPECS, strict=True), 1):
        case = _mapping(raw, f"API case {index}")
        (
            case_id,
            method,
            target,
            body,
            authorization,
            expected_status,
            code,
            param,
            message,
        ) = spec
        key = f"api-contract-{index:02d}-{case_id}"
        item = http.get(("api_contract", case_id))
        request = requests.get(key)
        if item is None or request is None:
            fail("API compact HTTP correlation is absent")
        if (
            case_ids[index - 1] != case_id
            or statuses[index - 1] != expected_status
            or request_keys[index - 1] != key
            or case.get("case_index") != index
            or case.get("case_id") != case_id
            or case.get("method") != method
            or case.get("target") != target
            or case.get("authorization_mode") != authorization
            or case.get("status") != expected_status
            or case.get("request_body_bytes") != len(body)
            or case.get("request_body_sha256") != hashlib.sha256(body).hexdigest()
            or _attr(item, "request_key", "API HTTP result") != key
            or _attr(item, "method", "API HTTP result") != method
            or _attr(item, "target", "API HTTP result") != target
            or _attr(item, "request_index", "API HTTP result") != index
            or _attr(item, "status", "API HTTP result") != expected_status
            or _attr(item, "outcome", "API HTTP result") != "eof"
            or _attr(item, "request_body_bytes", "API HTTP result") != len(body)
            or _attr(item, "request_body_sha256", "API HTTP result")
            != hashlib.sha256(body).hexdigest()
            or request.get("response_chunk_count") != 1
            or request.get("authorization_mode") != authorization
        ):
            fail("API case identity, body, status, or one-chunk framing differs")
        response_bytes = _integer(
            case.get("response_body_bytes"), "API response bytes", minimum=1
        )
        response_sha = _sha256(case.get("response_body_sha256"), "API response SHA-256")
        if (
            _attr(item, "response_body_bytes", "API HTTP result") != response_bytes
            or _attr(item, "response_body_sha256", "API HTTP result") != response_sha
            or case.get("content_type") != "application/json"
            or case.get("content_length") != response_bytes
            or case.get("www_authenticate")
            != (["Bearer"] if expected_status == 401 else [])
            or _attr(item, "sse", "API HTTP result") is not None
        ):
            fail("API compact response size or digest differs")
        timing_fields = (
            "connect_completed_monotonic_ns",
            "write_started_monotonic_ns",
            "last_body_byte_sent_monotonic_ns",
            "response_started_monotonic_ns",
            "response_end_monotonic_ns",
        )
        timings = tuple(
            _integer(case.get(field), f"API {field}") for field in timing_fields
        )
        if timings != tuple(
            _integer(_attr(item, field, "API HTTP result"), f"API HTTP {field}")
            for field in timing_fields
        ) or not (timings[0] <= timings[1] <= timings[2] <= timings[3] <= timings[4]):
            fail("API compact request/response timing differs")
        error = case.get("error")
        projected_error: dict[str, Any] | None
        if message is None:
            if error is not None:
                fail("successful API case contains an error")
            projected_error = None
        else:
            error_map = _mapping(error, "API error summary")
            expected_error = {
                "type": "invalid_request_error",
                "code": code,
                "param": param,
                "message_utf8_bytes": len(message.encode("utf-8")),
                "message_sha256": _hash_text(message),
            }
            if dict(error_map) != expected_error:
                fail("API error summary differs from the frozen contract")
            projected_error = expected_error
        projected.append(
            {
                "case_index": index,
                "case_id": case_id,
                "status": expected_status,
                "request_body_bytes": len(body),
                "request_body_sha256": hashlib.sha256(body).hexdigest(),
                "response_body_bytes": response_bytes,
                "response_body_sha256": response_sha,
                "error": projected_error,
            }
        )
    quiet_check_count = _validate_api_quiet_checks(session, http)
    return {
        "schema_version": API_RESULTS_SCHEMA,
        "case_count": 10,
        "http_record_count": 40,
        "quiet_check_count": quiet_check_count,
        "cases": projected,
    }


def reconstruct_sampling(session: SessionLike) -> dict[str, Any]:
    """Rebuild the twenty sampled normal-resource request summaries."""

    _validate_session_ready(session)
    http = _http_map(session)
    traces = _trace_map(session)
    cases: list[dict[str, Any]] = []
    for index in SAMPLED_NORMAL_INDICES:
        case_id = f"normal-measured-{index:03d}"
        result = http.get(("resource_normal", case_id))
        trace = traces.get(("resource_normal", case_id))
        if result is None or trace is None:
            fail("sampled normal HTTP/lifecycle correlation is absent")
        if (
            _attr(result, "request_index", "sampling HTTP result") != index
            or _attr(result, "request_key", "sampling HTTP result") != f"p8f-{case_id}"
            or _attr(result, "method", "sampling HTTP result") != "POST"
            or _attr(result, "target", "sampling HTTP result") != "/v1/chat/completions"
        ):
            fail("sampled normal HTTP identity differs")
        _validate_success_sse(result, trace, outcome="length")
        release = _single_event(trace, "request_released")
        if (
            release.get("outcome") != "length"
            or release.get("completion_tokens") != 2
            or release.get("reset_complete") is not True
        ):
            fail("sampled normal lifecycle release differs")
        cases.append(
            {
                "request_index": index,
                "temperature": 0.6,
                "top_p": 0.95,
                "seed": index,
                "http_status": 200,
                "http_outcome": "eof",
                "release_outcome": "length",
                "completion_tokens": 2,
                "reset_complete": True,
            }
        )
    return {
        "schema_version": SAMPLING_RESULTS_SCHEMA,
        "sampled_request_count": 20,
        "sampled_normal_indices": list(SAMPLED_NORMAL_INDICES),
        "cases": cases,
    }


@dataclasses.dataclass(frozen=True)
class _ActionSpec:
    name: str
    selector: str | None
    enabled: bool | None
    text: str
    input_sha256: str | None
    screenshot: str | None = None


def _prompt_sha(prompt: str) -> str:
    return _hash_text(prompt)


def _normal_prompt(index: int) -> str:
    return f"Reply with exactly {_normal_marker(index)} and nothing else."


def _normal_marker(index: int) -> str:
    return "OPENWEBUI_SMOKE_OK" if index == 0 else f"OPENWEBUI_SOAK_OK_{index:02d}"


def _normal_specs(index: int) -> tuple[_ActionSpec, ...]:
    return (
        _ActionSpec("navigate", None, None, "none", "navigation"),
        _ActionSpec("select_model", "body", None, "none", _prompt_sha(MODEL_ID)),
        _ActionSpec(
            "submit_chat",
            "#chat-input",
            True,
            "none",
            _prompt_sha(_normal_prompt(index)),
        ),
        _ActionSpec(
            "wait_visible", ".chat-assistant", None, _normal_marker(index), None
        ),
        _ActionSpec("wait_ready", "#chat-input", True, _normal_marker(index), None),
    )


def _stop_specs() -> tuple[_ActionSpec, ...]:
    return (
        _ActionSpec("navigate", None, None, "none", "navigation"),
        _ActionSpec("select_model", "body", None, "none", _prompt_sha(MODEL_ID)),
        _ActionSpec(
            "submit_chat", "#chat-input", True, "none", _prompt_sha(STOP_PROMPT)
        ),
        _ActionSpec("wait_visible", ".chat-assistant", None, "required", None),
        _ActionSpec(
            "click_stop", STOP_SELECTOR, True, "required", None, STOP_SCREENSHOT
        ),
        _ActionSpec("wait_ready", "#chat-input", True, "required", None),
        _ActionSpec(
            "submit_chat",
            "#chat-input",
            True,
            "none",
            _prompt_sha(STOP_RECOVERY_PROMPT),
        ),
        _ActionSpec("wait_visible", ".chat-assistant", None, "STOP_RECOVERY_OK", None),
        _ActionSpec("wait_ready", "#chat-input", True, "STOP_RECOVERY_OK", None),
    )


def _failure_specs() -> tuple[_ActionSpec, ...]:
    return (
        _ActionSpec("navigate", None, None, "none", "navigation"),
        _ActionSpec("select_model", "body", None, "none", _prompt_sha(MODEL_ID)),
        _ActionSpec(
            "submit_chat",
            "#chat-input",
            True,
            "none",
            _prompt_sha(FAILURE_PROMPT),
        ),
        _ActionSpec("wait_visible", ".chat-assistant", None, "required", None),
        _ActionSpec(
            "wait_failed",
            ".chat-assistant",
            None,
            "required",
            None,
            FAILURE_SCREENSHOT,
        ),
        _ActionSpec("wait_ready", "#chat-input", True, "none", None),
        _ActionSpec(
            "submit_chat",
            "#chat-input",
            True,
            "none",
            _prompt_sha(FAILURE_RECOVERY_PROMPT),
        ),
        _ActionSpec(
            "wait_visible", ".chat-assistant", None, "FAILURE_RECOVERY_OK", None
        ),
        _ActionSpec("wait_ready", "#chat-input", True, "FAILURE_RECOVERY_OK", None),
    )


def _actions_for_browser_case(
    session: SessionLike,
    *,
    phase: str,
    browser_case: str,
    specs: Sequence[_ActionSpec],
    case_partition: Sequence[str],
) -> list[Any]:
    actions = [
        action
        for action in _sequence(
            _attr(session, "browser_actions", "session"), "browser actions"
        )
        if _attr(action, "phase", "browser action") == phase
        and _attr(action, "browser_case", "browser action") == browser_case
    ]
    if len(actions) != len(specs) or len(case_partition) != len(specs):
        fail("browser action cardinality differs")
    navigation_digest: str | None = None
    prior_completed = -1
    for index, (action, spec, expected_case) in enumerate(
        zip(actions, specs, case_partition, strict=True)
    ):
        started = _integer(
            _attr(action, "started_monotonic_ns", "browser action"),
            "browser action start",
        )
        completed = _integer(
            _attr(action, "completed_monotonic_ns", "browser action"),
            "browser action completion",
        )
        input_digest = _attr(action, "input_sha256", "browser action")
        if spec.input_sha256 == "navigation":
            navigation_digest = _sha256(input_digest, "navigation input SHA-256")
            expected_input: str | None = navigation_digest
        else:
            expected_input = spec.input_sha256
        if (
            _attr(action, "case_id", "browser action") != expected_case
            or _attr(action, "action_index", "browser action") != index
            or _attr(action, "action", "browser action") != spec.name
            or _attr(action, "selector", "browser action") != spec.selector
            or input_digest != expected_input
            or started < prior_completed
            or completed < started
            or _attr(action, "result_visible", "browser action") is not True
            or _attr(action, "result_enabled", "browser action") is not spec.enabled
            or _attr(action, "screenshot_file", "browser action") != spec.screenshot
        ):
            fail(
                "browser action identity, ordering, selector, input, or result differs"
            )
        text_bytes = _attr(action, "result_text_utf8_bytes", "browser action")
        text_sha = _attr(action, "result_text_sha256", "browser action")
        if spec.text == "required":
            _integer(text_bytes, "browser result text bytes", minimum=1)
            _sha256(text_sha, "browser result text SHA-256")
        elif spec.text == "none":
            if text_bytes is not None or text_sha is not None:
                fail("browser action unexpectedly carries text evidence")
        elif text_bytes != len(spec.text.encode("utf-8")) or text_sha != _hash_text(
            spec.text
        ):
            fail("browser action expected text evidence differs")
        screenshot_sha = _attr(action, "screenshot_sha256", "browser action")
        if spec.screenshot is None:
            if screenshot_sha is not None:
                fail("browser action unexpectedly carries screenshot evidence")
        else:
            _sha256(screenshot_sha, "browser screenshot SHA-256")
        prior_completed = completed
    return actions


def _validate_browser_action_set(session: SessionLike) -> None:
    expected: Counter[tuple[str, str]] = Counter(
        {
            ("openwebui", "openwebui_smoke"): 5,
            ("cancellation", "openwebui_stop_after_visible_content"): 9,
            ("post_header_failure", "post_header_worker_failure"): 9,
        }
    )
    expected.update(
        {("openwebui", f"openwebui_soak_chat_{index:02d}"): 5 for index in range(1, 21)}
    )
    observed: Counter[tuple[str, str]] = Counter()
    for action in _sequence(
        _attr(session, "browser_actions", "session"), "browser actions"
    ):
        observed[
            (
                _text(_attr(action, "phase", "browser action"), "browser phase"),
                _text(
                    _attr(action, "browser_case", "browser action"),
                    "browser case",
                ),
            )
        ] += 1
    if observed != expected:
        fail("complete browser action identity/cardinality set differs")


def _normal_navigation_digest(session: SessionLike) -> str:
    actions = _sequence(_attr(session, "browser_actions", "session"), "browser actions")
    navigation = [
        action
        for action in actions
        if _attr(action, "phase", "browser action") == "openwebui"
        and _attr(action, "action", "browser action") == "navigate"
    ]
    if len(navigation) != 21:
        fail("normal browser navigation cardinality differs")
    digests = {
        _sha256(
            _attr(action, "input_sha256", "browser navigation"),
            "browser navigation SHA-256",
        )
        for action in navigation
    }
    if len(digests) != 1:
        fail("normal browser navigation URL digests differ")
    return next(iter(digests))


class _PngValidator:
    """Incrementally verify framing, CRCs, and bounded scanline decoding."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.signature_seen = False
        self.chunk_index = 0
        self.chunk_type: bytes | None = None
        self.remaining = 0
        self.crc = 0
        self.ihdr = bytearray()
        self.saw_idat = False
        self.idat_closed = False
        self.saw_plte = False
        self.bit_depth: int | None = None
        self.color_type: int | None = None
        self.decompressor: Any | None = None
        self.expected_decoded_bytes = 0
        self.decoded_bytes = 0
        self.row_stride = 0
        self.ended = False

    def consume(self, raw: bytes) -> None:
        if self.ended and raw:
            fail("PNG carries bytes after IEND")
        self.buffer.extend(raw)
        while True:
            if not self.signature_seen:
                if len(self.buffer) < len(PNG_SIGNATURE):
                    return
                if bytes(self.buffer[:8]) != PNG_SIGNATURE:
                    fail("screenshot lacks the PNG signature")
                del self.buffer[:8]
                self.signature_seen = True
            if self.chunk_type is None:
                if len(self.buffer) < 8:
                    return
                length = int.from_bytes(self.buffer[:4], "big")
                chunk_type = bytes(self.buffer[4:8])
                del self.buffer[:8]
                if (
                    length > MAX_PNG_CHUNK_BYTES
                    or re.fullmatch(rb"[A-Za-z]{4}", chunk_type) is None
                ):
                    fail("PNG chunk framing differs")
                if self.chunk_index == 0 and (chunk_type != b"IHDR" or length != 13):
                    fail("PNG does not begin with one exact IHDR")
                if chunk_type == b"IHDR" and self.chunk_index != 0:
                    fail("PNG IHDR is duplicated")
                if chunk_type[0] & 0x20 == 0 and chunk_type not in {
                    b"IHDR",
                    b"PLTE",
                    b"IDAT",
                    b"IEND",
                }:
                    fail("PNG contains an unknown critical chunk")
                if chunk_type == b"PLTE":
                    if (
                        self.saw_plte
                        or self.saw_idat
                        or self.color_type in {0, 4}
                        or length == 0
                        or length > 768
                        or length % 3 != 0
                        or (
                            self.color_type == 3
                            and self.bit_depth is not None
                            and length // 3 > 1 << self.bit_depth
                        )
                    ):
                        fail("PNG PLTE cardinality, length, or ordering differs")
                if chunk_type == b"IDAT" and self.color_type == 3 and not self.saw_plte:
                    fail("indexed PNG lacks PLTE before IDAT")
                if chunk_type == b"IEND" and (length != 0 or not self.saw_idat):
                    fail("PNG IEND ordering differs")
                if self.saw_idat and chunk_type != b"IDAT":
                    self.idat_closed = True
                if chunk_type == b"IDAT" and self.idat_closed:
                    fail("PNG IDAT chunks are not consecutive")
                self.chunk_type = chunk_type
                self.remaining = length
                self.crc = zlib.crc32(chunk_type)
                self.ihdr.clear()
            if self.remaining:
                if not self.buffer:
                    return
                take = min(self.remaining, len(self.buffer))
                part = bytes(self.buffer[:take])
                del self.buffer[:take]
                self.remaining -= take
                self.crc = zlib.crc32(part, self.crc)
                if self.chunk_type == b"IHDR":
                    self.ihdr.extend(part)
                elif self.chunk_type == b"IDAT":
                    self._decode_idat(part)
                if self.remaining:
                    return
            if len(self.buffer) < 4:
                return
            stored_crc = int.from_bytes(self.buffer[:4], "big")
            del self.buffer[:4]
            if stored_crc != self.crc & 0xFFFF_FFFF:
                fail("PNG chunk CRC differs")
            assert self.chunk_type is not None
            if self.chunk_type == b"IHDR":
                self._validate_ihdr(bytes(self.ihdr))
            elif self.chunk_type == b"PLTE":
                self.saw_plte = True
            elif self.chunk_type == b"IDAT":
                self.saw_idat = True
            elif self.chunk_type == b"IEND":
                self._finish_image_data()
                self.ended = True
                self.chunk_type = None
                self.chunk_index += 1
                if self.buffer:
                    fail("PNG carries bytes after IEND")
                return
            self.chunk_type = None
            self.chunk_index += 1

    def _validate_ihdr(self, raw: bytes) -> None:
        if len(raw) != 13:
            fail("PNG IHDR length differs")
        width, height, depth, color, compression, filtering, interlace = struct.unpack(
            ">IIBBBBB", raw
        )
        allowed = {
            0: {1, 2, 4, 8, 16},
            2: {8, 16},
            3: {1, 2, 4, 8},
            4: {8, 16},
            6: {8, 16},
        }
        if (
            not 1 <= width <= 16_384
            or not 1 <= height <= 16_384
            or depth not in allowed.get(color, set())
            or compression != 0
            or filtering != 0
            or interlace != 0
        ):
            fail("PNG IHDR values differ")
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
        row_bytes = (width * channels * depth + 7) // 8
        expected = height * (row_bytes + 1)
        if expected > MAX_PNG_DECODED_BYTES:
            fail("PNG decoded size exceeds its bound")
        self.row_stride = row_bytes + 1
        self.expected_decoded_bytes = expected
        self.bit_depth = depth
        self.color_type = color
        self.decompressor = zlib.decompressobj()

    def _accept_decoded(self, raw: bytes) -> None:
        if self.decoded_bytes + len(raw) > self.expected_decoded_bytes:
            fail("PNG decoded bytes exceed IHDR dimensions")
        first_filter = (-self.decoded_bytes) % self.row_stride
        if any(
            raw[offset] > 4 for offset in range(first_filter, len(raw), self.row_stride)
        ):
            fail("PNG scanline filter type differs")
        self.decoded_bytes += len(raw)

    def _decode_idat(self, raw: bytes) -> None:
        decompressor = self.decompressor
        if decompressor is None or (decompressor.eof and raw):
            fail("PNG IDAT zlib stream ordering differs")
        pending = raw
        while pending:
            before = len(pending)
            try:
                decoded = decompressor.decompress(
                    pending, min(COPY_CHUNK_BYTES, self.expected_decoded_bytes + 1)
                )
            except zlib.error as error:
                raise IndependentViewError("PNG IDAT zlib stream is invalid") from error
            self._accept_decoded(decoded)
            if decompressor.unused_data:
                fail("PNG IDAT contains excess compressed data")
            pending = decompressor.unconsumed_tail
            if pending and len(pending) >= before and not decoded:
                fail("PNG IDAT decompression made no progress")

    def _finish_image_data(self) -> None:
        if self.color_type == 3 and not self.saw_plte:
            fail("indexed PNG lacks its required PLTE")
        decompressor = self.decompressor
        if decompressor is None:
            fail("PNG lacks a decodable IDAT stream")
        try:
            decoded = decompressor.flush(COPY_CHUNK_BYTES)
        except zlib.error as error:
            raise IndependentViewError("PNG IDAT finalization failed") from error
        self._accept_decoded(decoded)
        if (
            not decompressor.eof
            or decompressor.unconsumed_tail
            or decompressor.unused_data
            or self.decoded_bytes != self.expected_decoded_bytes
        ):
            fail("PNG IDAT decoded size or terminator differs")

    def finish(self) -> None:
        if (
            not self.signature_seen
            or not self.ended
            or self.chunk_type is not None
            or self.buffer
        ):
            fail("PNG stream is incomplete")


def _identity(
    st: os.stat_result,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    return (
        st.st_dev,
        st.st_ino,
        st.st_mode,
        st.st_nlink,
        st.st_uid,
        st.st_gid,
        st.st_size,
        st.st_mtime_ns,
        st.st_ctime_ns,
    )


def _read_png(
    root: Path, relative: str, *, forbidden_values: tuple[bytes, ...] = ()
) -> ScreenshotEvidence:
    if relative not in {STOP_SCREENSHOT, FAILURE_SCREENSHOT}:
        fail("screenshot path is outside the fixed bundle set")
    if type(forbidden_values) is not tuple or any(
        type(value) is not bytes or not value for value in forbidden_values
    ):
        fail("screenshot forbidden value set differs")
    root_path = Path(os.path.abspath(root))
    root_fd = browser_fd = file_fd = -1
    digest = hashlib.sha256()
    total = 0
    validator = _PngValidator()
    overlap = max((len(value) for value in forbidden_values), default=1) - 1
    tail = b""
    try:
        root_fd = os.open(
            root_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        root_identity = _identity(os.fstat(root_fd))
        browser_entry = os.stat("browser", dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISDIR(browser_entry.st_mode):
            fail("bundle browser entry is not a directory")
        browser_fd = os.open(
            "browser",
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        browser_identity = _identity(os.fstat(browser_fd))
        if browser_identity != _identity(browser_entry):
            fail("bundle browser directory changed while opening")
        name = relative.split("/", 1)[1]
        entry = os.stat(name, dir_fd=browser_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(entry.st_mode)
            or stat.S_IMODE(entry.st_mode) != 0o600
            or entry.st_nlink != 1
            or not 1 <= entry.st_size <= MAX_PNG_BYTES
        ):
            fail("screenshot mode, links, type, or size differs")
        file_fd = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=browser_fd,
        )
        opened = os.fstat(file_fd)
        file_identity = _identity(opened)
        opened_entry = os.stat(name, dir_fd=browser_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(opened_entry.st_mode)
            or file_identity != _identity(entry)
            or file_identity != _identity(opened_entry)
        ):
            fail("screenshot changed while opening")
        while True:
            chunk = os.read(file_fd, COPY_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_PNG_BYTES:
                fail("screenshot exceeds its streaming bound")
            combined = tail + chunk
            if any(value and value in combined for value in forbidden_values):
                fail("screenshot contains forbidden cleartext")
            tail = combined[-overlap:] if overlap else b""
            digest.update(chunk)
            validator.consume(chunk)
        validator.finish()
        if (
            total != entry.st_size
            or _identity(os.fstat(file_fd)) != file_identity
            or _identity(os.stat(name, dir_fd=browser_fd, follow_symlinks=False))
            != file_identity
            or _identity(os.fstat(browser_fd)) != browser_identity
            or _identity(os.stat("browser", dir_fd=root_fd, follow_symlinks=False))
            != browser_identity
            or _identity(os.fstat(root_fd)) != root_identity
        ):
            fail("screenshot or bundle directory changed while streaming")
        return ScreenshotEvidence(relative, total, digest.hexdigest())
    except IndependentViewError:
        raise
    except OSError as error:
        raise IndependentViewError(
            "failed to open screenshot without symlinks"
        ) from error
    finally:
        for descriptor in (file_fd, browser_fd, root_fd):
            if descriptor >= 0:
                os.close(descriptor)


def _release(
    trace: Any, *, outcome: str, completion_tokens: int | None
) -> Mapping[str, Any]:
    release = _single_event(trace, "request_released")
    if (
        release.get("outcome") != outcome
        or release.get("reset_complete") is not True
        or (
            completion_tokens is not None
            and release.get("completion_tokens") != completion_tokens
        )
    ):
        fail("lifecycle release outcome, completion count, or reset differs")
    return release


def _direct_target(phase: str, result: Any, trace: Any) -> dict[str, Any]:
    if (
        _attr(result, "status", "direct target HTTP") != 200
        or _attr(result, "outcome", "direct target HTTP") != "client_closed"
    ):
        fail("direct target HTTP status or outcome differs")
    _validate_sse_identity(
        result, trace, require_id=phase == "decode_after_first_content"
    )
    sse = _attr(result, "sse", "direct target HTTP")
    items = _sequence(_attr(sse, "items", "SSE metadata"), "SSE items")
    if any(
        _attr(item, "done", "SSE item") is True
        or _attr(item, "finish_reason", "SSE item") is not None
        or _attr(item, "usage_present", "SSE item") is True
        for item in items
    ):
        fail("direct cancelled target contains terminal SSE data")
    content_count = sum(
        _attr(item, "content_utf8_bytes", "SSE item") is not None for item in items
    )
    if (phase == "decode_after_first_content") != (content_count > 0):
        fail("direct target content trigger differs")
    events = _events(trace)
    cancel = _single_event(trace, "request_cancel_requested")
    release = _release(trace, outcome="cancelled", completion_tokens=None)
    if (
        cancel.get("reason") != "client_disconnect"
        or release.get("cancel_reason") != "client_disconnect"
    ):
        fail("direct cancellation reason differs")
    cancel_ns = _integer(cancel.get("observed_monotonic_ns"), "cancel timestamp")
    release_ns = _integer(release.get("observed_monotonic_ns"), "release timestamp")
    delta = release_ns - cancel_ns
    if not 0 <= delta <= 5_000_000_000:
        fail("direct cancel-to-release deadline differs")
    prior_names = [event.get("event") for event in events[: events.index(cancel)]]
    progress = [
        _integer(event.get("processed_prompt_tokens"), "cancel progress", 1)
        for event in events[: events.index(cancel)]
        if event.get("event") == "request_progress"
    ]
    if "request_started" not in prior_names:
        fail("direct cancellation precedes request_started")
    if phase == "after_started_before_progress" and progress:
        fail("after-start cancellation contains prefill progress")
    if phase == "prefill_after_128" and (not progress or max(progress) != 128):
        fail("128-token cancellation progress differs")
    if phase == "prefill_after_2048" and (not progress or max(progress) != 2048):
        fail("2048-token cancellation progress differs")
    if (
        phase == "decode_after_first_content"
        and "request_first_token" not in prior_names
    ):
        fail("decode cancellation lacks first-token evidence")
    return {
        "case_id": f"direct-{phase}-target",
        "transport": "direct_http",
        "http_status": 200,
        "http_outcome": "client_closed",
        "release_outcome": "cancelled",
        "cancel_reason": "client_disconnect",
        "cancel_to_release_ns": delta,
        "completion_tokens": _integer(
            release.get("completion_tokens"), "cancelled completion tokens"
        ),
        "reset_complete": True,
    }


def reconstruct_cancellation(
    session: SessionLike,
    bundle_root: Path,
    *,
    forbidden_values: tuple[bytes, ...] = (),
) -> dict[str, Any]:
    """Rebuild direct and OpenWebUI Stop cancellation results."""

    _validate_session_ready(session)
    http = _http_map(session)
    traces = _trace_map(session)
    phases: list[dict[str, Any]] = []
    for phase in DIRECT_CANCEL_PHASES:
        target_id = f"direct-{phase}-target"
        recovery_id = f"direct-{phase}-recovery"
        target_http = http.get(("cancellation", target_id))
        recovery_http = http.get(("cancellation", recovery_id))
        target_trace = traces.get(("cancellation", target_id))
        recovery_trace = traces.get(("cancellation", recovery_id))
        if None in (target_http, recovery_http, target_trace, recovery_trace):
            fail("direct cancellation pair is incomplete")
        assert target_http is not None
        assert recovery_http is not None
        assert target_trace is not None
        assert recovery_trace is not None
        for role_index, result in enumerate((target_http, recovery_http)):
            expected_index = DIRECT_CANCEL_PHASES.index(phase) * 2 + role_index + 1
            expected_id = target_id if role_index == 0 else recovery_id
            if (
                _attr(result, "request_index", "direct HTTP result") != expected_index
                or _attr(result, "request_key", "direct HTTP result") != expected_id
                or _attr(result, "method", "direct HTTP result") != "POST"
                or _attr(result, "target", "direct HTTP result")
                != "/v1/chat/completions"
            ):
                fail("direct cancellation HTTP identity or order differs")
        target = _direct_target(phase, target_http, target_trace)
        _validate_success_sse(recovery_http, recovery_trace, outcome="length")
        _release(recovery_trace, outcome="length", completion_tokens=2)
        phases.append(
            {
                "phase": phase,
                "target": target,
                "recovery": {
                    "case_id": recovery_id,
                    "transport": "direct_http",
                    "http_status": 200,
                    "http_outcome": "eof",
                    "release_outcome": "length",
                    "completion_tokens": 2,
                    "reset_complete": True,
                },
            }
        )

    stop_case = CANCEL_PHASES[-1]
    recovery_case = f"{stop_case}-recovery"
    stop_trace = traces.get(("cancellation", stop_case))
    recovery_trace = traces.get(("cancellation", recovery_case))
    if stop_trace is None or recovery_trace is None:
        fail("OpenWebUI Stop lifecycle pair is incomplete")
    actions = _actions_for_browser_case(
        session,
        phase="cancellation",
        browser_case=stop_case,
        specs=_stop_specs(),
        case_partition=(stop_case,) * 6 + (recovery_case,) * 3,
    )
    if _attr(actions[0], "input_sha256", "Stop navigation") != (
        _normal_navigation_digest(session)
    ):
        fail("Stop and normal browser navigation URL digests differ")
    screenshot = _read_png(
        bundle_root, STOP_SCREENSHOT, forbidden_values=forbidden_values
    )
    screenshot_action_sha = _attr(actions[4], "screenshot_sha256", "Stop action")
    if screenshot_action_sha != screenshot.sha256:
        fail("Stop screenshot file and action digest differ")
    cancel = _single_event(stop_trace, "request_cancel_requested")
    release = _release(stop_trace, outcome="cancelled", completion_tokens=None)
    if (
        cancel.get("reason") != "client_disconnect"
        or release.get("cancel_reason") != "client_disconnect"
    ):
        fail("Stop lifecycle cancellation reason differs")
    cancel_ns = _integer(cancel.get("observed_monotonic_ns"), "Stop cancel timestamp")
    release_ns = _integer(
        release.get("observed_monotonic_ns"), "Stop release timestamp"
    )
    cancel_to_release = release_ns - cancel_ns
    if not 0 <= cancel_to_release <= 5_000_000_000:
        fail("Stop cancel-to-release deadline differs")
    wait_visible_completed = _integer(
        _attr(actions[3], "completed_monotonic_ns", "Stop action"),
        "Stop visible completion",
    )
    click_completed = _integer(
        _attr(actions[4], "completed_monotonic_ns", "Stop action"),
        "Stop click completion",
    )
    if not wait_visible_completed < click_completed < cancel_ns <= release_ns:
        fail("Stop browser/cancel/release timeline differs")
    _release(recovery_trace, outcome="stop", completion_tokens=None)
    recovery_admitted = _single_event(recovery_trace, "request_admitted")
    recovery_release = _single_event(recovery_trace, "request_released")
    if not (
        _integer(
            _attr(actions[6], "started_monotonic_ns", "Stop recovery submit"),
            "Stop recovery submit start",
        )
        <= _integer(
            recovery_admitted.get("observed_monotonic_ns"),
            "Stop recovery admission",
        )
        <= _integer(
            recovery_release.get("observed_monotonic_ns"), "Stop recovery release"
        )
        <= _integer(
            _attr(actions[8], "completed_monotonic_ns", "Stop recovery ready"),
            "Stop recovery ready completion",
        )
    ):
        fail("Stop recovery browser/lifecycle timeline differs")
    phases.append(
        {
            "phase": stop_case,
            "target": {
                "case_id": stop_case,
                "transport": "openwebui_browser",
                "http_status": None,
                "http_outcome": None,
                "release_outcome": "cancelled",
                "cancel_reason": "client_disconnect",
                "cancel_to_release_ns": cancel_to_release,
                "completion_tokens": None,
                "reset_complete": True,
            },
            "recovery": {
                "case_id": recovery_case,
                "transport": "openwebui_browser",
                "http_status": None,
                "http_outcome": None,
                "release_outcome": "stop",
                "completion_tokens": None,
                "reset_complete": True,
            },
            "browser_action_count": 9,
            "screenshot": dataclasses.asdict(screenshot),
        }
    )
    return {
        "schema_version": CANCEL_RESULTS_SCHEMA,
        "phase_count": 5,
        "request_count": 10,
        "maximum_active_requests": 1,
        "phases": phases,
    }


def _browser_release_case(
    traces: Mapping[tuple[str, str], Any], phase: str, case_id: str
) -> tuple[str, Mapping[str, Any]]:
    trace = traces.get((phase, case_id))
    if trace is None:
        fail("browser lifecycle trace is absent")
    release = _single_event(trace, "request_released")
    outcome = release.get("outcome")
    if outcome not in {"stop", "length"} or release.get("reset_complete") is not True:
        fail("browser release outcome or reset differs")
    return cast(str, outcome), release


def _normal_browser_cases(session: SessionLike) -> list[dict[str, Any]]:
    _normal_navigation_digest(session)
    traces = _trace_map(session)
    projected: list[dict[str, Any]] = []
    for index in range(21):
        case_id = (
            "openwebui_smoke" if index == 0 else f"openwebui_soak_chat_{index:02d}"
        )
        actions = _actions_for_browser_case(
            session,
            phase="openwebui",
            browser_case=case_id,
            specs=_normal_specs(index),
            case_partition=(case_id,) * 5,
        )
        outcome, release = _browser_release_case(traces, "openwebui", case_id)
        trace = traces[("openwebui", case_id)]
        admitted = _single_event(trace, "request_admitted")
        submit_started = _integer(
            _attr(actions[2], "started_monotonic_ns", "browser submit"),
            "browser submit start",
        )
        ready_completed = _integer(
            _attr(actions[4], "completed_monotonic_ns", "browser ready"),
            "browser ready completion",
        )
        if not (
            submit_started
            <= _integer(
                admitted.get("observed_monotonic_ns"), "browser admission timestamp"
            )
            <= _integer(
                release.get("observed_monotonic_ns"), "browser release timestamp"
            )
            <= ready_completed
        ):
            fail("normal browser action/lifecycle timeline differs")
        projected.append(
            {
                "case_index": index,
                "case_id": case_id,
                "action_count": 5,
                "release_outcome": outcome,
                "reset_complete": True,
            }
        )
    return projected


def reconstruct_browser_soak(session: SessionLike) -> list[dict[str, Any]]:
    """Return only the 20 browser cases later joined into soak-results.json."""

    _validate_session_ready(session)
    return [dict(item) for item in _normal_browser_cases(session)[1:]]


def reconstruct_openwebui_smoke(
    session: SessionLike,
    bundle_root: Path,
    *,
    forbidden_values: tuple[bytes, ...] = (),
) -> dict[str, Any]:
    """Rebuild the normal, post-header failure, and recovery browser view."""

    _validate_session_ready(session)
    normal = _normal_browser_cases(session)[0]
    traces = _trace_map(session)
    failure_case = "post-header-failure"
    recovery_case = "post-header-recovery"
    failure_trace = traces.get(("post_header_failure", failure_case))
    recovery_trace = traces.get(("post_header_failure", recovery_case))
    if failure_trace is None or recovery_trace is None:
        fail("post-header failure/recovery lifecycle pair is absent")
    actions = _actions_for_browser_case(
        session,
        phase="post_header_failure",
        browser_case="post_header_worker_failure",
        specs=_failure_specs(),
        case_partition=(failure_case,) * 5 + (recovery_case,) * 4,
    )
    if _attr(actions[0], "input_sha256", "failure navigation") != (
        _normal_navigation_digest(session)
    ):
        fail("failure and normal browser navigation URL digests differ")
    screenshot = _read_png(
        bundle_root, FAILURE_SCREENSHOT, forbidden_values=forbidden_values
    )
    if _attr(actions[4], "screenshot_sha256", "failure action") != screenshot.sha256:
        fail("failure screenshot file and action digest differ")
    terminal = _single_event(failure_trace, "worker_fatal")
    if any(
        event.get("event") == "request_released" for event in _events(failure_trace)
    ):
        fail("failed request contains a normal release")
    fault = _attr(session, "fault_injection", "session")
    if fault is None:
        fail("post-header fault injection is absent")
    normal_probe = _mapping(
        _mapping(_attr(session, "probes", "session"), "session probes").get(
            "normal-segment-start"
        ),
        "normal lifecycle probe",
    )
    restart_probe = _mapping(
        _mapping(_attr(session, "probes", "session"), "session probes").get(
            "post-header-restart-ready"
        ),
        "post-header restart probe",
    )
    if (
        _attr(fault, "phase", "fault") != "post_header_failure"
        or _attr(fault, "case_id", "fault") != failure_case
        or _attr(fault, "injection", "fault") != "post_header_worker_kill"
        or _attr(fault, "signal", "fault") != "SIGKILL"
        or _attr(fault, "command_utf8_bytes", "fault")
        != len(FAULT_COMMAND.encode("utf-8"))
        or _attr(fault, "command_sha256", "fault") != _hash_text(FAULT_COMMAND)
        or _attr(fault, "target_pid", "fault") != normal_probe.get("worker_pid")
        or _attr(fault, "target_starttime_ticks", "fault")
        != normal_probe.get("worker_starttime_ticks")
        or restart_probe.get("probe") != "post-header-restart-ready"
        or restart_probe.get("service_active") is not True
        or restart_probe.get("ready_http_status") != 200
    ):
        fail("post-header fault or restart probe identity differs")
    _release(recovery_trace, outcome="stop", completion_tokens=None)
    admitted = _single_event(recovery_trace, "request_admitted")
    released = _single_event(recovery_trace, "request_released")
    fault_started = _integer(
        _attr(fault, "started_monotonic_ns", "fault"), "fault start", 1
    )
    fault_completed = _integer(
        _attr(fault, "completed_monotonic_ns", "fault"), "fault completion", 1
    )
    fatal_ns = _integer(
        terminal.get("observed_monotonic_ns"), "worker fatal timestamp", 1
    )
    ready_ns = _integer(
        restart_probe.get("observed_monotonic_ns"), "restart ready timestamp", 1
    )
    admitted_ns = _integer(
        admitted.get("observed_monotonic_ns"), "recovery admission timestamp", 1
    )
    released_ns = _integer(
        released.get("observed_monotonic_ns"), "recovery release timestamp", 1
    )
    if (
        not fault_started
        <= fault_completed
        <= fatal_ns
        <= ready_ns
        <= admitted_ns
        <= released_ns
    ):
        fail("post-header fault/fatal/probe/recovery timeline differs")
    if not (
        _integer(
            _attr(actions[3], "completed_monotonic_ns", "failure visible action"),
            "failure visible completion",
        )
        <= fault_started
        <= fatal_ns
        <= _integer(
            _attr(actions[4], "completed_monotonic_ns", "failure failed action"),
            "failure failed completion",
        )
        <= _integer(
            _attr(actions[6], "started_monotonic_ns", "recovery submit action"),
            "failure recovery submit start",
        )
        <= admitted_ns
        <= released_ns
        <= _integer(
            _attr(actions[8], "completed_monotonic_ns", "recovery ready action"),
            "failure recovery ready completion",
        )
    ):
        fail("post-header browser/fault/lifecycle timeline differs")
    recovery_marker = "FAILURE_RECOVERY_OK"
    marker_bytes = len(recovery_marker.encode("utf-8"))
    marker_sha = _hash_text(recovery_marker)
    for index in (7, 8):
        if (
            _attr(actions[index], "result_text_utf8_bytes", "recovery action")
            != marker_bytes
            or _attr(actions[index], "result_text_sha256", "recovery action")
            != marker_sha
        ):
            fail("failure recovery marker evidence differs")
    return {
        "schema_version": OPENWEBUI_SMOKE_SCHEMA,
        "normal": normal,
        "post_header_failure": {
            "case_id": failure_case,
            "action_count": 5,
            "terminal_event": "worker_fatal",
            "fault_injection": "post_header_worker_kill",
            "screenshot": dataclasses.asdict(screenshot),
        },
        "recovery": {
            "case_id": recovery_case,
            "action_count": 4,
            "terminal_event": "request_released",
            "release_outcome": "stop",
            "reset_complete": True,
        },
    }


def reconstruct_front_views(
    session: SessionLike,
    bundle_root: Path,
    *,
    forbidden_values: tuple[bytes, ...] = (),
) -> IndependentFrontViews:
    """Reconstruct and canonically encode all non-metric derived material."""

    api = reconstruct_api_contract(session)
    sampling = reconstruct_sampling(session)
    cancellation = reconstruct_cancellation(
        session, bundle_root, forbidden_values=forbidden_values
    )
    smoke = reconstruct_openwebui_smoke(
        session, bundle_root, forbidden_values=forbidden_values
    )
    browser_soak = reconstruct_browser_soak(session)
    values = {
        "api-contract-results.json": api,
        "sampling-results.json": sampling,
        "cancel-results.json": cancellation,
        "openwebui-smoke.json": smoke,
    }
    return IndependentFrontViews(
        api,
        sampling,
        cancellation,
        smoke,
        browser_soak,
        {
            name: canonical_json_bytes(value, forbidden_values=forbidden_values)
            for name, value in values.items()
        },
    )


__all__ = [
    "API_RESULTS_SCHEMA",
    "CANCEL_RESULTS_SCHEMA",
    "IndependentFrontViews",
    "IndependentViewError",
    "OPENWEBUI_SMOKE_SCHEMA",
    "SAMPLING_RESULTS_SCHEMA",
    "SOAK_RESULTS_SCHEMA",
    "ScreenshotEvidence",
    "canonical_json_bytes",
    "reconstruct_api_contract",
    "reconstruct_browser_soak",
    "reconstruct_cancellation",
    "reconstruct_front_views",
    "reconstruct_openwebui_smoke",
    "reconstruct_sampling",
]
