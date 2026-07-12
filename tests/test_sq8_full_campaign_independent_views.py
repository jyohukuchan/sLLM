#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import struct
import sys
import tempfile
import unittest
import zlib
from collections import Counter
from pathlib import Path
from typing import Any
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "sq8_full_campaign_independent_views.py"
PRODUCER_PATH = ROOT / "tools" / "sq8_full_campaign_views.py"
VALIDATOR_PATH = ROOT / "tools" / "validate-sq8-openwebui-release.py"


def load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VIEWS = load(MODULE_PATH, "sq8_full_campaign_independent_views_test")
PRODUCER = load(PRODUCER_PATH, "sq8_full_campaign_views_parity_test")
VALIDATOR = load(VALIDATOR_PATH, "sq8_release_validator_parity_test")


def digest(value: bytes | str) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def png_chunk(kind: bytes, raw: bytes) -> bytes:
    return (
        len(raw).to_bytes(4, "big")
        + kind
        + raw
        + (zlib.crc32(kind + raw) & 0xFFFF_FFFF).to_bytes(4, "big")
    )


def one_pixel_png() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00\xff"))
        + png_chunk(b"IEND", b"")
    )


def indexed_png_without_plte() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 3, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(b"\x00\x00"))
        + png_chunk(b"IEND", b"")
    )


def png_with_unknown_critical_chunk() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"ABCD", b"")
        + png_chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00\xff"))
        + png_chunk(b"IEND", b"")
    )


@dataclasses.dataclass(frozen=True)
class Order:
    phases: tuple[str, ...]
    openwebui_successful_requests: int
    cancellation_phases: tuple[str, ...]
    normal_gateway_pid: int = 1200


@dataclasses.dataclass(frozen=True)
class Api:
    case_ids: tuple[str, ...]
    request_keys: tuple[str, ...]
    statuses: tuple[int, ...]
    cases: tuple[dict[str, Any], ...]


@dataclasses.dataclass(frozen=True)
class SseItem:
    done: bool
    completion_id_utf8_bytes: int | None = None
    completion_id_sha256: str | None = None
    content_utf8_bytes: int | None = None
    content_sha256: str | None = None
    finish_reason: str | None = None
    usage_present: bool = False
    usage_is_object: bool | None = None
    completion_tokens: int | None = None


@dataclasses.dataclass(frozen=True)
class Sse:
    items: tuple[SseItem, ...]


@dataclasses.dataclass(frozen=True)
class Http:
    phase: str
    case_id: str
    request_index: int
    request_key: str
    method: str
    target: str
    status: int
    outcome: str
    request_body_bytes: int
    request_body_sha256: str
    response_body_bytes: int
    response_body_sha256: str
    sse: Sse | None
    connect_completed_monotonic_ns: int = 1
    write_started_monotonic_ns: int = 2
    last_body_byte_sent_monotonic_ns: int = 3
    response_started_monotonic_ns: int = 4
    response_end_monotonic_ns: int = 5


@dataclasses.dataclass(frozen=True)
class Action:
    phase: str
    case_id: str
    browser_case: str
    action_index: int
    action: str
    selector: str | None
    input_sha256: str | None
    started_monotonic_ns: int
    completed_monotonic_ns: int
    result_visible: bool | None
    result_enabled: bool | None
    result_text_utf8_bytes: int | None
    result_text_sha256: str | None
    screenshot_file: str | None
    screenshot_sha256: str | None


@dataclasses.dataclass
class Trace:
    phase: str
    case_id: str
    completion_id: str
    events: list[dict[str, Any]]


@dataclasses.dataclass(frozen=True)
class Fault:
    phase: str
    case_id: str
    injection: str
    target_pid: int
    target_starttime_ticks: int
    signal: str
    started_monotonic_ns: int
    completed_monotonic_ns: int
    command_utf8_bytes: int = len(VIEWS.FAULT_COMMAND.encode("utf-8"))
    command_sha256: str = digest(VIEWS.FAULT_COMMAND)


@dataclasses.dataclass(frozen=True)
class QuietCheck:
    phase: str
    case_id: str
    quiet_sequence: int
    label: str
    checked_monotonic_ns: int
    observer_open: bool
    observer_event_count: int
    new_journal_record_count: int
    journal_record_count: int
    journal_cursor: str


@dataclasses.dataclass(frozen=True)
class JournalObservation:
    phase: str
    case_id: str
    observation_index: int
    journal_cursor: str
    journal_monotonic_usec: int
    journal_pid: int
    message_utf8_bytes: int
    message_sha256: str


@dataclasses.dataclass
class Session:
    full_campaign_order: Order
    api_contract: Api
    http_results: list[Http]
    http_requests: dict[str, dict[str, Any]]
    browser_actions: list[Action]
    api_journal_observations: list[JournalObservation]
    lifecycle_quiet_checks: list[QuietCheck]
    fault_injection: Fault
    traces: dict[str, Trace]
    releases_by_phase: dict[str, list[dict[str, Any]]]
    probes: dict[str, dict[str, Any]]


def event(
    request_id: str,
    completion_id: str,
    name: str,
    observed: int,
    **fields: Any,
) -> dict[str, Any]:
    return {
        "event": name,
        "request_id": request_id,
        "completion_id": completion_id,
        "observed_monotonic_ns": observed,
        **fields,
    }


def trace(
    phase: str,
    case_id: str,
    events: list[tuple[str, int, dict[str, Any]]],
) -> tuple[str, Trace]:
    request_id = f"req-{case_id}"
    completion_id = f"chatcmpl-{case_id}"
    return request_id, Trace(
        phase,
        case_id,
        completion_id,
        [
            event(request_id, completion_id, name, observed, **fields)
            for name, observed, fields in events
        ],
    )


def success_sse(completion_id: str, outcome: str = "length") -> Sse:
    completion_sha = digest(completion_id)
    return Sse(
        (
            SseItem(
                False,
                len(completion_id),
                completion_sha,
                1,
                digest("x"),
            ),
            SseItem(
                False,
                len(completion_id),
                completion_sha,
                finish_reason=outcome,
                usage_present=True,
                usage_is_object=True,
                completion_tokens=2,
            ),
            SseItem(True),
        )
    )


def response_http(
    phase: str,
    case_id: str,
    request_index: int,
    *,
    outcome: str = "eof",
    sse: Sse | None = None,
) -> Http:
    body = f"request-{case_id}".encode()
    response = f"response-{case_id}".encode()
    return Http(
        phase,
        case_id,
        request_index,
        case_id if phase == "cancellation" else f"p8f-{case_id}",
        "POST",
        "/v1/chat/completions",
        200,
        outcome,
        len(body),
        digest(body),
        len(response),
        digest(response),
        sse,
    )


class CompleteFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        browser = root / "browser"
        browser.mkdir(mode=0o700)
        self.png = one_pixel_png()
        for name in ("openwebui-stop-before.png", "post-header-failure.png"):
            path = browser / name
            path.write_bytes(self.png)
            path.chmod(0o600)
        self.http: list[Http] = []
        self.http_requests: dict[str, dict[str, Any]] = {}
        self.actions: list[Action] = []
        self.traces: dict[str, Trace] = {}
        self._api()
        self.journal_observations = self._journal_observations()
        self.quiet = self._quiet_checks()
        self._sampling()
        self._direct()
        self._normal_browser()
        self._stop()
        self._failure()
        releases: dict[str, list[dict[str, Any]]] = {}
        for value in self.traces.values():
            for lifecycle_event in value.events:
                if lifecycle_event["event"] == "request_released":
                    releases.setdefault(value.phase, []).append(lifecycle_event)
        order = Order(
            (
                "preflight",
                "api_contract",
                "openwebui",
                "cancellation",
                "resource_normal",
                "post_header_failure",
                "resource_restart",
                "latency",
                "final",
            ),
            21,
            VIEWS.CANCEL_PHASES,
        )
        self.session = Session(
            order,
            self.api,
            self.http,
            self.http_requests,
            self.actions,
            self.journal_observations,
            self.quiet,
            Fault(
                "post_header_failure",
                "post-header-failure",
                "post_header_worker_kill",
                1201,
                10_001,
                "SIGKILL",
                200_040,
                200_041,
            ),
            self.traces,
            releases,
            {
                "normal-segment-start": {
                    "probe": "normal-segment-start",
                    "worker_pid": 1201,
                    "worker_starttime_ticks": 10_001,
                },
                "post-header-restart-ready": {
                    "probe": "post-header-restart-ready",
                    "observed_monotonic_ns": 200_050,
                    "service_active": True,
                    "ready_http_status": 200,
                    "worker_pid": 2201,
                    "worker_starttime_ticks": 20_001,
                },
            },
        )

    def _quiet_checks(self) -> list[QuietCheck]:
        labels = [spec[0] for spec in VIEWS._API_SPECS] + [
            "http-client-shutdown",
            "post-observer-close",
            "final-readiness-and-identity",
        ]
        return [
            QuietCheck(
                "api_contract",
                label,
                sequence,
                label,
                (10 + sequence) * 1_000 + 100,
                sequence <= 10,
                0,
                1,
                sequence + 1,
                f"quiet-cursor-{sequence:02d}",
            )
            for sequence, label in enumerate(labels)
        ]

    def _journal_observations(self) -> list[JournalObservation]:
        return [
            JournalObservation(
                "api_contract",
                f"api-journal-{index + 1:02d}",
                index,
                f"quiet-cursor-{index:02d}",
                10 + index,
                1200,
                len(f"journal-message-{index}".encode()),
                digest(f"journal-message-{index}"),
            )
            for index in range(13)
        ]

    def add_trace(
        self,
        phase: str,
        case_id: str,
        values: list[tuple[str, int, dict[str, Any]]],
    ) -> Trace:
        request_id, value = trace(phase, case_id, values)
        self.traces[request_id] = value
        return value

    def _api(self) -> None:
        cases: list[dict[str, Any]] = []
        ids: list[str] = []
        keys: list[str] = []
        statuses: list[int] = []
        for index, spec in enumerate(VIEWS._API_SPECS, 1):
            (
                case_id,
                method,
                target,
                body,
                authorization,
                status,
                code,
                param,
                message,
            ) = spec
            response = f"api-response-{index}".encode()
            error = (
                None
                if message is None
                else {
                    "type": "invalid_request_error",
                    "code": code,
                    "param": param,
                    "message_utf8_bytes": len(message.encode()),
                    "message_sha256": digest(message),
                }
            )
            cases.append(
                {
                    "case_index": index,
                    "case_id": case_id,
                    "method": method,
                    "target": target,
                    "authorization_mode": authorization,
                    "request_body_bytes": len(body),
                    "request_body_sha256": digest(body),
                    "connect_completed_monotonic_ns": index * 10,
                    "write_started_monotonic_ns": index * 10 + 1,
                    "last_body_byte_sent_monotonic_ns": index * 10 + 2,
                    "status": status,
                    "response_started_monotonic_ns": index * 10 + 3,
                    "response_end_monotonic_ns": index * 10 + 4,
                    "content_type": "application/json",
                    "content_length": len(response),
                    "www_authenticate": ["Bearer"] if status == 401 else [],
                    "response_body_bytes": len(response),
                    "response_body_sha256": digest(response),
                    "error": error,
                }
            )
            key = f"api-contract-{index:02d}-{case_id}"
            self.http.append(
                Http(
                    "api_contract",
                    case_id,
                    index,
                    key,
                    method,
                    target,
                    status,
                    "eof",
                    len(body),
                    digest(body),
                    len(response),
                    digest(response),
                    None,
                    index * 10,
                    index * 10 + 1,
                    index * 10 + 2,
                    index * 10 + 3,
                    index * 10 + 4,
                )
            )
            self.http_requests[key] = {
                "response_chunk_count": 1,
                "authorization_mode": authorization,
            }
            ids.append(case_id)
            keys.append(key)
            statuses.append(status)
        self.api = Api(tuple(ids), tuple(keys), tuple(statuses), tuple(cases))

    def _sampling(self) -> None:
        for index in VIEWS.SAMPLED_NORMAL_INDICES:
            case_id = f"normal-measured-{index:03d}"
            value = self.add_trace(
                "resource_normal",
                case_id,
                [
                    ("request_admitted", 20_000 + index * 10, {}),
                    ("request_started", 20_001 + index * 10, {}),
                    ("request_first_token", 20_002 + index * 10, {}),
                    (
                        "request_released",
                        20_003 + index * 10,
                        {
                            "outcome": "length",
                            "completion_tokens": 2,
                            "reset_complete": True,
                            "cancel_reason": None,
                        },
                    ),
                ],
            )
            self.http.append(
                response_http(
                    "resource_normal",
                    case_id,
                    index,
                    sse=success_sse(value.completion_id),
                )
            )

    def _direct(self) -> None:
        for phase_index, phase in enumerate(VIEWS.DIRECT_CANCEL_PHASES):
            target_id = f"direct-{phase}-target"
            recovery_id = f"direct-{phase}-recovery"
            base = 40_000 + phase_index * 1_000
            target_events: list[tuple[str, int, dict[str, Any]]] = [
                ("request_admitted", base, {}),
                ("request_started", base + 1, {}),
            ]
            if phase == "prefill_after_128":
                target_events.append(
                    (
                        "request_progress",
                        base + 2,
                        {"processed_prompt_tokens": 128},
                    )
                )
            elif phase == "prefill_after_2048":
                target_events.extend(
                    (
                        (
                            "request_progress",
                            base + 2,
                            {"processed_prompt_tokens": 128},
                        ),
                        (
                            "request_progress",
                            base + 3,
                            {"processed_prompt_tokens": 2048},
                        ),
                    )
                )
            elif phase == "decode_after_first_content":
                target_events.append(("request_first_token", base + 2, {}))
            target_events.extend(
                (
                    (
                        "request_cancel_requested",
                        base + 10,
                        {"reason": "client_disconnect"},
                    ),
                    (
                        "request_released",
                        base + 100,
                        {
                            "outcome": "cancelled",
                            "completion_tokens": (
                                1 if phase == "decode_after_first_content" else 0
                            ),
                            "reset_complete": True,
                            "cancel_reason": "client_disconnect",
                        },
                    ),
                )
            )
            target_trace = self.add_trace("cancellation", target_id, target_events)
            recovery_trace = self.add_trace(
                "cancellation",
                recovery_id,
                [
                    ("request_admitted", base + 200, {}),
                    ("request_started", base + 201, {}),
                    ("request_first_token", base + 202, {}),
                    (
                        "request_released",
                        base + 203,
                        {
                            "outcome": "length",
                            "completion_tokens": 2,
                            "reset_complete": True,
                            "cancel_reason": None,
                        },
                    ),
                ],
            )
            target_items: tuple[SseItem, ...]
            if phase == "decode_after_first_content":
                target_items = (
                    SseItem(
                        False,
                        len(target_trace.completion_id),
                        digest(target_trace.completion_id),
                        1,
                        digest("x"),
                    ),
                )
            else:
                target_items = ()
            self.http.extend(
                (
                    response_http(
                        "cancellation",
                        target_id,
                        phase_index * 2 + 1,
                        outcome="client_closed",
                        sse=Sse(target_items),
                    ),
                    response_http(
                        "cancellation",
                        recovery_id,
                        phase_index * 2 + 2,
                        sse=success_sse(recovery_trace.completion_id),
                    ),
                )
            )

    def add_actions(
        self,
        phase: str,
        browser_case: str,
        case_partition: tuple[str, ...],
        specs: tuple[Any, ...],
        base: int,
    ) -> list[Action]:
        values: list[Action] = []
        for index, (case_id, spec) in enumerate(
            zip(case_partition, specs, strict=True)
        ):
            if spec.input_sha256 == "navigation":
                input_sha = digest("http://openwebui/?temporary-chat=true")
            else:
                input_sha = spec.input_sha256
            if spec.text == "required":
                text = (
                    "FAILURE_RECOVERY_OK"
                    if phase == "post_header_failure" and index in {7, 8}
                    else f"visible-text-{phase}-{browser_case}-{index}"
                )
                text_bytes = len(text.encode())
                text_sha = digest(text)
            elif spec.text != "none":
                text_bytes = len(spec.text.encode())
                text_sha = digest(spec.text)
            else:
                text_bytes = None
                text_sha = None
            values.append(
                Action(
                    phase,
                    case_id,
                    browser_case,
                    index,
                    spec.name,
                    spec.selector,
                    input_sha,
                    base + index * 10,
                    base + index * 10 + 5,
                    True,
                    spec.enabled,
                    text_bytes,
                    text_sha,
                    spec.screenshot,
                    digest(self.png) if spec.screenshot is not None else None,
                )
            )
        self.actions.extend(values)
        return values

    def _normal_browser(self) -> None:
        for index in range(21):
            case_id = (
                "openwebui_smoke" if index == 0 else f"openwebui_soak_chat_{index:02d}"
            )
            base = 60_000 + index * 1_000
            self.add_actions(
                "openwebui",
                case_id,
                (case_id,) * 5,
                VIEWS._normal_specs(index),
                base,
            )
            self.add_trace(
                "openwebui",
                case_id,
                [
                    ("request_admitted", base + 22, {}),
                    ("request_started", base + 23, {}),
                    ("request_first_token", base + 30, {}),
                    (
                        "request_released",
                        base + 42,
                        {
                            "outcome": "stop",
                            "completion_tokens": 1,
                            "reset_complete": True,
                            "cancel_reason": None,
                        },
                    ),
                ],
            )

    def _stop(self) -> None:
        case_id = "openwebui_stop_after_visible_content"
        recovery = f"{case_id}-recovery"
        base = 150_000
        self.add_actions(
            "cancellation",
            case_id,
            (case_id,) * 6 + (recovery,) * 3,
            VIEWS._stop_specs(),
            base,
        )
        self.add_trace(
            "cancellation",
            case_id,
            [
                ("request_admitted", base + 22, {}),
                ("request_started", base + 23, {}),
                ("request_first_token", base + 30, {}),
                (
                    "request_cancel_requested",
                    base + 50,
                    {"reason": "client_disconnect"},
                ),
                (
                    "request_released",
                    base + 60,
                    {
                        "outcome": "cancelled",
                        "completion_tokens": 1,
                        "reset_complete": True,
                        "cancel_reason": "client_disconnect",
                    },
                ),
            ],
        )
        self.add_trace(
            "cancellation",
            recovery,
            [
                ("request_admitted", base + 62, {}),
                ("request_started", base + 63, {}),
                ("request_first_token", base + 70, {}),
                (
                    "request_released",
                    base + 82,
                    {
                        "outcome": "stop",
                        "completion_tokens": 1,
                        "reset_complete": True,
                        "cancel_reason": None,
                    },
                ),
            ],
        )

    def _failure(self) -> None:
        failure = "post-header-failure"
        recovery = "post-header-recovery"
        base = 200_000
        self.add_actions(
            "post_header_failure",
            "post_header_worker_failure",
            (failure,) * 5 + (recovery,) * 4,
            VIEWS._failure_specs(),
            base,
        )
        self.add_trace(
            "post_header_failure",
            failure,
            [
                ("request_admitted", base + 22, {}),
                ("request_started", base + 23, {}),
                ("request_first_token", base + 30, {}),
                ("worker_fatal", base + 42, {"reason": "worker_exit"}),
            ],
        )
        self.add_trace(
            "post_header_failure",
            recovery,
            [
                ("request_admitted", base + 62, {}),
                ("request_started", base + 63, {}),
                ("request_first_token", base + 70, {}),
                (
                    "request_released",
                    base + 82,
                    {
                        "outcome": "stop",
                        "completion_tokens": 1,
                        "reset_complete": True,
                        "cancel_reason": None,
                    },
                ),
            ],
        )


def direct_producer_input(result: dict[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    request_index = 0
    for phase in result["phases"][:4]:
        for role in ("target", "recovery"):
            request_index += 1
            value = phase[role]
            item = {
                "request_index": request_index,
                "phase": phase["phase"],
                "role": role,
                "case_id": value["case_id"],
                "request_body_bytes": 10,
                "request_body_sha256": digest(f"request-{request_index}"),
                "http_status": value["http_status"],
                "http_outcome": value["http_outcome"],
                "response_body_bytes": 20,
                "response_body_sha256": digest(f"response-{request_index}"),
                "lifecycle_event_count": 4,
                "request_id_sha256": digest(f"request-id-{request_index}"),
                "completion_id_sha256": digest(f"completion-id-{request_index}"),
                "release_observed_monotonic_ns": 1000 + request_index,
                "release_outcome": value["release_outcome"],
                "reset_complete": True,
                "completion_tokens": value["completion_tokens"],
            }
            if role == "target":
                item.update(
                    {
                        "trigger_observed_monotonic_ns": 900,
                        "cancel_observed_monotonic_ns": 950,
                        "cancel_to_release_ns": value["cancel_to_release_ns"],
                        "progress": None,
                    }
                )
            cases.append(item)
    return {
        "schema_version": PRODUCER.DIRECT_CANCEL_INPUT_SCHEMA,
        "phase_order": list(PRODUCER.DIRECT_CANCEL_PHASES),
        "request_count": 8,
        "http_record_count": 32,
        "lifecycle_record_count": 55,
        "maximum_active_requests": 1,
        "component_summary_sha256": "a" * 64,
        "source_bindings": {},
        "cases": cases,
    }


def stop_producer_input(result: dict[str, Any]) -> dict[str, Any]:
    value = result["phases"][4]
    return {
        "schema_version": PRODUCER.STOP_INPUT_SCHEMA,
        "browser_case": value["phase"],
        "browser_action_count": 9,
        "browser_socket_event_count": 1,
        "lifecycle_event_count": 11,
        "request_count": 2,
        "maximum_active_requests": 1,
        "component_summary_sha256": "a" * 64,
        "screenshot": value["screenshot"],
        "source_bindings": {},
        "browser_evidence": {},
        "gateway_evidence": {
            "target_outcome": "cancelled",
            "cancel_reason": "client_disconnect",
            "cancel_to_release_ns": value["target"]["cancel_to_release_ns"],
            "recovery_outcome": "stop",
            "target_reset_complete": True,
            "recovery_reset_complete": True,
        },
        "raw_artifacts": {},
    }


def combined_producer_input() -> dict[str, Any]:
    schedule = [
        {
            "position": index,
            "case_index": index,
            "case_kind": "smoke" if index == 0 else "soak",
            "browser_case": (
                "openwebui_smoke" if index == 0 else f"openwebui_soak_chat_{index:02d}"
            ),
        }
        for index in range(21)
    ]
    return {
        "schema_version": PRODUCER.COMBINED_INPUT_SCHEMA,
        "mode": "smoke_then_soak20",
        "schedule": schedule,
        "chat_count": 21,
        "action_count": 105,
        "lifecycle_record_count": 105,
        "maximum_active_requests": 1,
        "stop_release_count": 21,
        "reset_complete_count": 21,
        "component_summary_sha256": "a" * 64,
        "source_bindings": {},
        "cases": [
            {
                **common,
                "browser_case_sha256": "a" * 64,
                "action_count": 5,
                "socket_event_count": 4,
                "chat_id_sha256": "a" * 64,
                "message_id_sha256": "a" * 64,
                "request_id_sha256": "a" * 64,
                "completion_id_sha256": "a" * 64,
                "admitted_monotonic_ns": index * 10 + 1,
                "released_monotonic_ns": index * 10 + 2,
                "outcome": "stop",
                "reset_complete": True,
            }
            for index, common in enumerate(schedule)
        ],
    }


def failure_producer_input(result: dict[str, Any]) -> dict[str, Any]:
    screenshot = result["post_header_failure"]["screenshot"]
    return {
        "schema_version": PRODUCER.FAILURE_INPUT_SCHEMA,
        "phase": "post_header_failure",
        "cases": {
            "failure": "post-header-failure",
            "recovery": "post-header-recovery",
        },
        "source_sha256": {},
        "summary_sha256": "a" * 64,
        "service": {},
        "browser": {
            "action_count": 9,
            "socket_event_count": 7,
            "screenshot_file": screenshot["file"],
            "screenshot_bytes": screenshot["bytes"],
            "screenshot_sha256": screenshot["sha256"],
        },
        "fault": {
            "target_pid": 1201,
            "started_monotonic_ns": 200_040,
            "completed_monotonic_ns": 200_041,
            "worker_fatal_monotonic_ns": 200_042,
        },
        "recovery": {
            "ready_completed_monotonic_ns": 200_050,
            "admitted_monotonic_ns": 200_062,
            "released_monotonic_ns": 200_082,
        },
        "journal": {},
    }


class IndependentFrontViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = CompleteFixture(Path(self.temporary.name))

    def test_full_positive_reconstructs_canonical_views(self) -> None:
        result = VIEWS.reconstruct_front_views(
            self.fixture.session,
            self.fixture.root,
            forbidden_values=(b"forbidden-secret",),
        )
        self.assertEqual(result.api_contract_results["case_count"], 10)
        self.assertEqual(result.sampling_results["sampled_request_count"], 20)
        self.assertEqual(result.cancel_results["phase_count"], 5)
        self.assertEqual(result.openwebui_smoke["recovery"]["action_count"], 4)
        self.assertEqual(len(result.browser_soak_cases), 20)
        for raw in result.canonical_bytes.values():
            self.assertTrue(raw.endswith(b"\n"))
            self.assertNotIn(b"passed", raw)
            self.assertEqual(
                raw,
                json.dumps(
                    json.loads(raw),
                    ensure_ascii=True,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
                + b"\n",
            )

    def test_api_sampling_and_cancellation_match_producer_projection(self) -> None:
        api = VIEWS.reconstruct_api_contract(self.fixture.session)
        api_input = {
            "schema_version": PRODUCER.API_INPUT_SCHEMA,
            "case_count": 10,
            "http_record_count": 40,
            "journal_record_count": 13,
            "lifecycle_event_count": 0,
            "quiet_check_count": 13,
            "cases": api["cases"],
            "source_bindings": {},
        }
        self.assertEqual(api, PRODUCER.project_api_contract(api_input))
        sampling = VIEWS.reconstruct_sampling(self.fixture.session)
        self.assertEqual(sampling, PRODUCER.project_sampling(sampling["cases"]))
        cancellation = VIEWS.reconstruct_cancellation(
            self.fixture.session, self.fixture.root
        )
        self.assertEqual(
            cancellation,
            PRODUCER.project_cancellation(
                direct_producer_input(cancellation),
                stop_producer_input(cancellation),
            ),
        )

    def test_frozen_api_specs_match_validator_and_extra_fields_are_dropped(
        self,
    ) -> None:
        expected = tuple(
            (
                case.case_id,
                case.method,
                case.target,
                case.body,
                case.authorization_mode,
                case.expected_status,
                case.expected_code,
                case.expected_param,
                case.expected_message,
            )
            for case in VALIDATOR.API_CONTRACT_CASES
        )
        self.assertEqual(VIEWS._API_SPECS, expected)
        result = VIEWS.reconstruct_api_contract(self.fixture.session)
        self.assertEqual(
            set(result["cases"][0]),
            {
                "case_index",
                "case_id",
                "status",
                "request_body_bytes",
                "request_body_sha256",
                "response_body_bytes",
                "response_body_sha256",
                "error",
            },
        )
        key = self.fixture.session.api_contract.request_keys[0]
        self.fixture.session.http_requests[key]["response_chunk_count"] = 2
        with self.assertRaises(VIEWS.IndependentViewError):
            VIEWS.reconstruct_api_contract(self.fixture.session)

    def test_api_quiet_check_deletion_and_field_tampering_fail(self) -> None:
        original = list(self.fixture.session.lifecycle_quiet_checks)

        def changed(index: int, **fields: Any) -> list[QuietCheck]:
            values = list(original)
            values[index] = dataclasses.replace(values[index], **fields)
            return values

        mutations = {
            "deleted": original[:-1],
            "label": changed(0, label="wrong-label"),
            "phase": changed(0, phase="preflight"),
            "case": changed(0, case_id="wrong-case"),
            "sequence": changed(1, quiet_sequence=0),
            "observer_open": changed(10, observer_open=False),
            "observer_count": changed(0, observer_event_count=1),
            "journal_delta": changed(1, new_journal_record_count=2),
            "time": changed(0, checked_monotonic_ns=0),
            "cursor": changed(1, journal_cursor=original[0].journal_cursor),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                self.fixture.session.lifecycle_quiet_checks = mutation
                with self.assertRaises(VIEWS.IndependentViewError):
                    VIEWS.reconstruct_api_contract(self.fixture.session)
        self.fixture.session.lifecycle_quiet_checks = original

        observations = list(self.fixture.session.api_journal_observations)

        def observation_changed(index: int, **fields: Any) -> list[JournalObservation]:
            values = list(observations)
            values[index] = dataclasses.replace(values[index], **fields)
            return values

        observation_mutations = {
            "observation_deleted": observations[:-1],
            "observation_index": observation_changed(1, observation_index=0),
            "observation_cursor": observation_changed(
                1, journal_cursor=observations[0].journal_cursor
            ),
            "observation_time": observation_changed(1, journal_monotonic_usec=9),
            "observation_pid": observation_changed(0, journal_pid=999),
        }
        for label, observation_mutation in observation_mutations.items():
            with self.subTest(label=label):
                self.fixture.session.api_journal_observations = observation_mutation
                with self.assertRaises(VIEWS.IndependentViewError):
                    VIEWS.reconstruct_api_contract(self.fixture.session)
        self.fixture.session.api_journal_observations = observations

    def test_api_quiet_checks_accept_one_anchor_without_journal_observations(self):
        self.fixture.session.api_journal_observations = []
        self.fixture.session.lifecycle_quiet_checks = [
            dataclasses.replace(
                check,
                new_journal_record_count=0,
                journal_record_count=0,
                journal_cursor="api-start-anchor",
            )
            for check in self.fixture.session.lifecycle_quiet_checks
        ]
        result = VIEWS.reconstruct_api_contract(self.fixture.session)
        self.assertEqual(result["quiet_check_count"], 13)

        self.fixture.session.lifecycle_quiet_checks[-1] = dataclasses.replace(
            self.fixture.session.lifecycle_quiet_checks[-1],
            journal_cursor="changed-anchor",
        )
        with self.assertRaises(VIEWS.IndependentViewError):
            VIEWS.reconstruct_api_contract(self.fixture.session)

    def test_browser_soak_matches_producer_projection(self) -> None:
        cases = VIEWS.reconstruct_browser_soak(self.fixture.session)
        self.assertEqual(
            cases, PRODUCER.project_browser_soak(combined_producer_input())
        )

    def test_openwebui_smoke_matches_producer_projection(self) -> None:
        result = VIEWS.reconstruct_openwebui_smoke(
            self.fixture.session, self.fixture.root
        )
        self.assertEqual(
            result,
            PRODUCER.project_openwebui_smoke(
                combined_producer_input(), failure_producer_input(result)
            ),
        )

    def test_accepts_the_validator_dataclass_shapes(self) -> None:
        source = self.fixture.session
        traces = {
            request_id: VALIDATOR.RequestTrace(
                value.phase,
                value.case_id,
                value.completion_id,
                value.events,
                value.events[-1]["event"],
            )
            for request_id, value in source.traces.items()
        }
        http_results = []
        for value in source.http_results:
            sse = None
            if value.sse is not None:
                items = tuple(
                    VALIDATOR.HttpSseItem(
                        index,
                        index + 1,
                        item.done,
                        item.completion_id_utf8_bytes,
                        item.completion_id_sha256,
                        item.content_utf8_bytes,
                        item.content_sha256,
                        item.finish_reason,
                        item.usage_present,
                        item.usage_is_object,
                        item.completion_tokens,
                    )
                    for index, item in enumerate(value.sse.items)
                )
                sse = VALIDATOR.HttpSseMetadata(
                    1,
                    1,
                    1,
                    items,
                )
            http_results.append(
                VALIDATOR.HttpCompactResult(
                    value.phase,
                    value.case_id,
                    value.request_index,
                    value.request_key,
                    value.method,
                    value.target,
                    value.status,
                    value.outcome,
                    value.request_body_bytes,
                    value.request_body_sha256,
                    value.response_body_bytes,
                    value.response_body_sha256,
                    value.connect_completed_monotonic_ns,
                    value.write_started_monotonic_ns,
                    value.last_body_byte_sent_monotonic_ns,
                    value.response_started_monotonic_ns,
                    value.response_end_monotonic_ns,
                    sse,
                )
            )
        actions = tuple(
            VALIDATOR.BrowserActionData(**dataclasses.asdict(value))
            for value in source.browser_actions
        )
        journal_observations = tuple(
            VALIDATOR.ApiJournalObservationData(**dataclasses.asdict(value))
            for value in source.api_journal_observations
        )
        quiet_checks = tuple(
            VALIDATOR.LifecycleQuietCheckData(**dataclasses.asdict(value))
            for value in source.lifecycle_quiet_checks
        )
        fault = VALIDATOR.FaultInjectionData(
            source.fault_injection.phase,
            source.fault_injection.case_id,
            source.fault_injection.injection,
            source.fault_injection.target_pid,
            source.fault_injection.target_starttime_ticks,
            source.fault_injection.signal,
            source.fault_injection.command_utf8_bytes,
            source.fault_injection.command_sha256,
            source.fault_injection.started_monotonic_ns,
            source.fault_injection.completed_monotonic_ns,
        )
        order = VALIDATOR.FullCampaignOrderResult(
            source.full_campaign_order.phases,
            source.full_campaign_order.openwebui_successful_requests,
            source.full_campaign_order.cancellation_phases,
            1200,
            2200,
            1201,
            2201,
            2,
            3,
        )
        api = VALIDATOR.ApiContractValidationResult(
            source.api_contract.case_ids,
            source.api_contract.request_keys,
            source.api_contract.statuses,
            source.api_contract.cases,
        )
        actual = VALIDATOR.SessionData(
            "run-id",
            "boot-id",
            {},
            {},
            traces,
            source.releases_by_phase,
            {},
            "cursor",
            Counter(),
            source.http_requests,
            [],
            source.probes,
            (),
            tuple(http_results),
            actions,
            journal_observations,
            quiet_checks,
            fault,
            order,
            api,
        )
        result = VIEWS.reconstruct_front_views(actual, self.fixture.root)
        self.assertEqual(result.sampling_results["sampled_request_count"], 20)
        self.assertEqual(result.cancel_results["phase_count"], 5)

    def test_sampling_order_status_outcome_and_sse_id_tampering_fail(self) -> None:
        case_id = "normal-measured-005"
        position = next(
            index
            for index, item in enumerate(self.fixture.session.http_results)
            if item.case_id == case_id
        )
        original = self.fixture.session.http_results[position]
        assert original.sse is not None
        original_sse = original.sse
        mutations = {
            "order": dataclasses.replace(original, request_index=6),
            "status": dataclasses.replace(original, status=500),
            "outcome": dataclasses.replace(original, outcome="client_closed"),
            "sse_id": dataclasses.replace(
                original,
                sse=dataclasses.replace(
                    original_sse,
                    items=(
                        dataclasses.replace(
                            original_sse.items[0], completion_id_sha256="f" * 64
                        ),
                        *original_sse.items[1:],
                    ),
                ),
            ),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                self.fixture.session.http_results[position] = mutation
                with self.assertRaises(VIEWS.IndependentViewError):
                    VIEWS.reconstruct_sampling(self.fixture.session)
                self.fixture.session.http_results[position] = original

    def test_direct_deadline_and_stop_action_result_tampering_fail(self) -> None:
        target = self.fixture.session.traces[
            "req-direct-after_started_before_progress-target"
        ]
        release = target.events[-1]
        original_time = release["observed_monotonic_ns"]
        cancel_time = target.events[-2]["observed_monotonic_ns"]
        release["observed_monotonic_ns"] = cancel_time + 5_000_000_001
        with self.assertRaises(VIEWS.IndependentViewError):
            VIEWS.reconstruct_cancellation(self.fixture.session, self.fixture.root)
        release["observed_monotonic_ns"] = original_time

        action_index = next(
            index
            for index, action in enumerate(self.fixture.session.browser_actions)
            if action.phase == "cancellation" and action.action_index == 4
        )
        original = self.fixture.session.browser_actions[action_index]
        for mutation in (
            dataclasses.replace(original, action="wait_ready"),
            dataclasses.replace(original, result_enabled=False),
        ):
            self.fixture.session.browser_actions[action_index] = mutation
            with self.assertRaises(VIEWS.IndependentViewError):
                VIEWS.reconstruct_cancellation(self.fixture.session, self.fixture.root)
        self.fixture.session.browser_actions[action_index] = original

    def test_final_markers_extra_actions_and_fault_command_tampering_fail(self) -> None:
        for phase, browser_case, action_index in (
            ("openwebui", "openwebui_smoke", 3),
            ("openwebui", "openwebui_smoke", 4),
            ("cancellation", "openwebui_stop_after_visible_content", 7),
            ("cancellation", "openwebui_stop_after_visible_content", 8),
            ("post_header_failure", "post_header_worker_failure", 7),
            ("post_header_failure", "post_header_worker_failure", 8),
        ):
            position = next(
                index
                for index, action in enumerate(self.fixture.session.browser_actions)
                if action.phase == phase
                and action.browser_case == browser_case
                and action.action_index == action_index
            )
            original = self.fixture.session.browser_actions[position]
            wrong = "WRONG_MODEL_OUTPUT"
            self.fixture.session.browser_actions[position] = dataclasses.replace(
                original,
                result_text_utf8_bytes=len(wrong.encode()),
                result_text_sha256=digest(wrong),
            )
            with self.assertRaises(VIEWS.IndependentViewError):
                VIEWS.reconstruct_front_views(self.fixture.session, self.fixture.root)
            self.fixture.session.browser_actions[position] = original

        original_action = self.fixture.session.browser_actions[0]
        self.fixture.session.browser_actions.append(
            dataclasses.replace(
                original_action,
                phase="latency",
                case_id="extra-browser-action",
                browser_case="extra-browser-action",
                action_index=0,
            )
        )
        with self.assertRaises(VIEWS.IndependentViewError):
            VIEWS.reconstruct_api_contract(self.fixture.session)
        self.fixture.session.browser_actions.pop()

        original_fault = self.fixture.session.fault_injection
        for mutation in (
            dataclasses.replace(original_fault, command_utf8_bytes=1),
            dataclasses.replace(original_fault, command_sha256=digest("os.kill")),
        ):
            self.fixture.session.fault_injection = mutation
            with self.assertRaises(VIEWS.IndependentViewError):
                VIEWS.reconstruct_openwebui_smoke(
                    self.fixture.session, self.fixture.root
                )
        self.fixture.session.fault_injection = original_fault

    def test_fault_order_and_screenshot_tampering_fail(self) -> None:
        probe = self.fixture.session.probes["post-header-restart-ready"]
        original_ready = probe["observed_monotonic_ns"]
        probe["observed_monotonic_ns"] = 200_041
        with self.assertRaises(VIEWS.IndependentViewError):
            VIEWS.reconstruct_openwebui_smoke(self.fixture.session, self.fixture.root)
        probe["observed_monotonic_ns"] = original_ready

        screenshot = self.fixture.root / "browser" / "post-header-failure.png"
        original = screenshot.read_bytes()
        screenshot.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        screenshot.chmod(0o600)
        with self.assertRaises(VIEWS.IndependentViewError):
            VIEWS.reconstruct_openwebui_smoke(self.fixture.session, self.fixture.root)

    def test_screenshot_mode_and_action_digest_tampering_fail(self) -> None:
        screenshot = self.fixture.root / "browser" / "openwebui-stop-before.png"
        screenshot.chmod(0o644)
        with self.assertRaises(VIEWS.IndependentViewError):
            VIEWS.reconstruct_cancellation(self.fixture.session, self.fixture.root)
        screenshot.chmod(0o600)
        hardlink = self.fixture.root / "stop-hardlink"
        os.link(screenshot, hardlink)
        with self.assertRaises(VIEWS.IndependentViewError):
            VIEWS.reconstruct_cancellation(self.fixture.session, self.fixture.root)
        hardlink.unlink()
        position = next(
            index
            for index, action in enumerate(self.fixture.session.browser_actions)
            if action.phase == "cancellation" and action.action_index == 4
        )
        original = self.fixture.session.browser_actions[position]
        self.fixture.session.browser_actions[position] = dataclasses.replace(
            original, screenshot_sha256="0" * 64
        )
        with self.assertRaises(VIEWS.IndependentViewError):
            VIEWS.reconstruct_cancellation(self.fixture.session, self.fixture.root)

    def test_png_rejects_missing_palette_and_unknown_critical_chunk(self) -> None:
        screenshot = self.fixture.root / "browser" / "openwebui-stop-before.png"
        position = next(
            index
            for index, action in enumerate(self.fixture.session.browser_actions)
            if action.phase == "cancellation" and action.action_index == 4
        )
        original_action = self.fixture.session.browser_actions[position]
        for raw, message in (
            (indexed_png_without_plte(), "PLTE"),
            (png_with_unknown_critical_chunk(), "unknown critical"),
        ):
            screenshot.write_bytes(raw)
            screenshot.chmod(0o600)
            self.fixture.session.browser_actions[position] = dataclasses.replace(
                original_action, screenshot_sha256=digest(raw)
            )
            with self.assertRaisesRegex(VIEWS.IndependentViewError, message):
                VIEWS.reconstruct_cancellation(self.fixture.session, self.fixture.root)

    def test_png_fifo_replacement_is_nonblocking_and_rejected(self) -> None:
        screenshot = self.fixture.root / "browser" / "openwebui-stop-before.png"
        real_open = os.open
        observed_flags: list[int] = []
        replaced = False

        def racing_open(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            nonlocal replaced
            if path == "openwebui-stop-before.png" and not replaced:
                replaced = True
                screenshot.unlink()
                os.mkfifo(screenshot, 0o600)
                observed_flags.append(flags)
            return real_open(path, flags, mode, dir_fd=dir_fd)

        with mock.patch.object(VIEWS.os, "open", side_effect=racing_open):
            with self.assertRaises(VIEWS.IndependentViewError):
                VIEWS.reconstruct_cancellation(self.fixture.session, self.fixture.root)
        self.assertTrue(replaced)
        self.assertEqual(len(observed_flags), 1)
        self.assertNotEqual(observed_flags[0] & os.O_NONBLOCK, 0)

    def test_canonical_rejects_passed_and_forbidden_cleartext(self) -> None:
        for value, secret in (
            ({"nested": {"passed": False}}, ()),
            ({"value": "secret-value"}, (b"secret-value",)),
            ({"value": 'escaped-secret-"-value'}, (b'escaped-secret-"-value',)),
            ({"value": "escaped-secret-\\-value"}, (b"escaped-secret-\\-value",)),
            ({"value": "escaped-secret-\t-value"}, (b"escaped-secret-\t-value",)),
        ):
            with self.assertRaises(VIEWS.IndependentViewError):
                VIEWS.canonical_json_bytes(value, forbidden_values=secret)

    def test_production_module_does_not_import_producer_or_validator(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("sq8_full_campaign_views", source)
        self.assertNotIn("validate-sq8-openwebui-release", source)


if __name__ == "__main__":
    unittest.main()
