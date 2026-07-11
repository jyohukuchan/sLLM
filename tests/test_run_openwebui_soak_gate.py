from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "run-openwebui-soak-gate.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("run_openwebui_soak_gate", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def digest(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def lifecycle(name: str, timestamp: int, request: str, completion: str, **fields):
    return {
        "schema_version": TOOL.LIFECYCLE_SCHEMA,
        "event": name,
        "observed_monotonic_ns": timestamp,
        "request_id": request,
        "completion_id": completion,
        **fields,
    }


def lifecycle_trace(request: str, completion: str, start: int):
    return [
        lifecycle(
            "request_admitted",
            start,
            request,
            completion,
            stream=True,
            prompt_tokens=16,
            max_completion_tokens=64,
        ),
        lifecycle(
            "request_started",
            start + 1,
            request,
            completion,
            stream=True,
            prompt_tokens=16,
            admit_to_start_ns=1,
        ),
        lifecycle(
            "request_progress",
            start + 2,
            request,
            completion,
            phase="prefill",
            processed_prompt_tokens=16,
            prompt_tokens=16,
        ),
        lifecycle(
            "request_first_token",
            start + 3,
            request,
            completion,
            stream=True,
            completion_tokens=1,
        ),
        lifecycle(
            "request_released",
            start + 4,
            request,
            completion,
            stream=True,
            outcome="stop",
            cancel_reason=None,
            prompt_tokens=16,
            completion_tokens=4,
            reset_complete=True,
            admit_to_start_ns=1,
            start_to_release_ns=3,
            admit_to_release_ns=4,
        ),
    ]


def action(
    case_index: int,
    action_index: int,
    name: str,
    base_url: str,
    *,
    include_smoke: bool = False,
):
    base = case_index * 1000
    inputs = (
        digest(TOOL.navigation_url(base_url)),
        digest(TOOL.MODEL_ID),
        digest(TOOL.case_prompt(case_index, include_smoke=include_smoke)),
        None,
        None,
    )
    selectors = (None, "body", "#chat-input", ".chat-assistant", "#chat-input")
    carries_text = name in {"wait_visible", "wait_ready"}
    return {
        "browser_case": TOOL.browser_case(case_index, include_smoke=include_smoke),
        "action_index": action_index,
        "action": name,
        "selector": selectors[action_index],
        "input_sha256": inputs[action_index],
        "started_monotonic_ns": str(base + action_index * 10),
        "completed_monotonic_ns": str(base + action_index * 10 + 5),
        "result": {
            "visible": True,
            "enabled": True if name in {"submit_chat", "wait_ready"} else None,
            "text_utf8_bytes": 12 if carries_text else None,
            "text_sha256": digest(f"response-{case_index}") if carries_text else None,
        },
        "screenshot_file": None,
        "screenshot_sha256": None,
    }


def socket_event(
    sequence: int,
    kind: str,
    timestamp: int,
    *,
    done: bool = False,
    content: bool = False,
):
    return {
        "sequence": sequence,
        "observed_monotonic_ns": str(timestamp),
        "correlation_target": "chat_target",
        "type": kind,
        "done": done,
        "has_error": False,
        "content_utf8_bytes": 12 if content else 0,
        "content_sha256": digest(f"content-{timestamp}") if content else None,
    }


def browser_case_fixture(
    case_index: int, base_url: str, *, include_smoke: bool = False
):
    base = case_index * 1000
    chat_id = f"temporary-chat-{case_index}"
    message_id = f"temporary-message-{case_index}"
    marker = TOOL.case_marker(case_index, include_smoke=include_smoke).encode()
    return {
        "schema_version": (
            TOOL.COMBINED_BROWSER_SCHEMA if include_smoke else TOOL.BROWSER_SCHEMA
        ),
        "record_type": TOOL.case_record_type(case_index, include_smoke=include_smoke),
        "browser_case": TOOL.browser_case(case_index, include_smoke=include_smoke),
        "case_index": case_index,
        "observed_monotonic_ns": str(base + 60),
        "browser_actions": [
            action(
                case_index,
                index,
                name,
                base_url,
                include_smoke=include_smoke,
            )
            for index, name in enumerate(TOOL.FINAL_ACTIONS)
        ],
        "socket_correlation": {
            "target": {
                "chat_id_utf8_bytes": len(chat_id.encode()),
                "chat_id_sha256": digest(chat_id),
                "message_id_utf8_bytes": len(message_id.encode()),
                "message_id_sha256": digest(message_id),
            },
            "submit_started_monotonic_ns": str(base + 20),
            "submit_completed_monotonic_ns": str(base + 25),
            "first_content_observed_monotonic_ns": str(base + 30),
            "done_observed_monotonic_ns": str(base + 42),
            "done_event_count": 1,
            "cancellation_event_count": 0,
            "provider_error_count": 0,
        },
        "socket_events": [
            socket_event(0, "chat:active", base + 26),
            socket_event(1, "chat:completion", base + 30, content=True),
            socket_event(2, "chat:outlet", base + 31),
            socket_event(3, "chat:completion", base + 42, done=True),
        ],
        "visible_marker": {
            "expected_marker_utf8_bytes": len(marker),
            "expected_marker_sha256": digest(marker),
            "observed": True,
        },
        "page_error_count": 0,
        "page_errors": [],
        "page_state": {
            "page_index": case_index,
            "temporary_chat": True,
            "created": True,
            "closed": True,
            "open_pages_after_close": 0,
        },
    }


def browser_values(base_url: str, *, include_smoke: bool = False):
    cases = [
        browser_case_fixture(index, base_url, include_smoke=include_smoke)
        for index in TOOL.case_indices(include_smoke=include_smoke)
    ]
    raws = [json.dumps(value, separators=(",", ":")).encode() for value in cases]
    summary = {
        "schema_version": (
            TOOL.COMBINED_BROWSER_SCHEMA if include_smoke else TOOL.BROWSER_SCHEMA
        ),
        "record_type": (
            TOOL.COMBINED_SUMMARY_RECORD_TYPE
            if include_smoke
            else "openwebui_soak_summary"
        ),
        "browser_case": (TOOL.COMBINED_RUN_CASE if include_smoke else TOOL.RUN_CASE),
        "observed_monotonic_ns": str(len(cases) * 1000 + 100),
        "chat_count": len(cases),
        "action_count": len(cases) * len(TOOL.FINAL_ACTIONS),
        "socket_event_count": sum(len(value["socket_events"]) for value in cases),
        "browser_process_count": 1,
        "browser_context_count": 1,
        "browser_context_closed_count": 1,
        "page_count_created": len(cases),
        "page_count_closed": len(cases),
        "maximum_open_pages": 1,
        "page_error_count": 0,
        "cancellation_event_count": 0,
        "provider_error_count": 0,
        "case_record_sha256": [digest(raw) for raw in raws],
    }
    if include_smoke:
        summary["mode"] = TOOL.COMBINED_MODE
        summary["schedule"] = TOOL.schedule_evidence(include_smoke=True)
    return cases, summary


def framed_lines(cases, summary):
    case_raws = [json.dumps(value, separators=(",", ":")).encode() for value in cases]
    summary = copy.deepcopy(summary)
    summary["case_record_sha256"] = [digest(raw) for raw in case_raws]
    summary["socket_event_count"] = sum(len(value["socket_events"]) for value in cases)
    summary_raw = json.dumps(summary, separators=(",", ":")).encode()
    return [*zip(case_raws, cases, strict=True), (summary_raw, summary)]


class BrowserEvidenceTests(unittest.TestCase):
    base_url = "http://127.0.0.1:3000"

    def test_twenty_case_browser_stdout_and_summary_validate(self):
        cases, summary = browser_values(self.base_url)
        lines = framed_lines(cases, summary)
        with tempfile.TemporaryDirectory() as temporary:
            summary_path = Path(temporary) / TOOL.BROWSER_SUMMARY_NAME
            summary_path.write_bytes(lines[-1][0] + b"\n")
            evidence, result = TOOL.validate_browser_stdout(
                lines, summary_path, TOOL.SecretGuard([]), base_url=self.base_url
            )
        self.assertEqual(len(evidence), 20)
        self.assertEqual(result["action_count"], 100)
        self.assertEqual(result["socket_event_count"], 80)

    def test_explicit_smoke_then_twenty_case_schedule_validates(self):
        cases, summary = browser_values(self.base_url, include_smoke=True)
        lines = framed_lines(cases, summary)
        with tempfile.TemporaryDirectory() as temporary:
            summary_path = Path(temporary) / TOOL.BROWSER_SUMMARY_NAME
            summary_path.write_bytes(lines[-1][0] + b"\n")
            evidence, result = TOOL.validate_browser_stdout(
                lines,
                summary_path,
                TOOL.SecretGuard([]),
                base_url=self.base_url,
                include_smoke=True,
            )
        self.assertEqual(len(evidence), 21)
        self.assertEqual(evidence[0]["case_index"], 0)
        self.assertEqual(evidence[0]["browser_case"], TOOL.SMOKE_CASE)
        self.assertEqual(evidence[1]["browser_case"], TOOL.browser_case(1))
        self.assertEqual(result["chat_count"], 21)
        self.assertEqual(result["action_count"], 105)
        self.assertEqual(result["mode"], TOOL.COMBINED_MODE)
        self.assertEqual(result["schedule"], TOOL.schedule_evidence(include_smoke=True))

    def test_combined_schedule_order_and_summary_identity_are_fail_closed(self):
        cases, summary = browser_values(self.base_url, include_smoke=True)
        changed_smoke = copy.deepcopy(cases[0])
        changed_smoke["visible_marker"]["expected_marker_sha256"] = "0" * 64
        changed_smoke_raw = json.dumps(changed_smoke, separators=(",", ":")).encode()
        with self.assertRaisesRegex(TOOL.SoakGateError, "marker evidence"):
            TOOL.validate_browser_case(
                changed_smoke,
                changed_smoke_raw,
                TOOL.SecretGuard([]),
                case_index=0,
                base_url=self.base_url,
                include_smoke=True,
            )

        reordered = [cases[1], cases[0], *cases[2:]]
        lines = framed_lines(reordered, summary)
        with tempfile.TemporaryDirectory() as temporary:
            summary_path = Path(temporary) / TOOL.BROWSER_SUMMARY_NAME
            summary_path.write_bytes(lines[-1][0] + b"\n")
            with self.assertRaisesRegex(TOOL.SoakGateError, "case identity"):
                TOOL.validate_browser_stdout(
                    lines,
                    summary_path,
                    TOOL.SecretGuard([]),
                    base_url=self.base_url,
                    include_smoke=True,
                )

        for field, replacement in (
            ("mode", "soak20"),
            ("schedule", list(reversed(summary["schedule"]))),
        ):
            changed = copy.deepcopy(summary)
            changed[field] = replacement
            lines = framed_lines(cases, changed)
            with tempfile.TemporaryDirectory() as temporary:
                summary_path = Path(temporary) / TOOL.BROWSER_SUMMARY_NAME
                summary_path.write_bytes(lines[-1][0] + b"\n")
                with self.assertRaisesRegex(
                    TOOL.SoakGateError, "summary mode or schedule"
                ):
                    TOOL.validate_browser_stdout(
                        lines,
                        summary_path,
                        TOOL.SecretGuard([]),
                        base_url=self.base_url,
                        include_smoke=True,
                    )

    def test_action_marker_and_case_hash_tampering_are_rejected(self):
        cases, summary = browser_values(self.base_url)
        changed = copy.deepcopy(cases[0])
        changed["browser_actions"][2]["input_sha256"] = "0" * 64
        raw = json.dumps(changed, separators=(",", ":")).encode()
        with self.assertRaisesRegex(TOOL.SoakGateError, "action identity"):
            TOOL.validate_browser_case(
                changed,
                raw,
                TOOL.SecretGuard([]),
                case_index=1,
                base_url=self.base_url,
            )

        changed = copy.deepcopy(cases[0])
        changed["visible_marker"]["expected_marker_sha256"] = "0" * 64
        raw = json.dumps(changed, separators=(",", ":")).encode()
        with self.assertRaisesRegex(TOOL.SoakGateError, "marker evidence"):
            TOOL.validate_browser_case(
                changed,
                raw,
                TOOL.SecretGuard([]),
                case_index=1,
                base_url=self.base_url,
            )

        lines = framed_lines(cases, summary)
        lines[-1][1]["case_record_sha256"][0] = "0" * 64
        changed_summary_raw = json.dumps(lines[-1][1], separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory() as temporary:
            summary_path = Path(temporary) / TOOL.BROWSER_SUMMARY_NAME
            summary_path.write_bytes(changed_summary_raw + b"\n")
            with self.assertRaisesRegex(TOOL.SoakGateError, "case hashes"):
                TOOL.validate_browser_summary(
                    lines[-1][1],
                    changed_summary_raw,
                    summary_path,
                    TOOL.SecretGuard([]),
                    [
                        TOOL.validate_browser_case(
                            value,
                            raw,
                            TOOL.SecretGuard([]),
                            case_index=index,
                            base_url=self.base_url,
                        )
                        for index, (raw, value) in enumerate(lines[:-1], start=1)
                    ],
                )

    def test_boolean_values_cannot_substitute_for_integer_evidence(self):
        cases, summary = browser_values(self.base_url)
        changed = copy.deepcopy(cases[0])
        changed["case_index"] = True
        raw = json.dumps(changed, separators=(",", ":")).encode()
        with self.assertRaisesRegex(TOOL.SoakGateError, "bounded integer"):
            TOOL.validate_browser_case(
                changed,
                raw,
                TOOL.SecretGuard([]),
                case_index=1,
                base_url=self.base_url,
            )

        changed = copy.deepcopy(cases[0])
        changed["browser_actions"][0]["action_index"] = False
        raw = json.dumps(changed, separators=(",", ":")).encode()
        with self.assertRaisesRegex(TOOL.SoakGateError, "bounded integer"):
            TOOL.validate_browser_case(
                changed,
                raw,
                TOOL.SecretGuard([]),
                case_index=1,
                base_url=self.base_url,
            )

        lines = framed_lines(cases, summary)
        changed_summary = copy.deepcopy(lines[-1][1])
        changed_summary["browser_process_count"] = True
        changed_raw = json.dumps(changed_summary, separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / TOOL.BROWSER_SUMMARY_NAME
            path.write_bytes(changed_raw + b"\n")
            with self.assertRaisesRegex(TOOL.SoakGateError, "bounded integer"):
                TOOL.validate_browser_summary(
                    changed_summary,
                    changed_raw,
                    path,
                    TOOL.SecretGuard([]),
                    [
                        TOOL.validate_browser_case(
                            value,
                            raw,
                            TOOL.SecretGuard([]),
                            case_index=index,
                            base_url=self.base_url,
                        )
                        for index, (raw, value) in enumerate(lines[:-1], start=1)
                    ],
                )

    def test_missing_extra_and_duplicate_chat_records_are_rejected(self):
        cases, summary = browser_values(self.base_url)
        lines = framed_lines(cases, summary)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / TOOL.BROWSER_SUMMARY_NAME
            path.write_bytes(lines[-1][0] + b"\n")
            with self.assertRaisesRegex(TOOL.SoakGateError, "record count"):
                TOOL.validate_browser_stdout(
                    lines[:-2] + [lines[-1]],
                    path,
                    TOOL.SecretGuard([]),
                    base_url=self.base_url,
                )
            with self.assertRaisesRegex(TOOL.SoakGateError, "record count"):
                TOOL.validate_browser_stdout(
                    [lines[0], *lines],
                    path,
                    TOOL.SecretGuard([]),
                    base_url=self.base_url,
                )

        changed_cases = copy.deepcopy(cases)
        changed_cases[1]["socket_correlation"]["target"]["chat_id_sha256"] = (
            changed_cases[0]["socket_correlation"]["target"]["chat_id_sha256"]
        )
        changed_lines = framed_lines(changed_cases, summary)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / TOOL.BROWSER_SUMMARY_NAME
            path.write_bytes(changed_lines[-1][0] + b"\n")
            with self.assertRaisesRegex(TOOL.SoakGateError, "duplicated"):
                TOOL.validate_browser_stdout(
                    changed_lines,
                    path,
                    TOOL.SecretGuard([]),
                    base_url=self.base_url,
                )

    def test_state_event_cancel_provider_and_page_errors_are_zero(self):
        case = browser_case_fixture(1, self.base_url)
        changed = copy.deepcopy(case["socket_events"])
        changed[0]["content_utf8_bytes"] = 1
        changed[0]["content_sha256"] = "a" * 64
        with self.assertRaisesRegex(TOOL.SoakGateError, "state event"):
            TOOL.validate_socket_events(changed)
        changed = copy.deepcopy(case["socket_events"])
        changed[0]["done"] = True
        with self.assertRaisesRegex(TOOL.SoakGateError, "state event"):
            TOOL.validate_socket_events(changed)
        changed = copy.deepcopy(case["socket_events"])
        changed[1]["has_error"] = True
        with self.assertRaisesRegex(TOOL.SoakGateError, "provider-error"):
            TOOL.validate_socket_events(changed)
        changed = copy.deepcopy(case["socket_events"])
        changed[2]["type"] = "chat:tasks:cancel"
        with self.assertRaisesRegex(TOOL.SoakGateError, "cancellation"):
            TOOL.validate_socket_events(changed)

        changed = copy.deepcopy(case["socket_events"])
        changed.append(socket_event(4, "chat:completion", 1043, content=True))
        with self.assertRaisesRegex(TOOL.SoakGateError, "content, done"):
            TOOL.validate_socket_events(changed)

    def test_socket_first_content_must_precede_visible_action_completion(self):
        case = browser_case_fixture(1, self.base_url)
        case["socket_events"][1]["observed_monotonic_ns"] = "1036"
        case["socket_events"][2]["observed_monotonic_ns"] = "1037"
        case["socket_correlation"]["first_content_observed_monotonic_ns"] = "1036"
        raw = json.dumps(case, separators=(",", ":")).encode()
        with self.assertRaisesRegex(TOOL.SoakGateError, "socket-to-action"):
            TOOL.validate_browser_case(
                case,
                raw,
                TOOL.SecretGuard([]),
                case_index=1,
                base_url=self.base_url,
            )


class GatewayTraceTests(unittest.TestCase):
    base_url = "http://127.0.0.1:3000"

    def machine(
        self,
        count=TOOL.CHAT_COUNT,
        *,
        expected_count=TOOL.CHAT_COUNT,
        include_smoke=False,
    ):
        machine = TOOL.SoakLifecycleMachine(expected_count=expected_count)
        indices = (
            TOOL.case_indices(include_smoke=True)
            if include_smoke
            else tuple(range(1, count + 1))
        )
        for case_index in indices[:count]:
            for event in lifecycle_trace(
                f"request-{case_index}",
                f"completion-{case_index}",
                case_index * 1000 + 21,
            ):
                machine.consume(event)
        return machine

    def browser_evidence(self):
        cases, _summary = browser_values(self.base_url)
        return [
            TOOL.validate_browser_case(
                value,
                json.dumps(value, separators=(",", ":")).encode(),
                TOOL.SecretGuard([]),
                case_index=index,
                base_url=self.base_url,
            )
            for index, value in enumerate(cases, start=1)
        ]

    def test_exact_twenty_serial_stop_reset_traces_correlate(self):
        correlations = TOOL.validate_gateway_traces(
            self.machine(), self.browser_evidence()
        )
        self.assertEqual(len(correlations), 20)
        self.assertTrue(all(item["outcome"] == "stop" for item in correlations))
        self.assertTrue(all(item["reset_complete"] for item in correlations))

    def test_smoke_then_twenty_gateway_traces_correlate_in_one_epoch(self):
        cases, _summary = browser_values(self.base_url, include_smoke=True)
        evidence = [
            TOOL.validate_browser_case(
                value,
                json.dumps(value, separators=(",", ":")).encode(),
                TOOL.SecretGuard([]),
                case_index=index,
                base_url=self.base_url,
                include_smoke=True,
            )
            for index, value in zip(
                TOOL.case_indices(include_smoke=True), cases, strict=True
            )
        ]
        correlations = TOOL.validate_gateway_traces(
            self.machine(21, expected_count=21, include_smoke=True),
            evidence,
            include_smoke=True,
        )
        self.assertEqual(len(correlations), 21)
        self.assertEqual(correlations[0]["case_index"], 0)
        self.assertEqual(correlations[0]["browser_case"], TOOL.SMOKE_CASE)
        self.assertEqual(correlations[1]["case_index"], 1)
        self.assertEqual(correlations[1]["browser_case"], TOOL.browser_case(1))

    def test_gateway_first_token_must_precede_browser_content(self):
        evidence = self.browser_evidence()
        evidence[0]["first_content_ns"] = 1023
        with self.assertRaisesRegex(TOOL.SoakGateError, "browser ordering"):
            TOOL.validate_gateway_traces(self.machine(), evidence)

    def test_missing_extra_and_overlap_are_rejected(self):
        with self.assertRaisesRegex(TOOL.SoakGateError, "request count"):
            TOOL.validate_gateway_traces(self.machine(19), self.browser_evidence())
        machine = self.machine()
        with self.assertRaisesRegex(TOOL.SoakGateError, "extra"):
            machine.consume(
                lifecycle(
                    "request_admitted",
                    30_000,
                    "request-extra",
                    "completion-extra",
                    stream=True,
                    prompt_tokens=16,
                    max_completion_tokens=64,
                )
            )
        overlapping = TOOL.SoakLifecycleMachine()
        overlapping.consume(lifecycle_trace("request-a", "completion-a", 10)[0])
        with self.assertRaisesRegex(TOOL.SoakGateError, "overlapping"):
            overlapping.consume(lifecycle_trace("request-b", "completion-b", 11)[0])

    def test_release_must_be_stop_reset_complete_without_cancel(self):
        trace = lifecycle_trace("request-a", "completion-a", 10)
        trace[-1]["outcome"] = "length"
        machine = TOOL.SoakLifecycleMachine()
        with self.assertRaisesRegex(TOOL.SoakGateError, "reset-complete stop"):
            for event in trace:
                machine.consume(event)

        trace = lifecycle_trace("request-a", "completion-a", 10)
        trace[-1]["reset_complete"] = False
        machine = TOOL.SoakLifecycleMachine()
        with self.assertRaisesRegex(TOOL.SoakGateError, "reset-complete stop"):
            for event in trace:
                machine.consume(event)

        trace = lifecycle_trace("request-a", "completion-a", 10)
        trace.insert(
            -1,
            lifecycle(
                "request_cancel_requested",
                14,
                "request-a",
                "completion-a",
                stream=True,
                reason="client_disconnect",
                admit_to_cancel_ns=4,
            ),
        )
        machine = TOOL.SoakLifecycleMachine()
        with self.assertRaisesRegex(TOOL.SoakGateError, "cancelled"):
            for event in trace:
                machine.consume(event)

    def test_prompt_completion_and_duration_identity_are_fail_closed(self):
        trace = lifecycle_trace("request-a", "completion-a", 10)
        trace[1]["prompt_tokens"] = 17
        machine = TOOL.SoakLifecycleMachine()
        with self.assertRaisesRegex(TOOL.SoakGateError, "prompt-token"):
            for event in trace:
                machine.consume(event)

        trace = lifecycle_trace("request-a", "completion-a", 10)
        trace[-1]["completion_tokens"] = 0
        machine = TOOL.SoakLifecycleMachine()
        with self.assertRaisesRegex(TOOL.SoakGateError, "reset-complete stop"):
            for event in trace:
                machine.consume(event)

        trace = lifecycle_trace("request-a", "completion-a", 10)
        trace[-1]["completion_tokens"] = 65
        machine = TOOL.SoakLifecycleMachine()
        with self.assertRaisesRegex(TOOL.SoakGateError, "reset-complete stop"):
            for event in trace:
                machine.consume(event)

        trace = lifecycle_trace("request-a", "completion-a", 10)
        trace[-1]["admit_to_release_ns"] = 5
        machine = TOOL.SoakLifecycleMachine()
        with self.assertRaisesRegex(TOOL.SoakGateError, "duration identity"):
            for event in trace:
                machine.consume(event)


class RawIdentitySecretAndAtomicTests(unittest.TestCase):
    def test_lifecycle_and_journal_payload_bytes_and_cursor_are_exact(self):
        event = lifecycle_trace("request-a", "completion-a", 10)[0]
        payload = TOOL.compact_json(event)
        self.assertEqual(TOOL.validate_lifecycle_payload(payload), event)
        with self.assertRaisesRegex(TOOL.SoakGateError, "canonical"):
            TOOL.validate_lifecycle_payload(b" " + payload)
        journal = {
            "__CURSOR": "cursor-1",
            "__MONOTONIC_TIMESTAMP": "10",
            "_BOOT_ID": "a" * 32,
            "_PID": "123",
            "_SYSTEMD_UNIT": "ullm-openai.service",
            "PRIORITY": "6",
            "MESSAGE": "INFO:     " + payload.decode(),
        }
        cursor, observed = TOOL.validate_journal_record(
            TOOL.compact_json(journal),
            service="ullm-openai.service",
            main_pid=123,
            boot_id="a" * 32,
            cursors=set(),
            lifecycle_payloads=set(),
        )
        self.assertEqual((cursor, observed), ("cursor-1", payload))
        missing_priority = dict(journal)
        del missing_priority["PRIORITY"]
        with self.assertRaisesRegex(TOOL.SoakGateError, "PRIORITY"):
            TOOL.validate_journal_record(
                TOOL.compact_json(missing_priority),
                service="ullm-openai.service",
                main_pid=123,
                boot_id="a" * 32,
                cursors=set(),
                lifecycle_payloads=set(),
            )
        changed = payload.replace(b"request-a", b"request-b")
        with self.assertRaisesRegex(TOOL.SoakGateError, "payload bytes differ"):
            TOOL.require_correlated_prefix([payload], [changed])

    def test_content_addressed_image_and_command_never_embed_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "browser.cjs"
            token = root / "token"
            output = root / "output"
            script.write_text("", encoding="ascii")
            token.write_text("cleartext-token", encoding="ascii")
            output.mkdir()
            command = TOOL.build_browser_command(
                docker="docker",
                image="browser@sha256:" + "a" * 64,
                name="soak-container",
                script=script,
                token_file=token,
                browser_output=output,
                openwebui_url="http://127.0.0.1:3000",
                uid=os.geteuid(),
                gid=os.getegid(),
            )
        self.assertEqual(command.count("--mount"), 3)
        self.assertIn("--network=host", command)
        self.assertNotIn("cleartext-token", "\n".join(command))
        self.assertEqual(command[-2:], ["node", TOOL.BROWSER_SCRIPT_CONTAINER_PATH])
        self.assertNotIn("OPENWEBUI_SOAK_MODE", command)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "browser.cjs"
            token = root / "token"
            output = root / "output"
            script.write_text("", encoding="ascii")
            token.write_text("cleartext-token", encoding="ascii")
            output.mkdir()
            combined = TOOL.build_browser_command(
                docker="docker",
                image="browser@sha256:" + "a" * 64,
                name="combined-container",
                script=script,
                token_file=token,
                browser_output=output,
                openwebui_url="http://127.0.0.1:3000",
                uid=os.geteuid(),
                gid=os.getegid(),
                include_smoke=True,
            )
        mode_index = combined.index("OPENWEBUI_SOAK_MODE=" + TOOL.COMBINED_MODE)
        self.assertEqual(combined[mode_index - 1], "--env")
        self.assertNotIn("cleartext-token", "\n".join(combined))
        self.assertEqual(
            TOOL.normalized_browser_image("sha256:" + "b" * 64),
            ("sha256:" + "b" * 64, "sha256:" + "b" * 64),
        )
        with self.assertRaisesRegex(TOOL.SoakGateError, "immutable"):
            TOOL.normalized_browser_image("browser:latest")
        source = TOOL_PATH.read_text(encoding="utf-8")
        self.assertIn('"image_reference_sha256"', source)
        self.assertNotIn('"image_reference": browser_image', source)

    def test_container_output_is_isolated_and_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            source = temporary_root / "source"
            source.write_bytes(b"stable")
            self.assertEqual(
                TOOL.stable_regular_snapshot(source, "test source", 1024), b"stable"
            )
            symlink = temporary_root / "source-link"
            symlink.symlink_to(source)
            with self.assertRaisesRegex(RuntimeError, "without following links"):
                TOOL.stable_regular_snapshot(symlink, "test source", 1024)

            root = temporary_root / TOOL.BROWSER_CONTAINER_OUTPUT_DIR_NAME
            root.mkdir(mode=0o700)
            summary = root / TOOL.BROWSER_SUMMARY_NAME
            summary.write_text("{}\n", encoding="ascii")
            self.assertEqual(TOOL.validate_container_output_directory(root), summary)
            published = temporary_root / "published.json"
            self.assertEqual(
                TOOL.snapshot_validated_browser_summary(summary, published, b"{}"),
                b"{}\n",
            )
            summary.write_text('{"changed":true}\n', encoding="ascii")
            with self.assertRaisesRegex(TOOL.SoakGateError, "changed after"):
                TOOL.snapshot_validated_browser_summary(
                    summary, temporary_root / "changed.json", b"{}"
                )
            (root / "unexpected").write_text("x", encoding="ascii")
            with self.assertRaisesRegex(TOOL.SoakGateError, "layout or identity"):
                TOOL.validate_container_output_directory(root)

    def test_journal_final_seal_rejects_event_arriving_during_stop(self):
        event = lifecycle_trace("request-a", "completion-a", 10)[0]
        payload = TOOL.compact_json(event)
        record = TOOL.ObserverRecord(
            payload, event, 10, 123, os.geteuid(), os.getegid()
        )

        class FakeJournal:
            def __init__(self, mutate_on_stop: bool):
                self.mutate_on_stop = mutate_on_stop
                self.records = [b"record"]
                self.cursors = {"cursor-1"}
                self.lifecycle = [payload]
                self.stderr_bytes = 0
                self.stderr_digest = hashlib.sha256()

            def stop(self):
                if self.mutate_on_stop:
                    self.lifecycle.append(b'{"extra":"lifecycle"}')

        sealed = TOOL.stop_and_validate_journal(FakeJournal(False), [record])
        self.assertEqual(sealed["lifecycle_records"], 1)
        with self.assertRaisesRegex(TOOL.SoakGateError, "final seal"):
            TOOL.stop_and_validate_journal(FakeJournal(True), [record])

    def test_secret_guard_failure_output_and_atomic_abort_are_redacted(self):
        guard = TOOL.SecretGuard([b"cleartext-secret"])
        with self.assertRaisesRegex(TOOL.SoakGateError, "forbidden"):
            guard.reject(b'{"value":"cleartext-secret"}', "fixture")
        stderr = io.StringIO()
        with (
            mock.patch.object(
                TOOL, "execute", side_effect=TOOL.SoakGateError("cleartext-secret")
            ),
            mock.patch.object(TOOL, "parse_args", return_value=object()),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertEqual(TOOL.main([]), 1)
        self.assertEqual(stderr.getvalue(), "OpenWebUI browser soak gate failed\n")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failed = TOOL.AtomicRunDirectory(root / "failed")
            writer = TOOL.AtomicLineWriter(
                failed.stage / "browser" / "raw.jsonl", maximum_bytes=1024
            )
            writer.write_line(b'{"partial":true}')
            writer.abort()
            failed.abort()
            self.assertFalse((root / "failed").exists())
            self.assertFalse(failed.stage.exists())

            passed = TOOL.AtomicRunDirectory(root / "passed")
            (passed.stage / "summary.json").write_text("complete", encoding="ascii")
            TOOL.fsync_bundle_tree(passed.stage)
            passed.publish()
            self.assertEqual((root / "passed" / "summary.json").read_text(), "complete")

            unsafe = TOOL.AtomicRunDirectory(root / "unsafe")
            (unsafe.stage / "browser" / "link").symlink_to("missing")
            with self.assertRaisesRegex(TOOL.SoakGateError, "unsafe artifact"):
                TOOL.fsync_bundle_tree(unsafe.stage)
            unsafe.abort()

    def test_gate_and_browser_scripts_are_executable(self):
        self.assertEqual(TOOL.GATE_SOURCE_RAW, TOOL_PATH.read_bytes())
        self.assertEqual(TOOL.SUPPORT_SOURCE_RAW, TOOL.SUPPORT_TOOL_PATH.read_bytes())
        self.assertTrue(os.access(TOOL_PATH, os.X_OK))
        self.assertTrue(
            os.access(ROOT / "deploy" / "openwebui" / "browser-soak.cjs", os.X_OK)
        )

    def test_include_smoke_is_explicit_and_default_remains_exact_twenty(self):
        required = [
            "--output-dir",
            "/tmp/output",
            "--token-file",
            "/tmp/token",
            "--browser-image",
            "sha256:" + "a" * 64,
            "--openwebui-url",
            "http://127.0.0.1:3000",
            "--service",
            "ullm-openai.service",
        ]
        default = TOOL.parse_args(required)
        combined = TOOL.parse_args([*required, "--include-smoke"])
        self.assertFalse(default.include_smoke)
        self.assertTrue(combined.include_smoke)
        self.assertEqual(TOOL.case_indices(include_smoke=False), tuple(range(1, 21)))
        self.assertEqual(TOOL.case_indices(include_smoke=True), tuple(range(0, 21)))

    def test_node_and_gate_combined_schedules_are_identical(self):
        script = ROOT / "deploy" / "openwebui" / "browser-soak.cjs"
        program = r"""
const m = require(process.argv[1]);
const defaults = m.caseSchedule("soak20");
const combined = m.caseSchedule(m.COMBINED_MODE);
process.stdout.write(JSON.stringify({
  defaults: m.scheduleEvidence(defaults),
  combined: m.scheduleEvidence(combined),
  smokeMarker: combined[0].marker,
  smokePrompt: combined[0].prompt,
}));
"""
        completed = subprocess.run(
            ["node", "-e", program, str(script)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        value = json.loads(completed.stdout)
        self.assertEqual(value["defaults"], TOOL.schedule_evidence(include_smoke=False))
        self.assertEqual(value["combined"], TOOL.schedule_evidence(include_smoke=True))
        self.assertEqual(value["smokeMarker"], TOOL.SMOKE_MARKER)
        self.assertEqual(
            digest(value["smokePrompt"]),
            digest(TOOL.case_prompt(0, include_smoke=True)),
        )


if __name__ == "__main__":
    unittest.main()
