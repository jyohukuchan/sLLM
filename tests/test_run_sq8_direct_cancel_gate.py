from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "tools" / "run-sq8-direct-cancel-gate.py"
CLIENT_PATH = ROOT / "tools" / "sq8-openwebui-http-client.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GATE = (
    load_module("run_sq8_direct_cancel_gate", GATE_PATH)
    if GATE_PATH.is_file()
    else None
)
CLIENT = load_module("sq8_openwebui_http_client_for_direct_gate", CLIENT_PATH)


ROLE_EVENT = b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
CONTENT_EVENT = b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
USAGE_EVENT = b'data: {"choices":[],"usage":{"completion_tokens":2}}\n\n'
DONE_EVENT = b"data: [DONE]\n\n"


EXPECTED_PHASES = (
    "after_started_before_progress",
    "prefill_after_128",
    "prefill_after_2048",
    "decode_after_first_content",
)
TARGET_PROMPT_TOKENS = {
    "after_started_before_progress": 3584,
    "prefill_after_128": 3584,
    "prefill_after_2048": 3584,
    "decode_after_first_content": 32,
}


def lifecycle(
    name: str,
    timestamp: int,
    *,
    request_id: str = "request-target",
    completion_id: str = "chatcmpl-target",
    **fields,
):
    assert GATE is not None
    return {
        "schema_version": GATE.COL.LIFECYCLE_SCHEMA,
        "event": name,
        "observed_monotonic_ns": timestamp,
        "request_id": request_id,
        "completion_id": completion_id,
        **fields,
    }


def admitted(
    timestamp: int,
    prompt_tokens: int,
    max_tokens: int,
    *,
    request_id: str = "request-target",
    completion_id: str = "chatcmpl-target",
):
    return lifecycle(
        "request_admitted",
        timestamp,
        request_id=request_id,
        completion_id=completion_id,
        stream=True,
        prompt_tokens=prompt_tokens,
        max_completion_tokens=max_tokens,
    )


def started(
    timestamp: int,
    prompt_tokens: int,
    *,
    request_id: str = "request-target",
    completion_id: str = "chatcmpl-target",
):
    return lifecycle(
        "request_started",
        timestamp,
        request_id=request_id,
        completion_id=completion_id,
        stream=True,
        prompt_tokens=prompt_tokens,
        admit_to_start_ns=1,
    )


def progress(
    timestamp: int,
    processed: int,
    prompt_tokens: int,
    *,
    request_id: str = "request-target",
    completion_id: str = "chatcmpl-target",
):
    return lifecycle(
        "request_progress",
        timestamp,
        request_id=request_id,
        completion_id=completion_id,
        phase="prefill",
        processed_prompt_tokens=processed,
        prompt_tokens=prompt_tokens,
    )


def first_token(
    timestamp: int,
    *,
    request_id: str = "request-target",
    completion_id: str = "chatcmpl-target",
):
    return lifecycle(
        "request_first_token",
        timestamp,
        request_id=request_id,
        completion_id=completion_id,
        stream=True,
        completion_tokens=1,
    )


def cancel_requested(
    timestamp: int,
    *,
    admitted_at: int = 1_000,
    request_id: str = "request-target",
    completion_id: str = "chatcmpl-target",
):
    return lifecycle(
        "request_cancel_requested",
        timestamp,
        request_id=request_id,
        completion_id=completion_id,
        stream=True,
        reason="client_disconnect",
        admit_to_cancel_ns=timestamp - admitted_at,
    )


def released(
    timestamp: int,
    prompt_tokens: int,
    *,
    cancelled: bool,
    completion_tokens: int,
    admitted_at: int = 1_000,
    request_id: str = "request-target",
    completion_id: str = "chatcmpl-target",
):
    admit_to_start = 1
    admit_to_release = timestamp - admitted_at
    return lifecycle(
        "request_released",
        timestamp,
        request_id=request_id,
        completion_id=completion_id,
        stream=True,
        outcome="cancelled" if cancelled else "length",
        cancel_reason="client_disconnect" if cancelled else None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reset_complete=True,
        admit_to_start_ns=admit_to_start,
        start_to_release_ns=admit_to_release - admit_to_start,
        admit_to_release_ns=admit_to_release,
    )


def valid_target_validator(phase: str, *, base: int = 1_000):
    assert GATE is not None
    prompt_tokens = TARGET_PROMPT_TOKENS[phase]
    validator = GATE.CancellationTraceValidator(GATE.PHASE_SPECS[phase], prompt_tokens)
    validator.consume(admitted(base, prompt_tokens, 512))
    validator.consume(started(base + 1, prompt_tokens))
    if phase == "prefill_after_128":
        validator.consume(progress(base + 2, 128, prompt_tokens))
        trigger = base + 2
    elif phase == "prefill_after_2048":
        for index, processed in enumerate(range(128, 2049, 128), start=2):
            validator.consume(progress(base + index, processed, prompt_tokens))
        trigger = base + 17
    elif phase == "decode_after_first_content":
        validator.consume(progress(base + 2, prompt_tokens, prompt_tokens))
        validator.consume(first_token(base + 3))
        validator.observe_content(base + 4, 0)
        trigger = base + 4
    else:
        trigger = base + 1
    validator.mark_close(trigger)
    cancel_time = trigger + 1
    validator.consume(cancel_requested(cancel_time))
    validator.consume(
        released(
            cancel_time + 1,
            prompt_tokens,
            cancelled=True,
            completion_tokens=0 if phase != "decode_after_first_content" else 1,
        )
    )
    return validator


def valid_recovery_validator(*, base: int = 10_000):
    assert GATE is not None
    validator = GATE.RecoveryTraceValidator(32)
    validator.consume(
        admitted(
            base,
            32,
            2,
            request_id="request-recovery",
            completion_id="chatcmpl-recovery",
        )
    )
    validator.consume(
        started(
            base + 1,
            32,
            request_id="request-recovery",
            completion_id="chatcmpl-recovery",
        )
    )
    validator.consume(
        progress(
            base + 2,
            32,
            32,
            request_id="request-recovery",
            completion_id="chatcmpl-recovery",
        )
    )
    validator.consume(
        first_token(
            base + 3,
            request_id="request-recovery",
            completion_id="chatcmpl-recovery",
        )
    )
    validator.consume(
        released(
            base + 4,
            32,
            cancelled=False,
            completion_tokens=2,
            admitted_at=base,
            request_id="request-recovery",
            completion_id="chatcmpl-recovery",
        )
    )
    return validator


def assert_rejected(test: unittest.TestCase, operation):
    assert GATE is not None
    with test.assertRaises((GATE.GateError, GATE.COL.CollectorError)):
        operation()


@unittest.skipIf(GATE is None, "direct cancellation gate is being implemented")
class PhaseContractTests(unittest.TestCase):
    def test_frozen_phase_order_and_specs(self):
        self.assertEqual(tuple(GATE.PHASE_ORDER), EXPECTED_PHASES)
        self.assertEqual(tuple(GATE.PHASE_SPECS), EXPECTED_PHASES)
        GATE.validate_phase_order(EXPECTED_PHASES)

        for mutation in (
            EXPECTED_PHASES[:-1],
            tuple(reversed(EXPECTED_PHASES)),
            EXPECTED_PHASES + (EXPECTED_PHASES[-1],),
            EXPECTED_PHASES[:-1] + ("not-a-phase",),
        ):
            with self.subTest(mutation=mutation):
                assert_rejected(
                    self,
                    lambda mutation=mutation: GATE.validate_phase_order(mutation),
                )

    def test_each_target_and_recovery_trace_accepts_exact_contract(self):
        for phase in EXPECTED_PHASES:
            with self.subTest(phase=phase):
                valid_target_validator(phase).finalize()
        valid_recovery_validator().finalize()

    def _overshoot_operation(self, phase: str):
        prompt_tokens = TARGET_PROMPT_TOKENS[phase]
        validator = GATE.CancellationTraceValidator(
            GATE.PHASE_SPECS[phase], prompt_tokens
        )
        validator.consume(admitted(1_000, prompt_tokens, 512))
        validator.consume(started(1_001, prompt_tokens))
        if phase == "after_started_before_progress":
            validator.mark_close(1_001)
            validator.consume(progress(1_002, 128, prompt_tokens))
            cancel_time = 1_003
        elif phase == "prefill_after_128":
            validator.consume(progress(1_002, 128, prompt_tokens))
            validator.mark_close(1_002)
            validator.consume(progress(1_003, 256, prompt_tokens))
            cancel_time = 1_004
        elif phase == "prefill_after_2048":
            for index, processed in enumerate(range(128, 2049, 128), start=2):
                validator.consume(progress(1_000 + index, processed, prompt_tokens))
            validator.mark_close(1_017)
            validator.consume(first_token(1_018))
            cancel_time = 1_019
        else:
            validator.consume(progress(1_002, prompt_tokens, prompt_tokens))
            validator.consume(first_token(1_003))
            validator.observe_content(1_004, 0)
            validator.mark_close(1_004)
            validator.observe_content(1_005, 1)
            cancel_time = 1_006
        validator.consume(cancel_requested(cancel_time))
        validator.consume(
            released(
                cancel_time + 1,
                prompt_tokens,
                cancelled=True,
                completion_tokens=1 if phase == "decode_after_first_content" else 0,
            )
        )
        validator.finalize()

    def test_trigger_overshoot_progress_first_token_and_content_fail_closed(self):
        for phase in EXPECTED_PHASES:
            with self.subTest(phase=phase):
                assert_rejected(
                    self, lambda phase=phase: self._overshoot_operation(phase)
                )

    def test_decode_requires_first_nonempty_content_before_close(self):
        def operation():
            validator = GATE.CancellationTraceValidator(
                GATE.PHASE_SPECS["decode_after_first_content"], 32
            )
            validator.consume(admitted(1_000, 32, 512))
            validator.consume(started(1_001, 32))
            validator.consume(progress(1_002, 32, 32))
            validator.consume(first_token(1_003))
            validator.mark_close(1_004)
            validator.consume(cancel_requested(1_005))
            validator.consume(
                released(
                    1_006,
                    32,
                    cancelled=True,
                    completion_tokens=1,
                )
            )
            validator.finalize()

        assert_rejected(self, operation)

    def test_decode_allows_multiple_content_objects_from_triggering_raw_chunk(self):
        validator = GATE.CancellationTraceValidator(
            GATE.PHASE_SPECS["decode_after_first_content"], 32
        )
        validator.consume(admitted(1_000, 32, 512))
        validator.consume(started(1_001, 32))
        validator.consume(progress(1_002, 32, 32))
        validator.consume(first_token(1_003))
        validator.observe_content(1_004, 0)
        validator.observe_content(1_004, 0)
        validator.mark_close(1_004)
        validator.consume(cancel_requested(1_005))
        validator.consume(
            released(
                1_006,
                32,
                cancelled=True,
                completion_tokens=1,
            )
        )
        validator.finalize()

    def test_explicit_close_cannot_precede_trigger_timestamp(self):
        def operation():
            validator = GATE.CancellationTraceValidator(
                GATE.PHASE_SPECS["after_started_before_progress"], 3584
            )
            validator.consume(admitted(1_000, 3584, 512))
            validator.consume(started(1_010, 3584))
            validator.mark_close(1_009)
            validator.consume(cancel_requested(1_011))
            validator.consume(
                released(
                    1_012,
                    3584,
                    cancelled=True,
                    completion_tokens=0,
                )
            )
            validator.finalize()

        assert_rejected(self, operation)

    def test_cancel_reason_release_reset_and_five_second_deadline_are_exact(self):
        def mutate(field: str):
            validator = GATE.CancellationTraceValidator(
                GATE.PHASE_SPECS["after_started_before_progress"], 3584
            )
            validator.consume(admitted(1_000, 3584, 512))
            validator.consume(started(1_001, 3584))
            validator.mark_close(1_001)
            cancellation = cancel_requested(1_002)
            release = released(
                1_003,
                3584,
                cancelled=True,
                completion_tokens=0,
            )
            if field == "reason":
                cancellation["reason"] = "server_shutdown"
            elif field == "cancel_reason":
                release["cancel_reason"] = "server_shutdown"
            elif field == "reset_complete":
                release["reset_complete"] = False
            validator.consume(cancellation)
            validator.consume(release)
            validator.finalize()

        for field in ("reason", "cancel_reason", "reset_complete"):
            with self.subTest(field=field):
                assert_rejected(self, lambda field=field: mutate(field))

        def late_release():
            validator = GATE.CancellationTraceValidator(
                GATE.PHASE_SPECS["after_started_before_progress"], 3584
            )
            validator.consume(admitted(1_000, 3584, 512))
            validator.consume(started(1_001, 3584))
            validator.mark_close(1_001)
            validator.consume(cancel_requested(1_002))
            validator.consume(
                released(
                    1_002 + 5_000_000_001,
                    3584,
                    cancelled=True,
                    completion_tokens=0,
                )
            )
            validator.finalize()

        assert_rejected(self, late_release)

        boundary = GATE.CancellationTraceValidator(
            GATE.PHASE_SPECS["after_started_before_progress"], 3584
        )
        boundary.consume(admitted(1_000, 3584, 512))
        boundary.consume(started(1_001, 3584))
        boundary.mark_close(1_001)
        boundary.consume(cancel_requested(1_002))
        boundary.consume(
            released(
                1_002 + 5_000_000_000,
                3584,
                cancelled=True,
                completion_tokens=0,
            )
        )
        boundary.finalize()

    def test_release_duration_arithmetic_is_exact(self):
        validator = GATE.CancellationTraceValidator(
            GATE.PHASE_SPECS["after_started_before_progress"], 3584
        )
        validator.consume(admitted(1_000, 3584, 512))
        validator.consume(started(1_001, 3584))
        validator.mark_close(1_001)
        validator.consume(cancel_requested(1_002))
        release = released(1_003, 3584, cancelled=True, completion_tokens=0)
        release["start_to_release_ns"] += 1
        assert_rejected(self, lambda: validator.consume(release))

        recovery = GATE.RecoveryTraceValidator(32)
        recovery.consume(
            admitted(
                10_000,
                32,
                2,
                request_id="request-recovery",
                completion_id="chatcmpl-recovery",
            )
        )
        recovery.consume(
            started(
                10_001,
                32,
                request_id="request-recovery",
                completion_id="chatcmpl-recovery",
            )
        )
        recovery.consume(
            progress(
                10_002,
                32,
                32,
                request_id="request-recovery",
                completion_id="chatcmpl-recovery",
            )
        )
        recovery.consume(
            first_token(
                10_003,
                request_id="request-recovery",
                completion_id="chatcmpl-recovery",
            )
        )
        release = released(
            10_004,
            32,
            cancelled=False,
            completion_tokens=2,
            admitted_at=10_000,
            request_id="request-recovery",
            completion_id="chatcmpl-recovery",
        )
        release["admit_to_release_ns"] += 1
        assert_rejected(self, lambda: recovery.consume(release))

    def test_recovery_first_token_requires_complete_prompt_progress(self):
        validator = GATE.RecoveryTraceValidator(32)
        validator.consume(
            admitted(
                10_000,
                32,
                2,
                request_id="request-recovery",
                completion_id="chatcmpl-recovery",
            )
        )
        validator.consume(
            started(
                10_001,
                32,
                request_id="request-recovery",
                completion_id="chatcmpl-recovery",
            )
        )
        validator.consume(
            progress(
                10_002,
                16,
                32,
                request_id="request-recovery",
                completion_id="chatcmpl-recovery",
            )
        )
        assert_rejected(
            self,
            lambda: validator.consume(
                first_token(
                    10_003,
                    request_id="request-recovery",
                    completion_id="chatcmpl-recovery",
                )
            ),
        )

    def test_recovery_is_exact_length_two_and_not_cancelled(self):
        def wrong_release():
            validator = GATE.RecoveryTraceValidator(32)
            validator.consume(
                admitted(
                    10_000,
                    32,
                    2,
                    request_id="request-recovery",
                    completion_id="chatcmpl-recovery",
                )
            )
            validator.consume(
                started(
                    10_001,
                    32,
                    request_id="request-recovery",
                    completion_id="chatcmpl-recovery",
                )
            )
            validator.consume(
                progress(
                    10_002,
                    32,
                    32,
                    request_id="request-recovery",
                    completion_id="chatcmpl-recovery",
                )
            )
            validator.consume(
                first_token(
                    10_003,
                    request_id="request-recovery",
                    completion_id="chatcmpl-recovery",
                )
            )
            validator.consume(
                released(
                    10_004,
                    32,
                    cancelled=False,
                    completion_tokens=1,
                    request_id="request-recovery",
                    completion_id="chatcmpl-recovery",
                    admitted_at=10_000,
                )
            )
            validator.finalize()

        assert_rejected(self, wrong_release)


@unittest.skipIf(GATE is None, "direct cancellation gate is being implemented")
class RunOrderingAndCommandsTests(unittest.TestCase):
    @staticmethod
    def _consume_target(run, phase: str, base: int, index: int):
        prompt = TARGET_PROMPT_TOKENS[phase]
        request = f"request-target-{index}"
        completion = f"chatcmpl-target-{index}"
        run.begin_target(phase)
        run.consume(
            admitted(
                base,
                prompt,
                512,
                request_id=request,
                completion_id=completion,
            )
        )
        run.consume(
            started(
                base + 1,
                prompt,
                request_id=request,
                completion_id=completion,
            )
        )
        timestamp = base + 1
        if phase == "prefill_after_128":
            timestamp += 1
            run.consume(
                progress(
                    timestamp,
                    128,
                    prompt,
                    request_id=request,
                    completion_id=completion,
                )
            )
        elif phase == "prefill_after_2048":
            for processed in range(128, 2049, 128):
                timestamp += 1
                run.consume(
                    progress(
                        timestamp,
                        processed,
                        prompt,
                        request_id=request,
                        completion_id=completion,
                    )
                )
        elif phase == "decode_after_first_content":
            timestamp += 1
            run.consume(
                progress(
                    timestamp,
                    prompt,
                    prompt,
                    request_id=request,
                    completion_id=completion,
                )
            )
            timestamp += 1
            run.consume(
                first_token(
                    timestamp,
                    request_id=request,
                    completion_id=completion,
                )
            )
            timestamp += 1
            assert run.active is not None
            run.active.observe_content(timestamp, 0)
        assert run.active is not None
        run.active.mark_close(timestamp)
        timestamp += 1
        run.consume(
            cancel_requested(
                timestamp,
                request_id=request,
                completion_id=completion,
                admitted_at=base,
            )
        )
        timestamp += 1
        run.consume(
            released(
                timestamp,
                prompt,
                cancelled=True,
                completion_tokens=1 if phase == "decode_after_first_content" else 0,
                admitted_at=base,
                request_id=request,
                completion_id=completion,
            )
        )
        run.complete_active()

    @staticmethod
    def _consume_recovery(run, phase: str, base: int, index: int):
        request = f"request-recovery-{index}"
        completion = f"chatcmpl-recovery-{index}"
        run.begin_recovery(phase)
        run.consume(
            admitted(
                base,
                32,
                2,
                request_id=request,
                completion_id=completion,
            )
        )
        run.consume(
            started(
                base + 1,
                32,
                request_id=request,
                completion_id=completion,
            )
        )
        run.consume(
            progress(
                base + 2,
                32,
                32,
                request_id=request,
                completion_id=completion,
            )
        )
        run.consume(
            first_token(
                base + 3,
                request_id=request,
                completion_id=completion,
            )
        )
        run.consume(
            released(
                base + 4,
                32,
                cancelled=False,
                completion_tokens=2,
                admitted_at=base,
                request_id=request,
                completion_id=completion,
            )
        )
        run.complete_active()

    def _complete_run(self):
        run = GATE.DirectCancelRunValidator()
        for index, phase in enumerate(EXPECTED_PHASES):
            base = 100_000 + index * 100
            self._consume_target(run, phase, base, index)
            self._consume_recovery(run, phase, base + 40, index)
        run.finalize()
        return run

    def test_exactly_eight_requests_are_target_recovery_pairs(self):
        run = self._complete_run()
        if hasattr(run, "max_active"):
            self.assertEqual(run.max_active, 1)

    def test_overlap_wrong_phase_recovery_before_release_and_extra_request_reject(self):
        def overlap():
            run = GATE.DirectCancelRunValidator()
            run.begin_target(EXPECTED_PHASES[0])
            run.begin_recovery(EXPECTED_PHASES[0])

        def wrong_phase():
            run = GATE.DirectCancelRunValidator()
            run.begin_target(EXPECTED_PHASES[1])

        def extra_request():
            run = self._complete_run()
            run.begin_target(EXPECTED_PHASES[0])

        for operation in (overlap, wrong_phase, extra_request):
            with self.subTest(operation=operation.__name__):
                assert_rejected(self, operation)

    def test_recovery_admission_timestamp_must_follow_target_release(self):
        run = GATE.DirectCancelRunValidator()
        self._consume_target(run, EXPECTED_PHASES[0], 100_000, 0)
        run.begin_recovery(EXPECTED_PHASES[0])
        assert_rejected(
            self,
            lambda: run.consume(
                admitted(
                    100_003,
                    32,
                    2,
                    request_id="request-recovery-0",
                    completion_id="chatcmpl-recovery-0",
                )
            ),
        )

    def test_request_and_close_commands_match_actual_client_protocol(self):
        body = b'{"model":"ullm-qwen3-14b-sq8","stream":true}'
        request = GATE.build_request_command("phase-0-target", body, True)
        self.assertEqual(
            request,
            {
                "schema_version": CLIENT.COMMAND_SCHEMA,
                "command": "request",
                "request_key": "phase-0-target",
                "method": "POST",
                "target": "/v1/chat/completions",
                "body_base64": base64.b64encode(body).decode("ascii"),
                "authorization_mode": "valid_bearer",
                "close_on_first_nonempty_sse_content": True,
            },
        )
        self.assertEqual(
            GATE.build_close_command("phase-0-target"),
            {
                "schema_version": CLIENT.COMMAND_SCHEMA,
                "command": "close",
                "request_key": "phase-0-target",
            },
        )


@unittest.skipIf(GATE is None, "direct cancellation gate is being implemented")
class RawSseBoundaryTests(unittest.TestCase):
    def test_split_content_is_attributed_to_the_completing_raw_chunk(self):
        parser = GATE.SseParser()
        parser.feed(
            GATE.HttpChunk(
                0,
                ROLE_EVENT + b'data: {"choices":[{"delta":{"con',
                1_000,
            )
        )
        parser.feed(
            GATE.HttpChunk(
                1,
                b'tent":"one"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"two"}}]}\n\n',
                1_001,
            )
        )
        items = parser.finish(None, allow_incomplete=False)
        content = GATE.nonempty_content_items(items)

        self.assertEqual(len(content), 2)
        self.assertEqual([item.chunk_index for item in content], [1, 1])
        self.assertEqual(
            [item.observed_monotonic_ns for item in content], [1_001, 1_001]
        )

    def test_content_in_a_later_chunk_remains_a_distinct_overshoot_boundary(self):
        parser = GATE.SseParser()
        parser.feed(GATE.HttpChunk(0, CONTENT_EVENT, 1_000))
        parser.feed(GATE.HttpChunk(1, CONTENT_EVENT, 1_001))
        content = GATE.nonempty_content_items(
            parser.finish(None, allow_incomplete=False)
        )
        self.assertEqual([item.chunk_index for item in content], [0, 1])


@unittest.skipIf(GATE is None, "direct cancellation gate is being implemented")
class EvidenceHttpProtocolValidationTests(unittest.TestCase):
    def _client_and_events(self):
        body = b'{"model":"test","stream":true}'
        plan = GATE.HttpPlan(
            request_key="target-request",
            phase=EXPECTED_PHASES[0],
            role="target",
            body=body,
            auto_close=False,
        )
        client = GATE.EvidenceHttpClient([], mock.Mock(), mock.Mock())
        client.active = plan
        request = {
            "schema_version": GATE.HTTP_EVENT_SCHEMA,
            "event": "http_request",
            "request_key": plan.request_key,
            "method": "POST",
            "target": GATE.HTTP_TARGET,
            "headers": {
                "content_type": "application/json",
                "content_length": len(body),
                "authorization_mode": "valid_bearer",
            },
            "body_base64": base64.b64encode(body).decode("ascii"),
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "body_bytes": len(body),
            "connect_completed_monotonic_ns": 1_000,
            "write_started_monotonic_ns": 1_001,
            "last_body_byte_sent_monotonic_ns": 1_002,
        }
        start = {
            "schema_version": GATE.HTTP_EVENT_SCHEMA,
            "event": "http_response_start",
            "request_key": plan.request_key,
            "status": 200,
            "headers": [["Content-Type", "text/event-stream"]],
            "observed_monotonic_ns": 1_003,
        }
        end = {
            "schema_version": GATE.HTTP_EVENT_SCHEMA,
            "event": "http_response_end",
            "request_key": plan.request_key,
            "outcome": "client_closed",
            "error": None,
            "body_bytes": 0,
            "body_sha256": hashlib.sha256(b"").hexdigest(),
            "observed_monotonic_ns": 1_004,
        }
        return client, request, start, end

    def test_request_start_end_sequence_is_accepted(self):
        client, request, start, end = self._client_and_events()
        client._read_event = mock.Mock(side_effect=[request, start, end])
        result = client.finish(time.monotonic_ns() + 1_000_000_000)
        self.assertEqual(result.status, 200)
        self.assertEqual(result.outcome, "client_closed")

    def test_explicit_close_may_win_before_response_start(self):
        client, request, _start, end = self._client_and_events()
        client._read_event = mock.Mock(side_effect=[request, end])
        result = client.finish(time.monotonic_ns() + 1_000_000_000)
        self.assertIsNone(result.status)
        self.assertEqual(result.outcome, "client_closed")
        active = GATE.CancellationTraceValidator(
            GATE.PHASE_SPECS["after_started_before_progress"]
        )
        active.consume(admitted(1_000, 3584, 512))
        active.consume(started(1_001, 3584))
        active.mark_close(1_001)
        active.consume(cancel_requested(1_002))
        active.consume(
            released(
                1_003,
                3584,
                cancelled=True,
                completion_tokens=0,
            )
        )
        GATE.DirectCancelGate._validate_target_http(mock.Mock(), active, result)
        recovery = valid_recovery_validator()
        assert_rejected(
            self,
            lambda: GATE.DirectCancelGate._validate_recovery_http(recovery, result),
        )

    def test_decode_content_requires_lifecycle_completion_id(self):
        active = GATE.CancellationTraceValidator(
            GATE.PHASE_SPECS["decode_after_first_content"]
        )
        active.consume(admitted(1_000, 32, 512))
        active.consume(started(1_001, 32))
        active.consume(progress(1_002, 32, 32))
        active.consume(first_token(1_003))
        active.consume(cancel_requested(1_005))
        active.consume(
            released(
                1_006,
                32,
                cancelled=True,
                completion_tokens=1,
            )
        )
        value = {"choices": [{"delta": {"content": "x"}}]}
        raw = GATE.compact_json(value)
        chunk = GATE.HttpChunk(0, b"data: " + raw + b"\n\n", 1_004)
        item = GATE.SseItem(raw, value, False, 0, 1_004)
        result = GATE.HttpResult(
            status=200,
            outcome="client_closed",
            response_body=chunk.raw,
            chunks=(chunk,),
            items=(item,),
            response_end_monotonic_ns=1_005,
        )

        assert_rejected(
            self,
            lambda: GATE.DirectCancelGate._validate_target_http(
                mock.Mock(), active, result
            ),
        )


@unittest.skipIf(GATE is None, "direct cancellation gate is being implemented")
class AtomicOutputTests(unittest.TestCase):
    def test_abort_removes_staged_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "direct-cancel-evidence"
            transaction = GATE.AtomicRunDirectory(output)
            self.assertFalse(output.exists())
            self.assertTrue(transaction.stage.is_dir())
            (transaction.stage / "partial.raw.jsonl").write_bytes(b"partial\n")

            transaction.abort()

            self.assertFalse(output.exists())
            self.assertFalse(transaction.stage.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_post_rename_fsync_failure_is_removed_by_abort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "direct-cancel-evidence"
            transaction = GATE.AtomicRunDirectory(output)
            (transaction.stage / "summary.json").write_bytes(b"{}\n")
            real_fsync = GATE.os.fsync
            calls = 0

            def fail_parent_fsync(descriptor):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected parent fsync failure")
                return real_fsync(descriptor)

            with mock.patch.object(GATE.os, "fsync", side_effect=fail_parent_fsync):
                with self.assertRaises((GATE.GateError, OSError)):
                    transaction.publish()
            transaction.abort()

            self.assertFalse(output.exists())
            self.assertFalse(transaction.stage.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_publish_is_single_rename_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "direct-cancel-evidence"
            transaction = GATE.AtomicRunDirectory(output)
            evidence = b'{"result":"passed"}\n'
            (transaction.stage / "summary.json").write_bytes(evidence)
            transaction.publish()

            self.assertFalse(transaction.stage.exists())
            self.assertEqual((output / "summary.json").read_bytes(), evidence)

            assert_rejected(self, lambda: GATE.AtomicRunDirectory(output))

    def test_publish_refuses_destination_created_after_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "direct-cancel-evidence"
            transaction = GATE.AtomicRunDirectory(output)
            (transaction.stage / "summary.json").write_bytes(b"{}\n")
            output.mkdir(mode=0o700)

            try:
                assert_rejected(self, transaction.publish)
            finally:
                transaction.abort()
                if output.exists():
                    shutil.rmtree(output)

    def test_failure_cleanup_never_leaves_a_pass_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "direct-cancel-evidence"
            transaction = GATE.AtomicRunDirectory(output)
            try:
                (transaction.stage / "summary.json").write_text(
                    '{"status":"passed"}', encoding="ascii"
                )
                raise RuntimeError("injected failure")
            except RuntimeError:
                transaction.abort()

            self.assertFalse(output.exists())
            self.assertFalse(transaction.stage.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_raw_writer_rejects_secret_without_retaining_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "direct-cancel-evidence"
            transaction = GATE.AtomicRunDirectory(output)
            secret = b"direct-gate-secret-should-never-be-retained"
            writer = GATE.RawWriter(
                transaction.stage / "http.raw.jsonl", GATE.COL.SecretGuard(secret)
            )
            try:
                with self.assertRaises(GATE.COL.CollectorError) as captured:
                    writer.write(b'{"credential":"' + secret + b'"}', "HTTP evidence")
                self.assertNotIn(secret.decode("ascii"), str(captured.exception))
            finally:
                writer.abort()
                transaction.abort()

            self.assertFalse(output.exists())
            self.assertFalse(transaction.stage.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_sealed_raw_and_json_artifact_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guard = GATE.COL.SecretGuard(b"artifact-test-secret-49f2")
            writer = GATE.RawWriter(root / "raw.jsonl", guard)
            writer.write(b'{"value":1}', "raw fixture")
            writer.close()
            GATE.verify_raw_writer(writer)
            (root / "raw.jsonl").write_bytes(b'{"value":2}\n')
            assert_rejected(self, lambda: GATE.verify_raw_writer(writer))

            value = {"schema_version": GATE.GATE_SCHEMA, "value": 1}
            GATE.write_json_file(root / "view.json", value, guard)
            GATE.verify_json_file(root / "view.json", value)
            (root / "view.json").write_bytes(
                GATE.compact_json({**value, "value": 2}) + b"\n"
            )
            assert_rejected(
                self, lambda: GATE.verify_json_file(root / "view.json", value)
            )


@unittest.skipIf(GATE is None, "direct cancellation gate is being implemented")
class ObserverJournalCorrelationTests(unittest.TestCase):
    gateway_pid = 12_345

    def _evidence(self):
        events = [
            admitted(1_000, 3584, 512),
            started(1_001, 3584),
        ]
        observer = []
        journal = []
        for index, event in enumerate(events):
            raw = GATE.compact_json(event)
            observer.append(
                GATE.CorrelatedObserverRecord(
                    raw_payload=raw,
                    event=copy.deepcopy(event),
                    received_monotonic_ns=event["observed_monotonic_ns"] + 1,
                    sender_pid=self.gateway_pid,
                    sender_uid=1_000,
                    sender_gid=1_000,
                )
            )
            journal.append(
                {
                    "__CURSOR": f"cursor-{index}",
                    "__MONOTONIC_TIMESTAMP": str(1_000_000 + index),
                    "_PID": str(self.gateway_pid),
                    "MESSAGE": raw.decode("ascii"),
                }
            )
        return observer, journal

    def test_exact_payload_pid_cursor_and_sequence_are_retained(self):
        observer, journal = self._evidence()
        correlated = GATE.correlate_records(observer, journal, self.gateway_pid)
        self.assertEqual([item["sequence"] for item in correlated], [0, 1])
        self.assertEqual(
            [item["cursor"] for item in correlated], ["cursor-0", "cursor-1"]
        )
        self.assertEqual(correlated[0]["journal_pid"], str(self.gateway_pid))
        self.assertEqual(
            correlated[0]["payload_sha256"],
            hashlib.sha256(observer[0].raw_payload).hexdigest(),
        )

    def test_payload_bytes_pid_duplicate_cursor_and_counts_fail_closed(self):
        def wrong_bytes():
            observer, journal = self._evidence()
            original = observer[0]
            observer[0] = GATE.CorrelatedObserverRecord(
                raw_payload=json.dumps(original.event, separators=(", ", ": ")).encode(
                    "ascii"
                ),
                event=original.event,
                received_monotonic_ns=original.received_monotonic_ns,
                sender_pid=original.sender_pid,
                sender_uid=original.sender_uid,
                sender_gid=original.sender_gid,
            )
            GATE.correlate_records(observer, journal, self.gateway_pid)

        def wrong_observer_pid():
            observer, journal = self._evidence()
            original = observer[0]
            observer[0] = GATE.CorrelatedObserverRecord(
                raw_payload=original.raw_payload,
                event=original.event,
                received_monotonic_ns=original.received_monotonic_ns,
                sender_pid=self.gateway_pid + 1,
                sender_uid=original.sender_uid,
                sender_gid=original.sender_gid,
            )
            GATE.correlate_records(observer, journal, self.gateway_pid)

        def wrong_journal_pid():
            observer, journal = self._evidence()
            journal[0]["_PID"] = str(self.gateway_pid + 1)
            GATE.correlate_records(observer, journal, self.gateway_pid)

        def duplicate_cursor():
            observer, journal = self._evidence()
            journal[1]["__CURSOR"] = journal[0]["__CURSOR"]
            GATE.correlate_records(observer, journal, self.gateway_pid)

        def extra_observer():
            observer, journal = self._evidence()
            observer.append(observer[-1])
            GATE.correlate_records(observer, journal, self.gateway_pid)

        def extra_journal_lifecycle():
            observer, journal = self._evidence()
            extra = dict(journal[-1])
            extra["__CURSOR"] = "cursor-extra"
            journal.append(extra)
            GATE.correlate_records(observer, journal, self.gateway_pid)

        for operation in (
            wrong_bytes,
            wrong_observer_pid,
            wrong_journal_pid,
            duplicate_cursor,
            extra_observer,
            extra_journal_lifecycle,
        ):
            with self.subTest(operation=operation.__name__):
                assert_rejected(self, operation)

    def test_raw_journal_boot_unit_and_lifecycle_pid_are_fixed(self):
        boot_id = "c" * 32
        event_raw = GATE.compact_json(admitted(1_000, 3584, 512))
        baseline = {
            "__CURSOR": "cursor-fixed-identity",
            "__MONOTONIC_TIMESTAMP": "1000",
            "_BOOT_ID": boot_id,
            "_PID": str(self.gateway_pid),
            "_SYSTEMD_UNIT": GATE.SERVICE_UNIT,
            "PRIORITY": "6",
            "MESSAGE": event_raw.decode("ascii"),
        }

        class Source:
            def __init__(self, record):
                self.record = record

            def poll(self):
                if self.record is None:
                    return []
                raw = GATE.compact_json(self.record)
                self.record = None
                return [raw]

        def consume(record):
            writer = mock.Mock()
            capture = GATE.JournalCapture(
                boot_id,
                self.gateway_pid,
                writer,
                GATE.COL.SecretGuard(b"journal-test-secret-9f72a4"),
            )
            capture.source = Source(record)
            capture.poll()
            return writer

        writer = consume(dict(baseline))
        writer.write.assert_called_once()

        mutations = (
            ("_BOOT_ID", "d" * 32),
            ("_SYSTEMD_UNIT", "other.service"),
            ("_PID", str(self.gateway_pid + 1)),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                record = dict(baseline)
                record[field] = value
                assert_rejected(self, lambda record=record: consume(record))

    def test_post_observer_close_journal_only_lifecycle_is_rejected(self):
        observer, journal = self._evidence()
        late = dict(journal[-1])
        late["__CURSOR"] = "cursor-after-observer-close"
        late_event = cancel_requested(1_002)
        late["MESSAGE"] = GATE.compact_json(late_event).decode("ascii")
        journal.append(late)

        assert_rejected(
            self,
            lambda: GATE.correlate_records(observer, journal, self.gateway_pid),
        )

        diagnostic = dict(late)
        diagnostic["MESSAGE"] = "late non-lifecycle diagnostic"
        self.assertEqual(
            len(
                GATE.correlate_records(
                    observer, [*journal[:-1], diagnostic], self.gateway_pid
                )
            ),
            len(observer),
        )


@unittest.skipIf(GATE is None, "direct cancellation gate is being implemented")
class ObserverSocketCredentialTests(unittest.TestCase):
    def _receive(
        self,
        root: Path,
        name: str,
        *,
        expected_pid: int,
        expected_uid: int,
        expected_gid: int,
    ):
        path = root / name
        guard = GATE.COL.SecretGuard(b"observer-test-secret-794d23")
        observer = GATE.COL.LifecycleObserver(
            path,
            guard,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        )
        sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        observer.open()
        try:
            raw = GATE.compact_json(admitted(1_000, 3584, 512))
            sender.sendto(raw, os.fspath(path))
            return observer.receive(
                time.monotonic_ns() + 2_000_000_000,
                expected_sender_pid=expected_pid,
            )
        finally:
            sender.close()
            observer.close()

    def test_kernel_pid_uid_and_gid_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current_pid = os.getpid()
            current_uid = os.geteuid()
            current_gid = os.getegid()
            record = self._receive(
                root,
                "valid.sock",
                expected_pid=current_pid,
                expected_uid=current_uid,
                expected_gid=current_gid,
            )
            self.assertEqual(record.sender_pid, current_pid)
            self.assertEqual(record.sender_uid, current_uid)
            self.assertEqual(record.sender_gid, current_gid)

            for label, pid, uid, gid in (
                ("pid", current_pid + 1, current_uid, current_gid),
                ("uid", current_pid, current_uid + 1, current_gid),
                ("gid", current_pid, current_uid, current_gid + 1),
            ):
                with self.subTest(label=label):
                    with self.assertRaises(GATE.COL.CollectorError):
                        self._receive(
                            root,
                            f"wrong-{label}.sock",
                            expected_pid=pid,
                            expected_uid=uid,
                            expected_gid=gid,
                        )


@unittest.skipIf(GATE is None, "direct cancellation gate is being implemented")
class ServiceIdentityTests(unittest.TestCase):
    def _identity(self):
        return GATE.ServiceIdentity(
            unit=GATE.SERVICE_UNIT,
            user="homelab1",
            uid=1_000,
            gid=1_000,
            control_group="/system.slice/ullm-openai.service",
            gateway_pid=12_000,
            gateway_starttime_ticks=100_000,
            worker_pid=12_001,
            worker_starttime_ticks=100_001,
            n_restarts=2,
            boot_id="a" * 32,
        )

    def test_pid_uid_boot_service_worker_and_restart_identity_cannot_change(self):
        baseline = self._identity()
        with mock.patch.object(GATE, "capture_service_identity", return_value=baseline):
            GATE.require_service_identity(baseline)

        mutations = {
            "unit": "other.service",
            "user": "other-user",
            "uid": 1_001,
            "gid": 1_001,
            "control_group": "/system.slice/other.service",
            "gateway_pid": 12_100,
            "gateway_starttime_ticks": 100_100,
            "worker_pid": 12_101,
            "worker_starttime_ticks": 100_101,
            "n_restarts": 3,
            "boot_id": "b" * 32,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                changed = dataclasses.replace(baseline, **{field: value})
                with mock.patch.object(
                    GATE, "capture_service_identity", return_value=changed
                ):
                    assert_rejected(
                        self, lambda: GATE.require_service_identity(baseline)
                    )


class ProtocolHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    hold_sent = threading.Event()
    release_hold = threading.Event()

    def log_message(self, _format, *args):
        del args

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.path == "/hold":
            self._stream_headers()
            self.wfile.write(b": hold\n\n")
            self.wfile.flush()
            type(self).hold_sent.set()
            type(self).release_hold.wait(3.0)
            return
        if self.path == "/content":
            self._stream_headers()
            for part in (ROLE_EVENT, CONTENT_EVENT, DONE_EVENT):
                try:
                    self.wfile.write(part)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                time.sleep(0.03)
            return
        if self.path == "/recovery":
            payload = CONTENT_EVENT + USAGE_EVENT + DONE_EVENT
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            return
        self.send_error(404)

    def _stream_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True


class ActualClient:
    def __init__(self, base_url: str, key_file: Path):
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(CLIENT_PATH),
                "--base-url",
                base_url,
                "--api-key-file",
                str(key_file),
                "--socket-timeout-seconds",
                "5",
                "--read-chunk-bytes",
                "8",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self.events: queue.Queue[dict[str, object]] = queue.Queue()
        self.raw_output: list[bytes] = []
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self._expect("ready")

    def _read(self):
        assert self.process.stdout is not None
        for raw in self.process.stdout:
            self.raw_output.append(raw)
            self.events.put(json.loads(raw))

    def send(self, value: dict[str, object]):
        assert self.process.stdin is not None
        raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
        self.process.stdin.write(raw + b"\n")
        self.process.stdin.flush()

    def _expect(self, name: str, timeout: float = 3.0):
        event = self.events.get(timeout=timeout)
        if event["event"] != name:
            raise AssertionError(f"expected {name}, got {event}")
        return event

    def through_end(self, timeout: float = 5.0):
        result = []
        deadline = time.monotonic() + timeout
        while True:
            event = self.events.get(timeout=max(0.01, deadline - time.monotonic()))
            result.append(event)
            if event["event"] == "http_response_end":
                return result

    def close(self):
        if self.process.poll() is None:
            self.send(
                {
                    "schema_version": CLIENT.COMMAND_SCHEMA,
                    "command": "shutdown",
                }
            )
            self._expect("shutdown_complete")
            self.process.wait(timeout=3.0)
        self._close_streams()

    def terminate(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=3.0)
        self._close_streams()

    def _close_streams(self):
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
        self.reader.join(timeout=1.0)


def request_command(key: str, target: str, *, auto_close: bool):
    body = json.dumps({"model": "test", "stream": True}).encode()
    return {
        "schema_version": CLIENT.COMMAND_SCHEMA,
        "command": "request",
        "request_key": key,
        "method": "POST",
        "target": target,
        "body_base64": base64.b64encode(body).decode("ascii"),
        "authorization_mode": "valid_bearer",
        "close_on_first_nonempty_sse_content": auto_close,
    }


class ActualHttpClientProtocolTests(unittest.TestCase):
    def setUp(self):
        ProtocolHandler.hold_sent = threading.Event()
        ProtocolHandler.release_hold = threading.Event()
        self.temporary = tempfile.TemporaryDirectory()
        self.key_file = Path(self.temporary.name) / "api-key"
        self.secret = b"direct-cancel-test-secret-174a39"
        self.key_file.write_bytes(self.secret + b"\n")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ProtocolHandler)
        self.server.daemon_threads = True
        self.server_thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.server_thread.start()
        self.client = ActualClient(
            f"http://127.0.0.1:{self.server.server_port}", self.key_file
        )

    def tearDown(self):
        ProtocolHandler.release_hold.set()
        self.client.terminate()
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=3.0)
        self.temporary.cleanup()

    def test_explicit_close_recovery_and_first_content_auto_close(self):
        self.client.send(request_command("target-manual", "/hold", auto_close=False))
        self.assertTrue(ProtocolHandler.hold_sent.wait(2.0))
        prefix = []
        while not any(event["event"] == "http_body_chunk" for event in prefix):
            prefix.append(self.client.events.get(timeout=2.0))
        self.client.send(
            {
                "schema_version": CLIENT.COMMAND_SCHEMA,
                "command": "close",
                "request_key": "target-manual",
            }
        )
        manual = prefix + self.client.through_end()
        self.assertEqual(manual[-1]["outcome"], "client_closed")

        self.client.send(request_command("recovery", "/recovery", auto_close=False))
        recovery = self.client.through_end()
        self.assertEqual(recovery[-1]["outcome"], "eof")
        recovery_body = b"".join(
            base64.b64decode(event["body_base64"], validate=True)
            for event in recovery
            if event["event"] == "http_body_chunk"
        )
        self.assertIn(USAGE_EVENT, recovery_body)
        self.assertTrue(recovery_body.endswith(DONE_EVENT))

        self.client.send(request_command("target-auto", "/content", auto_close=True))
        automatic = self.client.through_end()
        automatic_body = b"".join(
            base64.b64decode(event["body_base64"], validate=True)
            for event in automatic
            if event["event"] == "http_body_chunk"
        )
        self.assertIn(ROLE_EVENT, automatic_body)
        self.assertIn(CONTENT_EVENT, automatic_body)
        self.assertNotIn(DONE_EVENT, automatic_body)
        self.assertEqual(automatic[-1]["outcome"], "client_closed")

        self.client.close()
        raw = b"".join(self.client.raw_output)
        self.assertNotIn(self.secret, raw)


if __name__ == "__main__":
    unittest.main()
