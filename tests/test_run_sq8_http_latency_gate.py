from __future__ import annotations

import base64
import dataclasses
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "tools" / "run-sq8-http-latency-gate.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GATE = load_module("run_sq8_http_latency_gate", GATE_PATH)


def assert_rejected(test: unittest.TestCase, operation) -> None:
    with test.assertRaises(
        (GATE.LatencyGateError, GATE.DIRECT.GateError, GATE.COL.CollectorError)
    ):
        operation()


def lifecycle(
    name: str,
    timestamp: int,
    *,
    request_id: str = "request-latency",
    completion_id: str = "chatcmpl-latency",
    **fields,
):
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
    spec,
    *,
    request_id="request-latency",
    completion_id="chatcmpl-latency",
):
    return lifecycle(
        "request_admitted",
        timestamp,
        request_id=request_id,
        completion_id=completion_id,
        stream=True,
        prompt_tokens=spec.prompt_tokens,
        max_completion_tokens=spec.max_tokens,
    )


def started(
    timestamp: int,
    spec,
    *,
    request_id="request-latency",
    completion_id="chatcmpl-latency",
):
    return lifecycle(
        "request_started",
        timestamp,
        request_id=request_id,
        completion_id=completion_id,
        stream=True,
        prompt_tokens=spec.prompt_tokens,
        admit_to_start_ns=1,
    )


def progress(
    timestamp: int,
    spec,
    processed: int,
    *,
    request_id="request-latency",
    completion_id="chatcmpl-latency",
):
    return lifecycle(
        "request_progress",
        timestamp,
        request_id=request_id,
        completion_id=completion_id,
        phase="prefill",
        processed_prompt_tokens=processed,
        prompt_tokens=spec.prompt_tokens,
    )


def first_token(
    timestamp: int, *, request_id="request-latency", completion_id="chatcmpl-latency"
):
    return lifecycle(
        "request_first_token",
        timestamp,
        request_id=request_id,
        completion_id=completion_id,
        stream=True,
        completion_tokens=1,
    )


def cancel(
    timestamp: int,
    admitted_at: int,
    *,
    request_id="request-latency",
    completion_id="chatcmpl-latency",
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
    admitted_at: int,
    spec,
    *,
    request_id="request-latency",
    completion_id="chatcmpl-latency",
):
    admit_to_release = timestamp - admitted_at
    cancelled = spec.workload == "ttft"
    return lifecycle(
        "request_released",
        timestamp,
        request_id=request_id,
        completion_id=completion_id,
        stream=True,
        outcome="cancelled" if cancelled else "length",
        cancel_reason="client_disconnect" if cancelled else None,
        prompt_tokens=spec.prompt_tokens,
        completion_tokens=1 if cancelled else 64,
        reset_complete=True,
        admit_to_start_ns=1,
        start_to_release_ns=admit_to_release - 1,
        admit_to_release_ns=admit_to_release,
    )


def consume_prefill(trace, spec, base: int) -> int:
    timestamp = base + 2
    processed = min(128, spec.prompt_tokens)
    while True:
        trace.consume(progress(timestamp, spec, processed))
        timestamp += 1
        if processed == spec.prompt_tokens:
            return timestamp
        processed = min(processed + 128, spec.prompt_tokens)


def sse_item(value, timestamp: int, index: int = 0):
    raw = GATE.compact_json(value)
    return GATE.TimedSseItem(raw, value, False, index, timestamp)


def ttft_observation(
    *,
    completion_id: str = "chatcmpl-latency",
    sent: int = 1_010,
    content_time: int = 1_100,
):
    value = {
        "id": completion_id,
        "choices": [{"delta": {"content": "x"}, "finish_reason": None}],
    }
    raw = b"data: " + GATE.compact_json(value) + b"\n\n"
    chunk = GATE.TimedChunk(0, raw, content_time)
    return GATE.HttpObservation(
        status=200,
        outcome="client_closed",
        request_sent_monotonic_ns=sent,
        response_start_monotonic_ns=sent + 1,
        response_end_monotonic_ns=content_time + 1,
        body=raw,
        chunks=(chunk,),
        items=(sse_item(value, content_time),),
    )


def decode_observation(
    *,
    completion_id: str = "chatcmpl-latency",
    content_count: int = 64,
    interval_ns: int = 10_000_000,
    usage: int = 64,
):
    items = []
    base = 2_000
    for index in range(content_count):
        value = {
            "id": completion_id,
            "choices": [{"delta": {"content": "x"}, "finish_reason": None}],
        }
        items.append(sse_item(value, base + index * interval_ns, index))
    finish = {
        "id": completion_id,
        "choices": [{"delta": {}, "finish_reason": "length"}],
    }
    usage_value = {
        "id": completion_id,
        "choices": [],
        "usage": {"completion_tokens": usage},
    }
    items.append(sse_item(finish, base + content_count * interval_ns, content_count))
    items.append(
        sse_item(usage_value, base + content_count * interval_ns, content_count)
    )
    items.append(
        GATE.TimedSseItem(
            b"[DONE]",
            None,
            True,
            content_count,
            base + content_count * interval_ns,
        )
    )
    return GATE.HttpObservation(
        status=200,
        outcome="eof",
        request_sent_monotonic_ns=1_000,
        response_start_monotonic_ns=1_001,
        response_end_monotonic_ns=base + content_count * interval_ns + 1,
        body=b"body",
        chunks=(GATE.TimedChunk(0, b"body", base),),
        items=tuple(items),
    )


def valid_trace(spec, *, base: int = 1_000):
    trace = GATE.LifecycleTrace(spec)
    trace.consume(admitted(base, spec))
    trace.consume(started(base + 1, spec))
    next_time = consume_prefill(trace, spec, base)
    trace.consume(first_token(next_time))
    if spec.workload == "ttft":
        observation = ttft_observation(sent=base - 10, content_time=next_time + 1)
        trace.consume(cancel(next_time + 2, base))
        trace.consume(released(next_time + 3, base, spec))
    else:
        observation = decode_observation()
        trace.consume(released(next_time + 700_000_000, base, spec))
    return trace, observation


def valid_metric_samples():
    samples = []
    for spec in GATE.SCHEDULE:
        item = {
            "sequence": spec.sequence,
            "case_id": spec.case_id,
            "workload": spec.workload,
            "sample_kind": spec.sample_kind,
            "sample_index": spec.sample_index,
            "fixture_id": spec.fixture_id,
        }
        if spec.workload == "ttft":
            item["ttft_ns"] = 1_000_000_000
        else:
            item["decode_elapsed_ns"] = 630_000_000
            item["decode_intervals_ns"] = [10_000_000] * 63
        samples.append(item)
    return samples


class ScheduleAndFixtureTests(unittest.TestCase):
    def test_schedule_is_exactly_72_serial_cases(self):
        self.assertEqual(len(GATE.SCHEDULE), 72)
        self.assertEqual([item.sequence for item in GATE.SCHEDULE], list(range(1, 73)))
        self.assertEqual(GATE.SCHEDULE[0].case_id, "ttft-exact-p0032-warmup-01")
        self.assertEqual(GATE.SCHEDULE[59].case_id, "ttft-exact-p3584-measured-10")
        self.assertEqual(GATE.SCHEDULE[60].case_id, "decode64-warmup-01")
        self.assertEqual(GATE.SCHEDULE[-1].case_id, "decode64-measured-10")
        self.assertEqual(len({item.case_id for item in GATE.SCHEDULE}), 72)

    def test_reordered_or_missing_schedule_fails(self):
        assert_rejected(self, lambda: GATE.validate_schedule(GATE.SCHEDULE[1:]))
        mutated = list(GATE.SCHEDULE)
        mutated[0], mutated[1] = mutated[1], mutated[0]
        assert_rejected(self, lambda: GATE.validate_schedule(mutated))

    def test_all_frozen_fixtures_load_and_build_exact_request_bodies(self):
        root = ROOT / "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures"
        for fixture_id in GATE.FIXTURE_ORDER:
            fixture = GATE.load_fixture(root / f"{fixture_id}.json", fixture_id)
            self.assertEqual(
                fixture.prompt_tokens, GATE.FIXTURE_IDENTITIES[fixture_id][0]
            )
            body = json.loads(GATE.request_body(fixture, 512))
            self.assertEqual(
                set(body),
                {
                    "model",
                    "messages",
                    "stream",
                    "stream_options",
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "seed",
                },
            )
            self.assertEqual(body["messages"], fixture.messages)
            self.assertEqual(body["stream_options"], {"include_usage": True})

    def test_fixture_hash_mutation_fails(self):
        source = (
            ROOT
            / "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures/exact-p0032.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exact-p0032.json"
            path.write_bytes(source.read_bytes() + b" ")
            assert_rejected(self, lambda: GATE.load_fixture(path, "exact-p0032"))

    def test_request_body_rejects_non_frozen_generation_length(self):
        root = ROOT / "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures"
        fixture = GATE.load_fixture(root / "exact-p0032.json", "exact-p0032")
        assert_rejected(self, lambda: GATE.request_body(fixture, 63))


class TimedSseParserTests(unittest.TestCase):
    def test_split_object_uses_final_raw_chunk_timestamp(self):
        parser = GATE.TimedSseParser()
        parser.feed(GATE.TimedChunk(0, b'data: {"choices":[{"delta":{"con', 100))
        parser.feed(GATE.TimedChunk(1, b'tent":"x"}}]}\n\n', 200))
        items = parser.finish(allow_incomplete=False)
        content = GATE.nonempty_content_items(items)
        self.assertEqual(
            [(item.chunk_index, item.observed_monotonic_ns) for item in content],
            [(1, 200)],
        )

    def test_two_objects_in_one_chunk_share_authoritative_time(self):
        raw = (
            b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
        )
        items = GATE.parse_timed_sse(
            (GATE.TimedChunk(0, raw, 300),), allow_incomplete=False
        )
        content = GATE.nonempty_content_items(items)
        self.assertEqual([item.observed_monotonic_ns for item in content], [300, 300])

    def test_incomplete_trailing_event_is_only_allowed_for_close(self):
        chunk = GATE.TimedChunk(0, b'data: {"choices":', 100)
        self.assertEqual(GATE.parse_timed_sse((chunk,), allow_incomplete=True), ())
        assert_rejected(
            self, lambda: GATE.parse_timed_sse((chunk,), allow_incomplete=False)
        )

    def test_regressed_chunk_index_or_time_fails(self):
        parser = GATE.TimedSseParser()
        parser.feed(GATE.TimedChunk(0, b": keepalive\n", 100))
        assert_rejected(self, lambda: parser.feed(GATE.TimedChunk(2, b"x", 101)))
        parser = GATE.TimedSseParser()
        parser.feed(GATE.TimedChunk(0, b": keepalive\n", 100))
        assert_rejected(self, lambda: parser.feed(GATE.TimedChunk(1, b"x", 99)))

    def test_invalid_json_fails_at_dispatch(self):
        assert_rejected(
            self,
            lambda: GATE.parse_timed_sse(
                (GATE.TimedChunk(0, b"data: {\n\n", 1),), allow_incomplete=False
            ),
        )

    def test_many_raw_chunks_are_streamed_without_synthetic_boundaries(self):
        chunks = tuple(
            GATE.TimedChunk(index, b": keepalive\n", 1_000 + index)
            for index in range(2_048)
        ) + (
            GATE.TimedChunk(
                2_048,
                b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n',
                4_000,
            ),
        )
        items = GATE.parse_timed_sse(chunks, allow_incomplete=False)
        self.assertEqual(len(GATE.nonempty_content_items(items)), 1)
        self.assertEqual(items[0].chunk_index, 2_048)


class HttpSemanticTests(unittest.TestCase):
    def test_ttft_uses_last_body_to_first_content(self):
        spec = GATE.SCHEDULE[0]
        result = GATE.validate_ttft_http(spec, ttft_observation(), "chatcmpl-latency")
        self.assertEqual(result["ttft_ns"], 90)

    def test_ttft_rejects_completion_mismatch_and_later_chunk_overshoot(self):
        spec = GATE.SCHEDULE[0]
        assert_rejected(
            self,
            lambda: GATE.validate_ttft_http(spec, ttft_observation(), "different"),
        )
        observation = ttft_observation()
        value = observation.items[0].value
        assert value is not None
        extra = sse_item(value, 1_101, 1)
        mutated = dataclasses.replace(
            observation,
            chunks=observation.chunks + (GATE.TimedChunk(1, b"x", 1_101),),
            items=observation.items + (extra,),
        )
        assert_rejected(
            self,
            lambda: GATE.validate_ttft_http(spec, mutated, "chatcmpl-latency"),
        )

    def test_ttft_rejects_an_unidentified_non_done_object(self):
        spec = GATE.SCHEDULE[0]
        observation = ttft_observation()
        unidentified = sse_item({"choices": []}, 1_099, 0)
        assert_rejected(
            self,
            lambda: GATE.validate_ttft_http(
                spec,
                dataclasses.replace(
                    observation, items=(unidentified,) + observation.items
                ),
                "chatcmpl-latency",
            ),
        )

    def test_ttft_rejects_done_or_usage_after_auto_close(self):
        spec = GATE.SCHEDULE[0]
        observation = ttft_observation()
        done = GATE.TimedSseItem(b"[DONE]", None, True, 0, 1_100)
        assert_rejected(
            self,
            lambda: GATE.validate_ttft_http(
                spec,
                dataclasses.replace(observation, items=observation.items + (done,)),
                "chatcmpl-latency",
            ),
        )

    def test_decode_requires_exact_64_usage_length_and_done(self):
        spec = GATE.SCHEDULE[60]
        result = GATE.validate_decode_http(
            spec, decode_observation(), "chatcmpl-latency"
        )
        self.assertEqual(len(result["decode_intervals_ns"]), 63)
        self.assertEqual(result["decode_elapsed_ns"], 630_000_000)
        assert_rejected(
            self,
            lambda: GATE.validate_decode_http(
                spec, decode_observation(content_count=63), "chatcmpl-latency"
            ),
        )
        assert_rejected(
            self,
            lambda: GATE.validate_decode_http(
                spec, decode_observation(usage=63), "chatcmpl-latency"
            ),
        )

    def test_decode_rejects_zero_elapsed_without_inventing_spacing(self):
        spec = GATE.SCHEDULE[60]
        assert_rejected(
            self,
            lambda: GATE.validate_decode_http(
                spec, decode_observation(interval_ns=0), "chatcmpl-latency"
            ),
        )


class LifecycleStateTests(unittest.TestCase):
    def test_valid_ttft_and_decode_lifecycle_complete(self):
        for spec in (GATE.SCHEDULE[0], GATE.SCHEDULE[60]):
            trace, observation = valid_trace(spec)
            result = trace.complete(observation)
            self.assertEqual(result["case_id"], spec.case_id)
            self.assertTrue(result["release_completion_tokens"] in {1, 64})

    def test_progress_skip_or_first_token_before_complete_prefill_fails(self):
        spec = next(item for item in GATE.SCHEDULE if item.fixture_id == "exact-p0512")
        trace = GATE.LifecycleTrace(spec)
        trace.consume(admitted(1_000, spec))
        trace.consume(started(1_001, spec))
        assert_rejected(self, lambda: trace.consume(progress(1_002, spec, 256)))
        trace = GATE.LifecycleTrace(spec)
        trace.consume(admitted(1_000, spec))
        trace.consume(started(1_001, spec))
        trace.consume(progress(1_002, spec, 128))
        assert_rejected(self, lambda: trace.consume(first_token(1_003)))

    def test_ttft_requires_cancel_and_decode_forbids_cancel(self):
        ttft = GATE.SCHEDULE[0]
        trace = GATE.LifecycleTrace(ttft)
        trace.consume(admitted(1_000, ttft))
        trace.consume(started(1_001, ttft))
        next_time = consume_prefill(trace, ttft, 1_000)
        trace.consume(first_token(next_time))
        assert_rejected(
            self, lambda: trace.consume(released(next_time + 1, 1_000, ttft))
        )

        decode = GATE.SCHEDULE[60]
        trace = GATE.LifecycleTrace(decode)
        trace.consume(admitted(2_000, decode))
        trace.consume(started(2_001, decode))
        next_time = consume_prefill(trace, decode, 2_000)
        trace.consume(first_token(next_time))
        assert_rejected(self, lambda: trace.consume(cancel(next_time + 1, 2_000)))

    def test_ttft_content_must_precede_cancel(self):
        spec = GATE.SCHEDULE[0]
        trace, _observation = valid_trace(spec)
        late = ttft_observation(sent=1_005, content_time=9_000)
        assert_rejected(self, lambda: trace.complete(late))

    def test_run_validator_rejects_wrong_case_and_incomplete_schedule(self):
        validator = GATE.LatencyRunValidator()
        assert_rejected(self, lambda: validator.begin(GATE.SCHEDULE[1]))
        validator = GATE.LatencyRunValidator()
        validator.begin(GATE.SCHEDULE[0])
        assert_rejected(self, validator.finalize)

    def test_gate_drains_http_stdout_before_waiting_for_observer_release(self):
        spec = GATE.SCHEDULE[0]
        fixture = GATE.FixtureSnapshot(
            fixture_id=spec.fixture_id,
            prompt_tokens=spec.prompt_tokens,
            messages=[{"role": "user", "content": "x"}],
            raw=b"fixture",
            sha256="0" * 64,
            identity=(1, 2, 3, 4, 5),
        )
        order = []
        http = mock.Mock()
        http.begin.side_effect = lambda _plan: order.append("begin")
        observation = ttft_observation()
        http.finish.side_effect = lambda _deadline: (
            order.append("finish") or observation
        )
        sample_writer = mock.Mock()
        gate = GATE.LatencyGate(
            mock.Mock(gateway_pid=1, uid=1000, gid=1000),
            "sha256:" + "0" * 64,
            "1" * 64,
            {spec.fixture_id: fixture},
            mock.Mock(),
            mock.Mock(),
            mock.Mock(),
            sample_writer,
            mock.Mock(),
            http,
        )
        gate.validator = mock.Mock()
        gate.validator.complete.return_value = {"sequence": 1}
        gate._wait_for_release = mock.Mock(
            side_effect=lambda _deadline: order.append("release")
        )
        gate._correlate = mock.Mock()
        with (
            mock.patch.object(GATE, "require_epoch"),
            mock.patch.object(GATE.DIRECT, "require_ready"),
        ):
            gate._run_case(spec)
        self.assertEqual(order, ["begin", "finish", "release"])

    def test_run_validator_rejects_reused_request_or_completion_identity(self):
        validator = GATE.LatencyRunValidator()
        first = {
            "request_id": "request-reused",
            "completion_id": "chatcmpl-reused",
            "release_observed_monotonic_ns": 10,
        }
        validator.active = mock.Mock()
        validator.active.complete.return_value = first
        validator.complete(mock.Mock())
        validator.active = mock.Mock()
        validator.active.complete.return_value = {
            **first,
            "release_observed_monotonic_ns": 20,
        }
        assert_rejected(self, lambda: validator.complete(mock.Mock()))


class MetricTests(unittest.TestCase):
    def test_exact_linear_interpolation_and_valid_72_samples(self):
        self.assertEqual(
            GATE.linear_percentile([0, 10], Fraction(19, 20)), Fraction(19, 2)
        )
        result = GATE.derive_metrics(valid_metric_samples())
        self.assertEqual(result["decode64"]["interval_count"], 630)
        self.assertEqual(result["ttft"]["exact-p3584"]["count"], 10)

    def test_bool_is_not_accepted_as_an_integer_measurement(self):
        assert_rejected(self, lambda: GATE.linear_percentile([True], Fraction(1, 2)))

    def test_ttft_hard_limit_mutation_fails(self):
        samples = valid_metric_samples()
        for item in samples:
            if (
                item["workload"] == "ttft"
                and item["fixture_id"] == "exact-p0032"
                and item["sample_kind"] == "measured"
            ):
                item["ttft_ns"] = 3_100_000_000
        assert_rejected(self, lambda: GATE.derive_metrics(samples))

    def test_metric_derivation_rejects_reordered_case_metadata(self):
        samples = valid_metric_samples()
        samples[0], samples[1] = samples[1], samples[0]
        assert_rejected(self, lambda: GATE.derive_metrics(samples))

    def test_decode_throughput_and_interval_hard_limit_mutations_fail(self):
        samples = valid_metric_samples()
        for item in samples:
            if item["workload"] == "decode64" and item["sample_kind"] == "measured":
                item["decode_elapsed_ns"] = 5_000_000_000
        assert_rejected(self, lambda: GATE.derive_metrics(samples))
        samples = valid_metric_samples()
        for item in samples:
            if item["workload"] == "decode64" and item["sample_kind"] == "measured":
                item["decode_intervals_ns"] = [101_000_000] * 63
        assert_rejected(self, lambda: GATE.derive_metrics(samples))


class EpochAndAtomicEvidenceTests(unittest.TestCase):
    def epoch_document(self, *, n_restarts=2):
        identity = GATE.DIRECT.ServiceIdentity(
            unit=GATE.DIRECT.SERVICE_UNIT,
            user="homelab1",
            uid=1000,
            gid=1000,
            control_group="/system.slice/ullm-openai.service",
            gateway_pid=111,
            gateway_starttime_ticks=222,
            worker_pid=333,
            worker_starttime_ticks=444,
            n_restarts=n_restarts,
            boot_id="a" * 32,
        )
        return {
            "schema_version": GATE.EPOCH_SCHEMA,
            "phase": "resource_restart",
            "service_identity": dataclasses.asdict(identity),
        }

    def test_epoch_requires_exact_restarted_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch.json"
            path.write_bytes(GATE.compact_json(self.epoch_document()))
            epoch = GATE.load_epoch(path)
            self.assertEqual(epoch.service_identity.gateway_pid, 111)
            self.assertEqual(epoch.service_identity.n_restarts, 2)

    def test_epoch_rejects_zero_restart_bool_and_unknown_field(self):
        for document in (
            self.epoch_document(n_restarts=0),
            self.epoch_document(n_restarts=True),
            {**self.epoch_document(), "extra": 1},
        ):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "epoch.json"
                path.write_bytes(GATE.compact_json(document))
                assert_rejected(self, lambda path=path: GATE.load_epoch(path))

    def test_checksum_document_is_sorted_and_detects_mutation(self):
        guard = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            (stage / "b.json").write_bytes(b"b\n")
            (stage / "a.json").write_bytes(b"a\n")
            raw = GATE.write_sha256sums(stage, ["b.json", "a.json"], guard)
            self.assertTrue(raw.splitlines()[0].endswith(b"  a.json"))
            GATE.verify_sha256sums(stage, ["b.json", "a.json"], raw)
            (stage / "a.json").write_bytes(b"changed\n")
            assert_rejected(
                self,
                lambda: GATE.verify_sha256sums(stage, ["b.json", "a.json"], raw),
            )


class HttpEventProtocolTests(unittest.TestCase):
    def event_sequence(self):
        spec = GATE.SCHEDULE[0]
        body = b'{"model":"test"}'
        value = {
            "id": "chatcmpl-latency",
            "choices": [{"delta": {"content": "x"}, "finish_reason": None}],
        }
        chunk = b"data: " + GATE.compact_json(value) + b"\n\n"
        request = {
            "schema_version": GATE.HTTP_EVENT_SCHEMA,
            "event": "http_request",
            "request_key": spec.case_id,
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
            "connect_completed_monotonic_ns": 100,
            "write_started_monotonic_ns": 101,
            "last_body_byte_sent_monotonic_ns": 102,
        }
        start = {
            "schema_version": GATE.HTTP_EVENT_SCHEMA,
            "event": "http_response_start",
            "request_key": spec.case_id,
            "status": 200,
            "headers": [["Content-Type", "text/event-stream"]],
            "observed_monotonic_ns": 103,
        }
        body_event = {
            "schema_version": GATE.HTTP_EVENT_SCHEMA,
            "event": "http_body_chunk",
            "request_key": spec.case_id,
            "chunk_index": 0,
            "body_base64": base64.b64encode(chunk).decode("ascii"),
            "body_sha256": hashlib.sha256(chunk).hexdigest(),
            "body_bytes": len(chunk),
            "observed_monotonic_ns": 200,
        }
        end = {
            "schema_version": GATE.HTTP_EVENT_SCHEMA,
            "event": "http_response_end",
            "request_key": spec.case_id,
            "outcome": "client_closed",
            "error": None,
            "body_bytes": len(chunk),
            "body_sha256": hashlib.sha256(chunk).hexdigest(),
            "observed_monotonic_ns": 201,
        }
        return spec, body, [request, start, body_event, end]

    def test_client_preserves_last_body_and_chunk_timing(self):
        spec, body, events = self.event_sequence()
        client = GATE.LatencyHttpClient([], mock.Mock(), mock.Mock())
        client.active = GATE.HttpPlan(spec, body)
        client._read_event = mock.Mock(side_effect=events)
        result = client.finish(1_000)
        self.assertEqual(result.request_sent_monotonic_ns, 102)
        self.assertEqual(result.items[0].observed_monotonic_ns, 200)

    def test_client_rejects_request_body_hash_mutation(self):
        spec, body, events = self.event_sequence()
        events[0]["body_sha256"] = "0" * 64
        client = GATE.LatencyHttpClient([], mock.Mock(), mock.Mock())
        client.active = GATE.HttpPlan(spec, body)
        client._read_event = mock.Mock(side_effect=events)
        assert_rejected(self, lambda: client.finish(1_000))

    def test_client_rejects_begin_after_the_72_case_schedule(self):
        client = GATE.LatencyHttpClient([], mock.Mock(), mock.Mock())
        client.request_count = 72
        assert_rejected(
            self,
            lambda: client.begin(GATE.HttpPlan(GATE.SCHEDULE[-1], b"{}")),
        )


if __name__ == "__main__":
    unittest.main()
