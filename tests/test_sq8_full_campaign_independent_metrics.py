from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, cast


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


METRICS = load_module(
    "test_sq8_full_campaign_independent_metrics_tool",
    TOOLS / "sq8_full_campaign_independent_metrics.py",
)
PRODUCER = load_module(
    "test_sq8_full_campaign_independent_metrics_producer_oracle",
    TOOLS / "sq8_full_campaign_views.py",
)
RESOURCE_FIXTURES = load_module(
    "test_sq8_full_campaign_independent_metrics_resource_fixtures",
    ROOT / "tests" / "test_sq8_full_campaign_views.py",
)


SHA = "a" * 64
LATENCY_FIXTURE_ROOT = (
    ROOT / "tests" / "fixtures" / "sq8-serving-v0.1" / "chat-template" / "fixtures"
)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def reject_nonfinite(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def tracked_latency_request_body_identities() -> dict[tuple[str, int], tuple[int, str]]:
    result: dict[tuple[str, int], tuple[int, str]] = {}
    for fixture_id, prompt_tokens in cast(
        tuple[tuple[str, int], ...], METRICS.FIXTURE_ORDER
    ):
        raw = (LATENCY_FIXTURE_ROOT / f"{fixture_id}.json").read_bytes()
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
        if type(value) is not dict or set(value) != {
            "schema_version",
            "fixture_id",
            "kind",
            "messages",
            "construction",
            "template_options",
            "expected",
        }:
            raise AssertionError("tracked latency fixture fields differ")
        messages = value["messages"]
        expected = value["expected"]
        if (
            value["schema_version"] != "ullm.sq8.chat_template_fixture.v1"
            or value["fixture_id"] != fixture_id
            or value["kind"] != "exact_length"
            or value["template_options"]
            != {"add_generation_prompt": True, "enable_thinking": False}
            or type(expected) is not dict
            or expected.get("prompt_tokens") != prompt_tokens
            or type(messages) is not list
            or len(messages) != 1
            or type(messages[0]) is not dict
            or set(messages[0]) != {"role", "content"}
            or messages[0].get("role") != "user"
            or type(messages[0].get("content")) is not str
            or not messages[0]["content"]
        ):
            raise AssertionError("tracked latency fixture contract differs")
        max_tokens_values = (512, 64) if fixture_id == "exact-p0032" else (512,)
        for max_tokens in max_tokens_values:
            body = json.dumps(
                {
                    "model": "ullm-qwen3-14b-sq8",
                    "messages": messages,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 0,
                },
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            result[(fixture_id, max_tokens)] = (
                len(body),
                hashlib.sha256(body).hexdigest(),
            )
    return result


@dataclasses.dataclass(frozen=True)
class SseItem:
    chunk_index: int
    observed_monotonic_ns: int
    done: bool
    completion_id_utf8_bytes: int | None
    completion_id_sha256: str | None
    content_utf8_bytes: int | None
    content_sha256: str | None
    finish_reason: str | None
    usage_present: bool
    usage_is_object: bool | None
    completion_tokens: int | None


@dataclasses.dataclass(frozen=True)
class SseMetadata:
    chunk_count: int
    first_chunk_monotonic_ns: int
    last_chunk_monotonic_ns: int
    items: tuple[SseItem, ...]


@dataclasses.dataclass(frozen=True)
class HttpResult:
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
    connect_completed_monotonic_ns: int
    write_started_monotonic_ns: int
    last_body_byte_sent_monotonic_ns: int
    response_started_monotonic_ns: int
    response_end_monotonic_ns: int
    sse: SseMetadata | None


@dataclasses.dataclass
class Trace:
    phase: str
    case_id: str
    completion_id: str
    events: list[dict[str, Any]]
    terminal: str | None = "request_released"


@dataclasses.dataclass
class Session:
    http_results: tuple[HttpResult, ...]
    traces: dict[str, Trace]
    releases_by_phase: dict[str, list[dict[str, Any]]]
    probes: dict[str, dict[str, Any]]


def completion_binding(completion_id: str) -> tuple[int, str]:
    raw = completion_id.encode("utf-8")
    return len(raw), hashlib.sha256(raw).hexdigest()


def item(
    completion_id: str,
    chunk_index: int,
    observed_ns: int,
    *,
    content: bool = False,
    finish: str | None = None,
    usage: int | None = None,
    done: bool = False,
) -> SseItem:
    identifier_bytes, identifier_sha = completion_binding(completion_id)
    content_raw = f"token-{chunk_index}".encode("ascii") if content else None
    return SseItem(
        chunk_index=chunk_index,
        observed_monotonic_ns=observed_ns,
        done=done,
        completion_id_utf8_bytes=None if done else identifier_bytes,
        completion_id_sha256=None if done else identifier_sha,
        content_utf8_bytes=None if content_raw is None else len(content_raw),
        content_sha256=(
            None if content_raw is None else hashlib.sha256(content_raw).hexdigest()
        ),
        finish_reason=finish,
        usage_present=usage is not None,
        usage_is_object=True if usage is not None else None,
        completion_tokens=usage,
    )


def trace_event(
    request_id: str,
    completion_id: str,
    name: str,
    observed_ns: int,
    **fields: Any,
) -> dict[str, Any]:
    return {
        "event": name,
        "request_id": request_id,
        "completion_id": completion_id,
        "observed_monotonic_ns": observed_ns,
        **fields,
    }


def latency_trace(
    request_id: str,
    completion_id: str,
    spec: object,
    first_content_ns: int,
    release_ns: int,
) -> Trace:
    prompt_tokens = cast(int, getattr(spec, "prompt_tokens"))
    max_tokens = cast(int, getattr(spec, "max_tokens"))
    workload = cast(str, getattr(spec, "workload"))
    base = first_content_ns - 50
    events = [
        trace_event(
            request_id,
            completion_id,
            "request_admitted",
            base,
            stream=True,
            prompt_tokens=prompt_tokens,
            max_completion_tokens=max_tokens,
        ),
        trace_event(
            request_id,
            completion_id,
            "request_started",
            base + 1,
            stream=True,
            prompt_tokens=prompt_tokens,
            admit_to_start_ns=1,
        ),
    ]
    processed = 0
    while processed < prompt_tokens:
        processed = min(processed + 128, prompt_tokens)
        events.append(
            trace_event(
                request_id,
                completion_id,
                "request_progress",
                base + len(events),
                phase="prefill",
                prompt_tokens=prompt_tokens,
                processed_prompt_tokens=processed,
            )
        )
    events.append(
        trace_event(
            request_id,
            completion_id,
            "request_first_token",
            first_content_ns,
            stream=True,
            completion_tokens=1,
        )
    )
    admit_to_start = 1
    if workload == "ttft":
        cancel_time = first_content_ns
        admit_to_cancel = cancel_time - base
        events.append(
            trace_event(
                request_id,
                completion_id,
                "request_cancel_requested",
                cancel_time,
                stream=True,
                reason="client_disconnect",
                admit_to_cancel_ns=admit_to_cancel,
            )
        )
        outcome = "cancelled"
        cancel_reason: str | None = "client_disconnect"
        completion_tokens = 1
    else:
        outcome = "length"
        cancel_reason = None
        completion_tokens = 64
    events.append(
        trace_event(
            request_id,
            completion_id,
            "request_released",
            release_ns,
            stream=True,
            prompt_tokens=prompt_tokens,
            outcome=outcome,
            cancel_reason=cancel_reason,
            completion_tokens=completion_tokens,
            reset_complete=True,
            admit_to_start_ns=admit_to_start,
            start_to_release_ns=release_ns - base - admit_to_start,
            admit_to_release_ns=release_ns - base,
        )
    )
    return Trace("latency", cast(str, getattr(spec, "case_id")), completion_id, events)


def linear_percentile(values: list[int], probability: Fraction) -> int | dict[str, int]:
    ordered = sorted(values)
    rank = Fraction(len(ordered) - 1) * probability
    lower = rank.numerator // rank.denominator
    upper = lower if rank.denominator == 1 else lower + 1
    value = Fraction(ordered[lower])
    if lower != upper:
        value += (rank - lower) * (ordered[upper] - ordered[lower])
    if value.denominator == 1:
        return value.numerator
    return {"numerator": value.numerator, "denominator": value.denominator}


def latency_session(
    *,
    slow_ttft: bool = False,
    slow_decode_intervals: bool = False,
) -> tuple[Session, dict[str, Any]]:
    results: list[HttpResult] = []
    traces: dict[str, Trace] = {}
    samples: list[dict[str, Any]] = []
    body_identities = cast(
        dict[tuple[str, int], tuple[int, str]],
        METRICS.LATENCY_REQUEST_BODY_IDENTITIES,
    )
    cursor = 1_000_000_000_000
    for spec in cast(tuple[object, ...], METRICS.LATENCY_SCHEDULE):
        sequence = cast(int, getattr(spec, "sequence"))
        case_id = cast(str, getattr(spec, "case_id"))
        workload = cast(str, getattr(spec, "workload"))
        sample_kind = cast(str, getattr(spec, "sample_kind"))
        sample_index = cast(int, getattr(spec, "sample_index"))
        fixture_id = cast(str, getattr(spec, "fixture_id"))
        prompt_tokens = cast(int, getattr(spec, "prompt_tokens"))
        max_tokens = cast(int, getattr(spec, "max_tokens"))
        request_body_bytes, request_body_sha256 = body_identities[
            (fixture_id, max_tokens)
        ]
        completion_id = f"chatcmpl-latency-{sequence:03d}"
        request_id = f"req-latency-{sequence:03d}"
        sent = cursor + 100
        response_started = sent + 100
        sse_items: tuple[SseItem, ...]
        if workload == "ttft":
            ttft = (
                4_000_000_000
                if slow_ttft and fixture_id == "exact-p0032"
                else (1_000_000_000 + sample_index * 1_000_000)
            )
            first_content = sent + ttft
            sse_items = (item(completion_id, 0, first_content, content=True),)
            response_end = first_content + 10
            release = response_end + 100
            outcome = "client_closed"
            samples.append(
                {
                    "sequence": sequence,
                    "case_id": case_id,
                    "sample_kind": sample_kind,
                    "sample_index": sample_index,
                    "fixture_id": fixture_id,
                    "prompt_tokens": prompt_tokens,
                    "ttft_ns": ttft,
                    "content_object_count": 1,
                    "release_outcome": "cancelled",
                    "release_completion_tokens": 1,
                }
            )
        else:
            first_content = response_started + 100
            if slow_decode_intervals and sample_kind == "measured":
                intervals = [110_000_000] * 32 + [20_000_000] * 31
            else:
                intervals = [50_000_000] * 63
            content_times = [first_content]
            for interval in intervals:
                content_times.append(content_times[-1] + interval)
            mutable = [
                item(completion_id, index, observed, content=True)
                for index, observed in enumerate(content_times)
            ]
            mutable.extend(
                (
                    item(completion_id, 64, content_times[-1] + 1, finish="length"),
                    item(completion_id, 65, content_times[-1] + 2, usage=64),
                    item(completion_id, 66, content_times[-1] + 3, done=True),
                )
            )
            sse_items = tuple(mutable)
            response_end = content_times[-1] + 4
            release = content_times[-1] - 1
            outcome = "eof"
            elapsed = content_times[-1] - content_times[0]
            throughput = Fraction(63_000_000_000, elapsed)
            samples.append(
                {
                    "sequence": sequence,
                    "case_id": case_id,
                    "sample_kind": sample_kind,
                    "sample_index": sample_index,
                    "fixture_id": fixture_id,
                    "prompt_tokens": prompt_tokens,
                    "decode_elapsed_ns": elapsed,
                    "decode_intervals_ns": intervals,
                    "decode_tokens_per_second": (
                        throughput.numerator
                        if throughput.denominator == 1
                        else {
                            "numerator": throughput.numerator,
                            "denominator": throughput.denominator,
                        }
                    ),
                    "release_outcome": "length",
                    "release_completion_tokens": 64,
                }
            )
        sse = SseMetadata(
            len(sse_items),
            sse_items[0].observed_monotonic_ns,
            sse_items[-1].observed_monotonic_ns,
            sse_items,
        )
        results.append(
            HttpResult(
                "latency",
                case_id,
                sequence,
                case_id,
                "POST",
                "/v1/chat/completions",
                200,
                outcome,
                request_body_bytes,
                request_body_sha256,
                1000,
                SHA,
                cursor,
                cursor + 50,
                sent,
                response_started,
                response_end,
                sse,
            )
        )
        traces[request_id] = latency_trace(
            request_id, completion_id, spec, first_content, release
        )
        cursor = release + 1_000_000

    ttft_metrics: dict[str, Any] = {}
    for fixture_id, _prompt in cast(tuple[tuple[str, int], ...], METRICS.FIXTURE_ORDER):
        values = [
            cast(int, sample["ttft_ns"])
            for sample in samples
            if sample["fixture_id"] == fixture_id
            and sample["sample_kind"] == "measured"
            and "ttft_ns" in sample
        ]
        limits = cast(dict[str, tuple[int, int]], METRICS.TTFT_LIMITS_NS)[fixture_id]
        ttft_metrics[fixture_id] = {
            "count": 10,
            "p50_ns": linear_percentile(values, Fraction(1, 2)),
            "p95_ns": linear_percentile(values, Fraction(19, 20)),
            "p50_maximum_ns": limits[0],
            "p95_maximum_ns": limits[1],
        }
    decode_measured = [
        sample
        for sample in samples
        if "decode_elapsed_ns" in sample and sample["sample_kind"] == "measured"
    ]
    throughputs = sorted(
        Fraction(63_000_000_000, cast(int, sample["decode_elapsed_ns"]))
        for sample in decode_measured
    )
    throughput = (throughputs[4] + throughputs[5]) / 2
    pooled = [
        interval
        for sample in decode_measured
        for interval in cast(list[int], sample["decode_intervals_ns"])
    ]
    producer_input = {
        "schema_version": PRODUCER.LATENCY_INPUT_SCHEMA,
        "request_count": 72,
        "http_record_count": 1,
        "lifecycle_record_count": 1,
        "journal_record_count": 1,
        "prefill_ttft": {
            "request_count": 60,
            "metrics": ttft_metrics,
            "samples": samples[:60],
        },
        "decode64": {
            "request_count": 12,
            "metrics": {
                "request_count": 10,
                "interval_count": 630,
                "p50_tokens_per_second": (
                    throughput.numerator
                    if throughput.denominator == 1
                    else {
                        "numerator": throughput.numerator,
                        "denominator": throughput.denominator,
                    }
                ),
                "minimum_p50_tokens_per_second": 15,
                "p95_inter_content_ns": linear_percentile(pooled, Fraction(19, 20)),
                "maximum_p95_inter_content_ns": 100_000_000,
            },
            "samples": samples[60:],
        },
        "source_bindings": {},
    }
    return Session(tuple(results), traces, {}, {}), producer_input


def canonical(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def scaled_resource_records() -> list[dict[str, Any]]:
    records = cast(
        list[dict[str, Any]], copy.deepcopy(RESOURCE_FIXTURES.resource_records())
    )
    for record in records:
        if record.get("record_type") == "gpu_metric":
            record["captured_monotonic_ns"] *= 2
        elif record.get("record_type") == "resource_sample":
            for key in (
                "release_observed_monotonic_ns",
                "idle_settle_started_monotonic_ns",
                "sample_monotonic_ns",
            ):
                if record[key] is not None:
                    record[key] *= 2
    return records


def resource_trace(
    phase: str,
    case_id: str,
    request_id: str,
    completion_id: str,
    admission_ns: int,
    response_end_ns: int,
    release_ns: int,
) -> Trace:
    prompt_tokens = 32
    events = [
        trace_event(
            request_id,
            completion_id,
            "request_admitted",
            admission_ns,
            stream=True,
            prompt_tokens=prompt_tokens,
            max_completion_tokens=2,
        ),
        trace_event(
            request_id,
            completion_id,
            "request_started",
            admission_ns + 1,
            stream=True,
            prompt_tokens=prompt_tokens,
            admit_to_start_ns=1,
        ),
        trace_event(
            request_id,
            completion_id,
            "request_progress",
            admission_ns + 2,
            phase="prefill",
            prompt_tokens=prompt_tokens,
            processed_prompt_tokens=prompt_tokens,
        ),
        trace_event(
            request_id,
            completion_id,
            "request_first_token",
            release_ns - 1,
            stream=True,
            completion_tokens=1,
        ),
        trace_event(
            request_id,
            completion_id,
            "request_released",
            release_ns,
            stream=True,
            prompt_tokens=prompt_tokens,
            outcome="length",
            cancel_reason=None,
            completion_tokens=2,
            reset_complete=True,
            admit_to_start_ns=1,
            start_to_release_ns=release_ns - admission_ns - 1,
            admit_to_release_ns=release_ns - admission_ns,
        ),
    ]
    return Trace(phase, case_id, completion_id, events)


def resource_http_result(
    phase: str,
    case_id: str,
    request_index: int,
    completion_id: str,
    connect_ns: int,
    admission_ns: int,
    response_end_ns: int,
    *,
    positive: bool,
) -> HttpResult:
    if positive:
        items = (
            item(completion_id, 0, response_end_ns - 3, content=True),
            item(completion_id, 1, response_end_ns - 2, finish="length"),
            item(completion_id, 2, response_end_ns - 1, usage=2),
            item(completion_id, 3, response_end_ns, done=True),
        )
        sse = SseMetadata(4, items[0].observed_monotonic_ns, response_end_ns, items)
        status = 200
    else:
        sse = None
        status = 400
    return HttpResult(
        phase,
        case_id,
        request_index,
        f"p8f-{case_id}",
        "POST",
        "/v1/chat/completions",
        status,
        "eof",
        100,
        SHA,
        100,
        SHA,
        connect_ns,
        connect_ns + 1,
        admission_ns,
        admission_ns + 1,
        response_end_ns,
        sse,
    )


class ResourceFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "bundle"
        self.root.mkdir(mode=0o700)
        self.records = scaled_resource_records()
        metric_raw = b'{"gpu":2}\n'
        metric_sha = hashlib.sha256(metric_raw).hexdigest()
        for record in self.records:
            if record.get("record_type") == "gpu_metric":
                record["raw_output_sha256"] = metric_sha
                path = self.root / cast(str, record["raw_output_file"])
                path.write_bytes(metric_raw)
                os.chmod(path, 0o600)
        self.write_records()
        self.session = self.build_session()

    def write_records(self) -> None:
        path = self.root / "soak-resources.raw.jsonl"
        path.write_bytes(b"".join(canonical(record) for record in self.records))
        os.chmod(path, 0o600)

    def point(self, segment: str, request_index: int | None) -> dict[str, Any]:
        return next(
            record
            for record in self.records
            if record.get("record_type") == "resource_sample"
            and record["segment"] == segment
            and record["request_index"] == request_index
            and record["sample_index"] == 0
        )

    def metric_time(self, segment: str, boundary: str) -> int:
        return cast(
            int,
            next(
                record
                for record in self.records
                if record.get("record_type") == "gpu_metric"
                and record["segment"] == segment
                and record["boundary"] == boundary
            )["captured_monotonic_ns"],
        )

    def build_session(self) -> Session:
        results: list[HttpResult] = []
        traces: dict[str, Trace] = {}
        releases: dict[str, list[dict[str, Any]]] = {
            "resource_normal": [],
            "resource_restart": [],
        }
        prior_release = -1
        for phase, case_id, request_index, positive in cast(
            tuple[tuple[str, str, int, bool], ...], METRICS.RESOURCE_HTTP_SCHEDULE
        ):
            segment = "normal" if phase == "resource_normal" else "restart"
            warmup = "warmup" in case_id
            if warmup:
                before = self.metric_time(segment, "before")
                admission = before + request_index * 1_000_000_000
                release = admission + 500_000_000
                response_end = release + 100_000_000
                request_id = f"resource-{segment}-warmup-{request_index:03d}"
            elif positive:
                point = self.point(segment, request_index)
                release = cast(int, point["release_observed_monotonic_ns"])
                admission = release - 1_000_000_000
                response_end = release + 100_000_000
                request_id = cast(str, point["request_id"])
            else:
                prior_point = self.point(segment, request_index)
                prior_last = (
                    cast(int, prior_point["sample_monotonic_ns"]) + 8_000_000_000
                )
                admission = prior_last + 100_000_000
                response_end = admission + 100_000_000
                release = response_end
                request_id = ""
            connect = max(prior_release, admission - 100_000_000)
            completion_id = f"chatcmpl-{case_id}"
            result = resource_http_result(
                phase,
                case_id,
                request_index,
                completion_id,
                connect,
                admission,
                response_end,
                positive=positive,
            )
            results.append(result)
            if positive:
                trace = resource_trace(
                    phase,
                    case_id,
                    request_id,
                    completion_id,
                    admission,
                    response_end,
                    release,
                )
                traces[request_id] = trace
                releases[phase].append(trace.events[-1])
                prior_release = release
        normal_identity = self.point("normal", None)
        restart_identity = self.point("restart", None)
        probes = {
            "normal-segment-start": self.probe(
                "normal", normal_identity, self.metric_time("normal", "before"), 2
            ),
            "restart-segment-start": self.probe(
                "restart", restart_identity, self.metric_time("restart", "before"), 3
            ),
        }
        return Session(tuple(results), traces, releases, probes)

    @staticmethod
    def probe(
        segment: str, sample: dict[str, Any], observed_ns: int, restarts: int
    ) -> dict[str, Any]:
        name = f"{segment}-segment-start"
        return {
            "probe": name,
            "phase": f"resource_{segment}",
            "service_active": True,
            "ready_http_status": 200,
            "control_group": sample["systemd"]["control_group_before"],
            "gateway_pid": sample["gateway"]["pid"],
            "gateway_starttime_ticks": sample["gateway"]["starttime_ticks_before"],
            "worker_pid": sample["worker"]["pid"],
            "worker_starttime_ticks": sample["worker"]["starttime_ticks_before"],
            "n_restarts": restarts,
            "observed_monotonic_ns": observed_ns,
        }

    def close(self) -> None:
        self.temporary.cleanup()

    def __enter__(self) -> ResourceFixture:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class IndependentLatencyTests(unittest.TestCase):
    def test_request_body_identities_match_tracked_fixtures(self) -> None:
        self.assertEqual(
            METRICS.LATENCY_REQUEST_BODY_IDENTITIES,
            tracked_latency_request_body_identities(),
        )

    def test_reconstructs_exact_producer_latency_view(self) -> None:
        session, producer_input = latency_session()
        expected = cast(dict[str, Any], PRODUCER.project_latency(producer_input))
        observed = cast(dict[str, Any], METRICS.reconstruct_latency_results(session))
        self.assertEqual(observed, expected)
        self.assertEqual(observed["request_count"], 72)
        self.assertEqual(observed["decode64"]["metrics"]["interval_count"], 630)

    def test_schedule_lifecycle_and_sse_negatives_are_rejected(self) -> None:
        mutations: tuple[tuple[str, Callable[[Session], None]], ...] = (
            (
                "schedule",
                lambda session: setattr(
                    session,
                    "http_results",
                    (session.http_results[1], session.http_results[0])
                    + session.http_results[2:],
                ),
            ),
            (
                "request body bytes",
                lambda session: setattr(
                    session,
                    "http_results",
                    (
                        dataclasses.replace(
                            session.http_results[0],
                            request_body_bytes=(
                                session.http_results[0].request_body_bytes + 1
                            ),
                        ),
                    )
                    + session.http_results[1:],
                ),
            ),
            (
                "request body SHA",
                lambda session: setattr(
                    session,
                    "http_results",
                    (
                        dataclasses.replace(
                            session.http_results[0], request_body_sha256="0" * 64
                        ),
                    )
                    + session.http_results[1:],
                ),
            ),
            (
                "lifecycle",
                lambda session: (
                    session.traces[next(iter(session.traces))]
                    .events[-1]
                    .update({"outcome": "length"})
                ),
            ),
            (
                "SSE",
                lambda session: setattr(
                    session,
                    "http_results",
                    (
                        dataclasses.replace(
                            session.http_results[0],
                            sse=dataclasses.replace(
                                cast(SseMetadata, session.http_results[0].sse),
                                items=(
                                    dataclasses.replace(
                                        cast(
                                            SseMetadata, session.http_results[0].sse
                                        ).items[0],
                                        usage_present=True,
                                        usage_is_object=False,
                                    ),
                                ),
                            ),
                        ),
                    )
                    + session.http_results[1:],
                ),
            ),
        )
        for label, mutation in mutations:
            with self.subTest(label=label):
                session, _producer_input = latency_session()
                mutation(session)
                with self.assertRaises(METRICS.IndependentMetricsError):
                    METRICS.reconstruct_latency_results(session)

    def test_latency_threshold_negatives_are_rejected(self) -> None:
        for label, options in (
            ("TTFT", {"slow_ttft": True}),
            ("decode percentile", {"slow_decode_intervals": True}),
        ):
            with self.subTest(label=label):
                session, _producer = latency_session(**options)
                with self.assertRaises(METRICS.IndependentMetricsError):
                    METRICS.reconstruct_latency_results(session)


class IndependentResourceTests(unittest.TestCase):
    def test_streams_and_reconstructs_exact_producer_resource_view(self) -> None:
        with ResourceFixture() as fixture:
            expected = cast(
                dict[str, Any],
                PRODUCER.analyze_soak_resources(
                    fixture.root / "soak-resources.raw.jsonl"
                ),
            )
            observed = cast(
                dict[str, Any],
                METRICS.reconstruct_soak_resource_results(
                    fixture.root, fixture.session
                ),
            )
            self.assertEqual(observed, expected)
            self.assertEqual(observed["resource_sample_count"], 610)
            self.assertEqual(observed["gpu_metric_count"], 4)

    def test_schedule_identity_outcome_hash_and_window_negatives_are_rejected(
        self,
    ) -> None:
        defects: tuple[tuple[str, Callable[[ResourceFixture], None]], ...] = (
            (
                "schedule",
                lambda fixture: fixture.records[2].update({"sample_index": 1}),
            ),
            (
                "identity",
                lambda fixture: fixture.records[3]["worker"].update({"pid": 999}),
            ),
            (
                "outcome",
                lambda fixture: fixture.point("normal", 1).update(
                    {"release_outcome": "cancelled"}
                ),
            ),
            (
                "window",
                lambda fixture: next(
                    record
                    for record in fixture.records
                    if record.get("record_type") == "gpu_metric"
                    and record["segment"] == "normal"
                    and record["boundary"] == "after"
                ).update({"captured_monotonic_ns": 1}),
            ),
        )
        for label, mutation in defects:
            with self.subTest(label=label), ResourceFixture() as fixture:
                mutation(fixture)
                fixture.write_records()
                with self.assertRaises(METRICS.IndependentMetricsError):
                    METRICS.reconstruct_soak_resource_results(
                        fixture.root, fixture.session
                    )

        with ResourceFixture() as fixture:
            metric_path = fixture.root / "amd-smi-metric-normal-before.json"
            metric_path.write_bytes(b'{"gpu":3}\n')
            os.chmod(metric_path, 0o600)
            with self.assertRaises(METRICS.IndependentMetricsError):
                METRICS.reconstruct_soak_resource_results(fixture.root, fixture.session)

    def test_session_cross_binding_and_file_safety_negatives_are_rejected(self) -> None:
        with ResourceFixture() as fixture:
            point = fixture.point("normal", 1)
            point["request_id"] = "different-request"
            fixture.write_records()
            with self.assertRaises(METRICS.IndependentMetricsError):
                METRICS.reconstruct_soak_resource_results(fixture.root, fixture.session)

        with ResourceFixture() as fixture:
            path = fixture.root / "soak-resources.raw.jsonl"
            outside = fixture.root.parent / "outside-resource"
            path.rename(outside)
            path.symlink_to(outside)
            with self.assertRaises(METRICS.IndependentMetricsError):
                METRICS.reconstruct_soak_resource_results(fixture.root, fixture.session)

        with ResourceFixture() as fixture:
            path = fixture.root / "soak-resources.raw.jsonl"
            path.write_bytes(b"{" + b"x" * ((1 << 20) + 1) + b"\n")
            os.chmod(path, 0o600)
            with self.assertRaises(METRICS.IndependentMetricsError):
                METRICS.reconstruct_soak_resource_results(fixture.root, fixture.session)

        self.assertNotEqual(METRICS._file_flags() & os.O_NONBLOCK, 0)
        with ResourceFixture() as fixture:
            path = fixture.root / "soak-resources.raw.jsonl"
            saved = fixture.root / "soak-resources.saved.jsonl"
            original_entry_identity = METRICS._RootSnapshot.entry_identity
            swapped = False

            def replace_with_fifo_after_stat(snapshot: object, name: str) -> object:
                nonlocal swapped
                identity = original_entry_identity(snapshot, name)
                if name == path.name and not swapped:
                    path.rename(saved)
                    os.mkfifo(path, 0o600)
                    swapped = True
                return identity

            with (
                mock.patch.object(
                    METRICS._RootSnapshot,
                    "entry_identity",
                    replace_with_fifo_after_stat,
                ),
                self.assertRaises(METRICS.IndependentMetricsError),
            ):
                METRICS.reconstruct_soak_resource_results(fixture.root, fixture.session)


if __name__ == "__main__":
    unittest.main()
