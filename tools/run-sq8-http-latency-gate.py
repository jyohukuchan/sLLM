#!/usr/bin/env python3
"""Run the frozen 72-request SQ8 HTTP TTFT and decode latency gate.

The request client runs in the same Docker network as OpenWebUI.  The gate
measures only monotonic timestamps emitted by that client, synchronizes every
request with the gateway lifecycle observer, and later requires byte-exact
copies of those lifecycle messages in the authoritative service journal.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
import types
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO, NoReturn, Sequence, cast


MAX_SUPPORT_BYTES = 8 * 1024 * 1024
MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_HTTP_RESPONSE_BYTES = 8 * 1024 * 1024
REQUEST_TIMEOUT_NS = 300_000_000_000
RELEASE_DEADLINE_NS = 5_000_000_000
QUIET_DRAIN_NS = 250_000_000
GATE_SCHEMA = "ullm.sq8.http_latency_gate.v1"
EPOCH_SCHEMA = "ullm.sq8.resource_restart_epoch.v1"
HTTP_COMMAND_SCHEMA = "ullm.sq8.openwebui_http_client.command.v1"
HTTP_EVENT_SCHEMA = "ullm.sq8.openwebui_http_client.event.v1"
MODEL_ID = "ullm-qwen3-14b-sq8"
HTTP_TARGET = "/v1/chat/completions"
HTTP_NETWORK_NAME = "open-webui-network"
HTTP_CLIENT_SHA256 = "a64642a0f31bcdd92cf02883e195ee270b9752ee6117908b789cc66187053285"
OBSERVER_SOCKET = Path("/run/ullm/lifecycle-observer.sock")
FIXTURE_SCHEMA = "ullm.sq8.chat_template_fixture.v1"
FIXTURE_ORDER = (
    "exact-p0032",
    "exact-p0128",
    "exact-p0512",
    "exact-p2048",
    "exact-p3584",
)
FIXTURE_IDENTITIES = {
    "exact-p0032": (
        32,
        "c660c7fb3c25d2a3e25693e2beb2abc10295a06935772d17d23cedab04f24c07",
    ),
    "exact-p0128": (
        128,
        "f8fe81bacb8761f3aa10cce1c333a51f9a85d65b5bfc7b02499886fb9f550a37",
    ),
    "exact-p0512": (
        512,
        "e2f53c514a228e9e10871fc0df1867394aae12416215c9716770d2b420a3480f",
    ),
    "exact-p2048": (
        2048,
        "cd04c3339542f07731074ac0e00740a83061e620f6caff9c2a7e5316df1ccdcf",
    ),
    "exact-p3584": (
        3584,
        "e3cd6c722302f73d688492b73a182298f34cc0a1498def209c262e5e9aa92912",
    ),
}
TTFT_LIMITS_NS = {
    "exact-p0032": (2_500_000_000, 3_000_000_000),
    "exact-p0128": (4_000_000_000, 5_000_000_000),
    "exact-p0512": (10_000_000_000, 12_000_000_000),
    "exact-p2048": (30_000_000_000, 35_000_000_000),
    "exact-p3584": (50_000_000_000, 60_000_000_000),
}
DECODE_MIN_P50_TOKENS_PER_SECOND = 15
DECODE_MAX_P95_INTERVAL_NS = 100_000_000


class LatencyGateError(RuntimeError):
    """A fail-closed diagnostic containing only fixed implementation text."""


def fail(message: str) -> NoReturn:
    raise LatencyGateError(message)


def _bootstrap_snapshot(path: Path) -> tuple[bytes, tuple[int, ...]]:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > MAX_SUPPORT_BYTES
        ):
            fail("direct gate support is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = MAX_SUPPORT_BYTES + 1
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
            fail("direct gate support changed while it was read")
        return raw, identity
    except LatencyGateError:
        raise
    except OSError:
        fail("failed to read direct gate support without following links")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _load_direct_support() -> tuple[types.ModuleType, bytes, tuple[int, ...]]:
    path = Path(__file__).with_name("run-sq8-direct-cancel-gate.py")
    raw, identity = _bootstrap_snapshot(path)
    name = "_ullm_sq8_http_latency_direct_support"
    module = types.ModuleType(name)
    module.__file__ = os.fspath(path)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(raw, os.fspath(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, raw, identity


DIRECT, DIRECT_SUPPORT_RAW, DIRECT_SUPPORT_IDENTITY = _load_direct_support()
COL = DIRECT.COL
MODULE_IMPORT_RAW, MODULE_IMPORT_IDENTITY = DIRECT._single_fd_snapshot(
    Path(__file__), "imported HTTP latency gate implementation", MAX_INPUT_BYTES
)


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
        fail("failed to serialize bounded canonical JSON")


def exact_integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail(f"{label} is not an exact non-negative integer")
    return value


@dataclasses.dataclass(frozen=True)
class CaseSpec:
    sequence: int
    case_id: str
    workload: str
    sample_kind: str
    sample_index: int
    fixture_id: str
    prompt_tokens: int
    max_tokens: int
    auto_close: bool


def build_schedule() -> tuple[CaseSpec, ...]:
    result: list[CaseSpec] = []
    sequence = 0
    for fixture_id in FIXTURE_ORDER:
        prompt_tokens = FIXTURE_IDENTITIES[fixture_id][0]
        for sample_kind, count in (("warmup", 2), ("measured", 10)):
            for sample_index in range(1, count + 1):
                sequence += 1
                result.append(
                    CaseSpec(
                        sequence=sequence,
                        case_id=(f"ttft-{fixture_id}-{sample_kind}-{sample_index:02d}"),
                        workload="ttft",
                        sample_kind=sample_kind,
                        sample_index=sample_index,
                        fixture_id=fixture_id,
                        prompt_tokens=prompt_tokens,
                        max_tokens=512,
                        auto_close=True,
                    )
                )
    for sample_kind, count in (("warmup", 2), ("measured", 10)):
        for sample_index in range(1, count + 1):
            sequence += 1
            result.append(
                CaseSpec(
                    sequence=sequence,
                    case_id=f"decode64-{sample_kind}-{sample_index:02d}",
                    workload="decode64",
                    sample_kind=sample_kind,
                    sample_index=sample_index,
                    fixture_id="exact-p0032",
                    prompt_tokens=32,
                    max_tokens=64,
                    auto_close=False,
                )
            )
    if sequence != 72:
        fail("internal latency schedule length differs from 72")
    return tuple(result)


SCHEDULE = build_schedule()


def validate_schedule(schedule: Sequence[CaseSpec]) -> None:
    if tuple(schedule) != SCHEDULE:
        fail("HTTP latency request schedule differs from the frozen 72 cases")
    if len({item.case_id for item in schedule}) != 72:
        fail("HTTP latency case IDs are not unique")


@dataclasses.dataclass(frozen=True)
class FixtureSnapshot:
    fixture_id: str
    prompt_tokens: int
    messages: list[dict[str, str]]
    raw: bytes
    sha256: str
    identity: tuple[int, ...]


def load_fixture(path: Path, fixture_id: str) -> FixtureSnapshot:
    if fixture_id not in FIXTURE_IDENTITIES:
        fail("unknown frozen latency fixture")
    expected_prompt, expected_sha = FIXTURE_IDENTITIES[fixture_id]
    raw, identity = DIRECT._single_fd_snapshot(
        path, f"{fixture_id} fixture", MAX_INPUT_BYTES
    )
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        fail("fixed latency fixture SHA-256 differs")
    value = COL.strict_json_object(raw, f"{fixture_id} fixture")
    COL.exact_keys(
        value,
        {
            "schema_version",
            "fixture_id",
            "kind",
            "messages",
            "construction",
            "template_options",
            "expected",
        },
        f"{fixture_id} fixture",
    )
    expected = value["expected"]
    messages = value["messages"]
    if (
        value["schema_version"] != FIXTURE_SCHEMA
        or value["fixture_id"] != fixture_id
        or value["kind"] != "exact_length"
        or type(expected) is not dict
        or expected.get("prompt_tokens") != expected_prompt
        or value["template_options"]
        != {"add_generation_prompt": True, "enable_thinking": False}
        or type(messages) is not list
        or len(messages) != 1
    ):
        fail("fixed latency fixture contract differs")
    message = messages[0]
    if (
        type(message) is not dict
        or set(message) != {"role", "content"}
        or message["role"] != "user"
        or type(message["content"]) is not str
        or not message["content"]
    ):
        fail("fixed latency fixture message differs")
    return FixtureSnapshot(
        fixture_id=fixture_id,
        prompt_tokens=expected_prompt,
        messages=[{"role": "user", "content": message["content"]}],
        raw=raw,
        sha256=expected_sha,
        identity=identity,
    )


def request_body(fixture: FixtureSnapshot, max_tokens: int) -> bytes:
    if max_tokens not in {64, 512}:
        fail("HTTP latency max_tokens differs from the frozen cases")
    raw = compact_json(
        {
            "model": MODEL_ID,
            "messages": fixture.messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
        }
    )
    if len(raw) > COL.MAX_HTTP_BODY_BYTES:
        fail("HTTP latency request body exceeds its bound")
    return raw


@dataclasses.dataclass(frozen=True)
class TimedChunk:
    index: int
    raw: bytes
    observed_monotonic_ns: int


@dataclasses.dataclass(frozen=True)
class TimedSseItem:
    raw_data: bytes
    value: dict[str, Any] | None
    done: bool
    chunk_index: int
    observed_monotonic_ns: int


class TimedSseParser:
    """Parse SSE framing while retaining the final raw-chunk observation time."""

    def __init__(self) -> None:
        self.line = bytearray()
        self.data_lines: list[bytes] = []
        self.items: list[TimedSseItem] = []
        self.previous_cr = False
        self.event_bytes = 0
        self.last_chunk_index = -1
        self.last_timestamp = -1

    def feed(self, chunk: TimedChunk) -> None:
        if (
            type(chunk.index) is not int
            or chunk.index != self.last_chunk_index + 1
            or type(chunk.observed_monotonic_ns) is not int
            or chunk.observed_monotonic_ns < self.last_timestamp
            or type(chunk.raw) is not bytes
            or not chunk.raw
        ):
            fail("timed SSE chunk identity or ordering differs")
        self.last_chunk_index = chunk.index
        self.last_timestamp = chunk.observed_monotonic_ns
        for byte in chunk.raw:
            if self.previous_cr:
                self.previous_cr = False
                if byte == 0x0A:
                    continue
            if byte == 0x0D:
                self._finish_line(chunk)
                self.previous_cr = True
            elif byte == 0x0A:
                self._finish_line(chunk)
            else:
                self.line.append(byte)
                if len(self.line) > 1024 * 1024:
                    fail("SSE line exceeds its bounded size")

    def finish(self, *, allow_incomplete: bool) -> tuple[TimedSseItem, ...]:
        if self.previous_cr:
            self.previous_cr = False
        final_chunk = None
        if self.last_chunk_index >= 0:
            final_chunk = TimedChunk(self.last_chunk_index, b"x", self.last_timestamp)
        if self.line:
            if final_chunk is None:
                fail("SSE response lacks a final raw chunk")
            self._finish_line(final_chunk)
        if self.data_lines:
            if allow_incomplete:
                self.data_lines.clear()
                self.event_bytes = 0
            elif final_chunk is not None:
                self._dispatch(final_chunk)
        return tuple(self.items)

    def _finish_line(self, chunk: TimedChunk) -> None:
        line = bytes(self.line)
        self.line.clear()
        if not line:
            self._dispatch(chunk)
            return
        if line.startswith(b":"):
            return
        field, separator, value = line.partition(b":")
        if separator and value.startswith(b" "):
            value = value[1:]
        if field == b"data":
            self.event_bytes += len(value) + (1 if self.data_lines else 0)
            if self.event_bytes > 2 * 1024 * 1024:
                fail("SSE event exceeds its bounded size")
            self.data_lines.append(value)

    def _dispatch(self, chunk: TimedChunk) -> None:
        if not self.data_lines:
            self.event_bytes = 0
            return
        raw = b"\n".join(self.data_lines)
        self.data_lines.clear()
        self.event_bytes = 0
        if raw == b"[DONE]":
            item = TimedSseItem(
                raw, None, True, chunk.index, chunk.observed_monotonic_ns
            )
        else:
            value = cast(
                dict[str, Any], COL.strict_json_object(raw, "timed SSE data object")
            )
            item = TimedSseItem(
                raw, value, False, chunk.index, chunk.observed_monotonic_ns
            )
        self.items.append(item)


def parse_timed_sse(
    chunks: Sequence[TimedChunk], *, allow_incomplete: bool
) -> tuple[TimedSseItem, ...]:
    parser = TimedSseParser()
    for chunk in chunks:
        parser.feed(chunk)
    return parser.finish(allow_incomplete=allow_incomplete)


def nonempty_content_items(
    items: Sequence[TimedSseItem],
) -> list[TimedSseItem]:
    result: list[TimedSseItem] = []
    for item in items:
        if item.value is None:
            continue
        choices = item.value.get("choices")
        if type(choices) is not list or not choices or type(choices[0]) is not dict:
            continue
        delta = choices[0].get("delta")
        content = delta.get("content") if type(delta) is dict else None
        if type(content) is str and content:
            result.append(item)
    return result


@dataclasses.dataclass(frozen=True)
class HttpPlan:
    spec: CaseSpec
    body: bytes


@dataclasses.dataclass(frozen=True)
class HttpObservation:
    status: int
    outcome: str
    request_sent_monotonic_ns: int
    response_start_monotonic_ns: int
    response_end_monotonic_ns: int
    body: bytes
    chunks: tuple[TimedChunk, ...]
    items: tuple[TimedSseItem, ...]


def _completion_ids(items: Sequence[TimedSseItem]) -> set[str]:
    return {
        item.value["id"]
        for item in items
        if item.value is not None and type(item.value.get("id")) is str
    }


def validate_ttft_http(
    spec: CaseSpec, observation: HttpObservation, completion_id: str
) -> dict[str, Any]:
    if spec.workload != "ttft" or not spec.auto_close or spec.max_tokens != 512:
        fail("TTFT HTTP validator received a non-TTFT case")
    if observation.status != 200 or observation.outcome != "client_closed":
        fail("TTFT response did not end in the deliberate first-content close")
    contents = nonempty_content_items(observation.items)
    if not contents:
        fail("TTFT response lacks a non-empty content SSE object")
    if _completion_ids(observation.items) != {completion_id} or any(
        item.value is not None and item.value.get("id") != completion_id
        for item in observation.items
    ):
        fail("TTFT SSE and lifecycle completion identities differ")
    first = contents[0]
    if (
        first.observed_monotonic_ns < observation.request_sent_monotonic_ns
        or not observation.chunks
        or observation.chunks[-1].index != first.chunk_index
        or any(item.chunk_index != first.chunk_index for item in contents)
        or any(item.done for item in observation.items)
    ):
        fail("TTFT close or first-content raw-chunk boundary differs")
    for item in observation.items:
        if item.value is None:
            continue
        choices = item.value.get("choices")
        if type(choices) is list and choices and type(choices[0]) is dict:
            if choices[0].get("finish_reason") is not None:
                fail("TTFT close retained a terminal finish object")
        if item.value.get("usage") is not None:
            fail("TTFT close retained a terminal usage object")
    ttft_ns = first.observed_monotonic_ns - observation.request_sent_monotonic_ns
    if ttft_ns <= 0:
        fail("TTFT duration is not positive")
    return {
        "request_sent_monotonic_ns": observation.request_sent_monotonic_ns,
        "first_content_monotonic_ns": first.observed_monotonic_ns,
        "first_content_chunk_index": first.chunk_index,
        "ttft_ns": ttft_ns,
        "content_object_count": len(contents),
    }


def validate_decode_http(
    spec: CaseSpec, observation: HttpObservation, completion_id: str
) -> dict[str, Any]:
    if spec.workload != "decode64" or spec.auto_close or spec.max_tokens != 64:
        fail("decode HTTP validator received a non-decode case")
    if observation.status != 200 or observation.outcome != "eof":
        fail("decode response did not complete at EOF")
    contents = nonempty_content_items(observation.items)
    if len(contents) != 64:
        fail("decode response does not contain exactly 64 non-empty content objects")
    if _completion_ids(observation.items) != {completion_id} or any(
        item.value is not None and item.value.get("id") != completion_id
        for item in observation.items
    ):
        fail("decode SSE and lifecycle completion identities differ")
    done_positions = [
        index for index, item in enumerate(observation.items) if item.done
    ]
    if done_positions != [len(observation.items) - 1]:
        fail("decode [DONE] count or ordering differs")
    finish_reasons: list[Any] = []
    usage_counts: list[Any] = []
    for item in observation.items:
        if item.value is None:
            continue
        choices = item.value.get("choices")
        if type(choices) is list and choices and type(choices[0]) is dict:
            reason = choices[0].get("finish_reason")
            if reason is not None:
                finish_reasons.append(reason)
        usage = item.value.get("usage")
        if type(usage) is dict and "completion_tokens" in usage:
            usage_counts.append(usage["completion_tokens"])
    if finish_reasons != ["length"] or usage_counts != [64]:
        fail("decode terminal finish reason or usage count differs")
    timestamps = [item.observed_monotonic_ns for item in contents]
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        fail("decode content timestamps regressed")
    elapsed_ns = timestamps[-1] - timestamps[0]
    if elapsed_ns <= 0:
        fail("decode first-to-64th elapsed duration is not positive")
    intervals_ns = [right - left for left, right in zip(timestamps, timestamps[1:])]
    if len(intervals_ns) != 63:
        fail("decode consecutive-content interval count differs")
    return {
        "request_sent_monotonic_ns": observation.request_sent_monotonic_ns,
        "first_content_monotonic_ns": timestamps[0],
        "last_content_monotonic_ns": timestamps[-1],
        "content_timestamps_ns": timestamps,
        "decode_elapsed_ns": elapsed_ns,
        "decode_intervals_ns": intervals_ns,
        "decode_tokens_per_second": {
            "numerator": 63_000_000_000,
            "denominator": elapsed_ns,
        },
    }


def linear_percentile(values: Sequence[int], quantile: Fraction) -> Fraction:
    if (
        not values
        or not Fraction(0) <= quantile <= Fraction(1)
        or any(type(value) is not int or value < 0 for value in values)
    ):
        fail("linear percentile input differs from the frozen exact domain")
    ordered = sorted(values)
    rank = Fraction(len(ordered) - 1) * quantile
    lower = rank.numerator // rank.denominator
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return Fraction(ordered[lower]) + fraction * (
        Fraction(ordered[upper]) - Fraction(ordered[lower])
    )


def fraction_json(value: Fraction) -> int | dict[str, int]:
    if value.denominator == 1:
        return value.numerator
    return {"numerator": value.numerator, "denominator": value.denominator}


def derive_metrics(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) != 72:
        fail("latency metric input does not contain exactly 72 samples")
    for expected, item in zip(SCHEDULE, samples, strict=True):
        if type(item) is not dict or any(
            item.get(name) != value
            for name, value in (
                ("sequence", expected.sequence),
                ("case_id", expected.case_id),
                ("workload", expected.workload),
                ("sample_kind", expected.sample_kind),
                ("sample_index", expected.sample_index),
                ("fixture_id", expected.fixture_id),
            )
        ):
            fail("latency metric sample order or case identity differs")
        if expected.workload == "ttft":
            exact_integer(item.get("ttft_ns"), "latency TTFT sample", minimum=1)
        else:
            exact_integer(
                item.get("decode_elapsed_ns"), "latency decode elapsed", minimum=1
            )
            sample_intervals = item.get("decode_intervals_ns")
            if type(sample_intervals) is not list or len(sample_intervals) != 63:
                fail("latency decode sample interval count differs from 63")
            for interval in sample_intervals:
                exact_integer(interval, "latency decode interval")
    ttft: dict[str, dict[str, Any]] = {}
    for fixture_id in FIXTURE_ORDER:
        values = [
            exact_integer(item.get("ttft_ns"), "measured TTFT")
            for item in samples
            if item.get("workload") == "ttft"
            and item.get("fixture_id") == fixture_id
            and item.get("sample_kind") == "measured"
        ]
        if len(values) != 10:
            fail("measured TTFT population differs from ten")
        p50 = linear_percentile(values, Fraction(1, 2))
        p95 = linear_percentile(values, Fraction(19, 20))
        p50_limit, p95_limit = TTFT_LIMITS_NS[fixture_id]
        if p50 > p50_limit or p95 > p95_limit:
            fail("measured TTFT exceeds its frozen hard limit")
        ttft[fixture_id] = {
            "count": 10,
            "p50_ns": fraction_json(p50),
            "p95_ns": fraction_json(p95),
            "p50_maximum_ns": p50_limit,
            "p95_maximum_ns": p95_limit,
        }
    decode_samples = [
        item
        for item in samples
        if item.get("workload") == "decode64" and item.get("sample_kind") == "measured"
    ]
    if len(decode_samples) != 10:
        fail("measured decode population differs from ten")
    elapsed = [
        exact_integer(item.get("decode_elapsed_ns"), "decode elapsed", minimum=1)
        for item in decode_samples
    ]
    throughputs = [Fraction(63_000_000_000, value) for value in elapsed]
    ordered_throughputs = sorted(throughputs)
    throughput_p50 = (ordered_throughputs[4] + ordered_throughputs[5]) / 2
    intervals: list[int] = []
    for item in decode_samples:
        value = item.get("decode_intervals_ns")
        if type(value) is not list or len(value) != 63:
            fail("decode interval population per request differs from 63")
        intervals.extend(
            exact_integer(entry, "decode consecutive-content interval")
            for entry in value
        )
    if len(intervals) != 630:
        fail("pooled measured decode interval population differs from 630")
    interval_p95 = linear_percentile(intervals, Fraction(19, 20))
    if throughput_p50 < DECODE_MIN_P50_TOKENS_PER_SECOND:
        fail("measured decode p50 throughput is below its frozen hard limit")
    if interval_p95 > DECODE_MAX_P95_INTERVAL_NS:
        fail("measured decode p95 interval exceeds its frozen hard limit")
    return {
        "ttft": ttft,
        "decode64": {
            "request_count": 10,
            "interval_count": 630,
            "p50_tokens_per_second": fraction_json(throughput_p50),
            "minimum_p50_tokens_per_second": DECODE_MIN_P50_TOKENS_PER_SECOND,
            "p95_inter_content_ns": fraction_json(interval_p95),
            "maximum_p95_inter_content_ns": DECODE_MAX_P95_INTERVAL_NS,
        },
    }


class LifecycleTrace:
    """Validate one latency request's complete gateway lifecycle."""

    def __init__(self, spec: CaseSpec):
        self.spec = spec
        self.events: list[dict[str, Any]] = []
        self.request_id: str | None = None
        self.completion_id: str | None = None
        self.started = False
        self.last_progress = 0
        self.first_token = False
        self.cancel: dict[str, Any] | None = None
        self.release: dict[str, Any] | None = None
        self.last_timestamp = -1

    def consume(self, event: dict[str, Any]) -> None:
        value = cast(dict[str, Any], COL.validate_lifecycle_value(dict(event)))
        observed = exact_integer(
            value["observed_monotonic_ns"], "lifecycle observation timestamp"
        )
        if observed < self.last_timestamp:
            fail("latency lifecycle timestamps regressed")
        self.last_timestamp = observed
        name = value["event"]
        if name == "worker_fatal" or self.release is not None:
            fail("fatal or trailing lifecycle event occurred during latency")
        if name == "request_admitted":
            if self.request_id is not None or self.events:
                fail("latency admission is duplicated or out of order")
            if (
                value["prompt_tokens"] != self.spec.prompt_tokens
                or value["max_completion_tokens"] != self.spec.max_tokens
                or value["stream"] is not True
            ):
                fail("latency admission request shape differs")
            self.request_id = value["request_id"]
            self.completion_id = value["completion_id"]
        else:
            if self.request_id is None or self.completion_id is None:
                fail("latency lifecycle event precedes admission")
            if (
                value.get("request_id") != self.request_id
                or value.get("completion_id") != self.completion_id
            ):
                fail("latency lifecycle request identity differs")
            if value.get("stream", True) is not True:
                fail("latency lifecycle stream flag differs")
        if name == "request_started":
            if self.started or len(self.events) != 1:
                fail("latency start is duplicated or out of order")
            if value["prompt_tokens"] != self.spec.prompt_tokens:
                fail("latency started prompt length differs")
            self.started = True
        elif name == "request_progress":
            processed = value["processed_prompt_tokens"]
            expected = min(
                128 if self.last_progress == 0 else self.last_progress + 128,
                self.spec.prompt_tokens,
            )
            if (
                not self.started
                or self.first_token
                or self.cancel is not None
                or value["phase"] != "prefill"
                or value["prompt_tokens"] != self.spec.prompt_tokens
                or processed != expected
            ):
                fail("latency prefill progress sequence differs")
            self.last_progress = processed
        elif name == "request_first_token":
            if (
                not self.started
                or self.first_token
                or self.cancel is not None
                or self.last_progress != self.spec.prompt_tokens
                or value["completion_tokens"] != 1
            ):
                fail("latency first-token ordering differs")
            self.first_token = True
        elif name == "request_cancel_requested":
            if (
                self.spec.workload != "ttft"
                or not self.first_token
                or self.cancel is not None
                or value["reason"] != "client_disconnect"
            ):
                fail("TTFT cancellation ordering or reason differs")
            self.cancel = value
        elif name == "request_released":
            if (
                not self.started
                or not self.first_token
                or value["prompt_tokens"] != self.spec.prompt_tokens
                or value["reset_complete"] is not True
                or value["admit_to_release_ns"]
                != value["admit_to_start_ns"] + value["start_to_release_ns"]
            ):
                fail("latency release common result differs")
            if self.spec.workload == "ttft":
                if (
                    self.cancel is None
                    or value["outcome"] != "cancelled"
                    or value["cancel_reason"] != "client_disconnect"
                    or value["completion_tokens"] < 1
                ):
                    fail("TTFT release differs from cancelled/reset-complete")
                absolute_delay = observed - self.cancel["observed_monotonic_ns"]
                stored_delay = (
                    value["admit_to_release_ns"] - self.cancel["admit_to_cancel_ns"]
                )
                if (
                    absolute_delay < 0
                    or absolute_delay > RELEASE_DEADLINE_NS
                    or stored_delay < 0
                    or stored_delay > RELEASE_DEADLINE_NS
                ):
                    fail("TTFT cancellation release exceeded five seconds")
            elif (
                self.cancel is not None
                or value["outcome"] != "length"
                or value["cancel_reason"] is not None
                or value["completion_tokens"] != 64
            ):
                fail("decode release differs from length/64/reset-complete")
            self.release = value
        elif name not in {
            "request_admitted",
            "request_started",
            "request_progress",
            "request_first_token",
        }:
            fail("unsupported lifecycle event in latency trace")
        self.events.append(value)

    def complete(self, observation: HttpObservation) -> dict[str, Any]:
        if (
            self.release is None
            or self.request_id is None
            or self.completion_id is None
        ):
            fail("latency lifecycle trace is incomplete")
        http_result = (
            validate_ttft_http(self.spec, observation, self.completion_id)
            if self.spec.workload == "ttft"
            else validate_decode_http(self.spec, observation, self.completion_id)
        )
        if self.spec.workload == "ttft":
            assert self.cancel is not None
            first_content = http_result["first_content_monotonic_ns"]
            if first_content > self.cancel["observed_monotonic_ns"]:
                fail("TTFT gateway cancellation preceded client-visible content")
        return {
            "sequence": self.spec.sequence,
            "case_id": self.spec.case_id,
            "workload": self.spec.workload,
            "sample_kind": self.spec.sample_kind,
            "sample_index": self.spec.sample_index,
            "fixture_id": self.spec.fixture_id,
            "prompt_tokens": self.spec.prompt_tokens,
            "max_tokens": self.spec.max_tokens,
            "request_id": self.request_id,
            "completion_id": self.completion_id,
            "release_observed_monotonic_ns": self.release["observed_monotonic_ns"],
            "release_outcome": self.release["outcome"],
            "release_completion_tokens": self.release["completion_tokens"],
            **http_result,
        }


class LatencyRunValidator:
    """Enforce exact serial schedule and derive the hard latency metrics."""

    def __init__(self, schedule: Sequence[CaseSpec] = SCHEDULE):
        validate_schedule(schedule)
        self.schedule = tuple(schedule)
        self.index = 0
        self.active: LifecycleTrace | None = None
        self.samples: list[dict[str, Any]] = []
        self.last_release_ns = -1
        self.max_active = 0
        self.request_ids: set[str] = set()
        self.completion_ids: set[str] = set()

    def begin(self, spec: CaseSpec) -> None:
        if self.active is not None or self.index >= len(self.schedule):
            fail("latency request began while another was active or after schedule")
        if spec != self.schedule[self.index]:
            fail("latency request case differs from the frozen schedule")
        self.active = LifecycleTrace(spec)
        self.max_active = max(self.max_active, 1)

    def consume(self, event: dict[str, Any]) -> None:
        if self.active is None:
            fail("latency lifecycle event occurred without an active request")
        if event.get("event") == "request_admitted" and (
            type(event.get("observed_monotonic_ns")) is not int
            or event["observed_monotonic_ns"] <= self.last_release_ns
        ):
            fail("latency admission did not strictly follow the prior release")
        self.active.consume(event)

    def complete(self, observation: HttpObservation) -> dict[str, Any]:
        if self.active is None:
            fail("latency completion lacks an active request")
        sample = self.active.complete(observation)
        release_ns = sample["release_observed_monotonic_ns"]
        if type(release_ns) is not int or release_ns <= self.last_release_ns:
            fail("latency releases are not strictly ordered")
        request_id = sample["request_id"]
        completion_id = sample["completion_id"]
        if (
            type(request_id) is not str
            or type(completion_id) is not str
            or request_id in self.request_ids
            or completion_id in self.completion_ids
        ):
            fail("latency request or completion identity is reused")
        self.request_ids.add(request_id)
        self.completion_ids.add(completion_id)
        self.samples.append(sample)
        self.last_release_ns = release_ns
        self.index += 1
        self.active = None
        return sample

    def finalize(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if self.active is not None or self.index != len(self.schedule):
            fail("latency run did not complete the exact 72-request schedule")
        if self.max_active != 1 or len(self.samples) != 72:
            fail("latency request count or maximum active count differs")
        return list(self.samples), derive_metrics(self.samples)


class LatencyHttpClient:
    """Strict asynchronous controller for the fixed Docker HTTP client."""

    def __init__(self, command: Sequence[str], guard: Any, writer: Any):
        self.command = tuple(command)
        self.guard = guard
        self.writer = writer
        self.process: subprocess.Popen[bytes] | None = None
        self.reader: Any | None = None
        self.stderr: BinaryIO | None = None
        self.active: HttpPlan | None = None
        self.request_count = 0
        self.last_response_end_ns = -1

    def start(self) -> None:
        if self.process is not None:
            fail("HTTP latency client is already started")
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
            fail("failed to start the HTTP latency client")
        if self.process.stdin is None or self.process.stdout is None:
            fail("HTTP latency client pipes are unavailable")
        self.reader = COL.BoundedLineReader(self.process.stdout.fileno())
        event = self._read_event(time.monotonic_ns() + 30_000_000_000)
        COL.exact_keys(
            event,
            {"schema_version", "event", "observed_monotonic_ns"},
            "HTTP latency ready event",
        )
        if event["schema_version"] != HTTP_EVENT_SCHEMA or event["event"] != "ready":
            fail("HTTP latency client ready event differs")
        exact_integer(event["observed_monotonic_ns"], "HTTP client ready timestamp")

    def begin(self, plan: HttpPlan) -> None:
        if self.active is not None:
            fail("HTTP latency request began while another request was active")
        if (
            self.request_count >= len(SCHEDULE)
            or plan.spec != SCHEDULE[self.request_count]
        ):
            fail("HTTP client request order differs from the frozen schedule")
        self.guard.reject(plan.body, "HTTP latency request body")
        command = DIRECT.build_request_command(
            plan.spec.case_id, plan.body, plan.spec.auto_close
        )
        self._write_command(command)
        self.active = plan
        self.request_count += 1

    def finish(self, deadline_ns: int) -> HttpObservation:
        plan = self.active
        if plan is None:
            fail("HTTP latency response collection lacks an active request")
        saw_request = False
        saw_start = False
        status: int | None = None
        sent_ns = -1
        start_ns = -1
        end_ns = -1
        next_chunk = 0
        chunks: list[TimedChunk] = []
        digest = hashlib.sha256()
        total = 0
        outcome: str | None = None
        last_timestamp = -1
        while outcome is None:
            event = self._read_event(deadline_ns)
            if event.get("schema_version") != HTTP_EVENT_SCHEMA:
                fail("HTTP latency event schema_version differs")
            name = event.get("event")
            fields = {
                key: value
                for key, value in event.items()
                if key not in {"schema_version", "event"}
            }
            if name == "http_request":
                if saw_request:
                    fail("HTTP latency request evidence is duplicated")
                support_plan = COL.HttpPlan(
                    phase="latency",
                    case_id=plan.spec.case_id,
                    request_index=plan.spec.sequence,
                    request_key=plan.spec.case_id,
                    target=HTTP_TARGET,
                    body=plan.body,
                    expected_status=200,
                    expect_release=True,
                )
                connect_ns, sent_ns = (
                    COL.HttpClientProcess._validate_http_request_event(
                        fields, support_plan
                    )
                )
                if connect_ns < self.last_response_end_ns:
                    fail("HTTP latency request begins before the prior response ended")
                last_timestamp = sent_ns
                saw_request = True
            elif name == "http_response_start":
                if not saw_request or saw_start:
                    fail("HTTP latency response start ordering differs")
                COL.exact_keys(
                    fields,
                    {"request_key", "status", "headers", "observed_monotonic_ns"},
                    "HTTP latency response start",
                )
                if fields["request_key"] != plan.spec.case_id:
                    fail("HTTP latency response start request key differs")
                status = exact_integer(fields["status"], "HTTP response status", 100)
                if status != 200 or type(fields["headers"]) is not list:
                    fail("HTTP latency response status or headers differ")
                media: list[str] = []
                for pair in fields["headers"]:
                    if (
                        type(pair) is not list
                        or len(pair) != 2
                        or any(type(item) is not str for item in pair)
                    ):
                        fail("HTTP latency response header differs")
                    if pair[0].lower() == "content-type":
                        media.append(pair[1].split(";", 1)[0].strip().lower())
                if media != ["text/event-stream"]:
                    fail("HTTP latency Content-Type differs")
                start_ns = exact_integer(
                    fields["observed_monotonic_ns"], "HTTP response start timestamp"
                )
                if start_ns < last_timestamp:
                    fail("HTTP response start precedes the request send boundary")
                last_timestamp = start_ns
                saw_start = True
            elif name == "http_body_chunk":
                if not saw_start:
                    fail("HTTP latency body chunk precedes response start")
                COL.exact_keys(
                    fields,
                    {
                        "request_key",
                        "chunk_index",
                        "body_base64",
                        "body_sha256",
                        "body_bytes",
                        "observed_monotonic_ns",
                    },
                    "HTTP latency body chunk",
                )
                if (
                    fields["request_key"] != plan.spec.case_id
                    or fields["chunk_index"] != next_chunk
                ):
                    fail("HTTP latency body chunk correlation differs")
                raw = COL.decode_bound_bytes(fields, "HTTP latency body chunk")
                total += len(raw)
                if not raw or total > MAX_HTTP_RESPONSE_BYTES:
                    fail("HTTP latency response chunk or aggregate size differs")
                self.guard.reject(raw, "HTTP latency response body")
                observed = exact_integer(
                    fields["observed_monotonic_ns"], "HTTP body chunk timestamp"
                )
                if observed < last_timestamp:
                    fail("HTTP latency response timestamps regressed")
                chunks.append(TimedChunk(next_chunk, raw, observed))
                digest.update(raw)
                next_chunk += 1
                last_timestamp = observed
            elif name == "http_response_end":
                if not saw_request or not saw_start:
                    fail("HTTP latency response end precedes response evidence")
                COL.exact_keys(
                    fields,
                    {
                        "request_key",
                        "outcome",
                        "error",
                        "body_bytes",
                        "body_sha256",
                        "observed_monotonic_ns",
                    },
                    "HTTP latency response end",
                )
                if fields["request_key"] != plan.spec.case_id:
                    fail("HTTP latency response end request key differs")
                outcome = fields["outcome"]
                if outcome not in {"eof", "client_closed", "timeout", "error"}:
                    fail("HTTP latency response outcome differs")
                if (outcome in {"eof", "client_closed"}) != (fields["error"] is None):
                    fail("HTTP latency response error field differs from outcome")
                if (
                    fields["body_bytes"] != total
                    or fields["body_sha256"] != digest.hexdigest()
                ):
                    fail("HTTP latency response end aggregate differs")
                end_ns = exact_integer(
                    fields["observed_monotonic_ns"], "HTTP response end timestamp"
                )
                if end_ns < last_timestamp:
                    fail("HTTP latency response end timestamp regressed")
                self.last_response_end_ns = end_ns
            elif name == "command_error":
                fail("HTTP latency client rejected a gate command")
            else:
                fail("HTTP latency client emitted an unexpected event")
        assert status is not None and outcome is not None
        body = b"".join(chunk.raw for chunk in chunks)
        items = parse_timed_sse(chunks, allow_incomplete=outcome == "client_closed")
        self.active = None
        return HttpObservation(
            status=status,
            outcome=outcome,
            request_sent_monotonic_ns=sent_ns,
            response_start_monotonic_ns=start_ns,
            response_end_monotonic_ns=end_ns,
            body=body,
            chunks=tuple(chunks),
            items=items,
        )

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        pending: BaseException | None = None
        try:
            if self.active is not None or self.request_count != len(SCHEDULE):
                fail("HTTP latency client shutdown request count differs from 72")
            self._write_command(
                {"schema_version": HTTP_COMMAND_SCHEMA, "command": "shutdown"}
            )
            event = self._read_event(time.monotonic_ns() + 5_000_000_000)
            COL.exact_keys(
                event,
                {"schema_version", "event", "observed_monotonic_ns"},
                "HTTP latency shutdown event",
            )
            if (
                event["schema_version"] != HTTP_EVENT_SCHEMA
                or event["event"] != "shutdown_complete"
            ):
                fail("HTTP latency shutdown acknowledgement differs")
            exact_integer(
                event["observed_monotonic_ns"], "HTTP client shutdown timestamp"
            )
            if process.wait(timeout=5.0) != 0:
                fail("HTTP latency client exited nonzero")
            if self.reader is None or process.stdout is None:
                fail("HTTP latency shutdown stream state is unavailable")
            if self.reader.buffer or os.read(process.stdout.fileno(), 1):
                fail("HTTP latency client emitted trailing stdout")
        except BaseException as error:
            pending = error
            if process.poll() is None:
                COL.terminate_process_group(process)
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
            COL.terminate_process_group(process)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self._check_stderr(require_empty=False)
        self.process = None

    def _check_stderr(self, *, require_empty: bool = True) -> None:
        if self.stderr is None:
            return
        self.stderr.seek(0)
        raw = self.stderr.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            fail("HTTP latency client stderr exceeds its bound")
        self.guard.reject(raw, "HTTP latency client stderr")
        self.stderr.close()
        self.stderr = None
        if require_empty and raw:
            fail("HTTP latency client emitted stderr")

    def _write_command(self, value: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None or process.poll() is not None:
            fail("HTTP latency client is not running")
        raw = compact_json(value)
        self.guard.reject(raw, "HTTP latency client command")
        try:
            process.stdin.write(raw + b"\n")
            process.stdin.flush()
        except OSError:
            fail("failed to write an HTTP latency client command")

    def _read_event(self, deadline_ns: int) -> dict[str, Any]:
        if self.reader is None:
            fail("HTTP latency client reader is unavailable")
        raw = self.reader.read(deadline_ns, "HTTP latency client event")
        self.guard.reject(raw, "HTTP latency client event")
        self.writer.write(raw, "raw HTTP latency evidence")
        return cast(dict[str, Any], COL.strict_json_object(raw, "HTTP latency event"))


@dataclasses.dataclass(frozen=True)
class EpochSnapshot:
    service_identity: Any
    raw: bytes
    sha256: str
    identity: tuple[int, ...]


def load_epoch(path: Path) -> EpochSnapshot:
    raw, identity = DIRECT._single_fd_snapshot(
        path, "resource-restart epoch", MAX_INPUT_BYTES
    )
    value = COL.strict_json_object(raw, "resource-restart epoch")
    COL.exact_keys(
        value,
        {"schema_version", "phase", "service_identity"},
        "resource-restart epoch",
    )
    if value["schema_version"] != EPOCH_SCHEMA or value["phase"] != "resource_restart":
        fail("resource-restart epoch schema or phase differs")
    fields = COL.exact_keys(
        value["service_identity"],
        {field.name for field in dataclasses.fields(DIRECT.ServiceIdentity)},
        "resource-restart service identity",
    )
    for name in (
        "uid",
        "gid",
        "gateway_pid",
        "gateway_starttime_ticks",
        "worker_pid",
        "worker_starttime_ticks",
        "n_restarts",
    ):
        fields[name] = exact_integer(
            fields[name], f"resource-restart service identity {name}"
        )
    for name in ("unit", "user", "control_group", "boot_id"):
        if type(fields[name]) is not str or not fields[name]:
            fail("resource-restart service string identity differs")
    if (
        fields["unit"] != DIRECT.SERVICE_UNIT
        or fields["gateway_pid"] <= 0
        or fields["worker_pid"] <= 0
        or fields["gateway_starttime_ticks"] <= 0
        or fields["worker_starttime_ticks"] <= 0
        or fields["n_restarts"] < 1
        or re.fullmatch(r"[0-9a-f]{32}", fields["boot_id"]) is None
    ):
        fail("resource-restart epoch does not identify a restarted live generation")
    return EpochSnapshot(
        service_identity=DIRECT.ServiceIdentity(**fields),
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
        identity=identity,
    )


def require_epoch(expected: Any) -> Any:
    observed = DIRECT.capture_service_identity()
    if observed != expected:
        fail("service identity differs from the resource-restart epoch")
    return observed


class LatencyGate:
    def __init__(
        self,
        identity: Any,
        image_id: str,
        network_id: str,
        fixtures: dict[str, FixtureSnapshot],
        observer: Any,
        observer_writer: Any,
        correlation_writer: Any,
        sample_writer: Any,
        journal: Any,
        http: LatencyHttpClient,
    ):
        self.identity = identity
        self.image_id = image_id
        self.network_id = network_id
        self.fixtures = fixtures
        self.observer = observer
        self.observer_writer = observer_writer
        self.correlation_writer = correlation_writer
        self.sample_writer = sample_writer
        self.journal = journal
        self.http = http
        self.validator = LatencyRunValidator()
        self.observer_records: list[Any] = []
        self.correlation_count = 0

    def run(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        validate_schedule(SCHEDULE)
        for spec in SCHEDULE:
            self._run_case(spec)
        samples, metrics = self.validator.finalize()
        time.sleep(QUIET_DRAIN_NS / 1_000_000_000)
        self.observer.require_empty()
        self._correlate(time.monotonic_ns() + RELEASE_DEADLINE_NS)
        require_epoch(self.identity)
        DIRECT.require_ready(self.image_id)
        DIRECT.validate_docker_identity(self.image_id, self.network_id)
        return samples, metrics

    def _run_case(self, spec: CaseSpec) -> None:
        require_epoch(self.identity)
        fixture = self.fixtures[spec.fixture_id]
        self.validator.begin(spec)
        self.http.begin(HttpPlan(spec, request_body(fixture, spec.max_tokens)))
        deadline_ns = time.monotonic_ns() + REQUEST_TIMEOUT_NS
        observation = self.http.finish(deadline_ns)
        # Drain the HTTP child's bounded stdout before lifecycle synchronization.
        # A complete decode response can otherwise fill the pipe while this process
        # waits on the independent observer socket.
        self._wait_for_release(deadline_ns)
        sample = self.validator.complete(observation)
        self.sample_writer.write(compact_json(sample), "latency sample evidence")
        self._correlate(time.monotonic_ns() + RELEASE_DEADLINE_NS)
        require_epoch(self.identity)
        DIRECT.require_ready(self.image_id)

    def _receive_one(self, deadline_ns: int) -> None:
        datagram = self.observer.receive(
            deadline_ns, expected_sender_pid=self.identity.gateway_pid
        )
        if (
            datagram.sender_uid != self.identity.uid
            or datagram.sender_gid != self.identity.gid
        ):
            fail("lifecycle observer sender UID or GID differs from the service")
        self.observer_writer.write(
            datagram.raw_payload, "raw latency lifecycle observer payload"
        )
        self.observer_records.append(datagram)
        self.validator.consume(datagram.event)
        self.journal.poll()

    def _wait_for_release(self, request_deadline_ns: int) -> None:
        while True:
            active = self.validator.active
            if active is None:
                fail("latency lifecycle wait lacks an active trace")
            if active.release is not None:
                return
            deadline = request_deadline_ns
            if active.cancel is not None:
                deadline = min(
                    deadline,
                    active.cancel["observed_monotonic_ns"] + RELEASE_DEADLINE_NS,
                )
            self._receive_one(deadline)

    def _correlate(self, deadline_ns: int) -> None:
        correlations = self.journal.wait_correlated(self.observer_records, deadline_ns)
        if len(correlations) < self.correlation_count:
            fail("latency observer-to-journal correlation count regressed")
        for value in correlations[self.correlation_count :]:
            self.correlation_writer.write(
                compact_json({"schema_version": GATE_SCHEMA, **value}),
                "latency observer journal correlation evidence",
            )
        self.correlation_count = len(correlations)


def write_sha256sums(stage: Path, paths: Sequence[str], guard: Any) -> bytes:
    if len(paths) != len(set(paths)) or "SHA256SUMS" in paths:
        fail("checksum input path set differs")
    lines: list[bytes] = []
    for name in sorted(paths, key=lambda item: item.encode("ascii")):
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name) is None:
            fail("checksum artifact name syntax differs")
        raw, _identity = DIRECT._single_fd_snapshot(
            stage / name, f"checksum input {name}", DIRECT.MAX_RAW_BYTES
        )
        guard.reject(raw, f"checksum input {name}")
        lines.append(f"{hashlib.sha256(raw).hexdigest()}  {name}\n".encode("ascii"))
    document = b"".join(lines)
    guard.reject(document, "SHA256SUMS")
    descriptor = -1
    try:
        descriptor = os.open(
            stage / "SHA256SUMS",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        offset = 0
        while offset < len(document):
            written = os.write(descriptor, document[offset:])
            if written <= 0:
                fail("SHA256SUMS write was short")
            offset += written
        os.fsync(descriptor)
    except LatencyGateError:
        raise
    except OSError:
        fail("failed to create SHA256SUMS")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return document


def verify_sha256sums(stage: Path, paths: Sequence[str], expected: bytes) -> None:
    raw, _identity = DIRECT._single_fd_snapshot(
        stage / "SHA256SUMS", "sealed SHA256SUMS", DIRECT.MAX_RAW_BYTES
    )
    if raw != expected:
        fail("SHA256SUMS changed before publication")
    lines = raw.decode("ascii", errors="strict").splitlines()
    ordered = sorted(paths, key=lambda item: item.encode("ascii"))
    if len(lines) != len(ordered):
        fail("SHA256SUMS entry count differs")
    for line, name in zip(lines, ordered, strict=True):
        match = re.fullmatch(
            r"([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._-]{0,127})", line
        )
        if match is None or match.group(2) != name:
            fail("SHA256SUMS entry syntax or ordering differs")
        artifact, _artifact_identity = DIRECT._single_fd_snapshot(
            stage / name, f"verified checksum input {name}", DIRECT.MAX_RAW_BYTES
        )
        if hashlib.sha256(artifact).hexdigest() != match.group(1):
            fail("SHA256SUMS artifact digest differs")


@dataclasses.dataclass(frozen=True)
class Arguments:
    output_dir: Path
    api_key_file: Path
    http_image_id: str
    docker_network_id: str
    expected_epoch_file: Path


def execute(args: Arguments) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    fixture_root = repo_root / "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures"
    output = DIRECT.AtomicRunDirectory(args.output_dir)
    writers: list[Any] = []
    snapshots: Any | None = None
    observer: Any | None = None
    http: LatencyHttpClient | None = None
    published = False
    try:
        gate_raw, gate_identity = MODULE_IMPORT_RAW, MODULE_IMPORT_IDENTITY
        DIRECT.verify_snapshot(
            Path(__file__),
            "imported HTTP latency gate implementation",
            gate_raw,
            gate_identity,
        )
        client_path = Path(__file__).with_name("sq8-openwebui-http-client.py")
        client_raw, client_identity = DIRECT._single_fd_snapshot(
            client_path, "HTTP evidence client implementation", MAX_INPUT_BYTES
        )
        if hashlib.sha256(client_raw).hexdigest() != HTTP_CLIENT_SHA256:
            fail("HTTP evidence client implementation SHA-256 differs")
        epoch = load_epoch(args.expected_epoch_file)
        fixtures = {
            fixture_id: load_fixture(fixture_root / f"{fixture_id}.json", fixture_id)
            for fixture_id in FIXTURE_ORDER
        }
        guard, credential_raw = COL.SecretGuard.snapshot_from_file(args.api_key_file)
        for label, raw in (
            ("latency gate implementation", gate_raw),
            ("direct gate support", DIRECT_SUPPORT_RAW),
            ("collector support", DIRECT.COLLECTOR_SUPPORT_RAW),
            ("HTTP client implementation", client_raw),
            ("resource-restart epoch", epoch.raw),
            *((f"{item.fixture_id} fixture", item.raw) for item in fixtures.values()),
        ):
            guard.reject(raw, label)
        DIRECT.validate_docker_identity(args.http_image_id, args.docker_network_id)
        identity = require_epoch(epoch.service_identity)
        if identity.uid != COL.HTTP_CLIENT_UID or identity.gid != COL.HTTP_CLIENT_GID:
            fail("service identity differs from fixed HTTP client UID or GID")
        DIRECT.require_ready(args.http_image_id)

        snapshots = COL.RuntimeSnapshots.create(client_raw, credential_raw)
        config = types.SimpleNamespace(
            identities={"openwebui": {"derived_image_id": args.http_image_id}}
        )
        http_command = COL.build_http_client_command(config, snapshots)
        http_writer = DIRECT.RawWriter(output.stage / "http-client.raw.jsonl", guard)
        observer_writer = DIRECT.RawWriter(output.stage / "observer.raw.jsonl", guard)
        journal_writer = DIRECT.RawWriter(
            output.stage / "service-journal.raw.jsonl", guard
        )
        correlation_writer = DIRECT.RawWriter(
            output.stage / "observer-journal-correlation.raw.jsonl", guard
        )
        sample_writer = DIRECT.RawWriter(output.stage / "samples.raw.jsonl", guard)
        writers.extend(
            [
                http_writer,
                observer_writer,
                journal_writer,
                correlation_writer,
                sample_writer,
            ]
        )
        journal = DIRECT.JournalCapture(
            identity.boot_id, identity.gateway_pid, journal_writer, guard
        )
        observer = COL.LifecycleObserver(
            OBSERVER_SOCKET,
            guard,
            expected_uid=identity.uid,
            expected_gid=identity.gid,
        )
        observer.open()
        journal.start()
        http = LatencyHttpClient(http_command, guard, http_writer)
        http.start()
        snapshots.unlink_credential()

        gate = LatencyGate(
            identity,
            args.http_image_id,
            args.docker_network_id,
            fixtures,
            observer,
            observer_writer,
            correlation_writer,
            sample_writer,
            journal,
            http,
        )
        samples, metrics = gate.run()
        http.close()
        time.sleep(QUIET_DRAIN_NS / 1_000_000_000)
        observer.require_empty()
        journal.poll()
        final_correlations = journal.wait_correlated(
            gate.observer_records, time.monotonic_ns() + RELEASE_DEADLINE_NS
        )
        if (
            len(gate.observer_records) != gate.correlation_count
            or len(final_correlations) != gate.correlation_count
        ):
            fail("final latency observer-to-journal correlation is incomplete")
        observer.require_empty()
        observer.close()
        observer = None
        time.sleep(QUIET_DRAIN_NS / 1_000_000_000)
        journal.poll()
        post_close_correlations = DIRECT.correlate_records(
            gate.observer_records, journal.records, identity.gateway_pid
        )
        if len(post_close_correlations) != gate.correlation_count:
            fail("post-observer-close latency journal correlation differs")
        require_epoch(identity)
        DIRECT.require_ready(args.http_image_id)
        DIRECT.validate_docker_identity(args.http_image_id, args.docker_network_id)

        for writer in writers:
            writer.close()
        manifest = {
            "schema_version": GATE_SCHEMA,
            "record_type": "input_manifest",
            "inputs": [
                {
                    "path": "tools/run-sq8-http-latency-gate.py",
                    "bytes": len(gate_raw),
                    "sha256": hashlib.sha256(gate_raw).hexdigest(),
                },
                {
                    "path": "tools/run-sq8-direct-cancel-gate.py",
                    "bytes": len(DIRECT_SUPPORT_RAW),
                    "sha256": hashlib.sha256(DIRECT_SUPPORT_RAW).hexdigest(),
                },
                {
                    "path": "tools/collect-sq8-openwebui-release.py",
                    "bytes": len(DIRECT.COLLECTOR_SUPPORT_RAW),
                    "sha256": hashlib.sha256(DIRECT.COLLECTOR_SUPPORT_RAW).hexdigest(),
                },
                {
                    "path": "tools/sq8-openwebui-http-client.py",
                    "bytes": len(client_raw),
                    "sha256": hashlib.sha256(client_raw).hexdigest(),
                },
                {
                    "path": "resource-restart-epoch.json",
                    "bytes": len(epoch.raw),
                    "sha256": epoch.sha256,
                },
                *[
                    {
                        "path": (
                            "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures/"
                            f"{item.fixture_id}.json"
                        ),
                        "bytes": len(item.raw),
                        "sha256": item.sha256,
                    }
                    for item in fixtures.values()
                ],
            ],
            "schedule": [dataclasses.asdict(item) for item in SCHEDULE],
            "request_bodies": [
                {
                    "fixture_id": fixture_id,
                    "max_tokens": max_tokens,
                    "bytes": len(request_body(fixtures[fixture_id], max_tokens)),
                    "sha256": hashlib.sha256(
                        request_body(fixtures[fixture_id], max_tokens)
                    ).hexdigest(),
                }
                for fixture_id, max_tokens in (
                    *((fixture_id, 512) for fixture_id in FIXTURE_ORDER),
                    ("exact-p0032", 64),
                )
            ],
        }
        DIRECT.write_json_file(output.stage / "input-manifest.json", manifest, guard)
        summary = {
            "schema_version": GATE_SCHEMA,
            "record_type": "summary",
            "passed": True,
            "request_count": len(samples),
            "max_active": gate.validator.max_active,
            "service_identity": dataclasses.asdict(identity),
            "resource_restart_epoch_sha256": epoch.sha256,
            "http_image_id": args.http_image_id,
            "docker_network_name": HTTP_NETWORK_NAME,
            "docker_network_id": args.docker_network_id,
            "observer_socket": os.fspath(OBSERVER_SOCKET),
            "observer_event_count": len(gate.observer_records),
            "journal_correlation_count": gate.correlation_count,
            "metrics": metrics,
            "artifacts": {
                writer.path.name: {
                    "bytes": writer.bytes_written,
                    "lines": writer.lines_written,
                    "sha256": writer.digest.hexdigest(),
                }
                for writer in writers
            },
        }
        DIRECT.write_json_file(output.stage / "summary.json", summary, guard)

        DIRECT.verify_snapshot(
            Path(__file__),
            "HTTP latency gate implementation",
            gate_raw,
            gate_identity,
        )
        direct_path = Path(__file__).with_name("run-sq8-direct-cancel-gate.py")
        DIRECT.verify_snapshot(
            direct_path,
            "direct gate support",
            DIRECT_SUPPORT_RAW,
            DIRECT_SUPPORT_IDENTITY,
        )
        collector_path = Path(__file__).with_name("collect-sq8-openwebui-release.py")
        DIRECT.verify_snapshot(
            collector_path,
            "collector support",
            DIRECT.COLLECTOR_SUPPORT_RAW,
            DIRECT.COLLECTOR_SUPPORT_IDENTITY,
        )
        DIRECT.verify_snapshot(
            client_path,
            "HTTP evidence client implementation",
            client_raw,
            client_identity,
        )
        DIRECT.verify_snapshot(
            args.expected_epoch_file,
            "resource-restart epoch",
            epoch.raw,
            epoch.identity,
        )
        for item in fixtures.values():
            DIRECT.verify_snapshot(
                fixture_root / f"{item.fixture_id}.json",
                f"{item.fixture_id} fixture",
                item.raw,
                item.identity,
            )
        for writer in writers:
            DIRECT.verify_raw_writer(writer)
        DIRECT.verify_json_file(output.stage / "input-manifest.json", manifest)
        DIRECT.verify_json_file(output.stage / "summary.json", summary)
        artifact_names = [writer.path.name for writer in writers] + [
            "input-manifest.json",
            "summary.json",
        ]
        checksums = write_sha256sums(output.stage, artifact_names, guard)
        verify_sha256sums(output.stage, artifact_names, checksums)
        for path in output.stage.iterdir():
            guard.scan_file(path, f"staged latency output {path.name}")
        snapshots.close()
        snapshots = None
        output.publish()
        published = True
        return {
            "schema_version": GATE_SCHEMA,
            "output_dir": os.fspath(output.final_path),
            "request_count": len(samples),
        }
    finally:
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
    parser.add_argument("--expected-epoch-file", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    arguments = Arguments(
        output_dir=namespace.output_dir,
        api_key_file=namespace.api_key_file,
        http_image_id=namespace.http_image_id,
        docker_network_id=namespace.docker_network_id,
        expected_epoch_file=namespace.expected_epoch_file,
    )
    try:
        result = execute(arguments)
    except (LatencyGateError, DIRECT.GateError, COL.CollectorError) as error:
        print(f"SQ8 HTTP latency gate: {error}", file=sys.stderr)
        return 2
    print(compact_json(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
