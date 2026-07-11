#!/usr/bin/env python3
"""Independently reconstruct final latency and resource campaign metrics."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import stat
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, NoReturn, Sequence, cast


LATENCY_RESULTS_SCHEMA = "ullm.sq8.openwebui_release.prefill_latency_results.v1"
RESOURCE_INPUT_SCHEMA = "ullm.sq8.release_measurement.raw.v1"

HASH_CHUNK_BYTES = 64 << 10
MAX_RESOURCE_BYTES = 512 << 20
MAX_RESOURCE_LINE_BYTES = 1 << 20
MAX_GPU_METRIC_BYTES = 16 << 20
MAX_TEXT_BYTES = 8192
MAX_IDENTIFIER_BYTES = 256
RELEASE_MAX_NS = 5_000_000_000

FIXTURE_ORDER = (
    ("exact-p0032", 32),
    ("exact-p0128", 128),
    ("exact-p0512", 512),
    ("exact-p2048", 2048),
    ("exact-p3584", 3584),
)
LATENCY_REQUEST_BODY_IDENTITIES = {
    ("exact-p0032", 512): (
        218,
        "bb7fec4a8b357317ecfdb584a7619b2d07f52502f0b903453f1384ed6cbae045",
    ),
    ("exact-p0128", 512): (
        410,
        "bbb888a87865a2200ec167defa319ba1f02f3640f4b7b9fd77f3d0f8d964d364",
    ),
    ("exact-p0512", 512): (
        1178,
        "578d206ecdac06cafb322d751bbfd5137396646bddc636b58da8dd070130b33e",
    ),
    ("exact-p2048", 512): (
        4250,
        "9ff2c9350742aa91973f75139b5d987d7ad21268c98d5d88c648457cb53a7b95",
    ),
    ("exact-p3584", 512): (
        7322,
        "9dca9ab7c71e2c24b0979c3098341b7ae8716e71698c88aca92d1898458fa90a",
    ),
    ("exact-p0032", 64): (
        217,
        "cb76274bb192f1aeedce3ac989f7c63aaceccd8bcf53024517a574cc42d14dd1",
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

RESOURCE_SCHEDULE = {
    "normal_warmups": 10,
    "normal_requests": 100,
    "restart_warmups": 10,
    "restart_requests": 20,
    "idle_settle_ms": 5000,
    "samples_per_point": 5,
    "sample_interval_ms": 1000,
}
RESOURCE_COMMANDS = {
    "systemd_version": "systemctl --version",
    "service_identity": (
        "systemctl show ullm-openai.service --property=ControlGroup "
        "--property=MainPID --no-pager"
    ),
    "cgroup_type": "stat -fc %T /sys/fs/cgroup",
    "host_memory": "cat /sys/fs/cgroup${ControlGroup}/memory.current",
    "proc_stat": "cat /proc/${PID}/stat",
    "proc_status": "cat /proc/${PID}/status",
    "proc_exe": "readlink /proc/${PID}/exe",
    "proc_fds": "find -P /proc/${PID}/fd -mindepth 1 -maxdepth 1 -printf '%f\\n'",
    "proc_children": "cat /proc/${PID}/task/${PID}/children",
    "amd_smi_version": "amd-smi version",
    "amd_smi_list": "amd-smi list --json",
    "amd_smi_process": "amd-smi process --gpu 2 --general --json",
    "amd_smi_metric": "amd-smi metric --gpu 2 --json",
    "kfd_proc_probe": "test -d /sys/class/kfd/kfd/proc",
    "kfd_processes": (
        "find -P /sys/class/kfd/kfd/proc -mindepth 1 -maxdepth 1 -printf '%f\\n'"
    ),
    "kfd_vram": "cat /sys/class/kfd/kfd/proc/${PID}/vram_51545",
}
RESOURCE_TOOL_VERSIONS = {
    "amd_smi_tool": "26.2.2+e1a6bc5663",
    "amd_smi_library": "26.2.2",
    "rocm": "7.2.1",
}
GPU_INDEX = 2
GPU_BDF = "0000:47:00.0"
GPU_UUID = "a8ff7551-0000-1000-80e9-ddefa2d60f55"
KFD_GPU_ID = 51_545
RESOURCE_METRICS = (
    "memory_current_bytes",
    "process_vram_bytes",
    "gateway_rss_bytes",
    "worker_rss_bytes",
)
STABLE_COUNTS = (
    "gateway_threads",
    "gateway_fds",
    "gateway_children",
    "worker_threads",
    "worker_fds",
    "worker_children",
)
FINAL_DELTA_MAX_BYTES = 67_108_864
THEIL_SEN_MAX_BYTES_PER_REQUEST = 262_144

RESOURCE_HEADER_FIELDS = {
    "schema_version",
    "record_type",
    "service_unit",
    "commands",
    "tools",
    "probes",
    "schedule",
}
RESOURCE_SAMPLE_FIELDS = {
    "schema_version",
    "record_type",
    "segment",
    "phase",
    "request_index",
    "request_id",
    "release_outcome",
    "release_observed_monotonic_ns",
    "reset_complete",
    "idle_settle_started_monotonic_ns",
    "sample_index",
    "sample_monotonic_ns",
    "systemd",
    "host",
    "gateway",
    "worker",
    "gpu",
}
SYSTEMD_FIELDS = {
    "control_group_before",
    "control_group_after",
    "main_pid_before",
    "main_pid_after",
}
PROCESS_FIELDS = {
    "pid",
    "ppid",
    "exe",
    "starttime_ticks_before",
    "starttime_ticks_after",
    "vmrss_kb",
    "vmrss_bytes",
    "threads",
    "fd_count",
    "children",
}
GPU_FIELDS = {
    "index",
    "bdf",
    "uuid",
    "kfd_gpu_id",
    "process_record_count",
    "worker_pid",
    "mem_usage",
    "kfd_vram_bytes",
    "unrelated_process_pids",
}
GPU_METRIC_FIELDS = {
    "schema_version",
    "record_type",
    "segment",
    "boundary",
    "captured_monotonic_ns",
    "gpu_index",
    "raw_output_file",
    "raw_output_sha256",
}


class IndependentMetricsError(RuntimeError):
    """A fail-closed independent metric reconstruction error."""


def fail(message: str) -> NoReturn:
    raise IndependentMetricsError(message)


def _attribute(value: object, name: str, label: str) -> Any:
    try:
        return getattr(value, name)
    except AttributeError as error:
        raise IndependentMetricsError(f"{label} lacks {name}") from error


def _exact_int(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        fail(f"{label} is not an exact integer in range")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        fail(f"{label} is not a boolean")
    return value


def _text(value: Any, label: str, *, maximum_bytes: int = MAX_TEXT_BYTES) -> str:
    if type(value) is not str or not value or "\0" in value:
        fail(f"{label} is not bounded non-empty text")
    try:
        raw = value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise IndependentMetricsError(f"{label} is not strict UTF-8") from error
    if len(raw) > maximum_bytes:
        fail(f"{label} exceeds its UTF-8 byte bound")
    return value


def _sha256(value: Any, label: str) -> str:
    text = _text(value, label, maximum_bytes=64)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        fail(f"{label} is not lowercase SHA-256")
    return text


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        fail(f"{label} fields differ")
    return cast(dict[str, Any], value)


def _mapping_attribute(value: object, name: str, label: str) -> dict[Any, Any]:
    result = _attribute(value, name, label)
    if type(result) is not dict:
        fail(f"{label}.{name} is not an exact mapping")
    return result


def _tuple_attribute(value: object, name: str, label: str) -> tuple[Any, ...]:
    result = _attribute(value, name, label)
    if type(result) is not tuple:
        fail(f"{label}.{name} is not an exact tuple")
    return result


def _fraction_json(value: Fraction) -> int | dict[str, int]:
    if value.denominator == 1:
        return value.numerator
    return {"numerator": value.numerator, "denominator": value.denominator}


def _median(values: Iterable[int | Fraction]) -> Fraction:
    converted: list[Fraction] = []
    for value in values:
        if type(value) is int:
            converted.append(Fraction(value))
        elif type(value) is Fraction:
            converted.append(value)
        else:
            fail("median input contains a non-exact value")
    converted.sort()
    if not converted:
        fail("median input is empty")
    middle = len(converted) // 2
    if len(converted) % 2:
        return converted[middle]
    return (converted[middle - 1] + converted[middle]) / 2


def _linear_percentile(values: Sequence[int], probability: Fraction) -> Fraction:
    if (
        not values
        or type(probability) is not Fraction
        or probability < 0
        or probability > 1
        or any(type(value) is not int or value < 0 for value in values)
    ):
        fail("linear percentile input differs")
    ordered = sorted(values)
    rank = Fraction(len(ordered) - 1) * probability
    lower = rank.numerator // rank.denominator
    upper = lower if rank.denominator == 1 else lower + 1
    if lower == upper:
        return Fraction(ordered[lower])
    return Fraction(ordered[lower]) + (rank - lower) * (ordered[upper] - ordered[lower])


def _theil_sen(values: Sequence[Fraction]) -> Fraction:
    if len(values) < 2 or any(type(value) is not Fraction for value in values):
        fail("Theil-Sen input differs")
    slopes = [
        (values[right] - values[left]) / (right - left)
        for left in range(len(values))
        for right in range(left + 1, len(values))
    ]
    if len(slopes) != len(values) * (len(values) - 1) // 2:
        fail("Theil-Sen pair population differs")
    return _median(slopes)


@dataclasses.dataclass(frozen=True)
class _LatencySpec:
    sequence: int
    case_id: str
    workload: str
    sample_kind: str
    sample_index: int
    fixture_id: str
    prompt_tokens: int
    max_tokens: int


def _latency_schedule() -> tuple[_LatencySpec, ...]:
    result: list[_LatencySpec] = []
    sequence = 0
    for fixture_id, prompt_tokens in FIXTURE_ORDER:
        for sample_kind, count in (("warmup", 2), ("measured", 10)):
            for sample_index in range(1, count + 1):
                sequence += 1
                result.append(
                    _LatencySpec(
                        sequence,
                        f"ttft-{fixture_id}-{sample_kind}-{sample_index:02d}",
                        "ttft",
                        sample_kind,
                        sample_index,
                        fixture_id,
                        prompt_tokens,
                        512,
                    )
                )
    for sample_kind, count in (("warmup", 2), ("measured", 10)):
        for sample_index in range(1, count + 1):
            sequence += 1
            result.append(
                _LatencySpec(
                    sequence,
                    f"decode64-{sample_kind}-{sample_index:02d}",
                    "decode64",
                    sample_kind,
                    sample_index,
                    "exact-p0032",
                    32,
                    64,
                )
            )
    if sequence != 72:
        fail("internal latency schedule differs")
    return tuple(result)


LATENCY_SCHEDULE = _latency_schedule()


def _completion_binding(completion_id: str) -> tuple[int, str]:
    raw = completion_id.encode("utf-8", errors="strict")
    return len(raw), hashlib.sha256(raw).hexdigest()


def _trace_index(
    traces: Mapping[Any, Any], phase: str
) -> dict[tuple[str, str], tuple[str, object]]:
    result: dict[tuple[str, str], tuple[str, object]] = {}
    for request_id_value, trace in traces.items():
        if type(request_id_value) is not str:
            fail("session trace request ID type differs")
        trace_phase = _attribute(trace, "phase", "session trace")
        if trace_phase != phase:
            continue
        case_id = _text(
            _attribute(trace, "case_id", "session trace"),
            "session trace case ID",
            maximum_bytes=MAX_IDENTIFIER_BYTES,
        )
        key = (phase, case_id)
        if key in result:
            fail(f"{phase} lifecycle case is duplicated")
        result[key] = (request_id_value, trace)
    return result


def _event_time(event: Mapping[str, Any], label: str) -> int:
    return _exact_int(event.get("observed_monotonic_ns"), f"{label} timestamp")


def _validate_trace_common(
    request_id: str,
    trace: object,
    spec: _LatencySpec,
) -> tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    completion_id = _text(
        _attribute(trace, "completion_id", "latency trace"),
        "latency completion ID",
        maximum_bytes=MAX_IDENTIFIER_BYTES,
    )
    events_value = _attribute(trace, "events", "latency trace")
    if type(events_value) is not list or not events_value:
        fail("latency trace events differ")
    events = cast(list[dict[str, Any]], events_value)
    if any(type(event) is not dict for event in events):
        fail("latency trace contains a non-object event")
    times = [_event_time(event, "latency lifecycle") for event in events]
    if any(current < prior for prior, current in zip(times, times[1:])):
        fail("latency lifecycle timestamps regress")
    names = [event.get("event") for event in events]
    progress_values: list[int] = []
    processed = 0
    while processed < spec.prompt_tokens:
        processed = min(processed + 128, spec.prompt_tokens)
        progress_values.append(processed)
    expected_names = (
        ["request_admitted", "request_started"]
        + ["request_progress"] * len(progress_values)
        + ["request_first_token"]
    )
    if spec.workload == "ttft":
        expected_names.append("request_cancel_requested")
    expected_names.append("request_released")
    if names != expected_names:
        fail("latency lifecycle event schedule differs")
    for event in events:
        if (
            event.get("request_id") != request_id
            or event.get("completion_id") != completion_id
        ):
            fail("latency lifecycle request/completion identity differs")
    admitted = events[0]
    started = events[1]
    if (
        admitted.get("stream") is not True
        or _exact_int(
            admitted.get("prompt_tokens"), "latency admitted prompt tokens", minimum=1
        )
        != spec.prompt_tokens
        or _exact_int(
            admitted.get("max_completion_tokens"),
            "latency admitted completion limit",
            minimum=1,
        )
        != spec.max_tokens
        or started.get("stream") is not True
        or started.get("prompt_tokens") != spec.prompt_tokens
    ):
        fail("latency admission/start parameters differ")
    progress = events[2 : 2 + len(progress_values)]
    for event, expected in zip(progress, progress_values, strict=True):
        if (
            event.get("phase") != "prefill"
            or event.get("prompt_tokens") != spec.prompt_tokens
            or event.get("processed_prompt_tokens") != expected
        ):
            fail("latency prefill progress differs")
    first_token = events[2 + len(progress_values)]
    if (
        first_token.get("stream") is not True
        or first_token.get("completion_tokens") != 1
    ):
        fail("latency first-token event differs")
    release = events[-1]
    if (
        release.get("stream") is not True
        or release.get("prompt_tokens") != spec.prompt_tokens
        or release.get("reset_complete") is not True
        or _exact_int(
            release.get("admit_to_release_ns"), "latency admit-to-release", minimum=1
        )
        != _exact_int(release.get("admit_to_start_ns"), "latency admit-to-start")
        + _exact_int(release.get("start_to_release_ns"), "latency start-to-release")
    ):
        fail("latency release common fields differ")
    if _attribute(trace, "terminal", "latency trace") != "request_released":
        fail("latency trace terminal differs")
    return completion_id, events, first_token, release


def _sse_items(result: object, label: str) -> tuple[object, tuple[Any, ...]]:
    sse = _attribute(result, "sse", label)
    if sse is None:
        fail(f"{label} lacks SSE metadata")
    items = _tuple_attribute(sse, "items", f"{label} SSE")
    chunk_count = _exact_int(
        _attribute(sse, "chunk_count", f"{label} SSE"),
        f"{label} SSE chunk count",
        minimum=1,
    )
    if not items or len(items) > 2048:
        fail(f"{label} SSE item population differs")
    first_chunk = _attribute(sse, "first_chunk_monotonic_ns", f"{label} SSE")
    last_chunk = _attribute(sse, "last_chunk_monotonic_ns", f"{label} SSE")
    _exact_int(first_chunk, f"{label} first SSE chunk timestamp")
    _exact_int(last_chunk, f"{label} last SSE chunk timestamp")
    first_chunk_time = _exact_int(first_chunk, f"{label} first SSE chunk timestamp")
    last_chunk_time = _exact_int(last_chunk, f"{label} last SSE chunk timestamp")
    if first_chunk_time > last_chunk_time:
        fail(f"{label} SSE chunk timestamps regress")
    prior_time = -1
    prior_index = -1
    for item in items:
        index = _exact_int(
            _attribute(item, "chunk_index", f"{label} SSE item"),
            f"{label} SSE chunk index",
        )
        observed = _exact_int(
            _attribute(item, "observed_monotonic_ns", f"{label} SSE item"),
            f"{label} SSE item timestamp",
        )
        if (
            index >= chunk_count
            or index < prior_index
            or observed < prior_time
            or observed < first_chunk_time
            or observed > last_chunk_time
        ):
            fail(f"{label} SSE item order differs")
        done = _boolean(_attribute(item, "done", f"{label} SSE item"), f"{label} done")
        content_bytes = _attribute(item, "content_utf8_bytes", f"{label} SSE item")
        content_sha = _attribute(item, "content_sha256", f"{label} SSE item")
        finish = _attribute(item, "finish_reason", f"{label} SSE item")
        usage_present = _boolean(
            _attribute(item, "usage_present", f"{label} SSE item"),
            f"{label} usage-present",
        )
        usage_is_object = _attribute(item, "usage_is_object", f"{label} SSE item")
        completion_tokens = _attribute(item, "completion_tokens", f"{label} SSE item")
        if done:
            if (
                content_bytes is not None
                or content_sha is not None
                or finish is not None
                or usage_present
                or usage_is_object is not None
                or completion_tokens is not None
            ):
                fail(f"{label} [DONE] metadata differs")
        else:
            if content_bytes is None:
                if content_sha is not None:
                    fail(f"{label} content hash lacks content bytes")
            else:
                _exact_int(content_bytes, f"{label} content bytes", minimum=1)
                _sha256(content_sha, f"{label} content SHA-256")
            if finish is not None:
                _text(finish, f"{label} finish reason", maximum_bytes=64)
            if not usage_present:
                if usage_is_object is not None or completion_tokens is not None:
                    fail(f"{label} absent usage retains metadata")
            elif usage_is_object is True:
                _exact_int(completion_tokens, f"{label} usage completion tokens")
            elif usage_is_object is not False or completion_tokens is not None:
                fail(f"{label} non-object usage metadata differs")
        prior_index = index
        prior_time = observed
    return sse, items


def _validate_sse_completion_ids(
    items: Sequence[Any], completion_id: str, label: str
) -> None:
    expected_bytes, expected_sha = _completion_binding(completion_id)
    for item in items:
        done = _boolean(_attribute(item, "done", f"{label} SSE item"), f"{label} done")
        observed_bytes = _attribute(
            item, "completion_id_utf8_bytes", f"{label} SSE item"
        )
        observed_sha = _attribute(item, "completion_id_sha256", f"{label} SSE item")
        if done:
            if observed_bytes is not None or observed_sha is not None:
                fail(f"{label} [DONE] retains a completion ID")
        elif observed_bytes != expected_bytes or observed_sha != expected_sha:
            fail(f"{label} SSE completion ID differs from lifecycle")


def _validate_latency_http_common(result: object, spec: _LatencySpec) -> None:
    expected = {
        "phase": "latency",
        "case_id": spec.case_id,
        "request_index": spec.sequence,
        "request_key": spec.case_id,
        "method": "POST",
        "target": "/v1/chat/completions",
        "status": 200,
    }
    for name, value in expected.items():
        if _attribute(result, name, "latency HTTP result") != value:
            fail("latency HTTP schedule or identity differs")
    connect = _exact_int(
        _attribute(result, "connect_completed_monotonic_ns", "latency HTTP result"),
        "latency HTTP connect",
    )
    write = _exact_int(
        _attribute(result, "write_started_monotonic_ns", "latency HTTP result"),
        "latency HTTP write",
    )
    sent = _exact_int(
        _attribute(result, "last_body_byte_sent_monotonic_ns", "latency HTTP result"),
        "latency HTTP final send",
    )
    response_started = _exact_int(
        _attribute(result, "response_started_monotonic_ns", "latency HTTP result"),
        "latency HTTP response start",
    )
    response_end = _exact_int(
        _attribute(result, "response_end_monotonic_ns", "latency HTTP result"),
        "latency HTTP response end",
    )
    if not connect <= write <= sent <= response_started <= response_end:
        fail("latency HTTP timestamps differ")
    request_body_bytes = _exact_int(
        _attribute(result, "request_body_bytes", "latency HTTP result"),
        "latency HTTP request bytes",
        minimum=1,
    )
    request_body_sha256 = _sha256(
        _attribute(result, "request_body_sha256", "latency HTTP result"),
        "latency HTTP request SHA-256",
    )
    expected_body = LATENCY_REQUEST_BODY_IDENTITIES.get(
        (spec.fixture_id, spec.max_tokens)
    )
    if (
        expected_body is None
        or (request_body_bytes, request_body_sha256) != expected_body
    ):
        fail("latency HTTP request body differs from its tracked fixture")
    _exact_int(
        _attribute(result, "response_body_bytes", "latency HTTP result"),
        "latency HTTP response bytes",
        minimum=1,
    )
    _sha256(
        _attribute(result, "response_body_sha256", "latency HTTP result"),
        "latency HTTP response SHA-256",
    )


def _latency_sample(
    result: object,
    request_id: str,
    trace: object,
    spec: _LatencySpec,
) -> dict[str, Any]:
    _validate_latency_http_common(result, spec)
    completion_id, events, _first_token, release = _validate_trace_common(
        request_id, trace, spec
    )
    _sse, items = _sse_items(result, "latency HTTP result")
    _validate_sse_completion_ids(items, completion_id, "latency HTTP result")
    sent = cast(int, _attribute(result, "last_body_byte_sent_monotonic_ns", "HTTP"))
    response_end = cast(int, _attribute(result, "response_end_monotonic_ns", "HTTP"))
    response_started = cast(
        int, _attribute(result, "response_started_monotonic_ns", "HTTP")
    )
    if (
        cast(int, _attribute(_sse, "first_chunk_monotonic_ns", "latency SSE"))
        < response_started
        or cast(int, _attribute(_sse, "last_chunk_monotonic_ns", "latency SSE"))
        > response_end
        or _event_time(events[0], "latency admission") < sent
    ):
        fail("latency HTTP/SSE/lifecycle boundary differs")
    content_items = [
        item
        for item in items
        if type(_attribute(item, "content_utf8_bytes", "latency SSE item")) is int
        and cast(int, _attribute(item, "content_utf8_bytes", "latency SSE item")) > 0
    ]
    common = {
        "sequence": spec.sequence,
        "case_id": spec.case_id,
        "sample_kind": spec.sample_kind,
        "sample_index": spec.sample_index,
        "fixture_id": spec.fixture_id,
        "prompt_tokens": spec.prompt_tokens,
    }
    release_time = _event_time(release, "latency release")
    if spec.workload == "ttft":
        if _attribute(result, "outcome", "TTFT HTTP result") != "client_closed":
            fail("TTFT HTTP outcome differs")
        if not content_items:
            fail("TTFT response lacks non-empty content")
        chunk_count = cast(int, _attribute(_sse, "chunk_count", "TTFT SSE"))
        if (
            any(
                _attribute(item, "done", "TTFT SSE item") is not False for item in items
            )
            or any(
                _attribute(item, "finish_reason", "TTFT SSE item") is not None
                for item in items
            )
            or any(
                _attribute(item, "usage_present", "TTFT SSE item") is not False
                for item in items
            )
            or any(
                _attribute(item, "chunk_index", "TTFT content") != chunk_count - 1
                for item in content_items
            )
        ):
            fail("TTFT close/content/usage boundary differs")
        first_content = _exact_int(
            _attribute(content_items[0], "observed_monotonic_ns", "TTFT content"),
            "TTFT first content timestamp",
        )
        if not sent < first_content <= response_end:
            fail("TTFT timing order differs")
        cancel = events[-2]
        cancel_time = _event_time(cancel, "TTFT cancellation")
        if (
            cancel.get("reason") != "client_disconnect"
            or release.get("outcome") != "cancelled"
            or release.get("cancel_reason") != "client_disconnect"
            or release.get("reset_complete") is not True
            or _exact_int(
                release.get("completion_tokens"), "TTFT release tokens", minimum=1
            )
            < 1
            or not first_content <= cancel_time <= release_time
            or release_time - cancel_time > RELEASE_MAX_NS
            or _exact_int(cancel.get("admit_to_cancel_ns"), "TTFT admit-to-cancel")
            > _exact_int(release.get("admit_to_release_ns"), "TTFT admit-to-release")
            or _exact_int(release.get("admit_to_release_ns"), "TTFT admit-to-release")
            - _exact_int(cancel.get("admit_to_cancel_ns"), "TTFT admit-to-cancel")
            > RELEASE_MAX_NS
        ):
            fail("TTFT cancellation release differs")
        return {
            **common,
            "ttft_ns": first_content - sent,
            "content_object_count": len(content_items),
            "release_outcome": "cancelled",
            "release_completion_tokens": cast(int, release["completion_tokens"]),
        }

    if _attribute(result, "outcome", "decode HTTP result") != "eof":
        fail("decode HTTP outcome differs")
    if len(content_items) != 64:
        fail("decode content population differs from 64")
    done_positions = [
        index
        for index, item in enumerate(items)
        if _attribute(item, "done", "decode SSE item") is True
    ]
    finish_reasons = [
        _attribute(item, "finish_reason", "decode SSE item")
        for item in items
        if _attribute(item, "finish_reason", "decode SSE item") is not None
    ]
    content_positions = [
        index
        for index, item in enumerate(items)
        if type(_attribute(item, "content_utf8_bytes", "decode SSE item")) is int
        and cast(int, _attribute(item, "content_utf8_bytes", "decode SSE item")) > 0
    ]
    finish_positions = [
        index
        for index, item in enumerate(items)
        if _attribute(item, "finish_reason", "decode SSE item") is not None
    ]
    usage_positions = [
        index
        for index, item in enumerate(items)
        if _attribute(item, "usage_present", "decode SSE item") is True
    ]
    if (
        done_positions != [len(items) - 1]
        or finish_reasons != ["length"]
        or len(finish_positions) != 1
        or len(usage_positions) != 1
        or not max(content_positions)
        < finish_positions[0]
        < usage_positions[0]
        < done_positions[0]
        or _attribute(items[usage_positions[0]], "usage_is_object", "decode usage")
        is not True
        or _attribute(items[usage_positions[0]], "completion_tokens", "decode usage")
        != 64
        or _attribute(items[-1], "chunk_index", "decode [DONE]")
        != _attribute(_sse, "chunk_count", "decode SSE") - 1
    ):
        fail("decode finish/usage/[DONE] contract differs")
    timestamps = [
        _exact_int(
            _attribute(item, "observed_monotonic_ns", "decode content"),
            "decode content timestamp",
        )
        for item in content_items
    ]
    if (
        timestamps[0] < sent
        or timestamps[-1] > response_end
        or any(current < prior for prior, current in zip(timestamps, timestamps[1:]))
    ):
        fail("decode content timestamps differ")
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        fail("decode elapsed timing differs")
    intervals = [current - prior for prior, current in zip(timestamps, timestamps[1:])]
    if (
        len(intervals) != 63
        or any(interval <= 0 for interval in intervals)
        or release.get("outcome") != "length"
        or release.get("cancel_reason") is not None
        or release.get("completion_tokens") != 64
        or any(event.get("event") == "request_cancel_requested" for event in events)
    ):
        fail("decode release or interval contract differs")
    return {
        **common,
        "decode_elapsed_ns": elapsed,
        "decode_intervals_ns": intervals,
        "decode_tokens_per_second": _fraction_json(Fraction(63_000_000_000, elapsed)),
        "release_outcome": "length",
        "release_completion_tokens": 64,
    }


def _latency_metrics(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) != 72:
        fail("latency metric sample population differs")
    ttft: dict[str, Any] = {}
    for fixture_id, _prompt_tokens in FIXTURE_ORDER:
        values = [
            _exact_int(sample.get("ttft_ns"), "measured TTFT", minimum=1)
            for sample in samples
            if sample["fixture_id"] == fixture_id
            and sample["sample_kind"] == "measured"
            and "ttft_ns" in sample
        ]
        if len(values) != 10:
            fail("measured TTFT population differs")
        p50 = _linear_percentile(values, Fraction(1, 2))
        p95 = _linear_percentile(values, Fraction(19, 20))
        p50_limit, p95_limit = TTFT_LIMITS_NS[fixture_id]
        if p50 > p50_limit or p95 > p95_limit:
            fail("measured TTFT exceeds its frozen threshold")
        ttft[fixture_id] = {
            "count": 10,
            "p50_ns": _fraction_json(p50),
            "p95_ns": _fraction_json(p95),
            "p50_maximum_ns": p50_limit,
            "p95_maximum_ns": p95_limit,
        }
    decode = [
        sample
        for sample in samples
        if "decode_elapsed_ns" in sample and sample["sample_kind"] == "measured"
    ]
    if len(decode) != 10:
        fail("measured decode population differs")
    throughputs = [
        Fraction(63_000_000_000, cast(int, sample["decode_elapsed_ns"]))
        for sample in decode
    ]
    throughput_p50 = _median(throughputs)
    intervals = [
        interval
        for sample in decode
        for interval in cast(list[int], sample["decode_intervals_ns"])
    ]
    if len(intervals) != 630:
        fail("measured decode interval population differs")
    interval_p95 = _linear_percentile(intervals, Fraction(19, 20))
    if throughput_p50 < DECODE_MIN_P50_TOKENS_PER_SECOND:
        fail("measured decode throughput is below its frozen threshold")
    if interval_p95 > DECODE_MAX_P95_INTERVAL_NS:
        fail("measured decode interval exceeds its frozen threshold")
    return {
        "ttft": ttft,
        "decode64": {
            "request_count": 10,
            "interval_count": 630,
            "p50_tokens_per_second": _fraction_json(throughput_p50),
            "minimum_p50_tokens_per_second": DECODE_MIN_P50_TOKENS_PER_SECOND,
            "p95_inter_content_ns": _fraction_json(interval_p95),
            "maximum_p95_inter_content_ns": DECODE_MAX_P95_INTERVAL_NS,
        },
    }


def reconstruct_latency_results(session: object) -> dict[str, Any]:
    """Reconstruct the exact final latency view from bounded session projections."""

    results = [
        result
        for result in _tuple_attribute(session, "http_results", "session")
        if _attribute(result, "phase", "HTTP result") == "latency"
    ]
    if len(results) != 72:
        fail("latency HTTP result count differs from 72")
    traces = _mapping_attribute(session, "traces", "session")
    indexed = _trace_index(traces, "latency")
    if len(indexed) != 72:
        fail("latency lifecycle trace count differs from 72")
    samples: list[dict[str, Any]] = []
    prior_response_end = -1
    prior_release = -1
    for result, spec in zip(results, LATENCY_SCHEDULE, strict=True):
        entry = indexed.get(("latency", spec.case_id))
        if entry is None:
            fail("latency lifecycle case is absent")
        request_id, trace = entry
        connect = _exact_int(
            _attribute(result, "connect_completed_monotonic_ns", "latency HTTP"),
            "latency HTTP connect",
        )
        if connect < prior_response_end or connect < prior_release:
            fail("latency requests overlap or regress")
        sample = _latency_sample(result, request_id, trace, spec)
        samples.append(sample)
        prior_response_end = cast(
            int, _attribute(result, "response_end_monotonic_ns", "latency HTTP")
        )
        events = cast(
            list[dict[str, Any]], _attribute(trace, "events", "latency trace")
        )
        prior_release = _event_time(events[-1], "latency release")
    metrics = _latency_metrics(samples)
    return {
        "schema_version": LATENCY_RESULTS_SCHEMA,
        "request_count": 72,
        "prefill_ttft": {
            "request_count": 60,
            "metrics": metrics["ttft"],
            "samples": samples[:60],
        },
        "decode64": {
            "request_count": 12,
            "metrics": metrics["decode64"],
            "samples": samples[60:],
        },
    }


@dataclasses.dataclass(frozen=True)
class _Identity:
    device: int
    inode: int
    mode: int
    links: int
    uid: int
    gid: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _Identity:
        return cls(
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_uid,
            value.st_gid,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for independent resource validation")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _file_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        fail(
            "O_NOFOLLOW and O_NONBLOCK are required for independent resource validation"
        )
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


class _RootSnapshot:
    def __init__(self, root: Path):
        if not isinstance(root, os.PathLike):
            fail("resource root path type differs")
        self.path = Path(root)
        if not self.path.is_absolute() or ".." in self.path.parts:
            fail("resource root path is not absolute and normalized")
        self.fd = -1
        self.identity: _Identity | None = None

    def __enter__(self) -> _RootSnapshot:
        try:
            before = _Identity.from_stat(os.stat(self.path, follow_symlinks=False))
            self.fd = os.open(self.path, _directory_flags())
            opened = _Identity.from_stat(os.fstat(self.fd))
            if before != opened:
                fail("resource root changed while opening")
            if (
                not stat.S_ISDIR(opened.mode)
                or stat.S_IMODE(opened.mode) != 0o700
                or opened.uid != os.geteuid()
                or opened.gid != os.getegid()
            ):
                fail("resource root mode or owner differs")
            try:
                descriptor_path = Path(os.readlink(f"/proc/self/fd/{self.fd}"))
            except OSError as error:
                raise IndependentMetricsError(
                    "failed to resolve resource root descriptor"
                ) from error
            if descriptor_path != self.path:
                fail("resource root path contains a symbolic link")
            self.identity = opened
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self.fd >= 0:
            descriptor = self.fd
            self.fd = -1
            try:
                os.close(descriptor)
            except OSError as error:
                raise IndependentMetricsError(
                    "failed to close resource root descriptor"
                ) from error

    def entry_identity(self, name: str) -> _Identity:
        try:
            return _Identity.from_stat(
                os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            )
        except OSError as error:
            raise IndependentMetricsError(
                f"resource file is unavailable: {name}"
            ) from error

    def open_file(self, name: str, maximum: int) -> tuple[int, _Identity]:
        before = self.entry_identity(name)
        if (
            not stat.S_ISREG(before.mode)
            or stat.S_IMODE(before.mode) != 0o600
            or before.links != 1
            or self.identity is None
            or before.uid != self.identity.uid
            or before.gid != self.identity.gid
            or before.size < 1
            or before.size > maximum
        ):
            fail(f"resource file identity or size differs: {name}")
        try:
            descriptor = os.open(name, _file_flags(), dir_fd=self.fd)
        except OSError as error:
            raise IndependentMetricsError(
                f"failed to open resource file: {name}"
            ) from error
        opened = _Identity.from_stat(os.fstat(descriptor))
        if not stat.S_ISREG(opened.mode) or opened != before:
            os.close(descriptor)
            fail(f"resource file changed while opening: {name}")
        return descriptor, before

    def verify_file(self, name: str, descriptor: int, before: _Identity) -> None:
        if (
            _Identity.from_stat(os.fstat(descriptor)) != before
            or self.entry_identity(name) != before
        ):
            fail(f"resource file changed while streaming: {name}")

    def verify_root(self) -> None:
        if self.identity is None:
            fail("resource root snapshot is incomplete")
        if (
            _Identity.from_stat(os.fstat(self.fd)) != self.identity
            or _Identity.from_stat(os.stat(self.path, follow_symlinks=False))
            != self.identity
        ):
            fail("resource root changed during validation")


def _reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("resource JSON contains a duplicate key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    fail("resource JSON contains a non-finite number")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        fail("resource JSON contains a non-finite number")
    return parsed


def _reject_passed(value: Any, label: str) -> None:
    pending = [value]
    while pending:
        current = pending.pop()
        if type(current) is dict:
            if "passed" in current:
                fail(f"{label} contains forbidden key passed")
            pending.extend(current.values())
        elif type(current) is list:
            pending.extend(current)


def _resource_document(raw: bytes, line_number: int) -> dict[str, Any]:
    label = f"resource line {line_number}"
    if (
        not raw
        or len(raw) > MAX_RESOURCE_LINE_BYTES
        or not raw.endswith(b"\n")
        or raw.endswith(b"\r\n")
        or raw.count(b"\n") != 1
    ):
        fail(f"{label} framing differs")
    try:
        value = json.loads(
            raw[:-1].decode("ascii", errors="strict"),
            object_pairs_hook=_reject_pairs,
            parse_float=_finite_float,
            parse_constant=_reject_constant,
        )
        canonical = (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except IndependentMetricsError:
        raise
    except (UnicodeError, ValueError, TypeError, RecursionError) as error:
        raise IndependentMetricsError(
            f"{label} is not strict canonical JSON"
        ) from error
    if type(value) is not dict or canonical != raw:
        fail(f"{label} is not a canonical JSON object")
    _reject_passed(value, label)
    return cast(dict[str, Any], value)


def _resource_lines(root: _RootSnapshot) -> Iterator[tuple[int, dict[str, Any]]]:
    descriptor, before = root.open_file("soak-resources.raw.jsonl", MAX_RESOURCE_BYTES)
    handle: BinaryIO | None = None
    try:
        handle = os.fdopen(descriptor, "rb", buffering=0)
        descriptor = -1
        line_number = 0
        total = 0
        while True:
            raw = handle.readline(MAX_RESOURCE_LINE_BYTES + 1)
            if not raw:
                break
            line_number += 1
            total += len(raw)
            if len(raw) > MAX_RESOURCE_LINE_BYTES or total > MAX_RESOURCE_BYTES:
                fail("resource raw stream exceeds its bound")
            yield line_number, _resource_document(raw, line_number)
        root.verify_file("soak-resources.raw.jsonl", handle.fileno(), before)
        if total != before.size:
            fail("resource raw streamed byte count differs")
    except IndependentMetricsError:
        raise
    except OSError as error:
        raise IndependentMetricsError("failed to stream resource raw") from error
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError as error:
                raise IndependentMetricsError("failed to close resource raw") from error
        elif descriptor >= 0:
            os.close(descriptor)


def _snapshot_metric(root: _RootSnapshot, name: str) -> tuple[int, str]:
    descriptor, before = root.open_file(name, MAX_GPU_METRIC_BYTES)
    digest = hashlib.sha256()
    payload = bytearray()
    total = 0
    try:
        while True:
            raw = os.read(descriptor, HASH_CHUNK_BYTES)
            if not raw:
                break
            total += len(raw)
            if total > MAX_GPU_METRIC_BYTES:
                fail(f"GPU metric exceeds its bound: {name}")
            digest.update(raw)
            payload.extend(raw)
        root.verify_file(name, descriptor, before)
    except IndependentMetricsError:
        raise
    except OSError as error:
        raise IndependentMetricsError(f"failed to stream GPU metric: {name}") from error
    finally:
        os.close(descriptor)
    if total != before.size:
        fail(f"GPU metric byte count differs: {name}")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_pairs,
            parse_float=_finite_float,
            parse_constant=_reject_constant,
        )
    except IndependentMetricsError:
        raise
    except (UnicodeError, ValueError, RecursionError) as error:
        raise IndependentMetricsError(
            f"GPU metric is not strict JSON: {name}"
        ) from error
    if type(value) not in {dict, list}:
        fail(f"GPU metric JSON root differs: {name}")
    _reject_passed(value, name)
    return total, digest.hexdigest()


def _resource_header(record: Any) -> None:
    header = _object(record, RESOURCE_HEADER_FIELDS, "resource header")
    if (
        header["schema_version"] != RESOURCE_INPUT_SCHEMA
        or header["record_type"] != "header"
        or header["service_unit"] != "ullm-openai.service"
        or header["schedule"] != RESOURCE_SCHEDULE
        or header["commands"] != RESOURCE_COMMANDS
    ):
        fail("resource header identity differs")
    tools = _object(
        header["tools"],
        {
            "systemd_major",
            "systemd_version_line",
            "amd_smi_tool",
            "amd_smi_library",
            "rocm",
            "amd_smi_version_output",
        },
        "resource tools",
    )
    systemd_line = _text(tools["systemd_version_line"], "systemd version")
    version_output = _text(tools["amd_smi_version_output"], "AMD SMI version")
    if (
        tools["systemd_major"] != 255
        or not systemd_line.startswith("systemd 255 ")
        or any(tools[name] != value for name, value in RESOURCE_TOOL_VERSIONS.items())
        or any(value not in version_output for value in RESOURCE_TOOL_VERSIONS.values())
    ):
        fail("resource tool identity differs")
    probes = _object(
        header["probes"],
        {
            "cgroup_fs_type",
            "kfd_proc_present",
            "gpu_index",
            "gpu_bdf",
            "gpu_uuid",
            "kfd_gpu_id",
        },
        "resource probes",
    )
    if probes != {
        "cgroup_fs_type": "cgroup2fs",
        "kfd_proc_present": True,
        "gpu_index": GPU_INDEX,
        "gpu_bdf": GPU_BDF,
        "gpu_uuid": GPU_UUID,
        "kfd_gpu_id": KFD_GPU_ID,
    }:
        fail("resource physical probe identity differs")


def _ascending_pids(value: Any, label: str) -> list[int]:
    if type(value) is not list:
        fail(f"{label} is not an array")
    result = [_exact_int(item, f"{label} PID", minimum=1) for item in value]
    if result != sorted(set(result)):
        fail(f"{label} is not ascending and unique")
    return result


def _process(value: Any, label: str) -> dict[str, Any]:
    process = _object(value, PROCESS_FIELDS, label)
    pid = _exact_int(process["pid"], f"{label}.pid", minimum=1)
    _exact_int(process["ppid"], f"{label}.ppid", minimum=1)
    exe = _text(process["exe"], f"{label}.exe")
    pure = PurePosixPath(exe)
    if not pure.is_absolute() or ".." in pure.parts:
        fail(f"{label}.exe is not an absolute normalized path")
    before = _exact_int(
        process["starttime_ticks_before"], f"{label}.starttime", minimum=1
    )
    if process["starttime_ticks_after"] != before:
        fail(f"{label}.starttime changed")
    rss_kb = _exact_int(process["vmrss_kb"], f"{label}.rss kB")
    if process["vmrss_bytes"] != rss_kb * 1024:
        fail(f"{label}.rss conversion differs")
    _exact_int(process["threads"], f"{label}.threads", minimum=1)
    _exact_int(process["fd_count"], f"{label}.fds")
    _ascending_pids(process["children"], f"{label}.children")
    if process["pid"] != pid:
        fail(f"{label}.pid differs")
    return process


def _resource_identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    gateway = cast(Mapping[str, Any], record["gateway"])
    worker = cast(Mapping[str, Any], record["worker"])
    systemd_value = cast(Mapping[str, Any], record["systemd"])
    return (
        systemd_value["control_group_before"],
        gateway["pid"],
        gateway["ppid"],
        gateway["exe"],
        gateway["starttime_ticks_before"],
        worker["pid"],
        worker["ppid"],
        worker["exe"],
        worker["starttime_ticks_before"],
    )


def _resource_sample(
    record: Any, expected: tuple[str, str, int | None, int]
) -> dict[str, Any]:
    segment, phase, request_index, sample_index = expected
    item = _object(record, RESOURCE_SAMPLE_FIELDS, "resource sample")
    if (
        item["schema_version"] != RESOURCE_INPUT_SCHEMA
        or item["record_type"] != "resource_sample"
        or item["segment"] != segment
        or item["phase"] != phase
        or item["request_index"] != request_index
        or item["sample_index"] != sample_index
    ):
        fail("resource sample schedule differs")
    settle = _exact_int(item["idle_settle_started_monotonic_ns"], "resource settle")
    sampled = _exact_int(item["sample_monotonic_ns"], "resource sample time")
    if phase == "baseline":
        if any(
            item[name] is not None
            for name in (
                "request_id",
                "release_outcome",
                "release_observed_monotonic_ns",
                "reset_complete",
            )
        ):
            fail("resource baseline contains release data")
    else:
        _text(
            item["request_id"],
            "resource request ID",
            maximum_bytes=MAX_IDENTIFIER_BYTES,
        )
        released = _exact_int(item["release_observed_monotonic_ns"], "resource release")
        if (
            item["release_outcome"] != "length"
            or item["reset_complete"] is not True
            or settle < released
        ):
            fail("resource release outcome or timing differs")
    systemd_value = _object(item["systemd"], SYSTEMD_FIELDS, "resource systemd")
    control_group = _text(systemd_value["control_group_before"], "control group")
    if (
        systemd_value["control_group_after"] != control_group
        or not PurePosixPath(control_group).is_absolute()
        or ".." in PurePosixPath(control_group).parts
    ):
        fail("resource control group differs")
    main_pid = _exact_int(
        systemd_value["main_pid_before"], "resource MainPID", minimum=1
    )
    if systemd_value["main_pid_after"] != main_pid:
        fail("resource MainPID changed during a sample")
    host = _object(item["host"], {"memory_current_bytes"}, "resource host")
    _exact_int(host["memory_current_bytes"], "resource MemoryCurrent")
    gateway = _process(item["gateway"], "resource gateway")
    worker = _process(item["worker"], "resource worker")
    if (
        gateway["pid"] != main_pid
        or worker["ppid"] != gateway["pid"]
        or worker["pid"] not in gateway["children"]
        or PurePosixPath(worker["exe"]).name != "ullm-sq8-worker"
    ):
        fail("resource gateway/worker relationship differs")
    gpu = _object(item["gpu"], GPU_FIELDS, "resource GPU")
    if (
        gpu["index"] != GPU_INDEX
        or gpu["bdf"] != GPU_BDF
        or gpu["uuid"] != GPU_UUID
        or gpu["kfd_gpu_id"] != KFD_GPU_ID
        or gpu["process_record_count"] != 1
        or gpu["worker_pid"] != worker["pid"]
    ):
        fail("resource GPU identity differs")
    memory = _object(gpu["mem_usage"], {"value", "unit"}, "resource VRAM")
    vram = _exact_int(memory["value"], "resource VRAM", minimum=1)
    if memory["unit"] != "B" or gpu["kfd_vram_bytes"] != vram:
        fail("resource VRAM sources differ")
    if _ascending_pids(gpu["unrelated_process_pids"], "unrelated GPU PIDs"):
        fail("resource GPU contains an unrelated process")
    if sample_index == 0 and sampled - settle < 5_000_000_000:
        fail("resource idle settle is too short")
    return item


def _gpu_metric(
    root: _RootSnapshot,
    record: Any,
    expected: tuple[str, str],
) -> int:
    segment, boundary = expected
    item = _object(record, GPU_METRIC_FIELDS, "resource GPU metric")
    expected_file = f"amd-smi-metric-{segment}-{boundary}.json"
    if (
        item["schema_version"] != RESOURCE_INPUT_SCHEMA
        or item["record_type"] != "gpu_metric"
        or item["segment"] != segment
        or item["boundary"] != boundary
        or item["gpu_index"] != GPU_INDEX
        or item["raw_output_file"] != expected_file
    ):
        fail("resource GPU metric schedule differs")
    captured = _exact_int(item["captured_monotonic_ns"], "GPU metric timestamp")
    expected_sha = _sha256(item["raw_output_sha256"], "GPU metric SHA-256")
    _bytes, observed_sha = _snapshot_metric(root, expected_file)
    if observed_sha != expected_sha:
        fail("GPU metric raw output SHA-256 differs")
    return captured


def _expected_resource_records() -> Iterator[tuple[str, tuple[Any, ...]]]:
    yield "metric", ("normal", "before")
    for sample_index in range(5):
        yield "sample", ("normal", "baseline", None, sample_index)
    for request_index in range(1, 101):
        for sample_index in range(5):
            yield "sample", ("normal", "post_release", request_index, sample_index)
    yield "metric", ("normal", "after")
    yield "metric", ("restart", "before")
    for sample_index in range(5):
        yield "sample", ("restart", "baseline", None, sample_index)
    for request_index in range(1, 21):
        for sample_index in range(5):
            yield "sample", ("restart", "post_release", request_index, sample_index)
    yield "metric", ("restart", "after")


@dataclasses.dataclass(frozen=True)
class _ResourcePoint:
    segment: str
    phase: str
    request_index: int | None
    request_id: str | None
    release_outcome: str | None
    release_observed_monotonic_ns: int | None
    settle_started_ns: int
    sample_times: tuple[int, ...]
    identity: tuple[Any, ...]
    metrics: dict[str, Fraction]
    stable_counts: dict[str, Fraction]


def _resource_point(samples: Sequence[dict[str, Any]]) -> _ResourcePoint:
    if len(samples) != 5:
        fail("resource point sample population differs")
    first = samples[0]
    stable = (
        "segment",
        "phase",
        "request_index",
        "request_id",
        "release_outcome",
        "release_observed_monotonic_ns",
        "reset_complete",
        "idle_settle_started_monotonic_ns",
    )
    identity = _resource_identity(first)
    times: list[int] = []
    for index, sample in enumerate(samples):
        if sample["sample_index"] != index:
            fail("resource point sample index differs")
        if any(sample[name] != first[name] for name in stable):
            fail("resource point stable field changed")
        if _resource_identity(sample) != identity:
            fail("resource point process identity changed")
        current = cast(int, sample["sample_monotonic_ns"])
        if times and current - times[-1] < 1_000_000_000:
            fail("resource point sample interval is too short")
        times.append(current)
    return _ResourcePoint(
        cast(str, first["segment"]),
        cast(str, first["phase"]),
        cast(int | None, first["request_index"]),
        cast(str | None, first["request_id"]),
        cast(str | None, first["release_outcome"]),
        cast(int | None, first["release_observed_monotonic_ns"]),
        cast(int, first["idle_settle_started_monotonic_ns"]),
        tuple(times),
        identity,
        {
            "memory_current_bytes": _median(
                cast(int, sample["host"]["memory_current_bytes"]) for sample in samples
            ),
            "process_vram_bytes": _median(
                cast(int, sample["gpu"]["mem_usage"]["value"]) for sample in samples
            ),
            "gateway_rss_bytes": _median(
                cast(int, sample["gateway"]["vmrss_bytes"]) for sample in samples
            ),
            "worker_rss_bytes": _median(
                cast(int, sample["worker"]["vmrss_bytes"]) for sample in samples
            ),
        },
        {
            "gateway_threads": _median(
                cast(int, sample["gateway"]["threads"]) for sample in samples
            ),
            "gateway_fds": _median(
                cast(int, sample["gateway"]["fd_count"]) for sample in samples
            ),
            "gateway_children": _median(
                len(cast(list[Any], sample["gateway"]["children"]))
                for sample in samples
            ),
            "worker_threads": _median(
                cast(int, sample["worker"]["threads"]) for sample in samples
            ),
            "worker_fds": _median(
                cast(int, sample["worker"]["fd_count"]) for sample in samples
            ),
            "worker_children": _median(
                len(cast(list[Any], sample["worker"]["children"])) for sample in samples
            ),
        },
    )


def _resource_http_schedule() -> tuple[tuple[str, str, int, bool], ...]:
    result: list[tuple[str, str, int, bool]] = []
    for index in range(1, 11):
        result.append(("resource_normal", f"normal-warmup-{index:02d}", index, True))
    negatives = {
        25: "negative-after-025-context_overflow_1",
        50: "negative-after-050-malformed_json",
        75: "negative-after-075-context_overflow_2",
    }
    for index in range(1, 101):
        result.append(("resource_normal", f"normal-measured-{index:03d}", index, True))
        if index in negatives:
            result.append(("resource_normal", negatives[index], index, False))
    for index in range(1, 11):
        result.append(("resource_restart", f"restart-warmup-{index:02d}", index, True))
    for index in range(1, 21):
        result.append(
            ("resource_restart", f"restart-measured-{index:03d}", index, True)
        )
    return tuple(result)


RESOURCE_HTTP_SCHEDULE = _resource_http_schedule()


def _resource_trace(
    request_id: str, trace: object, phase: str, case_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        _attribute(trace, "phase", "resource trace") != phase
        or _attribute(trace, "case_id", "resource trace") != case_id
        or _attribute(trace, "terminal", "resource trace") != "request_released"
    ):
        fail("resource lifecycle trace identity differs")
    completion_id = _text(
        _attribute(trace, "completion_id", "resource trace"),
        "resource completion ID",
        maximum_bytes=MAX_IDENTIFIER_BYTES,
    )
    events_value = _attribute(trace, "events", "resource trace")
    if type(events_value) is not list or not events_value:
        fail("resource lifecycle events differ")
    events = cast(list[dict[str, Any]], events_value)
    times = [_event_time(event, "resource lifecycle") for event in events]
    if any(current < prior for prior, current in zip(times, times[1:])):
        fail("resource lifecycle timestamps regress")
    if any(
        type(event) is not dict
        or event.get("request_id") != request_id
        or event.get("completion_id") != completion_id
        for event in events
    ):
        fail("resource lifecycle correlation differs")
    admitted = events[0]
    release = events[-1]
    names = [event.get("event") for event in events]
    if (
        admitted.get("event") != "request_admitted"
        or admitted.get("stream") is not True
        or admitted.get("max_completion_tokens") != 2
        or "request_started" not in names
        or "request_first_token" not in names
        or "request_cancel_requested" in names
        or release.get("event") != "request_released"
        or release.get("outcome") != "length"
        or release.get("completion_tokens") != 2
        or release.get("cancel_reason") is not None
        or release.get("reset_complete") is not True
    ):
        fail("resource lifecycle outcome differs")
    return admitted, release


def _validate_resource_sse(result: object, trace: object, label: str) -> None:
    completion_id = cast(str, _attribute(trace, "completion_id", "resource trace"))
    sse, items = _sse_items(result, label)
    _validate_sse_completion_ids(items, completion_id, label)
    response_started = _exact_int(
        _attribute(result, "response_started_monotonic_ns", label),
        f"{label} response start",
    )
    response_end = _exact_int(
        _attribute(result, "response_end_monotonic_ns", label),
        f"{label} response end",
    )
    first_chunk = _exact_int(
        _attribute(sse, "first_chunk_monotonic_ns", f"{label} SSE"),
        f"{label} first SSE chunk",
    )
    last_chunk = _exact_int(
        _attribute(sse, "last_chunk_monotonic_ns", f"{label} SSE"),
        f"{label} last SSE chunk",
    )
    if not response_started <= first_chunk <= last_chunk <= response_end:
        fail("resource HTTP/SSE boundary differs")
    content_positions = [
        index
        for index, item in enumerate(items)
        if type(_attribute(item, "content_utf8_bytes", f"{label} item")) is int
        and cast(int, _attribute(item, "content_utf8_bytes", f"{label} item")) > 0
    ]
    done_positions = [
        index
        for index, item in enumerate(items)
        if _attribute(item, "done", f"{label} item") is True
    ]
    usage_positions = [
        index
        for index, item in enumerate(items)
        if _attribute(item, "usage_present", f"{label} item") is True
    ]
    finish_positions = [
        index
        for index, item in enumerate(items)
        if _attribute(item, "finish_reason", f"{label} item") is not None
    ]
    if (
        not content_positions
        or done_positions != [len(items) - 1]
        or len(usage_positions) != 1
        or len(finish_positions) != 1
        or not max(content_positions)
        < finish_positions[0]
        < usage_positions[0]
        < done_positions[0]
        or _attribute(items[usage_positions[0]], "usage_is_object", f"{label} usage")
        is not True
        or _attribute(items[usage_positions[0]], "completion_tokens", f"{label} usage")
        != 2
        or _attribute(items[finish_positions[0]], "finish_reason", f"{label} finish")
        != "length"
        or _attribute(items[-1], "chunk_index", f"{label} [DONE]")
        != _attribute(sse, "chunk_count", f"{label} SSE") - 1
    ):
        fail("resource SSE content/finish/usage differs")


def _session_resource_bindings(
    session: object,
    baselines: Mapping[str, _ResourcePoint],
    points: Mapping[str, Sequence[_ResourcePoint]],
    identities: Mapping[str, tuple[Any, ...]],
    metric_times: Mapping[tuple[str, str], int],
) -> None:
    all_results = _tuple_attribute(session, "http_results", "session")
    results = [
        result
        for result in all_results
        if _attribute(result, "phase", "HTTP result")
        in {"resource_normal", "resource_restart"}
    ]
    if len(results) != len(RESOURCE_HTTP_SCHEDULE):
        fail("resource HTTP result population differs")
    traces = _mapping_attribute(session, "traces", "session")
    trace_maps = {
        phase: _trace_index(traces, phase)
        for phase in ("resource_normal", "resource_restart")
    }
    if sum(len(value) for value in trace_maps.values()) != 140:
        fail("resource lifecycle trace population differs")
    releases_by_phase = _mapping_attribute(session, "releases_by_phase", "session")
    ordered_positive: dict[str, list[tuple[str, object, dict[str, Any], object]]] = {
        "resource_normal": [],
        "resource_restart": [],
    }
    prior_release = -1
    prior_response_end = -1
    for position, (result, expected) in enumerate(
        zip(results, RESOURCE_HTTP_SCHEDULE, strict=True)
    ):
        phase, case_id, request_index, positive = expected
        if any(
            _attribute(result, name, "resource HTTP result") != value
            for name, value in (
                ("phase", phase),
                ("case_id", case_id),
                ("request_index", request_index),
                ("request_key", f"p8f-{case_id}"),
                ("method", "POST"),
                ("target", "/v1/chat/completions"),
            )
        ):
            fail("resource HTTP schedule or identity differs")
        connect = _exact_int(
            _attribute(result, "connect_completed_monotonic_ns", "resource HTTP"),
            "resource HTTP connect",
        )
        sent = _exact_int(
            _attribute(result, "last_body_byte_sent_monotonic_ns", "resource HTTP"),
            "resource HTTP send",
        )
        response_end = _exact_int(
            _attribute(result, "response_end_monotonic_ns", "resource HTTP"),
            "resource HTTP response end",
        )
        if (
            not connect <= sent <= response_end
            or connect < prior_release
            or connect < prior_response_end
        ):
            fail("resource HTTP timing or serialization differs")
        write = _exact_int(
            _attribute(result, "write_started_monotonic_ns", "resource HTTP"),
            "resource HTTP write",
        )
        response_started = _exact_int(
            _attribute(result, "response_started_monotonic_ns", "resource HTTP"),
            "resource HTTP response start",
        )
        if not connect <= write <= sent <= response_started <= response_end:
            fail("resource HTTP detailed timestamp order differs")
        _exact_int(
            _attribute(result, "request_body_bytes", "resource HTTP"),
            "resource HTTP request bytes",
            minimum=1,
        )
        _sha256(
            _attribute(result, "request_body_sha256", "resource HTTP"),
            "resource HTTP request SHA-256",
        )
        _exact_int(
            _attribute(result, "response_body_bytes", "resource HTTP"),
            "resource HTTP response bytes",
            minimum=1,
        )
        _sha256(
            _attribute(result, "response_body_sha256", "resource HTTP"),
            "resource HTTP response SHA-256",
        )
        prior_response_end = response_end
        trace_entry = trace_maps[phase].get((phase, case_id))
        if not positive:
            if (
                trace_entry is not None
                or _attribute(result, "status", "negative resource HTTP") != 400
                or _attribute(result, "outcome", "negative resource HTTP") != "eof"
                or _attribute(result, "sse", "negative resource HTTP") is not None
            ):
                fail("negative resource HTTP/lifecycle outcome differs")
            if position + 1 >= len(results):
                fail("negative resource request lacks recovery")
            following = results[position + 1]
            following_case = cast(
                str, _attribute(following, "case_id", "resource HTTP")
            )
            following_phase = cast(str, _attribute(following, "phase", "resource HTTP"))
            following_trace = trace_maps[following_phase].get(
                (following_phase, following_case)
            )
            if following_trace is None:
                fail("negative resource request lacks recovery lifecycle")
            quiet_end = _event_time(
                cast(
                    list[dict[str, Any]],
                    _attribute(following_trace[1], "events", "trace"),
                )[0],
                "resource recovery admission",
            )
            if any(
                connect
                <= _event_time(
                    cast(list[dict[str, Any]], _attribute(trace, "events", "trace"))[0],
                    "resource admission",
                )
                < quiet_end
                for trace in traces.values()
                if _attribute(trace, "events", "trace")
            ):
                fail("negative resource quiet interval contains an admission")
            continue
        if trace_entry is None:
            fail("positive resource HTTP lacks lifecycle")
        request_id, trace = trace_entry
        admitted, release = _resource_trace(request_id, trace, phase, case_id)
        if (
            _attribute(result, "status", "resource HTTP") != 200
            or _attribute(result, "outcome", "resource HTTP") != "eof"
            or _event_time(admitted, "resource admission") < sent
        ):
            fail("positive resource HTTP/lifecycle timing differs")
        _validate_resource_sse(result, trace, "resource HTTP")
        prior_release = _event_time(release, "resource release")
        ordered_positive[phase].append((request_id, trace, release, result))

    all_admissions = sorted(
        _event_time(
            cast(list[dict[str, Any]], _attribute(trace, "events", "trace"))[0],
            "session admission",
        )
        for trace in traces.values()
        if _attribute(trace, "events", "trace")
        and cast(list[dict[str, Any]], _attribute(trace, "events", "trace"))[0].get(
            "event"
        )
        == "request_admitted"
    )
    for segment, phase, expected_count in (
        ("normal", "resource_normal", 110),
        ("restart", "resource_restart", 30),
    ):
        ordered = ordered_positive[phase]
        if len(ordered) != expected_count:
            fail(f"{phase} positive lifecycle population differs")
        declared = releases_by_phase.get(phase)
        if type(declared) is not list or declared != [entry[2] for entry in ordered]:
            fail(f"{phase} release projection differs from traces")
        baseline = baselines[segment]
        segment_points = list(points[segment])
        if baseline.settle_started_ns < _event_time(ordered[9][2], "warmup release"):
            fail(f"{phase} baseline begins before warmup release")
        first_measured_events = _attribute(
            ordered[10][1], "events", "first measured resource trace"
        )
        if type(first_measured_events) is not list or not first_measured_events:
            fail(f"{phase} first measured lifecycle is incomplete")
        if (
            _event_time(first_measured_events[0], "first measured admission")
            < baseline.sample_times[-1]
        ):
            fail(f"{phase} first measured admission overlaps baseline sampling")
        for point, entry in zip(segment_points, ordered[10:], strict=True):
            request_id, trace, release, _result = entry
            if (
                point.request_id != request_id
                or point.release_observed_monotonic_ns
                != _event_time(release, "measured resource release")
                or point.release_outcome != release.get("outcome")
            ):
                fail(f"{phase} resource point/lifecycle binding differs")
            if any(
                _event_time(release, "resource release")
                < admitted
                <= point.sample_times[-1]
                for admitted in all_admissions
            ):
                fail(f"{phase} resource sample interval contains an admission")
        for point, following in zip(segment_points[:-1], ordered[11:], strict=True):
            following_admit = _event_time(
                cast(list[dict[str, Any]], _attribute(following[1], "events", "trace"))[
                    0
                ],
                "resource following admission",
            )
            if following_admit < point.sample_times[-1]:
                fail(f"{phase} resource requests overlap post-release sampling")

        before = metric_times[(segment, "before")]
        after = metric_times[(segment, "after")]
        first_admission = _event_time(
            cast(list[dict[str, Any]], _attribute(ordered[0][1], "events", "trace"))[0],
            "resource first admission",
        )
        if (
            before > first_admission
            or before > baseline.settle_started_ns
            or after < segment_points[-1].sample_times[-1]
            or after < before
        ):
            fail(f"{segment} GPU metric window differs")
        for request_id_value, trace in traces.items():
            events = _attribute(trace, "events", "trace")
            if type(events) is not list or not events:
                continue
            if (
                _event_time(events[0], "trace start") <= after
                and _event_time(events[-1], "trace end") >= before
                and _attribute(trace, "phase", "trace") != phase
            ):
                fail(f"{segment} metric window contains foreign lifecycle")
        for result in all_results:
            if (
                _exact_int(
                    _attribute(result, "connect_completed_monotonic_ns", "HTTP result"),
                    "HTTP connect",
                )
                <= after
                and _exact_int(
                    _attribute(result, "response_end_monotonic_ns", "HTTP result"),
                    "HTTP end",
                )
                >= before
                and _attribute(result, "phase", "HTTP result") != phase
            ):
                fail(f"{segment} metric window contains foreign HTTP")

    if metric_times[("restart", "before")] < metric_times[("normal", "after")]:
        fail("restart GPU metric window overlaps normal")
    probes = _mapping_attribute(session, "probes", "session")
    normal_probe = probes.get("normal-segment-start")
    restart_probe = probes.get("restart-segment-start")
    if type(normal_probe) is not dict or type(restart_probe) is not dict:
        fail("resource lifecycle probes are absent")
    for segment, probe, name in (
        ("normal", normal_probe, "normal-segment-start"),
        ("restart", restart_probe, "restart-segment-start"),
    ):
        identity = identities[segment]
        if (
            probe.get("probe") != name
            or probe.get("phase")
            != ("resource_normal" if segment == "normal" else "resource_restart")
            or probe.get("service_active") is not True
            or probe.get("ready_http_status") != 200
            or probe.get("control_group") != identity[0]
            or probe.get("gateway_pid") != identity[1]
            or probe.get("gateway_starttime_ticks") != identity[4]
            or probe.get("worker_pid") != identity[5]
            or probe.get("worker_starttime_ticks") != identity[8]
        ):
            fail(f"{segment} lifecycle probe identity differs")
    normal_restarts = _exact_int(normal_probe.get("n_restarts"), "normal restart count")
    restart_restarts = _exact_int(
        restart_probe.get("n_restarts"), "restart restart count"
    )
    normal_probe_time = _exact_int(
        normal_probe.get("observed_monotonic_ns"), "normal probe timestamp"
    )
    restart_probe_time = _exact_int(
        restart_probe.get("observed_monotonic_ns"), "restart probe timestamp"
    )
    normal_first_events = _attribute(
        ordered_positive["resource_normal"][0][1], "events", "normal first trace"
    )
    restart_first_events = _attribute(
        ordered_positive["resource_restart"][0][1], "events", "restart first trace"
    )
    if (
        type(normal_first_events) is not list
        or not normal_first_events
        or type(restart_first_events) is not list
        or not restart_first_events
    ):
        fail("resource first lifecycle trace is incomplete")
    if (
        restart_restarts != normal_restarts + 1
        or normal_probe_time > metric_times[("normal", "before")]
        or restart_probe_time > metric_times[("restart", "before")]
        or normal_probe_time >= restart_probe_time
        or restart_probe_time
        > _event_time(restart_first_events[0], "restart admission")
        or normal_probe_time > _event_time(normal_first_events[0], "normal admission")
    ):
        fail("resource probe restart count or timing differs")


def _segment_result(
    baseline: _ResourcePoint, points: Sequence[_ResourcePoint]
) -> dict[str, Any]:
    expected_count = 100 if baseline.segment == "normal" else 20
    if (
        baseline.phase != "baseline"
        or len(points) != expected_count
        or [point.request_index for point in points]
        != list(range(1, expected_count + 1))
    ):
        fail("resource segment point schedule differs")
    for point in points:
        if (
            point.segment != baseline.segment
            or point.phase != "post_release"
            or point.identity != baseline.identity
        ):
            fail("resource segment point identity differs")
        for name in STABLE_COUNTS:
            if point.stable_counts[name] != baseline.stable_counts[name]:
                fail(f"resource stable process count differs: {name}")
    final = points[-1]
    slopes = {
        name: _theil_sen([point.metrics[name] for point in points])
        for name in RESOURCE_METRICS
    }
    deltas = {
        name: final.metrics[name] - baseline.metrics[name] for name in RESOURCE_METRICS
    }
    for name in ("memory_current_bytes", "process_vram_bytes"):
        if deltas[name] > FINAL_DELTA_MAX_BYTES:
            fail(f"resource final delta exceeds threshold: {name}")
        if slopes[name] > THEIL_SEN_MAX_BYTES_PER_REQUEST:
            fail(f"resource Theil-Sen slope exceeds threshold: {name}")
    return {
        "measured_point_count": expected_count,
        "baseline_median": {
            **{
                name: _fraction_json(baseline.metrics[name])
                for name in RESOURCE_METRICS
            },
            **{
                name: _fraction_json(baseline.stable_counts[name])
                for name in STABLE_COUNTS
            },
        },
        "final_signed_median_delta": {
            name: _fraction_json(deltas[name]) for name in RESOURCE_METRICS
        },
        "complete_theil_sen_per_request": {
            name: _fraction_json(slopes[name]) for name in RESOURCE_METRICS
        },
        "stable_process_counts": {
            "gateway": {
                "threads": _fraction_json(baseline.stable_counts["gateway_threads"]),
                "fds": _fraction_json(baseline.stable_counts["gateway_fds"]),
                "children": _fraction_json(baseline.stable_counts["gateway_children"]),
            },
            "worker": {
                "threads": _fraction_json(baseline.stable_counts["worker_threads"]),
                "fds": _fraction_json(baseline.stable_counts["worker_fds"]),
                "children": _fraction_json(baseline.stable_counts["worker_children"]),
            },
        },
    }


def reconstruct_soak_resource_results(root: Path, session: object) -> dict[str, Any]:
    """Stream, cross-bind, and reconstruct the final resource-only soak view."""

    with _RootSnapshot(root) as snapshot:
        lines = _resource_lines(snapshot)
        try:
            first_number, header = next(lines)
        except StopIteration:
            fail("resource raw file is empty")
        if first_number != 1:
            fail("resource header line number differs")
        _resource_header(header)
        expected = iter(_expected_resource_records())
        groups: list[dict[str, Any]] = []
        baselines: dict[str, _ResourcePoint] = {}
        points: dict[str, list[_ResourcePoint]] = {"normal": [], "restart": []}
        identities: dict[str, tuple[Any, ...]] = {}
        request_ids: set[str] = set()
        metric_times: dict[tuple[str, str], int] = {}
        sample_count = 0
        metric_count = 0
        record_count = 0
        for line_number, record in lines:
            record_count += 1
            try:
                kind, position = next(expected)
            except StopIteration:
                fail(f"resource line {line_number} is extra")
            if kind == "metric":
                typed = cast(tuple[str, str], position)
                metric_times[typed] = _gpu_metric(snapshot, record, typed)
                metric_count += 1
                continue
            sample = _resource_sample(
                record, cast(tuple[str, str, int | None, int], position)
            )
            sample_count += 1
            groups.append(sample)
            if len(groups) < 5:
                continue
            point = _resource_point(groups)
            groups = []
            prior_identity = identities.setdefault(point.segment, point.identity)
            if point.identity != prior_identity:
                fail("resource identity changed inside a segment")
            if point.phase == "baseline":
                if point.segment in baselines:
                    fail("resource baseline is duplicated")
                baselines[point.segment] = point
            else:
                if point.request_id is None or point.request_id in request_ids:
                    fail("resource request ID is absent or duplicated")
                request_ids.add(point.request_id)
                points[point.segment].append(point)
        try:
            next(expected)
        except StopIteration:
            pass
        else:
            fail("resource raw state machine is incomplete")
        if (
            groups
            or record_count != 614
            or sample_count != 610
            or metric_count != 4
            or set(baselines) != {"normal", "restart"}
            or len(request_ids) != 120
            or len(metric_times) != 4
        ):
            fail("resource 1+610+4 contract is incomplete")
        normal_identity = identities["normal"]
        restart_identity = identities["restart"]
        if (
            normal_identity[0] != restart_identity[0]
            or (normal_identity[1], normal_identity[4])
            == (restart_identity[1], restart_identity[4])
            or (normal_identity[5], normal_identity[8])
            == (restart_identity[5], restart_identity[8])
        ):
            fail("resource restart epoch identity differs")
        _session_resource_bindings(session, baselines, points, identities, metric_times)
        snapshot.verify_root()
        return {
            "resource_sample_count": 610,
            "gpu_metric_count": 4,
            "segments": {
                segment: _segment_result(baselines[segment], points[segment])
                for segment in ("normal", "restart")
            },
        }


__all__ = [
    "IndependentMetricsError",
    "LATENCY_RESULTS_SCHEMA",
    "LATENCY_SCHEDULE",
    "RESOURCE_HTTP_SCHEDULE",
    "RESOURCE_INPUT_SCHEMA",
    "reconstruct_latency_results",
    "reconstruct_soak_resource_results",
]
