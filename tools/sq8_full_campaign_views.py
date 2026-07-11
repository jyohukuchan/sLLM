#!/usr/bin/env python3
"""Build the six canonical producer views for one full SQ8 campaign."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import re
import stat
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence, cast


API_INPUT_SCHEMA = "ullm.sq8.api_contract_gate_ingest.view.v1"
COMBINED_INPUT_SCHEMA = "ullm.sq8.openwebui_gate_ingest.combined_view.v1"
DIRECT_CANCEL_INPUT_SCHEMA = "ullm.sq8.direct_cancel_gate_ingest.view.v1"
STOP_INPUT_SCHEMA = "ullm.sq8.openwebui_stop_gate_ingest.view.v1"
FAILURE_INPUT_SCHEMA = "ullm.sq8.openwebui_failure_gate_ingest.view.v1"
LATENCY_INPUT_SCHEMA = "ullm.sq8.http_latency_gate_ingest.view.v1"
RESOURCE_INPUT_SCHEMA = "ullm.sq8.release_measurement.raw.v1"

API_RESULTS_SCHEMA = "ullm.sq8.openwebui_release.api_contract_results.v1"
CANCEL_RESULTS_SCHEMA = "ullm.sq8.openwebui_release.cancel_results.v1"
LATENCY_RESULTS_SCHEMA = "ullm.sq8.openwebui_release.prefill_latency_results.v1"
OPENWEBUI_SMOKE_SCHEMA = "ullm.sq8.openwebui_release.openwebui_smoke.v1"
SAMPLING_RESULTS_SCHEMA = "ullm.sq8.openwebui_release.sampling_results.v1"
SOAK_RESULTS_SCHEMA = "ullm.sq8.openwebui_release.soak_results.v1"

VIEW_FILENAMES = (
    "sampling-results.json",
    "cancel-results.json",
    "prefill-latency-results.json",
    "api-contract-results.json",
    "openwebui-smoke.json",
    "soak-results.json",
)
DIRECT_CANCEL_PHASES = (
    "after_started_before_progress",
    "prefill_after_128",
    "prefill_after_2048",
    "decode_after_first_content",
)
CANCEL_PHASES = DIRECT_CANCEL_PHASES + ("openwebui_stop_after_visible_content",)
FIXTURE_ORDER = (
    ("exact-p0032", 32),
    ("exact-p0128", 128),
    ("exact-p0512", 512),
    ("exact-p2048", 2048),
    ("exact-p3584", 3584),
)
SAMPLED_NORMAL_INDICES = tuple(range(5, 101, 5))
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

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
MAX_RESOURCE_BYTES = 512 * 1024 * 1024
MAX_RESOURCE_LINE_BYTES = 1024 * 1024

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


class FullCampaignViewError(ValueError):
    """A fail-closed producer-view construction error."""


def fail(message: str) -> None:
    raise FullCampaignViewError(message)


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        fail(f"{label} fields differ")
    return cast(dict[str, Any], value)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        fail(f"{label} is not an object")
    return cast(dict[str, Any], value)


def _array(value: Any, label: str, *, length: int | None = None) -> list[Any]:
    if type(value) is not list or (length is not None and len(value) != length):
        fail(f"{label} array length or type differs")
    return cast(list[Any], value)


def _integer(
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
    return cast(int, value)


def _float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        fail(f"{label} is not a finite JSON float")
    return cast(float, value)


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        fail(f"{label} is not a boolean")
    return cast(bool, value)


def _text(value: Any, label: str, *, maximum: int = 4096) -> str:
    if type(value) is not str or not value or len(value) > maximum or "\0" in value:
        fail(f"{label} is not bounded non-empty text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        fail(f"{label} is not strict UTF-8")
    return cast(str, value)


def _sha256(value: Any, label: str) -> str:
    text = _text(value, label, maximum=64)
    if SHA256_RE.fullmatch(text) is None:
        fail(f"{label} is not lowercase SHA-256")
    return text


def _fraction_value(value: Any, label: str) -> Fraction:
    if type(value) is int:
        return Fraction(value)
    item = _object(value, {"numerator", "denominator"}, label)
    numerator = _integer(item["numerator"], f"{label}.numerator", minimum=-(1 << 255))
    denominator = _integer(item["denominator"], f"{label}.denominator", minimum=1)
    return Fraction(numerator, denominator)


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
    if not converted:
        fail("median input is empty")
    converted.sort()
    middle = len(converted) // 2
    if len(converted) % 2:
        return converted[middle]
    return (converted[middle - 1] + converted[middle]) / 2


def _theil_sen(values: Sequence[Fraction]) -> Fraction:
    if len(values) < 2 or any(type(value) is not Fraction for value in values):
        fail("Theil-Sen input differs")
    slopes = [
        (values[j] - values[i]) / (j - i)
        for i in range(len(values))
        for j in range(i + 1, len(values))
    ]
    if len(slopes) != len(values) * (len(values) - 1) // 2:
        fail("Theil-Sen did not include every pair")
    return _median(slopes)


def _reject_passed(value: Any, label: str) -> None:
    if type(value) is dict:
        if "passed" in value:
            fail(f"{label} contains forbidden key passed")
        for key, item in value.items():
            if type(key) is not str:
                fail(f"{label} contains a non-text object key")
            _reject_passed(item, label)
    elif type(value) is list:
        for item in value:
            _reject_passed(item, label)


def _forbidden(forbidden_values: tuple[bytes, ...]) -> tuple[bytes, ...]:
    if type(forbidden_values) is not tuple:
        fail("forbidden secret values are mutable")
    for value in forbidden_values:
        if type(value) is not bytes or len(value) < 4 or b"\0" in value:
            fail("forbidden secret value syntax differs")
    return forbidden_values


def _scan(raw: bytes, forbidden_values: tuple[bytes, ...], label: str) -> None:
    for value in forbidden_values:
        if value in raw:
            fail(f"{label} contains forbidden secret cleartext")


def _scan_json_semantics(
    value: Any, forbidden_values: tuple[bytes, ...], label: str
) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    visited = 0
    while pending:
        item, depth = pending.pop()
        visited += 1
        if depth > 128 or visited > 100_000:
            fail(f"{label} exceeds the semantic secret-scan bound")
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is str:
                    try:
                        _scan(
                            key.encode("utf-8", errors="strict"),
                            forbidden_values,
                            label,
                        )
                    except UnicodeError:
                        fail(f"{label} contains a non-UTF-8 object key")
                pending.append((child, depth + 1))
        elif type(item) in {list, tuple}:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            try:
                _scan(item.encode("utf-8", errors="strict"), forbidden_values, label)
            except UnicodeError:
                fail(f"{label} contains a non-UTF-8 string")
        elif type(item) in {bytes, bytearray}:
            _scan(bytes(item), forbidden_values, label)


def canonical_json_bytes(
    value: Mapping[str, Any], *, forbidden_values: tuple[bytes, ...] = ()
) -> bytes:
    """Return one canonical ASCII JSON document terminated by one LF."""

    forbidden = _forbidden(forbidden_values)
    if type(value) is not dict:
        fail("canonical view root is not an object")
    _reject_passed(value, "canonical view")
    _scan_json_semantics(value, forbidden, "canonical view")
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
        raise FullCampaignViewError("canonical view cannot be encoded") from error
    _scan(raw, forbidden, "canonical view")
    return raw


def project_api_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    source = _object(
        value,
        {
            "schema_version",
            "case_count",
            "http_record_count",
            "journal_record_count",
            "lifecycle_event_count",
            "quiet_check_count",
            "cases",
            "source_bindings",
        },
        "API ingest view",
    )
    if (
        source["schema_version"] != API_INPUT_SCHEMA
        or _integer(source["case_count"], "API case count") != 10
        or _integer(source["http_record_count"], "API HTTP record count") != 40
        or _integer(source["lifecycle_event_count"], "API lifecycle count") != 0
        or _integer(source["quiet_check_count"], "API quiet count") != 13
    ):
        fail("API ingest counts or schema differ")
    _mapping(source["source_bindings"], "API source bindings")
    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for position, raw_case in enumerate(
        _array(source["cases"], "API cases", length=10), 1
    ):
        case = _object(
            raw_case,
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
            f"API case {position}",
        )
        case_index = _integer(case["case_index"], "API case index", minimum=1)
        case_id = _text(case["case_id"], "API case ID", maximum=128)
        if case_index != position or case_id in case_ids:
            fail("API case order or uniqueness differs")
        case_ids.add(case_id)
        status = _integer(case["status"], "API status", minimum=100, maximum=599)
        request_bytes = _integer(case["request_body_bytes"], "API request bytes")
        response_bytes = _integer(case["response_body_bytes"], "API response bytes")
        request_sha = _sha256(case["request_body_sha256"], "API request SHA-256")
        response_sha = _sha256(case["response_body_sha256"], "API response SHA-256")
        error = case["error"]
        projected_error: dict[str, Any] | None
        if error is None:
            if status >= 400:
                fail("API error response lacks a redacted error summary")
            projected_error = None
        else:
            item = _object(
                error,
                {
                    "type",
                    "code",
                    "param",
                    "message_utf8_bytes",
                    "message_sha256",
                },
                f"API case {position} error",
            )
            if status < 400:
                fail("successful API case contains an error summary")
            projected_error = {
                "type": _text(item["type"], "API error type", maximum=128),
                "code": _text(item["code"], "API error code", maximum=128),
                "param": None
                if item["param"] is None
                else _text(item["param"], "API error param", maximum=128),
                "message_utf8_bytes": _integer(
                    item["message_utf8_bytes"], "API error message bytes", minimum=1
                ),
                "message_sha256": _sha256(
                    item["message_sha256"], "API error message SHA-256"
                ),
            }
        cases.append(
            {
                "case_index": case_index,
                "case_id": case_id,
                "status": status,
                "request_body_bytes": request_bytes,
                "request_body_sha256": request_sha,
                "response_body_bytes": response_bytes,
                "response_body_sha256": response_sha,
                "error": projected_error,
            }
        )
    return {
        "schema_version": API_RESULTS_SCHEMA,
        "case_count": 10,
        "http_record_count": 40,
        "quiet_check_count": 13,
        "cases": cases,
    }


def _combined_cases(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = _object(
        value,
        {
            "schema_version",
            "mode",
            "schedule",
            "chat_count",
            "action_count",
            "lifecycle_record_count",
            "maximum_active_requests",
            "stop_release_count",
            "reset_complete_count",
            "component_summary_sha256",
            "source_bindings",
            "cases",
        },
        "combined browser ingest view",
    )
    if (
        source["schema_version"] != COMBINED_INPUT_SCHEMA
        or source["mode"] != "smoke_then_soak20"
        or _integer(source["chat_count"], "combined chat count") != 21
        or _integer(source["action_count"], "combined action count") != 105
        or _integer(source["maximum_active_requests"], "combined active count") != 1
        or _integer(source["stop_release_count"], "combined stop count") != 21
        or _integer(source["reset_complete_count"], "combined reset count") != 21
    ):
        fail("combined browser counts or schema differ")
    _sha256(source["component_summary_sha256"], "combined summary SHA-256")
    _mapping(source["source_bindings"], "combined source bindings")
    schedule = _array(source["schedule"], "combined schedule", length=21)
    raw_cases = _array(source["cases"], "combined cases", length=21)
    projected: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for position, (raw_schedule, raw_case) in enumerate(
        zip(schedule, raw_cases, strict=True)
    ):
        expected_index = position
        expected_kind = "smoke" if position == 0 else "soak"
        expected_id = (
            "openwebui_smoke"
            if position == 0
            else f"openwebui_soak_chat_{position:02d}"
        )
        schedule_item = _object(
            raw_schedule,
            {"position", "case_index", "case_kind", "browser_case"},
            f"combined schedule {position}",
        )
        case = _object(
            raw_case,
            {
                "position",
                "case_index",
                "case_kind",
                "browser_case",
                "browser_case_sha256",
                "action_count",
                "socket_event_count",
                "chat_id_sha256",
                "message_id_sha256",
                "request_id_sha256",
                "completion_id_sha256",
                "admitted_monotonic_ns",
                "released_monotonic_ns",
                "outcome",
                "reset_complete",
            },
            f"combined case {position}",
        )
        for name, expected in (
            ("position", position),
            ("case_index", expected_index),
            ("case_kind", expected_kind),
            ("browser_case", expected_id),
        ):
            if schedule_item[name] != expected or case.get(name) != expected:
                fail("combined browser schedule or case order differs")
        if expected_id in case_ids:
            fail("combined browser case ID is duplicated")
        case_ids.add(expected_id)
        if _integer(case.get("action_count"), "combined action count") != 5:
            fail("combined per-chat action count differs")
        outcome = _text(case.get("outcome"), "combined release outcome", maximum=32)
        if outcome not in {"stop", "length"}:
            fail("combined release outcome differs")
        if (
            _boolean(case.get("reset_complete"), "combined reset acknowledgement")
            is not True
        ):
            fail("combined request was not reset")
        projected.append(
            {
                "case_index": expected_index,
                "case_id": expected_id,
                "action_count": 5,
                "release_outcome": outcome,
                "reset_complete": True,
            }
        )
    return projected


def project_browser_soak(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project the twenty browser chats used inside soak-results.json."""

    return copy.deepcopy(_combined_cases(value)[1:])


def _direct_cancel_pairs(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = _object(
        value,
        {
            "schema_version",
            "phase_order",
            "request_count",
            "http_record_count",
            "lifecycle_record_count",
            "maximum_active_requests",
            "component_summary_sha256",
            "source_bindings",
            "cases",
        },
        "direct cancellation ingest view",
    )
    if (
        source["schema_version"] != DIRECT_CANCEL_INPUT_SCHEMA
        or source["phase_order"] != list(DIRECT_CANCEL_PHASES)
        or _integer(source["request_count"], "direct request count") != 8
        or not 32
        <= _integer(source["http_record_count"], "direct HTTP record count")
        <= 2048
        or _integer(source["lifecycle_record_count"], "direct lifecycle count") != 55
        or _integer(source["maximum_active_requests"], "direct active count") != 1
    ):
        fail("direct cancellation counts or schema differ")
    _sha256(source["component_summary_sha256"], "direct summary SHA-256")
    _mapping(source["source_bindings"], "direct source bindings")
    raw_cases = _array(source["cases"], "direct cases", length=8)
    pairs: list[dict[str, Any]] = []
    request_indices: set[int] = set()
    case_ids: set[str] = set()
    for pair_index, phase in enumerate(DIRECT_CANCEL_PHASES):
        projected_roles: dict[str, dict[str, Any]] = {}
        for role_offset, role in enumerate(("target", "recovery")):
            common_fields = {
                "request_index",
                "phase",
                "role",
                "case_id",
                "request_body_bytes",
                "request_body_sha256",
                "http_status",
                "http_outcome",
                "response_body_bytes",
                "response_body_sha256",
                "lifecycle_event_count",
                "request_id_sha256",
                "completion_id_sha256",
                "release_observed_monotonic_ns",
                "release_outcome",
                "reset_complete",
                "completion_tokens",
            }
            target_fields = {
                "trigger_observed_monotonic_ns",
                "cancel_observed_monotonic_ns",
                "cancel_to_release_ns",
                "progress",
            }
            case = _object(
                raw_cases[pair_index * 2 + role_offset],
                common_fields | (target_fields if role == "target" else set()),
                "direct case",
            )
            if case.get("phase") != phase or case.get("role") != role:
                fail("direct target/recovery order differs")
            request_index = _integer(
                case.get("request_index"), "direct request index", minimum=1
            )
            case_id = _text(case.get("case_id"), "direct case ID", maximum=128)
            if request_index in request_indices or case_id in case_ids:
                fail("direct request index or case ID is duplicated")
            request_indices.add(request_index)
            case_ids.add(case_id)
            _integer(case["request_body_bytes"], "direct request bytes")
            _sha256(case["request_body_sha256"], "direct request SHA-256")
            _integer(case["response_body_bytes"], "direct response bytes")
            _sha256(case["response_body_sha256"], "direct response SHA-256")
            _integer(
                case["lifecycle_event_count"], "direct lifecycle event count", minimum=1
            )
            _sha256(case["request_id_sha256"], "direct request ID SHA-256")
            _sha256(case["completion_id_sha256"], "direct completion ID SHA-256")
            _integer(
                case["release_observed_monotonic_ns"],
                "direct release timestamp",
                minimum=1,
            )
            status = _integer(
                case.get("http_status"), "direct HTTP status", minimum=100, maximum=599
            )
            http_outcome = _text(
                case.get("http_outcome"), "direct HTTP outcome", maximum=32
            )
            release_outcome = _text(
                case.get("release_outcome"), "direct release outcome", maximum=32
            )
            completion_tokens = _integer(
                case.get("completion_tokens"), "direct completion tokens"
            )
            if (
                _boolean(case.get("reset_complete"), "direct reset acknowledgement")
                is not True
            ):
                fail("direct request was not reset")
            if role == "target":
                cancel_to_release = _integer(
                    case.get("cancel_to_release_ns"),
                    "direct cancel-to-release",
                    maximum=5_000_000_000,
                )
                if (
                    status != 200
                    or http_outcome != "client_closed"
                    or release_outcome != "cancelled"
                ):
                    fail("direct target HTTP or cancellation result differs")
                projected_roles[role] = {
                    "case_id": case_id,
                    "transport": "direct_http",
                    "http_status": status,
                    "http_outcome": http_outcome,
                    "release_outcome": release_outcome,
                    "cancel_reason": "client_disconnect",
                    "cancel_to_release_ns": cancel_to_release,
                    "completion_tokens": completion_tokens,
                    "reset_complete": True,
                }
            else:
                if (
                    status != 200
                    or http_outcome != "eof"
                    or release_outcome != "length"
                    or completion_tokens != 2
                ):
                    fail("direct recovery HTTP or release result differs")
                projected_roles[role] = {
                    "case_id": case_id,
                    "transport": "direct_http",
                    "http_status": status,
                    "http_outcome": http_outcome,
                    "release_outcome": release_outcome,
                    "completion_tokens": 2,
                    "reset_complete": True,
                }
        pairs.append({"phase": phase, **projected_roles})
    if request_indices != set(range(1, 9)):
        fail("direct request indices differ from one through eight")
    return pairs


def _stop_pair(value: Mapping[str, Any]) -> dict[str, Any]:
    source = _object(
        value,
        {
            "schema_version",
            "browser_case",
            "browser_action_count",
            "browser_socket_event_count",
            "lifecycle_event_count",
            "request_count",
            "maximum_active_requests",
            "component_summary_sha256",
            "screenshot",
            "source_bindings",
            "browser_evidence",
            "gateway_evidence",
            "raw_artifacts",
        },
        "Stop ingest view",
    )
    phase = CANCEL_PHASES[-1]
    if (
        source["schema_version"] != STOP_INPUT_SCHEMA
        or source["browser_case"] != phase
        or _integer(source["browser_action_count"], "Stop action count") != 9
        or _integer(source["lifecycle_event_count"], "Stop lifecycle count") != 11
        or _integer(source["request_count"], "Stop request count") != 2
        or _integer(source["maximum_active_requests"], "Stop active count") != 1
    ):
        fail("Stop counts or schema differ")
    _sha256(source["component_summary_sha256"], "Stop summary SHA-256")
    screenshot = _object(
        source["screenshot"], {"file", "bytes", "sha256"}, "Stop screenshot"
    )
    if screenshot["file"] != "browser/openwebui-stop-before.png":
        fail("Stop screenshot bundle path differs")
    gateway = _mapping(source["gateway_evidence"], "Stop gateway evidence")
    target_outcome = _text(
        gateway.get("target_outcome"), "Stop target outcome", maximum=32
    )
    cancel_reason = _text(
        gateway.get("cancel_reason"), "Stop cancel reason", maximum=64
    )
    recovery_outcome = _text(
        gateway.get("recovery_outcome"), "Stop recovery outcome", maximum=32
    )
    if (
        target_outcome != "cancelled"
        or cancel_reason != "client_disconnect"
        or recovery_outcome != "stop"
        or gateway.get("target_reset_complete") is not True
        or gateway.get("recovery_reset_complete") is not True
    ):
        fail("Stop target or recovery outcome differs")
    return {
        "phase": phase,
        "target": {
            "case_id": phase,
            "transport": "openwebui_browser",
            "http_status": None,
            "http_outcome": None,
            "release_outcome": target_outcome,
            "cancel_reason": cancel_reason,
            "cancel_to_release_ns": _integer(
                gateway.get("cancel_to_release_ns"),
                "Stop cancel-to-release",
                maximum=5_000_000_000,
            ),
            "completion_tokens": None,
            "reset_complete": True,
        },
        "recovery": {
            "case_id": f"{phase}-recovery",
            "transport": "openwebui_browser",
            "http_status": None,
            "http_outcome": None,
            "release_outcome": recovery_outcome,
            "completion_tokens": None,
            "reset_complete": True,
        },
        "browser_action_count": 9,
        "screenshot": {
            "file": screenshot["file"],
            "bytes": _integer(screenshot["bytes"], "Stop screenshot bytes", minimum=1),
            "sha256": _sha256(screenshot["sha256"], "Stop screenshot SHA-256"),
        },
    }


def project_cancellation(
    direct_cancel: Mapping[str, Any], stop: Mapping[str, Any]
) -> dict[str, Any]:
    phases = _direct_cancel_pairs(direct_cancel) + [_stop_pair(stop)]
    if [item["phase"] for item in phases] != list(CANCEL_PHASES):
        fail("complete cancellation phase order differs")
    return {
        "schema_version": CANCEL_RESULTS_SCHEMA,
        "phase_count": 5,
        "request_count": 10,
        "maximum_active_requests": 1,
        "phases": phases,
    }


def _expected_latency_schedule() -> Iterator[tuple[int, str, str, int, str, int]]:
    sequence = 0
    for fixture_id, prompt_tokens in FIXTURE_ORDER:
        for sample_kind, count in (("warmup", 2), ("measured", 10)):
            for sample_index in range(1, count + 1):
                sequence += 1
                yield (
                    sequence,
                    "ttft",
                    sample_kind,
                    sample_index,
                    fixture_id,
                    prompt_tokens,
                )
    for sample_kind, count in (("warmup", 2), ("measured", 10)):
        for sample_index in range(1, count + 1):
            sequence += 1
            yield sequence, "decode64", sample_kind, sample_index, "exact-p0032", 32


def _latency_sample(
    raw: Any,
    expected: tuple[int, str, str, int, str, int],
) -> dict[str, Any]:
    sequence, workload, sample_kind, sample_index, fixture_id, prompt_tokens = expected
    common = {
        "sequence",
        "case_id",
        "sample_kind",
        "sample_index",
        "fixture_id",
        "prompt_tokens",
        "release_outcome",
        "release_completion_tokens",
    }
    fields = common | (
        {"ttft_ns", "content_object_count"}
        if workload == "ttft"
        else {"decode_elapsed_ns", "decode_intervals_ns", "decode_tokens_per_second"}
    )
    sample = _object(raw, fields, f"latency sample {sequence}")
    expected_case = (
        f"ttft-{fixture_id}-{sample_kind}-{sample_index:02d}"
        if workload == "ttft"
        else f"decode64-{sample_kind}-{sample_index:02d}"
    )
    for name, value in (
        ("sequence", sequence),
        ("case_id", expected_case),
        ("sample_kind", sample_kind),
        ("sample_index", sample_index),
        ("fixture_id", fixture_id),
        ("prompt_tokens", prompt_tokens),
    ):
        if sample[name] != value:
            fail("latency sample schedule differs")
    release_outcome = _text(
        sample["release_outcome"], "latency release outcome", maximum=32
    )
    release_tokens = _integer(
        sample["release_completion_tokens"], "latency release tokens", minimum=1
    )
    if workload == "ttft":
        if release_outcome != "cancelled":
            fail("TTFT release outcome differs")
        return {
            "sequence": sequence,
            "case_id": expected_case,
            "sample_kind": sample_kind,
            "sample_index": sample_index,
            "fixture_id": fixture_id,
            "prompt_tokens": prompt_tokens,
            "ttft_ns": _integer(sample["ttft_ns"], "TTFT", minimum=1),
            "content_object_count": _integer(
                sample["content_object_count"], "TTFT content count", minimum=1
            ),
            "release_outcome": release_outcome,
            "release_completion_tokens": release_tokens,
        }
    if release_outcome != "length" or release_tokens != 64:
        fail("decode64 release outcome differs")
    intervals = _array(sample["decode_intervals_ns"], "decode64 intervals", length=63)
    projected_intervals = [
        _integer(item, "decode64 interval", minimum=1) for item in intervals
    ]
    throughput = _fraction_value(
        sample["decode_tokens_per_second"], "decode64 tokens per second"
    )
    return {
        "sequence": sequence,
        "case_id": expected_case,
        "sample_kind": sample_kind,
        "sample_index": sample_index,
        "fixture_id": fixture_id,
        "prompt_tokens": prompt_tokens,
        "decode_elapsed_ns": _integer(
            sample["decode_elapsed_ns"], "decode64 elapsed", minimum=1
        ),
        "decode_intervals_ns": projected_intervals,
        "decode_tokens_per_second": _fraction_json(throughput),
        "release_outcome": release_outcome,
        "release_completion_tokens": 64,
    }


def _latency_metrics(value: Any) -> dict[str, Any]:
    metrics = _object(value, {"ttft", "decode64"}, "latency metrics")
    ttft = _mapping(metrics["ttft"], "TTFT metrics")
    if set(ttft) != {fixture for fixture, _tokens in FIXTURE_ORDER}:
        fail("TTFT metric fixture set differs")
    projected_ttft: dict[str, Any] = {}
    for fixture_id, _tokens in FIXTURE_ORDER:
        item = _object(
            ttft[fixture_id],
            {"count", "p50_ns", "p95_ns", "p50_maximum_ns", "p95_maximum_ns"},
            f"TTFT metric {fixture_id}",
        )
        if _integer(item["count"], "TTFT metric count") != 10:
            fail("TTFT measured population differs")
        p50 = _fraction_value(item["p50_ns"], "TTFT p50_ns")
        p95 = _fraction_value(item["p95_ns"], "TTFT p95_ns")
        for name in ("p50_maximum_ns", "p95_maximum_ns"):
            _integer(item[name], f"TTFT {name}", minimum=1)
        projected_ttft[fixture_id] = {
            "count": 10,
            "p50_ns": _fraction_json(p50),
            "p95_ns": _fraction_json(p95),
            "p50_maximum_ns": item["p50_maximum_ns"],
            "p95_maximum_ns": item["p95_maximum_ns"],
        }
    decode = _object(
        metrics["decode64"],
        {
            "request_count",
            "interval_count",
            "p50_tokens_per_second",
            "minimum_p50_tokens_per_second",
            "p95_inter_content_ns",
            "maximum_p95_inter_content_ns",
        },
        "decode64 metrics",
    )
    if (
        _integer(decode["request_count"], "decode64 measured count") != 10
        or _integer(decode["interval_count"], "decode64 interval count") != 630
    ):
        fail("decode64 metric population differs")
    throughput = _fraction_value(
        decode["p50_tokens_per_second"], "decode64 p50 throughput"
    )
    interval = _fraction_value(decode["p95_inter_content_ns"], "decode64 p95 interval")
    _integer(
        decode["minimum_p50_tokens_per_second"],
        "decode64 throughput minimum",
        minimum=1,
    )
    _integer(
        decode["maximum_p95_inter_content_ns"], "decode64 interval maximum", minimum=1
    )
    return {
        "ttft": projected_ttft,
        "decode64": {
            "request_count": 10,
            "interval_count": 630,
            "p50_tokens_per_second": _fraction_json(throughput),
            "minimum_p50_tokens_per_second": decode["minimum_p50_tokens_per_second"],
            "p95_inter_content_ns": _fraction_json(interval),
            "maximum_p95_inter_content_ns": decode["maximum_p95_inter_content_ns"],
        },
    }


def project_latency(value: Mapping[str, Any]) -> dict[str, Any]:
    source = _object(
        value,
        {
            "schema_version",
            "request_count",
            "http_record_count",
            "lifecycle_record_count",
            "journal_record_count",
            "prefill_ttft",
            "decode64",
            "source_bindings",
        },
        "latency ingest view",
    )
    if (
        source["schema_version"] != LATENCY_INPUT_SCHEMA
        or _integer(source["request_count"], "latency request count") != 72
    ):
        fail("latency schema or request count differs")
    _mapping(source["source_bindings"], "latency source bindings")
    prefill = _object(
        source["prefill_ttft"], {"request_count", "metrics", "samples"}, "TTFT view"
    )
    decode = _object(
        source["decode64"], {"request_count", "metrics", "samples"}, "decode64 view"
    )
    if (
        _integer(prefill["request_count"], "TTFT request count") != 60
        or _integer(decode["request_count"], "decode64 request count") != 12
    ):
        fail("latency workload request count differs")
    raw_samples = _array(prefill["samples"], "TTFT samples", length=60) + _array(
        decode["samples"], "decode64 samples", length=12
    )
    expected = list(_expected_latency_schedule())
    samples = [
        _latency_sample(raw, schedule)
        for raw, schedule in zip(raw_samples, expected, strict=True)
    ]
    metrics = _latency_metrics(
        {"ttft": prefill["metrics"], "decode64": decode["metrics"]}
    )
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


def project_openwebui_smoke(
    combined: Mapping[str, Any], failure: Mapping[str, Any]
) -> dict[str, Any]:
    normal = _combined_cases(combined)[0]
    source = _object(
        failure,
        {
            "schema_version",
            "phase",
            "cases",
            "source_sha256",
            "summary_sha256",
            "service",
            "browser",
            "fault",
            "recovery",
            "journal",
        },
        "failure ingest view",
    )
    if (
        source["schema_version"] != FAILURE_INPUT_SCHEMA
        or source["phase"] != "post_header_failure"
    ):
        fail("failure schema or phase differs")
    cases = _object(source["cases"], {"failure", "recovery"}, "failure cases")
    if cases != {"failure": "post-header-failure", "recovery": "post-header-recovery"}:
        fail("failure case IDs differ")
    browser = _object(
        source["browser"],
        {
            "action_count",
            "socket_event_count",
            "screenshot_file",
            "screenshot_bytes",
            "screenshot_sha256",
        },
        "failure browser evidence",
    )
    if _integer(browser["action_count"], "failure action count") != 9:
        fail("failure action count differs")
    if browser["screenshot_file"] != "browser/post-header-failure.png":
        fail("failure screenshot path differs")
    fault = _object(
        source["fault"],
        {
            "target_pid",
            "started_monotonic_ns",
            "completed_monotonic_ns",
            "worker_fatal_monotonic_ns",
        },
        "failure fault",
    )
    recovery = _object(
        source["recovery"],
        {
            "ready_completed_monotonic_ns",
            "admitted_monotonic_ns",
            "released_monotonic_ns",
        },
        "failure recovery",
    )
    fault_started = _integer(fault["started_monotonic_ns"], "fault start", minimum=1)
    fault_completed = _integer(
        fault["completed_monotonic_ns"], "fault completion", minimum=1
    )
    worker_fatal = _integer(
        fault["worker_fatal_monotonic_ns"], "worker fatal", minimum=1
    )
    ready = _integer(
        recovery["ready_completed_monotonic_ns"], "recovery ready", minimum=1
    )
    admitted = _integer(
        recovery["admitted_monotonic_ns"], "recovery admission", minimum=1
    )
    released = _integer(
        recovery["released_monotonic_ns"], "recovery release", minimum=1
    )
    if (
        not fault_started
        <= fault_completed
        <= worker_fatal
        <= ready
        <= admitted
        <= released
    ):
        fail("failure and recovery timeline differs")
    return {
        "schema_version": OPENWEBUI_SMOKE_SCHEMA,
        "normal": normal,
        "post_header_failure": {
            "case_id": cases["failure"],
            "action_count": 5,
            "terminal_event": "worker_fatal",
            "fault_injection": "post_header_worker_kill",
            "screenshot": {
                "file": browser["screenshot_file"],
                "bytes": _integer(
                    browser["screenshot_bytes"], "failure screenshot bytes", minimum=1
                ),
                "sha256": _sha256(
                    browser["screenshot_sha256"], "failure screenshot SHA-256"
                ),
            },
        },
        "recovery": {
            "case_id": cases["recovery"],
            "action_count": 4,
            "terminal_event": "request_released",
            "release_outcome": "stop",
            "reset_complete": True,
        },
    }


def project_sampling(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes, bytearray)):
        fail("sampling cases are not an immutable-style sequence")
    if len(cases) != 20:
        fail("sampling case count differs")
    projected: list[dict[str, Any]] = []
    indices: set[int] = set()
    for expected_index, raw in zip(SAMPLED_NORMAL_INDICES, cases, strict=True):
        case = _object(
            raw,
            {
                "request_index",
                "temperature",
                "top_p",
                "seed",
                "http_status",
                "http_outcome",
                "release_outcome",
                "completion_tokens",
                "reset_complete",
            },
            f"sampling case {expected_index}",
        )
        request_index = _integer(
            case["request_index"], "sampling request index", minimum=1
        )
        temperature = _float(case["temperature"], "sampling temperature")
        top_p = _float(case["top_p"], "sampling top_p")
        seed = _integer(case["seed"], "sampling seed", minimum=1)
        if (
            request_index != expected_index
            or request_index in indices
            or temperature != 0.6
            or top_p != 0.95
            or seed != expected_index
            or _integer(case["http_status"], "sampling HTTP status") != 200
            or case["http_outcome"] != "eof"
            or case["release_outcome"] != "length"
            or _integer(case["completion_tokens"], "sampling completion tokens") != 2
            or case["reset_complete"] is not True
        ):
            fail("sampling case schedule or outcome differs")
        indices.add(request_index)
        projected.append(copy.deepcopy(case))
    return {
        "schema_version": SAMPLING_RESULTS_SCHEMA,
        "sampled_request_count": 20,
        "sampled_normal_indices": list(SAMPLED_NORMAL_INDICES),
        "cases": projected,
    }


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("resource JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    fail("resource JSON contains a non-finite number")


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        fail("resource JSON contains a non-finite number")
    return parsed


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
            object_pairs_hook=_reject_duplicate_pairs,
            parse_float=_parse_float,
            parse_constant=_reject_constant,
        )
    except FullCampaignViewError:
        raise
    except (UnicodeError, ValueError, RecursionError) as error:
        raise FullCampaignViewError(f"{label} is not strict ASCII JSON") from error
    if type(value) is not dict:
        fail(f"{label} root is not an object")
    if canonical_json_bytes(cast(dict[str, Any], value)) != raw:
        fail(f"{label} is not canonical JSON+LF")
    _reject_passed(value, label)
    return cast(dict[str, Any], value)


def _resource_lines(
    path: Path, forbidden_values: tuple[bytes, ...]
) -> Iterator[tuple[int, dict[str, Any]]]:
    if not isinstance(path, os.PathLike):
        fail("resource raw path has the wrong type")
    descriptor = -1
    handle: Any | None = None
    before: os.stat_result | None = None
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > MAX_RESOURCE_BYTES
        ):
            fail("resource raw file identity, links, or size differs")
        handle = os.fdopen(descriptor, "rb", buffering=0)
        descriptor = -1
        line_number = 0
        while True:
            raw = handle.readline(MAX_RESOURCE_LINE_BYTES + 1)
            if not raw:
                break
            line_number += 1
            if len(raw) > MAX_RESOURCE_LINE_BYTES:
                fail("resource raw line exceeds its size limit")
            _scan(raw, forbidden_values, f"resource line {line_number}")
            yield line_number, _resource_document(raw, line_number)
        after = os.fstat(handle.fileno())
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            fail("resource raw file changed while streaming")
    except FullCampaignViewError:
        raise
    except OSError as error:
        raise FullCampaignViewError("failed to stream resource raw file") from error
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError as error:
                raise FullCampaignViewError(
                    "failed to close resource raw file"
                ) from error
        elif descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as error:
                raise FullCampaignViewError(
                    "failed to close resource descriptor"
                ) from error


def _resource_header(record: Any) -> None:
    header = _object(record, RESOURCE_HEADER_FIELDS, "resource header")
    if (
        header["schema_version"] != RESOURCE_INPUT_SCHEMA
        or header["record_type"] != "header"
        or header["service_unit"] != "ullm-openai.service"
        or header["schedule"] != RESOURCE_SCHEDULE
    ):
        fail("resource header schema, service, or schedule differs")
    commands = _mapping(header["commands"], "resource commands")
    if commands != RESOURCE_COMMANDS:
        fail("resource command identity differs")
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
    systemd_line = _text(
        tools["systemd_version_line"], "resource systemd version line", maximum=8192
    )
    version_output = _text(
        tools["amd_smi_version_output"], "resource AMD SMI version output", maximum=8192
    )
    if (
        _integer(tools["systemd_major"], "systemd major") != 255
        or not systemd_line.startswith("systemd 255 ")
        or any(
            tools[name] != expected for name, expected in RESOURCE_TOOL_VERSIONS.items()
        )
        or any(
            expected not in version_output
            for expected in RESOURCE_TOOL_VERSIONS.values()
        )
    ):
        fail("resource systemd, AMD SMI, or ROCm tool identity differs")
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
    if (
        probes["cgroup_fs_type"] != "cgroup2fs"
        or probes["kfd_proc_present"] is not True
        or probes["gpu_index"] != 2
        or probes["gpu_bdf"] != "0000:47:00.0"
        or probes["gpu_uuid"] != "a8ff7551-0000-1000-80e9-ddefa2d60f55"
        or probes["kfd_gpu_id"] != 51545
    ):
        fail("resource physical probe identity differs")


def _pids(value: Any, label: str) -> list[int]:
    result = [
        _integer(item, f"{label}[{index}]", minimum=1)
        for index, item in enumerate(_array(value, label))
    ]
    if result != sorted(set(result)):
        fail(f"{label} is not ascending and unique")
    return result


def _process(value: Any, label: str) -> dict[str, Any]:
    process = _object(value, PROCESS_FIELDS, label)
    pid = _integer(process["pid"], f"{label}.pid", minimum=1)
    _integer(process["ppid"], f"{label}.ppid", minimum=1)
    exe = _text(process["exe"], f"{label}.exe")
    if not PurePosixPath(exe).is_absolute():
        fail(f"{label}.exe is not absolute")
    before = _integer(
        process["starttime_ticks_before"], f"{label}.starttime before", minimum=1
    )
    after = _integer(
        process["starttime_ticks_after"], f"{label}.starttime after", minimum=1
    )
    if before != after:
        fail(f"{label} starttime changed")
    rss_kb = _integer(process["vmrss_kb"], f"{label}.rss kB")
    if _integer(process["vmrss_bytes"], f"{label}.rss bytes") != rss_kb * 1024:
        fail(f"{label} RSS byte conversion differs")
    _integer(process["threads"], f"{label}.threads", minimum=1)
    _integer(process["fd_count"], f"{label}.fds")
    _pids(process["children"], f"{label}.children")
    if pid <= 0:
        fail(f"{label} PID differs")
    return process


def _resource_identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    gateway = cast(Mapping[str, Any], record["gateway"])
    worker = cast(Mapping[str, Any], record["worker"])
    systemd = cast(Mapping[str, Any], record["systemd"])
    return (
        systemd["control_group_before"],
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
        fail("resource sample state-machine position differs")
    settle = _integer(
        item["idle_settle_started_monotonic_ns"], "resource settle timestamp"
    )
    sampled = _integer(item["sample_monotonic_ns"], "resource sample timestamp")
    if phase == "baseline":
        for name in (
            "request_id",
            "release_outcome",
            "release_observed_monotonic_ns",
            "reset_complete",
        ):
            if item[name] is not None:
                fail("resource baseline contains request release data")
    else:
        _text(item["request_id"], "resource request ID", maximum=256)
        release_outcome = _text(
            item["release_outcome"], "resource release outcome", maximum=32
        )
        if (
            release_outcome not in {"stop", "length", "cancelled"}
            or item["reset_complete"] is not True
        ):
            fail("resource post-release outcome differs")
        released = _integer(
            item["release_observed_monotonic_ns"], "resource release timestamp"
        )
        if settle < released:
            fail("resource settle begins before release")
    systemd = _object(item["systemd"], SYSTEMD_FIELDS, "resource systemd sample")
    group_before = _text(systemd["control_group_before"], "resource control group")
    group_after = _text(systemd["control_group_after"], "resource control group")
    if (
        group_before != group_after
        or not PurePosixPath(group_before).is_absolute()
        or ".." in PurePosixPath(group_before).parts
    ):
        fail("resource control group changed or is unsafe")
    main_before = _integer(systemd["main_pid_before"], "resource main PID", minimum=1)
    if systemd["main_pid_after"] != main_before:
        fail("resource main PID changed during a sample")
    host = _object(item["host"], {"memory_current_bytes"}, "resource host sample")
    _integer(host["memory_current_bytes"], "resource MemoryCurrent")
    gateway = _process(item["gateway"], "resource gateway")
    worker = _process(item["worker"], "resource worker")
    if (
        gateway["pid"] != main_before
        or worker["ppid"] != gateway["pid"]
        or worker["pid"] not in gateway["children"]
        or PurePosixPath(worker["exe"]).name != "ullm-sq8-worker"
    ):
        fail("resource gateway/worker relationship differs")
    gpu = _object(item["gpu"], GPU_FIELDS, "resource GPU sample")
    if (
        gpu["index"] != 2
        or gpu["bdf"] != "0000:47:00.0"
        or gpu["uuid"] != "a8ff7551-0000-1000-80e9-ddefa2d60f55"
        or gpu["kfd_gpu_id"] != 51545
        or gpu["process_record_count"] != 1
        or gpu["worker_pid"] != worker["pid"]
    ):
        fail("resource GPU process identity differs")
    memory = _object(gpu["mem_usage"], {"value", "unit"}, "resource VRAM")
    vram = _integer(memory["value"], "resource VRAM bytes", minimum=1)
    if memory["unit"] != "B" or gpu["kfd_vram_bytes"] != vram:
        fail("resource VRAM sources differ")
    if _pids(gpu["unrelated_process_pids"], "resource unrelated GPU PIDs"):
        fail("resource GPU contains an unrelated process")
    if sample_index == 0:
        if sampled - settle < 5_000_000_000:
            fail("resource idle settle is shorter than five seconds")
    return item


def _gpu_metric(record: Any, expected: tuple[str, str]) -> None:
    segment, boundary = expected
    item = _object(record, GPU_METRIC_FIELDS, "resource GPU metric")
    expected_file = f"amd-smi-metric-{segment}-{boundary}.json"
    if (
        item["schema_version"] != RESOURCE_INPUT_SCHEMA
        or item["record_type"] != "gpu_metric"
        or item["segment"] != segment
        or item["boundary"] != boundary
        or item["gpu_index"] != 2
        or item["raw_output_file"] != expected_file
    ):
        fail("resource GPU metric state-machine position differs")
    _integer(item["captured_monotonic_ns"], "GPU metric timestamp")
    _sha256(item["raw_output_sha256"], "GPU metric SHA-256")


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
    sample_times: tuple[int, ...]
    identity: tuple[Any, ...]
    metrics: dict[str, Fraction]
    stable_counts: dict[str, Fraction]


def _resource_point(samples: Sequence[dict[str, Any]]) -> _ResourcePoint:
    if len(samples) != 5:
        fail("resource point does not contain five samples")
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
            fail("resource point stable fields changed")
        if _resource_identity(sample) != identity:
            fail("resource point process identity changed")
        current = cast(int, sample["sample_monotonic_ns"])
        if times and current - times[-1] < 1_000_000_000:
            fail("resource point sample interval is shorter than one second")
        times.append(current)
    return _ResourcePoint(
        segment=cast(str, first["segment"]),
        phase=cast(str, first["phase"]),
        request_index=cast(int | None, first["request_index"]),
        request_id=cast(str | None, first["request_id"]),
        sample_times=tuple(times),
        identity=identity,
        metrics={
            "memory_current_bytes": _median(
                cast(int, item["host"]["memory_current_bytes"]) for item in samples
            ),
            "process_vram_bytes": _median(
                cast(int, item["gpu"]["mem_usage"]["value"]) for item in samples
            ),
            "gateway_rss_bytes": _median(
                cast(int, item["gateway"]["vmrss_bytes"]) for item in samples
            ),
            "worker_rss_bytes": _median(
                cast(int, item["worker"]["vmrss_bytes"]) for item in samples
            ),
        },
        stable_counts={
            "gateway_threads": _median(
                cast(int, item["gateway"]["threads"]) for item in samples
            ),
            "gateway_fds": _median(
                cast(int, item["gateway"]["fd_count"]) for item in samples
            ),
            "gateway_children": _median(
                len(cast(list[Any], item["gateway"]["children"])) for item in samples
            ),
            "worker_threads": _median(
                cast(int, item["worker"]["threads"]) for item in samples
            ),
            "worker_fds": _median(
                cast(int, item["worker"]["fd_count"]) for item in samples
            ),
            "worker_children": _median(
                len(cast(list[Any], item["worker"]["children"])) for item in samples
            ),
        },
    )


def _segment_result(
    baseline: _ResourcePoint, points: Sequence[_ResourcePoint]
) -> dict[str, Any]:
    expected_count = 100 if baseline.segment == "normal" else 20
    if len(points) != expected_count or baseline.phase != "baseline":
        fail("resource segment point count differs")
    if [point.request_index for point in points] != list(range(1, expected_count + 1)):
        fail("resource segment request order differs")
    for point in points:
        if point.segment != baseline.segment or point.phase != "post_release":
            fail("resource segment point phase differs")
        if point.identity != baseline.identity:
            fail("resource process identity changed within a segment")
        for name in STABLE_COUNTS:
            if point.stable_counts[name] != baseline.stable_counts[name]:
                fail(f"resource {baseline.segment} {name} did not return to baseline")
    final = points[-1]
    slopes = {
        name: _theil_sen([point.metrics[name] for point in points])
        for name in RESOURCE_METRICS
    }
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
            name: _fraction_json(final.metrics[name] - baseline.metrics[name])
            for name in RESOURCE_METRICS
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


def analyze_soak_resources(
    path: Path, *, forbidden_values: tuple[bytes, ...] = ()
) -> dict[str, Any]:
    """Stream and independently derive the two resource segment summaries."""

    forbidden = _forbidden(forbidden_values)
    lines = _resource_lines(path, forbidden)
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
    sample_count = 0
    metric_count = 0
    record_count = 0
    for line_number, record in lines:
        record_count += 1
        try:
            kind, position = next(expected)
        except StopIteration:
            fail(f"resource line {line_number} is an extra record")
        if kind == "metric":
            _gpu_metric(record, cast(tuple[str, str], position))
            metric_count += 1
            continue
        sample = _resource_sample(
            record, cast(tuple[str, str, int | None, int], position)
        )
        sample_count += 1
        groups.append(sample)
        if len(groups) != 5:
            continue
        point = _resource_point(groups)
        groups = []
        existing_identity = identities.setdefault(point.segment, point.identity)
        if point.identity != existing_identity:
            fail("resource identity changed inside a segment")
        if point.phase == "baseline":
            if point.segment in baselines:
                fail("resource baseline is duplicated")
            baselines[point.segment] = point
        else:
            if point.request_id is None or point.request_id in request_ids:
                fail("resource request ID is absent or duplicated")
            request_id = point.request_id
            assert request_id is not None
            request_ids.add(request_id)
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
    return {
        "resource_sample_count": 610,
        "gpu_metric_count": 4,
        "segments": {
            segment: _segment_result(baselines[segment], points[segment])
            for segment in ("normal", "restart")
        },
    }


@dataclasses.dataclass(frozen=True)
class FullCampaignViews:
    sampling_results: dict[str, Any]
    cancel_results: dict[str, Any]
    prefill_latency_results: dict[str, Any]
    api_contract_results: dict[str, Any]
    openwebui_smoke: dict[str, Any]
    soak_results: dict[str, Any]

    def documents(self) -> dict[str, dict[str, Any]]:
        return {
            "sampling-results.json": copy.deepcopy(self.sampling_results),
            "cancel-results.json": copy.deepcopy(self.cancel_results),
            "prefill-latency-results.json": copy.deepcopy(self.prefill_latency_results),
            "api-contract-results.json": copy.deepcopy(self.api_contract_results),
            "openwebui-smoke.json": copy.deepcopy(self.openwebui_smoke),
            "soak-results.json": copy.deepcopy(self.soak_results),
        }

    def serialized(
        self, *, forbidden_values: tuple[bytes, ...] = ()
    ) -> dict[str, bytes]:
        return {
            name: canonical_json_bytes(value, forbidden_values=forbidden_values)
            for name, value in self.documents().items()
        }


def build_full_campaign_views(
    api_contract: Mapping[str, Any],
    combined: Mapping[str, Any],
    direct_cancel: Mapping[str, Any],
    stop: Mapping[str, Any],
    failure: Mapping[str, Any],
    latency: Mapping[str, Any],
    sampling_cases: Sequence[Mapping[str, Any]],
    resource_raw_path: Path,
    *,
    forbidden_values: tuple[bytes, ...] = (),
) -> FullCampaignViews:
    """Build and canonicalize every final producer view without validator imports."""

    forbidden = _forbidden(forbidden_values)
    resource = analyze_soak_resources(resource_raw_path, forbidden_values=forbidden)
    soak = {
        "schema_version": SOAK_RESULTS_SCHEMA,
        "browser": {
            "chat_count": 20,
            "cases": project_browser_soak(combined),
        },
        **resource,
    }
    result = FullCampaignViews(
        sampling_results=project_sampling(sampling_cases),
        cancel_results=project_cancellation(direct_cancel, stop),
        prefill_latency_results=project_latency(latency),
        api_contract_results=project_api_contract(api_contract),
        openwebui_smoke=project_openwebui_smoke(combined, failure),
        soak_results=soak,
    )
    serialized = result.serialized(forbidden_values=forbidden)
    if tuple(serialized) != VIEW_FILENAMES or any(
        not raw.endswith(b"\n") for raw in serialized.values()
    ):
        fail("full campaign view filename or framing contract differs")
    return result


__all__ = [
    "API_RESULTS_SCHEMA",
    "CANCEL_RESULTS_SCHEMA",
    "FullCampaignViewError",
    "FullCampaignViews",
    "LATENCY_RESULTS_SCHEMA",
    "OPENWEBUI_SMOKE_SCHEMA",
    "SAMPLING_RESULTS_SCHEMA",
    "SOAK_RESULTS_SCHEMA",
    "VIEW_FILENAMES",
    "analyze_soak_resources",
    "build_full_campaign_views",
    "canonical_json_bytes",
    "project_api_contract",
    "project_browser_soak",
    "project_cancellation",
    "project_latency",
    "project_openwebui_smoke",
    "project_sampling",
]
