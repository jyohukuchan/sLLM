#!/usr/bin/env python3
"""Independently validate the phase-1 SQ8 OpenWebUI release evidence.

Phase 1 validates the immutable bundle, lifecycle journal, and complete resource
measurement contract.  It deliberately does not publish release-validation.json
or claim the browser, cancellation, API, and latency gates are complete.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import stat
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Iterator


SESSION_SCHEMA = "ullm.sq8.openwebui_release.raw.v1"
RESOURCE_SCHEMA = "ullm.sq8.release_measurement.raw.v1"
LIFECYCLE_SCHEMA = "ullm.gateway.lifecycle.v1"
MATRIX_SCHEMA = "ullm.sq8.openwebui_release.matrix.v1"
PHASE1_REPORT_SCHEMA = "ullm.sq8.openwebui_release.validation.phase1.v1"

SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
MAX_JSON_BYTES = 16 * 1024 * 1024
U64_MAX = (1 << 64) - 1

FIXTURE_IDS = (
    "exact-p0032",
    "exact-p0128",
    "exact-p0512",
    "exact-p2048",
    "exact-p3584",
)
CANCEL_PHASES = (
    "after_started_before_progress",
    "prefill_after_128",
    "prefill_after_2048",
    "decode_after_first_content",
    "openwebui_stop_after_visible_content",
)
PHASES = {
    "preflight",
    "api_contract",
    "openwebui",
    "cancellation",
    "resource_normal",
    "post_header_failure",
    "resource_restart",
    "latency",
    "final",
}

SCHEDULE = {
    "openwebui_chats": 20,
    "cancel_phases": list(CANCEL_PHASES),
    "normal_warmups": 10,
    "normal_requests": 100,
    "sampled_normal_indices": list(range(5, 101, 5)),
    "restart_warmups": 10,
    "restart_requests": 20,
    "ttft_fixture_ids": list(FIXTURE_IDS),
    "latency_warmups_per_case": 2,
    "latency_measured_per_case": 10,
    "decode_warmups": 2,
    "decode_measured": 10,
    "idle_settle_ms": 5000,
    "samples_per_point": 5,
    "sample_interval_ms": 1000,
}
RESOURCE_SCHEDULE = {
    "normal_warmups": 10,
    "normal_requests": 100,
    "restart_warmups": 10,
    "restart_requests": 20,
    "idle_settle_ms": 5000,
    "samples_per_point": 5,
    "sample_interval_ms": 1000,
}
THRESHOLDS = {
    "ttft_seconds_maximum": {
        "exact-p0032": {"p50": Decimal("2.5"), "p95": 3},
        "exact-p0128": {"p50": 4, "p95": 5},
        "exact-p0512": {"p50": 10, "p95": 12},
        "exact-p2048": {"p50": 30, "p95": 35},
        "exact-p3584": {"p50": 50, "p95": 60},
    },
    "decode_p50_tokens_per_second_minimum": 15,
    "decode_p95_inter_content_seconds_maximum": Decimal("0.1"),
    "cancel_release_max_ns": 5_000_000_000,
    "final_delta_max_bytes": 67_108_864,
    "theil_sen_max_bytes_per_request": 262_144,
}

COMMANDS = {
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

EXPECTED_ROLES = {
    "environment.json": "environment",
    "model-identity.json": "model_identity",
    "raw-session-results.jsonl": "session_raw",
    "soak-resources.raw.jsonl": "resource_raw",
    "service-journal.raw.jsonl": "service_journal_raw",
    "amd-smi-metric-normal-before.json": "gpu_metric_raw",
    "amd-smi-metric-normal-after.json": "gpu_metric_raw",
    "amd-smi-metric-restart-before.json": "gpu_metric_raw",
    "amd-smi-metric-restart-after.json": "gpu_metric_raw",
    "sampling-results.json": "derived_view",
    "cancel-results.json": "derived_view",
    "prefill-latency-results.json": "derived_view",
    "api-contract-results.json": "derived_view",
    "openwebui-smoke.json": "derived_view",
    "soak-results.json": "derived_view",
    "browser/openwebui-stop-before.png": "browser_screenshot",
    "browser/post-header-failure.png": "browser_screenshot",
}
MATRIX_EXCLUDED = {
    "release-matrix.json",
    "release-validation.json",
    "summary.md",
    "SHA256SUMS",
}
BUNDLE_FILES = set(EXPECTED_ROLES) | {
    "release-matrix.json",
    "summary.md",
    "SHA256SUMS",
}

COMMON_SESSION_FIELDS = {
    "schema_version",
    "record_type",
    "sequence",
    "phase",
    "case_id",
}
SESSION_FIELDS = {
    "header": {
        "run_id",
        "started_utc",
        "clock",
        "boot_id",
        "identities",
        "input_files",
        "schedule",
        "thresholds",
    },
    "http_request": {
        "request_index",
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
    "http_response_start": {
        "request_key",
        "status",
        "headers",
        "observed_monotonic_ns",
    },
    "http_body_chunk": {
        "request_key",
        "chunk_index",
        "body_base64",
        "body_sha256",
        "body_bytes",
        "observed_monotonic_ns",
    },
    "http_response_end": {
        "request_key",
        "outcome",
        "error",
        "body_bytes",
        "body_sha256",
        "observed_monotonic_ns",
    },
    "gateway_event": {
        "journal_cursor",
        "journal_monotonic_usec",
        "journal_pid",
        "message",
        "message_sha256",
        "event",
    },
    "browser_action": {
        "browser_case",
        "action_index",
        "action",
        "selector",
        "input_sha256",
        "started_monotonic_ns",
        "completed_monotonic_ns",
        "result",
        "screenshot_file",
        "screenshot_sha256",
    },
    "lifecycle_probe": {
        "probe",
        "observed_monotonic_ns",
        "service_active",
        "ready_http_status",
        "control_group",
        "gateway_pid",
        "gateway_starttime_ticks",
        "worker_pid",
        "worker_starttime_ticks",
        "n_restarts",
    },
    "fault_injection": {
        "injection",
        "target_pid",
        "target_starttime_ticks",
        "signal",
        "command",
        "started_monotonic_ns",
        "completed_monotonic_ns",
    },
    "run_end": {
        "completed_utc",
        "completed_monotonic_ns",
        "final_git_commit",
        "final_git_status_raw",
        "final_git_status_sha256",
        "record_counts",
        "final_journal_cursor",
    },
}

LIFECYCLE_FIELDS = {
    "request_admitted": {
        "request_id",
        "completion_id",
        "stream",
        "prompt_tokens",
        "max_completion_tokens",
    },
    "request_started": {
        "request_id",
        "completion_id",
        "stream",
        "prompt_tokens",
        "admit_to_start_ns",
    },
    "request_progress": {
        "request_id",
        "completion_id",
        "phase",
        "processed_prompt_tokens",
        "prompt_tokens",
    },
    "request_first_token": {
        "request_id",
        "completion_id",
        "stream",
        "completion_tokens",
    },
    "request_cancel_requested": {
        "request_id",
        "completion_id",
        "stream",
        "reason",
        "admit_to_cancel_ns",
    },
    "request_released": {
        "request_id",
        "completion_id",
        "stream",
        "outcome",
        "cancel_reason",
        "prompt_tokens",
        "completion_tokens",
        "reset_complete",
        "admit_to_start_ns",
        "start_to_release_ns",
        "admit_to_release_ns",
    },
    "worker_fatal": {"request_id", "completion_id", "reason", "admit_to_fatal_ns"},
}

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
HOST_FIELDS = {"memory_current_bytes"}
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


class ValidationError(ValueError):
    pass


def fail(message: str) -> None:
    raise ValidationError(message)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> None:
    fail(f"JSON contains a non-finite numeric constant: {value}")


def _validate_unicode(value: Any, label: str) -> None:
    if isinstance(value, str):
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeError as error:
            fail(f"{label} contains an invalid Unicode scalar: {error}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_unicode(item, f"{label}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_unicode(key, f"{label} key")
            _validate_unicode(item, f"{label}.{key}")


def decode_json_bytes(
    raw: bytes,
    label: str,
    *,
    allow_outer_whitespace: bool = False,
    require_object: bool = True,
) -> Any:
    if not raw or len(raw) > MAX_JSON_BYTES:
        fail(f"{label} has an invalid size")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        fail(f"{label} is not strict UTF-8: {error}")
    if not allow_outer_whitespace and (
        not text.startswith("{") or not text.endswith("}")
    ):
        fail(f"{label} must contain exactly one JSON object without outer whitespace")
    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_float=Decimal,
            parse_constant=reject_json_constant,
        )
    except ValidationError:
        raise
    except (json.JSONDecodeError, ValueError) as error:
        fail(f"failed to decode {label}: {error}")
    _validate_unicode(value, label)
    if require_object and type(value) is not dict:
        fail(f"{label} must be an object")
    return value


def read_json(path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_JSON_BYTES:
            fail(f"{label} has an invalid size: {size}")
        raw = path.read_bytes()
    except OSError as error:
        fail(f"failed to read {label}: {error}")
    return decode_json_bytes(raw, label, allow_outer_whitespace=True)


def validate_json_document(path: Path, label: str) -> None:
    regular_file(path, label)
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_JSON_BYTES:
            fail(f"{label} has an invalid size: {size}")
        raw = path.read_bytes()
    except OSError as error:
        fail(f"failed to read {label}: {error}")
    decode_json_bytes(
        raw,
        label,
        allow_outer_whitespace=True,
        require_object=False,
    )


def iter_jsonl(path: Path, label: str) -> Iterator[tuple[int, dict[str, Any]]]:
    regular_file(path, label)
    try:
        handle: BinaryIO
        with path.open("rb") as handle:
            line_number = 0
            while True:
                raw = handle.readline(MAX_JSON_BYTES + 1)
                if not raw:
                    break
                line_number += 1
                if len(raw) > MAX_JSON_BYTES:
                    fail(f"{label} line {line_number} exceeds the size limit")
                if not raw.endswith(b"\n"):
                    fail(f"{label} line {line_number} is not LF-terminated")
                raw = raw[:-1]
                if raw.endswith(b"\r"):
                    fail(f"{label} line {line_number} uses CRLF")
                yield line_number, decode_json_bytes(raw, f"{label} line {line_number}")
            if line_number == 0:
                fail(f"{label} is empty")
    except ValidationError:
        raise
    except OSError as error:
        fail(f"failed to read {label}: {error}")


def exact_fields(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict:
        fail(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        fail(
            f"{label} field set differs: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )
    return value


def integer(value: Any, label: str, minimum: int = 0, maximum: int = U64_MAX) -> int:
    if type(value) is not int:
        fail(f"{label} must be an integer")
    if value < minimum or value > maximum:
        fail(f"{label} is outside {minimum}..={maximum}")
    return value


def boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        fail(f"{label} must be a boolean")
    return value


def string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        fail(f"{label} must be {'a non-empty ' if nonempty else 'a '}string")
    return value


def nullable_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return string(value, label)


def sha256_value(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-256")
    return value


def git_commit(value: Any, label: str) -> str:
    if type(value) is not str or GIT_COMMIT_RE.fullmatch(value) is None:
        fail(f"{label} must be a lowercase 40-hex Git commit")
    return value


def json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            json_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def reject_key_recursive(value: Any, forbidden: str, label: str) -> None:
    if type(value) is dict:
        if forbidden in value:
            fail(f"{label} contains forbidden key {forbidden!r}")
        for key, item in value.items():
            reject_key_recursive(item, forbidden, f"{label}.{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            reject_key_recursive(item, forbidden, f"{label}[{index}]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        fail(f"failed to hash {path}: {error}")
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def regular_file(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        fail(f"failed to stat {label}: {error}")
    if not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} must be a regular non-symlink file")
    return path


def _absolute_without_resolution(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def safe_bundle_root(path: Path) -> Path:
    absolute = _absolute_without_resolution(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as error:
            fail(f"failed to stat bundle path component {current}: {error}")
        if stat.S_ISLNK(metadata.st_mode):
            fail(f"bundle path contains a symlink component: {current}")
    if not absolute.is_dir():
        fail("bundle root must be a directory")
    return absolute


def safe_relative_file(root: Path, relative: str, label: str) -> Path:
    if type(relative) is not str or not relative or "\\" in relative:
        fail(f"{label} is not a safe relative path")
    pure = PurePosixPath(relative)
    lexical_parts = relative.split("/")
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in lexical_parts):
        fail(f"{label} is not a safe relative path")
    current = root
    for part in pure.parts:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as error:
            fail(f"failed to stat {label}: {error}")
        if stat.S_ISLNK(metadata.st_mode):
            fail(f"{label} contains a symlink")
    return regular_file(current, label)


def median(values: Iterable[int | Fraction]) -> Fraction:
    converted: list[Fraction] = []
    for value in values:
        if type(value) is int:
            converted.append(Fraction(value))
        elif type(value) is Fraction:
            converted.append(value)
        else:
            fail("median input contains a non-exact or non-finite value")
    ordered = sorted(converted)
    if not ordered:
        fail("median input must not be empty")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def percentile(values: Iterable[int | Fraction], probability: Fraction) -> Fraction:
    converted: list[Fraction] = []
    for value in values:
        if type(value) is int:
            converted.append(Fraction(value))
        elif type(value) is Fraction:
            converted.append(value)
        else:
            fail("percentile input contains a non-exact or non-finite value")
    ordered = sorted(converted)
    if (
        not ordered
        or type(probability) is not Fraction
        or probability < 0
        or probability > 1
    ):
        fail("percentile input or probability is invalid")
    rank = Fraction(len(ordered) - 1) * probability
    lower = rank.numerator // rank.denominator
    upper = lower if rank.denominator == 1 else lower + 1
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (rank - lower) * (ordered[upper] - ordered[lower])


def theil_sen(values: list[Fraction]) -> Fraction:
    if len(values) < 2:
        fail("Theil-Sen input must contain at least two points")
    if any(type(value) is not Fraction for value in values):
        fail("Theil-Sen input contains a non-exact or non-finite value")
    slopes = [
        (values[j] - values[i]) / (j - i)
        for i in range(len(values))
        for j in range(i + 1, len(values))
    ]
    return median(slopes)


def fraction_json(value: Fraction) -> int | dict[str, int]:
    if value.denominator == 1:
        return value.numerator
    return {"numerator": value.numerator, "denominator": value.denominator}


def decode_base64(value: Any, label: str) -> bytes:
    text = string(value, label, nonempty=False)
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as error:
        fail(f"{label} is not canonical base64: {error}")


def validate_schedule(value: Any, label: str) -> dict[str, Any]:
    expected = SCHEDULE
    exact_fields(value, set(expected), label)
    if not json_equal(value, expected):
        fail(f"{label} differs from the frozen release schedule")
    return value


def validate_thresholds(value: Any, label: str) -> dict[str, Any]:
    exact_fields(value, set(THRESHOLDS), label)
    ttft = exact_fields(
        value["ttft_seconds_maximum"], set(FIXTURE_IDS), f"{label}.ttft_seconds_maximum"
    )
    for fixture_id in FIXTURE_IDS:
        exact_fields(
            ttft[fixture_id],
            {"p50", "p95"},
            f"{label}.ttft_seconds_maximum.{fixture_id}",
        )
    if not json_equal(value, THRESHOLDS):
        fail(f"{label} differs from the frozen release thresholds")
    return value


@dataclass(frozen=True)
class MatrixData:
    run_id: str
    schedule: dict[str, Any]
    thresholds: dict[str, Any]


def validate_bundle_layout(root: Path) -> None:
    actual_files: set[str] = set()
    saw_browser = False
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.name == "browser":
                    if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                        fail(
                            "bundle browser entry must be a regular non-symlink directory"
                        )
                    saw_browser = True
                    with os.scandir(entry.path) as browser_entries:
                        for browser_entry in browser_entries:
                            relative = f"browser/{browser_entry.name}"
                            if browser_entry.is_symlink() or not browser_entry.is_file(
                                follow_symlinks=False
                            ):
                                fail(
                                    f"bundle contains a non-regular file or symlink: {relative}"
                                )
                            if relative not in BUNDLE_FILES:
                                fail(
                                    f"bundle contains an extra evidence file: {relative}"
                                )
                            actual_files.add(relative)
                    continue
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    fail(f"bundle contains a non-regular file or symlink: {entry.name}")
                if entry.name not in BUNDLE_FILES:
                    fail(f"bundle contains an extra evidence file: {entry.name}")
                actual_files.add(entry.name)
    except ValidationError:
        raise
    except OSError as error:
        fail(f"failed to enumerate bundle layout: {error}")
    if not saw_browser:
        fail("bundle lacks the browser evidence directory")
    if actual_files != BUNDLE_FILES:
        fail(
            f"bundle file set differs: missing={sorted(BUNDLE_FILES - actual_files)} "
            f"extra={sorted(actual_files - BUNDLE_FILES)}"
        )
    if (root / "release-validation.json").exists() or (
        root / "release-validation.json"
    ).is_symlink():
        fail("release-validation.json must be absent before validation")


def validate_sha256sums(root: Path) -> str:
    path = safe_relative_file(root, "SHA256SUMS", "SHA256SUMS")
    try:
        raw = path.read_bytes()
        text = raw.decode("ascii", errors="strict")
    except (OSError, UnicodeError) as error:
        fail(f"failed to read SHA256SUMS: {error}")
    if not text or not text.endswith("\n") or "\r" in text:
        fail("SHA256SUMS must be non-empty LF-terminated ASCII")
    expected_paths = sorted(
        BUNDLE_FILES - {"SHA256SUMS"}, key=lambda item: item.encode("utf-8")
    )
    lines = text.splitlines()
    if len(lines) != len(expected_paths):
        fail("SHA256SUMS entry count differs")
    observed_paths: list[str] = []
    for index, (line, expected_path) in enumerate(
        zip(lines, expected_paths, strict=True), start=1
    ):
        match = re.fullmatch(r"([0-9a-f]{64})  ([!-~]+)", line)
        if match is None:
            fail(f"SHA256SUMS line {index} is invalid")
        digest, relative = match.groups()
        observed_paths.append(relative)
        if relative != expected_path:
            fail(
                f"SHA256SUMS paths are not exact bytewise ascending paths at line {index}"
            )
        artifact = safe_relative_file(root, relative, f"SHA256SUMS artifact {relative}")
        if sha256_file(artifact) != digest:
            fail(f"SHA256SUMS digest mismatch for {relative}")
    return hashlib.sha256(raw).hexdigest()


def validate_matrix(root: Path) -> MatrixData:
    path = safe_relative_file(root, "release-matrix.json", "release-matrix.json")
    matrix = read_json(path, "release-matrix.json")
    reject_key_recursive(matrix, "passed", "release-matrix.json")
    exact_fields(
        matrix,
        {"schema_version", "run_id", "files", "schedule", "thresholds"},
        "release-matrix.json",
    )
    if matrix["schema_version"] != MATRIX_SCHEMA:
        fail("release-matrix.json schema_version differs")
    run_id = string(matrix["run_id"], "release-matrix.json.run_id")
    files = matrix["files"]
    if type(files) is not list or len(files) != len(EXPECTED_ROLES):
        fail("release-matrix.json.files has the wrong cardinality")
    paths: list[str] = []
    for index, entry in enumerate(files):
        label = f"release-matrix.json.files[{index}]"
        exact_fields(entry, {"role", "path", "bytes", "sha256"}, label)
        relative = string(entry["path"], f"{label}.path")
        if relative not in EXPECTED_ROLES:
            fail(f"{label}.path is not a defined matrix input")
        if entry["role"] != EXPECTED_ROLES[relative]:
            fail(f"{label}.role differs for {relative}")
        size = integer(entry["bytes"], f"{label}.bytes")
        digest = sha256_value(entry["sha256"], f"{label}.sha256")
        artifact = safe_relative_file(root, relative, f"matrix input {relative}")
        if artifact.stat().st_size != size:
            fail(f"matrix size differs for {relative}")
        if sha256_file(artifact) != digest:
            fail(f"matrix SHA-256 differs for {relative}")
        paths.append(relative)
    expected_paths = sorted(EXPECTED_ROLES, key=lambda item: item.encode("utf-8"))
    if paths != expected_paths:
        fail(
            "release-matrix.json.files paths are not exact bytewise ascending unique paths"
        )
    schedule = validate_schedule(matrix["schedule"], "release-matrix.json.schedule")
    thresholds = validate_thresholds(
        matrix["thresholds"], "release-matrix.json.thresholds"
    )
    return MatrixData(run_id=run_id, schedule=schedule, thresholds=thresholds)


@dataclass
class RequestTrace:
    phase: str
    case_id: str
    completion_id: str
    events: list[dict[str, Any]]
    terminal: str | None = None


@dataclass(frozen=True)
class GatewayEvidence:
    cursor: str
    journal_monotonic_usec: int
    journal_pid: int
    message: str
    message_sha256: str
    event: dict[str, Any]


@dataclass
class SessionData:
    run_id: str
    boot_id: str
    schedule: dict[str, Any]
    thresholds: dict[str, Any]
    traces: dict[str, RequestTrace]
    releases_by_phase: dict[str, list[dict[str, Any]]]
    journal_events: dict[str, GatewayEvidence]
    final_journal_cursor: str
    record_counts: Counter[str]


def validate_hash_bound_bytes(
    encoded: Any, byte_count: Any, digest_value: Any, label: str
) -> bytes:
    raw = decode_base64(encoded, f"{label}.body_base64")
    if base64.b64encode(raw).decode("ascii") != encoded:
        fail(f"{label}.body_base64 is not canonical")
    if integer(byte_count, f"{label}.body_bytes") != len(raw):
        fail(f"{label}.body_bytes differs from decoded bytes")
    digest = sha256_value(digest_value, f"{label}.body_sha256")
    if hashlib.sha256(raw).hexdigest() != digest:
        fail(f"{label}.body_sha256 differs")
    return raw


def _validate_lifecycle_common(value: Any, label: str) -> tuple[str, int]:
    if type(value) is not dict:
        fail(f"{label} must be an object")
    event_name = string(value.get("event"), f"{label}.event")
    if event_name not in LIFECYCLE_FIELDS:
        fail(f"{label}.event is unknown")
    exact_fields(
        value,
        {"schema_version", "event", "observed_monotonic_ns"}
        | LIFECYCLE_FIELDS[event_name],
        label,
    )
    if value["schema_version"] != LIFECYCLE_SCHEMA:
        fail(f"{label}.schema_version differs")
    observed = integer(value["observed_monotonic_ns"], f"{label}.observed_monotonic_ns")
    return event_name, observed


def validate_lifecycle(value: Any, label: str) -> dict[str, Any]:
    event_name, _ = _validate_lifecycle_common(value, label)
    request_id_value = value.get("request_id")
    completion_id_value = value.get("completion_id")
    if event_name == "worker_fatal" and request_id_value is None:
        if completion_id_value is not None or value["admit_to_fatal_ns"] is not None:
            fail(f"{label} nullable worker_fatal fields must be null together")
    else:
        string(request_id_value, f"{label}.request_id")
        string(completion_id_value, f"{label}.completion_id")

    if event_name == "request_admitted":
        boolean(value["stream"], f"{label}.stream")
        integer(value["prompt_tokens"], f"{label}.prompt_tokens", minimum=1)
        integer(
            value["max_completion_tokens"], f"{label}.max_completion_tokens", minimum=1
        )
    elif event_name == "request_started":
        boolean(value["stream"], f"{label}.stream")
        integer(value["prompt_tokens"], f"{label}.prompt_tokens", minimum=1)
        integer(value["admit_to_start_ns"], f"{label}.admit_to_start_ns")
    elif event_name == "request_progress":
        string(value["phase"], f"{label}.phase")
        processed = integer(
            value["processed_prompt_tokens"],
            f"{label}.processed_prompt_tokens",
            minimum=1,
        )
        prompt = integer(value["prompt_tokens"], f"{label}.prompt_tokens", minimum=1)
        if processed > prompt:
            fail(f"{label}.processed_prompt_tokens exceeds prompt_tokens")
    elif event_name == "request_first_token":
        boolean(value["stream"], f"{label}.stream")
        if integer(value["completion_tokens"], f"{label}.completion_tokens") != 1:
            fail(f"{label}.completion_tokens must equal one")
    elif event_name == "request_cancel_requested":
        boolean(value["stream"], f"{label}.stream")
        string(value["reason"], f"{label}.reason")
        integer(value["admit_to_cancel_ns"], f"{label}.admit_to_cancel_ns")
    elif event_name == "request_released":
        boolean(value["stream"], f"{label}.stream")
        outcome = string(value["outcome"], f"{label}.outcome")
        if outcome not in {"stop", "length", "cancelled"}:
            fail(f"{label}.outcome is invalid")
        cancel_reason = nullable_string(
            value["cancel_reason"], f"{label}.cancel_reason"
        )
        if (outcome == "cancelled") != (cancel_reason is not None):
            fail(f"{label}.cancel_reason does not match outcome")
        integer(value["prompt_tokens"], f"{label}.prompt_tokens", minimum=1)
        integer(value["completion_tokens"], f"{label}.completion_tokens")
        if boolean(value["reset_complete"], f"{label}.reset_complete") is not True:
            fail(f"{label}.reset_complete must be true")
        admit_to_start = integer(
            value["admit_to_start_ns"], f"{label}.admit_to_start_ns"
        )
        start_to_release = integer(
            value["start_to_release_ns"], f"{label}.start_to_release_ns"
        )
        admit_to_release = integer(
            value["admit_to_release_ns"], f"{label}.admit_to_release_ns"
        )
        if admit_to_release != admit_to_start + start_to_release:
            fail(f"{label}.admit_to_release_ns arithmetic differs")
    elif event_name == "worker_fatal":
        string(value["reason"], f"{label}.reason")
        if request_id_value is not None:
            integer(value["admit_to_fatal_ns"], f"{label}.admit_to_fatal_ns")
    return value


def decode_lifecycle_message(message: str, label: str) -> dict[str, Any]:
    raw = message.encode("utf-8")
    if raw.startswith(b"{"):
        payload = raw
    elif raw.startswith(b"INFO:     {"):
        payload = raw[len(b"INFO:     ") :]
    else:
        fail(f"{label} has a forbidden journal prefix")
    return validate_lifecycle(decode_json_bytes(payload, label), label)


def _validate_header(
    record: dict[str, Any],
    root: Path,
    matrix: MatrixData,
    expected_worker_sha256: str,
) -> tuple[str, str]:
    label = "raw-session header"
    if record["phase"] != "preflight" or record["case_id"] is not None:
        fail(f"{label} phase/case_id differs")
    run_id = string(record["run_id"], f"{label}.run_id")
    if run_id != matrix.run_id:
        fail(f"{label}.run_id differs from release matrix")
    string(record["started_utc"], f"{label}.started_utc")
    if record["clock"] != "python.time.monotonic_ns":
        fail(f"{label}.clock differs")
    boot_id = string(record["boot_id"], f"{label}.boot_id")

    identities = exact_fields(
        record["identities"],
        {
            "environment_file",
            "environment_sha256",
            "model_identity_file",
            "model_identity_sha256",
            "openwebui",
            "docker_network_id",
            "gateway_source_sha256",
            "worker_source_sha256",
            "worker_binary_sha256",
        },
        f"{label}.identities",
    )
    if (
        identities["environment_file"] != "environment.json"
        or identities["model_identity_file"] != "model-identity.json"
    ):
        fail(f"{label}.identities bundle filenames differ")
    for name, digest_key in (
        ("environment.json", "environment_sha256"),
        ("model-identity.json", "model_identity_sha256"),
    ):
        expected = sha256_value(
            identities[digest_key], f"{label}.identities.{digest_key}"
        )
        if sha256_file(safe_relative_file(root, name, name)) != expected:
            fail(f"{label}.identities.{digest_key} differs from {name}")
    openwebui = exact_fields(
        identities["openwebui"],
        {
            "version",
            "source_revision",
            "base_image_digest",
            "base_image_id",
            "derived_image_id",
            "Dockerfile_sha256",
            "patch_sha256",
            "patched_middleware_sha256",
        },
        f"{label}.identities.openwebui",
    )
    for key in (
        "version",
        "source_revision",
        "base_image_digest",
        "base_image_id",
        "derived_image_id",
    ):
        string(openwebui[key], f"{label}.identities.openwebui.{key}")
    for key in ("Dockerfile_sha256", "patch_sha256", "patched_middleware_sha256"):
        sha256_value(openwebui[key], f"{label}.identities.openwebui.{key}")
    string(identities["docker_network_id"], f"{label}.identities.docker_network_id")
    sha256_value(
        identities["gateway_source_sha256"], f"{label}.identities.gateway_source_sha256"
    )
    sha256_value(
        identities["worker_source_sha256"], f"{label}.identities.worker_source_sha256"
    )
    worker_sha = sha256_value(
        identities["worker_binary_sha256"], f"{label}.identities.worker_binary_sha256"
    )
    if worker_sha != expected_worker_sha256:
        fail(f"{label} worker binary differs from the trusted CLI anchor")

    input_files = record["input_files"]
    if type(input_files) is not list:
        fail(f"{label}.input_files must be an array")
    input_paths: list[str] = []
    for index, item in enumerate(input_files):
        item_label = f"{label}.input_files[{index}]"
        exact_fields(item, {"path", "bytes", "sha256"}, item_label)
        relative = string(item["path"], f"{item_label}.path")
        pure = PurePosixPath(relative)
        lexical_parts = relative.split("/")
        if (
            pure.is_absolute()
            or any(part in {"", ".", ".."} for part in lexical_parts)
            or "\\" in relative
        ):
            fail(f"{item_label}.path is unsafe")
        integer(item["bytes"], f"{item_label}.bytes")
        sha256_value(item["sha256"], f"{item_label}.sha256")
        input_paths.append(relative)
    if input_paths != sorted(set(input_paths), key=lambda item: item.encode("utf-8")):
        fail(f"{label}.input_files paths are not bytewise ascending and unique")
    validate_schedule(record["schedule"], f"{label}.schedule")
    validate_thresholds(record["thresholds"], f"{label}.thresholds")
    if not json_equal(record["schedule"], matrix.schedule) or not json_equal(
        record["thresholds"], matrix.thresholds
    ):
        fail(f"{label} schedule/thresholds differ from release matrix")
    return run_id, boot_id


def _validate_http_record(
    record: dict[str, Any],
    label: str,
    requests: dict[str, dict[str, Any]],
    response_started: set[str],
    response_ended: set[str],
    body_states: dict[str, tuple[hashlib._Hash, int, int]],
) -> None:
    record_type = record["record_type"]
    if record_type == "http_request":
        integer(record["request_index"], f"{label}.request_index")
        key = string(record["request_key"], f"{label}.request_key")
        if key in requests:
            fail(f"{label}.request_key is duplicated")
        string(record["method"], f"{label}.method")
        string(record["target"], f"{label}.target")
        headers = exact_fields(
            record["headers"],
            {"content_type", "content_length", "authorization_mode"},
            f"{label}.headers",
        )
        string(headers["content_type"], f"{label}.headers.content_type")
        content_length = integer(
            headers["content_length"], f"{label}.headers.content_length"
        )
        authorization_mode = string(
            headers["authorization_mode"], f"{label}.headers.authorization_mode"
        )
        if authorization_mode not in {"valid_bearer", "invalid_bearer", "missing"}:
            fail(f"{label}.headers.authorization_mode differs")
        raw = validate_hash_bound_bytes(
            record["body_base64"], record["body_bytes"], record["body_sha256"], label
        )
        if content_length != len(raw):
            fail(f"{label}.headers.content_length differs")
        connect = integer(
            record["connect_completed_monotonic_ns"],
            f"{label}.connect_completed_monotonic_ns",
        )
        started = integer(
            record["write_started_monotonic_ns"], f"{label}.write_started_monotonic_ns"
        )
        sent = integer(
            record["last_body_byte_sent_monotonic_ns"],
            f"{label}.last_body_byte_sent_monotonic_ns",
        )
        if not connect <= started <= sent:
            fail(f"{label} request timing order differs")
        # Retain only correlation metadata; request bodies may be large.
        requests[key] = {"request_index": record["request_index"]}
        body_states[key] = (hashlib.sha256(), 0, 0)
    elif record_type == "http_response_start":
        key = string(record["request_key"], f"{label}.request_key")
        if key not in requests or key in response_started:
            fail(f"{label} response start has an unknown or duplicated request_key")
        integer(record["status"], f"{label}.status", minimum=100, maximum=599)
        headers = record["headers"]
        if type(headers) is not list:
            fail(f"{label}.headers must be an array")
        for index, pair in enumerate(headers):
            if type(pair) is not list or len(pair) != 2:
                fail(f"{label}.headers[{index}] must be a two-string array")
            string(pair[0], f"{label}.headers[{index}][0]")
            string(pair[1], f"{label}.headers[{index}][1]", nonempty=False)
        integer(record["observed_monotonic_ns"], f"{label}.observed_monotonic_ns")
        response_started.add(key)
    elif record_type == "http_body_chunk":
        key = string(record["request_key"], f"{label}.request_key")
        if key not in response_started or key in response_ended:
            fail(f"{label} chunk has no active response")
        digest, byte_count, next_index = body_states[key]
        if integer(record["chunk_index"], f"{label}.chunk_index") != next_index:
            fail(f"{label}.chunk_index is not contiguous")
        raw = validate_hash_bound_bytes(
            record["body_base64"], record["body_bytes"], record["body_sha256"], label
        )
        digest.update(raw)
        body_states[key] = (digest, byte_count + len(raw), next_index + 1)
        integer(record["observed_monotonic_ns"], f"{label}.observed_monotonic_ns")
    elif record_type == "http_response_end":
        key = string(record["request_key"], f"{label}.request_key")
        if key not in response_started or key in response_ended:
            fail(f"{label} response end has no active response")
        outcome = string(record["outcome"], f"{label}.outcome")
        if outcome not in {"eof", "client_closed", "timeout", "error"}:
            fail(f"{label}.outcome differs")
        error = nullable_string(record["error"], f"{label}.error")
        if (outcome in {"eof", "client_closed"}) != (error is None):
            fail(f"{label}.error does not match outcome")
        digest, byte_count, _ = body_states[key]
        if integer(record["body_bytes"], f"{label}.body_bytes") != byte_count:
            fail(f"{label}.body_bytes differs from chunks")
        if (
            sha256_value(record["body_sha256"], f"{label}.body_sha256")
            != digest.hexdigest()
        ):
            fail(f"{label}.body_sha256 differs from chunks")
        integer(record["observed_monotonic_ns"], f"{label}.observed_monotonic_ns")
        response_ended.add(key)
        del body_states[key]


def _add_lifecycle_event(
    traces: dict[str, RequestTrace],
    completion_ids: dict[str, str],
    phase: str,
    case_id: str,
    event: dict[str, Any],
    label: str,
) -> None:
    name = event["event"]
    request_id = event["request_id"]
    if request_id is None:
        return
    completion_id = event["completion_id"]
    trace = traces.get(request_id)
    if name == "request_admitted":
        if trace is not None or completion_id in completion_ids:
            fail(f"{label} admitted request/completion ID is duplicated")
        trace = RequestTrace(
            phase=phase, case_id=case_id, completion_id=completion_id, events=[]
        )
        traces[request_id] = trace
        completion_ids[completion_id] = request_id
    elif trace is None:
        fail(f"{label} refers to a request before admission")
    assert trace is not None
    if (
        trace.phase != phase
        or trace.case_id != case_id
        or trace.completion_id != completion_id
    ):
        fail(f"{label} request correlation differs")
    if trace.terminal is not None:
        fail(f"{label} appears after terminal event {trace.terminal}")
    previous_time = trace.events[-1]["observed_monotonic_ns"] if trace.events else None
    if previous_time is not None and event["observed_monotonic_ns"] < previous_time:
        fail(f"{label} monotonic event order differs")
    names = [item["event"] for item in trace.events]
    if name == "request_started":
        if names != ["request_admitted"]:
            fail(f"{label} started event order differs")
        admitted = trace.events[0]
        if (
            event["stream"] is not admitted["stream"]
            or event["prompt_tokens"] != admitted["prompt_tokens"]
        ):
            fail(f"{label} started fields differ from admission")
    elif name == "request_progress":
        if (
            "request_started" not in names
            or "request_first_token" in names
            or "request_cancel_requested" in names
        ):
            fail(f"{label} progress event order differs")
        if event["prompt_tokens"] != trace.events[0]["prompt_tokens"]:
            fail(f"{label} progress prompt_tokens differs from admission")
        prior_progress = [
            item["processed_prompt_tokens"]
            for item in trace.events
            if item["event"] == "request_progress"
        ]
        if prior_progress and event["processed_prompt_tokens"] <= prior_progress[-1]:
            fail(f"{label} progress is not strictly increasing")
    elif name == "request_first_token":
        if (
            "request_started" not in names
            or "request_first_token" in names
            or "request_cancel_requested" in names
        ):
            fail(f"{label} first-token event order differs")
        if event["stream"] is not trace.events[0]["stream"]:
            fail(f"{label} first-token stream flag differs from admission")
    elif name == "request_cancel_requested":
        if "request_started" not in names or "request_cancel_requested" in names:
            fail(f"{label} cancel event order differs")
        if event["stream"] is not trace.events[0]["stream"]:
            fail(f"{label} cancel stream flag differs from admission")
    elif name == "request_released":
        if "request_started" not in names or "request_released" in names:
            fail(f"{label} release event order differs")
        admitted = trace.events[0]
        started = next(
            item for item in trace.events if item["event"] == "request_started"
        )
        if (
            event["stream"] is not admitted["stream"]
            or event["prompt_tokens"] != admitted["prompt_tokens"]
        ):
            fail(f"{label} release fields differ from admission")
        if event["admit_to_start_ns"] != started["admit_to_start_ns"]:
            fail(f"{label} release/start duration differs")
        maximum_completion = admitted["max_completion_tokens"]
        if event["completion_tokens"] > maximum_completion:
            fail(f"{label} completion count exceeds admission maximum")
        if (event["completion_tokens"] > 0) != ("request_first_token" in names):
            fail(f"{label} completion count and first-token event differ")
        if (
            event["outcome"] == "length"
            and event["completion_tokens"] != maximum_completion
        ):
            fail(f"{label} length outcome does not reach the admission maximum")
        if event["outcome"] == "cancelled" and "request_cancel_requested" not in names:
            fail(f"{label} cancelled release lacks cancellation event")
        if event["outcome"] != "cancelled" and "request_cancel_requested" in names:
            fail(f"{label} non-cancelled release follows cancellation")
        if event["outcome"] == "cancelled":
            cancel = next(
                item
                for item in trace.events
                if item["event"] == "request_cancel_requested"
            )
            if event["cancel_reason"] != cancel["reason"]:
                fail(f"{label} does not retain the cancellation reason")
        trace.terminal = name
    elif name == "worker_fatal":
        if "request_started" not in names:
            fail(f"{label} active worker_fatal precedes start")
        trace.terminal = name
    trace.events.append(event)


def validate_service_journal(
    root: Path,
    expected: dict[str, GatewayEvidence],
    boot_id: str,
    final_cursor: str,
) -> None:
    remaining = dict(expected)
    final_seen = False
    seen_cursors: set[str] = set()
    last_cursor: str | None = None
    last_monotonic = -1
    for line_number, record in iter_jsonl(
        root / "service-journal.raw.jsonl", "service-journal.raw.jsonl"
    ):
        label = f"service-journal.raw.jsonl line {line_number}"
        for field in (
            "__CURSOR",
            "__MONOTONIC_TIMESTAMP",
            "_BOOT_ID",
            "_PID",
            "_SYSTEMD_UNIT",
            "PRIORITY",
            "MESSAGE",
        ):
            if field not in record:
                fail(f"{label} lacks required field {field}")
        cursor = string(record["__CURSOR"], f"{label}.__CURSOR")
        if cursor in seen_cursors:
            fail(f"{label} journal cursor is duplicated")
        seen_cursors.add(cursor)
        last_cursor = cursor
        if record["_BOOT_ID"] != boot_id:
            fail(f"{label} boot ID differs")
        if record["_SYSTEMD_UNIT"] != "ullm-openai.service":
            fail(f"{label} systemd unit differs")
        monotonic_text = string(
            record["__MONOTONIC_TIMESTAMP"], f"{label}.__MONOTONIC_TIMESTAMP"
        )
        pid_text = string(record["_PID"], f"{label}._PID")
        if not monotonic_text.isdecimal() or not pid_text.isdecimal():
            fail(f"{label} numeric journal fields are invalid")
        monotonic = int(monotonic_text)
        if monotonic < last_monotonic:
            fail(f"{label} journal monotonic timestamps regress")
        last_monotonic = monotonic
        string(record["PRIORITY"], f"{label}.PRIORITY")
        message = string(record["MESSAGE"], f"{label}.MESSAGE", nonempty=False)
        if cursor == final_cursor:
            final_seen = True
        evidence = remaining.pop(cursor, None)
        if evidence is not None:
            if (
                monotonic != evidence.journal_monotonic_usec
                or int(pid_text) != evidence.journal_pid
            ):
                fail(f"{label} copied numeric journal fields differ")
            if (
                message != evidence.message
                or sha256_text(message) != evidence.message_sha256
            ):
                fail(f"{label} copied MESSAGE bytes/hash differ")
        payload_text: str | None = None
        if message.startswith("{"):
            payload_text = message
        elif message.startswith("INFO:     {"):
            payload_text = message[len("INFO:     ") :]
        if payload_text is not None:
            decoded = decode_json_bytes(
                payload_text.encode("utf-8"), f"{label}.MESSAGE"
            )
            if (
                type(decoded) is dict
                and decoded.get("schema_version") == LIFECYCLE_SCHEMA
            ):
                validate_lifecycle(decoded, f"{label}.MESSAGE")
                if cursor not in expected:
                    fail(
                        f"{label} lifecycle message is omitted from raw-session-results.jsonl"
                    )
    if remaining:
        fail(f"service journal lacks {len(remaining)} copied gateway event cursor(s)")
    if not final_seen:
        fail("service journal lacks the run_end final journal cursor")
    if last_cursor != final_cursor:
        fail("service journal is not bounded exactly by the run_end final cursor")


def validate_session(
    root: Path,
    matrix: MatrixData,
    expected_commit: str,
    expected_worker_sha256: str,
) -> SessionData:
    traces: dict[str, RequestTrace] = {}
    completion_ids: dict[str, str] = {}
    releases_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    journal_events: dict[str, GatewayEvidence] = {}
    requests: dict[str, dict[str, Any]] = {}
    response_started: set[str] = set()
    response_ended: set[str] = set()
    body_states: dict[str, tuple[hashlib._Hash, int, int]] = {}
    counts: Counter[str] = Counter()
    run_id: str | None = None
    boot_id: str | None = None
    final_cursor: str | None = None
    declared_counts: dict[str, Any] | None = None
    last_gateway_observed = -1
    saw_run_end = False

    for line_number, record in iter_jsonl(
        root / "raw-session-results.jsonl", "raw-session-results.jsonl"
    ):
        label = f"raw-session-results.jsonl line {line_number}"
        reject_key_recursive(record, "passed", label)
        record_type = record.get("record_type")
        if type(record_type) is not str or record_type not in SESSION_FIELDS:
            fail(f"{label}.record_type is unknown")
        exact_fields(record, COMMON_SESSION_FIELDS | SESSION_FIELDS[record_type], label)
        if record["schema_version"] != SESSION_SCHEMA:
            fail(f"{label}.schema_version differs")
        if integer(record["sequence"], f"{label}.sequence") != sum(counts.values()):
            fail(f"{label}.sequence is not contiguous from zero")
        phase = record["phase"]
        if type(phase) is not str or phase not in PHASES:
            fail(f"{label}.phase is invalid")
        case_id = record["case_id"]
        if record_type in {"header", "run_end"}:
            if case_id is not None:
                fail(f"{label}.case_id must be null")
        else:
            string(case_id, f"{label}.case_id")
        if saw_run_end:
            fail(f"{label} appears after run_end")
        counts[record_type] += 1

        if record_type == "header":
            if line_number != 1 or counts[record_type] != 1:
                fail("raw-session header must be the first and sole header")
            run_id, boot_id = _validate_header(
                record, root, matrix, expected_worker_sha256
            )
        elif run_id is None:
            fail(f"{label} appears before header")
        elif record_type.startswith("http_"):
            _validate_http_record(
                record, label, requests, response_started, response_ended, body_states
            )
        elif record_type == "gateway_event":
            cursor = string(record["journal_cursor"], f"{label}.journal_cursor")
            if cursor in journal_events:
                fail(f"{label}.journal_cursor is duplicated")
            usec = integer(
                record["journal_monotonic_usec"], f"{label}.journal_monotonic_usec"
            )
            journal_pid = integer(
                record["journal_pid"], f"{label}.journal_pid", minimum=1
            )
            message = string(record["message"], f"{label}.message", nonempty=False)
            message_digest = sha256_value(
                record["message_sha256"], f"{label}.message_sha256"
            )
            if sha256_text(message) != message_digest:
                fail(f"{label}.message_sha256 differs")
            event = decode_lifecycle_message(message, f"{label}.message")
            if not json_equal(event, record["event"]):
                fail(f"{label}.event differs from exactly decoded MESSAGE")
            name, observed = _validate_lifecycle_common(event, f"{label}.event")
            if usec < observed // 1000:
                fail(f"{label} journal observation precedes the lifecycle timestamp")
            if observed < last_gateway_observed:
                fail(f"{label} gateway events are not in monotonic order")
            last_gateway_observed = observed
            _add_lifecycle_event(traces, completion_ids, phase, case_id, event, label)
            if name == "request_released":
                releases_by_phase[phase].append(event)
            journal_events[cursor] = GatewayEvidence(
                cursor, usec, journal_pid, message, message_digest, event
            )
        elif record_type == "browser_action":
            string(record["browser_case"], f"{label}.browser_case")
            integer(record["action_index"], f"{label}.action_index")
            action = string(record["action"], f"{label}.action")
            if action not in {
                "navigate",
                "select_model",
                "submit_chat",
                "wait_visible",
                "click_stop",
                "wait_failed",
                "wait_ready",
            }:
                fail(f"{label}.action differs")
            nullable_string(record["selector"], f"{label}.selector")
            if record["input_sha256"] is not None:
                sha256_value(record["input_sha256"], f"{label}.input_sha256")
            started = integer(
                record["started_monotonic_ns"], f"{label}.started_monotonic_ns"
            )
            completed = integer(
                record["completed_monotonic_ns"], f"{label}.completed_monotonic_ns"
            )
            if completed < started:
                fail(f"{label} browser timing order differs")
            result = exact_fields(
                record["result"],
                {"visible", "enabled", "text_utf8_bytes", "text_sha256"},
                f"{label}.result",
            )
            for key in ("visible", "enabled"):
                if result[key] is not None:
                    boolean(result[key], f"{label}.result.{key}")
            if result["text_utf8_bytes"] is not None:
                integer(result["text_utf8_bytes"], f"{label}.result.text_utf8_bytes")
                sha256_value(result["text_sha256"], f"{label}.result.text_sha256")
            elif result["text_sha256"] is not None:
                fail(f"{label}.result text fields must be null together")
            screenshot = record["screenshot_file"]
            screenshot_sha = record["screenshot_sha256"]
            if screenshot is None:
                if screenshot_sha is not None:
                    fail(f"{label} screenshot fields must be null together")
            else:
                screenshot_path = string(screenshot, f"{label}.screenshot_file")
                if screenshot_path not in {
                    "browser/openwebui-stop-before.png",
                    "browser/post-header-failure.png",
                }:
                    fail(f"{label}.screenshot_file differs")
                expected_sha = sha256_value(
                    screenshot_sha, f"{label}.screenshot_sha256"
                )
                if (
                    sha256_file(
                        safe_relative_file(root, screenshot_path, screenshot_path)
                    )
                    != expected_sha
                ):
                    fail(f"{label}.screenshot_sha256 differs")
        elif record_type == "lifecycle_probe":
            string(record["probe"], f"{label}.probe")
            integer(record["observed_monotonic_ns"], f"{label}.observed_monotonic_ns")
            boolean(record["service_active"], f"{label}.service_active")
            integer(
                record["ready_http_status"], f"{label}.ready_http_status", maximum=599
            )
            string(record["control_group"], f"{label}.control_group")
            for key in (
                "gateway_pid",
                "gateway_starttime_ticks",
                "worker_pid",
                "worker_starttime_ticks",
            ):
                integer(record[key], f"{label}.{key}", minimum=1)
            integer(record["n_restarts"], f"{label}.n_restarts")
        elif record_type == "fault_injection":
            if (
                record["injection"] != "post_header_worker_kill"
                or record["signal"] != "SIGKILL"
            ):
                fail(f"{label} fault identity differs")
            integer(record["target_pid"], f"{label}.target_pid", minimum=1)
            integer(
                record["target_starttime_ticks"],
                f"{label}.target_starttime_ticks",
                minimum=1,
            )
            string(record["command"], f"{label}.command")
            started = integer(
                record["started_monotonic_ns"], f"{label}.started_monotonic_ns"
            )
            completed = integer(
                record["completed_monotonic_ns"], f"{label}.completed_monotonic_ns"
            )
            if completed < started:
                fail(f"{label} fault timing order differs")
        elif record_type == "run_end":
            saw_run_end = True
            if phase != "final" or counts[record_type] != 1:
                fail(f"{label} run_end placement differs")
            string(record["completed_utc"], f"{label}.completed_utc")
            completed_ns = integer(
                record["completed_monotonic_ns"], f"{label}.completed_monotonic_ns"
            )
            if completed_ns < last_gateway_observed:
                fail(f"{label}.completed_monotonic_ns precedes the final gateway event")
            if (
                git_commit(record["final_git_commit"], f"{label}.final_git_commit")
                != expected_commit
            ):
                fail(f"{label} final commit differs from trusted CLI anchor")
            status = string(
                record["final_git_status_raw"],
                f"{label}.final_git_status_raw",
                nonempty=False,
            )
            if sha256_text(status) != sha256_value(
                record["final_git_status_sha256"], f"{label}.final_git_status_sha256"
            ):
                fail(f"{label}.final_git_status_sha256 differs")
            declared_counts = exact_fields(
                record["record_counts"], set(counts), f"{label}.record_counts"
            )
            final_cursor = string(
                record["final_journal_cursor"], f"{label}.final_journal_cursor"
            )

    if (
        run_id is None
        or boot_id is None
        or not saw_run_end
        or final_cursor is None
        or declared_counts is None
    ):
        fail("raw-session-results.jsonl lacks header or run_end")
    if any(
        type(value) is not int for value in declared_counts.values()
    ) or declared_counts != dict(counts):
        fail("run_end.record_counts differs from independently counted raw records")
    if set(requests) != response_ended or body_states:
        fail(
            "one or more raw HTTP requests lack a complete response start/end correlation"
        )
    for request_id_value, trace in traces.items():
        if trace.terminal is None:
            fail(f"request {request_id_value} lacks a terminal lifecycle event")
    ordered_traces = sorted(
        traces.values(), key=lambda trace: trace.events[0]["observed_monotonic_ns"]
    )
    for prior, following in zip(ordered_traces, ordered_traces[1:]):
        terminal_time = prior.events[-1]["observed_monotonic_ns"]
        following_admission = following.events[0]["observed_monotonic_ns"]
        if following_admission <= terminal_time:
            fail("a request is admitted before the prior lifecycle terminal event")
        prior_release = prior.events[-1]
        if (
            prior.phase == "cancellation"
            and prior_release["event"] == "request_released"
            and prior_release["outcome"] == "cancelled"
        ):
            following_terminal = following.events[-1]
            if (
                following.phase != "cancellation"
                or following_terminal["event"] != "request_released"
                or following_terminal["outcome"] == "cancelled"
            ):
                fail(
                    "a cancellation is not followed by a successful recovery lifecycle"
                )
    validate_service_journal(root, journal_events, boot_id, final_cursor)
    return SessionData(
        run_id=run_id,
        boot_id=boot_id,
        schedule=matrix.schedule,
        thresholds=matrix.thresholds,
        traces=traces,
        releases_by_phase=dict(releases_by_phase),
        journal_events=journal_events,
        final_journal_cursor=final_cursor,
        record_counts=counts,
    )


@dataclass(frozen=True)
class ResourcePoint:
    segment: str
    phase: str
    request_index: int | None
    request_id: str | None
    release_outcome: str | None
    release_observed_monotonic_ns: int | None
    idle_settle_started_monotonic_ns: int
    sample_monotonic_ns: tuple[int, ...]
    host_memory: Fraction
    primary_vram: Fraction
    gateway_rss: Fraction
    worker_rss: Fraction
    gateway_threads: Fraction
    gateway_fds: Fraction
    gateway_children: Fraction
    worker_threads: Fraction
    worker_fds: Fraction
    worker_children: Fraction


@dataclass(frozen=True)
class ResourceResult:
    segments: dict[str, dict[str, Any]]
    sample_count: int
    gpu_metric_count: int


def _expected_resource_records() -> Iterator[
    tuple[str, str, str | None, int | None, int | None]
]:
    yield "gpu_metric", "normal", "before", None, None
    for sample_index in range(5):
        yield "resource_sample", "normal", "baseline", None, sample_index
    for request_index in range(1, 101):
        for sample_index in range(5):
            yield (
                "resource_sample",
                "normal",
                "post_release",
                request_index,
                sample_index,
            )
    yield "gpu_metric", "normal", "after", None, None
    yield "gpu_metric", "restart", "before", None, None
    for sample_index in range(5):
        yield "resource_sample", "restart", "baseline", None, sample_index
    for request_index in range(1, 21):
        for sample_index in range(5):
            yield (
                "resource_sample",
                "restart",
                "post_release",
                request_index,
                sample_index,
            )
    yield "gpu_metric", "restart", "after", None, None


def _validate_resource_header(record: dict[str, Any], label: str) -> None:
    exact_fields(record, RESOURCE_HEADER_FIELDS, label)
    if record["schema_version"] != RESOURCE_SCHEMA or record["record_type"] != "header":
        fail(f"{label} schema/record type differs")
    if record["service_unit"] != "ullm-openai.service":
        fail(f"{label}.service_unit differs")
    exact_fields(record["commands"], set(COMMANDS), f"{label}.commands")
    if not json_equal(record["commands"], COMMANDS):
        fail(f"{label}.commands differs from the frozen commands")
    tools_value = exact_fields(
        record["tools"],
        {
            "systemd_major",
            "systemd_version_line",
            "amd_smi_tool",
            "amd_smi_library",
            "rocm",
            "amd_smi_version_output",
        },
        f"{label}.tools",
    )
    if integer(tools_value["systemd_major"], f"{label}.tools.systemd_major") != 255:
        fail(f"{label}.tools.systemd_major differs")
    version_line = string(
        tools_value["systemd_version_line"], f"{label}.tools.systemd_version_line"
    )
    if not version_line.startswith("systemd 255 "):
        fail(f"{label}.tools.systemd_version_line differs")
    expected_versions = {
        "amd_smi_tool": "26.2.2+e1a6bc5663",
        "amd_smi_library": "26.2.2",
        "rocm": "7.2.1",
    }
    for key, expected in expected_versions.items():
        if tools_value[key] != expected:
            fail(f"{label}.tools.{key} differs")
    version_output = string(
        tools_value["amd_smi_version_output"], f"{label}.tools.amd_smi_version_output"
    )
    for expected in expected_versions.values():
        if expected not in version_output:
            fail(f"{label}.tools.amd_smi_version_output lacks {expected}")
    probes = exact_fields(
        record["probes"],
        {
            "cgroup_fs_type",
            "kfd_proc_present",
            "gpu_index",
            "gpu_bdf",
            "gpu_uuid",
            "kfd_gpu_id",
        },
        f"{label}.probes",
    )
    expected_probes = {
        "cgroup_fs_type": "cgroup2fs",
        "kfd_proc_present": True,
        "gpu_index": 2,
        "gpu_bdf": "0000:47:00.0",
        "gpu_uuid": "a8ff7551-0000-1000-80e9-ddefa2d60f55",
        "kfd_gpu_id": 51545,
    }
    if not json_equal(probes, expected_probes):
        fail(f"{label}.probes differs from the frozen R9700 identity")
    exact_fields(record["schedule"], set(RESOURCE_SCHEDULE), f"{label}.schedule")
    if not json_equal(record["schedule"], RESOURCE_SCHEDULE):
        fail(f"{label}.schedule differs from the frozen resource schedule")


def _ascending_unique_pids(value: Any, label: str) -> list[int]:
    if type(value) is not list:
        fail(f"{label} must be an array")
    result = [
        integer(item, f"{label}[{index}]", minimum=1)
        for index, item in enumerate(value)
    ]
    if result != sorted(set(result)):
        fail(f"{label} must be ascending and unique")
    return result


def _validate_process(value: Any, label: str) -> dict[str, Any]:
    process = exact_fields(value, PROCESS_FIELDS, label)
    integer(process["pid"], f"{label}.pid", minimum=1)
    integer(process["ppid"], f"{label}.ppid", minimum=1)
    exe = string(process["exe"], f"{label}.exe")
    if not exe.startswith("/"):
        fail(f"{label}.exe must be absolute")
    before = integer(
        process["starttime_ticks_before"], f"{label}.starttime_ticks_before", minimum=1
    )
    after = integer(
        process["starttime_ticks_after"], f"{label}.starttime_ticks_after", minimum=1
    )
    if before != after:
        fail(f"{label} starttime changed during sampling")
    rss_kb = integer(process["vmrss_kb"], f"{label}.vmrss_kb")
    rss_bytes = integer(process["vmrss_bytes"], f"{label}.vmrss_bytes")
    if rss_bytes != rss_kb * 1024:
        fail(f"{label}.vmrss_bytes differs from VmRSS kB")
    integer(process["threads"], f"{label}.threads")
    integer(process["fd_count"], f"{label}.fd_count")
    _ascending_unique_pids(process["children"], f"{label}.children")
    return process


def _validate_resource_sample(record: dict[str, Any], label: str) -> dict[str, Any]:
    reject_key_recursive(record, "passed", label)
    exact_fields(record, RESOURCE_SAMPLE_FIELDS, label)
    if (
        record["schema_version"] != RESOURCE_SCHEMA
        or record["record_type"] != "resource_sample"
    ):
        fail(f"{label} schema/record type differs")
    segment = record["segment"]
    phase = record["phase"]
    if (
        type(segment) is not str
        or segment not in {"normal", "restart"}
        or type(phase) is not str
        or phase not in {"baseline", "post_release"}
    ):
        fail(f"{label} segment/phase differs")
    integer(
        record["idle_settle_started_monotonic_ns"],
        f"{label}.idle_settle_started_monotonic_ns",
    )
    integer(record["sample_index"], f"{label}.sample_index", maximum=4)
    integer(record["sample_monotonic_ns"], f"{label}.sample_monotonic_ns")

    if record["phase"] == "baseline":
        for field in (
            "request_index",
            "request_id",
            "release_outcome",
            "release_observed_monotonic_ns",
            "reset_complete",
        ):
            if record[field] is not None:
                fail(f"{label}.{field} must be null for a baseline sample")
    else:
        integer(record["request_index"], f"{label}.request_index", minimum=1)
        string(record["request_id"], f"{label}.request_id")
        release_outcome = string(record["release_outcome"], f"{label}.release_outcome")
        if release_outcome not in {"stop", "length", "cancelled"}:
            fail(f"{label}.release_outcome differs")
        integer(
            record["release_observed_monotonic_ns"],
            f"{label}.release_observed_monotonic_ns",
        )
        if boolean(record["reset_complete"], f"{label}.reset_complete") is not True:
            fail(f"{label}.reset_complete must be true")

    systemd_value = exact_fields(record["systemd"], SYSTEMD_FIELDS, f"{label}.systemd")
    before_group = string(
        systemd_value["control_group_before"], f"{label}.systemd.control_group_before"
    )
    after_group = string(
        systemd_value["control_group_after"], f"{label}.systemd.control_group_after"
    )
    pure_group = PurePosixPath(before_group)
    if (
        not pure_group.is_absolute()
        or ".." in pure_group.parts
        or before_group != after_group
    ):
        fail(f"{label}.systemd control group is unsafe or changed")
    main_before = integer(
        systemd_value["main_pid_before"], f"{label}.systemd.main_pid_before", minimum=1
    )
    main_after = integer(
        systemd_value["main_pid_after"], f"{label}.systemd.main_pid_after", minimum=1
    )
    if main_before != main_after:
        fail(f"{label}.systemd MainPID changed during sampling")
    host = exact_fields(record["host"], HOST_FIELDS, f"{label}.host")
    integer(host["memory_current_bytes"], f"{label}.host.memory_current_bytes")
    gateway = _validate_process(record["gateway"], f"{label}.gateway")
    worker = _validate_process(record["worker"], f"{label}.worker")
    if gateway["pid"] != main_before:
        fail(f"{label} gateway PID differs from systemd MainPID")
    if worker["ppid"] != gateway["pid"] or worker["pid"] not in gateway["children"]:
        fail(f"{label} worker is not a direct gateway child")
    if Path(worker["exe"]).name != "ullm-sq8-worker":
        fail(f"{label} worker executable basename differs")

    gpu = exact_fields(record["gpu"], GPU_FIELDS, f"{label}.gpu")
    if (
        gpu["index"] != 2
        or gpu["bdf"] != "0000:47:00.0"
        or gpu["uuid"] != "a8ff7551-0000-1000-80e9-ddefa2d60f55"
        or gpu["kfd_gpu_id"] != 51545
    ):
        fail(f"{label}.gpu physical identity differs")
    if integer(gpu["process_record_count"], f"{label}.gpu.process_record_count") != 1:
        fail(f"{label}.gpu.process_record_count must equal one")
    if (
        integer(gpu["worker_pid"], f"{label}.gpu.worker_pid", minimum=1)
        != worker["pid"]
    ):
        fail(f"{label}.gpu.worker_pid differs")
    mem_usage = exact_fields(
        gpu["mem_usage"], {"value", "unit"}, f"{label}.gpu.mem_usage"
    )
    primary_vram = integer(mem_usage["value"], f"{label}.gpu.mem_usage.value")
    if mem_usage["unit"] != "B":
        fail(f"{label}.gpu.mem_usage.unit differs")
    if integer(gpu["kfd_vram_bytes"], f"{label}.gpu.kfd_vram_bytes") != primary_vram:
        fail(f"{label} AMD SMI and KFD VRAM differ")
    if _ascending_unique_pids(
        gpu["unrelated_process_pids"], f"{label}.gpu.unrelated_process_pids"
    ):
        fail(f"{label}.gpu.unrelated_process_pids is not empty")
    return record


def _resource_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    gateway = record["gateway"]
    worker = record["worker"]
    return (
        record["systemd"]["control_group_before"],
        gateway["pid"],
        gateway["ppid"],
        gateway["exe"],
        gateway["starttime_ticks_before"],
        worker["pid"],
        worker["ppid"],
        worker["exe"],
        worker["starttime_ticks_before"],
    )


def _point_from_samples(samples: list[dict[str, Any]], label: str) -> ResourcePoint:
    if len(samples) != 5:
        fail(f"{label} must contain exactly five samples")
    first = samples[0]
    stable_fields = {
        "segment",
        "phase",
        "request_index",
        "request_id",
        "release_outcome",
        "release_observed_monotonic_ns",
        "reset_complete",
        "idle_settle_started_monotonic_ns",
    }
    for index, sample in enumerate(samples):
        if sample["sample_index"] != index:
            fail(f"{label} sample indices differ")
        for field in stable_fields:
            if not json_equal(sample[field], first[field]):
                fail(f"{label}.{field} changes within the point")
        if _resource_identity(sample) != _resource_identity(first):
            fail(f"{label} process identity changes within the point")
    starts = [sample["sample_monotonic_ns"] for sample in samples]
    settle_start = first["idle_settle_started_monotonic_ns"]
    if starts[0] - settle_start < 5_000_000_000:
        fail(f"{label} idle settle is shorter than five seconds")
    for prior, current in zip(starts, starts[1:]):
        if current - prior < 1_000_000_000:
            fail(f"{label} sample interval is shorter than one second")
    if first["phase"] == "post_release":
        release_time = first["release_observed_monotonic_ns"]
        if settle_start < release_time:
            fail(f"{label} settle starts before release")
    return ResourcePoint(
        segment=first["segment"],
        phase=first["phase"],
        request_index=first["request_index"],
        request_id=first["request_id"],
        release_outcome=first["release_outcome"],
        release_observed_monotonic_ns=first["release_observed_monotonic_ns"],
        idle_settle_started_monotonic_ns=settle_start,
        sample_monotonic_ns=tuple(starts),
        host_memory=median(
            sample["host"]["memory_current_bytes"] for sample in samples
        ),
        primary_vram=median(sample["gpu"]["mem_usage"]["value"] for sample in samples),
        gateway_rss=median(sample["gateway"]["vmrss_bytes"] for sample in samples),
        worker_rss=median(sample["worker"]["vmrss_bytes"] for sample in samples),
        gateway_threads=median(sample["gateway"]["threads"] for sample in samples),
        gateway_fds=median(sample["gateway"]["fd_count"] for sample in samples),
        gateway_children=median(
            len(sample["gateway"]["children"]) for sample in samples
        ),
        worker_threads=median(sample["worker"]["threads"] for sample in samples),
        worker_fds=median(sample["worker"]["fd_count"] for sample in samples),
        worker_children=median(len(sample["worker"]["children"]) for sample in samples),
    )


def _validate_gpu_metric(root: Path, record: dict[str, Any], label: str) -> None:
    reject_key_recursive(record, "passed", label)
    exact_fields(record, GPU_METRIC_FIELDS, label)
    if (
        record["schema_version"] != RESOURCE_SCHEMA
        or record["record_type"] != "gpu_metric"
    ):
        fail(f"{label} schema/record type differs")
    segment = record["segment"]
    boundary = record["boundary"]
    if (
        type(segment) is not str
        or segment not in {"normal", "restart"}
        or type(boundary) is not str
        or boundary not in {"before", "after"}
    ):
        fail(f"{label} segment/boundary differs")
    integer(record["captured_monotonic_ns"], f"{label}.captured_monotonic_ns")
    if integer(record["gpu_index"], f"{label}.gpu_index") != 2:
        fail(f"{label}.gpu_index differs")
    expected_name = f"amd-smi-metric-{segment}-{boundary}.json"
    if record["raw_output_file"] != expected_name:
        fail(f"{label}.raw_output_file differs")
    digest = sha256_value(record["raw_output_sha256"], f"{label}.raw_output_sha256")
    path = safe_relative_file(root, expected_name, expected_name)
    if sha256_file(path) != digest:
        fail(f"{label}.raw_output_sha256 differs")
    validate_json_document(path, expected_name)


def _phase_release_trace(session: SessionData, request_id_value: str) -> RequestTrace:
    trace = session.traces.get(request_id_value)
    if trace is None or trace.terminal != "request_released":
        fail(f"resource request {request_id_value} lacks a released lifecycle trace")
    return trace


def _validate_resource_lifecycle(
    session: SessionData,
    segment: str,
    baseline: ResourcePoint,
    points: list[ResourcePoint],
) -> None:
    phase = "resource_normal" if segment == "normal" else "resource_restart"
    expected_measured = 100 if segment == "normal" else 20
    releases = session.releases_by_phase.get(phase, [])
    if len(releases) != expected_measured + 10:
        fail(
            f"{phase} must contain ten warmup and {expected_measured} measured releases"
        )
    phase_traces = [trace for trace in session.traces.values() if trace.phase == phase]
    if len(phase_traces) != len(releases) or any(
        trace.terminal != "request_released" for trace in phase_traces
    ):
        fail(f"{phase} contains an extra or non-released request lifecycle")
    release_ids = [event["request_id"] for event in releases]
    measured_ids = [point.request_id for point in points]
    if release_ids[10:] != measured_ids:
        fail(f"{phase} measured release order differs from resource points")
    if baseline.idle_settle_started_monotonic_ns < releases[9]["observed_monotonic_ns"]:
        fail(f"{phase} baseline settle starts before the tenth warmup release")
    all_admissions = sorted(
        trace.events[0]["observed_monotonic_ns"]
        for trace in session.traces.values()
        if trace.events and trace.events[0]["event"] == "request_admitted"
    )
    quiet_intervals = [
        (releases[9]["observed_monotonic_ns"], baseline.sample_monotonic_ns[-1])
    ] + [
        (event["observed_monotonic_ns"], point.sample_monotonic_ns[-1])
        for point, event in zip(points, releases[10:], strict=True)
    ]
    for interval_start, interval_end in quiet_intervals:
        if any(
            interval_start < admitted <= interval_end for admitted in all_admissions
        ):
            fail(
                f"{phase} has a request admission during a frozen idle/sample interval"
            )
    ordered_traces: list[RequestTrace] = []
    for event in releases:
        trace = _phase_release_trace(session, event["request_id"])
        ordered_traces.append(trace)
        if trace.phase != phase:
            fail(f"{phase} lifecycle trace phase differs")
        admitted = trace.events[0]
        if (
            admitted["event"] != "request_admitted"
            or admitted["stream"] is not True
            or admitted["max_completion_tokens"] != 2
        ):
            fail(f"{phase} resource request admission parameters differ")
        if event["reset_complete"] is not True:
            fail(f"{phase} resource release reset acknowledgement differs")
    for prior_event, next_trace in zip(releases, ordered_traces[1:]):
        if (
            next_trace.events[0]["observed_monotonic_ns"]
            < prior_event["observed_monotonic_ns"]
        ):
            fail(f"{phase} requests overlap")
    first_measured_admission = ordered_traces[10].events[0]["observed_monotonic_ns"]
    if first_measured_admission < baseline.sample_monotonic_ns[-1]:
        fail(f"{phase} admits the first measured request during baseline sampling")
    for point, event in zip(points, releases[10:], strict=True):
        if point.release_observed_monotonic_ns != event["observed_monotonic_ns"]:
            fail(f"{phase} release observation timestamp differs")
        if point.release_outcome != event["outcome"]:
            fail(f"{phase} release outcome differs")
    for point, next_trace in zip(points, ordered_traces[11:]):
        if (
            next_trace.events[0]["observed_monotonic_ns"]
            < point.sample_monotonic_ns[-1]
        ):
            fail(f"{phase} admits a request during post-release resource sampling")


def _segment_metrics(
    baseline: ResourcePoint, points: list[ResourcePoint], label: str
) -> dict[str, Any]:
    expected_points = 100 if label == "normal" else 20
    if len(points) != expected_points:
        fail(f"{label} resource point count differs")
    diagnostic_names = (
        "gateway_threads",
        "gateway_fds",
        "gateway_children",
        "worker_threads",
        "worker_fds",
        "worker_children",
    )
    for point in points:
        for name in diagnostic_names:
            if getattr(point, name) != getattr(baseline, name):
                fail(
                    f"{label} {name} median differs from its segment baseline at request {point.request_index}"
                )

    host_values = [point.host_memory for point in points]
    vram_values = [point.primary_vram for point in points]
    gateway_rss_values = [point.gateway_rss for point in points]
    worker_rss_values = [point.worker_rss for point in points]
    host_final_delta = host_values[-1] - baseline.host_memory
    vram_final_delta = vram_values[-1] - baseline.primary_vram
    host_slope = theil_sen(host_values)
    vram_slope = theil_sen(vram_values)
    maximum_delta = Fraction(THRESHOLDS["final_delta_max_bytes"])
    maximum_slope = Fraction(THRESHOLDS["theil_sen_max_bytes_per_request"])
    if host_final_delta > maximum_delta:
        fail(f"{label} final MemoryCurrent delta exceeds the release threshold")
    if vram_final_delta > maximum_delta:
        fail(f"{label} final process VRAM delta exceeds the release threshold")
    if host_slope > maximum_slope:
        fail(f"{label} MemoryCurrent Theil-Sen slope exceeds the release threshold")
    if vram_slope > maximum_slope:
        fail(f"{label} process VRAM Theil-Sen slope exceeds the release threshold")
    return {
        "point_count": len(points),
        "baseline": {
            "memory_current_bytes": fraction_json(baseline.host_memory),
            "process_vram_bytes": fraction_json(baseline.primary_vram),
            "gateway_rss_bytes": fraction_json(baseline.gateway_rss),
            "worker_rss_bytes": fraction_json(baseline.worker_rss),
            "gateway_threads": fraction_json(baseline.gateway_threads),
            "gateway_fds": fraction_json(baseline.gateway_fds),
            "gateway_children": fraction_json(baseline.gateway_children),
            "worker_threads": fraction_json(baseline.worker_threads),
            "worker_fds": fraction_json(baseline.worker_fds),
            "worker_children": fraction_json(baseline.worker_children),
        },
        "final_delta": {
            "memory_current_bytes": fraction_json(host_final_delta),
            "process_vram_bytes": fraction_json(vram_final_delta),
        },
        "theil_sen_bytes_per_request": {
            "memory_current": fraction_json(host_slope),
            "process_vram": fraction_json(vram_slope),
            "gateway_rss_diagnostic": fraction_json(theil_sen(gateway_rss_values)),
            "worker_rss_diagnostic": fraction_json(theil_sen(worker_rss_values)),
        },
    }


def validate_resources(root: Path, session: SessionData) -> ResourceResult:
    iterator = iter_jsonl(root / "soak-resources.raw.jsonl", "soak-resources.raw.jsonl")
    try:
        first_line, header = next(iterator)
    except StopIteration:
        fail("soak-resources.raw.jsonl is empty")
    if first_line != 1:
        fail("resource header line differs")
    reject_key_recursive(header, "passed", "resource header")
    _validate_resource_header(header, "resource header")

    expected = list(_expected_resource_records())
    point_samples: list[dict[str, Any]] = []
    baselines: dict[str, ResourcePoint] = {}
    points: dict[str, list[ResourcePoint]] = {"normal": [], "restart": []}
    identities: dict[str, tuple[Any, ...]] = {}
    sample_count = 0
    metric_count = 0
    metric_times: dict[tuple[str, str], int] = {}

    observed_count = 0
    for observed_count, (line_number, record) in enumerate(iterator, start=1):
        label = f"soak-resources.raw.jsonl line {line_number}"
        if observed_count > len(expected):
            fail(f"{label} is an extra resource record")
        (
            expected_type,
            expected_segment,
            expected_phase,
            expected_request,
            expected_sample,
        ) = expected[observed_count - 1]
        if record.get("record_type") != expected_type:
            fail(f"{label}.record_type violates the exact resource state machine")
        if record.get("segment") != expected_segment:
            fail(f"{label}.segment violates the exact resource state machine")
        if expected_type == "gpu_metric":
            if record.get("boundary") != expected_phase:
                fail(f"{label}.boundary violates the exact resource state machine")
            _validate_gpu_metric(root, record, label)
            metric_times[(expected_segment, expected_phase)] = record[
                "captured_monotonic_ns"
            ]
            metric_count += 1
            continue

        _validate_resource_sample(record, label)
        if (
            record["phase"] != expected_phase
            or record["request_index"] != expected_request
            or record["sample_index"] != expected_sample
        ):
            fail(f"{label} violates the exact resource sample state machine")
        identity = _resource_identity(record)
        previous_identity = identities.setdefault(expected_segment, identity)
        if identity != previous_identity:
            fail(
                f"{label} process identity changes within the {expected_segment} segment"
            )
        point_samples.append(record)
        sample_count += 1
        if len(point_samples) == 5:
            point = _point_from_samples(
                point_samples,
                f"{expected_segment} {expected_phase} point {expected_request}",
            )
            if expected_phase == "baseline":
                if expected_segment in baselines:
                    fail(f"{expected_segment} baseline is duplicated")
                baselines[expected_segment] = point
            else:
                points[expected_segment].append(point)
            point_samples = []

    if observed_count != len(expected):
        fail(
            f"resource record count differs: expected {len(expected) + 1} total records"
        )
    if (
        point_samples
        or sample_count != 610
        or metric_count != 4
        or set(baselines) != {"normal", "restart"}
    ):
        fail("resource 1+610+4 state machine is incomplete")
    normal_identity = identities["normal"]
    restart_identity = identities["restart"]
    if normal_identity[0] != restart_identity[0]:
        fail("systemd ControlGroup changes across the planned service restart")
    if (normal_identity[1], normal_identity[4]) == (
        restart_identity[1],
        restart_identity[4],
    ) or (normal_identity[5], normal_identity[8]) == (
        restart_identity[5],
        restart_identity[8],
    ):
        fail(
            "gateway and worker identities must both change across the planned restart"
        )

    for segment in ("normal", "restart"):
        before = metric_times[(segment, "before")]
        after = metric_times[(segment, "after")]
        baseline = baselines[segment]
        lifecycle_phase = (
            "resource_normal" if segment == "normal" else "resource_restart"
        )
        first_release = session.releases_by_phase[lifecycle_phase][0]
        first_trace = session.traces[first_release["request_id"]]
        first_admission = first_trace.events[0]["observed_monotonic_ns"]
        if before > first_admission:
            fail(f"{segment} gpu metric-before occurs after the first warmup admission")
        if before > baseline.idle_settle_started_monotonic_ns:
            fail(f"{segment} gpu metric-before occurs after baseline settle start")
        if after < points[segment][-1].sample_monotonic_ns[-1]:
            fail(f"{segment} gpu metric-after occurs before the final resource sample")
        _validate_resource_lifecycle(session, segment, baseline, points[segment])
    if metric_times[("restart", "before")] < metric_times[("normal", "after")]:
        fail("restart metric-before occurs before normal metric-after")

    reports = {
        segment: _segment_metrics(baselines[segment], points[segment], segment)
        for segment in ("normal", "restart")
    }
    return ResourceResult(
        segments=reports, sample_count=sample_count, gpu_metric_count=metric_count
    )


def validate_phase1(
    bundle: Path,
    *,
    expected_commit: str,
    expected_worker_binary_sha256: str,
) -> dict[str, Any]:
    trusted_commit = git_commit(expected_commit, "--expected-commit")
    trusted_worker_sha = sha256_value(
        expected_worker_binary_sha256, "--expected-worker-binary-sha256"
    )
    root = safe_bundle_root(bundle)
    validate_bundle_layout(root)
    sums_sha256 = validate_sha256sums(root)
    matrix = validate_matrix(root)
    session = validate_session(root, matrix, trusted_commit, trusted_worker_sha)
    resources = validate_resources(root, session)
    return {
        "schema_version": PHASE1_REPORT_SCHEMA,
        "release_status": "incomplete",
        "phase1_validated": True,
        "run_id": matrix.run_id,
        "trusted_anchors": {
            "git_commit": trusted_commit,
            "worker_binary_sha256": trusted_worker_sha,
        },
        "verified_sha256sums_sha256": sums_sha256,
        "raw_counts": {
            "session_records": sum(session.record_counts.values()),
            "gateway_events": len(session.journal_events),
            "resource_samples": resources.sample_count,
            "gpu_metrics": resources.gpu_metric_count,
        },
        "resource_segments": resources.segments,
        "unimplemented_release_gates": [
            "api_contract",
            "openwebui_browser_smoke_and_20_chat_soak",
            "five_phase_cancellation_and_recovery",
            "post_header_failure_presentation_and_restart",
            "http_sse_ttft_and_decode",
            "aggregate_view_reconstruction",
            "complete_identity_and_source_state",
            "exclusive_release_validation_publication",
        ],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-worker-binary-sha256", required=True)
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="validate implemented phase-1 gates and emit an explicitly incomplete report",
    )
    return parser.parse_args(argv)


def _json_default(value: Any) -> Any:
    if type(value) is Decimal:
        return str(value)
    raise TypeError(f"unsupported JSON output type: {type(value).__name__}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        report = validate_phase1(
            args.bundle,
            expected_commit=args.expected_commit,
            expected_worker_binary_sha256=args.expected_worker_binary_sha256,
        )
        if not args.phase1_only:
            fail(
                "phase-1 evidence is valid, but full P8-F release gates are not implemented; "
                "release-validation.json was not written"
            )
    except ValidationError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            report,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
