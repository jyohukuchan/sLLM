#!/usr/bin/env python3
"""Independently validate standalone SQ8 worker acceptance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO, Iterable


RAW_SCHEMA_VERSION = "ullm.sq8.worker_acceptance.raw.v2"
RESULT_SCHEMA_VERSION = "ullm.sq8.worker_acceptance.validation.v2"
WORKER_SCHEMA_VERSION = "ullm.worker.v1"
MAX_LINE_BYTES = 8 * 1024 * 1024
U64_MAX = (1 << 64) - 1
U32_MAX = (1 << 32) - 1
VOCAB_SIZE = 151_936

ARTIFACT_MANIFEST_SHA256 = "23977f4e9bed4bac4cc64c177c35d7f83355861426bf32027a69cf7a241552e2"
ARTIFACT_CONTENT_SHA256 = "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147"
PACKAGE_MANIFEST_SHA256 = "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb"
MODEL_REVISION = "9a283b4a5efbc09ce247e0ae5b02b744739e525a"
GPU_INDEX = 2
GPU_BDF = "0000:47:00.0"
GPU_UUID = "a8ff7551-0000-1000-80e9-ddefa2d60f55"
KFD_GPU_ID = 51_545

REQUIRED_HIP_GUARDS = {
    "ULLM_REQUIRE_HIP_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL",
    "ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL",
    "ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL",
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
    "ULLM_REQUIRE_HIP_ROPE_KERNEL",
    "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL",
}

SCHEDULE = {
    "latency_warmups": 2,
    "latency_measured": 10,
    "resource_warmups": 10,
    "resource_requests": 100,
    "cancel_block_size": 5,
    "cancel_block_offset": 4,
    "idle_settle_ms": 5_000,
    "samples_per_point": 5,
    "sample_interval_ms": 1_000,
}
THRESHOLDS = {
    "cancel_sample_max_ns": 5_000_000_000,
    "cancel_p95_max_ns": 2_000_000_000,
    "theil_sen_max_bytes_per_request": 262_144,
    "final_delta_max_bytes": 67_108_864,
    "request_max_ns": 180_000_000_000,
    "progress_max_ns": 30_000_000_000,
    "shutdown_max_ns": 30_000_000_000,
}

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
REQUEST_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")

HEADER_FIELDS = {
    "schema_version",
    "record_type",
    "clock",
    "build",
    "worker",
    "device",
    "environment",
    "schedule",
    "thresholds",
}
COMMAND_BASE_FIELDS = {
    "schema_version",
    "record_type",
    "phase",
    "request_index",
    "request_id",
    "command_type",
    "write_started_monotonic_ns",
    "write_completed_monotonic_ns",
    "raw_json",
    "raw_sha256",
}
GENERATE_FIELDS = COMMAND_BASE_FIELDS | {
    "prompt_tokens",
    "prompt_token_ids_sha256",
    "max_new_tokens",
    "sampling",
    "eos_token_ids",
}
CANCEL_FIELDS = COMMAND_BASE_FIELDS | {"cancel_reason", "cancel_target"}
WORKER_EVENT_FIELDS = {
    "schema_version",
    "record_type",
    "observed_monotonic_ns",
    "raw_json",
    "raw_sha256",
    "event",
}
RESOURCE_SAMPLE_FIELDS = {
    "schema_version",
    "record_type",
    "phase",
    "request_index",
    "request_id",
    "release_outcome",
    "release_observed_monotonic_ns",
    "settle_started_monotonic_ns",
    "sample_index",
    "sample_started_monotonic_ns",
    "worker",
    "gpu",
}
RESOURCE_WORKER_FIELDS = {
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
    "stat_before_raw",
    "stat_before_raw_sha256",
    "status_raw",
    "status_raw_sha256",
    "exe_target",
    "fd_names",
    "children_raw",
    "children_raw_sha256",
    "stat_after_raw",
    "stat_after_raw_sha256",
}
RESOURCE_GPU_FIELDS = {
    "index",
    "bdf",
    "uuid",
    "kfd_gpu_id",
    "process_raw_json",
    "process_raw_sha256",
    "worker_pid",
    "mem_usage_value",
    "mem_usage_unit",
    "kfd_snapshot",
}
ISOLATION_CHECK_FIELDS = {
    "schema_version",
    "record_type",
    "phase",
    "request_index",
    "request_id",
    "release_observed_monotonic_ns",
    "kfd_snapshot",
}
KFD_SNAPSHOT_FIELDS = {
    "acquisition_started_monotonic_ns",
    "acquisition_completed_monotonic_ns",
    "deadline_monotonic_ns",
    "attempt_count",
    "retry_reasons",
    "attempts",
    "before_identities",
    "processes",
    "after_identities",
}
KFD_ATTEMPT_FIELDS = {
    "attempt_index",
    "started_monotonic_ns",
    "completed_monotonic_ns",
    "outcome",
    "retry_reason",
    "retry_stage",
    "retry_pid",
    "pids_before",
    "before_identities",
    "processes",
    "pids_after",
    "after_identities",
}
GPU_METRIC_FIELDS = {
    "schema_version",
    "record_type",
    "boundary",
    "captured_monotonic_ns",
    "raw_json",
    "raw_sha256",
}
PROCESS_EXIT_FIELDS = {
    "schema_version",
    "record_type",
    "stdout_eof_monotonic_ns",
    "exit_observed_monotonic_ns",
    "exit_code",
    "stderr_file",
    "stderr_sha256",
    "final_git_commit",
    "final_git_status_raw",
    "final_git_status_raw_sha256",
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


def parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        fail("JSON contains a non-finite number")
    return parsed


def reject_json_constant(value: str) -> None:
    fail(f"JSON contains invalid numeric constant: {value}")


def decode_json(text: str, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_float=parse_finite_float,
            parse_constant=reject_json_constant,
        )
    except ValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        fail(f"failed to decode {label}: {error}")


def json_type_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int/float coercions."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            json_type_equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            json_type_equal(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    return left == right


def exact_fields(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        fail(f"{label} field set differs: missing={missing} unknown={unknown}")
    return value


def integer(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = U64_MAX,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} must be an integer")
    if value < minimum or value > maximum:
        fail(f"{label} is outside {minimum}..={maximum}")
    return value


def timestamp(value: Any, label: str) -> int:
    return integer(value, label)


def sha256_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-256")
    return value


def request_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or REQUEST_ID_RE.fullmatch(value) is None:
        fail(f"{label} violates the worker request-ID contract")
    return value


def validate_git_status_text(value: str, label: str) -> None:
    if value and not value.endswith("\n"):
        fail(f"{label} must preserve the final LF")
    for line in value.splitlines():
        if not line.startswith("?? .rocprofv3/") or line == "?? .rocprofv3/..":
            fail(f"{label} contains a forbidden path: {line!r}")
        relative = line.removeprefix("?? ")
        if ".." in Path(relative).parts:
            fail(f"{label} escapes the profiler directory: {line!r}")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        fail(f"failed to hash {path}: {error}")
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        fail(f"failed to stat {label} {path}: {error}")
    if not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} must be a regular file, not a symlink: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as error:
        fail(f"failed to resolve {label} {path}: {error}")


def median(values: Iterable[int | Fraction]) -> Fraction:
    ordered = sorted(Fraction(value) for value in values)
    if not ordered:
        fail("median input must not be empty")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def percentile(values: Iterable[int], probability: Fraction) -> Fraction:
    ordered = sorted(Fraction(value) for value in values)
    if not ordered or probability < 0 or probability > 1:
        fail("percentile input is invalid")
    rank = Fraction(len(ordered) - 1) * probability
    lower = rank.numerator // rank.denominator
    upper = lower if rank.denominator == 1 else lower + 1
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (rank - lower) * (ordered[upper] - ordered[lower])


def theil_sen(values: list[Fraction]) -> Fraction:
    if len(values) != 100:
        fail("Theil-Sen input must contain exactly 100 point medians")
    slopes = [
        (values[j] - values[i]) / (j - i)
        for i in range(len(values))
        for j in range(i + 1, len(values))
    ]
    if len(slopes) != 4_950:
        fail("Theil-Sen construction did not produce 4950 slopes")
    return median(slopes)


def fraction_json(value: Fraction) -> int | float:
    if value.denominator == 1:
        return value.numerator
    return float(value)


def ascending_prompt_sha256(prompt_tokens: int) -> str:
    digest = hashlib.sha256()
    for token_id in range(1, prompt_tokens + 1):
        digest.update(token_id.to_bytes(4, "little"))
    return digest.hexdigest()


def validated_raw_text(value: Any, digest_value: Any, label: str) -> str:
    if not isinstance(value, str):
        fail(f"{label} must be a string")
    expected = sha256_value(digest_value, f"{label}_sha256")
    if expected != sha256_text(value):
        fail(f"{label} SHA-256 differs")
    return value


def parse_kfd_pids(value: Any, label: str) -> list[int]:
    if not isinstance(value, list):
        fail(f"{label} must be an array")
    parsed = [
        integer(item, f"{label}[{index}]", minimum=1, maximum=U32_MAX)
        for index, item in enumerate(value)
    ]
    if parsed != sorted(set(parsed)):
        fail(f"{label} must have ascending unique PIDs")
    return parsed


def parse_kfd_identities(value: Any, label: str) -> list[tuple[int, int, int]]:
    if not isinstance(value, list):
        fail(f"{label} must be an array")
    parsed: list[tuple[int, int, int]] = []
    for index, entry_value in enumerate(value):
        entry = exact_fields(
            entry_value, {"pid", "st_dev", "st_ino"}, f"{label}[{index}]"
        )
        parsed.append(
            (
                integer(
                    entry["pid"], f"{label}[{index}].pid", minimum=1, maximum=U32_MAX
                ),
                integer(entry["st_dev"], f"{label}[{index}].st_dev", minimum=0),
                integer(entry["st_ino"], f"{label}[{index}].st_ino", minimum=1),
            )
        )
    if [item[0] for item in parsed] != sorted({item[0] for item in parsed}):
        fail(f"{label} must have ascending unique PIDs")
    return parsed


def parse_kfd_processes(value: Any, label: str) -> list[tuple[int, int, int, int]]:
    if not isinstance(value, list):
        fail(f"{label} must be an array")
    parsed: list[tuple[int, int, int, int]] = []
    for index, entry_value in enumerate(value):
        entry = exact_fields(
            entry_value,
            {"pid", "st_dev", "st_ino", "vram_raw", "vram_bytes"},
            f"{label}[{index}]",
        )
        pid = integer(entry["pid"], f"{label}[{index}].pid", minimum=1, maximum=U32_MAX)
        st_dev = integer(entry["st_dev"], f"{label}[{index}].st_dev", minimum=0)
        st_ino = integer(entry["st_ino"], f"{label}[{index}].st_ino", minimum=1)
        raw = entry["vram_raw"]
        if (
            not isinstance(raw, str)
            or not raw
            or not raw.isascii()
            or not raw.isdecimal()
            or raw != raw.strip()
            or len(raw) > 4096
        ):
            fail(f"{label}[{index}].vram_raw must be stripped decimal ASCII")
        amount = integer(entry["vram_bytes"], f"{label}[{index}].vram_bytes")
        if int(raw, 10) != amount:
            fail(f"{label}[{index}] raw KFD value differs from vram_bytes")
        parsed.append((pid, st_dev, st_ino, amount))
    if [item[0] for item in parsed] != sorted({item[0] for item in parsed}):
        fail(f"{label} must have ascending unique PIDs")
    return parsed


@dataclass(frozen=True)
class ParsedKfdSnapshot:
    started_ns: int
    completed_ns: int
    processes: list[tuple[int, int, int, int]]
    attempt_count: int
    retry_count: int


def parse_kfd_snapshot(
    value: Any, label: str, expected_worker_pid: int | None
) -> ParsedKfdSnapshot:
    snapshot = exact_fields(value, KFD_SNAPSHOT_FIELDS, label)
    started = timestamp(
        snapshot["acquisition_started_monotonic_ns"],
        f"{label}.acquisition_started_monotonic_ns",
    )
    completed = timestamp(
        snapshot["acquisition_completed_monotonic_ns"],
        f"{label}.acquisition_completed_monotonic_ns",
    )
    deadline = timestamp(
        snapshot["deadline_monotonic_ns"], f"{label}.deadline_monotonic_ns"
    )
    if deadline != started + 1_000_000_000 or completed <= started or completed > deadline:
        fail(f"{label} does not preserve the fixed one-second acquisition deadline")
    attempt_count = integer(
        snapshot["attempt_count"], f"{label}.attempt_count", minimum=1
    )
    attempts = snapshot["attempts"]
    if not isinstance(attempts, list) or len(attempts) != attempt_count:
        fail(f"{label}.attempts differs from attempt_count")

    parsed_attempts: list[dict[str, Any]] = []
    expected_retry_reasons: list[str] = []
    previous_completed = -1
    for index, attempt_value in enumerate(attempts):
        attempt_label = f"{label}.attempts[{index}]"
        attempt = exact_fields(attempt_value, KFD_ATTEMPT_FIELDS, attempt_label)
        if integer(attempt["attempt_index"], f"{attempt_label}.attempt_index") != index:
            fail(f"{attempt_label} index is not contiguous")
        attempt_started = timestamp(
            attempt["started_monotonic_ns"], f"{attempt_label}.started_monotonic_ns"
        )
        attempt_completed = timestamp(
            attempt["completed_monotonic_ns"],
            f"{attempt_label}.completed_monotonic_ns",
        )
        if (
            (index == 0 and attempt_started != started)
            or (index > 0 and attempt_started <= previous_completed)
            or attempt_completed <= attempt_started
            or attempt_completed > deadline
        ):
            fail(f"{attempt_label} timestamps violate acquisition ordering")
        previous_completed = attempt_completed
        pids_before = parse_kfd_pids(attempt["pids_before"], f"{attempt_label}.pids_before")
        before_identities = parse_kfd_identities(
            attempt["before_identities"], f"{attempt_label}.before_identities"
        )
        processes = parse_kfd_processes(
            attempt["processes"], f"{attempt_label}.processes"
        )
        pids_after_value = attempt["pids_after"]
        pids_after = (
            None
            if pids_after_value is None
            else parse_kfd_pids(pids_after_value, f"{attempt_label}.pids_after")
        )
        if expected_worker_pid is not None and expected_worker_pid not in pids_before:
            fail(f"{attempt_label} omits the required worker from the before PID set")
        if (
            expected_worker_pid is not None
            and pids_after is not None
            and expected_worker_pid not in pids_after
        ):
            fail(f"{attempt_label} omits the required worker from the after PID set")
        after_identities = parse_kfd_identities(
            attempt["after_identities"], f"{attempt_label}.after_identities"
        )
        before_identity_pids = [item[0] for item in before_identities]
        process_pids = [item[0] for item in processes]
        after_identity_pids = [item[0] for item in after_identities]
        if any(
            amount > 0 and pid != expected_worker_pid
            for pid, _, _, amount in processes
        ):
            fail(f"{attempt_label} contains an unexpected positive KFD owner")
        if any(
            (pid, st_dev, st_ino)
            != before_identities[before_identity_pids.index(pid)]
            for pid, st_dev, st_ino, _ in processes
            if pid in before_identity_pids
        ) or any(pid not in before_identity_pids for pid in process_pids):
            fail(f"{attempt_label} process directory identity differs from before")

        outcome = attempt["outcome"]
        retry_reason = attempt["retry_reason"]
        retry_stage = attempt["retry_stage"]
        retry_pid = attempt["retry_pid"]
        if outcome == "stable":
            if index != attempt_count - 1:
                fail(f"{attempt_label} stable attempt must be final")
            if any(item is not None for item in (retry_reason, retry_stage, retry_pid)):
                fail(f"{attempt_label} stable attempt contains retry metadata")
            if (
                pids_after is None
                or pids_before != pids_after
                or before_identity_pids != pids_before
                or process_pids != pids_before
                or after_identity_pids != pids_after
                or before_identities != after_identities
            ):
                fail(f"{attempt_label} stable KFD identity/process sets differ")
        elif outcome == "retry":
            if index == attempt_count - 1:
                fail(f"{attempt_label} final attempt must be stable")
            if retry_reason not in {"entry_disappeared", "pid_set_changed"}:
                fail(f"{attempt_label} retry reason is invalid")
            if retry_stage not in {"before", "read", "after"}:
                fail(f"{attempt_label} retry stage is invalid")
            parsed_retry_pid = (
                None
                if retry_pid is None
                else integer(
                    retry_pid,
                    f"{attempt_label}.retry_pid",
                    minimum=1,
                    maximum=U32_MAX,
                )
            )
            if retry_reason == "entry_disappeared":
                if (
                    expected_worker_pid is not None
                    and parsed_retry_pid == expected_worker_pid
                ):
                    fail(f"{attempt_label} retries a required worker disappearance")
                if retry_stage == "before":
                    if processes or pids_after is not None or after_identities:
                        fail(f"{attempt_label} before disappearance has later evidence")
                    if parsed_retry_pid is None or (
                        before_identity_pids != pids_before[: len(before_identity_pids)]
                        or len(before_identity_pids) >= len(pids_before)
                        or pids_before[len(before_identity_pids)] != parsed_retry_pid
                    ):
                        fail(f"{attempt_label} before disappearance prefix differs")
                elif retry_stage == "read":
                    if (
                        parsed_retry_pid is None
                        or before_identity_pids != pids_before
                        or process_pids != pids_before[: len(process_pids)]
                        or len(process_pids) >= len(pids_before)
                        or pids_before[len(process_pids)] != parsed_retry_pid
                        or pids_after is not None
                        or after_identities
                    ):
                        fail(f"{attempt_label} read disappearance prefix differs")
                else:
                    if (
                        before_identity_pids != pids_before
                        or process_pids != pids_before
                    ):
                        fail(f"{attempt_label} after retry lacks complete before reads")
                    if parsed_retry_pid is None or (
                        pids_after is None
                        or after_identity_pids != pids_after[: len(after_identity_pids)]
                        or len(after_identity_pids) >= len(pids_after)
                        or pids_after[len(after_identity_pids)] != parsed_retry_pid
                    ):
                        fail(f"{attempt_label} after disappearance prefix differs")
                    before_map = {
                        pid: (dev, ino) for pid, dev, ino in before_identities
                    }
                    after_map = {
                        pid: (dev, ino) for pid, dev, ino in after_identities
                    }
                    if any(
                        before_map[pid] != after_map[pid]
                        for pid in before_map.keys() & after_map.keys()
                    ):
                        fail(f"{attempt_label} reused a PID across retry identities")
            else:
                if (
                    retry_stage != "after"
                    or parsed_retry_pid is not None
                    or before_identity_pids != pids_before
                    or process_pids != pids_before
                    or pids_after is None
                    or after_identity_pids != pids_after
                    or pids_before == pids_after
                ):
                    fail(f"{attempt_label} PID-set retry evidence differs")
                before_map = {pid: (dev, ino) for pid, dev, ino in before_identities}
                after_map = {pid: (dev, ino) for pid, dev, ino in after_identities}
                if any(before_map[pid] != after_map[pid] for pid in before_map.keys() & after_map.keys()):
                    fail(f"{attempt_label} reused a PID across retry sets")
            expected_retry_reasons.append(
                f"{retry_reason}:{retry_stage}:"
                f"{parsed_retry_pid if parsed_retry_pid is not None else 'null'}"
            )
        else:
            fail(f"{attempt_label}.outcome must be stable or retry")
        parsed_attempts.append(
            {
                "completed": attempt_completed,
                "before_identities": before_identities,
                "processes": processes,
                "after_identities": after_identities,
            }
        )

    if completed != parsed_attempts[-1]["completed"]:
        fail(f"{label} completion differs from the final stable attempt")
    retry_reasons = snapshot["retry_reasons"]
    if not json_type_equal(retry_reasons, expected_retry_reasons):
        fail(f"{label}.retry_reasons differs from retry attempts")
    final = attempts[-1]
    if (
        not json_type_equal(snapshot["before_identities"], final["before_identities"])
        or not json_type_equal(snapshot["processes"], final["processes"])
        or not json_type_equal(snapshot["after_identities"], final["after_identities"])
    ):
        fail(f"{label} final snapshot summaries differ from the stable attempt")
    final_processes = parsed_attempts[-1]["processes"]
    final_positive = [
        (pid, amount) for pid, _, _, amount in final_processes if amount > 0
    ]
    if expected_worker_pid is None:
        if final_positive:
            fail(f"{label} preflight snapshot contains positive KFD VRAM")
    elif (
        len(final_positive) != 1
        or final_positive[0][0] != expected_worker_pid
        or final_positive[0][1] <= 0
    ):
        fail(f"{label} does not prove sole positive worker KFD ownership")
    return ParsedKfdSnapshot(
        started,
        completed,
        final_processes,
        attempt_count,
        attempt_count - 1,
    )


def parse_proc_stat(raw: str, expected_pid: int, label: str) -> tuple[int, int]:
    if not isinstance(raw, str) or not raw.startswith(f"{expected_pid} ("):
        fail(f"{label} PID prefix differs")
    for match in reversed(list(re.finditer(r"\) ([A-Za-z]) ", raw))):
        fields = [match.group(1), *raw[match.end() :].strip().split()]
        if len(fields) < 20:
            continue
        try:
            ppid = int(fields[1], 10)
            starttime = int(fields[19], 10)
        except ValueError:
            continue
        if ppid > 0 and starttime > 0:
            return ppid, starttime
    fail(f"{label} lacks a valid rightmost comm delimiter")


def parse_proc_status(raw: str, label: str) -> tuple[int, int]:
    if not isinstance(raw, str):
        fail(f"{label} must be a string")
    vmrss: int | None = None
    threads: int | None = None
    for line in raw.splitlines():
        if line.startswith("VmRSS:"):
            match = re.fullmatch(r"VmRSS:\s+([0-9]+) kB", line)
            if match is None or vmrss is not None:
                fail(f"{label} VmRSS is malformed or repeated")
            vmrss = int(match.group(1), 10)
        elif line.startswith("Threads:"):
            match = re.fullmatch(r"Threads:\s+([0-9]+)", line)
            if match is None or threads is not None:
                fail(f"{label} Threads is malformed or repeated")
            threads = int(match.group(1), 10)
    if vmrss is None or threads is None or threads < 1:
        fail(f"{label} lacks VmRSS or Threads")
    if vmrss > U64_MAX // 1024:
        fail(f"{label} VmRSS byte conversion overflows")
    return vmrss, threads


@dataclass(frozen=True)
class RequestExpectation:
    segment: str
    phase: str
    index: int
    kind: str
    cancel_target: str | None = None
    latency_recovery: bool = False

    @property
    def prompt_tokens(self) -> int:
        return 128 if self.cancel_target == "prompt" else 8

    @property
    def max_new_tokens(self) -> int:
        return 512 if self.kind == "cancel" else 2


@dataclass
class ActiveRequest:
    expectation: RequestExpectation
    request_id: str
    generate_started_ns: int
    started_observed_ns: int | None = None
    progress: list[int] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    cancel_write_started_ns: int | None = None
    cancel_write_completed_ns: int | None = None
    cancel_required_now: bool = False
    eos_seen: bool = False
    progress_reference_ns: int | None = None


@dataclass(frozen=True)
class ReleaseRecord:
    phase: str
    request_index: int
    request_id: str
    outcome: str
    observed_ns: int


@dataclass(frozen=True)
class IsolationExpectation:
    phase: str
    request_index: int | None
    request_id: str | None
    release_observed_ns: int | None
    not_before_ns: int


@dataclass
class ResourcePoint:
    phase: str
    release: ReleaseRecord | None
    settle_started_ns: int | None = None
    previous_sample_ns: int | None = None
    samples: list[dict[str, int]] = field(default_factory=list)


def latency_plan() -> list[RequestExpectation]:
    result: list[RequestExpectation] = []
    for phase, count in (("latency_warmup", 2), ("latency_measured", 10)):
        for index in range(1, count + 1):
            target = "prompt" if index % 2 else "decode"
            result.append(RequestExpectation("latency", phase, index, "cancel", target))
            result.append(
                RequestExpectation("latency", phase, index, "normal", latency_recovery=True)
            )
    return result


def resource_plan(segment: str, phase: str, count: int) -> list[RequestExpectation]:
    result = []
    cancel_ordinal = 0
    for index in range(1, count + 1):
        if index % 5 == 4:
            cancel_ordinal += 1
            target = "prompt" if cancel_ordinal % 2 else "decode"
            result.append(RequestExpectation(segment, phase, index, "cancel", target))
        else:
            result.append(RequestExpectation(segment, phase, index, "normal"))
    return result


class AcceptanceValidator:
    def __init__(self, raw_path: Path, expected_commit: str, expected_binary_sha256: str):
        self.raw_path = raw_path
        self.expected_commit = expected_commit
        self.expected_binary_sha256 = expected_binary_sha256
        self.stage = "header"
        self.worker_identity: dict[str, Any] | None = None
        self.worker_binary_path: Path | None = None
        self.active: ActiveRequest | None = None
        self.pending_isolation: IsolationExpectation | None = None
        self.seen_request_ids: set[str] = set()
        self.last_command_ns = -1
        self.last_event_ns = -1
        self.last_record_time_ns = -1
        self.latency_requests = latency_plan()
        self.latency_cursor = 0
        self.resource_warmups = resource_plan(
            "resource_warmup", "resource_warmup", SCHEDULE["resource_warmups"]
        )
        self.resource_warmup_cursor = 0
        self.resource_requests = resource_plan(
            "resource_measured", "resource_measured", SCHEDULE["resource_requests"]
        )
        self.resource_cursor = 0
        self.pending_latency_recovery_id: str | None = None
        self.last_resource_warmup_release: ReleaseRecord | None = None
        self.resource_point: ResourcePoint | None = None
        self.baseline: dict[str, Fraction] | None = None
        self.rss_points: list[Fraction] = []
        self.vram_points: list[Fraction] = []
        self.latency_warmup_bounds: list[int] = []
        self.latency_measured_bounds: list[int] = []
        self.all_cancel_bounds: list[int] = []
        self.non_latency_warmup_cancel_bounds: list[int] = []
        self.gate_errors: list[str] = []
        self.record_counts: dict[str, int] = {}
        self.release_count = 0
        self.command_count = 0
        self.worker_event_count = 0
        self.resource_sample_count = 0
        self.gpu_metric_count = 0
        self.isolation_check_count = 0
        self.shutdown_write_started_ns: int | None = None
        self.process_exit: dict[str, Any] | None = None
        self.stderr_path: Path | None = None
        self.stderr_sha256: str | None = None
        self.preflight_kfd_completed_ns = -1
        self.kfd_snapshot_count = 0
        self.kfd_attempt_count = 0
        self.kfd_retry_count = 0

    def _validate_kfd_snapshot(
        self, value: Any, label: str, expected_worker_pid: int | None
    ) -> ParsedKfdSnapshot:
        parsed = parse_kfd_snapshot(value, label, expected_worker_pid)
        self.kfd_snapshot_count += 1
        self.kfd_attempt_count += parsed.attempt_count
        self.kfd_retry_count += parsed.retry_count
        return parsed

    def consume(self, record: dict[str, Any], line_number: int) -> None:
        label = f"line {line_number}"
        if record.get("schema_version") != RAW_SCHEMA_VERSION:
            fail(f"{label} has the wrong schema_version")
        record_type = record.get("record_type")
        if not isinstance(record_type, str):
            fail(f"{label}.record_type must be a string")
        self.record_counts[record_type] = self.record_counts.get(record_type, 0) + 1

        if self.stage == "done":
            fail(f"{label} appears after the final process_exit")
        if self.stage == "header":
            self._consume_header(record, label)
            return
        if self.stage == "ready":
            self._consume_ready(record, label)
            return

        if self.pending_isolation is not None:
            self._consume_isolation_check(record, label)
            return

        if self.active is not None:
            self._consume_active(record, label)
            return

        if self.resource_point is not None:
            if record_type != "resource_sample":
                fail(f"{label} interrupts a five-sample resource point")
            self._consume_resource_sample(record, label)
            return

        if self.stage == "latency":
            if self.latency_cursor < len(self.latency_requests):
                self._consume_generate(record, self.latency_requests[self.latency_cursor], label)
                self.latency_cursor += 1
            else:
                self._consume_gpu_metric(record, "before", label)
                self.stage = "resource_warmup"
            return
        if self.stage == "resource_warmup":
            if self.resource_warmup_cursor < len(self.resource_warmups):
                expectation = self.resource_warmups[self.resource_warmup_cursor]
                self._consume_generate(record, expectation, label)
                self.resource_warmup_cursor += 1
            else:
                if self.last_resource_warmup_release is None:
                    fail("resource baseline has no tenth warmup release")
                self.resource_point = ResourcePoint("baseline", None)
                self._consume_resource_sample(record, label)
            return
        if self.stage == "resource_measured":
            if self.resource_cursor < len(self.resource_requests):
                expectation = self.resource_requests[self.resource_cursor]
                self._consume_generate(record, expectation, label)
                self.resource_cursor += 1
            else:
                self._consume_gpu_metric(record, "after", label)
                self.stage = "shutdown"
            return
        if self.stage == "shutdown":
            self._consume_shutdown(record, label)
            self.stage = "process_exit"
            return
        if self.stage == "process_exit":
            self._consume_process_exit(record, label)
            self.stage = "done"
            return
        fail(f"{label} reached unknown validator stage {self.stage}")

    def _consume_header(self, record: dict[str, Any], label: str) -> None:
        exact_fields(record, HEADER_FIELDS, f"{label} header")
        if record["record_type"] != "header" or record["clock"] != "python.time.monotonic_ns":
            fail("the first record is not the frozen header")

        build = exact_fields(
            record["build"],
            {
                "git_commit",
                "tracked_clean",
                "git_status_raw",
                "git_status_raw_sha256",
                "binary_sha256",
                "artifact_manifest_sha256",
                "artifact_content_sha256",
                "package_manifest_sha256",
            },
            "header.build",
        )
        git_commit = build["git_commit"]
        git_status_raw = validated_raw_text(
            build["git_status_raw"],
            build["git_status_raw_sha256"],
            "header.build.git_status_raw",
        )
        validate_git_status_text(git_status_raw, "header.build.git_status_raw")
        if (
            not isinstance(git_commit, str)
            or GIT_COMMIT_RE.fullmatch(git_commit) is None
            or git_commit != self.expected_commit
            or build["tracked_clean"] is not True
            or sha256_value(build["binary_sha256"], "header.build.binary_sha256")
            != self.expected_binary_sha256
            or build["artifact_manifest_sha256"] != ARTIFACT_MANIFEST_SHA256
            or build["artifact_content_sha256"] != ARTIFACT_CONTENT_SHA256
            or build["package_manifest_sha256"] != PACKAGE_MANIFEST_SHA256
        ):
            fail("header build/model identity differs from the frozen inputs")

        worker = exact_fields(
            record["worker"], {"pid", "ppid", "starttime_ticks", "exe"}, "header.worker"
        )
        worker_identity = {
            "pid": integer(worker["pid"], "header.worker.pid", minimum=1, maximum=U32_MAX),
            "ppid": integer(worker["ppid"], "header.worker.ppid", minimum=1, maximum=U32_MAX),
            "starttime_ticks": integer(
                worker["starttime_ticks"], "header.worker.starttime_ticks", minimum=1
            ),
            "exe": worker["exe"],
        }
        if not isinstance(worker_identity["exe"], str) or not worker_identity["exe"]:
            fail("header.worker.exe must be a nonempty string")
        worker_executable = regular_file(Path(worker_identity["exe"]), "header worker executable")
        if sha256_file(worker_executable) != self.expected_binary_sha256:
            fail("header worker executable does not match the expected binary SHA-256")
        self.worker_binary_path = worker_executable
        self.worker_identity = worker_identity

        device = exact_fields(
            record["device"],
            {
                "gpu_index",
                "bdf",
                "uuid",
                "kfd_gpu_id",
                "amd_smi_list_raw_json",
                "amd_smi_list_raw_sha256",
            },
            "header.device",
        )
        if any(
            device[key] != expected
            for key, expected in {
                "gpu_index": GPU_INDEX,
                "bdf": GPU_BDF,
                "uuid": GPU_UUID,
                "kfd_gpu_id": KFD_GPU_ID,
            }.items()
        ):
            fail("header.device does not identify the frozen physical R9700")
        list_raw = device["amd_smi_list_raw_json"]
        if not isinstance(list_raw, str) or sha256_value(
            device["amd_smi_list_raw_sha256"], "header.device.amd_smi_list_raw_sha256"
        ) != sha256_text(list_raw):
            fail("header.device AMD SMI list raw SHA-256 differs")
        list_document = decode_json(list_raw, "header.device.amd_smi_list_raw_json")
        if not isinstance(list_document, list):
            fail("header.device AMD SMI list raw root must be an array")
        index_matches = [
            entry
            for entry in list_document
            if isinstance(entry, dict) and entry.get("gpu") == GPU_INDEX
        ]
        if len(index_matches) != 1 or any(
            index_matches[0].get(key) != expected
            for key, expected in {
                "gpu": GPU_INDEX,
                "bdf": GPU_BDF,
                "uuid": GPU_UUID,
                "kfd_id": KFD_GPU_ID,
            }.items()
        ):
            fail("header.device AMD SMI list must contain one unique matching GPU index 2")

        environment = exact_fields(
            record["environment"],
            {
                "hip_visible_devices",
                "required_hip_guards",
                "amd_smi_version_raw",
                "amd_smi_version_raw_sha256",
                "preflight_kfd_snapshot",
            },
            "header.environment",
        )
        guards = exact_fields(
            environment["required_hip_guards"], REQUIRED_HIP_GUARDS, "header.environment.required_hip_guards"
        )
        if environment["hip_visible_devices"] != "1" or any(value != "1" for value in guards.values()):
            fail("header environment does not enable the frozen isolated HIP path")
        version_raw = environment["amd_smi_version_raw"]
        if not isinstance(version_raw, str) or sha256_value(
            environment["amd_smi_version_raw_sha256"],
            "header.environment.amd_smi_version_raw_sha256",
        ) != sha256_text(version_raw):
            fail("header.environment AMD SMI version raw SHA-256 differs")
        for required in (
            "AMDSMI Tool: 26.2.2+e1a6bc5663",
            "AMDSMI Library version: 26.2.2",
            "ROCm version: 7.2.1",
        ):
            if required not in version_raw:
                fail(f"header.environment AMD SMI version is missing {required!r}")
        preflight_kfd = self._validate_kfd_snapshot(
            environment["preflight_kfd_snapshot"],
            "header.environment.preflight_kfd_snapshot",
            None,
        )
        self.preflight_kfd_completed_ns = preflight_kfd.completed_ns

        schedule = exact_fields(record["schedule"], set(SCHEDULE), "header.schedule")
        thresholds = exact_fields(record["thresholds"], set(THRESHOLDS), "header.thresholds")
        if any(schedule[key] != expected for key, expected in SCHEDULE.items()):
            fail("header.schedule differs from the frozen schedule")
        if any(thresholds[key] != expected for key, expected in THRESHOLDS.items()):
            fail("header.thresholds differs from the frozen thresholds")
        for key, value in schedule.items():
            integer(value, f"header.schedule.{key}")
        for key, value in thresholds.items():
            integer(value, f"header.thresholds.{key}")
        self.stage = "ready"

    def _consume_ready(self, record: dict[str, Any], label: str) -> None:
        event, observed = self._outer_worker_event(record, label)
        expected = {
            "schema_version": WORKER_SCHEMA_VERSION,
            "type": "ready",
            "model": "ullm-qwen3-14b-sq8",
            "model_revision": MODEL_REVISION,
            "artifact_content_sha256": ARTIFACT_CONTENT_SHA256,
            "package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
            "device": "gfx1201",
            "execution_profile": "rdna4_w8a8_block_ck",
            "context_length": 4_096,
            "max_new_tokens": 512,
        }
        exact_fields(event, set(expected), "ready event")
        if not json_type_equal(event, expected):
            fail("ready event differs from the frozen worker identity")
        if observed <= self.preflight_kfd_completed_ns:
            fail("ready observation does not follow preflight KFD acquisition")
        self.last_record_time_ns = observed
        self.pending_isolation = IsolationExpectation(
            "ready", None, None, None, observed
        )
        self.stage = "latency"

    def _consume_generate(
        self, record: dict[str, Any], expectation: RequestExpectation, label: str
    ) -> None:
        exact_fields(record, GENERATE_FIELDS, f"{label} generate")
        if record["record_type"] != "command" or record["command_type"] != "generate":
            fail(f"{label} must be the next scheduled generate command")
        start, _ = self._validate_command_common(record, expectation.phase, label)
        raw_command = self._decode_command_raw(record, label)
        exact_fields(
            raw_command,
            {
                "schema_version",
                "type",
                "request_id",
                "prompt_token_ids",
                "max_new_tokens",
                "sampling",
                "eos_token_ids",
            },
            f"{label}.raw_json",
        )
        index = integer(record["request_index"], f"{label}.request_index", minimum=1)
        if index != expectation.index:
            fail(f"{label} request index differs from the frozen matrix")
        current_id = request_id(record["request_id"], f"{label}.request_id")
        expected_id = self._expected_request_id(expectation)
        if current_id != expected_id:
            fail(f"{label} request ID must be {expected_id}")
        if current_id in self.seen_request_ids:
            fail(f"{label} reuses request ID {current_id}")
        if expectation.latency_recovery:
            if self.pending_latency_recovery_id is None or current_id != self.pending_latency_recovery_id:
                fail(f"{label} latency recovery ID must be the cancel ID plus -recovery")
            self.pending_latency_recovery_id = None
        elif self.pending_latency_recovery_id is not None:
            fail(f"{label} omitted the immediate latency recovery request")
        self.seen_request_ids.add(current_id)

        prompt_tokens = integer(record["prompt_tokens"], f"{label}.prompt_tokens", minimum=1)
        max_new_tokens = integer(record["max_new_tokens"], f"{label}.max_new_tokens", minimum=1)
        if prompt_tokens != expectation.prompt_tokens or max_new_tokens != expectation.max_new_tokens:
            fail(f"{label} prompt/completion limits differ from the frozen request matrix")
        prompt_sha = sha256_value(
            record["prompt_token_ids_sha256"], f"{label}.prompt_token_ids_sha256"
        )
        if prompt_sha != ascending_prompt_sha256(prompt_tokens):
            fail(f"{label} prompt token digest is not ascending u32 little-endian IDs")
        sampling = exact_fields(
            record["sampling"], {"temperature", "top_p", "top_k", "seed"}, f"{label}.sampling"
        )
        temperature = sampling["temperature"]
        top_p = sampling["top_p"]
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) != 0.0
            or isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not math.isfinite(float(top_p))
            or float(top_p) != 1.0
            or sampling["top_k"] != 20
            or isinstance(sampling["top_k"], bool)
            or sampling["seed"] != 0
            or isinstance(sampling["seed"], bool)
        ):
            fail(f"{label} sampling differs from the frozen greedy request")
        if record["eos_token_ids"] != [151_645, 151_643]:
            fail(f"{label}.eos_token_ids differs from the product EOS IDs")
        expected_prompt_ids = list(range(1, prompt_tokens + 1))
        raw_prompt_ids = raw_command["prompt_token_ids"]
        if (
            not isinstance(raw_prompt_ids, list)
            or any(type(token_id) is not int for token_id in raw_prompt_ids)
            or type(raw_command["max_new_tokens"]) is not int
            or not isinstance(raw_command["sampling"], dict)
            or type(raw_command["sampling"].get("top_k")) is not int
            or type(raw_command["sampling"].get("seed")) is not int
            or not isinstance(raw_command["eos_token_ids"], list)
            or any(type(token_id) is not int for token_id in raw_command["eos_token_ids"])
        ):
            fail(f"{label} raw generate object uses the wrong JSON field types")
        if not json_type_equal(raw_command, {
            "schema_version": WORKER_SCHEMA_VERSION,
            "type": "generate",
            "request_id": current_id,
            "prompt_token_ids": expected_prompt_ids,
            "max_new_tokens": max_new_tokens,
            "sampling": sampling,
            "eos_token_ids": [151_645, 151_643],
        }):
            fail(f"{label} command summary differs from the complete raw generate object")

        self.command_count += 1
        self.active = ActiveRequest(
            expectation, current_id, start, progress_reference_ns=start
        )

    @staticmethod
    def _expected_request_id(expectation: RequestExpectation) -> str:
        if expectation.phase == "latency_warmup":
            base = f"p8c-latency-warmup-{expectation.index:02d}"
        elif expectation.phase == "latency_measured":
            base = f"p8c-latency-measured-{expectation.index:02d}"
        elif expectation.phase == "resource_warmup":
            base = f"p8c-resource-warmup-{expectation.index:02d}"
        elif expectation.phase == "resource_measured":
            base = f"p8c-resource-measured-{expectation.index:03d}"
        else:
            fail(f"unknown request phase {expectation.phase}")
        return base + ("-recovery" if expectation.latency_recovery else "")

    def _consume_active(self, record: dict[str, Any], label: str) -> None:
        assert self.active is not None
        if self.active.cancel_required_now:
            if record.get("record_type") != "command" or record.get("command_type") != "cancel":
                fail(f"{label} must record cancel immediately after its target event")
            self._consume_cancel(record, label)
            return
        if record.get("record_type") != "worker_event":
            fail(f"{label} is not a legal record while request {self.active.request_id} is active")
        event, observed = self._outer_worker_event(record, label)
        if (
            self.active.cancel_write_started_ns is None
            and self.active.progress_reference_ns is not None
        ):
            progress_gap = observed - self.active.progress_reference_ns
            if progress_gap <= 0:
                fail(f"{label} worker event does not follow its progress reference")
            if progress_gap > THRESHOLDS["progress_max_ns"]:
                self.gate_errors.append(
                    f"{self.active.request_id} progress gap {progress_gap} exceeds "
                    f"{THRESHOLDS['progress_max_ns']} ns"
                )
            self.active.progress_reference_ns = observed
        event_type = event.get("type")
        if event_type == "started":
            self._event_started(event, observed, label)
        elif event_type == "progress":
            self._event_progress(event, label)
        elif event_type == "token":
            self._event_token(event, observed, label)
        elif event_type == "released":
            self._event_released(event, observed, label)
        elif event_type == "error":
            fail(f"{label} contains an error event in an acceptance run")
        else:
            fail(f"{label} contains invalid active-request event type {event_type!r}")
        self.last_record_time_ns = max(self.last_record_time_ns, observed)

    def _consume_cancel(self, record: dict[str, Any], label: str) -> None:
        assert self.active is not None
        exact_fields(record, CANCEL_FIELDS, f"{label} cancel")
        if record["record_type"] != "command" or record["command_type"] != "cancel":
            fail(f"{label} is not a cancel command")
        start, completed = self._validate_command_common(record, self.active.expectation.phase, label)
        raw_command = self._decode_command_raw(record, label)
        exact_fields(
            raw_command,
            {"schema_version", "type", "request_id", "reason"},
            f"{label}.raw_json",
        )
        if integer(record["request_index"], f"{label}.request_index", minimum=1) != self.active.expectation.index:
            fail(f"{label} cancel request index differs from its generate")
        if record["request_id"] != self.active.request_id:
            fail(f"{label} cancel request ID differs from the active request")
        if record["cancel_reason"] != "operator" or record["cancel_target"] != self.active.expectation.cancel_target:
            fail(f"{label} cancel reason/target differs from the frozen matrix")
        if not json_type_equal(raw_command, {
            "schema_version": WORKER_SCHEMA_VERSION,
            "type": "cancel",
            "request_id": self.active.request_id,
            "reason": "operator",
        }):
            fail(f"{label} command summary differs from the complete raw cancel object")
        if self.active.eos_seen:
            fail(f"{label} cancel follows an EOS token and violates terminal precedence")
        trigger = (
            self.active.started_observed_ns
            if self.active.expectation.cancel_target == "prompt"
            else self.last_event_ns
        )
        if trigger is None or start <= trigger:
            fail(f"{label} cancel write does not follow its observed target event")
        trigger_gap = start - trigger
        if trigger_gap > THRESHOLDS["progress_max_ns"]:
            self.gate_errors.append(
                f"{self.active.request_id} cancel trigger gap {trigger_gap} exceeds "
                f"{THRESHOLDS['progress_max_ns']} ns"
            )
        self.active.cancel_write_started_ns = start
        self.active.cancel_write_completed_ns = completed
        self.active.cancel_required_now = False
        self.command_count += 1

    def _validate_command_common(
        self, record: dict[str, Any], phase: str, label: str
    ) -> tuple[int, int]:
        if record["phase"] != phase:
            fail(f"{label}.phase differs from the frozen matrix")
        started = timestamp(record["write_started_monotonic_ns"], f"{label}.write_started_monotonic_ns")
        completed = timestamp(record["write_completed_monotonic_ns"], f"{label}.write_completed_monotonic_ns")
        if (
            started <= self.last_command_ns
            or started <= self.last_record_time_ns
            or completed <= started
        ):
            fail(f"{label} command timestamps are not strictly increasing")
        self.last_command_ns = completed
        self.last_record_time_ns = max(self.last_record_time_ns, completed)
        return started, completed

    @staticmethod
    def _decode_command_raw(record: dict[str, Any], label: str) -> dict[str, Any]:
        raw = validated_raw_text(record["raw_json"], record["raw_sha256"], f"{label}.raw_json")
        if raw.endswith(("\n", "\r")):
            fail(f"{label}.raw_json must exclude the terminating LF")
        decoded = decode_json(raw, f"{label}.raw_json")
        if not isinstance(decoded, dict):
            fail(f"{label}.raw_json root must be an object")
        return decoded

    def _outer_worker_event(
        self, record: dict[str, Any], label: str
    ) -> tuple[dict[str, Any], int]:
        exact_fields(record, WORKER_EVENT_FIELDS, f"{label} worker_event")
        if record["record_type"] != "worker_event":
            fail(f"{label} is not a worker_event")
        observed = timestamp(record["observed_monotonic_ns"], f"{label}.observed_monotonic_ns")
        if observed <= self.last_event_ns:
            fail(f"{label} worker event timestamps are not strictly increasing")
        self.last_event_ns = observed
        self.worker_event_count += 1
        event = record["event"]
        if not isinstance(event, dict) or event.get("schema_version") != WORKER_SCHEMA_VERSION:
            fail(f"{label}.event is not an ullm.worker.v1 object")
        raw = validated_raw_text(
            record["raw_json"], record["raw_sha256"], f"{label}.raw_json"
        )
        if raw.endswith(("\n", "\r")):
            fail(f"{label}.raw_json must exclude the terminating LF")
        raw_event = decode_json(raw, f"{label}.raw_json")
        if not isinstance(raw_event, dict) or not json_type_equal(raw_event, event):
            fail(f"{label}.event differs from reparsed raw worker stdout")
        return event, observed

    def _event_started(self, event: dict[str, Any], observed: int, label: str) -> None:
        assert self.active is not None
        exact_fields(event, {"schema_version", "type", "request_id", "prompt_tokens"}, f"{label}.event")
        if self.active.started_observed_ns is not None or event["request_id"] != self.active.request_id:
            fail(f"{label} started event is duplicated or has a mismatched request ID")
        prompt_tokens = integer(
            event["prompt_tokens"], f"{label}.event.prompt_tokens", minimum=1
        )
        if prompt_tokens != self.active.expectation.prompt_tokens:
            fail(f"{label} started prompt count differs from generate")
        if observed <= self.active.generate_started_ns:
            fail(f"{label} started observation precedes generate write start")
        self.active.started_observed_ns = observed
        if self.active.expectation.cancel_target == "prompt":
            self.active.cancel_required_now = True

    def _event_progress(self, event: dict[str, Any], label: str) -> None:
        assert self.active is not None
        exact_fields(
            event,
            {"schema_version", "type", "request_id", "phase", "processed_prompt_tokens"},
            f"{label}.event",
        )
        if (
            self.active.started_observed_ns is None
            or self.active.tokens
            or event["request_id"] != self.active.request_id
        ):
            fail(f"{label} progress event is out of order or has the wrong request ID")
        processed = integer(
            event["processed_prompt_tokens"],
            f"{label}.event.processed_prompt_tokens",
            minimum=1,
        )
        previous = self.active.progress[-1] if self.active.progress else 0
        if (
            event["phase"] != "prefill"
            or processed <= previous
            or processed > self.active.expectation.prompt_tokens
            or (processed != self.active.expectation.prompt_tokens and processed % 128 != 0)
        ):
            fail(f"{label} progress event violates the M=128 prompt contract")
        self.active.progress.append(processed)

    def _event_token(self, event: dict[str, Any], observed: int, label: str) -> None:
        assert self.active is not None
        if self.active.eos_seen:
            fail(f"{label} token follows an EOS token")
        exact_fields(event, {"schema_version", "type", "request_id", "index", "token_id"}, f"{label}.event")
        if (
            self.active.started_observed_ns is None
            or not self.active.progress
            or self.active.progress[-1] != self.active.expectation.prompt_tokens
            or event["request_id"] != self.active.request_id
        ):
            fail(f"{label} token event occurs before complete prefill or has the wrong request ID")
        index = integer(event["index"], f"{label}.event.index")
        token_id = integer(event["token_id"], f"{label}.event.token_id", maximum=VOCAB_SIZE - 1)
        if index != len(self.active.tokens):
            fail(f"{label} token indices are not contiguous from zero")
        self.active.tokens.append(token_id)
        if token_id in (151_645, 151_643):
            self.active.eos_seen = True
        if self.active.expectation.cancel_target == "decode" and index == 0:
            self.active.cancel_required_now = True
            if observed != self.last_event_ns:
                fail(f"{label} decode trigger timestamp is inconsistent")

    def _event_released(self, event: dict[str, Any], observed: int, label: str) -> None:
        assert self.active is not None
        expectation = self.active.expectation
        cancelled = expectation.kind == "cancel"
        fields = {
            "schema_version",
            "type",
            "request_id",
            "outcome",
            "prompt_tokens",
            "completion_tokens",
            "reset_complete",
        }
        if cancelled:
            fields.add("cancel_reason")
        exact_fields(event, fields, f"{label}.event")
        if self.active.started_observed_ns is None or event["request_id"] != self.active.request_id:
            fail(f"{label} release lacks a matching started event")
        prompt_tokens = integer(event["prompt_tokens"], f"{label}.event.prompt_tokens", minimum=1)
        completion_tokens = integer(event["completion_tokens"], f"{label}.event.completion_tokens")
        if (
            prompt_tokens != expectation.prompt_tokens
            or completion_tokens != len(self.active.tokens)
            or event["reset_complete"] is not True
        ):
            fail(f"{label} release counters/reset do not match the observed request")
        request_duration = observed - self.active.generate_started_ns
        if request_duration <= 0:
            fail(f"{label} release does not follow generate write start")
        if request_duration > THRESHOLDS["request_max_ns"]:
            self.gate_errors.append(
                f"{self.active.request_id} request duration {request_duration} exceeds "
                f"{THRESHOLDS['request_max_ns']} ns"
            )

        if cancelled:
            if (
                self.active.cancel_write_started_ns is None
                or event["outcome"] != "cancelled"
                or event["cancel_reason"] != "operator"
                or completion_tokens >= 512
                or (expectation.cancel_target == "prompt" and completion_tokens != 0)
                or (expectation.cancel_target == "decode" and completion_tokens < 1)
                or self.active.eos_seen
            ):
                fail(f"{label} cancelled release differs from the frozen phase contract")
            bound = observed - self.active.cancel_write_started_ns
            if bound <= 0:
                fail(f"{label} cancellation release does not follow cancel write start")
            if expectation.segment == "latency":
                if expectation.phase == "latency_warmup":
                    self.latency_warmup_bounds.append(bound)
                else:
                    self.latency_measured_bounds.append(bound)
                self.pending_latency_recovery_id = self.active.request_id + "-recovery"
            self.all_cancel_bounds.append(bound)
            if expectation.phase != "latency_warmup":
                self.non_latency_warmup_cancel_bounds.append(bound)
            if bound > THRESHOLDS["cancel_sample_max_ns"]:
                self.gate_errors.append(
                    f"{expectation.phase}[{expectation.index}] cancel upper bound "
                    f"{bound} exceeds {THRESHOLDS['cancel_sample_max_ns']} ns"
                )
        else:
            if (
                event["outcome"] != "length"
                or completion_tokens != 2
                or self.active.progress != [8]
                or self.active.cancel_write_started_ns is not None
                or self.active.eos_seen
            ):
                fail(f"{label} normal release differs from Length completion-2 baseline")

        release = ReleaseRecord(
            expectation.phase,
            expectation.index,
            self.active.request_id,
            event["outcome"],
            observed,
        )
        self.release_count += 1
        if expectation.segment == "resource_warmup":
            self.last_resource_warmup_release = release
        elif expectation.segment == "resource_measured":
            self.resource_point = ResourcePoint("post_release", release)
        self.pending_isolation = IsolationExpectation(
            expectation.phase,
            expectation.index,
            self.active.request_id,
            observed,
            observed,
        )
        self.active = None

    def _consume_isolation_check(self, record: dict[str, Any], label: str) -> None:
        assert self.pending_isolation is not None
        if self.worker_identity is None:
            fail("isolation check appears before worker identity")
        expected = self.pending_isolation
        exact_fields(record, ISOLATION_CHECK_FIELDS, f"{label} isolation_check")
        if record["record_type"] != "isolation_check" or record["phase"] != expected.phase:
            fail(f"{label} is not the required immediate {expected.phase} isolation check")
        if (
            record["request_index"] != expected.request_index
            or record["request_id"] != expected.request_id
            or record["release_observed_monotonic_ns"] != expected.release_observed_ns
        ):
            fail(f"{label} isolation check does not match Ready/release identity")
        snapshot = self._validate_kfd_snapshot(
            record["kfd_snapshot"],
            f"{label}.kfd_snapshot",
            self.worker_identity["pid"],
        )
        if (
            snapshot.started_ns <= expected.not_before_ns
            or snapshot.started_ns <= self.last_record_time_ns
        ):
            fail(f"{label} isolation capture is not ordered after Ready/release")
        self.last_record_time_ns = snapshot.completed_ns
        self.isolation_check_count += 1
        self.pending_isolation = None

    def _consume_resource_sample(self, record: dict[str, Any], label: str) -> None:
        assert self.resource_point is not None
        point = self.resource_point
        exact_fields(record, RESOURCE_SAMPLE_FIELDS, f"{label} resource_sample")
        if record["record_type"] != "resource_sample" or record["phase"] != point.phase:
            fail(f"{label} resource phase differs from the scheduled point")
        sample_index = integer(record["sample_index"], f"{label}.sample_index", maximum=4)
        if sample_index != len(point.samples):
            fail(f"{label} resource sample indices are not 0..4 in order")
        settle = timestamp(record["settle_started_monotonic_ns"], f"{label}.settle_started_monotonic_ns")
        sample_started = timestamp(record["sample_started_monotonic_ns"], f"{label}.sample_started_monotonic_ns")
        release_floor = (
            self.last_resource_warmup_release.observed_ns
            if point.phase == "baseline" and self.last_resource_warmup_release is not None
            else point.release.observed_ns if point.release is not None else -1
        )
        if settle < release_floor:
            fail(f"{label} settle starts before its matching release")
        if point.settle_started_ns is None:
            point.settle_started_ns = settle
        elif settle != point.settle_started_ns:
            fail(f"{label} changes settle_started_monotonic_ns within one point")
        if sample_index == 0:
            if sample_started - settle < 5_000_000_000:
                fail(f"{label} first resource sample violates the five-second settle")
        elif point.previous_sample_ns is None or sample_started - point.previous_sample_ns < 1_000_000_000:
            fail(f"{label} resource sample interval is below one second")
        if sample_started <= self.last_record_time_ns:
            fail(f"{label} resource sample timestamp is not after prior evidence activity")

        if point.phase == "baseline":
            if any(
                record[key] is not None
                for key in (
                    "request_index",
                    "request_id",
                    "release_outcome",
                    "release_observed_monotonic_ns",
                )
            ):
                fail(f"{label} baseline request/release fields must be null")
        else:
            assert point.release is not None
            if (
                integer(record["request_index"], f"{label}.request_index", minimum=1, maximum=100)
                != point.release.request_index
                or record["request_id"] != point.release.request_id
                or record["release_outcome"] != point.release.outcome
                or timestamp(record["release_observed_monotonic_ns"], f"{label}.release_observed_monotonic_ns")
                != point.release.observed_ns
            ):
                fail(f"{label} does not match its measured request release")

        resource, kfd_completed = self._validate_resource_values(
            record["worker"], record["gpu"], sample_started, label
        )
        point.samples.append(resource)
        point.previous_sample_ns = sample_started
        self.last_record_time_ns = kfd_completed
        self.resource_sample_count += 1
        if len(point.samples) == 5:
            self._finish_resource_point(point)
            self.resource_point = None

    def _validate_resource_values(
        self, worker_value: Any, gpu_value: Any, sample_started: int, label: str
    ) -> tuple[dict[str, int], int]:
        if self.worker_identity is None:
            fail("resource sample appears before worker identity")
        worker = exact_fields(worker_value, RESOURCE_WORKER_FIELDS, f"{label}.worker")
        pid = integer(worker["pid"], f"{label}.worker.pid", minimum=1, maximum=U32_MAX)
        ppid = integer(worker["ppid"], f"{label}.worker.ppid", minimum=1, maximum=U32_MAX)
        start_before = integer(
            worker["starttime_ticks_before"], f"{label}.worker.starttime_ticks_before", minimum=1
        )
        start_after = integer(
            worker["starttime_ticks_after"], f"{label}.worker.starttime_ticks_after", minimum=1
        )
        if (
            pid != self.worker_identity["pid"]
            or ppid != self.worker_identity["ppid"]
            or worker["exe"] != self.worker_identity["exe"]
            or start_before != self.worker_identity["starttime_ticks"]
            or start_after != self.worker_identity["starttime_ticks"]
        ):
            fail(f"{label} worker process identity changed")
        vmrss_kb = integer(worker["vmrss_kb"], f"{label}.worker.vmrss_kb", minimum=1)
        vmrss_bytes = integer(worker["vmrss_bytes"], f"{label}.worker.vmrss_bytes", minimum=1)
        if vmrss_kb > U64_MAX // 1024 or vmrss_bytes != vmrss_kb * 1024:
            fail(f"{label} worker VmRSS conversion differs from kB * 1024")
        threads = integer(worker["threads"], f"{label}.worker.threads", minimum=1)
        fd_count = integer(worker["fd_count"], f"{label}.worker.fd_count")
        children = worker["children"]
        if not isinstance(children, list):
            fail(f"{label}.worker.children must be an array")
        parsed_children = [
            integer(value, f"{label}.worker.children[{index}]", minimum=1, maximum=U32_MAX)
            for index, value in enumerate(children)
        ]
        if parsed_children != sorted(set(parsed_children)):
            fail(f"{label}.worker.children must contain ascending unique PIDs")

        stat_before_raw = validated_raw_text(
            worker["stat_before_raw"],
            worker["stat_before_raw_sha256"],
            f"{label}.worker.stat_before_raw",
        )
        stat_after_raw = validated_raw_text(
            worker["stat_after_raw"],
            worker["stat_after_raw_sha256"],
            f"{label}.worker.stat_after_raw",
        )
        raw_ppid_before, raw_start_before = parse_proc_stat(
            stat_before_raw, pid, f"{label}.worker.stat_before_raw"
        )
        raw_ppid_after, raw_start_after = parse_proc_stat(
            stat_after_raw, pid, f"{label}.worker.stat_after_raw"
        )
        if (
            (raw_ppid_before, raw_start_before) != (ppid, start_before)
            or (raw_ppid_after, raw_start_after) != (ppid, start_after)
        ):
            fail(f"{label} derived worker identity differs from raw /proc stat")
        status_raw = validated_raw_text(
            worker["status_raw"],
            worker["status_raw_sha256"],
            f"{label}.worker.status_raw",
        )
        raw_vmrss_kb, raw_threads = parse_proc_status(
            status_raw, f"{label}.worker.status_raw"
        )
        if raw_vmrss_kb != vmrss_kb or raw_threads != threads:
            fail(f"{label} derived RSS/thread count differs from raw /proc status")
        if (
            not isinstance(worker["exe_target"], str)
            or worker["exe_target"] != worker["exe"]
            or worker["exe_target"] != self.worker_identity["exe"]
        ):
            fail(f"{label} executable summary differs from raw /proc exe target")
        fd_names = worker["fd_names"]
        if (
            not isinstance(fd_names, list)
            or any(
                not isinstance(name, str) or not name.isascii() or not name.isdecimal()
                for name in fd_names
            )
            or len(set(fd_names)) != len(fd_names)
            or len(fd_names) != fd_count
        ):
            fail(f"{label} FD count differs from raw /proc FD names")
        children_raw = validated_raw_text(
            worker["children_raw"],
            worker["children_raw_sha256"],
            f"{label}.worker.children_raw",
        )
        if not children_raw.isascii():
            fail(f"{label}.worker.children_raw must be ASCII")
        try:
            raw_children = [int(value, 10) for value in children_raw.split()]
        except ValueError:
            fail(f"{label}.worker.children_raw is malformed")
        if raw_children != parsed_children:
            fail(f"{label} child PID summary differs from raw /proc children")

        gpu = exact_fields(gpu_value, RESOURCE_GPU_FIELDS, f"{label}.gpu")
        if (
            gpu["index"] != GPU_INDEX
            or gpu["bdf"] != GPU_BDF
            or gpu["uuid"] != GPU_UUID
            or gpu["kfd_gpu_id"] != KFD_GPU_ID
            or gpu["worker_pid"] != pid
            or gpu["mem_usage_unit"] != "B"
        ):
            fail(f"{label} GPU identity/isolation differs from the frozen R9700")
        mem_usage = integer(gpu["mem_usage_value"], f"{label}.gpu.mem_usage_value", minimum=1)
        kfd_snapshot = self._validate_kfd_snapshot(
            gpu["kfd_snapshot"], f"{label}.gpu.kfd_snapshot", pid
        )
        if kfd_snapshot.started_ns < sample_started:
            fail(f"{label} KFD acquisition starts before resource sample")
        own_kfd = [
            amount
            for process_pid, _, _, amount in kfd_snapshot.processes
            if process_pid == pid and amount > 0
        ]
        if own_kfd != [mem_usage]:
            fail(f"{label} AMD SMI and final stable KFD VRAM bytes differ")
        raw_json = gpu["process_raw_json"]
        if not isinstance(raw_json, str):
            fail(f"{label}.gpu.process_raw_json must be a string")
        raw_sha = sha256_value(gpu["process_raw_sha256"], f"{label}.gpu.process_raw_sha256")
        if raw_sha != sha256_text(raw_json):
            fail(f"{label} AMD SMI process raw SHA-256 differs")
        raw_document = decode_json(raw_json, f"{label}.gpu.process_raw_json")
        if (
            not isinstance(raw_document, list)
            or len(raw_document) != 1
            or not isinstance(raw_document[0], dict)
        ):
            fail(f"{label} AMD SMI process raw JSON must contain one GPU object")
        raw_gpu = raw_document[0]
        processes = raw_gpu.get("process_list")
        if (
            raw_gpu.get("gpu") != GPU_INDEX
            or not isinstance(processes, list)
            or len(processes) != 1
            or not isinstance(processes[0], dict)
        ):
            fail(f"{label} AMD SMI raw JSON does not isolate GPU 2 to one process")
        info = processes[0].get("process_info")
        if not isinstance(info, dict) or not isinstance(info.get("mem_usage"), dict):
            fail(f"{label} AMD SMI raw process_info/mem_usage is missing")
        memory = info["mem_usage"]
        if (
            info.get("pid") != pid
            or memory.get("value") != mem_usage
            or memory.get("unit") != "B"
            or isinstance(info.get("pid"), bool)
            or isinstance(memory.get("value"), bool)
        ):
            fail(f"{label} parsed AMD SMI worker VRAM differs from recorded fields")
        integer(info["pid"], f"{label}.gpu.raw.pid", minimum=1, maximum=U32_MAX)
        integer(memory["value"], f"{label}.gpu.raw.mem_usage.value", minimum=1)
        return (
            {
                "rss": vmrss_bytes,
                "vram": mem_usage,
                "threads": threads,
                "fds": fd_count,
                "children": len(parsed_children),
            },
            kfd_snapshot.completed_ns,
        )

    def _finish_resource_point(self, point: ResourcePoint) -> None:
        medians = {
            key: median(sample[key] for sample in point.samples)
            for key in ("rss", "vram", "threads", "fds", "children")
        }
        if point.phase == "baseline":
            if self.baseline is not None:
                fail("resource baseline appears more than once")
            self.baseline = medians
            self.stage = "resource_measured"
            return
        if self.baseline is None or point.release is None:
            fail("post-release point appears before the baseline")
        expected_index = len(self.rss_points) + 1
        if point.release.request_index != expected_index:
            fail("post-release resource points are not ordered 1..100")
        self.rss_points.append(medians["rss"])
        self.vram_points.append(medians["vram"])
        for key in ("threads", "fds", "children"):
            if medians[key] != self.baseline[key]:
                self.gate_errors.append(
                    f"resource request {expected_index} {key} median "
                    f"{fraction_json(medians[key])} differs from baseline "
                    f"{fraction_json(self.baseline[key])}"
                )

    def _consume_gpu_metric(self, record: dict[str, Any], boundary: str, label: str) -> None:
        exact_fields(record, GPU_METRIC_FIELDS, f"{label} gpu_metric")
        if record["record_type"] != "gpu_metric" or record["boundary"] != boundary:
            fail(f"{label} is not the scheduled {boundary} GPU metric")
        captured = timestamp(record["captured_monotonic_ns"], f"{label}.captured_monotonic_ns")
        if captured <= self.last_record_time_ns:
            fail(f"{label} GPU metric timestamp is not ordered")
        raw_json = record["raw_json"]
        if not isinstance(raw_json, str):
            fail(f"{label}.raw_json must be a string")
        if sha256_value(record["raw_sha256"], f"{label}.raw_sha256") != sha256_text(raw_json):
            fail(f"{label} GPU metric raw SHA-256 differs")
        document = decode_json(raw_json, f"{label}.raw_json")
        if not isinstance(document, dict):
            fail(f"{label} GPU metric raw root must be an object")
        gpu_data = document.get("gpu_data")
        if (
            not isinstance(gpu_data, list)
            or len(gpu_data) != 1
            or not isinstance(gpu_data[0], dict)
            or gpu_data[0].get("gpu") != GPU_INDEX
        ):
            fail(f"{label} GPU metric raw JSON does not contain exactly GPU 2")
        self.last_record_time_ns = captured
        self.gpu_metric_count += 1

    def _consume_shutdown(self, record: dict[str, Any], label: str) -> None:
        exact_fields(record, COMMAND_BASE_FIELDS, f"{label} shutdown")
        if (
            record["record_type"] != "command"
            or record["command_type"] != "shutdown"
            or record["phase"] != "shutdown"
            or record["request_index"] is not None
            or record["request_id"] is not None
        ):
            fail(f"{label} is not the frozen idle shutdown command")
        raw_command = self._decode_command_raw(record, label)
        exact_fields(raw_command, {"schema_version", "type"}, f"{label}.raw_json")
        if not json_type_equal(
            raw_command,
            {"schema_version": WORKER_SCHEMA_VERSION, "type": "shutdown"},
        ):
            fail(f"{label} command summary differs from the complete raw shutdown object")
        started, _ = self._validate_command_common(record, "shutdown", label)
        self.shutdown_write_started_ns = started
        self.command_count += 1

    def _consume_process_exit(self, record: dict[str, Any], label: str) -> None:
        exact_fields(record, PROCESS_EXIT_FIELDS, f"{label} process_exit")
        if record["record_type"] != "process_exit":
            fail(f"{label} is not process_exit")
        stdout_eof = timestamp(record["stdout_eof_monotonic_ns"], f"{label}.stdout_eof_monotonic_ns")
        exit_observed = timestamp(record["exit_observed_monotonic_ns"], f"{label}.exit_observed_monotonic_ns")
        if self.shutdown_write_started_ns is None:
            fail(f"{label} has no matching shutdown write start")
        if (
            stdout_eof < self.shutdown_write_started_ns
            or exit_observed < stdout_eof
            or exit_observed < self.last_command_ns
        ):
            fail(f"{label} EOF/exit timestamps precede shutdown")
        shutdown_duration = exit_observed - self.shutdown_write_started_ns
        if shutdown_duration > THRESHOLDS["shutdown_max_ns"]:
            self.gate_errors.append(
                f"shutdown duration {shutdown_duration} exceeds "
                f"{THRESHOLDS['shutdown_max_ns']} ns"
            )
        if record["exit_code"] != 0 or isinstance(record["exit_code"], bool):
            fail(f"{label} worker exit code must be zero")
        final_git_commit = record["final_git_commit"]
        if (
            not isinstance(final_git_commit, str)
            or GIT_COMMIT_RE.fullmatch(final_git_commit) is None
            or final_git_commit != self.expected_commit
        ):
            fail(f"{label} final git commit differs from the frozen build identity")
        final_git_status_raw = validated_raw_text(
            record["final_git_status_raw"],
            record["final_git_status_raw_sha256"],
            f"{label}.final_git_status_raw",
        )
        validate_git_status_text(
            final_git_status_raw, f"{label}.final_git_status_raw"
        )
        stderr_name = record["stderr_file"]
        if (
            not isinstance(stderr_name, str)
            or stderr_name != "worker-stderr.jsonl"
            or Path(stderr_name).name != stderr_name
            or stderr_name in {".", ".."}
        ):
            fail(f"{label}.stderr_file must name a regular sibling file")
        stderr_path = regular_file(self.raw_path.parent / stderr_name, "worker stderr")
        try:
            same_as_raw = os.path.samefile(stderr_path, self.raw_path)
        except OSError as error:
            fail(f"failed to compare worker stderr and raw evidence identity: {error}")
        if same_as_raw:
            fail("worker stderr file must differ from the raw JSONL")
        expected_sha = sha256_value(record["stderr_sha256"], f"{label}.stderr_sha256")
        actual_sha = sha256_file(stderr_path)
        if expected_sha != actual_sha:
            fail(f"{label} worker stderr SHA-256 differs")
        self.stderr_path = stderr_path
        self.stderr_sha256 = actual_sha
        self.process_exit = {
            "stdout_eof_monotonic_ns": stdout_eof,
            "exit_observed_monotonic_ns": exit_observed,
            "exit_code": 0,
            "shutdown_duration_ns": shutdown_duration,
            "final_git_commit": final_git_commit,
        }

    def finish(self, raw_sha256: str) -> dict[str, Any]:
        if self.stage != "done" or self.process_exit is None:
            fail("raw evidence ended before the final process_exit")
        if self.active is not None or self.resource_point is not None:
            fail("raw evidence ended with active request/resource state")
        if self.pending_isolation is not None:
            fail("raw evidence ended before a required isolation check")
        if self.pending_latency_recovery_id is not None:
            fail("raw evidence ended before latency recovery")
        if len(self.latency_warmup_bounds) != 2 or len(self.latency_measured_bounds) != 10:
            fail("cancel latency evidence does not contain exact 2+10 samples")
        if self.resource_sample_count != 505 or len(self.rss_points) != 100 or len(self.vram_points) != 100:
            fail("resource evidence does not contain baseline + 100*5 samples")
        if self.baseline is None:
            fail("resource evidence has no baseline")
        if self.gpu_metric_count != 2:
            fail("resource evidence must contain before/after GPU metrics")
        if self.isolation_check_count != 135:
            fail(
                "worker evidence must contain Ready plus 134 release isolation checks, "
                f"got {self.isolation_check_count}"
            )
        if self.kfd_snapshot_count != 641:
            fail(
                "worker evidence must contain preflight + 135 isolation + 505 resource "
                f"KFD snapshots, got {self.kfd_snapshot_count}"
            )
        if self.command_count != 169:
            fail(f"request matrix must produce exactly 169 commands, got {self.command_count}")
        if self.release_count != 134:
            fail(f"request matrix must produce exactly 134 releases, got {self.release_count}")
        if len(self.all_cancel_bounds) != 34:
            fail(f"request matrix must produce exactly 34 cancellations, got {len(self.all_cancel_bounds)}")
        if len(self.non_latency_warmup_cancel_bounds) != 32:
            fail(
                "request matrix must produce exactly 32 non-latency-warmup cancellation bounds"
            )

        cancel_p50 = percentile(self.latency_measured_bounds, Fraction(1, 2))
        cancel_p95 = percentile(self.latency_measured_bounds, Fraction(95, 100))
        if cancel_p95 > THRESHOLDS["cancel_p95_max_ns"]:
            self.gate_errors.append(
                f"cancel upper-bound p95 {fraction_json(cancel_p95)} exceeds {THRESHOLDS['cancel_p95_max_ns']} ns"
            )

        rss_slope = theil_sen(self.rss_points)
        vram_slope = theil_sen(self.vram_points)
        rss_delta = self.rss_points[-1] - self.baseline["rss"]
        vram_delta = self.vram_points[-1] - self.baseline["vram"]
        for label, value in (("worker RSS", rss_slope), ("primary VRAM", vram_slope)):
            if value > THRESHOLDS["theil_sen_max_bytes_per_request"]:
                self.gate_errors.append(
                    f"{label} Theil-Sen slope {fraction_json(value)} exceeds "
                    f"{THRESHOLDS['theil_sen_max_bytes_per_request']} B/request"
                )
        for label, value in (("worker RSS", rss_delta), ("primary VRAM", vram_delta)):
            if value > THRESHOLDS["final_delta_max_bytes"]:
                self.gate_errors.append(
                    f"{label} final delta {fraction_json(value)} exceeds {THRESHOLDS['final_delta_max_bytes']} bytes"
                )

        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "passed": not self.gate_errors,
            "gate_errors": self.gate_errors,
            "build_identity": {
                "git_commit": self.expected_commit,
                "tracked_clean": True,
                "binary_sha256": self.expected_binary_sha256,
            },
            "counts": {
                "commands": self.command_count,
                "worker_events": self.worker_event_count,
                "releases": self.release_count,
                "resource_samples": self.resource_sample_count,
                "resource_points": len(self.rss_points),
                "gpu_metrics": self.gpu_metric_count,
                "isolation_checks": self.isolation_check_count,
                "kfd_snapshots": self.kfd_snapshot_count,
                "kfd_attempts": self.kfd_attempt_count,
                "kfd_retries": self.kfd_retry_count,
                "all_cancellations": len(self.all_cancel_bounds),
                "non_latency_warmup_cancellations": len(
                    self.non_latency_warmup_cancel_bounds
                ),
                "theil_sen_pairs_per_series": 4_950,
            },
            "cancellation": {
                "warmup_samples": len(self.latency_warmup_bounds),
                "measured_samples": len(self.latency_measured_bounds),
                "all_samples": len(self.all_cancel_bounds),
                "non_latency_warmup_samples": len(
                    self.non_latency_warmup_cancel_bounds
                ),
                "measured_upper_bound_p50_ns": fraction_json(cancel_p50),
                "measured_upper_bound_p95_ns": fraction_json(cancel_p95),
                "maximum_upper_bound_ns": max(self.all_cancel_bounds),
            },
            "resources": {
                "baseline_worker_rss_bytes": fraction_json(self.baseline["rss"]),
                "baseline_vram_bytes": fraction_json(self.baseline["vram"]),
                "final_worker_rss_delta_bytes": fraction_json(rss_delta),
                "final_vram_delta_bytes": fraction_json(vram_delta),
                "worker_rss_theil_sen_bytes_per_request": fraction_json(rss_slope),
                "vram_theil_sen_bytes_per_request": fraction_json(vram_slope),
            },
            "process": self.process_exit,
            "evidence": {
                "raw_file": str(self.raw_path),
                "raw_sha256": raw_sha256,
                "stderr_file": str(self.stderr_path),
                "stderr_sha256": self.stderr_sha256,
            },
        }


def iter_jsonl(handle: BinaryIO, digest: Any) -> Iterable[tuple[int, dict[str, Any]]]:
    line_number = 0
    while True:
        raw = handle.readline(MAX_LINE_BYTES + 2)
        if not raw:
            break
        line_number += 1
        if len(raw) > MAX_LINE_BYTES + 1:
            fail(f"line {line_number} exceeds the 8 MiB limit")
        if not raw.endswith(b"\n"):
            fail(f"line {line_number} is not LF terminated")
        payload = raw[:-1]
        if len(payload) > MAX_LINE_BYTES or payload.endswith(b"\r"):
            fail(f"line {line_number} violates the JSONL framing contract")
        digest.update(raw)
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            fail(f"line {line_number} is not valid UTF-8: {error}")
        value = decode_json(text, f"line {line_number}")
        if not isinstance(value, dict):
            fail(f"line {line_number} root must be an object")
        yield line_number, value
    if line_number == 0:
        fail("raw JSONL is empty")


def validate_evidence(
    raw_path: Path, expected_git_commit: str, expected_binary_sha256: str
) -> dict[str, Any]:
    if GIT_COMMIT_RE.fullmatch(expected_git_commit) is None:
        fail("expected git commit must be 40 lowercase hexadecimal characters")
    if SHA256_RE.fullmatch(expected_binary_sha256) is None:
        fail("expected worker binary SHA-256 must be 64 lowercase hexadecimal characters")
    canonical = regular_file(raw_path, "raw evidence")
    if canonical.name != "raw.jsonl":
        fail("successful raw evidence basename must be raw.jsonl")
    validator = AcceptanceValidator(canonical, expected_git_commit, expected_binary_sha256)
    digest = hashlib.sha256()
    try:
        with canonical.open("rb") as handle:
            for line_number, record in iter_jsonl(handle, digest):
                validator.consume(record, line_number)
    except OSError as error:
        fail(f"failed to read raw evidence {canonical}: {error}")
    return validator.finish(digest.hexdigest())


def write_json_create_new(path: Path, value: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        fail(f"failed to create validation output {path}: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_evidence", type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-binary-sha256", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validation = validate_evidence(
            args.raw_evidence,
            args.expected_git_commit,
            args.expected_binary_sha256,
        )
        if args.output is not None:
            write_json_create_new(args.output, validation)
    except ValidationError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1
    json.dump(validation, sys.stdout, ensure_ascii=True, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
