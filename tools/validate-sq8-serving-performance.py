#!/usr/bin/env python3
"""Independently validate SQ8 fixed-M8 serving performance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any


INPUT_SCHEMA_VERSION = "ullm.sq8.serving_performance.raw.v1"
RESULT_SCHEMA_VERSION = "ullm.sq8.serving_performance_validation.v1"
PREFILL_MODE = "m8-chunk8"
PREFILL_CHUNK_TOKENS = 8
PREFILL_IMPLEMENTATION = "sq8.fixed-m8-cached-prefix.v1"
WARMUP_RUNS = 2
MEASURED_RUNS = 5
PERCENTILE_METHOD = "linear_interpolation_rank_(n-1)*p"
VRAM_POLICY = "record_and_cross_check_each_sample_no_stability_gate"
PROMPT_TOKEN_PATTERN = "ascending_u32_1_through_prompt_tokens"
TTFT_MAX_NEW_TOKENS = 512
TTFT_LIMITS = {
    32: (2.5, 3.0),
    128: (4.0, 5.0),
    512: (10.0, 12.0),
    2_048: (30.0, 35.0),
    3_584: (50.0, 60.0),
}
DECODE_PROMPT_TOKENS = 32
DECODE_GENERATED_TOKENS = 64
DECODE_TIMED_TOKENS = DECODE_GENERATED_TOKENS - 1
DECODE_EXECUTION_CALLS = DECODE_PROMPT_TOKENS // PREFILL_CHUNK_TOKENS + DECODE_TIMED_TOKENS
DECODE_P50_TOKENS_PER_SECOND_MINIMUM = 15.0
DECODE_P95_INTER_TOKEN_SECONDS_MAXIMUM = 0.100
CONTEXT_TOKENS = 4_096
STACK_LAYERS = 40
BLOCK_TOKENS = 16
CACHE_BLOCKS = 256
VOCAB_SIZE = 151_936
KV_CACHE_BYTES = 1_342_177_280
AMD_SMI_GPU_INDEX = 2
R9700_KFD_GPU_ID = 51_545
R9700_BDF = "0000:47:00.0"
R9700_UUID = "a8ff7551-0000-1000-80e9-ddefa2d60f55"
EOS_TOKEN_IDS = [151_645, 151_643]
REQUIRED_AMD_SMI_VERSION_LINES = (
    "AMDSMI Tool: 26.2.2+e1a6bc5663",
    "AMDSMI Library version: 26.2.2",
    "ROCm version: 7.2.1",
)
EXPECTED_ARTIFACT_SHA256 = (
    "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147"
)
EXPECTED_PACKAGE_SHA256 = (
    "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb"
)
MAX_JSON_BYTES = 64 * 1024 * 1024
GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
U64_MAX = (1 << 64) - 1
U32_MAX = (1 << 32) - 1


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


def regular_json_file(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        fail(f"failed to stat {label} {path}: {error}")
    if not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} must be a regular file, not a symlink: {path}")
    if metadata.st_size <= 0 or metadata.st_size > MAX_JSON_BYTES:
        fail(
            f"{label} size must be in 1..={MAX_JSON_BYTES}: "
            f"path={path} bytes={metadata.st_size}"
        )
    try:
        return path.resolve(strict=True)
    except OSError as error:
        fail(f"failed to resolve {label} {path}: {error}")


def load_json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    canonical = regular_json_file(path, label)
    try:
        with canonical.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"failed to read {label} {canonical}: {error}")
    if not isinstance(value, dict):
        fail(f"{label} JSON root must be an object")
    return canonical, value


def load_embedded_json(value: Any, label: str) -> Any:
    if not isinstance(value, str) or not value:
        fail(f"{label} must be a nonempty JSON string")
    try:
        return json.loads(value, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        fail(f"failed to parse {label}: {error}")


def sha256_bytes(value: str) -> str:
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


def ascending_prompt_sha256(prompt_tokens: int) -> str:
    digest = hashlib.sha256()
    for token_id in range(1, prompt_tokens + 1):
        digest.update(token_id.to_bytes(4, "little"))
    return digest.hexdigest()


def integer(value: Any, label: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        fail(f"{label} is outside the permitted range")
    return value


def timestamp(value: Any, label: str) -> int:
    return integer(value, label, maximum=U64_MAX)


def finite_number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        fail(f"{label} must be finite and at least {minimum}")
    return result


def sha256_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-256")
    return value


def require_timestamp_order(earlier: int, later: int, label: str) -> None:
    if later <= earlier:
        fail(f"{label} timestamps are not strictly increasing")


def validate_expected_build_identity(git_commit: str, binary_sha256: str) -> None:
    if GIT_COMMIT_RE.fullmatch(git_commit) is None:
        fail("expected runner git commit must be 40 lowercase hexadecimal characters")
    if SHA256_RE.fullmatch(binary_sha256) is None:
        fail("expected binary SHA-256 must be 64 lowercase hexadecimal characters")


def validate_build_identity(
    result: dict[str, Any], expected_git_commit: str, expected_binary_sha256: str, label: str
) -> dict[str, Any]:
    git_commit = result.get("runner_git_commit")
    binary_sha256 = result.get("runner_binary_sha256")
    if (
        not isinstance(git_commit, str)
        or GIT_COMMIT_RE.fullmatch(git_commit) is None
        or git_commit != expected_git_commit
        or result.get("runner_worktree_clean") is not True
        or not isinstance(binary_sha256, str)
        or SHA256_RE.fullmatch(binary_sha256) is None
        or binary_sha256 != expected_binary_sha256
    ):
        fail(f"{label} does not match the required clean runner build identity")
    return {
        "runner_git_commit": git_commit,
        "runner_worktree_clean": True,
        "runner_binary_sha256": binary_sha256,
    }


def validate_device(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    arch = value.get("gcn_arch_name")
    memory = integer(value.get("total_global_mem"), f"{label}.total_global_mem")
    if (
        integer(value.get("device_id"), f"{label}.device_id") != 0
        or value.get("backend") != "hip"
        or value.get("name") != "AMD Radeon Graphics"
        or not isinstance(arch, str)
        or (arch and arch.split(":", 1)[0].lower() != "gfx1201")
        or integer(value.get("compute_major"), f"{label}.compute_major") != 12
        or integer(value.get("compute_minor"), f"{label}.compute_minor") != 0
        or not (30 * 1024**3 <= memory <= 34 * 1024**3)
    ):
        fail(f"{label} is not the isolated R9700 identity")


def validate_timer_and_sampling(result: dict[str, Any], label: str) -> None:
    timer = result.get("timer")
    sampling = result.get("sampling")
    if not isinstance(timer, dict) or not isinstance(sampling, dict):
        fail(f"{label} timer and sampling contracts must be objects")
    if timer != {
        "clock": "std::time::Instant_monotonic",
        "ttft_start": "immediately_before_session.start",
        "ttft_end": "immediately_after_first_token_return_before_snapshot",
        "fixture_construction_included": False,
        "model_load_included": False,
        "cleanup_included": False,
    }:
        fail(f"{label}.timer differs from the frozen TTFT timer contract")
    if (
        sampling.get("method") != "greedy_temperature_zero"
        or finite_number(sampling.get("temperature"), f"{label}.sampling.temperature") != 0.0
        or finite_number(sampling.get("top_p"), f"{label}.sampling.top_p") != 1.0
        or integer(sampling.get("top_k"), f"{label}.sampling.top_k") != 20
        or integer(sampling.get("seed"), f"{label}.sampling.seed") != 0
        or sampling.get("eos_token_ids") != EOS_TOKEN_IDS
    ):
        fail(f"{label}.sampling differs from the frozen greedy sampling contract")


def validate_environment(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    version_raw = value.get("amd_smi_version_raw")
    if not isinstance(version_raw, str) or not version_raw:
        fail(f"{label}.amd_smi_version_raw must be nonempty")
    version_sha = sha256_value(
        value.get("amd_smi_version_raw_sha256"), f"{label}.amd_smi_version_raw_sha256"
    )
    if version_sha != sha256_bytes(version_raw):
        fail(f"{label} AMD SMI version raw SHA-256 differs")
    for required in REQUIRED_AMD_SMI_VERSION_LINES:
        if required not in version_raw:
            fail(f"{label} AMD SMI version is missing {required!r}")

    list_raw = value.get("amd_smi_list_raw_json")
    list_sha = sha256_value(
        value.get("amd_smi_list_raw_sha256"), f"{label}.amd_smi_list_raw_sha256"
    )
    if not isinstance(list_raw, str) or list_sha != sha256_bytes(list_raw):
        fail(f"{label} AMD SMI list raw SHA-256 differs")
    document = load_embedded_json(list_raw, f"{label}.amd_smi_list_raw_json")
    if not isinstance(document, list):
        fail(f"{label} AMD SMI list root must be an array")
    matches = [
        item
        for item in document
        if isinstance(item, dict)
        and integer(item.get("gpu"), f"{label}.amd_smi_list.gpu") == AMD_SMI_GPU_INDEX
    ]
    if len(matches) != 1:
        fail(f"{label} AMD SMI list must contain exactly one GPU 2 entry")
    target = matches[0]
    if (
        target.get("bdf") != R9700_BDF
        or target.get("uuid") != R9700_UUID
        or integer(target.get("kfd_id"), f"{label}.amd_smi_list.kfd_id")
        != R9700_KFD_GPU_ID
        or value.get("hip_visible_devices") != "1"
        or integer(value.get("target_gpu_index"), f"{label}.target_gpu_index")
        != AMD_SMI_GPU_INDEX
        or value.get("target_gpu_bdf") != R9700_BDF
        or value.get("target_gpu_uuid") != R9700_UUID
        or integer(value.get("target_kfd_gpu_id"), f"{label}.target_kfd_gpu_id")
        != R9700_KFD_GPU_ID
    ):
        fail(f"{label} does not bind the frozen R9700 environment")
    return {
        "hip_visible_devices": "1",
        "amd_smi_version_raw_sha256": version_sha,
        "amd_smi_list_raw_sha256": list_sha,
        "target_gpu_index": AMD_SMI_GPU_INDEX,
        "target_gpu_bdf": R9700_BDF,
        "target_gpu_uuid": R9700_UUID,
        "target_kfd_gpu_id": R9700_KFD_GPU_ID,
    }


def validate_snapshot(
    value: Any,
    label: str,
    *,
    status: str,
    request_id: str | None,
    prompt_tokens: int,
    generated_tokens: int,
    cache_len: int,
) -> None:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    active = request_id is not None
    cache_lengths = value.get("cache_lengths")
    if (
        value.get("status") != status
        or value.get("active_request_id") != request_id
        or integer(value.get("prompt_tokens"), f"{label}.prompt_tokens") != prompt_tokens
        or integer(
            value.get("prompt_tokens_processed"), f"{label}.prompt_tokens_processed"
        )
        != prompt_tokens
        or integer(value.get("generated_tokens"), f"{label}.generated_tokens")
        != generated_tokens
        or not isinstance(cache_lengths, list)
        or len(cache_lengths) != STACK_LAYERS
        or any(integer(item, f"{label}.cache_lengths") != cache_len for item in cache_lengths)
        or integer(value.get("scheduler_active"), f"{label}.scheduler_active")
        != (1 if active else 0)
        or integer(value.get("scheduler_waiting"), f"{label}.scheduler_waiting") != 0
        or integer(value.get("block_size_tokens"), f"{label}.block_size_tokens")
        != BLOCK_TOKENS
        or integer(value.get("total_blocks"), f"{label}.total_blocks") != CACHE_BLOCKS
        or integer(value.get("free_blocks"), f"{label}.free_blocks")
        != (0 if active else CACHE_BLOCKS)
        or integer(value.get("allocated_blocks"), f"{label}.allocated_blocks")
        != (CACHE_BLOCKS if active else 0)
        or integer(value.get("free_runs"), f"{label}.free_runs") != (0 if active else 1)
        or integer(value.get("largest_free_run"), f"{label}.largest_free_run")
        != (0 if active else CACHE_BLOCKS)
    ):
        fail(f"{label} snapshot contract differs")


def validate_ready_snapshot(value: Any, label: str) -> None:
    validate_snapshot(
        value,
        label,
        status="ready",
        request_id=None,
        prompt_tokens=0,
        generated_tokens=0,
        cache_len=0,
    )


def validate_vram_capture(
    value: Any, label: str, expected_worker_pid: int | None
) -> dict[str, int]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    command_start = timestamp(
        value.get("amd_smi_command_start_elapsed_ns"),
        f"{label}.amd_smi_command_start_elapsed_ns",
    )
    command_end = timestamp(
        value.get("amd_smi_command_end_elapsed_ns"),
        f"{label}.amd_smi_command_end_elapsed_ns",
    )
    captured_ns = timestamp(value.get("captured_elapsed_ns"), f"{label}.captured_elapsed_ns")
    require_timestamp_order(command_start, command_end, f"{label} AMD SMI command")
    require_timestamp_order(command_end, captured_ns, f"{label} VRAM capture")
    worker_pid = integer(value.get("worker_pid"), f"{label}.worker_pid", minimum=1, maximum=U32_MAX)
    if expected_worker_pid is not None and worker_pid != expected_worker_pid:
        fail(f"{label} worker PID differs across captures")
    gpu_index = integer(value.get("amd_smi_gpu_index"), f"{label}.amd_smi_gpu_index")
    memory_bytes = integer(
        value.get("amd_smi_mem_usage_bytes"), f"{label}.amd_smi_mem_usage_bytes", minimum=1
    )
    raw = value.get("amd_smi_process_raw_json")
    raw_sha = sha256_value(
        value.get("amd_smi_process_raw_sha256"), f"{label}.amd_smi_process_raw_sha256"
    )
    if not isinstance(raw, str) or raw_sha != sha256_bytes(raw):
        fail(f"{label} AMD SMI process raw SHA-256 differs")
    document = load_embedded_json(raw, f"{label}.amd_smi_process_raw_json")
    if not isinstance(document, list) or len(document) != 1 or not isinstance(document[0], dict):
        fail(f"{label} AMD SMI process JSON must contain one GPU object")
    gpu = document[0]
    processes = gpu.get("process_list")
    if not isinstance(processes, list) or len(processes) != 1 or not isinstance(processes[0], dict):
        fail(f"{label} AMD SMI process JSON must contain one process")
    info = processes[0].get("process_info")
    if not isinstance(info, dict):
        fail(f"{label} AMD SMI process_info must be an object")
    memory = info.get("mem_usage")
    if not isinstance(memory, dict):
        fail(f"{label} AMD SMI mem_usage must be an object")
    if (
        integer(gpu.get("gpu"), f"{label}.raw.gpu") != AMD_SMI_GPU_INDEX
        or integer(info.get("pid"), f"{label}.raw.pid", minimum=1, maximum=U32_MAX)
        != worker_pid
        or integer(memory.get("value"), f"{label}.raw.mem_usage.value", minimum=1)
        != memory_bytes
        or memory.get("unit") != "B"
        or gpu_index != AMD_SMI_GPU_INDEX
        or integer(value.get("kfd_gpu_id"), f"{label}.kfd_gpu_id") != R9700_KFD_GPU_ID
        or integer(value.get("kfd_vram_bytes"), f"{label}.kfd_vram_bytes", minimum=1)
        != memory_bytes
    ):
        fail(f"{label} AMD SMI/KFD VRAM identity differs")

    kfd_processes = value.get("kfd_processes")
    if not isinstance(kfd_processes, list) or not kfd_processes:
        fail(f"{label}.kfd_processes must be nonempty")
    parsed_processes: list[tuple[int, int]] = []
    for index, process in enumerate(kfd_processes):
        process_label = f"{label}.kfd_processes[{index}]"
        if not isinstance(process, dict):
            fail(f"{process_label} must be an object")
        parsed_processes.append(
            (
                integer(process.get("pid"), f"{process_label}.pid", minimum=1, maximum=U32_MAX),
                integer(process.get("vram_bytes"), f"{process_label}.vram_bytes"),
            )
        )
    if parsed_processes != sorted(parsed_processes) or len(
        {pid for pid, _ in parsed_processes}
    ) != len(parsed_processes):
        fail(f"{label}.kfd_processes must have unique ascending PIDs")
    own = [amount for pid, amount in parsed_processes if pid == worker_pid]
    unrelated = [pid for pid, amount in parsed_processes if pid != worker_pid and amount > 0]
    if own != [memory_bytes] or unrelated or value.get("unrelated_positive_kfd_pids") != []:
        fail(f"{label} does not prove isolated KFD VRAM ownership")
    return {
        "command_start_ns": command_start,
        "command_end_ns": command_end,
        "timestamp_ns": captured_ns,
        "worker_pid": worker_pid,
        "vram_bytes": memory_bytes,
    }


def nested_object(root: dict[str, Any], keys: tuple[str, ...], label: str) -> dict[str, Any]:
    value: Any = root
    for key in keys:
        if not isinstance(value, dict) or not isinstance(value.get(key), dict):
            fail(f"{label} is missing {'.'.join(keys)}")
        value = value[key]
    return value


def metric_quantity(
    gpu: dict[str, Any], keys: tuple[str, ...], expected_unit: str, label: str
) -> float:
    quantity = nested_object(gpu, keys, label)
    result = finite_number(quantity.get("value"), f"{label}.value")
    if quantity.get("unit") != expected_unit:
        fail(f"{label}.unit must be {expected_unit}")
    return result


def validate_metric_capture(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    command_start = timestamp(
        value.get("command_start_elapsed_ns"), f"{label}.command_start_elapsed_ns"
    )
    command_end = timestamp(value.get("command_end_elapsed_ns"), f"{label}.command_end_elapsed_ns")
    captured_ns = timestamp(value.get("captured_elapsed_ns"), f"{label}.captured_elapsed_ns")
    require_timestamp_order(command_start, command_end, f"{label} AMD SMI command")
    if captured_ns != command_end:
        fail(f"{label}.captured_elapsed_ns must equal command_end_elapsed_ns")
    raw = value.get("raw_json")
    raw_sha = sha256_value(value.get("raw_sha256"), f"{label}.raw_sha256")
    if not isinstance(raw, str) or raw_sha != sha256_bytes(raw):
        fail(f"{label} AMD SMI metric raw SHA-256 differs")
    document = load_embedded_json(raw, f"{label}.raw_json")
    if not isinstance(document, dict):
        fail(f"{label} AMD SMI metric root must be an object")
    gpu_data = document.get("gpu_data")
    if not isinstance(gpu_data, list) or not gpu_data or not isinstance(gpu_data[0], dict):
        fail(f"{label} AMD SMI metric is missing gpu_data[0]")
    gpu = gpu_data[0]
    summaries = {
        "hotspot_temperature_c": metric_quantity(gpu, ("temperature", "hotspot"), "C", label),
        "socket_power_w": metric_quantity(gpu, ("power", "socket_power"), "W", label),
        "gfx_clock_mhz": metric_quantity(gpu, ("clock", "gfx_0", "clk"), "MHz", label),
        "memory_clock_mhz": metric_quantity(gpu, ("clock", "mem_0", "clk"), "MHz", label),
        "fabric_clock_mhz": metric_quantity(gpu, ("clock", "fclk_0", "clk"), "MHz", label),
    }
    if integer(gpu.get("gpu"), f"{label}.raw.gpu") != AMD_SMI_GPU_INDEX or integer(
        value.get("gpu_index"), f"{label}.gpu_index"
    ) != AMD_SMI_GPU_INDEX:
        fail(f"{label} metric does not identify GPU 2")
    for key, expected in summaries.items():
        if finite_number(value.get(key), f"{label}.{key}") != expected:
            fail(f"{label}.{key} differs from raw AMD SMI JSON")
    return {
        "command_start_ns": command_start,
        "command_end_ns": command_end,
        "timestamp_ns": captured_ns,
        "raw_sha256": raw_sha,
        **summaries,
    }


def expected_sample_order() -> list[tuple[str, int]]:
    return [("warmup", index) for index in range(WARMUP_RUNS)] + [
        ("measured", index) for index in range(MEASURED_RUNS)
    ]


def percentile(values: list[float], probability: float) -> float:
    if not values or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        fail("percentile input is invalid")
    ordered = sorted(finite_number(value, "percentile sample") for value in values)
    rank = (len(ordered) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    result = ordered[lower] + (rank - lower) * (ordered[upper] - ordered[lower])
    if not math.isfinite(result):
        fail("percentile result is non-finite")
    return result


def validate_ttft_sample(
    value: Any,
    label: str,
    *,
    prompt_tokens: int,
    phase: str,
    sample_index: int,
    expected_worker_pid: int,
    previous_timestamp_ns: int,
) -> tuple[int, int]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    request_id = f"perf-ttft-p{prompt_tokens:04d}-{phase}-{sample_index}"
    start = timestamp(value.get("request_start_elapsed_ns"), f"{label}.request_start_elapsed_ns")
    first = timestamp(value.get("first_token_elapsed_ns"), f"{label}.first_token_elapsed_ns")
    cancel_set = timestamp(value.get("cancel_set_elapsed_ns"), f"{label}.cancel_set_elapsed_ns")
    cancellation = timestamp(
        value.get("cancellation_observed_elapsed_ns"), f"{label}.cancellation_observed_elapsed_ns"
    )
    reset_start = timestamp(value.get("reset_start_elapsed_ns"), f"{label}.reset_start_elapsed_ns")
    reset_end = timestamp(value.get("reset_end_elapsed_ns"), f"{label}.reset_end_elapsed_ns")
    for earlier, later, order_label in (
        (previous_timestamp_ns, start, "sample start"),
        (start, first, "TTFT"),
        (first, cancel_set, "cancel set"),
        (cancel_set, cancellation, "cancellation"),
        (cancellation, reset_start, "reset start"),
        (reset_start, reset_end, "reset end"),
    ):
        require_timestamp_order(earlier, later, f"{label} {order_label}")
    ttft_ns = integer(value.get("ttft_ns"), f"{label}.ttft_ns", maximum=U64_MAX)
    reset_ns = integer(value.get("reset_ns"), f"{label}.reset_ns", maximum=U64_MAX)
    first_token_id = integer(value.get("first_token_id"), f"{label}.first_token_id")
    expected_calls = prompt_tokens // PREFILL_CHUNK_TOKENS
    if (
        value.get("phase") != phase
        or integer(value.get("sample_index"), f"{label}.sample_index") != sample_index
        or value.get("request_id") != request_id
        or ttft_ns != first - start
        or first_token_id >= VOCAB_SIZE
        or first_token_id in EOS_TOKEN_IDS
        or integer(value.get("first_token_cache_len"), f"{label}.first_token_cache_len")
        != prompt_tokens
        or integer(value.get("prompt_execution_calls"), f"{label}.prompt_execution_calls")
        != expected_calls
        or integer(value.get("prompt_progress_events"), f"{label}.prompt_progress_events")
        != expected_calls - 1
        or reset_ns != reset_end - reset_start
        or value.get("release_outcome") != "cancelled"
        or integer(value.get("release_generated_tokens"), f"{label}.release_generated_tokens")
        != 1
        or value.get("release_reset_complete") is not True
    ):
        fail(f"{label} TTFT execution contract differs")
    validate_snapshot(
        value.get("first_token_snapshot"),
        f"{label}.first_token_snapshot",
        status="decoding",
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        generated_tokens=1,
        cache_len=prompt_tokens,
    )
    validate_snapshot(
        value.get("cancellation_snapshot"),
        f"{label}.cancellation_snapshot",
        status="cancelling",
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        generated_tokens=1,
        cache_len=prompt_tokens,
    )
    validate_ready_snapshot(value.get("post_reset_snapshot"), f"{label}.post_reset_snapshot")
    vram = validate_vram_capture(
        value.get("vram_after_reset"), f"{label}.vram_after_reset", expected_worker_pid
    )
    require_timestamp_order(reset_end, vram["command_start_ns"], f"{label} post-reset VRAM")
    return vram["timestamp_ns"], ttft_ns


def validate_ttft_case(
    value: Any,
    label: str,
    *,
    prompt_tokens: int,
    expected_worker_pid: int,
    previous_timestamp_ns: int,
    gate_errors: list[str],
) -> tuple[int, dict[str, Any]]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    p50_limit, p95_limit = TTFT_LIMITS[prompt_tokens]
    if (
        integer(value.get("prompt_tokens"), f"{label}.prompt_tokens") != prompt_tokens
        or integer(value.get("max_new_tokens"), f"{label}.max_new_tokens")
        != TTFT_MAX_NEW_TOKENS
        or value.get("prompt_token_pattern") != PROMPT_TOKEN_PATTERN
        or sha256_value(
            value.get("prompt_token_ids_u32_le_sha256"),
            f"{label}.prompt_token_ids_u32_le_sha256",
        )
        != ascending_prompt_sha256(prompt_tokens)
        or finite_number(value.get("p50_limit_seconds"), f"{label}.p50_limit_seconds")
        != p50_limit
        or finite_number(value.get("p95_limit_seconds"), f"{label}.p95_limit_seconds")
        != p95_limit
    ):
        fail(f"{label} TTFT case contract differs")
    before = validate_metric_capture(value.get("metric_before"), f"{label}.metric_before")
    require_timestamp_order(
        previous_timestamp_ns, before["command_start_ns"], f"{label} metric before"
    )
    samples = value.get("samples")
    order = expected_sample_order()
    if not isinstance(samples, list) or len(samples) != len(order):
        fail(f"{label}.samples must contain two warmups and five measured runs")
    measured: list[float] = []
    current = before["timestamp_ns"]
    for index, ((phase, sample_index), sample) in enumerate(zip(order, samples, strict=True)):
        current, ttft_ns = validate_ttft_sample(
            sample,
            f"{label}.samples[{index}]",
            prompt_tokens=prompt_tokens,
            phase=phase,
            sample_index=sample_index,
            expected_worker_pid=expected_worker_pid,
            previous_timestamp_ns=current,
        )
        if phase == "measured":
            measured.append(ttft_ns / 1_000_000_000.0)
    after = validate_metric_capture(value.get("metric_after"), f"{label}.metric_after")
    require_timestamp_order(current, after["command_start_ns"], f"{label} metric after")
    p50 = percentile(measured, 0.50)
    p95 = percentile(measured, 0.95)
    errors = []
    if p50 > p50_limit:
        errors.append(f"prompt {prompt_tokens} TTFT p50 {p50:.9f}s exceeds {p50_limit:.9f}s")
    if p95 > p95_limit:
        errors.append(f"prompt {prompt_tokens} TTFT p95 {p95:.9f}s exceeds {p95_limit:.9f}s")
    gate_errors.extend(errors)
    return after["timestamp_ns"], {
        "prompt_tokens": prompt_tokens,
        "measured_ttft_seconds": measured,
        "p50_seconds": p50,
        "p95_seconds": p95,
        "p50_limit_seconds": p50_limit,
        "p95_limit_seconds": p95_limit,
        "passed": not errors,
        "metric_before": before,
        "metric_after": after,
    }


def validate_decode_sample(
    value: Any,
    label: str,
    *,
    phase: str,
    sample_index: int,
    expected_worker_pid: int,
    previous_timestamp_ns: int,
) -> tuple[int, float, list[float]]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    request_id = f"perf-decode-p0032-g0064-{phase}-{sample_index}"
    start = timestamp(value.get("request_start_elapsed_ns"), f"{label}.request_start_elapsed_ns")
    first = timestamp(value.get("first_token_elapsed_ns"), f"{label}.first_token_elapsed_ns")
    last = timestamp(value.get("last_token_elapsed_ns"), f"{label}.last_token_elapsed_ns")
    reset_start = timestamp(value.get("reset_start_elapsed_ns"), f"{label}.reset_start_elapsed_ns")
    reset_end = timestamp(value.get("reset_end_elapsed_ns"), f"{label}.reset_end_elapsed_ns")
    for earlier, later, order_label in (
        (previous_timestamp_ns, start, "sample start"),
        (start, first, "first token"),
        (first, last, "last token"),
        (last, reset_start, "reset start"),
        (reset_start, reset_end, "reset end"),
    ):
        require_timestamp_order(earlier, later, f"{label} {order_label}")
    ttft_ns = integer(value.get("ttft_ns"), f"{label}.ttft_ns", maximum=U64_MAX)
    duration_ns = integer(
        value.get("decode_duration_ns"), f"{label}.decode_duration_ns", minimum=1, maximum=U64_MAX
    )
    reset_ns = integer(value.get("reset_ns"), f"{label}.reset_ns", maximum=U64_MAX)
    if (
        value.get("phase") != phase
        or integer(value.get("sample_index"), f"{label}.sample_index") != sample_index
        or value.get("request_id") != request_id
        or ttft_ns != first - start
        or duration_ns != last - first
        or integer(value.get("prompt_execution_calls"), f"{label}.prompt_execution_calls") != 4
        or integer(value.get("prompt_progress_events"), f"{label}.prompt_progress_events") != 3
        or integer(value.get("execution_calls"), f"{label}.execution_calls")
        != DECODE_EXECUTION_CALLS
        or reset_ns != reset_end - reset_start
        or value.get("release_outcome") != "length"
        or integer(value.get("release_generated_tokens"), f"{label}.release_generated_tokens")
        != DECODE_GENERATED_TOKENS
        or value.get("release_reset_complete") is not True
    ):
        fail(f"{label} decode execution contract differs")
    generated = value.get("generated")
    if not isinstance(generated, list) or len(generated) != DECODE_GENERATED_TOKENS:
        fail(f"{label}.generated must contain exactly 64 tokens")
    availability: list[int] = []
    for index, item in enumerate(generated):
        item_label = f"{label}.generated[{index}]"
        if not isinstance(item, dict):
            fail(f"{item_label} must be an object")
        available = timestamp(
            item.get("available_elapsed_ns"), f"{item_label}.available_elapsed_ns"
        )
        expected_reason = "length" if index == DECODE_GENERATED_TOKENS - 1 else None
        token_id = integer(item.get("token_id"), f"{item_label}.token_id")
        if (
            integer(item.get("generated_index"), f"{item_label}.generated_index") != index
            or token_id >= VOCAB_SIZE
            or token_id in EOS_TOKEN_IDS
            or integer(item.get("cache_len"), f"{item_label}.cache_len")
            != DECODE_PROMPT_TOKENS + index
            or item.get("terminal_reason") != expected_reason
        ):
            fail(f"{item_label} token/cache transition differs")
        if availability:
            require_timestamp_order(availability[-1], available, f"{item_label} availability")
        availability.append(available)
    if availability[0] != first or availability[-1] != last:
        fail(f"{label} first/last availability differs from sample timestamps")
    intervals = [
        (availability[index] - availability[index - 1]) / 1_000_000_000.0
        for index in range(1, len(availability))
    ]
    validate_snapshot(
        value.get("terminal_snapshot"),
        f"{label}.terminal_snapshot",
        status="finishing",
        request_id=request_id,
        prompt_tokens=DECODE_PROMPT_TOKENS,
        generated_tokens=DECODE_GENERATED_TOKENS,
        cache_len=DECODE_PROMPT_TOKENS + DECODE_TIMED_TOKENS,
    )
    validate_ready_snapshot(value.get("post_reset_snapshot"), f"{label}.post_reset_snapshot")
    vram = validate_vram_capture(
        value.get("vram_after_reset"), f"{label}.vram_after_reset", expected_worker_pid
    )
    require_timestamp_order(reset_end, vram["command_start_ns"], f"{label} post-reset VRAM")
    throughput = DECODE_TIMED_TOKENS * 1_000_000_000.0 / duration_ns
    return vram["timestamp_ns"], throughput, intervals


def validate_decode_case(
    value: Any,
    label: str,
    *,
    expected_worker_pid: int,
    previous_timestamp_ns: int,
    gate_errors: list[str],
) -> tuple[int, dict[str, Any]]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    if (
        integer(value.get("prompt_tokens"), f"{label}.prompt_tokens") != DECODE_PROMPT_TOKENS
        or integer(value.get("max_new_tokens"), f"{label}.max_new_tokens")
        != DECODE_GENERATED_TOKENS
        or integer(value.get("generated_tokens"), f"{label}.generated_tokens")
        != DECODE_GENERATED_TOKENS
        or value.get("prompt_token_pattern") != PROMPT_TOKEN_PATTERN
        or sha256_value(
            value.get("prompt_token_ids_u32_le_sha256"),
            f"{label}.prompt_token_ids_u32_le_sha256",
        )
        != ascending_prompt_sha256(DECODE_PROMPT_TOKENS)
        or integer(value.get("decode_tokens_per_sample"), f"{label}.decode_tokens_per_sample")
        != DECODE_TIMED_TOKENS
        or finite_number(
            value.get("p50_tokens_per_second_minimum"),
            f"{label}.p50_tokens_per_second_minimum",
        )
        != DECODE_P50_TOKENS_PER_SECOND_MINIMUM
        or finite_number(
            value.get("p95_inter_token_seconds_maximum"),
            f"{label}.p95_inter_token_seconds_maximum",
        )
        != DECODE_P95_INTER_TOKEN_SECONDS_MAXIMUM
        or value.get("p95_inter_token_pooling") != "all_measured_inter_token_intervals"
    ):
        fail(f"{label} decode case contract differs")
    before = validate_metric_capture(value.get("metric_before"), f"{label}.metric_before")
    require_timestamp_order(
        previous_timestamp_ns, before["command_start_ns"], f"{label} metric before"
    )
    samples = value.get("samples")
    order = expected_sample_order()
    if not isinstance(samples, list) or len(samples) != len(order):
        fail(f"{label}.samples must contain two warmups and five measured runs")
    measured_throughputs: list[float] = []
    measured_intervals: list[float] = []
    current = before["timestamp_ns"]
    for index, ((phase, sample_index), sample) in enumerate(zip(order, samples, strict=True)):
        current, throughput, intervals = validate_decode_sample(
            sample,
            f"{label}.samples[{index}]",
            phase=phase,
            sample_index=sample_index,
            expected_worker_pid=expected_worker_pid,
            previous_timestamp_ns=current,
        )
        if phase == "measured":
            measured_throughputs.append(throughput)
            measured_intervals.extend(intervals)
    if len(measured_intervals) != MEASURED_RUNS * DECODE_TIMED_TOKENS:
        fail(f"{label} measured inter-token pool must contain 315 intervals")
    after = validate_metric_capture(value.get("metric_after"), f"{label}.metric_after")
    require_timestamp_order(current, after["command_start_ns"], f"{label} metric after")
    throughput_p50 = percentile(measured_throughputs, 0.50)
    inter_token_p95 = percentile(measured_intervals, 0.95)
    errors = []
    if throughput_p50 < DECODE_P50_TOKENS_PER_SECOND_MINIMUM:
        errors.append(
            f"decode p50 {throughput_p50:.9f} token/s is below "
            f"{DECODE_P50_TOKENS_PER_SECOND_MINIMUM:.9f} token/s"
        )
    if inter_token_p95 > DECODE_P95_INTER_TOKEN_SECONDS_MAXIMUM:
        errors.append(
            f"decode inter-token p95 {inter_token_p95:.9f}s exceeds "
            f"{DECODE_P95_INTER_TOKEN_SECONDS_MAXIMUM:.9f}s"
        )
    gate_errors.extend(errors)
    return after["timestamp_ns"], {
        "measured_tokens_per_second": measured_throughputs,
        "p50_tokens_per_second": throughput_p50,
        "p50_tokens_per_second_minimum": DECODE_P50_TOKENS_PER_SECOND_MINIMUM,
        "measured_inter_token_interval_count": len(measured_intervals),
        "p95_inter_token_seconds": inter_token_p95,
        "p95_inter_token_seconds_maximum": DECODE_P95_INTER_TOKEN_SECONDS_MAXIMUM,
        "passed": not errors,
        "metric_before": before,
        "metric_after": after,
    }


def validate_result(
    result_path: Path, expected_runner_git_commit: str, expected_binary_sha256: str
) -> dict[str, Any]:
    validate_expected_build_identity(expected_runner_git_commit, expected_binary_sha256)
    result_file, result = load_json(result_path, "performance result")
    label = str(result_file)
    build_identity = validate_build_identity(
        result, expected_runner_git_commit, expected_binary_sha256, label
    )
    if (
        result.get("schema_version") != INPUT_SCHEMA_VERSION
        or result.get("prefill_mode") != PREFILL_MODE
        or integer(result.get("prefill_chunk_tokens"), f"{label}.prefill_chunk_tokens")
        != PREFILL_CHUNK_TOKENS
        or result.get("prefill_implementation") != PREFILL_IMPLEMENTATION
        or integer(result.get("warmup_runs"), f"{label}.warmup_runs") != WARMUP_RUNS
        or integer(result.get("measured_runs"), f"{label}.measured_runs") != MEASURED_RUNS
        or result.get("percentile_method") != PERCENTILE_METHOD
        or result.get("vram_policy") != VRAM_POLICY
        or result.get("artifact_content_sha256") != EXPECTED_ARTIFACT_SHA256
        or result.get("package_manifest_sha256") != EXPECTED_PACKAGE_SHA256
        or integer(result.get("kv_cache_bytes"), f"{label}.kv_cache_bytes") != KV_CACHE_BYTES
        or integer(result.get("cache_blocks"), f"{label}.cache_blocks") != CACHE_BLOCKS
        or integer(result.get("context_tokens"), f"{label}.context_tokens") != CONTEXT_TOKENS
    ):
        fail(f"{label} has the wrong performance schema/model/runtime contract")
    finite_number(result.get("load_seconds"), f"{label}.load_seconds")
    validate_timer_and_sampling(result, label)
    validate_device(result.get("device"), f"{label}.device")
    environment = validate_environment(result.get("environment"), f"{label}.environment")
    initial_vram = validate_vram_capture(result.get("initial_vram"), f"{label}.initial_vram", None)
    worker_pid = initial_vram["worker_pid"]
    current = initial_vram["timestamp_ns"]
    gate_errors: list[str] = []

    ttft_cases = result.get("ttft_cases")
    prompt_lengths = tuple(TTFT_LIMITS)
    if not isinstance(ttft_cases, list) or len(ttft_cases) != len(prompt_lengths):
        fail(f"{label}.ttft_cases must contain the frozen five prompt lengths")
    ttft_results = []
    for index, (prompt_tokens, case) in enumerate(zip(prompt_lengths, ttft_cases, strict=True)):
        current, summary = validate_ttft_case(
            case,
            f"{label}.ttft_cases[{index}]",
            prompt_tokens=prompt_tokens,
            expected_worker_pid=worker_pid,
            previous_timestamp_ns=current,
            gate_errors=gate_errors,
        )
        ttft_results.append(summary)

    current, decode_result = validate_decode_case(
        result.get("decode_case"),
        f"{label}.decode_case",
        expected_worker_pid=worker_pid,
        previous_timestamp_ns=current,
        gate_errors=gate_errors,
    )
    final_vram = validate_vram_capture(
        result.get("final_vram"), f"{label}.final_vram", worker_pid
    )
    require_timestamp_order(current, final_vram["command_start_ns"], f"{label} final VRAM")
    validate_ready_snapshot(result.get("final_snapshot"), f"{label}.final_snapshot")

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "passed": not gate_errors,
        "gate_errors": gate_errors,
        "build_identity": build_identity,
        "environment": environment,
        "thresholds": {
            "percentile_method": PERCENTILE_METHOD,
            "ttft_seconds": {
                str(prompt): {"p50": limits[0], "p95": limits[1]}
                for prompt, limits in TTFT_LIMITS.items()
            },
            "decode_p50_tokens_per_second_minimum": DECODE_P50_TOKENS_PER_SECOND_MINIMUM,
            "decode_p95_inter_token_seconds_maximum": DECODE_P95_INTER_TOKEN_SECONDS_MAXIMUM,
        },
        "ttft_cases": ttft_results,
        "decode_case": decode_result,
        "vram": {
            "worker_pid": worker_pid,
            "initial_bytes": initial_vram["vram_bytes"],
            "final_bytes": final_vram["vram_bytes"],
        },
        "evidence": {"file": str(result_file), "sha256": sha256_file(result_file)},
    }


def write_json_create_new(path: Path, value: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        fail(f"failed to create validation output {path}: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--expected-runner-git-commit", required=True)
    parser.add_argument("--expected-binary-sha256", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validation = validate_result(
            args.result,
            args.expected_runner_git_commit,
            args.expected_binary_sha256,
        )
        if args.output is not None:
            write_json_create_new(args.output, validation)
    except ValidationError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1
    if not validation["passed"]:
        print(
            "performance gate failed: " + "; ".join(validation["gate_errors"]),
            file=sys.stderr,
        )
        return 1
    print(
        f"passed=true ttft_prompts={list(TTFT_LIMITS)} "
        f"decode_p50_tps={validation['decode_case']['p50_tokens_per_second']:.9f} "
        f"decode_p95_itl_s={validation['decode_case']['p95_inter_token_seconds']:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
