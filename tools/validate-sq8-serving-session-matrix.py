#!/usr/bin/env python3
"""Validate SQ8 serving boundary, cancellation, and exact-context evidence."""

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


INPUT_SCHEMA_VERSION = "ullm.sq8.serving_smoke.v2"
RESULT_SCHEMA_VERSION = "ullm.sq8.serving_session_matrix_validation.v1"
BOUNDARY_PROMPT_LENGTHS = (15, 16, 17, 255, 256, 257)
CONTEXT_PROMPT_LENGTH = 4_095
CONTEXT_TOKENS = 4_096
BLOCK_TOKENS = 16
CACHE_BLOCKS = 256
STACK_LAYERS = 40
VOCAB_SIZE = 151_936
KV_CACHE_BYTES = 1_342_177_280
PREFILL_CANCEL_PROGRESS = 8
DECODE_PROMPT_LENGTH = 8
EXPECTED_DECODE_G8 = (353, 10, 4_999, 1_725, 15, 16, 17, 18)
MAX_JSON_BYTES = 1024 * 1024
EXPECTED_ARTIFACT_SHA256 = (
    "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147"
)
EXPECTED_PACKAGE_SHA256 = (
    "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        fail(f"failed to hash {path}: {error}")
    return digest.hexdigest()


def integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} must be an integer")
    return value


def finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        fail(f"{label} must be finite and nonnegative")
    return result


def validate_device(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    arch = value.get("gcn_arch_name")
    memory = integer(value.get("total_global_mem"), f"{label}.total_global_mem")
    if (
        value.get("device_id") != 0
        or value.get("backend") != "hip"
        or value.get("name") != "AMD Radeon Graphics"
        or not isinstance(arch, str)
        or (arch and arch.split(":", 1)[0].lower() != "gfx1201")
        or value.get("compute_major") != 12
        or value.get("compute_minor") != 0
        or not (30 * 1024**3 <= memory <= 34 * 1024**3)
    ):
        fail(f"{label} is not the isolated R9700 identity")


def validate_common(result: dict[str, Any], label: str) -> list[dict[str, Any]]:
    if result.get("schema_version") != INPUT_SCHEMA_VERSION:
        fail(f"{label} has the wrong schema version")
    if result.get("artifact_content_sha256") != EXPECTED_ARTIFACT_SHA256:
        fail(f"{label} has the wrong artifact identity")
    if result.get("package_manifest_sha256") != EXPECTED_PACKAGE_SHA256:
        fail(f"{label} has the wrong package identity")
    validate_device(result.get("device"), f"{label}.device")
    finite_nonnegative(result.get("load_seconds"), f"{label}.load_seconds")
    if (
        result.get("kv_cache_bytes") != KV_CACHE_BYTES
        or result.get("cache_blocks") != CACHE_BLOCKS
        or result.get("context_tokens") != CONTEXT_TOKENS
        or result.get("post_reset_status") != "ready"
        or result.get("post_reset_active") != 0
        or result.get("post_reset_waiting") != 0
        or result.get("post_reset_allocated_blocks") != 0
        or result.get("post_reset_cache_lengths_all_zero") is not True
        or result.get("post_reset_cache_lengths") != [0] * STACK_LAYERS
    ):
        fail(f"{label} does not prove the reusable post-reset baseline")
    requests = result.get("requests")
    if not isinstance(requests, list) or not requests:
        fail(f"{label}.requests must be a nonempty array")
    return requests


def validate_completed_request(
    request: Any,
    prompt_tokens: int,
    request_id: str,
    label: str,
    *,
    max_new_tokens: int = 1,
    expected_generated: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    if not isinstance(request, dict):
        fail(f"{label} must be an object")
    prompt = request.get("prompt_token_ids")
    if prompt != list(range(1, prompt_tokens + 1)):
        fail(f"{label} does not contain the ascending {prompt_tokens}-token prompt")
    generated = request.get("generated_token_ids")
    if (
        not isinstance(generated, list)
        or not generated
        or len(generated) > max_new_tokens
    ):
        fail(f"{label}.generated_token_ids has an invalid length")
    generated_tokens = [
        integer(token, f"{label}.generated_token_ids[{index}]")
        for index, token in enumerate(generated)
    ]
    if any(not 0 <= token < VOCAB_SIZE for token in generated_tokens):
        fail(f"{label} generated token is outside the vocabulary")
    if expected_generated is not None and generated_tokens != list(expected_generated):
        fail(f"{label} generated tokens differ from the fixed sequence")
    expected_cache_len = prompt_tokens + len(generated_tokens) - 1
    cache_lengths = request.get("terminal_cache_lengths")
    if (
        request.get("request_id") != request_id
        or request.get("max_new_tokens") != max_new_tokens
        or request.get("prompt_progress_events") != prompt_tokens - 1
        or request.get("execution_units") != expected_cache_len
        or request.get("reserved_context_tokens") != prompt_tokens + max_new_tokens
        or request.get("terminal_sequence_tokens")
        != prompt_tokens + len(generated_tokens)
        or request.get("terminal_status") != "finishing"
        or request.get("terminal_expected_cache_len") != expected_cache_len
        or cache_lengths != [expected_cache_len] * STACK_LAYERS
        or request.get("terminal_cache_lengths_all_expected") is not True
        or request.get("terminal_last_cache_position") != expected_cache_len - 1
        or request.get("terminal_last_logical_block")
        != (expected_cache_len - 1) // BLOCK_TOKENS
        or request.get("terminal_scheduler_active") != 1
        or request.get("terminal_scheduler_waiting") != 0
        or request.get("terminal_allocated_blocks") != CACHE_BLOCKS
        or request.get("terminal_reason") != "length"
        or request.get("release_outcome") != "length"
        or request.get("oracle_capture") is not None
    ):
        fail(f"{label} terminal/cache contract differs")
    request_seconds = finite_nonnegative(
        request.get("request_seconds"), f"{label}.request_seconds"
    )
    reset_seconds = finite_nonnegative(
        request.get("reset_seconds"), f"{label}.reset_seconds"
    )
    return {
        "request_id": request_id,
        "prompt_tokens": prompt_tokens,
        "generated_token_ids": generated_tokens,
        "terminal_sequence_tokens": prompt_tokens + len(generated_tokens),
        "cache_length": expected_cache_len,
        "last_cache_position": expected_cache_len - 1,
        "last_logical_block": (expected_cache_len - 1) // BLOCK_TOKENS,
        "request_seconds": request_seconds,
        "reset_seconds": reset_seconds,
    }


def validate_prefill_cancellation(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    reset_seconds = finite_nonnegative(value.get("reset_seconds"), f"{label}.reset_seconds")
    if (
        value.get("request_id") != "serving-smoke-prefill-cancel"
        or value.get("cancellation_phase") != "prefill"
        or value.get("prompt_tokens") != BOUNDARY_PROMPT_LENGTHS[0]
        or value.get("prompt_progress_before_cancel") != PREFILL_CANCEL_PROGRESS
        or value.get("generated_before_cancel") != []
        or value.get("execution_units_before_cancel") != PREFILL_CANCEL_PROGRESS
        or value.get("status_before_cancel") != "prefilling"
        or value.get("cache_lengths_before_cancel")
        != [PREFILL_CANCEL_PROGRESS] * STACK_LAYERS
        or value.get("scheduler_active_before_cancel") != 1
        or value.get("scheduler_waiting_before_cancel") != 0
        or value.get("allocated_blocks_before_cancel") != CACHE_BLOCKS
        or value.get("status_after_observation") != "cancelling"
        or value.get("prompt_progress_after_observation") != PREFILL_CANCEL_PROGRESS
        or value.get("generated_tokens_after_observation") != 0
        or value.get("cache_lengths_after_observation")
        != [PREFILL_CANCEL_PROGRESS] * STACK_LAYERS
        or value.get("scheduler_active_after_observation") != 1
        or value.get("scheduler_waiting_after_observation") != 0
        or value.get("allocated_blocks_after_observation") != CACHE_BLOCKS
        or value.get("release_outcome") != "cancelled"
    ):
        fail(f"{label} cancellation/cache contract differs")
    return {
        "phase": "prefill",
        "prompt_tokens": BOUNDARY_PROMPT_LENGTHS[0],
        "prompt_progress_before_cancel": PREFILL_CANCEL_PROGRESS,
        "generated_before_cancel": 0,
        "cache_length_before_cancel": PREFILL_CANCEL_PROGRESS,
        "reset_seconds": reset_seconds,
    }


def validate_decode_cancellation(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    reset_seconds = finite_nonnegative(value.get("reset_seconds"), f"{label}.reset_seconds")
    if (
        value.get("request_id") != "serving-smoke-cancel"
        or value.get("cancellation_phase") != "decode"
        or value.get("prompt_tokens") != DECODE_PROMPT_LENGTH
        or value.get("prompt_progress_before_cancel") != DECODE_PROMPT_LENGTH - 1
        or value.get("generated_before_cancel") != [EXPECTED_DECODE_G8[0]]
        or value.get("execution_units_before_cancel") != DECODE_PROMPT_LENGTH
        or value.get("status_before_cancel") != "decoding"
        or value.get("cache_lengths_before_cancel")
        != [DECODE_PROMPT_LENGTH] * STACK_LAYERS
        or value.get("scheduler_active_before_cancel") != 1
        or value.get("scheduler_waiting_before_cancel") != 0
        or value.get("allocated_blocks_before_cancel") != CACHE_BLOCKS
        or value.get("status_after_observation") != "cancelling"
        or value.get("prompt_progress_after_observation") != DECODE_PROMPT_LENGTH
        or value.get("generated_tokens_after_observation") != 1
        or value.get("cache_lengths_after_observation")
        != [DECODE_PROMPT_LENGTH] * STACK_LAYERS
        or value.get("scheduler_active_after_observation") != 1
        or value.get("scheduler_waiting_after_observation") != 0
        or value.get("allocated_blocks_after_observation") != CACHE_BLOCKS
        or value.get("release_outcome") != "cancelled"
    ):
        fail(f"{label} cancellation/cache contract differs")
    return {
        "phase": "decode",
        "prompt_tokens": DECODE_PROMPT_LENGTH,
        "prompt_progress_before_cancel": DECODE_PROMPT_LENGTH - 1,
        "generated_before_cancel": [EXPECTED_DECODE_G8[0]],
        "cache_length_before_cancel": DECODE_PROMPT_LENGTH,
        "cache_length_after_observation": DECODE_PROMPT_LENGTH,
        "reset_seconds": reset_seconds,
    }


def validate_boundary_result(result: dict[str, Any], label: str) -> dict[str, Any]:
    requests = validate_common(result, label)
    if len(requests) != len(BOUNDARY_PROMPT_LENGTHS):
        fail(f"{label} has the wrong boundary request count")
    summaries = [
        validate_completed_request(
            request,
            prompt_tokens,
            f"serving-smoke-p{prompt_tokens:04}",
            f"{label}.requests[{index}]",
        )
        for index, (request, prompt_tokens) in enumerate(
            zip(requests, BOUNDARY_PROMPT_LENGTHS, strict=True)
        )
    ]
    cancellation = validate_prefill_cancellation(
        result.get("cancelled_request"), f"{label}.cancelled_request"
    )
    return {"requests": summaries, "cancellation": cancellation}


def validate_context_result(result: dict[str, Any], label: str) -> dict[str, Any]:
    requests = validate_common(result, label)
    if len(requests) != 1 or result.get("cancelled_request") is not None:
        fail(f"{label} must contain one completed request and no cancellation")
    summary = validate_completed_request(
        requests[0],
        CONTEXT_PROMPT_LENGTH,
        "serving-smoke-p4095",
        f"{label}.requests[0]",
    )
    if (
        summary["terminal_sequence_tokens"] != CONTEXT_TOKENS
        or summary["last_logical_block"] != CACHE_BLOCKS - 1
    ):
        fail(f"{label} does not reach the exact context boundary")
    return summary


def validate_decode_cancel_result(result: dict[str, Any], label: str) -> dict[str, Any]:
    requests = validate_common(result, label)
    if len(requests) != 1:
        fail(f"{label} must contain one completed decode reference request")
    completed = validate_completed_request(
        requests[0],
        DECODE_PROMPT_LENGTH,
        "serving-smoke-p0008",
        f"{label}.requests[0]",
        max_new_tokens=len(EXPECTED_DECODE_G8),
        expected_generated=EXPECTED_DECODE_G8,
    )
    cancellation = validate_decode_cancellation(
        result.get("cancelled_request"), f"{label}.cancelled_request"
    )
    return {"completed_reference": completed, "cancellation": cancellation}


def validate_results(
    boundary_path: Path, context_path: Path, decode_cancel_path: Path
) -> dict[str, Any]:
    boundary_file, boundary_json = load_json(boundary_path, "boundary result")
    context_file, context_json = load_json(context_path, "context result")
    decode_cancel_file, decode_cancel_json = load_json(
        decode_cancel_path, "decode cancellation result"
    )
    try:
        files = (boundary_file, context_file, decode_cancel_file)
        if any(
            os.path.samefile(first, second)
            for index, first in enumerate(files)
            for second in files[index + 1 :]
        ):
            fail("session matrix evidence inputs must be different files")
    except OSError as error:
        fail(f"failed to compare evidence file identity: {error}")
    boundary = validate_boundary_result(boundary_json, str(boundary_file))
    context = validate_context_result(context_json, str(context_file))
    decode_cancel = validate_decode_cancel_result(
        decode_cancel_json, str(decode_cancel_file)
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "passed": True,
        "boundary": boundary,
        "exact_context": context,
        "decode_cancel": decode_cancel,
        "evidence": [
            {
                "kind": "block_boundary_and_prefill_cancel",
                "file": str(boundary_file),
                "sha256": sha256_file(boundary_file),
            },
            {
                "kind": "exact_context_4096",
                "file": str(context_file),
                "sha256": sha256_file(context_file),
            },
            {
                "kind": "decode_cancel",
                "file": str(decode_cancel_file),
                "sha256": sha256_file(decode_cancel_file),
            },
        ],
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
    parser.add_argument("boundary_result", type=Path)
    parser.add_argument("context_result", type=Path)
    parser.add_argument("decode_cancel_result", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = validate_results(
            args.boundary_result, args.context_result, args.decode_cancel_result
        )
        if args.output is not None:
            write_json_create_new(args.output, result)
    except ValidationError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1
    context = result["exact_context"]
    print(
        f"passed=true boundary_prompts={list(BOUNDARY_PROMPT_LENGTHS)} "
        f"prefill_cancel={PREFILL_CANCEL_PROGRESS} "
        f"decode_cancel_token={EXPECTED_DECODE_G8[0]} "
        f"context_tokens={context['terminal_sequence_tokens']} "
        f"context_seconds={context['request_seconds']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
