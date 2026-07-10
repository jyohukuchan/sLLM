#!/usr/bin/env python3
"""Validate the SQ8 chunked-prefill 3584+512 deep-context boundary run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import struct
import sys
from pathlib import Path
from typing import Any


INPUT_SCHEMA_VERSION = "ullm.sq8.serving_deep_boundary.v1"
RESULT_SCHEMA_VERSION = "ullm.sq8.serving_deep_boundary_validation.v1"
PREFILL_MODE = "m8-chunk8"
PREFILL_CHUNK_TOKENS = 8
PREFILL_IMPLEMENTATION = "sq8.fixed-m8-cached-prefix.v1"
REQUEST_ID = "deep-boundary-p3584-g512"
PROMPT_TOKENS = 3_584
GENERATED_TOKENS = 512
CONTEXT_TOKENS = 4_096
STACK_LAYERS = 40
BLOCK_TOKENS = 16
CACHE_BLOCKS = 256
VOCAB_SIZE = 151_936
KV_CACHE_BYTES = 1_342_177_280
EOS_TOKEN_IDS = (151_645, 151_643)
TERMINAL_CACHE_LEN = CONTEXT_TOKENS - 1
TERMINAL_CACHE_POSITION = TERMINAL_CACHE_LEN - 1
TERMINAL_LOGICAL_BLOCK = TERMINAL_CACHE_POSITION // BLOCK_TOKENS
PREFILL_EXECUTION_CALLS = PROMPT_TOKENS // PREFILL_CHUNK_TOKENS
DECODE_EXECUTION_CALLS = GENERATED_TOKENS - 1
TOTAL_EXECUTION_CALLS = PREFILL_EXECUTION_CALLS + DECODE_EXECUTION_CALLS
PROMPT_PROGRESS_EVENTS = PREFILL_EXECUTION_CALLS - 1
MAX_JSON_BYTES = 8 * 1024 * 1024
EXPECTED_ARTIFACT_SHA256 = (
    "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147"
)
EXPECTED_PACKAGE_SHA256 = (
    "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb"
)
GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
U32 = struct.Struct("<I")


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


def validate_expected_build_identity(git_commit: str, binary_sha256: str) -> None:
    if GIT_COMMIT_RE.fullmatch(git_commit) is None:
        fail("expected runner git commit must be 40 lowercase hexadecimal characters")
    if SHA256_RE.fullmatch(binary_sha256) is None:
        fail("expected binary SHA-256 must be 64 lowercase hexadecimal characters")


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


def validate_zero_cache_baseline(result: dict[str, Any], label: str) -> None:
    cache_lengths = result.get("post_reset_cache_lengths")
    if (
        result.get("post_reset_status") != "ready"
        or result.get("post_reset_active") != 0
        or result.get("post_reset_waiting") != 0
        or result.get("post_reset_allocated_blocks") != 0
        or result.get("post_reset_cache_lengths_all_zero") is not True
        or not isinstance(cache_lengths, list)
        or len(cache_lengths) != STACK_LAYERS
        or any(integer(value, f"{label}.post_reset_cache_lengths") != 0 for value in cache_lengths)
    ):
        fail(f"{label} does not prove the reusable post-reset baseline")


def validate_generated_tokens(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != GENERATED_TOKENS:
        fail(f"{label} must contain exactly {GENERATED_TOKENS} tokens")
    tokens = [integer(token, f"{label}[{index}]") for index, token in enumerate(value)]
    if any(token < 0 or token >= VOCAB_SIZE for token in tokens):
        fail(f"{label} contains a token outside the vocabulary")
    return tokens


def validate_generated_steps(value: Any, generated: list[int], label: str) -> None:
    if not isinstance(value, list) or len(value) != GENERATED_TOKENS:
        fail(f"{label} must contain exactly {GENERATED_TOKENS} steps")
    for index, (step, token_id) in enumerate(zip(value, generated, strict=True)):
        step_label = f"{label}[{index}]"
        if not isinstance(step, dict):
            fail(f"{step_label} must be an object")
        cache_len = PROMPT_TOKENS + index
        cache_lengths = step.get("cache_lengths")
        expected_write_position = None if index == 0 else cache_len - 1
        expected_status = "finishing" if index == GENERATED_TOKENS - 1 else "decoding"
        expected_reason = "length" if index == GENERATED_TOKENS - 1 else None
        if (
            integer(step.get("generated_index"), f"{step_label}.generated_index") != index
            or integer(step.get("token_id"), f"{step_label}.token_id") != token_id
            or integer(step.get("cache_len"), f"{step_label}.cache_len") != cache_len
            or step.get("cache_write_position") != expected_write_position
            or step.get("status") != expected_status
            or not isinstance(cache_lengths, list)
            or len(cache_lengths) != STACK_LAYERS
            or any(
                integer(length, f"{step_label}.cache_lengths") != cache_len
                for length in cache_lengths
            )
            or integer(step.get("scheduler_active"), f"{step_label}.scheduler_active") != 1
            or integer(step.get("scheduler_waiting"), f"{step_label}.scheduler_waiting") != 0
            or integer(step.get("allocated_blocks"), f"{step_label}.allocated_blocks")
            != CACHE_BLOCKS
            or step.get("cache_lengths_all_expected") is not True
            or step.get("terminal_reason") != expected_reason
        ):
            fail(f"{step_label} does not prove its generated/cache transition")


def validate_prefill_execution_units(value: Any, label: str) -> None:
    if not isinstance(value, list) or len(value) != PREFILL_EXECUTION_CALLS:
        fail(f"{label} must contain exactly {PREFILL_EXECUTION_CALLS} units")
    for index, unit in enumerate(value):
        unit_label = f"{label}[{index}]"
        if not isinstance(unit, dict):
            fail(f"{unit_label} must be an object")
        start = index * PREFILL_CHUNK_TOKENS
        end = start + PREFILL_CHUNK_TOKENS
        cache_lengths = unit.get("cache_lengths")
        if (
            integer(unit.get("start_position"), f"{unit_label}.start_position") != start
            or integer(unit.get("width"), f"{unit_label}.width") != PREFILL_CHUNK_TOKENS
            or integer(unit.get("end_position"), f"{unit_label}.end_position") != end
            or unit.get("final_prompt_unit") is not (index == PREFILL_EXECUTION_CALLS - 1)
            or not isinstance(cache_lengths, list)
            or len(cache_lengths) != STACK_LAYERS
            or any(
                integer(length, f"{unit_label}.cache_lengths") != end
                for length in cache_lengths
            )
            or unit.get("cache_lengths_all_expected") is not True
            or integer(
                unit.get("last_cache_position"), f"{unit_label}.last_cache_position"
            )
            != end - 1
            or integer(unit.get("last_logical_block"), f"{unit_label}.last_logical_block")
            != (end - 1) // BLOCK_TOKENS
        ):
            fail(f"{unit_label} does not prove its M8 prefill/cache transition")


def validate_terminal_request(request: dict[str, Any], label: str) -> None:
    cache_lengths = request.get("terminal_cache_lengths")
    if (
        request.get("terminal_expected_cache_len") != TERMINAL_CACHE_LEN
        or request.get("terminal_cache_lengths_all_expected") is not True
        or not isinstance(cache_lengths, list)
        or len(cache_lengths) != STACK_LAYERS
        or any(
            integer(length, f"{label}.terminal_cache_lengths") != TERMINAL_CACHE_LEN
            for length in cache_lengths
        )
        or request.get("terminal_last_cache_position") != TERMINAL_CACHE_POSITION
        or request.get("terminal_last_logical_block") != TERMINAL_LOGICAL_BLOCK
        or integer(
            request.get("terminal_scheduler_active"), f"{label}.terminal_scheduler_active"
        )
        != 1
        or integer(
            request.get("terminal_scheduler_waiting"), f"{label}.terminal_scheduler_waiting"
        )
        != 0
        or integer(
            request.get("terminal_allocated_blocks"), f"{label}.terminal_allocated_blocks"
        )
        != CACHE_BLOCKS
        or request.get("terminal_status") != "finishing"
        or request.get("terminal_reason") != "length"
        or request.get("release_outcome") != "length"
    ):
        fail(f"{label} does not prove the terminal 4096-token boundary")


def token_sha256(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(U32.pack(token))
    return digest.hexdigest()


def validate_result(
    result_path: Path, expected_runner_git_commit: str, expected_binary_sha256: str
) -> dict[str, Any]:
    validate_expected_build_identity(expected_runner_git_commit, expected_binary_sha256)
    result_file, result = load_json(result_path, "deep-boundary result")
    label = str(result_file)
    build_identity = validate_build_identity(
        result, expected_runner_git_commit, expected_binary_sha256, label
    )
    if (
        result.get("schema_version") != INPUT_SCHEMA_VERSION
        or not isinstance(result.get("passed"), bool)
        or result.get("prefill_mode") != PREFILL_MODE
        or result.get("prefill_chunk_tokens") != PREFILL_CHUNK_TOKENS
        or result.get("prefill_implementation") != PREFILL_IMPLEMENTATION
        or result.get("artifact_content_sha256") != EXPECTED_ARTIFACT_SHA256
        or result.get("package_manifest_sha256") != EXPECTED_PACKAGE_SHA256
        or result.get("kv_cache_bytes") != KV_CACHE_BYTES
        or result.get("cache_blocks") != CACHE_BLOCKS
        or result.get("context_tokens") != CONTEXT_TOKENS
        or result.get("test_only_ignore_eos") is not True
        or result.get("cancelled_request") is not None
    ):
        fail(f"{label} has the wrong deep-boundary schema/model/runtime contract")
    validate_device(result.get("device"), f"{label}.device")
    validate_zero_cache_baseline(result, label)

    requests = result.get("requests")
    if not isinstance(requests, list) or len(requests) != 1:
        fail(f"{label}.requests must contain exactly one request")
    request = requests[0]
    request_label = f"{label}.requests[0]"
    if not isinstance(request, dict):
        fail(f"{request_label} must be an object")
    prompt = request.get("prompt_token_ids")
    if prompt != list(range(1, PROMPT_TOKENS + 1)):
        fail(f"{request_label} does not use the ascending {PROMPT_TOKENS}-token prompt")
    if (
        request.get("request_id") != REQUEST_ID
        or request.get("max_new_tokens") != GENERATED_TOKENS
        or request.get("test_only_ignore_eos") is not True
        or request.get("reserved_context_tokens") != CONTEXT_TOKENS
        or request.get("terminal_sequence_tokens") != CONTEXT_TOKENS
        or request.get("processed_prompt_tokens") != PROMPT_TOKENS
        or request.get("execution_calls") != TOTAL_EXECUTION_CALLS
        or request.get("execution_units") != TOTAL_EXECUTION_CALLS
        or request.get("prompt_progress_events") != PROMPT_PROGRESS_EVENTS
    ):
        fail(f"{request_label} does not bind the fixed deep-boundary execution contract")
    validate_prefill_execution_units(
        request.get("prefill_execution_units"),
        f"{request_label}.prefill_execution_units",
    )
    generated = validate_generated_tokens(
        request.get("generated_token_ids"), f"{request_label}.generated_token_ids"
    )
    validate_generated_steps(
        request.get("generated_steps"), generated, f"{request_label}.generated_steps"
    )
    validate_terminal_request(request, request_label)

    eos_indices = [index for index, token in enumerate(generated) if token in EOS_TOKEN_IDS]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "passed": True,
        "build_identity": build_identity,
        "prefill_mode": PREFILL_MODE,
        "prefill_chunk_tokens": PREFILL_CHUNK_TOKENS,
        "prefill_implementation": PREFILL_IMPLEMENTATION,
        "prompt_tokens": PROMPT_TOKENS,
        "generated_tokens": GENERATED_TOKENS,
        "reserved_context_tokens": PROMPT_TOKENS + GENERATED_TOKENS,
        "terminal_cache_len": TERMINAL_CACHE_LEN,
        "terminal_last_cache_position": TERMINAL_CACHE_POSITION,
        "terminal_last_logical_block": TERMINAL_LOGICAL_BLOCK,
        "prefill_execution_calls": PREFILL_EXECUTION_CALLS,
        "decode_execution_calls": DECODE_EXECUTION_CALLS,
        "total_execution_calls": TOTAL_EXECUTION_CALLS,
        "test_only_ignore_eos": True,
        "observed_eos_generated_indices": eos_indices,
        "generated_token_ids_sha256": token_sha256(generated),
        "evidence": {
            "file": str(result_file),
            "sha256": sha256_file(result_file),
        },
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
    print(
        f"passed=true prompt_tokens={validation['prompt_tokens']} "
        f"generated_tokens={validation['generated_tokens']} "
        f"terminal_cache_len={validation['terminal_cache_len']} "
        f"observed_eos={validation['observed_eos_generated_indices']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
