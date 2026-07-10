#!/usr/bin/env python3
"""Validate SQ8 fixed-M8 serving chunks against all-M1 and source oracles."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


RESULT_SCHEMA_VERSION = "ullm.sq8.serving_chunks_validation.v1"
CHUNK_SCHEMA_VERSION = "ullm.sq8.serving_chunks.v3"
M1_SCHEMA_VERSION = "ullm.sq8.serving_smoke.v2"
CHUNK_MODE = "m8-chunk8"
M1_MODE = "all-m1"
CHUNK_TOKENS = 8
STACK_LAYERS = 40
BLOCK_TOKENS = 16
DEFAULT_REQUIRED_PROMPTS = (1, 7, 8, 9, 15, 16, 17, 32, 128, 512, 4095)
DEFAULT_SOURCE_PROMPTS = (8, 32, 128, 512, 4095)
MAX_PATH_RELATIVE_L2 = 0.10
MIN_PATH_COSINE = 0.995
MIN_PATH_TOP_10_OVERLAP = 5


def load_common_validator() -> Any:
    path = Path(__file__).with_name("validate-sq8-serving-runtime-oracle.py")
    spec = importlib.util.spec_from_file_location("sq8_serving_oracle_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load common validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_common_validator()
ValidationError = common.ValidationError


def fail(message: str) -> None:
    raise ValidationError(message)


def parse_prompt_lengths(value: str, label: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        fail(f"{label} must be comma-separated integers: {error}")
    if not values or any(value <= 0 for value in values) or len(set(values)) != len(values):
        fail(f"{label} must contain unique positive prompt lengths")
    return values


def expected_widths(prompt_tokens: int, mode: str) -> list[int]:
    if prompt_tokens <= 0:
        fail("prompt length must be positive")
    if mode == M1_MODE:
        return [1] * prompt_tokens
    if mode == CHUNK_MODE:
        return [CHUNK_TOKENS] * (prompt_tokens // CHUNK_TOKENS) + [1] * (
            prompt_tokens % CHUNK_TOKENS
        )
    fail(f"unknown prefill mode: {mode}")


def resolve_capture(
    result_path: Path,
    value: Any,
    label: str,
    *,
    expected_bytes: int,
    source_root: Path,
) -> Path:
    if not isinstance(value, str) or not value:
        fail(f"{label} must be a nonempty path")
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [result_path.parent / raw, Path.cwd() / raw]
    resolved: list[Path] = []
    for candidate in candidates:
        try:
            common.regular_file(candidate, expected_bytes, label)
            path = candidate.resolve(strict=True)
        except ValidationError:
            continue
        except OSError:
            continue
        if path not in resolved:
            resolved.append(path)
    if len(resolved) != 1:
        fail(f"{label} must resolve to exactly one regular file, got {resolved}")
    capture = resolved[0]
    result_root = result_path.parent.resolve(strict=True)
    if not common.is_within(capture, result_root):
        fail(f"{label} must remain inside the result directory")
    if common.is_within(capture, source_root):
        fail(f"{label} must remain outside the source oracle")
    return capture


def validate_unit_trace(request: dict[str, Any], prompt_tokens: int, mode: str, label: str) -> None:
    widths = expected_widths(prompt_tokens, mode)
    units = request.get("prefill_execution_units")
    if not isinstance(units, list) or len(units) != len(widths):
        fail(f"{label} has the wrong prefill execution-unit count")
    position = 0
    for index, (unit, width) in enumerate(zip(units, widths, strict=True)):
        unit_label = f"{label}.prefill_execution_units[{index}]"
        if not isinstance(unit, dict):
            fail(f"{unit_label} must be an object")
        end = position + width
        cache_lengths = unit.get("cache_lengths")
        if (
            unit.get("start_position") != position
            or unit.get("width") != width
            or unit.get("end_position") != end
            or unit.get("final_prompt_unit") is not (end == prompt_tokens)
            or unit.get("cache_lengths_all_expected") is not True
            or not isinstance(cache_lengths, list)
            or len(cache_lengths) != STACK_LAYERS
            or any(value != end for value in cache_lengths)
            or unit.get("last_cache_position") != end - 1
            or unit.get("last_logical_block") != (end - 1) // BLOCK_TOKENS
        ):
            fail(f"{unit_label} does not bind its position/cache transition")
        position = end
    if position != prompt_tokens:
        fail(f"{label} prefill units do not cover the complete prompt")
    if (
        request.get("processed_prompt_tokens") != prompt_tokens
        or request.get("prompt_progress_events") != len(widths) - 1
        or request.get("execution_calls") != len(widths)
        or request.get("execution_units") != len(widths)
    ):
        fail(f"{label} confuses processed prompt tokens with execution calls")


def validate_result(
    result_path: Path,
    *,
    schema: str,
    mode: str,
    required_prompts: tuple[int, ...],
    source_root: Path,
    require_build_identity: bool,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    common.regular_file(result_path, None, f"{mode} result")
    result_path = result_path.resolve(strict=True)
    result = common.load_json(result_path)
    if (
        result.get("schema_version") != schema
        or result.get("prefill_mode") != mode
        or result.get("prefill_chunk_tokens") != CHUNK_TOKENS
        or result.get("artifact_content_sha256") != common.EXPECTED_ARTIFACT_SHA256
        or result.get("package_manifest_sha256") != common.EXPECTED_PACKAGE_SHA256
    ):
        fail(f"{result_path} has the wrong schema/mode/model identity")
    implementation = result.get("prefill_implementation")
    expected_implementation = {
        M1_MODE: "sq8.sequential-m1.v1",
        CHUNK_MODE: "sq8.fixed-m8-cached-prefix.v1",
    }[mode]
    if implementation != expected_implementation:
        fail(f"{result_path} has the wrong prefill implementation")
    git_commit = result.get("runner_git_commit")
    binary_sha256 = result.get("runner_binary_sha256")
    worktree_clean = result.get("runner_worktree_clean")
    has_build_identity = (
        isinstance(git_commit, str)
        and len(git_commit) == 40
        and all(character in "0123456789abcdef" for character in git_commit)
        and isinstance(binary_sha256, str)
        and common.SHA256_RE.fullmatch(binary_sha256) is not None
        and isinstance(worktree_clean, bool)
    )
    if require_build_identity and (not has_build_identity or worktree_clean is not True):
        fail(f"{result_path} does not bind a clean runner commit/binary")
    common.validate_device(result.get("device"), f"{result_path}.device")
    cache_lengths = result.get("post_reset_cache_lengths")
    if (
        result.get("kv_cache_bytes") != common.KV_CACHE_BYTES
        or result.get("cache_blocks") != common.CACHE_BLOCKS
        or result.get("context_tokens") != common.CONTEXT_TOKENS
        or result.get("post_reset_status") != "ready"
        or result.get("post_reset_active") != 0
        or result.get("post_reset_waiting") != 0
        or result.get("post_reset_allocated_blocks") != 0
        or result.get("post_reset_cache_lengths_all_zero") is not True
        or not isinstance(cache_lengths, list)
        or len(cache_lengths) != STACK_LAYERS
        or any(value != 0 for value in cache_lengths)
    ):
        fail(f"{result_path} does not prove the reusable reset baseline")

    requests = result.get("requests")
    if not isinstance(requests, list):
        fail(f"{result_path}.requests must be a list")
    values: dict[int, dict[str, Any]] = {}
    for index, request in enumerate(requests):
        label = f"{result_path}.requests[{index}]"
        if not isinstance(request, dict):
            fail(f"{label} must be an object")
        prompt = request.get("prompt_token_ids")
        if not isinstance(prompt, list) or prompt != list(range(1, len(prompt) + 1)):
            fail(f"{label} does not use ascending raw token IDs")
        prompt_tokens = len(prompt)
        if prompt_tokens in values or prompt_tokens not in required_prompts:
            fail(f"{label} has an unexpected or duplicate prompt length {prompt_tokens}")
        generated = request.get("generated_token_ids")
        terminal_cache = request.get("terminal_cache_lengths")
        if (
            request.get("max_new_tokens") != 1
            or not isinstance(generated, list)
            or len(generated) != 1
            or request.get("terminal_expected_cache_len") != prompt_tokens
            or request.get("terminal_cache_lengths_all_expected") is not True
            or not isinstance(terminal_cache, list)
            or len(terminal_cache) != STACK_LAYERS
            or any(value != prompt_tokens for value in terminal_cache)
            or request.get("terminal_last_cache_position") != prompt_tokens - 1
            or request.get("terminal_last_logical_block")
            != (prompt_tokens - 1) // BLOCK_TOKENS
            or request.get("terminal_scheduler_active") != 1
            or request.get("terminal_scheduler_waiting") != 0
            or request.get("terminal_allocated_blocks") != common.CACHE_BLOCKS
            or request.get("terminal_reason") != "length"
            or request.get("release_outcome") != "length"
        ):
            fail(f"{label} does not prove its terminal cache/scheduler contract")
        validate_unit_trace(request, prompt_tokens, mode, label)
        capture = request.get("oracle_capture")
        if not isinstance(capture, dict) or capture.get("position") != prompt_tokens - 1:
            fail(f"{label} has invalid oracle capture metadata")
        hidden = resolve_capture(
            result_path,
            capture.get("final_hidden_file"),
            f"{label}.final_hidden_file",
            expected_bytes=common.HIDDEN_SIZE * common.F32.size,
            source_root=source_root,
        )
        logits = resolve_capture(
            result_path,
            capture.get("logits_file"),
            f"{label}.logits_file",
            expected_bytes=common.VOCAB_SIZE * common.F32.size,
            source_root=source_root,
        )
        hidden_hash = common.sha256_value(
            capture.get("final_hidden_f32_le_sha256"), f"{label}.final_hidden hash"
        )
        logits_hash = common.sha256_value(
            capture.get("logits_f32_le_sha256"), f"{label}.logits hash"
        )
        if common.sha256_file(hidden) != hidden_hash or common.sha256_file(logits) != logits_hash:
            fail(f"{label} capture payload hash differs")
        token_id = common.integer(generated[0], f"{label}.generated_token_ids[0]")
        if token_id != capture.get("top1_token_id"):
            fail(f"{label} generated token differs from recorded capture top-1")
        values[prompt_tokens] = {
            "prompt_tokens": prompt_tokens,
            "token_id": token_id,
            "top1_logit": common.finite_f32(capture.get("top1_logit"), f"{label}.top1_logit"),
            "hidden": hidden,
            "hidden_sha256": hidden_hash,
            "logits": logits,
            "logits_sha256": logits_hash,
        }
    if set(values) != set(required_prompts):
        fail(
            f"{result_path} prompt coverage differs: "
            f"expected={list(required_prompts)} actual={sorted(values)}"
        )
    return values, {
        "result_file": str(result_path),
        "result_sha256": common.sha256_file(result_path),
        "schema_version": schema,
        "prefill_mode": mode,
        "prefill_implementation": implementation,
        "runner_git_commit": git_commit if has_build_identity else None,
        "runner_worktree_clean": worktree_clean if has_build_identity else None,
        "runner_binary_sha256": binary_sha256 if has_build_identity else None,
    }


def validate_ranked_capture(
    token_id: int,
    logit: float,
    ranked: list[dict[str, Any]],
    label: str,
) -> None:
    if not ranked or ranked[0].get("token_id") != token_id or ranked[0].get("logit") != logit:
        fail(f"{label} recorded top-1 differs from raw logits")


def validate_path_gate(metrics: dict[str, Any], *, logits: bool, label: str) -> None:
    if (
        metrics.get("nonfinite_count") != 0
        or metrics.get("relative_l2", math.inf) > MAX_PATH_RELATIVE_L2
        or metrics.get("cosine_similarity", -math.inf) < MIN_PATH_COSINE
    ):
        fail(f"{label} fails the chunk/all-M1 numerical gate: {metrics}")
    if logits and (
        metrics.get("top_1_exact") is not True
        or metrics.get("top_10_overlap", 0) < MIN_PATH_TOP_10_OVERLAP
    ):
        fail(f"{label} fails the chunk/all-M1 token gate: {metrics}")


def validate_results(
    chunk_result: Path,
    m1_result: Path,
    fixture_root: Path,
    required_prompts: tuple[int, ...],
    source_prompts: tuple[int, ...],
    require_build_identity: bool = False,
) -> dict[str, Any]:
    if not set(source_prompts).issubset(required_prompts):
        fail("source prompts must be a subset of required prompts")
    source = common.validate_source_oracle(fixture_root)
    chunk, chunk_evidence = validate_result(
        chunk_result,
        schema=CHUNK_SCHEMA_VERSION,
        mode=CHUNK_MODE,
        required_prompts=required_prompts,
        source_root=source["root"],
        require_build_identity=require_build_identity,
    )
    m1, m1_evidence = validate_result(
        m1_result,
        schema=M1_SCHEMA_VERSION,
        mode=M1_MODE,
        required_prompts=required_prompts,
        source_root=source["root"],
        require_build_identity=require_build_identity,
    )
    try:
        if os.path.samefile(chunk_result, m1_result):
            fail("chunk and all-M1 results must be different files")
    except OSError as error:
        fail(f"failed to compare result file identity: {error}")
    for field in ("runner_git_commit", "runner_binary_sha256"):
        chunk_value = chunk_evidence[field]
        m1_value = m1_evidence[field]
        if (chunk_value is not None or m1_value is not None) and chunk_value != m1_value:
            fail(f"chunk/all-M1 {field} differs")

    prompts = []
    for prompt_tokens in required_prompts:
        chunk_value = chunk[prompt_tokens]
        m1_value = m1[prompt_tokens]
        if chunk_value["token_id"] != m1_value["token_id"]:
            fail(f"prompt {prompt_tokens} chunk/all-M1 token differs")
        common.reject_same_file(
            chunk_value["hidden"], m1_value["hidden"], f"prompt {prompt_tokens} final hidden"
        )
        common.reject_same_file(
            chunk_value["logits"], m1_value["logits"], f"prompt {prompt_tokens} logits"
        )
        hidden_metrics = common.compare_f32_files(
            chunk_value["hidden"],
            m1_value["hidden"],
            elements=common.HIDDEN_SIZE,
            top_k=False,
            label=f"prompt {prompt_tokens} chunk/all-M1 final hidden",
        )
        logits_metrics = common.compare_f32_files(
            chunk_value["logits"],
            m1_value["logits"],
            elements=common.VOCAB_SIZE,
            top_k=True,
            label=f"prompt {prompt_tokens} chunk/all-M1 logits",
        )
        validate_path_gate(hidden_metrics, logits=False, label=f"prompt {prompt_tokens} hidden")
        validate_path_gate(logits_metrics, logits=True, label=f"prompt {prompt_tokens} logits")
        validate_ranked_capture(
            chunk_value["token_id"],
            chunk_value["top1_logit"],
            logits_metrics["actual_top_10"],
            f"prompt {prompt_tokens} chunk",
        )
        validate_ranked_capture(
            m1_value["token_id"],
            m1_value["top1_logit"],
            logits_metrics["reference_top_10"],
            f"prompt {prompt_tokens} all-M1",
        )
        prompt_result: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "token_id": chunk_value["token_id"],
            "chunk_vs_all_m1": {
                "final_hidden": hidden_metrics,
                "logits": logits_metrics,
            },
            "captures": {
                "chunk_hidden_sha256": chunk_value["hidden_sha256"],
                "chunk_logits_sha256": chunk_value["logits_sha256"],
                "all_m1_hidden_sha256": m1_value["hidden_sha256"],
                "all_m1_logits_sha256": m1_value["logits_sha256"],
            },
        }
        if prompt_tokens in source_prompts:
            prefix = f"prompts/raw-p{prompt_tokens:04d}"
            source_hidden, source_hidden_hash = common.source_payload(
                source, f"{prefix}/final-hidden.f32le", common.HIDDEN_SIZE * common.F32.size
            )
            source_logits, source_logits_hash = common.source_payload(
                source, f"{prefix}/prefill-logits.f32le", common.VOCAB_SIZE * common.F32.size
            )
            source_token_file, source_token_hash = common.source_payload(
                source, f"{prefix}/greedy-g1.u32le", common.U32.size
            )
            source_token = common.read_first_u32(source_token_file)
            if chunk_value["token_id"] != source_token:
                fail(f"prompt {prompt_tokens} chunk token differs from source oracle")
            common.reject_same_file(
                chunk_value["hidden"], source_hidden, f"prompt {prompt_tokens} source hidden"
            )
            common.reject_same_file(
                chunk_value["logits"], source_logits, f"prompt {prompt_tokens} source logits"
            )
            source_hidden_metrics = common.compare_f32_files(
                chunk_value["hidden"],
                source_hidden,
                elements=common.HIDDEN_SIZE,
                top_k=False,
                label=f"prompt {prompt_tokens} chunk/source final hidden",
            )
            source_logits_metrics = common.compare_f32_files(
                chunk_value["logits"],
                source_logits,
                elements=common.VOCAB_SIZE,
                top_k=True,
                label=f"prompt {prompt_tokens} chunk/source logits",
            )
            common.validate_gate(
                source_hidden_metrics, logits=False, label=f"prompt {prompt_tokens} source hidden"
            )
            common.validate_gate(
                source_logits_metrics, logits=True, label=f"prompt {prompt_tokens} source logits"
            )
            prompt_result["chunk_vs_source"] = {
                "final_hidden": source_hidden_metrics,
                "logits": source_logits_metrics,
                "source_hidden_sha256": source_hidden_hash,
                "source_logits_sha256": source_logits_hash,
                "source_greedy_g1_sha256": source_token_hash,
            }
        prompts.append(prompt_result)

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "passed": True,
        "required_prompts": list(required_prompts),
        "source_prompts": list(source_prompts),
        "thresholds": {
            "chunk_vs_all_m1": {
                "max_relative_l2": MAX_PATH_RELATIVE_L2,
                "min_cosine_similarity": MIN_PATH_COSINE,
                "top_1": "exact",
                "min_top_10_overlap": MIN_PATH_TOP_10_OVERLAP,
            },
            "chunk_vs_source": {
                "max_relative_l2": common.MAX_RELATIVE_L2,
                "min_cosine_similarity": common.MIN_COSINE_SIMILARITY,
                "top_1": "exact",
                "min_top_10_overlap": common.MIN_TOP_10_OVERLAP,
            },
        },
        "source_oracle": source["evidence"],
        "prompts": prompts,
        "evidence": [chunk_evidence, m1_evidence],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-result", type=Path, required=True)
    parser.add_argument("--all-m1-result", type=Path, required=True)
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=Path("tests/fixtures/sq8-serving-v0.1/oracles/vllm-source-v0.1"),
    )
    parser.add_argument(
        "--required-prompts", default=",".join(str(value) for value in DEFAULT_REQUIRED_PROMPTS)
    )
    parser.add_argument(
        "--source-prompts", default=",".join(str(value) for value in DEFAULT_SOURCE_PROMPTS)
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-build-identity", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        required_prompts = parse_prompt_lengths(args.required_prompts, "required prompts")
        source_prompts = parse_prompt_lengths(args.source_prompts, "source prompts")
        validation = validate_results(
            args.chunk_result,
            args.all_m1_result,
            args.fixture_root,
            required_prompts,
            source_prompts,
            args.require_build_identity,
        )
        if args.output is not None:
            common.write_json_create_new(args.output, validation)
    except ValidationError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1
    worst_relative_l2 = max(
        max(
            prompt["chunk_vs_all_m1"]["final_hidden"]["relative_l2"],
            prompt["chunk_vs_all_m1"]["logits"]["relative_l2"],
        )
        for prompt in validation["prompts"]
    )
    minimum_cosine = min(
        min(
            prompt["chunk_vs_all_m1"]["final_hidden"]["cosine_similarity"],
            prompt["chunk_vs_all_m1"]["logits"]["cosine_similarity"],
        )
        for prompt in validation["prompts"]
    )
    print(
        f"passed=true prompts={validation['required_prompts']} "
        f"source_prompts={validation['source_prompts']} "
        f"worst_relative_l2={worst_relative_l2:.9f} "
        f"minimum_cosine={minimum_cosine:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
