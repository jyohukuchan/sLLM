#!/usr/bin/env python3
"""Compare isolated baseline and tail-fix SQ8_0 prefill oracle captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any


SCHEMA = "ullm.sq8.prefill_tail_fix.oracle_comparison.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("requests"), list):
        raise ValueError(f"{path} is not an SQ8_0 serving result")
    return value


def resolve_capture(result_path: Path, capture: dict[str, Any], field: str) -> Path:
    raw = capture.get(field)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{result_path}: missing {field}")
    path = Path(raw)
    if not path.is_absolute():
        path = result_path.parent / path
    if not path.is_file():
        raise ValueError(f"{result_path}: capture file does not exist: {path}")
    return path


def f32_metrics(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference_bytes = reference_path.read_bytes()
    candidate_bytes = candidate_path.read_bytes()
    if len(reference_bytes) != len(candidate_bytes) or len(reference_bytes) % 4:
        raise ValueError(
            f"capture shape mismatch: {reference_path}={len(reference_bytes)} "
            f"{candidate_path}={len(candidate_bytes)}"
        )
    reference = struct.unpack(f"<{len(reference_bytes) // 4}f", reference_bytes)
    candidate = struct.unpack(f"<{len(candidate_bytes) // 4}f", candidate_bytes)
    max_abs = 0.0
    diff_l2_squared = 0.0
    reference_l2_squared = 0.0
    candidate_l2_squared = 0.0
    dot = 0.0
    nonfinite = 0
    for expected, actual in zip(reference, candidate, strict=True):
        if not math.isfinite(expected) or not math.isfinite(actual):
            nonfinite += 1
            continue
        difference = actual - expected
        max_abs = max(max_abs, abs(difference))
        diff_l2_squared += difference * difference
        reference_l2_squared += expected * expected
        candidate_l2_squared += actual * actual
        dot += expected * actual
    denominator = math.sqrt(reference_l2_squared)
    cosine_denominator = math.sqrt(reference_l2_squared * candidate_l2_squared)
    return {
        "elements": len(reference),
        "exact_f32_le_bytes": reference_bytes == candidate_bytes,
        "baseline_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "max_abs": max_abs,
        "relative_l2": math.sqrt(diff_l2_squared) / denominator if denominator else 0.0,
        "cosine_similarity": dot / cosine_denominator if cosine_denominator else 1.0,
        "nonfinite_count": nonfinite,
    }


def unit_summary(request: dict[str, Any]) -> dict[str, Any]:
    units = request.get("prefill_execution_units")
    if not isinstance(units, list):
        raise ValueError("request lacks prefill_execution_units")
    widths = [unit.get("width") for unit in units]
    if not all(isinstance(width, int) and width > 0 for width in widths):
        raise ValueError("prefill execution-unit width is invalid")
    return {
        "execution_calls": len(widths),
        "logical_widths": widths,
        "logical_width_sum": sum(widths),
        "last_logical_width": widths[-1],
    }


def request_index(result: dict[str, Any], result_path: Path) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for request in result["requests"]:
        if not isinstance(request, dict):
            raise ValueError(f"{result_path}: request is not an object")
        prompt = request.get("prompt_token_ids")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"{result_path}: prompt token IDs are invalid")
        length = len(prompt)
        if length in indexed:
            raise ValueError(f"{result_path}: duplicate prompt length {length}")
        indexed[length] = request
    return indexed


def compare_request(
    prompt_tokens: int,
    baseline_result_path: Path,
    candidate_result_path: Path,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if baseline.get("prompt_token_ids") != candidate.get("prompt_token_ids"):
        raise ValueError(f"prompt {prompt_tokens}: input tokens differ")
    baseline_capture = baseline.get("oracle_capture")
    candidate_capture = candidate.get("oracle_capture")
    if not isinstance(baseline_capture, dict) or not isinstance(candidate_capture, dict):
        raise ValueError(f"prompt {prompt_tokens}: missing oracle capture")
    if baseline_capture.get("position") != prompt_tokens - 1:
        raise ValueError(f"prompt {prompt_tokens}: baseline position is invalid")
    if candidate_capture.get("position") != prompt_tokens - 1:
        raise ValueError(f"prompt {prompt_tokens}: candidate position is invalid")
    baseline_generated = baseline.get("generated_token_ids")
    candidate_generated = candidate.get("generated_token_ids")
    if not isinstance(baseline_generated, list) or not isinstance(candidate_generated, list):
        raise ValueError(f"prompt {prompt_tokens}: generated token IDs are invalid")
    return {
        "prompt_tokens": prompt_tokens,
        "generated_token_ids": {
            "baseline": baseline_generated,
            "candidate": candidate_generated,
            "exact": baseline_generated == candidate_generated,
        },
        "top1": {
            "baseline_token_id": baseline_capture.get("top1_token_id"),
            "candidate_token_id": candidate_capture.get("top1_token_id"),
            "exact_token": baseline_capture.get("top1_token_id")
            == candidate_capture.get("top1_token_id"),
            "baseline_logit": baseline_capture.get("top1_logit"),
            "candidate_logit": candidate_capture.get("top1_logit"),
        },
        "baseline_schedule": unit_summary(baseline),
        "candidate_schedule": unit_summary(candidate),
        "final_hidden": f32_metrics(
            resolve_capture(baseline_result_path, baseline_capture, "final_hidden_file"),
            resolve_capture(candidate_result_path, candidate_capture, "final_hidden_file"),
        ),
        "logits": f32_metrics(
            resolve_capture(baseline_result_path, baseline_capture, "logits_file"),
            resolve_capture(candidate_result_path, candidate_capture, "logits_file"),
        ),
    }


def main() -> int:
    args = parse_args()
    baseline_result = load_result(args.baseline_result)
    candidate_result = load_result(args.candidate_result)
    baseline_requests = request_index(baseline_result, args.baseline_result)
    candidate_requests = request_index(candidate_result, args.candidate_result)
    if set(baseline_requests) != set(candidate_requests):
        raise ValueError(
            "baseline/candidate prompt coverage differs: "
            f"{sorted(baseline_requests)} != {sorted(candidate_requests)}"
        )
    comparisons = [
        compare_request(
            prompt_tokens,
            args.baseline_result,
            args.candidate_result,
            baseline_requests[prompt_tokens],
            candidate_requests[prompt_tokens],
        )
        for prompt_tokens in sorted(baseline_requests)
    ]
    output = {
        "schema_version": SCHEMA,
        "baseline": {
            "result": str(args.baseline_result),
            "prefill_mode": baseline_result.get("prefill_mode"),
            "prefill_implementation": baseline_result.get("prefill_implementation"),
            "runner_git_commit": baseline_result.get("runner_git_commit"),
            "runner_binary_sha256": baseline_result.get("runner_binary_sha256"),
        },
        "candidate": {
            "result": str(args.candidate_result),
            "prefill_mode": candidate_result.get("prefill_mode"),
            "prefill_implementation": candidate_result.get("prefill_implementation"),
            "runner_git_commit": candidate_result.get("runner_git_commit"),
            "runner_binary_sha256": candidate_result.get("runner_binary_sha256"),
        },
        "comparisons": comparisons,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
