#!/usr/bin/env python3
"""Record SQ8_0 baseline/candidate prefill oracle deltas without a threshold gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        result = json.load(source)
    if not isinstance(result, dict) or not isinstance(result.get("requests"), list):
        raise ValueError(f"not an SQ8_0 serving result: {path}")
    return result


def capture_path(result_path: Path, capture: dict[str, Any], field: str) -> Path:
    raw = capture.get(field)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{result_path}: missing {field}")
    path = Path(raw)
    if not path.is_absolute():
        path = result_path.parent / path
    if not path.is_file():
        raise ValueError(f"missing capture: {path}")
    return path


def f32_delta(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference_bytes = reference_path.read_bytes()
    candidate_bytes = candidate_path.read_bytes()
    if len(reference_bytes) != len(candidate_bytes) or len(reference_bytes) % 4:
        raise ValueError(f"capture shape mismatch: {reference_path} vs {candidate_path}")
    reference = struct.unpack(f"<{len(reference_bytes) // 4}f", reference_bytes)
    candidate = struct.unpack(f"<{len(candidate_bytes) // 4}f", candidate_bytes)
    max_abs = 0.0
    diff_l2 = 0.0
    reference_l2 = 0.0
    candidate_l2 = 0.0
    dot = 0.0
    nonfinite = 0
    for expected, actual in zip(reference, candidate, strict=True):
        if not math.isfinite(expected) or not math.isfinite(actual):
            nonfinite += 1
            continue
        difference = actual - expected
        max_abs = max(max_abs, abs(difference))
        diff_l2 += difference * difference
        reference_l2 += expected * expected
        candidate_l2 += actual * actual
        dot += expected * actual
    cosine_denominator = math.sqrt(reference_l2 * candidate_l2)
    return {
        "elements": len(reference),
        "exact_f32_le_bytes": reference_bytes == candidate_bytes,
        "baseline_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "max_abs": max_abs,
        "relative_l2": math.sqrt(diff_l2) / math.sqrt(reference_l2) if reference_l2 else 0.0,
        "cosine_similarity": dot / cosine_denominator if cosine_denominator else 1.0,
        "nonfinite_count": nonfinite,
    }


def index_requests(result: dict[str, Any], path: Path) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for request in result["requests"]:
        if not isinstance(request, dict):
            raise ValueError(f"invalid request record in {path}")
        prompt = request.get("prompt_token_ids")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"missing prompt tokens in {path}")
        length = len(prompt)
        if length in indexed:
            raise ValueError(f"duplicate prompt length {length} in {path}")
        indexed[length] = request
    return indexed


def compare(
    prompt_tokens: int,
    baseline_path: Path,
    candidate_path: Path,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if baseline.get("prompt_token_ids") != candidate.get("prompt_token_ids"):
        raise ValueError(f"prompt {prompt_tokens}: input token IDs differ")
    baseline_capture = baseline.get("oracle_capture")
    candidate_capture = candidate.get("oracle_capture")
    if not isinstance(baseline_capture, dict) or not isinstance(candidate_capture, dict):
        raise ValueError(f"prompt {prompt_tokens}: missing oracle capture")
    return {
        "prompt_tokens": prompt_tokens,
        "generated_token_ids": {
            "baseline": baseline.get("generated_token_ids"),
            "candidate": candidate.get("generated_token_ids"),
            "exact": baseline.get("generated_token_ids") == candidate.get("generated_token_ids"),
        },
        "top1": {
            "baseline_token_id": baseline_capture.get("top1_token_id"),
            "candidate_token_id": candidate_capture.get("top1_token_id"),
            "exact_token": baseline_capture.get("top1_token_id") == candidate_capture.get("top1_token_id"),
            "baseline_logit": baseline_capture.get("top1_logit"),
            "candidate_logit": candidate_capture.get("top1_logit"),
        },
        "final_hidden": f32_delta(
            capture_path(baseline_path, baseline_capture, "final_hidden_file"),
            capture_path(candidate_path, candidate_capture, "final_hidden_file"),
        ),
        "logits": f32_delta(
            capture_path(baseline_path, baseline_capture, "logits_file"),
            capture_path(candidate_path, candidate_capture, "logits_file"),
        ),
        "baseline_prefill_execution_units": baseline.get("prefill_execution_units"),
        "candidate_prefill_execution_units": candidate.get("prefill_execution_units"),
    }


def main() -> int:
    args = parse_args()
    baseline_result = load_result(args.baseline_result)
    candidate_result = load_result(args.candidate_result)
    baseline = index_requests(baseline_result, args.baseline_result)
    candidate = index_requests(candidate_result, args.candidate_result)
    if set(baseline) != set(candidate):
        raise ValueError(f"prompt coverage differs: {sorted(baseline)} vs {sorted(candidate)}")
    output = {
        "schema_version": "ullm.prefill_attention_redesign.oracle_comparison.v1",
        "policy": "metrics are recorded for review; no exact-match or scalar numerical threshold is a pass/fail gate",
        "baseline": {
            "result": str(args.baseline_result),
            "runner_git_commit": baseline_result.get("runner_git_commit"),
            "runner_binary_sha256": baseline_result.get("runner_binary_sha256"),
        },
        "candidate": {
            "result": str(args.candidate_result),
            "runner_git_commit": candidate_result.get("runner_git_commit"),
            "runner_binary_sha256": candidate_result.get("runner_binary_sha256"),
        },
        "comparisons": [
            compare(length, args.baseline_result, args.candidate_result, baseline[length], candidate[length])
            for length in sorted(baseline)
        ],
    }
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(output, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
