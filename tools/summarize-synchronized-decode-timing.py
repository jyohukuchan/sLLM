#!/usr/bin/env python3
"""Summarize full-model decode timing recorded by a serving result JSON.

The input is intentionally the synchronized per-generated-token timing emitted
by ``sq8_ck_serving --record-generated-timing``.  GPU event ranges, profiler
durations, model-load time, and the prefill/first-token step are excluded.
This makes the reported tok/s an end-to-end decode measurement, not a kernel
or profiler proxy.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


SCHEMA = "ullm.synchronized_decode_timing_summary.v1"


def strict_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root is not an object")
    return value


def finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: not a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label}: must be finite and positive")
    return result


def summarize_run(path: Path) -> dict[str, Any]:
    document = strict_object(path)
    if document.get("passed") is not True:
        raise ValueError(f"{path}: serving result did not pass")
    requests = document.get("requests")
    if not isinstance(requests, list) or len(requests) != 1 or not isinstance(requests[0], dict):
        raise ValueError(f"{path}: expected exactly one request")
    request = requests[0]
    steps = request.get("generated_steps")
    if not isinstance(steps, list):
        raise ValueError(f"{path}: lacks --record-generated-timing generated_steps")
    decode: list[dict[str, Any]] = []
    for offset, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"{path}: generated step {offset} is not an object")
        index = step.get("generated_index")
        if isinstance(index, bool) or not isinstance(index, int) or index != offset:
            raise ValueError(f"{path}: generated indices are not contiguous from zero")
        if index == 0:
            continue
        seconds = finite_positive(step.get("synchronized_seconds"), f"{path}: step {index}")
        decode.append({"generated_index": index, "synchronized_seconds": seconds})
    if not decode:
        raise ValueError(f"{path}: no feedback decode token was recorded")
    total_seconds = sum(float(item["synchronized_seconds"]) for item in decode)
    tokens = len(decode)
    return {
        "result_json": str(path),
        "schema_version": document.get("schema_version"),
        "runner_binary_sha256": document.get("runner_binary_sha256"),
        "handwritten_wmma_projection_prototype": document.get("handwritten_wmma_projection_prototype"),
        "device": document.get("device"),
        "tokens": tokens,
        "decode_seconds": total_seconds,
        "milliseconds_per_token": total_seconds * 1000.0 / tokens,
        "tokens_per_second": tokens / total_seconds,
        "per_token_seconds": [float(item["synchronized_seconds"]) for item in decode],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--result-json", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    paths = [path.expanduser().resolve() for path in args.result_json]
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    if len({str(path) for path in paths}) != len(paths):
        raise SystemExit("duplicate --result-json input")
    try:
        runs = [summarize_run(path) for path in paths]
    except ValueError as error:
        raise SystemExit(str(error)) from error
    values = sorted(float(run["tokens_per_second"]) for run in runs)
    milliseconds = sorted(float(run["milliseconds_per_token"]) for run in runs)
    tokens = sum(int(run["tokens"]) for run in runs)
    seconds = sum(float(run["decode_seconds"]) for run in runs)
    result = {
        "schema_version": SCHEMA,
        "label": args.label,
        "method": {
            "timing_source": "sq8_ck_serving generated_steps[].synchronized_seconds",
            "included_generated_indices": "1 and later (feedback decode only)",
            "excluded": ["model_load", "prefill", "generated_index=0", "profiler ranges", "GPU event timing"],
        },
        "runs": runs,
        "aggregate": {
            "run_count": len(runs),
            "feedback_decode_tokens": tokens,
            "feedback_decode_seconds": seconds,
            "pooled_tokens_per_second": tokens / seconds,
            "pooled_milliseconds_per_token": seconds * 1000.0 / tokens,
            "per_run_tokens_per_second": values,
            "median_tokens_per_second": statistics.median(values),
            "minimum_tokens_per_second": values[0],
            "maximum_tokens_per_second": values[-1],
            "median_milliseconds_per_token": statistics.median(milliseconds),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
