#!/usr/bin/env python3
"""Summarize isolated full-model M=1 tile-route timings without overwriting evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


SCHEMA = "ullm.sq8_0.paged_decode_tile_performance_summary.v1"
ROUTES = {"direct": None, "tile128": 128, "tile256": 256}
MULTI_TILE_POLICY = "direct-fallback-exact-state.v1"


class SummaryError(RuntimeError):
    """A malformed performance result."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SummaryError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SummaryError(f"cannot read {path}: {error}") from error
    require(isinstance(value, dict), f"{path} must contain an object")
    return value


def finite_seconds(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    seconds = float(value)
    require(math.isfinite(seconds) and seconds > 0.0, f"{label} must be positive and finite")
    return seconds


def summarize_route(root: Path, route: str, expected_tile: int | None) -> dict[str, Any]:
    path = root / "performance" / route / "result.json"
    result = load_json(path)
    require(result.get("passed") is True, f"{route} runner did not pass")
    require(result.get("paged_decode_split_source_tile") == expected_tile, f"{route} tile selection differs")
    expected_policy = None if expected_tile is None else MULTI_TILE_POLICY
    require(
        result.get("paged_decode_split_multi_tile_policy") == expected_policy,
        f"{route} multi-tile policy differs",
    )
    requests = result.get("requests")
    require(isinstance(requests, list) and len(requests) == 1 and isinstance(requests[0], dict), f"{route} request shape differs")
    request = requests[0]
    generated = request.get("generated_token_ids")
    steps = request.get("generated_steps")
    require(isinstance(generated, list) and len(generated) >= 2, f"{route} has fewer than two generated tokens")
    require(isinstance(steps, list) and len(steps) == len(generated), f"{route} generated steps differ")
    samples: list[float] = []
    for expected_index, step in enumerate(steps):
        require(isinstance(step, dict), f"{route} generated step {expected_index} is not an object")
        require(step.get("generated_index") == expected_index, f"{route} generated index differs")
        require(step.get("token_id") == generated[expected_index], f"{route} generated token differs")
        if expected_index > 0:
            samples.append(finite_seconds(step.get("synchronized_seconds"), f"{route} step {expected_index}"))
    return {
        "result": str(path),
        "generated_token_ids": generated,
        "m1_generated_indices": list(range(1, len(generated))),
        "m1_samples_seconds": samples,
        "m1_sample_count": len(samples),
        "mean_seconds": statistics.fmean(samples),
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "split_source_tile": expected_tile,
        "multi_tile_policy": expected_policy,
    }


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    output = root / "performance-summary.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing summary: {output}")
    try:
        routes = {route: summarize_route(root, route, tile) for route, tile in ROUTES.items()}
        reference = routes["direct"]
        for route, result in routes.items():
            result["token_exact_match_direct"] = result["generated_token_ids"] == reference["generated_token_ids"]
            result["speedup_vs_direct"] = reference["mean_seconds"] / result["mean_seconds"]
        summary = {
            "schema_version": SCHEMA,
            "result_dir": str(root),
            "timing_scope": "synchronized whole-model M=1 generated indices 1..N; excludes model load, prefill, reset, and oracle capture",
            "routes": routes,
        }
        write_json_new(output, summary)
    except SummaryError as error:
        raise SystemExit(f"performance summary failed: {error}") from error
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
