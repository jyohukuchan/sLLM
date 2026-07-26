#!/usr/bin/env python3
"""Summarize a conditional SQ8 v0.2 preliminary decode timing comparison.

This tool deliberately accepts only routes whose numerical preliminary result
has already passed and whose selected multi-tile branch was observed.  It is
not an admission or production benchmark summarizer.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


class SummaryError(RuntimeError):
    pass


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SummaryError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SummaryError(f"{path} is not an object")
    return value


def summarize(path: Path) -> dict[str, Any]:
    value = load(path)
    if value.get("passed") is not True:
        raise SummaryError(f"serving runner did not pass: {path}")
    requests = value.get("requests")
    if not isinstance(requests, list) or len(requests) != 1 or not isinstance(requests[0], dict):
        raise SummaryError(f"serving runner does not have exactly one request: {path}")
    request = requests[0]
    tokens = request.get("generated_token_ids")
    steps = request.get("generated_steps")
    if not isinstance(tokens, list) or not isinstance(steps, list) or len(tokens) != len(steps) or len(steps) < 2:
        raise SummaryError(f"serving runner has insufficient generated timing: {path}")
    samples: list[float] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("generated_index") != index:
            raise SummaryError(f"generated step indexing differs at {path} index={index}")
        if step.get("token_id") != tokens[index]:
            raise SummaryError(f"generated token binding differs at {path} index={index}")
        if index:
            seconds = step.get("synchronized_seconds")
            if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds <= 0:
                raise SummaryError(f"invalid synchronized_seconds at {path} index={index}")
            samples.append(float(seconds))
    return {
        "result": str(path.resolve()),
        "generated_token_ids": tokens,
        "m1_generated_indices": list(range(1, len(tokens))),
        "m1_samples_seconds": samples,
        "m1_sample_count": len(samples),
        "mean_seconds": statistics.fmean(samples),
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "split_source_tile": value.get("paged_decode_split_source_tile"),
        "multi_tile_policy": value.get("paged_decode_split_multi_tile_policy"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    try:
        direct = summarize(args.direct)
        candidates: dict[str, Any] = {}
        for path in args.candidate:
            route = path.parent.name
            if route in candidates:
                raise SummaryError(f"duplicate candidate route {route}")
            value = summarize(path)
            value["token_exact_match_direct"] = value["generated_token_ids"] == direct["generated_token_ids"]
            value["speedup_vs_direct"] = direct["mean_seconds"] / value["mean_seconds"]
            candidates[route] = value
        result = {
            "schema_version": "ullm.sq8.gate.v0.2.preliminary-performance.v1",
            "status": "conditional_preliminary_speed_measurement",
            "timing_scope": "synchronized whole-model M=1 generated indices 1..N; excludes model load, prefill, reset, and oracle capture",
            "direct": direct,
            "candidates": candidates,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except SummaryError as exc:
        raise SystemExit(f"performance summary failed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
