#!/usr/bin/env python3
"""Compare two synchronized full-model decode timing summaries.

The summaries must have been produced by
``summarize-synchronized-decode-timing.py``.  This utility deliberately makes
no judgement about numerical agreement or output quality: it answers only the
speed-first question of whether the candidate's pooled feedback-decode
throughput exceeds the baseline's under the recorded method.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "ullm.synchronized_decode_timing_comparison.v1"


def load_summary(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} summary {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} summary root is not an object")
    if document.get("schema_version") != "ullm.synchronized_decode_timing_summary.v1":
        raise ValueError(f"{label} summary has an unexpected schema")
    aggregate = document.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError(f"{label} summary has no aggregate")
    value = aggregate.get("pooled_tokens_per_second")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} pooled throughput is not numeric")
    throughput = float(value)
    if not math.isfinite(throughput) or throughput <= 0.0:
        raise ValueError(f"{label} pooled throughput is not finite and positive")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite output: {output}")
    baseline_path = args.baseline_summary.expanduser().resolve()
    candidate_path = args.candidate_summary.expanduser().resolve()
    try:
        baseline = load_summary(baseline_path, "baseline")
        candidate = load_summary(candidate_path, "candidate")
    except ValueError as error:
        raise SystemExit(str(error)) from error

    baseline_tps = float(baseline["aggregate"]["pooled_tokens_per_second"])
    candidate_tps = float(candidate["aggregate"]["pooled_tokens_per_second"])
    result = {
        "schema_version": SCHEMA,
        "method": "pooled synchronized feedback-decode tokens per second",
        "baseline_summary": str(baseline_path),
        "candidate_summary": str(candidate_path),
        "baseline_tokens_per_second": baseline_tps,
        "candidate_tokens_per_second": candidate_tps,
        "candidate_over_baseline_ratio": candidate_tps / baseline_tps,
        "candidate_delta_tokens_per_second": candidate_tps - baseline_tps,
        "candidate_faster": candidate_tps > baseline_tps,
        "decision_rule": "strictly greater pooled tokens/s; quality is evaluated separately",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
