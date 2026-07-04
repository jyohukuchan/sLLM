#!/usr/bin/env python3
"""Evaluate prefix-smoke candidate gates from a Qwen prefix smoke summary."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "qwen-prefix-candidate-gates-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--baseline-condition", default="baseline")
    parser.add_argument("--candidate-condition", action="append", default=[])
    parser.add_argument("--max-fixture-worsen", type=float, default=0.001)
    parser.add_argument("--min-median-improvement", type=float, default=0.005)
    parser.add_argument("--min-fixture-count", type=int, default=3)
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    reports = summary.get("reports")
    if not isinstance(reports, list):
        raise ValueError(f"summary has no reports array: {path}")
    return summary


def split_label(label: str) -> tuple[str, str]:
    if "-" not in label:
        raise ValueError(f"report label must be FIXTURE-CONDITION, got {label!r}")
    fixture, condition = label.split("-", 1)
    if not fixture or not condition:
        raise ValueError(f"report label must be FIXTURE-CONDITION, got {label!r}")
    return fixture, condition


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def reports_by_fixture_condition(summary: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    reports = {}
    for report in summary["reports"]:
        if not isinstance(report, dict):
            continue
        label = report.get("label")
        if not isinstance(label, str):
            continue
        fixture, condition = split_label(label)
        key = (fixture, condition)
        if key in reports:
            raise ValueError(f"duplicate report label fixture/condition: {label}")
        reports[key] = report
    return reports


def condition_names(
    reports: dict[tuple[str, str], dict[str, Any]],
    baseline_condition: str,
    selected: list[str],
) -> list[str]:
    if selected:
        return sorted(set(selected))
    return sorted({condition for _, condition in reports if condition != baseline_condition})


def evaluate_condition(
    condition: str,
    reports: dict[tuple[str, str], dict[str, Any]],
    baseline_condition: str,
    max_fixture_worsen: float,
    min_median_improvement: float,
    min_fixture_count: int,
) -> dict[str, Any]:
    fixtures = sorted({fixture for fixture, report_condition in reports if report_condition == condition})
    comparisons = []
    for fixture in fixtures:
        baseline = reports.get((fixture, baseline_condition))
        candidate = reports.get((fixture, condition))
        if baseline is None or candidate is None:
            continue
        baseline_max = number(baseline.get("overall_max_abs_diff"))
        candidate_max = number(candidate.get("overall_max_abs_diff"))
        if baseline_max is None or candidate_max is None:
            continue
        delta = candidate_max - baseline_max
        comparisons.append(
            {
                "fixture": fixture,
                "baseline_label": baseline.get("label"),
                "candidate_label": candidate.get("label"),
                "baseline_max_abs_diff": baseline_max,
                "candidate_max_abs_diff": candidate_max,
                "delta": delta,
                "improvement": baseline_max - candidate_max,
                "baseline_layer": baseline.get("overall_max_layer"),
                "candidate_layer": candidate.get("overall_max_layer"),
                "baseline_token_index": baseline.get("overall_max_token_index"),
                "candidate_token_index": candidate.get("overall_max_token_index"),
                "baseline_hidden_index": baseline.get("overall_max_hidden_index"),
                "candidate_hidden_index": candidate.get("overall_max_hidden_index"),
            }
        )
    improvements = [float(item["improvement"]) for item in comparisons]
    regressions = [float(item["delta"]) for item in comparisons]
    fixture_count = len(comparisons)
    mean_improvement = statistics.fmean(improvements) if improvements else None
    median_improvement = statistics.median(improvements) if improvements else None
    max_regression = max(0.0, max(regressions)) if regressions else None
    hard_reject = any(delta > max_fixture_worsen for delta in regressions)
    if fixture_count < min_fixture_count:
        decision = "needs_more_fixtures"
        reason = f"only {fixture_count} paired fixture(s), need {min_fixture_count}"
    elif hard_reject:
        decision = "reject"
        reason = f"fixture regression exceeds {max_fixture_worsen}"
    elif mean_improvement is not None and median_improvement is not None and mean_improvement >= 0.0 and median_improvement >= min_median_improvement:
        decision = "accept"
        reason = "aggregate and median gates passed"
    else:
        decision = "hold"
        reason = f"median improvement below {min_median_improvement} or aggregate did not improve"
    return {
        "condition": condition,
        "decision": decision,
        "reason": reason,
        "fixture_count": fixture_count,
        "mean_improvement": mean_improvement,
        "median_improvement": median_improvement,
        "max_regression": max_regression,
        "comparisons": comparisons,
    }


def fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.9g}"
    if value is None:
        return "-"
    return str(value)


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen Prefix Candidate Gates",
        "",
        f"- schema: `{summary['schema_version']}`",
        f"- baseline condition: `{summary['baseline_condition']}`",
        f"- max fixture worsen: `{summary['gates']['max_fixture_worsen']}`",
        f"- min median improvement: `{summary['gates']['min_median_improvement']}`",
        f"- min fixture count: `{summary['gates']['min_fixture_count']}`",
        "",
        "| condition | decision | fixtures | mean improvement | median improvement | max regression | reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for candidate in summary["candidates"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(candidate["condition"]),
                    str(candidate["decision"]),
                    fmt(candidate["fixture_count"]),
                    fmt(candidate.get("mean_improvement")),
                    fmt(candidate.get("median_improvement")),
                    fmt(candidate.get("max_regression")),
                    str(candidate["reason"]),
                ]
            )
            + " |"
        )
    for candidate in summary["candidates"]:
        lines.extend(
            [
                "",
                f"## {candidate['condition']}",
                "",
                "| fixture | baseline | candidate | delta | layer | token | hidden |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for comparison in candidate["comparisons"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(comparison["fixture"]),
                        fmt(comparison["baseline_max_abs_diff"]),
                        fmt(comparison["candidate_max_abs_diff"]),
                        fmt(comparison["delta"]),
                        fmt(comparison["candidate_layer"]),
                        fmt(comparison["candidate_token_index"]),
                        fmt(comparison["candidate_hidden_index"]),
                    ]
                )
                + " |"
            )
    return "\n".join(lines).rstrip() + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.min_fixture_count < 1:
        raise SystemExit("--min-fixture-count must be positive")
    summary = read_summary(args.summary_json)
    reports = reports_by_fixture_condition(summary)
    candidates = [
        evaluate_condition(
            condition,
            reports,
            args.baseline_condition,
            args.max_fixture_worsen,
            args.min_median_improvement,
            args.min_fixture_count,
        )
        for condition in condition_names(reports, args.baseline_condition, args.candidate_condition)
    ]
    output = {
        "schema_version": SCHEMA_VERSION,
        "source_summary": str(args.summary_json),
        "baseline_condition": args.baseline_condition,
        "gates": {
            "max_fixture_worsen": args.max_fixture_worsen,
            "min_median_improvement": args.min_median_improvement,
            "min_fixture_count": args.min_fixture_count,
        },
        "candidates": candidates,
    }
    if args.output_json:
        write_json(args.output_json, output)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(output), encoding="utf-8")
    print(f"qwen-prefix-candidate-gates candidates={len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
