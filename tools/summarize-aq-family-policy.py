#!/usr/bin/env python3
"""Summarize simple family-level aq precision policies from JSONL rows."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_sample_elements(row: dict[str, Any]) -> int:
    metrics = row.get("metrics", {})
    scope = row.get("scope", {})
    if "sampled_elements" in metrics:
        return int(metrics["sampled_elements"])
    if "sampled_elements" in scope:
        return int(scope["sampled_elements"])
    if "sampled_groups" in scope and "group_size" in scope:
        return int(scope["sampled_groups"]) * int(scope["group_size"])
    raise ValueError(f"row has no sampled element count: {scope.get('tensor_names')}")


def row_bpp(row: dict[str, Any]) -> float:
    metrics = row.get("metrics", {})
    quantization = row.get("quantization", {})
    if metrics.get("effective_bpp") is not None:
        return float(metrics["effective_bpp"])
    if quantization.get("effective_bpp") is not None:
        return float(quantization["effective_bpp"])
    raise ValueError(f"row has no effective bpp: {row.get('scope', {}).get('tensor_names')}")


def row_param_count(row: dict[str, Any]) -> int:
    shape = row.get("scope", {}).get("tensor_shape")
    if not shape:
        raise ValueError(f"row has no tensor shape: {row.get('scope', {}).get('tensor_names')}")
    return int(math.prod(int(dim) for dim in shape))


def weighted_components(row: dict[str, Any]) -> tuple[float, float]:
    metrics = row["metrics"]
    n = row_sample_elements(row)
    weighted_mse = float(metrics["weighted_mse"])
    weighted_rel = float(metrics["weighted_relative_mse"])
    sse = weighted_mse * n
    denom = sse / max(weighted_rel, 1e-30)
    return sse, denom


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    sse = denom = params = bpp_params = 0.0
    for row in rows:
        row_sse, row_denom = weighted_components(row)
        row_params = row_param_count(row)
        sse += row_sse
        denom += row_denom
        params += row_params
        bpp_params += row_bpp(row) * row_params
    return {
        "rows": len(rows),
        "parameter_weighted_bpp": bpp_params / params,
        "combined_weighted_relative_mse": sse / denom,
    }


def group_aq_rows(rows: list[dict[str, Any]], low_candidate: str, high_candidate: str) -> dict[str, dict[str, list[dict]]]:
    grouped: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        candidate = row.get("candidate", {}).get("candidate_id")
        if candidate not in {low_candidate, high_candidate}:
            continue
        family = row["scope"]["families"][0]
        grouped.setdefault(family, {}).setdefault(candidate, []).append(row)
    missing = [
        family
        for family, by_candidate in grouped.items()
        if low_candidate not in by_candidate or high_candidate not in by_candidate
    ]
    if missing:
        raise SystemExit(f"families missing low/high candidates: {', '.join(sorted(missing))}")
    return grouped


def summarize_family_options(
    grouped: dict[str, dict[str, list[dict]]],
    low_candidate: str,
    high_candidate: str,
) -> dict[str, dict[str, dict[str, float | int]]]:
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for family, by_candidate in sorted(grouped.items()):
        result[family] = {
            low_candidate: summarize_rows(by_candidate[low_candidate]),
            high_candidate: summarize_rows(by_candidate[high_candidate]),
        }
    return result


def policy_summary(
    grouped: dict[str, dict[str, list[dict]]],
    low_candidate: str,
    high_candidate: str,
    high_families: set[str],
) -> dict[str, Any]:
    rows = []
    for family, by_candidate in grouped.items():
        candidate = high_candidate if family in high_families else low_candidate
        rows.extend(by_candidate[candidate])
    summary = summarize_rows(rows)
    summary["high_candidate_families"] = sorted(high_families)
    return summary


def best_policies_by_cap(
    grouped: dict[str, dict[str, list[dict]]],
    low_candidate: str,
    high_candidate: str,
    bpp_caps: list[float],
) -> list[dict[str, Any]]:
    families = sorted(grouped)
    policies = []
    for flags in itertools.product([False, True], repeat=len(families)):
        high_families = {family for family, use_high in zip(families, flags, strict=True) if use_high}
        policies.append(policy_summary(grouped, low_candidate, high_candidate, high_families))

    best = []
    for cap in bpp_caps:
        feasible = [policy for policy in policies if float(policy["parameter_weighted_bpp"]) <= cap + 1e-12]
        if not feasible:
            continue
        selected = min(feasible, key=lambda item: float(item["combined_weighted_relative_mse"]))
        best.append({"bpp_cap": cap, **selected})
    return best


def parse_baseline(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("baseline must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("baseline name must not be empty")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aq-results", type=Path, required=True)
    parser.add_argument("--low-candidate", default="aq4_e4m3_g16_ts_flloyd16")
    parser.add_argument("--high-candidate", default="aq4_e4m3_g8_ts_flloyd16")
    parser.add_argument("--bpp-cap", type=float, action="append", default=[4.5, 4.6, 4.7, 4.8, 4.9, 5.0])
    parser.add_argument("--baseline", type=parse_baseline, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    aq_rows = load_rows(args.aq_results)
    grouped = group_aq_rows(aq_rows, args.low_candidate, args.high_candidate)
    baselines = {}
    for name, path in args.baseline:
        rows = [row for row in load_rows(path) if row.get("status") == "ok"]
        baselines[name] = summarize_rows(rows)

    result = {
        "schema_version": "aq-family-policy-summary-v0.1",
        "aq_results": str(args.aq_results),
        "low_candidate": args.low_candidate,
        "high_candidate": args.high_candidate,
        "family_options": summarize_family_options(grouped, args.low_candidate, args.high_candidate),
        "best_policies_by_bpp_cap": best_policies_by_cap(
            grouped,
            args.low_candidate,
            args.high_candidate,
            args.bpp_cap,
        ),
        "baselines": baselines,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
