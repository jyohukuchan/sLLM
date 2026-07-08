#!/usr/bin/env python3
"""Evaluate SQ FP8 overlay logits guard JSONs against a promotion policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "sq-fp8-overlay-acceptance-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", action="append", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument(
        "--promotion-rule",
        choices=["strict_top1"],
        default="strict_top1",
        help="Rule used to decide whether a case can be promoted beyond T2.",
    )
    parser.add_argument("--diagnostic-min-topk-common", type=int, default=5)
    parser.add_argument("--diagnostic-max-baseline-rank", type=int, default=2)
    parser.add_argument("--diagnostic-max-top1-gap", type=float, default=0.15)
    return parser.parse_args()


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return value


def numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def topk_common(case: dict[str, Any]) -> int | None:
    return int_or_none(case.get("topk_common", case.get("topk_common_count")))


def case_name(path: Path, case: dict[str, Any]) -> str:
    name = case.get("name")
    if isinstance(name, str) and name:
        return f"{path.stem}:{name}"
    return path.stem


def strict_top1_reasons(case: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if case.get("verified") is not True:
        reasons.append("case_not_verified")
    if case.get("top1_match") is not True:
        reasons.append("top1_mismatch")
    return reasons


def diagnostic_reasons(
    case: dict[str, Any],
    *,
    min_topk_common: int,
    max_baseline_rank: int,
    max_top1_gap: float,
) -> list[str]:
    reasons: list[str] = []
    common = topk_common(case)
    if common is None or common < min_topk_common:
        reasons.append("topk_common_below_threshold")
    rank = int_or_none(case.get("baseline_top1_rank_in_sq_topk"))
    if rank is None or rank > max_baseline_rank:
        reasons.append("baseline_top1_rank_above_threshold")
    gap = numeric_or_none(case.get("sq_top1_minus_baseline_top1_logit"))
    if gap is None or abs(gap) > max_top1_gap:
        reasons.append("top1_gap_above_threshold")
    return reasons


def evaluate_case(
    path: Path,
    root: dict[str, Any],
    case: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    strict_reasons = strict_top1_reasons(case)
    diagnostic_failure_reasons = diagnostic_reasons(
        case,
        min_topk_common=args.diagnostic_min_topk_common,
        max_baseline_rank=args.diagnostic_max_baseline_rank,
        max_top1_gap=args.diagnostic_max_top1_gap,
    )
    return {
        "input_json": str(path),
        "case": case_name(path, case),
        "layers": root.get("layers"),
        "token_ids": root.get("token_ids"),
        "fp8_tensor_count": case.get("fp8_tensor_count"),
        "verified": case.get("verified"),
        "baseline_top1": case.get("baseline_top1"),
        "sq_top1": case.get("sq_top1"),
        "top1_match": case.get("top1_match"),
        "baseline_top1_rank_in_sq_topk": case.get("baseline_top1_rank_in_sq_topk"),
        "topk_common": topk_common(case),
        "sq_top1_minus_baseline_top1_logit": case.get(
            "sq_top1_minus_baseline_top1_logit"
        ),
        "strict_top1_pass": not strict_reasons,
        "strict_top1_failure_reasons": strict_reasons,
        "diagnostic_topk_pass": not diagnostic_failure_reasons,
        "diagnostic_topk_failure_reasons": diagnostic_failure_reasons,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    evaluated_cases: list[dict[str, Any]] = []
    for path in args.input_json:
        root = load_json_object(path)
        raw_cases = root.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise SystemExit(f"{path}: expected non-empty cases[]")
        for raw_case in raw_cases:
            if not isinstance(raw_case, dict):
                raise SystemExit(f"{path}: cases[] entries must be objects")
            evaluated_cases.append(evaluate_case(path, root, raw_case, args))

    strict_pass_count = sum(1 for case in evaluated_cases if case["strict_top1_pass"])
    diagnostic_pass_count = sum(1 for case in evaluated_cases if case["diagnostic_topk_pass"])
    strict_passed = strict_pass_count == len(evaluated_cases)
    diagnostic_passed = diagnostic_pass_count == len(evaluated_cases)
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "promotion_rule": args.promotion_rule,
            "diagnostic_rule": "topk_rank_gap",
            "diagnostic_min_topk_common": args.diagnostic_min_topk_common,
            "diagnostic_max_baseline_rank": args.diagnostic_max_baseline_rank,
            "diagnostic_max_top1_gap": args.diagnostic_max_top1_gap,
        },
        "inputs": [str(path) for path in args.input_json],
        "summary": {
            "case_count": len(evaluated_cases),
            "strict_top1_pass_count": strict_pass_count,
            "strict_top1_passed": strict_passed,
            "diagnostic_topk_pass_count": diagnostic_pass_count,
            "diagnostic_topk_passed": diagnostic_passed,
            "accepted_for_t2_promotion": strict_passed,
        },
        "cases": evaluated_cases,
    }


def main() -> int:
    args = parse_args()
    if args.diagnostic_min_topk_common < 0:
        raise SystemExit("--diagnostic-min-topk-common must be non-negative")
    if args.diagnostic_max_baseline_rank <= 0:
        raise SystemExit("--diagnostic-max-baseline-rank must be greater than zero")
    if args.diagnostic_max_top1_gap < 0.0:
        raise SystemExit("--diagnostic-max-top1-gap must be non-negative")
    result = evaluate(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
