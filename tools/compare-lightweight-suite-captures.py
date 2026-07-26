#!/usr/bin/env python3
"""Compare two saved lightweight-promotion prompt-suite capture directories.

The tool is intentionally format-agnostic: both sides merely need one JSON
record named ``<case-id>.json`` per suite case, with the record layout emitted
by the generic served-suite capture or the SQ8_1 CPU reference generator.
It never invokes a model or changes a manifest.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_promotion_module() -> ModuleType:
    path = ROOT / "tools" / "lightweight_promotion.py"
    spec = importlib.util.spec_from_file_location("compare_lightweight_promotion", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared promotion helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROMOTION = load_promotion_module()


def read_record(path: Path, case_id: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict) or value.get("case_id") != case_id:
        raise ValueError(f"{path}: does not contain case {case_id!r}")
    return value


def load_records(directory: Path, suite: tuple[Any, ...]) -> list[dict[str, Any]]:
    if not directory.is_dir():
        raise ValueError(f"missing capture directory: {directory}")
    return [read_record(directory / f"{case.case_id}.json", case.case_id) for case in suite]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-suite", type=Path, default=PROMOTION.DEFAULT_PROMPT_SUITE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.baseline_dir = args.baseline_dir.expanduser().resolve()
    args.candidate_dir = args.candidate_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.prompt_suite = args.prompt_suite.expanduser().resolve()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to use an existing output directory: {args.output_dir}")
    suite = PROMOTION.load_suite(args.prompt_suite)
    try:
        baseline = load_records(args.baseline_dir, suite)
        candidate = load_records(args.candidate_dir, suite)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    args.output_dir.mkdir(parents=True, mode=0o750)
    comparison = PROMOTION.compare_suites(suite, baseline, candidate)
    PROMOTION.write_json_new(args.output_dir / "comparison.json", comparison, "suite comparison")
    PROMOTION.write_comparison_markdown(
        args.output_dir / "comparison.md",
        suite,
        baseline,
        candidate,
        comparison,
    )
    PROMOTION.write_json_new(
        args.output_dir / "comparison-manifest.json",
        {
            "schema_version": "ullm.lightweight_suite_capture_comparison.v1",
            "baseline_dir": str(args.baseline_dir),
            "candidate_dir": str(args.candidate_dir),
            "prompt_suite": str(args.prompt_suite),
            "case_count": len(suite),
            "blocking_findings": comparison["blocking_findings"],
            "passed": comparison["passed"],
        },
        "suite comparison manifest",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
