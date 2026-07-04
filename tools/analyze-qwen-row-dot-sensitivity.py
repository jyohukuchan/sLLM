#!/usr/bin/env python3
"""Summarize row-dot sensitivity from Qwen module trace JSONL files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "qwen-row-dot-sensitivity-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fullref-jsonl",
        action="append",
        required=True,
        type=Path,
        help="Full-reference module trace JSONL produced by export-qwen-layer-module-trace.py.",
    )
    parser.add_argument(
        "--projection",
        action="append",
        default=[],
        help="Projection key to include. Defaults to all row_dot projections.",
    )
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def f(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def rmse(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def optimal_scale(entries: list[dict[str, Any]]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for entry in entries:
        source = f(entry.get("source_row_dot"))
        package = f(entry.get("package_row_dot"))
        if source is None or package is None:
            continue
        numerator += source * package
        denominator += package * package
    if denominator <= 0.0:
        return None
    return numerator / denominator


def summarize_projection(
    source_path: Path,
    row: dict[str, Any],
    projection_name: str,
    projection: dict[str, Any],
) -> dict[str, Any] | None:
    entries = [
        entry
        for entry in projection.get("per_token", [])
        if isinstance(entry, dict)
        and f(entry.get("package_row_dot")) is not None
        and f(entry.get("source_row_dot")) is not None
    ]
    if not entries:
        return None

    errors = [
        float(entry["package_row_dot"]) - float(entry["source_row_dot"])
        for entry in entries
    ]
    scale = optimal_scale(entries)
    scaled_errors: list[float] = []
    if scale is not None:
        for entry in entries:
            source = float(entry["source_row_dot"])
            package = float(entry["package_row_dot"])
            scaled_errors.append(package * scale - source)

    worst_index, worst_error = max(
        enumerate(errors),
        key=lambda item: abs(item[1]),
    )
    worst = entries[worst_index]
    worst_scaled_error = (
        scaled_errors[worst_index]
        if scaled_errors and worst_index < len(scaled_errors)
        else None
    )
    original_rmse = rmse(errors)
    scaled_rmse = rmse(scaled_errors) if scaled_errors else None
    improvement_ratio = None
    if scaled_rmse is not None and original_rmse > 0.0:
        improvement_ratio = 1.0 - (scaled_rmse / original_rmse)

    return {
        "source_path": str(source_path),
        "package_dir": row.get("package_dir"),
        "fixture": row.get("fixture"),
        "layer_index": row.get("layer_index"),
        "layer_type": row.get("layer_type"),
        "hidden_index": row.get("hidden_index"),
        "projection": projection_name,
        "token_count": len(entries),
        "package_row_l2_norm": projection.get("package_row_l2_norm"),
        "original_rmse": original_rmse,
        "original_max_abs_error": max(abs(value) for value in errors),
        "original_mean_error": sum(errors) / len(errors),
        "optimal_scale": scale,
        "scaled_rmse": scaled_rmse,
        "scaled_max_abs_error": max(abs(value) for value in scaled_errors) if scaled_errors else None,
        "scaled_mean_error": sum(scaled_errors) / len(scaled_errors) if scaled_errors else None,
        "scale_improvement_ratio": improvement_ratio,
        "worst_token_index": worst.get("token_index"),
        "worst_error": worst_error,
        "worst_scaled_error": worst_scaled_error,
        "worst_source_row_dot": worst.get("source_row_dot"),
        "worst_package_row_dot": worst.get("package_row_dot"),
        "worst_module_output": worst.get("module_output"),
    }


def build_summary(paths: list[Path], projections: set[str] | None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            row_dot = row.get("row_dot")
            if not isinstance(row_dot, dict):
                continue
            for projection_name, projection in row_dot.items():
                if projections is not None and projection_name not in projections:
                    continue
                if not isinstance(projection, dict):
                    continue
                summary = summarize_projection(path, row, str(projection_name), projection)
                if summary is not None:
                    rows.append(summary)
    rows.sort(
        key=lambda item: (
            -float(item.get("original_max_abs_error") or 0.0),
            str(item.get("source_path") or ""),
            str(item.get("projection") or ""),
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_paths": [str(path) for path in paths],
        "projection_filter": sorted(projections) if projections is not None else None,
        "row_count": len(rows),
        "rows": rows,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 9) -> str:
    value_f = f(value)
    if value_f is None:
        return "-"
    return f"{value_f:.{digits}g}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "| layer | hidden | projection | tokens | original_rmse | original_max_abs | optimal_scale | scaled_rmse | scaled_max_abs | improvement | worst_token | worst_error | worst_scaled_error |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        improvement = row.get("scale_improvement_ratio")
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row.get("layer_index", "-"),
                row.get("hidden_index", "-"),
                row.get("projection", "-"),
                row.get("token_count", "-"),
                fmt(row.get("original_rmse")),
                fmt(row.get("original_max_abs_error")),
                fmt(row.get("optimal_scale"), 12),
                fmt(row.get("scaled_rmse")),
                fmt(row.get("scaled_max_abs_error")),
                fmt(improvement),
                row.get("worst_token_index", "-"),
                fmt(row.get("worst_error")),
                fmt(row.get("worst_scaled_error")),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    projection_filter = set(args.projection) if args.projection else None
    summary = build_summary(args.fullref_jsonl, projection_filter)
    write_json(args.summary_json, summary)
    if args.markdown is not None:
        write_markdown(args.markdown, summary)
    print(f"qwen-row-dot-sensitivity rows={summary['row_count']} output={args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
