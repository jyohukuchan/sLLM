#!/usr/bin/env python3
"""Summarize module contribution traces from golden-prefix JSONL rows."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "golden-prefix-module-contribution-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize package-golden-prefix-smoke module_contribution traces."
    )
    parser.add_argument("jsonl_paths", nargs="+", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def parse_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def parse_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def range_label(row: dict[str, Any]) -> str:
    start = parse_int(row.get("layer_start"))
    end = parse_int(row.get("layer_end_exclusive"))
    if start is None or end is None:
        return "-"
    return f"{start}..{end}"


def module_dominance(trace: dict[str, Any]) -> str:
    attention = abs(parse_float(trace.get("attention_output")) or 0.0)
    mlp = abs(parse_float(trace.get("mlp_output")) or 0.0)
    if attention >= mlp * 1.5:
        return "attention"
    if mlp >= attention * 1.5:
        return "mlp"
    return "mixed"


def failure_shape(trace: dict[str, Any]) -> str:
    actual_delta = parse_float(trace.get("actual_delta"))
    expected_delta = parse_float(trace.get("expected_delta"))
    if actual_delta is None or expected_delta is None:
        return "unknown"
    abs_actual = abs(actual_delta)
    abs_expected = abs(expected_delta)
    if abs_expected > 1.0 and abs_expected >= abs_actual * 3.0:
        return "missing_expected_delta"
    if abs_actual > 1.0 and abs_actual >= abs_expected * 3.0:
        return "spurious_actual_delta"
    if actual_delta * expected_delta < 0.0 and abs_actual > 0.5 and abs_expected > 0.5:
        return "opposite_delta"
    return "mixed_delta"


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            module = item.get("module_contribution")
            if not isinstance(module, dict):
                continue
            trace = module.get("max_output_diff_trace")
            delta_distribution = module.get("delta_distribution")
            if not isinstance(trace, dict) or not isinstance(delta_distribution, dict):
                continue
            delta_location = delta_distribution.get("max_abs_diff_location")
            if not isinstance(delta_location, dict):
                delta_location = {}
            output_diff = parse_float(trace.get("output_diff"))
            delta_diff = parse_float(trace.get("delta_diff"))
            input_diff = parse_float(trace.get("input_diff"))
            row = {
                "source": str(path),
                "line_number": line_number,
                "layer_index": parse_int(item.get("layer_index")),
                "layer_kind": item.get("layer_kind"),
                "backend": item.get("backend"),
                "device_index": parse_int(item.get("device_index")),
                "run_mode": item.get("run_mode"),
                "range": range_label(item),
                "hot_hidden_index": parse_int(module.get("hot_hidden_index")),
                "token_index": parse_int(trace.get("token_index")),
                "output_diff": output_diff,
                "input_diff": input_diff,
                "delta_diff": delta_diff,
                "attention_output": parse_float(trace.get("attention_output")),
                "mlp_output": parse_float(trace.get("mlp_output")),
                "actual_delta": parse_float(trace.get("actual_delta")),
                "expected_delta": parse_float(trace.get("expected_delta")),
                "dominant_actual_module": module_dominance(trace),
                "failure_shape": failure_shape(trace),
                "delta_max_abs": parse_float(delta_location.get("abs_diff")),
                "delta_max_token": parse_int(delta_location.get("token_index")),
                "delta_max_hidden": parse_int(delta_location.get("hidden_index")),
            }
            if output_diff not in (None, 0.0) and delta_diff is not None:
                row["abs_delta_diff_over_abs_output_diff"] = abs(delta_diff) / abs(output_diff)
            else:
                row["abs_delta_diff_over_abs_output_diff"] = None
            if input_diff is not None and output_diff is not None:
                row["input_diff_share"] = abs(input_diff) / abs(output_diff) if output_diff else None
            else:
                row["input_diff_share"] = None
            rows.append(row)
    return rows


def abs_value(row: dict[str, Any], key: str) -> float:
    value = parse_float(row.get(key))
    return abs(value) if value is not None else -1.0


def build_summary(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_rows(path))
    rows.sort(
        key=lambda row: (
            str(row.get("run_mode")),
            row.get("layer_index") if row.get("layer_index") is not None else 10**9,
            row.get("line_number") if row.get("line_number") is not None else 10**9,
        )
    )
    hidden_counts = Counter(str(row.get("hot_hidden_index")) for row in rows)
    dominance_counts = Counter(str(row.get("dominant_actual_module")) for row in rows)
    failure_shape_counts = Counter(str(row.get("failure_shape")) for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sources": [str(path) for path in paths],
        "row_count": len(rows),
        "hot_hidden_counts": dict(hidden_counts.most_common()),
        "dominance_counts": dict(dominance_counts.most_common()),
        "failure_shape_counts": dict(failure_shape_counts.most_common()),
        "worst_output_diff": max(rows, key=lambda row: abs_value(row, "output_diff"), default=None),
        "worst_delta_diff": max(rows, key=lambda row: abs_value(row, "delta_diff"), default=None),
        "rows": rows,
    }


def fmt(value: Any, digits: int = 6) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "-"
    return f"{parsed:.{digits}g}"


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| layer | kind | mode | range | hot | tok | output_diff | input_diff | delta_diff | attn | mlp | expected_delta | actual_delta | dominant | shape |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("layer_index") if row.get("layer_index") is not None else "-"),
                    str(row.get("layer_kind") or "-"),
                    str(row.get("run_mode") or "-"),
                    str(row.get("range") or "-"),
                    str(row.get("hot_hidden_index") if row.get("hot_hidden_index") is not None else "-"),
                    str(row.get("token_index") if row.get("token_index") is not None else "-"),
                    fmt(row.get("output_diff")),
                    fmt(row.get("input_diff")),
                    fmt(row.get("delta_diff")),
                    fmt(row.get("attention_output")),
                    fmt(row.get("mlp_output")),
                    fmt(row.get("expected_delta")),
                    fmt(row.get("actual_delta")),
                    str(row.get("dominant_actual_module") or "-"),
                    str(row.get("failure_shape") or "-"),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    summary = build_summary(args.jsonl_paths)
    if args.summary_json:
        write_json(args.summary_json, summary)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(build_markdown(summary["rows"]), encoding="utf-8")
    print(
        "golden-prefix-module-contribution "
        f"rows={summary['row_count']} "
        f"hot_hidden_counts={summary['hot_hidden_counts']} "
        f"dominance_counts={summary['dominance_counts']} "
        f"failure_shape_counts={summary['failure_shape_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
