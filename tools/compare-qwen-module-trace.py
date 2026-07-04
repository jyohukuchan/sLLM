#!/usr/bin/env python3
"""Compare package module-contribution rows with full-reference layer traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "qwen-module-trace-comparison-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-jsonl", type=Path, required=True)
    parser.add_argument("--fullref-jsonl", type=Path, required=True)
    parser.add_argument("--run-mode", default="golden_before_each_layer")
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


def float_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def trace_by_token(row: dict[str, Any], token_index: int) -> dict[str, Any] | None:
    for trace in row.get("per_token_hidden_trace", []):
        if int(trace.get("token_index", -1)) == token_index:
            return trace
    return None


def diff(package_value: Any, fullref_value: Any) -> float | None:
    package_float = float_value(package_value)
    fullref_float = float_value(fullref_value)
    if package_float is None or fullref_float is None:
        return None
    return package_float - fullref_float


def build_summary(package_rows: list[dict[str, Any]], fullref_rows: list[dict[str, Any]], run_mode: str) -> dict[str, Any]:
    fullref_by_layer = {int(row["layer_index"]): row for row in fullref_rows}
    comparisons: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in package_rows:
        if row.get("run_mode") != run_mode:
            continue
        layer = int(row["layer_index"])
        fullref = fullref_by_layer.get(layer)
        if fullref is None:
            skipped.append({"layer_index": layer, "reason": "missing_full_reference"})
            continue
        module = row.get("module_contribution", {})
        package_trace = module.get("max_output_diff_trace", {})
        token = int(package_trace.get("token_index"))
        full_trace = trace_by_token(fullref, token)
        if full_trace is None:
            skipped.append({"layer_index": layer, "token_index": token, "reason": "missing_full_reference_token"})
            continue
        comparison = {
            "layer_index": layer,
            "layer_kind": row.get("layer_kind"),
            "run_mode": row.get("run_mode"),
            "token_index": token,
            "hidden_index": package_trace.get("hidden_index"),
            "package_output_diff": package_trace.get("output_diff"),
            "expected_delta": package_trace.get("expected_delta"),
            "package_actual_delta": package_trace.get("actual_delta"),
            "fullref_actual_delta": full_trace.get("actual_delta"),
            "actual_delta_error": diff(package_trace.get("actual_delta"), full_trace.get("actual_delta")),
            "package_attention_output": package_trace.get("attention_output"),
            "fullref_attention_output": full_trace.get("attention_output"),
            "attention_error": diff(package_trace.get("attention_output"), full_trace.get("attention_output")),
            "package_mlp_output": package_trace.get("mlp_output"),
            "fullref_mlp_output": full_trace.get("mlp_output"),
            "mlp_error": diff(package_trace.get("mlp_output"), full_trace.get("mlp_output")),
            "fullref_fixture_max_abs_diff": fullref.get("fixture_match", {}).get("max_abs_diff"),
        }
        comparisons.append(comparison)

    worst_delta = max(
        comparisons,
        key=lambda item: abs(float_value(item.get("actual_delta_error")) or 0.0),
        default=None,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_mode": run_mode,
        "row_count": len(comparisons),
        "skipped": skipped,
        "worst_actual_delta_error": worst_delta,
        "rows": comparisons,
    }


def fmt(value: Any) -> str:
    parsed = float_value(value)
    if parsed is None:
        return "-"
    return f"{parsed:.6g}"


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | pkg_attn | full_attn | attn_error | pkg_mlp | full_mlp | mlp_error |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["layer_index"]),
                    str(row["token_index"]),
                    str(row["hidden_index"]),
                    fmt(row["package_output_diff"]),
                    fmt(row["expected_delta"]),
                    fmt(row["package_actual_delta"]),
                    fmt(row["fullref_actual_delta"]),
                    fmt(row["actual_delta_error"]),
                    fmt(row["package_attention_output"]),
                    fmt(row["fullref_attention_output"]),
                    fmt(row["attention_error"]),
                    fmt(row["package_mlp_output"]),
                    fmt(row["fullref_mlp_output"]),
                    fmt(row["mlp_error"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    package_rows = read_jsonl(args.package_jsonl)
    fullref_rows = read_jsonl(args.fullref_jsonl)
    summary = build_summary(package_rows, fullref_rows, args.run_mode)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(summary["rows"]), encoding="utf-8")
    print(
        "qwen-module-trace-comparison "
        f"rows={summary['row_count']} skipped={len(summary['skipped'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
