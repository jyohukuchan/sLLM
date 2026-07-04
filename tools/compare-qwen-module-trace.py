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


def row_dot_by_token(row: dict[str, Any], projection: str, token_index: int) -> dict[str, Any] | None:
    row_dot = row.get("row_dot")
    if not isinstance(row_dot, dict):
        return None
    projection_payload = row_dot.get(projection)
    if not isinstance(projection_payload, dict):
        return None
    for trace in projection_payload.get("per_token", []):
        if int(trace.get("token_index", -1)) == token_index:
            return trace
    return None


def diff(package_value: Any, fullref_value: Any) -> float | None:
    package_float = float_value(package_value)
    fullref_float = float_value(fullref_value)
    if package_float is None or fullref_float is None:
        return None
    return package_float - fullref_float


def get_hot_input_vectors(row: dict[str, Any]) -> dict[str, Any]:
    module = row.get("module_contribution")
    if isinstance(module, dict):
        vectors = module.get("hot_input_vectors")
        if isinstance(vectors, dict):
            return vectors
    legacy = row.get("hot_input_vectors")
    return legacy if isinstance(legacy, dict) else {}


def get_hot_input_vectors_for_token(row: dict[str, Any], token_index: int) -> dict[str, Any]:
    vectors = get_hot_input_vectors(row)
    vector_tokens = [
        summary.get("token_index")
        for summary in vectors.values()
        if isinstance(summary, dict) and "token_index" in summary
    ]
    if vector_tokens and all(token == token_index for token in vector_tokens):
        return vectors

    module = row.get("module_contribution")
    per_token = None
    if isinstance(module, dict):
        per_token = module.get("per_token_hot_input_vectors")
    if not isinstance(per_token, list):
        per_token = row.get("per_token_hot_input_vectors")
    if isinstance(per_token, list):
        for item in per_token:
            if isinstance(item, dict) and int(item.get("token_index", -1)) == token_index:
                return item
    return vectors


def compare_hot_input_vectors(
    package_summary: dict[str, Any] | None,
    fullref_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    if package_summary is None and fullref_summary is None:
        return {"status": "missing_both"}
    if package_summary is None:
        return {"status": "missing_package"}
    if fullref_summary is None:
        return {"status": "missing_fullref"}

    package_token = package_summary.get("token_index")
    fullref_token = fullref_summary.get("token_index")
    result: dict[str, Any] = {
        "status": "ok" if package_token == fullref_token else "token_mismatch",
        "package_token": package_token,
        "fullref_token": fullref_token,
        "feature_count_delta": diff(
            package_summary.get("feature_count"),
            fullref_summary.get("feature_count"),
        ),
    }

    package_stats = package_summary.get("stats")
    fullref_stats = fullref_summary.get("stats")
    if isinstance(package_stats, dict) and isinstance(fullref_stats, dict):
        result["stats_error"] = {
            key: diff(package_stats.get(key), fullref_stats.get(key))
            for key in (
                "mean",
                "abs_mean",
                "variance",
                "stddev",
                "rms",
                "l2_norm",
                "min",
                "max",
                "max_abs",
            )
            if key in package_stats and key in fullref_stats
        }

    top_feature_comparison = compare_feature_items(package_summary, fullref_summary, "top_abs_features")
    if top_feature_comparison:
        result["top_feature_comparison"] = top_feature_comparison[:8]
    sampled_feature_comparison = compare_feature_items(package_summary, fullref_summary, "sampled_features")
    if sampled_feature_comparison:
        result["sampled_feature_comparison"] = sampled_feature_comparison[:16]
    return result


def compare_feature_items(
    package_summary: dict[str, Any],
    fullref_summary: dict[str, Any],
    field_name: str,
) -> list[dict[str, Any]]:
    package_items = package_summary.get(field_name)
    fullref_items = fullref_summary.get(field_name)
    if not isinstance(package_items, list) or not isinstance(fullref_items, list):
        return []

    package_map = {
        item.get("feature_index"): item
        for item in package_items
        if isinstance(item, dict)
        and isinstance(item.get("feature_index"), int)
        and isinstance(item.get("value"), (int, float))
        and isinstance(item.get("abs_value"), (int, float))
    }
    fullref_map = {
        item.get("feature_index"): item
        for item in fullref_items
        if isinstance(item, dict)
        and isinstance(item.get("feature_index"), int)
        and isinstance(item.get("value"), (int, float))
        and isinstance(item.get("abs_value"), (int, float))
    }
    common = [
        int(index)
        for index in set(package_map.keys()) & set(fullref_map.keys())
        if isinstance(index, int)
    ]
    top_feature_comparison = []
    for index in common:
        package_item = package_map[index]
        fullref_item = fullref_map[index]
        value_diff = diff(package_item.get("value"), fullref_item.get("value"))
        abs_value_diff = diff(package_item.get("abs_value"), fullref_item.get("abs_value"))
        top_feature_comparison.append(
            {
                "feature_index": index,
                "package_value": package_item.get("value"),
                "fullref_value": fullref_item.get("value"),
                "package_abs_value": package_item.get("abs_value"),
                "fullref_abs_value": fullref_item.get("abs_value"),
                "value_diff": value_diff,
                "abs_value_diff": abs_value_diff,
            }
        )
        package_group_stats = package_item.get("group_stats")
        fullref_group_stats = fullref_item.get("group_stats")
        if isinstance(package_group_stats, dict) and isinstance(fullref_group_stats, dict):
            top_feature_comparison[-1]["package_group_stats"] = package_group_stats
            top_feature_comparison[-1]["fullref_group_stats"] = fullref_group_stats
            top_feature_comparison[-1]["group_stats_error"] = {
                key: diff(package_group_stats.get(key), fullref_group_stats.get(key))
                for key in (
                    "mean",
                    "abs_mean",
                    "rms",
                    "min",
                    "max",
                    "max_abs",
                )
                if key in package_group_stats and key in fullref_group_stats
            }
            for key in ("group_index", "group_offset", "group_width"):
                if key in package_item and key in fullref_item:
                    top_feature_comparison[-1][key] = package_item.get(key)

    top_feature_comparison.sort(
        key=lambda item: float(abs(float_value(item.get("abs_value_diff")) or 0.0)),
        reverse=True,
    )
    return top_feature_comparison


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
        package_hidden = int(package_trace.get("hidden_index"))
        fullref_hidden = int(fullref.get("hidden_index"))
        if package_hidden != fullref_hidden:
            skipped.append(
                {
                    "layer_index": layer,
                    "package_hidden_index": package_hidden,
                    "fullref_hidden_index": fullref_hidden,
                    "reason": "hidden_mismatch",
                }
            )
            continue
        full_trace = trace_by_token(fullref, token)
        if full_trace is None:
            skipped.append({"layer_index": layer, "token_index": token, "reason": "missing_full_reference_token"})
            continue
        comparison = {
            "layer_index": layer,
            "layer_kind": row.get("layer_kind"),
            "run_mode": row.get("run_mode"),
            "token_index": token,
            "hidden_index": package_hidden,
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

        package_hot = get_hot_input_vectors_for_token(row, token)
        fullref_hot = get_hot_input_vectors_for_token(fullref, token)
        comparison["attention_projection_input_hot"] = compare_hot_input_vectors(
            package_hot.get("attention_projection_input"),
            fullref_hot.get("attention_projection_input"),
        )
        comparison["mlp_activation_hot"] = compare_hot_input_vectors(
            package_hot.get("mlp_activation"),
            fullref_hot.get("mlp_activation"),
        )
        common_hot_names = sorted(set(package_hot.keys()) & set(fullref_hot.keys()))
        comparison["hot_input_vector_stage_errors"] = {
            name: compare_hot_input_vectors(package_hot.get(name), fullref_hot.get(name))
            for name in common_hot_names
            if name != "token_index"
        }

        attention_row_dot = row_dot_by_token(fullref, "attention_out_proj", token)
        if attention_row_dot is None:
            attention_row_dot = row_dot_by_token(fullref, "self_attention_o_proj", token)
        if attention_row_dot is not None:
            comparison.update(
                {
                    "fullref_package_attention_row_dot": attention_row_dot.get("package_row_dot"),
                    "attention_row_only_error": attention_row_dot.get("package_row_dot_error_vs_source_row"),
                    "attention_activation_path_error": diff(
                        package_trace.get("attention_output"),
                        attention_row_dot.get("package_row_dot"),
                    ),
                }
            )
        mlp_row_dot = row_dot_by_token(fullref, "mlp_down_proj", token)
        if mlp_row_dot is not None:
            comparison.update(
                {
                    "fullref_package_mlp_row_dot": mlp_row_dot.get("package_row_dot"),
                    "mlp_row_only_error": mlp_row_dot.get("package_row_dot_error_vs_source_row"),
                    "mlp_activation_path_error": diff(
                        package_trace.get("mlp_output"),
                        mlp_row_dot.get("package_row_dot"),
                    ),
                }
            )
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


def markdown(rows: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> str:
    def feature_label(value: Any, field_name: str) -> str:
        if not isinstance(value, dict):
            return "-"
        top = value.get(field_name)
        if not isinstance(top, list) or not top:
            return "-"
        first = top[0]
        if not isinstance(first, dict):
            return "-"
        index = first.get("feature_index")
        diff = first.get("value_diff")
        if index is None:
            return "-"
        if diff is None:
            return str(int(index))
        return f"{int(index)}:{fmt(diff)}"

    def top_feature_label(value: Any) -> str:
        return feature_label(value, "top_feature_comparison")

    def sampled_feature_label(value: Any) -> str:
        return feature_label(value, "sampled_feature_comparison")

    lines = [
        "| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | attn_row_only | attn_activation | mlp_row_only | mlp_activation | attn_hot_status | attn_hot_abs_mean_err | attn_hot_top1 | mlp_hot_status | mlp_hot_abs_mean_err | mlp_hot_top1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        attn_hot = row.get("attention_projection_input_hot", {})
        mlp_hot = row.get("mlp_activation_hot", {})
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
                    fmt(row.get("attention_row_only_error")),
                    fmt(row.get("attention_activation_path_error")),
                    fmt(row.get("mlp_row_only_error")),
                    fmt(row.get("mlp_activation_path_error")),
                    str(attn_hot.get("status", "-")),
                    fmt((attn_hot.get("stats_error") or {}).get("abs_mean")),
                    top_feature_label(attn_hot),
                    str(mlp_hot.get("status", "-")),
                    fmt((mlp_hot.get("stats_error") or {}).get("abs_mean")),
                    top_feature_label(mlp_hot),
                ]
            )
            + " |"
        )
    stage_rows: list[str] = []
    for row in rows:
        stage_errors = row.get("hot_input_vector_stage_errors")
        if not isinstance(stage_errors, dict):
            continue
        for stage, comparison in sorted(stage_errors.items()):
            if not isinstance(comparison, dict):
                continue
            stats_error = comparison.get("stats_error") or {}
            stage_rows.append(
                "| "
                + " | ".join(
                    [
                        str(row["layer_index"]),
                        str(row["token_index"]),
                        str(row["hidden_index"]),
                        str(stage),
                        str(comparison.get("status", "-")),
                        fmt(stats_error.get("abs_mean")),
                        fmt(stats_error.get("rms")),
                        fmt(stats_error.get("max_abs")),
                        top_feature_label(comparison),
                        sampled_feature_label(comparison),
                    ]
                )
                + " |"
            )
    if stage_rows:
        lines.extend(
            [
                "",
                "## Hot Input Vector Stage Errors",
                "",
                "| layer | token | hidden | stage | status | abs_mean_err | rms_err | max_abs_err | top1 | sampled_top1 |",
                "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
                *stage_rows,
            ]
        )
    if skipped:
        lines.extend(
            [
                "",
                "## Skipped",
                "",
                "| layer | reason | package_hidden | fullref_hidden | token |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for item in skipped:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("layer_index", "-")),
                        str(item.get("reason", "-")),
                        str(item.get("package_hidden_index", "-")),
                        str(item.get("fullref_hidden_index", "-")),
                        str(item.get("token_index", "-")),
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
        args.markdown.write_text(markdown(summary["rows"], summary["skipped"]), encoding="utf-8")
    print(
        "qwen-module-trace-comparison "
        f"rows={summary['row_count']} skipped={len(summary['skipped'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
