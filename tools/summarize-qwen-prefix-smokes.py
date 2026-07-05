#!/usr/bin/env python3
"""Summarize qwen package-golden-prefix-smoke JSONL reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "qwen-prefix-smoke-summary-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Named package-golden-prefix-smoke JSONL report. May be repeated.",
    )
    parser.add_argument(
        "--matrix-summary-json",
        action="append",
        default=[],
        type=Path,
        help="package-golden-prefix smoke matrix summary JSON. May be repeated.",
    )
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def parse_report_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--report must be LABEL=PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"--report label must not be empty: {spec!r}")
    return label, Path(path)


def parse_matrix_summary(path: Path) -> list[tuple[str, Path]]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise ValueError(f"matrix summary must be a JSON object: {path}")
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"matrix summary has no runs array: {path}")
    reports = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"matrix summary run must be a JSON object: {path}:{index}")
        fixture_label = run.get("fixture_label")
        condition = run.get("condition")
        report_path = run.get("report_path")
        returncode = run.get("returncode")
        if returncode not in (None, 0):
            continue
        if not isinstance(fixture_label, str) or not isinstance(condition, str):
            raise ValueError(f"matrix run is missing fixture_label or condition: {path}:{index}")
        if not isinstance(report_path, str) or not report_path:
            raise ValueError(f"matrix run is missing report_path: {path}:{index}")
        reports.append((f"{fixture_label}-{condition}", Path(report_path)))
    return reports


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"report has no rows: {path}")
    return rows


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def layer_summary(row: dict[str, Any]) -> dict[str, Any]:
    output_distribution = row.get("output_distribution")
    max_location = {}
    if isinstance(output_distribution, dict):
        location = output_distribution.get("max_abs_diff_location")
        if isinstance(location, dict):
            max_location = location
    return {
        "layer_index": row.get("layer_index"),
        "layer_kind": row.get("layer_kind"),
        "max_abs_diff": number(row.get("max_abs_diff")),
        "mean_abs_diff": number(row.get("mean_abs_diff")),
        "mse": number(row.get("mse")),
        "cosine_similarity": number(row.get("cosine_similarity")),
        "max_token_index": max_location.get("token_index"),
        "max_hidden_index": max_location.get("hidden_index"),
        "max_diff": number(max_location.get("diff")),
    }


def summarize_report(label: str, path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    layers = [layer_summary(row) for row in rows]
    max_layer = max(layers, key=lambda item: abs(float(item.get("max_abs_diff") or 0.0)))
    max_mean_abs = max(
        (float(item["mean_abs_diff"]) for item in layers if item.get("mean_abs_diff") is not None),
        default=None,
    )
    max_mse = max(
        (float(item["mse"]) for item in layers if item.get("mse") is not None),
        default=None,
    )
    min_cosine = min(
        (float(item["cosine_similarity"]) for item in layers if item.get("cosine_similarity") is not None),
        default=None,
    )
    first = rows[0]
    return {
        "label": label,
        "path": str(path),
        "fixture": first.get("fixture"),
        "package": first.get("package"),
        "run_mode": first.get("run_mode"),
        "backend": first.get("backend"),
        "device_name": first.get("device_name"),
        "layer_start": first.get("layer_start"),
        "layer_end_exclusive": first.get("layer_end_exclusive"),
        "sequence_len": first.get("sequence_len"),
        "row_scale_override_source": first.get("row_scale_override_source"),
        "cell_delta_override_source": first.get("cell_delta_override_source"),
        "manifest_row_scale_override_count": first.get("manifest_row_scale_override_count"),
        "row_count": len(rows),
        "overall_max_abs_diff": max_layer.get("max_abs_diff"),
        "overall_max_layer": max_layer.get("layer_index"),
        "overall_max_token_index": max_layer.get("max_token_index"),
        "overall_max_hidden_index": max_layer.get("max_hidden_index"),
        "overall_max_diff": max_layer.get("max_diff"),
        "max_mean_abs_diff": max_mean_abs,
        "max_mse": max_mse,
        "min_cosine_similarity": min_cosine,
        "layers": layers,
    }


def fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.9g}"
    if value is None:
        return "-"
    return str(value)


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Qwen Prefix Smoke Summary",
        "",
        f"- schema: `{summary['schema_version']}`",
        f"- report count: `{len(summary['reports'])}`",
        "",
        "## Overall",
        "",
        "| label | max_abs | layer | token | hidden | max_diff | max_mean_abs | max_mse | min_cosine |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in summary["reports"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(report["label"]),
                    fmt(report.get("overall_max_abs_diff")),
                    fmt(report.get("overall_max_layer")),
                    fmt(report.get("overall_max_token_index")),
                    fmt(report.get("overall_max_hidden_index")),
                    fmt(report.get("overall_max_diff")),
                    fmt(report.get("max_mean_abs_diff")),
                    fmt(report.get("max_mse")),
                    fmt(report.get("min_cosine_similarity")),
                ]
            )
            + " |"
        )
    lines.append("")
    for report in summary["reports"]:
        lines.extend(
            [
                f"## {report['label']}",
                "",
                "| layer | max_abs | token | hidden | diff | mean_abs | mse | cosine |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for layer in report["layers"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        fmt(layer.get("layer_index")),
                        fmt(layer.get("max_abs_diff")),
                        fmt(layer.get("max_token_index")),
                        fmt(layer.get("max_hidden_index")),
                        fmt(layer.get("max_diff")),
                        fmt(layer.get("mean_abs_diff")),
                        fmt(layer.get("mse")),
                        fmt(layer.get("cosine_similarity")),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    report_specs = [parse_report_spec(spec) for spec in args.report]
    for matrix_summary_json in args.matrix_summary_json:
        report_specs.extend(parse_matrix_summary(matrix_summary_json))
    if not report_specs:
        raise SystemExit("at least one --report or --matrix-summary-json is required")
    labels = set()
    duplicate_labels = []
    for label, _ in report_specs:
        if label in labels:
            duplicate_labels.append(label)
        labels.add(label)
    if duplicate_labels:
        raise SystemExit(f"duplicate report labels: {', '.join(sorted(duplicate_labels))}")
    reports = [summarize_report(label, path) for label, path in report_specs]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "reports": reports,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
