#!/usr/bin/env python3
"""Analyze golden-prefix drift JSONL outputs from one or more runs.

The script is intentionally conservative about schema assumptions so old and new
JSONL formats can be consumed together. Missing `run_mode` is treated as
`actual_prefix`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "golden-prefix-drift-analysis-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize golden-prefix drift JSONL across runs."
    )
    parser.add_argument(
        "jsonl_paths",
        nargs="+",
        type=Path,
        help="One or more JSONL files to analyze.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=False,
        help="Write summary JSON to this path.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        required=False,
        help="Write Markdown table to this path.",
    )
    return parser.parse_args()


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
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


def parse_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def normalize_run_mode(raw: Any) -> str:
    value = parse_str(raw)
    if not value:
        return "actual_prefix"
    return value


def range_label(start: int | None, end: int | None) -> str:
    if start is None and end is None:
        return "?"
    if start is None:
        return f"?..{end}"
    if end is None:
        return f"{start}..?"
    return f"{start}..{end}"


def device_backend_label(device_index: int | None, backend: str) -> str:
    if device_index is None:
        return backend
    return f"{device_index}/{backend}"


@dataclass(frozen=True)
class GroupKey:
    device_index: int | None
    backend: str
    run_mode: str
    layer_start: int | None
    layer_end_exclusive: int | None
    layer_index: int | None


@dataclass
class DriftRow:
    key: GroupKey
    source: str
    input_mse: float | None
    input_mean_abs_diff: float | None
    input_max_abs_diff: float | None
    input_cosine_similarity: float | None
    input_failure_class: str | None
    output_mse: float | None
    mean_abs_diff: float | None
    max_abs_diff: float | None
    cosine_similarity: float | None
    failure_class: str | None


def safe_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"JSONL path is not a file: {path}")


def read_jsonl(path: Path) -> list[DriftRow]:
    rows: list[DriftRow] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} invalid JSON: {exc}") from exc

            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number} row is not an object")

            key = GroupKey(
                device_index=parse_int(item.get("device_index")),
                backend=parse_str(item.get("backend")) or "unknown",
                run_mode=normalize_run_mode(item.get("run_mode")),
                layer_start=parse_int(item.get("layer_start")),
                layer_end_exclusive=parse_int(item.get("layer_end_exclusive")),
                layer_index=parse_int(item.get("layer_index")),
            )
            rows.append(
                DriftRow(
                    key=key,
                    source=str(path),
                    input_mse=parse_float(item.get("input_mse")),
                    input_mean_abs_diff=parse_float(item.get("input_mean_abs_diff")),
                    input_max_abs_diff=parse_float(item.get("input_max_abs_diff")),
                    input_cosine_similarity=parse_float(item.get("input_cosine_similarity")),
                    input_failure_class=parse_str(item.get("input_failure_class")),
                    output_mse=parse_float(item.get("mse")),
                    mean_abs_diff=parse_float(item.get("mean_abs_diff")),
                    max_abs_diff=parse_float(item.get("max_abs_diff")),
                    cosine_similarity=parse_float(item.get("cosine_similarity")),
                    failure_class=parse_str(item.get("failure_class")),
                )
            )
    return rows


def row_importance(row: DriftRow) -> int:
    score = 0
    for value in (
        row.output_mse,
        row.max_abs_diff,
        row.cosine_similarity,
        row.input_mse,
        row.input_max_abs_diff,
        row.input_cosine_similarity,
        row.failure_class,
        row.input_failure_class,
        row.key.layer_index,
        row.key.layer_start,
        row.key.layer_end_exclusive,
        row.key.device_index,
        row.key.backend,
        row.key.run_mode,
    ):
        if value is not None:
            score += 1
    return score


def group_rows(rows: list[DriftRow]) -> dict[GroupKey, DriftRow]:
    grouped: dict[GroupKey, DriftRow] = {}
    for row in rows:
        existing = grouped.get(row.key)
        if existing is None or row_importance(row) > row_importance(existing):
            grouped[row.key] = row
    return grouped


def fmt_num(value: float | None, width: int = 6) -> str:
    if value is None:
        return "-"
    return f"{value:.{width}f}"


def is_drift_failure(value: str | None) -> bool:
    if not value:
        return False
    return "drift" in value


def make_table_rows(grouped_rows: list[DriftRow]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in sorted(
        grouped_rows,
        key=lambda row: (
            row.key.layer_index if row.key.layer_index is not None else 1 << 30,
            (row.key.device_index if row.key.device_index is not None else 1 << 30),
            row.key.backend,
            row.key.run_mode,
            row.key.layer_start if row.key.layer_start is not None else -1,
            row.key.layer_end_exclusive if row.key.layer_end_exclusive is not None else -1,
        ),
    ):
        records.append(
            {
                "layer": row.key.layer_index,
                "device_backend": device_backend_label(
                    row.key.device_index,
                    row.key.backend,
                ),
                "run_mode": row.key.run_mode,
                "range": range_label(row.key.layer_start, row.key.layer_end_exclusive),
                "input_mse": row.input_mse,
                "output_mse": row.output_mse,
                "output_cosine": row.cosine_similarity,
                "failure_class": row.failure_class or "-",
            }
        )
    return records


def build_markdown_table(table_rows: list[dict[str, Any]]) -> str:
    lines = [
        "| layer | device/backend | run_mode | range | input_mse | output_mse | output_cosine | failure_class |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in table_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["layer"]) if row["layer"] is not None else "-",
                    str(row["device_backend"]),
                    row["run_mode"],
                    row["range"],
                    fmt_num(row["input_mse"]),
                    fmt_num(row["output_mse"]),
                    fmt_num(row["output_cosine"]),
                    row["failure_class"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def first_bad_layer(grouped_rows: list[DriftRow]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in grouped_rows
        if is_drift_failure(row.failure_class) and row.key.layer_index is not None
    ]
    if not candidates:
        return None
    row = sorted(candidates, key=lambda row: row.key.layer_index)[0]
    return {
        "layer_index": row.key.layer_index,
        "device_index": row.key.device_index,
        "backend": row.key.backend,
        "run_mode": row.key.run_mode,
        "range": range_label(row.key.layer_start, row.key.layer_end_exclusive),
        "failure_class": row.failure_class,
        "output_mse": row.output_mse,
    }


def largest_output_mse_layer(grouped_rows: list[DriftRow]) -> dict[str, Any] | None:
    candidates = [
        row for row in grouped_rows if row.output_mse is not None and row.key.layer_index is not None
    ]
    if not candidates:
        return None
    row = max(candidates, key=lambda row: row.output_mse)
    return {
        "layer_index": row.key.layer_index,
        "device_index": row.key.device_index,
        "backend": row.key.backend,
        "run_mode": row.key.run_mode,
        "range": range_label(row.key.layer_start, row.key.layer_end_exclusive),
        "output_mse": row.output_mse,
    }


def backend_deltas(grouped_rows: list[DriftRow]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int | None, int | None, int | None], dict[str, list[DriftRow]]] = {}
    for row in grouped_rows:
        key = (row.key.run_mode, row.key.layer_start, row.key.layer_end_exclusive, row.key.layer_index)
        backends = buckets.setdefault(key, {})
        device_backend = device_backend_label(row.key.device_index, row.key.backend or "unknown")
        backends.setdefault(device_backend, []).append(row)

    deltas: list[dict[str, Any]] = []
    for (run_mode, layer_start, layer_end_exclusive, layer_index), backend_rows in buckets.items():
        if len(backend_rows) < 2:
            continue

        # Prefer output_mse for backend delta, then fall back to max_abs_diff.
        backend_list = sorted(backend_rows.items(), key=lambda item: item[0])
        for i, (backend_a, rows_a) in enumerate(backend_list):
            for backend_b, rows_b in backend_list[i + 1 :]:
                for row_a in rows_a:
                    for row_b in rows_b:
                        metric_name = None
                        a_value = row_a.output_mse
                        b_value = row_b.output_mse
                        if a_value is not None and b_value is not None:
                            metric_name = "output_mse"
                        else:
                            a_value = row_a.max_abs_diff
                            b_value = row_b.max_abs_diff
                            if a_value is not None and b_value is not None:
                                metric_name = "max_abs_diff"
                        if metric_name is None:
                            continue
                        delta = a_value - b_value
                        deltas.append(
                            {
                                "run_mode": run_mode,
                                "layer_index": layer_index,
                                "range": range_label(layer_start, layer_end_exclusive),
                                "metric": metric_name,
                                "backend_a": backend_a,
                                "backend_b": backend_b,
                                "backend_a_value": a_value,
                                "backend_b_value": b_value,
                                "backend_delta": delta,
                                "backend_delta_abs": abs(delta),
                            }
                        )
    return deltas


def largest_backend_delta_layer(deltas: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not deltas:
        return None
    return max(deltas, key=lambda row: row["backend_delta_abs"])


def run_mode_deltas(grouped_rows: list[DriftRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[int | None, str | None, int | None], dict[str, list[DriftRow]]] = defaultdict(
        lambda: {"actual_prefix": [], "golden_before_each_layer": []}
    )

    for row in grouped_rows:
        if row.key.run_mode not in {"actual_prefix", "golden_before_each_layer"}:
            continue
        groups[(row.key.device_index, row.key.backend, row.key.layer_index)][row.key.run_mode].append(row)

    deltas: list[dict[str, Any]] = []
    for (device_index, backend, layer_index), rows_by_run_mode in groups.items():
        actual_rows = rows_by_run_mode["actual_prefix"]
        golden_rows = rows_by_run_mode["golden_before_each_layer"]
        if not actual_rows or not golden_rows:
            continue
        actual_unused = set(range(len(actual_rows)))
        golden_unused = set(range(len(golden_rows)))

        # Pair same-range rows first.
        for ai, actual in enumerate(actual_rows):
            if actual.output_mse is None:
                continue
            matched = None
            for gi, golden in enumerate(golden_rows):
                if (
                    gi in golden_unused
                    and actual.key.layer_start == golden.key.layer_start
                    and actual.key.layer_end_exclusive == golden.key.layer_end_exclusive
                    and golden.output_mse is not None
                ):
                    matched = gi
                    break
            if matched is None:
                continue
            golden = golden_rows[matched]
            deltas.append(
                {
                    "device_index": device_index,
                    "backend": backend,
                    "layer_index": layer_index,
                    "run_mode_a": "actual_prefix",
                    "run_mode_b": "golden_before_each_layer",
                    "range_a": range_label(
                        actual.key.layer_start,
                        actual.key.layer_end_exclusive,
                    ),
                    "range_b": range_label(
                        golden.key.layer_start,
                        golden.key.layer_end_exclusive,
                    ),
                    "range_match": True,
                    "output_mse_delta": actual.output_mse - golden.output_mse,
                }
            )
            actual_unused.discard(ai)
            golden_unused.discard(matched)

        # Fallback: pair leftover rows by layer/device/backend if range does not match.
        if actual_unused and golden_unused:
            ai = next(iter(actual_unused))
            gi = next(iter(golden_unused))
            actual = actual_rows[ai]
            golden = golden_rows[gi]
            if actual.output_mse is not None and golden.output_mse is not None:
                deltas.append(
                    {
                        "device_index": device_index,
                        "backend": backend,
                        "layer_index": layer_index,
                        "run_mode_a": "actual_prefix",
                        "run_mode_b": "golden_before_each_layer",
                        "range_a": range_label(
                            actual.key.layer_start,
                            actual.key.layer_end_exclusive,
                        ),
                        "range_b": range_label(
                            golden.key.layer_start,
                            golden.key.layer_end_exclusive,
                        ),
                        "range_match": False,
                        "output_mse_delta": actual.output_mse - golden.output_mse,
                    }
                )
    return deltas


def build_summary(
    grouped: dict[GroupKey, DriftRow], run_mode_path: list[Path], output_path: Path | None
) -> dict[str, Any]:
    rows = list(grouped.values())
    table_rows = make_table_rows(rows)
    markdown = build_markdown_table(table_rows)
    backend_delta_rows = backend_deltas(rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sources": [str(path) for path in run_mode_path],
        "summary": {
            "first_bad_layer": first_bad_layer(rows),
            "largest_output_mse_layer": largest_output_mse_layer(rows),
            "largest_backend_delta_layer": largest_backend_delta_layer(backend_delta_rows),
            "run_mode_output_mse_deltas": run_mode_deltas(rows),
            "backend_deltas": backend_delta_rows,
        },
        "table_rows": table_rows,
        "markdown": markdown,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    for path in args.jsonl_paths:
        safe_path(path)

    all_rows = []
    for path in args.jsonl_paths:
        all_rows.extend(read_jsonl(path))
    grouped = group_rows(all_rows)
    summary = build_summary(grouped, args.jsonl_paths, args.summary_json)

    markdown = summary["markdown"]
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown + "\n", encoding="utf-8")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
