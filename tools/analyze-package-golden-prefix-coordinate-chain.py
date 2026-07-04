#!/usr/bin/env python3
"""Extract one token/hidden coordinate chain from package golden prefix JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "package-golden-prefix-coordinate-chain-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--token-index", type=int, required=True)
    parser.add_argument("--hidden-index", type=int, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_coordinate_trace(row: dict[str, Any], token_index: int, hidden_index: int) -> dict[str, Any] | None:
    module_contribution = row.get("module_contribution")
    if not isinstance(module_contribution, dict):
        return None
    traces = module_contribution.get("per_token_hot_hidden_trace")
    if not isinstance(traces, list):
        return None
    for item in traces:
        if not isinstance(item, dict):
            continue
        if item.get("token_index") == token_index and item.get("hidden_index") == hidden_index:
            return item
    return None


def max_location(row: dict[str, Any], key: str) -> dict[str, Any] | None:
    distribution = row.get(key)
    if isinstance(distribution, dict):
        location = distribution.get("max_abs_diff_location")
        if isinstance(location, dict):
            return location
    return None


def build_summary(path: Path, token_index: int, hidden_index: int) -> dict[str, Any]:
    source_rows = read_jsonl(path)
    rows = []
    for row in source_rows:
        trace = find_coordinate_trace(row, token_index, hidden_index)
        entry: dict[str, Any] = {
            "layer_index": row.get("layer_index"),
            "layer_kind": row.get("layer_kind"),
            "run_mode": row.get("run_mode"),
            "input_max_abs_diff": row.get("input_max_abs_diff"),
            "output_max_abs_diff": row.get("max_abs_diff"),
            "input_max_abs_diff_location": max_location(row, "input_distribution"),
            "output_max_abs_diff_location": max_location(row, "output_distribution"),
            "coordinate_trace_available": trace is not None,
        }
        if trace is not None:
            for key in [
                "actual_input",
                "expected_input",
                "input_diff",
                "attention_output",
                "attention_block_output",
                "post_normed",
                "mlp_output",
                "actual_delta",
                "expected_delta",
                "delta_diff",
                "actual_output",
                "expected_output",
                "output_diff",
                "abs_output_diff",
            ]:
                if key in trace:
                    entry[key] = trace[key]
        rows.append(entry)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_path": str(path),
        "token_index": token_index,
        "hidden_index": hidden_index,
        "rows": rows,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.9g}"
    if value is None:
        return "-"
    return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "| layer | kind | input_diff | delta_diff | output_diff | attention | mlp | coord? | output_max_abs | output_max_location |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in payload["rows"]:
        loc = row.get("output_max_abs_diff_location")
        if isinstance(loc, dict):
            loc_text = "token {}, hidden {}".format(loc.get("token_index"), loc.get("hidden_index"))
        else:
            loc_text = "-"
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row.get("layer_index", "-"),
                row.get("layer_kind", "-"),
                fmt(row.get("input_diff")),
                fmt(row.get("delta_diff")),
                fmt(row.get("output_diff")),
                fmt(row.get("attention_output")),
                fmt(row.get("mlp_output")),
                "yes" if row.get("coordinate_trace_available") else "no",
                fmt(row.get("output_max_abs_diff")),
                loc_text,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    summary = build_summary(args.jsonl, args.token_index, args.hidden_index)
    write_json(args.summary_json, summary)
    if args.markdown is not None:
        write_markdown(args.markdown, summary)
    print(f"package-golden-prefix-coordinate-chain rows={len(summary['rows'])} output={args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
