#!/usr/bin/env python3
"""Generate smoke row-scale override grids for Qwen prefix validation."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "qwen-row-scale-grid-v0.1"
SMOKE_SCHEMA_VERSION = "package-row-scale-overrides-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-index", type=int, required=True)
    parser.add_argument("--tensor-suffix", required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--scale", action="append", default=[], metavar="START:END:COUNT")
    parser.add_argument("--scale-values", action="append", default=[], metavar="CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-prefix", default="row-scale")
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--conditions-txt", type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def finite_positive(value: float, label: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(f"{label} must be a finite positive number, got {value!r}")
    return value


def parse_scale_range(spec: str) -> list[float]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise SystemExit(f"--scale must be START:END:COUNT, got {spec!r}")
    try:
        start = finite_positive(float(parts[0]), "--scale START")
        end = finite_positive(float(parts[1]), "--scale END")
        count = int(parts[2])
    except ValueError as err:
        raise SystemExit(f"failed to parse --scale {spec!r}: {err}") from err
    if count < 1:
        raise SystemExit(f"--scale COUNT must be positive, got {count}")
    if count == 1:
        if start != end:
            raise SystemExit("--scale COUNT=1 requires START and END to match")
        return [start]
    step = (end - start) / float(count - 1)
    return [start + step * index for index in range(count)]


def parse_scale_values(spec: str) -> list[float]:
    values = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(finite_positive(float(part), "--scale-values"))
        except ValueError as err:
            raise SystemExit(f"failed to parse --scale-values item {part!r}: {err}") from err
    if not values:
        raise SystemExit(f"--scale-values must contain at least one value, got {spec!r}")
    return values


def scale_values(args: argparse.Namespace) -> list[float]:
    values: list[float] = []
    for spec in args.scale:
        values.extend(parse_scale_range(spec))
    for spec in args.scale_values:
        values.extend(parse_scale_values(spec))
    if not values:
        raise SystemExit("at least one --scale or --scale-values is required")
    unique: dict[str, float] = {}
    for value in values:
        key = f"{value:.12g}"
        unique.setdefault(key, value)
    return [unique[key] for key in sorted(unique, key=lambda item: unique[item])]


def safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unnamed"


def scale_label(scale: float) -> str:
    value = f"{scale:.9f}".rstrip("0").rstrip(".")
    return value.replace(".", "p")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def smoke_payload(layer_index: int, tensor_suffix: str, row_index: int, scale: float) -> dict[str, Any]:
    return {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "overrides": [
            {
                "layer_index": layer_index,
                "tensor_suffix": tensor_suffix,
                "row_index": row_index,
                "scale": scale,
            }
        ],
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen Row-Scale Grid",
        "",
        f"- schema: `{summary['schema_version']}`",
        f"- layer index: `{summary['layer_index']}`",
        f"- tensor suffix: `{summary['tensor_suffix']}`",
        f"- row index: `{summary['row_index']}`",
        f"- condition count: `{len(summary['conditions'])}`",
        "",
        "| condition | scale | row_scale_json |",
        "| --- | ---: | --- |",
    ]
    for condition in summary["conditions"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(condition["condition"]),
                    f"{float(condition['scale']):.12g}",
                    str(condition["row_scale_json"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    if args.layer_index < 0:
        raise SystemExit("--layer-index must be non-negative")
    if args.row_index < 0:
        raise SystemExit("--row-index must be non-negative")
    tensor_suffix = args.tensor_suffix.strip()
    if not tensor_suffix:
        raise SystemExit("--tensor-suffix must not be empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_prefix = safe_label(args.label_prefix)
    conditions = []
    seen_labels: set[str] = set()
    for scale in scale_values(args):
        label = f"{label_prefix}-s{scale_label(scale)}"
        if label in seen_labels:
            raise SystemExit(f"duplicate condition label generated: {label}")
        seen_labels.add(label)
        path = args.output_dir / f"{label}.json"
        write_json(path, smoke_payload(args.layer_index, tensor_suffix, args.row_index, scale))
        conditions.append(
            {
                "condition": label,
                "scale": scale,
                "row_scale_json": str(path),
                "runner_condition": f"{label},row_scale={path}",
            }
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "layer_index": args.layer_index,
        "tensor_suffix": tensor_suffix,
        "row_index": args.row_index,
        "output_dir": str(args.output_dir),
        "conditions": conditions,
    }
    if args.summary_json:
        write_json(args.summary_json, summary)
    if args.conditions_txt:
        args.conditions_txt.parent.mkdir(parents=True, exist_ok=True)
        args.conditions_txt.write_text(
            "".join(f"--condition {condition['runner_condition']}\n" for condition in conditions),
            encoding="utf-8",
        )
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(summary), encoding="utf-8")
    print(f"qwen-row-scale-grid conditions={len(conditions)} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
