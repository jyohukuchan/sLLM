#!/usr/bin/env python3
"""Generate manifest row-scale override grids for Qwen package validation."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "qwen-manifest-row-scale-grid-v0.1"
ROW_SCALE_SCHEMA_VERSION = "row-scale-overrides-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-row-scale-json", type=Path, required=True)
    parser.add_argument("--target-tensor-name", required=True)
    parser.add_argument("--target-row-index", type=int, required=True)
    parser.add_argument("--target-source-prefix", default="qwen-manifest-row-scale-grid")
    parser.add_argument("--scale", action="append", default=[], metavar="START:END:COUNT")
    parser.add_argument("--scale-values", action="append", default=[], metavar="CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-prefix", default="manifest-row-scale")
    parser.add_argument("--package-output-dir", type=Path)
    parser.add_argument("--package-name-prefix", default="")
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--conditions-txt", type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as err:
        raise SystemExit(f"failed to read JSON {path}: {err}") from err
    except json.JSONDecodeError as err:
        raise SystemExit(f"failed to parse JSON {path}: {err}") from err
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def base_entries(base: dict[str, Any]) -> list[dict[str, Any]]:
    schema_version = base.get("schema_version")
    if schema_version != ROW_SCALE_SCHEMA_VERSION:
        raise SystemExit(
            f"base row-scale schema_version must be {ROW_SCALE_SCHEMA_VERSION}, got {schema_version!r}"
        )
    entries = base.get("entries")
    if not isinstance(entries, list):
        raise SystemExit("base row-scale entries must be a list")
    copied: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SystemExit(f"base row-scale entry {index} must be an object")
        tensor_name = entry.get("tensor_name")
        row_index = entry.get("row_index")
        scale = entry.get("scale")
        if not isinstance(tensor_name, str) or not tensor_name:
            raise SystemExit(f"base row-scale entry {index} tensor_name must be non-empty")
        if not isinstance(row_index, int) or row_index < 0:
            raise SystemExit(f"base row-scale entry {index} row_index must be non-negative")
        if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or float(scale) <= 0.0:
            raise SystemExit(f"base row-scale entry {index} scale must be finite and positive")
        key = (tensor_name, row_index)
        if key in seen:
            raise SystemExit(f"duplicate base row-scale entry for {tensor_name} row {row_index}")
        seen.add(key)
        copied_entry = {
            "tensor_name": tensor_name,
            "row_index": row_index,
            "scale": float(scale),
        }
        source = entry.get("source")
        if source is not None:
            if not isinstance(source, str):
                raise SystemExit(f"base row-scale entry {index} source must be a string when present")
            copied_entry["source"] = source
        copied.append(copied_entry)
    return copied


def payload_for_scale(
    entries: list[dict[str, Any]],
    tensor_name: str,
    row_index: int,
    scale: float,
    source_prefix: str,
) -> dict[str, Any]:
    target = {
        "tensor_name": tensor_name,
        "row_index": row_index,
        "scale": scale,
        "source": f"{source_prefix}-s{scale_label(scale)}",
    }
    output_entries: list[dict[str, Any]] = []
    replaced = False
    for entry in entries:
        if entry["tensor_name"] == tensor_name and entry["row_index"] == row_index:
            output_entries.append(target)
            replaced = True
        else:
            output_entries.append(dict(entry))
    if not replaced:
        output_entries.append(target)
    return {
        "schema_version": ROW_SCALE_SCHEMA_VERSION,
        "entries": output_entries,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen Manifest Row-Scale Grid",
        "",
        f"- schema: `{summary['schema_version']}`",
        f"- base row-scale json: `{summary['base_row_scale_json']}`",
        f"- target tensor: `{summary['target_tensor_name']}`",
        f"- target row: `{summary['target_row_index']}`",
        f"- condition count: `{len(summary['conditions'])}`",
        "",
        "| condition | scale | row_scale_json | package_path |",
        "| --- | ---: | --- | --- |",
    ]
    for condition in summary["conditions"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(condition["condition"]),
                    f"{float(condition['scale']):.12g}",
                    str(condition["row_scale_json"]),
                    str(condition.get("package_path", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    if args.target_row_index < 0:
        raise SystemExit("--target-row-index must be non-negative")
    target_tensor_name = args.target_tensor_name.strip()
    if not target_tensor_name:
        raise SystemExit("--target-tensor-name must not be empty")
    source_prefix = args.target_source_prefix.strip()
    if not source_prefix:
        raise SystemExit("--target-source-prefix must not be empty")

    entries = base_entries(read_json(args.base_row_scale_json))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package_name_prefix = safe_label(args.package_name_prefix)
    label_prefix = safe_label(args.label_prefix)
    conditions = []
    seen_labels: set[str] = set()

    for scale in scale_values(args):
        label = f"{label_prefix}-s{scale_label(scale)}"
        if label in seen_labels:
            raise SystemExit(f"duplicate condition label generated: {label}")
        seen_labels.add(label)
        row_scale_path = args.output_dir / f"{label}.json"
        write_json(
            row_scale_path,
            payload_for_scale(entries, target_tensor_name, args.target_row_index, scale, source_prefix),
        )
        condition: dict[str, Any] = {
            "condition": label,
            "scale": scale,
            "row_scale_json": str(row_scale_path),
        }
        if args.package_output_dir:
            package_name = f"{package_name_prefix}-{label}" if package_name_prefix else label
            package_path = args.package_output_dir / f"{package_name}.ullm.d"
            condition["package_path"] = str(package_path)
            condition["runner_condition"] = f"{label},package={package_path}"
        conditions.append(condition)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "base_row_scale_json": str(args.base_row_scale_json),
        "target_tensor_name": target_tensor_name,
        "target_row_index": args.target_row_index,
        "target_source_prefix": source_prefix,
        "output_dir": str(args.output_dir),
        "package_output_dir": str(args.package_output_dir) if args.package_output_dir else None,
        "conditions": conditions,
    }
    if args.summary_json:
        write_json(args.summary_json, summary)
    if args.conditions_txt:
        args.conditions_txt.parent.mkdir(parents=True, exist_ok=True)
        args.conditions_txt.write_text(
            "".join(
                f"--condition {condition['runner_condition']}\n"
                for condition in conditions
                if "runner_condition" in condition
            ),
            encoding="utf-8",
        )
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(summary), encoding="utf-8")
    print(f"qwen-manifest-row-scale-grid conditions={len(conditions)} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
