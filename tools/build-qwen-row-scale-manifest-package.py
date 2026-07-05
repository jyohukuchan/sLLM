#!/usr/bin/env python3
"""Build a hardlink package copy with Qwen row-scale manifest overrides."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "qwen-row-scale-manifest-package-build-v0.1"
ROW_SCALE_SCHEMA_VERSION = "row-scale-overrides-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--row-scale-overrides-json", type=Path, required=True)
    parser.add_argument("--output-package", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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


def tensor_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        raise SystemExit("source package manifest must contain a tensors list")
    result: dict[str, dict[str, Any]] = {}
    for tensor in tensors:
        if not isinstance(tensor, dict):
            raise SystemExit("source package manifest tensors must be objects")
        name = tensor.get("name")
        if not isinstance(name, str) or not name:
            raise SystemExit("source package manifest tensor name must be a non-empty string")
        if name in result:
            raise SystemExit(f"duplicate tensor in source package manifest: {name}")
        result[name] = tensor
    return result


def validated_row_scale_overrides(
    overrides: dict[str, Any], tensors_by_name: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    schema_version = overrides.get("schema_version")
    if schema_version != ROW_SCALE_SCHEMA_VERSION:
        raise SystemExit(
            f"row_scale_overrides schema_version must be {ROW_SCALE_SCHEMA_VERSION}, got {schema_version!r}"
        )
    entries = overrides.get("entries")
    if not isinstance(entries, list):
        raise SystemExit("row_scale_overrides entries must be a list")

    seen: set[tuple[str, int]] = set()
    validated_entries: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SystemExit(f"row_scale_overrides entry {index} must be an object")
        tensor_name = entry.get("tensor_name")
        if not isinstance(tensor_name, str) or not tensor_name:
            raise SystemExit(f"row_scale_overrides entry {index} tensor_name must be non-empty")
        tensor = tensors_by_name.get(tensor_name)
        if tensor is None:
            raise SystemExit(
                f"row_scale_overrides entry {index} references tensor not in source package: {tensor_name}"
            )
        shape = tensor.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise SystemExit(f"row_scale_overrides tensor {tensor_name} must be 2D, got shape {shape!r}")
        rows = shape[0]
        if not isinstance(rows, int) or rows <= 0:
            raise SystemExit(f"row_scale_overrides tensor {tensor_name} has invalid row count {rows!r}")
        row_index = entry.get("row_index")
        if not isinstance(row_index, int) or row_index < 0:
            raise SystemExit(f"row_scale_overrides entry {index} row_index must be non-negative")
        if row_index >= rows:
            raise SystemExit(
                f"row_scale_overrides row out of range for {tensor_name}: row={row_index} rows={rows}"
            )
        scale = entry.get("scale")
        if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or float(scale) <= 0.0:
            raise SystemExit(
                f"row_scale_overrides entry for {tensor_name} row {row_index} "
                f"must have a finite positive scale"
            )
        key = (tensor_name, row_index)
        if key in seen:
            raise SystemExit(f"duplicate row_scale_overrides entry for {tensor_name} row {row_index}")
        seen.add(key)

        validated = {
            "tensor_name": tensor_name,
            "row_index": row_index,
            "scale": float(scale),
        }
        source = entry.get("source")
        if source is not None:
            if not isinstance(source, str):
                raise SystemExit(f"row_scale_overrides entry {index} source must be a string when present")
            validated["source"] = source
        validated_entries.append(validated)

    return {
        "schema_version": ROW_SCALE_SCHEMA_VERSION,
        "entries": validated_entries,
    }


def ensure_build_paths(source_package: Path, output_package: Path, overwrite: bool) -> None:
    if not source_package.is_dir():
        raise SystemExit(f"source package must be an existing directory: {source_package}")
    source_resolved = source_package.resolve()
    output_resolved = output_package.resolve()
    if source_resolved == output_resolved:
        raise SystemExit("--output-package must differ from --source-package")
    if source_resolved in output_resolved.parents:
        raise SystemExit("--output-package must not be inside --source-package")
    if output_package.exists():
        if not overwrite:
            raise SystemExit(f"output package already exists: {output_package}")
        if output_package.is_dir():
            shutil.rmtree(output_package)
        else:
            output_package.unlink()
    output_package.parent.mkdir(parents=True, exist_ok=True)


def build_package(source_package: Path, output_package: Path, manifest: dict[str, Any]) -> None:
    shutil.copytree(source_package, output_package, copy_function=os.link)
    output_manifest_path = output_package / "manifest.json"
    # copytree hardlinks every file, so break the manifest link before replacing metadata.
    output_manifest_path.unlink()
    write_json(output_manifest_path, manifest)


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen Row-Scale Manifest Package Build",
        "",
        f"- schema: `{summary['schema_version']}`",
        f"- source package: `{summary['source_package']}`",
        f"- output package: `{summary['output_package']}`",
        f"- dry run: `{summary['dry_run']}`",
        f"- row scale entries: `{summary['row_scale_entry_count']}`",
        "",
        "| tensor | row | scale | source |",
        "| --- | ---: | ---: | --- |",
    ]
    for entry in summary["row_scale_overrides"]["entries"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(entry["tensor_name"]),
                    str(entry["row_index"]),
                    f"{float(entry['scale']):.12g}",
                    str(entry.get("source", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    source_package = args.source_package
    output_package = args.output_package
    source_manifest_path = source_package / "manifest.json"
    source_manifest = read_json(source_manifest_path)
    overrides = read_json(args.row_scale_overrides_json)
    row_scale_overrides = validated_row_scale_overrides(overrides, tensor_map(source_manifest))

    output_manifest = dict(source_manifest)
    output_manifest["row_scale_overrides"] = row_scale_overrides

    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_package": str(source_package),
        "row_scale_overrides_json": str(args.row_scale_overrides_json),
        "output_package": str(output_package),
        "dry_run": bool(args.dry_run),
        "row_scale_entry_count": len(row_scale_overrides["entries"]),
        "row_scale_overrides": row_scale_overrides,
    }

    if not args.dry_run:
        ensure_build_paths(source_package, output_package, args.overwrite)
        build_package(source_package, output_package, output_manifest)

    if args.summary_json:
        write_json(args.summary_json, summary)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(summary), encoding="utf-8")

    action = "validated" if args.dry_run else "built"
    print(
        f"qwen-row-scale-manifest-package {action}=true "
        f"entries={len(row_scale_overrides['entries'])} output={output_package}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
