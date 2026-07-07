#!/usr/bin/env python3
"""Create a package copy with selected tensor name prefixes removed."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a hardlink package copy with selected tensor prefixes stripped."
    )
    parser.add_argument("--input-package", required=True, type=Path)
    parser.add_argument("--output-package", required=True, type=Path)
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        help="Tensor name prefix to remove. May be repeated.",
    )
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(package: Path) -> dict[str, Any]:
    manifest_path = package / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tensor_name(entry: dict[str, Any]) -> str:
    value = entry.get("name")
    return value if isinstance(value, str) else ""


def should_strip(entry: dict[str, Any], prefixes: list[str]) -> bool:
    name = tensor_name(entry)
    return any(name.startswith(prefix) for prefix in prefixes)


def referenced_files(manifest: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    refs.add("manifest.json")
    for entry in manifest.get("tensors", []):
        for key in ("index_file", "scale_file", "codebook_file"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                refs.add(value)
    for entry in manifest.get("passthrough_tensors", []):
        value = entry.get("payload_file")
        if isinstance(value, str) and value:
            refs.add(value)
    for entry in manifest.get("codebooks", []):
        value = entry.get("file")
        if isinstance(value, str) and value:
            refs.add(value)
    return refs


def file_size(package: Path, relative: str) -> int:
    path = package / relative
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def hardlink_referenced_files(input_package: Path, output_package: Path, refs: set[str]) -> None:
    for relative in sorted(refs):
        if relative == "manifest.json":
            continue
        src = input_package / relative
        dst = output_package / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)


def main() -> int:
    args = parse_args()
    if not args.strip_prefix:
        raise SystemExit("at least one --strip-prefix is required")
    input_package = args.input_package
    output_package = args.output_package
    manifest = read_manifest(input_package)

    original_tensors = list(manifest.get("tensors", []))
    original_passthrough = list(manifest.get("passthrough_tensors", []))
    kept_tensors = [entry for entry in original_tensors if not should_strip(entry, args.strip_prefix)]
    kept_passthrough = [
        entry for entry in original_passthrough if not should_strip(entry, args.strip_prefix)
    ]
    stripped_tensors = [entry for entry in original_tensors if should_strip(entry, args.strip_prefix)]
    stripped_passthrough = [
        entry for entry in original_passthrough if should_strip(entry, args.strip_prefix)
    ]

    output_manifest = dict(manifest)
    output_manifest["tensors"] = kept_tensors
    output_manifest["passthrough_tensors"] = kept_passthrough

    original_refs = referenced_files(manifest)
    output_refs = referenced_files(output_manifest)
    removed_refs = original_refs - output_refs
    summary = {
        "input_package": str(input_package),
        "output_package": str(output_package),
        "strip_prefixes": args.strip_prefix,
        "dry_run": args.dry_run,
        "original_quantized_tensors": len(original_tensors),
        "output_quantized_tensors": len(kept_tensors),
        "stripped_quantized_tensors": len(stripped_tensors),
        "original_passthrough_tensors": len(original_passthrough),
        "output_passthrough_tensors": len(kept_passthrough),
        "stripped_passthrough_tensors": len(stripped_passthrough),
        "removed_referenced_files": sorted(removed_refs),
        "removed_referenced_file_bytes": sum(file_size(input_package, ref) for ref in removed_refs),
        "output_referenced_files": len(output_refs),
        "output_referenced_file_bytes": sum(file_size(input_package, ref) for ref in output_refs),
    }

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if output_package.exists():
        if not args.overwrite:
            raise SystemExit(f"output package already exists: {output_package}")
        shutil.rmtree(output_package)
    output_package.mkdir(parents=True)
    hardlink_referenced_files(input_package, output_package, output_refs)
    (output_package / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
