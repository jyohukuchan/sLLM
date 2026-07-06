#!/usr/bin/env python3
"""Build a hardlink package copy with selected tensors replaced from overrides."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ullm-prototype-package-overlay-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--override-package", type=Path, action="append", required=True)
    parser.add_argument("--replace-tensor", action="append", required=True)
    parser.add_argument("--output-package", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def relative_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
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


def manifest_path(package: Path) -> Path:
    return package / "manifest.json"


def read_manifest(package: Path, label: str) -> dict[str, Any]:
    if not package.is_dir():
        raise SystemExit(f"{label} must be an existing directory: {package}")
    manifest = read_json(manifest_path(package))
    if manifest.get("schema_version") != "ullm-prototype-manifest-v0.1":
        raise SystemExit(f"{label} has unsupported schema_version: {manifest.get('schema_version')!r}")
    return manifest


def object_list(manifest: dict[str, Any], key: str, label: str) -> list[dict[str, Any]]:
    value = manifest.get(key, [])
    if not isinstance(value, list):
        raise SystemExit(f"{label} manifest field {key} must be a list")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise SystemExit(f"{label} manifest {key}[{index}] must be an object")
        result.append(item)
    return result


def entry_name(entry: dict[str, Any], label: str) -> str:
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise SystemExit(f"{label} manifest entry has invalid tensor name: {name!r}")
    return name


def tensor_entries_by_name(manifest: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key in ("tensors", "passthrough_tensors"):
        for entry in object_list(manifest, key, label):
            name = entry_name(entry, label)
            if name in result:
                raise SystemExit(f"{label} manifest has duplicate tensor entry: {name}")
            result[name] = entry
    return result


def codebook_key(entry: dict[str, Any], label: str) -> tuple[str, str]:
    family = entry.get("family")
    candidate_id = entry.get("candidate_id")
    if not isinstance(family, str) or not family:
        raise SystemExit(f"{label} codebook entry has invalid family: {family!r}")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise SystemExit(f"{label} codebook entry has invalid candidate_id: {candidate_id!r}")
    return family, candidate_id


def codebooks_by_key(manifest: dict[str, Any], label: str) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in object_list(manifest, "codebooks", label):
        key = codebook_key(entry, label)
        if key in result:
            raise SystemExit(f"{label} manifest has duplicate codebook entry: {key}")
        file_name = entry.get("file")
        if not isinstance(file_name, str) or not file_name:
            raise SystemExit(f"{label} codebook {key} has invalid file: {file_name!r}")
        result[key] = entry
    return result


def referenced_files_for_entries(entries: list[dict[str, Any]]) -> set[str]:
    refs: set[str] = set()
    for entry in entries:
        for key in ("index_file", "scale_file", "codebook_file", "payload_file"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                refs.add(value)
    return refs


def referenced_files(manifest: dict[str, Any]) -> set[str]:
    entries = object_list(manifest, "tensors", "manifest")
    entries.extend(object_list(manifest, "passthrough_tensors", "manifest"))
    entries.extend(object_list(manifest, "codebooks", "manifest"))
    return referenced_files_for_entries(entries) | {
        str(entry["file"])
        for entry in object_list(manifest, "codebooks", "manifest")
        if isinstance(entry.get("file"), str)
    }


def link_file(src: Path, dst: Path, dst_rel: Path, copied_files: list[dict[str, Any]]) -> int:
    if not src.is_file():
        raise SystemExit(f"referenced override file is missing: {src}")
    if dst.exists():
        raise SystemExit(f"overlay destination file already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)
    size = dst.stat().st_size
    copied_files.append({"path": relative_path(dst_rel), "bytes": size})
    return size


def ensure_output_path(base_package: Path, override_packages: list[Path], output_package: Path, overwrite: bool) -> None:
    output_resolved = output_package.resolve(strict=False)
    package_labels = [("base package", base_package)]
    package_labels.extend((f"override package {i}", path) for i, path in enumerate(override_packages))
    for label, package in package_labels:
        package_resolved = package.resolve(strict=False)
        if output_resolved == package_resolved:
            raise SystemExit(f"--output-package must differ from {label}: {package}")
        if package_resolved in output_resolved.parents:
            raise SystemExit(f"--output-package must not be inside {label}: {package}")
    if output_package.exists():
        if not overwrite:
            raise SystemExit(f"output package already exists: {output_package}")
        if output_package.is_dir():
            shutil.rmtree(output_package)
        else:
            output_package.unlink()
    output_package.parent.mkdir(parents=True, exist_ok=True)


def copy_base_package(base_package: Path, output_package: Path) -> None:
    shutil.copytree(base_package, output_package, copy_function=os.link)
    output_manifest_path = manifest_path(output_package)
    if output_manifest_path.exists():
        output_manifest_path.unlink()


def package_entry_source_files(package: Path, entry: dict[str, Any]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for key in ("index_file", "scale_file", "payload_file"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            sources[key] = package / value
    return sources


def overlay_tensor_file_name(kind: str, index: int, tensor_name: str, source: Path) -> Path:
    stem = f"override-{index:03d}-{sanitize(tensor_name)}"
    if kind == "index_file":
        return Path("tensors") / f"{stem}.idx4"
    if kind == "scale_file":
        return Path("tensors") / f"{stem}.scale_u8"
    if kind == "payload_file":
        suffix = source.suffix or ".raw"
        return Path("passthrough") / f"{stem}{suffix}"
    raise ValueError(kind)


def overlay_codebook_file_name(index: int, key: tuple[str, str]) -> Path:
    return Path("codebooks") / f"override-{index:03d}-{sanitize(key[0] + '__' + key[1])}.f32"


def same_file_bytes(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    return left.read_bytes() == right.read_bytes()


def validate_replacement_compatible(name: str, base_entry: dict[str, Any], replacement: dict[str, Any]) -> None:
    for key in ("shape", "dtype", "elements", "family"):
        base_value = base_entry.get(key)
        replacement_value = replacement.get(key)
        if base_value is not None and replacement_value is not None and base_value != replacement_value:
            raise SystemExit(
                f"replacement tensor {name} has incompatible {key}: "
                f"base={base_value!r} replacement={replacement_value!r}"
            )


def validate_replacement_payload(name: str, replacement: dict[str, Any]) -> str:
    has_index = isinstance(replacement.get("index_file"), str)
    has_scale = isinstance(replacement.get("scale_file"), str)
    has_codebook = isinstance(replacement.get("codebook_file"), str)
    has_payload = isinstance(replacement.get("payload_file"), str)
    if has_index or has_scale or has_codebook:
        if not (has_index and has_scale and has_codebook):
            raise SystemExit(
                f"replacement tensor {name} must provide index_file, scale_file, and codebook_file together"
            )
        if has_payload:
            raise SystemExit(f"replacement tensor {name} must not mix AQ files and payload_file")
        return "tensor"
    if has_payload:
        return "passthrough"
    raise SystemExit(f"replacement entry for {name} has no tensor payload files")


def build_overlay(args: argparse.Namespace) -> dict[str, Any]:
    replace_names = list(dict.fromkeys(args.replace_tensor))
    if not replace_names:
        raise SystemExit("at least one --replace-tensor is required")

    base_manifest = read_manifest(args.base_package, "base package")
    base_entries = tensor_entries_by_name(base_manifest, "base package")
    for name in replace_names:
        if name not in base_entries:
            raise SystemExit(f"replace tensor is not present in base package: {name}")

    overrides: dict[str, tuple[int, Path, dict[str, Any], dict[str, Any]]] = {}
    for package_index, package in enumerate(args.override_package):
        manifest = read_manifest(package, f"override package {package_index}")
        entries = tensor_entries_by_name(manifest, f"override package {package_index}")
        for name in replace_names:
            entry = entries.get(name)
            if entry is None:
                continue
            if name in overrides:
                raise SystemExit(f"multiple override packages provide replacement tensor: {name}")
            overrides[name] = (package_index, package, manifest, entry)

    missing = [name for name in replace_names if name not in overrides]
    if missing:
        raise SystemExit(f"replace tensors missing from override packages: {', '.join(missing)}")

    base_tensors = object_list(base_manifest, "tensors", "base package")
    base_passthrough = object_list(base_manifest, "passthrough_tensors", "base package")
    removed_base_entries = [
        entry
        for entry in [*base_tensors, *base_passthrough]
        if entry_name(entry, "base package") in replace_names
    ]
    removed_base_refs = referenced_files_for_entries(removed_base_entries)

    output_tensors = [
        dict(entry) for entry in base_tensors if entry_name(entry, "base package") not in replace_names
    ]
    output_passthrough = [
        dict(entry)
        for entry in base_passthrough
        if entry_name(entry, "base package") not in replace_names
    ]
    output_codebooks = {
        key: dict(entry) for key, entry in codebooks_by_key(base_manifest, "base package").items()
    }
    copied_files: list[dict[str, Any]] = []
    replaced: list[dict[str, Any]] = []

    if not args.dry_run:
        ensure_output_path(args.base_package, args.override_package, args.output_package, args.overwrite)
        copy_base_package(args.base_package, args.output_package)

    for name in replace_names:
        package_index, package, manifest, entry = overrides[name]
        replacement = dict(entry)
        validate_replacement_compatible(name, base_entries[name], replacement)
        source_files = package_entry_source_files(package, entry)
        for file_key, src in source_files.items():
            dst_rel = overlay_tensor_file_name(file_key, package_index, name, src)
            replacement[file_key] = relative_path(dst_rel)
            if not args.dry_run:
                link_file(src, args.output_package / dst_rel, dst_rel, copied_files)

        if "codebook_file" in replacement:
            key = (str(replacement.get("family")), str(replacement.get("candidate_id")))
            override_codebooks = codebooks_by_key(manifest, f"override package {package_index}")
            override_codebook = override_codebooks.get(key)
            if override_codebook is None:
                raise SystemExit(f"override tensor {name} references missing codebook {key}")
            src_codebook = package / str(override_codebook["file"])
            existing = output_codebooks.get(key)
            if existing is not None:
                base_codebook = args.base_package / str(existing["file"])
                if not same_file_bytes(base_codebook, src_codebook):
                    raise SystemExit(f"codebook conflict for {key}: base and override bytes differ")
                replacement["codebook_file"] = existing["file"]
            else:
                dst_rel = overlay_codebook_file_name(package_index, key)
                replacement["codebook_file"] = relative_path(dst_rel)
                codebook_entry = dict(override_codebook)
                codebook_entry["file"] = relative_path(dst_rel)
                output_codebooks[key] = codebook_entry
                if not args.dry_run:
                    link_file(src_codebook, args.output_package / dst_rel, dst_rel, copied_files)

        kind = validate_replacement_payload(name, replacement)
        if kind == "tensor":
            output_tensors.append(replacement)
        else:
            output_passthrough.append(replacement)

        replaced.append(
            {
                "name": name,
                "kind": kind,
                "override_package": str(package),
                "candidate_id": replacement.get("candidate_id"),
                "group_size": replacement.get("group_size"),
            }
        )

    used_codebook_keys = {
        (str(entry.get("family")), str(entry.get("candidate_id")))
        for entry in output_tensors
        if isinstance(entry.get("family"), str) and isinstance(entry.get("candidate_id"), str)
    }
    output_codebook_list = [
        entry for key, entry in sorted(output_codebooks.items()) if key in used_codebook_keys
    ]

    output_manifest = dict(base_manifest)
    output_manifest["tensors"] = output_tensors
    output_manifest["passthrough_tensors"] = output_passthrough
    output_manifest["codebooks"] = output_codebook_list
    final_refs = referenced_files(output_manifest)
    removed_files = sorted(removed_base_refs - final_refs)

    if not args.dry_run:
        for rel in removed_files:
            path = args.output_package / rel
            if path.exists():
                path.unlink()
        write_json(manifest_path(args.output_package), output_manifest)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "base_package": str(args.base_package),
        "override_packages": [str(path) for path in args.override_package],
        "output_package": str(args.output_package),
        "dry_run": bool(args.dry_run),
        "replace_tensors": replace_names,
        "replaced": replaced,
        "tensor_count": len(output_tensors),
        "passthrough_tensor_count": len(output_passthrough),
        "codebook_count": len(output_codebook_list),
        "removed_base_files": removed_files,
        "linked_files": copied_files,
    }
    if args.summary_json:
        write_json(args.summary_json, summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = build_overlay(args)
    action = "validated" if args.dry_run else "built"
    print(
        f"ullm-prototype-package-overlay {action}=true "
        f"replaced={len(summary['replaced'])} output={args.output_package}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
