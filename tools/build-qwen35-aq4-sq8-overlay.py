#!/usr/bin/env python3
"""Build the exact Qwen3.5-9B AQ4 QKV/Z SQ8_0 overlay artifact."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any


BINDING_SCHEMA = "ullm.qwen35_aq4_sq8_qkv_z_overlay.v2"
IMPLEMENTATION_ID = "qwen35_aq4_sq8_linear_qkv_z_overlay_v1"
FORMAT_ID = "AQ4_0"
OVERLAY_FORMAT_ID = "SQ8_0"
SQ_MANIFEST_SCHEMA = "sq-fp8-artifact-v0.1"
CONTENT_DOMAIN = b"ullm.qwen35-aq4-sq8-overlay-content.v1\0"
TENSOR_SET_DOMAIN = b"ullm.qwen35-aq4-sq8-overlay-tensor-set.v1\0"
EXPECTED_TENSOR_COUNT = 48
SCALE_BLOCK_COLS = 256
PROMOTION_SCHEMA = "ullm.aq4_resident_promotion.v1"
BASE_PROMOTION_NAME = "promotion-paged-decode-split-v1.json"
OVERLAY_PROMOTION_NAME = "promotion-sq8-linear-qkv-z-overlay-v0.1.json"


class BuildError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", required=True, type=Path)
    parser.add_argument("--base-package", required=True, type=Path)
    parser.add_argument("--output-artifact", required=True, type=Path)
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--replace-existing-hardening", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BuildError(f"invalid JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise BuildError(f"JSON root must be an object: {path}")
    return value


def _reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BuildError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise BuildError(f"expected regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file_range(path: Path, offset: int, size: int) -> str:
    if offset < 0 or size < 0 or path.is_symlink() or not path.is_file():
        raise BuildError(f"invalid logical tensor range: {path}")
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as handle:
        handle.seek(offset)
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise BuildError(f"logical tensor payload is truncated: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def safetensors_headers(path: Path) -> tuple[int, dict[str, dict[str, Any]]]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise BuildError(f"safetensors header length is truncated: {path}")
        header_length = int.from_bytes(raw_length, "little")
        if not 0 < header_length <= 16 * 1024 * 1024:
            raise BuildError(f"safetensors header length differs: {path}")
        raw_header = handle.read(header_length)
    try:
        header = json.loads(raw_header, object_pairs_hook=_reject_duplicate)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BuildError(f"invalid safetensors header {path}: {error}") from error
    if not isinstance(header, dict):
        raise BuildError(f"safetensors header root differs: {path}")
    header.pop("__metadata__", None)
    if not all(isinstance(name, str) and isinstance(value, dict) for name, value in header.items()):
        raise BuildError(f"safetensors tensor header differs: {path}")
    return 8 + header_length, header


def exact_tensor_names(config: dict[str, Any]) -> list[str]:
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise BuildError("source config.text_config must be an object")
    layer_types = text_config.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != 32:
        raise BuildError("Qwen3.5-9B source must declare exactly 32 layer_types")
    if any(value not in {"linear_attention", "full_attention"} for value in layer_types):
        raise BuildError("source contains an unsupported layer type")
    linear_layers = [index for index, value in enumerate(layer_types) if value == "linear_attention"]
    if len(linear_layers) != 24:
        raise BuildError("Qwen3.5-9B source must declare exactly 24 linear-attention layers")
    names: list[str] = []
    for layer in linear_layers:
        names.append(f"model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight")
        names.append(f"model.language_model.layers.{layer}.linear_attn.in_proj_z.weight")
    if len(names) != EXPECTED_TENSOR_COUNT:
        raise BuildError(f"overlay tensor count differs: {len(names)}")
    return names


def tensor_set_sha256(names: list[str]) -> str:
    digest = hashlib.sha256(TENSOR_SET_DOMAIN)
    for name in sorted(names):
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def exact_include_regex(names: list[str]) -> str:
    import re

    return "^(?:" + "|".join(re.escape(name) for name in names) + ")$"


def _contained_regular_file(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise BuildError(f"{label} path must be a non-empty string")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise BuildError(f"{label} path must be contained and relative")
    root = root.resolve()
    path = root / candidate
    if path.is_symlink() or not path.is_file():
        raise BuildError(f"{label} must be a regular file")
    canonical = path.resolve()
    try:
        canonical.relative_to(root)
    except ValueError as error:
        raise BuildError(f"{label} escapes artifact root") from error
    return canonical


def _expected_entry(name: str) -> tuple[str, list[int]]:
    if name.endswith(".linear_attn.in_proj_qkv.weight"):
        return "linear_attn_qkv", [8192, 4096]
    if name.endswith(".linear_attn.in_proj_z.weight"):
        return "linear_attn_z", [4096, 4096]
    raise BuildError(f"unexpected overlay tensor: {name}")


def validate_sq_manifest(artifact_dir: Path, names: list[str]) -> tuple[dict[str, Any], str]:
    manifest_path = _contained_regular_file(artifact_dir, "sq_manifest.json", "SQ manifest")
    manifest = read_json(manifest_path)
    candidate = manifest.get("candidate")
    storage = manifest.get("storage")
    entries = manifest.get("fp8_tensors")
    if manifest.get("schema_version") != SQ_MANIFEST_SCHEMA or not isinstance(candidate, dict):
        raise BuildError("SQ manifest schema/candidate differs")
    if candidate.get("id") != OVERLAY_FORMAT_ID or candidate.get("format_id") != OVERLAY_FORMAT_ID:
        raise BuildError("SQ manifest format identity differs")
    if not isinstance(storage, dict) or storage.get("fp8_tensor_count") != EXPECTED_TENSOR_COUNT:
        raise BuildError("SQ manifest tensor count differs")
    if not isinstance(entries, list) or len(entries) != EXPECTED_TENSOR_COUNT:
        raise BuildError("SQ manifest fp8_tensors count differs")
    by_name: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise BuildError("SQ manifest tensor entry differs")
        name = entry["name"]
        if name in by_name:
            raise BuildError(f"duplicate SQ tensor name: {name}")
        by_name[name] = entry
    if set(by_name) != set(names):
        raise BuildError("SQ manifest tensor set differs from exact QKV/Z set")

    digest = hashlib.sha256(CONTENT_DOMAIN)
    for name in sorted(names):
        entry = by_name[name]
        family, shape = _expected_entry(name)
        if (
            entry.get("family") != family
            or entry.get("source_dtype") != "BF16"
            or entry.get("shape") != shape
            or entry.get("scale_granularity") != "row_block"
            or entry.get("scale_block_cols") != SCALE_BLOCK_COLS
            or entry.get("scale_dtype") != "f32"
            or entry.get("payload_dtype") != "fp8_e4m3"
        ):
            raise BuildError(f"SQ tensor dtype/shape/layout differs: {name}")
        payload = _contained_regular_file(artifact_dir, entry.get("payload_file"), f"{name} payload")
        scale = _contained_regular_file(artifact_dir, entry.get("scale_file"), f"{name} scale")
        payload_sha = sha256_file(payload)
        scale_sha = sha256_file(scale)
        if entry.get("payload_sha256") != payload_sha or entry.get("scale_sha256") != scale_sha:
            raise BuildError(f"SQ tensor payload identity differs: {name}")
        if entry.get("payload_bytes") != payload.stat().st_size or entry.get("scale_bytes") != scale.stat().st_size:
            raise BuildError(f"SQ tensor payload size differs: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload_sha.encode("ascii"))
        digest.update(b"\0")
        digest.update(scale_sha.encode("ascii"))
        digest.update(b"\n")
    return manifest, digest.hexdigest()


def source_provenance(
    source_model_dir: Path,
    artifact_dir: Path,
    manifest: dict[str, Any],
    names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_root = source_model_dir.resolve()
    index = read_json(source_root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise BuildError("source model index weight_map must be an object")
    entries = manifest.get("fp8_tensors")
    if not isinstance(entries, list):
        raise BuildError("SQ manifest fp8_tensors must be an array")
    by_name = {entry.get("name"): entry for entry in entries if isinstance(entry, dict)}
    if len(by_name) != len(entries) or set(by_name) != set(names):
        raise BuildError("SQ manifest tensor mapping set differs")
    names_by_shard: dict[str, list[str]] = {}
    for name in names:
        shard = weight_map.get(name)
        if not isinstance(shard, str) or not shard or Path(shard).is_absolute() or ".." in Path(shard).parts:
            raise BuildError(f"source shard mapping differs: {name}")
        names_by_shard.setdefault(shard, []).append(name)

    shards: list[dict[str, Any]] = []
    tensors: list[dict[str, Any]] = []
    for shard_name in sorted(names_by_shard):
        shard_path = _contained_regular_file(source_root, shard_name, "source safetensors shard")
        data_start, headers = safetensors_headers(shard_path)
        shards.append(
            {
                "path": shard_name,
                "bytes": shard_path.stat().st_size,
                "sha256": sha256_file(shard_path),
            }
        )
        for name in sorted(names_by_shard[shard_name]):
            family, expected_shape = _expected_entry(name)
            del family
            header = headers.get(name)
            if not isinstance(header, dict):
                raise BuildError(f"source shard omits tensor: {name}")
            dtype = header.get("dtype")
            shape = header.get("shape")
            offsets = header.get("data_offsets")
            expected_bytes = 2
            for dimension in expected_shape:
                expected_bytes *= dimension
            if (
                dtype != "BF16"
                or shape != expected_shape
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) and value >= 0 for value in offsets)
                or offsets[1] - offsets[0] != expected_bytes
            ):
                raise BuildError(f"source logical tensor geometry differs: {name}")
            entry = by_name[name]
            payload_path = _contained_regular_file(
                artifact_dir, entry.get("payload_file"), f"{name} payload"
            )
            scale_path = _contained_regular_file(
                artifact_dir, entry.get("scale_file"), f"{name} scale"
            )
            tensors.append(
                {
                    "name": name,
                    "source": {
                        "file": shard_name,
                        "dtype": dtype,
                        "shape": shape,
                        "logical_sha256": sha256_file_range(
                            shard_path, data_start + offsets[0], expected_bytes
                        ),
                    },
                    "overlay": {
                        "payload": {
                            "path": entry["payload_file"],
                            "bytes": payload_path.stat().st_size,
                            "sha256": sha256_file(payload_path),
                        },
                        "scale": {
                            "path": entry["scale_file"],
                            "bytes": scale_path.stat().st_size,
                            "sha256": sha256_file(scale_path),
                        },
                    },
                }
            )
    tensors.sort(key=lambda item: item["name"])
    if len(tensors) != EXPECTED_TENSOR_COUNT or len({item["name"] for item in tensors}) != len(tensors):
        raise BuildError("source provenance exact tensor mapping differs")
    return shards, tensors


def validate_bound_source_provenance(
    binding: dict[str, Any],
    source_model_dir: Path,
    artifact_dir: Path,
    manifest: dict[str, Any],
    names: list[str],
) -> None:
    source = binding.get("source")
    if not isinstance(source, dict):
        raise BuildError("binding source provenance is missing")
    shards = source.get("shards")
    tensors = source.get("tensors")
    if not isinstance(shards, list) or not isinstance(tensors, list):
        raise BuildError("binding source provenance arrays are missing")
    shard_paths = [item.get("path") for item in shards if isinstance(item, dict)]
    tensor_names = [item.get("name") for item in tensors if isinstance(item, dict)]
    if (
        len(shard_paths) != len(shards)
        or len(set(shard_paths)) != len(shard_paths)
        or len(tensor_names) != len(tensors)
        or len(set(tensor_names)) != len(tensor_names)
        or set(tensor_names) != set(names)
    ):
        raise BuildError("binding source provenance set is missing, duplicated, or mismatched")
    expected_shards, expected_tensors = source_provenance(
        source_model_dir, artifact_dir, manifest, names
    )
    if shards != expected_shards or tensors != expected_tensors:
        raise BuildError("binding source provenance identity differs")


def create_binding(
    artifact_dir: Path,
    source_model_dir: Path,
    package_dir: Path,
    names: list[str],
) -> dict[str, Any]:
    manifest, content_sha = validate_sq_manifest(artifact_dir, names)
    source_root = source_model_dir.resolve()
    package_root = package_dir.resolve()
    config_path = source_root / "config.json"
    index_path = source_root / "model.safetensors.index.json"
    package_manifest = package_root / "manifest.json"
    sq_manifest = artifact_dir.resolve() / "sq_manifest.json"
    shards, tensors = source_provenance(
        source_model_dir, artifact_dir, manifest, names
    )
    binding = {
        "schema_version": BINDING_SCHEMA,
        "format_id": FORMAT_ID,
        "overlay_format_id": OVERLAY_FORMAT_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "sq_manifest": {"path": "sq_manifest.json", "sha256": sha256_file(sq_manifest)},
        "content_sha256": content_sha,
        "tensor_set_sha256": tensor_set_sha256(names),
        "tensor_names": sorted(names),
        "scale": {"granularity": "row_block", "block_cols": SCALE_BLOCK_COLS, "dtype": "f32"},
        "artifact_policy": {
            "uid": os.getuid(),
            "gid": os.getgid(),
            "directory_mode": "0555",
            "file_mode": "0444",
            "regular_file_nlink": 1,
        },
        "source": {
            "model_dir": str(source_root),
            "config_sha256": sha256_file(config_path),
            "index_sha256": sha256_file(index_path),
            "shards": shards,
            "tensors": tensors,
        },
        "package": {
            "root": str(package_root),
            "manifest_sha256": sha256_file(package_manifest),
        },
    }
    path = artifact_dir / "binding.json"
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(binding, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise BuildError(f"refusing to overwrite binding: {path}") from error
    return binding


def overlay_promotion_receipt(
    product_root: Path,
    binding: dict[str, Any],
    binding_sha256: str,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    base_path = product_root / BASE_PROMOTION_NAME
    base = read_json(base_path)
    if base.get("schema_version") != PROMOTION_SCHEMA:
        raise BuildError("base AQ4 promotion receipt schema differs")
    source_commit = base.get("source_commit")
    evidence = base.get("evidence")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise BuildError("base AQ4 promotion source_commit differs")
    if not isinstance(evidence, dict) or set(evidence) != {"path", "sha256"}:
        raise BuildError("base AQ4 promotion evidence identity differs")
    evidence_path = _contained_regular_file(product_root, evidence.get("path"), "AQ4 promotion evidence")
    if evidence.get("sha256") != sha256_file(evidence_path):
        raise BuildError("base AQ4 promotion evidence SHA-256 differs")
    receipt = {
        "schema_version": PROMOTION_SCHEMA,
        "source_commit": source_commit,
        "evidence": {"path": evidence["path"], "sha256": evidence["sha256"]},
        "overlay": {
            "content_sha256": binding["content_sha256"],
            "binding_manifest_sha256": binding_sha256,
            "tensor_set_sha256": binding["tensor_set_sha256"],
            "artifact_inventory": inventory,
        },
    }
    return receipt


def run_legacy_builder(args: argparse.Namespace, names: list[str], output_artifact: Path) -> None:
    builder = Path(__file__).resolve().with_name("build-sq-fp8-w8a16-artifact.py")
    command = [
        sys.executable,
        str(builder),
        "--source-model-dir",
        str(args.source_model_dir),
        "--output-artifact",
        str(output_artifact),
        "--base-package",
        str(args.base_package),
        "--candidate-id",
        OVERLAY_FORMAT_ID,
        "--scale-granularity",
        "row_block",
        "--scale-block-cols",
        str(SCALE_BLOCK_COLS),
        "--row-chunk",
        str(args.row_chunk),
        "--include-regex",
        exact_include_regex(names),
        "--max-tensors",
        str(EXPECTED_TENSOR_COUNT),
    ]
    subprocess.run(command, check=True)


def write_create_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise BuildError(f"refusing to overwrite JSON: {path}") from error


def harden_artifact(root: Path) -> None:
    paths = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in paths:
        metadata = path.lstat()
        if path.is_symlink():
            raise BuildError(f"artifact contains symlink: {path}")
        if metadata.st_uid != os.getuid() or metadata.st_gid != os.getgid():
            raise BuildError(f"artifact owner differs: {path}")
        if path.is_file():
            if metadata.st_nlink != 1:
                raise BuildError(f"artifact file link count differs: {path}")
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
        else:
            raise BuildError(f"artifact contains unsupported file type: {path}")
    root.chmod(0o555)


def artifact_inventory(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    expected = {Path("binding.json"), Path("sq_manifest.json")}
    for tensor in binding["source"]["tensors"]:
        expected.add(Path(tensor["overlay"]["payload"]["path"]))
        expected.add(Path(tensor["overlay"]["scale"]["path"]))
    found: set[Path] = set()
    directories = 0
    regular_file_bytes = 0
    for path in [root, *root.rglob("*")]:
        metadata = path.lstat()
        if path.is_symlink():
            raise BuildError(f"artifact contains symlink: {path}")
        if metadata.st_uid != os.getuid() or metadata.st_gid != os.getgid():
            raise BuildError(f"artifact owner differs: {path}")
        if path.is_dir():
            directories += 1
            if metadata.st_mode & 0o777 != 0o555:
                raise BuildError(f"artifact directory mode differs: {path}")
        elif path.is_file():
            relative = path.relative_to(root)
            if not relative.parts or relative in found:
                raise BuildError(f"artifact file path differs: {path}")
            found.add(relative)
            regular_file_bytes += metadata.st_size
            if metadata.st_mode & 0o777 != 0o444 or metadata.st_nlink != 1:
                raise BuildError(f"artifact file immutability differs: {path}")
        else:
            raise BuildError(f"artifact contains unsupported file type: {path}")
    if found != expected:
        raise BuildError("artifact exact file inventory differs")
    return {
        "uid": os.getuid(),
        "gid": os.getgid(),
        "directory_mode": "0555",
        "file_mode": "0444",
        "regular_file_nlink": 1,
        "directory_count": directories,
        "regular_file_count": len(found),
        "regular_file_bytes": regular_file_bytes,
        "symlink_count": 0,
    }


def _rename_exchange(left: Path, right: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise BuildError("atomic rename exchange is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(left), -100, os.fsencode(right), 2) != 0:
        error_number = ctypes.get_errno()
        raise BuildError(f"atomic rename exchange failed: {os.strerror(error_number)}")


def _make_tree_removable(root: Path) -> None:
    if not root.exists():
        return
    for path in [root, *root.rglob("*")]:
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o755)
        elif path.is_file() and not path.is_symlink():
            path.chmod(0o644)


def atomic_publish_directory(staged: Path, output: Path, replace: bool, content_sha256: str) -> None:
    if output.exists():
        if not replace or output.is_symlink() or not output.is_dir():
            raise BuildError(f"output artifact already exists: {output}")
        existing = read_json(output / "binding.json")
        if existing.get("content_sha256") != content_sha256:
            raise BuildError("existing artifact content differs; hardening replacement refused")
        _rename_exchange(staged, output)
        _make_tree_removable(staged)
        shutil.rmtree(staged)
    else:
        os.rename(staged, output)


def atomic_publish_json(staged: Path, output: Path, replace: bool) -> None:
    if output.exists():
        if not replace or output.is_symlink() or not output.is_file():
            raise BuildError(f"JSON output already exists: {output}")
        _rename_exchange(staged, output)
        staged.chmod(0o644)
        staged.unlink()
    else:
        os.rename(staged, output)


def main() -> int:
    args = parse_args()
    if args.row_chunk <= 0:
        raise BuildError("row-chunk must be positive")
    if args.output_artifact.exists() and not args.replace_existing_hardening:
        raise BuildError(f"output artifact already exists: {args.output_artifact}")
    product_root = args.base_package.resolve().parent
    promotion_path = product_root / OVERLAY_PROMOTION_NAME
    if promotion_path.exists() and not args.replace_existing_hardening:
        raise BuildError(f"promotion receipt already exists: {promotion_path}")
    output_parent = args.output_artifact.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    if args.summary_json is not None:
        summary_parent = args.summary_json.parent.resolve()
        try:
            summary_parent.relative_to(args.output_artifact.resolve())
        except ValueError:
            pass
        else:
            raise BuildError("summary JSON must be outside immutable artifact directory")
    config = read_json(args.source_model_dir / "config.json")
    names = exact_tensor_names(config)
    staged_artifact = output_parent / (
        f".{args.output_artifact.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staged_receipt = product_root / f".{OVERLAY_PROMOTION_NAME}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    binding: dict[str, Any]
    inventory: dict[str, Any]
    binding_sha256: str
    try:
        run_legacy_builder(args, names, staged_artifact)
        binding = create_binding(
            staged_artifact, args.source_model_dir, args.base_package, names
        )
        harden_artifact(staged_artifact)
        inventory = artifact_inventory(staged_artifact, binding)
        binding_sha256 = sha256_file(staged_artifact / "binding.json")
        receipt = overlay_promotion_receipt(
            product_root, binding, binding_sha256, inventory
        )
        write_create_new_json(staged_receipt, receipt)
        staged_receipt.chmod(0o444)
        atomic_publish_directory(
            staged_artifact,
            args.output_artifact,
            args.replace_existing_hardening,
            binding["content_sha256"],
        )
        atomic_publish_json(
            staged_receipt, promotion_path, args.replace_existing_hardening
        )
    finally:
        if staged_artifact.exists():
            _make_tree_removable(staged_artifact)
            shutil.rmtree(staged_artifact)
        if staged_receipt.exists():
            staged_receipt.chmod(0o644)
            staged_receipt.unlink()
    summary = {
        "schema_version": BINDING_SCHEMA,
        "artifact": str(args.output_artifact.resolve()),
        "binding_manifest": str((args.output_artifact / "binding.json").resolve()),
        "binding_manifest_sha256": binding_sha256,
        "content_sha256": binding["content_sha256"],
        "tensor_set_sha256": binding["tensor_set_sha256"],
        "tensor_count": len(names),
        "source_shard_count": len(binding["source"]["shards"]),
        "source_tensor_count": len(binding["source"]["tensors"]),
        "artifact_inventory": inventory,
        "atomic_publish": True,
        "promotion_receipt": str(promotion_path),
        "promotion_receipt_sha256": sha256_file(promotion_path),
    }
    if args.summary_json is not None:
        write_create_new_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
