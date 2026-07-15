#!/usr/bin/env python3
"""Build the exact Qwen3.5-9B AQ4 QKV/Z SQ8_0 overlay artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


BINDING_SCHEMA = "ullm.qwen35_aq4_sq8_qkv_z_overlay.v1"
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


def create_binding(
    artifact_dir: Path,
    source_model_dir: Path,
    package_dir: Path,
    names: list[str],
) -> dict[str, Any]:
    _, content_sha = validate_sq_manifest(artifact_dir, names)
    source_root = source_model_dir.resolve()
    package_root = package_dir.resolve()
    config_path = source_root / "config.json"
    index_path = source_root / "model.safetensors.index.json"
    package_manifest = package_root / "manifest.json"
    sq_manifest = artifact_dir.resolve() / "sq_manifest.json"
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
        "source": {
            "model_dir": str(source_root),
            "config_sha256": sha256_file(config_path),
            "index_sha256": sha256_file(index_path),
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


def create_overlay_promotion_receipt(
    product_root: Path,
    content_sha256: str,
) -> tuple[Path, dict[str, Any]]:
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
        "overlay": {"content_sha256": content_sha256},
    }
    output = product_root / OVERLAY_PROMOTION_NAME
    write_create_new_json(output, receipt)
    return output, receipt


def run_legacy_builder(args: argparse.Namespace, names: list[str]) -> None:
    builder = Path(__file__).resolve().with_name("build-sq-fp8-w8a16-artifact.py")
    command = [
        sys.executable,
        str(builder),
        "--source-model-dir",
        str(args.source_model_dir),
        "--output-artifact",
        str(args.output_artifact),
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


def main() -> int:
    args = parse_args()
    if args.row_chunk <= 0:
        raise BuildError("row-chunk must be positive")
    if args.output_artifact.exists():
        raise BuildError(f"output artifact already exists: {args.output_artifact}")
    product_root = args.base_package.resolve().parent
    promotion_path = product_root / OVERLAY_PROMOTION_NAME
    if promotion_path.exists():
        raise BuildError(f"promotion receipt already exists: {promotion_path}")
    config = read_json(args.source_model_dir / "config.json")
    names = exact_tensor_names(config)
    run_legacy_builder(args, names)
    binding = create_binding(args.output_artifact, args.source_model_dir, args.base_package, names)
    promotion_path, _ = create_overlay_promotion_receipt(product_root, binding["content_sha256"])
    summary = {
        "schema_version": BINDING_SCHEMA,
        "artifact": str(args.output_artifact.resolve()),
        "binding_manifest": str((args.output_artifact / "binding.json").resolve()),
        "binding_manifest_sha256": sha256_file(args.output_artifact / "binding.json"),
        "content_sha256": binding["content_sha256"],
        "tensor_set_sha256": binding["tensor_set_sha256"],
        "tensor_count": len(names),
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
