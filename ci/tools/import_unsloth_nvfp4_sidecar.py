#!/usr/bin/env python3
"""Import Unsloth compressed-tensors NVFP4 MLP weights into sLLM sidecar v1.

The importer is deliberately positional and bounded: it reads only reviewed
safetensors ranges, preserves the packed E2M1 and E4M3 block-scale bytes, and
converts compressed-tensors' reciprocal global scale into sLLM's multiplicative
FP32 tensor scale. It never materializes a BF16 copy of the imported weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, BinaryIO


SCHEMA = "sllm-nvfp4-sidecar-v1"
BLOCK_SIZE = 16
VALUE_SUFFIX = ".sllm_nvfp4_value"
BLOCK_SCALE_SUFFIX = ".sllm_nvfp4_block_scale"
TENSOR_SCALE_SUFFIX = ".sllm_nvfp4_tensor_scale"
SOURCE_REPOSITORY = "unsloth/gemma-4-12b-it-NVFP4"
SOURCE_REVISION = "b1f649734b34aa5575b03d186abd1b9be3d0d5c4"
SOURCE_ARTIFACT_SHA256 = "7c2ee23298e7c3a9247e8947597dca5a38f8b791a0322487466d2bfad8ce704b"


class ContractError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_header(path: Path) -> tuple[int, dict[str, Any], bytes]:
    with path.open("rb") as source:
        raw_length = source.read(8)
        if len(raw_length) != 8:
            raise ContractError(f"truncated safetensors header: {path}")
        length = struct.unpack("<Q", raw_length)[0]
        if length == 0 or length > 256 * 1024 * 1024:
            raise ContractError(f"invalid safetensors header length: {path}")
        raw_header = source.read(length)
        if len(raw_header) != length:
            raise ContractError(f"truncated safetensors header payload: {path}")
    header = json.loads(raw_header)
    if not isinstance(header, dict):
        raise ContractError(f"safetensors header is not an object: {path}")
    return 8 + length, header, raw_header


def require_tensor(header: dict[str, Any], name: str, dtype: str, shape: list[int]) -> dict[str, Any]:
    value = header.get(name)
    if not isinstance(value, dict) or value.get("dtype") != dtype or value.get("shape") != shape:
        raise ContractError(f"tensor contract differs: {name}")
    offsets = value.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(offset, int) for offset in offsets)
        or offsets[0] < 0
        or offsets[1] < offsets[0]
    ):
        raise ContractError(f"tensor range differs: {name}")
    return value


def hash_range(source: BinaryIO, data_start: int, offsets: list[int]) -> str:
    digest = hashlib.sha256()
    remaining = offsets[1] - offsets[0]
    source.seek(data_start + offsets[0])
    while remaining:
        chunk = source.read(min(remaining, 16 * 1024 * 1024))
        if not chunk:
            raise ContractError("source artifact is truncated")
        digest.update(chunk)
        remaining -= len(chunk)
    return "sha256:" + digest.hexdigest()


def read_range(source: BinaryIO, data_start: int, offsets: list[int]) -> bytes:
    source.seek(data_start + offsets[0])
    length = offsets[1] - offsets[0]
    value = source.read(length)
    if len(value) != length:
        raise ContractError("source artifact is truncated")
    return value


def copy_range(source: BinaryIO, destination: BinaryIO, data_start: int, offsets: list[int]) -> None:
    source.seek(data_start + offsets[0])
    remaining = offsets[1] - offsets[0]
    while remaining:
        chunk = source.read(min(remaining, 16 * 1024 * 1024))
        if not chunk:
            raise ContractError("source artifact is truncated")
        destination.write(chunk)
        remaining -= len(chunk)


def mlp_names() -> list[str]:
    return [
        f"model.language_model.layers.{layer}.mlp.{projection}_proj.weight"
        for layer in range(48)
        for projection in ("down", "gate", "up")
    ]


def lock_identity(lock_path: Path) -> tuple[dict[str, Any], str]:
    lock = json.loads(lock_path.read_text())
    try:
        model = lock["model"]
        repo_id = model["repo_id"]
        revision = model["resolved_revision"]
        fingerprint = lock["fingerprint"]
    except (KeyError, TypeError) as error:
        raise ContractError("source lock identity is incomplete") from error
    if not all(isinstance(value, str) and value for value in (repo_id, revision, fingerprint)):
        raise ContractError("source lock identity is invalid")
    return lock, sha256_file(lock_path)


def convert(args: argparse.Namespace) -> None:
    lock, lock_sha256 = lock_identity(args.source_lock)
    quantized = args.quantized_cache / "model.safetensors"
    bf16 = args.bf16_cache / "model.safetensors"
    if sha256_file(quantized) != "sha256:" + SOURCE_ARTIFACT_SHA256:
        raise ContractError("Unsloth artifact SHA-256 differs")
    q_start, q_header, q_header_bytes = read_header(quantized)
    b_start, b_header, _ = read_header(bf16)
    if hashlib.sha256(q_header_bytes).hexdigest() != args.expected_header_sha256:
        raise ContractError("Unsloth header SHA-256 differs")

    selected = mlp_names()
    if args.tensor:
        requested = set(args.tensor)
        missing = requested.difference(selected)
        if missing:
            raise ContractError(f"requested tensor is not a Gemma MLP weight: {sorted(missing)}")
        selected = sorted(requested)
    records: list[dict[str, Any]] = []
    payloads: list[tuple[str, str, list[int], int, list[int] | bytes]] = []
    with quantized.open("rb") as q_source, bf16.open("rb") as b_source:
        for name in selected:
            source_meta = b_header.get(name)
            if not isinstance(source_meta, dict) or source_meta.get("dtype") != "BF16":
                raise ContractError(f"BF16 source tensor is absent: {name}")
            shape = source_meta.get("shape")
            if not isinstance(shape, list) or len(shape) != 2 or not all(isinstance(v, int) and v > 0 for v in shape):
                raise ContractError(f"BF16 source shape differs: {name}")
            rows, columns = shape
            packed_name = name.removesuffix(".weight") + ".weight_packed"
            scale_name = name.removesuffix(".weight") + ".weight_scale"
            global_name = name.removesuffix(".weight") + ".weight_global_scale"
            packed_meta = require_tensor(q_header, packed_name, "U8", [rows, columns // 2])
            scale_meta = require_tensor(q_header, scale_name, "F8_E4M3", [rows, math.ceil(columns / BLOCK_SIZE)])
            global_meta = require_tensor(q_header, global_name, "F32", [1])
            if columns % 2 != 0:
                raise ContractError(f"Unsloth packed tensor has an odd K dimension: {name}")
            global_bytes = read_range(q_source, q_start, global_meta["data_offsets"])
            reciprocal_scale = struct.unpack("<f", global_bytes)[0]
            if not math.isfinite(reciprocal_scale) or reciprocal_scale <= 0.0:
                raise ContractError(f"Unsloth global scale is invalid: {name}")
            tensor_scale = struct.pack("<f", 1.0 / reciprocal_scale)
            source_hash = hash_range(b_source, b_start, source_meta["data_offsets"])
            value_hash = hash_range(q_source, q_start, packed_meta["data_offsets"])
            block_hash = hash_range(q_source, q_start, scale_meta["data_offsets"])
            tensor_hash = sha256_bytes(tensor_scale)
            value_name = name + VALUE_SUFFIX
            block_name = name + BLOCK_SCALE_SUFFIX
            tensor_name = name + TENSOR_SCALE_SUFFIX
            payloads.extend(
                [
                    (value_name, "U8", [rows * columns // 2], q_start, packed_meta["data_offsets"]),
                    (block_name, "U8", [rows, math.ceil(columns / BLOCK_SIZE)], q_start, scale_meta["data_offsets"]),
                    (tensor_name, "F32", [1], 0, tensor_scale),
                ]
            )
            records.append(
                {
                    "name": name,
                    "logical_shape": shape,
                    "source_dtype": "BF16",
                    "source_sha256": source_hash,
                    "value_name": value_name,
                    "value_sha256": value_hash,
                    "packing": "low-nibble-first-row-major",
                    "block_scale_name": block_name,
                    "block_scale_sha256": block_hash,
                    "block_size": BLOCK_SIZE,
                    "block_axis": 1,
                    "block_scale_dtype": "F8_E4M3FN",
                    "tensor_scale_name": tensor_name,
                    "tensor_scale_sha256": tensor_hash,
                    "tensor_scale_dtype": "F32",
                    "rounding": "nearest-even",
                    "saturation": "finite",
                    "zero_point": False,
                }
            )

    header: dict[str, Any] = {"__metadata__": {"format": SCHEMA}}
    cursor = 0
    for name, dtype, shape, _, source in payloads:
        length = len(source) if isinstance(source, bytes) else source[1] - source[0]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + length]}
        cursor += length
    header_bytes = canonical_bytes(header)
    header_bytes += b" " * ((-len(header_bytes)) % 8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with quantized.open("rb") as source, temporary.open("wb") as destination:
        destination.write(struct.pack("<Q", len(header_bytes)))
        destination.write(header_bytes)
        for _, _, _, data_start, payload in payloads:
            if isinstance(payload, bytes):
                destination.write(payload)
            else:
                copy_range(source, destination, data_start, payload)
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(args.output)

    model = lock["model"]
    manifest = {
        "schema_version": SCHEMA,
        "source": {
            "repo_id": model["repo_id"],
            "resolved_revision": model["resolved_revision"],
            "lock_fingerprint": lock["fingerprint"],
            "lock_sha256": lock_sha256,
        },
        "format_source": {
            "repository": "https://github.com/NVIDIA/TransformerEngine",
            "tag": "v2.18",
            "commit": "27486e03cfc1fa41f6932dcecdc47c71c47eac3e",
            "license": "BSD-3-Clause",
            "contract": "sllm-weight-nvfp4-v1",
        },
        "tool": {
            "repository": args.tool_repository,
            "commit": args.tool_commit,
            "path": "ci/tools/import_unsloth_nvfp4_sidecar.py",
            "sha256": sha256_file(Path(__file__).resolve()),
            "numpy": "not-used",
            "arguments": {
                "tensor": sorted(args.tensor),
                **({"selection": "gemma-mlp-subset"} if args.tensor else {}),
                "source_repository": SOURCE_REPOSITORY,
                "source_revision": SOURCE_REVISION,
                "source_artifact_sha256": "sha256:" + SOURCE_ARTIFACT_SHA256,
                "global_scale_convention": "multiplicative-reciprocal-of-compressed-tensors-weight-global-scale",
                "input_global_scales_applied": False,
            },
        },
        "artifact": {
            "path": args.output.name,
            "size_bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
            "tensor_count": len(records),
        },
        "tensors": records,
    }
    manifest["fingerprint"] = sha256_bytes(canonical_bytes(manifest))
    args.manifest.write_bytes(canonical_bytes(manifest) + b"\n")
    print(
        f"Unsloth NVFP4 import: PASS tensors={len(records)} bytes={args.output.stat().st_size} "
        f"fingerprint={manifest['fingerprint']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-lock", required=True, type=Path)
    parser.add_argument("--bf16-cache", required=True, type=Path)
    parser.add_argument("--quantized-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-header-sha256", required=True)
    parser.add_argument("--tool-repository", default="local")
    parser.add_argument("--tool-commit", default="working-tree")
    parser.add_argument("--tensor", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    try:
        convert(parse_args())
        return 0
    except (ContractError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Unsloth NVFP4 import: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
