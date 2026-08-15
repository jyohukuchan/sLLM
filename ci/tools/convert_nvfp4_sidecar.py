#!/usr/bin/env python3
"""Create a deterministic weight-only NVFP4 sidecar from reviewed BF16 weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "sllm-nvfp4-sidecar-v1"
BLOCK_SIZE = 16
FP4_MAX = np.float32(6.0)
FP8_MAX = np.float32(448.0)
VALUE_SUFFIX = ".sllm_nvfp4_value"
BLOCK_SCALE_SUFFIX = ".sllm_nvfp4_block_scale"
TENSOR_SCALE_SUFFIX = ".sllm_nvfp4_tensor_scale"
E2M1_POSITIVE = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


class ContractError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes | memoryview | np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as source:
        raw = source.read(8)
        if len(raw) != 8:
            raise ContractError(f"truncated safetensors header: {path}")
        length = struct.unpack("<Q", raw)[0]
        if length == 0 or length > 256 * 1024 * 1024:
            raise ContractError(f"invalid safetensors header length: {path}")
        header = json.loads(source.read(length))
    if not isinstance(header, dict):
        raise ContractError(f"safetensors header is not an object: {path}")
    return 8 + length, header


def source_catalog(cache: Path) -> dict[str, tuple[Path, int, dict[str, Any]]]:
    index_path = cache / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text()).get("weight_map")
        if not isinstance(weight_map, dict):
            raise ContractError("source index has no weight_map")
    else:
        direct = cache / "model.safetensors"
        if not direct.is_file():
            raise ContractError("source cache has neither indexed nor direct safetensors")
        _, direct_header = read_header(direct)
        weight_map = {name: direct.name for name in direct_header if name != "__metadata__"}
    headers: dict[str, tuple[int, dict[str, Any]]] = {}
    result: dict[str, tuple[Path, int, dict[str, Any]]] = {}
    for name, shard_name in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard_name, str) or Path(shard_name).name != shard_name:
            raise ContractError("source index contains an unsafe name")
        shard = cache / shard_name
        if shard_name not in headers:
            headers[shard_name] = read_header(shard)
        data_start, header = headers[shard_name]
        metadata = header.get(name)
        if not isinstance(metadata, dict):
            raise ContractError(f"source tensor is absent: {name}")
        result[name] = (shard, data_start, metadata)
    return result


def is_text_linear(name: str, metadata: dict[str, Any]) -> bool:
    if not name.startswith("model.language_model.layers.") or metadata.get("dtype") != "BF16":
        return False
    if not isinstance(metadata.get("shape"), list) or len(metadata["shape"]) != 2:
        return False
    return name.endswith((
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight", ".linear_attn.in_proj_qkv.weight",
        ".linear_attn.in_proj_z.weight", ".linear_attn.in_proj_b.weight",
        ".linear_attn.in_proj_a.weight", ".linear_attn.out_proj.weight",
        ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight",
    ))


def bf16_to_fp32(raw: memoryview, shape: list[int]) -> np.ndarray:
    elements = int(np.prod(shape, dtype=np.int64))
    if len(raw) != elements * 2:
        raise ContractError("BF16 payload length differs from shape")
    words = np.frombuffer(raw, dtype="<u2", count=elements).astype(np.uint32) << np.uint32(16)
    return words.view(np.float32).reshape(shape)


def e4m3_positive_table() -> np.ndarray:
    result = np.empty(127, dtype=np.float32)
    for bits in range(127):
        exponent, mantissa = (bits >> 3) & 0x0F, bits & 0x07
        result[bits] = np.float32(mantissa * 2.0**-9 if exponent == 0 else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))
    return result


E4M3_POSITIVE = e4m3_positive_table()


def nearest_even(values: np.ndarray, table: np.ndarray) -> np.ndarray:
    upper = np.minimum(np.searchsorted(table, values, side="left"), len(table) - 1)
    lower = np.maximum(upper - 1, 0)
    lower_error, upper_error = np.abs(values - table[lower]), np.abs(table[upper] - values)
    choose_upper = (upper_error < lower_error) | ((upper_error == lower_error) & ((upper & 1) == 0) & ((lower & 1) != 0))
    return np.where(choose_upper, upper, lower).astype(np.uint8)


def encode_e4m3(values: np.ndarray) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ContractError("non-finite E4M3 input")
    sign = np.signbit(values).astype(np.uint8) << np.uint8(7)
    return nearest_even(np.minimum(np.abs(values), FP8_MAX), E4M3_POSITIVE) | sign


def decode_e4m3(bits: np.ndarray) -> np.ndarray:
    sign = np.where((bits & np.uint8(0x80)) == 0, np.float32(1.0), np.float32(-1.0))
    return sign * E4M3_POSITIVE[bits & np.uint8(0x7F)]


def encode_e2m1(values: np.ndarray) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ContractError("non-finite E2M1 input")
    sign = np.signbit(values).astype(np.uint8) << np.uint8(3)
    return nearest_even(np.minimum(np.abs(values), FP4_MAX), E2M1_POSITIVE) | sign


def quantize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ContractError("NVFP4 source must be a finite rank-2 matrix")
    rows, columns = matrix.shape
    global_amax = np.max(np.abs(matrix)).astype(np.float32)
    tensor_scale = np.float32(1.0 if global_amax == 0 else global_amax / (FP8_MAX * FP4_MAX))
    blocks = (columns + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_scale_bits = np.empty((rows, blocks), dtype=np.uint8)
    codes = np.zeros(matrix.size, dtype=np.uint8)
    # Vectorize a bounded number of rows at a time. This retains the exact
    # row-major block and rounding contract while bounding temporary memory for
    # full-model conversion.
    rows_per_chunk = 32
    padded_columns = blocks * BLOCK_SIZE
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_count = row_end - row_start
        padded = np.zeros((row_count, padded_columns), dtype=np.float32)
        padded[:, :columns] = matrix[row_start:row_end]
        block_values = padded.reshape(row_count, blocks, BLOCK_SIZE)
        block_amax = np.max(np.abs(block_values), axis=2).astype(np.float32)
        # Transformer Engine v2.18 preserves zero and underflowed E4M3
        # decode scales. Such a block canonically quantizes to all-zero E2M1
        # values rather than dividing by a zero decoded scale.
        raw_scale = ((block_amax / FP4_MAX) / tensor_scale).astype(np.float32)
        scale_bits = encode_e4m3(raw_scale)
        block_scale_bits[row_start:row_end] = scale_bits
        decoded_scale = decode_e4m3(scale_bits).astype(np.float32)
        denominator = decoded_scale[..., np.newaxis] * tensor_scale
        normalized = np.zeros_like(block_values)
        np.divide(block_values, denominator, out=normalized, where=denominator != 0)
        chunk_codes = encode_e2m1(normalized).reshape(row_count, padded_columns)[:, :columns]
        codes[row_start * columns : row_end * columns] = chunk_codes.reshape(-1)
    packed = np.zeros((codes.size + 1) // 2, dtype=np.uint8)
    packed[:] = codes[0::2]
    packed[: codes[1::2].size] |= codes[1::2] << np.uint8(4)
    return packed, block_scale_bits, np.asarray([tensor_scale], dtype="<f4")


def convert(args: argparse.Namespace) -> None:
    catalog = source_catalog(args.cache)
    selected = sorted(name for name, (_, _, metadata) in catalog.items() if is_text_linear(name, metadata))
    if args.tensor:
        requested = set(args.tensor)
        missing = requested.difference(selected)
        if missing:
            raise ContractError(f"requested tensor is not a text linear: {sorted(missing)}")
        selected = [name for name in selected if name in requested]
    if not selected:
        raise ContractError("no NVFP4 tensor selected")

    payloads: list[tuple[str, str, list[int], bytes]] = []
    records: list[dict[str, Any]] = []
    for ordinal, name in enumerate(selected, 1):
        shard, data_start, metadata = catalog[name]
        start, end = metadata["data_offsets"]
        with shard.open("rb") as source_file, mmap.mmap(source_file.fileno(), 0, access=mmap.ACCESS_READ) as source_map:
            source_bytes = memoryview(source_map)[data_start + start : data_start + end]
            source_hash = sha256_bytes(source_bytes)
            matrix = bf16_to_fp32(source_bytes, metadata["shape"])
            source_bytes.release()
            packed, block_scales, tensor_scale = quantize(matrix)
        value_name, block_name, tensor_name = name + VALUE_SUFFIX, name + BLOCK_SCALE_SUFFIX, name + TENSOR_SCALE_SUFFIX
        payloads.extend([
            (value_name, "U8", [packed.size], packed.tobytes()),
            (block_name, "U8", list(block_scales.shape), block_scales.tobytes()),
            (tensor_name, "F32", [1], tensor_scale.tobytes()),
        ])
        records.append({
            "name": name, "logical_shape": metadata["shape"], "source_dtype": "BF16", "source_sha256": source_hash,
            "value_name": value_name, "value_sha256": sha256_bytes(packed), "packing": "low-nibble-first-row-major",
            "block_scale_name": block_name, "block_scale_sha256": sha256_bytes(block_scales),
            "block_size": BLOCK_SIZE, "block_axis": 1, "block_scale_dtype": "F8_E4M3FN",
            "tensor_scale_name": tensor_name, "tensor_scale_sha256": sha256_bytes(tensor_scale), "tensor_scale_dtype": "F32",
            "rounding": "nearest-even", "saturation": "finite", "zero_point": False,
        })
        print(f"[{ordinal}/{len(selected)}] {name}", file=sys.stderr)

    header: dict[str, Any] = {"__metadata__": {"format": SCHEMA}}
    cursor = 0
    for name, dtype, shape, payload in payloads:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + len(payload)]}
        cursor += len(payload)
    header_bytes = canonical_bytes(header)
    header_bytes += b" " * ((-len(header_bytes)) % 8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("wb") as destination:
        destination.write(struct.pack("<Q", len(header_bytes)))
        destination.write(header_bytes)
        for _, _, _, payload in payloads:
            destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(args.output)

    source_lock = json.loads(args.source_lock.read_text())
    manifest = {
        "schema_version": SCHEMA,
        "source": {
            "repo_id": source_lock["model"]["repo_id"], "resolved_revision": source_lock["model"]["resolved_revision"],
            "lock_fingerprint": source_lock["fingerprint"], "lock_sha256": sha256_file(args.source_lock),
        },
        "format_source": {
            "repository": "https://github.com/NVIDIA/TransformerEngine", "tag": "v2.18",
            "commit": "27486e03cfc1fa41f6932dcecdc47c71c47eac3e", "license": "BSD-3-Clause",
            "contract": "sllm-weight-nvfp4-v1",
        },
        "tool": {
            "repository": args.tool_repository, "commit": args.tool_commit,
            "path": "ci/tools/convert_nvfp4_sidecar.py", "sha256": sha256_file(Path(__file__).resolve()),
            "numpy": np.__version__, "arguments": {"tensor": sorted(args.tensor)},
        },
        "artifact": {"path": args.output.name, "size_bytes": args.output.stat().st_size, "sha256": sha256_file(args.output), "tensor_count": len(records)},
        "tensors": records,
    }
    manifest["fingerprint"] = sha256_bytes(canonical_bytes(manifest))
    args.manifest.write_bytes(canonical_bytes(manifest) + b"\n")
    print(f"NVFP4 sidecar: PASS tensors={len(records)} bytes={args.output.stat().st_size} fingerprint={manifest['fingerprint']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-lock", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--tool-repository", default="local")
    parser.add_argument("--tool-commit", default="working-tree")
    parser.add_argument("--tensor", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    try:
        convert(parse_args())
        return 0
    except (ContractError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"NVFP4 sidecar: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
