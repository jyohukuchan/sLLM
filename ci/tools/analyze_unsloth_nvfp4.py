#!/usr/bin/env python3
"""Independent source-identity and NVFP4 quality analysis for Phase 15Q."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np


BLOCK_SIZE = 16
FP4_MAX = np.float32(6.0)
FP8_MAX = np.float32(448.0)
E2M1 = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
SCALE_MULTIPLIERS = np.asarray([0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 2.0], dtype=np.float32)


class ContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_header(path: Path) -> tuple[int, dict[str, Any], bytes]:
    with path.open("rb") as source:
        raw = source.read(8)
        if len(raw) != 8:
            raise ContractError(f"truncated safetensors header: {path}")
        length = struct.unpack("<Q", raw)[0]
        if length == 0 or length > 256 * 1024 * 1024:
            raise ContractError(f"invalid safetensors header: {path}")
        header_bytes = source.read(length)
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise ContractError("safetensors header is not an object")
    return 8 + length, header, header_bytes


def catalog_hash(header: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(name for name in header if name != "__metadata__"):
        value = header[name]
        line = json.dumps(
            [name, value.get("dtype"), value.get("shape"), value.get("data_offsets")],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update(line.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def range_hash(path: Path, data_start: int, offsets: list[int]) -> str:
    digest = hashlib.sha256()
    remaining = offsets[1] - offsets[0]
    with path.open("rb") as source:
        source.seek(data_start + offsets[0])
        while remaining:
            chunk = source.read(min(remaining, 16 * 1024 * 1024))
            if not chunk:
                raise ContractError("tensor range is truncated")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def e4m3_table() -> np.ndarray:
    result = np.empty(127, dtype=np.float32)
    for bits in range(127):
        exponent, mantissa = (bits >> 3) & 0x0F, bits & 0x07
        result[bits] = np.float32(
            mantissa * 2.0**-9 if exponent == 0 else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        )
    return result


E4M3 = e4m3_table()


def nearest_even(values: np.ndarray, table: np.ndarray) -> np.ndarray:
    upper = np.minimum(np.searchsorted(table, values, side="left"), len(table) - 1)
    lower = np.maximum(upper - 1, 0)
    lower_error = np.abs(values - table[lower])
    upper_error = np.abs(table[upper] - values)
    choose_upper = (upper_error < lower_error) | (
        (upper_error == lower_error) & ((upper & 1) == 0) & ((lower & 1) != 0)
    )
    return np.where(choose_upper, upper, lower).astype(np.uint8)


def encode_e4m3(values: np.ndarray) -> np.ndarray:
    sign = np.signbit(values).astype(np.uint8) << np.uint8(7)
    return nearest_even(np.minimum(np.abs(values), FP8_MAX), E4M3) | sign


def decode_e4m3(bits: np.ndarray) -> np.ndarray:
    sign = np.where(bits & np.uint8(0x80), np.float32(-1.0), np.float32(1.0))
    return sign * E4M3[bits & np.uint8(0x7F)]


def encode_e2m1(values: np.ndarray) -> np.ndarray:
    sign = np.signbit(values).astype(np.uint8) << np.uint8(3)
    return nearest_even(np.minimum(np.abs(values), FP4_MAX), E2M1) | sign


def decode_e2m1(bits: np.ndarray) -> np.ndarray:
    sign = np.where(bits & np.uint8(0x08), np.float32(-1.0), np.float32(1.0))
    return sign * E2M1[bits & np.uint8(0x07)]


def bf16(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def tensor_memmap(path: Path, data_start: int, metadata: dict[str, Any], dtype: str) -> np.memmap:
    numpy_dtype = {"BF16": "<u2", "U8": "u1", "F8_E4M3": "u1", "F32": "<f4"}[dtype]
    return np.memmap(
        path,
        dtype=numpy_dtype,
        mode="r",
        offset=data_start + metadata["data_offsets"][0],
        shape=tuple(metadata["shape"]),
        order="C",
    )


def full_amax(words: np.memmap) -> float:
    flattened = words.reshape(-1)
    maximum = np.float32(0.0)
    for start in range(0, flattened.size, 16 * 1024 * 1024):
        values = bf16(flattened[start : start + 16 * 1024 * 1024])
        if not np.isfinite(values).all():
            raise ContractError("BF16 source contains a non-finite value")
        maximum = np.maximum(maximum, np.max(np.abs(values)))
    return float(maximum)


def quantize_sample(source: np.ndarray, tensor_scale: float) -> np.ndarray:
    block_amax = np.max(np.abs(source), axis=1).astype(np.float32)
    raw_scale = ((block_amax / FP4_MAX) / np.float32(tensor_scale)).astype(np.float32)
    scale_bits = encode_e4m3(raw_scale)
    scale = decode_e4m3(scale_bits) * np.float32(tensor_scale)
    normalized = np.zeros_like(source)
    np.divide(source, scale[:, None], out=normalized, where=scale[:, None] != 0)
    return decode_e2m1(encode_e2m1(normalized)) * scale[:, None]


def metrics(source: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    source64 = source.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    difference = candidate64 - source64
    signal = float(np.sum(source64 * source64))
    noise = float(np.sum(difference * difference))
    denominator = math.sqrt(signal * float(np.sum(candidate64 * candidate64)))
    saturation_rate = 0.0
    if source.size % BLOCK_SIZE == 0:
        source_blocks = np.abs(source.reshape(-1, BLOCK_SIZE))
        candidate_blocks = np.abs(candidate.reshape(-1, BLOCK_SIZE))
        block_peak = np.max(candidate_blocks, axis=1, keepdims=True)
        saturated = (block_peak > 0.0) & (candidate_blocks == block_peak) & (source_blocks > candidate_blocks)
        saturation_rate = float(np.mean(saturated))
    return {
        "samples": int(source.size),
        "mse": noise / source.size,
        "mae": float(np.mean(np.abs(difference))),
        "max_abs": float(np.max(np.abs(difference))),
        "cosine": float(np.sum(source64 * candidate64) / denominator) if denominator else 1.0,
        "sqnr_db": float(10.0 * math.log10(signal / noise)) if noise else float("inf"),
        "zero_rate": float(np.mean(candidate == 0.0)),
        "saturation_rate": saturation_rate,
    }


def compare_shared_bf16(
    bf16_path: Path,
    bf16_start: int,
    bf16_header: dict[str, Any],
    quantized_path: Path,
    quantized_start: int,
    quantized_header: dict[str, Any],
) -> dict[str, Any]:
    shared = sorted(
        name
        for name, metadata in quantized_header.items()
        if name != "__metadata__" and metadata.get("dtype") == "BF16" and name in bf16_header
    )
    mismatches: list[str] = []
    digest = hashlib.sha256()
    for ordinal, name in enumerate(shared, 1):
        q_meta, b_meta = quantized_header[name], bf16_header[name]
        if q_meta.get("shape") != b_meta.get("shape") or b_meta.get("dtype") != "BF16":
            mismatches.append(name)
            continue
        q_hash = range_hash(quantized_path, quantized_start, q_meta["data_offsets"])
        b_hash = range_hash(bf16_path, bf16_start, b_meta["data_offsets"])
        if q_hash != b_hash:
            mismatches.append(name)
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(q_hash.encode())
        if ordinal % 100 == 0:
            print(f"shared BF16 identity [{ordinal}/{len(shared)}]", file=sys.stderr)
    return {
        "shared_bf16_tensors": len(shared),
        "byte_identical_tensors": len(shared) - len(mismatches),
        "mismatches": mismatches,
        "identity_digest": digest.hexdigest(),
    }


def analyze_tensor(
    name: str,
    bf16_path: Path,
    bf16_start: int,
    bf16_header: dict[str, Any],
    quantized_path: Path,
    quantized_start: int,
    quantized_header: dict[str, Any],
    sample_blocks: int,
) -> dict[str, Any]:
    source_meta = bf16_header[name]
    rows, columns = source_meta["shape"]
    prefix = name.removesuffix(".weight")
    packed_meta = quantized_header[prefix + ".weight_packed"]
    scale_meta = quantized_header[prefix + ".weight_scale"]
    global_meta = quantized_header[prefix + ".weight_global_scale"]
    if (
        source_meta.get("dtype") != "BF16"
        or packed_meta != {
            "dtype": "U8",
            "shape": [rows, columns // 2],
            "data_offsets": packed_meta.get("data_offsets"),
        }
        or scale_meta.get("dtype") != "F8_E4M3"
        or scale_meta.get("shape") != [rows, math.ceil(columns / BLOCK_SIZE)]
        or global_meta.get("dtype") != "F32"
        or global_meta.get("shape") != [1]
    ):
        raise ContractError(f"quantized tensor metadata differs: {name}")
    source_words = tensor_memmap(bf16_path, bf16_start, source_meta, "BF16")
    packed = tensor_memmap(quantized_path, quantized_start, packed_meta, "U8")
    scales = tensor_memmap(quantized_path, quantized_start, scale_meta, "F8_E4M3")
    global_scale = float(tensor_memmap(quantized_path, quantized_start, global_meta, "F32")[0])
    if not math.isfinite(global_scale) or global_scale <= 0.0:
        raise ContractError(f"invalid global scale: {name}")
    blocks_per_row = math.ceil(columns / BLOCK_SIZE)
    total_blocks = rows * blocks_per_row
    count = min(sample_blocks, total_blocks)
    block_indices = np.unique(np.linspace(0, total_blocks - 1, count, dtype=np.int64))
    sample_rows = block_indices // blocks_per_row
    sample_blocks_in_row = block_indices % blocks_per_row
    offsets = np.arange(BLOCK_SIZE, dtype=np.int64)
    columns_2d = sample_blocks_in_row[:, None] * BLOCK_SIZE + offsets[None, :]
    valid = columns_2d < columns
    clipped_columns = np.minimum(columns_2d, columns - 1)
    sampled_source = bf16(source_words[sample_rows[:, None], clipped_columns])
    sampled_source = np.where(valid, sampled_source, np.float32(0.0))
    packed_bytes = packed[sample_rows[:, None], clipped_columns // 2]
    codes = np.where(clipped_columns & 1, packed_bytes >> np.uint8(4), packed_bytes & np.uint8(0x0F))
    u0 = decode_e2m1(codes) * decode_e4m3(scales[sample_rows, sample_blocks_in_row])[:, None] / np.float32(global_scale)
    u0 = np.where(valid, u0, np.float32(0.0))

    maximum = full_amax(source_words)
    s0_tensor_scale = 1.0 if maximum == 0.0 else maximum / float(FP8_MAX * FP4_MAX)
    s0 = quantize_sample(sampled_source, s0_tensor_scale)
    choices: list[tuple[float, float, np.ndarray]] = []
    for multiplier in SCALE_MULTIPLIERS:
        candidate = quantize_sample(sampled_source, s0_tensor_scale * float(multiplier))
        mse = float(np.mean((candidate.astype(np.float64) - sampled_source.astype(np.float64)) ** 2))
        choices.append((mse, float(multiplier), candidate))
    _, best_multiplier, o0 = min(choices, key=lambda item: (item[0], abs(item[1] - 1.0)))
    return {
        "name": name,
        "shape": [rows, columns],
        "source_sha256": range_hash(bf16_path, bf16_start, source_meta["data_offsets"]),
        "sample_blocks": int(block_indices.size),
        "sample_elements": int(np.count_nonzero(valid)),
        "source_amax": maximum,
        "s0_tensor_scale": s0_tensor_scale,
        "u0_reciprocal_global_scale": global_scale,
        "o0_tensor_scale_multiplier": best_multiplier,
        "s0": metrics(sampled_source[valid], s0[valid]),
        "u0": metrics(sampled_source[valid], u0[valid]),
        "o0": metrics(sampled_source[valid], o0[valid]),
    }


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def summarize(records: list[dict[str, Any]], variant: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in ("mse", "mae", "max_abs", "cosine", "sqnr_db", "zero_rate", "saturation_rate"):
        values = [float(record[variant][metric]) for record in records]
        result[f"{metric}_median"] = percentile(values, 50)
        result[f"{metric}_p90"] = percentile(values, 90)
        result[f"{metric}_max"] = max(values)
    return result


def run(args: argparse.Namespace) -> None:
    bf16_path = args.bf16_cache / "model.safetensors"
    quantized_path = args.quantized_cache / "model.safetensors"
    bf16_sha = sha256_file(bf16_path)
    quantized_sha = sha256_file(quantized_path)
    if bf16_sha != args.expected_bf16_sha256 or quantized_sha != args.expected_quantized_sha256:
        raise ContractError("artifact SHA-256 differs")
    bf16_start, bf16_header, bf16_header_bytes = read_header(bf16_path)
    quantized_start, quantized_header, quantized_header_bytes = read_header(quantized_path)
    shared = compare_shared_bf16(
        bf16_path,
        bf16_start,
        bf16_header,
        quantized_path,
        quantized_start,
        quantized_header,
    )
    if shared["mismatches"]:
        raise ContractError("shared BF16 tensors differ")
    names = [
        f"model.language_model.layers.{layer}.mlp.{projection}_proj.weight"
        for layer in range(48)
        for projection in ("down", "gate", "up")
    ]
    records = []
    for ordinal, name in enumerate(names, 1):
        records.append(
            analyze_tensor(
                name,
                bf16_path,
                bf16_start,
                bf16_header,
                quantized_path,
                quantized_start,
                quantized_header,
                args.sample_blocks,
            )
        )
        print(f"NVFP4 quality [{ordinal}/{len(names)}] {name}", file=sys.stderr)
    report = {
        "schema_version": "phase15q-unsloth-nvfp4-analysis-v1",
        "state": "PASS",
        "artifacts": {
            "bf16": {
                "sha256": bf16_sha,
                "size_bytes": bf16_path.stat().st_size,
                "header_sha256": hashlib.sha256(bf16_header_bytes).hexdigest(),
                "catalog_sha256": catalog_hash(bf16_header),
                "tensor_count": len(bf16_header) - int("__metadata__" in bf16_header),
            },
            "quantized": {
                "sha256": quantized_sha,
                "size_bytes": quantized_path.stat().st_size,
                "header_sha256": hashlib.sha256(quantized_header_bytes).hexdigest(),
                "catalog_sha256": catalog_hash(quantized_header),
                "tensor_count": len(quantized_header) - int("__metadata__" in quantized_header),
            },
        },
        "source_identity": shared,
        "sampling": {
            "method": "deterministic-evenly-spaced-k-axis-blocks",
            "blocks_per_tensor_cap": args.sample_blocks,
            "evaluation_set_used_for_tuning": False,
        },
        "summary": {variant: summarize(records, variant) for variant in ("s0", "u0", "o0")},
        "tensors": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(f"Phase 15Q analysis: PASS tensors={len(records)} output={args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-cache", required=True, type=Path)
    parser.add_argument("--quantized-cache", required=True, type=Path)
    parser.add_argument("--expected-bf16-sha256", required=True)
    parser.add_argument("--expected-quantized-sha256", required=True)
    parser.add_argument("--sample-blocks", type=int, default=4096)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.sample_blocks < 17 or args.sample_blocks > 1_000_000:
        parser.error("--sample-blocks must be in [17,1000000]")
    return args


def main() -> int:
    try:
        run(parse_args())
        return 0
    except (ContractError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Phase 15Q analysis: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
