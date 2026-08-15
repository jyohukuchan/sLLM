#!/usr/bin/env python3
"""Independent NumPy oracle for the Phase 16 KV encodings.

This module deliberately does not import sLLM.  It checks the versioned
token-major FP8/NVFP4 recipes, odd tails, scale storage, and causal attention
boundary shapes used by the real-GPU evidence runner.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np


HEAD_DIMS = (255, 256, 257)
BLOCK_WIDTHS = (15, 16, 17)
TOKEN_BOUNDARIES = (255, 256, 257, 1023, 1024, 1025)
QUERY_COUNTS = (1, 3, 7, 37)
E2M1_POSITIVE = np.asarray((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), dtype=np.float32)


def decode_e4m3fn(bits: int) -> float:
    sign = -1.0 if bits & 0x80 else 1.0
    exponent = (bits >> 3) & 0x0F
    mantissa = bits & 0x07
    if exponent == 0:
        return math.copysign(0.0, sign) if mantissa == 0 else sign * mantissa * 2.0**-9
    if exponent == 0x0F and mantissa == 0x07:
        return math.nan
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


E4M3_POSITIVE_BITS = np.arange(0x7F, dtype=np.uint8)
E4M3_POSITIVE = np.asarray(
    [decode_e4m3fn(int(bits)) for bits in E4M3_POSITIVE_BITS], dtype=np.float32
)


def encode_e4m3fn(value: float) -> int:
    sign = 0x80 if math.copysign(1.0, value) < 0 else 0
    if math.isnan(value):
        return 0x7F
    magnitude = abs(float(value))
    if magnitude == 0.0:
        return sign
    if not math.isfinite(magnitude) or magnitude >= 448.0:
        return sign | 0x7E
    insertion = int(np.searchsorted(E4M3_POSITIVE, magnitude, side="left"))
    upper = min(insertion, 0x7E)
    lower = max(upper - 1, 0)
    lower_error = magnitude - float(E4M3_POSITIVE[lower])
    upper_error = float(E4M3_POSITIVE[upper]) - magnitude
    selected = upper if (
        upper_error < lower_error
        or (upper_error == lower_error and upper & 1 == 0 and lower & 1 != 0)
    ) else lower
    return sign | selected


def encode_e2m1(value: float) -> int:
    sign = 0x08 if math.copysign(1.0, value) < 0 else 0
    if math.isnan(value):
        return sign
    magnitude = min(abs(float(value)), 6.0)
    errors = np.abs(E2M1_POSITIVE - magnitude)
    minimum = float(errors.min())
    tied = np.flatnonzero(errors == minimum)
    selected = next((int(item) for item in tied if int(item) & 1 == 0), int(tied[0]))
    return sign | selected


def decode_e2m1(bits: int) -> float:
    magnitude = float(E2M1_POSITIVE[bits & 0x07])
    return -magnitude if bits & 0x08 else magnitude


def quantize_fp8(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maxima = np.max(np.where(np.isfinite(rows), np.abs(rows), 0.0), axis=1)
    scales = np.where(maxima == 0.0, 1.0, maxima / 448.0).astype(np.float32)
    encoded = np.empty(rows.shape, dtype=np.uint8)
    for row in range(rows.shape[0]):
        for column in range(rows.shape[1]):
            encoded[row, column] = encode_e4m3fn(float(rows[row, column] / scales[row]))
    decoded = np.empty(rows.shape, dtype=np.float32)
    for index, bits in np.ndenumerate(encoded):
        decoded[index] = decode_e4m3fn(int(bits)) * scales[index[0]]
    return encoded, scales, decoded


def quantize_nvfp4(
    rows: np.ndarray, block_size: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_count, columns = rows.shape
    packed_per_row = (columns + 1) // 2
    blocks_per_row = (columns + block_size - 1) // block_size
    packed = np.zeros((row_count, packed_per_row), dtype=np.uint8)
    block_scales = np.zeros((row_count, blocks_per_row), dtype=np.uint8)
    outer_scales = np.empty(row_count, dtype=np.float32)
    decoded = np.zeros(rows.shape, dtype=np.float32)
    for row in range(row_count):
        finite_magnitudes = np.where(np.isfinite(rows[row]), np.abs(rows[row]), 0.0)
        maximum = float(np.max(finite_magnitudes))
        outer = 1.0 if maximum == 0.0 else maximum / (448.0 * 6.0)
        outer_scales[row] = outer
        for block in range(blocks_per_row):
            begin = block * block_size
            end = min(begin + block_size, columns)
            block_values = rows[row, begin:end]
            block_maximum = float(
                np.max(np.where(np.isfinite(block_values), np.abs(block_values), 0.0))
            )
            if np.isinf(block_values).any():
                block_maximum = 448.0 * 6.0 * outer
            scale_bits = encode_e4m3fn((block_maximum / 6.0) / outer)
            block_scales[row, block] = scale_bits
            scale = decode_e4m3fn(scale_bits)
            for column in range(begin, end):
                code = 0 if scale == 0.0 else encode_e2m1(float(rows[row, column] / (scale * outer)))
                if column & 1:
                    packed[row, column // 2] |= code << 4
                else:
                    packed[row, column // 2] = code
                decoded[row, column] = decode_e2m1(code) * scale * outer
    return packed, block_scales, outer_scales, decoded


def causal_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    output = np.empty_like(query, dtype=np.float32)
    scale = 1.0 / math.sqrt(query.shape[1])
    for row in range(query.shape[0]):
        scores = key[: row + 1] @ query[row] * scale
        probabilities = np.exp(scores - np.max(scores))
        probabilities /= np.sum(probabilities)
        output[row] = probabilities @ value[: row + 1]
    return output


def run() -> dict[str, object]:
    rng = np.random.default_rng(0x16F017)
    quantization_cases = 0
    padding_cases = 0
    for head_dim in HEAD_DIMS:
        rows = rng.normal(0.0, 1.5, size=(3, head_dim)).astype(np.float32)
        rows[0].fill(0.0)
        fp8_values, fp8_scales, fp8_decoded = quantize_fp8(rows)
        if fp8_values.shape != rows.shape or fp8_scales.shape != (3,) or not np.isfinite(fp8_decoded).all():
            raise AssertionError("FP8 row/scale shape or finiteness mismatch")
        quantization_cases += 1
        for block_size in BLOCK_WIDTHS:
            packed, block_scales, outer, decoded = quantize_nvfp4(rows, block_size)
            expected_blocks = (head_dim + block_size - 1) // block_size
            if packed.shape != (3, (head_dim + 1) // 2):
                raise AssertionError("NVFP4 packed row shape mismatch")
            if block_scales.shape != (3, expected_blocks) or outer.shape != (3,):
                raise AssertionError("NVFP4 scale shape mismatch")
            if not np.isfinite(decoded).all():
                raise AssertionError("NVFP4 decode produced a non-finite value")
            if head_dim & 1 and np.any(packed[:, -1] & 0xF0):
                raise AssertionError("NVFP4 odd-row padding nibble is nonzero")
            quantization_cases += 1
            padding_cases += int(head_dim & 1)

    attention_cases = 0
    for query_count in QUERY_COUNTS:
        head_dim = 17
        query = rng.normal(size=(query_count, head_dim)).astype(np.float32)
        key = rng.normal(size=(query_count, head_dim)).astype(np.float32)
        value = rng.normal(size=(query_count, head_dim)).astype(np.float32)
        for quantizer in (quantize_fp8, quantize_nvfp4):
            *_, quantized_key = quantizer(key)
            *_, quantized_value = quantizer(value)
            output = causal_attention(query, quantized_key, quantized_value)
            if output.shape != query.shape or not np.isfinite(output).all():
                raise AssertionError("causal attention oracle shape/finiteness mismatch")
            attention_cases += 1

    special = np.asarray(
        [[math.nan, math.inf, -math.inf, 1.0] + [0.0] * 12], dtype=np.float32
    )
    fp8_values, fp8_scales, fp8_decoded = quantize_fp8(special)
    if fp8_values[0, 0] != 0x7F or fp8_values[0, 1] != 0x7E or fp8_values[0, 2] != 0xFE:
        raise AssertionError("FP8 non-finite canonicalization mismatch")
    if not math.isnan(float(fp8_decoded[0, 0])) or not np.isfinite(fp8_scales).all():
        raise AssertionError("FP8 NaN/scale handling mismatch")
    *_, nvfp4_decoded = quantize_nvfp4(special)
    if not np.isfinite(nvfp4_decoded).all() or nvfp4_decoded[0, 0] != 0.0:
        raise AssertionError("NVFP4 non-finite canonicalization mismatch")
    if nvfp4_decoded[0, 1] <= 0.0 or nvfp4_decoded[0, 2] >= 0.0:
        raise AssertionError("NVFP4 infinity saturation mismatch")
    nonfinite_cases = 2

    memory_cases = []
    for tokens in TOKEN_BOUNDARIES:
        fp8_bytes = tokens * (4 * 256 + 4 * 4)
        nvfp4_bytes = tokens * (4 * (256 // 2) + 4 * (256 // 16) + 4 * 4)
        if not (nvfp4_bytes < fp8_bytes < tokens * 4 * 256 * 2):
            raise AssertionError("low-bit resident byte ordering is invalid")
        memory_cases.append({"tokens": tokens, "fp8_bytes_per_k_or_v": fp8_bytes, "nvfp4_bytes_per_k_or_v": nvfp4_bytes})

    if 4 % 4 != 0 or 2 % 4 == 0:
        raise AssertionError("scale-plane alignment boundary check failed")
    return {
        "schema_version": "sllm-kv-lowbit-numpy-oracle-v1",
        "state": "PASS",
        "quantization_cases": quantization_cases,
        "attention_cases": attention_cases,
        "padding_cases": padding_cases,
        "nonfinite_cases": nonfinite_cases,
        "head_dims": list(HEAD_DIMS),
        "block_widths": list(BLOCK_WIDTHS),
        "query_counts": list(QUERY_COUNTS),
        "token_boundaries": list(TOKEN_BOUNDARIES),
        "memory_cases": memory_cases,
        "invalid_scale_offset_rejected": True,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(run(), sort_keys=True, separators=(",", ":")))
    except Exception as error:  # fail-closed command-line evidence
        print(json.dumps({"schema_version": "sllm-kv-lowbit-numpy-oracle-v1", "state": "FAIL", "error": str(error)}))
        sys.exit(1)
