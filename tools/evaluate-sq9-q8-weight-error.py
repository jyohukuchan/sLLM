#!/usr/bin/env python3
"""CPU-only, bounded-memory SQ9_0 versus Q8_0 weight-error evaluation.

The source checkpoint is Qwen/Qwen3-14B-FP8.  Each selected F8_E4M3 payload
is reconstructed exactly according to its BF16 ``weight_scale_inv`` 128x128
block multiplier, then independently requantized to:

* ``SQ9_0``: signed E5M3, no reconstruction scale, RNE;
* ``Q8_0_g32_f16``: signed int8 plus one FP16 symmetric scale per 32 values;
* ``Q8_0_g128_f16``: the same codebook with a 128-value scale ablation.

This script never initializes HIP, CUDA, or a model runtime.  It reads source
bytes through read-only mappings and processes a bounded row chunk at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_VERSION = "2026-07-26.sq9-q8-error.v1"
SOURCE_BLOCK_ROWS = 128
SOURCE_BLOCK_COLS = 128
Q8_GROUP_SIZES = (32, 128)
E5M3_MIN_NORMAL = np.float32(2.0**-14)
E5M3_SUBNORMAL_UNIT = np.float32(2.0**-17)
E5M3_MAX_FINITE = np.float32(61440.0)
SPREAD_BINS: tuple[tuple[str, float, float], ...] = (
    ("[1,2)", 1.0, 2.0),
    ("[2,4)", 2.0, 4.0),
    ("[4,8)", 4.0, 8.0),
    ("[8,16)", 8.0, 16.0),
    ("[16,inf)", 16.0, math.inf),
)


class EvaluationError(RuntimeError):
    """Raised for a malformed source checkpoint or invalid evaluation input."""


@dataclass(frozen=True)
class TensorRegion:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offset: int
    data_length: int


@dataclass(frozen=True)
class WeightScalePair:
    name: str
    weight_path: Path
    weight: TensorRegion
    scale_path: Path
    scale: TensorRegion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-model-dir",
        type=Path,
        default=Path("/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8"),
        help="read-only Qwen3-14B-FP8 safetensors directory",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--row-chunk",
        type=int,
        default=128,
        help="maximum source rows held per tensor chunk (default: 128)",
    )
    parser.add_argument(
        "--tensor-regex",
        action="append",
        default=[],
        help="regular expression; repeat to OR-select names (default: all FP8 pairs)",
    )
    parser.add_argument(
        "--max-tensors",
        type=int,
        default=None,
        help="evaluate at most this many lexically ordered FP8 tensors",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of files in an existing output directory",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run deterministic quantizer checks without reading a model",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def parse_safetensors_header(path: Path) -> tuple[int, dict[str, TensorRegion]]:
    """Return data-base offset and validated tensor regions for one shard."""

    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_header_length = handle.read(8)
        if len(raw_header_length) != 8:
            raise EvaluationError(f"missing safetensors header length: {path}")
        header_length = int.from_bytes(raw_header_length, "little", signed=False)
        if header_length <= 0 or header_length > 64 * 1024 * 1024:
            raise EvaluationError(f"invalid safetensors header length {header_length}: {path}")
        raw_header = handle.read(header_length)
    if len(raw_header) != header_length:
        raise EvaluationError(f"truncated safetensors header: {path}")
    try:
        decoded = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"invalid safetensors header JSON: {path}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise EvaluationError(f"safetensors header root is not an object: {path}")

    data_base = 8 + header_length
    data_bytes = file_size - data_base
    regions: dict[str, TensorRegion] = {}
    for name, raw in decoded.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise EvaluationError(f"invalid tensor header entry: {path}:{name!r}")
        dtype = raw.get("dtype")
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list):
            raise EvaluationError(f"incomplete tensor header entry: {path}:{name}")
        if not all(_is_nonnegative_int(value) for value in shape):
            raise EvaluationError(f"invalid tensor shape: {path}:{name}")
        if len(offsets) != 2 or not all(_is_nonnegative_int(value) for value in offsets):
            raise EvaluationError(f"invalid tensor offsets: {path}:{name}")
        start, end = offsets
        if end < start or end > data_bytes:
            raise EvaluationError(f"out-of-range tensor offsets: {path}:{name}")
        regions[name] = TensorRegion(
            name=name,
            dtype=dtype,
            shape=tuple(shape),
            data_offset=data_base + start,
            data_length=end - start,
        )
    return data_base, regions


def read_index(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read source index {index_path}: {exc}") from exc
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(name, str) and isinstance(shard, str) for name, shard in weight_map.items()
    ):
        raise EvaluationError(f"invalid source weight_map: {index_path}")
    return dict(weight_map)


def discover_pairs(model_dir: Path) -> list[WeightScalePair]:
    weight_map = read_index(model_dir)
    shard_names = sorted(set(weight_map.values()))
    headers: dict[str, tuple[Path, dict[str, TensorRegion]]] = {}
    for shard_name in shard_names:
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            raise EvaluationError(f"indexed source shard is missing: {shard_path}")
        _, regions = parse_safetensors_header(shard_path)
        headers[shard_name] = (shard_path, regions)

    pairs: list[WeightScalePair] = []
    for name in sorted(weight_map):
        if not name.endswith(".weight"):
            continue
        scale_name = f"{name}_scale_inv"
        if scale_name not in weight_map:
            continue
        weight_shard = weight_map[name]
        scale_shard = weight_map[scale_name]
        weight_path, weight_regions = headers[weight_shard]
        scale_path, scale_regions = headers[scale_shard]
        weight = weight_regions.get(name)
        scale = scale_regions.get(scale_name)
        if weight is None or scale is None:
            raise EvaluationError(f"index/header mismatch for pair {name}")
        if weight.dtype != "F8_E4M3" or len(weight.shape) != 2:
            continue
        if scale.dtype != "BF16" or len(scale.shape) != 2:
            raise EvaluationError(f"invalid source scale dtype/shape for {name}")
        rows, cols = weight.shape
        expected_weight_bytes = rows * cols
        expected_scale_shape = (
            math.ceil(rows / SOURCE_BLOCK_ROWS),
            math.ceil(cols / SOURCE_BLOCK_COLS),
        )
        if weight.data_length != expected_weight_bytes:
            raise EvaluationError(
                f"unexpected F8 byte length for {name}: {weight.data_length} != {expected_weight_bytes}"
            )
        if scale.shape != expected_scale_shape or scale.data_length != math.prod(scale.shape) * 2:
            raise EvaluationError(
                f"invalid 128x128 BF16 scale geometry for {name}: "
                f"shape={scale.shape}, expected={expected_scale_shape}"
            )
        pairs.append(WeightScalePair(name, weight_path, weight, scale_path, scale))
    if not pairs:
        raise EvaluationError("source model has no paired F8_E4M3/BF16 2-D weights")
    return pairs


def e4m3fn_lookup() -> np.ndarray:
    result = np.empty(256, dtype=np.float32)
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        exponent = (code >> 3) & 0x0F
        mantissa = code & 0x07
        if exponent == 0:
            result[code] = np.float32(sign * mantissa * (2.0**-9))
        elif exponent == 0x0F and mantissa == 0x07:
            result[code] = np.float32(np.nan)
        else:
            result[code] = np.float32(sign * (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7)))
    return result


E4M3FN_LOOKUP = e4m3fn_lookup()


def bf16_bytes_to_f32(raw: memoryview) -> np.ndarray:
    bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (bits << np.uint32(16)).view("<f4")


def sum_squares(values: np.ndarray) -> float:
    flat = values.reshape(-1)
    return float(np.dot(flat.astype(np.float64), flat.astype(np.float64)))


def format_number(value: float) -> float:
    """Normalize NumPy scalar values before JSON serialization."""

    return float(value)


def new_metric_state() -> dict[str, Any]:
    return {
        "elements": 0,
        "reference_sse": 0.0,
        "error_sse": 0.0,
        "abs_error_sum": 0.0,
        "max_abs_error": -1.0,
        "max_abs_error_location": None,
        "max_relative_error": -1.0,
        "max_relative_error_location": None,
    }


def update_metric_state(
    state: dict[str, Any],
    reference: np.ndarray,
    reconstructed: np.ndarray,
    *,
    tensor_name: str,
    row_start: int,
) -> np.ndarray:
    error = reconstructed - reference
    abs_error = np.abs(error)
    state["elements"] += int(reference.size)
    state["reference_sse"] += sum_squares(reference)
    state["error_sse"] += sum_squares(error)
    state["abs_error_sum"] += float(np.sum(abs_error, dtype=np.float64))

    flat_index = int(np.argmax(abs_error))
    local_max = float(abs_error.reshape(-1)[flat_index])
    if local_max > state["max_abs_error"]:
        cols = reference.shape[1]
        row = flat_index // cols
        col = flat_index % cols
        state["max_abs_error"] = local_max
        state["max_abs_error_location"] = {
            "tensor": tensor_name,
            "row": row_start + row,
            "col": col,
            "reference": float(reference[row, col]),
            "reconstructed": float(reconstructed[row, col]),
            "signed_error": float(error[row, col]),
        }

    nonzero = np.abs(reference) > 0.0
    if bool(np.any(nonzero)):
        relative = np.zeros_like(abs_error)
        np.divide(abs_error, np.abs(reference), out=relative, where=nonzero)
        relative_index = int(np.argmax(relative))
        local_relative = float(relative.reshape(-1)[relative_index])
        if local_relative > state["max_relative_error"]:
            cols = reference.shape[1]
            row = relative_index // cols
            col = relative_index % cols
            state["max_relative_error"] = local_relative
            state["max_relative_error_location"] = {
                "tensor": tensor_name,
                "row": row_start + row,
                "col": col,
                "reference": float(reference[row, col]),
                "reconstructed": float(reconstructed[row, col]),
                "signed_error": float(error[row, col]),
            }
    return error


def finalize_metric_state(state: dict[str, Any]) -> dict[str, Any]:
    result = dict(state)
    reference_sse = result["reference_sse"]
    error_sse = result["error_sse"]
    result["relative_l2"] = math.sqrt(error_sse / reference_sse) if reference_sse else 0.0
    result["relative_mse"] = error_sse / reference_sse if reference_sse else 0.0
    result["mean_abs_error"] = result["abs_error_sum"] / result["elements"]
    return result


def add_metric_states(destination: dict[str, Any], source: dict[str, Any]) -> None:
    destination["elements"] += source["elements"]
    destination["reference_sse"] += source["reference_sse"]
    destination["error_sse"] += source["error_sse"]
    destination["abs_error_sum"] += source["abs_error_sum"]
    if source["max_abs_error"] > destination["max_abs_error"]:
        destination["max_abs_error"] = source["max_abs_error"]
        destination["max_abs_error_location"] = source["max_abs_error_location"]
    if source["max_relative_error"] > destination["max_relative_error"]:
        destination["max_relative_error"] = source["max_relative_error"]
        destination["max_relative_error_location"] = source["max_relative_error_location"]


def quantize_sq9_0(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """RNE quantize finite F32 values to the exact finite SQ9_0 E5M3 lattice."""

    if not bool(np.all(np.isfinite(values))):
        raise EvaluationError("SQ9_0 input contains non-finite reconstructed source values")
    absolute = np.abs(values)
    saturated = absolute > E5M3_MAX_FINITE
    clamped = np.minimum(absolute, E5M3_MAX_FINITE)
    reconstructed = np.zeros_like(values, dtype=np.float32)

    normal = clamped >= E5M3_MIN_NORMAL
    if bool(np.any(normal)):
        normal_values = clamped[normal]
        exponents = np.floor(np.log2(normal_values)).astype(np.int32)
        significands = np.rint(np.ldexp(normal_values, 3 - exponents)).astype(np.int32)
        carry = significands == 16
        significands[carry] = 8
        exponents[carry] += 1
        over = exponents > 15
        exponents[over] = 15
        significands[over] = 15
        reconstructed[normal] = np.ldexp(
            significands.astype(np.float32) * np.float32(0.125), exponents
        )

    subnormal = (clamped > 0.0) & ~normal
    if bool(np.any(subnormal)):
        subnormal_mantissas = np.rint(clamped[subnormal] / E5M3_SUBNORMAL_UNIT)
        subnormal_mantissas = np.clip(subnormal_mantissas, 0.0, 7.0).astype(np.float32)
        reconstructed[subnormal] = subnormal_mantissas * E5M3_SUBNORMAL_UNIT

    reconstructed = np.copysign(reconstructed, values)
    telemetry = {
        "input_nonzero": int(np.count_nonzero(values)),
        "input_zero": int(values.size - np.count_nonzero(values)),
        "saturated_to_finite_max": int(np.count_nonzero(saturated)),
        "normal_codes": int(np.count_nonzero(normal)),
        "subnormal_codes": int(np.count_nonzero((reconstructed != 0.0) & ~normal)),
        "rounded_to_zero": int(np.count_nonzero((values != 0.0) & (reconstructed == 0.0))),
    }
    return reconstructed, telemetry


def quantize_q8_0(values: np.ndarray, group_size: int) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """Quantize signed weights with Q8_0-style symmetric FP16 block scales."""

    if values.ndim != 2:
        raise EvaluationError("Q8_0 input must be a rank-two row chunk")
    rows, cols = values.shape
    groups = math.ceil(cols / group_size)
    padded_cols = groups * group_size
    if padded_cols == cols:
        grouped = values.reshape(rows, groups, group_size)
    else:
        padded = np.zeros((rows, padded_cols), dtype=np.float32)
        padded[:, :cols] = values
        grouped = padded.reshape(rows, groups, group_size)
    maxima = np.max(np.abs(grouped), axis=2)
    exact_scales = maxima / np.float32(127.0)
    fp16_scales = exact_scales.astype(np.float16)
    scales = fp16_scales.astype(np.float32)
    normalized = np.zeros_like(grouped, dtype=np.float32)
    np.divide(grouped, scales[:, :, None], out=normalized, where=scales[:, :, None] != 0.0)
    rounded = np.rint(normalized)
    clipped = np.clip(rounded, -127.0, 127.0)
    codes = clipped.astype(np.int8)
    reconstructed_grouped = codes.astype(np.float32) * scales[:, :, None]
    reconstructed = reconstructed_grouped.reshape(rows, padded_cols)[:, :cols]
    telemetry = {
        "group_size": group_size,
        "groups": int(rows * groups),
        "nonzero_groups": int(np.count_nonzero(maxima)),
        "scale_fp16_underflow": int(np.count_nonzero((exact_scales > 0.0) & (fp16_scales == 0.0))),
        "scale_fp16_overflow": int(np.count_nonzero(~np.isfinite(fp16_scales))),
        "clipped_values": int(np.count_nonzero(np.abs(rounded) > 127.0)),
        "codes_at_127": int(np.count_nonzero(np.abs(codes) == 127)),
    }
    return reconstructed, telemetry, grouped


def new_spread_state() -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "groups": 0,
            "elements": 0,
            "reference_sse": 0.0,
            "sq9_0_error_sse": 0.0,
            "q8_0_g32_f16_error_sse": 0.0,
        }
        for name, _, _ in SPREAD_BINS
    }


def add_spread_statistics(
    state: dict[str, dict[str, float | int]],
    values: np.ndarray,
    sq9_error: np.ndarray,
    q8_error: np.ndarray,
) -> None:
    rows, cols = values.shape
    group_size = 32
    groups = math.ceil(cols / group_size)
    padded_cols = groups * group_size
    if padded_cols == cols:
        grouped_values = values.reshape(rows, groups, group_size)
        grouped_sq9_error = sq9_error.reshape(rows, groups, group_size)
        grouped_q8_error = q8_error.reshape(rows, groups, group_size)
        valid = np.ones((rows, groups, group_size), dtype=bool)
    else:
        grouped_values = np.zeros((rows, padded_cols), dtype=np.float32)
        grouped_sq9_error = np.zeros((rows, padded_cols), dtype=np.float32)
        grouped_q8_error = np.zeros((rows, padded_cols), dtype=np.float32)
        grouped_values[:, :cols] = values
        grouped_sq9_error[:, :cols] = sq9_error
        grouped_q8_error[:, :cols] = q8_error
        grouped_values = grouped_values.reshape(rows, groups, group_size)
        grouped_sq9_error = grouped_sq9_error.reshape(rows, groups, group_size)
        grouped_q8_error = grouped_q8_error.reshape(rows, groups, group_size)
        valid = np.zeros((rows, padded_cols), dtype=bool)
        valid[:, :cols] = True
        valid = valid.reshape(rows, groups, group_size)

    maximum = np.max(np.abs(grouped_values), axis=2)
    rms = np.sqrt(np.mean(grouped_values * grouped_values, axis=2))
    spread = np.divide(maximum, rms, out=np.ones_like(maximum), where=rms != 0.0)
    reference_sse = np.sum(grouped_values * grouped_values, axis=2, dtype=np.float64)
    sq9_sse = np.sum(grouped_sq9_error * grouped_sq9_error, axis=2, dtype=np.float64)
    q8_sse = np.sum(grouped_q8_error * grouped_q8_error, axis=2, dtype=np.float64)
    valid_elements = np.sum(valid, axis=2, dtype=np.int64)
    for name, lower, upper in SPREAD_BINS:
        mask = spread >= lower
        if math.isfinite(upper):
            mask &= spread < upper
        bucket = state[name]
        bucket["groups"] += int(np.count_nonzero(mask))
        bucket["elements"] += int(np.sum(valid_elements[mask], dtype=np.int64))
        bucket["reference_sse"] += float(np.sum(reference_sse[mask], dtype=np.float64))
        bucket["sq9_0_error_sse"] += float(np.sum(sq9_sse[mask], dtype=np.float64))
        bucket["q8_0_g32_f16_error_sse"] += float(np.sum(q8_sse[mask], dtype=np.float64))


def merge_integer_telemetry(destination: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        if key == "group_size":
            previous = destination.get(key)
            if previous is not None and previous != int(value):
                raise EvaluationError(
                    f"inconsistent Q8_0 group-size telemetry: {previous} versus {value}"
                )
            destination[key] = int(value)
        else:
            destination[key] = destination.get(key, 0) + int(value)


def source_values(
    weight_map: mmap.mmap,
    weight: TensorRegion,
    row_start: int,
    row_stop: int,
    scales: np.ndarray,
) -> tuple[np.ndarray, int]:
    rows, cols = weight.shape
    if not (0 <= row_start <= row_stop <= rows):
        raise EvaluationError("invalid source row chunk")
    byte_start = weight.data_offset + row_start * cols
    raw = np.frombuffer(weight_map, dtype=np.uint8, count=(row_stop - row_start) * cols, offset=byte_start)
    nonfinite_codes = int(np.count_nonzero(~np.isfinite(E4M3FN_LOOKUP[raw])))
    if nonfinite_codes:
        raise EvaluationError(
            f"source tensor {weight.name} has {nonfinite_codes} non-finite E4M3FN codes "
            f"in rows [{row_start}, {row_stop})"
        )
    decoded = E4M3FN_LOOKUP[raw].reshape(row_stop - row_start, cols)
    row_blocks = np.arange(row_start, row_stop, dtype=np.int64) // SOURCE_BLOCK_ROWS
    expanded_columns = np.repeat(scales, SOURCE_BLOCK_COLS, axis=1)[:, :cols]
    values = decoded * expanded_columns[row_blocks]
    if not bool(np.all(np.isfinite(values))):
        raise EvaluationError(f"source reconstruction has non-finite values: {weight.name}")
    return values.astype(np.float32, copy=False), nonfinite_codes


def load_scale_values(scale_map: mmap.mmap, region: TensorRegion) -> np.ndarray:
    raw = memoryview(scale_map)[region.data_offset : region.data_offset + region.data_length]
    values = bf16_bytes_to_f32(raw).reshape(region.shape)
    if not bool(np.all(np.isfinite(values))) or not bool(np.all(values > 0.0)):
        raise EvaluationError(f"source scale has non-positive or non-finite values: {region.name}")
    return values


def projection_family(name: str) -> str:
    for suffix in (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ):
        if name.endswith(suffix):
            return suffix.removesuffix(".weight")
    return "other"


def evaluate_pair(pair: WeightScalePair, row_chunk: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rows, cols = pair.weight.shape
    metrics = {
        "SQ9_0": new_metric_state(),
        "Q8_0_g32_f16": new_metric_state(),
        "Q8_0_g128_f16": new_metric_state(),
    }
    telemetry: dict[str, dict[str, int]] = {
        "SQ9_0": {},
        "Q8_0_g32_f16": {},
        "Q8_0_g128_f16": {},
        "source": {"nonfinite_f8_codes": 0},
    }
    spread = new_spread_state()

    with pair.weight_path.open("rb") as weight_handle, pair.scale_path.open("rb") as scale_handle:
        with mmap.mmap(weight_handle.fileno(), 0, access=mmap.ACCESS_READ) as weight_map:
            with mmap.mmap(scale_handle.fileno(), 0, access=mmap.ACCESS_READ) as scale_map:
                scales = load_scale_values(scale_map, pair.scale)
                for row_start in range(0, rows, row_chunk):
                    row_stop = min(rows, row_start + row_chunk)
                    values, source_nonfinite = source_values(
                        weight_map, pair.weight, row_start, row_stop, scales
                    )
                    telemetry["source"]["nonfinite_f8_codes"] += source_nonfinite

                    sq9_reconstructed, sq9_telemetry = quantize_sq9_0(values)
                    sq9_error = update_metric_state(
                        metrics["SQ9_0"],
                        values,
                        sq9_reconstructed,
                        tensor_name=pair.name,
                        row_start=row_start,
                    )
                    merge_integer_telemetry(telemetry["SQ9_0"], sq9_telemetry)

                    q8_32_reconstructed, q8_32_telemetry, _ = quantize_q8_0(values, 32)
                    q8_32_error = update_metric_state(
                        metrics["Q8_0_g32_f16"],
                        values,
                        q8_32_reconstructed,
                        tensor_name=pair.name,
                        row_start=row_start,
                    )
                    merge_integer_telemetry(telemetry["Q8_0_g32_f16"], q8_32_telemetry)
                    add_spread_statistics(spread, values, sq9_error, q8_32_error)

                    q8_128_reconstructed, q8_128_telemetry, _ = quantize_q8_0(values, 128)
                    update_metric_state(
                        metrics["Q8_0_g128_f16"],
                        values,
                        q8_128_reconstructed,
                        tensor_name=pair.name,
                        row_start=row_start,
                    )
                    merge_integer_telemetry(telemetry["Q8_0_g128_f16"], q8_128_telemetry)

    row = {
        "tensor": pair.name,
        "family": projection_family(pair.name),
        "shape": [rows, cols],
        "elements": rows * cols,
        "source": {
            "weight_dtype": pair.weight.dtype,
            "weight_shard": pair.weight_path.name,
            "scale_name": pair.scale.name,
            "scale_dtype": pair.scale.dtype,
            "scale_shard": pair.scale_path.name,
            "scale_shape": list(pair.scale.shape),
            "scale_block_shape": [SOURCE_BLOCK_ROWS, SOURCE_BLOCK_COLS],
            "reconstruction": "decode_e4m3fn(weight) * bf16_to_f32(weight_scale_inv)",
            **telemetry["source"],
        },
        "formats": {
            format_name: {
                "metrics": finalize_metric_state(metric),
                "telemetry": telemetry[format_name],
            }
            for format_name, metric in metrics.items()
        },
        "q8_0_g32_block_spread": spread,
    }
    return row, metrics


def aggregate_spread(
    destination: dict[str, dict[str, float | int]], source: dict[str, dict[str, float | int]]
) -> None:
    for name in destination:
        for key in destination[name]:
            destination[name][key] += source[name][key]


def summarize_spread(state: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for name, values in state.items():
        entry = dict(values)
        reference_sse = float(entry["reference_sse"])
        entry["SQ9_0_relative_mse"] = (
            float(entry["sq9_0_error_sse"]) / reference_sse if reference_sse else 0.0
        )
        entry["Q8_0_g32_f16_relative_mse"] = (
            float(entry["q8_0_g32_f16_error_sse"]) / reference_sse if reference_sse else 0.0
        )
        result[name] = entry
    return result


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise EvaluationError(f"output directory is non-empty (use --overwrite): {path}")
    path.mkdir(parents=True, exist_ok=True)


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_self_test() -> None:
    input_values = np.array(
        [[0.0, 1.0, 1.0625, 1.1875, -1.0625, 2.0**-17, 2.0**-18, 70000.0]],
        dtype=np.float32,
    )
    reconstructed, telemetry = quantize_sq9_0(input_values)
    expected = np.array(
        [[0.0, 1.0, 1.0, 1.25, -1.0, 2.0**-17, 0.0, 61440.0]], dtype=np.float32
    )
    if not np.array_equal(reconstructed, expected):
        raise EvaluationError(
            f"SQ9_0 RNE self-test failed: got={reconstructed.tolist()} expected={expected.tolist()}"
        )
    if telemetry["saturated_to_finite_max"] != 1 or telemetry["rounded_to_zero"] != 1:
        raise EvaluationError(f"SQ9_0 telemetry self-test failed: {telemetry}")
    q8_values = np.array([[1.0, -1.0] + [0.0] * 30], dtype=np.float32)
    q8_reconstructed, q8_telemetry, _ = quantize_q8_0(q8_values, 32)
    # The canonical scale itself is stored as FP16, so 1/127 is rounded before
    # reconstruction; this checks the expected bounded FP16-scale endpoint error.
    if not np.allclose(q8_reconstructed, q8_values, rtol=0.0, atol=1.0 / 16384.0):
        raise EvaluationError("Q8_0 self-test exceeded the expected FP16-scale endpoint error")
    if q8_telemetry["groups"] != 1:
        raise EvaluationError(f"Q8_0 telemetry self-test failed: {q8_telemetry}")


def select_pairs(pairs: Iterable[WeightScalePair], patterns: list[str], maximum: int | None) -> list[WeightScalePair]:
    import re

    compiled = [re.compile(pattern) for pattern in patterns]
    selected = [
        pair
        for pair in pairs
        if not compiled or any(pattern.search(pair.name) for pattern in compiled)
    ]
    if maximum is not None:
        selected = selected[:maximum]
    if not selected:
        raise EvaluationError("no source FP8 tensor pairs match the requested selection")
    return selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.row_chunk <= 0:
        raise EvaluationError("--row-chunk must be positive")
    if args.max_tensors is not None and args.max_tensors <= 0:
        raise EvaluationError("--max-tensors must be positive")
    run_self_test()
    if args.self_test:
        return {"status": "self_test_passed", "script_version": SCRIPT_VERSION}
    if args.output_dir is None:
        raise EvaluationError("--output-dir is required unless --self-test is used")

    model_dir = args.source_model_dir.resolve()
    if not model_dir.is_dir():
        raise EvaluationError(f"source model directory does not exist: {model_dir}")
    pairs = select_pairs(discover_pairs(model_dir), args.tensor_regex, args.max_tensors)
    prepare_output_dir(args.output_dir, args.overwrite)

    metadata = {
        "schema_version": "ullm.sq9-q8-offline-error.v1",
        "script_version": SCRIPT_VERSION,
        "execution": {
            "accelerator_execution": "none",
            "cpu_only": True,
            "row_chunk": args.row_chunk,
            "pid": os.getpid(),
            "command_line": sys.argv,
            "numpy_version": np.__version__,
            "python_version": sys.version,
        },
        "source": {
            "model_dir": str(model_dir),
            "config_sha256": sha256_file(model_dir / "config.json"),
            "index_sha256": sha256_file(model_dir / "model.safetensors.index.json"),
            "weight_dtype": "F8_E4M3",
            "scale_dtype": "BF16",
            "scale_block_shape": [SOURCE_BLOCK_ROWS, SOURCE_BLOCK_COLS],
            "selected_pair_count": len(pairs),
            "selected_tensors": [pair.name for pair in pairs],
        },
        "quantizers": {
            "SQ9_0": {
                "value_encoding": "signed E5M3, finite E5M3 RNE, no reconstruction scale",
                "payload_bits_per_weight": 9.0,
                "finite_clamp": float(E5M3_MAX_FINITE),
            },
            "Q8_0_g32_f16": {
                "value_encoding": "signed int8 symmetric RNE, clamp [-127,127]",
                "scale": "one FP16 dequantization multiplier per contiguous 32 weights",
                "payload_bits_per_weight": 8.5,
            },
            "Q8_0_g128_f16": {
                "value_encoding": "signed int8 symmetric RNE, clamp [-127,127]",
                "scale": "one FP16 dequantization multiplier per contiguous 128 weights",
                "payload_bits_per_weight": 8.125,
            },
        },
    }
    json_dump(args.output_dir / "metadata.json", metadata)

    aggregate_metrics = {
        "SQ9_0": new_metric_state(),
        "Q8_0_g32_f16": new_metric_state(),
        "Q8_0_g128_f16": new_metric_state(),
    }
    aggregate_telemetry: dict[str, dict[str, int]] = {
        "SQ9_0": {},
        "Q8_0_g32_f16": {},
        "Q8_0_g128_f16": {},
        "source": {"nonfinite_f8_codes": 0},
    }
    aggregate_spread_state = new_spread_state()
    rows: list[dict[str, Any]] = []
    per_tensor_path = args.output_dir / "per-tensor.jsonl"
    with per_tensor_path.open("w", encoding="utf-8") as per_tensor:
        for index, pair in enumerate(pairs, start=1):
            print(f"[{index}/{len(pairs)}] {pair.name}", flush=True)
            row, raw_metrics = evaluate_pair(pair, args.row_chunk)
            per_tensor.write(json.dumps(row, sort_keys=True) + "\n")
            per_tensor.flush()
            rows.append(row)
            for format_name, metric in raw_metrics.items():
                add_metric_states(aggregate_metrics[format_name], metric)
                merge_integer_telemetry(
                    aggregate_telemetry[format_name], row["formats"][format_name]["telemetry"]
                )
            aggregate_telemetry["source"]["nonfinite_f8_codes"] += row["source"][
                "nonfinite_f8_codes"
            ]
            aggregate_spread(aggregate_spread_state, row["q8_0_g32_block_spread"])

    summary = {
        "schema_version": "ullm.sq9-q8-offline-error-summary.v1",
        "metadata_file": "metadata.json",
        "per_tensor_file": "per-tensor.jsonl",
        "selected_tensor_count": len(rows),
        "elements": sum(row["elements"] for row in rows),
        "formats": {
            format_name: {
                "metrics": finalize_metric_state(metric),
                "telemetry": aggregate_telemetry[format_name],
            }
            for format_name, metric in aggregate_metrics.items()
        },
        "q8_0_g32_block_spread": summarize_spread(aggregate_spread_state),
        "source_nonfinite_f8_codes": aggregate_telemetry["source"]["nonfinite_f8_codes"],
        "interpretation_boundary": (
            "Errors are incremental requantization errors relative to the source-correct "
            "F8_E4M3 plus BF16 128x128 reconstructed checkpoint, not errors versus an unavailable "
            "pre-FP8 training checkpoint."
        ),
    }
    json_dump(args.output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except EvaluationError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
