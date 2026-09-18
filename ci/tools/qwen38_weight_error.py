#!/usr/bin/env python3
"""Compare Qwen3.8 FP8 weight recipes against the pinned BF16 checkpoint.

This is an offline, tensor-level oracle.  It does not use either inference
engine's dequantization routine: the two byte formats are decoded here and
compared with the original BF16 values from the HF safetensors checkpoint.

The sLLM artifact stores OCP E4M3FN values with one E8M0 power-of-two scale
for every consecutive block of 32 values along K.  The vLLM checkpoint stores
OCP E4M3FN values with one BF16 inverse scale for each 128x128 tile.  Both
matrices are row-major in the source checkpoint.  GGUF exposes reversed
dimensions for ordinary tensors, so the logical shape comes from the source
checkpoint and recipe metadata rather than the GGUF physical shape.

Quantile values are computed from a deterministic systematic sample per
tensor; max and all moment statistics are exact streaming accumulations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import mmap
from pathlib import Path
import struct
import sys
import time
from typing import Any, Iterable

import numpy as np
from gguf import GGUFReader


BF16_EXPECTED = {
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "intermediate_size": 17408,
    "vocab_size": 248320,
}
MX_RECIPE_ENCODING = "mxfp8-e4m3-block32-e8m0"
GGUF_I8_CARRIER = 24
GGUF_BF16 = 30
DEFAULT_BF16 = Path("/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-BF16")
DEFAULT_FP8 = Path("/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-FP8")
DEFAULT_MX = Path(
    "/home/homelab1/datapool/qwen38-kld-20260918/sllm-mxfp8/"
    "Qwen3.8-27B-MXFP8.gguf"
)
DEFAULT_OUTPUT = Path(
    "/home/homelab1/datapool/qwen38-kld-20260918/results/weight-error-qwen38.json"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_file_info(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    return {"path": str(path), "size_bytes": len(raw), "sha256": sha256_bytes(raw), "value": value}


def decode_e4m3fn(bits: np.ndarray) -> np.ndarray:
    """Decode OCP E4M3FN bytes without using torch's float8 conversion."""
    bits = np.asarray(bits, dtype=np.uint8)
    sign = np.where((bits & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    exponent = ((bits >> 3) & 0x0F).astype(np.int16)
    mantissa = (bits & 0x07).astype(np.int16)
    result = np.where(
        exponent == 0,
        mantissa.astype(np.float32) * np.float32(2.0**-9),
        (1.0 + mantissa.astype(np.float32) / 8.0)
        * np.exp2(exponent.astype(np.float32) - 7.0),
    ).astype(np.float32)
    # 0x7f and 0xff are the two NaN encodings.  A valid model should never
    # contain them, but retaining NaN makes a bad artifact fail visibly.
    result[(exponent == 15) & (mantissa == 7)] = np.nan
    return result * sign


def decode_e8m0(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.exp2(bits.astype(np.float32) - 127.0).astype(np.float32)
    result[bits == 0] = np.float32(2.0**-127)
    result[bits == 255] = np.nan
    return result


E4_TABLE = decode_e4m3fn(np.arange(256, dtype=np.uint8))
E8_TABLE = decode_e8m0(np.arange(256, dtype=np.uint8))


def encode_e4m3fn(values: np.ndarray) -> np.ndarray:
    """Vectorized round-to-nearest-even OCP E4M3FN encoder.

    This is used only by the no-clipping counterfactual.  The production
    artifact is decoded above; re-encoding here keeps the counterfactual
    independent of both the Rust converter and torch float8 operators.
    """
    values = np.asarray(values, dtype=np.float32)
    negative = np.signbit(values)
    magnitude = np.abs(values)
    result = np.zeros(values.shape, dtype=np.uint8)
    finite = np.isfinite(magnitude)
    nonzero = finite & (magnitude != 0.0)
    small = nonzero & (magnitude < np.float32(2.0**-6))
    if np.any(small):
        result[small] = np.clip(np.rint(magnitude[small] * 512.0), 0, 7).astype(np.uint8)
    normal = nonzero & ~small
    if np.any(normal):
        exponent = np.floor(np.log2(magnitude[normal])).astype(np.int32)
        quantum = np.exp2(exponent - 3)
        significand = np.rint(magnitude[normal] / quantum).astype(np.int32)
        result[normal] = np.clip(exponent * 8 + 48 + significand, 0, 0x7E).astype(np.uint8)
    result[~finite] = 0x7E
    result[negative] |= 0x80
    return result


def u8_view(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.dtype != np.uint8:
        data = data.view(np.uint8)
    return data.reshape(-1)


@dataclass(frozen=True)
class SafeTensorDescriptor:
    """The small, dtype-neutral part of a safetensors tensor header."""

    shard: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int


class SafeTensorStore:
    """Read BF16/F8_E4M3 safetensors without a framework dependency.

    The usual safetensors framework adapters materialize BF16 through torch.
    This oracle intentionally parses the documented container header and
    widens BF16 bits itself, keeping its numerical reference independent of
    PyTorch and either inference engine.
    """

    def __init__(self, root: Path, index: dict[str, Any]):
        self.root = root
        self._files: dict[str, Any] = {}
        self._maps: dict[str, mmap.mmap] = {}
        self.descriptors: dict[str, SafeTensorDescriptor] = {}
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise RuntimeError(f"{root}: safetensors index has no weight_map")
        for shard in sorted(set(weight_map.values())):
            path = root / shard
            stream = path.open("rb")
            prefix = stream.read(8)
            if len(prefix) != 8:
                stream.close()
                raise RuntimeError(f"{path}: short safetensors header length")
            header_length = struct.unpack("<Q", prefix)[0]
            header = stream.read(header_length)
            if len(header) != header_length:
                stream.close()
                raise RuntimeError(f"{path}: short safetensors header")
            try:
                entries = json.loads(header)
            except json.JSONDecodeError as exc:
                stream.close()
                raise RuntimeError(f"{path}: invalid safetensors header: {exc}") from exc
            if not isinstance(entries, dict):
                stream.close()
                raise RuntimeError(f"{path}: safetensors header is not an object")
            data_base = 8 + header_length
            self._files[shard] = stream
            self._maps[shard] = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            for name, entry in entries.items():
                if name == "__metadata__":
                    continue
                if not isinstance(entry, dict):
                    raise RuntimeError(f"{path}: tensor {name} header is not an object")
                shape = entry.get("shape")
                offsets = entry.get("data_offsets")
                dtype = entry.get("dtype")
                if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2 or not isinstance(dtype, str):
                    raise RuntimeError(f"{path}: malformed tensor header for {name}")
                self.descriptors[name] = SafeTensorDescriptor(
                    shard=shard,
                    dtype=dtype,
                    shape=tuple(int(x) for x in shape),
                    data_start=data_base + int(offsets[0]),
                    data_end=data_base + int(offsets[1]),
                )
        missing = sorted(set(weight_map) - set(self.descriptors))
        if missing:
            raise RuntimeError(f"{root}: index names absent from shard headers: {missing[:4]}")

    def close(self) -> None:
        for mapping in self._maps.values():
            mapping.close()
        for stream in self._files.values():
            stream.close()
        self._maps.clear()
        self._files.clear()

    def __enter__(self) -> "SafeTensorStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def read(self, name: str, row_start: int | None = None, row_end: int | None = None) -> np.ndarray:
        descriptor = self.descriptors[name]
        shape = descriptor.shape
        if row_start is None:
            row_start = 0
        if row_end is None:
            row_end = shape[0] if len(shape) == 2 else 1
        if len(shape) == 2:
            if not 0 <= row_start <= row_end <= shape[0]:
                raise RuntimeError(f"{name}: invalid row range {row_start}:{row_end}")
            elements = (row_end - row_start) * shape[1]
            row_offset = row_start * shape[1]
        else:
            if row_start != 0 or row_end != 1:
                raise RuntimeError(f"{name}: row range requested for rank-{len(shape)} tensor")
            elements = int(math.prod(shape))
            row_offset = 0
        if descriptor.dtype == "BF16":
            item_bytes = 2
            raw_dtype = np.dtype("<u2")
        elif descriptor.dtype == "F8_E4M3":
            item_bytes = 1
            raw_dtype = np.dtype("u1")
        else:
            raise RuntimeError(f"{name}: unsupported safetensors dtype {descriptor.dtype}")
        begin = descriptor.data_start + row_offset * item_bytes
        end = begin + elements * item_bytes
        if end > descriptor.data_end:
            raise RuntimeError(f"{name}: requested bytes exceed tensor payload")
        raw = np.frombuffer(self._maps[descriptor.shard], dtype=raw_dtype, count=elements, offset=begin)
        if descriptor.dtype == "BF16":
            # A BF16 value is its 16-bit bits shifted into the high half of an
            # IEEE FP32 word.  The source bytes are little-endian by format.
            bits = raw.astype(np.uint32, copy=False) << 16
            values = bits.view("<f4")
        else:
            values = raw.copy()
        out_shape = (row_end - row_start, shape[1]) if len(shape) == 2 else shape
        return values.reshape(out_shape)


@dataclass
class MetricAccumulator:
    count: int = 0
    ref_sq: float = 0.0
    quant_sq: float = 0.0
    cross: float = 0.0
    error_sq: float = 0.0
    abs_sum: float = 0.0
    max_abs: float = 0.0
    max_rel: float = 0.0
    rel_epsilon: float = 1.0e-12
    samples: list[float] | None = None
    saturation_count: int = 0
    saturation_error_sq: float = 0.0
    saturation_abs_sum: float = 0.0

    def add(
        self,
        reference: np.ndarray,
        quantized: np.ndarray,
        global_offset: int,
        stride: int,
        saturation_mask: np.ndarray | None = None,
    ) -> None:
        reference = np.asarray(reference, dtype=np.float64).reshape(-1)
        quantized = np.asarray(quantized, dtype=np.float64).reshape(-1)
        if reference.shape != quantized.shape:
            raise RuntimeError("metric input shape mismatch")
        error = quantized - reference
        abs_error = np.abs(error)
        rel_error = abs_error / np.maximum(np.abs(reference), self.rel_epsilon)
        self.count += int(reference.size)
        self.ref_sq += float(np.dot(reference, reference))
        self.quant_sq += float(np.dot(quantized, quantized))
        self.cross += float(np.dot(reference, quantized))
        self.error_sq += float(np.dot(error, error))
        self.abs_sum += float(abs_error.sum())
        if abs_error.size:
            self.max_abs = max(self.max_abs, float(abs_error.max()))
            self.max_rel = max(self.max_rel, float(rel_error.max()))
        if saturation_mask is not None:
            saturation_mask = np.asarray(saturation_mask, dtype=bool).reshape(-1)
            if saturation_mask.shape != reference.shape:
                raise RuntimeError("saturation mask shape mismatch")
            self.saturation_count += int(saturation_mask.sum())
            self.saturation_error_sq += float(np.dot(error[saturation_mask], error[saturation_mask]))
            self.saturation_abs_sum += float(abs_error[saturation_mask].sum())
        if self.samples is not None and stride > 0:
            # Use a global systematic sample so chunk boundaries cannot bias
            # the reported p95/p99 values.
            first = (-global_offset) % stride
            if first < abs_error.size:
                self.samples.extend(abs_error[first::stride].astype(np.float32).tolist())

    def finish(self) -> dict[str, Any]:
        if self.count == 0:
            raise RuntimeError("empty metric")
        ref_norm = math.sqrt(max(self.ref_sq, 0.0))
        quant_norm = math.sqrt(max(self.quant_sq, 0.0))
        cosine = self.cross / (ref_norm * quant_norm) if ref_norm and quant_norm else None
        sample = np.asarray(self.samples or [], dtype=np.float32)
        return {
            "elements": self.count,
            "reference_sq_sum": self.ref_sq,
            "quantized_sq_sum": self.quant_sq,
            "cross_sum": self.cross,
            "error_sq_sum": self.error_sq,
            "abs_error_sum": self.abs_sum,
            "rms_abs_error": math.sqrt(max(self.error_sq, 0.0) / self.count),
            "relative_rms_error": math.sqrt(max(self.error_sq, 0.0) / max(self.ref_sq, 1.0e-30)),
            "mean_abs_error": self.abs_sum / self.count,
            "max_abs_error": self.max_abs,
            "max_element_relative_error": self.max_rel,
            "cosine": cosine,
            "saturation_max_code_count": self.saturation_count,
            "saturation_max_code_fraction": self.saturation_count / self.count,
            "saturation_error_sq_fraction": self.saturation_error_sq / self.error_sq if self.error_sq else 0.0,
            "saturation_error_sq_sum": self.saturation_error_sq,
            "saturation_abs_error_sum": self.saturation_abs_sum,
            "p95_abs_error_sample": float(np.percentile(sample, 95)) if sample.size else None,
            "p99_abs_error_sample": float(np.percentile(sample, 99)) if sample.size else None,
            "quantile_sample_count": int(sample.size),
        }


def new_accumulator(total_elements: int, sample_limit: int) -> tuple[MetricAccumulator, int]:
    stride = max(1, math.ceil(total_elements / sample_limit))
    return MetricAccumulator(samples=[]), stride


def stats_from_tensors(
    name: str,
    reference_store: SafeTensorStore,
    decoded: Iterable[tuple[int, np.ndarray, np.ndarray | None, np.ndarray | None]],
    shape: tuple[int, int],
    sample_limit: int,
    row_chunk: int,
    counterfactual_block_limit: int = 0,
) -> dict[str, Any]:
    rows, columns = shape
    accumulator, stride = new_accumulator(rows * columns, sample_limit)
    blocks_per_row = columns // 32
    counterfactual_accumulator = MetricAccumulator() if counterfactual_block_limit > 0 and columns % 32 == 0 else None
    paired_current_accumulator = MetricAccumulator() if counterfactual_accumulator is not None else None
    counterfactual_total_blocks = rows * blocks_per_row if counterfactual_accumulator is not None else 0
    counterfactual_stride = max(1, math.ceil(counterfactual_total_blocks / counterfactual_block_limit)) if counterfactual_accumulator is not None else 0
    counterfactual_block_count = 0
    counterfactual_scale_up_block_count = 0
    current_clipping_block_count = 0
    for start, values, saturation_limit, max_code_mask in decoded:
        end = start + values.shape[0]
        ref = reference_store.read(name, start, end)
        if saturation_limit is not None and max_code_mask is not None:
            saturation_mask = max_code_mask & (np.abs(ref) > saturation_limit * (1.0 + 1.0e-7))
        else:
            saturation_mask = None
        accumulator.add(ref, values, start * columns, stride, saturation_mask)
        if counterfactual_accumulator is not None:
            block_reference = ref.reshape(end - start, blocks_per_row, 32)
            block_ids = np.arange(start * blocks_per_row, end * blocks_per_row, dtype=np.int64).reshape(end - start, blocks_per_row)
            selected = (block_ids % counterfactual_stride) == 0
            if saturation_mask is not None:
                current_clipping_block_count += int(np.any(saturation_mask.reshape(end - start, blocks_per_row, 32), axis=2).sum())
            if np.any(selected):
                selected_reference = block_reference[selected]
                maxima = np.max(np.abs(selected_reference), axis=1)
                # Select the smallest representable E8M0 scale that puts the
                # block maximum at or below E4M3FN's finite limit.  The
                # production MX scale is floor(log2(max))-8, so this differs
                # only for blocks whose normalized maximum exceeds 1.75.
                finite_nonzero = np.isfinite(maxima) & (maxima != 0.0)
                exponents = np.zeros(maxima.shape, dtype=np.int32)
                exponents[finite_nonzero] = np.ceil(
                    np.log2(maxima[finite_nonzero] / np.float32(448.0))
                ).astype(np.int32)
                exponents = np.clip(exponents, -127, 127)
                scales = np.exp2(exponents).astype(np.float32)
                codes = encode_e4m3fn(selected_reference / scales[:, None])
                quantized = E4_TABLE[codes] * scales[:, None]
                if paired_current_accumulator is not None:
                    current_blocks = values.reshape(end - start, blocks_per_row, 32)[selected]
                    paired_current_accumulator.add(selected_reference, current_blocks, 0, 0)
                counterfactual_accumulator.add(
                    selected_reference,
                    quantized,
                    0,
                    0,
                )
                counterfactual_block_count += int(selected_reference.shape[0])
                current_exponents = np.zeros(maxima.shape, dtype=np.int32)
                current_exponents[finite_nonzero] = np.floor(np.log2(maxima[finite_nonzero])).astype(np.int32) - 8
                counterfactual_scale_up_block_count += int(np.count_nonzero(exponents > current_exponents))
    result = accumulator.finish()
    result.update({"name": name, "shape": [rows, columns], "row_chunk": row_chunk, "sample_stride": stride})
    if counterfactual_accumulator is not None:
        result["counterfactual_no_clipping_e8m0"] = counterfactual_accumulator.finish()
        result["counterfactual_no_clipping_e8m0"]["paired_current_same_sample"] = paired_current_accumulator.finish()
        result["counterfactual_no_clipping_e8m0"].update({
            "sampled_block_count": counterfactual_block_count,
            "sample_stride_blocks": counterfactual_stride,
            "scale_up_block_count": counterfactual_scale_up_block_count,
            "current_clipping_block_count": current_clipping_block_count,
            "description": "sampled blocks re-quantized with ceil(log2(max/448)) E8M0 scale",
        })
    return result


def decode_mx_rows(
    values: np.ndarray,
    scales: np.ndarray,
    rows: int,
    columns: int,
    row_chunk: int,
) -> Iterable[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    if columns % 32:
        raise RuntimeError(f"MX tensor K dimension is not block32 aligned: {columns}")
    expected_values = rows * columns
    expected_scales = rows * (columns // 32)
    if len(values) != expected_values or len(scales) != expected_scales:
        raise RuntimeError(
            f"MX plane length mismatch: values={len(values)} expected={expected_values}, "
            f"scales={len(scales)} expected={expected_scales}"
        )
    blocks_per_row = columns // 32
    for start in range(0, rows, row_chunk):
        end = min(rows, start + row_chunk)
        codes = values[start * columns : end * columns]
        scale_bits = scales[start * blocks_per_row : end * blocks_per_row]
        decoded = E4_TABLE[codes] * np.repeat(E8_TABLE[scale_bits], 32)
        scale = np.repeat(E8_TABLE[scale_bits], 32)
        limit = np.float32(448.0) * scale
        max_code = (codes & np.uint8(0x7F)) == np.uint8(0x7E)
        yield start, decoded.reshape(end - start, columns), limit.reshape(end - start, columns), max_code.reshape(end - start, columns)


def decode_fp8_rows(
    value_store: SafeTensorStore,
    value_name: str,
    scale_name: str,
    rows: int,
    columns: int,
    row_chunk: int,
) -> Iterable[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    if rows % 128 or columns % 128:
        raise RuntimeError(f"official FP8 matrix is not 128x128 aligned: {rows}x{columns}")
    expected_scales = (rows // 128, columns // 128)
    scales = np.asarray(value_store.read(scale_name), dtype=np.float32)
    if tuple(int(x) for x in scales.shape) != expected_scales:
        raise RuntimeError(f"FP8 inverse scales: expected shape {expected_scales}, got {scales.shape}")
    for start in range(0, rows, row_chunk):
        end = min(rows, start + row_chunk)
        if start % 128 or end % 128:
            raise RuntimeError("FP8 row chunk must align to 128-row tiles")
        codes = np.asarray(value_store.read(value_name, start, end), dtype=np.uint8).reshape(-1)
        tile_scales = scales[start // 128 : end // 128]
        expanded_scales = np.repeat(np.repeat(tile_scales, 128, axis=0), 128, axis=1)
        decoded = E4_TABLE[codes] * expanded_scales.reshape(-1)
        limit = np.float32(448.0) * expanded_scales
        max_code = (codes & np.uint8(0x7F)) == np.uint8(0x7E)
        yield start, decoded.reshape(end - start, columns), limit, max_code.reshape(end - start, columns)


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def hash_store_payload(store: SafeTensorStore, name: str) -> str:
    """Hash one safetensors payload in bounded chunks."""
    descriptor = store.descriptors[name]
    digest = hashlib.sha256()
    mapping = store._maps[descriptor.shard]
    for begin in range(descriptor.data_start, descriptor.data_end, 1 << 20):
        digest.update(mapping[begin : min(begin + (1 << 20), descriptor.data_end)])
    return digest.hexdigest()


def hash_gguf_payload(tensor: Any) -> str:
    """Hash a GGUF tensor's mapped bytes without materializing its payload."""
    digest = hashlib.sha256()
    raw = np.asarray(tensor.data).view(np.uint8).reshape(-1)
    for begin in range(0, raw.size, 1 << 20):
        digest.update(raw[begin : min(begin + (1 << 20), raw.size)].tobytes())
    return digest.hexdigest()


def read_root_config(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    config_info = json_file_info(root / "config.json")
    index_info = json_file_info(root / "model.safetensors.index.json")
    config = config_info["value"]
    index = index_info["value"]
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise RuntimeError(f"{root}: text_config is not an object")
    for key, expected in BF16_EXPECTED.items():
        if text_config.get(key) != expected:
            raise RuntimeError(f"{root}: {key}={text_config.get(key)!r}, expected {expected!r}")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"{root}: no safetensors weight_map")
    return config, index, {"config": config_info["sha256"], "index": index_info["sha256"]}


def compact_config(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config", config)
    result = {key: text.get(key) for key in BF16_EXPECTED}
    result.update({"model_type": config.get("model_type"), "architectures": config.get("architectures")})
    quant = config.get("quantization_config")
    if quant:
        result["quantization_config"] = {
            key: value for key, value in quant.items() if key != "modules_to_not_convert"
        }
        result["modules_to_not_convert_count"] = len(quant.get("modules_to_not_convert", []))
    return result


def gguf_recipe(reader: GGUFReader) -> dict[str, Any]:
    field = reader.fields.get("sllm.tensor_recipe")
    if field is None:
        raise RuntimeError("MX GGUF has no sllm.tensor_recipe")
    recipe = json.loads(field.contents())
    if not isinstance(recipe, dict):
        raise RuntimeError("sllm.tensor_recipe is not an object")
    return recipe


def classify_role(name: str) -> str:
    if ".mlp." in name:
        return "mlp"
    if ".linear_attn." in name:
        return "linear_attention"
    if ".self_attn." in name:
        return "full_attention"
    if name.endswith("lm_head.weight"):
        return "lm_head"
    if "embed_tokens" in name:
        return "embedding"
    return "other"


def is_text_weight(name: str) -> bool:
    return name.startswith("model.language_model.") or name in {
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
    }


def special_family(name: str) -> str | None:
    if name == "lm_head.weight":
        return "lm_head"
    if name == "model.language_model.embed_tokens.weight":
        return "embed_tokens"
    if ".linear_attn.in_proj_a.weight" in name:
        return "gdn_in_proj_a"
    if ".linear_attn.in_proj_b.weight" in name:
        return "gdn_in_proj_b"
    if ".linear_attn.in_proj_ba.weight" in name:
        return "gdn_in_proj_ba"
    return None


def layer_number(name: str) -> int | None:
    marker = ".layers."
    if marker not in name:
        return None
    tail = name.split(marker, 1)[1]
    try:
        return int(tail.split(".", 1)[0])
    except (ValueError, IndexError):
        return None


def sum_metric_groups(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate exact sums from per-tensor metrics.

    Per-tensor output retains all metrics.  Aggregates reconstruct moments from
    the reported RMS/cosine fields, so no giant dequantized matrix is retained.
    Quantiles remain the median of tensor quantiles and are explicitly labeled
    as such; they are not presented as an element-weighted global percentile.
    """
    if not metrics:
        return {"tensors": 0, "elements": 0}
    count = sum(int(x["elements"]) for x in metrics)
    ref_sq = sum(float(x.get("reference_sq_sum", 0.0)) for x in metrics)
    quant_sq = sum(float(x.get("quantized_sq_sum", 0.0)) for x in metrics)
    cross = sum(float(x.get("cross_sum", 0.0)) for x in metrics)
    error_sq = sum(float(x.get("error_sq_sum", 0.0)) for x in metrics)
    abs_sum = sum(float(x.get("abs_error_sum", 0.0)) for x in metrics)
    saturation_count = sum(int(x.get("saturation_max_code_count", 0)) for x in metrics)
    saturation_error_sq = sum(float(x.get("saturation_error_sq_sum", 0.0)) for x in metrics)
    saturation_abs_sum = sum(float(x.get("saturation_abs_error_sum", 0.0)) for x in metrics)
    cosine = cross / math.sqrt(ref_sq * quant_sq) if ref_sq and quant_sq else None
    return {
        "tensors": len(metrics),
        "elements": count,
        "reference_sq_sum": ref_sq,
        "quantized_sq_sum": quant_sq,
        "cross_sum": cross,
        "error_sq_sum": error_sq,
        "abs_error_sum": abs_sum,
        "rms_abs_error": math.sqrt(error_sq / count),
        "relative_rms_error": math.sqrt(error_sq / ref_sq) if ref_sq else None,
        "mean_abs_error": abs_sum / count,
        "cosine": cosine,
        "saturation_max_code_count": saturation_count,
        "saturation_max_code_fraction": saturation_count / count,
        "saturation_error_sq_sum": saturation_error_sq,
        "saturation_error_sq_fraction": saturation_error_sq / error_sq if error_sq else 0.0,
        "saturation_abs_error_sum": saturation_abs_sum,
        "p95_abs_error_sample_median_across_tensors": float(np.median([x["p95_abs_error_sample"] for x in metrics if x["p95_abs_error_sample"] is not None])),
        "p99_abs_error_sample_median_across_tensors": float(np.median([x["p99_abs_error_sample"] for x in metrics if x["p99_abs_error_sample"] is not None])),
    }


def load_identity(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-root", type=Path, default=DEFAULT_BF16)
    parser.add_argument("--fp8-root", type=Path, default=DEFAULT_FP8)
    parser.add_argument("--mxfp8-gguf", type=Path, default=DEFAULT_MX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tensors", type=int, default=None, help="debug limit; default processes every MXFP8 linear tensor")
    parser.add_argument("--sample-limit", type=int, default=65536)
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument(
        "--counterfactual-block-limit",
        type=int,
        default=4096,
        help="sampled MX blocks for the no-clipping scale counterfactual (0 disables)",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.sample_limit <= 0 or args.row_chunk <= 0 or args.counterfactual_block_limit < 0:
        raise RuntimeError("sample-limit and row-chunk must be positive; counterfactual-block-limit cannot be negative")
    bf_root = args.bf16_root.resolve()
    fp_root = args.fp8_root.resolve()
    mx_path = args.mxfp8_gguf.resolve()
    bf_config, bf_index, bf_hashes = read_root_config(bf_root)
    fp_config, fp_index, fp_hashes = read_root_config(fp_root)
    bf_base_names = set(bf_index["weight_map"])
    fp_base_names = {name for name in fp_index["weight_map"] if not name.endswith("_scale_inv")}
    if bf_base_names != fp_base_names:
        missing_in_fp = sorted(bf_base_names - fp_base_names)
        missing_in_bf = sorted(fp_base_names - bf_base_names)
        raise RuntimeError(
            "BF16 and FP8 safetensors base tensor inventories differ: "
            f"missing_in_fp={missing_in_fp[:4]}, missing_in_bf={missing_in_bf[:4]}"
        )

    reader = GGUFReader(str(mx_path))
    recipe = gguf_recipe(reader)
    bindings = recipe.get("bindings")
    if not isinstance(bindings, list):
        raise RuntimeError("MX recipe bindings are not a list")
    binding_map = {item.get("logical_tensor"): item for item in bindings if isinstance(item, dict)}
    gguf_tensors = {tensor.name: tensor for tensor in reader.tensors}
    all_mx_names = sorted(
        name for name, item in binding_map.items()
        if item.get("encoding") == MX_RECIPE_ENCODING and name in gguf_tensors and is_text_weight(name)
    )
    mx_names = all_mx_names
    if args.max_tensors is not None:
        if args.max_tensors <= 0:
            raise RuntimeError("max-tensors must be positive")
        mx_names = mx_names[: args.max_tensors]

    # Cache only memory-mapped shard handles. Each read widens one row chunk,
    # so the 2.5 GB head never needs to coexist with a full dequantized copy.
    metrics: list[dict[str, Any]] = []
    categories: dict[str, list[str]] = {"both_quantized": [], "mxfp8_only": [], "both_unquantized": [], "shape_or_dtype_mismatch": []}
    role_metrics: dict[str, list[dict[str, Any]]] = {}
    per_layer: dict[str, list[dict[str, Any]]] = {}
    started = time.time()
    with SafeTensorStore(bf_root, bf_index) as bf_store, SafeTensorStore(fp_root, fp_index) as fp_store:
        fp8_name_set: set[str] = set()
        fp8_dtype_set: set[str] = set()
        fp8_scale_shapes: dict[str, list[int]] = {}
        official_dtype_by_name: dict[str, str] = {}
        for name, shard in fp_index["weight_map"].items():
            if not name.endswith(".weight") or name not in bf_index["weight_map"] or not is_text_weight(name):
                continue
            descriptor = fp_store.descriptors[name]
            if len(descriptor.shape) == 2:
                dtype_name = descriptor.dtype
                fp8_dtype_set.add(dtype_name)
                official_dtype_by_name[name] = dtype_name
            if len(descriptor.shape) == 2 and descriptor.dtype == "F8_E4M3":
                fp8_name_set.add(name)
                scale_name = f"{name}_scale_inv"
                scale = fp_store.descriptors.get(scale_name)
                if scale is None:
                    raise RuntimeError(f"{name}: FP8 tensor has no {scale_name}")
                fp8_scale_shapes[name] = [int(x) for x in scale.shape]

        # Keep the BF16/head/GDN coverage visible even when a caller uses
        # --max-tensors for a quick smoke check.  This is the key distinction
        # between this MX artifact and the official FP8 checkpoint: the
        # latter leaves the small GDN input projections and output head in
        # BF16, while the former quantizes the two GDN projections it binds.
        special_inventory: list[dict[str, Any]] = []
        pair_coverage: dict[str, list[str]] = {}
        text_rank2_names: list[str] = []
        for name, shard in bf_index["weight_map"].items():
            if not name.endswith(".weight") or not is_text_weight(name):
                continue
            reference_descriptor = bf_store.descriptors[name]
            if len(reference_descriptor.shape) != 2:
                continue
            text_rank2_names.append(name)
            official_dtype = official_dtype_by_name.get(name, "absent")
            mx_tensor = gguf_tensors.get(name)
            mx_scale = gguf_tensors.get(f"{name}.sllm.scale.block32_e8m0")
            mx_quantized = (
                name in all_mx_names
                and mx_tensor is not None
                and mx_scale is not None
                and int(mx_tensor.tensor_type) == GGUF_I8_CARRIER
                and int(mx_scale.tensor_type) == GGUF_I8_CARRIER
            )
            if official_dtype == "F8_E4M3" and mx_quantized:
                pair = "official_fp8_and_mxfp8"
            elif official_dtype == "BF16" and mx_quantized:
                pair = "official_bf16_and_mxfp8"
            elif official_dtype == "BF16" and not mx_quantized:
                pair = "official_bf16_and_mxfp8_unquantized"
            else:
                pair = f"official_{official_dtype}_and_mxfp8_{'quantized' if mx_quantized else 'unquantized'}"
            pair_coverage.setdefault(pair, []).append(name)
            family = special_family(name)
            if family is not None:
                item = {
                    "name": name,
                    "family": family,
                    "shape": [int(x) for x in reference_descriptor.shape],
                    "official_dtype": official_dtype,
                    "mxfp8_quantized": mx_quantized,
                    "gguf_value_dtype": int(mx_tensor.tensor_type) if mx_tensor is not None else None,
                    "gguf_scale_present": mx_scale is not None,
                }
                if official_dtype == "BF16" and mx_tensor is not None and int(mx_tensor.tensor_type) == GGUF_BF16:
                    bf_hash = hash_store_payload(bf_store, name)
                    fp_hash = hash_store_payload(fp_store, name)
                    gg_hash = hash_gguf_payload(mx_tensor)
                    item.update({
                        "bf16_payload_sha256": bf_hash,
                        "official_fp8_payload_sha256": fp_hash,
                        "gguf_payload_sha256": gg_hash,
                        "all_three_payloads_byte_equal": bf_hash == fp_hash == gg_hash,
                    })
                special_inventory.append(item)

        for ordinal, name in enumerate(mx_names, 1):
            bf_shard = bf_index["weight_map"].get(name)
            if bf_shard is None:
                categories["shape_or_dtype_mismatch"].append(name)
                continue
            reference_descriptor = bf_store.descriptors[name]
            if reference_descriptor.dtype != "BF16" or len(reference_descriptor.shape) != 2:
                categories["shape_or_dtype_mismatch"].append(name)
                continue
            rows, columns = (int(reference_descriptor.shape[0]), int(reference_descriptor.shape[1]))
            binding_shape = tuple(int(x) for x in binding_map[name].get("logical_shape", []))
            if binding_shape != (rows, columns):
                raise RuntimeError(f"{name}: recipe shape {binding_shape} != BF16 shape {(rows, columns)}")
            value_tensor = gguf_tensors[name]
            scale_tensor = gguf_tensors.get(f"{name}.sllm.scale.block32_e8m0")
            if int(value_tensor.tensor_type) != GGUF_I8_CARRIER or scale_tensor is None or int(scale_tensor.tensor_type) != GGUF_I8_CARRIER:
                categories["shape_or_dtype_mismatch"].append(name)
                continue
            mx_values = u8_view(value_tensor.data)
            mx_scales = u8_view(scale_tensor.data)
            official_is_quantized = name in fp8_name_set
            category = "both_quantized" if official_is_quantized else "mxfp8_only"
            categories[category].append(name)
            mx_stat = stats_from_tensors(
                name,
                bf_store,
                decode_mx_rows(mx_values, mx_scales, rows, columns, args.row_chunk),
                (rows, columns),
                args.sample_limit,
                args.row_chunk,
                args.counterfactual_block_limit,
            )
            result: dict[str, Any] = {
                "name": name,
                "layer": layer_number(name),
                "role": classify_role(name),
                "shape": [rows, columns],
                "category": category,
                "reference_dtype": reference_descriptor.dtype,
                "mxfp8": mx_stat,
                "mxfp8_storage": {
                    "value_bytes": int(len(mx_values)),
                    "scale_bytes": int(len(mx_scales)),
                    "bits_per_element_including_scale": 8.0 * (len(mx_values) + len(mx_scales)) / (rows * columns),
                    "block_size": 32,
                    "scale_encoding": "E8M0 power-of-two, scale=2**(bits-127), bits=0 maps to 2**-127",
                    "value_encoding": "OCP E4M3FN",
                },
            }
            if official_is_quantized:
                fp_scale_name = f"{name}_scale_inv"
                if fp_store.descriptors[name].dtype != "F8_E4M3":
                    raise RuntimeError(f"{name}: FP8 name has dtype {fp_store.descriptors[name].dtype}")
                if fp_store.descriptors[name].shape != (rows, columns):
                    raise RuntimeError(f"{name}: FP8 shape {fp_store.descriptors[name].shape} != {(rows, columns)}")
                fp_scale_descriptor = fp_store.descriptors[fp_scale_name]
                fp_stat = stats_from_tensors(
                    name,
                    bf_store,
                    decode_fp8_rows(fp_store, name, fp_scale_name, rows, columns, args.row_chunk),
                    (rows, columns),
                    args.sample_limit,
                    args.row_chunk,
                )
                result["official_fp8"] = fp_stat
                result["official_fp8_storage"] = {
                    "value_bytes": int(fp_store.descriptors[name].data_end - fp_store.descriptors[name].data_start),
                    "scale_bytes": int(fp_scale_descriptor.data_end - fp_scale_descriptor.data_start),
                    "scale_dtype": fp_scale_descriptor.dtype,
                    "scale_shape": [int(x) for x in fp_scale_descriptor.shape],
                    "bits_per_element_including_scale": 8.0 * ((fp_store.descriptors[name].data_end - fp_store.descriptors[name].data_start) + (fp_scale_descriptor.data_end - fp_scale_descriptor.data_start)) / (rows * columns),
                    "block_shape": [128, 128],
                    "scale_encoding": "BF16 inverse scale, one per 128x128 tile",
                    "value_encoding": "OCP E4M3FN",
                }
            else:
                # The official checkpoint keeps these GDN projections in BF16.
                result["official_fp8"] = {
                    "name": name,
                    "shape": [rows, columns],
                    "reference_dtype": bf_store.descriptors[name].dtype,
                    "unquantized_exact_reference": True,
                    "rms_abs_error": 0.0,
                    "relative_rms_error": 0.0,
                    "mean_abs_error": 0.0,
                    "max_abs_error": 0.0,
                    "max_element_relative_error": 0.0,
                    "cosine": 1.0,
                }
            metrics.append(result)
            role_metrics.setdefault(result["role"], []).append(result["mxfp8"])
            per_layer.setdefault(str(result["layer"]), []).append(result["mxfp8"])
            if ordinal % 16 == 0:
                print(f"processed {ordinal}/{len(mx_names)} tensors", file=sys.stderr, flush=True)

    both = [x for x in metrics if x["category"] == "both_quantized"]
    m_only = [x for x in metrics if x["category"] == "mxfp8_only"]
    fp8_metrics = [x["official_fp8"] for x in both]
    mx_metrics = [x["mxfp8"] for x in metrics]
    comparison = {
        "both_quantized": {
            "tensor_count": len(both),
            "elements": sum(int(x["shape"][0]) * int(x["shape"][1]) for x in both),
            "mxfp8": sum_metric_groups([x["mxfp8"] for x in both]),
            "official_fp8": sum_metric_groups(fp8_metrics),
        },
        "mxfp8_only": {
            "tensor_count": len(m_only),
            "elements": sum(int(x["shape"][0]) * int(x["shape"][1]) for x in m_only),
            "mxfp8": sum_metric_groups([x["mxfp8"] for x in m_only]),
            "official_checkpoint": "BF16 exact (these matrices are excluded by modules_to_not_convert)",
        },
    }
    artifact_root = mx_path.parent.parent
    report = {
        "schema_version": "qwen38-weight-error-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": {
            "reference": "HF Qwen3.8-27B BF16 safetensors, values widened to FP32 for arithmetic",
            "mx": "independent OCP E4M3FN + E8M0 decoder in this script",
            "official_fp8": "independent OCP E4M3FN + BF16 inverse-scale decoder in this script",
            "relative_rms_definition": "sqrt(sum((q-ref)^2) / sum(ref^2))",
            "cosine_definition": "dot(ref,q)/(||ref||*||q||)",
            "quantile_definition": "p95/p99 of deterministic systematic sample per tensor; max and moments are exact",
            "mx_scale_formula": "scale = 2**(scale_byte-127), with byte 0 defined as 2**-127 and byte 255 invalid NaN",
            "fp8_scale_formula": "dequant = decode_e4m3fn(code) * BF16 scale_inv[row//128,col//128]",
            "mx_saturation_definition": "finite max code (0x7e/0xfe) whose BF16 source magnitude exceeds 448*block_scale",
            "mx_counterfactual": "sampled blocks use the smallest E8M0 scale ceil(log2(maximum/448)); this isolates scale-induced clipping from E4M3 rounding",
            "logical_layout": "BF16 and safetensors are [N,K] row-major; GGUF custom carrier planes retain source flattening",
            "selection": "all MXFP8 recipe bindings with a matching rank-2 language-model BF16 tensor",
        },
        "inputs": {
            "bf16": {"root": str(bf_root), "config": compact_config(bf_config), "hashes": bf_hashes, "identity": load_identity(artifact_root / "bf16-source-identity.json")},
            "official_fp8": {"root": str(fp_root), "config": compact_config(fp_config), "hashes": fp_hashes, "identity": load_identity(artifact_root / "fp8-source-identity.json")},
            "sllm_mxfp8": {
                "file": file_identity(mx_path),
                "derived_lock": load_identity(mx_path.with_suffix(".derived-lock.json")),
                "gguf_metadata": {
                    "general_name": reader.fields.get("general.name").contents() if reader.fields.get("general.name") else None,
                    "general_file_type": reader.fields.get("general.file_type").contents() if reader.fields.get("general.file_type") else None,
                    "tensor_count": len(reader.tensors),
                    "recipe_schema": recipe.get("schema_version"),
                    "recipe_encoding_counts": {
                        str(encoding): sum(1 for item in bindings if isinstance(item, dict) and item.get("encoding") == encoding)
                        for encoding in sorted({item.get("encoding") for item in bindings if isinstance(item, dict)})
                    },
                    "semantic_model_id": recipe.get("semantic_model_id"),
                },
            },
        },
        "coverage": {
            "bf16_text_rank2_weight_count": len(text_rank2_names),
            "official_fp8_rank2_quantized_count": len(fp8_name_set),
            "mxfp8_text_rank2_quantized_count": len(all_mx_names),
            "categories": {key: {"count": len(value), "names": value} for key, value in categories.items()},
            "quantization_pair_coverage": {key: {"count": len(value), "names": sorted(value)} for key, value in sorted(pair_coverage.items())},
            "special_inventory": sorted(special_inventory, key=lambda item: item["name"]),
            "official_fp8_dtype_set_seen": sorted(fp8_dtype_set),
            "official_fp8_scale_shape_examples": dict(list(fp8_scale_shapes.items())[:5]),
        },
        "comparison": comparison,
        "grouped_mxfp8": {
            "by_role": {key: sum_metric_groups(value) for key, value in sorted(role_metrics.items())},
            "by_layer": {key: sum_metric_groups(value) for key, value in sorted(per_layer.items(), key=lambda pair: int(pair[0]))},
        },
        "tensors": metrics,
        "runtime_seconds": time.time() - started,
    }
    return report


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"qwen38_weight_error.py: error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.output} ({len(report['tensors'])} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
