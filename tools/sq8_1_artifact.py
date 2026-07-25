#!/usr/bin/env python3
"""Build and verify isolated ``SQ8_1`` signed-int8 block-scale artifacts.

The source contract is deliberately narrow: a verified canonical ``SQ8_0``
artifact is reconstructed row by row, then requantized to the separate
``SQ8_1`` wire format.  It does not modify an SQ8_0 artifact, expand the
public release registry, or infer a different source contract.

The output contains a row-major signed-int8 payload plane and a separate
little-endian FP16 scale plane.  The implementation keeps only one source row,
one padded output row, and that row's scales in memory while packing a tensor.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator

from sq8_canonical_artifact import (
    ArtifactError as Sq8CanonicalArtifactError,
    bf16_bytes_to_f32,
    fp8_e4m3fn_to_f32,
    verify_canonical_artifact,
)


SCHEMA_VERSION = "sq8_1-artifact-v0.1"
ARTIFACT_KIND = "sq8_1_block_int8"
FORMAT_ID = "SQ8_1"
SOURCE_FORMAT_ID = "SQ8_0"
SOURCE_SCHEMA_VERSION = "sq-fp8-artifact-v0.2"
GROUP_SIZE = 32
SCALE_DTYPE = "F16"
ENDIANNESS = "little"
PAYLOAD_DTYPE = "I8"
PAYLOAD_ALIGNMENT_BYTES = 16
MANIFEST_FILE = "sq8_1_manifest.json"
DEFAULT_COPY_CHUNK_BYTES = 4 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
AT_FDCWD = -100
RENAME_NOREPLACE = 1
RENAME_EXCHANGE = 2


class ArtifactError(ValueError):
    """A format or source-contract violation."""


class _ArtifactPromotionError(ArtifactError):
    def __init__(self, message: str, *, preserve_temp: bool) -> None:
        super().__init__(message)
        self.preserve_temp = preserve_temp


@dataclass(frozen=True)
class PackedTensorStats:
    values: int
    blocks: int
    zero_blocks: int
    positive_scale_underflow_count: int
    positive_scale_overflow_count: int
    post_storage_clipping_count: int
    raw_scale_min: float
    raw_scale_max: float
    stored_scale_min: float
    stored_scale_max: float

    def as_manifest(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "blocks": self.blocks,
            "zero_blocks": self.zero_blocks,
            "positive_scale_underflow_count": self.positive_scale_underflow_count,
            "positive_scale_overflow_count": self.positive_scale_overflow_count,
            "post_storage_clipping_count": self.post_storage_clipping_count,
            "raw_scale_min": self.raw_scale_min,
            "raw_scale_max": self.raw_scale_max,
            "stored_scale_min": self.stored_scale_min,
            "stored_scale_max": self.stored_scale_max,
        }


@dataclass(frozen=True)
class Sq8_1Tensor:
    name: str
    rows: int
    cols: int
    payload_row_stride: int
    payload: bytes
    scales_f16_le: bytes

    @property
    def groups_per_row(self) -> int:
        return groups_per_row(self.cols)

    @property
    def actual_bpp(self) -> float:
        return actual_bpp(self.cols, self.payload_row_stride)

    def scale(self, row: int, block: int) -> float:
        if not 0 <= row < self.rows or not 0 <= block < self.groups_per_row:
            raise ArtifactError("SQ8_1 scale index is out of bounds")
        offset = 2 * (row * self.groups_per_row + block)
        return f16_bits_to_f32(int.from_bytes(self.scales_f16_le[offset : offset + 2], "little"))

    def code(self, row: int, col: int) -> int:
        if not 0 <= row < self.rows or not 0 <= col < self.cols:
            raise ArtifactError("SQ8_1 payload index is out of bounds")
        value = self.payload[row * self.payload_row_stride + col]
        return value - 256 if value >= 128 else value

    def reconstruct_row(self, row: int) -> list[float]:
        if not 0 <= row < self.rows:
            raise ArtifactError("SQ8_1 row index is out of bounds")
        return [
            self.code(row, col) * self.scale(row, col // GROUP_SIZE)
            for col in range(self.cols)
        ]


def _is_int(value: Any) -> bool:
    return type(value) is int


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = DEFAULT_COPY_CHUNK_BYTES) -> str:
    if chunk_bytes <= 0:
        raise ArtifactError("SHA-256 chunk size must be greater than zero")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_content_sha256(manifest_without_integrity: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(manifest_without_integrity))


def read_json(path: Path) -> dict[str, Any]:
    def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactError(f"JSON object contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_keys)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def groups_per_row(cols: int) -> int:
    if not _is_int(cols) or cols <= 0:
        raise ArtifactError("SQ8_1 cols must be a positive integer")
    return (cols + GROUP_SIZE - 1) // GROUP_SIZE


def payload_row_stride(cols: int) -> int:
    if not _is_int(cols) or cols <= 0:
        raise ArtifactError("SQ8_1 cols must be a positive integer")
    return ((cols + PAYLOAD_ALIGNMENT_BYTES - 1) // PAYLOAD_ALIGNMENT_BYTES) * PAYLOAD_ALIGNMENT_BYTES


def actual_bpp(cols: int, row_stride: int | None = None) -> float:
    stride = payload_row_stride(cols) if row_stride is None else row_stride
    if stride != payload_row_stride(cols):
        raise ArtifactError("SQ8_1 row stride does not match the canonical alignment rule")
    return 8.0 * (stride + 2 * groups_per_row(cols)) / cols


def f16_bits_to_f32(bits: int) -> float:
    if not _is_int(bits) or not 0 <= bits <= 0xFFFF:
        raise ArtifactError("FP16 bits must be an unsigned 16-bit integer")
    sign = -1.0 if bits & 0x8000 else 1.0
    exponent = (bits >> 10) & 0x1F
    mantissa = bits & 0x03FF
    if exponent == 0:
        return math.copysign(0.0, sign) if mantissa == 0 else sign * math.ldexp(float(mantissa), -24)
    if exponent == 0x1F:
        return math.copysign(math.inf, sign) if mantissa == 0 else math.nan
    return sign * math.ldexp(1.0 + float(mantissa) / 1024.0, exponent - 15)


def f32(value: float) -> float:
    """Round ``value`` to the canonical host-side binary32 domain.

    SQ8_1 derives its scale from F32 weights.  Python evaluates arithmetic in
    binary64, so make the boundary explicit before both scale derivation and
    code generation; otherwise a value infinitesimally above an F16 boundary
    could be packed differently from the runtime's F32 reference.
    """

    try:
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except OverflowError as exc:
        raise ArtifactError(f"SQ8_1 source weight is outside the F32 domain: {value!r}") from exc


def _round_shift_ties_even(value: int, shift: int) -> int:
    if shift <= 0:
        return value << (-shift)
    quotient = value >> shift
    remainder = value & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and quotient & 1):
        return quotient + 1
    return quotient


def f32_to_f16_bits_rne(value: float) -> int:
    """IEEE binary16 conversion with nearest-even rounding, without NumPy/Torch."""

    if not isinstance(value, float):
        value = float(value)
    raw = struct.unpack("<I", struct.pack("<f", value))[0]
    sign = (raw >> 16) & 0x8000
    exponent_bits = (raw >> 23) & 0xFF
    mantissa = raw & 0x7FFFFF
    if exponent_bits == 0xFF:
        return sign | (0x7C00 if mantissa == 0 else 0x7E00)
    exponent = exponent_bits - 127
    if exponent > 15:
        return sign | 0x7C00
    if exponent >= -14:
        rounded = _round_shift_ties_even(mantissa, 13)
        half_exponent = exponent + 15
        if rounded == 0x400:
            rounded = 0
            half_exponent += 1
            if half_exponent >= 0x1F:
                return sign | 0x7C00
        return sign | (half_exponent << 10) | rounded
    if exponent < -25:
        return sign
    significand = mantissa | 0x800000
    subnormal = _round_shift_ties_even(significand, -exponent - 1)
    return sign | subnormal


def ceil_fp16(value: float) -> int:
    """Return the least finite, positive IEEE binary16 value >= ``value``."""

    if not math.isfinite(value) or value <= 0.0:
        raise ArtifactError("ceil_fp16 requires a finite positive value")
    rounded = f32_to_f16_bits_rne(float(value))
    if rounded == 0:
        return 0x0001
    if rounded >= 0x7C00:
        raise ArtifactError(f"FP16 scale overflow for raw scale {value!r}")
    stored = f16_bits_to_f32(rounded)
    if stored < value:
        rounded += 1
        if rounded >= 0x7C00:
            raise ArtifactError(f"FP16 scale overflow for raw scale {value!r}")
        stored = f16_bits_to_f32(rounded)
    if not math.isfinite(stored) or stored <= 0.0 or stored < value:
        raise ArtifactError(f"could not encode finite positive ceil-FP16 scale {value!r}")
    return rounded


def _round_ties_even(value: float) -> int:
    if not math.isfinite(value):
        raise ArtifactError("cannot round a non-finite quantization ratio")
    return int(round(value))


def _quantize_block(values: list[float]) -> tuple[list[int], int, float, bool, int]:
    if not values:
        raise ArtifactError("SQ8_1 block must not be empty")
    values = [f32(value) for value in values]
    if any(not math.isfinite(value) for value in values):
        raise ArtifactError("SQ8_1 source weights must be finite")
    maximum = max(abs(value) for value in values)
    if maximum == 0.0:
        return [0] * len(values), 0x3C00, 0.0, False, 0
    raw_scale = f32(maximum / 127.0)
    underflow = raw_scale < math.ldexp(1.0, -24)
    try:
        scale_bits = ceil_fp16(raw_scale)
    except ArtifactError as exc:
        raise ArtifactError(str(exc)) from exc
    scale = f16_bits_to_f32(scale_bits)
    clipping = 0
    codes: list[int] = []
    for value in values:
        ratio = value / scale
        if ratio < -127.0 or ratio > 127.0:
            clipping += 1
        code = max(-127, min(127, _round_ties_even(ratio)))
        if code == -128:
            raise AssertionError("SQ8_1 must never emit code -128")
        codes.append(code)
    return codes, scale_bits, raw_scale, underflow, clipping


def pack_tensor_from_rows(
    name: str,
    rows: int,
    cols: int,
    source_rows: Iterable[list[float]],
    payload_handle: BinaryIO,
    scale_handle: BinaryIO,
) -> PackedTensorStats:
    """Pack a matrix from an iterator without retaining prior rows in memory."""

    if not isinstance(name, str) or not name:
        raise ArtifactError("SQ8_1 tensor name must be non-empty")
    if not _is_int(rows) or rows <= 0:
        raise ArtifactError("SQ8_1 rows must be a positive integer")
    stride = payload_row_stride(cols)
    group_count = groups_per_row(cols)
    iterator = iter(source_rows)
    blocks = 0
    zero_blocks = 0
    underflows = 0
    clipping = 0
    raw_scale_min = math.inf
    raw_scale_max = 0.0
    stored_scale_min = math.inf
    stored_scale_max = 0.0
    for row_index in range(rows):
        try:
            row = next(iterator)
        except StopIteration as exc:
            raise ArtifactError(f"SQ8_1 source tensor {name} ended before row {row_index}") from exc
        if len(row) != cols:
            raise ArtifactError(
                f"SQ8_1 source tensor {name} row {row_index} has {len(row)} columns, expected {cols}"
            )
        payload = bytearray(stride)
        scales = bytearray(group_count * 2)
        for block in range(group_count):
            start = block * GROUP_SIZE
            stop = min(start + GROUP_SIZE, cols)
            codes, scale_bits, raw_scale, underflow, block_clipping = _quantize_block(row[start:stop])
            payload[start:stop] = bytes(code & 0xFF for code in codes)
            scales[2 * block : 2 * block + 2] = scale_bits.to_bytes(2, "little")
            blocks += 1
            zero_blocks += int(raw_scale == 0.0)
            underflows += int(underflow)
            clipping += block_clipping
            raw_scale_min = min(raw_scale_min, raw_scale)
            raw_scale_max = max(raw_scale_max, raw_scale)
            stored_scale = f16_bits_to_f32(scale_bits)
            stored_scale_min = min(stored_scale_min, stored_scale)
            stored_scale_max = max(stored_scale_max, stored_scale)
        if any(payload[cols:]):
            raise AssertionError("SQ8_1 packer produced nonzero physical tail padding")
        payload_handle.write(payload)
        scale_handle.write(scales)
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ArtifactError(f"SQ8_1 source tensor {name} has more than {rows} rows")
    return PackedTensorStats(
        values=rows * cols,
        blocks=blocks,
        zero_blocks=zero_blocks,
        positive_scale_underflow_count=underflows,
        positive_scale_overflow_count=0,
        post_storage_clipping_count=clipping,
        raw_scale_min=0.0 if raw_scale_min == math.inf else raw_scale_min,
        raw_scale_max=raw_scale_max,
        stored_scale_min=0.0 if stored_scale_min == math.inf else stored_scale_min,
        stored_scale_max=stored_scale_max,
    )


def pack_tensor_from_values(name: str, rows: int, cols: int, values: Iterable[float]) -> Sq8_1Tensor:
    source = list(values)
    if len(source) != rows * cols:
        raise ArtifactError(
            f"SQ8_1 source tensor {name} has {len(source)} values, expected {rows * cols}"
        )
    import io

    payload = io.BytesIO()
    scales = io.BytesIO()
    pack_tensor_from_rows(
        name,
        rows,
        cols,
        (source[row * cols : (row + 1) * cols] for row in range(rows)),
        payload,
        scales,
    )
    result = Sq8_1Tensor(
        name=name,
        rows=rows,
        cols=cols,
        payload_row_stride=payload_row_stride(cols),
        payload=payload.getvalue(),
        scales_f16_le=scales.getvalue(),
    )
    validate_tensor(result)
    return result


def quantize_activation(values: Iterable[float]) -> tuple[bytes, bytes]:
    """Quantize one logical activation vector using the exact SQ8_1 rule."""

    source = list(values)
    if not source:
        raise ArtifactError("SQ8_1 activation vector must not be empty")
    codes = bytearray(len(source))
    scales = bytearray(groups_per_row(len(source)) * 2)
    for block in range(groups_per_row(len(source))):
        start = block * GROUP_SIZE
        stop = min(start + GROUP_SIZE, len(source))
        block_codes, scale_bits, _, _, _ = _quantize_block(source[start:stop])
        codes[start:stop] = bytes(code & 0xFF for code in block_codes)
        scales[2 * block : 2 * block + 2] = scale_bits.to_bytes(2, "little")
    return bytes(codes), bytes(scales)


def _validate_finite_input(values: Iterable[float], label: str) -> list[float]:
    result = [float(value) for value in values]
    if any(not math.isfinite(value) for value in result):
        raise ArtifactError(f"{label} contains non-finite values")
    return result


def matvec_w8a16(tensor: Sq8_1Tensor, activation: Iterable[float]) -> list[float]:
    """Default SQ8_1 reference path: i8-to-float weights and F32 activation."""

    validate_tensor(tensor)
    vector = _validate_finite_input(activation, "SQ8_1 W8A16 activation")
    if len(vector) != tensor.cols:
        raise ArtifactError("SQ8_1 W8A16 activation length does not match tensor cols")
    result: list[float] = []
    for row in range(tensor.rows):
        total = 0.0
        for block in range(tensor.groups_per_row):
            start = block * GROUP_SIZE
            stop = min(start + GROUP_SIZE, tensor.cols)
            partial = 0.0
            for col in range(start, stop):
                partial += float(tensor.code(row, col)) * vector[col]
            total += partial * tensor.scale(row, block)
        result.append(total)
    return result


def matvec_w8a8_explicit(tensor: Sq8_1Tensor, activation: Iterable[float]) -> list[float]:
    """Explicit-only W8A8 reference path using K=32 signed int32 partial dots."""

    validate_tensor(tensor)
    vector = _validate_finite_input(activation, "SQ8_1 W8A8 activation")
    if len(vector) != tensor.cols:
        raise ArtifactError("SQ8_1 W8A8 activation length does not match tensor cols")
    activation_codes, activation_scales = quantize_activation(vector)
    result: list[float] = []
    for row in range(tensor.rows):
        total = 0.0
        for block in range(tensor.groups_per_row):
            start = block * GROUP_SIZE
            stop = min(start + GROUP_SIZE, tensor.cols)
            dot = 0
            for col in range(start, stop):
                activation_code = activation_codes[col]
                if activation_code >= 128:
                    activation_code -= 256
                dot += tensor.code(row, col) * activation_code
            activation_scale = f16_bits_to_f32(
                int.from_bytes(activation_scales[2 * block : 2 * block + 2], "little")
            )
            total += float(dot) * tensor.scale(row, block) * activation_scale
        result.append(total)
    return result


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactError(f"{label} must be an object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ArtifactError(f"{label} must be lowercase hexadecimal SHA-256")
    return value


def _safe_artifact_path(artifact_dir: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ArtifactError(f"{label} must be a non-empty relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ArtifactError(f"{label} must stay inside the artifact directory")
    path = artifact_dir / candidate
    if path.is_symlink() or not path.is_file():
        raise ArtifactError(f"{label} must be a regular, non-symlink file")
    resolved = path.resolve()
    if artifact_dir.resolve() not in resolved.parents:
        raise ArtifactError(f"{label} escapes the artifact directory")
    return resolved


def _require_positive_shape(value: Any, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not _is_int(item) or item <= 0 for item in value)
    ):
        raise ArtifactError(f"{label} must contain two positive integers")
    return value[0], value[1]


def _verify_plane(
    artifact_dir: Path,
    value: Any,
    label: str,
    expected_bytes: int,
) -> Path:
    plane = _require_dict(value, label)
    path = _safe_artifact_path(artifact_dir, plane.get("file"), f"{label}.file")
    byte_count = plane.get("bytes")
    if not _is_int(byte_count) or byte_count != expected_bytes:
        raise ArtifactError(f"{label}.bytes must equal {expected_bytes}")
    if path.stat().st_size != byte_count:
        raise ArtifactError(f"{label} file length does not match its manifest")
    digest = _require_sha256(plane.get("sha256"), f"{label}.sha256")
    if sha256_file(path) != digest:
        raise ArtifactError(f"{label} SHA-256 mismatch")
    return path


def _validate_tensor_files(
    payload_path: Path,
    scale_path: Path,
    rows: int,
    cols: int,
    stride: int,
) -> None:
    group_count = groups_per_row(cols)
    with payload_path.open("rb") as handle:
        for row in range(rows):
            payload = handle.read(stride)
            if len(payload) != stride:
                raise ArtifactError(f"SQ8_1 payload is truncated at row {row}")
            if any(value == 0x80 for value in payload[:cols]):
                raise ArtifactError("SQ8_1 payload contains forbidden int8 code -128")
            if any(payload[cols:]):
                raise ArtifactError("SQ8_1 payload physical tail padding is nonzero")
    with scale_path.open("rb") as handle:
        for scale_index in range(rows * group_count):
            raw = handle.read(2)
            if len(raw) != 2:
                raise ArtifactError(f"SQ8_1 scale is truncated at index {scale_index}")
            scale = f16_bits_to_f32(int.from_bytes(raw, "little"))
            if not math.isfinite(scale) or scale <= 0.0:
                raise ArtifactError("SQ8_1 scale must be finite and strictly positive")


def validate_tensor(tensor: Sq8_1Tensor) -> None:
    if tensor.rows <= 0 or tensor.cols <= 0:
        raise ArtifactError("SQ8_1 tensor shape must be positive")
    stride = payload_row_stride(tensor.cols)
    if tensor.payload_row_stride != stride:
        raise ArtifactError("SQ8_1 payload row stride violates the canonical 16-byte rule")
    if len(tensor.payload) != tensor.rows * stride:
        raise ArtifactError("SQ8_1 payload length does not match the tensor shape")
    if len(tensor.scales_f16_le) != tensor.rows * groups_per_row(tensor.cols) * 2:
        raise ArtifactError("SQ8_1 scale length does not match the tensor shape")
    for row in range(tensor.rows):
        payload = tensor.payload[row * stride : (row + 1) * stride]
        if any(value == 0x80 for value in payload[: tensor.cols]):
            raise ArtifactError("SQ8_1 payload contains forbidden int8 code -128")
        if any(payload[tensor.cols :]):
            raise ArtifactError("SQ8_1 payload physical tail padding is nonzero")
    for offset in range(0, len(tensor.scales_f16_le), 2):
        scale = f16_bits_to_f32(int.from_bytes(tensor.scales_f16_le[offset : offset + 2], "little"))
        if not math.isfinite(scale) or scale <= 0.0:
            raise ArtifactError("SQ8_1 scale must be finite and strictly positive")


def _source_tensor_rows(
    artifact_dir: Path,
    entry: dict[str, Any],
) -> Iterator[list[float]]:
    shape = entry.get("shape")
    if not isinstance(shape, list) or len(shape) != 2 or any(not _is_int(dim) or dim <= 0 for dim in shape):
        raise ArtifactError("canonical SQ8_0 source tensor shape is invalid")
    rows, cols = shape
    weight = _require_dict(entry.get("weight"), "canonical SQ8_0 weight")
    scale = _require_dict(entry.get("scale"), "canonical SQ8_0 scale")
    weight_path = _safe_artifact_path(artifact_dir, weight.get("file"), "canonical SQ8_0 weight.file")
    scale_path = _safe_artifact_path(artifact_dir, scale.get("file"), "canonical SQ8_0 scale.file")
    block_shape = scale.get("block_shape")
    if block_shape != [128, 128]:
        raise ArtifactError("canonical SQ8_0 source tensor must use 128x128 scales")
    scale_shape = scale.get("shape")
    expected_scale_shape = [(rows + 127) // 128, (cols + 127) // 128]
    if scale_shape != expected_scale_shape:
        raise ArtifactError("canonical SQ8_0 source scale shape is invalid")
    raw_scales = scale_path.read_bytes()
    if len(raw_scales) != expected_scale_shape[0] * expected_scale_shape[1] * 2:
        raise ArtifactError("canonical SQ8_0 source scale payload length is invalid")
    scales = bf16_bytes_to_f32(raw_scales)
    if any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise ArtifactError("canonical SQ8_0 source scales must be finite and positive")
    if weight_path.stat().st_size != rows * cols:
        raise ArtifactError("canonical SQ8_0 source payload length is invalid")
    scale_cols = expected_scale_shape[1]
    with weight_path.open("rb") as handle:
        for row in range(rows):
            raw = handle.read(cols)
            if len(raw) != cols:
                raise ArtifactError(f"canonical SQ8_0 source payload is truncated at row {row}")
            scale_row = (row // 128) * scale_cols
            yield [
                f32(fp8_e4m3fn_to_f32(value) * scales[scale_row + col // 128])
                for col, value in enumerate(raw)
            ]


def _safe_tensor_file_stem(index: int) -> str:
    return f"{index:05d}"


def _directory_identity(path: Path, label: str) -> tuple[int, int]:
    try:
        result = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArtifactError(f"failed to stat {label}: {path}: {exc}") from exc
    if stat.S_ISLNK(result.st_mode) or not stat.S_ISDIR(result.st_mode):
        raise ArtifactError(f"{label} must be a non-symlink directory: {path}")
    return result.st_dev, result.st_ino


def _renameat2(source: Path, destination: Path, flags: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.renameat2
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    status = function(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        flags,
    )
    if status != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _rename_noreplace(source: Path, destination: Path) -> None:
    try:
        _renameat2(source, destination, RENAME_NOREPLACE)
    except AttributeError:
        if destination.exists():
            raise ArtifactError(f"output artifact already exists: {destination}")
        try:
            os.rename(source, destination)
        except OSError as exc:
            raise ArtifactError(f"failed to promote artifact: {exc}") from exc
    except OSError as exc:
        if exc.errno == 17:
            raise ArtifactError(f"output artifact already exists: {destination}") from exc
        raise ArtifactError(f"failed to promote artifact: {exc}") from exc


def _remove_owned_directory(path: Path, identity: tuple[int, int], label: str) -> None:
    try:
        if _directory_identity(path, label) != identity:
            raise ArtifactError(f"refusing to remove {label}; identity changed: {path}")
    except FileNotFoundError:
        return
    shutil.rmtree(path)


def _promote_artifact(temp_dir: Path, output_dir: Path, overwrite: bool, expected_identity: tuple[int, int] | None) -> None:
    if expected_identity is None:
        if os.path.lexists(output_dir):
            raise ArtifactError(f"output artifact appeared during build: {output_dir}")
        _rename_noreplace(temp_dir, output_dir)
        return
    if not overwrite:
        raise ArtifactError(f"output artifact already exists: {output_dir}")
    if _directory_identity(output_dir, "existing SQ8_1 output artifact") != expected_identity:
        raise ArtifactError("existing SQ8_1 output artifact changed during build")
    try:
        _renameat2(temp_dir, output_dir, RENAME_EXCHANGE)
    except (AttributeError, OSError) as exc:
        raise _ArtifactPromotionError(
            f"atomic SQ8_1 overwrite exchange failed: {exc}", preserve_temp=True
        ) from exc
    try:
        _remove_owned_directory(temp_dir, expected_identity, "exchanged previous SQ8_1 artifact")
    except Exception as exc:  # noqa: BLE001 - preserve a recoverable exchanged artifact.
        raise _ArtifactPromotionError(
            f"SQ8_1 artifact promoted but prior exchange cleanup failed at {temp_dir}: {exc}",
            preserve_temp=True,
        ) from exc


def build_sq8_1_artifact(
    source_sq8_0_artifact: Path,
    output_artifact: Path,
    *,
    tensor_names: Iterable[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Requantize a verified SQ8_0 artifact into a fresh, isolated SQ8_1 artifact."""

    source = source_sq8_0_artifact.resolve()
    output = Path(os.path.abspath(output_artifact)).resolve(strict=False)
    if output.is_symlink():
        raise ArtifactError(f"output artifact path must not be a symlink: {output}")
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ArtifactError("SQ8_0 source and SQ8_1 output must not contain one another")
    try:
        verify_canonical_artifact(source)
    except Sq8CanonicalArtifactError as exc:
        raise ArtifactError(f"SQ8_0 source artifact is not verified: {exc}") from exc
    source_manifest = read_json(source / "sq_manifest.json")
    if source_manifest.get("schema_version") != SOURCE_SCHEMA_VERSION or source_manifest.get("format_id") != SOURCE_FORMAT_ID:
        raise ArtifactError("SQ8_1 source contract requires SQ8_0 canonical artifact v0.2")
    entries = source_manifest.get("quantized_tensors")
    if not isinstance(entries, list) or not entries:
        raise ArtifactError("SQ8_0 source artifact has no quantized tensor entries")
    by_name = {entry.get("name"): entry for entry in entries if isinstance(entry, dict)}
    if len(by_name) != len(entries) or any(not isinstance(name, str) or not name for name in by_name):
        raise ArtifactError("SQ8_0 source tensor names are invalid")
    if tensor_names is None:
        selected_names = sorted(by_name)
    else:
        selected_names = sorted(tensor_names)
        if not selected_names or len(set(selected_names)) != len(selected_names):
            raise ArtifactError("SQ8_1 tensor selection must be non-empty and unique")
        missing = sorted(set(selected_names) - set(by_name))
        if missing:
            raise ArtifactError(f"SQ8_1 requested source tensors are absent: {missing}")
    expected_identity: tuple[int, int] | None = None
    if output.exists():
        if not output.is_dir():
            raise ArtifactError(f"existing SQ8_1 output is not a directory: {output}")
        if not overwrite:
            raise ArtifactError(f"output artifact already exists: {output}")
        expected_identity = _directory_identity(output, "existing SQ8_1 output artifact")
        verify_sq8_1_artifact(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent))
    temp_identity = _directory_identity(temp_dir, "new SQ8_1 temporary artifact")
    try:
        (temp_dir / "payload").mkdir()
        (temp_dir / "scales").mkdir()
        manifest_entries: list[dict[str, Any]] = []
        total_payload_bytes = 0
        total_scale_bytes = 0
        for index, name in enumerate(selected_names):
            entry = by_name[name]
            assert isinstance(entry, dict)
            shape = _require_positive_shape(entry.get("shape"), f"SQ8_0 source tensor {name}.shape")
            rows, cols = shape
            stem = _safe_tensor_file_stem(index)
            payload_relative = f"payload/{stem}.i8"
            scales_relative = f"scales/{stem}.f16"
            payload_path = temp_dir / payload_relative
            scales_path = temp_dir / scales_relative
            with payload_path.open("wb") as payload_handle, scales_path.open("wb") as scale_handle:
                stats = pack_tensor_from_rows(
                    name,
                    rows,
                    cols,
                    _source_tensor_rows(source, entry),
                    payload_handle,
                    scale_handle,
                )
            stride = payload_row_stride(cols)
            expected_payload_bytes = rows * stride
            expected_scale_bytes = rows * groups_per_row(cols) * 2
            if payload_path.stat().st_size != expected_payload_bytes or scales_path.stat().st_size != expected_scale_bytes:
                raise ArtifactError(f"SQ8_1 packer emitted an unexpected plane length for {name}")
            manifest_entries.append(
                {
                    "name": name,
                    "shape": [rows, cols],
                    "elements": rows * cols,
                    "payload": {
                        "file": payload_relative,
                        "dtype": PAYLOAD_DTYPE,
                        "bytes": expected_payload_bytes,
                        "sha256": sha256_file(payload_path),
                        "row_stride": stride,
                        "alignment_bytes": PAYLOAD_ALIGNMENT_BYTES,
                    },
                    "scale": {
                        "file": scales_relative,
                        "dtype": SCALE_DTYPE,
                        "bytes": expected_scale_bytes,
                        "sha256": sha256_file(scales_path),
                        "shape": [rows, groups_per_row(cols)],
                        "order": "row_major",
                    },
                    "storage": {
                        "nominal_full_block_bpp": 8.5,
                        "actual_bpp": actual_bpp(cols, stride),
                    },
                    "quantization": stats.as_manifest(),
                }
            )
            total_payload_bytes += expected_payload_bytes
            total_scale_bytes += expected_scale_bytes
        source_manifest_sha256 = sha256_file(source / "sq_manifest.json")
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "format_id": FORMAT_ID,
            "endianness": ENDIANNESS,
            "group_size": GROUP_SIZE,
            "source": {
                "format_id": SOURCE_FORMAT_ID,
                "schema_version": SOURCE_SCHEMA_VERSION,
                "artifact": str(source),
                "manifest_sha256": source_manifest_sha256,
                "contract": "reconstructed_row_major_f32_from_verified_sq8_0_canonical",
            },
            "storage": {
                "payload_bytes": total_payload_bytes,
                "scale_bytes": total_scale_bytes,
                "total_bytes": total_payload_bytes + total_scale_bytes,
            },
            "tensors": manifest_entries,
        }
        manifest["integrity"] = {"content_sha256": artifact_content_sha256(manifest)}
        write_json(temp_dir / MANIFEST_FILE, manifest)
        verify_sq8_1_artifact(temp_dir)
        _promote_artifact(temp_dir, output, overwrite, expected_identity)
        temp_dir = None  # type: ignore[assignment]
        return manifest
    finally:
        if temp_dir is not None:
            _remove_owned_directory(temp_dir, temp_identity, "new SQ8_1 temporary artifact")


def verify_sq8_1_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact = artifact_dir.resolve()
    manifest_path = artifact / MANIFEST_FILE
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ArtifactError(f"SQ8_1 manifest must be a regular file: {manifest_path}")
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ArtifactError("SQ8_1 manifest exceeds the maximum allowed size")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(f"SQ8_1 schema_version must be {SCHEMA_VERSION}")
    if manifest.get("artifact_kind") != ARTIFACT_KIND:
        raise ArtifactError(f"SQ8_1 artifact_kind must be {ARTIFACT_KIND}")
    if manifest.get("format_id") != FORMAT_ID:
        raise ArtifactError(f"SQ8_1 format_id must be {FORMAT_ID}")
    if manifest.get("endianness") != ENDIANNESS or manifest.get("group_size") != GROUP_SIZE:
        raise ArtifactError("SQ8_1 endianness or group_size contract is invalid")
    integrity = _require_dict(manifest.get("integrity"), "SQ8_1 integrity")
    expected_content = _require_sha256(integrity.get("content_sha256"), "SQ8_1 integrity.content_sha256")
    without_integrity = dict(manifest)
    del without_integrity["integrity"]
    if artifact_content_sha256(without_integrity) != expected_content:
        raise ArtifactError("SQ8_1 manifest content SHA-256 mismatch")
    source = _require_dict(manifest.get("source"), "SQ8_1 source")
    if source.get("format_id") != SOURCE_FORMAT_ID or source.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ArtifactError("SQ8_1 source contract must identify verified SQ8_0 v0.2")
    _require_sha256(source.get("manifest_sha256"), "SQ8_1 source.manifest_sha256")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ArtifactError("SQ8_1 tensors must be a non-empty list")
    names: set[str] = set()
    files: set[str] = set()
    payload_total = 0
    scale_total = 0
    previous_name = ""
    for index, raw_entry in enumerate(tensors):
        entry = _require_dict(raw_entry, f"SQ8_1 tensors[{index}]")
        name = entry.get("name")
        if not isinstance(name, str) or not name or name <= previous_name or name in names:
            raise ArtifactError("SQ8_1 tensor names must be unique and sorted")
        previous_name = name
        names.add(name)
        rows, cols = _require_positive_shape(entry.get("shape"), f"SQ8_1 tensor {name}.shape")
        if entry.get("elements") != rows * cols:
            raise ArtifactError(f"SQ8_1 tensor {name}.elements is invalid")
        payload = _require_dict(entry.get("payload"), f"SQ8_1 tensor {name}.payload")
        scale = _require_dict(entry.get("scale"), f"SQ8_1 tensor {name}.scale")
        stride = payload.get("row_stride")
        if not _is_int(stride) or stride != payload_row_stride(cols):
            raise ArtifactError(f"SQ8_1 tensor {name} payload_row_stride is invalid")
        if payload.get("dtype") != PAYLOAD_DTYPE or payload.get("alignment_bytes") != PAYLOAD_ALIGNMENT_BYTES:
            raise ArtifactError(f"SQ8_1 tensor {name} payload contract is invalid")
        if scale.get("dtype") != SCALE_DTYPE or scale.get("shape") != [rows, groups_per_row(cols)] or scale.get("order") != "row_major":
            raise ArtifactError(f"SQ8_1 tensor {name} scale contract is invalid")
        expected_payload_bytes = rows * stride
        expected_scale_bytes = rows * groups_per_row(cols) * 2
        payload_path = _verify_plane(artifact, payload, f"SQ8_1 tensor {name}.payload", expected_payload_bytes)
        scale_path = _verify_plane(artifact, scale, f"SQ8_1 tensor {name}.scale", expected_scale_bytes)
        for relative in (payload.get("file"), scale.get("file")):
            assert isinstance(relative, str)
            if relative in files:
                raise ArtifactError("SQ8_1 payload and scale files must be unique")
            files.add(relative)
        storage = _require_dict(entry.get("storage"), f"SQ8_1 tensor {name}.storage")
        if storage.get("nominal_full_block_bpp") != 8.5 or not math.isclose(storage.get("actual_bpp", -1.0), actual_bpp(cols, stride), rel_tol=0.0, abs_tol=0.0):
            raise ArtifactError(f"SQ8_1 tensor {name} storage bpp accounting is invalid")
        stats = _require_dict(entry.get("quantization"), f"SQ8_1 tensor {name}.quantization")
        if stats.get("values") != rows * cols or stats.get("blocks") != rows * groups_per_row(cols):
            raise ArtifactError(f"SQ8_1 tensor {name} quantization accounting is invalid")
        if stats.get("post_storage_clipping_count") != 0:
            raise ArtifactError(f"SQ8_1 tensor {name} records post-storage clipping")
        _validate_tensor_files(payload_path, scale_path, rows, cols, stride)
        payload_total += expected_payload_bytes
        scale_total += expected_scale_bytes
    storage = _require_dict(manifest.get("storage"), "SQ8_1 storage")
    if storage.get("payload_bytes") != payload_total or storage.get("scale_bytes") != scale_total or storage.get("total_bytes") != payload_total + scale_total:
        raise ArtifactError("SQ8_1 aggregate storage accounting is invalid")
    return {
        "format_id": FORMAT_ID,
        "schema_version": SCHEMA_VERSION,
        "tensor_count": len(tensors),
        "payload_bytes": payload_total,
        "scale_bytes": scale_total,
        "total_bytes": payload_total + scale_total,
        "content_sha256": expected_content,
    }


def read_sq8_1_tensor(artifact_dir: Path, tensor_name: str) -> Sq8_1Tensor:
    verify_sq8_1_artifact(artifact_dir)
    manifest = read_json(artifact_dir.resolve() / MANIFEST_FILE)
    for entry in manifest["tensors"]:
        assert isinstance(entry, dict)
        if entry.get("name") != tensor_name:
            continue
        rows, cols = _require_positive_shape(entry["shape"], f"SQ8_1 tensor {tensor_name}.shape")
        payload = _require_dict(entry["payload"], f"SQ8_1 tensor {tensor_name}.payload")
        scale = _require_dict(entry["scale"], f"SQ8_1 tensor {tensor_name}.scale")
        payload_path = _safe_artifact_path(artifact_dir.resolve(), payload["file"], "SQ8_1 payload.file")
        scale_path = _safe_artifact_path(artifact_dir.resolve(), scale["file"], "SQ8_1 scale.file")
        tensor = Sq8_1Tensor(
            name=tensor_name,
            rows=rows,
            cols=cols,
            payload_row_stride=payload["row_stride"],
            payload=payload_path.read_bytes(),
            scales_f16_le=scale_path.read_bytes(),
        )
        validate_tensor(tensor)
        return tensor
    raise ArtifactError(f"SQ8_1 tensor is absent from artifact: {tensor_name}")
