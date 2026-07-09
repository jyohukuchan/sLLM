"""Source-correct SQ8_0 canonical artifact import and verification."""

from __future__ import annotations

import hashlib
import ctypes
import json
import math
import os
import re
import shutil
import stat
import struct
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "sq-fp8-artifact-v0.2"
ARTIFACT_KIND = "canonical"
FORMAT_ID = "SQ8_0"
IMPORT_MODE = "fp8_checkpoint"
RAW_ENCODING = "raw_safetensors_payload"
WEIGHT_DTYPE = "F8_E4M3"
SCALE_DTYPE = "BF16"
SCALE_LAYOUT = "block_2d"
SCALE_ORDER = "row_major"
SCALE_SEMANTIC = "dequant_multiplier"
DEFAULT_COPY_CHUNK_BYTES = 64 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_BLOCK_SHAPE = (128, 128)
AT_FDCWD = -100
RENAME_NOREPLACE = 1
RENAME_EXCHANGE = 2


class ArtifactError(ValueError):
    pass


class _ArtifactPromotionError(ArtifactError):
    def __init__(self, message: str, *, preserve_temp: bool) -> None:
        super().__init__(message)
        self.preserve_temp = preserve_temp


def _is_int(value: Any) -> bool:
    return type(value) is int


@dataclass(frozen=True)
class TensorRegion:
    name: str
    source_file: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int

    @property
    def elements(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class SourceContract:
    config_file: str
    config_sha256: str
    index_file: str | None
    index_sha256: str | None
    quant_method: str
    format: str
    activation_scheme: str
    weight_block_shape: tuple[int, int]


@dataclass(frozen=True)
class WeightScalePair:
    weight: TensorRegion
    scale: TensorRegion


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ArtifactError(f"JSON object contains duplicate key: {key}")
        payload[key] = value
    return payload


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"JSON root must be an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_content_sha256(manifest_without_integrity: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(manifest_without_integrity))


def dtype_size(dtype: str) -> int:
    sizes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    try:
        return sizes[dtype]
    except KeyError as exc:
        raise ArtifactError(f"unsupported safetensors dtype: {dtype}") from exc


def parse_safetensors_header(path: Path) -> list[TensorRegion]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_header_len = handle.read(8)
        if len(raw_header_len) != 8:
            raise ArtifactError(f"safetensors file is missing header length: {path}")
        header_len = int.from_bytes(raw_header_len, "little", signed=False)
        if header_len <= 0 or header_len > MAX_SAFETENSORS_HEADER_BYTES:
            raise ArtifactError(f"invalid safetensors header length {header_len}: {path}")
        header_bytes = handle.read(header_len)
        if len(header_bytes) != header_len:
            raise ArtifactError(f"truncated safetensors header: {path}")
    try:
        header = json.loads(
            header_bytes,
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid safetensors header JSON {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ArtifactError(f"safetensors header root must be an object: {path}")

    data_base = 8 + header_len
    data_length = file_size - data_base
    regions_with_offsets: list[tuple[int, int, TensorRegion]] = []
    for name, raw in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(raw, dict):
            raise ArtifactError(f"tensor metadata must be an object: {path}:{name}")
        dtype = raw.get("dtype")
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        if not isinstance(dtype, str):
            raise ArtifactError(f"tensor dtype is missing: {path}:{name}")
        if not isinstance(shape, list) or any(
            not _is_int(dim) or dim < 0 for dim in shape
        ):
            raise ArtifactError(f"tensor shape is invalid: {path}:{name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not _is_int(value) for value in offsets)
        ):
            raise ArtifactError(f"tensor data_offsets are invalid: {path}:{name}")
        start, end = offsets
        if start < 0 or end < start or end > data_length:
            raise ArtifactError(f"tensor data_offsets are out of bounds: {path}:{name}")
        expected_length = math.prod(shape) * dtype_size(dtype)
        if end - start != expected_length:
            raise ArtifactError(
                f"tensor byte length mismatch for {name}: "
                f"expected {expected_length} got {end - start}"
            )
        region = TensorRegion(
            name=name,
            source_file=path,
            dtype=dtype,
            shape=tuple(shape),
            offset=data_base + start,
            length=end - start,
        )
        regions_with_offsets.append((start, end, region))

    cursor = 0
    for start, end, region in sorted(
        regions_with_offsets,
        key=lambda item: (item[0], item[1], item[2].name),
    ):
        if start < cursor:
            raise ArtifactError(
                f"safetensors tensor data regions overlap at {region.name}: "
                f"start={start} previous_end={cursor}"
            )
        if start > cursor:
            raise ArtifactError(
                f"safetensors tensor data regions contain a gap before {region.name}: "
                f"start={start} expected={cursor}"
            )
        cursor = end
    if cursor != data_length:
        raise ArtifactError(
            "safetensors tensor data regions do not cover the complete data buffer: "
            f"covered={cursor} data_length={data_length}"
        )
    return sorted((item[2] for item in regions_with_offsets), key=lambda item: item.name)


def _source_shard_path(model_dir: Path, raw_name: Any) -> Path:
    if not isinstance(raw_name, str) or not raw_name:
        raise ArtifactError(f"source shard name must be a non-empty string: {raw_name!r}")
    relative = Path(raw_name)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != raw_name:
        raise ArtifactError(f"source shard must be a basename inside the model directory: {raw_name}")
    path = model_dir / relative
    if path.is_symlink():
        raise ArtifactError(f"source shard must not be a symlink: {path}")
    resolved = path.resolve()
    if model_dir.resolve() not in resolved.parents or not resolved.is_file():
        raise ArtifactError(f"source shard is missing or escapes the model directory: {path}")
    return resolved


def source_files(
    model_dir: Path,
) -> tuple[list[Path], Path | None, dict[str, str] | None]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        if index_path.is_symlink():
            raise ArtifactError(f"source index must not be a symlink: {index_path}")
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ArtifactError(f"index weight_map must be a non-empty object: {index_path}")
        normalized_weight_map: dict[str, str] = {}
        files_by_name: dict[str, Path] = {}
        for tensor_name, raw_file in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise ArtifactError("index weight_map tensor names must be non-empty strings")
            source_file = _source_shard_path(model_dir, raw_file)
            normalized_weight_map[tensor_name] = source_file.name
            files_by_name[source_file.name] = source_file
        all_shards = {
            _source_shard_path(model_dir, path.name).name
            for path in model_dir.glob("*.safetensors")
        }
        unindexed_shards = sorted(all_shards - set(files_by_name))
        if unindexed_shards:
            raise ArtifactError(
                f"model directory contains safetensors shards absent from the index: {unindexed_shards}"
            )
        return sorted(files_by_name.values()), index_path.resolve(), normalized_weight_map
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise ArtifactError(f"no safetensors files found in {model_dir}")
    validated = [_source_shard_path(model_dir, path.name) for path in files]
    return validated, None, None


def collect_tensor_inventory(model_dir: Path) -> tuple[dict[str, TensorRegion], Path | None]:
    model_dir = model_dir.resolve()
    files, index_path, weight_map = source_files(model_dir)
    inventory: dict[str, TensorRegion] = {}
    for source_file in files:
        for region in parse_safetensors_header(source_file):
            if region.name in inventory:
                raise ArtifactError(f"duplicate tensor name across shards: {region.name}")
            inventory[region.name] = region
    if weight_map is not None:
        indexed_names = set(weight_map)
        inventory_names = set(inventory)
        missing = sorted(indexed_names - inventory_names)
        unindexed = sorted(inventory_names - indexed_names)
        if missing or unindexed:
            raise ArtifactError(
                "safetensors index/inventory tensor mismatch: "
                f"missing={missing} unindexed={unindexed}"
            )
        wrong_shards = sorted(
            name
            for name, expected_file in weight_map.items()
            if inventory[name].source_file.name != expected_file
        )
        if wrong_shards:
            raise ArtifactError(
                f"safetensors index maps tensors to the wrong shard: {wrong_shards}"
            )
    return inventory, index_path


def load_source_contract(model_dir: Path) -> SourceContract:
    model_dir = model_dir.resolve()
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ArtifactError(f"model config does not exist: {config_path}")
    if config_path.is_symlink() or model_dir not in config_path.resolve().parents:
        raise ArtifactError(f"model config must be a regular file inside the model directory: {config_path}")
    config = read_json(config_path)
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise ArtifactError("config.quantization_config must be an object")
    quant_method = quantization.get("quant_method")
    fmt = quantization.get("fmt")
    activation_scheme = quantization.get("activation_scheme")
    block_shape = quantization.get("weight_block_size")
    if quant_method != "fp8" or fmt != "e4m3" or activation_scheme != "dynamic":
        raise ArtifactError(
            "source quantization must be fp8/e4m3 with dynamic activation"
        )
    if (
        not isinstance(block_shape, list)
        or len(block_shape) != 2
        or any(not _is_int(dim) or dim <= 0 for dim in block_shape)
    ):
        raise ArtifactError("quantization_config.weight_block_size must contain two positive integers")
    if tuple(block_shape) != CANONICAL_BLOCK_SHAPE:
        raise ArtifactError(
            "sq-fp8-artifact-v0.2 requires weight_block_size [128, 128], "
            f"got {block_shape}"
        )
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists() and (
        index_path.is_symlink() or model_dir not in index_path.resolve().parents
    ):
        raise ArtifactError(f"model index must be a regular file inside the model directory: {index_path}")
    return SourceContract(
        config_file=config_path.name,
        config_sha256=sha256_file(config_path),
        index_file=index_path.name if index_path.is_file() else None,
        index_sha256=sha256_file(index_path) if index_path.is_file() else None,
        quant_method=quant_method,
        format=fmt,
        activation_scheme=activation_scheme,
        weight_block_shape=(block_shape[0], block_shape[1]),
    )


def expected_scale_shape(
    weight_shape: tuple[int, ...],
    block_shape: tuple[int, int],
) -> tuple[int, int]:
    if len(weight_shape) != 2:
        raise ArtifactError(f"FP8 weight shape must be 2D, got {weight_shape}")
    rows, cols = weight_shape
    block_rows, block_cols = block_shape
    if rows <= 0 or cols <= 0 or block_rows <= 0 or block_cols <= 0:
        raise ArtifactError("weight and block dimensions must be positive")
    return (
        (rows + block_rows - 1) // block_rows,
        (cols + block_cols - 1) // block_cols,
    )


def pair_fp8_weights(
    inventory: dict[str, TensorRegion],
    block_shape: tuple[int, int],
) -> tuple[list[WeightScalePair], list[TensorRegion]]:
    weights = sorted(
        (tensor for tensor in inventory.values() if tensor.dtype == WEIGHT_DTYPE),
        key=lambda tensor: tensor.name,
    )
    scale_tensors = {
        tensor.name: tensor
        for tensor in inventory.values()
        if tensor.name.endswith(".weight_scale_inv")
    }
    pairs: list[WeightScalePair] = []
    used_scales: set[str] = set()
    for weight in weights:
        if not weight.name.endswith(".weight"):
            raise ArtifactError(f"FP8 tensor is not a weight tensor: {weight.name}")
        if len(weight.shape) != 2:
            raise ArtifactError(f"FP8 weight must be 2D: {weight.name}")
        scale_name = f"{weight.name}_scale_inv"
        scale = inventory.get(scale_name)
        if scale is None:
            raise ArtifactError(f"missing scale tensor for FP8 weight {weight.name}: {scale_name}")
        if scale.dtype != SCALE_DTYPE:
            raise ArtifactError(
                f"scale tensor {scale.name} must use {SCALE_DTYPE}, got {scale.dtype}"
            )
        if len(scale.shape) != 2:
            raise ArtifactError(f"scale tensor must be 2D: {scale.name}")
        expected = expected_scale_shape(weight.shape, block_shape)
        if scale.shape != expected:
            raise ArtifactError(
                f"scale shape mismatch for {weight.name}: expected {expected} got {scale.shape}"
            )
        pairs.append(WeightScalePair(weight=weight, scale=scale))
        used_scales.add(scale.name)
    orphan_scales = sorted(set(scale_tensors) - used_scales)
    if orphan_scales:
        raise ArtifactError(f"orphan scale tensors: {orphan_scales}")
    passthrough = sorted(
        (
            tensor
            for tensor in inventory.values()
            if tensor.name not in used_scales and tensor.dtype != WEIGHT_DTYPE
        ),
        key=lambda tensor: tensor.name,
    )
    return pairs, passthrough


def tensor_family(name: str) -> str:
    suffixes = {
        ".self_attn.q_proj.weight": "attn_q",
        ".self_attn.k_proj.weight": "attn_k",
        ".self_attn.v_proj.weight": "attn_v",
        ".self_attn.o_proj.weight": "attn_o",
        ".mlp.gate_proj.weight": "mlp_gate",
        ".mlp.up_proj.weight": "mlp_up",
        ".mlp.down_proj.weight": "mlp_down",
    }
    for suffix, family in suffixes.items():
        if name.endswith(suffix):
            return family
    return "other_weight"


def payload_relative_paths(name: str) -> tuple[Path, Path]:
    tensor_id = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return (
        Path("weights") / f"{tensor_id}.f8_e4m3",
        Path("scales") / f"{tensor_id}.bf16",
    )


def _validate_fp8_chunk(chunk: bytes, name: str) -> None:
    if b"\x7f" in chunk or b"\xff" in chunk:
        raise ArtifactError(f"FP8 weight contains an E4M3FN NaN encoding: {name}")


def bf16_bytes_to_f32(payload: bytes) -> list[float]:
    if len(payload) % 2 != 0:
        raise ArtifactError("BF16 payload length must be divisible by two")
    values: list[float] = []
    for offset in range(0, len(payload), 2):
        bits = int.from_bytes(payload[offset : offset + 2], "little") << 16
        values.append(struct.unpack("<f", bits.to_bytes(4, "little"))[0])
    return values


def _validate_scale_chunk(chunk: bytes, name: str) -> None:
    for value in bf16_bytes_to_f32(chunk):
        if not math.isfinite(value) or value <= 0.0:
            raise ArtifactError(f"scale tensor contains a non-positive or non-finite value: {name}")


def copy_region_exact_and_hash(
    region: TensorRegion,
    destination: Path,
    chunk_bytes: int,
) -> str:
    if chunk_bytes <= 0:
        raise ArtifactError("copy chunk size must be positive")
    if region.dtype == SCALE_DTYPE and chunk_bytes % 2 != 0:
        chunk_bytes -= 1
        if chunk_bytes == 0:
            raise ArtifactError("scale copy chunk size must be at least two bytes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    remaining = region.length
    with region.source_file.open("rb") as source, destination.open("wb") as output:
        source.seek(region.offset)
        while remaining > 0:
            chunk = source.read(min(chunk_bytes, remaining))
            if not chunk:
                raise ArtifactError(f"unexpected EOF while copying {region.name}")
            if region.dtype == WEIGHT_DTYPE:
                _validate_fp8_chunk(chunk, region.name)
            elif region.dtype == SCALE_DTYPE:
                _validate_scale_chunk(chunk, region.name)
            output.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
    if destination.stat().st_size != region.length:
        raise ArtifactError(f"copied payload length mismatch for {region.name}")
    return digest.hexdigest()


def _safe_artifact_path(artifact_dir: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or any(part in {"", ".", ".."} for part in raw.parts):
        raise ArtifactError(f"artifact payload path must be a safe relative path: {relative}")
    resolved = (artifact_dir / raw).resolve()
    root = artifact_dir.resolve()
    if root not in resolved.parents:
        raise ArtifactError(f"artifact payload path escapes artifact root: {relative}")
    return resolved


def _read_region_bytes(region: TensorRegion) -> bytes:
    with region.source_file.open("rb") as handle:
        handle.seek(region.offset)
        payload = handle.read(region.length)
    if len(payload) != region.length:
        raise ArtifactError(f"failed to read complete tensor region: {region.name}")
    return payload


def _passthrough_entry(tensor: TensorRegion) -> dict[str, Any]:
    return {
        "name": tensor.name,
        "dtype": tensor.dtype,
        "shape": list(tensor.shape),
        "elements": tensor.elements,
        "source_file": tensor.source_file.name,
    }


def _pair_entry(
    pair: WeightScalePair,
    artifact_dir: Path,
    block_shape: tuple[int, int],
    chunk_bytes: int,
) -> dict[str, Any]:
    weight_file, scale_file = payload_relative_paths(pair.weight.name)
    weight_sha256 = copy_region_exact_and_hash(
        pair.weight,
        artifact_dir / weight_file,
        chunk_bytes,
    )
    scale_sha256 = copy_region_exact_and_hash(
        pair.scale,
        artifact_dir / scale_file,
        chunk_bytes,
    )
    return {
        "name": pair.weight.name,
        "family": tensor_family(pair.weight.name),
        "shape": list(pair.weight.shape),
        "elements": pair.weight.elements,
        "weight": {
            "dtype": pair.weight.dtype,
            "encoding": RAW_ENCODING,
            "file": weight_file.as_posix(),
            "bytes": pair.weight.length,
            "sha256": weight_sha256,
            "source_file": pair.weight.source_file.name,
        },
        "scale": {
            "name": pair.scale.name,
            "dtype": pair.scale.dtype,
            "encoding": RAW_ENCODING,
            "file": scale_file.as_posix(),
            "shape": list(pair.scale.shape),
            "elements": pair.scale.elements,
            "bytes": pair.scale.length,
            "sha256": scale_sha256,
            "source_file": pair.scale.source_file.name,
            "layout": SCALE_LAYOUT,
            "block_shape": list(block_shape),
            "order": SCALE_ORDER,
            "semantic": SCALE_SEMANTIC,
        },
    }


def _renameat2(
    left: Path,
    right: Path,
    flags: int,
    operation: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ArtifactError(
            f"atomic artifact {operation} requires Linux renameat2; "
            "this platform cannot safely promote the artifact"
        )
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(left),
        AT_FDCWD,
        os.fsencode(right),
        flags,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise ArtifactError(
            f"atomic artifact {operation} failed: "
            f"{os.strerror(error_number)} ({error_number})"
        )


def _rename_exchange(left: Path, right: Path) -> None:
    _renameat2(left, right, RENAME_EXCHANGE, "directory exchange")


def _rename_noreplace(left: Path, right: Path) -> None:
    _renameat2(left, right, RENAME_NOREPLACE, "initial promotion")


def _entry_metadata(path: Path, label: str) -> os.stat_result:
    try:
        return os.lstat(path)
    except OSError as exc:
        raise ArtifactError(f"failed to identify {label} {path}: {exc}") from exc


def _entry_identity(path: Path, label: str) -> tuple[int, int]:
    metadata = _entry_metadata(path, label)
    return metadata.st_dev, metadata.st_ino


def _directory_identity(path: Path, label: str) -> tuple[int, int]:
    metadata = _entry_metadata(path, label)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ArtifactError(f"{label} is not a regular directory: {path}")
    return metadata.st_dev, metadata.st_ino


def _remove_empty_quarantine(path: Path) -> None:
    try:
        path.rmdir()
    except OSError as exc:
        warnings.warn(
            f"empty artifact cleanup quarantine remains at {path}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def _remove_owned_directory(
    path: Path,
    expected_identity: tuple[int, int],
    label: str,
) -> None:
    quarantine = Path(
        tempfile.mkdtemp(
            prefix=f".{path.name}.cleanup.",
            dir=path.parent,
        )
    )
    quarantined_entry = quarantine / "owned-entry"
    try:
        _rename_noreplace(path, quarantined_entry)
    except ArtifactError as exc:
        _remove_empty_quarantine(quarantine)
        warnings.warn(
            f"{label} was not removed because it could not be moved into a private "
            f"cleanup quarantine: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    try:
        moved_identity = _entry_identity(quarantined_entry, f"quarantined {label}")
    except ArtifactError as exc:
        warnings.warn(
            f"{label} was moved to {quarantined_entry} but could not be identified; "
            f"leaving it untouched: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    if moved_identity != expected_identity:
        try:
            _rename_noreplace(quarantined_entry, path)
        except ArtifactError as exc:
            warnings.warn(
                f"{label} changed identity before cleanup; the conflicting entry remains "
                f"untouched at {quarantined_entry} because restoring {path} failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        try:
            restored_identity = _entry_identity(path, f"restored conflicting {label}")
        except ArtifactError as exc:
            warnings.warn(
                f"{label} changed identity before cleanup and was restored to {path}, but its "
                f"identity could not be confirmed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            _remove_empty_quarantine(quarantine)
            return
        _remove_empty_quarantine(quarantine)
        if restored_identity != moved_identity:
            warnings.warn(
                f"{label} changed again after it was restored to {path}; leaving the current "
                f"entry untouched with identity {restored_identity}",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        warnings.warn(
            f"{label} changed identity before cleanup; the conflicting entry was restored "
            f"without deletion at {path} with identity {restored_identity}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    try:
        shutil.rmtree(quarantine)
    except OSError as exc:
        warnings.warn(
            f"verified {label} remains in cleanup quarantine {quarantine}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def _remove_exchanged_directory(
    path: Path,
    expected_identity: tuple[int, int],
) -> None:
    _remove_owned_directory(
        path,
        expected_identity,
        "exchanged previous artifact",
    )


def _promote_artifact(
    temp_dir: Path,
    output_dir: Path,
    overwrite: bool,
    expected_output_identity: tuple[int, int] | None,
) -> None:
    output_exists = os.path.lexists(output_dir)
    if expected_output_identity is None:
        if output_exists:
            raise ArtifactError(f"output artifact appeared during build: {output_dir}")
        _rename_noreplace(temp_dir, output_dir)
        return

    if not overwrite:
        raise ArtifactError(f"output artifact already exists: {output_dir}")
    if not output_exists:
        raise ArtifactError(f"existing output artifact disappeared during build: {output_dir}")
    current_output_identity = _directory_identity(output_dir, "existing output artifact")
    if current_output_identity != expected_output_identity:
        raise ArtifactError(
            "existing output artifact changed after validation; refusing atomic overwrite: "
            f"{output_dir}"
        )

    new_artifact_identity = _directory_identity(temp_dir, "new temporary artifact")
    _rename_exchange(temp_dir, output_dir)
    exchanged_output_identity: tuple[int, int] | None = None
    post_exchange_identity_error: ArtifactError | None = None
    try:
        exchanged_output_identity = _entry_identity(
            temp_dir,
            "exchanged previous output entry",
        )
    except ArtifactError as exc:
        post_exchange_identity_error = exc

    if exchanged_output_identity == expected_output_identity:
        _remove_exchanged_directory(temp_dir, expected_output_identity)
        return

    try:
        _rename_exchange(temp_dir, output_dir)
    except ArtifactError as exc:
        raise _ArtifactPromotionError(
            "existing output artifact changed during atomic overwrite and rollback failed; "
            f"the conflicting directory remains untouched at {temp_dir}: {exc}",
            preserve_temp=True,
        ) from exc

    try:
        rolled_back_temp_identity = _directory_identity(
            temp_dir,
            "rolled-back new temporary artifact",
        )
    except ArtifactError as exc:
        raise _ArtifactPromotionError(
            "existing output artifact changed during atomic overwrite; rollback returned but "
            f"the temporary path could not be identified, so it will not be removed: {exc}",
            preserve_temp=True,
        ) from exc
    if rolled_back_temp_identity != new_artifact_identity:
        raise _ArtifactPromotionError(
            "existing output artifact changed during atomic overwrite; rollback did not return "
            f"the new artifact to {temp_dir}, so that path will not be removed",
            preserve_temp=True,
        )
    try:
        rolled_back_output_identity = _entry_identity(
            output_dir,
            "rolled-back conflicting output entry",
        )
    except ArtifactError as exc:
        raise _ArtifactPromotionError(
            "existing output artifact changed during atomic overwrite; the new artifact was "
            f"rolled back but the conflicting output entry could not be identified: {exc}",
            preserve_temp=True,
        ) from exc
    if exchanged_output_identity is not None and (
        rolled_back_output_identity != exchanged_output_identity
    ):
        raise _ArtifactPromotionError(
            "existing output artifact changed during atomic overwrite; rollback did not restore "
            f"the conflicting output entry identity at {output_dir}",
            preserve_temp=True,
        )
    if rolled_back_output_identity == new_artifact_identity:
        raise _ArtifactPromotionError(
            "existing output artifact changed during atomic overwrite; rollback left the new "
            f"artifact active at {output_dir}",
            preserve_temp=True,
        )
    identity_detail = (
        f"; post-exchange identity error was: {post_exchange_identity_error}"
        if post_exchange_identity_error is not None
        else ""
    )
    raise ArtifactError(
        "existing output artifact changed during atomic overwrite; promotion was rolled back "
        f"without deleting the conflicting entry at {output_dir}{identity_detail}"
    )


def build_canonical_artifact(
    model_dir: Path,
    output_dir: Path,
    *,
    tensor_names: Iterable[str] | None = None,
    copy_chunk_bytes: int = DEFAULT_COPY_CHUNK_BYTES,
    overwrite: bool = False,
) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    if output_dir.is_symlink():
        raise ArtifactError(f"output artifact path must not be a symlink: {output_dir}")
    output_dir = Path(os.path.abspath(output_dir)).resolve(strict=False)
    if (
        output_dir == model_dir
        or output_dir.is_relative_to(model_dir)
        or model_dir.is_relative_to(output_dir)
    ):
        raise ArtifactError(
            "source model and output artifact paths must not be equal or contain one another: "
            f"source={model_dir} output={output_dir}"
        )
    expected_output_identity: tuple[int, int] | None = None
    if output_dir.exists() and not output_dir.is_dir():
        raise ArtifactError(f"existing output artifact is not a directory: {output_dir}")
    if output_dir.exists() and not overwrite:
        raise ArtifactError(f"output artifact already exists: {output_dir}")
    if output_dir.exists():
        expected_output_identity = _directory_identity(
            output_dir,
            "existing output artifact",
        )
        try:
            verify_canonical_artifact(output_dir)
        except ArtifactError as exc:
            raise ArtifactError(
                "--overwrite only accepts an existing verified sq-fp8-artifact-v0.2 "
                f"canonical artifact: {output_dir}: {exc}"
            ) from exc
        if (
            _directory_identity(output_dir, "validated existing output artifact")
            != expected_output_identity
        ):
            raise ArtifactError(
                "existing output artifact changed identity during validation; "
                f"refusing atomic overwrite: {output_dir}"
            )
    contract = load_source_contract(model_dir)
    inventory, inventory_index_path = collect_tensor_inventory(model_dir)
    if contract.index_file is not None and (
        inventory_index_path is None or inventory_index_path.name != contract.index_file
    ):
        raise ArtifactError("source index identity changed while collecting tensors")
    pairs, passthrough = pair_fp8_weights(inventory, contract.weight_block_shape)
    if not pairs:
        raise ArtifactError("source checkpoint contains no complete F8_E4M3 weight/scale pairs")
    by_name = {pair.weight.name: pair for pair in pairs}
    if tensor_names is None:
        selected_pairs = pairs
    else:
        requested = list(tensor_names)
        if not requested:
            raise ArtifactError("tensor selection must not be empty")
        if len(set(requested)) != len(requested):
            raise ArtifactError("tensor selection contains duplicate names")
        missing = sorted(set(requested) - set(by_name))
        if missing:
            raise ArtifactError(f"selected FP8 weight tensors do not exist: {missing}")
        selected_pairs = [by_name[name] for name in sorted(requested)]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent)
    )
    temp_dir_identity: tuple[int, int] | None = None
    try:
        temp_dir_identity = _directory_identity(temp_dir, "new temporary artifact")
        quantized_entries = [
            _pair_entry(pair, temp_dir, contract.weight_block_shape, copy_chunk_bytes)
            for pair in selected_pairs
        ]
        weight_bytes = sum(entry["weight"]["bytes"] for entry in quantized_entries)
        scale_bytes = sum(entry["scale"]["bytes"] for entry in quantized_entries)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "format_id": FORMAT_ID,
            "source": {
                "model_name": model_dir.name,
                "config_file": contract.config_file,
                "config_sha256": contract.config_sha256,
                "index_file": contract.index_file,
                "index_sha256": contract.index_sha256,
                "quantization": {
                    "quant_method": contract.quant_method,
                    "format": contract.format,
                    "activation_scheme": contract.activation_scheme,
                    "weight_block_shape": list(contract.weight_block_shape),
                },
            },
            "import": {
                "mode": IMPORT_MODE,
                "encoding": RAW_ENCODING,
            },
            "coverage": {
                "scope": "full_model" if len(selected_pairs) == len(pairs) else "selected_tensors",
                "source_tensor_count": len(inventory),
                "source_fp8_weight_count": len(pairs),
                "source_scale_count": len(pairs),
                "paired_tensor_count": len(pairs),
                "selected_pair_count": len(selected_pairs),
                "unpaired_tensor_count": 0,
                "passthrough_tensor_count": len(passthrough),
            },
            "storage": {
                "weight_payload_bytes": weight_bytes,
                "scale_payload_bytes": scale_bytes,
                "total_payload_bytes": weight_bytes + scale_bytes,
            },
            "quantized_tensors": sorted(quantized_entries, key=lambda entry: entry["name"]),
            "passthrough_tensors": [_passthrough_entry(tensor) for tensor in passthrough],
        }
        manifest["integrity"] = {
            "content_sha256": artifact_content_sha256(manifest),
        }
        write_json(temp_dir / "sq_manifest.json", manifest)
        verify_canonical_artifact(temp_dir)
        try:
            _promote_artifact(
                temp_dir,
                output_dir,
                overwrite,
                expected_output_identity,
            )
        except _ArtifactPromotionError as exc:
            if exc.preserve_temp:
                temp_dir = None
            raise
        temp_dir = None
        return manifest
    finally:
        if temp_dir is not None and temp_dir_identity is not None:
            _remove_owned_directory(
                temp_dir,
                temp_dir_identity,
                "new temporary artifact",
            )


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactError(f"{label} must be an object")
    return value


def _require_shape(value: Any, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not _is_int(dim) or dim <= 0 for dim in value)
    ):
        raise ArtifactError(f"{label} must contain two positive integers")
    return value[0], value[1]


def _require_sha256(value: Any, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ArtifactError(f"{label} must be lowercase hexadecimal SHA-256")
    return value


def _require_source_file_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{label} must be a non-empty shard basename")
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name != value:
        raise ArtifactError(f"{label} must be a shard basename")
    return value


def _dynamic_shape_elements(value: Any, label: str) -> int:
    if not isinstance(value, list) or any(
        not _is_int(dim) or dim <= 0 for dim in value
    ):
        raise ArtifactError(f"{label} must contain positive integer dimensions")
    return math.prod(value)


def _qwen3_pair_alias(name: str) -> str | None:
    prefix = "model.layers."
    language_prefix = "model.language_model.layers."
    if name.startswith(prefix):
        return language_prefix + name[len(prefix) :]
    if name.startswith(language_prefix):
        return prefix + name[len(language_prefix) :]
    return None


def _verify_payload_file(
    artifact_dir: Path,
    payload: dict[str, Any],
    label: str,
) -> Path:
    relative = payload.get("file")
    if not isinstance(relative, str):
        raise ArtifactError(f"{label}.file must be a string")
    path = _safe_artifact_path(artifact_dir, relative)
    expected_bytes = payload.get("bytes")
    if not _is_int(expected_bytes) or expected_bytes < 0:
        raise ArtifactError(f"{label}.bytes must be a non-negative integer")
    try:
        actual_bytes = path.stat().st_size
    except OSError as exc:
        raise ArtifactError(f"failed to stat {label} file {path}: {exc}") from exc
    if actual_bytes != expected_bytes:
        raise ArtifactError(
            f"{label} byte length mismatch: expected {expected_bytes} got {actual_bytes}"
        )
    expected_sha256 = payload.get("sha256")
    if not isinstance(expected_sha256, str) or SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ArtifactError(f"{label}.sha256 must be lowercase hexadecimal SHA-256")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ArtifactError(
            f"{label} SHA-256 mismatch: expected {expected_sha256} got {actual_sha256}"
        )
    return path


def verify_canonical_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    manifest_path = artifact_dir / "sq_manifest.json"
    try:
        manifest_bytes = manifest_path.stat().st_size
    except OSError as exc:
        raise ArtifactError(f"failed to stat canonical manifest {manifest_path}: {exc}") from exc
    if manifest_bytes > MAX_MANIFEST_BYTES:
        raise ArtifactError(
            f"canonical manifest exceeds {MAX_MANIFEST_BYTES} bytes: {manifest_path}"
        )
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(f"schema_version must be {SCHEMA_VERSION}")
    if manifest.get("artifact_kind") != ARTIFACT_KIND:
        raise ArtifactError(f"artifact_kind must be {ARTIFACT_KIND}")
    if manifest.get("format_id") != FORMAT_ID:
        raise ArtifactError(f"format_id must be {FORMAT_ID}")
    import_info = _require_dict(manifest.get("import"), "import")
    if import_info.get("mode") != IMPORT_MODE or import_info.get("encoding") != RAW_ENCODING:
        raise ArtifactError("canonical import mode/encoding mismatch")
    integrity = _require_dict(manifest.get("integrity"), "integrity")
    expected_content_sha256 = integrity.get("content_sha256")
    if not isinstance(expected_content_sha256, str) or (
        SHA256_PATTERN.fullmatch(expected_content_sha256) is None
    ):
        raise ArtifactError("integrity.content_sha256 must be lowercase hexadecimal SHA-256")
    without_integrity = dict(manifest)
    del without_integrity["integrity"]
    actual_content_sha256 = artifact_content_sha256(without_integrity)
    if expected_content_sha256 != actual_content_sha256:
        raise ArtifactError(
            "artifact content SHA-256 mismatch: "
            f"expected {expected_content_sha256} got {actual_content_sha256}"
        )

    source = _require_dict(manifest.get("source"), "source")
    if not isinstance(source.get("model_name"), str) or not source.get("model_name"):
        raise ArtifactError("source.model_name must be a non-empty string")
    _require_source_file_name(source.get("config_file"), "source.config_file")
    _require_sha256(source.get("config_sha256"), "source.config_sha256")
    index_file = source.get("index_file")
    index_sha256 = source.get("index_sha256")
    if (index_file is None) != (index_sha256 is None):
        raise ArtifactError("source.index_file and index_sha256 must both be present or absent")
    if index_file is not None:
        _require_source_file_name(index_file, "source.index_file")
        _require_sha256(index_sha256, "source.index_sha256")
    quantization = _require_dict(source.get("quantization"), "source.quantization")
    if (
        quantization.get("quant_method") != "fp8"
        or quantization.get("format") != "e4m3"
        or quantization.get("activation_scheme") != "dynamic"
    ):
        raise ArtifactError("source quantization contract must be fp8/e4m3/dynamic")
    block_shape = _require_shape(
        quantization.get("weight_block_shape"),
        "source.quantization.weight_block_shape",
    )
    if block_shape != CANONICAL_BLOCK_SHAPE:
        raise ArtifactError(
            f"source weight_block_shape must be {CANONICAL_BLOCK_SHAPE}, got {block_shape}"
        )
    entries = manifest.get("quantized_tensors")
    if not isinstance(entries, list):
        raise ArtifactError("quantized_tensors must be a list")
    names: set[str] = set()
    scale_names: set[str] = set()
    payload_files: set[str] = set()
    weight_bytes = 0
    scale_bytes = 0
    ordered_names = [
        entry.get("name") if isinstance(entry, dict) else None for entry in entries
    ]
    if ordered_names != sorted(ordered_names, key=lambda value: str(value)):
        raise ArtifactError("quantized_tensors must be sorted by name")
    for index, raw_entry in enumerate(entries):
        entry = _require_dict(raw_entry, f"quantized_tensors[{index}]")
        name = entry.get("name")
        if not isinstance(name, str) or not name.endswith(".weight"):
            raise ArtifactError(f"quantized_tensors[{index}].name must be a non-empty .weight name")
        if name in names:
            raise ArtifactError(f"duplicate quantized tensor entry: {name}")
        names.add(name)
        if not isinstance(entry.get("family"), str) or not entry.get("family"):
            raise ArtifactError(f"quantized_tensors[{index}].family must be non-empty")
        shape = _require_shape(entry.get("shape"), f"quantized_tensors[{index}].shape")
        weight = _require_dict(entry.get("weight"), f"quantized_tensors[{index}].weight")
        scale = _require_dict(entry.get("scale"), f"quantized_tensors[{index}].scale")
        if weight.get("dtype") != WEIGHT_DTYPE or weight.get("encoding") != RAW_ENCODING:
            raise ArtifactError(f"invalid canonical weight contract for {name}")
        if scale.get("dtype") != SCALE_DTYPE or scale.get("encoding") != RAW_ENCODING:
            raise ArtifactError(f"invalid canonical scale contract for {name}")
        expected_scale_name = f"{name}_scale_inv"
        if scale.get("name") != expected_scale_name:
            raise ArtifactError(
                f"scale name mismatch for {name}: expected {expected_scale_name} "
                f"got {scale.get('name')}"
            )
        if expected_scale_name in scale_names:
            raise ArtifactError(f"duplicate scale tensor entry: {expected_scale_name}")
        scale_names.add(expected_scale_name)
        if (
            scale.get("layout") != SCALE_LAYOUT
            or scale.get("order") != SCALE_ORDER
            or scale.get("semantic") != SCALE_SEMANTIC
        ):
            raise ArtifactError(f"invalid canonical scale layout for {name}")
        entry_block_shape = _require_shape(
            scale.get("block_shape"),
            f"quantized_tensors[{index}].scale.block_shape",
        )
        if entry_block_shape != block_shape:
            raise ArtifactError(f"scale block shape differs from source contract for {name}")
        scale_shape = _require_shape(
            scale.get("shape"),
            f"quantized_tensors[{index}].scale.shape",
        )
        if scale_shape != expected_scale_shape(shape, block_shape):
            raise ArtifactError(f"scale shape mismatch for {name}")
        if not _is_int(entry.get("elements")) or entry.get("elements") != math.prod(shape):
            raise ArtifactError(f"weight element count mismatch for {name}")
        if not _is_int(scale.get("elements")) or scale.get("elements") != math.prod(scale_shape):
            raise ArtifactError(f"scale element count mismatch for {name}")
        if weight.get("bytes") != math.prod(shape):
            raise ArtifactError(f"weight payload byte count mismatch for {name}")
        if scale.get("bytes") != math.prod(scale_shape) * 2:
            raise ArtifactError(f"scale payload byte count mismatch for {name}")
        _require_source_file_name(weight.get("source_file"), f"{name}.weight.source_file")
        _require_source_file_name(scale.get("source_file"), f"{name}.scale.source_file")
        for kind, payload in (("weight", weight), ("scale", scale)):
            relative = payload.get("file")
            if not isinstance(relative, str) or relative in payload_files:
                raise ArtifactError(f"duplicate or invalid artifact payload file for {name} {kind}")
            payload_files.add(relative)
        weight_path = _verify_payload_file(artifact_dir, weight, f"{name} weight")
        scale_path = _verify_payload_file(artifact_dir, scale, f"{name} scale")
        weight_bytes += weight_path.stat().st_size
        scale_bytes += scale_path.stat().st_size
        with weight_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                _validate_fp8_chunk(chunk, name)
        with scale_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                _validate_scale_chunk(chunk, str(scale.get("name", name)))

    coverage = _require_dict(manifest.get("coverage"), "coverage")
    scope = coverage.get("scope")
    if scope not in {"selected_tensors", "full_model"}:
        raise ArtifactError("coverage.scope must be selected_tensors or full_model")
    coverage_count_fields = (
        "source_tensor_count",
        "source_fp8_weight_count",
        "source_scale_count",
        "paired_tensor_count",
        "selected_pair_count",
        "unpaired_tensor_count",
        "passthrough_tensor_count",
    )
    for field in coverage_count_fields:
        value = coverage.get(field)
        if not _is_int(value) or value < 0:
            raise ArtifactError(f"coverage.{field} must be a non-negative integer")
    if coverage.get("selected_pair_count") != len(entries):
        raise ArtifactError("coverage.selected_pair_count mismatch")
    if not entries:
        raise ArtifactError("canonical artifact must contain at least one selected pair")
    if coverage.get("unpaired_tensor_count") != 0:
        raise ArtifactError("coverage.unpaired_tensor_count must be zero")
    source_tensor_count = coverage.get("source_tensor_count")
    source_fp8_weight_count = coverage.get("source_fp8_weight_count")
    source_scale_count = coverage.get("source_scale_count")
    paired_tensor_count = coverage.get("paired_tensor_count")
    passthrough_tensor_count = coverage.get("passthrough_tensor_count")
    passthrough_entries = manifest.get("passthrough_tensors")
    if not isinstance(passthrough_entries, list):
        raise ArtifactError("passthrough_tensors must be a list")
    if passthrough_tensor_count != len(passthrough_entries):
        raise ArtifactError("coverage.passthrough_tensor_count mismatch")
    if (
        source_fp8_weight_count <= 0
        or source_fp8_weight_count < len(entries)
        or source_scale_count != source_fp8_weight_count
        or paired_tensor_count != source_fp8_weight_count
    ):
        raise ArtifactError("coverage FP8 weight/scale pair counts are inconsistent")
    if scope == "full_model" and len(entries) != paired_tensor_count:
        raise ArtifactError("full_model coverage requires every source pair to be selected")
    if source_tensor_count != (
        source_fp8_weight_count + source_scale_count + passthrough_tensor_count
    ):
        raise ArtifactError("coverage.source_tensor_count mismatch")
    passthrough_names: set[str] = set()
    ordered_passthrough_names = [
        entry.get("name") if isinstance(entry, dict) else None
        for entry in passthrough_entries
    ]
    if ordered_passthrough_names != sorted(
        ordered_passthrough_names,
        key=lambda value: str(value),
    ):
        raise ArtifactError("passthrough_tensors must be sorted by name")
    for index, raw_entry in enumerate(passthrough_entries):
        entry = _require_dict(raw_entry, f"passthrough_tensors[{index}]")
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in passthrough_names or name in names:
            raise ArtifactError(f"invalid or duplicate passthrough tensor name: {name}")
        if name in scale_names:
            raise ArtifactError(f"scale tensor also appears as passthrough: {name}")
        passthrough_names.add(name)
        dtype = entry.get("dtype")
        if not isinstance(dtype, str) or not dtype:
            raise ArtifactError(f"passthrough_tensors[{index}].dtype must be non-empty")
        expected_elements = _dynamic_shape_elements(
            entry.get("shape"),
            f"passthrough_tensors[{index}].shape",
        )
        if not _is_int(entry.get("elements")) or entry.get("elements") != expected_elements:
            raise ArtifactError(f"passthrough tensor element count mismatch: {name}")
        _require_source_file_name(
            entry.get("source_file"),
            f"passthrough_tensors[{index}].source_file",
        )
    for name in names:
        alias = _qwen3_pair_alias(name)
        if alias is not None and alias in names:
            raise ArtifactError(
                f"quantized_tensors contain both Qwen namespace aliases: {name} and {alias}"
            )
    storage = _require_dict(manifest.get("storage"), "storage")
    for field in ("weight_payload_bytes", "scale_payload_bytes", "total_payload_bytes"):
        value = storage.get(field)
        if not _is_int(value) or value < 0:
            raise ArtifactError(f"storage.{field} must be a non-negative integer")
    if storage.get("weight_payload_bytes") != weight_bytes:
        raise ArtifactError("storage.weight_payload_bytes mismatch")
    if storage.get("scale_payload_bytes") != scale_bytes:
        raise ArtifactError("storage.scale_payload_bytes mismatch")
    if storage.get("total_payload_bytes") != weight_bytes + scale_bytes:
        raise ArtifactError("storage.total_payload_bytes mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "selected_pair_count": len(entries),
        "weight_payload_bytes": weight_bytes,
        "scale_payload_bytes": scale_bytes,
        "content_sha256": actual_content_sha256,
        "verified": True,
    }


def fp8_e4m3fn_to_f32(byte: int) -> float:
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 0x0F
    mantissa = byte & 0x07
    if exponent == 0:
        return sign * mantissa * (2.0**-9)
    if exponent == 0x0F and mantissa == 0x07:
        return math.nan
    return sign * (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))


def reconstruct_artifact_points_f32(
    artifact_dir: Path,
    tensor_name: str,
    points: Iterable[tuple[int, int]],
) -> list[float]:
    artifact_dir = artifact_dir.resolve()
    verify_canonical_artifact(artifact_dir)
    manifest = read_json(artifact_dir / "sq_manifest.json")
    entries = manifest.get("quantized_tensors")
    if not isinstance(entries, list):
        raise ArtifactError("quantized_tensors must be a list")
    matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("name") == tensor_name]
    if len(matches) != 1:
        raise ArtifactError(f"expected one canonical tensor named {tensor_name}, found {len(matches)}")
    entry = matches[0]
    rows, cols = _require_shape(entry.get("shape"), f"{tensor_name}.shape")
    weight = _require_dict(entry.get("weight"), f"{tensor_name}.weight")
    scale = _require_dict(entry.get("scale"), f"{tensor_name}.scale")
    scale_rows, scale_cols = _require_shape(scale.get("shape"), f"{tensor_name}.scale.shape")
    block_rows, block_cols = _require_shape(
        scale.get("block_shape"),
        f"{tensor_name}.scale.block_shape",
    )
    weight_path = _safe_artifact_path(artifact_dir, str(weight.get("file")))
    scale_path = _safe_artifact_path(artifact_dir, str(scale.get("file")))
    output: list[float] = []
    with weight_path.open("rb") as weight_handle, scale_path.open("rb") as scale_handle:
        for row, col in points:
            if row < 0 or row >= rows or col < 0 or col >= cols:
                raise ArtifactError(f"reconstruction point is out of bounds: {(row, col)}")
            weight_handle.seek(row * cols + col)
            weight_byte = weight_handle.read(1)
            scale_row = row // block_rows
            scale_col = col // block_cols
            if scale_row >= scale_rows or scale_col >= scale_cols:
                raise ArtifactError("scale lookup is out of bounds")
            scale_handle.seek((scale_row * scale_cols + scale_col) * 2)
            scale_bytes = scale_handle.read(2)
            if len(weight_byte) != 1 or len(scale_bytes) != 2:
                raise ArtifactError("artifact payload ended during point reconstruction")
            weight_value = fp8_e4m3fn_to_f32(weight_byte[0])
            scale_value = bf16_bytes_to_f32(scale_bytes)[0]
            output.append(weight_value * scale_value)
    return output
