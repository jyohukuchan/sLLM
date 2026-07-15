#!/usr/bin/env python3
"""Offline AQ4 layer-0 matvec oracle.

The oracle decodes one package matrix (the first linear tensor consumed by the
layer-0 runtime) in bounded row/group chunks.  It can compare the CPU package
matvec with a tensor-level GPU trace and with a BF16 source matvec.  Existing
attempt-3 differential traces intentionally contain only stage samples, not a
tensor-level projection output; in that case this tool emits a fail-closed
``blocked_missing_tensor_output`` report and never infers a kernel diagnosis.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import stat
import struct
from pathlib import Path
from typing import Any


SCHEMA = "ullm.aq4_layer0_matvec_oracle.v1"
TENSOR_OUTPUT_SCHEMA = "ullm.aq4_layer0_matvec_tensor_output.v1"
TRACE_SCHEMA = "ullm.qwen35_aq4_differential_trace.v1"
DEFAULT_TENSOR = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
DEFAULT_ACTIVE_SHA = "feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44"
DEFAULT_PACKAGE_SHA = "a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad"
DEFAULT_SOURCE_TRACE_SHA = "9638b4e724c00747e8f0cd2eda1637e6e3679869349ce675463baf5fe393a692"
DEFAULT_GPU_OPERATION = "fused_qkv_aq4_matvec"
DEFAULT_GPU_RPB_ROWS = 4
DEFAULT_GPU_THREADS_PER_ROW = 64
MAX_TRACE_ROWS = 4096
MAX_GPU_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_GPU_OUTPUT_ROWS = 4096
MAX_GPU_ROW_ELEMENTS = 8192
EXPECTED_ROWS = (("fixture-prompt-0", 0), ("fixture-prompt-0", 1), ("fixture-prompt-1", 0))
F32 = struct.Struct("<f")
_AUDIT_SPEC = importlib.util.spec_from_file_location("aq4_input_controls", Path(__file__).with_name("audit_aq4_p2_input_controls.py"))
if _AUDIT_SPEC is None or _AUDIT_SPEC.loader is None:
    raise RuntimeError("AQ4 input-control reference is unavailable")
_AUDIT = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT)


class OracleError(ValueError):
    pass


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    if path.is_symlink() or not path.is_file():
        raise OracleError(f"{path} must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise OracleError(f"{label} must be a regular directory")
    return path.resolve()


def _canonical_file(path: Path, label: str, *, root: Path | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise OracleError(f"{label} must be a regular file")
    canonical = path.resolve()
    if canonical.is_symlink() or not canonical.is_file():
        raise OracleError(f"{label} must resolve to a regular file")
    if root is not None:
        root = root.resolve()
        try:
            canonical.relative_to(root)
        except ValueError as error:
            raise OracleError(f"{label} escapes canonical root {root}") from error
    return canonical


def _stat_signature(path: Path, label: str) -> dict[str, int]:
    try:
        info = path.stat()
    except OSError as error:
        raise OracleError(f"{label} stat failed: {error}") from error
    if not stat.S_ISREG(info.st_mode):
        raise OracleError(f"{label} is not a regular file")
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "size_bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _capture_file_identity(path: Path, label: str, *, root: Path | None = None) -> dict[str, Any]:
    canonical = _canonical_file(path, label, root=root)
    pre = _stat_signature(canonical, label)
    digest = sha256_file(canonical)
    post = _stat_signature(canonical, label)
    if pre != post:
        raise OracleError(f"{label} changed during identity capture")
    return {
        "canonical_path": str(canonical),
        "sha256": digest,
        "pre_stat": pre,
        "post_stat": post,
    }


def _assert_file_identity(identity: dict[str, Any], label: str) -> None:
    path = Path(str(identity.get("canonical_path", "")))
    current = _capture_file_identity(path, label)
    if current["canonical_path"] != identity["canonical_path"] or current["sha256"] != identity["sha256"] or current["pre_stat"] != identity["post_stat"]:
        raise OracleError(f"{label} changed after identity capture")


def _canonical_package_payload(package_dir: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise OracleError(f"{label} must be a relative package path")
    return _canonical_file(package_dir / relative, label, root=package_dir)


def read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise OracleError(f"{path} must be a regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OracleError(f"invalid JSON {path}: {error}") from error


def _reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OracleError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OracleError(f"{label} must be finite")
    value = float(value)
    if not math.isfinite(value):
        raise OracleError(f"{label} must be finite")
    return value


def scale_values_e4m3() -> tuple[float, ...]:
    """Return the runtime's sorted finite E4M3 scale table."""
    values: set[float] = set()
    exp_bits = 4
    mant_bits = 3
    bias = (1 << (exp_bits - 1)) - 1
    mant_count = 1 << mant_bits
    for exponent in range((1 << exp_bits) - 1):
        for mantissa in range(mant_count):
            if exponent == 0:
                if mantissa == 0:
                    continue
                value = (mantissa / mant_count) * (2.0 ** (1 - bias))
            else:
                value = (1.0 + mantissa / mant_count) * (2.0 ** (exponent - bias))
            values.add(value)
    return tuple(sorted(values))


class SafeTensorReader:
    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise OracleError("source safetensor header length is truncated")
            header_size = struct.unpack("<Q", raw)[0]
            if header_size > 64 * 1024 * 1024:
                raise OracleError("source safetensor header exceeds bound")
            header_raw = handle.read(header_size)
        try:
            self.header = json.loads(header_raw.decode("utf-8"), object_pairs_hook=_reject_duplicate)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise OracleError(f"invalid source safetensor header: {error}") from error
        if not isinstance(self.header, dict):
            raise OracleError("source safetensor header must be an object")
        self.data_start = 8 + header_size

    def metadata(self, name: str) -> dict[str, Any]:
        value = self.header.get(name)
        if not isinstance(value, dict) or name == "__metadata__":
            raise OracleError(f"source tensor metadata is missing: {name}")
        return value

    def read_tensor_row(self, name: str, row: int) -> list[float]:
        metadata = self.metadata(name)
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if metadata.get("dtype") != "BF16" or not isinstance(shape, list) or len(shape) not in (1, 2) or not isinstance(offsets, list) or len(offsets) != 2:
            raise OracleError(f"source tensor {name} must be a BF16 tensor")
        rows, cols = (1, int(shape[0])) if len(shape) == 1 else (int(shape[0]), int(shape[1]))
        start, end = (int(offsets[0]), int(offsets[1]))
        if rows <= 0 or cols <= 0 or start < 0 or end < start or end - start != rows * cols * 2:
            raise OracleError(f"source tensor {name} metadata geometry differs")
        if not 0 <= row < rows:
            raise OracleError(f"source row {row} is outside {rows}")
        with self.path.open("rb") as handle:
            handle.seek(self.data_start + start + row * cols * 2)
            payload = handle.read(cols * 2)
        if len(payload) != cols * 2:
            raise OracleError(f"source tensor {name} row is truncated")
        values: list[float] = []
        for (raw,) in struct.iter_unpack("<H", payload):
            values.append(F32.unpack(struct.pack("<I", raw << 16))[0])
        return values

    def read_tensor_vector(self, name: str) -> list[float]:
        metadata = self.metadata(name)
        shape = metadata.get("shape")
        if not isinstance(shape, list) or len(shape) != 1:
            raise OracleError(f"source tensor {name} must be a one-dimensional vector")
        return self.read_tensor_row(name, 0)

    def digest_tensor(self, name: str) -> str:
        metadata = self.metadata(name)
        offsets = metadata.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise OracleError(f"source tensor {name} offsets differ")
        start, end = int(offsets[0]), int(offsets[1])
        if start < 0 or end < start:
            raise OracleError(f"source tensor {name} offsets are invalid")
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            handle.seek(self.data_start + start)
            remaining = end - start
            while remaining:
                chunk = handle.read(min(1 << 20, remaining))
                if not chunk:
                    raise OracleError(f"source tensor {name} payload is truncated")
                digest.update(chunk)
                remaining -= len(chunk)
        return digest.hexdigest()


def _load_trace(root: Path, expected_package_sha: str | None, expected_active_sha: str | None) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    root = _canonical_directory(root, "trace root")
    manifest_path = _canonical_file(root / "manifest.json", "trace manifest", root=root)
    payload = _canonical_file(root / "payload.jsonl", "trace payload", root=root)
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != TRACE_SCHEMA or type(manifest.get("rows")) is not int or manifest.get("rows") != 3:
        raise OracleError("differential trace manifest schema/row count differs")
    identity = manifest.get("identity")
    if expected_package_sha is not None or expected_active_sha is not None:
        if not isinstance(identity, dict):
            raise OracleError("differential trace identity differs")
        if expected_package_sha is not None and identity.get("package_manifest_sha256") != expected_package_sha:
            raise OracleError("differential trace package identity differs")
        if expected_active_sha is not None and identity.get("active_manifest_sha256") != expected_active_sha:
            raise OracleError("differential trace active identity differs")
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with payload.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if len(line) > 64 * 1024 * 1024:
                raise OracleError(f"differential trace row {line_number} exceeds bound")
            if not line.strip():
                raise OracleError(f"differential trace row {line_number} is blank")
            try:
                value = json.loads(line, object_pairs_hook=_reject_duplicate)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise OracleError(f"invalid differential trace JSONL row {line_number}: {error}") from error
            if not isinstance(value, dict) or not isinstance(value.get("case_id"), str) or type(value.get("step")) is not int:
                raise OracleError("differential trace row identity differs")
            key = (value["case_id"], value["step"])
            if key in rows:
                raise OracleError(f"duplicate differential trace row: {key}")
            rows[key] = value
            if len(rows) > MAX_TRACE_ROWS:
                raise OracleError(f"differential trace rows exceed bound {MAX_TRACE_ROWS}")
    if set(rows) != set(EXPECTED_ROWS):
        raise OracleError("differential trace case set differs")
    return manifest, rows


def _load_package(package_dir: Path, tensor_name: str) -> tuple[dict[str, Any], dict[str, Any], Path, str, dict[str, Any]]:
    package_dir = _canonical_directory(package_dir, "package root")
    manifest_path = _canonical_file(package_dir / "manifest.json", "package manifest", root=package_dir)
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "ullm-prototype-manifest-v0.1":
        raise OracleError("package manifest schema differs")
    package_identity = _capture_file_identity(manifest_path, "package manifest", root=package_dir)
    package_sha = package_identity["sha256"]
    entries = {str(item.get("name")): item for item in manifest.get("tensors", []) if isinstance(item, dict)}
    item = entries.get(tensor_name)
    if item is None:
        raise OracleError(f"sentinel tensor is absent from package: {tensor_name}")
    if item.get("name") != DEFAULT_TENSOR:
        raise OracleError("sentinel must be the first layer-0 linear tensor")
    shape = item.get("shape")
    if not isinstance(shape, list) or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
        raise OracleError("sentinel tensor shape differs")
    rows, cols = shape
    group_size = item.get("group_size")
    if type(group_size) is not int or group_size <= 0 or cols % group_size:
        raise OracleError("sentinel group geometry differs")
    if item.get("dtype") != "BF16" or item.get("index_encoding") != "idx4_low_nibble_first" or item.get("scale_encoding") != "u8_scale_table_index" or item.get("scale_format") != "e4m3":
        raise OracleError("sentinel AQ4 encoding differs")
    for field in ("index_file", "scale_file", "codebook_file", "source_file"):
        if field not in item:
            raise OracleError(f"sentinel manifest field is missing: {field}")
    source_path = _canonical_file(Path(str(item["source_file"])), "sentinel source safetensor")
    item["source_file"] = str(source_path)
    return manifest, item, package_dir, package_sha, package_identity


def _read_f32_file(path: Path, count: int) -> list[float]:
    payload = path.read_bytes()
    if len(payload) != count * 4:
        raise OracleError(f"codebook byte length differs: {path}")
    return [value[0] for value in struct.iter_unpack("<f", payload)]


def _read_window(path: Path, offset: int, length: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(length)
    if len(payload) != length:
        raise OracleError(f"AQ4 payload is truncated: {path}")
    return payload


def _f32(value: float) -> float:
    return F32.unpack(F32.pack(value))[0]


def dequant_matvec(package_dir: Path, item: dict[str, Any], vector: list[float], *, scalar_f32: bool = False) -> list[float]:
    shape = item["shape"]
    rows, cols = shape
    if len(vector) != cols:
        raise OracleError(f"matvec input length differs: got {len(vector)} expected {cols}")
    vector = [finite(value, "matvec input") for value in vector]
    group_size = item["group_size"]
    groups_per_row = cols // group_size
    package_dir = _canonical_directory(package_dir, "package root")
    index_path = _canonical_package_payload(package_dir, item["index_file"], "AQ4 index payload")
    scale_path = _canonical_package_payload(package_dir, item["scale_file"], "AQ4 scale payload")
    codebook_path = _canonical_package_payload(package_dir, item["codebook_file"], "AQ4 codebook payload")
    codebook = _read_f32_file(codebook_path, 16)
    tensor_scale = _f32(finite(item.get("tensor_scale"), "tensor_scale"))
    if tensor_scale <= 0:
        raise OracleError("tensor_scale must be positive")
    scales = scale_values_e4m3()
    expected_index = rows * cols // 2
    expected_scale = rows * groups_per_row
    if index_path.stat().st_size != expected_index or scale_path.stat().st_size != expected_scale:
        raise OracleError("AQ4 index/scale byte lengths differ")
    result: list[float] = []
    scale_table = list(scales)
    for row in range(rows):
        indices = _read_window(index_path, row * cols // 2, cols // 2)
        scale_indices = _read_window(scale_path, row * groups_per_row, groups_per_row)
        if scalar_f32:
            total = 0.0
            for element in range(cols):
                packed = indices[element // 2]
                code = (packed & 0x0F) if element % 2 == 0 else packed >> 4
                group = element // group_size
                weight = _f32(_f32(codebook[code]) * _f32(scale_table[scale_indices[group]]))
                weight = _f32(weight * tensor_scale)
                product = _f32(weight * _f32(vector[element]))
                total = _f32(total + product)
        else:
            try:
                total = _AUDIT.aq4_matvec_reference(
                    indices,
                    scale_indices,
                    codebook,
                    scale_table,
                    tensor_scale,
                    vector,
                    1,
                    cols,
                    group_size,
                )[0]
            except Exception as error:
                raise OracleError(f"AQ4 CPU reference matvec failed at row {row}: {error}") from error
        result.append(finite(total, "CPU dequantized matvec output"))
    return result


def source_matvec(source: SafeTensorReader, item: dict[str, Any], vector: list[float]) -> list[float]:
    shape = item["shape"]
    rows, cols = shape
    if len(vector) != cols:
        raise OracleError("source matvec input length differs")
    result: list[float] = []
    for row in range(rows):
        values = source.read_tensor_row(item["name"], row)
        total = sum(value * vector[index] for index, value in enumerate(values))
        result.append(finite(total, "BF16 source matvec output"))
    return result


def _token_ids_hash(token_ids: list[int]) -> str:
    encoded = (json.dumps(token_ids, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _f32_bits_hex(value: float) -> str:
    return f"0x{struct.unpack('<I', F32.pack(_f32(value)))[0]:08x}"


def _source_tensor_paths(package_manifest: dict[str, Any], qkv_item: dict[str, Any]) -> tuple[dict[str, Path], Path]:
    passthrough = {str(item.get("name")): item for item in package_manifest.get("passthrough_tensors", []) if isinstance(item, dict)}
    names = {
        "qkv": qkv_item.get("source_file"),
        "embedding": (passthrough.get("model.language_model.embed_tokens.weight") or {}).get("source_file"),
        "input_norm": (passthrough.get("model.language_model.layers.0.input_layernorm.weight") or {}).get("source_file"),
    }
    paths: dict[str, Path] = {}
    for label, value in names.items():
        if not isinstance(value, str) or not value:
            raise OracleError(f"source {label} path is missing")
        paths[label] = _canonical_file(Path(value), f"source {label} safetensor")
    source_root = Path(os.path.commonpath([str(path.parent) for path in paths.values()])).resolve()
    if not source_root.is_dir():
        raise OracleError("source canonical root is unavailable")
    for label, path in paths.items():
        try:
            path.relative_to(source_root)
        except ValueError as error:
            raise OracleError(f"source {label} escapes canonical source root") from error
    return paths, source_root


def rmsnorm_f32_input(embedding: list[float], norm: list[float], epsilon: float) -> list[float]:
    if len(embedding) != len(norm) or not embedding:
        raise OracleError("RMSNorm input/weight dimensions differ")
    sum_squares = _f32(0.0)
    for value in embedding:
        product = _f32(_f32(value) * _f32(value))
        sum_squares = _f32(sum_squares + product)
    mean_square = _f32(sum_squares / _f32(len(embedding)))
    epsilon = _f32(epsilon)
    variance = _f32(mean_square + epsilon)
    rms = _f32(math.sqrt(variance))
    scale = _f32(_f32(1.0) / rms)
    # The host runtime computes f32(input * inv_rms), then f32(result * weight).
    # Keep this order: input * weight * inv_rms differs by an observable ULP.
    return [finite(_f32(_f32(_f32(value) * scale) * _f32(norm[index])), "layer-0 input_normed") for index, value in enumerate(embedding)]


def _build_input_normed(package_manifest: dict[str, Any], cases_path: Path, source_rows: dict[tuple[str, int], dict[str, Any]], source_paths: dict[str, Path]) -> tuple[dict[tuple[str, int], list[float]], dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    """Reconstruct the exact layer-0 input from embedding rows and RMSNorm weights.

    Only three token rows and one 4096-element norm vector are retained.  The full
    embedding and QKV tensors are never materialized.
    """
    cases_doc = read_json(cases_path)
    cases = {str(row["case_id"]): row for row in cases_doc.get("cases", []) if isinstance(row, dict)}
    passthrough = {str(item.get("name")): item for item in package_manifest.get("passthrough_tensors", []) if isinstance(item, dict)}
    embed_item = passthrough.get("model.language_model.embed_tokens.weight")
    norm_item = passthrough.get("model.language_model.layers.0.input_layernorm.weight")
    if embed_item is None or norm_item is None:
        raise OracleError("layer-0 embedding/input norm passthrough identity is missing")
    embed_reader = SafeTensorReader(source_paths["embedding"])
    norm_reader = SafeTensorReader(source_paths["input_norm"])
    raw_norm = norm_reader.read_tensor_vector(norm_item["name"])
    if len(raw_norm) != 4096:
        raise OracleError("layer-0 input norm length differs")
    mean_abs = sum(abs(value) for value in raw_norm) / len(raw_norm)
    if mean_abs >= 0.75:
        raise OracleError("layer-0 input norm is not the expected additive RMSNorm payload")
    # Qwen3.5 stores additive RMSNorm deltas.  Match the runtime's
    # effective_rmsnorm_weight_values() f32 transform before reconstructing
    # input_normed; multiplying the raw BF16 payload would bind a different
    # model input.
    norm = [_f32(value + 1.0) for value in raw_norm]
    norm_identity = {
        "tensor_name": norm_item["name"],
        "raw_payload_sha256": norm_reader.digest_tensor(norm_item["name"]),
        "raw_payload_sha256_expected": norm_item.get("payload_sha256"),
        "transform": "effective_rmsnorm_weight_values: raw_f32 + 1.0_f32",
        "effective_values_f32le_sha256": vector_sha(norm),
        "mean_abs_raw": mean_abs,
        "epsilon_f32": _f32(1.0e-6),
        "epsilon_f32_bits_hex": _f32_bits_hex(1.0e-6),
    }
    if norm_identity["raw_payload_sha256"] != norm_item.get("payload_sha256"):
        raise OracleError("layer-0 input norm payload identity differs")
    vectors: dict[tuple[str, int], list[float]] = {}
    bindings: dict[tuple[str, int], dict[str, Any]] = {}
    for case_id, step in EXPECTED_ROWS:
        case = cases.get(case_id)
        if case is None or not isinstance(case.get("prompt_token_ids"), list):
            raise OracleError(f"fixed case is missing from source cases: {case_id}")
        context = [int(token) for token in case["prompt_token_ids"]]
        for previous_step in range(step):
            previous = source_rows[(case_id, previous_step)]
            greedy = previous.get("greedy_token_id")
            if type(greedy) is not int or greedy < 0:
                raise OracleError(f"source trace greedy token is invalid: {(case_id, previous_step)}")
            context.append(greedy)
        token_id = context[-1]
        embedding = embed_reader.read_tensor_row(embed_item["name"], token_id)
        if len(embedding) != len(norm):
            raise OracleError("embedding/input norm dimensions differ")
        epsilon = _f32(1.0e-6)
        vector = rmsnorm_f32_input(embedding, norm, epsilon)
        vectors[(case_id, step)] = vector
        bindings[(case_id, step)] = {
            "token_id": token_id,
            "context_length": len(context),
            "context_token_ids_sha256": _token_ids_hash(context),
            "input_sha256": vector_sha(vector),
        }
    return vectors, bindings, norm_identity


def vector_sha(values: list[float]) -> str:
    payload = b"".join(F32.pack(finite(value, "input vector")) for value in values)
    return hashlib.sha256(payload).hexdigest()


def input_bindings_sha(bindings: dict[tuple[str, int], dict[str, Any]]) -> str:
    rows = []
    for case_id, step in EXPECTED_ROWS:
        binding = bindings.get((case_id, step))
        if not isinstance(binding, dict):
            raise OracleError(f"input binding is missing: {(case_id, step)}")
        rows.append({"case_id": case_id, "step": step, **binding})
    encoded = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def emit_runtime_input_jsonl(path: Path, vectors: dict[tuple[str, int], list[float]], bindings: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    if os.path.lexists(path):
        raise OracleError(f"refusing to overwrite runtime input sidecar: {path}")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise OracleError(f"runtime input sidecar parent must be a regular directory: {parent}")
    parent = parent.resolve()
    final_path = parent / path.name
    if os.path.lexists(final_path):
        raise OracleError(f"refusing to overwrite runtime input sidecar: {final_path}")
    temp_path = parent / f".{path.name}.tmp-{os.getpid()}"
    if os.path.lexists(temp_path):
        raise OracleError(f"runtime input sidecar temporary path already exists: {temp_path}")
    header = {
        "kind": "header",
        "schema_version": "ullm.aq4_layer0_input_normed_jsonl.v1",
        "tensor_name": DEFAULT_TENSOR,
        "dtype": "f32",
        "shape": [4096],
    }
    try:
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(header, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
            for case_id, step in EXPECTED_ROWS:
                values = vectors[(case_id, step)]
                binding = bindings[(case_id, step)]
                row = {
                    "kind": "case",
                    "case_id": case_id,
                    "step": step,
                    "context_token_ids_sha256": binding["context_token_ids_sha256"],
                    "context_length": binding["context_length"],
                    "input_sha256": binding["input_sha256"],
                    "values": [finite(value, "runtime input value") for value in values],
                }
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, final_path)
        except FileExistsError as error:
            raise OracleError(f"refusing to overwrite runtime input sidecar: {final_path}") from error
    except OSError as error:
        raise OracleError(f"failed to emit runtime input sidecar {final_path}: {error}") from error
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise OracleError(f"failed to remove runtime input sidecar temporary file: {error}") from error
    return _capture_file_identity(final_path, "runtime input sidecar")


def compare_vectors(left: list[float], right: list[float]) -> dict[str, float | bool]:
    if len(left) != len(right):
        raise OracleError("comparison vector lengths differ")
    delta = [a - b for a, b in zip(left, right)]
    max_abs = max((abs(value) for value in delta), default=0.0)
    left_norm = math.sqrt(sum(value * value for value in left))
    ref_norm = math.sqrt(sum(value * value for value in right))
    delta_norm = math.sqrt(sum(value * value for value in delta))
    relative_l2 = delta_norm / max(ref_norm, 1e-30)
    cosine = sum(a * b for a, b in zip(left, right)) / max(left_norm * ref_norm, 1e-30)
    bit_mismatch_count = sum(F32.pack(a) != F32.pack(b) for a, b in zip(left, right))
    return {
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "bit_mismatch_count": bit_mismatch_count,
        "bit_mismatch_rate": bit_mismatch_count / max(len(left), 1),
        "finite": all(math.isfinite(value) for value in delta),
    }


def _load_tensor_outputs(path: Path, item: dict[str, Any], expected_identity: dict[str, Any]) -> tuple[dict[tuple[str, int], list[float]], dict[str, Any]]:
    path = _canonical_file(path, "GPU tensor output")
    output_size = path.stat().st_size
    if output_size > MAX_GPU_OUTPUT_BYTES:
        raise OracleError(f"GPU tensor output exceeds bounded JSON size {MAX_GPU_OUTPUT_BYTES}")
    value = read_json(path)
    if value.get("schema_version") != TENSOR_OUTPUT_SCHEMA or value.get("tensor_name") != item["name"]:
        raise OracleError("GPU tensor output schema/tensor differs")
    identity = value.get("identity")
    if not isinstance(identity, dict):
        raise OracleError("GPU tensor output identity is missing")
    required = ("package_manifest_sha256", "active_manifest_sha256", "input_bindings_sha256", "device", "guard_set_sha256", "operation", "effective_rpb")
    missing = [field for field in required if field not in identity]
    if missing:
        raise OracleError(f"GPU tensor output identity fields are missing: {','.join(missing)}")
    for field in ("package_manifest_sha256", "active_manifest_sha256", "input_bindings_sha256", "guard_set_sha256", "operation"):
        if identity.get(field) != expected_identity.get(field):
            raise OracleError(f"GPU tensor output identity mismatch: {field}")
    if identity.get("device") != expected_identity.get("device"):
        raise OracleError("GPU tensor output device/backend identity mismatch")
    effective_rpb = identity.get("effective_rpb")
    if effective_rpb != expected_identity.get("effective_rpb"):
        raise OracleError("GPU tensor output effective RPB identity mismatch")
    loaded: dict[tuple[str, int], list[float]] = {}
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise OracleError("GPU tensor output rows are missing")
    if len(rows) > MAX_GPU_OUTPUT_ROWS:
        raise OracleError(f"GPU tensor output rows exceed bound {MAX_GPU_OUTPUT_ROWS}")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str) or type(row.get("step")) is not int or not isinstance(row.get("values"), list):
            raise OracleError("tensor output row fields differ")
        key = (row["case_id"], row["step"])
        if key in loaded or key not in EXPECTED_ROWS:
            raise OracleError("tensor output case set differs")
        if len(row["values"]) > MAX_GPU_ROW_ELEMENTS:
            raise OracleError("GPU tensor output row exceeds bounded element count")
        loaded[key] = [finite(number, "tensor output") for number in row["values"]]
    if set(loaded) != set(EXPECTED_ROWS):
        raise OracleError("tensor output must contain all fixed attempt-3 cases")
    return loaded, identity


def _trace_file_identities(root: Path, label: str) -> dict[str, Any]:
    root = _canonical_directory(root, f"{label} root")
    return {
        "manifest": _capture_file_identity(root / "manifest.json", f"{label} manifest", root=root),
        "payload": _capture_file_identity(root / "payload.jsonl", f"{label} payload", root=root),
    }


def _assert_identities(identities: dict[str, Any]) -> None:
    for label, identity in identities.items():
        _assert_file_identity(identity, label)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_gpu_operation != DEFAULT_GPU_OPERATION or args.expected_gpu_rpb_rows != DEFAULT_GPU_RPB_ROWS or args.expected_gpu_threads_per_row != DEFAULT_GPU_THREADS_PER_ROW:
        raise OracleError("GPU operation/RPB overrides are not approved; use the pinned qkv binding")
    if not math.isfinite(args.abs_tol) or not math.isfinite(args.relative_tol) or args.abs_tol < 0.0 or args.relative_tol < 0.0:
        raise OracleError("GPU comparison tolerances must be finite and non-negative")
    package_manifest, item, package_dir, package_sha, package_manifest_identity = _load_package(args.package_dir, args.tensor_name)
    if package_sha != args.expected_package_sha256:
        raise OracleError("package manifest SHA differs from pinned identity")
    package_payload_paths = {
        "package index payload": _canonical_package_payload(package_dir, item["index_file"], "package index payload"),
        "package scale payload": _canonical_package_payload(package_dir, item["scale_file"], "package scale payload"),
        "package codebook payload": _canonical_package_payload(package_dir, item["codebook_file"], "package codebook payload"),
    }
    file_identities: dict[str, Any] = {"package manifest": package_manifest_identity}
    file_identities.update({label: _capture_file_identity(path, label, root=package_dir) for label, path in package_payload_paths.items()})
    gpu_trace_file_identities = _trace_file_identities(args.gpu_trace, "GPU trace")
    source_trace_file_identities = _trace_file_identities(args.source_trace, "source trace")
    file_identities.update({f"GPU trace {key}": value for key, value in gpu_trace_file_identities.items()})
    file_identities.update({f"source trace {key}": value for key, value in source_trace_file_identities.items()})
    gpu_manifest, gpu_rows = _load_trace(args.gpu_trace, package_sha, args.expected_active_sha256)
    source_manifest, source_rows = _load_trace(args.source_trace, None, None)
    _assert_identities({**file_identities})
    if source_manifest.get("source_manifest_sha256") != args.expected_source_trace_sha256:
        raise OracleError("source trace identity differs from pinned identity")
    for key in EXPECTED_ROWS:
        if source_rows[key].get("context_token_ids_sha256") != gpu_rows[key].get("context_token_ids_sha256"):
            raise OracleError(f"source/GPU context binding differs: {key}")
    source_paths, source_root = _source_tensor_paths(package_manifest, item)
    file_identities.update({f"source {key} safetensor": _capture_file_identity(path, f"source {key} safetensor", root=source_root) for key, path in source_paths.items()})
    cases_value = source_manifest.get("cases_path")
    if not isinstance(cases_value, str) or not cases_value or not Path(cases_value).is_absolute():
        raise OracleError("source trace cases_path must be an absolute path")
    cases_path = _canonical_file(Path(cases_value), "cases file")
    file_identities["cases file"] = _capture_file_identity(cases_path, "cases file")
    source_tensor_reader = SafeTensorReader(source_paths["qkv"])
    scale_table = scale_values_e4m3()
    active_identity = gpu_manifest.get("identity")
    if not isinstance(active_identity, dict):
        raise OracleError("GPU trace identity is missing")
    gpu_device = active_identity.get("device")
    if not isinstance(gpu_device, dict) or not isinstance(gpu_device.get("backend"), str) or type(gpu_device.get("index")) is not int:
        raise OracleError("GPU trace device/backend identity is missing")
    guard_set_sha = active_identity.get("guard_set_sha256")
    if not isinstance(guard_set_sha, str) or len(guard_set_sha) != 64:
        raise OracleError("GPU trace guard identity is missing")
    base = {
        "schema_version": SCHEMA,
        "status": "blocked_missing_tensor_output",
        "classification": "inconclusive_missing_tensor_level_output",
        "tensor_name": item["name"],
        "tensor_shape": item["shape"],
        "group_size": item["group_size"],
        "package_manifest_sha256": package_sha,
        "active_manifest_sha256": active_identity["active_manifest_sha256"],
        "source_trace_manifest_sha256": source_manifest.get("source_manifest_sha256"),
        "fixed_case_rows": [{"case_id": case_id, "step": step} for case_id, step in EXPECTED_ROWS],
        "required_gpu_tensor_output_schema": TENSOR_OUTPUT_SCHEMA,
        "source_canonical_root": str(source_root),
        "cpu_f32_reference_contract": {
            "implementation": "explicit_f32_scalar_accumulation",
            "api_target": "ullm_runtime_aq4_matvec_f32",
            "execution": "offline_python_model; runtime API not invoked",
            "operation": "standalone_aq4_matvec_f32 (fused qkv/z/gate/beta wrapper intentionally not compared)",
            "row_scale_count": 0,
            "row_scale_override_present": False,
            "numeric_bound": {"max_abs": 1.0e-4, "relative_l2": 1.0e-4},
            "bit_exact_required": False,
        },
        "numeric_contract": {
            "input_dtype": "f32",
            "weight_source_dtype": "BF16",
            "aq4_accumulator_dtype": "f32",
            "codebook_dtype": "f32le",
            "scale_table_dtype": "f32le",
            "index_packing": "idx4_low_nibble_first",
            "source_reference_accumulator_dtype": "f64_python_reference",
        },
        "gpu_tensor_output_identity_contract": {
            "package_manifest_sha256": package_sha,
            "active_manifest_sha256": active_identity["active_manifest_sha256"],
            "device": {"backend": gpu_device["backend"], "index": gpu_device["index"]},
            "guard_set_sha256": guard_set_sha,
            "operation": args.expected_gpu_operation,
            "effective_rpb": {"rows_per_block": args.expected_gpu_rpb_rows, "threads_per_row": args.expected_gpu_threads_per_row},
        },
        "gpu_comparison_tolerance": {
            "abs_tol": args.abs_tol,
            "relative_tol": args.relative_tol,
            "basis": "predeclared AQ4-vs-GPU probe bound; not a promotion threshold; any finite mismatch is no-go",
        },
        "promotion_eligible": False,
        "nonfinite_counts": {"input_vectors": 0, "cpu_outputs": 0, "runtime_cpu_outputs": 0, "source_outputs": 0},
        "payload_identity": {
            "source_tensor_sha256": source_tensor_reader.digest_tensor(item["name"]),
            "index_sha256": file_identities["package index payload"]["sha256"],
            "scale_sha256": file_identities["package scale payload"]["sha256"],
            "codebook_sha256": file_identities["package codebook payload"]["sha256"],
            "scale_table_count": len(scale_table),
            "scale_table_f32le_sha256": hashlib.sha256(b"".join(F32.pack(value) for value in scale_table)).hexdigest(),
            "tensor_scale": item["tensor_scale"],
            "tensor_scale_f32": _f32(float(item["tensor_scale"])),
            "tensor_scale_f32_bits_hex": _f32_bits_hex(float(item["tensor_scale"])),
            "row_scale_count": 0,
            "row_scale_override_present": False,
        },
        "file_identities": file_identities,
    }
    inputs, input_bindings, norm_identity = _build_input_normed(package_manifest, cases_path, source_rows, source_paths)
    base["input_norm_identity"] = norm_identity
    base["input_bindings_sha256"] = input_bindings_sha(input_bindings)
    if args.emit_runtime_input_jsonl:
        sidecar_identity = emit_runtime_input_jsonl(args.emit_runtime_input_jsonl, inputs, input_bindings)
        file_identities["runtime input sidecar"] = sidecar_identity
        base["runtime_input_sidecar"] = {
            "canonical_path": sidecar_identity["canonical_path"],
            "sha256": sidecar_identity["sha256"],
            "schema_version": "ullm.aq4_layer0_input_normed_jsonl.v1",
            "rows": len(EXPECTED_ROWS),
        }
    expected_gpu_identity = {
        "package_manifest_sha256": package_sha,
        "active_manifest_sha256": active_identity["active_manifest_sha256"],
        "input_bindings_sha256": base["input_bindings_sha256"],
        "device": {"backend": gpu_device["backend"], "index": gpu_device["index"]},
        "guard_set_sha256": guard_set_sha,
        "operation": args.expected_gpu_operation,
        "effective_rpb": {"rows_per_block": args.expected_gpu_rpb_rows, "threads_per_row": args.expected_gpu_threads_per_row},
    }
    gpu_outputs = None
    if args.gpu_tensor_output:
        gpu_output_path = _canonical_file(args.gpu_tensor_output, "GPU tensor output")
        try:
            file_identities["GPU tensor output"] = _capture_file_identity(gpu_output_path, "GPU tensor output")
            gpu_outputs, gpu_output_identity = _load_tensor_outputs(gpu_output_path, item, expected_gpu_identity)
            base["gpu_tensor_output_identity"] = gpu_output_identity
        except OracleError as error:
            base["status"] = "no_go_gpu_tensor_output_identity_mismatch"
            base["classification"] = "no_go_gpu_tensor_output_identity_mismatch"
            base["reason"] = f"GPU tensor output identity rejected: {error}"
            base["rows"] = []
            return base
    rows: list[dict[str, Any]] = []
    gpu_numeric_mismatch = False
    for key in EXPECTED_ROWS:
        vector = inputs[key]
        cpu = dequant_matvec(package_dir, item, vector)
        cpu_runtime_f32 = dequant_matvec(package_dir, item, vector, scalar_f32=True)
        source = source_matvec(source_tensor_reader, item, vector)
        formula_vs_runtime = compare_vectors(cpu, cpu_runtime_f32)
        f32_bound_pass = formula_vs_runtime["max_abs"] <= 1e-4 and formula_vs_runtime["relative_l2"] <= 1e-4
        record: dict[str, Any] = {"case_id": key[0], "step": key[1], "input_binding": input_bindings[key], "cpu_vs_source": compare_vectors(cpu, source), "cpu_formula_vs_f32_reference": formula_vs_runtime, "runtime_cpu_f32_api_invoked": False, "runtime_cpu_f32_bit_exact": False, "runtime_cpu_f32_bound_pass": f32_bound_pass}
        if gpu_outputs is not None:
            gpu = gpu_outputs[key]
            if len(gpu) != len(cpu):
                raise OracleError(f"GPU output length differs: {key}")
            record["cpu_vs_gpu"] = compare_vectors(cpu, gpu)
            record["source_vs_gpu"] = compare_vectors(source, gpu)
            record["gpu_output_sha256"] = vector_sha(gpu)
            record["gpu_output_elements"] = len(gpu)
            close = record["cpu_vs_gpu"]["max_abs"] <= args.abs_tol and record["cpu_vs_gpu"]["relative_l2"] <= args.relative_tol
            record["classification"] = "expected_quantization" if close else "kernel_or_decode_bug"
            gpu_numeric_mismatch = gpu_numeric_mismatch or not close
        else:
            record["classification"] = "inconclusive_missing_gpu_tensor_output"
        rows.append(record)
    _assert_identities(file_identities)
    base["input_contract"] = "source embedding row + effective layer-0 additive RMSNorm (raw BF16 weight + 1.0_f32; epsilon=1e-6_f32)"
    base["cpu_reference"] = "tools/audit_aq4_p2_input_controls.py::aq4_matvec_reference"
    if gpu_outputs is None:
        base["reason"] = "attempt-3 trace has stage samples only; qkv tensor-level GPU output is absent"
    base["rows"] = rows
    if gpu_outputs is None:
        base["status"] = "blocked_missing_gpu_tensor_output"
        base["classification"] = "inconclusive_missing_gpu_tensor_output"
    elif gpu_numeric_mismatch:
        base["status"] = "no_go_gpu_numeric_mismatch"
        base["classification"] = "no_go_gpu_numeric_mismatch"
        base["reason"] = "GPU tensor output contained finite values outside the predeclared abs/relative tolerance"
    else:
        base["status"] = "valid"
        base["classification"] = "expected_quantization"
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--gpu-trace", type=Path, required=True)
    parser.add_argument("--gpu-tensor-output", type=Path)
    parser.add_argument("--emit-runtime-input-jsonl", type=Path)
    parser.add_argument("--tensor-name", default=DEFAULT_TENSOR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-active-sha256", dest="expected_active_sha256", default=DEFAULT_ACTIVE_SHA)
    parser.add_argument("--expected-package-sha256", dest="expected_package_sha256", default=DEFAULT_PACKAGE_SHA)
    parser.add_argument("--expected-source-trace-sha256", dest="expected_source_trace_sha256", default=DEFAULT_SOURCE_TRACE_SHA)
    parser.add_argument("--expected-gpu-operation", default=DEFAULT_GPU_OPERATION)
    parser.add_argument("--expected-gpu-rpb-rows", type=int, default=DEFAULT_GPU_RPB_ROWS)
    parser.add_argument("--expected-gpu-threads-per-row", type=int, default=DEFAULT_GPU_THREADS_PER_ROW)
    parser.add_argument("--abs-tol", type=float, default=1e-3)
    parser.add_argument("--relative-tol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.output.exists() or os.path.lexists(args.output):
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    try:
        report = run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True, sort_keys=True))
        return 0 if report["status"] == "valid" else 2
    except OracleError as error:
        print(f"AQ4 layer-0 matvec oracle blocked: {error}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
