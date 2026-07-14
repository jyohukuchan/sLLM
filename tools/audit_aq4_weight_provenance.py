#!/usr/bin/env python3
"""Streaming provenance and weight audit for an AQ4 package.

The production package is deliberately not materialized as one large tensor.  This
tool reads safetensors and AQ4 payloads in bounded group chunks and compares the
source values with the package dequantization equation used by the runtime::

    recon = codebook[idx4] * scale_table[scale_idx] * tensor_scale

It is read-only and is useful when a source/path oracle shows drift but the cause
(quantization, export, or runtime layout) is still unknown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import numpy as np


DEFAULT_SOURCE = Path("/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B")
DEFAULT_PACKAGE = Path("/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package")
DEFAULT_SERVED = Path("/etc/ullm/served-models/active.json")
DEFAULT_NAMES = (
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.input_layernorm.weight",
    "model.language_model.layers.0.post_attention_layernorm.weight",
    "model.language_model.layers.3.self_attn.q_proj.weight",
    "model.language_model.layers.3.self_attn.k_proj.weight",
    "model.language_model.layers.3.self_attn.v_proj.weight",
    "model.language_model.layers.3.self_attn.o_proj.weight",
    "model.language_model.layers.3.mlp.gate_proj.weight",
    "model.language_model.layers.3.mlp.up_proj.weight",
    "model.language_model.layers.3.mlp.down_proj.weight",
    "model.language_model.norm.weight",
    "lm_head.weight",
)


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class SafeTensorReader:
    """Small seek-based reader that never loads a tensor payload wholesale."""

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError(f"{path} has no safetensors header length")
            header_bytes = struct.unpack("<Q", raw)[0]
            header = handle.read(header_bytes)
        if len(header) != header_bytes:
            raise ValueError(f"{path} has a truncated safetensors header")
        self.data_start = 8 + header_bytes
        self.header = json.loads(header.decode("utf-8"))
        self.tensors = {k: v for k, v in self.header.items() if k != "__metadata__"}

    def metadata(self, name: str) -> dict[str, Any]:
        try:
            item = self.tensors[name]
        except KeyError as exc:
            raise KeyError(f"{name} is absent from {self.path}") from exc
        return item

    def _range(self, name: str) -> tuple[int, int]:
        offsets = self.metadata(name).get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"{name} has invalid data_offsets")
        start, end = (int(offsets[0]), int(offsets[1]))
        if start < 0 or end < start:
            raise ValueError(f"{name} has invalid data range")
        return start, end

    def read_range(self, name: str, offset: int, length: int) -> bytes:
        start, end = self._range(name)
        if offset < 0 or length < 0 or offset + length > end - start:
            raise ValueError(f"{name} range {offset}:{offset + length} is outside tensor")
        with self.path.open("rb") as handle:
            handle.seek(self.data_start + start + offset)
            payload = handle.read(length)
        if len(payload) != length:
            raise ValueError(f"{name} read {len(payload)} bytes, expected {length}")
        return payload

    def iter_bytes(self, name: str, chunk_bytes: int = 1 << 20) -> Iterable[bytes]:
        start, end = self._range(name)
        with self.path.open("rb") as handle:
            handle.seek(self.data_start + start)
            remaining = end - start
            while remaining:
                payload = handle.read(min(chunk_bytes, remaining))
                if not payload:
                    raise ValueError(f"{name} ended before its declared payload")
                remaining -= len(payload)
                yield payload

    def digest(self, name: str) -> str:
        digest = hashlib.sha256()
        for payload in self.iter_bytes(name):
            digest.update(payload)
        return digest.hexdigest()


def dtype_bytes(dtype: str) -> int:
    return {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}.get(dtype.upper(), 0)


def decode_values(payload: bytes, dtype: str) -> np.ndarray:
    dtype = dtype.upper()
    if dtype == "BF16":
        if len(payload) % 2:
            raise ValueError("BF16 payload is not 2-byte aligned")
        raw = np.frombuffer(payload, dtype="<u2").astype(np.uint32)
        return (raw << 16).view("<f4")
    if dtype == "F16":
        return np.frombuffer(payload, dtype="<f2").astype(np.float32)
    if dtype == "F32":
        return np.frombuffer(payload, dtype="<f4")
    if dtype == "F64":
        return np.frombuffer(payload, dtype="<f8").astype(np.float32)
    raise ValueError(f"unsupported safetensors dtype {dtype}")


def scale_values(scale_format: str) -> np.ndarray:
    fmt = scale_format.lower()
    if fmt == "e8m0":
        return np.asarray([2.0 ** (code - 127) for code in range(255)], dtype=np.float32)
    if fmt.startswith("u"):
        fmt = fmt[1:]
    if not (fmt.startswith("e") and "m" in fmt):
        raise ValueError(f"unknown AQ scale format {scale_format}")
    exp_bits, mant_bits = (int(part) for part in fmt[1:].split("m", 1))
    bias = (1 << (exp_bits - 1)) - 1
    mant_count = 1 << mant_bits
    values: list[float] = []
    for exponent in range((1 << exp_bits) - 1):
        for mantissa in range(mant_count):
            if exponent == 0:
                if mantissa == 0:
                    continue
                value = (mantissa / mant_count) * (2.0 ** (1 - bias))
            else:
                value = (1.0 + mantissa / mant_count) * (2.0 ** (exponent - bias))
            values.append(value)
    return np.asarray(sorted(set(values)), dtype=np.float32)


def read_binary_window(path: Path, offset: int, length: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        value = handle.read(length)
    if len(value) != length:
        raise ValueError(f"{path} read {len(value)} bytes at {offset}, expected {length}")
    return value


def decode_indices(payload: bytes, elements: int) -> np.ndarray:
    packed = np.frombuffer(payload, dtype=np.uint8)
    values = np.empty(packed.size * 2, dtype=np.uint8)
    values[0::2] = packed & 0x0F
    values[1::2] = packed >> 4
    return values[:elements]


def tensor_elements(shape: list[Any]) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def package_entries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for key in ("tensors", "passthrough_tensors"):
        for item in manifest.get(key, []):
            name = str(item.get("name", ""))
            if name:
                values[name] = item
    return values


def source_readers(entries: dict[str, dict[str, Any]]) -> dict[Path, SafeTensorReader]:
    readers: dict[Path, SafeTensorReader] = {}
    for item in entries.values():
        path = Path(str(item.get("source_file", "")))
        if path not in readers and path.is_file():
            readers[path] = SafeTensorReader(path)
    return readers


def compare_passthrough(
    package_dir: Path,
    item: dict[str, Any],
    source: SafeTensorReader,
    chunk_bytes: int,
) -> dict[str, Any]:
    name = str(item["name"])
    source_meta = source.metadata(name)
    payload_path = package_dir / str(item["payload_file"])
    package_digest = sha256_file(payload_path, chunk_bytes)
    source_digest = source.digest(name)
    source_bytes = int(source_meta["data_offsets"][1]) - int(source_meta["data_offsets"][0])
    package_bytes = payload_path.stat().st_size
    return {
        "kind": "passthrough",
        "name": name,
        "source_file": str(source.path),
        "source_dtype": source_meta.get("dtype"),
        "package_dtype": item.get("dtype"),
        "source_shape": source_meta.get("shape"),
        "package_shape": item.get("shape"),
        "shape_exact": source_meta.get("shape") == item.get("shape"),
        "dtype_exact": source_meta.get("dtype") == item.get("dtype"),
        "elements_exact": source_bytes // max(dtype_bytes(str(source_meta.get("dtype"))), 1)
        == int(item.get("elements", -1)),
        "orientation": "row_major_passthrough",
        "transpose_suspected": source_meta.get("shape") != item.get("shape"),
        "source_sha256": source_digest,
        "package_sha256": package_digest,
        "manifest_payload_sha256": item.get("payload_sha256"),
        "payload_hash_exact": package_digest == item.get("payload_sha256"),
        "source_payload_hash_exact": source_digest == package_digest,
        "source_bytes": source_bytes,
        "package_bytes": package_bytes,
        "payload_encoding": item.get("payload_encoding"),
        "read_chunk_bytes": chunk_bytes,
    }


def compare_quantized(
    package_dir: Path,
    item: dict[str, Any],
    source: SafeTensorReader,
    chunk_groups: int,
) -> dict[str, Any]:
    name = str(item["name"])
    source_meta = source.metadata(name)
    shape = [int(v) for v in item.get("shape", [])]
    elements = tensor_elements(shape)
    source_shape = [int(v) for v in source_meta.get("shape", [])]
    source_dtype = str(source_meta.get("dtype", ""))
    element_size = dtype_bytes(source_dtype)
    group_size = int(item.get("group_size", 0))
    if not shape or group_size <= 0 or elements % group_size:
        raise ValueError(f"{name} has invalid shape/group size")
    if source_shape != shape:
        # Still report the mismatch without trying to interpret a transposed stream.
        return {
            "kind": "quantized",
            "name": name,
            "source_file": str(source.path),
            "source_dtype": source_dtype,
            "package_dtype": item.get("dtype"),
            "source_shape": source_shape,
            "package_shape": shape,
            "shape_exact": False,
            "dtype_exact": source_dtype == item.get("dtype"),
            "elements_exact": False,
            "orientation": "shape_mismatch",
            "transpose_suspected": len(source_shape) == 2 and source_shape == shape[::-1],
            "source_sha256": source.digest(name),
            "index_sha256": (
                sha256_file(package_dir / str(item["index_file"]))
                if (package_dir / str(item["index_file"])).is_file()
                else None
            ),
            "scale_sha256": (
                sha256_file(package_dir / str(item["scale_file"]))
                if (package_dir / str(item["scale_file"])).is_file()
                else None
            ),
            "codebook_sha256": (
                sha256_file(package_dir / str(item["codebook_file"]))
                if (package_dir / str(item["codebook_file"])).is_file()
                else None
            ),
            "error": "source/package shape mismatch",
        }

    groups = elements // group_size
    index_path = package_dir / str(item["index_file"])
    scale_path = package_dir / str(item["scale_file"])
    codebook_path = package_dir / str(item["codebook_file"])
    codebook_bytes = codebook_path.read_bytes()
    if len(codebook_bytes) % 4:
        raise ValueError(f"{name} codebook is not f32 aligned")
    codebook = np.frombuffer(codebook_bytes, dtype="<f4")
    if codebook.size != 16:
        raise ValueError(f"{name} codebook has {codebook.size} entries, expected 16")
    scales = scale_values(str(item.get("scale_format", "")))
    tensor_scale = float(item.get("tensor_scale", 0.0))
    if not math.isfinite(tensor_scale) or tensor_scale <= 0:
        raise ValueError(f"{name} has invalid tensor_scale")
    expected_index_bytes = (elements + 1) // 2
    expected_scale_bytes = groups
    index_bytes = index_path.stat().st_size
    scale_bytes = scale_path.stat().st_size
    digest_source = hashlib.sha256()
    sse = 0.0
    source_sse = 0.0
    max_abs = 0.0
    finite = True
    seen = 0
    scale_min = 255
    scale_max = 0
    index_counts = np.zeros(16, dtype=np.int64)
    chunk_groups = max(1, int(chunk_groups))
    if element_size == 0:
        raise ValueError(f"unsupported source dtype {source_dtype}")
    for group_start in range(0, groups, chunk_groups):
        group_count = min(chunk_groups, groups - group_start)
        chunk_elements = group_count * group_size
        byte_offset = group_start * group_size * element_size
        source_payload = source.read_range(name, byte_offset, chunk_elements * element_size)
        digest_source.update(source_payload)
        source_values = decode_values(source_payload, source_dtype).astype(np.float32, copy=False)
        index_payload = read_binary_window(index_path, (group_start * group_size) // 2, (chunk_elements + 1) // 2)
        indices = decode_indices(index_payload, chunk_elements).astype(np.int64, copy=False)
        scale_indices = np.frombuffer(read_binary_window(scale_path, group_start, group_count), dtype=np.uint8).astype(np.int64)
        if np.any(scale_indices >= scales.size):
            raise ValueError(f"{name} contains scale index outside {scales.size}-entry table")
        scale_min = min(scale_min, int(scale_indices.min()))
        scale_max = max(scale_max, int(scale_indices.max()))
        index_counts += np.bincount(indices, minlength=16)
        combined = np.repeat(scales[scale_indices] * tensor_scale, group_size)
        recon = codebook[indices] * combined
        error = source_values - recon
        finite = finite and bool(np.isfinite(source_values).all()) and bool(np.isfinite(recon).all())
        sse += float(np.dot(error, error))
        source_sse += float(np.dot(source_values, source_values))
        max_abs = max(max_abs, float(np.max(np.abs(error))))
        seen += chunk_elements
    rel = sse / source_sse if source_sse else 0.0
    source_digest = digest_source.hexdigest()
    return {
        "kind": "quantized",
        "name": name,
        "source_file": str(source.path),
        "source_dtype": source_dtype,
        "package_dtype": item.get("dtype"),
        "source_shape": source_shape,
        "package_shape": shape,
        "shape_exact": True,
        "dtype_exact": source_dtype == item.get("dtype"),
        "elements_exact": seen == int(item.get("elements", -1)) == elements,
        "orientation": "row_major_grouped",
        "transpose_suspected": False,
        "source_sha256": source_digest,
        "index_sha256": sha256_file(index_path),
        "scale_sha256": sha256_file(scale_path),
        "codebook_sha256": sha256_file(codebook_path),
        "index_bytes": index_bytes,
        "expected_index_bytes": expected_index_bytes,
        "scale_bytes": scale_bytes,
        "expected_scale_bytes": expected_scale_bytes,
        "payload_lengths_exact": index_bytes == expected_index_bytes and scale_bytes == expected_scale_bytes,
        "group_size": group_size,
        "groups": groups,
        "family": item.get("family"),
        "candidate_id": item.get("candidate_id"),
        "index_file": item.get("index_file"),
        "scale_file": item.get("scale_file"),
        "codebook_file": item.get("codebook_file"),
        "scale_format": item.get("scale_format"),
        "tensor_scale": tensor_scale,
        "codebook": [float(value) for value in codebook],
        "scale_table_sha256": hashlib.sha256(scales.astype("<f4", copy=False).tobytes()).hexdigest(),
        "scale_index_min": scale_min,
        "scale_index_max": scale_max,
        "index_counts": [int(value) for value in index_counts],
        "finite": finite,
        "measured_mse": sse / seen if seen else 0.0,
        "measured_relative_mse": rel,
        "measured_max_abs_error": max_abs,
        "manifest_metrics": item.get("metrics", {}),
        "read_chunk_groups": chunk_groups,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def served_binding(path: Path, package_dir: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = load_json(path)
    product = value.get("product", {}) if isinstance(value, dict) else {}
    package = product.get("package", {}) if isinstance(product, dict) else {}
    manifest_path = package_dir / "manifest.json"
    expected = package.get("manifest_sha256")
    actual = sha256_file(manifest_path)
    return {
        "path": str(path),
        "schema_version": value.get("schema_version"),
        "served_package_manifest_path": package.get("manifest_path"),
        "served_manifest_sha256": expected,
        "actual_package_manifest_sha256": actual,
        "manifest_sha256_exact": expected == actual if expected else False,
        "served_product_root": product.get("root"),
        "package_dir": str(package_dir),
        "package_root_exact": str(package_dir.parent) == product.get("root"),
    }


def run_audit(
    package_dir: Path = DEFAULT_PACKAGE,
    source_dir: Path = DEFAULT_SOURCE,
    names: Iterable[str] = DEFAULT_NAMES,
    chunk_groups: int = 4096,
    served_manifest: Path | None = DEFAULT_SERVED,
) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    source_dir = source_dir.resolve()
    manifest_path = package_dir / "manifest.json"
    manifest = load_json(manifest_path)
    entries = package_entries(manifest)
    selected = [str(name) for name in names]
    selected_items = {name: entries[name] for name in selected if name in entries}
    readers = source_readers(selected_items)
    results: list[dict[str, Any]] = []
    for name in selected:
        item = entries.get(name)
        if item is None:
            results.append({"name": name, "status": "missing_from_package"})
            continue
        source_path = Path(str(item.get("source_file", "")))
        reader = readers.get(source_path)
        if reader is None:
            results.append({"name": name, "status": "missing_source_file", "source_file": str(source_path)})
            continue
        try:
            if "payload_file" in item:
                result = compare_passthrough(package_dir, item, reader, max(1, chunk_groups * 1024))
            else:
                result = compare_quantized(package_dir, item, reader, chunk_groups)
            result["status"] = "ok"
            results.append(result)
        except Exception as exc:  # report a single bad tensor and continue auditing others
            results.append({"name": name, "status": "error", "error": str(exc), "source_file": str(source_path)})
    source_root_exact = Path(str(manifest.get("source_model_dir", ""))).resolve() == source_dir
    return {
        "schema_version": "ullm.qwen35_aq4_weight_provenance_audit.v1",
        "package_dir": str(package_dir),
        "package_manifest_sha256": sha256_file(manifest_path),
        "source_dir": str(source_dir),
        "manifest_source_model_dir": manifest.get("source_model_dir"),
        "source_root_exact": source_root_exact,
        "selected_names": selected,
        "chunk_groups": chunk_groups,
        "read_only": True,
        "served_binding": served_binding(served_manifest, package_dir) if served_manifest else None,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--name", action="append", dest="names", help="Tensor name (repeatable).")
    parser.add_argument(
        "--all-quantized",
        action="store_true",
        help="Audit every manifest tensor (rather than only the representative defaults).",
    )
    parser.add_argument("--chunk-groups", type=int, default=4096)
    parser.add_argument("--served-manifest", type=Path, default=DEFAULT_SERVED)
    parser.add_argument("--no-served-manifest", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.all_quantized and args.names:
        parser_error = "--all-quantized cannot be combined with --name"
        raise SystemExit(parser_error)
    if args.all_quantized:
        manifest = load_json(args.package_dir / "manifest.json")
        names = [str(item["name"]) for item in manifest.get("tensors", [])]
    else:
        names = args.names or DEFAULT_NAMES
    served = None if args.no_served_manifest else args.served_manifest
    report = run_audit(args.package_dir, args.source_dir, names, args.chunk_groups, served)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if all(item.get("status") == "ok" for item in report["results"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
