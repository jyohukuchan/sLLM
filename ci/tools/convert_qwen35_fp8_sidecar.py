#!/usr/bin/env python3
"""Create or validate a Qwen3.5 text-linear E4M3FN sidecar.

The source BF16 checkpoint remains immutable and supplies embeddings, norms,
attention state, and every non-linear tensor.  This artifact owns only text
linear FP8 values and their outer-dimension FP32 scales, which lets model load
retain one verified source lock while making quantized resident ownership and
provenance explicit.
"""

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

SCHEMA = "sllm-qwen35-fp8-sidecar-v1"
VALUE_DTYPE = "F8_E4M3"
SCALE_DTYPE = "F32"
SCALE_SUFFIX = ".sllm_fp8_scale"
MAX_E4M3FN = np.float32(448.0)


class ContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def read_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as source:
        raw = source.read(8)
        if len(raw) != 8:
            raise ContractError(f"truncated safetensors header length: {path}")
        length = struct.unpack("<Q", raw)[0]
        if length == 0 or length > 256 * 1024 * 1024:
            raise ContractError(f"invalid safetensors header length: {path}")
        header = json.loads(source.read(length))
    if not isinstance(header, dict):
        raise ContractError(f"safetensors header is not an object: {path}")
    return 8 + length, header


def is_text_linear(name: str, metadata: dict[str, Any]) -> bool:
    if not name.startswith("model.language_model.layers."):
        return False
    if metadata.get("dtype") != "BF16" or len(metadata.get("shape", [])) != 2:
        return False
    return name.endswith(
        (
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
            ".self_attn.o_proj.weight",
            ".linear_attn.in_proj_qkv.weight",
            ".linear_attn.in_proj_z.weight",
            ".linear_attn.in_proj_b.weight",
            ".linear_attn.in_proj_a.weight",
            ".linear_attn.out_proj.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
        )
    )


def decode_table() -> np.ndarray:
    result = np.empty(127, dtype=np.float32)
    for bits in range(127):
        exponent = (bits >> 3) & 0x0F
        mantissa = bits & 0x07
        if exponent == 0:
            result[bits] = np.float32(mantissa * 2.0**-9)
        else:
            result[bits] = np.float32((1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))
    return result


POSITIVE_VALUES = decode_table()


def encode_e4m3fn(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.float32:
        values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ContractError("converter refuses non-finite source values")
    sign = np.signbit(values).astype(np.uint8) << np.uint8(7)
    magnitude = np.minimum(np.abs(values), MAX_E4M3FN)
    upper = np.searchsorted(POSITIVE_VALUES, magnitude, side="left")
    upper = np.minimum(upper, 126)
    lower = np.maximum(upper - 1, 0)
    lower_error = np.abs(magnitude - POSITIVE_VALUES[lower])
    upper_error = np.abs(POSITIVE_VALUES[upper] - magnitude)
    choose_upper = upper_error < lower_error
    ties = upper_error == lower_error
    choose_upper |= ties & ((upper & 1) == 0) & ((lower & 1) != 0)
    encoded = np.where(choose_upper, upper, lower).astype(np.uint8)
    return encoded | sign


def bf16_bytes_to_fp32(raw: memoryview, shape: list[int]) -> np.ndarray:
    elements = int(np.prod(shape, dtype=np.int64))
    if len(raw) != elements * 2:
        raise ContractError("BF16 tensor byte count differs from shape")
    bf16 = np.frombuffer(raw, dtype="<u2", count=elements)
    words = bf16.astype(np.uint32) << np.uint32(16)
    return words.view(np.float32).reshape(shape)


def quantize_outer_rows(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
        raise ContractError("FP8 source must be a non-empty rank-2 matrix")
    if not np.isfinite(source).all():
        raise ContractError("converter refuses non-finite source values")
    amax = np.max(np.abs(source), axis=1).astype(np.float32)
    scales = np.where(amax == 0, np.float32(1.0), amax / MAX_E4M3FN).astype("<f4")
    values = encode_e4m3fn(source / scales[:, None])
    return np.ascontiguousarray(values), np.ascontiguousarray(scales)


def source_catalog(cache: Path) -> dict[str, tuple[Path, int, dict[str, Any]]]:
    index = json.loads((cache / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ContractError("source index has no weight_map")
    headers: dict[str, tuple[int, dict[str, Any]]] = {}
    result: dict[str, tuple[Path, int, dict[str, Any]]] = {}
    for name, shard_name in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard_name, str) or Path(shard_name).name != shard_name:
            raise ContractError("source index contains an unsafe tensor or shard name")
        shard = cache / shard_name
        if shard_name not in headers:
            headers[shard_name] = read_header(shard)
        data_start, header = headers[shard_name]
        metadata = header.get(name)
        if not isinstance(metadata, dict):
            raise ContractError(f"source tensor is absent from shard header: {name}")
        result[name] = (shard, data_start, metadata)
    return result


def planned_tensors(catalog: dict[str, tuple[Path, int, dict[str, Any]]]) -> list[str]:
    selected = sorted(name for name, (_, _, metadata) in catalog.items() if is_text_linear(name, metadata))
    if not selected:
        raise ContractError("source checkpoint has no selected text-linear tensors")
    return selected


def artifact_header(catalog: dict[str, tuple[Path, int, dict[str, Any]]], selected: list[str]) -> tuple[bytes, dict[str, tuple[int, int]], int]:
    header: dict[str, Any] = {"__metadata__": {"format": "sllm-fp8-sidecar-v1"}}
    ranges: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name in selected:
        shape = catalog[name][2]["shape"]
        elements = int(np.prod(shape, dtype=np.int64))
        ranges[name] = (cursor, cursor + elements)
        header[name] = {"dtype": VALUE_DTYPE, "shape": shape, "data_offsets": list(ranges[name])}
        cursor += elements
        scale_name = name + SCALE_SUFFIX
        ranges[scale_name] = (cursor, cursor + shape[0] * 4)
        header[scale_name] = {"dtype": SCALE_DTYPE, "shape": [shape[0]], "data_offsets": list(ranges[scale_name])}
        cursor += shape[0] * 4
    encoded = canonical_bytes(header)
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    return encoded, ranges, cursor


def convert(args: argparse.Namespace) -> None:
    catalog = source_catalog(args.cache)
    selected = planned_tensors(catalog)
    header, ranges, payload_size = artifact_header(catalog, selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    records: list[dict[str, Any]] = []
    with temporary.open("w+b") as destination:
        destination.write(struct.pack("<Q", len(header)))
        destination.write(header)
        destination.truncate(8 + len(header) + payload_size)
        output_map = mmap.mmap(destination.fileno(), 0)
        try:
            for ordinal, name in enumerate(selected, 1):
                shard, data_start, metadata = catalog[name]
                start, end = metadata["data_offsets"]
                with shard.open("rb") as source_file, mmap.mmap(source_file.fileno(), 0, access=mmap.ACCESS_READ) as source_map:
                    source_bytes = memoryview(source_map)[data_start + start : data_start + end]
                    matrix = bf16_bytes_to_fp32(source_bytes, metadata["shape"])
                    values, scales = quantize_outer_rows(matrix)
                    source_hash = "sha256:" + hashlib.sha256(source_bytes).hexdigest()
                    del source_bytes
                value_start, value_end = ranges[name]
                scale_start, scale_end = ranges[name + SCALE_SUFFIX]
                payload_start = 8 + len(header)
                output_map[payload_start + value_start : payload_start + value_end] = values.tobytes()
                output_map[payload_start + scale_start : payload_start + scale_end] = scales.tobytes()
                records.append(
                    {
                        "name": name,
                        "shape": metadata["shape"],
                        "source_dtype": "BF16",
                        "source_sha256": source_hash,
                        "value_dtype": "F8_E4M3FN",
                        "value_sha256": "sha256:" + hashlib.sha256(values).hexdigest(),
                        "scale_dtype": "F32",
                        "scale_granularity": "outer-dimension",
                        "scale_axis": 0,
                        "scale_sha256": "sha256:" + hashlib.sha256(scales).hexdigest(),
                        "rounding": "nearest-even",
                        "saturation": "finite-448",
                        "zero_point": False,
                    }
                )
                print(f"[{ordinal}/{len(selected)}] {name}", file=sys.stderr)
        finally:
            output_map.flush()
            output_map.close()
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(args.output)

    source_lock = json.loads(args.source_lock.read_text())
    manifest = {
        "schema_version": SCHEMA,
        "source": {
            "repo_id": source_lock["model"]["repo_id"],
            "resolved_revision": source_lock["model"]["resolved_revision"],
            "lock_fingerprint": source_lock["fingerprint"],
            "lock_sha256": sha256_file(args.source_lock),
        },
        "tool": {
            "repository": args.tool_repository,
            "commit": args.tool_commit,
            "path": "ci/tools/convert_qwen35_fp8_sidecar.py",
            "sha256": sha256_file(Path(__file__).resolve()),
            "numpy": np.__version__,
            "arguments": {"scale": "outer-dimension-f32", "encoding": "OCP-E4M3FN"},
        },
        "artifact": {
            "path": args.output.name,
            "size_bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
            "tensor_count": len(records),
            "scale_tensor_count": len(records),
        },
        "tensors": records,
    }
    manifest["fingerprint"] = "sha256:" + hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    args.manifest.write_bytes(canonical_bytes(manifest) + b"\n")
    validate(args.output, args.manifest, args.source_lock)
    print(f"FP8 sidecar: PASS tensors={len(records)} artifact={args.output}")


def validate(artifact: Path, manifest_path: Path, source_lock_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA:
        raise ContractError("FP8 sidecar manifest schema is unsupported")
    claimed = manifest.pop("fingerprint", None)
    actual = "sha256:" + hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    manifest["fingerprint"] = claimed
    if claimed != actual:
        raise ContractError("FP8 sidecar manifest fingerprint differs")
    if manifest["source"]["lock_sha256"] != sha256_file(source_lock_path):
        raise ContractError("FP8 sidecar source lock digest differs")
    if manifest["artifact"]["sha256"] != sha256_file(artifact):
        raise ContractError("FP8 sidecar artifact digest differs")
    data_start, header = read_header(artifact)
    if data_start >= artifact.stat().st_size:
        raise ContractError("FP8 sidecar has no payload")
    records = manifest.get("tensors")
    if not isinstance(records, list) or len(records) != manifest["artifact"]["tensor_count"]:
        raise ContractError("FP8 sidecar tensor count differs")
    with artifact.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        for record in records:
            name = record["name"]
            value = header.get(name)
            scale = header.get(name + SCALE_SUFFIX)
            if value is None or scale is None or value["shape"] != record["shape"]:
                raise ContractError(f"FP8 sidecar tensor/scale metadata differs: {name}")
            if value["dtype"] != VALUE_DTYPE or scale["dtype"] != SCALE_DTYPE:
                raise ContractError(f"FP8 sidecar tensor/scale dtype differs: {name}")
            for metadata, digest_name in ((value, "value_sha256"), (scale, "scale_sha256")):
                begin, end = metadata["data_offsets"]
                digest = "sha256:" + hashlib.sha256(mapped[data_start + begin : data_start + end]).hexdigest()
                if digest != record[digest_name]:
                    raise ContractError(f"FP8 sidecar tensor payload digest differs: {name}")


def self_test() -> None:
    for columns in (127, 128, 129):
        source = np.linspace(-9.0, 7.0, columns, dtype=np.float32).reshape(1, columns)
        values, scales = quantize_outer_rows(source)
        if values.shape != source.shape or scales.shape != (1,):
            raise ContractError(f"boundary fixture failed: {columns}")
    fixtures = np.array([0.0, -0.0, 2.0**-9, 448.0, -448.0], dtype=np.float32)
    expected = np.array([0x00, 0x80, 0x01, 0x7E, 0xFE], dtype=np.uint8)
    if not np.array_equal(encode_e4m3fn(fixtures), expected):
        raise ContractError("E4M3FN special fixture differs")
    for value in (np.nan, np.inf, -np.inf):
        try:
            quantize_outer_rows(np.array([[value]], dtype=np.float32))
        except ContractError:
            pass
        else:
            raise ContractError("non-finite fixture was accepted")
    print("FP8 converter self-test: PASS boundaries=127,128,129 special=zero/subnormal/max/nan/inf")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--cache", type=Path, required=True)
    convert_parser.add_argument("--source-lock", type=Path, required=True)
    convert_parser.add_argument("--output", type=Path, required=True)
    convert_parser.add_argument("--manifest", type=Path, required=True)
    convert_parser.add_argument("--tool-repository", required=True)
    convert_parser.add_argument("--tool-commit", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--artifact", type=Path, required=True)
    validate_parser.add_argument("--manifest", type=Path, required=True)
    validate_parser.add_argument("--source-lock", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "convert":
            convert(args)
        else:
            validate(args.artifact, args.manifest, args.source_lock)
            print("FP8 sidecar validation: PASS")
        return 0
    except (ContractError, OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"FP8 sidecar: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
