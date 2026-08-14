#!/usr/bin/env python3
"""Hash one safetensors range without materializing or storing raw slice bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any

MAX_HEADER_BYTES = 64 * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024


class SliceError(RuntimeError):
    pass


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SliceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _regular_fd(path: Path) -> int:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise SliceError(f"cannot open regular non-symlink file {path}: {exc}") from exc
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(fd)
        raise SliceError(f"slice input is not a regular file: {path}")
    return fd


def _read_exact(fd: int, offset: int, length: int, label: str) -> bytes:
    data = os.pread(fd, length, offset)
    if len(data) != length:
        raise SliceError(f"short read for {label}")
    return data


def _load_json(fd: int, length: int, offset: int, label: str) -> Any:
    try:
        return json.loads(
            _read_exact(fd, offset, length, label),
            object_pairs_hook=_object_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SliceError(f"invalid UTF-8 JSON for {label}: {exc}") from exc


def _safe_shard_name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise SliceError("safetensors index has an invalid shard name")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.name != value:
        raise SliceError("safetensors shard must be one root-local basename")
    return value


def hash_slice(cache_root: Path, tensor_name: str, relative_offset: int, length: int) -> dict[str, Any]:
    if relative_offset < 0 or length <= 0:
        raise SliceError("slice offset must be non-negative and length must be positive")
    index_path = cache_root / "model.safetensors.index.json"
    index_fd = _regular_fd(index_path)
    try:
        index_size = os.fstat(index_fd).st_size
        index = _load_json(index_fd, index_size, 0, "safetensors index")
    finally:
        os.close(index_fd)
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise SliceError("safetensors index has no weight_map object")
    shard_name = _safe_shard_name(index["weight_map"].get(tensor_name))

    shard_path = cache_root / shard_name
    shard_fd = _regular_fd(shard_path)
    try:
        shard_size = os.fstat(shard_fd).st_size
        header_length = int.from_bytes(_read_exact(shard_fd, 0, 8, "header length"), "little")
        if header_length <= 0 or header_length > MAX_HEADER_BYTES or 8 + header_length > shard_size:
            raise SliceError("safetensors header length is outside the bounded file")
        header = _load_json(shard_fd, header_length, 8, "safetensors header")
        entry = header.get(tensor_name) if isinstance(header, dict) else None
        if not isinstance(entry, dict):
            raise SliceError("tensor is absent from the indexed shard header")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or any(type(value) is not int or value < 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int or value < 0 for value in offsets)
            or offsets[1] < offsets[0]
        ):
            raise SliceError("tensor metadata is malformed")
        tensor_bytes = offsets[1] - offsets[0]
        slice_end = relative_offset + length
        if slice_end < relative_offset or slice_end > tensor_bytes:
            raise SliceError("slice range is outside the tensor payload")
        absolute_start = 8 + header_length + offsets[0] + relative_offset
        absolute_end = absolute_start + length
        if absolute_end > shard_size:
            raise SliceError("slice range is outside the shard file")
        digest = hashlib.sha256()
        cursor = absolute_start
        while cursor < absolute_end:
            count = min(READ_CHUNK_BYTES, absolute_end - cursor)
            digest.update(_read_exact(shard_fd, cursor, count, "slice payload"))
            cursor += count
    finally:
        os.close(shard_fd)

    return {
        "schema_version": "safetensors-slice-hash-v1",
        "tensor_name": tensor_name,
        "source_shard": shard_name,
        "dtype": dtype,
        "shape": shape,
        "header_length_bytes": header_length,
        "tensor_data_offsets": offsets,
        "slice_relative_range": [relative_offset, slice_end],
        "absolute_byte_range": [absolute_start, absolute_end],
        "size_bytes": length,
        "sha256": digest.hexdigest(),
        "raw_stored": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--offset", required=True, type=int)
    parser.add_argument("--length", required=True, type=int)
    args = parser.parse_args()
    try:
        report = hash_slice(args.cache_root, args.tensor, args.offset, args.length)
    except (OSError, SliceError) as exc:
        parser.exit(1, f"slice hash: FAIL: {exc}\n")
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
