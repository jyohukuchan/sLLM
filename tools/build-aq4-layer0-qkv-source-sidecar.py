#!/usr/bin/env python3
"""Build a bounded BF16-source layer-0 QKV projection sidecar.

The tool streams output-row chunks so the BF16 tensor is never duplicated as
one full f32 matrix. It is an input to a CPU-only diagnostic command and does
not alter any worker or production model path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


INPUT_SCHEMA = "ullm.aq4_layer0_input_normed_jsonl.v1"
SCHEMA = "ullm.aq4_layer0_qkv_source_sidecar.v1"
TENSOR = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
INPUT_COLS = 4096
OUTPUT_ROWS = 8192
ROW_CHUNK = 256


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_input(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    raw = path.read_bytes()
    lines = [line for line in raw.splitlines() if line]
    if not lines:
        raise ValueError("input sidecar is empty")
    header = json.loads(lines[0])
    if (
        header.get("kind") != "header"
        or header.get("schema_version") != INPUT_SCHEMA
        or header.get("tensor_name") != TENSOR
        or header.get("dtype") != "f32"
        or header.get("shape") != [INPUT_COLS]
    ):
        raise ValueError("input contract differs")
    cases: list[dict[str, Any]] = []
    for line in lines[1:]:
        case = json.loads(line)
        values = case.get("values")
        if case.get("kind") != "case" or not isinstance(values, list) or len(values) != INPUT_COLS:
            raise ValueError(f"invalid input case {case.get('case_id')}")
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f"non-finite input case {case.get('case_id')}")
        actual = sha256_bytes(struct.pack(f"<{len(values)}f", *values))
        if case.get("input_sha256") != actual:
            raise ValueError(f"input hash differs for {case.get('case_id')}")
        cases.append(case)
    if not cases or len(cases) > 4096:
        raise ValueError("input case count is outside bounds")
    return header, cases, sha256_bytes(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    try:
        if args.output.exists() or metadata_path.exists():
            raise ValueError(f"refusing to overwrite output: {args.output}")
        header, cases, input_sha = load_input(args.input)
        index_path = args.source_model / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        shard_name = index["weight_map"][TENSOR]
        shard_path = args.source_model / shard_name
        source_payload_digest = hashlib.sha256()
        output_digest = hashlib.sha256()
        output_bytes = 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            weight = handle.get_tensor(TENSOR)
            if list(weight.shape) != [OUTPUT_ROWS, INPUT_COLS] or weight.dtype != torch.bfloat16:
                raise ValueError("source QKV tensor must be BF16 [8192,4096]")
            for row_start in range(0, OUTPUT_ROWS, ROW_CHUNK):
                row_end = min(row_start + ROW_CHUNK, OUTPUT_ROWS)
                bf16_chunk = weight[row_start:row_end].detach().contiguous()
                source_payload_digest.update(bf16_chunk.view(torch.uint16).numpy().tobytes())
            with args.output.open("xb") as output_handle:
                for case in cases:
                    values = torch.tensor(case["values"], dtype=torch.float32)
                    for row_start in range(0, OUTPUT_ROWS, ROW_CHUNK):
                        row_end = min(row_start + ROW_CHUNK, OUTPUT_ROWS)
                        weight_chunk = weight[row_start:row_end].to(dtype=torch.float32)
                        result = torch.matmul(values, weight_chunk.transpose(0, 1)).contiguous()
                        if not bool(torch.isfinite(result).all()):
                            raise ValueError(f"non-finite source QKV output for {case['case_id']}")
                        raw = result.numpy().tobytes()
                        output_handle.write(raw)
                        output_digest.update(raw)
                        output_bytes += len(raw)
                output_handle.flush()
                os.fsync(output_handle.fileno())
        expected_bytes = len(cases) * OUTPUT_ROWS * 4
        if output_bytes != expected_bytes:
            raise ValueError(f"output size differs: got {output_bytes} expected {expected_bytes}")
        report = {
            "schema_version": SCHEMA,
            "status": "valid",
            "tensor_name": TENSOR,
            "dtype": "f32",
            "shape": [len(cases), OUTPUT_ROWS],
            "row_order": "input_jsonl_order",
            "operation": "input_f32_matmul_source_bf16_weight_cast_f32_accumulate_f32",
            "source": {
                "model_dir": str(args.source_model),
                "index_path": str(index_path),
                "index_sha256": sha256_file(index_path),
                "shard_path": str(shard_path),
                "shard_bytes": shard_path.stat().st_size,
                "tensor_payload_sha256": source_payload_digest.hexdigest(),
                "tensor_shape": [OUTPUT_ROWS, INPUT_COLS],
                "tensor_dtype": "BF16",
            },
            "input": {
                "path": str(args.input),
                "sha256": input_sha,
                "schema": header["schema_version"],
                "rows": len(cases),
                "shape": [INPUT_COLS],
            },
            "output": {
                "path": str(args.output),
                "sha256": output_digest.hexdigest(),
                "bytes": output_bytes,
            },
            "memory_policy": {"output_row_chunk": ROW_CHUNK, "full_f32_weight_materialized": False},
            "promotion": False,
            "holdout": "not_run",
            "policy_evaluation": "policy_not_evaluated",
            "thresholds": None,
        }
        metadata_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, ensure_ascii=True, sort_keys=True))
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        if args.output.exists() and not metadata_path.exists():
            args.output.unlink()
        print(f"QKV source sidecar failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
