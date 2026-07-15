#!/usr/bin/env python3
"""Build a bounded BF16-source layer-0 Z projection sidecar.

The sidecar uses the exact fixed f32 input rows supplied by the existing
layer-0 family isolation input contract and computes ``input @ Z_bf16.T`` with
an explicit f32 cast/accumulate.  It is an input to the CPU-only diagnostic
command; it never changes a worker or production model path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


INPUT_SCHEMA = "ullm.aq4_layer0_input_normed_jsonl.v1"
SCHEMA = "ullm.aq4_layer0_z_source_sidecar.v1"
TENSOR = "model.language_model.layers.0.linear_attn.in_proj_z.weight"
INPUT_COLS = 4096
OUTPUT_ROWS = 4096
F32 = struct.Struct("<f")


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
    digest = sha256_bytes(raw)
    lines = [line for line in raw.splitlines() if line]
    if not lines:
        raise ValueError("input sidecar is empty")
    header = json.loads(lines[0])
    if header.get("kind") != "header" or header.get("schema_version") != INPUT_SCHEMA:
        raise ValueError("input schema differs")
    if header.get("dtype") != "f32" or header.get("shape") != [INPUT_COLS]:
        raise ValueError("input must be f32 [4096]")
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
    return header, cases, digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        header, cases, input_sha = load_input(args.input)
        index = json.loads((args.source_model / "model.safetensors.index.json").read_text())
        shard_name = index["weight_map"][TENSOR]
        shard_path = args.source_model / shard_name
        source_index_sha = sha256_file(args.source_model / "model.safetensors.index.json")
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            weight = handle.get_tensor(TENSOR)
            if list(weight.shape) != [OUTPUT_ROWS, INPUT_COLS] or weight.dtype != torch.bfloat16:
                raise ValueError("source Z tensor must be BF16 [4096,4096]")
            source_payload = weight.detach().contiguous().view(torch.uint16).numpy().tobytes()
            source_payload_sha = sha256_bytes(source_payload)
            source_f32 = weight.to(dtype=torch.float32)
            rows = []
            for case in cases:
                values = torch.tensor(case["values"], dtype=torch.float32)
                output = torch.matmul(values, source_f32.transpose(0, 1)).contiguous()
                if not bool(torch.isfinite(output).all()):
                    raise ValueError(f"non-finite source Z output for {case['case_id']}")
                rows.append(output)
        payload = b"".join(row.numpy().tobytes() for row in rows)
        if args.output.exists():
            raise ValueError(f"refusing to overwrite output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
        report = {
            "schema_version": SCHEMA,
            "status": "valid",
            "tensor_name": TENSOR,
            "dtype": "f32",
            "shape": [len(rows), OUTPUT_ROWS],
            "row_order": "input_jsonl_order",
            "operation": "input_f32_matmul_source_bf16_weight_cast_f32_accumulate_f32",
            "source": {
                "model_dir": str(args.source_model),
                "index_path": str(args.source_model / "model.safetensors.index.json"),
                "index_sha256": source_index_sha,
                "shard_path": str(shard_path),
                "shard_bytes": shard_path.stat().st_size,
                "tensor_payload_sha256": source_payload_sha,
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
            "output": {"path": str(args.output), "sha256": sha256_file(args.output), "bytes": len(payload)},
            "promotion": False,
            "holdout": "not_run",
            "policy_evaluation": "policy_not_evaluated",
            "thresholds": None,
        }
        report_path = args.output.with_suffix(args.output.suffix + ".json")
        report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, ensure_ascii=True, sort_keys=True))
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"Z source sidecar failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
