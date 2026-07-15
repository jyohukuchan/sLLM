#!/usr/bin/env python3
"""Build bounded BF16-source layer-0 A/B projection sidecars.

The layer-0 hybrid isolation diagnostic uses the same three normalized input
rows for each projection family.  This tool computes the source reference
matvecs directly from the BF16 safetensors payload and writes little-endian
f32 rows.  Source tensors are processed one projection row chunk at a time;
the full f32 weight matrix is never materialized.

The output is diagnostic evidence only.  It does not alter the worker or any
production model path, and the metadata deliberately keeps policy and
promotion fail-closed.
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
SCHEMA = "ullm.aq4_layer0_ab_source_sidecar.v1"
INPUT_TENSOR = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
INPUT_COLS = 4096
EXPECTED_ROWS = 3
OUTPUT_ROWS = 32
ROW_CHUNK = 32

FAMILIES: dict[str, tuple[str, str]] = {
    "a": (
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
        "source-a.f32le",
    ),
    "b": (
        "model.language_model.layers.0.linear_attn.in_proj_b.weight",
        "source-b.f32le",
    ),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _f32_bytes(values: torch.Tensor) -> bytes:
    """Return an explicit little-endian f32 representation."""

    array = values.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
    # The CI/production hosts are little-endian.  The explicit dtype keeps the
    # artifact contract independent of NumPy's native dtype spelling.
    return array.astype("<f4", copy=False).tobytes()


def _bf16_payload_bytes(values: torch.Tensor) -> bytes:
    """Hash the logical BF16 payload, not its f32 conversion."""

    payload = values.detach().to(device="cpu").contiguous().view(torch.uint16).numpy()
    return payload.astype("<u2", copy=False).tobytes()


def load_input(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Read and validate the existing normalized three-row input contract."""

    raw = path.read_bytes()
    lines = [line for line in raw.splitlines() if line]
    if not lines:
        raise ValueError("input sidecar is empty")
    header = json.loads(lines[0])
    if (
        header.get("kind") != "header"
        or header.get("schema_version") != INPUT_SCHEMA
        or header.get("tensor_name") != INPUT_TENSOR
        or header.get("dtype") != "f32"
        or header.get("shape") != [INPUT_COLS]
    ):
        raise ValueError("input contract differs")
    cases: list[dict[str, Any]] = []
    for line in lines[1:]:
        case = json.loads(line)
        values = case.get("values")
        if (
            case.get("kind") != "case"
            or not isinstance(values, list)
            or len(values) != INPUT_COLS
        ):
            raise ValueError(f"invalid input case {case.get('case_id')}")
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f"non-finite input case {case.get('case_id')}")
        actual = sha256_bytes(struct.pack(f"<{len(values)}f", *values))
        if case.get("input_sha256") != actual:
            raise ValueError(f"input hash differs for {case.get('case_id')}")
        cases.append(case)
    if len(cases) != EXPECTED_ROWS:
        raise ValueError(f"input must contain exactly {EXPECTED_ROWS} rows")
    return header, cases, sha256_bytes(raw)


def _output_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _resolve_outputs(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir is not None:
        if args.output_a is not None or args.output_b is not None:
            raise ValueError("--output-dir cannot be combined with --output-a/--output-b")
        return {family: args.output_dir / filename for family, (_, filename) in FAMILIES.items()}
    if args.output_a is None or args.output_b is None:
        raise ValueError("provide both --output-a and --output-b, or --output-dir")
    return {"a": args.output_a, "b": args.output_b}


def _report(
    *,
    family: str,
    tensor_name: str,
    source_model: Path,
    index_path: Path,
    shard_path: Path,
    source_payload_sha: str,
    input_path: Path,
    input_header: dict[str, Any],
    input_sha: str,
    cases: list[dict[str, Any]],
    output_path: Path,
    output_sha: str,
    output_bytes: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": "valid",
        "family": family,
        "tensor_name": tensor_name,
        "dtype": "f32",
        "shape": [len(cases), OUTPUT_ROWS],
        "row_order": "input_jsonl_order",
        "operation": "input_f32_matmul_source_bf16_weight_cast_f32_accumulate_f32",
        "source": {
            "model_dir": str(source_model),
            "index_path": str(index_path),
            "index_sha256": sha256_file(index_path),
            "shard_path": str(shard_path),
            "shard_bytes": shard_path.stat().st_size,
            "tensor_payload_sha256": source_payload_sha,
            "tensor_shape": [OUTPUT_ROWS, INPUT_COLS],
            "tensor_dtype": "BF16",
        },
        "input": {
            "path": str(input_path),
            "sha256": input_sha,
            "schema": input_header["schema_version"],
            "rows": len(cases),
            "shape": [INPUT_COLS],
            "steps": [case["step"] for case in cases],
        },
        "output": {
            "path": str(output_path),
            "sha256": output_sha,
            "bytes": output_bytes,
        },
        "rows": [
            {
                "case_id": case["case_id"],
                "step": case["step"],
                "input_sha256": case["input_sha256"],
                "offset_bytes": index * OUTPUT_ROWS * 4,
                "bytes": OUTPUT_ROWS * 4,
            }
            for index, case in enumerate(cases)
        ],
        "memory_policy": {
            "output_row_chunk": ROW_CHUNK,
            "full_f32_weight_materialized": False,
            "source_families_processed_sequentially": True,
        },
        "promotion": False,
        "holdout": "not_run",
        "policy_evaluation": "policy_not_evaluated",
        "thresholds": None,
    }


def build(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    outputs = _resolve_outputs(args)
    output_meta = {family: _output_metadata_path(path) for family, path in outputs.items()}
    all_paths = [*outputs.values(), *output_meta.values()]
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("output paths overlap")
    if any(path.exists() for path in all_paths):
        raise ValueError("refusing to overwrite output sidecar or metadata")

    header, cases, input_sha = load_input(args.input)
    index_path = args.source_model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("source index has no weight_map")

    for path in all_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_outputs: dict[str, Path] = {}
    committed_paths: list[Path] = []
    try:
        # Each family is opened and completed before the next one.  This keeps
        # the peak source tensor residency bounded even when shard placement
        # differs between A and B.
        for family, (tensor_name, _) in FAMILIES.items():
            shard_name = weight_map.get(tensor_name)
            if not isinstance(shard_name, str) or not shard_name:
                raise ValueError(f"source index has no shard for {family}")
            shard_path = args.source_model / shard_name
            if not shard_path.is_file():
                raise ValueError(f"source shard missing for {family}: {shard_path}")

            output_path = outputs[family]
            temporary = output_path.with_name(output_path.name + ".tmp")
            if temporary.exists():
                raise ValueError(f"temporary output exists: {temporary}")
            temporary_outputs[family] = temporary
            output_digest = hashlib.sha256()
            output_bytes = 0
            source_digest = hashlib.sha256()
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                weight = handle.get_tensor(tensor_name)
                if list(weight.shape) != [OUTPUT_ROWS, INPUT_COLS] or weight.dtype != torch.bfloat16:
                    raise ValueError(f"source {family} tensor must be BF16 [32,4096]")
                for row_start in range(0, OUTPUT_ROWS, ROW_CHUNK):
                    row_end = min(row_start + ROW_CHUNK, OUTPUT_ROWS)
                    source_digest.update(_bf16_payload_bytes(weight[row_start:row_end]))
                with temporary.open("xb") as output_handle:
                    for case in cases:
                        values = torch.tensor(case["values"], dtype=torch.float32)
                        for row_start in range(0, OUTPUT_ROWS, ROW_CHUNK):
                            row_end = min(row_start + ROW_CHUNK, OUTPUT_ROWS)
                            weight_chunk = weight[row_start:row_end].to(dtype=torch.float32)
                            result = torch.matmul(values, weight_chunk.transpose(0, 1)).contiguous()
                            if not bool(torch.isfinite(result).all()):
                                raise ValueError(f"non-finite source {family} output for {case['case_id']}")
                            raw = _f32_bytes(result)
                            output_handle.write(raw)
                            output_digest.update(raw)
                            output_bytes += len(raw)
                    output_handle.flush()
                    os.fsync(output_handle.fileno())

            expected_bytes = len(cases) * OUTPUT_ROWS * 4
            if output_bytes != expected_bytes:
                raise ValueError(
                    f"output size differs for {family}: got {output_bytes} expected {expected_bytes}"
                )
            report = _report(
                family=family,
                tensor_name=tensor_name,
                source_model=args.source_model,
                index_path=index_path,
                shard_path=shard_path,
                source_payload_sha=source_digest.hexdigest(),
                input_path=args.input,
                input_header=header,
                input_sha=input_sha,
                cases=cases,
                output_path=output_path,
                output_sha=output_digest.hexdigest(),
                output_bytes=output_bytes,
            )
            temporary_meta = output_meta[family].with_name(output_meta[family].name + ".tmp")
            if temporary_meta.exists():
                raise ValueError(f"temporary metadata exists: {temporary_meta}")
            temporary_meta.write_text(
                json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, output_path)
            os.replace(temporary_meta, output_meta[family])
            committed_paths.extend((output_path, output_meta[family]))
        return {
            family: json.loads(output_meta[family].read_text(encoding="utf-8"))
            for family in FAMILIES
        }
    except BaseException:
        for path in temporary_outputs.values():
            path.unlink(missing_ok=True)
        for path in output_meta.values():
            path.with_name(path.name + ".tmp").unlink(missing_ok=True)
        for path in committed_paths:
            path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-a", type=Path)
    parser.add_argument("--output-b", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        torch.set_num_threads(1)
        # Inter-op parallelism can already be initialized when this helper is
        # embedded by a test harness.  The worker CLI remains deterministic,
        # while an already-initialized process keeps its existing setting.
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        reports = build(args)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"A/B source sidecars failed: {error}")
        return 1
    print(json.dumps(reports, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
