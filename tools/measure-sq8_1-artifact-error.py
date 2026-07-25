#!/usr/bin/env python3
"""Measure a packed SQ8_1 tensor against its reconstructed SQ8_0 source.

This is intentionally a tensor-level measurement.  It does not claim a
full-model logit result, and it leaves both the verified SQ8_0 source and the
SQ8_1 artifact immutable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from sq8_1_artifact import (
    ArtifactError,
    _source_tensor_rows,
    f32,
    matvec_w8a16,
    matvec_w8a8_explicit,
    read_json,
    read_sq8_1_tensor,
    verify_sq8_1_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sq8_0-artifact", required=True, type=Path)
    parser.add_argument("--sq8_1-artifact", required=True, type=Path)
    parser.add_argument("--tensor-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def relative_l2(error_sumsq: float, reference_sumsq: float) -> float:
    if reference_sumsq == 0.0:
        return 0.0 if error_sumsq == 0.0 else math.inf
    return math.sqrt(error_sumsq / reference_sumsq)


def metrics(actual: list[float], expected: list[float]) -> dict[str, float]:
    if len(actual) != len(expected):
        raise ArtifactError("metric inputs have different lengths")
    error_sumsq = 0.0
    reference_sumsq = 0.0
    maximum = 0.0
    for observed, reference in zip(actual, expected, strict=True):
        error = observed - reference
        error_sumsq += error * error
        reference_sumsq += reference * reference
        maximum = max(maximum, abs(error))
    return {
        "relative_l2": relative_l2(error_sumsq, reference_sumsq),
        "max_abs": maximum,
    }


def main() -> int:
    args = parse_args()
    output = args.output.resolve(strict=False)
    if output.exists():
        raise SystemExit(f"refusing to overwrite output: {output}")
    artifact = args.sq8_1_artifact.resolve()
    source = args.source_sq8_0_artifact.resolve()
    verification = verify_sq8_1_artifact(artifact)
    manifest = read_json(artifact / "sq8_1_manifest.json")
    source_manifest = read_json(source / "sq_manifest.json")
    source_entries = {
        entry["name"]: entry
        for entry in source_manifest.get("quantized_tensors", [])
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    if args.tensor_name not in source_entries:
        raise SystemExit(f"source SQ8_0 tensor is absent: {args.tensor_name}")
    tensor = read_sq8_1_tensor(artifact, args.tensor_name)
    entry = source_entries[args.tensor_name]
    source_rows = list(_source_tensor_rows(source, entry))
    if len(source_rows) != tensor.rows or any(len(row) != tensor.cols for row in source_rows):
        raise SystemExit("source and SQ8_1 tensor shapes differ")

    weight_error_sumsq = 0.0
    weight_reference_sumsq = 0.0
    weight_max_abs = 0.0
    reconstructed_rows: list[list[float]] = []
    for row_index, source_row in enumerate(source_rows):
        reconstructed = tensor.reconstruct_row(row_index)
        reconstructed_rows.append(reconstructed)
        for actual, expected in zip(reconstructed, source_row, strict=True):
            error = actual - expected
            weight_error_sumsq += error * error
            weight_reference_sumsq += expected * expected
            weight_max_abs = max(weight_max_abs, abs(error))

    # The activation is deterministic and declared here solely to exercise the
    # two reference paths on this real tensor.  It is not a model-logit test.
    activation = [f32(((index * 19 + 7) % 253 - 126) / 127.0) for index in range(tensor.cols)]
    source_output = [sum(weight * activation[col] for col, weight in enumerate(row)) for row in source_rows]
    output_w8a16 = matvec_w8a16(tensor, activation)
    output_w8a8 = matvec_w8a8_explicit(tensor, activation)
    manifest_entry = next(entry for entry in manifest["tensors"] if entry["name"] == args.tensor_name)
    result = {
        "measurement": "single_tensor_against_reconstructed_verified_sq8_0_source",
        "scope_note": "not a full-model logit or release-quality gate",
        "source_artifact": str(source),
        "sq8_1_artifact": str(artifact),
        "tensor": args.tensor_name,
        "shape": [tensor.rows, tensor.cols],
        "values": tensor.rows * tensor.cols,
        "weight_error": {
            "relative_l2": relative_l2(weight_error_sumsq, weight_reference_sumsq),
            "max_abs": weight_max_abs,
        },
        "sampled_linear_output_error": {
            "activation": "deterministic_f32_(i*19+7)%253-126_over_127",
            "w8a16_default": metrics(output_w8a16, source_output),
            "w8a8_explicit": metrics(output_w8a8, source_output),
        },
        "quantization": manifest_entry["quantization"],
        "storage": manifest_entry["storage"],
        "artifact_verification": verification,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ArtifactError as exc:
        raise SystemExit(str(exc)) from exc
