#!/usr/bin/env python3
"""Compare CPU and standalone-GPU artifact-FP32 capture runs.

This is deliberately a pre-measurement fixed policy for qualifying the GPU
reference, rather than the v0.2 candidate gate.  Every captured tensor is
compared: logits, final hidden, and all forty post-layer hidden states at each
position.  CPU reference inputs and GPU teacher-forced inputs, plus greedy
token IDs, are exact contracts; F32 payloads use numerical tolerances because
the CPU's fixed increasing-K FMA order and standard hipBLAS SGEMM need not be
bitwise identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any


SCHEMA = "ullm.sq8.artifact_fp32_cpu_gpu_reference_comparison.v1"
CHUNK_BYTES = 1024 * 1024
LAYERS = 40
HIDDEN = 5120
VOCAB = 151936

# These are set before any GPU result is measured.  They match the existing
# SQ8 F32LE differential control used for direct/handwritten GPU paths.  The
# CPU strict-F32-to-CPU-F64 real-artifact projection cross-check was already
# 2.742e-6 max-abs and 1.037e-6 relative-L2, so this admits normal F32
# operation-order variation while remaining substantially tighter than a
# quantization-quality threshold.
MAX_ABS = 2.0e-5
RELATIVE_L2 = 1.0e-5
MIN_COSINE = 0.999999


class ComparisonError(ValueError):
    """An invalid or incomplete capture, rather than a numerical mismatch."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ComparisonError(f"failed to read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ComparisonError(f"failed to parse {path}: {error}") from error
    if not isinstance(value, dict):
        raise ComparisonError(f"{path} is not a JSON object")
    return value


def required_tensor_paths() -> list[tuple[str, int]]:
    tensors = [("logits.f32le", VOCAB), ("final-hidden.f32le", HIDDEN)]
    tensors.extend((f"layers/layer-{index:02}-hidden.f32le", HIDDEN) for index in range(LAYERS))
    return tensors


def nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComparisonError(f"{label} must be a nonempty string")
    return value


def integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ComparisonError(f"{label} must be an integer")
    return value


def load_steps(run_root: Path) -> list[dict[str, Any]]:
    receipt = read_json(run_root / "run.json")
    steps = receipt.get("forward_steps")
    if not isinstance(steps, list) or not steps:
        raise ComparisonError(f"{run_root}/run.json has no nonempty forward_steps")
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(steps):
        if not isinstance(record, dict):
            raise ComparisonError(f"run forward_steps[{index}] is not an object")
        if integer(record.get("forward_index"), f"run forward_steps[{index}].forward_index") != index:
            raise ComparisonError(f"run forward_steps[{index}] is not in canonical index order")
        summary = record.get("summary")
        if not isinstance(summary, dict):
            raise ComparisonError(f"run forward_steps[{index}].summary is not an object")
        layer_hashes = summary.get("layer_hidden_f32le_sha256")
        if not isinstance(layer_hashes, list) or len(layer_hashes) != LAYERS:
            raise ComparisonError(f"run forward_steps[{index}] does not have {LAYERS} layer hashes")
        normalized.append(summary)
    return normalized


def verify_capture(run_root: Path, index: int, expected_summary: dict[str, Any]) -> dict[str, Path]:
    root = run_root / f"forward-{index:04}"
    metadata = read_json(root / "metadata.json")
    summary = metadata.get("forward")
    files = metadata.get("files")
    if not isinstance(summary, dict) or not isinstance(files, dict):
        raise ComparisonError(f"{root}/metadata.json is missing forward or files objects")
    for field in (
        "position",
        "input_token_id",
        "greedy_token_id",
        "logits_f32le_sha256",
        "final_hidden_f32le_sha256",
        "layer_hidden_f32le_sha256",
    ):
        if summary.get(field) != expected_summary.get(field):
            raise ComparisonError(f"{root}/metadata.json forward.{field} disagrees with run.json")
    expected_paths = required_tensor_paths()
    if set(files) != {path for path, _ in expected_paths}:
        raise ComparisonError(f"{root}/metadata.json file set is not the complete canonical capture")
    paths: dict[str, Path] = {}
    for relative, elements in expected_paths:
        expected_hash = nonempty_string(files.get(relative), f"{root} files[{relative!r}]")
        path = root / relative
        try:
            size = path.stat().st_size
        except OSError as error:
            raise ComparisonError(f"failed to stat capture {path}: {error}") from error
        if size != elements * 4:
            raise ComparisonError(
                f"capture size mismatch for {path}: expected={elements * 4} actual={size}"
            )
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ComparisonError(
                f"capture SHA-256 mismatch for {path}: expected={expected_hash} actual={actual_hash}"
            )
        paths[relative] = path
    if files["logits.f32le"] != summary["logits_f32le_sha256"]:
        raise ComparisonError(f"{root} logits hash disagrees with its summary")
    if files["final-hidden.f32le"] != summary["final_hidden_f32le_sha256"]:
        raise ComparisonError(f"{root} final hidden hash disagrees with its summary")
    layer_hashes = summary["layer_hidden_f32le_sha256"]
    for layer in range(LAYERS):
        relative = f"layers/layer-{layer:02}-hidden.f32le"
        if files[relative] != layer_hashes[layer]:
            raise ComparisonError(f"{root} layer {layer} hash disagrees with its summary")
    return paths


def compare_tensor(reference: Path, candidate: Path) -> dict[str, object]:
    if reference.stat().st_size != candidate.stat().st_size:
        raise ComparisonError(
            f"length mismatch: reference={reference.stat().st_size} candidate={candidate.stat().st_size}"
        )
    count = 0
    bit_mismatch_count = 0
    nonfinite_count = 0
    max_abs = 0.0
    max_abs_index = 0
    squared_error = 0.0
    reference_squared = 0.0
    candidate_squared = 0.0
    dot = 0.0
    with reference.open("rb") as left, candidate.open("rb") as right:
        while True:
            expected_bytes = left.read(CHUNK_BYTES)
            actual_bytes = right.read(CHUNK_BYTES)
            if not expected_bytes and not actual_bytes:
                break
            if len(expected_bytes) != len(actual_bytes) or len(expected_bytes) % 4 != 0:
                raise ComparisonError("inconsistent f32le reads")
            for local_index, (expected_bits, actual_bits, expected, actual) in enumerate(
                zip(
                    struct.iter_unpack("<I", expected_bytes),
                    struct.iter_unpack("<I", actual_bytes),
                    struct.iter_unpack("<f", expected_bytes),
                    struct.iter_unpack("<f", actual_bytes),
                )
            ):
                expected_value = expected[0]
                actual_value = actual[0]
                if expected_bits[0] != actual_bits[0]:
                    bit_mismatch_count += 1
                if not math.isfinite(expected_value) or not math.isfinite(actual_value):
                    nonfinite_count += 1
                    continue
                delta = actual_value - expected_value
                absolute = abs(delta)
                if absolute > max_abs:
                    max_abs = absolute
                    max_abs_index = count + local_index
                squared_error += delta * delta
                reference_squared += expected_value * expected_value
                candidate_squared += actual_value * actual_value
                dot += expected_value * actual_value
            count += len(expected_bytes) // 4
    relative_l2 = math.sqrt(squared_error) / max(math.sqrt(reference_squared), 1.0e-30)
    cosine_denominator = math.sqrt(reference_squared) * math.sqrt(candidate_squared)
    cosine = dot / cosine_denominator if cosine_denominator else 1.0
    passed = (
        nonfinite_count == 0
        and math.isfinite(max_abs)
        and math.isfinite(relative_l2)
        and math.isfinite(cosine)
        and max_abs <= MAX_ABS
        and relative_l2 <= RELATIVE_L2
        and cosine >= MIN_COSINE
    )
    return {
        "element_count": count,
        "reference_sha256": sha256(reference),
        "candidate_sha256": sha256(candidate),
        "bit_mismatch_count": bit_mismatch_count,
        "nonfinite_count": nonfinite_count,
        "max_abs": max_abs,
        "max_abs_index": max_abs_index,
        "relative_l2": relative_l2,
        "cosine_similarity": cosine,
        "passed": passed,
    }


def compare_runs(cpu_root: Path, gpu_root: Path) -> dict[str, object]:
    cpu_steps = load_steps(cpu_root)
    gpu_steps = load_steps(gpu_root)
    if len(cpu_steps) != len(gpu_steps):
        raise ComparisonError(
            f"position count differs: CPU={len(cpu_steps)} GPU={len(gpu_steps)}"
        )

    positions: list[dict[str, object]] = []
    total_bit_mismatches = 0
    total_nonfinite = 0
    worst_max_abs: tuple[float, str, int] = (-1.0, "", 0)
    worst_relative_l2: tuple[float, str, int] = (-1.0, "", 0)
    minimum_cosine: tuple[float, str, int] = (math.inf, "", 0)
    passed = True
    for index, (cpu_step, gpu_step) in enumerate(zip(cpu_steps, gpu_steps)):
        cpu_input = integer(cpu_step.get("input_token_id"), f"CPU position {index} input_token_id")
        gpu_input = integer(gpu_step.get("input_token_id"), f"GPU position {index} input_token_id")
        cpu_greedy = integer(cpu_step.get("greedy_token_id"), f"CPU position {index} greedy_token_id")
        gpu_greedy = integer(gpu_step.get("greedy_token_id"), f"GPU position {index} greedy_token_id")
        cpu_paths = verify_capture(cpu_root, index, cpu_step)
        gpu_paths = verify_capture(gpu_root, index, gpu_step)
        tensors: dict[str, object] = {}
        position_passed = cpu_input == gpu_input and cpu_greedy == gpu_greedy
        for relative, _ in required_tensor_paths():
            metrics = compare_tensor(cpu_paths[relative], gpu_paths[relative])
            tensors[relative] = metrics
            metrics_passed = bool(metrics["passed"])
            position_passed = position_passed and metrics_passed
            total_bit_mismatches += int(metrics["bit_mismatch_count"])
            total_nonfinite += int(metrics["nonfinite_count"])
            if float(metrics["max_abs"]) > worst_max_abs[0]:
                worst_max_abs = (float(metrics["max_abs"]), relative, index)
            if float(metrics["relative_l2"]) > worst_relative_l2[0]:
                worst_relative_l2 = (float(metrics["relative_l2"]), relative, index)
            if float(metrics["cosine_similarity"]) < minimum_cosine[0]:
                minimum_cosine = (float(metrics["cosine_similarity"]), relative, index)
        positions.append(
            {
                "position": index,
                "input_token_id": {"cpu": cpu_input, "gpu": gpu_input, "equal": cpu_input == gpu_input},
                "greedy_token_id": {"cpu": cpu_greedy, "gpu": gpu_greedy, "equal": cpu_greedy == gpu_greedy},
                "tensors": tensors,
                "passed": position_passed,
            }
        )
        passed = passed and position_passed
    return {
        "schema_version": SCHEMA,
        "cpu_run": str(cpu_root),
        "gpu_run": str(gpu_root),
        "positions": positions,
        "thresholds_predeclared_before_gpu_measurement": {
            "max_abs": MAX_ABS,
            "relative_l2": RELATIVE_L2,
            "min_cosine_similarity": MIN_COSINE,
            "nonfinite_count": 0,
            "input_token_ids": "exact",
            "greedy_token_ids": "exact",
            "scope": "every logits/final-hidden/layer-00..39 tensor at every position",
            "rationale": (
                "existing SQ8 F32LE GPU differential control; fixed before GPU measurement; "
                "allows only expected F32 operation-order variation"
            ),
        },
        "aggregate": {
            "positions": len(positions),
            "tensors_compared": len(positions) * len(required_tensor_paths()),
            "total_bit_mismatch_count": total_bit_mismatches,
            "total_nonfinite_count": total_nonfinite,
            "worst_max_abs": {
                "value": worst_max_abs[0],
                "tensor": worst_max_abs[1],
                "position": worst_max_abs[2],
            },
            "worst_relative_l2": {
                "value": worst_relative_l2[0],
                "tensor": worst_relative_l2[1],
                "position": worst_relative_l2[2],
            },
            "minimum_cosine_similarity": {
                "value": minimum_cosine[0],
                "tensor": minimum_cosine[1],
                "position": minimum_cosine[2],
            },
        },
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-run", required=True, type=Path)
    parser.add_argument("--gpu-run", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    args = parser.parse_args()
    if args.result_json.exists():
        raise SystemExit(f"refusing to overwrite existing result: {args.result_json}")
    try:
        result = compare_runs(args.cpu_run, args.gpu_run)
    except ComparisonError as error:
        raise SystemExit(f"invalid reference capture: {error}") from error
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    try:
        with args.result_json.open("x", encoding="utf-8") as destination:
            destination.write(payload)
    except OSError as error:
        raise SystemExit(f"failed to create {args.result_json}: {error}") from error
    print(payload, end="")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
