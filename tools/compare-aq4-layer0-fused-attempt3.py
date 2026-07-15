#!/usr/bin/env python3
"""Create a descriptive, threshold-free AQ4 fused-output comparison artifact.

The comparator validates report identity/layout before reading outputs and streams
all f32 values in bounded chunks. It intentionally emits no correctness, holdout,
promotion, or policy verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pathlib
import struct
from typing import Any, Iterable


EXPECTED_LAYOUT = {
    "format": "concatenated_little_endian_f32_rows",
    "dtype": "f32",
    "row_order": "input_jsonl_order",
    "qkv_shape": [8192],
    "qkv_standalone_shape": [8192],
    "z_shape": [4096],
    "gate_shape": [32],
    "beta_shape": [32],
}
EXPECTED_SEGMENTS = [
    {"name": "Q", "start_row": 0, "end_row_exclusive": 2048},
    {"name": "K", "start_row": 2048, "end_row_exclusive": 4096},
    {"name": "V", "start_row": 4096, "end_row_exclusive": 8192},
]
OUTPUT_NAMES = {
    "qkv": "qkv.f32le",
    "qkv_standalone": "qkv-standalone.f32le",
    "z": "z.f32le",
    "gate": "gate.f32le",
    "beta": "beta.f32le",
}
OUTPUT_SHAPES = {"qkv": 8192, "qkv_standalone": 8192, "z": 4096, "gate": 32, "beta": 32}
CHUNK_BYTES = 64 * 1024


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def float_stream(path: pathlib.Path) -> Iterable[float]:
    carry = b""
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            data = carry + chunk
            usable = len(data) - (len(data) % 4)
            for (value,) in struct.iter_unpack("<f", data[:usable]):
                yield value
            carry = data[usable:]
    if carry:
        raise ValueError(f"unaligned f32 sidecar: {path}")


def byte_mismatch_counts(left: pathlib.Path, right: pathlib.Path, row_bytes: int, rows: int) -> list[int]:
    mismatches = [0] * rows
    offset = 0
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while True:
            left_chunk = left_stream.read(CHUNK_BYTES)
            right_chunk = right_stream.read(CHUNK_BYTES)
            if not left_chunk and not right_chunk:
                return mismatches
            common = min(len(left_chunk), len(right_chunk))
            for index, (left_byte, right_byte) in enumerate(zip(left_chunk[:common], right_chunk[:common])):
                if left_byte != right_byte:
                    row = (offset + index) // row_bytes
                    if row < rows:
                        mismatches[row] += 1
            for index in range(common, max(len(left_chunk), len(right_chunk))):
                row = (offset + index) // row_bytes
                if row < rows:
                    mismatches[row] += 1
            offset += max(len(left_chunk), len(right_chunk))


def input_identity(report: dict[str, Any]) -> dict[str, Any]:
    value = report["input"]
    identity = value["identity"]
    if identity["pre_stat"] != identity["post_stat"]:
        raise ValueError("input pre/post stat changed")
    return {
        "schema": value["schema"],
        "dtype": value["dtype"],
        "shape": value["shape"],
        "rows": value["rows"],
        "sidecar_sha256": value["sidecar_sha256"],
        "consumed_sha256": identity["consumed_sha256"],
    }


def resolve_output(report_path: pathlib.Path, output: dict[str, Any], expected_name: str) -> pathlib.Path:
    path = pathlib.Path(output["path"])
    if path.name != expected_name:
        raise ValueError(f"output basename differs: expected {expected_name}, got {path.name}")
    if not path.is_absolute():
        path = report_path.parent / path
    stat = path.stat()
    if not path.is_file() or path.is_symlink() or stat.st_nlink != 1:
        raise ValueError(f"output is not an immutable regular file: {path}")
    if output["sha256"] != sha256_file(path):
        raise ValueError(f"output SHA differs from report: {path}")
    return path


def load_report(path: pathlib.Path, label: str) -> tuple[dict[str, Any], dict[str, pathlib.Path], str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["schema_version"] != "ullm.aq4_layer0_qkv_z_gate_beta_runtime_probe.v2":
        raise ValueError(f"{label} report schema differs")
    if report["status"] != "valid" or report["classification"] != "unclassified" or report["fused"] is not True:
        raise ValueError(f"{label} report is not valid fused output")
    if report["input"]["rows"] != 3 or report["input"]["shape"] != [4096]:
        raise ValueError(f"{label} input rows/shape differs")
    if report["qkv_row_segments"] != EXPECTED_SEGMENTS or report["output_layout"] != EXPECTED_LAYOUT:
        raise ValueError(f"{label} output layout differs")
    identities = input_identity(report)
    if identities["schema"] != "ullm.aq4_layer0_input_normed_jsonl.v1" or identities["dtype"] != "f32":
        raise ValueError(f"{label} input identity schema differs")
    if identities["sidecar_sha256"] != identities["consumed_sha256"]:
        raise ValueError(f"{label} input consumed SHA differs")
    outputs: dict[str, pathlib.Path] = {}
    for key, name in OUTPUT_NAMES.items():
        output = report["outputs"][key]
        if output["row_shape"] != [OUTPUT_SHAPES[key]]:
            raise ValueError(f"{label} {key} row shape differs")
        if len(output["cases"]) != report["input"]["rows"]:
            raise ValueError(f"{label} {key} case count differs")
        path_value = resolve_output(path, output, name)
        expected_bytes = report["input"]["rows"] * OUTPUT_SHAPES[key] * 4
        if output["bytes"] != expected_bytes or path_value.stat().st_size != expected_bytes:
            raise ValueError(f"{label} {key} byte count differs")
        outputs[key] = path_value
    return report, outputs, sha256_file(path)


def metric_accumulator() -> dict[str, Any]:
    return {
        "element_count": 0,
        "nonfinite_left": 0,
        "nonfinite_right": 0,
        "max_abs": 0.0,
        "sum_sq_diff": 0.0,
        "sum_sq_left": 0.0,
        "sum_sq_right": 0.0,
        "dot": 0.0,
    }


def compare_streams(left: pathlib.Path, right: pathlib.Path, row_shape: int, rows: int) -> dict[str, Any]:
    aggregate = metric_accumulator()
    row_metrics = [metric_accumulator() for _ in range(rows)]
    left_stream = float_stream(left)
    right_stream = float_stream(right)
    for index, pair in enumerate(itertools.zip_longest(left_stream, right_stream, fillvalue=None)):
        left_value, right_value = pair
        if left_value is None or right_value is None:
            raise ValueError(f"element count differs: {left} vs {right}")
        row = index // row_shape
        if row >= rows:
            raise ValueError("more rows than report declares")
        item = row_metrics[row]
        aggregate["element_count"] += 1
        item["element_count"] += 1
        if not math.isfinite(left_value):
            aggregate["nonfinite_left"] += 1
            item["nonfinite_left"] += 1
        if not math.isfinite(right_value):
            aggregate["nonfinite_right"] += 1
            item["nonfinite_right"] += 1
        if math.isfinite(left_value) and math.isfinite(right_value):
            difference = left_value - right_value
            absolute = abs(difference)
            for target in (aggregate, item):
                target["max_abs"] = max(target["max_abs"], absolute)
                target["sum_sq_diff"] += difference * difference
                target["sum_sq_left"] += left_value * left_value
                target["sum_sq_right"] += right_value * right_value
                target["dot"] += left_value * right_value
    if aggregate["element_count"] != row_shape * rows:
        raise ValueError("element count does not match report layout")
    byte_mismatches = byte_mismatch_counts(left, right, row_shape * 4, rows)

    def finish(item: dict[str, Any], byte_mismatch: int, byte_count: int) -> dict[str, Any]:
        left_norm = math.sqrt(item["sum_sq_left"])
        right_norm = math.sqrt(item["sum_sq_right"])
        diff_norm = math.sqrt(item["sum_sq_diff"])
        valid = item["nonfinite_left"] == 0 and item["nonfinite_right"] == 0
        return {
            "element_count": item["element_count"],
            "byte_count_left": byte_count,
            "byte_count_right": byte_count,
            "byte_mismatch_count": byte_mismatch,
            "nonfinite_left": item["nonfinite_left"],
            "nonfinite_right": item["nonfinite_right"],
            "max_abs": item["max_abs"] if valid else None,
            "relative_l2": diff_norm / right_norm if valid and right_norm else None,
            "cosine": item["dot"] / (left_norm * right_norm) if valid and left_norm and right_norm else None,
        }

    return {
        "aggregate": finish(aggregate, sum(byte_mismatches), left.stat().st_size),
        "rows": [finish(item, byte_mismatches[index], row_shape * 4) for index, item in enumerate(row_metrics)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-report", type=pathlib.Path, required=True)
    parser.add_argument("--cpu-report", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    gpu_report, gpu_outputs, gpu_report_sha = load_report(args.gpu_report, "gpu")
    cpu_report, cpu_outputs, cpu_report_sha = load_report(args.cpu_report, "cpu")
    if gpu_report["device"]["backend"].lower() != "hip":
        raise SystemExit("GPU report backend is not HIP")
    if cpu_report["device"]["backend"].lower() != "cpu":
        raise SystemExit("CPU formal report backend is not CPU")
    gpu_identity = input_identity(gpu_report)
    cpu_identity = input_identity(cpu_report)
    if gpu_identity != cpu_identity:
        raise SystemExit(f"input identity differs: {gpu_identity} != {cpu_identity}")
    if gpu_report["package"]["manifest_sha256"] != cpu_report["package"]["manifest_sha256"]:
        raise SystemExit("package manifest identity differs")
    rows = gpu_identity["rows"]
    comparisons: dict[str, Any] = {}
    pairs = {
        "gpu_fused_qkv_vs_gpu_standalone_qkv": ("gpu", "qkv", "gpu", "qkv_standalone"),
        "gpu_fused_qkv_vs_cpu_formal_qkv": ("gpu", "qkv", "cpu", "qkv"),
        "gpu_standalone_qkv_vs_cpu_formal_qkv": ("gpu", "qkv_standalone", "cpu", "qkv_standalone"),
        "gpu_z_vs_cpu_formal_z": ("gpu", "z", "cpu", "z"),
        "gpu_gate_vs_cpu_formal_gate": ("gpu", "gate", "cpu", "gate"),
        "gpu_beta_vs_cpu_formal_beta": ("gpu", "beta", "cpu", "beta"),
    }
    for name, (left_label, left_key, right_label, right_key) in pairs.items():
        left_path = gpu_outputs[left_key] if left_label == "gpu" else cpu_outputs[left_key]
        right_path = gpu_outputs[right_key] if right_label == "gpu" else cpu_outputs[right_key]
        comparisons[name] = {
            "left": {"source": left_label, "key": left_key, "path": str(left_path), "sha256": sha256_file(left_path)},
            "right": {"source": right_label, "key": right_key, "path": str(right_path), "sha256": sha256_file(right_path)},
            "row_shape": OUTPUT_SHAPES[left_key],
            "rows": rows,
            "metrics": compare_streams(left_path, right_path, OUTPUT_SHAPES[left_key], rows),
        }
    result = {
        "schema_version": "ullm.aq4_layer0_qkv_fused_attempt3_comparison.v1",
        "status": "measured",
        "scope": "descriptive_numeric_comparison_only",
        "thresholds": None,
        "policy_decision": "not_evaluated",
        "holdout": "not_run",
        "promotion": "not_evaluated",
        "input_identity": gpu_identity,
        "output_layout": EXPECTED_LAYOUT,
        "qkv_row_segments": EXPECTED_SEGMENTS,
        "reports": {
            "gpu_attempt3": {"path": str(args.gpu_report), "sha256": gpu_report_sha},
            "cpu_formal": {"path": str(args.cpu_report), "sha256": cpu_report_sha},
        },
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "comparisons": list(comparisons)}))


if __name__ == "__main__":
    main()
