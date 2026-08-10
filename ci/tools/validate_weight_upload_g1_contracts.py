#!/usr/bin/env python3
"""Validate the verified weight-upload G1 contracts without GPU or model use."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "ci/matrix/weight-upload-semantic-g1-v1.json"
MATRIX_SCHEMA = ROOT / "ci/schema/weight-upload-g1-matrix-v1.schema.json"
REPORT_SCHEMA = ROOT / "ci/schema/weight-upload-g1-report-v1.schema.json"
CORE = ROOT / "crates/sllm-core/src/weights.rs"
BINARY = ROOT / "crates/sllm-hip/src/bin/sllm-weight-upload-g1-evidence.rs"
CARGO = ROOT / "crates/sllm-hip/Cargo.toml"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def validate_report(report: dict[str, Any]) -> None:
    Draft202012Validator(read_json(REPORT_SCHEMA)).validate(report)
    source_start, source_end = report["source_range"]
    if source_end - source_start != report["tensor_size_bytes"]:
        raise ValueError("report source range does not match the tensor size")
    expected_sizes = (16 * 1024 * 1024, 4 * 1024 * 1024)
    cursor = 0
    for order, (chunk, expected_size) in enumerate(zip(report["chunks"], expected_sizes, strict=True)):
        if chunk["order"] != order or chunk["tensor_offset"] != cursor:
            raise ValueError("report chunks are not ordered and contiguous")
        if chunk["source_offset"] != source_start + cursor:
            raise ValueError("report chunk source offset is not plan-relative")
        if chunk["destination_offset"] != report["destination_offset"] + cursor:
            raise ValueError("report chunk destination offset is not target-relative")
        if chunk["size_bytes"] != expected_size:
            raise ValueError("report chunk size differs from the reviewed split")
        cursor += expected_size
    if cursor != report["tensor_size_bytes"]:
        raise ValueError("report chunks do not cover the tensor")


def validate() -> None:
    matrix_schema = read_json(MATRIX_SCHEMA)
    report_schema = read_json(REPORT_SCHEMA)
    Draft202012Validator.check_schema(matrix_schema)
    Draft202012Validator.check_schema(report_schema)
    Draft202012Validator(matrix_schema).validate(read_json(MATRIX))

    core = CORE.read_text(encoding="utf-8")
    for required in (
        "pub fn upload_verified_weight(",
        "plan.recompute_digest()",
        "source.read_tensor_range(tensor_name, source_relative, length)?",
        ".upload(request.queue, range, bytes)",
        "destination range must exactly match the tensor byte range",
        "weight chunks are not contiguous source/destination peers",
    ):
        if required not in core:
            raise ValueError(f"weight bridge is missing contract fragment: {required}")
    binary = BINARY.read_text(encoding="utf-8")
    for required in (
        "build_verified_weight_load_plan(&lock, &cache)",
        "upload_verified_weight(WeightUploadRequest",
        ".readback(&queue, range)",
        "output != expected",
        "plan.digest_hex() != PLAN_DIGEST",
        "entry.source_range != SOURCE_RANGE",
        "entry.chunks[0].byte_length != 16 * 1024 * 1024",
        "cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0",
    ):
        if required not in binary:
            raise ValueError(f"weight evidence binary is missing contract fragment: {required}")
    for forbidden in ("copy_to_host(", "copy_to_device(", "sllm_hip_sys", "std::net", "Queue::"):
        if forbidden in binary:
            raise ValueError(f"weight evidence binary bypasses the generic boundary: {forbidden}")

    cargo = CARGO.read_text(encoding="utf-8")
    if cargo.count('name = "sllm-weight-upload-g1-evidence"') != 1:
        raise ValueError("weight evidence binary target is not registered exactly once")
    if cargo.count('path = "src/bin/sllm-weight-upload-g1-evidence.rs"') != 1:
        raise ValueError("weight evidence binary path is not registered exactly once")


def main() -> int:
    try:
        validate()
    except (OSError, ValueError) as error:
        print(f"weight upload G1 contracts: FAIL: {error}", file=sys.stderr)
        return 1
    print("weight upload G1 contracts: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
