#!/usr/bin/env python3
"""Validate the focused backend-neutral transfer G1 contracts without GPU use."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "ci/matrix/execution-transfer-g1-v1.json"
MATRIX_SCHEMA = ROOT / "ci/schema/execution-transfer-g1-matrix-v1.schema.json"
REPORT_SCHEMA = ROOT / "ci/schema/execution-transfer-g1-report-v1.schema.json"
BINARY = ROOT / "crates/sllm-hip/src/bin/sllm-execution-transfer-g1-evidence.rs"
CARGO = ROOT / "crates/sllm-hip/Cargo.toml"


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate() -> None:
    matrix_schema = read_json(MATRIX_SCHEMA)
    report_schema = read_json(REPORT_SCHEMA)
    Draft202012Validator.check_schema(matrix_schema)
    Draft202012Validator.check_schema(report_schema)
    Draft202012Validator(matrix_schema).validate(read_json(MATRIX))

    source = BINARY.read_text(encoding="utf-8")
    for required in (
        "const CASE_SIZES: [usize; 6] = [1, 3, 17, 255, 256, 257]",
        ".open_execution_session(request)",
        ".max_transfer_bytes()",
        ".upload(&queue, range.clone()",
        ".readback(&queue, range)",
        "output != input",
        "kernel_dispatch_count: 0",
        "cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0",
    ):
        if required not in source:
            raise ValueError(f"transfer evidence binary is missing contract fragment: {required}")
    for forbidden in ("copy_to_host(", "copy_to_device(", "sllm_hip_sys", "std::net"):
        if forbidden in source:
            raise ValueError(f"transfer evidence binary bypasses the generic boundary: {forbidden}")

    cargo = CARGO.read_text(encoding="utf-8")
    if cargo.count('name = "sllm-execution-transfer-g1-evidence"') != 1:
        raise ValueError("transfer evidence binary target is not registered exactly once")
    if cargo.count('path = "src/bin/sllm-execution-transfer-g1-evidence.rs"') != 1:
        raise ValueError("transfer evidence binary path is not registered exactly once")


def main() -> int:
    try:
        validate()
    except (OSError, ValueError) as error:
        print(f"execution transfer G1 contracts: FAIL: {error}", file=sys.stderr)
        return 1
    print("execution transfer G1 contracts: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
