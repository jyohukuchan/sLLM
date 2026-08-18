#!/usr/bin/env python3
"""Extract Phase 29 GDN device time from a rocprofv3 kernel trace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


class Phase29Error(RuntimeError):
    """Raised when an input cannot provide fail-closed Phase 29 evidence."""


GDN_KERNEL_MARKERS = (
    "linear_attention_recurrent_gated_norm",
    "linear_attention_gdn_prepare",
    "linear_attention_gdn_core",
    "linear_attention_gdn_reduce",
    "linear_attention_gdn_finalize",
    "linear_attention_gdn_copy",
)


def _duration_ns(row: dict[str, str]) -> int:
    duration = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
    if duration <= 0:
        raise Phase29Error("kernel duration must be positive")
    return duration


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        raise Phase29Error("cannot aggregate an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def is_gdn_kernel(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in GDN_KERNEL_MARKERS)


def extract_decode_steps(
    rows: Iterable[dict[str, str]], *, requests: int = 14, output_tokens: int = 16
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row["Dispatch_Id"]))
    argmax_positions = [
        index
        for index, row in enumerate(ordered)
        if "argmax" in row["Kernel_Name"].lower()
    ]
    expected_argmax = requests * output_tokens
    if len(argmax_positions) != expected_argmax:
        raise Phase29Error(
            f"expected {expected_argmax} Argmax dispatches, got {len(argmax_positions)}"
        )

    steps: list[dict[str, Any]] = []
    for token_index, position in enumerate(argmax_positions):
        if token_index % output_tokens == 0:
            continue
        previous_position = argmax_positions[token_index - 1]
        family = [
            row
            for row in ordered[previous_position + 1 : position + 1]
            if is_gdn_kernel(row["Kernel_Name"])
        ]
        if not family:
            raise Phase29Error(f"decode step {token_index} has no GDN device work")
        steps.append(
            {
                "device_ns": sum(_duration_ns(row) for row in family),
                "calls": len(family),
                "kernels": sorted({row["Kernel_Name"] for row in family}),
            }
        )

    expected_steps = requests * (output_tokens - 1)
    if len(steps) != expected_steps:
        raise Phase29Error(f"expected {expected_steps} decode steps, got {len(steps)}")
    return steps


def _token_digest(raw: dict[str, Any]) -> str:
    samples = raw.get("measured", {}).get("samples", [])
    if len(samples) != 10:
        raise Phase29Error("benchmark must contain exactly 10 measured samples")
    token_records = [sample.get("tokens") for sample in samples]
    if any(record != token_records[0] for record in token_records[1:]):
        raise Phase29Error("measured samples did not produce identical token records")
    payload = json.dumps(token_records[0], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_benchmark(raw: dict[str, Any], target: str, prompt_tokens: int) -> None:
    config = raw.get("config", {})
    audit = raw.get("audit", {})
    cleanup = raw.get("cleanup", {})
    if raw.get("state") != "PASS":
        raise Phase29Error("benchmark state is not PASS")
    if config.get("input_token_count") != prompt_tokens:
        raise Phase29Error("prompt token count drifted")
    if config.get("max_new_tokens") != 16 or config.get("warmups") != 3 or config.get("measured") != 10:
        raise Phase29Error("benchmark protocol drifted")
    if audit.get("target") != target or audit.get("selected_backend") != "hip":
        raise Phase29Error("benchmark target/backend drifted")
    if not audit.get("all_dispatches_hip") or audit.get("fallback_used"):
        raise Phase29Error("GPU evidence used fallback or a non-HIP dispatch")
    if cleanup.get("retryable_cleanup") != 0 or cleanup.get("durable_quarantine") != 0:
        raise Phase29Error("cleanup residue is nonzero")
    if not cleanup.get("all_requests_dropped"):
        raise Phase29Error("request cleanup is incomplete")


def build_run_report(
    trace_path: Path,
    benchmark_path: Path,
    *,
    target: str,
    pattern: str,
    variant: str,
    process_index: int,
) -> dict[str, Any]:
    prompt_tokens = {"B0": 17, "B1": 28, "B2": 255}.get(pattern)
    if prompt_tokens is None:
        raise Phase29Error(f"unsupported pattern: {pattern}")
    if target not in {"gfx1030", "gfx1201"}:
        raise Phase29Error(f"unsupported target: {target}")
    if variant not in {"baseline", "candidate"}:
        raise Phase29Error(f"unsupported variant: {variant}")
    if process_index < 1:
        raise Phase29Error("process index must be positive")

    with trace_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    raw = json.loads(benchmark_path.read_text())
    validate_benchmark(raw, target, prompt_tokens)
    steps = extract_decode_steps(rows)
    device_ns = [step["device_ns"] for step in steps]
    calls = [step["calls"] for step in steps]
    if len(set(calls)) != 1:
        raise Phase29Error("GDN calls per committed step are unstable")

    gdn_rows = [row for row in rows if is_gdn_kernel(row["Kernel_Name"])]
    resources = sorted(
        {
            (
                row["Kernel_Name"],
                int(row["Workgroup_Size_X"]),
                int(row["Grid_Size_X"]),
                int(row["LDS_Block_Size"]),
                int(row["Scratch_Size"]),
                int(row["VGPR_Count"]),
                int(row["SGPR_Count"]),
            )
            for row in gdn_rows
        }
    )
    return {
        "schema_version": "phase29-gdn-device-run-v1",
        "state": "PASS",
        "target": target,
        "pattern": pattern,
        "variant": variant,
        "process_index": process_index,
        "prompt_tokens": prompt_tokens,
        "output_tokens": 16,
        "committed_decode_steps": len(steps),
        "gdn_device_p50_ns": statistics.median(device_ns),
        "gdn_device_p90_ns": _percentile(device_ns, 0.9),
        "gdn_device_max_ns": max(device_ns),
        "gdn_calls_per_step": calls[0],
        "token_record_sha256": _token_digest(raw),
        "resources": [
            {
                "kernel": item[0],
                "workgroup_x": item[1],
                "grid_x": item[2],
                "lds_bytes": item[3],
                "scratch_bytes": item[4],
                "vgpr_count": item[5],
                "sgpr_count": item[6],
            }
            for item in resources
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--target", choices=("gfx1030", "gfx1201"), required=True)
    parser.add_argument("--pattern", choices=("B0", "B1", "B2"), required=True)
    parser.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--process-index", type=int, required=True)
    args = parser.parse_args()
    report = build_run_report(
        args.trace,
        args.benchmark,
        target=args.target,
        pattern=args.pattern,
        variant=args.variant,
        process_index=args.process_index,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
