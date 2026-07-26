#!/usr/bin/env python3
"""Summarize a rocprofv3 kernel trace without treating it as throughput.

The output deliberately reports summed dispatch duration and launch-supply
geometry separately.  Neither quantity is an achieved-residency or wall-clock
measurement; full-model tok/s is recorded by the independent driver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


WAVE_SIZE = 32
COMPUTE_UNITS = 64
MAX_WAVES_PER_CU = 32
MACHINE_WAVE_SLOTS = COMPUTE_UNITS * MAX_WAVES_PER_CU
ATTENTION_MARKERS = (
    "cached_prefix_attn",
    "flash_attn",
    "flashattention",
    "fattn",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--api-trace", type=Path)
    parser.add_argument(
        "--terminal-sync-window",
        action="store_true",
        help="select kernels between the penultimate and final long HIP stream synchronization",
    )
    parser.add_argument("--min-sync-ns", type=int, default=1_000_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def integer(row: dict[str, str], key: str) -> int:
    value = row.get(key)
    if value is None or not value.strip():
        raise ValueError(f"missing {key} in rocprof row")
    return int(value)


def duration_ns(row: dict[str, str]) -> int:
    return integer(row, "End_Timestamp") - integer(row, "Start_Timestamp")


def workgroups(row: dict[str, str]) -> int:
    return math.prod(
        math.ceil(integer(row, f"Grid_Size_{axis}") / integer(row, f"Workgroup_Size_{axis}"))
        for axis in ("X", "Y", "Z")
    )


def threads_per_workgroup(row: dict[str, str]) -> int:
    return math.prod(integer(row, f"Workgroup_Size_{axis}") for axis in ("X", "Y", "Z"))


def waves(row: dict[str, str]) -> int:
    return workgroups(row) * math.ceil(threads_per_workgroup(row) / WAVE_SIZE)


def geometry(row: dict[str, str]) -> dict[str, int]:
    return {
        "grid_size_x": integer(row, "Grid_Size_X"),
        "grid_size_y": integer(row, "Grid_Size_Y"),
        "grid_size_z": integer(row, "Grid_Size_Z"),
        "workgroup_size_x": integer(row, "Workgroup_Size_X"),
        "workgroup_size_y": integer(row, "Workgroup_Size_Y"),
        "workgroup_size_z": integer(row, "Workgroup_Size_Z"),
        "workgroups_per_dispatch": workgroups(row),
        "threads_per_workgroup": threads_per_workgroup(row),
        "wave32_per_dispatch": waves(row),
        "machine_wave_slots": MACHINE_WAVE_SLOTS,
        "wave_supply_percent_of_machine_slots": waves(row) * 100.0 / MACHINE_WAVE_SLOTS,
    }


def is_attention_kernel(name: str) -> bool:
    lower = name.lower()
    return any(marker in lower for marker in ATTENTION_MARKERS)


def terminal_sync_window(
    api_rows: list[dict[str, str]], min_sync_ns: int
) -> tuple[int, int, dict[str, Any]]:
    syncs: list[tuple[int, int, str]] = []
    for row in api_rows:
        function = row.get("Function", "")
        if "hipStreamSynchronize" not in function and "hipDeviceSynchronize" not in function:
            continue
        start = integer(row, "Start_Timestamp")
        end = integer(row, "End_Timestamp")
        if end - start >= min_sync_ns:
            syncs.append((start, end, function))
    if len(syncs) < 2:
        raise ValueError(f"need at least two long syncs, found {len(syncs)}")
    start = syncs[-2][1]
    end = syncs[-1][1]
    return start, end, {
        "method": "terminal interval bounded by final two long HIP synchronization calls",
        "min_sync_ns": min_sync_ns,
        "long_sync_count": len(syncs),
        "start_after_sync": {"start_ns": syncs[-2][0], "end_ns": syncs[-2][1], "function": syncs[-2][2]},
        "end_after_sync": {"start_ns": syncs[-1][0], "end_ns": syncs[-1][1], "function": syncs[-1][2]},
    }


def compact_geometry(geometries: dict[tuple[int, ...], int]) -> list[dict[str, int]]:
    result = []
    for values, dispatches in sorted(geometries.items()):
        (
            grid_x,
            grid_y,
            grid_z,
            wg_x,
            wg_y,
            wg_z,
            workgroup_count,
            threads,
            wave_count,
        ) = values
        result.append(
            {
                "dispatches": dispatches,
                "grid_size_x": grid_x,
                "grid_size_y": grid_y,
                "grid_size_z": grid_z,
                "workgroup_size_x": wg_x,
                "workgroup_size_y": wg_y,
                "workgroup_size_z": wg_z,
                "workgroups_per_dispatch": workgroup_count,
                "threads_per_workgroup": threads,
                "wave32_per_dispatch": wave_count,
                "wave_supply_percent_of_machine_slots": wave_count * 100.0 / MACHINE_WAVE_SLOTS,
            }
        )
    return result


def main() -> int:
    args = parse_args()
    rows = read_csv(args.kernel_trace)
    if not rows:
        raise ValueError(f"kernel trace is empty: {args.kernel_trace}")
    selected = rows
    selection: dict[str, Any] = {"method": "all kernel rows in supplied trace"}
    input_metadata: dict[str, Any] = {
        "kernel_trace": str(args.kernel_trace),
        "kernel_trace_sha256": sha256(args.kernel_trace),
    }
    if args.api_trace is not None:
        input_metadata["api_trace"] = str(args.api_trace)
        input_metadata["api_trace_sha256"] = sha256(args.api_trace)
    if args.terminal_sync_window:
        if args.api_trace is None:
            raise ValueError("--terminal-sync-window requires --api-trace")
        begin, end, selection = terminal_sync_window(read_csv(args.api_trace), args.min_sync_ns)
        selected = [
            row
            for row in rows
            if integer(row, "Start_Timestamp") >= begin and integer(row, "End_Timestamp") <= end
        ]
        if not selected:
            raise ValueError("terminal synchronization window selected no kernels")
        selection["selected_kernel_begin_ns"] = begin
        selection["selected_kernel_end_ns"] = end

    duration_total = sum(duration_ns(row) for row in selected)
    if duration_total <= 0:
        raise ValueError("selected kernel duration is not positive")
    aggregate: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "dispatches": 0,
            "duration_ns_sum": 0,
            "workgroups_sum": 0,
            "wave32_sum": 0,
            "geometries": defaultdict(int),
        }
    )
    for row in selected:
        name = row.get("Kernel_Name", "")
        if not name:
            raise ValueError("kernel trace row has no Kernel_Name")
        record = aggregate[name]
        record["dispatches"] += 1
        record["duration_ns_sum"] += duration_ns(row)
        record["workgroups_sum"] += workgroups(row)
        record["wave32_sum"] += waves(row)
        item = geometry(row)
        key = (
            item["grid_size_x"],
            item["grid_size_y"],
            item["grid_size_z"],
            item["workgroup_size_x"],
            item["workgroup_size_y"],
            item["workgroup_size_z"],
            item["workgroups_per_dispatch"],
            item["threads_per_workgroup"],
            item["wave32_per_dispatch"],
        )
        record["geometries"][key] += 1

    kernels: list[dict[str, Any]] = []
    for name, values in aggregate.items():
        kernels.append(
            {
                "kernel_name": name,
                "attention_family": is_attention_kernel(name),
                "dispatches": values["dispatches"],
                "kernel_duration_ns_sum": values["duration_ns_sum"],
                "kernel_duration_ms_sum": values["duration_ns_sum"] / 1_000_000.0,
                "kernel_duration_share_percent": values["duration_ns_sum"] * 100.0 / duration_total,
                "workgroups_sum": values["workgroups_sum"],
                "wave32_sum": values["wave32_sum"],
                "launch_geometries": compact_geometry(values["geometries"]),
            }
        )
    kernels.sort(key=lambda item: item["kernel_duration_ns_sum"], reverse=True)
    attention = [item for item in kernels if item["attention_family"]]
    attention_duration = sum(item["kernel_duration_ns_sum"] for item in attention)
    output = {
        "schema_version": "ullm.prefill_attention_redesign.kernel_trace.v1",
        "label": args.label,
        "inputs": input_metadata,
        "selection": selection,
        "device_assumptions": {
            "gpu": "R9700 gfx1201",
            "compute_units": COMPUTE_UNITS,
            "wavefront_size": WAVE_SIZE,
            "max_waves_per_cu": MAX_WAVES_PER_CU,
            "machine_wave_slots": MACHINE_WAVE_SLOTS,
        },
        "aggregate": {
            "kernel_dispatches": len(selected),
            "kernel_duration_ns_sum": duration_total,
            "kernel_duration_ms_sum": duration_total / 1_000_000.0,
            "attention_dispatches": sum(item["dispatches"] for item in attention),
            "attention_kernel_duration_ns_sum": attention_duration,
            "attention_kernel_duration_share_percent": attention_duration * 100.0 / duration_total,
        },
        "kernels": kernels,
        "limitations": [
            "Summed kernel dispatch duration is not wall-clock elapsed time and is never used as throughput.",
            "Wave32 supply is a launch-envelope calculation, not measured residency or achieved occupancy.",
            "Name-based attention classification is trace attribution; algorithm details require source or ISA evidence.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(output, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
