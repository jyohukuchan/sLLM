#!/usr/bin/env python3
"""Extract the terminal llama-bench decode region from rocprofv3 CSV traces.

The profiled command used one depth prefill followed by 16 single-token
generations.  llama-bench synchronizes after the depth prefill and after every
generated token.  rocprof records many internal synchronizes too, so this
script uses the 25 long (>1 ms) hipStreamSynchronize calls observed in this
capture: ordinal 9 is the depth completion; ordinals 10--25 are the sixteen
generation completions (one-based ordinals).

The selection is deliberately validated against the expected 40 vector FATTN
main and 40 FATTN-combine dispatches in every generated-token interval.  It is
an analysis helper only: it does not invoke a GPU or alter runtime sources.
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


MAIN_MARKER = "flash_attn_ext_vec"
COMBINE_MARKER = "flash_attn_combine_results"
SYNC_FUNCTION = "hipStreamSynchronize"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def integer(row: dict[str, str], key: str) -> int:
    return int(row[key])


def workgroups(row: dict[str, str]) -> int:
    axes = ("X", "Y", "Z")
    return math.prod(
        math.ceil(integer(row, f"Grid_Size_{axis}") / integer(row, f"Workgroup_Size_{axis}"))
        for axis in axes
    )


def wavefronts(row: dict[str, str]) -> int:
    threads = math.prod(integer(row, f"Workgroup_Size_{axis}") for axis in ("X", "Y", "Z"))
    return workgroups(row) * math.ceil(threads / 32)


def family(row: dict[str, str]) -> str | None:
    name = row["Kernel_Name"]
    if MAIN_MARKER in name:
        return "fattn_main"
    if COMBINE_MARKER in name:
        return "fattn_combine"
    return None


def duration_ns(row: dict[str, str]) -> int:
    return integer(row, "End_Timestamp") - integer(row, "Start_Timestamp")


def as_ms(value: int | float) -> float:
    return value / 1_000_000.0


def average(values: list[int | float]) -> float:
    return sum(values) / len(values)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--api-trace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    kernel_rows = read_csv(args.trace)
    for row in kernel_rows:
        row["_start"] = str(integer(row, "Start_Timestamp"))
        row["_end"] = str(integer(row, "End_Timestamp"))

    long_syncs: list[dict[str, int | str]] = []
    for row in read_csv(args.api_trace):
        if row["Function"] != SYNC_FUNCTION:
            continue
        start = integer(row, "Start_Timestamp")
        end = integer(row, "End_Timestamp")
        if end - start >= 1_000_000:
            long_syncs.append(
                {
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": end - start,
                    "correlation_id": row["Correlation_Id"],
                }
            )

    # Capture-specific, source-backed fence indexing described in the docstring.
    depth_completion_index = 8
    first_generation_completion_index = 9
    decode_steps = 16
    if len(long_syncs) != 25:
        raise SystemExit(f"expected 25 long synchronization calls, found {len(long_syncs)}")
    if first_generation_completion_index + decode_steps != len(long_syncs):
        raise SystemExit("generation-fence indexing no longer reaches the trace end")

    step_rows: list[dict[str, Any]] = []
    selected: list[dict[str, str]] = []
    previous_end = int(long_syncs[depth_completion_index]["end_ns"])
    for ordinal in range(decode_steps):
        fence = long_syncs[first_generation_completion_index + ordinal]
        end = int(fence["end_ns"])
        step = [
            row
            for row in kernel_rows
            if integer(row, "Start_Timestamp") >= previous_end
            and integer(row, "End_Timestamp") <= end
        ]
        main_rows = [row for row in step if family(row) == "fattn_main"]
        combine_rows = [row for row in step if family(row) == "fattn_combine"]
        if len(main_rows) != 40 or len(combine_rows) != 40:
            raise SystemExit(
                f"step {ordinal + 1}: expected 40 main + 40 combine FATTN dispatches, "
                f"found {len(main_rows)} + {len(combine_rows)}"
            )
        attention_rows = main_rows + combine_rows
        total_duration = sum(duration_ns(row) for row in step)
        attention_duration = sum(duration_ns(row) for row in attention_rows)
        main_duration = sum(duration_ns(row) for row in main_rows)
        combine_duration = sum(duration_ns(row) for row in combine_rows)
        step_rows.append(
            {
                "step": ordinal + 1,
                "begin_ns": previous_end,
                "end_ns": end,
                "wall_interval_ms": as_ms(end - previous_end),
                "completion_sync_correlation_id": fence["correlation_id"],
                "kernel_dispatches": len(step),
                "kernel_duration_ms_sum": as_ms(total_duration),
                "attention_dispatches": len(attention_rows),
                "fattn_main_dispatches": len(main_rows),
                "fattn_combine_dispatches": len(combine_rows),
                "attention_duration_ms_sum": as_ms(attention_duration),
                "fattn_main_duration_ms_sum": as_ms(main_duration),
                "fattn_combine_duration_ms_sum": as_ms(combine_duration),
                "attention_kernel_time_share_percent": attention_duration * 100.0 / total_duration,
                "all_kernel_workgroups": sum(workgroups(row) for row in step),
                "attention_workgroups": sum(workgroups(row) for row in attention_rows),
                "fattn_main_workgroups": sum(workgroups(row) for row in main_rows),
                "fattn_combine_workgroups": sum(workgroups(row) for row in combine_rows),
            }
        )
        selected.extend(step)
        previous_end = end

    main_selected = [row for row in selected if family(row) == "fattn_main"]
    combine_selected = [row for row in selected if family(row) == "fattn_combine"]
    attention_selected = main_selected + combine_selected
    total_duration = sum(duration_ns(row) for row in selected)
    attention_duration = sum(duration_ns(row) for row in attention_selected)

    families = []
    for label, rows in (("fattn_main", main_selected), ("fattn_combine", combine_selected)):
        geometries = {
            (
                row["Workgroup_Size_X"],
                row["Workgroup_Size_Y"],
                row["Workgroup_Size_Z"],
                row["Grid_Size_X"],
                row["Grid_Size_Y"],
                row["Grid_Size_Z"],
            )
            for row in rows
        }
        if len(geometries) != 1:
            raise SystemExit(f"{label}: expected one launch geometry, found {len(geometries)}")
        geometry = next(iter(geometries))
        families.append(
            {
                "family": label,
                "dispatches": len(rows),
                "dispatches_per_decode_step": len(rows) / decode_steps,
                "kernel_duration_ms_sum": as_ms(sum(duration_ns(row) for row in rows)),
                "kernel_duration_ms_per_decode_step": as_ms(sum(duration_ns(row) for row in rows) / decode_steps),
                "workgroups_per_dispatch": workgroups(rows[0]),
                "workgroups_per_decode_step": sum(workgroups(row) for row in rows) / decode_steps,
                "wavefronts_per_dispatch_wave32": wavefronts(rows[0]),
                "wavefronts_per_decode_step_wave32": sum(wavefronts(row) for row in rows) / decode_steps,
                "workgroup_size_x": geometry[0],
                "workgroup_size_y": geometry[1],
                "workgroup_size_z": geometry[2],
                "grid_size_x": geometry[3],
                "grid_size_y": geometry[4],
                "grid_size_z": geometry[5],
            }
        )

    by_kernel: dict[str, dict[str, int]] = defaultdict(lambda: {"dispatches": 0, "duration_ns_sum": 0, "workgroups": 0})
    for row in selected:
        record = by_kernel[row["Kernel_Name"]]
        record["dispatches"] += 1
        record["duration_ns_sum"] += duration_ns(row)
        record["workgroups"] += workgroups(row)
    kernel_summary = [
        {
            "kernel_name": name,
            "dispatches": values["dispatches"],
            "dispatches_per_decode_step": values["dispatches"] / decode_steps,
            "kernel_duration_ms_sum": as_ms(values["duration_ns_sum"]),
            "kernel_duration_ms_per_decode_step": as_ms(values["duration_ns_sum"] / decode_steps),
            "workgroups_sum": values["workgroups"],
            "workgroups_per_decode_step": values["workgroups"] / decode_steps,
        }
        for name, values in by_kernel.items()
    ]
    kernel_summary.sort(key=lambda row: row["kernel_duration_ms_sum"], reverse=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / "decode-step-summary.csv",
        step_rows,
        list(step_rows[0]),
    )
    write_csv(
        args.out_dir / "attention-kernel-summary.csv",
        families,
        list(families[0]),
    )
    write_csv(
        args.out_dir / "decode-kernel-summary.csv",
        kernel_summary,
        list(kernel_summary[0]),
    )

    summary = {
        "schema": "ullm.llamacpp_attention_profile_summary.v1",
        "inputs": {
            "kernel_trace": str(args.trace),
            "kernel_trace_sha256": sha256(args.trace),
            "hip_api_trace": str(args.api_trace),
            "hip_api_trace_sha256": sha256(args.api_trace),
        },
        "selection": {
            "method": "terminal long hipStreamSynchronize fences; source-confirmed llama-bench depth then 16 generation order",
            "long_sync_min_duration_ns": 1_000_000,
            "long_sync_count": len(long_syncs),
            "depth_completion_long_sync_ordinal_1_based": depth_completion_index + 1,
            "generation_completion_long_sync_ordinals_1_based": list(
                range(first_generation_completion_index + 1, first_generation_completion_index + decode_steps + 1)
            ),
            "decode_steps": decode_steps,
            "per_step_fattn_validation": "40 main vector FATTN + 40 combine FATTN dispatches",
        },
        "aggregate": {
            "kernel_dispatches": len(selected),
            "kernel_dispatches_per_decode_step": len(selected) / decode_steps,
            "kernel_duration_ms_sum": as_ms(total_duration),
            "kernel_duration_ms_per_decode_step": as_ms(total_duration / decode_steps),
            "attention_dispatches": len(attention_selected),
            "attention_dispatches_per_decode_step": len(attention_selected) / decode_steps,
            "attention_duration_ms_sum": as_ms(attention_duration),
            "attention_duration_ms_per_decode_step": as_ms(attention_duration / decode_steps),
            "attention_kernel_time_share_percent": attention_duration * 100.0 / total_duration,
            "all_kernel_workgroups": sum(workgroups(row) for row in selected),
            "all_kernel_workgroups_per_decode_step": sum(workgroups(row) for row in selected) / decode_steps,
            "attention_workgroups": sum(workgroups(row) for row in attention_selected),
            "attention_workgroups_per_decode_step": sum(workgroups(row) for row in attention_selected) / decode_steps,
        },
        "attention_families": families,
        "limitations": [
            "Kernel-duration share is the sum of selected dispatch durations, not wall-clock share and not achieved occupancy.",
            "Wavefront calculations assume the observed R9700 wave32 geometry; they are launch-supply proxies, not a sampled residency metric.",
            "The F16 profile is intentionally CPU-limited and profiler-instrumented; it is not a throughput replacement for the unprofiled baseline.",
        ],
    }
    with (args.out_dir / "profile-summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
