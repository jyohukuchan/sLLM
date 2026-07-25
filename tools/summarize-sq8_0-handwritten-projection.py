#!/usr/bin/env python3
"""Create a provenance-preserving summary for the SQ8_0 WMMA feasibility run."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


CK_STATIC_SOURCE = (
    "benchmarks/results/2026-07-26/"
    "sq8-r9700-handwritten-kernel-phase0-v0.1/static/offline-codegen-metadata.md"
)

CK_STATIC_FORMS = [
    {
        "form": "Default 16x128x128, VmemReadVec 8",
        "static_lds_bytes": 18_432,
        "vgpr_per_thread": 100,
        "sgpr_per_wave": 47,
        "lds_only_ceiling": "3 workgroups/CU = 24 wave32 = 75% of a 32-wave reference",
    },
    {
        "form": "KPadding 16x128x256, VmemReadVec 16",
        "static_lds_bytes": 36_864,
        "vgpr_per_thread": 242,
        "sgpr_per_wave": 48,
        "lds_only_ceiling": "1 workgroup/CU = 8 wave32 = 25% of a 32-wave reference",
    },
    {
        "form": "Default 16x128x256, VmemReadVec 16",
        "static_lds_bytes": 36_864,
        "vgpr_per_thread": 175,
        "sgpr_per_wave": 46,
        "lds_only_ceiling": "1 workgroup/CU = 8 wave32 = 25% of a 32-wave reference",
    },
    {
        "form": "Default 16x256x128, VmemReadVec 8",
        "static_lds_bytes": 34_816,
        "vgpr_per_thread": 154,
        "sgpr_per_wave": 49,
        "lds_only_ceiling": "1 workgroup/CU = 8 wave32 = 25% of a 32-wave reference",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--attempt", default="attempt-2")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def parse_watch(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    _, _, payload = text.partition("\n")
    decoder = json.JSONDecoder()
    index = 0
    samples: list[dict[str, Any]] = []
    while True:
        while index < len(payload) and payload[index].isspace():
            index += 1
        if index == len(payload):
            break
        value, index = decoder.raw_decode(payload, index)
        if not isinstance(value, list):
            raise ValueError(f"{path} contains a non-list watch record")
        for record in value:
            if isinstance(record, dict) and record.get("gpu") == 2:
                samples.append(record)
    if not samples:
        raise ValueError(f"{path} contains no GPU 2 samples")
    return samples


def range_and_mean(values: list[float | int]) -> dict[str, float | int]:
    return {"min": min(values), "max": max(values), "mean": statistics.fmean(values)}


def telemetry_summary(path: Path) -> dict[str, Any]:
    samples = parse_watch(path)
    throttles = Counter(str(sample["power"]["throttle_status"]) for sample in samples)
    return {
        "raw_watch": str(path),
        "samples": len(samples),
        "timestamp_first": samples[0].get("timestamp"),
        "timestamp_last": samples[-1].get("timestamp"),
        "throttle_status_counts": dict(sorted(throttles.items())),
        "edge_c": range_and_mean([sample["temperature"]["edge"]["value"] for sample in samples]),
        "hotspot_c": range_and_mean(
            [sample["temperature"]["hotspot"]["value"] for sample in samples]
        ),
        "memory_c": range_and_mean([sample["temperature"]["mem"]["value"] for sample in samples]),
        "gfx_mhz": range_and_mean([sample["clock"]["gfx_0"]["clk"]["value"] for sample in samples]),
        "memory_mhz": range_and_mean(
            [sample["clock"]["mem_0"]["clk"]["value"] for sample in samples]
        ),
        "socket_power_w": range_and_mean(
            [sample["power"]["socket_power"]["value"] for sample in samples]
        ),
        "interpretation": (
            "Raw AMD SMI samples. Throttle status is recorded, but its physical cause is not "
            "identified by this evidence."
        ),
    }


def component_baseline(component: dict[str, Any]) -> dict[str, Any]:
    total_time_us = 0.0
    total_bytes = 0
    shapes: list[dict[str, Any]] = []
    for shape in component["shapes"]:
        calls = int(shape["per_layer_calls"])
        timing = shape["timing"]
        time_us = timing["ck_us"]
        if not isinstance(time_us, (int, float)):
            raise ValueError(f"CK timing missing for {shape['family']}")
        route_bytes = int(shape["logical_route_bytes"]["ck_including_bf16_workspace_and_f32_output"])
        total_time_us += calls * time_us
        total_bytes += calls * route_bytes
        shapes.append(
            {
                "family": shape["family"],
                "m": shape["m"],
                "n": shape["n"],
                "k": shape["k"],
                "per_layer_calls": calls,
                "ck_instance": shape["ck_instance"],
                "ck_us_per_launch": time_us,
                "logical_route_bytes_per_launch": route_bytes,
                "logical_gb_s": timing["ck_logical_gb_s"],
                "logical_to_nominal_hbm_ratio": timing["ck_theoretical_hbm_ratio"],
                "ns_per_output_element": timing["ck_ns_per_output_element"],
            }
        )
    logical_gb_s = total_bytes / total_time_us / 1000.0
    return {
        "measurement": "HIP event timing of the exact selected CK helper plus its BF16-to-F32 boundary",
        "traffic_metric": (
            "logical route bytes / event time; this is not physical HBM bandwidth because the "
            "available PMC counters report unusable byte values"
        ),
        "nominal_hbm_gb_s_reference": component["theoretical_hbm_gb_s"],
        "shapes": shapes,
        "weighted_by_7_projections_per_layer": {
            "logical_route_bytes_per_layer": total_bytes,
            "time_us_per_layer": total_time_us,
            "logical_gb_s": logical_gb_s,
            "logical_to_nominal_hbm_ratio": logical_gb_s / component["theoretical_hbm_gb_s"],
        },
        "40_layer_decode_projection_subtotal": {
            "logical_route_bytes": total_bytes * 40,
            "time_us": total_time_us * 40,
        },
    }


def main() -> int:
    args = parse_args()
    root = args.result_root
    attempt = root / args.attempt
    static = read_json(root / "static/handwritten-isa-summary.json")
    component_gate = read_json(attempt / "component/gate.json")
    ck_baseline = read_json(attempt / "component/ck-baseline.json")
    full_model_gate = read_json(attempt / "full-model-multistep/gate.json")
    result = {
        "schema_version": "ullm.sq8_0.handwritten_projection_feasibility_summary.v1",
        "scope": "private gfx1201 WMMA M=1 SQ8_0 prototype; no default dispatch replacement",
        "result_root": str(root),
        "attempt": args.attempt,
        "ck_selector_and_static_reference": {
            "source": CK_STATIC_SOURCE,
            "m1_shape_mapping": {
                "q_o": "[1,5120] x [5120,5120] -> Default 16x128x128",
                "k_v": "[1,5120] x [1024,5120] -> Default 16x128x128",
                "gate_up": "[1,5120] x [17408,5120] -> KPadding 16x128x256",
                "down": "[1,17408] x [5120,17408] -> Default 16x128x256",
            },
            "m_tail": "M=1 is a tail against CK's MPerBlock=16; N and K are exact multiples of 128.",
            "forms": CK_STATIC_FORMS,
        },
        "handwritten_static_isa": static,
        "handwritten_runtime_resource_query": component_gate["handwritten_hip_resource_query"],
        "resource_query_caveat": (
            "The raw attempt-2 `threads_per_block=1024` field was populated from HIP's "
            "maxThreadsPerBlock capability by the pre-correction binary, not the 32-thread launch. "
            "The source was corrected after this window; the correction was not remeasured to avoid "
            "another service window. `active_blocks_per_cu` retains HIP's own per-multiprocessor term."
        ),
        "component_numerical_gate": component_gate["numeric_gate"],
        "ck_component_baseline": component_baseline(ck_baseline),
        "full_model_multistep_gate": full_model_gate,
        "candidate_timing": {
            "performed": False,
            "reason": "forbidden by the frozen policy because the full-model multi-step gate failed",
        },
        "telemetry": telemetry_summary(attempt / "telemetry/during-window.watch.txt"),
        "service_windows": {
            "attempt_1": {
                "start": (root / "service/window-start.txt").read_text().strip(),
                "end": (root / "service/window-end.txt").read_text().strip(),
                "gpu_work_started": False,
                "outcome": "aborted before GPU work because AMD SMI's no-process sentinel was parsed as a process; service restored",
            },
            "attempt_2": {
                "start": (attempt / "service/window-start.txt").read_text().strip(),
                "isolation_complete": (attempt / "service/isolation-complete.txt").read_text().strip(),
                "end": (attempt / "service/window-end.txt").read_text().strip(),
                "gpu_work_started": True,
                "restore_record": str(attempt / "service/restore.txt"),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
