#!/usr/bin/env python3
"""Attribute the DD ROCprof traces to the benchmark's measured token phases.

Token boundaries are reconstructed from one invariant per executed token:
Gemma emits 35 direct-attention dispatches and the MoE emits 40 route
dispatches.  The initial warm-up / untimed setup token groups are deliberately
excluded from the reported prefill/decode phases.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_int(value):
    return int(value)


def run(kernel_path, api_path, marker, markers_per_token, phases, wall_ms, out_path,
        stop_after_last_name=None):
    events = []
    for row in rows(kernel_path):
        events.append({
            "name": row["Kernel_Name"],
            "correlation": row["Correlation_Id"],
            "start": as_int(row["Start_Timestamp"]),
            "end": as_int(row["End_Timestamp"]),
        })
    events.sort(key=lambda item: (item["start"], item["end"]))
    if stop_after_last_name:
        last = max(i for i, item in enumerate(events) if item["name"] == stop_after_last_name)
        events = events[:last + 1]
    marker_indexes = [i for i, item in enumerate(events) if item["name"] == marker]
    if len(marker_indexes) % markers_per_token:
        raise SystemExit(f"{len(marker_indexes)} {marker} events is not divisible by {markers_per_token}")
    token_count = len(marker_indexes) // markers_per_token
    if sum(stop - start for start, stop in phases.values()) > token_count:
        raise SystemExit(f"phase tokens exceed reconstructed count {token_count}")

    # A token's interval begins at its first marker.  The small first-layer
    # prefix before that marker is assigned to the preceding token; this keeps
    # all whole dispatches, and phase boundaries move by at most one layer.
    token_for_event = {}
    start_indexes = [marker_indexes[t * markers_per_token] for t in range(token_count)]
    next_token = 0
    for event_index in range(start_indexes[0], len(events)):
        while next_token + 1 < token_count and event_index >= start_indexes[next_token + 1]:
            next_token += 1
        token_for_event[event_index] = next_token

    api_end = {}
    for row in rows(api_path):
        if row["Function"] in {"hipModuleLaunchKernel", "hipLaunchKernel"}:
            api_end[row["Correlation_Id"]] = as_int(row["End_Timestamp"])

    result = {"marker": marker, "markers_per_token": markers_per_token, "reconstructed_tokens": token_count,
              "phase_boundary_method": "first marker of the next token", "phases": {}}
    for phase, (token_start, token_stop) in phases.items():
        selected = [event for i, event in enumerate(events)
                    if token_start <= token_for_event.get(i, -1) < token_stop]
        totals = defaultdict(lambda: [0, 0])
        for event in selected:
            totals[event["name"]][0] += event["end"] - event["start"]
            totals[event["name"]][1] += 1
        kernel_ns = sum(total for total, _ in totals.values())
        gap_ns = 0
        post_launch_gap_ns = 0
        gap_count = 0
        post_launch_gap_count = 0
        for previous, following in zip(selected, selected[1:]):
            gap = following["start"] - previous["end"]
            if gap <= 0:
                continue
            gap_ns += gap
            gap_count += 1
            if api_end.get(following["correlation"], 2**63 - 1) <= previous["end"]:
                post_launch_gap_ns += gap
                post_launch_gap_count += 1
        wall_ns = int(wall_ms[phase] * 1_000_000)
        symbols = [
            {"symbol": name, "total_ms": total / 1e6, "launches": launches,
             "share_of_kernel_time": total / kernel_ns if kernel_ns else 0.0}
            for name, (total, launches) in totals.items()
        ]
        symbols.sort(key=lambda item: item["total_ms"], reverse=True)
        result["phases"][phase] = {
            "token_range": [token_start, token_stop],
            "wall_ms_from_driver": wall_ms[phase],
            "kernel_sum_ms": kernel_ns / 1e6,
            "wall_not_in_any_kernel_ms": (wall_ns - kernel_ns) / 1e6,
            "kernel_fraction_of_wall": kernel_ns / wall_ns if wall_ns else 0.0,
            "positive_inter_kernel_gap_ms": gap_ns / 1e6,
            "positive_inter_kernel_gaps": gap_count,
            "gap_after_next_launch_returned_ms": post_launch_gap_ns / 1e6,
            "gaps_after_next_launch_returned": post_launch_gap_count,
            "gap_time_fraction_after_next_launch_returned": post_launch_gap_ns / gap_ns if gap_ns else 0.0,
            "gap_count_fraction_after_next_launch_returned": post_launch_gap_count / gap_count if gap_count else 0.0,
            "symbols": symbols,
        }
        if marker == "ullm_paged_decode_attn_f32_kernel" and markers_per_token == 35:
            full_layers = {4, 9, 14, 19, 24, 29, 34}
            split = defaultdict(lambda: [0, 0])
            for token in range(token_start, token_stop):
                for layer in range(markers_per_token):
                    event = events[marker_indexes[token * markers_per_token + layer]]
                    key = "full_attention" if layer in full_layers else "sliding_attention"
                    split[key][0] += event["end"] - event["start"]
                    split[key][1] += 1
            attention_ns = sum(value[0] for value in split.values())
            result["phases"][phase]["attention_kind_split"] = {
                key: {"total_ms": value[0] / 1e6, "launches": value[1],
                      "share_of_attention_time": value[0] / attention_ns if attention_ns else 0.0}
                for key, value in split.items()
            }
    Path(out_path).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    # model, kernel trace, HIP API trace, wall-time JSON, result JSON
    model, kernel, api, timing, output = sys.argv[1:]
    timing = json.loads(Path(timing).read_text())
    if model == "gemma":
        workload = timing["workload"]
        run(kernel, api, "ullm_paged_decode_attn_f32_kernel", 35,
            {"prefill": (6, 24), "decode": (42, 54)},
            {"prefill": workload["prefill"]["total_elapsed_seconds"] * 1000,
             "decode": workload["decode"]["total_elapsed_seconds"] * 1000}, output)
    elif model == "moe":
        generation = timing["generation"]
        prompt_tokens = len(generation["prompt_token_ids"])
        decode_tokens = len(generation["generated_token_ids"])
        run(kernel, api, "ullm_moe_route_f32_kernel", 40,
            {"prefill": (0, prompt_tokens), "decode": (prompt_tokens, prompt_tokens + decode_tokens)},
            {"prefill": generation["prompt_wall_ms"], "decode": generation["decode_wall_ms"]}, output,
            stop_after_last_name="ullm_top1_f32_kernel")
    else:
        raise SystemExit("model must be gemma or moe")
