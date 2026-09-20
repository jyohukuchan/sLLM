#!/usr/bin/env python3
"""Separate GPU execution and observed GPU-idle intervals in a decode range.

HIP API wall durations overlap device execution. This tool intersects each API
category with GPU-idle time rather than adding CPU and GPU durations together.
Idle attribution describes where the host was observed, not a causal proof of
why the GPU was idle. Profiling overhead remains part of the observation.
"""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

from phase87_decode_profile import _family


def hip_category(name):
    if any(word in name for word in ("Synchronize", "EventQuery", "StreamQuery")):
        return "idle_in_sync_or_poll"
    if "Launch" in name:
        return "idle_in_launch"
    if "Memcpy" in name or "Memset" in name:
        return "idle_in_transfer_api"
    return "idle_in_other_hip_api"


def breakdown(analysis_path, hip_path, output):
    analysis = json.loads(analysis_path.read_text())
    segment = analysis["decode_segment"]
    start, end = segment["start_timestamp_ns"], segment["end_timestamp_ns"]
    transitions = analysis["benchmark"]["row"]["decode_transition_count"]
    if not isinstance(transitions, int) or transitions <= 0:
        raise ValueError("positive committed decode transition count required")
    kernel_path = Path(analysis["raw_sha256"]["kernel_trace"]["path"])
    events = [(start, "boundary", 0), (end, "boundary", 0)]
    family_ns = collections.Counter()
    kernel_count = 0
    with kernel_path.open() as stream:
        for row in csv.DictReader(stream):
            a, b = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
            if a < start or b > end:
                continue
            events.extend(((a, "gpu", 1), (b, "gpu", -1)))
            family_ns[_family(row["Kernel_Name"])] += b - a
            kernel_count += 1
    api_ns = collections.Counter()
    with hip_path.open() as stream:
        for row in csv.DictReader(stream):
            a = max(start, int(row["Start_Timestamp"]))
            b = min(end, int(row["End_Timestamp"]))
            if b <= a:
                continue
            category = hip_category(row["Function"])
            events.extend(((a, category, 1), (b, category, -1)))
            api_ns[category] += b - a
    priorities = ["gpu", "idle_in_sync_or_poll", "idle_in_launch",
                  "idle_in_transfer_api", "idle_in_other_hip_api"]
    active, partition = collections.Counter(), collections.Counter()
    previous = start
    for time_ns, category, delta in sorted(events):
        duration = time_ns - previous
        if duration:
            owner = next((kind for kind in priorities if active[kind] > 0),
                         "idle_outside_hip_api")
            partition[owner] += duration
        active[category] += delta
        previous = time_ns
    if sum(partition.values()) != end - start:
        raise ValueError("timeline partition does not cover decode")
    if partition["gpu"] != segment["gpu_busy_union_ns"]:
        raise ValueError("GPU coverage differs from profile analysis")
    if kernel_count != segment["dispatch_count"]:
        raise ValueError("kernel coverage differs from profile analysis")
    result = {
        "schema": "phase87-profile-breakdown-v1", "state": "PASS",
        "target": analysis["benchmark"]["target"],
        "mtp": analysis["benchmark"]["mtp"]["requested"],
        "committed_decode_transitions": transitions,
        "profile_wall_ms_per_token": (end - start) / transitions / 1e6,
        "family_ms_per_token": {key: value / transitions / 1e6
                                for key, value in family_ns.most_common()},
        "timeline_ms_per_token": {key: value / transitions / 1e6
                                  for key, value in partition.items()},
        "family_share_of_profile_wall": {key: value / (end - start)
                                         for key, value in family_ns.items()},
        "hip_api_wall_ns_including_gpu_overlap": dict(api_ns),
        "kernel_dispatch_count": kernel_count,
        "caveat": "profile diagnostic; GPU-idle attribution is observational, not causal; HIP API totals overlap GPU and must not be added",
        "inputs": {str(p): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
                   for p in (analysis_path, hip_path, kernel_path)},
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("target", "mtp", "family_ms_per_token", "timeline_ms_per_token")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--hip-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    breakdown(args.analysis, args.hip_trace, args.output)


if __name__ == "__main__":
    main()
