#!/usr/bin/env python3
"""Compare matched-input greedy/host/GPU sampling reports; no completion gate.

Run sllm-phase78-qwen38-benchmark with SLLM_PHASE81_REPLAY=1 and
SLLM_PHASE81_SAMPLING=greedy, host-fixed, gpu-fixed respectively. Enable the
same accepted fast paths for all three. GPU correctness and API/free-generation
measurements are separate evidence.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def compare(reports, clock_ticks):
    require(clock_ticks > 0, "CLK_TCK must be positive")
    a, b, c = reports
    for label, report, mode in zip("ABC", reports, ["greedy", "host_fixed", "gpu_fixed"]):
        require(report["state"] == "PASS" and report["cleanup"]["zero"], f"{label}: execution/cleanup failed")
        require(report["sampling"]["mode"] == mode and report["sampling"]["replay_inputs"], f"{label}: wrong mode or missing input replay")
        for key in ["target", "device_index", "model", "fixture", "selector_environment", "repetitions"]:
            require(report[key] == a[key], f"{label}: {key} differs")
        for key in ["kv_cache", "state_capacity_tokens", "prefill_chunk_capacity_tokens", "mtp", "termination"]:
            require(report["protocol"][key] == a["protocol"][key], f"{label}: protocol {key} differs")
        require(report["sampling"]["seed"] == a["sampling"]["seed"], f"{label}: seed differs")
        require([(r["prompt_tokens"], r["output_tokens"]) for r in report["rows"]] ==
                [(r["prompt_tokens"], r["output_tokens"]) for r in a["rows"]], f"{label}: row budgets differ")
    result = []
    for rows in zip(a["rows"], b["rows"], c["rows"]):
        for label, row in zip("ABC", rows):
            require(row["prompt_prefix_sha256"] == rows[0]["prompt_prefix_sha256"], f"{label}: prompt differs")
            measured = [run for run in row["runs"] if run["sample_kind"] == "measured"]
            require(len(measured) == a["repetitions"]["measured_per_row"] and len(measured) > 0, "measured samples missing")
            for run in row["runs"]:
                audit = run["audit"]
                require(audit["selected_backend"] == "hip" and audit["target"] == a["target"] and
                        audit["all_dispatches_hip"] and not audit["fallback_used"], f"{label}: HIP dispatch invalid")
                require(run["decode_transition_count"] == row["output_tokens"] - 1, f"{label}: decode count differs")
                require(len(run["generated_tokens"]) == row["output_tokens"], f"{label}: output count differs")
                # Both A and C must retain the accepted optimizations, rather
                # than compare against an artificially slow Argmax baseline.
                if label in "AC":
                    for key in ["graph_replay_count", "graph_span_count", "kv_append_attention_chain_count"]:
                        require(audit[key] > 0, f"{label}: {key} was not used")
        metrics = {}
        for key in ["prefill_ms", "ttft_ms", "tpot_ms", "decode_tokens_per_second", "e2e_ms"]:
            values = [row["measured_summary"][key] for row in rows]
            require(all(math.isfinite(v["median"]) and v["median"] > 0 for v in values), f"invalid {key}")
            metrics[key] = dict(zip("ABC", values))
            metrics[key]["C_vs_A_percent"] = (values[2]["median"] / values[0]["median"] - 1) * 100
            metrics[key]["C_vs_B_percent"] = (values[2]["median"] / values[1]["median"] - 1) * 100
        cpu = {}
        for label, row in zip("ABC", rows):
            measured = [run for run in row["runs"] if run["sample_kind"] == "measured"]
            cpu[label] = {}
            for phase in ["prefill", "decode"]:
                ticks = [run.get("cpu", {}).get(f"{phase}_user_system_ticks") for run in measured]
                cpu[label][f"{phase}_cpu_ms_median"] = (
                    None if any(t is None for t in ticks) else statistics.median(ticks) * 1000 / clock_ticks
                )
        result.append({"prompt_tokens": rows[0]["prompt_tokens"], "output_tokens": rows[0]["output_tokens"],
                       "metrics": metrics, "cpu": cpu})
    return {"schema": "phase81-matched-sampling-comparison-v1", "target": a["target"], "clock_ticks_per_second": clock_ticks,
            "rows": result, "scope": "Matched-input direct runtime timing; CPU tick resolution limits small intervals. Does not prove GPU numerical correctness, real API performance, model quality, or Phase completion."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("c", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--clock-ticks", type=int, required=True, help="getconf CLK_TCK on the measured host")
    args = parser.parse_args()
    paths = [args.a, args.b, args.c]
    result = compare([json.loads(p.read_text()) for p in paths], args.clock_ticks)
    result["input_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
