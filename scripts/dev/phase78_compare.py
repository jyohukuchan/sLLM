#!/usr/bin/env python3
"""Compare reports against the original Phase 78 performance targets.

Historical comparison only: the user accepted changed targets and incomplete
final measurements on 2026-09-05. This script does not decide Phase completion.
GPU/numerical evidence remains separate.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROWS = [(17, 17), (512, 32), (2048, 128), (9435, 128)]
LIMITS = {"gfx1030": (340.80, 16.86), "gfx1201": (779.06, 21.07)}
FIXTURE = "sha256:50ae5d562b673cf68ea58ee93989356bdb5955693d47b1756331da3988081b80"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(sllm, llama):
    target = sllm["target"]
    require(target in LIMITS and llama["config"]["target"] == target, "target mismatch")
    require(sllm["schema_version"] == "phase78-qwen38-resident-benchmark-v3", "sLLM must include the prefill token in the output budget")
    require(sllm["state"] == llama["state"] == "PASS", "run failed")
    require(sllm["fixture"]["sha256"] == llama["config"]["fixture_sha256"] == FIXTURE, "fixture mismatch")
    require(sllm["is_phase78_final"], "sLLM is not a full 3+10 protocol run")
    require(llama["config"]["warmups"] == 3 and llama["config"]["measured"] == 10, "llama repetitions differ")
    require(llama["config"]["schema"] == "phase78-llama-fixed-fixture-v2", "llama output accounting is not verified")
    require(llama["config"]["source_head"] == "4df29be4f4c3673f428170fda944a5b19f743bb8", "llama source differs")
    require(not llama["config"]["mtp"], "llama MTP is enabled")
    require(sllm["protocol"]["kv_cache"] == "FP16", "sLLM KV differs")
    require(sllm["cleanup"]["zero"] and llama["server_exit_code"] == 0, "cleanup failed")
    require([(r["prompt_tokens"], r["output_tokens"]) for r in sllm["rows"]] == ROWS, "sLLM rows differ")
    require([(r["prompt_tokens"], r["output_budget"]) for r in llama["rows"]] == ROWS, "llama rows differ")
    rows = []
    for spec, sr, lr in zip(ROWS, sllm["rows"], llama["rows"]):
        for engine, runs in [("sLLM", sr["runs"]), ("llama", lr["runs"])]:
            require(len(runs) == 13, f"{engine} {spec}: missing samples")
            require(sum(r["sample_kind"] == "warmup" for r in runs) == 3, f"{engine} warmups differ")
            require(sum(r["sample_kind"] == "measured" for r in runs) == 10, f"{engine} measured samples differ")
            require(all(len(r["generated_tokens"]) == spec[1] for r in runs), f"{engine} output count differs")
            require(all(r["generated_tokens"] == runs[0]["generated_tokens"] for r in runs), f"{engine} is not deterministic")
        require(all(r["decode_transition_count"] == spec[1] - 1 for r in sr["runs"]), "sLLM decode count differs")
        require(all(r["decoded_transition_count"] == spec[1] - 1 for r in lr["runs"]), "llama decode count differs")
        require(all(r["stop_reason"] == "length" for r in sr["runs"]), "sLLM stop reason differs")
        require(all(r["stop_reason"] == "limit" for r in lr["runs"]), "llama stop reason differs")
        require(all(r["audit"]["all_dispatches_hip"] and not r["audit"]["fallback_used"] for r in sr["runs"]), "sLLM dispatch failed")
        ss, ls = sr["measured_summary"], lr["measured_summary"]
        comparisons = {}
        for key, lk in [("prefill_ms", "prompt_ms"), ("ttft_ms", "ttft_ms")]:
            a, b = ss[key], ls[lk]
            allowance = max(a["mad"], b["mad"])
            comparisons[key] = {"sllm": a, "llama": b, "allowed_sllm_median_ms": b["median"] + allowance,
                                "pass": a["median"] <= b["median"] + allowance}
        rows.append({"prompt_tokens": spec[0], "output_tokens": spec[1], "relative_gates": comparisons,
                     "sllm_tpot_ms": ss["tpot_ms"], "llama_tpot_ms": ls["predicted_per_token_ms"],
                     "sllm_e2e_ms": ss["e2e_ms"], "llama_e2e_ms": ls["e2e_ms"]})
    long_summary = sllm["rows"][-1]["measured_summary"]
    absolute = {}
    for key, limit in zip(["prefill_tokens_per_second", "decode_tokens_per_second"], LIMITS[target]):
        value = long_summary[key]["median"]
        absolute[key] = {"sllm": value, "minimum": limit, "pass": value >= limit}
    return {"schema": "phase78-timing-comparison-v1", "target": target,
            "comparison": "E1 system-equivalent; model encodings and KV formats differ",
            "performance_gates_pass": all(g["pass"] for g in absolute.values()) and
                all(g["pass"] for row in rows for g in row["relative_gates"].values()),
            "absolute_gates": absolute, "rows": rows,
            "scope": "Timing comparison only. Numerical classification, exact binary/GPU identity, hardware read counters, GPU family times, and no GTT spill require separate evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sllm", type=Path)
    parser.add_argument("llama", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.sllm.read_text()), json.loads(args.llama.read_text()))
    result["inputs"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.sllm, args.llama]}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("PASS" if result["performance_gates_pass"] else "UNMET")


if __name__ == "__main__":
    main()
