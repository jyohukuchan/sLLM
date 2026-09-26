#!/usr/bin/env python3
"""Build and run the probe-only Phase 87 WU-P1 paged attention comparison.

Each case runs the current contiguous provider and its paged counterpart in
one process with AB/BA/AB ordering. The generated binary and raw case reports
stay in the caller's output directory; this runner does not modify runtime
source, the public ABI, or the production provider selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
SOURCE = "native/hip/tests/phase87_wup1_paged_attention_probe.hip.cpp"
SOURCES = (
    SOURCE,
    "native/hip/tests/phase87_wup1_paged_decode.hpp",
    "native/hip/tests/phase87_wup1_paged_prefill.hpp",
    "native/hip/src/causal_attention_kernel.hip.cpp",
    "native/hip/src/causal_attention_kernel_internal.hpp",
    "native/lowp/include/lowp/detail/low_precision_block_codec.hpp",
    "include/sllm/hip.h",
)
UUID = {
    "gfx1030": "GPU-76a08c022586fed6",
    "gfx1201": "GPU-a8e9ddefa2d60f55",
}
LENGTHS = (1023, 1024, 1025, 8192, 8193, 65536)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases() -> list[tuple[str, int, int, int, int, int]]:
    selected: list[tuple[str, int, int, int, int, int]] = []
    for length in LENGTHS:
        for m in (1, 2, 3):
            selected.append(("decode", length, m, 3, 5, 4 if length == 65536 else 16))
    for length in LENGTHS:
        selected.append(("prefill", length, 128, 3, 5, 1))
    # Both sides of the production qtile8 M=128 boundary and one non-aligned
    # representative prefill tile are covered at long KV lengths.
    selected.extend(
        [
            ("prefill", 8192, 127, 3, 5, 1),
            ("prefill", 8192, 129, 3, 5, 1),
            ("prefill", 8193, 219, 3, 5, 1),
        ]
    )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=tuple(UUID), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target: str = args.target
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    case_dir = output / "cases"
    case_dir.mkdir()
    binary = output / f"phase87-wup1-{target}"
    command = [
        "/opt/rocm/bin/amdclang++",
        "-O3",
        "-ffp-contract=off",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-x",
        "hip",
        f"--offload-arch={target}",
        "--hip-link",
        "-I",
        "include",
        "-I",
        "native/hip/src",
        "-I",
        "native/lowp/include",
        SOURCE,
        "-L/opt/rocm/lib",
        "-lamdhip64",
        "-lrocblas",
        "-o",
        str(binary),
    ]
    compiler_version = subprocess.check_output(
        [command[0], "--version"], text=True
    ).splitlines()[0]
    start = time.time()
    built = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
    (output / "build.stdout").write_text(built.stdout)
    (output / "build.stderr").write_text(built.stderr)
    if built.returncode != 0:
        print(f"WU-P1 {target} compile failed; see {output / 'build.stderr'}", file=sys.stderr)
        return 2

    env = dict(os.environ)
    for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                "SLLM_WUP1_IDENTITY_TABLE"):
        env.pop(key, None)
    env["ROCR_VISIBLE_DEVICES"] = UUID[target]
    env["LD_LIBRARY_PATH"] = "/opt/rocm/lib"
    if target == "gfx1030":
        env["DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING"] = "0"
    else:
        env.pop("DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING", None)

    entries: list[dict] = []
    for kind, length, m, rounds, samples, replays in cases():
        label = f"{kind}-kv{length}-m{m}"
        selected = [str(binary), target, kind, str(length), str(m),
                    str(rounds), str(samples), str(replays)]
        print(f"{target} {label} start", flush=True)
        launched = time.time()
        completed = subprocess.run(selected, cwd=REPO, env=env, text=True,
                                   capture_output=True)
        (case_dir / f"{label}.stderr").write_text(completed.stderr)
        (case_dir / f"{label}.stdout").write_text(completed.stdout)
        if completed.returncode != 0:
            entry = {"kind": kind, "length": length, "m": m,
                     "state": "FAIL", "exit_code": completed.returncode,
                     "elapsed_seconds": time.time() - launched}
        else:
            try:
                entry = json.loads(completed.stdout)
                reported_rounds = entry.get("rounds")
                if (not isinstance(reported_rounds, list) or
                        len(reported_rounds) != rounds or
                        any(not isinstance(item, dict) for item in reported_rounds)):
                    raise ValueError("invalid round reports")
                increases = [item.get("increase_percent") for item in reported_rounds]
                if any(not isinstance(value, (int, float)) or
                       not math.isfinite(value) for value in increases):
                    raise ValueError("invalid performance values")
                measured_performance_ok = all(value < 10.0 for value in increases)
                oracle = entry.get("oracle_max_abs")
                if (entry.get("target") != target or entry.get("kind") != kind or
                        entry.get("length") != length or entry.get("m") != m or
                        entry.get("page_tokens") != 128 or
                        entry.get("page_table_layout") != "reverse_by_page" or
                        entry.get("kv_encoding") != "mxfp8-e4" or
                        entry.get("round_count") != rounds or
                        entry.get("samples_per_variant") != samples or
                        entry.get("replays_per_sample") != replays or
                        not entry.get("gpu_execution") or entry.get("fallback_used") or
                        not entry.get("cleanup_zero") or
                        not entry.get("candidate_bitwise") or
                        not entry.get("repeat") or not entry.get("guards") or
                        not isinstance(oracle, (int, float)) or
                        not math.isfinite(oracle) or oracle > 0.03125 or
                        [item.get("order") for item in reported_rounds] !=
                        ["AB", "BA", "AB"] or
                        entry.get("performance_below_10pct") is not
                        measured_performance_ok or
                        entry.get("state") != ("PASS" if measured_performance_ok
                                               else "PERFORMANCE_REVIEW")):
                    raise ValueError("case report violates WU-P1 contract")
                entry["elapsed_seconds"] = time.time() - launched
            except (json.JSONDecodeError, ValueError) as exc:
                entry = {"kind": kind, "length": length, "m": m,
                         "state": "FAIL", "error": str(exc),
                         "exit_code": completed.returncode,
                         "elapsed_seconds": time.time() - launched}
        (case_dir / f"{label}.json").write_text(json.dumps(entry, indent=2) + "\n")
        entries.append(entry)
        print(f"{target} {label} {entry['state']}", flush=True)
        if entry["state"] == "FAIL":
            break

    complete = len(entries) == len(cases())
    numeric_ok = complete and all(
        e["state"] in ("PASS", "PERFORMANCE_REVIEW") for e in entries
    )
    performance_ok = numeric_ok and all(e["state"] == "PASS" for e in entries)
    summary = {}
    for kind in ("decode", "prefill"):
        observed = [r["increase_percent"] for e in entries
                    if e["state"] in ("PASS", "PERFORMANCE_REVIEW") and
                    e["kind"] == kind
                    for r in e["rounds"]]
        summary[kind] = {"cases": sum(e["kind"] == kind for e in entries),
                         "max_increase_percent": max(observed) if observed else None,
                         "rounds": len(observed)}
    state = "PASS" if performance_ok else "PERFORMANCE_REVIEW" if numeric_ok else "FAIL"
    report = {
        "schema_version": "phase87-wup1-paged-suite-v1",
        "state": state,
        "target": target,
        "visible_gpu_uuid": UUID[target],
        "compiler_version": compiler_version,
        "compile_command": command,
        "binary": str(binary),
        "binary_sha256": digest(binary),
        "source_sha256": {path: digest(REPO / path) for path in SOURCES},
        "device_only": True,
        "fallback_used": False,
        "cases_expected": len(cases()),
        "cases_completed": len(entries),
        "numeric_ok": numeric_ok,
        "performance_below_10pct": performance_ok,
        "summary": summary,
        "cases": entries,
        "elapsed_seconds": time.time() - start,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"WU-P1 {target}: {state} cases={len(entries)}/{len(cases())}", flush=True)
    return 0 if state == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
