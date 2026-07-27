#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0
# One-command standalone build/run wrapper. It never controls services.
set -euo pipefail
usage() { cat >&2 <<'EOF'
usage: run-hw-microbench-rdna4-cdna3.sh --arch gfx1201|gfx942 --results-dir /absolute/dir [--amd-smi-gpu INDEX] [--build-only]

The default run builds and ISA-audits, samples amd-smi continuously during a
clock warmup and every measurement group, and refuses to time throughput until
three consecutive active GFX-clock samples are stable.  Pass --amd-smi-gpu for
an actual run; it is optional only for --build-only.
EOF
}
arch= results= amd_smi_gpu= build_only=0
while [[ $# -gt 0 ]]; do case "$1" in --arch) arch=${2:-};shift 2;;--results-dir)results=${2:-};shift 2;;--amd-smi-gpu)amd_smi_gpu=${2:-};shift 2;;--build-only)build_only=1;shift;;-h|--help)usage;exit 0;;*)usage;exit 2;;esac;done
[[ $arch == gfx1201 || $arch == gfx942 ]] && [[ $results == /* ]] || { usage; exit 2; }
(( build_only )) || [[ $amd_smi_gpu =~ ^[0-9]+$ ]] || { echo '--amd-smi-gpu is required for a measured run' >&2; exit 2; }
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P); rocm=${ROCM_PATH:-/opt/rocm}; mkdir -p -- "$results"
binary="$results/hw-microbench-$arch"; "$rocm/bin/hipcc" -std=c++20 -O3 --offload-arch="$arch" "$repo/tools/hw-microbench-rdna4-cdna3.hip.cpp" -o "$binary" >"$results/build.log" 2>&1
"$repo/tools/audit-hw-microbench-isa.sh" --repo "$repo" --arch "$arch" --output-dir "$results/isa" >"$results/isa.log" 2>&1
if (( build_only )); then exit 0; fi

sample_pid=
sample_metrics() {
  local phase=$1 out=$2 raw now
  : >"$out"
  while :; do
    now=$(date --iso-8601=ns)
    # `--violation` was removed by amd-smi 26.x (ROCm 7.2.4).  Clock, power,
    # and temperature are the telemetry required by this benchmark and remain
    # available on both the older and newer CLI variants.
    if raw=$(amd-smi metric --gpu "$amd_smi_gpu" --temperature --clock --power --json 2>&1); then
      jq -cn --arg timestamp "$now" --arg phase "$phase" --argjson metric "$raw" '{timestamp:$timestamp,phase:$phase,metric:$metric}' >>"$out"
    else
      jq -cn --arg timestamp "$now" --arg phase "$phase" --arg error "$raw" '{timestamp:$timestamp,phase:$phase,error:$error}' >>"$out"
    fi
    sleep 0.25
  done
}
start_sampler() { sample_metrics "$1" "$results/telemetry-$1.jsonl" & sample_pid=$!; }
stop_sampler() { kill "$sample_pid" 2>/dev/null || true; wait "$sample_pid" 2>/dev/null || true; sample_pid=; }
trap '[[ -z ${sample_pid:-} ]] || stop_sampler' EXIT
copy_latest() { tail -n 1 "$results/telemetry-$1.jsonl" >"$results/telemetry-$1-latest.jsonl" || true; }
steady_clock() {
  python3 - "$results/telemetry-clock-warmup.jsonl" <<'PY'
import json, sys
from pathlib import Path
rows=[]
for line in Path(sys.argv[1]).read_text().splitlines():
    try:
        d=json.loads(line); value=d['metric']['gpu_data'][0]['clock']['gfx_0']['clk']['value']
        if isinstance(value, (int,float)): rows.append(float(value))
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError): pass
if len(rows) < 3:
    raise SystemExit('clock gate needs three valid amd-smi GFX samples')
tail=rows[-3:]
# An active, settled clock is defined by evidence, rather than a hard-coded
# boost target: at least 1 GHz and a <=5% spread across three samples.
if min(tail) < 1000 or (max(tail)-min(tail))/max(tail) > .05:
    raise SystemExit(f'clock did not settle: last_three_mhz={tail}')
print(json.dumps({'samples_mhz':rows,'settled_last_three_mhz':tail,'criterion':'three samples >=1000 MHz with <=5% spread'}))
PY
}
run_mode() {
  local mode=$1 start_ns end_ns
  start_sampler "$mode"
  start_ns=$(date --iso-8601=ns)
  HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" "$binary" --mode "$mode" \
    --memory-peak-gbps "${HW_MB_MEMORY_PEAK_GBPS:?set from official/observed source}" \
    --bf16-peak-tflops "${HW_MB_BF16_PEAK_TFLOPS:?set from official source}" \
    --fp8-peak-tflops "${HW_MB_FP8_PEAK_TFLOPS:?set from official source}" \
    --output "$results/${mode}.jsonl"
  end_ns=$(date --iso-8601=ns)
  stop_sampler; copy_latest "$mode"
  printf '%s\t%s\t%s\n' "$mode" "$start_ns" "$end_ns" >>"$results/runtime.tsv"
}
: >"$results/runtime.tsv"
printf 'phase\tstarted_at\tfinished_at\n' >"$results/runtime.tsv"
start_sampler clock-warmup
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" "$binary" --mode clock-warmup --clock-warmup-seconds 12 \
  --memory-peak-gbps "${HW_MB_MEMORY_PEAK_GBPS:?set from official/observed source}" \
  --bf16-peak-tflops "${HW_MB_BF16_PEAK_TFLOPS:?set from official source}" \
  --fp8-peak-tflops "${HW_MB_FP8_PEAK_TFLOPS:?set from official source}" --output "$results/clock-warmup.jsonl"
stop_sampler; copy_latest clock-warmup
steady_clock >"$results/clock-steady.json"
run_mode validate
run_mode bandwidth
run_mode gemm
