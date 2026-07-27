#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0
# One-command standalone build/run wrapper. It never controls services.
set -euo pipefail
usage() { cat >&2 <<'EOF'
usage: run-hw-microbench-rdna4-cdna3.sh --arch gfx1201|gfx942 --results-dir /absolute/dir [--build-only]

The default run uses HIP_VISIBLE_DEVICES=0, 5 warmups, 11 median samples and
10 inner launches/sample. It records amd-smi metrics before/after (when
available).  The 256 MiB-per-vector STREAM working set is intentionally larger
than both targets' cache hierarchy.
EOF
}
arch= results= build_only=0
while [[ $# -gt 0 ]]; do case "$1" in --arch) arch=${2:-};shift 2;;--results-dir)results=${2:-};shift 2;;--build-only)build_only=1;shift;;-h|--help)usage;exit 0;;*)usage;exit 2;;esac;done
[[ $arch == gfx1201 || $arch == gfx942 ]] && [[ $results == /* ]] || { usage; exit 2; }
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P); rocm=${ROCM_PATH:-/opt/rocm}; mkdir -p -- "$results"
binary="$results/hw-microbench-$arch"; "$rocm/bin/hipcc" -std=c++20 -O3 --offload-arch="$arch" "$repo/tools/hw-microbench-rdna4-cdna3.hip.cpp" -o "$binary" >"$results/build.log" 2>&1
"$repo/tools/audit-hw-microbench-isa.sh" --repo "$repo" --arch "$arch" --output-dir "$results/isa" >"$results/isa.log" 2>&1
if (( build_only )); then exit 0; fi
run_mode() {
  local mode=$1 start end
  if command -v amd-smi >/dev/null; then amd-smi metric --json >"$results/telemetry-${mode}-before.json" 2>&1 || true; fi
  start=$(date +%s)
  HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" "$binary" --mode "$mode" \
    --memory-peak-gbps "${HW_MB_MEMORY_PEAK_GBPS:?set from official/observed source}" \
    --bf16-peak-tflops "${HW_MB_BF16_PEAK_TFLOPS:?set from official source}" \
    --fp8-peak-tflops "${HW_MB_FP8_PEAK_TFLOPS:?set from official source}" \
    --output "$results/${mode}.jsonl"
  end=$(date +%s)
  if command -v amd-smi >/dev/null; then amd-smi metric --json >"$results/telemetry-${mode}-after.json" 2>&1 || true; fi
  printf '%s_seconds=%s\n' "$mode" "$((end-start))" >>"$results/runtime.txt"
}
: >"$results/runtime.txt"
run_mode validate
run_mode bandwidth
run_mode gemm
