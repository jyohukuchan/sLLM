#!/usr/bin/env bash
# R9700-only dispatch attribution for the current promoted Gemma4 path.
set -euo pipefail

repo=/home/homelab1/coding-local/ultimateLLM/uLLM-project
out="$repo/benchmarks/results/2026-07-28/gemma4-prefill-fresh-attribution-ej-v1.0"
raw="$out/raw"
telemetry="$out/telemetry"
model_dir=/home/homelab1/datapool/ai_models/safetensors/gemma-4-E2B
amd_smi=/opt/rocm/bin/amd-smi
binary="$repo/target/release/ullm-gemma4-resident"
guard_env=(
    ULLM_REQUIRE_HIP_ADD_KERNEL=1
    ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1
    ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1
    ULLM_REQUIRE_HIP_GEMMA_PROPORTIONAL_ROPE_KERNEL=1
    ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1
    ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1
    ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1
    ULLM_REQUIRE_HIP_ROPE_KERNEL=1
)

mkdir -p "$raw" "$telemetry" "$out/preflight"
exec 9>/run/ullm/r9700.lock
flock -n 9
printf 'lock acquired %s\n' "$(date --iso-8601=seconds)" > "$out/preflight/lock-acquired.txt"
"$amd_smi" list > "$out/preflight/amd-smi-list.txt"
sha256sum "$repo/runtime/src/ullm_runtime.cpp" "$repo/runtime/src/ullm_runtime_api.inc" \
    "$repo/runtime/src/ullm_runtime_hiprtc_sources.inc" > "$out/preflight/runtime-source-sentinels.sha256"
printf '%s\n' "${guard_env[@]}" > "$out/preflight/promoted-gemma4-guard-contract.txt"

(cd "$repo" && cargo build --release -p ullm-engine --bin ullm-gemma4-resident) > "$raw/build.log" 2>&1
sha256sum "$binary" > "$out/preflight/binary.sha256"

sample_temperature() {
    local label=$1 pid=$2
    while kill -0 "$pid" 2>/dev/null; do
        printf '%s ' "$(date --iso-8601=seconds)" >> "$telemetry/${label}.txt"
        "$amd_smi" metric --gpu 2 --temperature >> "$telemetry/${label}.txt" 2>&1 || true
        sleep 5
    done
}

run_case() {
    local n=$1
    local trace="$raw/rocprof-n${n}"
    "$amd_smi" metric --gpu 2 --temperature > "$telemetry/n${n}-before.txt" 2>&1
    env HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 "${guard_env[@]}" \
        "$binary" --model-dir "$model_dir" --output "$raw/attention-profile-n${n}.json" \
        --mode attention-profile --benchmark-prompt-tokens "$n" --benchmark-prompt-token-id 2 \
        --cooldown-hotspot-c 87 --cooldown-timeout-seconds 900 > "$raw/attention-profile-n${n}.stdout" 2> "$raw/attention-profile-n${n}.stderr" &
    local pid=$!
    sample_temperature "n${n}-plain" "$pid" &
    local sampler=$!
    wait "$pid"
    wait "$sampler" || true
    "$amd_smi" metric --gpu 2 --temperature > "$telemetry/n${n}-after-plain.txt" 2>&1

    "$amd_smi" metric --gpu 2 --temperature > "$telemetry/n${n}-before-rocprof.txt" 2>&1
    rocprofv3 --kernel-trace --output-directory "$trace" --output-format csv -- \
        env HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 "${guard_env[@]}" \
        "$binary" --model-dir "$model_dir" --output "$raw/rocprof-profile-n${n}.json" \
        --mode attention-profile --benchmark-prompt-tokens "$n" --benchmark-prompt-token-id 2 \
        --cooldown-hotspot-c 87 --cooldown-timeout-seconds 900 > "$raw/rocprof-n${n}.stdout" 2> "$raw/rocprof-n${n}.stderr" &
    pid=$!
    sample_temperature "n${n}-rocprof" "$pid" &
    sampler=$!
    wait "$pid"
    wait "$sampler" || true
    "$amd_smi" metric --gpu 2 --temperature > "$telemetry/n${n}-after-rocprof.txt" 2>&1
}

run_case 512
run_case 2048
printf 'complete %s\n' "$(date --iso-8601=seconds)" > "$out/COMPLETE"
