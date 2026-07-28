#!/usr/bin/env bash
# Durable R9700-only benchmark window. Invoked by the root wrapper so it can
# stop/restart ullm-openai, but all compilation and result writes run as
# homelab1. This script makes no source changes.
set -euo pipefail

repo=/home/homelab1/coding-local/ultimateLLM/uLLM-project
baseline=/tmp/ullm-gemma4-pre-residency-dn
out="$repo/benchmarks/results/2026-07-28/gemma4-prefill-definitive-eh-v1.0"
raw="$out/raw"
telemetry="$out/telemetry"
model_dir=/home/homelab1/datapool/ai_models/safetensors/gemma-4-E2B
gguf=/home/homelab1/datapool/ai_models/gguf/gemma-4-E2B-BF16.gguf
llama=/home/homelab1/llama.cpp-src/build-rdna4/bin/llama-bench
amd_smi=/opt/rocm/bin/amd-smi
baseline_ids='30380,171104,152367,98047,195323,143382,171713,200076,108906,107704,118515,241794,100410,79346,87276,33414,196111,166665,49894,243345,123088,205178,4960,174801,66630,183870,14519,18911,44134,195559,81407,246665,61658,91154,164480,193470,154890,136673,18728,187868,124,40569,35836,194520,118170,113409,210442,239970,6570,94132,74486,28062,129404,72166,20630,246440,139688,92857,8693,168946,193569,112603,168793,190640,145935,237973,43349,39886,87402,127644,113264,70640,85213,42477,156555,131226,161009,236206,218915,148972,47585,39802,77124,126641,4398,59031,156230,103146,236102,218355,218544,156311,19112,19705,159477,143062,54636,181922,167681,29101,63832,87733,186531,167051,230759,205717,158274,78568,132472,102192,244260,70413,70403,138892,218466,176318,428,27801,113003,58315,222479,143587,218628,62835,50247,185038,6744,10918'
aq4_guard_env=(
    ULLM_REQUIRE_HIP_AQ4_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1
    ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL=1
    ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_GROUP8_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_KERNEL=1
    ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_GROUP8_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_RAGGED_M_KERNEL=1
    ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_GROUP8_RAGGED_M_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL=1
    ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL=1
    ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL=1 ULLM_REQUIRE_HIP_ADD_KERNEL=1
    ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1 ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1
    ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL=1
    ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1
    ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL=1
    ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_WMMA_KERNEL=1
    ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL=1
    ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1 ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL=1
    ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL=1
    ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 ULLM_REQUIRE_HIP_ROPE_KERNEL=1
    ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL=1 ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL=1
    ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1 ULLM_REQUIRE_HIP_TOP1_KERNEL=1
)
baseline_gemma4_guard_env=(
    ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1
    ULLM_REQUIRE_HIP_F32_ROW_KERNEL=1
    ULLM_REQUIRE_HIP_PAGED_ATTENTION_KERNEL=1
    ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1
    ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1
)
promoted_gemma4_guard_env=(
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

timestamp() { date --iso-8601=seconds; }
metric() { "$amd_smi" metric --gpu 2 --temperature 2>&1; }

sample_temperature() {
    local label=$1
    local pid=$2
    local file="$telemetry/${label}.jsonl"
    while kill -0 "$pid" 2>/dev/null; do
        printf '{"timestamp":"%s","amd_smi_metric":%s}\n' "$(timestamp)" "$(metric | jq -Rsc .)" >> "$file"
        sleep 5
    done
    printf '{"timestamp":"%s","amd_smi_metric":%s}\n' "$(timestamp)" "$(metric | jq -Rsc .)" >> "$file"
}

run_timed() {
    local label=$1
    shift
    metric > "$telemetry/${label}.before.txt"
    "$@" > "$raw/${label}.stdout" 2> "$raw/${label}.stderr" &
    local pid=$!
    sample_temperature "$label" "$pid" &
    local sampler=$!
    local status=0
    wait "$pid" || status=$?
    wait "$sampler" || true
    metric > "$telemetry/${label}.after.txt"
    return "$status"
}

build_endpoint() {
    local name=$1
    local tree=$2
    (
        cd "$tree"
        cargo clean
        cargo build --release -p ullm-engine --bin ullm-gemma4-resident
    ) > "$raw/${name}.build.log" 2>&1
    if ! grep -q 'Compiling ullm-runtime-sys' "$raw/${name}.build.log"; then
        echo "native ullm-runtime-sys did not recompile" >&2
        return 1
    fi
    find "$tree/target/release/build" -path '*ullm-runtime-sys*/out/*' -type f -printf '%TY-%Tm-%TdT%TT%TZ %s %p\n' \
        | sort > "$raw/${name}.native-runtime-artifacts.txt"
    sha256sum "$tree/target/release/ullm-gemma4-resident" > "$raw/${name}.binary.sha256"
}

run_ullm_endpoint() {
    local name=$1
    local tree=$2
    local -a guard_env
    case "$name" in
        baseline) guard_env=("${baseline_gemma4_guard_env[@]}") ;;
        promoted) guard_env=("${promoted_gemma4_guard_env[@]}") ;;
        *) echo "unknown Gemma4 endpoint: $name" >&2; return 2 ;;
    esac
    build_endpoint "$name" "$tree"
    local binary="$tree/target/release/ullm-gemma4-resident"
    for n in 128 512 2048; do
        run_timed "${name}-n${n}" env \
            HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 \
            "${guard_env[@]}" \
            "$binary" --model-dir "$model_dir" --output "$raw/${name}-n${n}.json" \
            --mode benchmark --benchmark-repeats 5 --benchmark-prompt-tokens "$n" \
            --benchmark-decode-tokens 128 --benchmark-prompt-token-id 2 \
            --cooldown-hotspot-c 87 --cooldown-timeout-seconds 900
    done
}

run_llama() {
    for n in 128 512 2048; do
        run_timed "llama-n${n}" env HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 \
            "$llama" -m "$gguf" -ngl 999 -p "$n" -n 0 -r 5 -b "$n" -ub "$n" \
            -ctk f32 -ctv f32 -fa off -o json
        run_timed "llama-decode-n${n}" env HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 \
            "$llama" -m "$gguf" -ngl 999 -p 0 -n 128 -r 5 -b "$n" -ub "$n" \
            -ctk f32 -ctv f32 -fa off -o json
        mv "$raw/llama-n${n}.stdout" "$raw/llama-n${n}.json"
        mv "$raw/llama-decode-n${n}.stdout" "$raw/llama-decode-n${n}.json"
    done
}

exec 9>/run/ullm/r9700.lock
flock -n 9
printf 'lock acquired %s\n' "$(timestamp)" > "$out/preflight/lock-acquired.txt"
"$amd_smi" list > "$out/preflight/amd-smi-list.txt"
"$amd_smi" metric --gpu 2 --temperature > "$out/preflight/r9700-temperature-start.txt"
sha256sum "$repo/runtime/src/ullm_runtime.cpp" "$repo/runtime/src/ullm_runtime_api.inc" \
    "$repo/runtime/src/ullm_runtime_hiprtc_sources.inc" > "$out/preflight/runtime-source-sentinels-start.sha256"

if [[ "${ULLM_EH_RESUME:-0}" == 1 ]]; then
    printf '%s\n' "${promoted_gemma4_guard_env[@]}" > "$raw/gemma4-promoted-guard-contract.txt"
else
    (
        cd "$repo"
        cargo clean
        cargo build --release -p ullm-engine --bin ullm-qwen35-aq4-baseline-probe
    ) > "$raw/qwen35-aq4-probe.build.log" 2>&1
    printf '%s\n' "${aq4_guard_env[@]}" > "$raw/qwen35-aq4-guard-contract.txt"
    run_timed qwen35-aq4-start env HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 "${aq4_guard_env[@]}" \
        "$repo/target/release/ullm-qwen35-aq4-baseline-probe" --token-ids "$baseline_ids" \
        --package /home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package \
        --device-index 1 --context-length 128 --expected-top1 220
    mv "$raw/qwen35-aq4-start.stdout" "$raw/qwen35-aq4-start.json"
    run_ullm_endpoint baseline "$baseline"
fi
printf '%s\n' "${promoted_gemma4_guard_env[@]}" > "$raw/gemma4-promoted-guard-contract.txt"
run_ullm_endpoint promoted "$repo"
run_llama

(
    cd "$repo"
    cargo build --release -p ullm-engine --bin ullm-qwen35-aq4-baseline-probe
) > "$raw/qwen35-aq4-probe.end-build.log" 2>&1

run_timed qwen35-aq4-end env HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 "${aq4_guard_env[@]}" \
    "$repo/target/release/ullm-qwen35-aq4-baseline-probe" --token-ids "$baseline_ids" \
    --package /home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package \
    --device-index 1 --context-length 128 --expected-top1 220
mv "$raw/qwen35-aq4-end.stdout" "$raw/qwen35-aq4-end.json"

sha256sum "$repo/runtime/src/ullm_runtime.cpp" "$repo/runtime/src/ullm_runtime_api.inc" \
    "$repo/runtime/src/ullm_runtime_hiprtc_sources.inc" > "$out/preflight/runtime-source-sentinels-end.sha256"
"$amd_smi" metric --gpu 2 --temperature > "$out/preflight/r9700-temperature-end.txt"
printf 'complete %s\n' "$(timestamp)" > "$out/COMPLETE"
