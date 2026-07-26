#!/usr/bin/env bash
# Reproducible, offline R9700 evidence window for the Qwen3.5 MoE AQ4_0 loader.
#
# This script deliberately fails rather than taking ownership when the caller
# has not already confirmed that /run/ullm/r9700.lock has no holder or waiter.
# It does not mutate services, manifests, or active model selection.
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/../../../.." && pwd)
result_dir="$repo_root/benchmarks/results/2026-07-27/qwen35-moe-loader-wiring-v0.1"
cd "$repo_root"

# The main worktree can be concurrently edited by other tasks.  Keep the
# reproducible default, but allow a clean, separately built candidate to be
# injected for an isolated evidence window without copying it into target/.
baseline_binary=${ULLM_AQ4_BASELINE_PROBE_BIN:-"$repo_root/target/release/ullm-qwen35-aq4-baseline-probe"}
moe_binary=${ULLM_QWEN35_MOE_GENERATE_BIN:-"$repo_root/target/release/ullm-qwen35-moe-aq4-generate"}
attempt_tag=${ULLM_QWEN35_MOE_ATTEMPT_TAG:-attempt2}
if ! [[ "$attempt_tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
    printf '%s\n' "invalid evidence attempt tag: $attempt_tag" >&2
    exit 64
fi
baseline_result_prefix="$result_dir/qwen35-9b-baseline-probe-$attempt_tag"
moe_result_prefix="$result_dir/qwen35-moe-$attempt_tag"
if ! test -x "$baseline_binary"; then
    printf '%s\n' "baseline probe binary is not executable: $baseline_binary" >&2
    exit 64
fi
if ! test -x "$moe_binary"; then
    printf '%s\n' "MoE generator binary is not executable: $moe_binary" >&2
    exit 64
fi

baseline_ids='30380,171104,152367,98047,195323,143382,171713,200076,108906,107704,118515,241794,100410,79346,87276,33414,196111,166665,49894,243345,123088,205178,4960,174801,66630,183870,14519,18911,44134,195559,81407,246665,61658,91154,164480,193470,154890,136673,18728,187868,124,40569,35836,194520,118170,113409,210442,239970,6570,94132,74486,28062,129404,72166,20630,246440,139688,92857,8693,168946,193569,112603,168793,190640,145935,237973,43349,39886,87402,127644,113264,70640,85213,42477,156555,131226,161009,236206,218915,148972,47585,39802,77124,126641,4398,59031,156230,103146,236102,218355,218544,156311,19112,19705,159477,143062,54636,181922,167681,29101,63832,87733,186531,167051,230759,205717,158274,78568,132472,102192,244260,70413,70403,138892,218466,176318,428,27801,113003,58315,222479,143587,218628,62835,50247,185038,6744,10918'
prompt_ids='760,6511,314,9338,369'
export repo_root result_dir baseline_ids prompt_ids
export baseline_binary moe_binary attempt_tag baseline_result_prefix moe_result_prefix

# The physical R9700 is AMD SMI GPU 2, but ROCm orders it as HIP ordinal 1:
# gfx1030 (BDF 43:00.0), gfx1201 (BDF 47:00.0), gfx1030 (BDF 03:00.0).
# Do not substitute the AMD SMI index here; HIP ordinal 2 is the forbidden V620.
r9700_hip_ordinal=1
export r9700_hip_ordinal

capture_inner_preflight() {
    printf 'captured_at='; date --iso-8601=seconds
    systemctl show ullm-openai.service -p ActiveState -p NRestarts
    amd-smi process --gpu 2 --json 2>&1 || true
    amd-smi metric --gpu 2 --json 2>&1 || true
}

run_window() {
    printf 'acquired_at='; date --iso-8601=seconds
    capture_inner_preflight > "${moe_result_prefix}-inner-preflight.txt"
    # Keep the offline loader on the same guarded operation contract as the
    # existing Qwen3.5-9B AQ4_0 resident worker.  These are the exact names in
    # QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV, not a guessed subset.
    local -a aq4_guard_env=(
        ULLM_REQUIRE_HIP_AQ4_KERNEL=1
        ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1
        ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1
        ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL=1
        ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_GROUP8_KERNEL=1
        ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_KERNEL=1
        ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_GROUP8_KERNEL=1
        ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_RAGGED_M_KERNEL=1
        ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_GROUP8_RAGGED_M_KERNEL=1
        ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL=1
        ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL=1
        ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL=1
        ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL=1
        ULLM_REQUIRE_HIP_ADD_KERNEL=1
        ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1
        ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1
        ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL=1
        ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL=1
        ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1
        ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1
        ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL=1
        ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL=1
        ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL=1
        ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_WMMA_KERNEL=1
        ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1
        ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL=1
        ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1
        ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL=1
        ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL=1
        ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL=1
        ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1
        ULLM_REQUIRE_HIP_ROPE_KERNEL=1
        ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL=1
        ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL=1
        ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1
        ULLM_REQUIRE_HIP_TOP1_KERNEL=1
    )
    printf '%s\n' "${aq4_guard_env[@]}" > "$result_dir/qwen35-aq4-guard-contract-$attempt_tag.txt"
    if systemctl is-active --quiet ullm-openai.service; then
        printf '%s\n' 'ullm-openai.service became active after lock acquisition' >&2
        return 73
    fi
    if amd-smi process --gpu 2 --json | rg -q '"pid"|"process_id"'; then
        printf '%s\n' 'R9700 has a process after lock acquisition' >&2
        return 74
    fi
    edge_temp=$(amd-smi metric --gpu 2 --json | jq -r '.gpu_data[0].temperature.edge.value')
    if test "$edge_temp" -gt 45; then
        printf '%s\n' "R9700 edge temperature ${edge_temp}C is above 45C" >&2
        return 75
    fi

    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
        HIP_VISIBLE_DEVICES="$r9700_hip_ordinal" ULLM_HIP_VISIBLE_DEVICES="$r9700_hip_ordinal" \
        "${aq4_guard_env[@]}" \
        "$baseline_binary" \
        --token-ids "$baseline_ids" \
        --device-index 1 \
        --context-length 256 \
        --expected-top1 220 \
        > "${baseline_result_prefix}.json" \
        2> "${baseline_result_prefix}.stderr"

    : > "${moe_result_prefix}-vram-telemetry.jsonl"
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
        HIP_VISIBLE_DEVICES="$r9700_hip_ordinal" ULLM_HIP_VISIBLE_DEVICES="$r9700_hip_ordinal" \
        "${aq4_guard_env[@]}" \
        ULLM_KV_CACHE_DTYPE=f16 \
        ULLM_REQUIRE_HIP_TYPED_PAGED_DECODE_KERNEL=1 \
        ULLM_REQUIRE_HIP_TYPED_PAGED_DECODE_SPLIT_KERNEL=1 \
        ULLM_REQUIRE_HIP_TYPED_PAGED_KV_WRITE_KERNEL=1 \
        "$moe_binary" \
        --prompt-token-ids "$prompt_ids" \
        --new-tokens 8 \
        --device-index 1 \
        --context-length 262144 \
        --kv-block-size 256 \
        --hold-seconds 30 \
        --output "${moe_result_prefix}-generate.json" \
        > "${moe_result_prefix}-generate.stdout" \
        2> "${moe_result_prefix}-generate.stderr" &
    moe_pid=$!
    while kill -0 "$moe_pid" 2>/dev/null; do
        captured_at=$(date --iso-8601=seconds)
        amd-smi metric --gpu 2 --json \
            | jq --arg captured_at "$captured_at" '. + {captured_at: $captured_at}' \
            >> "${moe_result_prefix}-vram-telemetry.jsonl" || true
        sleep 1
    done
    set +e
    wait "$moe_pid"
    moe_status=$?
    set -e
    printf '%s\n' "$moe_status" > "${moe_result_prefix}-generate.exit-status.txt"
    if test "$moe_status" -ne 0; then
        return "$moe_status"
    fi
    printf 'released_at='; date --iso-8601=seconds
}

set +e
if lslocks -o PID,COMMAND,TYPE,MODE,PATH,BLOCKER | rg -q 'r9700\.lock'; then
    printf '%s\n' 'a prior R9700 lock holder or waiter appeared before acquisition' >&2
    printf '%s\n' '72' > "${moe_result_prefix}-window.exit-status.txt"
    exit 72
fi
flock -n /run/ullm/r9700.lock bash -ceu "$(declare -f capture_inner_preflight run_window); run_window"
window_status=$?
set -e
printf '%s\n' "$window_status" > "${moe_result_prefix}-window.exit-status.txt"
exit "$window_status"
