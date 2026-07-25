#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# One reversible, R9700-only full-model M=1 paged-decode timing window for
# SQ8_0.  The direct legacy route remains the control; 128/256/512 select the
# existing split API only through a test-only process environment variable.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 RESULT_DIR SERVING_BIN" >&2
    exit 2
fi

result_dir=$(realpath -m "$1")
serving_bin=$(realpath -e "$2")
repo_root=$(git rev-parse --show-toplevel)
artifact_dir=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact
package_dir=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/package
prompt_u32le="$repo_root/tests/fixtures/sq8-serving-v0.1/oracles/vllm-source-v0.1/inputs/raw-p0512.u32le"

if [[ -z ${ULLM_SUDO_PASSWORD:-} ]]; then
    echo "ULLM_SUDO_PASSWORD is required only for the approved service stop/start" >&2
    exit 2
fi
if [[ ! -x "$serving_bin" || ! -d "$artifact_dir" || ! -d "$package_dir" || ! -f "$prompt_u32le" ]]; then
    echo "required SQ8_0 full-model measurement input is unavailable" >&2
    exit 2
fi
if [[ -e "$result_dir/service/window-start.txt" ]]; then
    echo "refusing to overwrite existing window record: $result_dir/service/window-start.txt" >&2
    exit 2
fi

mkdir -p "$result_dir"/{service,telemetry,preflight,cases}

capture() {
    local relative=$1
    shift
    "$@" >"$result_dir/$relative" 2>&1
}

capture_required() {
    local relative=$1
    shift
    if "$@" >"$result_dir/$relative" 2>&1; then
        printf '0\n' >"$result_dir/$relative.exit-status"
    else
        local status=$?
        printf '%s\n' "$status" >"$result_dir/$relative.exit-status"
        return "$status"
    fi
}

sudo_service() {
    printf '%s\n' "$ULLM_SUDO_PASSWORD" | sudo -S -p '' "$@"
}

kernel_guards=(
    ULLM_REQUIRE_HIP_ADD_KERNEL=1
    ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1
    ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1
    ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL=1
    ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1
    ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1
    ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL=1
    ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1
    ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1
    ULLM_REQUIRE_HIP_ROPE_KERNEL=1
    ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1
)

capture_service_state() {
    {
        date --iso-8601=seconds
        systemctl show ullm-openai.service \
            -p ActiveState -p SubState -p Result -p MainPID -p NRestarts \
            -p StartLimitBurst -p StartLimitIntervalUSec
        printf 'llama-qwen35-udq4.active='; systemctl is-active llama-qwen35-udq4.service || true
        printf 'llama-qwen35-udq4.enabled='; systemctl is-enabled llama-qwen35-udq4.service || true
        printf 'gdm3.active='; systemctl is-active gdm3.service || true
    } >"$result_dir/service/service-state.txt"
}

capture_telemetry() {
    local name=$1
    amd-smi metric --gpu 2 --temperature --clock --power --json >"$result_dir/telemetry/$name.json" 2>&1
}

service_was_active=0
restored=0
restore_service() {
    local original_status=$?
    trap - EXIT
    if [[ $service_was_active -eq 1 && $restored -eq 0 ]]; then
        {
            date --iso-8601=seconds
            echo 'attempt=initial-start'
        } >"$result_dir/service/restore.txt"
        if sudo_service systemctl start ullm-openai.service >>"$result_dir/service/restore.txt" 2>&1 \
            && systemctl is-active --quiet ullm-openai.service; then
            restored=1
        else
            {
                echo 'initial-start-failed: checking isolated process residue before reset-failed'
                amd-smi process --gpu 2 --general --json || true
                pgrep -af 'sq8_ck_serving|sq8_0_r9700' || true
                echo 'attempt=reset-failed'
            } >>"$result_dir/service/restore.txt" 2>&1
            sudo_service systemctl reset-failed ullm-openai.service >>"$result_dir/service/restore.txt" 2>&1 || true
            if sudo_service systemctl start ullm-openai.service >>"$result_dir/service/restore.txt" 2>&1 \
                && systemctl is-active --quiet ullm-openai.service; then
                restored=1
            fi
        fi
        systemctl show ullm-openai.service -p ActiveState -p SubState -p Result -p MainPID -p NRestarts \
            >>"$result_dir/service/restore.txt" 2>&1 || true
        if [[ $restored -eq 0 ]]; then
            original_status=1
        fi
    fi
    capture_telemetry after-restore || true
    exit "$original_status"
}

trap restore_service EXIT

date --iso-8601=seconds >"$result_dir/service/window-start.txt"
capture_service_state
capture "preflight/amd-smi-list.json" amd-smi list --json
capture "preflight/r9700-static.json" amd-smi static --gpu 2 --asic --bus --json
capture "preflight/r9700-process-before.json" amd-smi process --gpu 2 --general --json
capture_telemetry before-stop

if [[ $(systemctl is-active ullm-openai.service) == active ]]; then
    service_was_active=1
fi
if [[ $(systemctl is-active llama-qwen35-udq4.service) != inactive ]]; then
    echo 'llama-qwen35-udq4.service is not inactive; refusing GPU measurement' >&2
    exit 1
fi
if [[ $(systemctl is-enabled llama-qwen35-udq4.service 2>/dev/null || true) != disabled ]]; then
    echo 'llama-qwen35-udq4.service is not disabled; refusing GPU measurement' >&2
    exit 1
fi
if [[ $(systemctl is-active gdm3.service) != inactive ]]; then
    echo 'gdm3.service is not inactive; refusing GPU measurement' >&2
    exit 1
fi

if [[ $service_was_active -eq 1 ]]; then
    capture_required "service/stop.txt" sudo_service systemctl stop ullm-openai.service
fi
capture_service_state
capture "service/r9700-process-after-stop.json" amd-smi process --gpu 2 --general --json
capture_telemetry after-stop

run_case() {
    local name=$1
    local tile=${2:-}
    local -a extra_env=(
        -u ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE
        -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE
        -u ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE
    )
    if [[ -n "$tile" ]]; then
        extra_env+=("ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE=$tile")
    fi
    capture_required "cases/$name.stdout.txt" \
        env "${extra_env[@]}" HIP_VISIBLE_DEVICES=1 "${kernel_guards[@]}" "$serving_bin" \
            --artifact "$artifact_dir" --package "$package_dir" \
            --prompt-token-ids-u32le "$prompt_u32le" --max-new-tokens 8 \
            --prefill-mode m128-chunk128 --test-only-ignore-eos \
            --result-json "$result_dir/cases/$name.result.json"
}

# Each request has one final M=128 prefill step and seven ensuing M=1 decode
# steps.  The runner records synchronized_seconds for every generated step;
# analysis uses generated_index 1..7 only.
run_case direct
run_case tile128 128
run_case tile256 256
run_case tile512 512

capture_telemetry after-measurement
date --iso-8601=seconds >"$result_dir/service/window-measurements-complete.txt"
