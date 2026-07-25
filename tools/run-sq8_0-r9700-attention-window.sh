#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run one reversible R9700-only SQ8_0 attention measurement window.  This
# script never changes a served-model manifest, a release, an authorization,
# or a systemd unit.  It only stops and restores the already-active worker so
# isolated HIP processes can own the R9700 for the duration of the window.

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 RESULT_DIR PROTOTYPE_BIN SPLIT_BENCH_BIN SERVING_BIN" >&2
    exit 2
fi

result_dir=$(realpath -m "$1")
prototype_bin=$(realpath -e "$2")
split_bench_bin=$(realpath -e "$3")
serving_bin=$(realpath -e "$4")
repo_root=$(git rev-parse --show-toplevel)
artifact_dir=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact
package_dir=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/package
prompt_u32le="$repo_root/tests/fixtures/sq8-serving-v0.1/oracles/vllm-source-v0.1/inputs/raw-p0512.u32le"
rocprof=/opt/rocm/bin/rocprofv3

if [[ -z ${ULLM_SUDO_PASSWORD:-} ]]; then
    echo "ULLM_SUDO_PASSWORD is required only for the approved service stop/start" >&2
    exit 2
fi
if [[ ! -x "$prototype_bin" || ! -x "$split_bench_bin" || ! -x "$serving_bin" ]]; then
    echo "one or more required binaries are not executable" >&2
    exit 2
fi
if [[ ! -f "$prompt_u32le" || ! -d "$artifact_dir" || ! -d "$package_dir" ]]; then
    echo "canonical SQ8_0 measurement inputs are unavailable" >&2
    exit 2
fi
if [[ -e "$result_dir/service/window-start.txt" ]]; then
    echo "refusing to overwrite an existing window record: $result_dir/service/window-start.txt" >&2
    exit 2
fi

mkdir -p "$result_dir"/{service,telemetry,pmc,flash2,decode,preflight}

capture() {
    local relative=$1
    shift
    "$@" >"$result_dir/$relative" 2>&1
}

capture_optional() {
    local relative=$1
    shift
    if "$@" >"$result_dir/$relative" 2>&1; then
        printf '0\n' >"$result_dir/$relative.exit-status"
    else
        local status=$?
        printf '%s\n' "$status" >"$result_dir/$relative.exit-status"
    fi
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

run_r9700() {
    env -u ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE \
        -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE \
        HIP_VISIBLE_DEVICES=1 "${kernel_guards[@]}" "$@"
}

run_r9700_staged_flash2() {
    env -u ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE \
        HIP_VISIBLE_DEVICES=1 ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE=1 \
        "${kernel_guards[@]}" "$@"
}

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
                echo 'initial-start-failed: checking R9700 isolated-process residue before reset-failed'
                amd-smi process --gpu 2 --general --json || true
                pgrep -af 'sq8_0_r9700_attention_prototype|sq8_0_paged_decode_split_bench|sq8_ck_serving' || true
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

mkdir -p "$result_dir/pmc"/{probe-raw-sq,probe-raw-gl2c,probe-derived,target-derived} \
    "$result_dir/flash2"/{standalone,serving-baseline,serving-staged} \
    "$result_dir/decode"

# These optional probes distinguish primitive counter availability from
# derived-metric evaluation.  A failed profiler invocation is evidence, not a
# reason to skip numerical or timing gates.
capture_optional "pmc/probe-raw-sq/rocprof.stdout.txt" \
    env HIP_VISIBLE_DEVICES=1 "$rocprof" --output-directory "$result_dir/pmc/probe-raw-sq/data" \
        --output-format csv --kernel-trace --kernel-include-regex '^ullm_sq8_0_pmc_probe_kernel$' \
        --pmc SQ_INSTS_VALU,SQ_WAVES -- "$prototype_bin" --mode pmc-probe \
        --output-dir "$result_dir/pmc/probe-raw-sq/app"
capture_optional "pmc/probe-raw-gl2c/rocprof.stdout.txt" \
    env HIP_VISIBLE_DEVICES=1 "$rocprof" --output-directory "$result_dir/pmc/probe-raw-gl2c/data" \
        --output-format csv --kernel-trace --kernel-include-regex '^ullm_sq8_0_pmc_probe_kernel$' \
        --pmc GL2C_EA_RDREQ_32B,GL2C_EA_RDREQ_64B,GL2C_EA_RDREQ_128B -- "$prototype_bin" --mode pmc-probe \
        --output-dir "$result_dir/pmc/probe-raw-gl2c/app"
capture_optional "pmc/probe-derived/rocprof.stdout.txt" \
    env HIP_VISIBLE_DEVICES=1 "$rocprof" --output-directory "$result_dir/pmc/probe-derived/data" \
        --output-format csv --kernel-trace --kernel-include-regex '^ullm_sq8_0_pmc_probe_kernel$' \
        --pmc FETCH_SIZE,VALUInsts,Wavefronts -- "$prototype_bin" --mode pmc-probe \
        --output-dir "$result_dir/pmc/probe-derived/app"

# First run the separate Flash2 symbols.  This establishes the staged QK,
# max, and sum numerical gate before any selected runtime body is used.
capture_required "flash2/standalone/stdout.txt" \
    env HIP_VISIBLE_DEVICES=1 "$prototype_bin" --mode flash2 \
        --output-dir "$result_dir/flash2/standalone" --warmups 5 --repeats 40

# The full-model baseline and staged runs use the canonical artifact and the
# vLLM-source raw-p0512 input fixture.  Each captures the first M=128 oracle
# output and synchronized per-unit prefill timing, without profiler overhead.
mkdir -p "$result_dir/flash2/serving-baseline/code-objects" \
    "$result_dir/flash2/serving-staged/code-objects"
(
    cd "$result_dir/flash2/serving-baseline/code-objects"
    run_r9700 env GPU_DUMP_CODE_OBJECT=1 "$serving_bin" \
        --artifact "$artifact_dir" --package "$package_dir" \
        --prompt-token-ids-u32le "$prompt_u32le" --max-new-tokens 1 \
        --prefill-mode m128-chunk128 \
        --oracle-capture-dir "$result_dir/flash2/serving-baseline/oracle" \
        --result-json "$result_dir/flash2/serving-baseline/result.json"
) >"$result_dir/flash2/serving-baseline/stdout.txt" 2>&1
printf '0\n' >"$result_dir/flash2/serving-baseline/stdout.txt.exit-status"
(
    cd "$result_dir/flash2/serving-staged/code-objects"
    run_r9700_staged_flash2 env GPU_DUMP_CODE_OBJECT=1 "$serving_bin" \
        --artifact "$artifact_dir" --package "$package_dir" \
        --prompt-token-ids-u32le "$prompt_u32le" --max-new-tokens 1 \
        --prefill-mode m128-chunk128 \
        --oracle-capture-dir "$result_dir/flash2/serving-staged/oracle" \
        --result-json "$result_dir/flash2/serving-staged/result.json"
) >"$result_dir/flash2/serving-staged/stdout.txt" 2>&1
printf '0\n' >"$result_dir/flash2/serving-staged/stdout.txt.exit-status"

# This is a scoped actual-body PMC pass.  Its timings are intentionally not
# used as throughput; the preceding two runs are the unprofiled timings.
capture_optional "pmc/target-derived/rocprof.stdout.txt" \
    env -u ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE \
        -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE \
        HIP_VISIBLE_DEVICES=1 "${kernel_guards[@]}" "$rocprof" \
        --output-directory "$result_dir/pmc/target-derived/data" --output-format csv --kernel-trace \
        --kernel-include-regex '^ullm_cached_prefix_attn_f32_flash2_kernel$' \
        --pmc FETCH_SIZE,VALUInsts,Wavefronts -- "$serving_bin" \
        --artifact "$artifact_dir" --package "$package_dir" \
        --prompt-token-ids-u32le "$prompt_u32le" --max-new-tokens 1 \
        --prefill-mode m128-chunk128 \
        --result-json "$result_dir/pmc/target-derived/runner-result.json"

capture_required "decode/split-bench.stdout.txt" \
    env HIP_VISIBLE_DEVICES=1 "$split_bench_bin" --output "$result_dir/decode/split-bench.json" \
        --cache-len 1036 --warmups 10 --repeats 100
capture_telemetry after-measurement
date --iso-8601=seconds >"$result_dir/service/window-measurements-complete.txt"
