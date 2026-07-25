#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# One reversible R9700-only window for diagnosing and fixing the SQ8_0
# paged-decode source-tile route.  It never changes an active-model manifest,
# power setting, systemd unit, campaign, or authorization.

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 RESULT_DIR UNFIXED_SERVING_BIN FIXED_SERVING_BIN SPLIT_BENCH_BIN" >&2
    exit 2
fi

result_dir=$(realpath -m "$1")
unfixed_serving_bin=$(realpath -e "$2")
fixed_serving_bin=$(realpath -e "$3")
split_bench_bin=$(realpath -e "$4")
repo_root=$(git rev-parse --show-toplevel)
criteria_source="$repo_root/benchmarks/results/2026-07-26/sq8_0-paged-decode-tile-gate/gate-criteria.json"
artifact_dir=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact
package_dir=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/package
input_dir="$repo_root/tests/fixtures/sq8-serving-v0.1/oracles/vllm-source-v0.1/inputs"
r9700_bdf=0000:47:00.0
r9700_sysfs="/sys/bus/pci/devices/$r9700_bdf"
expected_criteria_sha256=645df099030dcf3beca1289e0cc848f0f9c53c1725866896e06848631d962978

if [[ -z ${ULLM_SUDO_PASSWORD:-} ]]; then
    echo "ULLM_SUDO_PASSWORD is required for the approved service stop/start window" >&2
    exit 2
fi
for path in "$unfixed_serving_bin" "$fixed_serving_bin" "$split_bench_bin" "$criteria_source" \
    "$input_dir/raw-p0128.u32le" "$input_dir/raw-p0512.u32le"; do
    if [[ ! -f "$path" ]]; then
        echo "required input is unavailable: $path" >&2
        exit 2
    fi
done
for path in "$unfixed_serving_bin" "$fixed_serving_bin" "$split_bench_bin"; do
    if [[ ! -x "$path" ]]; then
        echo "required executable is not executable: $path" >&2
        exit 2
    fi
done
if [[ ! -d "$artifact_dir" || ! -d "$package_dir" ]]; then
    echo "SQ8_0 artifact or package directory is unavailable" >&2
    exit 2
fi
if [[ -e "$result_dir/service/window-start.txt" || -e "$result_dir/summary.json" ]]; then
    echo "refusing to overwrite an existing window or gate summary" >&2
    exit 2
fi
if [[ $(sha256sum "$criteria_source" | awk '{print $1}') != "$expected_criteria_sha256" ]]; then
    echo "the frozen tile-gate criterion hash differs from the required baseline" >&2
    exit 2
fi

mkdir -p "$result_dir"/{cases,diagnostics/api-sweep,diagnostics/unfixed-kv,directories,input,performance,preflight,service,telemetry}
cp -- "$criteria_source" "$result_dir/gate-criteria.json"
sha256sum "$unfixed_serving_bin" "$fixed_serving_bin" "$split_bench_bin" \
    >"$result_dir/preflight/binaries.sha256"
{
    printf 'unfixed_serving_bin=%s\n' "$unfixed_serving_bin"
    printf 'fixed_serving_bin=%s\n' "$fixed_serving_bin"
    printf 'split_bench_bin=%s\n' "$split_bench_bin"
    printf 'criteria_source=%s\n' "$criteria_source"
    printf 'criteria_sha256=%s\n' "$expected_criteria_sha256"
} >"$result_dir/preflight/inputs.env"

capture_optional() {
    local relative=$1
    shift
    local status=0
    if "$@" >"$result_dir/$relative" 2>&1; then
        status=0
    else
        status=$?
    fi
    printf '%s\n' "$status" >"$result_dir/$relative.exit-status"
    return 0
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

append_event() {
    printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >>"$result_dir/service/events.tsv"
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
    local name=$1
    {
        date --iso-8601=seconds
        systemctl show ullm-openai.service \
            -p ActiveState -p SubState -p Result -p MainPID -p NRestarts \
            -p StartLimitBurst -p StartLimitIntervalUSec
        printf 'llama-qwen35-udq4.active='; systemctl is-active llama-qwen35-udq4.service || true
        printf 'llama-qwen35-udq4.enabled='; systemctl is-enabled llama-qwen35-udq4.service || true
        printf 'gdm3.active='; systemctl is-active gdm3.service || true
    } >"$result_dir/service/$name.txt" 2>&1
}

capture_gpu_metrics_raw() {
    local name=$1
    {
        date --iso-8601=seconds
        printf 'bdf=%s\n' "$r9700_bdf"
        if [[ -r "$r9700_sysfs/gpu_metrics" ]]; then
            stat --format='size=%s' "$r9700_sysfs/gpu_metrics" || true
            printf '%s\n' '-- first-120-bytes-hex --'
            od -An -v -tx1 -N 120 "$r9700_sysfs/gpu_metrics" || true
            printf '%s\n' '-- u32-le-offset-68-throttle-status-current-v1.3-layout --'
            od -An -j 68 -N 4 -tu4 "$r9700_sysfs/gpu_metrics" || true
            printf '%s\n' '-- u64-le-offset-112-independent-throttle-status-current-v1.3-layout --'
            od -An -j 112 -N 8 -tu8 "$r9700_sysfs/gpu_metrics" || true
        else
            printf '%s\n' 'gpu_metrics-unreadable-or-absent'
        fi
    } >"$result_dir/telemetry/$name.gpu-metrics-raw.txt" 2>&1
}

capture_telemetry() {
    local name=$1
    capture_optional "telemetry/$name.metrics.json" \
        amd-smi metric --gpu 2 --temperature --clock --power --violation --json
    capture_optional "telemetry/$name.static-limit.json" amd-smi static --gpu 2 --limit --json
    capture_optional "telemetry/$name.process.json" amd-smi process --gpu 2 --general --json
    capture_gpu_metrics_raw "$name"
}

telemetry_pid=
start_telemetry_monitor() {
    (
        local sample=0
        while :; do
            capture_optional "telemetry/during-$(printf '%04d' "$sample").metrics.json" \
                amd-smi metric --gpu 2 --temperature --clock --power --violation --json
            capture_gpu_metrics_raw "during-$(printf '%04d' "$sample")"
            sample=$((sample + 1))
            sleep 1
        done
    ) &
    telemetry_pid=$!
}

stop_telemetry_monitor() {
    if [[ -n ${telemetry_pid:-} ]]; then
        kill "$telemetry_pid" 2>/dev/null || true
        wait "$telemetry_pid" 2>/dev/null || true
        telemetry_pid=
    fi
}

service_was_active=0
restored=0
restore_service() {
    local original_status=$?
    trap - EXIT
    stop_telemetry_monitor
    if [[ $service_was_active -eq 1 && $restored -eq 0 ]]; then
        {
            date --iso-8601=seconds
            echo 'attempt=initial-start'
        } >"$result_dir/service/restore.txt"
        if sudo_service systemctl start ullm-openai.service >>"$result_dir/service/restore.txt" 2>&1 \
            && systemctl is-active --quiet ullm-openai.service; then
            restored=1
            append_event 'service-restored-initial-start'
        else
            {
                echo 'initial-start-failed: checking isolated process residue before reset-failed'
                amd-smi process --gpu 2 --general --json || true
                pgrep -af 'sq8_ck_serving|sq8_0_paged_decode_split_bench' || true
                echo 'attempt=reset-failed'
            } >>"$result_dir/service/restore.txt" 2>&1
            sudo_service systemctl reset-failed ullm-openai.service >>"$result_dir/service/restore.txt" 2>&1 || true
            if sudo_service systemctl start ullm-openai.service >>"$result_dir/service/restore.txt" 2>&1 \
                && systemctl is-active --quiet ullm-openai.service; then
                restored=1
                append_event 'service-restored-after-reset-failed'
            fi
        fi
        if [[ $restored -eq 0 ]]; then
            original_status=1
            append_event 'service-restore-failed'
        fi
    fi
    capture_service_state after-restore || true
    capture_telemetry after-restore || true
    exit "$original_status"
}

trap restore_service EXIT

date --iso-8601=seconds >"$result_dir/service/window-start.txt"
printf 'timestamp\tevent\n' >"$result_dir/service/events.tsv"
append_event 'window-start'
capture_service_state pre-stop
capture_optional 'preflight/amd-smi-list.json' amd-smi list --json
capture_optional 'preflight/r9700-static.json' amd-smi static --gpu 2 --asic --bus --json
capture_optional 'preflight/r9700-process-before.json' amd-smi process --gpu 2 --general --json
capture_optional 'preflight/gate-criteria.sha256' sha256sum "$result_dir/gate-criteria.json"
capture_telemetry before-stop

if ! rg -q 'gfx1201' "$result_dir/preflight/r9700-static.json" \
    || ! rg -q '47:00\.0' "$result_dir/preflight/r9700-static.json"; then
    echo 'GPU 2 is not the required R9700 gfx1201 at 0000:47:00.0' >&2
    exit 1
fi
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
append_event 'preflight-passed-r9700-only-llama-inactive-gdm-inactive'

if [[ $service_was_active -eq 1 ]]; then
    capture_required 'service/stop.txt' sudo_service systemctl stop ullm-openai.service
    append_event 'service-stopped'
else
    append_event 'service-was-not-active-no-stop-issued'
fi
capture_service_state after-stop
capture_optional 'service/r9700-process-after-stop.json' amd-smi process --gpu 2 --general --json
capture_telemetry after-stop

make_prefix() {
    local source=$1
    local count=$2
    local destination=$3
    if [[ -e "$destination" ]]; then
        echo "refusing to overwrite prefix input: $destination" >&2
        exit 1
    fi
    dd if="$source" of="$destination" bs=4 count="$count" status=none
    [[ $(stat -c%s "$destination") -eq $((count * 4)) ]]
}

token_ids_csv() {
    od -An -v -tu4 "$1" | tr -s '[:space:]' '\n' | sed '/^$/d' | paste -sd, -
}

p0128="$input_dir/raw-p0128.u32le"
p0512="$input_dir/raw-p0512.u32le"
p0128_prefix127="$result_dir/input/raw-p0128-prefix127.u32le"
p0512_prefix511="$result_dir/input/raw-p0512-prefix511.u32le"
make_prefix "$p0128" 127 "$p0128_prefix127"
make_prefix "$p0512" 511 "$p0512_prefix511"
{
    printf 'label\tpath\tbytes\tsha256\n'
    for path in "$p0128_prefix127" "$p0128" "$p0512_prefix511" "$p0512"; do
        printf '%s\t%s\t%s\t%s\n' "$(basename "$path" .u32le)" \
            "${path#$result_dir/}" "$(stat -c%s "$path")" "$(sha256sum "$path" | awk '{print $1}')"
    done
} >"$result_dir/input/manifest.tsv"

run_unfixed_kv_route() {
    local route=$1
    local tile=${2:-}
    local output="$result_dir/diagnostics/unfixed-kv/$route"
    local -a route_env=(
        -u ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE
        -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE
        -u ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE
        -u ULLM_SQ8_0_PAGED_DECODE_DIRECT
    )
    if [[ -n "$tile" ]]; then
        route_env+=("ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE=$tile")
    fi
    mkdir -p "$output"
    capture_required "diagnostics/unfixed-kv/$route/stdout.txt" \
        env "${route_env[@]}" HIP_VISIBLE_DEVICES=1 "${kernel_guards[@]}" "$unfixed_serving_bin" \
            --artifact "$artifact_dir" --package "$package_dir" \
            --prompt-token-ids-u32le "$p0128" --max-new-tokens 2 \
            --prefill-mode m128-chunk128 --record-generated-timing \
            --decode-oracle-capture-dir "$output/oracle" \
            --kv-cache-prefix-capture-dir "$output/kv-prefix" \
            --result-json "$output/result.json"
    append_event "unfixed-kv-route-complete-$route"
}

run_fixed_group_route() {
    local group=$1
    local primary=$2
    local secondary=$3
    local route=$4
    local tile=${5:-}
    local output="$result_dir/cases/$route/$group"
    local -a route_env=(
        -u ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE
        -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE
        -u ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE
        -u ULLM_SQ8_0_PAGED_DECODE_DIRECT
    )
    if [[ -n "$tile" ]]; then
        route_env+=("ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE=$tile")
    fi
    mkdir -p "$output"
    capture_required "cases/$route/$group/stdout.txt" \
        env "${route_env[@]}" HIP_VISIBLE_DEVICES=1 "${kernel_guards[@]}" "$fixed_serving_bin" \
            --artifact "$artifact_dir" --package "$package_dir" \
            --prompt-token-ids-u32le "$primary" --max-new-tokens 4 \
            --second-prompt-token-ids "$(token_ids_csv "$secondary")" --second-max-new-tokens 4 \
            --prefill-mode m128-chunk128 --record-generated-timing \
            --decode-oracle-capture-dir "$output/oracle" \
            --result-json "$output/result.json"
    append_event "fixed-gate-route-complete-$route-$group"
}

run_fixed_performance_route() {
    local route=$1
    local tile=${2:-}
    local output="$result_dir/performance/$route"
    local -a route_env=(
        -u ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE
        -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE
        -u ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE
        -u ULLM_SQ8_0_PAGED_DECODE_DIRECT
    )
    if [[ -n "$tile" ]]; then
        route_env+=("ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE=$tile")
    fi
    mkdir -p "$output"
    capture_required "performance/$route/stdout.txt" \
        env "${route_env[@]}" HIP_VISIBLE_DEVICES=1 "${kernel_guards[@]}" "$fixed_serving_bin" \
            --artifact "$artifact_dir" --package "$package_dir" \
            --prompt-token-ids-u32le "$p0512" --max-new-tokens 8 \
            --prefill-mode m128-chunk128 --record-generated-timing \
            --result-json "$output/result.json"
    append_event "fixed-performance-route-complete-$route"
}

start_telemetry_monitor
append_event 'telemetry-monitor-started'

for cache_len in 128 129 130 256 257 512 513 514; do
    label=$(printf 'c%04d' "$cache_len")
    capture_required "diagnostics/api-sweep/$label.stdout.txt" \
        env HIP_VISIBLE_DEVICES=1 \
            ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1 \
            ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL=1 \
            "$split_bench_bin" --output "$result_dir/diagnostics/api-sweep/$label.json" \
            --cache-len "$cache_len" --warmups 1 --repeats 3
    append_event "api-sweep-complete-$label"
done

run_unfixed_kv_route direct
run_unfixed_kv_route tile128 128
capture_required 'diagnostics/unfixed-kv/summary.stdout.txt' \
    python3 "$repo_root/tools/evaluate-sq8_0-paged-decode-kv-diagnostic.py" \
        --direct-result "$result_dir/diagnostics/unfixed-kv/direct/result.json" \
        --candidate-result "$result_dir/diagnostics/unfixed-kv/tile128/result.json" \
        --output "$result_dir/diagnostics/unfixed-kv/summary.json"
append_event 'unfixed-kv-diagnostic-complete'

run_fixed_group_route p0128-boundary-tail "$p0128_prefix127" "$p0128" direct
run_fixed_group_route p0128-boundary-tail "$p0128_prefix127" "$p0128" tile128 128
run_fixed_group_route p0128-boundary-tail "$p0128_prefix127" "$p0128" tile256 256
run_fixed_group_route p0512-boundary-tail "$p0512_prefix511" "$p0512" direct
run_fixed_group_route p0512-boundary-tail "$p0512_prefix511" "$p0512" tile128 128
run_fixed_group_route p0512-boundary-tail "$p0512_prefix511" "$p0512" tile256 256
capture_optional 'gate-evaluator.stdout.txt' \
    python3 "$repo_root/tools/evaluate-sq8_0-paged-decode-tile-gate.py" --result-dir "$result_dir"
append_event 'fixed-gate-evaluated'

run_fixed_performance_route direct
run_fixed_performance_route tile128 128
run_fixed_performance_route tile256 256
capture_required 'performance-summary.stdout.txt' \
    python3 "$repo_root/tools/summarize-sq8_0-paged-decode-tile-performance.py" \
        --result-dir "$result_dir"
append_event 'fixed-performance-summarized'

stop_telemetry_monitor
append_event 'telemetry-monitor-stopped'
capture_telemetry after-measurement
date --iso-8601=seconds >"$result_dir/service/window-measurements-complete.txt"
append_event 'window-measurements-complete'
