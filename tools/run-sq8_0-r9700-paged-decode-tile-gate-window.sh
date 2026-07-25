#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# One reversible R9700-only service window for the frozen SQ8_0 paged-decode
# numerical gate.  It captures actual M=1 feedback decode vectors for direct,
# tile128, and tile256; it does not activate a served-model manifest or alter
# any persistent GPU setting.

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
input_dir="$repo_root/tests/fixtures/sq8-serving-v0.1/oracles/vllm-source-v0.1/inputs"
r9700_bdf=0000:47:00.0
r9700_sysfs="/sys/bus/pci/devices/$r9700_bdf"

if [[ -z ${ULLM_SUDO_PASSWORD:-} ]]; then
    echo "ULLM_SUDO_PASSWORD is required for the approved service stop/start window" >&2
    exit 2
fi
if [[ ! -x "$serving_bin" || ! -d "$artifact_dir" || ! -d "$package_dir" ]]; then
    echo "serving binary, artifact, or package is unavailable" >&2
    exit 2
fi
for source in "$input_dir/raw-p0128.u32le" "$input_dir/raw-p0512.u32le"; do
    if [[ ! -f "$source" ]]; then
        echo "required real-prompt token fixture is unavailable: $source" >&2
        exit 2
    fi
done
if [[ ! -f "$result_dir/gate-criteria.json" ]]; then
    echo "frozen gate-criteria.json must exist before the service window" >&2
    exit 2
fi
if [[ -e "$result_dir/service/window-start.txt" || -e "$result_dir/summary.json" ]]; then
    echo "refusing to overwrite an existing window or gate summary" >&2
    exit 2
fi

mkdir -p "$result_dir"/{cases,input,preflight,service,telemetry}

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

capture_sysfs_telemetry() {
    local name=$1
    {
        date --iso-8601=seconds
        printf '%s\n' "bdf=$r9700_bdf"
        for entry in \
            gpu_metrics \
            power_dpm_force_performance_level \
            pp_power_profile_mode \
            pp_features \
            thermal_throttling_logging; do
            printf '\n-- %s --\n' "$entry"
            if [[ -e "$r9700_sysfs/$entry" ]]; then
                if [[ $entry == gpu_metrics ]]; then
                    sudo_service od -An -v -tx1 "$r9700_sysfs/$entry" || true
                else
                    sudo_service cat "$r9700_sysfs/$entry" || true
                fi
            else
                printf 'absent\n'
            fi
        done
        if [[ -d "$r9700_sysfs/hwmon" ]]; then
            while IFS= read -r path; do
                printf '\n-- %s --\n' "${path#$r9700_sysfs/}"
                sudo_service cat "$path" || true
            done < <(find "$r9700_sysfs/hwmon" -maxdepth 2 -type f -print | sort)
        fi
    } >"$result_dir/telemetry/$name.sysfs.txt" 2>&1
}

capture_gpu_metrics_raw() {
    # Preserve the firmware metric table alongside amd-smi's decoded status.
    # In particular, byte offset 68 is throttle_status in the current
    # format_revision=1/content_revision=3 table.  Recording both the header
    # and raw bytes keeps this evidence usable if the driver/SMU layout differs
    # on a later run.
    local name=$1
    {
        date --iso-8601=seconds
        printf '%s\n' "bdf=$r9700_bdf"
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
    capture_sysfs_telemetry "$name"
    capture_gpu_metrics_raw "$name"
}

capture_kernel_events() {
    local name=$1
    {
        date --iso-8601=seconds
        if ! sudo_service dmesg --color=never | rg -i 'amdgpu.*(0000:47:00\.0|PTL|thrott)'; then
            true
        fi
    } >"$result_dir/telemetry/$name.kernel-events.txt" 2>&1
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
                pgrep -af 'sq8_ck_serving|sq8_0_r9700' || true
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
    capture_kernel_events after-restore || true
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
capture_kernel_events before-stop

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
capture_kernel_events after-stop

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

run_group_route() {
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
        env "${route_env[@]}" HIP_VISIBLE_DEVICES=1 "${kernel_guards[@]}" "$serving_bin" \
            --artifact "$artifact_dir" --package "$package_dir" \
            --prompt-token-ids-u32le "$primary" --max-new-tokens 4 \
            --second-prompt-token-ids "$(token_ids_csv "$secondary")" --second-max-new-tokens 4 \
            --prefill-mode m128-chunk128 --record-generated-timing \
            --decode-oracle-capture-dir "$output/oracle" \
            --result-json "$output/result.json"
    append_event "case-complete-$route-$group"
}

start_telemetry_monitor
append_event 'telemetry-monitor-started'
run_group_route p0128-boundary-tail "$p0128_prefix127" "$p0128" direct
run_group_route p0128-boundary-tail "$p0128_prefix127" "$p0128" tile128 128
run_group_route p0128-boundary-tail "$p0128_prefix127" "$p0128" tile256 256
run_group_route p0512-boundary-tail "$p0512_prefix511" "$p0512" direct
run_group_route p0512-boundary-tail "$p0512_prefix511" "$p0512" tile128 128
run_group_route p0512-boundary-tail "$p0512_prefix511" "$p0512" tile256 256
stop_telemetry_monitor
append_event 'telemetry-monitor-stopped'
capture_telemetry after-measurement
capture_kernel_events after-measurement
date --iso-8601=seconds >"$result_dir/service/window-measurements-complete.txt"
append_event 'window-measurements-complete'
