#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0

# One reversible, R9700-only numerical window for the private handwritten
# SQ8_0 projection-contract diagnostic. This never times or promotes a
# candidate, changes a served-model manifest, consumes an authorization, or
# changes default dispatch.

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    printf '%s\n' "usage: $0 RESULT_DIR DIAGNOSTIC_BIN" >&2
    exit 2
fi

result_dir="$(realpath -m "$1")"
diagnostic_bin="$(realpath -e "$2")"
repo_root="$(git rev-parse --show-toplevel)"
artifact_dir=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact
package_dir=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/package
prompt_u32le="$repo_root/tests/fixtures/sq8-serving-v0.1/oracles/vllm-source-v0.1/inputs/raw-p0512.u32le"
r9700_amd_smi_gpu=2
r9700_hip_visibility=1

if [[ -z ${ULLM_SUDO_PASSWORD:-} ]]; then
    printf '%s\n' 'ULLM_SUDO_PASSWORD is required for the approved stop/start only' >&2
    exit 2
fi
if [[ ! -x "$diagnostic_bin" || ! -d "$artifact_dir" || ! -d "$package_dir" || ! -f "$prompt_u32le" ]]; then
    printf '%s\n' 'diagnostic binary, canonical artifact, package, or raw-p0512 fixture is unavailable' >&2
    exit 2
fi
if [[ -e "$result_dir" ]]; then
    printf '%s\n' "refusing to overwrite existing evidence directory: $result_dir" >&2
    exit 2
fi

mkdir -p "$result_dir"/{service,telemetry,preflight}

sudo_service() {
    printf '%s\n' "$ULLM_SUDO_PASSWORD" | sudo -S -p '' "$@"
}

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
    } >"$result_dir/service/$name.txt"
}

capture_telemetry() {
    local name=$1
    amd-smi metric --gpu "$r9700_amd_smi_gpu" --temperature --clock --power --json \
        >"$result_dir/telemetry/$name.json" 2>&1
}

json_has_no_r9700_processes() {
    python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    data = json.load(source)
for record in data:
    if record.get("gpu") != 2:
        continue
    for entry in record.get("process_list", []):
        info = entry.get("process_info")
        # This driver represents an empty process list as a sentinel string.
        # A process-info object or any unknown representation is unsafe.
        if isinstance(info, dict) or info != "No running processes detected":
            raise SystemExit(1)
PY
}

wait_for_r9700_idle() {
    local relative=$1
    local attempt
    # The worker's HIP context can remain visible briefly after systemd has
    # reported the unit inactive. Give the driver 90 seconds before declaring
    # the isolation window unusable; starting a kernel while it is present is
    # never permitted.
    for attempt in $(seq 1 90); do
        amd-smi process --gpu "$r9700_amd_smi_gpu" --general --json \
            >"$result_dir/$relative" 2>&1 || true
        if json_has_no_r9700_processes "$result_dir/$relative"; then
            return 0
        fi
        sleep 1
    done
    return 1
}

run_r9700() {
    env -u HSA_VISIBLE_DEVICES \
        -u ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE \
        -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE \
        -u ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE \
        HIP_VISIBLE_DEVICES="$r9700_hip_visibility" \
        ULLM_REQUIRE_HIP_ADD_KERNEL=1 \
        ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1 \
        ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1 \
        ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL=1 \
        ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1 \
        ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1 \
        ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL=1 \
        ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1 \
        ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 \
        ULLM_REQUIRE_HIP_ROPE_KERNEL=1 \
        ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1 \
        "$@"
}

service_was_active=0
restored=0
telemetry_pid=''

stop_telemetry_watcher() {
    if [[ -n "$telemetry_pid" ]]; then
        kill "$telemetry_pid" 2>/dev/null || true
        wait "$telemetry_pid" 2>/dev/null || true
        telemetry_pid=''
    fi
}

restore_service() {
    local original_status=${1:-0}
    trap - EXIT
    set +e
    stop_telemetry_watcher
    if [[ $service_was_active -eq 1 && $restored -eq 0 ]]; then
        {
            date --iso-8601=seconds
            echo 'attempt=single-start'
        } >"$result_dir/service/restore.txt"
        if sudo_service systemctl start ullm-openai.service >>"$result_dir/service/restore.txt" 2>&1 \
            && systemctl is-active --quiet ullm-openai.service; then
            restored=1
        fi
        systemctl show ullm-openai.service -p ActiveState -p SubState -p Result -p MainPID -p NRestarts \
            >>"$result_dir/service/restore.txt" 2>&1 || true
    fi
    capture_telemetry after-restore || true
    capture_service_state after-restore || true
    date --iso-8601=seconds >"$result_dir/service/window-end.txt"
    if [[ $service_was_active -eq 1 && $restored -eq 0 ]]; then
        original_status=1
    fi
    return "$original_status"
}

trap 'restore_service "$?"' EXIT

date --iso-8601=seconds >"$result_dir/service/window-start.txt"
capture_service_state before-stop
capture preflight/amd-smi-list.json amd-smi list --json
capture preflight/r9700-static.json amd-smi static --gpu "$r9700_amd_smi_gpu" --asic --bus --json
capture preflight/r9700-process-before.json amd-smi process --gpu "$r9700_amd_smi_gpu" --general --json
capture_telemetry before-stop

if [[ "$(systemctl is-active ullm-openai.service || true)" != active ]]; then
    printf '%s\n' 'ullm-openai.service is not active; refusing an undefined restore state' >&2
    exit 1
fi
service_was_active=1
if [[ "$(systemctl is-active llama-qwen35-udq4.service || true)" != inactive ]] \
    || [[ "$(systemctl is-enabled llama-qwen35-udq4.service 2>/dev/null || true)" != disabled ]] \
    || [[ "$(systemctl is-active gdm3.service || true)" != inactive ]]; then
    printf '%s\n' 'required service isolation preconditions are not met' >&2
    exit 1
fi

capture_optional service/stop.txt sudo_service systemctl stop ullm-openai.service
capture_service_state after-stop
if [[ "$(systemctl is-active ullm-openai.service || true)" != inactive ]]; then
    printf '%s\n' 'ullm-openai.service did not reach inactive after stop request' >&2
    exit 1
fi
if ! wait_for_r9700_idle service/r9700-process-after-stop.json; then
    printf '%s\n' 'R9700 still has a process after service stop; not starting isolated work' >&2
    exit 1
fi
capture_telemetry after-stop

amd-smi metric --gpu "$r9700_amd_smi_gpu" --temperature --clock --power --watch 1 --watch_time 1800 --json \
    >"$result_dir/telemetry/during-window.watch.txt" 2>&1 &
telemetry_pid=$!

capture_optional diagnostic.stdout.txt run_r9700 "$diagnostic_bin" \
    --artifact "$artifact_dir" --package "$package_dir" \
    --prompt-token-ids-u32le "$prompt_u32le" --output "$result_dir/diagnostic"
capture_telemetry after-diagnostic
date --iso-8601=seconds >"$result_dir/service/isolation-complete.txt"
