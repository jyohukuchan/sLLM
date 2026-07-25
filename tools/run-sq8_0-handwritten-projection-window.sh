#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0

# One reversible, R9700-only SQ8_0 handwritten projection measurement window.
# This runner is intentionally limited to the private prototype binaries.  It
# never changes a served-model manifest, a campaign, an authorization, a
# release, a systemd unit, or the default CK dispatch.

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    printf '%s\n' "usage: $0 RESULT_DIR COMPONENT_BIN SERVING_BIN" >&2
    exit 2
fi

result_dir="$(realpath -m "$1")"
component_bin="$(realpath -e "$2")"
serving_bin="$(realpath -e "$3")"
repo_root="$(git rev-parse --show-toplevel)"
artifact_dir=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact
package_dir=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/package
prompt_u32le="$repo_root/tests/fixtures/sq8-serving-v0.1/oracles/vllm-source-v0.1/inputs/raw-p0512.u32le"

# AMD SMI numbers this card as GPU 2.  ROCr/HIP orders the exposed GPU agents
# differently, so the prior R9700 evidence maps its single-token visibility
# selection to HIP token 1.  Every workload also validates gfx1201 itself.
r9700_amd_smi_gpu=2
r9700_hip_visibility=1

if [[ -z ${ULLM_SUDO_PASSWORD:-} ]]; then
    printf '%s\n' 'ULLM_SUDO_PASSWORD is required for the approved stop/start only' >&2
    exit 2
fi
if [[ ! -x "$component_bin" || ! -x "$serving_bin" ]]; then
    printf '%s\n' 'component and serving binaries must be executable' >&2
    exit 2
fi
if [[ ! -d "$artifact_dir" || ! -d "$package_dir" || ! -f "$prompt_u32le" ]]; then
    printf '%s\n' 'canonical SQ8_0 artifact, package, or raw-p0512 fixture is unavailable' >&2
    exit 2
fi
if [[ -e "$result_dir/service/window-start.txt" ]]; then
    printf '%s\n' "refusing to overwrite existing window evidence: $result_dir/service/window-start.txt" >&2
    exit 2
fi

mkdir -p "$result_dir"/{component,serving/ck,serving/handwritten,full-model-multistep,service,telemetry,preflight}

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
    env -u HSA_VISIBLE_DEVICES \
        -u ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE \
        -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE \
        -u ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE \
        -u GPU_DUMP_CODE_OBJECT \
        HIP_VISIBLE_DEVICES="$r9700_hip_visibility" "${kernel_guards[@]}" "$@"
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
        # AMD SMI represents an empty list on this driver as one sentinel
        # string rather than an empty JSON array. Only a process-info object
        # denotes a resident process; any other unexpected value is unsafe.
        if isinstance(info, dict):
            raise SystemExit(1)
        if info != "No running processes detected":
            raise SystemExit(1)
PY
}

wait_for_r9700_idle() {
    local relative=$1
    local attempt
    for attempt in $(seq 1 30); do
        amd-smi process --gpu "$r9700_amd_smi_gpu" --general --json \
            >"$result_dir/$relative" 2>&1 || true
        if json_has_no_r9700_processes "$result_dir/$relative"; then
            return 0
        fi
        sleep 1
    done
    return 1
}

json_field_is_true() {
    python3 - "$1" "$2" <<'PY'
import json
import sys

path, field = sys.argv[1:]
with open(path, encoding="utf-8") as source:
    value = json.load(source)
for component in field.split("."):
    if not isinstance(value, dict):
        raise SystemExit(1)
    value = value.get(component)
raise SystemExit(0 if value is True else 1)
PY
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
    local original_status=$?
    trap - EXIT
    stop_telemetry_watcher
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
                echo 'initial-start-failed; inspect isolated R9700 residue before reset-failed'
                amd-smi process --gpu "$r9700_amd_smi_gpu" --general --json || true
                echo 'attempt=reset-failed-and-second-start'
            } >>"$result_dir/service/restore.txt" 2>&1
            sudo_service systemctl reset-failed ullm-openai.service >>"$result_dir/service/restore.txt" 2>&1 || true
            if sudo_service systemctl start ullm-openai.service >>"$result_dir/service/restore.txt" 2>&1 \
                && systemctl is-active --quiet ullm-openai.service; then
                restored=1
            fi
        fi
        systemctl show ullm-openai.service -p ActiveState -p SubState -p Result -p MainPID -p NRestarts \
            >>"$result_dir/service/restore.txt" 2>&1 || true
    fi
    capture_telemetry after-restore || true
    capture_service_state after-restore || true
    date --iso-8601=seconds >"$result_dir/service/window-end.txt"
    if [[ $restored -eq 0 && $service_was_active -eq 1 ]]; then
        original_status=1
    fi
    exit "$original_status"
}

trap restore_service EXIT

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
if [[ "$(systemctl is-active llama-qwen35-udq4.service || true)" != inactive ]]; then
    printf '%s\n' 'llama-qwen35-udq4.service is not inactive; refusing R9700 work' >&2
    exit 1
fi
if [[ "$(systemctl is-enabled llama-qwen35-udq4.service 2>/dev/null || true)" != disabled ]]; then
    printf '%s\n' 'llama-qwen35-udq4.service is not disabled; refusing R9700 work' >&2
    exit 1
fi
if [[ "$(systemctl is-active gdm3.service || true)" != inactive ]]; then
    printf '%s\n' 'gdm3.service is not inactive; refusing R9700 work' >&2
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

# Record power/clock/temperature/throttle status once per second throughout
# the isolated process sequence. The raw watch output is retained verbatim.
amd-smi metric --gpu "$r9700_amd_smi_gpu" --temperature --clock --power --watch 1 --watch_time 1800 --json \
    >"$result_dir/telemetry/during-window.watch.txt" 2>&1 &
telemetry_pid=$!

# First numerical component gate: no HIP event timing is performed in gate mode.
capture_optional component/gate.stdout.txt \
    run_r9700 "$component_bin" --output "$result_dir/component/gate.json" --mode gate --warmups 0 --repeats 1 --device 0
capture_telemetry after-component-gate

# Full-model control and candidate runs are numerical captures only. The
# candidate remains opt-in through the private CLI flag; the normal CK default
# is exercised by the first command.
capture_optional serving/ck/stdout.txt \
    run_r9700 "$serving_bin" \
        --artifact "$artifact_dir" --package "$package_dir" \
        --prompt-token-ids-u32le "$prompt_u32le" --max-new-tokens 4 \
        --prefill-mode m8-chunk8 \
        --decode-oracle-capture-dir "$result_dir/serving/ck/decode" \
        --result-json "$result_dir/serving/ck/result.json"
capture_telemetry after-serving-ck

capture_optional serving/handwritten/stdout.txt \
    run_r9700 "$serving_bin" \
        --artifact "$artifact_dir" --package "$package_dir" \
        --prompt-token-ids-u32le "$prompt_u32le" --max-new-tokens 4 \
        --prefill-mode m8-chunk8 \
        --handwritten-wmma-projection-prototype \
        --decode-oracle-capture-dir "$result_dir/serving/handwritten/decode" \
        --result-json "$result_dir/serving/handwritten/result.json"
capture_telemetry after-serving-handwritten

full_model_pass=0
if [[ -f "$result_dir/serving/ck/result.json" && -f "$result_dir/serving/handwritten/result.json" ]]; then
    capture_optional full-model-multistep/stdout.txt \
        python3 "$repo_root/tools/compare-sq8_0-handwritten-projection-gate.py" \
            --ck-result "$result_dir/serving/ck/result.json" \
            --handwritten-result "$result_dir/serving/handwritten/result.json" \
            --min-decode-steps 2 \
            --output "$result_dir/full-model-multistep/gate.json"
    if [[ -f "$result_dir/full-model-multistep/gate.json" ]] \
        && json_field_is_true "$result_dir/full-model-multistep/gate.json" passed; then
        full_model_pass=1
    fi
else
    printf '%s\n' 'not run: one or both full-model result JSON files were not created' \
        >"$result_dir/full-model-multistep/not-run.txt"
fi

# The CK-only control is measured after both frozen numerical gates, even if
# the candidate fails. This supplies the requested baseline without assigning
# any candidate speed claim to a numerically rejected route.
capture_optional component/baseline.stdout.txt \
    run_r9700 "$component_bin" --output "$result_dir/component/ck-baseline.json" \
        --mode baseline --warmups 10 --repeats 80 --device 0
capture_telemetry after-ck-baseline

component_pass=0
if [[ -f "$result_dir/component/gate.json" ]] \
    && json_field_is_true "$result_dir/component/gate.json" numeric_gate.passed; then
    component_pass=1
fi

if [[ $component_pass -eq 1 && $full_model_pass -eq 1 ]]; then
    capture_optional component/measure.stdout.txt \
        run_r9700 "$component_bin" --output "$result_dir/component/measure.json" \
            --mode measure --warmups 10 --repeats 80 --device 0
    capture_telemetry after-handwritten-measure
else
    {
        echo 'not run: candidate event timing is forbidden until both frozen numerical gates pass'
        printf 'component_gate_pass=%s\n' "$component_pass"
        printf 'full_model_multistep_pass=%s\n' "$full_model_pass"
    } >"$result_dir/component/handwritten-measurement-not-run.txt"
fi

date --iso-8601=seconds >"$result_dir/service/isolation-complete.txt"
