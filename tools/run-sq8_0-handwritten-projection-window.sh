#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0

# One reversible, R9700-only SQ8_0 handwritten projection measurement window.
# This runner is intentionally limited to the private prototype binaries.  It
# never changes a served-model manifest, a campaign, an authorization, a
# release, a systemd unit, or the default CK dispatch.

set -euo pipefail

speed_first=0
lightweight_prompt_dir=''
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --speed-first)
            speed_first=1
            shift
            ;;
        --lightweight-prompt-dir)
            shift
            if [[ "$#" -eq 0 ]]; then
                printf '%s\n' '--lightweight-prompt-dir requires a directory' >&2
                exit 2
            fi
            lightweight_prompt_dir="$(realpath -e "$1")"
            shift
            ;;
        *)
            break
            ;;
    esac
done

if [[ "$#" -ne 3 ]]; then
    printf '%s\n' "usage: $0 [--speed-first] [--lightweight-prompt-dir DIR] RESULT_DIR COMPONENT_BIN SERVING_BIN" >&2
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
if [[ -n "$lightweight_prompt_dir" && ! -f "$lightweight_prompt_dir/suite.json" ]]; then
    printf '%s\n' 'lightweight prompt directory must contain suite.json' >&2
    exit 2
fi
if [[ -e "$result_dir/service/window-start.txt" ]]; then
    printf '%s\n' "refusing to overwrite existing window evidence: $result_dir/service/window-start.txt" >&2
    exit 2
fi

mkdir -p "$result_dir"/{component,serving/ck,serving/handwritten,full-model-multistep,service,telemetry,preflight,timing/ck,timing/handwritten}

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
    amd-smi metric --gpu "$r9700_amd_smi_gpu" --temperature --clock --power --violation --json \
        >"$result_dir/telemetry/$name.json" 2>&1
}

thermal_sample_is_ready() {
    python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    document = json.load(source)
rows = document.get("gpu_data")
if not isinstance(rows, list):
    raise SystemExit(1)
row = next((item for item in rows if item.get("gpu") == 2), None)
if not isinstance(row, dict):
    raise SystemExit(1)
temperature = row.get("temperature")
power = row.get("power")
if not isinstance(temperature, dict) or not isinstance(power, dict):
    raise SystemExit(1)
edge = temperature.get("edge", {}).get("value")
hotspot = temperature.get("hotspot", {}).get("value")
socket_power = power.get("socket_power", {}).get("value")
throttle = power.get("throttle_status")
if not all(isinstance(value, (int, float)) for value in (edge, hotspot, socket_power)):
    raise SystemExit(1)
if edge <= 42 and hotspot <= 45 and socket_power <= 30 and throttle == "UNTHROTTLED":
    raise SystemExit(0)
raise SystemExit(1)
PY
}

wait_for_cooldown() {
    local label=$1
    local attempt
    local sample
    for attempt in $(seq 1 300); do
        sample="telemetry/cooldown-${label}-$(printf '%03d' "$attempt").json"
        amd-smi metric --gpu "$r9700_amd_smi_gpu" --temperature --clock --power --violation --json \
            >"$result_dir/$sample" 2>&1 || true
        if thermal_sample_is_ready "$result_dir/$sample"; then
            {
                date --iso-8601=seconds
                printf 'label=%s\n' "$label"
                printf 'accepted_sample=%s\n' "$sample"
                printf 'criteria=edge<=42C hotspot<=45C socket_power<=30W throttle=UNTHROTTLED\n'
            } >"$result_dir/timing/cooldown-${label}.txt"
            return 0
        fi
        sleep 1
    done
    return 1
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
capture preflight/required-pgrep-before.txt bash -c "pgrep -af 'ullm-sq8-r9700|run_measurements.py|llama-bench|llama-server|promote-served-model|ullm-aq4-worker' || true"
capture preflight/required-service-before.txt systemctl is-active ullm-openai.service
capture preflight/binary-sha256.txt sha256sum "$component_bin" "$serving_bin"
capture preflight/git-head.txt git rev-parse HEAD
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
capture preflight/required-pgrep-after-stop.txt bash -c "pgrep -af 'ullm-sq8-r9700|run_measurements.py|llama-bench|llama-server|promote-served-model|ullm-aq4-worker' || true"
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
amd-smi metric --gpu "$r9700_amd_smi_gpu" --temperature --clock --power --violation --watch 1 --watch_time 1800 --json \
    >"$result_dir/telemetry/during-window.watch.txt" 2>&1 &
telemetry_pid=$!

# New lightweight-promotion policy: collect matched, real full-model decode
# timing before examining numerical differences.  ``generated_index=0`` is
# the prefill/first-token transition; the summarizer intentionally includes
# only feedback decode indices 1 and later.
if [[ $speed_first -eq 1 ]]; then
    timing_repeats=5
    {
        printf 'method=synchronized full-model generated-step timing\n'
        printf 'repeats=%s\n' "$timing_repeats"
        printf 'prompt_tokens=1028\n'
        printf 'max_new_tokens=16\n'
        printf 'included_decode_indices=1_and_later\n'
        printf 'excluded=model_load,prefill,generated_index_0,profiler_ranges,gpu_event_timing\n'
    } >"$result_dir/timing/protocol.txt"
    for timing_repeat in $(seq 1 "$timing_repeats"); do
        timing_id=$(printf 'r%02d' "$timing_repeat")
        if ! wait_for_cooldown "ck-${timing_id}"; then
            printf '%s\n' "R9700 thermal cooldown timed out before CK ${timing_id}" >&2
            exit 1
        fi
        capture_optional "timing/ck/${timing_id}.stdout.txt" \
            run_r9700 "$serving_bin" \
                --artifact "$artifact_dir" --package "$package_dir" \
                --prompt-lengths 1028 --max-new-tokens 16 \
                --prefill-mode m8-chunk8 --record-generated-timing \
                --result-json "$result_dir/timing/ck/${timing_id}.json"
        capture_telemetry "after-timing-ck-${timing_id}"

        if ! wait_for_cooldown "handwritten-${timing_id}"; then
            printf '%s\n' "R9700 thermal cooldown timed out before handwritten ${timing_id}" >&2
            exit 1
        fi
        capture_optional "timing/handwritten/${timing_id}.stdout.txt" \
            run_r9700 "$serving_bin" \
                --artifact "$artifact_dir" --package "$package_dir" \
                --prompt-lengths 1028 --max-new-tokens 16 \
                --prefill-mode m8-chunk8 --record-generated-timing \
                --handwritten-wmma-projection-prototype \
                --decode-oracle-capture-dir "$result_dir/timing/handwritten/${timing_id}-decode" \
                --result-json "$result_dir/timing/handwritten/${timing_id}.json"
        capture_telemetry "after-timing-handwritten-${timing_id}"
    done

    timing_summary_ready=1
    for timing_repeat in $(seq 1 "$timing_repeats"); do
        timing_id=$(printf 'r%02d' "$timing_repeat")
        if [[ ! -f "$result_dir/timing/ck/${timing_id}.json" \
            || ! -f "$result_dir/timing/handwritten/${timing_id}.json" ]]; then
            timing_summary_ready=0
        fi
    done
    if [[ $timing_summary_ready -eq 1 ]]; then
        ck_summary=(python3 "$repo_root/tools/summarize-synchronized-decode-timing.py" \
            --label sq8_0_ck --output "$result_dir/timing/ck-summary.json")
        handwritten_summary=(python3 "$repo_root/tools/summarize-synchronized-decode-timing.py" \
            --label sq8_0_handwritten_wmma --output "$result_dir/timing/handwritten-summary.json")
        for timing_repeat in $(seq 1 "$timing_repeats"); do
            timing_id=$(printf 'r%02d' "$timing_repeat")
            ck_summary+=(--result-json "$result_dir/timing/ck/${timing_id}.json")
            handwritten_summary+=(--result-json "$result_dir/timing/handwritten/${timing_id}.json")
        done
        capture_required timing/ck-summary.stdout.txt "${ck_summary[@]}"
        capture_required timing/handwritten-summary.stdout.txt "${handwritten_summary[@]}"

        capture_required timing/speed-decision.stdout.txt \
            python3 "$repo_root/tools/compare-synchronized-decode-timing.py" \
                --baseline-summary "$result_dir/timing/ck-summary.json" \
                --candidate-summary "$result_dir/timing/handwritten-summary.json" \
                --output "$result_dir/timing/speed-decision.json"

        # The lightweight policy deliberately separates throughput from
        # quality.  Do not run the superseded exact-logit/component gates in
        # speed-first mode.  A non-faster candidate stops here; a faster one
        # gets actual fixed-prompt generations in this same isolation window.
        if ! json_field_is_true "$result_dir/timing/speed-decision.json" candidate_faster; then
            {
                printf 'decision=stop-after-speed\n'
                printf 'reason=handwritten WMMA pooled full-model feedback decode throughput did not exceed CK\n'
                printf 'quality_capture=not_run because the candidate is not faster\n'
            } >"$result_dir/timing/speed-first-outcome.txt"
            date --iso-8601=seconds >"$result_dir/service/isolation-complete.txt"
            exit 0
        fi

        if [[ -z "$lightweight_prompt_dir" ]]; then
            {
                printf 'decision=stop-after-speed\n'
                printf 'reason=candidate was faster but no fixed prompt suite was supplied\n'
                printf 'quality_capture=not_run\n'
            } >"$result_dir/timing/speed-first-outcome.txt"
            date --iso-8601=seconds >"$result_dir/service/isolation-complete.txt"
            exit 1
        fi

        mkdir -p "$result_dir/lightweight/ck" "$result_dir/lightweight/handwritten"
        mapfile -t lightweight_cases < <(
            jq -r '.cases[] | [.case_id, .max_completion_tokens, .prompt_u32le_file] | @tsv' \
                "$lightweight_prompt_dir/suite.json"
        )
        if [[ "${#lightweight_cases[@]}" -eq 0 ]]; then
            printf '%s\n' 'the supplied lightweight prompt suite contains no cases' >&2
            exit 1
        fi
        lightweight_complete=1
        for lightweight_case in "${lightweight_cases[@]}"; do
            IFS=$'\t' read -r case_id max_completion_tokens prompt_relative <<<"$lightweight_case"
            if [[ ! "$case_id" =~ ^[a-z][a-z0-9_]{0,63}$ ]] \
                || [[ ! "$max_completion_tokens" =~ ^[0-9]+$ ]] \
                || [[ "$max_completion_tokens" -lt 3 ]] \
                || [[ -z "$prompt_relative" ]]; then
                printf '%s\n' "invalid lightweight suite case record: $lightweight_case" >&2
                exit 1
            fi
            prompt_file="$(realpath -e "$lightweight_prompt_dir/$prompt_relative")"
            case "$prompt_file" in
                "$lightweight_prompt_dir"/*) ;;
                *)
                    printf '%s\n' "lightweight suite input escapes its directory: $case_id" >&2
                    exit 1
                    ;;
            esac
            if [[ ! -f "$prompt_file" ]]; then
                printf '%s\n' "lightweight suite input is not a regular file: $case_id" >&2
                exit 1
            fi
            capture_optional "lightweight/ck/${case_id}.stdout.txt" \
                run_r9700 "$serving_bin" \
                    --artifact "$artifact_dir" --package "$package_dir" \
                    --prompt-token-ids-u32le "$prompt_file" --max-new-tokens "$max_completion_tokens" \
                    --prefill-mode m8-chunk8 \
                    --result-json "$result_dir/lightweight/ck/${case_id}.json"
            capture_telemetry "after-lightweight-ck-${case_id}"
            capture_optional "lightweight/handwritten/${case_id}.stdout.txt" \
                run_r9700 "$serving_bin" \
                    --artifact "$artifact_dir" --package "$package_dir" \
                    --prompt-token-ids-u32le "$prompt_file" --max-new-tokens "$max_completion_tokens" \
                    --prefill-mode m8-chunk8 --handwritten-wmma-projection-prototype \
                    --decode-oracle-capture-dir "$result_dir/lightweight/handwritten/${case_id}-decode" \
                    --result-json "$result_dir/lightweight/handwritten/${case_id}.json"
            capture_telemetry "after-lightweight-handwritten-${case_id}"
            if [[ ! -f "$result_dir/lightweight/ck/${case_id}.json" \
                || ! -f "$result_dir/lightweight/handwritten/${case_id}.json" ]]; then
                lightweight_complete=0
            fi
        done
        if [[ $lightweight_complete -ne 1 ]]; then
            {
                printf 'decision=quality-capture-incomplete\n'
                printf 'reason=at least one fixed-prompt CK or handwritten result JSON is absent\n'
                printf 'suite_dir=%s\n' "$lightweight_prompt_dir"
            } >"$result_dir/timing/speed-first-outcome.txt"
            date --iso-8601=seconds >"$result_dir/service/isolation-complete.txt"
            exit 1
        fi
        {
            printf 'decision=continue-to-lightweight-output-capture\n'
            printf 'reason=handwritten WMMA pooled full-model feedback decode throughput exceeded CK\n'
            printf 'suite_dir=%s\n' "$lightweight_prompt_dir"
        } >"$result_dir/timing/speed-first-outcome.txt"
        date --iso-8601=seconds >"$result_dir/service/isolation-complete.txt"
        exit 0
    else
        printf '%s\n' 'not summarized: at least one CK or handwritten timing result is absent' \
            >"$result_dir/timing/summary-not-run.txt"
        date --iso-8601=seconds >"$result_dir/service/isolation-complete.txt"
        exit 1
    fi
fi

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
