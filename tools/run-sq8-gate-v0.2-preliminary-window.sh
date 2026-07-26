#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# One bounded R9700-only window for the SQ8 v0.2 preliminary snapshot.  It
# never reads or writes the served-model activation manifest, and it restores
# only the pre-existing ullm-openai.service state.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 RESULT_DIR" >&2
    exit 2
fi

result_dir=$(realpath -m "$1")
repo_root=$(git rev-parse --show-toplevel)
capture_bin="$repo_root/target/release/ullm-sq8-gate-capture"
serving_bin="$repo_root/target/release/examples/sq8_ck_serving"
preparer="$repo_root/tools/prepare-sq8-gate-v0.2-capture.py"
evaluator="$repo_root/tools/evaluate-sq8-gate-v0.2.py"
performance_summary="$repo_root/tools/summarize-sq8-gate-v0.2-preliminary-performance.py"
gate="$repo_root/docs/plans/sq8-numerical-gate-v0.2-relative-to-fp32-reference.json"
reference="$result_dir/reference-snapshot-2160.json"
artifact=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact
package=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/package
timing_prompt="$repo_root/tests/fixtures/sq8-serving-v0.1/oracles/vllm-source-v0.1/inputs/raw-p0512.u32le"
service=ullm-openai.service
r9700_bdf=0000:47:00.0

for path in "$capture_bin" "$serving_bin" "$preparer" "$evaluator" "$performance_summary" "$gate" "$reference" "$timing_prompt"; do
    [[ -e "$path" ]] || { echo "required path is unavailable: $path" >&2; exit 2; }
done
[[ -x "$capture_bin" && -x "$serving_bin" ]] || { echo "capture or serving binary is not executable" >&2; exit 2; }
[[ -d "$artifact" && -d "$package" ]] || { echo "artifact or package is unavailable" >&2; exit 2; }
[[ ! -e "$result_dir/service/window-start.txt" ]] || { echo "refusing to reuse service window output" >&2; exit 2; }
[[ ! -e "$result_dir/captures" && ! -e "$result_dir/evaluations" && ! -e "$result_dir/performance" ]] || {
    echo "refusing to overwrite capture, evaluation, or performance output" >&2
    exit 2
}

mkdir -p "$result_dir"/{captures,evaluations,performance,preflight,service,telemetry}

sudo_service() {
    if [[ -n ${ULLM_SUDO_PASSWORD:-} ]]; then
        printf '%s\n' "$ULLM_SUDO_PASSWORD" | sudo -S -p '' "$@"
    else
        sudo -n "$@"
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

capture_optional() {
    local relative=$1
    shift
    if "$@" >"$result_dir/$relative" 2>&1; then
        printf '0\n' >"$result_dir/$relative.exit-status"
    else
        printf '%s\n' "$?" >"$result_dir/$relative.exit-status"
    fi
    return 0
}

event() {
    printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >>"$result_dir/service/events.tsv"
}

capture_service_state() {
    local name=$1
    {
        date --iso-8601=seconds
        systemctl show "$service" -p ActiveState -p SubState -p Result -p MainPID -p NRestarts \
            -p StartLimitBurst -p StartLimitIntervalUSec --no-pager
        printf 'llama-qwen35-udq4.active='; systemctl is-active llama-qwen35-udq4.service || true
        printf 'llama-qwen35-udq4.enabled='; systemctl is-enabled llama-qwen35-udq4.service || true
        printf 'gdm3.active='; systemctl is-active gdm3.service || true
    } >"$result_dir/service/$name.txt" 2>&1
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
        if sudo_service systemctl start "$service" >>"$result_dir/service/restore.txt" 2>&1 \
            && systemctl is-active --quiet "$service"; then
            restored=1
            event 'service-restored-initial-start'
        else
            # The first start is the normal recovery.  Only after a verified
            # failure do we clear a failed state and consume the one remaining
            # recovery attempt, staying below StartLimitBurst=3.
            {
                echo 'initial-start-failed; one reset-failed recovery attempt follows'
                systemctl show "$service" -p ActiveState -p SubState -p Result -p NRestarts --no-pager || true
            } >>"$result_dir/service/restore.txt"
            sudo_service systemctl reset-failed "$service" >>"$result_dir/service/restore.txt" 2>&1 || true
            if sudo_service systemctl start "$service" >>"$result_dir/service/restore.txt" 2>&1 \
                && systemctl is-active --quiet "$service"; then
                restored=1
                event 'service-restored-after-reset-failed'
            else
                original_status=1
                event 'service-restore-failed'
            fi
        fi
    fi
    capture_service_state post-restore || true
    capture_optional telemetry/post-restore-r9700-process.json amd-smi process --gpu 2 --general --json
    exit "$original_status"
}

trap restore_service EXIT

printf 'timestamp\tevent\n' >"$result_dir/service/events.tsv"
date --iso-8601=seconds >"$result_dir/service/window-start.txt"
event 'window-start'
capture_service_state pre-stop
capture_required preflight/amd-smi-list.json amd-smi list --json
capture_required preflight/r9700-static.json amd-smi static --gpu 2 --asic --bus --json
capture_required preflight/r9700-process-before.json amd-smi process --gpu 2 --general --json
capture_required preflight/binaries.sha256 sha256sum "$capture_bin" "$serving_bin" "$gate" "$reference"

if ! rg -q "$r9700_bdf" "$result_dir/preflight/r9700-static.json"; then
    echo "GPU 2 is not the expected R9700 BDF $r9700_bdf" >&2
    exit 1
fi
if [[ $(systemctl is-active llama-qwen35-udq4.service) != inactive ]] \
    || [[ $(systemctl is-enabled llama-qwen35-udq4.service 2>/dev/null || true) != disabled ]] \
    || [[ $(systemctl is-active gdm3.service) != inactive ]]; then
    echo 'another R9700/UI service is active; refusing this isolated window' >&2
    exit 1
fi
if [[ $(systemctl is-active "$service") == active ]]; then
    service_was_active=1
fi
sudo_service true
event 'preflight-passed-r9700-bdf-and-service-state'

if [[ $service_was_active -eq 1 ]]; then
    capture_required service/stop.txt sudo_service systemctl stop "$service"
    event 'service-stopped'
else
    event 'service-was-inactive-no-stop-issued'
fi
capture_service_state post-stop
capture_required service/r9700-process-after-stop.json amd-smi process --gpu 2 --general --json
if rg -q '"pid"[[:space:]]*:' "$result_dir/service/r9700-process-after-stop.json"; then
    echo 'R9700 still has an owner after service stop' >&2
    exit 1
fi
event 'r9700-holder-check-passed'

run_capture() {
    local label=$1
    local plan=$2
    capture_required "captures/$label.stdout.txt" \
        env -u ROCR_VISIBLE_DEVICES \
            -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE \
            -u ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE \
            -u ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE \
            HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 "${kernel_guards[@]}" \
            python3 "$preparer" run --plan "$plan" --capture-binary "$capture_bin" \
                --output "$result_dir/captures/$label"
    event "capture-complete-$label"
}

run_capture control "$result_dir/control-plan.json"
run_capture tile128 "$result_dir/tile128-plan.json"
run_capture tile256 "$result_dir/tile256-plan.json"

evaluate_candidate() {
    local label=$1
    if env python3 "$evaluator" evaluate-preliminary \
        --gate "$gate" --reference "$reference" \
        --control "$result_dir/captures/control/capture-manifest.json" \
        --candidate "$result_dir/captures/$label/capture-manifest.json" \
        --output-json "$result_dir/evaluations/$label.json" \
        --output-markdown "$result_dir/evaluations/$label.md" \
        >"$result_dir/evaluations/$label.stdout.txt" 2>&1; then
        printf '0\n' >"$result_dir/evaluations/$label.stdout.txt.exit-status"
    else
        printf '%s\n' "$?" >"$result_dir/evaluations/$label.stdout.txt.exit-status"
    fi
    event "evaluation-complete-$label"
}

evaluate_candidate tile128
evaluate_candidate tile256

preliminary_speed_eligible() {
    local result=$1
    python3 - "$result" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
ok = (
    value.get("status") == "preliminary"
    and value.get("preliminary_outcome") == "pass_metric_subset"
    and value.get("selector_exposure", {}).get("multi_tile_exercised") is True
)
raise SystemExit(0 if ok else 1)
PY
}

run_speed_route() {
    local route=$1
    local tile=${2:-}
    local -a route_environment=(
        -u ROCR_VISIBLE_DEVICES
        -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE
        -u ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE
        -u ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE
    )
    if [[ -n "$tile" ]]; then
        route_environment+=(
            "ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE=$tile"
            ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE=1
        )
    fi
    mkdir -p "$result_dir/performance/$route"
    capture_required "performance/$route/stdout.txt" \
        env "${route_environment[@]}" HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 "${kernel_guards[@]}" \
            "$serving_bin" --artifact "$artifact" --package "$package" \
                --prompt-token-ids-u32le "$timing_prompt" --max-new-tokens 8 \
                --prefill-mode m128-chunk128 --record-generated-timing \
                --result-json "$result_dir/performance/$route/result.json"
    event "performance-complete-$route"
}

speed_candidates=()
if preliminary_speed_eligible "$result_dir/evaluations/tile128.json"; then
    speed_candidates+=(tile128)
fi
if preliminary_speed_eligible "$result_dir/evaluations/tile256.json"; then
    speed_candidates+=(tile256)
fi
if [[ ${#speed_candidates[@]} -gt 0 ]]; then
    run_speed_route direct
    for route in "${speed_candidates[@]}"; do
        case "$route" in
            tile128) run_speed_route tile128 128 ;;
            tile256) run_speed_route tile256 256 ;;
        esac
    done
    summary_args=(--direct "$result_dir/performance/direct/result.json")
    for route in "${speed_candidates[@]}"; do
        summary_args+=(--candidate "$result_dir/performance/$route/result.json")
    done
    capture_required performance/summary.stdout.txt \
        python3 "$performance_summary" "${summary_args[@]}" \
            --output "$result_dir/performance-summary.json"
else
    printf '%s\n' 'No candidate both passed the preliminary metric subset and exercised multi-tile execution; speed was intentionally not measured.' \
        >"$result_dir/performance/not-run.txt"
fi

capture_optional telemetry/post-measurement-r9700.json amd-smi metric --gpu 2 --temperature --clock --power --json
date --iso-8601=seconds >"$result_dir/service/window-measurements-complete.txt"
event 'window-measurements-complete'
