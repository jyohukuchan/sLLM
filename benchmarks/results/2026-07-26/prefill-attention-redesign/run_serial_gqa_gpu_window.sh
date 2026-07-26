#!/usr/bin/env bash
# Follow-up for the arithmetic-schedule-preserving GQA grouped candidate.
# The wrapper owns one isolated R9700 window; it never changes a manifest or
# starts the disabled llama.cpp unit.
set -Eeuo pipefail

ROOT="/home/homelab1/coding-local/ultimateLLM/uLLM-project"
RESULT="${ROOT}/benchmarks/results/2026-07-26/prefill-attention-redesign"
SERVICE="ullm-openai.service"
LLAMA_SERVICE="llama-qwen35-udq4.service"
GPU_LOCK="/run/ullm/r9700.lock"
GPU_INDEX="2"
ARTIFACT="/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact"
PACKAGE="/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/package"
SERVING="/tmp/ullm-br-prefill-final/target/release/examples/sq8_ck_serving"
DRIVER="/tmp/ullm-sq8-r9700-phase0-profile-20260726/target/release/ullm-sq8-r9700-phase0-profile"
ENGINE="/tmp/ullm-br-prefill-final/target/release/ullm-engine"
BASELINE_ENV="ULLM_DISABLE_SQ8_0_FLASH2_GQA_GROUPED"

read -r SUDO_PASSWORD
mkdir -p "${RESULT}/service" "${RESULT}/numerical/serial-baseline" \
    "${RESULT}/numerical/serial-gqa" "${RESULT}/raw/serial-attention-probe"
EVENTS="${RESULT}/service/serial-gqa-window-events.tsv"
if [[ -e "${EVENTS}" ]]; then
    echo "refusing to append a second serial-gqa window" >&2
    exit 1
fi

event() { printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >> "${EVENTS}"; }
sudo_systemctl() { printf '%s\n' "${SUDO_PASSWORD}" | sudo -S -p '' systemctl "$@"; }
record_service() {
    local label="$1"
    systemctl show "${SERVICE}" -p ActiveState -p SubState -p MainPID -p NRestarts \
        -p StartLimitBurst -p StartLimitIntervalUSec -p UnitFileState \
        > "${RESULT}/service/serial-gqa-${label}-ullm-openai.txt" 2>&1 || true
    systemctl show "${LLAMA_SERVICE}" -p ActiveState -p SubState -p MainPID -p UnitFileState \
        > "${RESULT}/service/serial-gqa-${label}-llama-qwen35-udq4.txt" 2>&1 || true
}

service_stopped=0
lock_held=0
release_lock() {
    if (( lock_held )); then
        flock -u 9 2>/dev/null || true
        exec 9>&-
        lock_held=0
        event "r9700-lock-released"
    fi
}
wait_service_stop() {
    local attempt
    for attempt in $(seq 1 30); do
        if ! systemctl is-active --quiet "${SERVICE}" && ! pgrep -x ullm-aq4-worker >/dev/null; then
            return 0
        fi
        sleep 1
    done
    return 1
}
wait_lock_release() {
    local attempt
    for attempt in $(seq 1 30); do
        if ! fuser "${GPU_LOCK}" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}
restore_service() {
    local status=$?
    trap - EXIT
    release_lock
    if (( service_stopped )); then
        event "restore-attempt"
        if sudo_systemctl start "${SERVICE}" > "${RESULT}/service/serial-gqa-restore.stdout" \
            2> "${RESULT}/service/serial-gqa-restore.stderr"; then
            event "restore-start-return=0"
        else
            local restore_status=$?
            printf '%s\n' "${restore_status}" > "${RESULT}/service/serial-gqa-restore.exit-status"
            event "restore-start-return=${restore_status}"
            systemctl show "${SERVICE}" -p ActiveState -p SubState -p Result -p NRestarts \
                > "${RESULT}/service/serial-gqa-restore-first-failure-state.txt" 2>&1 || true
            if grep -qx 'Result=start-limit' "${RESULT}/service/serial-gqa-restore-first-failure-state.txt"; then
                event "restore-reset-failed-attempt"
                if sudo_systemctl reset-failed "${SERVICE}" \
                    > "${RESULT}/service/serial-gqa-reset-failed.stdout" \
                    2> "${RESULT}/service/serial-gqa-reset-failed.stderr" && \
                    sudo_systemctl start "${SERVICE}" \
                    > "${RESULT}/service/serial-gqa-restore-retry.stdout" \
                    2> "${RESULT}/service/serial-gqa-restore-retry.stderr"; then
                    event "restore-retry-return=0"
                else
                    status=1
                    event "restore-retry-return=nonzero"
                fi
            else
                status=1
            fi
        fi
        for delay in 1 2 4 8 16 24; do
            if systemctl is-active --quiet "${SERVICE}"; then
                event "restore-active"
                break
            fi
            sleep "${delay}"
        done
    fi
    record_service "post-restore"
    amd-smi metric --gpu "${GPU_INDEX}" --json \
        > "${RESULT}/service/serial-gqa-r9700-metric-post-restore.json" 2>&1 || true
    sha256sum /etc/ullm/served-models/active.json \
        > "${RESULT}/service/serial-gqa-active-manifest-post-restore.sha256" 2>&1 || true
    event "window-finished"
    exit "${status}"
}
trap restore_service EXIT

event "preflight-begin"
fuser -v "${GPU_LOCK}" > "${RESULT}/service/serial-gqa-pre-window-fuser.txt" 2>&1 || true
pgrep -af 'ullm-sq8-r9700|run_measurements.py|llama-bench|llama-server|promote-served-model' \
    > "${RESULT}/service/serial-gqa-pre-window-pgrep.txt" || true
record_service "pre-window"
amd-smi static --gpu "${GPU_INDEX}" --json > "${RESULT}/service/serial-gqa-r9700-static.json" 2>&1 || true
amd-smi process --gpu "${GPU_INDEX}" --json \
    > "${RESULT}/service/serial-gqa-r9700-process-pre-stop.json" 2>&1 || true
amd-smi metric --gpu "${GPU_INDEX}" --json \
    > "${RESULT}/service/serial-gqa-r9700-metric-pre-stop.json" 2>&1 || true
sha256sum /etc/ullm/served-models/active.json \
    > "${RESULT}/service/serial-gqa-active-manifest-pre-stop.sha256"
sha256sum "${SERVING}" "${DRIVER}" "${ENGINE}" \
    > "${RESULT}/service/serial-gqa-executable-sha256.txt"

if ! grep -qx 'ActiveState=inactive' "${RESULT}/service/serial-gqa-pre-window-llama-qwen35-udq4.txt" \
    || ! grep -qx 'UnitFileState=disabled' "${RESULT}/service/serial-gqa-pre-window-llama-qwen35-udq4.txt"; then
    echo "${LLAMA_SERVICE} is not inactive+disabled" >&2
    exit 1
fi
if ! systemctl is-active --quiet "${SERVICE}"; then
    echo "${SERVICE} was not active before this owned window" >&2
    exit 1
fi
for path in "${SERVING}" "${DRIVER}" "${ENGINE}"; do
    if [[ ! -x "${path}" ]]; then
        echo "required serial-gqa executable is absent: ${path}" >&2
        exit 1
    fi
done

event "stop-attempt"
sudo_systemctl stop "${SERVICE}" > "${RESULT}/service/serial-gqa-stop.stdout" \
    2> "${RESULT}/service/serial-gqa-stop.stderr"
service_stopped=1
event "stop-complete"
record_service "post-stop"
amd-smi process --gpu "${GPU_INDEX}" --json \
    > "${RESULT}/service/serial-gqa-r9700-process-post-stop.json" 2>&1 || true
if ! wait_service_stop; then
    echo "${SERVICE} or AQ4_0 worker remains after stop; refusing isolated measurement" >&2
    exit 1
fi
fuser -v "${GPU_LOCK}" > "${RESULT}/service/serial-gqa-post-stop-fuser-before-acquire.txt" 2>&1 || true
if ! wait_lock_release; then
    echo "R9700 lock remains held after service stop; refusing to steal it" >&2
    exit 75
fi
exec 9<"${GPU_LOCK}"
if ! flock -n 9; then
    echo "R9700 lock was acquired by another process after service stop; refusing to steal it" >&2
    exec 9>&-
    exit 75
fi
lock_held=1
event "window-begin-lock-acquired-after-service-stop"
fuser -v "${GPU_LOCK}" > "${RESULT}/service/serial-gqa-post-acquire-fuser.txt" 2>&1 || true
amd-smi process --gpu "${GPU_INDEX}" --json \
    > "${RESULT}/service/serial-gqa-r9700-process-post-lock.json" 2>&1 || true
if pgrep -x ullm-aq4-worker >/dev/null; then
    echo "AQ4_0 worker appeared after lock acquisition; refusing isolated measurement" >&2
    exit 1
fi

HIP_GUARDS=(
    ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 ULLM_REQUIRE_HIP_ROPE_KERNEL=1
    ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_ADD_KERNEL=1
    ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1
    ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1
    ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL=1
    ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1 ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1
)

event "numerical-serial-baseline-begin"
env -u ROCR_VISIBLE_DEVICES -u ULLM_USE_SQ8_0_FLASH2_GQA_GROUPED_PROTOTYPE \
    -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE HIP_VISIBLE_DEVICES=1 \
    "${BASELINE_ENV}"=1 "${HIP_GUARDS[@]}" \
    "${SERVING}" --artifact "${ARTIFACT}" --package "${PACKAGE}" \
    --prompt-lengths 128,512,1024,2048,4095 --max-new-tokens 1 --prefill-mode m128-chunk128 \
    --oracle-capture-dir "${RESULT}/numerical/serial-baseline/oracle" \
    --result-json "${RESULT}/numerical/serial-baseline/result.json" \
    > "${RESULT}/numerical/serial-baseline/stdout.log" \
    2> "${RESULT}/numerical/serial-baseline/stderr.log"
event "numerical-serial-baseline-complete"

event "numerical-serial-gqa-begin"
env -u ROCR_VISIBLE_DEVICES -u ULLM_USE_SQ8_0_FLASH2_GQA_GROUPED_PROTOTYPE \
    -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE -u "${BASELINE_ENV}" HIP_VISIBLE_DEVICES=1 \
    "${HIP_GUARDS[@]}" \
    "${SERVING}" --artifact "${ARTIFACT}" --package "${PACKAGE}" \
    --prompt-lengths 128,512,1024,2048,4095 --max-new-tokens 1 --prefill-mode m128-chunk128 \
    --oracle-capture-dir "${RESULT}/numerical/serial-gqa/oracle" \
    --result-json "${RESULT}/numerical/serial-gqa/result.json" \
    > "${RESULT}/numerical/serial-gqa/stdout.log" \
    2> "${RESULT}/numerical/serial-gqa/stderr.log"
event "numerical-serial-gqa-complete"
python3 "${RESULT}/compare_oracles.py" \
    --baseline-result "${RESULT}/numerical/serial-baseline/result.json" \
    --candidate-result "${RESULT}/numerical/serial-gqa/result.json" \
    --output "${RESULT}/numerical/serial-gqa-comparison.json"
event "numerical-serial-gqa-compared"

event "attention-only-generic-begin"
if env -u ROCR_VISIBLE_DEVICES -u ULLM_USE_SQ8_0_FLASH2_GQA_GROUPED_PROTOTYPE \
    -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE HIP_VISIBLE_DEVICES=1 \
    "${BASELINE_ENV}"=1 "${ENGINE}" runtime-cached-prefix-attn-smoke \
    0 4095 128 1 40 8 128 128 cached_prefix_flash2 f32 \
    > "${RESULT}/raw/serial-attention-probe/generic.log" \
    2> "${RESULT}/raw/serial-attention-probe/generic.stderr.log"; then
    printf '0\n' > "${RESULT}/raw/serial-attention-probe/generic.exit-status"
    event "attention-only-generic-complete-return=0"
else
    smoke_status=$?
    printf '%s\n' "${smoke_status}" > "${RESULT}/raw/serial-attention-probe/generic.exit-status"
    event "attention-only-generic-complete-return=${smoke_status}"
fi
event "attention-only-serial-gqa-begin"
if env -u ROCR_VISIBLE_DEVICES -u ULLM_USE_SQ8_0_FLASH2_GQA_GROUPED_PROTOTYPE \
    -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE -u "${BASELINE_ENV}" HIP_VISIBLE_DEVICES=1 \
    "${ENGINE}" runtime-cached-prefix-attn-smoke \
    0 4095 128 1 40 8 128 128 cached_prefix_flash2 f32 \
    > "${RESULT}/raw/serial-attention-probe/serial-gqa.log" \
    2> "${RESULT}/raw/serial-attention-probe/serial-gqa.stderr.log"; then
    printf '0\n' > "${RESULT}/raw/serial-attention-probe/serial-gqa.exit-status"
    event "attention-only-serial-gqa-complete-return=0"
else
    smoke_status=$?
    printf '%s\n' "${smoke_status}" > "${RESULT}/raw/serial-attention-probe/serial-gqa.exit-status"
    event "attention-only-serial-gqa-complete-return=${smoke_status}"
fi

event "throughput-serial-gqa-begin"
env ULLM_PREFILL_RAW_SUBDIR=raw/serial-gqa-throughput \
    ULLM_PREFILL_DRIVER="${DRIVER}" \
    ULLM_PREFILL_CANDIDATE_LABEL=gqa_grouped_tile20_serial_generic_schedule \
    ULLM_PREFILL_CANDIDATE_ENV='' \
    ULLM_PREFILL_BASELINE_ENV="${BASELINE_ENV}" \
    ULLM_PREFILL_CANDIDATE_DESCRIPTION='automatic gfx1201 Q=40/KV=8 grouped Flash2; 20-token staged K/V, generic 256-thread per-Q arithmetic schedule' \
    ULLM_PREFILL_CONFIG_NAME=serial-gqa-throughput-run-configuration.json \
    ULLM_PREFILL_SUMMARY_NAME=serial-gqa-throughput-summary.json \
    python3 "${RESULT}/run_measurements.py" \
    > "${RESULT}/serial-gqa-throughput-runner.stdout" \
    2> "${RESULT}/serial-gqa-throughput-runner.stderr"
event "throughput-serial-gqa-complete"
amd-smi metric --gpu "${GPU_INDEX}" --json \
    > "${RESULT}/service/serial-gqa-r9700-metric-before-restore.json" 2>&1 || true
amd-smi process --gpu "${GPU_INDEX}" --json \
    > "${RESULT}/service/serial-gqa-r9700-process-before-restore.json" 2>&1 || true
event "measurement-complete"
