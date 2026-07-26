#!/usr/bin/env bash
# Complete the numerical and synchronized timing work after the Phase 0
# profiler window finished.  This owns a separate, recovery-safe R9700 window
# and intentionally does not rerun rocprof.
set -Eeuo pipefail

ROOT="/home/homelab1/coding-local/ultimateLLM/uLLM-project"
RESULT="${ROOT}/benchmarks/results/2026-07-26/prefill-attention-redesign"
SERVICE="ullm-openai.service"
LLAMA_SERVICE="llama-qwen35-udq4.service"
GPU_LOCK="/run/ullm/r9700.lock"
GPU_INDEX="2"
ARTIFACT="/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact"
PACKAGE="/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/package"
BASELINE_DRIVER="/tmp/ullm-br-prefill-baseline17-target/release/ullm-sq8-r9700-phase0-profile"
CANDIDATE_DRIVER="/tmp/ullm-br-prefill-gqa20-target/release/ullm-sq8-r9700-phase0-profile"
BASELINE_SERVING="/tmp/ullm-br-prefill-baseline17-target/release/examples/sq8_ck_serving"
CANDIDATE_SERVING="/tmp/ullm-br-prefill-gqa20-target/release/examples/sq8_ck_serving"

read -r SUDO_PASSWORD

mkdir -p "${RESULT}/service" "${RESULT}/raw/throughput" \
    "${RESULT}/numerical/baseline" "${RESULT}/numerical/candidate"
EVENTS="${RESULT}/service/remaining-window-events.tsv"
if [[ -e "${EVENTS}" ]]; then
    echo "refusing to append a second remaining-work GPU window to ${RESULT}" >&2
    exit 1
fi

event() {
    printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >> "${EVENTS}"
}

sudo_systemctl() {
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S -p '' systemctl "$@"
}

record_service() {
    local label="$1"
    systemctl show "${SERVICE}" \
        -p ActiveState -p SubState -p MainPID -p NRestarts -p StartLimitBurst -p StartLimitIntervalUSec -p UnitFileState \
        > "${RESULT}/service/remaining-${label}-ullm-openai.txt" 2>&1 || true
    systemctl show "${LLAMA_SERVICE}" -p ActiveState -p SubState -p MainPID -p UnitFileState \
        > "${RESULT}/service/remaining-${label}-llama-qwen35-udq4.txt" 2>&1 || true
}

service_stopped=0
r9700_lock_held=0

release_r9700_lock() {
    if (( r9700_lock_held )); then
        flock -u 9 2>/dev/null || true
        exec 9>&-
        r9700_lock_held=0
        event "r9700-lock-released"
    fi
}

wait_for_service_stop() {
    local attempt
    for attempt in $(seq 1 30); do
        if ! systemctl is-active --quiet "${SERVICE}" && ! pgrep -x 'ullm-aq4-worker' >/dev/null; then
            return 0
        fi
        sleep 1
    done
    return 1
}

wait_for_lock_release() {
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
    # The gateway takes this same singleton flock, so it cannot be restarted
    # until the owned measurement window has released it.
    release_r9700_lock
    if (( service_stopped )); then
        event "restore-attempt"
        if sudo_systemctl start "${SERVICE}" \
            > "${RESULT}/service/remaining-restore.stdout" \
            2> "${RESULT}/service/remaining-restore.stderr"; then
            event "restore-start-return=0"
        else
            local restore_status=$?
            printf '%s\n' "${restore_status}" > "${RESULT}/service/remaining-restore.exit-status"
            event "restore-start-return=${restore_status}"
            systemctl show "${SERVICE}" -p ActiveState -p SubState -p Result -p NRestarts \
                > "${RESULT}/service/remaining-restore-first-failure-state.txt" 2>&1 || true
            # This is a narrowly scoped recovery path after this script's own
            # stop; only an actual start-limit state permits a reset/retry.
            if grep -qx 'Result=start-limit' "${RESULT}/service/remaining-restore-first-failure-state.txt"; then
                event "restore-reset-failed-attempt"
                if sudo_systemctl reset-failed "${SERVICE}" \
                    > "${RESULT}/service/remaining-restore-reset-failed.stdout" \
                    2> "${RESULT}/service/remaining-restore-reset-failed.stderr" && \
                    sudo_systemctl start "${SERVICE}" \
                    > "${RESULT}/service/remaining-restore-retry.stdout" \
                    2> "${RESULT}/service/remaining-restore-retry.stderr"; then
                    event "restore-retry-return=0"
                else
                    local retry_status=$?
                    printf '%s\n' "${retry_status}" > "${RESULT}/service/remaining-restore-retry.exit-status"
                    event "restore-retry-return=${retry_status}"
                    status=1
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
    amd-smi metric --gpu "${GPU_INDEX}" --json > "${RESULT}/service/remaining-r9700-metric-post-restore.json" 2>&1 || true
    sha256sum /etc/ullm/served-models/active.json > "${RESULT}/service/remaining-active-manifest-post-restore.sha256" 2>&1 || true
    event "window-finished"
    exit "${status}"
}
trap restore_service EXIT

event "preflight-begin"
fuser -v "${GPU_LOCK}" > "${RESULT}/service/remaining-pre-window-fuser.txt" 2>&1 || true
pgrep -af 'ullm-sq8-r9700|run_measurements.py|llama-bench|llama-server|promote-served-model' \
    > "${RESULT}/service/remaining-pre-window-pgrep.txt" || true
record_service "pre-window"
amd-smi static --gpu "${GPU_INDEX}" --json > "${RESULT}/service/remaining-r9700-static.json" 2>&1 || true
amd-smi process --gpu "${GPU_INDEX}" --json > "${RESULT}/service/remaining-r9700-process-pre-stop.json" 2>&1 || true
amd-smi metric --gpu "${GPU_INDEX}" --json > "${RESULT}/service/remaining-r9700-metric-pre-stop.json" 2>&1 || true
sha256sum /etc/ullm/served-models/active.json > "${RESULT}/service/remaining-active-manifest-pre-stop.sha256"
sha256sum "${BASELINE_DRIVER}" "${CANDIDATE_DRIVER}" "${BASELINE_SERVING}" "${CANDIDATE_SERVING}" \
    > "${RESULT}/service/remaining-executable-sha256.txt"

if ! grep -qx 'ActiveState=inactive' "${RESULT}/service/remaining-pre-window-llama-qwen35-udq4.txt" \
    || ! grep -qx 'UnitFileState=disabled' "${RESULT}/service/remaining-pre-window-llama-qwen35-udq4.txt"; then
    echo "${LLAMA_SERVICE} is not inactive+disabled" >&2
    exit 1
fi
if ! systemctl is-active --quiet "${SERVICE}"; then
    echo "${SERVICE} was not active before this owned window" >&2
    exit 1
fi
for executable in "${BASELINE_DRIVER}" "${CANDIDATE_DRIVER}" "${BASELINE_SERVING}" "${CANDIDATE_SERVING}"; do
    if [[ ! -x "${executable}" ]]; then
        echo "required executable is absent: ${executable}" >&2
        exit 1
    fi
done

event "stop-attempt"
sudo_systemctl stop "${SERVICE}" > "${RESULT}/service/remaining-stop.stdout" 2> "${RESULT}/service/remaining-stop.stderr"
service_stopped=1
event "stop-complete"
record_service "post-stop"
amd-smi process --gpu "${GPU_INDEX}" --json > "${RESULT}/service/remaining-r9700-process-post-stop.json" 2>&1 || true
if ! wait_for_service_stop; then
    echo "${SERVICE} or its AQ4_0 worker remains after stop; refusing isolated measurement" >&2
    exit 1
fi
fuser -v "${GPU_LOCK}" > "${RESULT}/service/remaining-post-stop-fuser-before-acquire.txt" 2>&1 || true
if ! wait_for_lock_release; then
    echo "R9700 lock remains held after service stop; refusing to steal it" >&2
    exit 75
fi
if ! exec 9<"${GPU_LOCK}"; then
    echo "failed to open R9700 lock after service stop" >&2
    exit 1
fi
if ! flock -n 9; then
    echo "R9700 lock was acquired by another process after service stop; refusing to steal it" >&2
    exec 9>&-
    exit 75
fi
r9700_lock_held=1
event "window-begin-lock-acquired-after-service-stop"
fuser -v "${GPU_LOCK}" > "${RESULT}/service/remaining-post-acquire-fuser.txt" 2>&1 || true
amd-smi process --gpu "${GPU_INDEX}" --json > "${RESULT}/service/remaining-r9700-process-post-lock.json" 2>&1 || true
if pgrep -x 'ullm-aq4-worker' >/dev/null; then
    echo "AQ4_0 worker appeared after R9700 lock acquisition; refusing isolated measurement" >&2
    exit 1
fi

HIP_GUARDS=(
    ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1
    ULLM_REQUIRE_HIP_ROPE_KERNEL=1
    ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1
    ULLM_REQUIRE_HIP_ADD_KERNEL=1
    ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1
    ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1
    ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1
    ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL=1
    ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1
    ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1
)

event "numerical-baseline-begin"
env -u ROCR_VISIBLE_DEVICES \
    -u ULLM_USE_SQ8_0_FLASH2_GQA_GROUPED_PROTOTYPE \
    -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE \
    HIP_VISIBLE_DEVICES=1 "${HIP_GUARDS[@]}" \
    "${BASELINE_SERVING}" --artifact "${ARTIFACT}" --package "${PACKAGE}" \
    --prompt-lengths 128,512,1024,2048,4095 --max-new-tokens 1 --prefill-mode m128-chunk128 \
    --oracle-capture-dir "${RESULT}/numerical/baseline/oracle" \
    --result-json "${RESULT}/numerical/baseline/result.json" \
    > "${RESULT}/numerical/baseline/stdout.log" 2> "${RESULT}/numerical/baseline/stderr.log"
event "numerical-baseline-complete"

event "numerical-candidate-begin"
env -u ROCR_VISIBLE_DEVICES \
    -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE \
    HIP_VISIBLE_DEVICES=1 ULLM_USE_SQ8_0_FLASH2_GQA_GROUPED_PROTOTYPE=1 "${HIP_GUARDS[@]}" \
    "${CANDIDATE_SERVING}" --artifact "${ARTIFACT}" --package "${PACKAGE}" \
    --prompt-lengths 128,512,1024,2048,4095 --max-new-tokens 1 --prefill-mode m128-chunk128 \
    --oracle-capture-dir "${RESULT}/numerical/candidate/oracle" \
    --result-json "${RESULT}/numerical/candidate/result.json" \
    > "${RESULT}/numerical/candidate/stdout.log" 2> "${RESULT}/numerical/candidate/stderr.log"
event "numerical-candidate-complete"

event "numerical-compare-begin"
python3 "${RESULT}/compare_oracles.py" \
    --baseline-result "${RESULT}/numerical/baseline/result.json" \
    --candidate-result "${RESULT}/numerical/candidate/result.json" \
    --output "${RESULT}/numerical/comparison.json"
event "numerical-compare-complete"

event "throughput-begin"
python3 "${RESULT}/run_measurements.py" \
    > "${RESULT}/throughput-runner.stdout" 2> "${RESULT}/throughput-runner.stderr"
event "throughput-complete"

amd-smi metric --gpu "${GPU_INDEX}" --json > "${RESULT}/service/remaining-r9700-metric-before-restore.json" 2>&1 || true
amd-smi process --gpu "${GPU_INDEX}" --json > "${RESULT}/service/remaining-r9700-process-before-restore.json" 2>&1 || true
event "measurement-complete"
