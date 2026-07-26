#!/usr/bin/env bash
# One stop -> isolated numerical validation + prefill remeasurement -> restore.
# Read the already-authorized sudo password once from stdin; never persist it.
set -Eeuo pipefail

RESULT="/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-26/prefill-tail-fix"
SERVICE="ullm-openai.service"
GPU_INDEX="2"
BASELINE_ROOT="/tmp/ullm-prefill-clean-0216b131"
CANDIDATE_ROOT="/tmp/ullm-prefill-tail-fix-source"
BASELINE_SERVING="${BASELINE_ROOT}/target/release/examples/sq8_ck_serving"
CANDIDATE_SERVING="/tmp/ullm-prefill-tail-fix-target/release/examples/sq8_ck_serving"
ARTIFACT="/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact"
PACKAGE="/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/package"
ALLOW_INHERITED_STOPPED="${ALLOW_INHERITED_STOPPED:-0}"

read -r SUDO_PASSWORD

mkdir -p "${RESULT}/service" "${RESULT}/numerical/baseline" "${RESULT}/numerical/candidate"

event() {
    printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >> "${RESULT}/service/window-events.tsv"
}

sudo_systemctl() {
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S -p '' systemctl "$@"
}

record_service() {
    local prefix="$1"
    systemctl show "${SERVICE}" \
        --property=ActiveState,SubState,MainPID,NRestarts,StartLimitBurst,StartLimitIntervalUSec,UnitFileState \
        > "${RESULT}/service/${prefix}-systemctl-show.txt" 2>&1 || true
    systemctl is-active "${SERVICE}" > "${RESULT}/service/${prefix}-is-active.txt" 2>&1 || true
}

restore_service() {
    local prior_status=$?
    trap - EXIT
    set +e
    event "restore-attempt"
    sudo_systemctl start "${SERVICE}" \
        > "${RESULT}/service/restore-start.stdout" \
        2> "${RESULT}/service/restore-start.stderr"
    local start_status=$?
    printf '%s\n' "${start_status}" > "${RESULT}/service/restore-start.exit-status"
    event "restore-start-return=${start_status}"
    local delay
    for delay in 1 2 4 8 16 24; do
        if systemctl is-active --quiet "${SERVICE}"; then
            event "restore-active"
            break
        fi
        sleep "${delay}"
    done
    record_service "post-restore"
    sha256sum /etc/ullm/served-models/active.json > "${RESULT}/service/active-manifest-post-restore.sha256" 2>&1 || true
    event "restore-finished"
    exit "${prior_status}"
}

capture_preflight() {
    pgrep -af 'ullm-sq8-r9700|run_measurements.py|llama-bench|llama-server|promote-served-model|ullm-aq4-worker' \
        > "${RESULT}/service/pre-window-pgrep.txt" || true
    systemctl is-active "${SERVICE}" > "${RESULT}/service/pre-window-is-active.txt" 2>&1 || true
    record_service "pre-window"
    systemctl show llama-qwen35-udq4.service \
        --property=ActiveState,UnitFileState,SubState,MainPID \
        > "${RESULT}/service/llama-qwen35-state.txt" 2>&1 || true
    amd-smi process --gpu "${GPU_INDEX}" --json > "${RESULT}/service/r9700-process-pre-stop.json" 2>&1 || true
    amd-smi metric --gpu "${GPU_INDEX}" --json > "${RESULT}/service/r9700-metric-pre-stop.json" 2>&1 || true
    sha256sum /etc/ullm/served-models/active.json > "${RESULT}/service/active-manifest-pre-stop.sha256"
}

capture_preflight
if ! grep -qx 'ActiveState=inactive' "${RESULT}/service/llama-qwen35-state.txt" \
    || ! grep -qx 'UnitFileState=disabled' "${RESULT}/service/llama-qwen35-state.txt"; then
    echo "llama-qwen35-udq4.service is not inactive+disabled" >&2
    exit 1
fi

service_was_active=0
if grep -qx 'active' "${RESULT}/service/pre-window-is-active.txt"; then
    service_was_active=1
elif [[ "${ALLOW_INHERITED_STOPPED}" == "1" ]] \
    && grep -qx 'inactive' "${RESULT}/service/pre-window-is-active.txt"; then
    # Another owner may have already stopped the service.  Adopt that quiet
    # state only when explicitly requested, and always restore it on exit.
    event "adopt-inherited-inactive-window"
else
    echo "${SERVICE} was not active before the owned service window" >&2
    exit 1
fi

trap restore_service EXIT
if (( service_was_active )); then
    event "stop-attempt"
    sudo_systemctl stop "${SERVICE}" \
        > "${RESULT}/service/stop.stdout" \
        2> "${RESULT}/service/stop.stderr"
    printf '%s\n' "$?" > "${RESULT}/service/stop.exit-status"
    event "stop-complete"
else
    printf '%s\n' 'not-issued-inherited-inactive' > "${RESULT}/service/stop.exit-status"
    event "stop-not-issued-inherited-inactive"
fi
record_service "post-stop"
amd-smi process --gpu "${GPU_INDEX}" --json > "${RESULT}/service/r9700-process-post-stop.json" 2>&1 || true
pgrep -af 'ullm-sq8-r9700|run_measurements.py|llama-bench|llama-server|promote-served-model|ullm-aq4-worker' \
    > "${RESULT}/service/post-stop-pgrep.txt" || true

if pgrep -x 'ullm-aq4-worker' >/dev/null; then
    echo "AQ4_0 worker remains after service stop; refusing isolated GPU work" >&2
    exit 1
fi

export HIP_VISIBLE_DEVICES=1
unset ROCR_VISIBLE_DEVICES || true
export ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1
export ULLM_REQUIRE_HIP_ROPE_KERNEL=1
export ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1
export ULLM_REQUIRE_HIP_ADD_KERNEL=1
export ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1
export ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1
export ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1
export ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL=1
export ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1
export ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1

event "baseline-oracle-start"
(
    cd "${BASELINE_ROOT}"
    "${BASELINE_SERVING}" \
        --artifact "${ARTIFACT}" \
        --package "${PACKAGE}" \
        --prompt-lengths 128,512,1024,2048,4095,1000,129 \
        --max-new-tokens 1 \
        --prefill-mode m128-chunk128 \
        --oracle-capture-dir "${RESULT}/numerical/baseline/oracle" \
        --result-json "${RESULT}/numerical/baseline/result.json"
) > "${RESULT}/numerical/baseline/stdout.log" 2> "${RESULT}/numerical/baseline/stderr.log"
event "baseline-oracle-complete"

event "candidate-oracle-start"
(
    cd "${CANDIDATE_ROOT}"
    "${CANDIDATE_SERVING}" \
        --artifact "${ARTIFACT}" \
        --package "${PACKAGE}" \
        --prompt-lengths 128,512,1024,2048,4095,1000,129 \
        --max-new-tokens 1 \
        --prefill-mode m128-chunk128 \
        --oracle-capture-dir "${RESULT}/numerical/candidate/oracle" \
        --result-json "${RESULT}/numerical/candidate/result.json"
) > "${RESULT}/numerical/candidate/stdout.log" 2> "${RESULT}/numerical/candidate/stderr.log"
event "candidate-oracle-complete"

python3 "${RESULT}/compare_oracles.py" \
    --baseline-result "${RESULT}/numerical/baseline/result.json" \
    --candidate-result "${RESULT}/numerical/candidate/result.json" \
    --output "${RESULT}/numerical/comparison.json"
event "oracle-comparison-complete"

event "prefill-measurement-start"
python3 "${RESULT}/run_measurements.py" \
    > "${RESULT}/runner.stdout" \
    2> "${RESULT}/runner.stderr"
event "prefill-measurement-complete"
