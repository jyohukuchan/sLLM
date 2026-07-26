#!/usr/bin/env bash
# Run the remaining BQ GPU evidence in one owned service window.
#
# The script is intentionally evidence-local.  It never changes active.json,
# never invokes promotion, and always attempts exactly one service restoration
# after an owned stop.  It must start only while ullm-openai.service is active;
# that makes a stale inactive window fail closed instead of stealing another
# task's reservation.
set -Eeuo pipefail

ROOT='/home/homelab1/coding-local/ultimateLLM/uLLM-project'
RESULT="$ROOT/benchmarks/results/2026-07-26/attention-redesign-shipping"
SERVICE='ullm-openai.service'
LOCK='/run/ullm/r9700.lock'
ACTIVE='/etc/ullm/served-models/active.json'
TOKEN_FILE='/etc/ullm/openai-api-key'
GATEWAY_SOURCE='/tmp/ullm-bq-gateway-source-bfc76a72'
SQ8_SOURCE='/tmp/ullm-bq-sq8-quality-source-bfc76a72'
GATEWAY_PYTHON="$ROOT/services/openai-gateway/.venv/bin/python"
AQ4_DRIVER='/home/homelab1/coding-local/ultimateLLM/uLLM-aq4-p3-deployment-build-target-c4c9a9b3/release/ullm-aq4-decode-step-profile'
AQ4_SUMMARY="$ROOT/tools/summarize-aq4-decode-attention-trace.py"
CAPTURE_TOOL="$SQ8_SOURCE/tools/capture-lightweight-served-suite.py"
COMPARE_TOOL="$SQ8_SOURCE/tools/compare-lightweight-suite-captures.py"
PROMPT_SUITE="$SQ8_SOURCE/docs/plans/lightweight-promotion-prompt-suite-v0.1.json"
DIRECT_MANIFEST="$RESULT/sq8_0-quality/direct/manifest.json"
GROUPED_MANIFEST="$RESULT/sq8_0-quality/gqa-grouped-tile20/manifest.json"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
WINDOW="$RESULT/service-window-$RUN_ID"
PHASE1="$RESULT/phase1/current-p3-compatible-c1339-$RUN_ID"
AQ4_CAPTURE="$RESULT/manifest/active-aq4-p3-isolated-$RUN_ID"
DIRECT_CAPTURE="$RESULT/sq8_0-quality/direct/capture-$RUN_ID"
GROUPED_CAPTURE="$RESULT/sq8_0-quality/gqa-grouped-tile20/capture-$RUN_ID"
COMPARISON="$RESULT/sq8_0-quality/comparison-$RUN_ID"

read -r SUDO_PASSWORD
mkdir -p "$WINDOW" "$PHASE1" "$AQ4_CAPTURE" "$DIRECT_CAPTURE" "$GROUPED_CAPTURE"

service_stopped=0
gateway_pid=''

event() {
    printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >> "$WINDOW/events.tsv"
}

sudo_systemctl() {
    printf '%s\n' "$SUDO_PASSWORD" | sudo -S -p '' systemctl "$@"
}

record_preflight() {
    local label="$1"
    fuser -v "$LOCK" > "$WINDOW/${label}-fuser.txt" 2>&1 || true
    # The required pgrep pattern is run, but only PIDs are retained: another
    # task's command-line prompt is neither useful evidence nor ours to store.
    pgrep -af 'ullm-sq8-r9700|run_measurements.py|llama-bench|llama-server|promote-served-model' \
        | awk -v self="$$" '$1 != self {print $1}' \
        > "$WINDOW/${label}-pgrep-pids.txt" || true
    systemctl show "$SERVICE" -p ActiveState -p SubState -p MainPID -p NRestarts \
        > "$WINDOW/${label}-service.txt" 2>&1 || true
    sha256sum "$ACTIVE" > "$WINDOW/${label}-active-manifest.sha256"
}

stop_gateway() {
    local label="$1"
    local pid="$2"
    local attempt
    if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
        return 0
    fi
    kill -TERM "$pid" 2>/dev/null || true
    for attempt in $(seq 1 30); do
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        printf '%s\n' 'gateway-term-timeout; sending KILL' >> "$label/gateway-lifecycle.txt"
        kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
}

restore_service() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$gateway_pid" ]]; then
        stop_gateway "$gateway_pid_dir" "$gateway_pid"
        gateway_pid=''
    fi
    if (( service_stopped )); then
        event 'service-restore-attempt'
        if sudo_systemctl start "$SERVICE" > "$WINDOW/service-restore.stdout" \
            2> "$WINDOW/service-restore.stderr"; then
            printf '%s\n' 0 > "$WINDOW/service-restore.exit-status"
        else
            printf '%s\n' "$?" > "$WINDOW/service-restore.exit-status"
            status=1
        fi
        local attempt
        for attempt in $(seq 1 30); do
            if systemctl is-active --quiet "$SERVICE"; then
                event 'service-restore-active'
                break
            fi
            sleep 1
        done
    fi
    record_preflight 'post-window'
    event "window-finished-status=$status"
    exit "$status"
}
trap restore_service EXIT INT TERM

for required in "$ACTIVE" "$TOKEN_FILE" "$GATEWAY_PYTHON" "$AQ4_DRIVER" "$AQ4_SUMMARY" \
    "$CAPTURE_TOOL" "$COMPARE_TOOL" "$PROMPT_SUITE" "$DIRECT_MANIFEST" "$GROUPED_MANIFEST"; do
    [[ -e "$required" ]] || { printf 'missing required input: %s\n' "$required" >&2; exit 1; }
done
[[ -x "$GATEWAY_PYTHON" && -x "$AQ4_DRIVER" ]] || {
    printf '%s\n' 'required executable is not executable' >&2
    exit 1
}
git -C "$GATEWAY_SOURCE" diff --quiet
git -C "$GATEWAY_SOURCE" diff --cached --quiet
git -C "$SQ8_SOURCE" diff --quiet
git -C "$SQ8_SOURCE" diff --cached --quiet
git -C "$GATEWAY_SOURCE" rev-parse HEAD > "$WINDOW/gateway-source-commit.txt"
git -C "$SQ8_SOURCE" rev-parse HEAD > "$WINDOW/sq8-source-commit.txt"
sha256sum "$AQ4_DRIVER" "$DIRECT_MANIFEST" "$GROUPED_MANIFEST" \
    > "$WINDOW/input-sha256.txt"
record_preflight 'pre-window'
event 'preflight-complete'

if ! systemctl is-active --quiet "$SERVICE"; then
    printf '%s\n' "$SERVICE is not active; refusing to claim an unowned window" >&2
    exit 75
fi
if fuser "$LOCK" >/dev/null 2>&1; then
    service_main_pid="$(systemctl show "$SERVICE" -p MainPID --value)"
    lock_owner="$(lslocks -o PID,PATH 2>/dev/null | awk -v path="$LOCK" '$2 == path {print $1; exit}')"
    if [[ -n "$lock_owner" && "$lock_owner" != "$service_main_pid" ]]; then
        printf 'R9700 lock owner %s is not service main PID %s; refusing window\n' \
            "$lock_owner" "$service_main_pid" >&2
        exit 75
    fi
fi

event 'service-stop-attempt'
sudo_systemctl stop "$SERVICE" > "$WINDOW/service-stop.stdout" 2> "$WINDOW/service-stop.stderr"
service_stopped=1
event 'service-stop-returned'
for _ in $(seq 1 30); do
    if ! systemctl is-active --quiet "$SERVICE" && ! pgrep -x ullm-aq4-worker >/dev/null; then
        break
    fi
    sleep 1
done
if systemctl is-active --quiet "$SERVICE" || pgrep -x ullm-aq4-worker >/dev/null; then
    printf '%s\n' 'AQ4_0 service or worker remains after stop; refusing isolated work' >&2
    exit 1
fi
record_preflight 'post-stop'
if fuser "$LOCK" >/dev/null 2>&1; then
    printf '%s\n' 'R9700 lock remains held after owned service stop; refusing to steal it' >&2
    exit 75
fi

# Use the current active manifest's declared guards, so the profile is tied
# to the P3 production contract rather than a duplicated hand-maintained list.
mapfile -t AQ4_GUARDS < <(
    python3 - "$ACTIVE" <<'PY'
import json
import sys

document = json.load(open(sys.argv[1], encoding='utf-8'))
for name in document['worker']['required_environment']:
    print(f'{name}=1')
PY
)
printf '%s\n' "${AQ4_GUARDS[@]}" > "$PHASE1/required-environment.txt"

event 'aq4-rocprof-begin'
exec 9<"$LOCK"
if ! flock -n 9; then
    printf '%s\n' 'R9700 lock became busy before AQ4_0 rocprof; refusing to steal it' >&2
    exit 75
fi
fuser -v "$LOCK" > "$PHASE1/lock-acquired-fuser.txt" 2>&1 || true
mkdir -p "$PHASE1/rocprof"
env \
    -u ROCR_VISIBLE_DEVICES \
    -u ULLM_EXPERIMENTAL_HIP_PAGED_DECODE_SPLIT_TILE \
    -u ULLM_EXPERIMENTAL_HIP_PAGED_DECODE_SPLIT_MIN_CACHE_LEN \
    -u ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE \
    -u ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT \
    -u ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_PIPELINED_SPLIT \
    -u ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE \
    HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 "${AQ4_GUARDS[@]}" \
    rocprofv3 --runtime-trace --stats --output-format csv \
        --output-directory "$PHASE1/rocprof" --output-file aq4-p3-c1339 -- \
        "$AQ4_DRIVER" 1339 --warmup 6 --measured 32 \
        > "$PHASE1/profile.stdout.jsonl" 2> "$PHASE1/profile.stderr.log"
flock -u 9
exec 9>&-
kernel_trace="$(find "$PHASE1/rocprof" -type f -name '*_kernel_trace.csv' -print -quit)"
hip_trace="$(find "$PHASE1/rocprof" -type f -name '*_hip_api_trace.csv' -print -quit)"
marker_trace="$(find "$PHASE1/rocprof" -type f -name '*_marker_api_trace.csv' -print -quit)"
[[ -n "$kernel_trace" && -n "$hip_trace" && -n "$marker_trace" ]] || {
    printf '%s\n' 'rocprof did not produce the expected three trace CSV files' >&2
    exit 1
}
python3 "$AQ4_SUMMARY" \
    --kernel-trace "$kernel_trace" \
    --hip-api-trace "$hip_trace" \
    --marker-trace "$marker_trace" \
    --expected-cache-start 1339 \
    --output "$PHASE1/trace-summary.json"
event 'aq4-rocprof-complete'

run_gateway_capture() {
    local label="$1"
    local manifest="$2"
    local port="$3"
    local output="$4"
    mkdir -p "$output"
    event "${label}-gateway-begin"
    env \
        -u ROCR_VISIBLE_DEVICES \
        -u ULLM_EXPERIMENTAL_HIP_PAGED_DECODE_SPLIT_TILE \
        -u ULLM_EXPERIMENTAL_HIP_PAGED_DECODE_SPLIT_MIN_CACHE_LEN \
        -u ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE \
        -u ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT \
        -u ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_PIPELINED_SPLIT \
        -u ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE \
        PYTHONPATH="$GATEWAY_SOURCE/services/openai-gateway/src" \
        ULLM_SERVED_MODEL_MANIFEST="$manifest" \
        ULLM_API_KEY_FILE="$TOKEN_FILE" \
        ULLM_GPU_LOCK_FILE="$LOCK" \
        ULLM_HIP_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 \
        ULLM_BIND_HOST=127.0.0.1 ULLM_BIND_PORT="$port" \
        "$GATEWAY_PYTHON" -m ullm_openai_gateway \
        > "$output/gateway.stdout.log" 2> "$output/gateway.stderr.log" &
    gateway_pid=$!
    gateway_pid_dir="$output"
    printf '%s\n' "$gateway_pid" > "$output/gateway.pid"
    set +e
    python3 "$CAPTURE_TOOL" \
        --manifest "$manifest" \
        --output-dir "$output/capture" \
        --prompt-suite "$PROMPT_SUITE" \
        --base-url "http://127.0.0.1:${port}" \
        --gateway-container direct \
        --token-file "$TOKEN_FILE" \
        > "$output/capture.stdout.log" 2> "$output/capture.stderr.log"
    local capture_status=$?
    set -e
    stop_gateway "$output" "$gateway_pid"
    gateway_pid=''
    printf '%s\n' "$capture_status" > "$output/capture.exit-status"
    if (( capture_status != 0 )); then
        return "$capture_status"
    fi
    event "${label}-gateway-complete"
}

# This is a physical, isolated read-only regression smoke of the active P3
# worker through the same manifest parser that gained the optional field.
run_gateway_capture 'aq4-p3' "$ACTIVE" 18080 "$AQ4_CAPTURE"
run_gateway_capture 'sq8-direct' "$DIRECT_MANIFEST" 18081 "$DIRECT_CAPTURE"
run_gateway_capture 'sq8-grouped-tile20' "$GROUPED_MANIFEST" 18082 "$GROUPED_CAPTURE"

event 'sq8-comparison-begin'
python3 "$COMPARE_TOOL" \
    --baseline-dir "$DIRECT_CAPTURE/capture/cases" \
    --candidate-dir "$GROUPED_CAPTURE/capture/cases" \
    --output-dir "$COMPARISON" \
    --prompt-suite "$PROMPT_SUITE" \
    > "$WINDOW/sq8-comparison.stdout.log" 2> "$WINDOW/sq8-comparison.stderr.log"
event 'sq8-comparison-complete'
record_preflight 'pre-restore'
exit 0
