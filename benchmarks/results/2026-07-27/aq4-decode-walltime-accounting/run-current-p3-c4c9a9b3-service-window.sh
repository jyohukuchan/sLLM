#!/usr/bin/env bash
# Create one auditable, fail-closed maintenance window for current AQ4_0 capture.
#
# This script never touches active.json.  It stops ullm-openai.service only when
# that service itself owns the R9700 flock, delegates GPU work to the paired
# capture script, then makes exactly one restoration attempt even on failure.
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 OUTPUT_DIRECTORY" >&2
    exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
window_dir="$1"
capture_dir="$window_dir/capture"
service='ullm-openai.service'
llama_service='llama-qwen35-udq4.service'
gpu_lock='/run/ullm/r9700.lock'
active_manifest='/etc/ullm/served-models/active.json'
capture_script="$repo_root/benchmarks/results/2026-07-27/aq4-decode-walltime-accounting/capture-current-p3-c4c9a9b3.sh"

if [[ -e "$window_dir" ]]; then
    echo "refusing to overwrite existing output directory: $window_dir" >&2
    exit 2
fi
if [[ ! -x "$capture_script" || ! -r "$active_manifest" ]]; then
    echo "current AQ4_0 capture inputs are unavailable" >&2
    exit 1
fi

read -r sudo_password
mkdir -p "$window_dir"
service_stopped=0

event() {
    printf '%s\t%s\n' "$(date -Is)" "$1" >>"$window_dir/events.tsv"
}

sudo_systemctl() {
    printf '%s\n' "$sudo_password" | sudo -S -p '' systemctl "$@"
}

record_state() {
    local label="$1"
    fuser -v "$gpu_lock" >"$window_dir/${label}-r9700-lock.txt" 2>&1 || true
    pgrep -af 'ullm-sq8-r9700|run_measurements.py|llama-bench|llama-server|promote-served-model' \
        | awk -v self="$$" '$1 != self {print $1}' \
        >"$window_dir/${label}-gpu-process-pids.txt" || true
    systemctl show "$service" -p ActiveState -p SubState -p MainPID -p NRestarts \
        -p StartLimitBurst -p StartLimitIntervalUSec \
        >"$window_dir/${label}-ullm-openai.txt" 2>&1 || true
    systemctl show "$llama_service" -p ActiveState -p SubState -p MainPID -p UnitFileState \
        >"$window_dir/${label}-llama-qwen35-udq4.txt" 2>&1 || true
    sha256sum "$active_manifest" >"$window_dir/${label}-active-manifest.sha256"
}

restore_service() {
    local status=$?
    local start_status=0
    trap - EXIT INT TERM
    if (( service_stopped )); then
        event 'service-restore-attempt'
        if sudo_systemctl start "$service" >"$window_dir/service-restore.stdout" \
            2>"$window_dir/service-restore.stderr"; then
            printf '%s\n' 0 >"$window_dir/service-restore-first.exit-status"
        else
            start_status=$?
            printf '%s\n' "$start_status" >"$window_dir/service-restore-first.exit-status"
            systemctl show "$service" -p Result >"$window_dir/service-restore-first-result.txt" || true
            if grep -qx 'Result=start-limit-hit' "$window_dir/service-restore-first-result.txt"; then
                # A manual stop does not consume a restart, but a pre-existing
                # rate-limit state can still make the one required restore fail.
                # Clear only that failed unit, then make one final start attempt.
                event 'service-restore-reset-failed-attempt'
                if sudo_systemctl reset-failed "$service" \
                    >"$window_dir/service-restore-reset-failed.stdout" \
                    2>"$window_dir/service-restore-reset-failed.stderr"; then
                    event 'service-restore-reset-failed-complete'
                    if sudo_systemctl start "$service" \
                        >"$window_dir/service-restore-after-reset.stdout" \
                        2>"$window_dir/service-restore-after-reset.stderr"; then
                        start_status=0
                    else
                        start_status=$?
                    fi
                    printf '%s\n' "$start_status" \
                        >"$window_dir/service-restore-after-reset.exit-status"
                else
                    start_status=$?
                    printf '%s\n' "$start_status" \
                        >"$window_dir/service-restore-reset-failed.exit-status"
                fi
            fi
            if (( start_status != 0 )); then
                status=1
            fi
        fi
        for _ in $(seq 1 60); do
            if systemctl is-active --quiet "$service"; then
                event 'service-restore-active'
                break
            fi
            sleep 1
        done
        if ! systemctl is-active --quiet "$service"; then
            event 'service-restore-not-active'
            status=1
        fi
    fi
    record_state 'post-window'
    event "window-finished-status=$status"
    exit "$status"
}
trap restore_service EXIT INT TERM

record_state 'pre-window'
event 'preflight-complete'

if ! grep -qx 'ActiveState=inactive' "$window_dir/pre-window-llama-qwen35-udq4.txt" ||
    ! grep -qx 'UnitFileState=disabled' "$window_dir/pre-window-llama-qwen35-udq4.txt"; then
    echo "llama-qwen35-udq4.service is not inactive and disabled" >&2
    exit 1
fi
if ! systemctl is-active --quiet "$service"; then
    echo "$service is not active; refusing to claim an unowned service window" >&2
    exit 75
fi

service_main_pid="$(systemctl show "$service" -p MainPID --value)"
lock_owner="$(lslocks -n -o PID,PATH 2>/dev/null | awk -v path="$gpu_lock" '$2 == path {print $1; exit}')"
if [[ -z "$lock_owner" || "$lock_owner" != "$service_main_pid" ]]; then
    printf 'R9700 lock owner %s is not service main PID %s; refusing window\n' \
        "${lock_owner:-none}" "$service_main_pid" >&2
    exit 75
fi

event 'service-stop-attempt'
sudo_systemctl stop "$service" >"$window_dir/service-stop.stdout" \
    2>"$window_dir/service-stop.stderr"
service_stopped=1
event 'service-stop-returned'
for _ in $(seq 1 30); do
    if ! systemctl is-active --quiet "$service" &&
        ! pgrep -x ullm-aq4-worker >/dev/null &&
        ! fuser "$gpu_lock" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
record_state 'post-stop'
if systemctl is-active --quiet "$service" || pgrep -x ullm-aq4-worker >/dev/null ||
    fuser "$gpu_lock" >/dev/null 2>&1; then
    echo "service, AQ4_0 worker, or R9700 lock remained after stop; refusing to steal it" >&2
    exit 75
fi

event 'paired-capture-begin'
"$capture_script" "$capture_dir"
event 'paired-capture-complete'
record_state 'pre-restore'
