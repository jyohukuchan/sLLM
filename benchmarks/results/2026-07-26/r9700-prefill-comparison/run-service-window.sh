#!/usr/bin/env bash
# Run only after a credential has been supplied to sudo -v by the operator.
# This script never handles or records the credential itself.
set -Eeuo pipefail

result_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
service_dir="$result_dir/service"
mkdir -p "$service_dir"

record() {
    local label="$1"
    shift
    {
        date --iso-8601=seconds
        printf 'command:'
        printf ' %q' "$@"
        printf '\n'
        "$@"
        rc=$?
        printf 'returncode: %s\n' "$rc"
        return "$rc"
    } >"$service_dir/$label.txt" 2>&1
}

restore_service() {
    local original_status=$?
    trap - EXIT
    set +e
    {
        date --iso-8601=seconds
        printf '%s\n' 'restore command: sudo -n systemctl start ullm-openai.service'
        sudo -n systemctl start ullm-openai.service
        start_rc=$?
        printf 'start_returncode: %s\n' "$start_rc"
        for _ in $(seq 1 60); do
            if systemctl is-active --quiet ullm-openai.service; then
                break
            fi
            sleep 1
        done
        systemctl show ullm-openai.service --property=ActiveState,UnitFileState,SubState,MainPID,NRestarts
        show_rc=$?
        printf 'show_returncode: %s\n' "$show_rc"
        systemctl show llama-qwen35-udq4.service --property=ActiveState,UnitFileState,SubState,MainPID,NRestarts
        llama_rc=$?
        printf 'llama_qwen35_show_returncode: %s\n' "$llama_rc"
        date --iso-8601=seconds
        if ! systemctl is-active --quiet ullm-openai.service; then
            printf '%s\n' 'ERROR: ullm-openai.service did not return active'
            exit 70
        fi
        if [[ "$start_rc" -ne 0 ]]; then
            exit "$start_rc"
        fi
    } >"$service_dir/restore.txt" 2>&1
    restore_rc=$?
    printf '%s\n' "$restore_rc" >"$service_dir/restore.returncode"
    if [[ "$original_status" -ne 0 ]]; then
        exit "$original_status"
    fi
    exit "$restore_rc"
}

if ! sudo -n true; then
    printf '%s\n' 'sudo credential is not cached; prime with the approved command before running this wrapper' >&2
    exit 64
fi

{
    date --iso-8601=seconds
    printf '%s\n' 'window=1'
    printf '%s\n' 'approved sudo credential intentionally not recorded'
} >"$service_dir/window-start.txt"

record pre-stop-state systemctl show ullm-openai.service --property=ActiveState,UnitFileState,SubState,MainPID,NRestarts
record pre-stop-llama-qwen35-state systemctl show llama-qwen35-udq4.service --property=ActiveState,UnitFileState,SubState,MainPID,NRestarts
record pre-stop-r9700-processes amd-smi process --gpu 2 --json || true

if ! grep -qx 'ActiveState=active' "$service_dir/pre-stop-state.txt"; then
    printf '%s\n' 'ullm-openai.service was not active before the requested isolated window' >&2
    exit 65
fi
if ! grep -qx 'ActiveState=inactive' "$service_dir/pre-stop-llama-qwen35-state.txt" || ! grep -qx 'UnitFileState=disabled' "$service_dir/pre-stop-llama-qwen35-state.txt"; then
    printf '%s\n' 'llama-qwen35-udq4.service is not inactive+disabled; refusing to proceed' >&2
    exit 66
fi

trap restore_service EXIT

record stop sudo -n systemctl stop ullm-openai.service
record post-stop-state systemctl show ullm-openai.service --property=ActiveState,UnitFileState,SubState,MainPID,NRestarts
record post-stop-r9700-processes amd-smi process --gpu 2 --json || true

if ! grep -qx 'ActiveState=inactive' "$service_dir/post-stop-state.txt"; then
    printf '%s\n' 'ullm-openai.service did not become inactive' >&2
    exit 67
fi

python3 "$result_dir/run_measurements.py"
