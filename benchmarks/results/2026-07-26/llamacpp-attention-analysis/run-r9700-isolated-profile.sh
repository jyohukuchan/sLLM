#!/usr/bin/env bash
# One bounded R9700-only collection window.  It intentionally never reads or
# writes the served-model manifest, systemd unit files, or /opt/ullm.
set -Eeuo pipefail

RESULT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
SERVICE="ullm-openai.service"
LLAMA_SERVICE="llama-qwen35-udq4.service"
LLAMA_BENCH="/home/homelab1/llama.cpp-src/build-rdna4/bin/llama-bench"
MODEL="/home/homelab1/datapool/ai_models/gguf/Qwen/Qwen3-14B-GGUF-530227a7/Qwen3-14B-Q8_0.gguf"
ROCPROF="/opt/rocm/bin/rocprofv3"
PROFILE_DIR="$RESULT_DIR/rocprof"
LOG="$RESULT_DIR/service-window-events.tsv"

mkdir -p "$PROFILE_DIR"
printf 'timestamp\tevent\tdetail\n' > "$LOG"

event() {
    printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" | tee -a "$LOG"
}

record_service_state() {
    local label="$1"
    systemctl show "$SERVICE" -p ActiveState -p SubState -p UnitFileState -p NRestarts --no-pager > "$RESULT_DIR/$label-ullm-openai.txt"
    systemctl show "$LLAMA_SERVICE" -p ActiveState -p SubState -p UnitFileState -p NRestarts --no-pager > "$RESULT_DIR/$label-llama-qwen35-udq4.txt"
}

service_stopped=0
restore_service() {
    local status=$?
    trap - EXIT
    if [[ "$service_stopped" == "1" ]]; then
        event "restore_begin" "$SERVICE"
        if sudo systemctl start "$SERVICE"; then
            event "restore_ok" "$SERVICE"
        else
            event "restore_failed" "$SERVICE"
            status=1
        fi
        record_service_state "post-restore" || status=1
    fi
    exit "$status"
}
trap restore_service EXIT

event "window_begin" "R9700 gfx1201 only; HIP_VISIBLE_DEVICES=1"
record_service_state "pre-stop"
rocm-smi --showproductname --showuniqueid --showmeminfo vram --showuse > "$RESULT_DIR/pre-stop-rocm-smi.txt" 2>&1

if ! systemctl is-active --quiet "$SERVICE"; then
    event "precondition_failed" "$SERVICE was not active; no state change made"
    exit 1
fi
if systemctl is-active --quiet "$LLAMA_SERVICE"; then
    event "precondition_failed" "$LLAMA_SERVICE is active"
    exit 1
fi
if [[ "$(systemctl is-enabled "$LLAMA_SERVICE" 2>/dev/null || true)" != "disabled" ]]; then
    event "precondition_failed" "$LLAMA_SERVICE is not disabled"
    exit 1
fi

event "stop_begin" "$SERVICE"
sudo systemctl stop "$SERVICE"
service_stopped=1
event "stop_ok" "$SERVICE"
record_service_state "post-stop"

for _ in $(seq 1 30); do
    if ! systemctl is-active --quiet "$SERVICE"; then
        break
    fi
    sleep 1
done
if systemctl is-active --quiet "$SERVICE"; then
    event "isolation_failed" "$SERVICE is still active"
    exit 1
fi
if lsof /dev/dri/renderD129 /dev/kfd > "$RESULT_DIR/post-stop-gpu-fd-holders.txt" 2>&1; then
    event "isolation_failed" "GPU fd holders remained after stop; see post-stop-gpu-fd-holders.txt"
    exit 1
fi
event "isolation_ok" "no /dev/dri/renderD129 (R9700) or /dev/kfd holders"

env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 "$LLAMA_BENCH" --list-devices > "$RESULT_DIR/r9700-only-device-list.txt" 2> "$RESULT_DIR/r9700-only-device-list.stderr"
if ! rg -q 'gfx1201|Radeon AI|R9700' "$RESULT_DIR/r9700-only-device-list.txt" "$RESULT_DIR/r9700-only-device-list.stderr"; then
    event "isolation_failed" "llama-bench device list did not identify gfx1201"
    exit 1
fi
if rg -q 'gfx1030|V620' "$RESULT_DIR/r9700-only-device-list.txt" "$RESULT_DIR/r9700-only-device-list.stderr"; then
    event "isolation_failed" "llama-bench saw a V620"
    exit 1
fi

profile_cmd=(
    "$ROCPROF" --runtime-trace --stats --output-format csv
    --output-directory "$PROFILE_DIR" --output-file llama-q8_0-f16-kv-decode
    --
    /usr/bin/nice -n 19 /usr/bin/ionice -c3
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1
    "$LLAMA_BENCH"
    -m "$MODEL" -o json -r 1 -p 0 -n 16 -d 1028
    -b 128 -ub 128 -ctk f16 -ctv f16 -ngl 999 -sm none -mg 0
    -dev ROCm0 -nkvo 0 -fa on -t 8 -mmp 1 --no-warmup --progress
)
printf '%q ' "${profile_cmd[@]}" | sed 's/ $//' > "$RESULT_DIR/profile-command.txt"
printf '\n' >> "$RESULT_DIR/profile-command.txt"

event "profile_begin" "llama-bench Q8_0 F16 KV, d=1028, n=16, t=8, no warmup"
if "${profile_cmd[@]}" > "$RESULT_DIR/llama-bench.stdout" 2> "$RESULT_DIR/llama-bench.stderr"; then
    event "profile_ok" "rocprofv3 completed"
else
    rc=$?
    printf '%s\n' "$rc" > "$RESULT_DIR/profile-exit-code.txt"
    event "profile_failed" "rocprofv3 exit=$rc"
    exit "$rc"
fi
printf '0\n' > "$RESULT_DIR/profile-exit-code.txt"
rocm-smi --showproductname --showuniqueid --showmeminfo vram --showuse > "$RESULT_DIR/pre-restore-rocm-smi.txt" 2>&1
event "window_profile_complete" "restoration follows via EXIT trap"
