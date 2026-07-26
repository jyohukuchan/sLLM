#!/usr/bin/env bash
# Isolated BH grouped-tile-20 decode regression against the current HEAD.
set -Eeuo pipefail

root=/home/homelab1/coding-local/ultimateLLM/uLLM-project
result=$root/benchmarks/results/2026-07-26/prefill-attention-redesign
service=ullm-openai.service
llama_service=llama-qwen35-udq4.service
gpu_lock=/run/ullm/r9700.lock
driver=/tmp/ullm-br-main-driver-build/release/ullm-sq8-r9700-phase0-profile
out=$result/raw/current-head-bh-decode-regression
events=$result/service/current-head-bh-decode-regression-window-events.tsv

read -r sudo_password
mkdir -p "$out" "$result/service"
if [[ -e "$events" ]]; then
    echo "refusing to append a second current-head BH decode window" >&2
    exit 1
fi
event() { printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >> "$events"; }
sudo_systemctl() { printf '%s\n' "$sudo_password" | sudo -S -p '' systemctl "$@"; }
record_service() {
    local label=$1
    systemctl show "$service" -p ActiveState -p SubState -p MainPID -p NRestarts \
        -p StartLimitBurst -p StartLimitIntervalUSec -p UnitFileState \
        > "$result/service/current-head-bh-decode-$label-ullm-openai.txt" 2>&1 || true
    systemctl show "$llama_service" -p ActiveState -p SubState -p MainPID -p UnitFileState \
        > "$result/service/current-head-bh-decode-$label-llama-qwen35-udq4.txt" 2>&1 || true
}
service_stopped=0
lock_held=0
release_lock() {
    if (( lock_held )); then
        flock -u 9 2>/dev/null || true
        exec 9>&-
        lock_held=0
        event r9700-lock-released
    fi
}
restore_service() {
    local status=$?
    trap - EXIT
    release_lock
    if (( service_stopped )); then
        event restore-attempt
        if sudo_systemctl start "$service" \
            > "$result/service/current-head-bh-decode-restore.stdout" \
            2> "$result/service/current-head-bh-decode-restore.stderr"; then
            event restore-start-return=0
        else
            status=1
            event restore-start-return=nonzero
        fi
        for delay in 1 2 4 8 16; do
            if systemctl is-active --quiet "$service"; then
                event restore-active
                break
            fi
            sleep "$delay"
        done
    fi
    record_service post-restore
    amd-smi metric --gpu 2 --json \
        > "$result/service/current-head-bh-decode-r9700-metric-post-restore.json" 2>&1 || true
    sha256sum /etc/ullm/served-models/active.json \
        > "$result/service/current-head-bh-decode-active-manifest-post-restore.sha256" 2>&1 || true
    event window-finished
    exit "$status"
}
trap restore_service EXIT

event preflight-begin
fuser -v "$gpu_lock" > "$result/service/current-head-bh-decode-pre-window-fuser.txt" 2>&1 || true
pgrep -af 'ullm-sq8-r9700|run_measurements.py|llama-bench|llama-server|promote-served-model' \
    > "$result/service/current-head-bh-decode-pre-window-pgrep.txt" || true
record_service pre-window
amd-smi static --gpu 2 --json > "$result/service/current-head-bh-decode-r9700-static.json" 2>&1 || true
amd-smi metric --gpu 2 --json > "$result/service/current-head-bh-decode-r9700-metric-pre-stop.json" 2>&1 || true
sha256sum /etc/ullm/served-models/active.json \
    > "$result/service/current-head-bh-decode-active-manifest-pre-stop.sha256"
sha256sum "$driver" > "$result/service/current-head-bh-decode-driver.sha256"
git -C "$root" rev-parse HEAD > "$result/service/current-head-bh-decode-git-head.txt"

if [[ ! -x "$driver" ]]; then
    echo "current-head driver is absent" >&2
    exit 1
fi
if ! grep -qx 'ActiveState=inactive' "$result/service/current-head-bh-decode-pre-window-llama-qwen35-udq4.txt" ||
    ! grep -qx 'UnitFileState=disabled' "$result/service/current-head-bh-decode-pre-window-llama-qwen35-udq4.txt"; then
    echo "llama service is not inactive+disabled" >&2
    exit 1
fi
if ! systemctl is-active --quiet "$service"; then
    echo "uLLM service was not active before this owned window" >&2
    exit 1
fi

event stop-attempt
sudo_systemctl stop "$service" > "$result/service/current-head-bh-decode-stop.stdout" \
    2> "$result/service/current-head-bh-decode-stop.stderr"
service_stopped=1
event stop-complete
record_service post-stop
for attempt in $(seq 1 30); do
    if ! systemctl is-active --quiet "$service" &&
        ! pgrep -x ullm-aq4-worker >/dev/null &&
        ! fuser "$gpu_lock" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
if systemctl is-active --quiet "$service" || pgrep -x ullm-aq4-worker >/dev/null ||
    fuser "$gpu_lock" >/dev/null 2>&1; then
    echo "service or R9700 lock remained held after stop; refusing to steal it" >&2
    exit 75
fi
exec 9<"$gpu_lock"
if ! flock -n 9; then
    echo "R9700 lock was acquired by another process; refusing to steal it" >&2
    exec 9>&-
    exit 75
fi
lock_held=1
event window-begin-lock-acquired-after-service-stop
fuser -v "$gpu_lock" > "$result/service/current-head-bh-decode-post-acquire-fuser.txt" 2>&1 || true

event decode-bh-grouped-tile20-begin
env -u ROCR_VISIBLE_DEVICES -u ULLM_DISABLE_SQ8_0_FLASH2_GQA_GROUPED \
    -u ULLM_USE_SQ8_0_FLASH2_GQA_GROUPED_PROTOTYPE \
    -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE \
    HIP_VISIBLE_DEVICES=1 \
    ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 ULLM_REQUIRE_HIP_ROPE_KERNEL=1 \
    ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_ADD_KERNEL=1 \
    ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1 \
    ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1 \
    ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL=1 \
    ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1 ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1 \
    ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE=20 \
    ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE=1 \
    ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT=1 \
    "$driver" --phase decode --prompt-tokens 1024 --repeats 5 \
        --warmup-steps 4 --measured-steps 16 \
    > "$out/stdout.log" 2> "$out/stderr.log"
event decode-bh-grouped-tile20-complete
amd-smi metric --gpu 2 --json > "$out/amd-smi-metric-after.json" 2>&1 || true
event measurement-complete
