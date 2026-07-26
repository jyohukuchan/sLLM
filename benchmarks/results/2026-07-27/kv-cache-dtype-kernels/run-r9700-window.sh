#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0
#
# One owned R9700 window for KV-cache dtype validation.  This script keeps the
# service stop/restore, lock, thermal samples, and every full-model invocation
# in one auditable transaction.  It never changes an active manifest or starts
# llama-qwen35-udq4.service.
set -Eeuo pipefail

root="/home/homelab1/coding-local/ultimateLLM/uLLM-project"
result_root="$root/benchmarks/results/2026-07-27/kv-cache-dtype-kernels"
run_id="run-$(date +%Y%m%dT%H%M%S%z)"
out="$result_root/$run_id"
service="ullm-openai.service"
llama_service="llama-qwen35-udq4.service"
gpu_lock="/run/ullm/r9700.lock"
gpu_index=2
active_manifest="/etc/ullm/served-models/active.json"
aq4_binary="$root/target/release/ullm-aq4-kv-cache-dtype-measure"
sq8_driver="/tmp/ullm-sq8-r9700-phase0-profile-20260726/target/release/ullm-sq8-r9700-phase0-profile"
sq8_serving="$root/target/release/examples/sq8_ck_serving"
sq8_artifact="/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact"
sq8_package="/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/package"
sq8_oracle="$root/benchmarks/results/2026-07-26/prefill-attention-redesign/numerical/serial-gqa/oracle"

mkdir -p "$out" "$out/service" "$out/thermal" "$out/aq4" "$out/sq8-f32"
cp -- "$0" "$out/run-r9700-window.sh"
sha256sum "$aq4_binary" "$sq8_driver" "$sq8_serving" "$active_manifest" > "$out/input-sha256-before.txt"

event() {
    printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >> "$out/window-events.tsv"
}

sudo_systemctl() {
    printf '%s\n' 'Threadripper' | sudo -S -p '' systemctl "$@"
}

record_required_preflight() {
    fuser -v /run/ullm/r9700.lock > "$out/service/$1-fuser.txt" 2>&1 || true
    pgrep -af 'ullm-sq8-r9700|run_measurements.py|llama-bench|promote-served-model' \
        > "$out/service/$1-pgrep.txt" || true
    systemctl show ullm-openai.service -p ActiveState -p NRestarts \
        > "$out/service/$1-ullm-openai.txt"
}

record_service() {
    systemctl show "$service" -p ActiveState -p SubState -p MainPID -p NRestarts \
        -p Result -p StartLimitBurst -p StartLimitIntervalUSec -p UnitFileState \
        > "$out/service/$1-ullm-openai-full.txt" 2>&1 || true
    systemctl show "$llama_service" -p ActiveState -p SubState -p MainPID -p UnitFileState \
        > "$out/service/$1-llama-qwen35-udq4.txt" 2>&1 || true
}

metric() {
    amd-smi metric --gpu "$gpu_index" -t -c -p --json
}

thermal_gate() {
    local condition="$1"
    local path="$out/thermal/${condition}-gate.jsonl"
    local attempt edge
    for attempt in $(seq 1 180); do
        metric | tee -a "$path" > "$out/thermal/${condition}-latest.json"
        edge="$(jq -r '.gpu_data[0].temperature.edge.value // empty' "$out/thermal/${condition}-latest.json")"
        if [[ "$edge" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk "BEGIN { exit !($edge <= 45.0) }"; then
            event "thermal-gate-pass condition=${condition} edge_c=${edge} limit_c=45"
            return 0
        fi
        event "thermal-gate-wait condition=${condition} edge_c=${edge:-unknown} limit_c=45 attempt=${attempt}"
        sleep 5
    done
    echo "thermal gate timed out for ${condition}; edge never reached <=45 C" >&2
    return 1
}

service_stopped=0
lock_held=0
restore() {
    local status=$?
    trap - EXIT
    if (( lock_held )); then
        flock -u 9 || true
        exec 9>&-
        event r9700-lock-released
    fi
    metric > "$out/thermal/after-window.json" 2>&1 || true
    if (( service_stopped )); then
        event service-restore-attempt
        if sudo_systemctl start "$service" > "$out/service/restore.stdout" 2> "$out/service/restore.stderr"; then
            event service-restore-start-return=0
        else
            status=1
            event service-restore-start-return=nonzero
        fi
        for delay in 1 2 4 8 16 24; do
            if systemctl is-active --quiet "$service"; then
                event service-restore-active
                break
            fi
            sleep "$delay"
        done
    fi
    record_required_preflight after-restore
    record_service after-restore
    sha256sum "$aq4_binary" "$sq8_driver" "$sq8_serving" "$active_manifest" \
        > "$out/input-sha256-after.txt" || true
    if ! cmp -s "$out/input-sha256-before.txt" "$out/input-sha256-after.txt"; then
        event input-identity-changed-during-window
        status=1
    fi
    event "window-finished status=${status}"
    exit "$status"
}
trap restore EXIT

event preflight-begin
record_required_preflight before-stop
record_service before-stop
amd-smi static --gpu "$gpu_index" --json > "$out/service/r9700-static.json"
amd-smi process --gpu "$gpu_index" --json > "$out/service/r9700-process-before-stop.json" || true
metric > "$out/thermal/before-stop.json"

if ! grep -qx 'ActiveState=inactive' "$out/service/before-stop-llama-qwen35-udq4.txt" ||
    ! grep -qx 'UnitFileState=disabled' "$out/service/before-stop-llama-qwen35-udq4.txt"; then
    echo "${llama_service} must remain inactive and disabled" >&2
    exit 1
fi
if [[ ! -x "$aq4_binary" || ! -x "$sq8_driver" || ! -x "$sq8_serving" ]]; then
    echo "one or more current-source measurement binaries are missing" >&2
    exit 1
fi
if ! systemctl is-active --quiet "$service"; then
    echo "${service} was not active before this owned window" >&2
    exit 75
fi

event service-stop-attempt
sudo_systemctl stop "$service" > "$out/service/stop.stdout" 2> "$out/service/stop.stderr"
service_stopped=1
event service-stop-complete
record_service after-stop
for attempt in $(seq 1 30); do
    if ! systemctl is-active --quiet "$service" &&
        ! pgrep -x ullm-aq4-worker >/dev/null &&
        ! fuser "$gpu_lock" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
record_required_preflight after-stop-before-lock
if systemctl is-active --quiet "$service" || pgrep -x ullm-aq4-worker >/dev/null ||
    fuser "$gpu_lock" >/dev/null 2>&1; then
    echo "service or R9700 lock remained held after stop; refusing to steal it" >&2
    exit 75
fi
exec 9<>"$gpu_lock"
if ! flock -n 9; then
    echo "R9700 lock was acquired after service stop; refusing to steal it" >&2
    exec 9>&-
    exit 75
fi
lock_held=1
event r9700-lock-acquired
record_required_preflight lock-held
amd-smi process --gpu "$gpu_index" --json > "$out/service/r9700-process-lock-held.json" || true

export HIP_VISIBLE_DEVICES=1
export ULLM_HIP_VISIBLE_DEVICES=1
while IFS= read -r required_name; do
    export "$required_name=1"
done < <(jq -r '.worker.required_environment[]' "$active_manifest")
env | rg '^(HIP_VISIBLE_DEVICES|ULLM_HIP_VISIBLE_DEVICES|ULLM_REQUIRE_HIP_)=' | sort \
    > "$out/aq4/required-environment.txt"

run_aq4() {
    local label="$1"
    shift
    local dtype_env=()
    if [[ "$label" == "f16" ]]; then
        dtype_env=(ULLM_KV_CACHE_DTYPE=f16 ULLM_REQUIRE_HIP_TYPED_PAGED_DECODE_KERNEL=1 ULLM_REQUIRE_HIP_TYPED_PAGED_DECODE_SPLIT_KERNEL=1 ULLM_REQUIRE_HIP_TYPED_PAGED_KV_WRITE_KERNEL=1)
    elif [[ "$label" == "fp8_e4m3fn" ]]; then
        dtype_env=(ULLM_KV_CACHE_DTYPE=fp8_e4m3fn ULLM_REQUIRE_HIP_TYPED_PAGED_DECODE_KERNEL=1 ULLM_REQUIRE_HIP_TYPED_PAGED_DECODE_SPLIT_KERNEL=1 ULLM_REQUIRE_HIP_TYPED_PAGED_KV_WRITE_KERNEL=1)
    fi
    env -u ULLM_KV_CACHE_DTYPE -u ULLM_KV_CACHE_TYPE_K -u ULLM_KV_CACHE_TYPE_V \
        -u ULLM_REQUIRE_HIP_TYPED_PAGED_DECODE_KERNEL \
        -u ULLM_REQUIRE_HIP_TYPED_PAGED_DECODE_SPLIT_KERNEL \
        -u ULLM_REQUIRE_HIP_TYPED_PAGED_KV_WRITE_KERNEL \
        "${dtype_env[@]}" "$aq4_binary" "$@"
}

for pair in "f32 4096" "f16 8192" "fp8_e4m3fn 16256"; do
    read -r dtype context <<< "$pair"
    thermal_gate "capacity-${dtype}"
    event "aq4-capacity-begin dtype=${dtype} context=${context}"
    run_aq4 "$dtype" --mode capacity --context-length "$context" \
        --output "$out/aq4/capacity-${dtype}.json" \
        > "$out/aq4/capacity-${dtype}.stdout" 2> "$out/aq4/capacity-${dtype}.stderr"
    event "aq4-capacity-complete dtype=${dtype} context=${context}"
done

for dtype in f32 f16 fp8_e4m3fn; do
    for tokens in 128 512 1024 2048 4095; do
        thermal_gate "prefill-${dtype}-${tokens}"
        event "aq4-prefill-begin dtype=${dtype} tokens=${tokens}"
        run_aq4 "$dtype" --mode prefill --context-length 4096 --token-count "$tokens" \
            --prefill-width 128 --warmup 1 --repeats 5 \
            --output "$out/aq4/prefill-${dtype}-p${tokens}.json" \
            > "$out/aq4/prefill-${dtype}-p${tokens}.stdout" \
            2> "$out/aq4/prefill-${dtype}-p${tokens}.stderr"
        event "aq4-prefill-complete dtype=${dtype} tokens=${tokens}"
    done
done

for dtype in f32 f16 fp8_e4m3fn; do
    thermal_gate "decode-${dtype}"
    event "aq4-decode-begin dtype=${dtype}"
    run_aq4 "$dtype" --mode decode --context-length 4096 --prefix-tokens 3968 \
        --generated-tokens 128 --prefill-width 128 --warmup 1 --repeats 5 \
        --output "$out/aq4/decode-${dtype}-c4096.json" \
        > "$out/aq4/decode-${dtype}-c4096.stdout" \
        2> "$out/aq4/decode-${dtype}-c4096.stderr"
    event "aq4-decode-complete dtype=${dtype}"
done

python3 "$result_root/prepare-quality-prompt.py" --output-dir "$out/quality-input" --target-tokens 3968
for dtype in f32 f16 fp8_e4m3fn; do
    thermal_gate "quality-${dtype}"
    event "aq4-quality-generate-begin dtype=${dtype}"
    run_aq4 "$dtype" --mode generate --context-length 4096 --generated-tokens 64 \
        --prefill-width 128 --token-ids-file "$out/quality-input/quality-input-token-ids.json" \
        --output "$out/aq4/quality-${dtype}.json" \
        > "$out/aq4/quality-${dtype}.stdout" 2> "$out/aq4/quality-${dtype}.stderr"
    event "aq4-quality-generate-complete dtype=${dtype}"
done
python3 "$result_root/decode-quality-output.py" \
    --f32 "$out/aq4/quality-f32.json" --f16 "$out/aq4/quality-f16.json" \
    --fp8 "$out/aq4/quality-fp8_e4m3fn.json" --output "$out/aq4/quality-side-by-side.json"

# The native source additions compile into the SQ8_0 HIPRTC modules too. This
# current-source regression reruns BH/BR's full model F32 control and compares
# every retained hidden/logit byte stream to its serial-GQA oracle.
thermal_gate sq8-f32-prefill-oracle
event sq8-f32-prefill-oracle-begin
env -u ROCR_VISIBLE_DEVICES -u ULLM_KV_CACHE_DTYPE -u ULLM_KV_CACHE_TYPE_K -u ULLM_KV_CACHE_TYPE_V \
    -u ULLM_DISABLE_SQ8_0_FLASH2_GQA_GROUPED -u ULLM_USE_SQ8_0_FLASH2_GQA_GROUPED_PROTOTYPE \
    -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE HIP_VISIBLE_DEVICES=1 \
    ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 ULLM_REQUIRE_HIP_ROPE_KERNEL=1 \
    ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_ADD_KERNEL=1 \
    ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1 \
    ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1 \
    ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL=1 \
    ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1 ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1 \
    ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE=20 \
    ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE=1 \
    ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT=1 \
    "$sq8_serving" --artifact "$sq8_artifact" --package "$sq8_package" \
    --prompt-lengths 128,512,1024,2048,4095 --max-new-tokens 1 --prefill-mode m128-chunk128 \
    --oracle-capture-dir "$out/sq8-f32/oracle" --result-json "$out/sq8-f32/result.json" \
    > "$out/sq8-f32/prefill.stdout" 2> "$out/sq8-f32/prefill.stderr"
for source in "$sq8_oracle"/*; do
    name="$(basename "$source")"
    if cmp -s "$source" "$out/sq8-f32/oracle/$name"; then
        printf 'match\t%s\n' "$name"
    else
        printf 'DIFFER\t%s\n' "$name"
        exit 1
    fi
done > "$out/sq8-f32/f32-byte-comparison.tsv"
event sq8-f32-prefill-oracle-complete

thermal_gate sq8-f32-decode
event sq8-f32-decode-begin
env -u ROCR_VISIBLE_DEVICES -u ULLM_KV_CACHE_DTYPE -u ULLM_KV_CACHE_TYPE_K -u ULLM_KV_CACHE_TYPE_V \
    -u ULLM_DISABLE_SQ8_0_FLASH2_GQA_GROUPED -u ULLM_USE_SQ8_0_FLASH2_GQA_GROUPED_PROTOTYPE \
    -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE HIP_VISIBLE_DEVICES=1 \
    ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 ULLM_REQUIRE_HIP_ROPE_KERNEL=1 \
    ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_ADD_KERNEL=1 \
    ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1 \
    ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1 \
    ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL=1 \
    ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1 ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1 \
    ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE=20 \
    ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE=1 \
    ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT=1 \
    "$sq8_driver" --phase decode --prompt-tokens 1024 --repeats 5 --warmup-steps 4 --measured-steps 16 \
    > "$out/sq8-f32/decode.jsonl" 2> "$out/sq8-f32/decode.stderr"
event sq8-f32-decode-complete

metric > "$out/thermal/before-restore.json"
event measurements-complete
