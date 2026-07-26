#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0
#
# One R9700-exclusive validation window for the prompt-length adaptive SQ8_0
# prefill policy.  Every throughput sample times synchronized prefill advances
# only: session.start (including a possible width reconfiguration), model load,
# warm-up, reset, and profiler ranges are outside the timing sum.
set -Eeuo pipefail

root="/home/homelab1/coding-local/ultimateLLM/uLLM-project"
out="$root/benchmarks/results/2026-07-27/prefill-adaptive-chunk"
service="ullm-openai.service"
llama_service="llama-qwen35-udq4.service"
gpu_lock="/run/ullm/r9700.lock"
gpu_index=2
serving="$root/target/release/examples/sq8_ck_serving"
decode="$root/target/release/examples/sq8_0_paged_decode_steady_bench"
artifact="/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact"
package="/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/package"
suite="$root/benchmarks/results/2026-07-27/prefill-chunk-width/generation-input"
long_input="$root/benchmarks/results/2026-07-27/prefill-chunk-width/generation-input-long/long-prefill-p4000.u32le"
oracle_compare="$root/benchmarks/results/2026-07-26/prefill-attention-redesign/compare_oracles.py"
suite_decoder="$root/tools/decode-sq8-serving-lightweight-suite.py"
tokenizer_model="/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8"
tokenizer_python="$root/services/openai-gateway/.venv/bin/python3"
active_manifest="/etc/ullm/served-models/active.json"

service_stopped=0
lock_held=0

event() {
    printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >> "$out/window-events.tsv"
}

sudo_systemctl() {
    printf '%s\n' 'Threadripper' | sudo -S -p '' systemctl "$@"
}

record_preflight() {
    local label="$1"
    fuser -v "$gpu_lock" > "$out/preflight/${label}-fuser.txt" 2>&1 || true
    pgrep -af 'ullm-sq8-r9700|run_measurements.py|llama-bench|promote-served-model|ullm-aq4-|gemma4' \
        > "$out/preflight/${label}-pgrep.txt" || true
    systemctl show "$service" -p ActiveState -p NRestarts \
        > "$out/preflight/${label}-ullm-openai.txt"
}

record_service() {
    local label="$1"
    systemctl show "$service" -p ActiveState -p SubState -p MainPID -p NRestarts \
        -p Result -p StartLimitBurst -p StartLimitIntervalUSec -p UnitFileState \
        > "$out/service/${label}-ullm-openai.txt" 2>&1 || true
    systemctl show "$llama_service" -p ActiveState -p SubState -p MainPID -p UnitFileState \
        > "$out/service/${label}-llama-qwen35-udq4.txt" 2>&1 || true
}

metric() {
    amd-smi metric --gpu "$gpu_index" -t -c -p --json
}

thermal_gate() {
    local condition="$1"
    local path="$out/thermal/${condition}-gate.jsonl"
    local attempt edge
    for attempt in $(seq 1 120); do
        metric | tee -a "$path" > "$out/thermal/${condition}-latest.json"
        edge="$(jq -r '.gpu_data[0].temperature.edge.value // empty' "$out/thermal/${condition}-latest.json")"
        if [[ "$edge" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk "BEGIN { exit !($edge <= 45.0) }"; then
            event "thermal-gate-pass condition=${condition} edge_c=${edge} limit_c=45"
            return 0
        fi
        event "thermal-gate-wait condition=${condition} edge_c=${edge:-unknown} limit_c=45 attempt=${attempt}"
        sleep 5
    done
    echo "thermal gate timed out for ${condition}; edge did not reach <=45 C" >&2
    return 1
}

service_tree_pids() {
    local root_pid="$1"
    local -a queue=("$root_pid")
    local pid child
    while ((${#queue[@]})); do
        pid="${queue[0]}"
        queue=("${queue[@]:1}")
        printf '%s\n' "$pid"
        while IFS= read -r child; do
            [[ -n "$child" ]] && queue+=("$child")
        done < <(ps -o pid= --ppid "$pid" 2>/dev/null | tr -d ' ' || true)
    done
}

lock_is_held_only_by_service() {
    local main_pid holders pid service_pids
    main_pid="$(systemctl show "$service" -p MainPID --value)"
    [[ "$main_pid" =~ ^[1-9][0-9]*$ ]] || return 1
    holders="$(fuser "$gpu_lock" 2>/dev/null || true)"
    [[ -n "$holders" ]] || return 0
    service_pids="$(service_tree_pids "$main_pid")"
    for pid in $holders; do
        if ! grep -qx "$pid" <<<"$service_pids"; then
            echo "R9700 lock holder $pid is outside ${service}'s process tree" >&2
            return 1
        fi
    done
}

run_sq8() {
    env -u ROCR_VISIBLE_DEVICES \
        -u ULLM_KV_CACHE_DTYPE -u ULLM_KV_CACHE_TYPE_K -u ULLM_KV_CACHE_TYPE_V \
        -u ULLM_DISABLE_SQ8_0_FLASH2_GQA_GROUPED \
        -u ULLM_USE_SQ8_0_FLASH2_GQA_GROUPED_PROTOTYPE \
        -u ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE \
        -u ULLM_SQ8_PREFILL_CHUNK_TOKENS \
        HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 \
        ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 ULLM_REQUIRE_HIP_ROPE_KERNEL=1 \
        ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_ADD_KERNEL=1 \
        ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1 \
        ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1 \
        ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL=1 \
        ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL=1 \
        ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1 ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1 \
        ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE=20 \
        ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE=1 \
        ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT=1 \
        "$@"
}

restore() {
    local status=$?
    trap - EXIT
    if ((lock_held)); then
        flock -u 9 || true
        exec 9>&-
        event r9700-lock-released
    fi
    metric > "$out/thermal/after-window.json" 2>&1 || true
    if ((service_stopped)); then
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
        if ! systemctl is-active --quiet "$service"; then
            status=1
            event service-restore-not-active
        fi
    fi
    record_preflight after-restore
    record_service after-restore
    event "window-finished status=${status}"
    exit "$status"
}
trap restore EXIT

cd "$root"
record_preflight before-stop
record_service before-stop
sha256sum "$serving" "$decode" "$active_manifest" "$oracle_compare" "$suite_decoder" \
    > "$out/source/inputs.sha256"
cp -- "$0" "$out/source/run-adaptive-window.sh"
cp -- "$active_manifest" "$out/source/active-manifest-before.json"
metric > "$out/thermal/before-stop.json"

if [[ ! -x "$serving" || ! -x "$decode" || ! -d "$artifact" || ! -d "$package" ]]; then
    echo "required binary or SQ8_0 product path is unavailable" >&2
    exit 1
fi
if [[ ! -x "$tokenizer_python" || ! -f "$long_input" || ! -f "$suite/suite.json" ]]; then
    echo "generation evidence input is unavailable" >&2
    exit 1
fi
if ! grep -qx 'ActiveState=inactive' "$out/service/before-stop-llama-qwen35-udq4.txt" ||
    ! grep -qx 'UnitFileState=disabled' "$out/service/before-stop-llama-qwen35-udq4.txt"; then
    echo "${llama_service} must remain inactive and disabled" >&2
    exit 1
fi
if ! systemctl is-active --quiet "$service"; then
    echo "${service} was not active before this window" >&2
    exit 75
fi
if ! lock_is_held_only_by_service; then
    echo "R9700 lock is not held only by the active gateway; waiting owner must release it" >&2
    exit 75
fi

event service-stop-attempt
sudo_systemctl stop "$service" > "$out/service/stop.stdout" 2> "$out/service/stop.stderr"
service_stopped=1
event service-stop-complete
for attempt in $(seq 1 30); do
    if ! systemctl is-active --quiet "$service" && ! fuser "$gpu_lock" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
record_preflight after-stop-before-lock
record_service after-stop
if systemctl is-active --quiet "$service" || fuser "$gpu_lock" >/dev/null 2>&1; then
    echo "service or R9700 lock remained held after stop; refusing to steal it" >&2
    exit 75
fi
exec 9<>"$gpu_lock"
if ! flock -n 9; then
    echo "R9700 lock became busy after gateway stop; refusing to steal it" >&2
    exit 75
fi
lock_held=1
event r9700-lock-acquired
record_preflight lock-held
metric > "$out/thermal/after-lock.json"

for prompt in 128 512 1024 2048 4095; do
    condition="adaptive-prefill-p${prompt}"
    repeated="${prompt},${prompt},${prompt},${prompt},${prompt},${prompt}"
    thermal_gate "$condition"
    event "${condition}-begin"
    run_sq8 "$serving" --artifact "$artifact" --package "$package" \
        --prompt-lengths "$repeated" --max-new-tokens 1 --prefill-mode adaptive \
        --result-json "$out/throughput/p${prompt}.json" \
        > "$out/throughput/p${prompt}.stdout" 2> "$out/throughput/p${prompt}.stderr"
    event "${condition}-complete"
done

for mode in m128-chunk128 adaptive; do
    label="${mode//-/_}"
    oracle="$out/numerical/${label}-oracle"
    condition="numerical-${label}"
    thermal_gate "$condition"
    event "${condition}-begin"
    run_sq8 "$serving" --artifact "$artifact" --package "$package" \
        --prompt-lengths 128,512,1024,2048,4095 --max-new-tokens 1 --prefill-mode "$mode" \
        --oracle-capture-dir "$oracle" --result-json "$out/numerical/${label}.json" \
        > "$out/numerical/${label}.stdout" 2> "$out/numerical/${label}.stderr"
    event "${condition}-complete"
done
python3 "$oracle_compare" \
    --baseline-result "$out/numerical/m128_chunk128.json" \
    --candidate-result "$out/numerical/adaptive.json" \
    --output "$out/numerical/adaptive-vs-m128.json"

thermal_gate decode-p1024
event decode-p1024-begin
run_sq8 "$decode" --artifact "$artifact" --package "$package" \
    --output "$out/decode/p1024.json" --prompt-tokens 1024 --warmup-steps 4 --measured-steps 16 --repeats 5 \
    > "$out/decode/p1024.stdout" 2> "$out/decode/p1024.stderr"
event decode-p1024-complete

for mode in m128-chunk128 adaptive; do
    label="${mode//-/_}"
    condition="long-generation-${label}"
    thermal_gate "$condition"
    event "${condition}-begin"
    run_sq8 "$serving" --artifact "$artifact" --package "$package" \
        --prompt-token-ids-u32le "$long_input" --max-new-tokens 96 --prefill-mode "$mode" \
        --result-json "$out/generation/long-${label}.json" \
        > "$out/generation/long-${label}.stdout" 2> "$out/generation/long-${label}.stderr"
    event "${condition}-complete"
done

for mode in m128-chunk128 adaptive; do
    label="${mode//-/_}"
    mkdir "$out/generation/${label}"
    while IFS= read -r case_id; do
        max_tokens="$(jq -r --arg case_id "$case_id" '.cases[] | select(.case_id == $case_id) | .max_completion_tokens' "$suite/suite.json")"
        condition="suite-${label}-${case_id}"
        thermal_gate "$condition"
        event "${condition}-begin"
        run_sq8 "$serving" --artifact "$artifact" --package "$package" \
            --prompt-token-ids-u32le "$suite/tokens/${case_id}.u32le" --max-new-tokens "$max_tokens" \
            --prefill-mode "$mode" --result-json "$out/generation/${label}/${case_id}.json" \
            > "$out/generation/${label}/${case_id}.stdout" 2> "$out/generation/${label}/${case_id}.stderr"
        event "${condition}-complete"
    done < <(jq -r '.cases[].case_id' "$suite/suite.json")
done
"$tokenizer_python" "$suite_decoder" --model-dir "$tokenizer_model" --prepared-suite-dir "$suite" \
    --baseline-dir "$out/generation/m128_chunk128" --candidate-dir "$out/generation/adaptive" \
    --output-dir "$out/generation/decoded-comparison"

metric > "$out/thermal/before-restore.json"
event measurements-complete
