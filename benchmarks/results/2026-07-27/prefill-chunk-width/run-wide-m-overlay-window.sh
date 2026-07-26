#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0
#
# One owned R9700 window for the isolated SQ8_0 wide-prefill-M overlay.  The
# main checkout intentionally does not contain the lower-runtime admission
# changes while BP/BX own those files.  This script only exercises the
# separately-built overlay and restores the gateway on every exit path.
set -Eeuo pipefail

root="/home/homelab1/coding-local/ultimateLLM/uLLM-project"
result_root="$root/benchmarks/results/2026-07-27/prefill-chunk-width"
overlay_root="/tmp/ullm-sq8-wide-m-overlay.9JkzMM"
overlay_target="/tmp/ullm-sq8-wide-m-target"
run_id="run-$(date +%Y%m%dT%H%M%S%z)"
out="$result_root/$run_id"
service="ullm-openai.service"
llama_service="llama-qwen35-udq4.service"
gpu_lock="/run/ullm/r9700.lock"
gpu_index=2
active_manifest="/etc/ullm/served-models/active.json"
driver="$overlay_target/release/examples/sq8_ck_wide_m_prefill_driver"
serving="$overlay_target/release/examples/sq8_ck_serving"
artifact="/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact"
package="/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/package"
tokenizer_model="/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8"
generation_input="$result_root/generation-input"
long_generation_input="$result_root/generation-input-long"
oracle_compare="$root/benchmarks/results/2026-07-26/prefill-attention-redesign/compare_oracles.py"
trace_analyzer="$root/benchmarks/results/2026-07-26/prefill-attention-redesign/analyze_kernel_trace.py"
summary_tool="$result_root/summarize-wide-m-overlay.py"
generation_summary="$result_root/summarize-wide-m-generation.py"
long_generation_summary="$result_root/summarize-long-real-token-generation.py"
tokenizer_python="$root/services/openai-gateway/.venv/bin/python3"
rocprof="/opt/rocm/bin/rocprofv3"
phase="${ULLM_SQ8_WIDE_M_PHASES:-all}"

# M=4096 fits the allocation arithmetic, but N=4095 cannot execute its
# cached-prefix path without inventing a 4096th token.  The real-token tail
# contract therefore makes M=2048 the largest useful width in this grid.
widths=(128 256 512 1024 2048)
prompts=(128 512 1024 2048 4095)

if [[ "$phase" != "all" && "$phase" != "validation" ]]; then
    echo "ULLM_SQ8_WIDE_M_PHASES must be all or validation, got: $phase" >&2
    exit 2
fi

mkdir "$out"
mkdir -p "$out/service" "$out/thermal" "$out/throughput" "$out/traces" \
    "$out/numerical" "$out/decode" "$out/generation"
cp -- "$0" "$out/run-wide-m-overlay-window.sh"

event() {
    printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >> "$out/window-events.tsv"
}

sudo_systemctl() {
    printf '%s\n' 'Threadripper' | sudo -S -p '' systemctl "$@"
}

record_required_preflight() {
    local label="$1"
    fuser -v "$gpu_lock" > "$out/service/${label}-fuser.txt" 2>&1 || true
    pgrep -af 'ullm-sq8-r9700|run_measurements.py|llama-bench|promote-served-model' \
        > "$out/service/${label}-pgrep.txt" || true
    systemctl show "$service" -p ActiveState -p NRestarts \
        > "$out/service/${label}-ullm-openai.txt"
}

record_service() {
    local label="$1"
    systemctl show "$service" -p ActiveState -p SubState -p MainPID -p NRestarts \
        -p Result -p StartLimitBurst -p StartLimitIntervalUSec -p UnitFileState \
        > "$out/service/${label}-ullm-openai-full.txt" 2>&1 || true
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
    local main_pid holders pid
    main_pid="$(systemctl show "$service" -p MainPID --value)"
    [[ "$main_pid" =~ ^[1-9][0-9]*$ ]] || return 1
    holders="$(fuser "$gpu_lock" 2>/dev/null || true)"
    [[ -n "$holders" ]] || return 0
    local service_pids
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

service_stopped=0
lock_held=0
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
    record_required_preflight after-restore
    record_service after-restore
    sha256sum "$driver" "$serving" "$active_manifest" "$oracle_compare" "$trace_analyzer" \
        "$summary_tool" "$generation_summary" "$long_generation_summary" \
        "$long_generation_input/manifest.json" "$long_generation_input/long-prefill-p4000.u32le" \
        > "$out/input-sha256-after.txt" || true
    if ! cmp -s "$out/input-sha256-before.txt" "$out/input-sha256-after.txt"; then
        event input-identity-changed-during-window
        status=1
    fi
    event "window-finished status=${status}"
    exit "$status"
}
trap restore EXIT

cd "$root"
event preflight-begin
event "phase=${phase}"
record_required_preflight before-stop
record_service before-stop
amd-smi static --gpu "$gpu_index" --json > "$out/service/r9700-static.json"
amd-smi process --gpu "$gpu_index" --json > "$out/service/r9700-process-before-stop.json" || true
metric > "$out/thermal/before-stop.json"
sha256sum "$driver" "$serving" "$active_manifest" "$oracle_compare" "$trace_analyzer" \
    "$summary_tool" "$generation_summary" "$long_generation_summary" \
    "$long_generation_input/manifest.json" "$long_generation_input/long-prefill-p4000.u32le" \
    > "$out/input-sha256-before.txt"

if [[ ! -d "$overlay_root" || ! -x "$driver" || ! -x "$serving" ]]; then
    echo "wide-M overlay source or executables are missing" >&2
    exit 1
fi
if [[ ! -d "$artifact" || ! -d "$package" || ! -d "$tokenizer_model" || ! -f "$generation_input/suite.json" ||
    ! -f "$long_generation_input/manifest.json" || ! -f "$long_generation_input/long-prefill-p4000.u32le" ]]; then
    echo "SQ8_0 artifact/package/tokenizer or generation input is missing" >&2
    exit 1
fi
if [[ ! -x "$tokenizer_python" || ! -x "$rocprof" ]]; then
    echo "tokenizer Python or rocprofv3 is missing" >&2
    exit 1
fi
if ! grep -qx 'ActiveState=inactive' "$out/service/before-stop-llama-qwen35-udq4.txt" ||
    ! grep -qx 'UnitFileState=disabled' "$out/service/before-stop-llama-qwen35-udq4.txt"; then
    echo "${llama_service} must remain inactive and disabled" >&2
    exit 1
fi
if ! systemctl is-active --quiet "$service"; then
    echo "${service} was not active before this owned window; refusing to inherit another owner's inactive window" >&2
    exit 75
fi
if ! lock_is_held_only_by_service; then
    echo "R9700 lock is not owned only by the active gateway; refusing to stop the service" >&2
    exit 75
fi
# The required pgrep output is preserved above for audit.  Do not infer an
# active benchmark from it: Codex/Claude wrapper command lines can contain
# the literal search terms from their task prompts.  The authoritative
# exclusion is the lock-owner process-tree check immediately above, followed
# by a post-stop empty-lock check before flock acquisition.

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

if [[ "$phase" == "all" ]]; then
    for width in "${widths[@]}"; do
        for prompt in "${prompts[@]}"; do
            condition="throughput-m${width}-p${prompt}"
            thermal_gate "$condition"
            event "${condition}-begin"
            run_sq8 "$driver" --phase prefill --prompt-tokens "$prompt" \
                --chunk-tokens "$width" --repeats 5 \
                > "$out/throughput/m${width}-p${prompt}.jsonl" \
                2> "$out/throughput/m${width}-p${prompt}.stderr"
            event "${condition}-complete"
        done
    done

    for width in "${widths[@]}"; do
        condition="trace-m${width}-p4095"
        trace_dir="$out/traces/m${width}-p4095"
        mkdir -p "$trace_dir/rocprof"
        thermal_gate "$condition"
        event "${condition}-begin"
        run_sq8 "$rocprof" --runtime-trace --stats --selected-regions --output-format csv \
            --output-directory "$trace_dir/rocprof" --output-file "m${width}-p4095" -- \
            "$driver" --phase prefill --prompt-tokens 4095 --chunk-tokens "$width" --repeats 1 \
            > "$trace_dir/stdout.log" 2> "$trace_dir/stderr.log"
        python3 "$trace_analyzer" \
            --kernel-trace "$trace_dir/rocprof/m${width}-p4095_kernel_trace.csv" \
            --output "$out/traces/m${width}-p4095-analysis.json" \
            --label "wide-m-overlay-m${width}-p4095"
        event "${condition}-complete"
    done
fi

for width in "${widths[@]}"; do
    condition="numerical-m${width}"
    numerical_dir="$out/numerical/m${width}"
    mkdir -p "$numerical_dir"
    thermal_gate "$condition"
    event "${condition}-begin"
    run_sq8 "$serving" --artifact "$artifact" --package "$package" \
        --prompt-lengths 128,512,1024,2048,4095 --max-new-tokens 1 \
        --prefill-mode "m${width}-chunk${width}" \
        --oracle-capture-dir "$numerical_dir/oracle" \
        --result-json "$numerical_dir/result.json" \
        > "$numerical_dir/stdout.log" 2> "$numerical_dir/stderr.log"
    event "${condition}-complete"
done
for width in 256 512 1024 2048; do
    python3 "$oracle_compare" \
        --baseline-result "$out/numerical/m128/result.json" \
        --candidate-result "$out/numerical/m${width}/result.json" \
        --output "$out/numerical/m${width}-vs-m128.json"
done

thermal_gate decode-m128-p1024
event decode-m128-p1024-begin
run_sq8 "$driver" --phase decode --prompt-tokens 1024 --chunk-tokens 128 \
    --warmup-steps 4 --measured-steps 16 --repeats 5 \
    > "$out/decode/m128-p1024.jsonl" 2> "$out/decode/m128-p1024.stderr"
event decode-m128-p1024-complete

# Always retain direct text for the baseline and the largest useful M.  Add
# every numerically non-bit-identical candidate automatically, so a changed
# reduction cannot hide behind a single representative generation.
generation_widths=(128 2048)
for width in 256 512 1024 2048; do
    if jq -e '[.comparisons[] | select(
        (.final_hidden.exact_f32_le_bytes | not) or
        (.logits.exact_f32_le_bytes | not) or
        (.generated_token_ids.exact | not)
    )] | length > 0' "$out/numerical/m${width}-vs-m128.json" > /dev/null; then
        if [[ " ${generation_widths[*]} " != *" ${width} "* ]]; then
            generation_widths+=("$width")
        fi
    fi
done
printf '%s\n' "${generation_widths[@]}" | jq -R . | jq -s \
    '{schema_version: "ullm.sq8.prefill_chunk_width.generation_selection.v1", widths: ., reason: "M=128 baseline plus M=2048 and every numerical non-bit-identical candidate"}' \
    > "$out/generation/selected-widths.json"

for width in "${generation_widths[@]}"; do
    for case_id in $(jq -r '.cases[].case_id' "$generation_input/suite.json"); do
        max_tokens="$(jq -r --arg case_id "$case_id" '.cases[] | select(.case_id == $case_id) | .max_completion_tokens' "$generation_input/suite.json")"
        condition="generation-m${width}-${case_id}"
        generation_dir="$out/generation/m${width}"
        mkdir -p "$generation_dir"
        thermal_gate "$condition"
        event "${condition}-begin"
        run_sq8 "$serving" --artifact "$artifact" --package "$package" \
            --prompt-token-ids-u32le "$generation_input/tokens/${case_id}.u32le" \
            --max-new-tokens "$max_tokens" --prefill-mode "m${width}-chunk${width}" \
            --result-json "$generation_dir/${case_id}.json" \
            > "$generation_dir/${case_id}.stdout" 2> "$generation_dir/${case_id}.stderr"
        event "${condition}-complete"
    done
done
"$tokenizer_python" "$generation_summary" --run-root "$out" \
    --generation-input "$generation_input" --model-dir "$tokenizer_model" \
    --output-json "$out/generation/summary.json" \
    --output-markdown "$out/generation/summary.md"

# The fixed policy suite has short prompts and therefore reaches M=1 under
# wide residents.  This additional N=4000 real-token run exercises each M's
# actual prefill schedule while retaining a genuine final chat generation
# header and enough context room for 96 decoded tokens.
for width in "${widths[@]}"; do
    condition="generation-long-m${width}-p4000"
    generation_dir="$out/generation-long/m${width}"
    mkdir -p "$generation_dir"
    thermal_gate "$condition"
    event "${condition}-begin"
    run_sq8 "$serving" --artifact "$artifact" --package "$package" \
        --prompt-token-ids-u32le "$long_generation_input/long-prefill-p4000.u32le" \
        --max-new-tokens 96 --prefill-mode "m${width}-chunk${width}" \
        --result-json "$generation_dir/result.json" \
        > "$generation_dir/stdout.log" 2> "$generation_dir/stderr.log"
    event "${condition}-complete"
done
"$tokenizer_python" "$long_generation_summary" --run-root "$out" \
    --model-dir "$tokenizer_model" \
    --output-json "$out/generation-long/summary.json" \
    --output-markdown "$out/generation-long/summary.md"

if [[ "$phase" == "all" ]]; then
    python3 "$summary_tool" --run-root "$out" \
        --output-json "$out/summary.json" --output-markdown "$out/summary.md"
else
    event summary-skipped-validation-phase
fi

metric > "$out/thermal/before-restore.json"
amd-smi process --gpu "$gpu_index" --json > "$out/service/r9700-process-before-restore.json" || true
event measurements-complete
