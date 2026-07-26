#!/usr/bin/env bash
# Run one owned R9700 window for the AQ4_0 matvec-add group-specialization A/B.
#
# This is intentionally root-only.  It stops only ullm-openai.service after proving that
# the service owns the shared flock, uses the retained shuffle source as an in-process-identical
# reference, and always makes one bounded service restoration attempt.
set -Eeuo pipefail

if [[ ${EUID} -ne 0 || $# -ne 1 ]]; then
    echo "usage: run-candidate-window.sh ABSOLUTE_OUTPUT_DIRECTORY" >&2
    exit 64
fi

repo_root="/home/homelab1/coding-local/ultimateLLM/uLLM-project"
results_root="$repo_root/benchmarks/results/2026-07-27/aq4-matvec-add-optimization"
output_dir="$1"
service="ullm-openai.service"
llama_service="llama-qwen35-udq4.service"
gpu_lock="/run/ullm/r9700.lock"
active_manifest="/etc/ullm/served-models/active.json"
# `sudo` uses a secure PATH that does not include ROCm's bin directory.  Keep
# telemetry fail-closed rather than silently losing the thermal/clock evidence.
amd_smi="/opt/rocm/bin/amd-smi"
# The grouped production baseline lives on 9d864350, not on current main.  This isolated
# worktree has the same two runtime blobs as BZ's c8074928 grouped worker; do not substitute a
# main-built worker merely because it accepts the grouped environment variable.
candidate_source_root="/tmp/ullm-aq4-add-grouped-source-20260727T0416"
candidate_target="/tmp/ullm-aq4-add-grouped-target-20260727T0416"
candidate_worker="$candidate_target/release/ullm-aq4-worker"
profile_binary="$candidate_target/release/ullm-aq4-decode-step-profile"
prefill_binary="$candidate_target/release/ullm-aq4-e2e-prefill-timing"
test_binary=""
runtime_source="$candidate_source_root/runtime/src/ullm_runtime_hiprtc_sources.inc"
test_source="$candidate_source_root/crates/ullm-runtime-sys/src/test_parts/aq4_matvec_add_wide_load_prototype.rs"
candidate_provenance="$results_root/candidate-grouped-build-provenance.json"

case "$output_dir" in
    "$results_root"/*) ;;
    *)
        echo "output must be below $results_root" >&2
        exit 65
        ;;
esac
if [[ -e "$output_dir" ]]; then
    echo "refusing to overwrite existing output directory: $output_dir" >&2
    exit 65
fi
if [[ ! -r "$active_manifest" || ! -r "$candidate_provenance" || ! -x "$amd_smi" ]]; then
    echo "candidate binary or active manifest precondition failed" >&2
    exit 65
fi
test_binary="$(jq -r '.artifacts.direct_gpu_test.path // empty' "$candidate_provenance")"
if [[ ! "$test_binary" == "$candidate_target"/debug/deps/ullm_runtime_sys-* ||
    ! -x "$candidate_worker" || ! -x "$profile_binary" || ! -x "$prefill_binary" ||
    ! -x "$test_binary" ]]; then
    echo "candidate artifact paths or executability precondition failed" >&2
    exit 65
fi
declare -A candidate_artifacts=(
    [worker]="$candidate_worker"
    [decode_profile]="$profile_binary"
    [prefill_profile]="$prefill_binary"
    [direct_gpu_test]="$test_binary"
)
for artifact_name in "${!candidate_artifacts[@]}"; do
    artifact_path="${candidate_artifacts[$artifact_name]}"
    recorded_path="$(jq -r --arg name "$artifact_name" \
        '.artifacts[$name].path // empty' "$candidate_provenance")"
    expected_artifact_hash="$(jq -r --arg name "$artifact_name" \
        '.artifacts[$name].sha256 // empty' "$candidate_provenance")"
    actual_artifact_hash="$(sha256sum "$artifact_path" | awk '{print $1}')"
    if [[ "$recorded_path" != "$artifact_path" ||
        ! "$expected_artifact_hash" =~ ^[0-9a-f]{64}$ ||
        "$actual_artifact_hash" != "$expected_artifact_hash" ]]; then
        echo "candidate artifact provenance no longer matches $artifact_name" >&2
        exit 65
    fi
done
expected_base_commit="$(jq -r '.base_commit // empty' "$candidate_provenance")"
actual_base_commit="$(git -C "$candidate_source_root" rev-parse HEAD 2>/dev/null || true)"
if [[ "$expected_base_commit" != "9d8643506a36659ecec3fc2d931deba26d29f574" ||
    "$actual_base_commit" != "$expected_base_commit" ]]; then
    echo "candidate source is not the audited 9d864350 grouped baseline" >&2
    exit 65
fi
expected_dirty=$' M crates/ullm-runtime-sys/src/test_parts/aq4_matvec_add_wide_load_prototype.rs\n M runtime/src/ullm_runtime_hiprtc_sources.inc'
actual_dirty="$(git -C "$candidate_source_root" status --porcelain)"
if [[ "$actual_dirty" != "$expected_dirty" ]]; then
    echo "candidate source worktree contains an unexpected change set" >&2
    exit 65
fi
expected_patch="$(jq -r '.source_patch_sha256 // empty' "$candidate_provenance")"
actual_patch="$({ git -C "$candidate_source_root" diff -- \
    runtime/src/ullm_runtime_hiprtc_sources.inc \
    crates/ullm-runtime-sys/src/test_parts/aq4_matvec_add_wide_load_prototype.rs; } | sha256sum | awk '{print $1}')"
if [[ ! "$expected_patch" =~ ^[0-9a-f]{64}$ || "$actual_patch" != "$expected_patch" ]]; then
    echo "candidate source patch provenance no longer matches" >&2
    exit 65
fi
for source_spec in \
    "runtime/src/ullm_runtime_hiprtc_sources.inc:$runtime_source" \
    "crates/ullm-runtime-sys/src/test_parts/aq4_matvec_add_wide_load_prototype.rs:$test_source"; do
    source_key="${source_spec%%:*}"
    source_path="${source_spec#*:}"
    expected_hash="$(jq -r --arg key "$source_key" '.source_inputs[$key] // empty' "$candidate_provenance")"
    actual_hash="$(sha256sum "$source_path" | awk '{print $1}')"
    if [[ ! "$expected_hash" =~ ^[0-9a-f]{64}$ || "$actual_hash" != "$expected_hash" ]]; then
        echo "candidate source provenance no longer matches $source_key; rebuild before measuring" >&2
        exit 65
    fi
done
if ! jq -e '.worker.binary | strings | startswith("/opt/ullm/")' "$active_manifest" >/dev/null; then
    echo "active manifest does not yet point to a protected /opt/ullm worker; refusing window" >&2
    exit 66
fi
active_worker="$(jq -r '.worker.binary' "$active_manifest")"
if [[ ! -x "$active_worker" || "$(stat -c '%u:%g:%a' "$active_worker")" != "0:0:555" ]]; then
    echo "active /opt/ullm worker is not root:root mode 0555; refusing window" >&2
    exit 66
fi
if ! jq -e '
    .format.format_id == "AQ4_0" and
    .worker.identity.device == "gfx1201" and
    .worker.execution.paged_decode_attention.kernel == "aq4_gqa_grouped_split" and
    .worker.execution.paged_decode_attention.split_tile == 128
' "$active_manifest" >/dev/null; then
    echo "active manifest is not BZ's protected AQ4_0 grouped-decode contract; refusing window" >&2
    exit 66
fi

umask 027
mkdir --mode=750 "$output_dir"
mkdir --mode=750 "$output_dir/direct-test" "$output_dir/runtime" "$output_dir/counters" \
    "$output_dir/launch-invariant" \
    "$output_dir/decode" "$output_dir/prefill" "$output_dir/thermal" "$output_dir/service"
# `rocprofv3` is deliberately run as homelab1 so its HIPRTC/cache identity is
# the same as the worker.  It creates its own `-d` subtree, therefore the
# capture root must already be writable by that user even though this wrapper
# itself runs as root for the bounded service/lock lifecycle.
chown -R homelab1:homelab1 "$output_dir"

service_stopped=0
lock_held=0
metric_pid=""

record_required_preflight() {
    local label="$1"
    fuser -v "$gpu_lock" >"$output_dir/service/${label}-fuser.txt" 2>&1 || true
    pgrep -af 'ullm-sq8-r9700|run_measurements.py|llama-bench|promote-served-model' \
        | awk '{print $1, $2}' >"$output_dir/service/${label}-pgrep-pids.txt" || true
    systemctl show "$service" -p ActiveState -p NRestarts \
        >"$output_dir/service/${label}-ullm-openai.txt"
}

record_service() {
    local label="$1"
    systemctl show "$service" -p ActiveState -p SubState -p MainPID -p NRestarts \
        -p Result -p StartLimitBurst -p StartLimitIntervalUSec -p UnitFileState \
        >"$output_dir/service/${label}-ullm-openai-full.txt" 2>&1 || true
    systemctl show "$llama_service" -p ActiveState -p SubState -p MainPID -p UnitFileState \
        >"$output_dir/service/${label}-llama-qwen35-udq4.txt" 2>&1 || true
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
    local main_pid holders service_pids pid
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

metric() {
    "$amd_smi" metric -g 2 -t -c -p --json
}

thermal_gate() {
    local condition="$1"
    local path="$output_dir/thermal/${condition}-gate.jsonl"
    local attempt edge
    for attempt in $(seq 1 180); do
        metric | tee -a "$path" >"$output_dir/thermal/${condition}-latest.json"
        edge="$(jq -r '.gpu_data[0].temperature.edge.value // empty' \
            "$output_dir/thermal/${condition}-latest.json")"
        if [[ "$edge" =~ ^[0-9]+([.][0-9]+)?$ ]] &&
            awk "BEGIN { exit !($edge <= 45.0) }"; then
            printf '%s\tthermal-gate-pass condition=%s edge_c=%s limit_c=45\n' \
                "$(date --iso-8601=seconds)" "$condition" "$edge" >>"$output_dir/events.tsv"
            return 0
        fi
        printf '%s\tthermal-gate-wait condition=%s edge_c=%s limit_c=45 attempt=%s\n' \
            "$(date --iso-8601=seconds)" "$condition" "${edge:-unknown}" "$attempt" \
            >>"$output_dir/events.tsv"
        sleep 5
    done
    echo "thermal gate timed out for $condition (edge did not reach <=45 C)" >&2
    return 1
}

run_as_user() {
    local reference="$1"
    local grouped="$2"
    shift 2
    local -a environment=(
        "PATH=/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin"
        "HOME=/home/homelab1"
        "XDG_CACHE_HOME=/home/homelab1/.cache"
        "HIP_VISIBLE_DEVICES=1"
        "ULLM_HIP_VISIBLE_DEVICES=1"
    )
    local required
    while IFS= read -r required; do
        [[ "$required" =~ ^[A-Z_][A-Z0-9_]*$ ]] || {
            echo "unsafe required environment name in manifest: $required" >&2
            return 1
        }
        environment+=("${required}=1")
    done < <(jq -r '.worker.required_environment[]' "$active_manifest")
    if [[ "$reference" == 1 ]]; then
        environment+=("ULLM_AQ4_MATVEC_ADD_USE_SHUFFLE_REFERENCE=1")
    fi
    if [[ "$grouped" == 1 ]]; then
        environment+=("ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT=1")
    fi
    runuser -u homelab1 -- env -i "${environment[@]}" "$@"
}

run_profile() {
    local label="$1"
    local reference="$2"
    local grouped="$3"
    printf '%s\n' "$(date --iso-8601=seconds)" >"$output_dir/decode/${label}.started-at"
    run_as_user "$reference" "$grouped" "$profile_binary" 1339 --warmup 6 --measured 32 \
        >"$output_dir/decode/${label}.jsonl" \
        2>"$output_dir/decode/${label}.stderr"
    printf '%s\n' "$(date --iso-8601=seconds)" >"$output_dir/decode/${label}.finished-at"
    # Capture the actual gfx/memory clocks immediately after this load, rather than treating
    # AMD SMI's THROTTLED text as a clock verdict.  A read failure is recorded but does not
    # invalidate the unprofiled full-model throughput result.
    metric >"$output_dir/thermal/${label}-after.json" 2>&1 || true
}

run_counter_capture() {
    local label="$1"
    local reference="$2"
    # Counter availability is diagnostic evidence for Phase 0, not a correctness gate.  Preserve
    # a profiler incompatibility and continue to the mandatory unprofiled full-model comparison.
    if run_as_user "$reference" 0 rocprofv3 -d "$output_dir/counters/${label}" -f csv \
        --pmc OccupancyPercent,SQ_WAVES,SQ_WAVE_CYCLES,SQ_INSTS_VALU,SQ_WAIT_ANY \
        --kernel-include-regex 'ullm_aq4_matvec_add_f32_kernel' -- \
        "$profile_binary" 1339 --warmup 1 --measured 2 \
        >"$output_dir/counters/${label}.stdout" \
        2>"$output_dir/counters/${label}.stderr"; then
        printf '%s\n' 0 >"$output_dir/counters/${label}.exit-status"
    else
        local status=$?
        printf '%s\n' "$status" >"$output_dir/counters/${label}.exit-status"
        printf '%s\tcounter-capture-failed label=%s status=%s\n' \
            "$(date --iso-8601=seconds)" "$label" "$status" >>"$output_dir/events.tsv"
    fi
}

run_launch_invariant_capture() {
    # The profile's wall-clock JSON remains the only throughput source.  This
    # small trace is solely a cardinality/geometry check: both the retained
    # pre-specialization baseline and the candidate at C=1339 must retain 292
    # module launches per measured token, including 64 matvec-add calls.
    local label="$1"
    local reference="$2"
    local capture_dir="$output_dir/launch-invariant/$label"
    local root="$capture_dir/rocprof"
    mkdir -p "$capture_dir"
    # This child is created after the initial result-root handoff above.  Give
    # the profiler's unprivileged process ownership before rocprofv3 creates
    # its `-d` subtree beneath it.
    chown homelab1:homelab1 "$capture_dir"
    run_as_user "$reference" 0 rocprofv3 -d "$root" -f csv \
        --hip-trace --kernel-trace --marker-trace --log-level error -- \
        "$profile_binary" 1339 --warmup 1 --measured 2 \
        >"$capture_dir/profile.stdout" \
        2>"$capture_dir/profile.stderr"
    local kernel_trace hip_trace marker_trace
    kernel_trace="$(find "$root" -type f -name '*kernel_trace.csv' -print -quit)"
    hip_trace="$(find "$root" -type f -name '*hip_api_trace.csv' -print -quit)"
    marker_trace="$(find "$root" -type f -name '*marker_api_trace.csv' -print -quit)"
    if [[ -z "$kernel_trace" || -z "$hip_trace" || -z "$marker_trace" ]]; then
        echo "launch-invariant rocprof capture is incomplete" >&2
        return 1
    fi
    python3 "$repo_root/tools/analyze-aq4-decode-walltime-accounting.py" \
        --kernel-trace "$kernel_trace" --hip-api-trace "$hip_trace" \
        --marker-trace "$marker_trace" \
        --profile-stdout "$capture_dir/profile.stdout" \
        --expected-cache-start 1339 --expected-steps 2 \
        --output "$capture_dir/accounting.json"
    python3 - "$capture_dir/accounting.json" "$label" \
        >"$capture_dir/assertion.json" <<'PY'
import json
import sys

accounting = json.load(open(sys.argv[1], encoding="utf-8"))
label = sys.argv[2]
families = accounting["module_kernel"]["families"]
module_dispatches = sum(item["kernel_count"] for item in families.values())
add_dispatches = next(
    item["kernel_count"]
    for item in accounting["module_kernel"]["kernels"]
    if item["name"] == "ullm_aq4_matvec_add_f32_kernel"
)
if module_dispatches != 584 or add_dispatches != 128:
    raise SystemExit(
        f"C=1339 launch invariant failed: module={module_dispatches} "
        f"(expected 584), matvec_add={add_dispatches} (expected 128)"
    )
print(json.dumps({
    "status": "valid",
    "body": label,
    "measured_tokens": 2,
    "module_launches_per_token": module_dispatches // 2,
    "matvec_add_launches_per_token": add_dispatches // 2,
    "throughput_note": "trace is cardinality-only; unprofiled profile outputs supply tok/s",
}, indent=2, sort_keys=True))
PY
}

restore_service() {
    local status=$?
    local restored=0
    trap - EXIT INT TERM
    if [[ -n "$metric_pid" ]] && kill -0 "$metric_pid" 2>/dev/null; then
        kill "$metric_pid" 2>/dev/null || true
        wait "$metric_pid" 2>/dev/null || true
    fi
    if ((lock_held)); then
        flock -u 9 || true
        exec 9>&-
        lock_held=0
        printf '%s\tr9700-lock-released\n' "$(date --iso-8601=seconds)" >>"$output_dir/events.tsv"
    fi
    metric >"$output_dir/thermal/after-window.json" 2>&1 || true
    if ((service_stopped)); then
        printf '%s\tservice-restore-attempt\n' "$(date --iso-8601=seconds)" >>"$output_dir/events.tsv"
        if systemctl start "$service" >"$output_dir/service/restore.stdout" \
            2>"$output_dir/service/restore.stderr"; then
            restored=1
        elif [[ "$(systemctl show "$service" -p Result --value)" == start-limit-hit ]]; then
            printf '%s\tservice-restore-reset-failed-attempt\n' "$(date --iso-8601=seconds)" \
                >>"$output_dir/events.tsv"
            if systemctl reset-failed "$service" >>"$output_dir/service/restore.stdout" \
                2>>"$output_dir/service/restore.stderr" &&
                systemctl start "$service" >>"$output_dir/service/restore.stdout" \
                2>>"$output_dir/service/restore.stderr"; then
                restored=1
            fi
        fi
        printf '%s\n' "$restored" >"$output_dir/service/restore.exit-status"
        for delay in 1 2 4 8 16 24; do
            if systemctl is-active --quiet "$service"; then
                break
            fi
            sleep "$delay"
        done
        if ! systemctl is-active --quiet "$service"; then
            status=1
            printf '%s\tservice-restore-not-active\n' "$(date --iso-8601=seconds)" \
                >>"$output_dir/events.tsv"
        fi
    fi
    record_required_preflight after-restore
    record_service after-restore
    sha256sum "$active_manifest" "$candidate_worker" "$profile_binary" "$prefill_binary" \
        "$test_binary" "$runtime_source" "$test_source" "$candidate_provenance" \
        >"$output_dir/input-sha256-after.txt" || true
    if ! cmp -s "$output_dir/input-sha256-before.txt" "$output_dir/input-sha256-after.txt"; then
        status=1
        printf '%s\tinput-identity-changed-during-window\n' "$(date --iso-8601=seconds)" \
            >>"$output_dir/events.tsv"
    fi
    chown -R homelab1:homelab1 "$output_dir"
    exit "$status"
}
trap restore_service EXIT INT TERM

cd "$repo_root"
printf '%s\tpreflight-begin\n' "$(date --iso-8601=seconds)" >>"$output_dir/events.tsv"
record_required_preflight before-stop
record_service before-stop
sha256sum "$active_manifest" "$candidate_worker" "$profile_binary" "$prefill_binary" \
    "$test_binary" "$runtime_source" "$test_source" "$candidate_provenance" \
    >"$output_dir/input-sha256-before.txt"
cp -- "$0" "$output_dir/run-candidate-window.sh"

if ! grep -qx 'ActiveState=inactive' "$output_dir/service/before-stop-llama-qwen35-udq4.txt" ||
    ! grep -qx 'UnitFileState=disabled' "$output_dir/service/before-stop-llama-qwen35-udq4.txt"; then
    echo "$llama_service must remain inactive and disabled" >&2
    exit 67
fi
if ! systemctl is-active --quiet "$service"; then
    echo "$service is not active; refusing to inherit another owner's inactive window" >&2
    exit 75
fi
if ! lock_is_held_only_by_service; then
    echo "R9700 lock is not owned only by the active gateway; refusing to stop the service" >&2
    exit 75
fi

printf '%s\tservice-stop-attempt\n' "$(date --iso-8601=seconds)" >>"$output_dir/events.tsv"
systemctl stop "$service" >"$output_dir/service/stop.stdout" 2>"$output_dir/service/stop.stderr"
service_stopped=1
for attempt in $(seq 1 30); do
    if ! systemctl is-active --quiet "$service" && ! pgrep -x ullm-aq4-worker >/dev/null &&
        ! fuser "$gpu_lock" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
record_required_preflight after-stop-before-lock
record_service after-stop
if systemctl is-active --quiet "$service" || pgrep -x ullm-aq4-worker >/dev/null ||
    fuser "$gpu_lock" >/dev/null 2>&1; then
    echo "service, AQ4 worker, or R9700 lock remained held after stop; refusing to steal it" >&2
    exit 75
fi

exec 9<>"$gpu_lock"
if ! flock -n 9; then
    echo "R9700 lock was acquired after service stop; refusing to steal it" >&2
    exit 75
fi
lock_held=1
printf '%s\tr9700-lock-acquired\n' "$(date --iso-8601=seconds)" >>"$output_dir/events.tsv"
record_required_preflight lock-held
record_service lock-held
"$amd_smi" static -g 2 -a --json >"$output_dir/thermal/r9700-static.json"
metric >"$output_dir/thermal/before-window.json"
"$amd_smi" metric -g 2 -t -c -p --json -w 1 -W 14400 \
    --file "$output_dir/thermal/metrics-watch.json" >"$output_dir/thermal/metrics-watch.log" 2>&1 &
metric_pid=$!

thermal_gate direct-differential
printf '%s\tdirect-differential-begin\n' "$(date --iso-8601=seconds)" >>"$output_dir/events.tsv"
run_as_user 0 0 env ULLM_RUN_AQ4_MATVEC_ADD_PRODUCTION_DIFFERENTIAL=1 \
    "$test_binary" hip_aq4_matvec_add_production_model_shapes_match_cpu_when_enabled \
    --ignored --nocapture >"$output_dir/direct-test/production-differential.stdout" \
    2>"$output_dir/direct-test/production-differential.stderr"
run_as_user 0 0 env ULLM_RUN_AQ4_MATVEC_ADD_SHUFFLE_DIFFERENTIAL=1 \
    "$test_binary" hip_aq4_matvec_add_shuffle_prototype_matches_cpu_for_production_shapes_when_enabled \
    --ignored --nocapture >"$output_dir/direct-test/shuffle-baseline-differential.stdout" \
    2>"$output_dir/direct-test/shuffle-baseline-differential.stderr"
run_as_user 0 0 env ULLM_RUN_AQ4_MATVEC_ADD_SHUFFLE_TIMING=1 \
    "$test_binary" hip_aq4_matvec_add_shuffle_prototype_timing_vs_production_for_production_shapes_when_enabled \
    --ignored --nocapture >"$output_dir/direct-test/candidate-vs-reference-timing.stdout" \
    2>"$output_dir/direct-test/candidate-vs-reference-timing.stderr"
printf '%s\tdirect-differential-complete\n' "$(date --iso-8601=seconds)" >>"$output_dir/events.tsv"

thermal_gate runtime-greedy
run_as_user 1 0 "$profile_binary" 1339 --warmup 1 --measured 2 \
    >"$output_dir/runtime/shuffle-reference-greedy.jsonl" \
    2>"$output_dir/runtime/shuffle-reference-greedy.stderr"
run_as_user 0 0 "$profile_binary" 1339 --warmup 1 --measured 2 \
    >"$output_dir/runtime/group-specialized-greedy.jsonl" \
    2>"$output_dir/runtime/group-specialized-greedy.stderr"
python3 - "$output_dir/runtime/shuffle-reference-greedy.jsonl" \
    "$output_dir/runtime/group-specialized-greedy.jsonl" >"$output_dir/runtime/greedy-validation.json" <<'PY'
import json
import sys

def tokens(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    values = [row["token_id"] for row in rows if row.get("event") == "measured_decode_step"]
    if len(values) != 2:
        raise SystemExit(f"{path}: expected two measured greedy tokens, found {len(values)}")
    return values

reference = tokens(sys.argv[1])
candidate = tokens(sys.argv[2])
if candidate != reference:
    raise SystemExit(f"greedy token mismatch: reference={reference} candidate={candidate}")
print(json.dumps({"status": "valid", "reference_tokens": reference, "candidate_tokens": candidate}, indent=2))
PY

thermal_gate launch-invariant
run_launch_invariant_capture shuffle-reference 1
thermal_gate launch-invariant-candidate
run_launch_invariant_capture group-specialized 0

thermal_gate counter-reference
run_counter_capture shuffle-reference 1
thermal_gate counter-candidate
run_counter_capture group-specialized 0

for round in a b; do
    thermal_gate "decode-reference-direct-${round}"
    run_profile "reference-direct-${round}" 1 0
    thermal_gate "decode-specialized-direct-${round}"
    run_profile "specialized-direct-${round}" 0 0
    thermal_gate "decode-reference-grouped-${round}"
    run_profile "reference-grouped-${round}" 1 1
    thermal_gate "decode-specialized-grouped-${round}"
    run_profile "specialized-grouped-${round}" 0 1
done

python3 - "$output_dir/decode" >"$output_dir/decode/summary.json" <<'PY'
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
groups = {}
sequences = {}
per_run = {}
for path in sorted(root.glob("*.jsonl")):
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    summary = [row for row in rows if row.get("event") == "summary"]
    tokens = [row["token_id"] for row in rows if row.get("event") == "measured_decode_step"]
    if len(summary) != 1 or len(tokens) != 32:
        raise SystemExit(f"{path}: expected one summary and 32 measured tokens")
    label = path.stem
    mode = "grouped" if "grouped" in label else "direct"
    body = "reference" if label.startswith("reference-") else "specialized"
    throughput = summary[0]["mean_tokens_per_second"]
    groups.setdefault(f"{body}-{mode}", []).append(throughput)
    per_run[label] = throughput
    sequences[label] = tokens

baseline = sequences["reference-direct-a"]
if any(tokens != baseline for tokens in sequences.values()):
    raise SystemExit("32-step greedy token sequence differs across reference/candidate or direct/grouped runs")
means = {key: statistics.mean(values) for key, values in sorted(groups.items())}
result = {
    "schema_version": "ullm.aq4_matvec_add.full_model_ab.v1",
    "measured_steps_per_run": 32,
    "runs_per_mode": 2,
    "per_run_tokens_per_second": per_run,
    "means_tokens_per_second": means,
    "ranges_tokens_per_second": {
        key: {"min": min(values), "max": max(values)}
        for key, values in sorted(groups.items())
    },
    "speedups": {
        "specialized_over_reference_direct": means["specialized-direct"] / means["reference-direct"],
        "specialized_over_reference_grouped": means["specialized-grouped"] / means["reference-grouped"],
    },
    "greedy_token_sequence": baseline,
    "historical_controls_tokens_per_second": {
        "direct": 74.110977,
        "grouped": 74.509830,
    },
    "throughput_definition": "mean_tokens_per_second emitted by ullm-aq4-decode-step-profile over 32 full-model decode steps; not a rocprof range duration",
}
print(json.dumps(result, indent=2, sort_keys=True))
PY

thermal_gate prefill-reference
run_as_user 1 0 "$prefill_binary" >"$output_dir/prefill/shuffle-reference-m128-p2048.json" \
    2>"$output_dir/prefill/shuffle-reference-m128-p2048.stderr"
metric >"$output_dir/thermal/prefill-reference-after.json" 2>&1 || true
thermal_gate prefill-candidate
run_as_user 0 0 "$prefill_binary" >"$output_dir/prefill/group-specialized-m128-p2048.json" \
    2>"$output_dir/prefill/group-specialized-m128-p2048.stderr"
metric >"$output_dir/thermal/prefill-candidate-after.json" 2>&1 || true
python3 - "$output_dir/prefill/shuffle-reference-m128-p2048.json" \
    "$output_dir/prefill/group-specialized-m128-p2048.json" \
    >"$output_dir/prefill/summary.json" <<'PY'
import json
import sys

def load(path):
    value = json.load(open(path, encoding="utf-8"))
    required = {"tokens", "chunk_width", "elapsed_seconds", "tokens_per_second"}
    if set(value) != required or value["tokens"] != 2048 or value["chunk_width"] != 128:
        raise SystemExit(f"{path}: unexpected AQ4 M=128/p2048 prefill result")
    if not isinstance(value["tokens_per_second"], (int, float)) or value["tokens_per_second"] <= 0:
        raise SystemExit(f"{path}: invalid prefill throughput")
    return value

reference = load(sys.argv[1])
candidate = load(sys.argv[2])
print(json.dumps({
    "schema_version": "ullm.aq4_matvec_add.prefill_ab.v1",
    "condition": "cold AQ4_0 prefill; 2048 tokens; M=128",
    "reference_tokens_per_second": reference["tokens_per_second"],
    "candidate_tokens_per_second": candidate["tokens_per_second"],
    "candidate_over_reference": candidate["tokens_per_second"] / reference["tokens_per_second"],
    "historical_aq4_f32_m128_p2048_tokens_per_second": 966.9146012373889,
    "throughput_definition": "one cold full-model prefill timing after one in-process warmup; not a profiler range duration",
}, indent=2, sort_keys=True))
PY
printf '%s\tmeasurement-complete\n' "$(date --iso-8601=seconds)" >>"$output_dir/events.tsv"
