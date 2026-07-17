#!/usr/bin/env bash
# Run exactly one approved AQ4 Phase 3c service-stop window.
#
# This driver is deliberately strict: it owns one stop and one start of only
# ullm-openai.service, never creates the R9700 lock, and delegates all GPU
# identity/health reads to the committed guard that was rehearsed while the
# service was running.  It must be invoked as root.

set -u -o pipefail
umask 077

if [ "$#" -ne 3 ] || [ "$3" != "--confirm-single-window" ]; then
  echo "usage: $0 OUT_DIRECTORY HIP_GUARD_BINARY --confirm-single-window" >&2
  exit 64
fi
if [ "$(id -u)" -ne 0 ]; then
  echo "this driver must run as root" >&2
  exit 65
fi

OUT=$1
GUARD_BIN=$2
REPO=/home/homelab1/coding-local/ultimateLLM/uLLM-project
PACKAGE=/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package
CASES="$REPO/tests/fixtures/qwen35-aq4-p2-oracle/cases.json"
REPLAY="$REPO/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/differential-trace-gpu-v1-input/replay.json"
TRACE_TOOLING_COMMIT=a7cb46e252b2f1a3e045278fe112bce21af05d32
LOCK=/run/ullm/r9700.lock
TRACE_STAGE_DIR="$OUT/trace-binary-staging"
TRACE_BIN="$TRACE_STAGE_DIR/ullm-aq4-differential-trace"
TRACE_STAGING_TOOL="$REPO/tools/stage-aq4-phase3c-trace-binary.py"
GUARD_TOOL="$REPO/tools/run-aq4-phase3c-r9700-guard.py"
LOCK_PROBE="$REPO/tools/probe-aq4-phase3c-existing-lock.py"
SERVICE=ullm-openai.service
MANIFEST=/etc/ullm/served-models/active.json
EXPECTED_MANIFEST_SHA=feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44

# Keep the trace on the audited normal M=1 route.  The normal resident matrix
# loader eagerly resolves its M=2..=128 AQ4 plan cache, so Register-BM8 must be
# proven for gfx1201 even though this trace dispatches only M=1 tokens.  Do not
# inherit the rest of the worker's broader guard profile: those flags add probes
# or select a different fallback/dispatch path than this fixed measurement.
PHASE3C_TRACE_UNSET_ENV=(
  ULLM_SYNC_LINEAR_ATTN_COMPONENTS_FOR_TIMING
  ULLM_DISABLE_AQ4_MATVEC_QKV_Z_GATE_BETA
  ULLM_DISABLE_PAGED_DECODE_SIGMOID_GATE_SELF_ATTN
  ULLM_DISABLE_SIGMOID_MUL_IN_PLACE
  ULLM_DISABLE_AQ4_MATVEC_TRIPLE_SELF_ATTN_QKV
  ULLM_DISABLE_AQ4_MATVEC_PAIR_SELF_ATTN_QK
  ULLM_ENABLE_AQ4_LM_HEAD_DIRECT_TOP1
  ULLM_EXPERIMENTAL_HIP_PAGED_DECODE_SPLIT_TILE
  ULLM_EXPERIMENTAL_HIP_PAGED_DECODE_SPLIT_MIN_CACHE_LEN
  ULLM_REQUIRE_HIP_ADD_KERNEL
  ULLM_REQUIRE_HIP_AQ4_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_TOP1_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL
  ULLM_REQUIRE_HIP_AQ4_ROW_KERNEL
  ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL
  ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL
  ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_FP8_E4M3_FLASH2_KERNEL
  ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_FP8_E4M3_KERNEL
  ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_KERNEL
  ULLM_REQUIRE_HIP_CAUSAL_ATTN_BATCH_F32_FLASH2_KERNEL
  ULLM_REQUIRE_HIP_CAUSAL_ATTN_BATCH_KERNEL
  ULLM_REQUIRE_HIP_CAUSAL_ATTN_F32_FLASH2_KERNEL
  ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL
  ULLM_REQUIRE_HIP_DECODE_ATTN_KERNEL
  ULLM_REQUIRE_HIP_DEPTHWISE_CONV1D_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL
  ULLM_REQUIRE_HIP_MATVEC_KERNEL
  ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL
  ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL
  ULLM_REQUIRE_HIP_ROPE_KERNEL
  ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL
  ULLM_REQUIRE_HIP_SILU_MUL_KERNEL
  ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL
  ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL
  ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_PAIR_KERNEL
  ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL
  ULLM_REQUIRE_HIP_TOP1_PAIRS_KERNEL
  ULLM_REQUIRE_HIP_UNKNOWN_KERNEL
)
TRACE_ENV=()
for name in "${PHASE3C_TRACE_UNSET_ENV[@]}"; do
  TRACE_ENV+=(-u "$name")
done
TRACE_ENV+=(
  HOME=/home/homelab1
  HIP_VISIBLE_DEVICES=1
  ULLM_HIP_VISIBLE_DEVICES=1
  ULLM_SERVED_MODEL_MANIFEST="$MANIFEST"
  ULLM_BUILD_GIT_COMMIT="$TRACE_TOOLING_COMMIT"
  ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB=4
  ULLM_AQ4_MATVEC_SILU_MUL_RPB=8
  ULLM_AQ4_MATVEC_ADD_RPB=8
  ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1
  ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1
  ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL=1
  ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL=1
  ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL=1
  ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1
  ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1
  ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1
  ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL=1
  ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL=1
  ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1
  ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL=1
  ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL=1
  ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL=1
  ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL=1
  ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1
  ULLM_REQUIRE_HIP_TOP1_KERNEL=1
)

STOP_ATTEMPTED=0
RESTORE_DONE=0
RESTORE_RC=99
TRACE_RC=not_started
AFTER_GUARD_RC=not_started
OUTCOME=not_started

event() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds --utc)" "$1" "$2" >> "$OUT/service-window-events.tsv"
}

write_result() {
  printf '%s\n' \
    "recorded_at_utc=$(date --iso-8601=seconds --utc)" \
    "outcome=$OUTCOME" \
    "trace_exit_code=$TRACE_RC" \
    "post_trace_guard_exit_code=$AFTER_GUARD_RC" \
    "restore_start_exit_code=$RESTORE_RC" \
    > "$OUT/service-window-result.txt"
}

restore() {
  if [ "$STOP_ATTEMPTED" -ne 1 ] || [ "$RESTORE_DONE" -ne 0 ]; then
    return 0
  fi
  RESTORE_DONE=1
  local started_at
  local finished_at
  started_at="$(date --iso-8601=seconds --utc)"
  event restore_start_invoked ""
  systemctl start "$SERVICE" > "$OUT/service-window-start.stdout" 2> "$OUT/service-window-start.stderr"
  RESTORE_RC=$?
  finished_at="$(date --iso-8601=seconds --utc)"
  printf '%s\n' "started_at_utc=$started_at" "finished_at_utc=$finished_at" "exit_code=$RESTORE_RC" > "$OUT/service-window-start.txt"
  event restore_start_returned "exit_code=$RESTORE_RC"
  return "$RESTORE_RC"
}

on_exit() {
  local code=$?
  trap - EXIT
  if [ "$STOP_ATTEMPTED" -eq 1 ] && [ "$RESTORE_DONE" -eq 0 ]; then
    restore
  fi
  exit "$code"
}
trap on_exit EXIT

if [ ! -d "$OUT" ] || [ -e "$OUT/gpu-trace" ] || [ -e "$OUT/guard-before" ] || [ -e "$OUT/guard-after" ]; then
  echo "output contract failed" >&2
  exit 30
fi
if [ ! -f "$LOCK" ] || [ -L "$LOCK" ] || [ ! -x "$TRACE_BIN" ] || [ ! -x "$GUARD_BIN" ] || [ ! -f "$TRACE_STAGING_TOOL" ] || [ ! -f "$GUARD_TOOL" ] || [ ! -f "$LOCK_PROBE" ] || [ ! -x /opt/rocm/bin/amd-smi ]; then
  echo "pre-stop file contract failed" >&2
  exit 31
fi
if ! python3 "$TRACE_STAGING_TOOL" \
  --verify \
  --source "$REPO/target/release/ullm-aq4-differential-trace" \
  --output "$TRACE_STAGE_DIR" \
  --trace-tooling-commit "$TRACE_TOOLING_COMMIT" \
  > "$OUT/trace-binary-staging-verify.json"; then
  echo "staged trace binary identity contract failed" >&2
  exit 37
fi
if [ ! -d "$PACKAGE" ] || [ ! -f "$CASES" ] || [ ! -f "$REPLAY" ] || [ ! -r "$MANIFEST" ]; then
  echo "trace input contract failed" >&2
  exit 32
fi
if [ "$(git -C "$REPO" rev-parse "$TRACE_TOOLING_COMMIT")" != "$TRACE_TOOLING_COMMIT" ]; then
  echo "trace tooling commit is unavailable" >&2
  exit 34
fi
if ! git -C "$REPO" ls-files --error-unmatch tools/stage-aq4-phase3c-trace-binary.py >/dev/null \
  || ! git -C "$REPO" diff --quiet HEAD -- tools/stage-aq4-phase3c-trace-binary.py \
  || ! git -C "$REPO" diff --cached --quiet HEAD -- tools/stage-aq4-phase3c-trace-binary.py; then
  echo "trace-binary staging tool is not tracked and clean" >&2
  exit 38
fi
if ! git -C "$REPO" diff --quiet "$TRACE_TOOLING_COMMIT" -- \
  crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs \
  crates/ullm-engine/src/qwen35_aq4_model_runtime.rs \
  crates/ullm-engine/src/bin/ullm-aq4-differential-trace.rs \
  tools/verify-aq4-layer0-package-embedding-fixture.py \
  tools/compare-aq4-layer0-cpu-gpu-stage-stream.py; then
  echo "trace tooling worktree differs from the fixed commit" >&2
  exit 35
fi
if ! git -C "$REPO" diff --cached --quiet "$TRACE_TOOLING_COMMIT" -- \
  crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs \
  crates/ullm-engine/src/qwen35_aq4_model_runtime.rs \
  crates/ullm-engine/src/bin/ullm-aq4-differential-trace.rs \
  tools/verify-aq4-layer0-package-embedding-fixture.py \
  tools/compare-aq4-layer0-cpu-gpu-stage-stream.py; then
  echo "trace tooling index differs from the fixed commit" >&2
  exit 36
fi
if ! runuser -u homelab1 -- env "${TRACE_ENV[@]}" \
  "$TRACE_BIN" --print-phase3c-trace-guard-requirements \
  > "$OUT/trace-guard-diagnostic.json" \
  2> "$OUT/trace-guard-diagnostic.stderr"; then
  echo "complete Phase 3c trace guard diagnostic failed before service stop" >&2
  exit 39
fi
if ! python3 - "$OUT/trace-guard-diagnostic.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
expected = [
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL",
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
    "ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL",
    "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL",
    "ULLM_REQUIRE_HIP_TOP1_KERNEL",
]
assert payload["schema_version"] == "ullm.qwen35_aq4_phase3c_trace_guard_diagnostic.v1"
assert payload["status"] == "valid"
assert payload["required_environment"] == expected
assert len(payload["linear_stage_guard"]["required_environment"]) == len(expected)
PY
then
  echo "complete Phase 3c trace guard diagnostic contract failed before service stop" >&2
  exit 39
fi

active_state="$(systemctl show "$SERVICE" -p ActiveState --value)"
sub_state="$(systemctl show "$SERVICE" -p SubState --value)"
main_pid="$(systemctl show "$SERVICE" -p MainPID --value)"
restarts="$(systemctl show "$SERVICE" -p NRestarts --value)"
preserve="$(systemctl show "$SERVICE" -p RuntimeDirectoryPreserve --value)"
manifest_sha="$(sha256sum "$MANIFEST" | awk '{print $1}')"
printf '%s\n' \
  "captured_at_utc=$(date --iso-8601=seconds --utc)" \
  "active_state=$active_state" \
  "sub_state=$sub_state" \
  "main_pid=$main_pid" \
  "nrestarts=$restarts" \
  "runtime_directory_preserve=$preserve" \
  "manifest_sha256=$manifest_sha" \
  > "$OUT/service-window-pre-stop.txt"
if [ "$active_state" != active ] || [ "$sub_state" != running ] || [ "$main_pid" -le 0 ] || [ "$preserve" != yes ] || [ "$manifest_sha" != "$EXPECTED_MANIFEST_SHA" ]; then
  OUTCOME=pre_stop_baseline_failure
  write_result
  exit 33
fi

STOP_ATTEMPTED=1
stop_started_at="$(date --iso-8601=seconds --utc)"
event stop_invoked ""
systemctl stop "$SERVICE" > "$OUT/service-window-stop.stdout" 2> "$OUT/service-window-stop.stderr"
stop_rc=$?
systemctl show "$SERVICE" -p ActiveState -p SubState -p MainPID > "$OUT/service-window-stop-systemctl-show.txt" 2> "$OUT/service-window-stop-systemctl-show.stderr"
show_rc=$?
stop_finished_at="$(date --iso-8601=seconds --utc)"
printf '%s\n' "started_at_utc=$stop_started_at" "finished_at_utc=$stop_finished_at" "exit_code=$stop_rc" "post_stop_show_exit_code=$show_rc" > "$OUT/service-window-stop.txt"
event stop_returned "exit_code=$stop_rc"
if [ "$stop_rc" -ne 0 ]; then
  OUTCOME=stop_failure
  restore
  write_result
  exit 50
fi

runuser -u homelab1 -- python3 "$LOCK_PROBE" "$OUT/service-window-lock-after-stop.json" "$LOCK"
probe_rc=$?
event post_stop_lock_probe "exit_code=$probe_rc"
if [ "$probe_rc" -ne 0 ]; then
  OUTCOME=lock_acquisition_failure_no_trace
  restore
  write_result
  exit 40
fi

python3 "$GUARD_TOOL" --output "$OUT/guard-before" --guard-bin "$GUARD_BIN" --health-phase before
guard_rc=$?
event r9700_architecture_and_health_guard "exit_code=$guard_rc"
if [ "$guard_rc" -ne 0 ]; then
  OUTCOME=architecture_guard_failure_no_trace
  restore
  write_result
  exit 41
fi

trace_started_at="$(date --iso-8601=seconds --utc)"
event trace_invoked ""
runuser -u homelab1 -- env "${TRACE_ENV[@]}" \
  bash -ceu '
    lock=$1
    trace_bin=$2
    package=$3
    cases=$4
    replay=$5
    trace_root=$6
    out=$7
    exec 9< "$lock"
    flock -n 9
    exec "$trace_bin" "$package" "$cases" "$replay" "$trace_root" 1 \
      --enable-intermediate-trace \
      --enable-linear-stage-trace \
      > "$out/gpu-trace.stdout" \
      2> "$out/gpu-trace.stderr"
  ' bash "$LOCK" "$TRACE_BIN" "$PACKAGE" "$CASES" "$REPLAY" "$OUT/gpu-trace" "$OUT"
TRACE_RC=$?
trace_finished_at="$(date --iso-8601=seconds --utc)"
printf '%s\n' \
  "started_at_utc=$trace_started_at" \
  "finished_at_utc=$trace_finished_at" \
  "exit_code=$TRACE_RC" \
  "lock_open_policy=read_only_existing_file_no_create" \
  > "$OUT/gpu-trace.exit-status.txt"
event trace_returned "exit_code=$TRACE_RC"

python3 "$GUARD_TOOL" --output "$OUT/guard-after" --guard-bin "$GUARD_BIN" --health-phase after
AFTER_GUARD_RC=$?
event post_trace_health_guard "exit_code=$AFTER_GUARD_RC"
restore
if [ "$RESTORE_RC" -ne 0 ]; then
  OUTCOME=restore_start_failure
  write_result
  exit 70
fi
if [ "$TRACE_RC" -ne 0 ]; then
  OUTCOME=trace_failure_no_retry
  write_result
  exit "$TRACE_RC"
fi
if [ "$AFTER_GUARD_RC" -ne 0 ]; then
  OUTCOME=post_trace_health_guard_failure
  write_result
  exit 71
fi
OUTCOME=trace_completed_pending_post_restore_validation
write_result
exit 0
