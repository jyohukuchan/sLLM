#!/usr/bin/env bash
# Run exactly one approved AQ4 Phase 6 final-output service-stop window.
#
# The one GPU model execution is the same M=1 path-oracle route that produced
# the 07/14 bounded final hidden/logit comparison.  This driver never creates
# the R9700 lock, never queries an untargeted GPU, and restores only
# ullm-openai.service exactly once after a stop attempt.

set -u -o pipefail
umask 077

if [ "$#" -ne 6 ] || [ "$6" != "--confirm-single-window" ]; then
  echo "usage: $0 OUT_DIRECTORY HIP_GUARD_BINARY SOURCE_WORKTREE SOURCE_BINARY SOURCE_COMMIT --confirm-single-window" >&2
  exit 64
fi
if [ "$(id -u)" -ne 0 ]; then
  echo "this driver must run as root" >&2
  exit 65
fi

OUT=$1
GUARD_BIN=$2
SOURCE_TREE=$3
SOURCE_BIN=$4
SOURCE_COMMIT=$5
REPO=/home/homelab1/coding-local/ultimateLLM/uLLM-project
PACKAGE=/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package
PACKAGE_MANIFEST="$PACKAGE/manifest.json"
CASES="$REPO/tests/fixtures/qwen35-aq4-p2-oracle/cases.json"
SOURCE_ORACLE="$REPO/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/source-oracle-v2"
BASELINE_PATH_ORACLE="$REPO/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/path-oracle-v2"
TOKENIZER_ROOT=/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B
LOCK=/run/ullm/r9700.lock
SERVICE=ullm-openai.service
MANIFEST=/etc/ullm/served-models/active.json
EXPECTED_MANIFEST_SHA=feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44
REQUIRED_FIX_COMMIT=e992b3ea1d0427744dfd83abdc98283a74c1e3b4
STAGING_TOOL="$REPO/tools/stage-aq4-phase6-path-oracle-binary.py"
EXPORT_TOOL="$REPO/tools/export-qwen35-aq4-path-oracle.py"
COMPARISON_TOOL="$REPO/tools/compare-aq4-phase6-final-output.py"
GUARD_TOOL="$REPO/tools/run-aq4-phase3c-r9700-guard.py"
LOCK_PROBE="$REPO/tools/probe-aq4-phase3c-existing-lock.py"
STAGE_DIR="$OUT/path-oracle-binary-staging"
PATH_ORACLE_BIN="$STAGE_DIR/ullm-aq4-p2-path-oracle"

# This is the complete v0.7 Phase 3c guard set.  Phase 6 uses the same 17
# guard names as a mandatory subset, while the original path-oracle contract
# additionally needs every active production guard below.
PHASE3C_REQUIRED_GUARDS=(
  ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL
  ULLM_REQUIRE_HIP_RMSNORM_KERNEL
  ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL
  ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL
  ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL
  ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL
  ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL
  ULLM_REQUIRE_HIP_BF16_ROW_KERNEL
  ULLM_REQUIRE_HIP_TOP1_KERNEL
)

PHASE6_REQUIRED_GUARDS=(
  ULLM_REQUIRE_HIP_AQ4_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL
  ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL
  ULLM_REQUIRE_HIP_ADD_KERNEL
  ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL
  ULLM_REQUIRE_HIP_BF16_ROW_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL
  ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL
  ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL
  ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL
  ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL
  ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL
  ULLM_REQUIRE_HIP_RMSNORM_KERNEL
  ULLM_REQUIRE_HIP_ROPE_KERNEL
  ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL
  ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL
  ULLM_REQUIRE_HIP_SILU_MUL_KERNEL
  ULLM_REQUIRE_HIP_TOP1_KERNEL
)

PHASE6_ENV=(
  HOME=/home/homelab1
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  HIP_VISIBLE_DEVICES=1
  ULLM_HIP_VISIBLE_DEVICES=1
  ULLM_BUILD_GIT_COMMIT="$SOURCE_COMMIT"
  ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB=4
  ULLM_AQ4_MATVEC_SILU_MUL_RPB=8
  ULLM_AQ4_MATVEC_ADD_RPB=8
)
for guard in "${PHASE6_REQUIRED_GUARDS[@]}"; do
  PHASE6_ENV+=("$guard=1")
done

STOP_ATTEMPTED=0
RESTORE_DONE=0
RESTORE_RC=99
ORACLE_RC=not_started
AFTER_GUARD_RC=not_started
OUTCOME=not_started

event() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds --utc)" "$1" "$2" >> "$OUT/service-window-events.tsv"
}

write_result() {
  printf '%s\n' \
    "recorded_at_utc=$(date --iso-8601=seconds --utc)" \
    "outcome=$OUTCOME" \
    "path_oracle_exit_code=$ORACLE_RC" \
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

if ! [[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source commit must be a lowercase full SHA-1" >&2
  exit 66
fi
case "$SOURCE_TREE" in
  /*) ;;
  *) echo "source worktree must be an absolute path" >&2; exit 66 ;;
esac
case "$SOURCE_BIN" in
  "$SOURCE_TREE"/*) ;;
  *) echo "source binary must be under the clean source worktree" >&2; exit 66 ;;
esac
if [ ! -d "$OUT" ] || [ -L "$OUT" ] || [ -e "$OUT/path-oracle" ] || [ -e "$OUT/oracle-link" ] || [ -e "$OUT/final-output-comparison" ] || [ -e "$OUT/guard-before" ] || [ -e "$OUT/guard-after" ]; then
  echo "output contract failed" >&2
  exit 30
fi
if [ ! -f "$LOCK" ] || [ -L "$LOCK" ] || [ ! -x "$PATH_ORACLE_BIN" ] || [ ! -x "$GUARD_BIN" ] || [ ! -f "$STAGING_TOOL" ] || [ ! -f "$EXPORT_TOOL" ] || [ ! -f "$COMPARISON_TOOL" ] || [ ! -f "$GUARD_TOOL" ] || [ ! -f "$LOCK_PROBE" ] || [ ! -x /opt/rocm/bin/amd-smi ]; then
  echo "pre-stop file contract failed" >&2
  exit 31
fi
if [ ! -d "$SOURCE_TREE" ] || [ ! -x "$SOURCE_BIN" ] || [ ! -d "$PACKAGE" ] || [ ! -f "$PACKAGE_MANIFEST" ] || [ ! -f "$CASES" ] || [ ! -d "$SOURCE_ORACLE" ] || [ ! -d "$BASELINE_PATH_ORACLE" ] || [ ! -d "$TOKENIZER_ROOT" ] || [ ! -r "$MANIFEST" ]; then
  echo "input contract failed" >&2
  exit 32
fi
if [ "$(git -C "$SOURCE_TREE" rev-parse HEAD)" != "$SOURCE_COMMIT" ] || ! git -C "$SOURCE_TREE" diff --quiet || ! git -C "$SOURCE_TREE" diff --cached --quiet || [ -n "$(git -C "$SOURCE_TREE" status --porcelain --untracked-files=all)" ]; then
  echo "clean source worktree contract failed" >&2
  exit 34
fi
if ! git -C "$SOURCE_TREE" merge-base --is-ancestor "$REQUIRED_FIX_COMMIT" "$SOURCE_COMMIT"; then
  echo "source worktree does not include the final RMSNorm fix commit" >&2
  exit 34
fi
for tool in \
  tools/stage-aq4-phase6-path-oracle-binary.py \
  tools/compare-aq4-phase6-final-output.py \
  tools/run-aq4-phase6-service-window.sh \
  tools/export-qwen35-aq4-path-oracle.py \
  tools/run-aq4-phase3c-r9700-guard.py \
  tools/probe-aq4-phase3c-existing-lock.py; do
  if ! git -C "$REPO" ls-files --error-unmatch "$tool" >/dev/null || ! git -C "$REPO" diff --quiet HEAD -- "$tool" || ! git -C "$REPO" diff --cached --quiet HEAD -- "$tool"; then
    echo "window tooling is not tracked and clean: $tool" >&2
    exit 35
  fi
done
if ! python3 "$STAGING_TOOL" \
  --verify \
  --source "$SOURCE_BIN" \
  --output "$STAGE_DIR" \
  --source-commit "$SOURCE_COMMIT" \
  > "$OUT/path-oracle-binary-staging-verify-window.json"; then
  echo "staged path-oracle binary identity contract failed" >&2
  exit 37
fi
if ! python3 - "$MANIFEST" "$OUT/phase6-guard-contract.json" <<'PY'
import json
import sys

manifest_path, output_path = sys.argv[1:]
phase3c = [
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
phase6 = [
    "ULLM_REQUIRE_HIP_AQ4_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL",
    "ULLM_REQUIRE_HIP_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL",
    "ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL",
    "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL",
    "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL",
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
    "ULLM_REQUIRE_HIP_ROPE_KERNEL",
    "ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL",
    "ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL",
    "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL",
    "ULLM_REQUIRE_HIP_TOP1_KERNEL",
]
payload = json.load(open(manifest_path, encoding="utf-8"))
actual = payload["worker"]["required_environment"]
assert isinstance(actual, list)
assert len(actual) == len(set(actual))
assert set(actual) == set(phase6)
assert set(phase3c).issubset(set(phase6))
json.dump(
    {
        "schema_version": "ullm.aq4_phase6_guard_contract.v1",
        "status": "valid",
        "phase3c_required_guards": phase3c,
        "phase6_required_guards": phase6,
        "active_manifest_required_guards": actual,
        "phase3c_subset_of_phase6": True,
    },
    open(output_path, "w", encoding="utf-8"),
    indent=2,
    sort_keys=True,
)
PY
then
  echo "Phase 6 production guard contract failed before service stop" >&2
  exit 39
fi
if ! python3 - "$OUT/baseline-metric-reproduction/comparison.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["schema_version"] == "ullm.aq4_phase6_final_output_comparison.v1"
assert payload["status"] == "valid"
assert abs(payload["before_fix"]["agreement"]["logit_sample_bounded_relative_l2_max"] - 0.6151289249025698) <= 1e-12
PY
then
  echo "legacy Phase 6 baseline reproduction is unavailable or invalid" >&2
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
  "source_commit=$SOURCE_COMMIT" \
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

runuser -u homelab1 -- /usr/bin/python3 "$LOCK_PROBE" "$OUT/service-window-lock-after-stop.json" "$LOCK"
probe_rc=$?
event post_stop_lock_probe "exit_code=$probe_rc"
if [ "$probe_rc" -ne 0 ]; then
  OUTCOME=lock_acquisition_failure_no_trace
  restore
  write_result
  exit 40
fi

/usr/bin/python3 "$GUARD_TOOL" --output "$OUT/guard-before" --guard-bin "$GUARD_BIN" --health-phase before
guard_rc=$?
event r9700_architecture_and_health_guard "exit_code=$guard_rc"
if [ "$guard_rc" -ne 0 ]; then
  OUTCOME=architecture_guard_failure_no_trace
  restore
  write_result
  exit 41
fi

oracle_started_at="$(date --iso-8601=seconds --utc)"
event path_oracle_invoked ""
runuser -u homelab1 -- /usr/bin/env -i "${PHASE6_ENV[@]}" \
  /bin/bash -ceu '
    lock=$1
    exporter=$2
    package=$3
    package_manifest=$4
    cases=$5
    source_oracle=$6
    tokenizer_root=$7
    binary=$8
    output=$9
    link_output=${10}
    manifest=${11}
    exec 9< "$lock"
    flock -n 9
    exec /usr/bin/python3 "$exporter" \
      --package-dir "$package" \
      --package-manifest "$package_manifest" \
      --allow-package-only \
      --cases "$cases" \
      --source-oracle "$source_oracle" \
      --tokenizer-root "$tokenizer_root" \
      --output "$output" \
      --link-output "$link_output" \
      --binary "$binary" \
      --served-model-manifest "$manifest" \
      --evidence-class production \
      --device-kind gpu \
      --device-index 1 \
      --visible-devices 1 \
      --prefill-m 1 \
      --timeout-seconds 3600
  ' bash "$LOCK" "$EXPORT_TOOL" "$PACKAGE" "$PACKAGE_MANIFEST" "$CASES" "$SOURCE_ORACLE" "$TOKENIZER_ROOT" "$PATH_ORACLE_BIN" "$OUT/path-oracle" "$OUT/oracle-link" "$MANIFEST" \
  > "$OUT/path-oracle-export.stdout" \
  2> "$OUT/path-oracle-export.stderr"
ORACLE_RC=$?
oracle_finished_at="$(date --iso-8601=seconds --utc)"
printf '%s\n' \
  "started_at_utc=$oracle_started_at" \
  "finished_at_utc=$oracle_finished_at" \
  "exit_code=$ORACLE_RC" \
  "lock_open_policy=read_only_existing_file_no_create" \
  "model_load_contract=one_path_oracle_model_load" \
  > "$OUT/path-oracle.exit-status.txt"
event path_oracle_returned "exit_code=$ORACLE_RC"

/usr/bin/python3 "$GUARD_TOOL" --output "$OUT/guard-after" --guard-bin "$GUARD_BIN" --health-phase after
AFTER_GUARD_RC=$?
event post_trace_health_guard "exit_code=$AFTER_GUARD_RC"
restore
if [ "$RESTORE_RC" -ne 0 ]; then
  OUTCOME=restore_start_failure
  write_result
  exit 70
fi

post_active_state="$(systemctl show "$SERVICE" -p ActiveState --value)"
post_sub_state="$(systemctl show "$SERVICE" -p SubState --value)"
post_main_pid="$(systemctl show "$SERVICE" -p MainPID --value)"
post_restarts="$(systemctl show "$SERVICE" -p NRestarts --value)"
post_manifest_sha="$(sha256sum "$MANIFEST" | awk '{print $1}')"
printf '%s\n' \
  "captured_at_utc=$(date --iso-8601=seconds --utc)" \
  "active_state=$post_active_state" \
  "sub_state=$post_sub_state" \
  "main_pid=$post_main_pid" \
  "nrestarts=$post_restarts" \
  "manifest_sha256=$post_manifest_sha" \
  "pre_stop_nrestarts=$restarts" \
  > "$OUT/service-window-post-restore.txt"
if [ "$post_active_state" != active ] || [ "$post_sub_state" != running ] || [ "$post_main_pid" -le 0 ] || [ "$post_restarts" != "$restarts" ] || [ "$post_manifest_sha" != "$manifest_sha" ]; then
  OUTCOME=post_restore_service_contract_failure
  write_result
  exit 72
fi
if [ "$ORACLE_RC" -ne 0 ]; then
  OUTCOME=path_oracle_failure_no_retry
  write_result
  exit "$ORACLE_RC"
fi
if [ "$AFTER_GUARD_RC" -ne 0 ]; then
  OUTCOME=post_trace_health_guard_failure
  write_result
  exit 71
fi
if ! runuser -u homelab1 -- /usr/bin/env -i HOME=/home/homelab1 PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /usr/bin/python3 "$COMPARISON_TOOL" \
    --source-oracle "$SOURCE_ORACLE" \
    --baseline-path-oracle "$BASELINE_PATH_ORACLE" \
    --after-path-oracle "$OUT/path-oracle" \
    --cases "$CASES" \
    --output "$OUT/final-output-comparison" \
    > "$OUT/final-output-comparison.stdout" \
    2> "$OUT/final-output-comparison.stderr"; then
  OUTCOME=post_restore_comparison_failure
  write_result
  exit 73
fi
OUTCOME=path_oracle_completed_and_compared
write_result
exit 0
