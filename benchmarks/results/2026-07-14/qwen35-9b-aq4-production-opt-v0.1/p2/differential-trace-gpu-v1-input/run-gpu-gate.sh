#!/usr/bin/env bash
set -Eeuo pipefail
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PATH

REPO="/home/homelab1/coding-local/ultimateLLM/uLLM-project"
BASE="$REPO/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2"
INPUT_DIR="$BASE/differential-trace-gpu-v1-input"
OUTPUT="$BASE/differential-trace-gpu-v1"
GATE_LOG="$BASE/differential-trace-gpu-v1-gate.log"
MONITOR_LOG="$BASE/differential-trace-gpu-v1-monitor.log"
RUN_LOG="$BASE/differential-trace-gpu-v1-run.log"
STOP_MARKER="$BASE/differential-trace-gpu-v1-service-stopped.marker"
OBSERVER_FAIL_MARKER="$BASE/differential-trace-gpu-v1-observer-failed.marker"
OBSERVER_SAMPLE_MARKER="$BASE/differential-trace-gpu-v1-observer-sample.marker"
RUN_STARTED_MARKER="$BASE/differential-trace-gpu-v1-run-started.marker"
REQUIRED_COMMIT="28ec343ac59e6d22e710035d7874df9fbd8f890f"
SOURCE_BIN="$REPO/target/release/ullm-aq4-differential-trace"
CANDIDATE_BIN="$INPUT_DIR/ullm-aq4-differential-trace-detached"
EXPECTED_SOURCE_SHA="356d131fc578debea418f6c67d7b89272bfb02700495775be98471c44e3bd0b7"
EXPECTED_CASES_SHA="15fed90dd2e16a5b68d4498c8632257d80ac94c56ed614696b0884c65f4836f2"
EXPECTED_REPLAY_SHA="1ee0b9228e1bc3a0ae9175e5693bf3770f9b89e872349554562dbd4b6b4747dc"
PACKAGE="/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1"
CASES="$REPO/tests/fixtures/qwen35-aq4-p2-oracle/cases.json"
REPLAY="$INPUT_DIR/replay.json"
ACTIVE="/etc/ullm/served-models/active.json"
WORKER="$REPO/target/reasoning-v2/release/ullm-aq4-worker"
LOCK="/run/ullm/r9700.lock"
RUNTIME_DIR="${LOCK%/*}"
ROCM_SMI_LINK="/opt/rocm/bin/rocm-smi"
ROCM_SMI_REAL_EXPECTED="/opt/rocm-7.2.1/libexec/rocm_smi/rocm_smi.py"
ROCM_SMI_REAL="$(realpath "$ROCM_SMI_LINK")"
ROCM_SMI_SHA_EXPECTED="5a64729944c9bcd7eccd647ddc39cca1a942b3a51cf234fa1cb59a7b20591f46"
ROCM_PYTHON="/usr/bin/python3.12"
ROCM_PYTHON_SHA_EXPECTED="1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
ROCM_PYTHON_VERSION_EXPECTED="Python 3.12.3"
EXPECTED_ACTIVE_SHA="feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44"
EXPECTED_PACKAGE_SHA="a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad"
EXPECTED_WORKER_SHA="177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d"

[[ "$(git -C "$REPO" hash-object crates/ullm-engine/src/bin/ullm-aq4-differential-trace.rs)" = 73bbaf50eb04b9c3dc4ac934b02e3dcf79bab8ca ]]
[[ -x "$SOURCE_BIN" && "$(sha256sum "$SOURCE_BIN" | awk '{print $1}')" = "$EXPECTED_SOURCE_SHA" ]]
[[ "$ROCM_SMI_REAL" = "$ROCM_SMI_REAL_EXPECTED" ]]
[[ -f "$ROCM_SMI_REAL" && "$(stat -Lc '%F:%h' "$ROCM_SMI_REAL")" = 'regular file:1' ]]
[[ "$(sha256sum "$ROCM_SMI_REAL" | awk '{print $1}')" = "$ROCM_SMI_SHA_EXPECTED" ]]
[[ -x "$ROCM_PYTHON" && "$(sha256sum "$ROCM_PYTHON" | awk '{print $1}')" = "$ROCM_PYTHON_SHA_EXPECTED" ]]
[[ "$("$ROCM_PYTHON" --version 2>&1)" = "$ROCM_PYTHON_VERSION_EXPECTED" ]]
[[ -f "$CASES" && "$(sha256sum "$CASES" | awk '{print $1}')" = "$EXPECTED_CASES_SHA" ]]
[[ -f "$REPLAY" && "$(sha256sum "$REPLAY" | awk '{print $1}')" = "$EXPECTED_REPLAY_SHA" ]]
[[ "$(sha256sum "$ACTIVE" | awk '{print $1}')" = "$EXPECTED_ACTIVE_SHA" ]]
[[ "$(sha256sum "$PACKAGE/package/manifest.json" | awk '{print $1}')" = "$EXPECTED_PACKAGE_SHA" ]]
[[ "$(sha256sum "$WORKER" | awk '{print $1}')" = "$EXPECTED_WORKER_SHA" ]]
[[ "$(stat -Lc '%F:%h' "$ACTIVE")" = 'regular file:1' ]]
[[ "$(stat -Lc '%F:%h' "$PACKAGE/package/manifest.json")" = 'regular file:1' ]]
[[ "$(stat -Lc '%F:%h' "$WORKER")" = 'regular file:2' ]]
[[ ! -e "$OUTPUT" && ! -e "$GATE_LOG" && ! -e "$MONITOR_LOG" && ! -e "$RUN_LOG" && ! -e "$STOP_MARKER" && ! -e "$OBSERVER_FAIL_MARKER" && ! -e "$OBSERVER_SAMPLE_MARKER" && ! -e "$RUN_STARTED_MARKER" ]]
[[ ! -e "$CANDIDATE_BIN" ]]
"$ROCM_PYTHON" - "$ACTIVE" <<'PY'
import json, sys
expected = [
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
actual = json.load(open(sys.argv[1]))["worker"]["required_environment"]
assert actual == expected, (len(actual), actual)
print("required_environment_count=30")
PY

if [[ "${PREFLIGHT_ONLY:-0}" != 1 && "${PREFLIGHT_LOCKED_ONLY:-0}" != 1 ]]; then
  exec > >(tee "$GATE_LOG") 2>&1
fi
echo "gate_started=$(date --iso-8601=seconds)"
echo "required_commit=$REQUIRED_COMMIT"
echo "source_bin=$SOURCE_BIN"
echo "source_bin_sha256=$(sha256sum "$SOURCE_BIN" | awk '{print $1}')"
echo "cases_sha256=$(sha256sum "$CASES" | awk '{print $1}')"
echo "replay_sha256=$(sha256sum "$REPLAY" | awk '{print $1}')"
echo "rocm_smi_link=$ROCM_SMI_LINK"
echo "rocm_smi_real=$ROCM_SMI_REAL"
echo "rocm_smi_stat=$(stat -Lc '%F:%h:%s:%i' "$ROCM_SMI_REAL")"
echo "rocm_smi_sha256=$(sha256sum "$ROCM_SMI_REAL" | awk '{print $1}')"
echo "rocm_smi_shebang=$(head -1 "$ROCM_SMI_REAL")"
echo "rocm_smi_version=$("$ROCM_PYTHON" "$ROCM_SMI_REAL" --version 2>&1 | head -1)"
echo "rocm_python=$ROCM_PYTHON"
echo "rocm_python_version=$("$ROCM_PYTHON" --version 2>&1)"
echo "rocm_python_sha256=$(sha256sum "$ROCM_PYTHON" | awk '{print $1}')"
echo "active_manifest_sha256=$(sha256sum "$ACTIVE" | awk '{print $1}')"
echo "package_manifest_sha256=$(sha256sum "$PACKAGE/package/manifest.json" | awk '{print $1}')"
echo "worker_sha256=$(sha256sum "$WORKER" | awk '{print $1}')"
ACTIVE_STAT="$(stat -Lc '%D:%i:%h:%s:%Y:%Z:%F' "$ACTIVE")"
PACKAGE_STAT="$(stat -Lc '%D:%i:%h:%s:%Y:%Z:%F' "$PACKAGE/package/manifest.json")"
WORKER_STAT="$(stat -Lc '%D:%i:%h:%s:%Y:%Z:%F' "$WORKER")"
echo "worker_stat_pre=$WORKER_STAT"
EXPECTED_RUNTIME_UID="$(id -u homelab1)"
EXPECTED_RUNTIME_GID="$(id -g homelab1)"
RUNTIME_DIR_TYPE="$(stat -Lc '%F' "$RUNTIME_DIR")"
RUNTIME_DIR_MODE="$(stat -Lc '%a' "$RUNTIME_DIR")"
RUNTIME_DIR_UID="$(stat -Lc '%u' "$RUNTIME_DIR")"
RUNTIME_DIR_GID="$(stat -Lc '%g' "$RUNTIME_DIR")"
RUNTIME_DIR_NLINK="$(stat -Lc '%h' "$RUNTIME_DIR")"
LOCK_TYPE="$(stat -Lc '%F' "$LOCK")"
LOCK_MODE="$(stat -Lc '%a' "$LOCK")"
LOCK_UID="$(stat -Lc '%u' "$LOCK")"
LOCK_GID="$(stat -Lc '%g' "$LOCK")"
LOCK_NLINK="$(stat -Lc '%h' "$LOCK")"
LOCK_SIZE="$(stat -Lc '%s' "$LOCK")"
RUNTIME_DIR_IDENTITY="$RUNTIME_DIR_TYPE:$RUNTIME_DIR_MODE:$RUNTIME_DIR_UID:$RUNTIME_DIR_GID"
LOCK_IDENTITY="$LOCK_TYPE:$LOCK_MODE:$LOCK_UID:$LOCK_GID"
echo "runtime_dir_pre=$RUNTIME_DIR_IDENTITY"
echo "runtime_lock_pre=$LOCK_IDENTITY"
echo "runtime_expected_uid_gid=$EXPECTED_RUNTIME_UID:$EXPECTED_RUNTIME_GID"
echo "runtime_dir_pre_nlink=$RUNTIME_DIR_NLINK"
echo "runtime_lock_pre_nlink_size=$LOCK_NLINK:$LOCK_SIZE"
[[ ! -L "$RUNTIME_DIR" && ! -L "$LOCK" ]]
[[ "$RUNTIME_DIR_TYPE" = directory && "$RUNTIME_DIR_MODE" = 750 && "$RUNTIME_DIR_UID" = "$EXPECTED_RUNTIME_UID" && "$RUNTIME_DIR_GID" = "$EXPECTED_RUNTIME_GID" && "$RUNTIME_DIR_NLINK" = 2 ]]
[[ "$LOCK_TYPE" = regular* && "$LOCK_MODE" = 600 && "$LOCK_UID" = "$EXPECTED_RUNTIME_UID" && "$LOCK_GID" = "$EXPECTED_RUNTIME_GID" && "$LOCK_NLINK" = 1 && "$LOCK_SIZE" = 0 ]]
NRESTARTS_BEFORE="$(systemctl show ullm-openai.service -p NRestarts --value)"
[[ "$(systemctl is-active ullm-openai.service)" = active ]]
SERVICE_MAINPID="$(systemctl show ullm-openai.service -p MainPID --value)"
LOCK_OWNER_PID="$(lslocks -o PID,PATH 2>/dev/null | awk -v path="$LOCK" '$2 == path {print $1; exit}')"
LOCK_OWNER_COMMAND="$(ps -p "$LOCK_OWNER_PID" -o comm= 2>/dev/null || true)"
echo "service_mainpid=$SERVICE_MAINPID"
echo "lock_owner_pid=$LOCK_OWNER_PID"
echo "lock_owner_command=$LOCK_OWNER_COMMAND"
[[ "$LOCK_OWNER_PID" = "$SERVICE_MAINPID" ]]
[[ "$LOCK_OWNER_COMMAND" == ullm-openai-gat* ]]
systemctl show ullm-openai.service -p ActiveState -p SubState -p NRestarts -p MainPID
pgrep -af 'ullm-(openai|aq4-worker)' || true
"$ROCM_PYTHON" "$ROCM_SMI_REAL" --showproductname --showmeminfo vram --showuse --json
docker exec open-webui curl --max-time 5 -fsS http://172.20.0.1:8000/healthz
echo
API_KEY="$(cat /etc/ullm/openai-api-key)"
docker exec -e API_KEY="$API_KEY" open-webui sh -lc 'curl --max-time 5 -fsS -H "Authorization: Bearer $API_KEY" http://172.20.0.1:8000/v1/models'
echo

if [[ "${PREFLIGHT_ONLY:-0}" = 1 ]]; then
  echo "preflight_only=1"
  echo "preflight_stop_run=skipped"
  exit 0
fi

if [[ "${PREFLIGHT_LOCKED_ONLY:-0}" = 1 ]]; then
  LOCKED_TMP="$(mktemp -d /tmp/ullm-aq4-lock-owner.XXXXXX)"
  locked_observer=""
  locked_cleanup() {
    if [[ -n "$locked_observer" ]]; then
      kill "$locked_observer" 2>/dev/null || true
      wait "$locked_observer" 2>/dev/null || true
    fi
    rm -rf "$LOCKED_TMP"
  }
  trap locked_cleanup EXIT INT TERM
  (
    while :; do
      if ! "$ROCM_PYTHON" "$ROCM_SMI_REAL" --showmeminfo vram --showuse --showpower --json >"$LOCKED_TMP/sample.json" 2>&1; then
        exit 1
      fi
      printf '%s\n' "$(date +%s%N)" >"$LOCKED_TMP/sample.ts"
      sleep 1
    done
  ) >"$LOCKED_TMP/observer.log" 2>&1 &
  locked_observer=$!
  for _ in $(seq 1 20); do
    [[ -s "$LOCKED_TMP/sample.ts" ]] && break
    sleep 0.1
  done
  [[ -s "$LOCKED_TMP/sample.ts" ]]
  kill -0 "$locked_observer"
  locked_sample_ts="$(cat "$LOCKED_TMP/sample.ts")"
  locked_now_ts="$(date +%s%N)"
  [[ "$locked_sample_ts" =~ ^[0-9]+$ ]] && (( locked_now_ts >= locked_sample_ts && locked_now_ts - locked_sample_ts <= 5000000000 ))
  [[ "$(lslocks -o PID,PATH 2>/dev/null | awk -v path="$LOCK" '$2 == path {print $1; exit}')" = "$SERVICE_MAINPID" ]]
  kill "$locked_observer" 2>/dev/null || true
  wait "$locked_observer" 2>/dev/null || true
  locked_observer=""
  echo "lock_owner_preflight=expected_service_mainpid"
  echo "locked_preflight_systemctl_mutations=0"
  echo "locked_preflight_observer_cleanup=complete"
  exit 0
fi

pre_stop_cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  rm -f "$CANDIDATE_BIN" "$GATE_LOG" "$MONITOR_LOG" "$RUN_LOG" "$STOP_MARKER" "$OBSERVER_FAIL_MARKER" "$OBSERVER_SAMPLE_MARKER" "$RUN_STARTED_MARKER"
  exit "$rc"
}
trap pre_stop_cleanup EXIT INT TERM

# Cargo hard-links target/release and target/release/deps. The executable used by the trace
# must have nlink=1 so that required_regular_sha256 can attest the exact executable safely.
install -m 0755 "$SOURCE_BIN" "$CANDIDATE_BIN"
[[ "$(sha256sum "$CANDIDATE_BIN" | awk '{print $1}')" = "$EXPECTED_SOURCE_SHA" ]]
[[ "$(stat -Lc '%F:%h' "$CANDIDATE_BIN")" = 'regular file:1' ]]
EMBEDDED_MATCH="$(strings -a "$CANDIDATE_BIN" | grep -F "$REQUIRED_COMMIT" | head -1 || true)"
[[ -n "$EMBEDDED_MATCH" ]]
echo "candidate_bin=$CANDIDATE_BIN"
echo "candidate_bin_sha256=$(sha256sum "$CANDIDATE_BIN" | awk '{print $1}')"
echo "candidate_bin_stat=$(stat -Lc '%F:%h:%s:%i' "$CANDIDATE_BIN")"
echo "embedded_commit_match=$EMBEDDED_MATCH"

export REQUIRED_COMMIT ACTIVE CANDIDATE_BIN PACKAGE CASES REPLAY OUTPUT MONITOR_LOG RUN_LOG STOP_MARKER OBSERVER_FAIL_MARKER OBSERVER_SAMPLE_MARKER RUN_STARTED_MARKER ROCM_SMI_REAL ROCM_PYTHON EXPECTED_ACTIVE_SHA EXPECTED_PACKAGE_SHA EXPECTED_WORKER_SHA ACTIVE_STAT PACKAGE_STAT WORKER_STAT
RESTORE_NEEDED=0
RESTORE_RC=0
STOP_ARMED=0
RUN_STARTED=0
RUNTIME_DIR_CREATED=0
LOCK_CREATED=0
RUNTIME_DIR_CREATED_DEV_INO=""
LOCK_CREATED_DEV_INO=""
observer=""
stop_observer() {
  if [[ -n "$observer" ]]; then
    kill "$observer" 2>/dev/null || true
    wait "$observer" 2>/dev/null || true
    observer=""
  fi
}
cleanup_runtime_artifacts() {
  local ok=1 lock_owner lock_type lock_mode lock_uid lock_gid lock_nlink lock_size lock_dev_ino dir_type dir_mode dir_uid dir_gid dir_nlink dir_dev_ino dir_child
  if (( LOCK_CREATED )); then
    lock_owner="$(lslocks -o PID,PATH 2>/dev/null | awk -v path="$LOCK" '$2 == path {print $1; exit}')"
    if [[ -n "$lock_owner" ]]; then
      echo "refusing to remove runtime lock held by pid=$lock_owner" >&2
      ok=0
    elif [[ -L "$LOCK" ]]; then
      echo "refusing to remove symlink runtime lock" >&2
      ok=0
    elif [[ ! -e "$LOCK" && ! -L "$LOCK" ]]; then
      echo "runtime_lock_already_absent=1"
    else
      lock_type="$(stat -Lc '%F' "$LOCK" 2>/dev/null || true)"
      lock_mode="$(stat -Lc '%a' "$LOCK" 2>/dev/null || true)"
      lock_uid="$(stat -Lc '%u' "$LOCK" 2>/dev/null || true)"
      lock_gid="$(stat -Lc '%g' "$LOCK" 2>/dev/null || true)"
      lock_nlink="$(stat -Lc '%h' "$LOCK" 2>/dev/null || true)"
      lock_size="$(stat -Lc '%s' "$LOCK" 2>/dev/null || true)"
      lock_dev_ino="$(stat -Lc '%D:%i' "$LOCK" 2>/dev/null || true)"
      if [[ "$lock_type" = regular* && "$lock_mode" = "$LOCK_MODE" && "$lock_uid" = "$LOCK_UID" && "$lock_gid" = "$LOCK_GID" && "$lock_nlink" = 1 && "$lock_size" = 0 && "$lock_dev_ino" = "$LOCK_CREATED_DEV_INO" ]]; then
        if rm -f -- "$LOCK"; then
          echo "runtime_lock_removed=1"
        else
          echo "failed to remove owned runtime lock" >&2
          ok=0
        fi
      else
        echo "refusing to remove runtime lock with changed identity" >&2
        ok=0
      fi
    fi
  fi
  if (( RUNTIME_DIR_CREATED )); then
    if [[ ! -e "$RUNTIME_DIR" && ! -L "$RUNTIME_DIR" ]]; then
      echo "runtime_dir_already_absent=1"
    elif [[ -L "$RUNTIME_DIR" ]]; then
      echo "refusing to remove symlink runtime directory" >&2
      ok=0
    else
      dir_type="$(stat -Lc '%F' "$RUNTIME_DIR" 2>/dev/null || true)"
      dir_mode="$(stat -Lc '%a' "$RUNTIME_DIR" 2>/dev/null || true)"
      dir_uid="$(stat -Lc '%u' "$RUNTIME_DIR" 2>/dev/null || true)"
      dir_gid="$(stat -Lc '%g' "$RUNTIME_DIR" 2>/dev/null || true)"
      dir_nlink="$(stat -Lc '%h' "$RUNTIME_DIR" 2>/dev/null || true)"
      dir_dev_ino="$(stat -Lc '%D:%i' "$RUNTIME_DIR" 2>/dev/null || true)"
      dir_child="$(find "$RUNTIME_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)"
      if [[ "$dir_type" = directory && "$dir_mode" = "$RUNTIME_DIR_MODE" && "$dir_uid" = "$RUNTIME_DIR_UID" && "$dir_gid" = "$RUNTIME_DIR_GID" && "$dir_nlink" = "$RUNTIME_DIR_NLINK" && "$dir_dev_ino" = "$RUNTIME_DIR_CREATED_DEV_INO" && -z "$dir_child" ]]; then
        if rmdir -- "$RUNTIME_DIR"; then
          echo "runtime_dir_removed=1"
        else
          echo "failed to remove owned runtime directory" >&2
          ok=0
        fi
      else
        echo "refusing to remove runtime directory with changed identity or contents" >&2
        ok=0
      fi
    fi
  fi
  (( ok == 1 ))
}
restore_service() {
  set +e
  if (( RESTORE_NEEDED )); then
    local restore_ok=1
    echo "restore_started=$(date --iso-8601=seconds)"
    local restore_deadline_ns restore_now_ns restore_remaining_ns
    restore_deadline_ns="$("$ROCM_PYTHON" - <<'PY'
import time
print(time.monotonic_ns() + 60000000000)
PY
)"
    /usr/bin/timeout --signal=TERM --kill-after=5s 60s systemctl start ullm-openai.service
    local start_rc=$?
    (( start_rc == 0 )) || restore_ok=0
    local health_ok=0 restored_state="" restored_substate="" restored_mainpid="" restored_lock_owner=""
    while :; do
      restore_now_ns="$("$ROCM_PYTHON" - <<'PY'
import time
print(time.monotonic_ns())
PY
)"
      restore_remaining_ns=$((restore_deadline_ns - restore_now_ns))
      (( restore_remaining_ns > 1000000000 )) || break
      restored_state="$(systemctl is-active ullm-openai.service || true)"
      restored_substate="$(systemctl show ullm-openai.service -p SubState --value 2>/dev/null || true)"
      restored_mainpid="$(systemctl show ullm-openai.service -p MainPID --value 2>/dev/null || true)"
      restored_lock_owner="$(lslocks -o PID,PATH 2>/dev/null | awk -v path="$LOCK" '$2 == path {print $1; exit}')"
      if [[ "$restored_state" = active && "$restored_substate" = running && "$restored_mainpid" =~ ^[1-9][0-9]*$ && "$restored_lock_owner" = "$restored_mainpid" ]]; then
        restore_now_ns="$("$ROCM_PYTHON" - <<'PY'
import time
print(time.monotonic_ns())
PY
)"
        restore_remaining_ns=$((restore_deadline_ns - restore_now_ns))
        if (( restore_remaining_ns > 1000000000 )) \
            && docker exec open-webui curl --connect-timeout 1 --max-time 1 -fsS http://172.20.0.1:8000/healthz >/dev/null 2>&1; then
          restore_now_ns="$("$ROCM_PYTHON" - <<'PY'
import time
print(time.monotonic_ns())
PY
)"
          restore_remaining_ns=$((restore_deadline_ns - restore_now_ns))
          if (( restore_remaining_ns > 1000000000 )) \
              && docker exec -e API_KEY="$API_KEY" open-webui sh -lc 'curl --connect-timeout 1 --max-time 1 -fsS -H "Authorization: Bearer $API_KEY" http://172.20.0.1:8000/v1/models' >/dev/null 2>&1; then
            health_ok=1
            break
          fi
        fi
      fi
      sleep 0.2
    done
    echo "restore_rc=$start_rc"
    echo "service_restored=$restored_state"
    echo "service_restored_substate=$restored_substate"
    echo "service_restored_mainpid=$restored_mainpid"
    echo "service_restored_lock_owner=$restored_lock_owner"
    echo "restore_health_ready=$health_ok"
    (( health_ok )) || restore_ok=0
    systemctl show ullm-openai.service -p ActiveState -p SubState -p NRestarts -p MainPID
    local post_nrestarts
    post_nrestarts="$(systemctl show ullm-openai.service -p NRestarts --value)"
    [[ "$post_nrestarts" = "$NRESTARTS_BEFORE" ]] || {
      echo "NRestarts changed: before=$NRESTARTS_BEFORE after=$post_nrestarts"
      restore_ok=0
    }
    echo "active_manifest_sha256_after=$(sha256sum "$ACTIVE" | awk '{print $1}')"
    echo "package_manifest_sha256_after=$(sha256sum "$PACKAGE/package/manifest.json" | awk '{print $1}')"
    echo "worker_sha256_after=$(sha256sum "$WORKER" | awk '{print $1}')"
    [[ "$(stat -Lc '%D:%i:%h:%s:%Y:%Z:%F' "$ACTIVE")" = "$ACTIVE_STAT" ]] || restore_ok=0
    [[ "$(stat -Lc '%D:%i:%h:%s:%Y:%Z:%F' "$PACKAGE/package/manifest.json")" = "$PACKAGE_STAT" ]] || restore_ok=0
    [[ "$(stat -Lc '%D:%i:%h:%s:%Y:%Z:%F' "$WORKER")" = "$WORKER_STAT" ]] || restore_ok=0
    [[ "$(sha256sum "$ACTIVE" | awk '{print $1}')" = "$EXPECTED_ACTIVE_SHA" ]] || restore_ok=0
    [[ "$(sha256sum "$PACKAGE/package/manifest.json" | awk '{print $1}')" = "$EXPECTED_PACKAGE_SHA" ]] || restore_ok=0
    [[ "$(sha256sum "$WORKER" | awk '{print $1}')" = "$EXPECTED_WORKER_SHA" ]] || restore_ok=0
    [[ "$(sha256sum "$ROCM_SMI_REAL" | awk '{print $1}')" = "$ROCM_SMI_SHA_EXPECTED" ]] || restore_ok=0
    [[ "$(stat -Lc '%F:%h' "$ROCM_SMI_REAL")" = 'regular file:1' ]] || restore_ok=0
    [[ ! -L "$RUNTIME_DIR" && ! -L "$LOCK" ]] || restore_ok=0
    [[ "$(stat -Lc '%F:%a:%u:%g' "$RUNTIME_DIR" 2>/dev/null || true)" = "directory:$RUNTIME_DIR_MODE:$RUNTIME_DIR_UID:$RUNTIME_DIR_GID" && "$(stat -Lc '%h' "$RUNTIME_DIR" 2>/dev/null || true)" = "$RUNTIME_DIR_NLINK" ]] || restore_ok=0
    [[ "$(stat -Lc '%F:%a:%u:%g' "$LOCK" 2>/dev/null || true)" = "regular empty file:$LOCK_MODE:$LOCK_UID:$LOCK_GID" && "$(stat -Lc '%h:%s' "$LOCK" 2>/dev/null || true)" = "$LOCK_NLINK:$LOCK_SIZE" ]] || restore_ok=0
    (( restore_ok )) || RESTORE_RC=1
  fi
}
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ -v LOCK_FD ]]; then
    eval "exec $LOCK_FD>&-"
    LOCK_FD=""
  fi
  stop_observer
  if ! cleanup_runtime_artifacts; then
    RESTORE_RC=1
  fi
  restore_service
  if [[ ! -e "$RUN_STARTED_MARKER" ]]; then
    rm -f "$CANDIDATE_BIN" "$GATE_LOG" "$MONITOR_LOG" "$RUN_LOG" "$STOP_MARKER" "$OBSERVER_FAIL_MARKER" "$OBSERVER_SAMPLE_MARKER" "$RUN_STARTED_MARKER"
  fi
  (( RESTORE_RC == 0 )) || rc=90
  exit "$rc"
}
trap cleanup EXIT INT TERM

RESTORE_NEEDED=1
if ! (umask 077; printf 'armed_at=%s\n' "$(date --iso-8601=seconds)" > "$STOP_MARKER"); then
  echo "failed to arm durable stop marker; refusing service stop" >&2
  exit 91
fi
[[ -s "$STOP_MARKER" ]]
STOP_ARMED=1

run_locked() {
  (
    while :; do
      sample_tmp="$OBSERVER_SAMPLE_MARKER.tmp.$$"
      if ! "$ROCM_PYTHON" "$ROCM_SMI_REAL" --showmeminfo vram --showuse --showpower --json >"$sample_tmp" 2>&1; then
        touch "$OBSERVER_FAIL_MARKER"
        rm -f "$sample_tmp"
        exit 1
      fi
      printf "sample=%s " "$(date --iso-8601=seconds)" >>"$MONITOR_LOG"
      cat "$sample_tmp" >>"$MONITOR_LOG"
      rm -f "$sample_tmp"
      sample_ts="$(date +%s%N)"
      printf '%s\n' "$sample_ts" >"$OBSERVER_SAMPLE_MARKER.tmp.$$"
      mv -f "$OBSERVER_SAMPLE_MARKER.tmp.$$" "$OBSERVER_SAMPLE_MARKER"
      sleep 1
    done
  ) >"$MONITOR_LOG" 2>&1 &
  observer=$!
  for _ in $(seq 1 20); do
    [[ -s "$OBSERVER_SAMPLE_MARKER" ]] && break
    sleep 0.1
  done
  [[ -s "$OBSERVER_SAMPLE_MARKER" ]]
  kill -0 "$observer"
  [[ ! -e "$OBSERVER_FAIL_MARKER" ]]
  sample_ts="$(cat "$OBSERVER_SAMPLE_MARKER")"
  now_ts="$(date +%s%N)"
  [[ "$sample_ts" =~ ^[0-9]+$ ]] && (( now_ts >= sample_ts && now_ts - sample_ts <= 5000000000 ))
  [[ "$(sha256sum "$ACTIVE" | awk '{print $1}')" = "$EXPECTED_ACTIVE_SHA" ]]
  [[ "$(sha256sum "$PACKAGE/package/manifest.json" | awk '{print $1}')" = "$EXPECTED_PACKAGE_SHA" ]]
  [[ "$(sha256sum "$WORKER" | awk '{print $1}')" = "$EXPECTED_WORKER_SHA" ]]
  [[ "$(stat -Lc '%D:%i:%h:%s:%Y:%Z:%F' "$ACTIVE")" = "$ACTIVE_STAT" ]]
  [[ "$(stat -Lc '%D:%i:%h:%s:%Y:%Z:%F' "$PACKAGE/package/manifest.json")" = "$PACKAGE_STAT" ]]
  [[ "$(stat -Lc '%D:%i:%h:%s:%Y:%Z:%F' "$WORKER")" = "$WORKER_STAT" ]]
  [[ ! -L "$RUNTIME_DIR" && ! -L "$LOCK" ]]
  [[ "$(stat -Lc '%F:%a:%u:%g' "$RUNTIME_DIR")" = "$RUNTIME_DIR_IDENTITY" ]]
  [[ "$(stat -Lc '%F:%a:%u:%g' "$LOCK")" = "$LOCK_IDENTITY" ]]
  echo "observer_ready_sample=$sample_ts"
  if ! printf 'service_stop_attempt_at=%s\n' "$(date --iso-8601=seconds)" >>"$STOP_MARKER"; then
    return 74
  fi
  if ! systemctl stop ullm-openai.service; then
    echo "systemctl stop failed; refusing candidate run" >&2
    return 75
  fi
  if ! printf 'service_stop_returned_at=%s\n' "$(date --iso-8601=seconds)" >>"$STOP_MARKER"; then
    return 76
  fi
  for _ in $(seq 1 30); do
    [[ "$(systemctl is-active ullm-openai.service || true)" = inactive ]] && ! pgrep -x ullm-aq4-worker >/dev/null && break
    sleep 1
  done
  [[ "$(systemctl is-active ullm-openai.service || true)" = inactive ]]
  ! pgrep -x ullm-aq4-worker >/dev/null
  echo "service_after_stop=inactive"
  "$ROCM_PYTHON" "$ROCM_SMI_REAL" --showproductname --showmeminfo vram --showuse --json
  for _ in $(seq 1 30); do
    [[ -z "$(lslocks -o PID,PATH 2>/dev/null | awk -v path="$LOCK" '$2 == path {print $1; exit}')" ]] && break
    sleep 0.2
  done
  [[ -z "$(lslocks -o PID,PATH 2>/dev/null | awk -v path="$LOCK" '$2 == path {print $1; exit}')" ]]
  if [[ -e "$RUNTIME_DIR" || -L "$RUNTIME_DIR" ]]; then
    echo "runtime directory remained after RuntimeDirectory=ullm stop; refusing foreign path" >&2
    return 80
  fi
  if ! (umask 077; mkdir -- "$RUNTIME_DIR"); then
    echo "failed to create diagnostic runtime directory" >&2
    return 81
  fi
  RUNTIME_DIR_CREATED=1
  if ! chown "$RUNTIME_DIR_UID:$RUNTIME_DIR_GID" "$RUNTIME_DIR" || ! chmod "$RUNTIME_DIR_MODE" "$RUNTIME_DIR"; then
    echo "failed to apply diagnostic runtime directory identity" >&2
    return 82
  fi
  [[ ! -L "$RUNTIME_DIR" && "$(stat -Lc '%F:%a:%u:%g' "$RUNTIME_DIR")" = "$RUNTIME_DIR_IDENTITY" && "$(stat -Lc '%h' "$RUNTIME_DIR")" = "$RUNTIME_DIR_NLINK" ]]
  RUNTIME_DIR_CREATED_DEV_INO="$(stat -Lc '%D:%i' "$RUNTIME_DIR")"
  printf 'runtime_dir_created_at=%s identity=%s nlink=%s dev_ino=%s\n' "$(date --iso-8601=seconds)" "$RUNTIME_DIR_IDENTITY" "$RUNTIME_DIR_NLINK" "$RUNTIME_DIR_CREATED_DEV_INO" >>"$STOP_MARKER"
  if [[ -e "$LOCK" || -L "$LOCK" ]]; then
    echo "runtime lock path appeared before diagnostic creation; refusing foreign path" >&2
    return 83
  fi
  if ! (umask 077; set -C; : >"$LOCK"); then
    echo "failed to create diagnostic runtime lock" >&2
    return 84
  fi
  LOCK_CREATED=1
  if ! chown "$LOCK_UID:$LOCK_GID" "$LOCK" || ! chmod "$LOCK_MODE" "$LOCK"; then
    echo "failed to apply diagnostic runtime lock identity" >&2
    return 85
  fi
  [[ ! -L "$LOCK" && "$(stat -Lc '%F:%a:%u:%g' "$LOCK")" = "regular empty file:$LOCK_MODE:$LOCK_UID:$LOCK_GID" && "$(stat -Lc '%h:%s' "$LOCK")" = "$LOCK_NLINK:$LOCK_SIZE" ]]
  LOCK_CREATED_DEV_INO="$(stat -Lc '%D:%i' "$LOCK")"
  printf 'runtime_lock_created_at=%s identity=regular empty file:%s:%s:%s nlink=%s size=%s dev_ino=%s\n' "$(date --iso-8601=seconds)" "$LOCK_MODE" "$LOCK_UID" "$LOCK_GID" "$LOCK_NLINK" "$LOCK_SIZE" "$LOCK_CREATED_DEV_INO" >>"$STOP_MARKER"
  exec {LOCK_FD}> "$LOCK"
  if ! flock -n "$LOCK_FD"; then
    echo "failed to acquire released R9700 lock" >&2
    return 79
  fi
  echo "lock_acquired_after_service_stop=$(date --iso-8601=seconds)"
  pre_run_sample="$(cat "$OBSERVER_SAMPLE_MARKER")"
  for _ in $(seq 1 10); do
    [[ ! -e "$OBSERVER_FAIL_MARKER" ]] || return 77
    kill -0 "$observer"
    current_sample="$(cat "$OBSERVER_SAMPLE_MARKER")"
    [[ "$current_sample" != "$pre_run_sample" ]] && break
    sleep 0.5
  done
  [[ ! -e "$OBSERVER_FAIL_MARKER" ]]
  kill -0 "$observer"
  [[ "$(cat "$OBSERVER_SAMPLE_MARKER")" != "$pre_run_sample" ]]
  run_started_tmp="$RUN_STARTED_MARKER.tmp.$$"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)" >"$run_started_tmp"
  mv -f "$run_started_tmp" "$RUN_STARTED_MARKER"
  if ! env ULLM_BUILD_GIT_COMMIT="$REQUIRED_COMMIT" ULLM_SERVED_MODEL_MANIFEST="$ACTIVE" \
      ULLM_HIP_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 \
      ULLM_REQUIRE_HIP_AQ4_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1 \
      ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL=1 \
      ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL=1 \
      ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL=1 \
      ULLM_REQUIRE_HIP_ADD_KERNEL=1 ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1 \
      ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL=1 \
      ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1 \
      ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL=1 \
      ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL=1 \
      ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL=1 \
      ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1 ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL=1 \
      ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL=1 \
      ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 ULLM_REQUIRE_HIP_ROPE_KERNEL=1 \
      ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL=1 ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL=1 \
      ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1 ULLM_REQUIRE_HIP_TOP1_KERNEL=1 \
      runuser -u homelab1 -- timeout --signal=TERM --kill-after=30s 1200s "$CANDIDATE_BIN" \
        "$PACKAGE" "$CASES" "$REPLAY" "$OUTPUT" 1 --enable-intermediate-trace \
        >"$RUN_LOG" 2>&1; then
    echo "candidate run failed" >&2
    return 78
  fi
  [[ -s "$MONITOR_LOG" ]]
  [[ ! -e "$OBSERVER_FAIL_MARKER" ]]
}
run_locked
"$ROCM_PYTHON" - "$OUTPUT/manifest.json" "$CANDIDATE_BIN" "$ACTIVE" "$PACKAGE/package/manifest.json" "$CASES" "$REPLAY" "$REQUIRED_COMMIT" <<'PY'
import hashlib, json, pathlib, sys

manifest_path, binary, active, package_manifest, cases, replay, required = sys.argv[1:]
m = json.loads(pathlib.Path(manifest_path).read_text())
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()
assert m['rows'] == 3
assert m['identity']['build_git_commit'] == required
assert m['identity']['tool_binary'] == binary
assert m['identity']['tool_binary_sha256'] == sha(binary)
assert m['identity']['active_manifest_sha256'] == sha(active)
assert m['identity']['package_manifest_sha256'] == sha(package_manifest)
assert m['input_binding']['cases_sha256'] == sha(cases)
assert m['input_binding']['replay_sha256'] == sha(replay)
PY
sha256sum -c "$OUTPUT/SHA256SUMS"
echo "output_manifest_verified=1"
echo "gate_finished=$(date --iso-8601=seconds)"
