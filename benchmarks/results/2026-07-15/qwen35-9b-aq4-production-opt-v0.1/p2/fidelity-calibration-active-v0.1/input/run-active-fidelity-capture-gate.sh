#!/usr/bin/env bash
set -Eeuo pipefail
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PATH

# This is a template only.  It is deliberately fail-closed until the source capture has been
# completed and the three artifact/build placeholders below have been replaced and reviewed.
REPO="/home/homelab1/coding-local/ultimateLLM/uLLM-project"
BASE="$REPO/benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-calibration-active-v0.1"
INPUT_DIR="$BASE/input"
SPLIT_ROOT="$REPO/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-holdout-split-v0.1"
PLAN="$INPUT_DIR/plan.json"
CASES="$REPO/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-calibration-capture-v0.1/input/cases.json"

OUTPUT="$BASE/output"
METRICS="$BASE/metrics.json"
GATE_LOG="$BASE/gate.log"
MONITOR_LOG="$BASE/monitor.log"
RUN_LOG="$BASE/run.log"
STOP_MARKER="$BASE/service-stopped.marker"
OBSERVER_FAIL_MARKER="$BASE/observer-failed.marker"
OBSERVER_SAMPLE_MARKER="$BASE/observer-sample.marker"
RUN_STARTED_MARKER="$BASE/run-started.marker"

# Do not change SOURCE_ARTIFACT or either source digest until the source exporter has finished.
SOURCE_ARTIFACT="$BASE/SOURCE_ARTIFACT_ROOT_PLACEHOLDER"
EXPECTED_SOURCE_ARTIFACT_SHA256="__SOURCE_ARTIFACT_SHA256__"
EXPECTED_SOURCE_MANIFEST_SHA256="__SOURCE_MANIFEST_SHA256__"
EXPECTED_CAPTURE_BINARY_SHA256="__CAPTURE_BINARY_SHA256__"

# Fixed contract pins.  The binary must be built from this committed fidelity producer baseline,
# rather than from a P3 candidate or from the production worker.
REQUIRED_BUILD_COMMIT="b1755da2a8ed188e3afac52dc1303ebaec3d09f5"
FIDELITY_BIN="$REPO/target/release/ullm-aq4-fidelity-capture"
ACTIVE="/etc/ullm/served-models/active.json"
PACKAGE_ROOT="/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1"
PACKAGE_MANIFEST="$PACKAGE_ROOT/package/manifest.json"
WORKER="$REPO/target/reasoning-v2/release/ullm-aq4-worker"
GUARD_SHA256="4eafd9bc149792b9c9849fed07a70830a42cf8227b85431130eec8f41708abc0"
DEVICE_ARCHITECTURE="gfx1201"
QUANTIZED_ARTIFACT_REVISION="aq4-reasoning-v0.1-candidate"
EXPECTED_SERVED_SHA256="feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44"
EXPECTED_PACKAGE_SHA256="a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad"
EXPECTED_WORKER_SHA256="177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d"
EXPECTED_PLAN_SHA256="03e921b050d64ae75206ff561b19ba563c1fac69ccf40dfab15176dee2b63854"
EXPECTED_CASES_SHA256="53f256bc8f5ed4036cfb1a9a98c0c9d9197bb980e1ef91d7ff01cf73001369a8"
EXPECTED_SPLIT_SHA256="966878f3d9eb13f5b485825208f8072521724f308f5ee3d8a003b0b051198887"
EXPECTED_POLICY_SHA256="302c3219af286a970ddf39ed090021ef102b51b2d188c0ff337f6b9dd04d1a03"
EXPECTED_CALIBRATION_SHA256="20c09f22bb1ca4dfac907de09febddb01ed0228c3f4a17c01efd646491e0983f"
MIN_FREE_BYTES=29078324
MAX_ROWS=24
MAX_ROW_BYTES=65536
MAX_HIDDEN_BYTES=393216
MAX_LOGITS_BYTES=23838720
MAX_VECTOR_BYTES=24231936
PYTHON="/usr/bin/python3.12"
SERVICE="ullm-openai.service"
LOCK="/run/ullm/r9700.lock"
RUNTIME_DIR="${LOCK%/*}"
ROCM_SMI="/opt/rocm/bin/rocm-smi"
ROCM_SMI_REAL_EXPECTED="/opt/rocm-7.2.1/libexec/rocm_smi/rocm_smi.py"
ROCM_SMI_SHA_EXPECTED="5a64729944c9bcd7eccd647ddc39cca1a942b3a51cf234fa1cb59a7b20591f46"
REQUIRED_GUARDS=(
  ULLM_REQUIRE_HIP_AQ4_KERNEL ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL
  ULLM_REQUIRE_HIP_ADD_KERNEL ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL ULLM_REQUIRE_HIP_BF16_ROW_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL
  ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL
  ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL ULLM_REQUIRE_HIP_RMSNORM_KERNEL
  ULLM_REQUIRE_HIP_ROPE_KERNEL ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL
  ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL ULLM_REQUIRE_HIP_SILU_MUL_KERNEL ULLM_REQUIRE_HIP_TOP1_KERNEL
)

PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
PREFLIGHT_LOCKED_ONLY="${PREFLIGHT_LOCKED_ONLY:-0}"
MOCK_PREFLIGHT="${MOCK_PREFLIGHT:-0}"
if [[ "$PREFLIGHT_ONLY" = 1 && "$PREFLIGHT_LOCKED_ONLY" = 1 ]]; then
  echo "preflight modes are mutually exclusive" >&2
  exit 64
fi
if [[ "$MOCK_PREFLIGHT" = 1 ]]; then
  # Tests may redirect only the read-only inputs.  Mock mode can never enter the stop/run path.
  REPO="${MOCK_REPO:-$REPO}"
  BASE="${MOCK_BASE:-$BASE}"
fi

fail() { echo "active fidelity gate: $*" >&2; exit 1; }
sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }
require_digest() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || fail "$name is an unresolved placeholder"
}
require_regular() {
  local path="$1" label="$2"
  [[ ! -L "$path" && -f "$path" ]] || fail "$label must be a non-symlink regular file: $path"
}
require_absent() {
  local path="$1"
  [[ ! -e "$path" && ! -L "$path" ]] || fail "refusing to overwrite existing path: $path"
}

validate_plan_and_cases() {
  require_regular "$PLAN" plan
  require_regular "$CASES" cases
  [[ "$(sha256_file "$PLAN")" = "$EXPECTED_PLAN_SHA256" ]] || fail "plan SHA differs"
  [[ "$(sha256_file "$CASES")" = "$EXPECTED_CASES_SHA256" ]] || fail "cases SHA differs"
  "$PYTHON" - "$PLAN" "$CASES" "$EXPECTED_SPLIT_SHA256" "$EXPECTED_POLICY_SHA256" "$EXPECTED_CALIBRATION_SHA256" "$EXPECTED_SERVED_SHA256" "$EXPECTED_PACKAGE_SHA256" "$EXPECTED_WORKER_SHA256" "$GUARD_SHA256" "$DEVICE_ARCHITECTURE" "$QUANTIZED_ARTIFACT_REVISION" <<'PY'
import hashlib, json, pathlib, sys
plan_path, cases_path, split_sha, policy_sha, calibration_sha, served_sha, package_sha, worker_sha, guard_sha, arch, quant_rev = sys.argv[1:]
plan = json.loads(pathlib.Path(plan_path).read_text(encoding="utf-8"))
assert plan["schema_version"] == "ullm.aq4_p2_fidelity_capture_plan.v1"
assert plan["status"] == "ready_for_source_and_active_capture"
assert plan["promotion_eligible"] is False and plan["row_count"] == 24 and plan["full_context_step"] == 0
assert plan["one_source_model_load"] is True and plan["one_active_model_load"] is True
assert plan["split_manifest_sha256"] == split_sha and plan["policy_sha256"] == policy_sha and plan["calibration_cases_sha256"] == calibration_sha
assert plan["source_cases_sha256"] == hashlib.sha256(pathlib.Path(cases_path).read_bytes()).hexdigest()
assert plan["execution_contract"]["source_torch_threads"] == 16
assert plan["expected_active_identity"] == {"served_model_manifest_sha256": served_sha, "package_manifest_sha256": package_sha, "worker_binary_sha256": worker_sha, "guard_sha256": guard_sha, "device_architecture": arch, "quantized_artifact_revision": quant_rev}
cases = json.loads(pathlib.Path(cases_path).read_text(encoding="utf-8"))
assert cases["schema_version"] == "ullm.qwen35_aq4_source_calibration_cases.v1" and len(cases["cases"]) == 24
assert len({row["case_id"] for row in cases["cases"]}) == 24
print("plan_cases=24 one_source_load=1 one_active_load=1 source_threads=16")
PY
}

validate_active_identity() {
  require_regular "$ACTIVE" active_manifest
  require_regular "$PACKAGE_MANIFEST" package_manifest
  require_regular "$WORKER" worker
  [[ "$(sha256_file "$ACTIVE")" = "$EXPECTED_SERVED_SHA256" ]] || fail "served manifest SHA differs"
  [[ "$(sha256_file "$PACKAGE_MANIFEST")" = "$EXPECTED_PACKAGE_SHA256" ]] || fail "package manifest SHA differs"
  [[ "$(sha256_file "$WORKER")" = "$EXPECTED_WORKER_SHA256" ]] || fail "worker SHA differs"
  "$PYTHON" - "$ACTIVE" "$PACKAGE_MANIFEST" "$WORKER" "$EXPECTED_SERVED_SHA256" "$EXPECTED_PACKAGE_SHA256" "$EXPECTED_WORKER_SHA256" "$GUARD_SHA256" "$DEVICE_ARCHITECTURE" "$QUANTIZED_ARTIFACT_REVISION" <<'PY'
import hashlib, json, pathlib, sys
active, package, worker, served_sha, package_sha, worker_sha, guard_sha, arch, quant_rev = sys.argv[1:]
m = json.loads(pathlib.Path(active).read_text(encoding="utf-8"))
assert m["schema_version"] == "ullm.served_model.v2"
assert m["format"] == {"format_id": "AQ4_0", "implementation_id": "qwen35_aq4_rdna4_v1"}
assert m["public"]["revision"] == quant_rev and m["worker"]["identity"]["device"] == arch
assert m["worker"]["binary_sha256"] == worker_sha and m["product"]["package"]["manifest_sha256"] == package_sha
required = m["worker"]["required_environment"]
digest = hashlib.sha256(b"ullm-aq4-p2-resident-guards-v1\\0")
for name in sorted(required): digest.update(f"{name}=1\\n".encode())
assert digest.hexdigest() == guard_sha and len(required) == 30
assert hashlib.sha256(pathlib.Path(active).read_bytes()).hexdigest() == served_sha
assert hashlib.sha256(pathlib.Path(package).read_bytes()).hexdigest() == package_sha
assert hashlib.sha256(pathlib.Path(worker).read_bytes()).hexdigest() == worker_sha
print("active_identity=served package worker guard device build_contract_ready")
PY
}

validate_source_identity() {
  require_digest EXPECTED_SOURCE_ARTIFACT_SHA256 "$EXPECTED_SOURCE_ARTIFACT_SHA256"
  require_digest EXPECTED_SOURCE_MANIFEST_SHA256 "$EXPECTED_SOURCE_MANIFEST_SHA256"
  require_digest EXPECTED_CAPTURE_BINARY_SHA256 "$EXPECTED_CAPTURE_BINARY_SHA256"
  [[ "$SOURCE_ARTIFACT" != *PLACEHOLDER* ]] || fail "SOURCE_ARTIFACT path is an unresolved placeholder"
  [[ "$SOURCE_ARTIFACT" != "$BASE/SOURCE_ARTIFACT_ROOT_PLACEHOLDER" ]] || fail "SOURCE_ARTIFACT path is an unresolved placeholder"
  [[ -d "$SOURCE_ARTIFACT" && ! -L "$SOURCE_ARTIFACT" ]] || fail "source artifact root is not a real directory"
  require_regular "$SOURCE_ARTIFACT/manifest.json" source_manifest
  require_regular "$SOURCE_ARTIFACT/SHA256SUMS" source_sums
  [[ "$(sha256_file "$SOURCE_ARTIFACT/SHA256SUMS")" = "$EXPECTED_SOURCE_ARTIFACT_SHA256" ]] || fail "source artifact SHA differs"
  [[ "$(sha256_file "$SOURCE_ARTIFACT/manifest.json")" = "$EXPECTED_SOURCE_MANIFEST_SHA256" ]] || fail "source manifest SHA differs"
  (cd "$SOURCE_ARTIFACT" && sha256sum -c SHA256SUMS >/dev/null) || fail "source SHA256SUMS verification failed"
}

validate_binary_and_bounds() {
  require_regular "$FIDELITY_BIN" fidelity_binary
  [[ "$(sha256_file "$FIDELITY_BIN")" = "$EXPECTED_CAPTURE_BINARY_SHA256" ]] || fail "fidelity binary SHA differs"
  strings -a "$FIDELITY_BIN" | grep -Fq "$REQUIRED_BUILD_COMMIT" || fail "fidelity binary build commit differs"
  [[ "$PACKAGE_ROOT/package" != "$PACKAGE_ROOT" ]] || fail "package argument is not the package directory"
  require_regular "$PACKAGE_MANIFEST" package_manifest
  require_absent "$OUTPUT"; require_absent "$METRICS"; require_absent "$GATE_LOG"; require_absent "$MONITOR_LOG"; require_absent "$RUN_LOG"
  require_absent "$STOP_MARKER"; require_absent "$OBSERVER_FAIL_MARKER"; require_absent "$OBSERVER_SAMPLE_MARKER"; require_absent "$RUN_STARTED_MARKER"
  [[ "$MAX_ROWS" = 24 && "$MAX_ROW_BYTES" = 65536 && "$MAX_HIDDEN_BYTES" = 393216 && "$MAX_LOGITS_BYTES" = 23838720 ]] || fail "row/sidecar bounds changed"
  local rocm_real
  rocm_real="$(realpath "$ROCM_SMI")"
  [[ "$rocm_real" = "$ROCM_SMI_REAL_EXPECTED" ]] || fail "ROCm SMI realpath differs"
  require_regular "$rocm_real" rocm_smi
  [[ "$(stat -Lc '%F:%h' "$rocm_real")" = "regular file:1" ]] || fail "ROCm SMI identity differs"
  [[ "$(sha256_file "$rocm_real")" = "$ROCM_SMI_SHA_EXPECTED" ]] || fail "ROCm SMI SHA differs"
  local available
  available="$(df -P -B1 "$BASE" | awk 'NR==2 {print $4}')"
  [[ "$available" =~ ^[0-9]+$ && "$available" -ge "$MIN_FREE_BYTES" ]] || fail "insufficient output disk free space"
}

validate_service_identity() {
  local uid gid owner mainpid owner_command
  uid="$(id -u homelab1)"; gid="$(id -g homelab1)"
  [[ ! -L "$RUNTIME_DIR" && "$(stat -Lc '%F:%a:%u:%g:%h' "$RUNTIME_DIR")" = "directory:750:$uid:$gid:2" ]] || fail "RuntimeDirectory identity differs"
  [[ ! -L "$LOCK" && "$(stat -Lc '%F:%a:%u:%g:%h:%s' "$LOCK")" = "regular empty file:600:$uid:$gid:1:0" ]] || fail "runtime lock identity differs"
  [[ "$(systemctl is-active "$SERVICE")" = active ]] || fail "service is not active"
  mainpid="$(systemctl show "$SERVICE" -p MainPID --value)"
  owner="$(lslocks -o PID,PATH 2>/dev/null | awk -v path="$LOCK" '$2 == path {print $1; exit}')"
  owner_command="$(ps -p "$owner" -o comm= 2>/dev/null || true)"
  [[ "$owner" = "$mainpid" && "$owner" =~ ^[1-9][0-9]*$ && "$owner_command" == ullm-openai-gat* ]] || fail "runtime lock owner differs from service MainPID"
}

validate_preflight() {
  [[ -x "$PYTHON" ]] || fail "fixed Python runtime is unavailable"
  require_digest EXPECTED_SOURCE_ARTIFACT_SHA256 "$EXPECTED_SOURCE_ARTIFACT_SHA256"
  require_digest EXPECTED_SOURCE_MANIFEST_SHA256 "$EXPECTED_SOURCE_MANIFEST_SHA256"
  require_digest EXPECTED_CAPTURE_BINARY_SHA256 "$EXPECTED_CAPTURE_BINARY_SHA256"
  [[ "$SOURCE_ARTIFACT" != *PLACEHOLDER* ]] || fail "SOURCE_ARTIFACT path is an unresolved placeholder"
  validate_plan_and_cases
  validate_active_identity
  validate_source_identity
  validate_binary_and_bounds
  if [[ "$MOCK_PREFLIGHT" != 1 ]]; then
    validate_service_identity
  fi
}

if [[ "$MOCK_PREFLIGHT" = 1 ]]; then
  [[ "$PREFLIGHT_ONLY" = 1 || "$PREFLIGHT_LOCKED_ONLY" = 1 ]] || fail "mock mode is preflight-only"
fi

validate_preflight
echo "gate_template=active-fidelity-v0.1"
echo "rows=$MAX_ROWS output=$OUTPUT metrics=$METRICS"
echo "plan_sha256=$(sha256_file "$PLAN") cases_sha256=$(sha256_file "$CASES")"
echo "split_sha256=$EXPECTED_SPLIT_SHA256 policy_sha256=$EXPECTED_POLICY_SHA256 calibration_sha256=$EXPECTED_CALIBRATION_SHA256"
echo "served_sha256=$EXPECTED_SERVED_SHA256 package_sha256=$EXPECTED_PACKAGE_SHA256 worker_sha256=$EXPECTED_WORKER_SHA256 guard_sha256=$GUARD_SHA256 device_architecture=$DEVICE_ARCHITECTURE"
if [[ "$MOCK_PREFLIGHT" = 1 ]]; then
  echo "mock_preflight=1 service_stop=0 gpu_run=0"
  exit 0
fi
if [[ "$PREFLIGHT_ONLY" = 1 ]]; then
  systemctl is-active --quiet "$SERVICE" || fail "service is not active"
  echo "preflight_only=1 service_stop=0 gpu_run=0"
  exit 0
fi

# Locked preflight is read-only and never arms a restore trap or starts/stops the service.
if [[ "$PREFLIGHT_LOCKED_ONLY" = 1 ]]; then
  systemctl is-active --quiet "$SERVICE" || fail "service is not active"
  [[ -f "$LOCK" && ! -L "$LOCK" ]] || fail "runtime lock is missing"
  owner="$(lslocks -o PID,PATH 2>/dev/null | awk -v path="$LOCK" '$2 == path {print $1; exit}')"
  mainpid="$(systemctl show "$SERVICE" -p MainPID --value)"
  [[ "$owner" = "$mainpid" && "$owner" =~ ^[1-9][0-9]*$ ]] || fail "runtime lock owner differs from service MainPID"
  echo "locked_preflight_systemctl_mutations=0 lock_owner=$owner observer=not_started"
  exit 0
fi

[[ "$PREFLIGHT_ONLY" = 0 && "$PREFLIGHT_LOCKED_ONLY" = 0 ]] || fail "unknown preflight mode"
[[ "$MOCK_PREFLIGHT" = 0 ]] || fail "mock mode cannot enter the service stop path"

NRESTARTS_BEFORE="$(systemctl show "$SERVICE" -p NRestarts --value)"
SERVICE_MAINPID="$(systemctl show "$SERVICE" -p MainPID --value)"
[[ "$(systemctl is-active "$SERVICE")" = active ]] || fail "service is not active"
RESTORE_NEEDED=0
LOCK_FD=""
OBSERVER_PID=""
RESTORE_RC=0
RUNTIME_DIR_CREATED=0
LOCK_CREATED=0
RUNTIME_DIR_CREATED_DEV_INO=""
LOCK_CREATED_DEV_INO=""

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  [[ -n "$OBSERVER_PID" ]] && kill "$OBSERVER_PID" 2>/dev/null
  [[ -n "$OBSERVER_PID" ]] && wait "$OBSERVER_PID" 2>/dev/null
  [[ -n "$LOCK_FD" ]] && eval "exec $LOCK_FD>&-"
  if (( LOCK_CREATED )) && [[ -f "$LOCK" && ! -L "$LOCK" && -z "$(lslocks -o PID,PATH 2>/dev/null | awk -v path="$LOCK" '$2 == path {print $1; exit}')" ]]; then rm -f "$LOCK"; fi
  if (( RUNTIME_DIR_CREATED )) && [[ -d "$RUNTIME_DIR" && -z "$(find "$RUNTIME_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then rmdir "$RUNTIME_DIR"; fi
  if (( RESTORE_NEEDED )); then
    timeout --signal=TERM --kill-after=5s 60s systemctl start "$SERVICE" || RESTORE_RC=1
    deadline=$((SECONDS + 60))
    while (( SECONDS < deadline )); do
      [[ "$(systemctl is-active "$SERVICE" 2>/dev/null)" = active && "$(systemctl show "$SERVICE" -p SubState --value 2>/dev/null)" = running ]] && break
      sleep 1
    done
    [[ "$(systemctl is-active "$SERVICE" 2>/dev/null)" = active ]] || RESTORE_RC=1
    [[ "$(systemctl show "$SERVICE" -p NRestarts --value 2>/dev/null)" = "$NRESTARTS_BEFORE" ]] || RESTORE_RC=1
    [[ "$(sha256_file "$ACTIVE" 2>/dev/null)" = "$EXPECTED_SERVED_SHA256" ]] || RESTORE_RC=1
    [[ "$(sha256_file "$PACKAGE_MANIFEST" 2>/dev/null)" = "$EXPECTED_PACKAGE_SHA256" ]] || RESTORE_RC=1
    [[ "$(sha256_file "$WORKER" 2>/dev/null)" = "$EXPECTED_WORKER_SHA256" ]] || RESTORE_RC=1
    restore_uid="$(id -u homelab1)"; restore_gid="$(id -g homelab1)"
    [[ ! -L "$RUNTIME_DIR" && "$(stat -Lc '%F:%a:%u:%g:%h' "$RUNTIME_DIR" 2>/dev/null || true)" = "directory:750:$restore_uid:$restore_gid:2" ]] || RESTORE_RC=1
    [[ ! -L "$LOCK" && "$(stat -Lc '%F:%a:%u:%g:%h:%s' "$LOCK" 2>/dev/null || true)" = "regular empty file:600:$restore_uid:$restore_gid:1:0" ]] || RESTORE_RC=1
    restore_mainpid="$(systemctl show "$SERVICE" -p MainPID --value 2>/dev/null || true)"
    restore_owner="$(lslocks -o PID,PATH 2>/dev/null | awk -v path="$LOCK" '$2 == path {print $1; exit}')"
    [[ "$restore_owner" = "$restore_mainpid" ]] || RESTORE_RC=1
  fi
  [[ -e "$RUN_STARTED_MARKER" ]] || rm -f "$GATE_LOG" "$MONITOR_LOG" "$RUN_LOG" "$STOP_MARKER" "$OBSERVER_FAIL_MARKER" "$OBSERVER_SAMPLE_MARKER"
  (( RESTORE_RC == 0 )) || rc=90
  exit "$rc"
}
trap cleanup EXIT INT TERM

RESTORE_NEEDED=1
printf 'armed_at=%s\n' "$(date --iso-8601=seconds)" >"$STOP_MARKER"
(
  while :; do
    "$ROCM_SMI" --showmeminfo vram --showuse --showpower --json >"$BASE/.observer.sample.$$" 2>&1 || { : >"$OBSERVER_FAIL_MARKER"; exit 1; }
    mv -f "$BASE/.observer.sample.$$" "$MONITOR_LOG"
    date +%s%N >"$OBSERVER_SAMPLE_MARKER"
    sleep 1
  done
) &
OBSERVER_PID=$!
for _ in $(seq 1 20); do [[ -s "$OBSERVER_SAMPLE_MARKER" ]] && break; sleep 0.1; done
[[ -s "$OBSERVER_SAMPLE_MARKER" && ! -e "$OBSERVER_FAIL_MARKER" ]] || fail "resource observer did not become ready"
observer_sample_ts="$(cat "$OBSERVER_SAMPLE_MARKER")"
observer_now_ts="$(date +%s%N)"
[[ "$observer_sample_ts" =~ ^[0-9]+$ && "$observer_now_ts" -ge "$observer_sample_ts" && $((observer_now_ts - observer_sample_ts)) -le 5000000000 ]] || fail "resource observer sample is stale"
printf 'service_stop_attempt_at=%s\n' "$(date --iso-8601=seconds)" >>"$STOP_MARKER"
systemctl stop "$SERVICE" || fail "service stop failed"
printf 'service_stop_returned_at=%s\n' "$(date --iso-8601=seconds)" >>"$STOP_MARKER"
for _ in $(seq 1 30); do [[ "$(systemctl is-active "$SERVICE" || true)" = inactive ]] && break; sleep 1; done
[[ "$(systemctl is-active "$SERVICE" || true)" = inactive ]] || fail "service did not stop"
[[ ! -e "$RUNTIME_DIR" && ! -L "$RUNTIME_DIR" ]] || fail "RuntimeDirectory remained after service stop"
(umask 077; mkdir "$RUNTIME_DIR") || fail "runtime directory create failed"
RUNTIME_DIR_CREATED=1
chown homelab1:homelab1 "$RUNTIME_DIR"; chmod 750 "$RUNTIME_DIR"
(umask 077; set -C; : >"$LOCK") || fail "runtime lock create failed"
LOCK_CREATED=1
chown homelab1:homelab1 "$LOCK"; chmod 600 "$LOCK"
exec {LOCK_FD}>"$LOCK"
flock -n "$LOCK_FD" || fail "runtime lock acquisition failed"
printf 'runtime_lock_acquired_at=%s\n' "$(date --iso-8601=seconds)" >>"$STOP_MARKER"
touch "$RUN_STARTED_MARKER"
guard_env=("ULLM_BUILD_GIT_COMMIT=$REQUIRED_BUILD_COMMIT" "ULLM_SERVED_MODEL_MANIFEST=$ACTIVE" "ULLM_HIP_VISIBLE_DEVICES=1" "HIP_VISIBLE_DEVICES=1")
for guard_name in "${REQUIRED_GUARDS[@]}"; do guard_env+=("$guard_name=1"); done
env "${guard_env[@]}" \
  runuser -u homelab1 -- timeout --signal=TERM --kill-after=30s 1200s "$FIDELITY_BIN" \
    --served-model-manifest "$ACTIVE" --split-root "$SPLIT_ROOT" --source "$SOURCE_ARTIFACT" \
    --cases "$CASES" --output "$OUTPUT" --device-index 0 --chunk-elements 65536 \
    --expected-split-manifest-sha256 "$EXPECTED_SPLIT_SHA256" --expected-policy-sha256 "$EXPECTED_POLICY_SHA256" \
    --expected-calibration-cases-sha256 "$EXPECTED_CALIBRATION_SHA256" \
    --expected-served-model-manifest-sha256 "$EXPECTED_SERVED_SHA256" \
    --expected-package-manifest-sha256 "$EXPECTED_PACKAGE_SHA256" \
    --expected-worker-binary-sha256 "$EXPECTED_WORKER_SHA256" --expected-guard-sha256 "$GUARD_SHA256" \
    --expected-device-architecture "$DEVICE_ARCHITECTURE" --expected-quantized-artifact-revision "$QUANTIZED_ARTIFACT_REVISION" \
    >"$RUN_LOG" 2>&1 || fail "active fidelity capture failed"
[[ -s "$OUTPUT/SHA256SUMS" ]] || fail "active output SHA256SUMS is missing"
(cd "$OUTPUT" && sha256sum -c SHA256SUMS) || fail "active output SHA256SUMS failed"
"$PYTHON" tools/capture-qwen35-aq4-fidelity.py --split-root "$SPLIT_ROOT" --source "$SOURCE_ARTIFACT" --active "$OUTPUT" --output "$METRICS" \
  --expected-split-manifest-sha256 "$EXPECTED_SPLIT_SHA256" --expected-policy-sha256 "$EXPECTED_POLICY_SHA256" \
  --expected-calibration-cases-sha256 "$EXPECTED_CALIBRATION_SHA256" --expected-served-model-manifest-sha256 "$EXPECTED_SERVED_SHA256" \
  --expected-package-manifest-sha256 "$EXPECTED_PACKAGE_SHA256" --expected-worker-binary-sha256 "$EXPECTED_WORKER_SHA256" \
  --expected-guard-sha256 "$GUARD_SHA256" --expected-device-architecture "$DEVICE_ARCHITECTURE" \
  --expected-quantized-artifact-revision "$QUANTIZED_ARTIFACT_REVISION" >"$BASE/metrics.stdout" 2>"$BASE/metrics.stderr"
"$PYTHON" tools/validate-qwen35-aq4-fidelity-capture.py --metrics "$METRICS" --split-root "$SPLIT_ROOT" >"$BASE/metrics.validate.stdout" 2>"$BASE/metrics.validate.stderr"
echo "active_output_verified=1 metrics_verified=1 rows=$MAX_ROWS one_model_load=1 nonfinite_rows=0"
echo "gate_finished=$(date --iso-8601=seconds)"
