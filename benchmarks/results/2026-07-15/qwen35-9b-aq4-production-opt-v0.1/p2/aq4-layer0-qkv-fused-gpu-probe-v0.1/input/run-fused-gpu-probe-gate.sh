#!/usr/bin/env bash
set -Eeuo pipefail
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PATH

# Diagnostic-only gate for the fused layer-0 AQ4 QKV/Z/gate/beta probe. The default path is
# read-only preflight. The service/GPU path requires EXECUTE_GPU_PROBE=1 and never produces
# a numeric Go/No-Go or promotion decision.
REPO="${REPO:-/home/homelab1/coding-local/ultimateLLM/uLLM-project}"
BASE="${BASE:-$REPO/benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-qkv-fused-gpu-probe-v0.1}"
INPUT="${INPUT:-$REPO/benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-matvec-oracle-integration-v0.1/runtime-input.jsonl}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package}"
PACKAGE_MANIFEST="${PACKAGE_MANIFEST:-$PACKAGE_ROOT/manifest.json}"
ACTIVE="${ACTIVE:-/etc/ullm/served-models/active.json}"
PROFILE="${PROFILE:-$REPO/deploy/served-models/qwen35-9b-aq4-reasoning.profile.json}"
WORKER="${WORKER:-$REPO/target/reasoning-v2/release/ullm-aq4-worker}"

PROBE_ROOT="${PROBE_ROOT:-$BASE/input/probe-binary-v0.1}"
PROBE="${PROBE:-$PROBE_ROOT/ullm-aq4-layer0-qkv-z-gate-beta-runtime-probe}"
RECEIPT="${RECEIPT:-$PROBE_ROOT/build-receipt.json}"
SUMS="${SUMS:-$PROBE_ROOT/SHA256SUMS}"
OUTPUT="${OUTPUT:-$BASE/attempts/attempt1/output}"
RUN_LOG="${RUN_LOG:-$BASE/attempts/attempt1/run.log}"
MONITOR_LOG="${MONITOR_LOG:-$BASE/attempts/attempt1/monitor.log}"
STOP_MARKER="${STOP_MARKER:-$BASE/attempts/attempt1/service-stopped.marker}"
OBSERVER_FAIL_MARKER="${OBSERVER_FAIL_MARKER:-$BASE/attempts/attempt1/observer-failed.marker}"
OBSERVER_SAMPLE_MARKER="${OBSERVER_SAMPLE_MARKER:-$BASE/attempts/attempt1/observer-sample.marker}"

EXPECTED_INPUT_SHA256="c009a9bded30b1b9a7c704c622bd3106b3d17989c438f91eb20bb16817348e17"
EXPECTED_PACKAGE_SHA256="a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad"
EXPECTED_ACTIVE_SHA256="feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44"
EXPECTED_PROFILE_SHA256="1013cc803adce27a178856e9fe300fc5a26cc998b5381812b0c3bd17ebbb5937"
EXPECTED_WORKER_SHA256="177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d"
EXPECTED_BUILD_COMMIT="6082df4966190ae4977b699460a5ecb93fee8e34"
EXPECTED_BUILD_COMMAND="CARGO_BUILD_JOBS=1 cargo build --release -p ullm-engine --bin ullm-aq4-layer0-qkv-z-gate-beta-runtime-probe"
EXPECTED_PROBE_BINARY_SHA256="42752e7a29614f59f72f90bed6797c3e925b032bffb1a4196c462c8476386840"
EXPECTED_BUILD_RECEIPT_SHA256="90e9ef6d383f7ef25e9526659f035e40291ba1a5efa7f8ba36340c8b245d9504"
EXPECTED_DEVICE_ARCHITECTURE="gfx1201"
EXPECTED_PHYSICAL_CARD="2"
EXPECTED_HIP_VISIBLE_TOKEN="1"
EXPECTED_FILTERED_HIP_ORDINAL="0"
EXPECTED_CPU_GLOBAL_DEVICE_INDEX="0"
EXPECTED_LOGICAL_DEVICE_INDEX="1"
SERVICE="ullm-openai.service"
SYSTEMCTL=(sudo -n -- systemctl)
RUNTIME_DIR_INSTALL=(sudo -n -- install -d -o homelab1 -g homelab1 -m 0750)
RUNTIME_DIR_REMOVE=(sudo -n -- rmdir)
LOCK="/run/ullm/r9700.lock"
RUNTIME_DIR="${LOCK%/*}"
ROCM_SMI="/opt/rocm/bin/amd-smi"
SUDO=(sudo -n --)

PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
PREFLIGHT_LOCKED_ONLY="${PREFLIGHT_LOCKED_ONLY:-0}"
MOCK_PREFLIGHT="${MOCK_PREFLIGHT:-0}"
EXECUTE_GPU_PROBE="${EXECUTE_GPU_PROBE:-0}"
if [[ "$PREFLIGHT_ONLY" = 1 && "$PREFLIGHT_LOCKED_ONLY" = 1 ]]; then
  echo "preflight modes are mutually exclusive" >&2
  exit 64
fi
if [[ "$MOCK_PREFLIGHT" = 1 ]]; then
  # Mock mode is strictly read-only and cannot reach systemctl, rocm-smi, or the probe.
  REPO="${MOCK_REPO:-$REPO}"
  BASE="${MOCK_BASE:-$BASE}"
  INPUT="${MOCK_INPUT:-$INPUT}"
  PACKAGE_MANIFEST="${MOCK_PACKAGE_MANIFEST:-$PACKAGE_MANIFEST}"
  ACTIVE="${MOCK_ACTIVE:-$ACTIVE}"
  PROFILE="${MOCK_PROFILE:-$PROFILE}"
  WORKER="${MOCK_WORKER:-$WORKER}"
  PROBE_ROOT="${MOCK_PROBE_ROOT:-$PROBE_ROOT}"
  PROBE="${MOCK_PROBE:-$PROBE}"
  RECEIPT="${MOCK_RECEIPT:-$RECEIPT}"
  SUMS="${MOCK_SUMS:-$SUMS}"
  OUTPUT="${MOCK_OUTPUT:-$OUTPUT}"
  RUN_LOG="${MOCK_RUN_LOG:-$BASE/run.log}"
  MONITOR_LOG="${MOCK_MONITOR_LOG:-$BASE/monitor.log}"
  STOP_MARKER="${MOCK_STOP_MARKER:-$BASE/service-stopped.marker}"
  OBSERVER_FAIL_MARKER="${MOCK_OBSERVER_FAIL_MARKER:-$BASE/observer-failed.marker}"
  OBSERVER_SAMPLE_MARKER="${MOCK_OBSERVER_SAMPLE_MARKER:-$BASE/observer-sample.marker}"
  EXECUTE_GPU_PROBE=0
fi

fail() { echo "AQ4 GPU probe gate: $*" >&2; exit 1; }
sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }
require_regular() {
  local path="$1" label="$2"
  [[ -f "$path" && ! -L "$path" ]] || fail "$label must be a non-symlink regular file: $path"
  [[ "$(stat -Lc '%F:%h' "$path")" = "regular file:1" ]] || fail "$label must have nlink=1: $path"
}
require_digest() { [[ "$2" =~ ^[0-9a-f]{64}$ ]] || fail "$1 is not a fixed SHA-256 pin"; }
require_absent() { [[ ! -e "$1" && ! -L "$1" ]] || fail "refusing to overwrite existing path: $1"; }

validate_identity() {
  require_digest EXPECTED_INPUT_SHA256 "$EXPECTED_INPUT_SHA256"
  require_digest EXPECTED_PACKAGE_SHA256 "$EXPECTED_PACKAGE_SHA256"
  require_digest EXPECTED_ACTIVE_SHA256 "$EXPECTED_ACTIVE_SHA256"
  require_digest EXPECTED_PROFILE_SHA256 "$EXPECTED_PROFILE_SHA256"
  require_digest EXPECTED_WORKER_SHA256 "$EXPECTED_WORKER_SHA256"
  require_digest EXPECTED_PROBE_BINARY_SHA256 "$EXPECTED_PROBE_BINARY_SHA256"
  require_digest EXPECTED_BUILD_RECEIPT_SHA256 "$EXPECTED_BUILD_RECEIPT_SHA256"
  require_regular "$INPUT" input_sidecar
  require_regular "$PACKAGE_MANIFEST" package_manifest
  require_regular "$ACTIVE" active_manifest
  require_regular "$PROFILE" reasoning_profile
  require_regular "$WORKER" worker_binary
  [[ "$(sha256_file "$INPUT")" = "$EXPECTED_INPUT_SHA256" ]] || fail "input sidecar SHA differs"
  [[ "$(sha256_file "$PACKAGE_MANIFEST")" = "$EXPECTED_PACKAGE_SHA256" ]] || fail "package manifest SHA differs"
  [[ "$(sha256_file "$ACTIVE")" = "$EXPECTED_ACTIVE_SHA256" ]] || fail "active manifest SHA differs"
  [[ "$(sha256_file "$PROFILE")" = "$EXPECTED_PROFILE_SHA256" ]] || fail "reasoning profile SHA differs"
  [[ "$(sha256_file "$WORKER")" = "$EXPECTED_WORKER_SHA256" ]] || fail "worker SHA differs"
  require_regular "$PROBE" probe_binary
  require_regular "$RECEIPT" build_receipt
  require_regular "$SUMS" build_sums
  [[ "$(sha256_file "$PROBE")" = "$EXPECTED_PROBE_BINARY_SHA256" ]] || fail "probe binary SHA differs"
  [[ "$(sha256_file "$RECEIPT")" = "$EXPECTED_BUILD_RECEIPT_SHA256" ]] || fail "probe build receipt SHA differs"
  (cd "$PROBE_ROOT" && sha256sum -c SHA256SUMS >/dev/null) || fail "probe SHA256SUMS verification failed"
  "$PYTHON" - "$RECEIPT" "$PROBE" "$EXPECTED_BUILD_COMMIT" "$EXPECTED_BUILD_COMMAND" <<'PY'
import hashlib, json, pathlib, sys
receipt_path, binary_path, expected_commit, expected_command = sys.argv[1:]
r = json.loads(pathlib.Path(receipt_path).read_text(encoding="utf-8"))
assert r["schema_version"] == "ullm.aq4_layer0_qkv_z_gate_beta_runtime_probe_build_receipt.v1"
assert r["status"] == "ready"
assert r["source"]["commit"] == expected_commit and r["source"]["tree_clean"] is True
assert r["build"]["command"] == expected_command and r["build"]["jobs"] == 1 and r["build"]["exit_status"] == 0
b = r["binary"]
p = pathlib.Path(binary_path)
assert b["path"] == p.name and b["nlink"] == 1 and b["mode"] == 0o555 and b["bytes"] == p.stat().st_size
assert b["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
print("fused_probe_build=clean_commit receipt command jobs1 immutable_mode0555 binary_sha nlink1")
PY
}

validate_output_contract() {
  "$PYTHON" - "$OUTPUT" "$INPUT" "$EXPECTED_INPUT_SHA256" "$EXPECTED_PACKAGE_SHA256" <<'PY'
import hashlib, json, math, pathlib, struct, sys
output, input_path, input_sha, package_sha = sys.argv[1:]
out = pathlib.Path(output)
r = json.loads((out / "report.json").read_text(encoding="utf-8"))
assert r["schema_version"] == "ullm.aq4_layer0_qkv_z_gate_beta_runtime_probe.v2"
assert r["status"] == "valid" and r["classification"] == "unclassified"
assert r["promotion_eligible"] is False and r["fused"] is True
assert r["operation"] == "aq4_matvec_qkv_z_gate_beta_f32"
d = r["device"]
assert d["backend"].lower() == "hip" and str(d["device_index"]) == "1"
assert str(d["device_id"]) == "0" and d["gcn_arch_name"] == "gfx1201"
assert r["visibility"] == {"hip_visible_devices": "1", "ullm_hip_visible_devices": "1"}
g = r["guard"]
assert g["hip_aq4_matvec_kernel_required"] is True
assert g["fused_kernel_required"] is True and g["fallback_allowed"] is False
assert g["fused_rpb_effective"] == 4
env = g["relevant_environment"]
assert env["ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL"] == "1"
assert env["ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL"] == "1"
assert env["ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB"] == "4"
assert env["ULLM_AQ4_MATVEC_RPB"] is None
assert r["input"]["sidecar_sha256"] == input_sha
assert r["input"]["identity"]["pre_stat"] == r["input"]["identity"]["post_stat"]
assert r["package"]["manifest_sha256"] == package_sha
expected = {"qkv": "qkv.f32le", "qkv_standalone": "qkv-standalone.f32le", "z": "z.f32le", "gate": "gate.f32le", "beta": "beta.f32le"}
assert set(r["outputs"]) == set(expected)
for key, name in expected.items():
    path = out / name
    assert path.is_file() and not path.is_symlink()
    raw = path.read_bytes()
    assert r["outputs"][key]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert r["outputs"][key]["bytes"] == len(raw) and len(raw) % 4 == 0
    for (value,) in struct.iter_unpack("<f", raw):
        assert math.isfinite(value)
assert (out / "report.json").stat().st_size > 0
print("fused_report=valid schema2 hip device1 gfx1201 fused_rpb4 standalone_rpb=unset/default32 finite_outputs=5 promotion=false")
PY
}
validate_preflight() {
  PYTHON="${PYTHON:-/usr/bin/python3}"
  [[ -x "$PYTHON" ]] || fail "fixed Python runtime is unavailable"
  [[ "$EXPECTED_PHYSICAL_CARD" = 2 && "$EXPECTED_HIP_VISIBLE_TOKEN" = 1 && "$EXPECTED_FILTERED_HIP_ORDINAL" = 0 && "$EXPECTED_CPU_GLOBAL_DEVICE_INDEX" = 0 && "$EXPECTED_LOGICAL_DEVICE_INDEX" = 1 && "$EXPECTED_DEVICE_ARCHITECTURE" = gfx1201 ]] || fail "device mapping pins changed"
  validate_identity
  require_absent "$BASE/attempts/attempt1"
  require_absent "$OUTPUT"
  require_absent "$RUN_LOG"
  require_absent "$MONITOR_LOG"
  require_absent "$STOP_MARKER"
  require_absent "$OBSERVER_FAIL_MARKER"
  require_absent "$OBSERVER_SAMPLE_MARKER"
}

run_observer() {
  : > "$MONITOR_LOG"
  while :; do
    if ! "$ROCM_SMI" --showmeminfo vram --showuse --showpower --json >>"$MONITOR_LOG" 2>&1; then
      : > "$OBSERVER_FAIL_MARKER"
      return 1
    fi
    : > "$OBSERVER_SAMPLE_MARKER"
    sleep 2
  done
}

service_active_hashes() {
  printf '%s %s %s\n' "$(sha256_file "$ACTIVE")" "$(sha256_file "$PACKAGE_MANIFEST")" "$(sha256_file "$WORKER")"
}

worker_pids() { /usr/bin/pgrep -f -x "$WORKER.*" 2>/dev/null || true; }
kfd_owners() {
  "$PYTHON" - "$REPO/tools/launch-aq4-p2-resident-smoke.py" <<'PY'
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("aq4_launcher", sys.argv[1])
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
print(json.dumps(mod._kfd_owner_snapshot()["owners"], separators=(",", ":")))
PY
}
amd_owners() {
  local raw
  raw=$("$ROCM_SMI" process --gpu "$EXPECTED_PHYSICAL_CARD" --general --json)
  "$PYTHON" - "$REPO/tools/launch-aq4-p2-resident-smoke.py" "$raw" <<'PY'
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("aq4_launcher", sys.argv[1])
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
print(json.dumps(mod.parse_amd_process_owners(sys.argv[2].encode())["owners"], separators=(",", ":")))
PY
}
stopped_services() {
  [[ "$("${SYSTEMCTL[@]}" is-active "$SERVICE" 2>/dev/null)" = inactive ]] &&
  [[ "$("${SYSTEMCTL[@]}" is-active "llama-qwen35-udq4.service" 2>/dev/null)" = inactive ]]
}
stable2_stopped() {
  local stable=0 workers amd kfd
  for _ in $(seq 1 120); do
    workers="$(worker_pids)"; amd="$(amd_owners)"; kfd="$(kfd_owners)"
    if stopped_services && [[ -z "$workers" && "$amd" = "[]" && "$kfd" = "[]" ]]; then
      stable=$((stable + 1)); [[ "$stable" -ge 2 ]] && return 0
    else
      stable=0
    fi
    sleep 1
  done
  return 1
}
LOCK_CREATED=0
STOPPED=0
LOCK_CLEAN=0
OLD_PID=""
OLD_RESTARTS=""
OLD_HASHES=""
cleanup_lock_substrate() {
  exec 9>&- 2>/dev/null || true
  [[ "$LOCK_CREATED" = 1 ]] || { LOCK_CLEAN=1; return 0; }
  [[ -f "$LOCK" && ! -L "$LOCK" ]] || return 1
  local current
  current="$(stat -Lc '%d:%i:%h:%a:%u:%g' "$LOCK")"
  [[ "$current" = "$LOCK_IDENTITY" ]] || return 1
  rm -f -- "$LOCK" || return 1
  "${RUNTIME_DIR_REMOVE[@]}" "$RUNTIME_DIR" || return 1
  [[ ! -e "$LOCK" && ! -e "$RUNTIME_DIR" ]] || return 1
  LOCK_CLEAN=1
}
post_start_check() {
  local new_pid new_restarts
  for _ in $(seq 1 120); do
    [[ "$("${SYSTEMCTL[@]}" is-active "$SERVICE" 2>/dev/null)" = active ]] &&
      [[ "$("${SYSTEMCTL[@]}" show "$SERVICE" -p SubState --value 2>/dev/null)" = running ]] && break
    sleep 1
  done
  [[ "$("${SYSTEMCTL[@]}" is-active "$SERVICE")" = active ]] || return 1
  [[ "$("${SYSTEMCTL[@]}" show "$SERVICE" -p SubState --value)" = running ]] || return 1
  new_pid="$("${SYSTEMCTL[@]}" show "$SERVICE" -p MainPID --value)"
  new_restarts="$("${SYSTEMCTL[@]}" show "$SERVICE" -p NRestarts --value)"
  [[ "$new_pid" =~ ^[0-9]+$ && "$new_pid" -gt 0 && "$new_pid" != "$OLD_PID" ]] || return 1
  [[ "$new_restarts" = "$OLD_RESTARTS" ]] || return 1
  [[ "$(service_active_hashes)" = "$OLD_HASHES" ]] || return 1
  require_regular "$LOCK" restored_runtime_lock
  [[ "$(stat -Lc '%d:%i:%h:%a:%u:%g' "$LOCK")" = *":1:600:1000:1000" ]] || return 1
  /usr/bin/flock -n "$LOCK" -c true && return 1
  local owner
  owner="$(sudo -n -- /usr/bin/sh -c 'for f in /proc/'"$new_pid"'/fd/*; do [ "$(readlink "$f" 2>/dev/null)" = "'"$LOCK"'" ] && echo found && exit 0; done; exit 1' 2>/dev/null || true)"
  [[ "$owner" = found ]] || return 1
  /usr/bin/curl --fail --silent --show-error --max-time 5 http://127.0.0.1:3000/health >/dev/null || return 1
}
restore_after_failure() {
  local code=$?
  trap - EXIT INT TERM
  set +e
  if [[ "$STOPPED" = 1 ]]; then
    if [[ "$LOCK_CLEAN" != 1 ]]; then cleanup_lock_substrate || true; fi
    if [[ "$LOCK_CLEAN" = 1 ]]; then "${SYSTEMCTL[@]}" start "$SERVICE" >/dev/null 2>&1 || code=1; fi
  fi
  exit "$code"
}
execute_probe() {
  mkdir -m 0750 "$BASE/attempts/attempt1"
  [[ "$("${SYSTEMCTL[@]}" is-active "$SERVICE")" = active ]] || fail "service is not active"
  [[ "$("${SYSTEMCTL[@]}" show "$SERVICE" -p SubState --value)" = running ]] || fail "service is not running"
  OLD_PID="$("${SYSTEMCTL[@]}" show "$SERVICE" -p MainPID --value)"
  OLD_RESTARTS="$("${SYSTEMCTL[@]}" show "$SERVICE" -p NRestarts --value)"
  OLD_HASHES="$(service_active_hashes)"
  printf 'old_main_pid=%s\nold_nrestarts=%s\nactive_package_worker_sha=%s\n' "$OLD_PID" "$OLD_RESTARTS" "$OLD_HASHES" > "$BASE/attempts/attempt1/prestate.txt"
  "${SYSTEMCTL[@]}" stop "$SERVICE" >/dev/null
  STOPPED=1
  : > "$STOP_MARKER"
  stable2_stopped || fail "stable2 stopped gates did not pass"
  [[ ! -e "$RUNTIME_DIR" ]] || fail "runtime directory remains after service stop"
  "${RUNTIME_DIR_INSTALL[@]}" "$RUNTIME_DIR"
  install -o homelab1 -g homelab1 -m 0600 /dev/null "$LOCK"
  LOCK_CREATED=1
  LOCK_IDENTITY="$(stat -Lc '%d:%i:%h:%a:%u:%g' "$LOCK")"
  [[ "$LOCK_IDENTITY" = *":1:600:1000:1000" ]] || fail "runtime lock identity differs"
  require_absent "$OUTPUT"
  mkdir -m 0750 "$OUTPUT"
  {
    exec 9>"$LOCK"
    flock -n 9 || fail "failed to acquire exact runtime lock inode"
    timeout --signal=TERM --kill-after=30s 1200s env -u ULLM_AQ4_MATVEC_RPB -u ULLM_AQ4_FUSED_RPB HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL=1 ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB=4 \
      "$PROBE" --package "$PACKAGE_ROOT" --input "$INPUT" --output-dir "$OUTPUT" --device-index "$EXPECTED_LOGICAL_DEVICE_INDEX"
  } >"$RUN_LOG" 2>&1
  validate_output_contract
  cleanup_lock_substrate || fail "lock unlink+rmdir did not complete while service was stopped"
  "${SYSTEMCTL[@]}" start "$SERVICE" >/dev/null || fail "service start failed"
  STOPPED=0
  post_start_check || fail "new service epoch/health/lock owner validation failed"
}

validate_preflight
if [[ "$MOCK_PREFLIGHT" = 1 ]]; then
  echo "mock_preflight=1 service_stop=0 gpu_run=0 numeric_threshold=none promotion_eligible=false"
  exit 0
fi
if [[ "$PREFLIGHT_ONLY" = 1 ]]; then
  echo "preflight_only=1 service_stop=0 gpu_run=0 numeric_threshold=none promotion_eligible=false"
  exit 0
fi
if [[ "$PREFLIGHT_LOCKED_ONLY" = 1 ]]; then
  [[ "$("${SYSTEMCTL[@]}" is-active "$SERVICE")" = active ]] || fail "service is not active"
  echo "preflight_locked_only=1 service_stop=0 gpu_run=0 numeric_threshold=none promotion_eligible=false"
  exit 0
fi
[[ "$EXECUTE_GPU_PROBE" = 1 ]] || fail "set EXECUTE_GPU_PROBE=1 for the explicit service/GPU diagnostic run"
trap restore_after_failure EXIT INT TERM
execute_probe
trap - EXIT INT TERM
