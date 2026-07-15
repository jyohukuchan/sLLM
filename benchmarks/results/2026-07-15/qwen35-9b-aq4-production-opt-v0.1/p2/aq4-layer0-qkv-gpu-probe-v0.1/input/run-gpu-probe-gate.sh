#!/usr/bin/env bash
set -Eeuo pipefail
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PATH

# Diagnostic-only gate for the standalone layer-0 AQ4 matvec probe.  The default path is
# read-only preflight.  The service/GPU path requires EXECUTE_GPU_PROBE=1 and never produces
# a numeric Go/No-Go or promotion decision.
REPO="/home/homelab1/coding-local/ultimateLLM/uLLM-project"
BASE="$REPO/benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-qkv-gpu-probe-v0.1"
INPUT="$REPO/benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-matvec-oracle-integration-v0.1/runtime-input.jsonl"
PACKAGE_ROOT="/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package"
PACKAGE_MANIFEST="$PACKAGE_ROOT/manifest.json"
ACTIVE="/etc/ullm/served-models/active.json"
PROFILE="$REPO/deploy/served-models/qwen35-9b-aq4-reasoning.profile.json"
WORKER="$REPO/target/reasoning-v2/release/ullm-aq4-worker"

PROBE_ROOT="$BASE/input/probe-binary-v0.1"
PROBE="$PROBE_ROOT/ullm-aq4-layer0-qkv-runtime-probe"
RECEIPT="$PROBE_ROOT/build-receipt.json"
SUMS="$PROBE_ROOT/SHA256SUMS"
OUTPUT="$BASE/output"
RUN_LOG="$BASE/run.log"
MONITOR_LOG="$BASE/monitor.log"
STOP_MARKER="$BASE/service-stopped.marker"
OBSERVER_FAIL_MARKER="$BASE/observer-failed.marker"
OBSERVER_SAMPLE_MARKER="$BASE/observer-sample.marker"

EXPECTED_INPUT_SHA256="c009a9bded30b1b9a7c704c622bd3106b3d17989c438f91eb20bb16817348e17"
EXPECTED_PACKAGE_SHA256="a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad"
EXPECTED_ACTIVE_SHA256="feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44"
EXPECTED_PROFILE_SHA256="1013cc803adce27a178856e9fe300fc5a26cc998b5381812b0c3bd17ebbb5937"
EXPECTED_WORKER_SHA256="177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d"
EXPECTED_BUILD_COMMIT="4a4b0e28eb27fa6710a339e470ee80d21d602680"
EXPECTED_BUILD_COMMAND="CARGO_BUILD_JOBS=1 cargo build --release -p ullm-engine --bin ullm-aq4-layer0-qkv-runtime-probe"
EXPECTED_PROBE_BINARY_SHA256="f58f0734ec595d9a9cd76161d28d096b2b18fc6e437cf3ad9d526eb710c7cf69"
EXPECTED_BUILD_RECEIPT_SHA256="34a39bd6f0d20dd8a98991f3c66b6a538697bf0689894150139a929e3c3c32c4"
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
ROCM_SMI="/opt/rocm/bin/rocm-smi"

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
assert r["schema_version"] == "ullm.aq4_layer0_qkv_runtime_probe_build_receipt.v1"
assert r["status"] == "ready"
assert r["source"]["commit"] == expected_commit and r["source"]["tree_clean"] is True
assert r["build"]["command"] == expected_command and r["build"]["jobs"] == 1 and r["build"]["exit_status"] == 0
b = r["binary"]
p = pathlib.Path(binary_path)
assert b["path"] == p.name and b["nlink"] == 1 and b["bytes"] == p.stat().st_size
assert b["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
print("probe_build=clean_commit receipt command jobs1 binary_sha nlink1")
PY
}

validate_output_contract() {
  "$PYTHON" - "$BASE" "$OUTPUT" "$INPUT" "$EXPECTED_INPUT_SHA256" "$EXPECTED_PACKAGE_SHA256" "$EXPECTED_ACTIVE_SHA256" "$EXPECTED_WORKER_SHA256" "$EXPECTED_LOGICAL_DEVICE_INDEX" "$EXPECTED_FILTERED_HIP_ORDINAL" "$EXPECTED_DEVICE_ARCHITECTURE" <<'PY'
import hashlib, json, pathlib, sys
base, output, input_path, input_sha, package_sha, active_sha, worker_sha, logical, hip_ordinal, arch = sys.argv[1:]
out = pathlib.Path(output)
report_path, data_path = out / "report.json", out / "output.f32le"
assert report_path.is_file() and data_path.is_file()
r = json.loads(report_path.read_text(encoding="utf-8"))
assert r["schema_version"] == "ullm.aq4_layer0_qkv_runtime_probe.v1"
assert r["status"] == "valid" and r["classification"] == "unclassified"
assert r["promotion_eligible"] is False and r["fused"] is False
assert r["operation"] == "standalone_aq4_matvec_f32"
d = r["device"]
assert d["backend"].lower() == "hip" and str(d["device_index"]) == logical
assert str(d["device_id"]) == hip_ordinal and d["gcn_arch_name"] == arch
assert r["input"]["sidecar_sha256"] == input_sha
i = r["input"]["identity"]
assert i["consumed_sha256"] == input_sha and i["pre_stat"] == i["post_stat"] and i["pre_stat"]["nlink"] == 1
assert r["package"]["manifest_sha256"] == package_sha
assert hashlib.sha256(pathlib.Path(input_path).read_bytes()).hexdigest() == input_sha
assert r["output"]["sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest()
assert r["output"]["bytes"] == data_path.stat().st_size
assert r["output"]["cases"] and all(row["output_elements"] == 8192 for row in r["output"]["cases"])
assert all(v == v and abs(v) != float("inf") for v in [r["output"]["bytes"]])
print("gpu_probe=valid hip logical_device1 hip_ordinal0 gfx1201 standalone fused0 unclassified promotion0")
PY
}

validate_preflight() {
  PYTHON="${PYTHON:-/usr/bin/python3}"
  [[ -x "$PYTHON" ]] || fail "fixed Python runtime is unavailable"
  [[ "$EXPECTED_PHYSICAL_CARD" = 2 && "$EXPECTED_HIP_VISIBLE_TOKEN" = 1 && "$EXPECTED_FILTERED_HIP_ORDINAL" = 0 && "$EXPECTED_CPU_GLOBAL_DEVICE_INDEX" = 0 && "$EXPECTED_LOGICAL_DEVICE_INDEX" = 1 && "$EXPECTED_DEVICE_ARCHITECTURE" = gfx1201 ]] || fail "device mapping pins changed"
  validate_identity
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

execute_probe() {
  local observer_pid="" mainpid_before="" restart_before="" active_hashes_before="" service_started=0 lock_ready=0
  local cleanup_status=0
  cleanup() {
    cleanup_status=$?
    trap - EXIT INT TERM
    set +e
    if [[ -n "${observer_pid:-}" ]]; then kill "$observer_pid" 2>/dev/null || true; wait "$observer_pid" 2>/dev/null || true; fi
    if [[ "${service_started:-0}" = 1 ]]; then
      timeout --signal=TERM --kill-after=5s 60s "${SYSTEMCTL[@]}" start "$SERVICE" >/dev/null 2>&1 || true
      for _ in $(seq 1 60); do [[ "$("${SYSTEMCTL[@]}" is-active "$SERVICE" 2>/dev/null)" = active ]] && break; sleep 1; done
      [[ "$("${SYSTEMCTL[@]}" is-active "$SERVICE" 2>/dev/null)" = active ]] || cleanup_status=1
      [[ "$(service_active_hashes)" = "$active_hashes_before" ]] || cleanup_status=1
      [[ "$("${SYSTEMCTL[@]}" show "$SERVICE" -p NRestarts --value 2>/dev/null)" = "$restart_before" ]] || cleanup_status=1
    fi
    if [[ "${lock_ready:-0}" = 1 && -e "$LOCK" ]]; then rm -f -- "$LOCK" 2>/dev/null || true; fi
    if [[ -d "$RUNTIME_DIR" && -z "$(find "$RUNTIME_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then "${RUNTIME_DIR_REMOVE[@]}" "$RUNTIME_DIR" 2>/dev/null || true; fi
    exit "$cleanup_status"
  }
  trap cleanup EXIT INT TERM
  [[ "$("${SYSTEMCTL[@]}" is-active "$SERVICE")" = active ]] || fail "service is not active"
  mainpid_before=$("${SYSTEMCTL[@]}" show "$SERVICE" -p MainPID --value)
  restart_before=$("${SYSTEMCTL[@]}" show "$SERVICE" -p NRestarts --value)
  active_hashes_before=$(service_active_hashes)
  (trap - EXIT INT TERM; run_observer) & observer_pid=$!
  sleep 2
  [[ -e "$OBSERVER_SAMPLE_MARKER" ]] || fail "observer did not produce a sample"
  service_started=1
  timeout --signal=TERM --kill-after=5s 60s "${SYSTEMCTL[@]}" stop "$SERVICE" >/dev/null
  : > "$STOP_MARKER"
  for _ in $(seq 1 60); do [[ "$("${SYSTEMCTL[@]}" is-active "$SERVICE" 2>/dev/null)" = inactive ]] && break; sleep 1; done
  [[ "$("${SYSTEMCTL[@]}" is-active "$SERVICE" 2>/dev/null)" = inactive ]] || fail "service did not stop"
  [[ ! -e "$RUNTIME_DIR" ]] || fail "RuntimeDirectory still exists after stop"
  "${RUNTIME_DIR_INSTALL[@]}" "$RUNTIME_DIR"
  install -o homelab1 -g homelab1 -m 600 /dev/null "$LOCK"
  lock_ready=1
  require_absent "$OUTPUT"
  mkdir -m 0750 "$OUTPUT"
  {
    exec 9>"$LOCK"
    flock -n 9 || fail "failed to acquire probe lock"
    timeout --signal=TERM --kill-after=30s 1200s env HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1 \
      "$PROBE" --package "$PACKAGE_ROOT" --input "$INPUT" --output-dir "$OUTPUT" --device-index "$EXPECTED_LOGICAL_DEVICE_INDEX"
  } >"$RUN_LOG" 2>&1
  validate_output_contract
  echo "diagnostic_only=1 numeric_threshold=none holdout=unobserved promotion_eligible=false"
  # The trap restores the service and removes the temporary lock substrate.
  trap - EXIT
  cleanup
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
execute_probe
