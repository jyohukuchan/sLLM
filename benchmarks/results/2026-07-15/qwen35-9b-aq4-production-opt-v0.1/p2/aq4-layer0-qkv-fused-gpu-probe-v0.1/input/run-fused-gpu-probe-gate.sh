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
ROCM_SMI="${ROCM_SMI:-/opt/rocm/bin/amd-smi}"
SUDO=(sudo -n --)

PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
PREFLIGHT_LOCKED_ONLY="${PREFLIGHT_LOCKED_ONLY:-0}"
MOCK_PREFLIGHT="${MOCK_PREFLIGHT:-0}"
EXECUTE_GPU_PROBE="${EXECUTE_GPU_PROBE:-0}"
MOCK_OBSERVER="${MOCK_OBSERVER:-0}"
if [[ "$PREFLIGHT_ONLY" = 1 && "$PREFLIGHT_LOCKED_ONLY" = 1 ]]; then
  echo "preflight modes are mutually exclusive" >&2
  exit 64
fi
if [[ "$MOCK_PREFLIGHT" = 1 ]]; then
  # Mock mode cannot reach systemctl, rocm-smi, or the probe. Explicit archive setup
  # below is the only mock path that materializes the pinned runtime copy.
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

ATTEMPT_ROOT="${ATTEMPT_ROOT:-$BASE/attempts/attempt1}"
RUNTIME_PROBE="${RUNTIME_PROBE:-$ATTEMPT_ROOT/ullm-aq4-layer0-qkv-z-gate-beta-runtime-probe}"
RUNTIME_PROBE_META="${RUNTIME_PROBE_META:-$ATTEMPT_ROOT/runtime-probe-stat.json}"

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

prepare_runtime_probe() {
  require_absent "$RUNTIME_PROBE"
  require_absent "$RUNTIME_PROBE_META"
  install -m 0555 -- "$PROBE" "$RUNTIME_PROBE" || fail "runtime probe copy failed"
  chmod 0555 -- "$RUNTIME_PROBE" || fail "runtime probe chmod failed"
  require_regular "$RUNTIME_PROBE" runtime_probe_binary
  [[ "$(stat -Lc '%a' "$RUNTIME_PROBE")" = 555 ]] || fail "runtime probe mode differs"
  [[ "$(sha256_file "$RUNTIME_PROBE")" = "$EXPECTED_PROBE_BINARY_SHA256" ]] || fail "runtime probe SHA differs"
  "$PYTHON" - "$RUNTIME_PROBE_META" "$PROBE" "$RUNTIME_PROBE" <<'PY'
import hashlib, json, os, pathlib, stat, sys

meta_path, source_path, runtime_path = map(pathlib.Path, sys.argv[1:])
source_stat = source_path.stat()
runtime_stat = runtime_path.stat()
payload = {
    "schema_version": "ullm.aq4_layer0_qkv_fused_gpu_probe_runtime_binary.v1",
    "source": {
        "path": str(source_path),
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "bytes": source_stat.st_size,
        "mode": format(stat.S_IMODE(source_stat.st_mode), "04o"),
        "nlink": source_stat.st_nlink,
    },
    "runtime": {
        "path": str(runtime_path),
        "sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
        "bytes": runtime_stat.st_size,
        "mode": format(stat.S_IMODE(runtime_stat.st_mode), "04o"),
        "nlink": runtime_stat.st_nlink,
    },
}
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
fd = os.open(meta_path, flags, 0o644)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True, indent=2)
    stream.write("\n")
PY
  require_regular "$RUNTIME_PROBE_META" runtime_probe_archive
  "$PYTHON" - "$RUNTIME_PROBE_META" "$EXPECTED_PROBE_BINARY_SHA256" <<'PY'
import json, pathlib, sys

meta_path, expected_sha = sys.argv[1:]
r = json.loads(pathlib.Path(meta_path).read_text(encoding="utf-8"))
assert r["schema_version"] == "ullm.aq4_layer0_qkv_fused_gpu_probe_runtime_binary.v1"
assert r["runtime"]["sha256"] == expected_sha
assert r["runtime"]["mode"] == "0555" and r["runtime"]["nlink"] == 1
assert r["source"]["sha256"] == expected_sha
PY
}

validate_output_contract() {
  "$PYTHON" - "$OUTPUT" "$INPUT" "$EXPECTED_INPUT_SHA256" "$EXPECTED_PACKAGE_SHA256" <<'PY'
import hashlib, json, math, pathlib, struct, sys
output, input_path, input_sha, package_sha = sys.argv[1:]
out = pathlib.Path(output)
report_path = out / "report.json"
report_stat = report_path.stat()
assert report_path.is_file() and not report_path.is_symlink() and report_stat.st_nlink == 1
r = json.loads(report_path.read_text(encoding="utf-8"))
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
identity = r["input"]["identity"]
assert identity["pre_stat"] == identity["post_stat"]
assert identity["consumed_sha256"] == input_sha
assert r["package"]["manifest_sha256"] == package_sha
reference = r["qkv_component_reference"]
assert reference["reference_backend"] == "hip"
assert reference["reference_kind"] == "diagnostic_standalone_reference"
assert reference["operation"] == "standalone_aq4_matvec_f32"
assert reference["standalone_rpb_raw"] is None
assert reference["standalone_rpb_effective"] == 32
assert reference["standalone_rpb_source"] == "architecture_default:gfx1201"
assert reference["standalone_output_key"] == "qkv_standalone"
assert math.isfinite(reference["max_abs"]) and math.isfinite(reference["relative_l2"])
assert r["qkv_row_segments"] == [
    {"name": "Q", "start_row": 0, "end_row_exclusive": 2048},
    {"name": "K", "start_row": 2048, "end_row_exclusive": 4096},
    {"name": "V", "start_row": 4096, "end_row_exclusive": 8192},
]
assert r["output_layout"] == {
    "format": "concatenated_little_endian_f32_rows",
    "dtype": "f32",
    "row_order": "input_jsonl_order",
    "qkv_shape": [8192],
    "qkv_standalone_shape": [8192],
    "z_shape": [4096],
    "gate_shape": [32],
    "beta_shape": [32],
}
expected = {"qkv": "qkv.f32le", "qkv_standalone": "qkv-standalone.f32le", "z": "z.f32le", "gate": "gate.f32le", "beta": "beta.f32le"}
assert set(r["outputs"]) == set(expected)
rows = r["input"]["rows"]
assert isinstance(rows, int) and rows > 0
shapes = {"qkv": 8192, "qkv_standalone": 8192, "z": 4096, "gate": 32, "beta": 32}
for key, name in expected.items():
    path = out / name
    metadata = path.stat()
    assert path.is_file() and not path.is_symlink() and metadata.st_nlink == 1
    raw = path.read_bytes()
    output = r["outputs"][key]
    assert pathlib.Path(output["path"]).name == name
    assert output["row_shape"] == [shapes[key]]
    assert output["bytes"] == len(raw) == rows * shapes[key] * 4 and len(raw) % 4 == 0
    assert output["sha256"] == hashlib.sha256(raw).hexdigest()
    assert len(output["cases"]) == rows
    for index, case in enumerate(output["cases"]):
        start = index * shapes[key] * 4
        row = raw[start:start + shapes[key] * 4]
        assert case["output_offset_bytes"] == start
        assert case["output_elements"] == shapes[key]
        assert case["output_sha256"] == hashlib.sha256(row).hexdigest()
        assert case["finite"] is True
    for (value,) in struct.iter_unpack("<f", raw):
        assert math.isfinite(value)
assert report_stat.st_size > 0
print("fused_report=valid schema2 hip device1 gfx1201 fused_rpb4 standalone_rpb=unset/default32 finite_outputs=5 promotion=false")
PY
}
validate_preflight() {
  PYTHON="${PYTHON:-/usr/bin/python3}"
  [[ -x "$PYTHON" ]] || fail "fixed Python runtime is unavailable"
  [[ "$EXPECTED_PHYSICAL_CARD" = 2 && "$EXPECTED_HIP_VISIBLE_TOKEN" = 1 && "$EXPECTED_FILTERED_HIP_ORDINAL" = 0 && "$EXPECTED_CPU_GLOBAL_DEVICE_INDEX" = 0 && "$EXPECTED_LOGICAL_DEVICE_INDEX" = 1 && "$EXPECTED_DEVICE_ARCHITECTURE" = gfx1201 ]] || fail "device mapping pins changed"
  validate_identity
  require_absent "$ATTEMPT_ROOT"
  require_absent "$RUNTIME_PROBE"
  require_absent "$RUNTIME_PROBE_META"
  require_absent "$OUTPUT"
  require_absent "$RUN_LOG"
  require_absent "$MONITOR_LOG"
  require_absent "$STOP_MARKER"
  require_absent "$OBSERVER_FAIL_MARKER"
  require_absent "$OBSERVER_SAMPLE_MARKER"
}

observer_sample_once() {
  local sample normalized
  if ! sample="$("$ROCM_SMI" --showmeminfo vram --showuse --showpower --json 2>&1)"; then
    : > "$OBSERVER_FAIL_MARKER"
    return 1
  fi
  if ! normalized="$(printf '%s' "$sample" | "$PYTHON" -c '
import json
import sys

payload = json.load(sys.stdin)
card = "card" + sys.argv[1]
architecture = sys.argv[2]
value = payload.get(card)
if not isinstance(value, dict) or value.get("GFX Version") != architecture:
    raise SystemExit(1)
print(json.dumps({"physical_card": int(sys.argv[1]), "architecture": architecture, "sample": payload}, separators=(",", ":")))
' "$EXPECTED_PHYSICAL_CARD" "$EXPECTED_DEVICE_ARCHITECTURE")"; then
    : > "$OBSERVER_FAIL_MARKER"
    return 1
  fi
  printf '%s\n' "$normalized" >> "$MONITOR_LOG" || {
    : > "$OBSERVER_FAIL_MARKER"
    return 1
  }
  : > "$OBSERVER_SAMPLE_MARKER"
}

run_observer() {
  : > "$MONITOR_LOG" || {
    : > "$OBSERVER_FAIL_MARKER"
    return 1
  }
  while :; do
    observer_sample_once || return 1
    sleep 2 || return 0
  done
}

OBSERVER_PID=""
OBSERVER_STARTED=0
start_observer() {
  (
    # Do not inherit the parent EXIT/INT/TERM trap into the observer child.
    trap - EXIT INT TERM HUP
    run_observer
  ) &
  OBSERVER_PID=$!
  OBSERVER_STARTED=1
  for _ in $(seq 1 10); do
    [[ -e "$OBSERVER_FAIL_MARKER" ]] && fail "observer failed before first sample"
    [[ -e "$OBSERVER_SAMPLE_MARKER" ]] && return 0
    kill -0 "$OBSERVER_PID" 2>/dev/null || fail "observer exited before first sample"
    sleep 1
  done
  fail "observer did not produce a first sample"
}

stop_observer() {
  local wait_status=0
  if [[ "$OBSERVER_STARTED" = 1 ]]; then
    kill -TERM "$OBSERVER_PID" 2>/dev/null || true
    wait "$OBSERVER_PID" || wait_status=$?
    OBSERVER_STARTED=0
    OBSERVER_PID=""
  fi
  [[ "$wait_status" = 0 || "$wait_status" = 143 || "$wait_status" = 130 ]] || return 1
  [[ ! -e "$OBSERVER_FAIL_MARKER" ]] || return 1
  [[ -e "$OBSERVER_SAMPLE_MARKER" ]] || return 1
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
  local code=$? observer_rc=0 cleanup_rc=0 start_rc=0
  trap - EXIT INT TERM
  set +e
  if [[ "$OBSERVER_STARTED" = 1 ]]; then
    stop_observer
    observer_rc=$?
  fi
  if [[ "$STOPPED" = 1 ]]; then
    if [[ "$LOCK_CLEAN" != 1 ]]; then
      cleanup_lock_substrate
      cleanup_rc=$?
    fi
    # Service restoration is attempted even when lock cleanup fails. No health,
    # worker, or GPU operations are performed after this start attempt.
    "${SYSTEMCTL[@]}" start "$SERVICE" >/dev/null 2>&1
    start_rc=$?
    STOPPED=0
  fi
  [[ "$observer_rc" = 0 && "$cleanup_rc" = 0 && "$start_rc" = 0 ]] || code=1
  exit "$code"
}
execute_probe() {
  [[ ! -L "$BASE" ]] || fail "BASE must not be a symlink"
  mkdir -p -- "$BASE/attempts" || fail "attempt parent creation failed"
  [[ -d "$BASE/attempts" && ! -L "$BASE/attempts" ]] || fail "attempt parent must be a directory"
  require_absent "$ATTEMPT_ROOT"
  mkdir -m 0750 -- "$ATTEMPT_ROOT" || fail "attempt directory create-new failed"
  prepare_runtime_probe
  [[ "$("${SYSTEMCTL[@]}" is-active "$SERVICE")" = active ]] || fail "service is not active"
  [[ "$("${SYSTEMCTL[@]}" show "$SERVICE" -p SubState --value)" = running ]] || fail "service is not running"
  OLD_PID="$("${SYSTEMCTL[@]}" show "$SERVICE" -p MainPID --value)"
  OLD_RESTARTS="$("${SYSTEMCTL[@]}" show "$SERVICE" -p NRestarts --value)"
  OLD_HASHES="$(service_active_hashes)"
  printf 'old_main_pid=%s\nold_nrestarts=%s\nactive_package_worker_sha=%s\n' "$OLD_PID" "$OLD_RESTARTS" "$OLD_HASHES" > "$ATTEMPT_ROOT/prestate.txt"
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
    start_observer
    timeout --signal=TERM --kill-after=30s 1200s env -u ULLM_AQ4_MATVEC_RPB -u ULLM_AQ4_FUSED_RPB HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL=1 ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB=4 \
      "$RUNTIME_PROBE" --package "$PACKAGE_ROOT" --input "$INPUT" --output-dir "$OUTPUT" --device-index "$EXPECTED_LOGICAL_DEVICE_INDEX"
  } >"$RUN_LOG" 2>&1
  stop_observer || fail "observer did not stop cleanly"
  [[ ! -e "$OBSERVER_FAIL_MARKER" ]] || fail "observer failure marker is present"
  [[ -e "$OBSERVER_SAMPLE_MARKER" ]] || fail "observer sample marker is absent"
  validate_output_contract
  cleanup_lock_substrate || fail "lock unlink+rmdir did not complete while service was stopped"
  "${SYSTEMCTL[@]}" start "$SERVICE" >/dev/null || fail "service start failed"
  STOPPED=0
  post_start_check || fail "new service epoch/health/lock owner validation failed"
}

validate_preflight
if [[ "${MOCK_ARCHIVE_SETUP:-0}" = 1 ]]; then
  [[ ! -L "$BASE" ]] || fail "BASE must not be a symlink"
  mkdir -p -- "$BASE/attempts" || fail "attempt parent creation failed"
  [[ -d "$BASE/attempts" && ! -L "$BASE/attempts" ]] || fail "attempt parent must be a directory"
  require_absent "$ATTEMPT_ROOT"
  mkdir -m 0750 -- "$ATTEMPT_ROOT" || fail "attempt directory create-new failed"
  prepare_runtime_probe
  echo "mock_archive_setup=1 runtime_probe_mode=0555 runtime_probe_nlink=1 runtime_probe_sha256=$EXPECTED_PROBE_BINARY_SHA256 service_stop=0 gpu_run=0"
  exit 0
fi
if [[ "$MOCK_OBSERVER" = 1 ]]; then
  observer_sample_once || fail "mock observer sample failed"
  [[ ! -e "$OBSERVER_FAIL_MARKER" ]] || fail "mock observer failure marker is present"
  [[ -e "$OBSERVER_SAMPLE_MARKER" ]] || fail "mock observer sample marker is absent"
  echo "mock_observer=1 service_stop=0 gpu_run=0 numeric_threshold=none promotion_eligible=false"
  exit 0
fi
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
