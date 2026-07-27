#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0
#
# Resumable, no-activation validation runbook for a rented MI300X/gfx942 host.
# It deliberately covers only the CDNA3 A′/B decision gates.  It never edits
# /etc/ullm, starts a serving process, downloads a model, or enables a release.

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run-sq8-cdna3-mi300x-validation.sh [options]

Options:
  --repo /absolute/repository       Repository to validate (default: this script's parent).
  --results-dir /absolute/directory Persistent logs and resume stamps.
  --jobs N                          Cargo build jobs; default: 8.
  --hip-visible-devices TOKEN       One GPU token; default: $HIP_VISIBLE_DEVICES or 0.
  --stage NAME                      all, preflight, cpu, hiprtc, build, isa, physical, or hw_microbench.
  --allow-network                   Permit Cargo to fetch missing dependencies; default is offline.
  --rehearsal-no-gfx942             Permit offline rehearsal without gfx942; physical remains an expected failure.
  --dry-run                         Validate arguments and print the ordered plan without writing/running.
  --help                            Show this help.

Before renting, clone the checked revision and warm its Cargo cache.  The
default offline mode fails early when that provisioning was skipped, instead of
spending leased GPU time on downloads.  Re-run the same command after a failure:
completed stages have a .done stamp and are not repeated.

`hw_microbench` is deliberately a non-P0, opt-in stage. It requires every P0
stamp including `physical`, so P0 cannot be skipped or weakened. It is the
first item to cut when lease time is short (after the established profiler /
occupancy, full-model, model/image, external-engine, timing-sweep, and
hand-written-A priorities); never cut P0 for it. Set all three
HW_MB_{MEMORY_PEAK_GBPS,BF16_PEAK_TFLOPS,FP8_PEAK_TFLOPS} values from the
rental host's recorded official/amd-smi source before running this stage. Also
set HW_MB_AMD_SMI_GPU to the physical GPU index selected by the one-token
HIP_VISIBLE_DEVICES mapping; it is recorded with continuous clock telemetry.
EOF
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd -- "$script_dir/.." && pwd -P)
results_dir=
jobs=${ULLM_RENTAL_JOBS:-8}
hip_visible_devices=${HIP_VISIBLE_DEVICES:-0}
requested_stage=all
allow_network=0
rehearsal_no_gfx942=0
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      repo_dir=$2
      shift 2
      ;;
    --results-dir)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      results_dir=$2
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      jobs=$2
      shift 2
      ;;
    --hip-visible-devices)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      hip_visible_devices=$2
      shift 2
      ;;
    --stage)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      requested_stage=$2
      shift 2
      ;;
    --allow-network)
      allow_network=1
      shift
      ;;
    --rehearsal-no-gfx942)
      rehearsal_no_gfx942=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ $repo_dir != /* || ! -f $repo_dir/Cargo.toml ]]; then
  printf 'repository must be an absolute directory containing Cargo.toml: %s\n' "$repo_dir" >&2
  exit 2
fi
repo_dir=$(cd -- "$repo_dir" && pwd -P)
if [[ -z $results_dir ]]; then
  results_dir="$repo_dir/benchmarks/results/$(date +%F)/mi300x-rental-v2"
fi
if [[ $results_dir != /* || ! $jobs =~ ^[1-9][0-9]*$ || $hip_visible_devices == *,* || -z $hip_visible_devices ]]; then
  usage
  exit 2
fi
case "$requested_stage" in
  all|preflight|cpu|hiprtc|build|isa|physical|hw_microbench) ;;
  *)
    usage
    exit 2
    ;;
esac

rocm_path=${ROCM_PATH:-/opt/rocm}
rental_linker=${ULLM_RENTAL_LINKER:-cc}
target_dir="$results_dir/cargo-target"
state_dir="$results_dir/state"
logs_dir="$results_dir/logs"
audit_dir="$results_dir/isa"
hiprtc_dir="$results_dir/hiprtc"
smoke_binary="$target_dir/release/examples/sq8_gfx942_aprime_physical_smoke"
timings_file="$results_dir/stage-timings.tsv"

stages=(preflight cpu hiprtc build isa physical)
if (( dry_run )); then
  printf 'dry-run repo=%s\n' "$repo_dir"
  printf 'dry-run results_dir=%s\n' "$results_dir"
  printf 'dry-run HIP_VISIBLE_DEVICES=%s jobs=%s offline=%s\n' \
    "$hip_visible_devices" "$jobs" "$((1 - allow_network))"
  printf 'priority P0 stages: %s\n' "${stages[*]}"
  printf 'physical stage unsets ULLM_SMOKE_SKIP_B_CONTROL and runs the five A′ shapes plus B control.\n'
  exit 0
fi

mkdir -p -- "$state_dir" "$logs_dir" "$audit_dir" "$hiprtc_dir"
head_revision=$(cd -- "$repo_dir" && git rev-parse HEAD)
# A state directory must not be reused after a tracked source edit, even when
# HEAD itself did not move.  (A clean rental checkout has the stable hash of an
# empty patch.)  Untracked artifacts are deliberately outside this hash: the
# runner writes its own results below --results-dir.
tracked_patch_sha256=$(cd -- "$repo_dir" && git diff --no-ext-diff --binary HEAD | sha256sum | awk '{print $1}')
fingerprint="${head_revision}+${tracked_patch_sha256}"
fingerprint_file="$state_dir/revision.txt"
if [[ -f $fingerprint_file ]] && [[ $(<"$fingerprint_file") != "$fingerprint" ]]; then
  printf 'resume state belongs to revision %s, not %s; choose a new --results-dir\n' \
    "$(<"$fingerprint_file")" "$fingerprint" >&2
  exit 1
fi
printf '%s\n' "$fingerprint" >"$fingerprint_file"
if [[ ! -f $timings_file ]]; then
  printf 'stage\tstarted\tfinished\tduration_seconds\tstatus\n' >"$timings_file"
fi

run_step() {
  local name=$1
  shift
  local stamp="$state_dir/$name.done"
  local log="$logs_dir/$name.log"
  local status
  if [[ -f $stamp ]]; then
    printf 'SKIP %s (already complete)\n' "$name"
    return 0
  fi
  printf 'START %s\n' "$name"
  local started_at started_epoch finished_at duration_seconds
  started_at=$(date -Is)
  started_epoch=$(date +%s)
  if "$@" >"$log" 2>&1; then
    finished_at=$(date -Is)
    duration_seconds=$(( $(date +%s) - started_epoch ))
    date -Is >"$stamp"
    printf '%s\t%s\t%s\t%s\tpass\n' \
      "$name" "$started_at" "$finished_at" "$duration_seconds" >>"$timings_file"
    printf 'PASS %s\n' "$name"
    return 0
  else
    status=$?
  fi
  finished_at=$(date -Is)
  duration_seconds=$(( $(date +%s) - started_epoch ))
  printf '%s\t%s\t%s\t%s\tfail:%s\n' \
    "$name" "$started_at" "$finished_at" "$duration_seconds" "$status" >>"$timings_file"
  printf 'FAIL %s (exit %s); inspect %s and re-run this script to resume\n' \
    "$name" "$status" "$log" >&2
  return "$status"
}

command_is_available() {
  local candidate=$1
  if [[ $candidate == */* ]]; then
    [[ -x $candidate ]]
  else
    command -v "$candidate" >/dev/null 2>&1
  fi
}

optional_tool_status() {
  local candidate=$1
  local resolved
  if resolved=$(command -v "$candidate" 2>/dev/null); then
    printf 'optional_%s=%s (not required by the rental runner)\n' "$candidate" "$resolved"
  else
    printf 'optional_%s=unavailable (not required by the rental runner)\n' "$candidate"
  fi
}

rental_cargo() {
  local -a cargo_args=("$@")
  if (( ! allow_network )); then
    cargo_args+=(--offline)
  fi
  cargo_args+=(--locked)
  # These environment keys override .cargo/config.toml only for the rental
  # command.  Local developers retain clang+mold; a rental host can use cc and
  # an empty Rust flag set when mold is unavailable.
  env \
    HIP_VISIBLE_DEVICES=-1 \
    GPU_ARCH=gfx942 \
    ROCM_PATH="$rocm_path" \
    CARGO_BUILD_JOBS="$jobs" \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$rental_linker" \
    CARGO_ENCODED_RUSTFLAGS="${ULLM_RENTAL_ENCODED_RUSTFLAGS:-}" \
    CARGO_TARGET_DIR="$target_dir" \
    cargo "${cargo_args[@]}"
}

stage_preflight() {
  cd -- "$repo_dir"
  for command in cargo rustc git c++ "$rocm_path/bin/hipcc" "$rocm_path/llvm/bin/llvm-objdump" \
    "$rocm_path/llvm/bin/llvm-readelf" "$rocm_path/llvm/bin/llvm-objcopy" \
    "$rocm_path/llvm/bin/clang-offload-bundler" rocminfo zstd; do
    if ! command_is_available "$command"; then
      printf 'required command is unavailable: %s\n' "$command" >&2
      return 1
    fi
  done
  if ! command_is_available "$rental_linker"; then
    printf 'required rental linker is unavailable: %s (set ULLM_RENTAL_LINKER to an executable C linker)\n' \
      "$rental_linker" >&2
    return 1
  fi
  rocminfo >"$results_dir/rocminfo.txt"
  local observed_arches
  observed_arches=$(grep -oE 'gfx[0-9]+' "$results_dir/rocminfo.txt" | sort -u | paste -sd, - || true)
  if [[ -z $observed_arches ]]; then
    observed_arches=none
  fi
  {
    printf 'revision=%s\n' "$fingerprint"
    printf 'HIP_VISIBLE_DEVICES=%s\n' "$hip_visible_devices"
    printf 'ROCM_PATH=%s\n' "$rocm_path"
    printf 'rental_linker=%s\n' "$rental_linker"
    if [[ -z ${ULLM_RENTAL_ENCODED_RUSTFLAGS:-} ]]; then
      printf 'rental_rustflags=cleared (overrides .cargo/config.toml)\n'
    else
      printf 'rental_rustflags=custom ULLM_RENTAL_ENCODED_RUSTFLAGS\n'
    fi
    printf 'rocminfo_arches=%s\n' "$observed_arches"
    "$rocm_path/bin/hipcc" --version
    cargo --version
    rustc --version
    optional_tool_status clang
    optional_tool_status mold
    optional_tool_status rustup
  } >"$results_dir/environment.txt"
  cat "$results_dir/environment.txt"
  rental_cargo metadata --format-version 1 --no-deps >/dev/null
  if ! grep -q 'gfx942' "$results_dir/rocminfo.txt"; then
    if (( rehearsal_no_gfx942 )); then
      printf '%s\n' "$observed_arches" >"$state_dir/gfx942-missing.rehearsal"
      printf 'REHEARSAL: rocminfo did not report required gfx942 (observed: %s); continuing only so physical can fail closed without touching a GPU. This cannot pass P0.\n' \
        "$observed_arches"
      return 0
    fi
    printf 'rocminfo did not report required gfx942 (observed: %s); this host cannot run the physical P0 gate.\n' \
      "$observed_arches" >&2
    return 1
  fi
}

stage_cpu() {
  cd -- "$repo_dir"
  rental_cargo test -p ullm-engine b_control_hipblas_layout_oracle_reproduces_the_mi300x_tail_delta \
    --lib -- --nocapture
  rental_cargo test -p ullm-engine sq8_gfx942_aprime --lib
  rental_cargo test -p ullm-engine --test sq8_fnuz_prepack
  rental_cargo test -p ullm-runtime-sys sq8_ck_gfx942_aprime_tests
}

stage_hiprtc() {
  cd -- "$repo_dir"
  local audit_binary="$hiprtc_dir/sq8-cdna3-hiprtc-audit"
  c++ -std=c++20 -O2 -Iruntime/include -Iruntime/src \
    tools/sq8-cdna3-hiprtc-audit.cpp -ldl -o "$audit_binary"
  HIP_VISIBLE_DEVICES=-1 ROCM_PATH="$rocm_path" "$audit_binary" --arch gfx942
}

stage_build() {
  cd -- "$repo_dir"
  rental_cargo build --release -p ullm-engine --features rocm-ck-gfx942-aprime \
    --example sq8_gfx942_aprime_physical_smoke
  test -x "$smoke_binary"
}

stage_isa() {
  cd -- "$repo_dir"
  test -x "$smoke_binary"
  ROCM_PATH="$rocm_path" tools/audit-sq8-cdna3-gfx942-isa.sh \
    --binary "$smoke_binary" --output-dir "$audit_dir"
}

stage_physical() {
  cd -- "$repo_dir"
  local prerequisite
  for prerequisite in preflight cpu hiprtc build isa; do
    if [[ ! -f $state_dir/$prerequisite.done ]]; then
      if [[ $prerequisite == preflight && -f $results_dir/rocminfo.txt ]] && \
        ! grep -q 'gfx942' "$results_dir/rocminfo.txt"; then
        printf 'physical is blocked because preflight found no required gfx942 device; inspect %s/rocminfo.txt. This is a fail-closed non-success.\n' \
          "$results_dir" >&2
        return 1
      fi
      printf 'physical requires completed P0 stage %s; run --stage all to preserve the gate order\n' \
        "$prerequisite" >&2
      return 1
    fi
  done
  if [[ -f $state_dir/gfx942-missing.rehearsal ]]; then
    printf 'physical is deliberately blocked: --rehearsal-no-gfx942 observed no gfx942 (%s). This expected local failure is not a P0 pass.\n' \
      "$(<"$state_dir/gfx942-missing.rehearsal")" >&2
    return 1
  fi
  test -x "$smoke_binary"
  # Do not permit the historical skip switch to turn a green A′ result into a
  # false overall pass.  This is the only stage that touches the rented GPU.
  HIP_VISIBLE_DEVICES="$hip_visible_devices" \
    env -u ULLM_SMOKE_SKIP_B_CONTROL "$smoke_binary"
}

stage_hw_microbench() {
  cd -- "$repo_dir"
  local prerequisite
  for prerequisite in preflight cpu hiprtc build isa physical; do
    if [[ ! -f $state_dir/$prerequisite.done ]]; then
      printf 'hw_microbench requires completed P0 stage %s; it is intentionally non-P0 and cannot run after a P0 failure.\n' \
        "$prerequisite" >&2
      return 1
    fi
  done
  : "${HW_MB_MEMORY_PEAK_GBPS:?set from recorded MI300X official/amd-smi source}"
  : "${HW_MB_BF16_PEAK_TFLOPS:?set from recorded MI300X official source}"
  : "${HW_MB_FP8_PEAK_TFLOPS:?set from recorded MI300X official source}"
  : "${HW_MB_AMD_SMI_GPU:?set to the physical amd-smi GPU index for the selected MI300X}"
  HIP_VISIBLE_DEVICES="$hip_visible_devices" \
    HW_MB_MEMORY_PEAK_GBPS="$HW_MB_MEMORY_PEAK_GBPS" \
    HW_MB_BF16_PEAK_TFLOPS="$HW_MB_BF16_PEAK_TFLOPS" \
    HW_MB_FP8_PEAK_TFLOPS="$HW_MB_FP8_PEAK_TFLOPS" \
    HW_MB_AMD_SMI_GPU="$HW_MB_AMD_SMI_GPU" \
    tools/run-hw-microbench-rdna4-cdna3.sh --arch gfx942 \
      --amd-smi-gpu "$HW_MB_AMD_SMI_GPU" \
      --results-dir "$results_dir/hw-microbench"
}

run_named_stage() {
  local name=$1
  case "$name" in
    preflight) run_step preflight stage_preflight ;;
    cpu) run_step cpu stage_cpu ;;
    hiprtc) run_step hiprtc stage_hiprtc ;;
    build) run_step build stage_build ;;
    isa) run_step isa stage_isa ;;
    physical) run_step physical stage_physical ;;
    hw_microbench) run_step hw_microbench stage_hw_microbench ;;
  esac
}

if [[ $requested_stage == all ]]; then
  for stage in "${stages[@]}"; do
    run_named_stage "$stage"
  done
  printf 'PASS P0 A′/B rental gate; logs and resume state: %s\n' "$results_dir"
  printf 'UNCONFIRMED by this runner: full-model gfx942 integration, runtime occupancy/residency, and hand-written A.\n'
else
  run_named_stage "$requested_stage"
  if [[ $requested_stage == hw_microbench ]]; then
    printf 'PASS optional non-P0 hardware microbenchmark; logs and resume state: %s\n' "$results_dir"
  else
    printf 'PASS requested P0 validation stage; logs and resume state: %s\n' "$results_dir"
  fi
fi
