#!/usr/bin/env bash
# Rebuild the AQ4_0 matvec-add candidate against BZ's grouped-split source baseline.
# This is intentionally CPU-only; the resulting artifact remains under /tmp until a later,
# separately locked R9700 qualification window accepts it.
set -Eeuo pipefail

repo_root="/home/homelab1/coding-local/ultimateLLM/uLLM-project"
results_root="$repo_root/benchmarks/results/2026-07-27/aq4-matvec-add-optimization"
source_root="/tmp/ullm-aq4-add-grouped-source-20260727T0416"
target_root="/tmp/ullm-aq4-add-grouped-target-20260727T0416"
provenance="$results_root/candidate-grouped-build-provenance.json"
expected_base="9d8643506a36659ecec3fc2d931deba26d29f574"
expected_dirty=$' M crates/ullm-runtime-sys/src/test_parts/aq4_matvec_add_wide_load_prototype.rs\n M runtime/src/ullm_runtime_hiprtc_sources.inc'
build_jobs="${ULLM_AQ4_ADD_BUILD_JOBS:-4}"

if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "ULLM_AQ4_ADD_BUILD_JOBS must be a positive integer" >&2
    exit 64
fi
if [[ -e "$target_root" || -e "$provenance" ]]; then
    echo "refusing to overwrite existing candidate target or provenance" >&2
    exit 65
fi
if [[ ! -d "$source_root/.git" && ! -f "$source_root/.git" ]]; then
    echo "missing isolated grouped candidate source worktree" >&2
    exit 65
fi
if [[ "$(git -C "$source_root" rev-parse HEAD)" != "$expected_base" ||
    "$(git -C "$source_root" status --porcelain)" != "$expected_dirty" ]]; then
    echo "candidate source worktree is not the expected grouped baseline plus candidate patch" >&2
    exit 65
fi
git -C "$source_root" diff --check

runtime_source="$source_root/runtime/src/ullm_runtime_hiprtc_sources.inc"
test_source="$source_root/crates/ullm-runtime-sys/src/test_parts/aq4_matvec_add_wide_load_prototype.rs"
source_patch="$(git -C "$source_root" diff -- \
    runtime/src/ullm_runtime_hiprtc_sources.inc \
    crates/ullm-runtime-sys/src/test_parts/aq4_matvec_add_wide_load_prototype.rs | sha256sum | awk '{print $1}')"

cd "$source_root"
CARGO_TARGET_DIR="$target_root" CARGO_BUILD_JOBS="$build_jobs" \
    cargo build --release -p ullm-engine \
    --bin ullm-aq4-worker \
    --bin ullm-aq4-decode-step-profile \
    --bin ullm-aq4-e2e-prefill-timing
CARGO_TARGET_DIR="$target_root" CARGO_BUILD_JOBS="$build_jobs" \
    cargo test -p ullm-runtime-sys --no-run

worker="$target_root/release/ullm-aq4-worker"
decode_profile="$target_root/release/ullm-aq4-decode-step-profile"
prefill_profile="$target_root/release/ullm-aq4-e2e-prefill-timing"
mapfile -t test_bins < <(find "$target_root/debug/deps" -maxdepth 1 -type f -executable \
    -name 'ullm_runtime_sys-*' -print | sort)
if [[ ! -x "$worker" || ! -x "$decode_profile" || ! -x "$prefill_profile" ||
    ${#test_bins[@]} -ne 1 ]]; then
    echo "expected worker/profile/test artifacts were not built" >&2
    exit 65
fi
test_binary="${test_bins[0]}"

runtime_blob="$(git -C "$source_root" rev-parse "$expected_base:runtime/src/ullm_runtime_hiprtc_sources.inc")"
part01_blob="$(git -C "$source_root" rev-parse "$expected_base:runtime/src/ullm_runtime_parts/part_01.inc")"
jq -n \
    --arg source_root "$source_root" \
    --arg target_root "$target_root" \
    --arg base_commit "$expected_base" \
    --arg source_patch_sha256 "$source_patch" \
    --arg runtime_source_hash "$(sha256sum "$runtime_source" | awk '{print $1}')" \
    --arg test_source_hash "$(sha256sum "$test_source" | awk '{print $1}')" \
    --arg worker "$worker" \
    --arg worker_sha256 "$(sha256sum "$worker" | awk '{print $1}')" \
    --arg decode_profile "$decode_profile" \
    --arg decode_profile_sha256 "$(sha256sum "$decode_profile" | awk '{print $1}')" \
    --arg prefill_profile "$prefill_profile" \
    --arg prefill_profile_sha256 "$(sha256sum "$prefill_profile" | awk '{print $1}')" \
    --arg test_binary "$test_binary" \
    --arg test_binary_sha256 "$(sha256sum "$test_binary" | awk '{print $1}')" \
    --arg runtime_blob "$runtime_blob" \
    --arg part01_blob "$part01_blob" \
    '{
      schema_version: "ullm.aq4_matvec_add.grouped_candidate_build.v1",
      source_root: $source_root,
      base_commit: $base_commit,
      source_patch_sha256: $source_patch_sha256,
      source_inputs: {
        "runtime/src/ullm_runtime_hiprtc_sources.inc": $runtime_source_hash,
        "crates/ullm-runtime-sys/src/test_parts/aq4_matvec_add_wide_load_prototype.rs": $test_source_hash
      },
      grouped_baseline_equivalence: {
        active_recorded_source_commit: "c8074928e22b27801df78d65508fdd619d13a748",
        runtime_hiprtc_blob: $runtime_blob,
        runtime_part01_blob: $part01_blob,
        status: "identical blobs verified before candidate patch"
      },
      staging_directory: $source_root,
      artifact_target_directory: $target_root,
      promotion_status: "staging_only_not_a_served_artifact",
      measurement_status: "ready_for_locked_r9700_window",
      artifacts: {
        worker: {path: $worker, sha256: $worker_sha256},
        decode_profile: {path: $decode_profile, sha256: $decode_profile_sha256},
        prefill_profile: {path: $prefill_profile, sha256: $prefill_profile_sha256},
        direct_gpu_test: {path: $test_binary, sha256: $test_binary_sha256}
      }
    }' >"$provenance"

sha256sum "$runtime_source" "$test_source" "$worker" "$decode_profile" "$prefill_profile" \
    "$test_binary" "$provenance" >"$results_root/candidate-grouped-build-sha256s.txt"
printf 'candidate grouped build ready: %s\n' "$provenance"
