#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0
#
# Extract and inspect the gfx942 code objects embedded in the A′ physical-smoke
# binary.  This is a static audit only: it never opens or launches a GPU.

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: audit-sq8-cdna3-gfx942-isa.sh --binary /absolute/path/to/smoke --output-dir /absolute/output

The binary must have been built with rocm-ck-gfx942-aprime and GPU_ARCH=gfx942.
ROCM_PATH defaults to /opt/rocm.
EOF
}

binary=
output_dir=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      binary=$2
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      output_dir=$2
      shift 2
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

if [[ $binary != /* || $output_dir != /* || ! -f $binary ]]; then
  usage
  exit 2
fi

rocm_path=${ROCM_PATH:-/opt/rocm}
llvm_bin="$rocm_path/llvm/bin"
objcopy="$llvm_bin/llvm-objcopy"
objdump="$llvm_bin/llvm-objdump"
readelf="$llvm_bin/llvm-readelf"
bundler="$llvm_bin/clang-offload-bundler"
for command in "$objcopy" "$objdump" "$readelf" "$bundler" zstd dd od grep sha256sum awk sed head; do
  if ! command -v "$command" >/dev/null 2>&1 && [[ ! -x $command ]]; then
    printf 'required command is unavailable: %s\n' "$command" >&2
    exit 1
  fi
done

mkdir -p -- "$output_dir"
fatbin="$output_dir/smoke.hip_fatbin"
summary="$output_dir/summary.txt"
resource_tsv="$output_dir/gfx942-ck-resource-metadata.tsv"
"$objcopy" --dump-section .hip_fatbin="$fatbin" "$binary"

# ROCWMMA publishes the per-workgroup LDS limit used for gfx942 compilation.
# Keep the fallback explicit so an older installation still leaves a readable
# audit trail rather than silently accepting an arbitrary value.
lds_limit=65536
rocwmma_constants="$rocm_path/include/rocwmma/internal/constants.hpp"
if [[ -f $rocwmma_constants ]]; then
  parsed_lds_limit=$(sed -n \
    's/.*AMDGCN_LDS_MAX_SIZE_BYTES[[:space:]]*=[[:space:]]*\([0-9][0-9]*\)u.*/\1/p' \
    "$rocwmma_constants" | head -n 1)
  if [[ $parsed_lds_limit =~ ^[1-9][0-9]*$ ]]; then
    lds_limit=$parsed_lds_limit
  fi
fi
printf 'code_object\tsymbol\tvgpr\tsgpr\tagpr\tlds_bytes\tprivate_bytes\tvgpr_spills\tsgpr_spills\tmax_workgroup\n' >"$resource_tsv"

required_contracts=(
  'DeviceGemmXdlUniversal<Default, RCR> BlkSize: 256, BlkTile: 16x128x128'
  'DeviceGemmXdlUniversal<KPadding, RCR> BlkSize: 256, BlkTile: 16x128x256'
  'DeviceGemmXdlUniversal<Default, RCR> BlkSize: 256, BlkTile: 16x256x128'
  'DeviceGemmXdlUniversal<Default, RCR> BlkSize: 256, BlkTile: 16x128x256'
)
for contract in "${required_contracts[@]}"; do
  if ! grep -aFq "$contract" "$binary"; then
    printf 'the linked binary does not contain required A′ contract: %s\n' "$contract" >&2
    exit 1
  fi
done

mapfile -t ccob_offsets < <(LC_ALL=C grep -abo 'CCOB' "$fatbin" | cut -d: -f1)
if [[ ${#ccob_offsets[@]} -eq 0 ]]; then
  printf 'no CCOB payload was found in %s\n' "$fatbin" >&2
  exit 1
fi

{
  printf 'binary_sha256='
  sha256sum "$binary" | awk '{print $1}'
  printf 'selected_instance_contracts=%s\n' "${#required_contracts[@]}"
  printf 'ccob_count=%s\n' "${#ccob_offsets[@]}"
} >"$summary"

total_mfma=0
gfx942_objects=0
for index in "${!ccob_offsets[@]}"; do
  offset=${ccob_offsets[$index]}
  length=$(od -An -j "$((offset + 8))" -N 8 -tu8 "$fatbin" | tr -d '[:space:]')
  if [[ ! $length =~ ^[0-9]+$ ]] || (( length <= 32 )); then
    printf 'invalid CCOB length at offset %s: %s\n' "$offset" "$length" >&2
    exit 1
  fi
  payload="$output_dir/ccob-${index}.zst"
  bundle="$output_dir/ccob-${index}.bundle"
  targets="$output_dir/ccob-${index}.targets.txt"
  dd if="$fatbin" of="$payload" bs=1 skip="$((offset + 32))" count="$((length - 32))" status=none
  zstd -q -d -f -o "$bundle" "$payload"
  "$bundler" --list --type=o --input="$bundle" >"$targets"
  if ! grep -Fxq 'hipv4-amdgcn-amd-amdhsa--gfx942' "$targets"; then
    printf 'CCOB %s does not contain gfx942; retained its target list for diagnosis\n' "$index" >&2
    continue
  fi

  hsaco="$output_dir/ccob-${index}.gfx942.hsaco"
  disassembly="$output_dir/ccob-${index}.gfx942.disasm"
  notes="$output_dir/ccob-${index}.gfx942.notes.txt"
  "$bundler" --unbundle --type=o --input="$bundle" \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx942 --output="$hsaco"
  "$objdump" --disassemble --mcpu=gfx942 "$hsaco" >"$disassembly"
  "$readelf" --notes "$hsaco" >"$notes"
  # The CCOB may contain several CK tail/specialization kernels.  Preserve all
  # of their AMDGPU metadata rather than assigning a guessed resource tuple to
  # a selected GetTypeString.  Selection is separately asserted above.
  awk -v code_object="ccob-$index" '
    function emit() {
      if (seen && name ~ /kernel_gemm_xdl/) {
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
          code_object, symbol, vgpr, sgpr, agpr, lds, private_bytes, vspill, sspill, wgmax
      }
    }
    /^  - \.agpr_count:/ {
      emit()
      seen = 1
      agpr = $3
      name = symbol = ""
      vgpr = sgpr = lds = private_bytes = vspill = sspill = wgmax = ""
      next
    }
    /^    \.name:/ { name = $2; next }
    /^    \.symbol:/ { symbol = $2; next }
    /^    \.group_segment_fixed_size:/ { lds = $2; next }
    /^    \.private_segment_fixed_size:/ { private_bytes = $2; next }
    /^    \.sgpr_count:/ { sgpr = $2; next }
    /^    \.sgpr_spill_count:/ { sspill = $2; next }
    /^    \.vgpr_count:/ { vgpr = $2; next }
    /^    \.vgpr_spill_count:/ { vspill = $2; next }
    /^    \.max_flat_workgroup_size:/ { wgmax = $2; next }
    END { emit() }
  ' "$notes" >>"$resource_tsv"
  mfma_count=$(grep -c 'v_mfma_f32_16x16x32_fp8_fp8' "$disassembly" || true)
  total_mfma=$((total_mfma + mfma_count))
  gfx942_objects=$((gfx942_objects + 1))
  {
    printf 'ccob_%s_offset=%s\n' "$index" "$offset"
    printf 'ccob_%s_gfx942_mfma_16x16x32_fp8_fp8=%s\n' "$index" "$mfma_count"
    printf 'ccob_%s_notes=%s\n' "$index" "$(basename "$notes")"
  } >>"$summary"
done

{
  printf 'gfx942_code_objects=%s\n' "$gfx942_objects"
  printf 'total_mfma_16x16x32_fp8_fp8=%s\n' "$total_mfma"
  printf 'occupancy=unconfirmed_without_hipModuleOccupancy_query_on_gfx942\n'
} >>"$summary"

if (( gfx942_objects == 0 || total_mfma == 0 )); then
  printf 'required gfx942 FP8 MFMA ISA evidence is absent; see %s\n' "$summary" >&2
  exit 1
fi

if ! resource_summary=$(awk -F '\t' -v lds_limit="$lds_limit" '
  NR == 1 { next }
  {
    count += 1
    if ($3 + 0 > max_vgpr) max_vgpr = $3 + 0
    if ($4 + 0 > max_sgpr) max_sgpr = $4 + 0
    if ($5 + 0 > max_agpr) max_agpr = $5 + 0
    if ($6 + 0 > max_lds) max_lds = $6 + 0
    if ($7 + 0 > max_private) max_private = $7 + 0
    if ($8 + 0 > max_vspill) max_vspill = $8 + 0
    if ($9 + 0 > max_sspill) max_sspill = $9 + 0
    if ($6 + 0 > lds_limit || $7 + 0 != 0 || $8 + 0 != 0 || $9 + 0 != 0) bad += 1
  }
  END {
    if (count == 0) {
      print "no CK GEMM resource rows were parsed" > "/dev/stderr"
      exit 1
    }
    printf "ck_gemm_kernel_entries=%d\n", count
    printf "ck_gemm_vgpr_max=%d\n", max_vgpr
    printf "ck_gemm_sgpr_max=%d\n", max_sgpr
    printf "ck_gemm_agpr_max=%d\n", max_agpr
    printf "ck_gemm_lds_max_bytes=%d\n", max_lds
    printf "ck_gemm_lds_limit_bytes=%d\n", lds_limit
    printf "ck_gemm_private_max_bytes=%d\n", max_private
    printf "ck_gemm_vgpr_spills_max=%d\n", max_vspill
    printf "ck_gemm_sgpr_spills_max=%d\n", max_sspill
    printf "ck_gemm_static_single_workgroup_fit=%s\n", (bad == 0 ? "pass" : "fail")
    exit (bad == 0 ? 0 : 1)
  }
' "$resource_tsv"); then
  printf 'static CK resource audit failed; see %s\n' "$resource_tsv" >&2
  exit 1
fi
printf '%s\n' "$resource_summary" >>"$summary"

printf 'PASS static gfx942 ISA audit: %s MFMA instructions across %s code object(s)\n' \
  "$total_mfma" "$gfx942_objects"
printf 'evidence: %s\n' "$summary"
