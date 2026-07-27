#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0
# Static ISA/resource audit for tools/hw-microbench-rdna4-cdna3.hip.cpp.
set -euo pipefail

usage() { echo "usage: $0 --arch gfx1201|gfx942 --output-dir /absolute/dir [--repo /absolute/repo]" >&2; }
arch= output_dir= repo=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) arch=${2:-}; shift 2;; --output-dir) output_dir=${2:-}; shift 2;; --repo) repo=${2:-}; shift 2;;
    -h|--help) usage; exit 0;; *) usage; exit 2;; esac
done
[[ $arch == gfx1201 || $arch == gfx942 ]] || { usage; exit 2; }
[[ $output_dir == /* ]] || { usage; exit 2; }
if [[ -z $repo ]]; then repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P); fi
[[ $repo == /* && -f $repo/tools/hw-microbench-rdna4-cdna3.hip.cpp ]] || { usage; exit 2; }
rocm=${ROCM_PATH:-/opt/rocm}; hipcc=$rocm/bin/hipcc; objdump=$rocm/llvm/bin/llvm-objdump; readelf=$rocm/llvm/bin/llvm-readelf
[[ -x $hipcc && -x $objdump && -x $readelf ]] || { echo "ROCm compiler/audit tools unavailable" >&2; exit 1; }
mkdir -p -- "$output_dir/build"
binary="$output_dir/hw-microbench-$arch"
(cd -- "$output_dir/build" && "$hipcc" -std=c++20 -O3 --save-temps --offload-arch="$arch" "$repo/tools/hw-microbench-rdna4-cdna3.hip.cpp" -o "$binary") >"$output_dir/build.log" 2>&1
hsaco=$(find "$output_dir/build" -type f -name '*-hip-amdgcn-amd-amdhsa-'"$arch"'.out' -print -quit)
# hipcc writes --save-temps beside the cwd; compile in build explicitly above.
if [[ -z $hsaco ]]; then
  hsaco=$(find "$output_dir" -type f -name '*-hip-amdgcn-amd-amdhsa-'"$arch"'.out' -print -quit)
fi
[[ -n $hsaco && -f $hsaco ]] || { echo "did not find saved $arch hsaco" >&2; exit 1; }
"$objdump" --disassemble --mcpu="$arch" "$hsaco" >"$output_dir/$arch.disasm"
"$readelf" --notes "$hsaco" >"$output_dir/$arch.notes.txt"
if [[ $arch == gfx1201 ]]; then required='v_wmma_f32_16x16x16_fp8_fp8'; else required='v_mfma_f32_16x16x32_fp8_fp8'; fi
count=$(grep -c "$required" "$output_dir/$arch.disasm" || true)
[[ $count -gt 0 ]] || { echo "required instruction absent: $required" >&2; exit 1; }
printf 'kernel\tvgpr\tsgpr\tagpr\tlds_bytes\tprivate_bytes\tvgpr_spills\tsgpr_spills\twavefront\n' >"$output_dir/resources.tsv"
awk '
  function emit() { if (name ~ /bf16_gemm_kernel|fp8_gemm_kernel/) printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",name,vgpr,sgpr,agpr,lds,priv,vspill,sspill,wave }
  /^  - \.args:/ { emit(); name=""; vgpr=sgpr=lds=priv=vspill=sspill=wave=""; agpr="N/A"; next }
  /^  - \.agpr_count:/ { agpr=$3; next }
  /^    \.name:/ { name=$2; next } /^    \.group_segment_fixed_size:/ {lds=$2;next}
  /^    \.private_segment_fixed_size:/ {priv=$2;next} /^    \.sgpr_count:/ {sgpr=$2;next}
  /^    \.sgpr_spill_count:/ {sspill=$2;next} /^    \.vgpr_count:/ {vgpr=$2;next}
  /^    \.vgpr_spill_count:/ {vspill=$2;next} /^    \.wavefront_size:/ {wave=$2;next}
  END { emit() }
' "$output_dir/$arch.notes.txt" >>"$output_dir/resources.tsv"
rows=$(($(wc -l <"$output_dir/resources.tsv") - 1))
[[ $rows -ge 2 ]] || { echo "GEMM resource metadata absent" >&2; exit 1; }
if ! awk -F '\t' 'NR>1 { if ($6 != 0 || $7 != 0 || $8 != 0) bad=1 } END { exit bad }' "$output_dir/resources.tsv"; then
  echo "private memory or register spill detected; see resources.tsv" >&2; exit 1
fi
{
  echo "arch=$arch"; echo "hsaco=$hsaco"; echo "required_instruction=$required"; echo "required_instruction_count=$count"
  echo "gemm_resource_rows=$rows"; echo "static_occupancy=resource metadata recorded; runtime occupancy requires the target GPU"
} >"$output_dir/summary.txt"
echo "PASS $arch ISA audit: $required ($count); resources: $output_dir/resources.tsv"
