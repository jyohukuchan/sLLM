#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0
#
# Compile the exact SQ8_1 HIPRTC source as a static HIP translation unit and
# audit both reference kernels without selecting a GPU.

set -euo pipefail

if [[ $# -ne 1 || $1 != /* ]]; then
  echo "usage: $0 /absolute/output-directory" >&2
  exit 2
fi

output_dir=$1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
llvm_bin=/opt/rocm-7.2.1/lib/llvm/bin
hipcc_bin=/usr/bin/hipcc
bundler="$llvm_bin/clang-offload-bundler"
objdump="$llvm_bin/llvm-objdump"
readelf="$llvm_bin/llvm-readelf"
runtime_source="$repo_root/runtime/src/kernels/sq8_1/sq8_1_matvec_hiprtc.inc"

for command in "$hipcc_bin" "$bundler" "$objdump" "$readelf" python3; do
  if [[ ! -x $command ]] && ! command -v "$command" >/dev/null 2>&1; then
    echo "required command is unavailable: $command" >&2
    exit 1
  fi
done

mkdir -p -- "$output_dir"
extracted="$output_dir/sq8_1_matvec_hiprtc_static.hip.cpp"
python3 "$script_dir/extract-sq8_1-hiprtc-source.py" \
  --runtime-source "$runtime_source" --output "$extracted" --allow-existing-identical
for arch in gfx1030 gfx1100 gfx1201 gfx942 gfx950; do
  bundle="$output_dir/sq8_1-reference-$arch.bundle"
  code_object="$output_dir/sq8_1-reference-$arch.hsaco"
  disassembly="$output_dir/sq8_1-reference-$arch.disasm"
  notes="$output_dir/sq8_1-reference-$arch.notes.txt"
  "$hipcc_bin" -O3 -std=c++17 --offload-arch="$arch" --offload-device-only \
    "$extracted" -o "$bundle"
  "$bundler" --unbundle --type=o --input="$bundle" \
    --targets="hipv4-amdgcn-amd-amdhsa--$arch" --output="$code_object"
  "$objdump" --disassemble --mcpu="$arch" "$code_object" >"$disassembly"
  "$readelf" --notes "$code_object" | sed '${/^$/d;}' >"$notes"
  python3 "$script_dir/analyze-sq8_1-dot4-isa.py" \
    --arch "$arch" --kernel ullm_sq8_1_matvec_w8a16_f32_kernel \
    --disassembly "$disassembly" --notes "$notes" \
    --output "$output_dir/sq8_1-reference-w8a16-$arch.json"
  python3 "$script_dir/analyze-sq8_1-dot4-isa.py" \
    --arch "$arch" --kernel ullm_sq8_1_matvec_w8a8_explicit_f32_kernel --require-signed-dot \
    --disassembly "$disassembly" --notes "$notes" \
    --output "$output_dir/sq8_1-reference-w8a8-$arch.json"
done
"$hipcc_bin" --version >"$output_dir/compiler-version.txt"
"$objdump" --version >>"$output_dir/compiler-version.txt"
{
  sha256sum "$runtime_source" "$script_dir/extract-sq8_1-hiprtc-source.py" \
    "$script_dir/analyze-sq8_1-dot4-isa.py" "$script_dir/build-sq8_1-reference-isa.sh"
  find "$output_dir" -maxdepth 1 -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum
} >"$output_dir/SHA256SUMS"
