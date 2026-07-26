#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0
#
# Compile the exact SQ8_0 HIPRTC matvec source as an isolated device-only
# translation unit and retain ISA/resource evidence for gfx1201 and gfx1030.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || $1 != /* ]]; then
  echo "usage: $0 /absolute/output-directory [runtime-source]" >&2
  exit 2
fi

output_dir=$1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
runtime_source=${2:-"$repo_root/runtime/src/kernels/sq8_0/sq8_0_matvec_hiprtc.inc"}
llvm_bin=/opt/rocm-7.2.1/lib/llvm/bin
hipcc_bin=/usr/bin/hipcc
clang_bin="$llvm_bin/clang++"
bundler="$llvm_bin/clang-offload-bundler"
objdump="$llvm_bin/llvm-objdump"
readelf="$llvm_bin/llvm-readelf"
extractor="$script_dir/extract-sq8_0-hiprtc-source.py"
analyzer="$script_dir/analyze-sq8-decode-matvec-isa.py"

for command in "$hipcc_bin" "$clang_bin" "$bundler" "$objdump" "$readelf" python3; do
  if [[ ! -x $command ]] && ! command -v "$command" >/dev/null 2>&1; then
    echo "required command is unavailable: $command" >&2
    exit 1
  fi
done
if [[ ! -r $runtime_source ]]; then
  echo "runtime source is unreadable: $runtime_source" >&2
  exit 1
fi

mkdir -p -- "$output_dir"
extracted="$output_dir/sq8_0_matvec_hiprtc_static.hip.cpp"
python3 "$extractor" \
  --runtime-source "$runtime_source" --output "$extracted" --allow-existing-identical

for arch in gfx1201 gfx1030; do
  bundle="$output_dir/sq8_0-matvec-$arch.bundle"
  code_object="$output_dir/sq8_0-matvec-$arch.hsaco"
  disassembly="$output_dir/sq8_0-matvec-$arch.disasm"
  notes="$output_dir/sq8_0-matvec-$arch.notes.txt"
  "$hipcc_bin" -O3 -std=c++17 --offload-arch="$arch" --offload-device-only \
    "$extracted" -o "$bundle"
  "$bundler" --unbundle --type=o --input="$bundle" \
    --targets="hipv4-amdgcn-amd-amdhsa--$arch" --output="$code_object"
  "$objdump" --disassemble --mcpu="$arch" "$code_object" >"$disassembly"
  "$readelf" --notes "$code_object" | sed '${/^$/d;}' >"$notes"
  "$clang_bin" -x hip --cuda-device-only --offload-arch="$arch" -O3 -std=c++17 \
    -S -emit-llvm "$extracted" -o "$output_dir/sq8_0-matvec-$arch.ll"
  python3 "$analyzer" --arch "$arch" --notes "$notes" \
    --disassembly "$disassembly" --output "$output_dir/sq8_0-matvec-$arch.summary.json"
done

"$hipcc_bin" --version >"$output_dir/compiler-version.txt"
"$objdump" --version >>"$output_dir/compiler-version.txt"
{
  sha256sum "$runtime_source" "$extractor" "$analyzer" "$script_dir/build-sq8-decode-matvec-isa.sh"
  find "$output_dir" -maxdepth 1 -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum
} >"$output_dir/SHA256SUMS"
