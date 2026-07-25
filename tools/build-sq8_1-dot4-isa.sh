#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0
#
# Build and inspect SQ8_1's signed-I8 dot baseline without selecting a GPU.

set -euo pipefail

if [[ $# -ne 1 || $1 != /* ]]; then
  echo "usage: $0 /absolute/output-directory" >&2
  exit 2
fi

output_dir=$1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
llvm_bin=/opt/rocm-7.2.1/lib/llvm/bin
hipcc_bin=/usr/bin/hipcc
bundler="$llvm_bin/clang-offload-bundler"
objdump="$llvm_bin/llvm-objdump"
readelf="$llvm_bin/llvm-readelf"

for command in "$hipcc_bin" "$bundler" "$objdump" "$readelf" python3; do
  if [[ ! -x $command ]] && ! command -v "$command" >/dev/null 2>&1; then
    echo "required command is unavailable: $command" >&2
    exit 1
  fi
done

mkdir -p -- "$output_dir"
for arch in gfx1030 gfx1100 gfx1201 gfx942 gfx950; do
  bundle="$output_dir/sq8_1-dot4-$arch.bundle"
  code_object="$output_dir/sq8_1-dot4-$arch.hsaco"
  disassembly="$output_dir/sq8_1-dot4-$arch.disasm"
  notes="$output_dir/sq8_1-dot4-$arch.notes.txt"
  analysis="$output_dir/sq8_1-dot4-$arch.json"
  "$hipcc_bin" -O3 -std=c++17 --offload-arch="$arch" --offload-device-only \
    "$script_dir/sq8_1-dot4-isa.hip.cpp" -o "$bundle"
  "$bundler" --unbundle --type=o --input="$bundle" \
    --targets="hipv4-amdgcn-amd-amdhsa--$arch" --output="$code_object"
  "$objdump" --disassemble --mcpu="$arch" "$code_object" >"$disassembly"
  "$readelf" --notes "$code_object" | sed '${/^$/d;}' >"$notes"
  python3 "$script_dir/analyze-sq8_1-dot4-isa.py" \
    --arch "$arch" --require-signed-dot --disassembly "$disassembly" --notes "$notes" --output "$analysis"
done
"$hipcc_bin" --version >"$output_dir/compiler-version.txt"
"$objdump" --version >>"$output_dir/compiler-version.txt"
{
  sha256sum "$script_dir/sq8_1-dot4-isa.hip.cpp" "$script_dir/analyze-sq8_1-dot4-isa.py" \
    "$script_dir/build-sq8_1-dot4-isa.sh"
  find "$output_dir" -maxdepth 1 -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum
} >"$output_dir/SHA256SUMS"
