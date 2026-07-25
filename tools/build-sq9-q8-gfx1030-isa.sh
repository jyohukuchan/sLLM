#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0
#
# Build and disassemble offline-only SQ9_0/Q8_0 ISA probes. This script never
# launches a HIP executable or selects a GPU; hipcc only produces a gfx1030
# code object for static inspection.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/output-directory" >&2
  exit 2
fi

output_dir=$1
if [[ $output_dir != /* ]]; then
  echo "output directory must be absolute" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "$0")" && pwd -P)
hipcc_bin=/usr/bin/hipcc
llvm_bin=/opt/rocm-7.2.1/lib/llvm/bin
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
bundle="$output_dir/sq9-q8-gfx1030.bundle"
code_object="$output_dir/sq9-q8-gfx1030.hsaco"
disassembly="$output_dir/sq9-q8-gfx1030.disasm"
notes="$output_dir/sq9-q8-gfx1030.notes.txt"
counts="$output_dir/isa-counts.json"

build_command=(
  "$hipcc_bin" -O3 -std=c++17 --offload-arch=gfx1030 --offload-device-only
  "$script_dir/sq9-q8-gfx1030-isa.hip.cpp" -o "$bundle"
)
printf '%q' "${build_command[0]}" >"$output_dir/build-command.txt"
printf ' %q' "${build_command[@]:1}" >>"$output_dir/build-command.txt"
printf '\n' >>"$output_dir/build-command.txt"

"$hipcc_bin" -O3 -std=c++17 --offload-arch=gfx1030 --offload-device-only \
  "$script_dir/sq9-q8-gfx1030-isa.hip.cpp" -o "$bundle"
"$bundler" --unbundle --type=o --input="$bundle" \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx1030 --output="$code_object"
"$objdump" --disassemble --mcpu=gfx1030 "$code_object" >"$disassembly"
"$readelf" --notes "$code_object" | sed '${/^$/d;}' >"$notes"
"$hipcc_bin" --version >"$output_dir/compiler-version.txt"
"$objdump" --version >>"$output_dir/compiler-version.txt"
sha256sum "$script_dir/sq9-q8-gfx1030-isa.hip.cpp" \
  "$script_dir/analyze-sq9-q8-gfx1030-isa.py" "$bundle" "$code_object" \
  "$disassembly" "$notes" >"$output_dir/SHA256SUMS"
python3 "$script_dir/analyze-sq9-q8-gfx1030-isa.py" \
  --disassembly "$disassembly" --notes "$notes" --output "$counts"
