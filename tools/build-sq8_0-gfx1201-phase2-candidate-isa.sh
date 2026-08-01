#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0
#
# Build only the isolated SQ8_0 gfx1201 Phase 2 prototype and retain ISA.

set -euo pipefail

if [[ $# -ne 1 || $1 != /* ]]; then
  echo "usage: $0 /absolute/output-directory" >&2
  exit 2
fi

output_dir=$1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
source_file="$script_dir/sq8_0_gfx1201_phase2_candidate.hip.cpp"
llvm_bin=/opt/rocm-7.2.1/lib/llvm/bin
hipcc_bin=/usr/bin/hipcc
bundler="$llvm_bin/clang-offload-bundler"
objdump="$llvm_bin/llvm-objdump"
readelf="$llvm_bin/llvm-readelf"
analyzer="$script_dir/analyze-sq8-decode-matvec-isa.py"

for command in "$hipcc_bin" "$bundler" "$objdump" "$readelf" python3; do
  if [[ ! -x $command ]] && ! command -v "$command" >/dev/null 2>&1; then
    echo "required command is unavailable: $command" >&2
    exit 1
  fi
done

mkdir -p -- "$output_dir"
bundle="$output_dir/sq8_0-gfx1201-phase2.bundle"
code_object="$output_dir/sq8_0-gfx1201-phase2.hsaco"
disassembly="$output_dir/sq8_0-gfx1201-phase2.disasm"
notes="$output_dir/sq8_0-gfx1201-phase2.notes.txt"

"$hipcc_bin" -O3 -std=c++17 --offload-arch=gfx1201 --offload-device-only \
  "$source_file" -o "$bundle"
"$bundler" --unbundle --type=o --input="$bundle" \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx1201 --output="$code_object"
"$objdump" --disassemble --mcpu=gfx1201 "$code_object" >"$disassembly"
"$readelf" --notes "$code_object" | sed '${/^$/d;}' >"$notes"
python3 "$analyzer" --arch gfx1201 --notes "$notes" --disassembly "$disassembly" \
  --output "$output_dir/sq8_0-gfx1201-phase2.summary.json"
"$hipcc_bin" --version >"$output_dir/compiler-version.txt"
{
  sha256sum "$source_file" "$analyzer" "$script_dir/build-sq8_0-gfx1201-phase2-candidate-isa.sh"
  find "$output_dir" -maxdepth 1 -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum
} >"$output_dir/SHA256SUMS"
