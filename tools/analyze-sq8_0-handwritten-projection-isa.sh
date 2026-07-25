#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  printf '%s\n' "usage: $0 HIP_OBJECT OUTPUT_DIRECTORY" >&2
  exit 2
fi

object="$1"
output_dir="$2"
rocm_path="${ROCM_PATH:-/opt/rocm-7.2.1}"
mkdir -p "${output_dir}"

fatbin="${output_dir}/handwritten-fatbin.bin"
code_object="${output_dir}/handwritten-gfx1201.co"
notes="${output_dir}/handwritten-gfx1201.notes.txt"
disassembly="${output_dir}/handwritten-gfx1201.disassembly.txt"

"${rocm_path}/llvm/bin/llvm-objcopy" \
  --dump-section .hip_fatbin="${fatbin}" "${object}"
"${rocm_path}/llvm/bin/clang-offload-bundler" \
  -unbundle -type=o \
  -targets=hipv4-amdgcn-amd-amdhsa--gfx1201 \
  -input="${fatbin}" -output="${code_object}"
"${rocm_path}/llvm/bin/llvm-readelf" --notes "${code_object}" >"${notes}"
"${rocm_path}/llvm/bin/llvm-objdump" -d --mcpu=gfx1201 "${code_object}" >"${disassembly}"
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/analyze-sq8_0-handwritten-projection-isa.py" \
  --notes "${notes}" \
  --disassembly "${disassembly}" \
  --output "${output_dir}/handwritten-isa-summary.json"
