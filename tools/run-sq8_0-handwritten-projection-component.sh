#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0

# Builds/runs only the isolated SQ8_0 handwritten M=1 feasibility component.
# It has no relationship to the public runtime ABI or serving dispatcher.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rocm_path="${ROCM_PATH:-/opt/rocm-7.2.1}"
gpu_arch="${GPU_ARCH:-gfx1201}"
output="${ULLM_SQ8_HANDWRITTEN_COMPONENT_BIN:-/tmp/ullm-sq8_0-handwritten-projection/bench}"

if [[ "${gpu_arch}" != "gfx1201" ]]; then
  printf '%s\n' 'run-sq8_0-handwritten-projection-component.sh requires GPU_ARCH=gfx1201' >&2
  exit 2
fi

mkdir -p "$(dirname "${output}")"

"${rocm_path}/bin/hipcc" \
  -std=c++20 \
  -O3 \
  -ffunction-sections \
  -fdata-sections \
  -DCK_USE_OCP_FP8=1 \
  -DCK_ENABLE_FP8=1 \
  -DCK_ENABLE_BF16=1 \
  --offload-arch="${gpu_arch}" \
  -I"${repo_root}/runtime/src" \
  -I"${rocm_path}/include" \
  "${repo_root}/tools/bench-sq8_0-handwritten-projection.cpp" \
  "${repo_root}/runtime/src/sq8_ck_gfx1201.hip.cpp" \
  "${repo_root}/runtime/src/sq8_handwritten_gfx1201.hip.cpp" \
  -L"${rocm_path}/lib" \
  -ldevice_gemm_operations \
  -lamdhip64 \
  -Wl,--gc-sections \
  -o "${output}"

if [[ "${1:-}" == "--build-only" ]]; then
  exit 0
fi

visible_device="${ULLM_R9700_HIP_VISIBLE_DEVICE:-${HIP_VISIBLE_DEVICES:-}}"
if [[ -z "${visible_device}" || "${visible_device}" == *,* ]]; then
  printf '%s\n' \
    'run-sq8_0-handwritten-projection-component.sh requires exactly one explicit HIP visibility token' >&2
  exit 2
fi
export HIP_VISIBLE_DEVICES="${visible_device}"
exec "${output}" "$@" --device 0
