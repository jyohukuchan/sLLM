#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rocm_path="${ROCM_PATH:-/opt/rocm-7.2.1}"
gpu_arch="${GPU_ARCH:-gfx1201}"
output="${ULLM_CK_COMPONENT_BIN:-${repo_root}/build/tools/bench-sq8-ck-component}"

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
  -I"${rocm_path}/include" \
  "${repo_root}/tools/bench-sq8-ck-component.cpp" \
  -L"${rocm_path}/lib" \
  -ldevice_gemm_operations \
  -lhiprtc \
  -lamdhip64 \
  -Wl,--gc-sections \
  -o "${output}"

if [[ "${1:-}" == "--build-only" ]]; then
  exit 0
fi

visible_device="${ULLM_R9700_HIP_VISIBLE_DEVICE:-${HIP_VISIBLE_DEVICES:-1}}"
if [[ -z "${visible_device}" || "${visible_device}" == *,* ]]; then
  printf '%s\n' \
    'run-sq8-ck-component.sh requires exactly one HIP visibility token' >&2
  exit 2
fi
export HIP_VISIBLE_DEVICES="${visible_device}"
export ULLM_CK_COMPONENT_VISIBLE_DEVICE_TOKEN="${visible_device}"

exec "${output}" "$@" --device 0
