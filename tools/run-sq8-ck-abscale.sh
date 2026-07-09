#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rocm_path="${ROCM_PATH:-/opt/rocm-7.2.1}"
gpu_arch="${GPU_ARCH:-gfx1201}"
output="${ULLM_CK_ABSCALE_PROBE_BIN:-${repo_root}/build/tools/bench-sq8-ck-abscale}"

mkdir -p "$(dirname "${output}")"

"${rocm_path}/bin/hipcc" \
  -x hip \
  -std=c++20 \
  -O3 \
  -DCK_USE_OCP_FP8=1 \
  -DCK_ENABLE_FP8=1 \
  -DCK_ENABLE_BF16=1 \
  --offload-arch="${gpu_arch}" \
  --hip-link \
  -I"${rocm_path}/include" \
  "${repo_root}/tools/bench-sq8-ck-abscale.cpp" \
  -L"${rocm_path}/lib" \
  -ldevice_gemm_operations \
  -lamdhip64 \
  -pthread \
  -o "${output}"

if [[ "${1:-}" == "--build-only" ]]; then
  exit 0
fi

args=("$@")
has_device=0
for arg in "${args[@]}"; do
  if [[ "${arg}" == "--device" ]]; then
    has_device=1
    break
  fi
done

if [[ "${has_device}" == 0 ]]; then
  if [[ ! -v HIP_VISIBLE_DEVICES ]]; then
    export HIP_VISIBLE_DEVICES="${ULLM_CK_ABSCALE_DEVICE_SELECTOR:-1}"
  fi
  args=(--device 0 "${args[@]}")
fi

exec "${output}" "${args[@]}"
