#include <cstdint>
#include <hip/hip_runtime.h>

namespace sllm_gdn_projection_bundle_kernel {

__device__ __forceinline__ float bf16_to_float(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16U);
}

__device__ __forceinline__ uint16_t float_to_bf16_rne(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t round = UINT32_C(0x7fff) + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>((bits + round) >> 16U);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_gdn_projection_bundle_bf16_fp32_decode_v1(
    const uint16_t *activation, const uint16_t *qkv_weight,
    const uint16_t *z_weight, const uint16_t *b_weight,
    const uint16_t *a_weight, uint16_t *qkv_output, uint16_t *z_output,
    uint16_t *b_output, uint16_t *a_output, uint64_t k) {
  constexpr uint64_t qkv_n = 8192U;
  constexpr uint64_t z_n = 4096U;
  constexpr uint64_t b_n = 32U;
  const uint64_t index = static_cast<uint64_t>(blockIdx.x);
  const uint16_t *weight = nullptr;
  uint16_t *output = nullptr;
  uint64_t column = index;
  uint64_t n = qkv_n;
  if (index < qkv_n) {
    weight = qkv_weight;
    output = qkv_output;
  } else if (index < qkv_n + z_n) {
    column = index - qkv_n;
    weight = z_weight;
    output = z_output;
    n = z_n;
  } else if (index < qkv_n + z_n + b_n) {
    column = index - qkv_n - z_n;
    weight = b_weight;
    output = b_output;
    n = b_n;
  } else {
    column = index - qkv_n - z_n - b_n;
    weight = a_weight;
    output = a_output;
    n = b_n;
  }
  if (column >= n)
    return;
  float partial = 0.0F;
  const uint16_t *weight_row = weight + column * k;
  const bool paired =
      (k & 1U) == 0U && ((reinterpret_cast<uintptr_t>(activation) |
                          reinterpret_cast<uintptr_t>(weight_row)) &
                         3U) == 0U;
  if (paired) {
    const auto *activation_pairs =
        reinterpret_cast<const uint32_t *>(activation);
    const auto *weight_pairs = reinterpret_cast<const uint32_t *>(weight_row);
    for (uint64_t pair = threadIdx.x; pair < k / 2U; pair += blockDim.x) {
      const uint32_t activation_pair = activation_pairs[pair];
      const uint32_t weight_pair =
          __builtin_nontemporal_load(weight_pairs + pair);
      partial += bf16_to_float(static_cast<uint16_t>(activation_pair)) *
                 bf16_to_float(static_cast<uint16_t>(weight_pair));
      partial += bf16_to_float(static_cast<uint16_t>(activation_pair >> 16U)) *
                 bf16_to_float(static_cast<uint16_t>(weight_pair >> 16U));
    }
  } else {
    for (uint64_t inner = threadIdx.x; inner < k; inner += blockDim.x) {
      partial +=
          bf16_to_float(activation[inner]) * bf16_to_float(weight_row[inner]);
    }
  }
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U)
    partial += __shfl_down(partial, offset, 32U);
  __shared__ float wave_sums[8];
  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t wave = threadIdx.x >> 5U;
  if (lane == 0U)
    wave_sums[wave] = partial;
  __syncthreads();
  if (wave == 0U) {
    partial = lane < 8U ? wave_sums[lane] : 0.0F;
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U)
      partial += __shfl_down(partial, offset, 32U);
    if (lane == 0U)
      output[column] = float_to_bf16_rne(partial);
  }
}

hipError_t launch(const uint16_t *activation, const uint16_t *qkv_weight,
                  const uint16_t *z_weight, const uint16_t *b_weight,
                  const uint16_t *a_weight, uint16_t *qkv_output,
                  uint16_t *z_output, uint16_t *b_output, uint16_t *a_output,
                  const uint64_t k, const hipStream_t stream) noexcept {
  hipLaunchKernelGGL(sllm_gdn_projection_bundle_bf16_fp32_decode_v1,
                     dim3(8192U + 4096U + 32U + 32U), dim3(256U), 0U, stream,
                     activation, qkv_weight, z_weight, b_weight, a_weight,
                     qkv_output, z_output, b_output, a_output, k);
  return hipGetLastError();
}

} // namespace sllm_gdn_projection_bundle_kernel
