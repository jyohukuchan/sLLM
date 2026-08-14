#include "kv_state_kernel_internal.hpp"

#include <cstdint>

namespace {

using Bf16Input = const uint16_t *;

__device__ __forceinline__ uint16_t float_bits_to_f16(const uint32_t bits) {
  const uint32_t sign = (bits >> 16U) & UINT32_C(0x8000);
  const uint32_t exponent = (bits >> 23U) & UINT32_C(0xff);
  const uint32_t fraction = bits & UINT32_C(0x7fffff);
  if (exponent == UINT32_C(0xff)) {
    if (fraction == 0U) {
      return static_cast<uint16_t>(sign | UINT32_C(0x7c00));
    }
    return static_cast<uint16_t>(sign | UINT32_C(0x7e00));
  }
  const int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<uint16_t>(sign | UINT32_C(0x7c00));
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    const uint32_t mantissa = fraction | UINT32_C(0x800000);
    const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((UINT32_C(1) << shift) - 1U);
    const uint32_t halfway = UINT32_C(1) << (shift - 1U);
    if (remainder > halfway ||
        (remainder == halfway && (rounded & UINT32_C(1)) != 0U)) {
      ++rounded;
    }
    return static_cast<uint16_t>(sign | rounded);
  }
  uint32_t rounded_fraction = fraction >> 13U;
  const uint32_t remainder = fraction & UINT32_C(0x1fff);
  if (remainder > UINT32_C(0x1000) ||
      (remainder == UINT32_C(0x1000) &&
       (rounded_fraction & UINT32_C(1)) != 0U)) {
    ++rounded_fraction;
    if (rounded_fraction == UINT32_C(0x400)) {
      rounded_fraction = 0U;
      if (half_exponent + 1 >= 31) {
        return static_cast<uint16_t>(sign | UINT32_C(0x7c00));
      }
      return static_cast<uint16_t>(
          sign | (static_cast<uint32_t>(half_exponent + 1) << 10U));
    }
  }
  return static_cast<uint16_t>(
      sign | (static_cast<uint32_t>(half_exponent) << 10U) | rounded_fraction);
}

__device__ __forceinline__ uint16_t bf16_to_f16(const uint16_t value) {
  return float_bits_to_f16(static_cast<uint32_t>(value) << 16U);
}

} // namespace

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_KV_WORKGROUP_SIZE,
    1) void sllm_kv_state_bf16_to_f16_token_major_v2(Bf16Input key_input,
                                                     Bf16Input value_input,
                                                     uint16_t *const key_output,
                                                     uint16_t
                                                         *const value_output,
                                                     const uint32_t token_count,
                                                     const uint64_t
                                                     /*capacity_tokens*/,
                                                     const uint64_t
                                                         start_position,
                                                     const uint32_t head_count,
                                                     const uint32_t head_dim) {
  const uint64_t element = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                           static_cast<uint64_t>(threadIdx.x);
  const uint64_t row_width = static_cast<uint64_t>(head_count) * head_dim;
  const uint64_t total = static_cast<uint64_t>(token_count) * row_width;
  if (element >= total) {
    return;
  }
  const uint64_t row = element / row_width;
  const uint64_t within_row = element % row_width;
  const uint64_t output_offset =
      (start_position + row) * row_width + within_row;
  key_output[output_offset] = bf16_to_f16(key_input[element]);
  value_output[output_offset] = bf16_to_f16(value_input[element]);
}

namespace sllm_kv_state_kernel {

hipError_t launch(const uint16_t *const key_input,
                  const uint16_t *const value_input, uint16_t *const key_output,
                  uint16_t *const value_output, const uint32_t token_count,
                  const uint64_t capacity_tokens, const uint64_t start_position,
                  const uint32_t head_count, const uint32_t head_dim,
                  const hipStream_t stream) noexcept {
  const uint64_t total =
      static_cast<uint64_t>(token_count) * head_count * head_dim;
  const uint32_t grid_count = static_cast<uint32_t>(
      (total + SLLM_HIP_KV_WORKGROUP_SIZE - 1U) / SLLM_HIP_KV_WORKGROUP_SIZE);
  const dim3 grid(grid_count, 1U, 1U);
  const dim3 block(SLLM_HIP_KV_WORKGROUP_SIZE, 1U, 1U);
  hipLaunchKernelGGL(sllm_kv_state_bf16_to_f16_token_major_v2, grid, block, 0U,
                     stream, key_input, value_input, key_output, value_output,
                     token_count, capacity_tokens, start_position, head_count,
                     head_dim);
  return hipGetLastError();
}

} // namespace sllm_kv_state_kernel
