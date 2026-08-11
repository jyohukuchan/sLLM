#include "elementwise_kernel_internal.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

__device__ __forceinline__ float bf16_to_float(const uint16_t value) noexcept {
  return __uint_as_float(static_cast<uint32_t>(value) << 16U);
}

__device__ __forceinline__ uint16_t
float_to_bf16_rne_bits(const float value) noexcept {
  const uint32_t bits = __float_as_uint(value);
  constexpr uint32_t exponent_mask = UINT32_C(0x7f800000);
  constexpr uint32_t fraction_mask = UINT32_C(0x007fffff);
  if ((bits & exponent_mask) == exponent_mask) {
    if ((bits & fraction_mask) != 0U) {
      const uint16_t sign =
          static_cast<uint16_t>((bits >> 16U) & UINT32_C(0x8000));
      const uint16_t payload =
          static_cast<uint16_t>((bits >> 16U) & UINT32_C(0x003f));
      return static_cast<uint16_t>(sign | UINT16_C(0x7fc0) | payload);
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & UINT32_C(1)) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

} // namespace

extern "C" __global__ __launch_bounds__(
    256, 1) void sllm_elementwise_copy_bf16_v1(const uint16_t *const input,
                                               uint16_t *const output,
                                               const uint64_t element_count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    output[index] = input[index];
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_elementwise_add_bf16_fp32_v1(
    const uint16_t *const input0, const uint16_t *const input1,
    uint16_t *const output, const uint64_t element_count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    output[index] = float_to_bf16_rne_bits(bf16_to_float(input0[index]) +
                                           bf16_to_float(input1[index]));
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_elementwise_silu_mul_bf16_fp32_v1(
    const uint16_t *const gate, const uint16_t *const up,
    uint16_t *const output, const uint64_t element_count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    const float gate_value = bf16_to_float(gate[index]);
    const float silu = gate_value / (1.0F + ::expf(-gate_value));
    const uint16_t silu_bf16 = float_to_bf16_rne_bits(silu);
    output[index] = float_to_bf16_rne_bits(bf16_to_float(silu_bf16) *
                                           bf16_to_float(up[index]));
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_elementwise_sigmoid_mul_bf16_fp32_v1(
    const uint16_t *const gate, const uint16_t *const attention_value,
    uint16_t *const output, const uint64_t element_count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    const float gate_value = bf16_to_float(gate[index]);
    const float sigmoid = 1.0F / (1.0F + ::expf(-gate_value));
    const uint16_t sigmoid_bf16 = float_to_bf16_rne_bits(sigmoid);
    output[index] = float_to_bf16_rne_bits(
        bf16_to_float(sigmoid_bf16) * bf16_to_float(attention_value[index]));
  }
}

namespace sllm_elementwise_kernel {
namespace {

bool grid_for(const uint64_t element_count, dim3 *const grid) noexcept {
  if (grid == nullptr || element_count == 0U) {
    return false;
  }
  const uint64_t workgroup = static_cast<uint64_t>(kWorkgroupSize);
  const uint64_t blocks =
      element_count / workgroup +
      static_cast<uint64_t>(element_count % workgroup != 0U);
  if (blocks > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  *grid = dim3(static_cast<uint32_t>(blocks), 1U, 1U);
  return true;
}

} // namespace

hipError_t launch_copy(const uint16_t *const input, uint16_t *const output,
                       const uint64_t element_count,
                       const hipStream_t stream) noexcept {
  dim3 grid;
  if (!grid_for(element_count, &grid)) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_elementwise_copy_bf16_v1, grid, dim3(kWorkgroupSize),
                     0U, stream, input, output, element_count);
  return hipGetLastError();
}

hipError_t launch_add(const uint16_t *const input0,
                      const uint16_t *const input1, uint16_t *const output,
                      const uint64_t element_count,
                      const hipStream_t stream) noexcept {
  dim3 grid;
  if (!grid_for(element_count, &grid)) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_elementwise_add_bf16_fp32_v1, grid,
                     dim3(kWorkgroupSize), 0U, stream, input0, input1, output,
                     element_count);
  return hipGetLastError();
}

hipError_t launch_silu_mul(const uint16_t *const gate, const uint16_t *const up,
                           uint16_t *const output, const uint64_t element_count,
                           const hipStream_t stream) noexcept {
  dim3 grid;
  if (!grid_for(element_count, &grid)) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_elementwise_silu_mul_bf16_fp32_v1, grid,
                     dim3(kWorkgroupSize), 0U, stream, gate, up, output,
                     element_count);
  return hipGetLastError();
}

hipError_t launch_sigmoid_mul(const uint16_t *const gate,
                              const uint16_t *const attention_value,
                              uint16_t *const output,
                              const uint64_t element_count,
                              const hipStream_t stream) noexcept {
  dim3 grid;
  if (!grid_for(element_count, &grid)) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_elementwise_sigmoid_mul_bf16_fp32_v1, grid,
                     dim3(kWorkgroupSize), 0U, stream, gate, attention_value,
                     output, element_count);
  return hipGetLastError();
}

} // namespace sllm_elementwise_kernel
