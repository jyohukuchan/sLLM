#include "rmsnorm_kernel_internal.hpp"

#include <cstdint>

namespace {

__device__ __forceinline__ float bf16_to_float(const uint16_t value) noexcept {
  return __uint_as_float(static_cast<uint32_t>(value) << 16U);
}

__device__ __forceinline__ float wave32_sum(float value) noexcept {
  for (unsigned int delta = 16U; delta != 0U; delta >>= 1U) {
    value += __shfl_down(value, delta, 32);
  }
  return value;
}

} // namespace

extern "C" __global__
__launch_bounds__(256, 1) void sllm_rmsnorm_baseline_wave32_v1(
    const uint16_t *const activation, const uint16_t *const raw_scale,
    uint16_t *const output, const uint32_t normalized_size,
    const float epsilon) {
  __shared__ float wave_sums[8];
  __shared__ float inverse_rms;
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int wave = threadIdx.x >> 5U;
  const uint64_t row = static_cast<uint64_t>(blockIdx.x);
  const uint64_t row_offset = row * static_cast<uint64_t>(normalized_size);

  float partial = 0.0F;
  for (uint32_t column = threadIdx.x; column < normalized_size;
       column += blockDim.x) {
    const float value = bf16_to_float(activation[row_offset + column]);
    partial += value * value;
  }
  partial = wave32_sum(partial);
  if (lane == 0U) {
    wave_sums[wave] = partial;
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    float sum = 0.0F;
    for (unsigned int index = 0U; index != 8U; ++index) {
      sum += wave_sums[index];
    }
    const float mean = sum / static_cast<float>(normalized_size);
    inverse_rms = 1.0F / sqrtf(mean + epsilon);
  }
  __syncthreads();
  for (uint32_t column = threadIdx.x; column < normalized_size;
       column += blockDim.x) {
    const float value = bf16_to_float(activation[row_offset + column]);
    const float raw = bf16_to_float(raw_scale[column]);
    output[row_offset + column] = sllm_rmsnorm_kernel::float_to_bf16_rne_bits(
        value * inverse_rms * (1.0F + raw));
  }
}

namespace sllm_rmsnorm_kernel {

hipError_t launch(const uint16_t *const activation,
                  const uint16_t *const raw_scale, uint16_t *const output,
                  const uint32_t normalized_size, const uint32_t row_count,
                  const float epsilon, const hipStream_t stream) noexcept {
  const dim3 grid(row_count, 1U, 1U);
  const dim3 block(256U, 1U, 1U);
  hipLaunchKernelGGL(sllm_rmsnorm_baseline_wave32_v1, grid, block, 0U, stream,
                     activation, raw_scale, output, normalized_size, epsilon);
  return hipGetLastError();
}

} // namespace sllm_rmsnorm_kernel
