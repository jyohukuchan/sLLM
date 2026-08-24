#include "rmsnorm_kernel_internal.hpp"

#include "sllm/hip.h"

#include <cstdint>

namespace {

__device__ __forceinline__ float bf16_to_float(const uint16_t value) noexcept {
  return __uint_as_float(static_cast<uint32_t>(value) << 16U);
}

template <unsigned int WaveWidth>
__device__ __forceinline__ float wave_sum(float value) noexcept {
  for (unsigned int delta = WaveWidth / 2U; delta != 0U; delta >>= 1U) {
    value += __shfl_down(value, delta, WaveWidth);
  }
  return value;
}

template <unsigned int WaveWidth, unsigned int WaveCount>
__device__ __forceinline__ void
rmsnorm_body(const uint16_t *const activation, const uint16_t *const raw_scale,
             uint16_t *const output, const uint32_t normalized_size,
             const float epsilon, const uint32_t scale_mode) noexcept {
  __shared__ float wave_sums[WaveCount];
  __shared__ float inverse_rms;
  const unsigned int lane = threadIdx.x % WaveWidth;
  const unsigned int wave = threadIdx.x / WaveWidth;
  const uint64_t row = static_cast<uint64_t>(blockIdx.x);
  const uint64_t row_offset = row * static_cast<uint64_t>(normalized_size);

  float partial = 0.0F;
  for (uint32_t column = threadIdx.x; column < normalized_size;
       column += blockDim.x) {
    const float value = bf16_to_float(activation[row_offset + column]);
    partial += value * value;
  }
  partial = wave_sum<WaveWidth>(partial);
  if (lane == 0U) {
    wave_sums[wave] = partial;
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    float sum = 0.0F;
    for (unsigned int index = 0U; index != WaveCount; ++index) {
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
    const float scale =
        scale_mode == SLLM_RMSNORM_SCALE_MODE_DIRECT ? raw : (1.0F + raw);
    output[row_offset + column] = sllm_rmsnorm_kernel::float_to_bf16_rne_bits(
        value * inverse_rms * scale);
  }
}

template <unsigned int WaveWidth, unsigned int WaveCount>
__device__ __forceinline__ void residual_rmsnorm_body(
    const uint16_t *const residual, const uint16_t *const addend,
    const uint16_t *const raw_scale, uint16_t *const residual_output,
    uint16_t *const output, const uint32_t normalized_size, const float epsilon,
    const uint32_t scale_mode) noexcept {
  __shared__ float wave_sums[WaveCount];
  __shared__ float inverse_rms;
  const unsigned int lane = threadIdx.x % WaveWidth;
  const unsigned int wave = threadIdx.x / WaveWidth;
  const uint64_t row = static_cast<uint64_t>(blockIdx.x);
  const uint64_t row_offset = row * static_cast<uint64_t>(normalized_size);
  float partial = 0.0F;
  for (uint32_t column = threadIdx.x; column < normalized_size;
       column += blockDim.x) {
    const float sum = bf16_to_float(residual[row_offset + column]) +
                      bf16_to_float(addend[row_offset + column]);
    const uint16_t rounded = sllm_rmsnorm_kernel::float_to_bf16_rne_bits(sum);
    residual_output[row_offset + column] = rounded;
    const float value = bf16_to_float(rounded);
    partial += value * value;
  }
  partial = wave_sum<WaveWidth>(partial);
  if (lane == 0U)
    wave_sums[wave] = partial;
  __syncthreads();
  if (threadIdx.x == 0U) {
    float sum = 0.0F;
    for (unsigned int index = 0U; index != WaveCount; ++index)
      sum += wave_sums[index];
    inverse_rms =
        1.0F / sqrtf(sum / static_cast<float>(normalized_size) + epsilon);
  }
  __syncthreads();
  for (uint32_t column = threadIdx.x; column < normalized_size;
       column += blockDim.x) {
    const float value = bf16_to_float(residual_output[row_offset + column]);
    const float raw = bf16_to_float(raw_scale[column]);
    const float scale =
        scale_mode == SLLM_RMSNORM_SCALE_MODE_DIRECT ? raw : (1.0F + raw);
    output[row_offset + column] = sllm_rmsnorm_kernel::float_to_bf16_rne_bits(
        value * inverse_rms * scale);
  }
}

} // namespace

extern "C" __global__
__launch_bounds__(256, 1) void sllm_rmsnorm_baseline_wave32_v1(
    const uint16_t *const activation, const uint16_t *const raw_scale,
    uint16_t *const output, const uint32_t normalized_size, const float epsilon,
    const uint32_t scale_mode) {
  rmsnorm_body<32U, 8U>(activation, raw_scale, output, normalized_size, epsilon,
                        scale_mode);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_rmsnorm_baseline_wave64_v1(
    const uint16_t *const activation, const uint16_t *const raw_scale,
    uint16_t *const output, const uint32_t normalized_size, const float epsilon,
    const uint32_t scale_mode) {
  rmsnorm_body<64U, 4U>(activation, raw_scale, output, normalized_size, epsilon,
                        scale_mode);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_rmsnorm_residual_fused_wave32_v1(
    const uint16_t *residual, const uint16_t *addend, const uint16_t *raw_scale,
    uint16_t *residual_output, uint16_t *output, uint32_t normalized_size,
    float epsilon, uint32_t scale_mode) {
  residual_rmsnorm_body<32U, 8U>(residual, addend, raw_scale, residual_output,
                                 output, normalized_size, epsilon, scale_mode);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_rmsnorm_residual_fused_wave64_v1(
    const uint16_t *residual, const uint16_t *addend, const uint16_t *raw_scale,
    uint16_t *residual_output, uint16_t *output, uint32_t normalized_size,
    float epsilon, uint32_t scale_mode) {
  residual_rmsnorm_body<64U, 4U>(residual, addend, raw_scale, residual_output,
                                 output, normalized_size, epsilon, scale_mode);
}

namespace sllm_rmsnorm_kernel {

hipError_t launch(const uint16_t *const activation,
                  const uint16_t *const raw_scale, uint16_t *const output,
                  const uint32_t normalized_size, const uint32_t row_count,
                  const float epsilon, const uint32_t scale_mode,
                  const hipStream_t stream) noexcept {
  const dim3 grid(row_count, 1U, 1U);
  const dim3 block(256U, 1U, 1U);
#if defined(SLLM_HIP_COMPILE_WAVE64) && SLLM_HIP_COMPILE_WAVE64 == 1
  hipLaunchKernelGGL(sllm_rmsnorm_baseline_wave64_v1, grid, block, 0U, stream,
                     activation, raw_scale, output, normalized_size, epsilon,
                     scale_mode);
#else
  hipLaunchKernelGGL(sllm_rmsnorm_baseline_wave32_v1, grid, block, 0U, stream,
                     activation, raw_scale, output, normalized_size, epsilon,
                     scale_mode);
#endif
  return hipGetLastError();
}

hipError_t launch_residual_fused(
    const uint16_t *const residual, const uint16_t *const addend,
    const uint16_t *const raw_scale, uint16_t *const residual_output,
    uint16_t *const output, const uint32_t normalized_size,
    const uint32_t row_count, const float epsilon, const uint32_t scale_mode,
    const hipStream_t stream) noexcept {
  const dim3 grid(row_count, 1U, 1U);
  const dim3 block(256U, 1U, 1U);
#if defined(SLLM_HIP_COMPILE_WAVE64) && SLLM_HIP_COMPILE_WAVE64 == 1
  hipLaunchKernelGGL(sllm_rmsnorm_residual_fused_wave64_v1, grid, block, 0U,
                     stream, residual, addend, raw_scale, residual_output,
                     output, normalized_size, epsilon, scale_mode);
#else
  hipLaunchKernelGGL(sllm_rmsnorm_residual_fused_wave32_v1, grid, block, 0U,
                     stream, residual, addend, raw_scale, residual_output,
                     output, normalized_size, epsilon, scale_mode);
#endif
  return hipGetLastError();
}

} // namespace sllm_rmsnorm_kernel
