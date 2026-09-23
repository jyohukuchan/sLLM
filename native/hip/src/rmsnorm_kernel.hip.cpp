#include "rmsnorm_kernel_internal.hpp"

#include "sllm/hip.h"
#include <lowp/detail/low_precision_block_codec.hpp>

#include <cmath>
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

namespace {

// The rounded BF16 activation the decomposed RMSNorm writes and the FP8
// quantizer then reads. Recomputing it keeps the fused path bit-identical
// without staging the whole row in LDS.
__device__ __forceinline__ uint16_t rmsnorm_prequant_rounded(
    const uint16_t *const residual_output, const uint16_t *const raw_scale,
    const uint64_t row_offset, const uint32_t column, const float inverse_rms,
    const uint32_t scale_mode) noexcept {
  const float value = bf16_to_float(residual_output[row_offset + column]);
  const float raw = bf16_to_float(raw_scale[column]);
  const float scale =
      scale_mode == SLLM_RMSNORM_SCALE_MODE_DIRECT ? raw : (1.0F + raw);
  return sllm_rmsnorm_kernel::float_to_bf16_rne_bits(value * inverse_rms *
                                                     scale);
}

// Pass 1 of a fused producer: byte-identical to `residual_rmsnorm_body<32,8>`.
// The row-sum reduction is deliberately restricted to the first 256 threads
// with a stride of 256, because a floating-point sum is order dependent and
// the decomposed control launches with a 256-thread block. The remaining
// threads stay idle here and join for the quantization passes, where the extra
// waves are what hide memory latency.
__device__ __forceinline__ void rmsnorm_prequant_prepare(
    const uint16_t *const residual, const uint16_t *const addend,
    uint16_t *const residual_output, const uint64_t row_offset,
    const uint32_t normalized_size, const float epsilon, float *const wave_sums,
    float *const inverse_rms_slot) noexcept {
  constexpr uint32_t sum_threads = 256U;
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t wave_count = 8U;
  const uint32_t tid = threadIdx.x;
  if (tid < sum_threads) {
    const uint32_t lane = tid % wave_width;
    const uint32_t wave = tid / wave_width;
    float partial = 0.0F;
    for (uint32_t column = tid; column < normalized_size;
         column += sum_threads) {
      const float sum = bf16_to_float(residual[row_offset + column]) +
                        bf16_to_float(addend[row_offset + column]);
      const uint16_t rounded = sllm_rmsnorm_kernel::float_to_bf16_rne_bits(sum);
      residual_output[row_offset + column] = rounded;
      const float value = bf16_to_float(rounded);
      partial += value * value;
    }
    partial = wave_sum<wave_width>(partial);
    if (lane == 0U) {
      wave_sums[wave] = partial;
    }
  }
  __syncthreads();
  if (tid == 0U) {
    float sum = 0.0F;
    for (uint32_t index = 0U; index != wave_count; ++index) {
      sum += wave_sums[index];
    }
    *inverse_rms_slot =
        1.0F / sqrtf(sum / static_cast<float>(normalized_size) + epsilon);
  }
  __syncthreads();
}

} // namespace

// Phase 87 stage 7: producer-side activation quantization fusion.
//
// These variants fold the activation quantizer into the producer so the
// consumer matmul can skip its own quantize launch (one HIP graph node and
// one inter-kernel gap per removed launch).  Both variants reproduce the
// decomposed chain bit-for-bit: `residual_rmsnorm_body<32,8>` followed by
// `sllm_matmul_bf16_to_fp8_outer_v2` / `..._nvfp4_block16_wave8_v1`.
//
// The rounding helper comes from `rmsnorm_kernel_internal.hpp` and the
// FP8/E2M1 codecs from `lowp/detail/low_precision_block_codec.hpp`, so the
// arithmetic is shared with the standalone kernels rather than copied.

extern "C" __global__
__launch_bounds__(1024, 1) void sllm_rmsnorm_residual_prequant_fp8_v1(
    const uint16_t *residual, const uint16_t *addend, const uint16_t *raw_scale,
    uint16_t *residual_output, uint8_t *quantized, float *activation_scales,
    uint32_t normalized_size, float epsilon, uint32_t scale_mode,
    uint32_t fnuz) {
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t max_waves = 32U;
  __shared__ float wave_sums[8];
  __shared__ float wave_maxima[max_waves];
  __shared__ float inverse_rms;
  __shared__ float shared_scale;
  const uint32_t lane = threadIdx.x % wave_width;
  const uint32_t wave = threadIdx.x / wave_width;
  const uint32_t waves = blockDim.x / wave_width;
  const uint64_t row = static_cast<uint64_t>(blockIdx.x);
  const uint64_t row_offset = row * static_cast<uint64_t>(normalized_size);

  // Pass 1: byte-identical to residual_rmsnorm_body<32,8>. The helper keeps
  // the row-sum reduction on the first 256 threads so the launch width can
  // grow to 1024 (32 waves) without changing `inverse_rms`.
  rmsnorm_prequant_prepare(residual, addend, residual_output, row_offset,
                           normalized_size, epsilon, wave_sums, &inverse_rms);

  // Pass 2: the row amax the FP8 quantizer derives from the BF16 activation.
  float maximum = 0.0F;
  for (uint32_t column = threadIdx.x; column < normalized_size;
       column += blockDim.x) {
    const float value = bf16_to_float(residual_output[row_offset + column]);
    const float raw = bf16_to_float(raw_scale[column]);
    const float scale =
        scale_mode == SLLM_RMSNORM_SCALE_MODE_DIRECT ? raw : (1.0F + raw);
    const uint16_t normalized = sllm_rmsnorm_kernel::float_to_bf16_rne_bits(
        value * inverse_rms * scale);
    maximum = fmaxf(maximum, fabsf(bf16_to_float(normalized)));
  }
  for (uint32_t delta = wave_width / 2U; delta != 0U; delta >>= 1U) {
    maximum = fmaxf(maximum, __shfl_down(maximum, delta, wave_width));
  }
  if (lane == 0U) {
    wave_maxima[wave] = maximum;
  }
  __syncthreads();
  if (wave == 0U) {
    maximum = lane < waves ? wave_maxima[lane] : 0.0F;
    for (uint32_t delta = wave_width / 2U; delta != 0U; delta >>= 1U) {
      maximum = fmaxf(maximum, __shfl_down(maximum, delta, wave_width));
    }
    if (lane == 0U) {
      shared_scale =
          maximum == 0.0F ? 1.0F : maximum / (fnuz != 0U ? 240.0F : 448.0F);
      activation_scales[row] = shared_scale;
    }
  }
  __syncthreads();

  // Pass 3: recompute the rounded BF16 activation and encode it exactly as
  // sllm_matmul_bf16_to_fp8_outer_v2 does on the standalone activation buffer.
  if ((normalized_size & UINT32_C(1)) == 0U) {
    const uint64_t pairs = normalized_size / UINT32_C(2);
    for (uint64_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
      const uint64_t first_column = pair * UINT32_C(2);
      const uint64_t second_column = first_column + UINT32_C(1);
      const float first =
          bf16_to_float(rmsnorm_prequant_rounded(
              residual_output, raw_scale, row_offset,
              static_cast<uint32_t>(first_column), inverse_rms, scale_mode)) /
          shared_scale;
      const float second =
          bf16_to_float(rmsnorm_prequant_rounded(
              residual_output, raw_scale, row_offset,
              static_cast<uint32_t>(second_column), inverse_rms, scale_mode)) /
          shared_scale;
      uint16_t packed;
      if (isfinite(first) && isfinite(second)) {
        packed = __hip_cvt_float2_to_fp8x2(
            make_float2(first, second), __HIP_SATFINITE,
            fnuz != 0U ? __HIP_E4M3_FNUZ : __HIP_E4M3);
      } else {
        packed = static_cast<uint16_t>(
            static_cast<uint16_t>(
                sllm_lowp::float_to_fp8_native(first, fnuz != 0U)) |
            (static_cast<uint16_t>(
                 sllm_lowp::float_to_fp8_native(second, fnuz != 0U))
             << 8U));
      }
      reinterpret_cast<uint16_t *>(quantized + row_offset)[pair] = packed;
    }
  } else {
    for (uint32_t column = threadIdx.x; column < normalized_size;
         column += blockDim.x) {
      const float value = bf16_to_float(rmsnorm_prequant_rounded(
                              residual_output, raw_scale, row_offset, column,
                              inverse_rms, scale_mode)) /
                          shared_scale;
      quantized[row_offset + column] =
          sllm_lowp::float_to_fp8_native(value, fnuz != 0U);
    }
  }
}

extern "C" __global__
__launch_bounds__(1024, 1) void sllm_rmsnorm_residual_prequant_nvfp4_v1(
    const uint16_t *residual, const uint16_t *addend, const uint16_t *raw_scale,
    uint16_t *residual_output, uint8_t *packed_activation,
    uint8_t *activation_block_scales, const float *input_tensor_scale,
    uint32_t normalized_size, float epsilon, uint32_t scale_mode) {
  constexpr uint32_t group_width = 16U;
  __shared__ float wave_sums[8];
  __shared__ float inverse_rms;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_in_group = tid % group_width;
  const uint32_t group = tid / group_width;
  const uint32_t group_count = blockDim.x / group_width;
  const uint64_t row = static_cast<uint64_t>(blockIdx.x);
  const uint64_t row_offset = row * static_cast<uint64_t>(normalized_size);
  const uint64_t blocks_per_row =
      (normalized_size + UINT32_C(15)) / UINT32_C(16);
  const uint64_t packed_row_bytes =
      (normalized_size + UINT32_C(1)) / UINT32_C(2);

  rmsnorm_prequant_prepare(residual, addend, residual_output, row_offset,
                           normalized_size, epsilon, wave_sums, &inverse_rms);

  // Pass 2: one 16-element block per 16-lane group, mirroring
  // sllm_matmul_bf16_to_nvfp4_block16_wave8_v1 lane-for-lane. A 1024-thread
  // block gives 64 concurrent groups; the decomposed quantizer spreads the
  // same blocks over ceil(m*blocks_per_row/8) workgroups, so a 256-thread
  // producer would serialise roughly eight times as many rounds.
  const float global = input_tensor_scale[0];
  for (uint64_t block = group; block < blocks_per_row; block += group_count) {
    const uint64_t base = block * UINT32_C(16);
    const uint64_t column = base + lane_in_group;
    const float value =
        column < normalized_size
            ? bf16_to_float(rmsnorm_prequant_rounded(
                  residual_output, raw_scale, row_offset,
                  static_cast<uint32_t>(column), inverse_rms, scale_mode))
            : 0.0F;
    // Seed through fmaxf(0, ·) so a NaN element contributes nothing, exactly
    // as the decomposed quantizer's unused lanes 16..31 do.
    float maximum = fmaxf(0.0F, fabsf(value));
    for (uint32_t delta = group_width / 2U; delta != 0U; delta >>= 1U) {
      maximum = fmaxf(maximum, __shfl_down(maximum, delta, group_width));
    }
    uint32_t encoded_scale_bits = 0U;
    if (lane_in_group == 0U) {
      const float raw_scale_value = maximum == 0.0F || !(global > 0.0F)
                                        ? 0.0F
                                        : maximum / (6.0F * global);
      encoded_scale_bits = sllm_lowp::float_to_e4m3fn(raw_scale_value);
      activation_block_scales[row * blocks_per_row + block] =
          static_cast<uint8_t>(encoded_scale_bits);
    }
    encoded_scale_bits = __shfl(encoded_scale_bits, 0, group_width);
    const uint8_t encoded_scale = static_cast<uint8_t>(encoded_scale_bits);
    const float decoded_scale =
        sllm_lowp::e4m3fn_to_float(encoded_scale) * global;
    const uint32_t pair_lane = lane_in_group & 7U;
    const uint32_t shuffled_first = pair_lane * 2U;
    const float first_value =
        __shfl(value, static_cast<int>(shuffled_first), group_width);
    const float second_value =
        __shfl(value, static_cast<int>(shuffled_first + 1U), group_width);
    if (lane_in_group < 8U) {
      const uint32_t first = lane_in_group * 2U;
      const uint64_t first_column = base + first;
      const uint64_t second_column = first_column + UINT32_C(1);
      const uint8_t low =
          first_column < normalized_size && decoded_scale > 0.0F
              ? sllm_lowp::float_to_e2m1(first_value / decoded_scale)
              : 0U;
      const uint8_t high =
          second_column < normalized_size && decoded_scale > 0.0F
              ? sllm_lowp::float_to_e2m1(second_value / decoded_scale)
              : 0U;
      if (first_column < normalized_size) {
        packed_activation[row * packed_row_bytes + first_column / UINT32_C(2)] =
            static_cast<uint8_t>(low | static_cast<uint8_t>(high << 4U));
      }
    }
  }
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

// Phase 87 stage 7: producer-side activation quantization launchers.
hipError_t launch_residual_prequant_fp8(
    const uint16_t *const residual, const uint16_t *const addend,
    const uint16_t *const raw_scale, uint16_t *const residual_output,
    uint8_t *const quantized, float *const activation_scales,
    const uint32_t normalized_size, const uint32_t row_count,
    const float epsilon, const uint32_t scale_mode, const uint32_t fnuz,
    const hipStream_t stream) noexcept {
  const dim3 grid(row_count, 1U, 1U);
  const dim3 block(1024U, 1U, 1U);
  hipLaunchKernelGGL(sllm_rmsnorm_residual_prequant_fp8_v1, grid, block, 0U,
                     stream, residual, addend, raw_scale, residual_output,
                     quantized, activation_scales, normalized_size, epsilon,
                     scale_mode, fnuz);
  return hipGetLastError();
}

hipError_t launch_residual_prequant_nvfp4(
    const uint16_t *const residual, const uint16_t *const addend,
    const uint16_t *const raw_scale, uint16_t *const residual_output,
    uint8_t *const packed_activation, uint8_t *const activation_block_scales,
    const float *const input_tensor_scale, const uint32_t normalized_size,
    const uint32_t row_count, const float epsilon, const uint32_t scale_mode,
    const hipStream_t stream) noexcept {
  const dim3 grid(row_count, 1U, 1U);
  const dim3 block(1024U, 1U, 1U);
  hipLaunchKernelGGL(sllm_rmsnorm_residual_prequant_nvfp4_v1, grid, block, 0U,
                     stream, residual, addend, raw_scale, residual_output,
                     packed_activation, activation_block_scales,
                     input_tensor_scale, normalized_size, epsilon, scale_mode);
  return hipGetLastError();
}

} // namespace sllm_rmsnorm_kernel
