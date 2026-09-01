// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase9-mmvf-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// f5919bf458ef190468b5c329bb293f8a54a1e69c,
// ggml/src/ggml-cuda/mmvf.cu
// SPDX-License-Identifier: MIT

#include "low_precision_block_codec.hpp"
#include "matmul_kernel_internal.hpp"

#include <hip/hip_fp8.h>

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1201__)
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>
#define SLLM_MATMUL_HAS_GFX12_ROCWMMA 1
#endif

#include <cstdint>

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

__device__ __forceinline__ float e4m3fn_to_float(const uint8_t bits) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode(bits);
}

__device__ __forceinline__ uint8_t float_to_e4m3fn(float value) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::encode(value);
}

__device__ __forceinline__ uint8_t
float_to_fp8_native(const float value, const bool fnuz) noexcept {
  return sllm_lowp::float_to_fp8_native(value, fnuz);
}

__device__ __forceinline__ uint8_t float_to_e4m3fnuz(float value) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E4M3FnuZ>::encode(value);
}

__device__ __forceinline__ float e2m1_to_float(const uint8_t bits) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E2M1>::decode(bits);
}

__device__ __forceinline__ uint8_t float_to_e2m1(float value) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E2M1>::encode(value);
}

__device__ __forceinline__ float e3m2_to_float(const uint8_t raw) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E3M2>::decode(raw);
}

__device__ __forceinline__ uint8_t float_to_e3m2(float value) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E3M2>::encode(value);
}

__device__ __forceinline__ float e8m0_to_float(const uint8_t bits) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E8M0>::decode(bits);
}

__device__ __forceinline__ uint8_t
mxfp4_even_scale_code(const float maximum) noexcept {
  return sllm_lowp::mxfp4_even_scale_code(maximum);
}

__device__ __forceinline__ uint8_t
packed_e3m2_at(const uint8_t *const row, const uint64_t index) noexcept {
  return sllm_lowp::packed_e3m2_at(row, index);
}

} // namespace

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_to_fp8_outer_v1(
    const uint16_t *const activation, uint8_t *const quantized,
    float *const scales, const uint64_t m, const uint64_t k,
    const uint32_t fnuz) {
  const uint64_t row = blockIdx.x;
  if (row >= m) {
    return;
  }
  float maximum = 0.0F;
  for (uint64_t column = threadIdx.x; column < k; column += blockDim.x) {
    maximum =
        fmaxf(maximum, fabsf(bf16_to_float(activation[row * k + column])));
  }
  __shared__ float reductions[256];
  reductions[threadIdx.x] = maximum;
  __syncthreads();
  for (uint32_t offset = 128U; offset != 0U; offset >>= 1U) {
    if (threadIdx.x < offset) {
      reductions[threadIdx.x] =
          fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  const float scale = reductions[0] == 0.0F
                          ? 1.0F
                          : reductions[0] / (fnuz != 0U ? 240.0F : 448.0F);
  if (threadIdx.x == 0U) {
    scales[row] = scale;
  }
  for (uint64_t column = threadIdx.x; column < k; column += blockDim.x) {
    const float value = bf16_to_float(activation[row * k + column]) / scale;
    quantized[row * k + column] =
        fnuz != 0U ? float_to_e4m3fnuz(value) : float_to_e4m3fn(value);
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_to_fp8_outer_v2(
    const uint16_t *const activation, uint8_t *const quantized,
    float *const scales, const uint64_t m, const uint64_t k,
    const uint32_t fnuz) {
  const uint64_t row = blockIdx.x;
  if (row >= m) {
    return;
  }
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t wave_count = 8U;
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  float maximum = 0.0F;
  for (uint64_t column = threadIdx.x; column < k; column += blockDim.x) {
    maximum =
        fmaxf(maximum, fabsf(bf16_to_float(activation[row * k + column])));
  }
#pragma unroll
  for (uint32_t offset = wave_width / 2U; offset != 0U; offset >>= 1U) {
    maximum = fmaxf(maximum, __shfl_down(maximum, offset, wave_width));
  }
  __shared__ float wave_maxima[wave_count];
  __shared__ float shared_scale;
  if (lane == 0U) {
    wave_maxima[wave] = maximum;
  }
  __syncthreads();
  if (wave == 0U) {
    maximum = lane < wave_count ? wave_maxima[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = wave_width / 2U; offset != 0U; offset >>= 1U) {
      maximum = fmaxf(maximum, __shfl_down(maximum, offset, wave_width));
    }
    if (lane == 0U) {
      shared_scale =
          maximum == 0.0F ? 1.0F : maximum / (fnuz != 0U ? 240.0F : 448.0F);
      scales[row] = shared_scale;
    }
  }
  __syncthreads();
  const uint64_t row_offset = row * k;
  if ((k & UINT64_C(1)) == 0U) {
    const uint64_t pairs = k / UINT64_C(2);
    for (uint64_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
      const float first =
          bf16_to_float(activation[row_offset + pair * UINT64_C(2)]) /
          shared_scale;
      const float second =
          bf16_to_float(activation[row_offset + pair * UINT64_C(2) + 1U]) /
          shared_scale;
      uint16_t packed;
      if (isfinite(first) && isfinite(second)) {
        packed = __hip_cvt_float2_to_fp8x2(
            make_float2(first, second), __HIP_SATFINITE,
            fnuz != 0U ? __HIP_E4M3_FNUZ : __HIP_E4M3);
      } else {
        packed = static_cast<uint16_t>(
            static_cast<uint16_t>(float_to_fp8_native(first, fnuz != 0U)) |
            (static_cast<uint16_t>(float_to_fp8_native(second, fnuz != 0U))
             << 8U));
      }
      reinterpret_cast<uint16_t *>(quantized + row_offset)[pair] = packed;
    }
  } else {
    for (uint64_t column = threadIdx.x; column < k; column += blockDim.x) {
      const float value =
          bf16_to_float(activation[row_offset + column]) / shared_scale;
      quantized[row_offset + column] = float_to_fp8_native(value, fnuz != 0U);
    }
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_fp8_outer_emulation_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index >= m * n) {
    return;
  }
  const uint64_t row = index / n;
  const uint64_t column = index - row * n;
  float accumulator = 0.0F;
  for (uint64_t inner = 0U; inner < k; ++inner) {
    accumulator =
        fmaf(e4m3fn_to_float(activation[row * k + inner]),
             e4m3fn_to_float(weight[column * k + inner]), accumulator);
  }
  output[index] = float_to_bf16_rne_bits(accumulator * activation_scales[row] *
                                         weight_scales[column]);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_to_nvfp4_block16_v1(
    const uint16_t *const activation, uint8_t *const packed_activation,
    uint8_t *const block_scales, const float *const input_tensor_scale,
    const uint64_t m, const uint64_t k) {
  const uint64_t blocks_per_row = (k + UINT64_C(15)) / UINT64_C(16);
  const uint64_t block_index = blockIdx.x;
  if (block_index >= m * blocks_per_row) {
    return;
  }
  const uint64_t row = block_index / blocks_per_row;
  const uint64_t block = block_index - row * blocks_per_row;
  const uint64_t base = block * UINT64_C(16);
  const uint64_t packed_row_bytes = (k + UINT64_C(1)) / UINT64_C(2);
  __shared__ float values[16];
  __shared__ float decoded_scale;
  if (threadIdx.x < 16U) {
    const uint64_t column = base + threadIdx.x;
    values[threadIdx.x] =
        column < k ? bf16_to_float(activation[row * k + column]) : 0.0F;
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    float maximum = 0.0F;
    for (uint32_t index = 0U; index != 16U; ++index) {
      maximum = fmaxf(maximum, fabsf(values[index]));
    }
    const float global = input_tensor_scale[0];
    const float raw_scale =
        maximum == 0.0F || !(global > 0.0F) ? 0.0F : maximum / (6.0F * global);
    const uint8_t encoded_scale = float_to_e4m3fn(raw_scale);
    block_scales[block_index] = encoded_scale;
    decoded_scale = e4m3fn_to_float(encoded_scale) * global;
  }
  __syncthreads();
  if (threadIdx.x < 8U) {
    const uint32_t first = threadIdx.x * 2U;
    const uint64_t first_column = base + first;
    const uint64_t second_column = first_column + 1U;
    if (first_column < k) {
      const uint8_t low = decoded_scale > 0.0F
                              ? float_to_e2m1(values[first] / decoded_scale)
                              : 0U;
      const uint8_t high =
          second_column < k && decoded_scale > 0.0F
              ? float_to_e2m1(values[first + 1U] / decoded_scale)
              : 0U;
      packed_activation[row * packed_row_bytes + first_column / UINT64_C(2)] =
          static_cast<uint8_t>(low | static_cast<uint8_t>(high << 4U));
    }
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_block16_packed_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  const uint64_t output_index = blockIdx.x;
  if (output_index >= m * n) {
    return;
  }
  const uint64_t row = output_index / n;
  const uint64_t column = output_index - row * n;
  const uint64_t blocks_per_row = (k + UINT64_C(15)) / UINT64_C(16);
  const uint64_t packed_activation_row = (k + UINT64_C(1)) / UINT64_C(2);
  float partial = 0.0F;
  for (uint64_t inner = threadIdx.x; inner < k; inner += blockDim.x) {
    const uint8_t activation_pair = __builtin_nontemporal_load(
        packed_activation + row * packed_activation_row + inner / 2U);
    const uint8_t activation_code = (inner & UINT64_C(1)) == 0U
                                        ? activation_pair & UINT8_C(0x0f)
                                        : activation_pair >> 4U;
    const uint64_t weight_index = column * k + inner;
    const uint8_t weight_pair =
        __builtin_nontemporal_load(packed_weight + weight_index / UINT64_C(2));
    const uint8_t weight_code = (weight_index & UINT64_C(1)) == 0U
                                    ? weight_pair & UINT8_C(0x0f)
                                    : weight_pair >> 4U;
    const float activation_scale = e4m3fn_to_float(
        activation_block_scales[row * blocks_per_row + inner / 16U]);
    const float weight_scale = e4m3fn_to_float(
        weight_block_scales[column * blocks_per_row + inner / 16U]);
    partial += e2m1_to_float(activation_code) * activation_scale *
               e2m1_to_float(weight_code) * weight_scale;
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    partial += __shfl_down(partial, offset, 32U);
  }
  __shared__ float wave_sums[8];
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  if (lane == 0U) {
    wave_sums[wave] = partial;
  }
  __syncthreads();
  if (wave == 0U) {
    partial = lane < 8U ? wave_sums[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
      partial += __shfl_down(partial, offset, 32U);
    }
    if (lane == 0U) {
      output[output_index] = float_to_bf16_rne_bits(
          partial * weight_tensor_scale[0] * input_tensor_scale[0]);
    }
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_to_mxfp4_block32_even_v1(
    const uint16_t *const activation, uint8_t *const packed_activation,
    uint8_t *const block_scales, const uint64_t m, const uint64_t k) {
  const uint64_t blocks_per_row = (k + UINT64_C(31)) / UINT64_C(32);
  const uint64_t block_index = blockIdx.x;
  if (block_index >= m * blocks_per_row) {
    return;
  }
  const uint64_t row = block_index / blocks_per_row;
  const uint64_t block = block_index - row * blocks_per_row;
  const uint64_t base = block * UINT64_C(32);
  const uint64_t packed_row_bytes = (k + UINT64_C(1)) / UINT64_C(2);
  __shared__ float values[32];
  __shared__ float decoded_scale;
  if (threadIdx.x < 32U) {
    const uint64_t column = base + threadIdx.x;
    values[threadIdx.x] =
        column < k ? bf16_to_float(activation[row * k + column]) : 0.0F;
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    float maximum = 0.0F;
    for (uint32_t index = 0U; index != 32U; ++index) {
      maximum = fmaxf(maximum, fabsf(values[index]));
    }
    const uint8_t encoded_scale = mxfp4_even_scale_code(maximum);
    block_scales[block_index] = encoded_scale;
    decoded_scale = e8m0_to_float(encoded_scale);
  }
  __syncthreads();
  if (threadIdx.x < 16U) {
    const uint32_t first = threadIdx.x * 2U;
    const uint64_t first_column = base + first;
    const uint64_t second_column = first_column + 1U;
    if (first_column < k) {
      const uint8_t low = isfinite(decoded_scale) && decoded_scale > 0.0F
                              ? float_to_e2m1(values[first] / decoded_scale)
                              : 0U;
      const uint8_t high =
          second_column < k && isfinite(decoded_scale) && decoded_scale > 0.0F
              ? float_to_e2m1(values[first + 1U] / decoded_scale)
              : 0U;
      packed_activation[row * packed_row_bytes + first_column / UINT64_C(2)] =
          static_cast<uint8_t>(low | static_cast<uint8_t>(high << 4U));
    }
  }
}

__device__ __forceinline__ void sllm_matmul_mxfp4_w4a4_block32_body(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  const uint64_t output_index = blockIdx.x;
  if (output_index >= m * n) {
    return;
  }
  const uint64_t row = output_index / n;
  const uint64_t column = output_index - row * n;
  const uint64_t blocks_per_row = (k + UINT64_C(31)) / UINT64_C(32);
  const uint64_t packed_row_bytes = (k + UINT64_C(1)) / UINT64_C(2);
  float partial = 0.0F;
  for (uint64_t inner = threadIdx.x; inner < k; inner += blockDim.x) {
    const uint8_t activation_pair = __builtin_nontemporal_load(
        packed_activation + row * packed_row_bytes + inner / UINT64_C(2));
    const uint8_t activation_code = (inner & UINT64_C(1)) == 0U
                                        ? activation_pair & UINT8_C(0x0f)
                                        : activation_pair >> 4U;
    const uint8_t weight_pair = __builtin_nontemporal_load(
        packed_weight + column * packed_row_bytes + inner / UINT64_C(2));
    const uint8_t weight_code = (inner & UINT64_C(1)) == 0U
                                    ? weight_pair & UINT8_C(0x0f)
                                    : weight_pair >> 4U;
    const float activation_scale = e8m0_to_float(
        activation_block_scales[row * blocks_per_row + inner / UINT64_C(32)]);
    const float weight_scale = e8m0_to_float(
        weight_block_scales[column * blocks_per_row + inner / UINT64_C(32)]);
    partial += e2m1_to_float(activation_code) * activation_scale *
               e2m1_to_float(weight_code) * weight_scale;
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    partial += __shfl_down(partial, offset, 32U);
  }
  __shared__ float wave_sums[8];
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  if (lane == 0U) {
    wave_sums[wave] = partial;
  }
  __syncthreads();
  if (wave == 0U) {
    partial = lane < 8U ? wave_sums[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
      partial += __shfl_down(partial, offset, 32U);
    }
    if (lane == 0U) {
      output[output_index] = float_to_bf16_rne_bits(partial);
    }
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_mxfp4_w4a4_block32_decode_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  sllm_matmul_mxfp4_w4a4_block32_body(packed_activation,
                                      activation_block_scales, packed_weight,
                                      weight_block_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_mxfp4_w4a4_block32_prefill_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  sllm_matmul_mxfp4_w4a4_block32_body(packed_activation,
                                      activation_block_scales, packed_weight,
                                      weight_block_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(32, 1) void sllm_matmul_bf16_to_mxfp8_e4m3_block32_v1(
    const uint16_t *const activation, uint8_t *const quantized,
    uint8_t *const block_scales, const uint64_t m, const uint64_t k) {
  const uint64_t blocks_per_row = k / UINT64_C(32);
  const uint64_t block_index = blockIdx.x;
  if (block_index >= m * blocks_per_row) {
    return;
  }
  const uint64_t row = block_index / blocks_per_row;
  const uint64_t block = block_index - row * blocks_per_row;
  const uint64_t base = block * UINT64_C(32);
  const uint32_t lane = threadIdx.x;
  const float value = bf16_to_float(activation[row * k + base + lane]);
  uint32_t has_nan = static_cast<uint32_t>(isnan(value));
  float maximum = has_nan != 0U ? 0.0F : fabsf(value);
  maximum = sllm_lowp::wave_amax(maximum);
  has_nan = sllm_lowp::wave_or(has_nan);
  uint32_t scale = 0U;
  if (lane == 0U) {
    scale = sllm_lowp::BlockCodec<sllm_lowp::Mxfp8E4Block32>::scale_code(
        maximum, has_nan != 0U);
    block_scales[block_index] = static_cast<uint8_t>(scale);
  }
  scale = __shfl(scale, 0U, 32U);
  const float decoded_scale = e8m0_to_float(static_cast<uint8_t>(scale));
  quantized[row * k + base + lane] =
      isfinite(decoded_scale) && decoded_scale > 0.0F
          ? float_to_e4m3fn(value / decoded_scale)
          : 0U;
}

extern "C" __global__
__launch_bounds__(32, 1) void sllm_matmul_bf16_to_mxfp6_e3m2_block32_v1(
    const uint16_t *const activation, uint8_t *const packed_activation,
    uint8_t *const block_scales, const uint64_t m, const uint64_t k) {
  const uint64_t blocks_per_row = k / UINT64_C(32);
  const uint64_t block_index = blockIdx.x;
  if (block_index >= m * blocks_per_row) {
    return;
  }
  const uint64_t row = block_index / blocks_per_row;
  const uint64_t block = block_index - row * blocks_per_row;
  const uint64_t base = block * UINT64_C(32);
  const uint32_t lane = threadIdx.x;
  const float value = bf16_to_float(activation[row * k + base + lane]);
  uint32_t has_nan = static_cast<uint32_t>(isnan(value));
  float maximum = has_nan != 0U ? 0.0F : fabsf(value);
  maximum = sllm_lowp::wave_amax(maximum);
  has_nan = sllm_lowp::wave_or(has_nan);
  uint32_t scale = 0U;
  if (lane == 0U) {
    scale = sllm_lowp::BlockCodec<sllm_lowp::Mxfp6E3Block32>::scale_code(
        maximum, has_nan != 0U);
    block_scales[block_index] = static_cast<uint8_t>(scale);
  }
  scale = __shfl(scale, 0U, 32U);
  const float decoded_scale = e8m0_to_float(static_cast<uint8_t>(scale));
  const uint32_t code =
      isfinite(decoded_scale) && decoded_scale > 0.0F
          ? static_cast<uint32_t>(float_to_e3m2(value / decoded_scale))
          : 0U;
  const uint32_t group = lane & ~UINT32_C(3);
  const uint32_t packed =
      __shfl(code, static_cast<int>(group), 32U) |
      (__shfl(code, static_cast<int>(group + 1U), 32U) << 6U) |
      (__shfl(code, static_cast<int>(group + 2U), 32U) << 12U) |
      (__shfl(code, static_cast<int>(group + 3U), 32U) << 18U);
  if ((lane & UINT32_C(3)) == 0U) {
    const uint64_t row_bytes = k * UINT64_C(3) / UINT64_C(4);
    const uint64_t destination = row * row_bytes + block * UINT64_C(24) +
                                 (lane / UINT32_C(4)) * UINT64_C(3);
    packed_activation[destination] = static_cast<uint8_t>(packed);
    packed_activation[destination + 1U] = static_cast<uint8_t>(packed >> 8U);
    packed_activation[destination + 2U] = static_cast<uint8_t>(packed >> 16U);
  }
}

__device__ __forceinline__ void sllm_matmul_mxfp8_w8a8_block32_body(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  const uint64_t output_index = blockIdx.x;
  if (output_index >= m * n) {
    return;
  }
  const uint64_t row = output_index / n;
  const uint64_t column = output_index - row * n;
  const uint64_t blocks_per_row = k / UINT64_C(32);
  const sllm_lowp::BlockScaledView<sllm_lowp::Mxfp8E4Block32> activation_view{
      activation, activation_scales, nullptr, k, k, blocks_per_row};
  const sllm_lowp::BlockScaledView<sllm_lowp::Mxfp8E4Block32> weight_view{
      weight, weight_scales, nullptr, k, k, blocks_per_row};
  float partial = 0.0F;
  for (uint64_t inner = threadIdx.x; inner < k; inner += blockDim.x) {
    partial += sllm_lowp::BlockCodec<sllm_lowp::Mxfp8E4Block32>::load(
                   activation_view, row, static_cast<uint32_t>(inner)) *
               sllm_lowp::BlockCodec<sllm_lowp::Mxfp8E4Block32>::load(
                   weight_view, column, static_cast<uint32_t>(inner));
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    partial += __shfl_down(partial, offset, 32U);
  }
  __shared__ float wave_sums[8];
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  if (lane == 0U) {
    wave_sums[wave] = partial;
  }
  __syncthreads();
  if (wave == 0U) {
    partial = lane < 8U ? wave_sums[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
      partial += __shfl_down(partial, offset, 32U);
    }
    if (lane == 0U) {
      output[output_index] = float_to_bf16_rne_bits(partial);
    }
  }
}

__device__ __forceinline__ void sllm_matmul_mxfp6_w6a6_block32_body(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  const uint64_t output_index = blockIdx.x;
  if (output_index >= m * n) {
    return;
  }
  const uint64_t row = output_index / n;
  const uint64_t column = output_index - row * n;
  const uint64_t blocks_per_row = k / UINT64_C(32);
  const uint64_t row_bytes = k * UINT64_C(3) / UINT64_C(4);
  const sllm_lowp::BlockScaledView<sllm_lowp::Mxfp6E3Block32> activation_view{
      activation, activation_scales, nullptr, k, row_bytes, blocks_per_row};
  const sllm_lowp::BlockScaledView<sllm_lowp::Mxfp6E3Block32> weight_view{
      weight, weight_scales, nullptr, k, row_bytes, blocks_per_row};
  float partial = 0.0F;
  for (uint64_t inner = threadIdx.x; inner < k; inner += blockDim.x) {
    partial += sllm_lowp::BlockCodec<sllm_lowp::Mxfp6E3Block32>::load(
                   activation_view, row, static_cast<uint32_t>(inner)) *
               sllm_lowp::BlockCodec<sllm_lowp::Mxfp6E3Block32>::load(
                   weight_view, column, static_cast<uint32_t>(inner));
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    partial += __shfl_down(partial, offset, 32U);
  }
  __shared__ float wave_sums[8];
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  if (lane == 0U) {
    wave_sums[wave] = partial;
  }
  __syncthreads();
  if (wave == 0U) {
    partial = lane < 8U ? wave_sums[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
      partial += __shfl_down(partial, offset, 32U);
    }
    if (lane == 0U) {
      output[output_index] = float_to_bf16_rne_bits(partial);
    }
  }
}

#define SLLM_DEFINE_MX_WA_KERNEL(symbol, body)                                 \
  extern "C" __global__ __launch_bounds__(256, 1) void symbol(                 \
      const uint8_t *const activation, const uint8_t *const activation_scales, \
      const uint8_t *const weight, const uint8_t *const weight_scales,         \
      uint16_t *const output, const uint64_t m, const uint64_t k,              \
      const uint64_t n) {                                                      \
    body(activation, activation_scales, weight, weight_scales, output, m, k,   \
         n);                                                                   \
  }

SLLM_DEFINE_MX_WA_KERNEL(sllm_matmul_mxfp8_w8a8_e4m3_block32_decode_v1,
                         sllm_matmul_mxfp8_w8a8_block32_body)
SLLM_DEFINE_MX_WA_KERNEL(sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_v1,
                         sllm_matmul_mxfp8_w8a8_block32_body)
SLLM_DEFINE_MX_WA_KERNEL(sllm_matmul_mxfp6_w6a6_e3m2_block32_decode_v1,
                         sllm_matmul_mxfp6_w6a6_block32_body)
SLLM_DEFINE_MX_WA_KERNEL(sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_v1,
                         sllm_matmul_mxfp6_w6a6_block32_body)

#undef SLLM_DEFINE_MX_WA_KERNEL

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_row8_v2(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t rows_per_workgroup = 8U;
  constexpr uint32_t tile_k = 256U;
  constexpr uint32_t blocks_per_tile = tile_k / 32U;
  __shared__ float weight_tile[tile_k];
  __shared__ float weight_scale_tile[blocks_per_tile];
  __shared__ float activation_scale_tile[rows_per_workgroup][blocks_per_tile];
  const uint64_t column = static_cast<uint64_t>(blockIdx.x) % n;
  const uint64_t row_base =
      (static_cast<uint64_t>(blockIdx.x) / n) * rows_per_workgroup;
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  const uint64_t row = row_base + wave;
  const uint64_t blocks_per_row = k / UINT64_C(32);
  float accumulator = 0.0F;
  for (uint64_t base = 0U; base < k; base += tile_k) {
    if (threadIdx.x < blocks_per_tile) {
      const uint64_t block = base / UINT64_C(32) + threadIdx.x;
      weight_scale_tile[threadIdx.x] =
          block < blocks_per_row
              ? e8m0_to_float(weight_scales[column * blocks_per_row + block])
              : 0.0F;
    }
    if (threadIdx.x < rows_per_workgroup * blocks_per_tile) {
      const uint32_t scale_row = threadIdx.x / blocks_per_tile;
      const uint32_t scale_block = threadIdx.x % blocks_per_tile;
      const uint64_t source_row = row_base + scale_row;
      const uint64_t block = base / UINT64_C(32) + scale_block;
      activation_scale_tile[scale_row][scale_block] =
          source_row < m && block < blocks_per_row
              ? e8m0_to_float(
                    activation_scales[source_row * blocks_per_row + block])
              : 0.0F;
    }
    const uint64_t global_inner = base + threadIdx.x;
    weight_tile[threadIdx.x] = global_inner < k
                                   ? e4m3fn_to_float(__builtin_nontemporal_load(
                                         weight + column * k + global_inner))
                                   : 0.0F;
    __syncthreads();
    if (row < m) {
      const uint32_t valid = static_cast<uint32_t>(
          k - base < tile_k ? k - base : static_cast<uint64_t>(tile_k));
      for (uint32_t offset = lane; offset < valid; offset += wave_width) {
        float term = e4m3fn_to_float(activation[row * k + base + offset]) *
                     activation_scale_tile[wave][offset / 32U];
        term *= weight_tile[offset];
        term *= weight_scale_tile[offset / 32U];
        accumulator += term;
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    accumulator += __shfl_down(accumulator, offset, 32U);
  }
  if (lane == 0U && row < m) {
    output[row * n + column] = float_to_bf16_rne_bits(accumulator);
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_row8_v2(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t rows_per_workgroup = 8U;
  constexpr uint32_t tile_k = 256U;
  constexpr uint32_t blocks_per_tile = tile_k / 32U;
  __shared__ float weight_tile[tile_k];
  __shared__ float weight_scale_tile[blocks_per_tile];
  __shared__ float activation_scale_tile[rows_per_workgroup][blocks_per_tile];
  const uint64_t column = static_cast<uint64_t>(blockIdx.x) % n;
  const uint64_t row_base =
      (static_cast<uint64_t>(blockIdx.x) / n) * rows_per_workgroup;
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  const uint64_t row = row_base + wave;
  const uint64_t blocks_per_row = k / UINT64_C(32);
  const uint64_t row_bytes = k * UINT64_C(3) / UINT64_C(4);
  const uint8_t *const weight_row = weight + column * row_bytes;
  const uint8_t *const activation_row =
      activation + (row < m ? row : 0U) * row_bytes;
  float accumulator = 0.0F;
  for (uint64_t base = 0U; base < k; base += tile_k) {
    if (threadIdx.x < blocks_per_tile) {
      const uint64_t block = base / UINT64_C(32) + threadIdx.x;
      weight_scale_tile[threadIdx.x] =
          block < blocks_per_row
              ? e8m0_to_float(weight_scales[column * blocks_per_row + block])
              : 0.0F;
    }
    if (threadIdx.x < rows_per_workgroup * blocks_per_tile) {
      const uint32_t scale_row = threadIdx.x / blocks_per_tile;
      const uint32_t scale_block = threadIdx.x % blocks_per_tile;
      const uint64_t source_row = row_base + scale_row;
      const uint64_t block = base / UINT64_C(32) + scale_block;
      activation_scale_tile[scale_row][scale_block] =
          source_row < m && block < blocks_per_row
              ? e8m0_to_float(
                    activation_scales[source_row * blocks_per_row + block])
              : 0.0F;
    }
    const uint64_t global_inner = base + threadIdx.x;
    weight_tile[threadIdx.x] =
        global_inner < k
            ? e3m2_to_float(packed_e3m2_at(weight_row, global_inner))
            : 0.0F;
    __syncthreads();
    if (row < m) {
      const uint32_t valid = static_cast<uint32_t>(
          k - base < tile_k ? k - base : static_cast<uint64_t>(tile_k));
      for (uint32_t offset = lane; offset < valid; offset += wave_width) {
        float term =
            e3m2_to_float(packed_e3m2_at(activation_row, base + offset)) *
            activation_scale_tile[wave][offset / 32U];
        term *= weight_tile[offset];
        term *= weight_scale_tile[offset / 32U];
        accumulator += term;
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    accumulator += __shfl_down(accumulator, offset, 32U);
  }
  if (lane == 0U && row < m) {
    output[row * n + column] = float_to_bf16_rne_bits(accumulator);
  }
}

struct Mxfp8MmqFormat {
  __device__ __forceinline__ static uint64_t row_bytes(const uint64_t k) {
    return k;
  }

  __device__ __forceinline__ static float
  load_activation(const uint8_t *const row, const uint64_t index) {
    return sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode_mx_value_plane(
        row[index]);
  }

  __device__ __forceinline__ static float load_weight(const uint8_t *const row,
                                                      const uint64_t index) {
    return decode_weight_byte(__builtin_nontemporal_load(row + index));
  }

  __device__ __forceinline__ static float
  decode_weight_byte(const uint8_t bits) {
    return sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode_mx_value_plane(
        bits);
  }
};

struct Mxfp6MmqFormat {
  __device__ __forceinline__ static uint64_t row_bytes(const uint64_t k) {
    return k * UINT64_C(3) / UINT64_C(4);
  }

  __device__ __forceinline__ static float
  load_activation(const uint8_t *const row, const uint64_t index) {
    return e3m2_to_float(packed_e3m2_at(row, index));
  }

  __device__ __forceinline__ static float load_weight(const uint8_t *const row,
                                                      const uint64_t index) {
    return e3m2_to_float(packed_e3m2_at(row, index));
  }
};

// Keep packed-value ingress independent from the MMQ arithmetic schedule.
// The scalar policy remains the format-generic default used by MXFP8 and
// MXFP6.  The vector policy is instantiated only for byte-addressed MXFP8;
// future packed formats can provide their own ingress policy without cloning
// the row/column/K decomposition.
struct MmqScalarWeightIngress {
  static constexpr uint32_t values_per_load = 1U;

  template <typename Format>
  __device__ __forceinline__ static void
  stage(const uint8_t *const row, const uint64_t index, float *const output) {
    output[0] = Format::load_weight(row, index);
  }
};

struct Mxfp8MmqVector32WeightIngress {
  static constexpr uint32_t values_per_load = 4U;

  template <typename Format>
  __device__ __forceinline__ static void
  stage(const uint8_t *const row, const uint64_t index, float *const output) {
    const uint32_t packed = __builtin_nontemporal_load(
        reinterpret_cast<const uint32_t *>(row + index));
#pragma unroll
    for (uint32_t byte = 0U; byte < values_per_load; ++byte) {
      output[byte] = Format::decode_weight_byte(
          static_cast<uint8_t>(packed >> (byte * 8U)));
    }
  }
};

// This candidate borrows llama.cpp MMQ's multi-row/multi-column/K-tile
// decomposition, but intentionally retains sLLM's packed MX values, E8M0
// scales, FP32 accumulation, and row8 reduction order.  In particular, it
// does not introduce the llama.cpp Q8_1 activation or integer dot path.
template <typename Format, uint32_t Columns,
          typename WeightIngress = MmqScalarWeightIngress,
          bool RegisterBlockScales = false>
__device__ __forceinline__ void sllm_matmul_mx_wa_mmq_columns_body(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t rows_per_workgroup = 8U;
  constexpr uint32_t tile_k = 256U;
  constexpr uint32_t blocks_per_tile = tile_k / 32U;
  constexpr uint32_t ingress_values = WeightIngress::values_per_load;
  static_assert(tile_k % ingress_values == 0U);
  __shared__ float weight_tile[Columns][tile_k];
  __shared__ float weight_scale_tile[Columns][blocks_per_tile];
  __shared__ float activation_scale_tile[rows_per_workgroup][blocks_per_tile];
  const uint64_t column_tiles =
      (n + static_cast<uint64_t>(Columns) - 1U) / Columns;
  const uint64_t tile_index = static_cast<uint64_t>(blockIdx.x);
  const uint64_t column_base = (tile_index % column_tiles) * Columns;
  const uint64_t row_base = (tile_index / column_tiles) * rows_per_workgroup;
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  const uint64_t row = row_base + wave;
  const uint64_t blocks_per_row = k / UINT64_C(32);
  const uint64_t row_bytes = Format::row_bytes(k);
  const uint8_t *const activation_row =
      activation + (row < m ? row : 0U) * row_bytes;
  float accumulators[Columns] = {};
  for (uint64_t base = 0U; base < k; base += tile_k) {
    for (uint32_t index = threadIdx.x; index < Columns * blocks_per_tile;
         index += blockDim.x) {
      const uint32_t local_column = index / blocks_per_tile;
      const uint32_t scale_block = index % blocks_per_tile;
      const uint64_t column = column_base + local_column;
      const uint64_t block = base / UINT64_C(32) + scale_block;
      weight_scale_tile[local_column][scale_block] =
          column < n && block < blocks_per_row
              ? e8m0_to_float(weight_scales[column * blocks_per_row + block])
              : 0.0F;
    }
    if (threadIdx.x < rows_per_workgroup * blocks_per_tile) {
      const uint32_t scale_row = threadIdx.x / blocks_per_tile;
      const uint32_t scale_block = threadIdx.x % blocks_per_tile;
      const uint64_t source_row = row_base + scale_row;
      const uint64_t block = base / UINT64_C(32) + scale_block;
      activation_scale_tile[scale_row][scale_block] =
          source_row < m && block < blocks_per_row
              ? e8m0_to_float(
                    activation_scales[source_row * blocks_per_row + block])
              : 0.0F;
    }
    constexpr uint32_t ingress_groups_per_column = tile_k / ingress_values;
    for (uint32_t index = threadIdx.x;
         index < Columns * ingress_groups_per_column; index += blockDim.x) {
      const uint32_t local_column = index / ingress_groups_per_column;
      const uint32_t group = index % ingress_groups_per_column;
      const uint32_t offset = group * ingress_values;
      const uint64_t column = column_base + local_column;
      const uint64_t global_inner = base + offset;
      if (column < n && global_inner + ingress_values <= k) {
        WeightIngress::template stage<Format>(
            weight + column * row_bytes, global_inner,
            &weight_tile[local_column][offset]);
      } else {
#pragma unroll
        for (uint32_t value = 0U; value < ingress_values; ++value) {
          weight_tile[local_column][offset + value] = 0.0F;
        }
      }
    }
    __syncthreads();
    if (row < m) {
      const uint32_t valid = static_cast<uint32_t>(
          k - base < tile_k ? k - base : static_cast<uint64_t>(tile_k));
      if constexpr (RegisterBlockScales) {
#pragma unroll
        for (uint32_t scale_block = 0U; scale_block < blocks_per_tile;
             ++scale_block) {
          const uint32_t offset = scale_block * wave_width + lane;
          if (offset >= valid) {
            continue;
          }
          const float activation_scale =
              activation_scale_tile[wave][scale_block];
          const float activation_value =
              Format::load_activation(activation_row, base + offset) *
              activation_scale;
#pragma unroll
          for (uint32_t local_column = 0U; local_column < Columns;
               ++local_column) {
            const float weight_scale =
                weight_scale_tile[local_column][scale_block];
            float term = activation_value * weight_tile[local_column][offset];
            term *= weight_scale;
            accumulators[local_column] += term;
          }
        }
      } else {
        for (uint32_t offset = lane; offset < valid; offset += wave_width) {
          const float activation_value =
              Format::load_activation(activation_row, base + offset) *
              activation_scale_tile[wave][offset / 32U];
#pragma unroll
          for (uint32_t local_column = 0U; local_column < Columns;
               ++local_column) {
            float term = activation_value * weight_tile[local_column][offset];
            term *= weight_scale_tile[local_column][offset / 32U];
            accumulators[local_column] += term;
          }
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
#pragma unroll
    for (uint32_t local_column = 0U; local_column < Columns; ++local_column) {
      accumulators[local_column] +=
          __shfl_down(accumulators[local_column], offset, 32U);
    }
  }
  if (lane == 0U && row < m) {
#pragma unroll
    for (uint32_t local_column = 0U; local_column < Columns; ++local_column) {
      const uint64_t column = column_base + local_column;
      if (column < n) {
        output[row * n + column] =
            float_to_bf16_rne_bits(accumulators[local_column]);
      }
    }
  }
}

#define SLLM_DEFINE_MX_WA_MMQ_COLUMNS_KERNEL(symbol, format, columns)          \
  extern "C" __global__ __launch_bounds__(256, 1) void symbol(                 \
      const uint8_t *const activation, const uint8_t *const activation_scales, \
      const uint8_t *const weight, const uint8_t *const weight_scales,         \
      uint16_t *const output, const uint64_t m, const uint64_t k,              \
      const uint64_t n) {                                                      \
    sllm_matmul_mx_wa_mmq_columns_body<format, columns>(                       \
        activation, activation_scales, weight, weight_scales, output, m, k,    \
        n);                                                                    \
  }

SLLM_DEFINE_MX_WA_MMQ_COLUMNS_KERNEL(
    sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_mmq_col4_v4, Mxfp8MmqFormat, 4U)
SLLM_DEFINE_MX_WA_MMQ_COLUMNS_KERNEL(
    sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_mmq_col8_v4, Mxfp8MmqFormat, 8U)
SLLM_DEFINE_MX_WA_MMQ_COLUMNS_KERNEL(sllm_mxfp8_w8a8_gfx1030_mmq_col16_v1,
                                     Mxfp8MmqFormat, 16U)
SLLM_DEFINE_MX_WA_MMQ_COLUMNS_KERNEL(sllm_mxfp8_w8a8_gfx1030_mmq_col32_v1,
                                     Mxfp8MmqFormat, 32U)
SLLM_DEFINE_MX_WA_MMQ_COLUMNS_KERNEL(
    sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_mmq_col4_v4, Mxfp6MmqFormat, 4U)
SLLM_DEFINE_MX_WA_MMQ_COLUMNS_KERNEL(
    sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_mmq_col8_v4, Mxfp6MmqFormat, 8U)

#undef SLLM_DEFINE_MX_WA_MMQ_COLUMNS_KERNEL

#define SLLM_DEFINE_MXFP8_GFX1030_MMQ_PHASE69_KERNEL(symbol, ingress,          \
                                                     register_scales)          \
  extern "C" __global__ __launch_bounds__(256, 1) void symbol(                 \
      const uint8_t *const activation, const uint8_t *const activation_scales, \
      const uint8_t *const weight, const uint8_t *const weight_scales,         \
      uint16_t *const output, const uint64_t m, const uint64_t k,              \
      const uint64_t n) {                                                      \
    sllm_matmul_mx_wa_mmq_columns_body<Mxfp8MmqFormat, 8U, ingress,            \
                                       register_scales>(                       \
        activation, activation_scales, weight, weight_scales, output, m, k,    \
        n);                                                                    \
  }

SLLM_DEFINE_MXFP8_GFX1030_MMQ_PHASE69_KERNEL(
    sllm_mxfp8_w8a8_gfx1030_mmq_col8_regscale_v1, MmqScalarWeightIngress, true)
SLLM_DEFINE_MXFP8_GFX1030_MMQ_PHASE69_KERNEL(
    sllm_mxfp8_w8a8_gfx1030_mmq_col8_vector32_v1, Mxfp8MmqVector32WeightIngress,
    false)
SLLM_DEFINE_MXFP8_GFX1030_MMQ_PHASE69_KERNEL(
    sllm_mxfp8_w8a8_gfx1030_mmq_col8_regscale_vector32_v1,
    Mxfp8MmqVector32WeightIngress, true)

#undef SLLM_DEFINE_MXFP8_GFX1030_MMQ_PHASE69_KERNEL

#if defined(SLLM_MATMUL_HAS_GFX12_ROCWMMA)
static_assert(sizeof(rocwmma::float8_t) == sizeof(uint8_t));
#endif

// Eight independent waves cover a 128-row output tile. The raw OCP E4M3 value
// planes are staged as bytes and consumed directly by gfx12 FP8 WMMA. E8M0
// block-32 scales remain separate: each wave keeps its unscaled 16x16 WMMA
// contributions in registers, transforms them to row-major lane layout once
// per K block, and applies the row/column scale pair while accumulating. No
// whole-tensor BF16/FP32 expansion or contribution scratch tile is created;
// zero-padded LDS tiles make both M and N tails fail-safe.
template <uint32_t ColumnTiles>
__device__ __forceinline__ void sllm_matmul_mxfp8_w8a8_gfx1201_wmma_body(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
#if defined(__gfx1201__)
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t waves_per_workgroup = 8U;
  constexpr uint32_t tile_m = 16U;
  constexpr uint32_t tile_n = 16U;
  constexpr uint32_t column_tiles = ColumnTiles;
  constexpr uint32_t block_k = 32U;
  constexpr uint32_t tile_values = tile_m * block_k;
  constexpr uint32_t output_values = tile_m * tile_n;
  __shared__ rocwmma::float8_t activation_tile[waves_per_workgroup]
                                              [tile_values];
  __shared__ rocwmma::float8_t weight_tile[column_tiles][tile_values];
  __shared__ float activation_scale_tile[waves_per_workgroup][tile_m];
  __shared__ float weight_scale_tile[column_tiles * tile_n];

  using AFragment = rocwmma::fragment<rocwmma::matrix_a, tile_m, tile_n, tile_m,
                                      rocwmma::float8_t, rocwmma::row_major>;
  using BFragment = rocwmma::fragment<rocwmma::matrix_b, tile_m, tile_n, tile_m,
                                      rocwmma::float8_t, rocwmma::col_major>;
  using AccumulatorFragment =
      rocwmma::fragment<rocwmma::accumulator, tile_m, tile_n, tile_m, float>;

  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (wave_width - 1U);
  const uint32_t wave = thread / wave_width;
  const uint64_t row_group_base =
      static_cast<uint64_t>(blockIdx.y) *
      sllm_matmul_kernel::kMxfp8W8A8PrefillWmmaRowsPerWorkgroup;
  const uint64_t row_tile_base =
      row_group_base + static_cast<uint64_t>(wave) * tile_m;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * column_tiles * tile_n;
  const uint64_t blocks_per_row = k / block_k;
  float accumulators[column_tiles][output_values / wave_width] = {};

  for (uint64_t block = 0U; block < blocks_per_row; ++block) {
    const uint64_t inner_base = block * block_k;
    auto *const activation_raw = reinterpret_cast<uint8_t *>(activation_tile);
    auto *const weight_raw = reinterpret_cast<uint8_t *>(weight_tile);

    for (uint32_t index = thread; index < waves_per_workgroup * tile_values;
         index += blockDim.x) {
      const uint32_t source_wave = index / tile_values;
      const uint32_t wave_index = index - source_wave * tile_values;
      const uint32_t local_row = wave_index / block_k;
      const uint32_t local_inner = wave_index - local_row * block_k;
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      activation_raw[index] =
          row < m ? activation[row * k + inner_base + local_inner] : 0U;
    }
    for (uint32_t index = thread; index < column_tiles * tile_values;
         index += blockDim.x) {
      const uint32_t column_tile = index / tile_values;
      const uint32_t tile_index = index - column_tile * tile_values;
      const uint32_t local_column = tile_index / block_k;
      const uint32_t local_inner = tile_index - local_column * block_k;
      const uint64_t column = column_base + column_tile * tile_n + local_column;
      weight_raw[index] =
          column < n ? __builtin_nontemporal_load(weight + column * k +
                                                  inner_base + local_inner)
                     : 0U;
    }
    if (thread < waves_per_workgroup * tile_m) {
      const uint32_t source_wave = thread / tile_m;
      const uint32_t local_row = thread - source_wave * tile_m;
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      activation_scale_tile[source_wave][local_row] =
          row < m
              ? e8m0_to_float(activation_scales[row * blocks_per_row + block])
              : 0.0F;
    }
    if (thread < column_tiles * tile_n) {
      const uint64_t column = column_base + thread;
      weight_scale_tile[thread] =
          column < n
              ? e8m0_to_float(weight_scales[column * blocks_per_row + block])
              : 0.0F;
    }
    __syncthreads();

    AFragment activation_fragment;
    AccumulatorFragment contributions[column_tiles];
#pragma unroll
    for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
      rocwmma::fill_fragment(contributions[column_tile], 0.0F);
    }
    rocwmma::load_matrix_sync(activation_fragment, activation_tile[wave],
                              block_k);
#pragma unroll
    for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
      BFragment weight_fragment;
      rocwmma::load_matrix_sync(weight_fragment, weight_tile[column_tile],
                                block_k);
      rocwmma::mma_sync(contributions[column_tile], activation_fragment,
                        weight_fragment, contributions[column_tile]);
    }
    rocwmma::load_matrix_sync(activation_fragment,
                              activation_tile[wave] + tile_m, block_k);
#pragma unroll
    for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
      BFragment weight_fragment;
      rocwmma::load_matrix_sync(weight_fragment,
                                weight_tile[column_tile] + tile_m, block_k);
      rocwmma::mma_sync(contributions[column_tile], activation_fragment,
                        weight_fragment, contributions[column_tile]);
    }
#pragma unroll
    for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
      const auto contribution_row_major =
          rocwmma::apply_data_layout<rocwmma::row_major>(
              contributions[column_tile]);
#pragma unroll
      for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
        const uint32_t local_row =
            (lane / tile_n) * (output_values / wave_width) + slot;
        const uint32_t local_column = lane % tile_n;
        float term = contribution_row_major[slot] *
                     activation_scale_tile[wave][local_row];
        term *= weight_scale_tile[column_tile * tile_n + local_column];
        accumulators[column_tile][slot] += term;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
#pragma unroll
    for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
      const uint32_t local_row =
          (lane / tile_n) * (output_values / wave_width) + slot;
      const uint32_t local_column = lane % tile_n;
      const uint64_t row = row_tile_base + local_row;
      const uint64_t column = column_base + column_tile * tile_n + local_column;
      if (row < m && column < n) {
        output[row * n + column] =
            float_to_bf16_rne_bits(accumulators[column_tile][slot]);
      }
    }
  }
#else
  (void)activation;
  (void)activation_scales;
  (void)weight;
  (void)weight_scales;
  (void)output;
  (void)m;
  (void)k;
  (void)n;
#endif
}

// Phase 64 candidates keep the Phase 63 arithmetic order and output mapping
// fixed while varying only workgroup height, the physical LDS row stride, or
// the fragment load source. LdsStride=33 is a public-rocWMMA-compatible
// bank-conflict probe: rocWMMA does not expose a custom XOR-addressed LDS
// accessor, so padding is used to perturb the same bank mapping without
// depending on its private fragment layout. DirectActivation and DirectWeight
// bypass their respective value tiles; the small E8M0 scale tiles remain
// shared. DirectActivation is dispatched only for complete 128-row groups.
template <uint32_t WavesPerWorkgroup, uint32_t ColumnTiles, uint32_t LdsStride,
          bool DirectActivation, bool DirectWeight>
__device__ __forceinline__ void
sllm_matmul_mxfp8_w8a8_gfx1201_wmma_phase64_body(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
#if defined(__gfx1201__)
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t waves_per_workgroup = WavesPerWorkgroup;
  constexpr uint32_t tile_m = 16U;
  constexpr uint32_t tile_n = 16U;
  constexpr uint32_t column_tiles = ColumnTiles;
  constexpr uint32_t block_k = 32U;
  constexpr uint32_t lds_stride = LdsStride;
  constexpr uint32_t rows_per_workgroup = waves_per_workgroup * tile_m;
  constexpr uint32_t activation_lds_values =
      DirectActivation ? 1U : waves_per_workgroup * tile_m * lds_stride;
  constexpr uint32_t weight_lds_values =
      DirectWeight ? 1U : column_tiles * tile_m * lds_stride;
  constexpr uint32_t output_values = tile_m * tile_n;
  static_assert(waves_per_workgroup == 4U || waves_per_workgroup == 8U);
  static_assert(lds_stride >= block_k);

  __shared__ rocwmma::float8_t activation_tile[activation_lds_values];
  __shared__ rocwmma::float8_t weight_tile[weight_lds_values];
  __shared__ float activation_scale_tile[waves_per_workgroup][tile_m];
  __shared__ float weight_scale_tile[column_tiles * tile_n];

  using AFragment = rocwmma::fragment<rocwmma::matrix_a, tile_m, tile_n, tile_m,
                                      rocwmma::float8_t, rocwmma::row_major>;
  using BFragment = rocwmma::fragment<rocwmma::matrix_b, tile_m, tile_n, tile_m,
                                      rocwmma::float8_t, rocwmma::col_major>;
  using AccumulatorFragment =
      rocwmma::fragment<rocwmma::accumulator, tile_m, tile_n, tile_m, float>;

  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (wave_width - 1U);
  const uint32_t wave = thread / wave_width;
  const uint64_t row_group_base =
      static_cast<uint64_t>(blockIdx.y) * rows_per_workgroup;
  const uint64_t row_tile_base =
      row_group_base + static_cast<uint64_t>(wave) * tile_m;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * column_tiles * tile_n;
  const uint64_t blocks_per_row = k / block_k;
  float accumulators[column_tiles][output_values / wave_width] = {};

  for (uint64_t block = 0U; block < blocks_per_row; ++block) {
    const uint64_t inner_base = block * block_k;
    if constexpr (!DirectActivation) {
      auto *const activation_raw = reinterpret_cast<uint8_t *>(activation_tile);
      constexpr uint32_t activation_logical_values =
          waves_per_workgroup * tile_m * block_k;
      for (uint32_t index = thread; index < activation_logical_values;
           index += blockDim.x) {
        const uint32_t source_wave = index / (tile_m * block_k);
        const uint32_t wave_index = index - source_wave * tile_m * block_k;
        const uint32_t local_row = wave_index / block_k;
        const uint32_t local_inner = wave_index - local_row * block_k;
        const uint64_t row = row_group_base +
                             static_cast<uint64_t>(source_wave) * tile_m +
                             local_row;
        activation_raw[(source_wave * tile_m + local_row) * lds_stride +
                       local_inner] =
            row < m ? activation[row * k + inner_base + local_inner] : 0U;
      }
    }
    if constexpr (!DirectWeight) {
      auto *const weight_raw = reinterpret_cast<uint8_t *>(weight_tile);
      constexpr uint32_t weight_logical_values =
          column_tiles * tile_m * block_k;
      for (uint32_t index = thread; index < weight_logical_values;
           index += blockDim.x) {
        const uint32_t column_tile = index / (tile_m * block_k);
        const uint32_t tile_index = index - column_tile * tile_m * block_k;
        const uint32_t local_column = tile_index / block_k;
        const uint32_t local_inner = tile_index - local_column * block_k;
        const uint64_t column =
            column_base + column_tile * tile_n + local_column;
        weight_raw[(column_tile * tile_m + local_column) * lds_stride +
                   local_inner] =
            column < n ? __builtin_nontemporal_load(weight + column * k +
                                                    inner_base + local_inner)
                       : 0U;
      }
    }
    if (thread < waves_per_workgroup * tile_m) {
      const uint32_t source_wave = thread / tile_m;
      const uint32_t local_row = thread - source_wave * tile_m;
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      activation_scale_tile[source_wave][local_row] =
          row < m
              ? e8m0_to_float(activation_scales[row * blocks_per_row + block])
              : 0.0F;
    }
    if (thread < column_tiles * tile_n) {
      const uint64_t column = column_base + thread;
      weight_scale_tile[thread] =
          column < n
              ? e8m0_to_float(weight_scales[column * blocks_per_row + block])
              : 0.0F;
    }
    __syncthreads();

    AFragment activation_fragment;
    AccumulatorFragment contributions[column_tiles];
#pragma unroll
    for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
      rocwmma::fill_fragment(contributions[column_tile], 0.0F);
    }
    if constexpr (DirectActivation) {
      const auto *const activation_matrix =
          reinterpret_cast<const rocwmma::float8_t *>(
              activation + row_tile_base * k + inner_base);
      rocwmma::load_matrix_sync(activation_fragment, activation_matrix,
                                static_cast<uint32_t>(k));
    } else {
      const auto *const activation_wave_tile =
          activation_tile + wave * tile_m * lds_stride;
      rocwmma::load_matrix_sync(activation_fragment, activation_wave_tile,
                                lds_stride);
    }
#pragma unroll
    for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
      BFragment weight_fragment;
      if constexpr (DirectWeight) {
        const uint64_t column = column_base + column_tile * tile_n;
        const auto *const weight_matrix =
            reinterpret_cast<const rocwmma::float8_t *>(weight + column * k +
                                                        inner_base);
        rocwmma::load_matrix_sync(weight_fragment, weight_matrix,
                                  static_cast<uint32_t>(k));
      } else {
        rocwmma::load_matrix_sync(
            weight_fragment, weight_tile + column_tile * tile_m * lds_stride,
            lds_stride);
      }
      rocwmma::mma_sync(contributions[column_tile], activation_fragment,
                        weight_fragment, contributions[column_tile]);
    }
    if constexpr (DirectActivation) {
      const auto *const activation_matrix =
          reinterpret_cast<const rocwmma::float8_t *>(
              activation + row_tile_base * k + inner_base + tile_m);
      rocwmma::load_matrix_sync(activation_fragment, activation_matrix,
                                static_cast<uint32_t>(k));
    } else {
      const auto *const activation_wave_tile =
          activation_tile + wave * tile_m * lds_stride;
      rocwmma::load_matrix_sync(activation_fragment,
                                activation_wave_tile + tile_m, lds_stride);
    }
#pragma unroll
    for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
      BFragment weight_fragment;
      if constexpr (DirectWeight) {
        const uint64_t column = column_base + column_tile * tile_n;
        const auto *const weight_matrix =
            reinterpret_cast<const rocwmma::float8_t *>(weight + column * k +
                                                        inner_base + tile_m);
        rocwmma::load_matrix_sync(weight_fragment, weight_matrix,
                                  static_cast<uint32_t>(k));
      } else {
        rocwmma::load_matrix_sync(
            weight_fragment,
            weight_tile + column_tile * tile_m * lds_stride + tile_m,
            lds_stride);
      }
      rocwmma::mma_sync(contributions[column_tile], activation_fragment,
                        weight_fragment, contributions[column_tile]);
    }
#pragma unroll
    for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
      const auto contribution_row_major =
          rocwmma::apply_data_layout<rocwmma::row_major>(
              contributions[column_tile]);
#pragma unroll
      for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
        const uint32_t local_row =
            (lane / tile_n) * (output_values / wave_width) + slot;
        const uint32_t local_column = lane % tile_n;
        float term = contribution_row_major[slot] *
                     activation_scale_tile[wave][local_row];
        term *= weight_scale_tile[column_tile * tile_n + local_column];
        accumulators[column_tile][slot] += term;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
#pragma unroll
    for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
      const uint32_t local_row =
          (lane / tile_n) * (output_values / wave_width) + slot;
      const uint32_t local_column = lane % tile_n;
      const uint64_t row = row_tile_base + local_row;
      const uint64_t column = column_base + column_tile * tile_n + local_column;
      if (row < m && column < n) {
        output[row * n + column] =
            float_to_bf16_rne_bits(accumulators[column_tile][slot]);
      }
    }
  }
#else
  (void)activation;
  (void)activation_scales;
  (void)weight;
  (void)weight_scales;
  (void)output;
  (void)m;
  (void)k;
  (void)n;
#endif
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_wmma128x16x32_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mxfp8_w8a8_gfx1201_wmma_body<1U>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_wmma128x64x32_v2(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mxfp8_w8a8_gfx1201_wmma_body<4U>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(128, 1) void sllm_mxfp8_w8a8_gfx1201_wmma64x64_4w_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mxfp8_w8a8_gfx1201_wmma_phase64_body<4U, 4U, 32U, false, false>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_mxfp8_w8a8_gfx1201_wmma128x64_pad33_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mxfp8_w8a8_gfx1201_wmma_phase64_body<8U, 4U, 33U, false, false>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_mxfp8_w8a8_gfx1201_wmma128x64_direct_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mxfp8_w8a8_gfx1201_wmma_phase64_body<8U, 4U, 32U, false, true>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_mxfp8_w8a8_gfx1201_wmma128x64_adirect_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mxfp8_w8a8_gfx1201_wmma_phase64_body<8U, 4U, 32U, true, false>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_mxfp8_w8a8_gfx1201_wmma128x64_bdirect_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mxfp8_w8a8_gfx1201_wmma_phase64_body<8U, 4U, 32U, true, true>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_mxfp8_w8a8_gfx1201_wmma128x128_bdirect_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mxfp8_w8a8_gfx1201_wmma_phase64_body<8U, 8U, 32U, true, true>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_tiled16_v3(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t tile = 16U;
  __shared__ float activation_tile[tile][tile];
  __shared__ float weight_tile[tile][tile];
  const uint32_t local_row = threadIdx.y;
  const uint32_t local_column = threadIdx.x;
  const uint64_t row = static_cast<uint64_t>(blockIdx.y) * tile + local_row;
  const uint64_t column =
      static_cast<uint64_t>(blockIdx.x) * tile + local_column;
  const uint64_t blocks_per_row = k / UINT64_C(32);
  float accumulator = 0.0F;
  for (uint64_t base = 0U; base < k; base += tile) {
    const uint64_t activation_inner = base + local_column;
    const uint64_t weight_inner = base + local_row;
    if (row < m && activation_inner < k) {
      activation_tile[local_row][local_column] =
          e4m3fn_to_float(activation[row * k + activation_inner]) *
          e8m0_to_float(activation_scales[row * blocks_per_row +
                                          activation_inner / UINT64_C(32)]);
    } else {
      activation_tile[local_row][local_column] = 0.0F;
    }
    if (column < n && weight_inner < k) {
      weight_tile[local_row][local_column] =
          e4m3fn_to_float(
              __builtin_nontemporal_load(weight + column * k + weight_inner)) *
          e8m0_to_float(weight_scales[column * blocks_per_row +
                                      weight_inner / UINT64_C(32)]);
    } else {
      weight_tile[local_row][local_column] = 0.0F;
    }
    __syncthreads();
#pragma unroll
    for (uint32_t inner = 0U; inner != tile; ++inner) {
      accumulator +=
          activation_tile[local_row][inner] * weight_tile[inner][local_column];
    }
    __syncthreads();
  }
  if (row < m && column < n) {
    output[row * n + column] = float_to_bf16_rne_bits(accumulator);
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_tiled16_v3(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t tile = 16U;
  __shared__ float activation_tile[tile][tile];
  __shared__ float weight_tile[tile][tile];
  const uint32_t local_row = threadIdx.y;
  const uint32_t local_column = threadIdx.x;
  const uint64_t row = static_cast<uint64_t>(blockIdx.y) * tile + local_row;
  const uint64_t column =
      static_cast<uint64_t>(blockIdx.x) * tile + local_column;
  const uint64_t blocks_per_row = k / UINT64_C(32);
  const uint64_t row_bytes = k * UINT64_C(3) / UINT64_C(4);
  float accumulator = 0.0F;
  for (uint64_t base = 0U; base < k; base += tile) {
    const uint64_t activation_inner = base + local_column;
    const uint64_t weight_inner = base + local_row;
    if (row < m && activation_inner < k) {
      activation_tile[local_row][local_column] =
          e3m2_to_float(
              packed_e3m2_at(activation + row * row_bytes, activation_inner)) *
          e8m0_to_float(activation_scales[row * blocks_per_row +
                                          activation_inner / UINT64_C(32)]);
    } else {
      activation_tile[local_row][local_column] = 0.0F;
    }
    if (column < n && weight_inner < k) {
      weight_tile[local_row][local_column] =
          e3m2_to_float(
              packed_e3m2_at(weight + column * row_bytes, weight_inner)) *
          e8m0_to_float(weight_scales[column * blocks_per_row +
                                      weight_inner / UINT64_C(32)]);
    } else {
      weight_tile[local_row][local_column] = 0.0F;
    }
    __syncthreads();
#pragma unroll
    for (uint32_t inner = 0U; inner != tile; ++inner) {
      accumulator +=
          activation_tile[local_row][inner] * weight_tile[inner][local_column];
    }
    __syncthreads();
  }
  if (row < m && column < n) {
    output[row * n + column] = float_to_bf16_rne_bits(accumulator);
  }
}

#pragma clang fp contract(off)
extern "C" __global__ __launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  const uint64_t output_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                                static_cast<uint64_t>(threadIdx.x);
  const uint64_t output_elements = m * n;
  if (output_index < output_elements) {
    const uint64_t row = output_index / n;
    const uint64_t column = output_index - row * n;
    float accumulator = 0.0F;
    for (uint64_t reduction = 0U; reduction != k; ++reduction) {
      const float activation_value =
          bf16_to_float(activation[row * k + reduction]);
      const float weight_value = bf16_to_float(weight[column * k + reduction]);
      accumulator += activation_value * weight_value;
    }
    output[output_index] = float_to_bf16_rne_bits(accumulator);
  }
}

// Row-major [M,K] x transposed row-major [N,K].  A 16x16 output tile shares
// both input tiles, eliminating the baseline kernel's redundant global loads.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_tiled16_v2(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t tile = 16U;
  __shared__ uint16_t activation_tile[tile][tile];
  __shared__ uint16_t weight_tile[tile][tile];
  const uint32_t local_row = threadIdx.y;
  const uint32_t local_column = threadIdx.x;
  const uint64_t row = static_cast<uint64_t>(blockIdx.y) * tile + local_row;
  const uint64_t column =
      static_cast<uint64_t>(blockIdx.x) * tile + local_column;
  float accumulator = 0.0F;
  for (uint64_t base = 0U; base < k; base += tile) {
    const uint64_t activation_k = base + local_column;
    const uint64_t weight_k = base + local_row;
    activation_tile[local_row][local_column] =
        row < m && activation_k < k ? activation[row * k + activation_k] : 0U;
    weight_tile[local_row][local_column] =
        column < n && weight_k < k ? weight[column * k + weight_k] : 0U;
    __syncthreads();
#pragma unroll
    for (uint32_t inner = 0U; inner != tile; ++inner) {
      accumulator += bf16_to_float(activation_tile[local_row][inner]) *
                     bf16_to_float(weight_tile[inner][local_column]);
    }
    __syncthreads();
  }
  if (row < m && column < n) {
    output[row * n + column] = float_to_bf16_rne_bits(accumulator);
  }
}

// Decode is a matrix-vector product. One workgroup owns one output column and
// reduces K cooperatively; this avoids launching mostly idle 16x16 tiles.
//
// The paired BF16 loads and two-level wave reduction are adapted from the
// floating MMVF organization in llama.cpp mmvf.cu at fixed commit
// f5919bf458ef190468b5c329bb293f8a54a1e69c. The ggml tensor/runtime and
// fusion machinery are deliberately not imported; this kernel retains sLLM's
// BF16 input/output and FP32 accumulation contract.
template <uint32_t WaveWidth, uint32_t WaveCount>
__device__ __forceinline__ void
matmul_bf16_decode_body(const uint16_t *const activation,
                        const uint16_t *const weight, uint16_t *const output,
                        const uint64_t k, const uint64_t n,
                        const uint64_t column) {
  if (column >= n) {
    return;
  }
  float partial = 0.0F;
  const uint16_t *const weight_row = weight + column * k;
  const bool paired =
      (k & UINT64_C(1)) == 0U && ((reinterpret_cast<uintptr_t>(activation) |
                                   reinterpret_cast<uintptr_t>(weight_row)) &
                                  static_cast<uintptr_t>(3U)) == 0U;
  if (paired) {
    const auto *const activation_pairs =
        reinterpret_cast<const uint32_t *>(activation);
    const auto *const weight_pairs =
        reinterpret_cast<const uint32_t *>(weight_row);
    const uint64_t pair_count = k / 2U;
    for (uint64_t pair = threadIdx.x; pair < pair_count; pair += blockDim.x) {
      const uint32_t activation_pair = activation_pairs[pair];
      const uint32_t weight_pair =
          __builtin_nontemporal_load(weight_pairs + pair);
      partial += bf16_to_float(static_cast<uint16_t>(activation_pair)) *
                 bf16_to_float(static_cast<uint16_t>(weight_pair));
      partial += bf16_to_float(static_cast<uint16_t>(activation_pair >> 16U)) *
                 bf16_to_float(static_cast<uint16_t>(weight_pair >> 16U));
    }
  } else {
    for (uint64_t reduction = threadIdx.x; reduction < k;
         reduction += blockDim.x) {
      partial += bf16_to_float(activation[reduction]) *
                 bf16_to_float(weight_row[reduction]);
    }
  }

#pragma unroll
  for (uint32_t offset = WaveWidth / 2U; offset != 0U; offset >>= 1U) {
    partial += __shfl_down(partial, offset, WaveWidth);
  }
  __shared__ float wave_sums[WaveCount];
  const uint32_t lane = threadIdx.x % WaveWidth;
  const uint32_t wave = threadIdx.x / WaveWidth;
  if (lane == 0U) {
    wave_sums[wave] = partial;
  }
  __syncthreads();
  if (wave == 0U) {
    partial = lane < WaveCount ? wave_sums[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = WaveWidth / 2U; offset != 0U; offset >>= 1U) {
      partial += __shfl_down(partial, offset, WaveWidth);
    }
    if (lane == 0U) {
      output[column] = float_to_bf16_rne_bits(partial);
    }
  }
}

template <uint32_t WaveWidth, uint32_t WaveCount>
__device__ __forceinline__ void matmul_bf16_serial_rows_body(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const uint64_t row_start = 0U,
    const uint64_t column_override = UINT64_MAX) {
  constexpr uint32_t max_rows = 8U;
  const uint64_t column = column_override == UINT64_MAX
                              ? static_cast<uint64_t>(blockIdx.x)
                              : column_override;
  if (column >= n || m == 0U || m > max_rows) {
    return;
  }
  float partial[max_rows] = {};
  const uint16_t *const weight_row = weight + column * k;
  const uint16_t *const activation_start = activation + row_start * k;
  const bool paired = (k & UINT64_C(1)) == 0U &&
                      ((reinterpret_cast<uintptr_t>(activation_start) |
                        reinterpret_cast<uintptr_t>(weight_row)) &
                       static_cast<uintptr_t>(3U)) == 0U;
  if (paired) {
    const auto *const weight_pairs =
        reinterpret_cast<const uint32_t *>(weight_row);
    const uint64_t pair_count = k / 2U;
    for (uint64_t pair = threadIdx.x; pair < pair_count; pair += blockDim.x) {
      const uint32_t weight_pair =
          __builtin_nontemporal_load(weight_pairs + pair);
      const float weight0 = bf16_to_float(static_cast<uint16_t>(weight_pair));
      const float weight1 =
          bf16_to_float(static_cast<uint16_t>(weight_pair >> 16U));
      for (uint32_t row = 0U; row < m; ++row) {
        const auto *const activation_pairs = reinterpret_cast<const uint32_t *>(
            activation_start + static_cast<uint64_t>(row) * k);
        const uint32_t activation_pair = activation_pairs[pair];
        partial[row] +=
            bf16_to_float(static_cast<uint16_t>(activation_pair)) * weight0;
        partial[row] +=
            bf16_to_float(static_cast<uint16_t>(activation_pair >> 16U)) *
            weight1;
      }
    }
  } else {
    for (uint64_t reduction = threadIdx.x; reduction < k;
         reduction += blockDim.x) {
      const float weight_value = bf16_to_float(weight_row[reduction]);
      for (uint32_t row = 0U; row < m; ++row) {
        partial[row] +=
            bf16_to_float(
                activation_start[static_cast<uint64_t>(row) * k + reduction]) *
            weight_value;
      }
    }
  }

  for (uint32_t offset = WaveWidth / 2U; offset != 0U; offset >>= 1U) {
    for (uint32_t row = 0U; row < m; ++row) {
      partial[row] += __shfl_down(partial[row], offset, WaveWidth);
    }
  }
  __shared__ float wave_sums[max_rows][WaveCount];
  const uint32_t lane = threadIdx.x % WaveWidth;
  const uint32_t wave = threadIdx.x / WaveWidth;
  if (lane == 0U) {
    for (uint32_t row = 0U; row < m; ++row) {
      wave_sums[row][wave] = partial[row];
    }
  }
  __syncthreads();
  if (wave == 0U) {
    for (uint32_t row = 0U; row < m; ++row) {
      partial[row] = lane < WaveCount ? wave_sums[row][lane] : 0.0F;
    }
    for (uint32_t offset = WaveWidth / 2U; offset != 0U; offset >>= 1U) {
      for (uint32_t row = 0U; row < m; ++row) {
        partial[row] += __shfl_down(partial[row], offset, WaveWidth);
      }
    }
    if (lane == 0U) {
      for (uint32_t row = 0U; row < m; ++row) {
        output[(row_start + static_cast<uint64_t>(row)) * n + column] =
            float_to_bf16_rne_bits(partial[row]);
      }
    }
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_decode_v4(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t k, const uint64_t n) {
  matmul_bf16_decode_body<32U, 8U>(activation, weight, output, k, n,
                                   blockIdx.x);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_decode_wave64_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t k, const uint64_t n) {
  matmul_bf16_decode_body<64U, 4U>(activation, weight, output, k, n,
                                   blockIdx.x);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_decode_serial_rows_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  matmul_bf16_serial_rows_body<32U, 8U>(activation, weight, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_decode_serial_rows_wave64_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  matmul_bf16_serial_rows_body<64U, 4U>(activation, weight, output, m, k, n);
}

// Short prefill provider for the exact gfx1030 Qwen projection shapes.  Each
// block owns one output column and one consecutive group of up to eight rows;
// the existing serial-reduction body is reused unchanged for each group.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_prefill_short_serial_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  if (n == 0U || m == 0U) {
    return;
  }
  const uint64_t column = blockIdx.x % n;
  const uint64_t row_group = blockIdx.x / n;
  const uint64_t row_start = row_group * UINT64_C(8);
  if (row_start >= m) {
    return;
  }
  const uint64_t remaining_rows = m - row_start;
  const uint64_t rows =
      remaining_rows < UINT64_C(8) ? remaining_rows : UINT64_C(8);
  // gfx1030 uses wave32, matching the established M=2..8 provider.
  matmul_bf16_serial_rows_body<32U, 8U>(activation, weight, output, rows, k, n,
                                        row_start, column);
}

extern "C" __global__ void
sllm_matmul_fp32_to_bf16_short_mixed_v1(const float *const input,
                                        uint16_t *const output,
                                        const uint64_t element_count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    output[index] = float_to_bf16_rne_bits(input[index]);
  }
}

// The Phase 15 provider remains the decode path and is also the within-binary
// prefill performance control when SLLM_NVFP4_FORCE_BASELINE=1 is explicit.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_block16_packed_dequant_v1(
    const uint16_t *const activation, const uint8_t *const packed_weight,
    const uint8_t *const block_scales, const float *const tensor_scale,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  const uint64_t output_index = blockIdx.x;
  if (output_index >= m * n) {
    return;
  }
  const uint64_t row = output_index / n;
  const uint64_t column = output_index - row * n;
  const uint64_t blocks_per_weight_row = (k + UINT64_C(15)) / UINT64_C(16);
  float partial = 0.0F;
  for (uint64_t inner = threadIdx.x; inner < k; inner += blockDim.x) {
    const uint64_t weight_index = column * k + inner;
    const uint8_t packed =
        __builtin_nontemporal_load(packed_weight + weight_index / UINT64_C(2));
    const uint8_t code = (weight_index & UINT64_C(1)) == 0U
                             ? packed & UINT8_C(0x0f)
                             : packed >> 4U;
    const float scale = e4m3fn_to_float(
        block_scales[column * blocks_per_weight_row + inner / UINT64_C(16)]);
    partial += bf16_to_float(activation[row * k + inner]) *
               e2m1_to_float(code) * scale * tensor_scale[0];
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    partial += __shfl_down(partial, offset, 32U);
  }
  __shared__ float wave_sums[8];
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  if (lane == 0U) {
    wave_sums[wave] = partial;
  }
  __syncthreads();
  if (wave == 0U) {
    partial = lane < 8U ? wave_sums[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
      partial += __shfl_down(partial, offset, 32U);
    }
    if (lane == 0U) {
      output[output_index] = float_to_bf16_rne_bits(partial);
    }
  }
}

// Prefill maps one wave to one M row. Eight rows share the packed weight
// decode for each output column and keep the expansion bounded to one K tile.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_block16_prefill_row8_tiled256_v2(
    const uint16_t *const activation, const uint8_t *const packed_weight,
    const uint8_t *const block_scales, const float *const tensor_scale,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t rows_per_workgroup = 8U;
  constexpr uint32_t tile_k = 256U;
  __shared__ float weight_tile[tile_k];
  __shared__ float scale_tile[tile_k / 16U];
  __shared__ float shared_tensor_scale;
  const uint64_t column = static_cast<uint64_t>(blockIdx.x) % n;
  const uint64_t row_base =
      (static_cast<uint64_t>(blockIdx.x) / n) * rows_per_workgroup;
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  const uint64_t row = row_base + wave;
  const uint64_t blocks_per_weight_row = (k + UINT64_C(15)) / UINT64_C(16);
  float accumulator = 0.0F;
  if (threadIdx.x == 0U) {
    shared_tensor_scale = tensor_scale[0];
  }
  for (uint64_t base = 0U; base < k; base += tile_k) {
    if (threadIdx.x < tile_k / 16U) {
      const uint64_t scale_inner = base + threadIdx.x * UINT64_C(16);
      scale_tile[threadIdx.x] =
          scale_inner < k
              ? e4m3fn_to_float(block_scales[column * blocks_per_weight_row +
                                             scale_inner / UINT64_C(16)])
              : 0.0F;
    }
    __syncthreads();
    const uint64_t global_inner = base + threadIdx.x;
    if (global_inner < k) {
      const uint64_t weight_index = column * k + global_inner;
      const uint8_t packed = __builtin_nontemporal_load(
          packed_weight + weight_index / UINT64_C(2));
      const uint8_t code = (weight_index & UINT64_C(1)) == 0U
                               ? packed & UINT8_C(0x0f)
                               : packed >> 4U;
      weight_tile[threadIdx.x] =
          e2m1_to_float(code) * scale_tile[threadIdx.x / 16U];
    } else {
      weight_tile[threadIdx.x] = 0.0F;
    }
    __syncthreads();
    if (row < m) {
      const uint32_t valid = static_cast<uint32_t>(
          k - base < tile_k ? k - base : static_cast<uint64_t>(tile_k));
      for (uint32_t offset = lane; offset < valid; offset += wave_width) {
        accumulator += bf16_to_float(activation[row * k + base + offset]) *
                       weight_tile[offset];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    accumulator += __shfl_down(accumulator, offset, 32U);
  }
  if (lane == 0U && row < m) {
    output[row * n + column] =
        float_to_bf16_rne_bits(accumulator * shared_tensor_scale);
  }
}

namespace sllm_matmul_kernel {
hipError_t launch_fp8_quantize(const uint16_t *const activation,
                               uint8_t *const quantized, float *const scales,
                               const uint64_t m, const uint64_t k,
                               const bool fnuz,
                               const hipStream_t stream) noexcept {
  const char *const force_baseline =
      std::getenv("SLLM_FP8_QUANT_FORCE_BASELINE");
  // Phase 15O did not have a current MI300X tuple. Keep the verified FNUZ
  // provider on v1 until the OCP candidate is independently measured there.
  if (fnuz ||
      (force_baseline != nullptr && std::strcmp(force_baseline, "1") == 0)) {
    hipLaunchKernelGGL(sllm_matmul_bf16_to_fp8_outer_v1,
                       dim3(static_cast<uint32_t>(m)), dim3(kWorkgroupSize), 0U,
                       stream, activation, quantized, scales, m, k,
                       fnuz ? UINT32_C(1) : UINT32_C(0));
  } else {
    hipLaunchKernelGGL(sllm_matmul_bf16_to_fp8_outer_v2,
                       dim3(static_cast<uint32_t>(m)), dim3(kWorkgroupSize), 0U,
                       stream, activation, quantized, scales, m, k,
                       fnuz ? UINT32_C(1) : UINT32_C(0));
  }
  return hipGetLastError();
}

hipError_t launch_fp8_emulation(const uint8_t *const activation,
                                const float *const activation_scales,
                                const uint8_t *const weight,
                                const float *const weight_scales,
                                uint16_t *const output, const uint64_t m,
                                const uint64_t k, const uint64_t n,
                                const hipStream_t stream) noexcept {
  const uint64_t elements = m * n;
  const uint32_t blocks =
      static_cast<uint32_t>((elements + kWorkgroupSize - 1U) / kWorkgroupSize);
  hipLaunchKernelGGL(sllm_matmul_fp8_outer_emulation_v1, dim3(blocks),
                     dim3(kWorkgroupSize), 0U, stream, activation,
                     activation_scales, weight, weight_scales, output, m, k, n);
  return hipGetLastError();
}

hipError_t launch_nvfp4(const uint16_t *const activation,
                        const uint8_t *const packed_weight,
                        const uint8_t *const block_scales,
                        const float *const tensor_scale, uint16_t *const output,
                        const uint64_t m, const uint64_t k, const uint64_t n,
                        const KernelVariant variant,
                        const hipStream_t stream) noexcept {
  if (variant == KernelVariant::Nvfp4BaselinePackedDequant ||
      variant == KernelVariant::Nvfp4DecodePackedDequant) {
    hipLaunchKernelGGL(sllm_matmul_nvfp4_block16_packed_dequant_v1,
                       dim3(static_cast<uint32_t>(m * n)), dim3(kWorkgroupSize),
                       0U, stream, activation, packed_weight, block_scales,
                       tensor_scale, output, m, k, n);
  } else if (variant == KernelVariant::Nvfp4PrefillRow8Tiled256) {
    hipLaunchKernelGGL(sllm_matmul_nvfp4_block16_prefill_row8_tiled256_v2,
                       dim3(static_cast<uint32_t>(((m + 7U) / 8U) * n)),
                       dim3(kWorkgroupSize), 0U, stream, activation,
                       packed_weight, block_scales, tensor_scale, output, m, k,
                       n);
  } else {
    return hipErrorInvalidValue;
  }
  return hipGetLastError();
}

hipError_t launch_nvfp4_quantize(const uint16_t *const activation,
                                 uint8_t *const packed_activation,
                                 uint8_t *const block_scales,
                                 const float *const input_tensor_scale,
                                 const uint64_t m, const uint64_t k,
                                 const hipStream_t stream) noexcept {
  const uint64_t blocks_per_row = (k + UINT64_C(15)) / UINT64_C(16);
  hipLaunchKernelGGL(sllm_matmul_bf16_to_nvfp4_block16_v1,
                     dim3(static_cast<uint32_t>(m * blocks_per_row)),
                     dim3(kWorkgroupSize), 0U, stream, activation,
                     packed_activation, block_scales, input_tensor_scale, m, k);
  return hipGetLastError();
}

hipError_t launch_nvfp4_w4a4(const uint8_t *const packed_activation,
                             const uint8_t *const activation_block_scales,
                             const uint8_t *const packed_weight,
                             const uint8_t *const weight_block_scales,
                             const float *const weight_tensor_scale,
                             const float *const input_tensor_scale,
                             uint16_t *const output, const uint64_t m,
                             const uint64_t k, const uint64_t n,
                             const KernelVariant variant,
                             const hipStream_t stream) noexcept {
  if (variant != KernelVariant::Nvfp4W4A4Packed) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_matmul_nvfp4_w4a4_block16_packed_v1,
                     dim3(static_cast<uint32_t>(m * n)), dim3(kWorkgroupSize),
                     0U, stream, packed_activation, activation_block_scales,
                     packed_weight, weight_block_scales, weight_tensor_scale,
                     input_tensor_scale, output, m, k, n);
  return hipGetLastError();
}

hipError_t launch_mxfp4_quantize(const uint16_t *const activation,
                                 uint8_t *const packed_activation,
                                 uint8_t *const block_scales, const uint64_t m,
                                 const uint64_t k,
                                 const hipStream_t stream) noexcept {
  const uint64_t blocks_per_row = (k + UINT64_C(31)) / UINT64_C(32);
  hipLaunchKernelGGL(sllm_matmul_bf16_to_mxfp4_block32_even_v1,
                     dim3(static_cast<uint32_t>(m * blocks_per_row)),
                     dim3(kWorkgroupSize), 0U, stream, activation,
                     packed_activation, block_scales, m, k);
  return hipGetLastError();
}

hipError_t launch_mxfp4_w4a4(const uint8_t *const packed_activation,
                             const uint8_t *const activation_block_scales,
                             const uint8_t *const packed_weight,
                             const uint8_t *const weight_block_scales,
                             uint16_t *const output, const uint64_t m,
                             const uint64_t k, const uint64_t n,
                             const KernelVariant variant,
                             const hipStream_t stream) noexcept {
  if (variant == KernelVariant::Mxfp4W4A4Decode) {
    hipLaunchKernelGGL(sllm_matmul_mxfp4_w4a4_block32_decode_v1,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, packed_activation, activation_block_scales,
                       packed_weight, weight_block_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp4W4A4Prefill) {
    hipLaunchKernelGGL(sllm_matmul_mxfp4_w4a4_block32_prefill_v1,
                       dim3(static_cast<uint32_t>(m * n)), dim3(kWorkgroupSize),
                       0U, stream, packed_activation, activation_block_scales,
                       packed_weight, weight_block_scales, output, m, k, n);
  } else {
    return hipErrorInvalidValue;
  }
  return hipGetLastError();
}

hipError_t launch_mxfp8_quantize(const uint16_t *const activation,
                                 uint8_t *const quantized,
                                 uint8_t *const block_scales, const uint64_t m,
                                 const uint64_t k,
                                 const hipStream_t stream) noexcept {
  const uint64_t blocks_per_row = k / UINT64_C(32);
  hipLaunchKernelGGL(sllm_matmul_bf16_to_mxfp8_e4m3_block32_v1,
                     dim3(static_cast<uint32_t>(m * blocks_per_row)), dim3(32U),
                     0U, stream, activation, quantized, block_scales, m, k);
  return hipGetLastError();
}

hipError_t launch_mxfp8_w8a8(const uint8_t *const activation,
                             const uint8_t *const activation_scales,
                             const uint8_t *const weight,
                             const uint8_t *const weight_scales,
                             uint16_t *const output, const uint64_t m,
                             const uint64_t k, const uint64_t n,
                             const hipStream_t stream) noexcept {
  const KernelVariant variant = select_mxfp8_variant(m);
  return launch_mxfp8_w8a8(activation, activation_scales, weight, weight_scales,
                           output, m, k, n, variant, stream);
}

hipError_t launch_mxfp8_w8a8(const uint8_t *const activation,
                             const uint8_t *const activation_scales,
                             const uint8_t *const weight,
                             const uint8_t *const weight_scales,
                             uint16_t *const output, const uint64_t m,
                             const uint64_t k, const uint64_t n,
                             const KernelVariant variant,
                             const hipStream_t stream) noexcept {
  if (variant == KernelVariant::Mxfp8W8A8Decode) {
    hipLaunchKernelGGL(sllm_matmul_mxfp8_w8a8_e4m3_block32_decode_v1,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, activation, activation_scales, weight,
                       weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8Prefill) {
    hipLaunchKernelGGL(sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_v1,
                       dim3(static_cast<uint32_t>(m * n)), dim3(kWorkgroupSize),
                       0U, stream, activation, activation_scales, weight,
                       weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillRow8) {
    hipLaunchKernelGGL(sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_row8_v2,
                       dim3(static_cast<uint32_t>(((m + 7U) / 8U) * n)),
                       dim3(kWorkgroupSize), 0U, stream, activation,
                       activation_scales, weight, weight_scales, output, m, k,
                       n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillMmqCol4) {
    hipLaunchKernelGGL(
        sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_mmq_col4_v4,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 3U) / 4U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillMmqCol8) {
    hipLaunchKernelGGL(
        sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_mmq_col8_v4,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 7U) / 8U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col16) {
    if (!phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1030_mmq_col16_v1,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 15U) / 16U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col32) {
    if (!phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1030_mmq_col32_v1,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 31U) / 32U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Regscale) {
    if (!phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1030_mmq_col8_regscale_v1,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 7U) / 8U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Vector32) {
    if (!phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1030_mmq_col8_vector32_v1,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 7U) / 8U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant ==
             KernelVariant::Mxfp8W8A8PrefillMmqGfx1030RegscaleVector32) {
    if (!phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1030_mmq_col8_regscale_vector32_v1,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 7U) / 8U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillWmmaN16) {
    if (!phase63_mxfp8_wmma_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_wmma128x16x32_v1,
        dim3(static_cast<uint32_t>(
                 (n + kMxfp8W8A8PrefillWmmaN16ColumnsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaN16ColumnsPerWorkgroup),
             static_cast<uint32_t>(
                 (m + kMxfp8W8A8PrefillWmmaRowsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmmaWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillWmmaN64) {
    if (!phase63_mxfp8_wmma_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_wmma128x64x32_v2,
        dim3(static_cast<uint32_t>(
                 (n + kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup),
             static_cast<uint32_t>(
                 (m + kMxfp8W8A8PrefillWmmaRowsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmmaWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillWmma4Wave) {
    if (!phase64_mxfp8_wmma_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1201_wmma64x64_4w_v1,
        dim3(static_cast<uint32_t>(n /
                                   kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup),
             static_cast<uint32_t>(
                 (m + kMxfp8W8A8PrefillWmma4WaveRowsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmma4WaveRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmma4WaveWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillWmmaLdsPad) {
    if (!phase64_mxfp8_wmma_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1201_wmma128x64_pad33_v1,
        dim3(static_cast<uint32_t>(n /
                                   kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup),
             static_cast<uint32_t>(
                 (m + kMxfp8W8A8PrefillWmmaRowsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmmaWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillWmmaDirectWeight) {
    if (!phase64_mxfp8_wmma_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1201_wmma128x64_direct_v1,
        dim3(static_cast<uint32_t>(n /
                                   kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup),
             static_cast<uint32_t>(
                 (m + kMxfp8W8A8PrefillWmmaRowsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmmaWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillWmmaDirectActivation) {
    if (!phase65_mxfp8_wmma_direct_activation_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1201_wmma128x64_adirect_v1,
        dim3(static_cast<uint32_t>(n /
                                   kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup),
             static_cast<uint32_t>(m / kMxfp8W8A8PrefillWmmaRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmmaWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillWmmaDirectBoth) {
    if (!phase65_mxfp8_wmma_direct_activation_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1201_wmma128x64_bdirect_v1,
        dim3(static_cast<uint32_t>(n /
                                   kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup),
             static_cast<uint32_t>(m / kMxfp8W8A8PrefillWmmaRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmmaWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillWmmaN128DirectBoth) {
    if (!phase66_mxfp8_wmma_n128_direct_both_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1201_wmma128x128_bdirect_v1,
        dim3(static_cast<uint32_t>(
                 n / kMxfp8W8A8PrefillWmmaN128ColumnsPerWorkgroup),
             static_cast<uint32_t>(m / kMxfp8W8A8PrefillWmmaRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmmaWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillTiled16) {
    hipLaunchKernelGGL(sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_tiled16_v3,
                       dim3(static_cast<uint32_t>((n + 15U) / 16U),
                            static_cast<uint32_t>((m + 15U) / 16U)),
                       dim3(16U, 16U), 0U, stream, activation,
                       activation_scales, weight, weight_scales, output, m, k,
                       n);
  } else {
    return hipErrorInvalidValue;
  }
  return hipGetLastError();
}

hipError_t launch_mxfp6_quantize(const uint16_t *const activation,
                                 uint8_t *const packed,
                                 uint8_t *const block_scales, const uint64_t m,
                                 const uint64_t k,
                                 const hipStream_t stream) noexcept {
  const uint64_t blocks_per_row = k / UINT64_C(32);
  hipLaunchKernelGGL(sllm_matmul_bf16_to_mxfp6_e3m2_block32_v1,
                     dim3(static_cast<uint32_t>(m * blocks_per_row)), dim3(32U),
                     0U, stream, activation, packed, block_scales, m, k);
  return hipGetLastError();
}

hipError_t launch_mxfp6_w6a6(const uint8_t *const activation,
                             const uint8_t *const activation_scales,
                             const uint8_t *const weight,
                             const uint8_t *const weight_scales,
                             uint16_t *const output, const uint64_t m,
                             const uint64_t k, const uint64_t n,
                             const KernelVariant variant,
                             const hipStream_t stream) noexcept {
  if (variant == KernelVariant::Mxfp6W6A6Decode) {
    hipLaunchKernelGGL(sllm_matmul_mxfp6_w6a6_e3m2_block32_decode_v1,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, activation, activation_scales, weight,
                       weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp6W6A6Prefill) {
    hipLaunchKernelGGL(sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_v1,
                       dim3(static_cast<uint32_t>(m * n)), dim3(kWorkgroupSize),
                       0U, stream, activation, activation_scales, weight,
                       weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp6W6A6PrefillRow8) {
    hipLaunchKernelGGL(sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_row8_v2,
                       dim3(static_cast<uint32_t>(((m + 7U) / 8U) * n)),
                       dim3(kWorkgroupSize), 0U, stream, activation,
                       activation_scales, weight, weight_scales, output, m, k,
                       n);
  } else if (variant == KernelVariant::Mxfp6W6A6PrefillMmqCol4) {
    hipLaunchKernelGGL(
        sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_mmq_col4_v4,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 3U) / 4U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp6W6A6PrefillMmqCol8) {
    hipLaunchKernelGGL(
        sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_mmq_col8_v4,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 7U) / 8U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp6W6A6PrefillTiled16) {
    hipLaunchKernelGGL(sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_tiled16_v3,
                       dim3(static_cast<uint32_t>((n + 15U) / 16U),
                            static_cast<uint32_t>((m + 15U) / 16U)),
                       dim3(16U, 16U), 0U, stream, activation,
                       activation_scales, weight, weight_scales, output, m, k,
                       n);
  } else {
    return hipErrorInvalidValue;
  }
  return hipGetLastError();
}

hipError_t launch(const uint16_t *const activation,
                  const uint16_t *const weight, uint16_t *const output,
                  const uint64_t m, const uint64_t k, const uint64_t n,
                  const KernelVariant variant,
                  const hipStream_t stream) noexcept {
  if (variant == KernelVariant::HipBlas) {
    return hipErrorInvalidValue;
  }
  if (variant == KernelVariant::DecodeReductionWave64) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_decode_wave64_v1,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, activation, weight, output, k, n);
  } else if (variant == KernelVariant::DecodeReduction) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_decode_v4,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, activation, weight, output, k, n);
  } else if (variant == KernelVariant::SerialRowsReductionWave64) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_decode_serial_rows_wave64_v1,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, activation, weight, output, m, k, n);
  } else if (variant == KernelVariant::SerialRowsReduction) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_decode_serial_rows_v1,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, activation, weight, output, m, k, n);
  } else if (variant == KernelVariant::PrefillShortSerial) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_prefill_short_serial_v1,
                       dim3(grid_size_x(variant, m, n)), dim3(kWorkgroupSize),
                       0U, stream, activation, weight, output, m, k, n);
  } else if (variant == KernelVariant::PrefillTiled16) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_tiled16_v2,
                       dim3(static_cast<uint32_t>((n + 15U) / 16U),
                            static_cast<uint32_t>((m + 15U) / 16U)),
                       dim3(16U, 16U), 0U, stream, activation, weight, output,
                       m, k, n);
  } else {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_v1,
                       dim3(grid_size_x(variant, m, n)), dim3(kWorkgroupSize),
                       0U, stream, activation, weight, output, m, k, n);
  }
  return hipGetLastError();
}

hipError_t launch_short_mixed_f32_to_bf16(const float *const output_f32,
                                          uint16_t *const output,
                                          const uint64_t element_count,
                                          const hipStream_t stream) noexcept {
  if (element_count == 0U || element_count > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_matmul_fp32_to_bf16_short_mixed_v1,
                     dim3(static_cast<uint32_t>((element_count + 255U) / 256U)),
                     dim3(kWorkgroupSize), 0U, stream, output_f32, output,
                     element_count);
  return hipGetLastError();
}

} // namespace sllm_matmul_kernel
