#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#if !defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
#include <hipblas/hipblas.h>
#include <mutex>
#endif

#include "causal_attention_kernel_internal.hpp"
#include "sllm/hip.h"

namespace sllm_causal_attention_kernel {
namespace {

__device__ float f16_to_f32(const uint16_t raw) noexcept {
  const uint32_t sign = (static_cast<uint32_t>(raw) & 0x8000U) << 16U;
  const uint32_t exponent = (static_cast<uint32_t>(raw) >> 10U) & 0x1fU;
  const uint32_t fraction = static_cast<uint32_t>(raw) & 0x03ffU;
  uint32_t bits = 0U;
  if (exponent == 0U) {
    if (fraction == 0U) {
      bits = sign;
    } else {
      uint32_t normalized = fraction;
      uint32_t shift = 0U;
      while ((normalized & 0x0400U) == 0U) {
        normalized <<= 1U;
        ++shift;
      }
      normalized &= 0x03ffU;
      bits = sign | ((127U - 14U - shift) << 23U) | (normalized << 13U);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (fraction << 13U);
  } else {
    bits = sign | ((exponent + 112U) << 23U) | (fraction << 13U);
  }
  return __uint_as_float(bits);
}

__device__ float bf16_to_f32(const uint16_t raw) noexcept {
  return __uint_as_float(static_cast<uint32_t>(raw) << 16U);
}

__device__ float2 load_bf16_pair(const uint16_t *const values) noexcept {
  if ((reinterpret_cast<uintptr_t>(values) & 3U) == 0U) {
    const uint32_t packed = *reinterpret_cast<const uint32_t *>(values);
    return make_float2(bf16_to_f32(static_cast<uint16_t>(packed & 0xffffU)),
                       bf16_to_f32(static_cast<uint16_t>(packed >> 16U)));
  }
  return make_float2(bf16_to_f32(values[0]), bf16_to_f32(values[1]));
}

__device__ float2 load_fp16_pair(const uint16_t *const values) noexcept {
  if ((reinterpret_cast<uintptr_t>(values) & 3U) == 0U) {
    const __half2 packed = *reinterpret_cast<const __half2 *>(values);
    return __half22float2(packed);
  }
  return make_float2(f16_to_f32(values[0]), f16_to_f32(values[1]));
}

__device__ float e4m3fn_to_f32(const uint8_t bits) noexcept {
#if defined(__gfx1201__)
  return __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(bits), 0);
#else
  const float sign = (bits & UINT8_C(0x80)) == 0U ? 1.0F : -1.0F;
  const uint8_t exponent = static_cast<uint8_t>((bits >> 3U) & 0x0fU);
  const uint8_t mantissa = static_cast<uint8_t>(bits & 0x07U);
  if (exponent == 0U) {
    return mantissa == 0U
               ? copysignf(0.0F, sign)
               : sign * static_cast<float>(mantissa) * ldexpf(1.0F, -9);
  }
  if (exponent == 0x0fU && mantissa == 0x07U) {
    return NAN;
  }
  return sign * (1.0F + static_cast<float>(mantissa) / 8.0F) *
         ldexpf(1.0F, static_cast<int>(exponent) - 7);
#endif
}

__device__ float e2m1_to_f32(const uint8_t bits) noexcept {
  constexpr float positive[8] = {0.0F, 0.5F, 1.0F, 1.5F,
                                 2.0F, 3.0F, 4.0F, 6.0F};
  const float value = positive[bits & 0x07U];
  return (bits & 0x08U) == 0U ? value : -value;
}

__device__ float load_kv(const void *const values, const void *const scales,
                         const float *const outer_scales,
                         const uint32_t encoding, const uint64_t row,
                         const uint32_t dimension, const uint32_t head_dim,
                         const float static_scale) noexcept {
  if (encoding == SLLM_HIP_KV_ENCODING_FP16_V1) {
    return f16_to_f32(
        static_cast<const uint16_t *>(values)[row * head_dim + dimension]);
  }
  if (encoding == SLLM_HIP_KV_ENCODING_FP8_V1 ||
      encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1) {
    const float scale = encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1
                            ? static_scale
                            : static_cast<const float *>(scales)[row];
    return e4m3fn_to_f32(static_cast<const uint8_t *>(
               values)[row * head_dim + dimension]) *
           scale;
  }
  const uint64_t packed_per_row = (static_cast<uint64_t>(head_dim) + 1U) / 2U;
  const uint64_t blocks_per_row = (static_cast<uint64_t>(head_dim) + 15U) / 16U;
  const uint8_t packed = static_cast<const uint8_t *>(
      values)[row * packed_per_row + dimension / 2U];
  const uint8_t code = (dimension & 1U) == 0U ? packed & 0x0fU : packed >> 4U;
  return e2m1_to_f32(code) *
         e4m3fn_to_f32(static_cast<const uint8_t *>(
             scales)[row * blocks_per_row + dimension / 16U]) *
         outer_scales[row];
}

__device__ uint16_t f32_to_bf16_rne(const float value) noexcept {
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

#pragma clang fp contract(off)
template <bool UseWaveProvider>
__global__ __launch_bounds__(256, 1) void causal_attention_kernel(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const float *const key_outer_scales, const float *const value_outer_scales,
    uint16_t *const output, const uint32_t query_count,
    const uint64_t /*capacity_tokens*/, const uint64_t start_position,
    const uint64_t committed_kv_length, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim, const uint32_t encoding,
    const float static_key_scale, const float static_value_scale) {
  if (blockIdx.x >= query_count * q_heads) {
    return;
  }
  const uint64_t flat = blockIdx.x;
  const uint64_t row = flat / q_heads;
  const uint32_t query_head = static_cast<uint32_t>(flat % q_heads);
  const uint64_t query_position = start_position + row;
  if (query_position >= committed_kv_length) {
    return;
  }
  const uint16_t *const query_row =
      query + row * q_heads * head_dim +
      static_cast<uint64_t>(query_head) * head_dim;
  const uint32_t kv_head = query_head / (q_heads / kv_heads);
  uint16_t *const output_row = output + row * q_heads * head_dim +
                               static_cast<uint64_t>(query_head) * head_dim;

  const uint32_t dimension = threadIdx.x;
  __shared__ float reductions[SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE];
  __shared__ float rescale;
  __shared__ float contribution;
  __shared__ float running_maximum;
  __shared__ float running_denominator;
  if (dimension == 0U) {
    running_maximum = -std::numeric_limits<float>::infinity();
    running_denominator = 0.0F;
  }
  __syncthreads();

  float accumulation0 = 0.0F;
  float accumulation1 = 0.0F;
  for (uint64_t key_position = 0U; key_position <= query_position;
       ++key_position) {
    const uint64_t kv_row = key_position * kv_heads + kv_head;
#if defined(__gfx1201__)
    if constexpr (UseWaveProvider) {
      float partial = 0.0F;
      for (uint32_t current = dimension; current < head_dim;
           current += blockDim.x) {
        partial += bf16_to_f32(query_row[current]) *
                   load_kv(key, key_scales, key_outer_scales, encoding, kv_row,
                           current, head_dim, static_key_scale);
      }
      for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
        partial += __shfl_down(partial, offset, 32U);
      }
      const uint32_t lane = dimension & 31U;
      const uint32_t wave = dimension >> 5U;
      if (lane == 0U) {
        reductions[wave] = partial;
      }
      __syncthreads();
      if (wave == 0U) {
        float block_sum = lane < 8U ? reductions[lane] : 0.0F;
        for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
          block_sum += __shfl_down(block_sum, offset, 32U);
        }
        if (lane == 0U) {
          const float current_score =
              block_sum * rsqrtf(static_cast<float>(head_dim));
          const float next_maximum = fmaxf(running_maximum, current_score);
          rescale = expf(running_maximum - next_maximum);
          contribution = expf(current_score - next_maximum);
          running_denominator = running_denominator * rescale + contribution;
          running_maximum = next_maximum;
        }
      }
      __syncthreads();
    } else
#endif
    {
      reductions[dimension] = 0.0F;
      for (uint32_t current = dimension; current < head_dim;
           current += blockDim.x) {
        reductions[dimension] +=
            bf16_to_f32(query_row[current]) *
            load_kv(key, key_scales, key_outer_scales, encoding, kv_row,
                    current, head_dim, static_key_scale);
      }
      __syncthreads();
      for (uint32_t stride = SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE / 2U;
           stride != 0U; stride >>= 1U) {
        if (dimension < stride) {
          reductions[dimension] += reductions[dimension + stride];
        }
        __syncthreads();
      }
      if (dimension == 0U) {
        const float current_score =
            reductions[0] * rsqrtf(static_cast<float>(head_dim));
        const float next_maximum = fmaxf(running_maximum, current_score);
        rescale = expf(running_maximum - next_maximum);
        contribution = expf(current_score - next_maximum);
        running_denominator = running_denominator * rescale + contribution;
        running_maximum = next_maximum;
      }
      __syncthreads();
    }
    if (dimension < head_dim) {
      accumulation0 =
          accumulation0 * rescale +
          contribution * load_kv(value, value_scales, value_outer_scales,
                                 encoding, kv_row, dimension, head_dim,
                                 static_value_scale);
    }
    const uint32_t second = dimension + blockDim.x;
    if (second < head_dim) {
      accumulation1 =
          accumulation1 * rescale +
          contribution * load_kv(value, value_scales, value_outer_scales,
                                 encoding, kv_row, second, head_dim,
                                 static_value_scale);
    }
    __syncthreads();
  }
  if (dimension < head_dim) {
    output_row[dimension] =
        f32_to_bf16_rne(accumulation0 / running_denominator);
  }
  const uint32_t second = dimension + blockDim.x;
  if (second < head_dim) {
    output_row[second] = f32_to_bf16_rne(accumulation1 / running_denominator);
  }
}

// Each wave owns one contiguous KV interval and computes an independent
// online-softmax partial. The workgroup then merges the eight partials in
// increasing interval order. This removes the per-key workgroup barriers from
// long M=1 attention without publishing intermediate values outside the block.
template <bool UseQueryPreload>
__global__
__launch_bounds__(256, 1) void causal_attention_decode_wave_split_kernel(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const float *const key_outer_scales, const float *const value_outer_scales,
    uint16_t *const output, const uint64_t committed_kv_length,
    const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim,
    const uint32_t encoding, const float static_key_scale,
    const float static_value_scale) {
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kWaveCount = 8U;
  constexpr uint32_t kHeadDim = SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM;
  constexpr uint32_t kDimensionsPerLane = kHeadDim / kWaveSize;
  const uint32_t query_head = blockIdx.x;
  if (query_head >= q_heads) {
    return;
  }
  const uint32_t dimension = threadIdx.x;
  const uint32_t lane = dimension & (kWaveSize - 1U);
  const uint32_t wave = dimension / kWaveSize;
  const uint32_t kv_head = query_head / (q_heads / kv_heads);
  const uint16_t *const query_row =
      query + static_cast<uint64_t>(query_head) * head_dim;
  uint16_t *const output_row =
      output + static_cast<uint64_t>(query_head) * head_dim;

  __shared__ float partial_values[kWaveCount * kHeadDim];
  __shared__ float partial_maxima[kWaveCount];
  __shared__ float partial_denominators[kWaveCount];
  float accumulations[kDimensionsPerLane];
  float query_values[kDimensionsPerLane];
#pragma unroll
  for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
    accumulations[index] = 0.0F;
    // The opt-in variant only hoists the immutable BF16 query conversion. It
    // leaves the per-key product and accumulation order identical to the
    // control kernel.
    if constexpr (UseQueryPreload) {
      const uint32_t current = lane + index * kWaveSize;
      query_values[index] =
          current < head_dim ? bf16_to_f32(query_row[current]) : 0.0F;
    }
  }
  float local_maximum = -std::numeric_limits<float>::infinity();
  float local_denominator = 0.0F;
  const uint64_t split_begin = committed_kv_length * wave / kWaveCount;
  const uint64_t split_end = committed_kv_length * (wave + 1U) / kWaveCount;
  for (uint64_t key_position = split_begin; key_position < split_end;
       ++key_position) {
    const uint64_t kv_row = key_position * kv_heads + kv_head;
    float partial = 0.0F;
#pragma unroll
    for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
      const uint32_t current = lane + index * kWaveSize;
      if (current < head_dim) {
        if constexpr (UseQueryPreload) {
          partial += query_values[index] *
                     load_kv(key, key_scales, key_outer_scales, encoding,
                             kv_row, current, head_dim, static_key_scale);
        } else {
          partial += bf16_to_f32(query_row[current]) *
                     load_kv(key, key_scales, key_outer_scales, encoding,
                             kv_row, current, head_dim, static_key_scale);
        }
      }
    }
    for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
      partial += __shfl_down(partial, offset, kWaveSize);
    }
    float rescale = 0.0F;
    float contribution = 0.0F;
    if (lane == 0U) {
      const float current_score =
          partial * rsqrtf(static_cast<float>(head_dim));
      const float next_maximum = fmaxf(local_maximum, current_score);
      rescale = expf(local_maximum - next_maximum);
      contribution = expf(current_score - next_maximum);
      local_denominator = local_denominator * rescale + contribution;
      local_maximum = next_maximum;
    }
    rescale = __shfl(rescale, 0U, kWaveSize);
    contribution = __shfl(contribution, 0U, kWaveSize);
#pragma unroll
    for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
      const uint32_t current = lane + index * kWaveSize;
      if (current < head_dim) {
        accumulations[index] =
            accumulations[index] * rescale +
            contribution * load_kv(value, value_scales, value_outer_scales,
                                   encoding, kv_row, current, head_dim,
                                   static_value_scale);
      }
    }
  }
  if (lane == 0U) {
    partial_maxima[wave] = local_maximum;
    partial_denominators[wave] = local_denominator;
  }
#pragma unroll
  for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
    const uint32_t current = lane + index * kWaveSize;
    if (current < head_dim) {
      partial_values[wave * kHeadDim + current] = accumulations[index];
    }
  }
  __syncthreads();

  float global_maximum = partial_maxima[0];
#pragma unroll
  for (uint32_t split = 1U; split < kWaveCount; ++split) {
    global_maximum = fmaxf(global_maximum, partial_maxima[split]);
  }
  float global_denominator = 0.0F;
  float merged = 0.0F;
#pragma unroll
  for (uint32_t split = 0U; split < kWaveCount; ++split) {
    const float scale = expf(partial_maxima[split] - global_maximum);
    global_denominator += partial_denominators[split] * scale;
    if (dimension < head_dim) {
      merged += partial_values[split * kHeadDim + dimension] * scale;
    }
  }
  if (dimension < head_dim) {
    output_row[dimension] = f32_to_bf16_rne(merged / global_denominator);
  }
  const uint32_t second = dimension + blockDim.x;
  if (second < head_dim) {
    float merged_second = 0.0F;
#pragma unroll
    for (uint32_t split = 0U; split < kWaveCount; ++split) {
      const float scale = expf(partial_maxima[split] - global_maximum);
      merged_second += partial_values[split * kHeadDim + second] * scale;
    }
    output_row[second] = f32_to_bf16_rne(merged_second / global_denominator);
  }
}

// FP16-only long-decode experiment.  The block/wave decomposition and merge
// order match the measured wave-split provider, but each lane handles four
// adjacent half2 pairs (lane*2 + segment*64).  Aligned uint32/half2 loads are
// used for the common path, with scalar loads retained for unusual offsets.
__global__
__launch_bounds__(256, 1) void causal_attention_decode_wave_split_fp16_pair_kernel(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const float *const key_outer_scales, const float *const value_outer_scales,
    uint16_t *const output, const uint64_t committed_kv_length,
    const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim,
    const uint32_t encoding, const float static_key_scale,
    const float static_value_scale) {
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kWaveCount = 8U;
  constexpr uint32_t kHeadDim = SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM;
  constexpr uint32_t kPairsPerLane = 4U;
  (void)key_scales;
  (void)value_scales;
  (void)key_outer_scales;
  (void)value_outer_scales;
  (void)static_key_scale;
  (void)static_value_scale;
  const uint32_t dimension = threadIdx.x;
  const uint32_t lane = dimension & (kWaveSize - 1U);
  const uint32_t wave = dimension / kWaveSize;
  const uint32_t query_head = blockIdx.x;
  if (query_head >= q_heads || q_heads != 16U || kv_heads != 4U ||
      head_dim != kHeadDim || encoding != SLLM_HIP_KV_ENCODING_FP16_V1) {
    return;
  }
  const uint32_t kv_head = query_head / (q_heads / kv_heads);
  const uint16_t *const query_row =
      query + static_cast<uint64_t>(query_head) * head_dim;
  uint16_t *const output_row =
      output + static_cast<uint64_t>(query_head) * head_dim;

  float query_first[kPairsPerLane];
  float query_second[kPairsPerLane];
  float accumulations_first[kPairsPerLane] = {};
  float accumulations_second[kPairsPerLane] = {};
  float key_first[kPairsPerLane];
  float key_second[kPairsPerLane];
  float local_maximum = -std::numeric_limits<float>::infinity();
  float local_denominator = 0.0F;
#pragma unroll
  for (uint32_t segment = 0U; segment < kPairsPerLane; ++segment) {
    const uint32_t pair_start = lane * 2U + segment * 64U;
    const float2 pair = load_bf16_pair(query_row + pair_start);
    query_first[segment] = pair.x;
    query_second[segment] = pair.y;
  }

  __shared__ float partial_values[kWaveCount * kHeadDim];
  __shared__ float partial_maxima[kWaveCount];
  __shared__ float partial_denominators[kWaveCount];
  const uint64_t split_begin = committed_kv_length * wave / kWaveCount;
  const uint64_t split_end = committed_kv_length * (wave + 1U) / kWaveCount;
  for (uint64_t key_position = split_begin; key_position < split_end;
       ++key_position) {
    const uint64_t kv_row = key_position * kv_heads + kv_head;
    const uint16_t *const key_row = static_cast<const uint16_t *>(key) +
                                    kv_row * static_cast<uint64_t>(head_dim);
    const uint16_t *const value_row = static_cast<const uint16_t *>(value) +
                                      kv_row * static_cast<uint64_t>(head_dim);
#pragma unroll
    for (uint32_t segment = 0U; segment < kPairsPerLane; ++segment) {
      const uint32_t pair_start = lane * 2U + segment * 64U;
      const float2 key_pair = load_fp16_pair(key_row + pair_start);
      key_first[segment] = key_pair.x;
      key_second[segment] = key_pair.y;
    }
    float partial = 0.0F;
#pragma unroll
    for (uint32_t segment = 0U; segment < kPairsPerLane; ++segment) {
      partial += query_first[segment] * key_first[segment] +
                 query_second[segment] * key_second[segment];
    }
    for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
      partial += __shfl_down(partial, offset, kWaveSize);
    }
    float rescale = 0.0F;
    float contribution = 0.0F;
    if (lane == 0U) {
      const float current_score =
          partial * rsqrtf(static_cast<float>(head_dim));
      const float next_maximum = fmaxf(local_maximum, current_score);
      rescale = expf(local_maximum - next_maximum);
      contribution = expf(current_score - next_maximum);
      local_denominator = local_denominator * rescale + contribution;
      local_maximum = next_maximum;
    }
    rescale = __shfl(rescale, 0U, kWaveSize);
    contribution = __shfl(contribution, 0U, kWaveSize);
#pragma unroll
    for (uint32_t segment = 0U; segment < kPairsPerLane; ++segment) {
      const uint32_t pair_start = lane * 2U + segment * 64U;
      const float2 value_pair = load_fp16_pair(value_row + pair_start);
      accumulations_first[segment] =
          accumulations_first[segment] * rescale + contribution * value_pair.x;
      accumulations_second[segment] =
          accumulations_second[segment] * rescale + contribution * value_pair.y;
    }
  }
  if (lane == 0U) {
    partial_maxima[wave] = local_maximum;
    partial_denominators[wave] = local_denominator;
  }
#pragma unroll
  for (uint32_t segment = 0U; segment < kPairsPerLane; ++segment) {
    const uint32_t pair_start = lane * 2U + segment * 64U;
    partial_values[wave * kHeadDim + pair_start] = accumulations_first[segment];
    partial_values[wave * kHeadDim + pair_start + 1U] =
        accumulations_second[segment];
  }
  __syncthreads();

  float global_maximum = partial_maxima[0];
#pragma unroll
  for (uint32_t split = 1U; split < kWaveCount; ++split) {
    global_maximum = fmaxf(global_maximum, partial_maxima[split]);
  }
  float global_denominator = 0.0F;
#pragma unroll
  for (uint32_t split = 0U; split < kWaveCount; ++split) {
    const float scale = expf(partial_maxima[split] - global_maximum);
    global_denominator += partial_denominators[split] * scale;
  }
  if (wave == 0U) {
#pragma unroll
    for (uint32_t segment = 0U; segment < kPairsPerLane; ++segment) {
      const uint32_t pair_start = lane * 2U + segment * 64U;
      float merged_first = 0.0F;
      float merged_second = 0.0F;
#pragma unroll
      for (uint32_t split = 0U; split < kWaveCount; ++split) {
        const float scale = expf(partial_maxima[split] - global_maximum);
        merged_first += partial_values[split * kHeadDim + pair_start] * scale;
        merged_second +=
            partial_values[split * kHeadDim + pair_start + 1U] * scale;
      }
      output_row[pair_start] =
          f32_to_bf16_rne(merged_first / global_denominator);
      output_row[pair_start + 1U] =
          f32_to_bf16_rne(merged_second / global_denominator);
    }
  }
}

// Long-decode GQA4 experiment.  A block owns one KV head and one of sixteen
// fixed KV partitions.  Its four wave32 waves process the four query heads
// sharing that KV head.  K/V tiles are loaded into LDS once per block and then
// reused by all four waves.  Stage 1 writes one online-softmax partial per
// query-head/partition; stage 2 merges partitions in increasing order.
constexpr uint32_t kDecodeGqa4SplitPartitions = 16U;
constexpr uint32_t kDecodeGqa4SplitTileTokens = 16U;
constexpr uint32_t kDecodeGqa4SplitWaveSize = 32U;
constexpr uint32_t kDecodeGqa4SplitGqaRatio = 4U;
constexpr uint32_t kDecodeGqa4SplitHeadDim = SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM;
constexpr uint32_t kDecodeGqa4SplitWorkspaceStride =
    kDecodeGqa4SplitHeadDim + 2U;

template <uint32_t kPartitions>
__global__
__launch_bounds__(128, 1) void causal_attention_decode_gqa4_split_stage1_kernel(
    const uint16_t *const query, const uint16_t *const key,
    const uint16_t *const value, uint16_t *const output,
    const uint64_t committed_kv_length, float *const workspace) {
  const uint32_t block = blockIdx.x;
  const uint32_t kv_head = block / kPartitions;
  const uint32_t partition = block % kPartitions;
  if (kv_head >= 4U) {
    return;
  }
  const uint64_t split_begin = committed_kv_length * partition / kPartitions;
  const uint64_t split_end =
      committed_kv_length * (partition + 1U) / kPartitions;
  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (kDecodeGqa4SplitWaveSize - 1U);
  const uint32_t wave = thread / kDecodeGqa4SplitWaveSize;
  const uint32_t query_head = kv_head * kDecodeGqa4SplitGqaRatio + wave;
  const uint16_t *const query_row =
      query + static_cast<uint64_t>(query_head) * kDecodeGqa4SplitHeadDim;
  constexpr uint32_t kDimensionsPerLane =
      kDecodeGqa4SplitHeadDim / kDecodeGqa4SplitWaveSize;
  float query_values[kDimensionsPerLane];
  float accumulations[kDimensionsPerLane] = {};
#pragma unroll
  for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
    query_values[index] =
        bf16_to_f32(query_row[lane + index * kDecodeGqa4SplitWaveSize]);
  }

  const uint64_t workspace_index =
      (static_cast<uint64_t>(query_head) * kPartitions + partition) *
      kDecodeGqa4SplitWorkspaceStride;
  float *const partial = workspace + workspace_index;
  if (split_begin >= split_end) {
    for (uint32_t index = lane; index < kDecodeGqa4SplitHeadDim;
         index += kDecodeGqa4SplitWaveSize) {
      partial[index] = 0.0F;
    }
    if (lane == 0U) {
      partial[kDecodeGqa4SplitHeadDim] = -INFINITY;
      partial[kDecodeGqa4SplitHeadDim + 1U] = 0.0F;
    }
    return;
  }

  __shared__ float key_tile[kDecodeGqa4SplitTileTokens]
                           [kDecodeGqa4SplitHeadDim];
  __shared__ float value_tile[kDecodeGqa4SplitTileTokens]
                             [kDecodeGqa4SplitHeadDim];
  float local_maximum = -INFINITY;
  float local_denominator = 0.0F;
  for (uint64_t tile_begin = split_begin; tile_begin < split_end;
       tile_begin += kDecodeGqa4SplitTileTokens) {
    const uint64_t remaining = split_end - tile_begin;
    const uint32_t tile_count = remaining < kDecodeGqa4SplitTileTokens
                                    ? static_cast<uint32_t>(remaining)
                                    : kDecodeGqa4SplitTileTokens;
    for (uint32_t element = thread;
         element < tile_count * kDecodeGqa4SplitHeadDim; element += 128U) {
      const uint32_t token = element / kDecodeGqa4SplitHeadDim;
      const uint32_t dimension = element % kDecodeGqa4SplitHeadDim;
      const uint64_t kv_row = (tile_begin + token) * 4U + kv_head;
      key_tile[token][dimension] =
          f16_to_f32(key[kv_row * kDecodeGqa4SplitHeadDim + dimension]);
      value_tile[token][dimension] =
          f16_to_f32(value[kv_row * kDecodeGqa4SplitHeadDim + dimension]);
    }
    __syncthreads();
    for (uint32_t token = 0U; token < tile_count; ++token) {
      float score_partial = 0.0F;
#pragma unroll
      for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
        score_partial +=
            query_values[index] *
            key_tile[token][lane + index * kDecodeGqa4SplitWaveSize];
      }
      for (uint32_t offset = kDecodeGqa4SplitWaveSize / 2U; offset != 0U;
           offset >>= 1U) {
        score_partial +=
            __shfl_down(score_partial, offset, kDecodeGqa4SplitWaveSize);
      }
      float rescale = 0.0F;
      float contribution = 0.0F;
      if (lane == 0U) {
        const float current_score =
            score_partial * rsqrtf(static_cast<float>(kDecodeGqa4SplitHeadDim));
        const float next_maximum = fmaxf(local_maximum, current_score);
        rescale = expf(local_maximum - next_maximum);
        contribution = expf(current_score - next_maximum);
        local_denominator = local_denominator * rescale + contribution;
        local_maximum = next_maximum;
      }
      rescale = __shfl(rescale, 0U, kDecodeGqa4SplitWaveSize);
      contribution = __shfl(contribution, 0U, kDecodeGqa4SplitWaveSize);
#pragma unroll
      for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
        accumulations[index] =
            accumulations[index] * rescale +
            contribution *
                value_tile[token][lane + index * kDecodeGqa4SplitWaveSize];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
    partial[lane + index * kDecodeGqa4SplitWaveSize] = accumulations[index];
  }
  if (lane == 0U) {
    partial[kDecodeGqa4SplitHeadDim] = local_maximum;
    partial[kDecodeGqa4SplitHeadDim + 1U] = local_denominator;
  }
  (void)output;
}

template <uint32_t kPartitions>
__global__
__launch_bounds__(128, 1) void causal_attention_decode_gqa4_split_stage2_kernel(
    const uint16_t *const query, uint16_t *const output, const uint32_t q_heads,
    const float *const workspace) {
  const uint32_t query_head = blockIdx.x;
  if (query_head >= q_heads || q_heads != 16U) {
    return;
  }
  const uint32_t dimension = threadIdx.x;
  const uint64_t base = static_cast<uint64_t>(query_head) * kPartitions *
                        kDecodeGqa4SplitWorkspaceStride;
  __shared__ float maxima[kPartitions];
  __shared__ float denominators[kPartitions];
  __shared__ float global_maximum;
  __shared__ float global_denominator;
  if (dimension < kPartitions) {
    maxima[dimension] =
        workspace[base + dimension * kDecodeGqa4SplitWorkspaceStride +
                  kDecodeGqa4SplitHeadDim];
    denominators[dimension] =
        workspace[base + dimension * kDecodeGqa4SplitWorkspaceStride +
                  kDecodeGqa4SplitHeadDim + 1U];
  }
  __syncthreads();
  if (dimension == 0U) {
    float maximum = maxima[0];
#pragma unroll
    for (uint32_t partition = 1U; partition < kPartitions; ++partition) {
      maximum = fmaxf(maximum, maxima[partition]);
    }
    float denominator = 0.0F;
#pragma unroll
    for (uint32_t partition = 0U; partition < kPartitions; ++partition) {
      denominator +=
          denominators[partition] * expf(maxima[partition] - maximum);
    }
    global_maximum = maximum;
    global_denominator = denominator;
  }
  __syncthreads();
  const uint16_t *const query_row =
      query + static_cast<uint64_t>(query_head) * kDecodeGqa4SplitHeadDim;
  uint16_t *const output_row =
      output + static_cast<uint64_t>(query_head) * kDecodeGqa4SplitHeadDim;
  for (uint32_t current = dimension; current < kDecodeGqa4SplitHeadDim;
       current += 128U) {
    float merged = 0.0F;
#pragma unroll
    for (uint32_t partition = 0U; partition < kPartitions; ++partition) {
      merged += workspace[base + partition * kDecodeGqa4SplitWorkspaceStride +
                          current] *
                expf(maxima[partition] - global_maximum);
    }
    output_row[current] = f32_to_bf16_rne(merged / global_denominator);
  }
  (void)query_row;
}

// Phase 33's one-row provider remains available as a measured control for the
// Phase 35 provider-selection audit.
__global__
__launch_bounds__(256, 1) void causal_attention_prefill_gqa4_shared_kernel(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const float *const key_outer_scales, const float *const value_outer_scales,
    uint16_t *const output, const uint32_t query_count,
    const uint64_t start_position, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim, const uint32_t encoding,
    const float static_key_scale, const float static_value_scale) {
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kWaveCount = 8U;
  constexpr uint32_t kGqaRatio = 4U;
  const uint64_t flat = blockIdx.x;
  const uint64_t row = flat / kv_heads;
  const uint32_t kv_head = static_cast<uint32_t>(flat % kv_heads);
  if (row >= query_count) {
    return;
  }
  const uint64_t query_position = start_position + row;
  const uint32_t dimension = threadIdx.x;
  const uint32_t lane = dimension & (kWaveSize - 1U);
  const uint32_t wave = dimension / kWaveSize;
  const uint32_t first_query_head = kv_head * kGqaRatio;

  __shared__ float reductions[kGqaRatio][kWaveCount];
  __shared__ float rescale[kGqaRatio];
  __shared__ float contribution[kGqaRatio];
  __shared__ float running_maximum[kGqaRatio];
  __shared__ float running_denominator[kGqaRatio];
  if (dimension < kGqaRatio) {
    running_maximum[dimension] = -std::numeric_limits<float>::infinity();
    running_denominator[dimension] = 0.0F;
  }
  __syncthreads();

  float accumulations[kGqaRatio] = {0.0F, 0.0F, 0.0F, 0.0F};
  for (uint64_t key_position = 0U; key_position <= query_position;
       ++key_position) {
    const uint64_t kv_row = key_position * kv_heads + kv_head;
    const float key_value =
        dimension < head_dim
            ? load_kv(key, key_scales, key_outer_scales, encoding, kv_row,
                      dimension, head_dim, static_key_scale)
            : 0.0F;
#pragma unroll
    for (uint32_t head = 0U; head < kGqaRatio; ++head) {
      const uint16_t *const query_row =
          query + (row * q_heads + first_query_head + head) *
                      static_cast<uint64_t>(head_dim);
      float partial = dimension < head_dim
                          ? bf16_to_f32(query_row[dimension]) * key_value
                          : 0.0F;
      for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
        partial += __shfl_down(partial, offset, kWaveSize);
      }
      if (lane == 0U) {
        reductions[head][wave] = partial;
      }
    }
    __syncthreads();
    if (wave < kGqaRatio) {
      float block_sum = lane < kWaveCount ? reductions[wave][lane] : 0.0F;
      for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
        block_sum += __shfl_down(block_sum, offset, kWaveSize);
      }
      if (lane == 0U) {
        const float current_score =
            block_sum * rsqrtf(static_cast<float>(head_dim));
        const float next_maximum = fmaxf(running_maximum[wave], current_score);
        rescale[wave] = expf(running_maximum[wave] - next_maximum);
        contribution[wave] = expf(current_score - next_maximum);
        running_denominator[wave] =
            running_denominator[wave] * rescale[wave] + contribution[wave];
        running_maximum[wave] = next_maximum;
      }
    }
    __syncthreads();
    const float value_element =
        dimension < head_dim
            ? load_kv(value, value_scales, value_outer_scales, encoding, kv_row,
                      dimension, head_dim, static_value_scale)
            : 0.0F;
    if (dimension < head_dim) {
#pragma unroll
      for (uint32_t head = 0U; head < kGqaRatio; ++head) {
        accumulations[head] = accumulations[head] * rescale[head] +
                              contribution[head] * value_element;
      }
    }
    __syncthreads();
  }
  if (dimension < head_dim) {
#pragma unroll
    for (uint32_t head = 0U; head < kGqaRatio; ++head) {
      uint16_t *const output_row =
          output + (row * q_heads + first_query_head + head) *
                       static_cast<uint64_t>(head_dim);
      output_row[dimension] =
          f32_to_bf16_rne(accumulations[head] / running_denominator[head]);
    }
  }
}

// One block owns four adjacent query rows and one KV head. Eight waves each
// own two (row, GQA-head) pairs, so every decoded K/V row is reused by all
// sixteen logical queries while each query preserves causal key order and an
// independent online-softmax state.
__global__
__launch_bounds__(256, 1) void causal_attention_prefill_gqa4_qtile4_kernel(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const float *const key_outer_scales, const float *const value_outer_scales,
    uint16_t *const output, const uint32_t query_count,
    const uint64_t start_position, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim, const uint32_t encoding,
    const float static_key_scale, const float static_value_scale) {
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kWaveCount = 8U;
  constexpr uint32_t kGqaRatio = 4U;
  constexpr uint32_t kQueryTile = 4U;
  constexpr uint32_t kLogicalQueries = kGqaRatio * kQueryTile;
  constexpr uint32_t kQueriesPerWave = kLogicalQueries / kWaveCount;
  constexpr uint32_t kHeadDim = SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM;
  constexpr uint32_t kDimensionsPerLane = kHeadDim / kWaveSize;
  const uint64_t flat = blockIdx.x;
  const uint64_t tile = flat / kv_heads;
  const uint32_t kv_head = static_cast<uint32_t>(flat % kv_heads);
  const uint64_t first_row = tile * kQueryTile;
  if (first_row >= query_count) {
    return;
  }
  const uint32_t dimension = threadIdx.x;
  const uint32_t lane = dimension & (kWaveSize - 1U);
  const uint32_t wave = dimension / kWaveSize;
  const uint32_t first_query_head = kv_head * kGqaRatio;

  __shared__ float key_tile[kHeadDim];
  __shared__ float value_tile[kHeadDim];
  float query_values[kQueriesPerWave][kDimensionsPerLane];
  float accumulations[kQueriesPerWave][kDimensionsPerLane] = {};
  float running_maximum[kQueriesPerWave];
  float running_denominator[kQueriesPerWave];
#pragma unroll
  for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
    running_maximum[item] = -std::numeric_limits<float>::infinity();
    running_denominator[item] = 0.0F;
    const uint32_t logical_query = wave * kQueriesPerWave + item;
    const uint64_t row = first_row + logical_query / kGqaRatio;
    const uint64_t safe_row =
        row < query_count ? row : static_cast<uint64_t>(query_count - 1U);
    const uint32_t query_head = first_query_head + logical_query % kGqaRatio;
    const uint16_t *const query_row =
        query +
        (safe_row * q_heads + query_head) * static_cast<uint64_t>(head_dim);
#pragma unroll
    for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
      const uint32_t current = lane + index * kWaveSize;
      query_values[item][index] = row < query_count && current < head_dim
                                      ? bf16_to_f32(query_row[current])
                                      : 0.0F;
    }
  }

  const uint64_t tile_end = first_row + kQueryTile;
  const uint64_t last_row =
      (tile_end < query_count ? tile_end : query_count) - 1U;
  const uint64_t last_query_position = start_position + last_row;
  for (uint64_t key_position = 0U; key_position <= last_query_position;
       ++key_position) {
    const uint64_t kv_row = key_position * kv_heads + kv_head;
    if (dimension < head_dim) {
      key_tile[dimension] =
          load_kv(key, key_scales, key_outer_scales, encoding, kv_row,
                  dimension, head_dim, static_key_scale);
      value_tile[dimension] =
          load_kv(value, value_scales, value_outer_scales, encoding, kv_row,
                  dimension, head_dim, static_value_scale);
    }
    __syncthreads();

#pragma unroll
    for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
      const uint32_t logical_query = wave * kQueriesPerWave + item;
      const uint64_t row = first_row + logical_query / kGqaRatio;
      const bool active =
          row < query_count && key_position <= start_position + row;
      float products[kDimensionsPerLane];
#pragma unroll
      for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
        const uint32_t current = lane + index * kWaveSize;
        products[index] = active && current < head_dim
                              ? query_values[item][index] * key_tile[current]
                              : 0.0F;
      }
      const float pair0 = products[0] + products[1];
      const float pair1 = products[2] + products[3];
      const float pair2 = products[4] + products[5];
      const float pair3 = products[6] + products[7];
      float partial = (pair0 + pair1) + (pair2 + pair3);
      for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
        partial += __shfl_down(partial, offset, kWaveSize);
      }
      float rescale = 1.0F;
      float contribution = 0.0F;
      float next_maximum = running_maximum[item];
      if (lane == 0U) {
        if (active) {
          const float current_score =
              partial * rsqrtf(static_cast<float>(head_dim));
          next_maximum = fmaxf(running_maximum[item], current_score);
          rescale = expf(running_maximum[item] - next_maximum);
          contribution = expf(current_score - next_maximum);
        }
      }
      rescale = __shfl(rescale, 0U, kWaveSize);
      contribution = __shfl(contribution, 0U, kWaveSize);
      next_maximum = __shfl(next_maximum, 0U, kWaveSize);
      running_denominator[item] =
          running_denominator[item] * rescale + contribution;
      running_maximum[item] = next_maximum;
#pragma unroll
      for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
        const uint32_t current = lane + index * kWaveSize;
        if (active && current < head_dim) {
          accumulations[item][index] = accumulations[item][index] * rescale +
                                       contribution * value_tile[current];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
    const uint32_t logical_query = wave * kQueriesPerWave + item;
    const uint64_t row = first_row + logical_query / kGqaRatio;
    const uint32_t query_head = first_query_head + logical_query % kGqaRatio;
    if (row < query_count) {
      uint16_t *const output_row = output + (row * q_heads + query_head) *
                                                static_cast<uint64_t>(head_dim);
#pragma unroll
      for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
        const uint32_t current = lane + index * kWaveSize;
        if (current < head_dim) {
          output_row[current] = f32_to_bf16_rne(accumulations[item][index] /
                                                running_denominator[item]);
        }
      }
    }
  }
}

// Long-prefill v2 follows the llama.cpp tile shape (8 query rows x GQA4 and
// 32-key tiles), but keeps the public BF16-Q/FP16-KV contract.  Stage 1 splits
// the key range across independent blocks and stores online-softmax partials;
// stage 2 combines those partials.  The host launcher runs fixed 256-row
// chunks, so the workspace stays bounded for 100k-token prompts.
constexpr uint32_t kLongPrefillV2Partitions = 16U;
constexpr uint32_t kLongPrefillV2TileTokens = 32U;
constexpr uint32_t kLongPrefillV2QueryTile = 8U;
constexpr uint32_t kLongPrefillV2GqaRatio = 4U;
constexpr uint32_t kLongPrefillV2WaveSize = 32U;
constexpr uint32_t kLongPrefillV2Threads = 256U;
constexpr uint32_t kLongPrefillV2QueriesPerWave =
    (kLongPrefillV2QueryTile * kLongPrefillV2GqaRatio) /
    (kLongPrefillV2Threads / kLongPrefillV2WaveSize);
constexpr uint32_t kLongPrefillV2PairsPerLane =
    SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM / kLongPrefillV2WaveSize / 2U;
constexpr uint32_t kLongPrefillV2WorkspaceStride =
    SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM + 2U;

__global__ __launch_bounds__(
    kLongPrefillV2Threads,
    1) void causal_attention_long_prefill_v2_stage1_kernel(const uint16_t
                                                               *const query,
                                                           const uint16_t
                                                               *const key,
                                                           const uint16_t
                                                               *const value,
                                                           const uint32_t
                                                               query_count,
                                                           const uint64_t
                                                               start_position,
                                                           const uint64_t
                                                               row_offset,
                                                           const uint64_t
                                                               committed_kv_length,
                                                           const uint32_t
                                                               q_heads,
                                                           const uint32_t
                                                               kv_heads,
                                                           float *const
                                                               workspace) {
  constexpr uint32_t kHeadDim = SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM;
  const uint64_t flat = blockIdx.x;
  const uint64_t tile =
      flat / (static_cast<uint64_t>(kv_heads) * kLongPrefillV2Partitions);
  const uint32_t tile_remainder = static_cast<uint32_t>(
      flat % (static_cast<uint64_t>(kv_heads) * kLongPrefillV2Partitions));
  const uint32_t kv_head = tile_remainder / kLongPrefillV2Partitions;
  const uint32_t partition = tile_remainder % kLongPrefillV2Partitions;
  const uint64_t first_row = tile * kLongPrefillV2QueryTile;
  if (first_row >= query_count || kv_head >= kv_heads) {
    return;
  }
  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (kLongPrefillV2WaveSize - 1U);
  const uint32_t wave = thread / kLongPrefillV2WaveSize;
  const uint32_t first_query_head = kv_head * kLongPrefillV2GqaRatio;

  // Four logical query/head pairs per wave.  Each lane owns four half2
  // elements (eight dimensions), matching the 256-thread/8-wave tile shape.
  float2 query_values[kLongPrefillV2QueriesPerWave][kLongPrefillV2PairsPerLane];
  float2 accumulations[kLongPrefillV2QueriesPerWave]
                      [kLongPrefillV2PairsPerLane] = {};
  float running_maximum[kLongPrefillV2QueriesPerWave];
  float running_denominator[kLongPrefillV2QueriesPerWave];
#pragma unroll
  for (uint32_t item = 0U; item < kLongPrefillV2QueriesPerWave; ++item) {
    running_maximum[item] = -INFINITY;
    running_denominator[item] = 0.0F;
    const uint32_t logical_query = wave * kLongPrefillV2QueriesPerWave + item;
    const uint64_t row = first_row + logical_query / kLongPrefillV2GqaRatio;
    const uint32_t query_head =
        first_query_head + logical_query % kLongPrefillV2GqaRatio;
    const uint64_t safe_row = row < query_count ? row : query_count - 1U;
    const uint16_t *const query_row =
        query +
        (safe_row * q_heads + query_head) * static_cast<uint64_t>(kHeadDim);
#pragma unroll
    for (uint32_t pair = 0U; pair < kLongPrefillV2PairsPerLane; ++pair) {
      const uint32_t dimension = lane * 2U + pair * kLongPrefillV2WaveSize * 2U;
      query_values[item][pair] = row < query_count
                                     ? load_bf16_pair(query_row + dimension)
                                     : make_float2(0.0F, 0.0F);
    }
  }

  __shared__ uint16_t key_tile[kLongPrefillV2TileTokens][kHeadDim];
  __shared__ uint16_t value_tile[kLongPrefillV2TileTokens][kHeadDim];
  const uint64_t split_begin =
      committed_kv_length * partition / kLongPrefillV2Partitions;
  const uint64_t split_end =
      committed_kv_length * (partition + 1U) / kLongPrefillV2Partitions;
  for (uint64_t tile_begin = split_begin; tile_begin < split_end;
       tile_begin += kLongPrefillV2TileTokens) {
    const uint64_t remaining = split_end - tile_begin;
    const uint32_t tile_count = remaining < kLongPrefillV2TileTokens
                                    ? static_cast<uint32_t>(remaining)
                                    : kLongPrefillV2TileTokens;
    for (uint32_t element = thread; element < tile_count * kHeadDim;
         element += kLongPrefillV2Threads) {
      const uint32_t token = element / kHeadDim;
      const uint32_t dimension = element % kHeadDim;
      const uint64_t kv_row = (tile_begin + token) * kv_heads + kv_head;
      key_tile[token][dimension] = key[kv_row * kHeadDim + dimension];
      value_tile[token][dimension] = value[kv_row * kHeadDim + dimension];
    }
    __syncthreads();

    for (uint32_t token = 0U; token < tile_count; ++token) {
#pragma unroll
      for (uint32_t item = 0U; item < kLongPrefillV2QueriesPerWave; ++item) {
        const uint32_t logical_query =
            wave * kLongPrefillV2QueriesPerWave + item;
        const uint64_t row = first_row + logical_query / kLongPrefillV2GqaRatio;
        const uint64_t query_position = start_position + row_offset + row;
        const bool active =
            row < query_count && tile_begin + token <= query_position;
        if (!active) {
          continue;
        }
        float partial = 0.0F;
#pragma unroll
        for (uint32_t pair = 0U; pair < kLongPrefillV2PairsPerLane; ++pair) {
          const uint32_t dimension =
              lane * 2U + pair * kLongPrefillV2WaveSize * 2U;
          const float2 q = query_values[item][pair];
          const float2 k = load_fp16_pair(key_tile[token] + dimension);
          partial += q.x * k.x + q.y * k.y;
        }
        for (uint32_t offset = kLongPrefillV2WaveSize / 2U; offset != 0U;
             offset >>= 1U) {
          partial += __shfl_down(partial, offset, kLongPrefillV2WaveSize);
        }
        float rescale = 1.0F;
        float contribution = 0.0F;
        float next_maximum = running_maximum[item];
        if (lane == 0U) {
          const float current_score =
              partial * rsqrtf(static_cast<float>(kHeadDim));
          next_maximum = fmaxf(running_maximum[item], current_score);
          rescale = expf(running_maximum[item] - next_maximum);
          contribution = expf(current_score - next_maximum);
        }
        rescale = __shfl(rescale, 0U, kLongPrefillV2WaveSize);
        contribution = __shfl(contribution, 0U, kLongPrefillV2WaveSize);
        next_maximum = __shfl(next_maximum, 0U, kLongPrefillV2WaveSize);
        running_denominator[item] =
            running_denominator[item] * rescale + contribution;
        running_maximum[item] = next_maximum;
#pragma unroll
        for (uint32_t pair = 0U; pair < kLongPrefillV2PairsPerLane; ++pair) {
          const uint32_t dimension =
              lane * 2U + pair * kLongPrefillV2WaveSize * 2U;
          const float2 v = load_fp16_pair(value_tile[token] + dimension);
          accumulations[item][pair].x =
              accumulations[item][pair].x * rescale + contribution * v.x;
          accumulations[item][pair].y =
              accumulations[item][pair].y * rescale + contribution * v.y;
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t item = 0U; item < kLongPrefillV2QueriesPerWave; ++item) {
    const uint32_t logical_query = wave * kLongPrefillV2QueriesPerWave + item;
    const uint64_t row = first_row + logical_query / kLongPrefillV2GqaRatio;
    const uint32_t query_head =
        first_query_head + logical_query % kLongPrefillV2GqaRatio;
    if (row >= query_count) {
      continue;
    }
    const uint64_t base =
        ((row * q_heads + query_head) * kLongPrefillV2Partitions + partition) *
        kLongPrefillV2WorkspaceStride;
    float *const partial = workspace + base;
#pragma unroll
    for (uint32_t pair = 0U; pair < kLongPrefillV2PairsPerLane; ++pair) {
      const uint32_t dimension = lane * 2U + pair * kLongPrefillV2WaveSize * 2U;
      partial[dimension] = accumulations[item][pair].x;
      partial[dimension + 1U] = accumulations[item][pair].y;
    }
    if (lane == 0U) {
      partial[kHeadDim] = running_maximum[item];
      partial[kHeadDim + 1U] = running_denominator[item];
    }
  }
}

__global__ __launch_bounds__(
    kLongPrefillV2Threads,
    1) void causal_attention_long_prefill_v2_combine_kernel(const float *const
                                                                workspace,
                                                            uint16_t
                                                                *const output,
                                                            const uint32_t
                                                                query_count,
                                                            const uint32_t
                                                                q_heads,
                                                            const uint64_t
                                                                row_offset) {
  constexpr uint32_t kHeadDim = SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM;
  const uint64_t flat = blockIdx.x;
  const uint64_t row = flat / q_heads;
  const uint32_t query_head = static_cast<uint32_t>(flat % q_heads);
  if (row >= query_count) {
    return;
  }
  const uint32_t dimension = threadIdx.x;
  const uint64_t base = (row * q_heads + query_head) *
                        kLongPrefillV2Partitions *
                        kLongPrefillV2WorkspaceStride;
  __shared__ float maxima[kLongPrefillV2Partitions];
  __shared__ float denominators[kLongPrefillV2Partitions];
  __shared__ float global_maximum;
  __shared__ float global_denominator;
  if (dimension < kLongPrefillV2Partitions) {
    const float *const partial =
        workspace + base + dimension * kLongPrefillV2WorkspaceStride;
    maxima[dimension] = partial[kHeadDim];
    denominators[dimension] = partial[kHeadDim + 1U];
  }
  __syncthreads();
  if (dimension == 0U) {
    float maximum = maxima[0];
#pragma unroll
    for (uint32_t partition = 1U; partition < kLongPrefillV2Partitions;
         ++partition) {
      maximum = fmaxf(maximum, maxima[partition]);
    }
    float denominator = 0.0F;
#pragma unroll
    for (uint32_t partition = 0U; partition < kLongPrefillV2Partitions;
         ++partition) {
      denominator +=
          denominators[partition] * expf(maxima[partition] - maximum);
    }
    global_maximum = maximum;
    global_denominator = denominator;
  }
  __syncthreads();
  const uint64_t output_row_index = row + row_offset;
  uint16_t *const output_row =
      output + (row * q_heads + query_head) * static_cast<uint64_t>(kHeadDim);
  if (dimension < kHeadDim) {
    float merged = 0.0F;
#pragma unroll
    for (uint32_t partition = 0U; partition < kLongPrefillV2Partitions;
         ++partition) {
      const float *const partial =
          workspace + base + partition * kLongPrefillV2WorkspaceStride;
      merged += partial[dimension] * expf(maxima[partition] - global_maximum);
    }
    output_row[dimension] = global_denominator == 0.0F
                                ? UINT16_C(0)
                                : f32_to_bf16_rne(merged / global_denominator);
  }
  (void)output_row_index;
}

// The scaled prefill prototype keeps the public BF16-Q/FP16-KV storage
// contract while doing all GEMM operands in FP16.  Each query row/head gets a
// power-of-two scale before narrowing, and the inverse scale is applied only
// to the FP32 score during softmax.  This preserves large (2^20) and tiny
// BF16 values without changing the public tensor representation.
__device__ uint16_t scaled_f32_to_f16_rne(const float value) noexcept {
  const uint32_t bits = __float_as_uint(value);
  const uint16_t sign = static_cast<uint16_t>((bits >> 16U) & 0x8000U);
  const uint32_t exponent = (bits >> 23U) & 0xffU;
  const uint32_t fraction = bits & 0x007fffffU;
  if (exponent == 0xffU) {
    return static_cast<uint16_t>(sign | (fraction == 0U ? 0x7c00U : 0x7e00U));
  }
  int32_t unbiased = static_cast<int32_t>(exponent) - 127;
  uint32_t normalized_fraction = fraction;
  if (exponent == 0U && fraction != 0U) {
    // BF16 subnormals arrive as FP32 subnormals after conversion.  Normalize
    // them before applying the FP16 exponent/subnormal boundary logic.
    unbiased = -126;
    while ((normalized_fraction & 0x00800000U) == 0U) {
      normalized_fraction <<= 1U;
      --unbiased;
    }
    normalized_fraction &= 0x007fffffU;
  }
  if (unbiased > 15) {
    return static_cast<uint16_t>(sign | 0x7c00U);
  }
  if (unbiased >= -14) {
    uint32_t rounded = normalized_fraction + 0x1000U;
    uint32_t half_exponent = static_cast<uint32_t>(unbiased + 15);
    if (rounded > 0x7fffffU) {
      rounded = 0U;
      ++half_exponent;
    }
    if (half_exponent >= 0x1fU) {
      return static_cast<uint16_t>(sign | 0x7c00U);
    }
    return static_cast<uint16_t>(sign | (half_exponent << 10U) |
                                 ((rounded >> 13U) & 0x03ffU));
  }
  if (unbiased < -24) {
    return sign;
  }
  const uint32_t mantissa = normalized_fraction | 0x00800000U;
  const uint32_t shift = static_cast<uint32_t>(-unbiased - 14);
  const uint32_t rounding_shift = shift + 13U;
  uint32_t rounded = mantissa >> rounding_shift;
  const uint32_t remainder_mask = (UINT32_C(1) << rounding_shift) - 1U;
  const uint32_t remainder = mantissa & remainder_mask;
  const uint32_t halfway = UINT32_C(1) << (rounding_shift - 1U);
  if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0U)) {
    ++rounded;
  }
  if (rounded >= 0x0400U) {
    return static_cast<uint16_t>(sign | 0x0400U);
  }
  return static_cast<uint16_t>(sign | (rounded & 0x03ffU));
}

// Probabilities are non-negative.  Keep the representable FP16 part below
// the F32 value so the omitted mass can be carried by the residual pass.
__device__ uint16_t scaled_f32_to_f16_floor(const float value) noexcept {
  if (!(value > 0.0F) || !isfinite(value)) {
    return value == INFINITY ? UINT16_C(0x7c00) : UINT16_C(0);
  }
  uint16_t rounded = scaled_f32_to_f16_rne(value);
  if (rounded != 0U && f16_to_f32(rounded) > value) {
    --rounded;
  }
  return rounded;
}

__device__ int scaled_unbiased_exponent(const float value) noexcept {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t exponent = (bits >> 23U) & 0xffU;
  if (exponent != 0U) {
    return static_cast<int>(exponent) - 127;
  }
  uint32_t fraction = bits & 0x007fffffU;
  int unbiased = -126;
  while (fraction != 0U && (fraction & 0x00800000U) == 0U) {
    fraction <<= 1U;
    --unbiased;
  }
  return unbiased;
}

__global__ __launch_bounds__(256, 1) void scaled_prefill_pack_query_kernel(
    const uint16_t *const query, uint16_t *const packed, float *const scales,
    const uint32_t rows, const uint32_t q_heads, const uint32_t head_dim,
    const uint32_t kv_head, const uint32_t q_group, const uint32_t row_offset) {
  const uint32_t packed_row = blockIdx.x;
  const uint32_t packed_rows = rows * q_group;
  if (packed_row >= packed_rows) {
    return;
  }
  const uint32_t row = packed_row / q_group;
  const uint32_t local_head = packed_row % q_group;
  const uint32_t query_head = kv_head * q_group + local_head;
  const uint64_t source_row = static_cast<uint64_t>(row_offset) + row;
  __shared__ float maximums[256];
  __shared__ uint32_t nan_flags[256];
  const uint32_t dimension = threadIdx.x;
  float value = 0.0F;
  uint32_t nan = 0U;
  if (dimension < head_dim) {
    value = bf16_to_f32(
        query[(source_row * q_heads + query_head) * head_dim + dimension]);
    nan = isnan(value) ? 1U : 0U;
  }
  maximums[dimension] = nan != 0U ? 0.0F : fabsf(value);
  nan_flags[dimension] = nan;
  __syncthreads();
  for (uint32_t stride = 128U; stride != 0U; stride >>= 1U) {
    if (dimension < stride) {
      maximums[dimension] =
          fmaxf(maximums[dimension], maximums[dimension + stride]);
      nan_flags[dimension] |= nan_flags[dimension + stride];
    }
    __syncthreads();
  }
  float scale = 1.0F;
  if (nan_flags[0] == 0U && isfinite(maximums[0]) && maximums[0] != 0.0F) {
    int scale_exponent = 15 - scaled_unbiased_exponent(maximums[0]);
    scale_exponent = max(-127, min(127, scale_exponent));
    scale = ldexpf(1.0F, scale_exponent);
  }
  if (dimension == 0U) {
    scales[packed_row] = scale;
  }
  __syncthreads();
  if (dimension < head_dim) {
    packed[static_cast<uint64_t>(packed_row) * head_dim + dimension] =
        scaled_f32_to_f16_rne(value * scale);
  }
}

__global__ __launch_bounds__(256, 1) void scaled_prefill_pack_kv_kernel(
    const uint16_t *const key, const uint16_t *const value,
    uint16_t *const key_pack, uint16_t *const value_pack,
    uint64_t *const special_first, const uint64_t key_length,
    const uint32_t kv_heads, const uint32_t head_dim, const uint32_t kv_head) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t elements = key_length * static_cast<uint64_t>(head_dim);
  if (index >= elements) {
    return;
  }
  const uint64_t token = index / head_dim;
  const uint32_t dimension = static_cast<uint32_t>(index % head_dim);
  const uint64_t source =
      (token * kv_heads + kv_head) * static_cast<uint64_t>(head_dim) +
      dimension;
  key_pack[index] = key[source];
  const uint16_t value_raw = value[source];
  const uint32_t value_bits = static_cast<uint32_t>(value_raw);
  const uint32_t exponent = (value_bits >> 10U) & 0x1fU;
  const uint32_t fraction = value_bits & 0x03ffU;
  if (exponent == 0x1fU) {
    const uint32_t kind =
        fraction != 0U ? 0U : ((value_bits & 0x8000U) != 0U ? 2U : 1U);
    const uint64_t flag =
        (static_cast<uint64_t>(kv_head) * head_dim + dimension) * 3U + kind;
    atomicMin(reinterpret_cast<unsigned long long *>(special_first + flag),
              static_cast<unsigned long long>(token));
    value_pack[index] = 0U;
  } else {
    value_pack[index] = value_raw;
  }
}

__global__ __launch_bounds__(256, 1) void scaled_prefill_softmax_fp16_kernel(
    float *const scores, uint16_t *const probabilities,
    uint16_t *const residuals, const uint32_t rows, const uint64_t key_length,
    const uint64_t start_position, const uint32_t row_offset,
    const uint32_t q_group, const float *const scales, const float scale) {
  const uint32_t packed_row = blockIdx.x;
  if (packed_row >= rows || q_group == 0U) {
    return;
  }
  const uint32_t row = packed_row / q_group;
  const uint64_t requested_end = start_position + row_offset + row + 1U;
  const uint64_t causal_end =
      requested_end < key_length ? requested_end : key_length;
  __shared__ float reduction[256];
  __shared__ uint32_t nan_scores[256];
  const float inverse_query_scale = 1.0F / scales[packed_row];
  float local_maximum = -INFINITY;
  uint32_t local_nan = 0U;
  for (uint64_t key = threadIdx.x; key < causal_end; key += blockDim.x) {
    const float score =
        scores[static_cast<uint64_t>(packed_row) * key_length + key] *
        inverse_query_scale * scale;
    local_nan |= isnan(score) ? 1U : 0U;
    local_maximum = fmaxf(local_maximum, score);
  }
  reduction[threadIdx.x] = local_maximum;
  nan_scores[threadIdx.x] = local_nan;
  __syncthreads();
  for (uint32_t stride = 128U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] =
          fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
      nan_scores[threadIdx.x] |= nan_scores[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float maximum = reduction[0];
  float local_denominator = 0.0F;
  for (uint64_t key = threadIdx.x; key < causal_end; key += blockDim.x) {
    local_denominator +=
        expf(scores[static_cast<uint64_t>(packed_row) * key_length + key] *
                 inverse_query_scale * scale -
             maximum);
  }
  reduction[threadIdx.x] = local_denominator;
  __syncthreads();
  for (uint32_t stride = 128U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float denominator = reduction[0];
  constexpr float kResidualScale = 1024.0F;
  for (uint64_t key = threadIdx.x; key < key_length; key += blockDim.x) {
    if (nan_scores[0] != 0U) {
      probabilities[static_cast<uint64_t>(packed_row) * key_length + key] =
          UINT16_C(0x7e00);
      residuals[static_cast<uint64_t>(packed_row) * key_length + key] =
          UINT16_C(0x7e00);
      continue;
    }
    float probability = 0.0F;
    if (key < causal_end) {
      probability =
          expf(scores[static_cast<uint64_t>(packed_row) * key_length + key] *
                   inverse_query_scale * scale -
               maximum) /
          denominator;
    }
    const uint16_t high = scaled_f32_to_f16_floor(probability);
    const float residual =
        fmaxf(0.0F, probability - f16_to_f32(high)) * kResidualScale;
    probabilities[static_cast<uint64_t>(packed_row) * key_length + key] = high;
    residuals[static_cast<uint64_t>(packed_row) * key_length + key] =
        scaled_f32_to_f16_floor(residual);
  }
}

__global__ __launch_bounds__(256, 1) void scaled_prefill_combine_kernel(
    float *const output, const float *const residual_output,
    const uint16_t *const probabilities, const uint16_t *const residuals,
    const uint64_t *const special_first, const uint16_t *const value,
    const uint32_t rows, const uint32_t kv_heads, const uint64_t key_length,
    const uint64_t start_position, const uint32_t row_offset,
    const uint32_t q_group, const uint32_t head_dim, const uint32_t kv_head) {
  constexpr float kResidualScale = 1024.0F;
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t elements = static_cast<uint64_t>(rows) * q_group * head_dim;
  if (index >= elements || q_group == 0U) {
    return;
  }
  const uint32_t packed_row = static_cast<uint32_t>(index / head_dim);
  const uint32_t dimension = static_cast<uint32_t>(index % head_dim);
  const uint32_t row = packed_row / q_group;
  const uint64_t requested_end = start_position + row_offset + row + 1U;
  const uint64_t causal_end =
      requested_end < key_length ? requested_end : key_length;
  output[index] += residual_output[index] / kResidualScale;
  const uint64_t probability_row =
      static_cast<uint64_t>(packed_row) * key_length;
  const auto positive_probability = [&](const uint64_t token) {
    return token < causal_end &&
           f16_to_f32(probabilities[probability_row + token]) +
                   f16_to_f32(residuals[probability_row + token]) /
                       kResidualScale >
               0.0F;
  };
  const uint64_t base =
      (static_cast<uint64_t>(kv_head) * head_dim + dimension) * 3U;
  const bool has_special = special_first[base] != UINT64_MAX ||
                           special_first[base + 1U] != UINT64_MAX ||
                           special_first[base + 2U] != UINT64_MAX;
  if (has_special) {
    bool nan_visible = false;
    bool positive_visible = false;
    bool negative_visible = false;
    for (uint64_t token = 0U; token < causal_end; ++token) {
      if (!positive_probability(token)) {
        continue;
      }
      const uint16_t raw =
          value[(token * kv_heads + kv_head) * head_dim + dimension];
      const uint32_t bits = static_cast<uint32_t>(raw);
      if (((bits >> 10U) & 0x1fU) != 0x1fU) {
        continue;
      }
      if ((bits & 0x03ffU) != 0U) {
        nan_visible = true;
      } else if ((bits & 0x8000U) != 0U) {
        negative_visible = true;
      } else {
        positive_visible = true;
      }
    }
    if (nan_visible || (positive_visible && negative_visible)) {
      output[index] = __int_as_float(UINT32_C(0x7fc00000));
    } else if (positive_visible) {
      output[index] = INFINITY;
    } else if (negative_visible) {
      output[index] = -INFINITY;
    }
  }
}

__global__ __launch_bounds__(256, 1) void scaled_prefill_scatter_kernel(
    const float *const input, uint16_t *const output, const uint32_t rows,
    const uint32_t q_heads, const uint32_t head_dim, const uint32_t kv_head,
    const uint32_t q_group, const uint32_t row_offset) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t elements = static_cast<uint64_t>(rows) * q_group * head_dim;
  if (index >= elements) {
    return;
  }
  const uint32_t packed_row = static_cast<uint32_t>(index / head_dim);
  const uint32_t dimension = static_cast<uint32_t>(index % head_dim);
  const uint32_t row = packed_row / q_group;
  const uint32_t local_head = packed_row % q_group;
  const uint32_t query_head = kv_head * q_group + local_head;
  output[((static_cast<uint64_t>(row_offset) + row) * q_heads + query_head) *
             head_dim +
         dimension] = f32_to_bf16_rne(input[index]);
}

} // namespace

hipError_t launch_decode_wave_split_fp16_pair(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const float *const key_outer_scales, const float *const value_outer_scales,
    uint16_t *const output, const uint32_t query_count,
    const uint64_t start_position, const uint64_t committed_kv_length,
    const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim,
    const uint32_t encoding, const float static_key_scale,
    const float static_value_scale, const hipStream_t stream) noexcept {
  if (query == nullptr || key == nullptr || value == nullptr ||
      output == nullptr || query_count != 1U || q_heads != 16U ||
      kv_heads != 4U || head_dim != SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM ||
      encoding != SLLM_HIP_KV_ENCODING_FP16_V1 ||
      start_position + 1U != committed_kv_length ||
      committed_kv_length < 1024U) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(causal_attention_decode_wave_split_fp16_pair_kernel),
      dim3(q_heads), dim3(SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE), 0U, stream,
      query, key, value, key_scales, value_scales, key_outer_scales,
      value_outer_scales, output, committed_kv_length, q_heads, kv_heads,
      head_dim, encoding, static_key_scale, static_value_scale);
  return hipGetLastError();
}

template <uint32_t kPartitions>
hipError_t launch_decode_gqa4_split_impl(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const float *const key_outer_scales, const float *const value_outer_scales,
    uint16_t *const output, const uint32_t query_count,
    const uint64_t start_position, const uint64_t committed_kv_length,
    const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim,
    const uint32_t encoding, const float static_key_scale,
    const float static_value_scale, void *const workspace,
    const uint64_t workspace_bytes, const hipStream_t stream) noexcept {
  (void)key_scales;
  (void)value_scales;
  (void)key_outer_scales;
  (void)value_outer_scales;
  (void)static_key_scale;
  (void)static_value_scale;
  constexpr uint64_t kRequiredBytes =
      static_cast<uint64_t>(16U) * kPartitions * (256U + 2U) * sizeof(float);
  if (query == nullptr || key == nullptr || value == nullptr ||
      output == nullptr || workspace == nullptr || query_count != 1U ||
      start_position + 1U != committed_kv_length || q_heads != 16U ||
      kv_heads != 4U || head_dim != 256U ||
      encoding != SLLM_HIP_KV_ENCODING_FP16_V1 ||
      workspace_bytes < kRequiredBytes) {
    return hipErrorInvalidValue;
  }
  const auto *const key_fp16 = static_cast<const uint16_t *>(key);
  const auto *const value_fp16 = static_cast<const uint16_t *>(value);
  hipLaunchKernelGGL(
      causal_attention_decode_gqa4_split_stage1_kernel<kPartitions>,
      dim3(kv_heads * kPartitions), dim3(128U), 0U, stream, query, key_fp16,
      value_fp16, output, committed_kv_length, static_cast<float *>(workspace));
  hipError_t status = hipGetLastError();
  if (status != hipSuccess) {
    return status;
  }
  hipLaunchKernelGGL(
      causal_attention_decode_gqa4_split_stage2_kernel<kPartitions>,
      dim3(q_heads), dim3(128U), 0U, stream, query, output, q_heads,
      static_cast<const float *>(workspace));
  return hipGetLastError();
}

hipError_t launch_decode_gqa4_split(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const float *const key_outer_scales, const float *const value_outer_scales,
    uint16_t *const output, const uint32_t query_count,
    const uint64_t start_position, const uint64_t committed_kv_length,
    const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim,
    const uint32_t encoding, const float static_key_scale,
    const float static_value_scale, void *const workspace,
    const uint64_t workspace_bytes, const hipStream_t stream) noexcept {
  return launch_decode_gqa4_split_impl<kDecodeGqa4SplitPartitions>(
      query, key, value, key_scales, value_scales, key_outer_scales,
      value_outer_scales, output, query_count, start_position,
      committed_kv_length, q_heads, kv_heads, head_dim, encoding,
      static_key_scale, static_value_scale, workspace, workspace_bytes, stream);
}

hipError_t launch_decode_gqa4_split_p32(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const float *const key_outer_scales, const float *const value_outer_scales,
    uint16_t *const output, const uint32_t query_count,
    const uint64_t start_position, const uint64_t committed_kv_length,
    const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim,
    const uint32_t encoding, const float static_key_scale,
    const float static_value_scale, void *const workspace,
    const uint64_t workspace_bytes, const hipStream_t stream) noexcept {
  return launch_decode_gqa4_split_impl<32U>(
      query, key, value, key_scales, value_scales, key_outer_scales,
      value_outer_scales, output, query_count, start_position,
      committed_kv_length, q_heads, kv_heads, head_dim, encoding,
      static_key_scale, static_value_scale, workspace, workspace_bytes, stream);
}

hipError_t launch_long_prefill_v2(
    const uint16_t *const query, const void *const key, const void *const value,
    uint16_t *const output, const uint32_t query_count,
    const uint64_t start_position, const uint64_t committed_kv_length,
    const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim,
    void *const workspace, const uint64_t workspace_bytes,
    const hipStream_t stream) noexcept {
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
  (void)query;
  (void)key;
  (void)value;
  (void)output;
  (void)query_count;
  (void)start_position;
  (void)committed_kv_length;
  (void)q_heads;
  (void)kv_heads;
  (void)head_dim;
  (void)workspace;
  (void)workspace_bytes;
  (void)stream;
  return hipErrorNotSupported;
#else
  constexpr uint32_t kChunkRows = 256U;
  constexpr uint32_t kExpectedQHeads = 16U;
  constexpr uint32_t kExpectedKvHeads = 4U;
  constexpr uint32_t kExpectedHeadDim = SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM;
  constexpr uint64_t kRequiredBytes =
      static_cast<uint64_t>(kChunkRows) * kExpectedQHeads *
      kLongPrefillV2Partitions * (kExpectedHeadDim + 2U) * sizeof(float);
  if (query == nullptr || key == nullptr || value == nullptr ||
      output == nullptr || workspace == nullptr || query_count < 1024U ||
      query_count > SLLM_HIP_CAUSAL_ATTENTION_MAX_M ||
      q_heads != kExpectedQHeads || kv_heads != kExpectedKvHeads ||
      head_dim != kExpectedHeadDim || committed_kv_length == 0U ||
      committed_kv_length < query_count ||
      committed_kv_length > SLLM_HIP_CAUSAL_ATTENTION_MAX_M ||
      start_position > UINT64_MAX - query_count ||
      start_position + query_count != committed_kv_length ||
      workspace_bytes < kRequiredBytes) {
    return hipErrorInvalidValue;
  }
  const auto *const key_fp16 = static_cast<const uint16_t *>(key);
  const auto *const value_fp16 = static_cast<const uint16_t *>(value);
  const uint64_t row_elements =
      static_cast<uint64_t>(q_heads) * static_cast<uint64_t>(head_dim);
  for (uint32_t row_offset = 0U; row_offset < query_count;
       row_offset += kChunkRows) {
    const uint32_t rows = std::min(kChunkRows, query_count - row_offset);
    const uint64_t tile_count =
        (static_cast<uint64_t>(rows) + kLongPrefillV2QueryTile - 1U) /
        kLongPrefillV2QueryTile;
    const uint64_t stage1_blocks =
        tile_count * static_cast<uint64_t>(kv_heads) * kLongPrefillV2Partitions;
    if (stage1_blocks > std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(rows) * q_heads >
            std::numeric_limits<uint32_t>::max()) {
      return hipErrorInvalidValue;
    }
    const uint16_t *const query_chunk =
        query + static_cast<uint64_t>(row_offset) * row_elements;
    uint16_t *const output_chunk =
        output + static_cast<uint64_t>(row_offset) * row_elements;
    hipLaunchKernelGGL(causal_attention_long_prefill_v2_stage1_kernel,
                       dim3(static_cast<uint32_t>(stage1_blocks)),
                       dim3(kLongPrefillV2Threads), 0U, stream, query_chunk,
                       key_fp16, value_fp16, rows, start_position, row_offset,
                       committed_kv_length, q_heads, kv_heads,
                       static_cast<float *>(workspace));
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
      return status;
    }
    hipLaunchKernelGGL(causal_attention_long_prefill_v2_combine_kernel,
                       dim3(static_cast<uint32_t>(rows * q_heads)),
                       dim3(kLongPrefillV2Threads), 0U, stream,
                       static_cast<const float *>(workspace), output_chunk,
                       rows, q_heads, static_cast<uint64_t>(row_offset));
    status = hipGetLastError();
    if (status != hipSuccess) {
      return status;
    }
  }
  return hipSuccess;
#endif
}

hipError_t launch_scaled_prefill_gemm(
    const uint16_t *const query, const void *const key, const void *const value,
    uint16_t *const output, const uint32_t query_count,
    const uint64_t start_position, const uint64_t committed_kv_length,
    const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim,
    void *const workspace, const uint64_t workspace_bytes,
    void *const blas_handle, void *const blas_mutex,
    const hipStream_t stream) noexcept {
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
  (void)query;
  (void)key;
  (void)value;
  (void)output;
  (void)query_count;
  (void)start_position;
  (void)committed_kv_length;
  (void)q_heads;
  (void)kv_heads;
  (void)head_dim;
  (void)workspace;
  (void)workspace_bytes;
  (void)blas_handle;
  (void)blas_mutex;
  (void)stream;
  return hipErrorNotSupported;
#else
  constexpr uint32_t kChunkRows = 256U;
  constexpr uint32_t kExpectedQHeads = 16U;
  constexpr uint32_t kExpectedKvHeads = 4U;
  constexpr uint32_t kExpectedHeadDim = 256U;
  constexpr uint32_t kQGroup = 4U;
  constexpr uint32_t kPackedRows = kChunkRows * kQGroup;
  constexpr uint64_t kMaxWorkspaceBytes = UINT64_C(1) << 30;
  if (query == nullptr || key == nullptr || value == nullptr ||
      output == nullptr || workspace == nullptr || blas_handle == nullptr ||
      blas_mutex == nullptr || query_count < 1024U ||
      q_heads != kExpectedQHeads || kv_heads != kExpectedKvHeads ||
      head_dim != kExpectedHeadDim || committed_kv_length == 0U ||
      committed_kv_length < query_count ||
      start_position > UINT64_MAX - query_count ||
      start_position + query_count != committed_kv_length ||
      committed_kv_length > static_cast<uint64_t>(INT32_MAX) ||
      query_count > static_cast<uint32_t>(INT32_MAX)) {
    return hipErrorInvalidValue;
  }
  if (committed_kv_length > UINT64_MAX / kPackedRows ||
      committed_kv_length > UINT64_MAX / head_dim) {
    return hipErrorOutOfMemory;
  }
  const uint64_t score_elements =
      static_cast<uint64_t>(kPackedRows) * committed_kv_length;
  const uint64_t kv_elements =
      committed_kv_length * static_cast<uint64_t>(head_dim);
  const uint64_t query_elements = static_cast<uint64_t>(kPackedRows) * head_dim;
  const auto checked_bytes = [](const uint64_t elements, const uint64_t width,
                                uint64_t *const result) noexcept -> bool {
    if (result == nullptr || elements > UINT64_MAX / width) {
      return false;
    }
    *result = elements * width;
    return true;
  };
  uint64_t score_bytes = 0U;
  uint64_t probability_bytes = 0U;
  uint64_t residual_bytes = 0U;
  uint64_t kv_bytes = 0U;
  uint64_t query_bytes = 0U;
  uint64_t scale_bytes = 0U;
  uint64_t output_bytes = 0U;
  uint64_t special_bytes = 0U;
  if (!checked_bytes(score_elements, sizeof(float), &score_bytes) ||
      !checked_bytes(score_elements, sizeof(uint16_t), &probability_bytes) ||
      !checked_bytes(score_elements, sizeof(uint16_t), &residual_bytes) ||
      !checked_bytes(kv_elements, sizeof(uint16_t), &kv_bytes) ||
      !checked_bytes(query_elements, sizeof(uint16_t), &query_bytes) ||
      !checked_bytes(kPackedRows, sizeof(float), &scale_bytes) ||
      !checked_bytes(query_elements, sizeof(float), &output_bytes) ||
      !checked_bytes(static_cast<uint64_t>(kExpectedKvHeads) *
                         kExpectedHeadDim * 3U,
                     sizeof(uint64_t), &special_bytes)) {
    return hipErrorOutOfMemory;
  }
  uint64_t required_bytes = score_bytes;
  const auto append_bytes = [&required_bytes](const uint64_t bytes) noexcept {
    if (required_bytes > UINT64_MAX - bytes) {
      return false;
    }
    required_bytes += bytes;
    return true;
  };
  if (!append_bytes(probability_bytes) || !append_bytes(residual_bytes) ||
      !append_bytes(kv_bytes) || !append_bytes(kv_bytes) ||
      !append_bytes(query_bytes) || !append_bytes(scale_bytes) ||
      !append_bytes(output_bytes) || !append_bytes(output_bytes) ||
      !append_bytes(special_bytes)) {
    return hipErrorOutOfMemory;
  }
  if (required_bytes > static_cast<uint64_t>(SIZE_MAX) ||
      required_bytes > kMaxWorkspaceBytes || workspace_bytes < required_bytes) {
    return hipErrorInvalidValue;
  }
  auto *const scores = static_cast<float *>(workspace);
  auto *const probabilities = reinterpret_cast<uint16_t *>(
      static_cast<uint8_t *>(workspace) + score_bytes);
  auto *const residuals = reinterpret_cast<uint16_t *>(
      static_cast<uint8_t *>(workspace) + score_bytes + probability_bytes);
  auto *const key_pack = reinterpret_cast<uint16_t *>(
      static_cast<uint8_t *>(workspace) + score_bytes + probability_bytes +
      residual_bytes);
  auto *const value_pack = reinterpret_cast<uint16_t *>(
      static_cast<uint8_t *>(workspace) + score_bytes + probability_bytes +
      residual_bytes + kv_bytes);
  auto *const query_pack = reinterpret_cast<uint16_t *>(
      static_cast<uint8_t *>(workspace) + score_bytes + probability_bytes +
      residual_bytes + kv_bytes + kv_bytes);
  auto *const scales = reinterpret_cast<float *>(
      static_cast<uint8_t *>(workspace) + score_bytes + probability_bytes +
      residual_bytes + kv_bytes + kv_bytes + query_bytes);
  auto *const output_f32 = reinterpret_cast<float *>(
      static_cast<uint8_t *>(workspace) + score_bytes + probability_bytes +
      residual_bytes + kv_bytes + kv_bytes + query_bytes + scale_bytes);
  auto *const residual_output_f32 = reinterpret_cast<float *>(
      static_cast<uint8_t *>(workspace) + score_bytes + probability_bytes +
      residual_bytes + kv_bytes + kv_bytes + query_bytes + scale_bytes +
      output_bytes);
  auto *const special_first = reinterpret_cast<uint64_t *>(
      static_cast<uint8_t *>(workspace) + score_bytes + probability_bytes +
      residual_bytes + kv_bytes + kv_bytes + query_bytes + scale_bytes +
      output_bytes + output_bytes);
  hipError_t status = hipMemsetAsync(
      special_first, 0xff, static_cast<std::size_t>(special_bytes), stream);
  if (status != hipSuccess) {
    return status;
  }
  hipblasHandle_t handle = static_cast<hipblasHandle_t>(blas_handle);
  std::mutex *const mutex = static_cast<std::mutex *>(blas_mutex);
  const float alpha = 1.0F;
  const float beta = 0.0F;
  const auto gemm = [&](const hipblasOperation_t op_a,
                        const hipblasOperation_t op_b, const int m, const int n,
                        const int k, const void *const a, const int lda,
                        const void *const b, const int ldb, void *const c,
                        const int ldc, const hipDataType a_type,
                        const hipDataType b_type) noexcept -> hipError_t {
    std::lock_guard<std::mutex> lock(*mutex);
    if (hipblasSetStream(handle, stream) != HIPBLAS_STATUS_SUCCESS) {
      return hipErrorUnknown;
    }
    const hipblasStatus_t status =
        hipblasGemmEx(handle, op_a, op_b, m, n, k, &alpha, a, a_type, lda, b,
                      b_type, ldb, &beta, c, HIPBLAS_R_32F, ldc,
                      HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    return status == HIPBLAS_STATUS_SUCCESS ? hipSuccess : hipErrorUnknown;
  };
  for (uint32_t kv_head = 0U;
       status == hipSuccess && kv_head < kExpectedKvHeads; ++kv_head) {
    const uint64_t packed_kv_elements =
        committed_kv_length * static_cast<uint64_t>(head_dim);
    hipLaunchKernelGGL(
        scaled_prefill_pack_kv_kernel,
        dim3(static_cast<uint32_t>((packed_kv_elements + 255U) / 256U)),
        dim3(256U), 0U, stream, static_cast<const uint16_t *>(key),
        static_cast<const uint16_t *>(value), key_pack, value_pack,
        special_first, committed_kv_length, kv_heads, head_dim, kv_head);
    status = hipGetLastError();
    for (uint32_t row_offset = 0U;
         status == hipSuccess && row_offset < query_count;
         row_offset += kChunkRows) {
      const uint32_t rows = std::min(kChunkRows, query_count - row_offset);
      const uint32_t packed_rows = rows * kQGroup;
      hipLaunchKernelGGL(scaled_prefill_pack_query_kernel, dim3(packed_rows),
                         dim3(256U), 0U, stream, query, query_pack, scales,
                         rows, q_heads, head_dim, kv_head, kQGroup, row_offset);
      status = hipGetLastError();
      if (status != hipSuccess) {
        break;
      }
      status = gemm(
          HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(committed_kv_length),
          static_cast<int>(packed_rows), static_cast<int>(head_dim), key_pack,
          static_cast<int>(head_dim), query_pack, static_cast<int>(head_dim),
          scores, static_cast<int>(committed_kv_length), HIPBLAS_R_16F,
          HIPBLAS_R_16F);
      if (status != hipSuccess) {
        break;
      }
      hipLaunchKernelGGL(scaled_prefill_softmax_fp16_kernel, dim3(packed_rows),
                         dim3(256U), 0U, stream, scores, probabilities,
                         residuals, packed_rows, committed_kv_length,
                         start_position, row_offset, kQGroup, scales,
                         1.0F / std::sqrt(static_cast<float>(head_dim)));
      status = hipGetLastError();
      if (status != hipSuccess) {
        break;
      }
      status = gemm(HIPBLAS_OP_N, HIPBLAS_OP_N, static_cast<int>(head_dim),
                    static_cast<int>(packed_rows),
                    static_cast<int>(committed_kv_length), value_pack,
                    static_cast<int>(head_dim), probabilities,
                    static_cast<int>(committed_kv_length), output_f32,
                    static_cast<int>(head_dim), HIPBLAS_R_16F, HIPBLAS_R_16F);
      if (status != hipSuccess) {
        break;
      }
      status = gemm(HIPBLAS_OP_N, HIPBLAS_OP_N, static_cast<int>(head_dim),
                    static_cast<int>(packed_rows),
                    static_cast<int>(committed_kv_length), value_pack,
                    static_cast<int>(head_dim), residuals,
                    static_cast<int>(committed_kv_length), residual_output_f32,
                    static_cast<int>(head_dim), HIPBLAS_R_16F, HIPBLAS_R_16F);
      if (status != hipSuccess) {
        break;
      }
      const uint64_t scatter_elements =
          static_cast<uint64_t>(packed_rows) * head_dim;
      hipLaunchKernelGGL(
          scaled_prefill_combine_kernel,
          dim3(static_cast<uint32_t>((scatter_elements + 255U) / 256U)),
          dim3(256U), 0U, stream, output_f32, residual_output_f32,
          probabilities, residuals, special_first,
          static_cast<const uint16_t *>(value), rows, kv_heads,
          committed_kv_length, start_position, row_offset, kQGroup, head_dim,
          kv_head);
      status = hipGetLastError();
      if (status != hipSuccess) {
        break;
      }
      hipLaunchKernelGGL(
          scaled_prefill_scatter_kernel,
          dim3(static_cast<uint32_t>((scatter_elements + 255U) / 256U)),
          dim3(256U), 0U, stream, output_f32, output, rows, q_heads, head_dim,
          kv_head, kQGroup, row_offset);
      status = hipGetLastError();
    }
  }
  return status;
#endif
}

hipError_t
launch(const uint16_t *const query, const void *const key,
       const void *const value, const void *const key_scales,
       const void *const value_scales, const float *const key_outer_scales,
       const float *const value_outer_scales, uint16_t *const output,
       const uint32_t query_count, const uint64_t capacity_tokens,
       const uint64_t start_position, const uint64_t committed_kv_length,
       const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim,
       const uint32_t encoding, const float static_key_scale,
       const float static_value_scale, const bool use_gfx1201_wave_provider,
       const bool use_decode_wave_split,
       const bool use_decode_wave_split_q_preload, const bool use_prefill_gqa4,
       const bool use_prefill_gqa4_qtile4, const hipStream_t stream) noexcept {
  if (query == nullptr || key == nullptr || value == nullptr ||
      output == nullptr || query_count == 0U ||
      query_count > SLLM_HIP_CAUSAL_ATTENTION_MAX_M || capacity_tokens == 0U ||
      committed_kv_length == 0U || q_heads == 0U || kv_heads == 0U ||
      q_heads % kv_heads != 0U || head_dim == 0U ||
      head_dim > SLLM_HIP_KV_MAX_HEAD_DIM ||
      ((encoding == SLLM_HIP_KV_ENCODING_FP8_V1 ||
        encoding == SLLM_HIP_KV_ENCODING_NVFP4_V1) &&
       (key_scales == nullptr || value_scales == nullptr)) ||
      (encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1 &&
       (!std::isfinite(static_key_scale) || static_key_scale <= 0.0F ||
        !std::isfinite(static_value_scale) || static_value_scale <= 0.0F)) ||
      (encoding == SLLM_HIP_KV_ENCODING_NVFP4_V1 &&
       (key_outer_scales == nullptr || value_outer_scales == nullptr))) {
    return hipErrorInvalidValue;
  }
  const uint64_t block_count = static_cast<uint64_t>(query_count) * q_heads;
  if (block_count > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidValue;
  }
  if (use_decode_wave_split) {
    if (query_count != 1U || start_position + 1U != committed_kv_length ||
        head_dim != SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM) {
      return hipErrorInvalidValue;
    }
    if (use_decode_wave_split_q_preload) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(causal_attention_decode_wave_split_kernel<true>),
          dim3(q_heads), dim3(SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE), 0U,
          stream, query, key, value, key_scales, value_scales, key_outer_scales,
          value_outer_scales, output, committed_kv_length, q_heads, kv_heads,
          head_dim, encoding, static_key_scale, static_value_scale);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(causal_attention_decode_wave_split_kernel<false>),
          dim3(q_heads), dim3(SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE), 0U,
          stream, query, key, value, key_scales, value_scales, key_outer_scales,
          value_outer_scales, output, committed_kv_length, q_heads, kv_heads,
          head_dim, encoding, static_key_scale, static_value_scale);
    }
  } else if (use_prefill_gqa4_qtile4) {
    if (query_count < 64U || q_heads / kv_heads != 4U ||
        head_dim != SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM) {
      return hipErrorInvalidValue;
    }
    const uint64_t gqa_block_count =
        (static_cast<uint64_t>(query_count) + 3U) / 4U * kv_heads;
    if (gqa_block_count > std::numeric_limits<uint32_t>::max()) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(causal_attention_prefill_gqa4_qtile4_kernel,
                       dim3(static_cast<uint32_t>(gqa_block_count)),
                       dim3(SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE), 0U,
                       stream, query, key, value, key_scales, value_scales,
                       key_outer_scales, value_outer_scales, output,
                       query_count, start_position, q_heads, kv_heads, head_dim,
                       encoding, static_key_scale, static_value_scale);
  } else if (use_prefill_gqa4) {
    if (query_count < 64U || q_heads / kv_heads != 4U ||
        head_dim != SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM) {
      return hipErrorInvalidValue;
    }
    const uint64_t gqa_block_count =
        static_cast<uint64_t>(query_count) * kv_heads;
    if (gqa_block_count > std::numeric_limits<uint32_t>::max()) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(causal_attention_prefill_gqa4_shared_kernel,
                       dim3(static_cast<uint32_t>(gqa_block_count)),
                       dim3(SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE), 0U,
                       stream, query, key, value, key_scales, value_scales,
                       key_outer_scales, value_outer_scales, output,
                       query_count, start_position, q_heads, kv_heads, head_dim,
                       encoding, static_key_scale, static_value_scale);
  } else if (use_gfx1201_wave_provider) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(causal_attention_kernel<true>),
                       dim3(static_cast<uint32_t>(block_count)),
                       dim3(SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE), 0U,
                       stream, query, key, value, key_scales, value_scales,
                       key_outer_scales, value_outer_scales, output,
                       query_count, capacity_tokens, start_position,
                       committed_kv_length, q_heads, kv_heads, head_dim,
                       encoding, static_key_scale, static_value_scale);
  } else {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(causal_attention_kernel<false>),
                       dim3(static_cast<uint32_t>(block_count)),
                       dim3(SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE), 0U,
                       stream, query, key, value, key_scales, value_scales,
                       key_outer_scales, value_outer_scales, output,
                       query_count, capacity_tokens, start_position,
                       committed_kv_length, q_heads, kv_heads, head_dim,
                       encoding, static_key_scale, static_value_scale);
  }
  return hipGetLastError();
}

} // namespace sllm_causal_attention_kernel
