#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

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
                         const uint32_t dimension,
                         const uint32_t head_dim) noexcept {
  if (encoding == SLLM_HIP_KV_ENCODING_FP16_V1) {
    return f16_to_f32(
        static_cast<const uint16_t *>(values)[row * head_dim + dimension]);
  }
  if (encoding == SLLM_HIP_KV_ENCODING_FP8_V1 ||
      encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1) {
    return e4m3fn_to_f32(static_cast<const uint8_t *>(
               values)[row * head_dim + dimension]) *
           static_cast<const float *>(scales)[row];
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
    const uint32_t kv_heads, const uint32_t head_dim, const uint32_t encoding) {
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
                           current, head_dim);
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
        reductions[dimension] += bf16_to_f32(query_row[current]) *
                                 load_kv(key, key_scales, key_outer_scales,
                                         encoding, kv_row, current, head_dim);
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
                                 encoding, kv_row, dimension, head_dim);
    }
    const uint32_t second = dimension + blockDim.x;
    if (second < head_dim) {
      accumulation1 =
          accumulation1 * rescale +
          contribution * load_kv(value, value_scales, value_outer_scales,
                                 encoding, kv_row, second, head_dim);
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
__global__
__launch_bounds__(256, 1) void causal_attention_decode_wave_split_kernel(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const float *const key_outer_scales, const float *const value_outer_scales,
    uint16_t *const output, const uint64_t committed_kv_length,
    const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim,
    const uint32_t encoding) {
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
#pragma unroll
  for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
    accumulations[index] = 0.0F;
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
        partial += bf16_to_f32(query_row[current]) *
                   load_kv(key, key_scales, key_outer_scales, encoding, kv_row,
                           current, head_dim);
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
                                   encoding, kv_row, current, head_dim);
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

// Phase 33's one-row provider remains available as a measured control for the
// Phase 35 provider-selection audit.
__global__
__launch_bounds__(256, 1) void causal_attention_prefill_gqa4_shared_kernel(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const float *const key_outer_scales, const float *const value_outer_scales,
    uint16_t *const output, const uint32_t query_count,
    const uint64_t start_position, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim, const uint32_t encoding) {
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
    const float key_value = dimension < head_dim
                                ? load_kv(key, key_scales, key_outer_scales,
                                          encoding, kv_row, dimension, head_dim)
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
        dimension < head_dim ? load_kv(value, value_scales, value_outer_scales,
                                       encoding, kv_row, dimension, head_dim)
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
    const uint32_t kv_heads, const uint32_t head_dim, const uint32_t encoding) {
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
      key_tile[dimension] = load_kv(key, key_scales, key_outer_scales, encoding,
                                    kv_row, dimension, head_dim);
      value_tile[dimension] = load_kv(value, value_scales, value_outer_scales,
                                      encoding, kv_row, dimension, head_dim);
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

} // namespace

hipError_t
launch(const uint16_t *const query, const void *const key,
       const void *const value, const void *const key_scales,
       const void *const value_scales, const float *const key_outer_scales,
       const float *const value_outer_scales, uint16_t *const output,
       const uint32_t query_count, const uint64_t capacity_tokens,
       const uint64_t start_position, const uint64_t committed_kv_length,
       const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim,
       const uint32_t encoding, const bool use_gfx1201_wave_provider,
       const bool use_decode_wave_split, const bool use_prefill_gqa4,
       const bool use_prefill_gqa4_qtile4, const hipStream_t stream) noexcept {
  if (query == nullptr || key == nullptr || value == nullptr ||
      output == nullptr || query_count == 0U ||
      query_count > SLLM_HIP_CAUSAL_ATTENTION_MAX_M || capacity_tokens == 0U ||
      committed_kv_length == 0U || q_heads == 0U || kv_heads == 0U ||
      q_heads % kv_heads != 0U || head_dim == 0U ||
      head_dim > SLLM_HIP_KV_MAX_HEAD_DIM ||
      (encoding != SLLM_HIP_KV_ENCODING_FP16_V1 &&
       (key_scales == nullptr || value_scales == nullptr)) ||
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
    hipLaunchKernelGGL(
        causal_attention_decode_wave_split_kernel, dim3(q_heads),
        dim3(SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE), 0U, stream, query, key,
        value, key_scales, value_scales, key_outer_scales, value_outer_scales,
        output, committed_kv_length, q_heads, kv_heads, head_dim, encoding);
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
                       encoding);
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
                       encoding);
  } else if (use_gfx1201_wave_provider) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(causal_attention_kernel<true>),
        dim3(static_cast<uint32_t>(block_count)),
        dim3(SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE), 0U, stream, query, key,
        value, key_scales, value_scales, key_outer_scales, value_outer_scales,
        output, query_count, capacity_tokens, start_position,
        committed_kv_length, q_heads, kv_heads, head_dim, encoding);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(causal_attention_kernel<false>),
        dim3(static_cast<uint32_t>(block_count)),
        dim3(SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE), 0U, stream, query, key,
        value, key_scales, value_scales, key_outer_scales, value_outer_scales,
        output, query_count, capacity_tokens, start_position,
        committed_kv_length, q_heads, kv_heads, head_dim, encoding);
  }
  return hipGetLastError();
}

} // namespace sllm_causal_attention_kernel
