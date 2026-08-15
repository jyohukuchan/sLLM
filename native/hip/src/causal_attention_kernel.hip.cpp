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
    reductions[dimension] = 0.0F;
    for (uint32_t current = dimension; current < head_dim;
         current += blockDim.x) {
      reductions[dimension] +=
          bf16_to_f32(query_row[current]) *
          load_kv(key, key_scales, key_outer_scales, encoding, kv_row,
                  current, head_dim);
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
    output_row[second] =
        f32_to_bf16_rne(accumulation1 / running_denominator);
  }
}

} // namespace

hipError_t launch(const uint16_t *const query, const void *const key,
                  const void *const value, const void *const key_scales,
                  const void *const value_scales,
                  const float *const key_outer_scales,
                  const float *const value_outer_scales, uint16_t *const output,
                  const uint32_t query_count, const uint64_t capacity_tokens,
                  const uint64_t start_position,
                  const uint64_t committed_kv_length, const uint32_t q_heads,
                  const uint32_t kv_heads, const uint32_t head_dim,
                  const uint32_t encoding, const hipStream_t stream) noexcept {
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
  hipLaunchKernelGGL(
      causal_attention_kernel, dim3(static_cast<uint32_t>(block_count)),
      dim3(SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE), 0U, stream, query, key,
      value, key_scales, value_scales, key_outer_scales, value_outer_scales,
      output, query_count, capacity_tokens, start_position, committed_kv_length,
      q_heads, kv_heads, head_dim, encoding);
  return hipGetLastError();
}

} // namespace sllm_causal_attention_kernel
