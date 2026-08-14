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
    const uint16_t *const query, const uint16_t *const key,
    const uint16_t *const value, uint16_t *const output,
    const uint32_t query_count, const uint64_t /*capacity_tokens*/,
    const uint64_t start_position, const uint64_t committed_kv_length,
    const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim) {
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

  float accumulation = 0.0F;
  for (uint64_t key_position = 0U; key_position <= query_position;
       ++key_position) {
    const uint64_t key_base = (key_position * kv_heads + kv_head) * head_dim;
    reductions[dimension] = dimension < head_dim
                                ? bf16_to_f32(query_row[dimension]) *
                                      f16_to_f32(key[key_base + dimension])
                                : 0.0F;
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
      accumulation = accumulation * rescale +
                     contribution * f16_to_f32(value[key_base + dimension]);
    }
    __syncthreads();
  }
  if (dimension < head_dim) {
    output_row[dimension] = f32_to_bf16_rne(accumulation / running_denominator);
  }
}

} // namespace

hipError_t launch(const uint16_t *const query, const uint16_t *const key,
                  const uint16_t *const value, uint16_t *const output,
                  const uint32_t query_count, const uint64_t capacity_tokens,
                  const uint64_t start_position,
                  const uint64_t committed_kv_length, const uint32_t q_heads,
                  const uint32_t kv_heads, const uint32_t head_dim,
                  const hipStream_t stream) noexcept {
  if (query == nullptr || key == nullptr || value == nullptr ||
      output == nullptr || query_count == 0U ||
      query_count > SLLM_HIP_CAUSAL_ATTENTION_MAX_M || capacity_tokens == 0U ||
      committed_kv_length == 0U || q_heads == 0U || kv_heads == 0U ||
      q_heads % kv_heads != 0U || head_dim == 0U ||
      head_dim > SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE) {
    return hipErrorInvalidValue;
  }
  const uint64_t block_count = static_cast<uint64_t>(query_count) * q_heads;
  if (block_count > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(
      causal_attention_kernel, dim3(static_cast<uint32_t>(block_count)),
      dim3(SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE), 0U, stream, query, key,
      value, output, query_count, capacity_tokens, start_position,
      committed_kv_length, q_heads, kv_heads, head_dim);
  return hipGetLastError();
}

} // namespace sllm_causal_attention_kernel
