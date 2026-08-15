#include "gemma_attention_kernel_internal.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace sllm_gemma_attention_kernel {
namespace {

__device__ __forceinline__ float bf16_to_f32(const uint16_t raw) noexcept {
  return __uint_as_float(static_cast<uint32_t>(raw) << 16U);
}

__device__ __forceinline__ uint16_t
f32_to_bf16_rne(const float value) noexcept {
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
extern "C" __global__
__launch_bounds__(kWorkgroupSize, 1) void sllm_gemma_causal_attention_online_softmax_gqa_bf16_v1(
    const uint16_t *const query, const uint16_t *const key,
    const uint16_t *const value, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const uint64_t committed_kv_length, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim,
    const uint64_t sliding_window) {
  const uint64_t block = static_cast<uint64_t>(blockIdx.x);
  if (block >= static_cast<uint64_t>(query_count) * q_heads) {
    return;
  }
  const uint64_t row = block / q_heads;
  const uint32_t query_head = static_cast<uint32_t>(block % q_heads);
  const uint64_t query_position = start_position + row;
  if (query_position >= committed_kv_length) {
    return;
  }
  const uint32_t kv_head = query_head / (q_heads / kv_heads);
  const uint64_t query_base =
      (row * q_heads + query_head) * static_cast<uint64_t>(head_dim);
  const uint64_t output_base = query_base;
  uint64_t first_key = 0U;
  if (sliding_window != 0U && query_position + 1U > sliding_window) {
    first_key = query_position + 1U - sliding_window;
  }

  __shared__ float reductions[kWorkgroupSize];
  __shared__ float rescale;
  __shared__ float contribution;
  __shared__ float running_maximum;
  __shared__ float running_denominator;
  if (threadIdx.x == 0U) {
    running_maximum = -std::numeric_limits<float>::infinity();
    running_denominator = 0.0F;
  }
  __syncthreads();

  float accumulation0 = 0.0F;
  float accumulation1 = 0.0F;
  for (uint64_t key_position = first_key; key_position <= query_position;
       ++key_position) {
    const uint64_t key_base =
        (key_position * kv_heads + kv_head) * static_cast<uint64_t>(head_dim);
    float partial = 0.0F;
    for (uint32_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
      partial += bf16_to_f32(query[query_base + dimension]) *
                 bf16_to_f32(key[key_base + dimension]);
    }
    reductions[threadIdx.x] = partial;
    __syncthreads();
    for (uint32_t stride = kWorkgroupSize / 2U; stride != 0U; stride >>= 1U) {
      if (threadIdx.x < stride) {
        reductions[threadIdx.x] += reductions[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0U) {
      const float score = reductions[0];
      const float next_maximum = fmaxf(running_maximum, score);
      rescale = expf(running_maximum - next_maximum);
      contribution = expf(score - next_maximum);
      running_denominator = running_denominator * rescale + contribution;
      running_maximum = next_maximum;
    }
    __syncthreads();
    const uint32_t dimension0 = threadIdx.x;
    if (dimension0 < head_dim) {
      accumulation0 = accumulation0 * rescale +
                      contribution * bf16_to_f32(value[key_base + dimension0]);
    }
    const uint32_t dimension1 = threadIdx.x + kWorkgroupSize;
    if (dimension1 < head_dim) {
      accumulation1 = accumulation1 * rescale +
                      contribution * bf16_to_f32(value[key_base + dimension1]);
    }
    __syncthreads();
  }
  if (threadIdx.x < head_dim) {
    output[output_base + threadIdx.x] =
        f32_to_bf16_rne(accumulation0 / running_denominator);
  }
  const uint32_t dimension1 = threadIdx.x + kWorkgroupSize;
  if (dimension1 < head_dim) {
    output[output_base + dimension1] =
        f32_to_bf16_rne(accumulation1 / running_denominator);
  }
}

} // namespace

hipError_t launch(const uint16_t *const query, const uint16_t *const key,
                  const uint16_t *const value, uint16_t *const output,
                  const uint32_t query_count, const uint64_t start_position,
                  const uint64_t committed_kv_length, const uint32_t q_heads,
                  const uint32_t kv_heads, const uint32_t head_dim,
                  const uint64_t sliding_window,
                  const hipStream_t stream) noexcept {
  if (query == nullptr || key == nullptr || value == nullptr ||
      output == nullptr || query_count == 0U || q_heads == 0U ||
      kv_heads == 0U || q_heads % kv_heads != 0U || head_dim == 0U ||
      head_dim > 512U ||
      start_position > std::numeric_limits<uint64_t>::max() - query_count ||
      start_position + query_count != committed_kv_length ||
      committed_kv_length == 0U) {
    return hipErrorInvalidValue;
  }
  const uint64_t blocks = static_cast<uint64_t>(query_count) * q_heads;
  if (blocks > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_gemma_causal_attention_online_softmax_gqa_bf16_v1,
                     dim3(static_cast<uint32_t>(blocks)), dim3(kWorkgroupSize),
                     0U, stream, query, key, value, output, query_count,
                     start_position, committed_kv_length, q_heads, kv_heads,
                     head_dim, sliding_window);
  return hipGetLastError();
}

} // namespace sllm_gemma_attention_kernel
