#include "rotary_kernel_internal.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace sllm_rotary_kernel {
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
__launch_bounds__(kWorkgroupSize, 1) void sllm_rotary_split_half_bf16_fp32_v1(
    const uint16_t *const query, const uint16_t *const key,
    const int32_t *const positions, uint16_t *const query_output,
    uint16_t *const key_output, const uint32_t token_count,
    const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim,
    const uint32_t rotary_dim, const float theta) {
  const uint64_t q_head_count = static_cast<uint64_t>(token_count) * q_heads;
  const uint64_t block = static_cast<uint64_t>(blockIdx.x);
  const bool is_query = block < q_head_count;
  const uint64_t head_index = is_query ? block : block - q_head_count;
  const uint32_t heads = is_query ? q_heads : kv_heads;
  const uint64_t token = head_index / heads;
  const uint64_t head = head_index % heads;
  if (token >= token_count) {
    return;
  }
  const uint16_t *const input = is_query ? query : key;
  uint16_t *const output = is_query ? query_output : key_output;
  const uint64_t base = (token * heads + head) * head_dim;
  const uint32_t half = head_dim / 2U;
  const uint32_t active_pairs = rotary_dim / 2U;
  const float position = static_cast<float>(positions[token]);
  for (uint32_t dimension = threadIdx.x; dimension < head_dim;
       dimension += blockDim.x) {
    uint16_t result = input[base + dimension];
    uint32_t pair = 0U;
    bool first = false;
    bool active = false;
    if (dimension < active_pairs) {
      pair = dimension;
      first = true;
      active = true;
    } else if (dimension >= half && dimension < half + active_pairs) {
      pair = dimension - half;
      active = true;
    }
    if (active) {
      const float exponent =
          -2.0F * static_cast<float>(pair) / static_cast<float>(head_dim);
      const float angle = position * powf(theta, exponent);
      const float cosine = cosf(angle);
      const float sine = sinf(angle);
      const float left = bf16_to_f32(input[base + pair]);
      const float right = bf16_to_f32(input[base + half + pair]);
      const float rotated =
          first ? left * cosine - right * sine : right * cosine + left * sine;
      result = f32_to_bf16_rne(rotated);
    }
    output[base + dimension] = result;
  }
}

} // namespace

hipError_t launch(const uint16_t *const query, const uint16_t *const key,
                  const int32_t *const positions, uint16_t *const query_output,
                  uint16_t *const key_output, const uint32_t token_count,
                  const uint32_t q_heads, const uint32_t kv_heads,
                  const uint32_t head_dim, const uint32_t rotary_dim,
                  const float theta, const hipStream_t stream) noexcept {
  if (query == nullptr || key == nullptr || positions == nullptr ||
      query_output == nullptr || key_output == nullptr || token_count == 0U ||
      q_heads == 0U || kv_heads == 0U || q_heads % kv_heads != 0U ||
      head_dim == 0U || (head_dim & 1U) != 0U || rotary_dim == 0U ||
      (rotary_dim & 1U) != 0U || rotary_dim > head_dim || theta != theta ||
      theta > std::numeric_limits<float>::max() || theta <= 0.0F) {
    return hipErrorInvalidValue;
  }
  const uint64_t blocks =
      static_cast<uint64_t>(token_count) * (q_heads + kv_heads);
  if (blocks > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(
      sllm_rotary_split_half_bf16_fp32_v1, dim3(static_cast<uint32_t>(blocks)),
      dim3(kWorkgroupSize), 0U, stream, query, key, positions, query_output,
      key_output, token_count, q_heads, kv_heads, head_dim, rotary_dim, theta);
  return hipGetLastError();
}

} // namespace sllm_rotary_kernel
