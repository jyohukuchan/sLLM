#include "ministral3_yarn_kernel_internal.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace sllm_ministral3_yarn_kernel {
namespace {

constexpr uint32_t kHeadDim = 128U;
constexpr uint32_t kHalfDim = 64U;
constexpr uint32_t kLowCorrection = 20U;
constexpr uint32_t kHighCorrection = 37U;
constexpr float kTheta = 1000000.0F;
constexpr float kFactor = 16.0F;
constexpr uint32_t kOriginalContext = 16384U;

__device__ __forceinline__ float bf16_to_f32(const uint16_t raw) noexcept {
  const uint32_t bits = static_cast<uint32_t>(raw) << 16U;
  float value = 0.0F;
  __builtin_memcpy(&value, &bits, sizeof(value));
  return value;
}

__device__ __forceinline__ uint16_t
f32_to_bf16_rne(const float value) noexcept {
  uint32_t bits = 0U;
  __builtin_memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

__device__ __forceinline__ float
inverse_frequency(const uint32_t pair) noexcept {
  const float exponent = static_cast<float>(pair * 2U) / 128.0F;
  const float base = powf(kTheta, exponent);
  const float extrapolated = 1.0F / base;
  const float interpolated = 1.0F / (kFactor * base);
  const float ramp = fminf(
      1.0F,
      fmaxf(0.0F, (static_cast<float>(pair) - kLowCorrection) /
                      static_cast<float>(kHighCorrection - kLowCorrection)));
  return interpolated * ramp + extrapolated * (1.0F - ramp);
}

__device__ __forceinline__ float query_scale(const int32_t position) noexcept {
  const uint32_t block = static_cast<uint32_t>(position) / kOriginalContext;
  return 1.0F + 0.1F * log1pf(static_cast<float>(block));
}

__global__ __launch_bounds__(kWorkgroupSize, 1) void ministral3_yarn_kernel(
    const uint16_t *const query, const uint16_t *const key,
    const int32_t *const positions, uint16_t *const query_output,
    uint16_t *const key_output, const uint32_t token_count,
    const uint32_t q_heads, const uint32_t kv_heads,
    const bool adjacent_pairing) {
  const uint64_t q_blocks = static_cast<uint64_t>(token_count) * q_heads;
  const uint64_t block = static_cast<uint64_t>(blockIdx.x);
  const bool is_query = block < q_blocks;
  const uint64_t local = is_query ? block : block - q_blocks;
  const uint32_t heads = is_query ? q_heads : kv_heads;
  const uint64_t token = local / heads;
  const uint32_t head = static_cast<uint32_t>(local % heads);
  if (token >= token_count) {
    return;
  }
  const uint64_t base = (token * heads + head) * kHeadDim;
  const uint16_t *const input = is_query ? query : key;
  uint16_t *const output = is_query ? query_output : key_output;
  const int32_t position = positions[token];
  const float scale = is_query ? query_scale(position) : 1.0F;
  for (uint32_t dimension = threadIdx.x; dimension < kHeadDim;
       dimension += blockDim.x) {
    const uint32_t pair =
        adjacent_pairing
            ? dimension / 2U
            : (dimension < kHalfDim ? dimension : dimension - kHalfDim);
    const float angle = static_cast<float>(position) * inverse_frequency(pair);
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const uint32_t left_index = adjacent_pairing ? pair * 2U : pair;
    const uint32_t right_index =
        adjacent_pairing ? left_index + 1U : kHalfDim + pair;
    const float left = bf16_to_f32(input[base + left_index]);
    const float right = bf16_to_f32(input[base + right_index]);
    const bool first =
        adjacent_pairing ? (dimension & 1U) == 0U : dimension < kHalfDim;
    const float rotated = first ? (left * cosine - right * sine) * scale
                                : (right * cosine + left * sine) * scale;
    output[base + dimension] = f32_to_bf16_rne(rotated);
  }
}

} // namespace

hipError_t launch(const uint16_t *const query, const uint16_t *const key,
                  const int32_t *const positions, uint16_t *const query_output,
                  uint16_t *const key_output, const uint32_t token_count,
                  const uint32_t q_heads, const uint32_t kv_heads,
                  const bool adjacent_pairing,
                  const hipStream_t stream) noexcept {
  if (query == nullptr || key == nullptr || positions == nullptr ||
      query_output == nullptr || key_output == nullptr || token_count == 0U ||
      q_heads != 32U || kv_heads != 8U) {
    return hipErrorInvalidValue;
  }
  const uint64_t blocks = static_cast<uint64_t>(token_count) *
                          static_cast<uint64_t>(q_heads + kv_heads);
  if (blocks > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(
      ministral3_yarn_kernel, dim3(static_cast<uint32_t>(blocks)),
      dim3(kWorkgroupSize), 0U, stream, query, key, positions, query_output,
      key_output, token_count, q_heads, kv_heads, adjacent_pairing);
  return hipGetLastError();
}

} // namespace sllm_ministral3_yarn_kernel
