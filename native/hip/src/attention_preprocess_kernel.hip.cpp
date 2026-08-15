#include "attention_preprocess_kernel_internal.hpp"

#include "rmsnorm_kernel_internal.hpp"

#include <cmath>
#include <cstdint>

namespace {

__device__ __forceinline__ float bf16_to_float(const uint16_t value) noexcept {
  return __uint_as_float(static_cast<uint32_t>(value) << 16U);
}

__device__ __forceinline__ float norm_value(const uint16_t value,
                                            const uint16_t raw_scale,
                                            const float inverse_rms) noexcept {
  const float input = bf16_to_float(value);
  const float normalized = input * inverse_rms;
  const float effective_scale = 1.0F + bf16_to_float(raw_scale);
  return normalized * effective_scale;
}

__device__ __forceinline__ void
rotate_neox_64(float *const values, const int32_t position) noexcept {
  constexpr float theta = 10000000.0F;
  for (uint32_t pair = 0U; pair != 32U; ++pair) {
    /* NeoX pairs dim pair with dim pair+32. The angle is computed from the
     * exact f32 theta and the f32 exponent -2*pair/64, then applied to the
     * already-normalized values in ascending pair order. */
    const float exponent =
        -static_cast<float>(2U * pair) / static_cast<float>(64U);
    const float angle = static_cast<float>(position) * powf(theta, exponent);
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const float first = values[pair];
    const float second = values[pair + 32U];
    values[pair] = first * cosine - second * sine;
    values[pair + 32U] = first * sine + second * cosine;
  }
}

__device__ __forceinline__ void
rotate_mrope_64(float *const values, const int32_t *const positions) noexcept {
  constexpr float theta = 10000000.0F;
  for (uint32_t pair = 0U; pair != 32U; ++pair) {
    uint32_t axis = 0U;
    if (pair % 3U == 1U && pair < 33U) {
      axis = 1U;
    } else if (pair % 3U == 2U && pair < 30U) {
      axis = 2U;
    }
    const float exponent =
        -static_cast<float>(2U * pair) / static_cast<float>(64U);
    const float angle = static_cast<float>(positions[axis]) *
                        powf(theta, exponent);
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const float first = values[pair];
    const float second = values[pair + 32U];
    values[pair] = first * cosine - second * sine;
    values[pair + 32U] = first * sine + second * cosine;
  }
}

__device__ __forceinline__ void process_head(const uint16_t *const input,
                                             const uint16_t *const raw_scale,
                                             uint16_t *const output,
                                             const int32_t *const positions,
                                             const uint32_t position_components) noexcept {
  float values[256];
  float sum = 0.0F;
  /* One logical thread owns one head. This deliberately keeps the RMSNorm
   * sum and every subsequent operation in ascending dimension order. */
  for (uint32_t dim = 0U; dim != 256U; ++dim) {
    const float value = bf16_to_float(input[dim]);
    sum += value * value;
  }
  const float mean = sum / 256.0F;
  const float inverse_rms = 1.0F / sqrtf(mean + 1.0e-6F);
  for (uint32_t dim = 0U; dim != 256U; ++dim) {
    /* Preserve the semantic stage boundary: RMSNorm's FP32 result is
     * rounded to BF16 RNE before the subsequent RoPE stage reads it back. */
    const uint16_t normalized = sllm_rmsnorm_kernel::float_to_bf16_rne_bits(
        norm_value(input[dim], raw_scale[dim], inverse_rms));
    values[dim] = bf16_to_float(normalized);
  }
  if (position_components == 1U) {
    rotate_neox_64(values, positions[0]);
  } else {
    rotate_mrope_64(values, positions);
  }
  for (uint32_t dim = 0U; dim != 256U; ++dim) {
    output[dim] = sllm_rmsnorm_kernel::float_to_bf16_rne_bits(values[dim]);
  }
}

} // namespace

extern "C" __global__
__launch_bounds__(1, 1) void sllm_attention_preprocess_headwise_norm_rope_v1(
    const uint16_t *const packed_q_gate, const uint16_t *const k,
    const uint16_t *const q_raw_scale, const uint16_t *const k_raw_scale,
    const int32_t *const positions, uint16_t *const q_output,
    uint16_t *const gate_output, uint16_t *const k_output, const uint32_t m,
    const uint32_t q_heads, const uint32_t k_heads, const uint32_t head_dim,
    const uint32_t position_components) {
  if (threadIdx.x != 0U) {
    return;
  }
  const uint64_t q_block_count = static_cast<uint64_t>(m) * q_heads;
  const uint64_t block = static_cast<uint64_t>(blockIdx.x);
  uint64_t row = 0U;
  uint32_t head = 0U;
  bool is_q = false;
  if (block < q_block_count) {
    row = block / q_heads;
    head = static_cast<uint32_t>(block % q_heads);
    is_q = true;
  } else {
    const uint64_t k_block = block - q_block_count;
    row = k_block / k_heads;
    head = static_cast<uint32_t>(k_block % k_heads);
  }
  const int32_t *const position = positions + row * position_components;
  if (is_q) {
    const uint64_t input_offset = row * q_heads * head_dim * 2U +
                                  static_cast<uint64_t>(head) * head_dim * 2U;
    const uint64_t output_offset =
        row * q_heads * head_dim + static_cast<uint64_t>(head) * head_dim;
    const uint64_t scale_offset = static_cast<uint64_t>(head) * head_dim;
    process_head(packed_q_gate + input_offset, q_raw_scale + scale_offset,
                 q_output + output_offset, position, position_components);
    for (uint32_t dim = 0U; dim != 256U; ++dim) {
      gate_output[output_offset + dim] =
          packed_q_gate[input_offset + head_dim + dim];
    }
  } else {
    const uint64_t input_offset =
        row * k_heads * head_dim + static_cast<uint64_t>(head) * head_dim;
    const uint64_t output_offset =
        row * k_heads * head_dim + static_cast<uint64_t>(head) * head_dim;
    const uint64_t scale_offset = static_cast<uint64_t>(head) * head_dim;
    process_head(k + input_offset, k_raw_scale + scale_offset,
                 k_output + output_offset, position, position_components);
  }
}

namespace sllm_attention_preprocess_kernel {

hipError_t launch(const uint16_t *const packed_q_gate, const uint16_t *const k,
                  const uint16_t *const q_raw_scale,
                  const uint16_t *const k_raw_scale,
                  const int32_t *const positions, uint16_t *const q_output,
                  uint16_t *const gate_output, uint16_t *const k_output,
                  const uint32_t m, const uint32_t q_heads,
                  const uint32_t k_heads, const uint32_t head_dim,
                  const uint32_t position_components,
                  const hipStream_t stream) noexcept {
  const uint32_t block_count = m * (q_heads + k_heads);
  const dim3 grid(block_count, 1U, 1U);
  const dim3 block(1U, 1U, 1U);
  hipLaunchKernelGGL(sllm_attention_preprocess_headwise_norm_rope_v1, grid,
                     block, 0U, stream, packed_q_gate, k, q_raw_scale,
                     k_raw_scale, positions, q_output, gate_output, k_output, m,
                     q_heads, k_heads, head_dim, position_components);
  return hipGetLastError();
}

} // namespace sllm_attention_preprocess_kernel
