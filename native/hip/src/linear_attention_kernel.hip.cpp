#include "linear_attention_kernel_internal.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

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

__device__ __forceinline__ float softplus(const float value) noexcept {
  return fmaxf(value, 0.0F) + log1pf(expf(-fabsf(value)));
}

} // namespace

extern "C" __global__
__launch_bounds__(128, 1) void sllm_linear_attention_causal_conv_silu_v1(
    const uint16_t *const qkv, const uint16_t *const conv_weight,
    const uint16_t *const previous_conv_state, uint16_t *const convolved_qkv,
    uint16_t *const next_conv_state, const uint32_t token_count) {
  constexpr uint64_t width = sllm_linear_attention_kernel::kQkvWidth;
  constexpr uint64_t history = sllm_linear_attention_kernel::kConvHistory;
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  const uint64_t output_elements = static_cast<uint64_t>(token_count) * width;
  if (index < output_elements) {
    const uint64_t token = index / width;
    const uint64_t channel = index % width;
    float sum = 0.0F;
    for (uint64_t tap = 0U; tap != 4U; ++tap) {
      const int64_t source = static_cast<int64_t>(token + tap) - 3;
      const uint16_t value =
          source < 0
              ? previous_conv_state[static_cast<uint64_t>(source + 3) * width +
                                    channel]
              : qkv[static_cast<uint64_t>(source) * width + channel];
      sum +=
          bf16_to_float(value) * bf16_to_float(conv_weight[channel * 4U + tap]);
    }
    const float silu = sum / (1.0F + expf(-sum));
    convolved_qkv[index] = float_to_bf16_rne_bits(silu);
    return;
  }

  const uint64_t history_index = index - output_elements;
  if (history_index < history * width) {
    const uint64_t history_row = history_index / width;
    const uint64_t channel = history_index % width;
    const int64_t source = static_cast<int64_t>(token_count) - 3 +
                           static_cast<int64_t>(history_row);
    next_conv_state[history_index] =
        source < 0
            ? previous_conv_state[static_cast<uint64_t>(source + 3) * width +
                                  channel]
            : qkv[static_cast<uint64_t>(source) * width + channel];
  }
}

#pragma clang fp contract(off)
extern "C" __global__
__launch_bounds__(128, 1) void sllm_linear_attention_recurrent_gated_norm_v1(
    const uint16_t *const convolved_qkv, const uint16_t *const z,
    const uint16_t *const b_input, const uint16_t *const a_input,
    const float *const a_log, const uint16_t *const dt_bias,
    const float *const norm_weight, const float *const previous_recurrent_state,
    float *const next_recurrent_state, uint16_t *const output,
    const uint32_t token_count) {
  constexpr uint32_t head_dim = sllm_linear_attention_kernel::kHeadDim;
  constexpr uint32_t qkv_width = sllm_linear_attention_kernel::kQkvWidth;
  constexpr uint32_t output_width = sllm_linear_attention_kernel::kOutputWidth;
  const uint32_t value_head = blockIdx.x;
  const uint32_t dimension = threadIdx.x;
  if (value_head >= sllm_linear_attention_kernel::kValueHeads ||
      dimension >= head_dim) {
    return;
  }
  const uint32_t qk_head = value_head / 2U;
  const uint64_t state_row =
      (static_cast<uint64_t>(value_head) * head_dim + dimension) * head_dim;
  for (uint32_t key_dimension = 0U; key_dimension != head_dim;
       ++key_dimension) {
    next_recurrent_state[state_row + key_dimension] =
        previous_recurrent_state[state_row + key_dimension];
  }

  __shared__ float q_values[head_dim];
  __shared__ float k_values[head_dim];
  __shared__ float q_inverse_norm;
  __shared__ float k_inverse_norm;
  __shared__ float beta;
  __shared__ float decay;
  __shared__ float output_values[head_dim];
  __shared__ float output_inverse_rms;

  for (uint32_t token = 0U; token != token_count; ++token) {
    const uint64_t qkv_row = static_cast<uint64_t>(token) * qkv_width;
    q_values[dimension] = bf16_to_float(
        convolved_qkv[qkv_row + static_cast<uint64_t>(qk_head) * head_dim +
                      dimension]);
    k_values[dimension] = bf16_to_float(
        convolved_qkv[qkv_row + 2048U +
                      static_cast<uint64_t>(qk_head) * head_dim + dimension]);
    __syncthreads();
    if (dimension == 0U) {
      float q_sum = 0.0F;
      float k_sum = 0.0F;
      for (uint32_t index = 0U; index != head_dim; ++index) {
        q_sum += q_values[index] * q_values[index];
        k_sum += k_values[index] * k_values[index];
      }
      q_inverse_norm = 1.0F / sqrtf(q_sum + 1.0e-6F);
      k_inverse_norm = 1.0F / sqrtf(k_sum + 1.0e-6F);
      const uint64_t scalar_index =
          static_cast<uint64_t>(token) * 32U + value_head;
      const float b_value = bf16_to_float(b_input[scalar_index]);
      const float beta_f32 = 1.0F / (1.0F + expf(-b_value));
      beta = bf16_to_float(float_to_bf16_rne_bits(beta_f32));
      const float a_value = bf16_to_float(a_input[scalar_index]) +
                            bf16_to_float(dt_bias[value_head]);
      const float g = -expf(a_log[value_head]) * softplus(a_value);
      decay = expf(g);
    }
    __syncthreads();
    q_values[dimension] = bf16_to_float(
        float_to_bf16_rne_bits(q_values[dimension] * q_inverse_norm));
    q_values[dimension] *= 1.0F / sqrtf(128.0F);
    k_values[dimension] = bf16_to_float(
        float_to_bf16_rne_bits(k_values[dimension] * k_inverse_norm));
    __syncthreads();

    for (uint32_t key_dimension = 0U; key_dimension != head_dim;
         ++key_dimension) {
      next_recurrent_state[state_row + key_dimension] *= decay;
    }
    float previous_projection = 0.0F;
    for (uint32_t key_dimension = 0U; key_dimension != head_dim;
         ++key_dimension) {
      previous_projection += next_recurrent_state[state_row + key_dimension] *
                             k_values[key_dimension];
    }
    const float value = bf16_to_float(
        convolved_qkv[qkv_row + 4096U +
                      static_cast<uint64_t>(value_head) * head_dim +
                      dimension]);
    const float residual = value - previous_projection;
    float current_projection = 0.0F;
    for (uint32_t key_dimension = 0U; key_dimension != head_dim;
         ++key_dimension) {
      const uint64_t index = state_row + key_dimension;
      const float updated = next_recurrent_state[index] +
                            beta * residual * k_values[key_dimension];
      next_recurrent_state[index] = updated;
      current_projection += updated * q_values[key_dimension];
    }
    output_values[dimension] =
        bf16_to_float(float_to_bf16_rne_bits(current_projection));
    __syncthreads();
    if (dimension == 0U) {
      float sum = 0.0F;
      for (uint32_t index = 0U; index != head_dim; ++index) {
        sum += output_values[index] * output_values[index];
      }
      output_inverse_rms = 1.0F / sqrtf(sum / 128.0F + 1.0e-6F);
    }
    __syncthreads();
    const uint64_t output_index = static_cast<uint64_t>(token) * output_width +
                                  static_cast<uint64_t>(value_head) * head_dim +
                                  dimension;
    const float z_value = bf16_to_float(z[output_index]);
    const float z_silu = z_value / (1.0F + expf(-z_value));
    const float normalized = output_values[dimension] * output_inverse_rms;
    const float normalized_bf16 =
        bf16_to_float(float_to_bf16_rne_bits(normalized));
    output[output_index] = float_to_bf16_rne_bits(
        normalized_bf16 * norm_weight[dimension] * z_silu);
    __syncthreads();
  }
}

namespace sllm_linear_attention_kernel {

hipError_t launch_convolution(const uint16_t *const qkv,
                              const uint16_t *const conv_weight,
                              const uint16_t *const previous_conv_state,
                              uint16_t *const convolved_qkv,
                              uint16_t *const next_conv_state,
                              const uint32_t token_count,
                              const hipStream_t stream) noexcept {
  if (qkv == nullptr || conv_weight == nullptr ||
      previous_conv_state == nullptr || convolved_qkv == nullptr ||
      next_conv_state == nullptr || token_count == 0U) {
    return hipErrorInvalidValue;
  }
  const uint64_t elements =
      static_cast<uint64_t>(token_count) * kQkvWidth + kConvHistory * kQkvWidth;
  const uint64_t workgroup = static_cast<uint64_t>(kWorkgroupSize);
  const uint64_t blocks =
      elements / workgroup + static_cast<uint64_t>(elements % workgroup != 0U);
  if (blocks > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_linear_attention_causal_conv_silu_v1,
                     dim3(static_cast<uint32_t>(blocks)), dim3(kWorkgroupSize),
                     0U, stream, qkv, conv_weight, previous_conv_state,
                     convolved_qkv, next_conv_state, token_count);
  return hipGetLastError();
}

hipError_t launch_recurrent(
    const uint16_t *const convolved_qkv, const uint16_t *const z,
    const uint16_t *const b_input, const uint16_t *const a_input,
    const float *const a_log, const uint16_t *const dt_bias,
    const float *const norm_weight, const float *const previous_recurrent_state,
    float *const next_recurrent_state, uint16_t *const output,
    const uint32_t token_count, const hipStream_t stream) noexcept {
  if (convolved_qkv == nullptr || z == nullptr || b_input == nullptr ||
      a_input == nullptr || a_log == nullptr || dt_bias == nullptr ||
      norm_weight == nullptr || previous_recurrent_state == nullptr ||
      next_recurrent_state == nullptr || output == nullptr ||
      token_count == 0U) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_linear_attention_recurrent_gated_norm_v1,
                     dim3(kValueHeads), dim3(kWorkgroupSize), 0U, stream,
                     convolved_qkv, z, b_input, a_input, a_log, dt_bias,
                     norm_weight, previous_recurrent_state,
                     next_recurrent_state, output, token_count);
  return hipGetLastError();
}

} // namespace sllm_linear_attention_kernel
