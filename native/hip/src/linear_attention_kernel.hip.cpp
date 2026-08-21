// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase9-gdn-layout-001
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase35-gdn-column-state-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// f5919bf458ef190468b5c329bb293f8a54a1e69c,
// ggml/src/ggml-cuda/gated_delta_net.cu
// SPDX-License-Identifier: MIT

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

template <uint32_t WaveWidth>
__device__ __forceinline__ float wave_sum(float value) noexcept {
  for (uint32_t offset = WaveWidth / 2U; offset != 0U; offset >>= 1U) {
    value += __shfl_down(value, offset, WaveWidth);
  }
  return value;
}

__device__ __forceinline__ uint64_t recurrent_state_index(
    const uint64_t state_base, const uint32_t dimension,
    const uint32_t key_dimension, const uint32_t head_dim) noexcept {
#if defined(__gfx1030__)
  // RDNA2 benefits from the wave-coalesced transposed state layout adapted
  // from llama.cpp gated_delta_net.cu at fixed commit
  // f5919bf458ef190468b5c329bb293f8a54a1e69c.
  return state_base + static_cast<uint64_t>(key_dimension) * head_dim +
         dimension;
#else
  // On the measured RDNA4 target, retaining one contiguous state row per
  // thread is faster. The state is private to this exact-target runtime.
  return state_base + static_cast<uint64_t>(dimension) * head_dim +
         key_dimension;
#endif
}

} // namespace

extern "C" __global__
__launch_bounds__(128, 1) void sllm_linear_attention_causal_conv_silu_v1(
    const uint16_t *const qkv, const uint16_t *const conv_weight,
    const uint16_t *const previous_conv_state, uint16_t *const convolved_qkv,
    uint16_t *const next_conv_state, const uint32_t token_count,
    const uint32_t qkv_width, const uint32_t conv_kernel_size) {
  const uint64_t width = qkv_width;
  const uint64_t history = conv_kernel_size - 1U;
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  const uint64_t output_elements = static_cast<uint64_t>(token_count) * width;
  if (index < output_elements) {
    const uint64_t token = index / width;
    const uint64_t channel = index % width;
    float sum = 0.0F;
    for (uint64_t tap = 0U; tap != conv_kernel_size; ++tap) {
      const int64_t source =
          static_cast<int64_t>(token + tap) - static_cast<int64_t>(history);
      const uint16_t value =
          source < 0 ? previous_conv_state[static_cast<uint64_t>(
                                               source +
                                               static_cast<int64_t>(history)) *
                                               width +
                                           channel]
                     : qkv[static_cast<uint64_t>(source) * width + channel];
      sum += bf16_to_float(value) *
             bf16_to_float(conv_weight[channel * conv_kernel_size + tap]);
    }
    const float silu = sum / (1.0F + expf(-sum));
    convolved_qkv[index] = float_to_bf16_rne_bits(silu);
    return;
  }

  const uint64_t history_index = index - output_elements;
  if (history_index < history * width) {
    const uint64_t history_row = history_index / width;
    const uint64_t channel = history_index % width;
    const int64_t source = static_cast<int64_t>(token_count) -
                           static_cast<int64_t>(history) +
                           static_cast<int64_t>(history_row);
    next_conv_state[history_index] =
        source < 0
            ? previous_conv_state[static_cast<uint64_t>(
                                      source + static_cast<int64_t>(history)) *
                                      width +
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
    const uint32_t token_count, const uint32_t qk_heads,
    const uint32_t value_heads, const uint32_t head_dim,
    const uint32_t qkv_width, const uint32_t output_width) {
  const uint32_t value_head = blockIdx.x;
  const uint32_t dimension = threadIdx.x;
  if (value_head >= value_heads || dimension >= head_dim) {
    return;
  }
  const uint32_t qk_head = value_head / (value_heads / qk_heads);
  const uint64_t state_base =
      static_cast<uint64_t>(value_head) * head_dim * head_dim;
  __shared__ float q_values[sllm_linear_attention_kernel::kHeadDim];
  __shared__ float k_values[sllm_linear_attention_kernel::kHeadDim];
  __shared__ float q_inverse_norm;
  __shared__ float k_inverse_norm;
  __shared__ float beta;
  __shared__ float decay;
  __shared__ float output_values[sllm_linear_attention_kernel::kHeadDim];
  __shared__ float output_inverse_rms;
  __shared__ float q_wave_sums[4];
  __shared__ float k_wave_sums[4];
  __shared__ float output_wave_sums[4];
  const uint32_t lane = dimension & 31U;
  const uint32_t wave = dimension >> 5U;

  for (uint32_t token = 0U; token != token_count; ++token) {
    const uint64_t qkv_row = static_cast<uint64_t>(token) * qkv_width;
    q_values[dimension] = bf16_to_float(
        convolved_qkv[qkv_row + static_cast<uint64_t>(qk_head) * head_dim +
                      dimension]);
    k_values[dimension] = bf16_to_float(
        convolved_qkv[qkv_row + static_cast<uint64_t>(qk_heads) * head_dim +
                      static_cast<uint64_t>(qk_head) * head_dim + dimension]);
    __syncthreads();
    const float q_square = q_values[dimension] * q_values[dimension];
    const float k_square = k_values[dimension] * k_values[dimension];
    const float q_wave_sum = wave_sum<32U>(q_square);
    const float k_wave_sum = wave_sum<32U>(k_square);
    if (lane == 0U) {
      q_wave_sums[wave] = q_wave_sum;
      k_wave_sums[wave] = k_wave_sum;
    }
    __syncthreads();
    if (dimension == 0U) {
      float q_sum = 0.0F;
      float k_sum = 0.0F;
      // Phase 29 approved the wave32 tree only for gfx1030/gfx1201. Preserve
      // the Phase 28 sequential order in the exact wave64/gfx942 build so the
      // RDNA optimization does not silently widen its numerical scope.
#if defined(SLLM_HIP_COMPILE_WAVE64) && SLLM_HIP_COMPILE_WAVE64 == 1
      for (uint32_t index = 0U; index != head_dim; ++index) {
        q_sum += q_values[index] * q_values[index];
        k_sum += k_values[index] * k_values[index];
#else
      for (uint32_t index = 0U; index != 4U; ++index) {
        q_sum += q_wave_sums[index];
        k_sum += k_wave_sums[index];
#endif
      }
      q_inverse_norm = 1.0F / sqrtf(q_sum + 1.0e-6F);
      k_inverse_norm = 1.0F / sqrtf(k_sum + 1.0e-6F);
      const uint64_t scalar_index =
          static_cast<uint64_t>(token) * value_heads + value_head;
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
    q_values[dimension] *= 1.0F / sqrtf(static_cast<float>(head_dim));
    k_values[dimension] = bf16_to_float(
        float_to_bf16_rne_bits(k_values[dimension] * k_inverse_norm));
    __syncthreads();

    float previous_projection = 0.0F;
    for (uint32_t key_dimension = 0U; key_dimension != head_dim;
         ++key_dimension) {
      const uint64_t state_index =
          recurrent_state_index(state_base, dimension, key_dimension, head_dim);
      // The first token reads the previous transactional buffer directly;
      // later tokens continue from this request's next buffer. Combining the
      // copy, decay, and projection pass preserves the original FP32
      // operation order while removing one full recurrent-state traversal.
      const float state = token == 0U ? previous_recurrent_state[state_index]
                                      : next_recurrent_state[state_index];
      const float decayed = state * decay;
      next_recurrent_state[state_index] = decayed;
      previous_projection += decayed * k_values[key_dimension];
    }
    const float value = bf16_to_float(
        convolved_qkv[qkv_row +
                      static_cast<uint64_t>(2U * qk_heads) * head_dim +
                      static_cast<uint64_t>(value_head) * head_dim +
                      dimension]);
    const float residual = value - previous_projection;
    float current_projection = 0.0F;
    for (uint32_t key_dimension = 0U; key_dimension != head_dim;
         ++key_dimension) {
      const uint64_t index =
          recurrent_state_index(state_base, dimension, key_dimension, head_dim);
      const float updated = next_recurrent_state[index] +
                            beta * residual * k_values[key_dimension];
      next_recurrent_state[index] = updated;
      current_projection += updated * q_values[key_dimension];
    }
    output_values[dimension] =
        bf16_to_float(float_to_bf16_rne_bits(current_projection));
    const float output_square =
        output_values[dimension] * output_values[dimension];
    const float output_wave_sum = wave_sum<32U>(output_square);
    if (lane == 0U) {
      output_wave_sums[wave] = output_wave_sum;
    }
    __syncthreads();
    if (dimension == 0U) {
      float sum = 0.0F;
#if defined(SLLM_HIP_COMPILE_WAVE64) && SLLM_HIP_COMPILE_WAVE64 == 1
      for (uint32_t index = 0U; index != head_dim; ++index) {
        sum += output_values[index] * output_values[index];
#else
      for (uint32_t index = 0U; index != 4U; ++index) {
        sum += output_wave_sums[index];
#endif
      }
      output_inverse_rms =
          1.0F / sqrtf(sum / static_cast<float>(head_dim) + 1.0e-6F);
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

// Normalize each shared Q/K head once, then materialize the per-value-head
// scalar gates used by the column-owned recurrent kernel. The normalized Q/K
// BF16 stages are identical to the one-row provider and replace their input
// slots in the request-local convolution scratch.
#pragma clang fp contract(off)
extern "C" __global__
__launch_bounds__(128, 1) void sllm_linear_attention_column_preprocess_v2(
    uint16_t *const convolved_qkv, const uint16_t *const b_input,
    const uint16_t *const a_input, const float *const a_log,
    const uint16_t *const dt_bias, float *const beta, float *const decay,
    const uint32_t token_count, const uint32_t qk_heads,
    const uint32_t value_heads, const uint32_t head_dim,
    const uint32_t qkv_width) {
  const uint64_t flat = blockIdx.x;
  const uint32_t token = static_cast<uint32_t>(flat / qk_heads);
  const uint32_t qk_head = static_cast<uint32_t>(flat % qk_heads);
  const uint32_t dimension = threadIdx.x;
  if (token >= token_count || dimension >= head_dim) {
    return;
  }
  const uint32_t lane = dimension & 31U;
  const uint32_t wave = dimension >> 5U;
  const uint64_t row = static_cast<uint64_t>(token) * qkv_width;
  const uint64_t q_index =
      row + static_cast<uint64_t>(qk_head) * head_dim + dimension;
  const uint64_t k_index =
      row + static_cast<uint64_t>(qk_heads + qk_head) * head_dim + dimension;
  const float q_value = bf16_to_float(convolved_qkv[q_index]);
  const float k_value = bf16_to_float(convolved_qkv[k_index]);
  const float q_wave_sum = wave_sum<32U>(q_value * q_value);
  const float k_wave_sum = wave_sum<32U>(k_value * k_value);
  __shared__ float q_wave_sums[4];
  __shared__ float k_wave_sums[4];
  __shared__ float q_inverse_norm;
  __shared__ float k_inverse_norm;
  if (lane == 0U) {
    q_wave_sums[wave] = q_wave_sum;
    k_wave_sums[wave] = k_wave_sum;
  }
  __syncthreads();
  if (dimension == 0U) {
    float q_sum = 0.0F;
    float k_sum = 0.0F;
#pragma unroll
    for (uint32_t index = 0U; index != 4U; ++index) {
      q_sum += q_wave_sums[index];
      k_sum += k_wave_sums[index];
    }
    q_inverse_norm = 1.0F / sqrtf(q_sum + 1.0e-6F);
    k_inverse_norm = 1.0F / sqrtf(k_sum + 1.0e-6F);
  }
  __syncthreads();
  convolved_qkv[q_index] = float_to_bf16_rne_bits(q_value * q_inverse_norm);
  convolved_qkv[k_index] = float_to_bf16_rne_bits(k_value * k_inverse_norm);

  const uint32_t value_heads_per_qk = value_heads / qk_heads;
  if (dimension < value_heads_per_qk) {
    const uint32_t value_head = qk_head * value_heads_per_qk + dimension;
    const uint64_t scalar_index =
        static_cast<uint64_t>(token) * value_heads + value_head;
    const float b_value = bf16_to_float(b_input[scalar_index]);
    beta[scalar_index] =
        bf16_to_float(float_to_bf16_rne_bits(1.0F / (1.0F + expf(-b_value))));
    const float a_value = bf16_to_float(a_input[scalar_index]) +
                          bf16_to_float(dt_bias[value_head]);
    decay[scalar_index] = expf(-expf(a_log[value_head]) * softplus(a_value));
  }
}

// Each wave owns one output column. Its lanes keep the complete recurrent
// column in registers across the token loop, reducing S^T*k and S^T*q within
// the wave and publishing the transactional state only once at the end.
#pragma clang fp contract(off)
extern "C" __global__
__launch_bounds__(128, 1) void sllm_linear_attention_recurrent_column_state_v2(
    const uint16_t *const convolved_qkv, const float *const beta,
    const float *const decay, const float *const previous_recurrent_state,
    float *const next_recurrent_state, uint16_t *const output,
    const uint32_t token_count, const uint32_t qk_heads,
    const uint32_t value_heads, const uint32_t head_dim,
    const uint32_t qkv_width, const uint32_t output_width) {
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kWavesPerBlock = 4U;
  constexpr uint32_t kRowsPerLane =
      sllm_linear_attention_kernel::kHeadDim / kWaveSize;
  const uint32_t column_groups = head_dim / kWavesPerBlock;
  const uint32_t value_head = blockIdx.x / column_groups;
  const uint32_t column_group = blockIdx.x % column_groups;
  const uint32_t lane = threadIdx.x & (kWaveSize - 1U);
  const uint32_t wave = threadIdx.x / kWaveSize;
  const uint32_t column = column_group * kWavesPerBlock + wave;
  if (value_head >= value_heads || column >= head_dim) {
    return;
  }
  const uint32_t qk_head = value_head / (value_heads / qk_heads);
  const uint64_t state_base =
      static_cast<uint64_t>(value_head) * head_dim * head_dim;
  float state_values[kRowsPerLane];
#pragma unroll
  for (uint32_t index = 0U; index < kRowsPerLane; ++index) {
    const uint32_t row_dimension = lane + index * kWaveSize;
    const uint64_t state_index =
        recurrent_state_index(state_base, column, row_dimension, head_dim);
    state_values[index] = previous_recurrent_state[state_index];
  }

  const float q_scale = rsqrtf(static_cast<float>(head_dim));
  for (uint32_t token = 0U; token != token_count; ++token) {
    const uint64_t qkv_row = static_cast<uint64_t>(token) * qkv_width;
    float q_values[kRowsPerLane];
    float k_values[kRowsPerLane];
#pragma unroll
    for (uint32_t index = 0U; index < kRowsPerLane; ++index) {
      const uint32_t row_dimension = lane + index * kWaveSize;
      q_values[index] =
          bf16_to_float(
              convolved_qkv[qkv_row +
                            static_cast<uint64_t>(qk_head) * head_dim +
                            row_dimension]) *
          q_scale;
      k_values[index] = bf16_to_float(
          convolved_qkv[qkv_row +
                        static_cast<uint64_t>(qk_heads + qk_head) * head_dim +
                        row_dimension]);
    }
    const uint64_t scalar_index =
        static_cast<uint64_t>(token) * value_heads + value_head;
    const float current_decay = decay[scalar_index];
    float previous_partial = 0.0F;
#pragma unroll
    for (uint32_t index = 0U; index < kRowsPerLane; ++index) {
      state_values[index] *= current_decay;
      previous_partial += state_values[index] * k_values[index];
    }
    const float previous_projection = wave_sum<kWaveSize>(previous_partial);
    float delta = 0.0F;
    if (lane == 0U) {
      const float value = bf16_to_float(
          convolved_qkv[qkv_row +
                        static_cast<uint64_t>(2U * qk_heads + value_head) *
                            head_dim +
                        column]);
      delta = beta[scalar_index] * (value - previous_projection);
    }
    delta = __shfl(delta, 0U, kWaveSize);
    float current_partial = 0.0F;
#pragma unroll
    for (uint32_t index = 0U; index < kRowsPerLane; ++index) {
      state_values[index] += delta * k_values[index];
      current_partial += state_values[index] * q_values[index];
    }
    const float current_projection = wave_sum<kWaveSize>(current_partial);
    if (lane == 0U) {
      const uint64_t output_index =
          static_cast<uint64_t>(token) * output_width +
          static_cast<uint64_t>(value_head) * head_dim + column;
      output[output_index] = float_to_bf16_rne_bits(current_projection);
    }
  }

#pragma unroll
  for (uint32_t index = 0U; index < kRowsPerLane; ++index) {
    const uint32_t row_dimension = lane + index * kWaveSize;
    const uint64_t state_index =
        recurrent_state_index(state_base, column, row_dimension, head_dim);
    next_recurrent_state[state_index] = state_values[index];
  }
}

#pragma clang fp contract(off)
extern "C" __global__
__launch_bounds__(128, 1) void sllm_linear_attention_column_postprocess_v2(
    const uint16_t *const z, const float *const norm_weight,
    uint16_t *const output, const uint32_t token_count,
    const uint32_t value_heads, const uint32_t head_dim,
    const uint32_t output_width) {
  const uint64_t flat = blockIdx.x;
  const uint32_t token = static_cast<uint32_t>(flat / value_heads);
  const uint32_t value_head = static_cast<uint32_t>(flat % value_heads);
  const uint32_t dimension = threadIdx.x;
  if (token >= token_count || dimension >= head_dim) {
    return;
  }
  const uint32_t lane = dimension & 31U;
  const uint32_t wave = dimension >> 5U;
  const uint64_t output_index = static_cast<uint64_t>(token) * output_width +
                                static_cast<uint64_t>(value_head) * head_dim +
                                dimension;
  const float output_value = bf16_to_float(output[output_index]);
  const float output_wave_sum = wave_sum<32U>(output_value * output_value);
  __shared__ float output_wave_sums[4];
  __shared__ float output_inverse_rms;
  if (lane == 0U) {
    output_wave_sums[wave] = output_wave_sum;
  }
  __syncthreads();
  if (dimension == 0U) {
    float sum = 0.0F;
#pragma unroll
    for (uint32_t index = 0U; index != 4U; ++index) {
      sum += output_wave_sums[index];
    }
    output_inverse_rms =
        1.0F / sqrtf(sum / static_cast<float>(head_dim) + 1.0e-6F);
  }
  __syncthreads();
  const float normalized =
      bf16_to_float(float_to_bf16_rne_bits(output_value * output_inverse_rms));
  const float z_value = bf16_to_float(z[output_index]);
  const float z_silu = z_value / (1.0F + expf(-z_value));
  output[output_index] =
      float_to_bf16_rne_bits(normalized * norm_weight[dimension] * z_silu);
}

namespace sllm_linear_attention_kernel {

hipError_t
launch_convolution(const uint16_t *const qkv, const uint16_t *const conv_weight,
                   const uint16_t *const previous_conv_state,
                   uint16_t *const convolved_qkv,
                   uint16_t *const next_conv_state, const uint32_t token_count,
                   const uint32_t qkv_width, const uint32_t conv_kernel_size,
                   const hipStream_t stream) noexcept {
  if (qkv == nullptr || conv_weight == nullptr ||
      previous_conv_state == nullptr || convolved_qkv == nullptr ||
      next_conv_state == nullptr || token_count == 0U || qkv_width == 0U ||
      conv_kernel_size < 2U) {
    return hipErrorInvalidValue;
  }
  const uint64_t elements =
      static_cast<uint64_t>(token_count) * qkv_width +
      static_cast<uint64_t>(conv_kernel_size - 1U) * qkv_width;
  const uint64_t workgroup = static_cast<uint64_t>(kWorkgroupSize);
  const uint64_t blocks =
      elements / workgroup + static_cast<uint64_t>(elements % workgroup != 0U);
  if (blocks > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_linear_attention_causal_conv_silu_v1,
                     dim3(static_cast<uint32_t>(blocks)), dim3(kWorkgroupSize),
                     0U, stream, qkv, conv_weight, previous_conv_state,
                     convolved_qkv, next_conv_state, token_count, qkv_width,
                     conv_kernel_size);
  return hipGetLastError();
}

hipError_t
launch_recurrent(const uint16_t *const convolved_qkv, const uint16_t *const z,
                 const uint16_t *const b_input, const uint16_t *const a_input,
                 const float *const a_log, const uint16_t *const dt_bias,
                 const float *const norm_weight,
                 const float *const previous_recurrent_state,
                 float *const next_recurrent_state, uint16_t *const output,
                 const uint32_t token_count, const uint32_t qk_heads,
                 const uint32_t value_heads, const uint32_t head_dim,
                 const uint32_t qkv_width, const uint32_t output_width,
                 const hipStream_t stream) noexcept {
  if (convolved_qkv == nullptr || z == nullptr || b_input == nullptr ||
      a_input == nullptr || a_log == nullptr || dt_bias == nullptr ||
      norm_weight == nullptr || previous_recurrent_state == nullptr ||
      next_recurrent_state == nullptr || output == nullptr ||
      token_count == 0U || qk_heads == 0U || value_heads == 0U ||
      value_heads % qk_heads != 0U || head_dim != kHeadDim) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_linear_attention_recurrent_gated_norm_v1,
                     dim3(value_heads), dim3(kWorkgroupSize), 0U, stream,
                     convolved_qkv, z, b_input, a_input, a_log, dt_bias,
                     norm_weight, previous_recurrent_state,
                     next_recurrent_state, output, token_count, qk_heads,
                     value_heads, head_dim, qkv_width, output_width);
  return hipGetLastError();
}

hipError_t launch_column_preprocess(
    uint16_t *const convolved_qkv, const uint16_t *const b_input,
    const uint16_t *const a_input, const float *const a_log,
    const uint16_t *const dt_bias, float *const beta, float *const decay,
    const uint32_t token_count, const uint32_t qk_heads,
    const uint32_t value_heads, const uint32_t head_dim,
    const uint32_t qkv_width, const hipStream_t stream) noexcept {
  if (convolved_qkv == nullptr || b_input == nullptr || a_input == nullptr ||
      a_log == nullptr || dt_bias == nullptr || beta == nullptr ||
      decay == nullptr || token_count == 0U || qk_heads == 0U ||
      value_heads == 0U || value_heads % qk_heads != 0U ||
      head_dim != kHeadDim) {
    return hipErrorInvalidValue;
  }
  const uint64_t blocks = static_cast<uint64_t>(token_count) * qk_heads;
  if (blocks > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_linear_attention_column_preprocess_v2,
                     dim3(static_cast<uint32_t>(blocks)), dim3(kWorkgroupSize),
                     0U, stream, convolved_qkv, b_input, a_input, a_log,
                     dt_bias, beta, decay, token_count, qk_heads, value_heads,
                     head_dim, qkv_width);
  return hipGetLastError();
}

hipError_t launch_column_recurrent(
    const uint16_t *const convolved_qkv, const float *const beta,
    const float *const decay, const float *const previous_recurrent_state,
    float *const next_recurrent_state, uint16_t *const output,
    const uint32_t token_count, const uint32_t qk_heads,
    const uint32_t value_heads, const uint32_t head_dim,
    const uint32_t qkv_width, const uint32_t output_width,
    const hipStream_t stream) noexcept {
  if (convolved_qkv == nullptr || beta == nullptr || decay == nullptr ||
      previous_recurrent_state == nullptr || next_recurrent_state == nullptr ||
      output == nullptr || token_count == 0U || qk_heads == 0U ||
      value_heads == 0U || value_heads % qk_heads != 0U ||
      head_dim != kHeadDim || head_dim % 4U != 0U) {
    return hipErrorInvalidValue;
  }
  const uint64_t blocks = static_cast<uint64_t>(value_heads) * head_dim / 4U;
  if (blocks > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_linear_attention_recurrent_column_state_v2,
                     dim3(static_cast<uint32_t>(blocks)), dim3(kWorkgroupSize),
                     0U, stream, convolved_qkv, beta, decay,
                     previous_recurrent_state, next_recurrent_state, output,
                     token_count, qk_heads, value_heads, head_dim, qkv_width,
                     output_width);
  return hipGetLastError();
}

hipError_t launch_column_postprocess(
    const uint16_t *const z, const float *const norm_weight,
    uint16_t *const output, const uint32_t token_count,
    const uint32_t value_heads, const uint32_t head_dim,
    const uint32_t output_width, const hipStream_t stream) noexcept {
  if (z == nullptr || norm_weight == nullptr || output == nullptr ||
      token_count == 0U || value_heads == 0U || head_dim != kHeadDim) {
    return hipErrorInvalidValue;
  }
  const uint64_t blocks = static_cast<uint64_t>(token_count) * value_heads;
  if (blocks > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_linear_attention_column_postprocess_v2,
                     dim3(static_cast<uint32_t>(blocks)), dim3(kWorkgroupSize),
                     0U, stream, z, norm_weight, output, token_count,
                     value_heads, head_dim, output_width);
  return hipGetLastError();
}

} // namespace sllm_linear_attention_kernel
