#include "matmul_kernel_internal.hpp"
#include "moe_expert_kernel_internal.hpp"

#include <cmath>

namespace {

__device__ __forceinline__ float bf16_to_float(const uint16_t bits) noexcept {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16U);
}

__device__ __forceinline__ uint16_t float_to_bf16(const float value) noexcept {
  uint32_t bits = __float_as_uint(value);
  bits += UINT32_C(0x7fff) + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

__device__ __forceinline__ float e2m1(const uint8_t nibble) noexcept {
  const uint8_t magnitude = nibble & 7U;
  float value = 0.0F;
  switch (magnitude) {
  case 1U:
    value = 0.5F;
    break;
  case 2U:
    value = 1.0F;
    break;
  case 3U:
    value = 1.5F;
    break;
  case 4U:
    value = 2.0F;
    break;
  case 5U:
    value = 3.0F;
    break;
  case 6U:
    value = 4.0F;
    break;
  case 7U:
    value = 6.0F;
    break;
  default:
    value = 0.0F;
    break;
  }
  return (nibble & 8U) != 0U ? -value : value;
}

__device__ __forceinline__ float e8m0(const uint8_t bits) noexcept {
  return bits == 255U ? NAN : ldexpf(1.0F, static_cast<int>(bits) - 127);
}

__device__ __forceinline__ float packed(const uint8_t *const values,
                                        const uint8_t *const scales,
                                        const uint64_t row,
                                        const uint64_t width,
                                        const uint64_t column) noexcept {
  const uint64_t packed_width = (width + 1U) / 2U;
  const uint8_t byte = values[row * packed_width + column / 2U];
  const uint8_t nibble =
      (column & 1U) == 0U ? byte & 15U : static_cast<uint8_t>(byte >> 4U);
  return e2m1(nibble) *
         e8m0(scales[row * ((width + 31U) / 32U) + column / 32U]);
}

__device__ __forceinline__ const int32_t *ids(const uint8_t *metadata) {
  return reinterpret_cast<const int32_t *>(metadata);
}

__device__ __forceinline__ const float *weights(const uint8_t *metadata,
                                                const uint64_t pairs) {
  return reinterpret_cast<const float *>(metadata + pairs * 4U);
}

__device__ __forceinline__ const int32_t *
grouped_tokens(const uint8_t *metadata, const uint64_t pairs) {
  return reinterpret_cast<const int32_t *>(
      metadata + pairs * 8U + sllm_moe_expert_kernel::kExperts * 4U +
      (sllm_moe_expert_kernel::kExperts + 1U) * 4U);
}

__device__ __forceinline__ const int32_t *grouped_slots(const uint8_t *metadata,
                                                        const uint64_t pairs) {
  return grouped_tokens(metadata, pairs) + pairs;
}

extern "C" __global__ void sllm_moe_routed_gateup_v1(
    const uint8_t *const activation_values,
    const uint8_t *const activation_scales, const uint8_t *const metadata,
    const uint8_t *const blob, uint16_t *const intermediate,
    const uint64_t pair_count) {
  const uint64_t grouped_pair = blockIdx.x;
  if (grouped_pair >= pair_count)
    return;
  const int32_t token = grouped_tokens(metadata, pair_count)[grouped_pair];
  const int32_t slot = grouped_slots(metadata, pair_count)[grouped_pair];
  const uint64_t original_pair =
      static_cast<uint64_t>(token) * sllm_moe_expert_kernel::kTopK +
      static_cast<uint32_t>(slot);
  const uint64_t expert = static_cast<uint32_t>(ids(metadata)[original_pair]);
  const uint64_t expert_gate_row =
      expert * sllm_moe_expert_kernel::kIntermediate;
  const uint8_t *const gate_values =
      blob + sllm_moe_expert_kernel::kGateValuesOffset;
  const uint8_t *const gate_scales =
      blob + sllm_moe_expert_kernel::kGateScalesOffset;
  const uint8_t *const up_values =
      blob + sllm_moe_expert_kernel::kUpValuesOffset;
  const uint8_t *const up_scales =
      blob + sllm_moe_expert_kernel::kUpScalesOffset;
  for (uint64_t row = threadIdx.x; row < sllm_moe_expert_kernel::kIntermediate;
       row += blockDim.x) {
    float gate = 0.0F;
    float up = 0.0F;
    for (uint64_t column = 0U; column < sllm_moe_expert_kernel::kHidden;
         ++column) {
      const float activation = packed(activation_values, activation_scales,
                                      static_cast<uint32_t>(token),
                                      sllm_moe_expert_kernel::kHidden, column);
      gate +=
          activation * packed(gate_values, gate_scales, expert_gate_row + row,
                              sllm_moe_expert_kernel::kHidden, column);
      up += activation * packed(up_values, up_scales, expert_gate_row + row,
                                sllm_moe_expert_kernel::kHidden, column);
    }
    const float rounded_gate = bf16_to_float(float_to_bf16(gate));
    const float silu = rounded_gate / (1.0F + expf(-rounded_gate));
    intermediate[original_pair * sllm_moe_expert_kernel::kIntermediate + row] =
        float_to_bf16(silu * bf16_to_float(float_to_bf16(up)));
  }
}

extern "C" __global__ void sllm_moe_shared_gateup_v1(
    const uint16_t *const hidden, const uint8_t *const blob,
    uint16_t *const intermediate, float *const shared_gate) {
  const uint64_t token = blockIdx.x;
  const uint16_t *const gate = reinterpret_cast<const uint16_t *>(
      blob + sllm_moe_expert_kernel::kSharedGateOffset);
  const uint16_t *const up = reinterpret_cast<const uint16_t *>(
      blob + sllm_moe_expert_kernel::kSharedUpOffset);
  const uint16_t *const gate_vector = reinterpret_cast<const uint16_t *>(
      blob + sllm_moe_expert_kernel::kSharedExpertGateOffset);
  if (threadIdx.x == 0U) {
    float sum = 0.0F;
    for (uint64_t column = 0U; column < sllm_moe_expert_kernel::kHidden;
         ++column) {
      sum += bf16_to_float(
                 hidden[token * sllm_moe_expert_kernel::kHidden + column]) *
             bf16_to_float(gate_vector[column]);
    }
    shared_gate[token] = 1.0F / (1.0F + expf(-sum));
  }
  for (uint64_t row = threadIdx.x; row < sllm_moe_expert_kernel::kIntermediate;
       row += blockDim.x) {
    float gate_sum = 0.0F;
    float up_sum = 0.0F;
    for (uint64_t column = 0U; column < sllm_moe_expert_kernel::kHidden;
         ++column) {
      const float activation = bf16_to_float(
          hidden[token * sllm_moe_expert_kernel::kHidden + column]);
      gate_sum +=
          activation *
          bf16_to_float(gate[row * sllm_moe_expert_kernel::kHidden + column]);
      up_sum +=
          activation *
          bf16_to_float(up[row * sllm_moe_expert_kernel::kHidden + column]);
    }
    const float rounded_gate = bf16_to_float(float_to_bf16(gate_sum));
    const float silu = rounded_gate / (1.0F + expf(-rounded_gate));
    intermediate[token * sllm_moe_expert_kernel::kIntermediate + row] =
        float_to_bf16(silu * bf16_to_float(float_to_bf16(up_sum)));
  }
}

extern "C" __global__ void sllm_moe_down_combine_v1(
    const uint8_t *const intermediate_values,
    const uint8_t *const intermediate_scales,
    const uint16_t *const shared_intermediate, const float *const shared_gate,
    const uint8_t *const metadata, const uint8_t *const blob,
    uint16_t *const output, const uint64_t token_count) {
  const uint64_t token = blockIdx.x;
  const uint64_t pairs = token_count * sllm_moe_expert_kernel::kTopK;
  const uint16_t *const shared_down = reinterpret_cast<const uint16_t *>(
      blob + sllm_moe_expert_kernel::kSharedDownOffset);
  const uint8_t *const down_values =
      blob + sllm_moe_expert_kernel::kDownValuesOffset;
  const uint8_t *const down_scales =
      blob + sllm_moe_expert_kernel::kDownScalesOffset;
  for (uint64_t row = blockIdx.y * blockDim.x + threadIdx.x;
       row < sllm_moe_expert_kernel::kHidden; row += gridDim.y * blockDim.x) {
    float routed = 0.0F;
    for (uint64_t slot = 0U; slot < sllm_moe_expert_kernel::kTopK; ++slot) {
      const uint64_t pair = token * sllm_moe_expert_kernel::kTopK + slot;
      const uint64_t expert = static_cast<uint32_t>(ids(metadata)[pair]);
      float sum = 0.0F;
      for (uint64_t column = 0U; column < sllm_moe_expert_kernel::kIntermediate;
           ++column) {
        sum += packed(intermediate_values, intermediate_scales, pair,
                      sllm_moe_expert_kernel::kIntermediate, column) *
               packed(down_values, down_scales,
                      expert * sllm_moe_expert_kernel::kHidden + row,
                      sllm_moe_expert_kernel::kIntermediate, column);
      }
      routed +=
          bf16_to_float(float_to_bf16(sum)) * weights(metadata, pairs)[pair];
    }
    float shared = 0.0F;
    for (uint64_t column = 0U; column < sllm_moe_expert_kernel::kIntermediate;
         ++column) {
      shared +=
          bf16_to_float(
              shared_intermediate[token *
                                      sllm_moe_expert_kernel::kIntermediate +
                                  column]) *
          bf16_to_float(
              shared_down[row * sllm_moe_expert_kernel::kIntermediate +
                          column]);
    }
    output[token * sllm_moe_expert_kernel::kHidden + row] = float_to_bf16(
        routed + bf16_to_float(float_to_bf16(shared)) * shared_gate[token]);
  }
}

__device__ __forceinline__ uint16_t
gemma4_float_to_bf16_rne(const float value) noexcept {
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

__device__ __forceinline__ float
gemma4_e4m3fn_to_float(const uint8_t bits) noexcept {
  const float sign = (bits & UINT8_C(0x80)) == 0U ? 1.0F : -1.0F;
  const uint8_t exponent = static_cast<uint8_t>((bits >> 3U) & 15U);
  const uint8_t mantissa = static_cast<uint8_t>(bits & 7U);
  if (exponent == 0U) {
    return mantissa == 0U
               ? copysignf(0.0F, sign)
               : sign * static_cast<float>(mantissa) * ldexpf(1.0F, -9);
  }
  if (exponent == 15U && mantissa == 7U) {
    return NAN;
  }
  return sign * (1.0F + static_cast<float>(mantissa) / 8.0F) *
         ldexpf(1.0F, static_cast<int>(exponent) - 7);
}

__device__ __forceinline__ uint8_t
gemma4_float_to_e4m3fn(float value) noexcept {
  const uint8_t sign = signbit(value) ? UINT8_C(0x80) : 0U;
  value = fabsf(value);
  if (value == 0.0F) {
    return sign;
  }
  if (!isfinite(value) || value >= 448.0F) {
    return static_cast<uint8_t>(sign | UINT8_C(0x7e));
  }
  uint32_t low = 0U;
  uint32_t high = UINT32_C(0x7e);
  while (low < high) {
    const uint32_t middle = (low + high) >> 1U;
    if (gemma4_e4m3fn_to_float(static_cast<uint8_t>(middle)) < value) {
      low = middle + 1U;
    } else {
      high = middle;
    }
  }
  const uint8_t upper = static_cast<uint8_t>(low);
  const uint8_t lower = upper == 0U ? 0U : static_cast<uint8_t>(upper - 1U);
  const float lower_error = value - gemma4_e4m3fn_to_float(lower);
  const float upper_error = gemma4_e4m3fn_to_float(upper) - value;
  const bool select_upper =
      upper_error < lower_error ||
      (upper_error == lower_error && (upper & 1U) == 0U && (lower & 1U) != 0U);
  return static_cast<uint8_t>(sign | (select_upper ? upper : lower));
}

__device__ __forceinline__ uint8_t gemma4_float_to_e2m1(float value) noexcept {
  const uint8_t sign = signbit(value) ? UINT8_C(0x08) : 0U;
  value = fabsf(value);
  uint8_t selected = 0U;
  float selected_error = value;
  for (uint8_t code = 1U; code != 8U; ++code) {
    const float error = fabsf(value - e2m1(code));
    if (error < selected_error ||
        (error == selected_error && (code & 1U) == 0U &&
         (selected & 1U) != 0U)) {
      selected = code;
      selected_error = error;
    }
  }
  return static_cast<uint8_t>(sign | selected);
}

__device__ __forceinline__ float gemma4_nvfp4(const uint8_t *const values,
                                              const uint8_t *const scales,
                                              const uint64_t row,
                                              const uint64_t width,
                                              const uint64_t column) noexcept {
  const uint64_t packed_width = (width + 1U) / 2U;
  const uint8_t byte = values[row * packed_width + column / 2U];
  const uint8_t code = (column & 1U) == 0U ? byte & UINT8_C(0x0f)
                                           : static_cast<uint8_t>(byte >> 4U);
  return e2m1(code) * gemma4_e4m3fn_to_float(
                          scales[row * ((width + 15U) / 16U) + column / 16U]);
}

extern "C" __global__ void sllm_gemma4_moe_quantize_active_nvfp4_v2(
    const uint16_t *const input, const uint8_t *const metadata,
    const uint8_t *const blob, uint8_t *const packed_output,
    uint8_t *const block_scales, const uint64_t pair_count,
    const uint64_t width, const uint64_t input_scale_offset,
    const uint32_t input_rows_are_pairs) {
  const uint64_t blocks_per_row = (width + 15U) / 16U;
  const uint64_t flat_block = blockIdx.x;
  if (flat_block >= pair_count * blocks_per_row) {
    return;
  }
  const uint64_t pair = flat_block / blocks_per_row;
  const uint64_t block = flat_block - pair * blocks_per_row;
  const uint32_t expert = static_cast<uint32_t>(ids(metadata)[pair]);
  if (expert >= sllm_moe_expert_kernel::gemma4::kExperts) {
    return;
  }
  const uint64_t source_row =
      input_rows_are_pairs != 0U ? pair
                                 : pair / sllm_moe_expert_kernel::gemma4::kTopK;
  const uint64_t base = block * 16U;
  const uint64_t packed_width = (width + 1U) / 2U;
  const float *const input_scales =
      reinterpret_cast<const float *>(blob + input_scale_offset);
  __shared__ float values[16];
  __shared__ float decoded_block_scale;
  if (threadIdx.x < 16U) {
    const uint64_t column = base + threadIdx.x;
    values[threadIdx.x] =
        column < width ? bf16_to_float(input[source_row * width + column])
                       : 0.0F;
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    float maximum = 0.0F;
    for (uint32_t index = 0U; index != 16U; ++index) {
      maximum = fmaxf(maximum, fabsf(values[index]));
    }
    const float global = input_scales[expert];
    const float raw_scale =
        maximum == 0.0F || !(global > 0.0F) ? 0.0F : maximum / (6.0F * global);
    const uint8_t encoded = gemma4_float_to_e4m3fn(raw_scale);
    block_scales[flat_block] = encoded;
    decoded_block_scale = gemma4_e4m3fn_to_float(encoded) * global;
  }
  __syncthreads();
  if (threadIdx.x < 8U) {
    const uint64_t first = base + static_cast<uint64_t>(threadIdx.x) * 2U;
    if (first < width) {
      const uint8_t low = decoded_block_scale > 0.0F
                              ? gemma4_float_to_e2m1(values[threadIdx.x * 2U] /
                                                     decoded_block_scale)
                              : 0U;
      const uint8_t high =
          first + 1U < width && decoded_block_scale > 0.0F
              ? gemma4_float_to_e2m1(values[threadIdx.x * 2U + 1U] /
                                     decoded_block_scale)
              : 0U;
      packed_output[pair * packed_width + first / 2U] =
          static_cast<uint8_t>(low | static_cast<uint8_t>(high << 4U));
    }
  }
}

extern "C" __global__ void sllm_gemma4_moe_gateup_nvfp4_v2(
    const uint8_t *const activation_values,
    const uint8_t *const activation_scales, const uint8_t *const metadata,
    const uint8_t *const blob, uint16_t *const intermediate,
    const uint64_t pair_count) {
  const uint64_t pair = blockIdx.x;
  if (pair >= pair_count) {
    return;
  }
  const uint32_t expert = static_cast<uint32_t>(ids(metadata)[pair]);
  if (expert >= sllm_moe_expert_kernel::gemma4::kExperts) {
    return;
  }
  const uint8_t *const gate_values =
      blob + sllm_moe_expert_kernel::gemma4::kGateValuesOffset;
  const uint8_t *const gate_scales =
      blob + sllm_moe_expert_kernel::gemma4::kGateScalesOffset;
  const uint8_t *const up_values =
      blob + sllm_moe_expert_kernel::gemma4::kUpValuesOffset;
  const uint8_t *const up_scales =
      blob + sllm_moe_expert_kernel::gemma4::kUpScalesOffset;
  const float *const gate_outer = reinterpret_cast<const float *>(
      blob + sllm_moe_expert_kernel::gemma4::kGateOuterScalesOffset);
  const float *const gate_input = reinterpret_cast<const float *>(
      blob + sllm_moe_expert_kernel::gemma4::kGateInputScalesOffset);
  const float *const up_outer = reinterpret_cast<const float *>(
      blob + sllm_moe_expert_kernel::gemma4::kUpOuterScalesOffset);
  const uint64_t expert_row = static_cast<uint64_t>(expert) *
                              sllm_moe_expert_kernel::gemma4::kIntermediate;
  for (uint64_t row = threadIdx.x;
       row < sllm_moe_expert_kernel::gemma4::kIntermediate; row += blockDim.x) {
    float gate = 0.0F;
    float up = 0.0F;
    for (uint64_t column = 0U; column < sllm_moe_expert_kernel::gemma4::kHidden;
         ++column) {
      const float activation =
          gemma4_nvfp4(activation_values, activation_scales, pair,
                       sllm_moe_expert_kernel::gemma4::kHidden, column);
      gate = fmaf(activation,
                  gemma4_nvfp4(gate_values, gate_scales, expert_row + row,
                               sllm_moe_expert_kernel::gemma4::kHidden, column),
                  gate);
      up = fmaf(activation,
                gemma4_nvfp4(up_values, up_scales, expert_row + row,
                             sllm_moe_expert_kernel::gemma4::kHidden, column),
                up);
    }
    const float common_input_scale = gate_input[expert];
    const float gate_bf16 = bf16_to_float(gemma4_float_to_bf16_rne(
        gate * gate_outer[expert] * common_input_scale));
    const float up_bf16 = bf16_to_float(
        gemma4_float_to_bf16_rne(up * up_outer[expert] * common_input_scale));
    constexpr float gelu_coefficient = 0.7978845608028654F;
    const float inner =
        gelu_coefficient *
        (gate_bf16 + 0.044715F * gate_bf16 * gate_bf16 * gate_bf16);
    const float gelu_bf16 = bf16_to_float(
        gemma4_float_to_bf16_rne(0.5F * gate_bf16 * (1.0F + tanhf(inner))));
    intermediate[pair * sllm_moe_expert_kernel::gemma4::kIntermediate + row] =
        gemma4_float_to_bf16_rne(gelu_bf16 * up_bf16);
  }
}

extern "C" __global__ void sllm_gemma4_moe_down_combine_nvfp4_v2(
    const uint8_t *const intermediate_values,
    const uint8_t *const intermediate_scales, const uint8_t *const metadata,
    const uint8_t *const blob, uint16_t *const output,
    const uint64_t token_count) {
  const uint64_t token = blockIdx.x;
  const uint64_t pair_count =
      token_count * sllm_moe_expert_kernel::gemma4::kTopK;
  const uint8_t *const down_values =
      blob + sllm_moe_expert_kernel::gemma4::kDownValuesOffset;
  const uint8_t *const down_scales =
      blob + sllm_moe_expert_kernel::gemma4::kDownScalesOffset;
  const float *const down_outer = reinterpret_cast<const float *>(
      blob + sllm_moe_expert_kernel::gemma4::kDownOuterScalesOffset);
  const float *const down_input = reinterpret_cast<const float *>(
      blob + sllm_moe_expert_kernel::gemma4::kDownInputScalesOffset);
  const uint16_t *const expert_scales = reinterpret_cast<const uint16_t *>(
      blob + sllm_moe_expert_kernel::gemma4::kPerExpertScalesOffset);
  for (uint64_t row = blockIdx.y * blockDim.x + threadIdx.x;
       row < sllm_moe_expert_kernel::gemma4::kHidden;
       row += gridDim.y * blockDim.x) {
    float routed = 0.0F;
    for (uint64_t slot = 0U; slot < sllm_moe_expert_kernel::gemma4::kTopK;
         ++slot) {
      const uint64_t pair =
          token * sllm_moe_expert_kernel::gemma4::kTopK + slot;
      const uint32_t expert = static_cast<uint32_t>(ids(metadata)[pair]);
      if (expert >= sllm_moe_expert_kernel::gemma4::kExperts) {
        continue;
      }
      float sum = 0.0F;
      const uint64_t weight_row = static_cast<uint64_t>(expert) *
                                      sllm_moe_expert_kernel::gemma4::kHidden +
                                  row;
      for (uint64_t column = 0U;
           column < sllm_moe_expert_kernel::gemma4::kIntermediate; ++column) {
        sum = fmaf(
            gemma4_nvfp4(intermediate_values, intermediate_scales, pair,
                         sllm_moe_expert_kernel::gemma4::kIntermediate, column),
            gemma4_nvfp4(down_values, down_scales, weight_row,
                         sllm_moe_expert_kernel::gemma4::kIntermediate, column),
            sum);
      }
      const float projection = bf16_to_float(gemma4_float_to_bf16_rne(
          sum * down_outer[expert] * down_input[expert]));
      routed = fmaf(projection,
                    weights(metadata, pair_count)[pair] *
                        bf16_to_float(expert_scales[expert]),
                    routed);
    }
    output[token * sllm_moe_expert_kernel::gemma4::kHidden + row] =
        gemma4_float_to_bf16_rne(routed);
  }
}

} // namespace

namespace sllm_moe_expert_kernel {

uint64_t workspace_bytes(const uint64_t token_count) noexcept {
  const uint64_t pairs = token_count * kTopK;
  return token_count * (kHidden / 2U + kHidden / 32U) +
         pairs * kIntermediate * 2U +
         pairs * (kIntermediate / 2U + kIntermediate / 32U) +
         token_count * kIntermediate * 2U + token_count * 4U;
}

hipError_t launch(const uint16_t *const hidden,
                  const uint8_t *const routing_metadata,
                  const uint8_t *const layer_blob, uint8_t *const workspace,
                  uint16_t *const output, const uint64_t token_count,
                  const hipStream_t stream) noexcept {
  const uint64_t pairs = token_count * kTopK;
  uint8_t *cursor = workspace;
  uint8_t *const activation_values = cursor;
  cursor += token_count * (kHidden / 2U);
  uint8_t *const activation_scales = cursor;
  cursor += token_count * (kHidden / 32U);
  auto *const routed_intermediate = reinterpret_cast<uint16_t *>(cursor);
  cursor += pairs * kIntermediate * 2U;
  uint8_t *const intermediate_values = cursor;
  cursor += pairs * (kIntermediate / 2U);
  uint8_t *const intermediate_scales = cursor;
  cursor += pairs * (kIntermediate / 32U);
  auto *const shared_intermediate = reinterpret_cast<uint16_t *>(cursor);
  cursor += token_count * kIntermediate * 2U;
  auto *const shared_gate = reinterpret_cast<float *>(cursor);
  hipError_t status = sllm_matmul_kernel::launch_mxfp4_quantize(
      hidden, activation_values, activation_scales, token_count, kHidden,
      stream);
  if (status != hipSuccess)
    return status;
  hipLaunchKernelGGL(sllm_moe_routed_gateup_v1,
                     dim3(static_cast<uint32_t>(pairs)), dim3(256U), 0U, stream,
                     activation_values, activation_scales, routing_metadata,
                     layer_blob, routed_intermediate, pairs);
  status = hipGetLastError();
  if (status != hipSuccess)
    return status;
  status = sllm_matmul_kernel::launch_mxfp4_quantize(
      routed_intermediate, intermediate_values, intermediate_scales, pairs,
      kIntermediate, stream);
  if (status != hipSuccess)
    return status;
  hipLaunchKernelGGL(sllm_moe_shared_gateup_v1,
                     dim3(static_cast<uint32_t>(token_count)), dim3(256U), 0U,
                     stream, hidden, layer_blob, shared_intermediate,
                     shared_gate);
  status = hipGetLastError();
  if (status != hipSuccess)
    return status;
  hipLaunchKernelGGL(sllm_moe_down_combine_v1,
                     dim3(static_cast<uint32_t>(token_count), 8U), dim3(256U),
                     0U, stream, intermediate_values, intermediate_scales,
                     shared_intermediate, shared_gate, routing_metadata,
                     layer_blob, output, token_count);
  return hipGetLastError();
}

uint64_t gemma4_workspace_bytes(const uint64_t token_count) noexcept {
  return token_count * gemma4::kWorkspaceBytesPerToken;
}

hipError_t launch_gemma4(const uint16_t *const hidden,
                         const uint8_t *const routing_metadata,
                         const uint8_t *const layer_blob,
                         uint8_t *const workspace, uint16_t *const output,
                         const uint64_t token_count,
                         const hipStream_t stream) noexcept {
  const uint64_t pairs = token_count * gemma4::kTopK;
  uint8_t *cursor = workspace;
  uint8_t *const activation_values = cursor;
  cursor += pairs * gemma4::kActivationValueBytesPerPair;
  uint8_t *const activation_scales = cursor;
  cursor += pairs * gemma4::kActivationScaleBytesPerPair;
  auto *const routed_intermediate = reinterpret_cast<uint16_t *>(cursor);
  cursor += pairs * gemma4::kIntermediateBytesPerPair;
  uint8_t *const intermediate_values = cursor;
  cursor += pairs * gemma4::kIntermediateValueBytesPerPair;
  uint8_t *const intermediate_scales = cursor;

  const uint64_t activation_blocks =
      pairs * gemma4::kActivationScaleBytesPerPair;
  hipLaunchKernelGGL(sllm_gemma4_moe_quantize_active_nvfp4_v2,
                     dim3(static_cast<uint32_t>(activation_blocks)), dim3(256U),
                     0U, stream, hidden, routing_metadata, layer_blob,
                     activation_values, activation_scales, pairs,
                     gemma4::kHidden, gemma4::kGateInputScalesOffset, 0U);
  hipError_t status = hipGetLastError();
  if (status != hipSuccess) {
    return status;
  }
  hipLaunchKernelGGL(sllm_gemma4_moe_gateup_nvfp4_v2,
                     dim3(static_cast<uint32_t>(pairs)), dim3(256U), 0U, stream,
                     activation_values, activation_scales, routing_metadata,
                     layer_blob, routed_intermediate, pairs);
  status = hipGetLastError();
  if (status != hipSuccess) {
    return status;
  }
  const uint64_t intermediate_blocks =
      pairs * gemma4::kIntermediateScaleBytesPerPair;
  hipLaunchKernelGGL(sllm_gemma4_moe_quantize_active_nvfp4_v2,
                     dim3(static_cast<uint32_t>(intermediate_blocks)),
                     dim3(256U), 0U, stream, routed_intermediate,
                     routing_metadata, layer_blob, intermediate_values,
                     intermediate_scales, pairs, gemma4::kIntermediate,
                     gemma4::kDownInputScalesOffset, 1U);
  status = hipGetLastError();
  if (status != hipSuccess) {
    return status;
  }
  hipLaunchKernelGGL(sllm_gemma4_moe_down_combine_nvfp4_v2,
                     dim3(static_cast<uint32_t>(token_count), 11U), dim3(256U),
                     0U, stream, intermediate_values, intermediate_scales,
                     routing_metadata, layer_blob, output, token_count);
  return hipGetLastError();
}

} // namespace sllm_moe_expert_kernel
