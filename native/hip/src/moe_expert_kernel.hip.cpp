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

} // namespace sllm_moe_expert_kernel
