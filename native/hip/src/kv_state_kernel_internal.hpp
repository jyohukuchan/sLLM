#ifndef SLLM_KV_STATE_KERNEL_INTERNAL_HPP
#define SLLM_KV_STATE_KERNEL_INTERNAL_HPP

#include "sllm/hip.h"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_kv_state_kernel {

constexpr const char *kLogicalKernelId = "kv_state.bf16_to_f16_token_major.v2";
constexpr const char *kDeviceSymbol =
    "sllm_kv_state_bf16_to_f16_token_major_v2";
constexpr const char *kFp8LogicalKernelId =
    "kv_state.bf16_to_fp8_token_major.v1";
constexpr const char *kFp8DeviceSymbol =
    "sllm_kv_state_bf16_to_fp8_token_major_v1";
constexpr const char *kNvfp4LogicalKernelId =
    "kv_state.bf16_to_nvfp4_token_major.v1";
constexpr const char *kNvfp4DeviceSymbol =
    "sllm_kv_state_bf16_to_nvfp4_token_major_v1";

hipError_t launch(const uint16_t *key_input, const uint16_t *value_input,
                  void *key_output, void *value_output, void *key_scales,
                  void *value_scales, float *key_outer_scales,
                  float *value_outer_scales, uint32_t token_count,
                  uint64_t capacity_tokens, uint64_t start_position,
                  uint32_t head_count, uint32_t head_dim, uint32_t encoding,
                  hipStream_t stream) noexcept;

} // namespace sllm_kv_state_kernel

#endif // SLLM_KV_STATE_KERNEL_INTERNAL_HPP
