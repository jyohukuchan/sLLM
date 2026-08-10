#ifndef SLLM_KV_STATE_KERNEL_INTERNAL_HPP
#define SLLM_KV_STATE_KERNEL_INTERNAL_HPP

#include "sllm/hip.h"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_kv_state_kernel {

constexpr const char *kLogicalKernelId = "kv_state.bf16_to_f16_transpose.v1";
constexpr const char *kDeviceSymbol = "sllm_kv_state_bf16_to_f16_transpose_v1";

hipError_t launch(const uint16_t *key_input, const uint16_t *value_input,
                  uint16_t *key_output, uint16_t *value_output,
                  uint32_t token_count, uint64_t capacity_tokens,
                  uint64_t start_position, hipStream_t stream) noexcept;

} // namespace sllm_kv_state_kernel

#endif // SLLM_KV_STATE_KERNEL_INTERNAL_HPP
