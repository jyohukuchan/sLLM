#ifndef SLLM_GEMMA_ATTENTION_KERNEL_INTERNAL_HPP
#define SLLM_GEMMA_ATTENTION_KERNEL_INTERNAL_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_gemma_attention_kernel {

constexpr const char *kLogicalKernelId =
    "gemma_causal_attention.online_softmax_gqa_bf16.v1";
constexpr const char *kDeviceSymbol =
    "sllm_gemma_causal_attention_online_softmax_gqa_bf16_v1";
constexpr uint32_t kWorkgroupSize = 256U;

hipError_t launch(const uint16_t *query, const uint16_t *key,
                  const uint16_t *value, uint16_t *output, uint32_t query_count,
                  uint64_t start_position, uint64_t committed_kv_length,
                  uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                  uint64_t sliding_window, hipStream_t stream) noexcept;

} // namespace sllm_gemma_attention_kernel

#endif // SLLM_GEMMA_ATTENTION_KERNEL_INTERNAL_HPP
