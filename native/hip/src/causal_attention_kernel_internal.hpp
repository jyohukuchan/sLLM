#ifndef SLLM_CAUSAL_ATTENTION_KERNEL_INTERNAL_HPP
#define SLLM_CAUSAL_ATTENTION_KERNEL_INTERNAL_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_causal_attention_kernel {

constexpr const char *kLogicalKernelId =
    "causal_attention.online_softmax_gqa.v2";
constexpr const char *kDeviceSymbol =
    "sllm_causal_attention_online_softmax_gqa_v2";

hipError_t launch(const uint16_t *query, const uint16_t *key,
                  const uint16_t *value, uint16_t *output, uint32_t query_count,
                  uint64_t capacity_tokens, uint64_t start_position,
                  uint64_t committed_kv_length, uint32_t q_heads,
                  uint32_t kv_heads, uint32_t head_dim,
                  hipStream_t stream) noexcept;

} // namespace sllm_causal_attention_kernel

#endif // SLLM_CAUSAL_ATTENTION_KERNEL_INTERNAL_HPP
