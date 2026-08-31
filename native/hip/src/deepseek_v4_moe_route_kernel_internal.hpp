#ifndef SLLM_DEEPSEEK_V4_MOE_ROUTE_KERNEL_INTERNAL_HPP
#define SLLM_DEEPSEEK_V4_MOE_ROUTE_KERNEL_INTERNAL_HPP

#include "sllm/hip.h"
#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_deepseek_v4_moe_route_kernel {

inline constexpr const char *kLogicalKernelIdScore =
    "deepseek_v4_moe_route.bf16_f32.score.v1";
inline constexpr const char *kLogicalKernelIdHash =
    "deepseek_v4_moe_route.bf16_f32.hash.v1";
inline constexpr const char *kDeviceSymbol =
    "sllm_deepseek_v4_moe_route_score_hash_v1";

hipError_t launch(const uint16_t *logits, const float *selection_bias,
                  const int32_t *hash_expert_ids, int32_t *expert_ids,
                  float *expert_weights, int32_t *expert_counts,
                  int32_t *expert_offsets, int32_t *grouped_token_ids,
                  int32_t *grouped_topk_slots, int32_t *status,
                  uint64_t token_count, uint32_t mode, uint32_t renormalize,
                  float routed_scale, hipStream_t stream) noexcept;

} // namespace sllm_deepseek_v4_moe_route_kernel

#endif // SLLM_DEEPSEEK_V4_MOE_ROUTE_KERNEL_INTERNAL_HPP
