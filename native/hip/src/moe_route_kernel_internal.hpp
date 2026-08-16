#ifndef SLLM_MOE_ROUTE_KERNEL_INTERNAL_HPP
#define SLLM_MOE_ROUTE_KERNEL_INTERNAL_HPP

#include "sllm/hip.h"
#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_moe_route_kernel {

inline constexpr const char *kLogicalKernelId =
    "moe_route.bf16.stable_topk_group.v1";
inline constexpr const char *kDeviceSymbol =
    "sllm_moe_route_stable_topk_group_v1";

hipError_t launch(const uint16_t *logits, int32_t *expert_ids,
                  float *expert_weights, int32_t *expert_counts,
                  int32_t *expert_offsets, int32_t *grouped_token_ids,
                  int32_t *grouped_topk_slots, int32_t *status,
                  uint64_t token_count, uint64_t expert_count,
                  uint32_t selected_expert_count, hipStream_t stream) noexcept;

} // namespace sllm_moe_route_kernel

#endif // SLLM_MOE_ROUTE_KERNEL_INTERNAL_HPP
