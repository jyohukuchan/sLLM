#ifndef SLLM_MINIMAX_M3_MOE_ROUTE_KERNEL_INTERNAL_HPP
#define SLLM_MINIMAX_M3_MOE_ROUTE_KERNEL_INTERNAL_HPP

#include "sllm/hip.h"

#include <hip/hip_runtime.h>

namespace sllm_minimax_m3_moe_route_kernel {

inline constexpr const char *kLogicalKernelId =
    "sllm.minimax_m3_moe_route.sigmoid_top4.v1";
inline constexpr const char *kDeviceSymbol =
    "sllm_minimax_m3_moe_route_sigmoid_top4_v1";

hipError_t launch(const float *logits, const float *selection_bias,
                  int32_t *expert_ids, float *expert_weights,
                  int32_t *expert_counts, int32_t *expert_offsets,
                  int32_t *grouped_token_ids, int32_t *grouped_topk_slots,
                  int32_t *status, uint64_t token_count,
                  hipStream_t stream) noexcept;

} // namespace sllm_minimax_m3_moe_route_kernel

#endif // SLLM_MINIMAX_M3_MOE_ROUTE_KERNEL_INTERNAL_HPP
