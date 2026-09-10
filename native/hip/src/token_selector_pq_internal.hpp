#ifndef SLLM_TOKEN_SELECTOR_PQ_INTERNAL_HPP
#define SLLM_TOKEN_SELECTOR_PQ_INTERNAL_HPP

#include "token_selector_pq_algorithm.hpp"

#include <hip/hip_runtime.h>

namespace sllm_token_selector_pq_kernel {

inline constexpr const char *kLogicalKernelId =
    "token_selector.sparse_pq_k20.v1";
inline constexpr const char *kDeviceSymbol =
    "sllm_token_selector_sparse_pq_k20_v1";

hipError_t launch(const sllm_token_selector_pq::SupportRecordV1 *target_rows,
                  uint32_t target_row_count,
                  const sllm_token_selector_pq::SupportRecordV1 *draft_rows,
                  uint32_t draft_row_count,
                  sllm_token_selector_pq::DraftIdsV1 draft_ids,
                  uint32_t draft_id_count, uint32_t width, uint64_t seed,
                  uint64_t absolute_position,
                  sllm_token_selector_pq::DecisionRecordV1 *output,
                  hipStream_t stream) noexcept;

} // namespace sllm_token_selector_pq_kernel

#endif // SLLM_TOKEN_SELECTOR_PQ_INTERNAL_HPP
