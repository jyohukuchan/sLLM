#include "token_selector_pq_internal.hpp"

#include <hip/hip_runtime.h>

#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)

namespace sllm_token_selector_pq_kernel {

hipError_t launch(const sllm_token_selector_pq::SupportRecordV1 *const,
                  const uint32_t,
                  const sllm_token_selector_pq::SupportRecordV1 *const,
                  const uint32_t, const sllm_token_selector_pq::DraftIdsV1,
                  const uint32_t, const uint32_t, const uint64_t,
                  const uint64_t,
                  sllm_token_selector_pq::DecisionRecordV1 *const,
                  const hipStream_t) noexcept {
  return hipErrorNotSupported;
}

} // namespace sllm_token_selector_pq_kernel

#else

extern "C" __global__
__launch_bounds__(64, 1) void sllm_token_selector_sparse_pq_k20_v1(
    const sllm_token_selector_pq::SupportRecordV1 *const target_rows,
    const uint32_t target_row_count,
    const sllm_token_selector_pq::SupportRecordV1 *const draft_rows,
    const uint32_t draft_row_count,
    const sllm_token_selector_pq::DraftIdsV1 draft_ids,
    const uint32_t draft_id_count, const uint32_t width, const uint64_t seed,
    const uint64_t absolute_position,
    sllm_token_selector_pq::DecisionRecordV1 *const output) {
  if ((blockIdx.x == 0U) && (blockIdx.y == 0U) && (blockIdx.z == 0U) &&
      (threadIdx.x == 0U) && (threadIdx.y == 0U) && (threadIdx.z == 0U)) {
    (void)sllm_token_selector_pq::run_sparse_pq(
        target_rows, target_row_count, draft_rows, draft_row_count,
        draft_ids.ids, draft_id_count, width, seed, absolute_position, output);
  }
}

namespace sllm_token_selector_pq_kernel {

hipError_t
launch(const sllm_token_selector_pq::SupportRecordV1 *const target_rows,
       const uint32_t target_row_count,
       const sllm_token_selector_pq::SupportRecordV1 *const draft_rows,
       const uint32_t draft_row_count,
       const sllm_token_selector_pq::DraftIdsV1 draft_ids,
       const uint32_t draft_id_count, const uint32_t width, const uint64_t seed,
       const uint64_t absolute_position,
       sllm_token_selector_pq::DecisionRecordV1 *const output,
       const hipStream_t stream) noexcept {
  const dim3 block(64U, 1U, 1U);
  hipLaunchKernelGGL(sllm_token_selector_sparse_pq_k20_v1, dim3(1U, 1U, 1U),
                     block, 0U, stream, target_rows, target_row_count,
                     draft_rows, draft_row_count, draft_ids, draft_id_count,
                     width, seed, absolute_position, output);
  return hipGetLastError();
}

} // namespace sllm_token_selector_pq_kernel

#endif // SLLM_PUBLIC_RUNTIME_HOST_TEST
