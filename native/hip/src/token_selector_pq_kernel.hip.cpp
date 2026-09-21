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

hipError_t launch_graph(const sllm_token_selector_pq::SupportRecordV1 *const,
                        const uint32_t,
                        const sllm_token_selector_pq::SupportRecordV1 *const,
                        const uint32_t, const uint32_t *, const uint32_t,
                        const uint32_t,
                        const sllm_decode_control::SelectorRecordV1 *,
                        sllm_decode_control::ControlV1 *, const uint32_t,
                        sllm_token_selector_pq::DecisionRecordV1 *const,
                        const hipStream_t) noexcept {
  return hipErrorNotSupported;
}

} // namespace sllm_token_selector_pq_kernel

#else

__device__ void decision_from_target_selector(
    const sllm_decode_control::SelectorRecordV1 *const selector,
    sllm_token_selector_pq::DecisionRecordV1 *const output) {
  sllm_token_selector_pq::clear_decision(output);
  if (selector == nullptr || output == nullptr || selector->status != 0U ||
      selector->reserved != 0U || selector->token_id < 0 ||
      !__builtin_isfinite(selector->logprob) || selector->logprob > 0.0F) {
    if (output != nullptr) {
      output->status = static_cast<uint32_t>(
          sllm_token_selector_pq::DecisionStatus::kInputSupportStatus);
    }
    return;
  }
  output->width = 0U;
  output->accepted_count = 0U;
  output->emitted_count = 1U;
  output->rejected_at = sllm_token_selector_pq::kNoRejection;
  output->draws_used = 1U;
  output->emitted_ids[0] = static_cast<uint32_t>(selector->token_id);
  output->target_logprobs[0] = static_cast<double>(selector->logprob);
}

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

extern "C" __global__
__launch_bounds__(64, 1) void sllm_token_selector_sparse_pq_k20_graph_v1(
    const sllm_token_selector_pq::SupportRecordV1 *const target_rows,
    const uint32_t target_row_count,
    const sllm_token_selector_pq::SupportRecordV1 *const draft_rows,
    const uint32_t draft_row_count, const uint32_t *const draft_ids,
    const uint32_t draft_id_count, const uint32_t width,
    const sllm_decode_control::SelectorRecordV1 *const target_selector,
    sllm_decode_control::ControlV1 *const control, const uint32_t phase_row,
    sllm_token_selector_pq::DecisionRecordV1 *const output) {
  if ((blockIdx.x != 0U) || (threadIdx.x != 0U)) {
    return;
  }
  if (control == nullptr || output == nullptr ||
      control->version != sllm_decode_control::kVersion ||
      control->phase_active == 0U || control->halted != 0U ||
      phase_row >= control->phase_rows ||
      control->phase_counter > UINT64_MAX - phase_row || draft_ids == nullptr ||
      width == 0U || width > sllm_token_selector_pq::kMaxWidth ||
      target_row_count != width + 1U || draft_row_count != width ||
      draft_id_count != width || control->active_width > width) {
    sllm_token_selector_pq::clear_decision(output);
    if (output != nullptr) {
      output->status = static_cast<uint32_t>(
          sllm_token_selector_pq::DecisionStatus::kCounterOverflow);
    }
    return;
  }
  const uint32_t active_width = control->active_width;
  if (active_width == 0U) {
    decision_from_target_selector(target_selector, output);
    return;
  }
  sllm_token_selector_pq::DraftIdsV1 ids{};
  for (uint32_t index = 0U; index < active_width; ++index) {
    ids.ids[index] = draft_ids[index];
  }
  (void)sllm_token_selector_pq::run_sparse_pq(
      target_rows, active_width + 1U, draft_rows, active_width, ids.ids,
      active_width, active_width, control->seed, control->sampler_counter,
      output);
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

hipError_t
launch_graph(const sllm_token_selector_pq::SupportRecordV1 *const target_rows,
             const uint32_t target_row_count,
             const sllm_token_selector_pq::SupportRecordV1 *const draft_rows,
             const uint32_t draft_row_count, const uint32_t *const draft_ids,
             const uint32_t draft_id_count, const uint32_t width,
             const sllm_decode_control::SelectorRecordV1 *const target_selector,
             sllm_decode_control::ControlV1 *const control,
             const uint32_t phase_row,
             sllm_token_selector_pq::DecisionRecordV1 *const output,
             const hipStream_t stream) noexcept {
  if (target_rows == nullptr || draft_rows == nullptr || draft_ids == nullptr ||
      control == nullptr || output == nullptr || width == 0U ||
      width > sllm_token_selector_pq::kMaxWidth ||
      target_row_count != width + 1U || draft_row_count != width ||
      draft_id_count != width) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_token_selector_sparse_pq_k20_graph_v1,
                     dim3(1U, 1U, 1U), dim3(64U, 1U, 1U), 0U, stream,
                     target_rows, target_row_count, draft_rows, draft_row_count,
                     draft_ids, draft_id_count, width, target_selector, control,
                     phase_row, output);
  return hipGetLastError();
}

} // namespace sllm_token_selector_pq_kernel

#endif // SLLM_PUBLIC_RUNTIME_HOST_TEST
