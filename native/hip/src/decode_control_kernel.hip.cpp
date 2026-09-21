#include "decode_control_kernel_internal.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

using sllm_decode_control::ControlV1;
using sllm_decode_control::HaltAllAccept;
using sllm_decode_control::HaltBudget;
using sllm_decode_control::HaltInvalid;
using sllm_decode_control::HaltNoop;
using sllm_decode_control::HaltStop;
using sllm_decode_control::InputKind;
using sllm_decode_control::ResultV1;
using sllm_decode_control::SelectorRecordV1;
using sllm_decode_control::Status;
using DecisionRecord = sllm_token_selector_pq::DecisionRecordV1;

constexpr std::uint32_t kNoStop = sllm_decode_control::kNoStop;
constexpr std::uint32_t kMaxStopIds = sllm_decode_control::kMaxStopIds;
constexpr std::uint32_t kMaxWidth = sllm_decode_control::kMaxWidth;
constexpr std::uint32_t kMaxEmitted = sllm_decode_control::kMaxEmitted;
constexpr std::uint32_t kResultRingSlots =
    sllm_decode_control::kResultRingSlots;
constexpr std::uint64_t kMaxU64 = UINT64_MAX;

__device__ __host__ constexpr std::uint32_t status_code(const Status status) {
  return static_cast<std::uint32_t>(status);
}

__device__ bool finite_f32(const float value) noexcept {
  return __builtin_isfinite(value) != 0;
}

__device__ bool finite_f64(const double value) noexcept {
  return __builtin_isfinite(value) != 0;
}

__device__ void clear_result(ResultV1 *const result,
                             const std::uint64_t generation,
                             const std::uint32_t status) noexcept {
  result->generation = generation;
  result->status = status;
  result->count = 0U;
  result->commit_rows = 0U;
  result->accepted = 0U;
  result->width = 0U;
  result->halt_flags = 0U;
  result->reserved0 = 0U;
  result->reserved_padding = 0U;
  result->model_position_before = 0U;
  result->model_position_after = 0U;
  result->counter_before = 0U;
  result->counter_after = 0U;
  for (std::uint32_t index = 0U; index < kMaxEmitted; ++index) {
    result->selected_ids[index] = 0U;
    result->logprobs[index] = 0.0;
  }
  result->reserved1 = 0U;
  result->reserved_tail = 0U;
}

__device__ void clear_result_ring(ResultV1 *const ring,
                                  const std::uint64_t generation,
                                  const std::uint32_t status) noexcept {
  clear_result(&ring[generation % kResultRingSlots], generation, status);
}

__device__ void fill_noop_identity(ResultV1 *const result,
                                   const ControlV1 *const control) noexcept {
  if (result == nullptr || control == nullptr) {
    return;
  }
  result->width = control->width;
  result->model_position_before = control->model_position;
  result->model_position_after = control->model_position;
  result->counter_before = control->sampler_counter;
  result->counter_after = control->sampler_counter;
}

__device__ void write_noop_result(ControlV1 *const control,
                                  ResultV1 *const result_ring,
                                  const std::uint32_t halt_flags) noexcept {
  if (control == nullptr || result_ring == nullptr) {
    return;
  }
  ResultV1 *const result = &result_ring[control->generation % kResultRingSlots];
  clear_result(result, control->generation, status_code(Status::Noop));
  fill_noop_identity(result, control);
  result->halt_flags = halt_flags | HaltNoop;
}

__device__ void
write_sticky_error_result(ControlV1 *const control,
                          ResultV1 *const result_ring) noexcept {
  if (control == nullptr || result_ring == nullptr) {
    return;
  }
  ResultV1 *const result = &result_ring[control->generation % kResultRingSlots];
  clear_result(result, control->generation, control->status);
  fill_noop_identity(result, control);
  result->halt_flags = HaltInvalid;
}

__device__ bool checked_add_u64(const std::uint64_t left,
                                const std::uint64_t right,
                                std::uint64_t *const output) noexcept {
  if (output == nullptr || left > kMaxU64 - right) {
    return false;
  }
  *output = left + right;
  return true;
}

__device__ bool checked_mul_u64(const std::uint64_t left,
                                const std::uint64_t right,
                                std::uint64_t *const output) noexcept {
  if (output == nullptr || (left != 0U && right > kMaxU64 / left)) {
    return false;
  }
  *output = left * right;
  return true;
}

__device__ bool control_header_valid(const ControlV1 *const control) noexcept {
  return control != nullptr &&
         control->version == sllm_decode_control::kVersion &&
         control->status == status_code(Status::Ok) &&
         (control->mode == sllm_decode_control::kModeTargetOnly ||
          control->mode == sllm_decode_control::kModeMtp) &&
         control->width != 0U && control->width <= kMaxWidth &&
         control->active_width <= control->width && control->capacity != 0U &&
         control->model_position <= control->capacity &&
         control->output_count <= control->output_limit &&
         control->output_limit - control->output_count <=
             control->capacity - control->model_position &&
         control->generation != kMaxU64;
}

__device__ void set_control_status(ControlV1 *const control,
                                   const Status status) noexcept {
  if (control == nullptr) {
    return;
  }
  if (status == Status::Ok) {
    if (control->status == status_code(Status::Ok)) {
      control->status = status_code(Status::Ok);
    }
    return;
  }
  if (control->status == status_code(Status::Ok)) {
    control->status = status_code(status);
  }
  control->halted = 1U;
  control->phase_active = 0U;
  control->commit_rows = 0U;
  control->publish_count = 0U;
}

__device__ bool token_is_stop(const std::uint32_t token,
                              const std::uint32_t *const stop_ids,
                              const std::uint32_t stop_count) noexcept {
  for (std::uint32_t stop = 0U; stop < stop_count; ++stop) {
    if (stop_ids[stop] == token) {
      return true;
    }
  }
  return false;
}

__device__ bool
validate_stop_inputs(const std::uint32_t *const stop_ids,
                     const std::uint32_t stop_count,
                     const std::uint32_t vocabulary_size) noexcept {
  if (stop_count > kMaxStopIds || vocabulary_size == 0U ||
      (stop_count != 0U && stop_ids == nullptr)) {
    return false;
  }
  for (std::uint32_t index = 0U; index < stop_count; ++index) {
    if (stop_ids[index] >= vocabulary_size) {
      return false;
    }
    for (std::uint32_t prior = 0U; prior < index; ++prior) {
      if (stop_ids[prior] == stop_ids[index]) {
        return false;
      }
    }
  }
  return true;
}

struct ParsedRecord final {
  std::uint32_t width;
  std::uint32_t accepted;
  std::uint32_t emitted;
  std::uint32_t ids[kMaxEmitted];
  double logprobs[kMaxEmitted];
};

__device__ bool parse_selector(const SelectorRecordV1 *const selector,
                               const ControlV1 *const control,
                               const std::uint32_t vocabulary_size,
                               ParsedRecord *const parsed) noexcept {
  if (selector == nullptr || control == nullptr || parsed == nullptr ||
      control->width != 1U || selector->status != 0U ||
      selector->reserved != 0U || selector->token_id < 0 ||
      static_cast<std::uint32_t>(selector->token_id) >= vocabulary_size ||
      !finite_f32(selector->logprob) || selector->logprob > 0.0F) {
    return false;
  }
  parsed->width = 1U;
  parsed->accepted = 0U;
  parsed->emitted = 1U;
  parsed->ids[0] = static_cast<std::uint32_t>(selector->token_id);
  parsed->logprobs[0] = static_cast<double>(selector->logprob);
  for (std::uint32_t index = 1U; index < kMaxEmitted; ++index) {
    parsed->ids[index] = 0U;
    parsed->logprobs[index] = 0.0;
  }
  return true;
}

__device__ bool parse_decision(const DecisionRecord *const decision,
                               const ControlV1 *const control,
                               const std::uint32_t vocabulary_size,
                               ParsedRecord *const parsed) noexcept {
  if (decision == nullptr || control == nullptr || parsed == nullptr ||
      decision->version != sllm_token_selector_pq::kDecisionVersion ||
      decision->status != 0U || decision->reserved0 != 0U ||
      decision->reserved1 != 0U || decision->width != control->active_width ||
      decision->width > control->width || decision->width > kMaxWidth ||
      decision->accepted_count > decision->width ||
      decision->emitted_count != decision->accepted_count + 1U ||
      decision->emitted_count > kMaxEmitted || vocabulary_size == 0U) {
    return false;
  }
  const std::uint32_t expected_draws =
      decision->accepted_count == decision->width
          ? decision->width + 1U
          : decision->accepted_count + 2U;
  if (decision->draws_used != expected_draws ||
      ((decision->accepted_count == decision->width) !=
       (decision->rejected_at == sllm_token_selector_pq::kNoRejection)) ||
      (decision->accepted_count < decision->width &&
       decision->rejected_at != decision->accepted_count)) {
    return false;
  }
  parsed->width = decision->width;
  parsed->accepted = decision->accepted_count;
  parsed->emitted = decision->emitted_count;
  for (std::uint32_t index = 0U; index < kMaxEmitted; ++index) {
    parsed->ids[index] = decision->emitted_ids[index];
    parsed->logprobs[index] = decision->target_logprobs[index];
    if (index < decision->emitted_count) {
      if (decision->emitted_ids[index] >= vocabulary_size ||
          !finite_f64(decision->target_logprobs[index]) ||
          decision->target_logprobs[index] > 0.0) {
        return false;
      }
    } else if (decision->emitted_ids[index] != 0U ||
               decision->target_logprobs[index] != 0.0) {
      return false;
    }
  }
  return true;
}

__device__ void write_invalid_result(ControlV1 *const control,
                                     ResultV1 *const result_ring,
                                     const Status status,
                                     const std::uint32_t halt_flags) noexcept {
  const std::uint64_t generation =
      control == nullptr ? 0U : control->generation;
  if (result_ring != nullptr) {
    clear_result_ring(result_ring, generation, status_code(status));
    ResultV1 *const result = &result_ring[generation % kResultRingSlots];
    result->halt_flags = halt_flags;
  }
  set_control_status(control, status);
}

__device__ void commit_record(ControlV1 *const control,
                              const ParsedRecord &parsed,
                              const std::uint32_t *const stop_ids,
                              const std::uint32_t stop_count,
                              const std::uint32_t vocabulary_size,
                              ResultV1 *const result_ring) noexcept {
  if (control == nullptr || result_ring == nullptr ||
      !control_header_valid(control) ||
      !validate_stop_inputs(stop_ids, stop_count, vocabulary_size)) {
    write_invalid_result(
        control, result_ring,
        !validate_stop_inputs(stop_ids, stop_count, vocabulary_size)
            ? Status::InvalidStopIds
            : Status::InvalidVersion,
        HaltInvalid);
    return;
  }
  if (control->halted != 0U) {
    write_noop_result(control, result_ring, HaltNoop);
    control->status = status_code(Status::Noop);
    control->commit_rows = 0U;
    control->publish_count = 0U;
    return;
  }
  if (control->output_count >= control->output_limit) {
    write_noop_result(control, result_ring, HaltBudget);
    control->status = status_code(Status::Noop);
    control->halted = 1U;
    control->phase_active = 0U;
    control->commit_rows = 0U;
    control->publish_count = 0U;
    return;
  }

  const std::uint64_t remaining = control->output_limit - control->output_count;
  std::uint32_t publish_count = parsed.emitted;
  if (remaining < static_cast<std::uint64_t>(publish_count)) {
    publish_count = static_cast<std::uint32_t>(remaining);
  }
  std::uint32_t stop_row = kNoStop;
  for (std::uint32_t row = 0U; row < publish_count; ++row) {
    if (token_is_stop(parsed.ids[row], stop_ids, stop_count)) {
      stop_row = row;
      publish_count = row + 1U;
      break;
    }
  }
  if (publish_count == 0U) {
    write_noop_result(control, result_ring, HaltBudget);
    control->status = status_code(Status::Noop);
    control->halted = 1U;
    control->phase_active = 0U;
    control->commit_rows = 0U;
    control->publish_count = 0U;
    return;
  }

  const std::uint64_t commit_rows = publish_count;
  std::uint64_t model_after = 0U;
  std::uint64_t counter_after = 0U;
  std::uint64_t output_after = 0U;
  std::uint64_t generation_after = 0U;
  if (!checked_add_u64(control->model_position, commit_rows, &model_after) ||
      model_after > control->capacity ||
      !checked_add_u64(control->sampler_counter, publish_count,
                       &counter_after) ||
      !checked_add_u64(control->output_count, publish_count, &output_after) ||
      !checked_add_u64(control->generation, 1U, &generation_after)) {
    write_invalid_result(control, result_ring, Status::InvalidCapacity,
                         HaltInvalid);
    return;
  }

  ResultV1 *const result = &result_ring[generation_after % kResultRingSlots];
  clear_result(result, generation_after, status_code(Status::Ok));
  result->count = publish_count;
  result->commit_rows = static_cast<std::uint32_t>(commit_rows);
  result->accepted = parsed.accepted;
  result->width = parsed.width;
  result->model_position_before = control->model_position;
  result->model_position_after = model_after;
  result->counter_before = control->sampler_counter;
  result->counter_after = counter_after;
  for (std::uint32_t row = 0U; row < publish_count; ++row) {
    result->selected_ids[row] = parsed.ids[row];
    result->logprobs[row] = parsed.logprobs[row];
  }
  if (parsed.accepted == parsed.width) {
    result->halt_flags |= HaltAllAccept;
  }
  if (stop_row != kNoStop) {
    result->halt_flags |= HaltStop;
    result->reserved0 = stop_row;
  } else if (output_after >= control->output_limit) {
    result->halt_flags |= HaltBudget;
  }

  control->status = status_code(Status::Ok);
  control->generation = generation_after;
  control->model_position = model_after;
  control->sampler_counter = counter_after;
  control->output_count = output_after;
  control->commit_rows = static_cast<std::uint32_t>(commit_rows);
  control->publish_count = publish_count;
  control->accepted_count = parsed.accepted;
  control->stop_row = stop_row;
  control->hidden_row = static_cast<std::uint32_t>(commit_rows - 1U);
  if (stop_row == kNoStop) {
    control->pending_token = parsed.ids[publish_count - 1U];
  }
  if (stop_row != kNoStop || output_after >= control->output_limit) {
    control->halted = 1U;
    control->phase_active = 0U;
  }
}

extern "C" __global__ void sllm_decode_control_begin_phase_v1(
    ControlV1 *const control, const std::uint32_t kind,
    const std::uint32_t index, const std::uint32_t rows) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  const bool kind_valid = kind == sllm_decode_control::kPhaseTarget ||
                          kind == sllm_decode_control::kPhaseDraft ||
                          kind == sllm_decode_control::kPhaseMtpAlign;
  const bool target_bounds =
      control != nullptr && kind == sllm_decode_control::kPhaseTarget &&
      rows != 0U && rows <= control->width + 1U && index <= control->width &&
      (control->mode == sllm_decode_control::kModeMtp ||
       (index == 0U && rows == 1U));
  const bool draft_bounds = control != nullptr &&
                            kind == sllm_decode_control::kPhaseDraft &&
                            rows == 1U && index < control->width;
  const bool align_bounds = control != nullptr &&
                            kind == sllm_decode_control::kPhaseMtpAlign &&
                            rows == 1U && index == control->width;
  if (control != nullptr && control->halted != 0U) {
    control->phase_active = 0U;
    control->commit_rows = 0U;
    control->publish_count = 0U;
    return;
  }
  if (!control_header_valid(control) || !kind_valid ||
      !(target_bounds || draft_bounds || align_bounds)) {
    set_control_status(control, Status::InvalidPhase);
    if (control != nullptr) {
      control->phase_active = 0U;
    }
    return;
  }
  if (control->mode == sllm_decode_control::kModeMtp &&
      kind == sllm_decode_control::kPhaseDraft && index == 0U) {
    if (control->model_position >= control->capacity ||
        control->output_count >= control->output_limit) {
      set_control_status(control, Status::InvalidCapacity);
      return;
    }
    const std::uint64_t budget_rows =
        control->output_limit - control->output_count;
    const std::uint64_t capacity_rows =
        control->capacity - control->model_position - 1U;
    std::uint64_t active =
        budget_rows < capacity_rows ? budget_rows : capacity_rows;
    if (active > control->width) {
      active = control->width;
    }
    control->active_width = static_cast<std::uint32_t>(active);
  }
  const bool inactive_draft = control->mode == sllm_decode_control::kModeMtp &&
                              kind == sllm_decode_control::kPhaseDraft &&
                              index >= control->active_width;
  const bool inactive_target_row =
      control->mode == sllm_decode_control::kModeMtp &&
      kind == sllm_decode_control::kPhaseTarget && rows == 1U &&
      index > control->active_width;
  if (inactive_draft || inactive_target_row) {
    control->phase_position = control->model_position;
    control->phase_counter = control->sampler_counter;
    control->phase_seed = control->seed;
    control->phase_rows = 0U;
    control->phase_index = index;
    control->phase_kind = kind;
    control->phase_active = 0U;
    control->status = status_code(Status::Ok);
    return;
  }
  std::uint64_t phase_position = control->model_position;
  std::uint64_t phase_counter = control->sampler_counter;
  std::uint64_t phase_seed = control->seed;
  const std::uint32_t effective_index =
      control->mode == sllm_decode_control::kModeMtp &&
              kind == sllm_decode_control::kPhaseMtpAlign
          ? control->active_width
          : index;
  if (kind == sllm_decode_control::kPhaseTarget) {
    if (!checked_add_u64(phase_counter, index, &phase_counter)) {
      set_control_status(control, Status::InvalidCounter);
      control->phase_active = 0U;
      return;
    }
  } else {
    if (!checked_add_u64(phase_position, effective_index, &phase_position) ||
        !checked_mul_u64(phase_counter, 9U, &phase_counter) ||
        !checked_add_u64(phase_counter, effective_index, &phase_counter)) {
      set_control_status(control, Status::InvalidCounter);
      control->phase_active = 0U;
      return;
    }
    phase_seed ^= sllm_decode_control::kQDraftSeedDomain64;
  }
  control->phase_position = phase_position;
  control->phase_counter = phase_counter;
  control->phase_seed = phase_seed;
  control->phase_rows = control->mode == sllm_decode_control::kModeMtp &&
                                kind == sllm_decode_control::kPhaseTarget
                            ? control->active_width + 1U
                            : rows;
  control->phase_index = effective_index;
  control->phase_kind = kind;
  control->phase_active = 1U;
  control->status = status_code(Status::Ok);
}

extern "C" __global__ void sllm_decode_control_commit_selector_v1(
    ControlV1 *const control, const SelectorRecordV1 *const selector,
    const std::uint32_t *const stop_ids, const std::uint32_t stop_count,
    const std::uint32_t vocabulary_size, ResultV1 *const result_ring) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  if (control != nullptr && control->status != status_code(Status::Ok)) {
    write_sticky_error_result(control, result_ring);
    return;
  }
  if (control != nullptr && control->halted != 0U) {
    write_noop_result(control, result_ring, HaltNoop);
    return;
  }
  ParsedRecord parsed{};
  if (!parse_selector(selector, control, vocabulary_size, &parsed)) {
    write_invalid_result(control, result_ring, Status::InvalidSelector,
                         HaltInvalid);
    return;
  }
  commit_record(control, parsed, stop_ids, stop_count, vocabulary_size,
                result_ring);
}

extern "C" __global__ void sllm_decode_control_commit_decision_v1(
    ControlV1 *const control, const DecisionRecord *const decision,
    const std::uint32_t *const stop_ids, const std::uint32_t stop_count,
    const std::uint32_t vocabulary_size, ResultV1 *const result_ring) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  if (control != nullptr && control->status != status_code(Status::Ok)) {
    write_sticky_error_result(control, result_ring);
    return;
  }
  if (control != nullptr && control->halted != 0U) {
    write_noop_result(control, result_ring, HaltNoop);
    return;
  }
  ParsedRecord parsed{};
  if (!parse_decision(decision, control, vocabulary_size, &parsed)) {
    write_invalid_result(control, result_ring, Status::InvalidDecision,
                         HaltInvalid);
    return;
  }
  commit_record(control, parsed, stop_ids, stop_count, vocabulary_size,
                result_ring);
}

extern "C" __global__ void
sllm_decode_control_gather_selector_v1(const SelectorRecordV1 *const selector,
                                       std::int32_t *const output_token) {
  if (blockIdx.x == 0U && threadIdx.x == 0U && selector != nullptr &&
      output_token != nullptr && selector->status == 0U &&
      selector->reserved == 0U && selector->token_id >= 0) {
    *output_token = selector->token_id;
  }
}

extern "C" __global__ void sllm_decode_control_gather_decision_v1(
    const ControlV1 *const control, const DecisionRecord *const decision,
    std::int32_t *const output_tokens, const std::uint32_t output_capacity) {
  if (blockIdx.x != 0U || threadIdx.x != 0U || control == nullptr ||
      decision == nullptr || output_tokens == nullptr ||
      decision->status != 0U || control->publish_count > output_capacity) {
    return;
  }
  for (std::uint32_t row = 0U; row < control->publish_count; ++row) {
    output_tokens[row] = static_cast<std::int32_t>(decision->emitted_ids[row]);
  }
}

extern "C" __global__ void
sllm_decode_control_gather_result_v1(const ResultV1 *const result,
                                     std::int32_t *const output_tokens,
                                     const std::uint32_t output_capacity) {
  if (blockIdx.x != 0U || threadIdx.x != 0U || result == nullptr ||
      output_tokens == nullptr || result->count > output_capacity ||
      result->status != status_code(Status::Ok)) {
    return;
  }
  for (std::uint32_t row = 0U; row < result->count; ++row) {
    output_tokens[row] = static_cast<std::int32_t>(result->selected_ids[row]);
  }
}

extern "C" __global__ void sllm_decode_control_gather_hidden_v1(
    ControlV1 *const control, const std::uint16_t *const hidden_rows,
    const std::uint32_t row_count, const std::uint32_t hidden_width,
    std::uint16_t *const output_hidden) {
  const std::uint32_t lane =
      static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
      static_cast<std::uint32_t>(threadIdx.x);
  // A discarded lookahead must not replace the committed hidden carry.
  // Valid EOS commits keep status Ok and still publish their selected row.
  if (control != nullptr && control->status != status_code(Status::Ok))
    return;
  if (control == nullptr || hidden_rows == nullptr ||
      output_hidden == nullptr || hidden_width == 0U ||
      control->hidden_row >= row_count) {
    if (lane == 0U) {
      set_control_status(control, Status::InvalidPosition);
    }
    return;
  }
  if (lane < hidden_width) {
    const std::uint64_t offset =
        static_cast<std::uint64_t>(control->hidden_row) * hidden_width + lane;
    output_hidden[lane] = hidden_rows[offset];
  }
}

extern "C" __global__ void sllm_decode_control_gather_active_token_v1(
    ControlV1 *const control, const std::int32_t *const draft_ids,
    const std::uint32_t draft_capacity, std::int32_t *const output_token) {
  if (blockIdx.x != 0U || threadIdx.x != 0U || control == nullptr ||
      draft_ids == nullptr || output_token == nullptr ||
      control->status != status_code(Status::Ok) || control->halted != 0U ||
      control->active_width > draft_capacity) {
    return;
  }
  *output_token = control->active_width == 0U
                      ? static_cast<std::int32_t>(control->pending_token)
                      : draft_ids[control->active_width - 1U];
}

extern "C" __global__ void sllm_decode_control_gather_active_hidden_v1(
    ControlV1 *const control, const std::uint16_t *const target_hidden_rows,
    const std::uint32_t target_row_count,
    const std::uint16_t *const previous_hidden,
    const std::uint32_t hidden_width, std::uint16_t *const output_hidden) {
  const std::uint32_t lane =
      static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
      static_cast<std::uint32_t>(threadIdx.x);
  if (control == nullptr || target_hidden_rows == nullptr ||
      previous_hidden == nullptr || output_hidden == nullptr ||
      control->status != status_code(Status::Ok) || control->halted != 0U ||
      hidden_width == 0U || control->active_width >= target_row_count) {
    return;
  }
  if (lane < hidden_width) {
    output_hidden[lane] =
        control->active_width == 0U
            ? previous_hidden[lane]
            : target_hidden_rows[static_cast<std::uint64_t>(
                                     control->active_width - 1U) *
                                     hidden_width +
                                 lane];
  }
}

} // namespace

#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)

namespace sllm_decode_control {

hipError_t launch_begin_phase(ControlV1 *, const std::uint32_t,
                              const std::uint32_t, const std::uint32_t,
                              const hipStream_t) noexcept {
  return hipErrorNotSupported;
}

hipError_t launch_commit_step(ControlV1 *, const void *, const InputKind,
                              const std::uint32_t *, const std::uint32_t,
                              const std::uint32_t, ResultV1 *,
                              const hipStream_t) noexcept {
  return hipErrorNotSupported;
}

hipError_t launch_gather_selector_token(const SelectorRecordV1 *,
                                        std::int32_t *,
                                        const hipStream_t) noexcept {
  return hipErrorNotSupported;
}

hipError_t launch_gather_decision_tokens(
    const ControlV1 *, const sllm_token_selector_pq::DecisionRecordV1 *,
    std::int32_t *, const std::uint32_t, const hipStream_t) noexcept {
  return hipErrorNotSupported;
}

hipError_t launch_gather_result_tokens(const ResultV1 *, std::int32_t *,
                                       const std::uint32_t,
                                       const hipStream_t) noexcept {
  return hipErrorNotSupported;
}

hipError_t launch_gather_hidden(const ControlV1 *, const std::uint16_t *,
                                const std::uint32_t, const std::uint32_t,
                                std::uint16_t *, const hipStream_t) noexcept {
  return hipErrorNotSupported;
}

hipError_t launch_gather_active_token(ControlV1 *, const std::int32_t *,
                                      const std::uint32_t, std::int32_t *,
                                      const hipStream_t) noexcept {
  return hipErrorNotSupported;
}

hipError_t launch_gather_active_hidden(ControlV1 *, const std::uint16_t *,
                                       const std::uint32_t,
                                       const std::uint16_t *,
                                       const std::uint32_t, std::uint16_t *,
                                       const hipStream_t) noexcept {
  return hipErrorNotSupported;
}

} // namespace sllm_decode_control

#else

namespace sllm_decode_control {

hipError_t launch_begin_phase(ControlV1 *const control,
                              const std::uint32_t kind,
                              const std::uint32_t index,
                              const std::uint32_t rows,
                              const hipStream_t stream) noexcept {
  hipLaunchKernelGGL(sllm_decode_control_begin_phase_v1, dim3(1U), dim3(1U), 0U,
                     stream, control, kind, index, rows);
  return hipGetLastError();
}

hipError_t launch_commit_step(
    ControlV1 *const control, const void *const record,
    const InputKind input_kind, const std::uint32_t *const stop_ids,
    const std::uint32_t stop_count, const std::uint32_t vocabulary_size,
    ResultV1 *const result_ring, const hipStream_t stream) noexcept {
  if (input_kind == InputKind::Selector) {
    hipLaunchKernelGGL(sllm_decode_control_commit_selector_v1, dim3(1U),
                       dim3(1U), 0U, stream, control,
                       static_cast<const SelectorRecordV1 *>(record), stop_ids,
                       stop_count, vocabulary_size, result_ring);
  } else {
    hipLaunchKernelGGL(
        sllm_decode_control_commit_decision_v1, dim3(1U), dim3(1U), 0U, stream,
        control,
        static_cast<const sllm_token_selector_pq::DecisionRecordV1 *>(record),
        stop_ids, stop_count, vocabulary_size, result_ring);
  }
  return hipGetLastError();
}

hipError_t launch_gather_selector_token(const SelectorRecordV1 *const selection,
                                        std::int32_t *const output_token,
                                        const hipStream_t stream) noexcept {
  hipLaunchKernelGGL(sllm_decode_control_gather_selector_v1, dim3(1U), dim3(1U),
                     0U, stream, selection, output_token);
  return hipGetLastError();
}

hipError_t launch_gather_decision_tokens(
    const ControlV1 *const control,
    const sllm_token_selector_pq::DecisionRecordV1 *const decision,
    std::int32_t *const output_tokens, const std::uint32_t output_capacity,
    const hipStream_t stream) noexcept {
  hipLaunchKernelGGL(sllm_decode_control_gather_decision_v1, dim3(1U), dim3(1U),
                     0U, stream, control, decision, output_tokens,
                     output_capacity);
  return hipGetLastError();
}

hipError_t launch_gather_result_tokens(const ResultV1 *const result,
                                       std::int32_t *const output_tokens,
                                       const std::uint32_t output_capacity,
                                       const hipStream_t stream) noexcept {
  hipLaunchKernelGGL(sllm_decode_control_gather_result_v1, dim3(1U), dim3(1U),
                     0U, stream, result, output_tokens, output_capacity);
  return hipGetLastError();
}

hipError_t launch_gather_hidden(ControlV1 *const control,
                                const std::uint16_t *const hidden_rows,
                                const std::uint32_t row_count,
                                const std::uint32_t hidden_width,
                                std::uint16_t *const output_hidden,
                                const hipStream_t stream) noexcept {
  const std::uint32_t blocks = (hidden_width + 255U) / 256U;
  hipLaunchKernelGGL(sllm_decode_control_gather_hidden_v1,
                     dim3(blocks == 0U ? 1U : blocks), dim3(256U), 0U, stream,
                     control, hidden_rows, row_count, hidden_width,
                     output_hidden);
  return hipGetLastError();
}

hipError_t launch_gather_active_token(ControlV1 *const control,
                                      const std::int32_t *const draft_ids,
                                      const std::uint32_t draft_capacity,
                                      std::int32_t *const output_token,
                                      const hipStream_t stream) noexcept {
  hipLaunchKernelGGL(sllm_decode_control_gather_active_token_v1, dim3(1U),
                     dim3(1U), 0U, stream, control, draft_ids, draft_capacity,
                     output_token);
  return hipGetLastError();
}

hipError_t launch_gather_active_hidden(
    ControlV1 *const control, const std::uint16_t *const target_hidden_rows,
    const std::uint32_t target_row_count,
    const std::uint16_t *const previous_hidden,
    const std::uint32_t hidden_width, std::uint16_t *const output_hidden,
    const hipStream_t stream) noexcept {
  const std::uint32_t blocks = (hidden_width + 255U) / 256U;
  hipLaunchKernelGGL(sllm_decode_control_gather_active_hidden_v1,
                     dim3(blocks == 0U ? 1U : blocks), dim3(256U), 0U, stream,
                     control, target_hidden_rows, target_row_count,
                     previous_hidden, hidden_width, output_hidden);
  return hipGetLastError();
}

} // namespace sllm_decode_control

#endif
