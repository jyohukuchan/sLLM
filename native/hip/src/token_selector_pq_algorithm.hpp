#ifndef SLLM_TOKEN_SELECTOR_PQ_ALGORITHM_HPP
#define SLLM_TOKEN_SELECTOR_PQ_ALGORITHM_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace sllm_token_selector_pq {

constexpr std::uint32_t kSupportVersion = 1U;
constexpr std::uint32_t kDecisionVersion = 1U;
constexpr std::uint32_t kMaxSupport = 20U;
constexpr std::uint32_t kMaxWidth = 8U;
constexpr std::uint32_t kMaxEmitted = kMaxWidth + 1U;
constexpr std::uint32_t kNoRejection = UINT32_MAX;

// Fixed-size by-value ABI argument: the native wrapper never passes a host
// pointer to the device. Only the first `draft_id_count` entries are used.
struct DraftIdsV1 {
  std::uint32_t ids[kMaxWidth];
};

static_assert(sizeof(DraftIdsV1) == 32U,
              "draft id ABI must remain eight u32 values");

// This private record is deliberately separate from the public 16-byte
// TokenSelectorRecord. A producer must zero every inactive entry.
struct SupportRecordV1 {
  std::uint32_t version;
  std::uint32_t status;
  std::uint32_t count;
  std::uint32_t reserved;
  std::uint32_t ids[kMaxSupport];
  double probabilities[kMaxSupport];
};

enum class DecisionStatus : std::uint32_t {
  kOk = 0U,
  kInvalidWidth = 1U,
  kInvalidRowCount = 2U,
  kInvalidSupportVersion = 3U,
  kInputSupportStatus = 4U,
  kInvalidSupportCount = 5U,
  kSupportReservedNonzero = 6U,
  kDuplicateId = 7U,
  kInvalidProbability = 8U,
  kZeroOrInvalidMass = 9U,
  kUnusedEntryNonzero = 10U,
  kMissingDraftIdInQ = 11U,
  kZeroDraftProbabilityInQ = 12U,
  kEmptyResidual = 13U,
  kCounterOverflow = 14U,
  kNullPointer = 15U,
};

// The result contains at most width accepted/replacement tokens plus one
// bonus token. target_logprobs always refers to normalized target-p, including
// for a residual replacement; it never contains a residual log probability.
struct alignas(16) DecisionRecordV1 {
  std::uint32_t version;
  std::uint32_t status;
  std::uint32_t accepted_count;
  std::uint32_t emitted_count;
  std::uint32_t rejected_at;
  std::uint32_t width;
  std::uint32_t draws_used;
  std::uint32_t reserved0;
  std::uint32_t emitted_ids[kMaxEmitted];
  std::uint32_t reserved1;
  double target_logprobs[kMaxEmitted];
};

static_assert(std::is_standard_layout<SupportRecordV1>::value,
              "support record must remain standard-layout");
static_assert(sizeof(SupportRecordV1) == 256U,
              "support record ABI must remain exactly 256 bytes");
static_assert(alignof(SupportRecordV1) == alignof(double),
              "support record ABI must match the selector export alignment");
static_assert(offsetof(SupportRecordV1, ids) == 16U,
              "support ids ABI offset changed");
static_assert(offsetof(SupportRecordV1, probabilities) == 96U,
              "support probabilities ABI offset changed");

static_assert(std::is_standard_layout<DecisionRecordV1>::value,
              "decision record must remain standard-layout");
static_assert(sizeof(DecisionRecordV1) == 144U,
              "decision record ABI must remain exactly 144 bytes");
static_assert(alignof(DecisionRecordV1) == 16U,
              "decision record ABI must remain 16-byte aligned");
static_assert(offsetof(DecisionRecordV1, emitted_ids) == 32U,
              "decision ids ABI offset changed");
static_assert(offsetof(DecisionRecordV1, target_logprobs) == 72U,
              "decision log probabilities ABI offset changed");

} // namespace sllm_token_selector_pq

#include <cmath>
#include <cstdint>

#if defined(__HIPCC__)
#define SLLM_TOKEN_SELECTOR_PQ_HD __host__ __device__
#else
#define SLLM_TOKEN_SELECTOR_PQ_HD
#endif

namespace sllm_token_selector_pq {

// Stable private RNG contract. The four domains cannot alias even when the
// logical position and row are equal.
constexpr std::uint64_t kDomainQDraft = UINT64_C(0x5144524146540001);
constexpr std::uint64_t kDomainPAccept = UINT64_C(0x5041434345505401);
constexpr std::uint64_t kDomainPResidual = UINT64_C(0x5052455349440001);
constexpr std::uint64_t kDomainPBonus = UINT64_C(0x50424f4e55530001);
constexpr std::uint64_t kSplitMixGamma = UINT64_C(0x9e3779b97f4a7c15);
constexpr double kInvTwoTo53 = 1.0 / 9007199254740992.0;

SLLM_TOKEN_SELECTOR_PQ_HD inline std::uint64_t splitmix64(std::uint64_t value) {
  value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31U);
}

SLLM_TOKEN_SELECTOR_PQ_HD inline bool
checked_draw_counter(const std::uint64_t absolute_position,
                     const std::uint32_t row, std::uint64_t *const counter) {
  if ((counter == nullptr) || (row > kMaxWidth)) {
    return false;
  }
  constexpr std::uint64_t stride = static_cast<std::uint64_t>(kMaxWidth) + 1U;
  constexpr std::uint64_t maximum = UINT64_MAX;
  const std::uint64_t row64 = static_cast<std::uint64_t>(row);
  if (absolute_position > ((maximum - row64) / stride)) {
    return false;
  }
  const std::uint64_t result = (absolute_position * stride) + row64;
  // The selector-compatible state uses counter + 1, so UINT64_MAX is not a
  // valid logical counter even though unsigned wrap itself is defined.
  if (result == maximum) {
    return false;
  }
  *counter = result;
  return true;
}

SLLM_TOKEN_SELECTOR_PQ_HD inline bool
uniform_draw(const std::uint64_t seed, const std::uint64_t domain,
             const std::uint64_t absolute_position, const std::uint32_t row,
             double *const draw) {
  if (draw == nullptr) {
    return false;
  }
  std::uint64_t counter = 0U;
  if (!checked_draw_counter(absolute_position, row, &counter)) {
    return false;
  }
  const std::uint64_t state =
      (seed ^ domain) + ((counter + 1U) * kSplitMixGamma);
  *draw = static_cast<double>(splitmix64(state) >> 11U) * kInvTwoTo53;
  return true;
}

SLLM_TOKEN_SELECTOR_PQ_HD inline void
clear_decision(DecisionRecordV1 *const output) {
  if (output == nullptr) {
    return;
  }
  output->version = kDecisionVersion;
  output->status = static_cast<std::uint32_t>(DecisionStatus::kOk);
  output->accepted_count = 0U;
  output->emitted_count = 0U;
  output->rejected_at = kNoRejection;
  output->width = 0U;
  output->draws_used = 0U;
  output->reserved0 = 0U;
  for (std::uint32_t index = 0U; index < kMaxEmitted; ++index) {
    output->emitted_ids[index] = 0U;
    output->target_logprobs[index] = 0.0;
  }
  output->reserved1 = 0U;
}

SLLM_TOKEN_SELECTOR_PQ_HD inline DecisionStatus
fail_decision(DecisionRecordV1 *const output, const DecisionStatus status) {
  clear_decision(output);
  if (output != nullptr) {
    output->status = static_cast<std::uint32_t>(status);
  }
  return status;
}

SLLM_TOKEN_SELECTOR_PQ_HD inline bool finite_double(const double value) {
  return __builtin_isfinite(value) != 0;
}

SLLM_TOKEN_SELECTOR_PQ_HD inline DecisionStatus
validate_support(const SupportRecordV1 &support, double *const total_mass) {
  if (support.version != kSupportVersion) {
    return DecisionStatus::kInvalidSupportVersion;
  }
  if (support.status != 0U) {
    return DecisionStatus::kInputSupportStatus;
  }
  if ((support.count == 0U) || (support.count > kMaxSupport)) {
    return DecisionStatus::kInvalidSupportCount;
  }
  if (support.reserved != 0U) {
    return DecisionStatus::kSupportReservedNonzero;
  }

  double sum = 0.0;
  for (std::uint32_t index = 0U; index < support.count; ++index) {
    const double probability = support.probabilities[index];
    if ((!finite_double(probability)) || (probability < 0.0)) {
      return DecisionStatus::kInvalidProbability;
    }
    for (std::uint32_t prior = 0U; prior < index; ++prior) {
      if (support.ids[prior] == support.ids[index]) {
        return DecisionStatus::kDuplicateId;
      }
    }
    sum += probability;
    if (!finite_double(sum)) {
      return DecisionStatus::kZeroOrInvalidMass;
    }
  }
  if (!(sum > 0.0)) {
    return DecisionStatus::kZeroOrInvalidMass;
  }

  for (std::uint32_t index = support.count; index < kMaxSupport; ++index) {
    if ((support.ids[index] != 0U) || (support.probabilities[index] != 0.0)) {
      return DecisionStatus::kUnusedEntryNonzero;
    }
  }
  if (total_mass != nullptr) {
    *total_mass = sum;
  }
  return DecisionStatus::kOk;
}

SLLM_TOKEN_SELECTOR_PQ_HD inline bool
probability_for_id(const SupportRecordV1 &support, const double total_mass,
                   const std::uint32_t id, double *const probability,
                   bool *const found) {
  if ((probability == nullptr) || (found == nullptr) || !(total_mass > 0.0)) {
    return false;
  }
  *probability = 0.0;
  *found = false;
  for (std::uint32_t index = 0U; index < support.count; ++index) {
    if (support.ids[index] == id) {
      *probability = support.probabilities[index] / total_mass;
      *found = true;
      return true;
    }
  }
  return true;
}

SLLM_TOKEN_SELECTOR_PQ_HD inline bool
append_target_selection(DecisionRecordV1 *const output,
                        const SupportRecordV1 &target,
                        const double target_total, const std::uint32_t id) {
  if ((output == nullptr) || (output->emitted_count >= kMaxEmitted)) {
    return false;
  }
  double probability = 0.0;
  bool found = false;
  if ((!probability_for_id(target, target_total, id, &probability, &found)) ||
      (!found) || !(probability > 0.0)) {
    return false;
  }
  const std::uint32_t slot = output->emitted_count;
  output->emitted_ids[slot] = id;
  output->target_logprobs[slot] = ::log(probability);
  output->emitted_count = slot + 1U;
  return true;
}

SLLM_TOKEN_SELECTOR_PQ_HD inline bool
sample_support(const SupportRecordV1 &support, const double total_mass,
               const double draw, std::uint32_t *const selected_id) {
  if ((selected_id == nullptr) || !(total_mass > 0.0) || !(draw >= 0.0) ||
      !(draw < 1.0)) {
    return false;
  }
  const double target = draw * total_mass;
  double cumulative = 0.0;
  std::uint32_t last_positive_id = 0U;
  bool have_positive = false;
  for (std::uint32_t index = 0U; index < support.count; ++index) {
    const double probability = support.probabilities[index];
    if (probability > 0.0) {
      last_positive_id = support.ids[index];
      have_positive = true;
    }
    cumulative += probability;
    if ((probability > 0.0) && (target < cumulative)) {
      *selected_id = support.ids[index];
      return true;
    }
  }
  if (have_positive) {
    // Bounded summation roundoff can leave target infinitesimally above
    // the final cumulative sum; preserve total mass by choosing the final
    // positive entry.
    *selected_id = last_positive_id;
    return true;
  }
  return false;
}

SLLM_TOKEN_SELECTOR_PQ_HD inline bool
sample_residual(const SupportRecordV1 &target, const double target_total,
                const SupportRecordV1 &draft, const double draft_total,
                const double draw, std::uint32_t *const selected_id) {
  if ((selected_id == nullptr) || !(target_total > 0.0) ||
      !(draft_total > 0.0) || !(draw >= 0.0) || !(draw < 1.0)) {
    return false;
  }

  double residual_total = 0.0;
  std::uint32_t last_positive_id = 0U;
  bool have_positive = false;
  for (std::uint32_t index = 0U; index < target.count; ++index) {
    const double target_probability =
        target.probabilities[index] / target_total;
    double draft_probability = 0.0;
    bool found = false;
    if (!probability_for_id(draft, draft_total, target.ids[index],
                            &draft_probability, &found)) {
      return false;
    }
    (void)found;
    const double difference = target_probability - draft_probability;
    if (difference > 0.0) {
      residual_total += difference;
      if (!finite_double(residual_total)) {
        return false;
      }
      last_positive_id = target.ids[index];
      have_positive = true;
    }
  }
  if ((!have_positive) || !(residual_total > 0.0)) {
    return false;
  }

  const double target_value = draw * residual_total;
  double cumulative = 0.0;
  for (std::uint32_t index = 0U; index < target.count; ++index) {
    const double target_probability =
        target.probabilities[index] / target_total;
    double draft_probability = 0.0;
    bool found = false;
    if (!probability_for_id(draft, draft_total, target.ids[index],
                            &draft_probability, &found)) {
      return false;
    }
    (void)found;
    const double difference = target_probability - draft_probability;
    if (difference > 0.0) {
      cumulative += difference;
      if (target_value < cumulative) {
        *selected_id = target.ids[index];
        return true;
      }
    }
  }
  *selected_id = last_positive_id;
  return true;
}

SLLM_TOKEN_SELECTOR_PQ_HD inline DecisionStatus
run_sparse_pq(const SupportRecordV1 *const target_rows,
              const std::uint32_t target_row_count,
              const SupportRecordV1 *const draft_rows,
              const std::uint32_t draft_row_count,
              const std::uint32_t *const draft_ids,
              const std::uint32_t draft_id_count, const std::uint32_t width,
              const std::uint64_t seed, const std::uint64_t absolute_position,
              DecisionRecordV1 *const output) {
  if (output == nullptr) {
    return DecisionStatus::kNullPointer;
  }
  clear_decision(output);
  if ((width == 0U) || (width > kMaxWidth)) {
    return fail_decision(output, DecisionStatus::kInvalidWidth);
  }
  output->width = width;
  if ((target_rows == nullptr) || (draft_rows == nullptr) ||
      (draft_ids == nullptr)) {
    return fail_decision(output, DecisionStatus::kNullPointer);
  }
  if ((target_row_count != (width + 1U)) || (draft_row_count != width) ||
      (draft_id_count != width)) {
    return fail_decision(output, DecisionStatus::kInvalidRowCount);
  }

  // Validate the largest slot before consuming any input or RNG. Lower rows
  // are monotone and therefore safe once this checked affine counter is safe.
  std::uint64_t maximum_counter = 0U;
  if (!checked_draw_counter(absolute_position, width, &maximum_counter)) {
    return fail_decision(output, DecisionStatus::kCounterOverflow);
  }
  (void)maximum_counter;

  double target_totals[kMaxEmitted] = {};
  double draft_totals[kMaxWidth] = {};
  for (std::uint32_t row = 0U; row < (width + 1U); ++row) {
    const DecisionStatus status =
        validate_support(target_rows[row], &target_totals[row]);
    if (status != DecisionStatus::kOk) {
      return fail_decision(output, status);
    }
  }
  for (std::uint32_t row = 0U; row < width; ++row) {
    const DecisionStatus status =
        validate_support(draft_rows[row], &draft_totals[row]);
    if (status != DecisionStatus::kOk) {
      return fail_decision(output, status);
    }
    double draft_probability = 0.0;
    bool found = false;
    if (!probability_for_id(draft_rows[row], draft_totals[row], draft_ids[row],
                            &draft_probability, &found)) {
      return fail_decision(output, DecisionStatus::kZeroOrInvalidMass);
    }
    if (!found) {
      return fail_decision(output, DecisionStatus::kMissingDraftIdInQ);
    }
    if (!(draft_probability > 0.0)) {
      return fail_decision(output, DecisionStatus::kZeroDraftProbabilityInQ);
    }
  }

  output->width = width;
  for (std::uint32_t row = 0U; row < width; ++row) {
    double target_probability = 0.0;
    bool found_in_target = false;
    if (!probability_for_id(target_rows[row], target_totals[row],
                            draft_ids[row], &target_probability,
                            &found_in_target)) {
      return fail_decision(output, DecisionStatus::kZeroOrInvalidMass);
    }
    (void)found_in_target;
    double draft_probability = 0.0;
    bool found_in_draft = false;
    if (!probability_for_id(draft_rows[row], draft_totals[row], draft_ids[row],
                            &draft_probability, &found_in_draft)) {
      return fail_decision(output, DecisionStatus::kZeroOrInvalidMass);
    }
    (void)found_in_draft;

    double acceptance_threshold = 1.0;
    if (target_probability < draft_probability) {
      acceptance_threshold = target_probability / draft_probability;
    }
    double acceptance_draw = 0.0;
    if (!uniform_draw(seed, kDomainPAccept, absolute_position, row,
                      &acceptance_draw)) {
      return fail_decision(output, DecisionStatus::kCounterOverflow);
    }
    output->draws_used = row + 1U;
    if (acceptance_draw < acceptance_threshold) {
      if (!append_target_selection(output, target_rows[row], target_totals[row],
                                   draft_ids[row])) {
        return fail_decision(output, DecisionStatus::kZeroOrInvalidMass);
      }
      output->accepted_count = row + 1U;
      continue;
    }

    double residual_draw = 0.0;
    if (!uniform_draw(seed, kDomainPResidual, absolute_position, row,
                      &residual_draw)) {
      return fail_decision(output, DecisionStatus::kCounterOverflow);
    }
    std::uint32_t replacement_id = 0U;
    if (!sample_residual(target_rows[row], target_totals[row], draft_rows[row],
                         draft_totals[row], residual_draw, &replacement_id)) {
      return fail_decision(output, DecisionStatus::kEmptyResidual);
    }
    output->rejected_at = row;
    output->draws_used = row + 2U;
    if (!append_target_selection(output, target_rows[row], target_totals[row],
                                 replacement_id)) {
      return fail_decision(output, DecisionStatus::kZeroOrInvalidMass);
    }
    return DecisionStatus::kOk;
  }

  double bonus_draw = 0.0;
  if (!uniform_draw(seed, kDomainPBonus, absolute_position, width,
                    &bonus_draw)) {
    return fail_decision(output, DecisionStatus::kCounterOverflow);
  }
  std::uint32_t bonus_id = 0U;
  if (!sample_support(target_rows[width], target_totals[width], bonus_draw,
                      &bonus_id)) {
    return fail_decision(output, DecisionStatus::kZeroOrInvalidMass);
  }
  output->draws_used = width + 1U;
  if (!append_target_selection(output, target_rows[width], target_totals[width],
                               bonus_id)) {
    return fail_decision(output, DecisionStatus::kZeroOrInvalidMass);
  }
  return DecisionStatus::kOk;
}

} // namespace sllm_token_selector_pq

#undef SLLM_TOKEN_SELECTOR_PQ_HD

#endif // SLLM_TOKEN_SELECTOR_PQ_ALGORITHM_HPP
