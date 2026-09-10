#include "sllm/hip.h"
#include "token_selector_pq_algorithm.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "unknown"
#endif

extern "C" sllm_status_t sllm_token_selector_verify_fixed_k20_mtp_v1(
    const sllm_context_t *context, const sllm_queue_t *queue,
    const sllm_buffer_t *target_buffer, uint64_t target_offset,
    uint32_t target_rows, const sllm_buffer_t *draft_buffer,
    uint64_t draft_offset, uint32_t draft_rows,
    sllm_token_selector_pq::DraftIdsV1 draft_ids, uint32_t draft_id_count,
    uint32_t width, uint64_t seed, uint64_t absolute_position,
    const sllm_buffer_t *decision_buffer, uint64_t decision_offset,
    sllm_error_sink_t *error_sink) noexcept;

namespace {

using sllm_token_selector_pq::DecisionRecordV1;
using sllm_token_selector_pq::DecisionStatus;
using sllm_token_selector_pq::DraftIdsV1;
using sllm_token_selector_pq::SupportRecordV1;

constexpr std::size_t kOracleVocab = 64U;
constexpr uint64_t kTargetOffset = 8U;
constexpr uint64_t kDraftOffset = 24U;
constexpr uint64_t kDecisionOffset = 16U;
constexpr uint64_t kTargetBytes =
    sizeof(SupportRecordV1) * sllm_token_selector_pq::kMaxEmitted;
constexpr uint64_t kDraftBytes =
    sizeof(SupportRecordV1) * sllm_token_selector_pq::kMaxWidth;
constexpr uint64_t kDecisionBytes = sizeof(DecisionRecordV1);
constexpr double kLogprobTolerance = 5.0e-13;

struct TestContext final {
  uint32_t checks = 0U;
  uint32_t failures = 0U;

  void expect(const bool condition, const std::string &description) {
    ++checks;
    if (!condition) {
      ++failures;
      std::cerr << "FAIL: " << description << '\n';
    }
  }
};

struct Error final {
  char message[512]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};

  void reset() {
    std::memset(message, 0, sizeof(message));
    sink.message_length = 0U;
  }
};

bool expect_status(const sllm_status_t actual, const sllm_status_t expected,
                   const char *const operation, const Error &error) {
  if (actual == expected) {
    return true;
  }
  std::cerr << operation << " returned " << actual << ", expected " << expected
            << ": " << error.message << '\n';
  return false;
}

bool wait_release(sllm_completion_t **const completion,
                  const char *const operation) {
  if (completion == nullptr || *completion == nullptr) {
    std::cerr << operation << " returned no completion\n";
    return false;
  }
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.state = SLLM_COMPLETION_STATE_PENDING;
  const sllm_status_t wait_status =
      sllm_completion_wait(*completion, UINT32_MAX, &result, &error.sink);
  bool ok = expect_status(wait_status, SLLM_STATUS_OK, operation, error);
  if (ok && result.state != SLLM_COMPLETION_STATE_SUCCESS) {
    std::cerr << operation << " reached state " << result.state << '\n';
    ok = false;
  }
  error.reset();
  const sllm_status_t release_status =
      sllm_completion_release(completion, &error.sink);
  return expect_status(release_status, SLLM_STATUS_OK, "completion release",
                       error) &&
         ok;
}

bool upload(const sllm_queue_t *const queue, const sllm_buffer_t *const buffer,
            const uint64_t offset, const void *const source,
            const uint64_t bytes, const char *const operation) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = const_cast<void *>(source);
  transfer.buffer_offset_bytes = offset;
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect_status(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                          &error.sink),
                     SLLM_STATUS_OK, operation, error)) {
    return false;
  }
  return wait_release(&completion, operation);
}

bool download(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer, const uint64_t offset,
              void *const destination, const uint64_t bytes,
              const char *const operation) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.buffer_offset_bytes = offset;
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect_status(sllm_buffer_copy_d2h(queue, buffer, &transfer, &completion,
                                          &error.sink),
                     SLLM_STATUS_OK, operation, error)) {
    return false;
  }

  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.state = SLLM_COMPLETION_STATE_PENDING;
  bool ok = expect_status(
      sllm_completion_wait(completion, UINT32_MAX, &result, &error.sink),
      SLLM_STATUS_OK, operation, error);
  if (ok && result.state != SLLM_COMPLETION_STATE_SUCCESS) {
    std::cerr << operation << " reached state " << result.state << '\n';
    ok = false;
  }
  uint64_t written = 0U;
  if (ok) {
    error.reset();
    ok = expect_status(sllm_completion_read(completion, destination, bytes,
                                            &written, &error.sink),
                       SLLM_STATUS_OK, operation, error) &&
         written == bytes;
    if (written != bytes) {
      std::cerr << operation << " returned " << written << " bytes, expected "
                << bytes << '\n';
    }
  }
  error.reset();
  const sllm_status_t release_status =
      sllm_completion_release(&completion, &error.sink);
  return expect_status(release_status, SLLM_STATUS_OK, "completion release",
                       error) &&
         ok;
}

SupportRecordV1
make_support(const std::initializer_list<std::pair<uint32_t, double>> entries) {
  SupportRecordV1 support{};
  support.version = sllm_token_selector_pq::kSupportVersion;
  support.count = static_cast<uint32_t>(entries.size());
  uint32_t index = 0U;
  for (const auto &entry : entries) {
    if (index < sllm_token_selector_pq::kMaxSupport) {
      support.ids[index] = entry.first;
      support.probabilities[index] = entry.second;
    }
    ++index;
  }
  return support;
}

struct CaseData final {
  uint32_t width = 0U;
  std::array<SupportRecordV1, sllm_token_selector_pq::kMaxEmitted> target{};
  std::array<SupportRecordV1, sllm_token_selector_pq::kMaxWidth> draft{};
  std::array<uint32_t, sllm_token_selector_pq::kMaxWidth> draft_ids{};
};

CaseData repeat_case(const uint32_t width, const SupportRecordV1 &target,
                     const SupportRecordV1 &draft, const uint32_t draft_id,
                     const SupportRecordV1 &bonus) {
  CaseData result{};
  result.width = width;
  for (uint32_t row = 0U; row < width; ++row) {
    result.target[row] = target;
    result.draft[row] = draft;
    result.draft_ids[row] = draft_id;
  }
  result.target[width] = bonus;
  return result;
}

// This oracle does not call the production validation, lookup, sampling,
// counter, or RNG helpers. It expands each sparse input into a dense
// long-double distribution and retains target-record order only for inverse
// CDF selection.
struct DenseDistribution final {
  std::array<long double, kOracleVocab> mass{};
  std::array<uint32_t, sllm_token_selector_pq::kMaxSupport> order{};
  uint32_t count = 0U;
};

DenseDistribution oracle_expand(const SupportRecordV1 &support) {
  DenseDistribution dense{};
  long double total = 0.0L;
  for (uint32_t index = 0U; index < support.count; ++index) {
    total += static_cast<long double>(support.probabilities[index]);
  }
  dense.count = support.count;
  for (uint32_t index = 0U; index < support.count; ++index) {
    const uint32_t id = support.ids[index];
    dense.mass[static_cast<std::size_t>(id)] =
        static_cast<long double>(support.probabilities[index]) / total;
    dense.order[index] = id;
  }
  return dense;
}

uint64_t oracle_splitmix64(uint64_t value) {
  value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31U);
}

bool oracle_counter(const uint64_t position, const uint32_t row,
                    uint64_t *const counter) {
  if (counter == nullptr || row > 8U) {
    return false;
  }
  const uint64_t row64 = static_cast<uint64_t>(row);
  if (position > (UINT64_MAX - row64) / UINT64_C(9)) {
    return false;
  }
  const uint64_t value = position * UINT64_C(9) + row64;
  if (value == UINT64_MAX) {
    return false;
  }
  *counter = value;
  return true;
}

long double oracle_draw(const uint64_t seed, const uint64_t domain,
                        const uint64_t position, const uint32_t row) {
  uint64_t counter = 0U;
  if (!oracle_counter(position, row, &counter)) {
    return -1.0L;
  }
  const uint64_t state = (seed ^ domain) + ((counter + UINT64_C(1)) *
                                            UINT64_C(0x9e3779b97f4a7c15));
  const uint64_t bits = oracle_splitmix64(state) >> 11U;
  return static_cast<long double>(bits) / 9007199254740992.0L;
}

uint32_t oracle_sample(const DenseDistribution &distribution,
                       const long double draw) {
  long double cumulative = 0.0L;
  uint32_t last_positive = 0U;
  for (uint32_t index = 0U; index < distribution.count; ++index) {
    const uint32_t id = distribution.order[index];
    const long double probability = distribution.mass[id];
    if (probability > 0.0L) {
      last_positive = id;
    }
    cumulative += probability;
    if (probability > 0.0L && draw < cumulative) {
      return id;
    }
  }
  return last_positive;
}

uint32_t oracle_sample_residual(const DenseDistribution &target,
                                const DenseDistribution &draft,
                                const long double draw) {
  long double total = 0.0L;
  for (uint32_t index = 0U; index < target.count; ++index) {
    const uint32_t id = target.order[index];
    const long double difference = target.mass[id] - draft.mass[id];
    if (difference > 0.0L) {
      total += difference;
    }
  }
  const long double selected_mass = draw * total;
  long double cumulative = 0.0L;
  uint32_t last_positive = 0U;
  for (uint32_t index = 0U; index < target.count; ++index) {
    const uint32_t id = target.order[index];
    const long double difference = target.mass[id] - draft.mass[id];
    if (difference > 0.0L) {
      cumulative += difference;
      last_positive = id;
      if (selected_mass < cumulative) {
        return id;
      }
    }
  }
  return last_positive;
}

void oracle_append(DecisionRecordV1 *const output,
                   const DenseDistribution &target, const uint32_t id) {
  const uint32_t slot = output->emitted_count;
  output->emitted_ids[slot] = id;
  output->target_logprobs[slot] =
      static_cast<double>(std::log(target.mass[id]));
  output->emitted_count = slot + 1U;
}

DecisionRecordV1 oracle_run(const CaseData &data, const uint64_t seed,
                            const uint64_t position) {
  DecisionRecordV1 output{};
  output.version = sllm_token_selector_pq::kDecisionVersion;
  output.status = static_cast<uint32_t>(DecisionStatus::kOk);
  output.rejected_at = sllm_token_selector_pq::kNoRejection;
  output.width = data.width;

  std::array<DenseDistribution, sllm_token_selector_pq::kMaxEmitted> target{};
  std::array<DenseDistribution, sllm_token_selector_pq::kMaxWidth> draft{};
  for (uint32_t row = 0U; row < data.width + 1U; ++row) {
    target[row] = oracle_expand(data.target[row]);
  }
  for (uint32_t row = 0U; row < data.width; ++row) {
    draft[row] = oracle_expand(data.draft[row]);
  }

  constexpr uint64_t kAcceptDomain = UINT64_C(0x5041434345505401);
  constexpr uint64_t kResidualDomain = UINT64_C(0x5052455349440001);
  constexpr uint64_t kBonusDomain = UINT64_C(0x50424f4e55530001);
  for (uint32_t row = 0U; row < data.width; ++row) {
    const uint32_t id = data.draft_ids[row];
    const long double p = target[row].mass[id];
    const long double q = draft[row].mass[id];
    const long double threshold = p < q ? p / q : 1.0L;
    output.draws_used = row + 1U;
    if (oracle_draw(seed, kAcceptDomain, position, row) < threshold) {
      oracle_append(&output, target[row], id);
      output.accepted_count = row + 1U;
      continue;
    }
    const uint32_t replacement = oracle_sample_residual(
        target[row], draft[row],
        oracle_draw(seed, kResidualDomain, position, row));
    output.rejected_at = row;
    output.draws_used = row + 2U;
    oracle_append(&output, target[row], replacement);
    return output;
  }

  const uint32_t bonus =
      oracle_sample(target[data.width],
                    oracle_draw(seed, kBonusDomain, position, data.width));
  output.draws_used = data.width + 1U;
  oracle_append(&output, target[data.width], bonus);
  return output;
}

class Runtime final {
public:
  bool initialize() {
    sllm_context_create_info_t context_info{};
    context_info.struct_size = sizeof(context_info);
    context_info.abi_version = SLLM_HIP_ABI_VERSION;
    context_info.device_index = 0U;
    std::strncpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
                 sizeof(context_info.expected_gcn_arch_name) - 1U);
    Error error;
    if (!expect_status(
            sllm_context_create(&context_info, &context_, &error.sink),
            SLLM_STATUS_OK, "p/q context create", error)) {
      return false;
    }

    sllm_queue_create_info_t queue_info{};
    queue_info.struct_size = sizeof(queue_info);
    queue_info.abi_version = SLLM_HIP_ABI_VERSION;
    if (!expect_status(
            sllm_queue_create(context_, &queue_info, &queue_, &error.sink),
            SLLM_STATUS_OK, "p/q queue create", error)) {
      return false;
    }
    return create_buffer(kTargetOffset + kTargetBytes, &target_) &&
           create_buffer(kDraftOffset + kDraftBytes, &draft_) &&
           create_buffer(kDecisionOffset + kDecisionBytes + 16U, &decision_);
  }

  bool release(TestContext *const test) {
    bool ok = true;
    Error error;
    if (decision_ != nullptr) {
      error.reset();
      const bool released =
          expect_status(sllm_buffer_release(&decision_, &error.sink),
                        SLLM_STATUS_OK, "p/q decision release", error);
      test->expect(released && decision_ == nullptr,
                   "decision buffer lifetime cleanup");
      ok = released && ok;
    }
    if (draft_ != nullptr) {
      error.reset();
      const bool released =
          expect_status(sllm_buffer_release(&draft_, &error.sink),
                        SLLM_STATUS_OK, "p/q draft release", error);
      test->expect(released && draft_ == nullptr,
                   "draft buffer lifetime cleanup");
      ok = released && ok;
    }
    if (target_ != nullptr) {
      error.reset();
      const bool released =
          expect_status(sllm_buffer_release(&target_, &error.sink),
                        SLLM_STATUS_OK, "p/q target release", error);
      test->expect(released && target_ == nullptr,
                   "target buffer lifetime cleanup");
      ok = released && ok;
    }
    if (queue_ != nullptr) {
      error.reset();
      const bool released =
          expect_status(sllm_queue_release(&queue_, &error.sink),
                        SLLM_STATUS_OK, "p/q queue release", error);
      test->expect(released && queue_ == nullptr, "queue lifetime cleanup");
      ok = released && ok;
    }
    if (context_ != nullptr) {
      error.reset();
      const bool released =
          expect_status(sllm_context_release(&context_, &error.sink),
                        SLLM_STATUS_OK, "p/q context release", error);
      test->expect(released && context_ == nullptr, "context lifetime cleanup");
      ok = released && ok;
    }
    return ok;
  }

  bool stage(const CaseData &data) const {
    const std::array<uint8_t, sizeof(DecisionRecordV1)> poison = [] {
      std::array<uint8_t, sizeof(DecisionRecordV1)> bytes{};
      bytes.fill(UINT8_C(0xa5));
      return bytes;
    }();
    return upload(queue_, target_, kTargetOffset, data.target.data(),
                  kTargetBytes, "p/q target upload") &&
           upload(queue_, draft_, kDraftOffset, data.draft.data(), kDraftBytes,
                  "p/q draft upload") &&
           upload(queue_, decision_, kDecisionOffset, poison.data(),
                  kDecisionBytes, "p/q decision poison");
  }

  bool run(const CaseData &data, const uint64_t seed, const uint64_t position,
           DecisionRecordV1 *const output) const {
    if (output == nullptr || !stage(data)) {
      return false;
    }
    DraftIdsV1 draft_ids{};
    for (std::size_t index = 0U; index < data.draft_ids.size(); ++index) {
      draft_ids.ids[index] = data.draft_ids[index];
    }
    Error error;
    const sllm_status_t status = sllm_token_selector_verify_fixed_k20_mtp_v1(
        context_, queue_, target_, kTargetOffset, data.width + 1U, draft_,
        kDraftOffset, data.width, draft_ids, data.width, data.width, seed,
        position, decision_, kDecisionOffset, &error.sink);
    if (!expect_status(status, SLLM_STATUS_OK, "p/q production wrapper",
                       error)) {
      return false;
    }
    return download(queue_, decision_, kDecisionOffset, output, sizeof(*output),
                    "p/q decision download");
  }

  const sllm_context_t *context() const { return context_; }
  const sllm_queue_t *queue() const { return queue_; }
  const sllm_buffer_t *target() const { return target_; }
  const sllm_buffer_t *draft() const { return draft_; }
  const sllm_buffer_t *decision() const { return decision_; }

private:
  bool create_buffer(const uint64_t bytes, sllm_buffer_t **const output) const {
    sllm_buffer_create_info_t info{};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    info.size_bytes = bytes;
    info.alignment_bytes = 0U;
    Error error;
    return expect_status(
        sllm_buffer_create(context_, &info, output, &error.sink),
        SLLM_STATUS_OK, "p/q buffer create", error);
  }

  sllm_context_t *context_ = nullptr;
  sllm_queue_t *queue_ = nullptr;
  sllm_buffer_t *target_ = nullptr;
  sllm_buffer_t *draft_ = nullptr;
  sllm_buffer_t *decision_ = nullptr;
};

bool compare_valid(TestContext *const test, const Runtime &runtime,
                   const std::string &label, const CaseData &data,
                   const uint64_t seed, const uint64_t position,
                   DecisionRecordV1 *const observed = nullptr) {
  DecisionRecordV1 candidate{};
  if (!runtime.run(data, seed, position, &candidate)) {
    test->expect(false, label + ": production execution");
    return false;
  }
  const DecisionRecordV1 oracle = oracle_run(data, seed, position);
  test->expect(candidate.version == oracle.version, label + ": version");
  test->expect(candidate.status == oracle.status, label + ": status");
  test->expect(candidate.accepted_count == oracle.accepted_count,
               label + ": accepted count");
  test->expect(candidate.emitted_count == oracle.emitted_count,
               label + ": emitted count");
  test->expect(candidate.rejected_at == oracle.rejected_at,
               label + ": rejected row");
  test->expect(candidate.width == oracle.width, label + ": width");
  test->expect(candidate.draws_used == oracle.draws_used,
               label + ": draw count");
  test->expect(candidate.reserved0 == 0U && candidate.reserved1 == 0U,
               label + ": reserved output");
  for (uint32_t index = 0U; index < candidate.emitted_count; ++index) {
    test->expect(candidate.emitted_ids[index] == oracle.emitted_ids[index],
                 label + ": emitted id " + std::to_string(index));
    test->expect(std::fabs(candidate.target_logprobs[index] -
                           oracle.target_logprobs[index]) <= kLogprobTolerance,
                 label + ": target-p logprob " + std::to_string(index));
  }
  for (uint32_t index = candidate.emitted_count;
       index < sllm_token_selector_pq::kMaxEmitted; ++index) {
    test->expect(candidate.emitted_ids[index] == 0U &&
                     candidate.target_logprobs[index] == 0.0,
                 label + ": zero unused output " + std::to_string(index));
  }
  if (observed != nullptr) {
    *observed = candidate;
  }
  return true;
}

void expect_device_status(TestContext *const test, const Runtime &runtime,
                          const std::string &label, const CaseData &data,
                          const DecisionStatus expected,
                          const uint64_t position = 0U) {
  DecisionRecordV1 output{};
  if (!runtime.run(data, UINT64_C(0x12345678), position, &output)) {
    test->expect(false, label + ": production execution");
    return;
  }
  test->expect(output.version == sllm_token_selector_pq::kDecisionVersion,
               label + ": decision version");
  test->expect(output.status == static_cast<uint32_t>(expected),
               label + ": status");
  test->expect(output.accepted_count == 0U && output.emitted_count == 0U &&
                   output.draws_used == 0U,
               label + ": fail-closed counters");
  test->expect(output.rejected_at == sllm_token_selector_pq::kNoRejection &&
                   output.width == 0U && output.reserved0 == 0U &&
                   output.reserved1 == 0U,
               label + ": fail-closed metadata");
  for (uint32_t index = 0U; index < sllm_token_selector_pq::kMaxEmitted;
       ++index) {
    test->expect(output.emitted_ids[index] == 0U &&
                     output.target_logprobs[index] == 0.0,
                 label + ": fail-closed output " + std::to_string(index));
  }
}

void test_abi(TestContext *const test) {
  test->expect(sizeof(SupportRecordV1) == 256U, "support ABI size");
  test->expect(alignof(SupportRecordV1) == 8U, "support ABI alignment");
  test->expect(offsetof(SupportRecordV1, ids) == 16U, "support IDs ABI offset");
  test->expect(offsetof(SupportRecordV1, probabilities) == 96U,
               "support probabilities ABI offset");
  test->expect(sizeof(DraftIdsV1) == 32U, "draft IDs by-value ABI size");
  test->expect(sizeof(DecisionRecordV1) == 144U, "decision ABI size");
  test->expect(alignof(DecisionRecordV1) == 16U, "decision ABI alignment");
  test->expect(offsetof(DecisionRecordV1, emitted_ids) == 32U,
               "decision IDs ABI offset");
  test->expect(offsetof(DecisionRecordV1, target_logprobs) == 72U,
               "decision logprobs ABI offset");
}

void test_valid_cases(TestContext *const test, const Runtime &runtime) {
  const SupportRecordV1 equality = make_support({{1U, 0.25}, {2U, 0.75}});
  const SupportRecordV1 bonus = make_support({{6U, 0.2}, {7U, 0.3}, {8U, 0.5}});

  DecisionRecordV1 observed{};
  const CaseData width1 = repeat_case(1U, equality, equality, 1U, bonus);
  (void)compare_valid(test, runtime, "width1 equality", width1, 5U, 11U,
                      &observed);
  test->expect(observed.accepted_count == 1U && observed.emitted_count == 2U &&
                   observed.rejected_at == sllm_token_selector_pq::kNoRejection,
               "width1 all-accept plus bonus shape");

  CaseData different =
      repeat_case(2U, make_support({{1U, 0.30}, {2U, 0.70}}),
                  make_support({{1U, 0.60}, {2U, 0.40}}), 1U, bonus);
  different.target[1] = make_support({{1U, 0.80}, {2U, 0.20}});
  different.draft[1] = make_support({{1U, 0.40}, {2U, 0.60}});
  (void)compare_valid(test, runtime, "width2 different overlap", different, 17U,
                      3U);

  const CaseData disjoint =
      repeat_case(1U, make_support({{10U, 0.4}, {11U, 0.6}}),
                  make_support({{3U, 0.7}, {4U, 0.3}}), 3U, bonus);
  (void)compare_valid(test, runtime, "disjoint support", disjoint, 91U, 4U,
                      &observed);
  test->expect(
      observed.accepted_count == 0U && observed.rejected_at == 0U &&
          (observed.emitted_ids[0] == 10U || observed.emitted_ids[0] == 11U),
      "disjoint residual excludes q-only IDs");

  const CaseData partial =
      repeat_case(1U, make_support({{1U, 0.0}, {2U, 0.25}, {3U, 0.75}}),
                  make_support({{1U, 0.5}, {2U, 0.25}, {4U, 0.25}}), 1U, bonus);
  (void)compare_valid(test, runtime, "partial support", partial, 2U, 0U,
                      &observed);
  test->expect(observed.emitted_ids[0] == 3U &&
                   std::fabs(observed.target_logprobs[0] - std::log(0.75)) <
                       1.0e-15,
               "residual replacement reports target-p logprob");

  const CaseData first_reject =
      repeat_case(3U, make_support({{1U, 0.0}, {2U, 1.0}}),
                  make_support({{1U, 1.0}}), 1U, bonus);
  (void)compare_valid(test, runtime, "first reject", first_reject, 7U, 0U,
                      &observed);
  test->expect(observed.accepted_count == 0U && observed.rejected_at == 0U,
               "first rejection position");

  CaseData middle_reject = repeat_case(3U, equality, equality, 1U, bonus);
  middle_reject.target[1] = make_support({{1U, 0.0}, {2U, 1.0}});
  middle_reject.draft[1] = make_support({{1U, 1.0}});
  (void)compare_valid(test, runtime, "middle reject", middle_reject, 7U, 0U,
                      &observed);
  test->expect(observed.accepted_count == 1U && observed.rejected_at == 1U,
               "middle rejection position");

  CaseData last_reject = repeat_case(3U, equality, equality, 1U, bonus);
  last_reject.target[2] = make_support({{1U, 0.0}, {2U, 1.0}});
  last_reject.draft[2] = make_support({{1U, 1.0}});
  (void)compare_valid(test, runtime, "last reject", last_reject, 7U, 0U,
                      &observed);
  test->expect(observed.accepted_count == 2U && observed.rejected_at == 2U,
               "last rejection position");

  SupportRecordV1 full_k20{};
  full_k20.version = sllm_token_selector_pq::kSupportVersion;
  full_k20.count = sllm_token_selector_pq::kMaxSupport;
  for (uint32_t index = 0U; index < sllm_token_selector_pq::kMaxSupport;
       ++index) {
    full_k20.ids[index] = index + 20U;
    full_k20.probabilities[index] = static_cast<double>(index + 1U);
  }
  const CaseData width8 = repeat_case(8U, full_k20, full_k20, 20U, full_k20);
  (void)compare_valid(test, runtime, "width8 full K20", width8, 37U, 19U,
                      &observed);
  test->expect(observed.accepted_count == 8U && observed.emitted_count == 9U &&
                   observed.draws_used == 9U,
               "width8 all-accept counters");

  constexpr double kBoundaryDraw = 3927139806938913.0 / 9007199254740992.0;
  const CaseData boundary = repeat_case(
      1U, make_support({{1U, kBoundaryDraw}, {2U, 1.0 - kBoundaryDraw}}),
      make_support({{1U, 1.0}}), 1U, bonus);
  (void)compare_valid(test, runtime, "acceptance equality boundary", boundary,
                      0U, 0U, &observed);
  test->expect(observed.rejected_at == 0U,
               "acceptance equality uses strict less-than");

  const uint64_t last_width8_position =
      (UINT64_MAX - UINT64_C(8)) / UINT64_C(9);
  (void)compare_valid(test, runtime, "last valid width8 counter", width8, 123U,
                      last_width8_position);
}

void test_malformed_records(TestContext *const test, const Runtime &runtime) {
  const SupportRecordV1 valid = make_support({{1U, 0.4}, {2U, 0.6}});
  const SupportRecordV1 bonus = make_support({{3U, 1.0}});
  const CaseData base = repeat_case(1U, valid, valid, 1U, bonus);
  CaseData bad = base;

  bad.target[0].probabilities[0] = std::numeric_limits<double>::quiet_NaN();
  expect_device_status(test, runtime, "NaN probability", bad,
                       DecisionStatus::kInvalidProbability);
  bad = base;
  bad.target[0].probabilities[0] = std::numeric_limits<double>::infinity();
  expect_device_status(test, runtime, "infinite probability", bad,
                       DecisionStatus::kInvalidProbability);
  bad = base;
  bad.target[0].probabilities[0] = -0.25;
  expect_device_status(test, runtime, "negative probability", bad,
                       DecisionStatus::kInvalidProbability);
  bad = base;
  bad.draft[0].ids[1] = bad.draft[0].ids[0];
  expect_device_status(test, runtime, "duplicate ID", bad,
                       DecisionStatus::kDuplicateId);
  bad = base;
  bad.target[0] = make_support({{1U, 0.0}, {2U, 0.0}});
  expect_device_status(test, runtime, "zero mass", bad,
                       DecisionStatus::kZeroOrInvalidMass);
  bad = base;
  bad.draft[0] = make_support({{2U, 1.0}});
  expect_device_status(test, runtime, "missing q draft ID", bad,
                       DecisionStatus::kMissingDraftIdInQ);
  bad = base;
  bad.draft[0] = make_support({{1U, 0.0}, {2U, 1.0}});
  expect_device_status(test, runtime, "zero q draft probability", bad,
                       DecisionStatus::kZeroDraftProbabilityInQ);
  bad = base;
  bad.target[0].count = 0U;
  expect_device_status(test, runtime, "zero support count", bad,
                       DecisionStatus::kInvalidSupportCount);
  bad = base;
  bad.target[0].count = sllm_token_selector_pq::kMaxSupport + 1U;
  expect_device_status(test, runtime, "support count over K20", bad,
                       DecisionStatus::kInvalidSupportCount);
  bad = base;
  bad.target[0].version = 2U;
  expect_device_status(test, runtime, "bad support version", bad,
                       DecisionStatus::kInvalidSupportVersion);
  bad = base;
  bad.target[0].status = 77U;
  expect_device_status(test, runtime, "input support error", bad,
                       DecisionStatus::kInputSupportStatus);
  bad = base;
  bad.target[0].reserved = 1U;
  expect_device_status(test, runtime, "support reserved nonzero", bad,
                       DecisionStatus::kSupportReservedNonzero);
  bad = base;
  bad.target[0].ids[bad.target[0].count] = 9U;
  expect_device_status(test, runtime, "unused support ID nonzero", bad,
                       DecisionStatus::kUnusedEntryNonzero);
  bad = base;
  bad.target[0].probabilities[bad.target[0].count] = 0.125;
  expect_device_status(test, runtime, "unused support probability nonzero", bad,
                       DecisionStatus::kUnusedEntryNonzero);

  const CaseData width8 = repeat_case(8U, valid, valid, 1U, bonus);
  const uint64_t first_invalid_position =
      ((UINT64_MAX - UINT64_C(8)) / UINT64_C(9)) + UINT64_C(1);
  expect_device_status(test, runtime, "RNG counter overflow", width8,
                       DecisionStatus::kCounterOverflow,
                       first_invalid_position);
}

void expect_wrapper_status(TestContext *const test, const std::string &label,
                           const sllm_status_t expected,
                           const sllm_status_t actual, const Error &error) {
  test->expect(actual == expected,
               label + ": wrapper status " + std::to_string(actual));
  test->expect(error.sink.message_length != 0U,
               label + ": wrapper error message");
}

void test_wrapper_validation(TestContext *const test, const Runtime &runtime) {
  const SupportRecordV1 valid = make_support({{1U, 0.4}, {2U, 0.6}});
  const SupportRecordV1 bonus = make_support({{3U, 1.0}});
  const CaseData base = repeat_case(1U, valid, valid, 1U, bonus);
  if (!runtime.stage(base)) {
    test->expect(false, "stage wrapper validation fixture");
    return;
  }
  DraftIdsV1 ids{};
  ids.ids[0] = 1U;
  Error error;
  auto call = [&](const sllm_buffer_t *const target,
                  const uint64_t target_offset, const uint32_t target_rows,
                  const sllm_buffer_t *const draft, const uint64_t draft_offset,
                  const uint32_t draft_rows, const uint32_t id_count,
                  const uint32_t width, const sllm_buffer_t *const decision,
                  const uint64_t decision_offset) {
    error.reset();
    return sllm_token_selector_verify_fixed_k20_mtp_v1(
        runtime.context(), runtime.queue(), target, target_offset, target_rows,
        draft, draft_offset, draft_rows, ids, id_count, width, 7U, 0U, decision,
        decision_offset, &error.sink);
  };

  expect_wrapper_status(test, "zero width", SLLM_STATUS_INVALID_ARGUMENT,
                        call(runtime.target(), kTargetOffset, 1U,
                             runtime.draft(), kDraftOffset, 0U, 0U, 0U,
                             runtime.decision(), kDecisionOffset),
                        error);
  expect_wrapper_status(test, "width over eight", SLLM_STATUS_INVALID_ARGUMENT,
                        call(runtime.target(), kTargetOffset, 10U,
                             runtime.draft(), kDraftOffset, 9U, 9U, 9U,
                             runtime.decision(), kDecisionOffset),
                        error);
  expect_wrapper_status(
      test, "wrong target row count", SLLM_STATUS_INVALID_ARGUMENT,
      call(runtime.target(), kTargetOffset, 1U, runtime.draft(), kDraftOffset,
           1U, 1U, 1U, runtime.decision(), kDecisionOffset),
      error);
  expect_wrapper_status(
      test, "wrong draft row count", SLLM_STATUS_INVALID_ARGUMENT,
      call(runtime.target(), kTargetOffset, 2U, runtime.draft(), kDraftOffset,
           0U, 1U, 1U, runtime.decision(), kDecisionOffset),
      error);
  expect_wrapper_status(
      test, "wrong draft ID count", SLLM_STATUS_INVALID_ARGUMENT,
      call(runtime.target(), kTargetOffset, 2U, runtime.draft(), kDraftOffset,
           1U, 0U, 1U, runtime.decision(), kDecisionOffset),
      error);
  expect_wrapper_status(
      test, "misaligned target offset", SLLM_STATUS_MISALIGNED_OFFSET,
      call(runtime.target(), kTargetOffset + 4U, 2U, runtime.draft(),
           kDraftOffset, 1U, 1U, 1U, runtime.decision(), kDecisionOffset),
      error);
  expect_wrapper_status(
      test, "misaligned decision offset", SLLM_STATUS_MISALIGNED_OFFSET,
      call(runtime.target(), kTargetOffset, 2U, runtime.draft(), kDraftOffset,
           1U, 1U, 1U, runtime.decision(), kDecisionOffset + 8U),
      error);
  expect_wrapper_status(
      test, "target range out of bounds", SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
      call(runtime.target(), kTargetOffset + kTargetBytes, 2U, runtime.draft(),
           kDraftOffset, 1U, 1U, 1U, runtime.decision(), kDecisionOffset),
      error);
  expect_wrapper_status(
      test, "overlapping support ranges", SLLM_STATUS_ALIAS_OVERLAP,
      call(runtime.target(), kTargetOffset, 2U, runtime.target(), kTargetOffset,
           1U, 1U, 1U, runtime.decision(), kDecisionOffset),
      error);
  expect_wrapper_status(
      test, "null target handle", SLLM_STATUS_INVALID_ARGUMENT,
      call(nullptr, kTargetOffset, 2U, runtime.draft(), kDraftOffset, 1U, 1U,
           1U, runtime.decision(), kDecisionOffset),
      error);
}

} // namespace

int main() {
  TestContext test;
  test_abi(&test);
  Runtime runtime;
  if (!runtime.initialize()) {
    test.expect(false, "initialize production p/q runtime");
    (void)runtime.release(&test);
    return 2;
  }

  test_valid_cases(&test, runtime);
  test_malformed_records(&test, runtime);
  test_wrapper_validation(&test, runtime);
  (void)runtime.release(&test);

  if (test.failures != 0U) {
    std::cerr << "phase83.5 production p/q GPU test FAILED: " << test.failures
              << " / " << test.checks << " checks\n";
    return 1;
  }
  std::cout << "{\"state\":\"PASS\",\"target\":\"" << SLLM_TEST_EXPECTED_TARGET
            << "\",\"wrapper\":\"sllm_token_selector_verify_fixed_k20_mtp_v1\""
               ",\"widths\":[1,2,3,8],\"support_bytes\":256"
               ",\"decision_bytes\":144,\"d2h_bytes_per_case\":144"
               ",\"counter\":\"position*9+row\",\"valid_cases\":10"
               ",\"device_error_cases\":15,\"wrapper_error_cases\":10"
               ",\"oracle\":\"independent_dense_long_double\""
               ",\"fallback_allowed\":0,\"fallback_used\":0}\n";
  return 0;
}
