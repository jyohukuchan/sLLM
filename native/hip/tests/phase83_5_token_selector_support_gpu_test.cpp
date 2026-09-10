#include "token_selector_kernel_internal.hpp"
#include "token_selector_support_internal.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kTopK = 20U;
constexpr float kTopP = 0.95F;
constexpr uint64_t kSeed = UINT64_C(0x123456789abcdef0);
constexpr uint64_t kCounter = 7U;

struct Expected final {
  sllm_token_selector_record_t record{};
  uint32_t status = SLLM_STATUS_OK;
  std::vector<uint32_t> ids;
  std::vector<double> probabilities;
};

struct Fixture final {
  const char *name;
  uint64_t vocab;
  uint32_t flags;
  bool ties;
  bool all_masked;
  bool nonfinite;
};

void check(const hipError_t error, const char *const what) {
  if (error != hipSuccess) {
    std::cerr << what << ": " << hipGetErrorString(error) << '\n';
    std::exit(1);
  }
}

uint16_t float_to_bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += ((bits >> 16U) & 1U) + UINT32_C(0x7fff);
  return static_cast<uint16_t>(bits >> 16U);
}

float bf16_to_float(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint64_t splitmix64(uint64_t value) {
  value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31U);
}

Expected make_expected(const std::vector<uint16_t> &logits,
                       const std::vector<float> &additive,
                       const std::vector<uint8_t> &mask, const uint32_t flags) {
  Expected result;
  result.record.token_id = -1;
  result.record.status = SLLM_STATUS_OK;
  result.record.logprob = -INFINITY;
  result.record.reserved0 = 0U;
  std::vector<std::pair<float, uint32_t>> candidates;
  for (size_t index = 0U; index != logits.size(); ++index) {
    if ((flags & SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT) != 0U &&
        mask[index] == 0U) {
      continue;
    }
    const float score =
        bf16_to_float(logits[index]) +
        ((flags & SLLM_HIP_TOKEN_SELECTOR_FLAG_ADDITIVE_PRESENT) != 0U
             ? additive[index]
             : 0.0F);
    if (!std::isfinite(score)) {
      result.status = SLLM_STATUS_TOKEN_SELECTOR_NONFINITE;
      result.record.status = result.status;
      return result;
    }
    candidates.emplace_back(score, static_cast<uint32_t>(index));
  }
  if (candidates.empty()) {
    result.status = SLLM_STATUS_TOKEN_SELECTOR_ALL_MASKED;
    result.record.status = result.status;
    return result;
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const auto &left, const auto &right) {
              return left.first > right.first ||
                     (left.first == right.first && left.second < right.second);
            });
  const size_t count = std::min(static_cast<size_t>(kTopK), candidates.size());
  const float maximum = candidates.front().first;
  double sum = 0.0;
  for (size_t index = 0U; index != count; ++index) {
    sum += std::exp(static_cast<double>(candidates[index].first - maximum));
  }
  const double cutoff = static_cast<double>(kTopP) * sum;
  size_t included = 0U;
  double cumulative = 0.0;
  for (; included != count; ++included) {
    cumulative +=
        std::exp(static_cast<double>(candidates[included].first - maximum));
    if (cumulative >= cutoff) {
      ++included;
      break;
    }
  }
  included = included == 0U ? 1U : included;
  for (size_t index = 0U; index != included; ++index) {
    const double weight =
        std::exp(static_cast<double>(candidates[index].first - maximum));
    result.ids.push_back(candidates[index].second);
    result.probabilities.push_back(weight / cumulative);
  }
  const uint64_t gamma = UINT64_C(0x9e3779b97f4a7c15);
  const uint64_t random_bits = splitmix64(kSeed + (kCounter + 1U) * gamma);
  const double target = static_cast<double>(random_bits >> 11U) *
                        (1.0 / 9007199254740992.0) * cumulative;
  double running = 0.0;
  size_t selected = included - 1U;
  for (size_t index = 0U; index != included; ++index) {
    running += std::exp(static_cast<double>(candidates[index].first - maximum));
    if (target < running) {
      selected = index;
      break;
    }
  }
  const double selected_weight =
      std::exp(static_cast<double>(candidates[selected].first - maximum));
  result.record.token_id = static_cast<int32_t>(candidates[selected].second);
  result.record.logprob =
      static_cast<float>(std::log(selected_weight / cumulative));
  return result;
}

void make_fixture(const Fixture &fixture, std::vector<uint16_t> *const logits,
                  std::vector<float> *const additive,
                  std::vector<uint8_t> *const mask) {
  logits->resize(static_cast<size_t>(fixture.vocab));
  additive->resize(static_cast<size_t>(fixture.vocab));
  mask->assign(static_cast<size_t>(fixture.vocab), 1U);
  for (uint64_t index = 0U; index != fixture.vocab; ++index) {
    const uint32_t bucket = static_cast<uint32_t>(
        (index * UINT64_C(0x9e3779b97f4a7c15) + UINT64_C(0x243f6a88)) % 37U);
    const float base =
        fixture.ties ? 1.0F : (static_cast<float>(bucket) - 18.0F) * 0.125F;
    (*logits)[static_cast<size_t>(index)] = float_to_bf16(base);
    (*additive)[static_cast<size_t>(index)] =
        fixture.ties ? 0.0F
                     : (static_cast<float>(index % 11U) - 5.0F) * 0.015625F;
    if ((fixture.flags & SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT) != 0U) {
      (*mask)[static_cast<size_t>(index)] =
          (index % 7U == 1U || index % 13U == 5U) ? 0U : 1U;
    }
  }
  if (fixture.all_masked) {
    std::fill(mask->begin(), mask->end(), 0U);
  }
  if (fixture.nonfinite && fixture.vocab > 17U) {
    (*logits)[17U] = UINT16_C(0x7fc0);
  }
}

bool verify_device_target() {
  int device_count = 0;
  check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
  if (device_count != 1) {
    std::cerr << "expected exactly one visible GPU, got " << device_count
              << '\n';
    return false;
  }
  hipDeviceProp_t properties{};
  check(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
  if (std::strcmp(properties.gcnArchName, SLLM_TEST_EXPECTED_TARGET) != 0) {
    std::cerr << "target mismatch expected=" << SLLM_TEST_EXPECTED_TARGET
              << " actual=" << properties.gcnArchName << '\n';
    return false;
  }
  return true;
}

bool run_fixture(const Fixture &fixture) {
  std::vector<uint16_t> logits;
  std::vector<float> additive;
  std::vector<uint8_t> mask;
  make_fixture(fixture, &logits, &additive, &mask);
  const Expected expected =
      make_expected(logits, additive, mask, fixture.flags);
  const uint64_t block_count = (fixture.vocab + 1023U) / 1024U;
  const size_t workspace_bytes = static_cast<size_t>(
      2U * block_count * static_cast<uint64_t>(kTopK) * UINT64_C(8));
  hipStream_t stream = nullptr;
  uint16_t *device_logits = nullptr;
  float *device_additive = nullptr;
  uint8_t *device_mask = nullptr;
  uint8_t *device_workspace = nullptr;
  sllm_token_selector_record_t *device_record = nullptr;
  check(hipStreamCreate(&stream), "hipStreamCreate");
  check(hipMalloc(reinterpret_cast<void **>(&device_logits),
                  logits.size() * sizeof(uint16_t)),
        "hipMalloc logits");
  check(hipMalloc(reinterpret_cast<void **>(&device_additive),
                  additive.size() * sizeof(float)),
        "hipMalloc additive");
  check(hipMalloc(reinterpret_cast<void **>(&device_mask),
                  mask.size() * sizeof(uint8_t)),
        "hipMalloc mask");
  check(
      hipMalloc(reinterpret_cast<void **>(&device_workspace), workspace_bytes),
      "hipMalloc workspace");
  check(hipMalloc(reinterpret_cast<void **>(&device_record),
                  sizeof(*device_record)),
        "hipMalloc record");
  check(hipMemcpyAsync(device_logits, logits.data(),
                       logits.size() * sizeof(uint16_t), hipMemcpyHostToDevice,
                       stream),
        "copy logits");
  check(hipMemcpyAsync(device_additive, additive.data(),
                       additive.size() * sizeof(float), hipMemcpyHostToDevice,
                       stream),
        "copy additive");
  check(hipMemcpyAsync(device_mask, mask.data(), mask.size() * sizeof(uint8_t),
                       hipMemcpyHostToDevice, stream),
        "copy mask");
  check(sllm_token_selector_kernel::launch(
            device_logits, device_additive, device_mask, device_workspace,
            fixture.vocab, 1.0F, kTopK, kTopP, fixture.flags, kSeed, kCounter,
            device_record, stream),
        "selector launch");
  check(hipStreamSynchronize(stream), "selector synchronize");
  sllm_token_selector_record_t record_first{};
  std::vector<uint8_t> support_first(sllm_token_selector_support::kBytesV1);
  check(hipMemcpy(&record_first, device_record, sizeof(record_first),
                  hipMemcpyDeviceToHost),
        "copy first record");
  check(hipMemcpy(support_first.data(), device_workspace, support_first.size(),
                  hipMemcpyDeviceToHost),
        "copy first support");
  check(sllm_token_selector_kernel::launch(
            device_logits, device_additive, device_mask, device_workspace,
            fixture.vocab, 1.0F, kTopK, kTopP, fixture.flags, kSeed, kCounter,
            device_record, stream),
        "selector repeat");
  check(hipStreamSynchronize(stream), "selector repeat synchronize");
  sllm_token_selector_record_t record_repeat{};
  std::vector<uint8_t> support_repeat(sllm_token_selector_support::kBytesV1);
  check(hipMemcpy(&record_repeat, device_record, sizeof(record_repeat),
                  hipMemcpyDeviceToHost),
        "copy repeat record");
  check(hipMemcpy(support_repeat.data(), device_workspace,
                  support_repeat.size(), hipMemcpyDeviceToHost),
        "copy repeat support");
  sllm_token_selector_support::RecordV1 support{};
  std::memcpy(&support, support_first.data(), sizeof(support));
  bool ok =
      std::memcmp(&record_first, &record_repeat, sizeof(record_first)) == 0 &&
      support_first == support_repeat &&
      record_first.status == expected.status &&
      support.status == expected.status && support.version == 1U &&
      support.count == expected.ids.size() && support.reserved == 0U;
  if (expected.status == SLLM_STATUS_OK) {
    ok = ok && record_first.token_id == expected.record.token_id &&
         std::abs(record_first.logprob - expected.record.logprob) <= 2.0e-5F;
  }
  for (size_t index = 0U; index != sllm_token_selector_support::kMaxCountV1;
       ++index) {
    const uint32_t expected_id =
        index < expected.ids.size() ? expected.ids[index] : 0U;
    const double expected_probability = index < expected.probabilities.size()
                                            ? expected.probabilities[index]
                                            : 0.0;
    ok = ok && support.ids[index] == expected_id &&
         std::abs(support.probabilities[index] - expected_probability) <=
             2.0e-11;
  }
  std::cout << "case=" << fixture.name << " vocab=" << fixture.vocab
            << " status=" << record_first.status
            << " support_count=" << support.count << " public_record_repeat="
            << (std::memcmp(&record_first, &record_repeat,
                            sizeof(record_first)) == 0
                    ? 1
                    : 0)
            << " result=" << (ok ? "PASS" : "FAIL") << '\n';
  check(hipFree(device_record), "hipFree record");
  check(hipFree(device_workspace), "hipFree workspace");
  check(hipFree(device_mask), "hipFree mask");
  check(hipFree(device_additive), "hipFree additive");
  check(hipFree(device_logits), "hipFree logits");
  check(hipStreamDestroy(stream), "hipStreamDestroy");
  return ok;
}

} // namespace

int main() {
  if (!verify_device_target()) {
    return 1;
  }
  const std::vector<Fixture> fixtures = {
      {"baseline_1023", 1023U, 0U, false, false, false},
      {"mask_1024", 1024U, SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT, false,
       false, false},
      {"additive_1025", 1025U, SLLM_HIP_TOKEN_SELECTOR_FLAG_ADDITIVE_PRESENT,
       false, false, false},
      {"both_ties_2049", 2049U,
       SLLM_HIP_TOKEN_SELECTOR_FLAG_ADDITIVE_PRESENT |
           SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT,
       true, false, false},
      {"all_masked_1025", 1025U, SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT,
       false, true, false},
      {"nonfinite_2049", 2049U,
       SLLM_HIP_TOKEN_SELECTOR_FLAG_ADDITIVE_PRESENT |
           SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT,
       false, false, true},
  };
  bool ok = true;
  for (const Fixture &fixture : fixtures) {
    ok = run_fixture(fixture) && ok;
  }
  std::cout << "summary cases=" << fixtures.size()
            << " result=" << (ok ? "PASS" : "FAIL") << '\n';
  return ok ? 0 : 1;
}
