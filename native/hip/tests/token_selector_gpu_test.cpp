#include "sllm/hip.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx942"
#endif

namespace {

static_assert(sizeof(sllm_token_selector_record_t) ==
                  SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES,
              "the selected record must remain the fixed 16-byte D2H ABI");

constexpr float kLogprobTolerance = 5.0e-3F;

struct Error final {
  char message[512]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};
};

bool expect(const sllm_status_t actual, const sllm_status_t expected,
            const char *const operation, const Error &error) {
  if (actual == expected) {
    return true;
  }
  std::cerr << operation << " returned " << actual << ", expected " << expected
            << ": " << error.message << '\n';
  return false;
}

sllm_tensor_binding_t binding(const sllm_buffer_t *const buffer,
                              const uint32_t dtype, const uint32_t rank,
                              const uint64_t first,
                              const uint64_t second = 0U) {
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.dtype = dtype;
  result.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  result.rank = rank;
  result.shape[0] = first;
  result.stride_elements[0] = rank == 2U ? second : 1U;
  if (rank == 2U) {
    result.shape[1] = second;
    result.stride_elements[1] = 1U;
  }
  return result;
}

bool wait_release(sllm_completion_t **const completion,
                  const char *const label) {
  Error error;
  sllm_completion_result_t result{sizeof(result),
                                  SLLM_HIP_ABI_VERSION,
                                  SLLM_COMPLETION_STATE_PENDING,
                                  0U,
                                  0U,
                                  0U,
                                  {0U, 0U, 0U, 0U}};
  const bool waited = expect(
      sllm_completion_wait(*completion, UINT32_MAX, &result, &error.sink),
      SLLM_STATUS_OK, label, error);
  if (!waited || result.state != SLLM_COMPLETION_STATE_SUCCESS) {
    std::cerr << label
              << " did not reach terminal success (state=" << result.state
              << ")\n";
    return false;
  }
  return expect(sllm_completion_release(completion, &error.sink),
                SLLM_STATUS_OK, "completion release", error);
}

bool upload(const sllm_queue_t *const queue, const sllm_buffer_t *const buffer,
            const void *const data, const uint64_t bytes) {
  sllm_transfer_desc_t transfer{sizeof(transfer),
                                SLLM_HIP_ABI_VERSION,
                                const_cast<void *>(data),
                                0U,
                                bytes,
                                {0U, 0U, 0U, 0U}};
  sllm_completion_t *completion = nullptr;
  Error error;
  return expect(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                     &error.sink),
                SLLM_STATUS_OK, "buffer h2d", error) &&
         wait_release(&completion, "h2d completion");
}

bool download_record(const sllm_queue_t *const queue,
                     const sllm_buffer_t *const buffer,
                     sllm_token_selector_record_t *const output) {
  // This helper intentionally transfers exactly the fixed selected record;
  // no full-vocabulary D2H path is allowed in this correctness test.
  std::vector<uint8_t> bytes(SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES);
  sllm_transfer_desc_t transfer{sizeof(transfer), SLLM_HIP_ABI_VERSION,
                                nullptr,          0U,
                                bytes.size(),     {0U, 0U, 0U, 0U}};
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_buffer_copy_d2h(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, "record d2h", error)) {
    return false;
  }
  sllm_completion_result_t result{sizeof(result),
                                  SLLM_HIP_ABI_VERSION,
                                  SLLM_COMPLETION_STATE_PENDING,
                                  0U,
                                  0U,
                                  0U,
                                  {0U, 0U, 0U, 0U}};
  if (!expect(
          sllm_completion_wait(completion, UINT32_MAX, &result, &error.sink),
          SLLM_STATUS_OK, "record d2h completion", error)) {
    (void)sllm_completion_release(&completion, &error.sink);
    return false;
  }
  uint64_t written = 0U;
  const bool read_ok =
      expect(sllm_completion_read(completion, bytes.data(), bytes.size(),
                                  &written, &error.sink),
             SLLM_STATUS_OK, "record d2h read", error) &&
      written == SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES;
  const bool release_ok =
      expect(sllm_completion_release(&completion, &error.sink), SLLM_STATUS_OK,
             "record d2h release", error);
  if (read_ok && release_ok) {
    std::memcpy(output, bytes.data(), sizeof(*output));
  }
  return read_ok && release_ok;
}

uint16_t float_to_bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
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

struct Oracle final {
  int32_t token_id{-1};
  uint32_t status{SLLM_STATUS_TOKEN_SELECTOR_NONFINITE};
  float logprob{-INFINITY};
};

Oracle cpu_oracle(const std::vector<uint16_t> &logits,
                  const std::vector<float> &additive,
                  const std::vector<uint8_t> &mask, const float temperature,
                  const uint64_t seed, const uint64_t counter) {
  Oracle output;
  float maximum = -INFINITY;
  bool has_candidate = false;
  for (std::size_t index = 0U; index != logits.size(); ++index) {
    if (mask[index] == 0U) {
      continue;
    }
    const float value = bf16_to_float(logits[index]) + additive[index];
    if (!std::isfinite(value)) {
      output.status = SLLM_STATUS_TOKEN_SELECTOR_NONFINITE;
      return output;
    }
    if (!has_candidate || value > maximum) {
      maximum = value;
      has_candidate = true;
    }
  }
  if (!has_candidate) {
    output.status = SLLM_STATUS_TOKEN_SELECTOR_ALL_MASKED;
    return output;
  }
  double sum = 0.0;
  for (std::size_t index = 0U; index != logits.size(); ++index) {
    if (mask[index] != 0U) {
      sum += std::exp(static_cast<double>(bf16_to_float(logits[index]) +
                                          additive[index] - maximum) /
                      static_cast<double>(temperature));
    }
  }
  if (!(sum > 0.0F) || !std::isfinite(sum)) {
    output.status = SLLM_STATUS_TOKEN_SELECTOR_NONFINITE;
    return output;
  }
  const uint64_t gamma = UINT64_C(0x9e3779b97f4a7c15);
  const uint64_t draw_state = seed + (counter + UINT64_C(1)) * gamma;
  const uint64_t random_bits = splitmix64(draw_state);
  const double unit =
      static_cast<double>(random_bits >> 11U) * (1.0 / 9007199254740992.0);
  const double target = unit * sum;
  double cumulative = 0.0;
  std::size_t selected = logits.size() - 1U;
  double selected_probability = 0.0;
  std::vector<std::size_t> order;
  order.reserve(logits.size());
  for (std::size_t index = 0U; index != logits.size(); ++index) {
    if (mask[index] == 0U) {
      continue;
    }
    order.push_back(index);
  }
  std::sort(order.begin(), order.end(),
            [&](const std::size_t left, const std::size_t right) {
              const float left_value =
                  bf16_to_float(logits[left]) + additive[left];
              const float right_value =
                  bf16_to_float(logits[right]) + additive[right];
              if (left_value != right_value) {
                return left_value > right_value;
              }
              return left < right;
            });
  for (const std::size_t index : order) {
    const double probability =
        std::exp(static_cast<double>(bf16_to_float(logits[index]) +
                                     additive[index] - maximum) /
                 static_cast<double>(temperature));
    cumulative += probability;
    if (target < cumulative) {
      selected = index;
      selected_probability = probability / sum;
      break;
    }
  }
  if (selected_probability == 0.0) {
    selected_probability =
        std::exp(static_cast<double>(bf16_to_float(logits[selected]) +
                                     additive[selected] - maximum) /
                 static_cast<double>(temperature)) /
        sum;
  }
  output.token_id = static_cast<int32_t>(selected);
  output.status = SLLM_STATUS_OK;
  output.logprob = static_cast<float>(std::log(selected_probability));
  return output;
}

Oracle fixed_topk_oracle(const std::vector<uint16_t> &logits,
                         const std::vector<float> &additive,
                         const std::vector<uint8_t> &mask, const uint32_t top_k,
                         const float top_p, const uint64_t seed,
                         const uint64_t counter) {
  Oracle output;
  std::vector<std::pair<float, std::size_t>> candidates;
  for (std::size_t index = 0U; index != logits.size(); ++index) {
    if (mask[index] == 0U) {
      continue;
    }
    const float value = bf16_to_float(logits[index]) + additive[index];
    if (!std::isfinite(value)) {
      output.status = SLLM_STATUS_TOKEN_SELECTOR_NONFINITE;
      return output;
    }
    candidates.emplace_back(value, index);
  }
  if (candidates.empty()) {
    output.status = SLLM_STATUS_TOKEN_SELECTOR_ALL_MASKED;
    return output;
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const auto &left, const auto &right) {
              return left.first != right.first ? left.first > right.first
                                               : left.second < right.second;
            });
  const std::size_t count = std::min<std::size_t>(top_k, candidates.size());
  const double maximum = candidates.front().first;
  double sum = 0.0;
  for (std::size_t index = 0U; index != count; ++index) {
    sum += std::exp(static_cast<double>(candidates[index].first - maximum));
  }
  const double cutoff = static_cast<double>(top_p) * sum;
  std::size_t included = 0U;
  double cumulative = 0.0;
  for (; included != count; ++included) {
    cumulative +=
        std::exp(static_cast<double>(candidates[included].first - maximum));
    if (cumulative >= cutoff) {
      ++included;
      break;
    }
  }
  included = std::max<std::size_t>(1U, included);
  const uint64_t gamma = UINT64_C(0x9e3779b97f4a7c15);
  const uint64_t random_bits = splitmix64(seed + (counter + 1U) * gamma);
  const double target = static_cast<double>(random_bits >> 11U) *
                        (1.0 / 9007199254740992.0) * cumulative;
  double running = 0.0;
  std::size_t selected = included - 1U;
  for (std::size_t index = 0U; index != included; ++index) {
    running += std::exp(static_cast<double>(candidates[index].first - maximum));
    if (target < running) {
      selected = index;
      break;
    }
  }
  const double selected_weight =
      std::exp(static_cast<double>(candidates[selected].first - maximum));
  output.token_id = static_cast<int32_t>(candidates[selected].second);
  output.status = SLLM_STATUS_OK;
  output.logprob = static_cast<float>(std::log(selected_weight / cumulative));
  return output;
}

// Independent oracle for the K=0 fixed profile.  It retains the smallest
// token-ID prefix of the boundary score bucket needed to reach top-p; this is
// the tie rule exercised by the GPU histogram path.
Oracle fixed_topp_oracle(const std::vector<uint16_t> &logits,
                         const std::vector<float> &additive,
                         const std::vector<uint8_t> &mask, const float top_p,
                         const uint64_t seed, const uint64_t counter) {
  Oracle output;
  std::vector<std::pair<float, std::size_t>> candidates;
  for (std::size_t index = 0U; index != logits.size(); ++index) {
    if (mask[index] == 0U) {
      continue;
    }
    const float value = bf16_to_float(logits[index]) + additive[index];
    if (!std::isfinite(value)) {
      output.status = SLLM_STATUS_TOKEN_SELECTOR_NONFINITE;
      return output;
    }
    candidates.emplace_back(value, index);
  }
  if (candidates.empty()) {
    output.status = SLLM_STATUS_TOKEN_SELECTOR_ALL_MASKED;
    return output;
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const auto &left, const auto &right) {
              return left.first != right.first ? left.first > right.first
                                               : left.second < right.second;
            });
  const double maximum = candidates.front().first;
  double total = 0.0;
  for (const auto &candidate : candidates) {
    total += std::exp(static_cast<double>(candidate.first - maximum));
  }
  const double nucleus_target = static_cast<double>(top_p) * total;
  std::size_t included = 0U;
  double cumulative = 0.0;
  double sample_total = 0.0;
  while (included != candidates.size()) {
    const float score = candidates[included].first;
    std::size_t end = included + 1U;
    while (end != candidates.size() && candidates[end].first == score) {
      ++end;
    }
    const double weight = std::exp(static_cast<double>(score - maximum));
    const std::size_t bucket_count = end - included;
    const double bucket = static_cast<double>(bucket_count) * weight;
    if (cumulative + bucket >= nucleus_target) {
      const double remaining = nucleus_target - cumulative;
      const std::size_t needed = std::min<std::size_t>(
          bucket_count,
          std::max<std::size_t>(
              1U, static_cast<std::size_t>(std::ceil(remaining / weight))));
      included += needed;
      sample_total = cumulative + static_cast<double>(needed) * weight;
      break;
    }
    cumulative += bucket;
    included = end;
  }
  if (included == 0U) {
    included = 1U;
    sample_total =
        std::exp(static_cast<double>(candidates.front().first - maximum));
  }
  const uint64_t gamma = UINT64_C(0x9e3779b97f4a7c15);
  const uint64_t random_bits = splitmix64(seed + (counter + 1U) * gamma);
  const double unit =
      static_cast<double>(random_bits >> 11U) * (1.0 / 9007199254740992.0);
  const double target = unit * sample_total;
  double running = 0.0;
  std::size_t selected = included - 1U;
  for (std::size_t index = 0U; index != included; ++index) {
    const double weight =
        std::exp(static_cast<double>(candidates[index].first - maximum));
    running += weight;
    if (target < running) {
      selected = index;
      break;
    }
  }
  const double selected_weight =
      std::exp(static_cast<double>(candidates[selected].first - maximum));
  output.token_id = static_cast<int32_t>(candidates[selected].second);
  output.status = SLLM_STATUS_OK;
  output.logprob = static_cast<float>(std::log(selected_weight / sample_total));
  return output;
}

void fill_inputs(const uint64_t vocab, std::vector<uint16_t> *const logits,
                 std::vector<float> *const additive,
                 std::vector<uint8_t> *const mask) {
  logits->resize(vocab);
  additive->resize(vocab);
  mask->resize(vocab);
  for (uint64_t index = 0U; index != vocab; ++index) {
    // All values are exactly representable in BF16 and keep the oracle/device
    // comparison focused on selection and softmax rather than conversion.
    const int32_t centered = static_cast<int32_t>(index % 17U) - 8;
    const float base = static_cast<float>(centered) * 0.0625F;
    (*logits)[index] = float_to_bf16(base);
    (*additive)[index] = static_cast<float>(centered) * 0.03125F;
    (*mask)[index] =
        (vocab == 1U || ((index % 7U) != 1U && (index % 13U) != 5U)) ? 1U : 0U;
  }
  if (vocab == 248320U) {
    // Broad nonuniform BF16 fixture.  The integer mixing is also used by the
    // NumPy oracle fixture: it creates many score buckets across the full
    // model vocabulary while retaining a deterministic masked subset.
    for (uint64_t index = 0U; index != vocab; ++index) {
      const uint64_t mixed = (index * UINT64_C(0x9e3779b97f4a7c15)) ^
                             ((index + UINT64_C(0x243f6a8885a308d3)) >> 17U);
      const uint32_t bucket = static_cast<uint32_t>((mixed >> 11U) % 193U);
      (*logits)[index] =
          float_to_bf16(-12.0F + static_cast<float>(bucket) * 0.0625F);
      (*additive)[index] = 0.0F;
      (*mask)[index] = (mixed % 17U) != 0U ? 1U : 0U;
    }
  }
}

bool validate_info(const sllm_token_selector_dispatch_info_t &info,
                   const uint64_t vocab) {
  return info.backend == SLLM_BACKEND_HIP && info.dispatch_count == 1U &&
         info.kernel_id == SLLM_HIP_TOKEN_SELECTOR_KERNEL_ID_BF16_F32_MASK_V1 &&
         info.workgroup_size_x == SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE &&
         info.grid_size_x == 1U && info.vocab_size == vocab &&
         info.fallback_allowed == 0U && info.fallback_used == 0U &&
         std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;
}

bool validate_fixed_info(const sllm_token_selector_dispatch_info_t &info,
                         const uint64_t vocab, const uint32_t top_k) {
  uint64_t records = ((vocab + 1023U) / 1024U) * top_k;
  uint32_t launches = 2U;
  while (records > 1024U) {
    records = ((records + 1023U) / 1024U) * top_k;
    ++launches;
  }
  return info.backend == SLLM_BACKEND_HIP && info.dispatch_count == launches &&
         info.kernel_id ==
             SLLM_HIP_TOKEN_SELECTOR_KERNEL_ID_FIXED_TOPK_TOPP_V1 &&
         info.workgroup_size_x == SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE &&
         info.grid_size_x == (vocab + 1023U) / 1024U &&
         info.vocab_size == vocab && info.fallback_allowed == 0U &&
         info.fallback_used == 0U &&
         std::strcmp(info.kernel_symbol, "token_selector.fixed_topk_topp.v1") ==
             0 &&
         std::strcmp(info.device_symbol,
                     "sllm_token_selector_fixed_topk_topp_v1") == 0 &&
         std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;
}

bool validate_fixed_topp_info(const sllm_token_selector_dispatch_info_t &info,
                              const uint64_t vocab) {
  return info.backend == SLLM_BACKEND_HIP && info.dispatch_count == 8U &&
         info.kernel_id ==
             SLLM_HIP_TOKEN_SELECTOR_KERNEL_ID_FIXED_TOPK_TOPP_V1 &&
         info.workgroup_size_x == SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE &&
         info.grid_size_x == (vocab + 1023U) / 1024U &&
         info.vocab_size == vocab && info.fallback_allowed == 0U &&
         info.fallback_used == 0U &&
         std::strcmp(info.kernel_symbol, "token_selector.fixed_topk_topp.v1") ==
             0 &&
         std::strcmp(info.device_symbol,
                     "sllm_token_selector_fixed_topk_topp_v1") == 0 &&
         std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;
}

bool execute_and_compare(
    const sllm_context_t *const context, const sllm_queue_t *const queue,
    const uint64_t vocab, const float temperature, const uint64_t seed,
    const uint64_t counter, const std::vector<uint16_t> &host_logits,
    const std::vector<float> &host_additive,
    const std::vector<uint8_t> &host_mask, sllm_buffer_t *const logits,
    sllm_buffer_t *const additive, sllm_buffer_t *const mask,
    sllm_buffer_t *const output) {
  Error error;
  sllm_token_selector_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_TOKEN_SELECTOR_VERSION;
  descriptor.logits = binding(logits, SLLM_TENSOR_DTYPE_BF16, 2U, 1U, vocab);
  descriptor.additive_logits =
      binding(additive, SLLM_TENSOR_DTYPE_F32, 2U, 1U, vocab);
  descriptor.valid_mask = binding(mask, SLLM_TENSOR_DTYPE_U8, 2U, 1U, vocab);
  descriptor.output = binding(output, SLLM_TENSOR_DTYPE_U8, 1U,
                              SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES);
  descriptor.vocab_size = vocab;
  descriptor.temperature = temperature;
  descriptor.seed = seed;
  descriptor.counter = counter;
  sllm_token_selector_plan_t *plan = nullptr;
  bool ok = expect(
      sllm_token_selector_prepare(context, &descriptor, &plan, &error.sink),
      SLLM_STATUS_OK, "selector prepare", error);
  sllm_token_selector_record_t first{};
  sllm_token_selector_record_t second{};
  sllm_token_selector_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_TOKEN_SELECTOR_DISPATCH_INFO_VERSION;
  for (sllm_token_selector_record_t *const record : {&first, &second}) {
    sllm_completion_t *completion = nullptr;
    ok = ok &&
         expect(sllm_token_selector_execute(plan, queue, &completion, &info,
                                            &error.sink),
                SLLM_STATUS_OK, "selector execute", error) &&
         wait_release(&completion, "selector completion") &&
         download_record(queue, output, record);
    if (!ok) {
      break;
    }
  }
  const Oracle oracle = cpu_oracle(host_logits, host_additive, host_mask,
                                   temperature, seed, counter);
  ok = ok && validate_info(info, vocab) && first.token_id == oracle.token_id &&
       first.status == oracle.status &&
       (oracle.status != SLLM_STATUS_OK ||
        (std::isfinite(first.logprob) &&
         std::fabs(first.logprob - oracle.logprob) <= kLogprobTolerance)) &&
       std::memcmp(&first, &second, sizeof(first)) == 0;
  if (!ok) {
    std::cerr << "selector oracle mismatch vocab=" << vocab
              << " counter=" << counter << " gpu_token=" << first.token_id
              << " oracle_token=" << oracle.token_id
              << " gpu_status=" << first.status
              << " oracle_status=" << oracle.status
              << " gpu_logprob=" << first.logprob
              << " oracle_logprob=" << oracle.logprob << '\n';
  }
  if (plan != nullptr) {
    ok = expect(sllm_token_selector_plan_release(&plan, &error.sink),
                SLLM_STATUS_OK, "selector plan release", error) &&
         ok;
  }
  return ok;
}

bool execute_status_case(const sllm_context_t *const context,
                         const sllm_queue_t *const queue, const uint64_t vocab,
                         const float temperature, const uint64_t seed,
                         const uint64_t counter, sllm_buffer_t *const logits,
                         sllm_buffer_t *const additive,
                         sllm_buffer_t *const mask, sllm_buffer_t *const output,
                         const uint32_t expected) {
  Error error;
  sllm_token_selector_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_TOKEN_SELECTOR_VERSION;
  descriptor.logits = binding(logits, SLLM_TENSOR_DTYPE_BF16, 2U, 1U, vocab);
  descriptor.additive_logits =
      binding(additive, SLLM_TENSOR_DTYPE_F32, 2U, 1U, vocab);
  descriptor.valid_mask = binding(mask, SLLM_TENSOR_DTYPE_U8, 2U, 1U, vocab);
  descriptor.output = binding(output, SLLM_TENSOR_DTYPE_U8, 1U,
                              SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES);
  descriptor.vocab_size = vocab;
  descriptor.temperature = temperature;
  descriptor.seed = seed;
  descriptor.counter = counter;
  sllm_token_selector_plan_t *plan = nullptr;
  bool ok = expect(
      sllm_token_selector_prepare(context, &descriptor, &plan, &error.sink),
      SLLM_STATUS_OK, "status selector prepare", error);
  sllm_token_selector_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_TOKEN_SELECTOR_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  ok = ok &&
       expect(sllm_token_selector_execute(plan, queue, &completion, &info,
                                          &error.sink),
              SLLM_STATUS_OK, "status selector execute", error) &&
       wait_release(&completion, "status selector completion");
  sllm_token_selector_record_t record{};
  ok = ok && download_record(queue, output, &record) &&
       validate_info(info, vocab) && record.status == expected &&
       record.token_id == -1 && record.reserved0 == 0U &&
       std::isinf(record.logprob) && record.logprob < 0.0F;
  if (!ok) {
    std::cerr << "legacy status mismatch vocab=" << vocab
              << " expected=" << expected << " got=" << record.status
              << " token=" << record.token_id << " logprob=" << record.logprob
              << " dispatch=" << info.dispatch_count << '\n';
  }
  if (plan != nullptr) {
    ok = expect(sllm_token_selector_plan_release(&plan, &error.sink),
                SLLM_STATUS_OK, "status selector plan release", error) &&
         ok;
  }
  return ok;
}

bool execute_fixed_topp_case(
    const sllm_context_t *const context, const sllm_queue_t *const queue,
    const uint64_t vocab, const uint64_t seed, const uint64_t counter,
    const std::vector<uint16_t> &host_logits,
    const std::vector<float> &host_additive,
    const std::vector<uint8_t> &host_mask, sllm_buffer_t *const logits,
    sllm_buffer_t *const additive, sllm_buffer_t *const mask,
    sllm_buffer_t *const output, sllm_buffer_t *const workspace,
    const int32_t independent_expected_token = -1, const bool use_mask = true) {
  Error error;
  sllm_token_selector_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_TOKEN_SELECTOR_VERSION_FIXED_TOPK_TOPP;
  descriptor.logits = binding(logits, SLLM_TENSOR_DTYPE_BF16, 2U, 1U, vocab);
  descriptor.additive_logits =
      binding(additive, SLLM_TENSOR_DTYPE_F32, 2U, 1U, vocab);
  descriptor.valid_mask = binding(mask, SLLM_TENSOR_DTYPE_U8, 2U, 1U, vocab);
  descriptor.output = binding(output, SLLM_TENSOR_DTYPE_U8, 1U,
                              SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES);
  descriptor.vocab_size = vocab;
  descriptor.temperature = 1.0F;
  descriptor.seed = seed;
  descriptor.counter = counter;
  descriptor.top_k = 0U;
  descriptor.flags = use_mask ? SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT : 0U;
  descriptor.top_p = 0.95F;
  descriptor.workspace = binding(workspace, SLLM_TENSOR_DTYPE_U8, 1U,
                                 SLLM_HIP_TOKEN_SELECTOR_K0_WORKSPACE_BYTES);
  sllm_token_selector_plan_t *plan = nullptr;
  bool ok = expect(
      sllm_token_selector_prepare(context, &descriptor, &plan, &error.sink),
      SLLM_STATUS_OK, "fixed top-p prepare", error);
  sllm_token_selector_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_TOKEN_SELECTOR_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  ok = ok &&
       expect(sllm_token_selector_execute(plan, queue, &completion, &info,
                                          &error.sink),
              SLLM_STATUS_OK, "fixed top-p execute", error) &&
       wait_release(&completion, "fixed top-p completion");
  sllm_token_selector_record_t record{};
  ok = ok && download_record(queue, output, &record);
  const std::vector<uint8_t> oracle_mask =
      use_mask ? host_mask : std::vector<uint8_t>(host_mask.size(), 1U);
  const Oracle oracle = fixed_topp_oracle(host_logits, host_additive,
                                          oracle_mask, 0.95F, seed, counter);
  ok = ok && validate_fixed_topp_info(info, vocab) &&
       record.token_id == oracle.token_id && record.status == oracle.status &&
       (oracle.status != SLLM_STATUS_OK ||
        std::fabs(record.logprob - oracle.logprob) <= kLogprobTolerance) &&
       (independent_expected_token < 0 ||
        record.token_id == independent_expected_token);
  if (!ok) {
    std::cerr << "fixed top-p oracle mismatch vocab=" << vocab
              << " seed=" << seed << " counter=" << counter
              << " gpu_token=" << record.token_id
              << " oracle_token=" << oracle.token_id
              << " gpu_status=" << record.status
              << " oracle_status=" << oracle.status
              << " gpu_logprob=" << record.logprob
              << " oracle_logprob=" << oracle.logprob << '\n';
  }
  if (plan != nullptr) {
    ok = expect(sllm_token_selector_plan_release(&plan, &error.sink),
                SLLM_STATUS_OK, "fixed top-p plan release", error) &&
         ok;
  }
  return ok;
}

bool execute_fixed_status_case(
    const sllm_context_t *const context, const sllm_queue_t *const queue,
    const uint64_t vocab, const uint64_t seed, const uint64_t counter,
    sllm_buffer_t *const logits, sllm_buffer_t *const additive,
    sllm_buffer_t *const mask, sllm_buffer_t *const output,
    sllm_buffer_t *const workspace, const uint32_t top_k, const uint32_t flags,
    const uint64_t workspace_bytes, const uint32_t expected) {
  Error error;
  sllm_token_selector_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_TOKEN_SELECTOR_VERSION_FIXED_TOPK_TOPP;
  descriptor.logits = binding(logits, SLLM_TENSOR_DTYPE_BF16, 2U, 1U, vocab);
  descriptor.additive_logits =
      binding(additive, SLLM_TENSOR_DTYPE_F32, 2U, 1U, vocab);
  descriptor.valid_mask = binding(mask, SLLM_TENSOR_DTYPE_U8, 2U, 1U, vocab);
  descriptor.output = binding(output, SLLM_TENSOR_DTYPE_U8, 1U,
                              SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES);
  descriptor.vocab_size = vocab;
  descriptor.temperature = 1.0F;
  descriptor.seed = seed;
  descriptor.counter = counter;
  descriptor.top_k = top_k;
  descriptor.flags = flags;
  descriptor.top_p = 0.95F;
  descriptor.workspace =
      binding(workspace, SLLM_TENSOR_DTYPE_U8, 1U, workspace_bytes);
  sllm_token_selector_plan_t *plan = nullptr;
  bool ok = expect(
      sllm_token_selector_prepare(context, &descriptor, &plan, &error.sink),
      SLLM_STATUS_OK, "fixed status prepare", error);
  sllm_token_selector_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_TOKEN_SELECTOR_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  ok = ok &&
       expect(sllm_token_selector_execute(plan, queue, &completion, &info,
                                          &error.sink),
              SLLM_STATUS_OK, "fixed status execute", error) &&
       wait_release(&completion, "fixed status completion");
  sllm_token_selector_record_t record{};
  ok = ok && download_record(queue, output, &record) &&
       (top_k == 0U ? validate_fixed_topp_info(info, vocab)
                    : validate_fixed_info(info, vocab, top_k)) &&
       record.status == expected && record.token_id == -1 &&
       record.reserved0 == 0U && std::isinf(record.logprob) &&
       record.logprob < 0.0F;
  if (!ok) {
    std::cerr << "fixed status mismatch vocab=" << vocab << " k=" << top_k
              << " expected=" << expected << " got=" << record.status
              << " token=" << record.token_id << " logprob=" << record.logprob
              << " dispatch=" << info.dispatch_count << '\n';
  }
  if (plan != nullptr) {
    ok = expect(sllm_token_selector_plan_release(&plan, &error.sink),
                SLLM_STATUS_OK, "fixed status plan release", error) &&
         ok;
  }
  return ok;
}

bool run_case(const sllm_context_t *const context,
              const sllm_queue_t *const queue, const uint64_t vocab,
              const uint64_t seed) {
  std::cerr << "token-selector vocab=" << vocab << " start\n";
  const float temperature = 0.7F;
  std::vector<uint16_t> host_logits;
  std::vector<float> host_additive;
  std::vector<uint8_t> host_mask;
  if (vocab == 3U && seed == 0U) {
    // Regression fixture for legacy descending-logit categorical order:
    // logits [0,1,2], draw(seed=0,counter=0)=0.8833108 selects token 1,
    // whereas token-ID ascending accumulation would select token 2.
    host_logits = {float_to_bf16(0.0F), float_to_bf16(1.0F),
                   float_to_bf16(2.0F)};
    host_additive.assign(3U, 0.0F);
    host_mask.assign(3U, 1U);
    if (cpu_oracle(host_logits, host_additive, host_mask, temperature, seed, 0U)
            .token_id != 1) {
      std::cerr
          << "legacy categorical order fixture oracle did not select token 1\n";
      return false;
    }
  } else {
    if (vocab == 25U) {
      // Fixture emitted by token_selector_fixed_oracle.py (NumPy, with the
      // same BF16 truncation and seed/counter contract).
      host_logits = {16448, 16448, 16416, 16416, 16384, 16320, 16256,
                     16128, 0,     48896, 49024, 49024, 49088, 49152,
                     49152, 49184, 49216, 49216, 49216, 49248, 49280,
                     49296, 49312, 49328, 49344};
      host_additive = {0.0F,  0.0F,  0.1F,  -0.1F, 0.05F, 0.2F,  0.0F,
                       0.15F, 0.0F,  0.1F,  0.0F,  0.05F, 0.0F,  0.1F,
                       0.0F,  0.05F, 0.0F,  0.1F,  0.0F,  0.05F, 0.0F,
                       0.1F,  0.0F,  0.05F, 0.0F};
      host_mask.resize(vocab);
      for (uint64_t index = 0U; index != vocab; ++index) {
        host_mask[index] = (index % 7U) != 1U && (index % 11U) != 4U;
      }
    } else {
      fill_inputs(vocab, &host_logits, &host_additive, &host_mask);
    }
    if (vocab == 1023U || vocab == 1024U || vocab == 1025U) {
      // Boundary fixture: put equal high logits in the final tile so a
      // missing lane in the 256-thread bitonic stages cannot pass unnoticed.
      for (uint64_t index = 0U; index != vocab; ++index) {
        host_logits[index] = float_to_bf16(-8.0F);
        host_additive[index] = 0.0F;
        host_mask[index] = 1U;
      }
      for (uint64_t index = vocab - 20U; index != vocab; ++index) {
        host_logits[index] = float_to_bf16(4.0F);
      }
    } else if (vocab == 100U) {
      // Independent K=0 boundary fixture: every valid token has the same
      // BF16 score, so top-p=.95 must retain IDs [0, 95).
      std::fill(host_logits.begin(), host_logits.end(), float_to_bf16(0.0F));
      std::fill(host_additive.begin(), host_additive.end(), 0.0F);
      std::fill(host_mask.begin(), host_mask.end(), 1U);
    }
  }
  Error error;
  auto create = [&](const uint64_t bytes, sllm_buffer_t **const out) {
    sllm_buffer_create_info_t info{sizeof(info), SLLM_HIP_ABI_VERSION,
                                   bytes,        0U,
                                   0U,           {0U, 0U, 0U, 0U, 0U}};
    return expect(sllm_buffer_create(context, &info, out, &error.sink),
                  SLLM_STATUS_OK, "buffer create", error);
  };
  sllm_buffer_t *logits = nullptr;
  sllm_buffer_t *additive = nullptr;
  sllm_buffer_t *mask = nullptr;
  sllm_buffer_t *output = nullptr;
  sllm_buffer_t *workspace = nullptr;
  const uint64_t workspace_bytes =
      std::max<uint64_t>(SLLM_HIP_TOKEN_SELECTOR_K0_WORKSPACE_BYTES,
                         2U * ((vocab + 1023U) / 1024U) * 64U * 8U);
  bool ok = create(host_logits.size() * sizeof(uint16_t), &logits) &&
            create(host_additive.size() * sizeof(float), &additive) &&
            create(host_mask.size(), &mask) &&
            create(SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES, &output) &&
            create(workspace_bytes, &workspace) &&
            upload(queue, logits, host_logits.data(),
                   host_logits.size() * sizeof(uint16_t)) &&
            upload(queue, additive, host_additive.data(),
                   host_additive.size() * sizeof(float)) &&
            upload(queue, mask, host_mask.data(), host_mask.size());
  ok = ok && execute_and_compare(context, queue, vocab, temperature, seed, 0U,
                                 host_logits, host_additive, host_mask, logits,
                                 additive, mask, output);
  ok = ok && execute_and_compare(context, queue, vocab, temperature, seed, 1U,
                                 host_logits, host_additive, host_mask, logits,
                                 additive, mask, output);

  // Fixed v2 candidate reduction: the host oracle uses the same NumPy-style
  // stable ordering (score descending, token ID ascending), while the GPU
  // path must use the persistent two-region workspace and launch reductions.
  const uint32_t fixed_variant_count = vocab == 25U ? 3U : 1U;
  for (uint32_t fixed_variant = 0U; fixed_variant != fixed_variant_count;
       ++fixed_variant) {
    const bool fixed_use_additive = fixed_variant != 2U;
    const bool fixed_use_mask = fixed_variant == 0U;
    const uint32_t fixed_flags =
        (fixed_use_additive ? SLLM_HIP_TOKEN_SELECTOR_FLAG_ADDITIVE_PRESENT
                            : 0U) |
        (fixed_use_mask ? SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT : 0U);
    const std::vector<float> fixed_additive =
        fixed_use_additive ? host_additive
                           : std::vector<float>(host_additive.size(), 0.0F);
    const std::vector<uint8_t> fixed_mask =
        fixed_use_mask ? host_mask : std::vector<uint8_t>(host_mask.size(), 1U);
    for (const uint32_t fixed_top_k : {20U, 64U}) {
      Error fixed_error;
      const uint64_t fixed_workspace_bytes =
          2U * ((vocab + 1023U) / 1024U) * fixed_top_k * 8U;
      for (const uint64_t fixed_counter : {0U, 1U, 3U}) {
        sllm_token_selector_desc_t descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.abi_version = SLLM_HIP_ABI_VERSION;
        descriptor.op_version = SLLM_HIP_TOKEN_SELECTOR_VERSION_FIXED_TOPK_TOPP;
        descriptor.logits =
            binding(logits, SLLM_TENSOR_DTYPE_BF16, 2U, 1U, vocab);
        descriptor.additive_logits =
            binding(additive, SLLM_TENSOR_DTYPE_F32, 2U, 1U, vocab);
        descriptor.valid_mask =
            binding(mask, SLLM_TENSOR_DTYPE_U8, 2U, 1U, vocab);
        descriptor.output = binding(output, SLLM_TENSOR_DTYPE_U8, 1U,
                                    SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES);
        descriptor.vocab_size = vocab;
        descriptor.temperature = 1.0F;
        descriptor.seed = seed;
        descriptor.counter = fixed_counter;
        descriptor.top_k = fixed_top_k;
        descriptor.flags = fixed_flags;
        descriptor.top_p = 0.95F;
        descriptor.workspace =
            binding(workspace, SLLM_TENSOR_DTYPE_U8, 1U, fixed_workspace_bytes);
        sllm_token_selector_plan_t *fixed_plan = nullptr;
        sllm_token_selector_dispatch_info_t fixed_info{};
        fixed_info.struct_size = sizeof(fixed_info);
        fixed_info.abi_version = SLLM_HIP_ABI_VERSION;
        fixed_info.info_version = SLLM_HIP_TOKEN_SELECTOR_DISPATCH_INFO_VERSION;
        sllm_completion_t *fixed_completion = nullptr;
        sllm_token_selector_record_t fixed_record{};
        bool fixed_case_ok =
            expect(sllm_token_selector_prepare(context, &descriptor,
                                               &fixed_plan, &fixed_error.sink),
                   SLLM_STATUS_OK, "fixed selector prepare", fixed_error) &&
            expect(sllm_token_selector_execute(fixed_plan, queue,
                                               &fixed_completion, &fixed_info,
                                               &fixed_error.sink),
                   SLLM_STATUS_OK, "fixed selector execute", fixed_error) &&
            wait_release(&fixed_completion, "fixed selector completion") &&
            download_record(queue, output, &fixed_record);
        const Oracle fixed_oracle =
            fixed_topk_oracle(host_logits, fixed_additive, fixed_mask,
                              fixed_top_k, 0.95F, seed, fixed_counter);
        fixed_case_ok = fixed_case_ok &&
                        validate_fixed_info(fixed_info, vocab, fixed_top_k) &&
                        fixed_record.token_id == fixed_oracle.token_id &&
                        fixed_record.status == fixed_oracle.status &&
                        (fixed_oracle.status != SLLM_STATUS_OK ||
                         std::fabs(fixed_record.logprob -
                                   fixed_oracle.logprob) <= kLogprobTolerance);
        if (!fixed_case_ok && (vocab == 1U || vocab == 25U)) {
          std::cerr << "fixed top-k mismatch k=" << fixed_top_k
                    << " counter=" << fixed_counter
                    << " dispatch=" << fixed_info.dispatch_count
                    << " grid=" << fixed_info.grid_size_x
                    << " kernel=" << fixed_info.kernel_id
                    << " token=" << fixed_record.token_id
                    << " status=" << fixed_record.status
                    << " logprob=" << fixed_record.logprob
                    << " oracle=" << fixed_oracle.token_id << "/"
                    << fixed_oracle.status << "/" << fixed_oracle.logprob
                    << '\n';
        }
        if (vocab == 25U && fixed_variant == 0U && fixed_top_k == 20U &&
            fixed_counter == 3U) {
          fixed_case_ok = fixed_case_ok && fixed_record.token_id == 0 &&
                          std::fabs(fixed_record.logprob - (-1.0015019954F)) <=
                              kLogprobTolerance;
        }
        if (!fixed_case_ok && (vocab == 1U || vocab == 25U)) {
          std::cerr << "fixed top-k final mismatch k=" << fixed_top_k
                    << " counter=" << fixed_counter
                    << " dispatch=" << fixed_info.dispatch_count
                    << " token=" << fixed_record.token_id
                    << " status=" << fixed_record.status
                    << " logprob=" << fixed_record.logprob
                    << " oracle=" << fixed_oracle.token_id << "/"
                    << fixed_oracle.status << "/" << fixed_oracle.logprob
                    << '\n';
        }
        if (fixed_plan != nullptr) {
          fixed_case_ok = expect(sllm_token_selector_plan_release(
                                     &fixed_plan, &fixed_error.sink),
                                 SLLM_STATUS_OK, "fixed selector plan release",
                                 fixed_error) &&
                          fixed_case_ok;
        }
        ok = fixed_case_ok && ok;
      }
    }
  }

  // K=0 is the fixed nucleus profile.  The descriptor deliberately carries
  // an additive binding while the flag skips its read, matching normal
  // decode requests where the reusable full-vocab buffers remain bound.
  const std::vector<float> topp_additive(host_additive.size(), 0.0F);
  ok = ok && execute_fixed_topp_case(context, queue, vocab, seed, 0U,
                                     host_logits, topp_additive, host_mask,
                                     logits, additive, mask, output, workspace);
  if (vocab == 248320U) {
    // Exercise the full model-sized histogram with several independent draws
    // both with the deterministic mask and with the mask read disabled.  The
    // host oracle uses the same BF16 fixture but a separately constructed
    // all-valid mask for the latter path, so this covers both flag variants.
    constexpr uint64_t kNonuniformSeeds[] = {0U, 1U, 17U, 0x12345678U};
    for (const bool use_mask : {true, false}) {
      for (const uint64_t draw_seed : kNonuniformSeeds) {
        for (const uint64_t draw_counter : {0U, 1U}) {
          ok = ok && execute_fixed_topp_case(
                         context, queue, vocab, draw_seed, draw_counter,
                         host_logits, topp_additive, host_mask, logits,
                         additive, mask, output, workspace, -1, use_mask);
          if (!ok) {
            break;
          }
        }
        if (!ok) {
          break;
        }
      }
      if (!ok) {
        break;
      }
    }
  }
  if (vocab == 100U) {
    // 64 independent draws over the all-tied V=100 boundary fixture prove
    // both replay and token-ID ascending truncation (ceil(.95*100)=95).
    constexpr int32_t kNumPyTieTokens[64] = {
        83, 53, 56, 10, 40, 36, 70, 37, 58, 64, 3,  30, 55, 73, 39, 50,
        34, 47, 6,  69, 20, 2,  74, 86, 63, 60, 72, 56, 53, 69, 62, 79,
        87, 16, 50, 28, 86, 74, 87, 76, 20, 6,  70, 69, 93, 91, 69, 45,
        1,  10, 69, 34, 92, 74, 69, 40, 58, 20, 46, 55, 69, 24, 18, 52};
    for (uint64_t draw_seed = 0U; draw_seed != 64U; ++draw_seed) {
      ok = ok && execute_fixed_topp_case(context, queue, vocab, draw_seed, 0U,
                                         host_logits, topp_additive, host_mask,
                                         logits, additive, mask, output,
                                         workspace, kNumPyTieTokens[draw_seed]);
      if (!ok) {
        break;
      }
    }
    // A repeated seed/counter must replay the same token and logprob after
    // the 64-draw distribution sweep.
    for (uint32_t replay = 0U; replay != 2U; ++replay) {
      ok = ok && execute_fixed_topp_case(
                     context, queue, vocab, 17U, 0U, host_logits, topp_additive,
                     host_mask, logits, additive, mask, output, workspace);
    }
  }

  std::vector<uint8_t> all_masked(vocab, 0U);
  ok = ok && upload(queue, mask, all_masked.data(), all_masked.size()) &&
       execute_status_case(context, queue, vocab, temperature, seed, 0U, logits,
                           additive, mask, output,
                           SLLM_STATUS_TOKEN_SELECTOR_ALL_MASKED);
  ok = ok &&
       execute_fixed_status_case(context, queue, vocab, seed, 0U, logits,
                                 additive, mask, output, workspace, 0U,
                                 SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT,
                                 SLLM_HIP_TOKEN_SELECTOR_K0_WORKSPACE_BYTES,
                                 SLLM_STATUS_TOKEN_SELECTOR_ALL_MASKED);
  std::vector<float> nonfinite(host_additive);
  nonfinite[0U] = std::numeric_limits<float>::quiet_NaN();
  std::vector<uint8_t> one_mask(vocab, 0U);
  one_mask[0U] = 1U;
  ok = ok &&
       upload(queue, additive, nonfinite.data(),
              nonfinite.size() * sizeof(float)) &&
       upload(queue, mask, one_mask.data(), one_mask.size()) &&
       execute_status_case(context, queue, vocab, temperature, seed, 0U, logits,
                           additive, mask, output,
                           SLLM_STATUS_TOKEN_SELECTOR_NONFINITE);
  ok = ok &&
       execute_fixed_status_case(context, queue, vocab, seed, 0U, logits,
                                 additive, mask, output, workspace, 20U,
                                 SLLM_HIP_TOKEN_SELECTOR_FLAG_ADDITIVE_PRESENT |
                                     SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT,
                                 2U * ((vocab + 1023U) / 1024U) * 20U * 8U,
                                 SLLM_STATUS_TOKEN_SELECTOR_NONFINITE);

  if (output != nullptr) {
    ok = expect(sllm_buffer_release(&output, &error.sink), SLLM_STATUS_OK,
                "output release", error) &&
         ok;
  }
  if (workspace != nullptr) {
    ok = expect(sllm_buffer_release(&workspace, &error.sink), SLLM_STATUS_OK,
                "workspace release", error) &&
         ok;
  }
  if (mask != nullptr) {
    ok = expect(sllm_buffer_release(&mask, &error.sink), SLLM_STATUS_OK,
                "mask release", error) &&
         ok;
  }
  if (additive != nullptr) {
    ok = expect(sllm_buffer_release(&additive, &error.sink), SLLM_STATUS_OK,
                "additive release", error) &&
         ok;
  }
  if (logits != nullptr) {
    ok = expect(sllm_buffer_release(&logits, &error.sink), SLLM_STATUS_OK,
                "logits release", error) &&
         ok;
  }
  if (!ok) {
    std::cerr << "selector correctness case failed vocab=" << vocab << '\n';
  }
  std::cerr << "token-selector vocab=" << vocab << (ok ? " PASS\n" : " FAIL\n");
  return ok;
}

} // namespace

int main() {
  // The last case exercises the Gemma/Qwen vocabulary-sized path while the
  // smaller values cover alignment boundaries and non-power-of-two launches.
  constexpr uint64_t vocabularies[] = {
      1U, 3U, 17U, 25U, 100U, 255U, 256U, 257U, 1023U, 1024U, 1025U, 248320U};
  constexpr uint64_t seed = UINT64_C(0x123456789abcdef0);
  Error error;
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  std::strncpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
               sizeof(context_info.expected_gcn_arch_name) - 1U);
  sllm_context_t *context = nullptr;
  if (!expect(sllm_context_create(&context_info, &context, &error.sink),
              SLLM_STATUS_OK, "context create", error)) {
    return 1;
  }
  sllm_queue_create_info_t queue_info{
      sizeof(queue_info), SLLM_HIP_ABI_VERSION, 0U, {0U, 0U, 0U, 0U, 0U}};
  sllm_queue_t *queue = nullptr;
  bool ok = expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
                   SLLM_STATUS_OK, "queue create", error);
  if (!ok) {
    (void)sllm_context_release(&context, &error.sink);
    return 1;
  }
  for (const uint64_t vocab : vocabularies) {
    if (!run_case(context, queue, vocab, seed)) {
      ok = false;
      break;
    }
  }
  ok = ok && run_case(context, queue, 3U, 0U);
  ok = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
              "queue release", error) &&
       ok;
  ok = expect(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
              "context release", error) &&
       ok;
  if (ok) {
    std::cout
        << "{\"state\":\"PASS\",\"target\":\"" << SLLM_TEST_EXPECTED_TARGET
        << "\",\"vocabularies\":[1,3,17,25,100,255,256,257,1023,1024,1025,"
           "248320]"
           ",\"counters\":[0,1,3],\"record_bytes\":16"
           ",\"d2h_bytes\":16,\"status_cases\":[\"all_masked\",\"nonfinite\"]"
           ",\"fallback_allowed\":0,\"fallback_used\":0"
           ",\"oracle_logprob_tolerance\":0.005}\n";
  }
  return ok ? 0 : 1;
}
