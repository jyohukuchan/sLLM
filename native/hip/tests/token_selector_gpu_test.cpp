#include "sllm/hip.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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
  return expect(sllm_completion_wait(*completion, UINT32_MAX, &result,
                                     &error.sink),
                SLLM_STATUS_OK, label, error) &&
         expect(sllm_completion_release(completion, &error.sink),
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
  if (plan != nullptr) {
    ok = expect(sllm_token_selector_plan_release(&plan, &error.sink),
                SLLM_STATUS_OK, "status selector plan release", error) &&
         ok;
  }
  return ok;
}

bool run_case(const sllm_context_t *const context,
              const sllm_queue_t *const queue, const uint64_t vocab,
              const uint64_t seed) {
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
    fill_inputs(vocab, &host_logits, &host_additive, &host_mask);
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
  bool ok = create(host_logits.size() * sizeof(uint16_t), &logits) &&
            create(host_additive.size() * sizeof(float), &additive) &&
            create(host_mask.size(), &mask) &&
            create(SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES, &output) &&
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

  std::vector<uint8_t> all_masked(vocab, 0U);
  ok = ok && upload(queue, mask, all_masked.data(), all_masked.size()) &&
       execute_status_case(context, queue, vocab, temperature, seed, 0U, logits,
                           additive, mask, output,
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

  if (output != nullptr) {
    ok = expect(sllm_buffer_release(&output, &error.sink), SLLM_STATUS_OK,
                "output release", error) &&
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
  return ok;
}

} // namespace

int main() {
  // The last case exercises the Gemma/Qwen vocabulary-sized path while the
  // smaller values cover alignment boundaries and non-power-of-two launches.
  constexpr uint64_t vocabularies[] = {1U, 3U, 17U, 255U, 256U, 257U, 248320U};
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
        << "\",\"vocabularies\":[1,3,17,255,256,257,248320]"
           ",\"counters\":[0,1],\"record_bytes\":16"
           ",\"d2h_bytes\":16,\"status_cases\":[\"all_masked\",\"nonfinite\"]"
           ",\"fallback_allowed\":0,\"fallback_used\":0"
           ",\"oracle_logprob_tolerance\":0.005}\n";
  }
  return ok ? 0 : 1;
}
