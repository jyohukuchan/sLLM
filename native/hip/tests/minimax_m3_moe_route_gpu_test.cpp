#include "sllm/hip.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1201"
#endif

namespace {

constexpr uint64_t kExperts = SLLM_HIP_MINIMAX_M3_MOE_ROUTE_EXPERT_COUNT;
constexpr uint32_t kSelected =
    SLLM_HIP_MINIMAX_M3_MOE_ROUTE_SELECTED_EXPERT_COUNT;

struct Error final {
  char message[512]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};
};

struct Layout final {
  uint64_t pairs;
  uint64_t ids;
  uint64_t weights;
  uint64_t counts;
  uint64_t offsets;
  uint64_t grouped_tokens;
  uint64_t grouped_slots;
  uint64_t status;
  uint64_t bytes;
};

struct RouteCase final {
  uint64_t tokens;
  std::vector<float> logits;
  std::vector<float> bias;
  int32_t expected_status;
  const char *label;
};

struct Oracle final {
  std::vector<int32_t> ids;
  std::vector<float> weights;
  std::vector<int32_t> counts;
  std::vector<int32_t> offsets;
  std::vector<int32_t> grouped_tokens;
  std::vector<int32_t> grouped_slots;
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

float sigmoid(const float value) {
  if (value >= 0.0F) {
    const float negative_exp = std::exp(-value);
    return 1.0F / (1.0F + negative_exp);
  }
  const float positive_exp = std::exp(value);
  return positive_exp / (1.0F + positive_exp);
}

Layout layout(const uint64_t tokens) {
  const uint64_t pairs = tokens * kSelected;
  const uint64_t ids = 0U;
  const uint64_t weights = ids + pairs * sizeof(int32_t);
  const uint64_t counts = weights + pairs * sizeof(float);
  const uint64_t offsets = counts + kExperts * sizeof(int32_t);
  const uint64_t grouped_tokens = offsets + (kExperts + 1U) * sizeof(int32_t);
  const uint64_t grouped_slots = grouped_tokens + pairs * sizeof(int32_t);
  const uint64_t status = grouped_slots + pairs * sizeof(int32_t);
  return {pairs,         ids,     weights,
          counts,        offsets, grouped_tokens,
          grouped_slots, status,  status + sizeof(int32_t)};
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

bool wait_and_release(sllm_completion_t **const completion,
                      const sllm_status_t expected_status = SLLM_STATUS_OK) {
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  bool ok = expect(
      sllm_completion_wait(*completion, UINT32_MAX, &result, &error.sink),
      expected_status, "completion wait", error);
  Error cached_error;
  ok = expect(sllm_completion_query(*completion, &result, &cached_error.sink),
              expected_status, "cached completion query", cached_error) &&
       ok;
  Error release_error;
  return expect(sllm_completion_release(completion, &release_error.sink),
                SLLM_STATUS_OK, "completion release", release_error) &&
         ok;
}

bool upload(const sllm_queue_t *const queue, const sllm_buffer_t *const buffer,
            const void *const data, const uint64_t bytes) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = const_cast<void *>(data);
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  return expect(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                     &error.sink),
                SLLM_STATUS_OK, "route upload", error) &&
         wait_and_release(&completion);
}

bool download(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer,
              std::vector<uint8_t> *const output) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.size_bytes = output->size();
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_buffer_copy_d2h(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, "route download", error)) {
    return false;
  }
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(
          sllm_completion_wait(completion, UINT32_MAX, &result, &error.sink),
          SLLM_STATUS_OK, "download wait", error)) {
    return false;
  }
  uint64_t written = 0U;
  const bool read =
      expect(sllm_completion_read(completion, output->data(), output->size(),
                                  &written, &error.sink),
             SLLM_STATUS_OK, "download read", error) &&
      written == output->size();
  return expect(sllm_completion_release(&completion, &error.sink),
                SLLM_STATUS_OK, "download release", error) &&
         read;
}

sllm_buffer_t *create_buffer(const sllm_context_t *const context,
                             const uint64_t bytes, bool *const ok) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = bytes;
  sllm_buffer_t *buffer = nullptr;
  Error error;
  *ok = expect(sllm_buffer_create(context, &info, &buffer, &error.sink),
               SLLM_STATUS_OK, "buffer create", error) &&
        *ok;
  return buffer;
}

template <typename T>
const T *at(const std::vector<uint8_t> &bytes, const uint64_t offset) {
  return reinterpret_cast<const T *>(bytes.data() + offset);
}

Oracle make_oracle(const RouteCase &test_case) {
  Oracle result;
  const Layout metadata = layout(test_case.tokens);
  result.ids.resize(static_cast<std::size_t>(metadata.pairs));
  result.weights.resize(result.ids.size());
  result.counts.assign(static_cast<std::size_t>(kExperts), 0);
  result.offsets.assign(static_cast<std::size_t>(kExperts + 1U), 0);
  result.grouped_tokens.resize(result.ids.size());
  result.grouped_slots.resize(result.ids.size());
  for (uint64_t token = 0U; token < test_case.tokens; ++token) {
    std::vector<int32_t> order(static_cast<std::size_t>(kExperts));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(
        order.begin(), order.end(),
        [&](const int32_t left, const int32_t right) {
          const float left_value =
              sigmoid(test_case.logits[token * kExperts +
                                       static_cast<uint32_t>(left)]) +
              test_case.bias[static_cast<uint32_t>(left)];
          const float right_value =
              sigmoid(test_case.logits[token * kExperts +
                                       static_cast<uint32_t>(right)]) +
              test_case.bias[static_cast<uint32_t>(right)];
          return left_value == right_value ? left < right
                                           : left_value > right_value;
        });
    const uint64_t base = token * kSelected;
    float normalizer = 0.0F;
    for (uint32_t slot = 0U; slot < kSelected; ++slot) {
      result.ids[base + slot] = order[slot];
      normalizer += sigmoid(
          test_case
              .logits[token * kExperts + static_cast<uint32_t>(order[slot])]);
    }
    for (uint32_t slot = 0U; slot < kSelected; ++slot) {
      const uint32_t expert = static_cast<uint32_t>(result.ids[base + slot]);
      result.weights[base + slot] =
          sigmoid(test_case.logits[token * kExperts + expert]) / normalizer *
          2.0F;
      ++result.counts[expert];
    }
  }
  int32_t cursor = 0;
  for (uint32_t expert = 0U; expert < kExperts; ++expert) {
    result.offsets[expert] = cursor;
    for (uint64_t token = 0U; token < test_case.tokens; ++token) {
      for (uint32_t slot = 0U; slot < kSelected; ++slot) {
        const uint64_t pair = token * kSelected + slot;
        if (result.ids[pair] == static_cast<int32_t>(expert)) {
          result.grouped_tokens[static_cast<std::size_t>(cursor)] =
              static_cast<int32_t>(token);
          result.grouped_slots[static_cast<std::size_t>(cursor)] =
              static_cast<int32_t>(slot);
          ++cursor;
        }
      }
    }
  }
  result.offsets[kExperts] = cursor;
  return result;
}

bool validate_failure(const std::vector<uint8_t> &output,
                      const Layout &metadata, const int32_t status) {
  if (*at<int32_t>(output, metadata.status) != status) {
    return false;
  }
  const int32_t *const ids = at<int32_t>(output, metadata.ids);
  const float *const weights = at<float>(output, metadata.weights);
  const int32_t *const counts = at<int32_t>(output, metadata.counts);
  const int32_t *const offsets = at<int32_t>(output, metadata.offsets);
  for (uint64_t pair = 0U; pair < metadata.pairs; ++pair) {
    if (ids[pair] != -1 || !std::isnan(weights[pair])) {
      return false;
    }
  }
  for (uint64_t expert = 0U; expert <= kExperts; ++expert) {
    if (offsets[expert] != 0 || (expert < kExperts && counts[expert] != 0)) {
      return false;
    }
  }
  return true;
}

bool run_case(const sllm_context_t *const context,
              const sllm_queue_t *const queue, const RouteCase &test_case) {
  const Layout metadata = layout(test_case.tokens);
  bool ok = true;
  sllm_buffer_t *logits =
      create_buffer(context, test_case.logits.size() * sizeof(float), &ok);
  sllm_buffer_t *bias =
      create_buffer(context, test_case.bias.size() * sizeof(float), &ok);
  sllm_buffer_t *output = create_buffer(context, metadata.bytes, &ok);
  ok = ok &&
       upload(queue, logits, test_case.logits.data(),
              test_case.logits.size() * sizeof(float)) &&
       upload(queue, bias, test_case.bias.data(),
              test_case.bias.size() * sizeof(float));

  sllm_minimax_m3_moe_route_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_MINIMAX_M3_MOE_ROUTE_VERSION;
  descriptor.selected_expert_count = kSelected;
  descriptor.logits =
      binding(logits, SLLM_TENSOR_DTYPE_F32, 2U, test_case.tokens, kExperts);
  descriptor.selection_bias =
      binding(bias, SLLM_TENSOR_DTYPE_F32, 1U, kExperts);
  descriptor.metadata =
      binding(output, SLLM_TENSOR_DTYPE_U8, 1U, metadata.bytes);
  sllm_minimax_m3_moe_route_plan_t *plan = nullptr;
  Error error;
  ok = ok && expect(sllm_minimax_m3_moe_route_prepare(context, &descriptor,
                                                      &plan, &error.sink),
                    SLLM_STATUS_OK, "route prepare", error);
  sllm_minimax_m3_moe_route_dispatch_info_t dispatch{};
  dispatch.struct_size = sizeof(dispatch);
  dispatch.abi_version = SLLM_HIP_ABI_VERSION;
  dispatch.info_version = SLLM_HIP_MINIMAX_M3_MOE_ROUTE_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  ok = ok && expect(sllm_minimax_m3_moe_route_execute(plan, queue, &completion,
                                                      &dispatch, &error.sink),
                    SLLM_STATUS_OK, "route execute", error);
  const sllm_status_t completion_status =
      test_case.expected_status == SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_OK
          ? SLLM_STATUS_OK
          : SLLM_STATUS_INVALID_ARGUMENT;
  ok = ok && wait_and_release(&completion, completion_status);
  std::vector<uint8_t> bytes(static_cast<std::size_t>(metadata.bytes));
  ok = ok && download(queue, output, &bytes) && dispatch.dispatch_count == 2U &&
       dispatch.kernel_id ==
           SLLM_HIP_MINIMAX_M3_MOE_ROUTE_KERNEL_ID_SIGMOID_TOP4_V1 &&
       dispatch.token_count == test_case.tokens &&
       dispatch.expert_count == kExperts &&
       dispatch.pair_count == metadata.pairs &&
       dispatch.selected_expert_count == kSelected &&
       dispatch.fallback_allowed == 0U && dispatch.fallback_used == 0U &&
       std::strcmp(dispatch.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0 &&
       std::strcmp(dispatch.device_symbol,
                   "sllm_minimax_m3_moe_route_sigmoid_top4_v1") == 0;
  if (ok && test_case.expected_status != SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_OK) {
    ok = validate_failure(bytes, metadata, test_case.expected_status);
  } else if (ok) {
    const Oracle oracle = make_oracle(test_case);
    const int32_t *const ids = at<int32_t>(bytes, metadata.ids);
    const float *const weights = at<float>(bytes, metadata.weights);
    const int32_t *const counts = at<int32_t>(bytes, metadata.counts);
    const int32_t *const offsets = at<int32_t>(bytes, metadata.offsets);
    const int32_t *const grouped_tokens =
        at<int32_t>(bytes, metadata.grouped_tokens);
    const int32_t *const grouped_slots =
        at<int32_t>(bytes, metadata.grouped_slots);
    ok = *at<int32_t>(bytes, metadata.status) ==
         SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_OK;
    for (uint64_t pair = 0U; ok && pair < metadata.pairs; ++pair) {
      ok = ids[pair] == oracle.ids[pair] &&
           std::abs(weights[pair] - oracle.weights[pair]) <= 2.0e-6F &&
           grouped_tokens[pair] == oracle.grouped_tokens[pair] &&
           grouped_slots[pair] == oracle.grouped_slots[pair];
    }
    for (uint64_t expert = 0U; ok && expert <= kExperts; ++expert) {
      ok = offsets[expert] == oracle.offsets[expert] &&
           (expert == kExperts || counts[expert] == oracle.counts[expert]);
    }
  }
  if (!ok) {
    std::cerr << "MiniMax route oracle mismatch: " << test_case.label
              << " M=" << test_case.tokens << '\n';
  }
  ok = expect(sllm_minimax_m3_moe_route_plan_release(&plan, &error.sink),
              SLLM_STATUS_OK, "plan release", error) &&
       ok;
  ok = expect(sllm_buffer_release(&output, &error.sink), SLLM_STATUS_OK,
              "output release", error) &&
       expect(sllm_buffer_release(&bias, &error.sink), SLLM_STATUS_OK,
              "bias release", error) &&
       expect(sllm_buffer_release(&logits, &error.sink), SLLM_STATUS_OK,
              "logits release", error) &&
       ok;
  return ok;
}

RouteCase valid_case(const uint64_t tokens, const char *const label) {
  RouteCase result{
      tokens, std::vector<float>(static_cast<std::size_t>(tokens * kExperts)),
      std::vector<float>(static_cast<std::size_t>(kExperts), 0.0F),
      SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_OK, label};
  for (uint64_t token = 0U; token < tokens; ++token) {
    for (uint64_t expert = 0U; expert < kExperts; ++expert) {
      result.logits[token * kExperts + expert] =
          -4.0F + static_cast<float>((expert * 17U + token * 11U) % 31U) / 8.0F;
    }
    result.logits[token * kExperts] = 8.0F;
    result.logits[token * kExperts + 127U] = 7.0F;
  }
  return result;
}

} // namespace

int main() {
  Error error;
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::strncpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
               sizeof(context_info.expected_gcn_arch_name) - 1U);
  sllm_context_t *context = nullptr;
  if (!expect(sllm_context_create(&context_info, &context, &error.sink),
              SLLM_STATUS_OK, "context create", error)) {
    return 1;
  }
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  sllm_queue_t *queue = nullptr;
  bool ok = expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
                   SLLM_STATUS_OK, "queue create", error);
  for (const uint64_t tokens :
       {UINT64_C(1), UINT64_C(3), UINT64_C(5), UINT64_C(17)}) {
    RouteCase test_case = valid_case(tokens, "stable sigmoid selection");
    if (tokens == 1U) {
      std::fill(test_case.logits.begin(), test_case.logits.end(), 0.0F);
    }
    if (tokens == 3U) {
      test_case.bias[10] = 100.0F;
      test_case.logits[10] = -8.0F;
    }
    ok = run_case(context, queue, test_case) && ok;
  }
  RouteCase nonfinite = valid_case(1U, "nonfinite logit");
  nonfinite.logits[0] = std::numeric_limits<float>::quiet_NaN();
  nonfinite.expected_status = SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_NONFINITE;
  ok = run_case(context, queue, nonfinite) && ok;
  RouteCase nonfinite_bias = valid_case(1U, "nonfinite bias");
  nonfinite_bias.bias[127] = std::numeric_limits<float>::infinity();
  nonfinite_bias.expected_status = SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_NONFINITE;
  ok = run_case(context, queue, nonfinite_bias) && ok;
  RouteCase zero = valid_case(1U, "zero normalizer");
  std::fill(zero.logits.begin(), zero.logits.end(),
            -std::numeric_limits<float>::max());
  zero.expected_status = SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_ZERO_NORMALIZER;
  ok = run_case(context, queue, zero) && ok;

  ok = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
              "queue release", error) &&
       expect(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
              "context release", error) &&
       ok;
  if (!ok) {
    return 1;
  }
  std::cout << "MiniMax M3 MoE route exact GPU oracle ("
            << SLLM_TEST_EXPECTED_TARGET << "): PASS\n";
  return 0;
}
