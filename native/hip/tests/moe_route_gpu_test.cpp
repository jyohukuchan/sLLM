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

uint16_t f32_to_bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounded = bits + UINT32_C(0x7fff) + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(rounded >> 16U);
}

float bf16_to_f32(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

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

Layout layout(const uint64_t tokens, const uint64_t experts,
              const uint32_t selected) {
  const uint64_t pairs = tokens * selected;
  const uint64_t ids = 0U;
  const uint64_t weights = ids + pairs * 4U;
  const uint64_t counts = weights + pairs * 4U;
  const uint64_t offsets = counts + experts * 4U;
  const uint64_t grouped_tokens = offsets + (experts + 1U) * 4U;
  const uint64_t grouped_slots = grouped_tokens + pairs * 4U;
  const uint64_t status = grouped_slots + pairs * 4U;
  return {pairs,          ids,           weights, counts,     offsets,
          grouped_tokens, grouped_slots, status,  status + 4U};
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

bool wait_and_release(sllm_completion_t **const completion) {
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(
          sllm_completion_wait(*completion, UINT32_MAX, &result, &error.sink),
          SLLM_STATUS_OK, "completion wait", error)) {
    return false;
  }
  return expect(sllm_completion_release(completion, &error.sink),
                SLLM_STATUS_OK, "completion release", error);
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
          SLLM_STATUS_OK, "route download wait", error)) {
    return false;
  }
  uint64_t written = 0U;
  const bool read =
      expect(sllm_completion_read(completion, output->data(), output->size(),
                                  &written, &error.sink),
             SLLM_STATUS_OK, "route download read", error) &&
      written == output->size();
  return expect(sllm_completion_release(&completion, &error.sink),
                SLLM_STATUS_OK, "route download release", error) &&
         read;
}

template <typename T>
const T *at(const std::vector<uint8_t> &bytes, const uint64_t offset) {
  return reinterpret_cast<const T *>(bytes.data() + offset);
}

bool run_case(const sllm_context_t *const context,
              const sllm_queue_t *const queue, const uint64_t tokens,
              const bool all_ties) {
  constexpr uint64_t experts = 256U;
  constexpr uint32_t selected = 8U;
  std::vector<uint16_t> logits(static_cast<std::size_t>(tokens * experts));
  for (uint64_t token = 0U; token < tokens; ++token) {
    for (uint64_t expert = 0U; expert < experts; ++expert) {
      const float value =
          all_ties
              ? 0.0F
              : static_cast<float>(
                    static_cast<int64_t>((expert * 37U + token * 13U) % 97U) -
                    48) /
                    8.0F;
      logits[static_cast<std::size_t>(token * experts + expert)] =
          f32_to_bf16(value);
    }
  }
  std::vector<int32_t> ids(static_cast<std::size_t>(tokens * selected));
  std::vector<float> weights(ids.size());
  for (uint64_t token = 0U; token < tokens; ++token) {
    std::vector<int32_t> order(experts);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(
        order.begin(), order.end(),
        [&](const int32_t left, const int32_t right) {
          const float a = bf16_to_f32(
              logits[token * experts + static_cast<uint32_t>(left)]);
          const float b = bf16_to_f32(
              logits[token * experts + static_cast<uint32_t>(right)]);
          return a == b ? left < right : a > b;
        });
    float maximum = -std::numeric_limits<float>::infinity();
    for (uint64_t expert = 0U; expert < experts; ++expert) {
      maximum =
          std::max(maximum, bf16_to_f32(logits[token * experts + expert]));
    }
    float denominator = 0.0F;
    for (uint64_t expert = 0U; expert < experts; ++expert) {
      denominator +=
          std::exp(bf16_to_f32(logits[token * experts + expert]) - maximum);
    }
    float chosen_sum = 0.0F;
    for (uint32_t slot = 0U; slot < selected; ++slot) {
      const uint64_t pair = token * selected + slot;
      ids[pair] = order[slot];
      weights[pair] =
          std::exp(bf16_to_f32(logits[token * experts +
                                      static_cast<uint32_t>(order[slot])]) -
                   maximum) /
          denominator;
      chosen_sum += weights[pair];
    }
    for (uint32_t slot = 0U; slot < selected; ++slot) {
      weights[token * selected + slot] /= chosen_sum;
    }
  }
  const Layout metadata_layout = layout(tokens, experts, selected);
  sllm_buffer_create_info_t logits_info{};
  logits_info.struct_size = sizeof(logits_info);
  logits_info.abi_version = SLLM_HIP_ABI_VERSION;
  logits_info.size_bytes = logits.size() * sizeof(uint16_t);
  sllm_buffer_create_info_t output_info{};
  output_info.struct_size = sizeof(output_info);
  output_info.abi_version = SLLM_HIP_ABI_VERSION;
  output_info.size_bytes = metadata_layout.bytes;
  sllm_buffer_t *logits_buffer = nullptr;
  sllm_buffer_t *output_buffer = nullptr;
  Error error;
  bool ok = expect(sllm_buffer_create(context, &logits_info, &logits_buffer,
                                      &error.sink),
                   SLLM_STATUS_OK, "route logits buffer", error) &&
            expect(sllm_buffer_create(context, &output_info, &output_buffer,
                                      &error.sink),
                   SLLM_STATUS_OK, "route metadata buffer", error) &&
            upload(queue, logits_buffer, logits.data(), logits_info.size_bytes);
  sllm_moe_route_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_MOE_ROUTE_VERSION;
  descriptor.selected_expert_count = selected;
  descriptor.logits =
      binding(logits_buffer, SLLM_TENSOR_DTYPE_BF16, 2U, tokens, experts);
  descriptor.metadata =
      binding(output_buffer, SLLM_TENSOR_DTYPE_U8, 1U, metadata_layout.bytes);
  sllm_moe_route_plan_t *plan = nullptr;
  ok = ok &&
       expect(sllm_moe_route_prepare(context, &descriptor, &plan, &error.sink),
              SLLM_STATUS_OK, "route prepare", error);
  sllm_moe_route_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_MOE_ROUTE_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  ok = ok &&
       expect(
           sllm_moe_route_execute(plan, queue, &completion, &info, &error.sink),
           SLLM_STATUS_OK, "route execute", error) &&
       wait_and_release(&completion);
  std::vector<uint8_t> output(static_cast<std::size_t>(metadata_layout.bytes));
  ok = ok && download(queue, output_buffer, &output);
  const int32_t *const actual_ids = at<int32_t>(output, metadata_layout.ids);
  const float *const actual_weights =
      at<float>(output, metadata_layout.weights);
  const int32_t *const counts = at<int32_t>(output, metadata_layout.counts);
  const int32_t *const offsets = at<int32_t>(output, metadata_layout.offsets);
  const int32_t *const grouped_tokens =
      at<int32_t>(output, metadata_layout.grouped_tokens);
  const int32_t *const grouped_slots =
      at<int32_t>(output, metadata_layout.grouped_slots);
  ok = ok && *at<int32_t>(output, metadata_layout.status) == 0;
  for (uint64_t pair = 0U; ok && pair < metadata_layout.pairs; ++pair) {
    ok = actual_ids[pair] == ids[pair] &&
         std::abs(actual_weights[pair] - weights[pair]) <= 2.0e-6F;
  }
  int32_t cursor = 0;
  int32_t maximum_expert_count = 0;
  int32_t active_expert_count = 0;
  for (int32_t expert = 0; ok && expert < static_cast<int32_t>(experts);
       ++expert) {
    ok = offsets[expert] == cursor;
    int32_t count = 0;
    for (uint64_t token = 0U; token < tokens; ++token) {
      for (uint32_t slot = 0U; slot < selected; ++slot) {
        if (ids[token * selected + slot] == expert) {
          ok = ok && grouped_tokens[cursor] == static_cast<int32_t>(token) &&
               grouped_slots[cursor] == static_cast<int32_t>(slot);
          ++cursor;
          ++count;
        }
      }
    }
    ok = ok && counts[expert] == count;
    maximum_expert_count = std::max(maximum_expert_count, count);
    active_expert_count += count != 0 ? 1 : 0;
  }
  ok = ok && offsets[experts] == cursor && info.dispatch_count == 2U &&
       info.kernel_id == SLLM_HIP_MOE_ROUTE_KERNEL_ID_STABLE_TOPK_V1 &&
       info.token_count == tokens && info.expert_count == experts &&
       info.pair_count == metadata_layout.pairs &&
       info.selected_expert_count == selected && info.fallback_allowed == 0U &&
       info.fallback_used == 0U &&
       std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;
  if (!ok) {
    std::cerr << "MoE route oracle mismatch: tokens=" << tokens
              << " ties=" << all_ties << '\n';
  }
  if (plan != nullptr) {
    ok = expect(sllm_moe_route_plan_release(&plan, &error.sink), SLLM_STATUS_OK,
                "route plan release", error) &&
         ok;
  }
  ok = expect(sllm_buffer_release(&output_buffer, &error.sink), SLLM_STATUS_OK,
              "route metadata release", error) &&
       ok;
  ok = expect(sllm_buffer_release(&logits_buffer, &error.sink), SLLM_STATUS_OK,
              "route logits release", error) &&
       ok;
  if (ok) {
    std::cout << "PASS target=" << SLLM_TEST_EXPECTED_TARGET
              << " tokens=" << tokens << " experts=256 topk=8 ties=" << all_ties
              << " active_experts=" << active_expert_count
              << " max_expert_count=" << maximum_expert_count
              << " fallback=0\n";
  }
  return ok;
}

} // namespace

int main() {
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  std::strncpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
               sizeof(context_info.expected_gcn_arch_name) - 1U);
  sllm_context_t *context = nullptr;
  Error error;
  if (!expect(sllm_context_create(&context_info, &context, &error.sink),
              SLLM_STATUS_OK, "route context", error)) {
    return 1;
  }
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  sllm_queue_t *queue = nullptr;
  bool ok = expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
                   SLLM_STATUS_OK, "route queue", error);
  const uint64_t cases[] = {1U, 2U, 3U, 7U, 8U, 31U, 32U, 33U};
  for (const uint64_t tokens : cases) {
    ok = ok && run_case(context, queue, tokens, false);
  }
  ok = ok && run_case(context, queue, 3U, true);
  ok = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
              "route queue release", error) &&
       ok;
  ok = expect(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
              "route context release", error) &&
       ok;
  return ok ? 0 : 1;
}
