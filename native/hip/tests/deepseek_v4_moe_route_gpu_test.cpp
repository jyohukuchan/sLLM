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

constexpr uint64_t kExperts = SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_EXPERT_COUNT;
constexpr uint32_t kSelected =
    SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_SELECTED_EXPERT_COUNT;

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
  uint32_t mode;
  uint32_t renormalize;
  float routed_scale;
  std::vector<uint16_t> logits;
  std::vector<float> bias;
  std::vector<int32_t> hash_ids;
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

uint16_t f32_to_bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
    return static_cast<uint16_t>(bits >> 16U);
  }
  bits += UINT32_C(0x7fff) + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

float bf16_to_f32(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float score(const uint16_t raw) {
  const float value = bf16_to_f32(raw);
  const float softplus = value > 0.0F ? value + std::log1p(std::exp(-value))
                                      : std::log1p(std::exp(value));
  return std::sqrt(softplus);
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

bool wait_and_release(sllm_completion_t **const completion) {
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  return expect(sllm_completion_wait(*completion, UINT32_MAX, &result,
                                     &error.sink),
                SLLM_STATUS_OK, "completion wait", error) &&
         expect(sllm_completion_release(completion, &error.sink),
                SLLM_STATUS_OK, "completion release", error);
}

bool wait_route_and_release(sllm_completion_t **const completion,
                            const int32_t expected_device_status) {
  const sllm_status_t expected_status =
      expected_device_status == SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_OK
          ? SLLM_STATUS_OK
          : SLLM_STATUS_INVALID_ARGUMENT;
  const uint32_t expected_state =
      expected_device_status == SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_OK
          ? SLLM_COMPLETION_STATE_SUCCESS
          : SLLM_COMPLETION_STATE_FAILURE;
  Error wait_error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  bool ok = expect(sllm_completion_wait(*completion, UINT32_MAX, &result,
                                        &wait_error.sink),
                   expected_status, "route completion wait", wait_error) &&
            result.state == expected_state &&
            (expected_status == SLLM_STATUS_OK ||
             wait_error.sink.message_length != 0U);

  result = {};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  Error query_error;
  ok = expect(sllm_completion_query(*completion, &result, &query_error.sink),
              expected_status, "route cached completion query", query_error) &&
       result.state == expected_state && ok;
  Error release_error;
  ok = expect(sllm_completion_release(completion, &release_error.sink),
              SLLM_STATUS_OK, "route completion release", release_error) &&
       *completion == nullptr && ok;
  return ok;
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

sllm_buffer_t *create_buffer(const sllm_context_t *const context,
                             const uint64_t bytes, const char *const label,
                             bool *const ok) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = bytes;
  sllm_buffer_t *buffer = nullptr;
  Error error;
  *ok = expect(sllm_buffer_create(context, &info, &buffer, &error.sink),
               SLLM_STATUS_OK, label, error) &&
        *ok;
  return buffer;
}

template <typename T>
const T *at(const std::vector<uint8_t> &bytes, const uint64_t offset) {
  return reinterpret_cast<const T *>(bytes.data() + offset);
}

Oracle make_oracle(const RouteCase &test_case) {
  Oracle result;
  const Layout metadata_layout = layout(test_case.tokens);
  result.ids.resize(static_cast<std::size_t>(metadata_layout.pairs));
  result.weights.resize(result.ids.size());
  result.counts.assign(static_cast<std::size_t>(kExperts), 0);
  result.offsets.assign(static_cast<std::size_t>(kExperts + 1U), 0);
  result.grouped_tokens.resize(result.ids.size());
  result.grouped_slots.resize(result.ids.size());
  for (uint64_t token = 0U; token < test_case.tokens; ++token) {
    const uint64_t pair_base = token * kSelected;
    if (test_case.mode == SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE) {
      std::vector<int32_t> order(static_cast<std::size_t>(kExperts));
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(
          order.begin(), order.end(),
          [&](const int32_t left, const int32_t right) {
            const float left_value =
                score(test_case.logits[token * kExperts +
                                       static_cast<uint32_t>(left)]) +
                test_case.bias[static_cast<uint32_t>(left)];
            const float right_value =
                score(test_case.logits[token * kExperts +
                                       static_cast<uint32_t>(right)]) +
                test_case.bias[static_cast<uint32_t>(right)];
            return left_value == right_value ? left < right
                                             : left_value > right_value;
          });
      for (uint32_t slot = 0U; slot < kSelected; ++slot) {
        result.ids[pair_base + slot] = order[slot];
      }
    } else {
      for (uint32_t slot = 0U; slot < kSelected; ++slot) {
        result.ids[pair_base + slot] = test_case.hash_ids[pair_base + slot];
      }
    }
    float denominator = 1.0F;
    if (test_case.renormalize != 0U) {
      denominator = 0.0F;
      for (uint32_t slot = 0U; slot < kSelected; ++slot) {
        denominator += score(
            test_case
                .logits[token * kExperts +
                        static_cast<uint32_t>(result.ids[pair_base + slot])]);
      }
    }
    for (uint32_t slot = 0U; slot < kSelected; ++slot) {
      result.weights[pair_base + slot] =
          score(test_case.logits[token * kExperts +
                                 static_cast<uint32_t>(
                                     result.ids[pair_base + slot])]) /
          denominator * test_case.routed_scale;
      ++result.counts[static_cast<uint32_t>(result.ids[pair_base + slot])];
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
                      const Layout &metadata_layout,
                      const int32_t expected_status) {
  if (*at<int32_t>(output, metadata_layout.status) != expected_status) {
    return false;
  }
  const int32_t *const ids = at<int32_t>(output, metadata_layout.ids);
  const float *const weights = at<float>(output, metadata_layout.weights);
  const int32_t *const counts = at<int32_t>(output, metadata_layout.counts);
  const int32_t *const offsets = at<int32_t>(output, metadata_layout.offsets);
  const int32_t *const grouped_tokens =
      at<int32_t>(output, metadata_layout.grouped_tokens);
  const int32_t *const grouped_slots =
      at<int32_t>(output, metadata_layout.grouped_slots);
  for (uint64_t pair = 0U; pair < metadata_layout.pairs; ++pair) {
    if (ids[pair] != -1 || !std::isnan(weights[pair]) ||
        grouped_tokens[pair] != -1 || grouped_slots[pair] != -1) {
      return false;
    }
  }
  for (uint64_t expert = 0U; expert < kExperts; ++expert) {
    if (counts[expert] != 0 || offsets[expert] != 0) {
      return false;
    }
  }
  return offsets[kExperts] == 0;
}

bool run_case(const sllm_context_t *const context,
              const sllm_queue_t *const queue, const RouteCase &test_case) {
  const Layout metadata_layout = layout(test_case.tokens);
  bool ok = true;
  sllm_buffer_t *logits_buffer =
      create_buffer(context, test_case.logits.size() * sizeof(uint16_t),
                    "logits buffer", &ok);
  sllm_buffer_t *bias_buffer = nullptr;
  sllm_buffer_t *hash_buffer = nullptr;
  if (test_case.mode == SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE) {
    bias_buffer = create_buffer(context, test_case.bias.size() * sizeof(float),
                                "bias buffer", &ok);
  } else {
    hash_buffer =
        create_buffer(context, test_case.hash_ids.size() * sizeof(int32_t),
                      "hash buffer", &ok);
  }
  sllm_buffer_t *output_buffer =
      create_buffer(context, metadata_layout.bytes, "metadata buffer", &ok);
  ok = ok && upload(queue, logits_buffer, test_case.logits.data(),
                    test_case.logits.size() * sizeof(uint16_t));
  if (test_case.mode == SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE) {
    ok = ok && upload(queue, bias_buffer, test_case.bias.data(),
                      test_case.bias.size() * sizeof(float));
  } else {
    ok = ok && upload(queue, hash_buffer, test_case.hash_ids.data(),
                      test_case.hash_ids.size() * sizeof(int32_t));
  }

  sllm_deepseek_v4_moe_route_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_VERSION;
  descriptor.mode = test_case.mode;
  descriptor.selected_expert_count = kSelected;
  descriptor.renormalize = test_case.renormalize;
  descriptor.routed_scale = test_case.routed_scale;
  descriptor.logits = binding(logits_buffer, SLLM_TENSOR_DTYPE_BF16, 2U,
                              test_case.tokens, kExperts);
  if (test_case.mode == SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE) {
    descriptor.selection_bias =
        binding(bias_buffer, SLLM_TENSOR_DTYPE_F32, 1U, kExperts);
  } else {
    descriptor.hash_expert_ids = binding(hash_buffer, SLLM_TENSOR_DTYPE_I32, 2U,
                                         test_case.tokens, kSelected);
  }
  descriptor.metadata =
      binding(output_buffer, SLLM_TENSOR_DTYPE_U8, 1U, metadata_layout.bytes);

  Error error;
  sllm_deepseek_v4_moe_route_query_info_t query{};
  query.struct_size = sizeof(query);
  query.abi_version = SLLM_HIP_ABI_VERSION;
  query.info_version = SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_QUERY_INFO_VERSION;
  ok =
      ok &&
      expect(sllm_deepseek_v4_moe_route_query(&descriptor, &query, &error.sink),
             SLLM_STATUS_OK, "route query", error) &&
      query.token_count == test_case.tokens && query.expert_count == kExperts &&
      query.pair_count == metadata_layout.pairs &&
      query.metadata_bytes == metadata_layout.bytes &&
      query.selected_expert_count == kSelected &&
      query.mode == test_case.mode &&
      query.renormalize == test_case.renormalize &&
      query.routed_scale == test_case.routed_scale;

  sllm_deepseek_v4_moe_route_plan_t *plan = nullptr;
  ok = ok && expect(sllm_deepseek_v4_moe_route_prepare(context, &descriptor,
                                                       &plan, &error.sink),
                    SLLM_STATUS_OK, "route prepare", error);
  sllm_deepseek_v4_moe_route_dispatch_info_t dispatch{};
  dispatch.struct_size = sizeof(dispatch);
  dispatch.abi_version = SLLM_HIP_ABI_VERSION;
  dispatch.info_version = SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  ok = ok &&
       expect(sllm_deepseek_v4_moe_route_execute(plan, queue, &completion,
                                                 &dispatch, &error.sink),
              SLLM_STATUS_OK, "route execute", error) &&
       wait_route_and_release(&completion, test_case.expected_status);
  std::vector<uint8_t> output(static_cast<std::size_t>(metadata_layout.bytes));
  ok = ok && download(queue, output_buffer, &output);

  ok = ok && dispatch.dispatch_count == 2U &&
       dispatch.kernel_id ==
           (test_case.mode == SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE
                ? SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_KERNEL_ID_SCORE_V1
                : SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_KERNEL_ID_HASH_V1) &&
       dispatch.token_count == test_case.tokens &&
       dispatch.expert_count == kExperts &&
       dispatch.pair_count == metadata_layout.pairs &&
       dispatch.selected_expert_count == kSelected &&
       dispatch.mode == test_case.mode &&
       dispatch.renormalize == test_case.renormalize &&
       dispatch.fallback_allowed == 0U && dispatch.fallback_used == 0U &&
       std::strcmp(dispatch.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0 &&
       std::strcmp(dispatch.device_symbol,
                   "sllm_deepseek_v4_moe_route_score_hash_v1") == 0;

  if (ok && test_case.expected_status != SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_OK) {
    ok = validate_failure(output, metadata_layout, test_case.expected_status);
  } else if (ok) {
    const Oracle oracle = make_oracle(test_case);
    const int32_t *const ids = at<int32_t>(output, metadata_layout.ids);
    const float *const weights = at<float>(output, metadata_layout.weights);
    const int32_t *const counts = at<int32_t>(output, metadata_layout.counts);
    const int32_t *const offsets = at<int32_t>(output, metadata_layout.offsets);
    const int32_t *const grouped_tokens =
        at<int32_t>(output, metadata_layout.grouped_tokens);
    const int32_t *const grouped_slots =
        at<int32_t>(output, metadata_layout.grouped_slots);
    ok = *at<int32_t>(output, metadata_layout.status) ==
         SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_OK;
    for (uint64_t pair = 0U; ok && pair < metadata_layout.pairs; ++pair) {
      ok = ids[pair] == oracle.ids[pair] &&
           std::abs(weights[pair] - oracle.weights[pair]) <= 8.0e-6F &&
           grouped_tokens[pair] == oracle.grouped_tokens[pair] &&
           grouped_slots[pair] == oracle.grouped_slots[pair];
    }
    for (uint64_t expert = 0U; ok && expert < kExperts; ++expert) {
      ok = counts[expert] == oracle.counts[expert] &&
           offsets[expert] == oracle.offsets[expert];
    }
    ok = ok && offsets[kExperts] == oracle.offsets[kExperts];
  }
  if (!ok) {
    std::cerr << "DeepSeek V4 route oracle mismatch: " << test_case.label
              << " M=" << test_case.tokens << '\n';
  }

  if (plan != nullptr) {
    ok = expect(sllm_deepseek_v4_moe_route_plan_release(&plan, &error.sink),
                SLLM_STATUS_OK, "route plan release", error) &&
         ok;
  }
  if (output_buffer != nullptr) {
    ok = expect(sllm_buffer_release(&output_buffer, &error.sink),
                SLLM_STATUS_OK, "metadata release", error) &&
         ok;
  }
  if (hash_buffer != nullptr) {
    ok = expect(sllm_buffer_release(&hash_buffer, &error.sink), SLLM_STATUS_OK,
                "hash release", error) &&
         ok;
  }
  if (bias_buffer != nullptr) {
    ok = expect(sllm_buffer_release(&bias_buffer, &error.sink), SLLM_STATUS_OK,
                "bias release", error) &&
         ok;
  }
  if (logits_buffer != nullptr) {
    ok = expect(sllm_buffer_release(&logits_buffer, &error.sink),
                SLLM_STATUS_OK, "logits release", error) &&
         ok;
  }
  return ok;
}

RouteCase score_case(const uint64_t tokens, const bool ties,
                     const char *const label) {
  RouteCase result{
      tokens,
      SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE,
      1U,
      1.5F,
      std::vector<uint16_t>(static_cast<std::size_t>(tokens * kExperts)),
      std::vector<float>(static_cast<std::size_t>(kExperts), 0.0F),
      {},
      SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_OK,
      label};
  for (uint64_t token = 0U; token < tokens; ++token) {
    for (uint64_t expert = 0U; expert < kExperts; ++expert) {
      const float value =
          ties
              ? 0.0F
              : -4.0F + static_cast<float>((expert * 17U + token * 11U) % 23U) /
                            16.0F;
      result.logits[token * kExperts + expert] = f32_to_bf16(value);
    }
    if (!ties) {
      result.logits[token * kExperts] = f32_to_bf16(5.0F);
      result.logits[token * kExperts + 255U] = f32_to_bf16(4.5F);
    }
  }
  return result;
}

bool scale_boundary_contract(const RouteCase &valid,
                             const sllm_deepseek_v4_moe_route_desc_t &base) {
  const float invalid_values[] = {0.0F, -1.0F,
                                  std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::quiet_NaN()};
  for (const float value : invalid_values) {
    sllm_deepseek_v4_moe_route_desc_t descriptor = base;
    descriptor.routed_scale = value;
    sllm_deepseek_v4_moe_route_query_info_t query{};
    query.struct_size = sizeof(query);
    query.abi_version = SLLM_HIP_ABI_VERSION;
    query.info_version = SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_QUERY_INFO_VERSION;
    Error error;
    if (!expect(
            sllm_deepseek_v4_moe_route_query(&descriptor, &query, &error.sink),
            SLLM_STATUS_INVALID_ARGUMENT, "invalid routed scale", error)) {
      std::cerr << "routed scale boundary failed for " << valid.label << '\n';
      return false;
    }
  }
  return true;
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

  for (const uint64_t tokens : {1U, 3U, 5U, 17U}) {
    const RouteCase finite = score_case(tokens, false, "score-finite");
    ok = ok && run_case(context, queue, finite);
  }
  ok = ok && run_case(context, queue, score_case(3U, true, "score-tie"));

  RouteCase unnormalized =
      score_case(1U, false, "score-unnormalized-positive-scale");
  unnormalized.renormalize = 0U;
  unnormalized.routed_scale = 0.75F;
  ok = ok && run_case(context, queue, unnormalized);

  RouteCase bias = score_case(1U, false, "score-bias-unbiased-weight");
  std::fill(bias.logits.begin(), bias.logits.end(), f32_to_bf16(-8.0F));
  bias.logits[10U] = f32_to_bf16(2.0F);
  bias.logits[20U] = f32_to_bf16(1.0F);
  bias.bias[20U] = 1.0F;
  ok = ok && run_case(context, queue, bias);

  RouteCase hash = score_case(5U, false, "hash-finite");
  hash.mode = SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_HASH;
  hash.bias.clear();
  hash.hash_ids.resize(static_cast<std::size_t>(hash.tokens * kSelected));
  for (uint64_t token = 0U; token < hash.tokens; ++token) {
    const int32_t ids[kSelected] = {0,
                                    255,
                                    static_cast<int32_t>(1U + token),
                                    static_cast<int32_t>(17U + token),
                                    static_cast<int32_t>(64U + token),
                                    static_cast<int32_t>(128U + token)};
    std::copy(std::begin(ids), std::end(ids),
              hash.hash_ids.begin() +
                  static_cast<std::ptrdiff_t>(token * kSelected));
  }
  ok = ok && run_case(context, queue, hash);

  RouteCase duplicate = hash;
  duplicate.tokens = 1U;
  duplicate.logits.resize(static_cast<std::size_t>(kExperts));
  duplicate.hash_ids = {0, 255, 7, 7, 9, 10};
  duplicate.expected_status =
      SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_DUPLICATE_EXPERT;
  duplicate.label = "hash-duplicate";
  ok = ok && run_case(context, queue, duplicate);

  RouteCase out_of_range = duplicate;
  out_of_range.hash_ids = {0, 255, 7, 8, 9, 256};
  out_of_range.expected_status =
      SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_EXPERT_OUT_OF_RANGE;
  out_of_range.label = "hash-out-of-range";
  ok = ok && run_case(context, queue, out_of_range);

  RouteCase nonfinite = score_case(1U, false, "score-nonfinite-logit");
  nonfinite.logits[31U] = f32_to_bf16(std::numeric_limits<float>::quiet_NaN());
  nonfinite.expected_status = SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_NONFINITE;
  ok = ok && run_case(context, queue, nonfinite);

  RouteCase nonfinite_bias = score_case(1U, false, "score-nonfinite-bias");
  nonfinite_bias.bias[17U] = std::numeric_limits<float>::infinity();
  nonfinite_bias.expected_status = SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_NONFINITE;
  ok = ok && run_case(context, queue, nonfinite_bias);

  /* Query needs only well-formed bindings; use non-null opaque sentinels to
   * cover routed_scale host validation without allocating another graph. */
  const RouteCase scale_case = score_case(1U, false, "scale-boundary");
  const Layout scale_layout = layout(1U);
  const auto *const sentinel = reinterpret_cast<const sllm_buffer_t *>(1U);
  sllm_deepseek_v4_moe_route_desc_t scale_descriptor{};
  scale_descriptor.struct_size = sizeof(scale_descriptor);
  scale_descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  scale_descriptor.op_version = SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_VERSION;
  scale_descriptor.mode = SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE;
  scale_descriptor.selected_expert_count = kSelected;
  scale_descriptor.renormalize = 1U;
  scale_descriptor.routed_scale = 1.5F;
  scale_descriptor.logits =
      binding(sentinel, SLLM_TENSOR_DTYPE_BF16, 2U, 1U, kExperts);
  scale_descriptor.selection_bias =
      binding(sentinel, SLLM_TENSOR_DTYPE_F32, 1U, kExperts);
  scale_descriptor.metadata =
      binding(sentinel, SLLM_TENSOR_DTYPE_U8, 1U, scale_layout.bytes);
  ok = ok && scale_boundary_contract(scale_case, scale_descriptor);

  ok = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
              "route queue release", error) &&
       ok;
  ok = expect(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
              "route context release", error) &&
       ok;
  if (ok) {
    std::cout << "Phase57 DeepSeek V4 moe_route GPU PASS target="
              << SLLM_TEST_EXPECTED_TARGET
              << " M=1,3,5,17 E=256 K=6 score=pass hash=pass"
                 " tie=stable bias=selection-only nonfinite=fail-closed"
                 " duplicate=fail-closed out-of-range=fail-closed fallback=0"
                 " completion=fail-closed cleanup=0\n";
  }
  return ok ? 0 : 1;
}
