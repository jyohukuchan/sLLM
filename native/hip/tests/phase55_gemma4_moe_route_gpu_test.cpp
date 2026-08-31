#include "sllm/hip.h"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1201"
#endif

namespace {

constexpr uint64_t kExpertCount = 128U;
constexpr uint32_t kSelectedCount = 8U;

struct Error final {
  char message[512]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};
};

struct Device final {
  int index = 0;
  int count = 0;
  hipDeviceProp_t properties{};
};

struct Layout final {
  uint64_t pair_count = 0U;
  uint64_t ids = 0U;
  uint64_t weights = 0U;
  uint64_t counts = 0U;
  uint64_t offsets = 0U;
  uint64_t grouped_tokens = 0U;
  uint64_t grouped_slots = 0U;
  uint64_t status = 0U;
  uint64_t bytes = 0U;
};

struct Oracle final {
  std::vector<int32_t> ids;
  std::vector<float> weights;
  std::vector<int32_t> counts;
  std::vector<int32_t> offsets;
  std::vector<int32_t> grouped_tokens;
  std::vector<int32_t> grouped_slots;
};

struct Evidence final {
  uint64_t finite_cases = 0U;
  uint64_t nonfinite_cases = 0U;
  uint64_t metadata_mismatches = 0U;
  uint64_t weight_mismatches = 0U;
  uint64_t dispatches = 0U;
  double max_weight_abs_error = 0.0;
  double max_weight_rel_error = 0.0;
  double max_selected_sum_error = 0.0;
  bool selected_expert_zero = false;
  bool selected_expert_last = false;
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

bool hip_expect(const hipError_t actual, const hipError_t expected,
                const char *const operation) {
  if (actual == expected) {
    return true;
  }
  std::cerr << operation << " returned " << static_cast<int>(actual) << ": "
            << hipGetErrorString(actual) << '\n';
  return false;
}

bool parse_index(const char *const text, int *const result) {
  if (text == nullptr || text[0] == '\0' || text[0] == '-') {
    return false;
  }
  uint64_t value = 0U;
  for (const char *cursor = text; *cursor != '\0'; ++cursor) {
    if (*cursor < '0' || *cursor > '9') {
      return false;
    }
    const uint64_t digit = static_cast<uint64_t>(*cursor - '0');
    if (value >
        (static_cast<uint64_t>(std::numeric_limits<int>::max()) - digit) /
            10U) {
      return false;
    }
    value = value * 10U + digit;
  }
  *result = static_cast<int>(value);
  return true;
}

bool select_device(const int index, Device *const device) {
  if (!hip_expect(hipGetDeviceCount(&device->count), hipSuccess,
                  "hipGetDeviceCount") ||
      device->count <= 0 || index < 0 || index >= device->count) {
    std::cerr << "device index " << index << " is not in visible range [0, "
              << device->count << ")\n";
    return false;
  }
  device->index = index;
  return hip_expect(hipSetDevice(index), hipSuccess, "hipSetDevice") &&
         hip_expect(hipGetDeviceProperties(&device->properties, index),
                    hipSuccess, "hipGetDeviceProperties");
}

uint16_t f32_to_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr uint32_t exponent_mask = UINT32_C(0x7f800000);
  constexpr uint32_t fraction_mask = UINT32_C(0x007fffff);
  if ((bits & exponent_mask) == exponent_mask) {
    if ((bits & fraction_mask) != 0U) {
      const uint16_t sign =
          static_cast<uint16_t>((bits >> 16U) & UINT32_C(0x8000));
      const uint16_t payload =
          static_cast<uint16_t>((bits >> 16U) & UINT32_C(0x003f));
      return static_cast<uint16_t>(sign | UINT16_C(0x7fc0) | payload);
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & UINT32_C(1)) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

float bf16_to_f32(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

Layout make_layout(const uint64_t tokens) {
  Layout result;
  result.pair_count = tokens * kSelectedCount;
  result.ids = 0U;
  result.weights = result.ids + result.pair_count * sizeof(int32_t);
  result.counts = result.weights + result.pair_count * sizeof(float);
  result.offsets = result.counts + kExpertCount * sizeof(int32_t);
  result.grouped_tokens =
      result.offsets + (kExpertCount + 1U) * sizeof(int32_t);
  result.grouped_slots =
      result.grouped_tokens + result.pair_count * sizeof(int32_t);
  result.status = result.grouped_slots + result.pair_count * sizeof(int32_t);
  result.bytes = result.status + sizeof(int32_t);
  return result;
}

bool validate_layout_contract(const Layout &layout, const uint64_t tokens) {
  const uint64_t pairs = tokens * kSelectedCount;
  return layout.pair_count == pairs && layout.ids == 0U &&
         layout.weights == pairs * 4U && layout.counts == pairs * 8U &&
         layout.offsets == pairs * 8U + 512U &&
         layout.grouped_tokens == pairs * 8U + 1028U &&
         layout.grouped_slots == pairs * 12U + 1028U &&
         layout.status == pairs * 16U + 1028U &&
         layout.bytes == tokens * 128U + 1032U;
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

bool create_buffer(const sllm_context_t *const context, const uint64_t bytes,
                   sllm_buffer_t **const output) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = bytes;
  Error error;
  return expect(sllm_buffer_create(context, &info, output, &error.sink),
                SLLM_STATUS_OK, "sllm_buffer_create", error) &&
         *output != nullptr;
}

bool wait_and_release(sllm_completion_t **const completion,
                      const char *const operation) {
  if (completion == nullptr || *completion == nullptr) {
    std::cerr << operation << " returned a null completion\n";
    return false;
  }
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  const bool waited = expect(
      sllm_completion_wait(*completion, UINT32_MAX, &result, &error.sink),
      SLLM_STATUS_OK, operation, error);
  const bool successful =
      waited && result.state == SLLM_COMPLETION_STATE_SUCCESS;
  const bool released =
      expect(sllm_completion_release(completion, &error.sink), SLLM_STATUS_OK,
             "sllm_completion_release", error);
  return successful && released && *completion == nullptr;
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
                SLLM_STATUS_OK, "sllm_buffer_copy_h2d", error) &&
         wait_and_release(&completion, "sllm_completion_wait(h2d)");
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
              SLLM_STATUS_OK, "sllm_buffer_copy_d2h", error) ||
      completion == nullptr) {
    return false;
  }
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(
          sllm_completion_wait(completion, UINT32_MAX, &result, &error.sink),
          SLLM_STATUS_OK, "sllm_completion_wait(d2h)", error) ||
      result.state != SLLM_COMPLETION_STATE_SUCCESS) {
    (void)sllm_completion_release(&completion, &error.sink);
    return false;
  }
  uint64_t bytes_written = 0U;
  const bool read =
      expect(sllm_completion_read(completion, output->data(), output->size(),
                                  &bytes_written, &error.sink),
             SLLM_STATUS_OK, "sllm_completion_read", error) &&
      bytes_written == output->size();
  const bool released =
      expect(sllm_completion_release(&completion, &error.sink), SLLM_STATUS_OK,
             "sllm_completion_release(d2h)", error);
  return read && released && completion == nullptr;
}

template <typename T>
T load(const std::vector<uint8_t> &bytes, const uint64_t offset) {
  T result{};
  std::memcpy(&result, bytes.data() + offset, sizeof(result));
  return result;
}

std::vector<uint16_t> make_finite_logits(const uint64_t tokens) {
  std::vector<uint16_t> result(static_cast<std::size_t>(tokens * kExpertCount));
  for (uint64_t token = 0U; token != tokens; ++token) {
    const uint64_t mode = token % 4U;
    for (uint64_t expert = 0U; expert != kExpertCount; ++expert) {
      const int64_t bucket =
          static_cast<int64_t>((expert * 37U + token * 19U) % 113U) - 56;
      float value = static_cast<float>(bucket) / 12.0F;
      if (mode == 0U) {
        value = 0.0F;
      } else if (mode == 3U) {
        value = (static_cast<float>(expert) - 96.0F) * 0.09375F -
                static_cast<float>(expert % 7U) * 0.03125F;
      }
      result[static_cast<std::size_t>(token * kExpertCount + expert)] =
          f32_to_bf16_rne(value);
    }
    const std::size_t row = static_cast<std::size_t>(token * kExpertCount);
    if (mode == 1U) {
      result[row] = f32_to_bf16_rne(10.0F);
      result[row + 1U] = f32_to_bf16_rne(8.0F);
      result[row + 126U] = f32_to_bf16_rne(9.0F);
      result[row + 127U] = f32_to_bf16_rne(12.0F);
    } else if (mode == 2U) {
      result[row] = f32_to_bf16_rne(11.0F);
      result[row + 127U] = f32_to_bf16_rne(11.0F);
      result[row + 3U] = f32_to_bf16_rne(10.0F);
      result[row + 64U] = f32_to_bf16_rne(10.0F);
    }
  }
  return result;
}

Oracle make_oracle(const std::vector<uint16_t> &logits, const uint64_t tokens) {
  Oracle result;
  const uint64_t pair_count = tokens * kSelectedCount;
  result.ids.resize(static_cast<std::size_t>(pair_count));
  result.weights.resize(static_cast<std::size_t>(pair_count));
  result.counts.assign(static_cast<std::size_t>(kExpertCount), 0);
  result.offsets.assign(static_cast<std::size_t>(kExpertCount + 1U), 0);
  result.grouped_tokens.resize(static_cast<std::size_t>(pair_count));
  result.grouped_slots.resize(static_cast<std::size_t>(pair_count));
  for (uint64_t token = 0U; token != tokens; ++token) {
    std::vector<int32_t> order(static_cast<std::size_t>(kExpertCount));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(
        order.begin(), order.end(),
        [&](const int32_t left, const int32_t right) {
          const float left_value = bf16_to_f32(logits[static_cast<std::size_t>(
              token * kExpertCount + static_cast<uint32_t>(left))]);
          const float right_value = bf16_to_f32(logits[static_cast<std::size_t>(
              token * kExpertCount + static_cast<uint32_t>(right))]);
          return left_value == right_value ? left < right
                                           : left_value > right_value;
        });
    float maximum = -std::numeric_limits<float>::infinity();
    for (uint64_t expert = 0U; expert != kExpertCount; ++expert) {
      maximum = std::max(
          maximum,
          bf16_to_f32(
              logits[static_cast<std::size_t>(token * kExpertCount + expert)]));
    }
    float denominator = 0.0F;
    for (uint64_t expert = 0U; expert != kExpertCount; ++expert) {
      denominator += std::exp(
          bf16_to_f32(
              logits[static_cast<std::size_t>(token * kExpertCount + expert)]) -
          maximum);
    }
    float selected_sum = 0.0F;
    for (uint32_t slot = 0U; slot != kSelectedCount; ++slot) {
      const uint64_t pair = token * kSelectedCount + slot;
      result.ids[static_cast<std::size_t>(pair)] = order[slot];
      result.weights[static_cast<std::size_t>(pair)] =
          std::exp(
              bf16_to_f32(logits[static_cast<std::size_t>(
                  token * kExpertCount + static_cast<uint32_t>(order[slot]))]) -
              maximum) /
          denominator;
      selected_sum += result.weights[static_cast<std::size_t>(pair)];
    }
    for (uint32_t slot = 0U; slot != kSelectedCount; ++slot) {
      result.weights[static_cast<std::size_t>(token * kSelectedCount + slot)] /=
          selected_sum;
    }
  }
  int32_t cursor = 0;
  for (int32_t expert = 0; expert != static_cast<int32_t>(kExpertCount);
       ++expert) {
    result.offsets[static_cast<std::size_t>(expert)] = cursor;
    for (uint64_t token = 0U; token != tokens; ++token) {
      for (uint32_t slot = 0U; slot != kSelectedCount; ++slot) {
        const uint64_t pair = token * kSelectedCount + slot;
        if (result.ids[static_cast<std::size_t>(pair)] == expert) {
          result.grouped_tokens[static_cast<std::size_t>(cursor)] =
              static_cast<int32_t>(token);
          result.grouped_slots[static_cast<std::size_t>(cursor)] =
              static_cast<int32_t>(slot);
          ++cursor;
          ++result.counts[static_cast<std::size_t>(expert)];
        }
      }
    }
  }
  result.offsets[static_cast<std::size_t>(kExpertCount)] = cursor;
  return result;
}

bool validate_dispatch(const sllm_moe_route_dispatch_info_t &info,
                       const uint64_t tokens, const Layout &layout) {
  const bool valid =
      info.backend == SLLM_BACKEND_HIP && info.dispatch_id != 0U &&
      info.dispatch_count == 2U &&
      info.kernel_id == SLLM_HIP_MOE_ROUTE_KERNEL_ID_STABLE_TOPK_V1 &&
      info.workgroup_size_x == SLLM_HIP_MOE_ROUTE_WORKGROUP_SIZE &&
      info.grid_size_x == tokens && info.token_count == tokens &&
      info.expert_count == kExpertCount &&
      info.pair_count == layout.pair_count &&
      info.selected_expert_count == kSelectedCount &&
      info.fallback_allowed == 0U && info.fallback_used == 0U &&
      std::strcmp(info.kernel_symbol, "moe_route.bf16.stable_topk_group.v1") ==
          0 &&
      std::strcmp(info.device_symbol, "sllm_moe_route_stable_topk_group_v1") ==
          0 &&
      std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;
  if (!valid) {
    std::cerr << "route dispatch mismatch: id=" << info.dispatch_id
              << " count=" << info.dispatch_count
              << " kernel=" << info.kernel_id << " grid=" << info.grid_size_x
              << " tokens=" << info.token_count
              << " experts=" << info.expert_count
              << " pairs=" << info.pair_count
              << " fallback_allowed=" << info.fallback_allowed
              << " fallback_used=" << info.fallback_used
              << " target=" << info.gcn_arch_name << '\n';
  }
  return valid;
}

bool run_route(const sllm_context_t *const context,
               const sllm_queue_t *const queue,
               const std::vector<uint16_t> &logits, const uint64_t tokens,
               std::vector<uint8_t> *const output, Evidence *const evidence) {
  const Layout layout = make_layout(tokens);
  if (!validate_layout_contract(layout, tokens)) {
    std::cerr << "host route layout contract mismatch for M=" << tokens << '\n';
    return false;
  }
  const uint64_t logits_bytes = logits.size() * sizeof(uint16_t);
  std::array<sllm_buffer_t *, 2> buffers{};
  bool success = create_buffer(context, logits_bytes, &buffers[0]) &&
                 create_buffer(context, layout.bytes, &buffers[1]) &&
                 upload(queue, buffers[0], logits.data(), logits_bytes);
  sllm_moe_route_plan_t *plan = nullptr;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (success) {
    sllm_moe_route_desc_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.op_version = SLLM_HIP_MOE_ROUTE_VERSION;
    descriptor.selected_expert_count = kSelectedCount;
    descriptor.logits =
        binding(buffers[0], SLLM_TENSOR_DTYPE_BF16, 2U, tokens, kExpertCount);
    descriptor.metadata =
        binding(buffers[1], SLLM_TENSOR_DTYPE_U8, 1U, layout.bytes);
    success =
        expect(sllm_moe_route_prepare(context, &descriptor, &plan, &error.sink),
               SLLM_STATUS_OK, "sllm_moe_route_prepare", error) &&
        plan != nullptr;
  }
  sllm_moe_route_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_MOE_ROUTE_DISPATCH_INFO_VERSION;
  if (success) {
    success = expect(sllm_moe_route_execute(plan, queue, &completion, &info,
                                            &error.sink),
                     SLLM_STATUS_OK, "sllm_moe_route_execute", error) &&
              wait_and_release(&completion, "sllm_completion_wait(route)") &&
              validate_dispatch(info, tokens, layout);
    if (success) {
      evidence->dispatches += info.dispatch_count;
    }
  }
  output->assign(static_cast<std::size_t>(layout.bytes), 0U);
  if (success) {
    success = download(queue, buffers[1], output);
  }
  if (plan != nullptr) {
    success = expect(sllm_moe_route_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "sllm_moe_route_plan_release", error) &&
              success;
  }
  for (auto iterator = buffers.rbegin(); iterator != buffers.rend();
       ++iterator) {
    if (*iterator != nullptr) {
      success = expect(sllm_buffer_release(&*iterator, &error.sink),
                       SLLM_STATUS_OK, "sllm_buffer_release", error) &&
                success;
    }
  }
  const uint64_t live_handles =
      static_cast<uint64_t>(plan != nullptr) +
      static_cast<uint64_t>(completion != nullptr) +
      static_cast<uint64_t>(std::count_if(
          buffers.begin(), buffers.end(),
          [](const sllm_buffer_t *const buffer) { return buffer != nullptr; }));
  if (live_handles != 0U) {
    std::cerr << "route case cleanup retained " << live_handles
              << " handle(s)\n";
    success = false;
  }
  return success;
}

bool validate_finite(const std::vector<uint8_t> &output, const Layout &layout,
                     const Oracle &oracle, const uint64_t tokens,
                     Evidence *const evidence) {
  bool success = load<int32_t>(output, layout.status) == 0;
  evidence->metadata_mismatches += success ? 0U : 1U;
  for (uint64_t pair = 0U; pair != layout.pair_count; ++pair) {
    const int32_t actual_id =
        load<int32_t>(output, layout.ids + pair * sizeof(int32_t));
    const float actual_weight =
        load<float>(output, layout.weights + pair * sizeof(float));
    const int32_t expected_id = oracle.ids[static_cast<std::size_t>(pair)];
    const float expected_weight =
        oracle.weights[static_cast<std::size_t>(pair)];
    if (actual_id != expected_id) {
      ++evidence->metadata_mismatches;
      success = false;
    }
    const double abs_error =
        std::abs(static_cast<double>(actual_weight) - expected_weight);
    const double denominator =
        std::max(std::abs(static_cast<double>(expected_weight)), 1.0e-30);
    evidence->max_weight_abs_error =
        std::max(evidence->max_weight_abs_error, abs_error);
    evidence->max_weight_rel_error =
        std::max(evidence->max_weight_rel_error, abs_error / denominator);
    if (abs_error > 2.0e-6) {
      ++evidence->weight_mismatches;
      success = false;
    }
    evidence->selected_expert_zero =
        evidence->selected_expert_zero || actual_id == 0;
    evidence->selected_expert_last =
        evidence->selected_expert_last || actual_id == 127;
  }
  for (uint64_t token = 0U; token != tokens; ++token) {
    double selected_sum = 0.0;
    for (uint32_t slot = 0U; slot != kSelectedCount; ++slot) {
      const uint64_t pair = token * kSelectedCount + slot;
      selected_sum +=
          load<float>(output, layout.weights + pair * sizeof(float));
    }
    evidence->max_selected_sum_error = std::max(
        evidence->max_selected_sum_error, std::abs(selected_sum - 1.0));
  }
  for (uint64_t expert = 0U; expert != kExpertCount; ++expert) {
    const int32_t actual_count =
        load<int32_t>(output, layout.counts + expert * sizeof(int32_t));
    const int32_t actual_offset =
        load<int32_t>(output, layout.offsets + expert * sizeof(int32_t));
    if (actual_count != oracle.counts[static_cast<std::size_t>(expert)] ||
        actual_offset != oracle.offsets[static_cast<std::size_t>(expert)]) {
      ++evidence->metadata_mismatches;
      success = false;
    }
  }
  const int32_t final_offset =
      load<int32_t>(output, layout.offsets + kExpertCount * sizeof(int32_t));
  if (final_offset != oracle.offsets[static_cast<std::size_t>(kExpertCount)]) {
    ++evidence->metadata_mismatches;
    success = false;
  }
  for (uint64_t pair = 0U; pair != layout.pair_count; ++pair) {
    const int32_t actual_token =
        load<int32_t>(output, layout.grouped_tokens + pair * sizeof(int32_t));
    const int32_t actual_slot =
        load<int32_t>(output, layout.grouped_slots + pair * sizeof(int32_t));
    if (actual_token != oracle.grouped_tokens[static_cast<std::size_t>(pair)] ||
        actual_slot != oracle.grouped_slots[static_cast<std::size_t>(pair)]) {
      ++evidence->metadata_mismatches;
      success = false;
    }
  }
  for (uint64_t token = 0U; token < tokens; token += 4U) {
    for (uint32_t slot = 0U; slot != kSelectedCount; ++slot) {
      const int32_t actual_id =
          load<int32_t>(output, layout.ids + (token * kSelectedCount + slot) *
                                                 sizeof(int32_t));
      if (actual_id != static_cast<int32_t>(slot)) {
        ++evidence->metadata_mismatches;
        success = false;
      }
    }
  }
  ++evidence->finite_cases;
  return success;
}

bool run_finite_case(const sllm_context_t *const context,
                     const sllm_queue_t *const queue, const uint64_t tokens,
                     Evidence *const evidence) {
  const std::vector<uint16_t> logits = make_finite_logits(tokens);
  const Oracle oracle = make_oracle(logits, tokens);
  const Layout layout = make_layout(tokens);
  std::vector<uint8_t> output;
  const bool success =
      run_route(context, queue, logits, tokens, &output, evidence) &&
      validate_finite(output, layout, oracle, tokens, evidence);
  if (!success) {
    std::cerr << "finite Gemma4 route oracle mismatch for M=" << tokens << '\n';
  }
  return success;
}

bool validate_nonfinite(const std::vector<uint8_t> &output,
                        const Layout &layout, Evidence *const evidence) {
  bool success = load<int32_t>(output, layout.status) == 1;
  for (uint64_t pair = 0U; pair != layout.pair_count; ++pair) {
    const int32_t id =
        load<int32_t>(output, layout.ids + pair * sizeof(int32_t));
    const float weight =
        load<float>(output, layout.weights + pair * sizeof(float));
    const int32_t grouped_token =
        load<int32_t>(output, layout.grouped_tokens + pair * sizeof(int32_t));
    const int32_t grouped_slot =
        load<int32_t>(output, layout.grouped_slots + pair * sizeof(int32_t));
    success = success && id == -1 && std::isnan(weight) &&
              grouped_token == -1 && grouped_slot == -1;
  }
  for (uint64_t expert = 0U; expert != kExpertCount; ++expert) {
    success =
        success &&
        load<int32_t>(output, layout.counts + expert * sizeof(int32_t)) == 0 &&
        load<int32_t>(output, layout.offsets + expert * sizeof(int32_t)) == 0;
  }
  success = success &&
            load<int32_t>(output,
                          layout.offsets + kExpertCount * sizeof(int32_t)) == 0;
  if (!success) {
    ++evidence->metadata_mismatches;
  }
  ++evidence->nonfinite_cases;
  return success;
}

bool run_nonfinite_case(const sllm_context_t *const context,
                        const sllm_queue_t *const queue,
                        const uint16_t nonfinite, const uint64_t expert,
                        const char *const label, Evidence *const evidence) {
  std::vector<uint16_t> logits = make_finite_logits(1U);
  logits[static_cast<std::size_t>(expert)] = nonfinite;
  const Layout layout = make_layout(1U);
  std::vector<uint8_t> output;
  const bool success =
      run_route(context, queue, logits, 1U, &output, evidence) &&
      validate_nonfinite(output, layout, evidence);
  if (!success) {
    std::cerr << "nonfinite Gemma4 route fail-closed mismatch for " << label
              << '\n';
  }
  return success;
}

} // namespace

int main(const int argc, char **const argv) {
  if (argc > 2) {
    std::cerr << "usage: " << argv[0] << " [visible-device-index]\n";
    return 2;
  }
  int device_index = 0;
  if (argc == 2 && !parse_index(argv[1], &device_index)) {
    std::cerr << "device index must be a non-negative decimal integer\n";
    return 2;
  }
  Device device;
  if (!select_device(device_index, &device)) {
    return 1;
  }
  const std::string actual_arch(device.properties.gcnArchName);
  const char *const visible_devices = std::getenv("HIP_VISIBLE_DEVICES");
  std::cout << "device index=" << device.index
            << " visible_count=" << device.count << " name=\""
            << device.properties.name << "\" arch=" << actual_arch
            << " expected=" << SLLM_TEST_EXPECTED_TARGET
            << " HIP_VISIBLE_DEVICES=\""
            << (visible_devices == nullptr ? "" : visible_devices) << "\"\n";
  if (actual_arch != SLLM_TEST_EXPECTED_TARGET || device.index != 0 ||
      device.count != 1) {
    std::cerr << "exact single visible GPU contract mismatch\n";
    return 1;
  }
  if (!hip_expect(hipDeviceSynchronize(), hipSuccess, "hipDeviceSynchronize")) {
    return 1;
  }
  std::size_t free_before = 0U;
  std::size_t total_before = 0U;
  if (!hip_expect(hipMemGetInfo(&free_before, &total_before), hipSuccess,
                  "hipMemGetInfo(before)")) {
    return 1;
  }

  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::strncpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
               sizeof(context_info.expected_gcn_arch_name) - 1U);
  sllm_context_t *context = nullptr;
  Error error;
  if (!expect(sllm_context_create(&context_info, &context, &error.sink),
              SLLM_STATUS_OK, "sllm_context_create", error) ||
      context == nullptr) {
    return 1;
  }
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  sllm_queue_t *queue = nullptr;
  bool success =
      expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
             SLLM_STATUS_OK, "sllm_queue_create", error) &&
      queue != nullptr;
  Evidence evidence;
  constexpr std::array<uint64_t, 8> token_counts{
      {1U, 3U, 7U, 8U, 17U, 31U, 32U, 33U}};
  for (const uint64_t tokens : token_counts) {
    if (success && !run_finite_case(context, queue, tokens, &evidence)) {
      success = false;
    }
  }
  if (success && !run_nonfinite_case(context, queue, UINT16_C(0x7fc1), 0U,
                                     "quiet NaN", &evidence)) {
    success = false;
  }
  if (success && !run_nonfinite_case(context, queue, UINT16_C(0x7f80), 127U,
                                     "+infinity", &evidence)) {
    success = false;
  }
  if (success && !run_nonfinite_case(context, queue, UINT16_C(0xff80), 64U,
                                     "-infinity", &evidence)) {
    success = false;
  }
  if (queue != nullptr) {
    success = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
                     "sllm_queue_release", error) &&
              success;
  }
  if (context != nullptr) {
    success = expect(sllm_context_release(&context, &error.sink),
                     SLLM_STATUS_OK, "sllm_context_release", error) &&
              success;
  }
  success = success && queue == nullptr && context == nullptr;
  if (!hip_expect(hipDeviceSynchronize(), hipSuccess,
                  "hipDeviceSynchronize(cleanup)")) {
    success = false;
  }
  std::size_t free_after = 0U;
  std::size_t total_after = 0U;
  const bool memory_read = hip_expect(hipMemGetInfo(&free_after, &total_after),
                                      hipSuccess, "hipMemGetInfo(after)");
  constexpr std::size_t kRuntimeCacheTolerance =
      static_cast<std::size_t>(512U) * 1024U * 1024U;
  const bool memory_cleanup =
      memory_read && total_before == total_after &&
      (free_after >= free_before ||
       free_before - free_after <= kRuntimeCacheTolerance);
  success = success && memory_cleanup && evidence.finite_cases == 8U &&
            evidence.nonfinite_cases == 3U &&
            evidence.metadata_mismatches == 0U &&
            evidence.weight_mismatches == 0U && evidence.dispatches == 22U &&
            evidence.selected_expert_zero && evidence.selected_expert_last &&
            evidence.max_selected_sum_error <= 2.0e-6;
  if (success) {
    std::cout << "phase55 Gemma4 moe_route GPU PASS target="
              << SLLM_TEST_EXPECTED_TARGET << " device_index=" << device.index
              << " device_name=\"" << device.properties.name
              << "\" tokens=1,3,7,8,17,31,32,33 experts=128 topk=8"
              << " finite_cases=" << evidence.finite_cases
              << " nonfinite_fail_closed=3 stable_tie_lower_id=1"
              << " selected_expert0=" << evidence.selected_expert_zero
              << " selected_expert127=" << evidence.selected_expert_last
              << " metadata_mismatches=" << evidence.metadata_mismatches
              << " weight_mismatches=" << evidence.weight_mismatches
              << " max_weight_abs_error=" << evidence.max_weight_abs_error
              << " max_weight_rel_error=" << evidence.max_weight_rel_error
              << " max_selected_sum_error=" << evidence.max_selected_sum_error
              << " dispatches=" << evidence.dispatches
              << " fallback_allowed=0 fallback_used=0 cleanup=0"
              << " free_before=" << free_before << " free_after=" << free_after
              << " kernel=moe_route.bf16.stable_topk_group.v1"
              << " symbol=sllm_moe_route_stable_topk_group_v1\n";
  } else {
    std::cerr << "phase55 Gemma4 moe_route GPU FAIL target="
              << SLLM_TEST_EXPECTED_TARGET
              << " finite_cases=" << evidence.finite_cases
              << " nonfinite_cases=" << evidence.nonfinite_cases
              << " metadata_mismatches=" << evidence.metadata_mismatches
              << " weight_mismatches=" << evidence.weight_mismatches
              << " max_weight_abs_error=" << evidence.max_weight_abs_error
              << " max_weight_rel_error=" << evidence.max_weight_rel_error
              << " max_selected_sum_error=" << evidence.max_selected_sum_error
              << " memory_cleanup=" << memory_cleanup << '\n';
  }
  return success ? 0 : 1;
}
