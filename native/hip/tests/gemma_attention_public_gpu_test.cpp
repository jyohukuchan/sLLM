#include "sllm/hip.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <limits>
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

struct Case final {
  const char *name;
  uint32_t query_count;
  uint64_t start_position;
  uint32_t q_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint64_t sliding_window;
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

float bf16_to_f32(const uint16_t raw) {
  const uint32_t bits = static_cast<uint32_t>(raw) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
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

sllm_tensor_binding_t
tensor_binding(const sllm_buffer_t *const buffer, const uint32_t dtype,
               const std::initializer_list<uint64_t> shape) {
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.dtype = dtype;
  result.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  result.rank = static_cast<uint32_t>(shape.size());
  uint64_t stride = 1U;
  std::size_t index = shape.size();
  for (auto iterator = shape.end(); iterator != shape.begin();) {
    --iterator;
    --index;
    result.shape[index] = *iterator;
    result.stride_elements[index] = stride;
    stride *= *iterator;
  }
  return result;
}

bool wait_and_release(sllm_completion_t **const completion,
                      const char *const operation) {
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(
          sllm_completion_wait(*completion, UINT32_MAX, &result, &error.sink),
          SLLM_STATUS_OK, operation, error) ||
      result.state != SLLM_COMPLETION_STATE_SUCCESS) {
    return false;
  }
  return expect(sllm_completion_release(completion, &error.sink),
                SLLM_STATUS_OK, "sllm_completion_release", error);
}

bool create_buffer(const sllm_context_t *const context, const uint64_t bytes,
                   sllm_buffer_t **const output) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = bytes;
  Error error;
  return expect(sllm_buffer_create(context, &info, output, &error.sink),
                SLLM_STATUS_OK, "sllm_buffer_create", error);
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
              std::vector<uint16_t> *const output) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.size_bytes = output->size() * sizeof(uint16_t);
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_buffer_copy_d2h(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, "sllm_buffer_copy_d2h", error)) {
    return false;
  }
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(
          sllm_completion_wait(completion, UINT32_MAX, &result, &error.sink),
          SLLM_STATUS_OK, "sllm_completion_wait(d2h)", error)) {
    return false;
  }
  uint64_t bytes_written = 0U;
  const bool read = expect(sllm_completion_read(completion, output->data(),
                                                transfer.size_bytes,
                                                &bytes_written, &error.sink),
                           SLLM_STATUS_OK, "sllm_completion_read", error);
  const bool released =
      expect(sllm_completion_release(&completion, &error.sink), SLLM_STATUS_OK,
             "sllm_completion_release(d2h)", error);
  return read && bytes_written == transfer.size_bytes && released;
}

std::vector<uint16_t> make_input(const std::size_t count, const uint32_t salt) {
  std::vector<uint16_t> values(count);
  for (std::size_t index = 0U; index != count; ++index) {
    const uint64_t mixed =
        (static_cast<uint64_t>(index) * UINT64_C(53) + salt * 29U) % 257U;
    values[index] = f32_to_bf16_rne(
        static_cast<float>(static_cast<int32_t>(mixed) - 128) / 1024.0F);
  }
  return values;
}

std::vector<uint16_t> reference(const std::vector<uint16_t> &query,
                                const std::vector<uint16_t> &key,
                                const std::vector<uint16_t> &value,
                                const Case &test_case) {
  const uint64_t kv_length = test_case.start_position + test_case.query_count;
  std::vector<uint16_t> output(query.size());
  for (uint32_t row = 0U; row != test_case.query_count; ++row) {
    const uint64_t query_position = test_case.start_position + row;
    uint64_t first_key = 0U;
    if (test_case.sliding_window != 0U &&
        query_position + 1U > test_case.sliding_window) {
      first_key = query_position + 1U - test_case.sliding_window;
    }
    const std::size_t key_count =
        static_cast<std::size_t>(query_position - first_key + 1U);
    for (uint32_t query_head = 0U; query_head != test_case.q_heads;
         ++query_head) {
      const uint32_t kv_head =
          query_head / (test_case.q_heads / test_case.kv_heads);
      const std::size_t query_base =
          (static_cast<std::size_t>(row) * test_case.q_heads + query_head) *
          test_case.head_dim;
      std::vector<float> scores(key_count);
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t local_key = 0U; local_key != key_count; ++local_key) {
        const uint64_t key_position = first_key + local_key;
        const std::size_t key_base =
            (static_cast<std::size_t>(key_position) * test_case.kv_heads +
             kv_head) *
            test_case.head_dim;
        float score = 0.0F;
        for (uint32_t dimension = 0U; dimension != test_case.head_dim;
             ++dimension) {
          score += bf16_to_f32(query[query_base + dimension]) *
                   bf16_to_f32(key[key_base + dimension]);
        }
        scores[local_key] = score;
        maximum = std::max(maximum, score);
      }
      float denominator = 0.0F;
      for (float &score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      for (uint32_t dimension = 0U; dimension != test_case.head_dim;
           ++dimension) {
        float result = 0.0F;
        for (std::size_t local_key = 0U; local_key != key_count; ++local_key) {
          const uint64_t key_position = first_key + local_key;
          const std::size_t value_index =
              (static_cast<std::size_t>(key_position) * test_case.kv_heads +
               kv_head) *
                  test_case.head_dim +
              dimension;
          result += scores[local_key] * bf16_to_f32(value[value_index]);
        }
        output[query_base + dimension] = f32_to_bf16_rne(result / denominator);
      }
    }
  }
  (void)kv_length;
  return output;
}

bool compare(const std::vector<uint16_t> &actual,
             const std::vector<uint16_t> &expected, float *const max_abs,
             float *const max_scaled_rel) {
  constexpr float atol = 0.015625F;
  constexpr float rtol = 0.03125F;
  for (std::size_t index = 0U; index != actual.size(); ++index) {
    const float observed = bf16_to_f32(actual[index]);
    const float oracle = bf16_to_f32(expected[index]);
    const float absolute = std::abs(observed - oracle);
    const float scaled_relative = absolute / std::max(std::abs(oracle), atol);
    *max_abs = std::max(*max_abs, absolute);
    *max_scaled_rel = std::max(*max_scaled_rel, scaled_relative);
    if (!std::isfinite(observed) || absolute > atol + rtol * std::abs(oracle)) {
      std::cerr << "public attention mismatch at index " << index
                << ": actual=" << observed << " expected=" << oracle
                << " abs=" << absolute << " scaled_rel=" << scaled_relative
                << '\n';
      return false;
    }
  }
  return true;
}

bool run_case(const sllm_context_t *const context,
              const sllm_queue_t *const queue, const Case &test_case,
              float *const max_abs, float *const max_scaled_rel) {
  const uint64_t kv_length = test_case.start_position + test_case.query_count;
  const std::size_t query_elements =
      static_cast<std::size_t>(test_case.query_count) * test_case.q_heads *
      test_case.head_dim;
  const std::size_t kv_elements = static_cast<std::size_t>(kv_length) *
                                  test_case.kv_heads * test_case.head_dim;
  const std::vector<uint16_t> query = make_input(query_elements, 3U);
  const std::vector<uint16_t> key = make_input(kv_elements, 7U);
  const std::vector<uint16_t> value = make_input(kv_elements, 13U);
  const std::vector<uint16_t> oracle = reference(query, key, value, test_case);
  const uint64_t query_bytes = query_elements * sizeof(uint16_t);
  const uint64_t kv_bytes = kv_elements * sizeof(uint16_t);
  std::array<sllm_buffer_t *, 4> buffers{};
  bool success = create_buffer(context, query_bytes, &buffers[0]) &&
                 create_buffer(context, kv_bytes, &buffers[1]) &&
                 create_buffer(context, kv_bytes, &buffers[2]) &&
                 create_buffer(context, query_bytes, &buffers[3]) &&
                 upload(queue, buffers[0], query.data(), query_bytes) &&
                 upload(queue, buffers[1], key.data(), kv_bytes) &&
                 upload(queue, buffers[2], value.data(), kv_bytes);

  sllm_windowed_attention_plan_t *plan = nullptr;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (success) {
    sllm_windowed_attention_desc_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.op_version = SLLM_HIP_WINDOWED_ATTENTION_VERSION;
    descriptor.start_position = test_case.start_position;
    descriptor.expected_kv_length = kv_length;
    descriptor.sliding_window = test_case.sliding_window;
    descriptor.q_heads = test_case.q_heads;
    descriptor.kv_heads = test_case.kv_heads;
    descriptor.head_dim = test_case.head_dim;
    const float scaling = 1.0F;
    std::memcpy(&descriptor.scaling_bits, &scaling, sizeof(scaling));
    descriptor.query = tensor_binding(
        buffers[0], SLLM_TENSOR_DTYPE_BF16,
        {test_case.query_count, test_case.q_heads, test_case.head_dim});
    descriptor.key =
        tensor_binding(buffers[1], SLLM_TENSOR_DTYPE_BF16,
                       {kv_length, test_case.kv_heads, test_case.head_dim});
    descriptor.value =
        tensor_binding(buffers[2], SLLM_TENSOR_DTYPE_BF16,
                       {kv_length, test_case.kv_heads, test_case.head_dim});
    descriptor.output = tensor_binding(
        buffers[3], SLLM_TENSOR_DTYPE_BF16,
        {test_case.query_count, test_case.q_heads, test_case.head_dim});
    success = expect(sllm_windowed_attention_prepare(context, &descriptor,
                                                     &plan, &error.sink),
                     SLLM_STATUS_OK, "sllm_windowed_attention_prepare", error);
  }

  sllm_windowed_attention_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_WINDOWED_ATTENTION_DISPATCH_INFO_VERSION;
  success =
      success &&
      expect(sllm_windowed_attention_execute(plan, queue, &completion, &info,
                                             &error.sink),
             SLLM_STATUS_OK, "sllm_windowed_attention_execute", error) &&
      wait_and_release(&completion, "sllm_completion_wait(windowed_attention)");
  success =
      success && info.backend == SLLM_BACKEND_HIP &&
      info.dispatch_count == 1U &&
      info.kernel_id ==
          SLLM_HIP_WINDOWED_ATTENTION_KERNEL_ID_ONLINE_SOFTMAX_GQA_BF16_V1 &&
      info.workgroup_size_x == SLLM_HIP_WINDOWED_ATTENTION_WORKGROUP_SIZE &&
      info.grid_size_x == test_case.query_count * test_case.q_heads &&
      info.query_count == test_case.query_count &&
      info.start_position == test_case.start_position &&
      info.committed_kv_length == kv_length &&
      info.sliding_window == test_case.sliding_window &&
      info.q_heads == test_case.q_heads &&
      info.kv_heads == test_case.kv_heads &&
      info.head_dim == test_case.head_dim &&
      info.scaling_bits == UINT32_C(0x3f800000) &&
      info.fallback_allowed == 0U && info.fallback_used == 0U &&
      std::strcmp(info.kernel_symbol,
                  "gemma_causal_attention.online_softmax_gqa_bf16.v1") == 0 &&
      std::strcmp(info.device_symbol,
                  "sllm_gemma_causal_attention_online_softmax_gqa_bf16_v1") ==
          0 &&
      std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;

  std::vector<uint16_t> output(query_elements);
  success = success && download(queue, buffers[3], &output) &&
            compare(output, oracle, max_abs, max_scaled_rel);
  if (plan != nullptr) {
    success =
        expect(sllm_windowed_attention_plan_release(&plan, &error.sink),
               SLLM_STATUS_OK, "sllm_windowed_attention_plan_release", error) &&
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
  if (!success) {
    std::cerr << "public attention case failed: " << test_case.name << '\n';
  }
  return success;
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
              SLLM_STATUS_OK, "sllm_context_create", error)) {
    return 1;
  }
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  sllm_queue_t *queue = nullptr;
  if (!expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
              SLLM_STATUS_OK, "sllm_queue_create", error)) {
    return 1;
  }

  constexpr std::array<Case, 5> cases{{
      {"nonaligned-m3", 3U, 2U, 3U, 1U, 6U, 4U},
      {"sliding-short", 1U, 0U, 16U, 8U, 256U, 1'024U},
      {"sliding-window-boundary", 3U, 1'022U, 16U, 8U, 256U, 1'024U},
      {"full-short", 3U, 17U, 16U, 1U, 512U, 0U},
      {"full-nonaligned", 17U, 240U, 16U, 1U, 512U, 0U},
  }};
  float max_abs = 0.0F;
  float max_scaled_rel = 0.0F;
  bool success = true;
  for (const Case &test_case : cases) {
    if (!run_case(context, queue, test_case, &max_abs, &max_scaled_rel)) {
      success = false;
      break;
    }
  }
  success = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
                   "sllm_queue_release", error) &&
            success;
  success = expect(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
                   "sllm_context_release", error) &&
            success;
  if (success) {
    std::cout << "public Gemma causal attention PASS target="
              << SLLM_TEST_EXPECTED_TARGET << " cases=" << cases.size()
              << " max_abs=" << max_abs << " max_scaled_rel=" << max_scaled_rel
              << " fallback=false kernel="
              << "gemma_causal_attention.online_softmax_gqa_bf16.v1"
              << " symbol="
              << "sllm_gemma_causal_attention_online_softmax_gqa_bf16_v1\n";
  }
  return success ? 0 : 1;
}
