#include "sllm/hip.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
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
  uint32_t token_count;
  uint32_t start_position;
  uint32_t q_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t rotary_dim;
  float theta;
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
        (static_cast<uint64_t>(index) * UINT64_C(37) + salt * 19U) % 257U;
    values[index] = f32_to_bf16_rne(
        static_cast<float>(static_cast<int32_t>(mixed) - 128) / 96.0F);
  }
  return values;
}

std::vector<uint16_t> reference(const std::vector<uint16_t> &input,
                                const std::vector<int32_t> &positions,
                                const uint32_t heads, const uint32_t head_dim,
                                const uint32_t rotary_dim, const float theta) {
  std::vector<uint16_t> output = input;
  const uint32_t half = head_dim / 2U;
  const uint32_t active_pairs = rotary_dim / 2U;
  for (std::size_t token = 0U; token != positions.size(); ++token) {
    for (uint32_t head = 0U; head != heads; ++head) {
      const std::size_t base =
          (token * heads + static_cast<std::size_t>(head)) * head_dim;
      for (uint32_t pair = 0U; pair != active_pairs; ++pair) {
        const float exponent =
            -2.0F * static_cast<float>(pair) / static_cast<float>(head_dim);
        const float angle =
            static_cast<float>(positions[token]) * std::pow(theta, exponent);
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float left = bf16_to_f32(input[base + pair]);
        const float right = bf16_to_f32(input[base + half + pair]);
        output[base + pair] = f32_to_bf16_rne(left * cosine - right * sine);
        output[base + half + pair] =
            f32_to_bf16_rne(right * cosine + left * sine);
      }
    }
  }
  return output;
}

bool compare(const std::vector<uint16_t> &actual,
             const std::vector<uint16_t> &expected,
             const std::vector<uint16_t> &input, const uint32_t head_dim,
             const uint32_t rotary_dim, float *const max_abs,
             float *const max_rel) {
  constexpr float atol = 0.03125F;
  constexpr float rtol = 0.03125F;
  const uint32_t half = head_dim / 2U;
  const uint32_t active_pairs = rotary_dim / 2U;
  for (std::size_t index = 0U; index != actual.size(); ++index) {
    const uint32_t dimension = static_cast<uint32_t>(index % head_dim);
    const bool active = dimension < active_pairs ||
                        (dimension >= half && dimension < half + active_pairs);
    if (!active && actual[index] != input[index]) {
      std::cerr << "inactive rotary dimension changed at index " << index
                << '\n';
      return false;
    }
    const float observed = bf16_to_f32(actual[index]);
    const float oracle = bf16_to_f32(expected[index]);
    const float absolute = std::abs(observed - oracle);
    const float relative = absolute / std::max(std::abs(oracle), atol);
    *max_abs = std::max(*max_abs, absolute);
    *max_rel = std::max(*max_rel, relative);
    if (!std::isfinite(observed) || absolute > atol + rtol * std::abs(oracle)) {
      std::cerr << "rotary public ABI mismatch at index " << index
                << ": actual=" << observed << " expected=" << oracle
                << " abs=" << absolute << " rel=" << relative << '\n';
      return false;
    }
  }
  return true;
}

bool run_case(const sllm_context_t *const context,
              const sllm_queue_t *const queue, const Case &test_case,
              float *const max_abs, float *const max_rel) {
  const std::size_t q_count = static_cast<std::size_t>(test_case.token_count) *
                              test_case.q_heads * test_case.head_dim;
  const std::size_t k_count = static_cast<std::size_t>(test_case.token_count) *
                              test_case.kv_heads * test_case.head_dim;
  const std::vector<uint16_t> query = make_input(q_count, 3U);
  const std::vector<uint16_t> key = make_input(k_count, 11U);
  std::vector<int32_t> positions(test_case.token_count);
  for (uint32_t index = 0U; index != test_case.token_count; ++index) {
    positions[index] = static_cast<int32_t>(test_case.start_position + index);
  }
  const std::vector<uint16_t> query_oracle =
      reference(query, positions, test_case.q_heads, test_case.head_dim,
                test_case.rotary_dim, test_case.theta);
  const std::vector<uint16_t> key_oracle =
      reference(key, positions, test_case.kv_heads, test_case.head_dim,
                test_case.rotary_dim, test_case.theta);

  const uint64_t q_bytes = q_count * sizeof(uint16_t);
  const uint64_t k_bytes = k_count * sizeof(uint16_t);
  const uint64_t position_bytes = positions.size() * sizeof(int32_t);
  std::array<sllm_buffer_t *, 5> buffers{};
  bool success = create_buffer(context, q_bytes, &buffers[0]) &&
                 create_buffer(context, k_bytes, &buffers[1]) &&
                 create_buffer(context, position_bytes, &buffers[2]) &&
                 create_buffer(context, q_bytes, &buffers[3]) &&
                 create_buffer(context, k_bytes, &buffers[4]) &&
                 upload(queue, buffers[0], query.data(), q_bytes) &&
                 upload(queue, buffers[1], key.data(), k_bytes) &&
                 upload(queue, buffers[2], positions.data(), position_bytes);

  sllm_rotary_plan_t *plan = nullptr;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (success) {
    sllm_rotary_desc_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.op_version = SLLM_HIP_ROTARY_VERSION;
    descriptor.start_position = test_case.start_position;
    descriptor.q_heads = test_case.q_heads;
    descriptor.kv_heads = test_case.kv_heads;
    descriptor.head_dim = test_case.head_dim;
    descriptor.rotary_dim = test_case.rotary_dim;
    std::memcpy(&descriptor.theta_bits, &test_case.theta,
                sizeof(test_case.theta));
    descriptor.max_position = SLLM_HIP_ROTARY_MAX_POSITION;
    descriptor.query = tensor_binding(
        buffers[0], SLLM_TENSOR_DTYPE_BF16,
        {test_case.token_count, test_case.q_heads, test_case.head_dim});
    descriptor.key = tensor_binding(
        buffers[1], SLLM_TENSOR_DTYPE_BF16,
        {test_case.token_count, test_case.kv_heads, test_case.head_dim});
    descriptor.positions = tensor_binding(buffers[2], SLLM_TENSOR_DTYPE_I32,
                                          {test_case.token_count});
    descriptor.query_output = tensor_binding(
        buffers[3], SLLM_TENSOR_DTYPE_BF16,
        {test_case.token_count, test_case.q_heads, test_case.head_dim});
    descriptor.key_output = tensor_binding(
        buffers[4], SLLM_TENSOR_DTYPE_BF16,
        {test_case.token_count, test_case.kv_heads, test_case.head_dim});
    success =
        expect(sllm_rotary_prepare(context, &descriptor, &plan, &error.sink),
               SLLM_STATUS_OK, "sllm_rotary_prepare", error);
  }

  sllm_rotary_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_ROTARY_DISPATCH_INFO_VERSION;
  success =
      success &&
      expect(sllm_rotary_execute(plan, queue, &completion, &info, &error.sink),
             SLLM_STATUS_OK, "sllm_rotary_execute", error) &&
      wait_and_release(&completion, "sllm_completion_wait(rotary)");
  success =
      success && info.backend == SLLM_BACKEND_HIP &&
      info.dispatch_count == 1U &&
      info.kernel_id == SLLM_HIP_ROTARY_KERNEL_ID_SPLIT_HALF_BF16_FP32_V1 &&
      info.workgroup_size_x == SLLM_HIP_ROTARY_WORKGROUP_SIZE &&
      info.grid_size_x ==
          test_case.token_count * (test_case.q_heads + test_case.kv_heads) &&
      info.token_count == test_case.token_count &&
      info.q_heads == test_case.q_heads &&
      info.kv_heads == test_case.kv_heads &&
      info.head_dim == test_case.head_dim &&
      info.rotary_dim == test_case.rotary_dim &&
      info.start_position == test_case.start_position &&
      info.max_position == SLLM_HIP_ROTARY_MAX_POSITION &&
      info.fallback_allowed == 0U && info.fallback_used == 0U &&
      std::strcmp(info.kernel_symbol, "rotary.split_half.bf16_fp32.v1") == 0 &&
      std::strcmp(info.device_symbol, "sllm_rotary_split_half_bf16_fp32_v1") ==
          0 &&
      std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;

  std::vector<uint16_t> query_output(q_count);
  std::vector<uint16_t> key_output(k_count);
  success = success && download(queue, buffers[3], &query_output) &&
            download(queue, buffers[4], &key_output) &&
            compare(query_output, query_oracle, query, test_case.head_dim,
                    test_case.rotary_dim, max_abs, max_rel) &&
            compare(key_output, key_oracle, key, test_case.head_dim,
                    test_case.rotary_dim, max_abs, max_rel);

  if (plan != nullptr) {
    success = expect(sllm_rotary_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "sllm_rotary_plan_release", error) &&
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
    std::cerr << "public rotary case failed: " << test_case.name << '\n';
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

  constexpr std::array<Case, 7> cases{{
      {"non-aligned-m3-p255", 3U, 255U, 3U, 1U, 6U, 4U, 10'000.0F},
      {"sliding-m1-p0", 1U, 0U, 16U, 8U, 256U, 256U, 10'000.0F},
      {"sliding-m3-p255", 3U, 255U, 16U, 8U, 256U, 256U, 10'000.0F},
      {"sliding-m17-tail", 17U, 262'127U, 16U, 8U, 256U, 256U, 10'000.0F},
      {"full-m1-p0", 1U, 0U, 16U, 1U, 512U, 128U, 1'000'000.0F},
      {"full-m3-p255", 3U, 255U, 16U, 1U, 512U, 128U, 1'000'000.0F},
      {"full-m17-tail", 17U, 262'127U, 16U, 1U, 512U, 128U, 1'000'000.0F},
  }};
  float max_abs = 0.0F;
  float max_rel = 0.0F;
  bool success = true;
  for (const Case &test_case : cases) {
    if (!run_case(context, queue, test_case, &max_abs, &max_rel)) {
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
    std::cout << "public split-half rotary PASS target="
              << SLLM_TEST_EXPECTED_TARGET << " cases=" << cases.size()
              << " max_abs=" << max_abs << " max_scaled_rel=" << max_rel
              << " fallback=false kernel=rotary.split_half.bf16_fp32.v1"
              << " symbol=sllm_rotary_split_half_bf16_fp32_v1\n";
  }
  return success ? 0 : 1;
}
