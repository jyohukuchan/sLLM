#include "sllm/hip.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
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

bool expect(const sllm_status_t actual, const sllm_status_t expected,
            const char *const operation, const Error &error) {
  if (actual == expected) {
    return true;
  }
  std::cerr << operation << " returned " << actual << ", expected " << expected
            << ": " << error.message << '\n';
  return false;
}

float bf16_to_f32(const uint16_t bits) {
  const uint32_t raw = static_cast<uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

uint16_t f32_to_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += UINT32_C(0x7fff) + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

uint32_t f32_bits(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

sllm_tensor_binding_t tensor_binding(const sllm_buffer_t *const buffer,
                                     const uint32_t dtype,
                                     const std::array<uint64_t, 3> shape,
                                     const uint32_t rank) {
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.dtype = dtype;
  result.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  result.rank = rank;
  uint64_t stride = 1U;
  for (uint32_t index = rank; index != 0U; --index) {
    result.shape[index - 1U] = shape[index - 1U];
    result.stride_elements[index - 1U] = stride;
    stride *= shape[index - 1U];
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
             "sllm_completion_release", error);
  return read && released && bytes_written == transfer.size_bytes;
}

std::vector<uint16_t> reference(const std::vector<uint16_t> &input,
                                const std::vector<int32_t> &positions,
                                const uint32_t heads, const bool query,
                                const bool adjacent_pairing) {
  constexpr uint32_t head_dim = 128U;
  constexpr uint32_t half = 64U;
  constexpr uint32_t low = 20U;
  constexpr uint32_t high = 37U;
  std::vector<uint16_t> output(input.size());
  for (std::size_t token = 0U; token != positions.size(); ++token) {
    const uint32_t position = static_cast<uint32_t>(positions[token]);
    const uint32_t block = position / 16384U;
    const float scale =
        query ? 1.0F + 0.1F * std::log1p(static_cast<float>(block)) : 1.0F;
    for (uint32_t head = 0U; head != heads; ++head) {
      const std::size_t base =
          (token * heads + static_cast<std::size_t>(head)) * head_dim;
      for (uint32_t pair = 0U; pair != half; ++pair) {
        const float base_frequency =
            std::pow(1'000'000.0F, static_cast<float>(pair * 2U) / head_dim);
        const float extrapolated = 1.0F / base_frequency;
        const float interpolated = 1.0F / (16.0F * base_frequency);
        const float ramp = std::fmin(
            1.0F, std::fmax(0.0F, (static_cast<float>(pair) - low) /
                                      static_cast<float>(high - low)));
        const float angle =
            static_cast<float>(position) *
            (interpolated * ramp + extrapolated * (1.0F - ramp));
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const uint32_t left_index = adjacent_pairing ? pair * 2U : pair;
        const uint32_t right_index =
            adjacent_pairing ? left_index + 1U : half + pair;
        const float left = bf16_to_f32(input[base + left_index]);
        const float right = bf16_to_f32(input[base + right_index]);
        output[base + left_index] =
            f32_to_bf16_rne((left * cosine - right * sine) * scale);
        output[base + right_index] =
            f32_to_bf16_rne((right * cosine + left * sine) * scale);
      }
    }
  }
  return output;
}

bool close(const std::vector<uint16_t> &actual,
           const std::vector<uint16_t> &expected, const char *const name) {
  constexpr float tolerance = 0.03125F;
  for (std::size_t index = 0U; index != actual.size(); ++index) {
    const float observed = bf16_to_f32(actual[index]);
    const float oracle = bf16_to_f32(expected[index]);
    if (!std::isfinite(observed) ||
        std::abs(observed - oracle) >
            tolerance + tolerance * std::abs(oracle)) {
      std::cerr << name << " mismatch at " << index << ": actual=" << observed
                << " expected=" << oracle << '\n';
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  constexpr uint32_t token_count = 3U;
  constexpr uint32_t q_heads = 32U;
  constexpr uint32_t kv_heads = 8U;
  constexpr uint32_t head_dim = 128U;
  const std::vector<int32_t> positions{0, 16384, 262143};
  const std::size_t q_count =
      static_cast<std::size_t>(token_count) * q_heads * head_dim;
  const std::size_t k_count =
      static_cast<std::size_t>(token_count) * kv_heads * head_dim;
  std::vector<uint16_t> query(q_count);
  std::vector<uint16_t> key(k_count);
  for (std::size_t index = 0U; index != q_count; ++index) {
    query[index] = f32_to_bf16_rne(
        static_cast<float>(static_cast<int32_t>(index % 31U) - 15) / 16.0F);
  }
  for (std::size_t index = 0U; index != k_count; ++index) {
    key[index] = f32_to_bf16_rne(
        static_cast<float>(static_cast<int32_t>(index % 23U) - 11) / 16.0F);
  }
  Error error;
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  std::strncpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
               sizeof(context_info.expected_gcn_arch_name) - 1U);
  sllm_context_t *context = nullptr;
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
    (void)sllm_context_release(&context, &error.sink);
    return 1;
  }

  std::array<sllm_buffer_t *, 5> buffers{};
  bool success =
      create_buffer(context, q_count * sizeof(uint16_t), &buffers[0]) &&
      create_buffer(context, k_count * sizeof(uint16_t), &buffers[1]) &&
      create_buffer(context, positions.size() * sizeof(int32_t), &buffers[2]) &&
      create_buffer(context, q_count * sizeof(uint16_t), &buffers[3]) &&
      create_buffer(context, k_count * sizeof(uint16_t), &buffers[4]) &&
      upload(queue, buffers[0], query.data(), q_count * sizeof(uint16_t)) &&
      upload(queue, buffers[1], key.data(), k_count * sizeof(uint16_t)) &&
      upload(queue, buffers[2], positions.data(),
             positions.size() * sizeof(int32_t));

  if (success) {
    struct PairingCase final {
      uint32_t op_version;
      uint32_t kernel_id;
      bool adjacent_pairing;
      const char *name;
    };
    const std::array<PairingCase, 2> cases{{
        {SLLM_HIP_MINISTRAL3_YARN_VERSION,
         SLLM_HIP_MINISTRAL3_YARN_KERNEL_ID_BF16_SPLIT_HALF_QSCALE_V1, false,
         "split-half"},
        {SLLM_HIP_MINISTRAL3_YARN_ADJACENT_VERSION,
         SLLM_HIP_MINISTRAL3_YARN_KERNEL_ID_BF16_ADJACENT_QSCALE_V2, true,
         "adjacent"},
    }};
    for (const PairingCase &test_case : cases) {
      sllm_ministral3_yarn_plan_t *plan = nullptr;
      sllm_completion_t *completion = nullptr;
      sllm_ministral3_yarn_desc_t descriptor{};
      descriptor.struct_size = sizeof(descriptor);
      descriptor.abi_version = SLLM_HIP_ABI_VERSION;
      descriptor.op_version = test_case.op_version;
      descriptor.position_payload_mode =
          SLLM_HIP_POSITION_PAYLOAD_MODE_EXPLICIT_V1;
      descriptor.q_heads = q_heads;
      descriptor.kv_heads = kv_heads;
      descriptor.head_dim = head_dim;
      descriptor.rotary_dim = head_dim;
      descriptor.theta_bits = f32_bits(1'000'000.0F);
      descriptor.factor_bits = f32_bits(16.0F);
      descriptor.original_context = SLLM_HIP_MINISTRAL3_YARN_ORIGINAL_CONTEXT;
      descriptor.max_position = SLLM_HIP_MINISTRAL3_YARN_MAX_POSITION;
      descriptor.beta_fast_bits = f32_bits(32.0F);
      descriptor.beta_slow_bits = f32_bits(1.0F);
      descriptor.query_scale_beta_bits = f32_bits(0.1F);
      descriptor.query = tensor_binding(buffers[0], SLLM_TENSOR_DTYPE_BF16,
                                        {token_count, q_heads, head_dim}, 3U);
      descriptor.key = tensor_binding(buffers[1], SLLM_TENSOR_DTYPE_BF16,
                                      {token_count, kv_heads, head_dim}, 3U);
      descriptor.positions = tensor_binding(buffers[2], SLLM_TENSOR_DTYPE_I32,
                                            {token_count, 0U, 0U}, 1U);
      descriptor.query_output =
          tensor_binding(buffers[3], SLLM_TENSOR_DTYPE_BF16,
                         {token_count, q_heads, head_dim}, 3U);
      descriptor.key_output =
          tensor_binding(buffers[4], SLLM_TENSOR_DTYPE_BF16,
                         {token_count, kv_heads, head_dim}, 3U);
      bool case_success =
          expect(sllm_ministral3_yarn_prepare(context, &descriptor, &plan,
                                              &error.sink),
                 SLLM_STATUS_OK, "sllm_ministral3_yarn_prepare", error);
      sllm_ministral3_yarn_dispatch_info_t info{};
      info.struct_size = sizeof(info);
      info.abi_version = SLLM_HIP_ABI_VERSION;
      info.info_version = SLLM_HIP_MINISTRAL3_YARN_DISPATCH_INFO_VERSION;
      case_success =
          case_success &&
          expect(sllm_ministral3_yarn_execute(plan, queue, &completion, &info,
                                              &error.sink),
                 SLLM_STATUS_OK, "sllm_ministral3_yarn_execute", error) &&
          wait_and_release(&completion,
                           "sllm_completion_wait(ministral3_yarn)") &&
          info.backend == SLLM_BACKEND_HIP && info.dispatch_count == 1U &&
          info.kernel_id == test_case.kernel_id &&
          info.token_count == token_count && info.q_heads == q_heads &&
          info.kv_heads == kv_heads && info.head_dim == head_dim &&
          info.rotary_dim == head_dim && info.fallback_used == 0U;
      std::vector<uint16_t> query_output(q_count);
      std::vector<uint16_t> key_output(k_count);
      const std::vector<uint16_t> query_oracle = reference(
          query, positions, q_heads, true, test_case.adjacent_pairing);
      const std::vector<uint16_t> key_oracle = reference(
          key, positions, kv_heads, false, test_case.adjacent_pairing);
      case_success = case_success &&
                     download(queue, buffers[3], &query_output) &&
                     download(queue, buffers[4], &key_output) &&
                     close(query_output, query_oracle, test_case.name) &&
                     close(key_output, key_oracle, test_case.name);
      if (plan != nullptr) {
        case_success =
            expect(sllm_ministral3_yarn_plan_release(&plan, &error.sink),
                   SLLM_STATUS_OK, "sllm_ministral3_yarn_plan_release",
                   error) &&
            case_success;
      }
      success = success && case_success;
      if (!success) {
        break;
      }
    }
  }
  for (auto iterator = buffers.rbegin(); iterator != buffers.rend();
       ++iterator) {
    if (*iterator != nullptr) {
      success = expect(sllm_buffer_release(&*iterator, &error.sink),
                       SLLM_STATUS_OK, "sllm_buffer_release", error) &&
                success;
    }
  }
  success = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
                   "sllm_queue_release", error) &&
            success;
  success = expect(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
                   "sllm_context_release", error) &&
            success;
  if (success) {
    std::cout << "ministral3.yarn public ABI PASS\n";
  }
  return success ? 0 : 1;
}
