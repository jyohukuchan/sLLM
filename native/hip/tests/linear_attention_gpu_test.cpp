#include "sllm/hip.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Error final {
  char message[256]{};
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

uint16_t float_to_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
    if ((bits & UINT32_C(0x007fffff)) != 0U) {
      return static_cast<uint16_t>(((bits >> 16U) & UINT32_C(0x8000)) |
                                   UINT32_C(0x7fc0));
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

float bf16_to_float(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

bool completion_wait(sllm_completion_t *const completion) {
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  Error error;
  return expect(sllm_completion_wait(completion, 5000U, &result, &error.sink),
                SLLM_STATUS_OK, "completion wait", error) &&
         result.state == SLLM_COMPLETION_STATE_SUCCESS;
}

bool completion_release(sllm_completion_t **const completion) {
  Error error;
  return expect(sllm_completion_release(completion, &error.sink),
                SLLM_STATUS_OK, "completion release", error);
}

bool upload(const sllm_queue_t *const queue, const sllm_buffer_t *const buffer,
            void *const data, const uint64_t bytes) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = data;
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  return expect(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                     &error.sink),
                SLLM_STATUS_OK, "input upload", error) &&
         completion != nullptr && completion_wait(completion) &&
         completion_release(&completion);
}

bool read_output(const sllm_queue_t *const queue,
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
              SLLM_STATUS_OK, "output readback submit", error) ||
      completion == nullptr || !completion_wait(completion)) {
    return false;
  }
  uint64_t written = 0U;
  const bool read =
      expect(sllm_completion_read(completion, output->data(),
                                  transfer.size_bytes, &written, &error.sink),
             SLLM_STATUS_OK, "output readback", error) &&
      written == transfer.size_bytes;
  return completion_release(&completion) && read;
}

sllm_tensor_binding_t binding(const sllm_buffer_t *const buffer,
                              const uint32_t dtype, const uint32_t rank,
                              const uint64_t *const shape) {
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.dtype = dtype;
  result.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  result.rank = rank;
  uint64_t stride = 1U;
  for (uint32_t backwards = 0U; backwards != rank; ++backwards) {
    const uint32_t index = rank - 1U - backwards;
    result.shape[index] = shape[index];
    result.stride_elements[index] = stride;
    stride *= shape[index];
  }
  return result;
}

struct Oracle final {
  std::vector<uint16_t> conv_history = std::vector<uint16_t>(3U * 8192U, 0U);
  std::vector<float> recurrent = std::vector<float>(32U * 128U * 128U, 0.0F);

  std::vector<uint16_t>
  run(const std::vector<uint16_t> &qkv, const std::vector<uint16_t> &z,
      const std::vector<uint16_t> &b, const std::vector<uint16_t> &a,
      const std::vector<uint16_t> &conv_weight, const std::vector<float> &a_log,
      const std::vector<uint16_t> &dt_bias,
      const std::vector<float> &norm_weight, const uint32_t token_count) {
    std::vector<uint16_t> convolved(static_cast<std::size_t>(token_count) *
                                    8192U);
    for (uint32_t token = 0U; token != token_count; ++token) {
      for (uint32_t channel = 0U; channel != 8192U; ++channel) {
        float sum = 0.0F;
        for (uint32_t tap = 0U; tap != 4U; ++tap) {
          const int64_t source =
              static_cast<int64_t>(token) + static_cast<int64_t>(tap) - 3;
          const uint16_t value =
              source < 0
                  ? conv_history[static_cast<std::size_t>(source + 3) * 8192U +
                                 channel]
                  : qkv[static_cast<std::size_t>(source) * 8192U + channel];
          sum += bf16_to_float(value) *
                 bf16_to_float(conv_weight[channel * 4U + tap]);
        }
        const float silu = sum / (1.0F + std::exp(-sum));
        convolved[static_cast<std::size_t>(token) * 8192U + channel] =
            float_to_bf16_rne(silu);
      }
    }
    std::vector<uint16_t> next_history(3U * 8192U);
    for (uint32_t row = 0U; row != 3U; ++row) {
      const int64_t source = static_cast<int64_t>(token_count) - 3 + row;
      for (uint32_t channel = 0U; channel != 8192U; ++channel) {
        next_history[static_cast<std::size_t>(row) * 8192U + channel] =
            source < 0
                ? conv_history[static_cast<std::size_t>(source + 3) * 8192U +
                               channel]
                : qkv[static_cast<std::size_t>(source) * 8192U + channel];
      }
    }
    conv_history = std::move(next_history);

    std::vector<uint16_t> output(static_cast<std::size_t>(token_count) * 4096U);
    std::array<float, 128> q_values{};
    std::array<float, 128> k_values{};
    std::array<float, 128> values{};
    for (uint32_t token = 0U; token != token_count; ++token) {
      const std::size_t qkv_row = static_cast<std::size_t>(token) * 8192U;
      for (uint32_t value_head = 0U; value_head != 32U; ++value_head) {
        const uint32_t qk_head = value_head / 2U;
        float q_sum = 0.0F;
        float k_sum = 0.0F;
        for (uint32_t dimension = 0U; dimension != 128U; ++dimension) {
          q_values[dimension] =
              bf16_to_float(convolved[qkv_row + qk_head * 128U + dimension]);
          k_values[dimension] = bf16_to_float(
              convolved[qkv_row + 2048U + qk_head * 128U + dimension]);
          q_sum += q_values[dimension] * q_values[dimension];
          k_sum += k_values[dimension] * k_values[dimension];
        }
        const float q_inverse = 1.0F / std::sqrt(q_sum + 1.0e-6F);
        const float k_inverse = 1.0F / std::sqrt(k_sum + 1.0e-6F);
        for (uint32_t dimension = 0U; dimension != 128U; ++dimension) {
          q_values[dimension] =
              bf16_to_float(float_to_bf16_rne(q_values[dimension] * q_inverse));
          k_values[dimension] =
              bf16_to_float(float_to_bf16_rne(k_values[dimension] * k_inverse));
          q_values[dimension] *= 1.0F / std::sqrt(128.0F);
        }
        const std::size_t scalar_index =
            static_cast<std::size_t>(token) * 32U + value_head;
        const float beta_f32 =
            1.0F / (1.0F + std::exp(-bf16_to_float(b[scalar_index])));
        const float beta = bf16_to_float(float_to_bf16_rne(beta_f32));
        const float a_value =
            bf16_to_float(a[scalar_index]) + bf16_to_float(dt_bias[value_head]);
        const float softplus = std::fmax(a_value, 0.0F) +
                               std::log1p(std::exp(-std::fabs(a_value)));
        const float decay = std::exp(-std::exp(a_log[value_head]) * softplus);
        for (uint32_t dimension = 0U; dimension != 128U; ++dimension) {
          const std::size_t state_row =
              (static_cast<std::size_t>(value_head) * 128U + dimension) * 128U;
          for (uint32_t key_dimension = 0U; key_dimension != 128U;
               ++key_dimension) {
            recurrent[state_row + key_dimension] *= decay;
          }
          float previous_projection = 0.0F;
          for (uint32_t key_dimension = 0U; key_dimension != 128U;
               ++key_dimension) {
            previous_projection +=
                recurrent[state_row + key_dimension] * k_values[key_dimension];
          }
          const float value = bf16_to_float(
              convolved[qkv_row + 4096U + value_head * 128U + dimension]);
          const float residual = value - previous_projection;
          float projection = 0.0F;
          for (uint32_t key_dimension = 0U; key_dimension != 128U;
               ++key_dimension) {
            const std::size_t index = state_row + key_dimension;
            recurrent[index] += beta * residual * k_values[key_dimension];
            projection += recurrent[index] * q_values[key_dimension];
          }
          values[dimension] = bf16_to_float(float_to_bf16_rne(projection));
        }
        float square_sum = 0.0F;
        for (const float value : values) {
          square_sum += value * value;
        }
        const float inverse_rms =
            1.0F / std::sqrt(square_sum / 128.0F + 1.0e-6F);
        for (uint32_t dimension = 0U; dimension != 128U; ++dimension) {
          const std::size_t output_index =
              static_cast<std::size_t>(token) * 4096U + value_head * 128U +
              dimension;
          const float z_value = bf16_to_float(z[output_index]);
          const float z_silu = z_value / (1.0F + std::exp(-z_value));
          const float normalized = values[dimension] * inverse_rms;
          const float normalized_bf16 =
              bf16_to_float(float_to_bf16_rne(normalized));
          output[output_index] = float_to_bf16_rne(
              normalized_bf16 * norm_weight[dimension] * z_silu);
        }
      }
    }
    return output;
  }
};

struct BoundaryOptions final {
  bool scale_query;
  bool round_qk;
  bool round_beta;
  bool round_core;
  bool round_normalized;
  bool decay_before_memory;
};

std::array<uint16_t, 128U>
run_scalar_boundary_oracle(const BoundaryOptions options) {
  constexpr uint32_t seed = 1U;
  constexpr uint32_t head_dim = 128U;
  constexpr float beta_input = 0.63F;
  constexpr float decay = 0.81F;
  std::array<float, head_dim> query{};
  std::array<float, head_dim> key{};
  std::array<float, head_dim> value{};
  std::array<float, head_dim * head_dim> recurrent{};
  for (uint32_t dimension = 0U; dimension != head_dim; ++dimension) {
    const int32_t query_base =
        static_cast<int32_t>((dimension * 37U + seed) % 101U) - 50;
    const int32_t key_base =
        static_cast<int32_t>((dimension * 53U + seed * 3U) % 97U) - 48;
    const int32_t value_base =
        static_cast<int32_t>((dimension * 29U + seed * 7U) % 83U) - 41;
    const int32_t query_offset = static_cast<int32_t>(dimension % 7U) - 3;
    const int32_t key_offset = static_cast<int32_t>(dimension % 5U) - 2;
    const int32_t value_offset = static_cast<int32_t>(dimension % 3U) - 1;
    query[dimension] = bf16_to_float(
        float_to_bf16_rne(static_cast<float>(query_base) * 0.013F +
                          static_cast<float>(query_offset) * 0.004F));
    key[dimension] = bf16_to_float(
        float_to_bf16_rne(static_cast<float>(key_base) * 0.011F +
                          static_cast<float>(key_offset) * 0.006F));
    value[dimension] = bf16_to_float(
        float_to_bf16_rne(static_cast<float>(value_base) * 0.021F +
                          static_cast<float>(value_offset) * 0.007F));
    for (uint32_t key_dimension = 0U; key_dimension != head_dim;
         ++key_dimension) {
      const int32_t state_base = static_cast<int32_t>(key_dimension % 11U) - 5;
      const int32_t state_offset = static_cast<int32_t>(key_dimension % 4U) - 2;
      recurrent[static_cast<std::size_t>(dimension) * head_dim +
                key_dimension] = static_cast<float>(state_base) * 0.003F +
                                 static_cast<float>(state_offset) * 0.0007F;
    }
  }

  float query_square_sum = 0.0F;
  float key_square_sum = 0.0F;
  for (uint32_t dimension = 0U; dimension != head_dim; ++dimension) {
    query_square_sum += query[dimension] * query[dimension];
    key_square_sum += key[dimension] * key[dimension];
  }
  const float query_inverse = 1.0F / std::sqrt(query_square_sum + 1.0e-6F);
  const float key_inverse = 1.0F / std::sqrt(key_square_sum + 1.0e-6F);
  const float query_scale =
      options.scale_query ? 1.0F / std::sqrt(128.0F) : 1.0F;
  for (uint32_t dimension = 0U; dimension != head_dim; ++dimension) {
    const float normalized_query = query[dimension] * query_inverse;
    const float normalized_key = key[dimension] * key_inverse;
    query[dimension] = options.round_qk
                           ? bf16_to_float(float_to_bf16_rne(normalized_query))
                           : normalized_query;
    key[dimension] = options.round_qk
                         ? bf16_to_float(float_to_bf16_rne(normalized_key))
                         : normalized_key;
    query[dimension] *= query_scale;
  }

  const float beta_f32 = 1.0F / (1.0F + std::exp(-beta_input));
  const float beta = options.round_beta
                         ? bf16_to_float(float_to_bf16_rne(beta_f32))
                         : beta_f32;

  std::array<float, head_dim> core{};
  for (uint32_t value_dimension = 0U; value_dimension != head_dim;
       ++value_dimension) {
    const std::size_t state_row =
        static_cast<std::size_t>(value_dimension) * head_dim;
    if (options.decay_before_memory) {
      for (uint32_t key_dimension = 0U; key_dimension != head_dim;
           ++key_dimension) {
        recurrent[state_row + key_dimension] *= decay;
      }
    }
    float kv_memory = 0.0F;
    for (uint32_t key_dimension = 0U; key_dimension != head_dim;
         ++key_dimension) {
      kv_memory += recurrent[state_row + key_dimension] * key[key_dimension];
    }
    const float residual = value[value_dimension] - kv_memory;
    if (!options.decay_before_memory) {
      for (uint32_t key_dimension = 0U; key_dimension != head_dim;
           ++key_dimension) {
        recurrent[state_row + key_dimension] *= decay;
      }
    }
    for (uint32_t key_dimension = 0U; key_dimension != head_dim;
         ++key_dimension) {
      recurrent[state_row + key_dimension] +=
          beta * residual * key[key_dimension];
    }
    float projection = 0.0F;
    for (uint32_t key_dimension = 0U; key_dimension != head_dim;
         ++key_dimension) {
      projection += recurrent[state_row + key_dimension] * query[key_dimension];
    }
    core[value_dimension] = options.round_core
                                ? bf16_to_float(float_to_bf16_rne(projection))
                                : projection;
  }

  float core_square_sum = 0.0F;
  for (const float core_value : core) {
    core_square_sum += core_value * core_value;
  }
  const float inverse_rms =
      1.0F / std::sqrt(core_square_sum / 128.0F + 1.0e-6F);
  std::array<uint16_t, head_dim> result{};
  for (uint32_t dimension = 0U; dimension != head_dim; ++dimension) {
    const float normalized = core[dimension] * inverse_rms;
    const float normalized_value =
        options.round_normalized ? bf16_to_float(float_to_bf16_rne(normalized))
                                 : normalized;
    const int32_t z_offset =
        static_cast<int32_t>((dimension * 17U + seed) % 19U) - 9;
    const float z_input = 0.2F + static_cast<float>(z_offset) * 0.025F;
    const float z_value = bf16_to_float(float_to_bf16_rne(z_input));
    const float z_silu = z_value / (1.0F + std::exp(-z_value));
    const float norm_weight =
        0.7F + static_cast<float>(dimension % 11U) * 0.03F;
    result[dimension] =
        float_to_bf16_rne(normalized_value * norm_weight * z_silu);
  }
  return result;
}

bool check_scalar_boundary_regressions() {
  const BoundaryOptions exact{true, true, true, true, true, true};
  const auto expected = run_scalar_boundary_oracle(exact);
  const auto without_query_scale = run_scalar_boundary_oracle(
      BoundaryOptions{false, true, true, true, true, true});
  const auto without_qk_round = run_scalar_boundary_oracle(
      BoundaryOptions{true, false, true, true, true, true});
  const auto without_beta_round = run_scalar_boundary_oracle(
      BoundaryOptions{true, true, false, true, true, true});
  const auto without_core_round = run_scalar_boundary_oracle(
      BoundaryOptions{true, true, true, false, true, true});
  const auto without_normalized_round = run_scalar_boundary_oracle(
      BoundaryOptions{true, true, true, true, false, true});
  const auto wrong_decay_order = run_scalar_boundary_oracle(
      BoundaryOptions{true, true, true, true, true, false});
  const auto differs = [&expected](const std::array<uint16_t, 128U> &actual) {
    return actual != expected;
  };
  if (!differs(without_query_scale) || !differs(without_qk_round) ||
      !differs(without_beta_round) || !differs(without_core_round) ||
      !differs(without_normalized_round) || !differs(wrong_decay_order)) {
    std::cerr << "scalar GDN boundary regression did not distinguish an "
                 "omitted semantic boundary\n";
    return false;
  }
  return true;
}

uint32_t bf16_ulp_distance(const uint16_t left, const uint16_t right) {
  const auto ordered = [](const uint16_t value) {
    return (value & UINT16_C(0x8000)) != 0U
               ? static_cast<uint16_t>(~value)
               : static_cast<uint16_t>(value | UINT16_C(0x8000));
  };
  const uint32_t left_ordered = ordered(left);
  const uint32_t right_ordered = ordered(right);
  return left_ordered > right_ordered ? left_ordered - right_ordered
                                      : right_ordered - left_ordered;
}

} // namespace

int main() {
  constexpr uint64_t max_tokens = 3U;
  constexpr uint64_t capacity = 4U;
  if (!check_scalar_boundary_regressions()) {
    return 1;
  }
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::memcpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
              sizeof(SLLM_TEST_EXPECTED_TARGET));
  Error error;
  sllm_context_t *context = nullptr;
  if (!expect(sllm_context_create(&context_info, &context, &error.sink),
              SLLM_STATUS_OK, "context create", error)) {
    return 1;
  }
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  sllm_queue_t *queue = nullptr;
  if (!expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
              SLLM_STATUS_OK, "queue create", error)) {
    return 1;
  }

  const std::array<uint64_t, 9> sizes = {max_tokens * 8192U * 2U,
                                         max_tokens * 4096U * 2U,
                                         max_tokens * 32U * 2U,
                                         max_tokens * 32U * 2U,
                                         8192U * 4U * 2U,
                                         32U * 4U,
                                         32U * 2U,
                                         128U * 4U,
                                         max_tokens * 4096U * 2U};
  std::array<sllm_buffer_t *, 9> buffers{};
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    sllm_buffer_create_info_t info{};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    info.size_bytes = sizes[index];
    if (!expect(
            sllm_buffer_create(context, &info, &buffers[index], &error.sink),
            SLLM_STATUS_OK, "buffer create", error)) {
      return 1;
    }
  }

  const uint16_t zero = float_to_bf16_rne(0.0F);
  std::vector<uint16_t> conv(8192U * 4U, zero);
  for (std::size_t channel = 0U; channel != 8192U; ++channel) {
    conv[channel * 4U] = float_to_bf16_rne(0.125F);
    conv[channel * 4U + 1U] = float_to_bf16_rne(-0.25F);
    conv[channel * 4U + 2U] = float_to_bf16_rne(0.5F);
    conv[channel * 4U + 3U] = float_to_bf16_rne(0.75F);
  }
  std::vector<float> a_log(32U);
  std::vector<uint16_t> dt_bias(32U);
  for (std::size_t head = 0U; head != 32U; ++head) {
    a_log[head] = std::log(0.5F + static_cast<float>(head) * 0.01F);
    dt_bias[head] =
        float_to_bf16_rne((static_cast<float>(head % 5U) - 2.0F) * 0.03125F);
  }
  std::vector<float> norm(128U);
  for (std::size_t dimension = 0U; dimension != 128U; ++dimension) {
    norm[dimension] = 0.75F + static_cast<float>(dimension % 11U) * 0.03125F;
  }
  if (!upload(queue, buffers[4], conv.data(), sizes[4]) ||
      !upload(queue, buffers[5], a_log.data(), sizes[5]) ||
      !upload(queue, buffers[6], dt_bias.data(), sizes[6]) ||
      !upload(queue, buffers[7], norm.data(), sizes[7])) {
    return 1;
  }

  sllm_linear_attention_state_create_info_t state_info{};
  state_info.struct_size = sizeof(state_info);
  state_info.abi_version = SLLM_HIP_ABI_VERSION;
  state_info.session_id = 1U;
  state_info.layer_id = 0U;
  state_info.capacity_tokens = capacity;
  sllm_linear_attention_state_t *state = nullptr;
  if (!expect(sllm_linear_attention_state_create(context, &state_info, &state,
                                                 &error.sink),
              SLLM_STATUS_OK, "state create", error)) {
    return 1;
  }

  const uint64_t conv_shape[] = {8192U, 1U, 4U};
  const uint64_t head_shape[] = {32U};
  const uint64_t norm_shape[] = {128U};
  Oracle oracle;
  uint64_t start_position = 0U;
  uint32_t max_observed_ulp = 0U;
  constexpr std::array<uint32_t, 2> call_sizes = {3U, 1U};
  for (std::size_t call = 0U; call != call_sizes.size(); ++call) {
    const uint32_t token_count = call_sizes[call];
    const uint32_t phase = call == 0U ? 0U : 3U;
    std::vector<uint16_t> qkv(static_cast<std::size_t>(token_count) * 8192U);
    std::vector<uint16_t> z(static_cast<std::size_t>(token_count) * 4096U);
    std::vector<uint16_t> b(static_cast<std::size_t>(token_count) * 32U);
    std::vector<uint16_t> a(static_cast<std::size_t>(token_count) * 32U);
    for (uint32_t token = 0U; token != token_count; ++token) {
      for (uint32_t channel = 0U; channel != 8192U; ++channel) {
        const float centered =
            static_cast<float>(static_cast<int32_t>(channel % 17U) - 8);
        const float value = centered * 0.015625F +
                            static_cast<float>(token + phase + 1U) * 0.03125F;
        qkv[static_cast<std::size_t>(token) * 8192U + channel] =
            float_to_bf16_rne(value);
      }
      for (uint32_t index = 0U; index != 4096U; ++index) {
        const float centered =
            static_cast<float>(static_cast<int32_t>(index % 13U) - 6);
        z[static_cast<std::size_t>(token) * 4096U + index] =
            float_to_bf16_rne(centered * 0.03125F + 0.25F);
      }
      for (uint32_t head = 0U; head != 32U; ++head) {
        const std::size_t index = static_cast<std::size_t>(token) * 32U + head;
        b[index] = float_to_bf16_rne(
            static_cast<float>(static_cast<int32_t>(head % 5U) - 2) * 0.125F);
        a[index] = float_to_bf16_rne(
            static_cast<float>(static_cast<int32_t>(head % 7U) - 3) * 0.0625F);
      }
    }
    const uint64_t qkv_bytes = qkv.size() * sizeof(uint16_t);
    const uint64_t z_bytes = z.size() * sizeof(uint16_t);
    const uint64_t scalar_bytes = b.size() * sizeof(uint16_t);
    if (!upload(queue, buffers[0], qkv.data(), qkv_bytes) ||
        !upload(queue, buffers[1], z.data(), z_bytes) ||
        !upload(queue, buffers[2], b.data(), scalar_bytes) ||
        !upload(queue, buffers[3], a.data(), scalar_bytes)) {
      return 1;
    }
    const uint64_t qkv_shape[] = {token_count, 8192U};
    const uint64_t output_shape[] = {token_count, 4096U};
    const uint64_t scalar_shape[] = {token_count, 32U};
    sllm_linear_attention_desc_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.op_version = SLLM_HIP_LINEAR_ATTENTION_VERSION;
    descriptor.start_position = start_position;
    descriptor.expected_length = start_position + token_count;
    descriptor.state = state;
    descriptor.qkv = binding(buffers[0], SLLM_TENSOR_DTYPE_BF16, 2U, qkv_shape);
    descriptor.z =
        binding(buffers[1], SLLM_TENSOR_DTYPE_BF16, 2U, output_shape);
    descriptor.b_input =
        binding(buffers[2], SLLM_TENSOR_DTYPE_BF16, 2U, scalar_shape);
    descriptor.a_input =
        binding(buffers[3], SLLM_TENSOR_DTYPE_BF16, 2U, scalar_shape);
    descriptor.conv_weight =
        binding(buffers[4], SLLM_TENSOR_DTYPE_BF16, 3U, conv_shape);
    descriptor.a_log =
        binding(buffers[5], SLLM_TENSOR_DTYPE_F32, 1U, head_shape);
    descriptor.dt_bias =
        binding(buffers[6], SLLM_TENSOR_DTYPE_BF16, 1U, head_shape);
    descriptor.norm_weight =
        binding(buffers[7], SLLM_TENSOR_DTYPE_F32, 1U, norm_shape);
    descriptor.output =
        binding(buffers[8], SLLM_TENSOR_DTYPE_BF16, 2U, output_shape);
    const std::vector<uint16_t> expected =
        oracle.run(qkv, z, b, a, conv, a_log, dt_bias, norm, token_count);
    sllm_linear_attention_dispatch_info_t dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    dispatch.info_version = SLLM_HIP_LINEAR_ATTENTION_DISPATCH_INFO_VERSION;
    sllm_completion_t *completion = nullptr;
    if (!expect(sllm_linear_attention_execute(context, queue, &descriptor,
                                              &completion, &dispatch,
                                              &error.sink),
                SLLM_STATUS_OK, "linear attention execute", error) ||
        dispatch.dispatch_count != 2U || dispatch.token_count != token_count ||
        dispatch.fallback_used != 0U || !completion_wait(completion) ||
        !completion_release(&completion)) {
      return 1;
    }
    std::vector<uint16_t> output(static_cast<std::size_t>(token_count) * 4096U);
    if (!read_output(queue, buffers[8], &output)) {
      return 1;
    }
    for (std::size_t index = 0U; index != output.size(); ++index) {
      const uint32_t distance =
          bf16_ulp_distance(output[index], expected[index]);
      max_observed_ulp = std::max(max_observed_ulp, distance);
      if (distance > 3U) {
        std::cerr << "call " << call << " output mismatch at " << index
                  << ": actual=" << output[index]
                  << " expected=" << expected[index] << " ulp=" << distance
                  << '\n';
        return 1;
      }
    }
    start_position += token_count;
    sllm_linear_attention_view_info_t view{};
    view.struct_size = sizeof(view);
    view.abi_version = SLLM_HIP_ABI_VERSION;
    view.info_version = SLLM_HIP_LINEAR_ATTENTION_VIEW_INFO_VERSION;
    if (!expect(sllm_linear_attention_state_query(state, &view, &error.sink),
                SLLM_STATUS_OK, "state query", error) ||
        view.observed_length != start_position ||
        view.generation != call + 1U || view.active_slot != (call + 1U) % 2U) {
      return 1;
    }
  }

  if (!expect(sllm_linear_attention_state_release(&state, &error.sink),
              SLLM_STATUS_OK, "state release", error)) {
    return 1;
  }
  for (auto &buffer : buffers) {
    if (!expect(sllm_buffer_release(&buffer, &error.sink), SLLM_STATUS_OK,
                "buffer release", error)) {
      return 1;
    }
  }
  if (!expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
              "queue release", error) ||
      !expect(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
              "context release", error)) {
    return 1;
  }
  std::cout
      << "{\"state\":\"PASS\",\"target\":\"" << SLLM_TEST_EXPECTED_TARGET
      << "\",\"prefill_tokens\":3,\"decode_tokens\":1,\"max_bf16_ulp\":"
      << max_observed_ulp
      << ",\"scalar_boundary_regression\":true,\"fallback_used\":false}\n";
  return 0;
}
