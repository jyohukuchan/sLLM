#include "sllm/hip.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "unknown"
#endif

/* These private backend entry points intentionally stay outside the installed
 * public header. Keep their declarations aligned with the native runtime and
 * the internal Rust FFI. */
extern "C" {
sllm_status_t sllm_linear_attention_state_prepare_checkpoint(
    const sllm_linear_attention_state_t *state, uint64_t expected_start,
    uint32_t token_count, uint32_t rows,
    sllm_error_sink_t *error_sink) noexcept;
sllm_status_t sllm_linear_attention_state_validate_checkpoint(
    const sllm_linear_attention_state_t *state, uint64_t expected_start,
    uint64_t expected_end, uint32_t rows,
    sllm_error_sink_t *error_sink) noexcept;
sllm_status_t sllm_linear_attention_state_commit_checkpoint(
    const sllm_context_t *context, const sllm_queue_t *queue,
    const sllm_linear_attention_state_t *state, uint64_t expected_start,
    uint64_t expected_end, uint64_t prefix_end, uint32_t row_index,
    sllm_error_sink_t *error_sink) noexcept;
sllm_status_t sllm_linear_attention_state_commit_checkpoint_batch(
    const sllm_context_t *context, const sllm_queue_t *queue,
    const sllm_linear_attention_state_t *const *states, uint32_t state_count,
    uint64_t expected_start, uint64_t expected_end, uint64_t prefix_end,
    uint32_t row_index, sllm_error_sink_t *error_sink) noexcept;
sllm_status_t sllm_linear_attention_state_discard_checkpoint(
    const sllm_linear_attention_state_t *state,
    sllm_error_sink_t *error_sink) noexcept;
}

namespace {

constexpr uint32_t kQkHeads = 16U;
constexpr uint32_t kValueHeads = 48U;
constexpr uint32_t kHeadDim = 128U;
constexpr uint32_t kConvKernel = 4U;
constexpr uint32_t kConvHistory = kConvKernel - 1U;
constexpr uint32_t kQkvWidth = (2U * kQkHeads + kValueHeads) * kHeadDim;
constexpr uint32_t kOutputWidth = kValueHeads * kHeadDim;
constexpr uint32_t kMaxRows = 3U;
constexpr uint32_t kCapacity = 16U;
constexpr uint32_t kQkToValue = kValueHeads / kQkHeads;

struct Error final {
  char message[256]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};
};

bool expect_status(const sllm_status_t actual, const sllm_status_t expected,
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

bool bf16_finite(const uint16_t value) {
  return (value & UINT16_C(0x7f80)) != UINT16_C(0x7f80);
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

bool completion_wait_and_release(sllm_completion_t **const completion,
                                 const char *const operation) {
  if (completion == nullptr || *completion == nullptr) {
    std::cerr << operation << " returned a null completion\n";
    return false;
  }
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  bool ok = expect_status(
                sllm_completion_wait(*completion, 5000U, &result, &error.sink),
                SLLM_STATUS_OK, operation, error) &&
            result.state == SLLM_COMPLETION_STATE_SUCCESS;
  if (!ok) {
    std::cerr << operation << " completion state was " << result.state << '\n';
  }
  ok = expect_status(sllm_completion_release(completion, &error.sink),
                     SLLM_STATUS_OK, "completion release", error) &&
       ok;
  return ok;
}

bool upload(const sllm_queue_t *const queue, const sllm_buffer_t *const buffer,
            void *const data, const uint64_t bytes, const char *const label) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = data;
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect_status(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                          &error.sink),
                     SLLM_STATUS_OK, label, error)) {
    return false;
  }
  return completion_wait_and_release(&completion, label);
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

struct Batch final {
  uint32_t rows = 0U;
  std::vector<uint16_t> qkv;
  std::vector<uint16_t> z;
  std::vector<uint16_t> b;
  std::vector<uint16_t> a;
};

Batch make_batch(const uint32_t first_token, const uint32_t rows) {
  Batch batch;
  batch.rows = rows;
  batch.qkv.resize(static_cast<std::size_t>(rows) * kQkvWidth);
  batch.z.resize(static_cast<std::size_t>(rows) * kOutputWidth);
  batch.b.resize(static_cast<std::size_t>(rows) * kValueHeads);
  batch.a.resize(static_cast<std::size_t>(rows) * kValueHeads);
  for (uint32_t row = 0U; row != rows; ++row) {
    const uint32_t token = first_token + row;
    for (uint32_t channel = 0U; channel != kQkvWidth; ++channel) {
      const float phase = static_cast<float>((token + 1U) * 17U + channel);
      batch.qkv[static_cast<std::size_t>(row) * kQkvWidth + channel] =
          float_to_bf16_rne(0.17F * std::sin(phase * 0.013F) +
                            0.03F * std::cos(phase * 0.007F));
    }
    for (uint32_t index = 0U; index != kOutputWidth; ++index) {
      const float phase = static_cast<float>((token + 3U) * 11U + index);
      batch.z[static_cast<std::size_t>(row) * kOutputWidth + index] =
          float_to_bf16_rne(0.12F + 0.28F * std::cos(phase * 0.017F));
    }
    for (uint32_t head = 0U; head != kValueHeads; ++head) {
      const std::size_t index =
          static_cast<std::size_t>(row) * kValueHeads + head;
      batch.b[index] =
          float_to_bf16_rne(((head + token) & 1U) == 0U ? 0.35F : -0.45F);
      batch.a[index] = float_to_bf16_rne(((head + token) % 3U) == 0U   ? 0.2F
                                         : ((head + token) % 3U) == 1U ? -0.3F
                                                                       : 0.1F);
    }
  }
  return batch;
}

struct Parameters final {
  std::vector<uint16_t> conv_weight =
      std::vector<uint16_t>(static_cast<std::size_t>(kQkvWidth) * kConvKernel);
  std::vector<float> a_log = std::vector<float>(kValueHeads);
  std::vector<uint16_t> dt_bias = std::vector<uint16_t>(kValueHeads);
  std::vector<float> norm_weight = std::vector<float>(kHeadDim);
  std::vector<uint16_t> conv_seed =
      std::vector<uint16_t>(static_cast<std::size_t>(kConvHistory) * kQkvWidth);
  std::vector<float> recurrent_seed = std::vector<float>(
      static_cast<std::size_t>(kValueHeads) * kHeadDim * kHeadDim);

  Parameters() {
    for (uint32_t channel = 0U; channel != kQkvWidth; ++channel) {
      for (uint32_t tap = 0U; tap != kConvKernel; ++tap) {
        const float phase = static_cast<float>(channel + tap) * 0.021F;
        conv_weight[static_cast<std::size_t>(channel) * kConvKernel + tap] =
            float_to_bf16_rne(0.08F * std::sin(phase));
      }
    }
    for (uint32_t head = 0U; head != kValueHeads; ++head) {
      a_log[head] = (head & 1U) == 0U ? -0.75F : -1.15F;
      dt_bias[head] = float_to_bf16_rne((head & 1U) == 0U ? 0.07F : -0.05F);
    }
    for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
      norm_weight[dimension] =
          0.85F + 0.1F * std::cos(static_cast<float>(dimension) * 0.031F);
    }
    for (std::size_t index = 0U; index != conv_seed.size(); ++index) {
      conv_seed[index] = float_to_bf16_rne(
          0.06F * std::cos(static_cast<float>(index % 257U) * 0.019F));
    }
    for (std::size_t index = 0U; index != recurrent_seed.size(); ++index) {
      recurrent_seed[index] =
          0.0015F * std::sin(static_cast<float>(index % 4093U) * 0.017F);
    }
  }
};

struct OracleSnapshot final {
  std::vector<uint16_t> conv;
  std::vector<float> recurrent;
};

std::size_t recurrent_state_index(const uint32_t value_head,
                                  const uint32_t dimension,
                                  const uint32_t key_dimension) {
  const std::size_t state_base =
      static_cast<std::size_t>(value_head) * kHeadDim * kHeadDim;
  if (std::string_view(SLLM_TEST_EXPECTED_TARGET) == "gfx1030") {
    return state_base + static_cast<std::size_t>(key_dimension) * kHeadDim +
           dimension;
  }
  return state_base + static_cast<std::size_t>(dimension) * kHeadDim +
         key_dimension;
}

struct GdnOracle final {
  std::vector<uint16_t> conv_history;
  std::vector<float> recurrent;
  const Parameters &parameters;

  explicit GdnOracle(const Parameters &params)
      : conv_history(params.conv_seed), recurrent(params.recurrent_seed),
        parameters(params) {}

  OracleSnapshot snapshot() const { return {conv_history, recurrent}; }

  std::vector<uint16_t> step(const Batch &batch, const uint32_t row) {
    std::vector<uint16_t> convolved(kQkvWidth);
    const std::size_t qkv_offset = static_cast<std::size_t>(row) * kQkvWidth;
    for (uint32_t channel = 0U; channel != kQkvWidth; ++channel) {
      float sum = 0.0F;
      for (uint32_t tap = 0U; tap != kConvKernel; ++tap) {
        // This oracle advances history after each row, unlike the native
        // batch kernel, which reads the history from before the whole batch.
        const int64_t source = static_cast<int64_t>(tap) - 3;
        const uint16_t value =
            source < 0
                ? conv_history[static_cast<std::size_t>(source + 3) *
                                   kQkvWidth +
                               channel]
                : batch.qkv[qkv_offset +
                            static_cast<std::size_t>(source) * kQkvWidth +
                            channel];
        sum += bf16_to_float(value) *
               bf16_to_float(
                   parameters.conv_weight[static_cast<std::size_t>(channel) *
                                              kConvKernel +
                                          tap]);
      }
      const float silu = sum / (1.0F + std::exp(-sum));
      convolved[channel] = float_to_bf16_rne(silu);
    }
    std::memmove(conv_history.data(), conv_history.data() + kQkvWidth,
                 static_cast<std::size_t>(kConvHistory - 1U) * kQkvWidth *
                     sizeof(uint16_t));
    std::memcpy(conv_history.data() +
                    static_cast<std::size_t>(kConvHistory - 1U) * kQkvWidth,
                batch.qkv.data() + qkv_offset,
                static_cast<std::size_t>(kQkvWidth) * sizeof(uint16_t));

    std::vector<uint16_t> output(kOutputWidth);
    std::array<float, kHeadDim> q_values{};
    std::array<float, kHeadDim> k_values{};
    std::array<float, kHeadDim> values{};
    for (uint32_t value_head = 0U; value_head != kValueHeads; ++value_head) {
      const uint32_t qk_head = value_head / kQkToValue;
      float q_sum = 0.0F;
      float k_sum = 0.0F;
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        q_values[dimension] =
            bf16_to_float(convolved[qk_head * kHeadDim + dimension]);
        k_values[dimension] = bf16_to_float(
            convolved[kQkHeads * kHeadDim + qk_head * kHeadDim + dimension]);
        q_sum += q_values[dimension] * q_values[dimension];
        k_sum += k_values[dimension] * k_values[dimension];
      }
      const float q_inverse = 1.0F / std::sqrt(q_sum + 1.0e-6F);
      const float k_inverse = 1.0F / std::sqrt(k_sum + 1.0e-6F);
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        q_values[dimension] =
            bf16_to_float(float_to_bf16_rne(q_values[dimension] * q_inverse));
        k_values[dimension] =
            bf16_to_float(float_to_bf16_rne(k_values[dimension] * k_inverse));
        q_values[dimension] *= 1.0F / std::sqrt(128.0F);
      }
      const std::size_t scalar_index =
          static_cast<std::size_t>(row) * kValueHeads + value_head;
      const float beta_f32 =
          1.0F / (1.0F + std::exp(-bf16_to_float(batch.b[scalar_index])));
      const float beta = bf16_to_float(float_to_bf16_rne(beta_f32));
      const float a_value = bf16_to_float(batch.a[scalar_index]) +
                            bf16_to_float(parameters.dt_bias[value_head]);
      const float softplus =
          std::fmax(a_value, 0.0F) + std::log1p(std::exp(-std::fabs(a_value)));
      const float decay =
          std::exp(-std::exp(parameters.a_log[value_head]) * softplus);
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        for (uint32_t key_dimension = 0U; key_dimension != kHeadDim;
             ++key_dimension) {
          const std::size_t state_index =
              recurrent_state_index(value_head, dimension, key_dimension);
          recurrent[state_index] *= decay;
        }
        float previous_projection = 0.0F;
        for (uint32_t key_dimension = 0U; key_dimension != kHeadDim;
             ++key_dimension) {
          const std::size_t state_index =
              recurrent_state_index(value_head, dimension, key_dimension);
          previous_projection +=
              recurrent[state_index] * k_values[key_dimension];
        }
        const float value =
            bf16_to_float(convolved[2U * kQkHeads * kHeadDim +
                                    value_head * kHeadDim + dimension]);
        const float residual = value - previous_projection;
        float projection = 0.0F;
        for (uint32_t key_dimension = 0U; key_dimension != kHeadDim;
             ++key_dimension) {
          const std::size_t index =
              recurrent_state_index(value_head, dimension, key_dimension);
          recurrent[index] += beta * residual * k_values[key_dimension];
          projection += recurrent[index] * q_values[key_dimension];
        }
        values[dimension] = bf16_to_float(float_to_bf16_rne(projection));
      }
      float square_sum = 0.0F;
      for (const float value : values) {
        square_sum += value * value;
      }
      const float inverse_rms = 1.0F / std::sqrt(square_sum / 128.0F + 1.0e-6F);
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        const std::size_t output_index =
            static_cast<std::size_t>(value_head) * kHeadDim + dimension;
        const float z_value =
            bf16_to_float(batch.z[static_cast<std::size_t>(row) * kOutputWidth +
                                  output_index]);
        const float z_silu = z_value / (1.0F + std::exp(-z_value));
        const float normalized = values[dimension] * inverse_rms;
        const float normalized_bf16 =
            bf16_to_float(float_to_bf16_rne(normalized));
        output[output_index] = float_to_bf16_rne(
            normalized_bf16 * parameters.norm_weight[dimension] * z_silu);
      }
    }
    return output;
  }
};

bool upload_state_plane(const sllm_linear_attention_state_t *const state,
                        const uint32_t plane, void *const data,
                        const uint64_t bytes, const char *const operation) {
  sllm_state_chunk_t chunk{};
  chunk.struct_size = sizeof(chunk);
  chunk.abi_version = SLLM_HIP_ABI_VERSION;
  chunk.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  chunk.plane = plane;
  chunk.byte_length = bytes;
  chunk.host_pointer = data;
  chunk.host_capacity = bytes;
  Error error;
  return expect_status(
      sllm_linear_attention_state_import(state, &chunk, &error.sink),
      SLLM_STATUS_OK, operation, error);
}

bool export_state_plane(const sllm_linear_attention_state_t *const state,
                        const uint32_t plane, std::vector<uint8_t> *const data,
                        const char *const operation) {
  sllm_state_chunk_t chunk{};
  chunk.struct_size = sizeof(chunk);
  chunk.abi_version = SLLM_HIP_ABI_VERSION;
  chunk.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  chunk.plane = plane;
  chunk.byte_length = data->size();
  chunk.host_pointer = data->data();
  chunk.host_capacity = data->size();
  Error error;
  return expect_status(
      sllm_linear_attention_state_export(state, &chunk, &error.sink),
      SLLM_STATUS_OK, operation, error);
}

bool query_state(const sllm_linear_attention_state_t *const state,
                 sllm_linear_attention_view_info_t *const view,
                 const char *const operation) {
  *view = {};
  view->struct_size = sizeof(*view);
  view->abi_version = SLLM_HIP_ABI_VERSION;
  view->info_version = SLLM_HIP_LINEAR_ATTENTION_VIEW_INFO_VERSION;
  Error error;
  return expect_status(
      sllm_linear_attention_state_query(state, view, &error.sink),
      SLLM_STATUS_OK, operation, error);
}

bool compare_state(const sllm_linear_attention_state_t *const state,
                   const OracleSnapshot &expected, const char *const label) {
  sllm_linear_attention_view_info_t view{};
  if (!query_state(state, &view, "checkpoint state query")) {
    return false;
  }
  const uint32_t slot = view.active_slot;
  const uint32_t conv_plane = slot == 0U
                                  ? SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT0
                                  : SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT1;
  const uint32_t recurrent_plane =
      slot == 0U ? SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT0
                 : SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT1;
  std::vector<uint8_t> conv(expected.conv.size() * sizeof(uint16_t));
  std::vector<uint8_t> recurrent(expected.recurrent.size() * sizeof(float));
  if (!export_state_plane(state, conv_plane, &conv, "checkpoint conv export") ||
      !export_state_plane(state, recurrent_plane, &recurrent,
                          "checkpoint recurrent export")) {
    return false;
  }
  std::vector<uint16_t> actual_conv(expected.conv.size());
  std::vector<float> actual_recurrent(expected.recurrent.size());
  std::memcpy(actual_conv.data(), conv.data(), conv.size());
  std::memcpy(actual_recurrent.data(), recurrent.data(), recurrent.size());
  const bool conv_finite =
      std::all_of(actual_conv.begin(), actual_conv.end(), bf16_finite);
  float max_abs = 0.0F;
  float max_rel = 0.0F;
  bool recurrent_finite = true;
  for (std::size_t index = 0U; index != expected.recurrent.size(); ++index) {
    const float actual = actual_recurrent[index];
    const float wanted = expected.recurrent[index];
    recurrent_finite = recurrent_finite && std::isfinite(actual);
    max_abs = std::max(max_abs, std::fabs(actual - wanted));
    max_rel = std::max(max_rel, std::fabs(actual - wanted) /
                                    std::max(1.0e-2F, std::fabs(wanted)));
  }
  const bool conv_match =
      std::memcmp(conv.data(), expected.conv.data(), conv.size()) == 0;
  const bool recurrent_match = max_abs <= 3.0e-3F && max_rel <= 0.5F;
  std::cout << label << " slot=" << slot
            << " conv_bitwise=" << (conv_match ? "PASS" : "FAIL")
            << " recurrent_max_abs=" << max_abs
            << " recurrent_max_rel=" << max_rel
            << " finite=" << (conv_finite && recurrent_finite ? "PASS" : "FAIL")
            << '\n';
  return conv_finite && recurrent_finite && conv_match && recurrent_match;
}

struct StateSnapshot final {
  sllm_linear_attention_view_info_t view{};
  std::vector<uint8_t> conv;
  std::vector<uint8_t> recurrent;
};

bool capture_state_snapshot(const sllm_linear_attention_state_t *const state,
                            const Parameters &parameters,
                            StateSnapshot *const snapshot,
                            const char *const label) {
  if (snapshot == nullptr || !query_state(state, &snapshot->view, label)) {
    return false;
  }
  const uint32_t slot = snapshot->view.active_slot;
  const uint32_t conv_plane = slot == 0U
                                  ? SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT0
                                  : SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT1;
  const uint32_t recurrent_plane =
      slot == 0U ? SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT0
                 : SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT1;
  snapshot->conv.resize(parameters.conv_seed.size() * sizeof(uint16_t));
  snapshot->recurrent.resize(parameters.recurrent_seed.size() * sizeof(float));
  return export_state_plane(state, conv_plane, &snapshot->conv,
                            "batch snapshot conv") &&
         export_state_plane(state, recurrent_plane, &snapshot->recurrent,
                            "batch snapshot recurrent");
}

bool snapshot_metadata_equal(const sllm_linear_attention_view_info_t &left,
                             const sllm_linear_attention_view_info_t &right) {
  if (left.struct_size != right.struct_size ||
      left.abi_version != right.abi_version ||
      left.info_version != right.info_version ||
      left.reserved0 != right.reserved0 ||
      left.session_id != right.session_id || left.layer_id != right.layer_id ||
      left.conv_state_dtype != right.conv_state_dtype ||
      left.recurrent_state_dtype != right.recurrent_state_dtype ||
      left.encoding != right.encoding ||
      left.active_slot != right.active_slot ||
      left.capacity_tokens != right.capacity_tokens ||
      left.observed_length != right.observed_length ||
      left.generation != right.generation ||
      left.context_identity != right.context_identity ||
      left.state_identity != right.state_identity) {
    return false;
  }
  return std::memcmp(left.conv_state_shape, right.conv_state_shape,
                     sizeof(left.conv_state_shape)) == 0 &&
         std::memcmp(left.recurrent_state_shape, right.recurrent_state_shape,
                     sizeof(left.recurrent_state_shape)) == 0 &&
         std::memcmp(left.reserved, right.reserved, sizeof(left.reserved)) == 0;
}

bool snapshot_equal(const StateSnapshot &before, const StateSnapshot &after,
                    const char *const label) {
  const bool metadata = snapshot_metadata_equal(before.view, after.view);
  const bool payload =
      before.conv == after.conv && before.recurrent == after.recurrent;
  if (!metadata || !payload) {
    std::cerr << label << " mutated state during rejected batch preflight\n";
  }
  return metadata && payload;
}

bool compare_output(const std::vector<uint16_t> &actual,
                    const std::vector<uint16_t> &expected,
                    const char *const label) {
  if (actual.size() != expected.size()) {
    return false;
  }
  uint32_t max_ulp = 0U;
  bool finite = true;
  for (std::size_t index = 0U; index != actual.size(); ++index) {
    finite = finite && bf16_finite(actual[index]);
    max_ulp =
        std::max(max_ulp, bf16_ulp_distance(actual[index], expected[index]));
  }
  std::cout << label << " max_bf16_ulp=" << max_ulp
            << " finite=" << (finite ? "PASS" : "FAIL") << '\n';
  return finite && max_ulp <= 4U;
}

bool compare_view(const sllm_linear_attention_view_info_t &view,
                  const uint64_t observed_length, const uint32_t active_slot,
                  const char *const label) {
  const bool ok = view.observed_length == observed_length &&
                  view.active_slot == active_slot;
  if (!ok) {
    std::cerr << label << " observed=" << view.observed_length
              << " active_slot=" << view.active_slot
              << " expected=" << observed_length << '/' << active_slot << '\n';
  }
  return ok;
}

bool create_state(const sllm_context_t *const context,
                  sllm_linear_attention_state_t **const state,
                  const uint64_t session_id) {
  sllm_linear_attention_state_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.session_id = session_id;
  info.layer_id = 0U;
  info.capacity_tokens = kCapacity;
  info.qk_heads = kQkHeads;
  info.value_heads = kValueHeads;
  info.head_dim = kHeadDim;
  info.conv_kernel_size = kConvKernel;
  Error error;
  return expect_status(
      sllm_linear_attention_state_create(context, &info, state, &error.sink),
      SLLM_STATUS_OK, "checkpoint state create", error);
}

bool execute_batch(const sllm_context_t *const context,
                   const sllm_queue_t *const queue,
                   sllm_linear_attention_state_t *const state,
                   const std::array<sllm_buffer_t *, 9U> &buffers,
                   const Parameters &parameters, const Batch &batch,
                   const uint64_t start_position,
                   std::vector<uint16_t> *const output) {
  const uint64_t qkv_shape[] = {batch.rows, kQkvWidth};
  const uint64_t output_shape[] = {batch.rows, kOutputWidth};
  const uint64_t scalar_shape[] = {batch.rows, kValueHeads};
  const uint64_t conv_shape[] = {kQkvWidth, 1U, kConvKernel};
  const uint64_t head_shape[] = {kValueHeads};
  const uint64_t norm_shape[] = {kHeadDim};
  const uint64_t qkv_bytes =
      static_cast<uint64_t>(batch.qkv.size()) * sizeof(uint16_t);
  const uint64_t z_bytes =
      static_cast<uint64_t>(batch.z.size()) * sizeof(uint16_t);
  const uint64_t scalar_bytes =
      static_cast<uint64_t>(batch.b.size()) * sizeof(uint16_t);
  output->assign(static_cast<std::size_t>(batch.rows) * kOutputWidth, 0U);
  const uint64_t output_bytes =
      static_cast<uint64_t>(output->size()) * sizeof(uint16_t);
  if (!upload(queue, buffers[0], const_cast<uint16_t *>(batch.qkv.data()),
              qkv_bytes, "checkpoint qkv upload") ||
      !upload(queue, buffers[1], const_cast<uint16_t *>(batch.z.data()),
              z_bytes, "checkpoint z upload") ||
      !upload(queue, buffers[2], const_cast<uint16_t *>(batch.b.data()),
              scalar_bytes, "checkpoint b upload") ||
      !upload(queue, buffers[3], const_cast<uint16_t *>(batch.a.data()),
              scalar_bytes, "checkpoint a upload")) {
    return false;
  }
  const auto make_descriptor = [&]() {
    sllm_linear_attention_desc_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.op_version = SLLM_HIP_LINEAR_ATTENTION_VERSION;
    descriptor.start_position = start_position;
    descriptor.expected_length = start_position + batch.rows;
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
    return descriptor;
  };
  sllm_linear_attention_desc_t descriptor = make_descriptor();
  sllm_linear_attention_dispatch_info_t dispatch{};
  dispatch.struct_size = sizeof(dispatch);
  dispatch.abi_version = SLLM_HIP_ABI_VERSION;
  dispatch.info_version = SLLM_HIP_LINEAR_ATTENTION_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect_status(sllm_linear_attention_execute(context, queue, &descriptor,
                                                   &completion, &dispatch,
                                                   &error.sink),
                     SLLM_STATUS_OK, "checkpoint execute", error) ||
      !completion_wait_and_release(&completion, "checkpoint execute")) {
    return false;
  }
  if (dispatch.token_count != batch.rows ||
      dispatch.start_position != start_position ||
      dispatch.expected_length != start_position + batch.rows ||
      dispatch.fallback_used != 0U || dispatch.fallback_allowed != 0U ||
      std::string(dispatch.gcn_arch_name) != SLLM_TEST_EXPECTED_TARGET) {
    std::cerr << "checkpoint dispatch metadata mismatch\n";
    return false;
  }
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.size_bytes = output_bytes;
  completion = nullptr;
  if (!expect_status(sllm_buffer_copy_d2h(queue, buffers[8], &transfer,
                                          &completion, &error.sink),
                     SLLM_STATUS_OK, "checkpoint output readback", error) ||
      completion == nullptr) {
    return false;
  }
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect_status(
          sllm_completion_wait(completion, 5000U, &result, &error.sink),
          SLLM_STATUS_OK, "checkpoint output wait", error) ||
      result.state != SLLM_COMPLETION_STATE_SUCCESS) {
    (void)sllm_completion_release(&completion, &error.sink);
    return false;
  }
  uint64_t written = 0U;
  bool ok =
      expect_status(sllm_completion_read(completion, output->data(),
                                         output_bytes, &written, &error.sink),
                    SLLM_STATUS_OK, "checkpoint output read", error) &&
      written == output_bytes;
  ok = expect_status(sllm_completion_release(&completion, &error.sink),
                     SLLM_STATUS_OK, "checkpoint output release", error) &&
       ok;
  (void)parameters;
  return ok;
}

bool run_case(const sllm_context_t *const context,
              const sllm_queue_t *const queue,
              const std::array<sllm_buffer_t *, 9U> &buffers,
              const Parameters &parameters, const Batch &first_batch,
              const Batch &second_batch,
              const std::array<OracleSnapshot, 7U> &snapshots,
              const std::vector<std::vector<uint16_t>> &oracle_outputs,
              const uint32_t mode) {
  sllm_linear_attention_state_t *state = nullptr;
  bool ok = create_state(context, &state, 100U + mode);
  if (!ok) {
    return false;
  }
  const uint64_t recurrent_bytes =
      static_cast<uint64_t>(parameters.recurrent_seed.size()) * sizeof(float);
  const uint64_t conv_bytes =
      static_cast<uint64_t>(parameters.conv_seed.size()) * sizeof(uint16_t);
  ok = upload_state_plane(state, SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT0,
                          const_cast<uint16_t *>(parameters.conv_seed.data()),
                          conv_bytes, "checkpoint seed conv") &&
       upload_state_plane(state, SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT0,
                          const_cast<float *>(parameters.recurrent_seed.data()),
                          recurrent_bytes, "checkpoint seed recurrent") &&
       ok;
  if (!ok) {
    Error error;
    (void)sllm_linear_attention_state_release(&state, &error.sink);
    return false;
  }
  sllm_linear_attention_view_info_t initial_view{};
  ok = query_state(state, &initial_view, "checkpoint initial query") &&
       compare_view(initial_view, 0U, 0U, "checkpoint initial view");

  if (ok && mode != 3U) {
    sllm_linear_attention_view_info_t before = initial_view;
    Error error;
    ok = expect_status(sllm_linear_attention_state_prepare_checkpoint(
                           state, 1U, 3U, 2U, &error.sink),
                       SLLM_STATUS_INVALID_ARGUMENT, "checkpoint stale prepare",
                       error) &&
         query_state(state, &before, "checkpoint stale query") &&
         compare_view(before, 0U, 0U, "checkpoint stale unchanged");
  }
  Error error;
  if (ok) {
    ok = expect_status(sllm_linear_attention_state_prepare_checkpoint(
                           state, 0U, 3U, 2U, &error.sink),
                       SLLM_STATUS_OK, "checkpoint prepare", error);
  }
  std::vector<uint16_t> first_output;
  if (ok) {
    first_output.resize(3U * kOutputWidth);
    ok = execute_batch(context, queue, state, buffers, parameters, first_batch,
                       0U, &first_output);
  }
  if (ok) {
    ok = compare_output(first_output, oracle_outputs[0], "checkpoint M3") &&
         expect_status(sllm_linear_attention_state_validate_checkpoint(
                           state, 1U, 4U, 2U, &error.sink),
                       SLLM_STATUS_INVALID_ARGUMENT,
                       "checkpoint wrong validate", error) &&
         expect_status(sllm_linear_attention_state_validate_checkpoint(
                           state, 0U, 3U, 2U, &error.sink),
                       SLLM_STATUS_OK, "checkpoint validate", error);
  }

  if (ok && mode < 2U) {
    sllm_linear_attention_view_info_t before_invalid{};
    ok = query_state(state, &before_invalid, "checkpoint pre-invalid query");
    std::vector<uint8_t> conv_before(parameters.conv_seed.size() *
                                     sizeof(uint16_t));
    std::vector<uint8_t> recurrent_before(parameters.recurrent_seed.size() *
                                          sizeof(float));
    const uint32_t slot = before_invalid.active_slot;
    if (ok) {
      ok = export_state_plane(
               state,
               slot == 0U ? SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT0
                          : SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT1,
               &conv_before, "checkpoint invalid conv snapshot") &&
           export_state_plane(
               state,
               slot == 0U ? SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT0
                          : SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT1,
               &recurrent_before, "checkpoint invalid recurrent snapshot");
    }
    ok = expect_status(sllm_linear_attention_state_commit_checkpoint(
                           context, queue, state, 0U, 3U, 3U, 2U, &error.sink),
                       SLLM_STATUS_INVALID_ARGUMENT, "checkpoint invalid row",
                       error) &&
         ok;
    sllm_linear_attention_view_info_t after_invalid{};
    if (ok) {
      ok = query_state(state, &after_invalid, "checkpoint invalid query") &&
           after_invalid.observed_length == before_invalid.observed_length &&
           after_invalid.generation == before_invalid.generation &&
           after_invalid.active_slot == before_invalid.active_slot;
      std::vector<uint8_t> conv_after(conv_before.size());
      std::vector<uint8_t> recurrent_after(recurrent_before.size());
      if (ok) {
        ok = export_state_plane(
                 state,
                 slot == 0U ? SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT0
                            : SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT1,
                 &conv_after, "checkpoint invalid conv reread") &&
             export_state_plane(
                 state,
                 slot == 0U ? SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT0
                            : SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT1,
                 &recurrent_after, "checkpoint invalid recurrent reread") &&
             conv_before == conv_after && recurrent_before == recurrent_after;
      }
    }
  }

  if (ok && mode < 2U) {
    const uint64_t prefix_end = static_cast<uint64_t>(mode + 1U);
    ok = expect_status(
        sllm_linear_attention_state_commit_checkpoint(
            context, queue, state, 0U, 3U, prefix_end, mode, &error.sink),
        SLLM_STATUS_OK, "checkpoint commit", error);
    sllm_linear_attention_view_info_t view{};
    if (ok) {
      ok = query_state(state, &view, "checkpoint prefix query") &&
           compare_view(view, prefix_end, 1U, "checkpoint prefix view") &&
           compare_state(state, snapshots[prefix_end],
                         "checkpoint prefix state");
    }
    std::vector<uint16_t> next_output;
    if (ok) {
      Batch next = make_batch(static_cast<uint32_t>(prefix_end), 1U);
      ok = execute_batch(context, queue, state, buffers, parameters, next,
                         prefix_end, &next_output) &&
           compare_output(next_output, oracle_outputs[prefix_end],
                          "checkpoint next M1") &&
           compare_state(state, snapshots[prefix_end + 1U],
                         "checkpoint next state");
    }
  }

  if (ok && mode == 2U) {
    ok = expect_status(
        sllm_linear_attention_state_discard_checkpoint(state, &error.sink),
        SLLM_STATUS_OK, "checkpoint full3 discard", error);
    if (ok) {
      sllm_linear_attention_view_info_t view{};
      ok = query_state(state, &view, "checkpoint full3 query") &&
           compare_view(view, 3U, 1U, "checkpoint full3 view") &&
           compare_state(state, snapshots[3U], "checkpoint full3 state");
    }
    std::vector<uint16_t> second_output;
    if (ok) {
      ok = expect_status(sllm_linear_attention_state_prepare_checkpoint(
                             state, 3U, 3U, 2U, &error.sink),
                         SLLM_STATUS_OK, "checkpoint rearm", error) &&
           execute_batch(context, queue, state, buffers, parameters,
                         second_batch, 3U, &second_output) &&
           expect_status(sllm_linear_attention_state_validate_checkpoint(
                             state, 3U, 6U, 2U, &error.sink),
                         SLLM_STATUS_OK, "checkpoint rearm validate", error) &&
           compare_output(second_output, oracle_outputs[3U],
                          "checkpoint rearm M3") &&
           expect_status(sllm_linear_attention_state_discard_checkpoint(
                             state, &error.sink),
                         SLLM_STATUS_OK, "checkpoint rearm discard", error);
    }
    if (ok) {
      sllm_linear_attention_view_info_t view{};
      ok = query_state(state, &view, "checkpoint rearm query") &&
           compare_view(view, 6U, 0U, "checkpoint rearm view") &&
           compare_state(state, snapshots[6U], "checkpoint rearm state");
    }
  }

  if (state != nullptr) {
    ok = expect_status(sllm_linear_attention_state_release(&state, &error.sink),
                       SLLM_STATUS_OK, "checkpoint state release", error) &&
         ok;
  }
  return ok;
}

bool run_batch_commit_case(
    const sllm_context_t *const context, const sllm_queue_t *const queue,
    const std::array<sllm_buffer_t *, 9U> &buffers,
    const Parameters &parameters, const Batch &first_batch,
    const std::array<OracleSnapshot, 7U> &snapshots,
    const std::vector<std::vector<uint16_t>> &oracle_outputs) {
  constexpr std::size_t kStateCount = 3U;
  std::array<sllm_linear_attention_state_t *, kStateCount> states{};
  Error error;
  bool ok = true;
  const uint64_t recurrent_bytes =
      static_cast<uint64_t>(parameters.recurrent_seed.size()) * sizeof(float);
  const uint64_t conv_bytes =
      static_cast<uint64_t>(parameters.conv_seed.size()) * sizeof(uint16_t);

  for (std::size_t index = 0U; ok && index != kStateCount; ++index) {
    ok = create_state(context, &states[index], 300U + index) &&
         upload_state_plane(states[index],
                            SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT0,
                            const_cast<uint16_t *>(parameters.conv_seed.data()),
                            conv_bytes, "batch seed conv") &&
         upload_state_plane(
             states[index], SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT0,
             const_cast<float *>(parameters.recurrent_seed.data()),
             recurrent_bytes, "batch seed recurrent") &&
         expect_status(sllm_linear_attention_state_prepare_checkpoint(
                           states[index], 0U, 3U, 2U, &error.sink),
                       SLLM_STATUS_OK, "batch prepare", error);
    std::vector<uint16_t> output;
    if (ok) {
      ok = execute_batch(context, queue, states[index], buffers, parameters,
                         first_batch, 0U, &output) &&
           compare_output(output, oracle_outputs[0], "batch M3") &&
           expect_status(sllm_linear_attention_state_validate_checkpoint(
                             states[index], 0U, 3U, 2U, &error.sink),
                         SLLM_STATUS_OK, "batch validate", error);
    }
  }

  std::array<StateSnapshot, kStateCount> before_invalid{};
  for (std::size_t index = 0U; ok && index != kStateCount; ++index) {
    ok = capture_state_snapshot(states[index], parameters,
                                &before_invalid[index],
                                "batch pre-invalid query");
  }

  const auto verify_unchanged = [&](const char *const label) {
    bool unchanged = true;
    for (std::size_t index = 0U; index != kStateCount; ++index) {
      StateSnapshot after;
      unchanged = capture_state_snapshot(states[index], parameters, &after,
                                         "batch post-invalid query") &&
                  snapshot_equal(before_invalid[index], after, label) &&
                  unchanged;
    }
    return unchanged;
  };

  std::array<const sllm_linear_attention_state_t *, kStateCount> duplicate = {
      states[0], states[0], states[2]};
  if (ok) {
    ok = expect_status(sllm_linear_attention_state_commit_checkpoint_batch(
                           context, queue, duplicate.data(), kStateCount, 0U,
                           3U, 2U, 1U, &error.sink),
                       SLLM_STATUS_INVALID_ARGUMENT,
                       "batch duplicate preflight", error) &&
         verify_unchanged("batch duplicate preflight");
  }

  const std::array<const sllm_linear_attention_state_t *, kStateCount> valid = {
      states[0], states[1], states[2]};
  if (ok) {
    ok = expect_status(sllm_linear_attention_state_commit_checkpoint_batch(
                           context, queue, valid.data(), kStateCount, 1U, 4U,
                           2U, 1U, &error.sink),
                       SLLM_STATUS_INVALID_ARGUMENT,
                       "batch mismatched preflight", error) &&
         verify_unchanged("batch mismatched preflight");
  }

  sllm_linear_attention_state_t *stale_state = nullptr;
  const sllm_linear_attention_state_t *stale_raw = nullptr;
  if (ok) {
    ok = create_state(context, &stale_state, 399U);
    stale_raw = stale_state;
    if (ok) {
      ok = expect_status(
          sllm_linear_attention_state_release(&stale_state, &error.sink),
          SLLM_STATUS_OK, "batch stale state release", error);
    }
  }
  std::array<const sllm_linear_attention_state_t *, kStateCount> stale = {
      states[0], states[1], stale_raw};
  if (ok) {
    ok = expect_status(sllm_linear_attention_state_commit_checkpoint_batch(
                           context, queue, stale.data(), kStateCount, 0U, 3U,
                           2U, 1U, &error.sink),
                       SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                       "batch stale preflight", error) &&
         verify_unchanged("batch stale preflight");
  }

  if (ok) {
    ok = expect_status(sllm_linear_attention_state_commit_checkpoint_batch(
                           context, queue, valid.data(), kStateCount, 0U, 3U,
                           2U, 1U, &error.sink),
                       SLLM_STATUS_OK, "batch commit row one", error);
  }
  for (std::size_t index = 0U; ok && index != kStateCount; ++index) {
    sllm_linear_attention_view_info_t view{};
    ok = query_state(states[index], &view, "batch committed query") &&
         compare_view(view, 2U, 1U, "batch committed view") &&
         compare_state(states[index], snapshots[2U], "batch committed state");
  }

  const Batch next_batch = make_batch(2U, 1U);
  for (std::size_t index = 0U; ok && index != kStateCount; ++index) {
    std::vector<uint16_t> output;
    ok = execute_batch(context, queue, states[index], buffers, parameters,
                       next_batch, 2U, &output) &&
         compare_output(output, oracle_outputs[2U], "batch next M1") &&
         compare_state(states[index], snapshots[3U], "batch next state");
  }

  for (auto &state : states) {
    if (state != nullptr) {
      ok = expect_status(
               sllm_linear_attention_state_release(&state, &error.sink),
               SLLM_STATUS_OK, "batch state release", error) &&
           ok;
    }
  }
  if (stale_state != nullptr) {
    ok = expect_status(
             sllm_linear_attention_state_release(&stale_state, &error.sink),
             SLLM_STATUS_OK, "batch stale cleanup", error) &&
         ok;
  }
  return ok;
}

bool run_rewind_case(const sllm_context_t *const context,
                     const sllm_queue_t *const queue,
                     const std::array<sllm_buffer_t *, 9U> &buffers,
                     const Parameters &parameters, const Batch &first_batch,
                     const std::array<OracleSnapshot, 7U> &snapshots) {
  sllm_linear_attention_state_t *state = nullptr;
  bool ok = create_state(context, &state, 199U);
  Error error;
  const uint64_t recurrent_bytes =
      static_cast<uint64_t>(parameters.recurrent_seed.size()) * sizeof(float);
  const uint64_t conv_bytes =
      static_cast<uint64_t>(parameters.conv_seed.size()) * sizeof(uint16_t);
  if (ok) {
    ok = upload_state_plane(state, SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT0,
                            const_cast<uint16_t *>(parameters.conv_seed.data()),
                            conv_bytes, "rewind seed conv") &&
         upload_state_plane(
             state, SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT0,
             const_cast<float *>(parameters.recurrent_seed.data()),
             recurrent_bytes, "rewind seed recurrent") &&
         expect_status(sllm_linear_attention_state_prepare_checkpoint(
                           state, 0U, 3U, 2U, &error.sink),
                       SLLM_STATUS_OK, "rewind prepare", error);
  }
  std::vector<uint16_t> output;
  if (ok) {
    output.resize(3U * kOutputWidth);
    ok = execute_batch(context, queue, state, buffers, parameters, first_batch,
                       0U, &output) &&
         expect_status(sllm_linear_attention_state_validate_checkpoint(
                           state, 0U, 3U, 2U, &error.sink),
                       SLLM_STATUS_OK, "rewind validate", error) &&
         expect_status(sllm_linear_attention_state_commit_checkpoint(
                           context, queue, state, 0U, 3U, 2U, 1U, &error.sink),
                       SLLM_STATUS_OK, "rewind commit", error) &&
         compare_state(state, snapshots[2U], "rewind committed prefix");
  }
  if (ok) {
    ok = expect_status(
        sllm_linear_attention_state_rewind_last(state, 2U, 0U, &error.sink),
        SLLM_STATUS_OK, "rewind last", error);
    sllm_linear_attention_view_info_t view{};
    if (ok) {
      ok = query_state(state, &view, "rewind query") &&
           compare_view(view, 0U, 0U, "rewind restored view");
    }
    if (ok) {
      std::vector<uint8_t> conv(conv_bytes);
      std::vector<uint8_t> recurrent(recurrent_bytes);
      ok =
          export_state_plane(state, SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT0,
                             &conv, "rewind conv export") &&
          export_state_plane(state, SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT0,
                             &recurrent, "rewind recurrent export") &&
          std::memcmp(conv.data(), parameters.conv_seed.data(), conv_bytes) ==
              0 &&
          std::memcmp(recurrent.data(), parameters.recurrent_seed.data(),
                      recurrent_bytes) == 0;
      if (!ok) {
        std::cerr
            << "rewind did not restore the captured pre-M3 state payload\n";
      }
    }
  }
  if (state != nullptr) {
    ok = expect_status(sllm_linear_attention_state_release(&state, &error.sink),
                       SLLM_STATUS_OK, "rewind state release", error) &&
         ok;
  }
  return ok;
}

bool run_control_case(const sllm_context_t *const context,
                      const sllm_queue_t *const queue,
                      const std::array<sllm_buffer_t *, 9U> &buffers,
                      const Parameters &parameters, const Batch &first_batch,
                      const OracleSnapshot &expected_state,
                      const std::vector<uint16_t> &expected_output) {
  sllm_linear_attention_state_t *state = nullptr;
  bool ok = create_state(context, &state, 250U);
  Error error;
  const uint64_t recurrent_bytes =
      static_cast<uint64_t>(parameters.recurrent_seed.size()) * sizeof(float);
  const uint64_t conv_bytes =
      static_cast<uint64_t>(parameters.conv_seed.size()) * sizeof(uint16_t);
  if (ok) {
    ok = upload_state_plane(state, SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT0,
                            const_cast<uint16_t *>(parameters.conv_seed.data()),
                            conv_bytes, "control seed conv") &&
         upload_state_plane(
             state, SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT0,
             const_cast<float *>(parameters.recurrent_seed.data()),
             recurrent_bytes, "control seed recurrent");
  }
  std::vector<uint16_t> output;
  if (ok) {
    ok = execute_batch(context, queue, state, buffers, parameters, first_batch,
                       0U, &output) &&
         compare_output(output, expected_output, "checkpoint control M3") &&
         compare_state(state, expected_state, "checkpoint control state");
  }
  if (state != nullptr) {
    ok = expect_status(sllm_linear_attention_state_release(&state, &error.sink),
                       SLLM_STATUS_OK, "control state release", error) &&
         ok;
  }
  return ok;
}

} // namespace

int main() {
  const std::string expected_target = SLLM_TEST_EXPECTED_TARGET;
  if (expected_target != "gfx1030" && expected_target != "gfx1201") {
    std::cerr << "checkpoint test requires exact gfx1030 or gfx1201 target\n";
    return 2;
  }

  sllm_device_info_t device{};
  device.struct_size = sizeof(device);
  device.abi_version = SLLM_HIP_ABI_VERSION;
  Error error;
  if (!expect_status(sllm_device_query(0U, &device, &error.sink),
                     SLLM_STATUS_OK, "checkpoint device query", error) ||
      std::string(device.gcn_arch_name) != expected_target) {
    std::cerr << "checkpoint test selected an unexpected GPU target\n";
    return 2;
  }

  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  std::array<sllm_buffer_t *, 9U> buffers{};
  bool ok = true;
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::strncpy(context_info.expected_gcn_arch_name, expected_target.c_str(),
               sizeof(context_info.expected_gcn_arch_name) - 1U);
  ok = expect_status(sllm_context_create(&context_info, &context, &error.sink),
                     SLLM_STATUS_OK, "checkpoint context create", error);
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  if (ok) {
    ok = expect_status(
        sllm_queue_create(context, &queue_info, &queue, &error.sink),
        SLLM_STATUS_OK, "checkpoint queue create", error);
  }
  const std::array<uint64_t, 9U> sizes = {
      static_cast<uint64_t>(kMaxRows) * kQkvWidth * sizeof(uint16_t),
      static_cast<uint64_t>(kMaxRows) * kOutputWidth * sizeof(uint16_t),
      static_cast<uint64_t>(kMaxRows) * kValueHeads * sizeof(uint16_t),
      static_cast<uint64_t>(kMaxRows) * kValueHeads * sizeof(uint16_t),
      static_cast<uint64_t>(kQkvWidth) * kConvKernel * sizeof(uint16_t),
      static_cast<uint64_t>(kValueHeads) * sizeof(float),
      static_cast<uint64_t>(kValueHeads) * sizeof(uint16_t),
      static_cast<uint64_t>(kHeadDim) * sizeof(float),
      static_cast<uint64_t>(kMaxRows) * kOutputWidth * sizeof(uint16_t)};
  for (std::size_t index = 0U; ok && index != buffers.size(); ++index) {
    sllm_buffer_create_info_t info{};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    info.size_bytes = sizes[index];
    ok = expect_status(
        sllm_buffer_create(context, &info, &buffers[index], &error.sink),
        SLLM_STATUS_OK, "checkpoint buffer create", error);
  }

  Parameters parameters;
  const Batch first_batch = make_batch(0U, 3U);
  const Batch second_batch = make_batch(3U, 3U);
  std::array<OracleSnapshot, 7U> snapshots{};
  std::vector<std::vector<uint16_t>> oracle_outputs(4U);
  if (ok) {
    if (!upload(queue, buffers[4], parameters.conv_weight.data(), sizes[4],
                "checkpoint conv weight upload") ||
        !upload(queue, buffers[5], parameters.a_log.data(), sizes[5],
                "checkpoint a_log upload") ||
        !upload(queue, buffers[6], parameters.dt_bias.data(), sizes[6],
                "checkpoint dt_bias upload") ||
        !upload(queue, buffers[7], parameters.norm_weight.data(), sizes[7],
                "checkpoint norm upload")) {
      ok = false;
    }
  }
  if (ok) {
    GdnOracle oracle(parameters);
    snapshots[0] = oracle.snapshot();
    oracle_outputs[0].reserve(3U * kOutputWidth);
    for (uint32_t row = 0U; row != 3U; ++row) {
      const std::vector<uint16_t> output = oracle.step(first_batch, row);
      oracle_outputs[0].insert(oracle_outputs[0].end(), output.begin(),
                               output.end());
      snapshots[row + 1U] = oracle.snapshot();
    }
    oracle_outputs[1] = oracle_outputs[0];
    oracle_outputs[2] = oracle_outputs[0];
    oracle_outputs[3].reserve(3U * kOutputWidth);
    for (uint32_t row = 0U; row != 3U; ++row) {
      const std::vector<uint16_t> output = oracle.step(second_batch, row);
      oracle_outputs[3].insert(oracle_outputs[3].end(), output.begin(),
                               output.end());
      snapshots[row + 4U] = oracle.snapshot();
    }
  }
  if (ok) {
    /* The three acceptance modes use the same M3 oracle.  For accept1/2, the
     * next M1 expected output is derived below from a fresh sequential oracle.
     */
    for (uint32_t mode = 0U; mode != 2U; ++mode) {
      GdnOracle oracle(parameters);
      std::vector<uint16_t> expected;
      for (uint32_t row = 0U; row != mode + 2U; ++row) {
        const std::vector<uint16_t> one =
            row < 3U ? oracle.step(first_batch, row)
                     : oracle.step(make_batch(3U, 1U), 0U);
        if (row == mode + 1U) {
          expected = one;
        }
      }
      oracle_outputs[static_cast<std::size_t>(mode) + 1U] = expected;
    }
  }
  if (ok) {
    ok = run_control_case(context, queue, buffers, parameters, first_batch,
                          snapshots[3U], oracle_outputs[0]) &&
         run_case(context, queue, buffers, parameters, first_batch,
                  second_batch, snapshots, oracle_outputs, 0U) &&
         run_case(context, queue, buffers, parameters, first_batch,
                  second_batch, snapshots, oracle_outputs, 1U) &&
         run_case(context, queue, buffers, parameters, first_batch,
                  second_batch, snapshots, oracle_outputs, 2U) &&
         run_batch_commit_case(context, queue, buffers, parameters, first_batch,
                               snapshots, oracle_outputs) &&
         run_rewind_case(context, queue, buffers, parameters, first_batch,
                         snapshots);
  }

  for (auto &buffer : buffers) {
    if (buffer != nullptr) {
      ok = expect_status(sllm_buffer_release(&buffer, &error.sink),
                         SLLM_STATUS_OK, "checkpoint buffer release", error) &&
           ok;
    }
  }
  if (queue != nullptr) {
    ok = expect_status(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
                       "checkpoint queue release", error) &&
         ok;
  }
  if (context != nullptr) {
    ok = expect_status(sllm_context_release(&context, &error.sink),
                       SLLM_STATUS_OK, "checkpoint context release", error) &&
         ok;
  }
  std::cout << "phase83_5_linear_checkpoint_gpu_test status="
            << (ok ? "PASS" : "FAIL") << " target=" << expected_target << '\n';
  return ok ? 0 : 1;
}
