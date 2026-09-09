#include "sllm/hip.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1030"
#endif

namespace {

constexpr uint32_t kTimeoutMs = 30'000U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kQueryHeads = 24U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint64_t kElementsPerToken =
    static_cast<uint64_t>(kKvHeads) * kHeadDim;
constexpr uint64_t kQueryElements =
    static_cast<uint64_t>(kQueryHeads) * kHeadDim;
constexpr float kAttentionScale = 1.0F / 16.0F;

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

bool release_buffer(sllm_buffer_t **const buffer) {
  if (buffer == nullptr || *buffer == nullptr) {
    return true;
  }
  Error error;
  const bool ok = expect(sllm_buffer_release(buffer, &error.sink),
                         SLLM_STATUS_OK, "buffer release", error);
  return ok && *buffer == nullptr;
}

bool release_state(sllm_kv_state_t **const state) {
  if (state == nullptr || *state == nullptr) {
    return true;
  }
  Error error;
  const bool ok = expect(sllm_kv_state_release(state, &error.sink),
                         SLLM_STATUS_OK, "KV state release", error);
  return ok && *state == nullptr;
}

bool wait_and_release(sllm_completion_t **const completion,
                      const char *const operation) {
  if (completion == nullptr || *completion == nullptr) {
    std::cerr << operation << " returned no completion\n";
    return false;
  }
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  const bool waited = expect(sllm_completion_wait(*completion, kTimeoutMs,
                                                  &result, &error.sink),
                             SLLM_STATUS_OK, operation, error) &&
                      result.state == SLLM_COMPLETION_STATE_SUCCESS;
  const bool released = expect(sllm_completion_release(completion, &error.sink),
                               SLLM_STATUS_OK, "completion release", error) &&
                        *completion == nullptr;
  return waited && released;
}

bool upload(const sllm_queue_t *const queue, const sllm_buffer_t *const buffer,
            const void *const source, const uint64_t size_bytes) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = const_cast<void *>(source);
  transfer.size_bytes = size_bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, "buffer upload", error)) {
    return false;
  }
  return wait_and_release(&completion, "buffer upload wait");
}

bool download(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer, void *const destination,
              const uint64_t size_bytes) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.size_bytes = size_bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_buffer_copy_d2h(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, "buffer download", error) ||
      completion == nullptr) {
    return false;
  }
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(
          sllm_completion_wait(completion, kTimeoutMs, &result, &error.sink),
          SLLM_STATUS_OK, "buffer download wait", error) ||
      result.state != SLLM_COMPLETION_STATE_SUCCESS) {
    (void)sllm_completion_release(&completion, &error.sink);
    return false;
  }
  uint64_t written = 0U;
  const bool read =
      expect(sllm_completion_read(completion, destination, size_bytes, &written,
                                  &error.sink),
             SLLM_STATUS_OK, "buffer download read", error) &&
      written == size_bytes;
  const bool released =
      expect(sllm_completion_release(&completion, &error.sink), SLLM_STATUS_OK,
             "buffer download completion release", error) &&
      completion == nullptr;
  return read && released;
}

bool create_buffer(const sllm_context_t *const context, const uint64_t bytes,
                   sllm_buffer_t **const output) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = bytes;
  Error error;
  return expect(sllm_buffer_create(context, &info, output, &error.sink),
                SLLM_STATUS_OK, "buffer create", error) &&
         *output != nullptr;
}

sllm_tensor_binding_t binding(const sllm_buffer_t *const buffer,
                              const uint32_t dtype, const uint32_t rank,
                              const uint64_t *const shape,
                              const uint64_t byte_offset = 0U) {
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.byte_offset = byte_offset;
  result.dtype = dtype;
  result.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  result.rank = rank;
  uint64_t stride = 1U;
  for (uint32_t reverse = 0U; reverse != rank; ++reverse) {
    const uint32_t index = rank - reverse - 1U;
    result.shape[index] = shape[index];
    result.stride_elements[index] = stride;
    stride *= shape[index];
  }
  return result;
}

bool create_visible_context(sllm_context_t **const context,
                            sllm_queue_t **const queue) {
  Error error;
  uint32_t device_count = 0U;
  if (!expect(sllm_device_count(&device_count, &error.sink), SLLM_STATUS_OK,
              "device count", error) ||
      device_count != 1U) {
    std::cerr << "test requires exactly one visible GPU, got " << device_count
              << '\n';
    return false;
  }
  sllm_device_info_t device{};
  device.struct_size = sizeof(device);
  device.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(sllm_device_query(0U, &device, &error.sink), SLLM_STATUS_OK,
              "device query", error) ||
      std::strcmp(device.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) != 0) {
    std::cerr << "visible target is not " << SLLM_TEST_EXPECTED_TARGET << '\n';
    return false;
  }
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::snprintf(context_info.expected_gcn_arch_name,
                sizeof(context_info.expected_gcn_arch_name), "%s",
                SLLM_TEST_EXPECTED_TARGET);
  if (!expect(sllm_context_create(&context_info, context, &error.sink),
              SLLM_STATUS_OK, "context create", error) ||
      *context == nullptr) {
    return false;
  }
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  return expect(sllm_queue_create(*context, &queue_info, queue, &error.sink),
                SLLM_STATUS_OK, "queue create", error) &&
         *queue != nullptr;
}

uint16_t f32_to_bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t upper = bits >> 16U;
  bits += UINT32_C(0x7fff) + (upper & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

float bf16_to_f32(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

// The test's oracle deliberately re-quantizes and decodes the fractional BF16
// source.  This independent scalar implementation exercises the native OCP
// E4M3FN/E8M0 scale and rounding contract without copying a device codec into
// the test.
uint8_t host_scale_code(const float maximum) {
  if (!(maximum > 0.0F) || !std::isfinite(maximum)) {
    return UINT8_C(127);
  }
  const int exponent = std::ilogb(maximum) - 8;
  return static_cast<uint8_t>(std::clamp(exponent + 127, 0, 254));
}

uint8_t host_e4m3fn_encode(const float value) {
  const uint8_t sign = std::signbit(value) ? UINT8_C(0x80) : UINT8_C(0);
  const float magnitude = std::fabs(value);
  if (magnitude == 0.0F) {
    return sign;
  }
  if (magnitude >= 448.0F || !std::isfinite(magnitude)) {
    return static_cast<uint8_t>(sign | UINT8_C(0x7e));
  }
  const uint32_t bits = [&]() {
    uint32_t raw = 0U;
    std::memcpy(&raw, &magnitude, sizeof(raw));
    return raw;
  }();
  const uint32_t rounded =
      bits + UINT32_C(0x0007ffff) + ((bits >> 20U) & UINT32_C(1));
  const uint32_t exponent = ((rounded >> 23U) & UINT32_C(0xff)) - 120U;
  const uint32_t code = (exponent << 3U) | ((rounded >> 20U) & UINT32_C(7));
  return static_cast<uint8_t>(
      sign | static_cast<uint8_t>(std::min(code, static_cast<uint32_t>(0x7e))));
}

float host_e4m3fn_decode(const uint8_t value) {
  const float sign = (value & UINT8_C(0x80)) != 0U ? -1.0F : 1.0F;
  const uint32_t magnitude = value & UINT8_C(0x7f);
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & UINT32_C(7);
  if (exponent == 0U) {
    return sign * static_cast<float>(mantissa) * std::ldexp(1.0F, -9);
  }
  return sign * std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                           static_cast<int>(exponent) - 7);
}

struct QuantizedRow final {
  std::array<uint8_t, kHeadDim> values{};
  std::array<uint8_t, kHeadDim / 32U> scales{};
};

QuantizedRow quantize_row(const std::vector<uint16_t> &source,
                          const uint64_t token, const uint32_t head,
                          const uint64_t token_stride) {
  QuantizedRow result{};
  const uint64_t row = (token * kKvHeads + head) * token_stride;
  for (uint32_t block = 0U; block != kHeadDim / 32U; ++block) {
    float maximum = 0.0F;
    for (uint32_t lane = 0U; lane != 32U; ++lane) {
      maximum = std::max(
          maximum, std::fabs(bf16_to_f32(source[row + block * 32U + lane])));
    }
    const uint8_t scale = host_scale_code(maximum);
    result.scales[block] = scale;
    const float scale_value = std::ldexp(1.0F, static_cast<int>(scale) - 127);
    for (uint32_t lane = 0U; lane != 32U; ++lane) {
      const float normalized =
          bf16_to_f32(source[row + block * 32U + lane]) / scale_value;
      result.values[block * 32U + lane] = host_e4m3fn_encode(normalized);
    }
  }
  return result;
}

float decoded_value(const QuantizedRow &row, const uint32_t dimension) {
  const uint8_t scale = row.scales[dimension / 32U];
  const float scale_value = std::ldexp(1.0F, static_cast<int>(scale) - 127);
  return host_e4m3fn_decode(row.values[dimension]) * scale_value;
}

std::vector<uint16_t> make_kv(const bool value_plane, const bool mutated,
                              const uint64_t token_count) {
  std::vector<uint16_t> result(
      static_cast<std::size_t>(token_count * kElementsPerToken));
  for (uint64_t token = 0U; token != token_count; ++token) {
    for (uint32_t head = 0U; head != kKvHeads; ++head) {
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        const uint32_t block = dimension / 32U;
        // Keep the source in the normal finite range while varying each
        // block's amax, lane, head, token, and mutation.  The resulting
        // normalized values are intentionally non-powers-of-two so the
        // independent E4M3FN oracle exercises round-to-nearest encoding.
        const float block_base = value_plane ? 0.21F : 0.29F;
        const float block_step = value_plane ? 0.07F : 0.09F;
        const float head_step = value_plane ? 0.019F : 0.023F;
        const float token_step = value_plane ? 0.008F : 0.009F;
        const float lane_step = value_plane ? 0.0019F : 0.0023F;
        float magnitude = block_base + block_step * static_cast<float>(block) +
                          head_step * static_cast<float>(head) +
                          token_step * static_cast<float>(token % 5U) +
                          lane_step * static_cast<float>(dimension % 13U);
        if (mutated) {
          magnitude += value_plane ? 0.037F : 0.041F;
        }
        const bool negative =
            ((token + dimension + head + (mutated ? 1U : 0U)) % 5U) == 0U;
        if (negative) {
          magnitude = -magnitude;
        }
        const uint64_t index = (token * kKvHeads + head) * kHeadDim + dimension;
        result[static_cast<std::size_t>(index)] = f32_to_bf16(magnitude);
      }
    }
  }
  return result;
}

std::vector<uint16_t> make_query(const bool mutated) {
  std::vector<uint16_t> result(static_cast<std::size_t>(kQueryElements));
  for (uint32_t head = 0U; head != kQueryHeads; ++head) {
    for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
      const uint32_t block = dimension / 32U;
      // Smaller fractional queries keep the score range away from a hard
      // one-hot softmax while retaining sign/lane/head/block variation.
      float value = 0.035F + 0.002F * static_cast<float>(block) +
                    0.001F * static_cast<float>(head) +
                    0.0004F * static_cast<float>(dimension % 17U);
      if (mutated) {
        value += 0.006F;
      }
      if (((head + dimension + (mutated ? 1U : 0U)) % 7U) == 0U) {
        value = -value;
      }
      result[static_cast<std::size_t>(head * kHeadDim + dimension)] =
          f32_to_bf16(value);
    }
  }
  return result;
}

bool query_length(const sllm_kv_state_t *const state, const uint64_t expected) {
  sllm_kv_view_info_t view{};
  view.struct_size = sizeof(view);
  view.abi_version = SLLM_HIP_ABI_VERSION;
  view.info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
  Error error;
  return expect(sllm_kv_state_query(state, &view, &error.sink), SLLM_STATUS_OK,
                "KV state query", error) &&
         view.observed_length == expected &&
         view.encoding == SLLM_TENSOR_ENCODING_MXFP8_BLOCK32_E8M0;
}

uint32_t bf16_ulp_distance(const uint16_t lhs, const uint16_t rhs) {
  const auto ordered = [](const uint16_t value) {
    const uint32_t magnitude = value & UINT16_C(0x7fff);
    return (value & UINT16_C(0x8000)) != 0U ? UINT32_C(0x8000) - magnitude
                                            : UINT32_C(0x8000) + value;
  };
  const uint32_t lhs_ordered = ordered(lhs);
  const uint32_t rhs_ordered = ordered(rhs);
  return lhs_ordered >= rhs_ordered ? lhs_ordered - rhs_ordered
                                    : rhs_ordered - lhs_ordered;
}

sllm_kv_append_desc_t append_descriptor(const sllm_buffer_t *const key,
                                        const sllm_buffer_t *const value,
                                        const uint64_t count,
                                        const uint64_t position) {
  const uint64_t shape[] = {count, kKvHeads, kHeadDim};
  const uint64_t offset = position * kElementsPerToken * sizeof(uint16_t);
  sllm_kv_append_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.append_version = SLLM_HIP_KV_STATE_VERSION;
  descriptor.expected_length = position;
  descriptor.start_position = position;
  descriptor.key_input =
      binding(key, SLLM_TENSOR_DTYPE_BF16, 3U, shape, offset);
  descriptor.value_input =
      binding(value, SLLM_TENSOR_DTYPE_BF16, 3U, shape, offset);
  return descriptor;
}

bool validate_append_info(const sllm_kv_append_info_t &info,
                          const uint64_t count, const uint64_t position) {
  return info.backend == SLLM_BACKEND_HIP && info.dispatch_id != 0U &&
         info.dispatch_count == 1U &&
         info.kernel_id ==
             SLLM_HIP_KV_KERNEL_ID_BF16_TO_MXFP8_E4_TOKEN_MAJOR_V1 &&
         info.start_position == position && info.token_count == count &&
         info.end_position == position + count && info.commit_allowed == 1U &&
         info.fallback_allowed == 0U && info.fallback_used == 0U &&
         std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;
}

std::vector<uint16_t> make_query_multi(const uint32_t query_count,
                                       const bool mutated) {
  const std::vector<uint16_t> base = make_query(mutated);
  std::vector<uint16_t> result(static_cast<std::size_t>(
      static_cast<uint64_t>(query_count) * kQueryElements));
  for (uint32_t query_index = 0U; query_index != query_count; ++query_index) {
    for (std::size_t index = 0U; index != base.size(); ++index) {
      float value = bf16_to_f32(base[index]);
      value *= 1.0F + 0.007F * static_cast<float>(query_index);
      if (((query_index + index) % 29U) == 0U) {
        value = -value;
      }
      result[static_cast<std::size_t>(query_index * kQueryElements + index)] =
          f32_to_bf16(value);
    }
  }
  return result;
}

std::vector<uint16_t> public_oracle(const std::vector<uint16_t> &key,
                                    const std::vector<uint16_t> &value,
                                    const std::vector<uint16_t> &query,
                                    const uint64_t committed_length,
                                    const uint32_t query_count) {
  const uint64_t start_position = committed_length - query_count;
  std::array<std::vector<QuantizedRow>, kKvHeads> key_rows{};
  std::array<std::vector<QuantizedRow>, kKvHeads> value_rows{};
  for (uint32_t head = 0U; head != kKvHeads; ++head) {
    key_rows[head].reserve(static_cast<std::size_t>(committed_length));
    value_rows[head].reserve(static_cast<std::size_t>(committed_length));
    for (uint64_t token = 0U; token != committed_length; ++token) {
      key_rows[head].push_back(quantize_row(key, token, head, kHeadDim));
      value_rows[head].push_back(quantize_row(value, token, head, kHeadDim));
    }
  }
  std::vector<uint16_t> expected(static_cast<std::size_t>(
      static_cast<uint64_t>(query_count) * kQueryElements));
  for (uint32_t query_index = 0U; query_index != query_count; ++query_index) {
    const uint64_t row_length = start_position + query_index + 1U;
    for (uint32_t query_head = 0U; query_head != kQueryHeads; ++query_head) {
      const uint32_t kv_head = query_head / (kQueryHeads / kKvHeads);
      std::vector<float> scores(static_cast<std::size_t>(row_length));
      float maximum = -std::numeric_limits<float>::infinity();
      for (uint64_t token = 0U; token != row_length; ++token) {
        float score = 0.0F;
        const QuantizedRow &key_row = key_rows[kv_head][token];
        for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
          score += bf16_to_f32(query[static_cast<std::size_t>(
                       (query_index * kQueryHeads + query_head) * kHeadDim +
                       dimension)]) *
                   decoded_value(key_row, dimension);
        }
        score *= kAttentionScale;
        scores[static_cast<std::size_t>(token)] = score;
        maximum = std::max(maximum, score);
      }
      float denominator = 0.0F;
      for (float &score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        float accumulated = 0.0F;
        for (uint64_t token = 0U; token != row_length; ++token) {
          accumulated +=
              (scores[static_cast<std::size_t>(token)] / denominator) *
              decoded_value(value_rows[kv_head][token], dimension);
        }
        expected[static_cast<std::size_t>(
            (query_index * kQueryHeads + query_head) * kHeadDim + dimension)] =
            f32_to_bf16(accumulated);
      }
    }
  }
  return expected;
}

bool compare_public_output(const std::vector<uint16_t> &observed,
                           const std::vector<uint16_t> &expected,
                           const uint64_t length, const uint32_t query_count) {
  if (observed.size() != expected.size()) {
    std::cerr << "public output size mismatch\n";
    return false;
  }
  float max_abs = 0.0F;
  float max_relative = 0.0F;
  uint32_t max_ulp = 0U;
  for (std::size_t index = 0U; index != observed.size(); ++index) {
    const float actual = bf16_to_f32(observed[index]);
    const float wanted = bf16_to_f32(expected[index]);
    if (!std::isfinite(actual) || !std::isfinite(wanted)) {
      std::cerr << "public oracle nonfinite index=" << index << '\n';
      return false;
    }
    const float difference = std::fabs(actual - wanted);
    max_abs = std::max(max_abs, difference);
    if (std::fabs(wanted) >= 0.125F) {
      max_relative = std::max(max_relative, difference / std::fabs(wanted));
    }
    max_ulp =
        std::max(max_ulp, bf16_ulp_distance(observed[index], expected[index]));
  }
  std::cout << "public length=" << length << " M=" << query_count
            << " oracle max_abs=" << max_abs << " max_rel=" << max_relative
            << " max_ulp=" << max_ulp << '\n';
  return max_ulp <= 4U && max_abs <= 0.03125F && max_relative <= 0.04F;
}

bool execute_public_attention(const sllm_context_t *const context,
                              const sllm_queue_t *const queue,
                              const sllm_kv_state_t *const state,
                              const sllm_buffer_t *const query,
                              const sllm_buffer_t *const output,
                              const uint64_t start, const uint64_t length,
                              const uint32_t query_count,
                              std::vector<uint16_t> *const observed) {
  const uint64_t shape[] = {query_count, kQueryHeads, kHeadDim};
  sllm_causal_attention_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_CAUSAL_ATTENTION_VERSION;
  descriptor.start_position = start;
  descriptor.expected_kv_length = length;
  descriptor.kv_state = state;
  descriptor.query = binding(query, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
  descriptor.output = binding(output, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
  sllm_causal_attention_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_causal_attention_execute(context, queue, &descriptor,
                                            &completion, &info, &error.sink),
              SLLM_STATUS_OK, "public MXFP8 attention", error) ||
      completion == nullptr) {
    if (completion != nullptr) {
      (void)sllm_completion_release(&completion, &error.sink);
    }
    return false;
  }
  const bool staged = length >= 1024U;
  const bool metadata_ok =
      info.backend == SLLM_BACKEND_HIP && info.dispatch_id != 0U &&
      info.query_count == query_count && info.start_position == start &&
      info.committed_kv_length == length && info.q_heads == kQueryHeads &&
      info.kv_heads == kKvHeads && info.head_dim == kHeadDim &&
      info.scale_denominator == 16U && info.fallback_allowed == 0U &&
      info.fallback_used == 0U &&
      std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0 &&
      (staged
           ? (info.dispatch_count == 2U &&
              info.kernel_id ==
                  SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_DECODE_WAVE_SPLIT_STAGED_GFX1030_V1 &&
              info.workgroup_size_x == 256U &&
              info.grid_size_x == query_count * kQueryHeads * 8U &&
              std::strcmp(
                  info.kernel_symbol,
                  "causal_attention.decode.wave8_split.staged.gfx1030.v1") ==
                  0 &&
              std::strcmp(info.device_symbol,
                          "sllm_causal_attention_decode_wave8_split_staged_"
                          "gfx1030_v1") == 0)
           : (info.dispatch_count == 1U &&
              info.kernel_id ==
                  SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PACKED_KV_V3));
  std::cout << "public length=" << length << " M=" << query_count
            << " selected_kernel_id=" << info.kernel_id
            << " dispatch_count=" << info.dispatch_count
            << " grid=" << info.grid_size_x
            << " workgroup=" << info.workgroup_size_x
            << " workspace_expected=" << (staged ? query_count * 198144U : 0U)
            << " symbol=" << info.kernel_symbol << '\n';
  if (!metadata_ok) {
    std::cerr << "public selector metadata mismatch at length=" << length
              << " M=" << query_count << '\n';
    (void)sllm_completion_release(&completion, &error.sink);
    return false;
  }
  if (!wait_and_release(&completion, "public MXFP8 attention wait")) {
    return false;
  }
  observed->assign(static_cast<std::size_t>(static_cast<uint64_t>(query_count) *
                                            kQueryElements),
                   0U);
  return download(queue, output, observed->data(),
                  static_cast<uint64_t>(observed->size()) * sizeof(uint16_t));
}

bool run_public_case(const sllm_context_t *const context,
                     const sllm_queue_t *const queue, const uint64_t length,
                     const uint32_t query_count) {
  const uint64_t start = length - query_count;
  const std::vector<uint16_t> key = make_kv(false, false, length);
  const std::vector<uint16_t> value = make_kv(true, false, length);
  const std::vector<uint16_t> query = make_query_multi(query_count, false);
  const std::vector<uint16_t> expected =
      public_oracle(key, value, query, length, query_count);
  const uint64_t kv_bytes = length * kElementsPerToken * sizeof(uint16_t);
  const uint64_t query_bytes =
      static_cast<uint64_t>(query.size()) * sizeof(uint16_t);
  sllm_kv_state_t *state = nullptr;
  std::array<sllm_buffer_t *, 4> buffers{};
  Error error;
  bool success = false;
  do {
    sllm_kv_state_create_info_v2_t create{};
    create.struct_size = sizeof(create);
    create.abi_version = SLLM_HIP_ABI_VERSION;
    create.create_info_version = SLLM_HIP_KV_STATE_CREATE_INFO_V2_VERSION;
    create.session_id = UINT64_C(0x8302);
    create.layer_id = 83U;
    create.capacity_tokens = length;
    create.head_count = kKvHeads;
    create.head_dim = kHeadDim;
    create.memory_kind = SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS;
    create.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
    create.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
    create.encoding = SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;
    create.block_size = 32U;
    create.scale_dtype = SLLM_TENSOR_DTYPE_U8;
    if (!expect(sllm_kv_state_create_v2(context, &create, &state, &error.sink),
                SLLM_STATUS_OK, "public MXFP8 state create", error) ||
        state == nullptr || !create_buffer(context, kv_bytes, &buffers[0]) ||
        !create_buffer(context, kv_bytes, &buffers[1]) ||
        !create_buffer(context, query_bytes, &buffers[2]) ||
        !create_buffer(context, query_bytes, &buffers[3]) ||
        !upload(queue, buffers[0], key.data(), kv_bytes) ||
        !upload(queue, buffers[1], value.data(), kv_bytes) ||
        !upload(queue, buffers[2], query.data(), query_bytes)) {
      break;
    }
    sllm_kv_append_desc_t append =
        append_descriptor(buffers[0], buffers[1], length, 0U);
    sllm_kv_append_info_t append_info{};
    append_info.struct_size = sizeof(append_info);
    append_info.abi_version = SLLM_HIP_ABI_VERSION;
    append_info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
    sllm_completion_t *append_completion = nullptr;
    if (!expect(sllm_kv_state_append(state, queue, &append, &append_completion,
                                     &append_info, &error.sink),
                SLLM_STATUS_OK, "public MXFP8 append", error) ||
        append_completion == nullptr ||
        !validate_append_info(append_info, length, 0U) ||
        !wait_and_release(&append_completion, "public MXFP8 append wait") ||
        !query_length(state, length)) {
      if (append_completion != nullptr) {
        (void)sllm_kv_state_append_cancel(state, append_completion,
                                          &error.sink);
        (void)sllm_completion_release(&append_completion, &error.sink);
      }
      break;
    }
    std::vector<uint16_t> observed;
    if (!execute_public_attention(context, queue, state, buffers[2], buffers[3],
                                  start, length, query_count, &observed) ||
        !compare_public_output(observed, expected, length, query_count)) {
      break;
    }
    const std::vector<uint16_t> first_observed = observed;
    // A second submission verifies that completion release leaves the shared
    // workspace reusable and that the same request output remains stable.
    if (!upload(queue, buffers[2], query.data(), query_bytes) ||
        !execute_public_attention(context, queue, state, buffers[2], buffers[3],
                                  start, length, query_count, &observed) ||
        observed != first_observed) {
      std::cerr << "public repeated attention output changed\n";
      break;
    }
    success = true;
  } while (false);
  bool cleaned = release_state(&state);
  for (auto iterator = buffers.rbegin(); iterator != buffers.rend();
       ++iterator) {
    success = release_buffer(&*iterator) && success;
  }
  std::cout << "public cleanup length=" << length << " M=" << query_count
            << " logical_bytes=" << (2U * kv_bytes + 2U * query_bytes)
            << " state_released=" << (cleaned ? 1 : 0) << '\n';
  return success && cleaned;
}

} // namespace

int main() {
  // This test deliberately opts into only the staged provider.  Other
  // candidate envs are cleared so metadata proves the intended selector.
  (void)setenv("SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_WAVE_STAGED", "1", 1);
  (void)unsetenv("SLLM_CAUSAL_ATTENTION_FORCE_BASELINE");
  (void)unsetenv("SLLM_CAUSAL_ATTENTION_GFX1030_SCALED_PREFILL_GEMM");
  (void)unsetenv("SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_GQA4_SPLIT");
  (void)unsetenv("SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P128");
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  bool success = create_visible_context(&context, &queue);
  constexpr std::array<uint64_t, 3> lengths = {1023U, 1024U, 1025U};
  if (success) {
    for (const uint64_t length : lengths) {
      for (uint32_t query_count = 1U; query_count <= 4U; ++query_count) {
        if (length < query_count ||
            !run_public_case(context, queue, length, query_count)) {
          success = false;
          break;
        }
      }
      if (!success) {
        break;
      }
    }
  }
  Error error;
  if (queue != nullptr) {
    success = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
                     "queue release", error) &&
              queue == nullptr && success;
  }
  if (context != nullptr) {
    success = expect(sllm_context_release(&context, &error.sink),
                     SLLM_STATUS_OK, "context release", error) &&
              context == nullptr && success;
  }
  if (success) {
    std::cout << "phase83 MXFP8 staged public GPU PASS target="
              << SLLM_TEST_EXPECTED_TARGET
              << " lengths=1023,1024,1025 M=1..4 repeat=1 oracle=independent"
                 " cleanup=verified\n";
  }
  return success ? 0 : 1;
}
