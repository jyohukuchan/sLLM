#include "sllm/hip.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1201"
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
constexpr uint64_t kPrefixAppend = 31U;
constexpr uint64_t kFirstChainStart = 31U;
constexpr uint64_t kSecondChainStart = 32U;
constexpr uint64_t kFinalLength = 33U;
constexpr uint64_t kCapacity = 65U;
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

std::vector<uint16_t> oracle(const std::vector<uint16_t> &key,
                             const std::vector<uint16_t> &value,
                             const std::vector<uint16_t> &query,
                             const uint64_t length) {
  std::array<std::vector<QuantizedRow>, kKvHeads> rows{};
  for (uint32_t head = 0U; head != kKvHeads; ++head) {
    rows[head].reserve(static_cast<std::size_t>(length));
    for (uint64_t token = 0U; token != length; ++token) {
      rows[head].push_back(quantize_row(key, token, head, kHeadDim));
    }
  }
  std::vector<uint16_t> result(static_cast<std::size_t>(kQueryElements), 0U);
  for (uint32_t query_head = 0U; query_head != kQueryHeads; ++query_head) {
    const uint32_t kv_head = query_head / (kQueryHeads / kKvHeads);
    std::vector<float> scores(static_cast<std::size_t>(length));
    float maximum = -std::numeric_limits<float>::infinity();
    for (uint64_t token = 0U; token != length; ++token) {
      float score = 0.0F;
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        score += bf16_to_f32(query[query_head * kHeadDim + dimension]) *
                 decoded_value(rows[kv_head][token], dimension);
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
      for (uint64_t token = 0U; token != length; ++token) {
        const QuantizedRow value_row =
            quantize_row(value, token, kv_head, kHeadDim);
        accumulated += scores[static_cast<std::size_t>(token)] / denominator *
                       decoded_value(value_row, dimension);
      }
      result[static_cast<std::size_t>(query_head * kHeadDim + dimension)] =
          f32_to_bf16(accumulated);
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

bool compare_output(const std::vector<uint16_t> &observed,
                    const std::vector<uint16_t> &expected,
                    const char *const tag) {
  if (observed.size() != expected.size()) {
    return false;
  }
  float max_abs = 0.0F;
  float max_relative = 0.0F;
  uint32_t max_ulp = 0U;
  for (std::size_t index = 0U; index != observed.size(); ++index) {
    const float actual = bf16_to_f32(observed[index]);
    const float wanted = bf16_to_f32(expected[index]);
    const float difference = std::fabs(actual - wanted);
    max_abs = std::max(max_abs, difference);
    if (std::fabs(wanted) >= 0.125F) {
      max_relative = std::max(max_relative, difference / std::fabs(wanted));
    }
    max_ulp =
        std::max(max_ulp, bf16_ulp_distance(observed[index], expected[index]));
    if (!std::isfinite(actual) || !std::isfinite(wanted)) {
      std::cerr << tag << " oracle nonfinite index=" << index
                << " actual=" << actual << " expected=" << wanted << '\n';
      return false;
    }
  }
  std::cout << tag << " oracle max_abs=" << max_abs
            << " max_rel=" << max_relative << " max_ulp=" << max_ulp << '\n';
  // The oracle rounds the same BF16 output format as the native kernel.  A
  // few BF16 ULPs cover only FP32 reduction/exp ordering, while the absolute
  // and relative limits keep a large numerical drift from hiding behind a
  // single low-magnitude ULP.
  if (max_ulp > 4U || max_abs > 0.03125F || max_relative > 0.04F) {
    std::cerr << tag << " oracle tolerance exceeded max_abs=" << max_abs
              << " max_rel=" << max_relative << " max_ulp=" << max_ulp << '\n';
    return false;
  }
  return true;
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

bool execute_chained_attention(const sllm_context_t *const context,
                               const sllm_queue_t *const queue,
                               const sllm_kv_state_t *const state,
                               sllm_completion_t *const append,
                               const sllm_buffer_t *const query,
                               const sllm_buffer_t *const output,
                               const uint64_t start, const uint64_t length,
                               sllm_completion_t **const append_owner,
                               std::vector<uint16_t> *const observed) {
  const uint64_t shape[] = {1U, kQueryHeads, kHeadDim};
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
  sllm_completion_t *attention = nullptr;
  Error error;
  if (!expect(sllm_causal_attention_execute_after_kv_append(
                  context, queue, append, &descriptor, &attention, &info,
                  &error.sink),
              SLLM_STATUS_OK, "chained MXFP8 attention", error) ||
      attention == nullptr || info.dispatch_id == 0U ||
      info.dispatch_count != 1U ||
      info.kernel_id != SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PACKED_KV_V3 ||
      info.query_count != 1U || info.start_position != start ||
      info.committed_kv_length != length || info.q_heads != kQueryHeads ||
      info.kv_heads != kKvHeads || info.head_dim != kHeadDim ||
      info.fallback_allowed != 0U || info.fallback_used != 0U ||
      std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) != 0) {
    if (attention != nullptr) {
      (void)sllm_completion_release(&attention, &error.sink);
    }
    return false;
  }
  if (!wait_and_release(&attention, "chained attention wait") ||
      !wait_and_release(append_owner, "chained append wait")) {
    return false;
  }
  observed->assign(static_cast<std::size_t>(kQueryElements), 0U);
  return download(queue, output, observed->data(),
                  kQueryElements * sizeof(uint16_t));
}

bool execute_ordinary_attention(const sllm_context_t *const context,
                                const sllm_queue_t *const queue,
                                const sllm_kv_state_t *const state,
                                const sllm_buffer_t *const query,
                                const sllm_buffer_t *const output,
                                const uint64_t start, const uint64_t length,
                                std::vector<uint16_t> *const observed) {
  const uint64_t shape[] = {1U, kQueryHeads, kHeadDim};
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
              SLLM_STATUS_OK, "ordinary MXFP8 attention", error) ||
      completion == nullptr || info.dispatch_id == 0U ||
      info.dispatch_count != 1U ||
      info.kernel_id != SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PACKED_KV_V3 ||
      info.query_count != 1U || info.start_position != start ||
      info.committed_kv_length != length || info.q_heads != kQueryHeads ||
      info.kv_heads != kKvHeads || info.head_dim != kHeadDim ||
      info.fallback_allowed != 0U || info.fallback_used != 0U ||
      std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) != 0 ||
      !wait_and_release(&completion, "ordinary attention wait")) {
    if (completion != nullptr) {
      (void)sllm_completion_release(&completion, &error.sink);
    }
    return false;
  }
  observed->assign(static_cast<std::size_t>(kQueryElements), 0U);
  return download(queue, output, observed->data(),
                  kQueryElements * sizeof(uint16_t));
}

bool run_case(const sllm_context_t *const context,
              const sllm_queue_t *const queue) {
  constexpr uint64_t input_tokens = 50U;
  const uint64_t kv_bytes = input_tokens * kElementsPerToken * sizeof(uint16_t);
  const uint64_t query_bytes = kQueryElements * sizeof(uint16_t);
  sllm_kv_state_t *state = nullptr;
  std::array<sllm_buffer_t *, 4> buffers{};
  Error error;
  bool success = false;

  const std::vector<uint16_t> key_initial = make_kv(false, false, input_tokens);
  const std::vector<uint16_t> value_initial =
      make_kv(true, false, input_tokens);
  const std::vector<uint16_t> key_mutated = make_kv(false, true, input_tokens);
  const std::vector<uint16_t> value_mutated = make_kv(true, true, input_tokens);
  std::vector<uint16_t> key_second = key_initial;
  std::vector<uint16_t> value_second = value_initial;
  const std::size_t second_offset =
      static_cast<std::size_t>(kFirstChainStart * kElementsPerToken);
  const std::size_t second_elements =
      static_cast<std::size_t>(kElementsPerToken);
  std::copy_n(key_mutated.begin() + static_cast<std::ptrdiff_t>(second_offset),
              second_elements,
              key_second.begin() + static_cast<std::ptrdiff_t>(second_offset));
  std::copy_n(
      value_mutated.begin() + static_cast<std::ptrdiff_t>(second_offset),
      second_elements,
      value_second.begin() + static_cast<std::ptrdiff_t>(second_offset));
  std::vector<uint16_t> key_third = key_second;
  std::vector<uint16_t> value_third = value_second;
  const std::size_t third_offset =
      static_cast<std::size_t>(kSecondChainStart * kElementsPerToken);
  std::copy_n(key_mutated.begin() + static_cast<std::ptrdiff_t>(third_offset),
              second_elements,
              key_third.begin() + static_cast<std::ptrdiff_t>(third_offset));
  std::copy_n(value_mutated.begin() + static_cast<std::ptrdiff_t>(third_offset),
              second_elements,
              value_third.begin() + static_cast<std::ptrdiff_t>(third_offset));
  const std::vector<uint16_t> query_initial = make_query(false);
  const std::vector<uint16_t> query_mutated = make_query(true);
  std::vector<uint16_t> first_observed;
  std::vector<uint16_t> second_observed;
  do {
    sllm_kv_state_create_info_v2_t create{};
    create.struct_size = sizeof(create);
    create.abi_version = SLLM_HIP_ABI_VERSION;
    create.create_info_version = SLLM_HIP_KV_STATE_CREATE_INFO_V2_VERSION;
    create.session_id = UINT64_C(0x8301);
    create.layer_id = 83U;
    create.capacity_tokens = kCapacity;
    create.head_count = kKvHeads;
    create.head_dim = kHeadDim;
    create.memory_kind = SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS;
    create.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
    create.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
    create.encoding = SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;
    create.block_size = 32U;
    create.scale_dtype = SLLM_TENSOR_DTYPE_U8;
    if (!expect(sllm_kv_state_create_v2(context, &create, &state, &error.sink),
                SLLM_STATUS_OK, "MXFP8 state create", error) ||
        state == nullptr || !create_buffer(context, kv_bytes, &buffers[0]) ||
        !create_buffer(context, kv_bytes, &buffers[1]) ||
        !create_buffer(context, query_bytes, &buffers[2]) ||
        !create_buffer(context, query_bytes, &buffers[3]) ||
        !upload(queue, buffers[0], key_initial.data(), kv_bytes) ||
        !upload(queue, buffers[1], value_initial.data(), kv_bytes) ||
        !upload(queue, buffers[2], query_initial.data(), query_bytes)) {
      break;
    }

    sllm_kv_append_desc_t first =
        append_descriptor(buffers[0], buffers[1], kPrefixAppend, 0U);
    sllm_kv_append_info_t first_info{};
    first_info.struct_size = sizeof(first_info);
    first_info.abi_version = SLLM_HIP_ABI_VERSION;
    first_info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
    sllm_completion_t *first_completion = nullptr;
    if (!expect(sllm_kv_state_append(state, queue, &first, &first_completion,
                                     &first_info, &error.sink),
                SLLM_STATUS_OK, "first MXFP8 append", error) ||
        first_completion == nullptr ||
        !validate_append_info(first_info, kPrefixAppend, 0U) ||
        !wait_and_release(&first_completion, "prefix append wait") ||
        !query_length(state, kPrefixAppend) ||
        !execute_ordinary_attention(context, queue, state, buffers[2],
                                    buffers[3], kPrefixAppend - 1U,
                                    kPrefixAppend, &first_observed)) {
      if (first_completion != nullptr) {
        (void)sllm_kv_state_append_cancel(state, first_completion, &error.sink);
        (void)sllm_completion_release(&first_completion, &error.sink);
      }
      break;
    }
    if (!compare_output(
            first_observed,
            oracle(key_initial, value_initial, query_initial, kPrefixAppend),
            "prefix ordinary MXFP8")) {
      break;
    }
    if (!upload(queue, buffers[0], key_mutated.data(), kv_bytes) ||
        !upload(queue, buffers[1], value_mutated.data(), kv_bytes) ||
        !upload(queue, buffers[2], query_mutated.data(), query_bytes)) {
      break;
    }

    sllm_kv_append_desc_t second =
        append_descriptor(buffers[0], buffers[1], 1U, kFirstChainStart);
    sllm_kv_append_info_t second_info{};
    second_info.struct_size = sizeof(second_info);
    second_info.abi_version = SLLM_HIP_ABI_VERSION;
    second_info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
    sllm_completion_t *second_completion = nullptr;
    if (!expect(sllm_kv_state_append(state, queue, &second, &second_completion,
                                     &second_info, &error.sink),
                SLLM_STATUS_OK, "second MXFP8 append", error) ||
        second_completion == nullptr ||
        !validate_append_info(second_info, 1U, kFirstChainStart) ||
        !execute_chained_attention(context, queue, state, second_completion,
                                   buffers[2], buffers[3], kFirstChainStart,
                                   kSecondChainStart, &second_completion,
                                   &second_observed)) {
      if (second_completion != nullptr) {
        (void)sllm_kv_state_append_cancel(state, second_completion,
                                          &error.sink);
        (void)sllm_completion_release(&second_completion, &error.sink);
      }
      break;
    }
    if (!query_length(state, kSecondChainStart) ||
        !compare_output(
            second_observed,
            oracle(key_second, value_second, query_mutated, kSecondChainStart),
            "second chained MXFP8")) {
      break;
    }
    bool changed = false;
    for (std::size_t index = 0U; index != first_observed.size(); ++index) {
      changed = changed || first_observed[index] != second_observed[index];
    }
    if (!changed) {
      std::cerr << "input mutation did not change the chained output\n";
      break;
    }

    std::vector<uint16_t> third_observed;
    sllm_kv_append_desc_t third =
        append_descriptor(buffers[0], buffers[1], 1U, kSecondChainStart);
    sllm_kv_append_info_t third_info{};
    third_info.struct_size = sizeof(third_info);
    third_info.abi_version = SLLM_HIP_ABI_VERSION;
    third_info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
    sllm_completion_t *third_completion = nullptr;
    if (!expect(sllm_kv_state_append(state, queue, &third, &third_completion,
                                     &third_info, &error.sink),
                SLLM_STATUS_OK, "third MXFP8 append", error) ||
        third_completion == nullptr ||
        !validate_append_info(third_info, 1U, kSecondChainStart) ||
        !execute_chained_attention(context, queue, state, third_completion,
                                   buffers[2], buffers[3], kSecondChainStart,
                                   kFinalLength, &third_completion,
                                   &third_observed)) {
      if (third_completion != nullptr) {
        (void)sllm_kv_state_append_cancel(state, third_completion, &error.sink);
        (void)sllm_completion_release(&third_completion, &error.sink);
      }
      break;
    }
    if (!query_length(state, kFinalLength) ||
        !compare_output(
            third_observed,
            oracle(key_third, value_third, query_mutated, kFinalLength),
            "third chained MXFP8")) {
      break;
    }
    const uint64_t query_shape[] = {1U, kQueryHeads, kHeadDim};
    sllm_causal_attention_desc_t repeat{};
    repeat.struct_size = sizeof(repeat);
    repeat.abi_version = SLLM_HIP_ABI_VERSION;
    repeat.op_version = SLLM_HIP_CAUSAL_ATTENTION_VERSION;
    repeat.start_position = kSecondChainStart;
    repeat.expected_kv_length = kFinalLength;
    repeat.kv_state = state;
    repeat.query = binding(buffers[2], SLLM_TENSOR_DTYPE_BF16, 3U, query_shape);
    repeat.output =
        binding(buffers[3], SLLM_TENSOR_DTYPE_BF16, 3U, query_shape);
    sllm_causal_attention_dispatch_info_t repeat_info{};
    repeat_info.struct_size = sizeof(repeat_info);
    repeat_info.abi_version = SLLM_HIP_ABI_VERSION;
    repeat_info.info_version = SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION;
    sllm_completion_t *repeat_completion = nullptr;
    if (!expect(sllm_causal_attention_execute(context, queue, &repeat,
                                              &repeat_completion, &repeat_info,
                                              &error.sink),
                SLLM_STATUS_OK, "repeated MXFP8 attention", error) ||
        repeat_completion == nullptr || repeat_info.dispatch_count != 1U ||
        repeat_info.kernel_id !=
            SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PACKED_KV_V3 ||
        repeat_info.fallback_allowed != 0U || repeat_info.fallback_used != 0U ||
        !wait_and_release(&repeat_completion, "repeated attention wait")) {
      if (repeat_completion != nullptr) {
        (void)sllm_completion_release(&repeat_completion, &error.sink);
      }
      break;
    }
    std::vector<uint16_t> repeated(kQueryElements);
    if (!download(queue, buffers[3], repeated.data(), query_bytes) ||
        repeated != third_observed) {
      std::cerr << "repeated ordinary attention differs from chained output\n";
      break;
    }

    sllm_kv_append_desc_t canceled =
        append_descriptor(buffers[0], buffers[1], 16U, kFinalLength);
    sllm_kv_append_info_t canceled_info{};
    canceled_info.struct_size = sizeof(canceled_info);
    canceled_info.abi_version = SLLM_HIP_ABI_VERSION;
    canceled_info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
    sllm_completion_t *canceled_completion = nullptr;
    if (!expect(sllm_kv_state_append(state, queue, &canceled,
                                     &canceled_completion, &canceled_info,
                                     &error.sink),
                SLLM_STATUS_OK, "canceled MXFP8 append", error) ||
        canceled_completion == nullptr ||
        !expect(sllm_kv_state_append_cancel(state, canceled_completion,
                                            &error.sink),
                SLLM_STATUS_OK, "cancel MXFP8 append", error) ||
        !wait_and_release(&canceled_completion, "canceled append wait") ||
        !query_length(state, kFinalLength)) {
      if (canceled_completion != nullptr) {
        (void)sllm_completion_release(&canceled_completion, &error.sink);
      }
      break;
    }
    success = true;
  } while (false);
  const bool cleaned = release_state(&state);
  for (auto iterator = buffers.rbegin(); iterator != buffers.rend();
       ++iterator) {
    success = release_buffer(&*iterator) && success;
  }
  return success && cleaned;
}

} // namespace

int main() {
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  bool success = create_visible_context(&context, &queue);
  if (success) {
    success = run_case(context, queue);
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
    std::cout << "phase83 MXFP8 append-chain GPU PASS target="
              << SLLM_TEST_EXPECTED_TARGET
              << " append_lengths=31,32,33 chained_appends=2 repeat=1 cancel=1"
                 " oracle=independent"
                 " fallback=false cleanup_failures=0\n";
  }
  return success ? 0 : 1;
}
