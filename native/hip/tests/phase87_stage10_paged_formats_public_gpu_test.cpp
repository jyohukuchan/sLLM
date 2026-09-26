// Phase 87 Stage 10: public Paged KV format coverage.
//
// This is deliberately a small public-ABI test.  It exercises append followed
// by generic paged attention at the 127/128/129 token boundaries for dynamic
// FP8 E4, static FP8 E4, NVFP4 block16, and (on gfx1030) MXFP8 E5.  A
// A host calculation checks selected output elements independently.
#include "sllm/hip.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#error SLLM_TEST_EXPECTED_TARGET must name one exact GPU target
#endif

namespace {

constexpr uint32_t kKvHeads = 2U;
constexpr uint32_t kQueryHeads = 8U;
constexpr uint32_t kHeadDim = 128U;
constexpr uint32_t kTokenBlock = 128U;
constexpr uint64_t kCapacity = 129U;
constexpr uint64_t kLogicalBlocks = 2U;
constexpr uint64_t kMaxPhysicalBlocks = 4U;
constexpr uint32_t kCompletionTimeoutMs = 30'000U;

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
  if (actual == expected)
    return true;
  std::cerr << operation << " returned " << actual << ", expected " << expected
            << ": " << error.message << '\n';
  return false;
}

bool release_buffer(sllm_buffer_t **const buffer) {
  if (buffer == nullptr || *buffer == nullptr)
    return true;
  Error error;
  const bool result = expect(sllm_buffer_release(buffer, &error.sink),
                             SLLM_STATUS_OK, "buffer release", error);
  return result && *buffer == nullptr;
}

bool release_state(sllm_kv_state_t **const state) {
  if (state == nullptr || *state == nullptr)
    return true;
  Error error;
  const bool result = expect(sllm_kv_state_release(state, &error.sink),
                             SLLM_STATUS_OK, "KV state release", error);
  return result && *state == nullptr;
}

bool release_queue(sllm_queue_t **const queue) {
  if (queue == nullptr || *queue == nullptr)
    return true;
  Error error;
  const bool result = expect(sllm_queue_release(queue, &error.sink),
                             SLLM_STATUS_OK, "queue release", error);
  return result && *queue == nullptr;
}

bool release_context(sllm_context_t **const context) {
  if (context == nullptr || *context == nullptr)
    return true;
  Error error;
  const bool result = expect(sllm_context_release(context, &error.sink),
                             SLLM_STATUS_OK, "context release", error);
  return result && *context == nullptr;
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
  const bool waited =
      expect(sllm_completion_wait(*completion, kCompletionTimeoutMs, &result,
                                  &error.sink),
             SLLM_STATUS_OK, operation, error) &&
      result.state == SLLM_COMPLETION_STATE_SUCCESS;
  if (!waited)
    std::cerr << operation << " completion state=" << result.state << '\n';
  const bool released = expect(sllm_completion_release(completion, &error.sink),
                               SLLM_STATUS_OK, "completion release", error);
  return waited && released && *completion == nullptr;
}

bool create_buffer(const sllm_context_t *const context, const uint64_t bytes,
                   sllm_buffer_t **const buffer) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = bytes;
  Error error;
  return expect(sllm_buffer_create(context, &info, buffer, &error.sink),
                SLLM_STATUS_OK, "buffer create", error) &&
         *buffer != nullptr;
}

bool upload(const sllm_queue_t *const queue, const sllm_buffer_t *const buffer,
            const void *const source, const uint64_t bytes) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = const_cast<void *>(source);
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, "buffer upload", error))
    return false;
  return wait_and_release(&completion, "buffer upload wait");
}

bool download(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer, void *const destination,
              const uint64_t bytes) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_buffer_copy_d2h(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, "buffer download", error) ||
      completion == nullptr)
    return false;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(sllm_completion_wait(completion, kCompletionTimeoutMs, &result,
                                   &error.sink),
              SLLM_STATUS_OK, "buffer download wait", error) ||
      result.state != SLLM_COMPLETION_STATE_SUCCESS) {
    (void)sllm_completion_release(&completion, &error.sink);
    return false;
  }
  uint64_t written = 0U;
  const bool read = expect(sllm_completion_read(completion, destination, bytes,
                                                &written, &error.sink),
                           SLLM_STATUS_OK, "buffer download read", error) &&
                    written == bytes;
  const bool released =
      expect(sllm_completion_release(&completion, &error.sink), SLLM_STATUS_OK,
             "download completion release", error);
  return read && released && completion == nullptr;
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

uint16_t f32_to_bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  return static_cast<uint16_t>(
      upper + ((lower > UINT32_C(0x8000) ||
                (lower == UINT32_C(0x8000) && (upper & 1U) != 0U))
                   ? 1U
                   : 0U));
}

float bf16_to_f32(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float f16_to_f32(const uint16_t raw) {
  const uint32_t sign = (static_cast<uint32_t>(raw) & 0x8000U) << 16U;
  const uint32_t exponent = (static_cast<uint32_t>(raw) >> 10U) & 0x1fU;
  const uint32_t fraction = static_cast<uint32_t>(raw) & 0x03ffU;
  uint32_t bits = 0U;
  if (exponent == 0U) {
    if (fraction == 0U) {
      bits = sign;
    } else {
      uint32_t normalized = fraction;
      uint32_t shift = 0U;
      while ((normalized & 0x0400U) == 0U) {
        normalized <<= 1U;
        ++shift;
      }
      normalized &= 0x03ffU;
      bits = sign | ((127U - 14U - shift) << 23U) | (normalized << 13U);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (fraction << 13U);
  } else {
    bits = sign | ((exponent + 112U) << 23U) | (fraction << 13U);
  }
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float e4m3_decode(const uint8_t bits) {
  const float sign = (bits & 0x80U) != 0U ? -1.0F : 1.0F;
  const uint32_t magnitude = bits & 0x7fU;
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  if (exponent == 0U)
    return sign * static_cast<float>(mantissa) * std::ldexp(1.0F, -9);
  if (magnitude == 0x7fU)
    return NAN;
  return sign * std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                           static_cast<int>(exponent) - 7);
}

uint8_t nearest_e4m3(const float value) {
  const uint8_t sign = std::signbit(value) ? UINT8_C(0x80) : 0U;
  const float magnitude = std::fabs(value);
  uint8_t selected = 0U;
  float error = magnitude;
  for (uint32_t code = 1U; code <= 0x7eU; ++code) {
    const float candidate = e4m3_decode(static_cast<uint8_t>(code));
    const float candidate_error = std::fabs(magnitude - candidate);
    if (candidate_error < error ||
        (candidate_error == error && (code & 1U) == 0U &&
         (selected & 1U) != 0U)) {
      selected = static_cast<uint8_t>(code);
      error = candidate_error;
    }
  }
  return static_cast<uint8_t>(sign | selected);
}

float e5m2_decode(const uint8_t bits) {
  return f16_to_f32(static_cast<uint16_t>(bits) << 8U);
}

uint8_t nearest_e5m2(const float value) {
  const uint8_t sign = std::signbit(value) ? UINT8_C(0x80) : 0U;
  const float magnitude = std::fabs(value);
  uint8_t selected = 0U;
  float error = magnitude;
  for (uint32_t code = 1U; code <= 0x7bU; ++code) {
    const float candidate = e5m2_decode(static_cast<uint8_t>(code));
    const float candidate_error = std::fabs(magnitude - candidate);
    if (candidate_error < error ||
        (candidate_error == error && (code & 1U) == 0U &&
         (selected & 1U) != 0U)) {
      selected = static_cast<uint8_t>(code);
      error = candidate_error;
    }
  }
  return static_cast<uint8_t>(sign | selected);
}

float e2m1_decode(const uint8_t code) {
  constexpr float magnitudes[] = {0.0F, 0.5F, 1.0F, 1.5F,
                                  2.0F, 3.0F, 4.0F, 6.0F};
  const float magnitude = magnitudes[code & 7U];
  return (code & 8U) == 0U ? magnitude : -magnitude;
}

uint8_t nearest_e2m1(const float value) {
  const uint8_t sign = std::signbit(value) ? UINT8_C(8) : 0U;
  const float magnitude = std::min(std::fabs(value), 6.0F);
  uint8_t selected = 0U;
  float error = magnitude;
  for (uint8_t code = 1U; code != 8U; ++code) {
    const float candidate_error = std::fabs(magnitude - e2m1_decode(code));
    if (candidate_error < error ||
        (candidate_error == error && (code & 1U) == 0U &&
         (selected & 1U) != 0U)) {
      selected = code;
      error = candidate_error;
    }
  }
  return static_cast<uint8_t>(sign | selected);
}

struct Format final {
  const char *name;
  uint32_t encoding;
  uint32_t dtype;
  uint32_t scale_dtype;
  uint32_t quantization_block;
  uint32_t append_kernel;
  float static_key_scale;
  float static_value_scale;
  bool dynamic_fp8;
  bool static_fp8;
  bool nvfp4;
  bool mxfp8_e5;
};

constexpr std::array<Format, 4> kFormats = {{
    {"dynamic-fp8-e4", SLLM_HIP_KV_ENCODING_FP8_V1,
     SLLM_TENSOR_DTYPE_F8_E4M3_FN, SLLM_TENSOR_DTYPE_F32, 0U,
     SLLM_HIP_KV_KERNEL_ID_BF16_TO_PAGED_FP8_E4_V1, 1.0F, 1.0F, true, false,
     false, false},
    {"static-fp8-e4", SLLM_HIP_KV_ENCODING_FP8_STATIC_V1,
     SLLM_TENSOR_DTYPE_F8_E4M3_FN, SLLM_TENSOR_DTYPE_F32, 0U,
     SLLM_HIP_KV_KERNEL_ID_BF16_TO_PAGED_FP8_STATIC_E4_V1, 1.25F, 0.75F, false,
     true, false, false},
    {"nvfp4-block16", SLLM_HIP_KV_ENCODING_NVFP4_V1, SLLM_TENSOR_DTYPE_U8,
     SLLM_TENSOR_DTYPE_F8_E4M3_FN, 16U,
     SLLM_HIP_KV_KERNEL_ID_BF16_TO_PAGED_NVFP4_V1, 1.0F, 1.0F, false, false,
     true, false},
    {"mxfp8-e5", SLLM_HIP_KV_ENCODING_MXFP8_E5_V1, SLLM_TENSOR_DTYPE_F8_E5M2,
     SLLM_TENSOR_DTYPE_U8, 32U, SLLM_HIP_KV_KERNEL_ID_BF16_TO_PAGED_MXFP8_E5_V1,
     1.0F, 1.0F, false, false, false, true},
}};

uint32_t float_bits(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::vector<uint16_t> make_kv(const bool key) {
  std::vector<uint16_t> result(static_cast<size_t>(kCapacity) * kKvHeads *
                               kHeadDim);
  for (uint64_t token = 0U; token != kCapacity; ++token) {
    for (uint32_t head = 0U; head != kKvHeads; ++head) {
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        const float base =
            key ? (0.11F + 0.017F * static_cast<float>(head) +
                   0.0019F * static_cast<float>(token % 23U) +
                   0.0021F * static_cast<float>(dimension % 29U))
                : (0.23F + 0.013F * static_cast<float>(head) +
                   0.0023F * static_cast<float>(token % 19U) +
                   0.0017F * static_cast<float>(dimension % 31U));
        const float signed_value =
            ((token + head + dimension) % (key ? 17U : 19U)) == 0U ? -base
                                                                   : base;
        result[(static_cast<size_t>(token) * kKvHeads + head) * kHeadDim +
               dimension] = f32_to_bf16(signed_value);
      }
    }
  }
  return result;
}

std::vector<uint16_t> make_queries() {
  std::vector<uint16_t> result(static_cast<size_t>(kCapacity) * kQueryHeads *
                               kHeadDim);
  for (uint64_t token = 0U; token != kCapacity; ++token) {
    for (uint32_t head = 0U; head != kQueryHeads; ++head) {
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        float value = 0.019F + 0.0017F * static_cast<float>(token) +
                      0.0029F * static_cast<float>(head % 5U) +
                      0.00041F * static_cast<float>(dimension % 27U);
        if ((token + head * 5U + dimension) % 13U == 0U)
          value = -value;
        result[(static_cast<size_t>(token) * kQueryHeads + head) * kHeadDim +
               dimension] = f32_to_bf16(value);
      }
    }
  }
  return result;
}

float quantized(const Format &format, const std::vector<uint16_t> &values,
                const uint64_t token, const uint32_t head,
                const uint32_t dimension, const bool key_row) {
  const size_t row = (static_cast<size_t>(token) * kKvHeads + head) * kHeadDim;
  const float raw = bf16_to_f32(values[row + dimension]);
  if (format.dynamic_fp8) {
    float maximum = 0.0F;
    for (uint32_t lane = 0U; lane != kHeadDim; ++lane)
      maximum = std::max(maximum, std::fabs(bf16_to_f32(values[row + lane])));
    const float scale = maximum == 0.0F ? 1.0F : maximum / 448.0F;
    return e4m3_decode(nearest_e4m3(raw / scale)) * scale;
  }
  if (format.static_fp8) {
    const float scale =
        key_row ? format.static_key_scale : format.static_value_scale;
    return e4m3_decode(nearest_e4m3(raw / scale)) * scale;
  }
  if (format.nvfp4) {
    float maximum = 0.0F;
    for (uint32_t lane = 0U; lane != kHeadDim; ++lane)
      maximum = std::max(maximum, std::fabs(bf16_to_f32(values[row + lane])));
    const float outer = maximum == 0.0F ? 1.0F : maximum / (448.0F * 6.0F);
    const uint32_t begin = (dimension / 16U) * 16U;
    const uint32_t end = std::min(begin + 16U, kHeadDim);
    float block_maximum = 0.0F;
    for (uint32_t lane = begin; lane != end; ++lane)
      block_maximum =
          std::max(block_maximum, std::fabs(bf16_to_f32(values[row + lane])));
    const float block_scale =
        e4m3_decode(nearest_e4m3((block_maximum / 6.0F) / outer));
    return e2m1_decode(nearest_e2m1(raw / (block_scale * outer))) *
           block_scale * outer;
  }
  float maximum = 0.0F;
  const uint32_t begin = (dimension / 32U) * 32U;
  const uint32_t end = std::min(begin + 32U, kHeadDim);
  for (uint32_t lane = begin; lane != end; ++lane)
    maximum = std::max(maximum, std::fabs(bf16_to_f32(values[row + lane])));
  const uint8_t scale_code =
      maximum == 0.0F
          ? UINT8_C(127)
          : static_cast<uint8_t>(
                std::clamp(std::ilogb(maximum) - 15, -127, 127) + 127);
  const float scale = std::ldexp(1.0F, static_cast<int>(scale_code) - 127);
  return e5m2_decode(nearest_e5m2(raw / scale)) * scale;
}

float oracle(const Format &format, const std::vector<uint16_t> &keys,
             const std::vector<uint16_t> &values,
             const std::vector<uint16_t> &queries, const uint64_t position,
             const uint32_t query_head, const uint32_t dimension) {
  const uint32_t kv_head = query_head / (kQueryHeads / kKvHeads);
  const size_t query_row =
      (static_cast<size_t>(position) * kQueryHeads + query_head) * kHeadDim;
  std::vector<double> scores(static_cast<size_t>(position) + 1U);
  double maximum = -std::numeric_limits<double>::infinity();
  for (uint64_t token = 0U; token <= position; ++token) {
    double dot = 0.0;
    for (uint32_t lane = 0U; lane != kHeadDim; ++lane)
      dot += static_cast<double>(bf16_to_f32(queries[query_row + lane])) *
             quantized(format, keys, token, kv_head, lane, true);
    scores[static_cast<size_t>(token)] = dot / std::sqrt(kHeadDim);
    maximum = std::max(maximum, scores[static_cast<size_t>(token)]);
  }
  double denominator = 0.0;
  double numerator = 0.0;
  for (uint64_t token = 0U; token <= position; ++token) {
    const double weight =
        std::exp(scores[static_cast<size_t>(token)] - maximum);
    denominator += weight;
    numerator +=
        weight * quantized(format, values, token, kv_head, dimension, false);
  }
  return static_cast<float>(numerator / denominator);
}

sllm_kv_append_desc_t append_descriptor(const sllm_buffer_t *const key,
                                        const sllm_buffer_t *const value,
                                        const uint64_t count,
                                        const uint64_t start) {
  const uint64_t shape[] = {count, kKvHeads, kHeadDim};
  const uint64_t token_bytes = static_cast<uint64_t>(kKvHeads) * kHeadDim * 2U;
  sllm_kv_append_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.append_version = SLLM_HIP_KV_STATE_VERSION;
  descriptor.expected_length = start;
  descriptor.start_position = start;
  descriptor.key_input = binding(key, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
  descriptor.value_input = binding(value, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
  descriptor.key_input.byte_offset = start * token_bytes;
  descriptor.value_input.byte_offset = start * token_bytes;
  return descriptor;
}

sllm_kv_state_paged_create_info_t paged_info(const Format &format,
                                             const uint64_t session) {
  sllm_kv_state_paged_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.create_info_version = SLLM_HIP_KV_PAGED_CREATE_INFO_VERSION;
  info.session_id = session;
  info.capacity_tokens = kCapacity;
  info.head_count = kKvHeads;
  info.head_dim = kHeadDim;
  info.memory_kind = SLLM_HIP_KV_MEMORY_KIND_PAGED;
  info.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
  info.dtype = format.dtype;
  info.encoding = format.encoding;
  info.scale_dtype = format.scale_dtype;
  info.quantization_block_size = format.quantization_block;
  info.token_block_size = kTokenBlock;
  info.physical_layout_version = SLLM_HIP_KV_PAGED_LAYOUT_VERSION;
  info.logical_table_capacity = kLogicalBlocks;
  info.max_physical_blocks = kMaxPhysicalBlocks;
  if (format.static_fp8) {
    info.static_key_scale_bits = float_bits(format.static_key_scale);
    info.static_value_scale_bits = float_bits(format.static_value_scale);
  }
  return info;
}

bool append_state(const Format &format, const sllm_kv_state_t *const state,
                  const sllm_queue_t *const queue,
                  const sllm_buffer_t *const key,
                  const sllm_buffer_t *const value, const uint64_t count,
                  sllm_completion_t **const completion) {
  const sllm_kv_append_desc_t descriptor =
      append_descriptor(key, value, count, 0U);
  sllm_kv_append_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
  Error error;
  *completion = nullptr;
  if (!expect(sllm_kv_state_append(state, queue, &descriptor, completion, &info,
                                   &error.sink),
              SLLM_STATUS_OK, "paged append", error) ||
      *completion == nullptr || info.backend != SLLM_BACKEND_HIP ||
      info.dispatch_count != 1U || info.token_count != count ||
      info.end_position != count || info.fallback_allowed != 0U ||
      info.fallback_used != 0U ||
      std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) != 0 ||
      info.kernel_id != format.append_kernel) {
    std::cerr << format.name << " append provider contract failed\n";
    return false;
  }
  return true;
}

bool run_attention(const Format &format, const sllm_context_t *const context,
                   const sllm_queue_t *const queue,
                   const sllm_kv_state_t *const state,
                   const sllm_buffer_t *const query,
                   const sllm_buffer_t *const output, const uint64_t position,
                   const uint32_t query_count,
                   sllm_completion_t *const dependency,
                   sllm_completion_t **const result) {
  const uint64_t shape[] = {query_count, kQueryHeads, kHeadDim};
  sllm_causal_attention_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_CAUSAL_ATTENTION_VERSION;
  descriptor.start_position = position;
  descriptor.expected_kv_length = position + query_count;
  descriptor.kv_state = state;
  descriptor.query = binding(query, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
  descriptor.output = binding(output, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
  const uint64_t row_bytes = static_cast<uint64_t>(kQueryHeads) * kHeadDim * 2U;
  descriptor.query.byte_offset = position * row_bytes;
  sllm_causal_attention_dispatch_info_t dispatch{};
  dispatch.struct_size = sizeof(dispatch);
  dispatch.abi_version = SLLM_HIP_ABI_VERSION;
  dispatch.info_version = SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION;
  Error error;
  *result = nullptr;
  const sllm_status_t status =
      dependency == nullptr
          ? sllm_causal_attention_execute(context, queue, &descriptor, result,
                                          &dispatch, &error.sink)
          : sllm_causal_attention_execute_after_kv_append(
                context, queue, dependency, &descriptor, result, &dispatch,
                &error.sink);
  if (!expect(status, SLLM_STATUS_OK, "paged attention", error) ||
      *result == nullptr || dispatch.backend != SLLM_BACKEND_HIP ||
      dispatch.dispatch_count != 1U || dispatch.query_count != query_count ||
      dispatch.start_position != position ||
      dispatch.committed_kv_length != position + query_count ||
      dispatch.q_heads != kQueryHeads || dispatch.kv_heads != kKvHeads ||
      dispatch.head_dim != kHeadDim || dispatch.fallback_allowed != 0U ||
      dispatch.fallback_used != 0U ||
      std::strcmp(dispatch.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) != 0 ||
      dispatch.kernel_id !=
          SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_GENERIC_FORMATS_V1) {
    std::cerr << format.name << " attention provider contract failed\n";
    if (*result != nullptr)
      (void)wait_and_release(result, "failed attention cleanup");
    return false;
  }
  return true;
}

bool run_case(const Format &format, const sllm_context_t *const context,
              const sllm_queue_t *const queue,
              const std::vector<uint16_t> &keys,
              const std::vector<uint16_t> &values,
              const std::vector<uint16_t> &queries, const uint64_t prefix) {
  const uint64_t input_bytes = static_cast<uint64_t>(keys.size()) * 2U;
  const uint64_t query_bytes = static_cast<uint64_t>(queries.size()) * 2U;
  const uint64_t output_bytes =
      prefix * static_cast<uint64_t>(kQueryHeads) * kHeadDim * 2U;
  sllm_kv_state_t *paged = nullptr;
  std::array<sllm_buffer_t *, 4> buffers{};
  sllm_completion_t *paged_append = nullptr;
  sllm_completion_t *paged_attention = nullptr;
  Error error;
  bool passed = true;
  const auto cleanup = [&]() {
    bool result = true;
    if (paged_attention != nullptr)
      result = wait_and_release(&paged_attention, "paged attention cleanup") &&
               result;
    result = release_state(&paged) && result;
    for (auto it = buffers.rbegin(); it != buffers.rend(); ++it)
      result = release_buffer(&*it) && result;
    return result;
  };

  const auto pinfo = paged_info(format, UINT64_C(0x8710) + prefix);
  passed &=
      expect(sllm_kv_state_create_paged(context, &pinfo, &paged, &error.sink),
             SLLM_STATUS_OK, "paged state create", error);
  if (!passed || paged == nullptr)
    return cleanup() && passed;
  passed &= create_buffer(context, input_bytes, &buffers[0]);
  passed &= create_buffer(context, input_bytes, &buffers[1]);
  passed &= create_buffer(context, query_bytes, &buffers[2]);
  passed &= create_buffer(context, output_bytes, &buffers[3]);
  if (!passed || !upload(queue, buffers[0], keys.data(), input_bytes) ||
      !upload(queue, buffers[1], values.data(), input_bytes) ||
      !upload(queue, buffers[2], queries.data(), query_bytes))
    return cleanup() && false;

  if (!append_state(format, paged, queue, buffers[0], buffers[1], prefix,
                    &paged_append))
    return cleanup() && false;
  const uint64_t position = 0U;
  if (!run_attention(format, context, queue, paged, buffers[2], buffers[3],
                     position, static_cast<uint32_t>(prefix), paged_append,
                     &paged_attention) ||
      !wait_and_release(&paged_attention, "paged attention wait") ||
      !wait_and_release(&paged_append, "paged append wait"))
    return cleanup() && false;

  std::vector<uint16_t> paged_output(output_bytes / 2U);
  if (!download(queue, buffers[3], paged_output.data(), output_bytes))
    return cleanup() && false;

  const uint64_t final_position = prefix - 1U;
  for (const uint32_t head : {0U, kQueryHeads - 1U}) {
    for (const uint32_t dimension : {0U, 64U, kHeadDim - 1U}) {
      const float expected = oracle(format, keys, values, queries,
                                    final_position, head, dimension);
      const size_t output_index =
          (static_cast<size_t>(final_position) * kQueryHeads + head) *
              kHeadDim +
          dimension;
      const float actual = bf16_to_f32(paged_output[output_index]);
      if (!std::isfinite(actual) || std::fabs(actual - expected) > 0.2F) {
        std::cerr << format.name << " oracle mismatch prefix=" << prefix
                  << " head=" << head << " dim=" << dimension
                  << " actual=" << actual << " expected=" << expected << '\n';
        passed = false;
      }
    }
  }
  sllm_kv_paged_view_info_t view{};
  view.struct_size = sizeof(view);
  view.abi_version = SLLM_HIP_ABI_VERSION;
  view.info_version = SLLM_HIP_KV_PAGED_VIEW_INFO_VERSION;
  passed &= expect(sllm_kv_state_query_paged(paged, &view, &error.sink),
                   SLLM_STATUS_OK, "paged format query", error);
  passed &= view.encoding == format.encoding &&
            view.observed_length == prefix &&
            view.logical_table_capacity == kLogicalBlocks &&
            view.allocated_physical_blocks >=
                (prefix + kTokenBlock - 1U) / kTokenBlock;
  return cleanup() && passed;
}

} // namespace

int main() {
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  Error error;
  bool passed = true;
  uint32_t device_count = 0U;
  passed &= expect(sllm_device_count(&device_count, &error.sink),
                   SLLM_STATUS_OK, "device count", error) &&
            device_count == 1U;
  sllm_device_info_t device{};
  device.struct_size = sizeof(device);
  device.abi_version = SLLM_HIP_ABI_VERSION;
  passed &= expect(sllm_device_query(0U, &device, &error.sink), SLLM_STATUS_OK,
                   "device query", error) &&
            device.wavefront_size == 32U &&
            std::strcmp(device.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::snprintf(context_info.expected_gcn_arch_name,
                sizeof(context_info.expected_gcn_arch_name), "%s",
                SLLM_TEST_EXPECTED_TARGET);
  passed &= expect(sllm_context_create(&context_info, &context, &error.sink),
                   SLLM_STATUS_OK, "context create", error) &&
            context != nullptr;
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  passed &= context != nullptr &&
            expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
                   SLLM_STATUS_OK, "queue create", error) &&
            queue != nullptr;
  const std::vector<uint16_t> keys = make_kv(true);
  const std::vector<uint16_t> values = make_kv(false);
  const std::vector<uint16_t> queries = make_queries();
  uint32_t format_cases = 0U;
  for (const Format &format : kFormats) {
    if (format.mxfp8_e5 &&
        std::strncmp(device.gcn_arch_name, "gfx1030", 7U) != 0)
      continue;
    for (const uint64_t prefix :
         {UINT64_C(127), UINT64_C(128), UINT64_C(129)}) {
      if (!passed ||
          !run_case(format, context, queue, keys, values, queries, prefix)) {
        std::cerr << "public paged format case failed format=" << format.name
                  << " prefix=" << prefix << '\n';
        passed = false;
      }
      ++format_cases;
    }
  }
  passed = release_queue(&queue) && passed;
  passed = release_context(&context) && passed;
  if (passed) {
    std::cout << "phase87_stage10_paged_formats_public_gpu_test: PASS target="
              << SLLM_TEST_EXPECTED_TARGET << " cases=" << format_cases
              << " boundaries=127,128,129 q_heads=8 kv_heads=2 head_dim=128"
                 " numerical-oracle=1 fallback=0 cleanup=0\n";
  }
  return passed ? 0 : 1;
}
