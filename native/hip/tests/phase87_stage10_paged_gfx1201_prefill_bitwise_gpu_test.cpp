// Phase 87 Stage 10: gfx1201 Paged versus contiguous MXFP8-E4 prefill.
//
// This is intentionally a small public-ABI control test.  It records the
// selected providers for M=47 and M=64, compares every BF16 output bitwise,
// and checks representative values against an independent host oracle.
#include "sllm/hip.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#error SLLM_TEST_EXPECTED_TARGET must name one exact GPU target
#endif
#ifndef SLLM_TEST_EXPECTED_UUID
#error SLLM_TEST_EXPECTED_UUID must name one exact GPU UUID
#endif

namespace {

constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kQueryHeads = 24U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kQuantizationBlock = 32U;
constexpr uint32_t kTokenBlock = 128U;
constexpr uint64_t kCapacity = 128U;
constexpr uint64_t kLogicalBlocks = 1U;
constexpr uint64_t kMaxPhysicalBlocks = 6U;
constexpr uint32_t kTimeoutMs = 30'000U;

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
  std::cerr << operation << " status=" << actual << " expected=" << expected
            << " message=" << error.message << '\n';
  return false;
}

std::string uuid_text(const hipUUID &uuid) {
  std::string value(uuid.bytes, sizeof(uuid.bytes));
  for (char &character : value) {
    const auto byte = static_cast<unsigned char>(character);
    if (!std::isxdigit(byte))
      return {};
    character = static_cast<char>(std::tolower(byte));
  }
  return "GPU-" + value;
}

bool release_buffer(sllm_buffer_t **const buffer) {
  if (buffer == nullptr || *buffer == nullptr)
    return true;
  Error error;
  return expect(sllm_buffer_release(buffer, &error.sink), SLLM_STATUS_OK,
                "buffer release", error) &&
         *buffer == nullptr;
}

bool release_state(sllm_kv_state_t **const state) {
  if (state == nullptr || *state == nullptr)
    return true;
  Error error;
  return expect(sllm_kv_state_release(state, &error.sink), SLLM_STATUS_OK,
                "state release", error) &&
         *state == nullptr;
}

bool release_queue(sllm_queue_t **const queue) {
  if (queue == nullptr || *queue == nullptr)
    return true;
  Error error;
  return expect(sllm_queue_release(queue, &error.sink), SLLM_STATUS_OK,
                "queue release", error) &&
         *queue == nullptr;
}

bool release_context(sllm_context_t **const context) {
  if (context == nullptr || *context == nullptr)
    return true;
  Error error;
  return expect(sllm_context_release(context, &error.sink), SLLM_STATUS_OK,
                "context release", error) &&
         *context == nullptr;
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

bool wait_release(sllm_completion_t **const completion,
                  const char *const operation) {
  if (completion == nullptr || *completion == nullptr)
    return false;
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
            const void *const source, const uint64_t bytes,
            const char *const operation) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = const_cast<void *>(source);
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  return expect(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                     &error.sink),
                SLLM_STATUS_OK, operation, error) &&
         wait_release(&completion, operation);
}

bool download(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer, void *const destination,
              const uint64_t bytes, const char *const operation) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_buffer_copy_d2h(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, operation, error) ||
      completion == nullptr)
    return false;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(
          sllm_completion_wait(completion, kTimeoutMs, &result, &error.sink),
          SLLM_STATUS_OK, operation, error) ||
      result.state != SLLM_COMPLETION_STATE_SUCCESS) {
    (void)sllm_completion_release(&completion, &error.sink);
    return false;
  }
  uint64_t bytes_written = 0U;
  const bool read = expect(sllm_completion_read(completion, destination, bytes,
                                                &bytes_written, &error.sink),
                           SLLM_STATUS_OK, "download read", error) &&
                    bytes_written == bytes;
  const bool released =
      expect(sllm_completion_release(&completion, &error.sink), SLLM_STATUS_OK,
             "download release", error) &&
      completion == nullptr;
  return read && released;
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
  for (uint32_t index = rank; index != 0U; --index) {
    const uint32_t current = index - 1U;
    result.shape[current] = shape[current];
    result.stride_elements[current] = stride;
    stride *= shape[current];
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

uint8_t e4m3fn_encode(const float value) {
  const uint8_t sign = std::signbit(value) ? UINT8_C(0x80) : 0U;
  const float magnitude = std::fabs(value);
  if (magnitude == 0.0F)
    return sign;
  if (!std::isfinite(magnitude) || magnitude >= 448.0F)
    return static_cast<uint8_t>(sign | UINT8_C(0x7e));
  uint32_t bits = 0U;
  std::memcpy(&bits, &magnitude, sizeof(bits));
  const uint32_t rounded =
      bits + UINT32_C(0x0007ffff) + ((bits >> 20U) & UINT32_C(1));
  const uint32_t exponent = ((rounded >> 23U) & UINT32_C(0xff)) - 120U;
  const uint32_t code = (exponent << 3U) | ((rounded >> 20U) & UINT32_C(7));
  return static_cast<uint8_t>(sign |
                              static_cast<uint8_t>(std::min(code, 0x7eU)));
}

double e4m3fn_decode(const uint8_t value) {
  const double sign = (value & UINT8_C(0x80)) != 0U ? -1.0 : 1.0;
  const uint32_t magnitude = value & UINT8_C(0x7f);
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & UINT32_C(7);
  if (exponent == 0U)
    return sign * static_cast<double>(mantissa) * std::ldexp(1.0, -9);
  return sign * std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0,
                           static_cast<int>(exponent) - 7);
}

uint8_t mxfp8_scale(const std::vector<uint16_t> &values, const std::size_t row,
                    const uint32_t dimension) {
  const std::size_t begin =
      row + static_cast<std::size_t>(dimension / kQuantizationBlock) *
                kQuantizationBlock;
  float maximum = 0.0F;
  for (uint32_t lane = 0U; lane != kQuantizationBlock; ++lane)
    maximum = std::max(maximum, std::fabs(bf16_to_f32(values[begin + lane])));
  if (maximum == 0.0F)
    return UINT8_C(127);
  return static_cast<uint8_t>(std::clamp(std::ilogb(maximum) - 8, -127, 127) +
                              127);
}

double quantized_value(const std::vector<uint16_t> &values,
                       const uint64_t token, const uint32_t head,
                       const uint32_t dimension) {
  const std::size_t row =
      (static_cast<std::size_t>(token) * kKvHeads + head) * kHeadDim;
  const float scale = std::ldexp(
      1.0F, static_cast<int>(mxfp8_scale(values, row, dimension)) - 127);
  return e4m3fn_decode(
             e4m3fn_encode(bf16_to_f32(values[row + dimension]) / scale)) *
         scale;
}

double oracle_value(const std::vector<uint16_t> &keys,
                    const std::vector<uint16_t> &values,
                    const std::vector<uint16_t> &queries, const uint32_t length,
                    const uint32_t query_row, const uint32_t query_head,
                    const uint32_t output_dimension) {
  const uint32_t kv_head = query_head / (kQueryHeads / kKvHeads);
  std::vector<double> scores(length);
  double maximum = -std::numeric_limits<double>::infinity();
  for (uint32_t token = 0U; token != length; ++token) {
    double score = 0.0;
    for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
      const std::size_t query_index =
          (static_cast<std::size_t>(query_row) * kQueryHeads + query_head) *
              kHeadDim +
          dimension;
      score += static_cast<double>(bf16_to_f32(queries[query_index])) *
               quantized_value(keys, token, kv_head, dimension);
    }
    scores[token] = score / 16.0;
    maximum = std::max(maximum, scores[token]);
  }
  double denominator = 0.0;
  double numerator = 0.0;
  for (uint32_t token = 0U; token != length; ++token) {
    const double weight = std::exp(scores[token] - maximum);
    denominator += weight;
    numerator +=
        weight * quantized_value(values, token, kv_head, output_dimension);
  }
  return numerator / denominator;
}

std::vector<uint16_t> make_kv(const bool key) {
  std::vector<uint16_t> result(static_cast<std::size_t>(kCapacity) * kKvHeads *
                               kHeadDim);
  for (uint64_t token = 0U; token != kCapacity; ++token) {
    for (uint32_t head = 0U; head != kKvHeads; ++head) {
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        const float base =
            key ? (0.19F + 0.013F * static_cast<float>(head) +
                   0.0007F * static_cast<float>(token % 17U) +
                   0.0011F * static_cast<float>(dimension % 19U))
                : (0.31F + 0.021F * static_cast<float>(head) +
                   0.0009F * static_cast<float>(token % 13U) +
                   0.0013F * static_cast<float>(dimension % 17U));
        const float signed_value =
            ((token + head + dimension) % (key ? 13U : 17U)) == 0U ? -base
                                                                   : base;
        result[(static_cast<std::size_t>(token) * kKvHeads + head) * kHeadDim +
               dimension] = f32_to_bf16(signed_value);
      }
    }
  }
  return result;
}

std::vector<uint16_t> make_queries() {
  std::vector<uint16_t> result(static_cast<std::size_t>(kCapacity) *
                               kQueryHeads * kHeadDim);
  for (uint64_t row = 0U; row != kCapacity; ++row) {
    for (uint32_t head = 0U; head != kQueryHeads; ++head) {
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        float value = 0.026F + 0.0021F * static_cast<float>(row) +
                      0.0017F * static_cast<float>(head % 9U) +
                      0.00031F * static_cast<float>(dimension % 23U);
        if ((row + head * 3U + dimension) % 11U == 0U)
          value = -value;
        result[(static_cast<std::size_t>(row) * kQueryHeads + head) * kHeadDim +
               dimension] = f32_to_bf16(value);
      }
    }
  }
  return result;
}

uint32_t next_bits(uint32_t *const state) {
  uint32_t value = *state;
  value ^= value << 13U;
  value ^= value >> 17U;
  value ^= value << 5U;
  *state = value == 0U ? UINT32_C(0x9e3779b9) : value;
  return *state;
}

void make_stress_vectors(const uint32_t seed, std::vector<uint16_t> *const keys,
                         std::vector<uint16_t> *const values,
                         std::vector<uint16_t> *const queries) {
  uint32_t state = seed | 1U;
  keys->resize(static_cast<std::size_t>(kCapacity) * kKvHeads * kHeadDim);
  values->resize(keys->size());
  queries->resize(static_cast<std::size_t>(kCapacity) * kQueryHeads * kHeadDim);
  for (uint64_t token = 0U; token != kCapacity; ++token) {
    for (uint32_t head = 0U; head != kKvHeads; ++head) {
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        const uint32_t random = next_bits(&state);
        const int exponent = static_cast<int>((random >> 3U) % 15U) - 7;
        const float mantissa =
            0.996F + static_cast<float>((random >> 12U) & 0x1fU) / 4096.0F;
        const float magnitude = std::ldexp(mantissa, exponent);
        const float signed_value = (random & 1U) != 0U ? -magnitude : magnitude;
        const std::size_t index =
            (static_cast<std::size_t>(token) * kKvHeads + head) * kHeadDim +
            dimension;
        (*keys)[index] = f32_to_bf16(signed_value);
        const float value = ((random >> 1U) & 1U) != 0U ? -magnitude * 0.9375F
                                                        : magnitude * 0.9375F;
        (*values)[index] = f32_to_bf16(value);
      }
    }
  }
  for (uint64_t token = 0U; token != kCapacity; ++token) {
    for (uint32_t head = 0U; head != kQueryHeads; ++head) {
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        const uint32_t random = next_bits(&state);
        const float tie =
            0.125F + static_cast<float>((random >> 8U) & 0x0fU) / 4096.0F;
        const float signed_value =
            (head + dimension + token + seed) % 2U == 0U ? tie : -tie;
        (*queries)[(static_cast<std::size_t>(token) * kQueryHeads + head) *
                       kHeadDim +
                   dimension] = f32_to_bf16(signed_value);
      }
    }
  }
}

sllm_kv_state_paged_create_info_t paged_create(const uint64_t session,
                                               const uint32_t layer) {
  sllm_kv_state_paged_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.create_info_version = SLLM_HIP_KV_PAGED_CREATE_INFO_VERSION;
  info.session_id = session;
  info.layer_id = layer;
  info.capacity_tokens = kCapacity;
  info.head_count = kKvHeads;
  info.head_dim = kHeadDim;
  info.memory_kind = SLLM_HIP_KV_MEMORY_KIND_PAGED;
  info.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
  info.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
  info.encoding = SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;
  info.scale_dtype = SLLM_TENSOR_DTYPE_U8;
  info.quantization_block_size = kQuantizationBlock;
  info.token_block_size = kTokenBlock;
  info.physical_layout_version = SLLM_HIP_KV_PAGED_LAYOUT_VERSION;
  info.logical_table_capacity = kLogicalBlocks;
  info.max_physical_blocks = kMaxPhysicalBlocks;
  return info;
}

sllm_kv_state_create_info_v2_t contiguous_create(const uint64_t session,
                                                 const uint32_t layer) {
  sllm_kv_state_create_info_v2_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.create_info_version = SLLM_HIP_KV_STATE_CREATE_INFO_V2_VERSION;
  info.session_id = session;
  info.layer_id = layer;
  info.capacity_tokens = kCapacity;
  info.head_count = kKvHeads;
  info.head_dim = kHeadDim;
  info.memory_kind = SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS;
  info.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
  info.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
  info.encoding = SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;
  info.block_size = kQuantizationBlock;
  info.scale_dtype = SLLM_TENSOR_DTYPE_U8;
  return info;
}

bool run_case(const sllm_context_t *const context,
              const sllm_queue_t *const queue, const uint32_t count,
              const std::vector<uint16_t> &keys,
              const std::vector<uint16_t> &values,
              const std::vector<uint16_t> &queries) {
  sllm_kv_state_t *paged = nullptr;
  sllm_kv_state_t *contiguous = nullptr;
  std::array<sllm_buffer_t *, 5U> buffers{};
  bool passed = true;
  Error error;
  const uint64_t kv_bytes =
      static_cast<uint64_t>(kCapacity) * kKvHeads * kHeadDim * sizeof(uint16_t);
  const uint64_t query_bytes = static_cast<uint64_t>(kCapacity) * kQueryHeads *
                               kHeadDim * sizeof(uint16_t);
  const uint64_t output_bytes =
      static_cast<uint64_t>(count) * kQueryHeads * kHeadDim * sizeof(uint16_t);
  const auto cleanup = [&]() {
    bool result = true;
    result = release_state(&contiguous) && result;
    result = release_state(&paged) && result;
    for (sllm_buffer_t *&buffer : buffers)
      result = release_buffer(&buffer) && result;
    return result;
  };
  const auto paged_info = paged_create(UINT64_C(0x873000) + count, count);
  const auto contiguous_info = contiguous_create(paged_info.session_id, count);
  passed &= expect(
      sllm_kv_state_create_paged(context, &paged_info, &paged, &error.sink),
      SLLM_STATUS_OK, "paged prefill create", error);
  passed &= expect(sllm_kv_state_create_v2(context, &contiguous_info,
                                           &contiguous, &error.sink),
                   SLLM_STATUS_OK, "contiguous prefill create", error);
  passed &= create_buffer(context, kv_bytes, &buffers[0]);
  passed &= create_buffer(context, kv_bytes, &buffers[1]);
  passed &= create_buffer(context, query_bytes, &buffers[2]);
  passed &= create_buffer(context, output_bytes, &buffers[3]);
  passed &= create_buffer(context, output_bytes, &buffers[4]);
  if (passed) {
    passed &= upload(queue, buffers[0], keys.data(), kv_bytes, "key upload");
    passed &=
        upload(queue, buffers[1], values.data(), kv_bytes, "value upload");
    passed &=
        upload(queue, buffers[2], queries.data(), query_bytes, "query upload");
  }
  auto append_state = [&](const sllm_kv_state_t *const state,
                          sllm_completion_t **const completion,
                          const char *const operation) {
    const uint64_t shape[] = {count, kKvHeads, kHeadDim};
    sllm_kv_append_desc_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.append_version = SLLM_HIP_KV_STATE_VERSION;
    descriptor.expected_length = 0U;
    descriptor.start_position = 0U;
    descriptor.key_input =
        binding(buffers[0], SLLM_TENSOR_DTYPE_BF16, 3U, shape);
    descriptor.value_input =
        binding(buffers[1], SLLM_TENSOR_DTYPE_BF16, 3U, shape);
    sllm_kv_append_info_t info{};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
    return expect(sllm_kv_state_append(state, queue, &descriptor, completion,
                                       &info, &error.sink),
                  SLLM_STATUS_OK, operation, error) &&
           *completion != nullptr && info.backend == SLLM_BACKEND_HIP &&
           info.fallback_allowed == 0U && info.fallback_used == 0U &&
           info.token_count == count && wait_release(completion, operation);
  };
  if (passed) {
    sllm_completion_t *paged_append = nullptr;
    sllm_completion_t *contiguous_append = nullptr;
    passed &= append_state(paged, &paged_append, "paged prefill append") &&
              append_state(contiguous, &contiguous_append,
                           "contiguous prefill append");
  }
  std::vector<uint16_t> paged_output(static_cast<std::size_t>(count) *
                                     kQueryHeads * kHeadDim);
  std::vector<uint16_t> contiguous_output(paged_output.size());
  if (passed) {
    const uint64_t shape[] = {count, kQueryHeads, kHeadDim};
    auto attention = [&](const sllm_kv_state_t *const state,
                         const sllm_buffer_t *const output,
                         const char *const operation,
                         sllm_causal_attention_dispatch_info_t *const info) {
      sllm_causal_attention_desc_t descriptor{};
      descriptor.struct_size = sizeof(descriptor);
      descriptor.abi_version = SLLM_HIP_ABI_VERSION;
      descriptor.op_version = SLLM_HIP_CAUSAL_ATTENTION_VERSION;
      descriptor.expected_kv_length = count;
      descriptor.kv_state = state;
      descriptor.query = binding(buffers[2], SLLM_TENSOR_DTYPE_BF16, 3U, shape);
      descriptor.output = binding(output, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
      sllm_completion_t *completion = nullptr;
      info->struct_size = sizeof(*info);
      info->abi_version = SLLM_HIP_ABI_VERSION;
      info->info_version = SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION;
      const bool submitted =
          expect(sllm_causal_attention_execute(context, queue, &descriptor,
                                               &completion, info, &error.sink),
                 SLLM_STATUS_OK, operation, error) &&
          completion != nullptr && info->backend == SLLM_BACKEND_HIP &&
          info->fallback_allowed == 0U && info->fallback_used == 0U;
      return submitted && wait_release(&completion, operation);
    };
    sllm_causal_attention_dispatch_info_t paged_dispatch{};
    sllm_causal_attention_dispatch_info_t contiguous_dispatch{};
    passed &= attention(paged, buffers[3], "paged prefill attention",
                        &paged_dispatch) &&
              attention(contiguous, buffers[4], "contiguous prefill attention",
                        &contiguous_dispatch) &&
              download(queue, buffers[3], paged_output.data(), output_bytes,
                       "paged output") &&
              download(queue, buffers[4], contiguous_output.data(),
                       output_bytes, "contiguous output");
    std::cout << "prefill m=" << count
              << " paged_kernel=" << paged_dispatch.kernel_symbol
              << " contiguous_kernel=" << contiguous_dispatch.kernel_symbol
              << '\n';
    if (paged_output != contiguous_output) {
      const auto mismatch = std::mismatch(
          paged_output.begin(), paged_output.end(), contiguous_output.begin());
      std::size_t mismatch_count = 0U;
      for (std::size_t index = 0U; index != paged_output.size(); ++index)
        mismatch_count += paged_output[index] != contiguous_output[index];
      std::cerr << "prefill m=" << count << " first_mismatch="
                << std::distance(paged_output.begin(), mismatch.first)
                << " mismatch_count=" << mismatch_count << " paged=0x"
                << std::hex
                << paged_output[static_cast<std::size_t>(
                       std::distance(paged_output.begin(), mismatch.first))]
                << " contiguous=0x"
                << contiguous_output[static_cast<std::size_t>(
                       std::distance(paged_output.begin(), mismatch.first))]
                << std::dec << '\n';
      passed = false;
    }
    for (const uint32_t dimension : {0U, 31U, 32U, 255U}) {
      const std::size_t index =
          (static_cast<std::size_t>(count - 1U) * kQueryHeads) * kHeadDim +
          dimension;
      const double expected =
          oracle_value(keys, values, queries, count, count - 1U, 0U, dimension);
      const double actual = bf16_to_f32(paged_output[index]);
      const double tolerance = std::max(0.08, std::fabs(expected) * 0.01);
      if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
        std::cerr << "prefill m=" << count << " oracle dimension=" << dimension
                  << " actual=" << actual << " expected=" << expected << '\n';
        passed = false;
      }
    }
  }
  const bool cleaned = cleanup();
  std::cout << "phase87 gfx1201 prefill m=" << count
            << " status=" << (passed && cleaned ? "PASS" : "FAIL") << '\n';
  return passed && cleaned;
}

bool run_decode_case(const sllm_context_t *const context,
                     const sllm_queue_t *const queue, const uint32_t count,
                     const std::vector<uint16_t> &keys,
                     const std::vector<uint16_t> &values,
                     const std::vector<uint16_t> &queries) {
  constexpr uint32_t kPrefix = 47U;
  sllm_kv_state_t *paged = nullptr;
  sllm_kv_state_t *contiguous = nullptr;
  std::array<sllm_buffer_t *, 5U> buffers{};
  sllm_completion_t *completion = nullptr;
  bool passed = true;
  Error error;
  const uint64_t kv_bytes =
      static_cast<uint64_t>(kCapacity) * kKvHeads * kHeadDim * sizeof(uint16_t);
  const uint64_t query_bytes = static_cast<uint64_t>(kCapacity) * kQueryHeads *
                               kHeadDim * sizeof(uint16_t);
  const uint64_t output_bytes =
      static_cast<uint64_t>(count) * kQueryHeads * kHeadDim * sizeof(uint16_t);
  const auto cleanup = [&]() {
    bool result = true;
    if (completion != nullptr) {
      (void)sllm_kv_state_append_cancel(paged, completion, &error.sink);
      result = wait_release(&completion, "decode cancel cleanup") && result;
    }
    result = release_state(&contiguous) && result;
    result = release_state(&paged) && result;
    for (sllm_buffer_t *&buffer : buffers)
      result = release_buffer(&buffer) && result;
    return result;
  };
  const auto paged_info = paged_create(UINT64_C(0x874700) + count, count);
  const auto contiguous_info = contiguous_create(paged_info.session_id, count);
  passed &= expect(
      sllm_kv_state_create_paged(context, &paged_info, &paged, &error.sink),
      SLLM_STATUS_OK, "decode paged create", error);
  passed &= expect(sllm_kv_state_create_v2(context, &contiguous_info,
                                           &contiguous, &error.sink),
                   SLLM_STATUS_OK, "decode contiguous create", error);
  passed &= create_buffer(context, kv_bytes, &buffers[0]);
  passed &= create_buffer(context, kv_bytes, &buffers[1]);
  passed &= create_buffer(context, query_bytes, &buffers[2]);
  passed &= create_buffer(context, output_bytes, &buffers[3]);
  passed &= create_buffer(context, output_bytes, &buffers[4]);
  if (passed) {
    passed &=
        upload(queue, buffers[0], keys.data(), kv_bytes, "decode key upload");
    passed &= upload(queue, buffers[1], values.data(), kv_bytes,
                     "decode value upload");
    passed &= upload(queue, buffers[2], queries.data(), query_bytes,
                     "decode query upload");
  }
  auto append = [&](const sllm_kv_state_t *const state,
                    const uint64_t count_value, const uint64_t start,
                    const char *const operation) {
    const uint64_t shape[] = {count_value, kKvHeads, kHeadDim};
    const uint64_t token_bytes =
        static_cast<uint64_t>(kKvHeads) * kHeadDim * sizeof(uint16_t);
    sllm_kv_append_desc_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.append_version = SLLM_HIP_KV_STATE_VERSION;
    descriptor.expected_length = start;
    descriptor.start_position = start;
    descriptor.key_input =
        binding(buffers[0], SLLM_TENSOR_DTYPE_BF16, 3U, shape);
    descriptor.value_input =
        binding(buffers[1], SLLM_TENSOR_DTYPE_BF16, 3U, shape);
    descriptor.key_input.byte_offset = start * token_bytes;
    descriptor.value_input.byte_offset = start * token_bytes;
    sllm_kv_append_info_t info{};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
    return expect(sllm_kv_state_append(state, queue, &descriptor, &completion,
                                       &info, &error.sink),
                  SLLM_STATUS_OK, operation, error) &&
           completion != nullptr && wait_release(&completion, operation);
  };
  if (passed)
    passed = append(paged, kPrefix, 0U, "decode paged prefix") &&
             append(contiguous, kPrefix, 0U, "decode contiguous prefix") &&
             append(paged, count, kPrefix, "decode paged tail") &&
             append(contiguous, count, kPrefix, "decode contiguous tail");
  std::vector<uint16_t> paged_output(static_cast<std::size_t>(count) *
                                     kQueryHeads * kHeadDim);
  std::vector<uint16_t> contiguous_output(paged_output.size());
  if (passed) {
    const uint64_t shape[] = {count, kQueryHeads, kHeadDim};
    const uint64_t query_offset = static_cast<uint64_t>(kPrefix) * kQueryHeads *
                                  kHeadDim * sizeof(uint16_t);
    auto attention = [&](const sllm_kv_state_t *const state,
                         const sllm_buffer_t *const output,
                         const char *const operation) {
      sllm_causal_attention_desc_t descriptor{};
      descriptor.struct_size = sizeof(descriptor);
      descriptor.abi_version = SLLM_HIP_ABI_VERSION;
      descriptor.op_version = SLLM_HIP_CAUSAL_ATTENTION_VERSION;
      descriptor.start_position = kPrefix;
      descriptor.expected_kv_length = kPrefix + count;
      descriptor.kv_state = state;
      descriptor.query = binding(buffers[2], SLLM_TENSOR_DTYPE_BF16, 3U, shape);
      descriptor.output = binding(output, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
      descriptor.query.byte_offset = query_offset;
      sllm_causal_attention_dispatch_info_t info{};
      info.struct_size = sizeof(info);
      info.abi_version = SLLM_HIP_ABI_VERSION;
      info.info_version = SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION;
      sllm_completion_t *attention_completion = nullptr;
      return expect(sllm_causal_attention_execute(context, queue, &descriptor,
                                                  &attention_completion, &info,
                                                  &error.sink),
                    SLLM_STATUS_OK, operation, error) &&
             attention_completion != nullptr && info.fallback_allowed == 0U &&
             info.fallback_used == 0U &&
             wait_release(&attention_completion, operation);
    };
    passed &=
        attention(paged, buffers[3], "decode paged attention") &&
        attention(contiguous, buffers[4], "decode contiguous attention") &&
        download(queue, buffers[3], paged_output.data(), output_bytes,
                 "decode paged output") &&
        download(queue, buffers[4], contiguous_output.data(), output_bytes,
                 "decode contiguous output");
    if (paged_output != contiguous_output) {
      const auto mismatch = std::mismatch(
          paged_output.begin(), paged_output.end(), contiguous_output.begin());
      std::size_t mismatch_count = 0U;
      for (std::size_t index = 0U; index != paged_output.size(); ++index)
        mismatch_count += paged_output[index] != contiguous_output[index];
      std::cerr << "decode m=" << count << " first_mismatch="
                << std::distance(paged_output.begin(), mismatch.first)
                << " mismatch_count=" << mismatch_count << '\n';
      passed = false;
    }
  }
  const bool cleaned = cleanup();
  std::cout << "phase87 decode prefix47 m=" << count
            << " status=" << (passed && cleaned ? "PASS" : "FAIL") << '\n';
  return passed && cleaned;
}

} // namespace

int main() {
  hipUUID uuid{};
  if (hipDeviceGetUuid(&uuid, 0) != hipSuccess ||
      uuid_text(uuid) != SLLM_TEST_EXPECTED_UUID) {
    std::cerr << "unexpected GPU UUID: " << uuid_text(uuid) << '\n';
    return 2;
  }
  Error error;
  sllm_device_info_t device{};
  device.struct_size = sizeof(device);
  device.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(sllm_device_query(0U, &device, &error.sink), SLLM_STATUS_OK,
              "device query", error) ||
      std::string(device.gcn_arch_name) != SLLM_TEST_EXPECTED_TARGET)
    return 2;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  std::strncpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
               sizeof(context_info.expected_gcn_arch_name) - 1U);
  bool passed =
      expect(sllm_context_create(&context_info, &context, &error.sink),
             SLLM_STATUS_OK, "context create", error);
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  if (passed)
    passed &=
        expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
               SLLM_STATUS_OK, "queue create", error);
  if (passed)
    passed &=
        expect(sllm_queue_set_completion_mode(
                   queue, SLLM_QUEUE_COMPLETION_MODE_PROFILED, &error.sink),
               SLLM_STATUS_OK, "profiled completion mode", error);
  const std::vector<uint16_t> keys = make_kv(true);
  const std::vector<uint16_t> values = make_kv(false);
  const std::vector<uint16_t> queries = make_queries();
  if (passed)
    passed = run_case(context, queue, 47U, keys, values, queries);
  if (passed)
    passed = run_case(context, queue, 64U, keys, values, queries);
  if (passed)
    passed = run_case(context, queue, 128U, keys, values, queries);
  uint32_t seed_index = 0U;
  for (const uint32_t seed : {UINT32_C(0x13579bdf), UINT32_C(0x2468ace1),
                              UINT32_C(0xdeadbeef), UINT32_C(0x31415927)}) {
    if (!passed)
      break;
    std::vector<uint16_t> stress_keys;
    std::vector<uint16_t> stress_values;
    std::vector<uint16_t> stress_queries;
    make_stress_vectors(seed, &stress_keys, &stress_values, &stress_queries);
    passed = run_case(context, queue, 47U, stress_keys, stress_values,
                      stress_queries) &&
             run_case(context, queue, 64U, stress_keys, stress_values,
                      stress_queries) &&
             (seed_index != 0U || run_case(context, queue, 128U, stress_keys,
                                           stress_values, stress_queries)) &&
             run_decode_case(context, queue, 1U, stress_keys, stress_values,
                             stress_queries) &&
             run_decode_case(context, queue, 2U, stress_keys, stress_values,
                             stress_queries);
    ++seed_index;
  }
  passed = release_queue(&queue) && passed;
  passed = release_context(&context) && passed;
  if (passed)
    std::cout << "phase87_stage10_paged_gfx1201_prefill_bitwise_gpu_test: PASS "
                 "target=gfx1201 "
                 "cases=m47,m64,m128+4x(m47,m64,m128(seed0),decode1,decode2) "
                 "fallback=0 cleanup=0\n";
  return passed ? 0 : 1;
}
