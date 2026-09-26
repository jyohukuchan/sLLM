// Phase 87 Stage 10: public Paged FP16 KV request correctness.
//
// This test deliberately uses the additive public ABI.  It covers the
// reviewed Qwen GQA6 [24,4,256] and GQA4 [16,4,256] geometries, all three
// sides of the 128-token boundary, M=1..5 decode, and a modest prefill.
// A host FP16 attention oracle independently checks selected output elements.
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
#error SLLM_TEST_EXPECTED_TARGET must name one exact GPU target
#endif

namespace {

constexpr uint32_t kKvHeads = 4U;
uint32_t kQueryHeads = 24U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kTokenBlock = 128U;
constexpr uint64_t kCapacity = 257U;
constexpr uint64_t kLogicalBlocks = 3U;
constexpr uint64_t kMaxPhysicalBlocks = 6U;
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
  if (!waited) {
    std::cerr << operation
              << " did not complete successfully (state=" << result.state
              << ")\n";
  }
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
              SLLM_STATUS_OK, "buffer upload", error)) {
    return false;
  }
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
      completion == nullptr) {
    return false;
  }
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
             "buffer download completion release", error);
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

uint16_t f32_to_f16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint16_t sign = static_cast<uint16_t>((bits >> 16U) & 0x8000U);
  const uint32_t exponent = (bits >> 23U) & 0xffU;
  const uint32_t mantissa = bits & 0x7fffffU;
  if (exponent == 0xffU) {
    if (mantissa == 0U)
      return static_cast<uint16_t>(sign | 0x7c00U);
    return static_cast<uint16_t>(sign | 0x7e00U);
  }
  const int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
  if (half_exponent >= 0x1f)
    return static_cast<uint16_t>(sign | 0x7c00U);
  if (half_exponent <= 0) {
    if (half_exponent < -10)
      return sign;
    const uint32_t significand = mantissa | 0x800000U;
    const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);
    uint32_t rounded = significand >> shift;
    const uint32_t remainder = significand & ((UINT32_C(1) << shift) - 1U);
    const uint32_t halfway = UINT32_C(1) << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0U))
      ++rounded;
    return static_cast<uint16_t>(sign | rounded);
  }
  uint32_t rounded = mantissa >> 13U;
  const uint32_t remainder = mantissa & UINT32_C(0x1fff);
  if (remainder > UINT32_C(0x1000) ||
      (remainder == UINT32_C(0x1000) && (rounded & 1U) != 0U)) {
    ++rounded;
    if (rounded == 0x400U)
      return static_cast<uint16_t>(sign | ((half_exponent + 1) << 10U));
  }
  return static_cast<uint16_t>(
      sign | (static_cast<uint32_t>(half_exponent) << 10U) | rounded);
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

double fp16_value(const std::vector<uint16_t> &values, const uint64_t token,
                  const uint32_t head, const uint32_t dimension) {
  const std::size_t row =
      (static_cast<std::size_t>(token) * kKvHeads + head) * kHeadDim;
  return static_cast<double>(
      f16_to_f32(f32_to_f16(bf16_to_f32(values[row + dimension]))));
}

float query_value(const std::vector<uint16_t> &queries, const uint64_t row,
                  const uint32_t head, const uint32_t dimension) {
  const std::size_t index =
      (static_cast<std::size_t>(row) * kQueryHeads + head) * kHeadDim +
      dimension;
  return bf16_to_f32(queries[index]);
}

double oracle_value(const std::vector<uint16_t> &keys,
                    const std::vector<uint16_t> &values,
                    const std::vector<uint16_t> &queries, const uint64_t length,
                    const uint32_t query_row, const uint32_t query_head,
                    const uint32_t output_dimension) {
  const uint32_t kv_head = query_head / (kQueryHeads / kKvHeads);
  std::vector<double> scores(static_cast<std::size_t>(length));
  double maximum = -std::numeric_limits<double>::infinity();
  for (uint64_t token = 0U; token != length; ++token) {
    double score = 0.0;
    for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
      score += static_cast<double>(
                   query_value(queries, query_row, query_head, dimension)) *
               fp16_value(keys, token, kv_head, dimension);
    }
    scores[static_cast<std::size_t>(token)] = score / 16.0;
    maximum = std::max(maximum, scores[static_cast<std::size_t>(token)]);
  }
  double denominator = 0.0;
  double numerator = 0.0;
  for (uint64_t token = 0U; token != length; ++token) {
    const double weight =
        std::exp(scores[static_cast<std::size_t>(token)] - maximum);
    denominator += weight;
    numerator += weight * fp16_value(values, token, kv_head, output_dimension);
  }
  return numerator / denominator;
}

sllm_kv_append_desc_t append_descriptor(const sllm_buffer_t *const key,
                                        const sllm_buffer_t *const value,
                                        const uint64_t count,
                                        const uint64_t start,
                                        const uint64_t byte_offset) {
  const uint64_t shape[] = {count, kKvHeads, kHeadDim};
  sllm_kv_append_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.append_version = SLLM_HIP_KV_STATE_VERSION;
  descriptor.expected_length = start;
  descriptor.start_position = start;
  descriptor.key_input = binding(key, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
  descriptor.value_input = binding(value, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
  descriptor.key_input.byte_offset = byte_offset;
  descriptor.value_input.byte_offset = byte_offset;
  return descriptor;
}

bool append_state(const sllm_kv_state_t *const state,
                  const sllm_queue_t *const queue,
                  const sllm_buffer_t *const key,
                  const sllm_buffer_t *const value, const uint64_t count,
                  const uint64_t start, sllm_completion_t **const output) {
  const uint64_t token_bytes =
      static_cast<uint64_t>(kKvHeads) * kHeadDim * sizeof(uint16_t);
  const sllm_kv_append_desc_t descriptor =
      append_descriptor(key, value, count, start, start * token_bytes);
  sllm_kv_append_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
  Error error;
  *output = nullptr;
  const uint32_t expected_paged_kernel =
      SLLM_HIP_KV_KERNEL_ID_BF16_TO_PAGED_F16_V1;
  if (!expect(sllm_kv_state_append(state, queue, &descriptor, output, &info,
                                   &error.sink),
              SLLM_STATUS_OK, "paged append", error) ||
      *output == nullptr || info.backend != SLLM_BACKEND_HIP ||
      info.dispatch_count != 1U || info.start_position != start ||
      info.token_count != count || info.end_position != start + count ||
      info.kernel_id != expected_paged_kernel || info.fallback_allowed != 0U ||
      info.fallback_used != 0U ||
      std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) != 0) {
    std::cerr << "append did not report exact HIP/no-fallback target\n";
    return false;
  }
  return true;
}

bool attention_descriptor(const sllm_kv_state_t *const state,
                          const sllm_buffer_t *const query,
                          const sllm_buffer_t *const output,
                          const uint64_t start, const uint64_t length,
                          const uint32_t count, const uint64_t query_offset,
                          sllm_causal_attention_desc_t *const descriptor) {
  const uint64_t shape[] = {count, kQueryHeads, kHeadDim};
  *descriptor = {};
  descriptor->struct_size = sizeof(*descriptor);
  descriptor->abi_version = SLLM_HIP_ABI_VERSION;
  descriptor->op_version = SLLM_HIP_CAUSAL_ATTENTION_VERSION;
  descriptor->start_position = start;
  descriptor->expected_kv_length = length;
  descriptor->kv_state = state;
  descriptor->query = binding(query, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
  descriptor->output = binding(output, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
  descriptor->query.byte_offset = query_offset;
  return true;
}

bool run_attention(const sllm_context_t *const context,
                   const sllm_queue_t *const queue,
                   const sllm_kv_state_t *const state,
                   const sllm_buffer_t *const query,
                   const sllm_buffer_t *const output, const uint64_t start,
                   const uint64_t length, const uint32_t count,
                   sllm_completion_t *const append_dependency, const bool paged,
                   const uint64_t query_offset,
                   std::vector<uint16_t> *const observed) {
  sllm_causal_attention_desc_t descriptor{};
  attention_descriptor(state, query, output, start, length, count, query_offset,
                       &descriptor);
  sllm_causal_attention_dispatch_info_t dispatch{};
  dispatch.struct_size = sizeof(dispatch);
  dispatch.abi_version = SLLM_HIP_ABI_VERSION;
  dispatch.info_version = SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  Error error;
  const sllm_status_t status =
      append_dependency == nullptr
          ? sllm_causal_attention_execute(context, queue, &descriptor,
                                          &completion, &dispatch, &error.sink)
          : sllm_causal_attention_execute_after_kv_append(
                context, queue, append_dependency, &descriptor, &completion,
                &dispatch, &error.sink);
  const uint32_t expected_kernel =
      paged ? (count <= 5U
                   ? SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_DECODE_FP16_V1
                   : SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_PREFILL_FP16_V1)
            : 0U;
  if (!expect(status, SLLM_STATUS_OK,
              append_dependency == nullptr ? "causal attention"
                                           : "append->attention",
              error) ||
      completion == nullptr || dispatch.backend != SLLM_BACKEND_HIP ||
      dispatch.query_count != count || dispatch.start_position != start ||
      dispatch.committed_kv_length != length ||
      dispatch.q_heads != kQueryHeads || dispatch.kv_heads != kKvHeads ||
      dispatch.head_dim != kHeadDim || dispatch.fallback_allowed != 0U ||
      dispatch.fallback_used != 0U ||
      std::strcmp(dispatch.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) != 0 ||
      (paged && dispatch.kernel_id != expected_kernel)) {
    std::cerr << "attention did not report exact HIP/no-fallback provider\n";
    if (completion != nullptr)
      (void)wait_and_release(&completion, "failed attention cleanup");
    return false;
  }
  const bool waited = wait_and_release(
      &completion, append_dependency == nullptr ? "causal attention wait"
                                                : "append->attention wait");
  if (!waited)
    return false;
  observed->resize(static_cast<std::size_t>(count) * kQueryHeads * kHeadDim);
  return download(queue, output, observed->data(),
                  static_cast<uint64_t>(observed->size()) * sizeof(uint16_t));
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
  info.dtype = SLLM_TENSOR_DTYPE_F16;
  info.encoding = SLLM_HIP_KV_ENCODING_FP16_V1;
  info.scale_dtype = 0U;
  info.quantization_block_size = 0U;
  info.token_block_size = kTokenBlock;
  info.physical_layout_version = SLLM_HIP_KV_PAGED_LAYOUT_VERSION;
  info.logical_table_capacity = kLogicalBlocks;
  info.max_physical_blocks = kMaxPhysicalBlocks;
  return info;
}

bool run_case(const sllm_context_t *const context,
              const sllm_queue_t *const queue, const uint64_t prefix_length,
              const uint32_t child_count, const std::vector<uint16_t> &keys,
              const std::vector<uint16_t> &values,
              const std::vector<uint16_t> &queries) {
  const uint64_t input_bytes =
      static_cast<uint64_t>(keys.size()) * sizeof(uint16_t);
  const uint64_t parent_output_bytes =
      prefix_length * kQueryHeads * kHeadDim * sizeof(uint16_t);
  const uint64_t child_output_bytes = static_cast<uint64_t>(child_count) *
                                      kQueryHeads * kHeadDim * sizeof(uint16_t);
  sllm_kv_state_t *paged_parent = nullptr;
  sllm_kv_state_t *paged_child = nullptr;
  std::array<sllm_buffer_t *, 6> buffers{};
  sllm_completion_t *paged_append_completion = nullptr;
  sllm_completion_t *paged_child_append_completion = nullptr;
  sllm_completion_t *cancelled_paged_append = nullptr;
  bool passed = true;
  Error error;

  const auto cleanup = [&]() {
    bool result = true;
    if (paged_child_append_completion != nullptr)
      result = wait_and_release(&paged_child_append_completion,
                                "child append cleanup") &&
               result;
    if (cancelled_paged_append != nullptr)
      result = wait_and_release(&cancelled_paged_append,
                                "cancelled append cleanup") &&
               result;
    if (paged_append_completion != nullptr)
      result = wait_and_release(&paged_append_completion, "append cleanup") &&
               result;
    result = release_state(&paged_child) && result;
    result = release_state(&paged_parent) && result;
    for (auto iterator = buffers.rbegin(); iterator != buffers.rend();
         ++iterator)
      result = release_buffer(&*iterator) && result;
    return result;
  };

  const auto create_states = [&]() {
    const sllm_kv_state_paged_create_info_t paged_info =
        paged_create(UINT64_C(0x871000) + prefix_length,
                     static_cast<uint32_t>(prefix_length));
    return expect(sllm_kv_state_create_paged(context, &paged_info,
                                             &paged_parent, &error.sink),
                  SLLM_STATUS_OK, "paged parent create", error);
  };
  if (!create_states()) {
    (void)cleanup();
    return false;
  }

  const uint64_t query_bytes =
      kCapacity * kQueryHeads * kHeadDim * sizeof(uint16_t);
  if (!create_buffer(context, input_bytes, &buffers[0]) ||
      !create_buffer(context, input_bytes, &buffers[1]) ||
      !create_buffer(context, query_bytes, &buffers[2]) ||
      !create_buffer(context, parent_output_bytes, &buffers[3]) ||
      !create_buffer(context, child_output_bytes, &buffers[4]) ||
      !create_buffer(context, child_output_bytes, &buffers[5]) ||
      !upload(queue, buffers[0], keys.data(), input_bytes) ||
      !upload(queue, buffers[1], values.data(), input_bytes) ||
      !upload(queue, buffers[2], queries.data(), query_bytes)) {
    (void)cleanup();
    return false;
  }

  if (!append_state(paged_parent, queue, buffers[0], buffers[1], prefix_length,
                    0U, &paged_append_completion)) {
    (void)cleanup();
    return false;
  }

  std::vector<uint16_t> paged_parent_output;
  const uint64_t query_row_bytes =
      static_cast<uint64_t>(kQueryHeads) * kHeadDim * sizeof(uint16_t);
  passed =
      run_attention(context, queue, paged_parent, buffers[2], buffers[3], 0U,
                    prefix_length, static_cast<uint32_t>(prefix_length),
                    paged_append_completion, true, 0U, &paged_parent_output) &&
      wait_and_release(&paged_append_completion, "parent paged append wait");
  if (!passed)
    return cleanup() && passed;

  const uint32_t oracle_head = 0U;
  const std::array<uint32_t, 4> oracle_dimensions{0U, 31U, 32U, 255U};
  for (const uint32_t dimension : oracle_dimensions) {
    const std::size_t index =
        (static_cast<std::size_t>(prefix_length - 1U) * kQueryHeads +
         oracle_head) *
            kHeadDim +
        dimension;
    const double expected = oracle_value(
        keys, values, queries, prefix_length,
        static_cast<uint32_t>(prefix_length - 1U), oracle_head, dimension);
    const double actual = bf16_to_f32(paged_parent_output[index]);
    if (!std::isfinite(actual) || std::fabs(actual - expected) > 0.08) {
      std::cerr << "parent oracle mismatch prefix=" << prefix_length
                << " dimension=" << dimension << " actual=" << actual
                << " expected=" << expected << '\n';
      passed = false;
    }
  }

  sllm_kv_paged_view_info_t parent_view{};
  parent_view.struct_size = sizeof(parent_view);
  parent_view.abi_version = SLLM_HIP_ABI_VERSION;
  parent_view.info_version = SLLM_HIP_KV_PAGED_VIEW_INFO_VERSION;
  passed &=
      expect(sllm_kv_state_query_paged(paged_parent, &parent_view, &error.sink),
             SLLM_STATUS_OK, "paged parent query", error);
  passed &= parent_view.dtype == SLLM_TENSOR_DTYPE_F16 &&
            parent_view.encoding == SLLM_HIP_KV_ENCODING_FP16_V1 &&
            parent_view.head_count == kKvHeads &&
            parent_view.head_dim == kHeadDim &&
            parent_view.memory_kind == SLLM_HIP_KV_MEMORY_KIND_PAGED &&
            parent_view.observed_length == prefix_length &&
            parent_view.allocated_physical_blocks >=
                (prefix_length + kTokenBlock - 1U) / kTokenBlock &&
            parent_view.committed_bytes_total != 0U;
  if (!passed) {
    std::cerr << "paged parent view mismatch prefix=" << prefix_length
              << " dtype=" << parent_view.dtype
              << " encoding=" << parent_view.encoding
              << " heads=" << parent_view.head_count
              << " dim=" << parent_view.head_dim
              << " memory=" << parent_view.memory_kind
              << " observed=" << parent_view.observed_length
              << " blocks=" << parent_view.allocated_physical_blocks
              << " committed=" << parent_view.committed_bytes_total << '\n';
  }

  const sllm_kv_state_paged_create_info_t child_info = paged_create(
      UINT64_C(0x871000) + prefix_length, static_cast<uint32_t>(prefix_length));
  sllm_kv_paged_state_fork_info_t fork_info{};
  fork_info.struct_size = sizeof(fork_info);
  fork_info.abi_version = SLLM_HIP_ABI_VERSION;
  fork_info.info_version = SLLM_HIP_KV_PAGED_STATE_FORK_INFO_VERSION;
  passed &=
      expect(sllm_kv_state_fork_paged(paged_parent, &child_info, &paged_child,
                                      &fork_info, &error.sink),
             SLLM_STATUS_OK, "paged fork", error);
  const uint64_t shared_blocks =
      (prefix_length + kTokenBlock - 1U) / kTokenBlock;
  passed &= paged_child != nullptr &&
            fork_info.published_length == prefix_length &&
            fork_info.shared_physical_blocks == shared_blocks &&
            fork_info.copied_physical_blocks == 0U &&
            fork_info.child_state_identity != 0U;
  if (!passed) {
    std::cerr << "paged fork metadata mismatch prefix=" << prefix_length
              << " published=" << fork_info.published_length
              << " shared=" << fork_info.shared_physical_blocks
              << " copied=" << fork_info.copied_physical_blocks
              << " child_state=" << fork_info.child_state_identity << '\n';
  }

  if (passed && prefix_length == 129U && child_count == 1U) {
    if (!append_state(paged_child, queue, buffers[0], buffers[1], 1U,
                      prefix_length, &cancelled_paged_append) ||
        !expect(sllm_kv_state_append_cancel(paged_child, cancelled_paged_append,
                                            &error.sink),
                SLLM_STATUS_OK, "paged append cancel", error) ||
        !wait_and_release(&cancelled_paged_append,
                          "cancelled paged append wait"))
      return cleanup() && false;
    sllm_kv_paged_view_info_t cancelled_view{};
    cancelled_view.struct_size = sizeof(cancelled_view);
    cancelled_view.abi_version = SLLM_HIP_ABI_VERSION;
    cancelled_view.info_version = SLLM_HIP_KV_PAGED_VIEW_INFO_VERSION;
    passed &= expect(
        sllm_kv_state_query_paged(paged_child, &cancelled_view, &error.sink),
        SLLM_STATUS_OK, "cancelled paged child query", error);
    passed &= cancelled_view.observed_length == prefix_length;
  }

  if (!append_state(paged_child, queue, buffers[0], buffers[1], child_count,
                    prefix_length, &paged_child_append_completion)) {
    return cleanup() && false;
  }
  std::vector<uint16_t> paged_child_output;
  passed =
      run_attention(context, queue, paged_child, buffers[2], buffers[4],
                    prefix_length, prefix_length + child_count, child_count,
                    paged_child_append_completion, true,
                    prefix_length * query_row_bytes, &paged_child_output) &&
      wait_and_release(&paged_child_append_completion,
                       "child paged append wait");
  if (!passed)
    return cleanup() && passed;
  for (const uint32_t dimension : oracle_dimensions) {
    const std::size_t index =
        (static_cast<std::size_t>(child_count - 1U) * kQueryHeads +
         oracle_head) *
            kHeadDim +
        dimension;
    const double expected =
        oracle_value(keys, values, queries, prefix_length + child_count,
                     static_cast<uint32_t>(prefix_length + child_count - 1U),
                     oracle_head, dimension);
    const double actual = bf16_to_f32(paged_child_output[index]);
    if (!std::isfinite(actual) || std::fabs(actual - expected) > 0.08) {
      std::cerr << "child oracle mismatch prefix=" << prefix_length
                << " dimension=" << dimension << " actual=" << actual
                << " expected=" << expected << '\n';
      passed = false;
    }
  }

  sllm_kv_paged_view_info_t parent_after_cow{};
  parent_after_cow.struct_size = sizeof(parent_after_cow);
  parent_after_cow.abi_version = SLLM_HIP_ABI_VERSION;
  parent_after_cow.info_version = SLLM_HIP_KV_PAGED_VIEW_INFO_VERSION;
  sllm_kv_paged_view_info_t child_view{};
  child_view.struct_size = sizeof(child_view);
  child_view.abi_version = SLLM_HIP_ABI_VERSION;
  child_view.info_version = SLLM_HIP_KV_PAGED_VIEW_INFO_VERSION;
  passed &=
      expect(sllm_kv_state_query_paged(paged_parent, &parent_after_cow,
                                       &error.sink),
             SLLM_STATUS_OK, "paged parent post-COW query", error) &&
      expect(sllm_kv_state_query_paged(paged_child, &child_view, &error.sink),
             SLLM_STATUS_OK, "paged child query", error) &&
      parent_after_cow.observed_length == prefix_length &&
      child_view.observed_length == prefix_length + child_count;
  sllm_kv_paged_state_fork_info_t after_cow{};
  after_cow.struct_size = sizeof(after_cow);
  after_cow.abi_version = SLLM_HIP_ABI_VERSION;
  after_cow.info_version = SLLM_HIP_KV_PAGED_STATE_FORK_INFO_VERSION;
  passed &= expect(
      sllm_kv_state_fork_query_paged(paged_child, &after_cow, &error.sink),
      SLLM_STATUS_OK, "paged post-COW fork query", error);
  const uint64_t expected_shared_blocks = prefix_length == 127U ? 0U : 1U;
  const uint64_t expected_copied_blocks =
      prefix_length % kTokenBlock == 0U ? 0U : 1U;
  const uint64_t bytes_per_token =
      static_cast<uint64_t>(kKvHeads) * kHeadDim * 4U;
  passed &=
      after_cow.shared_physical_blocks == expected_shared_blocks &&
      after_cow.copied_physical_blocks == expected_copied_blocks &&
      after_cow.shared_bytes ==
          expected_shared_blocks * bytes_per_token * kTokenBlock &&
      after_cow.copied_bytes == (prefix_length % kTokenBlock) * bytes_per_token;
  if (!passed) {
    std::cerr << "paged COW metadata mismatch prefix=" << prefix_length
              << " child_count=" << child_count
              << " shared=" << after_cow.shared_physical_blocks
              << " copied=" << after_cow.copied_physical_blocks
              << " shared_bytes=" << after_cow.shared_bytes
              << " copied_bytes=" << after_cow.copied_bytes << '\n';
  }
  return cleanup() && passed;
}

} // namespace

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
        const std::size_t index =
            (static_cast<std::size_t>(token) * kKvHeads + head) * kHeadDim +
            dimension;
        result[index] = f32_to_bf16(signed_value);
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
            device.visible_device_count == 1U && device.wavefront_size == 32U &&
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
  sllm_context_probe_result_t probe{};
  probe.struct_size = sizeof(probe);
  probe.abi_version = SLLM_HIP_ABI_VERSION;
  passed &= context != nullptr &&
            expect(sllm_context_probe(context, &probe, &error.sink),
                   SLLM_STATUS_OK, "context probe", error) &&
            probe.context_present == 1U && probe.hip_available == 1U;
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  passed &= context != nullptr &&
            expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
                   SLLM_STATUS_OK, "queue create", error) &&
            queue != nullptr;

  const std::vector<uint16_t> keys = make_kv(true);
  const std::vector<uint16_t> values = make_kv(false);
  for (const uint32_t query_heads : {24U, 16U}) {
    kQueryHeads = query_heads;
    const std::vector<uint16_t> queries = make_queries();
    for (const uint64_t prefix :
         {UINT64_C(127), UINT64_C(128), UINT64_C(129)}) {
      if (passed &&
          !run_case(context, queue, prefix, 1U, keys, values, queries)) {
        std::cerr << "public paged FP16 request failed q_heads=" << query_heads
                  << " prefix=" << prefix << '\n';
        passed = false;
      }
    }
    for (uint32_t count = 2U; passed && count <= 5U; ++count) {
      passed = run_case(context, queue, 129U, count, keys, values, queries);
    }
  }
  passed = release_queue(&queue) && passed;
  passed = release_context(&context) && passed;
  if (passed) {
    std::cout << "phase87_stage10_paged_fp16_public_gpu_test: PASS target="
              << SLLM_TEST_EXPECTED_TARGET
              << " cases=14 geometries=gqa6[24,4,256]+gqa4[16,4,256] "
                 "boundaries=127,128,129 decode_m=1..5 prefill=1 "
                 " oracle=independent-fp16-attention fallback=0 cleanup=0\n";
  }
  return passed ? 0 : 1;
}
