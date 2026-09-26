// Phase 87 Stage 10: public Paged KV whole-graph capture/replay.
//
// This test covers the production Qwen3.8 MXFP8-E4 GQA6 graph path.  The
// phase command is captured ahead of the Paged append and attention nodes;
// replay then selects the active position and row count from ControlV1.
#include "decode_control_kernel_internal.hpp"
#include "decode_graph_capture_internal.hpp"
#include "sllm/hip.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
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
constexpr uint64_t kPrefix = 127U;
constexpr uint64_t kCapacity = 257U;
constexpr uint64_t kLogicalBlocks = 3U;
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
  if (actual == expected) {
    return true;
  }
  std::cerr << operation << " status=" << actual << " expected=" << expected
            << " message=" << error.message << '\n';
  return false;
}

std::string uuid_text(const hipUUID &uuid) {
  std::string value(uuid.bytes, sizeof(uuid.bytes));
  for (char &character : value) {
    const auto byte = static_cast<unsigned char>(character);
    if (!std::isxdigit(byte)) {
      return {};
    }
    character = static_cast<char>(std::tolower(byte));
  }
  return "GPU-" + value;
}

bool release_buffer(sllm_buffer_t **const buffer) {
  if (buffer == nullptr || *buffer == nullptr) {
    return true;
  }
  Error error;
  return expect(sllm_buffer_release(buffer, &error.sink), SLLM_STATUS_OK,
                "buffer release", error) &&
         *buffer == nullptr;
}

bool release_state(sllm_kv_state_t **const state) {
  if (state == nullptr || *state == nullptr) {
    return true;
  }
  Error error;
  return expect(sllm_kv_state_release(state, &error.sink), SLLM_STATUS_OK,
                "state release", error) &&
         *state == nullptr;
}

bool release_queue(sllm_queue_t **const queue) {
  if (queue == nullptr || *queue == nullptr) {
    return true;
  }
  Error error;
  return expect(sllm_queue_release(queue, &error.sink), SLLM_STATUS_OK,
                "queue release", error) &&
         *queue == nullptr;
}

bool release_context(sllm_context_t **const context) {
  if (context == nullptr || *context == nullptr) {
    return true;
  }
  Error error;
  return expect(sllm_context_release(context, &error.sink), SLLM_STATUS_OK,
                "context release", error) &&
         *context == nullptr;
}

bool set_completion_mode(const sllm_queue_t *const queue,
                         const sllm_queue_completion_mode_t mode,
                         const char *const operation) {
  Error error;
  return expect(sllm_queue_set_completion_mode(queue, mode, &error.sink),
                SLLM_STATUS_OK, operation, error);
}

bool release_span(sllm_graph_span_t **const span) {
  if (span == nullptr || *span == nullptr) {
    return true;
  }
  Error error;
  return expect(sllm_graph_span_release(span, &error.sink), SLLM_STATUS_OK,
                "graph release", error) &&
         *span == nullptr;
}

bool wait_release(sllm_completion_t **const completion,
                  const char *const operation,
                  uint64_t *const timing_ns = nullptr) {
  if (completion == nullptr || *completion == nullptr) {
    std::cerr << operation << " returned no completion\n";
    return false;
  }
  if (timing_ns != nullptr)
    *timing_ns = 0U;
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  const bool waited = expect(sllm_completion_wait(*completion, kTimeoutMs,
                                                  &result, &error.sink),
                             SLLM_STATUS_OK, operation, error) &&
                      result.state == SLLM_COMPLETION_STATE_SUCCESS;
  bool timed = true;
  if (waited && timing_ns != nullptr) {
    sllm_completion_timing_t timing{};
    timing.struct_size = sizeof(timing);
    timing.abi_version = SLLM_HIP_ABI_VERSION;
    timed = expect(sllm_completion_timing(*completion, &timing, &error.sink),
                   SLLM_STATUS_OK, "completion timing", error) &&
            timing.valid != 0U && timing.elapsed_ns != 0U;
    if (timed)
      *timing_ns = timing.elapsed_ns;
  }
  const bool released = expect(sllm_completion_release(completion, &error.sink),
                               SLLM_STATUS_OK, "completion release", error) &&
                        *completion == nullptr;
  return waited && timed && released;
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
            const void *const source, const uint64_t bytes,
            const char *const operation = "buffer upload") {
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
              SLLM_STATUS_OK, operation, error)) {
    return false;
  }
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
                           SLLM_STATUS_OK, "buffer download read", error) &&
                    bytes_written == bytes;
  const bool released =
      expect(sllm_completion_release(&completion, &error.sink), SLLM_STATUS_OK,
             "buffer download release", error) &&
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

sllm_tensor_binding_t byte_binding(const sllm_buffer_t *const buffer,
                                   const uint64_t bytes) {
  const uint64_t shape[] = {bytes};
  return binding(buffer, SLLM_TENSOR_DTYPE_U8, 1U, shape);
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
  if (magnitude == 0.0F) {
    return sign;
  }
  if (!std::isfinite(magnitude) || magnitude >= 448.0F) {
    return static_cast<uint8_t>(sign | UINT8_C(0x7e));
  }
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
  if (exponent == 0U) {
    return sign * static_cast<double>(mantissa) * std::ldexp(1.0, -9);
  }
  return sign * std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0,
                           static_cast<int>(exponent) - 7);
}

uint8_t mxfp8_scale(const std::vector<uint16_t> &values, const std::size_t row,
                    const uint32_t dimension) {
  const std::size_t begin =
      row + static_cast<std::size_t>(dimension / kQuantizationBlock) *
                kQuantizationBlock;
  float maximum = 0.0F;
  for (uint32_t lane = 0U; lane != kQuantizationBlock; ++lane) {
    maximum = std::max(maximum, std::fabs(bf16_to_f32(values[begin + lane])));
  }
  if (maximum == 0.0F) {
    return UINT8_C(127);
  }
  const int exponent = std::ilogb(maximum);
  return static_cast<uint8_t>(std::clamp(exponent - 8, -127, 127) + 127);
}

double quantized_value(const std::vector<uint16_t> &values,
                       const uint64_t token, const uint32_t head,
                       const uint32_t dimension) {
  const std::size_t row =
      (static_cast<std::size_t>(token) * kKvHeads + head) * kHeadDim;
  const uint8_t scale_code = mxfp8_scale(values, row, dimension);
  const float scale = std::ldexp(1.0F, static_cast<int>(scale_code) - 127);
  return e4m3fn_decode(
             e4m3fn_encode(bf16_to_f32(values[row + dimension]) / scale)) *
         scale;
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
               quantized_value(keys, token, kv_head, dimension);
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
        if ((row + head * 3U + dimension) % 11U == 0U) {
          value = -value;
        }
        result[(static_cast<std::size_t>(row) * kQueryHeads + head) * kHeadDim +
               dimension] = f32_to_bf16(value);
      }
    }
  }
  return result;
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

sllm_kv_append_desc_t append_descriptor(const sllm_buffer_t *const key,
                                        const sllm_buffer_t *const value,
                                        const uint64_t count,
                                        const uint64_t start) {
  const uint64_t shape[] = {count, kKvHeads, kHeadDim};
  const uint64_t token_bytes =
      static_cast<uint64_t>(kKvHeads) * kHeadDim * sizeof(uint16_t);
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

bool append_and_wait(const sllm_kv_state_t *const state,
                     const sllm_queue_t *const queue,
                     const sllm_buffer_t *const key,
                     const sllm_buffer_t *const value, const uint64_t count,
                     const uint64_t start, sllm_completion_t **const retained,
                     const bool wait, uint64_t *const timing_ns = nullptr) {
  const sllm_kv_append_desc_t descriptor =
      append_descriptor(key, value, count, start);
  sllm_kv_append_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
  Error error;
  *retained = nullptr;
  if (!expect(sllm_kv_state_append(state, queue, &descriptor, retained, &info,
                                   &error.sink),
              SLLM_STATUS_OK, "KV append", error) ||
      *retained == nullptr || info.backend != SLLM_BACKEND_HIP ||
      info.dispatch_count != 1U || info.start_position != start ||
      info.token_count != count || info.end_position != start + count ||
      info.fallback_allowed != 0U || info.fallback_used != 0U ||
      info.kernel_id != SLLM_HIP_KV_KERNEL_ID_BF16_TO_PAGED_MXFP8_E4_V1 ||
      std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) != 0) {
    std::cerr << "append did not report exact HIP/no-fallback target\n";
    return false;
  }
  return !wait || wait_release(retained, "paged KV append wait", timing_ns);
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

bool check_oracle(const std::vector<uint16_t> &keys,
                  const std::vector<uint16_t> &values,
                  const std::vector<uint16_t> &queries,
                  const std::vector<uint16_t> &output, const uint32_t m) {
  const std::array<uint32_t, 4U> dimensions = {0U, 31U, 32U, 255U};
  for (uint32_t row = 0U; row != m; ++row) {
    for (const uint32_t dimension : dimensions) {
      const std::size_t index =
          (static_cast<std::size_t>(row) * kQueryHeads) * kHeadDim + dimension;
      const double expected = oracle_value(keys, values, queries, kPrefix + m,
                                           kPrefix + row, 0U, dimension);
      const double actual = bf16_to_f32(output[index]);
      if (!std::isfinite(actual) || std::fabs(actual - expected) > 0.08) {
        std::cerr << "graph oracle mismatch m=" << m << " row=" << row
                  << " dimension=" << dimension << " actual=" << actual
                  << " expected=" << expected << '\n';
        return false;
      }
    }
  }
  return true;
}

bool run_graph_case(
    const sllm_context_t *const context, const sllm_queue_t *const graph_queue,
    const sllm_buffer_t *const key, const sllm_buffer_t *const value,
    const sllm_buffer_t *const query, const sllm_buffer_t *const control_buffer,
    const std::vector<uint16_t> &keys, const std::vector<uint16_t> &values,
    const std::vector<uint16_t> &queries, const uint32_t m,
    const uint64_t session) {
  sllm_kv_state_t *paged = nullptr;
  sllm_buffer_t *graph_output = nullptr;
  sllm_graph_span_t *span = nullptr;
  sllm_completion_t *paged_prefix = nullptr;
  sllm_completion_t *graph_append = nullptr;
  sllm_completion_t *graph_attention = nullptr;
  sllm_completion_t *graph_replay = nullptr;
  sllm_completion_t *fence = nullptr;
  uint64_t paged_prefix_append_timing_ns = 0U;
  uint64_t graph_execute_to_fence_wait_wall_ns = 0U;
  bool capturing = false;
  bool passed = true;
  Error error;
  const uint64_t query_row_bytes =
      static_cast<uint64_t>(kQueryHeads) * kHeadDim * sizeof(uint16_t);
  const uint64_t output_bytes = static_cast<uint64_t>(m) * query_row_bytes;

  const auto cleanup = [&]() {
    bool result = true;
    if (capturing && span != nullptr) {
      Error abort_error;
      result = expect(sllm_graph_span_abort_capture(&span, &abort_error.sink),
                      SLLM_STATUS_OK, "graph abort cleanup", abort_error) &&
               result;
      capturing = false;
    }
    if (graph_append != nullptr) {
      (void)sllm_kv_state_append_cancel(paged, graph_append, &error.sink);
      result = wait_release(&graph_append, "graph append cleanup") && result;
    }
    if (graph_attention != nullptr) {
      result =
          wait_release(&graph_attention, "graph attention cleanup") && result;
    }
    if (graph_replay != nullptr) {
      result = wait_release(&graph_replay, "graph replay cleanup") && result;
    }
    if (fence != nullptr) {
      result = wait_release(&fence, "graph fence cleanup") && result;
    }
    result = release_span(&span) && result;
    result = release_state(&paged) && result;
    result = release_buffer(&graph_output) && result;
    return result;
  };

  const auto paged_info = paged_create(session, static_cast<uint32_t>(kPrefix));
  passed &= expect(
      sllm_kv_state_create_paged(context, &paged_info, &paged, &error.sink),
      SLLM_STATUS_OK, "graph paged create", error);
  passed &= create_buffer(context, output_bytes, &graph_output);
  if (!passed) {
    return cleanup() && passed;
  }

  // Keep Paged pool/table publication on the same queue that will own graph
  // preflight and replay.  The public pool contract is stream ordered; using
  // a second setup queue would make this test depend on cross-stream table
  // publication that the graph ABI does not promise.
  passed &=
      append_and_wait(paged, graph_queue, key, value, kPrefix, 0U,
                      &paged_prefix, true, &paged_prefix_append_timing_ns);
  if (!passed) {
    return cleanup() && passed;
  }

  sllm_decode_control::ControlV1 control{};
  control.version = sllm_decode_control::kVersion;
  control.status = static_cast<uint32_t>(sllm_decode_control::Status::Ok);
  control.mode = m == 1U ? sllm_decode_control::kModeTargetOnly
                         : sllm_decode_control::kModeMtp;
  control.width = m == 1U ? 1U : m - 1U;
  control.active_width = m == 1U ? 0U : m - 1U;
  control.model_position = kPrefix;
  control.capacity = kCapacity;
  control.output_limit = m;
  control.generation = 1U;
  control.stop_row = sllm_decode_control::kNoStop;
  control.hidden_row = sllm_decode_control::kNoStop;
  passed &= upload(graph_queue, control_buffer, &control, sizeof(control),
                   "graph control upload");
  passed &=
      set_completion_mode(graph_queue, SLLM_QUEUE_COMPLETION_MODE_DEFERRED,
                          "graph capture deferred mode");
  if (!passed) {
    return cleanup() && passed;
  }

  const sllm_tensor_binding_t control_binding =
      byte_binding(control_buffer, sizeof(control));
  passed &=
      expect(sllm_graph_span_begin_capture(
                 context, graph_queue, &control_binding, &span, &error.sink),
             SLLM_STATUS_OK, "graph begin", error) &&
      span != nullptr;
  capturing = passed;

  if (passed) {
    sllm_graph_span_decode_command_desc_t phase{};
    phase.struct_size = sizeof(phase);
    phase.abi_version = SLLM_HIP_ABI_VERSION;
    phase.info_version = SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_VERSION;
    phase.opcode = SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_BEGIN_PHASE;
    phase.phase_kind = sllm_decode_control::kPhaseTarget;
    phase.phase_index = 0U;
    phase.rows = m;
    passed &= expect(sllm_graph_span_decode_command(span, &phase, &error.sink),
                     SLLM_STATUS_OK, "graph begin phase", error);
  }
  if (passed) {
    sllm_kv_append_info_t append_info{};
    append_info.struct_size = sizeof(append_info);
    append_info.abi_version = SLLM_HIP_ABI_VERSION;
    append_info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
    const auto descriptor = append_descriptor(key, value, m, kPrefix);
    passed &=
        expect(sllm_kv_state_append(paged, graph_queue, &descriptor,
                                    &graph_append, &append_info, &error.sink),
               SLLM_STATUS_OK, "captured paged append", error) &&
        graph_append != nullptr && append_info.dispatch_count == 1U &&
        append_info.kernel_id ==
            SLLM_HIP_KV_KERNEL_ID_BF16_TO_PAGED_MXFP8_E4_V1 &&
        append_info.fallback_allowed == 0U && append_info.fallback_used == 0U;
    if (passed) {
      passed &= expect(sllm_graph_span_capture_marker(span, &graph_append,
                                                      &error.sink),
                       SLLM_STATUS_OK, "captured append marker", error) &&
                graph_append == nullptr;
    }
  }
  if (passed) {
    sllm_causal_attention_desc_t descriptor{};
    attention_descriptor(paged, query, graph_output, kPrefix, kPrefix + m, m,
                         kPrefix * query_row_bytes, &descriptor);
    sllm_causal_attention_dispatch_info_t dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    dispatch.info_version = SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION;
    passed &=
        expect(sllm_causal_attention_execute(context, graph_queue, &descriptor,
                                             &graph_attention, &dispatch,
                                             &error.sink),
               SLLM_STATUS_OK, "captured paged attention", error) &&
        graph_attention != nullptr && dispatch.backend == SLLM_BACKEND_HIP &&
        dispatch.query_count == m && dispatch.fallback_allowed == 0U &&
        dispatch.fallback_used == 0U &&
        dispatch.kernel_id ==
            SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_DECODE_GQA6_V1;
    if (passed) {
      passed &= expect(sllm_graph_span_capture_marker(span, &graph_attention,
                                                      &error.sink),
                       SLLM_STATUS_OK, "captured attention marker", error) &&
                graph_attention == nullptr;
    }
  }

  sllm_graph_span_capture_info_t capture_info{};
  if (passed) {
    capture_info.struct_size = sizeof(capture_info);
    capture_info.abi_version = SLLM_HIP_ABI_VERSION;
    capture_info.info_version = SLLM_HIP_GRAPH_SPAN_CAPTURE_INFO_VERSION;
    passed &=
        expect(sllm_graph_span_end_capture(span, &capture_info, &error.sink),
               SLLM_STATUS_OK, "graph end", error) &&
        capture_info.kernel_node_count >= 3U;
    capturing = false;
  }
  if (!passed) {
    return cleanup() && passed;
  }
  passed &=
      expect(sllm_graph_span_prepare_paged_kv(span, kPrefix + m, &error.sink),
             SLLM_STATUS_OK, "graph paged preflight", error);
  const auto graph_execute_started = std::chrono::steady_clock::now();
  if (passed) {
    passed &= expect(sllm_graph_span_execute(span, &graph_replay, &error.sink),
                     SLLM_STATUS_OK, "graph replay", error) &&
              graph_replay != nullptr;
  }
  if (passed) {
    passed &= expect(sllm_queue_fence(graph_queue, &fence, &error.sink),
                     SLLM_STATUS_OK, "graph fence", error) &&
              fence != nullptr;
    sllm_completion_result_t fence_result{};
    fence_result.struct_size = sizeof(fence_result);
    fence_result.abi_version = SLLM_HIP_ABI_VERSION;
    passed &= expect(sllm_completion_wait(fence, kTimeoutMs, &fence_result,
                                          &error.sink),
                     SLLM_STATUS_OK, "graph fence wait", error) &&
              fence_result.state == SLLM_COMPLETION_STATE_SUCCESS;
    if (passed) {
      graph_execute_to_fence_wait_wall_ns = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - graph_execute_started)
              .count());
    }
    sllm_completion_result_t replay_result{};
    replay_result.struct_size = sizeof(replay_result);
    replay_result.abi_version = SLLM_HIP_ABI_VERSION;
    passed &= expect(sllm_completion_finalize_after(
                         graph_replay, fence, &replay_result, &error.sink),
                     SLLM_STATUS_OK, "graph replay finalize", error);
    passed &= expect(sllm_completion_release(&graph_replay, &error.sink),
                     SLLM_STATUS_OK, "graph replay release", error) &&
              graph_replay == nullptr;
    passed &= expect(sllm_completion_release(&fence, &error.sink),
                     SLLM_STATUS_OK, "graph fence release", error) &&
              fence == nullptr;
  }

  std::vector<uint16_t> graph_result(static_cast<std::size_t>(m) * kQueryHeads *
                                     kHeadDim);
  if (passed) {
    passed &= download(graph_queue, graph_output, graph_result.data(),
                       output_bytes, "graph output readback");
    passed &= check_oracle(keys, values, queries, graph_result, m);
  }
  if (passed) {
    passed &= expect(sllm_graph_span_publish_state_metadata(
                         span, kPrefix, kPrefix + m, 1U, &error.sink),
                     SLLM_STATUS_OK, "graph metadata publish", error);
  }
  sllm_kv_paged_view_info_t view{};
  view.struct_size = sizeof(view);
  view.abi_version = SLLM_HIP_ABI_VERSION;
  view.info_version = SLLM_HIP_KV_PAGED_VIEW_INFO_VERSION;
  if (passed) {
    passed &= expect(sllm_kv_state_query_paged(paged, &view, &error.sink),
                     SLLM_STATUS_OK, "graph state query", error) &&
              view.observed_length == kPrefix + m &&
              view.committed_bytes_total != 0U;
  }
  passed &= cleanup();
  passed &=
      set_completion_mode(graph_queue, SLLM_QUEUE_COMPLETION_MODE_PROFILED,
                          "graph next setup profiled mode");
  std::cout << "phase87 graph m=" << m
            << " status=" << (passed ? "PASS" : "FAIL")
            << " kv_append_timing_ns=" << paged_prefix_append_timing_ns
            << " graph_execute_to_fence_wait_wall_ns="
            << graph_execute_to_fence_wait_wall_ns
            << " graph_timing_source=host_steady_clock_execute_to_fence_wait"
            << " graph_completion_timing=unsupported" << '\n';
  return passed;
}

bool run_cancel_case(const sllm_context_t *const context,
                     const sllm_queue_t *const queue,
                     const sllm_buffer_t *const key,
                     const sllm_buffer_t *const value) {
  sllm_kv_state_t *state = nullptr;
  sllm_completion_t *prefix = nullptr;
  sllm_completion_t *append = nullptr;
  Error error;
  bool passed = true;
  passed &= set_completion_mode(queue, SLLM_QUEUE_COMPLETION_MODE_PROFILED,
                                "cancel setup profiled mode");
  const auto info = paged_create(UINT64_C(0x87abc), 0x87U);
  passed &=
      expect(sllm_kv_state_create_paged(context, &info, &state, &error.sink),
             SLLM_STATUS_OK, "cancel state create", error);
  passed &=
      append_and_wait(state, queue, key, value, kPrefix, 0U, &prefix, true);
  if (passed) {
    passed &= append_and_wait(state, queue, key, value, 1U, kPrefix, &append,
                              false) &&
              expect(sllm_kv_state_append_cancel(state, append, &error.sink),
                     SLLM_STATUS_OK, "cancel append", error) &&
              wait_release(&append, "cancel append wait");
  }
  sllm_kv_paged_view_info_t view{};
  view.struct_size = sizeof(view);
  view.abi_version = SLLM_HIP_ABI_VERSION;
  view.info_version = SLLM_HIP_KV_PAGED_VIEW_INFO_VERSION;
  passed &= expect(sllm_kv_state_query_paged(state, &view, &error.sink),
                   SLLM_STATUS_OK, "cancel state query", error) &&
            view.observed_length == kPrefix;
  if (append != nullptr) {
    (void)sllm_kv_state_append_cancel(state, append, &error.sink);
    passed &= wait_release(&append, "cancel append cleanup");
  }
  passed &= release_state(&state);
  std::cout << "phase87 graph cancel status=" << (passed ? "PASS" : "FAIL")
            << '\n';
  return passed;
}

// Keep this small eager-DEFERRED transaction beside the graph test because
// graph setup uses the same Paged append completion contract.  The append is
// finalized only after one queue fence; waiting on the append itself would
// bypass the deferred completion path under test.
bool run_deferred_append_case(const sllm_context_t *const context,
                              const sllm_queue_t *const queue,
                              const sllm_buffer_t *const key,
                              const sllm_buffer_t *const value) {
  sllm_kv_state_t *state = nullptr;
  sllm_completion_t *append = nullptr;
  sllm_completion_t *fence = nullptr;
  Error error;
  bool passed = set_completion_mode(queue, SLLM_QUEUE_COMPLETION_MODE_DEFERRED,
                                    "deferred append mode");
  bool append_finalized = false;
  const auto info =
      paged_create(UINT64_C(0x87def), static_cast<uint32_t>(kPrefix));
  passed &=
      expect(sllm_kv_state_create_paged(context, &info, &state, &error.sink),
             SLLM_STATUS_OK, "deferred state create", error);
  if (passed) {
    passed &=
        append_and_wait(state, queue, key, value, kPrefix, 0U, &append, false);
  }
  if (passed) {
    passed &= expect(sllm_queue_fence(queue, &fence, &error.sink),
                     SLLM_STATUS_OK, "deferred append fence", error) &&
              fence != nullptr;
  }
  if (passed) {
    sllm_completion_result_t fence_result{};
    fence_result.struct_size = sizeof(fence_result);
    fence_result.abi_version = SLLM_HIP_ABI_VERSION;
    passed &= expect(sllm_completion_wait(fence, kTimeoutMs, &fence_result,
                                          &error.sink),
                     SLLM_STATUS_OK, "deferred append fence wait", error) &&
              fence_result.state == SLLM_COMPLETION_STATE_SUCCESS;
    sllm_completion_result_t append_result{};
    append_result.struct_size = sizeof(append_result);
    append_result.abi_version = SLLM_HIP_ABI_VERSION;
    passed &= expect(sllm_completion_finalize_after(
                         append, fence, &append_result, &error.sink),
                     SLLM_STATUS_OK, "deferred append finalize", error) &&
              append_result.state == SLLM_COMPLETION_STATE_SUCCESS;
    append_finalized = passed;
  }
  if (append != nullptr) {
    if (!append_finalized) {
      (void)sllm_kv_state_append_cancel(state, append, &error.sink);
      passed &= wait_release(&append, "deferred append cleanup");
    } else {
      passed &= expect(sllm_completion_release(&append, &error.sink),
                       SLLM_STATUS_OK, "deferred append release", error) &&
                append == nullptr;
    }
  }
  if (fence != nullptr) {
    passed &= expect(sllm_completion_release(&fence, &error.sink),
                     SLLM_STATUS_OK, "deferred fence release", error) &&
              fence == nullptr;
  }
  sllm_kv_paged_view_info_t view{};
  view.struct_size = sizeof(view);
  view.abi_version = SLLM_HIP_ABI_VERSION;
  view.info_version = SLLM_HIP_KV_PAGED_VIEW_INFO_VERSION;
  if (state != nullptr) {
    passed &= expect(sllm_kv_state_query_paged(state, &view, &error.sink),
                     SLLM_STATUS_OK, "deferred state query", error) &&
              view.observed_length == kPrefix;
  }
  passed &= release_state(&state);
  passed &= set_completion_mode(queue, SLLM_QUEUE_COMPLETION_MODE_PROFILED,
                                "deferred append restore profiled mode");
  std::cout << "phase87 graph deferred append status="
            << (passed ? "PASS" : "FAIL") << '\n';
  return passed;
}

} // namespace

int main() {
  hipUUID uuid{};
  if (hipDeviceGetUuid(&uuid, 0) != hipSuccess ||
      uuid_text(uuid) != SLLM_TEST_EXPECTED_UUID) {
    std::cerr << "unexpected GPU UUID: " << uuid_text(uuid) << '\n';
    return 2;
  }
  sllm_device_info_t device{};
  device.struct_size = sizeof(device);
  device.abi_version = SLLM_HIP_ABI_VERSION;
  Error error;
  if (!expect(sllm_device_query(0U, &device, &error.sink), SLLM_STATUS_OK,
              "device query", error) ||
      std::string(device.gcn_arch_name) != SLLM_TEST_EXPECTED_TARGET) {
    return 2;
  }

  sllm_context_t *context = nullptr;
  sllm_queue_t *eager_queue = nullptr;
  sllm_queue_t *graph_queue = nullptr;
  std::array<sllm_buffer_t *, 4U> buffers{};
  bool passed = true;
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  std::strncpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
               sizeof(context_info.expected_gcn_arch_name) - 1U);
  passed &= expect(sllm_context_create(&context_info, &context, &error.sink),
                   SLLM_STATUS_OK, "context create", error);
  if (passed) {
    sllm_context_probe_result_t probe{};
    probe.struct_size = sizeof(probe);
    probe.abi_version = SLLM_HIP_ABI_VERSION;
    passed &= expect(sllm_context_probe(context, &probe, &error.sink),
                     SLLM_STATUS_OK, "context probe", error) &&
              probe.context_present == 1U && probe.hip_available == 1U;
  }
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  if (passed) {
    passed &= expect(
        sllm_queue_create(context, &queue_info, &eager_queue, &error.sink),
        SLLM_STATUS_OK, "eager queue create", error);
    passed &= expect(
        sllm_queue_set_completion_mode(
            eager_queue, SLLM_QUEUE_COMPLETION_MODE_PROFILED, &error.sink),
        SLLM_STATUS_OK, "eager completion mode", error);
  }
  const uint64_t kv_bytes =
      static_cast<uint64_t>(kCapacity) * kKvHeads * kHeadDim * sizeof(uint16_t);
  const uint64_t query_bytes = static_cast<uint64_t>(kCapacity) * kQueryHeads *
                               kHeadDim * sizeof(uint16_t);
  const std::vector<uint16_t> keys = make_kv(true);
  const std::vector<uint16_t> values = make_kv(false);
  const std::vector<uint16_t> queries = make_queries();
  if (passed) {
    passed &= create_buffer(context, kv_bytes, &buffers[0]);
    passed &= create_buffer(context, kv_bytes, &buffers[1]);
    passed &= create_buffer(context, query_bytes, &buffers[2]);
    passed &= create_buffer(context, sizeof(sllm_decode_control::ControlV1),
                            &buffers[3]);
  }
  if (passed) {
    passed &=
        upload(eager_queue, buffers[0], keys.data(), kv_bytes, "key upload");
    passed &= upload(eager_queue, buffers[1], values.data(), kv_bytes,
                     "value upload");
    passed &= upload(eager_queue, buffers[2], queries.data(), query_bytes,
                     "query upload");
  }
  if (passed) {
    for (uint32_t m = 1U; m <= 5U && passed; ++m) {
      passed = run_graph_case(context, eager_queue, buffers[0], buffers[1],
                              buffers[2], buffers[3], keys, values, queries, m,
                              UINT64_C(0x871000) + kPrefix);
    }
  }
  if (passed) {
    passed = run_cancel_case(context, eager_queue, buffers[0], buffers[1]);
  }
  if (passed) {
    passed =
        run_deferred_append_case(context, eager_queue, buffers[0], buffers[1]);
  }
  for (sllm_buffer_t *&buffer : buffers) {
    passed = release_buffer(&buffer) && passed;
  }
  passed = release_span(nullptr) && passed;
  if (graph_queue != nullptr) {
    passed = release_queue(&graph_queue) && passed;
  }
  if (eager_queue != nullptr) {
    passed = release_queue(&eager_queue) && passed;
  }
  if (context != nullptr) {
    passed = release_context(&context) && passed;
  }
  if (passed) {
    std::cout << "phase87_stage10_paged_graph_public_gpu_test: PASS target="
              << SLLM_TEST_EXPECTED_TARGET
              << " cases=5 graph_replay=m1..m5 cancel_rollback=1 "
                 "deferred_append_fence=1 "
                 "oracle=independent-mxfp8 fallback=0 cleanup=0\n";
  }
  return passed ? 0 : 1;
}
