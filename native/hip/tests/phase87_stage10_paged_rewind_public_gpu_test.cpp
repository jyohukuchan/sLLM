// Phase 87 Stage 10: public Paged KV rewind and tail reuse correctness.
//
// The test exercises the Paged state at a partial block tail and at the
// 128-token block boundary. Rewind is a host metadata operation; the following
// append must reuse the shortened tail without exposing the discarded token.
#include "sllm/hip.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <iostream>
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

bool append(const sllm_kv_state_t *const state, const sllm_queue_t *const queue,
            const sllm_buffer_t *const key, const sllm_buffer_t *const value,
            const uint64_t count, const uint64_t start,
            sllm_completion_t **const completion, const bool wait,
            const char *const operation) {
  const auto descriptor = append_descriptor(key, value, count, start);
  sllm_kv_append_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
  Error error;
  *completion = nullptr;
  if (!expect(sllm_kv_state_append(state, queue, &descriptor, completion, &info,
                                   &error.sink),
              SLLM_STATUS_OK, operation, error) ||
      *completion == nullptr || info.backend != SLLM_BACKEND_HIP ||
      info.fallback_allowed != 0U || info.fallback_used != 0U ||
      info.start_position != start || info.token_count != count ||
      info.end_position != start + count ||
      info.kernel_id != SLLM_HIP_KV_KERNEL_ID_BF16_TO_PAGED_MXFP8_E4_V1)
    return false;
  return !wait || wait_release(completion, operation);
}

bool query_paged(const sllm_kv_state_t *const state, const uint64_t length,
                 const uint64_t generation, const char *const operation) {
  sllm_kv_paged_view_info_t view{};
  view.struct_size = sizeof(view);
  view.abi_version = SLLM_HIP_ABI_VERSION;
  view.info_version = SLLM_HIP_KV_PAGED_VIEW_INFO_VERSION;
  Error error;
  const bool queried =
      expect(sllm_kv_state_query_paged(state, &view, &error.sink),
             SLLM_STATUS_OK, operation, error);
  if (queried &&
      (view.observed_length != length || view.generation != generation))
    std::cerr << operation << " observed=" << view.observed_length
              << " generation=" << view.generation << " expected=" << length
              << "/" << generation << '\n';
  return queried && view.observed_length == length &&
         view.generation == generation;
}

bool run_case(const sllm_context_t *const context,
              const sllm_queue_t *const queue, const uint64_t initial_length,
              const uint64_t rewind_length, const uint64_t session,
              const std::vector<uint16_t> &keys,
              const std::vector<uint16_t> &values,
              const std::vector<uint16_t> &queries) {
  sllm_kv_state_t *paged = nullptr;
  std::array<sllm_buffer_t *, 5U> buffers{};
  sllm_completion_t *paged_append = nullptr;
  sllm_completion_t *paged_reappend = nullptr;
  sllm_completion_t *cancelled = nullptr;
  bool passed = true;
  Error error;
  const uint64_t input_bytes =
      static_cast<uint64_t>(keys.size()) * sizeof(uint16_t);
  const uint64_t query_bytes =
      static_cast<uint64_t>(queries.size()) * sizeof(uint16_t);
  const uint64_t output_bytes =
      static_cast<uint64_t>(kQueryHeads) * kHeadDim * sizeof(uint16_t);

  const auto cleanup = [&]() {
    bool result = true;
    if (cancelled != nullptr) {
      (void)sllm_kv_state_append_cancel(paged, cancelled, &error.sink);
      result = wait_release(&cancelled, "cancel cleanup") && result;
    }
    if (paged_reappend != nullptr)
      result =
          wait_release(&paged_reappend, "paged reappend cleanup") && result;
    if (paged_append != nullptr)
      result = wait_release(&paged_append, "paged append cleanup") && result;
    result = release_state(&paged) && result;
    for (sllm_buffer_t *&buffer : buffers)
      result = release_buffer(&buffer) && result;
    return result;
  };

  const auto paged_info = paged_create(session, static_cast<uint32_t>(session));
  passed &= expect(
      sllm_kv_state_create_paged(context, &paged_info, &paged, &error.sink),
      SLLM_STATUS_OK, "rewind paged create", error);
  passed &= create_buffer(context, input_bytes, &buffers[0]);
  passed &= create_buffer(context, input_bytes, &buffers[1]);
  passed &= create_buffer(context, query_bytes, &buffers[2]);
  passed &= create_buffer(context, output_bytes, &buffers[3]);
  passed &= create_buffer(context, output_bytes, &buffers[4]);
  if (passed) {
    passed &= upload(queue, buffers[0], keys.data(), input_bytes, "key upload");
    passed &=
        upload(queue, buffers[1], values.data(), input_bytes, "value upload");
    passed &=
        upload(queue, buffers[2], queries.data(), query_bytes, "query upload");
  }
  if (passed) {
    passed &= append(paged, queue, buffers[0], buffers[1], initial_length, 0U,
                     &paged_append, true, "initial paged append") &&
              query_paged(paged, initial_length, 1U, "initial paged query");
  }
  if (!passed)
    std::cerr << "rewind case failed during initial append/query\n";
  if (passed) {
    passed &= expect(sllm_kv_state_rewind_last(paged, initial_length,
                                               rewind_length, &error.sink),
                     SLLM_STATUS_OK, "paged rewind", error) &&
              query_paged(paged, rewind_length, 2U, "rewound paged query");
  }
  if (!passed)
    std::cerr << "rewind case failed during rewind/query\n";
  if (passed) {
    passed &=
        append(paged, queue, buffers[0], buffers[1], 1U, rewind_length,
               &paged_reappend, true, "paged tail reappend") &&
        query_paged(paged, rewind_length + 1U, 3U, "reappended paged query");
  }
  if (!passed)
    std::cerr << "rewind case failed during reappend/query\n";

  std::vector<uint16_t> paged_output(kQueryHeads * kHeadDim);
  if (passed) {
    const uint64_t query_row_bytes =
        static_cast<uint64_t>(kQueryHeads) * kHeadDim * sizeof(uint16_t);
    const uint64_t query_offset = rewind_length * query_row_bytes;
    const uint64_t expected_length = rewind_length + 1U;
    const uint64_t shape[] = {1U, kQueryHeads, kHeadDim};
    sllm_causal_attention_desc_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.op_version = SLLM_HIP_CAUSAL_ATTENTION_VERSION;
    descriptor.start_position = rewind_length;
    descriptor.expected_kv_length = expected_length;
    descriptor.kv_state = paged;
    descriptor.query = binding(buffers[2], SLLM_TENSOR_DTYPE_BF16, 3U, shape);
    descriptor.output = binding(buffers[3], SLLM_TENSOR_DTYPE_BF16, 3U, shape);
    descriptor.query.byte_offset = query_offset;
    sllm_causal_attention_dispatch_info_t dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    dispatch.info_version = SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION;
    sllm_completion_t *completion = nullptr;
    passed = expect(sllm_causal_attention_execute(context, queue, &descriptor,
                                                  &completion, &dispatch,
                                                  &error.sink),
                    SLLM_STATUS_OK, "paged rewind attention", error) &&
             completion != nullptr && dispatch.backend == SLLM_BACKEND_HIP &&
             dispatch.fallback_allowed == 0U && dispatch.fallback_used == 0U &&
             wait_release(&completion, "paged rewind attention") &&
             download(queue, buffers[3], paged_output.data(), output_bytes,
                      "paged rewind output");
  }
  if (passed) {
    passed &=
        append(paged, queue, buffers[0], buffers[1], 1U, rewind_length + 1U,
               &cancelled, false, "cancelled tail append") &&
        expect(sllm_kv_state_append_cancel(paged, cancelled, &error.sink),
               SLLM_STATUS_OK, "cancelled tail revoke", error) &&
        wait_release(&cancelled, "cancelled tail wait") &&
        query_paged(paged, rewind_length + 1U, 3U, "post-cancel paged query");
  }
  const bool cleaned = cleanup();
  std::cout << "phase87 rewind initial=" << initial_length
            << " rewind=" << rewind_length
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
    passed = run_case(context, queue, 49U, 48U, UINT64_C(0x871049), keys,
                      values, queries);
  if (passed)
    passed = run_case(context, queue, 129U, 128U, UINT64_C(0x871129), keys,
                      values, queries);
  passed = release_queue(&queue) && passed;
  passed = release_context(&context) && passed;
  if (passed)
    std::cout << "phase87_stage10_paged_rewind_public_gpu_test: PASS target="
              << SLLM_TEST_EXPECTED_TARGET
              << " cases=49_to_48,129_to_128 cancel=1 bitwise_control=1 "
                 "fallback=0 cleanup=0\n";
  return passed ? 0 : 1;
}
