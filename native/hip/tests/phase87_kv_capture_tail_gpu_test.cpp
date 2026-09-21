#include "decode_control_kernel_internal.hpp"
#include "decode_graph_capture_internal.hpp"
#include "sllm/hip.h"

#include <hip/hip_runtime.h>

#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1201"
#endif
#ifndef SLLM_TEST_EXPECTED_UUID
#define SLLM_TEST_EXPECTED_UUID "GPU-a8e9ddefa2d60f55"
#endif

namespace {

constexpr uint64_t kCapacity = 22U;
constexpr uint64_t kPublished = 21U;
constexpr uint32_t kStaticRows = 2U;
constexpr uint32_t kHeads = 4U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint64_t kElementsPerRow = static_cast<uint64_t>(kHeads) * kHeadDim;

struct Error final {
  char message[512]{};
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

sllm_tensor_binding_t binding(const sllm_buffer_t *const buffer,
                              const uint64_t rows) {
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.dtype = SLLM_TENSOR_DTYPE_BF16;
  result.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  result.rank = 3U;
  result.shape[0] = rows;
  result.shape[1] = kHeads;
  result.shape[2] = kHeadDim;
  result.stride_elements[0] = kElementsPerRow;
  result.stride_elements[1] = kHeadDim;
  result.stride_elements[2] = 1U;
  return result;
}

sllm_tensor_binding_t byte_binding(const sllm_buffer_t *const buffer,
                                   const uint64_t bytes) {
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.dtype = SLLM_TENSOR_DTYPE_U8;
  result.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  result.rank = 1U;
  result.shape[0] = bytes;
  result.stride_elements[0] = 1U;
  return result;
}

bool create_buffer(const sllm_context_t *const context, const uint64_t bytes,
                   sllm_buffer_t **const output) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = bytes;
  Error error;
  return expect_status(sllm_buffer_create(context, &info, output, &error.sink),
                       SLLM_STATUS_OK, "buffer_create", error);
}

bool wait_and_release(sllm_completion_t **const completion,
                      const char *const operation) {
  if (completion == nullptr || *completion == nullptr) {
    return false;
  }
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  const bool waited =
      expect_status(
          sllm_completion_wait(*completion, UINT32_MAX, &result, &error.sink),
          SLLM_STATUS_OK, operation, error) &&
      result.state == SLLM_COMPLETION_STATE_SUCCESS;
  return expect_status(sllm_completion_release(completion, &error.sink),
                       SLLM_STATUS_OK, "completion_release", error) &&
         *completion == nullptr && waited;
}

bool upload(const sllm_queue_t *const queue, const sllm_buffer_t *const buffer,
            void *const source, const uint64_t bytes,
            const char *const operation) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = source;
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  return expect_status(sllm_buffer_copy_h2d(queue, buffer, &transfer,
                                            &completion, &error.sink),
                       SLLM_STATUS_OK, operation, error) &&
         wait_and_release(&completion, operation);
}

bool query_state(const sllm_kv_state_t *const state, const uint32_t memory_kind,
                 const uint64_t expected_length) {
  sllm_kv_view_info_t view{};
  view.struct_size = sizeof(view);
  view.abi_version = SLLM_HIP_ABI_VERSION;
  view.info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
  Error error;
  if (!expect_status(sllm_kv_state_query(state, &view, &error.sink),
                     SLLM_STATUS_OK, "kv_state_query", error)) {
    return false;
  }
  if (view.memory_kind != memory_kind || view.capacity_tokens != kCapacity ||
      view.observed_length != expected_length ||
      view.mapped_token_capacity < expected_length ||
      view.mapped_token_capacity > kCapacity) {
    std::cerr << "unexpected KV view memory_kind=" << view.memory_kind
              << " capacity=" << view.capacity_tokens
              << " observed=" << view.observed_length
              << " mapped=" << view.mapped_token_capacity << '\n';
    return false;
  }
  return true;
}

bool run_memory_kind(const sllm_context_t *const context,
                     const sllm_queue_t *const eager_queue,
                     const sllm_queue_t *const graph_queue,
                     const sllm_buffer_t *const key_buffer,
                     const sllm_buffer_t *const value_buffer,
                     const sllm_buffer_t *const control_buffer,
                     const uint32_t memory_kind, const uint64_t session_id) {
  Error error;
  sllm_kv_state_t *state = nullptr;
  sllm_graph_span_t *span = nullptr;
  sllm_completion_t *completion = nullptr;
  bool ok = true;

  sllm_kv_state_create_info_t create{};
  create.struct_size = sizeof(create);
  create.abi_version = SLLM_HIP_ABI_VERSION;
  create.session_id = session_id;
  create.layer_id = static_cast<uint32_t>(session_id);
  create.capacity_tokens = kCapacity;
  create.head_count = kHeads;
  create.head_dim = kHeadDim;
  create.memory_kind = memory_kind;
  create.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
  ok &=
      expect_status(sllm_kv_state_create(context, &create, &state, &error.sink),
                    SLLM_STATUS_OK, "kv_state_create", error) &&
      state != nullptr;

  if (ok) {
    sllm_kv_append_desc_t warm{};
    warm.struct_size = sizeof(warm);
    warm.abi_version = SLLM_HIP_ABI_VERSION;
    warm.append_version = SLLM_HIP_KV_STATE_VERSION;
    warm.expected_length = 0U;
    warm.start_position = 0U;
    warm.key_input = binding(key_buffer, kPublished);
    warm.value_input = binding(value_buffer, kPublished);
    sllm_kv_append_info_t info{};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
    ok &= expect_status(sllm_kv_state_append(state, eager_queue, &warm,
                                             &completion, &info, &error.sink),
                        SLLM_STATUS_OK, "warm_kv_append", error) &&
          completion != nullptr && info.start_position == 0U &&
          info.token_count == kPublished && info.end_position == kPublished &&
          wait_and_release(&completion, "warm_kv_wait") &&
          query_state(state, memory_kind, kPublished);
  }

  if (ok) {
    const auto control_binding =
        byte_binding(control_buffer, sizeof(sllm_decode_control::ControlV1));
    ok &= expect_status(sllm_graph_span_begin_capture(context, graph_queue,
                                                      &control_binding, &span,
                                                      &error.sink),
                        SLLM_STATUS_OK, "graph_begin", error) &&
          span != nullptr;
  }
  if (ok) {
    sllm_graph_span_decode_command_desc_t phase{};
    phase.struct_size = sizeof(phase);
    phase.abi_version = SLLM_HIP_ABI_VERSION;
    phase.info_version = SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_VERSION;
    phase.opcode = SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_BEGIN_PHASE;
    phase.phase_kind = sllm_decode_control::kPhaseTarget;
    phase.phase_index = 0U;
    phase.rows = kStaticRows;
    ok &=
        expect_status(sllm_graph_span_decode_command(span, &phase, &error.sink),
                      SLLM_STATUS_OK, "graph_begin_phase", error);
  }
  if (ok) {
    sllm_kv_append_desc_t tail{};
    tail.struct_size = sizeof(tail);
    tail.abi_version = SLLM_HIP_ABI_VERSION;
    tail.append_version = SLLM_HIP_KV_STATE_VERSION;
    tail.expected_length = kPublished;
    tail.start_position = kPublished;
    tail.key_input = binding(key_buffer, kStaticRows);
    tail.value_input = binding(value_buffer, kStaticRows);
    sllm_kv_append_info_t info{};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
    ok &= expect_status(sllm_kv_state_append(state, graph_queue, &tail,
                                             &completion, &info, &error.sink),
                        SLLM_STATUS_OK, "captured_tail_kv_append", error) &&
          completion != nullptr && info.start_position == kPublished &&
          info.token_count == kStaticRows &&
          info.end_position == kPublished + kStaticRows &&
          info.commit_allowed == 1U;
    if (ok) {
      ok &= expect_status(
                sllm_graph_span_capture_marker(span, &completion, &error.sink),
                SLLM_STATUS_OK, "captured_tail_marker", error) &&
            completion == nullptr;
    }
  }
  if (ok && span != nullptr) {
    sllm_graph_span_capture_info_t capture_info{};
    capture_info.struct_size = sizeof(capture_info);
    capture_info.abi_version = SLLM_HIP_ABI_VERSION;
    capture_info.info_version = SLLM_HIP_GRAPH_SPAN_CAPTURE_INFO_VERSION;
    ok &= expect_status(
              sllm_graph_span_end_capture(span, &capture_info, &error.sink),
              SLLM_STATUS_OK, "graph_end", error) &&
          capture_info.kernel_node_count >= 2U;
  }
  if (span != nullptr) {
    if (ok) {
      ok &= expect_status(sllm_graph_span_release(&span, &error.sink),
                          SLLM_STATUS_OK, "graph_release", error) &&
            span == nullptr;
    } else {
      (void)sllm_graph_span_abort_capture(&span, &error.sink);
    }
  }
  if (completion != nullptr) {
    (void)sllm_kv_state_append_cancel(state, completion, &error.sink);
    (void)sllm_completion_release(&completion, &error.sink);
  }
  if (state != nullptr) {
    ok &= query_state(state, memory_kind, kPublished);
    ok &= expect_status(sllm_kv_state_release(&state, &error.sink),
                        SLLM_STATUS_OK, "kv_state_release", error) &&
          state == nullptr;
  }
  std::cout << "kv_capture_tail memory_kind=" << memory_kind
            << " status=" << (ok ? "PASS" : "FAIL") << '\n';
  return ok;
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
  if (!expect_status(sllm_device_query(0U, &device, &error.sink),
                     SLLM_STATUS_OK, "device_query", error) ||
      std::string(device.gcn_arch_name) != SLLM_TEST_EXPECTED_TARGET) {
    return 2;
  }

  sllm_context_t *context = nullptr;
  sllm_queue_t *eager_queue = nullptr;
  sllm_queue_t *graph_queue = nullptr;
  sllm_buffer_t *key_buffer = nullptr;
  sllm_buffer_t *value_buffer = nullptr;
  sllm_buffer_t *control_buffer = nullptr;
  bool ok = true;
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  std::strncpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
               sizeof(context_info.expected_gcn_arch_name) - 1U);
  ok &= expect_status(sllm_context_create(&context_info, &context, &error.sink),
                      SLLM_STATUS_OK, "context_create", error);
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  if (ok) {
    ok &= expect_status(
        sllm_queue_create(context, &queue_info, &eager_queue, &error.sink),
        SLLM_STATUS_OK, "eager_queue_create", error);
    ok &= expect_status(
        sllm_queue_create(context, &queue_info, &graph_queue, &error.sink),
        SLLM_STATUS_OK, "graph_queue_create", error);
    ok &= expect_status(
        sllm_queue_set_completion_mode(
            eager_queue, SLLM_QUEUE_COMPLETION_MODE_PROFILED, &error.sink),
        SLLM_STATUS_OK, "eager_profiled", error);
    ok &= expect_status(
        sllm_queue_set_completion_mode(
            graph_queue, SLLM_QUEUE_COMPLETION_MODE_DEFERRED, &error.sink),
        SLLM_STATUS_OK, "graph_deferred", error);
  }
  const uint64_t input_bytes = kPublished * kElementsPerRow * sizeof(uint16_t);
  if (ok) {
    ok &= create_buffer(context, input_bytes, &key_buffer);
    ok &= create_buffer(context, input_bytes, &value_buffer);
    ok &= create_buffer(context, sizeof(sllm_decode_control::ControlV1),
                        &control_buffer);
  }
  std::vector<uint16_t> key(input_bytes / sizeof(uint16_t));
  std::vector<uint16_t> value(key.size());
  for (std::size_t index = 0U; index < key.size(); ++index) {
    key[index] = static_cast<uint16_t>(UINT16_C(0x3d00) + index % 251U);
    value[index] = static_cast<uint16_t>(UINT16_C(0x3e00) + index % 239U);
  }
  sllm_decode_control::ControlV1 control{};
  control.version = sllm_decode_control::kVersion;
  control.status = static_cast<uint32_t>(sllm_decode_control::Status::Ok);
  control.mode = sllm_decode_control::kModeMtp;
  control.width = 1U;
  control.active_width = 0U;
  control.model_position = kPublished;
  control.capacity = kCapacity;
  control.output_limit = 1U;
  control.pending_token = 1U;
  control.stop_row = sllm_decode_control::kNoStop;
  control.hidden_row = sllm_decode_control::kNoStop;
  if (ok) {
    ok &=
        upload(eager_queue, key_buffer, key.data(), input_bytes, "key_upload");
    ok &= upload(eager_queue, value_buffer, value.data(), input_bytes,
                 "value_upload");
    ok &= upload(eager_queue, control_buffer, &control, sizeof(control),
                 "control_upload");
  }
  if (ok) {
    ok &= run_memory_kind(context, eager_queue, graph_queue, key_buffer,
                          value_buffer, control_buffer,
                          SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT, 8701U);
    ok &= run_memory_kind(context, eager_queue, graph_queue, key_buffer,
                          value_buffer, control_buffer,
                          SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS, 8702U);
  }

  for (sllm_buffer_t **buffer : {&control_buffer, &value_buffer, &key_buffer}) {
    if (*buffer != nullptr) {
      ok &= expect_status(sllm_buffer_release(buffer, &error.sink),
                          SLLM_STATUS_OK, "buffer_release", error);
    }
  }
  if (graph_queue != nullptr) {
    ok &= expect_status(sllm_queue_release(&graph_queue, &error.sink),
                        SLLM_STATUS_OK, "graph_queue_release", error);
  }
  if (eager_queue != nullptr) {
    ok &= expect_status(sllm_queue_release(&eager_queue, &error.sink),
                        SLLM_STATUS_OK, "eager_queue_release", error);
  }
  if (context != nullptr) {
    ok &= expect_status(sllm_context_release(&context, &error.sink),
                        SLLM_STATUS_OK, "context_release", error);
  }
  std::cout << "phase87_kv_capture_tail_gpu_test status="
            << (ok ? "PASS" : "FAIL") << " target=" << SLLM_TEST_EXPECTED_TARGET
            << '\n';
  return ok ? 0 : 1;
}
