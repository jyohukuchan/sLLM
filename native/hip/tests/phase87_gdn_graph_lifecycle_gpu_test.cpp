#include "decode_control_kernel_internal.hpp"
#include "decode_graph_capture_internal.hpp"
#include "sllm/hip.h"
#include <hip/hip_runtime.h>

#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1030"
#endif
#ifndef SLLM_TEST_EXPECTED_UUID
#define SLLM_TEST_EXPECTED_UUID "GPU-76a08c022586fed6"
#endif

extern "C" sllm_status_t
sllm_linear_attention_state_release(sllm_linear_attention_state_t **state,
                                    sllm_error_sink_t *error_sink) noexcept;

namespace {
constexpr uint32_t kQkHeads = 16U;
constexpr uint32_t kValueHeads = 48U;
constexpr uint32_t kHeadDim = 128U;
constexpr uint32_t kConvKernel = 4U;
constexpr uint32_t kQkvWidth = (2U * kQkHeads + kValueHeads) * kHeadDim;
constexpr uint32_t kOutputWidth = kValueHeads * kHeadDim;
constexpr uint64_t kCapacity = 16U;

struct Error {
  char message[512]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};
};

bool expect_status(const sllm_status_t actual, const sllm_status_t expected,
                   const char *where, const Error &error) {
  if (actual == expected)
    return true;
  std::cerr << where << " status=" << actual << " expected=" << expected
            << " message=" << error.message << '\n';
  return false;
}

std::string uuid_text(const hipUUID &uuid) {
  std::string value(uuid.bytes, sizeof(uuid.bytes));
  for (char &ch : value) {
    const unsigned char byte = static_cast<unsigned char>(ch);
    if (!std::isxdigit(byte))
      return {};
    ch = static_cast<char>(std::tolower(byte));
  }
  return "GPU-" + value;
}

uint16_t bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

sllm_tensor_binding_t binding(const sllm_buffer_t *buffer, uint32_t dtype,
                              std::initializer_list<uint64_t> dimensions) {
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.dtype = dtype;
  result.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  result.rank = static_cast<uint32_t>(dimensions.size());
  uint32_t index = 0U;
  for (const uint64_t dimension : dimensions) {
    result.shape[index] = dimension;
    ++index;
  }
  uint64_t stride = 1U;
  for (uint32_t backwards = 0U; backwards != result.rank; ++backwards) {
    const uint32_t current = result.rank - 1U - backwards;
    result.stride_elements[current] = stride;
    stride *= result.shape[current];
  }
  return result;
}

bool create_buffer(const sllm_context_t *context, uint64_t bytes,
                   sllm_buffer_t **buffer) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = bytes;
  Error error;
  return expect_status(sllm_buffer_create(context, &info, buffer, &error.sink),
                       SLLM_STATUS_OK, "buffer_create", error);
}

bool release_buffer(sllm_buffer_t **buffer, const char *where) {
  if (*buffer == nullptr)
    return true;
  Error error;
  return expect_status(sllm_buffer_release(buffer, &error.sink), SLLM_STATUS_OK,
                       where, error);
}

bool wait_release(sllm_completion_t **completion, const char *where) {
  if (*completion == nullptr)
    return false;
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  bool ok = expect_status(
                sllm_completion_wait(*completion, 5000U, &result, &error.sink),
                SLLM_STATUS_OK, where, error) &&
            result.state == SLLM_COMPLETION_STATE_SUCCESS;
  ok = expect_status(sllm_completion_release(completion, &error.sink),
                     SLLM_STATUS_OK, "completion_release", error) &&
       ok;
  return ok;
}

bool upload(const sllm_queue_t *queue, const sllm_buffer_t *buffer, void *data,
            uint64_t bytes, const char *where) {
  sllm_transfer_desc_t desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = SLLM_HIP_ABI_VERSION;
  desc.host_pointer = data;
  desc.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  return expect_status(sllm_buffer_copy_h2d(queue, buffer, &desc, &completion,
                                            &error.sink),
                       SLLM_STATUS_OK, where, error) &&
         wait_release(&completion, where);
}

bool import_seed(const sllm_linear_attention_state_t *state,
                 const uint32_t plane, void *data, uint64_t bytes,
                 const char *where) {
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
      SLLM_STATUS_OK, where, error);
}

bool export_slot(const sllm_linear_attention_state_t *state,
                 const uint32_t slot, std::vector<uint8_t> *conv,
                 std::vector<uint8_t> *recurrent) {
  const uint32_t conv_plane = slot == 0U
                                  ? SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT0
                                  : SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT1;
  const uint32_t recurrent_plane =
      slot == 0U ? SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT0
                 : SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT1;
  sllm_state_chunk_t conv_chunk{};
  conv_chunk.struct_size = sizeof(conv_chunk);
  conv_chunk.abi_version = SLLM_HIP_ABI_VERSION;
  conv_chunk.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  conv_chunk.plane = conv_plane;
  conv_chunk.byte_length = conv->size();
  conv_chunk.host_pointer = conv->data();
  conv_chunk.host_capacity = conv->size();
  sllm_state_chunk_t recurrent_chunk = conv_chunk;
  recurrent_chunk.plane = recurrent_plane;
  recurrent_chunk.byte_length = recurrent->size();
  recurrent_chunk.host_pointer = recurrent->data();
  recurrent_chunk.host_capacity = recurrent->size();
  Error error;
  return expect_status(sllm_linear_attention_state_export(state, &conv_chunk,
                                                          &error.sink),
                       SLLM_STATUS_OK, "conv_export", error) &&
         expect_status(sllm_linear_attention_state_export(
                           state, &recurrent_chunk, &error.sink),
                       SLLM_STATUS_OK, "recurrent_export", error);
}

bool export_active(const sllm_linear_attention_state_t *state,
                   std::vector<uint8_t> *conv, std::vector<uint8_t> *recurrent,
                   uint32_t *slot) {
  sllm_linear_attention_view_info_t view{};
  view.struct_size = sizeof(view);
  view.abi_version = SLLM_HIP_ABI_VERSION;
  view.info_version = SLLM_HIP_LINEAR_ATTENTION_VIEW_INFO_VERSION;
  Error error;
  if (!expect_status(
          sllm_linear_attention_state_query(state, &view, &error.sink),
          SLLM_STATUS_OK, "state_query", error)) {
    return false;
  }
  *slot = view.active_slot;
  return export_slot(state, *slot, conv, recurrent);
}

sllm_linear_attention_desc_t
make_descriptor(const sllm_linear_attention_state_t *state,
                const std::array<sllm_buffer_t *, 9U> &buffers,
                uint64_t start) {
  sllm_linear_attention_desc_t d{};
  d.struct_size = sizeof(d);
  d.abi_version = SLLM_HIP_ABI_VERSION;
  d.op_version = SLLM_HIP_LINEAR_ATTENTION_VERSION;
  d.start_position = start;
  d.expected_length = start + 1U;
  d.state = state;
  d.qkv = binding(buffers[0], SLLM_TENSOR_DTYPE_BF16, {1U, kQkvWidth});
  d.z = binding(buffers[1], SLLM_TENSOR_DTYPE_BF16, {1U, kOutputWidth});
  d.b_input = binding(buffers[2], SLLM_TENSOR_DTYPE_BF16, {1U, kValueHeads});
  d.a_input = binding(buffers[3], SLLM_TENSOR_DTYPE_BF16, {1U, kValueHeads});
  d.conv_weight =
      binding(buffers[4], SLLM_TENSOR_DTYPE_BF16, {kQkvWidth, 1U, kConvKernel});
  d.a_log = binding(buffers[5], SLLM_TENSOR_DTYPE_F32, {kValueHeads});
  d.dt_bias = binding(buffers[6], SLLM_TENSOR_DTYPE_BF16, {kValueHeads});
  d.norm_weight = binding(buffers[7], SLLM_TENSOR_DTYPE_F32, {kHeadDim});
  d.output = binding(buffers[8], SLLM_TENSOR_DTYPE_BF16, {1U, kOutputWidth});
  return d;
}

bool run_eager(const sllm_context_t *context, const sllm_queue_t *queue,
               sllm_linear_attention_state_t *state,
               const std::array<sllm_buffer_t *, 9U> &buffers,
               const uint32_t count = 2U, const uint64_t start = 0U) {
  for (uint64_t offset = 0U; offset != count; ++offset) {
    const uint64_t position = start + offset;
    sllm_linear_attention_desc_t desc =
        make_descriptor(state, buffers, position);
    sllm_linear_attention_dispatch_info_t info{};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    info.info_version = SLLM_HIP_LINEAR_ATTENTION_DISPATCH_INFO_VERSION;
    sllm_completion_t *completion = nullptr;
    Error error;
    if (!expect_status(sllm_linear_attention_execute(context, queue, &desc,
                                                     &completion, &info,
                                                     &error.sink),
                       SLLM_STATUS_OK, "eager_linear_execute", error) ||
        !wait_release(&completion, "eager_linear_wait")) {
      return false;
    }
  }
  return true;
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
  bool ok = true;
  sllm_context_t *context = nullptr;
  sllm_queue_t *graph_queue = nullptr;
  sllm_queue_t *eager_queue = nullptr;
  std::array<sllm_buffer_t *, 9U> buffers{};
  sllm_buffer_t *control_buffer = nullptr;
  sllm_buffer_t *selector_buffer = nullptr;
  sllm_buffer_t *result_buffer = nullptr;
  sllm_linear_attention_state_t *graph_state = nullptr;
  sllm_linear_attention_state_t *eager_state = nullptr;
  sllm_graph_span_t *span = nullptr;

  sllm_context_create_info_t ci{};
  ci.struct_size = sizeof(ci);
  ci.abi_version = SLLM_HIP_ABI_VERSION;
  ci.expected_gcn_arch_name[0] = SLLM_TEST_EXPECTED_TARGET[0];
  std::strncpy(ci.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
               sizeof(ci.expected_gcn_arch_name) - 1U);
  ok &= expect_status(sllm_context_create(&ci, &context, &error.sink),
                      SLLM_STATUS_OK, "context_create", error);
  sllm_queue_create_info_t qi{};
  qi.struct_size = sizeof(qi);
  qi.abi_version = SLLM_HIP_ABI_VERSION;
  if (ok) {
    ok &= expect_status(
        sllm_queue_create(context, &qi, &graph_queue, &error.sink),
        SLLM_STATUS_OK, "graph_queue_create", error);
    ok &= expect_status(
        sllm_queue_create(context, &qi, &eager_queue, &error.sink),
        SLLM_STATUS_OK, "eager_queue_create", error);
    ok &= expect_status(
        sllm_queue_set_completion_mode(
            graph_queue, SLLM_QUEUE_COMPLETION_MODE_DEFERRED, &error.sink),
        SLLM_STATUS_OK, "graph_deferred", error);
    ok &= expect_status(
        sllm_queue_set_completion_mode(
            eager_queue, SLLM_QUEUE_COMPLETION_MODE_PROFILED, &error.sink),
        SLLM_STATUS_OK, "eager_profiled", error);
  }
  const uint64_t bytes[9] = {2U * kQkvWidth,
                             2U * kOutputWidth,
                             2U * kValueHeads,
                             2U * kValueHeads,
                             static_cast<uint64_t>(kQkvWidth) * kConvKernel *
                                 2U,
                             static_cast<uint64_t>(kValueHeads) * sizeof(float),
                             2U * kValueHeads,
                             static_cast<uint64_t>(kHeadDim) * sizeof(float),
                             2U * kOutputWidth};
  for (std::size_t index = 0U; ok && index != buffers.size(); ++index) {
    ok &= create_buffer(context, bytes[index], &buffers[index]);
  }
  ok &= create_buffer(context, 256U, &control_buffer);
  ok &= create_buffer(context, sizeof(sllm_decode_control::SelectorRecordV1),
                      &selector_buffer);
  ok &= create_buffer(context, 2U * sizeof(sllm_decode_control::ResultV1),
                      &result_buffer);

  std::vector<uint16_t> qkv(kQkvWidth), z(kOutputWidth), b(kValueHeads),
      a(kValueHeads),
      conv_weight(static_cast<std::size_t>(kQkvWidth) * kConvKernel),
      dt_bias(kValueHeads);
  std::vector<float> a_log(kValueHeads), norm_weight(kHeadDim);
  for (std::size_t index = 0U; index != qkv.size(); ++index) {
    qkv[index] = bf16(0.1F + static_cast<float>(index % 17U) * 0.001F);
  }
  for (std::size_t index = 0U; index != z.size(); ++index)
    z[index] = bf16(0.2F);
  for (std::size_t index = 0U; index != b.size(); ++index)
    b[index] = bf16(0.1F);
  for (std::size_t index = 0U; index != a.size(); ++index)
    a[index] = bf16(0.0F);
  for (std::size_t index = 0U; index != conv_weight.size(); ++index)
    conv_weight[index] = bf16(0.01F);
  for (std::size_t index = 0U; index != dt_bias.size(); ++index)
    dt_bias[index] = bf16(0.0F);
  for (float &value : a_log)
    value = -1.0F;
  for (float &value : norm_weight)
    value = 1.0F;
  if (ok) {
    ok &= upload(graph_queue, buffers[0], qkv.data(), bytes[0], "qkv_upload");
    ok &= upload(graph_queue, buffers[1], z.data(), bytes[1], "z_upload");
    ok &= upload(graph_queue, buffers[2], b.data(), bytes[2], "b_upload");
    ok &= upload(graph_queue, buffers[3], a.data(), bytes[3], "a_upload");
    ok &= upload(graph_queue, buffers[4], conv_weight.data(), bytes[4],
                 "conv_upload");
    ok &=
        upload(graph_queue, buffers[5], a_log.data(), bytes[5], "alog_upload");
    ok &=
        upload(graph_queue, buffers[6], dt_bias.data(), bytes[6], "dt_upload");
    ok &= upload(graph_queue, buffers[7], norm_weight.data(), bytes[7],
                 "norm_upload");
  }
  const uint64_t conv_state_bytes =
      static_cast<uint64_t>(kConvKernel - 1U) * kQkvWidth * sizeof(uint16_t);
  const uint64_t recurrent_state_bytes =
      static_cast<uint64_t>(kValueHeads) * kHeadDim * kHeadDim * sizeof(float);
  std::vector<uint16_t> conv_seed(conv_state_bytes / sizeof(uint16_t),
                                  bf16(0.0F));
  std::vector<float> recurrent_seed(recurrent_state_bytes / sizeof(float),
                                    0.0F);
  std::vector<uint8_t> eager_two_conv(conv_state_bytes),
      eager_two_recurrent(recurrent_state_bytes);
  uint32_t eager_two_slot = 0U;
  sllm_linear_attention_state_create_info_t si{};
  si.struct_size = sizeof(si);
  si.abi_version = SLLM_HIP_ABI_VERSION;
  si.session_id = 901U;
  si.capacity_tokens = kCapacity;
  si.qk_heads = kQkHeads;
  si.value_heads = kValueHeads;
  si.head_dim = kHeadDim;
  si.conv_kernel_size = kConvKernel;
  if (ok) {
    ok &= expect_status(sllm_linear_attention_state_create(
                            context, &si, &graph_state, &error.sink),
                        SLLM_STATUS_OK, "graph_state_create", error);
    si.session_id = 902U;
    ok &= expect_status(sllm_linear_attention_state_create(
                            context, &si, &eager_state, &error.sink),
                        SLLM_STATUS_OK, "eager_state_create", error);
    ok &= import_seed(graph_state, SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT0,
                      conv_seed.data(), conv_state_bytes, "graph_conv_seed");
    ok &= import_seed(graph_state, SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT0,
                      recurrent_seed.data(), recurrent_state_bytes,
                      "graph_recurrent_seed");
    ok &= import_seed(eager_state, SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT0,
                      conv_seed.data(), conv_state_bytes, "eager_conv_seed");
    ok &= import_seed(eager_state, SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT0,
                      recurrent_seed.data(), recurrent_state_bytes,
                      "eager_recurrent_seed");
    /* Warm the graph state's request-local scratch before HIP stream capture;
     * the capture path must contain kernels and state transitions only. */
    ok &= run_eager(context, eager_queue, graph_state, buffers, 1U);
    if (ok) {
      ok &= expect_status(sllm_linear_attention_state_rewind_last(
                              graph_state, 1U, 0U, &error.sink),
                          SLLM_STATUS_OK, "graph_state_scratch_rewind1", error);
    }
    ok &= run_eager(context, eager_queue, eager_state, buffers);
    if (ok) {
      ok &= export_active(eager_state, &eager_two_conv, &eager_two_recurrent,
                          &eager_two_slot) &&
            eager_two_slot == 0U;
    }
    ok &= run_eager(context, eager_queue, eager_state, buffers, 1U, 2U);
  }
  sllm_decode_control::ControlV1 control_host{};
  control_host.version = sllm_decode_control::kVersion;
  control_host.status = static_cast<uint32_t>(sllm_decode_control::Status::Ok);
  control_host.mode = sllm_decode_control::kModeTargetOnly;
  control_host.width = 1U;
  control_host.capacity = kCapacity;
  // Three valid generations are followed by one budget-halted one-ahead
  // replay. The fourth replay must not modify either parity state plane.
  control_host.output_limit = 3U;
  control_host.seed = 7U;
  sllm_decode_control::SelectorRecordV1 selector{1, 0U, 0.0F, 0U};
  if (ok) {
    ok &= upload(graph_queue, control_buffer, &control_host,
                 sizeof(control_host), "control_upload");
    ok &= upload(graph_queue, selector_buffer, &selector, sizeof(selector),
                 "selector_upload");
  }
  if (ok) {
    sllm_tensor_binding_t control_binding =
        binding(control_buffer, SLLM_TENSOR_DTYPE_U8, {144U});
    ok &= expect_status(sllm_graph_span_begin_capture(context, graph_queue,
                                                      &control_binding, &span,
                                                      &error.sink),
                        SLLM_STATUS_OK, "graph_begin", error);
    ok &= expect_status(sllm_graph_span_bind_linear_state(span, graph_state,
                                                          nullptr, nullptr, 1U,
                                                          0U, &error.sink),
                        SLLM_STATUS_OK, "graph_bind_gdn_rows0", error);
  }
  if (ok) {
    sllm_graph_span_decode_command_desc_t phase{};
    phase.struct_size = sizeof(phase);
    phase.abi_version = SLLM_HIP_ABI_VERSION;
    phase.info_version = SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_VERSION;
    phase.opcode = SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_BEGIN_PHASE;
    phase.phase_kind = sllm_decode_control::kPhaseTarget;
    phase.rows = 1U;
    ok &=
        expect_status(sllm_graph_span_decode_command(span, &phase, &error.sink),
                      SLLM_STATUS_OK, "graph_phase", error);
    sllm_linear_attention_desc_t desc =
        make_descriptor(graph_state, buffers, 0U);
    sllm_linear_attention_dispatch_info_t dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    dispatch.info_version = SLLM_HIP_LINEAR_ATTENTION_DISPATCH_INFO_VERSION;
    sllm_completion_t *marker = nullptr;
    ok &= expect_status(sllm_linear_attention_execute(context, graph_queue,
                                                      &desc, &marker, &dispatch,
                                                      &error.sink),
                        SLLM_STATUS_OK, "graph_linear_capture", error);
    ok &= expect_status(
        sllm_graph_span_capture_marker(span, &marker, &error.sink),
        SLLM_STATUS_OK, "graph_linear_marker", error);
    sllm_graph_span_decode_command_desc_t commit{};
    commit.struct_size = sizeof(commit);
    commit.abi_version = SLLM_HIP_ABI_VERSION;
    commit.info_version = SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_VERSION;
    commit.opcode = SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_COMMIT_STEP;
    commit.input_kind = 0U;
    commit.vocabulary_size = 65536U;
    commit.input0 = binding(selector_buffer, SLLM_TENSOR_DTYPE_U8, {16U});
    commit.output = binding(result_buffer, SLLM_TENSOR_DTYPE_U8, {384U});
    ok &= expect_status(
        sllm_graph_span_decode_command(span, &commit, &error.sink),
        SLLM_STATUS_OK, "graph_commit_selector", error);
    ok &= expect_status(
        sllm_graph_span_select_linear_state(span, graph_state, 1U, &error.sink),
        SLLM_STATUS_OK, "graph_select_anchor", error);
    sllm_graph_span_capture_info_t info{};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    info.info_version = SLLM_HIP_GRAPH_SPAN_CAPTURE_INFO_VERSION;
    ok &= expect_status(sllm_graph_span_end_capture(span, &info, &error.sink),
                        SLLM_STATUS_OK, "graph_end", error);
  }
  std::array<sllm_completion_t *, 4U> replays{};
  if (ok) {
    for (uint32_t index = 0U; index != replays.size(); ++index) {
      ok &= expect_status(
          sllm_graph_span_execute(span, &replays[index], &error.sink),
          SLLM_STATUS_OK, "graph_replay", error);
    }
    sllm_completion_t *fence = nullptr;
    ok &= expect_status(sllm_queue_fence(graph_queue, &fence, &error.sink),
                        SLLM_STATUS_OK, "graph_fence", error);
    sllm_completion_result_t result{};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_ABI_VERSION;
    ok &= expect_status(
        sllm_completion_wait(fence, UINT32_MAX, &result, &error.sink),
        SLLM_STATUS_OK, "graph_fence_wait", error);
    for (sllm_completion_t *&completion : replays) {
      if (completion != nullptr) {
        sllm_completion_result_t replay_result{};
        replay_result.struct_size = sizeof(replay_result);
        replay_result.abi_version = SLLM_HIP_ABI_VERSION;
        ok &= expect_status(sllm_completion_finalize_after(
                                completion, fence, &replay_result, &error.sink),
                            SLLM_STATUS_OK, "graph_replay_finalize", error);
        ok &= expect_status(sllm_completion_release(&completion, &error.sink),
                            SLLM_STATUS_OK, "graph_replay_release", error);
      }
    }
    ok &= expect_status(sllm_completion_release(&fence, &error.sink),
                        SLLM_STATUS_OK, "graph_fence_release", error);
    ok &= expect_status(
        sllm_graph_span_publish_state_metadata(span, 0U, 3U, 3U, &error.sink),
        SLLM_STATUS_OK, "graph_publish_metadata", error);
  }
  if (ok) {
    constexpr uint64_t kReadbackBytes = 192U;
    std::array<std::array<uint8_t, kReadbackBytes>, 3U> readbacks{};
    std::array<sllm_completion_t *, 3U> readback_completions{};
    auto submit_readback = [&](const uint32_t index,
                               sllm_completion_t **const completion,
                               const char *const where) {
      sllm_transfer_desc_t transfer{};
      transfer.struct_size = sizeof(transfer);
      transfer.abi_version = SLLM_HIP_ABI_VERSION;
      transfer.host_pointer = readbacks[index].data();
      transfer.buffer_offset_bytes =
          static_cast<uint64_t>(index == 1U ? 1U : 0U) * kReadbackBytes;
      transfer.size_bytes = kReadbackBytes;
      return expect_status(sllm_buffer_copy_d2h(graph_queue, result_buffer,
                                                &transfer, completion,
                                                &error.sink),
                           SLLM_STATUS_OK, where, error);
    };
    ok &= submit_readback(0U, &readback_completions[0], "pinned_d2h_submit0");
    ok &= submit_readback(1U, &readback_completions[1], "pinned_d2h_submit1");
    sllm_completion_t *rejected_readback = nullptr;
    sllm_transfer_desc_t rejected_transfer{};
    rejected_transfer.struct_size = sizeof(rejected_transfer);
    rejected_transfer.abi_version = SLLM_HIP_ABI_VERSION;
    rejected_transfer.host_pointer = readbacks[2].data();
    rejected_transfer.buffer_offset_bytes = 0U;
    rejected_transfer.size_bytes = kReadbackBytes;
    ok &= expect_status(sllm_buffer_copy_d2h(graph_queue, result_buffer,
                                             &rejected_transfer,
                                             &rejected_readback, &error.sink),
                        SLLM_STATUS_PUBLIC_BUSY, "pinned_d2h_third_busy",
                        error) &&
          rejected_readback == nullptr;
    auto drain_readback = [&](const uint32_t index) {
      sllm_completion_result_t readback_result{};
      readback_result.struct_size = sizeof(readback_result);
      readback_result.abi_version = SLLM_HIP_ABI_VERSION;
      bool drained = expect_status(sllm_completion_wait(
                                       readback_completions[index], UINT32_MAX,
                                       &readback_result, &error.sink),
                                   SLLM_STATUS_OK, "pinned_d2h_wait", error) &&
                     readback_result.state == SLLM_COMPLETION_STATE_SUCCESS;
      uint64_t bytes_written = 0U;
      drained = expect_status(sllm_completion_read(readback_completions[index],
                                                   readbacks[index].data(),
                                                   kReadbackBytes,
                                                   &bytes_written, &error.sink),
                              SLLM_STATUS_OK, "pinned_d2h_read", error) &&
                bytes_written == kReadbackBytes && drained;
      drained = expect_status(sllm_completion_release(
                                  &readback_completions[index], &error.sink),
                              SLLM_STATUS_OK, "pinned_d2h_release", error) &&
                readback_completions[index] == nullptr && drained;
      return drained;
    };
    ok &= drain_readback(0U);
    ok &= drain_readback(1U);
    ok &=
        submit_readback(2U, &readback_completions[2], "pinned_d2h_reuse_slot");
    ok &= drain_readback(2U);
    // Distinct generation slots must not alias the same host staging memory.
    // The third copy reuses a released slot but must reproduce generation 2;
    // slot one contains the generation-3 budget Noop from the extra replay.
    for (uint32_t index = 0U; index < 3U; ++index) {
      sllm_decode_control::ResultV1 observed{};
      std::memcpy(&observed, readbacks[index].data(), sizeof(observed));
      const bool noop = index == 1U;
      const uint64_t expected_generation = noop ? 3U : 2U;
      const bool bytes_valid =
          observed.generation == expected_generation &&
          observed.status ==
              static_cast<uint32_t>(noop ? sllm_decode_control::Status::Noop
                                         : sllm_decode_control::Status::Ok) &&
          observed.count == (noop ? 0U : 1U) &&
          observed.commit_rows == (noop ? 0U : 1U) &&
          observed.model_position_after == expected_generation &&
          observed.selected_ids[0] == (noop ? 0U : 1U);
      if (!bytes_valid) {
        std::cerr << "pinned D2H payload differs for slot " << index << '\n';
      }
      ok &= bytes_valid;
    }
    ok &= readbacks[0] == readbacks[2] && readbacks[0] != readbacks[1];
  }
  if (span != nullptr) {
    ok &= expect_status(sllm_graph_span_release(&span, &error.sink),
                        SLLM_STATUS_OK, "graph_release", error);
  }
  sllm_linear_attention_view_info_t view{};
  view.struct_size = sizeof(view);
  view.abi_version = SLLM_HIP_ABI_VERSION;
  view.info_version = SLLM_HIP_LINEAR_ATTENTION_VIEW_INFO_VERSION;
  ok &= expect_status(
            sllm_linear_attention_state_query(graph_state, &view, &error.sink),
            SLLM_STATUS_OK, "graph_state_query", error) &&
        /* One eager warmup is rewound for metadata/position but its
         * generation remains part of the native monotonic history. */
        view.observed_length == 3U && view.generation == 5U &&
        view.active_slot == 1U;
  if (graph_state && eager_state) {
    std::vector<uint8_t> graph_slot0_conv(conv_state_bytes),
        graph_slot0_recurrent(recurrent_state_bytes),
        graph_active_conv(conv_state_bytes),
        graph_active_recurrent(recurrent_state_bytes);
    std::vector<uint8_t> eager_conv(conv_state_bytes),
        eager_recurrent(recurrent_state_bytes);
    uint32_t graph_slot = 0U, eager_slot = 0U;
    const bool export_ok =
        export_slot(graph_state, 0U, &graph_slot0_conv,
                    &graph_slot0_recurrent) &&
        export_active(graph_state, &graph_active_conv, &graph_active_recurrent,
                      &graph_slot) &&
        export_active(eager_state, &eager_conv, &eager_recurrent, &eager_slot);
    const bool state_match = graph_slot == 1U && graph_slot == eager_slot &&
                             graph_slot0_conv == eager_two_conv &&
                             graph_slot0_recurrent == eager_two_recurrent &&
                             graph_active_conv == eager_conv &&
                             graph_active_recurrent == eager_recurrent;
    ok &= export_ok && state_match;
  }
  if (graph_state)
    ok &= expect_status(
        sllm_linear_attention_state_release(&graph_state, &error.sink),
        SLLM_STATUS_OK, "graph_state_release", error);
  if (eager_state)
    ok &= expect_status(
        sllm_linear_attention_state_release(&eager_state, &error.sink),
        SLLM_STATUS_OK, "eager_state_release", error);
  for (auto &buffer : buffers)
    ok &= release_buffer(&buffer, "linear_buffer_release");
  ok &= release_buffer(&control_buffer, "control_release");
  ok &= release_buffer(&selector_buffer, "selector_release");
  ok &= release_buffer(&result_buffer, "result_release");
  if (eager_queue)
    ok &= expect_status(sllm_queue_release(&eager_queue, &error.sink),
                        SLLM_STATUS_OK, "eager_queue_release", error);
  if (graph_queue)
    ok &= expect_status(sllm_queue_release(&graph_queue, &error.sink),
                        SLLM_STATUS_OK, "graph_queue_release", error);
  if (context)
    ok &= expect_status(sllm_context_release(&context, &error.sink),
                        SLLM_STATUS_OK, "context_release", error);
  std::cout << "phase87_gdn_graph_lifecycle_gpu_test status="
            << (ok ? "PASS" : "FAIL") << " target=" << SLLM_TEST_EXPECTED_TARGET
            << '\n';
  return ok ? 0 : 1;
}
