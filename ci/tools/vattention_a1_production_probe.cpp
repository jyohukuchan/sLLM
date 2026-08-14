// SPDX-License-Identifier: MIT
// Focused production-C-ABI proof for vAttention KV B-1/B/B+1 growth.

#include "evidence_abi.h"
#include "sllm/hip.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t kCapacity = 1025;
constexpr std::uint64_t kRowWords = 4 * 256;
constexpr std::uint64_t kPageBytes = 2 * 1024 * 1024;

struct Error {
  char bytes[512]{};
  sllm_error_sink_t sink{};
  Error() {
    sink.struct_size = sizeof(sink);
    sink.abi_version = SLLM_HIP_ABI_VERSION;
    sink.message = bytes;
    sink.message_capacity = sizeof(bytes);
  }
};

void require_status(std::uint32_t actual, std::uint32_t expected,
                    const char *operation, const Error &error) {
  if (actual != expected) {
    throw std::runtime_error(std::string(operation) + " status=" +
                             std::to_string(actual) + " error=" + error.bytes);
  }
}

void wait_and_release(sllm_completion_t *&completion, Error &error) {
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  require_status(sllm_completion_wait(completion, 30000U, &result, &error.sink),
                 SLLM_STATUS_OK, "sllm_completion_wait", error);
  if (result.state != SLLM_COMPLETION_STATE_SUCCESS) {
    throw std::runtime_error("completion did not reach success");
  }
  require_status(sllm_completion_release(&completion, &error.sink),
                 SLLM_STATUS_OK, "sllm_completion_release", error);
  if (completion != nullptr) {
    throw std::runtime_error("completion release did not clear handle");
  }
}

std::uint16_t bf16_to_f16(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::uint32_t exponent = (bits >> 23U) & 0xffU;
  const std::uint32_t fraction = bits & 0x7fffffU;
  if (exponent == 0xffU) {
    return static_cast<std::uint16_t>(sign |
                                      (fraction == 0U ? 0x7c00U : 0x7e00U));
  }
  const std::int32_t half_exponent =
      static_cast<std::int32_t>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t mantissa = fraction | 0x800000U;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - half_exponent);
    std::uint32_t rounded = mantissa >> shift;
    const std::uint32_t remainder = mantissa & ((1U << shift) - 1U);
    const std::uint32_t halfway = 1U << (shift - 1U);
    if (remainder > halfway ||
        (remainder == halfway && (rounded & 1U) != 0U)) {
      ++rounded;
    }
    return static_cast<std::uint16_t>(sign | rounded);
  }
  std::uint32_t rounded_fraction = fraction >> 13U;
  const std::uint32_t remainder = fraction & 0x1fffU;
  if (remainder > 0x1000U ||
      (remainder == 0x1000U && (rounded_fraction & 1U) != 0U)) {
    ++rounded_fraction;
    if (rounded_fraction == 0x400U) {
      rounded_fraction = 0U;
      if (half_exponent + 1 >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00U);
      }
      return static_cast<std::uint16_t>(
          sign | (static_cast<std::uint32_t>(half_exponent + 1) << 10U));
    }
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(half_exponent) << 10U) |
      rounded_fraction);
}

sllm_tensor_binding_t binding(const sllm_buffer_t *buffer,
                              std::uint64_t tokens) {
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.dtype = SLLM_TENSOR_DTYPE_BF16;
  result.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  result.rank = 3;
  result.shape[0] = tokens;
  result.shape[1] = 4;
  result.shape[2] = 256;
  result.stride_elements[0] = 4 * 256;
  result.stride_elements[1] = 256;
  result.stride_elements[2] = 1;
  return result;
}

std::vector<std::uint16_t> input_words(std::uint64_t tokens,
                                       std::uint32_t seed) {
  std::vector<std::uint16_t> words(
      static_cast<std::size_t>(tokens * kRowWords));
  for (std::size_t index = 0; index < words.size(); ++index) {
    words[index] = static_cast<std::uint16_t>(
        0x3e00U + ((index * 37U + seed * 13U) % 0x180U));
  }
  return words;
}

void upload(sllm_queue_t *queue, sllm_buffer_t *buffer,
            std::vector<std::uint16_t> &words, Error &error) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = words.data();
  transfer.size_bytes = words.size() * sizeof(std::uint16_t);
  sllm_completion_t *completion = nullptr;
  require_status(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                      &error.sink),
                 SLLM_STATUS_OK, "sllm_buffer_copy_h2d", error);
  wait_and_release(completion, error);
}

void append(sllm_kv_state_t *state, sllm_queue_t *queue,
            sllm_buffer_t *key, sllm_buffer_t *value, std::uint64_t tokens,
            std::uint64_t position, Error &error) {
  sllm_kv_append_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.append_version = SLLM_HIP_KV_STATE_VERSION;
  descriptor.expected_length = position;
  descriptor.start_position = position;
  descriptor.key_input = binding(key, tokens);
  descriptor.value_input = binding(value, tokens);
  sllm_kv_append_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  require_status(sllm_kv_state_append(state, queue, &descriptor, &completion,
                                      &info, &error.sink),
                 SLLM_STATUS_OK, "sllm_kv_state_append", error);
  if (info.kernel_id != SLLM_HIP_KV_KERNEL_ID_BF16_TO_F16_TOKEN_MAJOR_V2 ||
      info.fallback_used != 0U || info.token_count != tokens) {
    throw std::runtime_error("append dispatch metadata drifted");
  }
  wait_and_release(completion, error);
}

sllm_kv_view_info_t query(sllm_kv_state_t *state, Error &error) {
  sllm_kv_view_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
  require_status(sllm_kv_state_query(state, &info, &error.sink),
                 SLLM_STATUS_OK, "sllm_kv_state_query", error);
  if (info.memory_kind != SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS ||
      info.layout != SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR ||
      info.k_stride_elements[0] != 1024U ||
      info.v_stride_elements[0] != 1024U) {
    throw std::runtime_error("production KV layout metadata drifted");
  }
  return info;
}

std::vector<std::uint16_t> read_plane(sllm_kv_state_t *state,
                                      std::uint32_t plane, Error &error) {
  sllm_kv_view_t *view = nullptr;
  require_status(sllm_kv_state_snapshot(state, &view, &error.sink),
                 SLLM_STATUS_OK, "sllm_kv_state_snapshot", error);
  std::vector<std::uint16_t> words(
      static_cast<std::size_t>(kCapacity * kRowWords));
  sllm_hip_kv_readback_request_t request{};
  request.struct_size = sizeof(request);
  request.abi_version = SLLM_HIP_KV_EVIDENCE_ABI_VERSION;
  request.view = view;
  request.plane = plane;
  request.byte_length = words.size() * sizeof(std::uint16_t);
  request.host_capacity = request.byte_length;
  request.host_output = reinterpret_cast<std::uint8_t *>(words.data());
  require_status(sllm_hip_kv_view_readback(&request, &error.sink),
                 SLLM_STATUS_OK, "sllm_hip_kv_view_readback", error);
  require_status(sllm_kv_view_release(&view, &error.sink), SLLM_STATUS_OK,
                 "sllm_kv_view_release", error);
  return words;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error("usage: production-probe <expected-target>");
    }
    const std::string target(argv[1]);
    Error error;
    sllm_context_create_info_t context_info{};
    context_info.struct_size = sizeof(context_info);
    context_info.abi_version = SLLM_HIP_ABI_VERSION;
    std::strncpy(context_info.expected_gcn_arch_name, target.c_str(),
                 sizeof(context_info.expected_gcn_arch_name) - 1U);
    sllm_context_t *context = nullptr;
    require_status(sllm_context_create(&context_info, &context, &error.sink),
                   SLLM_STATUS_OK, "sllm_context_create", error);
    sllm_queue_create_info_t queue_info{};
    queue_info.struct_size = sizeof(queue_info);
    queue_info.abi_version = SLLM_HIP_ABI_VERSION;
    sllm_queue_t *queue = nullptr;
    require_status(sllm_queue_create(context, &queue_info, &queue, &error.sink),
                   SLLM_STATUS_OK, "sllm_queue_create", error);
    sllm_kv_state_create_info_t state_info{};
    state_info.struct_size = sizeof(state_info);
    state_info.abi_version = SLLM_HIP_ABI_VERSION;
    state_info.session_id = 0xa1U;
    state_info.layer_id = 7U;
    state_info.capacity_tokens = kCapacity;
    state_info.head_count = 4U;
    state_info.head_dim = 256U;
    state_info.memory_kind = SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS;
    state_info.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
    sllm_kv_state_t *state = nullptr;
    require_status(sllm_kv_state_create(context, &state_info, &state,
                                        &error.sink),
                   SLLM_STATUS_OK, "sllm_kv_state_create", error);
    const auto initial = query(state, error);
    if (initial.observed_length != 0U || initial.mapped_token_capacity != 0U ||
        initial.committed_bytes_per_plane != 0U) {
      throw std::runtime_error("new KV state was physically committed");
    }
    sllm_kv_view_t *empty_view = nullptr;
    require_status(sllm_kv_state_snapshot(state, &empty_view, &error.sink),
                   SLLM_STATUS_OK, "empty snapshot", error);
    std::uint16_t empty_word = 0U;
    sllm_hip_kv_readback_request_t empty_request{};
    empty_request.struct_size = sizeof(empty_request);
    empty_request.abi_version = SLLM_HIP_KV_EVIDENCE_ABI_VERSION;
    empty_request.view = empty_view;
    empty_request.plane = SLLM_HIP_KV_EVIDENCE_PLANE_K;
    empty_request.byte_length = sizeof(empty_word);
    empty_request.host_capacity = sizeof(empty_word);
    empty_request.host_output = reinterpret_cast<std::uint8_t *>(&empty_word);
    require_status(sllm_hip_kv_view_readback(&empty_request, &error.sink),
                   SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
                   "unmapped readback rejection", error);
    require_status(sllm_kv_view_release(&empty_view, &error.sink),
                   SLLM_STATUS_OK, "empty view release", error);

    const std::uint64_t input_bytes = 1023U * kRowWords * 2U;
    sllm_buffer_create_info_t buffer_info{};
    buffer_info.struct_size = sizeof(buffer_info);
    buffer_info.abi_version = SLLM_HIP_ABI_VERSION;
    buffer_info.size_bytes = input_bytes;
    sllm_buffer_t *key = nullptr;
    sllm_buffer_t *value = nullptr;
    require_status(sllm_buffer_create(context, &buffer_info, &key, &error.sink),
                   SLLM_STATUS_OK, "key buffer create", error);
    require_status(
        sllm_buffer_create(context, &buffer_info, &value, &error.sink),
        SLLM_STATUS_OK, "value buffer create", error);
    auto key_input = input_words(1023U, 3U);
    auto value_input = input_words(1023U, 11U);
    upload(queue, key, key_input, error);
    upload(queue, value, value_input, error);
    append(state, queue, key, value, 1023U, 0U, error);
    const auto before = query(state, error);
    append(state, queue, key, value, 1U, 1023U, error);
    const auto boundary = query(state, error);
    append(state, queue, key, value, 1U, 1024U, error);
    const auto after = query(state, error);
    if (before.observed_length != 1023U ||
        before.mapped_token_capacity != 1024U ||
        before.committed_bytes_per_plane != kPageBytes ||
        boundary.observed_length != 1024U ||
        boundary.mapped_token_capacity != 1024U ||
        boundary.committed_bytes_per_plane != kPageBytes ||
        after.observed_length != 1025U ||
        after.mapped_token_capacity != 1025U ||
        after.committed_bytes_per_plane != 2U * kPageBytes) {
      throw std::runtime_error("B-1/B/B+1 commitment metadata mismatch");
    }
    const auto key_output =
        read_plane(state, SLLM_HIP_KV_EVIDENCE_PLANE_K, error);
    const auto value_output =
        read_plane(state, SLLM_HIP_KV_EVIDENCE_PLANE_V, error);
    for (std::uint64_t token = 0U; token < kCapacity; ++token) {
      for (std::uint64_t within = 0U; within < kRowWords; ++within) {
        const std::uint64_t source = token < 1023U ? token * kRowWords + within
                                                   : within;
        const std::uint64_t output = token * kRowWords + within;
        if (key_output[output] != bf16_to_f16(key_input[source]) ||
            value_output[output] != bf16_to_f16(value_input[source])) {
          throw std::runtime_error("token-major FP16 numerical oracle failed");
        }
      }
    }
    require_status(sllm_buffer_release(&key, &error.sink), SLLM_STATUS_OK,
                   "key buffer release", error);
    require_status(sllm_buffer_release(&value, &error.sink), SLLM_STATUS_OK,
                   "value buffer release", error);
    require_status(sllm_kv_state_release(&state, &error.sink), SLLM_STATUS_OK,
                   "state release", error);
    require_status(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
                   "queue release", error);
    require_status(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
                   "context release", error);
    std::cout << "{\"protocol\":\"sllm-vattention-a1-production-v1\","
                 "\"state\":\"PASS\",\"target\":\""
              << target
              << "\",\"layout\":\"token-major\","
                 "\"memory_kind\":\"virtual-contiguous\","
                 "\"boundary_tokens\":[1023,1024,1025],"
                 "\"committed_bytes_per_plane\":[2097152,2097152,4194304],"
                 "\"unmapped_readback_rejected\":true,"
                 "\"numerical_oracle\":true,\"fallback_used\":false,"
                 "\"cleanup_complete\":true}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "vAttention A1 production probe failed: " << error.what()
              << '\n';
    return 1;
  }
}
