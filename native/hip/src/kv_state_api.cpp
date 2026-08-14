#include "kv_state_api.hpp"

#include <cstring>
#include <limits>

namespace sllm_kv_state {
namespace {

bool all_zero(const uint32_t *const values, const std::size_t count) noexcept {
  for (std::size_t index = 0U; index != count; ++index) {
    if (values[index] != 0U) {
      return false;
    }
  }
  return true;
}

bool all_zero(const uint64_t *const values, const std::size_t count) noexcept {
  for (std::size_t index = 0U; index != count; ++index) {
    if (values[index] != 0U) {
      return false;
    }
  }
  return true;
}

bool multiply_overflows(const uint64_t left, const uint64_t right,
                        uint64_t *const result) noexcept {
  if (left != 0U && right > std::numeric_limits<uint64_t>::max() / left) {
    return true;
  }
  *result = left * right;
  return false;
}

sllm_status_t exact_struct(const uint32_t struct_size,
                           const uint32_t abi_version,
                           const std::size_t expected_size,
                           sllm_error_sink_t *const sink,
                           const char *const size_message) noexcept {
  if (struct_size != expected_size) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                                            size_message);
  }
  if (abi_version != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "KV state public ABI version is unsupported");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_tensor(const sllm_tensor_binding_t *const binding,
                              TensorMetadata *const metadata,
                              sllm_error_sink_t *const sink,
                              const char *const name) noexcept {
  if (binding == nullptr || metadata == nullptr) {
    return sllm_public_runtime::write_error(sink,
                                            SLLM_STATUS_INVALID_TENSOR_BINDING,
                                            "KV append tensor binding is null");
  }
  const sllm_status_t struct_status = exact_struct(
      binding->struct_size, binding->abi_version, sizeof(*binding), sink,
      "KV append tensor binding has an unsupported struct size");
  if (struct_status != SLLM_STATUS_OK) {
    return struct_status;
  }
  if (binding->reserved0 != 0U || !all_zero(binding->reserved, 2U)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "KV append tensor binding reserved fields must be zero");
  }
  if (binding->buffer == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "KV append tensor binding buffer is null");
  }
  if (binding->dtype != SLLM_TENSOR_DTYPE_BF16) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_UNSUPPORTED_DTYPE,
                                            "KV append inputs must use BF16");
  }
  if (binding->encoding != SLLM_TENSOR_ENCODING_UNQUANTIZED) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_ENCODING,
        "KV append inputs must be unquantized");
  }
  if (binding->rank != 3U ||
      (binding->shape[1] != 2U && binding->shape[1] != 4U) ||
      binding->shape[2] != SLLM_HIP_KV_HEAD_DIM || binding->shape[0] == 0U ||
      binding->shape[0] > SLLM_HIP_KV_MAX_M) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "KV append inputs must have a reviewed shape [M, Hkv, 256]");
  }
  if ((binding->byte_offset % UINT64_C(2)) != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_MISALIGNED_OFFSET,
        "KV append input offset is not FP16/BF16 aligned");
  }
  const uint64_t expected_strides[3] = {binding->shape[1] * binding->shape[2],
                                        binding->shape[2], 1U};
  for (uint32_t index = 0U; index != 3U; ++index) {
    if (binding->stride_elements[index] != expected_strides[index]) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_STRIDE_MISMATCH,
          "KV append inputs must be row-major contiguous");
    }
  }
  for (uint32_t index = 3U; index != SLLM_HIP_TENSOR_MAX_RANK; ++index) {
    if (binding->shape[index] != 0U || binding->stride_elements[index] != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
          "KV append unused tensor metadata must be zero");
    }
  }

  uint64_t elements = 0U;
  uint64_t payload_bytes = 0U;
  if (multiply_overflows(binding->shape[0], binding->shape[1], &elements) ||
      multiply_overflows(elements, binding->shape[2], &elements) ||
      multiply_overflows(elements, UINT64_C(2), &payload_bytes) ||
      sllm_public_runtime::add_overflows(binding->byte_offset, payload_bytes)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "KV append tensor byte interval overflowed u64");
  }
  metadata->buffer = binding->buffer;
  metadata->byte_offset = binding->byte_offset;
  metadata->payload_bytes = payload_bytes;
  metadata->end_offset = binding->byte_offset + payload_bytes;
  metadata->token_count = binding->shape[0];
  metadata->head_count = static_cast<uint32_t>(binding->shape[1]);
  metadata->head_dim = static_cast<uint32_t>(binding->shape[2]);
  (void)name;
  return SLLM_STATUS_OK;
}

} // namespace

sllm_status_t
validate_state_create_info(const sllm_kv_state_create_info_t *const info,
                           sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_KV_STATE_DESCRIPTOR,
        "KV state create info is null");
  }
  const sllm_status_t struct_status =
      exact_struct(info->struct_size, info->abi_version, sizeof(*info), sink,
                   "KV state create info has an unsupported struct size");
  if (struct_status != SLLM_STATUS_OK) {
    return struct_status;
  }
  const uint32_t head_count =
      info->head_count == 0U ? SLLM_HIP_KV_HEAD_COUNT : info->head_count;
  const uint32_t head_dim =
      info->head_dim == 0U ? SLLM_HIP_KV_HEAD_DIM : info->head_dim;
  if (info->flags != 0U || info->session_id == 0U ||
      info->capacity_tokens == 0U ||
      info->capacity_tokens > SLLM_HIP_KV_MAX_CAPACITY ||
      (head_count != 2U && head_count != 4U) ||
      head_dim != SLLM_HIP_KV_HEAD_DIM ||
      (info->memory_kind != SLLM_HIP_KV_MEMORY_KIND_CAPABILITY_SELECTED &&
       info->memory_kind != SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS &&
       info->memory_kind != SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT) ||
      info->layout != SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_KV_STATE_DESCRIPTOR,
        "KV state create info has an invalid session, shape, memory kind, or "
        "layout");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_append_prefix(const sllm_kv_append_desc_t *const descriptor,
                       sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_KV_APPEND_DESCRIPTOR,
        "KV append descriptor is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, descriptor, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_kv_append_desc_t)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "KV append descriptor has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "KV append public ABI version is unsupported");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_and_copy_append(const sllm_kv_append_desc_t *const descriptor,
                         AppendMetadata *const metadata,
                         sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr || metadata == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_KV_APPEND_DESCRIPTOR,
        "KV append descriptor or metadata output is null");
  }
  const sllm_status_t prefix_status = validate_append_prefix(descriptor, sink);
  if (prefix_status != SLLM_STATUS_OK) {
    return prefix_status;
  }
  const sllm_status_t struct_status = exact_struct(
      descriptor->struct_size, descriptor->abi_version, sizeof(*descriptor),
      sink, "KV append descriptor has an unsupported struct size");
  if (struct_status != SLLM_STATUS_OK) {
    return struct_status;
  }
  if (descriptor->append_version != SLLM_HIP_KV_STATE_VERSION ||
      descriptor->reserved0 != 0U || !all_zero(descriptor->reserved, 4U)) {
    return sllm_public_runtime::write_error(
        sink,
        descriptor->reserved0 != 0U || !all_zero(descriptor->reserved, 4U)
            ? SLLM_STATUS_RESERVED_NONZERO
            : SLLM_STATUS_INVALID_KV_APPEND_DESCRIPTOR,
        "KV append descriptor version or reserved fields are invalid");
  }
  if (descriptor->expected_length != descriptor->start_position) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_KV_LENGTH_MISMATCH,
        "KV append expected length and start position must match");
  }
  metadata->expected_length = descriptor->expected_length;
  metadata->start_position = descriptor->start_position;
  sllm_status_t status =
      validate_tensor(&descriptor->key_input, &metadata->key_input, sink,
                      "KV key input shape is invalid");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(&descriptor->value_input, &metadata->value_input,
                           sink, "KV value input shape is invalid");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (metadata->key_input.token_count != metadata->value_input.token_count) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "KV key and value append token counts must match");
  }
  if (metadata->key_input.head_count != metadata->value_input.head_count ||
      metadata->key_input.head_dim != metadata->value_input.head_dim) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "KV key and value append head layouts must match");
  }
  metadata->token_count = metadata->key_input.token_count;
  if (sllm_public_runtime::add_overflows(metadata->start_position,
                                         metadata->token_count)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "KV append position interval overflowed u64");
  }
  metadata->end_position = metadata->start_position + metadata->token_count;
  return SLLM_STATUS_OK;
}

void initialize_view_info(sllm_kv_view_info_t *const info) noexcept {
  if (info == nullptr) {
    return;
  }
  std::memset(info, 0, sizeof(*info));
  info->struct_size = sizeof(*info);
  info->abi_version = SLLM_HIP_ABI_VERSION;
  info->info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
}

void initialize_append_info(sllm_kv_append_info_t *const info) noexcept {
  if (info == nullptr) {
    return;
  }
  std::memset(info, 0, sizeof(*info));
  info->struct_size = sizeof(*info);
  info->abi_version = SLLM_HIP_ABI_VERSION;
  info->info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
  info->backend = SLLM_BACKEND_HIP;
}

} // namespace sllm_kv_state
