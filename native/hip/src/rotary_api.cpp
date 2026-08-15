#include "rotary_api.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace sllm_rotary {
namespace {

sllm_status_t exact_struct(const uint32_t struct_size,
                           const uint32_t abi_version,
                           const std::size_t expected_size,
                           sllm_error_sink_t *const sink,
                           const char *const name) noexcept {
  if (struct_size != expected_size) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                                            name);
  }
  if (abi_version != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "rotary public ABI version is unsupported");
  }
  return SLLM_STATUS_OK;
}

bool multiply_overflows(const uint64_t left, const uint64_t right,
                        uint64_t *const result) noexcept {
  if (left != 0U && right > std::numeric_limits<uint64_t>::max() / left) {
    return true;
  }
  *result = left * right;
  return false;
}

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

sllm_status_t validate_tensor(const sllm_tensor_binding_t *const binding,
                              const uint32_t expected_dtype,
                              const uint32_t expected_rank,
                              TensorMetadata *const copied,
                              sllm_error_sink_t *const sink,
                              const char *const name) noexcept {
  if (binding == nullptr || copied == nullptr) {
    return sllm_public_runtime::write_error(sink,
                                            SLLM_STATUS_INVALID_TENSOR_BINDING,
                                            "rotary tensor binding is null");
  }
  const sllm_status_t struct_status = exact_struct(
      binding->struct_size, binding->abi_version, sizeof(*binding), sink,
      "rotary tensor binding has an unsupported struct size");
  if (struct_status != SLLM_STATUS_OK) {
    return struct_status;
  }
  if (binding->reserved0 != 0U || !all_zero(binding->reserved, 2U)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "rotary tensor binding reserved fields must be zero");
  }
  if (binding->buffer == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "rotary tensor binding buffer is null");
  }
  if (binding->rank != expected_rank) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_SHAPE_MISMATCH,
                                            name);
  }
  if (binding->dtype != expected_dtype) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_DTYPE,
        "rotary tensor has an unsupported dtype");
  }
  if (binding->encoding != SLLM_TENSOR_ENCODING_UNQUANTIZED) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_ENCODING,
        "rotary tensors must be unquantized");
  }
  const uint64_t element_bytes =
      expected_dtype == SLLM_TENSOR_DTYPE_I32 ? UINT64_C(4) : UINT64_C(2);
  if ((binding->byte_offset % element_bytes) != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_MISALIGNED_OFFSET,
        "rotary tensor offset is not element aligned");
  }

  uint64_t expected_stride = 1U;
  uint64_t elements = 1U;
  for (uint32_t backwards = 0U; backwards != binding->rank; ++backwards) {
    const uint32_t index = binding->rank - 1U - backwards;
    const uint64_t extent = binding->shape[index];
    if (extent == 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_ZERO_EXTENT,
          "rotary tensor extents must be non-zero");
    }
    if (binding->stride_elements[index] != expected_stride) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_STRIDE_MISMATCH,
          "rotary tensors must be row-major contiguous");
    }
    if (multiply_overflows(expected_stride, extent, &expected_stride) ||
        multiply_overflows(elements, extent, &elements)) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_METADATA_OVERFLOW,
          "rotary tensor metadata overflowed u64");
    }
  }
  for (uint32_t index = binding->rank; index != SLLM_HIP_TENSOR_MAX_RANK;
       ++index) {
    if (binding->shape[index] != 0U || binding->stride_elements[index] != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
          "rotary unused tensor metadata must be zero");
    }
  }
  uint64_t payload_bytes = 0U;
  if (multiply_overflows(elements, element_bytes, &payload_bytes) ||
      sllm_public_runtime::add_overflows(binding->byte_offset, payload_bytes)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "rotary tensor byte interval overflowed u64");
  }
  copied->byte_offset = binding->byte_offset;
  copied->payload_bytes = payload_bytes;
  copied->end_offset = binding->byte_offset + payload_bytes;
  copied->rank = binding->rank;
  copied->element_bytes = static_cast<uint32_t>(element_bytes);
  std::memcpy(copied->shape, binding->shape, sizeof(copied->shape));
  std::memcpy(copied->strides, binding->stride_elements,
              sizeof(copied->strides));
  return SLLM_STATUS_OK;
}

bool exact_shape(const TensorMetadata &tensor, const uint32_t rank,
                 const uint64_t *const shape) noexcept {
  if (tensor.rank != rank) {
    return false;
  }
  for (uint32_t index = 0U; index != rank; ++index) {
    if (tensor.shape[index] != shape[index]) {
      return false;
    }
  }
  return true;
}

} // namespace

sllm_status_t
validate_descriptor_prefix(const sllm_rotary_desc_t *const descriptor,
                           sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ROTARY_DESCRIPTOR,
        "rotary descriptor is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, descriptor, sizeof(prefix));
  if (prefix[0] < sizeof(prefix) || prefix[0] != sizeof(*descriptor)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "rotary descriptor prefix has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "rotary public ABI version is unsupported");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_and_copy_descriptor(const sllm_rotary_desc_t *const descriptor,
                             DescriptorMetadata *const metadata,
                             sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr || metadata == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ROTARY_DESCRIPTOR,
        "rotary descriptor or metadata output is null");
  }
  const sllm_status_t prefix_status =
      validate_descriptor_prefix(descriptor, sink);
  if (prefix_status != SLLM_STATUS_OK) {
    return prefix_status;
  }
  const sllm_status_t struct_status = exact_struct(
      descriptor->struct_size, descriptor->abi_version, sizeof(*descriptor),
      sink, "rotary descriptor has an unsupported struct size");
  if (struct_status != SLLM_STATUS_OK) {
    return struct_status;
  }
  if (descriptor->reserved0 != 0U || !all_zero(descriptor->reserved, 2U)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "rotary descriptor reserved fields must be zero");
  }
  float theta = 0.0F;
  std::memcpy(&theta, &descriptor->theta_bits, sizeof(theta));
  if (descriptor->op_version != SLLM_HIP_ROTARY_VERSION ||
      descriptor->q_heads == 0U || descriptor->kv_heads == 0U ||
      descriptor->q_heads % descriptor->kv_heads != 0U ||
      descriptor->head_dim == 0U || (descriptor->head_dim & 1U) != 0U ||
      descriptor->rotary_dim == 0U || (descriptor->rotary_dim & 1U) != 0U ||
      descriptor->rotary_dim > descriptor->head_dim || !std::isfinite(theta) ||
      theta <= 0.0F || descriptor->max_position == 0U ||
      descriptor->max_position > SLLM_HIP_ROTARY_MAX_POSITION ||
      descriptor->start_position >= descriptor->max_position) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ROTARY_DESCRIPTOR,
        "rotary operation contract is invalid or unsupported");
  }

  sllm_status_t status = validate_tensor(
      &descriptor->query, SLLM_TENSOR_DTYPE_BF16, 3U, &metadata->query, sink,
      "rotary query tensor must have rank three");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(&descriptor->key, SLLM_TENSOR_DTYPE_BF16, 3U,
                           &metadata->key, sink,
                           "rotary key tensor must have rank three");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(&descriptor->positions, SLLM_TENSOR_DTYPE_I32, 1U,
                           &metadata->positions, sink,
                           "rotary positions tensor must have rank one");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(&descriptor->query_output, SLLM_TENSOR_DTYPE_BF16,
                           3U, &metadata->query_output, sink,
                           "rotary query output must have rank three");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(&descriptor->key_output, SLLM_TENSOR_DTYPE_BF16, 3U,
                           &metadata->key_output, sink,
                           "rotary key output must have rank three");
  if (status != SLLM_STATUS_OK) {
    return status;
  }

  const uint64_t token_count = metadata->query.shape[0];
  const uint64_t query_shape[] = {token_count, descriptor->q_heads,
                                  descriptor->head_dim};
  const uint64_t key_shape[] = {token_count, descriptor->kv_heads,
                                descriptor->head_dim};
  const uint64_t position_shape[] = {token_count};
  const uint64_t launch_blocks =
      token_count *
      (static_cast<uint64_t>(descriptor->q_heads) + descriptor->kv_heads);
  if (token_count == 0U || token_count > SLLM_HIP_ROTARY_MAX_M ||
      token_count > descriptor->max_position ||
      descriptor->start_position >
          static_cast<uint64_t>(descriptor->max_position) - token_count ||
      launch_blocks > std::numeric_limits<uint32_t>::max() ||
      !exact_shape(metadata->query, 3U, query_shape) ||
      !exact_shape(metadata->key, 3U, key_shape) ||
      !exact_shape(metadata->positions, 1U, position_shape) ||
      !exact_shape(metadata->query_output, 3U, query_shape) ||
      !exact_shape(metadata->key_output, 3U, key_shape)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "rotary tensors or position range do not match the descriptor");
  }

  metadata->token_count = token_count;
  metadata->start_position = descriptor->start_position;
  metadata->q_heads = descriptor->q_heads;
  metadata->kv_heads = descriptor->kv_heads;
  metadata->head_dim = descriptor->head_dim;
  metadata->rotary_dim = descriptor->rotary_dim;
  metadata->theta_bits = descriptor->theta_bits;
  metadata->max_position = descriptor->max_position;
  return SLLM_STATUS_OK;
}

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept {
  return left.byte_offset < right.end_offset &&
         right.byte_offset < left.end_offset;
}

} // namespace sllm_rotary
