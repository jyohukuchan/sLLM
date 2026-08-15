#include "windowed_attention_api.hpp"

#include <cstring>
#include <limits>

namespace sllm_windowed_attention {
namespace {

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

sllm_status_t validate_tensor(const sllm_tensor_binding_t &binding,
                              TensorMetadata *const copied,
                              sllm_error_sink_t *const sink) noexcept {
  if (copied == nullptr || binding.struct_size != sizeof(binding)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "windowed attention tensor binding has an unsupported struct size");
  }
  if (binding.abi_version != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "windowed attention tensor ABI version is unsupported");
  }
  if (binding.buffer == nullptr || binding.rank != 3U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "windowed attention tensors require a buffer and rank three");
  }
  if (binding.reserved0 != 0U || !all_zero(binding.reserved, 2U)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "windowed attention tensor reserved fields must be zero");
  }
  if (binding.dtype != SLLM_TENSOR_DTYPE_BF16) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_DTYPE,
        "windowed attention tensors must use BF16");
  }
  if (binding.encoding != SLLM_TENSOR_ENCODING_UNQUANTIZED) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_ENCODING,
        "windowed attention tensors must be unquantized");
  }
  if ((binding.byte_offset & 1U) != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_MISALIGNED_OFFSET,
        "windowed attention tensor offset must be BF16 aligned");
  }
  uint64_t stride = 1U;
  uint64_t elements = 1U;
  for (uint32_t backwards = 0U; backwards != 3U; ++backwards) {
    const uint32_t index = 2U - backwards;
    if (binding.shape[index] == 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_ZERO_EXTENT,
          "windowed attention tensor extents must be non-zero");
    }
    if (binding.stride_elements[index] != stride) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_STRIDE_MISMATCH,
          "windowed attention tensors must be contiguous");
    }
    if (multiply_overflows(stride, binding.shape[index], &stride) ||
        multiply_overflows(elements, binding.shape[index], &elements)) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_METADATA_OVERFLOW,
          "windowed attention tensor metadata overflowed u64");
    }
  }
  for (uint32_t index = 3U; index != SLLM_HIP_TENSOR_MAX_RANK; ++index) {
    if (binding.shape[index] != 0U || binding.stride_elements[index] != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
          "windowed attention unused tensor metadata must be zero");
    }
  }
  uint64_t payload_bytes = 0U;
  if (multiply_overflows(elements, UINT64_C(2), &payload_bytes) ||
      sllm_public_runtime::add_overflows(binding.byte_offset, payload_bytes)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "windowed attention tensor byte interval overflowed u64");
  }
  copied->byte_offset = binding.byte_offset;
  copied->payload_bytes = payload_bytes;
  copied->end_offset = binding.byte_offset + payload_bytes;
  copied->rank = binding.rank;
  copied->element_bytes = 2U;
  std::memcpy(copied->shape, binding.shape, sizeof(copied->shape));
  std::memcpy(copied->strides, binding.stride_elements,
              sizeof(copied->strides));
  return SLLM_STATUS_OK;
}

bool exact_shape(const TensorMetadata &tensor, const uint64_t dim0,
                 const uint64_t dim1, const uint64_t dim2) noexcept {
  return tensor.shape[0] == dim0 && tensor.shape[1] == dim1 &&
         tensor.shape[2] == dim2;
}

} // namespace

sllm_status_t validate_descriptor_prefix(
    const sllm_windowed_attention_desc_t *const descriptor,
    sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_WINDOWED_ATTENTION_DESCRIPTOR,
        "windowed attention descriptor is null");
  }
  uint32_t prefix[2]{};
  std::memcpy(prefix, descriptor, sizeof(prefix));
  if (prefix[0] != sizeof(*descriptor)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "windowed attention descriptor has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "windowed attention ABI version is unsupported");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_and_copy_descriptor(
    const sllm_windowed_attention_desc_t *const descriptor,
    DescriptorMetadata *const metadata,
    sllm_error_sink_t *const sink) noexcept {
  if (metadata == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_WINDOWED_ATTENTION_DESCRIPTOR,
        "windowed attention metadata output is null");
  }
  const sllm_status_t prefix_status =
      validate_descriptor_prefix(descriptor, sink);
  if (prefix_status != SLLM_STATUS_OK) {
    return prefix_status;
  }
  if (descriptor->reserved0 != 0U || !all_zero(descriptor->reserved, 4U)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "windowed attention reserved fields must be zero");
  }
  if (descriptor->op_version != SLLM_HIP_WINDOWED_ATTENTION_VERSION ||
      descriptor->q_heads == 0U || descriptor->kv_heads == 0U ||
      descriptor->q_heads % descriptor->kv_heads != 0U ||
      descriptor->head_dim == 0U ||
      descriptor->head_dim > SLLM_HIP_WINDOWED_ATTENTION_MAX_HEAD_DIM ||
      descriptor->scaling_bits != UINT32_C(0x3f800000) ||
      descriptor->expected_kv_length == 0U ||
      descriptor->expected_kv_length > SLLM_HIP_WINDOWED_ATTENTION_MAX_KV ||
      descriptor->start_position >= descriptor->expected_kv_length ||
      descriptor->sliding_window > SLLM_HIP_WINDOWED_ATTENTION_MAX_KV) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_WINDOWED_ATTENTION_DESCRIPTOR,
        "windowed attention operation contract is invalid or unsupported");
  }
  sllm_status_t status =
      validate_tensor(descriptor->query, &metadata->query, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(descriptor->key, &metadata->key, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(descriptor->value, &metadata->value, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(descriptor->output, &metadata->output, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  const uint64_t query_count = metadata->query.shape[0];
  if (query_count == 0U || query_count > SLLM_HIP_WINDOWED_ATTENTION_MAX_M ||
      descriptor->start_position >
          std::numeric_limits<uint64_t>::max() - query_count ||
      descriptor->start_position + query_count !=
          descriptor->expected_kv_length) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_WINDOWED_ATTENTION_DESCRIPTOR,
        "windowed attention query range does not end at the KV length");
  }
  if (!exact_shape(metadata->query, query_count, descriptor->q_heads,
                   descriptor->head_dim) ||
      !exact_shape(metadata->output, query_count, descriptor->q_heads,
                   descriptor->head_dim) ||
      !exact_shape(metadata->key, descriptor->expected_kv_length,
                   descriptor->kv_heads, descriptor->head_dim) ||
      !exact_shape(metadata->value, descriptor->expected_kv_length,
                   descriptor->kv_heads, descriptor->head_dim)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "windowed attention tensor shapes do not match the contract");
  }
  if (query_count > std::numeric_limits<uint32_t>::max() ||
      query_count >
          std::numeric_limits<uint32_t>::max() / descriptor->q_heads) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "windowed attention launch geometry exceeds u32");
  }
  metadata->start_position = descriptor->start_position;
  metadata->expected_kv_length = descriptor->expected_kv_length;
  metadata->sliding_window = descriptor->sliding_window;
  metadata->query_count = query_count;
  metadata->q_heads = descriptor->q_heads;
  metadata->kv_heads = descriptor->kv_heads;
  metadata->head_dim = descriptor->head_dim;
  metadata->scaling_bits = descriptor->scaling_bits;
  return SLLM_STATUS_OK;
}

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept {
  return left.byte_offset < right.end_offset &&
         right.byte_offset < left.end_offset;
}

} // namespace sllm_windowed_attention
