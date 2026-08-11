#include "embedding_api.hpp"

#include <cstring>
#include <limits>

namespace sllm_embedding {
namespace {

bool all_zero(const void *const bytes, const std::size_t size) noexcept {
  const auto *const values = static_cast<const unsigned char *>(bytes);
  for (std::size_t index = 0U; index != size; ++index) {
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

sllm_status_t validate_tensor(const sllm_tensor_binding_t &binding,
                              const uint32_t expected_dtype,
                              const uint32_t expected_rank,
                              TensorMetadata *const copied,
                              sllm_error_sink_t *const sink) noexcept {
  if (binding.struct_size != sizeof(binding)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "embedding tensor binding has an unsupported struct size");
  }
  if (binding.abi_version != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "embedding tensor binding ABI version is unsupported");
  }
  if (binding.reserved0 != 0U ||
      !all_zero(binding.reserved, sizeof(binding.reserved))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "embedding tensor binding reserved fields must be zero");
  }
  if (binding.buffer == nullptr || binding.rank != expected_rank) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "embedding tensor binding has the wrong rank or no buffer");
  }
  if (binding.dtype != expected_dtype) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_DTYPE,
        "embedding tensor binding has an unsupported dtype");
  }
  if (binding.encoding != SLLM_TENSOR_ENCODING_UNQUANTIZED) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_ENCODING,
        "embedding tensors must be unquantized");
  }
  const uint64_t element_bytes =
      expected_dtype == SLLM_TENSOR_DTYPE_I32 ? UINT64_C(4) : UINT64_C(2);
  if (binding.byte_offset % element_bytes != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_MISALIGNED_OFFSET,
        "embedding tensor offset is not naturally aligned");
  }
  uint64_t elements = 1U;
  uint64_t expected_stride = 1U;
  for (uint32_t backwards = 0U; backwards != expected_rank; ++backwards) {
    const uint32_t index = expected_rank - 1U - backwards;
    if (binding.shape[index] == 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_ZERO_EXTENT,
          "embedding tensor extents must be non-zero");
    }
    if (binding.stride_elements[index] != expected_stride) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_STRIDE_MISMATCH,
          "embedding tensors must be row-major contiguous");
    }
    if (multiply_overflows(expected_stride, binding.shape[index],
                           &expected_stride) ||
        multiply_overflows(elements, binding.shape[index], &elements)) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_METADATA_OVERFLOW,
          "embedding tensor metadata overflowed u64");
    }
  }
  for (uint32_t index = expected_rank; index != SLLM_HIP_TENSOR_MAX_RANK;
       ++index) {
    if (binding.shape[index] != 0U || binding.stride_elements[index] != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
          "embedding unused tensor metadata must be zero");
    }
  }
  uint64_t payload_bytes = 0U;
  if (multiply_overflows(elements, element_bytes, &payload_bytes) ||
      sllm_public_runtime::add_overflows(binding.byte_offset, payload_bytes)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "embedding tensor byte interval overflowed u64");
  }
  copied->byte_offset = binding.byte_offset;
  copied->payload_bytes = payload_bytes;
  copied->end_offset = binding.byte_offset + payload_bytes;
  copied->shape[0] = binding.shape[0];
  copied->shape[1] = expected_rank == 2U ? binding.shape[1] : 0U;
  return SLLM_STATUS_OK;
}

} // namespace

sllm_status_t
validate_descriptor_prefix(const sllm_embedding_desc_t *const descriptor,
                           sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_EMBEDDING_DESCRIPTOR,
        "embedding descriptor is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, descriptor, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_embedding_desc_t)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "embedding descriptor prefix has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "embedding public ABI version is unsupported");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_and_copy_descriptor(const sllm_embedding_desc_t *const descriptor,
                             DescriptorMetadata *const metadata,
                             sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr || metadata == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_EMBEDDING_DESCRIPTOR,
        "embedding descriptor or metadata output is null");
  }
  const sllm_status_t prefix_status =
      validate_descriptor_prefix(descriptor, sink);
  if (prefix_status != SLLM_STATUS_OK) {
    return prefix_status;
  }
  if (descriptor->op_version != SLLM_HIP_EMBEDDING_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_EMBEDDING_DESCRIPTOR,
        "embedding descriptor version is unsupported");
  }
  if (!all_zero(descriptor->reserved, sizeof(descriptor->reserved))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "embedding descriptor reserved fields must be zero");
  }
  sllm_status_t status = validate_tensor(
      descriptor->weight, SLLM_TENSOR_DTYPE_BF16, 2U, &metadata->weight, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(descriptor->token_ids, SLLM_TENSOR_DTYPE_I32, 1U,
                           &metadata->token_ids, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(descriptor->output, SLLM_TENSOR_DTYPE_BF16, 2U,
                           &metadata->output, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  metadata->vocab_size = metadata->weight.shape[0];
  metadata->hidden_size = metadata->weight.shape[1];
  metadata->token_count = metadata->token_ids.shape[0];
  if (metadata->output.shape[0] != metadata->token_count ||
      metadata->output.shape[1] != metadata->hidden_size) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "embedding output shape must be [tokens, hidden]");
  }
  if (multiply_overflows(metadata->token_count, metadata->hidden_size,
                         &metadata->output_elements)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "embedding output element count overflowed u64");
  }
  return SLLM_STATUS_OK;
}

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept {
  return left.byte_offset < right.end_offset &&
         right.byte_offset < left.end_offset;
}

} // namespace sllm_embedding
