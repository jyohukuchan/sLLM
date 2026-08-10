#include "argmax_api.hpp"

#include <cstring>
#include <limits>

namespace sllm_argmax {
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
                              const uint32_t expected_rank,
                              const uint32_t expected_dtype,
                              const uint64_t element_bytes,
                              TensorMetadata *const copied,
                              sllm_error_sink_t *const sink) noexcept {
  if (binding.struct_size != sizeof(binding)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "argmax tensor binding has an unsupported struct size");
  }
  if (binding.abi_version != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "argmax tensor binding ABI version is unsupported");
  }
  if (binding.reserved0 != 0U ||
      !all_zero(binding.reserved, sizeof(binding.reserved))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "argmax tensor binding reserved fields must be zero");
  }
  if (binding.buffer == nullptr || binding.rank != expected_rank) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "argmax tensor binding has the wrong buffer or rank");
  }
  if (binding.dtype != expected_dtype) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_DTYPE,
        "argmax tensor has an unsupported dtype");
  }
  if (binding.encoding != SLLM_TENSOR_ENCODING_UNQUANTIZED) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_ENCODING,
        "argmax tensors must be unquantized");
  }
  if ((binding.byte_offset % element_bytes) != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_MISALIGNED_OFFSET,
        "argmax tensor offset does not satisfy its dtype alignment");
  }
  if (binding.shape[0] == 0U ||
      (expected_rank == 2U && binding.shape[1] == 0U)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_ZERO_EXTENT,
        "argmax tensor extents must be non-zero");
  }
  if (expected_rank == 2U) {
    if (binding.stride_elements[1] != 1U ||
        binding.stride_elements[0] != binding.shape[1]) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_STRIDE_MISMATCH,
          "argmax logits must be row-major contiguous");
    }
  } else if (binding.stride_elements[0] != 1U) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_STRIDE_MISMATCH,
                                            "argmax output must be contiguous");
  }
  for (uint32_t index = expected_rank; index != SLLM_HIP_TENSOR_MAX_RANK;
       ++index) {
    if (binding.shape[index] != 0U || binding.stride_elements[index] != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
          "argmax unused tensor metadata must be zero");
    }
  }
  uint64_t elements = binding.shape[0];
  if (expected_rank == 2U &&
      multiply_overflows(elements, binding.shape[1], &elements)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "argmax tensor element count overflowed u64");
  }
  uint64_t payload_bytes = 0U;
  if (multiply_overflows(elements, element_bytes, &payload_bytes) ||
      sllm_public_runtime::add_overflows(binding.byte_offset, payload_bytes)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "argmax tensor byte interval overflowed u64");
  }
  copied->byte_offset = binding.byte_offset;
  copied->payload_bytes = payload_bytes;
  copied->end_offset = binding.byte_offset + payload_bytes;
  return SLLM_STATUS_OK;
}

} // namespace

sllm_status_t
validate_descriptor_prefix(const sllm_argmax_desc_t *const descriptor,
                           sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGMAX_DESCRIPTOR,
        "argmax descriptor is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, descriptor, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_argmax_desc_t)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "argmax descriptor prefix has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "argmax public ABI version is unsupported");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_and_copy_descriptor(const sllm_argmax_desc_t *const descriptor,
                             DescriptorMetadata *const metadata,
                             sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr || metadata == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGMAX_DESCRIPTOR,
        "argmax descriptor or metadata output is null");
  }
  const sllm_status_t prefix_status =
      validate_descriptor_prefix(descriptor, sink);
  if (prefix_status != SLLM_STATUS_OK) {
    return prefix_status;
  }
  if (descriptor->op_version != SLLM_HIP_ARGMAX_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGMAX_DESCRIPTOR,
        "argmax descriptor version is unsupported");
  }
  if (!all_zero(descriptor->reserved, sizeof(descriptor->reserved))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "argmax descriptor reserved fields must be zero");
  }
  sllm_status_t status =
      validate_tensor(descriptor->logits, 2U, SLLM_TENSOR_DTYPE_BF16,
                      UINT64_C(2), &metadata->logits, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(descriptor->output, 1U, SLLM_TENSOR_DTYPE_I32,
                           UINT64_C(4), &metadata->output, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  metadata->m = descriptor->logits.shape[0];
  metadata->v = descriptor->logits.shape[1];
  if (metadata->v > SLLM_HIP_ARGMAX_MAX_V ||
      metadata->m > SLLM_HIP_ARGMAX_MAX_M) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED,
        "argmax shape exceeds the baseline kernel launch contract");
  }
  if (descriptor->output.shape[0] != metadata->m) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "argmax output must have shape [M] for logits [M,V]");
  }
  return SLLM_STATUS_OK;
}

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept {
  return left.byte_offset < right.end_offset &&
         right.byte_offset < left.end_offset;
}

} // namespace sllm_argmax
