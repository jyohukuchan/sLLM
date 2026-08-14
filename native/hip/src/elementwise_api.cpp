#include "elementwise_api.hpp"

#include <cstring>
#include <limits>

namespace sllm_elementwise {
namespace {

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
        "elementwise public ABI version is unsupported");
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

bool all_zero(const void *const bytes, const std::size_t size) noexcept {
  const auto *const values = static_cast<const unsigned char *>(bytes);
  for (std::size_t index = 0U; index != size; ++index) {
    if (values[index] != 0U) {
      return false;
    }
  }
  return true;
}

sllm_status_t validate_tensor(const sllm_tensor_binding_t &binding,
                              TensorMetadata *const copied,
                              sllm_error_sink_t *const sink) noexcept {
  if (copied == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "elementwise tensor metadata output is null");
  }
  const sllm_status_t struct_status = exact_struct(
      binding.struct_size, binding.abi_version, sizeof(binding), sink,
      "elementwise tensor binding has an unsupported struct size");
  if (struct_status != SLLM_STATUS_OK) {
    return struct_status;
  }
  if (binding.reserved0 != 0U || binding.reserved[0] != 0U ||
      binding.reserved[1] != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "elementwise tensor binding reserved fields must be zero");
  }
  if (binding.buffer == nullptr || binding.rank == 0U ||
      binding.rank > SLLM_HIP_TENSOR_MAX_RANK) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "elementwise tensor binding requires a buffer and rank in 1..=8");
  }
  if (binding.dtype != SLLM_TENSOR_DTYPE_BF16) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_DTYPE,
        "elementwise tensors must use BF16 storage");
  }
  if (binding.encoding != SLLM_TENSOR_ENCODING_UNQUANTIZED) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_ENCODING,
        "elementwise tensors must be unquantized");
  }
  if ((binding.byte_offset & UINT64_C(1)) != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_MISALIGNED_OFFSET,
        "elementwise BF16 tensor offset must be two-byte aligned");
  }

  uint64_t expected_stride = 1U;
  uint64_t elements = 1U;
  for (uint32_t backwards = 0U; backwards != binding.rank; ++backwards) {
    const uint32_t index = binding.rank - 1U - backwards;
    const uint64_t extent = binding.shape[index];
    if (extent == 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_ZERO_EXTENT,
          "elementwise tensor extents must be non-zero");
    }
    if (binding.stride_elements[index] != expected_stride) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_STRIDE_MISMATCH,
          "elementwise tensors must be row-major contiguous");
    }
    if (multiply_overflows(expected_stride, extent, &expected_stride) ||
        multiply_overflows(elements, extent, &elements)) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_METADATA_OVERFLOW,
          "elementwise tensor metadata overflowed u64");
    }
  }
  for (uint32_t index = binding.rank; index != SLLM_HIP_TENSOR_MAX_RANK;
       ++index) {
    if (binding.shape[index] != 0U || binding.stride_elements[index] != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
          "elementwise unused tensor metadata must be zero");
    }
  }
  uint64_t payload_bytes = 0U;
  if (multiply_overflows(elements, UINT64_C(2), &payload_bytes) ||
      sllm_public_runtime::add_overflows(binding.byte_offset, payload_bytes)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "elementwise tensor byte interval overflowed u64");
  }

  copied->byte_offset = binding.byte_offset;
  copied->payload_bytes = payload_bytes;
  copied->end_offset = binding.byte_offset + payload_bytes;
  copied->rank = binding.rank;
  std::memcpy(copied->shape, binding.shape, sizeof(copied->shape));
  std::memcpy(copied->strides, binding.stride_elements,
              sizeof(copied->strides));
  return SLLM_STATUS_OK;
}

bool equal_layout(const TensorMetadata &left,
                  const TensorMetadata &right) noexcept {
  return left.rank == right.rank &&
         std::memcmp(left.shape, right.shape, sizeof(left.shape)) == 0 &&
         std::memcmp(left.strides, right.strides, sizeof(left.strides)) == 0;
}

} // namespace

sllm_status_t
validate_descriptor_prefix(const sllm_elementwise_desc_t *const descriptor,
                           sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ELEMENTWISE_DESCRIPTOR,
        "elementwise descriptor is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, descriptor, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_elementwise_desc_t)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "elementwise descriptor prefix has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "elementwise public ABI version is unsupported");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_and_copy_descriptor(const sllm_elementwise_desc_t *const descriptor,
                             DescriptorMetadata *const metadata,
                             sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr || metadata == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ELEMENTWISE_DESCRIPTOR,
        "elementwise descriptor or metadata output is null");
  }
  const sllm_status_t prefix_status =
      validate_descriptor_prefix(descriptor, sink);
  if (prefix_status != SLLM_STATUS_OK) {
    return prefix_status;
  }
  if (descriptor->op_version != SLLM_HIP_ELEMENTWISE_VERSION ||
      (descriptor->operation != SLLM_ELEMENTWISE_OPERATION_COPY &&
       descriptor->operation != SLLM_ELEMENTWISE_OPERATION_ADD &&
       descriptor->operation != SLLM_ELEMENTWISE_OPERATION_SILU_MUL &&
       descriptor->operation != SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ELEMENTWISE_DESCRIPTOR,
        "elementwise descriptor has an unsupported operation contract");
  }
  if (!all_zero(descriptor->reserved, sizeof(descriptor->reserved))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "elementwise descriptor reserved fields must be zero");
  }
  if (descriptor->operation == SLLM_ELEMENTWISE_OPERATION_COPY &&
      !all_zero(&descriptor->input1, sizeof(descriptor->input1))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ELEMENTWISE_DESCRIPTOR,
        "copy requires a zero-initialized second input binding");
  }

  sllm_status_t status =
      validate_tensor(descriptor->input0, &metadata->input0, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (descriptor->operation != SLLM_ELEMENTWISE_OPERATION_COPY) {
    status = validate_tensor(descriptor->input1, &metadata->input1, sink);
    if (status != SLLM_STATUS_OK) {
      return status;
    }
  } else {
    metadata->input1 = {};
  }
  status = validate_tensor(descriptor->output, &metadata->output, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (!equal_layout(metadata->input0, metadata->output) ||
      (descriptor->operation != SLLM_ELEMENTWISE_OPERATION_COPY &&
       !equal_layout(metadata->input0, metadata->input1))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "elementwise operands must have exactly equal layouts");
  }
  if (descriptor->operation == SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL &&
      (metadata->input0.rank != 3U ||
       (metadata->input0.shape[1] != 8U && metadata->input0.shape[1] != 16U) ||
       metadata->input0.shape[2] != 256U)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "sigmoid multiply requires a reviewed contiguous BF16 [M,H,256] "
        "layout");
  }
  const auto overlaps = [](const sllm_tensor_binding_t &left_binding,
                           const TensorMetadata &left,
                           const sllm_tensor_binding_t &right_binding,
                           const TensorMetadata &right) {
    return left_binding.buffer == right_binding.buffer &&
           intervals_overlap(left, right);
  };
  if (overlaps(descriptor->input0, metadata->input0, descriptor->output,
               metadata->output) ||
      (descriptor->operation != SLLM_ELEMENTWISE_OPERATION_COPY &&
       (overlaps(descriptor->input0, metadata->input0, descriptor->input1,
                 metadata->input1) ||
        overlaps(descriptor->input1, metadata->input1, descriptor->output,
                 metadata->output)))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_ALIAS_OVERLAP,
        "elementwise tensor intervals overlap within one binding identity");
  }
  metadata->element_count = metadata->input0.payload_bytes / UINT64_C(2);
  metadata->operation = descriptor->operation;
  return SLLM_STATUS_OK;
}

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept {
  return left.byte_offset < right.end_offset &&
         right.byte_offset < left.end_offset;
}

} // namespace sllm_elementwise
