#include "rmsnorm_api.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace sllm_rmsnorm {
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
        "RMSNorm public ABI version is unsupported");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_descriptor_prefix_impl(const sllm_rmsnorm_desc_t *const descriptor,
                                sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR,
        "RMSNorm descriptor is null");
  }

  uint32_t prefix[2] = {};
  std::memcpy(prefix, descriptor, sizeof(prefix));
  if (prefix[0] < static_cast<uint32_t>(sizeof(prefix)) ||
      prefix[0] != static_cast<uint32_t>(sizeof(sllm_rmsnorm_desc_t))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "RMSNorm descriptor prefix has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "RMSNorm public ABI version is unsupported");
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

sllm_status_t
validate_tensor_binding_impl(const sllm_tensor_binding_t *const binding,
                             TensorMetadata *const copied,
                             sllm_error_sink_t *const sink) noexcept {
  if (binding == nullptr || copied == nullptr) {
    return sllm_public_runtime::write_error(sink,
                                            SLLM_STATUS_INVALID_TENSOR_BINDING,
                                            "RMSNorm tensor binding is null");
  }
  const sllm_status_t struct_status = exact_struct(
      binding->struct_size, binding->abi_version, sizeof(*binding), sink,
      "RMSNorm tensor binding has an unsupported struct size");
  if (struct_status != SLLM_STATUS_OK) {
    return struct_status;
  }
  if (binding->reserved0 != 0U || binding->reserved[0] != 0U ||
      binding->reserved[1] != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "RMSNorm tensor binding reserved fields must be zero");
  }
  if (binding->buffer == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "RMSNorm tensor binding buffer is null");
  }
  if (binding->rank == 0U || binding->rank > SLLM_HIP_TENSOR_MAX_RANK) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "RMSNorm tensor binding rank must be in 1..=8");
  }
  if (binding->dtype != SLLM_TENSOR_DTYPE_BF16) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_DTYPE,
        "RMSNorm tensors must use BF16 storage");
  }
  if (binding->encoding != SLLM_TENSOR_ENCODING_UNQUANTIZED) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_ENCODING,
        "RMSNorm tensors must be unquantized");
  }
  if ((binding->byte_offset & UINT64_C(1)) != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_MISALIGNED_OFFSET,
        "RMSNorm BF16 tensor offset must be two-byte aligned");
  }

  uint64_t expected_stride = 1U;
  uint64_t elements = 1U;
  for (uint32_t backwards = 0U; backwards != binding->rank; ++backwards) {
    const uint32_t index = binding->rank - 1U - backwards;
    const uint64_t extent = binding->shape[index];
    if (extent == 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_ZERO_EXTENT,
          "RMSNorm tensor extents must be non-zero");
    }
    if (binding->stride_elements[index] != expected_stride) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_STRIDE_MISMATCH,
          "RMSNorm tensors must be row-major contiguous");
    }
    if (multiply_overflows(expected_stride, extent, &expected_stride) ||
        multiply_overflows(elements, extent, &elements)) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_METADATA_OVERFLOW,
          "RMSNorm tensor metadata overflowed u64");
    }
  }
  for (uint32_t index = binding->rank; index != SLLM_HIP_TENSOR_MAX_RANK;
       ++index) {
    if (binding->shape[index] != 0U || binding->stride_elements[index] != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
          "RMSNorm unused tensor metadata must be zero");
    }
  }
  uint64_t payload_bytes = 0U;
  if (multiply_overflows(elements, UINT64_C(2), &payload_bytes) ||
      sllm_public_runtime::add_overflows(binding->byte_offset, payload_bytes)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "RMSNorm tensor byte interval overflowed u64");
  }

  copied->byte_offset = binding->byte_offset;
  copied->payload_bytes = payload_bytes;
  copied->end_offset = binding->byte_offset + payload_bytes;
  copied->rank = binding->rank;
  std::memcpy(copied->shape, binding->shape, sizeof(copied->shape));
  std::memcpy(copied->strides, binding->stride_elements,
              sizeof(copied->strides));
  return SLLM_STATUS_OK;
}

bool equal_shape(const TensorMetadata &left,
                 const TensorMetadata &right) noexcept {
  return left.rank == right.rank &&
         std::memcmp(left.shape, right.shape, sizeof(left.shape)) == 0 &&
         std::memcmp(left.strides, right.strides, sizeof(left.strides)) == 0;
}

} // namespace

sllm_status_t
validate_tensor_binding(const sllm_tensor_binding_t *const binding,
                        TensorMetadata *const metadata,
                        sllm_error_sink_t *const sink) noexcept {
  return validate_tensor_binding_impl(binding, metadata, sink);
}

sllm_status_t
validate_descriptor_prefix(const sllm_rmsnorm_desc_t *const descriptor,
                           sllm_error_sink_t *const sink) noexcept {
  return validate_descriptor_prefix_impl(descriptor, sink);
}

sllm_status_t
validate_and_copy_descriptor(const sllm_rmsnorm_desc_t *const descriptor,
                             DescriptorMetadata *const metadata,
                             sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr || metadata == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR,
        "RMSNorm descriptor or metadata output is null");
  }
  const sllm_status_t prefix_status =
      validate_descriptor_prefix_impl(descriptor, sink);
  if (prefix_status != SLLM_STATUS_OK) {
    return prefix_status;
  }
  const sllm_status_t struct_status = exact_struct(
      descriptor->struct_size, descriptor->abi_version, sizeof(*descriptor),
      sink, "RMSNorm descriptor has an unsupported struct size");
  if (struct_status != SLLM_STATUS_OK) {
    return struct_status;
  }
  if (descriptor->reserved[0] != 0U || descriptor->reserved[1] != 0U ||
      descriptor->reserved[2] != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "RMSNorm descriptor reserved fields must be zero");
  }
  if (descriptor->op_version != SLLM_HIP_RMSNORM_VERSION ||
      descriptor->accumulation_dtype != SLLM_RMSNORM_ACCUMULATION_F32 ||
      descriptor->alias_policy != SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR,
        "RMSNorm descriptor has an unsupported operation contract");
  }
  if (descriptor->scale_mode != SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE &&
      descriptor->scale_mode != SLLM_RMSNORM_SCALE_MODE_DIRECT) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_SCALE_MODE,
        "RMSNorm scale mode is unsupported");
  }
  float epsilon = 0.0F;
  std::memcpy(&epsilon, &descriptor->epsilon_bits, sizeof(epsilon));
  if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_EPSILON,
        "RMSNorm epsilon must be finite and positive");
  }
  sllm_status_t status = validate_tensor_binding(&descriptor->activation,
                                                 &metadata->activation, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor_binding(&descriptor->raw_scale, &metadata->raw_scale,
                                   sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status =
      validate_tensor_binding(&descriptor->output, &metadata->output, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (!equal_shape(metadata->activation, metadata->output)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "RMSNorm activation and output must have exactly equal layouts");
  }
  if (metadata->raw_scale.rank != 1U ||
      metadata->raw_scale.shape[0] !=
          metadata->activation.shape[metadata->activation.rank - 1U]) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "RMSNorm raw scale must be rank one and match the final dimension");
  }
  const auto overlaps_if_same_binding =
      [](const sllm_tensor_binding_t &left, const TensorMetadata &left_metadata,
         const sllm_tensor_binding_t &right,
         const TensorMetadata &right_metadata) {
        return left.buffer == right.buffer &&
               intervals_overlap(left_metadata, right_metadata);
      };
  if (overlaps_if_same_binding(descriptor->activation, metadata->activation,
                               descriptor->raw_scale, metadata->raw_scale) ||
      overlaps_if_same_binding(descriptor->activation, metadata->activation,
                               descriptor->output, metadata->output) ||
      overlaps_if_same_binding(descriptor->raw_scale, metadata->raw_scale,
                               descriptor->output, metadata->output)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_ALIAS_OVERLAP,
        "RMSNorm tensor intervals overlap within one binding identity");
  }
  metadata->epsilon_bits = descriptor->epsilon_bits;
  metadata->scale_mode = descriptor->scale_mode;
  return SLLM_STATUS_OK;
}

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept {
  return left.byte_offset < right.end_offset &&
         right.byte_offset < left.end_offset;
}

} // namespace sllm_rmsnorm
