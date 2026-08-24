#include "attention_preprocess_api.hpp"

#include <cstring>
#include <limits>

namespace sllm_attention_preprocess {
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
        "attention preprocess public ABI version is unsupported");
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
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "attention preprocess tensor binding is null");
  }
  const sllm_status_t struct_status = exact_struct(
      binding->struct_size, binding->abi_version, sizeof(*binding), sink,
      "attention preprocess tensor binding has an unsupported struct size");
  if (struct_status != SLLM_STATUS_OK) {
    return struct_status;
  }
  if (binding->reserved0 != 0U || !all_zero(binding->reserved, 2U)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "attention preprocess tensor binding reserved fields must be zero");
  }
  if (binding->buffer == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "attention preprocess tensor binding buffer is null");
  }
  if (binding->rank != expected_rank) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_SHAPE_MISMATCH,
                                            name);
  }
  if (binding->dtype != expected_dtype) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_DTYPE,
        "attention preprocess tensor has an unsupported dtype");
  }
  if (binding->encoding != SLLM_TENSOR_ENCODING_UNQUANTIZED) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_ENCODING,
        "attention preprocess tensors must be unquantized");
  }
  const uint64_t element_bytes =
      expected_dtype == SLLM_TENSOR_DTYPE_I32 ? UINT64_C(4) : UINT64_C(2);
  if ((binding->byte_offset % element_bytes) != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_MISALIGNED_OFFSET,
        "attention preprocess tensor offset is not element aligned");
  }

  uint64_t expected_stride = 1U;
  uint64_t elements = 1U;
  for (uint32_t backwards = 0U; backwards != binding->rank; ++backwards) {
    const uint32_t index = binding->rank - 1U - backwards;
    const uint64_t extent = binding->shape[index];
    if (extent == 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_ZERO_EXTENT,
          "attention preprocess tensor extents must be non-zero");
    }
    if (binding->stride_elements[index] != expected_stride) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_STRIDE_MISMATCH,
          "attention preprocess tensors must be row-major contiguous");
    }
    if (multiply_overflows(expected_stride, extent, &expected_stride) ||
        multiply_overflows(elements, extent, &elements)) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_METADATA_OVERFLOW,
          "attention preprocess tensor metadata overflowed u64");
    }
  }
  for (uint32_t index = binding->rank; index != SLLM_HIP_TENSOR_MAX_RANK;
       ++index) {
    if (binding->shape[index] != 0U || binding->stride_elements[index] != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
          "attention preprocess unused tensor metadata must be zero");
    }
  }
  uint64_t payload_bytes = 0U;
  if (multiply_overflows(elements, element_bytes, &payload_bytes) ||
      sllm_public_runtime::add_overflows(binding->byte_offset, payload_bytes)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "attention preprocess tensor byte interval overflowed u64");
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

sllm_status_t validate_descriptor_prefix(
    const sllm_attention_preprocess_desc_t *const descriptor,
    sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ATTENTION_PREPROCESS_DESCRIPTOR,
        "attention preprocess descriptor is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, descriptor, sizeof(prefix));
  if (prefix[0] < sizeof(prefix) ||
      prefix[0] != sizeof(sllm_attention_preprocess_desc_t)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "attention preprocess descriptor prefix has an unsupported struct "
        "size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "attention preprocess public ABI version is unsupported");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_and_copy_descriptor(
    const sllm_attention_preprocess_desc_t *const descriptor,
    DescriptorMetadata *const metadata,
    sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr || metadata == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ATTENTION_PREPROCESS_DESCRIPTOR,
        "attention preprocess descriptor or metadata output is null");
  }
  const sllm_status_t prefix_status =
      validate_descriptor_prefix(descriptor, sink);
  if (prefix_status != SLLM_STATUS_OK) {
    return prefix_status;
  }
  const sllm_status_t struct_status = exact_struct(
      descriptor->struct_size, descriptor->abi_version, sizeof(*descriptor),
      sink, "attention preprocess descriptor has an unsupported struct size");
  if (struct_status != SLLM_STATUS_OK) {
    return struct_status;
  }
  if ((descriptor->reserved[0] !=
           SLLM_HIP_POSITION_PAYLOAD_MODE_CONTIGUOUS_V1 &&
       descriptor->reserved[0] != SLLM_HIP_POSITION_PAYLOAD_MODE_EXPLICIT_V1 &&
       descriptor->reserved[0] !=
           SLLM_HIP_POSITION_PAYLOAD_MODE_DERIVED_CONTIGUOUS_V1) ||
      !all_zero(descriptor->reserved + 1, 3U)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "attention preprocess descriptor reserved fields must be zero");
  }
  if (descriptor->op_version != SLLM_HIP_ATTENTION_PREPROCESS_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ATTENTION_PREPROCESS_DESCRIPTOR,
        "attention preprocess operation version is unsupported");
  }
  if (descriptor->start_position >=
      SLLM_HIP_ATTENTION_PREPROCESS_MAX_POSITION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ATTENTION_PREPROCESS_DESCRIPTOR,
        "attention preprocess start position is outside the text position "
        "range");
  }

  sllm_status_t status =
      validate_tensor(&descriptor->packed_q_gate, SLLM_TENSOR_DTYPE_BF16, 3U,
                      &metadata->packed_q_gate, sink,
                      "packed Q/gate tensor must have rank three");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(&descriptor->k, SLLM_TENSOR_DTYPE_BF16, 3U,
                           &metadata->k, sink, "K tensor must have rank three");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(&descriptor->q_raw_scale, SLLM_TENSOR_DTYPE_BF16, 2U,
                           &metadata->q_raw_scale, sink,
                           "Q raw norm scale must have rank two");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(&descriptor->k_raw_scale, SLLM_TENSOR_DTYPE_BF16, 2U,
                           &metadata->k_raw_scale, sink,
                           "K raw norm scale must have rank two");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (descriptor->positions.rank != 1U && descriptor->positions.rank != 2U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "positions tensor must have rank one or rank two");
  }
  status = validate_tensor(&descriptor->positions, SLLM_TENSOR_DTYPE_I32,
                           descriptor->positions.rank, &metadata->positions,
                           sink, "positions tensor rank differs");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(&descriptor->q_output, SLLM_TENSOR_DTYPE_BF16, 3U,
                           &metadata->q_output, sink,
                           "Q output tensor must have rank three");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(&descriptor->gate_output, SLLM_TENSOR_DTYPE_BF16, 3U,
                           &metadata->gate_output, sink,
                           "gate output tensor must have rank three");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(&descriptor->k_output, SLLM_TENSOR_DTYPE_BF16, 3U,
                           &metadata->k_output, sink,
                           "K output tensor must have rank three");
  if (status != SLLM_STATUS_OK) {
    return status;
  }

  const uint64_t m = metadata->packed_q_gate.shape[0];
  const uint64_t q_heads = metadata->packed_q_gate.shape[1];
  const uint64_t k_heads = metadata->k.shape[1];
  const uint64_t head_dim = metadata->k.shape[2];
  const uint64_t packed_shape[] = {m, q_heads, head_dim * 2U};
  const uint64_t k_shape[] = {m, k_heads, head_dim};
  const uint64_t scale_q_shape[] = {q_heads, head_dim};
  const uint64_t scale_k_shape[] = {k_heads, head_dim};
  const uint64_t position_shape[] = {m};
  const uint64_t mrope_position_shape[] = {m, 3U};
  const uint64_t q_output_shape[] = {m, q_heads, head_dim};
  if (m == 0U || m > SLLM_HIP_ATTENTION_PREPROCESS_MAX_M ||
      (q_heads != 8U && q_heads != 16U) || (k_heads != 2U && k_heads != 4U) ||
      q_heads % k_heads != 0U ||
      head_dim != SLLM_HIP_ATTENTION_PREPROCESS_Q_HEAD_DIM ||
      static_cast<uint64_t>(descriptor->start_position) + m >
          SLLM_HIP_ATTENTION_PREPROCESS_MAX_POSITION ||
      (descriptor->reserved[0] ==
           SLLM_HIP_POSITION_PAYLOAD_MODE_DERIVED_CONTIGUOUS_V1 &&
       static_cast<uint64_t>(descriptor->start_position) + m - 1U >
           static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) ||
      !exact_shape(metadata->packed_q_gate, 3U, packed_shape) ||
      !exact_shape(metadata->k, 3U, k_shape) ||
      !exact_shape(metadata->q_raw_scale, 2U, scale_q_shape) ||
      !exact_shape(metadata->k_raw_scale, 2U, scale_k_shape) ||
      (!exact_shape(metadata->positions, 1U, position_shape) &&
       !exact_shape(metadata->positions, 2U, mrope_position_shape)) ||
      (descriptor->reserved[0] ==
           SLLM_HIP_POSITION_PAYLOAD_MODE_DERIVED_CONTIGUOUS_V1 &&
       !exact_shape(metadata->positions, 1U, position_shape)) ||
      !exact_shape(metadata->q_output, 3U, q_output_shape) ||
      !exact_shape(metadata->gate_output, 3U, q_output_shape) ||
      !exact_shape(metadata->k_output, 3U, k_shape)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "attention preprocess tensors do not match the fixed head-wise layout");
  }
  metadata->m = m;
  metadata->start_position = descriptor->start_position;
  metadata->q_heads = static_cast<uint32_t>(q_heads);
  metadata->k_heads = static_cast<uint32_t>(k_heads);
  metadata->head_dim = static_cast<uint32_t>(head_dim);
  metadata->position_components = metadata->positions.rank == 2U ? 3U : 1U;
  metadata->position_payload_mode = descriptor->reserved[0];
  return SLLM_STATUS_OK;
}

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept {
  return left.byte_offset < right.end_offset &&
         right.byte_offset < left.end_offset;
}

} // namespace sllm_attention_preprocess
