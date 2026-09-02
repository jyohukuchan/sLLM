#include "linear_attention_api.hpp"

#include <array>
#include <cstring>
#include <limits>

namespace sllm_linear_attention {
namespace {

bool all_zero(const void *const bytes, const std::size_t count) noexcept {
  const auto *const values = static_cast<const unsigned char *>(bytes);
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

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept {
  return left.byte_offset < right.end_offset &&
         right.byte_offset < left.end_offset;
}

sllm_status_t validate_tensor(const sllm_tensor_binding_t &binding,
                              const uint32_t dtype, const uint32_t rank,
                              const uint64_t *const expected_shape,
                              TensorMetadata *const metadata,
                              sllm_error_sink_t *const sink,
                              const char *const role) noexcept {
  if (metadata == nullptr || binding.buffer == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "linear attention tensor binding requires a buffer");
  }
  if (binding.struct_size != sizeof(binding)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "linear attention tensor binding has an unsupported struct size");
  }
  if (binding.abi_version != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "linear attention tensor binding ABI version is unsupported");
  }
  if (binding.reserved0 != 0U ||
      !all_zero(binding.reserved, sizeof(binding.reserved))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "linear attention tensor binding reserved fields must be zero");
  }
  if (binding.dtype != dtype) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_DTYPE,
        "linear attention tensor binding has an unsupported dtype");
  }
  if (binding.encoding != SLLM_TENSOR_ENCODING_UNQUANTIZED) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_ENCODING,
        "linear attention tensors must be unquantized");
  }
  if (binding.rank != rank) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "linear attention tensor rank does not match its fixed role");
  }
  const uint64_t element_bytes = dtype == SLLM_TENSOR_DTYPE_F32 ? 4U : 2U;
  if ((binding.byte_offset % element_bytes) != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_MISALIGNED_OFFSET,
        "linear attention tensor offset is not dtype-aligned");
  }
  uint64_t elements = 1U;
  uint64_t expected_stride = 1U;
  for (uint32_t backwards = 0U; backwards != rank; ++backwards) {
    const uint32_t index = rank - 1U - backwards;
    if (binding.shape[index] != expected_shape[index]) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_SHAPE_MISMATCH,
          "linear attention tensor shape does not match its fixed role");
    }
    if (binding.shape[index] == 0U ||
        binding.stride_elements[index] != expected_stride ||
        multiply_overflows(elements, binding.shape[index], &elements) ||
        multiply_overflows(expected_stride, binding.shape[index],
                           &expected_stride)) {
      return sllm_public_runtime::write_error(
          sink,
          binding.shape[index] == 0U ? SLLM_STATUS_ZERO_EXTENT
                                     : SLLM_STATUS_STRIDE_MISMATCH,
          "linear attention tensors must be contiguous and non-empty");
    }
  }
  for (uint32_t index = rank; index != SLLM_HIP_TENSOR_MAX_RANK; ++index) {
    if (binding.shape[index] != 0U || binding.stride_elements[index] != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
          "linear attention unused tensor metadata must be zero");
    }
  }
  uint64_t payload_bytes = 0U;
  if (multiply_overflows(elements, element_bytes, &payload_bytes) ||
      sllm_public_runtime::add_overflows(binding.byte_offset, payload_bytes)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "linear attention tensor interval overflowed u64");
  }
  metadata->buffer = binding.buffer;
  metadata->byte_offset = binding.byte_offset;
  metadata->payload_bytes = payload_bytes;
  metadata->end_offset = binding.byte_offset + payload_bytes;
  (void)role;
  return SLLM_STATUS_OK;
}

} // namespace

sllm_status_t validate_state_create_info(
    const sllm_linear_attention_state_create_info_t *const info,
    sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_LINEAR_ATTENTION_STATE_DESCRIPTOR,
        "linear attention state create info is null");
  }
  if (info->struct_size != sizeof(*info)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "linear attention state create info has an unsupported struct size");
  }
  if (info->abi_version != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "linear attention state ABI version is unsupported");
  }
  const uint32_t qk_heads = info->qk_heads == 0U
                                ? SLLM_HIP_LINEAR_ATTENTION_QK_HEADS
                                : info->qk_heads;
  const uint32_t value_heads = info->value_heads == 0U
                                   ? SLLM_HIP_LINEAR_ATTENTION_VALUE_HEADS
                                   : info->value_heads;
  const uint32_t head_dim = info->head_dim == 0U
                                ? SLLM_HIP_LINEAR_ATTENTION_HEAD_DIM
                                : info->head_dim;
  const uint32_t conv_kernel_size =
      info->conv_kernel_size == 0U ? SLLM_HIP_LINEAR_ATTENTION_CONV_KERNEL_SIZE
                                   : info->conv_kernel_size;
  if (info->session_id == 0U || info->flags != 0U ||
      info->capacity_tokens == 0U ||
      info->capacity_tokens > SLLM_HIP_LINEAR_ATTENTION_MAX_CAPACITY ||
      qk_heads != SLLM_HIP_LINEAR_ATTENTION_QK_HEADS ||
      (value_heads != 16U && value_heads != 32U && value_heads != 48U) ||
      head_dim != SLLM_HIP_LINEAR_ATTENTION_HEAD_DIM ||
      conv_kernel_size != SLLM_HIP_LINEAR_ATTENTION_CONV_KERNEL_SIZE) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_LINEAR_ATTENTION_STATE_DESCRIPTOR,
        "linear attention state has an invalid session, flags, or capacity");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_descriptor_prefix(const sllm_linear_attention_desc_t *const descriptor,
                           sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_LINEAR_ATTENTION_DESCRIPTOR,
        "linear attention descriptor is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, descriptor, sizeof(prefix));
  if (prefix[0] != sizeof(*descriptor)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "linear attention descriptor has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "linear attention descriptor ABI version is unsupported");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_and_copy_descriptor(
    const sllm_linear_attention_desc_t *const descriptor,
    DescriptorMetadata *const metadata,
    sllm_error_sink_t *const sink) noexcept {
  const sllm_status_t prefix = validate_descriptor_prefix(descriptor, sink);
  if (prefix != SLLM_STATUS_OK || metadata == nullptr) {
    return prefix != SLLM_STATUS_OK
               ? prefix
               : sllm_public_runtime::write_error(
                     sink, SLLM_STATUS_INVALID_LINEAR_ATTENTION_DESCRIPTOR,
                     "linear attention metadata output is null");
  }
  if (descriptor->op_version != SLLM_HIP_LINEAR_ATTENTION_VERSION ||
      descriptor->state == nullptr || descriptor->reserved0 != 0U ||
      !all_zero(descriptor->reserved, sizeof(descriptor->reserved))) {
    return sllm_public_runtime::write_error(
        sink,
        descriptor->reserved0 != 0U ||
                !all_zero(descriptor->reserved, sizeof(descriptor->reserved))
            ? SLLM_STATUS_RESERVED_NONZERO
            : SLLM_STATUS_INVALID_LINEAR_ATTENTION_DESCRIPTOR,
        "linear attention descriptor version, state, or reserved fields are "
        "invalid");
  }
  const uint64_t token_count = descriptor->qkv.shape[0];
  if (token_count == 0U || token_count > SLLM_HIP_LINEAR_ATTENTION_MAX_M ||
      descriptor->start_position >
          std::numeric_limits<uint64_t>::max() - token_count ||
      descriptor->start_position + token_count != descriptor->expected_length ||
      descriptor->expected_length > SLLM_HIP_LINEAR_ATTENTION_MAX_CAPACITY) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_LINEAR_ATTENTION_LENGTH_MISMATCH,
        "linear attention token interval is invalid or exceeds capacity");
  }

  const uint64_t value_heads =
      descriptor->z.shape[1] / SLLM_HIP_LINEAR_ATTENTION_HEAD_DIM;
  if ((value_heads != 16U && value_heads != 32U && value_heads != 48U) ||
      descriptor->z.shape[1] % SLLM_HIP_LINEAR_ATTENTION_HEAD_DIM != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "linear attention output width is not a reviewed value-head layout");
  }
  const uint64_t qkv_width =
      (2U * SLLM_HIP_LINEAR_ATTENTION_QK_HEADS + value_heads) *
      SLLM_HIP_LINEAR_ATTENTION_HEAD_DIM;
  const uint64_t output_width =
      value_heads * SLLM_HIP_LINEAR_ATTENTION_HEAD_DIM;
  const uint64_t qkv_shape[] = {token_count, qkv_width};
  const uint64_t output_shape[] = {token_count, output_width};
  const uint64_t scalar_shape[] = {token_count, value_heads};
  const uint64_t conv_shape[] = {qkv_width, 1U,
                                 SLLM_HIP_LINEAR_ATTENTION_CONV_KERNEL_SIZE};
  const uint64_t head_scalar_shape[] = {value_heads};
  const uint64_t norm_shape[] = {SLLM_HIP_LINEAR_ATTENTION_HEAD_DIM};
  struct Validation final {
    const sllm_tensor_binding_t *binding;
    uint32_t dtype;
    uint32_t rank;
    const uint64_t *shape;
    TensorMetadata *metadata;
    const char *role;
  };
  const std::array<Validation, 9> validations = {{
      {&descriptor->qkv, SLLM_TENSOR_DTYPE_BF16, 2U, qkv_shape, &metadata->qkv,
       "qkv"},
      {&descriptor->z, SLLM_TENSOR_DTYPE_BF16, 2U, output_shape, &metadata->z,
       "z"},
      {&descriptor->b_input, SLLM_TENSOR_DTYPE_BF16, 2U, scalar_shape,
       &metadata->b_input, "b"},
      {&descriptor->a_input, SLLM_TENSOR_DTYPE_BF16, 2U, scalar_shape,
       &metadata->a_input, "a"},
      {&descriptor->conv_weight, SLLM_TENSOR_DTYPE_BF16, 3U, conv_shape,
       &metadata->conv_weight, "conv_weight"},
      {&descriptor->a_log, SLLM_TENSOR_DTYPE_F32, 1U, head_scalar_shape,
       &metadata->a_log, "A_log"},
      {&descriptor->dt_bias, SLLM_TENSOR_DTYPE_BF16, 1U, head_scalar_shape,
       &metadata->dt_bias, "dt_bias"},
      {&descriptor->norm_weight, SLLM_TENSOR_DTYPE_F32, 1U, norm_shape,
       &metadata->norm_weight, "norm_weight"},
      {&descriptor->output, SLLM_TENSOR_DTYPE_BF16, 2U, output_shape,
       &metadata->output, "output"},
  }};
  for (const Validation &validation : validations) {
    const sllm_status_t status = validate_tensor(
        *validation.binding, validation.dtype, validation.rank,
        validation.shape, validation.metadata, sink, validation.role);
    if (status != SLLM_STATUS_OK) {
      return status;
    }
  }
  for (std::size_t left = 0U; left != validations.size(); ++left) {
    for (std::size_t right = left + 1U; right != validations.size(); ++right) {
      if (validations[left].metadata->buffer ==
              validations[right].metadata->buffer &&
          intervals_overlap(*validations[left].metadata,
                            *validations[right].metadata)) {
        return sllm_public_runtime::write_error(
            sink, SLLM_STATUS_ALIAS_OVERLAP,
            "linear attention tensor intervals overlap");
      }
    }
  }
  metadata->token_count = token_count;
  metadata->start_position = descriptor->start_position;
  metadata->expected_length = descriptor->expected_length;
  metadata->qk_heads = SLLM_HIP_LINEAR_ATTENTION_QK_HEADS;
  metadata->value_heads = static_cast<uint32_t>(value_heads);
  metadata->head_dim = SLLM_HIP_LINEAR_ATTENTION_HEAD_DIM;
  metadata->conv_kernel_size = SLLM_HIP_LINEAR_ATTENTION_CONV_KERNEL_SIZE;
  metadata->qkv_width = static_cast<uint32_t>(qkv_width);
  metadata->output_width = static_cast<uint32_t>(output_width);
  return SLLM_STATUS_OK;
}

void initialize_view_info(
    sllm_linear_attention_view_info_t *const info) noexcept {
  if (info == nullptr) {
    return;
  }
  std::memset(info, 0, sizeof(*info));
  info->struct_size = sizeof(*info);
  info->abi_version = SLLM_HIP_ABI_VERSION;
  info->info_version = SLLM_HIP_LINEAR_ATTENTION_VIEW_INFO_VERSION;
}

void initialize_dispatch_info(
    sllm_linear_attention_dispatch_info_t *const info) noexcept {
  if (info == nullptr) {
    return;
  }
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = SLLM_HIP_LINEAR_ATTENTION_DISPATCH_INFO_VERSION;
}

} // namespace sllm_linear_attention
