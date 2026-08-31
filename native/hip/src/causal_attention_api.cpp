#include "causal_attention_api.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace sllm_causal_attention {
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

sllm_status_t validate_tensor(const sllm_tensor_binding_t &binding,
                              TensorMetadata *const metadata,
                              sllm_error_sink_t *const sink,
                              const char *const name) noexcept {
  if (metadata == nullptr || binding.buffer == nullptr || binding.rank != 3U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "causal attention Q/output binding requires a buffer and rank three");
  }
  if (binding.struct_size != sizeof(binding)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "causal attention tensor binding has an unsupported struct size");
  }
  if (binding.abi_version != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "causal attention tensor binding ABI version is unsupported");
  }
  if (binding.reserved0 != 0U ||
      !all_zero(binding.reserved, sizeof(binding.reserved))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "causal attention tensor binding reserved fields must be zero");
  }
  if (binding.dtype != SLLM_TENSOR_DTYPE_BF16) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_DTYPE,
        "causal attention Q/output tensors must use BF16");
  }
  if (binding.encoding != SLLM_TENSOR_ENCODING_UNQUANTIZED) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_ENCODING,
        "causal attention Q/output tensors must be unquantized");
  }
  if ((binding.byte_offset & UINT64_C(1)) != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_MISALIGNED_OFFSET,
        "causal attention BF16 tensor offset is not aligned");
  }
  uint64_t elements = 1U;
  uint64_t expected_stride = 1U;
  for (uint32_t backwards = 0U; backwards != 3U; ++backwards) {
    const uint32_t index = 2U - backwards;
    if (binding.shape[index] == 0U ||
        binding.stride_elements[index] != expected_stride ||
        multiply_overflows(elements, binding.shape[index], &elements) ||
        multiply_overflows(expected_stride, binding.shape[index],
                           &expected_stride)) {
      return sllm_public_runtime::write_error(
          sink,
          binding.shape[index] == 0U ? SLLM_STATUS_ZERO_EXTENT
                                     : SLLM_STATUS_STRIDE_MISMATCH,
          "causal attention Q/output tensors must be contiguous and non-empty");
    }
  }
  for (uint32_t index = 3U; index != SLLM_HIP_TENSOR_MAX_RANK; ++index) {
    if (binding.shape[index] != 0U || binding.stride_elements[index] != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
          "causal attention unused tensor metadata must be zero");
    }
  }
  uint64_t bytes = 0U;
  if (multiply_overflows(elements, UINT64_C(2), &bytes) ||
      sllm_public_runtime::add_overflows(binding.byte_offset, bytes)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "causal attention tensor interval overflowed u64");
  }
  metadata->byte_offset = binding.byte_offset;
  metadata->payload_bytes = bytes;
  metadata->end_offset = binding.byte_offset + bytes;
  metadata->query_count = binding.shape[0];
  metadata->q_heads = static_cast<uint32_t>(binding.shape[1]);
  metadata->head_dim = static_cast<uint32_t>(binding.shape[2]);
  (void)name;
  return SLLM_STATUS_OK;
}

} // namespace

sllm_status_t
validate_descriptor_prefix(const sllm_causal_attention_desc_t *const descriptor,
                           sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_CAUSAL_ATTENTION_DESCRIPTOR,
        "causal attention descriptor is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, descriptor, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_causal_attention_desc_t)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "causal attention descriptor has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "causal attention descriptor ABI version is unsupported");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_and_copy_descriptor(
    const sllm_causal_attention_desc_t *const descriptor,
    DescriptorMetadata *const metadata,
    sllm_error_sink_t *const sink) noexcept {
  const sllm_status_t prefix = validate_descriptor_prefix(descriptor, sink);
  if (prefix != SLLM_STATUS_OK || metadata == nullptr) {
    return prefix != SLLM_STATUS_OK
               ? prefix
               : sllm_public_runtime::write_error(
                     sink, SLLM_STATUS_INVALID_CAUSAL_ATTENTION_DESCRIPTOR,
                     "causal attention metadata output is null");
  }
  const uint64_t sliding_window =
      static_cast<uint64_t>(descriptor->reserved[0]) |
      (static_cast<uint64_t>(descriptor->reserved[1]) << UINT32_C(32));
  const bool full =
      descriptor->op_version == SLLM_HIP_CAUSAL_ATTENTION_VERSION &&
      all_zero(descriptor->reserved, sizeof(descriptor->reserved));
  const bool sliding =
      descriptor->op_version == SLLM_HIP_CAUSAL_ATTENTION_SLIDING_VERSION &&
      sliding_window == SLLM_HIP_KV_SLIDING_WINDOW_GEMMA4 &&
      descriptor->reserved[2] == 0U && descriptor->reserved[3] == 0U;
  float explicit_score_scale = 0.0F;
  std::memcpy(&explicit_score_scale, &descriptor->reserved[2], sizeof(float));
  const bool explicitly_scaled =
      descriptor->op_version ==
          SLLM_HIP_CAUSAL_ATTENTION_EXPLICIT_SCALE_VERSION &&
      (sliding_window == 0U ||
       sliding_window == SLLM_HIP_KV_SLIDING_WINDOW_GEMMA4) &&
      std::isfinite(explicit_score_scale) && explicit_score_scale > 0.0F &&
      descriptor->reserved[3] == 0U;
  if ((!full && !sliding && !explicitly_scaled) ||
      descriptor->kv_state == nullptr || descriptor->reserved0 != 0U) {
    return sllm_public_runtime::write_error(
        sink,
        descriptor->reserved0 != 0U ||
                !all_zero(descriptor->reserved, sizeof(descriptor->reserved))
            ? SLLM_STATUS_RESERVED_NONZERO
            : SLLM_STATUS_INVALID_CAUSAL_ATTENTION_DESCRIPTOR,
        "causal attention descriptor version, state, or reserved fields are "
        "invalid");
  }
  sllm_status_t status =
      validate_tensor(descriptor->query, &metadata->query, sink, "query");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status =
      validate_tensor(descriptor->output, &metadata->output, sink, "output");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (metadata->query.query_count != metadata->output.query_count ||
      metadata->query.query_count > SLLM_HIP_CAUSAL_ATTENTION_MAX_M ||
      descriptor->start_position >
          std::numeric_limits<uint64_t>::max() - metadata->query.query_count ||
      descriptor->start_position + metadata->query.query_count !=
          descriptor->expected_kv_length) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_CAUSAL_ATTENTION_LENGTH_MISMATCH,
        "causal attention query range does not equal the committed KV length");
  }
  if (metadata->query.query_count == 0U ||
      descriptor->expected_kv_length > SLLM_HIP_KV_MAX_CAPACITY) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_KV_CAPACITY_EXCEEDED,
        "causal attention range is outside the bounded KV capacity");
  }
  if ((metadata->query.q_heads != 8U && metadata->query.q_heads != 16U &&
       metadata->query.q_heads != 32U) ||
      (metadata->query.head_dim != 128U &&
       metadata->query.head_dim != SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM &&
       metadata->query.head_dim != SLLM_HIP_KV_MAX_HEAD_DIM) ||
      metadata->output.q_heads != metadata->query.q_heads ||
      metadata->output.head_dim != metadata->query.head_dim) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "causal attention Q/output must share a reviewed "
        "[M,Hq,128|256|512] shape");
  }
  if (descriptor->query.buffer == descriptor->output.buffer &&
      intervals_overlap(metadata->query, metadata->output)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_ALIAS_OVERLAP,
        "causal attention Q and output intervals overlap");
  }
  metadata->start_position = descriptor->start_position;
  metadata->expected_kv_length = descriptor->expected_kv_length;
  metadata->query_count = metadata->query.query_count;
  metadata->sliding_window = sliding_window;
  metadata->score_scale = explicit_score_scale;
  metadata->explicit_score_scale = explicitly_scaled;
  return SLLM_STATUS_OK;
}

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept {
  return left.byte_offset < right.end_offset &&
         right.byte_offset < left.end_offset;
}

void initialize_dispatch_info(
    sllm_causal_attention_dispatch_info_t *const info) noexcept {
  if (info == nullptr) {
    return;
  }
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION;
}

} // namespace sllm_causal_attention
