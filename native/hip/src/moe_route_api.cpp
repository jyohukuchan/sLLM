#include "moe_route_api.hpp"

#include <cstring>
#include <limits>

namespace sllm_moe_route {
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

bool add_overflows(const uint64_t left, const uint64_t right,
                   uint64_t *const result) noexcept {
  if (right > std::numeric_limits<uint64_t>::max() - left) {
    return true;
  }
  *result = left + right;
  return false;
}

sllm_status_t validate_tensor(const sllm_tensor_binding_t &binding,
                              const uint32_t rank,
                              const uint32_t expected_dtype,
                              const uint64_t element_bytes,
                              TensorMetadata *const copied,
                              sllm_error_sink_t *const sink) noexcept {
  if (binding.struct_size != sizeof(binding) ||
      binding.abi_version != SLLM_HIP_ABI_VERSION || binding.reserved0 != 0U ||
      !all_zero(binding.reserved, sizeof(binding.reserved))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "MoE route tensor binding prefix or reserved fields differ");
  }
  if (binding.buffer == nullptr || binding.rank != rank ||
      binding.dtype != expected_dtype ||
      binding.encoding != SLLM_TENSOR_ENCODING_UNQUANTIZED) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "MoE route tensor buffer, rank, dtype, or encoding differs");
  }
  if (binding.byte_offset % element_bytes != 0U || binding.shape[0] == 0U ||
      (rank == 2U && binding.shape[1] == 0U)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "MoE route tensor alignment or extent differs");
  }
  if (binding.stride_elements[rank - 1U] != 1U ||
      (rank == 2U && binding.stride_elements[0] != binding.shape[1])) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_STRIDE_MISMATCH,
        "MoE route tensors must be contiguous");
  }
  for (uint32_t index = rank; index != SLLM_HIP_TENSOR_MAX_RANK; ++index) {
    if (binding.shape[index] != 0U || binding.stride_elements[index] != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
          "MoE route unused tensor metadata must be zero");
    }
  }
  uint64_t elements = binding.shape[0];
  if ((rank == 2U &&
       multiply_overflows(elements, binding.shape[1], &elements)) ||
      multiply_overflows(elements, element_bytes, &copied->payload_bytes) ||
      sllm_public_runtime::add_overflows(binding.byte_offset,
                                         copied->payload_bytes)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "MoE route tensor interval overflowed u64");
  }
  copied->byte_offset = binding.byte_offset;
  copied->end_offset = binding.byte_offset + copied->payload_bytes;
  return SLLM_STATUS_OK;
}

} // namespace

sllm_status_t
validate_descriptor_prefix(const sllm_moe_route_desc_t *const descriptor,
                           sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                                            "MoE route descriptor is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, descriptor, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_moe_route_desc_t) ||
      prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "MoE route descriptor prefix differs");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_and_copy_descriptor(const sllm_moe_route_desc_t *const descriptor,
                             DescriptorMetadata *const metadata,
                             sllm_error_sink_t *const sink) noexcept {
  if (metadata == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "MoE route metadata output is null");
  }
  const sllm_status_t prefix = validate_descriptor_prefix(descriptor, sink);
  if (prefix != SLLM_STATUS_OK) {
    return prefix;
  }
  if (descriptor->op_version != SLLM_HIP_MOE_ROUTE_VERSION ||
      !all_zero(descriptor->reserved, sizeof(descriptor->reserved))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "MoE route version or reserved fields differ");
  }
  sllm_status_t status =
      validate_tensor(descriptor->logits, 2U, SLLM_TENSOR_DTYPE_BF16,
                      UINT64_C(2), &metadata->logits, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(descriptor->metadata, 1U, SLLM_TENSOR_DTYPE_U8,
                           UINT64_C(1), &metadata->output, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  metadata->token_count = descriptor->logits.shape[0];
  metadata->expert_count = descriptor->logits.shape[1];
  metadata->selected_expert_count = descriptor->selected_expert_count;
  if (metadata->token_count > SLLM_HIP_MOE_ROUTE_MAX_TOKENS ||
      metadata->expert_count > SLLM_HIP_MOE_ROUTE_MAX_EXPERTS ||
      metadata->selected_expert_count == 0U ||
      metadata->selected_expert_count > SLLM_HIP_MOE_ROUTE_MAX_SELECTED ||
      metadata->selected_expert_count > metadata->expert_count) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED, "MoE route shape exceeds its contract");
  }
  if (multiply_overflows(metadata->token_count, metadata->selected_expert_count,
                         &metadata->pair_count)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "MoE route pair count overflowed u64");
  }
  uint64_t cursor = 0U;
  metadata->ids_offset = cursor;
  if (multiply_overflows(metadata->pair_count, UINT64_C(4), &cursor)) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_METADATA_OVERFLOW,
                                            "MoE route layout overflowed");
  }
  metadata->weights_offset = cursor;
  if (add_overflows(cursor, metadata->pair_count * UINT64_C(4), &cursor)) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_METADATA_OVERFLOW,
                                            "MoE route layout overflowed");
  }
  metadata->counts_offset = cursor;
  if (add_overflows(cursor, metadata->expert_count * UINT64_C(4), &cursor)) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_METADATA_OVERFLOW,
                                            "MoE route layout overflowed");
  }
  metadata->offsets_offset = cursor;
  if (add_overflows(cursor,
                    (metadata->expert_count + UINT64_C(1)) * UINT64_C(4),
                    &cursor)) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_METADATA_OVERFLOW,
                                            "MoE route layout overflowed");
  }
  metadata->grouped_tokens_offset = cursor;
  if (add_overflows(cursor, metadata->pair_count * UINT64_C(4), &cursor)) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_METADATA_OVERFLOW,
                                            "MoE route layout overflowed");
  }
  metadata->grouped_slots_offset = cursor;
  if (add_overflows(cursor, metadata->pair_count * UINT64_C(4), &cursor)) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_METADATA_OVERFLOW,
                                            "MoE route layout overflowed");
  }
  metadata->status_offset = cursor;
  if (add_overflows(cursor, UINT64_C(4), &metadata->metadata_bytes) ||
      descriptor->metadata.shape[0] != metadata->metadata_bytes) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "MoE route metadata buffer length differs from the reviewed layout");
  }
  return SLLM_STATUS_OK;
}

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept {
  return left.byte_offset < right.end_offset &&
         right.byte_offset < left.end_offset;
}

} // namespace sllm_moe_route
