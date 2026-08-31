#include "minimax_m3_moe_route_api.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace sllm_minimax_m3_moe_route {
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

sllm_status_t validate_tensor_prefix(const sllm_tensor_binding_t &binding,
                                     sllm_error_sink_t *const sink) noexcept {
  if (binding.struct_size != sizeof(binding)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "MiniMax M3 route tensor binding struct size is unsupported");
  }
  if (binding.abi_version != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "MiniMax M3 route tensor binding ABI is unsupported");
  }
  if (binding.reserved0 != 0U ||
      !all_zero(binding.reserved, sizeof(binding.reserved))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "MiniMax M3 route tensor binding reserved fields must be zero");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_tensor(const sllm_tensor_binding_t &binding, const uint32_t rank,
                const uint32_t expected_dtype, const uint64_t element_bytes,
                const uint64_t first_extent, const uint64_t second_extent,
                TensorMetadata *const copied,
                sllm_error_sink_t *const sink) noexcept {
  const sllm_status_t prefix = validate_tensor_prefix(binding, sink);
  if (prefix != SLLM_STATUS_OK) {
    return prefix;
  }
  if (binding.buffer == nullptr || binding.rank != rank ||
      binding.dtype != expected_dtype ||
      binding.encoding != SLLM_TENSOR_ENCODING_UNQUANTIZED) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "MiniMax M3 route tensor buffer, rank, dtype, or encoding differs");
  }
  if (binding.byte_offset % element_bytes != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_MISALIGNED_OFFSET,
        "MiniMax M3 route tensor offset is misaligned");
  }
  if (binding.shape[0] != first_extent ||
      (rank == 2U && binding.shape[1] != second_extent) ||
      binding.stride_elements[rank - 1U] != 1U ||
      (rank == 2U && binding.stride_elements[0] != second_extent)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "MiniMax M3 route tensor shape or contiguous stride differs");
  }
  for (uint32_t index = rank; index != SLLM_HIP_TENSOR_MAX_RANK; ++index) {
    if (binding.shape[index] != 0U || binding.stride_elements[index] != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
          "MiniMax M3 route unused tensor metadata must be zero");
    }
  }
  uint64_t elements = first_extent;
  if ((rank == 2U && multiply_overflows(elements, second_extent, &elements)) ||
      multiply_overflows(elements, element_bytes, &copied->payload_bytes) ||
      sllm_public_runtime::add_overflows(binding.byte_offset,
                                         copied->payload_bytes)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "MiniMax M3 route tensor interval overflowed u64");
  }
  copied->byte_offset = binding.byte_offset;
  copied->end_offset = binding.byte_offset + copied->payload_bytes;
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_query_info(const sllm_minimax_m3_moe_route_query_info_t *const info,
                    sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "MiniMax M3 route query info output is null");
  }
  uint32_t prefix[3] = {};
  std::memcpy(prefix, info, sizeof(prefix));
  if (prefix[0] != sizeof(*info)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "MiniMax M3 route query info struct size is unsupported");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "MiniMax M3 route query info ABI is unsupported");
  }
  if (prefix[2] != SLLM_HIP_MINIMAX_M3_MOE_ROUTE_QUERY_INFO_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "MiniMax M3 route query info version is unsupported");
  }
  if (!all_zero(info->reserved, sizeof(info->reserved))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "MiniMax M3 route query info reserved fields must be zero");
  }
  return SLLM_STATUS_OK;
}

} // namespace

sllm_status_t validate_descriptor_prefix(
    const sllm_minimax_m3_moe_route_desc_t *const descriptor,
    sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "MiniMax M3 route descriptor is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, descriptor, sizeof(prefix));
  if (prefix[0] != sizeof(*descriptor)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "MiniMax M3 route descriptor struct size is unsupported");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "MiniMax M3 route descriptor ABI is unsupported");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_and_copy_descriptor(
    const sllm_minimax_m3_moe_route_desc_t *const descriptor,
    DescriptorMetadata *const metadata,
    sllm_error_sink_t *const sink) noexcept {
  if (metadata == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "MiniMax M3 route descriptor metadata output is null");
  }
  const sllm_status_t prefix = validate_descriptor_prefix(descriptor, sink);
  if (prefix != SLLM_STATUS_OK) {
    return prefix;
  }
  if (descriptor->op_version != SLLM_HIP_MINIMAX_M3_MOE_ROUTE_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "MiniMax M3 route operation version is unsupported");
  }
  if (!all_zero(descriptor->reserved, sizeof(descriptor->reserved))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "MiniMax M3 route descriptor reserved fields must be zero");
  }
  if (descriptor->selected_expert_count !=
      SLLM_HIP_MINIMAX_M3_MOE_ROUTE_SELECTED_EXPERT_COUNT) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED,
        "MiniMax M3 route requires the reviewed top-4 shape");
  }

  if (descriptor->logits.shape[0] == 0U ||
      descriptor->logits.shape[0] > SLLM_HIP_MINIMAX_M3_MOE_ROUTE_MAX_TOKENS) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED,
        "MiniMax M3 route token count exceeds its launch contract");
  }
  metadata->token_count = descriptor->logits.shape[0];
  metadata->expert_count = SLLM_HIP_MINIMAX_M3_MOE_ROUTE_EXPERT_COUNT;
  metadata->selected_expert_count = descriptor->selected_expert_count;

  sllm_status_t status = validate_tensor(
      descriptor->logits, 2U, SLLM_TENSOR_DTYPE_F32, UINT64_C(4),
      metadata->token_count, metadata->expert_count, &metadata->logits, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(
      descriptor->selection_bias, 1U, SLLM_TENSOR_DTYPE_F32, UINT64_C(4),
      metadata->expert_count, 0U, &metadata->selection_bias, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }

  if (multiply_overflows(metadata->token_count, metadata->selected_expert_count,
                         &metadata->pair_count)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "MiniMax M3 route pair count overflowed u64");
  }
  uint64_t cursor = 0U;
  metadata->ids_offset = cursor;
  if (multiply_overflows(metadata->pair_count, UINT64_C(4), &cursor)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "MiniMax M3 route metadata layout overflowed");
  }
  metadata->weights_offset = cursor;
  if (add_overflows(cursor, metadata->pair_count * UINT64_C(4), &cursor)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "MiniMax M3 route metadata layout overflowed");
  }
  metadata->counts_offset = cursor;
  if (add_overflows(cursor, metadata->expert_count * UINT64_C(4), &cursor)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "MiniMax M3 route metadata layout overflowed");
  }
  metadata->offsets_offset = cursor;
  if (add_overflows(cursor, (metadata->expert_count + 1U) * UINT64_C(4),
                    &cursor)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "MiniMax M3 route metadata layout overflowed");
  }
  metadata->grouped_tokens_offset = cursor;
  if (add_overflows(cursor, metadata->pair_count * UINT64_C(4), &cursor)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "MiniMax M3 route metadata layout overflowed");
  }
  metadata->grouped_slots_offset = cursor;
  if (add_overflows(cursor, metadata->pair_count * UINT64_C(4), &cursor)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "MiniMax M3 route metadata layout overflowed");
  }
  metadata->status_offset = cursor;
  if (add_overflows(cursor, UINT64_C(4), &cursor)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "MiniMax M3 route metadata layout overflowed");
  }
  metadata->metadata_bytes = cursor;
  status = validate_tensor(descriptor->metadata, 1U, SLLM_TENSOR_DTYPE_U8,
                           UINT64_C(1), metadata->metadata_bytes, 0U,
                           &metadata->output, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (descriptor->metadata.byte_offset % alignof(int32_t) != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_MISALIGNED_OFFSET,
        "MiniMax M3 route metadata output is not i32 aligned");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_dispatch_info(
    const sllm_minimax_m3_moe_route_dispatch_info_t *const info,
    sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "MiniMax M3 route dispatch info output is null");
  }
  uint32_t prefix[3] = {};
  std::memcpy(prefix, info, sizeof(prefix));
  if (prefix[0] != sizeof(*info)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "MiniMax M3 route dispatch info struct size is unsupported");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "MiniMax M3 route dispatch info ABI is unsupported");
  }
  if (prefix[2] != SLLM_HIP_MINIMAX_M3_MOE_ROUTE_DISPATCH_INFO_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "MiniMax M3 route dispatch info version is unsupported");
  }
  if (info->reserved0 != 0U ||
      !all_zero(info->reserved, sizeof(info->reserved))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "MiniMax M3 route dispatch info reserved fields must be zero");
  }
  return SLLM_STATUS_OK;
}

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept {
  return left.byte_offset < right.end_offset &&
         right.byte_offset < left.end_offset;
}

} // namespace sllm_minimax_m3_moe_route

extern "C" sllm_status_t sllm_minimax_m3_moe_route_query(
    const sllm_minimax_m3_moe_route_desc_t *const descriptor,
    sllm_minimax_m3_moe_route_query_info_t *const info,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    sllm_minimax_m3_moe_route::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_minimax_m3_moe_route::validate_and_copy_descriptor(
            descriptor, &metadata, error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }
    const sllm_status_t info_status =
        sllm_minimax_m3_moe_route::validate_query_info(info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    const uint32_t struct_size = info->struct_size;
    const uint32_t abi_version = info->abi_version;
    std::memset(info, 0, sizeof(*info));
    info->struct_size = struct_size;
    info->abi_version = abi_version;
    info->info_version = SLLM_HIP_MINIMAX_M3_MOE_ROUTE_QUERY_INFO_VERSION;
    info->token_count = metadata.token_count;
    info->expert_count = metadata.expert_count;
    info->pair_count = metadata.pair_count;
    info->metadata_bytes = metadata.metadata_bytes;
    info->selected_expert_count = metadata.selected_expert_count;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in MiniMax M3 route query");
  }
}
