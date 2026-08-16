#include "moe_expert_api.hpp"

#include <cstring>
#include <limits>

namespace sllm_moe_expert {
namespace {

bool all_zero(const void *const bytes, const std::size_t size) noexcept {
  const auto *const values = static_cast<const unsigned char *>(bytes);
  for (std::size_t index = 0U; index != size; ++index) {
    if (values[index] != 0U)
      return false;
  }
  return true;
}

sllm_status_t tensor(const sllm_tensor_binding_t &binding, const uint32_t dtype,
                     const uint32_t rank, const uint64_t first,
                     const uint64_t second, const uint64_t element_bytes,
                     TensorMetadata *const out,
                     sllm_error_sink_t *const sink) noexcept {
  if (binding.struct_size != sizeof(binding) ||
      binding.abi_version != SLLM_HIP_ABI_VERSION || binding.reserved0 != 0U ||
      !all_zero(binding.reserved, sizeof(binding.reserved)) ||
      binding.buffer == nullptr || binding.dtype != dtype ||
      binding.encoding != SLLM_TENSOR_ENCODING_UNQUANTIZED ||
      binding.rank != rank || binding.shape[0] != first ||
      (rank == 2U && binding.shape[1] != second) ||
      binding.stride_elements[rank - 1U] != 1U ||
      (rank == 2U && binding.stride_elements[0] != second) ||
      binding.byte_offset % element_bytes != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "MoE expert tensor prefix, shape, layout, or dtype differs");
  }
  for (uint32_t index = rank; index < SLLM_HIP_TENSOR_MAX_RANK; ++index) {
    if (binding.shape[index] != 0U || binding.stride_elements[index] != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
          "MoE expert unused tensor metadata must be zero");
    }
  }
  if (first != 0U && second > std::numeric_limits<uint64_t>::max() / first) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_METADATA_OVERFLOW,
                                            "MoE expert tensor overflow");
  }
  const uint64_t elements = rank == 2U ? first * second : first;
  if (elements != 0U &&
      element_bytes > std::numeric_limits<uint64_t>::max() / elements) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_METADATA_OVERFLOW,
                                            "MoE expert tensor overflow");
  }
  out->byte_offset = binding.byte_offset;
  out->payload_bytes = elements * element_bytes;
  if (sllm_public_runtime::add_overflows(out->byte_offset,
                                         out->payload_bytes)) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_METADATA_OVERFLOW,
                                            "MoE expert interval overflow");
  }
  out->end_offset = out->byte_offset + out->payload_bytes;
  return SLLM_STATUS_OK;
}

} // namespace

sllm_status_t
validate_descriptor_prefix(const sllm_moe_expert_desc_t *const descriptor,
                           sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                                            "MoE expert descriptor is null");
  }
  uint32_t prefix[2]{};
  std::memcpy(prefix, descriptor, sizeof(prefix));
  if (prefix[0] != sizeof(*descriptor) || prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "MoE expert descriptor prefix differs");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_and_copy_descriptor(const sllm_moe_expert_desc_t *const descriptor,
                             DescriptorMetadata *const metadata,
                             sllm_error_sink_t *const sink) noexcept {
  const sllm_status_t prefix = validate_descriptor_prefix(descriptor, sink);
  if (prefix != SLLM_STATUS_OK)
    return prefix;
  if (metadata == nullptr ||
      descriptor->op_version != SLLM_HIP_MOE_EXPERT_VERSION ||
      descriptor->reserved0 != 0U ||
      !all_zero(descriptor->reserved, sizeof(descriptor->reserved))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "MoE expert version or reserved fields differ");
  }
  const uint64_t tokens = descriptor->hidden.shape[0];
  if (tokens == 0U || tokens > SLLM_HIP_MOE_EXPERT_MAX_TOKENS) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_UNSUPPORTED,
                                            "MoE expert token count differs");
  }
  metadata->token_count = tokens;
  metadata->active_pair_count = tokens * SLLM_HIP_MOE_EXPERT_TOPK;
  metadata->routing_metadata_bytes = metadata->active_pair_count * 16U +
                                     SLLM_HIP_MOE_EXPERT_COUNT * 4U +
                                     (SLLM_HIP_MOE_EXPERT_COUNT + 1U) * 4U + 4U;
  metadata->workspace_bytes = tokens * UINT64_C(12484);
  const sllm_tensor_binding_t bindings[] = {
      descriptor->hidden, descriptor->routing_metadata, descriptor->layer_blob,
      descriptor->workspace, descriptor->output};
  const uint32_t dtypes[] = {SLLM_TENSOR_DTYPE_BF16, SLLM_TENSOR_DTYPE_U8,
                             SLLM_TENSOR_DTYPE_U8, SLLM_TENSOR_DTYPE_U8,
                             SLLM_TENSOR_DTYPE_BF16};
  const uint32_t ranks[] = {2U, 1U, 1U, 1U, 2U};
  const uint64_t first[] = {tokens, metadata->routing_metadata_bytes,
                            SLLM_HIP_MOE_EXPERT_LAYER_BLOB_BYTES,
                            metadata->workspace_bytes, tokens};
  const uint64_t second[] = {SLLM_HIP_MOE_EXPERT_HIDDEN_SIZE, 0U, 0U, 0U,
                             SLLM_HIP_MOE_EXPERT_HIDDEN_SIZE};
  const uint64_t bytes[] = {2U, 1U, 1U, 1U, 2U};
  for (uint32_t index = 0U; index < 5U; ++index) {
    const sllm_status_t status =
        tensor(bindings[index], dtypes[index], ranks[index], first[index],
               second[index], bytes[index], &metadata->tensors[index], sink);
    if (status != SLLM_STATUS_OK)
      return status;
  }
  return SLLM_STATUS_OK;
}

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept {
  return left.byte_offset < right.end_offset &&
         right.byte_offset < left.end_offset;
}

} // namespace sllm_moe_expert
