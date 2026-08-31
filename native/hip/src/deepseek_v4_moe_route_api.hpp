#ifndef SLLM_DEEPSEEK_V4_MOE_ROUTE_API_HPP
#define SLLM_DEEPSEEK_V4_MOE_ROUTE_API_HPP

#include "public_runtime_internal.hpp"

#include <cstdint>

namespace sllm_deepseek_v4_moe_route {

struct TensorMetadata final {
  uint64_t byte_offset;
  uint64_t payload_bytes;
  uint64_t end_offset;
};

struct DescriptorMetadata final {
  TensorMetadata logits;
  TensorMetadata selection_bias;
  TensorMetadata hash_expert_ids;
  TensorMetadata output;
  uint64_t token_count;
  uint64_t expert_count;
  uint64_t pair_count;
  uint64_t ids_offset;
  uint64_t weights_offset;
  uint64_t counts_offset;
  uint64_t offsets_offset;
  uint64_t grouped_tokens_offset;
  uint64_t grouped_slots_offset;
  uint64_t status_offset;
  uint64_t metadata_bytes;
  uint32_t selected_expert_count;
  uint32_t mode;
  uint32_t renormalize;
  float routed_scale;
};

sllm_status_t
validate_descriptor_prefix(const sllm_deepseek_v4_moe_route_desc_t *descriptor,
                           sllm_error_sink_t *sink) noexcept;

sllm_status_t validate_and_copy_descriptor(
    const sllm_deepseek_v4_moe_route_desc_t *descriptor,
    DescriptorMetadata *metadata, sllm_error_sink_t *sink) noexcept;

sllm_status_t
validate_dispatch_info(const sllm_deepseek_v4_moe_route_dispatch_info_t *info,
                       sllm_error_sink_t *sink) noexcept;

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept;

} // namespace sllm_deepseek_v4_moe_route

#endif // SLLM_DEEPSEEK_V4_MOE_ROUTE_API_HPP
