#ifndef SLLM_MOE_EXPERT_API_HPP
#define SLLM_MOE_EXPERT_API_HPP

#include "public_runtime_internal.hpp"
#include "sllm/hip.h"

#include <array>
#include <cstdint>

namespace sllm_moe_expert {

struct TensorMetadata final {
  uint64_t byte_offset;
  uint64_t payload_bytes;
  uint64_t end_offset;
};

struct DescriptorMetadata final {
  std::array<TensorMetadata, 5> tensors;
  uint32_t op_version;
  uint32_t expert_count;
  uint32_t selected_expert_count;
  uint32_t shared_expert_count;
  uint32_t hidden_size;
  uint32_t intermediate_size;
  uint64_t token_count;
  uint64_t active_pair_count;
  uint64_t routing_metadata_bytes;
  uint64_t workspace_bytes;
};

sllm_status_t
validate_descriptor_prefix(const sllm_moe_expert_desc_t *descriptor,
                           sllm_error_sink_t *sink) noexcept;
sllm_status_t
validate_and_copy_descriptor(const sllm_moe_expert_desc_t *descriptor,
                             DescriptorMetadata *metadata,
                             sllm_error_sink_t *sink) noexcept;
bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept;

} // namespace sllm_moe_expert

#endif // SLLM_MOE_EXPERT_API_HPP
