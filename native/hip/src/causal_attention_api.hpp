#ifndef SLLM_CAUSAL_ATTENTION_API_HPP
#define SLLM_CAUSAL_ATTENTION_API_HPP

#include <cstdint>

#include "public_runtime_internal.hpp"

namespace sllm_causal_attention {

struct TensorMetadata final {
  uint64_t byte_offset = 0U;
  uint64_t payload_bytes = 0U;
  uint64_t end_offset = 0U;
  uint64_t query_count = 0U;
};

struct DescriptorMetadata final {
  TensorMetadata query;
  TensorMetadata output;
  uint64_t start_position = 0U;
  uint64_t expected_kv_length = 0U;
  uint64_t query_count = 0U;
};

sllm_status_t
validate_descriptor_prefix(const sllm_causal_attention_desc_t *descriptor,
                           sllm_error_sink_t *sink) noexcept;

sllm_status_t
validate_and_copy_descriptor(const sllm_causal_attention_desc_t *descriptor,
                             DescriptorMetadata *metadata,
                             sllm_error_sink_t *sink) noexcept;

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept;

void initialize_dispatch_info(
    sllm_causal_attention_dispatch_info_t *info) noexcept;

} // namespace sllm_causal_attention

#endif // SLLM_CAUSAL_ATTENTION_API_HPP
