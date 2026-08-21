#ifndef SLLM_TOKEN_SELECTOR_API_HPP
#define SLLM_TOKEN_SELECTOR_API_HPP

#include "public_runtime_internal.hpp"

namespace sllm_token_selector {

struct TensorMetadata final {
  uint64_t byte_offset;
  uint64_t payload_bytes;
  uint64_t end_offset;
};

struct DescriptorMetadata final {
  TensorMetadata logits;
  TensorMetadata additive_logits;
  TensorMetadata valid_mask;
  TensorMetadata output;
  uint64_t vocab_size;
  float temperature;
  uint64_t seed;
  uint64_t counter;
};

sllm_status_t validate_descriptor_prefix(
    const sllm_token_selector_desc_t *descriptor,
    sllm_error_sink_t *sink) noexcept;

sllm_status_t validate_and_copy_descriptor(
    const sllm_token_selector_desc_t *descriptor, DescriptorMetadata *metadata,
    sllm_error_sink_t *sink) noexcept;

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept;

} // namespace sllm_token_selector

#endif // SLLM_TOKEN_SELECTOR_API_HPP
