#ifndef SLLM_EMBEDDING_API_HPP
#define SLLM_EMBEDDING_API_HPP

#include "public_runtime_internal.hpp"

#include <cstdint>

namespace sllm_embedding {

struct TensorMetadata final {
  uint64_t byte_offset;
  uint64_t payload_bytes;
  uint64_t end_offset;
  uint64_t shape[2];
};

struct DescriptorMetadata final {
  TensorMetadata weight;
  TensorMetadata token_ids;
  TensorMetadata output;
  uint64_t vocab_size;
  uint64_t hidden_size;
  uint64_t token_count;
  uint64_t output_elements;
};

sllm_status_t
validate_descriptor_prefix(const sllm_embedding_desc_t *descriptor,
                           sllm_error_sink_t *sink) noexcept;

sllm_status_t
validate_and_copy_descriptor(const sllm_embedding_desc_t *descriptor,
                             DescriptorMetadata *metadata,
                             sllm_error_sink_t *sink) noexcept;

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept;

} // namespace sllm_embedding

#endif // SLLM_EMBEDDING_API_HPP
