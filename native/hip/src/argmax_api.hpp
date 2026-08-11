#ifndef SLLM_ARGMAX_API_HPP
#define SLLM_ARGMAX_API_HPP

#include "public_runtime_internal.hpp"

#include <cstdint>

namespace sllm_argmax {

struct TensorMetadata final {
  uint64_t byte_offset;
  uint64_t payload_bytes;
  uint64_t end_offset;
};

struct DescriptorMetadata final {
  TensorMetadata logits;
  TensorMetadata output;
  uint64_t m;
  uint64_t v;
};

sllm_status_t validate_descriptor_prefix(const sllm_argmax_desc_t *descriptor,
                                         sllm_error_sink_t *sink) noexcept;

sllm_status_t validate_and_copy_descriptor(const sllm_argmax_desc_t *descriptor,
                                           DescriptorMetadata *metadata,
                                           sllm_error_sink_t *sink) noexcept;

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept;

} // namespace sllm_argmax

#endif // SLLM_ARGMAX_API_HPP
