#ifndef SLLM_WINDOWED_ATTENTION_API_HPP
#define SLLM_WINDOWED_ATTENTION_API_HPP

#include "public_runtime_internal.hpp"

#include <cstdint>

namespace sllm_windowed_attention {

struct TensorMetadata final {
  uint64_t byte_offset;
  uint64_t payload_bytes;
  uint64_t end_offset;
  uint64_t shape[SLLM_HIP_TENSOR_MAX_RANK];
  uint64_t strides[SLLM_HIP_TENSOR_MAX_RANK];
  uint32_t rank;
  uint32_t element_bytes;
};

struct DescriptorMetadata final {
  uint64_t start_position;
  uint64_t expected_kv_length;
  uint64_t sliding_window;
  uint64_t query_count;
  uint32_t q_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t scaling_bits;
  TensorMetadata query;
  TensorMetadata key;
  TensorMetadata value;
  TensorMetadata output;
};

sllm_status_t
validate_descriptor_prefix(const sllm_windowed_attention_desc_t *descriptor,
                           sllm_error_sink_t *sink) noexcept;

sllm_status_t
validate_and_copy_descriptor(const sllm_windowed_attention_desc_t *descriptor,
                             DescriptorMetadata *metadata,
                             sllm_error_sink_t *sink) noexcept;

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept;

} // namespace sllm_windowed_attention

#endif // SLLM_WINDOWED_ATTENTION_API_HPP
