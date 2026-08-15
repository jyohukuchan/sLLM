#ifndef SLLM_ROTARY_API_HPP
#define SLLM_ROTARY_API_HPP

#include "public_runtime_internal.hpp"

#include <cstdint>

namespace sllm_rotary {

constexpr uint32_t kTensorCount = 5U;

struct TensorMetadata final {
  uint64_t byte_offset;
  uint64_t payload_bytes;
  uint64_t end_offset;
  uint32_t rank;
  uint32_t element_bytes;
  uint64_t shape[SLLM_HIP_TENSOR_MAX_RANK];
  uint64_t strides[SLLM_HIP_TENSOR_MAX_RANK];
};

struct DescriptorMetadata final {
  TensorMetadata query;
  TensorMetadata key;
  TensorMetadata positions;
  TensorMetadata query_output;
  TensorMetadata key_output;
  uint64_t token_count;
  uint64_t start_position;
  uint32_t q_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t rotary_dim;
  uint32_t theta_bits;
  uint32_t max_position;
};

sllm_status_t validate_descriptor_prefix(const sllm_rotary_desc_t *descriptor,
                                         sllm_error_sink_t *sink) noexcept;

sllm_status_t validate_and_copy_descriptor(const sllm_rotary_desc_t *descriptor,
                                           DescriptorMetadata *metadata,
                                           sllm_error_sink_t *sink) noexcept;

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept;

} // namespace sllm_rotary

#endif // SLLM_ROTARY_API_HPP
