#ifndef SLLM_ATTENTION_PREPROCESS_API_HPP
#define SLLM_ATTENTION_PREPROCESS_API_HPP

#include "public_runtime_internal.hpp"

#include <cstdint>

namespace sllm_attention_preprocess {

constexpr uint32_t kTensorCount = 8U;

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
  TensorMetadata packed_q_gate;
  TensorMetadata k;
  TensorMetadata q_raw_scale;
  TensorMetadata k_raw_scale;
  TensorMetadata positions;
  TensorMetadata q_output;
  TensorMetadata gate_output;
  TensorMetadata k_output;
  uint64_t m;
  uint32_t start_position;
  uint32_t q_heads;
  uint32_t k_heads;
  uint32_t head_dim;
};

sllm_status_t
validate_descriptor_prefix(const sllm_attention_preprocess_desc_t *descriptor,
                           sllm_error_sink_t *sink) noexcept;

sllm_status_t
validate_and_copy_descriptor(const sllm_attention_preprocess_desc_t *descriptor,
                             DescriptorMetadata *metadata,
                             sllm_error_sink_t *sink) noexcept;

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept;

} // namespace sllm_attention_preprocess

#endif // SLLM_ATTENTION_PREPROCESS_API_HPP
