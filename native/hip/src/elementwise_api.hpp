#ifndef SLLM_ELEMENTWISE_API_HPP
#define SLLM_ELEMENTWISE_API_HPP

#include "public_runtime_internal.hpp"

#include <cstdint>

namespace sllm_elementwise {

struct TensorMetadata final {
  uint64_t byte_offset;
  uint64_t payload_bytes;
  uint64_t end_offset;
  uint32_t rank;
  uint64_t shape[SLLM_HIP_TENSOR_MAX_RANK];
  uint64_t strides[SLLM_HIP_TENSOR_MAX_RANK];
};

struct DescriptorMetadata final {
  TensorMetadata input0;
  TensorMetadata input1;
  TensorMetadata output;
  uint64_t element_count;
  sllm_elementwise_operation_t operation;
};

sllm_status_t
validate_descriptor_prefix(const sllm_elementwise_desc_t *descriptor,
                           sllm_error_sink_t *sink) noexcept;

sllm_status_t
validate_and_copy_descriptor(const sllm_elementwise_desc_t *descriptor,
                             DescriptorMetadata *metadata,
                             sllm_error_sink_t *sink) noexcept;

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept;

} // namespace sllm_elementwise

#endif // SLLM_ELEMENTWISE_API_HPP
