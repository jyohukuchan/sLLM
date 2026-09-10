#ifndef SLLM_MATMUL_API_HPP
#define SLLM_MATMUL_API_HPP

#include "public_runtime_internal.hpp"

#include <cstdint>

namespace sllm_matmul {

struct TensorMetadata final {
  uint64_t byte_offset;
  uint64_t payload_bytes;
  uint64_t end_offset;
  uint64_t shape[2];
};

struct DescriptorMetadata final {
  TensorMetadata activation;
  TensorMetadata weight;
  TensorMetadata output;
  uint64_t m;
  uint64_t k;
  uint64_t n;
  uint64_t output_elements;
  uint64_t weight_value_bytes;
  uint64_t weight_scale_offset;
  uint64_t weight_tensor_scale_offset;
  uint64_t input_tensor_scale_offset;
  uint32_t fp8_dtype;
  bool fp8_outer;
  bool nvfp4;
  bool nvfp4_w4a4;
  bool mxfp4_w4a4;
  bool mxfp8_w8a8;
  bool mxfp6_w6a6;
};

sllm_status_t validate_descriptor_prefix(const sllm_matmul_desc_t *descriptor,
                                         sllm_error_sink_t *sink) noexcept;

sllm_status_t validate_and_copy_descriptor(const sllm_matmul_desc_t *descriptor,
                                           DescriptorMetadata *metadata,
                                           sllm_error_sink_t *sink) noexcept;

/* Metadata-only validation is used by the allocation-free workspace query.
 * It accepts only null buffer handles and therefore never treats an arbitrary
 * stale handle as a substitute for normal bound-descriptor validation. */
sllm_status_t
validate_and_copy_unbound_descriptor(const sllm_matmul_desc_t *descriptor,
                                     DescriptorMetadata *metadata,
                                     sllm_error_sink_t *sink) noexcept;

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept;

} // namespace sllm_matmul

/* Internal, allocation-free admission query.  The descriptor must contain
 * metadata-only tensor bindings (all three buffer handles null); normal
 * prepare continues to require live bound buffers. */
extern "C" sllm_status_t sllm_hip_matmul_workspace_footprint(
    const sllm_context_t *context, const sllm_matmul_desc_t *descriptor,
    uint64_t *persistent_bytes, uint64_t *queue_bytes, uint64_t *context_bytes,
    sllm_error_sink_t *error_sink) noexcept;

#endif // SLLM_MATMUL_API_HPP
