#ifndef SLLM_RMSNORM_API_HPP
#define SLLM_RMSNORM_API_HPP

#include "public_runtime_internal.hpp"

#include <cstdint>

namespace sllm_rmsnorm {

struct TensorMetadata final {
  uint64_t byte_offset;
  uint64_t payload_bytes;
  uint64_t end_offset;
  uint32_t rank;
  uint64_t shape[SLLM_HIP_TENSOR_MAX_RANK];
  uint64_t strides[SLLM_HIP_TENSOR_MAX_RANK];
};

struct DescriptorMetadata final {
  TensorMetadata activation;
  TensorMetadata raw_scale;
  TensorMetadata output;
  uint32_t epsilon_bits;
};

/* The first eight bytes are the fixed ABI prefix.  Callers may provide only
 * this prefix when reporting an intentionally unsupported/truncated
 * descriptor, so this check must not inspect any nested field. */
sllm_status_t validate_descriptor_prefix(const sllm_rmsnorm_desc_t *descriptor,
                                         sllm_error_sink_t *sink) noexcept;

/* Validates only caller-owned descriptor bytes and copies the accepted
 * metadata.  Handle ownership, context identity, buffer bounds, and interval
 * aliasing are resolved by the public runtime after this function returns. */
sllm_status_t
validate_and_copy_descriptor(const sllm_rmsnorm_desc_t *descriptor,
                             DescriptorMetadata *metadata,
                             sllm_error_sink_t *sink) noexcept;

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept;

} // namespace sllm_rmsnorm

#endif // SLLM_RMSNORM_API_HPP
