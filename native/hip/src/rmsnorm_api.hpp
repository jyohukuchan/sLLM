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
  uint32_t scale_mode;
  /* Phase 87 stage 7: sllm_public_runtime::PrequantMode of `output`
   * (0 = legacy BF16). */
  uint32_t output_prequant;
  /* Raw FP32 bits of the NVFP4 activation tensor scale. Zero for BF16/FP8. */
  uint32_t input_global_scale_f32_bits;
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

/* Shared contiguous tensor binding validation for semantic operations
 * that use the same storage contract. Only legacy unquantized BF16 bindings
 * are accepted; producer outputs that may carry a Phase 87 stage 7
 * prequantized encoding use validate_output_tensor_binding instead. */
sllm_status_t validate_tensor_binding(const sllm_tensor_binding_t *binding,
                                      TensorMetadata *metadata,
                                      sllm_error_sink_t *sink) noexcept;

/* Shared `output` validation: legacy BF16/UNQUANTIZED, or a prequantized
 * producer output (FP8 outer per-row, NVFP4 block16 W4A4). The copied payload
 * includes the scale plane, so bounds and overlap checks see the full encoded
 * span. Row/width derivation follows the RMSNorm row structure:
 * rows = product(shape[0..rank-2]) (1 for rank one), width = last extent. */
sllm_status_t
validate_output_tensor_binding(const sllm_tensor_binding_t *binding,
                               TensorMetadata *metadata,
                               sllm_error_sink_t *sink) noexcept;

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept;

} // namespace sllm_rmsnorm

#endif // SLLM_RMSNORM_API_HPP
