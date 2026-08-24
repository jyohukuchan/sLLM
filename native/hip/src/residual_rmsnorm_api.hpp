#ifndef SLLM_RESIDUAL_RMSNORM_API_HPP
#define SLLM_RESIDUAL_RMSNORM_API_HPP

#include "rmsnorm_api.hpp"

namespace sllm_residual_rmsnorm {

struct DescriptorMetadata final {
  sllm_rmsnorm::TensorMetadata residual;
  sllm_rmsnorm::TensorMetadata addend;
  sllm_rmsnorm::TensorMetadata raw_scale;
  sllm_rmsnorm::TensorMetadata residual_output;
  sllm_rmsnorm::TensorMetadata output;
  uint32_t epsilon_bits;
  uint32_t scale_mode;
};

sllm_status_t
validate_descriptor_prefix(const sllm_residual_rmsnorm_desc_t *descriptor,
                           sllm_error_sink_t *sink) noexcept;

sllm_status_t
validate_and_copy_descriptor(const sllm_residual_rmsnorm_desc_t *descriptor,
                             DescriptorMetadata *metadata,
                             sllm_error_sink_t *sink) noexcept;

} // namespace sllm_residual_rmsnorm

#endif // SLLM_RESIDUAL_RMSNORM_API_HPP
