#ifndef SLLM_LINEAR_ATTENTION_API_HPP
#define SLLM_LINEAR_ATTENTION_API_HPP

#include "public_runtime_internal.hpp"

#include <cstdint>

namespace sllm_linear_attention {

struct TensorMetadata final {
  const sllm_buffer_t *buffer = nullptr;
  uint64_t byte_offset = 0U;
  uint64_t payload_bytes = 0U;
  uint64_t end_offset = 0U;
};

struct DescriptorMetadata final {
  TensorMetadata qkv;
  TensorMetadata z;
  TensorMetadata b_input;
  TensorMetadata a_input;
  TensorMetadata conv_weight;
  TensorMetadata a_log;
  TensorMetadata dt_bias;
  TensorMetadata norm_weight;
  TensorMetadata output;
  uint64_t token_count = 0U;
  uint64_t start_position = 0U;
  uint64_t expected_length = 0U;
  uint32_t qk_heads = 0U;
  uint32_t value_heads = 0U;
  uint32_t head_dim = 0U;
  uint32_t conv_kernel_size = 0U;
  uint32_t qkv_width = 0U;
  uint32_t output_width = 0U;
};

sllm_status_t validate_state_create_info(
    const sllm_linear_attention_state_create_info_t *info,
    sllm_error_sink_t *sink) noexcept;

sllm_status_t
validate_descriptor_prefix(const sllm_linear_attention_desc_t *descriptor,
                           sllm_error_sink_t *sink) noexcept;

sllm_status_t
validate_and_copy_descriptor(const sllm_linear_attention_desc_t *descriptor,
                             DescriptorMetadata *metadata,
                             sllm_error_sink_t *sink) noexcept;

void initialize_view_info(sllm_linear_attention_view_info_t *info) noexcept;
void initialize_dispatch_info(
    sllm_linear_attention_dispatch_info_t *info) noexcept;

} // namespace sllm_linear_attention

#endif // SLLM_LINEAR_ATTENTION_API_HPP
