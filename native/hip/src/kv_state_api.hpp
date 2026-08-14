#ifndef SLLM_KV_STATE_API_HPP
#define SLLM_KV_STATE_API_HPP

#include "public_runtime_internal.hpp"

#include <cstdint>

namespace sllm_kv_state {

struct TensorMetadata final {
  const sllm_buffer_t *buffer = nullptr;
  uint64_t byte_offset = 0U;
  uint64_t payload_bytes = 0U;
  uint64_t end_offset = 0U;
  uint64_t token_count = 0U;
  uint32_t head_count = 0U;
  uint32_t head_dim = 0U;
};

struct AppendMetadata final {
  uint64_t expected_length = 0U;
  uint64_t start_position = 0U;
  uint64_t token_count = 0U;
  uint64_t end_position = 0U;
  TensorMetadata key_input;
  TensorMetadata value_input;
};

sllm_status_t
validate_state_create_info(const sllm_kv_state_create_info_t *info,
                           sllm_error_sink_t *sink) noexcept;

sllm_status_t validate_append_prefix(const sllm_kv_append_desc_t *descriptor,
                                     sllm_error_sink_t *sink) noexcept;

sllm_status_t validate_and_copy_append(const sllm_kv_append_desc_t *descriptor,
                                       AppendMetadata *metadata,
                                       sllm_error_sink_t *sink) noexcept;

void initialize_view_info(sllm_kv_view_info_t *info) noexcept;
void initialize_append_info(sllm_kv_append_info_t *info) noexcept;

} // namespace sllm_kv_state

#endif // SLLM_KV_STATE_API_HPP
