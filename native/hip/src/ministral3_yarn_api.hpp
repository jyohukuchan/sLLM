#ifndef SLLM_MINISTRAL3_YARN_API_HPP
#define SLLM_MINISTRAL3_YARN_API_HPP

#include "rotary_api.hpp"

#include <cstdint>

namespace sllm_ministral3_yarn {

constexpr uint32_t kTensorCount = 5U;
constexpr uint32_t kQHeads = 32U;
constexpr uint32_t kKvHeads = 8U;
constexpr uint32_t kHeadDim = 128U;
constexpr uint32_t kRotaryDim = 128U;
constexpr uint32_t kOriginalContext = 16384U;
constexpr uint32_t kMaxPosition = 262144U;
constexpr uint32_t kMaxTokens = 262144U;
constexpr float kTheta = 1000000.0F;
constexpr float kFactor = 16.0F;
constexpr float kBetaFast = 32.0F;
constexpr float kBetaSlow = 1.0F;
constexpr float kQueryScaleBeta = 0.1F;

struct DescriptorMetadata final {
  sllm_rotary::TensorMetadata query;
  sllm_rotary::TensorMetadata key;
  sllm_rotary::TensorMetadata positions;
  sllm_rotary::TensorMetadata query_output;
  sllm_rotary::TensorMetadata key_output;
  uint64_t token_count;
  uint64_t start_position;
  uint32_t op_version;
  uint32_t position_payload_mode;
};

sllm_status_t
validate_descriptor_prefix(const sllm_ministral3_yarn_desc_t *descriptor,
                           sllm_error_sink_t *sink) noexcept;

sllm_status_t
validate_and_copy_descriptor(const sllm_ministral3_yarn_desc_t *descriptor,
                             DescriptorMetadata *metadata,
                             sllm_error_sink_t *sink) noexcept;

bool intervals_overlap(const sllm_rotary::TensorMetadata &left,
                       const sllm_rotary::TensorMetadata &right) noexcept;

} // namespace sllm_ministral3_yarn

#endif // SLLM_MINISTRAL3_YARN_API_HPP
