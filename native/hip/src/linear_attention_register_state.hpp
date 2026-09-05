#ifndef SLLM_LINEAR_ATTENTION_REGISTER_STATE_HPP
#define SLLM_LINEAR_ATTENTION_REGISTER_STATE_HPP

#include <cstdint>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

namespace sllm_linear_attention_kernel {

constexpr uint32_t kRegisterStateQkHeads = 16U;
constexpr uint32_t kRegisterStateValueHeads = 48U;
constexpr uint32_t kRegisterStateHeadDim = 128U;
constexpr uint32_t kRegisterStateQkvWidth = 10240U;
constexpr uint32_t kRegisterStateOutputWidth = 6144U;

constexpr bool register_state_shape_supported(
    const uint32_t token_count, const uint32_t qk_heads,
    const uint32_t value_heads, const uint32_t head_dim,
    const uint32_t qkv_width, const uint32_t output_width) noexcept {
  return token_count >= 2U && token_count <= 32U &&
         qk_heads == kRegisterStateQkHeads &&
         value_heads == kRegisterStateValueHeads &&
         head_dim == kRegisterStateHeadDim &&
         qkv_width == kRegisterStateQkvWidth &&
         output_width == kRegisterStateOutputWidth;
}

#if defined(__HIPCC__)
hipError_t launch_register_state(
    const uint16_t *convolved_qkv, const uint16_t *z, const uint16_t *b_input,
    const uint16_t *a_input, const float *a_log, const uint16_t *dt_bias,
    const float *norm_weight, const float *previous_recurrent_state,
    float *next_recurrent_state, uint16_t *output, uint32_t token_count,
    uint32_t qk_heads, uint32_t value_heads, uint32_t head_dim,
    uint32_t qkv_width, uint32_t output_width, hipStream_t stream) noexcept;
#endif

} // namespace sllm_linear_attention_kernel

#endif // SLLM_LINEAR_ATTENTION_REGISTER_STATE_HPP
