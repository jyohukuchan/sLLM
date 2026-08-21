#ifndef SLLM_LINEAR_ATTENTION_KERNEL_INTERNAL_HPP
#define SLLM_LINEAR_ATTENTION_KERNEL_INTERNAL_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_linear_attention_kernel {

constexpr uint32_t kQkHeads = 16U;
constexpr uint32_t kValueHeads = 32U;
constexpr uint32_t kHeadDim = 128U;
constexpr uint32_t kQkvWidth = 8192U;
constexpr uint32_t kOutputWidth = 4096U;
constexpr uint32_t kConvHistory = 3U;
constexpr uint32_t kConvKernelSize = 4U;
constexpr uint32_t kWorkgroupSize = 128U;
constexpr const char *kLogicalKernelId = "linear_attention.gdn.v1";
constexpr const char *kConvDeviceSymbol =
    "sllm_linear_attention_causal_conv_silu_v1";
constexpr const char *kRecurrentDeviceSymbol =
    "sllm_linear_attention_recurrent_gated_norm_v1";
constexpr const char *kColumnLogicalKernelId =
    "linear_attention.gdn.column_state.v2";
constexpr const char *kColumnRecurrentDeviceSymbol =
    "sllm_linear_attention_recurrent_column_state_v2";

hipError_t launch_convolution(const uint16_t *qkv, const uint16_t *conv_weight,
                              const uint16_t *previous_conv_state,
                              uint16_t *convolved_qkv,
                              uint16_t *next_conv_state, uint32_t token_count,
                              uint32_t qkv_width, uint32_t conv_kernel_size,
                              hipStream_t stream) noexcept;

hipError_t launch_recurrent(
    const uint16_t *convolved_qkv, const uint16_t *z, const uint16_t *b_input,
    const uint16_t *a_input, const float *a_log, const uint16_t *dt_bias,
    const float *norm_weight, const float *previous_recurrent_state,
    float *next_recurrent_state, uint16_t *output, uint32_t token_count,
    uint32_t qk_heads, uint32_t value_heads, uint32_t head_dim,
    uint32_t qkv_width, uint32_t output_width, hipStream_t stream) noexcept;

hipError_t launch_column_preprocess(
    uint16_t *convolved_qkv, const uint16_t *b_input, const uint16_t *a_input,
    const float *a_log, const uint16_t *dt_bias, float *beta, float *decay,
    uint32_t token_count, uint32_t qk_heads, uint32_t value_heads,
    uint32_t head_dim, uint32_t qkv_width, hipStream_t stream) noexcept;

hipError_t launch_column_recurrent(
    const uint16_t *convolved_qkv, const float *beta, const float *decay,
    const float *previous_recurrent_state, float *next_recurrent_state,
    uint16_t *output, uint32_t token_count, uint32_t qk_heads,
    uint32_t value_heads, uint32_t head_dim, uint32_t qkv_width,
    uint32_t output_width, hipStream_t stream) noexcept;

hipError_t launch_column_postprocess(const uint16_t *z,
                                     const float *norm_weight, uint16_t *output,
                                     uint32_t token_count, uint32_t value_heads,
                                     uint32_t head_dim, uint32_t output_width,
                                     hipStream_t stream) noexcept;

} // namespace sllm_linear_attention_kernel

#endif // SLLM_LINEAR_ATTENTION_KERNEL_INTERNAL_HPP
