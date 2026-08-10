#ifndef SLLM_ATTENTION_PREPROCESS_KERNEL_INTERNAL_HPP
#define SLLM_ATTENTION_PREPROCESS_KERNEL_INTERNAL_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_attention_preprocess_kernel {

constexpr const char *kLogicalKernelId =
    "attention_preprocess.headwise_norm_rope.v1";
constexpr const char *kDeviceSymbol =
    "sllm_attention_preprocess_headwise_norm_rope_v1";

hipError_t launch(const uint16_t *packed_q_gate, const uint16_t *k,
                  const uint16_t *q_raw_scale, const uint16_t *k_raw_scale,
                  const int32_t *positions, uint16_t *q_output,
                  uint16_t *gate_output, uint16_t *k_output, uint32_t m,
                  hipStream_t stream) noexcept;

} // namespace sllm_attention_preprocess_kernel

#endif // SLLM_ATTENTION_PREPROCESS_KERNEL_INTERNAL_HPP
