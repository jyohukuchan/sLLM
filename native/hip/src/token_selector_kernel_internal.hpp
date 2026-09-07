#ifndef SLLM_TOKEN_SELECTOR_KERNEL_INTERNAL_HPP
#define SLLM_TOKEN_SELECTOR_KERNEL_INTERNAL_HPP

#include "sllm/hip.h"

#include <hip/hip_runtime.h>

namespace sllm_token_selector_kernel {

inline constexpr const char *kLogicalKernelId =
    "token_selector.bf16_f32_mask.v1";
inline constexpr const char *kDeviceSymbol =
    "sllm_token_selector_bf16_f32_mask_v1";
inline constexpr const char *kFixedLogicalKernelId =
    "token_selector.fixed_topk_topp.v1";
inline constexpr const char *kFixedDeviceSymbol =
    "sllm_token_selector_fixed_topk_topp_v1";

hipError_t launch(const uint16_t *bf16_logits, const float *additive_logits,
                  const uint8_t *valid_mask, uint8_t *workspace,
                  uint64_t vocab_size, float temperature, uint32_t top_k,
                  float top_p, uint32_t flags, uint64_t seed, uint64_t counter,
                  sllm_token_selector_record_t *output,
                  hipStream_t stream) noexcept;

// Source-compatible baseline entry point retained for focused ABI tests and
// callers built against the v1 kernel helper.
hipError_t launch(const uint16_t *bf16_logits, const float *additive_logits,
                  const uint8_t *valid_mask, uint64_t vocab_size,
                  float temperature, uint64_t seed, uint64_t counter,
                  sllm_token_selector_record_t *output,
                  hipStream_t stream) noexcept;

} // namespace sllm_token_selector_kernel

#endif // SLLM_TOKEN_SELECTOR_KERNEL_INTERNAL_HPP
