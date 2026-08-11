#ifndef SLLM_ARGMAX_KERNEL_INTERNAL_HPP
#define SLLM_ARGMAX_KERNEL_INTERNAL_HPP

#include "sllm/hip.h"
#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_argmax_kernel {

inline constexpr const char *kLogicalKernelId = "argmax.bf16_f32.v1";
inline constexpr const char *kDeviceSymbol = "sllm_argmax_bf16_f32_v1";

hipError_t launch(const uint16_t *logits, int32_t *output, uint64_t m,
                  uint64_t v, hipStream_t stream) noexcept;

} // namespace sllm_argmax_kernel

#endif // SLLM_ARGMAX_KERNEL_INTERNAL_HPP
