#ifndef SLLM_ROTARY_KERNEL_INTERNAL_HPP
#define SLLM_ROTARY_KERNEL_INTERNAL_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_rotary_kernel {

constexpr const char *kLogicalKernelId = "rotary.split_half.bf16_fp32.v1";
constexpr const char *kDeviceSymbol = "sllm_rotary_split_half_bf16_fp32_v1";
constexpr uint32_t kWorkgroupSize = 256U;

hipError_t launch(const uint16_t *query, const uint16_t *key,
                  const int32_t *positions, uint16_t *query_output,
                  uint16_t *key_output, uint32_t token_count, uint32_t q_heads,
                  uint32_t kv_heads, uint32_t head_dim, uint32_t rotary_dim,
                  float theta, hipStream_t stream) noexcept;

} // namespace sllm_rotary_kernel

#endif // SLLM_ROTARY_KERNEL_INTERNAL_HPP
