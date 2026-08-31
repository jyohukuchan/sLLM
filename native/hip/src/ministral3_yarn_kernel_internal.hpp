#ifndef SLLM_MINISTRAL3_YARN_KERNEL_INTERNAL_HPP
#define SLLM_MINISTRAL3_YARN_KERNEL_INTERNAL_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_ministral3_yarn_kernel {

constexpr const char *kLogicalKernelId =
    "ministral3.yarn_split_half.bf16_fp32.qscale.v1";
constexpr const char *kDeviceSymbol = "sllm_ministral3_yarn_bf16_v1";
constexpr uint32_t kWorkgroupSize = 256U;

hipError_t launch(const uint16_t *query, const uint16_t *key,
                  const int32_t *positions, uint16_t *query_output,
                  uint16_t *key_output, uint32_t token_count, uint32_t q_heads,
                  uint32_t kv_heads, hipStream_t stream) noexcept;

} // namespace sllm_ministral3_yarn_kernel

#endif // SLLM_MINISTRAL3_YARN_KERNEL_INTERNAL_HPP
