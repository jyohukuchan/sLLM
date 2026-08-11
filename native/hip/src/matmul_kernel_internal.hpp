#ifndef SLLM_MATMUL_KERNEL_INTERNAL_HPP
#define SLLM_MATMUL_KERNEL_INTERNAL_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_matmul_kernel {

constexpr uint32_t kWorkgroupSize = 256U;
constexpr const char *kLogicalKernelId = "matmul.bf16_fp32.v1";
constexpr const char *kDeviceSymbol = "sllm_matmul_bf16_fp32_v1";

hipError_t launch(const uint16_t *activation, const uint16_t *weight,
                  uint16_t *output, uint64_t m, uint64_t k, uint64_t n,
                  hipStream_t stream) noexcept;

} // namespace sllm_matmul_kernel

#endif // SLLM_MATMUL_KERNEL_INTERNAL_HPP
