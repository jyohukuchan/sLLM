#ifndef SLLM_ELEMENTWISE_KERNEL_INTERNAL_HPP
#define SLLM_ELEMENTWISE_KERNEL_INTERNAL_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_elementwise_kernel {

constexpr const char *kCopyLogicalKernelId = "elementwise.copy.bf16.v1";
constexpr const char *kAddLogicalKernelId = "elementwise.add.bf16_fp32.v1";
constexpr const char *kSiluMulLogicalKernelId =
    "elementwise.silu_mul.bf16_fp32.v1";
constexpr const char *kSigmoidMulLogicalKernelId =
    "elementwise.sigmoid_mul.bf16_fp32.v1";
constexpr const char *kCopyDeviceSymbol = "sllm_elementwise_copy_bf16_v1";
constexpr const char *kAddDeviceSymbol = "sllm_elementwise_add_bf16_fp32_v1";
constexpr const char *kSiluMulDeviceSymbol =
    "sllm_elementwise_silu_mul_bf16_fp32_v1";
constexpr const char *kSigmoidMulDeviceSymbol =
    "sllm_elementwise_sigmoid_mul_bf16_fp32_v1";
constexpr uint32_t kWorkgroupSize = 256U;

hipError_t launch_copy(const uint16_t *input, uint16_t *output,
                       uint64_t element_count, hipStream_t stream) noexcept;

hipError_t launch_add(const uint16_t *input0, const uint16_t *input1,
                      uint16_t *output, uint64_t element_count,
                      hipStream_t stream) noexcept;

hipError_t launch_silu_mul(const uint16_t *gate, const uint16_t *up,
                           uint16_t *output, uint64_t element_count,
                           hipStream_t stream) noexcept;

hipError_t launch_sigmoid_mul(const uint16_t *gate,
                              const uint16_t *attention_value, uint16_t *output,
                              uint64_t element_count,
                              hipStream_t stream) noexcept;

} // namespace sllm_elementwise_kernel

#endif // SLLM_ELEMENTWISE_KERNEL_INTERNAL_HPP
