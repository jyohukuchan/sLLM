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
constexpr const char *kScalarMulLogicalKernelId =
    "elementwise.scalar_mul.bf16_fp32.v1";
constexpr const char *kGeluTanhMulLogicalKernelId =
    "elementwise.gelu_tanh_mul.bf16_fp32.v1";
constexpr const char *kTanhSoftcapLogicalKernelId =
    "elementwise.tanh_softcap.bf16_fp32.v1";
constexpr const char *kBroadcastAddLogicalKernelId =
    "elementwise.broadcast_add.bf16_fp32.v1";
constexpr const char *kBroadcastMulLogicalKernelId =
    "elementwise.broadcast_mul.bf16_fp32.v1";
// Phase 87 stage 7: producer-side activation quantization. The BF16
// activation is not written; the fused producer writes FP8 codes plus
// per-row scales, or NVFP4 packed values plus per-16-block scales.
constexpr const char *kPrequantSiluMulFp8LogicalKernelId =
    "elementwise.silu_mul_prequant.fp8.v1";
constexpr const char *kPrequantSiluMulNvfp4LogicalKernelId =
    "elementwise.silu_mul_prequant.nvfp4.v1";
constexpr const char *kPrequantSigmoidMulFp8LogicalKernelId =
    "elementwise.sigmoid_mul_prequant.fp8.v1";
constexpr const char *kCopyDeviceSymbol = "sllm_elementwise_copy_bf16_v1";
constexpr const char *kAddDeviceSymbol = "sllm_elementwise_add_bf16_fp32_v1";
constexpr const char *kSiluMulDeviceSymbol =
    "sllm_elementwise_silu_mul_bf16_fp32_v1";
constexpr const char *kSigmoidMulDeviceSymbol =
    "sllm_elementwise_sigmoid_mul_bf16_fp32_v1";
constexpr const char *kScalarMulDeviceSymbol =
    "sllm_elementwise_scalar_mul_bf16_fp32_v1";
constexpr const char *kGeluTanhMulDeviceSymbol =
    "sllm_elementwise_gelu_tanh_mul_bf16_fp32_v1";
constexpr const char *kTanhSoftcapDeviceSymbol =
    "sllm_elementwise_tanh_softcap_bf16_fp32_v1";
constexpr const char *kBroadcastAddDeviceSymbol =
    "sllm_elementwise_broadcast_add_bf16_fp32_v1";
constexpr const char *kBroadcastMulDeviceSymbol =
    "sllm_elementwise_broadcast_mul_bf16_fp32_v1";
constexpr const char *kPrequantSiluMulFp8DeviceSymbol =
    "sllm_elementwise_silu_mul_prequant_fp8_v1";
constexpr const char *kPrequantSiluMulNvfp4DeviceSymbol =
    "sllm_elementwise_silu_mul_prequant_nvfp4_v1";
constexpr const char *kPrequantSigmoidMulFp8DeviceSymbol =
    "sllm_elementwise_sigmoid_mul_prequant_fp8_v1";
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

hipError_t launch_scalar_mul(const uint16_t *input, const uint16_t *scalar,
                             uint16_t *output, uint64_t element_count,
                             hipStream_t stream) noexcept;

hipError_t launch_gelu_tanh_mul(const uint16_t *gate, const uint16_t *up,
                                uint16_t *output, uint64_t element_count,
                                hipStream_t stream) noexcept;

hipError_t launch_tanh_softcap(const uint16_t *input, const uint16_t *cap,
                               uint16_t *output, uint64_t element_count,
                               hipStream_t stream) noexcept;

hipError_t launch_broadcast_add(const uint16_t *input, const uint16_t *vector,
                                uint16_t *output, uint64_t element_count,
                                uint64_t width, hipStream_t stream) noexcept;

hipError_t launch_broadcast_mul(const uint16_t *input, const uint16_t *vector,
                                uint16_t *output, uint64_t element_count,
                                uint64_t width, hipStream_t stream) noexcept;

// Phase 87 stage 7: producer-side activation quantization launchers. m and k
// come from the output binding shape; for sigmoid the rank-3
// [m, heads, head_dim] binding flattens to k = heads * head_dim.
hipError_t launch_silu_mul_prequant_fp8(const uint16_t *gate,
                                        const uint16_t *up, uint8_t *quantized,
                                        float *activation_scales, uint32_t m,
                                        uint32_t k,
                                        hipStream_t stream) noexcept;

hipError_t launch_silu_mul_prequant_nvfp4(
    const uint16_t *gate, const uint16_t *up, uint8_t *packed_activation,
    uint8_t *activation_block_scales, const float *input_tensor_scale,
    uint32_t m, uint32_t k, hipStream_t stream) noexcept;

hipError_t launch_sigmoid_mul_prequant_fp8(const uint16_t *gate,
                                           const uint16_t *attention_value,
                                           uint8_t *quantized,
                                           float *activation_scales, uint32_t m,
                                           uint32_t k,
                                           hipStream_t stream) noexcept;

} // namespace sllm_elementwise_kernel

#endif // SLLM_ELEMENTWISE_KERNEL_INTERNAL_HPP
