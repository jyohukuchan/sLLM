#ifndef SLLM_MLP_GATE_UP_SILU_BUNDLE_KERNEL_INTERNAL_HPP
#define SLLM_MLP_GATE_UP_SILU_BUNDLE_KERNEL_INTERNAL_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

#include "sllm/hip.h"

namespace sllm_mlp_gate_up_silu_bundle_kernel {

// Dispatch metadata uses a logical operator id and the concrete HIP device
// symbol separately.
constexpr uint32_t kBaselineKernelId =
    SLLM_HIP_MLP_GATE_UP_SILU_BUNDLE_KERNEL_ID_V1;
constexpr const char *kBaselineLogicalKernelId =
    "mlp_gate_up_silu_bundle.bf16_fp32.decode.v1";
constexpr const char *kBaselineDeviceSymbol =
    "sllm_mlp_gate_up_silu_bundle_bf16_fp32_decode_v1";

hipError_t launch(const uint16_t *, const uint16_t *, const uint16_t *,
                  uint16_t *, uint16_t *, uint16_t *, uint64_t,
                  hipStream_t) noexcept;
} // namespace sllm_mlp_gate_up_silu_bundle_kernel

#endif
