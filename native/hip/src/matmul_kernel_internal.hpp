#ifndef SLLM_MATMUL_KERNEL_INTERNAL_HPP
#define SLLM_MATMUL_KERNEL_INTERNAL_HPP

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace sllm_matmul_kernel {

constexpr uint32_t kWorkgroupSize = 256U;
constexpr const char *kLogicalKernelId = "matmul.bf16_fp32.v1";
constexpr const char *kDeviceSymbol = "sllm_matmul_bf16_fp32_v1";
constexpr const char *kPrefillLogicalKernelId = "matmul.bf16_fp32.tiled16.v2";
constexpr const char *kPrefillDeviceSymbol = "sllm_matmul_bf16_fp32_tiled16_v2";
constexpr const char *kDecodeLogicalKernelId = "matmul.bf16_fp32.decode.v3";
constexpr const char *kDecodeDeviceSymbol = "sllm_matmul_bf16_fp32_decode_v3";
constexpr const char *kDecodeWave64LogicalKernelId =
    "matmul.bf16_fp32.decode.wave64.v1";
constexpr const char *kDecodeWave64DeviceSymbol =
    "sllm_matmul_bf16_fp32_decode_wave64_v1";
constexpr const char *kHipBlasLogicalKernelId = "matmul.hipblas.gemm_ex.v2";
constexpr const char *kHipBlasDeviceSymbol = "hipblasGemmEx";
constexpr const char *kFp8NativeLogicalKernelId =
    "matmul.fp8.outer.hipblaslt.v1";
constexpr const char *kFp8NativeDeviceSymbol = "hipblasLtMatmul";
constexpr const char *kFp8EmulationLogicalKernelId =
    "matmul.fp8.outer.byte_decode.v1";
constexpr const char *kFp8EmulationDeviceSymbol =
    "sllm_matmul_fp8_outer_emulation_v1";

enum class KernelVariant : uint32_t {
  Baseline = 1U,
  PrefillTiled16 = 2U,
  DecodeReduction = 3U,
  HipBlas = 4U,
  Fp8Native = 5U,
  Fp8Emulation = 6U,
  DecodeReductionWave64 = 7U,
};

inline KernelVariant select_variant(const uint64_t m, const uint64_t k,
                                    const uint64_t n,
                                    const char *const target) noexcept {
  const char *const force_baseline = std::getenv("SLLM_MATMUL_FORCE_BASELINE");
  if (force_baseline != nullptr && std::strcmp(force_baseline, "1") == 0) {
    return KernelVariant::Baseline;
  }
  (void)k;
  (void)n;
  if (m > 1U && target != nullptr &&
      (std::strcmp(target, "gfx1201") == 0 ||
       std::strcmp(target, "gfx942") == 0)) {
    return KernelVariant::HipBlas;
  }
  return m == 1U ? (target != nullptr && std::strcmp(target, "gfx942") == 0
                        ? KernelVariant::DecodeReductionWave64
                        : KernelVariant::DecodeReduction)
                 : KernelVariant::PrefillTiled16;
}

constexpr const char *logical_kernel_id(const KernelVariant variant) noexcept {
  return variant == KernelVariant::Fp8Native      ? kFp8NativeLogicalKernelId
         : variant == KernelVariant::Fp8Emulation ? kFp8EmulationLogicalKernelId
         : variant == KernelVariant::HipBlas      ? kHipBlasLogicalKernelId
         : variant == KernelVariant::DecodeReductionWave64
             ? kDecodeWave64LogicalKernelId
         : variant == KernelVariant::DecodeReduction
             ? kDecodeLogicalKernelId
             : (variant == KernelVariant::PrefillTiled16
                    ? kPrefillLogicalKernelId
                    : kLogicalKernelId);
}

constexpr const char *device_symbol(const KernelVariant variant) noexcept {
  return variant == KernelVariant::Fp8Native      ? kFp8NativeDeviceSymbol
         : variant == KernelVariant::Fp8Emulation ? kFp8EmulationDeviceSymbol
         : variant == KernelVariant::HipBlas      ? kHipBlasDeviceSymbol
         : variant == KernelVariant::DecodeReductionWave64
             ? kDecodeWave64DeviceSymbol
         : variant == KernelVariant::DecodeReduction
             ? kDecodeDeviceSymbol
             : (variant == KernelVariant::PrefillTiled16 ? kPrefillDeviceSymbol
                                                         : kDeviceSymbol);
}

constexpr uint32_t grid_size_x(const KernelVariant variant, const uint64_t m,
                               const uint64_t n) noexcept {
  return variant == KernelVariant::Fp8Native ? static_cast<uint32_t>(n)
         : variant == KernelVariant::Fp8Emulation
             ? static_cast<uint32_t>((m * n + kWorkgroupSize - 1U) /
                                     kWorkgroupSize)
         : variant == KernelVariant::HipBlas ? static_cast<uint32_t>(n)
         : variant == KernelVariant::DecodeReductionWave64
             ? static_cast<uint32_t>(n)
         : variant == KernelVariant::DecodeReduction
             ? static_cast<uint32_t>(n)
             : (variant == KernelVariant::PrefillTiled16
                    ? static_cast<uint32_t>((n + 15U) / 16U)
                    : static_cast<uint32_t>((m * n + kWorkgroupSize - 1U) /
                                            kWorkgroupSize));
}

hipError_t launch(const uint16_t *activation, const uint16_t *weight,
                  uint16_t *output, uint64_t m, uint64_t k, uint64_t n,
                  KernelVariant variant, hipStream_t stream) noexcept;

hipError_t launch_fp8_quantize(const uint16_t *activation, uint8_t *quantized,
                               float *scales, uint64_t m, uint64_t k, bool fnuz,
                               hipStream_t stream) noexcept;

hipError_t launch_fp8_emulation(const uint8_t *activation,
                                const float *activation_scales,
                                const uint8_t *weight,
                                const float *weight_scales, uint16_t *output,
                                uint64_t m, uint64_t k, uint64_t n,
                                hipStream_t stream) noexcept;

} // namespace sllm_matmul_kernel

#endif // SLLM_MATMUL_KERNEL_INTERNAL_HPP
