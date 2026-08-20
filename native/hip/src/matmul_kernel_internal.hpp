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
constexpr const char *kDecodeLogicalKernelId = "matmul.bf16_fp32.decode.v4";
constexpr const char *kDecodeDeviceSymbol = "sllm_matmul_bf16_fp32_decode_v4";
constexpr const char *kDecodeWave64LogicalKernelId =
    "matmul.bf16_fp32.decode.wave64.v1";
constexpr const char *kDecodeWave64DeviceSymbol =
    "sllm_matmul_bf16_fp32_decode_wave64_v1";
constexpr const char *kSerialRowsLogicalKernelId =
    "matmul.bf16_fp32.decode.serial_rows.v1";
constexpr const char *kSerialRowsDeviceSymbol =
    "sllm_matmul_bf16_fp32_decode_serial_rows_v1";
constexpr const char *kSerialRowsWave64LogicalKernelId =
    "matmul.bf16_fp32.decode.serial_rows.wave64.v1";
constexpr const char *kSerialRowsWave64DeviceSymbol =
    "sllm_matmul_bf16_fp32_decode_serial_rows_wave64_v1";
constexpr const char *kHipBlasLogicalKernelId = "matmul.hipblas.gemm_ex.v2";
constexpr const char *kHipBlasDeviceSymbol = "hipblasGemmEx";
constexpr const char *kFp8NativeLogicalKernelId =
    "matmul.fp8.outer.hipblaslt.v1";
constexpr const char *kFp8NativeDeviceSymbol = "hipblasLtMatmul";
constexpr const char *kFp8EmulationLogicalKernelId =
    "matmul.fp8.outer.byte_decode.v1";
constexpr const char *kFp8EmulationDeviceSymbol =
    "sllm_matmul_fp8_outer_emulation_v1";
constexpr const char *kNvfp4DecodeLogicalKernelId =
    "matmul.nvfp4.block16.decode.packed_dequant.v1";
constexpr const char *kNvfp4DecodeDeviceSymbol =
    "sllm_matmul_nvfp4_block16_packed_dequant_v1";
constexpr const char *kNvfp4PrefillLogicalKernelId =
    "matmul.nvfp4.block16.prefill.row8_tiled256.v2";
constexpr const char *kNvfp4PrefillDeviceSymbol =
    "sllm_matmul_nvfp4_block16_prefill_row8_tiled256_v2";
constexpr const char *kNvfp4BaselineLogicalKernelId =
    "matmul.nvfp4.block16.packed_dequant.v1";
constexpr const char *kNvfp4BaselineDeviceSymbol =
    "sllm_matmul_nvfp4_block16_packed_dequant_v1";
constexpr const char *kNvfp4W4A4LogicalKernelId =
    "matmul.nvfp4.w4a4.block16.packed.v1";
constexpr const char *kNvfp4W4A4DeviceSymbol =
    "sllm_matmul_nvfp4_w4a4_block16_packed_v1";
constexpr const char *kMxfp4W4A4DecodeLogicalKernelId =
    "matmul.mxfp4.w4a4.block32.decode.v1";
constexpr const char *kMxfp4W4A4DecodeDeviceSymbol =
    "sllm_matmul_mxfp4_w4a4_block32_decode_v1";
constexpr const char *kMxfp4W4A4PrefillLogicalKernelId =
    "matmul.mxfp4.w4a4.block32.prefill.v1";
constexpr const char *kMxfp4W4A4PrefillDeviceSymbol =
    "sllm_matmul_mxfp4_w4a4_block32_prefill_v1";

enum class KernelVariant : uint32_t {
  Baseline = 1U,
  PrefillTiled16 = 2U,
  DecodeReduction = 3U,
  HipBlas = 4U,
  Fp8Native = 5U,
  Fp8Emulation = 6U,
  DecodeReductionWave64 = 7U,
  Nvfp4DecodePackedDequant = 8U,
  Nvfp4PrefillRow8Tiled256 = 9U,
  Nvfp4BaselinePackedDequant = 10U,
  Nvfp4W4A4Packed = 11U,
  SerialRowsReduction = 12U,
  SerialRowsReductionWave64 = 13U,
  Mxfp4W4A4Decode = 14U,
  Mxfp4W4A4Prefill = 15U,
};

inline KernelVariant select_mxfp4_variant(const uint64_t m) noexcept {
  return m == 1U ? KernelVariant::Mxfp4W4A4Decode
                 : KernelVariant::Mxfp4W4A4Prefill;
}

inline KernelVariant select_nvfp4_variant(const uint64_t m) noexcept {
  const char *const force_baseline = std::getenv("SLLM_NVFP4_FORCE_BASELINE");
  if (force_baseline != nullptr && std::strcmp(force_baseline, "1") == 0) {
    return KernelVariant::Nvfp4BaselinePackedDequant;
  }
  return m == 1U ? KernelVariant::Nvfp4DecodePackedDequant
                 : KernelVariant::Nvfp4PrefillRow8Tiled256;
}

inline bool target_is(const char *const target,
                      const char *const expected) noexcept {
  if (target == nullptr || expected == nullptr) {
    return false;
  }
  if (std::strcmp(target, expected) == 0) {
    return true;
  }
  return std::strcmp(expected, "gfx942") == 0 &&
         std::strcmp(target, "gfx942:sramecc+:xnack-") == 0;
}

constexpr bool phase34_gfx1030_hipblas_shape(const uint64_t m, const uint64_t k,
                                             const uint64_t n) noexcept {
  const bool main_projection =
      (k == 2560U && (n == 9216U || n == 8192U || n == 4096U)) ||
      (k == 9216U && n == 2560U) || (k == 4096U && n == 2560U);
  if (main_projection) {
    return m >= 128U;
  }
  // full-attention K/V is too small to amortize the provider until a larger
  // row block. The GDN b/a N=32 projection remains on tiled16: its measured
  // crossover was unstable and its weighted absolute contribution is small.
  return k == 2560U && n == 1024U && m >= 1024U;
}

static_assert(!phase34_gfx1030_hipblas_shape(127U, 2560U, 9216U));
static_assert(phase34_gfx1030_hipblas_shape(128U, 2560U, 9216U));
static_assert(phase34_gfx1030_hipblas_shape(129U, 4096U, 2560U));
static_assert(!phase34_gfx1030_hipblas_shape(1023U, 2560U, 1024U));
static_assert(phase34_gfx1030_hipblas_shape(1024U, 2560U, 1024U));
static_assert(!phase34_gfx1030_hipblas_shape(10001U, 2560U, 32U));
static_assert(!phase34_gfx1030_hipblas_shape(10001U, 2560U, 248320U));

inline KernelVariant select_variant(const uint64_t m, const uint64_t k,
                                    const uint64_t n,
                                    const char *const target) noexcept {
  const char *const force_baseline = std::getenv("SLLM_MATMUL_FORCE_BASELINE");
  if (force_baseline != nullptr && std::strcmp(force_baseline, "1") == 0) {
    return KernelVariant::Baseline;
  }
  // Speculative target verification uses a small logical row block. Keep each
  // row's dot-product arithmetic identical to canonical M=1 decode; only the
  // independent row/column workgroups are grouped into one submission.
  if (m > 1U && m <= 8U) {
    return target_is(target, "gfx942")
               ? KernelVariant::SerialRowsReductionWave64
               : KernelVariant::SerialRowsReduction;
  }
  if (m > 1U && (target_is(target, "gfx1201") || target_is(target, "gfx942"))) {
    return KernelVariant::HipBlas;
  }
  if (target_is(target, "gfx1030") && phase34_gfx1030_hipblas_shape(m, k, n)) {
    return KernelVariant::HipBlas;
  }
  return m == 1U ? (target_is(target, "gfx942")
                        ? KernelVariant::DecodeReductionWave64
                        : KernelVariant::DecodeReduction)
                 : KernelVariant::PrefillTiled16;
}

constexpr const char *logical_kernel_id(const KernelVariant variant) noexcept {
  return variant == KernelVariant::Fp8Native      ? kFp8NativeLogicalKernelId
         : variant == KernelVariant::Fp8Emulation ? kFp8EmulationLogicalKernelId
         : variant == KernelVariant::Nvfp4DecodePackedDequant
             ? kNvfp4DecodeLogicalKernelId
         : variant == KernelVariant::Nvfp4PrefillRow8Tiled256
             ? kNvfp4PrefillLogicalKernelId
         : variant == KernelVariant::Nvfp4BaselinePackedDequant
             ? kNvfp4BaselineLogicalKernelId
         : variant == KernelVariant::Nvfp4W4A4Packed ? kNvfp4W4A4LogicalKernelId
         : variant == KernelVariant::Mxfp4W4A4Decode
             ? kMxfp4W4A4DecodeLogicalKernelId
         : variant == KernelVariant::Mxfp4W4A4Prefill
             ? kMxfp4W4A4PrefillLogicalKernelId
         : variant == KernelVariant::HipBlas ? kHipBlasLogicalKernelId
         : variant == KernelVariant::DecodeReductionWave64
             ? kDecodeWave64LogicalKernelId
         : variant == KernelVariant::SerialRowsReductionWave64
             ? kSerialRowsWave64LogicalKernelId
         : variant == KernelVariant::SerialRowsReduction
             ? kSerialRowsLogicalKernelId
         : variant == KernelVariant::DecodeReduction
             ? kDecodeLogicalKernelId
             : (variant == KernelVariant::PrefillTiled16
                    ? kPrefillLogicalKernelId
                    : kLogicalKernelId);
}

constexpr const char *device_symbol(const KernelVariant variant) noexcept {
  return variant == KernelVariant::Fp8Native      ? kFp8NativeDeviceSymbol
         : variant == KernelVariant::Fp8Emulation ? kFp8EmulationDeviceSymbol
         : variant == KernelVariant::Nvfp4DecodePackedDequant
             ? kNvfp4DecodeDeviceSymbol
         : variant == KernelVariant::Nvfp4PrefillRow8Tiled256
             ? kNvfp4PrefillDeviceSymbol
         : variant == KernelVariant::Nvfp4BaselinePackedDequant
             ? kNvfp4BaselineDeviceSymbol
         : variant == KernelVariant::Nvfp4W4A4Packed ? kNvfp4W4A4DeviceSymbol
         : variant == KernelVariant::Mxfp4W4A4Decode
             ? kMxfp4W4A4DecodeDeviceSymbol
         : variant == KernelVariant::Mxfp4W4A4Prefill
             ? kMxfp4W4A4PrefillDeviceSymbol
         : variant == KernelVariant::HipBlas ? kHipBlasDeviceSymbol
         : variant == KernelVariant::DecodeReductionWave64
             ? kDecodeWave64DeviceSymbol
         : variant == KernelVariant::SerialRowsReductionWave64
             ? kSerialRowsWave64DeviceSymbol
         : variant == KernelVariant::SerialRowsReduction
             ? kSerialRowsDeviceSymbol
         : variant == KernelVariant::DecodeReduction
             ? kDecodeDeviceSymbol
             : (variant == KernelVariant::PrefillTiled16 ? kPrefillDeviceSymbol
                                                         : kDeviceSymbol);
}

constexpr uint32_t grid_size_x(const KernelVariant variant, const uint64_t m,
                               const uint64_t n) noexcept {
  return variant == KernelVariant::Fp8Native ? static_cast<uint32_t>(n)
         : variant == KernelVariant::Nvfp4DecodePackedDequant
             ? static_cast<uint32_t>(n)
         : variant == KernelVariant::Nvfp4PrefillRow8Tiled256
             ? static_cast<uint32_t>(((m + 7U) / 8U) * n)
         : variant == KernelVariant::Nvfp4BaselinePackedDequant
             ? static_cast<uint32_t>(m * n)
         : variant == KernelVariant::Nvfp4W4A4Packed
             ? static_cast<uint32_t>(m * n)
         : variant == KernelVariant::Mxfp4W4A4Decode ? static_cast<uint32_t>(n)
         : variant == KernelVariant::Mxfp4W4A4Prefill
             ? static_cast<uint32_t>(m * n)
         : variant == KernelVariant::Fp8Emulation
             ? static_cast<uint32_t>((m * n + kWorkgroupSize - 1U) /
                                     kWorkgroupSize)
         : variant == KernelVariant::HipBlas ? static_cast<uint32_t>(n)
         : variant == KernelVariant::DecodeReductionWave64
             ? static_cast<uint32_t>(n)
         : variant == KernelVariant::SerialRowsReductionWave64
             ? static_cast<uint32_t>(n)
         : variant == KernelVariant::SerialRowsReduction
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

hipError_t launch_nvfp4(const uint16_t *activation,
                        const uint8_t *packed_weight,
                        const uint8_t *block_scales, const float *tensor_scale,
                        uint16_t *output, uint64_t m, uint64_t k, uint64_t n,
                        hipStream_t stream) noexcept;

hipError_t launch_nvfp4_quantize(const uint16_t *activation,
                                 uint8_t *packed_activation,
                                 uint8_t *block_scales,
                                 const float *input_tensor_scale, uint64_t m,
                                 uint64_t k, hipStream_t stream) noexcept;

hipError_t launch_nvfp4_w4a4(const uint8_t *packed_activation,
                             const uint8_t *activation_block_scales,
                             const uint8_t *packed_weight,
                             const uint8_t *weight_block_scales,
                             const float *weight_tensor_scale,
                             const float *input_tensor_scale, uint16_t *output,
                             uint64_t m, uint64_t k, uint64_t n,
                             hipStream_t stream) noexcept;

hipError_t launch_mxfp4_quantize(const uint16_t *activation,
                                 uint8_t *packed_activation,
                                 uint8_t *block_scales, uint64_t m, uint64_t k,
                                 hipStream_t stream) noexcept;

hipError_t launch_mxfp4_w4a4(const uint8_t *packed_activation,
                             const uint8_t *activation_block_scales,
                             const uint8_t *packed_weight,
                             const uint8_t *weight_block_scales,
                             uint16_t *output, uint64_t m, uint64_t k,
                             uint64_t n, hipStream_t stream) noexcept;

} // namespace sllm_matmul_kernel

#endif // SLLM_MATMUL_KERNEL_INTERNAL_HPP
