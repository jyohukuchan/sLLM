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
constexpr const char *kShortSerialLogicalKernelId =
    "matmul.bf16_fp32.prefill.short_serial.v1";
constexpr const char *kShortSerialDeviceSymbol =
    "sllm_matmul_bf16_fp32_prefill_short_serial_v1";
constexpr const char *kShortMixedLogicalKernelId =
    "matmul.bf16_fp32.prefill.short_mixed_bss.v2";
constexpr const char *kShortMixedDeviceSymbol = "hipblasGemmExBbsF32Output";
constexpr const char *kHipBlasLogicalKernelId = "matmul.hipblas.gemm_ex.v2";
constexpr const char *kHipBlasDeviceSymbol = "hipblasGemmEx";
constexpr int32_t kPhase49Gfx1030RocblasSolution445 = -445;
constexpr const char *kPhase49Gfx1030ShortMixedRocblasSolutionEnvironment =
    "SLLM_MATMUL_GFX1030_SHORT_MIXED_ROCBLAS_SOLUTION";
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
constexpr const char *kMxfp8W8A8DecodeLogicalKernelId =
    "matmul.mxfp8.w8a8.e4m3.block32.decode.v1";
constexpr const char *kMxfp8W8A8DecodeDeviceSymbol =
    "sllm_matmul_mxfp8_w8a8_e4m3_block32_decode_v1";
constexpr const char *kMxfp8W8A8PrefillLogicalKernelId =
    "matmul.mxfp8.w8a8.e4m3.block32.prefill.v1";
constexpr const char *kMxfp8W8A8PrefillDeviceSymbol =
    "sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_v1";
constexpr const char *kMxfp8W8A8PrefillRow8LogicalKernelId =
    "matmul.mxfp8.w8a8.e4m3.block32.prefill.row8.v2";
constexpr const char *kMxfp8W8A8PrefillRow8DeviceSymbol =
    "sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_row8_v2";
constexpr const char *kMxfp8W8A8PrefillTiled16LogicalKernelId =
    "matmul.mxfp8.w8a8.e4m3.block32.prefill.tiled16.v3";
constexpr const char *kMxfp8W8A8PrefillTiled16DeviceSymbol =
    "sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_tiled16_v3";
constexpr const char *kMxfp8W8A8PrefillMmqCol4LogicalKernelId =
    "matmul.mxfp8.w8a8.e4m3.block32.prefill.mmq-col4.v4";
constexpr const char *kMxfp8W8A8PrefillMmqCol4DeviceSymbol =
    "sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_mmq_col4_v4";
constexpr const char *kMxfp8W8A8PrefillMmqCol8LogicalKernelId =
    "matmul.mxfp8.w8a8.e4m3.block32.prefill.mmq-col8.v4";
constexpr const char *kMxfp8W8A8PrefillMmqCol8DeviceSymbol =
    "sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_mmq_col8_v4";
constexpr const char *kMxfp8W8A8PrefillMmqGfx1030Col16LogicalKernelId =
    "matmul.mxfp8.w8a8.gfx1030.mmq-col16.v1";
constexpr const char *kMxfp8W8A8PrefillMmqGfx1030Col16DeviceSymbol =
    "sllm_mxfp8_w8a8_gfx1030_mmq_col16_v1";
constexpr const char *kMxfp8W8A8PrefillMmqGfx1030Col32LogicalKernelId =
    "matmul.mxfp8.w8a8.gfx1030.mmq-col32.v1";
constexpr const char *kMxfp8W8A8PrefillMmqGfx1030Col32DeviceSymbol =
    "sllm_mxfp8_w8a8_gfx1030_mmq_col32_v1";
constexpr const char *kMxfp8W8A8PrefillWmmaN16LogicalKernelId =
    "matmul.mxfp8.w8a8.e4m3.block32.prefill.wmma128x16x32.v1";
constexpr const char *kMxfp8W8A8PrefillWmmaN16DeviceSymbol =
    "sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_wmma128x16x32_v1";
constexpr const char *kMxfp8W8A8PrefillWmmaN64LogicalKernelId =
    "matmul.mxfp8.w8a8.e4m3.block32.prefill.wmma128x64x32.v2";
constexpr const char *kMxfp8W8A8PrefillWmmaN64DeviceSymbol =
    "sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_wmma128x64x32_v2";
constexpr const char *kMxfp8W8A8PrefillWmma4WaveLogicalKernelId =
    "matmul.mxfp8.w8a8.gfx1201.wmma64x64.4w.v1";
constexpr const char *kMxfp8W8A8PrefillWmma4WaveDeviceSymbol =
    "sllm_mxfp8_w8a8_gfx1201_wmma64x64_4w_v1";
constexpr const char *kMxfp8W8A8PrefillWmmaLdsPadLogicalKernelId =
    "matmul.mxfp8.w8a8.gfx1201.wmma128x64.pad33.v1";
constexpr const char *kMxfp8W8A8PrefillWmmaLdsPadDeviceSymbol =
    "sllm_mxfp8_w8a8_gfx1201_wmma128x64_pad33_v1";
constexpr const char *kMxfp8W8A8PrefillWmmaDirectWeightLogicalKernelId =
    "matmul.mxfp8.w8a8.gfx1201.wmma128x64.direct.v1";
constexpr const char *kMxfp8W8A8PrefillWmmaDirectWeightDeviceSymbol =
    "sllm_mxfp8_w8a8_gfx1201_wmma128x64_direct_v1";
constexpr const char *kMxfp8W8A8PrefillWmmaDirectActivationLogicalKernelId =
    "matmul.mxfp8.w8a8.gfx1201.wmma128x64.adirect.v1";
constexpr const char *kMxfp8W8A8PrefillWmmaDirectActivationDeviceSymbol =
    "sllm_mxfp8_w8a8_gfx1201_wmma128x64_adirect_v1";
constexpr const char *kMxfp8W8A8PrefillWmmaDirectBothLogicalKernelId =
    "matmul.mxfp8.w8a8.gfx1201.wmma128x64.bdirect.v1";
constexpr const char *kMxfp8W8A8PrefillWmmaDirectBothDeviceSymbol =
    "sllm_mxfp8_w8a8_gfx1201_wmma128x64_bdirect_v1";
constexpr const char *kMxfp8W8A8PrefillWmmaN128DirectBothLogicalKernelId =
    "matmul.mxfp8.w8a8.gfx1201.wmma128x128.bdirect.v1";
constexpr const char *kMxfp8W8A8PrefillWmmaN128DirectBothDeviceSymbol =
    "sllm_mxfp8_w8a8_gfx1201_wmma128x128_bdirect_v1";
static_assert(sizeof("matmul.mxfp8.w8a8.gfx1201.wmma64x64.4w.v1") <= 64U);
static_assert(sizeof("matmul.mxfp8.w8a8.gfx1201.wmma128x64.pad33.v1") <= 64U);
static_assert(sizeof("matmul.mxfp8.w8a8.gfx1201.wmma128x64.direct.v1") <= 64U);
static_assert(sizeof("matmul.mxfp8.w8a8.gfx1201.wmma128x64.adirect.v1") <= 64U);
static_assert(sizeof("matmul.mxfp8.w8a8.gfx1201.wmma128x64.bdirect.v1") <= 64U);
static_assert(sizeof("matmul.mxfp8.w8a8.gfx1201.wmma128x128.bdirect.v1") <=
              64U);
static_assert(sizeof("sllm_mxfp8_w8a8_gfx1201_wmma64x64_4w_v1") <= 64U);
static_assert(sizeof("sllm_mxfp8_w8a8_gfx1201_wmma128x64_pad33_v1") <= 64U);
static_assert(sizeof("sllm_mxfp8_w8a8_gfx1201_wmma128x64_direct_v1") <= 64U);
static_assert(sizeof("sllm_mxfp8_w8a8_gfx1201_wmma128x64_adirect_v1") <= 64U);
static_assert(sizeof("sllm_mxfp8_w8a8_gfx1201_wmma128x64_bdirect_v1") <= 64U);
static_assert(sizeof("sllm_mxfp8_w8a8_gfx1201_wmma128x128_bdirect_v1") <= 64U);
constexpr const char *kMxfp8W8A8PrefillWmmaEnvironment =
    "SLLM_MXFP8_PREFILL_FORCE_WMMA_GFX1201";
constexpr const char *kMxfp8W8A8PrefillWmmaN16Environment =
    "SLLM_MXFP8_PREFILL_FORCE_WMMA_N16_GFX1201";
constexpr const char *kMxfp8W8A8PrefillWmma4WaveEnvironment =
    "SLLM_MXFP8_PREFILL_FORCE_WMMA_4W_GFX1201";
constexpr const char *kMxfp8W8A8PrefillWmmaLdsPadEnvironment =
    "SLLM_MXFP8_PREFILL_FORCE_WMMA_LDS_PAD_GFX1201";
constexpr const char *kMxfp8W8A8PrefillWmmaDirectWeightEnvironment =
    "SLLM_MXFP8_PREFILL_FORCE_WMMA_DIRECT_WEIGHT_GFX1201";
constexpr const char *kMxfp8W8A8PrefillWmmaDirectActivationEnvironment =
    "SLLM_MXFP8_PREFILL_FORCE_WMMA_DIRECT_ACTIVATION_GFX1201";
constexpr const char *kMxfp8W8A8PrefillWmmaDirectBothEnvironment =
    "SLLM_MXFP8_PREFILL_FORCE_WMMA_DIRECT_BOTH_GFX1201";
constexpr const char *kMxfp8W8A8PrefillWmmaN128DirectBothEnvironment =
    "SLLM_MXFP8_PREFILL_FORCE_WMMA_N128_DIRECT_BOTH_GFX1201";
constexpr const char *kMxfp8W8A8PrefillRow8Environment =
    "SLLM_MXFP8_PREFILL_FORCE_ROW8";
constexpr const char *kMxfp8W8A8PrefillMmqGfx1030ColumnsEnvironment =
    "SLLM_MXFP8_PREFILL_FORCE_MMQ_GFX1030_COLUMNS";
constexpr uint32_t kMxfp8W8A8PrefillWmmaWorkgroupSize = 256U;
constexpr uint32_t kMxfp8W8A8PrefillWmmaRowsPerWorkgroup = 128U;
constexpr uint32_t kMxfp8W8A8PrefillWmma4WaveWorkgroupSize = 128U;
constexpr uint32_t kMxfp8W8A8PrefillWmma4WaveRowsPerWorkgroup = 64U;
constexpr uint32_t kMxfp8W8A8PrefillWmmaN16ColumnsPerWorkgroup = 16U;
constexpr uint32_t kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup = 64U;
constexpr uint32_t kMxfp8W8A8PrefillWmmaN128ColumnsPerWorkgroup = 128U;
constexpr uint32_t kMxfp8W8A8PrefillWmmaBlockK = 32U;
constexpr const char *kMxfp6W6A6DecodeLogicalKernelId =
    "matmul.mxfp6.w6a6.e3m2.block32.decode.v1";
constexpr const char *kMxfp6W6A6DecodeDeviceSymbol =
    "sllm_matmul_mxfp6_w6a6_e3m2_block32_decode_v1";
constexpr const char *kMxfp6W6A6PrefillLogicalKernelId =
    "matmul.mxfp6.w6a6.e3m2.block32.prefill.v1";
constexpr const char *kMxfp6W6A6PrefillDeviceSymbol =
    "sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_v1";
constexpr const char *kMxfp6W6A6PrefillRow8LogicalKernelId =
    "matmul.mxfp6.w6a6.e3m2.block32.prefill.row8.v2";
constexpr const char *kMxfp6W6A6PrefillRow8DeviceSymbol =
    "sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_row8_v2";
constexpr const char *kMxfp6W6A6PrefillTiled16LogicalKernelId =
    "matmul.mxfp6.w6a6.e3m2.block32.prefill.tiled16.v3";
constexpr const char *kMxfp6W6A6PrefillTiled16DeviceSymbol =
    "sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_tiled16_v3";
constexpr const char *kMxfp6W6A6PrefillMmqCol4LogicalKernelId =
    "matmul.mxfp6.w6a6.e3m2.block32.prefill.mmq-col4.v4";
constexpr const char *kMxfp6W6A6PrefillMmqCol4DeviceSymbol =
    "sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_mmq_col4_v4";
constexpr const char *kMxfp6W6A6PrefillMmqCol8LogicalKernelId =
    "matmul.mxfp6.w6a6.e3m2.block32.prefill.mmq-col8.v4";
constexpr const char *kMxfp6W6A6PrefillMmqCol8DeviceSymbol =
    "sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_mmq_col8_v4";

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
  PrefillShortSerial = 16U,
  PrefillShortMixed = 17U,
  Mxfp8W8A8Decode = 18U,
  Mxfp8W8A8Prefill = 19U,
  Mxfp6W6A6Decode = 20U,
  Mxfp6W6A6Prefill = 21U,
  Mxfp8W8A8PrefillRow8 = 22U,
  Mxfp6W6A6PrefillRow8 = 23U,
  Mxfp8W8A8PrefillTiled16 = 24U,
  Mxfp6W6A6PrefillTiled16 = 25U,
  Mxfp8W8A8PrefillMmqCol4 = 26U,
  Mxfp8W8A8PrefillMmqCol8 = 27U,
  Mxfp6W6A6PrefillMmqCol4 = 28U,
  Mxfp6W6A6PrefillMmqCol8 = 29U,
  Mxfp8W8A8PrefillWmmaN16 = 30U,
  Mxfp8W8A8PrefillWmmaN64 = 31U,
  Mxfp8W8A8PrefillWmma4Wave = 32U,
  Mxfp8W8A8PrefillWmmaLdsPad = 33U,
  Mxfp8W8A8PrefillWmmaDirectWeight = 34U,
  Mxfp8W8A8PrefillWmmaDirectActivation = 35U,
  Mxfp8W8A8PrefillWmmaDirectBoth = 36U,
  Mxfp8W8A8PrefillWmmaN128DirectBoth = 37U,
  Mxfp8W8A8PrefillMmqGfx1030Col16 = 38U,
  Mxfp8W8A8PrefillMmqGfx1030Col32 = 39U,
};

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

inline KernelVariant
select_mx_wa_mmq_variant(const bool mxfp8,
                         const KernelVariant fallback) noexcept {
  const char *const columns =
      std::getenv("SLLM_MX_WA_PREFILL_FORCE_MMQ_COLUMNS");
  if (columns != nullptr && std::strcmp(columns, "4") == 0) {
    return mxfp8 ? KernelVariant::Mxfp8W8A8PrefillMmqCol4
                 : KernelVariant::Mxfp6W6A6PrefillMmqCol4;
  }
  if (columns != nullptr && std::strcmp(columns, "8") == 0) {
    return mxfp8 ? KernelVariant::Mxfp8W8A8PrefillMmqCol8
                 : KernelVariant::Mxfp6W6A6PrefillMmqCol8;
  }
  return fallback;
}

inline KernelVariant select_mxfp4_variant(const uint64_t m) noexcept {
  return m == 1U ? KernelVariant::Mxfp4W4A4Decode
                 : KernelVariant::Mxfp4W4A4Prefill;
}

inline KernelVariant select_mxfp8_variant(const uint64_t m) noexcept {
  if (m == 1U) {
    return KernelVariant::Mxfp8W8A8Decode;
  }
  const char *const force_baseline =
      std::getenv("SLLM_MX_WA_PREFILL_FORCE_BASELINE");
  if (force_baseline != nullptr && std::strcmp(force_baseline, "1") == 0) {
    return KernelVariant::Mxfp8W8A8Prefill;
  }
  const char *const force_row8 = std::getenv(kMxfp8W8A8PrefillRow8Environment);
  if (force_row8 != nullptr && std::strcmp(force_row8, "1") == 0) {
    return KernelVariant::Mxfp8W8A8PrefillRow8;
  }
  const KernelVariant mmq =
      select_mx_wa_mmq_variant(true, KernelVariant::Mxfp8W8A8PrefillRow8);
  if (mmq != KernelVariant::Mxfp8W8A8PrefillRow8) {
    return mmq;
  }
  const char *const force_tiled16 =
      std::getenv("SLLM_MXFP8_PREFILL_FORCE_TILED16");
  return force_tiled16 != nullptr && std::strcmp(force_tiled16, "1") == 0
             ? KernelVariant::Mxfp8W8A8PrefillTiled16
             : KernelVariant::Mxfp8W8A8PrefillRow8;
}

// Phase 63's production candidate is intentionally a shape-family rule rather
// than a model-name table.  It starts with large-M, wide dense projections and
// excludes narrow auxiliary projections and vocabulary heads until the shape
// sweep establishes their own stable crossover.
constexpr bool phase63_gfx1201_mxfp8_wmma_shape(const uint64_t m,
                                                const uint64_t k,
                                                const uint64_t n) noexcept {
  return m >= 128U && k >= 2048U && n >= 1024U && n <= 16384U &&
         (k % kMxfp8W8A8PrefillWmmaBlockK) == 0U &&
         (n % kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup) == 0U;
}

constexpr bool phase63_mxfp8_wmma_supported_shape(const uint64_t m,
                                                  const uint64_t k,
                                                  const uint64_t n) noexcept {
  return m > 1U && n > 0U && (k % kMxfp8W8A8PrefillWmmaBlockK) == 0U;
}

constexpr bool phase64_mxfp8_wmma_supported_shape(const uint64_t m,
                                                  const uint64_t k,
                                                  const uint64_t n) noexcept {
  return phase63_mxfp8_wmma_supported_shape(m, k, n) &&
         (n % kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup) == 0U;
}

// The gfx1030 staged-MMQ probes keep the established row8 arithmetic and
// permit output-column tails. They are benchmark-only selectors, so admission
// remains limited to the OCP block-32 K contract and prefill M > 1.
constexpr bool
phase67_mxfp8_mmq_gfx1030_supported_shape(const uint64_t m, const uint64_t k,
                                          const uint64_t n) noexcept {
  return m > 1U && k != 0U && n != 0U &&
         (k % kMxfp8W8A8PrefillWmmaBlockK) == 0U;
}

// The existing col8 MMQ provider wins across the measured large projection
// family on gfx1030, while short-M and M=128/N=1024 can reverse. Keep the
// production rule dimension-only and limited to the measured crossover; the
// narrow projection joins only at M>=512 where both measured lengths win.
constexpr bool phase67_gfx1030_mxfp8_mmq_col8_shape(const uint64_t m,
                                                    const uint64_t k,
                                                    const uint64_t n) noexcept {
  return phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n) && m >= 128U &&
         k >= 2048U &&
         ((n >= 2560U && n <= 16384U) || (m >= 512U && n == 1024U));
}

static_assert(!phase67_gfx1030_mxfp8_mmq_col8_shape(127U, 2560U, 9216U));
static_assert(phase67_gfx1030_mxfp8_mmq_col8_shape(128U, 2560U, 9216U));
static_assert(!phase67_gfx1030_mxfp8_mmq_col8_shape(128U, 2047U, 9216U));
static_assert(!phase67_gfx1030_mxfp8_mmq_col8_shape(128U, 2048U, 2559U));
static_assert(phase67_gfx1030_mxfp8_mmq_col8_shape(128U, 2048U, 2560U));
static_assert(!phase67_gfx1030_mxfp8_mmq_col8_shape(128U, 2049U, 2560U));
static_assert(!phase67_gfx1030_mxfp8_mmq_col8_shape(511U, 2560U, 1024U));
static_assert(phase67_gfx1030_mxfp8_mmq_col8_shape(512U, 2560U, 1024U));
static_assert(!phase67_gfx1030_mxfp8_mmq_col8_shape(512U, 2560U, 1025U));
static_assert(phase67_gfx1030_mxfp8_mmq_col8_shape(512U, 2560U, 16384U));
static_assert(!phase67_gfx1030_mxfp8_mmq_col8_shape(512U, 2560U, 16385U));
static_assert(!phase67_gfx1030_mxfp8_mmq_col8_shape(512U, 2560U, 248320U));

// Direct activation fragments cannot safely read a partial final 128-row
// workgroup. Non-aligned M therefore stays on the existing zero-padded LDS
// provider instead of issuing an out-of-bounds global fragment load.
constexpr bool phase65_mxfp8_wmma_direct_activation_supported_shape(
    const uint64_t m, const uint64_t k, const uint64_t n) noexcept {
  return phase64_mxfp8_wmma_supported_shape(m, k, n) &&
         (m % kMxfp8W8A8PrefillWmmaRowsPerWorkgroup) == 0U;
}

// Phase 66 doubles only the output-column tile count of the established
// direct-both provider. Direct fragment loads require complete M/N tiles and
// the OCP encoding requires complete block-32 K groups. Keep this candidate
// independent of model names and available only through its explicit force
// control until its operator and full-model evidence is complete.
constexpr bool phase66_mxfp8_wmma_n128_direct_both_supported_shape(
    const uint64_t m, const uint64_t k, const uint64_t n) noexcept {
  return m != 0U && k != 0U && n != 0U &&
         (m % kMxfp8W8A8PrefillWmmaRowsPerWorkgroup) == 0U &&
         (k % kMxfp8W8A8PrefillWmmaBlockK) == 0U &&
         (n % kMxfp8W8A8PrefillWmmaN128ColumnsPerWorkgroup) == 0U;
}

// The Phase 65 sweep covers the dense Qwen projection family, including the
// small GQA projections down to N=64. Keep the production rule dimension-only
// and bounded to the measured family; vocabulary heads remain on row8.
constexpr bool
phase65_gfx1201_mxfp8_wmma_direct_both_shape(const uint64_t m, const uint64_t k,
                                             const uint64_t n) noexcept {
  return phase65_mxfp8_wmma_direct_activation_supported_shape(m, k, n) &&
         m >= 128U && k >= 2048U && n >= 64U && n <= 16384U;
}

// Phase 66 adopts the wider output tile only inside the measured Phase 65
// production family. This excludes the short-K operator boundary where N128
// was slower, preserves N=64 on the established provider, and leaves tails
// and vocabulary heads on their existing fail-closed routes.
constexpr bool phase66_gfx1201_mxfp8_wmma_n128_direct_both_shape(
    const uint64_t m, const uint64_t k, const uint64_t n) noexcept {
  return phase65_gfx1201_mxfp8_wmma_direct_both_shape(m, k, n) &&
         (n % kMxfp8W8A8PrefillWmmaN128ColumnsPerWorkgroup) == 0U;
}

// Direct weight loads trade wave-local global traffic for removal of the
// workgroup-wide LDS staging/barrier.  The Phase 64 operator sweep found that
// this pays off for the wide projection family shared by the 2B/4B/9B models.
// The down-projection sweep was non-monotonic, so only the measured 9B pair is
// added; adjacent K values are not generalized.  Keep the production choice
// based on dimensions rather than a model identity.
constexpr bool phase64_gfx1201_mxfp8_wmma_direct_weight_shape(
    const uint64_t m, const uint64_t k, const uint64_t n) noexcept {
  return phase63_gfx1201_mxfp8_wmma_shape(m, k, n) && k != 0U &&
         (n / k >= 3U || (k == 12288U && n == 4096U));
}

inline KernelVariant select_mxfp8_variant(const uint64_t m, const uint64_t k,
                                          const uint64_t n,
                                          const char *const target) noexcept {
  const KernelVariant fallback = select_mxfp8_variant(m);
  if (m == 1U || fallback == KernelVariant::Mxfp8W8A8Prefill) {
    return fallback;
  }

  const bool exact_gfx1030 = target_is(target, "gfx1030");
  const bool exact_gfx1201 = target_is(target, "gfx1201");
  const char *const force_row8 = std::getenv(kMxfp8W8A8PrefillRow8Environment);
  const char *const force_mmq =
      std::getenv("SLLM_MX_WA_PREFILL_FORCE_MMQ_COLUMNS");
  const char *const force_tiled16 =
      std::getenv("SLLM_MXFP8_PREFILL_FORCE_TILED16");
  if ((force_row8 != nullptr && std::strcmp(force_row8, "1") == 0) ||
      (force_mmq != nullptr && (std::strcmp(force_mmq, "4") == 0 ||
                                std::strcmp(force_mmq, "8") == 0)) ||
      (force_tiled16 != nullptr && std::strcmp(force_tiled16, "1") == 0)) {
    return fallback;
  }
  const char *const force_gfx1030_mmq =
      std::getenv(kMxfp8W8A8PrefillMmqGfx1030ColumnsEnvironment);
  if (exact_gfx1030 && phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
    if (force_gfx1030_mmq != nullptr &&
        std::strcmp(force_gfx1030_mmq, "16") == 0) {
      return KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col16;
    }
    if (force_gfx1030_mmq != nullptr &&
        std::strcmp(force_gfx1030_mmq, "32") == 0) {
      return KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col32;
    }
  }
  if (exact_gfx1030 && phase67_gfx1030_mxfp8_mmq_col8_shape(m, k, n)) {
    return KernelVariant::Mxfp8W8A8PrefillMmqCol8;
  }
  const char *const force_wmma = std::getenv(kMxfp8W8A8PrefillWmmaEnvironment);
  const char *const force_wmma_n16 =
      std::getenv(kMxfp8W8A8PrefillWmmaN16Environment);
  const char *const force_wmma_4wave =
      std::getenv(kMxfp8W8A8PrefillWmma4WaveEnvironment);
  const char *const force_wmma_lds_pad =
      std::getenv(kMxfp8W8A8PrefillWmmaLdsPadEnvironment);
  const char *const force_wmma_direct_weight =
      std::getenv(kMxfp8W8A8PrefillWmmaDirectWeightEnvironment);
  const char *const force_wmma_direct_activation =
      std::getenv(kMxfp8W8A8PrefillWmmaDirectActivationEnvironment);
  const char *const force_wmma_direct_both =
      std::getenv(kMxfp8W8A8PrefillWmmaDirectBothEnvironment);
  const char *const force_wmma_n128_direct_both =
      std::getenv(kMxfp8W8A8PrefillWmmaN128DirectBothEnvironment);
  if (force_wmma_n128_direct_both != nullptr &&
      std::strcmp(force_wmma_n128_direct_both, "1") == 0 && exact_gfx1201 &&
      phase66_mxfp8_wmma_n128_direct_both_supported_shape(m, k, n)) {
    return KernelVariant::Mxfp8W8A8PrefillWmmaN128DirectBoth;
  }
  if (force_wmma_4wave != nullptr && std::strcmp(force_wmma_4wave, "1") == 0) {
    return exact_gfx1201 && phase64_mxfp8_wmma_supported_shape(m, k, n)
               ? KernelVariant::Mxfp8W8A8PrefillWmma4Wave
               : fallback;
  }
  if (force_wmma_lds_pad != nullptr &&
      std::strcmp(force_wmma_lds_pad, "1") == 0) {
    return exact_gfx1201 && phase64_mxfp8_wmma_supported_shape(m, k, n)
               ? KernelVariant::Mxfp8W8A8PrefillWmmaLdsPad
               : fallback;
  }
  if (force_wmma_direct_weight != nullptr &&
      std::strcmp(force_wmma_direct_weight, "1") == 0) {
    return exact_gfx1201 && phase64_mxfp8_wmma_supported_shape(m, k, n)
               ? KernelVariant::Mxfp8W8A8PrefillWmmaDirectWeight
               : fallback;
  }
  if (force_wmma_direct_activation != nullptr &&
      std::strcmp(force_wmma_direct_activation, "1") == 0) {
    if (!exact_gfx1201 || !phase64_mxfp8_wmma_supported_shape(m, k, n)) {
      return fallback;
    }
    return phase65_mxfp8_wmma_direct_activation_supported_shape(m, k, n)
               ? KernelVariant::Mxfp8W8A8PrefillWmmaDirectActivation
               : KernelVariant::Mxfp8W8A8PrefillWmmaN64;
  }
  if (force_wmma_direct_both != nullptr &&
      std::strcmp(force_wmma_direct_both, "1") == 0) {
    if (!exact_gfx1201 || !phase64_mxfp8_wmma_supported_shape(m, k, n)) {
      return fallback;
    }
    return phase65_mxfp8_wmma_direct_activation_supported_shape(m, k, n)
               ? KernelVariant::Mxfp8W8A8PrefillWmmaDirectBoth
               : KernelVariant::Mxfp8W8A8PrefillWmmaN64;
  }
  if (force_wmma_n16 != nullptr && std::strcmp(force_wmma_n16, "1") == 0) {
    return exact_gfx1201 && phase63_mxfp8_wmma_supported_shape(m, k, n)
               ? KernelVariant::Mxfp8W8A8PrefillWmmaN16
               : fallback;
  }
  if (force_wmma != nullptr && std::strcmp(force_wmma, "1") == 0) {
    return exact_gfx1201 && phase63_mxfp8_wmma_supported_shape(m, k, n)
               ? KernelVariant::Mxfp8W8A8PrefillWmmaN64
               : fallback;
  }

  // Explicit Phase 62 candidates remain authoritative unless Phase 63 is
  // explicitly forced. This keeps their benchmark controls independent.
  if (fallback != KernelVariant::Mxfp8W8A8PrefillRow8) {
    return fallback;
  }
  if (exact_gfx1201 &&
      phase66_gfx1201_mxfp8_wmma_n128_direct_both_shape(m, k, n)) {
    return KernelVariant::Mxfp8W8A8PrefillWmmaN128DirectBoth;
  }
  if (!exact_gfx1201 || !phase63_gfx1201_mxfp8_wmma_shape(m, k, n)) {
    return exact_gfx1201 &&
                   phase65_gfx1201_mxfp8_wmma_direct_both_shape(m, k, n)
               ? KernelVariant::Mxfp8W8A8PrefillWmmaDirectBoth
               : fallback;
  }
  if (phase65_gfx1201_mxfp8_wmma_direct_both_shape(m, k, n)) {
    return KernelVariant::Mxfp8W8A8PrefillWmmaDirectBoth;
  }
  return phase64_gfx1201_mxfp8_wmma_direct_weight_shape(m, k, n)
             ? KernelVariant::Mxfp8W8A8PrefillWmmaDirectWeight
             : KernelVariant::Mxfp8W8A8PrefillWmmaN64;
}

static_assert(!phase63_gfx1201_mxfp8_wmma_shape(127U, 2560U, 9216U));
static_assert(phase63_gfx1201_mxfp8_wmma_shape(128U, 2560U, 9216U));
static_assert(phase63_gfx1201_mxfp8_wmma_shape(129U, 9216U, 2560U));
static_assert(!phase63_gfx1201_mxfp8_wmma_shape(128U, 2559U, 9216U));
static_assert(phase63_gfx1201_mxfp8_wmma_shape(128U, 2560U, 1024U));
static_assert(!phase63_gfx1201_mxfp8_wmma_shape(128U, 2560U, 248320U));
static_assert(!phase63_mxfp8_wmma_supported_shape(1U, 2560U, 9216U));
static_assert(phase63_mxfp8_wmma_supported_shape(3U, 2560U, 9217U));
static_assert(!phase63_mxfp8_wmma_supported_shape(128U, 257U, 9216U));
static_assert(!phase67_mxfp8_mmq_gfx1030_supported_shape(1U, 32U, 1U));
static_assert(!phase67_mxfp8_mmq_gfx1030_supported_shape(2U, 31U, 1U));
static_assert(phase67_mxfp8_mmq_gfx1030_supported_shape(2U, 32U, 1U));
static_assert(!phase67_mxfp8_mmq_gfx1030_supported_shape(2U, 33U, 1U));
static_assert(!phase65_mxfp8_wmma_direct_activation_supported_shape(127U, 2560U,
                                                                    9216U));
static_assert(phase65_mxfp8_wmma_direct_activation_supported_shape(128U, 2560U,
                                                                   9216U));
static_assert(!phase65_mxfp8_wmma_direct_activation_supported_shape(129U, 2560U,
                                                                    9216U));
static_assert(!phase66_mxfp8_wmma_n128_direct_both_supported_shape(127U, 32U,
                                                                   128U));
static_assert(phase66_mxfp8_wmma_n128_direct_both_supported_shape(128U, 32U,
                                                                  128U));
static_assert(!phase66_mxfp8_wmma_n128_direct_both_supported_shape(129U, 32U,
                                                                   128U));
static_assert(!phase66_mxfp8_wmma_n128_direct_both_supported_shape(128U, 31U,
                                                                   128U));
static_assert(!phase66_mxfp8_wmma_n128_direct_both_supported_shape(128U, 33U,
                                                                   128U));
static_assert(!phase66_mxfp8_wmma_n128_direct_both_supported_shape(128U, 32U,
                                                                   127U));
static_assert(!phase66_mxfp8_wmma_n128_direct_both_supported_shape(128U, 32U,
                                                                   129U));
static_assert(phase66_mxfp8_wmma_n128_direct_both_supported_shape(256U, 2560U,
                                                                  1024U));
static_assert(!phase66_gfx1201_mxfp8_wmma_n128_direct_both_shape(128U, 32U,
                                                                 128U));
static_assert(!phase66_gfx1201_mxfp8_wmma_n128_direct_both_shape(128U, 2560U,
                                                                 64U));
static_assert(phase66_gfx1201_mxfp8_wmma_n128_direct_both_shape(128U, 2560U,
                                                                128U));
static_assert(phase66_gfx1201_mxfp8_wmma_n128_direct_both_shape(2048U, 9216U,
                                                                2560U));
static_assert(!phase66_gfx1201_mxfp8_wmma_n128_direct_both_shape(129U, 2560U,
                                                                 9216U));
static_assert(phase65_gfx1201_mxfp8_wmma_direct_both_shape(128U, 2560U, 64U));
static_assert(phase65_gfx1201_mxfp8_wmma_direct_both_shape(2048U, 4096U,
                                                           12288U));
static_assert(!phase65_gfx1201_mxfp8_wmma_direct_both_shape(129U, 2560U,
                                                            9216U));
static_assert(!phase65_gfx1201_mxfp8_wmma_direct_both_shape(128U, 2560U, 32U));
static_assert(!phase65_gfx1201_mxfp8_wmma_direct_both_shape(128U, 2560U,
                                                            248320U));
static_assert(phase64_gfx1201_mxfp8_wmma_direct_weight_shape(128U, 2048U,
                                                             6144U));
static_assert(phase64_gfx1201_mxfp8_wmma_direct_weight_shape(128U, 2560U,
                                                             9216U));
static_assert(phase64_gfx1201_mxfp8_wmma_direct_weight_shape(128U, 4096U,
                                                             12288U));
static_assert(!phase64_gfx1201_mxfp8_wmma_direct_weight_shape(128U, 4096U,
                                                              12224U));
static_assert(!phase64_gfx1201_mxfp8_wmma_direct_weight_shape(128U, 9216U,
                                                              2560U));
static_assert(phase64_gfx1201_mxfp8_wmma_direct_weight_shape(128U, 12288U,
                                                             4096U));
static_assert(!phase64_gfx1201_mxfp8_wmma_direct_weight_shape(128U, 11264U,
                                                              4096U));

inline KernelVariant select_mxfp6_variant(const uint64_t m) noexcept {
  if (m == 1U) {
    return KernelVariant::Mxfp6W6A6Decode;
  }
  const char *const force_baseline =
      std::getenv("SLLM_MX_WA_PREFILL_FORCE_BASELINE");
  if (force_baseline != nullptr && std::strcmp(force_baseline, "1") == 0) {
    return KernelVariant::Mxfp6W6A6Prefill;
  }
  const KernelVariant mmq =
      select_mx_wa_mmq_variant(false, KernelVariant::Mxfp6W6A6PrefillTiled16);
  if (mmq != KernelVariant::Mxfp6W6A6PrefillTiled16) {
    return mmq;
  }
  const char *const force_row8 = std::getenv("SLLM_MXFP6_PREFILL_FORCE_ROW8");
  return force_row8 != nullptr && std::strcmp(force_row8, "1") == 0
             ? KernelVariant::Mxfp6W6A6PrefillRow8
             : KernelVariant::Mxfp6W6A6PrefillTiled16;
}

inline KernelVariant select_nvfp4_variant(const uint64_t m) noexcept {
  const char *const force_baseline = std::getenv("SLLM_NVFP4_FORCE_BASELINE");
  if (force_baseline != nullptr && std::strcmp(force_baseline, "1") == 0) {
    return KernelVariant::Nvfp4BaselinePackedDequant;
  }
  return m == 1U ? KernelVariant::Nvfp4DecodePackedDequant
                 : KernelVariant::Nvfp4PrefillRow8Tiled256;
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

// ROCm 7.14 exposes a faster Tensile solution for the same BF16/F32 GEMM
// contract through rocblas_gemm_algo_solution_index. Keep this accepted
// default candidate constrained to the already-adopted Phase 34 gfx1030 shape
// set; an explicit environment value of 0 (or any unknown value) rolls back
// to the ordinary hipBLAS path, as do all other targets and shapes.
constexpr bool
phase49_gfx1030_rocblas_solution_445_shape(const uint64_t m, const uint64_t k,
                                           const uint64_t n) noexcept {
  return phase34_gfx1030_hipblas_shape(m, k, n);
}

static_assert(!phase34_gfx1030_hipblas_shape(127U, 2560U, 9216U));
static_assert(phase34_gfx1030_hipblas_shape(128U, 2560U, 9216U));
static_assert(phase34_gfx1030_hipblas_shape(129U, 4096U, 2560U));
static_assert(!phase34_gfx1030_hipblas_shape(1023U, 2560U, 1024U));
static_assert(phase34_gfx1030_hipblas_shape(1024U, 2560U, 1024U));
static_assert(!phase34_gfx1030_hipblas_shape(10001U, 2560U, 32U));
static_assert(!phase34_gfx1030_hipblas_shape(10001U, 2560U, 248320U));
static_assert(!phase49_gfx1030_rocblas_solution_445_shape(127U, 2560U, 9216U));
static_assert(phase49_gfx1030_rocblas_solution_445_shape(128U, 2560U, 9216U));
static_assert(phase49_gfx1030_rocblas_solution_445_shape(10001U, 2560U, 9216U));
static_assert(!phase49_gfx1030_rocblas_solution_445_shape(10001U, 2560U, 32U));
static_assert(!phase49_gfx1030_rocblas_solution_445_shape(10001U, 2560U,
                                                          248320U));

// The short-serial rollback provider covers the five dense Qwen projection
// shapes that are known to benefit from row grouping at small prefill M.
constexpr bool phase49_gfx1030_short_serial_shape(const uint64_t m,
                                                  const uint64_t k,
                                                  const uint64_t n) noexcept {
  const bool main_projection =
      (k == 2560U && (n == 9216U || n == 8192U || n == 4096U)) ||
      (k == 9216U && n == 2560U) || (k == 4096U && n == 2560U);
  return m >= 9U && m <= 63U && main_projection;
}

static_assert(!phase49_gfx1030_short_serial_shape(8U, 2560U, 9216U));
static_assert(phase49_gfx1030_short_serial_shape(9U, 2560U, 9216U));
static_assert(phase49_gfx1030_short_serial_shape(17U, 2560U, 8192U));
static_assert(phase49_gfx1030_short_serial_shape(31U, 2560U, 4096U));
static_assert(phase49_gfx1030_short_serial_shape(33U, 9216U, 2560U));
static_assert(phase49_gfx1030_short_serial_shape(63U, 4096U, 2560U));
static_assert(!phase49_gfx1030_short_serial_shape(64U, 4096U, 2560U));
static_assert(!phase49_gfx1030_short_serial_shape(17U, 2560U, 1024U));
static_assert(!phase49_gfx1030_short_serial_shape(17U, 2560U, 248320U));

constexpr bool phase49_gfx1030_short_mixed_shape(const uint64_t m,
                                                 const uint64_t k,
                                                 const uint64_t n) noexcept {
  const bool qwen_projection =
      (k == 2560U && (n == 32U || n == 1024U || n == 4096U || n == 8192U ||
                      n == 9216U || n == 248320U)) ||
      (k == 4096U && n == 2560U) || (k == 9216U && n == 2560U);
  return m >= 9U && m <= 63U && qwen_projection;
}

// Phase 49 short-mixed solution table.  The table is deliberately narrower
// than the default short-mixed provider: M=17 keeps its per-shape fastest
// choices, while M=32 uses the -473 candidate uniformly because it is the
// measured exact-output solution across all non-vocabulary shapes.  The
// vocabulary head (N=248320) stays on the hipBLAS baseline.  A zero return
// means baseline.  The companion environment is default-on when unset or
// exactly "1"; exactly "0" and unknown values disable the candidate.
constexpr int32_t
phase49_gfx1030_short_mixed_rocblas_solution(const uint64_t m, const uint64_t k,
                                             const uint64_t n) noexcept {
  if (m != 17U && m != 32U) {
    return 0;
  }
  if (m == 32U) {
    if ((k == 2560U &&
         (n == 32U || n == 1024U || n == 4096U || n == 8192U || n == 9216U)) ||
        ((k == 4096U || k == 9216U) && n == 2560U)) {
      return -473;
    }
    return 0;
  }
  if (k == 2560U) {
    if (n == 9216U || n == 8192U || n == 32U) {
      return -473;
    }
    if (n == 4096U) {
      return -472;
    }
    if (n == 1024U && m == 17U) {
      return -473;
    }
    return 0;
  }
  if ((k == 4096U || k == 9216U) && n == 2560U) {
    return -472;
  }
  return 0;
}

inline bool
phase49_gfx1030_short_mixed_rocblas_enabled(const char *const target,
                                            const uint64_t m, const uint64_t k,
                                            const uint64_t n) noexcept {
  const char *const force_baseline = std::getenv("SLLM_MATMUL_FORCE_BASELINE");
  if (force_baseline != nullptr && std::strcmp(force_baseline, "1") == 0) {
    return false;
  }
  const char *const environment =
      std::getenv(kPhase49Gfx1030ShortMixedRocblasSolutionEnvironment);
  const bool enabled =
      environment == nullptr || std::strcmp(environment, "1") == 0;
  return target_is(target, "gfx1030") && enabled &&
         phase49_gfx1030_short_mixed_rocblas_solution(m, k, n) != 0;
}

constexpr bool
phase49_gfx1030_mixed_workspace_bytes(const uint64_t m, const uint64_t n,
                                      uint64_t *const bytes) noexcept {
  if (bytes == nullptr || m == 0U || n == 0U || m > UINT64_MAX / n) {
    return false;
  }
  const uint64_t elements = m * n;
  if (elements > UINT64_MAX / UINT64_C(4)) {
    return false;
  }
  *bytes = elements * UINT64_C(4);
  return true;
}

static_assert(!phase49_gfx1030_short_mixed_shape(8U, 2560U, 32U));
static_assert(phase49_gfx1030_short_mixed_shape(9U, 2560U, 32U));
static_assert(phase49_gfx1030_short_mixed_shape(17U, 2560U, 1024U));
static_assert(phase49_gfx1030_short_mixed_shape(32U, 2560U, 248320U));
static_assert(phase49_gfx1030_short_mixed_shape(63U, 4096U, 2560U));
static_assert(!phase49_gfx1030_short_mixed_shape(64U, 2560U, 9216U));
static_assert(!phase49_gfx1030_short_mixed_shape(17U, 2560U, 33U));
static_assert(phase49_gfx1030_short_mixed_rocblas_solution(17U, 2560U, 9216U) ==
              -473);
static_assert(phase49_gfx1030_short_mixed_rocblas_solution(17U, 2560U, 1024U) ==
              -473);
static_assert(phase49_gfx1030_short_mixed_rocblas_solution(32U, 2560U, 1024U) ==
              -473);
static_assert(phase49_gfx1030_short_mixed_rocblas_solution(17U, 2560U,
                                                           248320U) == 0);
static_assert(phase49_gfx1030_short_mixed_rocblas_solution(16U, 2560U, 9216U) ==
              0);

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
  const char *const disable_short_serial =
      std::getenv("SLLM_MATMUL_GFX1030_SHORT_SERIAL");
  const char *const disable_short_mixed =
      std::getenv("SLLM_MATMUL_GFX1030_SHORT_MIXED");
  if (target_is(target, "gfx1030") &&
      !(disable_short_mixed != nullptr &&
        std::strcmp(disable_short_mixed, "0") == 0) &&
      phase49_gfx1030_short_mixed_shape(m, k, n)) {
    return KernelVariant::PrefillShortMixed;
  }
  if (target_is(target, "gfx1030") &&
      !(disable_short_serial != nullptr &&
        std::strcmp(disable_short_serial, "0") == 0) &&
      phase49_gfx1030_short_serial_shape(m, k, n)) {
    return KernelVariant::PrefillShortSerial;
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
         : variant == KernelVariant::Mxfp8W8A8Decode
             ? kMxfp8W8A8DecodeLogicalKernelId
         : variant == KernelVariant::Mxfp8W8A8Prefill
             ? kMxfp8W8A8PrefillLogicalKernelId
         : variant == KernelVariant::Mxfp8W8A8PrefillRow8
             ? kMxfp8W8A8PrefillRow8LogicalKernelId
         : variant == KernelVariant::Mxfp8W8A8PrefillTiled16
             ? kMxfp8W8A8PrefillTiled16LogicalKernelId
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqCol4
             ? kMxfp8W8A8PrefillMmqCol4LogicalKernelId
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqCol8
             ? kMxfp8W8A8PrefillMmqCol8LogicalKernelId
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col16
             ? kMxfp8W8A8PrefillMmqGfx1030Col16LogicalKernelId
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col32
             ? kMxfp8W8A8PrefillMmqGfx1030Col32LogicalKernelId
         : variant == KernelVariant::Mxfp8W8A8PrefillWmmaN16
             ? kMxfp8W8A8PrefillWmmaN16LogicalKernelId
         : variant == KernelVariant::Mxfp8W8A8PrefillWmmaN64
             ? kMxfp8W8A8PrefillWmmaN64LogicalKernelId
         : variant == KernelVariant::Mxfp8W8A8PrefillWmma4Wave
             ? kMxfp8W8A8PrefillWmma4WaveLogicalKernelId
         : variant == KernelVariant::Mxfp8W8A8PrefillWmmaLdsPad
             ? kMxfp8W8A8PrefillWmmaLdsPadLogicalKernelId
         : variant == KernelVariant::Mxfp8W8A8PrefillWmmaDirectWeight
             ? kMxfp8W8A8PrefillWmmaDirectWeightLogicalKernelId
         : variant == KernelVariant::Mxfp8W8A8PrefillWmmaDirectActivation
             ? kMxfp8W8A8PrefillWmmaDirectActivationLogicalKernelId
         : variant == KernelVariant::Mxfp8W8A8PrefillWmmaDirectBoth
             ? kMxfp8W8A8PrefillWmmaDirectBothLogicalKernelId
         : variant == KernelVariant::Mxfp8W8A8PrefillWmmaN128DirectBoth
             ? kMxfp8W8A8PrefillWmmaN128DirectBothLogicalKernelId
         : variant == KernelVariant::Mxfp6W6A6Decode
             ? kMxfp6W6A6DecodeLogicalKernelId
         : variant == KernelVariant::Mxfp6W6A6Prefill
             ? kMxfp6W6A6PrefillLogicalKernelId
         : variant == KernelVariant::Mxfp6W6A6PrefillRow8
             ? kMxfp6W6A6PrefillRow8LogicalKernelId
         : variant == KernelVariant::Mxfp6W6A6PrefillTiled16
             ? kMxfp6W6A6PrefillTiled16LogicalKernelId
         : variant == KernelVariant::Mxfp6W6A6PrefillMmqCol4
             ? kMxfp6W6A6PrefillMmqCol4LogicalKernelId
         : variant == KernelVariant::Mxfp6W6A6PrefillMmqCol8
             ? kMxfp6W6A6PrefillMmqCol8LogicalKernelId
         : variant == KernelVariant::PrefillShortSerial
             ? kShortSerialLogicalKernelId
         : variant == KernelVariant::PrefillShortMixed
             ? kShortMixedLogicalKernelId
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
         : variant == KernelVariant::Mxfp8W8A8Decode
             ? kMxfp8W8A8DecodeDeviceSymbol
         : variant == KernelVariant::Mxfp8W8A8Prefill
             ? kMxfp8W8A8PrefillDeviceSymbol
         : variant == KernelVariant::Mxfp8W8A8PrefillRow8
             ? kMxfp8W8A8PrefillRow8DeviceSymbol
         : variant == KernelVariant::Mxfp8W8A8PrefillTiled16
             ? kMxfp8W8A8PrefillTiled16DeviceSymbol
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqCol4
             ? kMxfp8W8A8PrefillMmqCol4DeviceSymbol
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqCol8
             ? kMxfp8W8A8PrefillMmqCol8DeviceSymbol
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col16
             ? kMxfp8W8A8PrefillMmqGfx1030Col16DeviceSymbol
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col32
             ? kMxfp8W8A8PrefillMmqGfx1030Col32DeviceSymbol
         : variant == KernelVariant::Mxfp8W8A8PrefillWmmaN16
             ? kMxfp8W8A8PrefillWmmaN16DeviceSymbol
         : variant == KernelVariant::Mxfp8W8A8PrefillWmmaN64
             ? kMxfp8W8A8PrefillWmmaN64DeviceSymbol
         : variant == KernelVariant::Mxfp8W8A8PrefillWmma4Wave
             ? kMxfp8W8A8PrefillWmma4WaveDeviceSymbol
         : variant == KernelVariant::Mxfp8W8A8PrefillWmmaLdsPad
             ? kMxfp8W8A8PrefillWmmaLdsPadDeviceSymbol
         : variant == KernelVariant::Mxfp8W8A8PrefillWmmaDirectWeight
             ? kMxfp8W8A8PrefillWmmaDirectWeightDeviceSymbol
         : variant == KernelVariant::Mxfp8W8A8PrefillWmmaDirectActivation
             ? kMxfp8W8A8PrefillWmmaDirectActivationDeviceSymbol
         : variant == KernelVariant::Mxfp8W8A8PrefillWmmaDirectBoth
             ? kMxfp8W8A8PrefillWmmaDirectBothDeviceSymbol
         : variant == KernelVariant::Mxfp8W8A8PrefillWmmaN128DirectBoth
             ? kMxfp8W8A8PrefillWmmaN128DirectBothDeviceSymbol
         : variant == KernelVariant::Mxfp6W6A6Decode
             ? kMxfp6W6A6DecodeDeviceSymbol
         : variant == KernelVariant::Mxfp6W6A6Prefill
             ? kMxfp6W6A6PrefillDeviceSymbol
         : variant == KernelVariant::Mxfp6W6A6PrefillRow8
             ? kMxfp6W6A6PrefillRow8DeviceSymbol
         : variant == KernelVariant::Mxfp6W6A6PrefillTiled16
             ? kMxfp6W6A6PrefillTiled16DeviceSymbol
         : variant == KernelVariant::Mxfp6W6A6PrefillMmqCol4
             ? kMxfp6W6A6PrefillMmqCol4DeviceSymbol
         : variant == KernelVariant::Mxfp6W6A6PrefillMmqCol8
             ? kMxfp6W6A6PrefillMmqCol8DeviceSymbol
         : variant == KernelVariant::PrefillShortSerial
             ? kShortSerialDeviceSymbol
         : variant == KernelVariant::PrefillShortMixed ? kShortMixedDeviceSymbol
         : variant == KernelVariant::HipBlas           ? kHipBlasDeviceSymbol
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
         : variant == KernelVariant::Mxfp8W8A8Decode ? static_cast<uint32_t>(n)
         : variant == KernelVariant::Mxfp8W8A8Prefill
             ? static_cast<uint32_t>(m * n)
         : variant == KernelVariant::Mxfp8W8A8PrefillRow8
             ? static_cast<uint32_t>(((m + 7U) / 8U) * n)
         : variant == KernelVariant::Mxfp8W8A8PrefillTiled16
             ? static_cast<uint32_t>((n + 15U) / 16U)
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqCol4
             ? static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 3U) / 4U))
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqCol8
             ? static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 7U) / 8U))
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col16
             ? static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 15U) / 16U))
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col32
             ? static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 31U) / 32U))
         : variant == KernelVariant::Mxfp8W8A8PrefillWmmaN16
             ? static_cast<uint32_t>(
                   (n + kMxfp8W8A8PrefillWmmaN16ColumnsPerWorkgroup - 1U) /
                   kMxfp8W8A8PrefillWmmaN16ColumnsPerWorkgroup)
         : variant == KernelVariant::Mxfp8W8A8PrefillWmmaN64
             ? static_cast<uint32_t>(
                   (n + kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup - 1U) /
                   kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup)
         : variant == KernelVariant::Mxfp8W8A8PrefillWmma4Wave ||
                 variant == KernelVariant::Mxfp8W8A8PrefillWmmaLdsPad ||
                 variant == KernelVariant::Mxfp8W8A8PrefillWmmaDirectWeight ||
                 variant ==
                     KernelVariant::Mxfp8W8A8PrefillWmmaDirectActivation ||
                 variant == KernelVariant::Mxfp8W8A8PrefillWmmaDirectBoth
             ? static_cast<uint32_t>(
                   (n + kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup - 1U) /
                   kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup)
         : variant == KernelVariant::Mxfp8W8A8PrefillWmmaN128DirectBoth
             ? static_cast<uint32_t>(
                   (n + kMxfp8W8A8PrefillWmmaN128ColumnsPerWorkgroup - 1U) /
                   kMxfp8W8A8PrefillWmmaN128ColumnsPerWorkgroup)
         : variant == KernelVariant::Mxfp6W6A6Decode ? static_cast<uint32_t>(n)
         : variant == KernelVariant::Mxfp6W6A6Prefill
             ? static_cast<uint32_t>(m * n)
         : variant == KernelVariant::Mxfp6W6A6PrefillRow8
             ? static_cast<uint32_t>(((m + 7U) / 8U) * n)
         : variant == KernelVariant::Mxfp6W6A6PrefillTiled16
             ? static_cast<uint32_t>((n + 15U) / 16U)
         : variant == KernelVariant::Mxfp6W6A6PrefillMmqCol4
             ? static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 3U) / 4U))
         : variant == KernelVariant::Mxfp6W6A6PrefillMmqCol8
             ? static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 7U) / 8U))
         : variant == KernelVariant::PrefillShortSerial
             ? static_cast<uint32_t>(((m + 7U) / 8U) * n)
         : variant == KernelVariant::PrefillShortMixed
             ? static_cast<uint32_t>(n)
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

constexpr uint32_t workgroup_size_x(const KernelVariant variant) noexcept {
  return variant == KernelVariant::Mxfp8W8A8PrefillWmma4Wave
             ? kMxfp8W8A8PrefillWmma4WaveWorkgroupSize
         : variant == KernelVariant::Mxfp8W8A8PrefillWmmaN16 ||
                 variant == KernelVariant::Mxfp8W8A8PrefillWmmaN64 ||
                 variant == KernelVariant::Mxfp8W8A8PrefillWmmaLdsPad ||
                 variant == KernelVariant::Mxfp8W8A8PrefillWmmaDirectWeight ||
                 variant ==
                     KernelVariant::Mxfp8W8A8PrefillWmmaDirectActivation ||
                 variant == KernelVariant::Mxfp8W8A8PrefillWmmaDirectBoth ||
                 variant == KernelVariant::Mxfp8W8A8PrefillWmmaN128DirectBoth
             ? kMxfp8W8A8PrefillWmmaWorkgroupSize
             : kWorkgroupSize;
}

static_assert(workgroup_size_x(KernelVariant::Mxfp8W8A8PrefillWmmaN16) == 256U);
static_assert(workgroup_size_x(KernelVariant::Mxfp8W8A8PrefillWmmaN64) == 256U);
static_assert(workgroup_size_x(KernelVariant::Mxfp8W8A8PrefillWmma4Wave) ==
              128U);
static_assert(workgroup_size_x(
                  KernelVariant::Mxfp8W8A8PrefillWmmaN128DirectBoth) == 256U);
static_assert(grid_size_x(KernelVariant::Mxfp8W8A8PrefillWmmaN128DirectBoth,
                          128U, 128U) == 1U);
static_assert(grid_size_x(KernelVariant::Mxfp8W8A8PrefillWmmaN128DirectBoth,
                          128U, 256U) == 2U);
static_assert(grid_size_x(KernelVariant::Mxfp8W8A8PrefillWmmaN64, 128U, 15U) ==
              1U);
static_assert(grid_size_x(KernelVariant::Mxfp8W8A8PrefillWmmaN64, 128U, 16U) ==
              1U);
static_assert(grid_size_x(KernelVariant::Mxfp8W8A8PrefillWmmaN64, 128U, 17U) ==
              1U);
static_assert(grid_size_x(KernelVariant::Mxfp8W8A8PrefillWmmaN64, 128U, 33U) ==
              1U);
static_assert(grid_size_x(KernelVariant::Mxfp8W8A8PrefillWmmaN64, 128U, 65U) ==
              2U);
static_assert(grid_size_x(KernelVariant::Mxfp8W8A8PrefillWmmaN16, 128U, 33U) ==
              3U);
static_assert(grid_size_x(KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col16, 8U,
                          15U) == 1U);
static_assert(grid_size_x(KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col16, 9U,
                          17U) == 4U);
static_assert(grid_size_x(KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col32, 8U,
                          31U) == 1U);
static_assert(grid_size_x(KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col32, 9U,
                          33U) == 4U);

hipError_t launch(const uint16_t *activation, const uint16_t *weight,
                  uint16_t *output, uint64_t m, uint64_t k, uint64_t n,
                  KernelVariant variant, hipStream_t stream) noexcept;

hipError_t launch_short_mixed_f32_to_bf16(const float *output_f32,
                                          uint16_t *output,
                                          uint64_t element_count,
                                          hipStream_t stream) noexcept;

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
                        KernelVariant variant, hipStream_t stream) noexcept;

hipError_t launch_nvfp4_quantize(const uint16_t *activation,
                                 uint8_t *packed_activation,
                                 uint8_t *block_scales,
                                 const float *input_tensor_scale, uint64_t m,
                                 uint64_t k, hipStream_t stream) noexcept;

hipError_t launch_nvfp4_w4a4(
    const uint8_t *packed_activation, const uint8_t *activation_block_scales,
    const uint8_t *packed_weight, const uint8_t *weight_block_scales,
    const float *weight_tensor_scale, const float *input_tensor_scale,
    uint16_t *output, uint64_t m, uint64_t k, uint64_t n, KernelVariant variant,
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
                             uint64_t n, KernelVariant variant,
                             hipStream_t stream) noexcept;

hipError_t launch_mxfp8_quantize(const uint16_t *activation, uint8_t *quantized,
                                 uint8_t *block_scales, uint64_t m, uint64_t k,
                                 hipStream_t stream) noexcept;

hipError_t launch_mxfp8_w8a8(const uint8_t *activation,
                             const uint8_t *activation_block_scales,
                             const uint8_t *weight,
                             const uint8_t *weight_block_scales,
                             uint16_t *output, uint64_t m, uint64_t k,
                             uint64_t n, KernelVariant variant,
                             hipStream_t stream) noexcept;

hipError_t launch_mxfp6_quantize(const uint16_t *activation,
                                 uint8_t *packed_activation,
                                 uint8_t *block_scales, uint64_t m, uint64_t k,
                                 hipStream_t stream) noexcept;

hipError_t launch_mxfp6_w6a6(const uint8_t *packed_activation,
                             const uint8_t *activation_block_scales,
                             const uint8_t *packed_weight,
                             const uint8_t *weight_block_scales,
                             uint16_t *output, uint64_t m, uint64_t k,
                             uint64_t n, KernelVariant variant,
                             hipStream_t stream) noexcept;

} // namespace sllm_matmul_kernel

#endif // SLLM_MATMUL_KERNEL_INTERNAL_HPP
