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
constexpr const char *kFp8OuterPrefillTiled16LogicalKernelId =
    "matmul.fp8.outer.prefill.tiled16.v1";
constexpr const char *kFp8OuterPrefillTiled16DeviceSymbol =
    "sllm_matmul_fp8_outer_prefill_tiled16_v1";
constexpr const char *kFp8OuterPrefillGfx1030Half2LogicalKernelId =
    "matmul.fp8.outer.prefill.gfx1030.half2.128x64.v1";
constexpr const char *kFp8OuterPrefillGfx1030Half2DeviceSymbol =
    "sllm_matmul_fp8_outer_prefill_gfx1030_half2_128x64_v1";
constexpr const char *kFp8OuterPrefillGfx1030Half2_64x64LogicalKernelId =
    "matmul.fp8.outer.prefill.gfx1030.half2.64x64.v1";
constexpr const char *kFp8OuterPrefillGfx1030Half2_64x64DeviceSymbol =
    "sllm_matmul_fp8_outer_prefill_gfx1030_half2_64x64_v1";
constexpr const char *kFp8OuterPrefillGfx1030Half2_64x64Environment =
    "SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_HALF2_64X64";
// Phase 78 ID85: the gfx1030 64x64/K32 prefill tile with a resident padded
// E4M3FN-to-FP16 LUT.  The candidate is opt-in and keeps ID71's broad prefill
// shape contract; FNUZ and other targets retain their existing providers.
constexpr const char *kFp8OuterPrefillGfx1030LdsLutLogicalKernelId =
    "matmul.fp8.outer.prefill.gfx1030.lds_lut.64x64.v1";
constexpr const char *kFp8OuterPrefillGfx1030LdsLutDeviceSymbol =
    "sllm_matmul_fp8_outer_prefill_gfx1030_lds_lut_v1";
constexpr const char *kFp8OuterPrefillGfx1030LdsLutEnvironment =
    "SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_LDS_LUT";
constexpr uint32_t kFp8OuterPrefillGfx1030LdsLutWorkgroupSize = 256U;
constexpr uint32_t kFp8OuterPrefillGfx1030LdsLutStaticLdsBytes = 9248U;
// Phase 78 ID70: exact gfx1030 OCP E4M3FN prefill pipeline. The resident
// bytes are expanded only into a context-owned transient FP16 workspace,
// consumed by rocBLAS with FP32 accumulation, and scaled into BF16 output.
// The opt-in intentionally excludes decode, FNUZ, the vocabulary projection,
// and non-aligned shapes while its N2 accumulation-order impact is evaluated.
constexpr const char *kFp8OuterPrefillGfx1030F16StagingLogicalKernelId =
    "matmul.fp8.outer.prefill.gfx1030.f16_staging.v1";
constexpr const char *kFp8OuterPrefillGfx1030F16StagingDeviceSymbol =
    "rocblas_gemm_ex";
constexpr const char *kFp8OuterPrefillGfx1030F16StagingEnvironment =
    "SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_F16_STAGING";
constexpr uint64_t kMatmulF16StagingAlignment = 256U;
constexpr uint64_t kFp8OuterPrefillGfx1030F16StagingMinM = 128U;
constexpr uint64_t kFp8OuterPrefillGfx1030F16StagingMaxK = 17408U;
constexpr uint64_t kFp8OuterPrefillGfx1030F16StagingMaxN = 17408U;
// Phase 78 ID86: the same transient FP16 ingress as ID70, followed by the
// measured gfx1030 64x64/K32 half2 consumer. The FP8 model storage remains
// resident; only the context-owned staging arena is used during execution.
constexpr const char *kFp8OuterPrefillGfx1030F16TileStagingLogicalKernelId =
    "matmul.fp8.outer.prefill.gfx1030.f16_tile.v1";
constexpr const char *kFp8OuterPrefillGfx1030F16TileStagingDeviceSymbol =
    "sllm_matmul_fp8_outer_prefill_gfx1030_f16_tile_staging_v1";
constexpr const char *kFp8OuterPrefillGfx1030F16TileStagingEnvironment =
    "SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_F16_TILE_STAGING";
struct F16StagingWorkspaceLayout final {
  uint64_t activation_offset;
  uint64_t activation_bytes;
  uint64_t weight_offset;
  uint64_t weight_bytes;
  uint64_t output_offset;
  uint64_t output_bytes;
  uint64_t total_bytes;
};
constexpr const char *kFp8OuterDecodeGfx1030Half2Wave4Col32LogicalKernelId =
    "matmul.fp8.outer.decode.gfx1030.half2.wave4col32.v1";
constexpr const char *kFp8OuterDecodeGfx1030Half2Wave4Col32DeviceSymbol =
    "sllm_matmul_fp8_outer_decode_gfx1030_half2_wave4col32_v1";
constexpr const char *kFp8OuterDecodeGfx1030Half2Environment =
    "SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_HALF2";
constexpr const char *kFp8OuterDecodeGfx1030Dword8LogicalKernelId =
    "matmul.fp8.outer.decode.gfx1030.dword8.wave4col32.v1";
constexpr const char *kFp8OuterDecodeGfx1030Dword8DeviceSymbol =
    "sllm_matmul_fp8_outer_decode_gfx1030_dword8_wave4col32_v1";
constexpr const char *kFp8OuterDecodeGfx1030Dword8Environment =
    "SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_DWORD8";
// Phase 78 ID75/76: exact gfx1030 FP8 outer-scale M=1 activation-LDS reuse.
// The single literal opt-in selects only the measured Qwen3.8 shape cache;
// all other shapes remain on the ID68 dword8 rollback under this opt-in.
constexpr const char *kFp8OuterDecodeGfx1030ActivationSharedEnvironment =
    "SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_ACTIVATION_SHARED";
constexpr const char
    *kFp8OuterDecodeGfx1030ActivationSharedWave4LogicalKernelId =
        "matmul.fp8.outer.decode.gfx1030.activation_shared.wave4col32.v1";
constexpr const char *kFp8OuterDecodeGfx1030ActivationSharedWave4DeviceSymbol =
    "sllm_matmul_fp8_outer_decode_gfx1030_actshared_wave4col32_v1";
constexpr const char
    *kFp8OuterDecodeGfx1030ActivationSharedWave8LogicalKernelId =
        "matmul.fp8.outer.decode.gfx1030.activation_shared.wave8col64.v1";
constexpr const char *kFp8OuterDecodeGfx1030ActivationSharedWave8DeviceSymbol =
    "sllm_matmul_fp8_outer_decode_gfx1030_actshared_wave8col64_v1";
// Phase 78 ID82: exact gfx1030 M=1 E4M3FN ingress through a resident padded
// 256-entry FP16-bit LUT in LDS.  The selector is opt-in and reuses the ID68
// shape boundary; baseline, FNUZ, and other targets retain their rollback.
constexpr const char *kFp8OuterDecodeGfx1030LdsLutLogicalKernelId =
    "matmul.fp8.outer.decode.gfx1030.lds_lut.wave4col32.v1";
constexpr const char *kFp8OuterDecodeGfx1030LdsLutDeviceSymbol =
    "sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_wave4col32_v1";
// The four exact Qwen38 decode tuples use separate code objects so their
// high-register rolled loops cannot affect the broad ID82 shape family.
constexpr const char *kFp8OuterDecodeGfx1030LdsLutK5120N17408DeviceSymbol =
    "sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_k5120n17408_v1";
constexpr const char *kFp8OuterDecodeGfx1030LdsLutK6144N5120DeviceSymbol =
    "sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_k6144n5120_v1";
constexpr const char *kFp8OuterDecodeGfx1030LdsLutK5120N10240DeviceSymbol =
    "sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_k5120n10240_v1";
constexpr const char *kFp8OuterDecodeGfx1030LdsLutK5120N6144DeviceSymbol =
    "sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_k5120n6144_v1";
constexpr const char *kFp8OuterDecodeGfx1030LdsLutEnvironment =
    "SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_LDS_LUT";
constexpr uint32_t kFp8OuterDecodeGfx1030LdsLutWorkgroupSize = 256U;
constexpr uint32_t kFp8OuterDecodeGfx1030LdsLutStaticLdsBytes = 544U;
constexpr const char *kFp8OuterDecodeBaselineEnvironment =
    "SLLM_FP8_OUTER_DECODE_FORCE_BASELINE";
constexpr uint32_t kFp8OuterDecodeGfx1030Half2WorkgroupSize = 256U;
constexpr uint32_t kFp8OuterDecodeGfx1030Half2WaveSize = 32U;
constexpr uint32_t kFp8OuterDecodeGfx1030Half2ColumnsPerWave = 4U;
constexpr uint32_t kFp8OuterDecodeGfx1030Half2WavesPerWorkgroup =
    kFp8OuterDecodeGfx1030Half2WorkgroupSize /
    kFp8OuterDecodeGfx1030Half2WaveSize;
constexpr uint32_t kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup =
    kFp8OuterDecodeGfx1030Half2WavesPerWorkgroup *
    kFp8OuterDecodeGfx1030Half2ColumnsPerWave;
constexpr uint64_t kFp8OuterDecodeGfx1030Half2MinK = 64U;
constexpr uint64_t kFp8OuterDecodeGfx1030Half2MaxK = 17408U;
static_assert(sizeof("matmul.fp8.outer.prefill.tiled16.v1") <= 64U);
static_assert(sizeof("sllm_matmul_fp8_outer_prefill_tiled16_v1") <= 64U);
static_assert(sizeof("matmul.fp8.outer.prefill.gfx1030.half2.128x64.v1") <=
              64U);
static_assert(sizeof("sllm_matmul_fp8_outer_prefill_gfx1030_half2_128x64_v1") <=
              64U);
static_assert(sizeof("matmul.fp8.outer.prefill.gfx1030.half2.64x64.v1") <= 64U);
static_assert(sizeof("sllm_matmul_fp8_outer_prefill_gfx1030_half2_64x64_v1") <=
              64U);
static_assert(sizeof("matmul.fp8.outer.prefill.gfx1030.lds_lut.64x64.v1") <=
              64U);
static_assert(sizeof("sllm_matmul_fp8_outer_prefill_gfx1030_lds_lut_v1") <=
              64U);
static_assert(kFp8OuterPrefillGfx1030LdsLutStaticLdsBytes == 9248U);
static_assert(sizeof("matmul.fp8.outer.prefill.gfx1030.f16_staging.v1") <= 64U);
static_assert(sizeof("rocblas_gemm_ex") <= 64U);
static_assert(sizeof("matmul.fp8.outer.prefill.gfx1030.f16_tile.v1") <= 64U);
static_assert(
    sizeof("sllm_matmul_fp8_outer_prefill_gfx1030_f16_tile_staging_v1") <= 64U);
static_assert(sizeof("matmul.fp8.outer.decode.gfx1030.half2.wave4col32.v1") <=
              64U);
static_assert(
    sizeof("sllm_matmul_fp8_outer_decode_gfx1030_half2_wave4col32_v1") <= 64U);
static_assert(sizeof("matmul.fp8.outer.decode.gfx1030.dword8.wave4col32.v1") <=
              64U);
static_assert(
    sizeof("sllm_matmul_fp8_outer_decode_gfx1030_dword8_wave4col32_v1") <= 64U);
static_assert(
    sizeof("matmul.fp8.outer.decode.gfx1030.activation_shared.wave4col32.v1") <=
    64U);
static_assert(
    sizeof("sllm_matmul_fp8_outer_decode_gfx1030_actshared_wave4col32_v1") <=
    64U);
static_assert(
    sizeof("matmul.fp8.outer.decode.gfx1030.activation_shared.wave8col64.v1") <=
    64U);
static_assert(
    sizeof("sllm_matmul_fp8_outer_decode_gfx1030_actshared_wave8col64_v1") <=
    64U);
static_assert(sizeof("matmul.fp8.outer.decode.gfx1030.lds_lut.wave4col32.v1") <=
              64U);
static_assert(
    sizeof("sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_wave4col32_v1") <=
    64U);
static_assert(
    sizeof("sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_k5120n17408_v1") <=
    64U);
static_assert(
    sizeof("sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_k6144n5120_v1") <=
    64U);
static_assert(
    sizeof("sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_k5120n10240_v1") <=
    64U);
static_assert(kFp8OuterDecodeGfx1030LdsLutStaticLdsBytes == 544U);
static_assert(kFp8OuterDecodeGfx1030Half2WavesPerWorkgroup == 8U);
static_assert(kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup == 32U);
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
constexpr const char *kNvfp4W4A4DecodeLogicalKernelId =
    "matmul.nvfp4.w4a4.block16.decode.v1";
constexpr const char *kNvfp4W4A4DecodeDeviceSymbol =
    "sllm_matmul_nvfp4_w4a4_block16_decode_v1";
constexpr const char *kNvfp4W4A4PrefillRow8LogicalKernelId =
    "matmul.nvfp4.w4a4.block16.prefill.row8_tiled256.v1";
constexpr const char *kNvfp4W4A4PrefillRow8DeviceSymbol =
    "sllm_matmul_nvfp4_w4a4_block16_prefill_row8_tiled256_v1";
constexpr const char *kNvfp4W4A4PrefillRow8Col8LogicalKernelId =
    "matmul.nvfp4.w4a4.block16.prefill.row8_col8_tiled256.v1";
constexpr const char *kNvfp4W4A4PrefillRow8Col8DeviceSymbol =
    "sllm_matmul_nvfp4_w4a4_block16_prefill_row8_col8_tiled256_v1";
constexpr const char *kNvfp4W4A4PrefillDp4a64x64LogicalKernelId =
    "matmul.nvfp4.w4a4.block16.prefill.dp4a64x64.v1";
constexpr const char *kNvfp4W4A4PrefillDp4a64x64DeviceSymbol =
    "sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_v1";
constexpr const char *kNvfp4W4A4PrefillDp4a64x64Index32PipelineDeviceSymbol =
    "sllm_nvfp4_w4a4_prefill_dp4a64x64_index32_pipeline_v1";
static_assert(sizeof("sllm_nvfp4_w4a4_prefill_dp4a64x64_index32_pipeline_v1") <=
              64U);
constexpr const char *kNvfp4W4A4PrefillDp4a64x64K128LogicalKernelId =
    "matmul.nvfp4.w4a4.block16.prefill.dp4a64x64_k128.v1";
constexpr const char *kNvfp4W4A4PrefillDp4a64x64K128DeviceSymbol =
    "sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_k128_v1";
constexpr const char *kNvfp4W4A4PrefillDp4a64x64K128Environment =
    "SLLM_NVFP4_W4A4_PREFILL_FORCE_DP4A_K128";
constexpr const char *kNvfp4W4A4PrefillGfx1201Wmma128x64LogicalKernelId =
    "matmul.nvfp4.w4a4.prefill.gfx1201.wmma128x64.v1";
constexpr const char *kNvfp4W4A4PrefillGfx1201Wmma128x64DeviceSymbol =
    "sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_v1";
constexpr const char *kNvfp4W4A4Gfx1201WmmaOrdinaryDeviceSymbol =
    "sllm_nvfp4_w4a4_prefill_gfx1201_wmma_ordinary_v1";
static_assert(sizeof("sllm_nvfp4_w4a4_prefill_gfx1201_wmma_ordinary_v1") <=
              64U);

constexpr bool
phase78_gfx1201_nvfp4_wmma_ordinary_shape(const uint64_t m, const uint64_t k,
                                          const uint64_t n) noexcept {
  return m == 17U && k == 5120U && n == 17408U;
}
static_assert(phase78_gfx1201_nvfp4_wmma_ordinary_shape(17U, 5120U, 17408U));
static_assert(!phase78_gfx1201_nvfp4_wmma_ordinary_shape(16U, 5120U, 17408U));
static_assert(!phase78_gfx1201_nvfp4_wmma_ordinary_shape(18U, 5120U, 17408U));
static_assert(!phase78_gfx1201_nvfp4_wmma_ordinary_shape(17U, 17408U, 5120U));

// Phase 78 ID81: the ID64-order 128x32 geometry validated by the standalone
// tile sweep.  Keep the candidate explicitly opt-in and exact-gfx1201 only.
constexpr const char *kNvfp4W4A4PrefillGfx1201Wmma128x32LogicalKernelId =
    "matmul.nvfp4.w4a4.prefill.gfx1201.wmma128x32.v1";
constexpr const char *kNvfp4W4A4PrefillGfx1201Wmma128x32DeviceSymbol =
    "sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x32_v1";
constexpr const char *kNvfp4W4A4PrefillGfx1201Wmma128x32Environment =
    "SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA_128X32";
// Phase 78 ID69: gfx1201 FP16 WMMA with NVFP4 block scales absorbed at LDS
// ingress.  This remains opt-in until numerical and resource evidence is
// available; ID64 is intentionally left as a separate candidate.
constexpr const char
    *kNvfp4W4A4PrefillGfx1201WmmaF16Scale128x64LogicalKernelId =
        "matmul.nvfp4.w4a4.prefill.gfx1201.wmma_f16scale128x64.v1";
constexpr const char *kNvfp4W4A4PrefillGfx1201WmmaF16Scale128x64DeviceSymbol =
    "sllm_nvfp4_w4a4_prefill_gfx1201_wmma_f16scale128x64_v1";
constexpr const char *kNvfp4W4A4PrefillGfx1201WmmaF16ScaleEnvironment =
    "SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA_F16SCALE";
constexpr uint32_t kNvfp4W4A4PrefillGfx1201WmmaF16ScaleWorkgroupSize = 256U;
constexpr uint32_t kNvfp4W4A4PrefillGfx1201WmmaF16ScaleRowsPerWorkgroup = 128U;
constexpr uint32_t kNvfp4W4A4PrefillGfx1201WmmaF16ScaleColumnsPerWorkgroup =
    64U;
constexpr uint32_t kNvfp4W4A4PrefillGfx1201WmmaF16ScaleBlockK = 16U;
constexpr uint32_t kNvfp4W4A4PrefillGfx1201WmmaF16ScaleStageK = 32U;
// Phase 78 ID72: exact-gfx1201 NVFP4 W4A4 prefill pipeline.  Packed E2M1
// values and their block16 E4M3 scales are expanded into transient FP16,
// rocBLAS accumulates in FP32, and a device-side tensor-scale epilogue emits
// BF16.  It remains an explicit N2 candidate until the reduction tree is
// accepted; ID64 stays available as the rollback provider.
constexpr const char *kNvfp4W4A4PrefillGfx1201F16StagingLogicalKernelId =
    "matmul.nvfp4.w4a4.prefill.gfx1201.f16_staging.v1";
constexpr const char *kNvfp4W4A4PrefillGfx1201F16StagingDeviceSymbol =
    "rocblas_gemm_ex";
constexpr const char *kNvfp4W4A4PrefillGfx1201F16StagingEnvironment =
    "SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_F16_STAGING";
// Phase 78 ID83: exact gfx1201 NVFP4 W4A4 prefill staging into reusable FP8
// byte planes, followed by native hipBLASLt E4M3FN/BF16 GEMM.  This remains an
// explicit opt-in; ID64 and all non-Qwen/tail shapes are the rollback path.
constexpr const char *kNvfp4W4A4PrefillGfx1201Fp8StagingLogicalKernelId =
    "matmul.nvfp4.w4a4.prefill.gfx1201.fp8_staging.v1";
constexpr const char *kNvfp4W4A4PrefillGfx1201Fp8StagingDeviceSymbol =
    "hipblasLtMatmul";
constexpr const char *kNvfp4W4A4PrefillGfx1201Fp8StagingEnvironment =
    "SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_FP8_STAGING";
constexpr uint32_t kNvfp4W4A4PrefillGfx1201Fp8StagingWorkgroupSize = 256U;
// Phase 78 opt-in M=1 decode candidate. One 128-thread workgroup owns 128
// adjacent columns and reuses the packed activation row from dynamic LDS.
constexpr const char *kNvfp4W4A4DecodeColumns128LogicalKernelId =
    "matmul.nvfp4.w4a4.decode.columns128.v1";
constexpr const char *kNvfp4W4A4DecodeColumns128DeviceSymbol =
    "sllm_matmul_nvfp4_w4a4_decode_columns128_v1";
constexpr const char *kNvfp4W4A4DecodeColumns128Environment =
    "SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_COLUMNS";
constexpr uint32_t kNvfp4W4A4DecodeColumns128WorkgroupSize = 128U;
// Qwen3.8-27B's largest Phase 78 projection K is intermediate_size=17,408.
// This bounds dynamic LDS to 13,056 bytes (packed activation plus FP32 scales).
constexpr uint64_t kNvfp4W4A4DecodeColumns128MaxK = 17408U;
constexpr const char *kNvfp4ActivationQuantizeWave8Environment =
    "SLLM_NVFP4_ACTIVATION_QUANTIZE_WAVE8";
constexpr const char *kNvfp4W4A4DecodeWave4Column32LogicalKernelId =
    "matmul.nvfp4.w4a4.decode.dp4a.wave4col32.v1";
constexpr const char *kNvfp4W4A4DecodeWave4Column32DeviceSymbol =
    "sllm_matmul_nvfp4_w4a4_decode_dp4a_wave4col32_v1";
constexpr const char *kNvfp4W4A4DecodeWave4Column32Environment =
    "SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_WAVE4";
constexpr uint32_t kNvfp4W4A4DecodeWave4Column32WorkgroupSize = 256U;
constexpr const char *kNvfp4W4A4DecodeActivationSharedLogicalKernelId =
    "matmul.nvfp4.w4a4.decode.dp4a.activation_shared.wave4col32.v1";
constexpr const char *kNvfp4W4A4DecodeActivationSharedDeviceSymbol =
    "sllm_matmul_nvfp4_w4a4_decode_dp4a_activation_shared_v1";
constexpr const char *kNvfp4W4A4DecodeActivationSharedEnvironment =
    "SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_ACTIVATION_SHARED";
constexpr uint32_t kNvfp4W4A4DecodeActivationSharedWorkgroupSize = 256U;
// Phase 78 ID84: target-specific decode implementations share one opt-in
// candidate identity.  gfx1201 uses the ID67 geometry and gfx1030 uses the
// ID73 activation-shared geometry; both retain the existing exact shape
// predicates and FP32 block-scale accumulation order.
constexpr const char *kNvfp4W4A4DecodeScaleLutLogicalKernelId =
    "matmul.nvfp4.w4a4.decode.scale_lut.v1";
// The public ABI reserves 64 bytes including its terminator.  The target
// implementation symbols in the include are longer than that, so production
// dispatch uses this short wrapper entry point and keeps the implementation
// symbols private to the translation unit.
constexpr const char *kNvfp4W4A4DecodeScaleLutDeviceSymbol =
    "sllm_matmul_nvfp4_w4a4_decode_scale_lut_v1";
constexpr const char
    *kNvfp4W4A4DecodeScaleLutGfx1201ActivationSharedDeviceSymbol =
        "sllm_nvfp4_w4a4_decode_scale_lut_gfx1201_actshared_v1";
constexpr const char *kNvfp4W4A4DecodeScaleLutEnvironment =
    "SLLM_NVFP4_W4A4_DECODE_FORCE_LDS_F32_LUT";
constexpr uint32_t kNvfp4W4A4DecodeScaleLutWorkgroupSize = 256U;
constexpr uint32_t kNvfp4W4A4DecodeScaleLutStaticLdsBytes = 1056U;
static_assert(sizeof("matmul.nvfp4.w4a4.decode.scale_lut.v1") <= 64U);
static_assert(sizeof("sllm_matmul_nvfp4_w4a4_decode_scale_lut_v1") <= 64U);
static_assert(sizeof("sllm_nvfp4_w4a4_decode_scale_lut_gfx1201_actshared_v1") <=
              64U);
static_assert(kNvfp4W4A4DecodeScaleLutStaticLdsBytes == 1056U);
static_assert(sizeof("matmul.nvfp4.w4a4.prefill.gfx1201.wmma128x64.v1") <= 64U);
static_assert(sizeof("sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_v1") <= 64U);
static_assert(sizeof("matmul.nvfp4.w4a4.block16.prefill.dp4a64x64_k128.v1") <=
              64U);
static_assert(
    sizeof("sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_k128_v1") <= 64U);
static_assert(
    sizeof("matmul.nvfp4.w4a4.prefill.gfx1201.wmma_f16scale128x64.v1") <= 64U);
static_assert(
    sizeof("sllm_nvfp4_w4a4_prefill_gfx1201_wmma_f16scale128x64_v1") <= 64U);
static_assert(sizeof("matmul.nvfp4.w4a4.prefill.gfx1201.f16_staging.v1") <=
              64U);
static_assert(sizeof("matmul.nvfp4.w4a4.prefill.gfx1201.fp8_staging.v1") <=
              64U);
static_assert(sizeof("hipblasLtMatmul") <= 64U);
static_assert(kNvfp4W4A4PrefillGfx1201WmmaF16ScaleWorkgroupSize == 256U);
static_assert(kNvfp4W4A4PrefillGfx1201WmmaF16ScaleRowsPerWorkgroup == 128U);
static_assert(kNvfp4W4A4PrefillGfx1201WmmaF16ScaleColumnsPerWorkgroup == 64U);
static_assert(kNvfp4W4A4PrefillGfx1201WmmaF16ScaleStageK %
                  kNvfp4W4A4PrefillGfx1201WmmaF16ScaleBlockK ==
              0U);
static_assert(sizeof("matmul.nvfp4.w4a4.decode.columns128.v1") <= 64U);
static_assert(sizeof("sllm_matmul_nvfp4_w4a4_decode_columns128_v1") <= 64U);
static_assert(sizeof("matmul.nvfp4.w4a4.decode.dp4a.wave4col32.v1") <= 64U);
static_assert(sizeof("sllm_matmul_nvfp4_w4a4_decode_dp4a_wave4col32_v1") <=
              64U);
static_assert(
    sizeof("matmul.nvfp4.w4a4.decode.dp4a.activation_shared.wave4col32.v1") <=
    64U);
static_assert(
    sizeof("sllm_matmul_nvfp4_w4a4_decode_dp4a_activation_shared_v1") <= 64U);
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
constexpr const char *kMxfp8W8A8PrefillMmqGfx1030RegscaleLogicalKernelId =
    "matmul.mxfp8.w8a8.gfx1030.mmq-col8.regscale.v1";
constexpr const char *kMxfp8W8A8PrefillMmqGfx1030RegscaleDeviceSymbol =
    "sllm_mxfp8_w8a8_gfx1030_mmq_col8_regscale_v1";
constexpr const char *kMxfp8W8A8PrefillMmqGfx1030Vector32LogicalKernelId =
    "matmul.mxfp8.w8a8.gfx1030.mmq-col8.vector32.v1";
constexpr const char *kMxfp8W8A8PrefillMmqGfx1030Vector32DeviceSymbol =
    "sllm_mxfp8_w8a8_gfx1030_mmq_col8_vector32_v1";
constexpr const char
    *kMxfp8W8A8PrefillMmqGfx1030RegscaleVector32LogicalKernelId =
        "matmul.mxfp8.w8a8.gfx1030.mmq-col8.regscale-vector32.v1";
constexpr const char *kMxfp8W8A8PrefillMmqGfx1030RegscaleVector32DeviceSymbol =
    "sllm_mxfp8_w8a8_gfx1030_mmq_col8_regscale_vector32_v1";
constexpr const char *kMxfp8W8A8PrefillGfx1030Half2_32x32K32LogicalKernelId =
    "matmul.mxfp8.w8a8.gfx1030.half2.32x32.k32.v1";
constexpr const char *kMxfp8W8A8PrefillGfx1030Half2_32x32K32DeviceSymbol =
    "sllm_mxfp8_w8a8_gfx1030_half2_32x32_k32_v1";
constexpr const char *kMxfp8W8A8PrefillGfx1030Half2_64x64K32LogicalKernelId =
    "matmul.mxfp8.w8a8.gfx1030.half2.64x64.k32.v1";
constexpr const char *kMxfp8W8A8PrefillGfx1030Half2_64x64K32DeviceSymbol =
    "sllm_mxfp8_w8a8_gfx1030_half2_64x64_k32_v1";
constexpr const char *kMxfp8W8A8PrefillGfx1030Half2_128x32K32LogicalKernelId =
    "matmul.mxfp8.w8a8.gfx1030.half2.128x32.k32.v1";
constexpr const char *kMxfp8W8A8PrefillGfx1030Half2_128x32K32DeviceSymbol =
    "sllm_mxfp8_w8a8_gfx1030_half2_128x32_k32_v1";
constexpr const char *kMxfp8W8A8PrefillGfx1030Half2_128x64K32LogicalKernelId =
    "matmul.mxfp8.w8a8.gfx1030.half2.128x64.k32.v1";
constexpr const char *kMxfp8W8A8PrefillGfx1030Half2_128x64K32DeviceSymbol =
    "sllm_mxfp8_w8a8_gfx1030_half2_128x64_k32_v1";
constexpr const char *kMxfp8W8A8PrefillGfx1030Half2_128x64K64LogicalKernelId =
    "matmul.mxfp8.w8a8.gfx1030.half2.128x64.k64.v1";
constexpr const char *kMxfp8W8A8PrefillGfx1030Half2_128x64K64DeviceSymbol =
    "sllm_mxfp8_w8a8_gfx1030_half2_128x64_k64_v1";
constexpr const char *kMxfp8W8A8PrefillGfx1030Half2_128x64K128LogicalKernelId =
    "matmul.mxfp8.w8a8.gfx1030.half2.128x64.k128.v1";
constexpr const char *kMxfp8W8A8PrefillGfx1030Half2_128x64K128DeviceSymbol =
    "sllm_mxfp8_w8a8_gfx1030_half2_128x64_k128_v1";
constexpr const char
    *kMxfp8W8A8PrefillGfx1030Half2_128x64K32DoubleLogicalKernelId =
        "matmul.mxfp8.w8a8.gfx1030.half2.128x64.k32.double.v1";
constexpr const char
    *kMxfp8W8A8PrefillGfx1030Half2_128x64K32DoubleDeviceSymbol =
        "sllm_mxfp8_w8a8_gfx1030_half2_128x64_k32_double_v1";
static_assert(sizeof("matmul.mxfp8.w8a8.gfx1030.mmq-col8.regscale.v1") <= 64U);
static_assert(sizeof("matmul.mxfp8.w8a8.gfx1030.mmq-col8.vector32.v1") <= 64U);
static_assert(
    sizeof("matmul.mxfp8.w8a8.gfx1030.mmq-col8.regscale-vector32.v1") <= 64U);
static_assert(sizeof("sllm_mxfp8_w8a8_gfx1030_mmq_col8_regscale_v1") <= 64U);
static_assert(sizeof("sllm_mxfp8_w8a8_gfx1030_mmq_col8_vector32_v1") <= 64U);
static_assert(sizeof("sllm_mxfp8_w8a8_gfx1030_mmq_col8_regscale_vector32_v1") <=
              64U);
static_assert(sizeof("matmul.mxfp8.w8a8.gfx1030.half2.32x32.k32.v1") <= 64U);
static_assert(sizeof("matmul.mxfp8.w8a8.gfx1030.half2.64x64.k32.v1") <= 64U);
static_assert(sizeof("matmul.mxfp8.w8a8.gfx1030.half2.128x32.k32.v1") <= 64U);
static_assert(sizeof("matmul.mxfp8.w8a8.gfx1030.half2.128x64.k32.v1") <= 64U);
static_assert(sizeof("sllm_mxfp8_w8a8_gfx1030_half2_32x32_k32_v1") <= 64U);
static_assert(sizeof("sllm_mxfp8_w8a8_gfx1030_half2_64x64_k32_v1") <= 64U);
static_assert(sizeof("sllm_mxfp8_w8a8_gfx1030_half2_128x32_k32_v1") <= 64U);
static_assert(sizeof("sllm_mxfp8_w8a8_gfx1030_half2_128x64_k32_v1") <= 64U);
static_assert(sizeof("matmul.mxfp8.w8a8.gfx1030.half2.128x64.k64.v1") <= 64U);
static_assert(sizeof("sllm_mxfp8_w8a8_gfx1030_half2_128x64_k64_v1") <= 64U);
static_assert(sizeof("matmul.mxfp8.w8a8.gfx1030.half2.128x64.k128.v1") <= 64U);
static_assert(sizeof("sllm_mxfp8_w8a8_gfx1030_half2_128x64_k128_v1") <= 64U);
static_assert(sizeof("matmul.mxfp8.w8a8.gfx1030.half2.128x64.k32.double.v1") <=
              64U);
static_assert(sizeof("sllm_mxfp8_w8a8_gfx1030_half2_128x64_k32_double_v1") <=
              64U);
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
constexpr const char *kMxfp8W8A8PrefillMmqGfx1030Phase69Environment =
    "SLLM_MXFP8_PREFILL_FORCE_MMQ_GFX1030_PHASE69";
constexpr const char *kMxfp8W8A8PrefillGfx1030Phase75Environment =
    "SLLM_MXFP8_PREFILL_FORCE_PHASE75";
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
constexpr const char *kMxfp6W6A6PrefillMmqGfx1030ViaE4M3LogicalKernelId =
    "matmul.mxfp6.w6a6.gfx1030.mmq-col8.via-e4m3.v1";
constexpr const char *kMxfp6W6A6PrefillMmqGfx1030ViaE4M3DeviceSymbol =
    "sllm_mxfp6_w6a6_gfx1030_mmq_col8_via_e4m3_v1";
constexpr const char *kMxfp6W6A6PrefillGfx1030Half2Dot2LogicalKernelId =
    "matmul.mxfp6.w6a6.gfx1030.half2.32x32.v1";
constexpr const char *kMxfp6W6A6PrefillGfx1030Half2Dot2DeviceSymbol =
    "sllm_mxfp6_w6a6_gfx1030_half2_32x32_v1";
constexpr const char
    *kMxfp6W6A6PrefillGfx1030Half2_128x64K32DoubleScalarLogicalKernelId =
        "matmul.mxfp6.w6a6.gfx1030.half2.128x64.k32d.scalar.v1";
constexpr const char
    *kMxfp6W6A6PrefillGfx1030Half2_128x64K32DoubleScalarDeviceSymbol =
        "sllm_mxfp6_w6a6_gfx1030_half2_128x64_k32d_scalar_v1";
constexpr const char
    *kMxfp6W6A6PrefillGfx1030Half2_128x64K32DoublePack4LogicalKernelId =
        "matmul.mxfp6.w6a6.gfx1030.half2.128x64.k32d.pack4.v1";
constexpr const char
    *kMxfp6W6A6PrefillGfx1030Half2_128x64K32DoublePack4DeviceSymbol =
        "sllm_mxfp6_w6a6_gfx1030_half2_128x64_k32d_pack4_v1";
constexpr const char *kMxfp6W6A6PrefillWmmaGfx1201ViaE4M3N64LogicalKernelId =
    "matmul.mxfp6.w6a6.gfx1201.wmma128x64.via-e4m3.v1";
constexpr const char *kMxfp6W6A6PrefillWmmaGfx1201ViaE4M3N64DeviceSymbol =
    "sllm_mxfp6_w6a6_gfx1201_wmma128x64_via_e4m3_v1";
constexpr const char *kMxfp6W6A6PrefillWmmaGfx1201Pack4N64LogicalKernelId =
    "matmul.mxfp6.w6a6.gfx1201.wmma128x64.pack4.v2";
constexpr const char *kMxfp6W6A6PrefillWmmaGfx1201Pack4N64DeviceSymbol =
    "sllm_mxfp6_w6a6_gfx1201_wmma128x64_pack4_v2";
constexpr const char *kMxfp6W6A6PrefillWmmaGfx1201Pack4SwarLogicalKernelId =
    "matmul.mxfp6.w6a6.gfx1201.wmma128x64.pack4-swar.v1";
constexpr const char *kMxfp6W6A6PrefillWmmaGfx1201Pack4SwarDeviceSymbol =
    "sllm_mxfp6_w6a6_gfx1201_wmma128x64_pack4_swar_v1";
constexpr const char *kMxfp6W6A6PrefillWmmaGfx1201Pack4N128LogicalKernelId =
    "matmul.mxfp6.w6a6.gfx1201.wmma128x128.pack4.v1";
constexpr const char *kMxfp6W6A6PrefillWmmaGfx1201Pack4N128DeviceSymbol =
    "sllm_mxfp6_w6a6_gfx1201_wmma128x128_pack4_v1";
constexpr const char *kMxfp6W6A6PrefillPhase70Environment =
    "SLLM_MXFP6_PREFILL_FORCE_PHASE70";
constexpr const char *kMxfp6W6A6PrefillPhase74Environment =
    "SLLM_MXFP6_PREFILL_FORCE_PHASE74";
constexpr const char *kMxfp6W6A6PrefillPhase75Environment =
    "SLLM_MXFP6_PREFILL_FORCE_PHASE75";
constexpr const char *kMxfp6W6A6PrefillTiled16Environment =
    "SLLM_MXFP6_PREFILL_FORCE_TILED16";
static_assert(sizeof("matmul.mxfp6.w6a6.gfx1030.mmq-col8.via-e4m3.v1") <= 64U);
static_assert(sizeof("matmul.mxfp6.w6a6.gfx1201.wmma128x64.via-e4m3.v1") <=
              64U);
static_assert(sizeof("sllm_mxfp6_w6a6_gfx1030_mmq_col8_via_e4m3_v1") <= 64U);
static_assert(sizeof("matmul.mxfp6.w6a6.gfx1030.half2.32x32.v1") <= 64U);
static_assert(sizeof("sllm_mxfp6_w6a6_gfx1030_half2_32x32_v1") <= 64U);
static_assert(sizeof("matmul.mxfp6.w6a6.gfx1030.half2.128x64.k32d.scalar.v1") <=
              64U);
static_assert(sizeof("sllm_mxfp6_w6a6_gfx1030_half2_128x64_k32d_scalar_v1") <=
              64U);
static_assert(sizeof("matmul.mxfp6.w6a6.gfx1030.half2.128x64.k32d.pack4.v1") <=
              64U);
static_assert(sizeof("sllm_mxfp6_w6a6_gfx1030_half2_128x64_k32d_pack4_v1") <=
              64U);
static_assert(sizeof("sllm_mxfp6_w6a6_gfx1201_wmma128x64_via_e4m3_v1") <= 64U);
static_assert(sizeof("matmul.mxfp6.w6a6.gfx1201.wmma128x64.pack4.v2") <= 64U);
static_assert(sizeof("matmul.mxfp6.w6a6.gfx1201.wmma128x64.pack4-swar.v1") <=
              64U);
static_assert(sizeof("matmul.mxfp6.w6a6.gfx1201.wmma128x128.pack4.v1") <= 64U);
static_assert(sizeof("sllm_mxfp6_w6a6_gfx1201_wmma128x64_pack4_v2") <= 64U);
static_assert(sizeof("sllm_mxfp6_w6a6_gfx1201_wmma128x64_pack4_swar_v1") <=
              64U);
static_assert(sizeof("sllm_mxfp6_w6a6_gfx1201_wmma128x128_pack4_v1") <= 64U);
static_assert(sizeof("matmul.nvfp4.w4a4.block16.prefill.row8_tiled256.v1") <=
              64U);
static_assert(
    sizeof("sllm_matmul_nvfp4_w4a4_block16_prefill_row8_tiled256_v1") <= 64U);
static_assert(
    sizeof("matmul.nvfp4.w4a4.block16.prefill.row8_col8_tiled256.v1") <= 64U);
static_assert(
    sizeof("sllm_matmul_nvfp4_w4a4_block16_prefill_row8_col8_tiled256_v1") <=
    64U);
static_assert(sizeof("matmul.nvfp4.w4a4.block16.prefill.dp4a64x64.v1") <= 64U);
static_assert(sizeof("sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_v1") <=
              64U);

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
  Nvfp4W4A4Decode = 58U,
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
  Mxfp8W8A8PrefillMmqGfx1030Regscale = 40U,
  Mxfp8W8A8PrefillMmqGfx1030Vector32 = 41U,
  Mxfp8W8A8PrefillMmqGfx1030RegscaleVector32 = 42U,
  Mxfp6W6A6PrefillMmqGfx1030ViaE4M3 = 43U,
  Mxfp6W6A6PrefillWmmaGfx1201ViaE4M3N64 = 44U,
  Mxfp6W6A6PrefillWmmaGfx1201Pack4N64 = 45U,
  Mxfp6W6A6PrefillWmmaGfx1201Pack4N128 = 46U,
  Mxfp6W6A6PrefillGfx1030Half2Dot2 = 47U,
  Mxfp6W6A6PrefillWmmaGfx1201Pack4Swar = 48U,
  Mxfp8W8A8PrefillGfx1030Half2_32x32K32 = 49U,
  Mxfp8W8A8PrefillGfx1030Half2_64x64K32 = 50U,
  Mxfp8W8A8PrefillGfx1030Half2_128x32K32 = 51U,
  Mxfp8W8A8PrefillGfx1030Half2_128x64K32 = 52U,
  Mxfp8W8A8PrefillGfx1030Half2_128x64K64 = 53U,
  Mxfp8W8A8PrefillGfx1030Half2_128x64K128 = 54U,
  Mxfp8W8A8PrefillGfx1030Half2_128x64K32Double = 55U,
  Mxfp6W6A6PrefillGfx1030Half2_128x64K32DoubleScalar = 56U,
  Mxfp6W6A6PrefillGfx1030Half2_128x64K32DoublePack4 = 57U,
  Nvfp4W4A4PrefillRow8Tiled256 = 59U,
  Fp8OuterPrefillTiled16 = 60U,
  Nvfp4W4A4PrefillRow8Col8Tiled256 = 61U,
  Nvfp4W4A4PrefillDp4a64x64 = 62U,
  Fp8OuterPrefillGfx1030Half2_128x64 = 63U,
  Nvfp4W4A4PrefillGfx1201Wmma128x64 = 64U,
  Nvfp4W4A4DecodeColumns128 = 65U,
  Fp8OuterDecodeGfx1030Half2Wave4Col32 = 66U,
  Nvfp4W4A4DecodeWave4Column32 = 67U,
  Fp8OuterDecodeGfx1030Dword8Wave4Col32 = 68U,
  Nvfp4W4A4PrefillGfx1201WmmaF16Scale128x64 = 69U,
  Fp8OuterPrefillGfx1030F16Staging = 70U,
  Fp8OuterPrefillGfx1030Half2_64x64 = 71U,
  Nvfp4W4A4PrefillGfx1201F16Staging = 72U,
  Nvfp4W4A4DecodeActivationShared = 73U,
  Fp8OuterDecodeGfx1030ActivationSharedWave4Col32 = 75U,
  Fp8OuterDecodeGfx1030ActivationSharedWave8Col64 = 76U,
  Nvfp4W4A4PrefillDp4a64x64K128 = 80U,
  Nvfp4W4A4PrefillGfx1201Wmma128x32 = 81U,
  Fp8OuterDecodeGfx1030LdsLutWave4Col32 = 82U,
  Nvfp4W4A4PrefillGfx1201Fp8Staging = 83U,
  Nvfp4W4A4DecodeScaleLut = 84U,
  Fp8OuterPrefillGfx1030LdsLut = 85U,
  Fp8OuterPrefillGfx1030F16TileStaging = 86U,
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
// than a model-name table. It starts with large-M, wide projections; the
// user-approved upper bound includes model-independent shapes through N=32768
// while preserving the existing alignment and M/K admission conditions.
constexpr bool phase63_gfx1201_mxfp8_wmma_shape(const uint64_t m,
                                                const uint64_t k,
                                                const uint64_t n) noexcept {
  return m >= 128U && k >= 2048U && n >= 1024U && n <= 32768U &&
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
// small GQA projections down to N=64. Keep the production rule dimension-only;
// irregular or wider-than-32768 vocabulary heads remain on row8.
constexpr bool
phase65_gfx1201_mxfp8_wmma_direct_both_shape(const uint64_t m, const uint64_t k,
                                             const uint64_t n) noexcept {
  return phase65_mxfp8_wmma_direct_activation_supported_shape(m, k, n) &&
         m >= 128U && k >= 2048U && n >= 64U && n <= 32768U;
}

// Phase 66 adopts the wider output tile only inside the measured Phase 65
// production family. This excludes the short-K operator boundary where N128
// was slower, preserves N=64 on the established provider, and leaves tails
// and shapes wider than N=32768 on their existing fail-closed routes.
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
  const char *const force_gfx1030_phase75 =
      std::getenv(kMxfp8W8A8PrefillGfx1030Phase75Environment);
  if (exact_gfx1030 && phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n) &&
      force_gfx1030_phase75 != nullptr) {
    if (std::strcmp(force_gfx1030_phase75, "half2-32x32-k32") == 0) {
      return KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_32x32K32;
    }
    if (std::strcmp(force_gfx1030_phase75, "half2-64x64-k32") == 0) {
      return KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_64x64K32;
    }
    if (std::strcmp(force_gfx1030_phase75, "half2-128x32-k32") == 0) {
      return KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x32K32;
    }
    if (std::strcmp(force_gfx1030_phase75, "half2-128x64-k32") == 0) {
      return KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K32;
    }
    if (std::strcmp(force_gfx1030_phase75, "half2-128x64-k64") == 0) {
      return KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K64;
    }
    if (std::strcmp(force_gfx1030_phase75, "half2-128x64-k128") == 0) {
      return KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K128;
    }
    if (std::strcmp(force_gfx1030_phase75, "half2-128x64-k32-double") == 0) {
      return KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K32Double;
    }
  }
  const char *const force_gfx1030_mmq =
      std::getenv(kMxfp8W8A8PrefillMmqGfx1030ColumnsEnvironment);
  const char *const force_gfx1030_phase69 =
      std::getenv(kMxfp8W8A8PrefillMmqGfx1030Phase69Environment);
  if (exact_gfx1030 && phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
    if (force_gfx1030_phase69 != nullptr &&
        std::strcmp(force_gfx1030_phase69, "regscale") == 0) {
      return KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Regscale;
    }
    if (force_gfx1030_phase69 != nullptr &&
        std::strcmp(force_gfx1030_phase69, "vector32") == 0) {
      return KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Vector32;
    }
    if (force_gfx1030_phase69 != nullptr &&
        std::strcmp(force_gfx1030_phase69, "combined") == 0) {
      return KernelVariant::Mxfp8W8A8PrefillMmqGfx1030RegscaleVector32;
    }
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
    return force_gfx1030_phase69 != nullptr &&
                   std::strcmp(force_gfx1030_phase69, "control") == 0
               ? KernelVariant::Mxfp8W8A8PrefillMmqCol8
               : KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K32Double;
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
static_assert(phase63_gfx1201_mxfp8_wmma_shape(128U, 4096U, 32768U));
static_assert(!phase63_gfx1201_mxfp8_wmma_shape(128U, 4096U, 32832U));
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
static_assert(phase65_gfx1201_mxfp8_wmma_direct_both_shape(128U, 4096U,
                                                           32768U));
static_assert(!phase65_gfx1201_mxfp8_wmma_direct_both_shape(128U, 4096U,
                                                            32832U));
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

constexpr bool
phase70_mxfp6_via_e4m3_supported_shape(const uint64_t m, const uint64_t k,
                                       const uint64_t n) noexcept {
  return m > 1U && n > 0U && k > 0U && (k % kMxfp8W8A8PrefillWmmaBlockK) == 0U;
}

constexpr bool
phase70_gfx1201_mxfp6_wmma_via_e4m3_shape(const uint64_t m, const uint64_t k,
                                          const uint64_t n) noexcept {
  return phase70_mxfp6_via_e4m3_supported_shape(m, k, n) && m >= 17U &&
         k >= 2048U && n >= 1024U && n <= 32768U;
}

constexpr bool
phase70_gfx1201_mxfp6_wmma_pack4_n128_shape(const uint64_t m, const uint64_t k,
                                            const uint64_t n) noexcept {
  return phase70_gfx1201_mxfp6_wmma_via_e4m3_shape(m, k, n) &&
         (n % kMxfp8W8A8PrefillWmmaN128ColumnsPerWorkgroup) == 0U;
}

// Phase 74's gfx1030 half2 dot2 candidate uses complete 32-value blocks and
// has zero-padded 32x32 output tiles, so arbitrary positive M/N tails remain
// safe while M=1 stays on the decode route.
constexpr bool
phase74_gfx1030_mxfp6_half2_dot2_shape(const uint64_t m, const uint64_t k,
                                       const uint64_t n) noexcept {
  return m > 1U && n > 0U && k > 0U && (k % kMxfp8W8A8PrefillWmmaBlockK) == 0U;
}

constexpr bool phase74_gfx1030_mxfp6_half2_dot2_default_shape(
    const uint64_t m, const uint64_t k, const uint64_t n) noexcept {
  return phase74_gfx1030_mxfp6_half2_dot2_shape(m, k, n) && m >= 128U &&
         k >= 2048U && n >= 1024U && n <= 32768U;
}

static_assert(!phase74_gfx1030_mxfp6_half2_dot2_shape(1U, 2560U, 9216U));
static_assert(!phase74_gfx1030_mxfp6_half2_dot2_shape(2U, 2559U, 9216U));
static_assert(phase74_gfx1030_mxfp6_half2_dot2_shape(2U, 2560U, 1U));
static_assert(phase74_gfx1030_mxfp6_half2_dot2_shape(32U, 2560U, 32U));
static_assert(!phase74_gfx1030_mxfp6_half2_dot2_default_shape(127U, 2048U,
                                                              1024U));
static_assert(phase74_gfx1030_mxfp6_half2_dot2_default_shape(128U, 2048U,
                                                             1024U));
static_assert(!phase74_gfx1030_mxfp6_half2_dot2_default_shape(128U, 2016U,
                                                              1024U));
static_assert(!phase74_gfx1030_mxfp6_half2_dot2_default_shape(128U, 2048U,
                                                              1023U));
static_assert(phase74_gfx1030_mxfp6_half2_dot2_default_shape(128U, 5120U,
                                                             17408U));
static_assert(phase74_gfx1030_mxfp6_half2_dot2_default_shape(2048U, 9216U,
                                                             32768U));
static_assert(!phase74_gfx1030_mxfp6_half2_dot2_default_shape(2048U, 9216U,
                                                              32769U));

static_assert(!phase70_gfx1201_mxfp6_wmma_via_e4m3_shape(16U, 2048U, 1024U));
static_assert(phase70_gfx1201_mxfp6_wmma_via_e4m3_shape(17U, 2048U, 1024U));
static_assert(!phase70_gfx1201_mxfp6_wmma_via_e4m3_shape(17U, 2016U, 1024U));
static_assert(!phase70_gfx1201_mxfp6_wmma_via_e4m3_shape(17U, 2048U, 1023U));
static_assert(phase70_gfx1201_mxfp6_wmma_via_e4m3_shape(2048U, 9216U, 32768U));
static_assert(!phase70_gfx1201_mxfp6_wmma_via_e4m3_shape(2048U, 9216U, 32769U));

inline KernelVariant select_mxfp6_variant(const uint64_t m, const uint64_t k,
                                          const uint64_t n,
                                          const char *const target) noexcept {
  if (m == 1U) {
    return KernelVariant::Mxfp6W6A6Decode;
  }
  const char *const force_baseline =
      std::getenv("SLLM_MX_WA_PREFILL_FORCE_BASELINE");
  if (force_baseline != nullptr && std::strcmp(force_baseline, "1") == 0) {
    return KernelVariant::Mxfp6W6A6Prefill;
  }
  const char *const phase70 = std::getenv(kMxfp6W6A6PrefillPhase70Environment);
  const char *const phase74 = std::getenv(kMxfp6W6A6PrefillPhase74Environment);
  const char *const phase75 = std::getenv(kMxfp6W6A6PrefillPhase75Environment);
  if (phase74_gfx1030_mxfp6_half2_dot2_shape(m, k, n) &&
      target_is(target, "gfx1030") && phase75 != nullptr) {
    if (std::strcmp(phase75, "half2-128x64-k32-double-scalar") == 0) {
      return KernelVariant::Mxfp6W6A6PrefillGfx1030Half2_128x64K32DoubleScalar;
    }
    if (std::strcmp(phase75, "half2-128x64-k32-double-pack4") == 0) {
      return KernelVariant::Mxfp6W6A6PrefillGfx1030Half2_128x64K32DoublePack4;
    }
  }
  if (phase74_gfx1030_mxfp6_half2_dot2_shape(m, k, n) &&
      target_is(target, "gfx1030") && phase74 != nullptr &&
      std::strcmp(phase74, "gfx1030-half2-32x32") == 0) {
    return KernelVariant::Mxfp6W6A6PrefillGfx1030Half2Dot2;
  }
  if (phase70_mxfp6_via_e4m3_supported_shape(m, k, n) &&
      target_is(target, "gfx1201") && phase74 != nullptr &&
      std::strcmp(phase74, "gfx1201-swar-pack4") == 0) {
    return KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4Swar;
  }
  if (phase70_mxfp6_via_e4m3_supported_shape(m, k, n) && phase70 != nullptr) {
    if (target_is(target, "gfx1030") && std::strcmp(phase70, "gfx1030") == 0) {
      return KernelVariant::Mxfp6W6A6PrefillMmqGfx1030ViaE4M3;
    }
    if (target_is(target, "gfx1201") &&
        std::strcmp(phase70, "gfx1201-n64") == 0) {
      return KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201ViaE4M3N64;
    }
    if (target_is(target, "gfx1201") &&
        std::strcmp(phase70, "gfx1201-n64-pack4") == 0) {
      return KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4N64;
    }
    if (target_is(target, "gfx1201") &&
        std::strcmp(phase70, "gfx1201-n128-pack4") == 0 &&
        phase70_gfx1201_mxfp6_wmma_pack4_n128_shape(m, k, n)) {
      return KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4N128;
    }
  }
  const KernelVariant mmq =
      select_mx_wa_mmq_variant(false, KernelVariant::Mxfp6W6A6PrefillTiled16);
  if (mmq != KernelVariant::Mxfp6W6A6PrefillTiled16) {
    return mmq;
  }
  const char *const force_row8 = std::getenv("SLLM_MXFP6_PREFILL_FORCE_ROW8");
  if (force_row8 != nullptr && std::strcmp(force_row8, "1") == 0) {
    return KernelVariant::Mxfp6W6A6PrefillRow8;
  }
  const char *const force_tiled16 =
      std::getenv(kMxfp6W6A6PrefillTiled16Environment);
  if (force_tiled16 != nullptr && std::strcmp(force_tiled16, "1") == 0) {
    return KernelVariant::Mxfp6W6A6PrefillTiled16;
  }
  if (target_is(target, "gfx1030") &&
      phase74_gfx1030_mxfp6_half2_dot2_default_shape(m, k, n)) {
    return KernelVariant::Mxfp6W6A6PrefillGfx1030Half2_128x64K32DoublePack4;
  }
  if (target_is(target, "gfx1201") &&
      phase70_gfx1201_mxfp6_wmma_via_e4m3_shape(m, k, n)) {
    return KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4Swar;
  }
  return KernelVariant::Mxfp6W6A6PrefillTiled16;
}

inline KernelVariant select_mxfp6_variant(const uint64_t m) noexcept {
  return select_mxfp6_variant(m, 0U, 0U, "");
}

inline KernelVariant select_nvfp4_variant(const uint64_t m) noexcept {
  const char *const force_baseline = std::getenv("SLLM_NVFP4_FORCE_BASELINE");
  if (force_baseline != nullptr && std::strcmp(force_baseline, "1") == 0) {
    return KernelVariant::Nvfp4BaselinePackedDequant;
  }
  return m == 1U ? KernelVariant::Nvfp4DecodePackedDequant
                 : KernelVariant::Nvfp4PrefillRow8Tiled256;
}

constexpr bool
phase78_gfx1201_nvfp4_w4a4_wmma128x64_shape(const uint64_t m, const uint64_t k,
                                            const uint64_t n) noexcept {
  return m > 1U && k > 0U && (k % 16U) == 0U && n > 0U;
}

constexpr bool
phase78_gfx1201_nvfp4_w4a4_wmma128x32_shape(const uint64_t m, const uint64_t k,
                                            const uint64_t n) noexcept {
  return m <= 512U && phase78_gfx1201_nvfp4_w4a4_wmma128x64_shape(m, k, n);
}

constexpr bool
phase78_gfx1201_nvfp4_w4a4_f16_staging_shape(const uint64_t m, const uint64_t k,
                                             const uint64_t n) noexcept {
  return m >= 128U && (m % 128U) == 0U && k != 0U && k <= 17408U &&
         (k % 16U) == 0U && n != 0U && n <= 17408U && (n % 16U) == 0U;
}

// ID83 is limited to the two Qwen3.8-27B NVFP4 MLP projection orientations
// measured by the standalone candidate.  Keep the opt-in exact-target and
// shape boundaries here so prepare and execute share one frozen contract.
constexpr bool
phase78_gfx1201_nvfp4_w4a4_fp8_staging_shape(const uint64_t m, const uint64_t k,
                                             const uint64_t n) noexcept {
  return m >= 128U && (m % 128U) == 0U &&
         ((k == UINT64_C(5120) && n == UINT64_C(17408)) ||
          (k == UINT64_C(17408) && n == UINT64_C(5120)));
}

// ID62's separate gfx1030 index32 body is selected only when every logical
// extent and byte-plane product fits in uint32_t. The 128-element headroom
// keeps tile-rounding additions and the final 64x64 grid arithmetic from
// wrapping; base pointers retain native 64-bit pointer arithmetic.
constexpr uint64_t kNvfp4W4A4Dp4aIndex32MaxExtent =
    static_cast<uint64_t>(UINT32_MAX) - UINT64_C(128);

constexpr bool nvfp4_w4a4_index32_product_fits(const uint64_t left,
                                               const uint64_t right) noexcept {
  return left == 0U || right <= static_cast<uint64_t>(UINT32_MAX) / left;
}

constexpr bool
phase78_nvfp4_w4a4_dp4a_index32_shape(const uint64_t m, const uint64_t k,
                                      const uint64_t n) noexcept {
  if (m <= 32U || m > kNvfp4W4A4Dp4aIndex32MaxExtent || n == 0U ||
      n > kNvfp4W4A4Dp4aIndex32MaxExtent || k < 16U ||
      k > kNvfp4W4A4Dp4aIndex32MaxExtent || (k % 16U) != 0U) {
    return false;
  }
  const uint64_t packed_k = k / UINT64_C(2);
  const uint64_t scale_k = k / UINT64_C(16);
  return nvfp4_w4a4_index32_product_fits(m, n) &&
         nvfp4_w4a4_index32_product_fits(m, packed_k) &&
         nvfp4_w4a4_index32_product_fits(n, packed_k) &&
         nvfp4_w4a4_index32_product_fits(m, scale_k) &&
         nvfp4_w4a4_index32_product_fits(n, scale_k);
}

// The raw next-stage software pipeline is beneficial only for the measured
// Qwen3.8 wide projection on gfx1030. The target check remains in the launcher
// and dispatch-symbol mapper so neighboring and transposed shapes stay on the
// ordinary Index32 provider.
constexpr bool phase78_nvfp4_w4a4_dp4a_index32_pipeline_shape(
    const uint64_t m, const uint64_t k, const uint64_t n) noexcept {
  return m == UINT64_C(1024) && k == UINT64_C(5120) && n == UINT64_C(17408);
}

static_assert(!phase78_nvfp4_w4a4_dp4a_index32_shape(32U, 16U, 64U));
static_assert(phase78_nvfp4_w4a4_dp4a_index32_shape(33U, 16U, 64U));
static_assert(!phase78_nvfp4_w4a4_dp4a_index32_shape(33U, 15U, 64U));
static_assert(!phase78_nvfp4_w4a4_dp4a_index32_shape(
    33U, 16U, static_cast<uint64_t>(UINT32_MAX)));
static_assert(phase78_nvfp4_w4a4_dp4a_index32_shape(
    static_cast<uint64_t>(UINT32_MAX) / 8U, 16U, 1U));
static_assert(!phase78_nvfp4_w4a4_dp4a_index32_shape(
    static_cast<uint64_t>(UINT32_MAX) / 8U + 1U, 16U, 1U));
static_assert(phase78_nvfp4_w4a4_dp4a_index32_pipeline_shape(1024U, 5120U,
                                                             17408U));
static_assert(!phase78_nvfp4_w4a4_dp4a_index32_pipeline_shape(1023U, 5120U,
                                                              17408U));
static_assert(!phase78_nvfp4_w4a4_dp4a_index32_pipeline_shape(1025U, 5120U,
                                                              17408U));
static_assert(!phase78_nvfp4_w4a4_dp4a_index32_pipeline_shape(1024U, 5136U,
                                                              17408U));
static_assert(!phase78_nvfp4_w4a4_dp4a_index32_pipeline_shape(1024U, 5120U,
                                                              17344U));
static_assert(!phase78_nvfp4_w4a4_dp4a_index32_pipeline_shape(1024U, 17408U,
                                                              5120U));

static_assert(!phase78_gfx1201_nvfp4_w4a4_wmma128x64_shape(1U, 16U, 1U));
static_assert(!phase78_gfx1201_nvfp4_w4a4_wmma128x64_shape(2U, 15U, 1U));
static_assert(phase78_gfx1201_nvfp4_w4a4_wmma128x64_shape(2U, 16U, 1U));
static_assert(phase78_gfx1201_nvfp4_w4a4_wmma128x64_shape(129U, 48U, 65U));
static_assert(!phase78_gfx1201_nvfp4_w4a4_wmma128x64_shape(2U, 16U, 0U));
static_assert(!phase78_gfx1201_nvfp4_w4a4_wmma128x32_shape(1U, 16U, 1U));
static_assert(phase78_gfx1201_nvfp4_w4a4_wmma128x32_shape(2U, 16U, 1U));
static_assert(phase78_gfx1201_nvfp4_w4a4_wmma128x32_shape(127U, 48U, 31U));
static_assert(phase78_gfx1201_nvfp4_w4a4_wmma128x32_shape(512U, 48U, 33U));
static_assert(!phase78_gfx1201_nvfp4_w4a4_wmma128x32_shape(513U, 48U, 33U));
static_assert(!phase78_gfx1201_nvfp4_w4a4_f16_staging_shape(1U, 5120U, 17408U));
static_assert(!phase78_gfx1201_nvfp4_w4a4_f16_staging_shape(127U, 5120U,
                                                            17408U));
static_assert(phase78_gfx1201_nvfp4_w4a4_f16_staging_shape(128U, 5120U,
                                                           17408U));
static_assert(phase78_gfx1201_nvfp4_w4a4_f16_staging_shape(512U, 17408U,
                                                           5120U));
static_assert(!phase78_gfx1201_nvfp4_w4a4_f16_staging_shape(128U, 17424U,
                                                            5120U));
static_assert(!phase78_gfx1201_nvfp4_w4a4_f16_staging_shape(128U, 5120U,
                                                            248320U));
static_assert(phase78_gfx1201_nvfp4_w4a4_fp8_staging_shape(128U, 5120U,
                                                           17408U));
static_assert(phase78_gfx1201_nvfp4_w4a4_fp8_staging_shape(1024U, 17408U,
                                                           5120U));
static_assert(!phase78_gfx1201_nvfp4_w4a4_fp8_staging_shape(127U, 5120U,
                                                            17408U));
static_assert(!phase78_gfx1201_nvfp4_w4a4_fp8_staging_shape(128U, 5120U,
                                                            5120U));

constexpr bool
phase78_nvfp4_w4a4_decode_columns128_shape(const uint64_t m, const uint64_t k,
                                           const uint64_t n) noexcept {
  return m == 1U && k > 0U && (k % 16U) == 0U &&
         k <= kNvfp4W4A4DecodeColumns128MaxK && n > 0U;
}

static_assert(!phase78_nvfp4_w4a4_decode_columns128_shape(2U, 16U, 128U));
static_assert(!phase78_nvfp4_w4a4_decode_columns128_shape(1U, 15U, 128U));
static_assert(phase78_nvfp4_w4a4_decode_columns128_shape(1U, 16U, 1U));
static_assert(phase78_nvfp4_w4a4_decode_columns128_shape(1U, 17408U, 129U));
static_assert(!phase78_nvfp4_w4a4_decode_columns128_shape(1U, 17424U, 128U));

constexpr bool
phase78_nvfp4_w4a4_decode_wave4col32_shape(const uint64_t m, const uint64_t k,
                                           const uint64_t n) noexcept {
  return m == 1U && k > 0U && (k % 16U) == 0U &&
         k <= kNvfp4W4A4DecodeColumns128MaxK && n > 0U;
}

static_assert(!phase78_nvfp4_w4a4_decode_wave4col32_shape(2U, 16U, 32U));
static_assert(!phase78_nvfp4_w4a4_decode_wave4col32_shape(1U, 15U, 32U));
static_assert(phase78_nvfp4_w4a4_decode_wave4col32_shape(1U, 16U, 31U));
static_assert(phase78_nvfp4_w4a4_decode_wave4col32_shape(1U, 32U, 32U));
static_assert(phase78_nvfp4_w4a4_decode_wave4col32_shape(1U, 48U, 33U));
static_assert(phase78_nvfp4_w4a4_decode_wave4col32_shape(1U, 17408U, 32U));
static_assert(!phase78_nvfp4_w4a4_decode_wave4col32_shape(1U, 17424U, 32U));

constexpr bool phase78_nvfp4_w4a4_decode_activation_shared_shape(
    const uint64_t m, const uint64_t k, const uint64_t n) noexcept {
  return m == 1U && ((k == UINT64_C(5120) && n == UINT64_C(17408)) ||
                     (k == UINT64_C(17408) && n == UINT64_C(5120)));
}

constexpr bool
phase78_nvfp4_w4a4_decode_scale_lut_gfx1201_activation_shared_shape(
    const uint64_t m, const uint64_t k, const uint64_t n) noexcept {
  return m == 1U && ((k == UINT64_C(5120) && n == UINT64_C(17408)) ||
                     (k == UINT64_C(17408) && n == UINT64_C(5120)));
}

constexpr uint64_t
nvfp4_w4a4_decode_activation_shared_lds_bytes(const uint64_t k) noexcept {
  return (k / UINT64_C(16)) * UINT64_C(5) * sizeof(uint32_t);
}

static_assert(phase78_nvfp4_w4a4_decode_activation_shared_shape(1U, 5120U,
                                                                17408U));
static_assert(phase78_nvfp4_w4a4_decode_activation_shared_shape(1U, 17408U,
                                                                5120U));
static_assert(!phase78_nvfp4_w4a4_decode_activation_shared_shape(2U, 5120U,
                                                                 17408U));
static_assert(!phase78_nvfp4_w4a4_decode_activation_shared_shape(1U, 5120U,
                                                                 5120U));
static_assert(!phase78_nvfp4_w4a4_decode_activation_shared_shape(1U, 17408U,
                                                                 17408U));
static_assert(
    phase78_nvfp4_w4a4_decode_scale_lut_gfx1201_activation_shared_shape(
        1U, 5120U, 17408U));
static_assert(
    phase78_nvfp4_w4a4_decode_scale_lut_gfx1201_activation_shared_shape(1U,
                                                                        17408U,
                                                                        5120U));
static_assert(
    !phase78_nvfp4_w4a4_decode_scale_lut_gfx1201_activation_shared_shape(
        1U, 5120U, 5120U));
static_assert(nvfp4_w4a4_decode_activation_shared_lds_bytes(5120U) == 6400U);
static_assert(nvfp4_w4a4_decode_activation_shared_lds_bytes(17408U) == 21760U);

inline KernelVariant
select_nvfp4_w4a4_variant(const uint64_t m, const uint64_t k, const uint64_t n,
                          const char *const target) noexcept {
  const char *const force_baseline =
      std::getenv("SLLM_NVFP4_W4A4_FORCE_BASELINE");
  if (force_baseline != nullptr && std::strcmp(force_baseline, "1") == 0) {
    return KernelVariant::Nvfp4W4A4Packed;
  }
  if (m == 1U) {
    const char *const force_scale_lut =
        std::getenv(kNvfp4W4A4DecodeScaleLutEnvironment);
    if (force_scale_lut != nullptr && std::strcmp(force_scale_lut, "1") == 0) {
      if (target_is(target, "gfx1030") &&
          phase78_nvfp4_w4a4_decode_activation_shared_shape(m, k, n)) {
        return KernelVariant::Nvfp4W4A4DecodeScaleLut;
      }
      if (target_is(target, "gfx1201") &&
          phase78_nvfp4_w4a4_decode_wave4col32_shape(m, k, n)) {
        return KernelVariant::Nvfp4W4A4DecodeScaleLut;
      }
    }
    const char *const force_activation_shared =
        std::getenv(kNvfp4W4A4DecodeActivationSharedEnvironment);
    if (force_activation_shared != nullptr &&
        std::strcmp(force_activation_shared, "1") == 0 &&
        target_is(target, "gfx1030") &&
        phase78_nvfp4_w4a4_decode_activation_shared_shape(m, k, n)) {
      return KernelVariant::Nvfp4W4A4DecodeActivationShared;
    }
    const char *const force_wave4 =
        std::getenv(kNvfp4W4A4DecodeWave4Column32Environment);
    const bool supported_target =
        target_is(target, "gfx1030") || target_is(target, "gfx1201");
    if (force_wave4 != nullptr && std::strcmp(force_wave4, "1") == 0 &&
        supported_target &&
        phase78_nvfp4_w4a4_decode_wave4col32_shape(m, k, n)) {
      return KernelVariant::Nvfp4W4A4DecodeWave4Column32;
    }
    const char *const force_columns =
        std::getenv(kNvfp4W4A4DecodeColumns128Environment);
    if (force_columns != nullptr && std::strcmp(force_columns, "1") == 0 &&
        supported_target &&
        phase78_nvfp4_w4a4_decode_columns128_shape(m, k, n)) {
      return KernelVariant::Nvfp4W4A4DecodeColumns128;
    }
    return KernelVariant::Nvfp4W4A4Decode;
  }
  // ID83's FP8 staging recipe remains an isolated research primitive.  Keep
  // its identity and workspace helpers available to the probe, but do not
  // expose it through the public selector while its numerical classification
  // is unresolved.
  const char *const force_row8 =
      std::getenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_ROW8");
  if (force_row8 != nullptr && std::strcmp(force_row8, "1") == 0) {
    return KernelVariant::Nvfp4W4A4PrefillRow8Tiled256;
  }
  const char *const force_col8 =
      std::getenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_COL8");
  if (force_col8 != nullptr && std::strcmp(force_col8, "1") == 0) {
    return KernelVariant::Nvfp4W4A4PrefillRow8Col8Tiled256;
  }
  const char *const force_gfx1201_wmma_f16scale =
      std::getenv(kNvfp4W4A4PrefillGfx1201WmmaF16ScaleEnvironment);
  if (force_gfx1201_wmma_f16scale != nullptr &&
      std::strcmp(force_gfx1201_wmma_f16scale, "1") == 0 &&
      target_is(target, "gfx1201") &&
      phase78_gfx1201_nvfp4_w4a4_wmma128x64_shape(m, k, n)) {
    return KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaF16Scale128x64;
  }
  const char *const force_gfx1201_wmma128x32 =
      std::getenv(kNvfp4W4A4PrefillGfx1201Wmma128x32Environment);
  if (force_gfx1201_wmma128x32 != nullptr &&
      std::strcmp(force_gfx1201_wmma128x32, "1") == 0 &&
      target_is(target, "gfx1201") &&
      phase78_gfx1201_nvfp4_w4a4_wmma128x64_shape(m, k, n)) {
    return phase78_gfx1201_nvfp4_w4a4_wmma128x32_shape(m, k, n)
               ? KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x32
               : KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x64;
  }
  const char *const force_dp4a =
      std::getenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_DP4A");
  const char *const force_dp4a_k128 =
      std::getenv(kNvfp4W4A4PrefillDp4a64x64K128Environment);
  const char *const force_gfx1201_wmma =
      std::getenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA");
  if (force_gfx1201_wmma != nullptr &&
      std::strcmp(force_gfx1201_wmma, "1") == 0 &&
      target_is(target, "gfx1201") &&
      phase78_gfx1201_nvfp4_w4a4_wmma128x64_shape(m, k, n)) {
    return KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x64;
  }
  const char *const force_gfx1201_f16_staging =
      std::getenv(kNvfp4W4A4PrefillGfx1201F16StagingEnvironment);
  if (force_gfx1201_f16_staging != nullptr &&
      std::strcmp(force_gfx1201_f16_staging, "1") == 0 &&
      target_is(target, "gfx1201") &&
      phase78_gfx1201_nvfp4_w4a4_f16_staging_shape(m, k, n)) {
    return KernelVariant::Nvfp4W4A4PrefillGfx1201F16Staging;
  }
  // Keep the ID72 opt-in useful for a full prefill where the final chunk is
  // not an M=128 multiple.  Such a tail retains the existing ID64 WMMA
  // reduction contract when its K/N shape is otherwise eligible.  The
  // fallback is scoped to the same exact target and literal opt-in so it
  // cannot change the default selector or enable ID64 independently.
  if (force_gfx1201_f16_staging != nullptr &&
      std::strcmp(force_gfx1201_f16_staging, "1") == 0 &&
      target_is(target, "gfx1201") &&
      phase78_gfx1201_nvfp4_w4a4_wmma128x64_shape(m, k, n)) {
    return KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x64;
  }
  if (force_dp4a_k128 != nullptr && std::strcmp(force_dp4a_k128, "1") == 0 &&
      target_is(target, "gfx1030") && m > 1U && k != 0U && (k % 16U) == 0U &&
      n != 0U) {
    return KernelVariant::Nvfp4W4A4PrefillDp4a64x64K128;
  }
  return force_dp4a != nullptr && std::strcmp(force_dp4a, "1") == 0 &&
                 k != 0U && (k % 16U) == 0U
             ? KernelVariant::Nvfp4W4A4PrefillDp4a64x64
             : KernelVariant::Nvfp4W4A4PrefillRow8Tiled256;
}

inline KernelVariant select_nvfp4_w4a4_variant(const uint64_t m) noexcept {
  return select_nvfp4_w4a4_variant(m, 0U, 0U, "");
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

constexpr bool fp8_outer_decode_gfx1030_half2_shape(const uint64_t m,
                                                    const uint64_t k,
                                                    const uint64_t n) noexcept {
  return m == 1U && n != 0U && k >= kFp8OuterDecodeGfx1030Half2MinK &&
         k <= kFp8OuterDecodeGfx1030Half2MaxK && (k % 64U) == 0U;
}

constexpr bool
fp8_outer_decode_gfx1030_lds_lut_tuple_shape(const uint64_t m, const uint64_t k,
                                             const uint64_t n) noexcept {
  return m == 1U && ((k == UINT64_C(5120) &&
                      (n == UINT64_C(17408) || n == UINT64_C(10240) ||
                       n == UINT64_C(6144))) ||
                     (k == UINT64_C(6144) && n == UINT64_C(5120)));
}

static_assert(fp8_outer_decode_gfx1030_lds_lut_tuple_shape(1U, 5120U, 17408U));
static_assert(fp8_outer_decode_gfx1030_lds_lut_tuple_shape(1U, 6144U, 5120U));
static_assert(fp8_outer_decode_gfx1030_lds_lut_tuple_shape(1U, 5120U, 10240U));
static_assert(fp8_outer_decode_gfx1030_lds_lut_tuple_shape(1U, 5120U, 6144U));
static_assert(!fp8_outer_decode_gfx1030_lds_lut_tuple_shape(2U, 5120U, 6144U));
static_assert(!fp8_outer_decode_gfx1030_lds_lut_tuple_shape(1U, 5120U, 6143U));
static_assert(!fp8_outer_decode_gfx1030_lds_lut_tuple_shape(1U, 5120U, 6145U));
static_assert(!fp8_outer_decode_gfx1030_lds_lut_tuple_shape(1U, 5120U, 5120U));
static_assert(!fp8_outer_decode_gfx1030_lds_lut_tuple_shape(2U, 5120U, 17408U));

constexpr bool fp8_outer_decode_gfx1030_activation_shared_wave4_shape(
    const uint64_t m, const uint64_t k, const uint64_t n) noexcept {
  return m == 1U && k == UINT64_C(5120) && n == UINT64_C(17408);
}

constexpr bool fp8_outer_decode_gfx1030_activation_shared_wave8_shape(
    const uint64_t m, const uint64_t k, const uint64_t n) noexcept {
  // K5120/N6144 regressed in the full-model A/B and remains on ID68.
  return m == 1U && ((k == UINT64_C(5120) &&
                      (n == UINT64_C(12288) || n == UINT64_C(10240))) ||
                     (k == UINT64_C(6144) && n == UINT64_C(5120)));
}

constexpr bool fp8_outer_decode_gfx1030_activation_shared_rollback_shape(
    const uint64_t m, const uint64_t k, const uint64_t n) noexcept {
  return m == 1U &&
         ((k == UINT64_C(17408) && n == UINT64_C(5120)) ||
          (k == UINT64_C(5120) && (n == UINT64_C(1024) || n == UINT64_C(6144) ||
                                   n == UINT64_C(248320))));
}

constexpr uint64_t fp8_outer_decode_gfx1030_activation_shared_lds_bytes(
    const uint64_t k) noexcept {
  return k <= UINT64_MAX / UINT64_C(2) ? k * UINT64_C(2) : 0U;
}

static_assert(fp8_outer_decode_gfx1030_activation_shared_wave4_shape(1U, 5120U,
                                                                     17408U));
static_assert(fp8_outer_decode_gfx1030_activation_shared_wave8_shape(1U, 6144U,
                                                                     5120U));
static_assert(fp8_outer_decode_gfx1030_activation_shared_rollback_shape(
    1U, 5120U, 248320U));
static_assert(fp8_outer_decode_gfx1030_activation_shared_lds_bytes(5120U) ==
              10240U);
static_assert(fp8_outer_decode_gfx1030_activation_shared_lds_bytes(17408U) ==
              34816U);

constexpr bool
fp8_outer_prefill_gfx1030_f16_staging_shape(const uint64_t m, const uint64_t k,
                                            const uint64_t n) noexcept {
  return m >= kFp8OuterPrefillGfx1030F16StagingMinM &&
         (m % kFp8OuterPrefillGfx1030F16StagingMinM) == 0U && k != 0U &&
         k <= kFp8OuterPrefillGfx1030F16StagingMaxK && (k % 16U) == 0U &&
         n != 0U && n <= kFp8OuterPrefillGfx1030F16StagingMaxN &&
         (n % 16U) == 0U;
}

constexpr bool fp8_outer_prefill_gfx1030_f16_tile_staging_shape(
    const uint64_t m, const uint64_t k, const uint64_t n) noexcept {
  // ID86 keeps the measured ID71 tile and accepts row/column tails. K and N
  // retain the existing FP8 outer staging boundaries; K is a multiple of four
  // so every transient FP16 row has a dword-aligned packed load.
  return m >= kFp8OuterPrefillGfx1030F16StagingMinM && k != 0U &&
         k <= kFp8OuterPrefillGfx1030F16StagingMaxK && (k % 16U) == 0U &&
         n != 0U && n <= kFp8OuterPrefillGfx1030F16StagingMaxN &&
         (n % 16U) == 0U;
}

static_assert(!fp8_outer_prefill_gfx1030_f16_tile_staging_shape(127U, 16U,
                                                                16U));
static_assert(fp8_outer_prefill_gfx1030_f16_tile_staging_shape(128U, 16U, 16U));
static_assert(fp8_outer_prefill_gfx1030_f16_tile_staging_shape(219U, 6144U,
                                                               5120U));
static_assert(!fp8_outer_prefill_gfx1030_f16_tile_staging_shape(128U, 12U,
                                                                16U));
static_assert(!fp8_outer_prefill_gfx1030_f16_tile_staging_shape(128U, 16U,
                                                                17U));

// ID71's short-M specialization is exact only for the measured K/N contract.
// Keep the predicates central so launcher, grid calculation, and public
// dispatch metadata cannot report different tile geometry. All other ID71
// shapes remain 64x64.
constexpr bool fp8_outer_prefill_gfx1030_half2_short_m32_n64_shape(
    const uint64_t m, const uint64_t k, const uint64_t n) noexcept {
  return m > 1U && m <= 32U && k == UINT64_C(5120) &&
         (n == UINT64_C(10240) || n == UINT64_C(12288) || n == UINT64_C(17408));
}

constexpr bool fp8_outer_prefill_gfx1030_half2_short_m32_n32_shape(
    const uint64_t m, const uint64_t k, const uint64_t n) noexcept {
  return m > 1U && m <= 32U &&
         ((k == UINT64_C(17408) && n == UINT64_C(5120)) ||
          (k == UINT64_C(5120) &&
           (n == UINT64_C(1024) || n == UINT64_C(6144))) ||
          (k == UINT64_C(6144) && n == UINT64_C(5120)));
}

constexpr bool fp8_outer_prefill_gfx1030_half2_short_m32_shape(
    const uint64_t m, const uint64_t k, const uint64_t n) noexcept {
  return fp8_outer_prefill_gfx1030_half2_short_m32_n64_shape(m, k, n) ||
         fp8_outer_prefill_gfx1030_half2_short_m32_n32_shape(m, k, n);
}

static_assert(fp8_outer_prefill_gfx1030_half2_short_m32_n64_shape(2U, 5120U,
                                                                  10240U));
static_assert(fp8_outer_prefill_gfx1030_half2_short_m32_n64_shape(32U, 5120U,
                                                                  17408U));
static_assert(!fp8_outer_prefill_gfx1030_half2_short_m32_n64_shape(1U, 5120U,
                                                                   17408U));
static_assert(!fp8_outer_prefill_gfx1030_half2_short_m32_n64_shape(33U, 5120U,
                                                                   17408U));
static_assert(!fp8_outer_prefill_gfx1030_half2_short_m32_n64_shape(32U, 5120U,
                                                                   17409U));
static_assert(fp8_outer_prefill_gfx1030_half2_short_m32_n32_shape(2U, 17408U,
                                                                  5120U));
static_assert(fp8_outer_prefill_gfx1030_half2_short_m32_n32_shape(32U, 6144U,
                                                                  5120U));
static_assert(fp8_outer_prefill_gfx1030_half2_short_m32_shape(17U, 6144U,
                                                              5120U));
static_assert(fp8_outer_prefill_gfx1030_half2_short_m32_shape(32U, 6144U,
                                                              5120U));
static_assert(!fp8_outer_prefill_gfx1030_half2_short_m32_shape(33U, 6144U,
                                                               5120U));
static_assert(!fp8_outer_prefill_gfx1030_half2_short_m32_shape(32U, 5120U,
                                                               2048U));

constexpr bool
fp8_outer_prefill_gfx1030_lds_lut_shape(const uint64_t m, const uint64_t k,
                                        const uint64_t n) noexcept {
  // ID85 preserves ID71's broad prefill selection while keeping the tile
  // mapping well-defined for direct launcher callers.
  return m > 1U && k > 0U && n > 0U;
}

static_assert(!fp8_outer_prefill_gfx1030_lds_lut_shape(1U, 32U, 64U));
static_assert(fp8_outer_prefill_gfx1030_lds_lut_shape(2U, 32U, 64U));
static_assert(fp8_outer_prefill_gfx1030_lds_lut_shape(128U, 5120U, 17408U));
static_assert(!fp8_outer_prefill_gfx1030_lds_lut_shape(128U, 0U, 17408U));

constexpr bool
f16_staging_workspace_layout(const uint64_t m, const uint64_t k,
                             const uint64_t n,
                             F16StagingWorkspaceLayout *const layout) noexcept {
  if (layout == nullptr || m == 0U || k == 0U || n == 0U) {
    return false;
  }
  *layout = {};
  constexpr uint64_t max = UINT64_MAX;
  constexpr uint64_t alignment = kMatmulF16StagingAlignment;
  const auto checked_product = [](const uint64_t left, const uint64_t right,
                                  uint64_t *const output) constexpr noexcept {
    if (output == nullptr || (left != 0U && right > UINT64_MAX / left)) {
      return false;
    }
    *output = left * right;
    return true;
  };
  const auto checked_align = [](const uint64_t value,
                                uint64_t *const output) constexpr noexcept {
    constexpr uint64_t mask = kMatmulF16StagingAlignment - 1U;
    if (output == nullptr || value > UINT64_MAX - mask) {
      return false;
    }
    *output = (value + mask) & ~mask;
    return true;
  };
  static_assert((alignment & (alignment - 1U)) == 0U);

  uint64_t activation_elements = 0U;
  uint64_t weight_elements = 0U;
  uint64_t output_elements = 0U;
  if (!checked_product(m, k, &activation_elements) ||
      !checked_product(n, k, &weight_elements) ||
      !checked_product(m, n, &output_elements) ||
      !checked_product(activation_elements, UINT64_C(2),
                       &layout->activation_bytes) ||
      !checked_product(weight_elements, UINT64_C(2), &layout->weight_bytes) ||
      !checked_product(output_elements, UINT64_C(4), &layout->output_bytes)) {
    return false;
  }
  layout->activation_offset = 0U;
  uint64_t activation_end = layout->activation_bytes;
  if (!checked_align(activation_end, &layout->weight_offset) ||
      layout->weight_offset > max - layout->weight_bytes) {
    return false;
  }
  const uint64_t weight_end = layout->weight_offset + layout->weight_bytes;
  if (!checked_align(weight_end, &layout->output_offset) ||
      layout->output_offset > max - layout->output_bytes) {
    return false;
  }
  const uint64_t output_end = layout->output_offset + layout->output_bytes;
  return checked_align(output_end, &layout->total_bytes) &&
         layout->total_bytes != 0U;
}

constexpr bool fp8_outer_prefill_gfx1030_f16_staging_workspace(
    const uint64_t m, const uint64_t k, const uint64_t n,
    F16StagingWorkspaceLayout *const layout) noexcept {
  return fp8_outer_prefill_gfx1030_f16_staging_shape(m, k, n) &&
         f16_staging_workspace_layout(m, k, n, layout);
}

constexpr bool fp8_outer_prefill_gfx1030_f16_tile_staging_workspace(
    const uint64_t m, const uint64_t k, const uint64_t n,
    F16StagingWorkspaceLayout *const layout) noexcept {
  return fp8_outer_prefill_gfx1030_f16_tile_staging_shape(m, k, n) &&
         f16_staging_workspace_layout(m, k, n, layout);
}

struct Nvfp4Fp8StagingWorkspaceLayout final {
  uint64_t activation_offset;
  uint64_t activation_bytes;
  uint64_t weight_offset;
  uint64_t weight_bytes;
  uint64_t scale_product_offset;
  uint64_t total_bytes;
};

// ID83 uses the existing format-neutral context arena.  Keep the FP8 layout
// separate from the FP16 layout so a future change cannot accidentally feed a
// BF16/FP16 pointer to the native FP8 descriptor.
constexpr bool nvfp4_w4a4_fp8_staging_shape(const uint64_t m, const uint64_t k,
                                            const uint64_t n) noexcept {
  return m >= 128U && (m % 128U) == 0U &&
         ((k == UINT64_C(5120) && n == UINT64_C(17408)) ||
          (k == UINT64_C(17408) && n == UINT64_C(5120)));
}

constexpr bool nvfp4_w4a4_fp8_staging_workspace(
    const uint64_t m, const uint64_t k, const uint64_t n,
    Nvfp4Fp8StagingWorkspaceLayout *const layout) noexcept {
  if (layout == nullptr || !nvfp4_w4a4_fp8_staging_shape(m, k, n)) {
    return false;
  }
  constexpr uint64_t max = UINT64_MAX;
  const auto product = [](const uint64_t left, const uint64_t right,
                          uint64_t *const result) constexpr noexcept {
    if (result == nullptr || (left != 0U && right > UINT64_MAX / left)) {
      return false;
    }
    *result = left * right;
    return true;
  };
  const auto align = [](const uint64_t value,
                        uint64_t *const result) constexpr noexcept {
    constexpr uint64_t mask = kMatmulF16StagingAlignment - 1U;
    if (result == nullptr || value > UINT64_MAX - mask)
      return false;
    *result = (value + mask) & ~mask;
    return true;
  };
  *layout = {};
  if (!product(m, k, &layout->activation_bytes) ||
      !product(n, k, &layout->weight_bytes) ||
      !align(layout->activation_bytes, &layout->weight_offset) ||
      layout->weight_offset > max - layout->weight_bytes ||
      !align(layout->weight_offset + layout->weight_bytes,
             &layout->scale_product_offset) ||
      layout->scale_product_offset > max - sizeof(float) ||
      !align(layout->scale_product_offset + sizeof(float),
             &layout->total_bytes)) {
    return false;
  }
  layout->activation_offset = 0U;
  return layout->total_bytes != 0U;
}

static_assert(nvfp4_w4a4_fp8_staging_shape(128U, 5120U, 17408U));
static_assert(nvfp4_w4a4_fp8_staging_shape(1024U, 17408U, 5120U));
static_assert(!nvfp4_w4a4_fp8_staging_shape(127U, 5120U, 17408U));
static_assert(!nvfp4_w4a4_fp8_staging_shape(128U, 5120U, 5120U));
static_assert([] {
  Nvfp4Fp8StagingWorkspaceLayout layout{};
  return nvfp4_w4a4_fp8_staging_workspace(128U, 5120U, 17408U, &layout) &&
         layout.activation_bytes == UINT64_C(655360) &&
         layout.weight_bytes == UINT64_C(89128960) &&
         layout.total_bytes > layout.scale_product_offset;
}());

// Phase 78 split4: only the two measured M=17 Qwen projection shapes are
// eligible.  The target-specific runtime gate further restricts gfx1201 to
// the down shape and gfx1030 to both shapes; no selector or public ID changes.
constexpr bool phase78_nvfp4_w4a4_split4_shape(const uint64_t m,
                                               const uint64_t k,
                                               const uint64_t n) noexcept {
  return m == UINT64_C(17) && ((k == UINT64_C(5120) && n == UINT64_C(17408)) ||
                               (k == UINT64_C(17408) && n == UINT64_C(5120)));
}

constexpr bool
phase78_gfx1030_nvfp4_w4a4_split4_shape(const uint64_t m, const uint64_t k,
                                        const uint64_t n) noexcept {
  return phase78_nvfp4_w4a4_split4_shape(m, k, n);
}

constexpr bool
phase78_gfx1201_nvfp4_w4a4_split4_shape(const uint64_t m, const uint64_t k,
                                        const uint64_t n) noexcept {
  return m == UINT64_C(17) && k == UINT64_C(17408) && n == UINT64_C(5120);
}

constexpr bool phase78_nvfp4_w4a4_split4_workspace(
    const uint64_t m, const uint64_t k, const uint64_t n,
    uint64_t *const partial_offset, uint64_t *const total_bytes) noexcept {
  if (partial_offset == nullptr || total_bytes == nullptr ||
      !phase78_nvfp4_w4a4_split4_shape(m, k, n)) {
    return false;
  }
  constexpr uint64_t max = UINT64_MAX;
  const uint64_t packed_activation_bytes_per_row = k / UINT64_C(2);
  const uint64_t activation_scale_bytes_per_row = k / UINT64_C(16);
  if (m != 0U && packed_activation_bytes_per_row > max / m) {
    return false;
  }
  if (m != 0U && activation_scale_bytes_per_row > max / m) {
    return false;
  }
  const uint64_t packed_activation_bytes = m * packed_activation_bytes_per_row;
  const uint64_t activation_scale_bytes = m * activation_scale_bytes_per_row;
  if (packed_activation_bytes > max - activation_scale_bytes) {
    return false;
  }
  const uint64_t base_bytes = packed_activation_bytes + activation_scale_bytes;
  if (m != 0U && n > max / m) {
    return false;
  }
  const uint64_t output_elements = m * n;
  if (output_elements > max / (UINT64_C(4) * sizeof(float))) {
    return false;
  }
  const uint64_t partial_bytes = output_elements * UINT64_C(4) * sizeof(float);
  if (base_bytes > max - partial_bytes) {
    return false;
  }
  *partial_offset = base_bytes;
  *total_bytes = base_bytes + partial_bytes;
  return true;
}

static_assert(phase78_gfx1201_nvfp4_w4a4_split4_shape(17U, 17408U, 5120U));
static_assert(phase78_gfx1030_nvfp4_w4a4_split4_shape(17U, 5120U, 17408U));
static_assert(!phase78_nvfp4_w4a4_split4_shape(16U, 17408U, 5120U));
static_assert(!phase78_nvfp4_w4a4_split4_shape(18U, 17408U, 5120U));
static_assert(!phase78_nvfp4_w4a4_split4_shape(17U, 17408U, 5119U));
static_assert(!phase78_nvfp4_w4a4_split4_shape(17U, 17408U, 5121U));
static_assert([] {
  uint64_t partial_offset = 0U;
  uint64_t total_bytes = 0U;
  return phase78_nvfp4_w4a4_split4_workspace(17U, 17408U, 5120U,
                                             &partial_offset, &total_bytes) &&
         partial_offset == UINT64_C(166464) && total_bytes == UINT64_C(1559104);
}());
static_assert([] {
  uint64_t partial_offset = 0U;
  uint64_t total_bytes = 0U;
  return phase78_nvfp4_w4a4_split4_workspace(17U, 5120U, 17408U,
                                             &partial_offset, &total_bytes) &&
         partial_offset == UINT64_C(48960) && total_bytes == UINT64_C(4783936);
}());

// A Qwen3.8 execution segment prepares several projection shapes before the
// previous same-queue submissions complete.  Reserve the largest exact
// wide/down layout on the first staging plan so later prepares never need to
// grow the shared arena while it is in flight.  The lease itself remains
// format-neutral and is shared by FP8 and NVFP4 staging pipelines.
constexpr bool
qwen38_f16_staging_workspace_reservation(const uint64_t m,
                                         uint64_t *const bytes) noexcept {
  if (bytes == nullptr) {
    return false;
  }
  *bytes = 0U;
  F16StagingWorkspaceLayout wide{};
  F16StagingWorkspaceLayout down{};
  if (!f16_staging_workspace_layout(m, UINT64_C(5120), UINT64_C(17408),
                                    &wide) ||
      !f16_staging_workspace_layout(m, UINT64_C(17408), UINT64_C(5120),
                                    &down)) {
    return false;
  }
  *bytes =
      wide.total_bytes > down.total_bytes ? wide.total_bytes : down.total_bytes;
  return *bytes != 0U;
}

static_assert([] {
  uint64_t bytes = 0U;
  return qwen38_f16_staging_workspace_reservation(512U, &bytes) &&
         bytes >= UINT64_C(219152384);
}());

// The legacy FP8 outer-vector path uses hipBLASLt on matrix-capable targets
// and a scalar byte-decode kernel on gfx1030. Decode keeps the scalar provider
// by default and exposes the exact-E4M3FN half2 wave/column kernel only as an
// opt-in. For prefill on gfx1030, select the tiled software provider so one
// workgroup reuses activation/weight tiles across output rows and columns.
// Each lane of the decode candidate loads one activation pair and shares it
// across four adjacent weight rows before the wave reduction.
inline KernelVariant
select_fp8_outer_variant(const uint64_t m, const uint64_t k, const uint64_t n,
                         const char *const target,
                         const bool fnuz = false) noexcept {
  if (target_is(target, "gfx1201") || target_is(target, "gfx942")) {
    return KernelVariant::Fp8Native;
  }
  if (!target_is(target, "gfx1030")) {
    return KernelVariant::Fp8Emulation;
  }
  if (m == 1U) {
    const char *const force_baseline =
        std::getenv(kFp8OuterDecodeBaselineEnvironment);
    if (fnuz ||
        (force_baseline != nullptr && std::strcmp(force_baseline, "1") == 0)) {
      return KernelVariant::Fp8Emulation;
    }
    const char *const force_lds_lut =
        std::getenv(kFp8OuterDecodeGfx1030LdsLutEnvironment);
    if (force_lds_lut != nullptr && std::strcmp(force_lds_lut, "1") == 0 &&
        fp8_outer_decode_gfx1030_half2_shape(m, k, n)) {
      return KernelVariant::Fp8OuterDecodeGfx1030LdsLutWave4Col32;
    }
    const char *const force_activation_shared =
        std::getenv(kFp8OuterDecodeGfx1030ActivationSharedEnvironment);
    if (force_activation_shared != nullptr &&
        std::strcmp(force_activation_shared, "1") == 0) {
      if (fp8_outer_decode_gfx1030_activation_shared_wave4_shape(m, k, n)) {
        return KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave4Col32;
      }
      if (fp8_outer_decode_gfx1030_activation_shared_wave8_shape(m, k, n)) {
        return KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave8Col64;
      }
      if (fp8_outer_decode_gfx1030_activation_shared_rollback_shape(m, k, n)) {
        return KernelVariant::Fp8OuterDecodeGfx1030Dword8Wave4Col32;
      }
    }
    const char *const force_half2 =
        std::getenv(kFp8OuterDecodeGfx1030Half2Environment);
    const char *const force_dword8 =
        std::getenv(kFp8OuterDecodeGfx1030Dword8Environment);
    if (force_dword8 != nullptr && std::strcmp(force_dword8, "1") == 0 &&
        fp8_outer_decode_gfx1030_half2_shape(m, k, n)) {
      return KernelVariant::Fp8OuterDecodeGfx1030Dword8Wave4Col32;
    }
    return force_half2 != nullptr && std::strcmp(force_half2, "1") == 0 &&
                   fp8_outer_decode_gfx1030_half2_shape(m, k, n)
               ? KernelVariant::Fp8OuterDecodeGfx1030Half2Wave4Col32
               : KernelVariant::Fp8Emulation;
  }
  const char *const force_baseline =
      std::getenv("SLLM_FP8_OUTER_PREFILL_FORCE_BASELINE");
  if (force_baseline != nullptr && std::strcmp(force_baseline, "1") == 0) {
    return KernelVariant::Fp8Emulation;
  }
  const char *const force_f16_tile_staging =
      std::getenv(kFp8OuterPrefillGfx1030F16TileStagingEnvironment);
  if (!fnuz && force_f16_tile_staging != nullptr &&
      std::strcmp(force_f16_tile_staging, "1") == 0 &&
      fp8_outer_prefill_gfx1030_f16_tile_staging_shape(m, k, n)) {
    return KernelVariant::Fp8OuterPrefillGfx1030F16TileStaging;
  }
  const char *const force_f16_staging =
      std::getenv(kFp8OuterPrefillGfx1030F16StagingEnvironment);
  if (!fnuz && force_f16_staging != nullptr &&
      std::strcmp(force_f16_staging, "1") == 0 &&
      fp8_outer_prefill_gfx1030_f16_staging_shape(m, k, n)) {
    return KernelVariant::Fp8OuterPrefillGfx1030F16Staging;
  }
  const char *const force_lds_lut =
      std::getenv(kFp8OuterPrefillGfx1030LdsLutEnvironment);
  if (!fnuz && force_lds_lut != nullptr &&
      std::strcmp(force_lds_lut, "1") == 0 &&
      fp8_outer_prefill_gfx1030_lds_lut_shape(m, k, n)) {
    return KernelVariant::Fp8OuterPrefillGfx1030LdsLut;
  }
  const char *const force_half2_64x64 =
      std::getenv(kFp8OuterPrefillGfx1030Half2_64x64Environment);
  if (!fnuz && force_half2_64x64 != nullptr &&
      std::strcmp(force_half2_64x64, "1") == 0) {
    return KernelVariant::Fp8OuterPrefillGfx1030Half2_64x64;
  }
  const char *const force_half2 =
      std::getenv("SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_HALF2");
  if (force_half2 != nullptr && std::strcmp(force_half2, "1") == 0 &&
      (k % 2U) == 0U) {
    return KernelVariant::Fp8OuterPrefillGfx1030Half2_128x64;
  }
  // ID71 is the measured gfx1030 OCP E4M3FN prefill default.  Keep every
  // explicit selector above it so the scalar baseline, research-only FP16
  // staging path, and ID63 128x64 rollback remain deterministic controls.
  return !fnuz ? KernelVariant::Fp8OuterPrefillGfx1030Half2_64x64
               : KernelVariant::Fp8OuterPrefillTiled16;
}

constexpr const char *logical_kernel_id(const KernelVariant variant) noexcept {
  if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_32x32K32) {
    return kMxfp8W8A8PrefillGfx1030Half2_32x32K32LogicalKernelId;
  }
  if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_64x64K32) {
    return kMxfp8W8A8PrefillGfx1030Half2_64x64K32LogicalKernelId;
  }
  if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x32K32) {
    return kMxfp8W8A8PrefillGfx1030Half2_128x32K32LogicalKernelId;
  }
  if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K32) {
    return kMxfp8W8A8PrefillGfx1030Half2_128x64K32LogicalKernelId;
  }
  if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K64) {
    return kMxfp8W8A8PrefillGfx1030Half2_128x64K64LogicalKernelId;
  }
  if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K128) {
    return kMxfp8W8A8PrefillGfx1030Half2_128x64K128LogicalKernelId;
  }
  if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K32Double) {
    return kMxfp8W8A8PrefillGfx1030Half2_128x64K32DoubleLogicalKernelId;
  }
  if (variant ==
      KernelVariant::Mxfp6W6A6PrefillGfx1030Half2_128x64K32DoubleScalar) {
    return kMxfp6W6A6PrefillGfx1030Half2_128x64K32DoubleScalarLogicalKernelId;
  }
  if (variant ==
      KernelVariant::Mxfp6W6A6PrefillGfx1030Half2_128x64K32DoublePack4) {
    return kMxfp6W6A6PrefillGfx1030Half2_128x64K32DoublePack4LogicalKernelId;
  }
  if (variant ==
      KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave4Col32) {
    return kFp8OuterDecodeGfx1030ActivationSharedWave4LogicalKernelId;
  }
  if (variant ==
      KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave8Col64) {
    return kFp8OuterDecodeGfx1030ActivationSharedWave8LogicalKernelId;
  }
  if (variant == KernelVariant::Fp8OuterDecodeGfx1030LdsLutWave4Col32) {
    return kFp8OuterDecodeGfx1030LdsLutLogicalKernelId;
  }
  if (variant == KernelVariant::Fp8OuterPrefillGfx1030LdsLut) {
    return kFp8OuterPrefillGfx1030LdsLutLogicalKernelId;
  }
  if (variant == KernelVariant::Fp8OuterPrefillGfx1030F16TileStaging) {
    return kFp8OuterPrefillGfx1030F16TileStagingLogicalKernelId;
  }
  if (variant == KernelVariant::Nvfp4W4A4DecodeScaleLut) {
    return kNvfp4W4A4DecodeScaleLutLogicalKernelId;
  }
  return variant == KernelVariant::Fp8Native      ? kFp8NativeLogicalKernelId
         : variant == KernelVariant::Fp8Emulation ? kFp8EmulationLogicalKernelId
         : variant == KernelVariant::Fp8OuterPrefillTiled16
             ? kFp8OuterPrefillTiled16LogicalKernelId
         : variant == KernelVariant::Fp8OuterPrefillGfx1030F16Staging
             ? kFp8OuterPrefillGfx1030F16StagingLogicalKernelId
         : variant == KernelVariant::Fp8OuterPrefillGfx1030Half2_64x64
             ? kFp8OuterPrefillGfx1030Half2_64x64LogicalKernelId
         : variant == KernelVariant::Fp8OuterPrefillGfx1030LdsLut
             ? kFp8OuterPrefillGfx1030LdsLutLogicalKernelId
         : variant == KernelVariant::Fp8OuterPrefillGfx1030F16TileStaging
             ? kFp8OuterPrefillGfx1030F16TileStagingLogicalKernelId
         : variant == KernelVariant::Fp8OuterPrefillGfx1030Half2_128x64
             ? kFp8OuterPrefillGfx1030Half2LogicalKernelId
         : variant == KernelVariant::Fp8OuterDecodeGfx1030Half2Wave4Col32
             ? kFp8OuterDecodeGfx1030Half2Wave4Col32LogicalKernelId
         : variant == KernelVariant::Fp8OuterDecodeGfx1030Dword8Wave4Col32
             ? kFp8OuterDecodeGfx1030Dword8LogicalKernelId
         : variant == KernelVariant::Nvfp4DecodePackedDequant
             ? kNvfp4DecodeLogicalKernelId
         : variant == KernelVariant::Nvfp4PrefillRow8Tiled256
             ? kNvfp4PrefillLogicalKernelId
         : variant == KernelVariant::Nvfp4BaselinePackedDequant
             ? kNvfp4BaselineLogicalKernelId
         : variant == KernelVariant::Nvfp4W4A4Decode
             ? kNvfp4W4A4DecodeLogicalKernelId
         : variant == KernelVariant::Nvfp4W4A4PrefillRow8Tiled256
             ? kNvfp4W4A4PrefillRow8LogicalKernelId
         : variant == KernelVariant::Nvfp4W4A4PrefillRow8Col8Tiled256
             ? kNvfp4W4A4PrefillRow8Col8LogicalKernelId
         : variant == KernelVariant::Nvfp4W4A4PrefillDp4a64x64
             ? kNvfp4W4A4PrefillDp4a64x64LogicalKernelId
         : variant == KernelVariant::Nvfp4W4A4PrefillDp4a64x64K128
             ? kNvfp4W4A4PrefillDp4a64x64K128LogicalKernelId
         : variant == KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x64
             ? kNvfp4W4A4PrefillGfx1201Wmma128x64LogicalKernelId
         : variant == KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x32
             ? kNvfp4W4A4PrefillGfx1201Wmma128x32LogicalKernelId
         : variant == KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaF16Scale128x64
             ? kNvfp4W4A4PrefillGfx1201WmmaF16Scale128x64LogicalKernelId
         : variant == KernelVariant::Nvfp4W4A4PrefillGfx1201F16Staging
             ? kNvfp4W4A4PrefillGfx1201F16StagingLogicalKernelId
         : variant == KernelVariant::Nvfp4W4A4PrefillGfx1201Fp8Staging
             ? kNvfp4W4A4PrefillGfx1201Fp8StagingLogicalKernelId
         : variant == KernelVariant::Nvfp4W4A4DecodeColumns128
             ? kNvfp4W4A4DecodeColumns128LogicalKernelId
         : variant == KernelVariant::Nvfp4W4A4DecodeWave4Column32
             ? kNvfp4W4A4DecodeWave4Column32LogicalKernelId
         : variant == KernelVariant::Nvfp4W4A4DecodeActivationShared
             ? kNvfp4W4A4DecodeActivationSharedLogicalKernelId
         : variant == KernelVariant::Nvfp4W4A4DecodeScaleLut
             ? kNvfp4W4A4DecodeScaleLutLogicalKernelId
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
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Regscale
             ? kMxfp8W8A8PrefillMmqGfx1030RegscaleLogicalKernelId
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Vector32
             ? kMxfp8W8A8PrefillMmqGfx1030Vector32LogicalKernelId
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030RegscaleVector32
             ? kMxfp8W8A8PrefillMmqGfx1030RegscaleVector32LogicalKernelId
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
         : variant == KernelVariant::Mxfp6W6A6PrefillMmqGfx1030ViaE4M3
             ? kMxfp6W6A6PrefillMmqGfx1030ViaE4M3LogicalKernelId
         : variant == KernelVariant::Mxfp6W6A6PrefillGfx1030Half2Dot2
             ? kMxfp6W6A6PrefillGfx1030Half2Dot2LogicalKernelId
         : variant == KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201ViaE4M3N64
             ? kMxfp6W6A6PrefillWmmaGfx1201ViaE4M3N64LogicalKernelId
         : variant == KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4N64
             ? kMxfp6W6A6PrefillWmmaGfx1201Pack4N64LogicalKernelId
         : variant == KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4Swar
             ? kMxfp6W6A6PrefillWmmaGfx1201Pack4SwarLogicalKernelId
         : variant == KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4N128
             ? kMxfp6W6A6PrefillWmmaGfx1201Pack4N128LogicalKernelId
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
  if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_32x32K32) {
    return kMxfp8W8A8PrefillGfx1030Half2_32x32K32DeviceSymbol;
  }
  if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_64x64K32) {
    return kMxfp8W8A8PrefillGfx1030Half2_64x64K32DeviceSymbol;
  }
  if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x32K32) {
    return kMxfp8W8A8PrefillGfx1030Half2_128x32K32DeviceSymbol;
  }
  if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K32) {
    return kMxfp8W8A8PrefillGfx1030Half2_128x64K32DeviceSymbol;
  }
  if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K64) {
    return kMxfp8W8A8PrefillGfx1030Half2_128x64K64DeviceSymbol;
  }
  if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K128) {
    return kMxfp8W8A8PrefillGfx1030Half2_128x64K128DeviceSymbol;
  }
  if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K32Double) {
    return kMxfp8W8A8PrefillGfx1030Half2_128x64K32DoubleDeviceSymbol;
  }
  if (variant ==
      KernelVariant::Mxfp6W6A6PrefillGfx1030Half2_128x64K32DoubleScalar) {
    return kMxfp6W6A6PrefillGfx1030Half2_128x64K32DoubleScalarDeviceSymbol;
  }
  if (variant ==
      KernelVariant::Mxfp6W6A6PrefillGfx1030Half2_128x64K32DoublePack4) {
    return kMxfp6W6A6PrefillGfx1030Half2_128x64K32DoublePack4DeviceSymbol;
  }
  if (variant ==
      KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave4Col32) {
    return kFp8OuterDecodeGfx1030ActivationSharedWave4DeviceSymbol;
  }
  if (variant ==
      KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave8Col64) {
    return kFp8OuterDecodeGfx1030ActivationSharedWave8DeviceSymbol;
  }
  if (variant == KernelVariant::Fp8OuterDecodeGfx1030LdsLutWave4Col32) {
    return kFp8OuterDecodeGfx1030LdsLutDeviceSymbol;
  }
  if (variant == KernelVariant::Fp8OuterPrefillGfx1030LdsLut) {
    return kFp8OuterPrefillGfx1030LdsLutDeviceSymbol;
  }
  if (variant == KernelVariant::Fp8OuterPrefillGfx1030F16TileStaging) {
    return kFp8OuterPrefillGfx1030F16TileStagingDeviceSymbol;
  }
  if (variant == KernelVariant::Nvfp4W4A4DecodeScaleLut) {
    return kNvfp4W4A4DecodeScaleLutDeviceSymbol;
  }
  return variant == KernelVariant::Fp8Native      ? kFp8NativeDeviceSymbol
         : variant == KernelVariant::Fp8Emulation ? kFp8EmulationDeviceSymbol
         : variant == KernelVariant::Fp8OuterPrefillTiled16
             ? kFp8OuterPrefillTiled16DeviceSymbol
         : variant == KernelVariant::Fp8OuterPrefillGfx1030F16Staging
             ? kFp8OuterPrefillGfx1030F16StagingDeviceSymbol
         : variant == KernelVariant::Fp8OuterPrefillGfx1030Half2_64x64
             ? kFp8OuterPrefillGfx1030Half2_64x64DeviceSymbol
         : variant == KernelVariant::Fp8OuterPrefillGfx1030LdsLut
             ? kFp8OuterPrefillGfx1030LdsLutDeviceSymbol
         : variant == KernelVariant::Fp8OuterPrefillGfx1030F16TileStaging
             ? kFp8OuterPrefillGfx1030F16TileStagingDeviceSymbol
         : variant == KernelVariant::Fp8OuterPrefillGfx1030Half2_128x64
             ? kFp8OuterPrefillGfx1030Half2DeviceSymbol
         : variant == KernelVariant::Fp8OuterDecodeGfx1030Half2Wave4Col32
             ? kFp8OuterDecodeGfx1030Half2Wave4Col32DeviceSymbol
         : variant == KernelVariant::Fp8OuterDecodeGfx1030Dword8Wave4Col32
             ? kFp8OuterDecodeGfx1030Dword8DeviceSymbol
         : variant == KernelVariant::Nvfp4DecodePackedDequant
             ? kNvfp4DecodeDeviceSymbol
         : variant == KernelVariant::Nvfp4PrefillRow8Tiled256
             ? kNvfp4PrefillDeviceSymbol
         : variant == KernelVariant::Nvfp4BaselinePackedDequant
             ? kNvfp4BaselineDeviceSymbol
         : variant == KernelVariant::Nvfp4W4A4Decode
             ? kNvfp4W4A4DecodeDeviceSymbol
         : variant == KernelVariant::Nvfp4W4A4PrefillRow8Tiled256
             ? kNvfp4W4A4PrefillRow8DeviceSymbol
         : variant == KernelVariant::Nvfp4W4A4PrefillRow8Col8Tiled256
             ? kNvfp4W4A4PrefillRow8Col8DeviceSymbol
         : variant == KernelVariant::Nvfp4W4A4PrefillDp4a64x64
             ? kNvfp4W4A4PrefillDp4a64x64DeviceSymbol
         : variant == KernelVariant::Nvfp4W4A4PrefillDp4a64x64K128
             ? kNvfp4W4A4PrefillDp4a64x64K128DeviceSymbol
         : variant == KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x64
             ? kNvfp4W4A4PrefillGfx1201Wmma128x64DeviceSymbol
         : variant == KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x32
             ? kNvfp4W4A4PrefillGfx1201Wmma128x32DeviceSymbol
         : variant == KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaF16Scale128x64
             ? kNvfp4W4A4PrefillGfx1201WmmaF16Scale128x64DeviceSymbol
         : variant == KernelVariant::Nvfp4W4A4PrefillGfx1201F16Staging
             ? kNvfp4W4A4PrefillGfx1201F16StagingDeviceSymbol
         : variant == KernelVariant::Nvfp4W4A4PrefillGfx1201Fp8Staging
             ? kNvfp4W4A4PrefillGfx1201Fp8StagingDeviceSymbol
         : variant == KernelVariant::Nvfp4W4A4DecodeColumns128
             ? kNvfp4W4A4DecodeColumns128DeviceSymbol
         : variant == KernelVariant::Nvfp4W4A4DecodeWave4Column32
             ? kNvfp4W4A4DecodeWave4Column32DeviceSymbol
         : variant == KernelVariant::Nvfp4W4A4DecodeActivationShared
             ? kNvfp4W4A4DecodeActivationSharedDeviceSymbol
         : variant == KernelVariant::Nvfp4W4A4DecodeScaleLut
             ? kNvfp4W4A4DecodeScaleLutDeviceSymbol
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
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Regscale
             ? kMxfp8W8A8PrefillMmqGfx1030RegscaleDeviceSymbol
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Vector32
             ? kMxfp8W8A8PrefillMmqGfx1030Vector32DeviceSymbol
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030RegscaleVector32
             ? kMxfp8W8A8PrefillMmqGfx1030RegscaleVector32DeviceSymbol
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
         : variant == KernelVariant::Mxfp6W6A6PrefillMmqGfx1030ViaE4M3
             ? kMxfp6W6A6PrefillMmqGfx1030ViaE4M3DeviceSymbol
         : variant == KernelVariant::Mxfp6W6A6PrefillGfx1030Half2Dot2
             ? kMxfp6W6A6PrefillGfx1030Half2Dot2DeviceSymbol
         : variant == KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201ViaE4M3N64
             ? kMxfp6W6A6PrefillWmmaGfx1201ViaE4M3N64DeviceSymbol
         : variant == KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4N64
             ? kMxfp6W6A6PrefillWmmaGfx1201Pack4N64DeviceSymbol
         : variant == KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4Swar
             ? kMxfp6W6A6PrefillWmmaGfx1201Pack4SwarDeviceSymbol
         : variant == KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4N128
             ? kMxfp6W6A6PrefillWmmaGfx1201Pack4N128DeviceSymbol
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

// ID84 has one candidate ID but two target-specific code-object symbols.  The
// public dispatch registry uses this overload so reports describe the symbol
// that the exact target launcher actually enqueues.
inline const char *
logical_kernel_id_for_target(const KernelVariant variant,
                             const char *const target) noexcept {
  (void)target;
  return logical_kernel_id(variant);
}

inline const char *device_symbol_for_target(const KernelVariant variant,
                                            const char *const target) noexcept {
  if (variant == KernelVariant::Nvfp4W4A4DecodeScaleLut) {
    (void)target;
    return kNvfp4W4A4DecodeScaleLutDeviceSymbol;
  }
  return device_symbol(variant);
}

inline const char *device_symbol_for_target(const KernelVariant variant,
                                            const char *const target,
                                            const uint64_t m, const uint64_t k,
                                            const uint64_t n) noexcept {
  if (variant == KernelVariant::Nvfp4W4A4PrefillDp4a64x64 &&
      target_is(target, "gfx1030") &&
      phase78_nvfp4_w4a4_dp4a_index32_pipeline_shape(m, k, n)) {
    return kNvfp4W4A4PrefillDp4a64x64Index32PipelineDeviceSymbol;
  }
  if (variant == KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x64 &&
      target_is(target, "gfx1201") &&
      phase78_gfx1201_nvfp4_wmma_ordinary_shape(m, k, n)) {
    return kNvfp4W4A4Gfx1201WmmaOrdinaryDeviceSymbol;
  }
  if (variant == KernelVariant::Nvfp4W4A4DecodeScaleLut &&
      target_is(target, "gfx1201") &&
      phase78_nvfp4_w4a4_decode_scale_lut_gfx1201_activation_shared_shape(m, k,
                                                                          n)) {
    return kNvfp4W4A4DecodeScaleLutGfx1201ActivationSharedDeviceSymbol;
  }
  if (variant == KernelVariant::Fp8OuterDecodeGfx1030LdsLutWave4Col32 &&
      target_is(target, "gfx1030") &&
      fp8_outer_decode_gfx1030_lds_lut_tuple_shape(m, k, n)) {
    if (k == UINT64_C(5120) && n == UINT64_C(17408)) {
      return kFp8OuterDecodeGfx1030LdsLutK5120N17408DeviceSymbol;
    }
    if (k == UINT64_C(6144) && n == UINT64_C(5120)) {
      return kFp8OuterDecodeGfx1030LdsLutK6144N5120DeviceSymbol;
    }
    if (k == UINT64_C(5120) && n == UINT64_C(6144)) {
      return kFp8OuterDecodeGfx1030LdsLutK5120N6144DeviceSymbol;
    }
    return kFp8OuterDecodeGfx1030LdsLutK5120N10240DeviceSymbol;
  }
  return device_symbol_for_target(variant, target);
}

constexpr uint32_t grid_size_x(const KernelVariant variant, const uint64_t m,
                               const uint64_t n,
                               const uint64_t k = 0U) noexcept {
  if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_32x32K32) {
    return static_cast<uint32_t>(((m + 31U) / 32U) * ((n + 31U) / 32U));
  }
  if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_64x64K32) {
    return static_cast<uint32_t>(((m + 63U) / 64U) * ((n + 63U) / 64U));
  }
  if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x32K32) {
    return static_cast<uint32_t>(((m + 127U) / 128U) * ((n + 31U) / 32U));
  }
  if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K32 ||
      variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K64 ||
      variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K128 ||
      variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K32Double ||
      variant ==
          KernelVariant::Mxfp6W6A6PrefillGfx1030Half2_128x64K32DoubleScalar ||
      variant ==
          KernelVariant::Mxfp6W6A6PrefillGfx1030Half2_128x64K32DoublePack4) {
    return static_cast<uint32_t>(((m + 127U) / 128U) * ((n + 63U) / 64U));
  }
  return variant == KernelVariant::Fp8OuterPrefillTiled16
             ? static_cast<uint32_t>(((m + 15U) / 16U) * ((n + 15U) / 16U))
         : variant == KernelVariant::Fp8OuterPrefillGfx1030F16Staging
             ? static_cast<uint32_t>((m * n + kWorkgroupSize - 1U) /
                                     kWorkgroupSize)
         : variant == KernelVariant::Fp8OuterPrefillGfx1030F16TileStaging
             ? static_cast<uint32_t>(((m + 63U) / 64U) * ((n + 63U) / 64U))
         : variant == KernelVariant::Fp8OuterPrefillGfx1030Half2_64x64
             ? static_cast<uint32_t>(
                   fp8_outer_prefill_gfx1030_half2_short_m32_n64_shape(m, k, n)
                       ? ((m + 31U) / 32U) * ((n + 63U) / 64U)
                   : fp8_outer_prefill_gfx1030_half2_short_m32_n32_shape(m, k,
                                                                         n)
                       ? ((m + 31U) / 32U) * ((n + 31U) / 32U)
                       : ((m + 63U) / 64U) * ((n + 63U) / 64U))
         : variant == KernelVariant::Fp8OuterPrefillGfx1030LdsLut
             ? static_cast<uint32_t>(((m + 63U) / 64U) * ((n + 63U) / 64U))
         : variant == KernelVariant::Fp8OuterPrefillGfx1030Half2_128x64
             ? static_cast<uint32_t>(((m + 127U) / 128U) * ((n + 63U) / 64U))
         : variant == KernelVariant::Fp8OuterDecodeGfx1030Half2Wave4Col32
             ? static_cast<uint32_t>(
                   (n + kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup - 1U) /
                   kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup)
         : variant == KernelVariant::Fp8OuterDecodeGfx1030Dword8Wave4Col32
             ? static_cast<uint32_t>(
                   (n + kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup - 1U) /
                   kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup)
         : variant == KernelVariant::Fp8OuterDecodeGfx1030LdsLutWave4Col32
             ? static_cast<uint32_t>(
                   (n + kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup - 1U) /
                   kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup)
         : variant ==
                 KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave4Col32
             ? static_cast<uint32_t>(
                   (n + kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup - 1U) /
                   kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup)
         : variant ==
                 KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave8Col64
             ? static_cast<uint32_t>((n + 63U) / 64U)
         : variant == KernelVariant::Fp8Native ? static_cast<uint32_t>(n)
         : variant == KernelVariant::Nvfp4DecodePackedDequant
             ? static_cast<uint32_t>(n)
         : variant == KernelVariant::Nvfp4PrefillRow8Tiled256
             ? static_cast<uint32_t>(((m + 7U) / 8U) * n)
         : variant == KernelVariant::Nvfp4BaselinePackedDequant
             ? static_cast<uint32_t>(m * n)
         : variant == KernelVariant::Nvfp4W4A4Decode ? static_cast<uint32_t>(n)
         : variant == KernelVariant::Nvfp4W4A4PrefillRow8Tiled256
             ? static_cast<uint32_t>(((m + 7U) / 8U) * n)
         : variant == KernelVariant::Nvfp4W4A4PrefillRow8Col8Tiled256
             ? static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 7U) / 8U))
         : variant == KernelVariant::Nvfp4W4A4PrefillDp4a64x64
             ? static_cast<uint32_t>(((m + 63U) / 64U) * ((n + 63U) / 64U))
         : variant == KernelVariant::Nvfp4W4A4PrefillDp4a64x64K128
             ? static_cast<uint32_t>(((m + 63U) / 64U) * ((n + 63U) / 64U))
         : variant == KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x64
             ? static_cast<uint32_t>((n + 63U) / 64U)
         : variant == KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x32
             ? static_cast<uint32_t>((n + 31U) / 32U)
         : variant == KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaF16Scale128x64
             ? static_cast<uint32_t>(
                   ((m + kNvfp4W4A4PrefillGfx1201WmmaF16ScaleRowsPerWorkgroup -
                     1U) /
                    kNvfp4W4A4PrefillGfx1201WmmaF16ScaleRowsPerWorkgroup) *
                   ((n +
                     kNvfp4W4A4PrefillGfx1201WmmaF16ScaleColumnsPerWorkgroup -
                     1U) /
                    kNvfp4W4A4PrefillGfx1201WmmaF16ScaleColumnsPerWorkgroup))
         : variant == KernelVariant::Nvfp4W4A4PrefillGfx1201F16Staging
             ? static_cast<uint32_t>((m * n + kWorkgroupSize - 1U) /
                                     kWorkgroupSize)
         : variant == KernelVariant::Nvfp4W4A4PrefillGfx1201Fp8Staging
             ? static_cast<uint32_t>((m * n + kWorkgroupSize - 1U) /
                                     kWorkgroupSize)
         : variant == KernelVariant::Nvfp4W4A4DecodeColumns128
             ? static_cast<uint32_t>((n + 127U) / 128U)
         : variant == KernelVariant::Nvfp4W4A4DecodeWave4Column32
             ? static_cast<uint32_t>((n + 31U) / 32U)
         : variant == KernelVariant::Nvfp4W4A4DecodeActivationShared
             ? static_cast<uint32_t>((n + 31U) / 32U)
         : variant == KernelVariant::Nvfp4W4A4DecodeScaleLut
             ? static_cast<uint32_t>((n + 31U) / 32U)
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
         : variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Regscale ||
                 variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Vector32 ||
                 variant ==
                     KernelVariant::Mxfp8W8A8PrefillMmqGfx1030RegscaleVector32
             ? static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 7U) / 8U))
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
         : variant == KernelVariant::Mxfp6W6A6PrefillMmqGfx1030ViaE4M3
             ? static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 7U) / 8U))
         : variant == KernelVariant::Mxfp6W6A6PrefillGfx1030Half2Dot2
             ? static_cast<uint32_t>(((m + 31U) / 32U) * ((n + 31U) / 32U))
         : variant == KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201ViaE4M3N64 ||
                 variant ==
                     KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4N64 ||
                 variant == KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4Swar
             ? static_cast<uint32_t>(
                   (n + kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup - 1U) /
                   kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup)
         : variant == KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4N128
             ? static_cast<uint32_t>(
                   (n + kMxfp8W8A8PrefillWmmaN128ColumnsPerWorkgroup - 1U) /
                   kMxfp8W8A8PrefillWmmaN128ColumnsPerWorkgroup)
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
                 variant == KernelVariant::Mxfp8W8A8PrefillWmmaN128DirectBoth ||
                 variant ==
                     KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201ViaE4M3N64 ||
                 variant ==
                     KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4N64 ||
                 variant ==
                     KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4N128 ||
                 variant == KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x64 ||
                 variant == KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x32
             ? kMxfp8W8A8PrefillWmmaWorkgroupSize
         : variant == KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaF16Scale128x64
             ? kNvfp4W4A4PrefillGfx1201WmmaF16ScaleWorkgroupSize
         : variant == KernelVariant::Nvfp4W4A4PrefillGfx1201Fp8Staging
             ? kNvfp4W4A4PrefillGfx1201Fp8StagingWorkgroupSize
         : variant == KernelVariant::Nvfp4W4A4DecodeScaleLut
             ? kNvfp4W4A4DecodeScaleLutWorkgroupSize
         : variant == KernelVariant::Nvfp4W4A4DecodeColumns128
             ? kNvfp4W4A4DecodeColumns128WorkgroupSize
         : variant == KernelVariant::Nvfp4W4A4DecodeWave4Column32
             ? kNvfp4W4A4DecodeWave4Column32WorkgroupSize
         : variant == KernelVariant::Nvfp4W4A4DecodeActivationShared
             ? kNvfp4W4A4DecodeActivationSharedWorkgroupSize
         : variant == KernelVariant::Fp8OuterDecodeGfx1030Half2Wave4Col32
             ? kFp8OuterDecodeGfx1030Half2WorkgroupSize
         : variant == KernelVariant::Fp8OuterDecodeGfx1030Dword8Wave4Col32
             ? kFp8OuterDecodeGfx1030Half2WorkgroupSize
         : variant == KernelVariant::Fp8OuterDecodeGfx1030LdsLutWave4Col32
             ? kFp8OuterDecodeGfx1030LdsLutWorkgroupSize
         : variant == KernelVariant::Fp8OuterPrefillGfx1030LdsLut
             ? kFp8OuterPrefillGfx1030LdsLutWorkgroupSize
         : variant == KernelVariant::Fp8OuterPrefillGfx1030F16TileStaging
             ? kWorkgroupSize
         : variant ==
                 KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave4Col32
             ? kFp8OuterDecodeGfx1030Half2WorkgroupSize
         : variant ==
                 KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave8Col64
             ? kFp8OuterDecodeGfx1030Half2WorkgroupSize
             : kWorkgroupSize;
}

static_assert(workgroup_size_x(KernelVariant::Mxfp8W8A8PrefillWmmaN16) == 256U);
static_assert(workgroup_size_x(KernelVariant::Mxfp8W8A8PrefillWmmaN64) == 256U);
static_assert(workgroup_size_x(KernelVariant::Mxfp8W8A8PrefillWmma4Wave) ==
              128U);
static_assert(workgroup_size_x(
                  KernelVariant::Mxfp8W8A8PrefillWmmaN128DirectBoth) == 256U);
static_assert(
    workgroup_size_x(KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x64) == 256U);
static_assert(
    workgroup_size_x(KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x32) == 256U);
static_assert(workgroup_size_x(
                  KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaF16Scale128x64) ==
              256U);
static_assert(
    workgroup_size_x(KernelVariant::Nvfp4W4A4PrefillGfx1201Fp8Staging) == 256U);
static_assert(workgroup_size_x(KernelVariant::Nvfp4W4A4DecodeScaleLut) ==
              kNvfp4W4A4DecodeScaleLutWorkgroupSize);
static_assert(workgroup_size_x(KernelVariant::Nvfp4W4A4DecodeColumns128) ==
              128U);
static_assert(workgroup_size_x(KernelVariant::Nvfp4W4A4DecodeWave4Column32) ==
              kNvfp4W4A4DecodeWave4Column32WorkgroupSize);
static_assert(
    workgroup_size_x(KernelVariant::Nvfp4W4A4DecodeActivationShared) ==
    kNvfp4W4A4DecodeActivationSharedWorkgroupSize);
static_assert(
    workgroup_size_x(KernelVariant::Fp8OuterDecodeGfx1030Dword8Wave4Col32) ==
    kFp8OuterDecodeGfx1030Half2WorkgroupSize);
static_assert(
    workgroup_size_x(KernelVariant::Fp8OuterDecodeGfx1030LdsLutWave4Col32) ==
    kFp8OuterDecodeGfx1030LdsLutWorkgroupSize);
static_assert(workgroup_size_x(KernelVariant::Fp8OuterPrefillGfx1030LdsLut) ==
              kFp8OuterPrefillGfx1030LdsLutWorkgroupSize);
static_assert(
    workgroup_size_x(
        KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave4Col32) ==
    kFp8OuterDecodeGfx1030Half2WorkgroupSize);
static_assert(
    workgroup_size_x(
        KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave8Col64) ==
    kFp8OuterDecodeGfx1030Half2WorkgroupSize);
static_assert(grid_size_x(KernelVariant::Mxfp8W8A8PrefillWmmaN128DirectBoth,
                          128U, 128U) == 1U);
static_assert(grid_size_x(KernelVariant::Mxfp8W8A8PrefillWmmaN128DirectBoth,
                          128U, 256U) == 2U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4Decode, 1U, 7U) == 7U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4DecodeColumns128, 1U, 1U) ==
              1U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4DecodeColumns128, 1U, 128U) ==
              1U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4DecodeColumns128, 1U, 129U) ==
              2U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4DecodeWave4Column32, 1U,
                          31U) == 1U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4DecodeWave4Column32, 1U,
                          32U) == 1U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4DecodeWave4Column32, 1U,
                          33U) == 2U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4DecodeActivationShared, 1U,
                          5120U) == 160U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4DecodeActivationShared, 1U,
                          17408U) == 544U);
static_assert(
    grid_size_x(KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave4Col32,
                1U, 17408U) == 544U);
static_assert(
    grid_size_x(KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave8Col64,
                1U, 12288U) == 192U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4PrefillRow8Tiled256, 17U,
                          7U) == 21U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4PrefillRow8Col8Tiled256, 17U,
                          7U) == 3U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4PrefillDp4a64x64, 65U, 65U) ==
              4U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x64,
                          127U, 63U) == 1U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x64,
                          129U, 65U) == 2U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x32,
                          127U, 31U) == 1U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x32,
                          129U, 33U) == 2U);
static_assert(
    grid_size_x(KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaF16Scale128x64, 127U,
                63U) == 1U);
static_assert(
    grid_size_x(KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaF16Scale128x64, 129U,
                65U) == 4U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4PrefillGfx1201Fp8Staging,
                          128U, 17408U) == 8704U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4Packed, 2U, 7U) == 14U);
static_assert(grid_size_x(KernelVariant::Fp8OuterPrefillTiled16, 17U, 17U) ==
              4U);
static_assert(grid_size_x(KernelVariant::Fp8OuterPrefillGfx1030Half2_128x64,
                          129U, 65U) == 4U);
static_assert(grid_size_x(KernelVariant::Fp8OuterPrefillGfx1030Half2_64x64, 65U,
                          65U) == 4U);
static_assert(grid_size_x(KernelVariant::Fp8OuterPrefillGfx1030Half2_64x64, 17U,
                          5120U) == 80U);
static_assert(grid_size_x(KernelVariant::Fp8OuterPrefillGfx1030Half2_64x64, 17U,
                          5120U, 6144U) == 160U);
static_assert(grid_size_x(KernelVariant::Fp8OuterPrefillGfx1030Half2_64x64, 17U,
                          10240U, 5120U) == 160U);
static_assert(grid_size_x(KernelVariant::Fp8OuterPrefillGfx1030Half2_64x64, 17U,
                          12288U, 5120U) == 192U);
static_assert(grid_size_x(KernelVariant::Fp8OuterPrefillGfx1030Half2_64x64, 17U,
                          17408U, 5120U) == 272U);
static_assert(grid_size_x(KernelVariant::Fp8OuterPrefillGfx1030Half2_64x64, 33U,
                          5120U, 6144U) == 80U);
static_assert(grid_size_x(KernelVariant::Fp8OuterPrefillGfx1030LdsLut, 65U,
                          65U) == 4U);
static_assert(grid_size_x(KernelVariant::Fp8OuterPrefillGfx1030F16Staging, 128U,
                          16U) == 8U);
static_assert(grid_size_x(KernelVariant::Fp8OuterPrefillGfx1030F16TileStaging,
                          219U, 5120U) == 320U);
static_assert(grid_size_x(KernelVariant::Fp8OuterDecodeGfx1030Half2Wave4Col32,
                          1U, 32U) == 1U);
static_assert(grid_size_x(KernelVariant::Fp8OuterDecodeGfx1030Half2Wave4Col32,
                          1U, 33U) == 2U);
static_assert(grid_size_x(KernelVariant::Fp8OuterDecodeGfx1030Dword8Wave4Col32,
                          1U, 32U) == 1U);
static_assert(grid_size_x(KernelVariant::Fp8OuterDecodeGfx1030Dword8Wave4Col32,
                          1U, 33U) == 2U);
static_assert(grid_size_x(KernelVariant::Fp8OuterDecodeGfx1030LdsLutWave4Col32,
                          1U, 32U) == 1U);
static_assert(grid_size_x(KernelVariant::Fp8OuterDecodeGfx1030LdsLutWave4Col32,
                          1U, 33U) == 2U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4DecodeScaleLut, 1U, 32U) ==
              1U);
static_assert(grid_size_x(KernelVariant::Nvfp4W4A4DecodeScaleLut, 1U, 33U) ==
              2U);
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
static_assert(grid_size_x(KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Regscale, 8U,
                          7U) == 1U);
static_assert(grid_size_x(KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Vector32, 9U,
                          9U) == 4U);
static_assert(
    grid_size_x(KernelVariant::Mxfp8W8A8PrefillMmqGfx1030RegscaleVector32, 9U,
                17U) == 6U);

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

hipError_t launch_fp8_outer_prefill_tiled16(
    const uint8_t *activation, const float *activation_scales,
    const uint8_t *weight, const float *weight_scales, uint16_t *output,
    uint64_t m, uint64_t k, uint64_t n, hipStream_t stream) noexcept;

hipError_t launch_fp8_outer_prefill_gfx1030_half2_128x64(
    const uint8_t *activation, const float *activation_scales,
    const uint8_t *weight, const float *weight_scales, uint16_t *output,
    uint64_t m, uint64_t k, uint64_t n, hipStream_t stream) noexcept;

hipError_t launch_fp8_outer_prefill_gfx1030_half2_64x64(
    const uint8_t *activation, const float *activation_scales,
    const uint8_t *weight, const float *weight_scales, uint16_t *output,
    uint64_t m, uint64_t k, uint64_t n, hipStream_t stream) noexcept;

hipError_t launch_fp8_outer_prefill_gfx1030_lds_lut(
    const uint8_t *activation, const float *activation_scales,
    const uint8_t *weight, const float *weight_scales, uint16_t *output,
    uint64_t m, uint64_t k, uint64_t n, hipStream_t stream) noexcept;

// ID86 consumes the FP16 matrices produced by the shared ID70 ingress stage.
// The launcher itself does not allocate or retain staging storage.
hipError_t launch_fp8_outer_prefill_gfx1030_f16_tile_staging(
    const uint16_t *activation, const float *activation_scales,
    const uint16_t *weight, const float *weight_scales, uint16_t *output,
    uint64_t m, uint64_t k, uint64_t n, hipStream_t stream) noexcept;

hipError_t launch_fp8_e4m3fn_to_fp16_staging(const uint8_t *input,
                                             uint16_t *output,
                                             uint64_t element_count,
                                             hipStream_t stream) noexcept;

hipError_t launch_fp8_outer_f16scale_epilogue(const float *input,
                                              const float *activation_scales,
                                              const float *weight_scales,
                                              uint16_t *output, uint64_t m,
                                              uint64_t n,
                                              hipStream_t stream) noexcept;

hipError_t launch_nvfp4_block16_to_fp16_staging(const uint8_t *packed,
                                                const uint8_t *block_scales,
                                                uint16_t *output, uint64_t rows,
                                                uint64_t k,
                                                hipStream_t stream) noexcept;

hipError_t launch_nvfp4_block16_to_fp8_staging(const uint8_t *packed,
                                               const uint8_t *block_scales,
                                               uint8_t *output, uint64_t rows,
                                               uint64_t k,
                                               hipStream_t stream) noexcept;

hipError_t launch_nvfp4_tensor_scale_product(const float *weight_tensor_scale,
                                             const float *input_tensor_scale,
                                             float *output,
                                             hipStream_t stream) noexcept;

hipError_t launch_nvfp4_tensor_scale_epilogue(const float *input,
                                              const float *weight_tensor_scale,
                                              const float *input_tensor_scale,
                                              uint16_t *output, uint64_t m,
                                              uint64_t n,
                                              hipStream_t stream) noexcept;

hipError_t launch_fp8_outer_decode_gfx1030_half2_wave4col32(
    const uint8_t *activation, const float *activation_scales,
    const uint8_t *weight, const float *weight_scales, uint16_t *output,
    uint64_t m, uint64_t k, uint64_t n, hipStream_t stream) noexcept;

hipError_t launch_fp8_outer_decode_gfx1030_dword8_wave4col32(
    const uint8_t *activation, const float *activation_scales,
    const uint8_t *weight, const float *weight_scales, uint16_t *output,
    uint64_t m, uint64_t k, uint64_t n, hipStream_t stream) noexcept;

hipError_t launch_fp8_outer_decode_gfx1030_lds_lut_wave4col32(
    const uint8_t *activation, const float *activation_scales,
    const uint8_t *weight, const float *weight_scales, uint16_t *output,
    uint64_t m, uint64_t k, uint64_t n, hipStream_t stream) noexcept;

hipError_t launch_fp8_outer_decode_gfx1030_lds_lut_k5120n17408(
    const uint8_t *activation, const float *activation_scales,
    const uint8_t *weight, const float *weight_scales, uint16_t *output,
    uint64_t m, uint64_t k, uint64_t n, hipStream_t stream) noexcept;

hipError_t launch_fp8_outer_decode_gfx1030_lds_lut_k6144n5120(
    const uint8_t *activation, const float *activation_scales,
    const uint8_t *weight, const float *weight_scales, uint16_t *output,
    uint64_t m, uint64_t k, uint64_t n, hipStream_t stream) noexcept;

hipError_t launch_fp8_outer_decode_gfx1030_lds_lut_k5120n10240(
    const uint8_t *activation, const float *activation_scales,
    const uint8_t *weight, const float *weight_scales, uint16_t *output,
    uint64_t m, uint64_t k, uint64_t n, hipStream_t stream) noexcept;

hipError_t launch_fp8_outer_decode_gfx1030_lds_lut_k5120n6144(
    const uint8_t *activation, const float *activation_scales,
    const uint8_t *weight, const float *weight_scales, uint16_t *output,
    uint64_t m, uint64_t k, uint64_t n, hipStream_t stream) noexcept;

hipError_t launch_fp8_outer_decode_gfx1030_activation_shared_wave4col32(
    const uint8_t *activation, const float *activation_scales,
    const uint8_t *weight, const float *weight_scales, uint16_t *output,
    uint64_t m, uint64_t k, uint64_t n, hipStream_t stream) noexcept;

hipError_t launch_fp8_outer_decode_gfx1030_activation_shared_wave8col64(
    const uint8_t *activation, const float *activation_scales,
    const uint8_t *weight, const float *weight_scales, uint16_t *output,
    uint64_t m, uint64_t k, uint64_t n, hipStream_t stream) noexcept;

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

// Private ID62 gfx1030 M=17 split-K4 candidate. The caller reserves
// 4 * m * n * sizeof(float) bytes for partial_workspace; the launcher accepts
// only (17, 5120, 17408) and (17, 17408, 5120).
hipError_t launch_nvfp4_w4a4_prefill_dp4a_short_split4(
    const uint8_t *packed_activation, const uint8_t *activation_block_scales,
    const uint8_t *packed_weight, const uint8_t *weight_block_scales,
    const float *weight_tensor_scale, const float *input_tensor_scale,
    uint16_t *output, float *partial_workspace, uint64_t m, uint64_t k,
    uint64_t n, hipStream_t stream) noexcept;

// Private ID64 gfx1201 M=17 down-shape split-K4 candidate. The partial planes
// are appended to the existing per-plan NVFP4 workspace.
hipError_t launch_nvfp4_w4a4_prefill_gfx1201_wmma128x64_split4(
    const uint8_t *packed_activation, const uint8_t *activation_block_scales,
    const uint8_t *packed_weight, const uint8_t *weight_block_scales,
    const float *weight_tensor_scale, const float *input_tensor_scale,
    uint16_t *output, float *partial_workspace, uint64_t m, uint64_t k,
    uint64_t n, hipStream_t stream) noexcept;

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
