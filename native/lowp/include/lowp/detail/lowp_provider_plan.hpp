#ifndef LOWP_PROVIDER_PLAN_HPP
#define LOWP_PROVIDER_PLAN_HPP
#include "low_precision_matmul_provider.hpp"
#include "lowp_kernel_internal.hpp"
#include <limits>
#include <optional>
namespace sllm_lowp {
inline std::optional<sllm_lowp::PreparedProviderPlan> concrete_provider_plan(
    const sllm_lowp::PreparedProviderPlan &provider,
    const sllm_matmul_kernel::KernelVariant variant) noexcept {
  using sllm_lowp::InnerProduct;
  using sllm_lowp::MatmulFormat;
  using sllm_lowp::ProviderKind;
  using sllm_lowp::TilePolicy;
  using sllm_matmul_kernel::KernelVariant;

  ProviderKind concrete_provider = ProviderKind::Unsupported;
  TilePolicy concrete_tile = TilePolicy::None;
  InnerProduct concrete_inner_product = InnerProduct::None;

  switch (variant) {
  case KernelVariant::Fp8Emulation:
    if (provider.format != MatmulFormat::Fp8OuterE4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Fp8OuterGfx1030Software;
    concrete_tile = TilePolicy::Elementwise;
    concrete_inner_product = InnerProduct::E4M3OuterFp32;
    break;
  case KernelVariant::Fp8OuterPrefillTiled16:
    if (provider.format != MatmulFormat::Fp8OuterE4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Fp8OuterGfx1030Software;
    concrete_tile = TilePolicy::BlockTiled16x16;
    concrete_inner_product = InnerProduct::E4M3OuterFp32;
    break;
  case KernelVariant::Fp8OuterPrefillGfx1030Half2_128x64:
    if (provider.format != MatmulFormat::Fp8OuterE4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Fp8OuterGfx1030Software;
    concrete_tile = TilePolicy::BlockRow128Column64;
    concrete_inner_product = InnerProduct::E4M3Fp16Dot2Fp32;
    break;
  case KernelVariant::Fp8OuterPrefillGfx1030Half2_64x64:
    if (provider.format != MatmulFormat::Fp8OuterE4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030 ||
        provider.m <= 1U) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Fp8OuterGfx1030Software;
    concrete_tile = ::sllm_matmul_kernel::
                            fp8_outer_prefill_gfx1030_half2_short_m32_n64_shape(
                                provider.m, provider.k, provider.n)
                        ? TilePolicy::BlockRow32Column64
                    : ::sllm_matmul_kernel::
                            fp8_outer_prefill_gfx1030_half2_short_m32_n32_shape(
                                provider.m, provider.k, provider.n)
                        ? TilePolicy::BlockRow32Column32
                        : TilePolicy::BlockRow64Column64;
    concrete_inner_product = InnerProduct::E4M3Fp16Dot2Fp32;
    break;
  case KernelVariant::Fp8OuterDecodeGfx1030Half2Wave4Col32:
    if (provider.format != MatmulFormat::Fp8OuterE4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030 ||
        provider.m != 1U) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Fp8OuterGfx1030Software;
    concrete_tile = TilePolicy::DecodeWave4Column32;
    concrete_inner_product = InnerProduct::E4M3Fp16Dot2Fp32;
    break;
  case KernelVariant::Fp8OuterDecodeGfx1030Dword8Wave4Col32:
    if (provider.format != MatmulFormat::Fp8OuterE4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030 ||
        provider.m != 1U) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Fp8OuterGfx1030Software;
    concrete_tile = TilePolicy::DecodeDword8Wave4Column32;
    concrete_inner_product = InnerProduct::E4M3Fp16Dot2Fp32;
    break;
  case KernelVariant::Fp8OuterDecodeGfx1030LdsLutWave4Col32:
    if (provider.format != MatmulFormat::Fp8OuterE4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030 ||
        !::sllm_matmul_kernel::fp8_outer_decode_gfx1030_half2_shape(
            provider.m, provider.k, provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Fp8OuterGfx1030Software;
    concrete_tile = TilePolicy::DecodeDword8Wave4Column32;
    concrete_inner_product = InnerProduct::E4M3Fp16Dot2Fp32;
    break;
  case KernelVariant::Fp8OuterDecodeGfx1030FusedM2_4:
    if (provider.format != MatmulFormat::Fp8OuterE4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030 ||
        !::sllm_matmul_kernel::fp8_outer_decode_gfx1030_fused_m2_4_shape(
            provider.m, provider.k, provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Fp8OuterGfx1030Software;
    concrete_tile = TilePolicy::DecodeDword8Wave4Column32;
    concrete_inner_product = InnerProduct::E4M3Fp16Dot2Fp32;
    break;
  case KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave4Col32:
    if (provider.format != MatmulFormat::Fp8OuterE4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030 ||
        !::sllm_matmul_kernel::
            fp8_outer_decode_gfx1030_activation_shared_wave4_shape(
                provider.m, provider.k, provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Fp8OuterGfx1030Software;
    // Keep the existing public decode tile contract; kernel identity carries
    // the activation-shared distinction without widening the provider ABI.
    concrete_tile = TilePolicy::DecodeDword8Wave4Column32;
    concrete_inner_product = InnerProduct::E4M3Fp16Dot2Fp32;
    break;
  case KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave8Col64:
    if (provider.format != MatmulFormat::Fp8OuterE4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030 ||
        !::sllm_matmul_kernel::
            fp8_outer_decode_gfx1030_activation_shared_wave8_shape(
                provider.m, provider.k, provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Fp8OuterGfx1030Software;
    concrete_tile = TilePolicy::DecodeWave4Column32;
    concrete_inner_product = InnerProduct::E4M3Fp16Dot2Fp32;
    break;
  case KernelVariant::Nvfp4W4A4SmallMVgprReuse:
    if (provider.format != MatmulFormat::Nvfp4W4A4 ||
        (provider.target != sllm_lowp::ExactTarget::Gfx1030 &&
         provider.target != sllm_lowp::ExactTarget::Gfx1201) ||
        !::sllm_matmul_kernel::phase83_nvfp4_w4a4_small_m_vgpr_reuse_shape(
            provider.m, provider.k, provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::DecodeWave4Column32;
    concrete_inner_product = InnerProduct::E2M1BlockScaledDp4aFp32;
    break;
  case KernelVariant::Nvfp4W4A4SmallMRowGrid:
    if (provider.format != MatmulFormat::Nvfp4W4A4 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030 || provider.m < 2U ||
        provider.m > 4U ||
        !::sllm_matmul_kernel::
            phase78_nvfp4_w4a4_decode_activation_shared_shape(1U, provider.k,
                                                              provider.n))
      return std::nullopt;
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::DecodeWave4Column32;
    concrete_inner_product = InnerProduct::E2M1BlockScaledDp4aFp32;
    break;
  case KernelVariant::Nvfp4W4A4SmallMGfx1201RowGrid:
    if (provider.format != MatmulFormat::Nvfp4W4A4 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1201 ||
        !::sllm_matmul_kernel::phase83_gfx1201_nvfp4_w4a4_small_m_rowgrid_shape(
            provider.m, provider.k, provider.n))
      return std::nullopt;
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::DecodeWave4Column32;
    concrete_inner_product = InnerProduct::E2M1BlockScaledDp4aFp32;
    break;
  case KernelVariant::Nvfp4W4A4DecodeScaleLut:
    if (provider.format != MatmulFormat::Nvfp4W4A4 || provider.m != 1U ||
        ((provider.target == sllm_lowp::ExactTarget::Gfx1030 &&
          !::sllm_matmul_kernel::
              phase78_nvfp4_w4a4_decode_activation_shared_shape(
                  provider.m, provider.k, provider.n)) ||
         (provider.target == sllm_lowp::ExactTarget::Gfx1201 &&
          !::sllm_matmul_kernel::phase78_nvfp4_w4a4_decode_wave4col32_shape(
              provider.m, provider.k, provider.n)) ||
         (provider.target != sllm_lowp::ExactTarget::Gfx1030 &&
          provider.target != sllm_lowp::ExactTarget::Gfx1201))) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::DecodeWave4Column32;
    concrete_inner_product = InnerProduct::E2M1BlockScaledDp4aFp32;
    break;
  case KernelVariant::Mxfp8W8A8Decode:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A8) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8Block32;
    concrete_tile = TilePolicy::DecodeRowReduction;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp8W8A8M1Col2:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A8 ||
        (provider.target != sllm_lowp::ExactTarget::Gfx1030 &&
         provider.target != sllm_lowp::ExactTarget::Gfx1201) ||
        !::sllm_matmul_kernel::phase85_mxfp_m1_col2_shape(
            provider.m, provider.k, provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8Block32;
    concrete_tile = TilePolicy::DecodeRowReduction;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp8W8A16M1Col2:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A16 ||
        (provider.target != sllm_lowp::ExactTarget::Gfx1030 &&
         provider.target != sllm_lowp::ExactTarget::Gfx1201) ||
        !::sllm_matmul_kernel::phase85_mxfp_m1_a16_shape(provider.m, provider.k,
                                                         provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8A16Block32;
    concrete_tile = TilePolicy::DecodeRowReduction;
    concrete_inner_product = InnerProduct::E4M3Bf16Fp32;
    break;
  case KernelVariant::Mxfp8W8A8Prefill:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A8) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8Block32;
    concrete_tile = TilePolicy::Elementwise;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp8W8A8PrefillRow8:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A8) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8Block32;
    concrete_tile = TilePolicy::BlockRow8;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp8W8A8PrefillTiled16:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A8) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8Block32;
    concrete_tile = TilePolicy::BlockTiled16x16;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp8W8A8PrefillMmqCol4:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A8) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8Block32;
    concrete_tile = TilePolicy::BlockRow8Column4;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp8W8A8PrefillMmqCol8:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A8) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8Block32;
    concrete_tile = TilePolicy::BlockRow8Column8;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp8W8A8PrefillPhase85SmallM:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A8 ||
        !::sllm_matmul_kernel::phase85_mxfp_small_m_force_shape(
            provider.m, provider.k, provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8Block32;
    concrete_tile = TilePolicy::BlockRow4Column8;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Vector32:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8Block32;
    concrete_tile = TilePolicy::BlockRow8Column8;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K32Double:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8Block32;
    concrete_tile = TilePolicy::BlockRow128Column64;
    concrete_inner_product = InnerProduct::E4M3Fp16Dot2Fp32;
    break;
  case KernelVariant::Mxfp8W8A8PrefillWmmaN16:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A8) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8Gfx1201Wmma;
    concrete_tile = TilePolicy::Wmma128x16x32;
    concrete_inner_product = InnerProduct::E4M3WmmaFp32;
    break;
  case KernelVariant::Mxfp8W8A8PrefillWmmaN64:
  case KernelVariant::Mxfp8W8A8PrefillWmmaDirectWeight:
  case KernelVariant::Mxfp8W8A8PrefillWmmaDirectBoth:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A8) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8Gfx1201Wmma;
    concrete_tile = TilePolicy::Wmma128x64x32;
    concrete_inner_product = InnerProduct::E4M3WmmaFp32;
    break;
  case KernelVariant::Mxfp8W8A8PrefillWmmaN128DirectBoth:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A8) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8Gfx1201Wmma;
    concrete_tile = TilePolicy::Wmma128x128x32;
    concrete_inner_product = InnerProduct::E4M3WmmaFp32;
    break;
  case KernelVariant::Mxfp6W6A6Decode:
    if (provider.format != MatmulFormat::Mxfp6E3M2W6A6) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp6Block32;
    concrete_tile = TilePolicy::DecodeRowReduction;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp6W6A6M1Col2:
    if (provider.format != MatmulFormat::Mxfp6E3M2W6A6 ||
        (provider.target != sllm_lowp::ExactTarget::Gfx1030 &&
         provider.target != sllm_lowp::ExactTarget::Gfx1201) ||
        !::sllm_matmul_kernel::phase85_mxfp_m1_col2_shape(
            provider.m, provider.k, provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp6Block32;
    concrete_tile = TilePolicy::DecodeRowReduction;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp6W6A16M1Col2:
    if (provider.format != MatmulFormat::Mxfp6E3M2W6A16 ||
        (provider.target != sllm_lowp::ExactTarget::Gfx1030 &&
         provider.target != sllm_lowp::ExactTarget::Gfx1201) ||
        !::sllm_matmul_kernel::phase85_mxfp_m1_a16_shape(provider.m, provider.k,
                                                         provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp6A16Block32;
    concrete_tile = TilePolicy::DecodeRowReduction;
    concrete_inner_product = InnerProduct::E3M2Bf16Fp32;
    break;
  case KernelVariant::Mxfp6W6A6Prefill:
    if (provider.format != MatmulFormat::Mxfp6E3M2W6A6) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp6Block32;
    concrete_tile = TilePolicy::Elementwise;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp6W6A6PrefillRow8:
    if (provider.format != MatmulFormat::Mxfp6E3M2W6A6) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp6Block32;
    concrete_tile = TilePolicy::BlockRow8;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp6W6A6PrefillTiled16:
    if (provider.format != MatmulFormat::Mxfp6E3M2W6A6) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp6Block32;
    concrete_tile = TilePolicy::BlockTiled16x16;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp6W6A6PrefillMmqCol4:
    if (provider.format != MatmulFormat::Mxfp6E3M2W6A6) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp6Block32;
    concrete_tile = TilePolicy::BlockRow8Column4;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp6W6A6PrefillMmqCol8:
    if (provider.format != MatmulFormat::Mxfp6E3M2W6A6) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp6Block32;
    concrete_tile = TilePolicy::BlockRow8Column8;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp6W6A6PrefillPhase85SmallM:
    if (provider.format != MatmulFormat::Mxfp6E3M2W6A6 ||
        !::sllm_matmul_kernel::phase85_mxfp_small_m_force_shape(
            provider.m, provider.k, provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp6Block32;
    concrete_tile = TilePolicy::BlockRow4Column8;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp6W6A6PrefillGfx1030Half2Dot2:
    if (provider.format != MatmulFormat::Mxfp6E3M2W6A6 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp6Gfx1030Half2Dot2;
    concrete_tile = TilePolicy::BlockRow32Column32;
    concrete_inner_product = InnerProduct::E3M2Fp16Dot2Fp32;
    break;
  case KernelVariant::Mxfp6W6A6PrefillGfx1030Half2_128x64K32DoublePack4:
    if (provider.format != MatmulFormat::Mxfp6E3M2W6A6 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp6Gfx1030Half2Dot2;
    concrete_tile = TilePolicy::BlockRow128Column64;
    concrete_inner_product = InnerProduct::E3M2Fp16Dot2Fp32;
    break;
  case KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201ViaE4M3N64:
    if (provider.format != MatmulFormat::Mxfp6E3M2W6A6 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1201) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp6Gfx1201WmmaViaE4M3;
    concrete_tile = TilePolicy::Wmma128x64x32;
    concrete_inner_product = InnerProduct::E3M2ViaE4M3WmmaFp32;
    break;
  case KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4N64:
  case KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4Swar:
    if (provider.format != MatmulFormat::Mxfp6E3M2W6A6 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1201) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp6Gfx1201WmmaViaE4M3;
    concrete_tile = TilePolicy::Wmma128x64x32;
    concrete_inner_product = InnerProduct::E3M2ViaE4M3WmmaFp32;
    break;
  case KernelVariant::Nvfp4DecodePackedDequant:
    if (provider.format != MatmulFormat::Nvfp4W4A16) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A16Block16;
    concrete_tile = TilePolicy::DecodeRowReduction;
    concrete_inner_product = InnerProduct::E2M1Bf16Fp32;
    break;
  case KernelVariant::Nvfp4PrefillRow8Tiled256:
    if (provider.format != MatmulFormat::Nvfp4W4A16) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A16Block16;
    concrete_tile = TilePolicy::PackedRow8;
    concrete_inner_product = InnerProduct::E2M1Bf16Fp32;
    break;
  case KernelVariant::Nvfp4BaselinePackedDequant:
    if (provider.format != MatmulFormat::Nvfp4W4A16) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A16Block16;
    concrete_tile = TilePolicy::Elementwise;
    concrete_inner_product = InnerProduct::E2M1Bf16Fp32;
    break;
  case KernelVariant::Nvfp4W4A4Packed:
    if (provider.format != MatmulFormat::Nvfp4W4A4) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::Elementwise;
    concrete_inner_product = InnerProduct::E2M1BlockScaledFp32;
    break;
  case KernelVariant::Nvfp4W4A4PrefillRow8Tiled256:
    if (provider.format != MatmulFormat::Nvfp4W4A4 || provider.m <= 1U) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::PackedRow8;
    concrete_inner_product = InnerProduct::E2M1BlockScaledFp32;
    break;
  case KernelVariant::Nvfp4W4A4PrefillRow8Col8Tiled256:
    if (provider.format != MatmulFormat::Nvfp4W4A4 || provider.m <= 1U) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::BlockRow8Column8;
    concrete_inner_product = InnerProduct::E2M1BlockScaledFp32;
    break;
  case KernelVariant::Nvfp4W4A4PrefillDp4a64x64:
  case KernelVariant::Nvfp4W4A4PrefillCompensated64x64:
    if (provider.format != MatmulFormat::Nvfp4W4A4 || provider.m <= 1U ||
        provider.k == 0U || (provider.k % 16U) != 0U) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile =
        variant == KernelVariant::Nvfp4W4A4PrefillCompensated64x64 &&
                provider.target == sllm_lowp::ExactTarget::Gfx1030 &&
                ::sllm_matmul_kernel::
                    phase83_gfx1030_nvfp4_w4a4_compensated128x64_shape(
                        provider.m, provider.k, provider.n)
            ? TilePolicy::BlockRow128Column64
            : TilePolicy::BlockRow64Column64;
    concrete_inner_product = InnerProduct::E2M1BlockScaledFp32;
    break;
  case KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan:
    if (provider.format != MatmulFormat::Nvfp4W4A4 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1201 ||
        !::sllm_matmul_kernel::phase83_gfx1201_nvfp4_w4a4_wmma_kahan_shape(
            provider.m, provider.k, provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::Wmma128x64x32;
    concrete_inner_product = InnerProduct::E2M1ViaE4M3WmmaFp32;
    break;
  case KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x64:
    if (provider.format != MatmulFormat::Nvfp4W4A4 || provider.m <= 1U ||
        provider.k == 0U || (provider.k % 16U) != 0U ||
        provider.target != sllm_lowp::ExactTarget::Gfx1201) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::Wmma128x64x32;
    concrete_inner_product = InnerProduct::E2M1ViaE4M3WmmaFp32;
    break;
  case KernelVariant::Nvfp4W4A4PrefillGfx1201F16Staging:
    if (provider.format != MatmulFormat::Nvfp4W4A4 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1201 ||
        !::sllm_matmul_kernel::phase78_gfx1201_nvfp4_w4a4_f16_staging_shape(
            provider.m, provider.k, provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::BlockTiled16x16;
    concrete_inner_product = InnerProduct::E2M1Fp16ScaleWmmaFp32;
    break;
  case KernelVariant::Nvfp4W4A4Decode:
    if (provider.format != MatmulFormat::Nvfp4W4A4 || provider.m != 1U) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::DecodeRowReduction;
    concrete_inner_product = InnerProduct::E2M1BlockScaledFp32;
    break;
  case KernelVariant::Nvfp4W4A4DecodeWave4Column32:
    if (provider.format != MatmulFormat::Nvfp4W4A4 || provider.m != 1U ||
        !::sllm_matmul_kernel::phase78_nvfp4_w4a4_decode_wave4col32_shape(
            provider.m, provider.k, provider.n) ||
        (provider.target != sllm_lowp::ExactTarget::Gfx1030 &&
         provider.target != sllm_lowp::ExactTarget::Gfx1201)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::DecodeWave4Column32;
    concrete_inner_product = InnerProduct::E2M1BlockScaledDp4aFp32;
    break;
  case KernelVariant::Nvfp4W4A4DecodeActivationShared:
    if (provider.format != MatmulFormat::Nvfp4W4A4 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030 ||
        !::sllm_matmul_kernel::
            phase78_nvfp4_w4a4_decode_activation_shared_shape(
                provider.m, provider.k, provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::DecodeWave4Column32;
    concrete_inner_product = InnerProduct::E2M1BlockScaledDp4aFp32;
    break;
  case KernelVariant::Mxfp4W4A4Decode:
    if (provider.format != MatmulFormat::Mxfp4W4A4) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp4W4A4Block32;
    concrete_tile = TilePolicy::DecodeRowReduction;
    concrete_inner_product = InnerProduct::E2M1BlockScaledFp32;
    break;
  case KernelVariant::Mxfp4W4A4Prefill:
    if (provider.format != MatmulFormat::Mxfp4W4A4) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp4W4A4Block32;
    concrete_tile = TilePolicy::Elementwise;
    concrete_inner_product = InnerProduct::E2M1BlockScaledFp32;
    break;
  default:
    return std::nullopt;
  }

  return sllm_lowp::with_execution_semantics(
      provider, concrete_provider, concrete_tile, concrete_inner_product);
}

inline bool
activation_workspace_bytes(const sllm_lowp::PreparedProviderPlan &provider,
                           uint64_t *const workspace_bytes) noexcept {
  if (workspace_bytes == nullptr) {
    return false;
  }
  *workspace_bytes = 0U;
  const uint64_t block_size =
      static_cast<uint64_t>(provider.block_contract.activation_block_size);
  const uint64_t bits =
      static_cast<uint64_t>(provider.block_contract.activation_bits);
  if (block_size == 0U || bits == 0U) {
    return true;
  }
  // Every accepted low-precision matmul has K aligned to its block size, but
  // retain the rounded expressions so this remains correct at descriptor
  // boundaries and documents the storage contract at the allocation site.
  const uint64_t max_value = std::numeric_limits<uint64_t>::max();
  if (provider.k > (max_value - UINT64_C(7)) / bits ||
      provider.k > max_value - (block_size - UINT64_C(1))) {
    return false;
  }
  const uint64_t value_bytes = (provider.k * bits + UINT64_C(7)) / UINT64_C(8);
  const uint64_t scale_bytes =
      (provider.k + block_size - UINT64_C(1)) / block_size;
  if (provider.m > max_value / value_bytes ||
      provider.m > max_value / scale_bytes) {
    return false;
  }
  const uint64_t value_total = provider.m * value_bytes;
  const uint64_t scale_total = provider.m * scale_bytes;
  if (value_total > max_value - scale_total) {
    return false;
  }
  *workspace_bytes = value_total + scale_total;
  return true;
}

} // namespace sllm_lowp
#endif
