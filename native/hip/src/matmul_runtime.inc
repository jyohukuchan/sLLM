namespace {

bool matmul_lowp_format(const sllm_matmul::DescriptorMetadata &metadata,
                        sllm_lowp::MatmulFormat *const format) noexcept {
  if (format == nullptr) {
    return false;
  }
  if (metadata.mxfp8_w8a8) {
    *format = sllm_lowp::MatmulFormat::Mxfp8E4M3W8A8;
    return true;
  }
  if (metadata.mxfp6_w6a6) {
    *format = sllm_lowp::MatmulFormat::Mxfp6E3M2W6A6;
    return true;
  }
  if (metadata.nvfp4_w4a4) {
    *format = sllm_lowp::MatmulFormat::Nvfp4W4A4;
    return true;
  }
  if (metadata.nvfp4) {
    *format = sllm_lowp::MatmulFormat::Nvfp4W4A16;
    return true;
  }
  if (metadata.mxfp4_w4a4) {
    *format = sllm_lowp::MatmulFormat::Mxfp4W4A4;
    return true;
  }
  return false;
}

bool matmul_lowp_path(
    const sllm_matmul::DescriptorMetadata &metadata) noexcept {
  sllm_lowp::MatmulFormat unused{};
  return matmul_lowp_format(metadata, &unused);
}

sllm_lowp::ExactTarget
matmul_lowp_target(const char *const arch_name) noexcept {
  return matches_runtime_gcn_arch(arch_name, "gfx1030")
             ? sllm_lowp::ExactTarget::Gfx1030
         : matches_runtime_gcn_arch(arch_name, "gfx1201")
             ? sllm_lowp::ExactTarget::Gfx1201
         : matches_runtime_gcn_arch(arch_name, "gfx942")
             ? sllm_lowp::ExactTarget::Gfx942SrameccOnXnackOff
             : sllm_lowp::ExactTarget::Unknown;
}

const char *
matmul_lowp_selector_target(const sllm_lowp::ExactTarget target) noexcept {
  switch (target) {
  case sllm_lowp::ExactTarget::Gfx1030:
    return "gfx1030";
  case sllm_lowp::ExactTarget::Gfx1201:
    return "gfx1201";
  case sllm_lowp::ExactTarget::Gfx942SrameccOnXnackOff:
    return "gfx942:sramecc+:xnack-";
  case sllm_lowp::ExactTarget::Unknown:
    break;
  }
  return "";
}

sllm_matmul_kernel::KernelVariant
matmul_lowp_kernel_variant(const sllm_matmul::DescriptorMetadata &metadata,
                           const sllm_lowp::PreparedProviderPlan &provider,
                           const char *const arch_name) noexcept {
  (void)arch_name;
  const char *const selector_target =
      matmul_lowp_selector_target(provider.target);
  switch (provider.provider) {
  case sllm_lowp::ProviderKind::Mxfp8Block32:
  case sllm_lowp::ProviderKind::Mxfp8Gfx1201Wmma:
    return ::sllm_matmul_kernel::select_mxfp8_variant(
        metadata.m, metadata.k, metadata.n, selector_target);
  case sllm_lowp::ProviderKind::Mxfp6Block32:
  case sllm_lowp::ProviderKind::Mxfp6Gfx1030MmqViaE4M3:
  case sllm_lowp::ProviderKind::Mxfp6Gfx1030Half2Dot2:
  case sllm_lowp::ProviderKind::Mxfp6Gfx1201WmmaViaE4M3:
    return ::sllm_matmul_kernel::select_mxfp6_variant(
        metadata.m, metadata.k, metadata.n, selector_target);
  case sllm_lowp::ProviderKind::Nvfp4W4A16Block16:
    return ::sllm_matmul_kernel::select_nvfp4_variant(metadata.m);
  case sllm_lowp::ProviderKind::Nvfp4W4A4Block16:
    return ::sllm_matmul_kernel::select_nvfp4_w4a4_variant(
        metadata.m, metadata.k, metadata.n, selector_target);
  case sllm_lowp::ProviderKind::Mxfp4W4A4Block32:
    return ::sllm_matmul_kernel::select_mxfp4_variant(metadata.m);
  case sllm_lowp::ProviderKind::Fp8OuterGfx1030Software:
  case sllm_lowp::ProviderKind::Unsupported:
    break;
  }
  return ::sllm_matmul_kernel::KernelVariant::Baseline;
}

std::optional<sllm_lowp::PreparedProviderPlan>
matmul_lowp_concrete_provider_plan(
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
  case KernelVariant::Fp8OuterPrefillGfx1030LdsLut:
    if (provider.format != MatmulFormat::Fp8OuterE4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030 ||
        !::sllm_matmul_kernel::fp8_outer_prefill_gfx1030_lds_lut_shape(
            provider.m, provider.k, provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Fp8OuterGfx1030Software;
    concrete_tile = TilePolicy::BlockRow64Column64;
    concrete_inner_product = InnerProduct::E4M3Fp16Dot2Fp32;
    break;
  case KernelVariant::Fp8OuterPrefillGfx1030F16Staging:
    if (provider.format != MatmulFormat::Fp8OuterE4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030 ||
        !::sllm_matmul_kernel::fp8_outer_prefill_gfx1030_f16_staging_shape(
            provider.m, provider.k, provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Fp8OuterGfx1030Software;
    concrete_tile = TilePolicy::BlockTiled16x16;
    concrete_inner_product = InnerProduct::E4M3OuterFp32;
    break;
  case KernelVariant::Fp8OuterPrefillGfx1030F16TileStaging:
    if (provider.format != MatmulFormat::Fp8OuterE4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030 ||
        !::sllm_matmul_kernel::fp8_outer_prefill_gfx1030_f16_tile_staging_shape(
            provider.m, provider.k, provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Fp8OuterGfx1030Software;
    concrete_tile = TilePolicy::BlockRow64Column64;
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
  case KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col16:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8Block32;
    concrete_tile = TilePolicy::BlockRow8Column16;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col32:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8Block32;
    concrete_tile = TilePolicy::BlockRow8Column32;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Regscale:
  case KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Vector32:
  case KernelVariant::Mxfp8W8A8PrefillMmqGfx1030RegscaleVector32:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8Block32;
    concrete_tile = TilePolicy::BlockRow8Column8;
    concrete_inner_product = InnerProduct::DecodedBlockScaledFp32;
    break;
  case KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_32x32K32:
  case KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_64x64K32:
  case KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x32K32:
  case KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K32:
  case KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K64:
  case KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K128:
  case KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K32Double:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A8 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8Block32;
    concrete_tile =
        variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_32x32K32
            ? TilePolicy::BlockRow32Column32
        : variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_64x64K32
            ? TilePolicy::BlockRow64Column64
        : variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x32K32
            ? TilePolicy::BlockRow128Column32
            : TilePolicy::BlockRow128Column64;
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
  case KernelVariant::Mxfp8W8A8PrefillWmma4Wave:
    if (provider.format != MatmulFormat::Mxfp8E4M3W8A8) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp8Gfx1201Wmma;
    concrete_tile = TilePolicy::Wmma64x64x32;
    concrete_inner_product = InnerProduct::E4M3WmmaFp32;
    break;
  case KernelVariant::Mxfp8W8A8PrefillWmmaN64:
  case KernelVariant::Mxfp8W8A8PrefillWmmaLdsPad:
  case KernelVariant::Mxfp8W8A8PrefillWmmaDirectWeight:
  case KernelVariant::Mxfp8W8A8PrefillWmmaDirectActivation:
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
  case KernelVariant::Mxfp6W6A6PrefillMmqGfx1030ViaE4M3:
    if (provider.format != MatmulFormat::Mxfp6E3M2W6A6 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1030) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp6Gfx1030MmqViaE4M3;
    concrete_tile = TilePolicy::BlockRow8Column8;
    concrete_inner_product = InnerProduct::E3M2ViaE4M3DecodedFp32;
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
  case KernelVariant::Mxfp6W6A6PrefillGfx1030Half2_128x64K32DoubleScalar:
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
  case KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4N128:
    if (provider.format != MatmulFormat::Mxfp6E3M2W6A6 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1201) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Mxfp6Gfx1201WmmaViaE4M3;
    concrete_tile =
        variant == KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4N128
            ? TilePolicy::Wmma128x128x32
            : TilePolicy::Wmma128x64x32;
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
  case KernelVariant::Nvfp4W4A4PrefillDp4a64x64K128:
    if (provider.format != MatmulFormat::Nvfp4W4A4 || provider.m <= 1U ||
        provider.k == 0U || (provider.k % 16U) != 0U) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::BlockRow64Column64;
    concrete_inner_product = InnerProduct::E2M1BlockScaledFp32;
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
  case KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x32:
    if (provider.format != MatmulFormat::Nvfp4W4A4 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1201 ||
        !::sllm_matmul_kernel::phase78_gfx1201_nvfp4_w4a4_wmma128x32_shape(
            provider.m, provider.k, provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::BlockRow128Column32;
    concrete_inner_product = InnerProduct::E2M1ViaE4M3WmmaFp32;
    break;
  case KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaF16Scale128x64:
    if (provider.format != MatmulFormat::Nvfp4W4A4 || provider.m <= 1U ||
        provider.k == 0U || (provider.k % 16U) != 0U ||
        provider.target != sllm_lowp::ExactTarget::Gfx1201 ||
        !::sllm_matmul_kernel::phase78_gfx1201_nvfp4_w4a4_wmma128x64_shape(
            provider.m, provider.k, provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::Wmma128x64x32;
    concrete_inner_product = InnerProduct::E2M1Fp16ScaleWmmaFp32;
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
  case KernelVariant::Nvfp4W4A4PrefillGfx1201Fp8Staging:
    if (provider.format != MatmulFormat::Nvfp4W4A4 ||
        provider.target != sllm_lowp::ExactTarget::Gfx1201 ||
        !::sllm_matmul_kernel::phase78_gfx1201_nvfp4_w4a4_fp8_staging_shape(
            provider.m, provider.k, provider.n)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::BlockTiled16x16;
    concrete_inner_product = InnerProduct::E2M1ViaE4M3NativeFp8Fp32;
    break;
  case KernelVariant::Nvfp4W4A4Decode:
    if (provider.format != MatmulFormat::Nvfp4W4A4 || provider.m != 1U) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::DecodeRowReduction;
    concrete_inner_product = InnerProduct::E2M1BlockScaledFp32;
    break;
  case KernelVariant::Nvfp4W4A4DecodeColumns128:
    if (provider.format != MatmulFormat::Nvfp4W4A4 || provider.m != 1U ||
        provider.k == 0U ||
        !::sllm_matmul_kernel::phase78_nvfp4_w4a4_decode_columns128_shape(
            provider.m, provider.k, provider.n) ||
        (provider.target != sllm_lowp::ExactTarget::Gfx1030 &&
         provider.target != sllm_lowp::ExactTarget::Gfx1201)) {
      return std::nullopt;
    }
    concrete_provider = ProviderKind::Nvfp4W4A4Block16;
    concrete_tile = TilePolicy::DecodeColumns128;
    concrete_inner_product = InnerProduct::E2M1BlockScaledDp4aFp32;
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

bool matmul_lowp_workspace_bytes(
    const sllm_lowp::PreparedProviderPlan &provider,
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

// The caller holds the context accounting lock. The cache is intentionally
// format-neutral so a later packed-low-precision ingress can reuse the same
// FP16/F32 scratch and queue-ownership contract without adding a persistent
// expanded weight representation.
hipError_t
reserve_matmul_f16_staging_workspace_locked(Context *const context,
                                            const uint64_t required) noexcept {
  if (context == nullptr || required == 0U ||
      required >
          static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return hipErrorOutOfMemory;
  }
  if ((context->matmul_f16_staging_in_flight == 0U) !=
      (context->matmul_f16_staging_owner_queue == nullptr)) {
    context->poisoned.store(true);
    return hipErrorUnknown;
  }
  if (context->matmul_f16_staging_workspace != nullptr &&
      context->matmul_f16_staging_workspace_bytes >= required) {
    context->matmul_f16_staging_workspace_high_water_bytes = std::max(
        context->matmul_f16_staging_workspace_high_water_bytes, required);
    return hipSuccess;
  }
  if (context->matmul_f16_staging_in_flight != 0U) {
    return hipErrorNotReady;
  }
  void *replacement = nullptr;
  const hipError_t allocation_status =
      hipMalloc(&replacement, static_cast<std::size_t>(required));
  if (allocation_status != hipSuccess) {
    return allocation_status;
  }
  void *const previous = context->matmul_f16_staging_workspace;
  if (previous != nullptr) {
    const hipError_t free_status =
        free_allocation_with_fault_injection(previous);
    if (free_status != hipSuccess) {
      context->matmul_f16_staging_workspace = nullptr;
      context->matmul_f16_staging_workspace_bytes = 0U;
      context->matmul_f16_staging_owner_queue = nullptr;
      context->poisoned.store(true);
      orphan_owner.retain_allocation(nullptr, previous);
      const hipError_t replacement_free_status =
          free_allocation_with_fault_injection(replacement);
      if (replacement_free_status != hipSuccess) {
        orphan_owner.retain_allocation(nullptr, replacement);
      }
      return free_status;
    }
  }
  context->matmul_f16_staging_workspace = replacement;
  context->matmul_f16_staging_workspace_bytes = required;
  context->matmul_f16_staging_workspace_high_water_bytes = std::max(
      context->matmul_f16_staging_workspace_high_water_bytes, required);
  return hipSuccess;
}

hipError_t acquire_matmul_f16_staging_workspace_locked(
    Context *const context, Queue *const queue, const uint64_t required,
    void **const workspace, uint64_t *const workspace_bytes) noexcept {
  if (context == nullptr || queue == nullptr || workspace == nullptr ||
      workspace_bytes == nullptr || required == 0U ||
      context->matmul_f16_staging_workspace == nullptr ||
      context->matmul_f16_staging_workspace_bytes < required) {
    return hipErrorInvalidValue;
  }
  if ((context->matmul_f16_staging_in_flight == 0U) !=
      (context->matmul_f16_staging_owner_queue == nullptr)) {
    context->poisoned.store(true);
    return hipErrorUnknown;
  }
  if (context->matmul_f16_staging_in_flight != 0U &&
      context->matmul_f16_staging_owner_queue != queue) {
    return hipErrorNotReady;
  }
  if (context->matmul_f16_staging_in_flight == UINT64_MAX) {
    return hipErrorOutOfMemory;
  }
  if (context->matmul_f16_staging_owner_queue == nullptr) {
    context->matmul_f16_staging_owner_queue = queue;
  }
  ++context->matmul_f16_staging_in_flight;
  *workspace = context->matmul_f16_staging_workspace;
  *workspace_bytes = context->matmul_f16_staging_workspace_bytes;
  return hipSuccess;
}

#if !defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
// Format-neutral compute stage for packed formats that expand their values to
// transient FP16.  Inputs use the runtime's row-major logical convention, so
// the rocBLAS column-major call computes C^T = W * A^T.  Scales remain the
// responsibility of the format-specific epilogue.
hipError_t launch_matmul_f16_staging_f32_gemm(
    Context *const context, const hipStream_t stream,
    const uint16_t *const activation_f16, const uint16_t *const weight_f16,
    float *const output_f32, const uint64_t m, const uint64_t k,
    const uint64_t n) noexcept {
  if (context == nullptr || activation_f16 == nullptr ||
      weight_f16 == nullptr || output_f32 == nullptr || m == 0U || k == 0U ||
      n == 0U ||
      m > static_cast<uint64_t>(std::numeric_limits<rocblas_int>::max()) ||
      k > static_cast<uint64_t>(std::numeric_limits<rocblas_int>::max()) ||
      n > static_cast<uint64_t>(std::numeric_limits<rocblas_int>::max()) ||
      context->matmul_rocblas_handle == nullptr) {
    return hipErrorInvalidValue;
  }

  std::lock_guard<std::mutex> blas_lock(context->matmul_blas_mutex);
  if (rocblas_set_stream(context->matmul_rocblas_handle, stream) !=
      rocblas_status_success) {
    return hipErrorInvalidHandle;
  }
  const float alpha = 1.0F;
  const float beta = 0.0F;
  const std::array<int64_t, 3> shape = {static_cast<int64_t>(m),
                                        static_cast<int64_t>(k),
                                        static_cast<int64_t>(n)};
  int32_t solution = 0;
  bool solution_is_cached = false;
  for (const auto &cached : context->matmul_f16_staging_rocblas_solutions) {
    if (cached[0] == shape[0] && cached[1] == shape[1] &&
        cached[2] == shape[2]) {
      solution = static_cast<int32_t>(cached[3]);
      solution_is_cached = true;
      break;
    }
  }
  if (!solution_is_cached) {
    // Query the exact contract and only choose a measured solution when it is
    // present for this shape.  ROCm 7.14 exposes 933 candidates for the
    // gfx1201 production shapes, where solution 136321 was selected by the
    // Phase 78 probe.  gfx1030 retains its measured -587 policy.  If either
    // preferred solution is absent (or the query cannot be represented by the
    // bounded local list), fail closed to rocBLAS' standard selector.
    rocblas_int solution_count = 0;
    rocblas_status query_status = rocblas_gemm_ex_get_solutions(
        context->matmul_rocblas_handle, rocblas_operation_transpose,
        rocblas_operation_none, static_cast<rocblas_int>(n),
        static_cast<rocblas_int>(m), static_cast<rocblas_int>(k), &alpha,
        weight_f16, rocblas_datatype_f16_r, static_cast<rocblas_int>(k),
        activation_f16, rocblas_datatype_f16_r, static_cast<rocblas_int>(k),
        &beta, output_f32, rocblas_datatype_f32_r, static_cast<rocblas_int>(n),
        output_f32, rocblas_datatype_f32_r, static_cast<rocblas_int>(n),
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0U, nullptr,
        &solution_count);
    constexpr rocblas_int kMaxQueriedSolutions = 1024;
    std::array<rocblas_int, static_cast<std::size_t>(kMaxQueriedSolutions)>
        queried_solutions{};
    bool solutions_loaded = false;
    if (query_status == rocblas_status_success && solution_count > 0 &&
        solution_count <= kMaxQueriedSolutions) {
      query_status = rocblas_gemm_ex_get_solutions(
          context->matmul_rocblas_handle, rocblas_operation_transpose,
          rocblas_operation_none, static_cast<rocblas_int>(n),
          static_cast<rocblas_int>(m), static_cast<rocblas_int>(k), &alpha,
          weight_f16, rocblas_datatype_f16_r, static_cast<rocblas_int>(k),
          activation_f16, rocblas_datatype_f16_r, static_cast<rocblas_int>(k),
          &beta, output_f32, rocblas_datatype_f32_r,
          static_cast<rocblas_int>(n), output_f32, rocblas_datatype_f32_r,
          static_cast<rocblas_int>(n), rocblas_datatype_f32_r,
          rocblas_gemm_algo_standard, 0U, queried_solutions.data(),
          &solution_count);
      solutions_loaded = query_status == rocblas_status_success;
    }
    const bool gfx1201 = matches_runtime_gcn_arch(
        context->device_properties.gcnArchName, "gfx1201");
    const int32_t preferred_solution =
        gfx1201 ? 136321 : (m <= 128U || k > n ? -587 : 0);
    if (solutions_loaded && preferred_solution != 0) {
      for (rocblas_int index = 0; index < solution_count; ++index) {
        if (queried_solutions[static_cast<std::size_t>(index)] ==
            preferred_solution) {
          solution = preferred_solution;
          break;
        }
      }
    }
    try {
      context->matmul_f16_staging_rocblas_solutions.push_back(
          {shape[0], shape[1], shape[2], static_cast<int64_t>(solution)});
    } catch (...) {
      // Allocation failure only disables the performance cache.  The exact
      // solution remains valid for this submission and can be queried again.
    }
  }
  const rocblas_gemm_algo algorithm = solution == 0
                                          ? rocblas_gemm_algo_standard
                                          : rocblas_gemm_algo_solution_index;
  const rocblas_status status =
      (rocblas_gemm_ex)(context->matmul_rocblas_handle,
                        rocblas_operation_transpose, rocblas_operation_none,
                        static_cast<rocblas_int>(n),
                        static_cast<rocblas_int>(m),
                        static_cast<rocblas_int>(k), &alpha, weight_f16,
                        rocblas_datatype_f16_r, static_cast<rocblas_int>(k),
                        activation_f16, rocblas_datatype_f16_r,
                        static_cast<rocblas_int>(k), &beta, output_f32,
                        rocblas_datatype_f32_r, static_cast<rocblas_int>(n),
                        output_f32, rocblas_datatype_f32_r,
                        static_cast<rocblas_int>(n), rocblas_datatype_f32_r,
                        algorithm, solution, 0U);
  return status == rocblas_status_success ? hipSuccess : hipErrorUnknown;
}
#endif

} // namespace

extern "C" sllm_status_t
sllm_matmul_prepare(const sllm_context_t *const raw_context,
                    const sllm_matmul_desc_t *const descriptor,
                    sllm_matmul_plan_t **const raw_plan,
                    sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_plan != nullptr) {
      *raw_plan = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_context == nullptr || raw_plan == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "matmul context or plan output is null");
    }
    const sllm_status_t prefix_status =
        sllm_matmul::validate_descriptor_prefix(descriptor, error_sink);
    if (prefix_status != SLLM_STATUS_OK) {
      return prefix_status;
    }
    const sllm_matmul_desc_t descriptor_copy = *descriptor;
    sllm_matmul::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_matmul::validate_and_copy_descriptor(&descriptor_copy, &metadata,
                                                  error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }

    std::unique_ptr<ElementwisePlan> candidate;
    uintptr_t token = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      Context *const context =
          lookup<Context>(raw_context, HandleKind::Context);
      Buffer *const activation =
          lookup<Buffer>(descriptor_copy.activation.buffer, HandleKind::Buffer);
      Buffer *const weight =
          lookup<Buffer>(descriptor_copy.weight.buffer, HandleKind::Buffer);
      Buffer *const output =
          lookup<Buffer>(descriptor_copy.output.buffer, HandleKind::Buffer);
      if (context == nullptr || activation == nullptr || weight == nullptr ||
          output == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "matmul context or buffer handle is stale or wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active ||
          activation->release_active || weight->release_active ||
          output->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "matmul cannot prepare while a context or buffer is releasing");
      }
      if (activation->context != context || weight->context != context ||
          output->context != context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH,
            "matmul buffers must belong to the supplied context and device");
      }
      if (metadata.activation.end_offset > activation->size_bytes ||
          metadata.weight.end_offset > weight->size_bytes ||
          metadata.output.end_offset > output->size_bytes) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "matmul tensor interval exceeds its backing buffer");
      }
      const auto overlaps_if_same =
          [](const Buffer *const left_buffer,
             const sllm_matmul::TensorMetadata &left,
             const Buffer *const right_buffer,
             const sllm_matmul::TensorMetadata &right) {
            return left_buffer == right_buffer &&
                   sllm_matmul::intervals_overlap(left, right);
          };
      if (overlaps_if_same(activation, metadata.activation, weight,
                           metadata.weight) ||
          overlaps_if_same(activation, metadata.activation, output,
                           metadata.output) ||
          overlaps_if_same(weight, metadata.weight, output, metadata.output)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_ALIAS_OVERLAP,
            "matmul tensor intervals overlap within one backing buffer");
      }
      if (!sllm_public_runtime::AccountingState::reserve_prepared_plan(
              context->accounting, activation->accounting, weight->accounting,
              output->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "matmul prepared-plan accounting is exhausted");
      }
      try {
        candidate = std::make_unique<ElementwisePlan>(context, activation,
                                                      weight, output, metadata);
        // The gfx1030 short-mixed provider is a BF16-only execution path.  In
        // particular, its M*N FP32 output workspace is not interchangeable
        // with the activation value/scale workspace required by the dynamic
        // FP8/NVFP4/MX quantizers.  Keeping the predicates exclusive also
        // prevents a quantized plan from being sized for short-mixed at
        // prepare time and then correctly dispatched to its quantized kernel
        // at execute time.
        const bool quantized_weight_path =
            metadata.fp8_outer || metadata.nvfp4 || metadata.nvfp4_w4a4 ||
            metadata.mxfp4_w4a4 || metadata.mxfp8_w8a8 || metadata.mxfp6_w6a6;
        bool short_mixed_shape =
            !quantized_weight_path &&
            ::sllm_matmul_kernel::phase49_gfx1030_short_mixed_shape(
                metadata.m, metadata.k, metadata.n);
#if !defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
        if (short_mixed_shape) {
          const char *const disable_short_mixed =
              std::getenv("SLLM_MATMUL_GFX1030_SHORT_MIXED");
          if (disable_short_mixed != nullptr &&
              std::strcmp(disable_short_mixed, "0") == 0) {
            short_mixed_shape = false;
          } else {
            const sllm_status_t mixed_device_status =
                select_context_device(context, error_sink);
            if (mixed_device_status != SLLM_STATUS_OK) {
              (void)sllm_public_runtime::AccountingState::release_prepared_plan(
                  context->accounting, activation->accounting,
                  weight->accounting, output->accounting);
              return mixed_device_status;
            }
            hipDeviceProp_t mixed_properties = {};
            const sllm_status_t mixed_property_status =
                get_context_device_properties(context, &mixed_properties,
                                              error_sink);
            if (mixed_property_status != SLLM_STATUS_OK) {
              (void)sllm_public_runtime::AccountingState::release_prepared_plan(
                  context->accounting, activation->accounting,
                  weight->accounting, output->accounting);
              return mixed_property_status;
            }
            short_mixed_shape = matches_runtime_gcn_arch(
                mixed_properties.gcnArchName, "gfx1030");
          }
        }
#else
        short_mixed_shape = false;
#endif
        const bool lowp_path = matmul_lowp_path(metadata);
        bool split4_plan = false;
        if (metadata.fp8_outer || lowp_path || short_mixed_shape) {
          const sllm_status_t device_status =
              select_context_device(context, error_sink);
          if (device_status != SLLM_STATUS_OK) {
            (void)sllm_public_runtime::AccountingState::release_prepared_plan(
                context->accounting, activation->accounting, weight->accounting,
                output->accounting);
            return device_status;
          }
          if (metadata.fp8_outer) {
            hipDeviceProp_t properties = {};
            const sllm_status_t property_status =
                get_context_device_properties(context, &properties, error_sink);
            if (property_status != SLLM_STATUS_OK) {
              (void)sllm_public_runtime::AccountingState::release_prepared_plan(
                  context->accounting, activation->accounting,
                  weight->accounting, output->accounting);
              return property_status;
            }
            const bool fnuz =
                metadata.fp8_dtype == SLLM_TENSOR_DTYPE_F8_E4M3_FNUZ;
            const sllm_matmul_kernel::KernelVariant kernel_variant =
                ::sllm_matmul_kernel::select_fp8_outer_variant(
                    metadata.m, metadata.k, metadata.n, properties.gcnArchName,
                    fnuz);
            candidate->matmul_kernel_variant = kernel_variant;

            // The exact-E4M3FN gfx1030 software route publishes the same
            // immutable provider semantics as the typed block-scaled paths.
            // Native gfx1201/gfx942 and FNUZ plans retain their established
            // hipBLASLt/emulation contracts and never select ID66.
            if (!fnuz &&
                matches_runtime_gcn_arch(properties.gcnArchName, "gfx1030")) {
              const sllm_lowp::ProviderRequest provider_request =
                  sllm_lowp::make_provider_request(
                      sllm_lowp::MatmulFormat::Fp8OuterE4M3W8A8,
                      sllm_lowp::ExactTarget::Gfx1030, metadata.m, metadata.n,
                      metadata.k);
              const sllm_lowp::PreparedProviderPlan provider_plan =
                  sllm_lowp::prepare_provider_plan(provider_request);
              const auto concrete_provider =
                  provider_plan.supported()
                      ? matmul_lowp_concrete_provider_plan(provider_plan,
                                                           kernel_variant)
                      : std::nullopt;
              if (!concrete_provider.has_value()) {
                (void)
                    sllm_public_runtime::AccountingState::release_prepared_plan(
                        context->accounting, activation->accounting,
                        weight->accounting, output->accounting);
                return sllm_public_runtime::write_error(
                    error_sink, SLLM_STATUS_UNSUPPORTED,
                    "FP8 outer software kernel does not match the prepared "
                    "provider contract");
              }
              candidate->matmul_provider_plan.emplace(*concrete_provider);
            }
          }
          if (lowp_path) {
            hipDeviceProp_t properties = {};
            const sllm_status_t property_status =
                get_context_device_properties(context, &properties, error_sink);
            if (property_status != SLLM_STATUS_OK) {
              (void)sllm_public_runtime::AccountingState::release_prepared_plan(
                  context->accounting, activation->accounting,
                  weight->accounting, output->accounting);
              return property_status;
            }
            sllm_lowp::MatmulFormat format{};
            if (!matmul_lowp_format(metadata, &format)) {
              (void)sllm_public_runtime::AccountingState::release_prepared_plan(
                  context->accounting, activation->accounting,
                  weight->accounting, output->accounting);
              return sllm_public_runtime::write_error(
                  error_sink, SLLM_STATUS_UNSUPPORTED,
                  "matmul low-precision format is not supported");
            }
            const sllm_lowp::ExactTarget target =
                matmul_lowp_target(properties.gcnArchName);
            const sllm_lowp::ProviderRequest provider_request =
                sllm_lowp::make_provider_request(format, target, metadata.m,
                                                 metadata.n, metadata.k);
            const sllm_lowp::PreparedProviderPlan provider_plan =
                sllm_lowp::prepare_provider_plan(provider_request);
            if (!provider_plan.supported()) {
              (void)sllm_public_runtime::AccountingState::release_prepared_plan(
                  context->accounting, activation->accounting,
                  weight->accounting, output->accounting);
              return sllm_public_runtime::write_error(
                  error_sink, SLLM_STATUS_UNSUPPORTED,
                  "matmul low-precision provider rejected target, layout, "
                  "shape, or numerics");
            }
            // Freeze both semantic provider and concrete kernel variant here.
            // Execute and dispatch-info must not observe later environment or
            // target changes by running a selector again.
            const sllm_matmul_kernel::KernelVariant kernel_variant =
                matmul_lowp_kernel_variant(metadata, provider_plan,
                                           properties.gcnArchName);
            const auto concrete_provider = matmul_lowp_concrete_provider_plan(
                provider_plan, kernel_variant);
            if (!concrete_provider.has_value()) {
              (void)sllm_public_runtime::AccountingState::release_prepared_plan(
                  context->accounting, activation->accounting,
                  weight->accounting, output->accounting);
              return sllm_public_runtime::write_error(
                  error_sink, SLLM_STATUS_UNSUPPORTED,
                  "matmul low-precision concrete kernel does not match the "
                  "prepared provider contract");
            }
            candidate->matmul_provider_plan.emplace(*concrete_provider);
            candidate->matmul_kernel_variant = kernel_variant;
            split4_plan =
                metadata.nvfp4_w4a4 &&
                ((kernel_variant == ::sllm_matmul_kernel::KernelVariant::
                                        Nvfp4W4A4PrefillDp4a64x64 &&
                  matches_runtime_gcn_arch(properties.gcnArchName, "gfx1030") &&
                  ::sllm_matmul_kernel::phase78_gfx1030_nvfp4_w4a4_split4_shape(
                      metadata.m, metadata.k, metadata.n)) ||
                 (kernel_variant == ::sllm_matmul_kernel::KernelVariant::
                                        Nvfp4W4A4PrefillGfx1201Wmma128x64 &&
                  matches_runtime_gcn_arch(properties.gcnArchName, "gfx1201") &&
                  ::sllm_matmul_kernel::phase78_gfx1201_nvfp4_w4a4_split4_shape(
                      metadata.m, metadata.k, metadata.n)));
          }
          const bool fp8_f16_staging =
              metadata.fp8_outer &&
              (candidate->matmul_kernel_variant ==
                   ::sllm_matmul_kernel::KernelVariant::
                       Fp8OuterPrefillGfx1030F16Staging ||
               candidate->matmul_kernel_variant ==
                   ::sllm_matmul_kernel::KernelVariant::
                       Fp8OuterPrefillGfx1030F16TileStaging);
          const bool fp8_f16_tile_staging =
              metadata.fp8_outer &&
              candidate->matmul_kernel_variant ==
                  ::sllm_matmul_kernel::KernelVariant::
                      Fp8OuterPrefillGfx1030F16TileStaging;
          const bool nvfp4_f16_staging =
              metadata.nvfp4_w4a4 && candidate->matmul_kernel_variant ==
                                         ::sllm_matmul_kernel::KernelVariant::
                                             Nvfp4W4A4PrefillGfx1201F16Staging;
          const bool nvfp4_fp8_staging =
              metadata.nvfp4_w4a4 && candidate->matmul_kernel_variant ==
                                         ::sllm_matmul_kernel::KernelVariant::
                                             Nvfp4W4A4PrefillGfx1201Fp8Staging;
          if (fp8_f16_staging || nvfp4_f16_staging || nvfp4_fp8_staging) {
            ::sllm_matmul_kernel::F16StagingWorkspaceLayout
                f16_staging_layout{};
            ::sllm_matmul_kernel::Nvfp4Fp8StagingWorkspaceLayout
                fp8_staging_layout{};
            uint64_t staged_total_bytes = 0U;
            bool layout_valid = false;
            if (fp8_f16_staging) {
              layout_valid =
                  fp8_f16_tile_staging
                      ? ::sllm_matmul_kernel::
                            fp8_outer_prefill_gfx1030_f16_tile_staging_workspace(
                                metadata.m, metadata.k, metadata.n,
                                &f16_staging_layout)
                      : ::sllm_matmul_kernel::
                            fp8_outer_prefill_gfx1030_f16_staging_workspace(
                                metadata.m, metadata.k, metadata.n,
                                &f16_staging_layout);
              staged_total_bytes = f16_staging_layout.total_bytes;
            } else if (nvfp4_f16_staging) {
              layout_valid =
                  ::sllm_matmul_kernel::
                      phase78_gfx1201_nvfp4_w4a4_f16_staging_shape(
                          metadata.m, metadata.k, metadata.n) &&
                  ::sllm_matmul_kernel::f16_staging_workspace_layout(
                      metadata.m, metadata.k, metadata.n, &f16_staging_layout);
              staged_total_bytes = f16_staging_layout.total_bytes;
            } else {
              layout_valid =
                  ::sllm_matmul_kernel::nvfp4_w4a4_fp8_staging_workspace(
                      metadata.m, metadata.k, metadata.n, &fp8_staging_layout);
              staged_total_bytes = fp8_staging_layout.total_bytes;
            }
            uint64_t reservation_bytes = 0U;
            if (!layout_valid || staged_total_bytes == 0U ||
                !::sllm_matmul_kernel::qwen38_f16_staging_workspace_reservation(
                    metadata.m, &reservation_bytes) ||
                reservation_bytes < staged_total_bytes ||
                reservation_bytes >
                    static_cast<uint64_t>(
                        std::numeric_limits<std::size_t>::max())) {
              (void)sllm_public_runtime::AccountingState::release_prepared_plan(
                  context->accounting, activation->accounting,
                  weight->accounting, output->accounting);
              return sllm_public_runtime::write_error(
                  error_sink, SLLM_STATUS_UNSUPPORTED,
                  "FP16 staging matmul workspace size overflowed");
            }
            const hipError_t staging_status =
                reserve_matmul_f16_staging_workspace_locked(context,
                                                            reservation_bytes);
            if (staging_status != hipSuccess) {
              (void)sllm_public_runtime::AccountingState::release_prepared_plan(
                  context->accounting, activation->accounting,
                  weight->accounting, output->accounting);
              if (staging_status == hipErrorNotReady) {
                return sllm_public_runtime::write_error(
                    error_sink, SLLM_STATUS_PUBLIC_NOT_READY,
                    "FP16 staging matmul workspace is owned by another queue");
              }
              return hip_failure(error_sink, staging_status,
                                 "hipMalloc FP16 staging matmul workspace");
            }
            candidate->matmul_context_workspace_bytes = reservation_bytes;
          }
          uint64_t scale_offset = 0U;
          if (short_mixed_shape) {
            uint64_t workspace_bytes = 0U;
            const bool workspace_valid =
                ::sllm_matmul_kernel::phase49_gfx1030_mixed_workspace_bytes(
                    metadata.m, metadata.n, &workspace_bytes);
            if (!workspace_valid ||
                workspace_bytes >
                    static_cast<uint64_t>(
                        std::numeric_limits<std::size_t>::max()) ||
                metadata.m >
                    static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
                metadata.k >
                    static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
                metadata.n >
                    static_cast<uint64_t>(std::numeric_limits<int>::max())) {
              (void)sllm_public_runtime::AccountingState::release_prepared_plan(
                  context->accounting, activation->accounting,
                  weight->accounting, output->accounting);
              return sllm_public_runtime::write_error(
                  error_sink, SLLM_STATUS_UNSUPPORTED,
                  "mixed matmul workspace or dispatch size overflowed");
            }
            candidate->matmul_workspace_bytes = workspace_bytes;
          } else if (lowp_path) {
            if (!matmul_lowp_workspace_bytes(
                    *candidate->matmul_provider_plan,
                    &candidate->matmul_workspace_bytes)) {
              (void)sllm_public_runtime::AccountingState::release_prepared_plan(
                  context->accounting, activation->accounting,
                  weight->accounting, output->accounting);
              return sllm_public_runtime::write_error(
                  error_sink, SLLM_STATUS_UNSUPPORTED,
                  "low-precision matmul workspace size overflowed");
            }
            if (split4_plan) {
              uint64_t partial_offset = 0U;
              uint64_t split4_workspace_bytes = 0U;
              if (!::sllm_matmul_kernel::phase78_nvfp4_w4a4_split4_workspace(
                      metadata.m, metadata.k, metadata.n, &partial_offset,
                      &split4_workspace_bytes) ||
                  candidate->matmul_workspace_bytes != partial_offset) {
                (void)
                    sllm_public_runtime::AccountingState::release_prepared_plan(
                        context->accounting, activation->accounting,
                        weight->accounting, output->accounting);
                return sllm_public_runtime::write_error(
                    error_sink, SLLM_STATUS_UNSUPPORTED,
                    "NVFP4 split4 workspace layout is incompatible");
              }
              candidate->matmul_workspace_bytes = split4_workspace_bytes;
            }
          } else {
            const uint64_t value_bytes =
                metadata.mxfp6_w6a6
                    ? metadata.m * (metadata.k * UINT64_C(3) / UINT64_C(4))
                : metadata.mxfp8_w8a8 ? metadata.m * metadata.k
                : metadata.nvfp4_w4a4 || metadata.mxfp4_w4a4
                    ? metadata.m * ((metadata.k + UINT64_C(1)) / UINT64_C(2))
                    : metadata.m * metadata.k;
            scale_offset = metadata.nvfp4_w4a4 || metadata.mxfp4_w4a4 ||
                                   metadata.mxfp8_w8a8 || metadata.mxfp6_w6a6
                               ? value_bytes
                               : (value_bytes + UINT64_C(3)) & ~UINT64_C(3);
            candidate->matmul_workspace_bytes =
                scale_offset +
                (metadata.nvfp4_w4a4 || metadata.mxfp4_w4a4 ||
                         metadata.mxfp8_w8a8 || metadata.mxfp6_w6a6
                     ? metadata.m *
                           ((metadata.k +
                             ((metadata.mxfp4_w4a4 || metadata.mxfp8_w8a8 ||
                               metadata.mxfp6_w6a6)
                                  ? UINT64_C(31)
                                  : UINT64_C(15))) /
                            ((metadata.mxfp4_w4a4 || metadata.mxfp8_w8a8 ||
                              metadata.mxfp6_w6a6)
                                 ? UINT64_C(32)
                                 : UINT64_C(16)))
                     : metadata.m * UINT64_C(4));
          }
          const hipError_t workspace_status =
              candidate->matmul_workspace_bytes == 0U
                  ? hipSuccess
                  : hipMalloc(&candidate->matmul_workspace,
                              static_cast<std::size_t>(
                                  candidate->matmul_workspace_bytes));
          if (workspace_status != hipSuccess) {
            (void)sllm_public_runtime::AccountingState::release_prepared_plan(
                context->accounting, activation->accounting, weight->accounting,
                output->accounting);
            return hip_failure(
                error_sink, workspace_status,
                "hipMalloc low-precision matmul prepared workspace");
          }
#if !defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
          if (metadata.nvfp4_w4a4 &&
              candidate->matmul_kernel_variant ==
                  ::sllm_matmul_kernel::KernelVariant::
                      Nvfp4W4A4PrefillGfx1201Fp8Staging) {
            ::sllm_matmul_kernel::Nvfp4Fp8StagingWorkspaceLayout
                staging_layout{};
            if (!::sllm_matmul_kernel::nvfp4_w4a4_fp8_staging_workspace(
                    metadata.m, metadata.k, metadata.n, &staging_layout) ||
                context->matmul_f16_staging_workspace == nullptr ||
                context->matmul_f16_staging_workspace_bytes <
                    staging_layout.total_bytes) {
              (void)sllm_public_runtime::AccountingState::release_prepared_plan(
                  context->accounting, activation->accounting,
                  weight->accounting, output->accounting);
              return sllm_public_runtime::write_error(
                  error_sink, SLLM_STATUS_INTERNAL_ERROR,
                  "NVFP4 FP8 staging workspace is unavailable");
            }
            auto *const staging_bytes =
                static_cast<uint8_t *>(context->matmul_f16_staging_workspace);
            auto *const scale_product = reinterpret_cast<float *>(
                staging_bytes + staging_layout.scale_product_offset);
            std::lock_guard<std::mutex> lt_lock(context->matmul_lt_mutex);
            const hipError_t lt_status = create_fp8_lt_plan(
                context->matmul_lt_handle, &candidate->fp8_lt_plan, nullptr,
                nullptr, SLLM_TENSOR_DTYPE_F8_E4M3_FN, metadata.m, metadata.k,
                metadata.n, context->device_properties.gcnArchName, false, true,
                scale_product);
            if (lt_status != hipSuccess) {
              (void)sllm_public_runtime::AccountingState::release_prepared_plan(
                  context->accounting, activation->accounting,
                  weight->accounting, output->accounting);
              return hip_failure(error_sink, lt_status,
                                 "prepare NVFP4 FP8 hipBLASLt solution");
            }
          } else if (metadata.fp8_outer &&
                     candidate->matmul_kernel_variant ==
                         ::sllm_matmul_kernel::KernelVariant::Fp8Native) {
            hipDeviceProp_t properties = {};
            const sllm_status_t property_status =
                get_context_device_properties(context, &properties, error_sink);
            if (property_status != SLLM_STATUS_OK) {
              (void)sllm_public_runtime::AccountingState::release_prepared_plan(
                  context->accounting, activation->accounting,
                  weight->accounting, output->accounting);
              return property_status;
            }
            auto *const activation_scales = reinterpret_cast<float *>(
                static_cast<uint8_t *>(candidate->matmul_workspace) +
                scale_offset);
            const auto *const weight_scales = reinterpret_cast<const float *>(
                static_cast<const uint8_t *>(weight->device_pointer) +
                metadata.weight_scale_offset);
            std::lock_guard<std::mutex> lt_lock(context->matmul_lt_mutex);
            const hipError_t lt_status = create_fp8_lt_plan(
                context->matmul_lt_handle, &candidate->fp8_lt_plan,
                activation_scales, weight_scales, metadata.fp8_dtype,
                metadata.m, metadata.k, metadata.n, properties.gcnArchName,
                true, false, nullptr);
            if (lt_status != hipSuccess) {
              (void)sllm_public_runtime::AccountingState::release_prepared_plan(
                  context->accounting, activation->accounting,
                  weight->accounting, output->accounting);
              return hip_failure(error_sink, lt_status,
                                 "prepare FP8 hipBLASLt solution");
            }
          }
#endif
        }
        token = register_handle(candidate.get(), HandleKind::MatmulPlan);
      } catch (...) {
        (void)sllm_public_runtime::AccountingState::release_prepared_plan(
            context->accounting, activation->accounting, weight->accounting,
            output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "matmul prepared-plan allocation or registration failed");
      }
      if (token == 0U) {
        (void)sllm_public_runtime::AccountingState::release_prepared_plan(
            context->accounting, activation->accounting, weight->accounting,
            output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "matmul prepared-plan handle allocation failed");
      }
    }
    *raw_plan = reinterpret_cast<sllm_matmul_plan_t *>(token);
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in matmul prepare");
  }
}

extern "C" sllm_status_t
sllm_matmul_plan_release(sllm_matmul_plan_t **const raw_plan,
                         sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_plan == nullptr || *raw_plan == nullptr) {
      return sllm_public_runtime::write_error(error_sink,
                                              SLLM_STATUS_INVALID_ARGUMENT,
                                              "matmul plan handle is null");
    }
    ElementwisePlan *plan = nullptr;
    bool quarantine_plan = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<ElementwisePlan>(*raw_plan, HandleKind::MatmulPlan);
      if (plan == nullptr || !plan->matmul) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "matmul plan handle is stale or has the wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->release_active || plan->in_flight || plan->graph_pins != 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "matmul plan is releasing, in flight, or owned by a graph span");
      }
      plan->release_active = true;
      bool workspace_released = true;
      if (plan->matmul_workspace != nullptr) {
        const sllm_status_t device_status =
            select_context_device(plan->context, error_sink);
        const hipError_t workspace_status =
            device_status == SLLM_STATUS_OK ? hipFree(plan->matmul_workspace)
                                            : hipErrorInvalidValue;
        if (workspace_status == hipSuccess) {
          plan->matmul_workspace = nullptr;
          plan->matmul_workspace_bytes = 0U;
        } else {
          workspace_released = false;
        }
      }
#if !defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
      destroy_fp8_lt_plan(plan->fp8_lt_plan);
      plan->fp8_lt_plan = nullptr;
#endif
      const bool accounting_released =
          workspace_released &&
          !sllm_public_runtime::FaultInjector::consume(
              sllm_public_runtime::FaultPoint::AccountingFailure) &&
          sllm_public_runtime::AccountingState::release_prepared_plan(
              plan->context->accounting, plan->input0->accounting,
              plan->input1->accounting, plan->output->accounting);
      unregister_handle(*raw_plan);
      *raw_plan = nullptr;
      if (!accounting_released) {
        plan->context->poisoned.store(true);
        quarantine_plan = true;
      }
    }
    if (quarantine_plan) {
      poison_owner.retain(plan);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "matmul workspace or prepared-plan accounting release failed; plan "
          "quarantined");
    }
    delete plan;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in matmul plan release");
  }
}

extern "C" sllm_status_t
sllm_matmul_execute(const sllm_matmul_plan_t *const raw_plan,
                    const sllm_queue_t *const raw_queue,
                    sllm_completion_t **const completion_output,
                    sllm_matmul_dispatch_info_t *const dispatch_info,
                    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_plan == nullptr || raw_queue == nullptr ||
        completion_output == nullptr || dispatch_info == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "matmul execute plan, queue, completion, or dispatch info is null");
    }
    const sllm_status_t dispatch_status =
        validate_matmul_dispatch_info(dispatch_info, error_sink);
    if (dispatch_status != SLLM_STATUS_OK) {
      return dispatch_status;
    }

    ElementwisePlan *plan = nullptr;
    Queue *queue = nullptr;
    uint64_t dispatch_id = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<ElementwisePlan>(raw_plan, HandleKind::MatmulPlan);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      if (plan == nullptr || queue == nullptr || !plan->matmul) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "matmul execute plan or queue handle is stale or wrong kind");
      }
      if (plan->context != queue->context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            "matmul plan and queue belong to different contexts");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->context->poisoned.load() || plan->context->release_active ||
          queue->release_active || plan->input0->release_active ||
          plan->input1->release_active || plan->output->release_active ||
          plan->release_active || plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "matmul execute owner graph is releasing or already in flight");
      }
      const auto &metadata = plan->matmul_metadata;
      if (metadata.m == 0U || metadata.k == 0U || metadata.n == 0U ||
          metadata.m > SLLM_HIP_MATMUL_MAX_M ||
          metadata.k > SLLM_HIP_MATMUL_MAX_K ||
          metadata.n > SLLM_HIP_MATMUL_MAX_N ||
          metadata.output_elements == 0U ||
          metadata.output_elements > SLLM_HIP_MATMUL_MAX_OUTPUT_ELEMENTS ||
          metadata.output_elements > UINT32_MAX) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_UNSUPPORTED,
            "matmul shape exceeds the baseline kernel launch contract");
      }
      if (!sllm_public_runtime::AccountingState::reserve_rmsnorm_submission(
              plan->context->accounting, queue->accounting,
              plan->input0->accounting, plan->input1->accounting,
              plan->output->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "matmul submission accounting is exhausted");
      }
      plan->in_flight = true;
      dispatch_id = plan->context->next_dispatch_id;
      if (dispatch_id == 0U) {
        plan->in_flight = false;
        (void)sllm_public_runtime::AccountingState::rollback_rmsnorm_submission(
            plan->context->accounting, queue->accounting,
            plan->input0->accounting, plan->input1->accounting,
            plan->output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "matmul context dispatch id is exhausted");
      }
      plan->context->next_dispatch_id =
          dispatch_id == std::numeric_limits<uint64_t>::max()
              ? 0U
              : dispatch_id + 1U;
      *completion_output = nullptr;
    }

    std::unique_ptr<Completion> candidate;
    NativeEventGuard event_guard;
    void *matmul_f16_staging_workspace = nullptr;
    uint64_t matmul_f16_staging_workspace_bytes = 0U;
    ElementwiseExecuteScopeGuard execute_guard(plan, queue, &candidate,
                                               &event_guard, error_sink);
    const sllm_status_t device_status =
        select_context_device(plan->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      if (!rollback_reserved_elementwise_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return device_status;
    }
    hipDeviceProp_t properties = {};
    const sllm_status_t property_status =
        get_context_device_properties(plan->context, &properties, error_sink);
    if (property_status != SLLM_STATUS_OK) {
      if (!rollback_reserved_elementwise_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return property_status;
    }
    char arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME] = {};
    if (!copy_property(arch_name, sizeof(arch_name), properties.gcnArchName,
                       sizeof(properties.gcnArchName))) {
      if (!rollback_reserved_elementwise_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
          "HIP gcnArchName is not NUL terminated or is too long");
    }
#if defined(SLLM_HIP_COMPILE_TARGET)
    if (!matches_runtime_gcn_arch(arch_name, SLLM_HIP_COMPILE_TARGET)) {
      if (!rollback_reserved_elementwise_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
          "matmul device target does not match the compiled exact target");
    }
#endif

    try {
      candidate = std::make_unique<Completion>(
          plan->context, queue, plan->input0, 0U, false, std::vector<uint8_t>{},
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, plan, plan->input0, plan->input1, plan->output);
      execute_guard.candidate_allocated();
    } catch (...) {
      if (!rollback_reserved_elementwise_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "matmul completion allocation failed before enqueue");
    }

    if (plan->matmul_context_workspace_bytes != 0U) {
      hipError_t staging_status = hipSuccess;
      {
        std::lock_guard<std::mutex> accounting_lock(
            plan->context->accounting_mutex);
        staging_status = acquire_matmul_f16_staging_workspace_locked(
            plan->context, queue, plan->matmul_context_workspace_bytes,
            &matmul_f16_staging_workspace, &matmul_f16_staging_workspace_bytes);
        if (staging_status == hipSuccess) {
          candidate->matmul_f16_staging_workspace_acquired = true;
        }
      }
      if (staging_status != hipSuccess) {
        if (!rollback_reserved_elementwise_submission(plan, queue,
                                                      error_sink)) {
          execute_guard.disarm();
          return SLLM_STATUS_INTERNAL_ERROR;
        }
        execute_guard.disarm();
        if (staging_status == hipErrorNotReady) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_PUBLIC_NOT_READY,
              "FP16 staging matmul workspace is owned by another queue");
        }
        return hip_failure(error_sink, staging_status,
                           "acquire FP16 staging matmul workspace");
      }
    }

    if (queue_profiles_completions(queue)) {
      hipEvent_t native_event = nullptr;
      const hipError_t event_status =
          hipEventCreateWithFlags(&native_event, 0U);
      if (event_status != hipSuccess) {
        const bool rollback_succeeded =
            candidate->matmul_f16_staging_workspace_acquired
                ? rollback_submission_references(candidate.get())
                : rollback_reserved_elementwise_submission(plan, queue,
                                                           error_sink);
        if (!rollback_succeeded) {
          execute_guard.disarm();
          return SLLM_STATUS_INTERNAL_ERROR;
        }
        execute_guard.disarm();
        return hip_failure(error_sink, event_status, "hipEventCreateWithFlags");
      }
      event_guard.adopt(plan->context, native_event);
      candidate->event = native_event;
      hipEvent_t timing_start_event = nullptr;
      const hipError_t timing_event_status =
          hipEventCreateWithFlags(&timing_start_event, 0U);
      if (timing_event_status != hipSuccess) {
        execute_guard.disarm();
        return rollback_unpublished_submission(
            candidate, event_guard,
            "matmul timing event creation failed before enqueue", error_sink);
      }
      candidate->timing_start_event = timing_start_event;
    }

    uintptr_t token = 0U;
    try {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      token = register_handle(candidate.get(), HandleKind::Completion);
    } catch (...) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "matmul completion registry allocation failed before enqueue",
          error_sink);
    }
    if (token == 0U) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "matmul completion handle token allocation failed", error_sink);
    }
    if (candidate->event != nullptr) {
      event_guard.release();
    }
    execute_guard.completion_registered(token);

    if (candidate->timing_start_event != nullptr) {
      const hipError_t timing_record_status =
          hipEventRecord(candidate->timing_start_event, queue->stream);
      if (timing_record_status != hipSuccess) {
        execute_guard.disarm();
        return cleanup_failed_submission(candidate, token, timing_record_status,
                                         "hipEventRecord timing start", queue,
                                         error_sink);
      }
    }
    const auto byte_pointer = [](Buffer *const buffer,
                                 const uint64_t offset) -> void * {
      return static_cast<char *>(buffer->device_pointer) +
             static_cast<std::size_t>(offset);
    };
    const uint16_t *const activation =
        static_cast<const uint16_t *>(byte_pointer(
            plan->input0, plan->matmul_metadata.activation.byte_offset));
    const void *const weight_storage =
        byte_pointer(plan->input1, plan->matmul_metadata.weight.byte_offset);
    const uint16_t *const weight =
        static_cast<const uint16_t *>(weight_storage);
    uint16_t *const output = static_cast<uint16_t *>(
        byte_pointer(plan->output, plan->matmul_metadata.output.byte_offset));
    // Typed low-precision and outer-FP8 plans carry the concrete kernel chosen
    // during prepare. Execute and dispatch reporting must not observe an
    // environment change after the plan was prepared.
    const auto variant =
        plan->matmul_metadata.fp8_outer ||
                plan->matmul_provider_plan.has_value()
            ? plan->matmul_kernel_variant
            : ::sllm_matmul_kernel::select_variant(
                  plan->matmul_metadata.m, plan->matmul_metadata.k,
                  plan->matmul_metadata.n, arch_name);
    hipError_t launch_status = hipSuccess;
    if (plan->matmul_metadata.mxfp8_w8a8) {
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
      launch_status = hipErrorInvalidValue;
#else
      if (!matches_runtime_gcn_arch(arch_name, "gfx1201") &&
          !matches_runtime_gcn_arch(arch_name, "gfx1030")) {
        launch_status = hipErrorNotSupported;
      } else {
        const uint64_t activation_value_bytes =
            plan->matmul_metadata.m * plan->matmul_metadata.k;
        const uint64_t activation_scale_bytes =
            plan->matmul_metadata.m * plan->matmul_metadata.k / UINT64_C(32);
        if (plan->matmul_workspace == nullptr ||
            plan->matmul_workspace_bytes <
                activation_value_bytes + activation_scale_bytes) {
          launch_status = hipErrorOutOfMemory;
        } else {
          auto *const quantized =
              static_cast<uint8_t *>(plan->matmul_workspace);
          auto *const activation_scales = quantized + activation_value_bytes;
          const auto *const fp8_weight =
              static_cast<const uint8_t *>(weight_storage);
          const auto *const weight_scales =
              fp8_weight + plan->matmul_metadata.weight_value_bytes;
          launch_status = ::sllm_matmul_kernel::launch_mxfp8_quantize(
              activation, quantized, activation_scales, plan->matmul_metadata.m,
              plan->matmul_metadata.k, queue->stream);
          if (launch_status == hipSuccess) {
            launch_status = ::sllm_matmul_kernel::launch_mxfp8_w8a8(
                quantized, activation_scales, fp8_weight, weight_scales, output,
                plan->matmul_metadata.m, plan->matmul_metadata.k,
                plan->matmul_metadata.n, variant, queue->stream);
          }
        }
      }
#endif
    } else if (plan->matmul_metadata.mxfp6_w6a6) {
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
      launch_status = hipErrorInvalidValue;
#else
      if (!matches_runtime_gcn_arch(arch_name, "gfx1201") &&
          !matches_runtime_gcn_arch(arch_name, "gfx1030")) {
        launch_status = hipErrorNotSupported;
      } else {
        const uint64_t activation_value_bytes =
            plan->matmul_metadata.m *
            (plan->matmul_metadata.k * UINT64_C(3) / UINT64_C(4));
        const uint64_t activation_scale_bytes =
            plan->matmul_metadata.m * plan->matmul_metadata.k / UINT64_C(32);
        if (plan->matmul_workspace == nullptr ||
            plan->matmul_workspace_bytes <
                activation_value_bytes + activation_scale_bytes) {
          launch_status = hipErrorOutOfMemory;
        } else {
          auto *const packed = static_cast<uint8_t *>(plan->matmul_workspace);
          auto *const activation_scales = packed + activation_value_bytes;
          const auto *const packed_weight =
              static_cast<const uint8_t *>(weight_storage);
          const auto *const weight_scales =
              packed_weight + plan->matmul_metadata.weight_value_bytes;
          launch_status = ::sllm_matmul_kernel::launch_mxfp6_quantize(
              activation, packed, activation_scales, plan->matmul_metadata.m,
              plan->matmul_metadata.k, queue->stream);
          if (launch_status == hipSuccess) {
            launch_status = ::sllm_matmul_kernel::launch_mxfp6_w6a6(
                packed, activation_scales, packed_weight, weight_scales, output,
                plan->matmul_metadata.m, plan->matmul_metadata.k,
                plan->matmul_metadata.n, variant, queue->stream);
          }
        }
      }
#endif
    } else if (plan->matmul_metadata.mxfp4_w4a4) {
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
      launch_status = hipErrorInvalidValue;
#else
      if (!matches_runtime_gcn_arch(arch_name, "gfx1201") &&
          !matches_runtime_gcn_arch(arch_name, "gfx1030")) {
        launch_status = hipErrorNotSupported;
      } else {
        const uint64_t packed_activation_bytes =
            plan->matmul_metadata.m *
            ((plan->matmul_metadata.k + UINT64_C(1)) / UINT64_C(2));
        const uint64_t activation_scale_bytes =
            plan->matmul_metadata.m *
            ((plan->matmul_metadata.k + UINT64_C(31)) / UINT64_C(32));
        if (plan->matmul_workspace == nullptr ||
            plan->matmul_workspace_bytes <
                packed_activation_bytes + activation_scale_bytes) {
          launch_status = hipErrorOutOfMemory;
        } else {
          auto *const packed_activation =
              static_cast<uint8_t *>(plan->matmul_workspace);
          auto *const activation_block_scales =
              packed_activation + packed_activation_bytes;
          const auto *const packed_weight =
              static_cast<const uint8_t *>(weight_storage);
          const auto *const weight_block_scales =
              packed_weight + plan->matmul_metadata.weight_value_bytes;
          launch_status = ::sllm_matmul_kernel::launch_mxfp4_quantize(
              activation, packed_activation, activation_block_scales,
              plan->matmul_metadata.m, plan->matmul_metadata.k, queue->stream);
          if (launch_status == hipSuccess) {
            launch_status = ::sllm_matmul_kernel::launch_mxfp4_w4a4(
                packed_activation, activation_block_scales, packed_weight,
                weight_block_scales, output, plan->matmul_metadata.m,
                plan->matmul_metadata.k, plan->matmul_metadata.n, variant,
                queue->stream);
          }
        }
      }
#endif
    } else if (plan->matmul_metadata.nvfp4_w4a4) {
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
      // Production kernels are absent from the host-contract binary.  Permit
      // only ID73's ordered quantize+compute route to publish a completion so
      // selector identity, two-dispatch accounting, and async ownership can
      // be tested without claiming a GPU numerical result.
      launch_status = variant == ::sllm_matmul_kernel::KernelVariant::
                                     Nvfp4W4A4DecodeActivationShared
                          ? hipSuccess
                          : hipErrorInvalidValue;
#else
      if (!matches_runtime_gcn_arch(arch_name, "gfx1201") &&
          !matches_runtime_gcn_arch(arch_name, "gfx1030")) {
        launch_status = hipErrorNotSupported;
      } else {
        const uint64_t packed_activation_bytes =
            plan->matmul_metadata.m *
            ((plan->matmul_metadata.k + UINT64_C(1)) / UINT64_C(2));
        const uint64_t activation_scale_bytes =
            plan->matmul_metadata.m *
            ((plan->matmul_metadata.k + UINT64_C(15)) / UINT64_C(16));
        if (plan->matmul_workspace == nullptr ||
            plan->matmul_workspace_bytes <
                packed_activation_bytes + activation_scale_bytes) {
          launch_status = hipErrorOutOfMemory;
        } else {
          auto *const packed_activation =
              static_cast<uint8_t *>(plan->matmul_workspace);
          auto *const activation_block_scales =
              packed_activation + packed_activation_bytes;
          const auto *const packed_weight =
              static_cast<const uint8_t *>(weight_storage);
          const auto *const weight_block_scales =
              packed_weight + plan->matmul_metadata.weight_value_bytes;
          const auto *const weight_tensor_scale =
              reinterpret_cast<const float *>(
                  static_cast<const uint8_t *>(weight_storage) +
                  (plan->matmul_metadata.weight_tensor_scale_offset -
                   plan->matmul_metadata.weight.byte_offset));
          const auto *const input_tensor_scale =
              reinterpret_cast<const float *>(
                  static_cast<const uint8_t *>(weight_storage) +
                  (plan->matmul_metadata.input_tensor_scale_offset -
                   plan->matmul_metadata.weight.byte_offset));
          launch_status = ::sllm_matmul_kernel::launch_nvfp4_quantize(
              activation, packed_activation, activation_block_scales,
              input_tensor_scale, plan->matmul_metadata.m,
              plan->matmul_metadata.k, queue->stream);
          if (launch_status == hipSuccess) {
            const bool split4_execute =
                ((variant == ::sllm_matmul_kernel::KernelVariant::
                                 Nvfp4W4A4PrefillDp4a64x64 &&
                  matches_runtime_gcn_arch(arch_name, "gfx1030") &&
                  ::sllm_matmul_kernel::phase78_gfx1030_nvfp4_w4a4_split4_shape(
                      plan->matmul_metadata.m, plan->matmul_metadata.k,
                      plan->matmul_metadata.n)) ||
                 (variant == ::sllm_matmul_kernel::KernelVariant::
                                 Nvfp4W4A4PrefillGfx1201Wmma128x64 &&
                  matches_runtime_gcn_arch(arch_name, "gfx1201") &&
                  ::sllm_matmul_kernel::phase78_gfx1201_nvfp4_w4a4_split4_shape(
                      plan->matmul_metadata.m, plan->matmul_metadata.k,
                      plan->matmul_metadata.n)));
            if (split4_execute) {
              uint64_t partial_offset = 0U;
              uint64_t split4_workspace_bytes = 0U;
              if (!::sllm_matmul_kernel::phase78_nvfp4_w4a4_split4_workspace(
                      plan->matmul_metadata.m, plan->matmul_metadata.k,
                      plan->matmul_metadata.n, &partial_offset,
                      &split4_workspace_bytes) ||
                  plan->matmul_workspace == nullptr ||
                  plan->matmul_workspace_bytes < split4_workspace_bytes) {
                launch_status = hipErrorOutOfMemory;
              } else {
                auto *const partial = reinterpret_cast<float *>(
                    packed_activation + partial_offset);
                if (variant == ::sllm_matmul_kernel::KernelVariant::
                                   Nvfp4W4A4PrefillDp4a64x64) {
                  launch_status = ::sllm_matmul_kernel::
                      launch_nvfp4_w4a4_prefill_dp4a_short_split4(
                          packed_activation, activation_block_scales,
                          packed_weight, weight_block_scales,
                          weight_tensor_scale, input_tensor_scale, output,
                          partial, plan->matmul_metadata.m,
                          plan->matmul_metadata.k, plan->matmul_metadata.n,
                          queue->stream);
                } else {
                  launch_status = ::sllm_matmul_kernel::
                      launch_nvfp4_w4a4_prefill_gfx1201_wmma128x64_split4(
                          packed_activation, activation_block_scales,
                          packed_weight, weight_block_scales,
                          weight_tensor_scale, input_tensor_scale, output,
                          partial, plan->matmul_metadata.m,
                          plan->matmul_metadata.k, plan->matmul_metadata.n,
                          queue->stream);
                }
              }
            } else if (variant == ::sllm_matmul_kernel::KernelVariant::
                                      Nvfp4W4A4PrefillGfx1201Fp8Staging) {
              ::sllm_matmul_kernel::Nvfp4Fp8StagingWorkspaceLayout
                  staging_layout{};
              const bool staging_valid =
                  ::sllm_matmul_kernel::nvfp4_w4a4_fp8_staging_workspace(
                      plan->matmul_metadata.m, plan->matmul_metadata.k,
                      plan->matmul_metadata.n, &staging_layout) &&
                  matmul_f16_staging_workspace != nullptr &&
                  matmul_f16_staging_workspace_bytes >=
                      staging_layout.total_bytes &&
                  plan->fp8_lt_plan != nullptr;
              if (!staging_valid) {
                launch_status = hipErrorOutOfMemory;
              } else {
                auto *const staging_bytes =
                    static_cast<uint8_t *>(matmul_f16_staging_workspace);
                auto *const staged_activation =
                    staging_bytes + staging_layout.activation_offset;
                auto *const staged_weight =
                    staging_bytes + staging_layout.weight_offset;
                auto *const scale_product = reinterpret_cast<float *>(
                    staging_bytes + staging_layout.scale_product_offset);
                launch_status =
                    ::sllm_matmul_kernel::launch_nvfp4_block16_to_fp8_staging(
                        packed_activation, activation_block_scales,
                        staged_activation, plan->matmul_metadata.m,
                        plan->matmul_metadata.k, queue->stream);
                if (launch_status == hipSuccess) {
                  launch_status =
                      ::sllm_matmul_kernel::launch_nvfp4_block16_to_fp8_staging(
                          packed_weight, weight_block_scales, staged_weight,
                          plan->matmul_metadata.n, plan->matmul_metadata.k,
                          queue->stream);
                }
                if (launch_status == hipSuccess) {
                  launch_status =
                      ::sllm_matmul_kernel::launch_nvfp4_tensor_scale_product(
                          weight_tensor_scale, input_tensor_scale,
                          scale_product, queue->stream);
                }
                if (launch_status == hipSuccess) {
                  std::lock_guard<std::mutex> lt_lock(
                      plan->context->matmul_lt_mutex);
                  launch_status =
                      launch_fp8_lt_plan(plan->fp8_lt_plan, staged_activation,
                                         staged_weight, output, queue->stream);
                }
              }
            } else if (variant == ::sllm_matmul_kernel::KernelVariant::
                                      Nvfp4W4A4PrefillGfx1201F16Staging) {
              ::sllm_matmul_kernel::F16StagingWorkspaceLayout staging_layout{};
              const bool staging_valid =
                  ::sllm_matmul_kernel::
                      phase78_gfx1201_nvfp4_w4a4_f16_staging_shape(
                          plan->matmul_metadata.m, plan->matmul_metadata.k,
                          plan->matmul_metadata.n) &&
                  ::sllm_matmul_kernel::f16_staging_workspace_layout(
                      plan->matmul_metadata.m, plan->matmul_metadata.k,
                      plan->matmul_metadata.n, &staging_layout) &&
                  matmul_f16_staging_workspace != nullptr &&
                  matmul_f16_staging_workspace_bytes >=
                      staging_layout.total_bytes;
              if (!staging_valid) {
                launch_status = hipErrorOutOfMemory;
              } else {
                auto *const staging_bytes =
                    static_cast<uint8_t *>(matmul_f16_staging_workspace);
                auto *const activation_f16 = reinterpret_cast<uint16_t *>(
                    staging_bytes + staging_layout.activation_offset);
                auto *const weight_f16 = reinterpret_cast<uint16_t *>(
                    staging_bytes + staging_layout.weight_offset);
                auto *const output_f32 = reinterpret_cast<float *>(
                    staging_bytes + staging_layout.output_offset);
                launch_status =
                    ::sllm_matmul_kernel::launch_nvfp4_block16_to_fp16_staging(
                        packed_activation, activation_block_scales,
                        activation_f16, plan->matmul_metadata.m,
                        plan->matmul_metadata.k, queue->stream);
                if (launch_status == hipSuccess) {
                  launch_status = ::sllm_matmul_kernel::
                      launch_nvfp4_block16_to_fp16_staging(
                          packed_weight, weight_block_scales, weight_f16,
                          plan->matmul_metadata.n, plan->matmul_metadata.k,
                          queue->stream);
                }
                if (launch_status == hipSuccess) {
                  launch_status = launch_matmul_f16_staging_f32_gemm(
                      plan->context, queue->stream, activation_f16, weight_f16,
                      output_f32, plan->matmul_metadata.m,
                      plan->matmul_metadata.k, plan->matmul_metadata.n);
                }
                if (launch_status == hipSuccess) {
                  launch_status =
                      ::sllm_matmul_kernel::launch_nvfp4_tensor_scale_epilogue(
                          output_f32, weight_tensor_scale, input_tensor_scale,
                          output, plan->matmul_metadata.m,
                          plan->matmul_metadata.n, queue->stream);
                }
              }
            } else {
              launch_status = ::sllm_matmul_kernel::launch_nvfp4_w4a4(
                  packed_activation, activation_block_scales, packed_weight,
                  weight_block_scales, weight_tensor_scale, input_tensor_scale,
                  output, plan->matmul_metadata.m, plan->matmul_metadata.k,
                  plan->matmul_metadata.n, variant, queue->stream);
            }
          }
        }
      }
#endif
    } else if (plan->matmul_metadata.nvfp4) {
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
      launch_status = hipErrorInvalidValue;
#else
      if (!matches_runtime_gcn_arch(arch_name, "gfx1201") &&
          !matches_runtime_gcn_arch(arch_name, "gfx1030")) {
        launch_status = hipErrorNotSupported;
      } else {
        const auto *const packed_weight =
            static_cast<const uint8_t *>(weight_storage);
        const auto *const block_scales =
            static_cast<const uint8_t *>(weight_storage) +
            plan->matmul_metadata.weight_value_bytes;
        const auto *const tensor_scale = reinterpret_cast<const float *>(
            static_cast<const uint8_t *>(weight_storage) +
            (plan->matmul_metadata.weight_tensor_scale_offset -
             plan->matmul_metadata.weight.byte_offset));
        launch_status = ::sllm_matmul_kernel::launch_nvfp4(
            activation, packed_weight, block_scales, tensor_scale, output,
            plan->matmul_metadata.m, plan->matmul_metadata.k,
            plan->matmul_metadata.n, variant, queue->stream);
      }
#endif
    } else if (plan->matmul_metadata.fp8_outer) {
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
      // The production kernels are not linked into the host-contract binary.
      // Treat the model-free staging/LUT routes as successful ordered enqueues
      // so their context-workspace lease, completion release, and dispatch
      // metadata can be exercised without claiming a numerical GPU result.
      launch_status = variant == ::sllm_matmul_kernel::KernelVariant::
                                      Fp8OuterPrefillGfx1030F16Staging ||
                              variant ==
                                  ::sllm_matmul_kernel::KernelVariant::
                                      Fp8OuterPrefillGfx1030F16TileStaging ||
                              variant == ::sllm_matmul_kernel::KernelVariant::
                                             Fp8OuterPrefillGfx1030LdsLut
                          ? hipSuccess
                          : hipErrorInvalidValue;
#else
      const uint64_t activation_value_bytes =
          plan->matmul_metadata.m * plan->matmul_metadata.k;
      const uint64_t scale_offset =
          (activation_value_bytes + UINT64_C(3)) & ~UINT64_C(3);
      if (plan->matmul_workspace == nullptr ||
          plan->matmul_workspace_bytes <
              scale_offset + plan->matmul_metadata.m * UINT64_C(4)) {
        launch_status = hipErrorOutOfMemory;
      } else {
        auto *const quantized = static_cast<uint8_t *>(plan->matmul_workspace);
        auto *const activation_scales = reinterpret_cast<float *>(
            static_cast<uint8_t *>(plan->matmul_workspace) + scale_offset);
        const auto *const fp8_weight =
            static_cast<const uint8_t *>(weight_storage);
        const auto *const weight_scales = reinterpret_cast<const float *>(
            static_cast<const uint8_t *>(weight_storage) +
            plan->matmul_metadata.weight_value_bytes);
        launch_status = ::sllm_matmul_kernel::launch_fp8_quantize(
            activation, quantized, activation_scales, plan->matmul_metadata.m,
            plan->matmul_metadata.k,
            plan->matmul_metadata.fp8_dtype == SLLM_TENSOR_DTYPE_F8_E4M3_FNUZ,
            queue->stream);
        if (launch_status == hipSuccess) {
          if (variant == ::sllm_matmul_kernel::KernelVariant::
                             Fp8OuterPrefillGfx1030F16Staging) {
            ::sllm_matmul_kernel::F16StagingWorkspaceLayout staging_layout{};
            const bool staging_valid =
                ::sllm_matmul_kernel::
                    fp8_outer_prefill_gfx1030_f16_staging_workspace(
                        plan->matmul_metadata.m, plan->matmul_metadata.k,
                        plan->matmul_metadata.n, &staging_layout) &&
                matmul_f16_staging_workspace != nullptr &&
                matmul_f16_staging_workspace_bytes >=
                    staging_layout.total_bytes;
            if (!staging_valid) {
              launch_status = hipErrorOutOfMemory;
            } else {
              auto *const staging_bytes =
                  static_cast<uint8_t *>(matmul_f16_staging_workspace);
              auto *const activation_f16 = reinterpret_cast<uint16_t *>(
                  staging_bytes + staging_layout.activation_offset);
              auto *const weight_f16 = reinterpret_cast<uint16_t *>(
                  staging_bytes + staging_layout.weight_offset);
              auto *const output_f32 = reinterpret_cast<float *>(
                  staging_bytes + staging_layout.output_offset);
              launch_status =
                  ::sllm_matmul_kernel::launch_fp8_e4m3fn_to_fp16_staging(
                      quantized, activation_f16,
                      staging_layout.activation_bytes / UINT64_C(2),
                      queue->stream);
              if (launch_status == hipSuccess) {
                launch_status =
                    ::sllm_matmul_kernel::launch_fp8_e4m3fn_to_fp16_staging(
                        fp8_weight, weight_f16,
                        staging_layout.weight_bytes / UINT64_C(2),
                        queue->stream);
              }
              if (launch_status == hipSuccess) {
                launch_status = launch_matmul_f16_staging_f32_gemm(
                    plan->context, queue->stream, activation_f16, weight_f16,
                    output_f32, plan->matmul_metadata.m,
                    plan->matmul_metadata.k, plan->matmul_metadata.n);
              }
              if (launch_status == hipSuccess) {
                launch_status =
                    ::sllm_matmul_kernel::launch_fp8_outer_f16scale_epilogue(
                        output_f32, activation_scales, weight_scales, output,
                        plan->matmul_metadata.m, plan->matmul_metadata.n,
                        queue->stream);
              }
            }
          } else if (variant == ::sllm_matmul_kernel::KernelVariant::
                                    Fp8OuterPrefillGfx1030F16TileStaging) {
            ::sllm_matmul_kernel::F16StagingWorkspaceLayout staging_layout{};
            const bool staging_valid =
                ::sllm_matmul_kernel::
                    fp8_outer_prefill_gfx1030_f16_tile_staging_workspace(
                        plan->matmul_metadata.m, plan->matmul_metadata.k,
                        plan->matmul_metadata.n, &staging_layout) &&
                matmul_f16_staging_workspace != nullptr &&
                matmul_f16_staging_workspace_bytes >=
                    staging_layout.total_bytes;
            if (!staging_valid) {
              launch_status = hipErrorOutOfMemory;
            } else {
              auto *const staging_bytes =
                  static_cast<uint8_t *>(matmul_f16_staging_workspace);
              auto *const activation_f16 = reinterpret_cast<uint16_t *>(
                  staging_bytes + staging_layout.activation_offset);
              auto *const weight_f16 = reinterpret_cast<uint16_t *>(
                  staging_bytes + staging_layout.weight_offset);
              launch_status =
                  ::sllm_matmul_kernel::launch_fp8_e4m3fn_to_fp16_staging(
                      quantized, activation_f16,
                      staging_layout.activation_bytes / UINT64_C(2),
                      queue->stream);
              if (launch_status == hipSuccess) {
                launch_status =
                    ::sllm_matmul_kernel::launch_fp8_e4m3fn_to_fp16_staging(
                        fp8_weight, weight_f16,
                        staging_layout.weight_bytes / UINT64_C(2),
                        queue->stream);
              }
              if (launch_status == hipSuccess) {
                launch_status = ::sllm_matmul_kernel::
                    launch_fp8_outer_prefill_gfx1030_f16_tile_staging(
                        activation_f16, activation_scales, weight_f16,
                        weight_scales, output, plan->matmul_metadata.m,
                        plan->matmul_metadata.k, plan->matmul_metadata.n,
                        queue->stream);
              }
            }
          } else if (variant ==
                     ::sllm_matmul_kernel::KernelVariant::Fp8Native) {
            std::lock_guard<std::mutex> lt_lock(plan->context->matmul_lt_mutex);
            launch_status =
                launch_fp8_lt_plan(plan->fp8_lt_plan, quantized, fp8_weight,
                                   output, queue->stream);
          } else if (matches_runtime_gcn_arch(arch_name, "gfx1030")) {
            if (variant == ::sllm_matmul_kernel::KernelVariant::
                               Fp8OuterDecodeGfx1030Dword8Wave4Col32) {
              launch_status = ::sllm_matmul_kernel::
                  launch_fp8_outer_decode_gfx1030_dword8_wave4col32(
                      quantized, activation_scales, fp8_weight, weight_scales,
                      output, plan->matmul_metadata.m, plan->matmul_metadata.k,
                      plan->matmul_metadata.n, queue->stream);
            } else if (variant == ::sllm_matmul_kernel::KernelVariant::
                                      Fp8OuterDecodeGfx1030LdsLutWave4Col32) {
              launch_status = ::sllm_matmul_kernel::
                  launch_fp8_outer_decode_gfx1030_lds_lut_wave4col32(
                      quantized, activation_scales, fp8_weight, weight_scales,
                      output, plan->matmul_metadata.m, plan->matmul_metadata.k,
                      plan->matmul_metadata.n, queue->stream);
            } else if (variant ==
                       ::sllm_matmul_kernel::KernelVariant::
                           Fp8OuterDecodeGfx1030ActivationSharedWave4Col32) {
              launch_status = ::sllm_matmul_kernel::
                  launch_fp8_outer_decode_gfx1030_activation_shared_wave4col32(
                      quantized, activation_scales, fp8_weight, weight_scales,
                      output, plan->matmul_metadata.m, plan->matmul_metadata.k,
                      plan->matmul_metadata.n, queue->stream);
            } else if (variant ==
                       ::sllm_matmul_kernel::KernelVariant::
                           Fp8OuterDecodeGfx1030ActivationSharedWave8Col64) {
              launch_status = ::sllm_matmul_kernel::
                  launch_fp8_outer_decode_gfx1030_activation_shared_wave8col64(
                      quantized, activation_scales, fp8_weight, weight_scales,
                      output, plan->matmul_metadata.m, plan->matmul_metadata.k,
                      plan->matmul_metadata.n, queue->stream);
            } else if (variant == ::sllm_matmul_kernel::KernelVariant::
                                      Fp8OuterDecodeGfx1030Half2Wave4Col32) {
              launch_status = ::sllm_matmul_kernel::
                  launch_fp8_outer_decode_gfx1030_half2_wave4col32(
                      quantized, activation_scales, fp8_weight, weight_scales,
                      output, plan->matmul_metadata.m, plan->matmul_metadata.k,
                      plan->matmul_metadata.n, queue->stream);
            } else if (variant == ::sllm_matmul_kernel::KernelVariant::
                                      Fp8OuterPrefillGfx1030Half2_64x64) {
              launch_status = ::sllm_matmul_kernel::
                  launch_fp8_outer_prefill_gfx1030_half2_64x64(
                      quantized, activation_scales, fp8_weight, weight_scales,
                      output, plan->matmul_metadata.m, plan->matmul_metadata.k,
                      plan->matmul_metadata.n, queue->stream);
            } else if (variant == ::sllm_matmul_kernel::KernelVariant::
                                      Fp8OuterPrefillGfx1030LdsLut) {
              launch_status = ::sllm_matmul_kernel::
                  launch_fp8_outer_prefill_gfx1030_lds_lut(
                      quantized, activation_scales, fp8_weight, weight_scales,
                      output, plan->matmul_metadata.m, plan->matmul_metadata.k,
                      plan->matmul_metadata.n, queue->stream);
            } else if (variant == ::sllm_matmul_kernel::KernelVariant::
                                      Fp8OuterPrefillGfx1030Half2_128x64) {
              launch_status = ::sllm_matmul_kernel::
                  launch_fp8_outer_prefill_gfx1030_half2_128x64(
                      quantized, activation_scales, fp8_weight, weight_scales,
                      output, plan->matmul_metadata.m, plan->matmul_metadata.k,
                      plan->matmul_metadata.n, queue->stream);
            } else if (variant == ::sllm_matmul_kernel::KernelVariant::
                                      Fp8OuterPrefillTiled16) {
              launch_status =
                  ::sllm_matmul_kernel::launch_fp8_outer_prefill_tiled16(
                      quantized, activation_scales, fp8_weight, weight_scales,
                      output, plan->matmul_metadata.m, plan->matmul_metadata.k,
                      plan->matmul_metadata.n, queue->stream);
            } else {
              launch_status = ::sllm_matmul_kernel::launch_fp8_emulation(
                  quantized, activation_scales, fp8_weight, weight_scales,
                  output, plan->matmul_metadata.m, plan->matmul_metadata.k,
                  plan->matmul_metadata.n, queue->stream);
            }
          } else {
            launch_status = hipErrorNotSupported;
          }
        }
      }
#endif
    } else
#if !defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
        if (variant == ::sllm_matmul_kernel::KernelVariant::PrefillShortMixed) {
      uint64_t workspace_bytes = 0U;
      uint64_t output_elements = 0U;
      bool workspace_valid =
          ::sllm_matmul_kernel::phase49_gfx1030_mixed_workspace_bytes(
              plan->matmul_metadata.m, plan->matmul_metadata.n,
              &workspace_bytes);
      if (workspace_valid) {
        output_elements = plan->matmul_metadata.m * plan->matmul_metadata.n;
        workspace_valid =
            plan->matmul_workspace != nullptr &&
            plan->matmul_workspace_bytes >= workspace_bytes &&
            output_elements <= UINT32_MAX &&
            plan->matmul_metadata.m <=
                static_cast<uint64_t>(std::numeric_limits<int>::max()) &&
            plan->matmul_metadata.n <=
                static_cast<uint64_t>(std::numeric_limits<int>::max()) &&
            plan->matmul_metadata.k <=
                static_cast<uint64_t>(std::numeric_limits<int>::max());
      }
      if (!workspace_valid) {
        launch_status = hipErrorOutOfMemory;
      } else if (plan->context->matmul_blas_handle == nullptr) {
        launch_status = hipErrorInvalidHandle;
      } else {
        auto *const output_f32 = static_cast<float *>(plan->matmul_workspace);
        std::lock_guard<std::mutex> blas_lock(plan->context->matmul_blas_mutex);
        const float alpha = 1.0F;
        const float beta = 0.0F;
        const int32_t short_mixed_solution =
            ::sllm_matmul_kernel::phase49_gfx1030_short_mixed_rocblas_solution(
                plan->matmul_metadata.m, plan->matmul_metadata.k,
                plan->matmul_metadata.n);
        const char *const force_baseline =
            std::getenv("SLLM_MATMUL_FORCE_BASELINE");
        const bool force_baseline_enabled =
            force_baseline != nullptr && std::strcmp(force_baseline, "1") == 0;
        bool short_mixed_rocblas_succeeded = false;
        if (!force_baseline_enabled && short_mixed_solution != 0 &&
            ::sllm_matmul_kernel::phase49_gfx1030_short_mixed_rocblas_enabled(
                arch_name, plan->matmul_metadata.m, plan->matmul_metadata.k,
                plan->matmul_metadata.n) &&
            plan->context->matmul_rocblas_handle != nullptr &&
            rocblas_set_stream(plan->context->matmul_rocblas_handle,
                               queue->stream) == rocblas_status_success) {
          const rocblas_status rocblas_status =
              (rocblas_gemm_ex)(plan->context->matmul_rocblas_handle,
                                rocblas_operation_transpose,
                                rocblas_operation_none,
                                static_cast<rocblas_int>(
                                    plan->matmul_metadata.n),
                                static_cast<rocblas_int>(
                                    plan->matmul_metadata.m),
                                static_cast<rocblas_int>(
                                    plan->matmul_metadata.k),
                                &alpha, weight, rocblas_datatype_bf16_r,
                                static_cast<rocblas_int>(
                                    plan->matmul_metadata.k),
                                activation, rocblas_datatype_bf16_r,
                                static_cast<rocblas_int>(
                                    plan->matmul_metadata.k),
                                &beta, output_f32, rocblas_datatype_f32_r,
                                static_cast<rocblas_int>(
                                    plan->matmul_metadata.n),
                                output_f32, rocblas_datatype_f32_r,
                                static_cast<rocblas_int>(
                                    plan->matmul_metadata.n),
                                rocblas_datatype_f32_r,
                                rocblas_gemm_algo_solution_index,
                                short_mixed_solution, 0U);
          short_mixed_rocblas_succeeded =
              rocblas_status == rocblas_status_success;
        }
        if (!short_mixed_rocblas_succeeded) {
          const hipblasStatus_t stream_status = hipblasSetStream(
              plan->context->matmul_blas_handle, queue->stream);
          const hipblasStatus_t blas_status =
              stream_status == HIPBLAS_STATUS_SUCCESS
                  ? hipblasGemmEx(plan->context->matmul_blas_handle,
                                  HIPBLAS_OP_T, HIPBLAS_OP_N,
                                  static_cast<int>(plan->matmul_metadata.n),
                                  static_cast<int>(plan->matmul_metadata.m),
                                  static_cast<int>(plan->matmul_metadata.k),
                                  &alpha, weight, HIPBLAS_R_16B,
                                  static_cast<int>(plan->matmul_metadata.k),
                                  activation, HIPBLAS_R_16B,
                                  static_cast<int>(plan->matmul_metadata.k),
                                  &beta, output_f32, HIPBLAS_R_32F,
                                  static_cast<int>(plan->matmul_metadata.n),
                                  HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT)
                  : stream_status;
          if (blas_status != HIPBLAS_STATUS_SUCCESS) {
            launch_status = hipErrorUnknown;
          }
        }
        if (launch_status == hipSuccess) {
          launch_status = ::sllm_matmul_kernel::launch_short_mixed_f32_to_bf16(
              output_f32, output, output_elements, queue->stream);
        }
      }
    } else if (variant == ::sllm_matmul_kernel::KernelVariant::HipBlas) {
      if (plan->context->matmul_blas_handle == nullptr) {
        launch_status = hipErrorInvalidHandle;
      } else {
        std::lock_guard<std::mutex> blas_lock(plan->context->matmul_blas_mutex);
        const float alpha = 1.0F;
        const float beta = 0.0F;
        const hipblasStatus_t stream_status =
            hipblasSetStream(plan->context->matmul_blas_handle, queue->stream);
        const char *const solution_env =
            std::getenv("SLLM_MATMUL_GFX1030_ROCBLAS_SOLUTION_445");
        const bool use_rocblas_solution_445 =
            matches_runtime_gcn_arch(arch_name, "gfx1030") &&
            ::sllm_matmul_kernel::phase49_gfx1030_rocblas_solution_445_shape(
                plan->matmul_metadata.m, plan->matmul_metadata.k,
                plan->matmul_metadata.n) &&
            (solution_env == nullptr || std::strcmp(solution_env, "1") == 0);
        if (stream_status != HIPBLAS_STATUS_SUCCESS) {
          launch_status = hipErrorUnknown;
        } else if (use_rocblas_solution_445) {
          if (plan->context->matmul_rocblas_handle == nullptr ||
              rocblas_set_stream(plan->context->matmul_rocblas_handle,
                                 queue->stream) != rocblas_status_success) {
            launch_status = hipErrorInvalidHandle;
          } else {
            const std::array<uint64_t, 3> shape = {plan->matmul_metadata.m,
                                                   plan->matmul_metadata.k,
                                                   plan->matmul_metadata.n};
            bool shape_is_primed = false;
            for (const auto &primed_shape :
                 plan->context->matmul_rocblas_primed_shapes) {
              if (primed_shape == shape) {
                shape_is_primed = true;
                break;
              }
            }
            rocblas_status query_status = rocblas_status_success;
            if (!shape_is_primed) {
              rocblas_int solution_count = 0;
              query_status = rocblas_gemm_ex_get_solutions(
                  plan->context->matmul_rocblas_handle,
                  rocblas_operation_transpose, rocblas_operation_none,
                  static_cast<rocblas_int>(plan->matmul_metadata.n),
                  static_cast<rocblas_int>(plan->matmul_metadata.m),
                  static_cast<rocblas_int>(plan->matmul_metadata.k), &alpha,
                  weight, rocblas_datatype_bf16_r,
                  static_cast<rocblas_int>(plan->matmul_metadata.k), activation,
                  rocblas_datatype_bf16_r,
                  static_cast<rocblas_int>(plan->matmul_metadata.k), &beta,
                  output, rocblas_datatype_bf16_r,
                  static_cast<rocblas_int>(plan->matmul_metadata.n), output,
                  rocblas_datatype_bf16_r,
                  static_cast<rocblas_int>(plan->matmul_metadata.n),
                  rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0U,
                  nullptr, &solution_count);
              if (query_status == rocblas_status_success &&
                  solution_count > 0) {
                plan->context->matmul_rocblas_primed_shapes.push_back(shape);
              } else if (query_status == rocblas_status_success) {
                query_status = rocblas_status_not_implemented;
              }
            }
            rocblas_status execution_status = query_status;
            if (query_status == rocblas_status_success) {
              execution_status =
                  (rocblas_gemm_ex)(plan->context->matmul_rocblas_handle,
                                    rocblas_operation_transpose,
                                    rocblas_operation_none,
                                    static_cast<rocblas_int>(
                                        plan->matmul_metadata.n),
                                    static_cast<rocblas_int>(
                                        plan->matmul_metadata.m),
                                    static_cast<rocblas_int>(
                                        plan->matmul_metadata.k),
                                    &alpha, weight, rocblas_datatype_bf16_r,
                                    static_cast<rocblas_int>(
                                        plan->matmul_metadata.k),
                                    activation, rocblas_datatype_bf16_r,
                                    static_cast<rocblas_int>(
                                        plan->matmul_metadata.k),
                                    &beta, output, rocblas_datatype_bf16_r,
                                    static_cast<rocblas_int>(
                                        plan->matmul_metadata.n),
                                    output, rocblas_datatype_bf16_r,
                                    static_cast<rocblas_int>(
                                        plan->matmul_metadata.n),
                                    rocblas_datatype_f32_r,
                                    rocblas_gemm_algo_solution_index,
                                    ::sllm_matmul_kernel::
                                        kPhase49Gfx1030RocblasSolution445,
                                    0U);
            }
            if (execution_status != rocblas_status_success) {
              launch_status = hipErrorUnknown;
            }
          }
        } else {
          const hipblasStatus_t blas_status = hipblasGemmEx(
              plan->context->matmul_blas_handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
              static_cast<int>(plan->matmul_metadata.n),
              static_cast<int>(plan->matmul_metadata.m),
              static_cast<int>(plan->matmul_metadata.k), &alpha, weight,
              HIPBLAS_R_16B, static_cast<int>(plan->matmul_metadata.k),
              activation, HIPBLAS_R_16B,
              static_cast<int>(plan->matmul_metadata.k), &beta, output,
              HIPBLAS_R_16B, static_cast<int>(plan->matmul_metadata.n),
              HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
          if (blas_status != HIPBLAS_STATUS_SUCCESS) {
            launch_status = hipErrorUnknown;
          }
        }
      }
    } else
#endif
    {
      launch_status = ::sllm_matmul_kernel::launch(
          activation, weight, output, plan->matmul_metadata.m,
          plan->matmul_metadata.k, plan->matmul_metadata.n, variant,
          queue->stream);
    }
    if (launch_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, launch_status,
                                       "matmul kernel launch", queue,
                                       error_sink);
    }
    if (candidate->event != nullptr) {
      const hipError_t record_status =
          hipEventRecord(candidate->event, queue->stream);
      if (record_status != hipSuccess) {
        execute_guard.disarm();
        return cleanup_failed_submission(candidate, token, record_status,
                                         "hipEventRecord", queue, error_sink);
      }
    }
    initialize_matmul_dispatch_info(dispatch_info, dispatch_id,
                                    plan->matmul_metadata, variant, arch_name);
    *completion_output = reinterpret_cast<sllm_completion_t *>(token);
    (void)candidate.release();
    execute_guard.disarm();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in matmul execute");
  }
}
