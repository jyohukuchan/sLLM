#ifndef SLLM_LOW_PRECISION_MATMUL_PROVIDER_HPP
#define SLLM_LOW_PRECISION_MATMUL_PROVIDER_HPP

#include <cstdint>

namespace sllm_lowp {

enum class MatmulFormat : uint8_t {
  Mxfp8E4M3W8A8,
  Mxfp6E3M2W6A6,
  Nvfp4W4A16,
  Nvfp4W4A4,
  Mxfp4W4A4,
  Fp8OuterE4M3W8A8,
};

enum class ScalarType : uint8_t {
  Bf16,
  Fp32,
  E4M3Fn,
  E3M2,
  E2M1,
};

enum class BlockScaleType : uint8_t {
  None,
  E8M0,
  E4M3FnWithFp32TensorScale,
  Fp32OuterVector,
};

enum class BlockKind : uint8_t {
  None,
  Mxfp8E4M3Block32,
  Mxfp6E3M2Block32,
  Nvfp4E2M1Block16,
  Mxfp4E2M1Block32,
};

enum class BlockLayout : uint8_t {
  RowMajor,
  RowMajorBlockScaled,
  ConsumerTiledBlockScaled,
};

enum class TargetArchitecture : uint8_t {
  Unknown,
  Rdna2Wave32,
  Rdna4Wave32,
  Cdna3Wave64,
};

enum class ExactTarget : uint8_t {
  Unknown,
  Gfx1030,
  Gfx1201,
  Gfx942SrameccOnXnackOff,
};

enum class TilePolicy : uint8_t {
  None,
  DecodeRowReduction,
  Elementwise,
  BlockRow8,
  BlockTiled16x16,
  BlockRow8Column4,
  BlockRow8Column8,
  PackedRow8,
  Wmma128x16x32,
  Wmma64x64x32,
  Wmma128x64x32,
  Wmma128x128x32,
  BlockRow8Column16,
  BlockRow8Column32,
  BlockRow32Column32,
  BlockRow64Column64,
  BlockRow128Column32,
  BlockRow128Column64,
  DecodeColumns128,
  DecodeWave4Column32,
  DecodeDword8Wave4Column32,
  // Phase 78 ID71 short-M FP8 specialization with a 32x64 output tile.
  // Appended to preserve the numeric values of the existing audit ABI.
  BlockRow32Column64,
};

enum class ActivationPack : uint8_t {
  NoneBf16,
  Mxfp8E4M3Block32,
  Mxfp6E3M2Block32,
  Nvfp4E2M1Block16,
  Mxfp4E2M1Block32,
  Fp8E4M3Outer,
};

enum class InnerProduct : uint8_t {
  None,
  DecodedBlockScaledFp32,
  E4M3WmmaFp32,
  E3M2ViaE4M3DecodedFp32,
  E3M2ViaE4M3WmmaFp32,
  E3M2Fp16Dot2Fp32,
  E4M3Fp16Dot2Fp32,
  E2M1Bf16Fp32,
  E2M1BlockScaledFp32,
  E2M1ViaE4M3WmmaFp32,
  E2M1BlockScaledDp4aFp32,
  E4M3OuterFp32,
  // NVFP4 W4A4 gfx1201 candidate: decode E2M1 and absorb each E4M3
  // block-16 scale into FP16 WMMA operands at tile ingress.
  E2M1Fp16ScaleWmmaFp32,
  // NVFP4 W4A4 gfx1201 candidate: consume block scales while staging to
  // native E4M3 FP8, then accumulate with the hipBLASLt FP8 provider.
  E2M1ViaE4M3NativeFp8Fp32,
};

enum class AccumulationType : uint8_t {
  Fp32,
};

enum class OutputType : uint8_t {
  Bf16Rne,
};

// ProviderKind deliberately does not mirror matmul KernelVariant. Runtime
// dispatch can map these semantic providers to one or more target kernels
// without making this reusable contract depend on kernel symbol inventory.
enum class ProviderKind : uint8_t {
  Unsupported,
  Mxfp8Block32,
  Mxfp8Gfx1201Wmma,
  Mxfp6Block32,
  Mxfp6Gfx1030MmqViaE4M3,
  Mxfp6Gfx1030Half2Dot2,
  Mxfp6Gfx1201WmmaViaE4M3,
  Nvfp4W4A16Block16,
  Nvfp4W4A4Block16,
  Mxfp4W4A4Block32,
  Fp8OuterGfx1030Software,
};

enum class ProviderRejection : uint8_t {
  None,
  UnknownTarget,
  UnsupportedTarget,
  EmptyDimension,
  KNotBlockAligned,
  InvalidLayout,
  UnsupportedNumerics,
};

struct FormatContract {
  MatmulFormat format;
  ScalarType weight_element;
  ScalarType activation_element;
  BlockKind weight_block;
  BlockKind activation_block;
  BlockScaleType weight_scale;
  BlockScaleType activation_scale;
  uint32_t weight_block_size;
  uint32_t activation_block_size;
  uint32_t weight_bits;
  uint32_t activation_bits;
  bool weight_has_tensor_scale;
  bool activation_has_tensor_scale;
};

struct ProviderRequest {
  MatmulFormat format;
  BlockLayout weight_layout;
  BlockLayout activation_layout;
  ExactTarget target;
  uint64_t m;
  uint64_t n;
  uint64_t k;
  AccumulationType accumulation;
  OutputType output;
};

struct PreparedProviderPlan {
  const ProviderKind provider;
  const ProviderRejection rejection;
  const MatmulFormat format;
  const FormatContract block_contract;
  const BlockLayout weight_layout;
  const BlockLayout activation_layout;
  const TargetArchitecture architecture;
  const ExactTarget target;
  const TilePolicy tile;
  const ActivationPack activation_pack;
  const InnerProduct inner_product;
  const uint64_t m;
  const uint64_t n;
  const uint64_t k;
  const AccumulationType accumulation;
  const OutputType output;

  constexpr bool supported() const noexcept {
    return provider != ProviderKind::Unsupported &&
           rejection == ProviderRejection::None;
  }
};

constexpr bool c_string_equal(const char *left, const char *right) noexcept {
  if (left == nullptr || right == nullptr) {
    return false;
  }
  while (*left != '\0' && *right != '\0') {
    if (*left != *right) {
      return false;
    }
    ++left;
    ++right;
  }
  return *left == *right;
}

constexpr ExactTarget exact_target_from_name(const char *target) noexcept {
  return c_string_equal(target, "gfx1030")   ? ExactTarget::Gfx1030
         : c_string_equal(target, "gfx1201") ? ExactTarget::Gfx1201
         : c_string_equal(target, "gfx942:sramecc+:xnack-")
             ? ExactTarget::Gfx942SrameccOnXnackOff
             : ExactTarget::Unknown;
}

constexpr TargetArchitecture
target_architecture(const ExactTarget target) noexcept {
  return target == ExactTarget::Gfx1030   ? TargetArchitecture::Rdna2Wave32
         : target == ExactTarget::Gfx1201 ? TargetArchitecture::Rdna4Wave32
         : target == ExactTarget::Gfx942SrameccOnXnackOff
             ? TargetArchitecture::Cdna3Wave64
             : TargetArchitecture::Unknown;
}

constexpr FormatContract format_contract(const MatmulFormat format) noexcept {
  switch (format) {
  case MatmulFormat::Mxfp8E4M3W8A8:
    return {format,
            ScalarType::E4M3Fn,
            ScalarType::E4M3Fn,
            BlockKind::Mxfp8E4M3Block32,
            BlockKind::Mxfp8E4M3Block32,
            BlockScaleType::E8M0,
            BlockScaleType::E8M0,
            32U,
            32U,
            8U,
            8U,
            false,
            false};
  case MatmulFormat::Mxfp6E3M2W6A6:
    return {format,
            ScalarType::E3M2,
            ScalarType::E3M2,
            BlockKind::Mxfp6E3M2Block32,
            BlockKind::Mxfp6E3M2Block32,
            BlockScaleType::E8M0,
            BlockScaleType::E8M0,
            32U,
            32U,
            6U,
            6U,
            false,
            false};
  case MatmulFormat::Nvfp4W4A16:
    return {format,
            ScalarType::E2M1,
            ScalarType::Bf16,
            BlockKind::Nvfp4E2M1Block16,
            BlockKind::None,
            BlockScaleType::E4M3FnWithFp32TensorScale,
            BlockScaleType::None,
            16U,
            0U,
            4U,
            16U,
            true,
            false};
  case MatmulFormat::Nvfp4W4A4:
    return {format,
            ScalarType::E2M1,
            ScalarType::E2M1,
            BlockKind::Nvfp4E2M1Block16,
            BlockKind::Nvfp4E2M1Block16,
            BlockScaleType::E4M3FnWithFp32TensorScale,
            BlockScaleType::E4M3FnWithFp32TensorScale,
            16U,
            16U,
            4U,
            4U,
            true,
            true};
  case MatmulFormat::Mxfp4W4A4:
    return {format,
            ScalarType::E2M1,
            ScalarType::E2M1,
            BlockKind::Mxfp4E2M1Block32,
            BlockKind::Mxfp4E2M1Block32,
            BlockScaleType::E8M0,
            BlockScaleType::E8M0,
            32U,
            32U,
            4U,
            4U,
            false,
            false};
  case MatmulFormat::Fp8OuterE4M3W8A8:
    return {format,
            ScalarType::E4M3Fn,
            ScalarType::E4M3Fn,
            BlockKind::None,
            BlockKind::None,
            BlockScaleType::Fp32OuterVector,
            BlockScaleType::Fp32OuterVector,
            0U,
            0U,
            8U,
            8U,
            false,
            false};
  }
  return {format,
          ScalarType::Bf16,
          ScalarType::Bf16,
          BlockKind::None,
          BlockKind::None,
          BlockScaleType::None,
          BlockScaleType::None,
          0U,
          0U,
          0U,
          0U,
          false,
          false};
}

constexpr BlockLayout
default_activation_layout(const MatmulFormat format) noexcept {
  return format == MatmulFormat::Nvfp4W4A16 ||
                 format == MatmulFormat::Fp8OuterE4M3W8A8
             ? BlockLayout::RowMajor
             : BlockLayout::RowMajorBlockScaled;
}

constexpr ProviderRequest make_provider_request(const MatmulFormat format,
                                                const ExactTarget target,
                                                const uint64_t m,
                                                const uint64_t n,
                                                const uint64_t k) noexcept {
  return {format,
          format == MatmulFormat::Fp8OuterE4M3W8A8
              ? BlockLayout::RowMajor
              : BlockLayout::RowMajorBlockScaled,
          default_activation_layout(format),
          target,
          m,
          n,
          k,
          AccumulationType::Fp32,
          OutputType::Bf16Rne};
}

constexpr bool is_rdna_target(const ExactTarget target) noexcept {
  return target == ExactTarget::Gfx1030 || target == ExactTarget::Gfx1201;
}

constexpr bool is_block_scaled_layout(const BlockLayout layout) noexcept {
  return layout == BlockLayout::RowMajorBlockScaled ||
         layout == BlockLayout::ConsumerTiledBlockScaled;
}

constexpr PreparedProviderPlan
rejected_plan(const ProviderRequest &request,
              const ProviderRejection rejection) noexcept {
  return {ProviderKind::Unsupported,
          rejection,
          request.format,
          format_contract(request.format),
          request.weight_layout,
          request.activation_layout,
          target_architecture(request.target),
          request.target,
          TilePolicy::None,
          ActivationPack::NoneBf16,
          InnerProduct::None,
          request.m,
          request.n,
          request.k,
          request.accumulation,
          request.output};
}

constexpr PreparedProviderPlan
with_execution_semantics(const PreparedProviderPlan &plan,
                         const ProviderKind provider, const TilePolicy tile,
                         const InnerProduct inner_product) noexcept {
  return {provider,
          plan.rejection,
          plan.format,
          plan.block_contract,
          plan.weight_layout,
          plan.activation_layout,
          plan.architecture,
          plan.target,
          tile,
          plan.activation_pack,
          inner_product,
          plan.m,
          plan.n,
          plan.k,
          plan.accumulation,
          plan.output};
}

constexpr bool
gfx1201_mxfp8_wmma_n64_shape(const ProviderRequest &request) noexcept {
  const bool phase63_wide_family = request.n >= 1024U;
  const bool phase65_complete_row_family =
      (request.m % 128U) == 0U && request.n >= 64U;
  return request.target == ExactTarget::Gfx1201 && request.m >= 128U &&
         request.k >= 2048U && request.n <= 32768U && (request.n % 64U) == 0U &&
         (phase63_wide_family || phase65_complete_row_family);
}

constexpr bool
gfx1201_mxfp8_wmma_n128_shape(const ProviderRequest &request) noexcept {
  return gfx1201_mxfp8_wmma_n64_shape(request) && (request.m % 128U) == 0U &&
         (request.n % 128U) == 0U;
}

constexpr bool
gfx1201_mxfp6_wmma_via_e4m3_shape(const ProviderRequest &request) noexcept {
  return request.target == ExactTarget::Gfx1201 && request.m >= 17U &&
         request.k >= 2048U && request.n >= 1024U && request.n <= 32768U;
}

constexpr PreparedProviderPlan
prepare_provider_plan(const ProviderRequest &request) noexcept {
  const FormatContract contract = format_contract(request.format);
  if (request.target == ExactTarget::Unknown) {
    return rejected_plan(request, ProviderRejection::UnknownTarget);
  }
  if (!is_rdna_target(request.target)) {
    return rejected_plan(request, ProviderRejection::UnsupportedTarget);
  }
  if (request.m == 0U || request.n == 0U || request.k == 0U) {
    return rejected_plan(request, ProviderRejection::EmptyDimension);
  }
  const bool k_requires_complete_blocks =
      request.format == MatmulFormat::Mxfp8E4M3W8A8 ||
      request.format == MatmulFormat::Mxfp6E3M2W6A6;
  const bool outer_vector_format =
      request.format == MatmulFormat::Fp8OuterE4M3W8A8;
  if ((!outer_vector_format && contract.weight_block_size == 0U) ||
      (k_requires_complete_blocks &&
       (request.k % contract.weight_block_size) != 0U)) {
    return rejected_plan(request, ProviderRejection::KNotBlockAligned);
  }
  if (request.accumulation != AccumulationType::Fp32 ||
      request.output != OutputType::Bf16Rne) {
    return rejected_plan(request, ProviderRejection::UnsupportedNumerics);
  }
  const bool valid_weight_layout =
      outer_vector_format ? request.weight_layout == BlockLayout::RowMajor
                          : is_block_scaled_layout(request.weight_layout);
  const bool valid_activation_layout =
      request.format == MatmulFormat::Nvfp4W4A16 || outer_vector_format
          ? request.activation_layout == BlockLayout::RowMajor
          : is_block_scaled_layout(request.activation_layout);
  if (!valid_weight_layout || !valid_activation_layout) {
    return rejected_plan(request, ProviderRejection::InvalidLayout);
  }

  const TilePolicy base_tile =
      request.m == 1U ? TilePolicy::DecodeRowReduction : TilePolicy::BlockRow8;
  switch (request.format) {
  case MatmulFormat::Fp8OuterE4M3W8A8:
    if (request.target != ExactTarget::Gfx1030) {
      return rejected_plan(request, ProviderRejection::UnsupportedTarget);
    }
    return {ProviderKind::Fp8OuterGfx1030Software,
            ProviderRejection::None,
            request.format,
            contract,
            request.weight_layout,
            request.activation_layout,
            target_architecture(request.target),
            request.target,
            request.m == 1U ? TilePolicy::DecodeRowReduction
                            : TilePolicy::BlockTiled16x16,
            ActivationPack::Fp8E4M3Outer,
            InnerProduct::E4M3OuterFp32,
            request.m,
            request.n,
            request.k,
            request.accumulation,
            request.output};
  case MatmulFormat::Mxfp8E4M3W8A8:
    if (gfx1201_mxfp8_wmma_n128_shape(request)) {
      return {ProviderKind::Mxfp8Gfx1201Wmma,
              ProviderRejection::None,
              request.format,
              contract,
              request.weight_layout,
              request.activation_layout,
              target_architecture(request.target),
              request.target,
              TilePolicy::Wmma128x128x32,
              ActivationPack::Mxfp8E4M3Block32,
              InnerProduct::E4M3WmmaFp32,
              request.m,
              request.n,
              request.k,
              request.accumulation,
              request.output};
    }
    if (gfx1201_mxfp8_wmma_n64_shape(request)) {
      return {ProviderKind::Mxfp8Gfx1201Wmma,
              ProviderRejection::None,
              request.format,
              contract,
              request.weight_layout,
              request.activation_layout,
              target_architecture(request.target),
              request.target,
              TilePolicy::Wmma128x64x32,
              ActivationPack::Mxfp8E4M3Block32,
              InnerProduct::E4M3WmmaFp32,
              request.m,
              request.n,
              request.k,
              request.accumulation,
              request.output};
    }
    return {ProviderKind::Mxfp8Block32,
            ProviderRejection::None,
            request.format,
            contract,
            request.weight_layout,
            request.activation_layout,
            target_architecture(request.target),
            request.target,
            base_tile,
            ActivationPack::Mxfp8E4M3Block32,
            InnerProduct::DecodedBlockScaledFp32,
            request.m,
            request.n,
            request.k,
            request.accumulation,
            request.output};
  case MatmulFormat::Mxfp6E3M2W6A6:
    if (gfx1201_mxfp6_wmma_via_e4m3_shape(request)) {
      return {ProviderKind::Mxfp6Gfx1201WmmaViaE4M3,
              ProviderRejection::None,
              request.format,
              contract,
              request.weight_layout,
              request.activation_layout,
              target_architecture(request.target),
              request.target,
              TilePolicy::Wmma128x64x32,
              ActivationPack::Mxfp6E3M2Block32,
              InnerProduct::E3M2ViaE4M3WmmaFp32,
              request.m,
              request.n,
              request.k,
              request.accumulation,
              request.output};
    }
    return {ProviderKind::Mxfp6Block32,
            ProviderRejection::None,
            request.format,
            contract,
            request.weight_layout,
            request.activation_layout,
            target_architecture(request.target),
            request.target,
            base_tile,
            ActivationPack::Mxfp6E3M2Block32,
            InnerProduct::DecodedBlockScaledFp32,
            request.m,
            request.n,
            request.k,
            request.accumulation,
            request.output};
  case MatmulFormat::Nvfp4W4A16:
    return {ProviderKind::Nvfp4W4A16Block16,
            ProviderRejection::None,
            request.format,
            contract,
            request.weight_layout,
            request.activation_layout,
            target_architecture(request.target),
            request.target,
            request.m == 1U ? TilePolicy::DecodeRowReduction
                            : TilePolicy::PackedRow8,
            ActivationPack::NoneBf16,
            InnerProduct::E2M1Bf16Fp32,
            request.m,
            request.n,
            request.k,
            request.accumulation,
            request.output};
  case MatmulFormat::Nvfp4W4A4:
    return {ProviderKind::Nvfp4W4A4Block16,
            ProviderRejection::None,
            request.format,
            contract,
            request.weight_layout,
            request.activation_layout,
            target_architecture(request.target),
            request.target,
            request.m == 1U ? TilePolicy::DecodeRowReduction
                            : TilePolicy::PackedRow8,
            ActivationPack::Nvfp4E2M1Block16,
            InnerProduct::E2M1BlockScaledFp32,
            request.m,
            request.n,
            request.k,
            request.accumulation,
            request.output};
  case MatmulFormat::Mxfp4W4A4:
    return {ProviderKind::Mxfp4W4A4Block32,
            ProviderRejection::None,
            request.format,
            contract,
            request.weight_layout,
            request.activation_layout,
            target_architecture(request.target),
            request.target,
            request.m == 1U ? TilePolicy::DecodeRowReduction
                            : TilePolicy::PackedRow8,
            ActivationPack::Mxfp4E2M1Block32,
            InnerProduct::E2M1BlockScaledFp32,
            request.m,
            request.n,
            request.k,
            request.accumulation,
            request.output};
  }
  return rejected_plan(request, ProviderRejection::UnsupportedNumerics);
}

} // namespace sllm_lowp

#endif // SLLM_LOW_PRECISION_MATMUL_PROVIDER_HPP
