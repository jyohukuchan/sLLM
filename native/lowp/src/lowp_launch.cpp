#include <cstring>
#include <hip/hip_runtime.h>
#include <lowp/detail/lowp_api_internal.hpp>
#include <lowp/lowp.h>

namespace {
using namespace sllm_lowp;
using namespace sllm_matmul_kernel;
lowp_status_t valid(const lowp_matmul_plan_t *p, bool legacy) noexcept {
  if (!p || p->struct_size != sizeof(*p) || p->version != LOWP_ABI_VERSION ||
      p->request.struct_size != sizeof(p->request) ||
      p->request.version != LOWP_ABI_VERSION || !p->supported ||
      !p->request.m || !p->request.n || !p->request.k)
    return LOWP_INVALID_ARGUMENT;
  if (p->request.format > 7 || (p->request.format == 4 && !legacy))
    return LOWP_NOT_SUPPORTED;
#ifdef LOWP_COMPILED_TARGET
  if (std::strcmp(lowp_target_name(p->request.target), LOWP_COMPILED_TARGET) !=
      0)
    return LOWP_NOT_SUPPORTED;
#endif
  const ProviderRequest r{
      static_cast<MatmulFormat>(p->request.format),
      static_cast<BlockLayout>(p->request.weight_layout),
      static_cast<BlockLayout>(p->request.activation_layout),
      static_cast<ExactTarget>(p->request.target),
      p->request.m,
      p->request.n,
      p->request.k,
      AccumulationType::Fp32,
      OutputType::Bf16Rne};
  const auto provider = prepare_provider_plan(r);
  const auto concrete =
      concrete_provider_plan(provider, static_cast<KernelVariant>(p->variant));
  if (!provider.supported() || !concrete ||
      static_cast<uint32_t>(concrete->provider) != p->provider ||
      static_cast<uint32_t>(concrete->tile) != p->tile ||
      static_cast<uint32_t>(concrete->inner_product) != p->inner_product)
    return LOWP_INVALID_ARGUMENT;
  return LOWP_SUCCESS;
}
lowp_status_t quantize(const lowp_matmul_plan_t *p, const uint16_t *a, void *v,
                       void *s, const float *tensor, void *stream,
                       bool legacy) noexcept {
  const auto status = valid(p, legacy);
  if (status != LOWP_SUCCESS)
    return status;
  if (!a || !v || !s)
    return LOWP_INVALID_ARGUMENT;
  const auto m = p->request.m, k = p->request.k;
  auto q = static_cast<uint8_t *>(v), scales = static_cast<uint8_t *>(s);
  const auto st = static_cast<hipStream_t>(stream);
  switch (p->request.format) {
  case LOWP_MXFP8_E4M3_W8A8:
    return launch_mxfp8_quantize(a, q, scales, m, k, st);
  case LOWP_MXFP6_E3M2_W6A6:
    return launch_mxfp6_quantize(a, q, scales, m, k, st);
  case LOWP_NVFP4_W4A4:
    if (!tensor)
      return LOWP_INVALID_ARGUMENT;
    return launch_nvfp4_quantize(a, q, scales, tensor, m, k, st);
  case 4:
    return launch_mxfp4_quantize(a, q, scales, m, k, st);
  case LOWP_FP8_OUTER_E4M3_W8A8:
    return launch_fp8_quantize(a, q, static_cast<float *>(s), m, k, false, st);
  default:
    return LOWP_NOT_SUPPORTED;
  }
}
} // namespace
extern "C" lowp_status_t lowp_quantize_activation(const lowp_matmul_plan_t *p,
                                                  const uint16_t *a,
                                                  void *values, void *scales,
                                                  const float *tensor,
                                                  void *stream) {
  return quantize(p, a, values, scales, tensor, stream, false);
}
namespace sllm_lowp {
lowp_status_t launch_internal(const lowp_matmul_plan_t *p,
                              const lowp_matmul_buffers_t *b, void *stream,
                              bool legacy) noexcept {
  const auto status = valid(p, legacy);
  if (status != LOWP_SUCCESS)
    return status;
  if (!b || b->struct_size != sizeof(*b) ||
      (b->flags & ~static_cast<uint32_t>(LOWP_ACTIVATION_PREQUANTIZED)) ||
      !b->activation || !b->weight || !b->weight_scales || !b->output)
    return LOWP_INVALID_ARGUMENT;
  const auto m = p->request.m, n = p->request.n, k = p->request.k;
  const auto variant = static_cast<KernelVariant>(p->variant);
  const auto stream_handle = static_cast<hipStream_t>(stream);
  const auto weight = static_cast<const uint8_t *>(b->weight);
  const auto weight_scales = static_cast<const uint8_t *>(b->weight_scales);
  const auto bf16 = static_cast<const uint16_t *>(b->activation);
  const uint8_t *activation = nullptr, *activation_scales = nullptr;
  if (b->flags & LOWP_ACTIVATION_PREQUANTIZED) {
    if (!b->activation_scales)
      return LOWP_INVALID_ARGUMENT;
    activation = static_cast<const uint8_t *>(b->activation);
    activation_scales = static_cast<const uint8_t *>(b->activation_scales);
  } else {
    if (!b->workspace || b->workspace_bytes < p->workspace_bytes)
      return LOWP_OUT_OF_MEMORY;
    auto *values = static_cast<uint8_t *>(b->workspace);
    auto *scales = values + p->activation_scale_offset;
    const auto q = quantize(p, bf16, values, scales, b->activation_tensor_scale,
                            stream, legacy);
    if (q != LOWP_SUCCESS)
      return q;
    activation = values;
    activation_scales = scales;
  }
  switch (p->request.format) {
  case LOWP_MXFP8_E4M3_W8A8:
    return launch_mxfp8_w8a8(activation, activation_scales, weight,
                             weight_scales, b->output, m, k, n, variant,
                             stream_handle);
  case LOWP_MXFP6_E3M2_W6A6:
    return launch_mxfp6_w6a6(activation, activation_scales, weight,
                             weight_scales, b->output, m, k, n, variant,
                             stream_handle);
  case 4:
    return launch_mxfp4_w4a4(activation, activation_scales, weight,
                             weight_scales, b->output, m, k, n, variant,
                             stream_handle);
  case LOWP_NVFP4_W4A4: {
    if (!b->weight_tensor_scale || !b->activation_tensor_scale)
      return LOWP_INVALID_ARGUMENT;
    if (p->launch_flags & 1U) {
      if (!b->workspace || b->workspace_bytes < p->workspace_bytes)
        return LOWP_OUT_OF_MEMORY;
      float *partial = reinterpret_cast<float *>(
          static_cast<uint8_t *>(b->workspace) + p->scratch_offset);
      if (p->request.target == LOWP_TARGET_GFX1030)
        return launch_nvfp4_w4a4_prefill_dp4a_short_split4(
            activation, activation_scales, weight, weight_scales,
            b->weight_tensor_scale, b->activation_tensor_scale, b->output,
            partial, m, k, n, stream_handle);
      return launch_nvfp4_w4a4_prefill_gfx1201_wmma128x64_split4(
          activation, activation_scales, weight, weight_scales,
          b->weight_tensor_scale, b->activation_tensor_scale, b->output,
          partial, m, k, n, stream_handle);
    }
    if (p->launch_flags & 2U) {
      F16StagingWorkspaceLayout l{};
      if (!b->staging_workspace ||
          b->staging_workspace_bytes < p->staging_workspace_bytes ||
          !f16_staging_workspace_layout(m, k, n, &l) || !b->f16_gemm)
        return LOWP_INVALID_ARGUMENT;
      auto *base = static_cast<uint8_t *>(b->staging_workspace);
      auto *aa = reinterpret_cast<uint16_t *>(base + l.activation_offset);
      auto *ww = reinterpret_cast<uint16_t *>(base + l.weight_offset);
      auto *oo = reinterpret_cast<float *>(base + l.output_offset);
      auto rc = launch_nvfp4_block16_to_fp16_staging(
          activation, activation_scales, aa, m, k, stream_handle);
      if (rc == hipSuccess)
        rc = launch_nvfp4_block16_to_fp16_staging(weight, weight_scales, ww, n,
                                                  k, stream_handle);
      if (rc != hipSuccess)
        return rc;
      const auto gemm =
          b->f16_gemm(b->f16_gemm_user, aa, ww, oo, m, n, k, stream);
      if (gemm != LOWP_SUCCESS)
        return gemm;
      return launch_nvfp4_tensor_scale_epilogue(oo, b->weight_tensor_scale,
                                                b->activation_tensor_scale,
                                                b->output, m, n, stream_handle);
    }
    return launch_nvfp4_w4a4(activation, activation_scales, weight,
                             weight_scales, b->weight_tensor_scale,
                             b->activation_tensor_scale, b->output, m, k, n,
                             variant, stream_handle);
  }
  case LOWP_FP8_OUTER_E4M3_W8A8: {
    const auto *as = reinterpret_cast<const float *>(activation_scales);
    const auto *ws = static_cast<const float *>(b->weight_scales);
    switch (variant) {
    case KernelVariant::Fp8Emulation:
      return launch_fp8_emulation(activation, as, weight, ws, b->output, m, k,
                                  n, stream_handle);
    case KernelVariant::Fp8OuterPrefillTiled16:
      return launch_fp8_outer_prefill_tiled16(
          activation, as, weight, ws, b->output, m, k, n, stream_handle);
    case KernelVariant::Fp8OuterPrefillGfx1030Half2_128x64:
      return launch_fp8_outer_prefill_gfx1030_half2_128x64(
          activation, as, weight, ws, b->output, m, k, n, stream_handle);
    case KernelVariant::Fp8OuterPrefillGfx1030Half2_64x64:
      return launch_fp8_outer_prefill_gfx1030_half2_64x64(
          activation, as, weight, ws, b->output, m, k, n, stream_handle);
    case KernelVariant::Fp8OuterDecodeGfx1030Half2Wave4Col32:
      return launch_fp8_outer_decode_gfx1030_half2_wave4col32(
          activation, as, weight, ws, b->output, m, k, n, stream_handle);
    case KernelVariant::Fp8OuterDecodeGfx1030Dword8Wave4Col32:
      return launch_fp8_outer_decode_gfx1030_dword8_wave4col32(
          activation, as, weight, ws, b->output, m, k, n, stream_handle);
    case KernelVariant::Fp8OuterDecodeGfx1030LdsLutWave4Col32:
      return launch_fp8_outer_decode_gfx1030_lds_lut_wave4col32(
          activation, as, weight, ws, b->output, m, k, n, stream_handle);
    case KernelVariant::Fp8OuterDecodeGfx1030FusedM2_4:
      return launch_fp8_outer_decode_gfx1030_fused_m2_4(
          activation, as, weight, ws, b->output, m, k, n, stream_handle);
    case KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave4Col32:
      return launch_fp8_outer_decode_gfx1030_activation_shared_wave4col32(
          activation, as, weight, ws, b->output, m, k, n, stream_handle);
    case KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave8Col64:
      return launch_fp8_outer_decode_gfx1030_activation_shared_wave8col64(
          activation, as, weight, ws, b->output, m, k, n, stream_handle);
    default:
      return LOWP_NOT_SUPPORTED;
    }
  }
  default:
    return LOWP_NOT_SUPPORTED;
  }
}
} // namespace sllm_lowp
extern "C" lowp_status_t lowp_matmul_launch(const lowp_matmul_plan_t *p,
                                            const lowp_matmul_buffers_t *b,
                                            void *stream) {
  return sllm_lowp::launch_internal(p, b, stream, false);
}
