#include <limits>
#include <lowp/detail/lowp_api_internal.hpp>
#include <lowp/lowp.h>

namespace {
using namespace sllm_lowp;
using namespace sllm_matmul_kernel;
bool multiply(uint64_t a, uint64_t b, uint64_t *out) noexcept {
  if (a && b > UINT64_MAX / a)
    return false;
  *out = a * b;
  return true;
}
bool add(uint64_t a, uint64_t b, uint64_t *out) noexcept {
  if (b > UINT64_MAX - a)
    return false;
  *out = a + b;
  return true;
}
SelectorDecision select(const PreparedProviderPlan &p) noexcept {
  const char *t = lowp_target_name(static_cast<lowp_target_t>(p.target));
  KernelVariant v = KernelVariant::Unspecialized;
  switch (p.format) {
  case MatmulFormat::Mxfp8E4M3W8A8:
    v = select_mxfp8_variant(p.m, p.k, p.n, t);
    break;
  case MatmulFormat::Mxfp6E3M2W6A6:
    v = select_mxfp6_variant(p.m, p.k, p.n, t);
    break;
  case MatmulFormat::Nvfp4W4A4:
    return select_nvfp4_w4a4_decision(p.m, p.k, p.n, t);
  case MatmulFormat::Mxfp4W4A4:
    v = select_mxfp4_variant(p.m);
    break;
  case MatmulFormat::Fp8OuterE4M3W8A8:
    return select_fp8_software_decision(p.m, p.k, p.n, t, false);
  case MatmulFormat::Nvfp4W4A16:
  case MatmulFormat::Mxfp8E4M3W8A16:
  case MatmulFormat::Mxfp6E3M2W6A16:
    return make_selector_decision(KernelVariant::Unspecialized, false, false,
                                  false, kSelectorReasonUnsupported);
  }
  return make_selector_decision(v, true, true, true, kSelectorReasonAdopted);
}
} // namespace

extern "C" uint32_t lowp_version(void) { return LOWP_ABI_VERSION; }
extern "C" lowp_target_t lowp_target_from_name(const char *name) {
  return static_cast<lowp_target_t>(sllm_lowp::exact_target_from_name(name));
}
extern "C" const char *lowp_target_name(lowp_target_t target) {
  switch (target) {
  case LOWP_TARGET_GFX1030:
    return "gfx1030";
  case LOWP_TARGET_GFX1201:
    return "gfx1201";
  case LOWP_TARGET_GFX942_SRAMECC_ON_XNACK_OFF:
    return "gfx942:sramecc+:xnack-";
  default:
    return "";
  }
}
extern "C" lowp_status_t lowp_get_format_info(lowp_format_t format,
                                              lowp_format_info_t *info) {
  if (!info)
    return LOWP_INVALID_ARGUMENT;
  *info = {};
  if (format == LOWP_MXFP4_W4A6_V1) {
    *info = {LOWP_MXFP4_W4A6_CONTRACT_VERSION,
             4,
             6,
             32,
             32,
             static_cast<uint32_t>(sllm_lowp::BlockScaleType::E8M0),
             static_cast<uint32_t>(sllm_lowp::BlockScaleType::E8M0),
             0,
             0};
    return LOWP_SUCCESS;
  }
  if (format > 7 || format == 4 || format == LOWP_RETIRED_NVFP4_W4A16 ||
      format == LOWP_RETIRED_MXFP8_E4M3_W8A16 ||
      format == LOWP_RETIRED_MXFP6_E3M2_W6A16)
    return LOWP_NOT_SUPPORTED;
  const auto c =
      sllm_lowp::format_contract(static_cast<sllm_lowp::MatmulFormat>(format));
  *info = {1,
           c.weight_bits,
           c.activation_bits,
           c.weight_block_size,
           c.activation_block_size,
           static_cast<uint32_t>(c.weight_scale),
           static_cast<uint32_t>(c.activation_scale),
           static_cast<uint32_t>(c.weight_has_tensor_scale),
           static_cast<uint32_t>(c.activation_has_tensor_scale)};
  return LOWP_SUCCESS;
}

namespace sllm_lowp {
lowp_status_t make_plan_impl(const lowp_matmul_request_t *r,
                             lowp_matmul_plan_t *p, bool legacy) noexcept {
  if (!p)
    return LOWP_INVALID_ARGUMENT;
  *p = {};
  p->struct_size = sizeof(*p);
  p->version = LOWP_ABI_VERSION;
  p->reason = "invalid request";
  if (!r || r->struct_size != sizeof(*r) || r->version != LOWP_ABI_VERSION)
    return LOWP_INVALID_ARGUMENT;
  p->request = *r;
  if (r->format > 7 || (r->format == 4 && !legacy)) {
    p->rejection =
        static_cast<uint32_t>(ProviderRejection::UnsupportedNumerics);
    p->reason = r->format == LOWP_MXFP4_W4A6_V1
                    ? "MXFP4 W4A6 v1 is not implemented"
                    : "format is not public/supported";
    return LOWP_NOT_SUPPORTED;
  }
  if (r->target > LOWP_TARGET_GFX942_SRAMECC_ON_XNACK_OFF) {
    p->rejection = static_cast<uint32_t>(ProviderRejection::UnknownTarget);
    p->reason = "unknown exact target";
    return LOWP_NOT_SUPPORTED;
  }
  if (r->weight_layout > LOWP_CONSUMER_TILED_BLOCK_SCALED ||
      r->activation_layout > LOWP_CONSUMER_TILED_BLOCK_SCALED) {
    p->rejection = static_cast<uint32_t>(ProviderRejection::InvalidLayout);
    p->reason = "invalid layout";
    return LOWP_INVALID_ARGUMENT;
  }
  ProviderRequest request{static_cast<MatmulFormat>(r->format),
                          static_cast<BlockLayout>(r->weight_layout),
                          static_cast<BlockLayout>(r->activation_layout),
                          static_cast<ExactTarget>(r->target),
                          r->m,
                          r->n,
                          r->k,
                          AccumulationType::Fp32,
                          OutputType::Bf16Rne};
  const auto prepared = prepare_provider_plan(request);
  p->provider = static_cast<uint32_t>(prepared.provider);
  p->rejection = static_cast<uint32_t>(prepared.rejection);
  if (!prepared.supported()) {
    p->reason = "provider rejected target, shape or layout";
    return LOWP_NOT_SUPPORTED;
  }
  const auto decision = select(prepared);
  p->variant = static_cast<uint32_t>(decision.variant);
  p->reason = decision.reason;
  p->selector_supported = decision.supported;
  p->selector_enabled = decision.enabled;
  p->adopted = decision.adopted;
  // Selector flags describe adoption/benchmark eligibility. The historical
  // runtime also executes a concrete baseline when these flags are false.
  const auto concrete = concrete_provider_plan(prepared, decision.variant);
  if (!concrete) {
    p->supported = 0;
    p->reason = "concrete variant does not match format/shape";
    return LOWP_NOT_SUPPORTED;
  }
  p->supported = 1U;
  p->provider = static_cast<uint32_t>(concrete->provider);
  p->tile = static_cast<uint32_t>(concrete->tile);
  p->inner_product = static_cast<uint32_t>(concrete->inner_product);
  p->activation_pack = static_cast<uint32_t>(concrete->activation_pack);
  const auto c = concrete->block_contract;
  uint64_t count = 0, bits = 0, activation_input_bytes = 0;
  if (!multiply(r->m, r->k, &count) ||
      !multiply(count, 2, &activation_input_bytes))
    return LOWP_INVALID_ARGUMENT;
  (void)activation_input_bytes;
  if (!multiply(r->n, r->k, &count) || !multiply(r->m, r->n, &bits) ||
      !multiply(bits, 2, &p->output_bytes))
    return LOWP_INVALID_ARGUMENT;
  if (r->format == LOWP_NVFP4_W4A4) {
    // Historical NVFP4 weights pack the complete logical plane continuously.
    p->weight_value_bytes = count / 2U + (count % 2U != 0U ? 1U : 0U);
  } else {
    uint64_t row_bits = 0U;
    if (!multiply(r->k, c.weight_bits, &row_bits) ||
        !multiply(r->n, row_bits / 8U + (row_bits % 8U != 0U ? 1U : 0U),
                  &p->weight_value_bytes))
      return LOWP_INVALID_ARGUMENT;
  }
  if (c.weight_block_size) {
    if (!multiply(r->n,
                  (r->k / c.weight_block_size +
                   (r->k % c.weight_block_size != 0U ? 1U : 0U)),
                  &p->weight_scale_bytes))
      return LOWP_INVALID_ARGUMENT;
  } else if (!multiply(r->n, 4, &p->weight_scale_bytes))
    return LOWP_INVALID_ARGUMENT;
  if (r->format == LOWP_FP8_OUTER_E4M3_W8A8) {
    if (!multiply(r->m, r->k, &p->activation_value_bytes) ||
        !add(p->activation_value_bytes, 3, &p->activation_scale_offset) ||
        !multiply(r->m, 4, &p->activation_scale_bytes))
      return LOWP_INVALID_ARGUMENT;
    p->activation_scale_offset &= ~UINT64_C(3);
    if (!add(p->activation_scale_offset, p->activation_scale_bytes,
             &p->workspace_bytes))
      return LOWP_INVALID_ARGUMENT;
  } else if (c.activation_block_size) {
    uint64_t row_bits = 0U;
    if (!multiply(r->k, c.activation_bits, &row_bits) ||
        !multiply(r->m, row_bits / 8U + (row_bits % 8U != 0U ? 1U : 0U),
                  &p->activation_value_bytes))
      return LOWP_INVALID_ARGUMENT;
    p->activation_scale_offset = p->activation_value_bytes;
    if (!multiply(r->m,
                  (r->k / c.activation_block_size +
                   (r->k % c.activation_block_size != 0U ? 1U : 0U)),
                  &p->activation_scale_bytes) ||
        !add(p->activation_scale_offset, p->activation_scale_bytes,
             &p->workspace_bytes))
      return LOWP_INVALID_ARGUMENT;
  }
  const bool split4 =
      r->format == LOWP_NVFP4_W4A4 &&
      ((decision.variant == KernelVariant::Nvfp4W4A4PrefillDp4a64x64 &&
        r->target == LOWP_TARGET_GFX1030 &&
        phase78_gfx1030_nvfp4_w4a4_split4_shape(r->m, r->k, r->n)) ||
       (decision.variant == KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x64 &&
        r->target == LOWP_TARGET_GFX1201 &&
        phase78_gfx1201_nvfp4_w4a4_split4_shape(r->m, r->k, r->n)));
  if (split4) {
    uint64_t total = 0;
    if (!phase78_nvfp4_w4a4_split4_workspace(r->m, r->k, r->n,
                                             &p->scratch_offset, &total) ||
        p->scratch_offset != p->workspace_bytes)
      return LOWP_INVALID_ARGUMENT;
    p->scratch_bytes = total - p->scratch_offset;
    p->workspace_bytes = total;
    p->launch_flags |= 1U;
  }
  if (decision.variant == KernelVariant::Nvfp4W4A4PrefillGfx1201F16Staging) {
    F16StagingWorkspaceLayout layout{};
    if (!f16_staging_workspace_layout(r->m, r->k, r->n, &layout) ||
        !qwen38_f16_staging_workspace_reservation(
            r->m, &p->staging_workspace_bytes) ||
        p->staging_workspace_bytes < layout.total_bytes)
      return LOWP_INVALID_ARGUMENT;
    p->launch_flags |= 2U;
  }
  return LOWP_SUCCESS;
}
lowp_status_t plan_internal(const lowp_matmul_request_t *r,
                            lowp_matmul_plan_t *p, bool legacy) noexcept {
  const auto status = make_plan_impl(r, p, legacy);
  if (p && status != LOWP_SUCCESS)
    p->supported = 0;
  return status;
}
} // namespace sllm_lowp
extern "C" lowp_status_t lowp_matmul_plan(const lowp_matmul_request_t *r,
                                          lowp_matmul_plan_t *p) {
  return sllm_lowp::plan_internal(r, p, false);
}
extern "C" uint64_t lowp_supported_formats(lowp_target_t target) {
  uint64_t result = 0;
  for (uint32_t f = 0; f <= 7; ++f) {
    if (f == 4)
      continue;
    const auto req = sllm_lowp::make_provider_request(
        static_cast<sllm_lowp::MatmulFormat>(f),
        static_cast<sllm_lowp::ExactTarget>(target), 1, 256, 256);
    lowp_matmul_request_t r{sizeof(r),
                            LOWP_ABI_VERSION,
                            f,
                            target,
                            static_cast<uint32_t>(req.weight_layout),
                            static_cast<uint32_t>(req.activation_layout),
                            1,
                            256,
                            256};
    lowp_matmul_plan_t p{};
    if (lowp_matmul_plan(&r, &p) == LOWP_SUCCESS)
      result |= UINT64_C(1) << f;
  }
  return result;
}
