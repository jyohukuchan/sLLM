#include "matmul_kernel_internal.hpp"

#include <cstdlib>
#include <iostream>
#include <string_view>
#include <utility>

namespace {

using sllm_matmul_kernel::KernelVariant;

constexpr const char *kSmallMRowGrid = "SLLM_NVFP4_W4A4_SMALL_M_ROWGRID";
constexpr const char *kBaseline = "SLLM_NVFP4_W4A4_FORCE_BASELINE";

void clear_controls() {
  unsetenv(kSmallMRowGrid);
  unsetenv(kBaseline);
  unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_COMPENSATED");
  unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_WMMA_COMPENSATED");
  unsetenv("SLLM_NVFP4_W4A4_SMALL_M_ROWGRID_GFX1201");
  unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_ROW8");
  unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_COL8");
  unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_DP4A");
  unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA");
  unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_F16_STAGING");
}

bool check_default(const uint64_t m, const uint64_t k, const uint64_t n,
                   const char *const target) {
  using namespace sllm_matmul_kernel;
  const SelectorDecision decision = select_nvfp4_w4a4_decision(m, k, n, target);
  return decision.variant == KernelVariant::Nvfp4W4A4SmallMVgprReuse &&
         decision.supported && decision.enabled && decision.adopted &&
         decision.runnable() &&
         std::string_view(decision.reason) == kSelectorReasonAdopted &&
         std::string_view(logical_kernel_id(decision.variant)) ==
             "matmul.nvfp4.w4a4.small_m.vgpr_reuse.v1" &&
         std::string_view(device_symbol(decision.variant)) ==
             "sllm_nvfp4_w4a4_small_m_vgpr_reuse_v1" &&
         grid_size_x(decision.variant, m, n) == (n + 31U) / 32U &&
         workgroup_size_x(decision.variant) == 256U;
}

bool check_kahan_stage_boundary(const uint64_t m, const uint64_t k,
                                const uint64_t n) {
  using namespace sllm_matmul_kernel;
  const SelectorDecision decision =
      select_nvfp4_w4a4_decision(m, k, n, "gfx1201");
  const char *const expected_symbol =
      m >= 256U ? "sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_kahan_stage64_v1"
                : "sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_kahan_v1";
  return decision.variant == KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan &&
         decision.supported && decision.enabled && decision.adopted &&
         std::string_view(device_symbol_for_target(decision.variant, "gfx1201",
                                                   m, k, n)) == expected_symbol;
}

} // namespace

int main() {
  using namespace sllm_matmul_kernel;
  clear_controls();
  bool ok = true;

  for (const char *const target : {"gfx1030", "gfx1201"}) {
    for (const auto &shape : {std::pair<uint64_t, uint64_t>{5120U, 17408U},
                              std::pair<uint64_t, uint64_t>{17408U, 5120U}}) {
      for (const uint64_t m : {2U, 3U, 4U}) {
        ok = ok && check_default(m, shape.first, shape.second, target);
      }
    }
  }

  // The exact M and K/N boundaries stay out of ID94.
  for (const auto &shape : {std::pair<uint64_t, uint64_t>{5120U, 17408U},
                            std::pair<uint64_t, uint64_t>{17408U, 5120U}}) {
    for (const char *const target : {"gfx1030", "gfx1201"}) {
      for (const uint64_t m : {1U, 5U}) {
        ok = ok &&
             select_nvfp4_w4a4_variant(m, shape.first, shape.second, target) !=
                 KernelVariant::Nvfp4W4A4SmallMVgprReuse;
      }
    }
  }
  ok = ok && select_nvfp4_w4a4_variant(3U, 5120U, 5120U, "gfx1030") !=
                 KernelVariant::Nvfp4W4A4SmallMVgprReuse;
  ok = ok && select_nvfp4_w4a4_variant(3U, 5120U, 17408U, "gfx1201") ==
                 KernelVariant::Nvfp4W4A4SmallMVgprReuse;
  ok = ok && select_nvfp4_w4a4_variant(3U, 5120U, 17408U, "gfx942") !=
                 KernelVariant::Nvfp4W4A4SmallMVgprReuse;

  // ID90 remains the explicit gfx1201 row-grid control.  A present value of
  // "0" disables both the control and ID94 adoption for an easy rollback.
  setenv("SLLM_NVFP4_W4A4_SMALL_M_ROWGRID_GFX1201", "1", 1);
  const SelectorDecision gfx1201_control =
      select_nvfp4_w4a4_decision(3U, 5120U, 17408U, "gfx1201");
  ok =
      ok &&
      gfx1201_control.variant == KernelVariant::Nvfp4W4A4SmallMGfx1201RowGrid &&
      gfx1201_control.supported && gfx1201_control.enabled &&
      !gfx1201_control.adopted;
  setenv("SLLM_NVFP4_W4A4_SMALL_M_ROWGRID_GFX1201", "0", 1);
  ok = ok &&
       select_nvfp4_w4a4_variant(3U, 5120U, 17408U, "gfx1201") !=
           KernelVariant::Nvfp4W4A4SmallMVgprReuse &&
       select_nvfp4_w4a4_variant(3U, 5120U, 17408U, "gfx1201") !=
           KernelVariant::Nvfp4W4A4SmallMGfx1201RowGrid;
  unsetenv("SLLM_NVFP4_W4A4_SMALL_M_ROWGRID_GFX1201");

  // ID89 keeps the logical identity while changing only the measured
  // gfx1201 exact-tuple launch symbol at the StageK64 boundary.
  for (const auto &shape : {std::pair<uint64_t, uint64_t>{5120U, 17408U},
                            std::pair<uint64_t, uint64_t>{17408U, 5120U}}) {
    for (const uint64_t m : {255U, 256U, 257U}) {
      ok = ok && check_kahan_stage_boundary(m, shape.first, shape.second);
    }
  }

  // ID88 remains the explicit rollback/control path.
  setenv(kSmallMRowGrid, "1", 1);
  const SelectorDecision forced =
      select_nvfp4_w4a4_decision(3U, 5120U, 17408U, "gfx1030");
  ok = ok && forced.variant == KernelVariant::Nvfp4W4A4SmallMRowGrid &&
       forced.supported && forced.enabled && !forced.adopted;

  // Any presence of the control, including "0", suppresses ID94 adoption.
  setenv(kSmallMRowGrid, "0", 1);
  const KernelVariant suppressed =
      select_nvfp4_w4a4_variant(3U, 5120U, 17408U, "gfx1030");
  ok = ok && suppressed != KernelVariant::Nvfp4W4A4SmallMVgprReuse &&
       suppressed != KernelVariant::Nvfp4W4A4SmallMRowGrid;

  setenv(kBaseline, "1", 1);
  ok = ok && select_nvfp4_w4a4_variant(3U, 5120U, 17408U, "gfx1030") ==
                 KernelVariant::Nvfp4W4A4Packed;

  clear_controls();
  std::cout << (ok ? "PASS" : "FAIL") << '\n';
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
