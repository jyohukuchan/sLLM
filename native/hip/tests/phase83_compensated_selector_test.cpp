#include "matmul_kernel_internal.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>

int main() {
  using namespace sllm_matmul_kernel;
  constexpr const char *flag = "SLLM_NVFP4_W4A4_PREFILL_FORCE_COMPENSATED";
  unsetenv(flag);
  unsetenv("SLLM_NVFP4_W4A4_FORCE_BASELINE");
  const auto old = select_nvfp4_w4a4_decision(65, 5120, 17408, "gfx1030");
  bool ok = old.variant != KernelVariant::Nvfp4W4A4PrefillCompensated64x64;
  setenv(flag, "1", 1);
  for (const auto target : {"gfx1030", "gfx1201"}) {
    const auto selected = select_nvfp4_w4a4_decision(65, 5120, 17408, target);
    ok = ok &&
         selected.variant == KernelVariant::Nvfp4W4A4PrefillCompensated64x64 &&
         selected.supported && selected.enabled && !selected.adopted;
    for (const auto k : {15U, 17U, 5119U, 5121U}) {
      const auto tail = select_nvfp4_w4a4_decision(65, k, 17408, target);
      ok =
          ok && tail.variant != KernelVariant::Nvfp4W4A4PrefillCompensated64x64;
    }
    const auto decode = select_nvfp4_w4a4_decision(1, 5120, 17408, target);
    ok =
        ok && decode.variant != KernelVariant::Nvfp4W4A4PrefillCompensated64x64;
  }
  for (const auto target : {"gfx942", "gfx1100", "unknown"})
    ok = ok && select_nvfp4_w4a4_decision(65, 5120, 17408, target).variant !=
                   KernelVariant::Nvfp4W4A4PrefillCompensated64x64;
  setenv("SLLM_NVFP4_W4A4_FORCE_BASELINE", "1", 1);
  ok = ok && select_nvfp4_w4a4_decision(65, 5120, 17408, "gfx1030").variant !=
                 KernelVariant::Nvfp4W4A4PrefillCompensated64x64;
  unsetenv("SLLM_NVFP4_W4A4_FORCE_BASELINE");
  unsetenv(flag);
  setenv("SLLM_NVFP4_W4A4_SMALL_M_ROWGRID", "1", 1);
  for (const auto m : {0U, 1U, 2U, 3U, 4U, 5U}) {
    const auto d = select_nvfp4_w4a4_decision(m, 5120, 17408, "gfx1030");
    const bool eligible = m >= 2U && m <= 4U;
    ok = ok &&
         ((d.variant == KernelVariant::Nvfp4W4A4SmallMRowGrid) == eligible);
    if (eligible)
      ok = ok && d.supported && d.enabled && !d.adopted;
  }
  for (const auto target : {"gfx1201", "gfx942", "unknown"})
    ok = ok && select_nvfp4_w4a4_decision(3, 5120, 17408, target).variant !=
                   KernelVariant::Nvfp4W4A4SmallMRowGrid;
  for (const auto k : {5119U, 5121U})
    ok = ok && select_nvfp4_w4a4_decision(3, k, 17408, "gfx1030").variant !=
                   KernelVariant::Nvfp4W4A4SmallMRowGrid;
  setenv("SLLM_NVFP4_W4A4_FORCE_BASELINE", "1", 1);
  ok = ok && select_nvfp4_w4a4_decision(3, 5120, 17408, "gfx1030").variant !=
                 KernelVariant::Nvfp4W4A4SmallMRowGrid;
  std::cout << (ok ? "PASS" : "FAIL") << '\n';
  return ok ? 0 : 1;
}
