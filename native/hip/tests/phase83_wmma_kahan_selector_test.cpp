#include "matmul_kernel_internal.hpp"

#include <cstdlib>
#include <iostream>
#include <string_view>
#include <utility>

int main() {
  using namespace sllm_matmul_kernel;
  constexpr const char *const kahan =
      "SLLM_NVFP4_W4A4_PREFILL_FORCE_WMMA_COMPENSATED";
  constexpr const char *const old_compensated =
      "SLLM_NVFP4_W4A4_PREFILL_FORCE_COMPENSATED";
  constexpr const char *const baseline = "SLLM_NVFP4_W4A4_FORCE_BASELINE";
  unsetenv(kahan);
  unsetenv(old_compensated);
  unsetenv(baseline);

  bool ok = true;
  ok =
      ok && select_nvfp4_w4a4_decision(65U, 5120U, 17408U, "gfx1201").variant !=
                KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan;
  setenv(kahan, "1", 1);
  for (const auto target : {"gfx1201", "gfx1030", "gfx942", "unknown"}) {
    for (const auto shape :
         {std::pair<uint64_t, uint64_t>{5120U, 17408U}, {17408U, 5120U}}) {
      const auto selected =
          select_nvfp4_w4a4_decision(65U, shape.first, shape.second, target);
      const bool eligible = target == std::string_view("gfx1201");
      ok =
          ok && ((selected.variant ==
                  KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan) == eligible);
      if (eligible) {
        ok = ok && selected.supported && selected.enabled && !selected.adopted;
      }
    }
  }
  for (const auto shape : {std::pair<uint64_t, uint64_t>{2048U, 17408U},
                           {5119U, 17408U},
                           {5121U, 17408U},
                           {5120U, 5120U}}) {
    const auto rejected =
        select_nvfp4_w4a4_decision(65U, shape.first, shape.second, "gfx1201");
    ok = ok &&
         rejected.variant != KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan;
  }
  ok = ok && select_nvfp4_w4a4_decision(1U, 5120U, 17408U, "gfx1201").variant !=
                 KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan;

  // The previous ID87 flag retains precedence if both explicit opt-ins are set.
  setenv(old_compensated, "1", 1);
  ok =
      ok && select_nvfp4_w4a4_decision(65U, 5120U, 17408U, "gfx1201").variant ==
                KernelVariant::Nvfp4W4A4PrefillCompensated64x64;
  unsetenv(old_compensated);
  setenv(baseline, "1", 1);
  ok =
      ok && select_nvfp4_w4a4_decision(65U, 5120U, 17408U, "gfx1201").variant !=
                KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan;

  std::cout << (ok ? "PASS" : "FAIL") << '\n';
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
