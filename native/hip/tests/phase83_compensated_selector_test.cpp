#include "matmul_kernel_internal.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>

int main() {
  using namespace sllm_matmul_kernel;
  constexpr const char *flag = "SLLM_NVFP4_W4A4_PREFILL_FORCE_COMPENSATED";
  constexpr const char *small_m = "SLLM_NVFP4_W4A4_SMALL_M_ROWGRID";
  constexpr const char *small_m_gfx1201 =
      "SLLM_NVFP4_W4A4_SMALL_M_ROWGRID_GFX1201";
  unsetenv(flag);
  unsetenv(small_m);
  unsetenv(small_m_gfx1201);
  unsetenv("SLLM_NVFP4_W4A4_FORCE_BASELINE");
  const auto default_compensated =
      select_nvfp4_w4a4_decision(65, 5120, 17408, "gfx1030");
  const auto default_m_boundary =
      select_nvfp4_w4a4_decision(63, 5120, 17408, "gfx1030");
  const auto default_shape_boundary =
      select_nvfp4_w4a4_decision(65, 5120, 5120, "gfx1030");
  const auto tile_m128_511 =
      select_nvfp4_w4a4_decision(511, 5120, 17408, "gfx1030");
  const auto tile_m128_512 =
      select_nvfp4_w4a4_decision(512, 5120, 17408, "gfx1030");
  const auto tile_m128_513 =
      select_nvfp4_w4a4_decision(513, 5120, 17408, "gfx1030");
  const auto tile_m128_1023 =
      select_nvfp4_w4a4_decision(1023, 17408, 5120, "gfx1030");
  const auto tile_m128_1024 =
      select_nvfp4_w4a4_decision(1024, 17408, 5120, "gfx1030");
  const auto tile_m128_1025 =
      select_nvfp4_w4a4_decision(1025, 17408, 5120, "gfx1030");
  bool ok =
      default_compensated.variant ==
          KernelVariant::Nvfp4W4A4PrefillCompensated64x64 &&
      default_compensated.supported && default_compensated.enabled &&
      default_compensated.adopted &&
      default_m_boundary.variant !=
          KernelVariant::Nvfp4W4A4PrefillCompensated64x64 &&
      default_shape_boundary.variant !=
          KernelVariant::Nvfp4W4A4PrefillCompensated64x64 &&
      tile_m128_511.variant ==
          KernelVariant::Nvfp4W4A4PrefillCompensated64x64 &&
      tile_m128_512.variant ==
          KernelVariant::Nvfp4W4A4PrefillCompensated64x64 &&
      tile_m128_513.variant ==
          KernelVariant::Nvfp4W4A4PrefillCompensated64x64 &&
      tile_m128_1023.variant ==
          KernelVariant::Nvfp4W4A4PrefillCompensated64x64 &&
      tile_m128_1024.variant ==
          KernelVariant::Nvfp4W4A4PrefillCompensated64x64 &&
      tile_m128_1025.variant ==
          KernelVariant::Nvfp4W4A4PrefillCompensated64x64 &&
      phase83_gfx1030_nvfp4_w4a4_compensated128x64_shape(512, 5120, 17408) &&
      !phase83_gfx1030_nvfp4_w4a4_compensated128x64_shape(511, 5120, 17408) &&
      !phase83_gfx1030_nvfp4_w4a4_compensated128x64_shape(513, 5120, 17408) &&
      grid_size_x(KernelVariant::Nvfp4W4A4PrefillCompensated64x64, 512, 17408,
                  5120) == ((512U + 127U) / 128U) * ((17408U + 63U) / 64U) &&
      grid_size_x(KernelVariant::Nvfp4W4A4PrefillCompensated64x64, 513, 17408,
                  5120) == ((513U + 63U) / 64U) * ((17408U + 63U) / 64U) &&
      std::strcmp(device_symbol_for_target(
                      KernelVariant::Nvfp4W4A4PrefillCompensated64x64,
                      "gfx1030", 512, 5120, 17408),
                  "sllm_nvfp4_w4a4_prefill_compensated128x64_v1") == 0 &&
      std::strcmp(device_symbol_for_target(
                      KernelVariant::Nvfp4W4A4PrefillCompensated64x64,
                      "gfx1030", 513, 5120, 17408),
                  "sllm_nvfp4_w4a4_prefill_compensated64x64_v1") == 0 &&
      std::strcmp(device_symbol_for_target(
                      KernelVariant::Nvfp4W4A4PrefillCompensated64x64,
                      "gfx1201", 512, 5120, 17408),
                  "sllm_nvfp4_w4a4_prefill_compensated64x64_v1") == 0;
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
  unsetenv("SLLM_NVFP4_W4A4_FORCE_BASELINE");
  unsetenv(small_m);
  unsetenv(small_m_gfx1201);
  for (const auto target : {"gfx1030", "gfx1201"}) {
    for (const auto &shape :
         {std::pair<uint64_t, uint64_t>{5120U, 17408U}, {17408U, 5120U}}) {
      for (const auto m : {2U, 3U, 4U}) {
        const auto d =
            select_nvfp4_w4a4_decision(m, shape.first, shape.second, target);
        const auto expected = KernelVariant::Nvfp4W4A4SmallMVgprReuse;
        ok = ok && d.variant == expected && d.supported && d.enabled &&
             d.adopted;
      }
    }
  }
  setenv(small_m, "0", 1);
  setenv(small_m_gfx1201, "0", 1);
  ok = ok &&
       select_nvfp4_w4a4_decision(3, 5120, 17408, "gfx1030").variant !=
           KernelVariant::Nvfp4W4A4SmallMRowGrid &&
       select_nvfp4_w4a4_decision(3, 5120, 17408, "gfx1201").variant !=
           KernelVariant::Nvfp4W4A4SmallMGfx1201RowGrid;
  std::cout << (ok ? "PASS" : "FAIL") << '\n';
  return ok ? 0 : 1;
}
