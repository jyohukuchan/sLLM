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
  const auto default_m64 =
      select_nvfp4_w4a4_decision(64U, 5120U, 17408U, "gfx1201");
  const auto default_m63 =
      select_nvfp4_w4a4_decision(63U, 5120U, 17408U, "gfx1201");
  const auto default_shape_boundary =
      select_nvfp4_w4a4_decision(64U, 5120U, 5120U, "gfx1201");
  const auto stage32_symbol =
      device_symbol_for_target(KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan,
                               "gfx1201", 65U, 5120U, 17408U);
  const auto stage32_boundary_symbol =
      device_symbol_for_target(KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan,
                               "gfx1201", 255U, 5120U, 17408U);
  const auto stage32_reverse_boundary_symbol =
      device_symbol_for_target(KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan,
                               "gfx1201", 255U, 17408U, 5120U);
  const auto stage64_symbol =
      device_symbol_for_target(KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan,
                               "gfx1201", 256U, 5120U, 17408U);
  const auto stage64_reverse_symbol =
      device_symbol_for_target(KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan,
                               "gfx1201", 256U, 17408U, 5120U);
  const auto stage64_tail_symbol =
      device_symbol_for_target(KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan,
                               "gfx1201", 257U, 5120U, 17408U);
  const auto stage64_reverse_tail_symbol =
      device_symbol_for_target(KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan,
                               "gfx1201", 257U, 17408U, 5120U);
  ok = ok &&
       default_m64.variant == KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan &&
       default_m64.supported && default_m64.enabled && default_m64.adopted &&
       default_m63.variant != KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan &&
       default_shape_boundary.variant !=
           KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan &&
       std::string_view(stage32_symbol) ==
           kNvfp4W4A4PrefillGfx1201WmmaKahanDeviceSymbol &&
       std::string_view(stage32_boundary_symbol) ==
           kNvfp4W4A4PrefillGfx1201WmmaKahanDeviceSymbol &&
       std::string_view(stage32_reverse_boundary_symbol) ==
           kNvfp4W4A4PrefillGfx1201WmmaKahanDeviceSymbol &&
       std::string_view(stage64_symbol) ==
           kNvfp4W4A4PrefillGfx1201WmmaKahanAlignedK5120N17408DeviceSymbol &&
       std::string_view(stage64_reverse_symbol) ==
           kNvfp4W4A4PrefillGfx1201WmmaKahanAlignedK17408N5120DeviceSymbol &&
       std::string_view(stage64_tail_symbol) ==
           kNvfp4W4A4PrefillGfx1201WmmaKahanLookaheadDeviceSymbol &&
       std::string_view(stage64_reverse_tail_symbol) ==
           kNvfp4W4A4PrefillGfx1201WmmaKahanLookaheadDeviceSymbol;
  // Exact measured padding scope: adjacent and other aligned M retain their
  // previous symbols; both K/N orientations reach padding only at M2048.
  for (const auto &shape :
       {std::pair<uint64_t, uint64_t>{5120U, 17408U}, {17408U, 5120U}}) {
    for (const uint64_t rows : {1920U, 2047U, 2048U, 2049U, 2176U}) {
      const std::string_view symbol = device_symbol_for_target(
          KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan, "gfx1201", rows,
          shape.first, shape.second);
      const std::string_view expected =
          rows == 2048U
              ? (shape.first == 5120U
                     ? "sllm_nvfp4_gfx1201_wmma128x64_pad68_k5120n17408_v1"
                     : "sllm_nvfp4_gfx1201_wmma128x64_pad68_k17408n5120_v1")
              : (rows % 128U == 0U
                     ? (shape.first == 5120U
                            ? kNvfp4W4A4PrefillGfx1201WmmaKahanAlignedK5120N17408DeviceSymbol
                            : kNvfp4W4A4PrefillGfx1201WmmaKahanAlignedK17408N5120DeviceSymbol)
                     : kNvfp4W4A4PrefillGfx1201WmmaKahanLookaheadDeviceSymbol);
      ok = ok && symbol == expected;
    }
  }
  setenv(kahan, "1", 1);
  for (const auto target : {"gfx1201", "gfx1030", "gfx942", "unknown"}) {
    for (const auto &shape :
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
  for (const auto &shape : {std::pair<uint64_t, uint64_t>{2048U, 17408U},
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
  unsetenv(baseline);
  setenv(kahan, "0", 1);
  ok =
      ok && select_nvfp4_w4a4_decision(64U, 5120U, 17408U, "gfx1201").variant !=
                KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaKahan;

  std::cout << (ok ? "PASS" : "FAIL") << '\n';
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
