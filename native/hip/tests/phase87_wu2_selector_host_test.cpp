#include "matmul_kernel_internal.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using sllm_matmul_kernel::HostKernelVariant;
using sllm_matmul_kernel::KernelVariant;

struct Shape final {
  uint64_t k;
  uint64_t n;
};

constexpr std::array<Shape, 8> kDot4Shapes = {{
    {5120U, 1024U},
    {5120U, 6144U},
    {5120U, 10240U},
    {5120U, 12288U},
    {5120U, 17408U},
    {5120U, 248320U},
    {6144U, 5120U},
    {17408U, 5120U},
}};

bool check_dot4_selection(const Shape &shape) {
  using namespace sllm_matmul_kernel;
  const SelectorDecision decision =
      select_fp8_outer_decision(1U, shape.k, shape.n, "gfx1201", false);
  return decision.variant == HostKernelVariant::Fp8OuterGfx1201Dot4 &&
         decision.supported && decision.enabled && decision.adopted &&
         decision.runnable() &&
         std::string_view(decision.reason) == kSelectorReasonAdopted &&
         std::string_view(logical_kernel_id(decision.variant)) ==
             "matmul.fp8.outer.gfx1201.dot4.v1" &&
         std::string_view(device_symbol(decision.variant)) ==
             "sllm_matmul_fp8_outer_gfx1201_dot4_v1" &&
         grid_size_x(decision.variant, 1U, shape.n, shape.k) ==
             (shape.n + 7U) / 8U &&
         workgroup_size_x(decision.variant) == 256U;
}

bool check_fallback_selection(const Shape &shape) {
  using namespace sllm_matmul_kernel;
  const KernelVariant expected_old = HostKernelVariant::Fp8Native;
  bool valid = true;
  for (const uint64_t rows : {2U, 3U, 4U, 5U}) {
    valid = valid && select_fp8_outer_variant(rows, shape.k, shape.n, "gfx1201",
                                              false) == expected_old;
  }
  valid = valid && select_fp8_outer_variant(1U, shape.k, shape.n, "gfx1201",
                                            true) == expected_old;
  return valid;
}

bool check_nonaligned_fallback() {
  using namespace sllm_matmul_kernel;
  constexpr std::array<Shape, 5> kNonaligned = {{
      {5119U, 1024U},
      {5120U, 1023U},
      {5121U, 1024U},
      {6144U, 5121U},
      {17408U, 5119U},
  }};
  bool valid = true;
  for (const Shape &shape : kNonaligned) {
    valid = valid &&
            select_fp8_outer_variant(1U, shape.k, shape.n, "gfx1201", false) ==
                HostKernelVariant::Fp8Native;
  }
  // The new provider is exact gfx1201-only.  The existing gfx942 native
  // route and gfx1030 software route remain selected for the same tuples.
  for (const Shape &shape : kDot4Shapes) {
    valid = valid &&
            select_fp8_outer_variant(1U, shape.k, shape.n, "gfx942", false) ==
                HostKernelVariant::Fp8Native;
    valid = valid &&
            select_fp8_outer_variant(1U, shape.k, shape.n, "gfx1030", false) !=
                HostKernelVariant::Fp8OuterGfx1201Dot4;
  }
  return valid;
}

} // namespace

int main() {
  bool valid = true;
  for (const Shape &shape : kDot4Shapes) {
    valid = valid && check_dot4_selection(shape);
    valid = valid && check_fallback_selection(shape);
  }
  valid = valid && check_nonaligned_fallback();
  std::cout << "phase87_wu2_selector_host status=" << (valid ? "PASS" : "FAIL")
            << " dot4_shapes=" << kDot4Shapes.size() << '\n';
  return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
