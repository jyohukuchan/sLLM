// Host-only before/after contract test for the lowp selection boundary.
// The CSV is generated from the isolated pre-move baseline tree. This test
// calls only the moved lowp provider/kernel headers and the host planning API.

#include <lowp/detail/low_precision_matmul_provider.hpp>
#include <lowp/detail/lowp_api_internal.hpp>
#include <lowp/detail/lowp_kernel_internal.hpp>
#include <lowp/detail/lowp_provider_plan.hpp>
#include <lowp/lowp.h>

#include <array>
#include <cassert>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifndef LOWP_SELECTION_FIXTURE_PATH
#define LOWP_SELECTION_FIXTURE_PATH "native/lowp/tests/selection_fixture.csv"
#endif

namespace {
using sllm_lowp::ExactTarget;
using sllm_lowp::MatmulFormat;
using sllm_matmul_kernel::KernelVariant;

struct Expected {
  std::string format, target, decision_reason;
  uint64_t m = 0, k = 0, n = 0;
  int native_variant = -1, decision_variant = -1, provider = -1;
  int rejection = -1, architecture = -1, tile = -1, activation_pack = -1;
  int inner_product = -1, concrete_provider = -1, concrete_tile = -1;
  int concrete_inner_product = -1;
  bool decision_supported = false, decision_enabled = false;
  bool decision_adopted = false, provider_supported = false;
  uint64_t activation_value_bytes = 0, activation_scale_offset = 0;
  uint64_t activation_scale_bytes = 0, activation_workspace_bytes = 0;
  uint64_t total_workspace_bytes = 0, split4_workspace_bytes = 0;
  uint64_t staging_workspace_bytes = 0;
};

std::vector<std::string> split(const std::string &line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ','))
    fields.push_back(field);
  return fields;
}

template <typename T> T number(const std::string &text) {
  T result{};
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), result);
  assert(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size());
  return result;
}

bool boolean(const std::string &text) { return text == "1"; }

std::vector<Expected> load_fixture() {
  std::ifstream input(LOWP_SELECTION_FIXTURE_PATH);
  assert(input.good());
  std::string line;
  assert(std::getline(input, line));
  std::vector<Expected> rows;
  while (std::getline(input, line)) {
    if (line.empty())
      continue;
    const auto fields = split(line);
    assert(fields.size() == 28U);
    Expected row{};
    row.format = fields[0];
    row.target = fields[1];
    row.m = number<uint64_t>(fields[2]);
    row.k = number<uint64_t>(fields[3]);
    row.n = number<uint64_t>(fields[4]);
    row.native_variant = number<int>(fields[5]);
    row.decision_variant = number<int>(fields[6]);
    row.decision_supported = boolean(fields[7]);
    row.decision_enabled = boolean(fields[8]);
    row.decision_adopted = boolean(fields[9]);
    row.decision_reason = fields[10];
    row.provider = number<int>(fields[11]);
    row.rejection = number<int>(fields[12]);
    row.architecture = number<int>(fields[13]);
    row.tile = number<int>(fields[14]);
    row.activation_pack = number<int>(fields[15]);
    row.inner_product = number<int>(fields[16]);
    row.provider_supported = boolean(fields[17]);
    row.concrete_provider = number<int>(fields[18]);
    row.concrete_tile = number<int>(fields[19]);
    row.concrete_inner_product = number<int>(fields[20]);
    row.activation_value_bytes = number<uint64_t>(fields[21]);
    row.activation_scale_offset = number<uint64_t>(fields[22]);
    row.activation_scale_bytes = number<uint64_t>(fields[23]);
    row.activation_workspace_bytes = number<uint64_t>(fields[24]);
    row.total_workspace_bytes = number<uint64_t>(fields[25]);
    row.split4_workspace_bytes = number<uint64_t>(fields[26]);
    row.staging_workspace_bytes = number<uint64_t>(fields[27]);
    rows.push_back(std::move(row));
  }
  return rows;
}

MatmulFormat format(const std::string &name) {
  if (name == "mxfp8_w8a8")
    return MatmulFormat::Mxfp8E4M3W8A8;
  if (name == "mxfp6_w6a6")
    return MatmulFormat::Mxfp6E3M2W6A6;
  if (name == "nvfp4_w4a4")
    return MatmulFormat::Nvfp4W4A4;
  if (name == "mxfp4_w4a4_internal")
    return MatmulFormat::Mxfp4W4A4;
  assert(name == "fp8_outer_w8a8");
  return MatmulFormat::Fp8OuterE4M3W8A8;
}

ExactTarget target(const std::string &name) {
  if (name == "gfx1030")
    return ExactTarget::Gfx1030;
  assert(name == "gfx1201");
  return ExactTarget::Gfx1201;
}

KernelVariant native_variant(const Expected &row) {
  const auto f = format(row.format);
  if (f == MatmulFormat::Mxfp8E4M3W8A8)
    return sllm_matmul_kernel::select_mxfp8_variant(row.m, row.k, row.n,
                                                    row.target.c_str());
  if (f == MatmulFormat::Mxfp6E3M2W6A6)
    return sllm_matmul_kernel::select_mxfp6_variant(row.m, row.k, row.n,
                                                    row.target.c_str());
  if (f == MatmulFormat::Nvfp4W4A4)
    return sllm_matmul_kernel::select_nvfp4_w4a4_variant(row.m, row.k, row.n,
                                                         row.target.c_str());
  if (f == MatmulFormat::Mxfp4W4A4)
    return sllm_matmul_kernel::select_mxfp4_variant(row.m);
  return sllm_matmul_kernel::select_fp8_software_variant(row.m, row.k, row.n,
                                                         row.target.c_str());
}

lowp_matmul_request_t api_request(const Expected &row) {
  const auto request = sllm_lowp::make_provider_request(
      format(row.format), target(row.target), row.m, row.n, row.k);
  return {sizeof(lowp_matmul_request_t),
          LOWP_ABI_VERSION,
          static_cast<lowp_format_t>(request.format),
          static_cast<lowp_target_t>(request.target),
          static_cast<lowp_layout_t>(request.weight_layout),
          static_cast<lowp_layout_t>(request.activation_layout),
          row.m,
          row.n,
          row.k};
}

void compare_row(const Expected &row) {
  const auto f = format(row.format);
  const auto t = target(row.target);
  // gfx1201 FP8 is deliberately outside this library. Its unchanged native
  // selector is exercised by native/hip's host contracts; here preserve the
  // historical ID and assert the public library rejects the format/target.
  if (f == MatmulFormat::Fp8OuterE4M3W8A8 && t == ExactTarget::Gfx1201) {
    // The historical native FP8 provider ID is owned by the runtime.
    constexpr int kRuntimeNativeFp8VariantId = 5;
    assert(kRuntimeNativeFp8VariantId == row.native_variant);
    const auto outside = sllm_lowp::prepare_provider_plan(
        sllm_lowp::make_provider_request(f, t, row.m, row.n, row.k));
    assert(static_cast<int>(outside.provider) == row.provider);
    assert(static_cast<int>(outside.rejection) == row.rejection);
    lowp_matmul_plan_t plan{};
    const auto request = api_request(row);
    assert(lowp_matmul_plan(&request, &plan) != LOWP_SUCCESS);
    return;
  }
  // The frozen fixture predates the WU-3P exact MTP prefix adoption. Keep its
  // source rows intact, and check the four affected rows against the current
  // selector and public plan contract here.
  if (f == MatmulFormat::Nvfp4W4A4 &&
      (t == ExactTarget::Gfx1030 || t == ExactTarget::Gfx1201) &&
      sllm_matmul_kernel::phase87_wu3p_mtp_nvfp4_prefix_shape(row.m, row.k,
                                                              row.n)) {
    const auto expected =
        t == ExactTarget::Gfx1030
            ? KernelVariant::Nvfp4W4A4PrefillDp4a64x64
            : KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x64;
    const auto decision = sllm_matmul_kernel::select_nvfp4_w4a4_decision(
        row.m, row.k, row.n, row.target.c_str());
    assert(decision.variant == expected && decision.supported &&
           decision.enabled && decision.adopted);
    const auto prepared = sllm_lowp::prepare_provider_plan(
        sllm_lowp::make_provider_request(f, t, row.m, row.n, row.k));
    assert(prepared.supported() &&
           static_cast<int>(prepared.provider) == row.provider);
    const auto concrete = sllm_lowp::concrete_provider_plan(prepared, expected);
    assert(concrete.has_value());
    lowp_matmul_plan_t plan{};
    const auto request = api_request(row);
    assert(lowp_matmul_plan(&request, &plan) == LOWP_SUCCESS);
    assert(plan.variant == static_cast<uint32_t>(expected));
    assert(plan.provider == static_cast<uint32_t>(concrete->provider));
    assert(plan.tile == static_cast<uint32_t>(concrete->tile));
    assert(plan.inner_product ==
           static_cast<uint32_t>(concrete->inner_product));
    // ID62 keeps its logical 64x64 plan identity. The production launcher
    // uses the 32x64 device body for M<=32; the WU-3P GPU probe covers it.
    if (t == ExactTarget::Gfx1030 && row.m <= 32U)
      assert(concrete->tile == sllm_lowp::TilePolicy::BlockRow64Column64);
    assert(plan.selector_supported == 1U && plan.selector_enabled == 1U &&
           plan.adopted == 1U);
    assert(plan.activation_value_bytes == row.activation_value_bytes);
    assert(plan.activation_scale_bytes == row.activation_scale_bytes);
    assert(plan.workspace_bytes == row.total_workspace_bytes);
    return;
  }
  const auto native = native_variant(row);
  assert(static_cast<int>(native) == row.native_variant);
  const auto prepared = sllm_lowp::prepare_provider_plan(
      sllm_lowp::make_provider_request(f, t, row.m, row.n, row.k));
  assert(static_cast<int>(prepared.provider) == row.provider);
  assert(static_cast<int>(prepared.rejection) == row.rejection);
  assert(prepared.supported() == row.provider_supported);

  if (f == MatmulFormat::Nvfp4W4A4) {
    const auto decision = sllm_matmul_kernel::select_nvfp4_w4a4_decision(
        row.m, row.k, row.n, row.target.c_str());
    assert(static_cast<int>(decision.variant) == row.decision_variant);
    assert(decision.supported == row.decision_supported);
    assert(decision.enabled == row.decision_enabled);
    assert(decision.adopted == row.decision_adopted);
    assert(row.decision_reason == decision.reason);
  } else if (f == MatmulFormat::Fp8OuterE4M3W8A8) {
    const auto decision = sllm_matmul_kernel::select_fp8_software_decision(
        row.m, row.k, row.n, row.target.c_str(), false);
    assert(static_cast<int>(decision.variant) == row.decision_variant);
    assert(decision.supported == row.decision_supported);
    assert(decision.enabled == row.decision_enabled);
    assert(decision.adopted == row.decision_adopted);
    assert(row.decision_reason == decision.reason);
  }

  lowp_matmul_plan_t plan{};
  const auto request = api_request(row);
  const bool legacy_mxfp4 = f == MatmulFormat::Mxfp4W4A4;
  const lowp_status_t status =
      legacy_mxfp4 ? sllm_lowp::plan_internal(&request, &plan, true)
                   : lowp_matmul_plan(&request, &plan);
  if (!row.provider_supported || row.concrete_provider < 0) {
    assert(status != LOWP_SUCCESS);
    return;
  }
  assert(status == LOWP_SUCCESS);
  if (row.decision_variant >= 0) {
    assert(plan.selector_supported ==
           static_cast<uint32_t>(row.decision_supported));
    assert(plan.selector_enabled ==
           static_cast<uint32_t>(row.decision_enabled));
    assert(plan.adopted == static_cast<uint32_t>(row.decision_adopted));
  }
  assert(plan.variant == static_cast<uint32_t>(native));
  assert(plan.provider == static_cast<uint32_t>(row.concrete_provider));
  assert(plan.tile == static_cast<uint32_t>(row.concrete_tile));
  assert(plan.inner_product ==
         static_cast<uint32_t>(row.concrete_inner_product));
  if (plan.activation_scale_bytes != row.activation_scale_bytes) {
    std::cerr << "footprint mismatch " << row.format << '/' << row.target << ' '
              << row.m << 'x' << row.k << 'x' << row.n << " expected "
              << row.activation_scale_bytes << " got "
              << plan.activation_scale_bytes << '\n';
    assert(false);
  }
  assert(plan.activation_value_bytes == row.activation_value_bytes);
  assert(plan.activation_scale_offset == row.activation_scale_offset);
  assert(plan.activation_scale_bytes == row.activation_scale_bytes);
  assert(plan.workspace_bytes == row.total_workspace_bytes);
  assert(plan.staging_workspace_bytes == row.staging_workspace_bytes);
  if (row.split4_workspace_bytes != 0U) {
    assert(plan.scratch_bytes != 0U);
  }
}
} // namespace

int main() {
  for (const uint32_t id : {8U, 9U, 10U}) {
    const auto retired = static_cast<KernelVariant>(id);
    assert(sllm_matmul_kernel::lowp_logical_kernel_id(retired) == nullptr);
    assert(sllm_matmul_kernel::lowp_device_symbol(retired) == nullptr);
  }
  const auto rows = load_fixture();
  assert(rows.size() == 3002U);
  for (const auto &row : rows)
    compare_row(row);

  // WU-3P selects only the measured Qwen3.8 MTP prefix projection tuples.
  // The non-aligned final chunk and both sides of the M gate stay explicit.
  for (const auto target : {"gfx1030", "gfx1201"}) {
    const auto adopted = std::string_view(target) == "gfx1030"
                             ? KernelVariant::Nvfp4W4A4PrefillDp4a64x64
                             : KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x64;
    for (const auto m : {32U, 33U, 1023U, 1024U}) {
      for (const auto shape : {std::array<uint64_t, 2>{10240U, 5120U},
                               {5120U, 12288U},
                               {5120U, 1024U}}) {
        const auto decision = sllm_matmul_kernel::select_nvfp4_w4a4_decision(
            m, shape[0], shape[1], target);
        assert(decision.variant == adopted && decision.supported &&
               decision.enabled && decision.adopted);
      }
    }
    for (const auto m : {31U, 1025U}) {
      const auto decision = sllm_matmul_kernel::select_nvfp4_w4a4_decision(
          m, 5120U, 12288U, target);
      assert(decision.variant == KernelVariant::Nvfp4W4A4PrefillRow8Tiled256);
    }
    const auto body = sllm_matmul_kernel::select_nvfp4_w4a4_decision(
        1024U, 5120U, 17408U, target);
    assert(body.variant != adopted);
  }

  // WU-12C extends the direct ID94 contract through M=5 for the two exact
  // Qwen3.8 projection tuples. The projection-pack lane keeps its separate
  // M<=4 admission rule; this check covers only the decomposed lowp API.
  for (const auto target_name : {"gfx1030", "gfx1201"}) {
    const auto target_value = target(target_name);
    for (const auto shape :
         {std::array<uint64_t, 2>{5120U, 17408U}, {17408U, 5120U}}) {
      Expected m5{};
      m5.format = "nvfp4_w4a4";
      m5.target = target_name;
      m5.m = 5U;
      m5.k = shape[0];
      m5.n = shape[1];
      const auto m5_decision = sllm_matmul_kernel::select_nvfp4_w4a4_decision(
          m5.m, m5.k, m5.n, target_name);
      assert(m5_decision.variant == KernelVariant::Nvfp4W4A4SmallMVgprReuse &&
             m5_decision.supported && m5_decision.enabled &&
             m5_decision.adopted);
      const auto m5_prepared =
          sllm_lowp::prepare_provider_plan(sllm_lowp::make_provider_request(
              MatmulFormat::Nvfp4W4A4, target_value, m5.m, m5.n, m5.k));
      assert(m5_prepared.supported());
      const auto m5_concrete =
          sllm_lowp::concrete_provider_plan(m5_prepared, m5_decision.variant);
      assert(m5_concrete.has_value());
      lowp_matmul_plan_t m5_plan{};
      const auto m5_request = api_request(m5);
      assert(lowp_matmul_plan(&m5_request, &m5_plan) == LOWP_SUCCESS);
      assert(m5_plan.variant ==
             static_cast<uint32_t>(KernelVariant::Nvfp4W4A4SmallMVgprReuse));
      assert(m5_plan.provider == static_cast<uint32_t>(m5_concrete->provider));
      assert(m5_plan.tile == static_cast<uint32_t>(m5_concrete->tile));
      assert(m5_plan.inner_product ==
             static_cast<uint32_t>(m5_concrete->inner_product));
      assert(m5_plan.selector_supported == 1U &&
             m5_plan.selector_enabled == 1U && m5_plan.adopted == 1U);

      const auto m6_decision = sllm_matmul_kernel::select_nvfp4_w4a4_decision(
          6U, m5.k, m5.n, target_name);
      assert(m6_decision.variant != KernelVariant::Nvfp4W4A4SmallMVgprReuse);
    }
  }
  return 0;
}
