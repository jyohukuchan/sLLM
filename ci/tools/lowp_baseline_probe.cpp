// Host-only selection fixture generator.
//
// This intentionally includes the current production headers directly.  The
// boundary move must keep the emitted rows byte-for-byte stable; after the
// move this include pair is the only path that needs to be retargeted to the
// extracted lowp library headers.

#include "low_precision_matmul_provider.hpp"
#include "matmul_kernel_internal.hpp"
#include "lowp_baseline_concrete.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace {

using sllm_lowp::ExactTarget;
using sllm_lowp::MatmulFormat;
using sllm_matmul_kernel::KernelVariant;

struct FixtureCase {
  MatmulFormat format;
  ExactTarget target;
  const char *target_name;
  uint64_t m;
  uint64_t k;
  uint64_t n;
};

struct FormatSpec {
  MatmulFormat format;
  const char *name;
  uint64_t base_k;
  uint64_t base_n;
};

constexpr std::array<FormatSpec, 8> kFormats = {{
    {MatmulFormat::Mxfp8E4M3W8A8, "mxfp8_w8a8", 2048U, 9216U},
    {MatmulFormat::Mxfp8E4M3W8A16, "mxfp8_w8a16", 2048U, 1024U},
    {MatmulFormat::Mxfp6E3M2W6A6, "mxfp6_w6a6", 2048U, 9216U},
    {MatmulFormat::Mxfp6E3M2W6A16, "mxfp6_w6a16", 2048U, 1024U},
    {MatmulFormat::Nvfp4W4A16, "nvfp4_w4a16", 5120U, 17408U},
    {MatmulFormat::Nvfp4W4A4, "nvfp4_w4a4", 5120U, 17408U},
    {MatmulFormat::Mxfp4W4A4, "mxfp4_w4a4_internal", 5120U, 17408U},
    {MatmulFormat::Fp8OuterE4M3W8A8, "fp8_outer_w8a8", 5120U, 17408U},
}};

constexpr std::array<std::pair<ExactTarget, const char *>, 2> kTargets = {{
    {ExactTarget::Gfx1030, "gfx1030"},
    {ExactTarget::Gfx1201, "gfx1201"},
}};

void add_case(std::vector<FixtureCase> &cases, const FormatSpec &format,
              const ExactTarget target, const char *const target_name,
              const uint64_t m, const uint64_t k, const uint64_t n) {
  cases.push_back({format.format, target, target_name, m, k, n});
}

const char *format_name(const MatmulFormat format) {
  for (const auto &item : kFormats) {
    if (item.format == format) {
      return item.name;
    }
  }
  return "unknown";
}

std::vector<FixtureCase> make_cases() {
  std::vector<FixtureCase> cases;
  // Every list contains both sides of the important provider/variant gates;
  // values such as 33, 65 and 1025 also exercise non-aligned tails.
  constexpr std::array<uint64_t, 30> m_values = {
      0U, 1U, 2U, 3U, 4U, 7U, 8U, 9U, 16U, 17U, 18U, 31U, 32U,
      33U, 63U, 64U, 65U, 127U, 128U, 129U, 511U, 512U, 513U,
      1023U, 1024U, 1025U, 4095U, 4096U, 4097U, 4098U};
  constexpr std::array<uint64_t, 38> k_values = {
      0U, 1U, 15U, 16U, 17U, 31U, 32U, 33U, 63U, 64U, 65U, 127U,
      128U, 129U, 2015U, 2016U, 2017U, 2047U, 2048U, 2049U, 5119U,
      5120U, 5121U, 6143U, 6144U, 6145U, 8191U, 8192U, 8193U,
      12287U, 12288U, 12289U, 16383U, 16384U, 16385U, 17407U,
      17408U, 17409U};
  constexpr std::array<uint64_t, 44> n_values = {
      0U, 1U, 15U, 16U, 17U, 31U, 32U, 33U, 63U, 64U, 65U, 127U,
      128U, 129U, 255U, 256U, 257U, 511U, 512U, 513U, 1023U, 1024U,
      1025U, 2047U, 2048U, 2049U, 6143U, 6144U, 6145U, 8191U, 8192U,
      8193U, 12287U, 12288U, 12289U, 16383U, 16384U, 16385U, 32767U,
      32768U, 32769U, 65535U, 65536U, 65537U};

  for (const auto &format : kFormats) {
    for (const auto &[target, target_name] : kTargets) {
      for (const uint64_t m : m_values) {
        add_case(cases, format, target, target_name, m, format.base_k,
                 format.base_n);
      }
      for (const uint64_t k : k_values) {
        add_case(cases, format, target, target_name, 1U, k, format.base_n);
        add_case(cases, format, target, target_name, 17U, k, format.base_n);
        add_case(cases, format, target, target_name, 128U, k, format.base_n);
      }
      for (const uint64_t n : n_values) {
        add_case(cases, format, target, target_name, 1U, format.base_k, n);
        add_case(cases, format, target, target_name, 17U, format.base_k, n);
        add_case(cases, format, target, target_name, 128U, format.base_k, n);
      }
      // Provider WMMA gates and the adopted M=1 families have independent
      // dimensions.  Keep these explicit even when a generic sweep happens
      // to contain the same coordinate.
      constexpr std::array<std::tuple<uint64_t, uint64_t, uint64_t>, 22>
          special_cases = {{{127U, 2048U, 1024U},
                            {128U, 2048U, 1024U},
                            {129U, 2048U, 1024U},
                            {128U, 2047U, 1024U},
                            {128U, 2049U, 1024U},
                            {128U, 2048U, 1023U},
                            {128U, 2048U, 1025U},
                            {128U, 2048U, 32768U},
                            {128U, 2048U, 32769U},
                            {17U, 2048U, 1024U},
                            {16U, 2048U, 1024U},
                            {17U, 2047U, 1024U},
                            {17U, 2049U, 1024U},
                            {17U, 2048U, 1023U},
                            {17U, 2048U, 1025U},
                            {1U, 64U, 33U},
                            {1U, 63U, 33U},
                            {1U, 65U, 33U},
                            {3U, 2017U, 1025U},
                            {17U, 2049U, 1023U},
                            {129U, 2081U, 1025U},
                            {513U, 5121U, 32769U}}};
      for (const auto &[m, k, n] : special_cases) {
        add_case(cases, format, target, target_name, m, k, n);
      }
      if (format.format == MatmulFormat::Fp8OuterE4M3W8A8) {
        constexpr std::array<std::tuple<uint64_t, uint64_t, uint64_t>, 11>
            fp8_tuples = {{{1U, 64U, 33U},
                           {1U, 128U, 64U},
                           {1U, 5120U, 1024U},
                           {1U, 5120U, 12288U},
                           {1U, 5120U, 17408U},
                           {1U, 6144U, 5120U},
                           {1U, 17408U, 5120U},
                           {2U, 5120U, 10240U},
                           {3U, 5120U, 12288U},
                           {4U, 5120U, 17408U},
                           {4U, 6144U, 5120U}}};
        for (const auto &[m, k, n] : fp8_tuples) {
          add_case(cases, format, target, target_name, m, k, n);
          if (k > 0U) {
            add_case(cases, format, target, target_name, m, k - 1U, n);
          }
          add_case(cases, format, target, target_name, m, k + 1U, n);
          if (n > 0U) {
            add_case(cases, format, target, target_name, m, k, n - 1U);
          }
          add_case(cases, format, target, target_name, m, k, n + 1U);
        }
      }
    }
  }

  // Deduplicate the Cartesian boundary sweeps while preserving deterministic
  // lexical order for a stable digest.
  std::set<std::tuple<uint8_t, uint8_t, uint64_t, uint64_t, uint64_t>> seen;
  std::vector<FixtureCase> unique;
  unique.reserve(cases.size());
  for (const auto &item : cases) {
    const auto key = std::make_tuple(static_cast<uint8_t>(item.format),
                                     static_cast<uint8_t>(item.target), item.m,
                                     item.k, item.n);
    if (seen.insert(key).second) {
      unique.push_back(item);
    }
  }
  std::sort(unique.begin(), unique.end(), [](const FixtureCase &left,
                                             const FixtureCase &right) {
    return std::tie(left.format, left.target, left.m, left.k, left.n) <
           std::tie(right.format, right.target, right.m, right.k, right.n);
  });
  return unique;
}

struct Result {
  FixtureCase input;
  int native_variant = -1;
  int decision_variant = -1;
  bool decision_supported = false;
  bool decision_enabled = false;
  bool decision_adopted = false;
  const char *decision_reason = "not_applicable";
  int provider = -1;
  int rejection = -1;
  int architecture = -1;
  int tile = -1;
  int activation_pack = -1;
  int inner_product = -1;
  bool provider_supported = false;
  int concrete_provider = -1;
  int concrete_tile = -1;
  int concrete_inner_product = -1;
  uint64_t activation_value_bytes = 0U;
  uint64_t activation_scale_offset = 0U;
  uint64_t activation_scale_bytes = 0U;
  uint64_t activation_workspace_bytes = 0U;
  uint64_t total_workspace_bytes = 0U;
  uint64_t split4_workspace_bytes = 0U;
  uint64_t staging_workspace_bytes = 0U;
};

KernelVariant native_variant(const FixtureCase &item) {
  const char *target = item.target_name;
  switch (item.format) {
  case MatmulFormat::Mxfp8E4M3W8A8:
    return sllm_matmul_kernel::select_mxfp8_variant(item.m, item.k, item.n,
                                                     target);
  case MatmulFormat::Mxfp8E4M3W8A16:
    return item.m == 1U &&
                   sllm_matmul_kernel::phase85_mxfp_m1_a16_shape(item.m,
                                                                  item.k,
                                                                  item.n)
               ? KernelVariant::Mxfp8W8A16M1Col2
               : KernelVariant::Baseline;
  case MatmulFormat::Mxfp6E3M2W6A6:
    return sllm_matmul_kernel::select_mxfp6_variant(item.m, item.k, item.n,
                                                     target);
  case MatmulFormat::Mxfp6E3M2W6A16:
    return item.m == 1U &&
                   sllm_matmul_kernel::phase85_mxfp_m1_a16_shape(item.m,
                                                                  item.k,
                                                                  item.n)
               ? KernelVariant::Mxfp6W6A16M1Col2
               : KernelVariant::Baseline;
  case MatmulFormat::Nvfp4W4A16:
    return sllm_matmul_kernel::select_nvfp4_variant(item.m);
  case MatmulFormat::Nvfp4W4A4:
    return sllm_matmul_kernel::select_nvfp4_w4a4_variant(
        item.m, item.k, item.n, target);
  case MatmulFormat::Mxfp4W4A4:
    return sllm_matmul_kernel::select_mxfp4_variant(item.m);
  case MatmulFormat::Fp8OuterE4M3W8A8:
    return sllm_matmul_kernel::select_fp8_outer_variant(item.m, item.k,
                                                         item.n, target);
  }
  return KernelVariant::Baseline;
}

Result evaluate(const FixtureCase &item) {
  Result result{item};
  result.native_variant = static_cast<int>(native_variant(item));
  if (item.format == MatmulFormat::Nvfp4W4A4) {
    const auto decision = sllm_matmul_kernel::select_nvfp4_w4a4_decision(
        item.m, item.k, item.n, item.target_name);
    result.decision_variant = static_cast<int>(decision.variant);
    result.decision_supported = decision.supported;
    result.decision_enabled = decision.enabled;
    result.decision_adopted = decision.adopted;
    result.decision_reason = decision.reason;
  } else if (item.format == MatmulFormat::Fp8OuterE4M3W8A8) {
    const auto decision = sllm_matmul_kernel::select_fp8_outer_decision(
        item.m, item.k, item.n, item.target_name);
    result.decision_variant = static_cast<int>(decision.variant);
    result.decision_supported = decision.supported;
    result.decision_enabled = decision.enabled;
    result.decision_adopted = decision.adopted;
    result.decision_reason = decision.reason;
  }

  const auto request = sllm_lowp::make_provider_request(
      item.format, item.target, item.m, item.n, item.k);
  const auto plan = sllm_lowp::prepare_provider_plan(request);
  result.provider = static_cast<int>(plan.provider);
  result.rejection = static_cast<int>(plan.rejection);
  result.architecture = static_cast<int>(plan.architecture);
  result.tile = static_cast<int>(plan.tile);
  result.activation_pack = static_cast<int>(plan.activation_pack);
  result.inner_product = static_cast<int>(plan.inner_product);
  result.provider_supported = plan.supported();
  const auto concrete = plan.supported()
                            ? selection_fixture_concrete_provider_plan(
                                  plan,
                                  static_cast<KernelVariant>(result.native_variant))
                            : std::nullopt;
  if (concrete.has_value()) {
    result.concrete_provider = static_cast<int>(concrete->provider);
    result.concrete_tile = static_cast<int>(concrete->tile);
    result.concrete_inner_product = static_cast<int>(concrete->inner_product);
    const uint64_t block_size = concrete->block_contract.activation_block_size;
    const uint64_t bits = concrete->block_contract.activation_bits;
    if (concrete->format == MatmulFormat::Fp8OuterE4M3W8A8 &&
        concrete->supported()) {
      result.activation_value_bytes = concrete->m * concrete->k;
      result.activation_scale_offset =
          (result.activation_value_bytes + UINT64_C(3)) & ~UINT64_C(3);
      result.activation_scale_bytes = concrete->m * UINT64_C(4);
      result.activation_workspace_bytes =
          result.activation_scale_offset + result.activation_scale_bytes;
    } else if (block_size != 0U && bits != 0U && concrete->supported()) {
      result.activation_value_bytes =
          concrete->m * ((concrete->k * bits + UINT64_C(7)) / UINT64_C(8));
      result.activation_scale_offset = result.activation_value_bytes;
      result.activation_scale_bytes =
          concrete->m * ((concrete->k + block_size - UINT64_C(1)) / block_size);
      (void)selection_fixture_activation_workspace_bytes(
          *concrete, &result.activation_workspace_bytes);
    }
    result.total_workspace_bytes = result.activation_workspace_bytes;
    if (concrete->format == MatmulFormat::Nvfp4W4A4) {
      const bool split4 =
          (concrete->target == sllm_lowp::ExactTarget::Gfx1030 &&
           native_variant(item) == KernelVariant::Nvfp4W4A4PrefillDp4a64x64 &&
           sllm_matmul_kernel::phase78_gfx1030_nvfp4_w4a4_split4_shape(
               concrete->m, concrete->k, concrete->n)) ||
          (concrete->target == sllm_lowp::ExactTarget::Gfx1201 &&
           native_variant(item) == KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x64 &&
           sllm_matmul_kernel::phase78_gfx1201_nvfp4_w4a4_split4_shape(
               concrete->m, concrete->k, concrete->n));
      if (split4) {
        uint64_t partial = 0U;
        uint64_t total = 0U;
        if (sllm_matmul_kernel::phase78_nvfp4_w4a4_split4_workspace(
                concrete->m, concrete->k, concrete->n, &partial, &total)) {
          result.total_workspace_bytes = total;
          result.split4_workspace_bytes = total;
        }
      }
      if (native_variant(item) ==
              KernelVariant::Nvfp4W4A4PrefillGfx1201F16Staging &&
          sllm_matmul_kernel::phase78_gfx1201_nvfp4_w4a4_f16_staging_shape(
              concrete->m, concrete->k, concrete->n)) {
        (void)sllm_matmul_kernel::qwen38_f16_staging_workspace_reservation(
            concrete->m, &result.staging_workspace_bytes);
      }
    }
  }
  return result;
}

std::string json_quote(const char *value) {
  std::string result = "\"";
  for (const char *cursor = value == nullptr ? "" : value; *cursor != '\0';
       ++cursor) {
    if (*cursor == '\\' || *cursor == '"') {
      result.push_back('\\');
    }
    result.push_back(*cursor);
  }
  result.push_back('"');
  return result;
}

void print_csv(const std::vector<Result> &rows) {
  std::cout << "format,target,m,k,n,native_variant,decision_variant,"
               "decision_supported,decision_enabled,decision_adopted,"
               "decision_reason,provider,rejection,architecture,tile,"
               "activation_pack,inner_product,provider_supported,"
               "concrete_provider,concrete_tile,concrete_inner_product,"
               "activation_value_bytes,activation_scale_offset,"
               "activation_scale_bytes,activation_workspace_bytes,"
               "total_workspace_bytes,split4_workspace_bytes,"
               "staging_workspace_bytes\n";
  for (const auto &row : rows) {
    const auto &input = row.input;
    std::cout << format_name(input.format) << ',' << input.target_name << ','
              << input.m
              << ',' << input.k << ',' << input.n << ',' << row.native_variant
              << ',' << row.decision_variant << ','
              << (row.decision_supported ? 1 : 0) << ','
              << (row.decision_enabled ? 1 : 0) << ','
              << (row.decision_adopted ? 1 : 0) << ',' << row.decision_reason
              << ',' << row.provider << ',' << row.rejection << ','
              << row.architecture << ',' << row.tile << ','
              << row.activation_pack << ',' << row.inner_product << ','
              << (row.provider_supported ? 1 : 0) << ','
              << row.concrete_provider << ',' << row.concrete_tile << ','
              << row.concrete_inner_product << ','
              << row.activation_value_bytes << ','
              << row.activation_scale_offset << ','
              << row.activation_scale_bytes << ','
              << row.activation_workspace_bytes << ','
              << row.total_workspace_bytes << ','
              << row.split4_workspace_bytes << ','
              << row.staging_workspace_bytes << '\n';
  }
}

void print_json(const std::vector<Result> &rows) {
  std::cout << "{\n  \"schema_version\": \"lowp-selection-baseline-v1\",\n"
               "  \"source\": \"native/hip/src production selectors\",\n"
               "  \"case_count\": "
            << rows.size() << ",\n  \"cases\": [\n";
  for (size_t index = 0; index < rows.size(); ++index) {
    const auto &row = rows[index];
    const auto &input = row.input;
    std::cout << "    {\"format\": " << json_quote(format_name(input.format))
              << ", \"target\": " << json_quote(input.target_name)
              << ", \"m\": " << input.m << ", \"k\": " << input.k
              << ", \"n\": " << input.n
              << ", \"native_variant\": " << row.native_variant
              << ", \"decision_variant\": " << row.decision_variant
              << ", \"decision_supported\": "
              << (row.decision_supported ? "true" : "false")
              << ", \"decision_enabled\": "
              << (row.decision_enabled ? "true" : "false")
              << ", \"decision_adopted\": "
              << (row.decision_adopted ? "true" : "false")
              << ", \"decision_reason\": "
              << json_quote(row.decision_reason) << ", \"provider\": "
              << row.provider << ", \"rejection\": " << row.rejection
              << ", \"architecture\": " << row.architecture
              << ", \"tile\": " << row.tile
              << ", \"activation_pack\": " << row.activation_pack
              << ", \"inner_product\": " << row.inner_product
              << ", \"provider_supported\": "
              << (row.provider_supported ? "true" : "false")
              << ", \"concrete_provider\": " << row.concrete_provider
              << ", \"concrete_tile\": " << row.concrete_tile
              << ", \"concrete_inner_product\": "
              << row.concrete_inner_product
              << ", \"activation_value_bytes\": "
              << row.activation_value_bytes
              << ", \"activation_scale_offset\": "
              << row.activation_scale_offset
              << ", \"activation_scale_bytes\": "
              << row.activation_scale_bytes
              << ", \"activation_workspace_bytes\": "
              << row.activation_workspace_bytes
              << ", \"total_workspace_bytes\": "
              << row.total_workspace_bytes
              << ", \"split4_workspace_bytes\": "
              << row.split4_workspace_bytes
              << ", \"staging_workspace_bytes\": "
              << row.staging_workspace_bytes << "}"
              << (index + 1 == rows.size() ? "\n" : ",\n");
  }
  std::cout << "  ]\n}\n";
}

} // namespace

#ifndef SLLM_LOWP_SELECTION_FIXTURE_NO_MAIN
int main(int argc, char **argv) {
  const auto cases = make_cases();
  std::vector<Result> rows;
  rows.reserve(cases.size());
  for (const auto &item : cases) {
    rows.push_back(evaluate(item));
  }
  if (argc > 1 && std::string(argv[1]) == "--csv") {
    print_csv(rows);
  } else {
    print_json(rows);
  }
  return 0;
}
#endif
