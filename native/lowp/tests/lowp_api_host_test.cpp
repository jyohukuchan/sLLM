#include <lowp/lowp.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

lowp_matmul_request_t request(const lowp_format_t format,
                              const lowp_target_t target, const uint64_t m,
                              const uint64_t n, const uint64_t k) {
  return {sizeof(lowp_matmul_request_t),
          LOWP_ABI_VERSION,
          format,
          target,
          LOWP_ROW_MAJOR_BLOCK_SCALED,
          LOWP_ROW_MAJOR_BLOCK_SCALED,
          m,
          n,
          k};
}

void expect_rejected(const lowp_matmul_request_t &input,
                     const lowp_status_t expected_status,
                     const uint32_t expected_rejection) {
  lowp_matmul_plan_t plan{};
  assert(lowp_matmul_plan(&input, &plan) == expected_status);
  assert(plan.rejection == expected_rejection);
  assert(plan.supported == 0U);
}

} // namespace

int main() {
  assert(lowp_version() == LOWP_ABI_VERSION);
  assert(lowp_target_from_name("gfx1030") == LOWP_TARGET_GFX1030);
  assert(lowp_target_from_name("gfx1201") == LOWP_TARGET_GFX1201);
  assert(lowp_target_from_name("gfx942:sramecc+:xnack-") ==
         LOWP_TARGET_GFX942_SRAMECC_ON_XNACK_OFF);
  assert(lowp_target_from_name(nullptr) == LOWP_TARGET_UNKNOWN);
  assert(lowp_target_from_name("gfx9999") == LOWP_TARGET_UNKNOWN);
  assert(std::strcmp(lowp_target_name(LOWP_TARGET_GFX1030), "gfx1030") == 0);
  assert(std::strcmp(lowp_target_name(LOWP_TARGET_UNKNOWN), "") == 0);

  lowp_format_info_t info{};
  assert(lowp_get_format_info(LOWP_MXFP8_E4M3_W8A8, &info) == LOWP_SUCCESS);
  assert(info.weight_bits == 8U && info.activation_bits == 8U);
  assert(info.weight_block_size == 32U && info.activation_block_size == 32U);
  assert(lowp_get_format_info(LOWP_MXFP4_W4A8_V1, &info) == LOWP_SUCCESS);
  assert(info.weight_bits == 4U && info.activation_bits == 8U);
  assert(lowp_get_format_info(LOWP_MXFP8_E4M3_W8A8, nullptr) ==
         LOWP_INVALID_ARGUMENT);
  assert(lowp_get_format_info(4U, &info) == LOWP_NOT_SUPPORTED);

  const auto valid =
      request(LOWP_MXFP8_E4M3_W8A8, LOWP_TARGET_GFX1030, 1U, 9216U, 2048U);
  lowp_matmul_plan_t plan{};
  assert(lowp_matmul_plan(&valid, &plan) == LOWP_SUCCESS);
  assert(plan.supported != 0U);
  assert(plan.provider == 1U && plan.variant == 99U);
  assert(plan.tile == 1U && plan.inner_product == 1U);
  assert(plan.activation_value_bytes == 2048U);
  assert(plan.activation_scale_offset == 2048U);
  assert(plan.activation_scale_bytes == 64U);
  assert(plan.workspace_bytes == 2112U);

  // Partial NVFP4 blocks are supported by the historical baseline launcher.
  // Its selector audit flag is not an operational rejection criterion.
  auto partial_nvfp4 =
      request(LOWP_NVFP4_W4A4, LOWP_TARGET_GFX1030, 1U, 17U, 15U);
  lowp_matmul_plan_t partial_plan{};
  assert(lowp_matmul_plan(&partial_nvfp4, &partial_plan) == LOWP_SUCCESS);
  assert(partial_plan.supported == 1U && partial_plan.selector_supported == 0U);
  assert(partial_plan.activation_value_bytes == 8U);
  assert(partial_plan.activation_scale_bytes == 1U);
  assert(partial_plan.workspace_bytes == 9U);

  auto wrong_size = valid;
  wrong_size.struct_size = sizeof(wrong_size) - 1U;
  expect_rejected(wrong_size, LOWP_INVALID_ARGUMENT, 0U);
  auto wrong_version = valid;
  wrong_version.version = LOWP_ABI_VERSION + 1U;
  expect_rejected(wrong_version, LOWP_INVALID_ARGUMENT, 0U);
  expect_rejected(request(LOWP_MXFP8_E4M3_W8A8, 99U, 1U, 9216U, 2048U),
                  LOWP_NOT_SUPPORTED, 1U);
  expect_rejected(
      request(LOWP_MXFP8_E4M3_W8A8, LOWP_TARGET_GFX1030, 0U, 9216U, 2048U),
      LOWP_NOT_SUPPORTED, 3U);
  expect_rejected(
      request(LOWP_MXFP8_E4M3_W8A8, LOWP_TARGET_GFX1030, 1U, 9216U, 2049U),
      LOWP_NOT_SUPPORTED, 4U);

  auto bad_layout = valid;
  bad_layout.weight_layout = 99U;
  expect_rejected(bad_layout, LOWP_INVALID_ARGUMENT, 5U);
  auto overflow = valid;
  overflow.m = std::numeric_limits<uint64_t>::max();
  expect_rejected(overflow, LOWP_INVALID_ARGUMENT, 0U);

  auto reserved = valid;
  reserved.format = 4U;
  expect_rejected(reserved, LOWP_NOT_SUPPORTED, 6U);
  auto public_mxfp4 = valid;
  public_mxfp4.format = LOWP_MXFP4_W4A8_V1;
  expect_rejected(public_mxfp4, LOWP_NOT_SUPPORTED, 6U);

  auto unscaled_layout = valid;
  unscaled_layout.activation_layout = LOWP_ROW_MAJOR;
  expect_rejected(unscaled_layout, LOWP_NOT_SUPPORTED, 5U);
  assert(lowp_supported_formats(LOWP_TARGET_GFX1030) != 0U);
  assert((lowp_supported_formats(LOWP_TARGET_GFX1030) & (UINT64_C(1) << 4U)) ==
         0U);
  return 0;
}
