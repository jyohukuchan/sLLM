#include "../src/paged_kv_slab_plan.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using sllm_paged_kv::BlockLocation;
using sllm_paged_kv::DescriptorOffsetPlan;
using sllm_paged_kv::SlabPlan;
using sllm_paged_kv::SlabPlanInput;

void expect(const bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void expect_throw(Function &&function, const std::string &message) {
  bool threw = false;
  try {
    function();
  } catch (const Exception &) {
    threw = true;
  }
  expect(threw, message);
}

SlabPlan make_plan(const std::uint64_t value, const std::uint64_t scale,
                   const std::uint64_t outer,
                   const std::uint64_t logical_tokens,
                   const std::uint64_t pool_blocks) {
  return SlabPlan::make(
      SlabPlanInput{value, scale, outer, logical_tokens, pool_blocks});
}

void expect_plane_bytes(const SlabPlan &plan, const std::uint64_t key_value,
                        const std::uint64_t scale, const std::uint64_t outer,
                        const std::string &label) {
  const auto &bytes = plan.plane_bytes_per_block();
  expect(bytes[0] == key_value && bytes[1] == key_value,
         label + " key/value plane bytes mismatch");
  expect(bytes[2] == scale && bytes[3] == scale,
         label + " key/value scale bytes mismatch");
  expect(bytes[4] == outer && bytes[5] == outer,
         label + " key/value outer-scale bytes mismatch");
}

void expect_offsets(const SlabPlan &plan,
                    const std::array<std::size_t, 6U> &expected,
                    const std::array<bool, 6U> &present,
                    const std::string &label) {
  expect(plan.plane_offsets() == expected, label + " plane offsets mismatch");
  expect(plan.present() == present, label + " optional plane flags mismatch");
}

void test_exact_qwen_layouts() {
  // Qwen H=4,D=256.  The supplied per-token values are for one K or V.
  const SlabPlan fp16 = make_plan(2048U, 0U, 0U, 128U, 8U);
  expect_plane_bytes(fp16, 262144U, 0U, 0U, "FP16");
  expect(fp16.block_stride() == 524288U, "FP16 block stride mismatch");
  expect(fp16.slab_footprint() == 4194304U, "FP16 slab footprint mismatch");
  expect_offsets(fp16, {0U, 262144U, 0U, 0U, 0U, 0U},
                 {true, true, false, false, false, false}, "FP16");

  const SlabPlan mxfp8 = make_plan(1024U, 32U, 0U, 128U, 8U);
  expect_plane_bytes(mxfp8, 131072U, 4096U, 0U, "MXFP8 E4");
  expect(mxfp8.block_stride() == 270336U, "MXFP8 E4 block stride mismatch");
  expect(mxfp8.slab_footprint() == 2162688U,
         "MXFP8 E4 slab footprint mismatch");
  expect_offsets(mxfp8, {0U, 131072U, 262144U, 266240U, 0U, 0U},
                 {true, true, true, true, false, false}, "MXFP8 E4");

  const SlabPlan nvfp4 = make_plan(512U, 64U, 16U, 128U, 8U);
  expect_plane_bytes(nvfp4, 65536U, 8192U, 2048U, "NVFP4");
  expect(nvfp4.block_stride() == 151552U, "NVFP4 block stride mismatch");
  expect(nvfp4.slab_footprint() == 1212416U, "NVFP4 slab footprint mismatch");
  expect_offsets(nvfp4, {0U, 65536U, 131072U, 139264U, 147456U, 149504U},
                 {true, true, true, true, true, true}, "NVFP4");
}

void test_logical_boundaries() {
  for (const std::uint64_t tokens :
       {127U, 128U, 129U, 65535U, 65536U, 65537U}) {
    const SlabPlan plan = make_plan(512U, 64U, 16U, tokens, 513U);
    const std::uint64_t expected_blocks =
        tokens / 128U + (tokens % 128U != 0U ? 1U : 0U);
    expect(plan.logical_block_count() == expected_blocks,
           "logical block boundary mismatch at " + std::to_string(tokens));
    expect(plan.slab_count() == 65U,
           "physical slab count mismatch at " + std::to_string(tokens));
  }

  const SlabPlan plan = make_plan(512U, 64U, 16U, 129U, 17U);
  expect(plan.slab_count() == 3U, "partial final slab count mismatch");
  const BlockLocation seven = plan.location(7U);
  const BlockLocation eight = plan.location(8U);
  const BlockLocation fifteen = plan.location(15U);
  const BlockLocation sixteen = plan.location(16U);
  expect(seven.slab == 0U && seven.slot == 7U, "slab 0 edge mismatch");
  expect(eight.slab == 1U && eight.slot == 0U, "slab 1 edge mismatch");
  expect(fifteen.slab == 1U && fifteen.slot == 7U, "slab 1 end mismatch");
  expect(sixteen.slab == 2U && sixteen.slot == 0U, "slab 2 edge mismatch");

  const DescriptorOffsetPlan first = plan.descriptor_offsets(0U);
  const DescriptorOffsetPlan last_of_slab = plan.descriptor_offsets(7U);
  const DescriptorOffsetPlan next_slab = plan.descriptor_offsets(8U);
  expect(first.offsets[0] == 0U && first.offsets[5] == 149504U,
         "first descriptor offsets mismatch");
  expect(last_of_slab.offsets[0] == 7U * plan.block_stride(),
         "last descriptor slot offset mismatch");
  expect(next_slab.offsets == first.offsets,
         "new slab did not restart slot offsets");
  expect(next_slab.present == first.present,
         "new slab descriptor flags changed");
  expect(last_of_slab.offsets[5] + plan.plane_bytes_per_block()[5] ==
             plan.slab_footprint(),
         "last descriptor does not fill the slab footprint");
  expect_throw<std::out_of_range>(
      [&] { (void)plan.location(17U); },
      "physical id beyond a partial slab was accepted");
}

void test_validation_and_overflow() {
  expect_throw<std::invalid_argument>(
      [] { (void)make_plan(0U, 0U, 0U, 128U, 1U); },
      "empty value plane accepted");
  expect_throw<std::invalid_argument>(
      [] { (void)make_plan(1U, 0U, 0U, 0U, 1U); },
      "empty logical capacity accepted");
  expect_throw<std::invalid_argument>(
      [] { (void)make_plan(1U, 0U, 0U, 128U, 0U); },
      "empty physical capacity accepted");
  expect_throw<std::invalid_argument>(
      [] {
        (void)make_plan(1U, 0U, 0U, 128U,
                        std::numeric_limits<std::uint32_t>::max());
      },
      "physical invalid sentinel accepted as a pool block");
  expect_throw<std::invalid_argument>(
      [] {
        (void)make_plan(1U, 0U, 0U,
                        (static_cast<std::uint64_t>(
                             std::numeric_limits<std::uint32_t>::max()) +
                         1U) *
                            128U,
                        1U);
      },
      "logical table ID space overflow accepted");

  expect_throw<std::overflow_error>(
      [] {
        (void)make_plan(std::numeric_limits<std::uint64_t>::max(), 0U, 0U, 128U,
                        1U);
      },
      "per-plane uint64 overflow accepted");
  expect_throw<std::overflow_error>(
      [] {
        const std::uint64_t value =
            std::numeric_limits<std::uint64_t>::max() / 128U;
        (void)make_plan(value, 0U, 0U, 128U, 1U);
      },
      "six-plane stride overflow accepted");
  expect_throw<std::overflow_error>(
      [] {
        const std::uint64_t value =
            std::numeric_limits<std::uint64_t>::max() / 1024U + 1U;
        (void)make_plan(value, 0U, 0U, 128U, 8U);
      },
      "eight-block slab overflow accepted");
}

} // namespace

int main() {
  try {
    test_exact_qwen_layouts();
    test_logical_boundaries();
    test_validation_and_overflow();
  } catch (const std::exception &error) {
    std::cerr << "phase87_stage10_slab_plan_host_test: FAIL: " << error.what()
              << '\n';
    return 1;
  }
  std::cout << "phase87_stage10_slab_plan_host_test: PASS\n";
  return 0;
}
