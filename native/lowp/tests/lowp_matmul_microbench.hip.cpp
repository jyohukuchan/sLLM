#include "lowp_mxfp8_test_common.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool run_benchmark() {
  constexpr uint64_t m = 8U;
  constexpr uint64_t n = 256U;
  constexpr uint64_t k = 256U;
  constexpr uint32_t warmups = 1U;
  constexpr uint32_t measured = 3U;
  const lowp_test::Mxfp8Fixture fixture =
      lowp_test::make_mxfp8_fixture(m, n, k);
  lowp_matmul_plan_t plan{};
  if (!lowp_test::make_mxfp8_plan(m, n, k, &plan)) {
    return false;
  }
  lowp_test::DeviceBuffer activation;
  lowp_test::DeviceBuffer weight;
  lowp_test::DeviceBuffer weight_scales;
  lowp_test::DeviceBuffer output;
  lowp_test::DeviceBuffer workspace;
  if (!activation.allocate(fixture.activation.size() * sizeof(uint16_t)) ||
      !weight.allocate(fixture.weight.size()) ||
      !weight_scales.allocate(fixture.weight_scales.size()) ||
      !output.allocate(static_cast<std::size_t>(plan.output_bytes)) ||
      !workspace.allocate(static_cast<std::size_t>(
          plan.workspace_bytes == 0U ? 1U : plan.workspace_bytes)) ||
      !lowp_test::copy_to_device(activation, fixture.activation.data(),
                                 fixture.activation.size() *
                                     sizeof(uint16_t)) ||
      !lowp_test::copy_to_device(weight, fixture.weight.data(),
                                 fixture.weight.size()) ||
      !lowp_test::copy_to_device(weight_scales, fixture.weight_scales.data(),
                                 fixture.weight_scales.size())) {
    return false;
  }
  lowp_matmul_buffers_t buffers{};
  buffers.struct_size = sizeof(buffers);
  buffers.flags = 0U;
  buffers.activation = activation.data;
  buffers.weight = weight.data;
  buffers.weight_scales = weight_scales.data;
  buffers.output = static_cast<uint16_t *>(output.data);
  buffers.workspace = workspace.data;
  buffers.workspace_bytes = plan.workspace_bytes;

  for (uint32_t iteration = 0U; iteration < warmups; ++iteration) {
    if (lowp_matmul_launch(&plan, &buffers, nullptr) != LOWP_SUCCESS ||
        !lowp_test::hip_ok(hipDeviceSynchronize(), "microbench warmup")) {
      return false;
    }
  }
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  bool valid =
      lowp_test::hip_ok(hipEventCreate(&start), "hipEventCreate start") &&
      lowp_test::hip_ok(hipEventCreate(&stop), "hipEventCreate stop");
  std::vector<float> milliseconds;
  milliseconds.reserve(measured);
  for (uint32_t iteration = 0U; iteration < measured && valid; ++iteration) {
    valid = lowp_test::hip_ok(hipEventRecord(start, nullptr),
                              "hipEventRecord start") &&
            lowp_matmul_launch(&plan, &buffers, nullptr) == LOWP_SUCCESS &&
            lowp_test::hip_ok(hipEventRecord(stop, nullptr),
                              "hipEventRecord stop") &&
            lowp_test::hip_ok(hipEventSynchronize(stop),
                              "hipEventSynchronize stop");
    float elapsed = 0.0F;
    valid =
        valid && lowp_test::hip_ok(hipEventElapsedTime(&elapsed, start, stop),
                                   "hipEventElapsedTime");
    if (valid) {
      milliseconds.push_back(elapsed);
    }
  }
  if (start != nullptr) {
    valid =
        lowp_test::hip_ok(hipEventDestroy(start), "hipEventDestroy start") &&
        valid;
  }
  if (stop != nullptr) {
    valid = lowp_test::hip_ok(hipEventDestroy(stop), "hipEventDestroy stop") &&
            valid;
  }
  std::vector<uint16_t> observed(
      static_cast<std::size_t>(plan.output_bytes / 2U));
  valid =
      valid && lowp_test::copy_from_device(observed.data(), output,
                                           observed.size() * sizeof(uint16_t));
  for (const uint16_t value : observed) {
    valid = valid && lowp_test::finite_bf16(value);
  }
  if (valid && milliseconds.size() == measured) {
    std::sort(milliseconds.begin(), milliseconds.end());
    std::cout << std::fixed << std::setprecision(6)
              << "lowp_mxfp8_microbench target=" << SLLM_TEST_EXPECTED_TARGET
              << " M=" << m << " N=" << n << " K=" << k
              << " provider=" << plan.provider << " variant=" << plan.variant
              << " warmups=" << warmups << " measured=" << measured
              << " median_ms=" << milliseconds[milliseconds.size() / 2U]
              << " status=PASS\n";
  }
  return valid && milliseconds.size() == measured;
}

} // namespace

int main() {
  hipDeviceProp_t properties{};
  if (!lowp_test::hip_ok(hipGetDeviceProperties(&properties, 0),
                         "hipGetDeviceProperties") ||
      std::string(properties.gcnArchName) != SLLM_TEST_EXPECTED_TARGET) {
    std::cerr << "exact target mismatch: observed=" << properties.gcnArchName
              << " expected=" << SLLM_TEST_EXPECTED_TARGET << '\n';
    return 1;
  }
  return run_benchmark() && lowp_test::hip_cleanup_ok() ? 0 : 1;
}
