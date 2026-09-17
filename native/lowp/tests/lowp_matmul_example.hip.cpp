#include "lowp_mxfp8_test_common.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int run_example() {
  hipDeviceProp_t properties{};
  if (!lowp_test::hip_ok(hipGetDeviceProperties(&properties, 0),
                         "hipGetDeviceProperties") ||
      std::string(properties.gcnArchName) != SLLM_TEST_EXPECTED_TARGET) {
    std::cerr << "exact target mismatch: observed=" << properties.gcnArchName
              << " expected=" << SLLM_TEST_EXPECTED_TARGET << '\n';
    return 1;
  }

  constexpr uint64_t m = 1U;
  constexpr uint64_t n = 4U;
  constexpr uint64_t k = 32U;
  const lowp_test::Mxfp8Fixture fixture =
      lowp_test::make_mxfp8_fixture(m, n, k);
  lowp_matmul_plan_t plan{};
  if (!lowp_test::make_mxfp8_plan(m, n, k, &plan)) {
    return 1;
  }
  lowp_test::DeviceBuffer activation;
  lowp_test::DeviceBuffer values;
  lowp_test::DeviceBuffer scales;
  lowp_test::DeviceBuffer weight;
  lowp_test::DeviceBuffer weight_scales;
  lowp_test::DeviceBuffer output;
  lowp_test::DeviceBuffer workspace;
  if (!activation.allocate(fixture.activation.size() * sizeof(uint16_t)) ||
      !values.allocate(static_cast<std::size_t>(plan.activation_value_bytes)) ||
      !scales.allocate(static_cast<std::size_t>(plan.activation_scale_bytes)) ||
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
    return 1;
  }
  if (lowp_quantize_activation(
          &plan, static_cast<const uint16_t *>(activation.data), values.data,
          scales.data, nullptr, nullptr) != LOWP_SUCCESS ||
      !lowp_test::hip_ok(hipDeviceSynchronize(), "example quantize")) {
    return 1;
  }
  std::vector<uint8_t> host_values(fixture.activation.size());
  std::vector<uint8_t> host_scales(
      static_cast<std::size_t>(plan.activation_scale_bytes));
  if (!lowp_test::copy_from_device(host_values.data(), values,
                                   host_values.size()) ||
      !lowp_test::copy_from_device(host_scales.data(), scales,
                                   host_scales.size())) {
    return 1;
  }
  const std::vector<uint16_t> expected =
      lowp_test::oracle_output(fixture, host_values, host_scales);
  lowp_matmul_buffers_t buffers{};
  buffers.struct_size = sizeof(buffers);
  buffers.flags = LOWP_ACTIVATION_PREQUANTIZED;
  buffers.activation = values.data;
  buffers.activation_scales = scales.data;
  buffers.weight = weight.data;
  buffers.weight_scales = weight_scales.data;
  buffers.output = static_cast<uint16_t *>(output.data);
  buffers.workspace = workspace.data;
  buffers.workspace_bytes = plan.workspace_bytes;
  if (lowp_matmul_launch(&plan, &buffers, nullptr) != LOWP_SUCCESS ||
      !lowp_test::hip_ok(hipDeviceSynchronize(), "example matmul")) {
    return 1;
  }
  std::vector<uint16_t> observed(expected.size());
  if (!lowp_test::copy_from_device(observed.data(), output,
                                   observed.size() * sizeof(uint16_t)) ||
      !lowp_test::check_output(observed, expected, "example")) {
    return 1;
  }
  std::cout << "lowp_mxfp8_example target=" << SLLM_TEST_EXPECTED_TARGET
            << " M=" << m << " N=" << n << " K=" << k << " status=PASS\n";
  return 0;
}

int main() { return run_example() == 0 && lowp_test::hip_cleanup_ok() ? 0 : 1; }
