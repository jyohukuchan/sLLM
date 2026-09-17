#include "lowp_mxfp8_test_common.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

struct CaseSpec final {
  uint64_t m;
  uint64_t n;
  uint64_t k;
};

bool run_case(const CaseSpec &spec) {
  const uint64_t m = spec.m;
  const uint64_t n = spec.n;
  const uint64_t k = spec.k;
  const lowp_test::Mxfp8Fixture fixture =
      lowp_test::make_mxfp8_fixture(m, n, k);
  lowp_matmul_plan_t plan{};
  if (!lowp_test::make_mxfp8_plan(m, n, k, &plan)) {
    return false;
  }
  if (plan.activation_value_bytes != m * k ||
      plan.activation_scale_bytes != m * (k / 32U) ||
      plan.weight_value_bytes != n * k ||
      plan.weight_scale_bytes != n * (k / 32U) ||
      plan.output_bytes != m * n * sizeof(uint16_t)) {
    std::cerr << "lowp plan byte contract mismatch\n";
    return false;
  }

  lowp_test::DeviceBuffer activation;
  lowp_test::DeviceBuffer activation_values;
  lowp_test::DeviceBuffer activation_scales;
  lowp_test::DeviceBuffer weight;
  lowp_test::DeviceBuffer weight_scales;
  lowp_test::DeviceBuffer output;
  lowp_test::DeviceBuffer workspace;
  const bool allocated =
      activation.allocate(fixture.activation.size() * sizeof(uint16_t)) &&
      activation_values.allocate(
          static_cast<std::size_t>(plan.activation_value_bytes)) &&
      activation_scales.allocate(
          static_cast<std::size_t>(plan.activation_scale_bytes)) &&
      weight.allocate(fixture.weight.size()) &&
      weight_scales.allocate(fixture.weight_scales.size()) &&
      output.allocate(static_cast<std::size_t>(plan.output_bytes)) &&
      workspace.allocate(static_cast<std::size_t>(
          plan.workspace_bytes == 0U ? 1U : plan.workspace_bytes));
  if (!allocated ||
      !lowp_test::copy_to_device(activation, fixture.activation.data(),
                                 fixture.activation.size() *
                                     sizeof(uint16_t)) ||
      !lowp_test::copy_to_device(weight, fixture.weight.data(),
                                 fixture.weight.size()) ||
      !lowp_test::copy_to_device(weight_scales, fixture.weight_scales.data(),
                                 fixture.weight_scales.size())) {
    return false;
  }

  if (lowp_quantize_activation(&plan,
                               static_cast<const uint16_t *>(activation.data),
                               activation_values.data, activation_scales.data,
                               nullptr, nullptr) != LOWP_SUCCESS) {
    std::cerr << "lowp_quantize_activation failed\n";
    return false;
  }
  if (!lowp_test::hip_ok(hipDeviceSynchronize(), "quantize synchronize")) {
    return false;
  }
  std::vector<uint8_t> quantized_values(fixture.activation.size());
  std::vector<uint8_t> quantized_scales(
      static_cast<std::size_t>(plan.activation_scale_bytes));
  if (!lowp_test::copy_from_device(quantized_values.data(), activation_values,
                                   quantized_values.size()) ||
      !lowp_test::copy_from_device(quantized_scales.data(), activation_scales,
                                   quantized_scales.size())) {
    return false;
  }
  const std::vector<uint16_t> expected =
      lowp_test::oracle_output(fixture, quantized_values, quantized_scales);

  lowp_matmul_buffers_t buffers{};
  buffers.struct_size = sizeof(buffers);
  buffers.flags = LOWP_ACTIVATION_PREQUANTIZED;
  buffers.activation = activation_values.data;
  buffers.activation_scales = activation_scales.data;
  buffers.weight = weight.data;
  buffers.weight_scales = weight_scales.data;
  buffers.output = static_cast<uint16_t *>(output.data);
  buffers.workspace = workspace.data;
  buffers.workspace_bytes = plan.workspace_bytes;
  if (lowp_matmul_launch(&plan, &buffers, nullptr) != LOWP_SUCCESS ||
      !lowp_test::hip_ok(hipDeviceSynchronize(), "prequantized synchronize")) {
    std::cerr << "prequantized lowp_matmul_launch failed\n";
    return false;
  }
  std::vector<uint16_t> observed(expected.size());
  if (!lowp_test::copy_from_device(observed.data(), output,
                                   observed.size() * sizeof(uint16_t)) ||
      !lowp_test::check_output(observed, expected, "prequantized")) {
    return false;
  }

  buffers.flags = 0U;
  buffers.activation = activation.data;
  buffers.activation_scales = nullptr;
  if (lowp_matmul_launch(&plan, &buffers, nullptr) != LOWP_SUCCESS ||
      !lowp_test::hip_ok(hipDeviceSynchronize(), "automatic synchronize")) {
    std::cerr << "automatic lowp_matmul_launch failed\n";
    return false;
  }
  if (!lowp_test::copy_from_device(observed.data(), output,
                                   observed.size() * sizeof(uint16_t)) ||
      !lowp_test::check_output(observed, expected, "automatic")) {
    return false;
  }
  if (!lowp_test::hip_cleanup_ok()) {
    std::cerr << "lowp oracle cleanup observed a hipFree error\n";
    return false;
  }
  std::cout << "lowp_mxfp8_oracle target=" << SLLM_TEST_EXPECTED_TARGET
            << " M=" << m << " N=" << n << " K=" << k
            << " provider=" << plan.provider << " variant=" << plan.variant
            << " status=PASS\n";
  return true;
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
  constexpr std::array<CaseSpec, 8> cases = {{
      {1U, 7U, 32U},
      {2U, 33U, 64U},
      {7U, 65U, 64U},
      {8U, 65U, 64U},
      {9U, 127U, 96U},
      {127U, 256U, 256U},
      {128U, 256U, 256U},
      {129U, 256U, 256U},
  }};
  bool valid = true;
  for (const CaseSpec &spec : cases) {
    valid = run_case(spec) && valid;
  }
  return valid && lowp_test::hip_cleanup_ok() ? 0 : 1;
}
