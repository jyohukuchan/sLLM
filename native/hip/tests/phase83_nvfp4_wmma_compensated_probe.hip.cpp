// Phase 83 ID89 gfx1201 NVFP4 WMMA/Kahan candidate probe.
//
// This standalone probe links the production ID64 control and the opt-in ID89
// candidate.  It intentionally keeps the oracle independent: packed E2M1 and
// E4M3FN bytes are decoded in long double, then the result is rounded to BF16
// with the ID59 ordered tensor-scale epilogue.
//
// Build/link against the production HIP kernel object with
// --offload-arch=gfx1201. The probe is not part of the default production or
// CMake test graph.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string_view>
#include <vector>

extern "C" __global__ void sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_v1(
    const uint8_t *packed_activation, const uint8_t *activation_block_scales,
    const uint8_t *packed_weight, const uint8_t *weight_block_scales,
    const float *weight_tensor_scale, const float *input_tensor_scale,
    uint16_t *output, uint64_t m, uint64_t k, uint64_t n);

extern "C" __global__ void sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_kahan_v1(
    const uint8_t *packed_activation, const uint8_t *activation_block_scales,
    const uint8_t *packed_weight, const uint8_t *weight_block_scales,
    const float *weight_tensor_scale, const float *input_tensor_scale,
    uint16_t *output, uint64_t m, uint64_t k, uint64_t n);

namespace {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kWarmups = 1U;
constexpr uint32_t kMeasured = 2U;
constexpr uint32_t kOracleRows = 2U;
constexpr uint32_t kOracleColumns = 4U;
constexpr float kWeightTensorScale = 0.75F;
constexpr float kInputTensorScale = 1.125F;
constexpr uint64_t kBoundaryK = 5120U;
constexpr uint64_t kBoundaryN = 17408U;
constexpr uint64_t kDownK = 17408U;
constexpr uint64_t kDownN = 5120U;

struct Shape final {
  uint64_t m;
  uint64_t k;
  uint64_t n;
  const char *name;
};

constexpr std::array<Shape, 5> kShapes = {{
    {63U, kBoundaryK, kBoundaryN, "wide-m63"},
    {64U, kBoundaryK, kBoundaryN, "wide-m64"},
    {65U, kBoundaryK, kBoundaryN, "wide-m65"},
    {1024U, kBoundaryK, kBoundaryN, "wide-m1024"},
    {1024U, kDownK, kDownN, "down-m1024"},
}};

bool hip_ok(hipError_t status, const char *operation);

struct DeviceBuffer final {
  void *ptr = nullptr;

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() {
    if (ptr != nullptr) {
      (void)hipFree(ptr);
    }
  }

  bool allocate(const size_t bytes) {
    return hip_ok(hipMalloc(&ptr, bytes), "hipMalloc");
  }

  template <typename T> T *as() const { return static_cast<T *>(ptr); }
};

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::cerr << operation << ": " << hipGetErrorString(status) << '\n';
  return false;
}

uint32_t next_u32(uint32_t &state) {
  state ^= state << 13U;
  state ^= state >> 17U;
  state ^= state << 5U;
  return state;
}

uint8_t e2m1_code(const uint8_t packed, const uint32_t nibble) {
  return static_cast<uint8_t>((packed >> (nibble * 4U)) & 0x0fU);
}

long double e2m1_value(const uint8_t code) {
  constexpr std::array<long double, 8> positive = {0.0L, 0.5L, 1.0L, 1.5L,
                                                   2.0L, 3.0L, 4.0L, 6.0L};
  const long double magnitude = positive[code & 7U];
  return (code & 8U) == 0U ? magnitude : -magnitude;
}

long double e4m3fn_value(const uint8_t bits) {
  const bool negative = (bits & 0x80U) != 0U;
  const uint32_t magnitude = bits & 0x7fU;
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  long double value = 0.0L;
  if (exponent == 0U) {
    value = static_cast<long double>(mantissa) * 0x1p-9L;
  } else if (magnitude == 0x7fU) {
    // The generated scale alphabet never emits this code. Keep the oracle
    // finite if a future seed accidentally reaches the reserved NaN code.
    return std::numeric_limits<long double>::quiet_NaN();
  } else {
    value = static_cast<long double>(8U + mantissa) *
            std::ldexp(1.0L, static_cast<int>(exponent) - 10);
  }
  return negative ? -value : value;
}

uint16_t float_to_bf16_rne(const float value) {
  uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
    if ((bits & UINT32_C(0x007fffff)) != 0U) {
      return static_cast<uint16_t>(((bits >> 16U) & UINT32_C(0x8000)) |
                                   UINT32_C(0x7fc0) |
                                   ((bits >> 16U) & UINT32_C(0x003f)));
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

float bf16_to_float(const uint16_t bits) {
  const uint32_t raw = static_cast<uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

uint32_t bf16_order(const uint16_t bits) {
  return (bits & UINT16_C(0x8000)) != 0U
             ? static_cast<uint32_t>(~bits & UINT16_C(0xffff))
             : static_cast<uint32_t>(bits | UINT16_C(0x8000));
}

uint32_t bf16_ulp_distance(const uint16_t lhs, const uint16_t rhs) {
  const uint32_t left = bf16_order(lhs);
  const uint32_t right = bf16_order(rhs);
  return left > right ? left - right : right - left;
}

// Positive finite E4M3FN scales only, with several exponents and mantissas.
uint8_t generated_scale(uint32_t &state) {
  constexpr std::array<uint8_t, 12> values = {0x20U, 0x24U, 0x28U, 0x2cU,
                                              0x30U, 0x34U, 0x38U, 0x3cU,
                                              0x40U, 0x44U, 0x48U, 0x4cU};
  return values[next_u32(state) % values.size()];
}

void fill_case(const Shape shape, const uint32_t seed,
               std::vector<uint8_t> &activation,
               std::vector<uint8_t> &activation_scales,
               std::vector<uint8_t> &weight,
               std::vector<uint8_t> &weight_scales) {
  uint32_t state = seed;
  const size_t activation_bytes =
      static_cast<size_t>(shape.m) * static_cast<size_t>(shape.k / 2U);
  const size_t activation_scale_bytes =
      static_cast<size_t>(shape.m) * static_cast<size_t>(shape.k / 16U);
  const size_t weight_bytes =
      static_cast<size_t>(shape.n) * static_cast<size_t>(shape.k / 2U);
  const size_t weight_scale_bytes =
      static_cast<size_t>(shape.n) * static_cast<size_t>(shape.k / 16U);
  activation.resize(activation_bytes);
  activation_scales.resize(activation_scale_bytes);
  weight.resize(weight_bytes);
  weight_scales.resize(weight_scale_bytes);
  for (uint8_t &value : activation) {
    value = static_cast<uint8_t>(next_u32(state));
  }
  for (uint8_t &value : weight) {
    value = static_cast<uint8_t>(next_u32(state));
  }
  for (uint8_t &value : activation_scales) {
    value = generated_scale(state);
  }
  for (uint8_t &value : weight_scales) {
    value = generated_scale(state);
  }
}

bool copy_to_device(const std::vector<uint8_t> &host, DeviceBuffer &device) {
  return hip_ok(
      hipMemcpy(device.ptr, host.data(), host.size(), hipMemcpyHostToDevice),
      "copy input");
}

bool launch_control(const dim3 grid, const dim3 block, const DeviceBuffer &a,
                    const DeviceBuffer &as, const DeviceBuffer &w,
                    const DeviceBuffer &ws, const DeviceBuffer &wts,
                    const DeviceBuffer &ats, DeviceBuffer &output,
                    const Shape shape) {
  hipLaunchKernelGGL(sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_v1, grid, block,
                     0U, 0, a.as<uint8_t>(), as.as<uint8_t>(), w.as<uint8_t>(),
                     ws.as<uint8_t>(), wts.as<float>(), ats.as<float>(),
                     output.as<uint16_t>(), shape.m, shape.k, shape.n);
  return hip_ok(hipGetLastError(), "ID64 launch");
}

bool launch_candidate(const dim3 grid, const dim3 block, const DeviceBuffer &a,
                      const DeviceBuffer &as, const DeviceBuffer &w,
                      const DeviceBuffer &ws, const DeviceBuffer &wts,
                      const DeviceBuffer &ats, DeviceBuffer &output,
                      const Shape shape) {
  hipLaunchKernelGGL(sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_kahan_v1, grid,
                     block, 0U, 0, a.as<uint8_t>(), as.as<uint8_t>(),
                     w.as<uint8_t>(), ws.as<uint8_t>(), wts.as<float>(),
                     ats.as<float>(), output.as<uint16_t>(), shape.m, shape.k,
                     shape.n);
  return hip_ok(hipGetLastError(), "ID89 launch");
}

long double oracle_value(const Shape shape, const uint64_t row,
                         const uint64_t column,
                         const std::vector<uint8_t> &activation,
                         const std::vector<uint8_t> &activation_scales,
                         const std::vector<uint8_t> &weight,
                         const std::vector<uint8_t> &weight_scales) {
  const uint64_t packed_row_bytes = shape.k / 2U;
  const uint64_t blocks_per_row = shape.k / 16U;
  long double accumulator = 0.0L;
  for (uint64_t block = 0U; block < blocks_per_row; ++block) {
    long double dot = 0.0L;
    for (uint64_t index = 0U; index < 16U; ++index) {
      const uint64_t inner = block * 16U + index;
      const uint8_t a_byte =
          activation[static_cast<size_t>(row * packed_row_bytes + inner / 2U)];
      const uint8_t w_byte =
          weight[static_cast<size_t>(column * packed_row_bytes + inner / 2U)];
      dot += e2m1_value(e2m1_code(a_byte, static_cast<uint32_t>(inner & 1U))) *
             e2m1_value(e2m1_code(w_byte, static_cast<uint32_t>(inner & 1U)));
    }
    const long double activation_scale = e4m3fn_value(
        activation_scales[static_cast<size_t>(row * blocks_per_row + block)]);
    const long double weight_scale = e4m3fn_value(
        weight_scales[static_cast<size_t>(column * blocks_per_row + block)]);
    accumulator += dot * activation_scale * weight_scale;
  }
  return (accumulator * static_cast<long double>(kWeightTensorScale)) *
         static_cast<long double>(kInputTensorScale);
}

struct Metrics final {
  uint32_t max_control_candidate_ulp = 0U;
  uint32_t max_candidate_oracle_ulp = 0U;
  float max_candidate_oracle_abs = 0.0F;
  bool candidate_repeat_equal = true;
};

Metrics inspect_outputs(const Shape shape, const std::vector<uint16_t> &control,
                        const std::vector<uint16_t> &candidate,
                        const std::vector<uint16_t> &repeat,
                        const std::vector<uint8_t> &activation,
                        const std::vector<uint8_t> &activation_scales,
                        const std::vector<uint8_t> &weight,
                        const std::vector<uint8_t> &weight_scales) {
  Metrics metrics{};
  for (size_t index = 0U; index < candidate.size(); ++index) {
    metrics.max_control_candidate_ulp =
        std::max(metrics.max_control_candidate_ulp,
                 bf16_ulp_distance(control[index], candidate[index]));
    if (candidate[index] != repeat[index]) {
      metrics.candidate_repeat_equal = false;
    }
  }
  const uint64_t rows = std::min<uint64_t>(shape.m, kOracleRows);
  const uint64_t columns = std::min<uint64_t>(shape.n, kOracleColumns);
  for (uint64_t row = 0U; row < rows; ++row) {
    for (uint64_t column = 0U; column < columns; ++column) {
      const size_t index = static_cast<size_t>(row * shape.n + column);
      const long double expected =
          oracle_value(shape, row, column, activation, activation_scales,
                       weight, weight_scales);
      const uint16_t expected_bf16 =
          float_to_bf16_rne(static_cast<float>(expected));
      metrics.max_candidate_oracle_ulp =
          std::max(metrics.max_candidate_oracle_ulp,
                   bf16_ulp_distance(candidate[index], expected_bf16));
      metrics.max_candidate_oracle_abs =
          std::max(metrics.max_candidate_oracle_abs,
                   std::abs(bf16_to_float(candidate[index]) -
                            static_cast<float>(expected)));
    }
  }
  return metrics;
}

bool run_case(const Shape shape, const uint32_t seed) {
  std::vector<uint8_t> activation;
  std::vector<uint8_t> activation_scales;
  std::vector<uint8_t> weight;
  std::vector<uint8_t> weight_scales;
  fill_case(shape, seed, activation, activation_scales, weight, weight_scales);

  DeviceBuffer device_activation;
  DeviceBuffer device_activation_scales;
  DeviceBuffer device_weight;
  DeviceBuffer device_weight_scales;
  DeviceBuffer device_weight_tensor_scale;
  DeviceBuffer device_input_tensor_scale;
  DeviceBuffer device_control;
  DeviceBuffer device_candidate;
  DeviceBuffer device_repeat;
  const size_t output_bytes = static_cast<size_t>(shape.m) *
                              static_cast<size_t>(shape.n) * sizeof(uint16_t);
  if (!device_activation.allocate(activation.size()) ||
      !device_activation_scales.allocate(activation_scales.size()) ||
      !device_weight.allocate(weight.size()) ||
      !device_weight_scales.allocate(weight_scales.size()) ||
      !device_weight_tensor_scale.allocate(sizeof(float)) ||
      !device_input_tensor_scale.allocate(sizeof(float)) ||
      !device_control.allocate(output_bytes) ||
      !device_candidate.allocate(output_bytes) ||
      !device_repeat.allocate(output_bytes)) {
    return false;
  }
  if (!copy_to_device(activation, device_activation) ||
      !copy_to_device(activation_scales, device_activation_scales) ||
      !copy_to_device(weight, device_weight) ||
      !copy_to_device(weight_scales, device_weight_scales) ||
      !hip_ok(hipMemcpy(device_weight_tensor_scale.ptr, &kWeightTensorScale,
                        sizeof(float), hipMemcpyHostToDevice),
              "copy weight tensor scale") ||
      !hip_ok(hipMemcpy(device_input_tensor_scale.ptr, &kInputTensorScale,
                        sizeof(float), hipMemcpyHostToDevice),
              "copy input tensor scale")) {
    return false;
  }

  const dim3 block(kThreads, 1U, 1U);
  const dim3 grid(static_cast<uint32_t>((shape.n + 63U) / 64U),
                  static_cast<uint32_t>((shape.m + 127U) / 128U), 1U);
  for (uint32_t iteration = 0U; iteration < kWarmups; ++iteration) {
    if (!launch_control(grid, block, device_activation,
                        device_activation_scales, device_weight,
                        device_weight_scales, device_weight_tensor_scale,
                        device_input_tensor_scale, device_control, shape) ||
        !launch_candidate(grid, block, device_activation,
                          device_activation_scales, device_weight,
                          device_weight_scales, device_weight_tensor_scale,
                          device_input_tensor_scale, device_candidate, shape) ||
        !hip_ok(hipDeviceSynchronize(), "warmup synchronize")) {
      return false;
    }
  }

  float control_ms = 0.0F;
  float candidate_ms = 0.0F;
  for (uint32_t iteration = 0U; iteration < kMeasured; ++iteration) {
    hipEvent_t control_start = nullptr;
    hipEvent_t control_end = nullptr;
    hipEvent_t candidate_start = nullptr;
    hipEvent_t candidate_end = nullptr;
    if (!hip_ok(hipEventCreate(&control_start), "create control start") ||
        !hip_ok(hipEventCreate(&control_end), "create control end") ||
        !hip_ok(hipEventCreate(&candidate_start), "create candidate start") ||
        !hip_ok(hipEventCreate(&candidate_end), "create candidate end")) {
      return false;
    }
    (void)hipEventRecord(control_start, nullptr);
    if (!launch_control(grid, block, device_activation,
                        device_activation_scales, device_weight,
                        device_weight_scales, device_weight_tensor_scale,
                        device_input_tensor_scale, device_control, shape) ||
        !hip_ok(hipEventRecord(control_end, nullptr), "record control end") ||
        !hip_ok(hipEventSynchronize(control_end), "control synchronize") ||
        !hip_ok(hipEventElapsedTime(&control_ms, control_start, control_end),
                "control timing")) {
      (void)hipEventDestroy(control_start);
      (void)hipEventDestroy(control_end);
      (void)hipEventDestroy(candidate_start);
      (void)hipEventDestroy(candidate_end);
      return false;
    }
    (void)hipEventRecord(candidate_start, nullptr);
    if (!launch_candidate(grid, block, device_activation,
                          device_activation_scales, device_weight,
                          device_weight_scales, device_weight_tensor_scale,
                          device_input_tensor_scale, device_candidate, shape) ||
        !hip_ok(hipEventRecord(candidate_end, nullptr),
                "record candidate end") ||
        !hip_ok(hipEventSynchronize(candidate_end), "candidate synchronize") ||
        !hip_ok(
            hipEventElapsedTime(&candidate_ms, candidate_start, candidate_end),
            "candidate timing")) {
      (void)hipEventDestroy(control_start);
      (void)hipEventDestroy(control_end);
      (void)hipEventDestroy(candidate_start);
      (void)hipEventDestroy(candidate_end);
      return false;
    }
    (void)hipEventDestroy(control_start);
    (void)hipEventDestroy(control_end);
    (void)hipEventDestroy(candidate_start);
    (void)hipEventDestroy(candidate_end);
  }

  if (!launch_candidate(grid, block, device_activation,
                        device_activation_scales, device_weight,
                        device_weight_scales, device_weight_tensor_scale,
                        device_input_tensor_scale, device_candidate, shape) ||
      !launch_candidate(grid, block, device_activation,
                        device_activation_scales, device_weight,
                        device_weight_scales, device_weight_tensor_scale,
                        device_input_tensor_scale, device_repeat, shape) ||
      !hip_ok(hipDeviceSynchronize(), "repeat synchronize")) {
    return false;
  }

  std::vector<uint16_t> host_control(static_cast<size_t>(shape.m) * shape.n);
  std::vector<uint16_t> host_candidate(host_control.size());
  std::vector<uint16_t> host_repeat(host_control.size());
  if (!hip_ok(hipMemcpy(host_control.data(), device_control.ptr, output_bytes,
                        hipMemcpyDeviceToHost),
              "copy control output") ||
      !hip_ok(hipMemcpy(host_candidate.data(), device_candidate.ptr,
                        output_bytes, hipMemcpyDeviceToHost),
              "copy candidate output") ||
      !hip_ok(hipMemcpy(host_repeat.data(), device_repeat.ptr, output_bytes,
                        hipMemcpyDeviceToHost),
              "copy repeat output")) {
    return false;
  }
  const Metrics metrics =
      inspect_outputs(shape, host_control, host_candidate, host_repeat,
                      activation, activation_scales, weight, weight_scales);
  std::cout << "case=" << shape.name << " seed=" << seed
            << " control_ms=" << std::fixed << std::setprecision(3)
            << control_ms << " candidate_ms=" << candidate_ms
            << " max_control_candidate_ulp="
            << metrics.max_control_candidate_ulp
            << " max_candidate_oracle_ulp=" << metrics.max_candidate_oracle_ulp
            << " max_candidate_oracle_abs=" << std::scientific
            << metrics.max_candidate_oracle_abs << " candidate_bitwise_repeat="
            << (metrics.candidate_repeat_equal ? "PASS" : "FAIL") << '\n';
  return metrics.candidate_repeat_equal;
}

bool exact_gfx1201(const char *const arch) {
  return arch != nullptr &&
         std::string_view(arch).compare(0U, 7U, "gfx1201") == 0U;
}

} // namespace

int main() {
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0),
              "get device properties")) {
    return EXIT_FAILURE;
  }
  if (!exact_gfx1201(properties.gcnArchName)) {
    std::cerr << "ID89 probe requires exact gfx1201; observed "
              << properties.gcnArchName << '\n';
    return EXIT_FAILURE;
  }
  bool passed = true;
  constexpr std::array<uint32_t, 2> seeds = {UINT32_C(0x243f6a88),
                                             UINT32_C(0x9e3779b9)};
  for (const Shape shape : kShapes) {
    for (const uint32_t seed : seeds) {
      passed = run_case(shape, seed) && passed;
    }
  }
  std::cout << "summary status=" << (passed ? "PASS" : "FAIL")
            << " target=" << properties.gcnArchName
            << " control=id64 candidate=id89 oracle=long-double"
            << " finite_positive_e4m3_scales=1 bitwise_repeat=1\n";
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
