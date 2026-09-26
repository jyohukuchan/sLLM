// Phase 87 WU-3P numeric probe for the Qwen3.8 MTP NVFP4 prefill path.
//
// This developer-only binary compares the unchanged row8_tiled256 kernel with
// the target-specific production candidate (ID62 DP4A on gfx1030, or ID64
// WMMA on gfx1201).  It deliberately calls the device entry points directly:
// selector behavior and full-model timing are owned by the caller of this
// probe.  The host oracle decodes the packed E2M1/E4M3 values independently
// in long double and applies the public BF16 RNE epilogue.
//
// Compile and link this file with the exact-target production lowp HIP object.
// It is intentionally outside the default CMake test graph.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "unknown"
#endif

extern "C" __global__ void
sllm_matmul_nvfp4_w4a4_block16_prefill_row8_tiled256_v1(
    const uint8_t *, const uint8_t *, const uint8_t *, const uint8_t *,
    const float *, const float *, uint16_t *, uint64_t, uint64_t, uint64_t);

extern "C" __global__ void sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_v1(
    const uint8_t *, const uint8_t *, const uint8_t *, const uint8_t *,
    const float *, const float *, uint16_t *, uint64_t, uint64_t, uint64_t);

extern "C" __global__ void sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_32x64_v1(
    const uint8_t *, const uint8_t *, const uint8_t *, const uint8_t *,
    const float *, const float *, uint16_t *, uint64_t, uint64_t, uint64_t);

extern "C" __global__ void sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_v1(
    const uint8_t *, const uint8_t *, const uint8_t *, const uint8_t *,
    const float *, const float *, uint16_t *, uint64_t, uint64_t, uint64_t);

namespace {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kWarmups = 1U;
constexpr uint32_t kMeasured = 3U;
constexpr float kWeightTensorScale = 0.75F;
constexpr float kInputTensorScale = 1.125F;
constexpr std::array<uint8_t, 12> kScaleAlphabet = {0x20U, 0x24U, 0x28U, 0x2cU,
                                                    0x30U, 0x34U, 0x38U, 0x3cU,
                                                    0x40U, 0x44U, 0x48U, 0x4cU};

struct Shape final {
  uint64_t m;
  uint64_t k;
  uint64_t n;
  const char *name;
};

// These are the three Qwen3.8 MTP prefix projections, with both sides of the
// 1024-row boundary.  The smaller cases exercise non-aligned M/N and the
// neighboring tile boundaries without claiming to model a full projection.
constexpr std::array<Shape, 10> kShapes = {{
    {1023U, 10240U, 5120U, "mtp-m1023-k10240-n5120"},
    {1024U, 10240U, 5120U, "mtp-m1024-k10240-n5120"},
    {1023U, 5120U, 12288U, "mtp-m1023-k5120-n12288"},
    {1024U, 5120U, 12288U, "mtp-m1024-k5120-n12288"},
    {1023U, 5120U, 1024U, "mtp-m1023-k5120-n1024"},
    {1024U, 5120U, 1024U, "mtp-m1024-k5120-n1024"},
    {3U, 32U, 17U, "tiny-m3-k32-n17"},
    {63U, 48U, 65U, "boundary-m63-k48-n65"},
    {64U, 64U, 64U, "boundary-m64-k64-n64"},
    {65U, 80U, 63U, "boundary-m65-k80-n63"},
}};

// Representative wide projection rows around the 64/128/512 tile
// boundaries.  The exact MTP rows above cover all three projection pairs;
// this set checks arbitrary-prefix tails on the two larger representative
// projection shapes without multiplying every small boundary by every pair.
constexpr std::array<Shape, 21> kRepresentativeRows = {{
    {63U, 10240U, 5120U, "rep-m63-k10240-n5120"},
    {64U, 10240U, 5120U, "rep-m64-k10240-n5120"},
    {65U, 10240U, 5120U, "rep-m65-k10240-n5120"},
    {127U, 10240U, 5120U, "rep-m127-k10240-n5120"},
    {128U, 10240U, 5120U, "rep-m128-k10240-n5120"},
    {129U, 10240U, 5120U, "rep-m129-k10240-n5120"},
    {511U, 10240U, 5120U, "rep-m511-k10240-n5120"},
    {512U, 10240U, 5120U, "rep-m512-k10240-n5120"},
    {513U, 10240U, 5120U, "rep-m513-k10240-n5120"},
    {31U, 5120U, 12288U, "rep-m31-k5120-n12288"},
    {32U, 5120U, 12288U, "rep-m32-k5120-n12288"},
    {33U, 5120U, 12288U, "rep-m33-k5120-n12288"},
    {63U, 5120U, 12288U, "rep-m63-k5120-n12288"},
    {64U, 5120U, 12288U, "rep-m64-k5120-n12288"},
    {65U, 5120U, 12288U, "rep-m65-k5120-n12288"},
    {127U, 5120U, 12288U, "rep-m127-k5120-n12288"},
    {128U, 5120U, 12288U, "rep-m128-k5120-n12288"},
    {129U, 5120U, 12288U, "rep-m129-k5120-n12288"},
    {511U, 5120U, 12288U, "rep-m511-k5120-n12288"},
    {512U, 5120U, 12288U, "rep-m512-k5120-n12288"},
    {513U, 5120U, 12288U, "rep-m513-k5120-n12288"},
}};

std::vector<Shape> all_shapes() {
  std::vector<Shape> result;
  result.reserve(kShapes.size() + kRepresentativeRows.size());
  result.insert(result.end(), kShapes.begin(), kShapes.end());
  result.insert(result.end(), kRepresentativeRows.begin(),
                kRepresentativeRows.end());
  return result;
}

bool hip_ok(const hipError_t status, const char *const operation);

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

  bool allocate(const std::size_t bytes) {
    return hip_ok(hipMalloc(&ptr, bytes), "hipMalloc");
  }

  template <typename T> T *as() const { return static_cast<T *>(ptr); }
};

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::cerr << operation << ": " << hipGetErrorName(status) << " ("
            << hipGetErrorString(status) << ")\n";
  return false;
}

uint32_t mix32(uint32_t value) {
  value ^= value >> 16U;
  value *= UINT32_C(0x7feb352d);
  value ^= value >> 15U;
  value *= UINT32_C(0x846ca68b);
  return value ^ (value >> 16U);
}

uint16_t f32_to_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

bool finite_bf16(const uint16_t value) {
  return (value & UINT16_C(0x7f80)) != UINT16_C(0x7f80);
}

uint32_t bf16_order(const uint16_t value) {
  return (value & UINT16_C(0x8000)) != 0U
             ? static_cast<uint32_t>(~value & UINT16_C(0xffff))
             : static_cast<uint32_t>(value | UINT16_C(0x8000));
}

uint32_t bf16_ulp(const uint16_t lhs, const uint16_t rhs) {
  const uint32_t left = bf16_order(lhs);
  const uint32_t right = bf16_order(rhs);
  return left > right ? left - right : right - left;
}

uint8_t e2m1_code(const uint8_t packed, const uint64_t index) {
  return static_cast<uint8_t>((packed >> ((index & 1U) * 4U)) & 0x0fU);
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
  if (exponent == 0U) {
    const long double result = static_cast<long double>(mantissa) * 0x1p-9L;
    return negative ? -result : result;
  }
  if (magnitude == 0x7fU) {
    return std::numeric_limits<long double>::quiet_NaN();
  }
  const long double result = static_cast<long double>(8U + mantissa) *
                             std::ldexp(1.0L, static_cast<int>(exponent) - 10);
  return negative ? -result : result;
}

struct HostInputs final {
  std::vector<uint8_t> activation;
  std::vector<uint8_t> activation_scales;
  std::vector<uint8_t> weight;
  std::vector<uint8_t> weight_scales;
};

HostInputs make_inputs(const Shape shape, const uint32_t seed) {
  HostInputs inputs;
  inputs.activation.resize(static_cast<std::size_t>(shape.m * (shape.k / 2U)));
  inputs.activation_scales.resize(
      static_cast<std::size_t>(shape.m * (shape.k / 16U)));
  inputs.weight.resize(static_cast<std::size_t>(shape.n * (shape.k / 2U)));
  inputs.weight_scales.resize(
      static_cast<std::size_t>(shape.n * (shape.k / 16U)));
  for (std::size_t index = 0U; index < inputs.activation.size(); ++index) {
    inputs.activation[index] = static_cast<uint8_t>(
        mix32(seed + static_cast<uint32_t>(index * UINT32_C(17))));
  }
  for (std::size_t index = 0U; index < inputs.weight.size(); ++index) {
    inputs.weight[index] = static_cast<uint8_t>(mix32(
        seed + UINT32_C(0x9e3779b9) + static_cast<uint32_t>(index * 29U)));
  }
  for (std::size_t index = 0U; index < inputs.activation_scales.size();
       ++index) {
    inputs.activation_scales[index] =
        kScaleAlphabet[mix32(seed + static_cast<uint32_t>(index)) %
                       kScaleAlphabet.size()];
  }
  for (std::size_t index = 0U; index < inputs.weight_scales.size(); ++index) {
    inputs.weight_scales[index] =
        kScaleAlphabet[mix32(seed + UINT32_C(0x243f6a88) +
                             static_cast<uint32_t>(index)) %
                       kScaleAlphabet.size()];
  }
  return inputs;
}

bool copy_to_device(const std::vector<uint8_t> &host, DeviceBuffer &device) {
  return hip_ok(
      hipMemcpy(device.ptr, host.data(), host.size(), hipMemcpyHostToDevice),
      "copy input");
}

struct Buffers final {
  DeviceBuffer activation;
  DeviceBuffer activation_scales;
  DeviceBuffer weight;
  DeviceBuffer weight_scales;
  DeviceBuffer weight_tensor_scale;
  DeviceBuffer input_tensor_scale;
  DeviceBuffer baseline;
  DeviceBuffer candidate;
  DeviceBuffer repeat;
};

bool allocate_buffers(const Shape shape, const HostInputs &inputs,
                      Buffers &buffers) {
  const std::size_t output_bytes =
      static_cast<std::size_t>(shape.m * shape.n * sizeof(uint16_t));
  return buffers.activation.allocate(inputs.activation.size()) &&
         buffers.activation_scales.allocate(inputs.activation_scales.size()) &&
         buffers.weight.allocate(inputs.weight.size()) &&
         buffers.weight_scales.allocate(inputs.weight_scales.size()) &&
         buffers.weight_tensor_scale.allocate(sizeof(float)) &&
         buffers.input_tensor_scale.allocate(sizeof(float)) &&
         buffers.baseline.allocate(output_bytes) &&
         buffers.candidate.allocate(output_bytes) &&
         buffers.repeat.allocate(output_bytes) &&
         copy_to_device(inputs.activation, buffers.activation) &&
         copy_to_device(inputs.activation_scales, buffers.activation_scales) &&
         copy_to_device(inputs.weight, buffers.weight) &&
         copy_to_device(inputs.weight_scales, buffers.weight_scales) &&
         hip_ok(hipMemcpy(buffers.weight_tensor_scale.ptr, &kWeightTensorScale,
                          sizeof(float), hipMemcpyHostToDevice),
                "copy weight tensor scale") &&
         hip_ok(hipMemcpy(buffers.input_tensor_scale.ptr, &kInputTensorScale,
                          sizeof(float), hipMemcpyHostToDevice),
                "copy input tensor scale");
}

bool launch_baseline(const Shape shape, const Buffers &buffers) {
  hipLaunchKernelGGL(
      sllm_matmul_nvfp4_w4a4_block16_prefill_row8_tiled256_v1,
      dim3(static_cast<uint32_t>(((shape.m + 7U) / 8U) * shape.n)),
      dim3(kThreads), 0U, 0, buffers.activation.as<uint8_t>(),
      buffers.activation_scales.as<uint8_t>(), buffers.weight.as<uint8_t>(),
      buffers.weight_scales.as<uint8_t>(),
      buffers.weight_tensor_scale.as<float>(),
      buffers.input_tensor_scale.as<float>(), buffers.baseline.as<uint16_t>(),
      shape.m, shape.k, shape.n);
  return hip_ok(hipGetLastError(), "baseline launch");
}

bool launch_candidate(const Shape shape, const Buffers &buffers) {
  if (std::string_view(SLLM_TEST_EXPECTED_TARGET) == "gfx1030") {
    if (shape.m <= 32U) {
      hipLaunchKernelGGL(
          sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_32x64_v1,
          dim3(static_cast<uint32_t>((shape.n + 63U) / 64U)), dim3(kThreads),
          0U, 0, buffers.activation.as<uint8_t>(),
          buffers.activation_scales.as<uint8_t>(), buffers.weight.as<uint8_t>(),
          buffers.weight_scales.as<uint8_t>(),
          buffers.weight_tensor_scale.as<float>(),
          buffers.input_tensor_scale.as<float>(),
          buffers.candidate.as<uint16_t>(), shape.m, shape.k, shape.n);
      return hip_ok(hipGetLastError(), "ID62 M<=32 candidate launch");
    }
    hipLaunchKernelGGL(
        sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_v1,
        dim3(static_cast<uint32_t>(((shape.m + 63U) / 64U) *
                                   ((shape.n + 63U) / 64U))),
        dim3(kThreads), 0U, 0, buffers.activation.as<uint8_t>(),
        buffers.activation_scales.as<uint8_t>(), buffers.weight.as<uint8_t>(),
        buffers.weight_scales.as<uint8_t>(),
        buffers.weight_tensor_scale.as<float>(),
        buffers.input_tensor_scale.as<float>(),
        buffers.candidate.as<uint16_t>(), shape.m, shape.k, shape.n);
    return hip_ok(hipGetLastError(), "ID62 candidate launch");
  }
  if (std::string_view(SLLM_TEST_EXPECTED_TARGET) == "gfx1201") {
    hipLaunchKernelGGL(
        sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_v1,
        dim3(static_cast<uint32_t>((shape.n + 63U) / 64U),
             static_cast<uint32_t>((shape.m + 127U) / 128U)),
        dim3(kThreads), 0U, 0, buffers.activation.as<uint8_t>(),
        buffers.activation_scales.as<uint8_t>(), buffers.weight.as<uint8_t>(),
        buffers.weight_scales.as<uint8_t>(),
        buffers.weight_tensor_scale.as<float>(),
        buffers.input_tensor_scale.as<float>(),
        buffers.candidate.as<uint16_t>(), shape.m, shape.k, shape.n);
    return hip_ok(hipGetLastError(), "ID64 candidate launch");
  }
  std::cerr << "unsupported target\n";
  return false;
}

bool launch_repeat(const Shape shape, const Buffers &buffers) {
  if (std::string_view(SLLM_TEST_EXPECTED_TARGET) == "gfx1030") {
    if (shape.m <= 32U) {
      hipLaunchKernelGGL(
          sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_32x64_v1,
          dim3(static_cast<uint32_t>((shape.n + 63U) / 64U)), dim3(kThreads),
          0U, 0, buffers.activation.as<uint8_t>(),
          buffers.activation_scales.as<uint8_t>(), buffers.weight.as<uint8_t>(),
          buffers.weight_scales.as<uint8_t>(),
          buffers.weight_tensor_scale.as<float>(),
          buffers.input_tensor_scale.as<float>(), buffers.repeat.as<uint16_t>(),
          shape.m, shape.k, shape.n);
      return hip_ok(hipGetLastError(), "ID62 M<=32 repeat launch");
    }
    hipLaunchKernelGGL(
        sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_v1,
        dim3(static_cast<uint32_t>(((shape.m + 63U) / 64U) *
                                   ((shape.n + 63U) / 64U))),
        dim3(kThreads), 0U, 0, buffers.activation.as<uint8_t>(),
        buffers.activation_scales.as<uint8_t>(), buffers.weight.as<uint8_t>(),
        buffers.weight_scales.as<uint8_t>(),
        buffers.weight_tensor_scale.as<float>(),
        buffers.input_tensor_scale.as<float>(), buffers.repeat.as<uint16_t>(),
        shape.m, shape.k, shape.n);
  } else if (std::string_view(SLLM_TEST_EXPECTED_TARGET) == "gfx1201") {
    hipLaunchKernelGGL(
        sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_v1,
        dim3(static_cast<uint32_t>((shape.n + 63U) / 64U),
             static_cast<uint32_t>((shape.m + 127U) / 128U)),
        dim3(kThreads), 0U, 0, buffers.activation.as<uint8_t>(),
        buffers.activation_scales.as<uint8_t>(), buffers.weight.as<uint8_t>(),
        buffers.weight_scales.as<uint8_t>(),
        buffers.weight_tensor_scale.as<float>(),
        buffers.input_tensor_scale.as<float>(), buffers.repeat.as<uint16_t>(),
        shape.m, shape.k, shape.n);
  } else {
    return false;
  }
  return hip_ok(hipGetLastError(), "candidate repeat launch");
}

long double oracle_value(const Shape shape, const HostInputs &inputs,
                         const uint64_t row, const uint64_t column) {
  const uint64_t packed_row_bytes = shape.k / 2U;
  const uint64_t blocks_per_row = shape.k / 16U;
  long double accumulator = 0.0L;
  for (uint64_t inner = 0U; inner < shape.k; ++inner) {
    const uint8_t activation = inputs.activation[static_cast<std::size_t>(
        row * packed_row_bytes + inner / 2U)];
    const uint8_t weight = inputs.weight[static_cast<std::size_t>(
        column * packed_row_bytes + inner / 2U)];
    const uint8_t activation_scale =
        inputs.activation_scales[static_cast<std::size_t>(row * blocks_per_row +
                                                          inner / 16U)];
    const uint8_t weight_scale = inputs.weight_scales[static_cast<std::size_t>(
        column * blocks_per_row + inner / 16U)];
    accumulator += e2m1_value(e2m1_code(activation, inner)) *
                   e2m1_value(e2m1_code(weight, inner)) *
                   e4m3fn_value(activation_scale) * e4m3fn_value(weight_scale);
  }
  return accumulator * static_cast<long double>(kWeightTensorScale) *
         static_cast<long double>(kInputTensorScale);
}

struct Metrics final {
  uint64_t mismatch_count = 0U;
  uint32_t max_mismatch_ulp = 0U;
  uint64_t baseline_nonfinite = 0U;
  uint64_t candidate_nonfinite = 0U;
  uint64_t repeat_mismatch_count = 0U;
  uint64_t oracle_samples = 0U;
  uint64_t baseline_oracle_mismatch = 0U;
  uint64_t candidate_oracle_mismatch = 0U;
  uint32_t baseline_oracle_max_ulp = 0U;
  uint32_t candidate_oracle_max_ulp = 0U;
};

Metrics inspect(const Shape shape, const HostInputs &inputs,
                const std::vector<uint16_t> &baseline,
                const std::vector<uint16_t> &candidate,
                const std::vector<uint16_t> &repeat) {
  Metrics metrics{};
  for (std::size_t index = 0U; index < candidate.size(); ++index) {
    if (baseline[index] != candidate[index]) {
      ++metrics.mismatch_count;
      metrics.max_mismatch_ulp =
          std::max(metrics.max_mismatch_ulp,
                   bf16_ulp(baseline[index], candidate[index]));
    }
    metrics.baseline_nonfinite += finite_bf16(baseline[index]) ? 0U : 1U;
    metrics.candidate_nonfinite += finite_bf16(candidate[index]) ? 0U : 1U;
    metrics.repeat_mismatch_count +=
        candidate[index] == repeat[index] ? 0U : 1U;
  }

  // Cover both row/column edges and deterministic interior samples.  Full
  // output comparison above remains mandatory; the high-precision oracle is
  // sampled because an exact 12M-by-10K long-double product is not bounded.
  const std::array<std::pair<uint64_t, uint64_t>, 12> fixed = {
      {{0U, 0U},
       {0U, shape.n - 1U},
       {shape.m - 1U, 0U},
       {shape.m - 1U, shape.n - 1U},
       {shape.m / 2U, shape.n / 2U},
       {shape.m / 3U, shape.n / 3U},
       {shape.m / 7U, shape.n / 5U},
       {shape.m / 5U, shape.n / 7U},
       {shape.m / 11U, shape.n / 13U},
       {shape.m / 13U, shape.n / 11U},
       {shape.m / 17U, shape.n / 19U},
       {shape.m / 19U, shape.n / 17U}}};
  for (const auto [row, column] : fixed) {
    const std::size_t index = static_cast<std::size_t>(row * shape.n + column);
    const uint16_t expected = f32_to_bf16_rne(
        static_cast<float>(oracle_value(shape, inputs, row, column)));
    const uint32_t baseline_ulp = bf16_ulp(baseline[index], expected);
    const uint32_t candidate_ulp = bf16_ulp(candidate[index], expected);
    metrics.baseline_oracle_max_ulp =
        std::max(metrics.baseline_oracle_max_ulp, baseline_ulp);
    metrics.candidate_oracle_max_ulp =
        std::max(metrics.candidate_oracle_max_ulp, candidate_ulp);
    metrics.baseline_oracle_mismatch += baseline_ulp == 0U ? 0U : 1U;
    metrics.candidate_oracle_mismatch += candidate_ulp == 0U ? 0U : 1U;
    ++metrics.oracle_samples;
  }
  return metrics;
}

struct Timings final {
  float baseline_ms = 0.0F;
  float candidate_ms = 0.0F;
};

bool timed_launch(bool (*launch)(Shape, const Buffers &), const Shape shape,
                  const Buffers &buffers, float &milliseconds) {
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (!hip_ok(hipEventCreate(&start), "create start") ||
      !hip_ok(hipEventCreate(&stop), "create stop")) {
    if (start != nullptr)
      (void)hipEventDestroy(start);
    if (stop != nullptr)
      (void)hipEventDestroy(stop);
    return false;
  }
  bool ok =
      hip_ok(hipEventRecord(start, 0), "record start") &&
      launch(shape, buffers) &&
      hip_ok(hipEventRecord(stop, 0), "record stop") &&
      hip_ok(hipEventSynchronize(stop), "event synchronize") &&
      hip_ok(hipEventElapsedTime(&milliseconds, start, stop), "event elapsed");
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
  return ok;
}

void print_resources(const char *const name, const void *const function) {
  hipFuncAttributes attributes{};
  const hipError_t attribute_status =
      hipFuncGetAttributes(&attributes, function);
  int active_blocks = 0;
  const hipError_t occupancy_status =
      hipOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, function,
                                                   kThreads, 0U);
  std::printf("resources kernel=%s vgpr=%d lds_static=%zu scratch=%zu "
              "active_blocks=%d attr=%s occupancy=%s\n",
              name, attributes.numRegs, attributes.sharedSizeBytes,
              attributes.localSizeBytes, active_blocks,
              hipGetErrorString(attribute_status),
              hipGetErrorString(occupancy_status));
}

bool run_case(const Shape shape, const uint32_t seed) {
  const HostInputs inputs = make_inputs(shape, seed);
  Buffers buffers;
  if (!allocate_buffers(shape, inputs, buffers)) {
    return false;
  }

  for (uint32_t iteration = 0U; iteration < kWarmups; ++iteration) {
    if (!launch_baseline(shape, buffers) || !launch_candidate(shape, buffers) ||
        !hip_ok(hipDeviceSynchronize(), "warmup synchronize")) {
      return false;
    }
  }
  std::array<float, kMeasured> baseline_timings{};
  std::array<float, kMeasured> candidate_timings{};
  for (uint32_t iteration = 0U; iteration < kMeasured; ++iteration) {
    if (!timed_launch(launch_baseline, shape, buffers,
                      baseline_timings[iteration]) ||
        !timed_launch(launch_candidate, shape, buffers,
                      candidate_timings[iteration])) {
      return false;
    }
  }
  if (!launch_candidate(shape, buffers) || !launch_repeat(shape, buffers) ||
      !hip_ok(hipDeviceSynchronize(), "repeat synchronize")) {
    return false;
  }

  std::sort(baseline_timings.begin(), baseline_timings.end());
  std::sort(candidate_timings.begin(), candidate_timings.end());
  const std::size_t output_count = static_cast<std::size_t>(shape.m * shape.n);
  std::vector<uint16_t> baseline(output_count);
  std::vector<uint16_t> candidate(output_count);
  std::vector<uint16_t> repeat(output_count);
  const std::size_t output_bytes = output_count * sizeof(uint16_t);
  if (!hip_ok(hipMemcpy(baseline.data(), buffers.baseline.ptr, output_bytes,
                        hipMemcpyDeviceToHost),
              "copy baseline") ||
      !hip_ok(hipMemcpy(candidate.data(), buffers.candidate.ptr, output_bytes,
                        hipMemcpyDeviceToHost),
              "copy candidate") ||
      !hip_ok(hipMemcpy(repeat.data(), buffers.repeat.ptr, output_bytes,
                        hipMemcpyDeviceToHost),
              "copy repeat")) {
    return false;
  }
  const Metrics metrics = inspect(shape, inputs, baseline, candidate, repeat);
  const bool passed =
      metrics.baseline_nonfinite == 0U && metrics.candidate_nonfinite == 0U &&
      metrics.repeat_mismatch_count == 0U &&
      metrics.baseline_oracle_mismatch == 0U &&
      metrics.candidate_oracle_mismatch == 0U &&
      metrics.baseline_oracle_max_ulp == 0U &&
      metrics.candidate_oracle_max_ulp == 0U && metrics.max_mismatch_ulp <= 1U;
  std::printf(
      "case=%s seed=%u M=%llu K=%llu N=%llu baseline_ms=%.6f "
      "candidate_ms=%.6f speedup_pct=%.4f mismatch_count=%llu max_ulp=%u "
      "baseline_nonfinite=%llu candidate_nonfinite=%llu repeat_mismatch=%llu "
      "oracle_samples=%llu baseline_oracle_mismatch=%llu "
      "candidate_oracle_mismatch=%llu baseline_oracle_max_ulp=%u "
      "candidate_oracle_max_ulp=%u state=%s\n",
      shape.name, seed, static_cast<unsigned long long>(shape.m),
      static_cast<unsigned long long>(shape.k),
      static_cast<unsigned long long>(shape.n), baseline_timings[1],
      candidate_timings[1],
      (static_cast<double>(baseline_timings[1]) /
           static_cast<double>(candidate_timings[1]) -
       1.0) *
          100.0,
      static_cast<unsigned long long>(metrics.mismatch_count),
      metrics.max_mismatch_ulp,
      static_cast<unsigned long long>(metrics.baseline_nonfinite),
      static_cast<unsigned long long>(metrics.candidate_nonfinite),
      static_cast<unsigned long long>(metrics.repeat_mismatch_count),
      static_cast<unsigned long long>(metrics.oracle_samples),
      static_cast<unsigned long long>(metrics.baseline_oracle_mismatch),
      static_cast<unsigned long long>(metrics.candidate_oracle_mismatch),
      metrics.baseline_oracle_max_ulp, metrics.candidate_oracle_max_ulp,
      passed ? "PASS" : "FAIL");
  std::fflush(stdout);
  return passed;
}

} // namespace

int main() {
  hipDeviceProp_t properties{};
  if (!hip_ok(hipSetDevice(0), "set device") ||
      !hip_ok(hipGetDeviceProperties(&properties, 0), "device properties")) {
    return EXIT_FAILURE;
  }
  const std::string_view observed(properties.gcnArchName);
  const std::string_view expected(SLLM_TEST_EXPECTED_TARGET);
  if (expected == "unknown" || observed != expected) {
    std::cerr << "exact target mismatch observed=" << observed
              << " expected=" << expected << '\n';
    return EXIT_FAILURE;
  }
  const char *candidate_name = "unknown";
  if (expected == "gfx1030") {
    candidate_name = "id62-dp4a64x64";
    print_resources(
        "row8-tiled256-baseline",
        reinterpret_cast<const void *>(
            sllm_matmul_nvfp4_w4a4_block16_prefill_row8_tiled256_v1));
    print_resources("id62-dp4a64x64",
                    reinterpret_cast<const void *>(
                        sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_v1));
    print_resources("id62-dp4a32x64-M<=32",
                    reinterpret_cast<const void *>(
                        sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_32x64_v1));
  } else if (expected == "gfx1201") {
    candidate_name = "id64-wmma128x64";
    print_resources(
        "row8-tiled256-baseline",
        reinterpret_cast<const void *>(
            sllm_matmul_nvfp4_w4a4_block16_prefill_row8_tiled256_v1));
    print_resources("id64-wmma128x64",
                    reinterpret_cast<const void *>(
                        sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_v1));
  } else {
    std::cerr << "unsupported target\n";
    return EXIT_FAILURE;
  }
  std::printf("target=%s candidate=%s baseline=row8-tiled256 "
              "oracle=long-double-packed-e2m1-e4m3fn-bf16-rne "
              "full_output_compare=1 repeat=1\n",
              properties.gcnArchName, candidate_name);

  bool passed = true;
  constexpr std::array<uint32_t, 2> seeds = {UINT32_C(0x243f6a88),
                                             UINT32_C(0x9e3779b9)};
  const std::vector<Shape> shapes = all_shapes();
  for (const Shape shape : shapes) {
    for (const uint32_t seed : seeds) {
      passed = run_case(shape, seed) && passed;
    }
  }
  std::printf("summary target=%s candidate=%s cases=%zu seeds=2 "
              "full_output_compare=1 status=%s\n",
              properties.gcnArchName, candidate_name,
              shapes.size() * seeds.size(), passed ? "PASS" : "FAIL");
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
