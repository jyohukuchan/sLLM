// Phase 78 standalone gfx1030 FP8 outer-scale decode direct-wave probe.
//
// ID68 (dword8, four columns per wave) is the control.  The candidates keep
// the same E4M3FN -> FP16 ingress, FP32 dot2 accumulation, outer scales, and
// BF16 RNE epilogue, but never stage the activation row in LDS:
//   * direct-wave8-sequential: eight columns per wave, one column at a time;
//   * direct-wave8-pair2: eight columns per wave, load four-column pairs
//     before conversion/compute (higher VGPR pressure);
//   * direct-wave4-pair2: four columns per wave with the explicit pair2 load
//     schedule, as a low-register comparison against ID68.
//
// This is a probe only.  It deliberately does not include production headers
// or alter the production selector.  Four distinct resident weight matrices
// are allocated per shape and cycled during measurement so the reported
// value is not an identical-weight warm-cache number.

#include "low_precision_block_codec.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

namespace {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kWave = 32U;
constexpr uint32_t kWaves = kThreads / kWave;
constexpr uint32_t kColdCopies = 4U;
constexpr int kWarmups = 3;
constexpr int kMeasured = 10;

struct ShapeCase final {
  uint64_t k;
  uint64_t n;
  uint32_t occurrences;
  const char *roles;
};

constexpr std::array<ShapeCase, 8> kQwen38Shapes = {{
    {5120U, 17408U, 16U, "layers56-63.mlp.gate+up"},
    {17408U, 5120U, 8U, "layers56-63.mlp.down"},
    {5120U, 12288U, 16U, "16.full-attn.q"},
    {5120U, 1024U, 32U, "16.full-attn.k+v"},
    {6144U, 5120U, 64U, "full-attn.o+linear-attn.out"},
    {5120U, 10240U, 48U, "48.linear-attn.qkv"},
    {5120U, 6144U, 48U, "48.linear-attn.z"},
    {5120U, 248320U, 1U, "lm_head"},
}};

constexpr uint32_t shape_occurrences() {
  uint32_t result = 0U;
  for (const ShapeCase &shape : kQwen38Shapes)
    result += shape.occurrences;
  return result;
}
static_assert(shape_occurrences() == 233U);

// The finite input set covers zero, normal and subnormal E4M3FN values.  NaN
// and infinity are checked separately by the all-code ingress oracle.
constexpr std::array<uint8_t, 16> kFiniteCodes = {
    0x00U, 0x01U, 0x08U, 0x10U, 0x18U, 0x20U, 0x28U, 0x30U,
    0x80U, 0x81U, 0x88U, 0x90U, 0x98U, 0xa0U, 0xa8U, 0xb0U};

__device__ __constant__ uint8_t kDeviceFiniteCodes[16] = {
    0x00U, 0x01U, 0x08U, 0x10U, 0x18U, 0x20U, 0x28U, 0x30U,
    0x80U, 0x81U, 0x88U, 0x90U, 0x98U, 0xa0U, 0xa8U, 0xb0U};

__device__ __forceinline__ uint8_t device_weight_code(const uint32_t copy,
                                                      const uint64_t column,
                                                      const uint64_t inner) {
  return kDeviceFiniteCodes[(copy * 11U + column * 3U + inner * 7U + 9U) & 15U];
}

__device__ __forceinline__ uint16_t device_bf16_rne(const float value) {
  const uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7f800000U) == 0x7f800000U) {
    if ((bits & 0x007fffffU) != 0U)
      return static_cast<uint16_t>(((bits >> 16U) & 0x8000U) | 0x7fc0U |
                                   ((bits >> 16U) & 0x003fU));
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & 0xffffU;
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U))
    ++upper;
  return static_cast<uint16_t>(upper);
}

__device__ __forceinline__ __half2 half2_from_bits(const uint32_t bits) {
  return *reinterpret_cast<const __half2 *>(&bits);
}

template <uint32_t ColumnsPerWave, bool PairLoad>
__global__ __launch_bounds__(kThreads, 1) void direct_wave_kernel(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  static_assert(ColumnsPerWave == 4U || ColumnsPerWave == 8U);
  constexpr uint32_t values_per_iteration = 8U;
  constexpr uint32_t columns_per_workgroup = kWaves * ColumnsPerWave;
  if (m != 1U || k == 0U || n == 0U || (k % 64U) != 0U)
    return;
  const uint32_t lane = threadIdx.x & (kWave - 1U);
  const uint32_t wave = threadIdx.x / kWave;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * columns_per_workgroup +
      static_cast<uint64_t>(wave) * ColumnsPerWave;
  if (column_base >= n)
    return;

  float accumulators[ColumnsPerWave] = {};
  const uint64_t iteration_count = k / values_per_iteration;
  const auto *const activation_dwords =
      reinterpret_cast<const uint32_t *>(activation);
  for (uint64_t iteration = lane; iteration < iteration_count;
       iteration += kWave) {
    const uint32_t activation_first =
        __builtin_nontemporal_load(activation_dwords + iteration * 2U);
    const uint32_t activation_second =
        __builtin_nontemporal_load(activation_dwords + iteration * 2U + 1U);
    uint32_t weight_first[ColumnsPerWave];
    uint32_t weight_second[ColumnsPerWave];
    if constexpr (PairLoad) {
#pragma unroll
      for (uint32_t local_column = 0U; local_column < ColumnsPerWave;
           ++local_column) {
        const uint64_t column = column_base + local_column;
        const auto *const column_dwords =
            reinterpret_cast<const uint32_t *>(weight + column * k);
        if (column < n) {
          weight_first[local_column] =
              __builtin_nontemporal_load(column_dwords + iteration * 2U);
          weight_second[local_column] =
              __builtin_nontemporal_load(column_dwords + iteration * 2U + 1U);
        } else {
          weight_first[local_column] = 0U;
          weight_second[local_column] = 0U;
        }
      }
    }
    const sllm_lowp::E4M3FnFp16x4Bits activation_first_pairs =
        sllm_lowp::e4m3fnx4_to_fp16x2_bits(activation_first);
    const sllm_lowp::E4M3FnFp16x4Bits activation_second_pairs =
        sllm_lowp::e4m3fnx4_to_fp16x2_bits(activation_second);
    const __half2 activation_first_low =
        half2_from_bits(activation_first_pairs.low);
    const __half2 activation_first_high =
        half2_from_bits(activation_first_pairs.high);
    const __half2 activation_second_low =
        half2_from_bits(activation_second_pairs.low);
    const __half2 activation_second_high =
        half2_from_bits(activation_second_pairs.high);
#pragma unroll
    for (uint32_t local_column = 0U; local_column < ColumnsPerWave;
         ++local_column) {
      const uint64_t column = column_base + local_column;
      if (column >= n)
        continue;
      if constexpr (!PairLoad) {
        const auto *const column_dwords =
            reinterpret_cast<const uint32_t *>(weight + column * k);
        weight_first[local_column] =
            __builtin_nontemporal_load(column_dwords + iteration * 2U);
        weight_second[local_column] =
            __builtin_nontemporal_load(column_dwords + iteration * 2U + 1U);
      }
      const sllm_lowp::E4M3FnFp16x4Bits weight_first_pairs =
          sllm_lowp::e4m3fnx4_to_fp16x2_bits(weight_first[local_column]);
      const sllm_lowp::E4M3FnFp16x4Bits weight_second_pairs =
          sllm_lowp::e4m3fnx4_to_fp16x2_bits(weight_second[local_column]);
      accumulators[local_column] = amd_mixed_dot(
          activation_first_low, half2_from_bits(weight_first_pairs.low),
          accumulators[local_column], false);
      accumulators[local_column] = amd_mixed_dot(
          activation_first_high, half2_from_bits(weight_first_pairs.high),
          accumulators[local_column], false);
      accumulators[local_column] = amd_mixed_dot(
          activation_second_low, half2_from_bits(weight_second_pairs.low),
          accumulators[local_column], false);
      accumulators[local_column] = amd_mixed_dot(
          activation_second_high, half2_from_bits(weight_second_pairs.high),
          accumulators[local_column], false);
    }
  }
#pragma unroll
  for (uint32_t local_column = 0U; local_column < ColumnsPerWave;
       ++local_column) {
#pragma unroll
    for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U)
      accumulators[local_column] +=
          __shfl_down(accumulators[local_column], offset, kWave);
    const uint64_t column = column_base + local_column;
    if (lane == 0U && column < n)
      output[column] =
          device_bf16_rne(accumulators[local_column] * activation_scales[0] *
                          weight_scales[column]);
  }
}

enum class CandidateId : uint32_t {
  Control68,
  Direct8Sequential,
  Direct8Pair2,
  Direct4Pair2
};

struct Candidate final {
  CandidateId id;
  const char *name;
  const void *function;
  uint32_t columns_per_wave;
};

Candidate candidate(const CandidateId id) {
  switch (id) {
  case CandidateId::Control68:
    return {id, "id68-dword8-wave4col32-control",
            reinterpret_cast<const void *>(direct_wave_kernel<4U, true>), 4U};
  case CandidateId::Direct8Sequential:
    return {id, "direct-wave8-sequential-col64",
            reinterpret_cast<const void *>(direct_wave_kernel<8U, false>), 8U};
  case CandidateId::Direct8Pair2:
    return {id, "direct-wave8-pair2-col64",
            reinterpret_cast<const void *>(direct_wave_kernel<8U, true>), 8U};
  case CandidateId::Direct4Pair2:
    return {id, "direct-wave4-pair2-col32",
            reinterpret_cast<const void *>(direct_wave_kernel<4U, true>), 4U};
  }
  return {CandidateId::Control68, "invalid", nullptr, 0U};
}

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "hip error operation=%s status=%s (%s)\n", operation,
               hipGetErrorName(status), hipGetErrorString(status));
  return false;
}

bool exact_gfx1030(const char *const arch) {
  if (arch == nullptr)
    return false;
  const std::string_view value(arch);
  constexpr std::string_view prefix = "gfx1030";
  return value == prefix || (value.size() > prefix.size() &&
                             value.compare(0U, prefix.size(), prefix) == 0 &&
                             value[prefix.size()] == ':');
}

float host_e4m3(const uint8_t bits) {
  const uint8_t magnitude = bits & 0x7fU;
  const uint8_t exponent = magnitude >> 3U;
  const uint8_t mantissa = magnitude & 0x07U;
  float value = 0.0F;
  if (exponent == 0U) {
    value = static_cast<float>(mantissa) * 0x1p-9F;
  } else if (magnitude == 0x7fU) {
    value = std::numeric_limits<float>::quiet_NaN();
  } else {
    value = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                       static_cast<int>(exponent) - 7);
  }
  return (bits & 0x80U) != 0U ? -value : value;
}

uint16_t host_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000U) == 0x7f800000U) {
    if ((bits & 0x007fffffU) != 0U)
      return static_cast<uint16_t>(((bits >> 16U) & 0x8000U) | 0x7fc0U |
                                   ((bits >> 16U) & 0x003fU));
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & 0xffffU;
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U))
    ++upper;
  return static_cast<uint16_t>(upper);
}

uint8_t host_weight_code(const uint32_t copy, const uint64_t column,
                         const uint64_t inner) {
  return kFiniteCodes[(copy * 11U + column * 3U + inner * 7U + 9U) & 15U];
}

struct WeightPlane final {
  uint8_t *weight = nullptr;
  float *scales = nullptr;
};

struct Buffers final {
  uint8_t *activation = nullptr;
  float *activation_scale = nullptr;
  std::array<WeightPlane, kColdCopies> planes{};
  uint16_t *output = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
};

void cleanup(Buffers *const buffers) {
  if (buffers == nullptr)
    return;
  if (buffers->stop != nullptr)
    (void)hipEventDestroy(buffers->stop);
  if (buffers->start != nullptr)
    (void)hipEventDestroy(buffers->start);
  if (buffers->stream != nullptr)
    (void)hipStreamDestroy(buffers->stream);
  if (buffers->output != nullptr)
    (void)hipFree(buffers->output);
  for (WeightPlane &plane : buffers->planes) {
    if (plane.scales != nullptr)
      (void)hipFree(plane.scales);
    if (plane.weight != nullptr)
      (void)hipFree(plane.weight);
  }
  if (buffers->activation_scale != nullptr)
    (void)hipFree(buffers->activation_scale);
  if (buffers->activation != nullptr)
    (void)hipFree(buffers->activation);
  *buffers = {};
}

__global__ void fill_weight_kernel(uint8_t *const weight, const uint64_t k,
                                   const uint64_t n, const uint32_t copy) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = k * n;
  if (index >= total)
    return;
  const uint64_t column = index / k;
  const uint64_t inner = index % k;
  weight[index] = device_weight_code(copy, column, inner);
}

void fill_activation(const ShapeCase &shape, std::vector<uint8_t> *const a) {
  a->resize(static_cast<std::size_t>(shape.k));
  for (uint64_t inner = 0U; inner < shape.k; ++inner)
    (*a)[static_cast<std::size_t>(inner)] =
        kFiniteCodes[(inner * 5U + 3U) & 15U];
}

void fill_scales(const ShapeCase &shape, const uint32_t copy,
                 std::vector<float> *const scales) {
  scales->resize(static_cast<std::size_t>(shape.n));
  for (uint64_t column = 0U; column < shape.n; ++column)
    (*scales)[static_cast<std::size_t>(column)] =
        0.625F + static_cast<float>(column % 11U) * 0.03125F +
        static_cast<float>(copy) * 0.0078125F;
}

bool make_buffers(const ShapeCase &shape, Buffers *const buffers) {
  if (buffers == nullptr || shape.k == 0U || shape.n == 0U ||
      (shape.k % 64U) != 0U || shape.k > SIZE_MAX / shape.n) {
    return false;
  }
  const std::size_t weight_bytes = static_cast<std::size_t>(shape.k * shape.n);
  const std::size_t scale_bytes =
      static_cast<std::size_t>(shape.n) * sizeof(float);
  const std::size_t output_bytes =
      static_cast<std::size_t>(shape.n) * sizeof(uint16_t);
  if (!hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->activation),
                        static_cast<std::size_t>(shape.k)),
              "malloc activation") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->activation_scale),
                        sizeof(float)),
              "malloc activation scale")) {
    cleanup(buffers);
    return false;
  }
  for (WeightPlane &plane : buffers->planes) {
    if (!hip_ok(
            hipMalloc(reinterpret_cast<void **>(&plane.weight), weight_bytes),
            "malloc cold weight") ||
        !hip_ok(
            hipMalloc(reinterpret_cast<void **>(&plane.scales), scale_bytes),
            "malloc cold scales")) {
      cleanup(buffers);
      return false;
    }
  }
  if (!hip_ok(
          hipMalloc(reinterpret_cast<void **>(&buffers->output), output_bytes),
          "malloc output") ||
      !hip_ok(hipStreamCreate(&buffers->stream), "create stream") ||
      !hip_ok(hipEventCreate(&buffers->start), "create start event") ||
      !hip_ok(hipEventCreate(&buffers->stop), "create stop event")) {
    cleanup(buffers);
    return false;
  }
  return true;
}

bool upload_inputs(const ShapeCase &shape,
                   const std::vector<uint8_t> &activation,
                   Buffers *const buffers) {
  const float activation_scale = 0.875F;
  if (!hip_ok(hipMemcpy(buffers->activation, activation.data(),
                        activation.size(), hipMemcpyHostToDevice),
              "copy activation") ||
      !hip_ok(hipMemcpy(buffers->activation_scale, &activation_scale,
                        sizeof(activation_scale), hipMemcpyHostToDevice),
              "copy activation scale"))
    return false;
  const uint64_t total = shape.k * shape.n;
  const uint32_t blocks =
      static_cast<uint32_t>((total + kThreads - 1U) / kThreads);
  for (uint32_t copy = 0U; copy < kColdCopies; ++copy) {
    hipLaunchKernelGGL(fill_weight_kernel, dim3(blocks), dim3(kThreads), 0U,
                       buffers->stream, buffers->planes[copy].weight, shape.k,
                       shape.n, copy);
    if (!hip_ok(hipGetLastError(), "fill cold weight"))
      return false;
    std::vector<float> scales;
    fill_scales(shape, copy, &scales);
    if (!hip_ok(hipMemcpy(buffers->planes[copy].scales, scales.data(),
                          scales.size() * sizeof(float), hipMemcpyHostToDevice),
                "copy cold scales"))
      return false;
  }
  return hip_ok(hipStreamSynchronize(buffers->stream), "sync input fill");
}

std::size_t shared_bytes(const Candidate &) { return 0U; }

bool launch(const Candidate &current, const ShapeCase &shape,
            const uint32_t copy, Buffers *const buffers) {
  const uint64_t columns_per_workgroup =
      static_cast<uint64_t>(kWaves) * current.columns_per_wave;
  const uint64_t blocks =
      (shape.n + columns_per_workgroup - 1U) / columns_per_workgroup;
  if (blocks == 0U || blocks > UINT32_MAX || copy >= kColdCopies)
    return false;
  const dim3 grid(static_cast<uint32_t>(blocks));
  const dim3 block(kThreads);
  const WeightPlane &plane = buffers->planes[copy];
  switch (current.id) {
  case CandidateId::Control68:
    hipLaunchKernelGGL((direct_wave_kernel<4U, true>), grid, block, 0U,
                       buffers->stream, buffers->activation,
                       buffers->activation_scale, plane.weight, plane.scales,
                       buffers->output, 1U, shape.k, shape.n);
    break;
  case CandidateId::Direct8Sequential:
    hipLaunchKernelGGL((direct_wave_kernel<8U, false>), grid, block, 0U,
                       buffers->stream, buffers->activation,
                       buffers->activation_scale, plane.weight, plane.scales,
                       buffers->output, 1U, shape.k, shape.n);
    break;
  case CandidateId::Direct8Pair2:
    hipLaunchKernelGGL((direct_wave_kernel<8U, true>), grid, block, 0U,
                       buffers->stream, buffers->activation,
                       buffers->activation_scale, plane.weight, plane.scales,
                       buffers->output, 1U, shape.k, shape.n);
    break;
  case CandidateId::Direct4Pair2:
    hipLaunchKernelGGL((direct_wave_kernel<4U, true>), grid, block, 0U,
                       buffers->stream, buffers->activation,
                       buffers->activation_scale, plane.weight, plane.scales,
                       buffers->output, 1U, shape.k, shape.n);
    break;
  }
  return hipGetLastError() == hipSuccess;
}

std::vector<uint16_t> cpu_oracle(const ShapeCase &shape,
                                 const std::vector<uint8_t> &activation,
                                 const uint32_t copy) {
  std::vector<uint16_t> expected(static_cast<std::size_t>(shape.n));
  for (uint64_t column = 0U; column < shape.n; ++column) {
    float accumulator = 0.0F;
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      accumulator = std::fmaf(
          host_e4m3(activation[static_cast<std::size_t>(inner)]),
          host_e4m3(host_weight_code(copy, column, inner)), accumulator);
    }
    const float scale = 0.625F + static_cast<float>(column % 11U) * 0.03125F +
                        static_cast<float>(copy) * 0.0078125F;
    expected[static_cast<std::size_t>(column)] =
        host_bf16_rne(accumulator * 0.875F * scale);
  }
  return expected;
}

uint32_t bf16_ulp(const uint16_t left, const uint16_t right) {
  const uint32_t a = left;
  const uint32_t b = right;
  return a > b ? a - b : b - a;
}

bool compare_outputs(const char *const name, const ShapeCase &shape,
                     const std::vector<uint16_t> &actual,
                     const std::vector<uint16_t> &expected,
                     const char *const relation) {
  std::size_t mismatches = 0U;
  uint32_t max_ulp = 0U;
  std::size_t printed = 0U;
  for (std::size_t index = 0U; index < actual.size(); ++index) {
    if (actual[index] != expected[index]) {
      ++mismatches;
      if (printed < 3U) {
        std::printf("oracle_mismatch candidate=%s relation=%s index=%zu "
                    "actual=0x%04x expected=0x%04x\n",
                    name, relation, index, static_cast<unsigned>(actual[index]),
                    static_cast<unsigned>(expected[index]));
        ++printed;
      }
    }
    max_ulp = std::max(max_ulp, bf16_ulp(actual[index], expected[index]));
  }
  std::printf("oracle candidate=%s relation=%s K=%llu N=%llu mismatches=%zu "
              "max_bf16_ulp=%u status=%s\n",
              name, relation, static_cast<unsigned long long>(shape.k),
              static_cast<unsigned long long>(shape.n), mismatches, max_ulp,
              mismatches == 0U ? "PASS" : "FAIL");
  return mismatches == 0U;
}

bool finite_output(const std::vector<uint16_t> &output) {
  for (const uint16_t bits : output) {
    if ((bits & 0x7f80U) == 0x7f80U)
      return false;
  }
  return true;
}

bool measure_all(const std::vector<CandidateId> &ids, const ShapeCase &shape,
                 Buffers *const buffers, std::vector<double> *const medians,
                 std::vector<double> *const mads) {
  if (ids.empty() || medians == nullptr || mads == nullptr)
    return false;
  std::vector<std::array<double, kMeasured>> samples(ids.size());
  // Warm up every candidate on each cold plane.  The measured order below is
  // rotated, so a candidate cannot win merely by following the control after
  // the same cache line was used by the preceding launch.
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    for (std::size_t position = 0U; position < ids.size(); ++position) {
      const std::size_t index =
          (position + static_cast<std::size_t>(warmup)) % ids.size();
      const Candidate current = candidate(ids[index]);
      if (!launch(current, shape, static_cast<uint32_t>(warmup) % kColdCopies,
                  buffers) ||
          !hip_ok(hipStreamSynchronize(buffers->stream), "warmup sync"))
        return false;
    }
  }
  for (int iteration = 0; iteration < kMeasured; ++iteration) {
    const uint32_t copy = static_cast<uint32_t>(iteration) % kColdCopies;
    for (std::size_t position = 0U; position < ids.size(); ++position) {
      const std::size_t index =
          (position + static_cast<std::size_t>(iteration)) % ids.size();
      const Candidate current = candidate(ids[index]);
      if (!hip_ok(hipEventRecord(buffers->start, buffers->stream),
                  "event start") ||
          !launch(current, shape, copy, buffers) ||
          !hip_ok(hipEventRecord(buffers->stop, buffers->stream),
                  "event stop") ||
          !hip_ok(hipEventSynchronize(buffers->stop), "event sync"))
        return false;
      float elapsed_ms = 0.0F;
      if (!hip_ok(
              hipEventElapsedTime(&elapsed_ms, buffers->start, buffers->stop),
              "event elapsed"))
        return false;
      samples[index][static_cast<std::size_t>(iteration)] = elapsed_ms;
    }
  }
  medians->resize(ids.size());
  mads->resize(ids.size());
  for (std::size_t index = 0U; index < ids.size(); ++index) {
    auto &values = samples[index];
    std::sort(values.begin(), values.end());
    (*medians)[index] = values[kMeasured / 2U];
    std::array<double, kMeasured> deviations{};
    for (int i = 0; i < kMeasured; ++i)
      deviations[static_cast<std::size_t>(i)] =
          std::fabs(values[static_cast<std::size_t>(i)] - (*medians)[index]);
    std::sort(deviations.begin(), deviations.end());
    (*mads)[index] = deviations[kMeasured / 2U];
  }
  return true;
}

bool copy_output(const ShapeCase &shape, Buffers *const buffers,
                 std::vector<uint16_t> *const output) {
  output->resize(static_cast<std::size_t>(shape.n));
  return hip_ok(hipMemcpy(output->data(), buffers->output,
                          output->size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "copy output");
}

__global__ void e4m3_decode_oracle_kernel(const uint8_t *const input,
                                          uint16_t *const output);

bool all_codes_oracle() {
  std::array<uint8_t, 256> input{};
  std::array<uint16_t, 256> actual{};
  for (uint32_t i = 0U; i < 256U; ++i)
    input[i] = static_cast<uint8_t>(i);
  uint8_t *device_input = nullptr;
  uint16_t *device_output = nullptr;
  bool ok =
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_input), 256U),
             "malloc all-code input") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_output),
                       actual.size() * sizeof(uint16_t)),
             "malloc all-code output") &&
      hip_ok(hipMemcpy(device_input, input.data(), 256U, hipMemcpyHostToDevice),
             "copy all-code input");
  if (ok) {
    hipLaunchKernelGGL(e4m3_decode_oracle_kernel, dim3(1U), dim3(256U), 0U,
                       nullptr, device_input, device_output);
    ok = hip_ok(hipGetLastError(), "launch all-code oracle") &&
         hip_ok(hipDeviceSynchronize(), "sync all-code oracle") &&
         hip_ok(hipMemcpy(actual.data(), device_output,
                          actual.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "copy all-code output");
  }
  std::size_t mismatches = 0U;
  if (ok) {
    for (uint32_t i = 0U; i < 256U; ++i)
      if (actual[i] != sllm_lowp::e4m3fn_to_fp16_bits(input[i]))
        ++mismatches;
  }
  if (device_output != nullptr)
    (void)hipFree(device_output);
  if (device_input != nullptr)
    (void)hipFree(device_input);
  std::printf("oracle all_e4m3_codes=256 mismatches=%zu status=%s\n",
              mismatches, ok && mismatches == 0U ? "PASS" : "FAIL");
  return ok && mismatches == 0U;
}

__global__ void e4m3_decode_oracle_kernel(const uint8_t *const input,
                                          uint16_t *const output) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 256U)
    output[index] = sllm_lowp::e4m3fn_to_fp16_bits(input[index]);
}

void print_resources(const Candidate &current, const uint64_t k) {
  hipFuncAttributes attributes{};
  const hipError_t attr_status =
      hipFuncGetAttributes(&attributes, current.function);
  int active_blocks = 0;
  const hipError_t occupancy_status =
      hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks, current.function, kThreads, shared_bytes(current));
  std::printf("resources candidate=%s columns_per_wave=%u vgpr=%d sgpr=%d "
              "static_lds=%zu dynamic_lds=%zu scratch=%zu active_blocks=%d "
              "K=%llu attr=%s occupancy=%s\n",
              current.name, current.columns_per_wave, attributes.numRegs, -1,
              attributes.sharedSizeBytes, shared_bytes(current),
              attributes.localSizeBytes, active_blocks,
              static_cast<unsigned long long>(k),
              hipGetErrorString(attr_status),
              hipGetErrorString(occupancy_status));
}

bool run_shape(const ShapeCase &shape, const std::vector<CandidateId> &ids,
               double *const weighted_control, double *const weighted_best) {
  std::vector<uint8_t> activation;
  fill_activation(shape, &activation);
  Buffers buffers;
  if (!make_buffers(shape, &buffers) ||
      !upload_inputs(shape, activation, &buffers)) {
    cleanup(&buffers);
    return false;
  }
  std::printf("cold_buffers K=%llu N=%llu copies=%u total_weight_bytes=%llu\n",
              static_cast<unsigned long long>(shape.k),
              static_cast<unsigned long long>(shape.n), kColdCopies,
              static_cast<unsigned long long>(shape.k * shape.n * kColdCopies));

  std::vector<std::vector<uint16_t>> controls(kColdCopies);
  std::vector<double> medians;
  std::vector<double> mads;
  if (!measure_all(ids, shape, &buffers, &medians, &mads)) {
    cleanup(&buffers);
    return false;
  }
  bool all_ok = true;
  for (std::size_t candidate_index = 0U; candidate_index < ids.size();
       ++candidate_index) {
    const CandidateId id = ids[candidate_index];
    const Candidate current = candidate(id);
    print_resources(current, shape.k);
    const double median_ms = medians[candidate_index];
    const double mad_ms = mads[candidate_index];
    for (uint32_t copy = 0U; copy < kColdCopies; ++copy) {
      if (!launch(current, shape, copy, &buffers) ||
          !hip_ok(hipStreamSynchronize(buffers.stream), "oracle sync")) {
        cleanup(&buffers);
        return false;
      }
      std::vector<uint16_t> actual;
      if (!copy_output(shape, &buffers, &actual)) {
        cleanup(&buffers);
        return false;
      }
      if (!finite_output(actual)) {
        std::printf("finite candidate=%s copy=%u status=FAIL\n", current.name,
                    copy);
        all_ok = false;
      }
      if (id == CandidateId::Control68) {
        controls[copy] = actual;
      } else {
        all_ok = compare_outputs(current.name, shape, actual, controls[copy],
                                 "vs-id68") &&
                 all_ok;
      }
      if (shape.n <= 128U) {
        const std::vector<uint16_t> expected =
            cpu_oracle(shape, activation, copy);
        all_ok =
            compare_outputs(current.name, shape, actual, expected, "host") &&
            all_ok;
      }
    }
    if (!launch(current, shape, 0U, &buffers) ||
        !hip_ok(hipStreamSynchronize(buffers.stream), "determinism sync")) {
      cleanup(&buffers);
      return false;
    }
    std::vector<uint16_t> first;
    if (!copy_output(shape, &buffers, &first) ||
        !launch(current, shape, 0U, &buffers) ||
        !hip_ok(hipStreamSynchronize(buffers.stream),
                "determinism repeat sync")) {
      cleanup(&buffers);
      return false;
    }
    std::vector<uint16_t> second;
    if (!copy_output(shape, &buffers, &second)) {
      cleanup(&buffers);
      return false;
    }
    const bool deterministic = first == second;
    std::printf("determinism candidate=%s K=%llu N=%llu bitwise=%s finite=%s\n",
                current.name, static_cast<unsigned long long>(shape.k),
                static_cast<unsigned long long>(shape.n),
                deterministic ? "PASS" : "FAIL",
                finite_output(second) ? "PASS" : "FAIL");
    all_ok = deterministic && finite_output(second) && all_ok;
    std::printf(
        "result candidate=%s K=%llu N=%llu median_ms=%.6f mad_ms=%.6f "
        "sustained_GBps=%.6f\n",
        current.name, static_cast<unsigned long long>(shape.k),
        static_cast<unsigned long long>(shape.n), median_ms, mad_ms,
        (static_cast<double>(shape.k) * static_cast<double>(shape.n) +
         static_cast<double>(shape.n) * static_cast<double>(sizeof(float)) +
         static_cast<double>(shape.k) + static_cast<double>(sizeof(float))) /
            median_ms / 1.0e6);
  }
  if (medians.empty() || controls[0].empty()) {
    cleanup(&buffers);
    return false;
  }
  const double control_ms = medians.front();
  const auto best = std::min_element(medians.begin(), medians.end());
  const std::size_t best_index =
      static_cast<std::size_t>(best - medians.begin());
  if (weighted_control != nullptr)
    *weighted_control += control_ms * shape.occurrences;
  if (weighted_best != nullptr)
    *weighted_best += *best * shape.occurrences;
  std::printf("shape_summary K=%llu N=%llu occurrences=%u roles=%s "
              "control_ms=%.6f best=%s best_ms=%.6f speedup=%.6f%%\n",
              static_cast<unsigned long long>(shape.k),
              static_cast<unsigned long long>(shape.n), shape.occurrences,
              shape.roles, control_ms, candidate(ids[best_index]).name, *best,
              (control_ms / *best - 1.0) * 100.0);
  cleanup(&buffers);
  std::printf("cleanup K=%llu N=%llu status=complete\n",
              static_cast<unsigned long long>(shape.k),
              static_cast<unsigned long long>(shape.n));
  return all_ok;
}

} // namespace

int main() {
  if (!hip_ok(hipSetDevice(0), "hipSetDevice"))
    return EXIT_FAILURE;
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0), "get properties") ||
      !exact_gfx1030(properties.gcnArchName)) {
    std::fprintf(stderr, "exact gfx1030 required; got %s\n",
                 properties.gcnArchName);
    return EXIT_FAILURE;
  }
  size_t free_bytes = 0U;
  size_t total_bytes = 0U;
  (void)hipMemGetInfo(&free_bytes, &total_bytes);
  std::printf("target=%s device=0 pci=%04x:%02x:%02x name=%s l2=%d "
              "free_bytes=%zu total_bytes=%zu\n",
              properties.gcnArchName, properties.pciDomainID,
              properties.pciBusID, properties.pciDeviceID, properties.name,
              properties.l2CacheSize, free_bytes, total_bytes);
  bool all_ok = all_codes_oracle();
  const std::vector<CandidateId> ids = {
      CandidateId::Control68, CandidateId::Direct8Sequential,
      CandidateId::Direct8Pair2, CandidateId::Direct4Pair2};
  for (const CandidateId id : ids)
    print_resources(candidate(id), 17408U);

  double ignored_control = 0.0;
  double ignored_best = 0.0;
  const std::array<ShapeCase, 2> oracle_shapes = {{
      {128U, 37U, 0U, "nonaligned-N37"},
      {192U, 67U, 0U, "nonaligned-N67"},
  }};
  for (const ShapeCase &shape : oracle_shapes)
    all_ok = run_shape(shape, ids, &ignored_control, &ignored_best) && all_ok;

  double weighted_control = 0.0;
  double weighted_best = 0.0;
  for (const ShapeCase &shape : kQwen38Shapes)
    all_ok = run_shape(shape, ids, &weighted_control, &weighted_best) && all_ok;
  std::printf("weighted_total exact_qwen38_fp8_tensors=%u cold_copies=%u "
              "control_ms_per_token=%.6f best_ms_per_token=%.6f speedup=%.6fx "
              "target_speedup=1.25x status=%s\n",
              shape_occurrences(), kColdCopies, weighted_control, weighted_best,
              weighted_best > 0.0 ? weighted_control / weighted_best : 0.0,
              all_ok ? "PASS" : "FAIL");
  size_t free_after = 0U;
  size_t total_after = 0U;
  const bool mem_ok = hipMemGetInfo(&free_after, &total_after) == hipSuccess;
  std::printf("cleanup_final free_bytes=%zu total_bytes=%zu status=%s\n",
              free_after, total_after, mem_ok ? "PASS" : "FAIL");
  std::printf("summary candidates=%zu warmups=%d measured=%d status=%s\n",
              ids.size(), kWarmups, kMeasured,
              all_ok && mem_ok ? "PASS" : "FAIL");
  return all_ok && mem_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
