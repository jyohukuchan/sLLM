// Phase 78 standalone gfx1030 FP8 decode software-pipeline probe.
//
// The production ID68 geometry (wave32, four output columns per wave, eight
// waves per workgroup, dword8 E4M3FN ingress) is retained as the control.
// Pipe2 and Pipe4 issue two/four independent K chunks per lane before doing
// the corresponding decode/dot work.  The intent is to expose more memory
// requests while keeping the same FP16 ingress, FP32 accumulation, outer
// scales, and BF16-RNE epilogue.  Four distinct matrices are cycled in every
// measured set; a repeated single-weight cache warm result is not accepted.

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

constexpr std::array<uint8_t, 16> kFiniteCodes = {
    0x00U, 0x01U, 0x08U, 0x10U, 0x18U, 0x20U, 0x28U, 0x30U,
    0x80U, 0x81U, 0x88U, 0x90U, 0x98U, 0xa0U, 0xa8U, 0xb0U};

__device__ __constant__ uint8_t kDeviceFiniteCodes[16] = {
    0x00U, 0x01U, 0x08U, 0x10U, 0x18U, 0x20U, 0x28U, 0x30U,
    0x80U, 0x81U, 0x88U, 0x90U, 0x98U, 0xa0U, 0xa8U, 0xb0U};

__device__ __forceinline__ uint8_t weight_code(const uint32_t copy,
                                               const uint64_t column,
                                               const uint64_t inner) {
  return kDeviceFiniteCodes[(copy * 11U + column * 3U + inner * 7U + 9U) & 15U];
}

__device__ __forceinline__ __half2 half2_from_bits(const uint32_t bits) {
  return *reinterpret_cast<const __half2 *>(&bits);
}

__device__ __forceinline__ uint16_t bf16_rne(const float value) {
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

// KUnroll=1 is the ID68 dword8 control. KUnroll=2/4 preload independent K
// chunks (spaced by one wave's lane assignment) before conversion and dot.
template <uint32_t KUnroll>
__global__ __launch_bounds__(kThreads, 1) void pipeline_kernel(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  static_assert(KUnroll == 1U || KUnroll == 2U || KUnroll == 4U);
  constexpr uint32_t columns_per_wave = 4U;
  constexpr uint32_t values_per_chunk = 8U;
  constexpr uint32_t columns_per_workgroup = kWaves * columns_per_wave;
  if (m != 1U || k == 0U || n == 0U || (k % 64U) != 0U)
    return;
  const uint32_t lane = threadIdx.x & (kWave - 1U);
  const uint32_t wave = threadIdx.x / kWave;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * columns_per_workgroup +
      static_cast<uint64_t>(wave) * columns_per_wave;
  if (column_base >= n)
    return;
  const uint64_t iteration_count = k / values_per_chunk;
  const auto *const activation_dwords =
      reinterpret_cast<const uint32_t *>(activation);
  float accumulators[columns_per_wave] = {};
  for (uint64_t base = lane; base < iteration_count;
       base += static_cast<uint64_t>(kWave) * KUnroll) {
    uint32_t activation_first[KUnroll];
    uint32_t activation_second[KUnroll];
    uint32_t weight_first[KUnroll][columns_per_wave];
    uint32_t weight_second[KUnroll][columns_per_wave];
#pragma unroll
    for (uint32_t unroll = 0U; unroll < KUnroll; ++unroll) {
      const uint64_t iteration = base + static_cast<uint64_t>(unroll) * kWave;
      if (iteration < iteration_count) {
        activation_first[unroll] =
            __builtin_nontemporal_load(activation_dwords + iteration * 2U);
        activation_second[unroll] =
            __builtin_nontemporal_load(activation_dwords + iteration * 2U + 1U);
#pragma unroll
        for (uint32_t local_column = 0U; local_column < columns_per_wave;
             ++local_column) {
          const uint64_t column = column_base + local_column;
          if (column < n) {
            const auto *const column_dwords =
                reinterpret_cast<const uint32_t *>(weight + column * k);
            weight_first[unroll][local_column] =
                __builtin_nontemporal_load(column_dwords + iteration * 2U);
            weight_second[unroll][local_column] =
                __builtin_nontemporal_load(column_dwords + iteration * 2U + 1U);
          } else {
            weight_first[unroll][local_column] = 0U;
            weight_second[unroll][local_column] = 0U;
          }
        }
      } else {
        activation_first[unroll] = 0U;
        activation_second[unroll] = 0U;
#pragma unroll
        for (uint32_t local_column = 0U; local_column < columns_per_wave;
             ++local_column) {
          weight_first[unroll][local_column] = 0U;
          weight_second[unroll][local_column] = 0U;
        }
      }
    }
#pragma unroll
    for (uint32_t unroll = 0U; unroll < KUnroll; ++unroll) {
      const uint64_t iteration = base + static_cast<uint64_t>(unroll) * kWave;
      if (iteration >= iteration_count)
        continue;
      const sllm_lowp::E4M3FnFp16x4Bits activation_low =
          sllm_lowp::e4m3fnx4_to_fp16x2_bits(activation_first[unroll]);
      const sllm_lowp::E4M3FnFp16x4Bits activation_high =
          sllm_lowp::e4m3fnx4_to_fp16x2_bits(activation_second[unroll]);
      const __half2 a0 = half2_from_bits(activation_low.low);
      const __half2 a1 = half2_from_bits(activation_low.high);
      const __half2 a2 = half2_from_bits(activation_high.low);
      const __half2 a3 = half2_from_bits(activation_high.high);
#pragma unroll
      for (uint32_t local_column = 0U; local_column < columns_per_wave;
           ++local_column) {
        const uint64_t column = column_base + local_column;
        if (column >= n)
          continue;
        const sllm_lowp::E4M3FnFp16x4Bits w0 =
            sllm_lowp::e4m3fnx4_to_fp16x2_bits(
                weight_first[unroll][local_column]);
        const sllm_lowp::E4M3FnFp16x4Bits w1 =
            sllm_lowp::e4m3fnx4_to_fp16x2_bits(
                weight_second[unroll][local_column]);
        accumulators[local_column] = amd_mixed_dot(
            a0, half2_from_bits(w0.low), accumulators[local_column], false);
        accumulators[local_column] = amd_mixed_dot(
            a1, half2_from_bits(w0.high), accumulators[local_column], false);
        accumulators[local_column] = amd_mixed_dot(
            a2, half2_from_bits(w1.low), accumulators[local_column], false);
        accumulators[local_column] = amd_mixed_dot(
            a3, half2_from_bits(w1.high), accumulators[local_column], false);
      }
    }
  }
#pragma unroll
  for (uint32_t local_column = 0U; local_column < columns_per_wave;
       ++local_column) {
#pragma unroll
    for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U)
      accumulators[local_column] +=
          __shfl_down(accumulators[local_column], offset, kWave);
    const uint64_t column = column_base + local_column;
    if (lane == 0U && column < n)
      output[column] = bf16_rne(accumulators[local_column] *
                                activation_scales[0] * weight_scales[column]);
  }
}

enum class CandidateId : uint32_t { Control68, Pipe2, Pipe4 };
struct Candidate final {
  CandidateId id;
  const char *name;
  const void *function;
  uint32_t k_unroll;
};

Candidate candidate(const CandidateId id) {
  switch (id) {
  case CandidateId::Control68:
    return {id, "id68-dword8-wave4col32-control",
            reinterpret_cast<const void *>(pipeline_kernel<1U>), 1U};
  case CandidateId::Pipe2:
    return {id, "direct-wave4-kpipe2-col32",
            reinterpret_cast<const void *>(pipeline_kernel<2U>), 2U};
  case CandidateId::Pipe4:
    return {id, "direct-wave4-kpipe4-col32",
            reinterpret_cast<const void *>(pipeline_kernel<4U>), 4U};
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

__global__ void fill_weight(uint8_t *const weight, const uint64_t k,
                            const uint64_t n, const uint32_t copy) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= k * n)
    return;
  weight[index] = weight_code(copy, index / k, index % k);
}

__global__ void e4m3_oracle(const uint8_t *const input,
                            uint16_t *const output) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 256U)
    output[index] = sllm_lowp::e4m3fn_to_fp16_bits(input[index]);
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
      (shape.k % 64U) != 0U || shape.k > SIZE_MAX / shape.n)
    return false;
  const std::size_t weight_bytes = static_cast<std::size_t>(shape.k * shape.n);
  const std::size_t scale_bytes =
      static_cast<std::size_t>(shape.n) * sizeof(float);
  const std::size_t output_bytes =
      static_cast<std::size_t>(shape.n) * sizeof(uint16_t);
  if (!hip_ok(
          hipMalloc(reinterpret_cast<void **>(&buffers->activation), shape.k),
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
                        sizeof(float), hipMemcpyHostToDevice),
              "copy activation scale"))
    return false;
  const uint64_t total = shape.k * shape.n;
  const uint32_t blocks =
      static_cast<uint32_t>((total + kThreads - 1U) / kThreads);
  for (uint32_t copy = 0U; copy < kColdCopies; ++copy) {
    hipLaunchKernelGGL(fill_weight, dim3(blocks), dim3(kThreads), 0U,
                       buffers->stream, buffers->planes[copy].weight, shape.k,
                       shape.n, copy);
    if (!hip_ok(hipGetLastError(), "fill weight"))
      return false;
    std::vector<float> scales;
    fill_scales(shape, copy, &scales);
    if (!hip_ok(hipMemcpy(buffers->planes[copy].scales, scales.data(),
                          scales.size() * sizeof(float), hipMemcpyHostToDevice),
                "copy scales"))
      return false;
  }
  return hip_ok(hipStreamSynchronize(buffers->stream), "sync fill");
}

bool launch(const Candidate &current, const ShapeCase &shape,
            const uint32_t copy, Buffers *const buffers) {
  if (copy >= kColdCopies)
    return false;
  const uint64_t columns_per_workgroup = static_cast<uint64_t>(kWaves) * 4U;
  const uint64_t blocks =
      (shape.n + columns_per_workgroup - 1U) / columns_per_workgroup;
  if (blocks == 0U || blocks > UINT32_MAX)
    return false;
  const WeightPlane &plane = buffers->planes[copy];
  const dim3 grid(static_cast<uint32_t>(blocks));
  const dim3 block(kThreads);
  switch (current.id) {
  case CandidateId::Control68:
    hipLaunchKernelGGL((pipeline_kernel<1U>), grid, block, 0U, buffers->stream,
                       buffers->activation, buffers->activation_scale,
                       plane.weight, plane.scales, buffers->output, 1U, shape.k,
                       shape.n);
    break;
  case CandidateId::Pipe2:
    hipLaunchKernelGGL((pipeline_kernel<2U>), grid, block, 0U, buffers->stream,
                       buffers->activation, buffers->activation_scale,
                       plane.weight, plane.scales, buffers->output, 1U, shape.k,
                       shape.n);
    break;
  case CandidateId::Pipe4:
    hipLaunchKernelGGL((pipeline_kernel<4U>), grid, block, 0U, buffers->stream,
                       buffers->activation, buffers->activation_scale,
                       plane.weight, plane.scales, buffers->output, 1U, shape.k,
                       shape.n);
    break;
  }
  return hipGetLastError() == hipSuccess;
}

bool copy_output(const ShapeCase &shape, Buffers *const buffers,
                 std::vector<uint16_t> *const out) {
  out->resize(static_cast<std::size_t>(shape.n));
  return hip_ok(hipMemcpy(out->data(), buffers->output,
                          out->size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "copy output");
}

std::vector<uint16_t> cpu_oracle(const ShapeCase &shape,
                                 const std::vector<uint8_t> &activation,
                                 const uint32_t copy) {
  std::vector<uint16_t> result(static_cast<std::size_t>(shape.n));
  for (uint64_t column = 0U; column < shape.n; ++column) {
    float accumulator = 0.0F;
    for (uint64_t inner = 0U; inner < shape.k; ++inner)
      accumulator = std::fmaf(
          host_e4m3(activation[static_cast<std::size_t>(inner)]),
          host_e4m3(host_weight_code(copy, column, inner)), accumulator);
    const float scale = 0.625F + static_cast<float>(column % 11U) * 0.03125F +
                        static_cast<float>(copy) * 0.0078125F;
    result[static_cast<std::size_t>(column)] =
        host_bf16_rne(accumulator * 0.875F * scale);
  }
  return result;
}

bool compare(const char *const name, const char *const relation,
             const ShapeCase &shape, const std::vector<uint16_t> &actual,
             const std::vector<uint16_t> &expected) {
  std::size_t mismatches = 0U;
  uint32_t max_ulp = 0U;
  for (std::size_t i = 0U; i < actual.size(); ++i) {
    if (actual[i] != expected[i])
      ++mismatches;
    const uint32_t left = actual[i];
    const uint32_t right = expected[i];
    max_ulp = std::max(max_ulp, left > right ? left - right : right - left);
  }
  std::printf("oracle candidate=%s relation=%s K=%llu N=%llu mismatches=%zu "
              "max_bf16_ulp=%u status=%s\n",
              name, relation, static_cast<unsigned long long>(shape.k),
              static_cast<unsigned long long>(shape.n), mismatches, max_ulp,
              mismatches == 0U ? "PASS" : "FAIL");
  return mismatches == 0U;
}

bool finite_output(const std::vector<uint16_t> &output) {
  for (const uint16_t bits : output)
    if ((bits & 0x7f80U) == 0x7f80U)
      return false;
  return true;
}

bool measure_all(const std::vector<CandidateId> &ids, const ShapeCase &shape,
                 Buffers *const buffers, std::vector<double> *const medians,
                 std::vector<double> *const mads) {
  if (ids.empty() || medians == nullptr || mads == nullptr)
    return false;
  std::vector<std::array<double, kMeasured>> samples(ids.size());
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    for (std::size_t position = 0U; position < ids.size(); ++position) {
      const std::size_t index =
          (position + static_cast<std::size_t>(warmup)) % ids.size();
      if (!launch(candidate(ids[index]), shape,
                  static_cast<uint32_t>(warmup) % kColdCopies, buffers) ||
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

bool all_codes_oracle() {
  std::array<uint8_t, 256> input{};
  std::array<uint16_t, 256> actual{};
  for (uint32_t i = 0U; i < 256U; ++i)
    input[i] = static_cast<uint8_t>(i);
  uint8_t *device_input = nullptr;
  uint16_t *device_output = nullptr;
  bool ok =
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_input), 256U),
             "malloc code input") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_output),
                       actual.size() * sizeof(uint16_t)),
             "malloc code output") &&
      hip_ok(hipMemcpy(device_input, input.data(), 256U, hipMemcpyHostToDevice),
             "copy code input");
  if (ok) {
    hipLaunchKernelGGL(e4m3_oracle, dim3(1U), dim3(256U), 0U, nullptr,
                       device_input, device_output);
    ok = hip_ok(hipGetLastError(), "launch code oracle") &&
         hip_ok(hipDeviceSynchronize(), "sync code oracle") &&
         hip_ok(hipMemcpy(actual.data(), device_output,
                          actual.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "copy code output");
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

void resources(const Candidate &current) {
  hipFuncAttributes attributes{};
  const hipError_t attr = hipFuncGetAttributes(&attributes, current.function);
  int active = 0;
  const hipError_t occupancy = hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, current.function, kThreads, 0U);
  std::printf("resources candidate=%s K_unroll=%u vgpr=%d sgpr=unavailable "
              "static_lds=%zu dynamic_lds=0 scratch=%zu active_blocks=%d "
              "attr=%s occupancy=%s\n",
              current.name, current.k_unroll, attributes.numRegs,
              attributes.sharedSizeBytes, attributes.localSizeBytes, active,
              hipGetErrorString(attr), hipGetErrorString(occupancy));
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
  std::vector<double> medians;
  std::vector<double> mads;
  if (!measure_all(ids, shape, &buffers, &medians, &mads)) {
    cleanup(&buffers);
    return false;
  }
  std::vector<std::vector<uint16_t>> controls(kColdCopies);
  bool all_ok = true;
  for (std::size_t index = 0U; index < ids.size(); ++index) {
    const Candidate current = candidate(ids[index]);
    resources(current);
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
      if (!finite_output(actual))
        all_ok = false;
      if (current.id == CandidateId::Control68) {
        controls[copy] = actual;
      } else {
        all_ok =
            compare(current.name, "vs-id68", shape, actual, controls[copy]) &&
            all_ok;
      }
      if (shape.n <= 128U) {
        const std::vector<uint16_t> expected =
            cpu_oracle(shape, activation, copy);
        all_ok =
            compare(current.name, "host", shape, actual, expected) && all_ok;
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
    const bool deterministic = first == second && finite_output(second);
    std::printf("determinism candidate=%s K=%llu N=%llu bitwise=%s finite=%s\n",
                current.name, static_cast<unsigned long long>(shape.k),
                static_cast<unsigned long long>(shape.n),
                deterministic ? "PASS" : "FAIL",
                finite_output(second) ? "PASS" : "FAIL");
    all_ok = deterministic && all_ok;
    const double bytes =
        static_cast<double>(shape.k) * static_cast<double>(shape.n) +
        static_cast<double>(shape.n) * static_cast<double>(sizeof(float)) +
        static_cast<double>(shape.k) + static_cast<double>(sizeof(float));
    std::printf("result candidate=%s K=%llu N=%llu median_ms=%.6f mad_ms=%.6f "
                "effective_weight_GBps=%.6f\n",
                current.name, static_cast<unsigned long long>(shape.k),
                static_cast<unsigned long long>(shape.n), medians[index],
                mads[index], bytes / medians[index] / 1.0e6);
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
  const std::vector<CandidateId> ids = {CandidateId::Control68,
                                        CandidateId::Pipe2, CandidateId::Pipe4};
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
              "target=1.15x status=%s\n",
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
