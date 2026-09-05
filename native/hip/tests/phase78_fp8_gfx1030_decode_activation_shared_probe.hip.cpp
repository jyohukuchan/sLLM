// Phase 78 standalone gfx1030 FP8 outer-scale decode activation-LDS probe.
//
// The control reproduces the production ID68 dword8 wave4col32 arithmetic:
// every lane loads two activation dwords and four weight-column dword pairs,
// expands E4M3FN values to exact FP16, and accumulates four FP16 dot2 results
// in FP32.  Candidates decode the activation row once per workgroup into
// dynamic LDS (K uint16 FP16 bits), then let all eight wave32s reuse that row.
// The 8-columns/wave candidate is included only as a resource/performance
// probe; the required production-shaped candidate is 4-columns/wave.
//
// This file is intentionally standalone and is not part of production.  The
// exact Qwen3.8-27B FP8 shape/occurrence manifest is copied from
// crates/sllm-core/src/quantized_model.rs::build_qwen38_inventory:
// layers 56..63 MLP, 16 full-attention layers, 48 linear-attention layers,
// and one lm_head, totaling 233 FP8 tensors.

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
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kWaveWidth = 32U;
constexpr uint32_t kWaves = kThreads / kWaveWidth;
constexpr int kWarmups = 3;
constexpr int kMeasured = 10;

struct ShapeCase final {
  uint64_t k;
  uint64_t n;
  uint32_t occurrences;
  const char *roles;
};

constexpr std::array<ShapeCase, 8> kQwen38Fp8Shapes = {{
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
  for (const ShapeCase &shape : kQwen38Fp8Shapes) {
    result += shape.occurrences;
  }
  return result;
}

static_assert(shape_occurrences() == 233U);

struct ScaledHalfPacks final {
  __half2 first;
  __half2 second;
};

__device__ __forceinline__ ScaledHalfPacks
e4m3x4_to_half2(const uint32_t packed) noexcept {
  ScaledHalfPacks result{};
  sllm_lowp::e4m3fnx4_to_half2x2(packed, &result.first, &result.second);
  return result;
}

__device__ __forceinline__ uint16_t bf16_rne(const float value) noexcept {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t exponent = bits & UINT32_C(0x7f800000);
  const uint32_t fraction = bits & UINT32_C(0x007fffff);
  if (exponent == UINT32_C(0x7f800000)) {
    if (fraction != 0U) {
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

template <uint32_t ColumnsPerWave>
__global__ __launch_bounds__(kThreads, 1) void dword8_control_kernel(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  static_assert(ColumnsPerWave == 4U);
  constexpr uint32_t values_per_dword = 4U;
  constexpr uint32_t dwords_per_iteration = 2U;
  constexpr uint32_t values_per_iteration =
      values_per_dword * dwords_per_iteration;
  constexpr uint32_t columns_per_workgroup = kWaves * ColumnsPerWave;
  if (m != 1U || k == 0U || n == 0U || (k % 64U) != 0U) {
    return;
  }
  const uint32_t lane = threadIdx.x & (kWaveWidth - 1U);
  const uint32_t wave = threadIdx.x / kWaveWidth;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * columns_per_workgroup +
      static_cast<uint64_t>(wave) * ColumnsPerWave;
  if (column_base >= n) {
    return;
  }
  const uint64_t iteration_count = k / values_per_iteration;
  const auto *const activation_dwords =
      reinterpret_cast<const uint32_t *>(activation);
  float accumulators[ColumnsPerWave] = {};
  for (uint64_t iteration = lane; iteration < iteration_count;
       iteration += kWaveWidth) {
    const uint32_t activation_first = __builtin_nontemporal_load(
        activation_dwords + iteration * dwords_per_iteration);
    const uint32_t activation_second = __builtin_nontemporal_load(
        activation_dwords + iteration * dwords_per_iteration + 1U);
    uint32_t weight_first[ColumnsPerWave];
    uint32_t weight_second[ColumnsPerWave];
#pragma unroll
    for (uint32_t local_column = 0U; local_column < ColumnsPerWave;
         ++local_column) {
      const uint64_t column = column_base + local_column;
      if (column < n) {
        const auto *const column_dwords =
            reinterpret_cast<const uint32_t *>(weight + column * k);
        weight_first[local_column] = __builtin_nontemporal_load(
            column_dwords + iteration * dwords_per_iteration);
        weight_second[local_column] = __builtin_nontemporal_load(
            column_dwords + iteration * dwords_per_iteration + 1U);
      } else {
        weight_first[local_column] = 0U;
        weight_second[local_column] = 0U;
      }
    }
    const ScaledHalfPacks activation_first_pairs =
        e4m3x4_to_half2(activation_first);
    const ScaledHalfPacks activation_second_pairs =
        e4m3x4_to_half2(activation_second);
#pragma unroll
    for (uint32_t local_column = 0U; local_column < ColumnsPerWave;
         ++local_column) {
      if (column_base + local_column < n) {
        const ScaledHalfPacks weight_first_pairs =
            e4m3x4_to_half2(weight_first[local_column]);
        const ScaledHalfPacks weight_second_pairs =
            e4m3x4_to_half2(weight_second[local_column]);
        accumulators[local_column] = amd_mixed_dot(
            activation_first_pairs.first, weight_first_pairs.first,
            accumulators[local_column], false);
        accumulators[local_column] = amd_mixed_dot(
            activation_first_pairs.second, weight_first_pairs.second,
            accumulators[local_column], false);
        accumulators[local_column] = amd_mixed_dot(
            activation_second_pairs.first, weight_second_pairs.first,
            accumulators[local_column], false);
        accumulators[local_column] = amd_mixed_dot(
            activation_second_pairs.second, weight_second_pairs.second,
            accumulators[local_column], false);
      }
    }
  }
#pragma unroll
  for (uint32_t local_column = 0U; local_column < ColumnsPerWave;
       ++local_column) {
#pragma unroll
    for (uint32_t offset = kWaveWidth / 2U; offset != 0U; offset >>= 1U) {
      accumulators[local_column] +=
          __shfl_down(accumulators[local_column], offset, kWaveWidth);
    }
    const uint64_t column = column_base + local_column;
    if (lane == 0U && column < n) {
      output[column] = bf16_rne(accumulators[local_column] *
                                activation_scales[0] * weight_scales[column]);
    }
  }
}

template <uint32_t ColumnsPerWave>
__global__ __launch_bounds__(kThreads, 1) void activation_shared_kernel(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t columns_per_workgroup = kWaves * ColumnsPerWave;
  static_assert(ColumnsPerWave == 4U || ColumnsPerWave == 8U);
  if (m != 1U || k == 0U || n == 0U || (k % 64U) != 0U) {
    return;
  }
  const uint32_t lane = threadIdx.x & (kWaveWidth - 1U);
  const uint32_t wave = threadIdx.x / kWaveWidth;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * columns_per_workgroup +
      static_cast<uint64_t>(wave) * ColumnsPerWave;
  extern __shared__ __align__(16) uint16_t activation_fp16[];
  // K is a multiple of 64 for every exact production shape, so these dword
  // stores cover the row exactly and perform one global activation expansion
  // per workgroup.  The LDS row is then read by all eight waves below.
  for (uint64_t index = static_cast<uint64_t>(threadIdx.x) * 4U; index < k;
       index += static_cast<uint64_t>(kThreads) * 4U) {
    const uint32_t packed = __builtin_nontemporal_load(
        reinterpret_cast<const uint32_t *>(activation + index));
    const sllm_lowp::E4M3FnFp16x4Bits expanded =
        sllm_lowp::e4m3fnx4_to_fp16x2_bits(packed);
    auto *const destination =
        reinterpret_cast<uint32_t *>(activation_fp16 + index);
    destination[0] = expanded.low;
    destination[1] = expanded.high;
  }
  __syncthreads();
  if (column_base >= n) {
    return;
  }
  const uint64_t iteration_count = k / 8U;
  float accumulators[ColumnsPerWave] = {};
  for (uint64_t iteration = lane; iteration < iteration_count;
       iteration += kWaveWidth) {
    const uint64_t index = iteration * 8U;
    const auto *const activation_pairs =
        reinterpret_cast<const __half2 *>(activation_fp16 + index);
    const __half2 activation_first = activation_pairs[0];
    const __half2 activation_second = activation_pairs[1];
    const __half2 activation_third = activation_pairs[2];
    const __half2 activation_fourth = activation_pairs[3];
#pragma unroll
    for (uint32_t local_column = 0U; local_column < ColumnsPerWave;
         ++local_column) {
      const uint64_t column = column_base + local_column;
      if (column < n) {
        const auto *const column_dwords =
            reinterpret_cast<const uint32_t *>(weight + column * k + index);
        const ScaledHalfPacks weight_first =
            e4m3x4_to_half2(__builtin_nontemporal_load(column_dwords));
        const ScaledHalfPacks weight_second =
            e4m3x4_to_half2(__builtin_nontemporal_load(column_dwords + 1U));
        accumulators[local_column] =
            amd_mixed_dot(activation_first, weight_first.first,
                          accumulators[local_column], false);
        accumulators[local_column] =
            amd_mixed_dot(activation_second, weight_first.second,
                          accumulators[local_column], false);
        accumulators[local_column] =
            amd_mixed_dot(activation_third, weight_second.first,
                          accumulators[local_column], false);
        accumulators[local_column] =
            amd_mixed_dot(activation_fourth, weight_second.second,
                          accumulators[local_column], false);
      }
    }
  }
#pragma unroll
  for (uint32_t local_column = 0U; local_column < ColumnsPerWave;
       ++local_column) {
#pragma unroll
    for (uint32_t offset = kWaveWidth / 2U; offset != 0U; offset >>= 1U) {
      accumulators[local_column] +=
          __shfl_down(accumulators[local_column], offset, kWaveWidth);
    }
    const uint64_t column = column_base + local_column;
    if (lane == 0U && column < n) {
      output[column] = bf16_rne(accumulators[local_column] *
                                activation_scales[0] * weight_scales[column]);
    }
  }
}

__global__ void e4m3_decode_oracle_kernel(const uint8_t *const input,
                                          uint16_t *const output) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 256U) {
    output[index] = sllm_lowp::e4m3fn_to_fp16_bits(input[index]);
  }
}

enum class CandidateId { Control, ActivationShared4, ActivationShared8 };

struct Candidate final {
  CandidateId id;
  const char *name;
  const void *function;
  uint32_t columns_per_wave;
};

Candidate candidate(const CandidateId id) {
  switch (id) {
  case CandidateId::Control:
    return {id, "id68-dword8-wave4col32-control",
            reinterpret_cast<const void *>(dword8_control_kernel<4U>), 4U};
  case CandidateId::ActivationShared4:
    return {id, "candidate-activation-lds-wave4col32",
            reinterpret_cast<const void *>(activation_shared_kernel<4U>), 4U};
  case CandidateId::ActivationShared8:
    return {id, "candidate-activation-lds-wave8col64",
            reinterpret_cast<const void *>(activation_shared_kernel<8U>), 8U};
  }
  return {CandidateId::Control, "invalid", nullptr, 0U};
}

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::fprintf(stderr, "hip error operation=%s status=%s (%s)\n", operation,
               hipGetErrorName(status), hipGetErrorString(status));
  return false;
}

bool exact_gfx1030(const char *const arch) {
  if (arch == nullptr) {
    return false;
  }
  const std::string_view value(arch);
  constexpr std::string_view prefix = "gfx1030";
  return value == prefix || (value.size() > prefix.size() &&
                             value.compare(0U, prefix.size(), prefix) == 0 &&
                             value[prefix.size()] == ':');
}

float host_e4m3(const uint8_t bits) {
  const uint8_t magnitude = bits & UINT8_C(0x7f);
  const uint8_t exponent = magnitude >> 3U;
  const uint8_t mantissa = magnitude & UINT8_C(0x07);
  float value = 0.0F;
  if (exponent == 0U) {
    value = static_cast<float>(mantissa) * 0x1p-9F;
  } else if (magnitude == UINT8_C(0x7f)) {
    value = std::numeric_limits<float>::quiet_NaN();
  } else {
    value = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                       static_cast<int>(exponent) - 7);
  }
  return (bits & UINT8_C(0x80)) != 0U ? -value : value;
}

uint16_t host_bf16_rne(const float value) {
  uint32_t bits = 0U;
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

struct Buffers final {
  uint8_t *activation = nullptr;
  float *activation_scales = nullptr;
  uint8_t *weight = nullptr;
  float *weight_scales = nullptr;
  uint16_t *output = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
};

void cleanup(Buffers *const buffers) {
  if (buffers == nullptr) {
    return;
  }
  if (buffers->stop != nullptr) {
    (void)hipEventDestroy(buffers->stop);
  }
  if (buffers->start != nullptr) {
    (void)hipEventDestroy(buffers->start);
  }
  if (buffers->stream != nullptr) {
    (void)hipStreamDestroy(buffers->stream);
  }
  if (buffers->output != nullptr) {
    (void)hipFree(buffers->output);
  }
  if (buffers->weight_scales != nullptr) {
    (void)hipFree(buffers->weight_scales);
  }
  if (buffers->weight != nullptr) {
    (void)hipFree(buffers->weight);
  }
  if (buffers->activation_scales != nullptr) {
    (void)hipFree(buffers->activation_scales);
  }
  if (buffers->activation != nullptr) {
    (void)hipFree(buffers->activation);
  }
  *buffers = {};
}

bool make_buffers(const ShapeCase &shape, Buffers *const buffers) {
  if (buffers == nullptr || shape.k == 0U || shape.n == 0U ||
      (shape.k % 64U) != 0U || shape.k > SIZE_MAX / shape.n) {
    return false;
  }
  const std::size_t weight_bytes = static_cast<std::size_t>(shape.k * shape.n);
  const std::size_t output_bytes = static_cast<std::size_t>(shape.n) * 2U;
  return hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->activation),
                          static_cast<std::size_t>(shape.k)),
                "hipMalloc activation") &&
         hip_ok(
             hipMalloc(reinterpret_cast<void **>(&buffers->activation_scales),
                       sizeof(float)),
             "hipMalloc activation scales") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->weight),
                          weight_bytes),
                "hipMalloc weight") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->weight_scales),
                          static_cast<std::size_t>(shape.n) * sizeof(float)),
                "hipMalloc weight scales") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->output),
                          output_bytes),
                "hipMalloc output") &&
         hip_ok(hipStreamCreate(&buffers->stream), "hipStreamCreate") &&
         hip_ok(hipEventCreate(&buffers->start), "hipEventCreate start") &&
         hip_ok(hipEventCreate(&buffers->stop), "hipEventCreate stop");
}

void fill_inputs(const ShapeCase &shape, std::vector<uint8_t> *const activation,
                 std::vector<float> *const activation_scales,
                 std::vector<uint8_t> *const weight,
                 std::vector<float> *const weight_scales) {
  activation->resize(static_cast<std::size_t>(shape.k));
  activation_scales->assign(1U, 0.875F);
  weight->resize(static_cast<std::size_t>(shape.k * shape.n));
  weight_scales->resize(static_cast<std::size_t>(shape.n));
  // Keep finite values in a small range so CPU FMA and the exact FP16 ingress
  // produce a stable BF16 oracle while still exercising signed/subnormal lanes.
  constexpr std::array<uint8_t, 16> codes = {
      UINT8_C(0x00), UINT8_C(0x01), UINT8_C(0x08), UINT8_C(0x10),
      UINT8_C(0x18), UINT8_C(0x20), UINT8_C(0x28), UINT8_C(0x30),
      UINT8_C(0x80), UINT8_C(0x81), UINT8_C(0x88), UINT8_C(0x90),
      UINT8_C(0x98), UINT8_C(0xa0), UINT8_C(0xa8), UINT8_C(0xb0)};
  for (uint64_t inner = 0U; inner < shape.k; ++inner) {
    (*activation)[static_cast<std::size_t>(inner)] =
        codes[(inner * 5U + 3U) % codes.size()];
  }
  for (uint64_t column = 0U; column < shape.n; ++column) {
    (*weight_scales)[static_cast<std::size_t>(column)] =
        0.625F + static_cast<float>(column % 11U) * 0.03125F;
    const std::size_t base = static_cast<std::size_t>(column * shape.k);
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      (*weight)[base + static_cast<std::size_t>(inner)] =
          codes[(column * 3U + inner * 7U + 9U) % codes.size()];
    }
  }
}

bool upload(const ShapeCase &shape, const std::vector<uint8_t> &activation,
            const std::vector<float> &activation_scales,
            const std::vector<uint8_t> &weight,
            const std::vector<float> &weight_scales, Buffers *const buffers) {
  return hip_ok(hipMemcpy(buffers->activation, activation.data(),
                          activation.size(), hipMemcpyHostToDevice),
                "copy activation") &&
         hip_ok(hipMemcpy(buffers->activation_scales, activation_scales.data(),
                          sizeof(float), hipMemcpyHostToDevice),
                "copy activation scale") &&
         hip_ok(hipMemcpy(buffers->weight, weight.data(), weight.size(),
                          hipMemcpyHostToDevice),
                "copy weight") &&
         hip_ok(hipMemcpy(buffers->weight_scales, weight_scales.data(),
                          weight_scales.size() * sizeof(float),
                          hipMemcpyHostToDevice),
                "copy weight scales") &&
         hip_ok(hipMemset(buffers->output, 0,
                          static_cast<std::size_t>(shape.n) * sizeof(uint16_t)),
                "clear output");
}

std::size_t dynamic_shared_bytes(const Candidate &current, const uint64_t k) {
  return current.id == CandidateId::Control ? 0U
                                            : static_cast<std::size_t>(k) * 2U;
}

bool launch(const Candidate &current, const ShapeCase &shape,
            Buffers *const buffers) {
  const uint64_t blocks = (shape.n + kWaves * current.columns_per_wave - 1U) /
                          (kWaves * current.columns_per_wave);
  if (blocks == 0U || blocks > UINT32_MAX) {
    return false;
  }
  const dim3 grid(static_cast<uint32_t>(blocks));
  const dim3 block(kThreads);
  const size_t shared = dynamic_shared_bytes(current, shape.k);
  switch (current.id) {
  case CandidateId::Control:
    hipLaunchKernelGGL(
        (dword8_control_kernel<4U>), grid, block, 0U, buffers->stream,
        buffers->activation, buffers->activation_scales, buffers->weight,
        buffers->weight_scales, buffers->output, 1U, shape.k, shape.n);
    break;
  case CandidateId::ActivationShared4:
    hipLaunchKernelGGL(
        (activation_shared_kernel<4U>), grid, block, shared, buffers->stream,
        buffers->activation, buffers->activation_scales, buffers->weight,
        buffers->weight_scales, buffers->output, 1U, shape.k, shape.n);
    break;
  case CandidateId::ActivationShared8:
    hipLaunchKernelGGL(
        (activation_shared_kernel<8U>), grid, block, shared, buffers->stream,
        buffers->activation, buffers->activation_scales, buffers->weight,
        buffers->weight_scales, buffers->output, 1U, shape.k, shape.n);
    break;
  }
  return hipGetLastError() == hipSuccess;
}

std::vector<uint16_t> cpu_oracle(const ShapeCase &shape,
                                 const std::vector<uint8_t> &activation,
                                 const float activation_scale,
                                 const std::vector<uint8_t> &weight,
                                 const std::vector<float> &weight_scales) {
  std::vector<uint16_t> expected(static_cast<std::size_t>(shape.n));
  for (uint64_t column = 0U; column < shape.n; ++column) {
    float accumulator = 0.0F;
    const std::size_t base = static_cast<std::size_t>(column * shape.k);
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      accumulator =
          std::fmaf(host_e4m3(activation[static_cast<std::size_t>(inner)]),
                    host_e4m3(weight[base + static_cast<std::size_t>(inner)]),
                    accumulator);
    }
    expected[static_cast<std::size_t>(column)] =
        host_bf16_rne(accumulator * activation_scale *
                      weight_scales[static_cast<std::size_t>(column)]);
  }
  return expected;
}

bool compare_outputs(const char *const name, const ShapeCase &shape,
                     const std::vector<uint16_t> &actual,
                     const std::vector<uint16_t> &expected) {
  std::size_t mismatches = 0U;
  uint32_t max_ulp = 0U;
  std::size_t printed = 0U;
  for (std::size_t index = 0U; index < actual.size(); ++index) {
    if (actual[index] != expected[index]) {
      ++mismatches;
      if (printed < 4U) {
        std::printf("oracle_mismatch candidate=%s index=%zu actual=0x%04x "
                    "expected=0x%04x\n",
                    name, index, static_cast<unsigned>(actual[index]),
                    static_cast<unsigned>(expected[index]));
        ++printed;
      }
    }
    const uint32_t left = actual[index];
    const uint32_t right = expected[index];
    max_ulp = std::max(max_ulp, left > right ? left - right : right - left);
  }
  std::printf("oracle candidate=%s K=%llu N=%llu mismatches=%zu "
              "max_bf16_ulp=%u status=%s\n",
              name, static_cast<unsigned long long>(shape.k),
              static_cast<unsigned long long>(shape.n), mismatches, max_ulp,
              mismatches == 0U ? "PASS" : "FAIL");
  return mismatches == 0U;
}

bool measure(const Candidate &current, const ShapeCase &shape,
             Buffers *const buffers, double *const median_ms,
             double *const mad_ms, double *const minimum_ms,
             double *const maximum_ms) {
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    if (!launch(current, shape, buffers) ||
        !hip_ok(hipStreamSynchronize(buffers->stream), "warmup synchronize")) {
      return false;
    }
  }
  std::array<double, kMeasured> samples{};
  for (int iteration = 0; iteration < kMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(buffers->start, buffers->stream),
                "event start") ||
        !launch(current, shape, buffers) ||
        !hip_ok(hipEventRecord(buffers->stop, buffers->stream), "event stop") ||
        !hip_ok(hipEventSynchronize(buffers->stop), "event synchronize")) {
      return false;
    }
    float elapsed_ms = 0.0F;
    if (!hip_ok(hipEventElapsedTime(&elapsed_ms, buffers->start, buffers->stop),
                "event elapsed")) {
      return false;
    }
    samples[static_cast<std::size_t>(iteration)] = elapsed_ms;
  }
  std::sort(samples.begin(), samples.end());
  *minimum_ms = samples.front();
  *maximum_ms = samples.back();
  *median_ms = samples[kMeasured / 2U];
  std::array<double, kMeasured> deviations{};
  for (int index = 0; index < kMeasured; ++index) {
    deviations[static_cast<std::size_t>(index)] =
        std::fabs(samples[static_cast<std::size_t>(index)] - *median_ms);
  }
  std::sort(deviations.begin(), deviations.end());
  *mad_ms = deviations[kMeasured / 2U];
  return true;
}

bool all_codes_oracle() {
  std::array<uint8_t, 256> input{};
  std::array<uint16_t, 256> actual{};
  for (uint32_t index = 0U; index < 256U; ++index) {
    input[index] = static_cast<uint8_t>(index);
  }
  uint8_t *device_input = nullptr;
  uint16_t *device_output = nullptr;
  bool ok =
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_input), input.size()),
             "malloc all-code input") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_output),
                       actual.size() * sizeof(uint16_t)),
             "malloc all-code output") &&
      hip_ok(hipMemcpy(device_input, input.data(), input.size(),
                       hipMemcpyHostToDevice),
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
    for (uint32_t index = 0U; index < 256U; ++index) {
      if (actual[index] != sllm_lowp::e4m3fn_to_fp16_bits(input[index])) {
        ++mismatches;
      }
    }
  }
  std::printf("oracle all_e4m3_codes=256 mismatches=%zu status=%s\n",
              mismatches, ok && mismatches == 0U ? "PASS" : "FAIL");
  if (device_output != nullptr) {
    (void)hipFree(device_output);
  }
  if (device_input != nullptr) {
    (void)hipFree(device_input);
  }
  return ok && mismatches == 0U;
}

void print_resources(const Candidate &current, const uint64_t k) {
  hipFuncAttributes attributes{};
  const hipError_t attr_status =
      hipFuncGetAttributes(&attributes, current.function);
  int active_blocks = 0;
  const hipError_t occupancy_status =
      hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks, current.function, kThreads,
          dynamic_shared_bytes(current, k));
  std::printf(
      "resources candidate=%s registers=%d static_lds=%zu dynamic_lds=%zu "
      "spill_local=%zu active_blocks=%d attr=%s occupancy=%s\n",
      current.name, attributes.numRegs, attributes.sharedSizeBytes,
      dynamic_shared_bytes(current, k), attributes.localSizeBytes,
      active_blocks, hipGetErrorString(attr_status),
      hipGetErrorString(occupancy_status));
}

bool run_shape(const ShapeCase &shape,
               const std::vector<CandidateId> &candidates,
               double *const weighted_control, double *const weighted_best) {
  std::vector<uint8_t> activation;
  std::vector<float> activation_scales;
  std::vector<uint8_t> weight;
  std::vector<float> weight_scales;
  fill_inputs(shape, &activation, &activation_scales, &weight, &weight_scales);
  const std::vector<uint16_t> cpu_expected = cpu_oracle(
      shape, activation, activation_scales[0], weight, weight_scales);
  Buffers buffers;
  if (!make_buffers(shape, &buffers) ||
      !upload(shape, activation, activation_scales, weight, weight_scales,
              &buffers)) {
    cleanup(&buffers);
    return false;
  }
  std::vector<uint16_t> control_output;
  std::vector<double> medians;
  medians.reserve(candidates.size());
  bool all_ok = true;
  for (const CandidateId id : candidates) {
    const Candidate current = candidate(id);
    print_resources(current, shape.k);
    double median_ms = 0.0;
    double mad_ms = 0.0;
    double minimum_ms = 0.0;
    double maximum_ms = 0.0;
    if (!measure(current, shape, &buffers, &median_ms, &mad_ms, &minimum_ms,
                 &maximum_ms)) {
      cleanup(&buffers);
      return false;
    }
    medians.push_back(median_ms);
    std::vector<uint16_t> actual(static_cast<std::size_t>(shape.n));
    if (!hip_ok(hipMemcpy(actual.data(), buffers.output,
                          actual.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "copy candidate output") ||
        !hip_ok(hipStreamSynchronize(buffers.stream),
                "synchronize candidate output")) {
      cleanup(&buffers);
      return false;
    }
    const bool cpu_ok =
        compare_outputs(current.name, shape, actual, cpu_expected);
    if (id == CandidateId::Control) {
      control_output = actual;
    } else {
      all_ok = compare_outputs(current.name, shape, actual, control_output) &&
               all_ok;
    }
    all_ok = cpu_ok && all_ok;
    // Repeat the control/candidate output once after the measured section to
    // make deterministic BF16 bit identity explicit in the evidence.
    if (!launch(current, shape, &buffers) ||
        !hip_ok(hipStreamSynchronize(buffers.stream),
                "determinism synchronize")) {
      cleanup(&buffers);
      return false;
    }
    std::vector<uint16_t> repeat(actual.size());
    if (!hip_ok(hipMemcpy(repeat.data(), buffers.output,
                          repeat.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "copy deterministic output")) {
      cleanup(&buffers);
      return false;
    }
    const bool deterministic = repeat == actual;
    std::printf("determinism candidate=%s K=%llu N=%llu bitwise=%s\n",
                current.name, static_cast<unsigned long long>(shape.k),
                static_cast<unsigned long long>(shape.n),
                deterministic ? "PASS" : "FAIL");
    all_ok = deterministic && all_ok;
    const double bytes = static_cast<double>(shape.k) * shape.n +
                         static_cast<double>(shape.n) * sizeof(float) +
                         static_cast<double>(shape.k) + sizeof(float);
    std::printf("result candidate=%s K=%llu N=%llu median_ms=%.6f mad_ms=%.6f "
                "min_ms=%.6f max_ms=%.6f effective_GBps=%.6f\n",
                current.name, static_cast<unsigned long long>(shape.k),
                static_cast<unsigned long long>(shape.n), median_ms, mad_ms,
                minimum_ms, maximum_ms, bytes / median_ms / 1.0e6);
  }
  if (medians.size() != candidates.size() || control_output.empty()) {
    cleanup(&buffers);
    return false;
  }
  const double control_ms = medians.front();
  const auto best = std::min_element(medians.begin(), medians.end());
  const std::size_t best_index =
      static_cast<std::size_t>(best - medians.begin());
  if (weighted_control != nullptr) {
    *weighted_control += control_ms * shape.occurrences;
  }
  if (weighted_best != nullptr) {
    *weighted_best += *best * shape.occurrences;
  }
  std::printf("shape_summary K=%llu N=%llu occurrences=%u roles=%s control=%s "
              "control_ms=%.6f best=%s best_ms=%.6f speedup=%.6f%%\n",
              static_cast<unsigned long long>(shape.k),
              static_cast<unsigned long long>(shape.n), shape.occurrences,
              shape.roles, candidate(candidates[0]).name, control_ms,
              candidate(candidates[best_index]).name, *best,
              (control_ms / *best - 1.0) * 100.0);
  cleanup(&buffers);
  std::printf("cleanup K=%llu N=%llu status=complete\n",
              static_cast<unsigned long long>(shape.k),
              static_cast<unsigned long long>(shape.n));
  return all_ok;
}

} // namespace

int main() {
  if (!hip_ok(hipSetDevice(0), "hipSetDevice")) {
    return EXIT_FAILURE;
  }
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0),
              "hipGetDeviceProperties") ||
      !exact_gfx1030(properties.gcnArchName)) {
    std::fprintf(stderr, "exact gfx1030 required; got %s\n",
                 properties.gcnArchName);
    return EXIT_FAILURE;
  }
  std::printf("target=%s device=0 pci=%04x:%02x:%02x name=%s\n",
              properties.gcnArchName, properties.pciDomainID,
              properties.pciBusID, properties.pciDeviceID, properties.name);
  bool all_ok = all_codes_oracle();
  const std::vector<CandidateId> candidates = {CandidateId::Control,
                                               CandidateId::ActivationShared4,
                                               CandidateId::ActivationShared8};
  for (const CandidateId id : candidates) {
    print_resources(candidate(id), 17408U);
  }
  // Non-aligned N=37 exercises the wave4col32 and wave8col64 tail checks while
  // keeping K=128 valid for the dword8 path.
  const ShapeCase oracle_shape{128U, 37U, 0U, "nonaligned-oracle"};
  double ignored_control = 0.0;
  double ignored_best = 0.0;
  all_ok =
      run_shape(oracle_shape, candidates, &ignored_control, &ignored_best) &&
      all_ok;

  double weighted_control = 0.0;
  double weighted_best = 0.0;
  for (const ShapeCase &shape : kQwen38Fp8Shapes) {
    all_ok = run_shape(shape, candidates, &weighted_control, &weighted_best) &&
             all_ok;
  }
  std::printf("weighted_total exact_qwen38_fp8_tensors=%u "
              "control_ms_per_token=%.6f best_cache_ms_per_token=%.6f "
              "shortfall_ms_per_token=%.6f speedup=%.6fx status=%s\n",
              shape_occurrences(), weighted_control, weighted_best,
              weighted_control - weighted_best,
              weighted_best > 0.0 ? weighted_control / weighted_best : 0.0,
              all_ok ? "PASS" : "FAIL");
  std::printf("summary candidates=%zu warmups=%d measured=%d status=%s\n",
              candidates.size(), kWarmups, kMeasured, all_ok ? "PASS" : "FAIL");
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
