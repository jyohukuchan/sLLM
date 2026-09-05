// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase78-nvfp4-byte-permute-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// 3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70, ggml/src/ggml-cuda/vecdotq.cuh
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

// Phase 78 ID84 production-kernel verification probe.
//
// This probe directly includes the production LDS-FP32-LUT kernels and keeps
// a private direct-decode control for the same ID67/ID73 accumulation order.
// It is prepared for a later GPU run on either exact target; this source is
// compile-only in the current Phase 78 handoff.
//
// Suggested compile-only commands (no GPU execution):
//   amdclang++ -D__HIP_ROCclr__=1 -O3 -DNDEBUG -std=gnu++17 \
//     --offload-arch=gfx1030 \
//     -mcode-object-version=6 -mno-wavefrontsize64 -x hip -c this-file \
//     -o phase78-id84-probe-gfx1030.o
//   amdclang++ -D__HIP_ROCclr__=1 -O3 -DNDEBUG -std=gnu++17 \
//     --offload-arch=gfx1201 \
//     -mcode-object-version=6 -mno-wavefrontsize64 -x hip -c this-file \
//     -o phase78-id84-probe-gfx1201.o
//
// A future run requires an explicit --target gfx1030 or --target gfx1201.
// gfx1030 routes the artifact wide/down pair through ID73 activation-shared;
// gfx1201 routes it through ID67 wave4/col32.  The tiny K48/N37 case always
// exercises ID67 and its 32-column tail.

#include "../src/nvfp4_decode_scale_lut.inc"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

namespace {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kWave = 32U;
constexpr uint32_t kWaves = kThreads / kWave;
constexpr uint32_t kColumnsPerWave = 4U;
constexpr uint32_t kColumnsPerWorkgroup = kWaves * kColumnsPerWave;
constexpr uint32_t kWarmups = 3U;
constexpr uint32_t kMeasured = 10U;
constexpr uint64_t kMaxK = UINT64_C(17408);
constexpr uint32_t kScaleCodes = 256U;

// Actual model scale planes are unsigned-positive E4M3FN values.  The full
// 254-code finite check below remains separate from these runtime inputs.
constexpr std::array<uint8_t, 16> kPositiveScaleCodes = {
    0x01U, 0x08U, 0x10U, 0x18U, 0x20U, 0x28U, 0x30U, 0x38U,
    0x40U, 0x48U, 0x50U, 0x58U, 0x60U, 0x68U, 0x70U, 0x7eU,
};

struct Shape final {
  uint64_t k;
  uint64_t n;
  uint32_t occurrences;
  const char *label;
};

constexpr std::array<Shape, 2> kArtifactShapes = {{
    {5120U, 17408U, 112U, "artifact-wide"},
    {17408U, 5120U, 56U, "artifact-down"},
}};

struct Options final {
  std::string_view target;
};

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "hip operation=%s status=%s (%s)\n", operation,
               hipGetErrorName(status), hipGetErrorString(status));
  return false;
}

bool exact_target(const char *const actual, const std::string_view expected) {
  if (actual == nullptr)
    return false;
  const std::string_view value(actual);
  return value == expected ||
         (value.size() > expected.size() &&
          value.compare(0U, expected.size(), expected) == 0 &&
          value[expected.size()] == ':');
}

// Private direct control helpers.  Their names are deliberately distinct from
// both matmul_kernel.hip.cpp and the production include.
struct ControlScaledPacks final {
  uint32_t even;
  uint32_t odd;
};

__device__ __forceinline__ ControlScaledPacks
sllm_id84_probe_scaled_packs(const uint32_t packed) {
  constexpr uint32_t table_0_3 = UINT32_C(0x03020100);
  constexpr uint32_t table_4_7 = UINT32_C(0x0c080604);
  constexpr uint32_t table_8_11 = UINT32_C(0xfdfeff00);
  constexpr uint32_t table_12_15 = UINT32_C(0xf4f8fafc);
  constexpr uint32_t low_mask = UINT32_C(0x07070707);
  constexpr uint32_t identity = UINT32_C(0x03020100);
  constexpr uint32_t sign_mask = UINT32_C(0x08080808);
  const uint32_t even_indices = packed;
  const uint32_t odd_indices = packed >> 4U;
  const uint32_t even_low =
      __builtin_amdgcn_perm(table_4_7, table_0_3, even_indices & low_mask);
  const uint32_t odd_low =
      __builtin_amdgcn_perm(table_4_7, table_0_3, odd_indices & low_mask);
  const uint32_t even_high =
      __builtin_amdgcn_perm(table_12_15, table_8_11, even_indices & low_mask);
  const uint32_t odd_high =
      __builtin_amdgcn_perm(table_12_15, table_8_11, odd_indices & low_mask);
  const uint32_t even_select = identity | ((even_indices & sign_mask) >> 1U);
  const uint32_t odd_select = identity | ((odd_indices & sign_mask) >> 1U);
  return {__builtin_amdgcn_perm(even_high, even_low, even_select),
          __builtin_amdgcn_perm(odd_high, odd_low, odd_select)};
}

__device__ __forceinline__ int32_t sllm_id84_probe_dot4(
    const uint32_t lhs, const uint32_t rhs, const int32_t accumulator) {
#if __has_builtin(__builtin_amdgcn_sdot4)
  return __builtin_amdgcn_sdot4(lhs, rhs, accumulator, false);
#else
  int32_t result = accumulator;
#pragma unroll
  for (uint32_t lane = 0U; lane < 4U; ++lane) {
    result += static_cast<int8_t>(lhs >> (lane * 8U)) *
              static_cast<int8_t>(rhs >> (lane * 8U));
  }
  return result;
#endif
}

__device__ __forceinline__ float
sllm_id84_probe_e4m3fn_to_float(const uint8_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & UINT8_C(0x80)) << 24U;
  const uint32_t magnitude = static_cast<uint32_t>(bits) & UINT32_C(0x7f);
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & UINT32_C(0x07);
  if (exponent == 0U) {
    if (mantissa == 0U)
      return __uint_as_float(sign);
    const float value = static_cast<float>(mantissa) * 0x1p-9F;
    return __uint_as_float(__float_as_uint(value) | sign);
  }
  if (magnitude == UINT32_C(0x7f)) {
    return __uint_as_float(sign | UINT32_C(0x7fc00000));
  }
  return __uint_as_float(sign | ((exponent + 120U) << 23U) | (mantissa << 20U));
}

__device__ __forceinline__ uint16_t
sllm_id84_probe_bf16_rne(const float value) {
  const uint32_t bits = __float_as_uint(value);
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

__device__ __forceinline__ bool sllm_id84_probe_id67_shape(const uint64_t m,
                                                           const uint64_t k,
                                                           const uint64_t n) {
  return m == 1U && k != 0U && (k % UINT64_C(16)) == 0U && k <= kMaxK &&
         n != 0U;
}

__device__ __forceinline__ bool sllm_id84_probe_id73_shape(const uint64_t m,
                                                           const uint64_t k,
                                                           const uint64_t n) {
  return m == 1U && ((k == UINT64_C(5120) && n == UINT64_C(17408)) ||
                     (k == UINT64_C(17408) && n == UINT64_C(5120)));
}

__device__ __forceinline__ void sllm_id84_probe_id67_body(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_scales, const uint8_t *const packed_weight,
    const uint8_t *const weight_scales, const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  if (!sllm_id84_probe_id67_shape(m, k, n))
    return;
  const uint32_t lane = threadIdx.x & (kWave - 1U);
  const uint32_t wave = threadIdx.x / kWave;
  const uint64_t blocks_per_row = k / UINT64_C(16);
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * kColumnsPerWorkgroup +
      static_cast<uint64_t>(wave) * kColumnsPerWave;
  const uint64_t packed_row_bytes = k / UINT64_C(2);
  float accumulators[kColumnsPerWave] = {};

  for (uint64_t block = lane; block < blocks_per_row; block += kWave) {
    const uint64_t packed_offset = block * UINT64_C(8);
    const auto *const activation_words =
        reinterpret_cast<const uint32_t *>(packed_activation + packed_offset);
    const ControlScaledPacks activation_pack0 = sllm_id84_probe_scaled_packs(
        __builtin_nontemporal_load(activation_words + 0U));
    const ControlScaledPacks activation_pack1 = sllm_id84_probe_scaled_packs(
        __builtin_nontemporal_load(activation_words + 1U));
    const float activation_scale = sllm_id84_probe_e4m3fn_to_float(
        __builtin_nontemporal_load(activation_scales + block));
#pragma unroll
    for (uint32_t column_offset = 0U; column_offset < kColumnsPerWave;
         ++column_offset) {
      const uint64_t column = column_base + column_offset;
      if (column >= n)
        continue;
      const auto *const weight_words = reinterpret_cast<const uint32_t *>(
          packed_weight + column * packed_row_bytes + packed_offset);
      const ControlScaledPacks weight_pack0 = sllm_id84_probe_scaled_packs(
          __builtin_nontemporal_load(weight_words + 0U));
      const ControlScaledPacks weight_pack1 = sllm_id84_probe_scaled_packs(
          __builtin_nontemporal_load(weight_words + 1U));
      int32_t block_sum = 0;
      block_sum = sllm_id84_probe_dot4(activation_pack0.even, weight_pack0.even,
                                       block_sum);
      block_sum = sllm_id84_probe_dot4(activation_pack0.odd, weight_pack0.odd,
                                       block_sum);
      block_sum = sllm_id84_probe_dot4(activation_pack1.even, weight_pack1.even,
                                       block_sum);
      block_sum = sllm_id84_probe_dot4(activation_pack1.odd, weight_pack1.odd,
                                       block_sum);
      const float weight_scale =
          sllm_id84_probe_e4m3fn_to_float(__builtin_nontemporal_load(
              weight_scales + column * blocks_per_row + block));
      accumulators[column_offset] += static_cast<float>(block_sum) * 0.25F *
                                     activation_scale * weight_scale;
    }
  }

#pragma unroll
  for (uint32_t column_offset = 0U; column_offset < kColumnsPerWave;
       ++column_offset) {
#pragma unroll
    for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U) {
      accumulators[column_offset] +=
          __shfl_down(accumulators[column_offset], offset, kWave);
    }
    const uint64_t column = column_base + column_offset;
    if (lane == 0U && column < n) {
      output[column] = sllm_id84_probe_bf16_rne(accumulators[column_offset] *
                                                weight_tensor_scale[0] *
                                                input_tensor_scale[0]);
    }
  }
}

__device__ __forceinline__ void sllm_id84_probe_id73_body(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_scales, const uint8_t *const packed_weight,
    const uint8_t *const weight_scales, const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  if (!sllm_id84_probe_id73_shape(m, k, n))
    return;
  const uint32_t lane = threadIdx.x & (kWave - 1U);
  const uint32_t wave = threadIdx.x / kWave;
  const uint64_t blocks_per_row = k / UINT64_C(16);
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * kColumnsPerWorkgroup +
      static_cast<uint64_t>(wave) * kColumnsPerWave;
  const uint64_t packed_row_bytes = k / UINT64_C(2);

  extern __shared__ uint32_t shared[];
  int32_t *const activation_packs = reinterpret_cast<int32_t *>(shared);
  float *const activation_scale_values =
      reinterpret_cast<float *>(shared + blocks_per_row * UINT64_C(4));
  for (uint64_t block = threadIdx.x; block < blocks_per_row;
       block += kThreads) {
    const auto *const activation_words = reinterpret_cast<const uint32_t *>(
        packed_activation + block * UINT64_C(8));
    const ControlScaledPacks first = sllm_id84_probe_scaled_packs(
        __builtin_nontemporal_load(activation_words + 0U));
    const ControlScaledPacks second = sllm_id84_probe_scaled_packs(
        __builtin_nontemporal_load(activation_words + 1U));
    activation_packs[block * UINT64_C(4) + UINT64_C(0)] =
        static_cast<int32_t>(first.even);
    activation_packs[block * UINT64_C(4) + UINT64_C(1)] =
        static_cast<int32_t>(first.odd);
    activation_packs[block * UINT64_C(4) + UINT64_C(2)] =
        static_cast<int32_t>(second.even);
    activation_packs[block * UINT64_C(4) + UINT64_C(3)] =
        static_cast<int32_t>(second.odd);
    activation_scale_values[block] = sllm_id84_probe_e4m3fn_to_float(
        __builtin_nontemporal_load(activation_scales + block));
  }
  __syncthreads();

  float accumulators[kColumnsPerWave] = {};
  for (uint64_t block = lane; block < blocks_per_row; block += kWave) {
    const ControlScaledPacks activation_pack0 = {
        static_cast<uint32_t>(
            activation_packs[block * UINT64_C(4) + UINT64_C(0)]),
        static_cast<uint32_t>(
            activation_packs[block * UINT64_C(4) + UINT64_C(1)])};
    const ControlScaledPacks activation_pack1 = {
        static_cast<uint32_t>(
            activation_packs[block * UINT64_C(4) + UINT64_C(2)]),
        static_cast<uint32_t>(
            activation_packs[block * UINT64_C(4) + UINT64_C(3)])};
    const float activation_scale = activation_scale_values[block];
#pragma unroll
    for (uint32_t column_offset = 0U; column_offset < kColumnsPerWave;
         ++column_offset) {
      const uint64_t column = column_base + column_offset;
      if (column >= n)
        continue;
      const auto *const weight_words = reinterpret_cast<const uint32_t *>(
          packed_weight + column * packed_row_bytes + block * UINT64_C(8));
      const ControlScaledPacks weight_pack0 = sllm_id84_probe_scaled_packs(
          __builtin_nontemporal_load(weight_words + 0U));
      const ControlScaledPacks weight_pack1 = sllm_id84_probe_scaled_packs(
          __builtin_nontemporal_load(weight_words + 1U));
      int32_t block_sum = 0;
      block_sum = sllm_id84_probe_dot4(activation_pack0.even, weight_pack0.even,
                                       block_sum);
      block_sum = sllm_id84_probe_dot4(activation_pack0.odd, weight_pack0.odd,
                                       block_sum);
      block_sum = sllm_id84_probe_dot4(activation_pack1.even, weight_pack1.even,
                                       block_sum);
      block_sum = sllm_id84_probe_dot4(activation_pack1.odd, weight_pack1.odd,
                                       block_sum);
      const float weight_scale =
          sllm_id84_probe_e4m3fn_to_float(__builtin_nontemporal_load(
              weight_scales + column * blocks_per_row + block));
      accumulators[column_offset] += static_cast<float>(block_sum) * 0.25F *
                                     activation_scale * weight_scale;
    }
  }

#pragma unroll
  for (uint32_t column_offset = 0U; column_offset < kColumnsPerWave;
       ++column_offset) {
#pragma unroll
    for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U) {
      accumulators[column_offset] +=
          __shfl_down(accumulators[column_offset], offset, kWave);
    }
    const uint64_t column = column_base + column_offset;
    if (lane == 0U && column < n) {
      output[column] = sllm_id84_probe_bf16_rne(accumulators[column_offset] *
                                                weight_tensor_scale[0] *
                                                input_tensor_scale[0]);
    }
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_id84_probe_id67_control(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_scales, const uint8_t *const packed_weight,
    const uint8_t *const weight_scales, const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  sllm_id84_probe_id67_body(packed_activation, activation_scales, packed_weight,
                            weight_scales, weight_tensor_scale,
                            input_tensor_scale, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_id84_probe_id73_control(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_scales, const uint8_t *const packed_weight,
    const uint8_t *const weight_scales, const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  sllm_id84_probe_id73_body(packed_activation, activation_scales, packed_weight,
                            weight_scales, weight_tensor_scale,
                            input_tensor_scale, output, m, k, n);
}

float host_e2m1(const uint8_t code) {
  constexpr std::array<float, 8> values = {0.0F, 0.5F, 1.0F, 1.5F,
                                           2.0F, 3.0F, 4.0F, 6.0F};
  const float value = values[code & 7U];
  return (code & 8U) == 0U ? value : -value;
}

float host_e4m3fn(const uint8_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x80U) << 24U;
  const uint8_t magnitude = bits & 0x7fU;
  const uint8_t exponent = magnitude >> 3U;
  const uint8_t mantissa = magnitude & 7U;
  if (exponent == 0U) {
    float result = static_cast<float>(mantissa) * 0x1p-9F;
    uint32_t result_bits = 0U;
    std::memcpy(&result_bits, &result, sizeof(result_bits));
    result_bits |= sign;
    std::memcpy(&result, &result_bits, sizeof(result));
    return result;
  }
  if (magnitude == 0x7fU)
    return std::numeric_limits<float>::quiet_NaN();
  const uint32_t result_bits = sign |
                               (static_cast<uint32_t>(exponent + 120U) << 23U) |
                               (static_cast<uint32_t>(mantissa) << 20U);
  float result = 0.0F;
  std::memcpy(&result, &result_bits, sizeof(result));
  return result;
}

uint16_t host_e4m3fn_to_fp16_bits(const uint8_t bits) {
  const uint16_t sign = static_cast<uint16_t>(bits & 0x80U) << 8U;
  const uint8_t magnitude = bits & 0x7fU;
  const uint8_t exponent = magnitude >> 3U;
  const uint8_t mantissa = magnitude & 7U;
  if (exponent == 0U) {
    constexpr uint16_t subnormal[8] = {0x0000U, 0x1800U, 0x1c00U, 0x1e00U,
                                       0x2000U, 0x2100U, 0x2200U, 0x2300U};
    return static_cast<uint16_t>(sign | subnormal[mantissa]);
  }
  if (magnitude == 0x7fU)
    return static_cast<uint16_t>(sign | 0x7e00U);
  return static_cast<uint16_t>(sign | (exponent + 8U) << 10U |
                               static_cast<uint16_t>(mantissa) << 7U);
}

float host_fp16_bits_to_float(const uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16U;
  const uint32_t exponent = (bits >> 10U) & 0x1fU;
  const uint32_t mantissa = bits & 0x03ffU;
  if (exponent == 0U) {
    if (mantissa == 0U) {
      float result = 0.0F;
      uint32_t result_bits = sign;
      std::memcpy(&result, &result_bits, sizeof(result));
      return result;
    }
    uint32_t normalized = mantissa;
    int32_t exponent_value = -14;
    while ((normalized & 0x0400U) == 0U) {
      normalized <<= 1U;
      --exponent_value;
    }
    normalized &= 0x03ffU;
    const uint32_t result_bits =
        sign | (static_cast<uint32_t>(exponent_value + 127) << 23U) |
        (normalized << 13U);
    float result = 0.0F;
    std::memcpy(&result, &result_bits, sizeof(result));
    return result;
  }
  const uint32_t result_bits =
      exponent == 0x1fU ? sign | 0x7f800000U | (mantissa << 13U)
                        : sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
  float result = 0.0F;
  std::memcpy(&result, &result_bits, sizeof(result));
  return result;
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

void fill_inputs(const Shape &shape, std::vector<uint8_t> *const activation,
                 std::vector<uint8_t> *const activation_scales,
                 std::vector<uint8_t> *const weight,
                 std::vector<uint8_t> *const weight_scales) {
  const uint64_t blocks = shape.k / UINT64_C(16);
  activation->resize(static_cast<size_t>(shape.k / UINT64_C(2)));
  activation_scales->resize(static_cast<size_t>(blocks));
  weight->resize(static_cast<size_t>(shape.k * shape.n / UINT64_C(2)));
  weight_scales->resize(static_cast<size_t>(shape.n * blocks));
  for (uint64_t byte = 0U; byte < shape.k / UINT64_C(2); ++byte) {
    const uint8_t low = static_cast<uint8_t>((byte * 5U + 3U) & 0x0fU);
    const uint8_t high = static_cast<uint8_t>((byte * 11U + 7U) & 0x0fU);
    (*activation)[static_cast<size_t>(byte)] =
        static_cast<uint8_t>(low | (high << 4U));
  }
  for (uint64_t block = 0U; block < blocks; ++block)
    (*activation_scales)[static_cast<size_t>(block)] =
        kPositiveScaleCodes[(block * 3U + 5U) & 15U];
  for (uint64_t byte = 0U; byte < shape.k * shape.n / UINT64_C(2); ++byte) {
    const uint8_t low = static_cast<uint8_t>((byte * 7U + 9U) & 0x0fU);
    const uint8_t high = static_cast<uint8_t>((byte * 13U + 1U) & 0x0fU);
    (*weight)[static_cast<size_t>(byte)] =
        static_cast<uint8_t>(low | (high << 4U));
  }
  for (uint64_t index = 0U; index < shape.n * blocks; ++index)
    (*weight_scales)[static_cast<size_t>(index)] =
        kPositiveScaleCodes[(index * 5U + 9U) & 15U];
}

std::vector<uint16_t> cpu_oracle(const Shape &shape,
                                 const std::vector<uint8_t> &activation,
                                 const std::vector<uint8_t> &activation_scales,
                                 const std::vector<uint8_t> &weight,
                                 const std::vector<uint8_t> &weight_scales) {
  const uint64_t blocks = shape.k / UINT64_C(16);
  std::vector<uint16_t> result(static_cast<size_t>(shape.n), 0U);
  for (uint64_t column = 0U; column < shape.n; ++column) {
    float accumulator = 0.0F;
    for (uint64_t block = 0U; block < blocks; ++block) {
      int32_t block_sum = 0;
      for (uint64_t index = 0U; index < 16U; ++index) {
        const uint64_t packed_index = block * UINT64_C(8) + index / 2U;
        const uint8_t a_byte = activation[static_cast<size_t>(packed_index)];
        const uint8_t w_byte =
            weight[static_cast<size_t>(column * (shape.k / 2U) + packed_index)];
        const uint8_t a_code =
            (index & 1U) == 0U ? a_byte & 0x0fU : a_byte >> 4U;
        const uint8_t w_code =
            (index & 1U) == 0U ? w_byte & 0x0fU : w_byte >> 4U;
        block_sum += static_cast<int32_t>(host_e2m1(a_code) * 2.0F) *
                     static_cast<int32_t>(host_e2m1(w_code) * 2.0F);
      }
      const float activation_scale =
          host_e4m3fn(activation_scales[static_cast<size_t>(block)]);
      const float weight_scale = host_e4m3fn(
          weight_scales[static_cast<size_t>(column * blocks + block)]);
      accumulator += static_cast<float>(block_sum) * 0.25F * activation_scale *
                     weight_scale;
    }
    result[static_cast<size_t>(column)] =
        host_bf16_rne(accumulator * 0.75F * 1.125F);
  }
  return result;
}

struct Buffers final {
  uint8_t *activation = nullptr;
  uint8_t *activation_scales = nullptr;
  uint8_t *weight = nullptr;
  uint8_t *weight_scales = nullptr;
  float *weight_tensor_scale = nullptr;
  float *input_tensor_scale = nullptr;
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
  if (buffers->input_tensor_scale != nullptr)
    (void)hipFree(buffers->input_tensor_scale);
  if (buffers->weight_tensor_scale != nullptr)
    (void)hipFree(buffers->weight_tensor_scale);
  if (buffers->weight_scales != nullptr)
    (void)hipFree(buffers->weight_scales);
  if (buffers->weight != nullptr)
    (void)hipFree(buffers->weight);
  if (buffers->activation_scales != nullptr)
    (void)hipFree(buffers->activation_scales);
  if (buffers->activation != nullptr)
    (void)hipFree(buffers->activation);
  *buffers = {};
}

bool make_buffers(const Shape &shape, Buffers *const buffers) {
  if (buffers == nullptr || shape.k == 0U || shape.n == 0U ||
      (shape.k % UINT64_C(16)) != 0U || shape.k > UINT64_MAX / shape.n)
    return false;
  const uint64_t weight_bytes_u64 = shape.k * shape.n / UINT64_C(2);
  const uint64_t weight_scale_bytes_u64 = shape.n * (shape.k / UINT64_C(16));
  if (weight_bytes_u64 > SIZE_MAX || weight_scale_bytes_u64 > SIZE_MAX ||
      shape.n > SIZE_MAX / sizeof(uint16_t))
    return false;
  const size_t weight_bytes = static_cast<size_t>(weight_bytes_u64);
  const size_t weight_scale_bytes = static_cast<size_t>(weight_scale_bytes_u64);
  const size_t output_bytes = static_cast<size_t>(shape.n * sizeof(uint16_t));
  return hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->activation),
                          static_cast<size_t>(shape.k / UINT64_C(2))),
                "malloc activation") &&
         hip_ok(
             hipMalloc(reinterpret_cast<void **>(&buffers->activation_scales),
                       static_cast<size_t>(shape.k / UINT64_C(16))),
             "malloc activation scales") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->weight),
                          weight_bytes),
                "malloc weight") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->weight_scales),
                          weight_scale_bytes),
                "malloc weight scales") &&
         hip_ok(
             hipMalloc(reinterpret_cast<void **>(&buffers->weight_tensor_scale),
                       sizeof(float)),
             "malloc weight tensor scale") &&
         hip_ok(
             hipMalloc(reinterpret_cast<void **>(&buffers->input_tensor_scale),
                       sizeof(float)),
             "malloc input tensor scale") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->output),
                          output_bytes),
                "malloc output") &&
         hip_ok(hipStreamCreate(&buffers->stream), "create stream") &&
         hip_ok(hipEventCreate(&buffers->start), "create start event") &&
         hip_ok(hipEventCreate(&buffers->stop), "create stop event");
}

bool upload_inputs(const Shape &shape, const std::vector<uint8_t> &activation,
                   const std::vector<uint8_t> &activation_scales,
                   const std::vector<uint8_t> &weight,
                   const std::vector<uint8_t> &weight_scales,
                   Buffers *const buffers) {
  const float weight_tensor_scale = 0.75F;
  const float input_tensor_scale = 1.125F;
  return hip_ok(hipMemcpy(buffers->activation, activation.data(),
                          activation.size(), hipMemcpyHostToDevice),
                "copy activation") &&
         hip_ok(hipMemcpy(buffers->activation_scales, activation_scales.data(),
                          activation_scales.size(), hipMemcpyHostToDevice),
                "copy activation scales") &&
         hip_ok(hipMemcpy(buffers->weight, weight.data(), weight.size(),
                          hipMemcpyHostToDevice),
                "copy weight") &&
         hip_ok(hipMemcpy(buffers->weight_scales, weight_scales.data(),
                          weight_scales.size(), hipMemcpyHostToDevice),
                "copy weight scales") &&
         hip_ok(hipMemcpy(buffers->weight_tensor_scale, &weight_tensor_scale,
                          sizeof(float), hipMemcpyHostToDevice),
                "copy weight tensor scale") &&
         hip_ok(hipMemcpy(buffers->input_tensor_scale, &input_tensor_scale,
                          sizeof(float), hipMemcpyHostToDevice),
                "copy input tensor scale") &&
         hip_ok(hipMemset(buffers->output, 0,
                          static_cast<size_t>(shape.n * sizeof(uint16_t))),
                "clear output");
}

size_t activation_shared_lds_bytes(const uint64_t k) {
  return static_cast<size_t>((k / UINT64_C(16)) * UINT64_C(5) *
                             sizeof(uint32_t));
}

bool launch_control(const Shape &shape, const bool activation_shared,
                    Buffers *const buffers) {
  const uint64_t grid_u64 =
      (shape.n + kColumnsPerWorkgroup - 1U) / kColumnsPerWorkgroup;
  if (grid_u64 == 0U || grid_u64 > UINT32_MAX)
    return false;
  const dim3 grid(static_cast<uint32_t>(grid_u64));
  const dim3 block(kThreads);
  const size_t dynamic_lds =
      activation_shared ? activation_shared_lds_bytes(shape.k) : 0U;
  if (activation_shared) {
    hipLaunchKernelGGL(
        sllm_id84_probe_id73_control, grid, block, dynamic_lds, buffers->stream,
        buffers->activation, buffers->activation_scales, buffers->weight,
        buffers->weight_scales, buffers->weight_tensor_scale,
        buffers->input_tensor_scale, buffers->output, 1U, shape.k, shape.n);
  } else {
    hipLaunchKernelGGL(
        sllm_id84_probe_id67_control, grid, block, 0U, buffers->stream,
        buffers->activation, buffers->activation_scales, buffers->weight,
        buffers->weight_scales, buffers->weight_tensor_scale,
        buffers->input_tensor_scale, buffers->output, 1U, shape.k, shape.n);
  }
  return hipGetLastError() == hipSuccess;
}

bool launch_candidate(const Shape &shape, const bool activation_shared,
                      Buffers *const buffers) {
  const uint64_t grid_u64 =
      (shape.n + kColumnsPerWorkgroup - 1U) / kColumnsPerWorkgroup;
  if (grid_u64 == 0U || grid_u64 > UINT32_MAX)
    return false;
  const dim3 grid(static_cast<uint32_t>(grid_u64));
  const dim3 block(kThreads);
  const size_t dynamic_lds =
      activation_shared ? activation_shared_lds_bytes(shape.k) : 0U;
  if (activation_shared) {
    hipLaunchKernelGGL(
        sllm_matmul_nvfp4_w4a4_decode_dp4a_activation_shared_lds_f32_lut_v1,
        grid, block, dynamic_lds, buffers->stream, buffers->activation,
        buffers->activation_scales, buffers->weight, buffers->weight_scales,
        buffers->weight_tensor_scale, buffers->input_tensor_scale,
        buffers->output, 1U, shape.k, shape.n);
  } else {
    hipLaunchKernelGGL(
        sllm_matmul_nvfp4_w4a4_decode_dp4a_wave4col32_lds_f32_lut_v1, grid,
        block, 0U, buffers->stream, buffers->activation,
        buffers->activation_scales, buffers->weight, buffers->weight_scales,
        buffers->weight_tensor_scale, buffers->input_tensor_scale,
        buffers->output, 1U, shape.k, shape.n);
  }
  return hipGetLastError() == hipSuccess;
}

bool launch_constant_candidate(const Shape &shape, const bool activation_shared,
                               Buffers *const buffers) {
  const uint64_t grid_u64 =
      (shape.n + kColumnsPerWorkgroup - 1U) / kColumnsPerWorkgroup;
  if (grid_u64 == 0U || grid_u64 > UINT32_MAX)
    return false;
  const dim3 grid(static_cast<uint32_t>(grid_u64));
  const dim3 block(kThreads);
  const size_t dynamic_lds =
      activation_shared ? activation_shared_lds_bytes(shape.k) : 0U;
  if (activation_shared) {
    hipLaunchKernelGGL(
        sllm_matmul_nvfp4_w4a4_decode_dp4a_activation_shared_lds_f32_const_lut_v1,
        grid, block, dynamic_lds, buffers->stream, buffers->activation,
        buffers->activation_scales, buffers->weight, buffers->weight_scales,
        buffers->weight_tensor_scale, buffers->input_tensor_scale,
        buffers->output, 1U, shape.k, shape.n);
  } else {
    hipLaunchKernelGGL(
        sllm_matmul_nvfp4_w4a4_decode_dp4a_wave4col32_lds_f32_const_lut_v1,
        grid, block, 0U, buffers->stream, buffers->activation,
        buffers->activation_scales, buffers->weight, buffers->weight_scales,
        buffers->weight_tensor_scale, buffers->input_tensor_scale,
        buffers->output, 1U, shape.k, shape.n);
  }
  return hipGetLastError() == hipSuccess;
}

bool capture(const Shape &shape, Buffers *const buffers,
             std::vector<uint16_t> *const output) {
  output->resize(static_cast<size_t>(shape.n));
  return hip_ok(hipMemcpy(output->data(), buffers->output,
                          output->size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "copy output");
}

size_t mismatch_count(const std::vector<uint16_t> &left,
                      const std::vector<uint16_t> &right) {
  const size_t count = std::min(left.size(), right.size());
  size_t mismatches = left.size() == right.size() ? 0U : 1U;
  for (size_t index = 0U; index < count; ++index)
    if (left[index] != right[index])
      ++mismatches;
  return mismatches;
}

bool run_scale_contract_oracle() {
  size_t finite_mismatches = 0U;
  size_t positive_input_violations = 0U;
  for (uint32_t code = 0U; code < kScaleCodes; ++code) {
    const uint8_t bits = static_cast<uint8_t>(code);
    if ((bits & 0x7fU) != 0x7fU) {
      const float direct = host_e4m3fn(bits);
      const float via_fp16 =
          host_fp16_bits_to_float(host_e4m3fn_to_fp16_bits(bits));
      uint32_t direct_bits = 0U;
      uint32_t via_bits = 0U;
      std::memcpy(&direct_bits, &direct, sizeof(direct_bits));
      std::memcpy(&via_bits, &via_fp16, sizeof(via_bits));
      if (direct_bits != via_bits)
        ++finite_mismatches;
    }
  }
  for (const uint8_t code : kPositiveScaleCodes)
    if ((code & 0x80U) != 0U || (code & 0x7fU) == 0x7fU)
      ++positive_input_violations;
  const bool nan_7f = std::isnan(host_e4m3fn(0x7fU));
  const bool nan_ff = std::isnan(host_e4m3fn(0xffU));
  std::printf("scale_contract finite_codes=254 finite_mismatches=%zu "
              "invalid_codes=0x7f,0xff nan=%s positive_input_violations=%zu "
              "status=%s\n",
              finite_mismatches, nan_7f && nan_ff ? "PASS" : "FAIL",
              positive_input_violations,
              finite_mismatches == 0U && nan_7f && nan_ff &&
                      positive_input_violations == 0U
                  ? "PASS"
                  : "FAIL");
  return finite_mismatches == 0U && nan_7f && nan_ff &&
         positive_input_violations == 0U;
}

bool run_tiny_oracle() {
  const Shape shape{48U, 37U, 0U, "tiny-k48-n37"};
  std::vector<uint8_t> activation;
  std::vector<uint8_t> activation_scales;
  std::vector<uint8_t> weight;
  std::vector<uint8_t> weight_scales;
  fill_inputs(shape, &activation, &activation_scales, &weight, &weight_scales);
  Buffers buffers;
  if (!make_buffers(shape, &buffers) ||
      !upload_inputs(shape, activation, activation_scales, weight,
                     weight_scales, &buffers)) {
    cleanup(&buffers);
    return false;
  }
  const std::vector<uint16_t> expected =
      cpu_oracle(shape, activation, activation_scales, weight, weight_scales);
  if (!launch_control(shape, false, &buffers) ||
      !hip_ok(hipStreamSynchronize(buffers.stream), "tiny control sync")) {
    cleanup(&buffers);
    return false;
  }
  std::vector<uint16_t> control;
  if (!capture(shape, &buffers, &control) ||
      !launch_candidate(shape, false, &buffers) ||
      !hip_ok(hipStreamSynchronize(buffers.stream), "tiny candidate sync")) {
    cleanup(&buffers);
    return false;
  }
  std::vector<uint16_t> candidate;
  const bool captured = capture(shape, &buffers, &candidate);
  std::vector<uint16_t> constant_candidate;
  const bool constant_captured =
      launch_constant_candidate(shape, false, &buffers) &&
      hip_ok(hipStreamSynchronize(buffers.stream), "tiny constant sync") &&
      capture(shape, &buffers, &constant_candidate);
  const size_t control_mismatches = mismatch_count(control, expected);
  const size_t candidate_mismatches = mismatch_count(candidate, expected);
  const size_t constant_mismatches =
      mismatch_count(constant_candidate, expected);
  const size_t direct_mismatches = mismatch_count(candidate, control);
  const size_t constant_vs_candidate =
      mismatch_count(constant_candidate, candidate);
  std::printf(
      "tiny_oracle K=48 N=37 blocks=3 tail_columns=5 control_mismatches=%zu "
      "candidate_mismatches=%zu constant_mismatches=%zu "
      "candidate_vs_control=%zu constant_vs_candidate=%zu status=%s\n",
      control_mismatches, candidate_mismatches, constant_mismatches,
      direct_mismatches, constant_vs_candidate,
      captured && control_mismatches == 0U && candidate_mismatches == 0U &&
              constant_captured && constant_mismatches == 0U &&
              direct_mismatches == 0U && constant_vs_candidate == 0U
          ? "PASS"
          : "FAIL");
  cleanup(&buffers);
  return captured && constant_captured && control_mismatches == 0U &&
         candidate_mismatches == 0U && constant_mismatches == 0U &&
         direct_mismatches == 0U && constant_vs_candidate == 0U;
}

struct Timing final {
  double median_us = 0.0;
  double mad_us = 0.0;
};

bool measure(const Shape &shape, const bool activation_shared,
             const bool candidate, const bool constant_lut,
             Buffers *const buffers, Timing *const timing) {
  for (uint32_t warmup = 0U; warmup < kWarmups; ++warmup) {
    const bool launched =
        candidate ? (constant_lut
                         ? launch_constant_candidate(shape, activation_shared,
                                                     buffers)
                         : launch_candidate(shape, activation_shared, buffers))
                  : launch_control(shape, activation_shared, buffers);
    if (!launched ||
        !hip_ok(hipStreamSynchronize(buffers->stream), "warmup sync"))
      return false;
  }
  std::array<float, kMeasured> samples{};
  for (uint32_t iteration = 0U; iteration < kMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(buffers->start, buffers->stream), "event start"))
      return false;
    const bool launched =
        candidate ? (constant_lut
                         ? launch_constant_candidate(shape, activation_shared,
                                                     buffers)
                         : launch_candidate(shape, activation_shared, buffers))
                  : launch_control(shape, activation_shared, buffers);
    if (!launched ||
        !hip_ok(hipEventRecord(buffers->stop, buffers->stream), "event stop") ||
        !hip_ok(hipEventSynchronize(buffers->stop), "event sync"))
      return false;
    float elapsed_ms = 0.0F;
    if (!hip_ok(hipEventElapsedTime(&elapsed_ms, buffers->start, buffers->stop),
                "event elapsed"))
      return false;
    samples[static_cast<size_t>(iteration)] = elapsed_ms * 1000.0F;
  }
  std::sort(samples.begin(), samples.end());
  timing->median_us = samples[kMeasured / 2U];
  std::array<float, kMeasured> deviations{};
  for (uint32_t index = 0U; index < kMeasured; ++index)
    deviations[static_cast<size_t>(index)] =
        std::fabs(samples[static_cast<size_t>(index)] - timing->median_us);
  std::sort(deviations.begin(), deviations.end());
  timing->mad_us = deviations[kMeasured / 2U];
  return true;
}

bool run_artifact_shapes(const std::string_view target) {
  const bool activation_shared = target == "gfx1030";
  double weighted_control_us = 0.0;
  double weighted_candidate_us = 0.0;
  double weighted_constant_us = 0.0;
  uint32_t weighted_calls = 0U;
  bool all_ok = true;
  for (const Shape &shape : kArtifactShapes) {
    std::vector<uint8_t> activation;
    std::vector<uint8_t> activation_scales;
    std::vector<uint8_t> weight;
    std::vector<uint8_t> weight_scales;
    fill_inputs(shape, &activation, &activation_scales, &weight,
                &weight_scales);
    Buffers buffers;
    if (!make_buffers(shape, &buffers) ||
        !upload_inputs(shape, activation, activation_scales, weight,
                       weight_scales, &buffers)) {
      cleanup(&buffers);
      return false;
    }
    if (!launch_control(shape, activation_shared, &buffers) ||
        !hip_ok(hipStreamSynchronize(buffers.stream),
                "artifact control sync")) {
      cleanup(&buffers);
      return false;
    }
    std::vector<uint16_t> control;
    if (!capture(shape, &buffers, &control) ||
        !launch_candidate(shape, activation_shared, &buffers) ||
        !hip_ok(hipStreamSynchronize(buffers.stream),
                "artifact candidate sync")) {
      cleanup(&buffers);
      return false;
    }
    std::vector<uint16_t> candidate;
    if (!capture(shape, &buffers, &candidate)) {
      cleanup(&buffers);
      return false;
    }
    if (!launch_constant_candidate(shape, activation_shared, &buffers) ||
        !hip_ok(hipStreamSynchronize(buffers.stream),
                "artifact constant sync")) {
      cleanup(&buffers);
      return false;
    }
    std::vector<uint16_t> constant_candidate;
    if (!capture(shape, &buffers, &constant_candidate)) {
      cleanup(&buffers);
      return false;
    }
    const size_t mismatches = mismatch_count(candidate, control);
    const size_t constant_vs_candidate =
        mismatch_count(constant_candidate, candidate);
    std::printf("artifact_compare target=%.*s shape=%s K=%llu N=%llu "
                "control_mismatches=%zu constant_vs_candidate=%zu status=%s\n",
                static_cast<int>(target.size()), target.data(), shape.label,
                static_cast<unsigned long long>(shape.k),
                static_cast<unsigned long long>(shape.n), mismatches,
                constant_vs_candidate,
                mismatches == 0U && constant_vs_candidate == 0U ? "PASS"
                                                                : "FAIL");
    all_ok = mismatches == 0U && constant_vs_candidate == 0U && all_ok;

    Timing control_timing;
    Timing candidate_timing;
    Timing constant_timing;
    if (!measure(shape, activation_shared, false, false, &buffers,
                 &control_timing) ||
        !measure(shape, activation_shared, true, false, &buffers,
                 &candidate_timing) ||
        !measure(shape, activation_shared, true, true, &buffers,
                 &constant_timing)) {
      cleanup(&buffers);
      return false;
    }
    std::printf(
        "artifact_perf target=%.*s shape=%s path=%s warmups=3 measured=10 "
        "control_median_us=%.3f control_mad_us=%.3f candidate_median_us=%.3f "
        "candidate_mad_us=%.3f constant_median_us=%.3f "
        "constant_mad_us=%.3f speedup=%.6fx constant_speedup=%.6fx\n",
        static_cast<int>(target.size()), target.data(), shape.label,
        activation_shared ? "ID73" : "ID67", control_timing.median_us,
        control_timing.mad_us, candidate_timing.median_us,
        candidate_timing.mad_us, constant_timing.median_us,
        constant_timing.mad_us,
        candidate_timing.median_us > 0.0
            ? control_timing.median_us / candidate_timing.median_us
            : 0.0,
        constant_timing.median_us > 0.0
            ? control_timing.median_us / constant_timing.median_us
            : 0.0);
    weighted_calls += shape.occurrences;
    weighted_control_us +=
        static_cast<double>(shape.occurrences) * control_timing.median_us;
    weighted_candidate_us +=
        static_cast<double>(shape.occurrences) * candidate_timing.median_us;
    weighted_constant_us +=
        static_cast<double>(shape.occurrences) * constant_timing.median_us;
    cleanup(&buffers);
  }
  const double speedup = weighted_candidate_us > 0.0
                             ? weighted_control_us / weighted_candidate_us
                             : 0.0;
  std::printf(
      "weighted_perf target=%.*s weighting=112wide+56down calls=%u "
      "warmups=3 measured=10 control_ms=%.6f candidate_ms=%.6f "
      "constant_ms=%.6f speedup=%.6fx constant_speedup=%.6fx status=%s\n",
      static_cast<int>(target.size()), target.data(), weighted_calls,
      weighted_control_us / 1000.0, weighted_candidate_us / 1000.0,
      weighted_constant_us / 1000.0, speedup,
      weighted_constant_us > 0.0 ? weighted_control_us / weighted_constant_us
                                 : 0.0,
      all_ok ? "PASS" : "FAIL");
  return all_ok;
}

bool parse_options(const int argc, char **const argv, Options *const options) {
  if (options == nullptr)
    return false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      std::printf("usage: %s --target gfx1030|gfx1201\n", argv[0]);
      return false;
    }
    if (argument == "--target" && index + 1 < argc) {
      options->target = argv[++index];
      continue;
    }
    constexpr std::string_view prefix = "--target=";
    if (argument.size() > prefix.size() &&
        argument.compare(0U, prefix.size(), prefix) == 0) {
      options->target = argument.substr(prefix.size());
      continue;
    }
    std::fprintf(stderr, "unknown argument: %s\n", argv[index]);
    return false;
  }
  return options->target == "gfx1030" || options->target == "gfx1201";
}

} // namespace

int main(const int argc, char **const argv) {
  Options options;
  if (!parse_options(argc, argv, &options))
    return EXIT_FAILURE;
  if (!hip_ok(hipSetDevice(0), "set device"))
    return EXIT_FAILURE;
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0),
              "get device properties") ||
      !exact_target(properties.gcnArchName, options.target)) {
    std::fprintf(stderr, "exact target=%.*s required; got %s\n",
                 static_cast<int>(options.target.size()), options.target.data(),
                 properties.gcnArchName);
    return EXIT_FAILURE;
  }
  std::printf("target=%.*s device=0 name=%s\n",
              static_cast<int>(options.target.size()), options.target.data(),
              properties.name);
  const bool all_ok = run_scale_contract_oracle() && run_tiny_oracle() &&
                      run_artifact_shapes(options.target);
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
