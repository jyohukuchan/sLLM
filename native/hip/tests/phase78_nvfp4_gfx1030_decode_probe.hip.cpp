// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase78-nvfp4-byte-permute-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// 3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70, ggml/src/ggml-cuda/vecdotq.cuh
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

// Phase 78 standalone gfx1030 NVFP4 W4A4 decode probe.
//
// The control is the production ID67 DP4A wave4col32 layout.  The candidates
// are intentionally isolated here: a packed-64-bit load variant, a wave8col64
// variant, and a workgroup-shared activation decode variant.  No public
// selector or existing source file is modified by this probe.

#include "low_precision_block_codec.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

struct ScaledPacks final {
  uint32_t even;
  uint32_t odd;
};

// One packed dword contains eight E2M1 nibbles.  The byte-permute tables are
// the exact signed-byte representation of value*2; DP4A then returns four
// times the original E2M1 subtotal.  This is the same arithmetic transform as
// ID67, kept local so the probe can be compiled without the runtime object.
__device__ __forceinline__ ScaledPacks scaled_packs(const uint32_t packed) {
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

__device__ __forceinline__ int32_t dot4(const uint32_t lhs, const uint32_t rhs,
                                        const int32_t accumulator) {
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

__device__ __forceinline__ float e2m1(const uint8_t code) {
  constexpr float positive[8] = {0.0F, 0.5F, 1.0F, 1.5F,
                                 2.0F, 3.0F, 4.0F, 6.0F};
  const float value = positive[code & 7U];
  return (code & 8U) == 0U ? value : -value;
}

__device__ __forceinline__ float e4m3(const uint8_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x80U) << 24U;
  const uint32_t magnitude = bits & 0x7fU;
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  if (exponent == 0U) {
    if (mantissa == 0U) {
      return __uint_as_float(sign);
    }
    const float value = static_cast<float>(mantissa) * 0x1p-9F;
    return __uint_as_float(__float_as_uint(value) | sign);
  }
  if (magnitude == 0x7fU) {
    return __uint_as_float(sign | UINT32_C(0x7fc00000));
  }
  return __uint_as_float(sign | ((exponent + 120U) << 23U) | (mantissa << 20U));
}

__device__ __forceinline__ uint16_t bf16_rne(const float value) {
  const uint32_t bits = __float_as_uint(value);
  constexpr uint32_t exponent_mask = UINT32_C(0x7f800000);
  constexpr uint32_t fraction_mask = UINT32_C(0x007fffff);
  if ((bits & exponent_mask) == exponent_mask) {
    if ((bits & fraction_mask) != 0U) {
      return static_cast<uint16_t>(((bits >> 16U) & 0x8000U) | 0x7fc0U |
                                   ((bits >> 16U) & 0x003fU));
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & 0xffffU;
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

template <uint32_t ColumnsPerWave, bool Packed64>
__global__ __launch_bounds__(kThreads, 1) void decode_wave_kernel(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_scales, const uint8_t *const packed_weight,
    const uint8_t *const weight_scales, const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  if (m != 1U || k == 0U || (k % 16U) != 0U) {
    return;
  }
  constexpr uint32_t columns_per_workgroup = kWaves * ColumnsPerWave;
  const uint32_t lane = threadIdx.x & (kWaveWidth - 1U);
  const uint32_t wave = threadIdx.x / kWaveWidth;
  const uint64_t blocks_per_row = k / 16U;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * columns_per_workgroup +
      static_cast<uint64_t>(wave) * ColumnsPerWave;
  const uint64_t packed_row_bytes = k / 2U;
  float accumulators[ColumnsPerWave] = {};

  for (uint64_t block = lane; block < blocks_per_row; block += kWaveWidth) {
    const uint64_t packed_offset = block * 8U;
    const auto *const activation_words =
        reinterpret_cast<const uint32_t *>(packed_activation + packed_offset);
    ScaledPacks activation_pack0;
    ScaledPacks activation_pack1;
    if constexpr (Packed64) {
      const uint64_t packed =
          __builtin_nontemporal_load(reinterpret_cast<const uint64_t *>(
              packed_activation + packed_offset));
      activation_pack0 = scaled_packs(static_cast<uint32_t>(packed));
      activation_pack1 = scaled_packs(static_cast<uint32_t>(packed >> 32U));
    } else {
      activation_pack0 = scaled_packs(
          __builtin_nontemporal_load(activation_words + UINT32_C(0)));
      activation_pack1 = scaled_packs(
          __builtin_nontemporal_load(activation_words + UINT32_C(1)));
    }
    const float activation_scale =
        e4m3(__builtin_nontemporal_load(activation_scales + block));
#pragma unroll
    for (uint32_t column_offset = 0U; column_offset < ColumnsPerWave;
         ++column_offset) {
      const uint64_t column = column_base + column_offset;
      if (column >= n) {
        continue;
      }
      const uint8_t *const weight_row =
          packed_weight + column * packed_row_bytes + packed_offset;
      ScaledPacks weight_pack0;
      ScaledPacks weight_pack1;
      if constexpr (Packed64) {
        const uint64_t packed = __builtin_nontemporal_load(
            reinterpret_cast<const uint64_t *>(weight_row));
        weight_pack0 = scaled_packs(static_cast<uint32_t>(packed));
        weight_pack1 = scaled_packs(static_cast<uint32_t>(packed >> 32U));
      } else {
        const auto *const weight_words =
            reinterpret_cast<const uint32_t *>(weight_row);
        weight_pack0 = scaled_packs(
            __builtin_nontemporal_load(weight_words + UINT32_C(0)));
        weight_pack1 = scaled_packs(
            __builtin_nontemporal_load(weight_words + UINT32_C(1)));
      }
      int32_t block_sum = 0;
      block_sum = dot4(activation_pack0.even, weight_pack0.even, block_sum);
      block_sum = dot4(activation_pack0.odd, weight_pack0.odd, block_sum);
      block_sum = dot4(activation_pack1.even, weight_pack1.even, block_sum);
      block_sum = dot4(activation_pack1.odd, weight_pack1.odd, block_sum);
      const float weight_scale = e4m3(__builtin_nontemporal_load(
          weight_scales + column * blocks_per_row + block));
      accumulators[column_offset] += static_cast<float>(block_sum) * 0.25F *
                                     activation_scale * weight_scale;
    }
  }

#pragma unroll
  for (uint32_t column_offset = 0U; column_offset < ColumnsPerWave;
       ++column_offset) {
#pragma unroll
    for (uint32_t offset = kWaveWidth / 2U; offset != 0U; offset >>= 1U) {
      accumulators[column_offset] +=
          __shfl_down(accumulators[column_offset], offset, kWaveWidth);
    }
    const uint64_t column = column_base + column_offset;
    if (lane == 0U && column < n) {
      output[column] = bf16_rne(accumulators[column_offset] *
                                weight_tensor_scale[0] * input_tensor_scale[0]);
    }
  }
}

// Decode activation packs and E4M3 block scales once per workgroup.  The
// wave4col32 output layout remains unchanged, but eight waves reuse this
// activation representation from LDS instead of repeating global decode.
__global__ __launch_bounds__(kThreads, 1) void decode_activation_shared_kernel(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_scales, const uint8_t *const packed_weight,
    const uint8_t *const weight_scales, const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  if (m != 1U || k == 0U || (k % 16U) != 0U) {
    return;
  }
  constexpr uint32_t columns_per_wave = 4U;
  constexpr uint32_t columns_per_workgroup = kWaves * columns_per_wave;
  const uint32_t lane = threadIdx.x & (kWaveWidth - 1U);
  const uint32_t wave = threadIdx.x / kWaveWidth;
  const uint64_t blocks_per_row = k / 16U;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * columns_per_workgroup +
      static_cast<uint64_t>(wave) * columns_per_wave;
  const uint64_t packed_row_bytes = k / 2U;
  extern __shared__ uint32_t shared[];
  uint32_t *const activation_packs = shared;
  float *const activation_scale_values =
      reinterpret_cast<float *>(shared + blocks_per_row * 4U);

  for (uint64_t block = threadIdx.x; block < blocks_per_row;
       block += kThreads) {
    const auto *const words =
        reinterpret_cast<const uint32_t *>(packed_activation + block * 8U);
    const ScaledPacks first =
        scaled_packs(__builtin_nontemporal_load(words + UINT32_C(0)));
    const ScaledPacks second =
        scaled_packs(__builtin_nontemporal_load(words + UINT32_C(1)));
    activation_packs[block * 4U + 0U] = first.even;
    activation_packs[block * 4U + 1U] = first.odd;
    activation_packs[block * 4U + 2U] = second.even;
    activation_packs[block * 4U + 3U] = second.odd;
    activation_scale_values[block] =
        e4m3(__builtin_nontemporal_load(activation_scales + block));
  }
  __syncthreads();

  float accumulators[columns_per_wave] = {};
  for (uint64_t block = lane; block < blocks_per_row; block += kWaveWidth) {
    const ScaledPacks activation_pack0 = {activation_packs[block * 4U + 0U],
                                          activation_packs[block * 4U + 1U]};
    const ScaledPacks activation_pack1 = {activation_packs[block * 4U + 2U],
                                          activation_packs[block * 4U + 3U]};
    const float activation_scale = activation_scale_values[block];
#pragma unroll
    for (uint32_t column_offset = 0U; column_offset < columns_per_wave;
         ++column_offset) {
      const uint64_t column = column_base + column_offset;
      if (column >= n) {
        continue;
      }
      const auto *const weight_words = reinterpret_cast<const uint32_t *>(
          packed_weight + column * packed_row_bytes + block * 8U);
      const ScaledPacks weight_pack0 =
          scaled_packs(__builtin_nontemporal_load(weight_words + UINT32_C(0)));
      const ScaledPacks weight_pack1 =
          scaled_packs(__builtin_nontemporal_load(weight_words + UINT32_C(1)));
      int32_t block_sum = 0;
      block_sum = dot4(activation_pack0.even, weight_pack0.even, block_sum);
      block_sum = dot4(activation_pack0.odd, weight_pack0.odd, block_sum);
      block_sum = dot4(activation_pack1.even, weight_pack1.even, block_sum);
      block_sum = dot4(activation_pack1.odd, weight_pack1.odd, block_sum);
      const float weight_scale = e4m3(__builtin_nontemporal_load(
          weight_scales + column * blocks_per_row + block));
      accumulators[column_offset] += static_cast<float>(block_sum) * 0.25F *
                                     activation_scale * weight_scale;
    }
  }

#pragma unroll
  for (uint32_t column_offset = 0U; column_offset < columns_per_wave;
       ++column_offset) {
#pragma unroll
    for (uint32_t offset = kWaveWidth / 2U; offset != 0U; offset >>= 1U) {
      accumulators[column_offset] +=
          __shfl_down(accumulators[column_offset], offset, kWaveWidth);
    }
    const uint64_t column = column_base + column_offset;
    if (lane == 0U && column < n) {
      output[column] = bf16_rne(accumulators[column_offset] *
                                weight_tensor_scale[0] * input_tensor_scale[0]);
    }
  }
}

__global__ void block16_subtotal_kernel(const uint8_t *const activation,
                                        const uint8_t *const weight,
                                        int32_t *const output) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    const auto *const a = reinterpret_cast<const uint32_t *>(activation);
    const auto *const w = reinterpret_cast<const uint32_t *>(weight);
    const ScaledPacks ap0 = scaled_packs(a[0]);
    const ScaledPacks ap1 = scaled_packs(a[1]);
    const ScaledPacks wp0 = scaled_packs(w[0]);
    const ScaledPacks wp1 = scaled_packs(w[1]);
    int32_t subtotal = 0;
    subtotal = dot4(ap0.even, wp0.even, subtotal);
    subtotal = dot4(ap0.odd, wp0.odd, subtotal);
    subtotal = dot4(ap1.even, wp1.even, subtotal);
    output[0] = dot4(ap1.odd, wp1.odd, subtotal);
  }
}

enum class CandidateId { Control, Packed64, Wave8Col64, ActivationShared };

struct Candidate final {
  CandidateId id;
  const char *name;
  const void *function;
  uint32_t columns_per_workgroup;
};

Candidate candidate(const CandidateId id) {
  switch (id) {
  case CandidateId::Control:
    return {id, "id67-wave4col32",
            reinterpret_cast<const void *>(decode_wave_kernel<4U, false>), 32U};
  case CandidateId::Packed64:
    return {id, "candidate-packed64-wave4col32",
            reinterpret_cast<const void *>(decode_wave_kernel<4U, true>), 32U};
  case CandidateId::Wave8Col64:
    return {id, "candidate-wave8col64",
            reinterpret_cast<const void *>(decode_wave_kernel<8U, false>), 64U};
  case CandidateId::ActivationShared:
    return {id, "candidate-activation-shared-wave4col32",
            reinterpret_cast<const void *>(decode_activation_shared_kernel),
            32U};
  }
  return {CandidateId::Control, "invalid", nullptr, 0U};
}

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::fprintf(stderr, "hip error operation=%s status=%s\n", operation,
               hipGetErrorString(status));
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

float host_e2m1(const uint8_t code) {
  constexpr std::array<float, 8> positive = {0.0F, 0.5F, 1.0F, 1.5F,
                                             2.0F, 3.0F, 4.0F, 6.0F};
  const float value = positive[code & 7U];
  return (code & 8U) == 0U ? value : -value;
}

float host_e4m3(const uint8_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x80U) << 24U;
  const uint32_t magnitude = bits & 0x7fU;
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  if (exponent == 0U) {
    if (mantissa == 0U) {
      return 0.0F;
    }
    const float value = static_cast<float>(mantissa) * 0x1p-9F;
    uint32_t result = 0U;
    std::memcpy(&result, &value, sizeof(result));
    result |= sign;
    float signed_value = 0.0F;
    std::memcpy(&signed_value, &result, sizeof(signed_value));
    return signed_value;
  }
  if (magnitude == 0x7fU) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  uint32_t result = sign | ((exponent + 120U) << 23U) | (mantissa << 20U);
  float value = 0.0F;
  std::memcpy(&value, &result, sizeof(value));
  return value;
}

uint16_t host_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000U) == 0x7f800000U) {
    if ((bits & 0x007fffffU) != 0U) {
      return static_cast<uint16_t>(((bits >> 16U) & 0x8000U) | 0x7fc0U |
                                   ((bits >> 16U) & 0x003fU));
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & 0xffffU;
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

float host_bf16_to_float(const uint16_t bits) {
  const uint32_t expanded = static_cast<uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &expanded, sizeof(value));
  return value;
}

uint8_t nibble(const uint8_t code) {
  return static_cast<uint8_t>(code & 0x0fU);
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
  if (buffers->input_tensor_scale != nullptr) {
    (void)hipFree(buffers->input_tensor_scale);
  }
  if (buffers->weight_tensor_scale != nullptr) {
    (void)hipFree(buffers->weight_tensor_scale);
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

bool make_buffers(const uint64_t k, const uint64_t n, Buffers *const buffers) {
  const uint64_t blocks = k / 16U;
  if (buffers == nullptr || k == 0U || n == 0U || (k % 16U) != 0U ||
      k > UINT64_MAX / n || n * k / 2U > SIZE_MAX || n * blocks > SIZE_MAX) {
    return false;
  }
  const size_t activation_bytes = static_cast<size_t>(k / 2U);
  const size_t weight_bytes = static_cast<size_t>(n * k / 2U);
  const size_t weight_scale_bytes = static_cast<size_t>(n * blocks);
  return hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->activation),
                          activation_bytes),
                "hipMalloc activation") &&
         hip_ok(
             hipMalloc(reinterpret_cast<void **>(&buffers->activation_scales),
                       static_cast<size_t>(blocks)),
             "hipMalloc activation scales") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->weight),
                          weight_bytes),
                "hipMalloc weight") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->weight_scales),
                          weight_scale_bytes),
                "hipMalloc weight scales") &&
         hip_ok(
             hipMalloc(reinterpret_cast<void **>(&buffers->weight_tensor_scale),
                       sizeof(float)),
             "hipMalloc weight tensor scale") &&
         hip_ok(
             hipMalloc(reinterpret_cast<void **>(&buffers->input_tensor_scale),
                       sizeof(float)),
             "hipMalloc input tensor scale") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->output),
                          static_cast<size_t>(n * sizeof(uint16_t))),
                "hipMalloc output") &&
         hip_ok(hipStreamCreate(&buffers->stream), "hipStreamCreate") &&
         hip_ok(hipEventCreate(&buffers->start), "hipEventCreate start") &&
         hip_ok(hipEventCreate(&buffers->stop), "hipEventCreate stop");
}

void fill_inputs(const uint64_t k, const uint64_t n,
                 std::vector<uint8_t> *const activation,
                 std::vector<uint8_t> *const activation_scales,
                 std::vector<uint8_t> *const weight,
                 std::vector<uint8_t> *const weight_scales) {
  const uint64_t blocks = k / 16U;
  activation->assign(static_cast<size_t>(k / 2U), 0U);
  activation_scales->assign(static_cast<size_t>(blocks), 0x38U);
  weight->assign(static_cast<size_t>(n * k / 2U), 0U);
  weight_scales->assign(static_cast<size_t>(n * blocks), 0x38U);
  for (uint64_t inner = 0U; inner < k; ++inner) {
    const uint8_t code = static_cast<uint8_t>((inner * 5U + 3U) & 0x0fU);
    const size_t byte = static_cast<size_t>(inner / 2U);
    if ((inner & 1U) == 0U) {
      (*activation)[byte] = code;
    } else {
      (*activation)[byte] |= static_cast<uint8_t>(code << 4U);
    }
  }
  for (uint64_t column = 0U; column < n; ++column) {
    for (uint64_t inner = 0U; inner < k; ++inner) {
      const uint8_t code =
          static_cast<uint8_t>((column * 3U + inner * 7U + 9U) & 0x0fU);
      const size_t byte = static_cast<size_t>(column * k / 2U + inner / 2U);
      if ((inner & 1U) == 0U) {
        (*weight)[byte] = code;
      } else {
        (*weight)[byte] |= static_cast<uint8_t>(code << 4U);
      }
    }
    // Alternate scale domains so the epilogue tests both block16 subtotal and
    // scale application without introducing non-finite values.
    for (uint64_t block = 0U; block < blocks; ++block) {
      (*weight_scales)[static_cast<size_t>(column * blocks + block)] =
          static_cast<uint8_t>((block & 1U) == 0U ? 0x38U : 0x40U);
    }
  }
  for (uint64_t block = 0U; block < blocks; ++block) {
    (*activation_scales)[static_cast<size_t>(block)] =
        static_cast<uint8_t>((block & 1U) == 0U ? 0x38U : 0x30U);
  }
}

bool upload(const uint64_t k, const uint64_t n,
            const std::vector<uint8_t> &activation,
            const std::vector<uint8_t> &activation_scales,
            const std::vector<uint8_t> &weight,
            const std::vector<uint8_t> &weight_scales, Buffers *const buffers) {
  const uint64_t blocks = k / 16U;
  const float weight_tensor_scale = 0.75F;
  const float input_tensor_scale = 1.125F;
  return hip_ok(hipMemcpy(buffers->activation, activation.data(), k / 2U,
                          hipMemcpyHostToDevice),
                "hipMemcpy activation") &&
         hip_ok(hipMemcpy(buffers->activation_scales, activation_scales.data(),
                          blocks, hipMemcpyHostToDevice),
                "hipMemcpy activation scales") &&
         hip_ok(hipMemcpy(buffers->weight, weight.data(), n * k / 2U,
                          hipMemcpyHostToDevice),
                "hipMemcpy weight") &&
         hip_ok(hipMemcpy(buffers->weight_scales, weight_scales.data(),
                          n * blocks, hipMemcpyHostToDevice),
                "hipMemcpy weight scales") &&
         hip_ok(hipMemcpy(buffers->weight_tensor_scale, &weight_tensor_scale,
                          sizeof(float), hipMemcpyHostToDevice),
                "hipMemcpy weight tensor scale") &&
         hip_ok(hipMemcpy(buffers->input_tensor_scale, &input_tensor_scale,
                          sizeof(float), hipMemcpyHostToDevice),
                "hipMemcpy input tensor scale") &&
         hip_ok(hipMemset(buffers->output, 0, n * sizeof(uint16_t)),
                "hipMemset output");
}

size_t dynamic_shared_bytes(const Candidate &candidate, const uint64_t k) {
  if (candidate.id != CandidateId::ActivationShared) {
    return 0U;
  }
  const uint64_t blocks = k / 16U;
  return static_cast<size_t>(blocks * 5U * sizeof(uint32_t));
}

bool launch(const Candidate &candidate, const uint64_t k, const uint64_t n,
            Buffers *const buffers) {
  const uint64_t blocks = (n + candidate.columns_per_workgroup - 1U) /
                          candidate.columns_per_workgroup;
  const size_t shared = dynamic_shared_bytes(candidate, k);
  if (blocks == 0U || blocks > UINT32_MAX) {
    return false;
  }
  switch (candidate.id) {
  case CandidateId::Control:
    hipLaunchKernelGGL((decode_wave_kernel<4U, false>),
                       dim3(static_cast<uint32_t>(blocks)), dim3(kThreads), 0U,
                       buffers->stream, buffers->activation,
                       buffers->activation_scales, buffers->weight,
                       buffers->weight_scales, buffers->weight_tensor_scale,
                       buffers->input_tensor_scale, buffers->output, 1U, k, n);
    break;
  case CandidateId::Packed64:
    hipLaunchKernelGGL((decode_wave_kernel<4U, true>),
                       dim3(static_cast<uint32_t>(blocks)), dim3(kThreads), 0U,
                       buffers->stream, buffers->activation,
                       buffers->activation_scales, buffers->weight,
                       buffers->weight_scales, buffers->weight_tensor_scale,
                       buffers->input_tensor_scale, buffers->output, 1U, k, n);
    break;
  case CandidateId::Wave8Col64:
    hipLaunchKernelGGL((decode_wave_kernel<8U, false>),
                       dim3(static_cast<uint32_t>(blocks)), dim3(kThreads), 0U,
                       buffers->stream, buffers->activation,
                       buffers->activation_scales, buffers->weight,
                       buffers->weight_scales, buffers->weight_tensor_scale,
                       buffers->input_tensor_scale, buffers->output, 1U, k, n);
    break;
  case CandidateId::ActivationShared:
    hipLaunchKernelGGL(decode_activation_shared_kernel,
                       dim3(static_cast<uint32_t>(blocks)), dim3(kThreads),
                       shared, buffers->stream, buffers->activation,
                       buffers->activation_scales, buffers->weight,
                       buffers->weight_scales, buffers->weight_tensor_scale,
                       buffers->input_tensor_scale, buffers->output, 1U, k, n);
    break;
  }
  return hipGetLastError() == hipSuccess;
}

bool measure(const Candidate &candidate, const uint64_t k, const uint64_t n,
             Buffers *const buffers, float *const median_us) {
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    if (!launch(candidate, k, n, buffers) ||
        !hip_ok(hipStreamSynchronize(buffers->stream), "warmup synchronize")) {
      return false;
    }
  }
  std::array<float, kMeasured> samples{};
  for (int iteration = 0; iteration < kMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(buffers->start, buffers->stream),
                "event start") ||
        !launch(candidate, k, n, buffers) ||
        !hip_ok(hipEventRecord(buffers->stop, buffers->stream), "event stop") ||
        !hip_ok(hipEventSynchronize(buffers->stop), "event synchronize") ||
        !hip_ok(hipEventElapsedTime(&samples[iteration], buffers->start,
                                    buffers->stop),
                "event elapsed")) {
      return false;
    }
    samples[iteration] *= 1000.0F;
  }
  std::sort(samples.begin(), samples.end());
  *median_us = samples[kMeasured / 2U];
  return true;
}

bool check_matrix_oracle(const Candidate &candidate, const uint64_t k,
                         const uint64_t n,
                         const std::vector<uint8_t> &activation,
                         const std::vector<uint8_t> &activation_scales,
                         const std::vector<uint8_t> &weight,
                         const std::vector<uint8_t> &weight_scales,
                         const std::vector<uint16_t> &actual) {
  const uint64_t blocks = k / 16U;
  size_t mismatches = 0U;
  double max_abs = 0.0;
  double max_rel = 0.0;
  for (uint64_t column = 0U; column < n; ++column) {
    float accumulator = 0.0F;
    for (uint64_t inner = 0U; inner < k; ++inner) {
      const uint8_t a_byte = activation[static_cast<size_t>(inner / 2U)];
      const uint8_t w_byte =
          weight[static_cast<size_t>(column * k / 2U + inner / 2U)];
      const uint8_t a_code = (inner & 1U) == 0U ? a_byte & 0x0fU : a_byte >> 4U;
      const uint8_t w_code = (inner & 1U) == 0U ? w_byte & 0x0fU : w_byte >> 4U;
      accumulator +=
          host_e2m1(a_code) *
          host_e4m3(activation_scales[static_cast<size_t>(inner / 16U)]) *
          host_e2m1(w_code) *
          host_e4m3(weight_scales[static_cast<size_t>(column * blocks +
                                                      inner / 16U)]);
    }
    const uint16_t expected_bits = host_bf16_rne(accumulator * 0.75F * 1.125F);
    const float expected = host_bf16_to_float(expected_bits);
    const float observed =
        host_bf16_to_float(actual[static_cast<size_t>(column)]);
    const double absolute = std::abs(static_cast<double>(observed) - expected);
    const double relative =
        absolute / std::max(1.0e-6, std::abs(static_cast<double>(expected)));
    max_abs = std::max(max_abs, absolute);
    max_rel = std::max(max_rel, relative);
    if (expected_bits != actual[static_cast<size_t>(column)]) {
      ++mismatches;
    }
  }
  std::printf("oracle candidate=%s k=%llu n=%llu max_abs=%.8g max_rel=%.8g "
              "mismatches=%zu status=%s\n",
              candidate.name, static_cast<unsigned long long>(k),
              static_cast<unsigned long long>(n), max_abs, max_rel, mismatches,
              mismatches == 0U ? "PASS" : "FAIL");
  return mismatches == 0U;
}

bool check_block16_subtotal() {
  std::array<uint8_t, 8> activation{};
  std::array<uint8_t, 8> weight{};
  for (uint32_t index = 0U; index < 16U; ++index) {
    const uint8_t a = static_cast<uint8_t>(index & 0x0fU);
    const uint8_t w = static_cast<uint8_t>((15U - index) & 0x0fU);
    if ((index & 1U) == 0U) {
      activation[index / 2U] = a;
      weight[index / 2U] = w;
    } else {
      activation[index / 2U] |= static_cast<uint8_t>(a << 4U);
      weight[index / 2U] |= static_cast<uint8_t>(w << 4U);
    }
  }
  uint8_t *device_activation = nullptr;
  uint8_t *device_weight = nullptr;
  int32_t *device_output = nullptr;
  if (!hip_ok(hipMalloc(reinterpret_cast<void **>(&device_activation), 8U),
              "hipMalloc subtotal activation") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&device_weight), 8U),
              "hipMalloc subtotal weight") ||
      !hip_ok(
          hipMalloc(reinterpret_cast<void **>(&device_output), sizeof(int32_t)),
          "hipMalloc subtotal output") ||
      !hip_ok(hipMemcpy(device_activation, activation.data(), 8U,
                        hipMemcpyHostToDevice),
              "hipMemcpy subtotal activation") ||
      !hip_ok(
          hipMemcpy(device_weight, weight.data(), 8U, hipMemcpyHostToDevice),
          "hipMemcpy subtotal weight")) {
    if (device_output != nullptr)
      (void)hipFree(device_output);
    if (device_weight != nullptr)
      (void)hipFree(device_weight);
    if (device_activation != nullptr)
      (void)hipFree(device_activation);
    return false;
  }
  hipLaunchKernelGGL(block16_subtotal_kernel, dim3(1), dim3(1), 0U, nullptr,
                     device_activation, device_weight, device_output);
  bool ok = hip_ok(hipGetLastError(), "launch subtotal") &&
            hip_ok(hipDeviceSynchronize(), "synchronize subtotal");
  int32_t actual = 0;
  ok = ok && hip_ok(hipMemcpy(&actual, device_output, sizeof(actual),
                              hipMemcpyDeviceToHost),
                    "hipMemcpy subtotal output");
  int32_t expected = 0;
  for (uint32_t index = 0U; index < 16U; ++index) {
    expected += static_cast<int32_t>(host_e2m1(index) * 2.0F) *
                static_cast<int32_t>(host_e2m1(15U - index) * 2.0F);
  }
  std::printf("oracle block16 subtotal expected=%d actual=%d status=%s\n",
              expected, actual, ok && expected == actual ? "PASS" : "FAIL");
  (void)hipFree(device_output);
  (void)hipFree(device_weight);
  (void)hipFree(device_activation);
  return ok && expected == actual;
}

void print_resources(const Candidate &candidate, const uint64_t k) {
  hipFuncAttributes attributes{};
  const hipError_t attr_status =
      hipFuncGetAttributes(&attributes, candidate.function);
  int active_blocks = 0;
  const hipError_t occupancy_status =
      hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks, candidate.function, kThreads,
          dynamic_shared_bytes(candidate, k));
  std::printf("resources candidate=%s registers=%d sgpr=ISA-metadata lds=%zu "
              "scratch=%zu active_blocks=%d attr=%s occupancy=%s\n",
              candidate.name, attributes.numRegs, attributes.sharedSizeBytes,
              attributes.localSizeBytes, active_blocks,
              hipGetErrorString(attr_status),
              hipGetErrorString(occupancy_status));
}

} // namespace

int main() {
  constexpr int device = 0;
  if (!hip_ok(hipSetDevice(device), "hipSetDevice")) {
    return EXIT_FAILURE;
  }
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device),
              "hipGetDeviceProperties") ||
      !exact_gfx1030(properties.gcnArchName)) {
    std::fprintf(stderr, "exact gfx1030 required; got %s\n",
                 properties.gcnArchName);
    return EXIT_FAILURE;
  }
  std::printf("target=%s device=%d pci=%04x:%02x:%02x name=%s\n",
              properties.gcnArchName, device, properties.pciDomainID,
              properties.pciBusID, properties.pciDeviceID, properties.name);
  bool all_ok = check_block16_subtotal();

  const std::array<CandidateId, 4> candidate_ids = {
      CandidateId::Control, CandidateId::Packed64, CandidateId::Wave8Col64,
      CandidateId::ActivationShared};
  for (const CandidateId id : candidate_ids) {
    print_resources(candidate(id), 17408U);
  }

  // Numerical oracle: all E2M1 codes appear in the packed data, two block16
  // scale domains are used, and N=17 exercises the tail of a 32-column tile.
  constexpr uint64_t oracle_k = 32U;
  constexpr uint64_t oracle_n = 17U;
  std::vector<uint8_t> activation;
  std::vector<uint8_t> activation_scales;
  std::vector<uint8_t> weight;
  std::vector<uint8_t> weight_scales;
  fill_inputs(oracle_k, oracle_n, &activation, &activation_scales, &weight,
              &weight_scales);
  Buffers oracle_buffers;
  if (!make_buffers(oracle_k, oracle_n, &oracle_buffers) ||
      !upload(oracle_k, oracle_n, activation, activation_scales, weight,
              weight_scales, &oracle_buffers)) {
    cleanup(&oracle_buffers);
    return EXIT_FAILURE;
  }
  std::vector<uint16_t> oracle_output(static_cast<size_t>(oracle_n));
  for (const CandidateId id : candidate_ids) {
    const Candidate current = candidate(id);
    if (!launch(current, oracle_k, oracle_n, &oracle_buffers) ||
        !hip_ok(hipDeviceSynchronize(), "oracle synchronize") ||
        !hip_ok(hipMemcpy(oracle_output.data(), oracle_buffers.output,
                          oracle_output.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "hipMemcpy oracle")) {
      cleanup(&oracle_buffers);
      return EXIT_FAILURE;
    }
    all_ok = check_matrix_oracle(current, oracle_k, oracle_n, activation,
                                 activation_scales, weight, weight_scales,
                                 oracle_output) &&
             all_ok;
  }
  cleanup(&oracle_buffers);

  struct Shape final {
    uint64_t k;
    uint64_t n;
    uint32_t calls_per_token;
  };
  const std::array<Shape, 2> shapes = {Shape{5120U, 17408U, 112U},
                                       Shape{17408U, 5120U, 56U}};
  std::array<std::array<float, 2>, 4> measured{};
  for (size_t shape_index = 0U; shape_index < shapes.size(); ++shape_index) {
    const Shape shape = shapes[shape_index];
    fill_inputs(shape.k, shape.n, &activation, &activation_scales, &weight,
                &weight_scales);
    Buffers buffers;
    if (!make_buffers(shape.k, shape.n, &buffers) ||
        !upload(shape.k, shape.n, activation, activation_scales, weight,
                weight_scales, &buffers)) {
      cleanup(&buffers);
      return EXIT_FAILURE;
    }
    for (size_t candidate_index = 0U; candidate_index < candidate_ids.size();
         ++candidate_index) {
      const Candidate current = candidate(candidate_ids[candidate_index]);
      float median_us = 0.0F;
      if (!measure(current, shape.k, shape.n, &buffers, &median_us)) {
        cleanup(&buffers);
        return EXIT_FAILURE;
      }
      measured[candidate_index][shape_index] = median_us;
      const double bytes = static_cast<double>(shape.k) * shape.n / 2.0;
      std::printf(
          "result candidate=%s k=%llu n=%llu median_us=%.3f gbps=%.6f\n",
          current.name, static_cast<unsigned long long>(shape.k),
          static_cast<unsigned long long>(shape.n), median_us,
          bytes / static_cast<double>(median_us) / 1000.0);
      // Decode output is only N BF16 values.  Capture one result after timing
      // and compare every candidate against the ID67 control for this exact
      // production shape, including the N tail when present.
      if (!launch(current, shape.k, shape.n, &buffers) ||
          !hip_ok(hipDeviceSynchronize(), "large-shape compare synchronize")) {
        cleanup(&buffers);
        return EXIT_FAILURE;
      }
      std::vector<uint16_t> observed(static_cast<size_t>(shape.n));
      if (!hip_ok(hipMemcpy(observed.data(), buffers.output,
                            observed.size() * sizeof(uint16_t),
                            hipMemcpyDeviceToHost),
                  "large-shape compare output")) {
        cleanup(&buffers);
        return EXIT_FAILURE;
      }
      static std::array<std::vector<uint16_t>, 2> controls;
      if (candidate_index == 0U) {
        controls[shape_index] = std::move(observed);
      } else {
        size_t output_mismatches = 0U;
        for (size_t output_index = 0U; output_index < observed.size();
             ++output_index) {
          if (observed[output_index] != controls[shape_index][output_index]) {
            ++output_mismatches;
          }
        }
        std::printf(
            "compare candidate=%s k=%llu n=%llu mismatches=%zu status=%s\n",
            current.name, static_cast<unsigned long long>(shape.k),
            static_cast<unsigned long long>(shape.n), output_mismatches,
            output_mismatches == 0U ? "PASS" : "FAIL");
        all_ok = output_mismatches == 0U && all_ok;
      }
    }
    cleanup(&buffers);
  }

  const double weighted_call_time_us =
      static_cast<double>(shapes[0].calls_per_token) * measured[0][0] +
      static_cast<double>(shapes[1].calls_per_token) * measured[0][1];
  const double weighted_bytes = static_cast<double>(shapes[0].calls_per_token) *
                                    shapes[0].k * shapes[0].n / 2.0 +
                                static_cast<double>(shapes[1].calls_per_token) *
                                    shapes[1].k * shapes[1].n / 2.0;
  for (size_t index = 0U; index < candidate_ids.size(); ++index) {
    const double time_us =
        static_cast<double>(shapes[0].calls_per_token) * measured[index][0] +
        static_cast<double>(shapes[1].calls_per_token) * measured[index][1];
    const double gbps = weighted_bytes / time_us / 1000.0;
    std::printf("weighted candidate=%s calls=112+56 ms_per_token=%.6f "
                "gbps=%.6f speedup_vs_id67=%.6f%%\n",
                candidate(candidate_ids[index]).name, time_us / 1000.0, gbps,
                index == 0U ? 0.0
                            : (weighted_call_time_us / time_us - 1.0) * 100.0);
  }

  std::printf("summary status=%s candidates=%zu warmups=%d measured=%d\n",
              all_ok ? "PASS" : "FAIL", candidate_ids.size(), kWarmups,
              kMeasured);
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
