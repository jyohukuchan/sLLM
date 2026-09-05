// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase78-nvfp4-byte-permute-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// 3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70, ggml/src/ggml-cuda/vecdotq.cuh
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

// Phase 78 standalone gfx1030 NVFP4 W4A4 transient int8 staging probe.
//
// This is an evidence-only comparison for ID62.  The control kernel keeps
// ID62's packed-NVFP4 -> LDS signed-byte conversion in each K tile.  The
// candidate first expands the packed E2M1 values to exact value*2 int8
// bytes, then runs the same 64x64/K32/block16 DP4A body from that transient
// workspace.  The resident model representation and production selector are
// not changed by this file.

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
#include <vector>

namespace {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kTileM = 64U;
constexpr uint32_t kTileN = 64U;
constexpr uint32_t kTileK = 32U;
constexpr uint32_t kBlockK = 16U;
constexpr uint32_t kRowsPerThread = 4U;
constexpr uint32_t kColumnsPerThread = 4U;
constexpr uint32_t kWarmups = 3U;
constexpr uint32_t kMeasured = 10U;

struct ProbePacks final {
  int32_t even;
  int32_t odd;
};

__device__ __forceinline__ ProbePacks
sllm_phase78_i8_probe_decode_x8(const uint32_t packed) noexcept {
  // This is the same byte-permute lookup used by the production ID62 body.
  constexpr uint32_t table_0_3 = UINT32_C(0x03020100);
  constexpr uint32_t table_4_7 = UINT32_C(0x0c080604);
  constexpr uint32_t table_8_11 = UINT32_C(0xfdfeff00);
  constexpr uint32_t table_12_15 = UINT32_C(0xf4f8fafc);
  constexpr uint32_t low_index_mask = UINT32_C(0x07070707);
  constexpr uint32_t byte_identity = UINT32_C(0x03020100);
  constexpr uint32_t sign_index_mask = UINT32_C(0x08080808);
  const uint32_t even_indices = packed;
  const uint32_t odd_indices = packed >> 4U;
  const uint32_t even_low = __builtin_amdgcn_perm(
      table_4_7, table_0_3, even_indices & low_index_mask);
  const uint32_t odd_low =
      __builtin_amdgcn_perm(table_4_7, table_0_3, odd_indices & low_index_mask);
  const uint32_t even_high = __builtin_amdgcn_perm(
      table_12_15, table_8_11, even_indices & low_index_mask);
  const uint32_t odd_high = __builtin_amdgcn_perm(table_12_15, table_8_11,
                                                  odd_indices & low_index_mask);
  const uint32_t even_select =
      byte_identity | ((even_indices & sign_index_mask) >> 1U);
  const uint32_t odd_select =
      byte_identity | ((odd_indices & sign_index_mask) >> 1U);
  return {static_cast<int32_t>(
              __builtin_amdgcn_perm(even_high, even_low, even_select)),
          static_cast<int32_t>(
              __builtin_amdgcn_perm(odd_high, odd_low, odd_select))};
}

__device__ __forceinline__ int32_t sllm_phase78_i8_probe_dot4(
    const int32_t lhs, const int32_t rhs, const int32_t accumulator) noexcept {
#if __has_builtin(__builtin_amdgcn_sdot4)
  return __builtin_amdgcn_sdot4(lhs, rhs, accumulator, false);
#else
  int32_t result = accumulator;
#pragma unroll
  for (uint32_t lane = 0U; lane < 4U; ++lane) {
    const int32_t left =
        static_cast<int8_t>(static_cast<uint32_t>(lhs) >> (lane * 8U));
    const int32_t right =
        static_cast<int8_t>(static_cast<uint32_t>(rhs) >> (lane * 8U));
    result += left * right;
  }
  return result;
#endif
}

__device__ __forceinline__ float
sllm_phase78_i8_probe_e4m3(const uint8_t bits) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode(bits);
}

__device__ __forceinline__ uint16_t
sllm_phase78_i8_probe_bf16(const float value) noexcept {
  const uint32_t bits = __float_as_uint(value);
  constexpr uint32_t exponent_mask = UINT32_C(0x7f800000);
  constexpr uint32_t fraction_mask = UINT32_C(0x007fffff);
  if ((bits & exponent_mask) == exponent_mask) {
    if ((bits & fraction_mask) != 0U) {
      const uint16_t sign =
          static_cast<uint16_t>((bits >> 16U) & UINT32_C(0x8000));
      const uint16_t payload =
          static_cast<uint16_t>((bits >> 16U) & UINT32_C(0x003f));
      return static_cast<uint16_t>(sign | UINT16_C(0x7fc0) | payload);
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & UINT32_C(1)) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

// Exact ID62 control: packed values are permuted into signed int8x4 packs
// inside every K32 stage, then consumed by the original block16 reduction.
__global__
__launch_bounds__(kThreads, 1) void sllm_phase78_i8_probe_id62_control(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_scales, const uint8_t *const packed_weight,
    const uint8_t *const weight_scales, const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  constexpr uint32_t blocks_per_stage = kTileK / kBlockK;
  constexpr uint32_t packed_chunks_per_stage = kTileK / 8U;
  constexpr uint32_t lds_group_stride = kTileK / 4U + 1U;
  constexpr uint32_t lds_scale_stride = blocks_per_stage + 1U;
  __shared__ int32_t activation_tile[kTileM][lds_group_stride];
  __shared__ int32_t weight_tile[kTileN][lds_group_stride];
  __shared__ float activation_scale_tile[kTileM][lds_scale_stride];
  __shared__ float weight_scale_tile[kTileN][lds_scale_stride];

  const uint64_t column_tiles = (n + kTileN - 1U) / kTileN;
  const uint64_t tile_index = blockIdx.x;
  const uint64_t row_base = (tile_index / column_tiles) * kTileM;
  const uint64_t column_base = (tile_index % column_tiles) * kTileN;
  const uint32_t thread = threadIdx.x;
  const uint32_t local_row = thread >> 4U;
  const uint32_t local_column = thread & 15U;
  const uint64_t packed_row_bytes = k / 2U;
  const uint64_t blocks_per_row = k / kBlockK;
  float accumulators[kRowsPerThread][kColumnsPerThread] = {};

  for (uint64_t base = 0U; base < k; base += kTileK) {
    for (uint32_t index = thread; index < kTileM * packed_chunks_per_stage;
         index += kThreads) {
      const uint32_t row = index / packed_chunks_per_stage;
      const uint32_t chunk = index % packed_chunks_per_stage;
      const uint64_t source_row = row_base + row;
      const uint64_t inner = base + static_cast<uint64_t>(chunk) * 8U;
      const ProbePacks values =
          source_row < m && inner + 8U <= k
              ? sllm_phase78_i8_probe_decode_x8(__builtin_nontemporal_load(
                    reinterpret_cast<const uint32_t *>(
                        packed_activation + source_row * packed_row_bytes +
                        inner / 2U)))
              : ProbePacks{0, 0};
      activation_tile[row][chunk * 2U] = values.even;
      activation_tile[row][chunk * 2U + 1U] = values.odd;
    }
    for (uint32_t index = thread; index < kTileN * packed_chunks_per_stage;
         index += kThreads) {
      const uint32_t column = index / packed_chunks_per_stage;
      const uint32_t chunk = index % packed_chunks_per_stage;
      const uint64_t source_column = column_base + column;
      const uint64_t inner = base + static_cast<uint64_t>(chunk) * 8U;
      const ProbePacks values =
          source_column < n && inner + 8U <= k
              ? sllm_phase78_i8_probe_decode_x8(__builtin_nontemporal_load(
                    reinterpret_cast<const uint32_t *>(
                        packed_weight + source_column * packed_row_bytes +
                        inner / 2U)))
              : ProbePacks{0, 0};
      weight_tile[column][chunk * 2U] = values.even;
      weight_tile[column][chunk * 2U + 1U] = values.odd;
    }
    for (uint32_t index = thread; index < kTileM * blocks_per_stage;
         index += kThreads) {
      const uint32_t row = index / blocks_per_stage;
      const uint32_t block = index % blocks_per_stage;
      const uint64_t source_row = row_base + row;
      const uint64_t source_block = base / kBlockK + block;
      activation_scale_tile[row][block] =
          source_row < m && source_block < blocks_per_row
              ? sllm_phase78_i8_probe_e4m3(__builtin_nontemporal_load(
                    activation_scales + source_row * blocks_per_row +
                    source_block))
              : 0.0F;
    }
    for (uint32_t index = thread; index < kTileN * blocks_per_stage;
         index += kThreads) {
      const uint32_t column = index / blocks_per_stage;
      const uint32_t block = index % blocks_per_stage;
      const uint64_t source_column = column_base + column;
      const uint64_t source_block = base / kBlockK + block;
      weight_scale_tile[column][block] =
          source_column < n && source_block < blocks_per_row
              ? sllm_phase78_i8_probe_e4m3(__builtin_nontemporal_load(
                    weight_scales + source_column * blocks_per_row +
                    source_block))
              : 0.0F;
    }
    __syncthreads();

#pragma unroll
    for (uint32_t block = 0U; block < blocks_per_stage; ++block) {
      if (base + static_cast<uint64_t>(block) * kBlockK >= k)
        continue;
      int32_t block_sums[kRowsPerThread][kColumnsPerThread] = {};
#pragma unroll
      for (uint32_t group = 0U; group < kBlockK / 4U; ++group) {
        int32_t activation_packs[kRowsPerThread];
        int32_t weight_packs[kColumnsPerThread];
#pragma unroll
        for (uint32_t row = 0U; row < kRowsPerThread; ++row)
          activation_packs[row] =
              activation_tile[local_row + row * 16U]
                             [block * (kBlockK / 4U) + group];
#pragma unroll
        for (uint32_t column = 0U; column < kColumnsPerThread; ++column)
          weight_packs[column] = weight_tile[local_column + column * 16U]
                                            [block * (kBlockK / 4U) + group];
#pragma unroll
        for (uint32_t row = 0U; row < kRowsPerThread; ++row)
#pragma unroll
          for (uint32_t column = 0U; column < kColumnsPerThread; ++column)
            block_sums[row][column] = sllm_phase78_i8_probe_dot4(
                activation_packs[row], weight_packs[column],
                block_sums[row][column]);
      }
#pragma unroll
      for (uint32_t row = 0U; row < kRowsPerThread; ++row) {
        const float activation_scale =
            activation_scale_tile[local_row + row * 16U][block];
#pragma unroll
        for (uint32_t column = 0U; column < kColumnsPerThread; ++column) {
          const float weight_scale =
              weight_scale_tile[local_column + column * 16U][block];
          accumulators[row][column] +=
              static_cast<float>(block_sums[row][column]) * 0.25F *
              activation_scale * weight_scale;
        }
      }
    }
    __syncthreads();
  }

  const float tensor_scale = weight_tensor_scale[0] * input_tensor_scale[0];
#pragma unroll
  for (uint32_t row = 0U; row < kRowsPerThread; ++row)
#pragma unroll
    for (uint32_t column = 0U; column < kColumnsPerThread; ++column) {
      const uint64_t output_row = row_base + local_row + row * 16U;
      const uint64_t output_column = column_base + local_column + column * 16U;
      if (output_row < m && output_column < n) {
        output[output_row * n + output_column] = sllm_phase78_i8_probe_bf16(
            accumulators[row][column] * tensor_scale);
      }
    }
}

// One thread owns one packed 8-value chunk.  The output is row-major for A
// and column-major for B, matching the source strides consumed by ID62.
__device__ __forceinline__ int8_t
sllm_phase78_i8_probe_scaled2(const uint8_t code) noexcept {
  constexpr int8_t magnitudes[8] = {0, 1, 2, 3, 4, 6, 8, 12};
  const int8_t value = magnitudes[code & 7U];
  return (code & 8U) == 0U ? value : static_cast<int8_t>(-value);
}

__global__
__launch_bounds__(kThreads, 1) void sllm_phase78_i8_probe_stage_packed(
    const uint8_t *const packed, int8_t *const staged, const uint64_t rows,
    const uint64_t k) {
  const uint64_t chunks_per_row = k / 8U;
  const uint64_t chunk_index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t chunk_count = rows * chunks_per_row;
  if (chunk_index >= chunk_count)
    return;
  const uint64_t row = chunk_index / chunks_per_row;
  const uint64_t chunk = chunk_index - row * chunks_per_row;
  const uint64_t packed_offset = row * (k / 2U) + chunk * 4U;
  const uint32_t packed_values = __builtin_nontemporal_load(
      reinterpret_cast<const uint32_t *>(packed + packed_offset));
  const uint64_t output_offset = row * k + chunk * 8U;
#pragma unroll
  for (uint32_t lane = 0U; lane < 8U; ++lane) {
    const uint8_t pair =
        static_cast<uint8_t>(packed_values >> ((lane / 2U) * 8U));
    const uint8_t code = (lane & 1U) == 0U ? pair & UINT8_C(0x0f) : pair >> 4U;
    staged[output_offset + lane] = sllm_phase78_i8_probe_scaled2(code);
  }
}

// The candidate has the same LDS shape and block16 operations as ID62.  Only
// the source of each four-byte pack differs: staged int8 bytes instead of a
// repeated packed-NVFP4 permute.  Keep this arithmetic sequence identical to
// the control so any mismatch is visible as a BF16 bit difference.
__global__
__launch_bounds__(kThreads, 1) void sllm_phase78_i8_probe_staged_matmul(
    const int8_t *const staged_activation,
    const uint8_t *const activation_scales, const int8_t *const staged_weight,
    const uint8_t *const weight_scales, const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  constexpr uint32_t blocks_per_stage = kTileK / kBlockK;
  constexpr uint32_t groups_per_stage = kTileK / 4U;
  constexpr uint32_t lds_group_stride = groups_per_stage + 1U;
  constexpr uint32_t lds_scale_stride = blocks_per_stage + 1U;
  __shared__ int32_t activation_tile[kTileM][lds_group_stride];
  __shared__ int32_t weight_tile[kTileN][lds_group_stride];
  __shared__ float activation_scale_tile[kTileM][lds_scale_stride];
  __shared__ float weight_scale_tile[kTileN][lds_scale_stride];

  const uint64_t column_tiles = (n + kTileN - 1U) / kTileN;
  const uint64_t tile_index = blockIdx.x;
  const uint64_t row_base = (tile_index / column_tiles) * kTileM;
  const uint64_t column_base = (tile_index % column_tiles) * kTileN;
  const uint32_t thread = threadIdx.x;
  const uint32_t local_row = thread >> 4U;
  const uint32_t local_column = thread & 15U;
  const uint64_t blocks_per_row = k / kBlockK;
  float accumulators[kRowsPerThread][kColumnsPerThread] = {};

  for (uint64_t base = 0U; base < k; base += kTileK) {
    for (uint32_t index = thread; index < kTileM * groups_per_stage;
         index += kThreads) {
      const uint32_t row = index / groups_per_stage;
      const uint32_t group = index % groups_per_stage;
      const uint64_t source_row = row_base + row;
      const uint64_t inner = base + static_cast<uint64_t>(group) * 4U;
      activation_tile[row][group] =
          source_row < m && inner + 4U <= k
              ? static_cast<int32_t>(__builtin_nontemporal_load(
                    reinterpret_cast<const uint32_t *>(staged_activation +
                                                       source_row * k + inner)))
              : 0;
    }
    for (uint32_t index = thread; index < kTileN * groups_per_stage;
         index += kThreads) {
      const uint32_t column = index / groups_per_stage;
      const uint32_t group = index % groups_per_stage;
      const uint64_t source_column = column_base + column;
      const uint64_t inner = base + static_cast<uint64_t>(group) * 4U;
      weight_tile[column][group] =
          source_column < n && inner + 4U <= k
              ? static_cast<int32_t>(__builtin_nontemporal_load(
                    reinterpret_cast<const uint32_t *>(
                        staged_weight + source_column * k + inner)))
              : 0;
    }
    for (uint32_t index = thread; index < kTileM * blocks_per_stage;
         index += kThreads) {
      const uint32_t row = index / blocks_per_stage;
      const uint32_t block = index % blocks_per_stage;
      const uint64_t source_row = row_base + row;
      const uint64_t source_block = base / kBlockK + block;
      activation_scale_tile[row][block] =
          source_row < m && source_block < blocks_per_row
              ? sllm_phase78_i8_probe_e4m3(__builtin_nontemporal_load(
                    activation_scales + source_row * blocks_per_row +
                    source_block))
              : 0.0F;
    }
    for (uint32_t index = thread; index < kTileN * blocks_per_stage;
         index += kThreads) {
      const uint32_t column = index / blocks_per_stage;
      const uint32_t block = index % blocks_per_stage;
      const uint64_t source_column = column_base + column;
      const uint64_t source_block = base / kBlockK + block;
      weight_scale_tile[column][block] =
          source_column < n && source_block < blocks_per_row
              ? sllm_phase78_i8_probe_e4m3(__builtin_nontemporal_load(
                    weight_scales + source_column * blocks_per_row +
                    source_block))
              : 0.0F;
    }
    __syncthreads();

#pragma unroll
    for (uint32_t block = 0U; block < blocks_per_stage; ++block) {
      if (base + static_cast<uint64_t>(block) * kBlockK >= k)
        continue;
      int32_t block_sums[kRowsPerThread][kColumnsPerThread] = {};
#pragma unroll
      for (uint32_t group = 0U; group < kBlockK / 4U; ++group) {
        int32_t activation_packs[kRowsPerThread];
        int32_t weight_packs[kColumnsPerThread];
#pragma unroll
        for (uint32_t row = 0U; row < kRowsPerThread; ++row)
          activation_packs[row] =
              activation_tile[local_row + row * 16U]
                             [block * (kBlockK / 4U) + group];
#pragma unroll
        for (uint32_t column = 0U; column < kColumnsPerThread; ++column)
          weight_packs[column] = weight_tile[local_column + column * 16U]
                                            [block * (kBlockK / 4U) + group];
#pragma unroll
        for (uint32_t row = 0U; row < kRowsPerThread; ++row)
#pragma unroll
          for (uint32_t column = 0U; column < kColumnsPerThread; ++column)
            block_sums[row][column] = sllm_phase78_i8_probe_dot4(
                activation_packs[row], weight_packs[column],
                block_sums[row][column]);
      }
#pragma unroll
      for (uint32_t row = 0U; row < kRowsPerThread; ++row) {
        const float activation_scale =
            activation_scale_tile[local_row + row * 16U][block];
#pragma unroll
        for (uint32_t column = 0U; column < kColumnsPerThread; ++column) {
          const float weight_scale =
              weight_scale_tile[local_column + column * 16U][block];
          accumulators[row][column] +=
              static_cast<float>(block_sums[row][column]) * 0.25F *
              activation_scale * weight_scale;
        }
      }
    }
    __syncthreads();
  }

  const float tensor_scale = weight_tensor_scale[0] * input_tensor_scale[0];
#pragma unroll
  for (uint32_t row = 0U; row < kRowsPerThread; ++row)
#pragma unroll
    for (uint32_t column = 0U; column < kColumnsPerThread; ++column) {
      const uint64_t output_row = row_base + local_row + row * 16U;
      const uint64_t output_column = column_base + local_column + column * 16U;
      if (output_row < m && output_column < n) {
        output[output_row * n + output_column] = sllm_phase78_i8_probe_bf16(
            accumulators[row][column] * tensor_scale);
      }
    }
}

float sllm_phase78_i8_probe_host_e2m1(const uint8_t code) {
  constexpr float values[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  const float value = values[code & 7U];
  return (code & 8U) == 0U ? value : -value;
}

float sllm_phase78_i8_probe_host_e4m3(const uint8_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x80U) << 24U;
  const uint32_t magnitude = bits & 0x7fU;
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  if (exponent == 0U) {
    if (mantissa == 0U)
      return 0.0F;
    const float value = static_cast<float>(mantissa) * 0x1p-9F;
    uint32_t value_bits = 0U;
    std::memcpy(&value_bits, &value, sizeof(value_bits));
    value_bits |= sign;
    float result = 0.0F;
    std::memcpy(&result, &value_bits, sizeof(result));
    return result;
  }
  if (magnitude == 0x7fU)
    return std::numeric_limits<float>::quiet_NaN();
  const uint32_t bits32 = sign | ((exponent + 120U) << 23U) | (mantissa << 20U);
  float result = 0.0F;
  std::memcpy(&result, &bits32, sizeof(result));
  return result;
}

uint16_t sllm_phase78_i8_probe_host_bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000U) == 0x7f800000U) {
    if ((bits & 0x007fffffU) != 0U) {
      const uint16_t sign =
          static_cast<uint16_t>((bits >> 16U) & UINT32_C(0x8000));
      const uint16_t payload =
          static_cast<uint16_t>((bits >> 16U) & UINT32_C(0x003f));
      return static_cast<uint16_t>(sign | UINT16_C(0x7fc0) | payload);
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & 0xffffU;
  if (lower > 0x8000U || (lower == 0x8000U && (upper & UINT32_C(1)) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

float sllm_phase78_i8_probe_host_bf16_to_float(const uint16_t bits) {
  const uint32_t expanded = static_cast<uint32_t>(bits) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &expanded, sizeof(result));
  return result;
}

int8_t sllm_phase78_i8_probe_host_scaled2(const uint8_t code) {
  constexpr int8_t magnitudes[8] = {0, 1, 2, 3, 4, 6, 8, 12};
  const int8_t value = magnitudes[code & 7U];
  return (code & 8U) == 0U ? value : static_cast<int8_t>(-value);
}

struct HostInputs final {
  std::vector<uint8_t> activation;
  std::vector<uint8_t> activation_scales;
  std::vector<uint8_t> weight;
  std::vector<uint8_t> weight_scales;
};

void sllm_phase78_i8_probe_fill_inputs(const uint64_t m, const uint64_t k,
                                       const uint64_t n, HostInputs *const in) {
  const uint64_t blocks = k / kBlockK;
  in->activation.assign(static_cast<size_t>(m * k / 2U), 0U);
  in->activation_scales.assign(static_cast<size_t>(m * blocks), 0x38U);
  in->weight.assign(static_cast<size_t>(n * k / 2U), 0U);
  in->weight_scales.assign(static_cast<size_t>(n * blocks), 0x38U);
  for (uint64_t row = 0U; row < m; ++row) {
    for (uint64_t inner = 0U; inner < k; ++inner) {
      const uint8_t code =
          static_cast<uint8_t>((row * 5U + inner * 3U + 1U) & 0x0fU);
      const size_t index = static_cast<size_t>(row * k / 2U + inner / 2U);
      if ((inner & 1U) == 0U)
        in->activation[index] = code;
      else
        in->activation[index] |= static_cast<uint8_t>(code << 4U);
    }
    for (uint64_t block = 0U; block < blocks; ++block)
      in->activation_scales[static_cast<size_t>(row * blocks + block)] =
          (block & 1U) == 0U ? 0x38U : 0x40U;
  }
  for (uint64_t column = 0U; column < n; ++column) {
    for (uint64_t inner = 0U; inner < k; ++inner) {
      const uint8_t code =
          static_cast<uint8_t>((column * 7U + inner * 9U + 2U) & 0x0fU);
      const size_t index = static_cast<size_t>(column * k / 2U + inner / 2U);
      if ((inner & 1U) == 0U)
        in->weight[index] = code;
      else
        in->weight[index] |= static_cast<uint8_t>(code << 4U);
    }
    for (uint64_t block = 0U; block < blocks; ++block)
      in->weight_scales[static_cast<size_t>(column * blocks + block)] =
          (block & 1U) == 0U ? 0x38U : 0x40U;
  }
}

void sllm_phase78_i8_probe_stage_host(const std::vector<uint8_t> &packed,
                                      const uint64_t rows, const uint64_t k,
                                      std::vector<int8_t> *const staged) {
  staged->assign(static_cast<size_t>(rows * k), 0);
  for (uint64_t row = 0U; row < rows; ++row) {
    for (uint64_t inner = 0U; inner < k; ++inner) {
      const uint8_t pair =
          packed[static_cast<size_t>(row * k / 2U + inner / 2U)];
      const uint8_t code = (inner & 1U) == 0U ? pair & 0x0fU : pair >> 4U;
      (*staged)[static_cast<size_t>(row * k + inner)] =
          sllm_phase78_i8_probe_host_scaled2(code);
    }
  }
}

void sllm_phase78_i8_probe_host_id62(const uint64_t m, const uint64_t k,
                                     const uint64_t n, const HostInputs &in,
                                     std::vector<uint16_t> *const output) {
  output->assign(static_cast<size_t>(m * n), 0U);
  for (uint64_t row = 0U; row < m; ++row) {
    for (uint64_t column = 0U; column < n; ++column) {
      float accumulator = 0.0F;
      for (uint64_t base = 0U; base < k; base += kBlockK) {
        int32_t block_sum = 0;
        for (uint32_t inner = 0U; inner < kBlockK; ++inner) {
          const uint64_t absolute = base + inner;
          const uint8_t ap =
              in.activation[static_cast<size_t>(row * k / 2U + absolute / 2U)];
          const uint8_t wp =
              in.weight[static_cast<size_t>(column * k / 2U + absolute / 2U)];
          const uint8_t ac = (absolute & 1U) == 0U ? ap & 0x0fU : ap >> 4U;
          const uint8_t wc = (absolute & 1U) == 0U ? wp & 0x0fU : wp >> 4U;
          block_sum +=
              static_cast<int32_t>(sllm_phase78_i8_probe_host_scaled2(ac)) *
              static_cast<int32_t>(sllm_phase78_i8_probe_host_scaled2(wc));
        }
        const float activation_scale = sllm_phase78_i8_probe_host_e4m3(
            in.activation_scales[static_cast<size_t>(row * (k / kBlockK) +
                                                     base / kBlockK)]);
        const float weight_scale = sllm_phase78_i8_probe_host_e4m3(
            in.weight_scales[static_cast<size_t>(column * (k / kBlockK) +
                                                 base / kBlockK)]);
        accumulator += static_cast<float>(block_sum) * 0.25F *
                       activation_scale * weight_scale;
      }
      (*output)[static_cast<size_t>(row * n + column)] =
          sllm_phase78_i8_probe_host_bf16(accumulator * 0.75F * 1.125F);
    }
  }
}

bool sllm_phase78_i8_probe_host_oracle() {
  constexpr uint64_t m = 17U;
  constexpr uint64_t k = 48U;
  constexpr uint64_t n = 37U;
  HostInputs in;
  sllm_phase78_i8_probe_fill_inputs(m, k, n, &in);
  std::vector<int8_t> staged_a, staged_w;
  sllm_phase78_i8_probe_stage_host(in.activation, m, k, &staged_a);
  sllm_phase78_i8_probe_stage_host(in.weight, n, k, &staged_w);
  bool mapping_ok = true;
  for (uint32_t code = 0U; code < 16U; ++code) {
    const float expected =
        sllm_phase78_i8_probe_host_e2m1(static_cast<uint8_t>(code)) * 2.0F;
    const float observed = static_cast<float>(
        sllm_phase78_i8_probe_host_scaled2(static_cast<uint8_t>(code)));
    mapping_ok = mapping_ok && expected == observed;
  }
  std::vector<uint16_t> expected;
  sllm_phase78_i8_probe_host_id62(m, k, n, in, &expected);
  // The staged integer representation computes the same block sums.  Use the
  // same scale and BF16 order to make the oracle independent of GPU launch.
  std::vector<uint16_t> observed(static_cast<size_t>(m * n), 0U);
  for (uint64_t row = 0U; row < m; ++row) {
    for (uint64_t column = 0U; column < n; ++column) {
      float accumulator = 0.0F;
      for (uint64_t base = 0U; base < k; base += kBlockK) {
        int32_t block_sum = 0;
        for (uint32_t inner = 0U; inner < kBlockK; ++inner)
          block_sum +=
              static_cast<int32_t>(
                  staged_a[static_cast<size_t>(row * k + base + inner)]) *
              static_cast<int32_t>(
                  staged_w[static_cast<size_t>(column * k + base + inner)]);
        accumulator += static_cast<float>(block_sum) * 0.25F *
                       sllm_phase78_i8_probe_host_e4m3(
                           in.activation_scales[static_cast<size_t>(
                               row * (k / kBlockK) + base / kBlockK)]) *
                       sllm_phase78_i8_probe_host_e4m3(
                           in.weight_scales[static_cast<size_t>(
                               column * (k / kBlockK) + base / kBlockK)]);
      }
      observed[static_cast<size_t>(row * n + column)] =
          sllm_phase78_i8_probe_host_bf16(accumulator * 0.75F * 1.125F);
    }
  }
  size_t mismatches = 0U;
  for (size_t index = 0U; index < expected.size(); ++index)
    mismatches += expected[index] == observed[index] ? 0U : 1U;
  const bool ok = mapping_ok && mismatches == 0U;
  std::printf("host-oracle m=%llu k=%llu n=%llu mapping=%s "
              "bf16_bit_mismatch=%zu status=%s\n",
              static_cast<unsigned long long>(m),
              static_cast<unsigned long long>(k),
              static_cast<unsigned long long>(n), mapping_ok ? "PASS" : "N2",
              mismatches, ok ? "PASS" : "N2");
  return ok;
}

struct DeviceBuffers final {
  uint8_t *packed_activation = nullptr;
  uint8_t *activation_scales = nullptr;
  uint8_t *packed_weight = nullptr;
  uint8_t *weight_scales = nullptr;
  int8_t *staged_activation = nullptr;
  int8_t *staged_weight = nullptr;
  uint16_t *control_output = nullptr;
  uint16_t *candidate_output = nullptr;
  float *weight_tensor_scale = nullptr;
  float *input_tensor_scale = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
};

bool sllm_phase78_i8_probe_hip_ok(const hipError_t status,
                                  const char *const operation) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "hip error operation=%s status=%s\n", operation,
               hipGetErrorString(status));
  return false;
}

bool sllm_phase78_i8_probe_cleanup(DeviceBuffers *const b) {
  if (b == nullptr)
    return true;
  bool ok = true;
  if (b->stop != nullptr)
    ok = sllm_phase78_i8_probe_hip_ok(hipEventDestroy(b->stop), "event stop") &&
         ok;
  if (b->start != nullptr)
    ok = sllm_phase78_i8_probe_hip_ok(hipEventDestroy(b->start),
                                      "event start") &&
         ok;
  if (b->stream != nullptr)
    ok = sllm_phase78_i8_probe_hip_ok(hipStreamDestroy(b->stream), "stream") &&
         ok;
  if (b->candidate_output != nullptr)
    ok = sllm_phase78_i8_probe_hip_ok(hipFree(b->candidate_output),
                                      "candidate output") &&
         ok;
  if (b->control_output != nullptr)
    ok = sllm_phase78_i8_probe_hip_ok(hipFree(b->control_output),
                                      "control output") &&
         ok;
  if (b->input_tensor_scale != nullptr)
    ok = sllm_phase78_i8_probe_hip_ok(hipFree(b->input_tensor_scale),
                                      "input tensor scale") &&
         ok;
  if (b->weight_tensor_scale != nullptr)
    ok = sllm_phase78_i8_probe_hip_ok(hipFree(b->weight_tensor_scale),
                                      "weight tensor scale") &&
         ok;
  if (b->staged_weight != nullptr)
    ok = sllm_phase78_i8_probe_hip_ok(hipFree(b->staged_weight),
                                      "staged weight") &&
         ok;
  if (b->staged_activation != nullptr)
    ok = sllm_phase78_i8_probe_hip_ok(hipFree(b->staged_activation),
                                      "staged activation") &&
         ok;
  if (b->weight_scales != nullptr)
    ok = sllm_phase78_i8_probe_hip_ok(hipFree(b->weight_scales),
                                      "weight scales") &&
         ok;
  if (b->packed_weight != nullptr)
    ok = sllm_phase78_i8_probe_hip_ok(hipFree(b->packed_weight),
                                      "packed weight") &&
         ok;
  if (b->activation_scales != nullptr)
    ok = sllm_phase78_i8_probe_hip_ok(hipFree(b->activation_scales),
                                      "activation scales") &&
         ok;
  if (b->packed_activation != nullptr)
    ok = sllm_phase78_i8_probe_hip_ok(hipFree(b->packed_activation),
                                      "packed activation") &&
         ok;
  *b = {};
  return ok;
}

bool sllm_phase78_i8_probe_make_buffers(const uint64_t m, const uint64_t k,
                                        const uint64_t n,
                                        DeviceBuffers *const b) {
  if (b == nullptr || m == 0U || n == 0U || k == 0U || (k % 16U) != 0U)
    return false;
  const uint64_t packed_a = m * k / 2U;
  const uint64_t packed_w = n * k / 2U;
  const uint64_t scale_a = m * (k / kBlockK);
  const uint64_t scale_w = n * (k / kBlockK);
  const uint64_t output = m * n * sizeof(uint16_t);
  return sllm_phase78_i8_probe_hip_ok(
             hipMalloc(reinterpret_cast<void **>(&b->packed_activation),
                       packed_a),
             "packed activation") &&
         sllm_phase78_i8_probe_hip_ok(
             hipMalloc(reinterpret_cast<void **>(&b->activation_scales),
                       scale_a),
             "activation scales") &&
         sllm_phase78_i8_probe_hip_ok(
             hipMalloc(reinterpret_cast<void **>(&b->packed_weight), packed_w),
             "packed weight") &&
         sllm_phase78_i8_probe_hip_ok(
             hipMalloc(reinterpret_cast<void **>(&b->weight_scales), scale_w),
             "weight scales") &&
         sllm_phase78_i8_probe_hip_ok(
             hipMalloc(reinterpret_cast<void **>(&b->staged_activation), m * k),
             "staged activation") &&
         sllm_phase78_i8_probe_hip_ok(
             hipMalloc(reinterpret_cast<void **>(&b->staged_weight), n * k),
             "staged weight") &&
         sllm_phase78_i8_probe_hip_ok(
             hipMalloc(reinterpret_cast<void **>(&b->control_output), output),
             "control output") &&
         sllm_phase78_i8_probe_hip_ok(
             hipMalloc(reinterpret_cast<void **>(&b->candidate_output), output),
             "candidate output") &&
         sllm_phase78_i8_probe_hip_ok(
             hipMalloc(reinterpret_cast<void **>(&b->weight_tensor_scale),
                       sizeof(float)),
             "weight tensor scale") &&
         sllm_phase78_i8_probe_hip_ok(
             hipMalloc(reinterpret_cast<void **>(&b->input_tensor_scale),
                       sizeof(float)),
             "input tensor scale") &&
         sllm_phase78_i8_probe_hip_ok(hipStreamCreate(&b->stream), "stream") &&
         sllm_phase78_i8_probe_hip_ok(hipEventCreate(&b->start),
                                      "event start") &&
         sllm_phase78_i8_probe_hip_ok(hipEventCreate(&b->stop), "event stop");
}

bool sllm_phase78_i8_probe_upload(const uint64_t m, const uint64_t k,
                                  const uint64_t n, const HostInputs &in,
                                  DeviceBuffers *const b) {
  const float weight_tensor_scale = 0.75F;
  const float input_tensor_scale = 1.125F;
  return sllm_phase78_i8_probe_hip_ok(
             hipMemcpy(b->packed_activation, in.activation.data(), m * k / 2U,
                       hipMemcpyHostToDevice),
             "upload activation") &&
         sllm_phase78_i8_probe_hip_ok(
             hipMemcpy(b->activation_scales, in.activation_scales.data(),
                       m * (k / kBlockK), hipMemcpyHostToDevice),
             "upload activation scales") &&
         sllm_phase78_i8_probe_hip_ok(hipMemcpy(b->packed_weight,
                                                in.weight.data(), n * k / 2U,
                                                hipMemcpyHostToDevice),
                                      "upload weight") &&
         sllm_phase78_i8_probe_hip_ok(
             hipMemcpy(b->weight_scales, in.weight_scales.data(),
                       n * (k / kBlockK), hipMemcpyHostToDevice),
             "upload weight scales") &&
         sllm_phase78_i8_probe_hip_ok(
             hipMemcpy(b->weight_tensor_scale, &weight_tensor_scale,
                       sizeof(float), hipMemcpyHostToDevice),
             "upload weight tensor scale") &&
         sllm_phase78_i8_probe_hip_ok(
             hipMemcpy(b->input_tensor_scale, &input_tensor_scale,
                       sizeof(float), hipMemcpyHostToDevice),
             "upload input tensor scale");
}

uint32_t sllm_phase78_i8_probe_grid(const uint64_t m, const uint64_t n) {
  const uint64_t tiles =
      ((m + kTileM - 1U) / kTileM) * ((n + kTileN - 1U) / kTileN);
  return static_cast<uint32_t>(tiles);
}

bool sllm_phase78_i8_probe_launch_control(const uint64_t m, const uint64_t k,
                                          const uint64_t n,
                                          DeviceBuffers *const b) {
  hipLaunchKernelGGL(sllm_phase78_i8_probe_id62_control,
                     dim3(sllm_phase78_i8_probe_grid(m, n)), dim3(kThreads), 0U,
                     b->stream, b->packed_activation, b->activation_scales,
                     b->packed_weight, b->weight_scales, b->weight_tensor_scale,
                     b->input_tensor_scale, b->control_output, m, k, n);
  return hipGetLastError() == hipSuccess;
}

bool sllm_phase78_i8_probe_launch_stage(const uint64_t m, const uint64_t k,
                                        const uint64_t n,
                                        DeviceBuffers *const b) {
  const uint64_t activation_chunks = m * (k / 8U);
  const uint64_t weight_chunks = n * (k / 8U);
  hipLaunchKernelGGL(sllm_phase78_i8_probe_stage_packed,
                     dim3(static_cast<uint32_t>(
                         (activation_chunks + kThreads - 1U) / kThreads)),
                     dim3(kThreads), 0U, b->stream, b->packed_activation,
                     b->staged_activation, m, k);
  hipLaunchKernelGGL(
      sllm_phase78_i8_probe_stage_packed,
      dim3(static_cast<uint32_t>((weight_chunks + kThreads - 1U) / kThreads)),
      dim3(kThreads), 0U, b->stream, b->packed_weight, b->staged_weight, n, k);
  return hipGetLastError() == hipSuccess;
}

bool sllm_phase78_i8_probe_launch_candidate(const uint64_t m, const uint64_t k,
                                            const uint64_t n,
                                            DeviceBuffers *const b) {
  hipLaunchKernelGGL(sllm_phase78_i8_probe_staged_matmul,
                     dim3(sllm_phase78_i8_probe_grid(m, n)), dim3(kThreads), 0U,
                     b->stream, b->staged_activation, b->activation_scales,
                     b->staged_weight, b->weight_scales, b->weight_tensor_scale,
                     b->input_tensor_scale, b->candidate_output, m, k, n);
  return hipGetLastError() == hipSuccess;
}

bool sllm_phase78_i8_probe_common_prewarm(const uint64_t m, const uint64_t k,
                                          const uint64_t n,
                                          DeviceBuffers *const b) {
  // Put both paths through the same short alternating warmup before any
  // samples.  This reduces clock/order bias; the reported candidate interval
  // still includes both staging launches and the staged matmul launch.
  for (uint32_t iteration = 0U; iteration < kWarmups; ++iteration) {
    const bool candidate_first = (iteration & 1U) != 0U;
    const bool first =
        candidate_first ? sllm_phase78_i8_probe_launch_stage(m, k, n, b) &&
                              sllm_phase78_i8_probe_launch_candidate(m, k, n, b)
                        : sllm_phase78_i8_probe_launch_control(m, k, n, b);
    const bool second =
        candidate_first
            ? sllm_phase78_i8_probe_launch_control(m, k, n, b)
            : sllm_phase78_i8_probe_launch_stage(m, k, n, b) &&
                  sllm_phase78_i8_probe_launch_candidate(m, k, n, b);
    if (!first || !second ||
        !sllm_phase78_i8_probe_hip_ok(hipStreamSynchronize(b->stream),
                                      "common prewarm"))
      return false;
  }
  return true;
}

bool sllm_phase78_i8_probe_measure_control(const uint64_t m, const uint64_t k,
                                           const uint64_t n,
                                           DeviceBuffers *const b,
                                           float *const median_us) {
  for (uint32_t i = 0U; i < kWarmups; ++i)
    if (!sllm_phase78_i8_probe_launch_control(m, k, n, b) ||
        !sllm_phase78_i8_probe_hip_ok(hipStreamSynchronize(b->stream),
                                      "control warmup"))
      return false;
  std::array<float, kMeasured> samples{};
  for (uint32_t i = 0U; i < kMeasured; ++i) {
    if (!sllm_phase78_i8_probe_hip_ok(hipEventRecord(b->start, b->stream),
                                      "control start") ||
        !sllm_phase78_i8_probe_launch_control(m, k, n, b) ||
        !sllm_phase78_i8_probe_hip_ok(hipEventRecord(b->stop, b->stream),
                                      "control stop") ||
        !sllm_phase78_i8_probe_hip_ok(hipEventSynchronize(b->stop),
                                      "control event") ||
        !sllm_phase78_i8_probe_hip_ok(
            hipEventElapsedTime(&samples[i], b->start, b->stop),
            "control elapsed"))
      return false;
    samples[i] *= 1000.0F;
  }
  std::sort(samples.begin(), samples.end());
  *median_us = samples[kMeasured / 2U];
  return true;
}

bool sllm_phase78_i8_probe_measure_pipeline(const uint64_t m, const uint64_t k,
                                            const uint64_t n,
                                            DeviceBuffers *const b,
                                            float *const total_us,
                                            float *const stage_us,
                                            float *const matmul_us) {
  for (uint32_t i = 0U; i < kWarmups; ++i)
    if (!sllm_phase78_i8_probe_launch_stage(m, k, n, b) ||
        !sllm_phase78_i8_probe_launch_candidate(m, k, n, b) ||
        !sllm_phase78_i8_probe_hip_ok(hipStreamSynchronize(b->stream),
                                      "candidate warmup"))
      return false;
  std::array<float, kMeasured> total{}, stage{}, matmul{};
  for (uint32_t i = 0U; i < kMeasured; ++i) {
    if (!sllm_phase78_i8_probe_hip_ok(hipEventRecord(b->start, b->stream),
                                      "candidate start") ||
        !sllm_phase78_i8_probe_launch_stage(m, k, n, b) ||
        !sllm_phase78_i8_probe_launch_candidate(m, k, n, b) ||
        !sllm_phase78_i8_probe_hip_ok(hipEventRecord(b->stop, b->stream),
                                      "candidate stop") ||
        !sllm_phase78_i8_probe_hip_ok(hipEventSynchronize(b->stop),
                                      "candidate event") ||
        !sllm_phase78_i8_probe_hip_ok(
            hipEventElapsedTime(&total[i], b->start, b->stop),
            "candidate elapsed"))
      return false;
    total[i] *= 1000.0F;
    // The separate components are measured in the same stream immediately
    // after the total sample.  They are diagnostic; total_us is the decision
    // metric because staging is transient and must be paid by each call.
    if (!sllm_phase78_i8_probe_hip_ok(hipEventRecord(b->start, b->stream),
                                      "stage start") ||
        !sllm_phase78_i8_probe_launch_stage(m, k, n, b) ||
        !sllm_phase78_i8_probe_hip_ok(hipEventRecord(b->stop, b->stream),
                                      "stage stop") ||
        !sllm_phase78_i8_probe_hip_ok(hipEventSynchronize(b->stop),
                                      "stage event") ||
        !sllm_phase78_i8_probe_hip_ok(
            hipEventElapsedTime(&stage[i], b->start, b->stop), "stage elapsed"))
      return false;
    stage[i] *= 1000.0F;
    if (!sllm_phase78_i8_probe_hip_ok(hipEventRecord(b->start, b->stream),
                                      "matmul start") ||
        !sllm_phase78_i8_probe_launch_candidate(m, k, n, b) ||
        !sllm_phase78_i8_probe_hip_ok(hipEventRecord(b->stop, b->stream),
                                      "matmul stop") ||
        !sllm_phase78_i8_probe_hip_ok(hipEventSynchronize(b->stop),
                                      "matmul event") ||
        !sllm_phase78_i8_probe_hip_ok(
            hipEventElapsedTime(&matmul[i], b->start, b->stop),
            "matmul elapsed"))
      return false;
    matmul[i] *= 1000.0F;
  }
  std::sort(total.begin(), total.end());
  std::sort(stage.begin(), stage.end());
  std::sort(matmul.begin(), matmul.end());
  *total_us = total[kMeasured / 2U];
  *stage_us = stage[kMeasured / 2U];
  *matmul_us = matmul[kMeasured / 2U];
  return true;
}

bool sllm_phase78_i8_probe_compare(const std::vector<uint16_t> &lhs,
                                   const std::vector<uint16_t> &rhs,
                                   const char *const name) {
  size_t mismatches = 0U;
  size_t max_bf16_distance = 0U;
  double max_abs = 0.0;
  for (size_t index = 0U; index < lhs.size(); ++index) {
    if (lhs[index] != rhs[index])
      ++mismatches;
    max_bf16_distance =
        std::max(max_bf16_distance,
                 static_cast<size_t>(std::abs(static_cast<int>(lhs[index]) -
                                              static_cast<int>(rhs[index]))));
    max_abs = std::max(
        max_abs,
        std::abs(static_cast<double>(
                     sllm_phase78_i8_probe_host_bf16_to_float(lhs[index])) -
                 static_cast<double>(
                     sllm_phase78_i8_probe_host_bf16_to_float(rhs[index]))));
  }
  std::printf("compare candidate=%s elements=%zu bf16_bit_mismatch=%zu "
              "max_bf16_distance=%zu max_abs=%.8g status=%s\n",
              name, lhs.size(), mismatches, max_bf16_distance, max_abs,
              mismatches == 0U ? "PASS" : "N2");
  return mismatches == 0U;
}

bool sllm_phase78_i8_probe_run_shape(const uint64_t m, const uint64_t k,
                                     const uint64_t n) {
  HostInputs in;
  sllm_phase78_i8_probe_fill_inputs(m, k, n, &in);
  DeviceBuffers b;
  if (!sllm_phase78_i8_probe_make_buffers(m, k, n, &b) ||
      !sllm_phase78_i8_probe_upload(m, k, n, in, &b)) {
    sllm_phase78_i8_probe_cleanup(&b);
    return false;
  }
  float control_us = 0.0F, total_us = 0.0F, stage_us = 0.0F;
  float matmul_us = 0.0F;
  const bool measured =
      sllm_phase78_i8_probe_common_prewarm(m, k, n, &b) &&
      sllm_phase78_i8_probe_measure_control(m, k, n, &b, &control_us) &&
      sllm_phase78_i8_probe_measure_pipeline(m, k, n, &b, &total_us, &stage_us,
                                             &matmul_us);
  std::vector<uint16_t> control(static_cast<size_t>(m * n));
  std::vector<uint16_t> candidate(static_cast<size_t>(m * n));
  bool copied =
      measured && sllm_phase78_i8_probe_launch_control(m, k, n, &b) &&
      sllm_phase78_i8_probe_hip_ok(hipDeviceSynchronize(), "control compare") &&
      sllm_phase78_i8_probe_hip_ok(hipMemcpy(control.data(), b.control_output,
                                             control.size() * sizeof(uint16_t),
                                             hipMemcpyDeviceToHost),
                                   "control copy") &&
      sllm_phase78_i8_probe_launch_stage(m, k, n, &b) &&
      sllm_phase78_i8_probe_launch_candidate(m, k, n, &b) &&
      sllm_phase78_i8_probe_hip_ok(hipDeviceSynchronize(),
                                   "candidate compare") &&
      sllm_phase78_i8_probe_hip_ok(
          hipMemcpy(candidate.data(), b.candidate_output,
                    candidate.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
          "candidate copy");
  const bool bitwise = copied && sllm_phase78_i8_probe_compare(
                                     control, candidate, "staged-vs-ID62");
  const uint64_t staged_bytes = m * k + n * k;
  const uint64_t source_bytes = m * k / 2U + n * k / 2U;
  std::printf(
      "result m=%llu k=%llu n=%llu control_us=%.3f stage_us=%.3f "
      "matmul_us=%.3f candidate_total_us=%.3f speedup=%.6f "
      "source_bytes=%llu staged_workspace_bytes=%llu status=%s\n",
      static_cast<unsigned long long>(m), static_cast<unsigned long long>(k),
      static_cast<unsigned long long>(n), control_us, stage_us, matmul_us,
      total_us,
      total_us > 0.0F ? static_cast<double>(control_us) / total_us : 0.0,
      static_cast<unsigned long long>(source_bytes),
      static_cast<unsigned long long>(staged_bytes),
      measured && bitwise ? "PASS" : "N2");
  const bool cleaned = sllm_phase78_i8_probe_cleanup(&b);
  return measured && bitwise && cleaned;
}

bool sllm_phase78_i8_probe_exact_gfx1030(const char *const arch) {
  return arch != nullptr &&
         std::string_view(arch).compare(0U, 7U, "gfx1030") == 0;
}

} // namespace

int main() {
  const bool oracle_ok = sllm_phase78_i8_probe_host_oracle();
  const char *const run =
      std::getenv("SLLM_PHASE78_NVFP4_GFX1030_I8_STAGING_RUN");
  if (run == nullptr || std::string_view(run) != "1") {
    std::printf("gpu=SKIPPED reason=set "
                "SLLM_PHASE78_NVFP4_GFX1030_I8_STAGING_RUN=1; "
                "compile-only/default mode\n");
    return oracle_ok ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  if (!sllm_phase78_i8_probe_hip_ok(hipSetDevice(0), "hipSetDevice"))
    return EXIT_FAILURE;
  hipDeviceProp_t properties{};
  if (!sllm_phase78_i8_probe_hip_ok(hipGetDeviceProperties(&properties, 0),
                                    "hipGetDeviceProperties") ||
      !sllm_phase78_i8_probe_exact_gfx1030(properties.gcnArchName)) {
    std::fprintf(stderr, "exact gfx1030 required; got %s\n",
                 properties.gcnArchName);
    return EXIT_FAILURE;
  }
  std::printf("target=%s device=0 pci=%04x:%02x:%02x name=%s\n",
              properties.gcnArchName, properties.pciDomainID,
              properties.pciBusID, properties.pciDeviceID, properties.name);
  bool all_ok = oracle_ok;
  // Default GPU set is bounded: tiny/tail plus both wide/down core M values.
  // Set *_ALL=1 only when a follow-up explicitly requests the extra tails.
  const bool all_shapes =
      std::getenv("SLLM_PHASE78_NVFP4_GFX1030_I8_STAGING_ALL") != nullptr;
  std::vector<std::array<uint64_t, 3>> shapes = {
      {17U, 32U, 37U},       {17U, 48U, 37U},        {128U, 5120U, 17408U},
      {128U, 17408U, 5120U}, {1024U, 5120U, 17408U}, {1024U, 17408U, 5120U}};
  if (all_shapes) {
    shapes.insert(shapes.begin() + 2, {{127U, 5120U, 17408U},
                                       {129U, 5120U, 17408U},
                                       {219U, 5120U, 17408U},
                                       {512U, 5120U, 17408U}});
  }
  for (const auto &shape : shapes)
    all_ok =
        sllm_phase78_i8_probe_run_shape(shape[0], shape[1], shape[2]) && all_ok;
  std::printf("summary status=%s warmups=%u measured=%u\n",
              all_ok ? "PASS" : "N2", kWarmups, kMeasured);
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
