// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase78-nvfp4-byte-permute-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// 3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70, ggml/src/ggml-cuda/vecdotq.cuh
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

// Phase 78 standalone gfx1201 NVFP4 decode dot-product probe.
//
// This developer evidence tool compares the current signed-byte DP4A recipe,
// the gfx12 mixed-sign sudot4 instruction, and the scalar FP8 dot4 builtin.
// It deliberately does not enter the production build.  The input contract
// is the exact NVFP4 W4A4 block16 layout used by the Phase 78 Qwen3.8 probe:
// eight packed E2M1 bytes per block16, one positive E4M3FN block scale per
// block, and one output per weight column.

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
constexpr uint32_t kWaveWidth = 32U;
constexpr uint32_t kWaves = kThreads / kWaveWidth;
constexpr int kWarmups = 3;
constexpr int kMeasured = 10;

enum class DotKind : uint32_t { Sdot4, Sudot4, Fp8Dot4 };

struct PackedI8Pair final {
  int32_t even;
  int32_t odd;
};

// Every E2M1 value is exactly representable by OCP E4M3FN.  This is kept as
// a constexpr host/device transform so the register-pack oracle and the
// device FP8 dot path share one mapping.
__host__ __device__ constexpr uint8_t e2m1_to_e4m3fn(const uint8_t code) {
  constexpr uint8_t map[8] = {0x00U, 0x30U, 0x38U, 0x3cU,
                              0x40U, 0x44U, 0x48U, 0x4cU};
  const uint8_t magnitude = map[code & 0x07U];
  return static_cast<uint8_t>(magnitude | (code & 0x08U) << 4U);
}

__host__ __device__ inline uint32_t e2m1x4_to_e4m3fn(const uint16_t packed) {
#if defined(__HIP_DEVICE_COMPILE__)
  // Expand packed nibbles into byte lanes, then use one v_perm_b32 for the
  // positive E4M3 magnitudes.  The sign bit is copied independently; this is
  // the intended register-only FP8 ingress (no table load per nibble).
  const uint32_t lanes =
      (static_cast<uint32_t>(packed) & UINT32_C(0x000f)) |
      ((static_cast<uint32_t>(packed) & UINT32_C(0x00f0)) << 4U) |
      ((static_cast<uint32_t>(packed) & UINT32_C(0x0f00)) << 8U) |
      ((static_cast<uint32_t>(packed) & UINT32_C(0xf000)) << 12U);
  constexpr uint32_t positive_0_3 = UINT32_C(0x3c383000);
  constexpr uint32_t positive_4_7 = UINT32_C(0x4c484440);
  constexpr uint32_t low_index_mask = UINT32_C(0x07070707);
  const uint32_t magnitude =
      __builtin_amdgcn_perm(positive_4_7, positive_0_3, lanes & low_index_mask);
  return magnitude | ((lanes & UINT32_C(0x08080808)) << 4U);
#else
  uint32_t result = 0U;
  for (uint32_t lane = 0U; lane < 4U; ++lane) {
    result |= static_cast<uint32_t>(e2m1_to_e4m3fn(
                  static_cast<uint8_t>((packed >> (lane * 4U)) & 0x0fU)))
              << (lane * 8U);
  }
  return result;
#endif
}

__host__ __device__ constexpr float e2m1_value(const uint8_t code) {
  constexpr float values[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  const float value = values[code & 0x07U];
  return (code & 0x08U) == 0U ? value : -value;
}

// This is the same AMD byte-permute packing used by the current production
// sdot4 provider.  Values are multiplied by two to remove the E2M1 half-step;
// four signed-byte products are therefore divided by four at block epilogue.
__device__ __forceinline__ PackedI8Pair
e2m1x8_scaled2_to_i8x4_pair(const uint32_t packed) noexcept {
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
  return PackedI8Pair{
      static_cast<int32_t>(
          __builtin_amdgcn_perm(even_high, even_low, even_select)),
      static_cast<int32_t>(
          __builtin_amdgcn_perm(odd_high, odd_low, odd_select)),
  };
}

__device__ __forceinline__ int32_t dot_i8x4(const int32_t lhs,
                                            const int32_t rhs,
                                            const int32_t accumulator,
                                            const DotKind kind) noexcept {
  if (kind == DotKind::Sudot4) {
#if __has_builtin(__builtin_amdgcn_sudot4)
    // Both sign selectors are immediate true: this is the signed/signed
    // canary, with the same byte interpretation as sdot4.
    return __builtin_amdgcn_sudot4(true, lhs, true, rhs, accumulator, false);
#else
    return accumulator;
#endif
  }
#if __has_builtin(__builtin_amdgcn_sdot4)
  return __builtin_amdgcn_sdot4(lhs, rhs, accumulator, false);
#else
  int32_t result = accumulator;
#pragma unroll
  for (uint32_t lane = 0U; lane < 4U; ++lane) {
    const int32_t a = static_cast<int8_t>(lhs >> (lane * 8U));
    const int32_t b = static_cast<int8_t>(rhs >> (lane * 8U));
    result += a * b;
  }
  return result;
#endif
}

template <DotKind Kind>
__device__ __forceinline__ float
nvfp4_block_dot(const uint8_t *const activation,
                const uint8_t *const activation_scales,
                const uint8_t *const weight, const uint8_t *const weight_scales,
                const uint32_t block, const uint32_t blocks_per_row) noexcept {
  const uint8_t *const activation_block = activation + block * 8U;
  const uint8_t *const weight_block = weight + block * 8U;
  const float activation_scale =
      sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode(
          activation_scales[block]);
  const float weight_scale =
      sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode(weight_scales[block]);
  if constexpr (Kind == DotKind::Fp8Dot4) {
    float result = 0.0F;
#pragma unroll
    for (uint32_t group = 0U; group < 4U; ++group) {
      const uint16_t packed_activation = __builtin_nontemporal_load(
          reinterpret_cast<const uint16_t *>(activation_block + group * 2U));
      const uint16_t packed_weight = __builtin_nontemporal_load(
          reinterpret_cast<const uint16_t *>(weight_block + group * 2U));
#if __has_builtin(__builtin_amdgcn_dot4_f32_fp8_fp8)
      result = __builtin_amdgcn_dot4_f32_fp8_fp8(
          e2m1x4_to_e4m3fn(packed_activation), e2m1x4_to_e4m3fn(packed_weight),
          result);
#else
      (void)packed_activation;
      (void)packed_weight;
#endif
    }
    return result * activation_scale * weight_scale;
  }

  const uint32_t activation_word0 = __builtin_nontemporal_load(
      reinterpret_cast<const uint32_t *>(activation_block));
  const uint32_t activation_word1 = __builtin_nontemporal_load(
      reinterpret_cast<const uint32_t *>(activation_block + 4U));
  const uint32_t weight_word0 = __builtin_nontemporal_load(
      reinterpret_cast<const uint32_t *>(weight_block));
  const uint32_t weight_word1 = __builtin_nontemporal_load(
      reinterpret_cast<const uint32_t *>(weight_block + 4U));
  const PackedI8Pair activation_pack0 =
      e2m1x8_scaled2_to_i8x4_pair(activation_word0);
  const PackedI8Pair activation_pack1 =
      e2m1x8_scaled2_to_i8x4_pair(activation_word1);
  const PackedI8Pair weight_pack0 = e2m1x8_scaled2_to_i8x4_pair(weight_word0);
  const PackedI8Pair weight_pack1 = e2m1x8_scaled2_to_i8x4_pair(weight_word1);
  int32_t sum = 0;
  sum = dot_i8x4(activation_pack0.even, weight_pack0.even, sum, Kind);
  sum = dot_i8x4(activation_pack0.odd, weight_pack0.odd, sum, Kind);
  sum = dot_i8x4(activation_pack1.even, weight_pack1.even, sum, Kind);
  sum = dot_i8x4(activation_pack1.odd, weight_pack1.odd, sum, Kind);
  (void)blocks_per_row;
  return static_cast<float>(sum) * 0.25F * activation_scale * weight_scale;
}

template <DotKind Kind, bool StoreSubtotals, bool KVector2>
__global__ __launch_bounds__(kThreads, 1) void nvfp4_decode_dot_kernel(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    float *const output, float *const block_subtotals,
    const uint32_t blocks_per_row, const uint32_t n) {
  const uint32_t lane = threadIdx.x & (kWaveWidth - 1U);
  const uint32_t wave = threadIdx.x / kWaveWidth;
  const uint32_t column = blockIdx.x * kWaves + wave;
  if (column >= n) {
    return;
  }
  const uint8_t *const weight_row =
      weight + static_cast<uint64_t>(column) * blocks_per_row * 8U;
  const uint8_t *const weight_scale_row =
      weight_scales + static_cast<uint64_t>(column) * blocks_per_row;
  float partial = 0.0F;
  if constexpr (KVector2) {
    for (uint32_t block = lane * 2U; block < blocks_per_row; block += 64U) {
      const float subtotal =
          nvfp4_block_dot<Kind>(activation, activation_scales, weight_row,
                                weight_scale_row, block, blocks_per_row);
      if constexpr (StoreSubtotals) {
        block_subtotals[static_cast<uint64_t>(column) * blocks_per_row +
                        block] = subtotal;
      }
      partial += subtotal;
      if (block + 1U < blocks_per_row) {
        const float second =
            nvfp4_block_dot<Kind>(activation, activation_scales, weight_row,
                                  weight_scale_row, block + 1U, blocks_per_row);
        if constexpr (StoreSubtotals) {
          block_subtotals[static_cast<uint64_t>(column) * blocks_per_row +
                          block + 1U] = second;
        }
        partial += second;
      }
    }
  } else {
    for (uint32_t block = lane; block < blocks_per_row; block += kWaveWidth) {
      const float subtotal =
          nvfp4_block_dot<Kind>(activation, activation_scales, weight_row,
                                weight_scale_row, block, blocks_per_row);
      if constexpr (StoreSubtotals) {
        block_subtotals[static_cast<uint64_t>(column) * blocks_per_row +
                        block] = subtotal;
      }
      partial += subtotal;
    }
  }

#pragma unroll
  for (uint32_t offset = kWaveWidth / 2U; offset != 0U; offset >>= 1U) {
    partial += __shfl_down(partial, offset, kWaveWidth);
  }
  if (lane == 0U) {
    output[column] = partial;
  }
}

// Production-shape decode geometry: one workgroup owns 32 adjacent output
// columns, each wave owns four columns, and each lane walks block16 indices
// strided by 32.  This matches the Phase 78 ID67 launch geometry.  The LDS
// variant decodes the activation row once for the whole workgroup and shares
// the resulting four dwords plus FP32 block scale across all eight waves.
template <DotKind Kind, bool StoreSubtotals, bool ActivationLds>
__global__ __launch_bounds__(kThreads, 1) void nvfp4_decode_wave4col32_kernel(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    float *const output, float *const block_subtotals,
    const uint32_t blocks_per_row, const uint32_t n) {
  extern __shared__ uint8_t shared[];
  uint8_t *const activation_tile = shared;
  float *const activation_scale_tile = reinterpret_cast<float *>(
      shared + static_cast<uint64_t>(blocks_per_row) * 16U);
  if constexpr (ActivationLds) {
    auto *const decoded_words = reinterpret_cast<uint32_t *>(activation_tile);
    for (uint32_t block = threadIdx.x; block < blocks_per_row;
         block += kThreads) {
      const uint8_t *const source =
          activation + static_cast<uint64_t>(block) * 8U;
      if constexpr (Kind == DotKind::Fp8Dot4) {
#pragma unroll
        for (uint32_t group = 0U; group < 4U; ++group) {
          const uint16_t packed = __builtin_nontemporal_load(
              reinterpret_cast<const uint16_t *>(source + group * 2U));
          decoded_words[static_cast<uint64_t>(block) * 4U + group] =
              e2m1x4_to_e4m3fn(packed);
        }
      } else {
        const uint32_t word0 = __builtin_nontemporal_load(
            reinterpret_cast<const uint32_t *>(source));
        const uint32_t word1 = __builtin_nontemporal_load(
            reinterpret_cast<const uint32_t *>(source + 4U));
        const PackedI8Pair first = e2m1x8_scaled2_to_i8x4_pair(word0);
        const PackedI8Pair second = e2m1x8_scaled2_to_i8x4_pair(word1);
        decoded_words[static_cast<uint64_t>(block) * 4U + 0U] =
            static_cast<uint32_t>(first.even);
        decoded_words[static_cast<uint64_t>(block) * 4U + 1U] =
            static_cast<uint32_t>(first.odd);
        decoded_words[static_cast<uint64_t>(block) * 4U + 2U] =
            static_cast<uint32_t>(second.even);
        decoded_words[static_cast<uint64_t>(block) * 4U + 3U] =
            static_cast<uint32_t>(second.odd);
      }
      activation_scale_tile[block] =
          sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode(
              activation_scales[block]);
    }
    __syncthreads();
  }

  const uint32_t lane = threadIdx.x & (kWaveWidth - 1U);
  const uint32_t wave = threadIdx.x / kWaveWidth;
  const uint32_t column_base = blockIdx.x * 32U + wave * 4U;
  float accumulators[4] = {};
  const uint64_t packed_row_bytes = static_cast<uint64_t>(blocks_per_row) * 8U;

  for (uint32_t block = lane; block < blocks_per_row; block += kWaveWidth) {
    uint32_t activation_words[4];
    float activation_scale = 0.0F;
    if constexpr (ActivationLds) {
      const auto *const decoded_words =
          reinterpret_cast<const uint32_t *>(activation_tile) + block * 4U;
#pragma unroll
      for (uint32_t group = 0U; group < 4U; ++group) {
        activation_words[group] = decoded_words[group];
      }
      activation_scale = activation_scale_tile[block];
    } else if constexpr (Kind == DotKind::Fp8Dot4) {
      const uint8_t *const source =
          activation + static_cast<uint64_t>(block) * 8U;
#pragma unroll
      for (uint32_t group = 0U; group < 4U; ++group) {
        const uint16_t packed = __builtin_nontemporal_load(
            reinterpret_cast<const uint16_t *>(source + group * 2U));
        activation_words[group] = e2m1x4_to_e4m3fn(packed);
      }
      activation_scale = sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode(
          activation_scales[block]);
    } else {
      const uint8_t *const source =
          activation + static_cast<uint64_t>(block) * 8U;
      const uint32_t word0 = __builtin_nontemporal_load(
          reinterpret_cast<const uint32_t *>(source));
      const uint32_t word1 = __builtin_nontemporal_load(
          reinterpret_cast<const uint32_t *>(source + 4U));
      const PackedI8Pair first = e2m1x8_scaled2_to_i8x4_pair(word0);
      const PackedI8Pair second = e2m1x8_scaled2_to_i8x4_pair(word1);
      activation_words[0] = static_cast<uint32_t>(first.even);
      activation_words[1] = static_cast<uint32_t>(first.odd);
      activation_words[2] = static_cast<uint32_t>(second.even);
      activation_words[3] = static_cast<uint32_t>(second.odd);
      activation_scale = sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode(
          activation_scales[block]);
    }

#pragma unroll
    for (uint32_t column_offset = 0U; column_offset < 4U; ++column_offset) {
      const uint32_t column = column_base + column_offset;
      if (column >= n) {
        continue;
      }
      const uint8_t *const weight_block =
          weight + static_cast<uint64_t>(column) * packed_row_bytes +
          static_cast<uint64_t>(block) * 8U;
      const uint8_t *const weight_scale_row =
          weight_scales + static_cast<uint64_t>(column) * blocks_per_row;
      const float weight_scale =
          sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode(
              weight_scale_row[block]);
      float subtotal = 0.0F;
      if constexpr (Kind == DotKind::Fp8Dot4) {
#pragma unroll
        for (uint32_t group = 0U; group < 4U; ++group) {
          const uint16_t packed = __builtin_nontemporal_load(
              reinterpret_cast<const uint16_t *>(weight_block + group * 2U));
#if __has_builtin(__builtin_amdgcn_dot4_f32_fp8_fp8)
          subtotal = __builtin_amdgcn_dot4_f32_fp8_fp8(
              activation_words[group], e2m1x4_to_e4m3fn(packed), subtotal);
#else
          (void)packed;
#endif
        }
        subtotal *= activation_scale * weight_scale;
      } else {
        const uint32_t word0 = __builtin_nontemporal_load(
            reinterpret_cast<const uint32_t *>(weight_block));
        const uint32_t word1 = __builtin_nontemporal_load(
            reinterpret_cast<const uint32_t *>(weight_block + 4U));
        const PackedI8Pair first = e2m1x8_scaled2_to_i8x4_pair(word0);
        const PackedI8Pair second = e2m1x8_scaled2_to_i8x4_pair(word1);
        int32_t integer_sum = 0;
        integer_sum = dot_i8x4(static_cast<int32_t>(activation_words[0]),
                               first.even, integer_sum, Kind);
        integer_sum = dot_i8x4(static_cast<int32_t>(activation_words[1]),
                               first.odd, integer_sum, Kind);
        integer_sum = dot_i8x4(static_cast<int32_t>(activation_words[2]),
                               second.even, integer_sum, Kind);
        integer_sum = dot_i8x4(static_cast<int32_t>(activation_words[3]),
                               second.odd, integer_sum, Kind);
        subtotal = static_cast<float>(integer_sum) * 0.25F * activation_scale *
                   weight_scale;
      }
      if constexpr (StoreSubtotals) {
        block_subtotals[static_cast<uint64_t>(column) * blocks_per_row +
                        block] = subtotal;
      }
      accumulators[column_offset] += subtotal;
    }
  }

#pragma unroll
  for (uint32_t column_offset = 0U; column_offset < 4U; ++column_offset) {
#pragma unroll
    for (uint32_t offset = kWaveWidth / 2U; offset != 0U; offset >>= 1U) {
      accumulators[column_offset] +=
          __shfl_down(accumulators[column_offset], offset, kWaveWidth);
    }
    const uint32_t column = column_base + column_offset;
    if (lane == 0U && column < n) {
      output[column] = accumulators[column_offset];
    }
  }
}

struct Candidate final {
  const char *name;
  DotKind kind;
  bool k_vector2;
  bool activation_lds;
  const void *kernel;
  const void *oracle_kernel;
};

[[maybe_unused]] Candidate make_candidate(const uint32_t index) {
  switch (index) {
  case 0U:
    return {"sdot4-current",
            DotKind::Sdot4,
            false,
            false,
            reinterpret_cast<const void *>(
                nvfp4_decode_dot_kernel<DotKind::Sdot4, false, false>),
            reinterpret_cast<const void *>(
                nvfp4_decode_dot_kernel<DotKind::Sdot4, true, false>)};
  case 1U:
    return {"sudot4-canary",
            DotKind::Sudot4,
            false,
            false,
            reinterpret_cast<const void *>(
                nvfp4_decode_dot_kernel<DotKind::Sudot4, false, false>),
            reinterpret_cast<const void *>(
                nvfp4_decode_dot_kernel<DotKind::Sudot4, true, false>)};
  case 2U:
    return {"fp8-dot4-register-pack",
            DotKind::Fp8Dot4,
            false,
            false,
            reinterpret_cast<const void *>(
                nvfp4_decode_dot_kernel<DotKind::Fp8Dot4, false, false>),
            reinterpret_cast<const void *>(
                nvfp4_decode_dot_kernel<DotKind::Fp8Dot4, true, false>)};
  default:
    return {"sdot4-kvector2",
            DotKind::Sdot4,
            true,
            false,
            reinterpret_cast<const void *>(
                nvfp4_decode_dot_kernel<DotKind::Sdot4, false, true>),
            reinterpret_cast<const void *>(
                nvfp4_decode_dot_kernel<DotKind::Sdot4, true, true>)};
  }
}

Candidate make_production_candidate(const uint32_t index) {
  const bool activation_lds = index >= 3U;
  const uint32_t kind_index = index % 3U;
  switch (kind_index) {
  case 0U:
    if (activation_lds) {
      return {"sdot4-wave4col32-activation-lds",
              DotKind::Sdot4,
              false,
              true,
              reinterpret_cast<const void *>(
                  nvfp4_decode_wave4col32_kernel<DotKind::Sdot4, false, true>),
              reinterpret_cast<const void *>(
                  nvfp4_decode_wave4col32_kernel<DotKind::Sdot4, true, true>)};
    }
    return {activation_lds ? "sdot4-wave4col32-activation-lds"
                           : "sdot4-wave4col32",
            DotKind::Sdot4,
            false,
            false,
            reinterpret_cast<const void *>(
                nvfp4_decode_wave4col32_kernel<DotKind::Sdot4, false, false>),
            reinterpret_cast<const void *>(
                nvfp4_decode_wave4col32_kernel<DotKind::Sdot4, true, false>)};
  case 1U:
    if (activation_lds) {
      return {"sudot4-wave4col32-activation-lds",
              DotKind::Sudot4,
              false,
              true,
              reinterpret_cast<const void *>(
                  nvfp4_decode_wave4col32_kernel<DotKind::Sudot4, false, true>),
              reinterpret_cast<const void *>(
                  nvfp4_decode_wave4col32_kernel<DotKind::Sudot4, true, true>)};
    }
    return {activation_lds ? "sudot4-wave4col32-activation-lds"
                           : "sudot4-wave4col32",
            DotKind::Sudot4,
            false,
            false,
            reinterpret_cast<const void *>(
                nvfp4_decode_wave4col32_kernel<DotKind::Sudot4, false, false>),
            reinterpret_cast<const void *>(
                nvfp4_decode_wave4col32_kernel<DotKind::Sudot4, true, false>)};
  default:
    if (activation_lds) {
      return {
          "fp8-dot4-wave4col32-activation-lds",
          DotKind::Fp8Dot4,
          false,
          true,
          reinterpret_cast<const void *>(
              nvfp4_decode_wave4col32_kernel<DotKind::Fp8Dot4, false, true>),
          reinterpret_cast<const void *>(
              nvfp4_decode_wave4col32_kernel<DotKind::Fp8Dot4, true, true>)};
    }
    return {activation_lds ? "fp8-dot4-wave4col32-activation-lds"
                           : "fp8-dot4-wave4col32",
            DotKind::Fp8Dot4,
            false,
            false,
            reinterpret_cast<const void *>(
                nvfp4_decode_wave4col32_kernel<DotKind::Fp8Dot4, false, false>),
            reinterpret_cast<const void *>(
                nvfp4_decode_wave4col32_kernel<DotKind::Fp8Dot4, true, false>)};
  }
}

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::fprintf(stderr, "hip error operation=%s status=%s\n", operation,
               hipGetErrorString(status));
  return false;
}

bool exact_gfx1201(const char *const arch) {
  if (arch == nullptr) {
    return false;
  }
  const std::string_view value(arch);
  constexpr std::string_view prefix = "gfx1201";
  return value == prefix || (value.size() > prefix.size() &&
                             value.compare(0U, prefix.size(), prefix) == 0 &&
                             value[prefix.size()] == ':');
}

float fp16_to_float(const uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16U;
  const uint32_t exponent = (bits >> 10U) & 0x1fU;
  const uint32_t mantissa = bits & 0x3ffU;
  uint32_t result = sign;
  if (exponent == 0U) {
    if (mantissa != 0U) {
      float value = static_cast<float>(mantissa) * 0x1p-24F;
      std::memcpy(&result, &value, sizeof(result));
      result = (result & UINT32_C(0x7fffffff)) | sign;
    }
  } else if (exponent == 31U) {
    result |= UINT32_C(0x7f800000) | (mantissa << 13U);
  } else {
    result |= ((exponent + 112U) << 23U) | (mantissa << 13U);
  }
  float value = 0.0F;
  std::memcpy(&value, &result, sizeof(value));
  return value;
}

float e4m3_to_float_host(const uint8_t code) {
  return fp16_to_float(sllm_lowp::e4m3fn_to_fp16_bits(code));
}

struct Buffers final {
  uint8_t *activation = nullptr;
  uint8_t *activation_scales = nullptr;
  uint8_t *weight = nullptr;
  uint8_t *weight_scales = nullptr;
  float *output = nullptr;
  float *block_subtotals = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
};

void free_buffers(Buffers *const buffers) {
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
  if (buffers->block_subtotals != nullptr) {
    (void)hipFree(buffers->block_subtotals);
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

bool make_buffers(const uint32_t blocks_per_row, const uint32_t n,
                  Buffers *const buffers) {
  const uint64_t weight_bytes = static_cast<uint64_t>(n) * blocks_per_row * 8U;
  const uint64_t subtotal_count = static_cast<uint64_t>(n) * blocks_per_row;
  if (weight_bytes > SIZE_MAX || subtotal_count > SIZE_MAX / sizeof(float)) {
    return false;
  }
  return hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->activation),
                          blocks_per_row * 8U),
                "hipMalloc activation") &&
         hip_ok(
             hipMalloc(reinterpret_cast<void **>(&buffers->activation_scales),
                       blocks_per_row),
             "hipMalloc activation scales") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->weight),
                          static_cast<size_t>(weight_bytes)),
                "hipMalloc weight") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->weight_scales),
                          subtotal_count),
                "hipMalloc weight scales") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->output),
                          static_cast<size_t>(n) * sizeof(float)),
                "hipMalloc output") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->block_subtotals),
                          static_cast<size_t>(subtotal_count) * sizeof(float)),
                "hipMalloc block subtotals") &&
         hip_ok(hipStreamCreate(&buffers->stream), "hipStreamCreate") &&
         hip_ok(hipEventCreate(&buffers->start), "hipEventCreate start") &&
         hip_ok(hipEventCreate(&buffers->stop), "hipEventCreate stop");
}

void fill_inputs(const uint32_t blocks_per_row, const uint32_t n,
                 std::vector<uint8_t> *const activation,
                 std::vector<uint8_t> *const activation_scales,
                 std::vector<uint8_t> *const weight,
                 std::vector<uint8_t> *const weight_scales) {
  activation->resize(static_cast<size_t>(blocks_per_row) * 8U);
  activation_scales->resize(blocks_per_row);
  weight->resize(static_cast<size_t>(n) * blocks_per_row * 8U);
  weight_scales->resize(static_cast<size_t>(n) * blocks_per_row);
  constexpr std::array<uint8_t, 8> scale_codes = {0x30U, 0x38U, 0x3cU, 0x40U,
                                                  0x44U, 0x48U, 0x4cU, 0x34U};
  for (uint32_t block = 0U; block < blocks_per_row; ++block) {
    (*activation_scales)[block] = scale_codes[block % scale_codes.size()];
    for (uint32_t byte = 0U; byte < 8U; ++byte) {
      const uint8_t low =
          static_cast<uint8_t>((block * 5U + byte * 3U) & 0x0fU);
      const uint8_t high =
          static_cast<uint8_t>((block * 11U + byte * 7U + 1U) & 0x0fU);
      (*activation)[static_cast<size_t>(block) * 8U + byte] =
          static_cast<uint8_t>(low | static_cast<uint8_t>(high << 4U));
    }
  }
  for (uint32_t column = 0U; column < n; ++column) {
    for (uint32_t block = 0U; block < blocks_per_row; ++block) {
      const size_t offset =
          (static_cast<size_t>(column) * blocks_per_row + block) * 8U;
      (*weight_scales)[static_cast<size_t>(column) * blocks_per_row + block] =
          scale_codes[(column * 3U + block * 5U + 2U) % scale_codes.size()];
      for (uint32_t byte = 0U; byte < 8U; ++byte) {
        const uint8_t low = static_cast<uint8_t>(
            (column * 13U + block * 5U + byte * 9U + 2U) & 0x0fU);
        const uint8_t high = static_cast<uint8_t>(
            (column * 7U + block * 11U + byte * 3U + 4U) & 0x0fU);
        (*weight)[offset + byte] =
            static_cast<uint8_t>(low | static_cast<uint8_t>(high << 4U));
      }
    }
  }
}

bool upload_inputs(const uint32_t blocks_per_row, const uint32_t n,
                   const std::vector<uint8_t> &activation,
                   const std::vector<uint8_t> &activation_scales,
                   const std::vector<uint8_t> &weight,
                   const std::vector<uint8_t> &weight_scales,
                   Buffers *const buffers) {
  return hip_ok(hipMemcpy(buffers->activation, activation.data(),
                          activation.size(), hipMemcpyHostToDevice),
                "hipMemcpy activation") &&
         hip_ok(hipMemcpy(buffers->activation_scales, activation_scales.data(),
                          activation_scales.size(), hipMemcpyHostToDevice),
                "hipMemcpy activation scales") &&
         hip_ok(hipMemcpy(buffers->weight, weight.data(), weight.size(),
                          hipMemcpyHostToDevice),
                "hipMemcpy weight") &&
         hip_ok(hipMemcpy(buffers->weight_scales, weight_scales.data(),
                          weight_scales.size(), hipMemcpyHostToDevice),
                "hipMemcpy weight scales") &&
         hip_ok(hipMemset(buffers->output, 0, n * sizeof(float)),
                "hipMemset output") &&
         hip_ok(
             hipMemset(buffers->block_subtotals, 0,
                       static_cast<size_t>(n) * blocks_per_row * sizeof(float)),
             "hipMemset block subtotals");
}

bool launch(const Candidate &candidate, const uint32_t blocks_per_row,
            const uint32_t n, Buffers *const buffers, const bool oracle) {
  const dim3 grid((n + kWaves - 1U) / kWaves);
  const void *const function =
      oracle ? candidate.oracle_kernel : candidate.kernel;
  float *subtotals = oracle ? buffers->block_subtotals : nullptr;
  uint32_t blocks_argument = blocks_per_row;
  uint32_t columns_argument = n;
  void *arguments[] = {&buffers->activation, &buffers->activation_scales,
                       &buffers->weight,     &buffers->weight_scales,
                       &buffers->output,     &subtotals,
                       &blocks_argument,     &columns_argument};
  // The kernel argument array contains pointers to the argument values, not
  // the device pointers themselves.  A null block-subtotal pointer is the
  // performance form of the same kernel contract.
  const size_t dynamic_shared_bytes =
      candidate.activation_lds ? static_cast<size_t>(blocks_per_row) * 20U : 0U;
  return hip_ok(hipLaunchKernel(function, grid, dim3(kThreads), arguments,
                                dynamic_shared_bytes, buffers->stream),
                "kernel launch");
}

bool measure(const Candidate &candidate, const uint32_t blocks_per_row,
             const uint32_t n, Buffers *const buffers, float *const median_us) {
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    if (!launch(candidate, blocks_per_row, n, buffers, false) ||
        !hip_ok(hipStreamSynchronize(buffers->stream), "warmup synchronize")) {
      return false;
    }
  }
  std::array<float, kMeasured> samples{};
  for (int iteration = 0; iteration < kMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(buffers->start, buffers->stream),
                "event start") ||
        !launch(candidate, blocks_per_row, n, buffers, false) ||
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

bool check_pack_oracle() {
  size_t pack_mismatches = 0U;
  size_t fp8_mismatches = 0U;
  for (uint32_t code = 0U; code < 16U; ++code) {
    const uint16_t packed = static_cast<uint16_t>(code | (code << 4U) |
                                                  (code << 8U) | (code << 12U));
    const uint32_t actual = e2m1x4_to_e4m3fn(packed);
    uint32_t expected = 0U;
    for (uint32_t lane = 0U; lane < 4U; ++lane) {
      expected |= static_cast<uint32_t>(e2m1_to_e4m3fn(code)) << (lane * 8U);
    }
    if (actual != expected) {
      ++fp8_mismatches;
    }
  }
  for (uint32_t packed = 0U; packed < 65536U; ++packed) {
    const uint32_t word = packed;
    const std::array<uint8_t, 4> codes = {
        static_cast<uint8_t>(word & 0x0fU),
        static_cast<uint8_t>((word >> 4U) & 0x0fU),
        static_cast<uint8_t>((word >> 8U) & 0x0fU),
        static_cast<uint8_t>((word >> 12U) & 0x0fU)};
    uint32_t expected_fp8 = 0U;
    for (uint32_t lane = 0U; lane < 4U; ++lane) {
      expected_fp8 |= static_cast<uint32_t>(e2m1_to_e4m3fn(codes[lane]))
                      << (lane * 8U);
    }
    if (e2m1x4_to_e4m3fn(static_cast<uint16_t>(word)) != expected_fp8) {
      ++fp8_mismatches;
    }
    for (uint32_t lane = 0U; lane < 4U; ++lane) {
      const uint8_t extracted =
          static_cast<uint8_t>((word >> (lane * 4U)) & 0x0fU);
      if (extracted != codes[lane] ||
          static_cast<int8_t>(static_cast<int>(e2m1_value(extracted) * 2.0F)) <
              -128 ||
          static_cast<int8_t>(static_cast<int>(e2m1_value(extracted) * 2.0F)) >
              127) {
        ++pack_mismatches;
      }
    }
  }
  std::printf(
      "oracle e2m1_codes=16 packed_nibbles=65536 packed_lane_mismatches=%zu "
      "fp8_pack_mismatches=%zu status=%s\n",
      pack_mismatches, fp8_mismatches,
      pack_mismatches == 0U && fp8_mismatches == 0U ? "PASS" : "FAIL");
  return pack_mismatches == 0U && fp8_mismatches == 0U;
}

bool check_variant_oracle(const Candidate &candidate, const uint32_t blocks,
                          const uint32_t n,
                          const std::vector<uint8_t> &activation,
                          const std::vector<uint8_t> &activation_scales,
                          const std::vector<uint8_t> &weight,
                          const std::vector<uint8_t> &weight_scales,
                          Buffers *const buffers,
                          const std::vector<float> *const control_output,
                          std::vector<float> *const captured_output) {
  if (!launch(candidate, blocks, n, buffers, true) ||
      !hip_ok(hipStreamSynchronize(buffers->stream), "oracle synchronize")) {
    return false;
  }
  std::vector<float> actual_output(n);
  std::vector<float> actual_subtotals(static_cast<size_t>(n) * blocks);
  if (!hip_ok(hipMemcpy(actual_output.data(), buffers->output,
                        actual_output.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "hipMemcpy oracle output") ||
      !hip_ok(hipMemcpy(actual_subtotals.data(), buffers->block_subtotals,
                        actual_subtotals.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "hipMemcpy oracle subtotals")) {
    return false;
  }
  if (captured_output != nullptr) {
    *captured_output = actual_output;
  }
  size_t control_mismatches = 0U;
  if (control_output != nullptr &&
      control_output->size() == actual_output.size()) {
    for (uint32_t column = 0U; column < n; ++column) {
      uint32_t actual_bits = 0U;
      uint32_t control_bits = 0U;
      std::memcpy(&actual_bits, &actual_output[column], sizeof(actual_bits));
      std::memcpy(&control_bits, &(*control_output)[column],
                  sizeof(control_bits));
      if (actual_bits != control_bits) {
        ++control_mismatches;
      }
    }
  }

  // Check all block subtotals for a small oracle shape.  This exercises every
  // packed nibble code and every scale pairing before the production shapes;
  // for the FP8 candidate it also validates the device-side v_perm register
  // pack (the host all-65536 loop above validates the corresponding mapping).
  const uint32_t columns_to_check = std::min<uint32_t>(n, 32U);
  double max_abs = 0.0;
  size_t mismatches = 0U;
  for (uint32_t column = 0U; column < columns_to_check; ++column) {
    float expected_total = 0.0F;
    for (uint32_t block = 0U; block < blocks; ++block) {
      int32_t integer_sum = 0;
      float fp8_sum = 0.0F;
      const size_t activation_offset = static_cast<size_t>(block) * 8U;
      const size_t weight_offset =
          (static_cast<size_t>(column) * blocks + block) * 8U;
      for (uint32_t index = 0U; index < 16U; ++index) {
        const uint8_t a_code = (activation[activation_offset + index / 2U] >>
                                ((index & 1U) * 4U)) &
                               0x0fU;
        const uint8_t b_code =
            (weight[weight_offset + index / 2U] >> ((index & 1U) * 4U)) & 0x0fU;
        integer_sum += static_cast<int32_t>(e2m1_value(a_code) * 2.0F) *
                       static_cast<int32_t>(e2m1_value(b_code) * 2.0F);
        fp8_sum = std::fmaf(e2m1_value(a_code), e2m1_value(b_code), fp8_sum);
      }
      const float scale_a = e4m3_to_float_host(activation_scales[block]);
      const float scale_b = e4m3_to_float_host(
          weight_scales[static_cast<size_t>(column) * blocks + block]);
      const float expected =
          candidate.kind == DotKind::Fp8Dot4
              ? fp8_sum * scale_a * scale_b
              : static_cast<float>(integer_sum) * 0.25F * scale_a * scale_b;
      const float actual =
          actual_subtotals[static_cast<size_t>(column) * blocks + block];
      const double error = std::abs(static_cast<double>(actual) - expected);
      max_abs = std::max(max_abs, error);
      const double tolerance =
          2.0e-4 * std::max(1.0, std::abs(static_cast<double>(expected)));
      if (!std::isfinite(actual) || error > tolerance) {
        ++mismatches;
      }
      expected_total += expected;
    }
    const double output_error =
        std::abs(static_cast<double>(actual_output[column]) - expected_total);
    max_abs = std::max(max_abs, output_error);
    if (!std::isfinite(actual_output[column]) ||
        output_error > 2.0e-3 * std::max(1.0, std::abs(static_cast<double>(
                                                  expected_total)))) {
      ++mismatches;
    }
  }
  std::printf(
      "oracle candidate=%s blocks=%u columns=%u max_abs=%.8g mismatches=%zu "
      "control_bitwise_mismatches=%zu status=%s\n",
      candidate.name, blocks, columns_to_check, max_abs, mismatches,
      control_mismatches,
      mismatches == 0U && control_mismatches == 0U ? "PASS" : "FAIL");
  return mismatches == 0U && control_mismatches == 0U;
}

void print_resources(const Candidate &candidate,
                     const uint32_t blocks_per_row) {
  hipFuncAttributes attributes{};
  const hipError_t attr_status =
      hipFuncGetAttributes(&attributes, candidate.kernel);
  int active_blocks = 0;
  const hipError_t occupancy_status =
      hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks, candidate.kernel, kThreads,
          candidate.activation_lds ? static_cast<size_t>(blocks_per_row) * 20U
                                   : 0U);
  std::printf(
      "resources candidate=%s vgpr=%d lds_static=%zu lds_dynamic=%zu "
      "scratch=%zu max_threads=%d active_blocks=%d attr=%s occupancy=%s\n",
      candidate.name, attributes.numRegs, attributes.sharedSizeBytes,
      candidate.activation_lds ? static_cast<size_t>(blocks_per_row) * 20U : 0U,
      attributes.localSizeBytes, attributes.maxThreadsPerBlock, active_blocks,
      hipGetErrorString(attr_status), hipGetErrorString(occupancy_status));
}

bool run_shape(const uint32_t k, const uint32_t n,
               std::array<float, 6> *const measured_us = nullptr) {
  const uint32_t blocks = k / 16U;
  std::vector<uint8_t> activation;
  std::vector<uint8_t> activation_scales;
  std::vector<uint8_t> weight;
  std::vector<uint8_t> weight_scales;
  fill_inputs(blocks, n, &activation, &activation_scales, &weight,
              &weight_scales);
  Buffers buffers;
  if (!make_buffers(blocks, n, &buffers) ||
      !upload_inputs(blocks, n, activation, activation_scales, weight,
                     weight_scales, &buffers)) {
    free_buffers(&buffers);
    return false;
  }
  bool all_ok = true;
  std::vector<float> control_output;
  for (uint32_t index = 0U; index < 6U; ++index) {
    const Candidate current = make_production_candidate(index);
    print_resources(current, blocks);
    all_ok =
        check_variant_oracle(current, blocks, n, activation, activation_scales,
                             weight, weight_scales, &buffers,
                             index == 0U ? nullptr : &control_output,
                             index == 0U ? &control_output : nullptr) &&
        all_ok;
    float median_us = 0.0F;
    if (!measure(current, blocks, n, &buffers, &median_us)) {
      free_buffers(&buffers);
      return false;
    }
    if (measured_us != nullptr) {
      (*measured_us)[index] = median_us;
    }
    const double weight_bytes = static_cast<double>(n) * k / 2.0;
    const double gigabytes_per_second = weight_bytes / median_us / 1000.0;
    std::printf(
        "result candidate=%s k=%u n=%u median_us=%.3f parameter_gbps=%.6f\n",
        current.name, k, n, median_us, gigabytes_per_second);
  }
  free_buffers(&buffers);
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
      !exact_gfx1201(properties.gcnArchName)) {
    std::fprintf(stderr, "exact gfx1201 required; got %s\n",
                 properties.gcnArchName);
    return EXIT_FAILURE;
  }
  std::printf("target=%s device=0 pci=%04x:%02x:%02x name=%s\n",
              properties.gcnArchName, properties.pciDomainID,
              properties.pciBusID, properties.pciDeviceID, properties.name);
  if (!check_pack_oracle()) {
    return EXIT_FAILURE;
  }

  // A small complete block oracle precedes the production exact shapes.  The
  // production K values are both divisible by 16, matching the model lock.
  std::array<float, 6> first_production_shape_us{};
  std::array<float, 6> second_production_shape_us{};
  if (!run_shape(512U, 32U) ||
      !run_shape(5120U, 17408U, &first_production_shape_us) ||
      !run_shape(17408U, 5120U, &second_production_shape_us)) {
    return EXIT_FAILURE;
  }
  for (uint32_t index = 0U; index < 6U; ++index) {
    const Candidate current = make_production_candidate(index);
    // Both exact production shapes have identical K*N traffic.  The
    // arithmetic mean is therefore the FLOP/byte-weighted ms/token summary.
    const double weighted_ms_per_token =
        (static_cast<double>(first_production_shape_us[index]) +
         static_cast<double>(second_production_shape_us[index])) /
        2000.0;
    std::printf("weighted candidate=%s weighted_ms_per_token=%.6f\n",
                current.name, weighted_ms_per_token);
  }
  std::printf(
      "summary status=PASS variants=6 shapes=3 warmups=%d measured=%d\n",
      kWarmups, kMeasured);
  return EXIT_SUCCESS;
}
