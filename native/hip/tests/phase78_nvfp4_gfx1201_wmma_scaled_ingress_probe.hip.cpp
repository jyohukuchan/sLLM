// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase78-nvfp4-byte-permute-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// 3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70, ggml/src/ggml-cuda/vecdotq.cuh
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

// Phase 78 standalone gfx1201 NVFP4 scaled-ingress WMMA probe.
//
// ID64 is the production FP8-WMMA control: it converts E2M1 to E4M3 exactly,
// runs one K=16 MMA per scale domain, and applies the two block scales to the
// FP32 contribution.  ID69 is the current FP16-WMMA baseline: scalar lanes
// repeatedly load a packed source byte, decode one nibble, and absorb block
// scales into FP16 LDS operands before MMA.
//
// The two candidates retain ID69's arithmetic order but replace its ingress.
// One thread owns four adjacent E2M1 values, issues one aligned 16-bit load,
// expands all four nibbles to two packed half2 values, broadcasts the E4M3
// block scale to half2, and writes two adjacent dwords to LDS.  StageK=32 is
// mandatory.  StageK=64 is benchmarked only when the compiled kernel reports
// at most 64 KiB LDS and no scratch spill.
//
// This developer probe is intentionally outside the production build and
// does not alter provider selection.

#include "low_precision_block_codec.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kWarmups = 3U;
constexpr uint32_t kMeasured = 10U;
constexpr uint32_t kOracleSamples = 64U;
constexpr uint32_t kSeed = UINT32_C(0x243f6a88);
constexpr float kWeightTensorScale = 0.75F;
constexpr float kInputTensorScale = 1.125F;
constexpr uint64_t kMaximumCandidateLds = UINT64_C(64) * 1024U;

static_assert(HIP_VERSION_MAJOR == 7 && HIP_VERSION_MINOR == 14,
              "the probe is pinned to ROCm/HIP 7.14");

struct Shape final {
  uint64_t m;
  uint64_t k;
  uint64_t n;
  uint32_t occurrences;
  const char *name;
};

constexpr std::array<Shape, 6> kShapes = {{
    {128U, 5120U, 17408U, 112U, "wide-m128"},
    {512U, 5120U, 17408U, 112U, "wide-m512"},
    {1024U, 5120U, 17408U, 112U, "wide-m1024"},
    {128U, 17408U, 5120U, 56U, "down-m128"},
    {512U, 17408U, 5120U, 56U, "down-m512"},
    {1024U, 17408U, 5120U, 56U, "down-m1024"},
}};

enum class Variant : uint32_t { Id64, Id69, Vector32, Vector64 };

constexpr std::array<Variant, 4> kVariants = {
    Variant::Id64, Variant::Id69, Variant::Vector32, Variant::Vector64};

const char *variant_name(const Variant variant) {
  switch (variant) {
  case Variant::Id64:
    return "id64-fp8-contribution-scale";
  case Variant::Id69:
    return "id69-scalar-fp16-ingress";
  case Variant::Vector32:
    return "vector-fp16-ingress-stagek32";
  case Variant::Vector64:
    return "vector-fp16-ingress-stagek64";
  }
  return "unknown";
}

[[maybe_unused]] __device__ __forceinline__ float
e2m1_to_float(const uint8_t code) {
  constexpr float positive[8] = {0.0F, 0.5F, 1.0F, 1.5F,
                                 2.0F, 3.0F, 4.0F, 6.0F};
  const float value = positive[code & UINT8_C(7)];
  return (code & UINT8_C(8)) == 0U ? value : -value;
}

__host__ __device__ __forceinline__ float e4m3fn_to_float(const uint8_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & UINT8_C(0x80)) << 24U;
  const uint32_t magnitude = bits & UINT8_C(0x7f);
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  if (exponent == 0U) {
    if (mantissa == 0U) {
#if defined(__HIP_DEVICE_COMPILE__)
      return __uint_as_float(sign);
#else
      uint32_t raw = sign;
      float value = 0.0F;
      std::memcpy(&value, &raw, sizeof(value));
      return value;
#endif
    }
    const float value = static_cast<float>(mantissa) * 0x1p-9F;
#if defined(__HIP_DEVICE_COMPILE__)
    return __uint_as_float(__float_as_uint(value) | sign);
#else
    uint32_t raw = 0U;
    std::memcpy(&raw, &value, sizeof(raw));
    raw |= sign;
    float signed_value = 0.0F;
    std::memcpy(&signed_value, &raw, sizeof(signed_value));
    return signed_value;
#endif
  }
  if (magnitude == UINT32_C(0x7f)) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __uint_as_float(sign | UINT32_C(0x7fc00000));
#else
    uint32_t raw = sign | UINT32_C(0x7fc00000);
    float value = 0.0F;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
#endif
  }
#if defined(__HIP_DEVICE_COMPILE__)
  return __uint_as_float(sign | ((exponent + 120U) << 23U) | (mantissa << 20U));
#else
  const uint32_t raw = sign | ((exponent + 120U) << 23U) | (mantissa << 20U);
  float value = 0.0F;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
#endif
}

[[maybe_unused]] __device__ __forceinline__ uint16_t
bf16_rne(const float value) {
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

// Four E2M1 nibbles become four byte-addressable E4M3 values for ID64.
[[maybe_unused]] __device__ __forceinline__ uint32_t
e2m1x4_to_e4m3fn_exact(const uint16_t packed) {
  const uint32_t lanes =
      (static_cast<uint32_t>(packed) & UINT32_C(0x000f)) |
      ((static_cast<uint32_t>(packed) & UINT32_C(0x00f0)) << 4U) |
      ((static_cast<uint32_t>(packed) & UINT32_C(0x0f00)) << 8U) |
      ((static_cast<uint32_t>(packed) & UINT32_C(0xf000)) << 12U);
  constexpr uint32_t positive_0_3 = UINT32_C(0x3c383000);
  constexpr uint32_t positive_4_7 = UINT32_C(0x4c484440);
  const uint32_t positive = __builtin_amdgcn_perm(positive_4_7, positive_0_3,
                                                  lanes & UINT32_C(0x07070707));
  return positive | ((lanes & UINT32_C(0x08080808)) << 4U);
}

struct Fp16x4Bits final {
  uint32_t low;
  uint32_t high;
};

// Vector E2M1 ingress.  A packed-byte permutation directly selects the high
// byte of each exact FP16 encoding; two integer spreads then place those bytes
// into four independent 16-bit lanes.  No scalar nibble decode or source-byte
// reload remains.
[[maybe_unused]] __device__ __forceinline__ Fp16x4Bits
e2m1x4_to_fp16x2_bits(const uint16_t packed) {
  const uint32_t lanes =
      (static_cast<uint32_t>(packed) & UINT32_C(0x000f)) |
      ((static_cast<uint32_t>(packed) & UINT32_C(0x00f0)) << 4U) |
      ((static_cast<uint32_t>(packed) & UINT32_C(0x0f00)) << 8U) |
      ((static_cast<uint32_t>(packed) & UINT32_C(0xf000)) << 12U);
  // High bytes of FP16 {0, .5, 1, 1.5, 2, 3, 4, 6}.
  constexpr uint32_t positive_0_3 = UINT32_C(0x3e3c3800);
  constexpr uint32_t positive_4_7 = UINT32_C(0x46444240);
  const uint32_t positive = __builtin_amdgcn_perm(positive_4_7, positive_0_3,
                                                  lanes & UINT32_C(0x07070707));
  const uint32_t fp16_high_bytes =
      positive | ((lanes & UINT32_C(0x08080808)) << 4U);
  return Fp16x4Bits{
      ((fp16_high_bytes & UINT32_C(0x000000ff)) << 8U) |
          ((fp16_high_bytes & UINT32_C(0x0000ff00)) << 16U),
      ((fp16_high_bytes & UINT32_C(0x00ff0000)) >> 8U) |
          (fp16_high_bytes & UINT32_C(0xff000000)),
  };
}

[[maybe_unused]] __device__ __forceinline__ uint32_t
half2_bits(const __half2 value) {
  static_assert(sizeof(__half2_raw) == sizeof(uint32_t));
  union Bits {
    __half2_raw half;
    uint32_t raw;
  } bits{static_cast<__half2_raw>(value)};
  return bits.raw;
}

// Production ID64 control copied into the standalone evidence binary.
__global__ __launch_bounds__(kThreads, 1) void id64_control_kernel(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
#if defined(__gfx1201__)
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t waves_per_workgroup = 8U;
  constexpr uint32_t tile_m = 16U;
  constexpr uint32_t tile_n = 16U;
  constexpr uint32_t column_tiles = 4U;
  constexpr uint32_t fragment_k = 16U;
  constexpr uint32_t stage_k = 32U;
  constexpr uint32_t scale_blocks_per_stage = stage_k / fragment_k;
  constexpr uint32_t rows_per_workgroup = waves_per_workgroup * tile_m;
  constexpr uint32_t columns_per_workgroup = column_tiles * tile_n;
  constexpr uint32_t tile_values = tile_m * stage_k;
  constexpr uint32_t values_per_group = 4U;
  constexpr uint32_t groups_per_tile = tile_values / values_per_group;
  constexpr uint32_t output_values = tile_m * tile_n;
  __shared__ __align__(4)
      rocwmma::float8_t activation_tile[waves_per_workgroup][tile_values];
  __shared__ __align__(4)
      rocwmma::float8_t weight_tile[column_tiles][tile_values];
  __shared__ float activation_scale_tile[waves_per_workgroup][tile_m]
                                        [scale_blocks_per_stage];
  __shared__ float weight_scale_tile[column_tiles * tile_n]
                                    [scale_blocks_per_stage];
  __shared__ float tensor_scale;
  using AFragment =
      rocwmma::fragment<rocwmma::matrix_a, tile_m, tile_n, fragment_k,
                        rocwmma::float8_t, rocwmma::row_major>;
  using BFragment =
      rocwmma::fragment<rocwmma::matrix_b, tile_m, tile_n, fragment_k,
                        rocwmma::float8_t, rocwmma::col_major>;
  using AccumulatorFragment = rocwmma::fragment<rocwmma::accumulator, tile_m,
                                                tile_n, fragment_k, float>;
  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (wave_width - 1U);
  const uint32_t wave = thread / wave_width;
  const uint64_t row_group_base =
      static_cast<uint64_t>(blockIdx.y) * rows_per_workgroup;
  const uint64_t row_tile_base =
      row_group_base + static_cast<uint64_t>(wave) * tile_m;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * columns_per_workgroup;
  const uint64_t blocks_per_row = k / fragment_k;
  const uint64_t stages =
      (blocks_per_row + scale_blocks_per_stage - 1U) / scale_blocks_per_stage;
  const uint64_t packed_row_bytes = k / 2U;
  float accumulators[column_tiles][output_values / wave_width] = {};
  if (thread == 0U) {
    tensor_scale = weight_tensor_scale[0] * input_tensor_scale[0];
  }
  for (uint64_t stage = 0U; stage < stages; ++stage) {
    const uint64_t inner_base = stage * stage_k;
    auto *const activation_groups =
        reinterpret_cast<uint32_t *>(activation_tile);
    auto *const weight_groups = reinterpret_cast<uint32_t *>(weight_tile);
    for (uint32_t group = thread; group < waves_per_workgroup * groups_per_tile;
         group += blockDim.x) {
      const uint32_t source_wave = group / groups_per_tile;
      const uint32_t wave_group = group - source_wave * groups_per_tile;
      const uint32_t local_row = wave_group / (stage_k / values_per_group);
      const uint32_t local_group =
          wave_group - local_row * (stage_k / values_per_group);
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      uint16_t packed = 0U;
      if (row < m && inner_base + local_group * values_per_group < k) {
        packed = __builtin_nontemporal_load(reinterpret_cast<const uint16_t *>(
            packed_activation + row * packed_row_bytes + inner_base / 2U +
            local_group * 2U));
      }
      activation_groups[group] = e2m1x4_to_e4m3fn_exact(packed);
    }
    for (uint32_t group = thread; group < column_tiles * groups_per_tile;
         group += blockDim.x) {
      const uint32_t column_tile = group / groups_per_tile;
      const uint32_t tile_group = group - column_tile * groups_per_tile;
      const uint32_t local_column = tile_group / (stage_k / values_per_group);
      const uint32_t local_group =
          tile_group - local_column * (stage_k / values_per_group);
      const uint64_t column = column_base +
                              static_cast<uint64_t>(column_tile) * tile_n +
                              local_column;
      uint16_t packed = 0U;
      if (column < n && inner_base + local_group * values_per_group < k) {
        packed = __builtin_nontemporal_load(reinterpret_cast<const uint16_t *>(
            packed_weight + column * packed_row_bytes + inner_base / 2U +
            local_group * 2U));
      }
      weight_groups[group] = e2m1x4_to_e4m3fn_exact(packed);
    }
    if (thread < waves_per_workgroup * tile_m * scale_blocks_per_stage) {
      const uint32_t scale_block = thread % scale_blocks_per_stage;
      const uint32_t row_index = thread / scale_blocks_per_stage;
      const uint32_t source_wave = row_index / tile_m;
      const uint32_t local_row = row_index - source_wave * tile_m;
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      activation_scale_tile[source_wave][local_row][scale_block] =
          row < m && block < blocks_per_row
              ? e4m3fn_to_float(
                    activation_block_scales[row * blocks_per_row + block])
              : 0.0F;
    }
    if (thread < column_tiles * tile_n * scale_blocks_per_stage) {
      const uint32_t scale_block = thread % scale_blocks_per_stage;
      const uint32_t local_column = thread / scale_blocks_per_stage;
      const uint64_t column = column_base + local_column;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      weight_scale_tile[local_column][scale_block] =
          column < n && block < blocks_per_row
              ? e4m3fn_to_float(
                    weight_block_scales[column * blocks_per_row + block])
              : 0.0F;
    }
    __syncthreads();
    for (uint32_t scale_block = 0U; scale_block < scale_blocks_per_stage;
         ++scale_block) {
      AFragment activation_fragment;
      rocwmma::load_matrix_sync(
          activation_fragment, activation_tile[wave] + scale_block * fragment_k,
          stage_k);
#pragma unroll
      for (uint32_t column_tile = 0U; column_tile < column_tiles;
           ++column_tile) {
        BFragment weight_fragment;
        AccumulatorFragment contribution;
        rocwmma::fill_fragment(contribution, 0.0F);
        rocwmma::load_matrix_sync(
            weight_fragment,
            weight_tile[column_tile] + scale_block * fragment_k, stage_k);
        rocwmma::mma_sync(contribution, activation_fragment, weight_fragment,
                          contribution);
        const auto row_major =
            rocwmma::apply_data_layout<rocwmma::row_major>(contribution);
#pragma unroll
        for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
          const uint32_t local_row =
              (lane / tile_n) * (output_values / wave_width) + slot;
          const uint32_t local_column = lane % tile_n;
          float term = row_major[slot] *
                       activation_scale_tile[wave][local_row][scale_block];
          term *= weight_scale_tile[column_tile * tile_n + local_column]
                                   [scale_block];
          accumulators[column_tile][slot] += term;
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
#pragma unroll
    for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
      const uint32_t local_row =
          (lane / tile_n) * (output_values / wave_width) + slot;
      const uint32_t local_column = lane % tile_n;
      const uint64_t row = row_tile_base + local_row;
      const uint64_t column = column_base + column_tile * tile_n + local_column;
      if (row < m && column < n) {
        output[row * n + column] =
            bf16_rne(accumulators[column_tile][slot] * tensor_scale);
      }
    }
  }
#else
  (void)packed_activation;
  (void)activation_block_scales;
  (void)packed_weight;
  (void)weight_block_scales;
  (void)weight_tensor_scale;
  (void)input_tensor_scale;
  (void)output;
  (void)m;
  (void)k;
  (void)n;
#endif
}

// Current production ID69 candidate, retained byte-for-byte in its relevant
// ingress and MMA ordering as the scalar baseline.
__global__ __launch_bounds__(kThreads, 1) void id69_baseline_kernel(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
#if defined(__gfx1201__)
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t waves_per_workgroup = 8U;
  constexpr uint32_t tile_m = 16U;
  constexpr uint32_t tile_n = 16U;
  constexpr uint32_t column_tiles = 4U;
  constexpr uint32_t fragment_k = 16U;
  constexpr uint32_t stage_k = 32U;
  constexpr uint32_t scale_blocks_per_stage = stage_k / fragment_k;
  constexpr uint32_t rows_per_workgroup = waves_per_workgroup * tile_m;
  constexpr uint32_t columns_per_workgroup = column_tiles * tile_n;
  constexpr uint32_t tile_values = tile_m * stage_k;
  constexpr uint32_t output_values = tile_m * tile_n;
  constexpr uint32_t activation_tile_values = waves_per_workgroup * tile_values;
  constexpr uint32_t weight_tile_values = column_tiles * tile_values;
  __shared__ __align__(4)
      rocwmma::float16_t activation_tile[activation_tile_values];
  __shared__ __align__(4) rocwmma::float16_t weight_tile[weight_tile_values];
  __shared__ rocwmma::float16_t activation_scale_tile[waves_per_workgroup]
                                                     [tile_m]
                                                     [scale_blocks_per_stage];
  __shared__ rocwmma::float16_t weight_scale_tile[column_tiles * tile_n]
                                                 [scale_blocks_per_stage];
  __shared__ float tensor_scale;
  using AFragment =
      rocwmma::fragment<rocwmma::matrix_a, tile_m, tile_n, fragment_k,
                        rocwmma::float16_t, rocwmma::row_major>;
  using BFragment =
      rocwmma::fragment<rocwmma::matrix_b, tile_m, tile_n, fragment_k,
                        rocwmma::float16_t, rocwmma::col_major>;
  using AccumulatorFragment = rocwmma::fragment<rocwmma::accumulator, tile_m,
                                                tile_n, fragment_k, float>;
  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (wave_width - 1U);
  const uint32_t wave = thread / wave_width;
  const uint64_t row_group_base =
      static_cast<uint64_t>(blockIdx.y) * rows_per_workgroup;
  const uint64_t row_tile_base =
      row_group_base + static_cast<uint64_t>(wave) * tile_m;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * columns_per_workgroup;
  const uint64_t blocks_per_row = k / fragment_k;
  const uint64_t stages =
      (blocks_per_row + scale_blocks_per_stage - 1U) / scale_blocks_per_stage;
  const uint64_t packed_row_bytes = k / 2U;
  if (thread == 0U) {
    tensor_scale = weight_tensor_scale[0] * input_tensor_scale[0];
  }
  AccumulatorFragment contributions[column_tiles];
#pragma unroll
  for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
    rocwmma::fill_fragment(contributions[column_tile], 0.0F);
  }
  for (uint64_t stage = 0U; stage < stages; ++stage) {
    const uint64_t inner_base = stage * stage_k;
    for (uint32_t index = thread;
         index < waves_per_workgroup * tile_m * scale_blocks_per_stage;
         index += blockDim.x) {
      const uint32_t scale_block = index % scale_blocks_per_stage;
      const uint32_t row_index = index / scale_blocks_per_stage;
      const uint32_t source_wave = row_index / tile_m;
      const uint32_t local_row = row_index - source_wave * tile_m;
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      activation_scale_tile[source_wave][local_row][scale_block] =
          row < m && block < blocks_per_row
              ? static_cast<rocwmma::float16_t>(e4m3fn_to_float(
                    activation_block_scales[row * blocks_per_row + block]))
              : static_cast<rocwmma::float16_t>(0.0F);
    }
    for (uint32_t index = thread;
         index < column_tiles * tile_n * scale_blocks_per_stage;
         index += blockDim.x) {
      const uint32_t scale_block = index % scale_blocks_per_stage;
      const uint32_t local_column = index / scale_blocks_per_stage;
      const uint64_t column = column_base + local_column;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      weight_scale_tile[local_column][scale_block] =
          column < n && block < blocks_per_row
              ? static_cast<rocwmma::float16_t>(e4m3fn_to_float(
                    weight_block_scales[column * blocks_per_row + block]))
              : static_cast<rocwmma::float16_t>(0.0F);
    }
    __syncthreads();
    for (uint32_t index = thread; index < activation_tile_values;
         index += blockDim.x) {
      const uint32_t source_wave = index / tile_values;
      const uint32_t tile_index = index - source_wave * tile_values;
      const uint32_t local_row = tile_index / stage_k;
      const uint32_t local_inner = tile_index - local_row * stage_k;
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      const uint64_t inner = inner_base + local_inner;
      if (row < m && inner < k) {
        const uint8_t packed = __builtin_nontemporal_load(
            packed_activation + row * packed_row_bytes + inner / 2U);
        const uint8_t code =
            (inner & 1U) == 0U ? packed & UINT8_C(0x0f) : packed >> 4U;
        const uint32_t scale_block = local_inner / fragment_k;
        const float scale = static_cast<float>(
            activation_scale_tile[source_wave][local_row][scale_block]);
        activation_tile[index] =
            static_cast<rocwmma::float16_t>(e2m1_to_float(code) * scale);
      } else {
        activation_tile[index] = static_cast<rocwmma::float16_t>(0.0F);
      }
    }
    for (uint32_t index = thread; index < weight_tile_values;
         index += blockDim.x) {
      const uint32_t column_tile = index / tile_values;
      const uint32_t tile_index = index - column_tile * tile_values;
      const uint32_t local_column = tile_index / stage_k;
      const uint32_t local_inner = tile_index - local_column * stage_k;
      const uint64_t column = column_base +
                              static_cast<uint64_t>(column_tile) * tile_n +
                              local_column;
      const uint64_t inner = inner_base + local_inner;
      if (column < n && inner < k) {
        const uint8_t packed = __builtin_nontemporal_load(
            packed_weight + column * packed_row_bytes + inner / 2U);
        const uint8_t code =
            (inner & 1U) == 0U ? packed & UINT8_C(0x0f) : packed >> 4U;
        const uint32_t scale_block = local_inner / fragment_k;
        const float scale =
            static_cast<float>(weight_scale_tile[column_tile * tile_n +
                                                 local_column][scale_block]);
        weight_tile[index] =
            static_cast<rocwmma::float16_t>(e2m1_to_float(code) * scale);
      } else {
        weight_tile[index] = static_cast<rocwmma::float16_t>(0.0F);
      }
    }
    __syncthreads();
#pragma unroll
    for (uint32_t scale_block = 0U; scale_block < scale_blocks_per_stage;
         ++scale_block) {
      AFragment activation_fragment;
      rocwmma::load_matrix_sync(activation_fragment,
                                activation_tile + wave * tile_values +
                                    scale_block * fragment_k,
                                stage_k);
#pragma unroll
      for (uint32_t column_tile = 0U; column_tile < column_tiles;
           ++column_tile) {
        BFragment weight_fragment;
        rocwmma::load_matrix_sync(weight_fragment,
                                  weight_tile + column_tile * tile_values +
                                      scale_block * fragment_k,
                                  stage_k);
        rocwmma::mma_sync(contributions[column_tile], activation_fragment,
                          weight_fragment, contributions[column_tile]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
    const auto contribution_row_major =
        rocwmma::apply_data_layout<rocwmma::row_major>(
            contributions[column_tile]);
#pragma unroll
    for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
      const uint32_t local_row =
          (lane / tile_n) * (output_values / wave_width) + slot;
      const uint32_t local_column = lane % tile_n;
      const uint64_t row = row_tile_base + local_row;
      const uint64_t column = column_base + column_tile * tile_n + local_column;
      if (row < m && column < n) {
        output[row * n + column] =
            bf16_rne(contribution_row_major[slot] * tensor_scale);
      }
    }
  }
#else
  (void)packed_activation;
  (void)activation_block_scales;
  (void)packed_weight;
  (void)weight_block_scales;
  (void)weight_tensor_scale;
  (void)input_tensor_scale;
  (void)output;
  (void)m;
  (void)k;
  (void)n;
#endif
}

template <uint32_t StageK>
__global__ __launch_bounds__(kThreads, 1) void vector_ingress_kernel(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
#if defined(__gfx1201__)
  static_assert(StageK == 32U || StageK == 64U);
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t waves_per_workgroup = 8U;
  constexpr uint32_t tile_m = 16U;
  constexpr uint32_t tile_n = 16U;
  constexpr uint32_t column_tiles = 4U;
  constexpr uint32_t fragment_k = 16U;
  constexpr uint32_t values_per_group = 4U;
  constexpr uint32_t scale_blocks_per_stage = StageK / fragment_k;
  constexpr uint32_t groups_per_row = StageK / values_per_group;
  constexpr uint32_t rows_per_workgroup = waves_per_workgroup * tile_m;
  constexpr uint32_t columns_per_workgroup = column_tiles * tile_n;
  constexpr uint32_t tile_values = tile_m * StageK;
  constexpr uint32_t groups_per_tile = tile_values / values_per_group;
  constexpr uint32_t activation_groups = waves_per_workgroup * groups_per_tile;
  constexpr uint32_t weight_groups = column_tiles * groups_per_tile;
  constexpr uint32_t output_values = tile_m * tile_n;
  __shared__ __align__(4)
      rocwmma::float16_t activation_tile[waves_per_workgroup * tile_values];
  __shared__ __align__(4)
      rocwmma::float16_t weight_tile[column_tiles * tile_values];
  __shared__ float tensor_scale;
  using AFragment =
      rocwmma::fragment<rocwmma::matrix_a, tile_m, tile_n, fragment_k,
                        rocwmma::float16_t, rocwmma::row_major>;
  using BFragment =
      rocwmma::fragment<rocwmma::matrix_b, tile_m, tile_n, fragment_k,
                        rocwmma::float16_t, rocwmma::col_major>;
  using AccumulatorFragment = rocwmma::fragment<rocwmma::accumulator, tile_m,
                                                tile_n, fragment_k, float>;
  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (wave_width - 1U);
  const uint32_t wave = thread / wave_width;
  const uint64_t row_group_base =
      static_cast<uint64_t>(blockIdx.y) * rows_per_workgroup;
  const uint64_t row_tile_base =
      row_group_base + static_cast<uint64_t>(wave) * tile_m;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * columns_per_workgroup;
  const uint64_t blocks_per_row = k / fragment_k;
  const uint64_t stages =
      (blocks_per_row + scale_blocks_per_stage - 1U) / scale_blocks_per_stage;
  const uint64_t packed_row_bytes = k / 2U;
  if (thread == 0U) {
    tensor_scale = weight_tensor_scale[0] * input_tensor_scale[0];
  }
  AccumulatorFragment contributions[column_tiles];
#pragma unroll
  for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
    rocwmma::fill_fragment(contributions[column_tile], 0.0F);
  }
  for (uint64_t stage = 0U; stage < stages; ++stage) {
    const uint64_t inner_base = stage * StageK;
    auto *const activation_dwords =
        reinterpret_cast<uint32_t *>(activation_tile);
    auto *const weight_dwords = reinterpret_cast<uint32_t *>(weight_tile);

    // One thread per four values: one aligned 16-bit load, four exact FP16
    // encodings, two half2 scale multiplies, and two adjacent LDS dwords.
    for (uint32_t group = thread; group < activation_groups;
         group += blockDim.x) {
      const uint32_t source_wave = group / groups_per_tile;
      const uint32_t tile_group = group - source_wave * groups_per_tile;
      const uint32_t local_row = tile_group / groups_per_row;
      const uint32_t local_group = tile_group - local_row * groups_per_row;
      const uint32_t local_inner = local_group * values_per_group;
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      const uint64_t inner = inner_base + local_inner;
      uint16_t packed = 0U;
      uint8_t scale_code = 0U;
      if (row < m && inner + values_per_group <= k) {
        packed = __builtin_nontemporal_load(reinterpret_cast<const uint16_t *>(
            packed_activation + row * packed_row_bytes + inner / 2U));
        scale_code = __builtin_nontemporal_load(activation_block_scales +
                                                row * blocks_per_row +
                                                inner / fragment_k);
      }
      const Fp16x4Bits expanded = e2m1x4_to_fp16x2_bits(packed);
      const __half scale = __float2half_rn(e4m3fn_to_float(scale_code));
      const __half2 broadcast = __halves2half2(scale, scale);
      const __half2 low =
          __hmul2(sllm_lowp::fp16x2_bits_to_half2(expanded.low), broadcast);
      const __half2 high =
          __hmul2(sllm_lowp::fp16x2_bits_to_half2(expanded.high), broadcast);
      activation_dwords[group * 2U] = half2_bits(low);
      activation_dwords[group * 2U + 1U] = half2_bits(high);
    }
    for (uint32_t group = thread; group < weight_groups; group += blockDim.x) {
      const uint32_t column_tile = group / groups_per_tile;
      const uint32_t tile_group = group - column_tile * groups_per_tile;
      const uint32_t local_column = tile_group / groups_per_row;
      const uint32_t local_group = tile_group - local_column * groups_per_row;
      const uint32_t local_inner = local_group * values_per_group;
      const uint64_t column = column_base +
                              static_cast<uint64_t>(column_tile) * tile_n +
                              local_column;
      const uint64_t inner = inner_base + local_inner;
      uint16_t packed = 0U;
      uint8_t scale_code = 0U;
      if (column < n && inner + values_per_group <= k) {
        packed = __builtin_nontemporal_load(reinterpret_cast<const uint16_t *>(
            packed_weight + column * packed_row_bytes + inner / 2U));
        scale_code = __builtin_nontemporal_load(
            weight_block_scales + column * blocks_per_row + inner / fragment_k);
      }
      const Fp16x4Bits expanded = e2m1x4_to_fp16x2_bits(packed);
      const __half scale = __float2half_rn(e4m3fn_to_float(scale_code));
      const __half2 broadcast = __halves2half2(scale, scale);
      const __half2 low =
          __hmul2(sllm_lowp::fp16x2_bits_to_half2(expanded.low), broadcast);
      const __half2 high =
          __hmul2(sllm_lowp::fp16x2_bits_to_half2(expanded.high), broadcast);
      weight_dwords[group * 2U] = half2_bits(low);
      weight_dwords[group * 2U + 1U] = half2_bits(high);
    }
    __syncthreads();
#pragma unroll
    for (uint32_t scale_block = 0U; scale_block < scale_blocks_per_stage;
         ++scale_block) {
      AFragment activation_fragment;
      rocwmma::load_matrix_sync(activation_fragment,
                                activation_tile + wave * tile_values +
                                    scale_block * fragment_k,
                                StageK);
#pragma unroll
      for (uint32_t column_tile = 0U; column_tile < column_tiles;
           ++column_tile) {
        BFragment weight_fragment;
        rocwmma::load_matrix_sync(weight_fragment,
                                  weight_tile + column_tile * tile_values +
                                      scale_block * fragment_k,
                                  StageK);
        rocwmma::mma_sync(contributions[column_tile], activation_fragment,
                          weight_fragment, contributions[column_tile]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
    const auto contribution_row_major =
        rocwmma::apply_data_layout<rocwmma::row_major>(
            contributions[column_tile]);
#pragma unroll
    for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
      const uint32_t local_row =
          (lane / tile_n) * (output_values / wave_width) + slot;
      const uint32_t local_column = lane % tile_n;
      const uint64_t row = row_tile_base + local_row;
      const uint64_t column = column_base + column_tile * tile_n + local_column;
      if (row < m && column < n) {
        output[row * n + column] =
            bf16_rne(contribution_row_major[slot] * tensor_scale);
      }
    }
  }
#else
  (void)packed_activation;
  (void)activation_block_scales;
  (void)packed_weight;
  (void)weight_block_scales;
  (void)weight_tensor_scale;
  (void)input_tensor_scale;
  (void)output;
  (void)m;
  (void)k;
  (void)n;
#endif
}

struct DeviceBuffers final {
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
  std::size_t allocations = 0U;
};

struct HostInputs final {
  std::vector<uint8_t> activation;
  std::vector<uint8_t> activation_scales;
  std::vector<uint8_t> weight;
  std::vector<uint8_t> weight_scales;
};

struct ResourceInfo final {
  bool available = false;
  int vgpr = -1;
  std::size_t lds = 0U;
  std::size_t scratch = 0U;
  int max_threads = 0;
  int active_blocks_per_cu = 0;
  double occupancy = 0.0;
};

struct Measurement final {
  bool ran = false;
  bool deterministic = false;
  std::array<float, kMeasured> samples_us{};
  float median_us = 0.0F;
  float minimum_us = 0.0F;
  float maximum_us = 0.0F;
  std::size_t repeat_mismatches = 0U;
  std::vector<uint16_t> output;
};

struct ShapeResult final {
  Shape shape{};
  std::array<Measurement, 4> measurements;
};

struct CleanupTotals final {
  std::size_t allocations = 0U;
  std::size_t frees = 0U;
  bool ok = true;
};

std::size_t variant_index(const Variant variant) {
  return static_cast<std::size_t>(variant);
}

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::cerr << operation << " failed: " << hipGetErrorName(status) << " ("
            << hipGetErrorString(status) << ")\n";
  return false;
}

bool exact_gfx1201(const char *const arch) {
  if (arch == nullptr) {
    return false;
  }
  const std::string_view value(arch);
  constexpr std::string_view target = "gfx1201";
  return value == target ||
         (value.size() > target.size() && value.starts_with(target) &&
          value[target.size()] == ':');
}

bool parse_device(const char *const text, int *const device) {
  if (text == nullptr || device == nullptr) {
    return false;
  }
  const std::string_view input(text);
  int parsed = -1;
  const auto result =
      std::from_chars(input.data(), input.data() + input.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != input.data() + input.size() ||
      parsed < 0) {
    return false;
  }
  *device = parsed;
  return true;
}

uint32_t mix32(uint32_t value) {
  value ^= value >> 16U;
  value *= UINT32_C(0x7feb352d);
  value ^= value >> 15U;
  value *= UINT32_C(0x846ca68b);
  return value ^ (value >> 16U);
}

uint8_t positive_finite_e4m3(const uint64_t index, const uint32_t seed) {
  return static_cast<uint8_t>((index * UINT64_C(73) + seed) % UINT64_C(127));
}

bool shape_byte_sizes(const Shape &shape, std::size_t *const activation,
                      std::size_t *const activation_scales,
                      std::size_t *const weight,
                      std::size_t *const weight_scales,
                      std::size_t *const output) {
  if (shape.m == 0U || shape.n == 0U || shape.k == 0U ||
      (shape.k % UINT64_C(64)) != 0U || shape.m > SIZE_MAX / shape.k ||
      shape.n > SIZE_MAX / shape.k || shape.m > SIZE_MAX / shape.n ||
      shape.m * shape.n > SIZE_MAX / sizeof(uint16_t)) {
    return false;
  }
  *activation = static_cast<std::size_t>(shape.m * shape.k / 2U);
  *activation_scales = static_cast<std::size_t>(shape.m * shape.k / 16U);
  *weight = static_cast<std::size_t>(shape.n * shape.k / 2U);
  *weight_scales = static_cast<std::size_t>(shape.n * shape.k / 16U);
  *output = static_cast<std::size_t>(shape.m * shape.n * sizeof(uint16_t));
  return true;
}

HostInputs make_inputs(const Shape &shape) {
  const uint64_t blocks = shape.k / UINT64_C(16);
  HostInputs inputs;
  inputs.activation.assign(static_cast<std::size_t>(shape.m * shape.k / 2U),
                           0U);
  inputs.activation_scales.resize(static_cast<std::size_t>(shape.m * blocks));
  inputs.weight.assign(static_cast<std::size_t>(shape.n * shape.k / 2U), 0U);
  inputs.weight_scales.resize(static_cast<std::size_t>(shape.n * blocks));
  for (uint64_t row = 0U; row < shape.m; ++row) {
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      const uint32_t ordinal = static_cast<uint32_t>(row * shape.k + inner);
      const uint8_t code =
          static_cast<uint8_t>(mix32(ordinal ^ kSeed) & UINT32_C(0x0f));
      const std::size_t index =
          static_cast<std::size_t>(row * shape.k / 2U + inner / 2U);
      if ((inner & 1U) == 0U) {
        inputs.activation[index] = code;
      } else {
        inputs.activation[index] |= static_cast<uint8_t>(code << 4U);
      }
    }
    for (uint64_t block = 0U; block < blocks; ++block) {
      inputs.activation_scales[static_cast<std::size_t>(row * blocks + block)] =
          positive_finite_e4m3(row * blocks + block, kSeed);
    }
  }
  for (uint64_t column = 0U; column < shape.n; ++column) {
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      const uint32_t ordinal = static_cast<uint32_t>(column * shape.k + inner);
      const uint8_t code = static_cast<uint8_t>(
          mix32(ordinal ^ kSeed ^ UINT32_C(0x9e3779b9)) & UINT32_C(0x0f));
      const std::size_t index =
          static_cast<std::size_t>(column * shape.k / 2U + inner / 2U);
      if ((inner & 1U) == 0U) {
        inputs.weight[index] = code;
      } else {
        inputs.weight[index] |= static_cast<uint8_t>(code << 4U);
      }
    }
    for (uint64_t block = 0U; block < blocks; ++block) {
      inputs.weight_scales[static_cast<std::size_t>(column * blocks + block)] =
          positive_finite_e4m3(column * blocks + block,
                               kSeed ^ UINT32_C(0xa5a5a5a5));
    }
  }
  return inputs;
}

bool device_allocate(void **const pointer, const std::size_t bytes,
                     DeviceBuffers *const buffers) {
  if (!hip_ok(hipMalloc(pointer, bytes), "hipMalloc")) {
    return false;
  }
  ++buffers->allocations;
  return true;
}

bool allocate_and_upload(const Shape &shape, const HostInputs &inputs,
                         DeviceBuffers *const buffers) {
  std::size_t activation_bytes = 0U;
  std::size_t activation_scale_bytes = 0U;
  std::size_t weight_bytes = 0U;
  std::size_t weight_scale_bytes = 0U;
  std::size_t output_bytes = 0U;
  if (!shape_byte_sizes(shape, &activation_bytes, &activation_scale_bytes,
                        &weight_bytes, &weight_scale_bytes, &output_bytes)) {
    return false;
  }
  return device_allocate(reinterpret_cast<void **>(&buffers->activation),
                         activation_bytes, buffers) &&
         device_allocate(reinterpret_cast<void **>(&buffers->activation_scales),
                         activation_scale_bytes, buffers) &&
         device_allocate(reinterpret_cast<void **>(&buffers->weight),
                         weight_bytes, buffers) &&
         device_allocate(reinterpret_cast<void **>(&buffers->weight_scales),
                         weight_scale_bytes, buffers) &&
         device_allocate(
             reinterpret_cast<void **>(&buffers->weight_tensor_scale),
             sizeof(float), buffers) &&
         device_allocate(
             reinterpret_cast<void **>(&buffers->input_tensor_scale),
             sizeof(float), buffers) &&
         device_allocate(reinterpret_cast<void **>(&buffers->output),
                         output_bytes, buffers) &&
         hip_ok(hipStreamCreate(&buffers->stream), "hipStreamCreate") &&
         hip_ok(hipEventCreate(&buffers->start), "hipEventCreate start") &&
         hip_ok(hipEventCreate(&buffers->stop), "hipEventCreate stop") &&
         hip_ok(hipMemcpy(buffers->activation, inputs.activation.data(),
                          activation_bytes, hipMemcpyHostToDevice),
                "upload activation") &&
         hip_ok(hipMemcpy(buffers->activation_scales,
                          inputs.activation_scales.data(),
                          activation_scale_bytes, hipMemcpyHostToDevice),
                "upload activation scales") &&
         hip_ok(hipMemcpy(buffers->weight, inputs.weight.data(), weight_bytes,
                          hipMemcpyHostToDevice),
                "upload weight") &&
         hip_ok(hipMemcpy(buffers->weight_scales, inputs.weight_scales.data(),
                          weight_scale_bytes, hipMemcpyHostToDevice),
                "upload weight scales") &&
         hip_ok(hipMemcpy(buffers->weight_tensor_scale, &kWeightTensorScale,
                          sizeof(float), hipMemcpyHostToDevice),
                "upload weight tensor scale") &&
         hip_ok(hipMemcpy(buffers->input_tensor_scale, &kInputTensorScale,
                          sizeof(float), hipMemcpyHostToDevice),
                "upload input tensor scale") &&
         hip_ok(hipMemset(buffers->output, 0, output_bytes), "clear output");
}

void cleanup(DeviceBuffers *const buffers, CleanupTotals *const totals) {
  totals->allocations += buffers->allocations;
  const auto destroy_event = [&](hipEvent_t *const event) {
    if (*event != nullptr) {
      totals->ok =
          hip_ok(hipEventDestroy(*event), "hipEventDestroy") && totals->ok;
      *event = nullptr;
    }
  };
  destroy_event(&buffers->stop);
  destroy_event(&buffers->start);
  if (buffers->stream != nullptr) {
    totals->ok =
        hip_ok(hipStreamDestroy(buffers->stream), "hipStreamDestroy") &&
        totals->ok;
    buffers->stream = nullptr;
  }
  const auto free_device = [&](auto **const pointer) {
    if (*pointer != nullptr) {
      if (hip_ok(hipFree(*pointer), "hipFree")) {
        ++totals->frees;
      } else {
        totals->ok = false;
      }
      *pointer = nullptr;
    }
  };
  free_device(&buffers->output);
  free_device(&buffers->input_tensor_scale);
  free_device(&buffers->weight_tensor_scale);
  free_device(&buffers->weight_scales);
  free_device(&buffers->weight);
  free_device(&buffers->activation_scales);
  free_device(&buffers->activation);
}

const void *kernel_pointer(const Variant variant) {
  switch (variant) {
  case Variant::Id64:
    return reinterpret_cast<const void *>(id64_control_kernel);
  case Variant::Id69:
    return reinterpret_cast<const void *>(id69_baseline_kernel);
  case Variant::Vector32:
    return reinterpret_cast<const void *>(vector_ingress_kernel<32U>);
  case Variant::Vector64:
    return reinterpret_cast<const void *>(vector_ingress_kernel<64U>);
  }
  return nullptr;
}

ResourceInfo resource_info(const Variant variant,
                           const hipDeviceProp_t &properties) {
  ResourceInfo result;
  hipFuncAttributes attributes{};
  const void *const kernel = kernel_pointer(variant);
  if (!hip_ok(hipFuncGetAttributes(&attributes, kernel),
              "hipFuncGetAttributes")) {
    return result;
  }
  int active_blocks = 0;
  if (!hip_ok(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                  &active_blocks, kernel, static_cast<int>(kThreads), 0U),
              "hipOccupancyMaxActiveBlocksPerMultiprocessor")) {
    return result;
  }
  result.available = true;
  result.vgpr = attributes.numRegs;
  result.lds = attributes.sharedSizeBytes;
  result.scratch = attributes.localSizeBytes;
  result.max_threads = attributes.maxThreadsPerBlock;
  result.active_blocks_per_cu = active_blocks;
  result.occupancy = properties.maxThreadsPerMultiProcessor == 0
                         ? 0.0
                         : static_cast<double>(active_blocks * kThreads) /
                               properties.maxThreadsPerMultiProcessor;
  std::cout << "resources variant=" << variant_name(variant)
            << " vgpr=" << result.vgpr << " lds=" << result.lds
            << " scratch_per_thread=" << result.scratch
            << " max_threads=" << result.max_threads
            << " active_blocks_per_cu=" << result.active_blocks_per_cu
            << " active_waves_per_cu=" << result.active_blocks_per_cu * 8
            << " occupancy=" << std::fixed << std::setprecision(6)
            << result.occupancy << "\n";
  return result;
}

bool launch(const Variant variant, const Shape &shape,
            const DeviceBuffers &buffers) {
  const dim3 grid(static_cast<uint32_t>((shape.n + 63U) / 64U),
                  static_cast<uint32_t>((shape.m + 127U) / 128U));
  const dim3 block(kThreads);
  switch (variant) {
  case Variant::Id64:
    hipLaunchKernelGGL(id64_control_kernel, grid, block, 0U, buffers.stream,
                       buffers.activation, buffers.activation_scales,
                       buffers.weight, buffers.weight_scales,
                       buffers.weight_tensor_scale, buffers.input_tensor_scale,
                       buffers.output, shape.m, shape.k, shape.n);
    break;
  case Variant::Id69:
    hipLaunchKernelGGL(id69_baseline_kernel, grid, block, 0U, buffers.stream,
                       buffers.activation, buffers.activation_scales,
                       buffers.weight, buffers.weight_scales,
                       buffers.weight_tensor_scale, buffers.input_tensor_scale,
                       buffers.output, shape.m, shape.k, shape.n);
    break;
  case Variant::Vector32:
    hipLaunchKernelGGL(
        (vector_ingress_kernel<32U>), grid, block, 0U, buffers.stream,
        buffers.activation, buffers.activation_scales, buffers.weight,
        buffers.weight_scales, buffers.weight_tensor_scale,
        buffers.input_tensor_scale, buffers.output, shape.m, shape.k, shape.n);
    break;
  case Variant::Vector64:
    hipLaunchKernelGGL(
        (vector_ingress_kernel<64U>), grid, block, 0U, buffers.stream,
        buffers.activation, buffers.activation_scales, buffers.weight,
        buffers.weight_scales, buffers.weight_tensor_scale,
        buffers.input_tensor_scale, buffers.output, shape.m, shape.k, shape.n);
    break;
  }
  return hip_ok(hipGetLastError(), "kernel launch");
}

bool measure(const Variant variant, const Shape &shape,
             const DeviceBuffers &buffers, Measurement *const result) {
  const std::size_t elements = static_cast<std::size_t>(shape.m * shape.n);
  const std::size_t bytes = elements * sizeof(uint16_t);
  for (uint32_t warmup = 0U; warmup < kWarmups; ++warmup) {
    if (!launch(variant, shape, buffers)) {
      return false;
    }
  }
  if (!hip_ok(hipStreamSynchronize(buffers.stream), "warmup synchronize")) {
    return false;
  }
  result->output.resize(elements);
  std::vector<uint16_t> current(elements);
  for (uint32_t iteration = 0U; iteration < kMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(buffers.start, buffers.stream),
                "timing start") ||
        !launch(variant, shape, buffers) ||
        !hip_ok(hipEventRecord(buffers.stop, buffers.stream), "timing stop") ||
        !hip_ok(hipEventSynchronize(buffers.stop), "timing synchronize")) {
      return false;
    }
    float milliseconds = 0.0F;
    if (!hip_ok(hipEventElapsedTime(&milliseconds, buffers.start, buffers.stop),
                "timing elapsed") ||
        !hip_ok(hipMemcpy(current.data(), buffers.output, bytes,
                          hipMemcpyDeviceToHost),
                "copy output")) {
      return false;
    }
    result->samples_us[iteration] = milliseconds * 1000.0F;
    if (iteration == 0U) {
      result->output = current;
    } else {
      for (std::size_t index = 0U; index < current.size(); ++index) {
        result->repeat_mismatches +=
            static_cast<std::size_t>(current[index] != result->output[index]);
      }
    }
  }
  std::array<float, kMeasured> sorted = result->samples_us;
  std::sort(sorted.begin(), sorted.end());
  result->minimum_us = sorted.front();
  result->median_us = sorted[sorted.size() / 2U];
  result->maximum_us = sorted.back();
  result->ran = true;
  result->deterministic = result->repeat_mismatches == 0U;
  std::cout << "timing shape=" << shape.name
            << " variant=" << variant_name(variant) << " warmups=" << kWarmups
            << " measured=" << kMeasured << " samples_us=";
  for (std::size_t index = 0U; index < result->samples_us.size(); ++index) {
    if (index != 0U) {
      std::cout << ',';
    }
    std::cout << std::fixed << std::setprecision(3)
              << result->samples_us[index];
  }
  std::cout << " median_us=" << result->median_us
            << " min_us=" << result->minimum_us
            << " max_us=" << result->maximum_us
            << " repeat_bf16_mismatches=" << result->repeat_mismatches
            << " deterministic=" << (result->deterministic ? "PASS" : "FAIL")
            << "\n";
  return result->deterministic;
}

float host_e2m1(const uint8_t code) {
  constexpr std::array<float, 8> positive = {0.0F, 0.5F, 1.0F, 1.5F,
                                             2.0F, 3.0F, 4.0F, 6.0F};
  const float value = positive[code & UINT8_C(7)];
  return (code & UINT8_C(8)) == 0U ? value : -value;
}

float host_bf16_to_float(const uint16_t bits) {
  const uint32_t expanded = static_cast<uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &expanded, sizeof(value));
  return value;
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

uint32_t ordered_bf16(const uint16_t bits) {
  if ((bits & UINT16_C(0x7fff)) == 0U) {
    return UINT32_C(0x8000);
  }
  return (bits & UINT16_C(0x8000)) != 0U
             ? static_cast<uint16_t>(~bits)
             : static_cast<uint16_t>(bits | UINT16_C(0x8000));
}

uint64_t hash_bf16(const std::span<const uint16_t> values) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (const uint16_t value : values) {
    hash ^= static_cast<uint8_t>(value & UINT16_C(0xff));
    hash *= UINT64_C(1099511628211);
    hash ^= static_cast<uint8_t>(value >> 8U);
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

struct OraclePoint final {
  std::size_t index;
  double expected;
  double absolute_sum;
  uint16_t expected_bf16;
};

std::array<OraclePoint, kOracleSamples> host_oracle(const Shape &shape,
                                                    const HostInputs &inputs) {
  std::array<OraclePoint, kOracleSamples> result{};
  const std::size_t elements = static_cast<std::size_t>(shape.m * shape.n);
  const uint64_t blocks = shape.k / UINT64_C(16);
  for (uint32_t sample = 0U; sample < kOracleSamples; ++sample) {
    const std::size_t output_index = static_cast<std::size_t>(sample) *
                                     (elements - 1U) / (kOracleSamples - 1U);
    const uint64_t row = output_index / shape.n;
    const uint64_t column = output_index - row * shape.n;
    double expected = 0.0;
    double absolute_sum = 0.0;
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      const uint8_t activation_pair =
          inputs.activation[static_cast<std::size_t>(row * shape.k / 2U +
                                                     inner / 2U)];
      const uint8_t weight_pair = inputs.weight[static_cast<std::size_t>(
          column * shape.k / 2U + inner / 2U)];
      const uint8_t activation_code = (inner & 1U) == 0U
                                          ? activation_pair & UINT8_C(0x0f)
                                          : activation_pair >> 4U;
      const uint8_t weight_code =
          (inner & 1U) == 0U ? weight_pair & UINT8_C(0x0f) : weight_pair >> 4U;
      const double term =
          static_cast<double>(host_e2m1(activation_code)) *
          e4m3fn_to_float(inputs.activation_scales[static_cast<std::size_t>(
              row * blocks + inner / UINT64_C(16))]) *
          static_cast<double>(host_e2m1(weight_code)) *
          e4m3fn_to_float(inputs.weight_scales[static_cast<std::size_t>(
              column * blocks + inner / UINT64_C(16))]);
      expected += term;
      absolute_sum += std::abs(term);
    }
    constexpr double tensor_scale =
        static_cast<double>(kWeightTensorScale) * kInputTensorScale;
    expected *= tensor_scale;
    absolute_sum *= tensor_scale;
    result[sample] = OraclePoint{output_index, expected, absolute_sum,
                                 host_bf16_rne(static_cast<float>(expected))};
  }
  return result;
}

bool check_host_oracle(const Shape &shape, const Variant variant,
                       const std::array<OraclePoint, kOracleSamples> &oracle,
                       const std::vector<uint16_t> &output) {
  std::size_t bf16_mismatches = 0U;
  uint32_t max_ulp = 0U;
  double max_abs = 0.0;
  double max_normalized = 0.0;
  for (const OraclePoint &point : oracle) {
    const uint16_t observed_bits = output[point.index];
    const double observed = host_bf16_to_float(observed_bits);
    const double absolute_error = std::abs(observed - point.expected);
    max_abs = std::max(max_abs, absolute_error);
    max_normalized =
        std::max(max_normalized,
                 absolute_error / std::max(point.absolute_sum,
                                           std::numeric_limits<double>::min()));
    bf16_mismatches +=
        static_cast<std::size_t>(observed_bits != point.expected_bf16);
    const uint32_t lhs = ordered_bf16(observed_bits);
    const uint32_t rhs = ordered_bf16(point.expected_bf16);
    max_ulp = std::max(max_ulp, lhs > rhs ? lhs - rhs : rhs - lhs);
  }
  const bool pass = max_normalized <= 0.01;
  std::cout << "host_oracle shape=" << shape.name
            << " variant=" << variant_name(variant)
            << " samples=" << kOracleSamples
            << " bf16_mismatches=" << bf16_mismatches
            << " max_bf16_ulp=" << max_ulp
            << " max_abs=" << std::setprecision(10) << max_abs
            << " max_normalized_error=" << max_normalized
            << " tolerance=0.01 status=" << (pass ? "PASS" : "FAIL") << "\n";
  return pass;
}

struct Comparison final {
  std::size_t mismatches = 0U;
  uint32_t max_ulp = 0U;
  double max_abs = 0.0;
  double max_rel = 0.0;
};

Comparison compare(const std::vector<uint16_t> &reference,
                   const std::vector<uint16_t> &candidate) {
  Comparison result;
  for (std::size_t index = 0U; index < reference.size(); ++index) {
    result.mismatches +=
        static_cast<std::size_t>(reference[index] != candidate[index]);
    const float lhs_value = host_bf16_to_float(reference[index]);
    const float rhs_value = host_bf16_to_float(candidate[index]);
    const double absolute =
        std::abs(static_cast<double>(lhs_value) - rhs_value);
    result.max_abs = std::max(result.max_abs, absolute);
    result.max_rel =
        std::max(result.max_rel,
                 absolute / std::max(1.0e-100,
                                     std::abs(static_cast<double>(lhs_value))));
    const uint32_t lhs = ordered_bf16(reference[index]);
    const uint32_t rhs = ordered_bf16(candidate[index]);
    result.max_ulp =
        std::max(result.max_ulp, lhs > rhs ? lhs - rhs : rhs - lhs);
  }
  return result;
}

void print_comparison(const Shape &shape, const char *const name,
                      const Comparison &comparison) {
  std::cout << "compare shape=" << shape.name << " pair=" << name
            << " bf16_mismatches=" << comparison.mismatches
            << " max_bf16_ulp=" << comparison.max_ulp
            << " max_abs=" << std::setprecision(10) << comparison.max_abs
            << " max_rel=" << comparison.max_rel << "\n";
}

bool run_shape(const Shape &shape, const bool run_stage64,
               ShapeResult *const result, CleanupTotals *const cleanup_totals) {
  std::cout << "shape_begin name=" << shape.name << " m=" << shape.m
            << " k=" << shape.k << " n=" << shape.n
            << " occurrences=" << shape.occurrences << "\n";
  const HostInputs inputs = make_inputs(shape);
  DeviceBuffers buffers;
  if (!allocate_and_upload(shape, inputs, &buffers)) {
    cleanup(&buffers, cleanup_totals);
    return false;
  }
  result->shape = shape;
  bool ok = true;
  for (const Variant variant : kVariants) {
    if (variant == Variant::Vector64 && !run_stage64) {
      continue;
    }
    Measurement &measurement = result->measurements[variant_index(variant)];
    if (!measure(variant, shape, buffers, &measurement)) {
      ok = false;
      break;
    }
    std::cout << "signature shape=" << shape.name
              << " variant=" << variant_name(variant) << " bf16_fnv64=0x"
              << std::hex << hash_bf16(measurement.output) << std::dec << "\n";
  }
  if (ok) {
    const auto oracle = host_oracle(shape, inputs);
    for (const Variant variant : kVariants) {
      const Measurement &measurement =
          result->measurements[variant_index(variant)];
      if (measurement.ran &&
          !check_host_oracle(shape, variant, oracle, measurement.output)) {
        ok = false;
      }
    }
    const auto &id64 = result->measurements[variant_index(Variant::Id64)];
    const auto &id69 = result->measurements[variant_index(Variant::Id69)];
    const auto &vector32 =
        result->measurements[variant_index(Variant::Vector32)];
    const Comparison id64_id69 = compare(id64.output, id69.output);
    const Comparison id64_vector32 = compare(id64.output, vector32.output);
    const Comparison id69_vector32 = compare(id69.output, vector32.output);
    print_comparison(shape, "id64-vs-id69", id64_id69);
    print_comparison(shape, "id64-vs-vector32", id64_vector32);
    print_comparison(shape, "id69-vs-vector32", id69_vector32);
    ok = id69_vector32.mismatches == 0U && ok;
    const auto &vector64 =
        result->measurements[variant_index(Variant::Vector64)];
    if (vector64.ran) {
      const Comparison id64_vector64 = compare(id64.output, vector64.output);
      const Comparison id69_vector64 = compare(id69.output, vector64.output);
      print_comparison(shape, "id64-vs-vector64", id64_vector64);
      print_comparison(shape, "id69-vs-vector64", id69_vector64);
      ok = id69_vector64.mismatches == 0U && ok;
    }
  }
  cleanup(&buffers, cleanup_totals);
  std::cout << "shape_end name=" << shape.name
            << " status=" << (ok ? "PASS" : "FAIL") << "\n";
  return ok;
}

void print_weighted_results(
    const std::array<ShapeResult, kShapes.size()> &results,
    const bool run_stage64) {
  for (const Variant variant : kVariants) {
    if (variant == Variant::Vector64 && !run_stage64) {
      continue;
    }
    double weighted_total_us = 0.0;
    uint64_t total_weight = 0U;
    for (const ShapeResult &result : results) {
      weighted_total_us +=
          result.measurements[variant_index(variant)].median_us *
          result.shape.occurrences;
      total_weight += result.shape.occurrences;
    }
    const double weighted_mean_us = weighted_total_us / total_weight;
    const double id64_total = [&]() {
      double value = 0.0;
      for (const ShapeResult &result : results) {
        value += result.measurements[variant_index(Variant::Id64)].median_us *
                 result.shape.occurrences;
      }
      return value;
    }();
    const double id69_total = [&]() {
      double value = 0.0;
      for (const ShapeResult &result : results) {
        value += result.measurements[variant_index(Variant::Id69)].median_us *
                 result.shape.occurrences;
      }
      return value;
    }();
    std::cout << "weighted variant=" << variant_name(variant)
              << " shapes=6 qwen_projection_weight=" << total_weight
              << " weighted_total_us=" << std::fixed << std::setprecision(3)
              << weighted_total_us << " weighted_mean_us=" << weighted_mean_us
              << " speedup_vs_id64=" << id64_total / weighted_total_us
              << " speedup_vs_id69=" << id69_total / weighted_total_us << "\n";
  }
  for (const uint64_t m : {UINT64_C(128), UINT64_C(512), UINT64_C(1024)}) {
    for (const Variant variant : kVariants) {
      if (variant == Variant::Vector64 && !run_stage64) {
        continue;
      }
      double total = 0.0;
      uint64_t weight = 0U;
      for (const ShapeResult &result : results) {
        if (result.shape.m == m) {
          total += result.measurements[variant_index(variant)].median_us *
                   result.shape.occurrences;
          weight += result.shape.occurrences;
        }
      }
      std::cout << "weighted_by_m m=" << m
                << " variant=" << variant_name(variant)
                << " qwen_projection_weight=" << weight
                << " weighted_mean_us=" << std::fixed << std::setprecision(3)
                << total / weight << "\n";
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  int device = 0;
  if (argc > 2 || (argc == 2 && !parse_device(argv[1], &device))) {
    std::cerr << "usage: phase78_nvfp4_gfx1201_wmma_scaled_ingress_probe "
                 "[DEVICE]\n";
    return EXIT_FAILURE;
  }
  if (!hip_ok(hipSetDevice(device), "hipSetDevice")) {
    return EXIT_FAILURE;
  }
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device),
              "hipGetDeviceProperties")) {
    return EXIT_FAILURE;
  }
  int runtime_version = 0;
  if (!hip_ok(hipRuntimeGetVersion(&runtime_version), "hipRuntimeGetVersion")) {
    return EXIT_FAILURE;
  }
  std::cout << "identity device=" << device
            << " arch=" << properties.gcnArchName
            << " hip_header=" << HIP_VERSION_MAJOR << '.' << HIP_VERSION_MINOR
            << '.' << HIP_VERSION_PATCH << " hip_runtime=" << runtime_version
            << " pci=" << std::hex << std::setw(4) << std::setfill('0')
            << properties.pciDomainID << ':' << std::setw(2)
            << properties.pciBusID << ':' << std::setw(2)
            << properties.pciDeviceID << std::dec << std::setfill(' ') << "\n";
  if (!exact_gfx1201(properties.gcnArchName)) {
    std::cerr << "exact gfx1201 is required\n";
    return EXIT_FAILURE;
  }

  std::array<ResourceInfo, 4> resources{};
  for (const Variant variant : kVariants) {
    resources[variant_index(variant)] = resource_info(variant, properties);
    if (!resources[variant_index(variant)].available) {
      std::cout << "PHASE78_NVFP4_SCALED_INGRESS_RESULT=FAIL\n";
      return EXIT_FAILURE;
    }
  }
  const ResourceInfo &stage64 = resources[variant_index(Variant::Vector64)];
  const bool run_stage64 = stage64.lds <= kMaximumCandidateLds &&
                           stage64.scratch == 0U &&
                           stage64.active_blocks_per_cu > 0;
  std::cout << "stage64_eligibility lds=" << stage64.lds
            << " limit=" << kMaximumCandidateLds
            << " scratch_per_thread=" << stage64.scratch
            << " active_blocks_per_cu=" << stage64.active_blocks_per_cu
            << " status=" << (run_stage64 ? "INCLUDE" : "SKIP") << "\n";

  std::array<ShapeResult, kShapes.size()> results{};
  CleanupTotals cleanup_totals;
  bool ok = true;
  for (std::size_t index = 0U; index < kShapes.size(); ++index) {
    if (!run_shape(kShapes[index], run_stage64, &results[index],
                   &cleanup_totals)) {
      ok = false;
      break;
    }
  }
  if (ok) {
    print_weighted_results(results, run_stage64);
  }
  const bool cleanup_ok =
      cleanup_totals.ok && cleanup_totals.allocations == cleanup_totals.frees;
  std::cout << "cleanup allocations=" << cleanup_totals.allocations
            << " frees=" << cleanup_totals.frees
            << " status=" << (cleanup_ok ? "PASS" : "FAIL") << "\n";
  ok = cleanup_ok && ok;
  std::cout << "PHASE78_NVFP4_SCALED_INGRESS_RESULT=" << (ok ? "PASS" : "FAIL")
            << "\n";
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
