// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase78-nvfp4-byte-permute-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// 3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70, ggml/src/ggml-cuda/vecdotq.cuh
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

// Phase 78 ID84 NVFP4 W4A4 decode kernels using a bank-padded FP32 scale LUT.
//
// This include is intentionally self-contained.  It is included by the HIP
// matmul translation unit after the existing ID67/ID73 kernels are available,
// but does not refer to their helpers.  The two kernels below retain the
// probe's E2M1 byte permutation, four DP4A operations per block16, FP32
// block-scale accumulation, wave reduction, and BF16-RNE epilogue order.
// Only the E4M3FN block-scale loads use a 256-entry FP32 LUT in bank-padded
// LDS.  The FP16-to-FP32 conversion during LUT initialization matches the
// Phase 78 LDS-FP32 probe candidate exactly for all E4M3FN encodings.

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_id84_nvfp4_scale_lut_detail {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kWave = 32U;
constexpr uint32_t kWaves = kThreads / kWave;
constexpr uint32_t kColumnsPerWave = 4U;
constexpr uint32_t kColumnsPerWorkgroup = kWaves * kColumnsPerWave;
constexpr uint32_t kScaleLutEntries = 256U;
// One padding element after every wave-sized group avoids a same-index bank
// conflict when all lanes read a scale code from LDS.
constexpr uint32_t kScaleLutStride = 33U;
constexpr uint32_t kScaleLutSlots =
    kScaleLutEntries + kScaleLutEntries / (kScaleLutStride - 1U);
constexpr uint64_t kDecodeMaxK = UINT64_C(17408);

// This is a compile-time copy of the old probe's device constant table.  The
// old probe fills this table from the host with
// sllm_lowp::e4m3fn_to_fp16_bits(); keeping the same 256 half encodings here
// lets the focused variant measure the table-reference path without a host
// hipMemcpyToSymbol setup.  0x7f/0xff retain the signed quiet-NaN encodings.
__device__ __constant__ uint16_t sllm_id84_nvfp4_scale_fp16_lut[256] = {
    UINT16_C(0x0000), UINT16_C(0x1800), UINT16_C(0x1c00), UINT16_C(0x1e00),
    UINT16_C(0x2000), UINT16_C(0x2100), UINT16_C(0x2200), UINT16_C(0x2300),
    UINT16_C(0x2400), UINT16_C(0x2480), UINT16_C(0x2500), UINT16_C(0x2580),
    UINT16_C(0x2600), UINT16_C(0x2680), UINT16_C(0x2700), UINT16_C(0x2780),
    UINT16_C(0x2800), UINT16_C(0x2880), UINT16_C(0x2900), UINT16_C(0x2980),
    UINT16_C(0x2a00), UINT16_C(0x2a80), UINT16_C(0x2b00), UINT16_C(0x2b80),
    UINT16_C(0x2c00), UINT16_C(0x2c80), UINT16_C(0x2d00), UINT16_C(0x2d80),
    UINT16_C(0x2e00), UINT16_C(0x2e80), UINT16_C(0x2f00), UINT16_C(0x2f80),
    UINT16_C(0x3000), UINT16_C(0x3080), UINT16_C(0x3100), UINT16_C(0x3180),
    UINT16_C(0x3200), UINT16_C(0x3280), UINT16_C(0x3300), UINT16_C(0x3380),
    UINT16_C(0x3400), UINT16_C(0x3480), UINT16_C(0x3500), UINT16_C(0x3580),
    UINT16_C(0x3600), UINT16_C(0x3680), UINT16_C(0x3700), UINT16_C(0x3780),
    UINT16_C(0x3800), UINT16_C(0x3880), UINT16_C(0x3900), UINT16_C(0x3980),
    UINT16_C(0x3a00), UINT16_C(0x3a80), UINT16_C(0x3b00), UINT16_C(0x3b80),
    UINT16_C(0x3c00), UINT16_C(0x3c80), UINT16_C(0x3d00), UINT16_C(0x3d80),
    UINT16_C(0x3e00), UINT16_C(0x3e80), UINT16_C(0x3f00), UINT16_C(0x3f80),
    UINT16_C(0x4000), UINT16_C(0x4080), UINT16_C(0x4100), UINT16_C(0x4180),
    UINT16_C(0x4200), UINT16_C(0x4280), UINT16_C(0x4300), UINT16_C(0x4380),
    UINT16_C(0x4400), UINT16_C(0x4480), UINT16_C(0x4500), UINT16_C(0x4580),
    UINT16_C(0x4600), UINT16_C(0x4680), UINT16_C(0x4700), UINT16_C(0x4780),
    UINT16_C(0x4800), UINT16_C(0x4880), UINT16_C(0x4900), UINT16_C(0x4980),
    UINT16_C(0x4a00), UINT16_C(0x4a80), UINT16_C(0x4b00), UINT16_C(0x4b80),
    UINT16_C(0x4c00), UINT16_C(0x4c80), UINT16_C(0x4d00), UINT16_C(0x4d80),
    UINT16_C(0x4e00), UINT16_C(0x4e80), UINT16_C(0x4f00), UINT16_C(0x4f80),
    UINT16_C(0x5000), UINT16_C(0x5080), UINT16_C(0x5100), UINT16_C(0x5180),
    UINT16_C(0x5200), UINT16_C(0x5280), UINT16_C(0x5300), UINT16_C(0x5380),
    UINT16_C(0x5400), UINT16_C(0x5480), UINT16_C(0x5500), UINT16_C(0x5580),
    UINT16_C(0x5600), UINT16_C(0x5680), UINT16_C(0x5700), UINT16_C(0x5780),
    UINT16_C(0x5800), UINT16_C(0x5880), UINT16_C(0x5900), UINT16_C(0x5980),
    UINT16_C(0x5a00), UINT16_C(0x5a80), UINT16_C(0x5b00), UINT16_C(0x5b80),
    UINT16_C(0x5c00), UINT16_C(0x5c80), UINT16_C(0x5d00), UINT16_C(0x5d80),
    UINT16_C(0x5e00), UINT16_C(0x5e80), UINT16_C(0x5f00), UINT16_C(0x7e00),
    UINT16_C(0x8000), UINT16_C(0x9800), UINT16_C(0x9c00), UINT16_C(0x9e00),
    UINT16_C(0xa000), UINT16_C(0xa100), UINT16_C(0xa200), UINT16_C(0xa300),
    UINT16_C(0xa400), UINT16_C(0xa480), UINT16_C(0xa500), UINT16_C(0xa580),
    UINT16_C(0xa600), UINT16_C(0xa680), UINT16_C(0xa700), UINT16_C(0xa780),
    UINT16_C(0xa800), UINT16_C(0xa880), UINT16_C(0xa900), UINT16_C(0xa980),
    UINT16_C(0xaa00), UINT16_C(0xaa80), UINT16_C(0xab00), UINT16_C(0xab80),
    UINT16_C(0xac00), UINT16_C(0xac80), UINT16_C(0xad00), UINT16_C(0xad80),
    UINT16_C(0xae00), UINT16_C(0xae80), UINT16_C(0xaf00), UINT16_C(0xaf80),
    UINT16_C(0xb000), UINT16_C(0xb080), UINT16_C(0xb100), UINT16_C(0xb180),
    UINT16_C(0xb200), UINT16_C(0xb280), UINT16_C(0xb300), UINT16_C(0xb380),
    UINT16_C(0xb400), UINT16_C(0xb480), UINT16_C(0xb500), UINT16_C(0xb580),
    UINT16_C(0xb600), UINT16_C(0xb680), UINT16_C(0xb700), UINT16_C(0xb780),
    UINT16_C(0xb800), UINT16_C(0xb880), UINT16_C(0xb900), UINT16_C(0xb980),
    UINT16_C(0xba00), UINT16_C(0xba80), UINT16_C(0xbb00), UINT16_C(0xbb80),
    UINT16_C(0xbc00), UINT16_C(0xbc80), UINT16_C(0xbd00), UINT16_C(0xbd80),
    UINT16_C(0xbe00), UINT16_C(0xbe80), UINT16_C(0xbf00), UINT16_C(0xbf80),
    UINT16_C(0xc000), UINT16_C(0xc080), UINT16_C(0xc100), UINT16_C(0xc180),
    UINT16_C(0xc200), UINT16_C(0xc280), UINT16_C(0xc300), UINT16_C(0xc380),
    UINT16_C(0xc400), UINT16_C(0xc480), UINT16_C(0xc500), UINT16_C(0xc580),
    UINT16_C(0xc600), UINT16_C(0xc680), UINT16_C(0xc700), UINT16_C(0xc780),
    UINT16_C(0xc800), UINT16_C(0xc880), UINT16_C(0xc900), UINT16_C(0xc980),
    UINT16_C(0xca00), UINT16_C(0xca80), UINT16_C(0xcb00), UINT16_C(0xcb80),
    UINT16_C(0xcc00), UINT16_C(0xcc80), UINT16_C(0xcd00), UINT16_C(0xcd80),
    UINT16_C(0xce00), UINT16_C(0xce80), UINT16_C(0xcf00), UINT16_C(0xcf80),
    UINT16_C(0xd000), UINT16_C(0xd080), UINT16_C(0xd100), UINT16_C(0xd180),
    UINT16_C(0xd200), UINT16_C(0xd280), UINT16_C(0xd300), UINT16_C(0xd380),
    UINT16_C(0xd400), UINT16_C(0xd480), UINT16_C(0xd500), UINT16_C(0xd580),
    UINT16_C(0xd600), UINT16_C(0xd680), UINT16_C(0xd700), UINT16_C(0xd780),
    UINT16_C(0xd800), UINT16_C(0xd880), UINT16_C(0xd900), UINT16_C(0xd980),
    UINT16_C(0xda00), UINT16_C(0xda80), UINT16_C(0xdb00), UINT16_C(0xdb80),
    UINT16_C(0xdc00), UINT16_C(0xdc80), UINT16_C(0xdd00), UINT16_C(0xdd80),
    UINT16_C(0xde00), UINT16_C(0xde80), UINT16_C(0xdf00), UINT16_C(0xfe00),
};

struct ScaledPacks final {
  uint32_t even;
  uint32_t odd;
};

// This is the exact E4M3FN-to-FP16 mapping used to populate the probe's
// constant LUT.  Every finite E4M3FN value is exactly representable by FP16;
// 0x7f/0xff retain their signed quiet-NaN encodings.
__device__ __forceinline__ uint16_t
sllm_id84_nvfp4_e4m3fn_to_fp16_bits(const uint8_t bits) {
  const uint16_t sign =
      static_cast<uint16_t>(static_cast<uint32_t>(bits & UINT8_C(0x80)) << 8U);
  const uint8_t magnitude = bits & UINT8_C(0x7f);
  const uint8_t exponent = magnitude >> 3U;
  const uint8_t mantissa = magnitude & UINT8_C(0x07);
  if (exponent == 0U) {
    const uint16_t subnormal = mantissa == 0U   ? UINT16_C(0x0000)
                               : mantissa == 1U ? UINT16_C(0x1800)
                               : mantissa == 2U ? UINT16_C(0x1c00)
                               : mantissa == 3U ? UINT16_C(0x1e00)
                               : mantissa == 4U ? UINT16_C(0x2000)
                               : mantissa == 5U ? UINT16_C(0x2100)
                               : mantissa == 6U ? UINT16_C(0x2200)
                                                : UINT16_C(0x2300);
    return static_cast<uint16_t>(sign | subnormal);
  }
  if (magnitude == UINT8_C(0x7f)) {
    return static_cast<uint16_t>(sign | UINT16_C(0x7e00));
  }
  return static_cast<uint16_t>(
      sign | static_cast<uint16_t>(exponent + UINT8_C(8)) << 10U |
      static_cast<uint16_t>(mantissa) << 7U);
}

__device__ __forceinline__ float
sllm_id84_nvfp4_fp16_bits_to_float(const uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & UINT16_C(0x8000)) << 16U;
  const uint32_t exponent = (bits >> 10U) & UINT32_C(0x1f);
  const uint32_t mantissa = bits & UINT16_C(0x03ff);
  if (exponent == 0U) {
    if (mantissa == 0U)
      return __uint_as_float(sign);
    uint32_t normalized = mantissa;
    int32_t exponent_value = -14;
    while ((normalized & UINT32_C(0x0400)) == 0U) {
      normalized <<= 1U;
      --exponent_value;
    }
    normalized &= UINT32_C(0x03ff);
    return __uint_as_float(
        sign | (static_cast<uint32_t>(exponent_value + 127) << 23U) |
        (normalized << 13U));
  }
  if (exponent == UINT32_C(0x1f)) {
    return __uint_as_float(sign | UINT32_C(0x7f800000) | (mantissa << 13U));
  }
  return __uint_as_float(sign | ((exponent + 112U) << 23U) | (mantissa << 13U));
}

__device__ __forceinline__ void
sllm_id84_nvfp4_populate_constant_lut(float *const scale_lut) {
  if (threadIdx.x < kScaleLutEntries) {
    const uint32_t code = threadIdx.x;
    scale_lut[code + code / (kScaleLutStride - 1U)] =
        sllm_id84_nvfp4_fp16_bits_to_float(
            sllm_id84_nvfp4_scale_fp16_lut[code]);
  }
  __syncthreads();
}

__device__ __forceinline__ float
sllm_id84_nvfp4_load_scale(const uint8_t code, const float *const lds_fp32) {
  const uint32_t index = static_cast<uint32_t>(code) +
                         static_cast<uint32_t>(code) / (kScaleLutStride - 1U);
  return lds_fp32[index];
}

__device__ __forceinline__ ScaledPacks
sllm_id84_nvfp4_scaled_packs(const uint32_t packed) {
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

__device__ __forceinline__ int32_t sllm_id84_nvfp4_dot4(
    const uint32_t lhs, const uint32_t rhs, const int32_t accumulator) {
#if defined(__gfx1201__) && __has_builtin(__builtin_amdgcn_sudot4)
  // The packed operands are signed bytes.  On gfx12, sudot4 with both sign
  // selectors set to true is the exact signed/signed replacement for sdot4;
  // the NVFP4 block subtotal is bounded by 2304, so the int32 result cannot
  // overflow or saturate.
  return __builtin_amdgcn_sudot4(true, static_cast<int32_t>(lhs), true,
                                 static_cast<int32_t>(rhs), accumulator, false);
#elif __has_builtin(__builtin_amdgcn_sdot4)
  return __builtin_amdgcn_sdot4(static_cast<int32_t>(lhs),
                                static_cast<int32_t>(rhs), accumulator, false);
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

__device__ __forceinline__ uint16_t
sllm_id84_nvfp4_bf16_rne(const float value) {
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

__device__ __forceinline__ bool sllm_id84_nvfp4_id67_shape(const uint64_t m,
                                                           const uint64_t k,
                                                           const uint64_t n) {
  return m == 1U && k != 0U && (k % UINT64_C(16)) == 0U && k <= kDecodeMaxK &&
         n != 0U;
}

__device__ __forceinline__ bool sllm_id84_nvfp4_id73_shape(const uint64_t m,
                                                           const uint64_t k,
                                                           const uint64_t n) {
  return m == 1U && ((k == UINT64_C(5120) && n == UINT64_C(17408)) ||
                     (k == UINT64_C(17408) && n == UINT64_C(5120)));
}

// The gfx1201 path keeps packed NVFP4 resident and stages only four raw
// block iterations in registers.  This preserves the resident format while
// moving the next weight/scale loads ahead of the corresponding dot products.
struct sllm_id84_nvfp4_prefetch4_raw_block final {
  uint32_t activation_word0;
  uint32_t activation_word1;
  uint8_t activation_scale;
  uint32_t weight_word0[kColumnsPerWave];
  uint32_t weight_word1[kColumnsPerWave];
  uint8_t weight_scale[kColumnsPerWave];
};

__device__ __forceinline__ void sllm_id84_nvfp4_prefetch4_load(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales, const uint64_t blocks_per_row,
    const uint64_t column_base, const uint64_t n, const uint64_t block,
    sllm_id84_nvfp4_prefetch4_raw_block *const result) {
  *result = {};
  if (block >= blocks_per_row)
    return;
  const uint64_t packed_offset = block * UINT64_C(8);
  const auto *const activation_words =
      reinterpret_cast<const uint32_t *>(packed_activation + packed_offset);
  result->activation_word0 = __builtin_nontemporal_load(activation_words + 0U);
  result->activation_word1 = __builtin_nontemporal_load(activation_words + 1U);
  result->activation_scale =
      __builtin_nontemporal_load(activation_block_scales + block);
#pragma unroll
  for (uint32_t column_offset = 0U; column_offset < kColumnsPerWave;
       ++column_offset) {
    const uint64_t column = column_base + column_offset;
    if (column >= n)
      continue;
    const uint64_t weight_offset =
        column * (blocks_per_row * UINT64_C(8)) + packed_offset;
    const auto *const weight_words =
        reinterpret_cast<const uint32_t *>(packed_weight + weight_offset);
    result->weight_word0[column_offset] =
        __builtin_nontemporal_load(weight_words + 0U);
    result->weight_word1[column_offset] =
        __builtin_nontemporal_load(weight_words + 1U);
    result->weight_scale[column_offset] = __builtin_nontemporal_load(
        weight_block_scales + column * blocks_per_row + block);
  }
}

__device__ __forceinline__ void
sllm_id84_nvfp4_id67_body(const uint8_t *const packed_activation,
                          const uint8_t *const activation_block_scales,
                          const uint8_t *const packed_weight,
                          const uint8_t *const weight_block_scales,
                          const float *const weight_tensor_scale,
                          const float *const input_tensor_scale,
                          uint16_t *const output, const uint64_t m,
                          const uint64_t k, const uint64_t n,
                          const float *const scale_lut) {
  if (!sllm_id84_nvfp4_id67_shape(m, k, n))
    return;
  const uint32_t lane = threadIdx.x & (kWave - 1U);
  const uint32_t wave = threadIdx.x / kWave;
  const uint64_t blocks_per_row = k / UINT64_C(16);
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * kColumnsPerWorkgroup +
      static_cast<uint64_t>(wave) * kColumnsPerWave;
  float accumulators[kColumnsPerWave] = {};

  for (uint64_t group = lane; group < blocks_per_row; group += kWave * 4U) {
    sllm_id84_nvfp4_prefetch4_raw_block prefetched[4];
#pragma unroll
    for (uint32_t lookahead = 0U; lookahead < 4U; ++lookahead) {
      sllm_id84_nvfp4_prefetch4_load(
          packed_activation, activation_block_scales, packed_weight,
          weight_block_scales, blocks_per_row, column_base, n,
          group + static_cast<uint64_t>(lookahead) * kWave,
          &prefetched[lookahead]);
    }
#pragma unroll
    for (uint32_t lookahead = 0U; lookahead < 4U; ++lookahead) {
      const uint64_t block = group + static_cast<uint64_t>(lookahead) * kWave;
      if (block >= blocks_per_row)
        continue;
      const ScaledPacks activation_pack0 =
          sllm_id84_nvfp4_scaled_packs(prefetched[lookahead].activation_word0);
      const ScaledPacks activation_pack1 =
          sllm_id84_nvfp4_scaled_packs(prefetched[lookahead].activation_word1);
      const float activation_scale = sllm_id84_nvfp4_load_scale(
          prefetched[lookahead].activation_scale, scale_lut);
#pragma unroll
      for (uint32_t column_offset = 0U; column_offset < kColumnsPerWave;
           ++column_offset) {
        const uint64_t column = column_base + column_offset;
        if (column >= n)
          continue;
        const ScaledPacks weight_pack0 = sllm_id84_nvfp4_scaled_packs(
            prefetched[lookahead].weight_word0[column_offset]);
        const ScaledPacks weight_pack1 = sllm_id84_nvfp4_scaled_packs(
            prefetched[lookahead].weight_word1[column_offset]);
        int32_t block_sum = 0;
        block_sum = sllm_id84_nvfp4_dot4(activation_pack0.even,
                                         weight_pack0.even, block_sum);
        block_sum = sllm_id84_nvfp4_dot4(activation_pack0.odd, weight_pack0.odd,
                                         block_sum);
        block_sum = sllm_id84_nvfp4_dot4(activation_pack1.even,
                                         weight_pack1.even, block_sum);
        block_sum = sllm_id84_nvfp4_dot4(activation_pack1.odd, weight_pack1.odd,
                                         block_sum);
        const float weight_scale = sllm_id84_nvfp4_load_scale(
            prefetched[lookahead].weight_scale[column_offset], scale_lut);
        // Finite E4M3 scales have four significand bits; with |block_sum| <=
        // 2304, this product needs at most 20 FP32 significand bits (range
        // 2^-20..1.156e8), so its multiplications are exact.  Explicit fmaf
        // therefore has the same RN result as the old product-then-add,
        // including signed zero, while making the intended final rounding
        // explicit when -ffp-contract=off is used.
        accumulators[column_offset] =
            fmaf((static_cast<float>(block_sum) * 0.25F) * activation_scale,
                 weight_scale, accumulators[column_offset]);
      }
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
      output[column] = sllm_id84_nvfp4_bf16_rne(accumulators[column_offset] *
                                                weight_tensor_scale[0] *
                                                input_tensor_scale[0]);
    }
  }
}

// R9700 ID84 tuple candidate.  Keep ID67's P4 weight prefetch and the exact
// four-dot/fmaf/reduction order, but decode each activation block once per
// workgroup into dynamic LDS.  The launcher admits this body only for the two
// measured Qwen3.8 projection tuples; the broad ID84 wrapper remains intact.
__device__ __forceinline__ void sllm_id84_nvfp4_id67_activation_shared_body(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n,
    const float *const scale_lut, uint32_t *const dynamic_shared) {
  if (!sllm_id84_nvfp4_id67_shape(m, k, n))
    return;
  const uint32_t lane = threadIdx.x & (kWave - 1U);
  const uint32_t wave = threadIdx.x / kWave;
  const uint64_t blocks_per_row = k / UINT64_C(16);
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * kColumnsPerWorkgroup +
      static_cast<uint64_t>(wave) * kColumnsPerWave;
  int32_t *const activation_packs = reinterpret_cast<int32_t *>(dynamic_shared);
  float *const activation_scale_values =
      reinterpret_cast<float *>(dynamic_shared + blocks_per_row * UINT64_C(4));

  for (uint64_t block = threadIdx.x; block < blocks_per_row;
       block += kThreads) {
    const auto *const activation_words = reinterpret_cast<const uint32_t *>(
        packed_activation + block * UINT64_C(8));
    const ScaledPacks first = sllm_id84_nvfp4_scaled_packs(
        __builtin_nontemporal_load(activation_words + 0U));
    const ScaledPacks second = sllm_id84_nvfp4_scaled_packs(
        __builtin_nontemporal_load(activation_words + 1U));
    activation_packs[block * UINT64_C(4) + UINT64_C(0)] =
        static_cast<int32_t>(first.even);
    activation_packs[block * UINT64_C(4) + UINT64_C(1)] =
        static_cast<int32_t>(first.odd);
    activation_packs[block * UINT64_C(4) + UINT64_C(2)] =
        static_cast<int32_t>(second.even);
    activation_packs[block * UINT64_C(4) + UINT64_C(3)] =
        static_cast<int32_t>(second.odd);
    activation_scale_values[block] = sllm_id84_nvfp4_load_scale(
        __builtin_nontemporal_load(activation_block_scales + block), scale_lut);
  }
  __syncthreads();

  float accumulators[kColumnsPerWave] = {};
  for (uint64_t group = lane; group < blocks_per_row; group += kWave * 4U) {
    sllm_id84_nvfp4_prefetch4_raw_block prefetched[4];
#pragma unroll
    for (uint32_t lookahead = 0U; lookahead < 4U; ++lookahead) {
      sllm_id84_nvfp4_prefetch4_load(
          packed_activation, activation_block_scales, packed_weight,
          weight_block_scales, blocks_per_row, column_base, n,
          group + static_cast<uint64_t>(lookahead) * kWave,
          &prefetched[lookahead]);
    }
#pragma unroll
    for (uint32_t lookahead = 0U; lookahead < 4U; ++lookahead) {
      const uint64_t block = group + static_cast<uint64_t>(lookahead) * kWave;
      if (block >= blocks_per_row)
        continue;
      const ScaledPacks activation_pack0 = {
          static_cast<uint32_t>(activation_packs[block * UINT64_C(4) + 0U]),
          static_cast<uint32_t>(activation_packs[block * UINT64_C(4) + 1U])};
      const ScaledPacks activation_pack1 = {
          static_cast<uint32_t>(activation_packs[block * UINT64_C(4) + 2U]),
          static_cast<uint32_t>(activation_packs[block * UINT64_C(4) + 3U])};
      const float activation_scale = activation_scale_values[block];
#pragma unroll
      for (uint32_t column_offset = 0U; column_offset < kColumnsPerWave;
           ++column_offset) {
        const uint64_t column = column_base + column_offset;
        if (column >= n)
          continue;
        const ScaledPacks weight_pack0 = sllm_id84_nvfp4_scaled_packs(
            prefetched[lookahead].weight_word0[column_offset]);
        const ScaledPacks weight_pack1 = sllm_id84_nvfp4_scaled_packs(
            prefetched[lookahead].weight_word1[column_offset]);
        int32_t block_sum = 0;
        block_sum = sllm_id84_nvfp4_dot4(activation_pack0.even,
                                         weight_pack0.even, block_sum);
        block_sum = sllm_id84_nvfp4_dot4(activation_pack0.odd, weight_pack0.odd,
                                         block_sum);
        block_sum = sllm_id84_nvfp4_dot4(activation_pack1.even,
                                         weight_pack1.even, block_sum);
        block_sum = sllm_id84_nvfp4_dot4(activation_pack1.odd, weight_pack1.odd,
                                         block_sum);
        const float weight_scale = sllm_id84_nvfp4_load_scale(
            prefetched[lookahead].weight_scale[column_offset], scale_lut);
        accumulators[column_offset] =
            fmaf((static_cast<float>(block_sum) * 0.25F) * activation_scale,
                 weight_scale, accumulators[column_offset]);
      }
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
      output[column] = sllm_id84_nvfp4_bf16_rne(accumulators[column_offset] *
                                                weight_tensor_scale[0] *
                                                input_tensor_scale[0]);
    }
  }
}

// Keep the prefetch-free ID73 body available for focused comparisons.  The
// production gfx1030 ID84 wrapper below uses the P=2 body; gfx1201 ID67 and
// its existing P=4 body remain unchanged.
__device__ __forceinline__ void
sllm_id84_nvfp4_id73_body_baseline(const uint8_t *const packed_activation,
                                   const uint8_t *const activation_block_scales,
                                   const uint8_t *const packed_weight,
                                   const uint8_t *const weight_block_scales,
                                   const float *const weight_tensor_scale,
                                   const float *const input_tensor_scale,
                                   uint16_t *const output, const uint64_t m,
                                   const uint64_t k, const uint64_t n,
                                   const float *const scale_lut) {
  if (!sllm_id84_nvfp4_id73_shape(m, k, n))
    return;
  const uint32_t lane = threadIdx.x & (kWave - 1U);
  const uint32_t wave = threadIdx.x / kWave;
  const uint64_t blocks_per_row = k / UINT64_C(16);
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * kColumnsPerWorkgroup +
      static_cast<uint64_t>(wave) * kColumnsPerWave;
  const uint64_t packed_row_bytes = k / UINT64_C(2);

  // Four int8x4 packs (16 bytes) plus one FP32 decoded scale (4 bytes) are
  // shared per block16: dynamic activation LDS is exactly K*5/4 bytes.
  extern __shared__ uint32_t shared[];
  int32_t *const activation_packs = reinterpret_cast<int32_t *>(shared);
  float *const activation_scale_values =
      reinterpret_cast<float *>(shared + blocks_per_row * UINT64_C(4));
  for (uint64_t block = threadIdx.x; block < blocks_per_row;
       block += kThreads) {
    const auto *const activation_words = reinterpret_cast<const uint32_t *>(
        packed_activation + block * UINT64_C(8));
    const ScaledPacks first = sllm_id84_nvfp4_scaled_packs(
        __builtin_nontemporal_load(activation_words + 0U));
    const ScaledPacks second = sllm_id84_nvfp4_scaled_packs(
        __builtin_nontemporal_load(activation_words + 1U));
    activation_packs[block * UINT64_C(4) + UINT64_C(0)] =
        static_cast<int32_t>(first.even);
    activation_packs[block * UINT64_C(4) + UINT64_C(1)] =
        static_cast<int32_t>(first.odd);
    activation_packs[block * UINT64_C(4) + UINT64_C(2)] =
        static_cast<int32_t>(second.even);
    activation_packs[block * UINT64_C(4) + UINT64_C(3)] =
        static_cast<int32_t>(second.odd);
    activation_scale_values[block] = sllm_id84_nvfp4_load_scale(
        __builtin_nontemporal_load(activation_block_scales + block), scale_lut);
  }
  __syncthreads();

  float accumulators[kColumnsPerWave] = {};
  for (uint64_t block = lane; block < blocks_per_row; block += kWave) {
    const ScaledPacks activation_pack0 = {
        static_cast<uint32_t>(
            activation_packs[block * UINT64_C(4) + UINT64_C(0)]),
        static_cast<uint32_t>(
            activation_packs[block * UINT64_C(4) + UINT64_C(1)])};
    const ScaledPacks activation_pack1 = {
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
      const ScaledPacks weight_pack0 = sllm_id84_nvfp4_scaled_packs(
          __builtin_nontemporal_load(weight_words + 0U));
      const ScaledPacks weight_pack1 = sllm_id84_nvfp4_scaled_packs(
          __builtin_nontemporal_load(weight_words + 1U));
      int32_t block_sum = 0;
      block_sum = sllm_id84_nvfp4_dot4(activation_pack0.even, weight_pack0.even,
                                       block_sum);
      block_sum = sllm_id84_nvfp4_dot4(activation_pack0.odd, weight_pack0.odd,
                                       block_sum);
      block_sum = sllm_id84_nvfp4_dot4(activation_pack1.even, weight_pack1.even,
                                       block_sum);
      block_sum = sllm_id84_nvfp4_dot4(activation_pack1.odd, weight_pack1.odd,
                                       block_sum);
      const float weight_scale = sllm_id84_nvfp4_load_scale(
          __builtin_nontemporal_load(weight_block_scales +
                                     column * blocks_per_row + block),
          scale_lut);
      // The finite-domain product is exact in FP32 (the integer subtotal is
      // bounded by 2304 and the two four-bit-significand E4M3 scales require
      // at most 20 significand bits), so fmaf preserves the old RN
      // product-then-add result, including signed zero.
      accumulators[column_offset] =
          fmaf((static_cast<float>(block_sum) * 0.25F) * activation_scale,
               weight_scale, accumulators[column_offset]);
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
      output[column] = sllm_id84_nvfp4_bf16_rne(accumulators[column_offset] *
                                                weight_tensor_scale[0] *
                                                input_tensor_scale[0]);
    }
  }
}

// gfx1030/V620 ID73 candidate.  Activation packs and activation scales remain
// in the production dynamic LDS layout.  Only raw weight words and weight
// scales are loaded in a two-iteration lookahead window; each lane processes
// the same ascending block sequence and retains the original FP32/FMA order.
struct sllm_id84_nvfp4_prefetch2_weight_block final {
  uint32_t weight_word0[kColumnsPerWave];
  uint32_t weight_word1[kColumnsPerWave];
  uint8_t weight_scale[kColumnsPerWave];
};

__device__ __forceinline__ void sllm_id84_nvfp4_prefetch2_weight_load(
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales, const uint64_t blocks_per_row,
    const uint64_t column_base, const uint64_t n, const uint64_t block,
    sllm_id84_nvfp4_prefetch2_weight_block *const result) {
  *result = {};
  if (block >= blocks_per_row)
    return;
  const uint64_t packed_offset = block * UINT64_C(8);
#pragma unroll
  for (uint32_t column_offset = 0U; column_offset < kColumnsPerWave;
       ++column_offset) {
    const uint64_t column = column_base + column_offset;
    if (column >= n)
      continue;
    const uint64_t weight_offset =
        column * (blocks_per_row * UINT64_C(8)) + packed_offset;
    const auto *const weight_words =
        reinterpret_cast<const uint32_t *>(packed_weight + weight_offset);
    result->weight_word0[column_offset] =
        __builtin_nontemporal_load(weight_words + 0U);
    result->weight_word1[column_offset] =
        __builtin_nontemporal_load(weight_words + 1U);
    result->weight_scale[column_offset] = __builtin_nontemporal_load(
        weight_block_scales + column * blocks_per_row + block);
  }
}

__device__ __forceinline__ void
sllm_id84_nvfp4_id73_body(const uint8_t *const packed_activation,
                          const uint8_t *const activation_block_scales,
                          const uint8_t *const packed_weight,
                          const uint8_t *const weight_block_scales,
                          const float *const weight_tensor_scale,
                          const float *const input_tensor_scale,
                          uint16_t *const output, const uint64_t m,
                          const uint64_t k, const uint64_t n,
                          const float *const scale_lut) {
  if (!sllm_id84_nvfp4_id73_shape(m, k, n))
    return;
  const uint32_t lane = threadIdx.x & (kWave - 1U);
  const uint32_t wave = threadIdx.x / kWave;
  const uint64_t blocks_per_row = k / UINT64_C(16);
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * kColumnsPerWorkgroup +
      static_cast<uint64_t>(wave) * kColumnsPerWave;

  extern __shared__ uint32_t shared[];
  int32_t *const activation_packs = reinterpret_cast<int32_t *>(shared);
  float *const activation_scale_values =
      reinterpret_cast<float *>(shared + blocks_per_row * UINT64_C(4));
  for (uint64_t block = threadIdx.x; block < blocks_per_row;
       block += kThreads) {
    const auto *const activation_words = reinterpret_cast<const uint32_t *>(
        packed_activation + block * UINT64_C(8));
    const ScaledPacks first = sllm_id84_nvfp4_scaled_packs(
        __builtin_nontemporal_load(activation_words + 0U));
    const ScaledPacks second = sllm_id84_nvfp4_scaled_packs(
        __builtin_nontemporal_load(activation_words + 1U));
    activation_packs[block * UINT64_C(4) + UINT64_C(0)] =
        static_cast<int32_t>(first.even);
    activation_packs[block * UINT64_C(4) + UINT64_C(1)] =
        static_cast<int32_t>(first.odd);
    activation_packs[block * UINT64_C(4) + UINT64_C(2)] =
        static_cast<int32_t>(second.even);
    activation_packs[block * UINT64_C(4) + UINT64_C(3)] =
        static_cast<int32_t>(second.odd);
    activation_scale_values[block] = sllm_id84_nvfp4_load_scale(
        __builtin_nontemporal_load(activation_block_scales + block), scale_lut);
  }
  __syncthreads();

  float accumulators[kColumnsPerWave] = {};
  for (uint64_t group = lane; group < blocks_per_row; group += kWave * 2U) {
    sllm_id84_nvfp4_prefetch2_weight_block prefetched[2];
#pragma unroll
    for (uint32_t lookahead = 0U; lookahead < 2U; ++lookahead) {
      sllm_id84_nvfp4_prefetch2_weight_load(
          packed_weight, weight_block_scales, blocks_per_row, column_base, n,
          group + static_cast<uint64_t>(lookahead) * kWave,
          &prefetched[lookahead]);
    }
#pragma unroll
    for (uint32_t lookahead = 0U; lookahead < 2U; ++lookahead) {
      const uint64_t block = group + static_cast<uint64_t>(lookahead) * kWave;
      if (block >= blocks_per_row)
        continue;
      const ScaledPacks activation_pack0 = {
          static_cast<uint32_t>(activation_packs[block * UINT64_C(4) + 0U]),
          static_cast<uint32_t>(activation_packs[block * UINT64_C(4) + 1U])};
      const ScaledPacks activation_pack1 = {
          static_cast<uint32_t>(activation_packs[block * UINT64_C(4) + 2U]),
          static_cast<uint32_t>(activation_packs[block * UINT64_C(4) + 3U])};
      const float activation_scale = activation_scale_values[block];
#pragma unroll
      for (uint32_t column_offset = 0U; column_offset < kColumnsPerWave;
           ++column_offset) {
        const uint64_t column = column_base + column_offset;
        if (column >= n)
          continue;
        const ScaledPacks weight_pack0 = sllm_id84_nvfp4_scaled_packs(
            prefetched[lookahead].weight_word0[column_offset]);
        const ScaledPacks weight_pack1 = sllm_id84_nvfp4_scaled_packs(
            prefetched[lookahead].weight_word1[column_offset]);
        int32_t block_sum = 0;
        block_sum = sllm_id84_nvfp4_dot4(activation_pack0.even,
                                         weight_pack0.even, block_sum);
        block_sum = sllm_id84_nvfp4_dot4(activation_pack0.odd, weight_pack0.odd,
                                         block_sum);
        block_sum = sllm_id84_nvfp4_dot4(activation_pack1.even,
                                         weight_pack1.even, block_sum);
        block_sum = sllm_id84_nvfp4_dot4(activation_pack1.odd, weight_pack1.odd,
                                         block_sum);
        const float weight_scale = sllm_id84_nvfp4_load_scale(
            prefetched[lookahead].weight_scale[column_offset], scale_lut);
        accumulators[column_offset] =
            fmaf((static_cast<float>(block_sum) * 0.25F) * activation_scale,
                 weight_scale, accumulators[column_offset]);
      }
    }
  }

#pragma unroll
  for (uint32_t column_offset = 0U; column_offset < kColumnsPerWave;
       ++column_offset) {
#pragma unroll
    for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U)
      accumulators[column_offset] +=
          __shfl_down(accumulators[column_offset], offset, kWave);
    const uint64_t column = column_base + column_offset;
    if (lane == 0U && column < n) {
      output[column] = sllm_id84_nvfp4_bf16_rne(accumulators[column_offset] *
                                                weight_tensor_scale[0] *
                                                input_tensor_scale[0]);
    }
  }
}

} // namespace sllm_id84_nvfp4_scale_lut_detail

// R9700 (gfx1201) normal decode path.  The selector/launcher owns the exact
// target check; this kernel itself remains target-neutral for code-object
// compilation and is shape guarded like the ID67 production path.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_decode_dp4a_wave4col32_lds_f32_lut_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  __shared__ float scale_lut[sllm_id84_nvfp4_scale_lut_detail::kScaleLutSlots];
  if (threadIdx.x < sllm_id84_nvfp4_scale_lut_detail::kScaleLutEntries) {
    const uint32_t code = threadIdx.x;
    scale_lut[code +
              code / (sllm_id84_nvfp4_scale_lut_detail::kScaleLutStride - 1U)] =
        sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_fp16_bits_to_float(
            sllm_id84_nvfp4_scale_lut_detail::
                sllm_id84_nvfp4_e4m3fn_to_fp16_bits(
                    static_cast<uint8_t>(code)));
  }
  __syncthreads();
  sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_id67_body(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n, scale_lut);
}

// R9700 ID84 exact-tuple body.  The static LUT keeps the existing 1056-byte
// FP32 scale-LUT resource; dynamic LDS is reserved by the host launcher for
// decoded activation packs and FP32 activation scales.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_nvfp4_w4a4_decode_scale_lut_gfx1201_actshared_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
#if defined(__gfx1201__)
  __shared__ float scale_lut[sllm_id84_nvfp4_scale_lut_detail::kScaleLutSlots];
  extern __shared__ uint32_t dynamic_shared[];
  if (threadIdx.x < sllm_id84_nvfp4_scale_lut_detail::kScaleLutEntries) {
    const uint32_t code = threadIdx.x;
    scale_lut[code +
              code / (sllm_id84_nvfp4_scale_lut_detail::kScaleLutStride - 1U)] =
        sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_fp16_bits_to_float(
            sllm_id84_nvfp4_scale_lut_detail::
                sllm_id84_nvfp4_e4m3fn_to_fp16_bits(
                    static_cast<uint8_t>(code)));
  }
  __syncthreads();
  sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_id67_activation_shared_body(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n, scale_lut, dynamic_shared);
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

// V620 (gfx1030) activation-shared decode path.  Static LDS holds the
// bank-padded FP32 LUT and dynamic LDS holds the activation packs/scales.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_decode_dp4a_activation_shared_lds_f32_lut_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  __shared__ float scale_lut[sllm_id84_nvfp4_scale_lut_detail::kScaleLutSlots];
  if (threadIdx.x < sllm_id84_nvfp4_scale_lut_detail::kScaleLutEntries) {
    const uint32_t code = threadIdx.x;
    scale_lut[code +
              code / (sllm_id84_nvfp4_scale_lut_detail::kScaleLutStride - 1U)] =
        sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_fp16_bits_to_float(
            sllm_id84_nvfp4_scale_lut_detail::
                sllm_id84_nvfp4_e4m3fn_to_fp16_bits(
                    static_cast<uint8_t>(code)));
  }
  __syncthreads();
  sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_id73_body(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n, scale_lut);
}

// Focused comparison variants that use the old probe's device constant
// FP16 table as the source of each FP32 LDS entry.  They intentionally keep
// separate symbols so a later integration probe can compare this path with
// the arithmetic table population above without changing the production
// wrapper or requiring host-side constant-memory initialization.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_decode_dp4a_wave4col32_lds_f32_const_lut_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  __shared__ float scale_lut[sllm_id84_nvfp4_scale_lut_detail::kScaleLutSlots];
  sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_populate_constant_lut(
      scale_lut);
  sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_id67_body(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n, scale_lut);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_decode_dp4a_activation_shared_lds_f32_const_lut_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  __shared__ float scale_lut[sllm_id84_nvfp4_scale_lut_detail::kScaleLutSlots];
  sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_populate_constant_lut(
      scale_lut);
  sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_id73_body(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n, scale_lut);
}
