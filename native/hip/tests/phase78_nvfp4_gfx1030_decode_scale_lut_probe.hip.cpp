// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase78-nvfp4-byte-permute-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// 3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70, ggml/src/ggml-cuda/vecdotq.cuh
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

// Phase 78 standalone gfx1030 NVFP4 decode scale-LUT probe.
//
// This file deliberately has no production linkage.  ID67's wave4/col32
// kernel and ID73's activation-shared variant are kept bit-for-bit in their
// E2M1 unpack, DP4A order, block16 subtotal, and BF16 epilogue.  Only the
// E4M3FN block-scale conversion is changed: direct decode, a read-only FP16
// LUT, a bank-padded LDS FP16 LUT, and a bank-padded LDS FP32 LUT.

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
constexpr uint32_t kColumnsPerWave = 4U;
constexpr uint32_t kColumnsPerWorkgroup = kWaves * kColumnsPerWave;
constexpr uint32_t kColdCopies = 4U;
constexpr int kWarmups = 3;
constexpr int kMeasured = 10;
constexpr uint32_t kScaleLutEntries = 256U;
// One padding element after every 32 entries prevents a wave's same-index
// access from becoming a bank-conflict pattern on gfx1030 LDS.
constexpr uint32_t kScaleLutStride = 33U;
constexpr uint32_t kScaleLutSlots = kScaleLutEntries + kScaleLutEntries / 32U;

struct Shape final {
  uint64_t k;
  uint64_t n;
  uint32_t occurrences;
  const char *role;
  bool id73;
};

constexpr std::array<Shape, 8> kQwenShapes = {{
    {5120U, 17408U, 16U, "layers56-63.mlp.gate+up", true},
    {17408U, 5120U, 8U, "layers56-63.mlp.down", true},
    {5120U, 12288U, 16U, "16.full-attn.q", false},
    {5120U, 1024U, 32U, "16.full-attn.k+v", false},
    {6144U, 5120U, 64U, "full-attn.o+linear-attn.out", false},
    {5120U, 10240U, 48U, "48.linear-attn.qkv", false},
    {5120U, 6144U, 48U, "48.linear-attn.z", false},
    {5120U, 248320U, 1U, "lm_head", false},
}};

constexpr uint32_t shape_occurrences() {
  uint32_t result = 0U;
  for (const Shape &shape : kQwenShapes)
    result += shape.occurrences;
  return result;
}
static_assert(shape_occurrences() == 233U, "Qwen occurrence recipe changed");

constexpr std::array<uint8_t, 16> kFiniteScaleCodes = {
    0x00U, 0x01U, 0x08U, 0x10U, 0x18U, 0x20U, 0x28U, 0x30U,
    0x80U, 0x81U, 0x88U, 0x90U, 0x98U, 0xa0U, 0xa8U, 0xb0U,
};

__device__ __constant__ uint16_t kScaleFp16Lut[kScaleLutEntries];

struct ScaledPacks final {
  uint32_t even;
  uint32_t odd;
};

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

// All finite E4M3FN values are exactly representable by FP16.  This helper
// intentionally accepts the full half format so the 256-code oracle also
// covers signed zero, subnormal, and NaN entries.
__device__ __forceinline__ float fp16_bits_to_float(const uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16U;
  const uint32_t exponent = (bits >> 10U) & 0x1fU;
  const uint32_t mantissa = bits & 0x03ffU;
  if (exponent == 0U) {
    if (mantissa == 0U)
      return __uint_as_float(sign);
    uint32_t normalized = mantissa;
    int32_t exponent_value = -14;
    while ((normalized & 0x0400U) == 0U) {
      normalized <<= 1U;
      --exponent_value;
    }
    normalized &= 0x03ffU;
    return __uint_as_float(sign |
                           static_cast<uint32_t>(exponent_value + 127) << 23U |
                           normalized << 13U);
  }
  if (exponent == 0x1fU) {
    return __uint_as_float(sign | UINT32_C(0x7f800000) | (mantissa << 13U));
  }
  return __uint_as_float(sign | ((exponent + 112U) << 23U) | (mantissa << 13U));
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

enum class ScaleMode : uint32_t { Direct, ConstantFp16, LdsFp16, LdsF32 };

template <ScaleMode Mode>
__device__ __forceinline__ float load_scale(const uint8_t code,
                                            const uint16_t *const lds_fp16,
                                            const float *const lds_fp32) {
  if constexpr (Mode == ScaleMode::Direct) {
    return sllm_lowp::e4m3fn_to_float(code);
  } else if constexpr (Mode == ScaleMode::ConstantFp16) {
    return fp16_bits_to_float(kScaleFp16Lut[code]);
  } else if constexpr (Mode == ScaleMode::LdsFp16) {
    return fp16_bits_to_float(
        lds_fp16[static_cast<uint32_t>(code) +
                 static_cast<uint32_t>(code) / (kScaleLutStride - 1U)]);
  } else {
    return lds_fp32[static_cast<uint32_t>(code) +
                    static_cast<uint32_t>(code) / (kScaleLutStride - 1U)];
  }
}

template <ScaleMode Mode>
__device__ __forceinline__ void id67_body(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_scales, const uint8_t *const packed_weight,
    const uint8_t *const weight_scales, const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n,
    const uint16_t *const lds_fp16, const float *const lds_fp32) {
  if (m != 1U || k == 0U || n == 0U || (k % 16U) != 0U)
    return;
  const uint32_t lane = threadIdx.x & (kWave - 1U);
  const uint32_t wave = threadIdx.x / kWave;
  const uint64_t blocks_per_row = k / 16U;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * kColumnsPerWorkgroup +
      static_cast<uint64_t>(wave) * kColumnsPerWave;
  const uint64_t packed_row_bytes = k / 2U;
  float accumulators[kColumnsPerWave] = {};

  for (uint64_t block = lane; block < blocks_per_row; block += kWave) {
    const uint64_t packed_offset = block * 8U;
    const auto *const activation_words =
        reinterpret_cast<const uint32_t *>(packed_activation + packed_offset);
    const ScaledPacks activation_pack0 =
        scaled_packs(__builtin_nontemporal_load(activation_words + 0U));
    const ScaledPacks activation_pack1 =
        scaled_packs(__builtin_nontemporal_load(activation_words + 1U));
    const float activation_scale =
        load_scale<Mode>(__builtin_nontemporal_load(activation_scales + block),
                         lds_fp16, lds_fp32);
#pragma unroll
    for (uint32_t column_offset = 0U; column_offset < kColumnsPerWave;
         ++column_offset) {
      const uint64_t column = column_base + column_offset;
      if (column >= n)
        continue;
      const auto *const weight_words = reinterpret_cast<const uint32_t *>(
          packed_weight + column * packed_row_bytes + packed_offset);
      const ScaledPacks weight_pack0 =
          scaled_packs(__builtin_nontemporal_load(weight_words + 0U));
      const ScaledPacks weight_pack1 =
          scaled_packs(__builtin_nontemporal_load(weight_words + 1U));
      int32_t block_sum = 0;
      block_sum = dot4(activation_pack0.even, weight_pack0.even, block_sum);
      block_sum = dot4(activation_pack0.odd, weight_pack0.odd, block_sum);
      block_sum = dot4(activation_pack1.even, weight_pack1.even, block_sum);
      block_sum = dot4(activation_pack1.odd, weight_pack1.odd, block_sum);
      const float weight_scale =
          load_scale<Mode>(__builtin_nontemporal_load(
                               weight_scales + column * blocks_per_row + block),
                           lds_fp16, lds_fp32);
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
      output[column] = bf16_rne(accumulators[column_offset] *
                                weight_tensor_scale[0] * input_tensor_scale[0]);
    }
  }
}

template <ScaleMode Mode>
__global__ __launch_bounds__(kThreads, 1) void id67_kernel(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_scales, const uint8_t *const packed_weight,
    const uint8_t *const weight_scales, const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  if constexpr (Mode == ScaleMode::LdsFp16) {
    __shared__ uint16_t lut[kScaleLutSlots];
    if (threadIdx.x < kScaleLutEntries) {
      const uint32_t code = threadIdx.x;
      lut[code + code / 32U] = kScaleFp16Lut[code];
    }
    __syncthreads();
    id67_body<Mode>(packed_activation, activation_scales, packed_weight,
                    weight_scales, weight_tensor_scale, input_tensor_scale,
                    output, m, k, n, lut, nullptr);
  } else if constexpr (Mode == ScaleMode::LdsF32) {
    __shared__ float lut[kScaleLutSlots];
    if (threadIdx.x < kScaleLutEntries) {
      const uint32_t code = threadIdx.x;
      lut[code + code / 32U] = fp16_bits_to_float(kScaleFp16Lut[code]);
    }
    __syncthreads();
    id67_body<Mode>(packed_activation, activation_scales, packed_weight,
                    weight_scales, weight_tensor_scale, input_tensor_scale,
                    output, m, k, n, nullptr, lut);
  } else {
    id67_body<Mode>(packed_activation, activation_scales, packed_weight,
                    weight_scales, weight_tensor_scale, input_tensor_scale,
                    output, m, k, n, nullptr, nullptr);
  }
}

template <ScaleMode Mode>
__device__ __forceinline__ void id73_body(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_scales, const uint8_t *const packed_weight,
    const uint8_t *const weight_scales, const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n,
    const uint16_t *const lds_fp16, const float *const lds_fp32) {
  if (m != 1U || k == 0U || n == 0U || (k % 16U) != 0U)
    return;
  const uint32_t lane = threadIdx.x & (kWave - 1U);
  const uint32_t wave = threadIdx.x / kWave;
  const uint64_t blocks_per_row = k / 16U;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * kColumnsPerWorkgroup +
      static_cast<uint64_t>(wave) * kColumnsPerWave;
  const uint64_t packed_row_bytes = k / 2U;
  extern __shared__ uint32_t shared[];
  int32_t *const activation_packs = reinterpret_cast<int32_t *>(shared);
  float *const activation_scale_values =
      reinterpret_cast<float *>(shared + blocks_per_row * 4U);
  for (uint64_t block = threadIdx.x; block < blocks_per_row;
       block += kThreads) {
    const auto *const words =
        reinterpret_cast<const uint32_t *>(packed_activation + block * 8U);
    const ScaledPacks first =
        scaled_packs(__builtin_nontemporal_load(words + 0U));
    const ScaledPacks second =
        scaled_packs(__builtin_nontemporal_load(words + 1U));
    activation_packs[block * 4U + 0U] = static_cast<int32_t>(first.even);
    activation_packs[block * 4U + 1U] = static_cast<int32_t>(first.odd);
    activation_packs[block * 4U + 2U] = static_cast<int32_t>(second.even);
    activation_packs[block * 4U + 3U] = static_cast<int32_t>(second.odd);
    activation_scale_values[block] =
        load_scale<Mode>(__builtin_nontemporal_load(activation_scales + block),
                         lds_fp16, lds_fp32);
  }
  __syncthreads();

  float accumulators[kColumnsPerWave] = {};
  for (uint64_t block = lane; block < blocks_per_row; block += kWave) {
    const ScaledPacks activation_pack0 = {
        static_cast<uint32_t>(activation_packs[block * 4U + 0U]),
        static_cast<uint32_t>(activation_packs[block * 4U + 1U])};
    const ScaledPacks activation_pack1 = {
        static_cast<uint32_t>(activation_packs[block * 4U + 2U]),
        static_cast<uint32_t>(activation_packs[block * 4U + 3U])};
    const float activation_scale = activation_scale_values[block];
#pragma unroll
    for (uint32_t column_offset = 0U; column_offset < kColumnsPerWave;
         ++column_offset) {
      const uint64_t column = column_base + column_offset;
      if (column >= n)
        continue;
      const auto *const words = reinterpret_cast<const uint32_t *>(
          packed_weight + column * packed_row_bytes + block * 8U);
      const ScaledPacks weight_pack0 =
          scaled_packs(__builtin_nontemporal_load(words + 0U));
      const ScaledPacks weight_pack1 =
          scaled_packs(__builtin_nontemporal_load(words + 1U));
      int32_t block_sum = 0;
      block_sum = dot4(activation_pack0.even, weight_pack0.even, block_sum);
      block_sum = dot4(activation_pack0.odd, weight_pack0.odd, block_sum);
      block_sum = dot4(activation_pack1.even, weight_pack1.even, block_sum);
      block_sum = dot4(activation_pack1.odd, weight_pack1.odd, block_sum);
      const float weight_scale =
          load_scale<Mode>(__builtin_nontemporal_load(
                               weight_scales + column * blocks_per_row + block),
                           lds_fp16, lds_fp32);
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
      output[column] = bf16_rne(accumulators[column_offset] *
                                weight_tensor_scale[0] * input_tensor_scale[0]);
    }
  }
}

template <ScaleMode Mode>
__global__ __launch_bounds__(kThreads, 1) void id73_kernel(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_scales, const uint8_t *const packed_weight,
    const uint8_t *const weight_scales, const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  if constexpr (Mode == ScaleMode::LdsFp16) {
    __shared__ uint16_t lut[kScaleLutSlots];
    if (threadIdx.x < kScaleLutEntries) {
      const uint32_t code = threadIdx.x;
      lut[code + code / 32U] = kScaleFp16Lut[code];
    }
    __syncthreads();
    id73_body<Mode>(packed_activation, activation_scales, packed_weight,
                    weight_scales, weight_tensor_scale, input_tensor_scale,
                    output, m, k, n, lut, nullptr);
  } else if constexpr (Mode == ScaleMode::LdsF32) {
    __shared__ float lut[kScaleLutSlots];
    if (threadIdx.x < kScaleLutEntries) {
      const uint32_t code = threadIdx.x;
      lut[code + code / 32U] = fp16_bits_to_float(kScaleFp16Lut[code]);
    }
    __syncthreads();
    id73_body<Mode>(packed_activation, activation_scales, packed_weight,
                    weight_scales, weight_tensor_scale, input_tensor_scale,
                    output, m, k, n, nullptr, lut);
  } else {
    id73_body<Mode>(packed_activation, activation_scales, packed_weight,
                    weight_scales, weight_tensor_scale, input_tensor_scale,
                    output, m, k, n, nullptr, nullptr);
  }
}

__global__ void scale_lut_oracle_kernel(uint16_t *const lut_bits,
                                        uint32_t *const direct_bits,
                                        uint32_t *const lut_float_bits) {
  const uint32_t code = blockIdx.x * blockDim.x + threadIdx.x;
  if (code >= kScaleLutEntries)
    return;
  lut_bits[code] = kScaleFp16Lut[code];
  direct_bits[code] =
      __float_as_uint(sllm_lowp::e4m3fn_to_float(static_cast<uint8_t>(code)));
  lut_float_bits[code] =
      __float_as_uint(fp16_bits_to_float(kScaleFp16Lut[code]));
}

__global__ void block16_subtotal_kernel(const uint8_t *const activation,
                                        const uint8_t *const weight,
                                        int32_t *const output) {
  if (blockIdx.x != 0U || threadIdx.x != 0U)
    return;
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

enum class CandidateKind : uint32_t {
  Id67Direct,
  Id67Constant,
  Id67LdsFp16,
  Id67LdsF32,
  Id73Direct,
  Id73Constant,
  Id73LdsFp16,
  Id73LdsF32,
};

struct Candidate final {
  CandidateKind kind;
  const char *name;
  const void *function;
  uint32_t kernel_id;
  bool id73;
  size_t dynamic_lds;
};

Candidate candidate(const CandidateKind kind, const uint64_t k) {
  const size_t id73_lds =
      static_cast<size_t>((k / 16U) * 5U * sizeof(uint32_t));
  switch (kind) {
  case CandidateKind::Id67Direct:
    return {kind,
            "ID67-direct-scale",
            reinterpret_cast<const void *>(id67_kernel<ScaleMode::Direct>),
            67U,
            false,
            0U};
  case CandidateKind::Id67Constant:
    return {
        kind,
        "ID67-constant-fp16-lut",
        reinterpret_cast<const void *>(id67_kernel<ScaleMode::ConstantFp16>),
        67U,
        false,
        0U};
  case CandidateKind::Id67LdsFp16:
    return {kind,
            "ID67-lds-fp16-lut",
            reinterpret_cast<const void *>(id67_kernel<ScaleMode::LdsFp16>),
            67U,
            false,
            0U};
  case CandidateKind::Id67LdsF32:
    return {kind,
            "ID67-lds-f32-lut",
            reinterpret_cast<const void *>(id67_kernel<ScaleMode::LdsF32>),
            67U,
            false,
            0U};
  case CandidateKind::Id73Direct:
    return {kind,
            "ID73-direct-scale",
            reinterpret_cast<const void *>(id73_kernel<ScaleMode::Direct>),
            73U,
            true,
            id73_lds};
  case CandidateKind::Id73Constant:
    return {
        kind,
        "ID73-constant-fp16-lut",
        reinterpret_cast<const void *>(id73_kernel<ScaleMode::ConstantFp16>),
        73U,
        true,
        id73_lds};
  case CandidateKind::Id73LdsFp16:
    return {kind,
            "ID73-lds-fp16-lut",
            reinterpret_cast<const void *>(id73_kernel<ScaleMode::LdsFp16>),
            73U,
            true,
            id73_lds};
  case CandidateKind::Id73LdsF32:
    return {kind,
            "ID73-lds-f32-lut",
            reinterpret_cast<const void *>(id73_kernel<ScaleMode::LdsF32>),
            73U,
            true,
            id73_lds};
  }
  return {CandidateKind::Id67Direct, "invalid", nullptr, 0U, false, 0U};
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

float host_e2m1(const uint8_t code) {
  constexpr std::array<float, 8> values = {0.0F, 0.5F, 1.0F, 1.5F,
                                           2.0F, 3.0F, 4.0F, 6.0F};
  const float value = values[code & 7U];
  return (code & 8U) == 0U ? value : -value;
}

int32_t host_e2m1_scaled2(const uint8_t code) {
  return static_cast<int32_t>(host_e2m1(code) * 2.0F);
}

float host_e4m3(const uint8_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x80U) << 24U;
  const uint8_t magnitude = bits & 0x7fU;
  const uint8_t exponent = magnitude >> 3U;
  const uint8_t mantissa = magnitude & 7U;
  if (exponent == 0U) {
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
  const uint32_t value_bits = sign |
                              (static_cast<uint32_t>(exponent + 120U) << 23U) |
                              (static_cast<uint32_t>(mantissa) << 20U);
  float result = 0.0F;
  std::memcpy(&result, &value_bits, sizeof(result));
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

struct WeightPlane final {
  uint8_t *weight = nullptr;
  uint8_t *scales = nullptr;
};

struct Buffers final {
  uint8_t *activation = nullptr;
  uint8_t *activation_scales = nullptr;
  std::array<WeightPlane, kColdCopies> planes{};
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
  for (WeightPlane &plane : buffers->planes) {
    if (plane.scales != nullptr)
      (void)hipFree(plane.scales);
    if (plane.weight != nullptr)
      (void)hipFree(plane.weight);
  }
  if (buffers->activation_scales != nullptr)
    (void)hipFree(buffers->activation_scales);
  if (buffers->activation != nullptr)
    (void)hipFree(buffers->activation);
  *buffers = {};
}

bool make_buffers(const Shape &shape, Buffers *const buffers) {
  if (buffers == nullptr || shape.k == 0U || shape.n == 0U ||
      (shape.k % 16U) != 0U || shape.k > UINT64_MAX / shape.n)
    return false;
  const uint64_t weight_bytes_u64 = shape.k * shape.n / 2U;
  const uint64_t scale_bytes_u64 = shape.n * (shape.k / 16U);
  if (weight_bytes_u64 > SIZE_MAX || scale_bytes_u64 > SIZE_MAX ||
      shape.n > SIZE_MAX / sizeof(uint16_t))
    return false;
  const size_t weight_bytes = static_cast<size_t>(weight_bytes_u64);
  const size_t scale_bytes = static_cast<size_t>(scale_bytes_u64);
  const size_t output_bytes = static_cast<size_t>(shape.n * sizeof(uint16_t));
  if (!hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->activation),
                        static_cast<size_t>(shape.k / 2U)),
              "malloc activation") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->activation_scales),
                        static_cast<size_t>(shape.k / 16U)),
              "malloc activation scales")) {
    cleanup(buffers);
    return false;
  }
  for (WeightPlane &plane : buffers->planes) {
    if (!hip_ok(
            hipMalloc(reinterpret_cast<void **>(&plane.weight), weight_bytes),
            "malloc cold weight") ||
        !hip_ok(
            hipMalloc(reinterpret_cast<void **>(&plane.scales), scale_bytes),
            "malloc cold weight scales")) {
      cleanup(buffers);
      return false;
    }
  }
  if (!hip_ok(
          hipMalloc(reinterpret_cast<void **>(&buffers->weight_tensor_scale),
                    sizeof(float)),
          "malloc weight tensor scale") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->input_tensor_scale),
                        sizeof(float)),
              "malloc input tensor scale") ||
      !hip_ok(
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

void fill_inputs(const Shape &shape, std::vector<uint8_t> *const activation,
                 std::vector<uint8_t> *const activation_scales,
                 std::vector<uint8_t> *const weight,
                 std::vector<uint8_t> *const weight_scales) {
  const uint64_t blocks = shape.k / 16U;
  activation->assign(static_cast<size_t>(shape.k / 2U), 0U);
  activation_scales->resize(static_cast<size_t>(blocks));
  weight->resize(static_cast<size_t>(shape.k * shape.n / 2U));
  weight_scales->resize(static_cast<size_t>(shape.n * blocks));
  for (uint64_t byte = 0U; byte < shape.k / 2U; ++byte) {
    const uint8_t low = static_cast<uint8_t>((byte * 5U + 3U) & 0x0fU);
    const uint8_t high = static_cast<uint8_t>((byte * 11U + 7U) & 0x0fU);
    (*activation)[static_cast<size_t>(byte)] =
        static_cast<uint8_t>(low | (high << 4U));
  }
  for (uint64_t block = 0U; block < blocks; ++block)
    (*activation_scales)[static_cast<size_t>(block)] =
        kFiniteScaleCodes[(block * 3U + 5U) & 15U];
  for (uint64_t byte = 0U; byte < shape.k * shape.n / 2U; ++byte) {
    const uint8_t low = static_cast<uint8_t>((byte * 7U + 9U) & 0x0fU);
    const uint8_t high = static_cast<uint8_t>((byte * 13U + 1U) & 0x0fU);
    (*weight)[static_cast<size_t>(byte)] =
        static_cast<uint8_t>(low | (high << 4U));
  }
  for (uint64_t index = 0U; index < shape.n * blocks; ++index)
    (*weight_scales)[static_cast<size_t>(index)] =
        kFiniteScaleCodes[(index * 5U + 9U) & 15U];
}

bool upload_inputs(const Shape &shape, const std::vector<uint8_t> &activation,
                   const std::vector<uint8_t> &activation_scales,
                   const std::vector<uint8_t> &weight,
                   const std::vector<uint8_t> &weight_scales,
                   Buffers *const buffers) {
  const float weight_tensor_scale = 0.75F;
  const float input_tensor_scale = 1.125F;
  if (!hip_ok(hipMemcpy(buffers->activation, activation.data(),
                        activation.size(), hipMemcpyHostToDevice),
              "copy activation") ||
      !hip_ok(hipMemcpy(buffers->activation_scales, activation_scales.data(),
                        activation_scales.size(), hipMemcpyHostToDevice),
              "copy activation scales") ||
      !hip_ok(hipMemcpy(buffers->weight_tensor_scale, &weight_tensor_scale,
                        sizeof(float), hipMemcpyHostToDevice),
              "copy weight tensor scale") ||
      !hip_ok(hipMemcpy(buffers->input_tensor_scale, &input_tensor_scale,
                        sizeof(float), hipMemcpyHostToDevice),
              "copy input tensor scale"))
    return false;
  for (uint32_t copy = 0U; copy < kColdCopies; ++copy) {
    if (!hip_ok(hipMemcpy(buffers->planes[copy].weight, weight.data(),
                          weight.size(), hipMemcpyHostToDevice),
                "copy cold weight") ||
        !hip_ok(hipMemcpy(buffers->planes[copy].scales, weight_scales.data(),
                          weight_scales.size(), hipMemcpyHostToDevice),
                "copy cold weight scales"))
      return false;
  }
  return hip_ok(hipMemset(buffers->output, 0,
                          static_cast<size_t>(shape.n * sizeof(uint16_t))),
                "clear output");
}

size_t dynamic_lds(const Candidate &current) { return current.dynamic_lds; }

bool launch(const Candidate &current, const Shape &shape, const uint32_t copy,
            Buffers *const buffers) {
  if (copy >= kColdCopies || current.function == nullptr)
    return false;
  const uint64_t grid_u64 =
      (shape.n + kColumnsPerWorkgroup - 1U) / kColumnsPerWorkgroup;
  if (grid_u64 == 0U || grid_u64 > UINT32_MAX)
    return false;
  const dim3 grid(static_cast<uint32_t>(grid_u64));
  const dim3 block(kThreads);
  const WeightPlane &plane = buffers->planes[copy];
  const size_t shared = dynamic_lds(current);
#define LAUNCH_ID67(mode)                                                      \
  hipLaunchKernelGGL(                                                          \
      (id67_kernel<mode>), grid, block, shared, buffers->stream,               \
      buffers->activation, buffers->activation_scales, plane.weight,           \
      plane.scales, buffers->weight_tensor_scale, buffers->input_tensor_scale, \
      buffers->output, 1U, shape.k, shape.n)
#define LAUNCH_ID73(mode)                                                      \
  hipLaunchKernelGGL(                                                          \
      (id73_kernel<mode>), grid, block, shared, buffers->stream,               \
      buffers->activation, buffers->activation_scales, plane.weight,           \
      plane.scales, buffers->weight_tensor_scale, buffers->input_tensor_scale, \
      buffers->output, 1U, shape.k, shape.n)
  switch (current.kind) {
  case CandidateKind::Id67Direct:
    LAUNCH_ID67(ScaleMode::Direct);
    break;
  case CandidateKind::Id67Constant:
    LAUNCH_ID67(ScaleMode::ConstantFp16);
    break;
  case CandidateKind::Id67LdsFp16:
    LAUNCH_ID67(ScaleMode::LdsFp16);
    break;
  case CandidateKind::Id67LdsF32:
    LAUNCH_ID67(ScaleMode::LdsF32);
    break;
  case CandidateKind::Id73Direct:
    LAUNCH_ID73(ScaleMode::Direct);
    break;
  case CandidateKind::Id73Constant:
    LAUNCH_ID73(ScaleMode::ConstantFp16);
    break;
  case CandidateKind::Id73LdsFp16:
    LAUNCH_ID73(ScaleMode::LdsFp16);
    break;
  case CandidateKind::Id73LdsF32:
    LAUNCH_ID73(ScaleMode::LdsF32);
    break;
  }
#undef LAUNCH_ID67
#undef LAUNCH_ID73
  return hipGetLastError() == hipSuccess;
}

std::vector<CandidateKind> candidate_kinds(const bool id73) {
  if (id73)
    return {CandidateKind::Id73Direct, CandidateKind::Id73Constant,
            CandidateKind::Id73LdsFp16, CandidateKind::Id73LdsF32};
  return {CandidateKind::Id67Direct, CandidateKind::Id67Constant,
          CandidateKind::Id67LdsFp16, CandidateKind::Id67LdsF32};
}

bool capture_output(const Shape &shape, Buffers *const buffers,
                    std::vector<uint16_t> *const output) {
  output->resize(static_cast<size_t>(shape.n));
  return hip_ok(hipMemcpy(output->data(), buffers->output,
                          output->size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "copy output");
}

std::vector<uint16_t> cpu_oracle(const Shape &shape,
                                 const std::vector<uint8_t> &activation,
                                 const std::vector<uint8_t> &activation_scales,
                                 const std::vector<uint8_t> &weight,
                                 const std::vector<uint8_t> &weight_scales) {
  const uint64_t blocks = shape.k / 16U;
  std::vector<uint16_t> result(static_cast<size_t>(shape.n));
  for (uint64_t column = 0U; column < shape.n; ++column) {
    float accumulator = 0.0F;
    for (uint64_t block = 0U; block < blocks; ++block) {
      int32_t block_sum = 0;
      for (uint32_t inner = 0U; inner < 16U; ++inner) {
        const uint64_t index = block * 16U + inner;
        const uint8_t a_byte = activation[static_cast<size_t>(index / 2U)];
        const uint8_t w_byte =
            weight[static_cast<size_t>(column * shape.k / 2U + index / 2U)];
        const uint8_t a_code =
            (index & 1U) == 0U ? a_byte & 0x0fU : a_byte >> 4U;
        const uint8_t w_code =
            (index & 1U) == 0U ? w_byte & 0x0fU : w_byte >> 4U;
        block_sum += host_e2m1_scaled2(a_code) * host_e2m1_scaled2(w_code);
      }
      const float activation_scale =
          host_e4m3(activation_scales[static_cast<size_t>(block)]);
      const float weight_scale = host_e4m3(
          weight_scales[static_cast<size_t>(column * blocks + block)]);
      accumulator += static_cast<float>(block_sum) * 0.25F * activation_scale *
                     weight_scale;
    }
    result[static_cast<size_t>(column)] =
        host_bf16_rne(accumulator * 0.75F * 1.125F);
  }
  return result;
}

bool compare_exact(const char *const label, const std::vector<uint16_t> &actual,
                   const std::vector<uint16_t> &expected) {
  size_t mismatches = 0U;
  uint32_t max_ulp = 0U;
  const size_t count = std::min(actual.size(), expected.size());
  for (size_t index = 0U; index < count; ++index) {
    if (actual[index] != expected[index])
      ++mismatches;
    const uint32_t left = actual[index];
    const uint32_t right = expected[index];
    max_ulp = std::max(max_ulp, left > right ? left - right : right - left);
  }
  mismatches += actual.size() == expected.size() ? 0U : 1U;
  std::printf("oracle=%s count=%zu mismatches=%zu max_bf16_ulp=%u status=%s\n",
              label, count, mismatches, max_ulp,
              mismatches == 0U ? "PASS" : "FAIL");
  return mismatches == 0U;
}

bool finite_output(const std::vector<uint16_t> &output) {
  for (const uint16_t bits : output)
    if ((bits & 0x7f80U) == 0x7f80U)
      return false;
  return true;
}

bool run_scale_oracle() {
  std::array<uint8_t, kScaleLutEntries> input{};
  std::array<uint16_t, kScaleLutEntries> actual_lut{};
  std::array<uint32_t, kScaleLutEntries> actual_direct{};
  std::array<uint32_t, kScaleLutEntries> actual_converted{};
  for (uint32_t index = 0U; index < kScaleLutEntries; ++index)
    input[index] = static_cast<uint8_t>(index);
  uint16_t *device_lut = nullptr;
  uint32_t *device_direct = nullptr;
  uint32_t *device_converted = nullptr;
  bool ok = hip_ok(hipMalloc(reinterpret_cast<void **>(&device_lut),
                             actual_lut.size() * sizeof(uint16_t)),
                   "malloc LUT oracle") &&
            hip_ok(hipMalloc(reinterpret_cast<void **>(&device_direct),
                             actual_direct.size() * sizeof(uint32_t)),
                   "malloc direct oracle") &&
            hip_ok(hipMalloc(reinterpret_cast<void **>(&device_converted),
                             actual_converted.size() * sizeof(uint32_t)),
                   "malloc converted oracle");
  if (ok) {
    hipLaunchKernelGGL(scale_lut_oracle_kernel, dim3(1U), dim3(256U), 0U,
                       nullptr, device_lut, device_direct, device_converted);
    ok = hip_ok(hipGetLastError(), "launch scale oracle") &&
         hip_ok(hipDeviceSynchronize(), "sync scale oracle") &&
         hip_ok(hipMemcpy(actual_lut.data(), device_lut,
                          actual_lut.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "copy LUT oracle") &&
         hip_ok(hipMemcpy(actual_direct.data(), device_direct,
                          actual_direct.size() * sizeof(uint32_t),
                          hipMemcpyDeviceToHost),
                "copy direct oracle") &&
         hip_ok(hipMemcpy(actual_converted.data(), device_converted,
                          actual_converted.size() * sizeof(uint32_t),
                          hipMemcpyDeviceToHost),
                "copy converted oracle");
  }
  size_t lut_mismatches = 0U;
  size_t value_mismatches = 0U;
  if (ok) {
    for (uint32_t index = 0U; index < kScaleLutEntries; ++index) {
      const uint16_t expected = sllm_lowp::e4m3fn_to_fp16_bits(input[index]);
      if (actual_lut[index] != expected)
        ++lut_mismatches;
      const bool direct_nan =
          (actual_direct[index] & 0x7f800000U) == 0x7f800000U &&
          (actual_direct[index] & 0x007fffffU) != 0U;
      const bool converted_nan =
          (actual_converted[index] & 0x7f800000U) == 0x7f800000U &&
          (actual_converted[index] & 0x007fffffU) != 0U;
      if (direct_nan || converted_nan) {
        if (!(direct_nan && converted_nan))
          ++value_mismatches;
      } else if (actual_direct[index] != actual_converted[index]) {
        ++value_mismatches;
      }
    }
  }
  std::printf("oracle scale_lut codes=256 lut_mismatches=%zu "
              "value_mismatches=%zu status=%s\n",
              lut_mismatches, value_mismatches,
              ok && lut_mismatches == 0U && value_mismatches == 0U ? "PASS"
                                                                   : "FAIL");
  if (device_converted != nullptr)
    (void)hipFree(device_converted);
  if (device_direct != nullptr)
    (void)hipFree(device_direct);
  if (device_lut != nullptr)
    (void)hipFree(device_lut);
  return ok && lut_mismatches == 0U && value_mismatches == 0U;
}

bool initialize_scale_lut() {
  std::array<uint16_t, kScaleLutEntries> host_lut{};
  for (uint32_t code = 0U; code < kScaleLutEntries; ++code)
    host_lut[code] = sllm_lowp::e4m3fn_to_fp16_bits(static_cast<uint8_t>(code));
  return hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(kScaleFp16Lut), host_lut.data(),
                                  host_lut.size() * sizeof(uint16_t), 0U,
                                  hipMemcpyHostToDevice),
                "copy scale constant LUT");
}

bool run_block16_oracle() {
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
  bool ok =
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_activation), 8U),
             "malloc subtotal activation") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_weight), 8U),
             "malloc subtotal weight") &&
      hip_ok(
          hipMalloc(reinterpret_cast<void **>(&device_output), sizeof(int32_t)),
          "malloc subtotal output") &&
      hip_ok(hipMemcpy(device_activation, activation.data(), 8U,
                       hipMemcpyHostToDevice),
             "copy subtotal activation") &&
      hip_ok(hipMemcpy(device_weight, weight.data(), 8U, hipMemcpyHostToDevice),
             "copy subtotal weight");
  if (ok) {
    hipLaunchKernelGGL(block16_subtotal_kernel, dim3(1U), dim3(1U), 0U, nullptr,
                       device_activation, device_weight, device_output);
    ok = hip_ok(hipGetLastError(), "launch subtotal") &&
         hip_ok(hipDeviceSynchronize(), "sync subtotal");
  }
  int32_t actual = 0;
  if (ok)
    ok = hip_ok(hipMemcpy(&actual, device_output, sizeof(actual),
                          hipMemcpyDeviceToHost),
                "copy subtotal output");
  int32_t expected = 0;
  for (uint32_t index = 0U; index < 16U; ++index)
    expected += host_e2m1_scaled2(static_cast<uint8_t>(index)) *
                host_e2m1_scaled2(static_cast<uint8_t>(15U - index));
  std::printf("oracle block16 subtotal expected=%d actual=%d status=%s\n",
              expected, actual, ok && expected == actual ? "PASS" : "FAIL");
  if (device_output != nullptr)
    (void)hipFree(device_output);
  if (device_weight != nullptr)
    (void)hipFree(device_weight);
  if (device_activation != nullptr)
    (void)hipFree(device_activation);
  return ok && expected == actual;
}

bool measure(const Candidate &current, const Shape &shape,
             Buffers *const buffers, float *const median_us,
             float *const mad_us, float *const min_us, float *const max_us) {
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    if (!launch(current, shape, static_cast<uint32_t>(warmup) % kColdCopies,
                buffers) ||
        !hip_ok(hipStreamSynchronize(buffers->stream), "warmup sync"))
      return false;
  }
  std::array<float, kMeasured> samples{};
  for (int iteration = 0; iteration < kMeasured; ++iteration) {
    const uint32_t copy = static_cast<uint32_t>(iteration) % kColdCopies;
    if (!hip_ok(hipEventRecord(buffers->start, buffers->stream),
                "event start") ||
        !launch(current, shape, copy, buffers) ||
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
  *median_us = samples[kMeasured / 2U];
  *min_us = samples.front();
  *max_us = samples.back();
  std::array<float, kMeasured> deviations{};
  for (int index = 0; index < kMeasured; ++index)
    deviations[static_cast<size_t>(index)] =
        std::fabs(samples[static_cast<size_t>(index)] - *median_us);
  std::sort(deviations.begin(), deviations.end());
  *mad_us = deviations[kMeasured / 2U];
  return true;
}

void print_resources(const Candidate &current) {
  hipFuncAttributes attributes{};
  const hipError_t attr_status =
      hipFuncGetAttributes(&attributes, current.function);
  int active_blocks = 0;
  const hipError_t occupancy_status =
      hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks, current.function, kThreads, current.dynamic_lds);
  std::printf(
      "resources candidate=%s kernel_id=%u registers=%d static_lds=%zu "
      "local=%zu dynamic_lds=%zu active_blocks=%d attr=%s occupancy=%s\n",
      current.name, current.kernel_id, attributes.numRegs,
      attributes.sharedSizeBytes, attributes.localSizeBytes,
      current.dynamic_lds, active_blocks, hipGetErrorString(attr_status),
      hipGetErrorString(occupancy_status));
}

bool compare_candidates(const Candidate &current, const Shape &shape,
                        Buffers *const buffers,
                        const std::vector<uint16_t> &control_output,
                        const std::vector<uint16_t> *const expected,
                        bool *const all_ok) {
  if (!launch(current, shape, 0U, buffers) ||
      !hip_ok(hipStreamSynchronize(buffers->stream), "correctness sync"))
    return false;
  std::vector<uint16_t> observed;
  if (!capture_output(shape, buffers, &observed))
    return false;
  const bool finite = finite_output(observed);
  size_t mismatches = 0U;
  for (size_t index = 0U; index < observed.size(); ++index)
    if (observed[index] != control_output[index])
      ++mismatches;
  std::printf("compare candidate=%s K=%llu N=%llu control_mismatches=%zu "
              "finite=%s status=%s\n",
              current.name, static_cast<unsigned long long>(shape.k),
              static_cast<unsigned long long>(shape.n), mismatches,
              finite ? "PASS" : "FAIL",
              mismatches == 0U && finite ? "PASS" : "FAIL");
  if (expected != nullptr)
    *all_ok = compare_exact(current.name, observed, *expected) && *all_ok;
  *all_ok = mismatches == 0U && finite && *all_ok;
  // A second execution on a different cold-buffer slot checks deterministic
  // output without relying on a warm single-pointer cache.
  if (!launch(current, shape, 1U, buffers) ||
      !hip_ok(hipStreamSynchronize(buffers->stream), "determinism sync"))
    return false;
  std::vector<uint16_t> repeated;
  if (!capture_output(shape, buffers, &repeated))
    return false;
  size_t deterministic_mismatches = 0U;
  for (size_t index = 0U; index < observed.size(); ++index)
    if (observed[index] != repeated[index])
      ++deterministic_mismatches;
  std::printf("determinism candidate=%s mismatches=%zu status=%s\n",
              current.name, deterministic_mismatches,
              deterministic_mismatches == 0U ? "PASS" : "FAIL");
  *all_ok = deterministic_mismatches == 0U && *all_ok;
  return true;
}

} // namespace

int main() {
  constexpr int device = 0;
  if (!hip_ok(hipSetDevice(device), "set device"))
    return EXIT_FAILURE;
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device),
              "get device properties") ||
      !exact_gfx1030(properties.gcnArchName)) {
    std::fprintf(stderr, "exact gfx1030 required; got %s\n",
                 properties.gcnArchName);
    return EXIT_FAILURE;
  }
  size_t free_bytes = 0U;
  size_t total_bytes = 0U;
  (void)hipMemGetInfo(&free_bytes, &total_bytes);
  std::printf("target=%s device=%d pci=%04x:%02x:%02x name=%s free_bytes=%zu "
              "total_bytes=%zu\n",
              properties.gcnArchName, device, properties.pciDomainID,
              properties.pciBusID, properties.pciDeviceID, properties.name,
              free_bytes, total_bytes);

  bool all_ok =
      initialize_scale_lut() && run_scale_oracle() && run_block16_oracle();
  // Small non-aligned boundary: N=37 exercises the 32-column tail and K=48
  // exercises three block16 scale domains.  The CPU oracle is independent of
  // the device LUT and validates every candidate's full BF16 output.
  const Shape oracle_shape{48U, 37U, 0U, "nonaligned-boundary", false};
  std::vector<uint8_t> activation;
  std::vector<uint8_t> activation_scales;
  std::vector<uint8_t> weight;
  std::vector<uint8_t> weight_scales;
  fill_inputs(oracle_shape, &activation, &activation_scales, &weight,
              &weight_scales);
  Buffers oracle_buffers;
  if (!make_buffers(oracle_shape, &oracle_buffers) ||
      !upload_inputs(oracle_shape, activation, activation_scales, weight,
                     weight_scales, &oracle_buffers)) {
    cleanup(&oracle_buffers);
    return EXIT_FAILURE;
  }
  const std::vector<uint16_t> oracle_expected = cpu_oracle(
      oracle_shape, activation, activation_scales, weight, weight_scales);
  std::vector<CandidateKind> oracle_kinds = candidate_kinds(false);
  std::vector<uint16_t> oracle_control;
  for (size_t index = 0U; index < oracle_kinds.size(); ++index) {
    const Candidate current = candidate(oracle_kinds[index], oracle_shape.k);
    std::vector<uint16_t> observed;
    if (!launch(current, oracle_shape, 0U, &oracle_buffers) ||
        !hip_ok(hipStreamSynchronize(oracle_buffers.stream), "oracle sync") ||
        !capture_output(oracle_shape, &oracle_buffers, &observed)) {
      cleanup(&oracle_buffers);
      return EXIT_FAILURE;
    }
    if (index == 0U) {
      oracle_control = observed;
      all_ok =
          compare_exact("ID67-direct CPU-oracle", observed, oracle_expected) &&
          all_ok;
    } else {
      all_ok =
          compare_exact(candidate(oracle_kinds[index], oracle_shape.k).name,
                        observed, oracle_expected) &&
          all_ok;
      all_ok = compare_exact("ID67-candidate-vs-control", observed,
                             oracle_control) &&
               all_ok;
    }
  }
  cleanup(&oracle_buffers);

  struct Measurement final {
    double median_us = 0.0;
    double mad_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
  };
  std::array<std::array<Measurement, 4>, kQwenShapes.size()> measurements{};
  for (size_t shape_index = 0U; shape_index < kQwenShapes.size();
       ++shape_index) {
    const Shape &shape = kQwenShapes[shape_index];
    const std::vector<CandidateKind> kinds = candidate_kinds(shape.id73);
    std::vector<uint8_t> shape_activation;
    std::vector<uint8_t> shape_activation_scales;
    std::vector<uint8_t> shape_weight;
    std::vector<uint8_t> shape_weight_scales;
    fill_inputs(shape, &shape_activation, &shape_activation_scales,
                &shape_weight, &shape_weight_scales);
    Buffers buffers;
    if (!make_buffers(shape, &buffers) ||
        !upload_inputs(shape, shape_activation, shape_activation_scales,
                       shape_weight, shape_weight_scales, &buffers)) {
      cleanup(&buffers);
      return EXIT_FAILURE;
    }
    std::vector<uint16_t> control_output;
    for (size_t candidate_index = 0U; candidate_index < kinds.size();
         ++candidate_index) {
      const Candidate current = candidate(kinds[candidate_index], shape.k);
      print_resources(current);
      float median_us = 0.0F;
      float mad_us = 0.0F;
      float min_us = 0.0F;
      float max_us = 0.0F;
      if (!measure(current, shape, &buffers, &median_us, &mad_us, &min_us,
                   &max_us)) {
        cleanup(&buffers);
        return EXIT_FAILURE;
      }
      measurements[shape_index][candidate_index] = {median_us, mad_us, min_us,
                                                    max_us};
      const double weight_bytes =
          static_cast<double>(shape.k) * static_cast<double>(shape.n) / 2.0 +
          static_cast<double>(shape.n) * static_cast<double>(shape.k / 16U);
      std::printf(
          "result candidate=%s role=%s K=%llu N=%llu median_us=%.3f "
          "mad_us=%.3f min_us=%.3f max_us=%.3f gbps_weight_plus_scale=%.6f\n",
          current.name, shape.role, static_cast<unsigned long long>(shape.k),
          static_cast<unsigned long long>(shape.n), median_us, mad_us, min_us,
          max_us, weight_bytes / static_cast<double>(median_us) / 1000.0);
      if (!launch(current, shape, 0U, &buffers) ||
          !hip_ok(hipStreamSynchronize(buffers.stream), "shape compare sync")) {
        cleanup(&buffers);
        return EXIT_FAILURE;
      }
      std::vector<uint16_t> observed;
      if (!capture_output(shape, &buffers, &observed)) {
        cleanup(&buffers);
        return EXIT_FAILURE;
      }
      if (candidate_index == 0U) {
        control_output = observed;
      } else {
        all_ok = compare_candidates(current, shape, &buffers, control_output,
                                    nullptr, &all_ok) &&
                 all_ok;
      }
    }
    cleanup(&buffers);
    (void)hipMemGetInfo(&free_bytes, &total_bytes);
    std::printf("cleanup role=%s free_bytes=%zu total_bytes=%zu\n", shape.role,
                free_bytes, total_bytes);
  }

  auto print_weighted = [&](const char *const label, const bool id73_only,
                            const uint32_t occurrence_filter) {
    for (size_t candidate_index = 0U; candidate_index < 4U; ++candidate_index) {
      double weighted_time_us = 0.0;
      double weighted_bytes = 0.0;
      uint32_t weighted_calls = 0U;
      for (size_t shape_index = 0U; shape_index < kQwenShapes.size();
           ++shape_index) {
        const Shape &shape = kQwenShapes[shape_index];
        if (shape.id73 != id73_only)
          continue;
        const uint32_t occurrences =
            occurrence_filter == 0U ? shape.occurrences : occurrence_filter;
        weighted_calls += occurrences;
        const Measurement &measurement =
            measurements[shape_index][candidate_index];
        weighted_time_us +=
            static_cast<double>(occurrences) * measurement.median_us;
        weighted_bytes +=
            static_cast<double>(occurrences) *
            (static_cast<double>(shape.k) * static_cast<double>(shape.n) / 2.0 +
             static_cast<double>(shape.n) * static_cast<double>(shape.k / 16U));
      }
      const Measurement &first = measurements[id73_only ? 0U : 2U][0U];
      (void)first;
      const double baseline_time_us = [&]() {
        double result = 0.0;
        for (size_t shape_index = 0U; shape_index < kQwenShapes.size();
             ++shape_index) {
          if (kQwenShapes[shape_index].id73 != id73_only)
            continue;
          const uint32_t occurrences =
              occurrence_filter == 0U ? kQwenShapes[shape_index].occurrences
                                      : occurrence_filter;
          result += static_cast<double>(occurrences) *
                    measurements[shape_index][0U].median_us;
        }
        return result;
      }();
      std::printf(
          "weighted label=%s candidates_index=%zu calls=%u ms_per_token=%.6f "
          "gbps_weight_plus_scale=%.6f speedup_vs_direct=%.6f%%\n",
          label, candidate_index, weighted_calls, weighted_time_us / 1000.0,
          weighted_bytes / weighted_time_us / 1000.0,
          candidate_index == 0U
              ? 0.0
              : (baseline_time_us / weighted_time_us - 1.0) * 100.0);
    }
  };
  print_weighted("ID73-wide-down", true, 0U);
  print_weighted("ID67-other-shapes", false, 0U);
  // The direct NVFP4 probe's production weighting is also retained for the
  // exact ID73 pair: 112 wide and 56 down calls per token.
  for (size_t candidate_index = 0U; candidate_index < 4U; ++candidate_index) {
    const double time_us = 112.0 * measurements[0][candidate_index].median_us +
                           56.0 * measurements[1][candidate_index].median_us;
    const double bytes = 112.0 * (5120.0 * 17408.0 / 2.0 + 17408.0 * 320.0) +
                         56.0 * (17408.0 * 5120.0 / 2.0 + 5120.0 * 1088.0);
    std::printf("weighted label=ID73-production-112+56 candidates_index=%zu "
                "ms_per_token=%.6f gbps_weight_plus_scale=%.6f "
                "speedup_vs_direct=%.6f%%\n",
                candidate_index, time_us / 1000.0, bytes / time_us / 1000.0,
                candidate_index == 0U ? 0.0
                                      : ((112.0 * measurements[0][0].median_us +
                                          56.0 * measurements[1][0].median_us) /
                                             time_us -
                                         1.0) *
                                            100.0);
  }
  (void)hipMemGetInfo(&free_bytes, &total_bytes);
  std::printf("cleanup final free_bytes=%zu total_bytes=%zu\n", free_bytes,
              total_bytes);
  std::printf("summary status=%s qwen_shapes=%zu occurrences=%u warmups=%d "
              "measured=%d goal_speedup=1.10x\n",
              all_ok ? "PASS" : "FAIL", kQwenShapes.size(), shape_occurrences(),
              kWarmups, kMeasured);
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
