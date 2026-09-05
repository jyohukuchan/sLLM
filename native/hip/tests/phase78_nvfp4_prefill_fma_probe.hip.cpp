// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase78-nvfp4-byte-permute-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// 3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70, ggml/src/ggml-cuda/vecdotq.cuh
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

// Phase 78 bounded NVFP4 prefill ID62 explicit-FMA probe.
//
// This developer-only binary compares the linked gfx1030 production ID62
// prefill launcher with a probe-local clone whose only arithmetic change is
// explicit fmaf for scale accumulation. Packed NVFP4 decoding, 64x64/K32
// geometry, block16 sums, stage/dtype, tensor scale, and BF16 RNE remain the
// same. No production selector or source is changed by this probe.

#include "matmul_kernel_internal.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
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
constexpr uint32_t kWarmups = 3U;
constexpr uint32_t kMeasured = 10U;
constexpr uint32_t kOracleSamples = 64U;
constexpr uint32_t kSeed = UINT32_C(0x243f6a88);
constexpr float kWeightTensorScale = 0.75F;
constexpr float kInputTensorScale = 1.125F;

struct Shape final {
  uint64_t m;
  uint64_t k;
  uint64_t n;
  uint32_t occurrences;
  const char *name;
};

struct HostInputs final {
  std::vector<uint8_t> activation;
  std::vector<uint8_t> activation_scales;
  std::vector<uint8_t> weight;
  std::vector<uint8_t> weight_scales;
};

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

struct OraclePoint final {
  std::size_t index;
  double expected;
  double absolute_sum;
  uint16_t expected_bf16;
};

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "hip error operation=%s status=%s\n", operation,
               hipGetErrorString(status));
  return false;
}

bool exact_gfx1030(const char *const arch) {
  if (arch == nullptr)
    return false;
  const std::string_view value(arch);
  constexpr std::string_view target = "gfx1030";
  return value == target || (value.size() > target.size() &&
                             value.compare(0U, target.size(), target) == 0U &&
                             value[target.size()] == ':');
}

bool parse_device(const char *const text, int *const device) {
  if (text == nullptr || device == nullptr)
    return false;
  char *end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed < 0L || parsed > INT32_MAX)
    return false;
  *device = static_cast<int>(parsed);
  return true;
}

__host__ __device__ __forceinline__ float e4m3fn_to_float(const uint8_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x80U) << 24U;
  const uint32_t magnitude = bits & 0x7fU;
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
    float result = 0.0F;
    std::memcpy(&result, &raw, sizeof(result));
    return result;
#endif
  }
  if (magnitude == 0x7fU) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __uint_as_float(sign | 0x7fc00000U);
#else
    uint32_t raw = sign | 0x7fc00000U;
    float result = 0.0F;
    std::memcpy(&result, &raw, sizeof(result));
    return result;
#endif
  }
#if defined(__HIP_DEVICE_COMPILE__)
  return __uint_as_float(sign | ((exponent + 120U) << 23U) | (mantissa << 20U));
#else
  const uint32_t raw = sign | ((exponent + 120U) << 23U) | (mantissa << 20U);
  float result = 0.0F;
  std::memcpy(&result, &raw, sizeof(result));
  return result;
#endif
}

__device__ __forceinline__ uint16_t bf16_rne(const float value) {
  const uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7f800000U) == 0x7f800000U) {
    if ((bits & 0x007fffffU) != 0U) {
      return static_cast<uint16_t>(((bits >> 16U) & 0x8000U) | 0x7fc0U |
                                   ((bits >> 16U) & 0x003fU));
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & 0xffffU;
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U))
    ++upper;
  return static_cast<uint16_t>(upper);
}

float host_e2m1(const uint8_t code) {
  constexpr float values[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  const float value = values[code & 7U];
  return (code & 8U) == 0U ? value : -value;
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
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U))
    ++upper;
  return static_cast<uint16_t>(upper);
}

float host_bf16_to_float(const uint16_t bits) {
  const uint32_t expanded = static_cast<uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &expanded, sizeof(value));
  return value;
}

uint32_t ordered_bf16(const uint16_t bits) {
  if ((bits & 0x7fffU) == 0U)
    return 0x8000U;
  return (bits & 0x8000U) != 0U ? static_cast<uint16_t>(~bits)
                                : static_cast<uint16_t>(bits | 0x8000U);
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

HostInputs make_inputs(const Shape &shape) {
  const uint64_t blocks = shape.k / 16U;
  HostInputs inputs;
  inputs.activation.assign(static_cast<size_t>(shape.m * shape.k / 2U), 0U);
  inputs.activation_scales.resize(static_cast<size_t>(shape.m * blocks));
  inputs.weight.assign(static_cast<size_t>(shape.n * shape.k / 2U), 0U);
  inputs.weight_scales.resize(static_cast<size_t>(shape.n * blocks));
  for (uint64_t row = 0U; row < shape.m; ++row) {
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      const uint8_t code = static_cast<uint8_t>(
          mix32(static_cast<uint32_t>(row * shape.k + inner) ^ kSeed) & 0x0fU);
      const size_t index = static_cast<size_t>(row * shape.k / 2U + inner / 2U);
      if ((inner & 1U) == 0U)
        inputs.activation[index] = code;
      else
        inputs.activation[index] |= static_cast<uint8_t>(code << 4U);
    }
    for (uint64_t block = 0U; block < blocks; ++block) {
      inputs.activation_scales[static_cast<size_t>(row * blocks + block)] =
          positive_finite_e4m3(row * blocks + block, kSeed);
    }
  }
  for (uint64_t column = 0U; column < shape.n; ++column) {
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      const uint8_t code = static_cast<uint8_t>(
          mix32(static_cast<uint32_t>(column * shape.k + inner) ^ kSeed ^
                UINT32_C(0x9e3779b9)) &
          0x0fU);
      const size_t index =
          static_cast<size_t>(column * shape.k / 2U + inner / 2U);
      if ((inner & 1U) == 0U)
        inputs.weight[index] = code;
      else
        inputs.weight[index] |= static_cast<uint8_t>(code << 4U);
    }
    for (uint64_t block = 0U; block < blocks; ++block) {
      inputs.weight_scales[static_cast<size_t>(column * blocks + block)] =
          positive_finite_e4m3(column * blocks + block,
                               kSeed ^ UINT32_C(0xa5a5a5a5));
    }
  }
  return inputs;
}

void cleanup(DeviceBuffers *const buffers) {
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

struct Comparison final {
  size_t mismatches = 0U;
  uint32_t max_ulp = 0U;
  double max_abs = 0.0;
  double max_rel = 0.0;
};

Comparison compare(const std::vector<uint16_t> &reference,
                   const std::vector<uint16_t> &candidate) {
  Comparison result;
  for (size_t index = 0U; index < reference.size(); ++index) {
    result.mismatches +=
        static_cast<size_t>(reference[index] != candidate[index]);
    const double absolute =
        std::fabs(static_cast<double>(host_bf16_to_float(reference[index]) -
                                      host_bf16_to_float(candidate[index])));
    result.max_abs = std::max(result.max_abs, absolute);
    result.max_rel = std::max(
        result.max_rel,
        absolute /
            std::max(1.0e-100, std::fabs(static_cast<double>(
                                   host_bf16_to_float(reference[index])))));
    const uint32_t lhs = ordered_bf16(reference[index]);
    const uint32_t rhs = ordered_bf16(candidate[index]);
    result.max_ulp =
        std::max(result.max_ulp, lhs > rhs ? lhs - rhs : rhs - lhs);
  }
  return result;
}

void print_comparison(const Shape &shape, const char *const name,
                      const Comparison &comparison) {
  std::printf("compare shape=%s pair=%s bf16_mismatches=%zu max_bf16_ulp=%u "
              "max_abs=%.9e max_rel=%.9e status=%s\n",
              shape.name, name, comparison.mismatches, comparison.max_ulp,
              comparison.max_abs, comparison.max_rel,
              comparison.mismatches == 0U ? "PASS" : "FAIL");
}

struct Dp4aScaledPacks final {
  int32_t even;
  int32_t odd;
};

__device__ __forceinline__ Dp4aScaledPacks
dp4a_e2m1x8_scaled2_to_i8x4_pair(const uint32_t packed) {
  // Exact signed bytes for E2M1 value * 2.  Even and odd nibbles are split
  // because one v_dot4_i32_i8 consumes four byte lanes.
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
  return {
      static_cast<int32_t>(
          __builtin_amdgcn_perm(even_high, even_low, even_select)),
      static_cast<int32_t>(
          __builtin_amdgcn_perm(odd_high, odd_low, odd_select)),
  };
}

__device__ __forceinline__ int32_t dp4a_signed_dot4(const int32_t lhs,
                                                    const int32_t rhs,
                                                    const int32_t accumulator) {
#if __has_builtin(__builtin_amdgcn_sdot4)
  return __builtin_amdgcn_sdot4(lhs, rhs, accumulator, false);
#else
  int32_t result = accumulator;
#pragma unroll
  for (uint32_t lane = 0U; lane < 4U; ++lane) {
    result += static_cast<int8_t>(static_cast<uint32_t>(lhs) >> (lane * 8U)) *
              static_cast<int8_t>(static_cast<uint32_t>(rhs) >> (lane * 8U));
  }
  return result;
#endif
}

// Exact copy of ID62's 64x64/K32 geometry and arithmetic contract. The only
// candidate change is explicit fmaf for the scale accumulate; the production
// launcher remains the linked control.
__global__ __launch_bounds__(256, 1) void fma_candidate_id62_kernel(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  constexpr uint32_t tile_m = 64U;
  constexpr uint32_t tile_n = 64U;
  constexpr uint32_t tile_k = 32U;
  constexpr uint32_t block_k = 16U;
  constexpr uint32_t blocks_per_stage = tile_k / block_k;
  constexpr uint32_t packed_groups_per_stage = tile_k / 4U;
  constexpr uint32_t packed_chunks_per_stage = tile_k / 8U;
  constexpr uint32_t lds_group_stride = packed_groups_per_stage + 1U;
  constexpr uint32_t lds_scale_stride = blocks_per_stage + 1U;
  constexpr uint32_t thread_rows = 16U;
  constexpr uint32_t thread_columns = 16U;
  constexpr uint32_t rows_per_thread = tile_m / thread_rows;
  constexpr uint32_t columns_per_thread = tile_n / thread_columns;

  __shared__ int32_t activation_tile[tile_m][lds_group_stride];
  __shared__ int32_t weight_tile[tile_n][lds_group_stride];
  __shared__ float activation_scale_tile[tile_m][lds_scale_stride];
  __shared__ float weight_scale_tile[tile_n][lds_scale_stride];

  const uint64_t column_tiles = (n + tile_n - 1U) / tile_n;
  const uint64_t tile_index = blockIdx.x;
  const uint64_t row_base = (tile_index / column_tiles) * tile_m;
  const uint64_t column_base = (tile_index % column_tiles) * tile_n;
  const uint32_t thread = threadIdx.x;
  const uint32_t local_row = thread >> 4U;
  const uint32_t local_column = thread & UINT32_C(15);
  const uint64_t packed_row_bytes = k / UINT64_C(2);
  const uint64_t blocks_per_row = k / block_k;
  float accumulators[rows_per_thread][columns_per_thread] = {};

  for (uint64_t base = 0U; base < k; base += tile_k) {
    for (uint32_t index = thread; index < tile_m * packed_chunks_per_stage;
         index += blockDim.x) {
      const uint32_t row = index / packed_chunks_per_stage;
      const uint32_t chunk = index % packed_chunks_per_stage;
      const uint64_t source_row = row_base + row;
      const uint64_t inner = base + static_cast<uint64_t>(chunk) * 8U;
      const Dp4aScaledPacks values =
          source_row < m && inner + 8U <= k
              ? dp4a_e2m1x8_scaled2_to_i8x4_pair(__builtin_nontemporal_load(
                    reinterpret_cast<const uint32_t *>(
                        packed_activation + source_row * packed_row_bytes +
                        inner / UINT64_C(2))))
              : Dp4aScaledPacks{0, 0};
      activation_tile[row][chunk * 2U] = values.even;
      activation_tile[row][chunk * 2U + 1U] = values.odd;
    }
    for (uint32_t index = thread; index < tile_n * packed_chunks_per_stage;
         index += blockDim.x) {
      const uint32_t column = index / packed_chunks_per_stage;
      const uint32_t chunk = index % packed_chunks_per_stage;
      const uint64_t source_column = column_base + column;
      const uint64_t inner = base + static_cast<uint64_t>(chunk) * 8U;
      const Dp4aScaledPacks values =
          source_column < n && inner + 8U <= k
              ? dp4a_e2m1x8_scaled2_to_i8x4_pair(__builtin_nontemporal_load(
                    reinterpret_cast<const uint32_t *>(
                        packed_weight + source_column * packed_row_bytes +
                        inner / UINT64_C(2))))
              : Dp4aScaledPacks{0, 0};
      weight_tile[column][chunk * 2U] = values.even;
      weight_tile[column][chunk * 2U + 1U] = values.odd;
    }
    for (uint32_t index = thread; index < tile_m * blocks_per_stage;
         index += blockDim.x) {
      const uint32_t row = index / blocks_per_stage;
      const uint32_t block = index % blocks_per_stage;
      const uint64_t source_row = row_base + row;
      const uint64_t source_block = base / block_k + block;
      activation_scale_tile[row][block] =
          source_row < m && source_block < blocks_per_row
              ? e4m3fn_to_float(
                    activation_block_scales[source_row * blocks_per_row +
                                            source_block])
              : 0.0F;
    }
    for (uint32_t index = thread; index < tile_n * blocks_per_stage;
         index += blockDim.x) {
      const uint32_t column = index / blocks_per_stage;
      const uint32_t block = index % blocks_per_stage;
      const uint64_t source_column = column_base + column;
      const uint64_t source_block = base / block_k + block;
      weight_scale_tile[column][block] =
          source_column < n && source_block < blocks_per_row
              ? e4m3fn_to_float(
                    weight_block_scales[source_column * blocks_per_row +
                                        source_block])
              : 0.0F;
    }
    __syncthreads();

#pragma unroll
    for (uint32_t block = 0U; block < blocks_per_stage; ++block) {
      int32_t block_sums[rows_per_thread][columns_per_thread] = {};
#pragma unroll
      for (uint32_t group = 0U; group < block_k / 4U; ++group) {
        int32_t activation_packs[rows_per_thread];
        int32_t weight_packs[columns_per_thread];
#pragma unroll
        for (uint32_t row = 0U; row < rows_per_thread; ++row) {
          activation_packs[row] =
              activation_tile[local_row + row * thread_rows]
                             [block * (block_k / 4U) + group];
        }
#pragma unroll
        for (uint32_t column = 0U; column < columns_per_thread; ++column) {
          weight_packs[column] =
              weight_tile[local_column + column * thread_columns]
                         [block * (block_k / 4U) + group];
        }
#pragma unroll
        for (uint32_t row = 0U; row < rows_per_thread; ++row) {
#pragma unroll
          for (uint32_t column = 0U; column < columns_per_thread; ++column) {
            block_sums[row][column] =
                dp4a_signed_dot4(activation_packs[row], weight_packs[column],
                                 block_sums[row][column]);
          }
        }
      }
#pragma unroll
      for (uint32_t row = 0U; row < rows_per_thread; ++row) {
        const float activation_scale =
            activation_scale_tile[local_row + row * thread_rows][block];
#pragma unroll
        for (uint32_t column = 0U; column < columns_per_thread; ++column) {
          const float weight_scale =
              weight_scale_tile[local_column + column * thread_columns][block];
          accumulators[row][column] =
              fmaf((static_cast<float>(block_sums[row][column]) * 0.25F) *
                       activation_scale,
                   weight_scale, accumulators[row][column]);
        }
      }
    }
    __syncthreads();
  }

  const float tensor_scale = weight_tensor_scale[0] * input_tensor_scale[0];
#pragma unroll
  for (uint32_t row = 0U; row < rows_per_thread; ++row) {
#pragma unroll
    for (uint32_t column = 0U; column < columns_per_thread; ++column) {
      const uint64_t output_row = row_base + local_row + row * thread_rows;
      const uint64_t output_column =
          column_base + local_column + column * thread_columns;
      if (output_row < m && output_column < n) {
        output[output_row * n + output_column] =
            bf16_rne(accumulators[row][column] * tensor_scale);
      }
    }
  }
}

enum class FmaVariant : uint32_t { ProductionId62 = 0U, CandidateFma = 1U };

constexpr std::array<FmaVariant, 2> kFmaVariants = {FmaVariant::ProductionId62,
                                                    FmaVariant::CandidateFma};

const char *fma_variant_name(const FmaVariant variant) {
  switch (variant) {
  case FmaVariant::ProductionId62:
    return "production-id62";
  case FmaVariant::CandidateFma:
    return "candidate-id62-fmaf";
  }
  return "unknown";
}

uint32_t fma_grid(const Shape &shape) {
  const uint64_t rows = (shape.m + 63U) / 64U;
  const uint64_t columns = (shape.n + 63U) / 64U;
  return static_cast<uint32_t>(rows * columns);
}

bool launch_production(const Shape &shape, DeviceBuffers *const buffers) {
  const hipError_t status = sllm_matmul_kernel::launch_nvfp4_w4a4(
      buffers->activation, buffers->activation_scales, buffers->weight,
      buffers->weight_scales, buffers->weight_tensor_scale,
      buffers->input_tensor_scale, buffers->output, shape.m, shape.k, shape.n,
      sllm_matmul_kernel::KernelVariant::Nvfp4W4A4PrefillDp4a64x64,
      buffers->stream);
  return hip_ok(status, "production ID62 launch");
}

bool launch_candidate(const Shape &shape, DeviceBuffers *const buffers) {
  hipLaunchKernelGGL(
      fma_candidate_id62_kernel, dim3(fma_grid(shape)), dim3(kThreads), 0U,
      buffers->stream, buffers->activation, buffers->activation_scales,
      buffers->weight, buffers->weight_scales, buffers->weight_tensor_scale,
      buffers->input_tensor_scale, buffers->output, shape.m, shape.k, shape.n);
  return hip_ok(hipGetLastError(), "candidate FMA launch");
}

bool launch_variant(const FmaVariant variant, const Shape &shape,
                    DeviceBuffers *const buffers) {
  return variant == FmaVariant::ProductionId62
             ? launch_production(shape, buffers)
             : launch_candidate(shape, buffers);
}

bool make_fma_buffers(const Shape &shape, DeviceBuffers *const buffers) {
  if (buffers == nullptr || shape.m == 0U || shape.n == 0U || shape.k == 0U ||
      (shape.k % 16U) != 0U || shape.m > SIZE_MAX / shape.k ||
      shape.n > SIZE_MAX / shape.k || shape.m > SIZE_MAX / shape.n) {
    return false;
  }
  const size_t activation_bytes = static_cast<size_t>(shape.m * shape.k / 2U);
  const size_t activation_scale_bytes =
      static_cast<size_t>(shape.m * (shape.k / 16U));
  const size_t weight_bytes = static_cast<size_t>(shape.n * shape.k / 2U);
  const size_t weight_scale_bytes =
      static_cast<size_t>(shape.n * (shape.k / 16U));
  const size_t output_bytes =
      static_cast<size_t>(shape.m * shape.n) * sizeof(uint16_t);
  return hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->activation),
                          activation_bytes),
                "FMA malloc activation") &&
         hip_ok(
             hipMalloc(reinterpret_cast<void **>(&buffers->activation_scales),
                       activation_scale_bytes),
             "FMA malloc activation scales") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->weight),
                          weight_bytes),
                "FMA malloc weight") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->weight_scales),
                          weight_scale_bytes),
                "FMA malloc weight scales") &&
         hip_ok(
             hipMalloc(reinterpret_cast<void **>(&buffers->weight_tensor_scale),
                       sizeof(float)),
             "FMA malloc weight tensor scale") &&
         hip_ok(
             hipMalloc(reinterpret_cast<void **>(&buffers->input_tensor_scale),
                       sizeof(float)),
             "FMA malloc input tensor scale") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->output),
                          output_bytes),
                "FMA malloc output") &&
         hip_ok(hipStreamCreate(&buffers->stream), "FMA stream create") &&
         hip_ok(hipEventCreate(&buffers->start), "FMA event start") &&
         hip_ok(hipEventCreate(&buffers->stop), "FMA event stop");
}

bool upload_fma_inputs(const Shape &shape, const HostInputs &inputs,
                       DeviceBuffers *const buffers) {
  const float weight_tensor_scale = kWeightTensorScale;
  const float input_tensor_scale = kInputTensorScale;
  return hip_ok(hipMemcpy(buffers->activation, inputs.activation.data(),
                          inputs.activation.size(), hipMemcpyHostToDevice),
                "FMA upload activation") &&
         hip_ok(hipMemcpy(
                    buffers->activation_scales, inputs.activation_scales.data(),
                    inputs.activation_scales.size(), hipMemcpyHostToDevice),
                "FMA upload activation scales") &&
         hip_ok(hipMemcpy(buffers->weight, inputs.weight.data(),
                          inputs.weight.size(), hipMemcpyHostToDevice),
                "FMA upload weight") &&
         hip_ok(hipMemcpy(buffers->weight_scales, inputs.weight_scales.data(),
                          inputs.weight_scales.size(), hipMemcpyHostToDevice),
                "FMA upload weight scales") &&
         hip_ok(hipMemcpy(buffers->weight_tensor_scale, &weight_tensor_scale,
                          sizeof(float), hipMemcpyHostToDevice),
                "FMA upload weight tensor scale") &&
         hip_ok(hipMemcpy(buffers->input_tensor_scale, &input_tensor_scale,
                          sizeof(float), hipMemcpyHostToDevice),
                "FMA upload input tensor scale");
}

bool common_prewarm(const Shape &shape, DeviceBuffers *const buffers) {
  for (uint32_t iteration = 0U; iteration < kWarmups; ++iteration) {
    const FmaVariant first = (iteration & 1U) == 0U ? FmaVariant::ProductionId62
                                                    : FmaVariant::CandidateFma;
    const FmaVariant second = first == FmaVariant::ProductionId62
                                  ? FmaVariant::CandidateFma
                                  : FmaVariant::ProductionId62;
    if (!launch_variant(first, shape, buffers) ||
        !launch_variant(second, shape, buffers) ||
        !hip_ok(hipStreamSynchronize(buffers->stream), "FMA prewarm")) {
      return false;
    }
  }
  return true;
}

bool measure_variant(const FmaVariant variant, const Shape &shape,
                     DeviceBuffers *const buffers, Measurement *const result) {
  for (uint32_t iteration = 0U; iteration < kWarmups; ++iteration) {
    if (!launch_variant(variant, shape, buffers) ||
        !hip_ok(hipStreamSynchronize(buffers->stream), "FMA warmup")) {
      return false;
    }
  }
  for (uint32_t iteration = 0U; iteration < kMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(buffers->start, buffers->stream),
                "FMA event start") ||
        !launch_variant(variant, shape, buffers) ||
        !hip_ok(hipEventRecord(buffers->stop, buffers->stream),
                "FMA event stop") ||
        !hip_ok(hipEventSynchronize(buffers->stop), "FMA event sync") ||
        !hip_ok(hipEventElapsedTime(&result->samples_us[iteration],
                                    buffers->start, buffers->stop),
                "FMA event elapsed")) {
      return false;
    }
    result->samples_us[iteration] *= 1000.0F;
  }
  std::array<float, kMeasured> sorted = result->samples_us;
  std::sort(sorted.begin(), sorted.end());
  result->minimum_us = sorted.front();
  result->median_us = sorted[kMeasured / 2U];
  result->maximum_us = sorted.back();
  result->output.resize(static_cast<size_t>(shape.m * shape.n));
  if (!hip_ok(hipMemcpy(result->output.data(), buffers->output,
                        result->output.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              "FMA output copy")) {
    return false;
  }
  std::vector<uint16_t> repeat(result->output.size());
  if (!launch_variant(variant, shape, buffers) ||
      !hip_ok(hipStreamSynchronize(buffers->stream), "FMA repeat sync") ||
      !hip_ok(hipMemcpy(repeat.data(), buffers->output,
                        repeat.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              "FMA repeat copy")) {
    return false;
  }
  for (size_t index = 0U; index < repeat.size(); ++index) {
    result->repeat_mismatches +=
        static_cast<size_t>(repeat[index] != result->output[index]);
  }
  result->ran = true;
  result->deterministic = result->repeat_mismatches == 0U;
  std::printf(
      "timing variant=%s shape=%s m=%llu k=%llu n=%llu warmups=%u measured=%u "
      "median_us=%.3f min_us=%.3f max_us=%.3f repeat_bf16_mismatches=%zu "
      "deterministic=%s\n",
      fma_variant_name(variant), shape.name,
      static_cast<unsigned long long>(shape.m),
      static_cast<unsigned long long>(shape.k),
      static_cast<unsigned long long>(shape.n), kWarmups, kMeasured,
      result->median_us, result->minimum_us, result->maximum_us,
      result->repeat_mismatches, result->deterministic ? "PASS" : "FAIL");
  return result->deterministic;
}

std::array<OraclePoint, kOracleSamples> f32_oracle(const Shape &shape,
                                                   const HostInputs &inputs) {
  std::array<OraclePoint, kOracleSamples> result{};
  const size_t elements = static_cast<size_t>(shape.m * shape.n);
  const uint64_t blocks = shape.k / 16U;
  for (uint32_t sample = 0U; sample < kOracleSamples; ++sample) {
    const size_t output_index =
        sample * (elements - 1U) / (kOracleSamples - 1U);
    const uint64_t row = output_index / shape.n;
    const uint64_t column = output_index % shape.n;
    float accumulator = 0.0F;
    float absolute_sum = 0.0F;
    for (uint64_t base = 0U; base < shape.k; base += 32U) {
      for (uint64_t local_block = 0U; local_block < 2U; ++local_block) {
        const uint64_t block_begin = base + local_block * 16U;
        if (block_begin >= shape.k)
          continue;
        int32_t block_sum = 0;
        const uint64_t block_end = std::min(shape.k, block_begin + 16U);
        for (uint64_t inner = block_begin; inner < block_end; ++inner) {
          const uint8_t a_pair = inputs.activation[static_cast<size_t>(
              row * shape.k / 2U + inner / 2U)];
          const uint8_t w_pair = inputs.weight[static_cast<size_t>(
              column * shape.k / 2U + inner / 2U)];
          const uint8_t a_code =
              (inner & 1U) == 0U ? a_pair & 0x0fU : a_pair >> 4U;
          const uint8_t w_code =
              (inner & 1U) == 0U ? w_pair & 0x0fU : w_pair >> 4U;
          block_sum += static_cast<int32_t>(host_e2m1(a_code) * 2.0F) *
                       static_cast<int32_t>(host_e2m1(w_code) * 2.0F);
        }
        const uint64_t block = block_begin / 16U;
        const float activation_scale = e4m3fn_to_float(
            inputs
                .activation_scales[static_cast<size_t>(row * blocks + block)]);
        const float weight_scale = e4m3fn_to_float(
            inputs.weight_scales[static_cast<size_t>(column * blocks + block)]);
        const float term = static_cast<float>(block_sum) * 0.25F *
                           activation_scale * weight_scale;
        accumulator += term;
        absolute_sum += std::fabs(term);
      }
    }
    accumulator *= kWeightTensorScale * kInputTensorScale;
    absolute_sum *= kWeightTensorScale * kInputTensorScale;
    result[sample] = OraclePoint{output_index, static_cast<double>(accumulator),
                                 static_cast<double>(absolute_sum),
                                 host_bf16_rne(accumulator)};
  }
  return result;
}

bool check_f32_oracle(const Shape &shape, const FmaVariant variant,
                      const std::array<OraclePoint, kOracleSamples> &oracle,
                      const Measurement &measurement) {
  size_t mismatches = 0U;
  uint32_t max_ulp = 0U;
  double max_abs = 0.0;
  double max_normalized = 0.0;
  for (const OraclePoint &point : oracle) {
    const uint16_t observed_bits = measurement.output[point.index];
    const double observed = host_bf16_to_float(observed_bits);
    const double absolute = std::fabs(observed - point.expected);
    mismatches += static_cast<size_t>(observed_bits != point.expected_bf16);
    max_abs = std::max(max_abs, absolute);
    max_normalized =
        std::max(max_normalized,
                 absolute / std::max(point.absolute_sum,
                                     std::numeric_limits<double>::min()));
    const uint32_t lhs = ordered_bf16(observed_bits);
    const uint32_t rhs = ordered_bf16(point.expected_bf16);
    max_ulp = std::max(max_ulp, lhs > rhs ? lhs - rhs : rhs - lhs);
  }
  std::printf("oracle variant=%s shape=%s samples=%u bf16_mismatches=%zu "
              "max_bf16_ulp=%u max_abs=%.9e max_normalized_error=%.9e "
              "repeat_bf16_mismatches=%zu status=%s\n",
              fma_variant_name(variant), shape.name, kOracleSamples, mismatches,
              max_ulp, max_abs, max_normalized, measurement.repeat_mismatches,
              (mismatches == 0U && measurement.deterministic) ? "PASS"
                                                              : "FAIL");
  return mismatches == 0U && measurement.deterministic;
}

bool check_resources() {
  hipFuncAttributes attributes{};
  if (!hip_ok(hipFuncGetAttributes(&attributes, reinterpret_cast<const void *>(
                                                    fma_candidate_id62_kernel)),
              "FMA candidate attributes")) {
    return false;
  }
  int active_blocks = 0;
  if (!hip_ok(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                  &active_blocks,
                  reinterpret_cast<const void *>(fma_candidate_id62_kernel),
                  kThreads, 0U),
              "FMA candidate occupancy")) {
    return false;
  }
  std::printf(
      "resources candidate=%s registers=%d lds=%zu scratch=%zu "
      "active_blocks_per_cu=%d expected_id62_static_lds=6144 status=%s\n",
      fma_variant_name(FmaVariant::CandidateFma), attributes.numRegs,
      attributes.sharedSizeBytes, attributes.localSizeBytes, active_blocks,
      attributes.sharedSizeBytes == 6144U && attributes.localSizeBytes == 0U &&
              active_blocks > 0
          ? "PASS"
          : "FAIL");
  return attributes.sharedSizeBytes == 6144U &&
         attributes.localSizeBytes == 0U && active_blocks > 0;
}

bool run_shape(const Shape &shape) {
  const HostInputs inputs = make_inputs(shape);
  DeviceBuffers buffers;
  if (!make_fma_buffers(shape, &buffers) ||
      !upload_fma_inputs(shape, inputs, &buffers)) {
    cleanup(&buffers);
    return false;
  }
  bool ok = common_prewarm(shape, &buffers);
  std::array<Measurement, 2> measurements{};
  const size_t first = (shape.m + shape.k + shape.n) & 1U;
  const FmaVariant order[] = {
      first == 0U ? FmaVariant::ProductionId62 : FmaVariant::CandidateFma,
      first == 0U ? FmaVariant::CandidateFma : FmaVariant::ProductionId62};
  for (const FmaVariant variant : order) {
    Measurement &measurement = measurements[static_cast<size_t>(variant)];
    ok = measure_variant(variant, shape, &buffers, &measurement) && ok;
  }
  if (ok) {
    const auto oracle = f32_oracle(shape, inputs);
    for (const FmaVariant variant : kFmaVariants) {
      ok = check_f32_oracle(shape, variant, oracle,
                            measurements[static_cast<size_t>(variant)]) &&
           ok;
    }
    const Comparison comparison =
        compare(measurements[0].output, measurements[1].output);
    print_comparison(shape, "production-id62-vs-candidate-id62-fmaf",
                     comparison);
    ok = comparison.mismatches == 0U && ok;
  }
  cleanup(&buffers);
  return ok;
}

} // namespace

int main(int argc, char **argv) {
  int device = 0;
  if (argc > 2 || (argc == 2 && !parse_device(argv[1], &device))) {
    std::fprintf(stderr, "usage: phase78_nvfp4_prefill_fma_probe [DEVICE]\n");
    return EXIT_FAILURE;
  }
  if (!hip_ok(hipSetDevice(device), "FMA hipSetDevice"))
    return EXIT_FAILURE;
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device),
              "FMA hipGetDeviceProperties")) {
    return EXIT_FAILURE;
  }
  if (!exact_gfx1030(properties.gcnArchName)) {
    std::fprintf(stderr, "exact gfx1030 required; got %s\n",
                 properties.gcnArchName);
    return EXIT_FAILURE;
  }
  std::printf("identity target=%s device=%d pci=%04x:%02x:%02x name=%s "
              "oracle=independent_fp32_block16_v1 contexts=tiny+wide/down "
              "warmups=%u measured=%u classification=N0_prototype\n",
              properties.gcnArchName, device, properties.pciDomainID,
              properties.pciBusID, properties.pciDeviceID, properties.name,
              kWarmups, kMeasured);
  bool ok = check_resources();
  const std::array<Shape, 5> shapes = {{
      {17U, 48U, 37U, 1U, "tiny-m17-k48-n37"},
      {128U, 5120U, 17408U, 112U, "wide-m128"},
      {1024U, 5120U, 17408U, 112U, "wide-m1024"},
      {128U, 17408U, 5120U, 56U, "down-m128"},
      {1024U, 17408U, 5120U, 56U, "down-m1024"},
  }};
  for (const Shape &shape : shapes)
    ok = run_shape(shape) && ok;
  std::printf("summary status=%s shapes=%zu warmups=%u measured=%u "
              "decision=N0_no_production_routing\n",
              ok ? "PASS" : "FAIL", shapes.size(), kWarmups, kMeasured);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
