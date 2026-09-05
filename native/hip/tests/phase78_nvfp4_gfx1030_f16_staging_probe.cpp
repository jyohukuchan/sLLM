// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase78-nvfp4-byte-permute-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// 3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70, ggml/src/ggml-cuda/vecdotq.cuh
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

// Phase 78 standalone gfx1030 NVFP4 W4A4 FP16-staging probe.
//
// ID62 is reproduced locally as the DP4A 64x64 prefill control.  The
// candidate expands each resident block16 NVFP4 row/column to FP16, runs
// rocBLAS F16/F16->F32, then applies tensor scales and BF16 RNE.  This file is
// evidence-only; no production selector or existing file is modified.

#include "low_precision_block_codec.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

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

extern "C" rocblas_status rocblas_gemm_ex_get_solutions(
    rocblas_handle handle, rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k, const void *alpha,
    const void *a, rocblas_datatype a_type, rocblas_int lda, const void *b,
    rocblas_datatype b_type, rocblas_int ldb, const void *beta, const void *c,
    rocblas_datatype c_type, rocblas_int ldc, void *d, rocblas_datatype d_type,
    rocblas_int ldd, rocblas_datatype compute_type, rocblas_gemm_algo algorithm,
    uint32_t flags, rocblas_int *solutions, rocblas_int *solution_count);

namespace {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kWaveWidth = 32U;
constexpr uint32_t kWaves = kThreads / kWaveWidth;
constexpr uint32_t kWarmups = 3U;
constexpr uint32_t kMeasured = 10U;

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
    if (mantissa == 0U)
      return __uint_as_float(sign);
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

// Exact ID62 DP4A 64x64 prefill control, copied as a standalone kernel.
__global__ __launch_bounds__(kThreads, 1) void id62_dp4a64x64_kernel(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_scales, const uint8_t *const packed_weight,
    const uint8_t *const weight_scales, const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  constexpr uint32_t tile_m = 64U;
  constexpr uint32_t tile_n = 64U;
  constexpr uint32_t tile_k = 32U;
  constexpr uint32_t block_k = 16U;
  constexpr uint32_t blocks_per_stage = tile_k / block_k;
  constexpr uint32_t packed_chunks_per_stage = tile_k / 8U;
  constexpr uint32_t lds_group_stride = tile_k / 4U + 1U;
  constexpr uint32_t lds_scale_stride = blocks_per_stage + 1U;
  constexpr uint32_t rows_per_thread = 4U;
  constexpr uint32_t columns_per_thread = 4U;
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
  const uint32_t local_column = thread & 15U;
  const uint64_t packed_row_bytes = k / 2U;
  const uint64_t blocks_per_row = k / block_k;
  float accumulators[rows_per_thread][columns_per_thread] = {};

  for (uint64_t base = 0U; base < k; base += tile_k) {
    for (uint32_t index = thread; index < tile_m * packed_chunks_per_stage;
         index += kThreads) {
      const uint32_t row = index / packed_chunks_per_stage;
      const uint32_t chunk = index % packed_chunks_per_stage;
      const uint64_t source_row = row_base + row;
      const uint64_t inner = base + static_cast<uint64_t>(chunk) * 8U;
      const ScaledPacks values =
          source_row < m && inner + 8U <= k
              ? scaled_packs(__builtin_nontemporal_load(
                    reinterpret_cast<const uint32_t *>(
                        packed_activation + source_row * packed_row_bytes +
                        inner / 2U)))
              : ScaledPacks{0U, 0U};
      activation_tile[row][chunk * 2U] = values.even;
      activation_tile[row][chunk * 2U + 1U] = values.odd;
    }
    for (uint32_t index = thread; index < tile_n * packed_chunks_per_stage;
         index += kThreads) {
      const uint32_t column = index / packed_chunks_per_stage;
      const uint32_t chunk = index % packed_chunks_per_stage;
      const uint64_t source_column = column_base + column;
      const uint64_t inner = base + static_cast<uint64_t>(chunk) * 8U;
      const ScaledPacks values =
          source_column < n && inner + 8U <= k
              ? scaled_packs(__builtin_nontemporal_load(
                    reinterpret_cast<const uint32_t *>(
                        packed_weight + source_column * packed_row_bytes +
                        inner / 2U)))
              : ScaledPacks{0U, 0U};
      weight_tile[column][chunk * 2U] = values.even;
      weight_tile[column][chunk * 2U + 1U] = values.odd;
    }
    for (uint32_t index = thread; index < tile_m * blocks_per_stage;
         index += kThreads) {
      const uint32_t row = index / blocks_per_stage;
      const uint32_t block = index % blocks_per_stage;
      const uint64_t source_row = row_base + row;
      const uint64_t source_block = base / block_k + block;
      activation_scale_tile[row][block] =
          source_row < m && source_block < blocks_per_row
              ? e4m3(__builtin_nontemporal_load(activation_scales +
                                                source_row * blocks_per_row +
                                                source_block))
              : 0.0F;
    }
    for (uint32_t index = thread; index < tile_n * blocks_per_stage;
         index += kThreads) {
      const uint32_t column = index / blocks_per_stage;
      const uint32_t block = index % blocks_per_stage;
      const uint64_t source_column = column_base + column;
      const uint64_t source_block = base / block_k + block;
      weight_scale_tile[column][block] =
          source_column < n && source_block < blocks_per_row
              ? e4m3(__builtin_nontemporal_load(weight_scales +
                                                source_column * blocks_per_row +
                                                source_block))
              : 0.0F;
    }
    __syncthreads();

#pragma unroll
    for (uint32_t block = 0U; block < blocks_per_stage; ++block) {
      if (base + static_cast<uint64_t>(block) * block_k >= k)
        continue;
      int32_t block_sums[rows_per_thread][columns_per_thread] = {};
#pragma unroll
      for (uint32_t group = 0U; group < block_k / 4U; ++group) {
        int32_t activation_packs[rows_per_thread];
        int32_t weight_packs[columns_per_thread];
#pragma unroll
        for (uint32_t row = 0U; row < rows_per_thread; ++row)
          activation_packs[row] =
              activation_tile[local_row + row * 16U]
                             [block * (block_k / 4U) + group];
#pragma unroll
        for (uint32_t column = 0U; column < columns_per_thread; ++column)
          weight_packs[column] = weight_tile[local_column + column * 16U]
                                            [block * (block_k / 4U) + group];
#pragma unroll
        for (uint32_t row = 0U; row < rows_per_thread; ++row)
#pragma unroll
          for (uint32_t column = 0U; column < columns_per_thread; ++column)
            block_sums[row][column] =
                dot4(activation_packs[row], weight_packs[column],
                     block_sums[row][column]);
      }
#pragma unroll
      for (uint32_t row = 0U; row < rows_per_thread; ++row) {
        const float activation_scale =
            activation_scale_tile[local_row + row * 16U][block];
#pragma unroll
        for (uint32_t column = 0U; column < columns_per_thread; ++column) {
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
  for (uint32_t row = 0U; row < rows_per_thread; ++row)
#pragma unroll
    for (uint32_t column = 0U; column < columns_per_thread; ++column) {
      const uint64_t output_row = row_base + local_row + row * 16U;
      const uint64_t output_column = column_base + local_column + column * 16U;
      if (output_row < m && output_column < n)
        output[output_row * n + output_column] =
            bf16_rne(accumulators[row][column] * tensor_scale);
    }
}

// One thread owns one block16: one 64-bit packed load and one scale load.
__global__ __launch_bounds__(kThreads, 1) void nvfp4_to_fp16_block16_kernel(
    const uint8_t *const packed, const uint8_t *const scales,
    uint16_t *const output, const uint64_t rows, const uint64_t k) {
  const uint64_t blocks_per_row = k / 16U;
  const uint64_t block_index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block_index >= rows * blocks_per_row)
    return;
  const uint64_t row = block_index / blocks_per_row;
  const uint64_t block = block_index - row * blocks_per_row;
  const uint64_t packed_offset = row * (k / 2U) + block * 8U;
  const uint64_t packed_values = __builtin_nontemporal_load(
      reinterpret_cast<const uint64_t *>(packed + packed_offset));
  const float scale = e4m3(__builtin_nontemporal_load(scales + block_index));
  const uint64_t output_offset = row * k + block * 16U;
#pragma unroll
  for (uint32_t index = 0U; index < 16U; ++index) {
    const uint8_t pair =
        static_cast<uint8_t>(packed_values >> ((index / 2U) * 8U));
    const uint8_t code = (index & 1U) == 0U ? pair & 0x0fU : pair >> 4U;
    const __half value = __float2half_rn(e2m1(code) * scale);
    output[output_offset + index] = static_cast<__half_raw>(value).x;
  }
}

__global__ __launch_bounds__(kThreads, 1) void bf16_epilogue_kernel(
    const float *const input, uint16_t *const output, const uint64_t elements,
    const float tensor_scale) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements)
    output[index] = bf16_rne(input[index] * tensor_scale);
}

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "hip error operation=%s status=%s\n", operation,
               hipGetErrorString(status));
  return false;
}

bool rocblas_ok(const rocblas_status status, const char *const operation) {
  if (status == rocblas_status_success)
    return true;
  std::fprintf(stderr, "rocblas error operation=%s status=%d\n", operation,
               static_cast<int>(status));
  return false;
}

bool exact_gfx1030(const char *const arch) {
  return arch != nullptr &&
         std::string_view(arch).compare(0U, 7U, "gfx1030") == 0;
}

float host_e2m1(const uint8_t code) {
  constexpr float positive[8] = {0.0F, 0.5F, 1.0F, 1.5F,
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
    if (mantissa == 0U)
      return 0.0F;
    float value = static_cast<float>(mantissa) * 0x1p-9F;
    uint32_t value_bits = 0U;
    std::memcpy(&value_bits, &value, sizeof(value_bits));
    value_bits |= sign;
    std::memcpy(&value, &value_bits, sizeof(value));
    return value;
  }
  if (magnitude == 0x7fU)
    return std::numeric_limits<float>::quiet_NaN();
  const uint32_t bits32 = sign | ((exponent + 120U) << 23U) | (mantissa << 20U);
  float value = 0.0F;
  std::memcpy(&value, &bits32, sizeof(value));
  return value;
}

uint16_t host_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000U) == 0x7f800000U)
    return static_cast<uint16_t>(bits >> 16U);
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

struct Buffers final {
  uint8_t *activation = nullptr;
  uint8_t *activation_scales = nullptr;
  uint8_t *weight = nullptr;
  uint8_t *weight_scales = nullptr;
  uint16_t *activation_fp16 = nullptr;
  uint16_t *weight_fp16 = nullptr;
  float *gemm_output = nullptr;
  uint16_t *output = nullptr;
  uint16_t *direct_output = nullptr;
  float *weight_tensor_scale = nullptr;
  float *input_tensor_scale = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  rocblas_handle rocblas = nullptr;
};

bool cleanup(Buffers *const b) {
  if (b == nullptr)
    return true;
  bool ok = true;
  if (b->rocblas != nullptr)
    ok = rocblas_ok(rocblas_destroy_handle(b->rocblas),
                    "rocblas_destroy_handle") &&
         ok;
  if (b->stop != nullptr)
    ok = hip_ok(hipEventDestroy(b->stop), "hipEventDestroy stop") && ok;
  if (b->start != nullptr)
    ok = hip_ok(hipEventDestroy(b->start), "hipEventDestroy start") && ok;
  if (b->stream != nullptr)
    ok = hip_ok(hipStreamDestroy(b->stream), "hipStreamDestroy") && ok;
  if (b->output != nullptr)
    ok = hip_ok(hipFree(b->output), "hipFree output") && ok;
  if (b->direct_output != nullptr)
    ok = hip_ok(hipFree(b->direct_output), "hipFree direct output") && ok;
  if (b->gemm_output != nullptr)
    ok = hip_ok(hipFree(b->gemm_output), "hipFree gemm output") && ok;
  if (b->weight_fp16 != nullptr)
    ok = hip_ok(hipFree(b->weight_fp16), "hipFree weight fp16") && ok;
  if (b->activation_fp16 != nullptr)
    ok = hip_ok(hipFree(b->activation_fp16), "hipFree activation fp16") && ok;
  if (b->input_tensor_scale != nullptr)
    ok = hip_ok(hipFree(b->input_tensor_scale), "hipFree input scale") && ok;
  if (b->weight_tensor_scale != nullptr)
    ok = hip_ok(hipFree(b->weight_tensor_scale), "hipFree weight scale") && ok;
  if (b->weight_scales != nullptr)
    ok = hip_ok(hipFree(b->weight_scales), "hipFree weight scales") && ok;
  if (b->weight != nullptr)
    ok = hip_ok(hipFree(b->weight), "hipFree weight") && ok;
  if (b->activation_scales != nullptr)
    ok = hip_ok(hipFree(b->activation_scales), "hipFree activation scales") &&
         ok;
  if (b->activation != nullptr)
    ok = hip_ok(hipFree(b->activation), "hipFree activation") && ok;
  *b = {};
  return ok;
}

bool make_buffers(const uint64_t m, const uint64_t k, const uint64_t n,
                  Buffers *const b) {
  if (b == nullptr || m == 0U || k == 0U || n == 0U || (k % 16U) != 0U ||
      m > SIZE_MAX / k || n > SIZE_MAX / k)
    return false;
  const uint64_t packed_a = m * k / 2U, packed_w = n * k / 2U;
  const uint64_t scales_a = m * (k / 16U), scales_w = n * (k / 16U);
  const uint64_t f16_a = m * k * sizeof(uint16_t),
                 f16_w = n * k * sizeof(uint16_t);
  const uint64_t f32_c = m * n * sizeof(float),
                 bf16_c = m * n * sizeof(uint16_t);
  return hip_ok(hipMalloc(reinterpret_cast<void **>(&b->activation), packed_a),
                "hipMalloc activation") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->activation_scales),
                          scales_a),
                "hipMalloc activation scales") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->weight), packed_w),
                "hipMalloc weight") &&
         hip_ok(
             hipMalloc(reinterpret_cast<void **>(&b->weight_scales), scales_w),
             "hipMalloc weight scales") &&
         hip_ok(
             hipMalloc(reinterpret_cast<void **>(&b->activation_fp16), f16_a),
             "hipMalloc activation fp16") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->weight_fp16), f16_w),
                "hipMalloc weight fp16") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->gemm_output), f32_c),
                "hipMalloc gemm output") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->output), bf16_c),
                "hipMalloc output") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->direct_output), bf16_c),
                "hipMalloc direct output") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->weight_tensor_scale),
                          sizeof(float)),
                "hipMalloc weight tensor scale") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->input_tensor_scale),
                          sizeof(float)),
                "hipMalloc input tensor scale") &&
         hip_ok(hipStreamCreate(&b->stream), "hipStreamCreate") &&
         hip_ok(hipEventCreate(&b->start), "hipEventCreate start") &&
         hip_ok(hipEventCreate(&b->stop), "hipEventCreate stop") &&
         rocblas_ok(rocblas_create_handle(&b->rocblas),
                    "rocblas_create_handle") &&
         rocblas_ok(
             rocblas_set_pointer_mode(b->rocblas, rocblas_pointer_mode_host),
             "rocblas_set_pointer_mode") &&
         rocblas_ok(rocblas_set_stream(b->rocblas, b->stream),
                    "rocblas_set_stream");
}

void fill_inputs(const uint64_t m, const uint64_t k, const uint64_t n,
                 std::vector<uint8_t> *const activation,
                 std::vector<uint8_t> *const activation_scales,
                 std::vector<uint8_t> *const weight,
                 std::vector<uint8_t> *const weight_scales) {
  const uint64_t blocks = k / 16U;
  activation->assign(static_cast<size_t>(m * k / 2U), 0U);
  activation_scales->assign(static_cast<size_t>(m * blocks), 0x38U);
  weight->assign(static_cast<size_t>(n * k / 2U), 0U);
  weight_scales->assign(static_cast<size_t>(n * blocks), 0x38U);
  for (uint64_t row = 0U; row < m; ++row) {
    for (uint64_t inner = 0U; inner < k; ++inner) {
      const uint8_t code =
          static_cast<uint8_t>((row * 5U + inner * 3U + 1U) & 0x0fU);
      const size_t index = static_cast<size_t>(row * k / 2U + inner / 2U);
      if ((inner & 1U) == 0U)
        (*activation)[index] = code;
      else
        (*activation)[index] |= static_cast<uint8_t>(code << 4U);
    }
    for (uint64_t block = 0U; block < blocks; ++block)
      (*activation_scales)[static_cast<size_t>(row * blocks + block)] =
          (block & 1U) == 0U ? 0x38U : 0x40U;
  }
  for (uint64_t column = 0U; column < n; ++column) {
    for (uint64_t inner = 0U; inner < k; ++inner) {
      const uint8_t code =
          static_cast<uint8_t>((column * 7U + inner * 9U + 2U) & 0x0fU);
      const size_t index = static_cast<size_t>(column * k / 2U + inner / 2U);
      if ((inner & 1U) == 0U)
        (*weight)[index] = code;
      else
        (*weight)[index] |= static_cast<uint8_t>(code << 4U);
    }
    for (uint64_t block = 0U; block < blocks; ++block)
      (*weight_scales)[static_cast<size_t>(column * blocks + block)] =
          (block & 1U) == 0U ? 0x38U : 0x40U;
  }
}

bool upload(const uint64_t m, const uint64_t k, const uint64_t n,
            const std::vector<uint8_t> &activation,
            const std::vector<uint8_t> &activation_scales,
            const std::vector<uint8_t> &weight,
            const std::vector<uint8_t> &weight_scales, Buffers *const b) {
  const float weight_tensor_scale = 0.75F, input_tensor_scale = 1.125F;
  const uint64_t blocks = k / 16U;
  return hip_ok(hipMemcpy(b->activation, activation.data(), m * k / 2U,
                          hipMemcpyHostToDevice),
                "upload activation") &&
         hip_ok(hipMemcpy(b->activation_scales, activation_scales.data(),
                          m * blocks, hipMemcpyHostToDevice),
                "upload activation scales") &&
         hip_ok(hipMemcpy(b->weight, weight.data(), n * k / 2U,
                          hipMemcpyHostToDevice),
                "upload weight") &&
         hip_ok(hipMemcpy(b->weight_scales, weight_scales.data(), n * blocks,
                          hipMemcpyHostToDevice),
                "upload weight scales") &&
         hip_ok(hipMemcpy(b->weight_tensor_scale, &weight_tensor_scale,
                          sizeof(float), hipMemcpyHostToDevice),
                "upload weight tensor scale") &&
         hip_ok(hipMemcpy(b->input_tensor_scale, &input_tensor_scale,
                          sizeof(float), hipMemcpyHostToDevice),
                "upload input tensor scale") &&
         hip_ok(hipMemset(b->output, 0, m * n * sizeof(uint16_t)),
                "clear output");
}

bool launch_control(const uint64_t m, const uint64_t k, const uint64_t n,
                    Buffers *const b) {
  const uint64_t tiles_x = (n + 63U) / 64U;
  hipLaunchKernelGGL(id62_dp4a64x64_kernel,
                     dim3(static_cast<uint32_t>(((m + 63U) / 64U) * tiles_x)),
                     dim3(kThreads), 0U, b->stream, b->activation,
                     b->activation_scales, b->weight, b->weight_scales,
                     b->weight_tensor_scale, b->input_tensor_scale, b->output,
                     m, k, n);
  return hipGetLastError() == hipSuccess;
}

bool launch_ingress(const uint64_t m, const uint64_t k, const uint64_t n,
                    Buffers *const b) {
  const uint64_t a_blocks = m * (k / 16U), w_blocks = n * (k / 16U);
  hipLaunchKernelGGL(
      nvfp4_to_fp16_block16_kernel,
      dim3(static_cast<uint32_t>((a_blocks + kThreads - 1U) / kThreads)),
      dim3(kThreads), 0U, b->stream, b->activation, b->activation_scales,
      b->activation_fp16, m, k);
  hipLaunchKernelGGL(
      nvfp4_to_fp16_block16_kernel,
      dim3(static_cast<uint32_t>((w_blocks + kThreads - 1U) / kThreads)),
      dim3(kThreads), 0U, b->stream, b->weight, b->weight_scales,
      b->weight_fp16, n, k);
  return hipGetLastError() == hipSuccess;
}

bool launch_staging(const uint64_t m, const uint64_t k, const uint64_t n,
                    const int32_t solution, Buffers *const b) {
  if (!launch_ingress(m, k, n, b))
    return false;
  const float alpha = 1.0F, beta = 0.0F;
  const rocblas_gemm_algo algorithm = solution == 0
                                          ? rocblas_gemm_algo_standard
                                          : rocblas_gemm_algo_solution_index;
  const int32_t algorithm_solution = solution == 0 ? 0 : solution;
  if (!rocblas_ok(
          (rocblas_gemm_ex)(b->rocblas, rocblas_operation_transpose,
                            rocblas_operation_none, static_cast<rocblas_int>(n),
                            static_cast<rocblas_int>(m),
                            static_cast<rocblas_int>(k), &alpha, b->weight_fp16,
                            rocblas_datatype_f16_r, static_cast<rocblas_int>(k),
                            b->activation_fp16, rocblas_datatype_f16_r,
                            static_cast<rocblas_int>(k), &beta, b->gemm_output,
                            rocblas_datatype_f32_r, static_cast<rocblas_int>(n),
                            b->gemm_output, rocblas_datatype_f32_r,
                            static_cast<rocblas_int>(n), rocblas_datatype_f32_r,
                            algorithm, algorithm_solution, 0U),
          "rocblas_gemm_ex"))
    return false;
  const uint64_t elements = m * n;
  hipLaunchKernelGGL(
      bf16_epilogue_kernel,
      dim3(static_cast<uint32_t>((elements + kThreads - 1U) / kThreads)),
      dim3(kThreads), 0U, b->stream, b->gemm_output, b->output, elements,
      0.75F * 1.125F);
  return hipGetLastError() == hipSuccess;
}

// Optional comparison path: let rocBLAS round directly to BF16 while using
// host alpha for the tensor-scale product.  This removes the F32 C workspace;
// numerical acceptance is based on bitwise BF16 comparison with staging.
bool launch_staging_direct_bf16(const uint64_t m, const uint64_t k,
                                const uint64_t n, const int32_t solution,
                                Buffers *const b) {
  if (!launch_ingress(m, k, n, b))
    return false;
  const float alpha = 0.75F * 1.125F, beta = 0.0F;
  const rocblas_gemm_algo algorithm = solution == 0
                                          ? rocblas_gemm_algo_standard
                                          : rocblas_gemm_algo_solution_index;
  const int32_t algorithm_solution = solution == 0 ? 0 : solution;
  if (!rocblas_ok(
          (rocblas_gemm_ex)(b->rocblas, rocblas_operation_transpose,
                            rocblas_operation_none, static_cast<rocblas_int>(n),
                            static_cast<rocblas_int>(m),
                            static_cast<rocblas_int>(k), &alpha, b->weight_fp16,
                            rocblas_datatype_f16_r, static_cast<rocblas_int>(k),
                            b->activation_fp16, rocblas_datatype_f16_r,
                            static_cast<rocblas_int>(k), &beta,
                            b->direct_output, rocblas_datatype_bf16_r,
                            static_cast<rocblas_int>(n), b->output,
                            rocblas_datatype_bf16_r,
                            static_cast<rocblas_int>(n), rocblas_datatype_f32_r,
                            algorithm, algorithm_solution, 0U),
          "rocblas_gemm_ex direct BF16"))
    return false;
  return hipGetLastError() == hipSuccess;
}

bool measure_control(const uint64_t m, const uint64_t k, const uint64_t n,
                     Buffers *const b, float *const median_us) {
  for (uint32_t i = 0U; i < kWarmups; ++i)
    if (!launch_control(m, k, n, b) ||
        !hip_ok(hipStreamSynchronize(b->stream), "control warmup"))
      return false;
  std::array<float, kMeasured> samples{};
  for (uint32_t i = 0U; i < kMeasured; ++i) {
    if (!hip_ok(hipEventRecord(b->start, b->stream), "control event start") ||
        !launch_control(m, k, n, b) ||
        !hip_ok(hipEventRecord(b->stop, b->stream), "control event stop") ||
        !hip_ok(hipEventSynchronize(b->stop), "control event sync") ||
        !hip_ok(hipEventElapsedTime(&samples[i], b->start, b->stop),
                "control elapsed"))
      return false;
    samples[i] *= 1000.0F;
  }
  std::sort(samples.begin(), samples.end());
  *median_us = samples[kMeasured / 2U];
  return true;
}

bool measure_staging(const uint64_t m, const uint64_t k, const uint64_t n,
                     const int32_t solution, Buffers *const b,
                     float *const median_us) {
  for (uint32_t i = 0U; i < kWarmups; ++i)
    if (!launch_staging(m, k, n, solution, b) ||
        !hip_ok(hipStreamSynchronize(b->stream), "staging warmup"))
      return false;
  std::array<float, kMeasured> samples{};
  for (uint32_t i = 0U; i < kMeasured; ++i) {
    if (!hip_ok(hipEventRecord(b->start, b->stream), "staging event start") ||
        !launch_staging(m, k, n, solution, b) ||
        !hip_ok(hipEventRecord(b->stop, b->stream), "staging event stop") ||
        !hip_ok(hipEventSynchronize(b->stop), "staging event sync") ||
        !hip_ok(hipEventElapsedTime(&samples[i], b->start, b->stop),
                "staging elapsed"))
      return false;
    samples[i] *= 1000.0F;
  }
  std::sort(samples.begin(), samples.end());
  *median_us = samples[kMeasured / 2U];
  return true;
}

bool measure_direct_bf16(const uint64_t m, const uint64_t k, const uint64_t n,
                         const int32_t solution, Buffers *const b,
                         float *const median_us) {
  for (uint32_t i = 0U; i < kWarmups; ++i)
    if (!launch_staging_direct_bf16(m, k, n, solution, b) ||
        !hip_ok(hipStreamSynchronize(b->stream), "direct BF16 warmup"))
      return false;
  std::array<float, kMeasured> samples{};
  for (uint32_t i = 0U; i < kMeasured; ++i) {
    if (!hip_ok(hipEventRecord(b->start, b->stream),
                "direct BF16 event start") ||
        !launch_staging_direct_bf16(m, k, n, solution, b) ||
        !hip_ok(hipEventRecord(b->stop, b->stream), "direct BF16 event stop") ||
        !hip_ok(hipEventSynchronize(b->stop), "direct BF16 event sync") ||
        !hip_ok(hipEventElapsedTime(&samples[i], b->start, b->stop),
                "direct BF16 elapsed"))
      return false;
    samples[i] *= 1000.0F;
  }
  std::sort(samples.begin(), samples.end());
  *median_us = samples[kMeasured / 2U];
  return true;
}

bool compare_outputs(const std::vector<uint16_t> &lhs,
                     const std::vector<uint16_t> &rhs, const char *const name,
                     const uint64_t m, const uint64_t n) {
  size_t ulp = 0U, top = 0U;
  double max_abs = 0.0, max_rel = 0.0;
  for (size_t i = 0U; i < lhs.size(); ++i) {
    const float a = host_bf16_to_float(lhs[i]), b = host_bf16_to_float(rhs[i]);
    const double abs = std::abs(static_cast<double>(a) - b);
    max_abs = std::max(max_abs, abs);
    max_rel = std::max(
        max_rel, abs / std::max(1.0e-6, std::abs(static_cast<double>(a))));
    if (lhs[i] != rhs[i]) {
      ++ulp;
      if (std::abs(static_cast<int>(lhs[i]) - static_cast<int>(rhs[i])) > 1)
        ++top;
    }
  }
  std::printf("compare candidate=%s m=%llu n=%llu max_abs=%.8g max_rel=%.8g "
              "bf16_ulp_mismatch=%zu top_mismatch=%zu\n",
              name, static_cast<unsigned long long>(m),
              static_cast<unsigned long long>(n), max_abs, max_rel, ulp, top);
  return ulp == 0U;
}

bool print_solutions(rocblas_handle handle, const uint64_t m, const uint64_t k,
                     const uint64_t n, const Buffers &b) {
  const float alpha = 1.0F, beta = 0.0F;
  rocblas_int count = 0;
  rocblas_status status = rocblas_gemm_ex_get_solutions(
      handle, rocblas_operation_transpose, rocblas_operation_none,
      static_cast<rocblas_int>(n), static_cast<rocblas_int>(m),
      static_cast<rocblas_int>(k), &alpha, b.weight_fp16,
      rocblas_datatype_f16_r, static_cast<rocblas_int>(k), b.activation_fp16,
      rocblas_datatype_f16_r, static_cast<rocblas_int>(k), &beta, b.gemm_output,
      rocblas_datatype_f32_r, static_cast<rocblas_int>(n), b.gemm_output,
      rocblas_datatype_f32_r, static_cast<rocblas_int>(n),
      rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0U, nullptr, &count);
  if (status != rocblas_status_success || count <= 0)
    return false;
  std::vector<rocblas_int> solutions(static_cast<size_t>(count));
  status = rocblas_gemm_ex_get_solutions(
      handle, rocblas_operation_transpose, rocblas_operation_none,
      static_cast<rocblas_int>(n), static_cast<rocblas_int>(m),
      static_cast<rocblas_int>(k), &alpha, b.weight_fp16,
      rocblas_datatype_f16_r, static_cast<rocblas_int>(k), b.activation_fp16,
      rocblas_datatype_f16_r, static_cast<rocblas_int>(k), &beta, b.gemm_output,
      rocblas_datatype_f32_r, static_cast<rocblas_int>(n), b.gemm_output,
      rocblas_datatype_f32_r, static_cast<rocblas_int>(n),
      rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0U, solutions.data(),
      &count);
  if (status != rocblas_status_success)
    return false;
  std::printf("solutions m=%llu k=%llu n=%llu count=%d first=",
              static_cast<unsigned long long>(m),
              static_cast<unsigned long long>(k),
              static_cast<unsigned long long>(n), count);
  for (size_t i = 0U; i < std::min<size_t>(solutions.size(), 16U); ++i)
    std::printf("%s%d", i == 0U ? "" : ",", solutions[i]);
  std::printf("\n");
  return true;
}

void print_resources(const char *const name, const void *const function) {
  hipFuncAttributes attributes{};
  const hipError_t attr = hipFuncGetAttributes(&attributes, function);
  int active_blocks = 0;
  const hipError_t occupancy = hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, function, kThreads, 0U);
  std::printf(
      "resources candidate=%s vgpr=%d sgpr=ISA-metadata lds_static=%zu "
      "scratch=%zu max_threads=%d active_blocks=%d attr=%s occupancy=%s\n",
      name, attributes.numRegs, attributes.sharedSizeBytes,
      attributes.localSizeBytes, attributes.maxThreadsPerBlock, active_blocks,
      hipGetErrorString(attr), hipGetErrorString(occupancy));
}

} // namespace

int main() {
  constexpr int device = 0;
  if (!hip_ok(hipSetDevice(device), "hipSetDevice"))
    return EXIT_FAILURE;
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
  print_resources("id62-dp4a64x64",
                  reinterpret_cast<const void *>(id62_dp4a64x64_kernel));
  print_resources("nvfp4-ingress-block16",
                  reinterpret_cast<const void *>(nvfp4_to_fp16_block16_kernel));
  print_resources("bf16-epilogue",
                  reinterpret_cast<const void *>(bf16_epilogue_kernel));

  std::vector<uint8_t> activation, activation_scales, weight, weight_scales;
  constexpr uint64_t oracle_m = 17U, oracle_k = 32U, oracle_n = 17U;
  fill_inputs(oracle_m, oracle_k, oracle_n, &activation, &activation_scales,
              &weight, &weight_scales);
  Buffers oracle;
  if (!make_buffers(oracle_m, oracle_k, oracle_n, &oracle) ||
      !upload(oracle_m, oracle_k, oracle_n, activation, activation_scales,
              weight, weight_scales, &oracle)) {
    cleanup(&oracle);
    return EXIT_FAILURE;
  }
  std::vector<uint16_t> oracle_control(
      static_cast<size_t>(oracle_m * oracle_n));
  std::vector<uint16_t> oracle_staging(
      static_cast<size_t>(oracle_m * oracle_n));
  std::vector<uint16_t> oracle_direct(static_cast<size_t>(oracle_m * oracle_n));
  bool oracle_runtime =
      launch_control(oracle_m, oracle_k, oracle_n, &oracle) &&
      hip_ok(hipDeviceSynchronize(), "oracle control sync") &&
      hip_ok(hipMemcpy(oracle_control.data(), oracle.output,
                       oracle_control.size() * sizeof(uint16_t),
                       hipMemcpyDeviceToHost),
             "oracle control copy") &&
      launch_staging(oracle_m, oracle_k, oracle_n, 0, &oracle) &&
      hip_ok(hipDeviceSynchronize(), "oracle staging sync") &&
      hip_ok(hipMemcpy(oracle_staging.data(), oracle.output,
                       oracle_staging.size() * sizeof(uint16_t),
                       hipMemcpyDeviceToHost),
             "oracle staging copy");
  bool oracle_ok = oracle_runtime;
  double oracle_max_abs = 0.0, oracle_max_rel = 0.0;
  size_t oracle_ulp = 0U, oracle_top = 0U;
  for (uint64_t row = 0U; oracle_runtime && row < oracle_m; ++row) {
    for (uint64_t column = 0U; column < oracle_n; ++column) {
      float expected = 0.0F;
      for (uint64_t inner = 0U; inner < oracle_k; ++inner) {
        const uint8_t ap =
            activation[static_cast<size_t>(row * oracle_k / 2U + inner / 2U)];
        const uint8_t wp =
            weight[static_cast<size_t>(column * oracle_k / 2U + inner / 2U)];
        const uint8_t ac = (inner & 1U) == 0U ? ap & 0x0fU : ap >> 4U;
        const uint8_t wc = (inner & 1U) == 0U ? wp & 0x0fU : wp >> 4U;
        expected +=
            host_e2m1(ac) *
            host_e4m3(activation_scales[static_cast<size_t>(row * 2U +
                                                            inner / 16U)]) *
            host_e2m1(wc) *
            host_e4m3(
                weight_scales[static_cast<size_t>(column * 2U + inner / 16U)]);
      }
      const uint16_t expected_bits = host_bf16_rne(expected * 0.75F * 1.125F);
      const uint16_t observed =
          oracle_staging[static_cast<size_t>(row * oracle_n + column)];
      const double absolute =
          std::abs(static_cast<double>(host_bf16_to_float(expected_bits)) -
                   host_bf16_to_float(observed));
      oracle_max_abs = std::max(oracle_max_abs, absolute);
      oracle_max_rel = std::max(
          oracle_max_rel,
          absolute / std::max(1.0e-6, std::abs(static_cast<double>(
                                          host_bf16_to_float(expected_bits)))));
      if (expected_bits != observed) {
        ++oracle_ulp;
        if (std::abs(static_cast<int>(expected_bits) -
                     static_cast<int>(observed)) > 1)
          ++oracle_top;
      }
    }
  }
  oracle_ok = oracle_ok && oracle_ulp == 0U;
  std::printf("oracle staging-vs-independent m=%llu k=%llu n=%llu max_abs=%.8g "
              "max_rel=%.8g bf16_ulp_mismatch=%zu top_mismatch=%zu status=%s\n",
              static_cast<unsigned long long>(oracle_m),
              static_cast<unsigned long long>(oracle_k),
              static_cast<unsigned long long>(oracle_n), oracle_max_abs,
              oracle_max_rel, oracle_ulp, oracle_top,
              oracle_ok ? "PASS" : "N2");
  const bool oracle_compare =
      compare_outputs(oracle_control, oracle_staging, "staging-vs-ID62-oracle",
                      oracle_m, oracle_n);
  bool direct_oracle_available =
      launch_staging_direct_bf16(oracle_m, oracle_k, oracle_n, 0, &oracle) &&
      hip_ok(hipDeviceSynchronize(), "oracle direct BF16 sync") &&
      hip_ok(hipMemcpy(oracle_direct.data(), oracle.output,
                       oracle_direct.size() * sizeof(uint16_t),
                       hipMemcpyDeviceToHost),
             "oracle direct BF16 copy");
  bool direct_oracle_compare = false;
  if (direct_oracle_available) {
    direct_oracle_compare =
        compare_outputs(oracle_staging, oracle_direct, "direct-BF16-vs-staging",
                        oracle_m, oracle_n);
  }
  std::printf(
      "direct-bf16 oracle_available=%s bitwise_match=%s "
      "workspace_reduction_bytes=%llu status=%s\n",
      direct_oracle_available ? "yes" : "no",
      direct_oracle_available && direct_oracle_compare ? "yes" : "no",
      static_cast<unsigned long long>(oracle_m * oracle_n * sizeof(float)),
      direct_oracle_available ? (direct_oracle_compare ? "PASS" : "N2")
                              : "UNAVAILABLE");
  const bool oracle_cleanup = cleanup(&oracle);

  const int32_t solution = [] {
    const char *const text = std::getenv("SLLM_PHASE78_NVFP4_GFX1030_SOLUTION");
    return text == nullptr
               ? 0
               : static_cast<int32_t>(std::strtol(text, nullptr, 10));
  }();
  const bool long_run =
      std::getenv("SLLM_PHASE78_NVFP4_GFX1030_LONG") != nullptr;
  const bool direct_bf16 =
      std::getenv("SLLM_PHASE78_NVFP4_GFX1030_DIRECT_BF16") != nullptr;
  const std::array<std::array<uint64_t, 3>, 4> short_shapes = {
      std::array<uint64_t, 3>{128U, 5120U, 17408U},
      std::array<uint64_t, 3>{128U, 17408U, 5120U},
      std::array<uint64_t, 3>{512U, 5120U, 17408U},
      std::array<uint64_t, 3>{512U, 17408U, 5120U}};
  bool all_ok = oracle_ok && oracle_compare && oracle_cleanup;
  for (const auto &shape : short_shapes) {
    const uint64_t m = shape[0], k = shape[1], n = shape[2];
    fill_inputs(m, k, n, &activation, &activation_scales, &weight,
                &weight_scales);
    Buffers b;
    if (!make_buffers(m, k, n, &b) ||
        !upload(m, k, n, activation, activation_scales, weight, weight_scales,
                &b)) {
      cleanup(&b);
      return EXIT_FAILURE;
    }
    if (!launch_staging(m, k, n, solution, &b) ||
        !hip_ok(hipStreamSynchronize(b.stream), "solution preparation")) {
      cleanup(&b);
      return EXIT_FAILURE;
    }
    (void)print_solutions(b.rocblas, m, k, n, b);
    float control_us = 0.0F, staging_us = 0.0F, direct_us = 0.0F;
    if (!measure_control(m, k, n, &b, &control_us) ||
        !measure_staging(m, k, n, solution, &b, &staging_us)) {
      cleanup(&b);
      return EXIT_FAILURE;
    }
    const double resident_weight_bytes = static_cast<double>(n) * k / 2.0;
    const double pipeline_bytes =
        resident_weight_bytes + static_cast<double>(m) * k / 2.0;
    const uint64_t workspace = m * k * sizeof(uint16_t) +
                               n * k * sizeof(uint16_t) + m * n * sizeof(float);
    std::printf(
        "result m=%llu k=%llu n=%llu control_us=%.3f staging_us=%.3f "
        "speedup=%.6f staging_gbps=%.6f workspace_bytes=%llu solution=%d\n",
        static_cast<unsigned long long>(m), static_cast<unsigned long long>(k),
        static_cast<unsigned long long>(n), control_us, staging_us,
        static_cast<double>(control_us) / staging_us,
        pipeline_bytes / staging_us / 1000.0,
        static_cast<unsigned long long>(workspace), solution);
    std::vector<uint16_t> control_output(static_cast<size_t>(m * n));
    std::vector<uint16_t> staging_output(static_cast<size_t>(m * n));
    const bool compared =
        launch_control(m, k, n, &b) &&
        hip_ok(hipDeviceSynchronize(), "control compare sync") &&
        hip_ok(hipMemcpy(control_output.data(), b.output,
                         control_output.size() * sizeof(uint16_t),
                         hipMemcpyDeviceToHost),
               "control compare copy") &&
        launch_staging(m, k, n, solution, &b) &&
        hip_ok(hipDeviceSynchronize(), "staging compare sync") &&
        hip_ok(hipMemcpy(staging_output.data(), b.output,
                         staging_output.size() * sizeof(uint16_t),
                         hipMemcpyDeviceToHost),
               "staging compare copy");
    all_ok = compared &&
             compare_outputs(control_output, staging_output, "staging-vs-ID62",
                             m, n) &&
             all_ok;
    if (direct_bf16) {
      const bool direct_measured =
          measure_direct_bf16(m, k, n, solution, &b, &direct_us);
      const uint64_t direct_workspace =
          m * k * sizeof(uint16_t) + n * k * sizeof(uint16_t);
      std::printf(
          "direct-bf16 m=%llu k=%llu n=%llu available=%s direct_us=%.3f "
          "workspace_bytes=%llu reduction_bytes=%llu solution=%d\n",
          static_cast<unsigned long long>(m),
          static_cast<unsigned long long>(k),
          static_cast<unsigned long long>(n), direct_measured ? "yes" : "no",
          direct_us, static_cast<unsigned long long>(direct_workspace),
          static_cast<unsigned long long>(m * n * sizeof(float)), solution);
      if (direct_measured) {
        std::vector<uint16_t> direct_output(static_cast<size_t>(m * n));
        const bool direct_compare_ready =
            hip_ok(hipStreamSynchronize(b.stream),
                   "direct BF16 compare sync") &&
            hip_ok(hipMemcpy(direct_output.data(), b.output,
                             direct_output.size() * sizeof(uint16_t),
                             hipMemcpyDeviceToHost),
                   "direct BF16 compare copy");
        all_ok = direct_compare_ready &&
                 compare_outputs(staging_output, direct_output,
                                 "direct-BF16-vs-staging", m, n) &&
                 all_ok;
      }
    }
    all_ok = cleanup(&b) && all_ok;
  }
  if (long_run) {
    const std::array<std::array<uint64_t, 3>, 4> long_shapes = {
        std::array<uint64_t, 3>{1024U, 5120U, 17408U},
        std::array<uint64_t, 3>{1024U, 17408U, 5120U},
        std::array<uint64_t, 3>{2048U, 5120U, 17408U},
        std::array<uint64_t, 3>{2048U, 17408U, 5120U}};
    for (const auto &shape : long_shapes) {
      const uint64_t m = shape[0], k = shape[1], n = shape[2];
      fill_inputs(m, k, n, &activation, &activation_scales, &weight,
                  &weight_scales);
      Buffers b;
      if (!make_buffers(m, k, n, &b) ||
          !upload(m, k, n, activation, activation_scales, weight, weight_scales,
                  &b)) {
        cleanup(&b);
        return EXIT_FAILURE;
      }
      float control_us = 0.0F, staging_us = 0.0F;
      const bool measured = measure_control(m, k, n, &b, &control_us) &&
                            measure_staging(m, k, n, solution, &b, &staging_us);
      const uint64_t workspace = m * k * sizeof(uint16_t) +
                                 n * k * sizeof(uint16_t) +
                                 m * n * sizeof(float);
      std::printf(
          "result-long m=%llu k=%llu n=%llu control_us=%.3f staging_us=%.3f "
          "speedup=%.6f workspace_bytes=%llu solution=%d\n",
          static_cast<unsigned long long>(m),
          static_cast<unsigned long long>(k),
          static_cast<unsigned long long>(n), control_us, staging_us,
          measured ? static_cast<double>(control_us) / staging_us : 0.0,
          static_cast<unsigned long long>(workspace), solution);
      if (direct_bf16 && measured) {
        const uint64_t direct_workspace =
            m * k * sizeof(uint16_t) + n * k * sizeof(uint16_t);
        std::vector<uint16_t> staging_output(static_cast<size_t>(m * n));
        std::vector<uint16_t> direct_output(static_cast<size_t>(m * n));
        const bool staging_copy =
            hip_ok(hipMemcpy(staging_output.data(), b.output,
                             staging_output.size() * sizeof(uint16_t),
                             hipMemcpyDeviceToHost),
                   "long staging output copy");
        float direct_us = 0.0F;
        const bool direct_measured =
            staging_copy &&
            measure_direct_bf16(m, k, n, solution, &b, &direct_us);
        std::printf(
            "direct-bf16-long m=%llu k=%llu n=%llu available=%s direct_us=%.3f "
            "workspace_bytes=%llu reduction_bytes=%llu solution=%d\n",
            static_cast<unsigned long long>(m),
            static_cast<unsigned long long>(k),
            static_cast<unsigned long long>(n), direct_measured ? "yes" : "no",
            direct_us, static_cast<unsigned long long>(direct_workspace),
            static_cast<unsigned long long>(m * n * sizeof(float)), solution);
        if (direct_measured) {
          const bool direct_copy =
              hip_ok(hipStreamSynchronize(b.stream),
                     "long direct BF16 compare sync") &&
              hip_ok(hipMemcpy(direct_output.data(), b.output,
                               direct_output.size() * sizeof(uint16_t),
                               hipMemcpyDeviceToHost),
                     "long direct BF16 output copy");
          all_ok = direct_copy &&
                   compare_outputs(staging_output, direct_output,
                                   "direct-BF16-vs-staging-long", m, n) &&
                   all_ok;
        }
      }
      all_ok = measured && all_ok;
      all_ok = cleanup(&b) && all_ok;
    }
  }
  std::printf("summary status=%s warmups=%u measured=%u\n",
              all_ok ? "PASS" : "N2", kWarmups, kMeasured);
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
