// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase78-nvfp4-byte-permute-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// 3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70, ggml/src/ggml-cuda/vecdotq.cuh
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

// Phase 78 standalone gfx1030 NVFP4 W4A4 DP4A tile sweep.
//
// This evidence-only probe reproduces the ID62 64x64/K32 kernel and compares
// smaller output tiles.  All variants preserve the production block16 order:
// integer dot4, convert to FP32, multiply activation/weight E4M3 scales, then
// apply the two tensor scales and BF16 RNE.  No production source is changed.

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
  for (uint32_t lane = 0U; lane < 4U; ++lane)
    result += static_cast<int8_t>(lhs >> (lane * 8U)) *
              static_cast<int8_t>(rhs >> (lane * 8U));
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
  if (magnitude == 0x7fU)
    return __uint_as_float(sign | UINT32_C(0x7fc00000));
  return __uint_as_float(sign | ((exponent + 120U) << 23U) | (mantissa << 20U));
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

template <uint32_t TileM, uint32_t TileN, uint32_t TileK>
__global__ __launch_bounds__(kThreads, 1) void dp4a_tile_kernel(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_scales, const uint8_t *const packed_weight,
    const uint8_t *const weight_scales, const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  static_assert(TileM == 32U || TileM == 64U);
  static_assert(TileN == 32U || TileN == 64U);
  static_assert(TileK == 32U || TileK == 64U || TileK == 128U || TileK == 256U);
  constexpr uint32_t blocks_per_stage = TileK / 16U;
  constexpr uint32_t packed_chunks_per_stage = TileK / 8U;
  constexpr uint32_t lds_group_stride = TileK / 4U + 1U;
  constexpr uint32_t lds_scale_stride = blocks_per_stage + 1U;
  constexpr uint32_t thread_columns = TileN >= 64U ? 16U : 8U;
  constexpr uint32_t thread_rows = kThreads / thread_columns;
  constexpr uint32_t rows_per_thread = TileM / thread_rows;
  constexpr uint32_t columns_per_thread = TileN / thread_columns;
  static_assert(thread_rows * rows_per_thread == TileM);
  static_assert(thread_columns * columns_per_thread == TileN);

  __shared__ int32_t activation_tile[TileM][lds_group_stride];
  __shared__ int32_t weight_tile[TileN][lds_group_stride];
  __shared__ float activation_scale_tile[TileM][lds_scale_stride];
  __shared__ float weight_scale_tile[TileN][lds_scale_stride];

  const uint64_t column_tiles = (n + TileN - 1U) / TileN;
  const uint64_t tile_index = blockIdx.x;
  const uint64_t row_base = (tile_index / column_tiles) * TileM;
  const uint64_t column_base = (tile_index % column_tiles) * TileN;
  const uint32_t thread = threadIdx.x;
  const uint32_t local_row = thread / thread_columns;
  const uint32_t local_column = thread % thread_columns;
  const uint64_t packed_row_bytes = k / 2U;
  const uint64_t blocks_per_row = k / 16U;
  float accumulators[rows_per_thread][columns_per_thread] = {};

  for (uint64_t base = 0U; base < k; base += TileK) {
    for (uint32_t index = thread; index < TileM * packed_chunks_per_stage;
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
    for (uint32_t index = thread; index < TileN * packed_chunks_per_stage;
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
    for (uint32_t index = thread; index < TileM * blocks_per_stage;
         index += kThreads) {
      const uint32_t row = index / blocks_per_stage;
      const uint32_t block = index % blocks_per_stage;
      const uint64_t source_row = row_base + row;
      const uint64_t source_block = base / 16U + block;
      activation_scale_tile[row][block] =
          source_row < m && source_block < blocks_per_row
              ? e4m3(__builtin_nontemporal_load(activation_scales +
                                                source_row * blocks_per_row +
                                                source_block))
              : 0.0F;
    }
    for (uint32_t index = thread; index < TileN * blocks_per_stage;
         index += kThreads) {
      const uint32_t column = index / blocks_per_stage;
      const uint32_t block = index % blocks_per_stage;
      const uint64_t source_column = column_base + column;
      const uint64_t source_block = base / 16U + block;
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
      if (base + static_cast<uint64_t>(block) * 16U >= k)
        continue;
      int32_t block_sums[rows_per_thread][columns_per_thread] = {};
#pragma unroll
      for (uint32_t group = 0U; group < 16U / 4U; ++group) {
        int32_t activation_packs[rows_per_thread];
        int32_t weight_packs[columns_per_thread];
#pragma unroll
        for (uint32_t row = 0U; row < rows_per_thread; ++row)
          activation_packs[row] = activation_tile[local_row + row * thread_rows]
                                                 [block * 4U + group];
#pragma unroll
        for (uint32_t column = 0U; column < columns_per_thread; ++column)
          weight_packs[column] =
              weight_tile[local_column + column * thread_columns]
                         [block * 4U + group];
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
            activation_scale_tile[local_row + row * thread_rows][block];
#pragma unroll
        for (uint32_t column = 0U; column < columns_per_thread; ++column) {
          const float weight_scale =
              weight_scale_tile[local_column + column * thread_columns][block];
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
      const uint64_t output_row = row_base + local_row + row * thread_rows;
      const uint64_t output_column =
          column_base + local_column + column * thread_columns;
      if (output_row < m && output_column < n)
        output[output_row * n + output_column] =
            bf16_rne(accumulators[row][column] * tensor_scale);
    }
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
  const uint32_t result = sign | ((exponent + 120U) << 23U) | (mantissa << 20U);
  float value = 0.0F;
  std::memcpy(&value, &result, sizeof(value));
  return value;
}

struct HostInputs final {
  uint64_t m = 0U;
  uint64_t k = 0U;
  uint64_t n = 0U;
  std::vector<uint8_t> activation, activation_scales, weight, weight_scales;
};

HostInputs make_inputs(const uint64_t m, const uint64_t k, const uint64_t n) {
  HostInputs h{m, k, n};
  const uint64_t blocks = k / 16U;
  h.activation.assign(static_cast<size_t>(m * k / 2U), 0U);
  h.activation_scales.assign(static_cast<size_t>(m * blocks), 0x38U);
  h.weight.assign(static_cast<size_t>(n * k / 2U), 0U);
  h.weight_scales.assign(static_cast<size_t>(n * blocks), 0x38U);
  for (uint64_t row = 0U; row < m; ++row) {
    for (uint64_t inner = 0U; inner < k; ++inner) {
      const uint8_t code =
          static_cast<uint8_t>((row * 5U + inner * 3U + 1U) & 0x0fU);
      const size_t index = static_cast<size_t>(row * k / 2U + inner / 2U);
      if ((inner & 1U) == 0U)
        h.activation[index] = code;
      else
        h.activation[index] |= static_cast<uint8_t>(code << 4U);
    }
    for (uint64_t block = 0U; block < blocks; ++block)
      h.activation_scales[static_cast<size_t>(row * blocks + block)] =
          (block & 1U) == 0U ? 0x38U : 0x40U;
  }
  for (uint64_t column = 0U; column < n; ++column) {
    for (uint64_t inner = 0U; inner < k; ++inner) {
      const uint8_t code =
          static_cast<uint8_t>((column * 7U + inner * 9U + 2U) & 0x0fU);
      const size_t index = static_cast<size_t>(column * k / 2U + inner / 2U);
      if ((inner & 1U) == 0U)
        h.weight[index] = code;
      else
        h.weight[index] |= static_cast<uint8_t>(code << 4U);
    }
    for (uint64_t block = 0U; block < blocks; ++block)
      h.weight_scales[static_cast<size_t>(column * blocks + block)] =
          (block & 1U) == 0U ? 0x38U : 0x40U;
  }
  return h;
}

std::vector<uint16_t> host_oracle(const HostInputs &h) {
  std::vector<uint16_t> output(static_cast<size_t>(h.m * h.n), 0U);
  const uint64_t blocks = h.k / 16U;
  for (uint64_t row = 0U; row < h.m; ++row) {
    for (uint64_t column = 0U; column < h.n; ++column) {
      float sum = 0.0F;
      for (uint64_t block = 0U; block < blocks; ++block) {
        float block_sum = 0.0F;
        for (uint32_t inner = 0U; inner < 16U; ++inner) {
          const uint64_t index = block * 16U + inner;
          const uint8_t ap =
              h.activation[static_cast<size_t>(row * h.k / 2U + index / 2U)];
          const uint8_t wp =
              h.weight[static_cast<size_t>(column * h.k / 2U + index / 2U)];
          const uint8_t ac = (index & 1U) == 0U ? ap & 0x0fU : ap >> 4U;
          const uint8_t wc = (index & 1U) == 0U ? wp & 0x0fU : wp >> 4U;
          block_sum += host_e2m1(ac) * host_e2m1(wc);
        }
        sum +=
            block_sum *
            host_e4m3(h.activation_scales[static_cast<size_t>(row * blocks +
                                                              block)]) *
            host_e4m3(
                h.weight_scales[static_cast<size_t>(column * blocks + block)]);
      }
      output[static_cast<size_t>(row * h.n + column)] =
          host_bf16_rne(sum * 0.75F * 1.125F);
    }
  }
  return output;
}

struct Buffers final {
  uint8_t *activation = nullptr, *activation_scales = nullptr;
  uint8_t *weight = nullptr, *weight_scales = nullptr;
  uint16_t *output = nullptr;
  float *weight_tensor_scale = nullptr, *input_tensor_scale = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr, stop = nullptr;
};

bool hip_ok(const hipError_t status, const char *const op) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "hip error operation=%s status=%s\n", op,
               hipGetErrorString(status));
  return false;
}

void cleanup(Buffers *const b) {
  if (b == nullptr)
    return;
  if (b->stop)
    (void)hipEventDestroy(b->stop);
  if (b->start)
    (void)hipEventDestroy(b->start);
  if (b->stream)
    (void)hipStreamDestroy(b->stream);
  if (b->input_tensor_scale)
    (void)hipFree(b->input_tensor_scale);
  if (b->weight_tensor_scale)
    (void)hipFree(b->weight_tensor_scale);
  if (b->output)
    (void)hipFree(b->output);
  if (b->weight_scales)
    (void)hipFree(b->weight_scales);
  if (b->weight)
    (void)hipFree(b->weight);
  if (b->activation_scales)
    (void)hipFree(b->activation_scales);
  if (b->activation)
    (void)hipFree(b->activation);
  *b = {};
}

bool make_buffers(const HostInputs &h, Buffers *const b) {
  const size_t a = h.activation.size(), as = h.activation_scales.size();
  const size_t w = h.weight.size(), ws = h.weight_scales.size();
  const size_t out = static_cast<size_t>(h.m * h.n * sizeof(uint16_t));
  return hip_ok(hipMalloc(reinterpret_cast<void **>(&b->activation), a),
                "malloc activation") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->activation_scales), as),
                "malloc activation scales") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->weight), w),
                "malloc weight") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->weight_scales), ws),
                "malloc weight scales") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->output), out),
                "malloc output") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->weight_tensor_scale),
                          sizeof(float)),
                "malloc weight tensor scale") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->input_tensor_scale),
                          sizeof(float)),
                "malloc input tensor scale") &&
         hip_ok(hipStreamCreate(&b->stream), "stream") &&
         hip_ok(hipEventCreate(&b->start), "event start") &&
         hip_ok(hipEventCreate(&b->stop), "event stop");
}

bool upload(const HostInputs &h, Buffers *const b) {
  const float wt = 0.75F, at = 1.125F;
  return hip_ok(hipMemcpy(b->activation, h.activation.data(),
                          h.activation.size(), hipMemcpyHostToDevice),
                "upload activation") &&
         hip_ok(hipMemcpy(b->activation_scales, h.activation_scales.data(),
                          h.activation_scales.size(), hipMemcpyHostToDevice),
                "upload activation scales") &&
         hip_ok(hipMemcpy(b->weight, h.weight.data(), h.weight.size(),
                          hipMemcpyHostToDevice),
                "upload weight") &&
         hip_ok(hipMemcpy(b->weight_scales, h.weight_scales.data(),
                          h.weight_scales.size(), hipMemcpyHostToDevice),
                "upload weight scales") &&
         hip_ok(hipMemcpy(b->weight_tensor_scale, &wt, sizeof(wt),
                          hipMemcpyHostToDevice),
                "upload weight tensor scale") &&
         hip_ok(hipMemcpy(b->input_tensor_scale, &at, sizeof(at),
                          hipMemcpyHostToDevice),
                "upload input tensor scale") &&
         hip_ok(hipMemset(b->output, 0, h.m * h.n * sizeof(uint16_t)),
                "clear output");
}

template <uint32_t TileM, uint32_t TileN, uint32_t TileK>
bool launch_tile(const HostInputs &h, Buffers *const b) {
  const uint64_t tiles_x = (h.n + TileN - 1U) / TileN;
  const uint64_t blocks = ((h.m + TileM - 1U) / TileM) * tiles_x;
  hipLaunchKernelGGL((dp4a_tile_kernel<TileM, TileN, TileK>),
                     dim3(static_cast<uint32_t>(blocks)), dim3(kThreads), 0U,
                     b->stream, b->activation, b->activation_scales, b->weight,
                     b->weight_scales, b->weight_tensor_scale,
                     b->input_tensor_scale, b->output, h.m, h.k, h.n);
  return hip_ok(hipGetLastError(), "tile launch");
}

template <uint32_t TileM, uint32_t TileN, uint32_t TileK>
bool measure_tile(const HostInputs &h, Buffers *const b,
                  float *const median_us) {
  for (uint32_t i = 0U; i < kWarmups; ++i)
    if (!launch_tile<TileM, TileN, TileK>(h, b) ||
        !hip_ok(hipStreamSynchronize(b->stream), "warmup sync"))
      return false;
  std::array<float, kMeasured> samples{};
  for (uint32_t i = 0U; i < kMeasured; ++i) {
    if (!hip_ok(hipEventRecord(b->start, b->stream), "event start") ||
        !launch_tile<TileM, TileN, TileK>(h, b) ||
        !hip_ok(hipEventRecord(b->stop, b->stream), "event stop") ||
        !hip_ok(hipEventSynchronize(b->stop), "event sync") ||
        !hip_ok(hipEventElapsedTime(&samples[i], b->start, b->stop), "elapsed"))
      return false;
    samples[i] *= 1000.0F;
  }
  std::sort(samples.begin(), samples.end());
  *median_us = samples[kMeasured / 2U];
  return true;
}

uint32_t ulp_distance(const uint16_t a, const uint16_t b) {
  if ((a & 0x7fffU) == 0U && (b & 0x7fffU) == 0U)
    return 0U;
  const int32_t ai = (a & 0x8000U) ? 0x8000 - (a & 0x7fffU) : 0x8000 + a;
  const int32_t bi = (b & 0x8000U) ? 0x8000 - (b & 0x7fffU) : 0x8000 + b;
  return static_cast<uint32_t>(std::abs(ai - bi));
}

void compare(const char *const name, const std::vector<uint16_t> &ref,
             const std::vector<uint16_t> &actual) {
  uint32_t max_ulp = 0U;
  uint64_t over_one = 0U;
  for (size_t i = 0U; i < ref.size(); ++i) {
    const uint32_t ulp = ulp_distance(ref[i], actual[i]);
    max_ulp = std::max(max_ulp, ulp);
    if (ulp > 1U)
      ++over_one;
  }
  std::printf(
      "compare candidate=%s values=%zu max_bf16_ulp=%u over1=%llu status=%s\n",
      name, ref.size(), max_ulp, static_cast<unsigned long long>(over_one),
      over_one == 0U ? "PASS" : "INFO");
}

template <uint32_t TileM, uint32_t TileN, uint32_t TileK>
void print_resource(const char *const name) {
  hipFuncAttributes attr{};
  const void *const fn =
      reinterpret_cast<const void *>(dp4a_tile_kernel<TileM, TileN, TileK>);
  const hipError_t attrs = hipFuncGetAttributes(&attr, fn);
  int active = 0;
  const hipError_t occ =
      hipOccupancyMaxActiveBlocksPerMultiprocessor(&active, fn, kThreads, 0U);
  std::printf("resources candidate=%s registers=%d lds=%zu scratch=%zu "
              "active_blocks=%d occupancy=%s attrs=%s\n",
              name, attr.numRegs, attr.sharedSizeBytes, attr.localSizeBytes,
              active, hipGetErrorString(occ), hipGetErrorString(attrs));
}

template <uint32_t TileM, uint32_t TileN, uint32_t TileK>
bool run_candidate(const char *const name, const HostInputs &h,
                   Buffers *const b, const std::vector<uint16_t> &control,
                   const float control_us, const double traffic_bytes) {
  float us = 0.0F;
  if (!measure_tile<TileM, TileN, TileK>(h, b, &us))
    return false;
  std::vector<uint16_t> actual;
  actual.resize(static_cast<size_t>(h.m * h.n));
  if (!hip_ok(hipMemcpy(actual.data(), b->output,
                        actual.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              "copy candidate"))
    return false;
  compare(name, control, actual);
  std::printf(
      "result candidate=%s m=%llu k=%llu n=%llu ms=%.6f effective_gbps=%.3f "
      "speedup_vs_control=%.3f traffic_bytes=%.0f\n",
      name, static_cast<unsigned long long>(h.m),
      static_cast<unsigned long long>(h.k),
      static_cast<unsigned long long>(h.n), us / 1000.0F,
      traffic_bytes / (static_cast<double>(us) * 1000.0), control_us / us,
      traffic_bytes);
  return true;
}

bool run_shape(const uint64_t m, const uint64_t k, const uint64_t n) {
  HostInputs h = make_inputs(m, k, n);
  Buffers b;
  if (!make_buffers(h, &b) || !upload(h, &b)) {
    cleanup(&b);
    return false;
  }
  float control_us = 0.0F;
  if (!measure_tile<64U, 64U, 32U>(h, &b, &control_us)) {
    cleanup(&b);
    return false;
  }
  std::vector<uint16_t> control(static_cast<size_t>(m * n));
  if (!hip_ok(hipMemcpy(control.data(), b.output,
                        control.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              "copy control")) {
    cleanup(&b);
    return false;
  }
  const double traffic =
      static_cast<double>(m * k / 2U + m * (k / 16U) + n * k / 2U +
                          n * (k / 16U) + m * n * sizeof(uint16_t));
  std::printf(
      "control m=%llu k=%llu n=%llu ms=%.6f traffic_bytes=%.0f\n",
      static_cast<unsigned long long>(m), static_cast<unsigned long long>(k),
      static_cast<unsigned long long>(n), control_us / 1000.0F, traffic);
  bool ok = run_candidate<32U, 64U, 32U>("tile32x64_k32", h, &b, control,
                                         control_us, traffic);
  ok = run_candidate<64U, 32U, 32U>("tile64x32_k32", h, &b, control, control_us,
                                    traffic) &&
       ok;
  ok = run_candidate<32U, 32U, 32U>("tile32x32_k32", h, &b, control, control_us,
                                    traffic) &&
       ok;
  ok = run_candidate<64U, 64U, 64U>("tile64x64_k64", h, &b, control, control_us,
                                    traffic) &&
       ok;
  ok = run_candidate<64U, 64U, 128U>("tile64x64_k128", h, &b, control,
                                     control_us, traffic) &&
       ok;
  ok = run_candidate<64U, 64U, 256U>("tile64x64_k256", h, &b, control,
                                     control_us, traffic) &&
       ok;
  cleanup(&b);
  return ok;
}

} // namespace

int main() {
  if (!hip_ok(hipSetDevice(0), "hipSetDevice"))
    return EXIT_FAILURE;
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0), "device properties"))
    return EXIT_FAILURE;
  std::printf("target=%s pci=%04x:%02x:%02x name=%s\n", properties.gcnArchName,
              properties.pciDomainID, properties.pciBusID,
              properties.pciDeviceID, properties.name);
  if (std::string_view(properties.gcnArchName).compare(0U, 7U, "gfx1030") !=
      0U) {
    std::fprintf(stderr, "gfx1030 required\n");
    return EXIT_FAILURE;
  }
  print_resource<64U, 64U, 32U>("control64x64_k32");
  print_resource<32U, 64U, 32U>("tile32x64_k32");
  print_resource<64U, 32U, 32U>("tile64x32_k32");
  print_resource<32U, 32U, 32U>("tile32x32_k32");
  print_resource<64U, 64U, 64U>("tile64x64_k64");
  print_resource<64U, 64U, 128U>("tile64x64_k128");
  print_resource<64U, 64U, 256U>("tile64x64_k256");

  // Tail oracle first: non-aligned M/N and a K value crossing multiple K32
  // stages, with a host reference that uses the same block16 scale order.
  {
    HostInputs h = make_inputs(7U, 160U, 65U);
    const std::vector<uint16_t> expected = host_oracle(h);
    Buffers b;
    if (!make_buffers(h, &b) || !upload(h, &b) ||
        !launch_tile<64U, 64U, 32U>(h, &b) ||
        !hip_ok(hipStreamSynchronize(b.stream), "oracle control sync")) {
      cleanup(&b);
      return EXIT_FAILURE;
    }
    std::vector<uint16_t> actual(expected.size());
    if (!hip_ok(hipMemcpy(actual.data(), b.output,
                          actual.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "oracle control copy")) {
      cleanup(&b);
      return EXIT_FAILURE;
    }
    compare("control_small_host_oracle", expected, actual);
    for (const auto candidate : {0U, 1U, 2U, 3U, 4U, 5U}) {
      bool launch_ok = candidate == 0U   ? launch_tile<32U, 64U, 32U>(h, &b)
                       : candidate == 1U ? launch_tile<64U, 32U, 32U>(h, &b)
                       : candidate == 2U ? launch_tile<32U, 32U, 32U>(h, &b)
                       : candidate == 3U ? launch_tile<64U, 64U, 64U>(h, &b)
                       : candidate == 4U ? launch_tile<64U, 64U, 128U>(h, &b)
                                         : launch_tile<64U, 64U, 256U>(h, &b);
      if (!launch_ok ||
          !hip_ok(hipStreamSynchronize(b.stream), "oracle candidate sync")) {
        cleanup(&b);
        return EXIT_FAILURE;
      }
      if (!hip_ok(hipMemcpy(actual.data(), b.output,
                            actual.size() * sizeof(uint16_t),
                            hipMemcpyDeviceToHost),
                  "oracle candidate copy")) {
        cleanup(&b);
        return EXIT_FAILURE;
      }
      compare(candidate == 0U   ? "tile32x64_small_oracle"
              : candidate == 1U ? "tile64x32_small_oracle"
              : candidate == 2U ? "tile32x32_small_oracle"
              : candidate == 3U ? "tile64x64_k64_small_oracle"
              : candidate == 4U ? "tile64x64_k128_small_oracle"
                                : "tile64x64_k256_small_oracle",
              expected, actual);
    }
    cleanup(&b);
  }

  bool ok = true;
  for (const uint64_t m : {128U, 512U, 1024U}) {
    ok = run_shape(m, 5120U, 17408U) && ok;
    ok = run_shape(m, 17408U, 5120U) && ok;
  }
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
