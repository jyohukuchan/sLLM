// Phase 78 short-prefill FP8 specialization.
//
// This is the ID71 half2 K32 schedule with a short-M 32x32 or 32x64 output
// tile. The source bytes, packed E4M3FN expansion, FP32 accumulation order,
// scale placement, and BF16-RNE epilogue intentionally match the production
// 64x64 body. The smaller row/column ownership is useful only when the caller
// has selected a short-M shape; the launcher keeps the existing ID71 body for
// every other shape.

#include "low_precision_block_codec.hpp"

#include <hip/hip_fp16.h>

#include <cstdint>

namespace sllm_phase78_fp8_short_m32 {

constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kTileM = 32U;
constexpr uint32_t kTileK = 32U;
constexpr uint32_t kLdsStride = kTileK + 2U;
constexpr uint32_t kValuesPerPackedLoad = 4U;

template <uint32_t kTileN>
__device__ __forceinline__ void fp8_outer_prefill_short_m32_body(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  static_assert(kTileN == 32U || kTileN == 64U);
  constexpr uint32_t kThreadRows = kTileN == 32U ? 16U : 8U;
  constexpr uint32_t kThreadColumns = kTileN == 32U ? 16U : 32U;
  constexpr uint32_t kThreadColumnBits = kTileN == 32U ? 4U : 5U;
  constexpr uint32_t rows_per_thread = kTileM / kThreadRows;
  constexpr uint32_t columns_per_thread = kTileN / kThreadColumns;
  static_assert(kTileN == 32U ? rows_per_thread == 2U : rows_per_thread == 4U);
  static_assert(columns_per_thread == 2U);
  __shared__ __align__(4) uint16_t activation_tile[kTileM][kLdsStride];
  __shared__ __align__(4) uint16_t weight_tile[kTileN][kLdsStride];

  const uint64_t column_tiles = (n + kTileN - 1U) / kTileN;
  const uint64_t tile_index = static_cast<uint64_t>(blockIdx.x);
  const uint64_t row_base = (tile_index / column_tiles) * kTileM;
  const uint64_t column_base = (tile_index % column_tiles) * kTileN;
  const uint32_t thread = threadIdx.x;
  const uint32_t local_row = thread >> kThreadColumnBits;
  const uint32_t local_column =
      thread & ((UINT32_C(1) << kThreadColumnBits) - UINT32_C(1));
  float accumulators[rows_per_thread][columns_per_thread] = {};

  for (uint64_t base = 0U; base < k; base += kTileK) {
    constexpr uint32_t loads_per_row = kTileK / kValuesPerPackedLoad;
    for (uint32_t index = thread; index < kTileM * loads_per_row;
         index += kWorkgroupSize) {
      const uint32_t row = index / loads_per_row;
      const uint32_t inner = (index % loads_per_row) * kValuesPerPackedLoad;
      const uint64_t source_row = row_base + row;
      const uint64_t source_inner = base + inner;
      if ((k % kValuesPerPackedLoad) == 0U && source_row < m &&
          source_inner + kValuesPerPackedLoad <= k) {
        const uint32_t packed =
            __builtin_nontemporal_load(reinterpret_cast<const uint32_t *>(
                activation + source_row * k + source_inner));
        const sllm_lowp::E4M3FnFp16x4Bits expanded =
            sllm_lowp::e4m3fnx4_to_fp16x2_bits(packed);
        auto *const packed_output =
            reinterpret_cast<uint32_t *>(&activation_tile[row][inner]);
        packed_output[0] = expanded.low;
        packed_output[1] = expanded.high;
      } else {
#pragma unroll
        for (uint32_t lane = 0U; lane < kValuesPerPackedLoad; ++lane) {
          activation_tile[row][inner + lane] =
              source_row < m && source_inner + lane < k
                  ? sllm_lowp::e4m3fn_to_fp16_bits(__builtin_nontemporal_load(
                        activation + source_row * k + source_inner + lane))
                  : UINT16_C(0);
        }
      }
    }
    for (uint32_t index = thread; index < kTileN * loads_per_row;
         index += kWorkgroupSize) {
      const uint32_t column = index / loads_per_row;
      const uint32_t inner = (index % loads_per_row) * kValuesPerPackedLoad;
      const uint64_t source_column = column_base + column;
      const uint64_t source_inner = base + inner;
      if ((k % kValuesPerPackedLoad) == 0U && source_column < n &&
          source_inner + kValuesPerPackedLoad <= k) {
        const uint32_t packed =
            __builtin_nontemporal_load(reinterpret_cast<const uint32_t *>(
                weight + source_column * k + source_inner));
        const sllm_lowp::E4M3FnFp16x4Bits expanded =
            sllm_lowp::e4m3fnx4_to_fp16x2_bits(packed);
        auto *const packed_output =
            reinterpret_cast<uint32_t *>(&weight_tile[column][inner]);
        packed_output[0] = expanded.low;
        packed_output[1] = expanded.high;
      } else {
#pragma unroll
        for (uint32_t lane = 0U; lane < kValuesPerPackedLoad; ++lane) {
          weight_tile[column][inner + lane] =
              source_column < n && source_inner + lane < k
                  ? sllm_lowp::e4m3fn_to_fp16_bits(__builtin_nontemporal_load(
                        weight + source_column * k + source_inner + lane))
                  : UINT16_C(0);
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (uint32_t inner = 0U; inner < kTileK; inner += 2U) {
      __half2 activation_pairs[rows_per_thread];
      __half2 weight_pairs[columns_per_thread];
#pragma unroll
      for (uint32_t row = 0U; row < rows_per_thread; ++row) {
        activation_pairs[row] = *reinterpret_cast<const __half2 *>(
            &activation_tile[local_row + row * kThreadRows][inner]);
      }
#pragma unroll
      for (uint32_t column = 0U; column < columns_per_thread; ++column) {
        weight_pairs[column] = *reinterpret_cast<const __half2 *>(
            &weight_tile[local_column + column * kThreadColumns][inner]);
      }
#pragma unroll
      for (uint32_t row = 0U; row < rows_per_thread; ++row) {
#pragma unroll
        for (uint32_t column = 0U; column < columns_per_thread; ++column) {
          accumulators[row][column] =
              amd_mixed_dot(activation_pairs[row], weight_pairs[column],
                            accumulators[row][column], false);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t row = 0U; row < rows_per_thread; ++row) {
    const uint64_t output_row = row_base + local_row + row * kThreadRows;
#pragma unroll
    for (uint32_t column = 0U; column < columns_per_thread; ++column) {
      const uint64_t output_column =
          column_base + local_column + column * kThreadColumns;
      if (output_row < m && output_column < n) {
        output[output_row * n + output_column] = float_to_bf16_rne_bits(
            accumulators[row][column] * activation_scales[output_row] *
            weight_scales[output_column]);
      }
    }
  }
}

extern "C" __global__
__launch_bounds__(kWorkgroupSize, 1) void sllm_matmul_fp8_outer_prefill_gfx1030_half2_32x32_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  fp8_outer_prefill_short_m32_body<32U>(activation, activation_scales, weight,
                                        weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(kWorkgroupSize, 1) void sllm_matmul_fp8_outer_prefill_gfx1030_half2_32x64_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  fp8_outer_prefill_short_m32_body<64U>(activation, activation_scales, weight,
                                        weight_scales, output, m, k, n);
}

} // namespace sllm_phase78_fp8_short_m32
