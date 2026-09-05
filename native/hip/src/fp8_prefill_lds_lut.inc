// Phase 78 gfx1030 FP8 prefill ingress candidate.
//
// The include is intentionally self-contained so that it can be compiled as
// a focused HIP translation unit before the dispatch registry includes it in
// matmul_kernel.hip.cpp.  It keeps the ID71 64x64/K32 half2 tile and changes
// only E4M3FN ingress: the exact FP16 bit values are looked up from a padded
// 256-entry table staged once in LDS per workgroup.

#include "low_precision_block_codec.hpp"

#include <hip/hip_fp16.h>

#include <cstdint>

namespace sllm_phase78_fp8_prefill_lds_lut {

constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kTileM = 64U;
constexpr uint32_t kTileN = 64U;
constexpr uint32_t kTileK = 32U;
constexpr uint32_t kLdsStride = kTileK + 2U;
constexpr uint32_t kLutEntries = 256U;
constexpr uint32_t kLutStorageEntries = 272U;
constexpr uint32_t kValuesPerPackedLoad = 4U;
constexpr uint32_t kThreadRows = 16U;
constexpr uint32_t kThreadColumns = 16U;

// Keep the table in the code object, as ID82 does, so workgroups do not need
// a host-side transfer before the first candidate launch.
__device__ __constant__ uint16_t
    sllm_phase78_fp8_prefill_lds_lut_table[kLutEntries] = {
        0x0000U, 0x1800U, 0x1c00U, 0x1e00U, 0x2000U, 0x2100U, 0x2200U, 0x2300U,
        0x2400U, 0x2480U, 0x2500U, 0x2580U, 0x2600U, 0x2680U, 0x2700U, 0x2780U,
        0x2800U, 0x2880U, 0x2900U, 0x2980U, 0x2a00U, 0x2a80U, 0x2b00U, 0x2b80U,
        0x2c00U, 0x2c80U, 0x2d00U, 0x2d80U, 0x2e00U, 0x2e80U, 0x2f00U, 0x2f80U,
        0x3000U, 0x3080U, 0x3100U, 0x3180U, 0x3200U, 0x3280U, 0x3300U, 0x3380U,
        0x3400U, 0x3480U, 0x3500U, 0x3580U, 0x3600U, 0x3680U, 0x3700U, 0x3780U,
        0x3800U, 0x3880U, 0x3900U, 0x3980U, 0x3a00U, 0x3a80U, 0x3b00U, 0x3b80U,
        0x3c00U, 0x3c80U, 0x3d00U, 0x3d80U, 0x3e00U, 0x3e80U, 0x3f00U, 0x3f80U,
        0x4000U, 0x4080U, 0x4100U, 0x4180U, 0x4200U, 0x4280U, 0x4300U, 0x4380U,
        0x4400U, 0x4480U, 0x4500U, 0x4580U, 0x4600U, 0x4680U, 0x4700U, 0x4780U,
        0x4800U, 0x4880U, 0x4900U, 0x4980U, 0x4a00U, 0x4a80U, 0x4b00U, 0x4b80U,
        0x4c00U, 0x4c80U, 0x4d00U, 0x4d80U, 0x4e00U, 0x4e80U, 0x4f00U, 0x4f80U,
        0x5000U, 0x5080U, 0x5100U, 0x5180U, 0x5200U, 0x5280U, 0x5300U, 0x5380U,
        0x5400U, 0x5480U, 0x5500U, 0x5580U, 0x5600U, 0x5680U, 0x5700U, 0x5780U,
        0x5800U, 0x5880U, 0x5900U, 0x5980U, 0x5a00U, 0x5a80U, 0x5b00U, 0x5b80U,
        0x5c00U, 0x5c80U, 0x5d00U, 0x5d80U, 0x5e00U, 0x5e80U, 0x5f00U, 0x7e00U,
        0x8000U, 0x9800U, 0x9c00U, 0x9e00U, 0xa000U, 0xa100U, 0xa200U, 0xa300U,
        0xa400U, 0xa480U, 0xa500U, 0xa580U, 0xa600U, 0xa680U, 0xa700U, 0xa780U,
        0xa800U, 0xa880U, 0xa900U, 0xa980U, 0xaa00U, 0xaa80U, 0xab00U, 0xab80U,
        0xac00U, 0xac80U, 0xad00U, 0xad80U, 0xae00U, 0xae80U, 0xaf00U, 0xaf80U,
        0xb000U, 0xb080U, 0xb100U, 0xb180U, 0xb200U, 0xb280U, 0xb300U, 0xb380U,
        0xb400U, 0xb480U, 0xb500U, 0xb580U, 0xb600U, 0xb680U, 0xb700U, 0xb780U,
        0xb800U, 0xb880U, 0xb900U, 0xb980U, 0xba00U, 0xba80U, 0xbb00U, 0xbb80U,
        0xbc00U, 0xbc80U, 0xbd00U, 0xbd80U, 0xbe00U, 0xbe80U, 0xbf00U, 0xbf80U,
        0xc000U, 0xc080U, 0xc100U, 0xc180U, 0xc200U, 0xc280U, 0xc300U, 0xc380U,
        0xc400U, 0xc480U, 0xc500U, 0xc580U, 0xc600U, 0xc680U, 0xc700U, 0xc780U,
        0xc800U, 0xc880U, 0xc900U, 0xc980U, 0xca00U, 0xca80U, 0xcb00U, 0xcb80U,
        0xcc00U, 0xcc80U, 0xcd00U, 0xcd80U, 0xce00U, 0xce80U, 0xcf00U, 0xcf80U,
        0xd000U, 0xd080U, 0xd100U, 0xd180U, 0xd200U, 0xd280U, 0xd300U, 0xd380U,
        0xd400U, 0xd480U, 0xd500U, 0xd580U, 0xd600U, 0xd680U, 0xd700U, 0xd780U,
        0xd800U, 0xd880U, 0xd900U, 0xd980U, 0xda00U, 0xda80U, 0xdb00U, 0xdb80U,
        0xdc00U, 0xdc80U, 0xdd00U, 0xdd80U, 0xde00U, 0xde80U, 0xdf00U, 0xfe00U};

__device__ __forceinline__ uint16_t
sllm_phase78_fp8_prefill_bf16_rne(const float value) noexcept {
  const uint32_t bits = __float_as_uint(value);
  if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
    if ((bits & UINT32_C(0x007fffff)) != 0U) {
      return static_cast<uint16_t>(((bits >> 16U) & UINT32_C(0x8000)) |
                                   UINT16_C(0x7fc0) |
                                   ((bits >> 16U) & UINT32_C(0x003f)));
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

// Add one unused entry after each 32-code group.  This is the same padded
// address mapping used by the Phase 78 probe and avoids a bank-aliasing LUT
// layout on gfx1030.
__device__ __forceinline__ uint32_t
sllm_phase78_fp8_prefill_lut_slot(const uint32_t code) noexcept {
  return code + (code >> 5U);
}

__device__ __forceinline__ uint16_t sllm_phase78_fp8_prefill_decode(
    const uint8_t code, const uint16_t *const lds_lut) noexcept {
  return lds_lut[sllm_phase78_fp8_prefill_lut_slot(
      static_cast<uint32_t>(code))];
}

__device__ __forceinline__ void
sllm_phase78_fp8_prefill_decode_four(const uint32_t packed,
                                     const uint16_t *const lds_lut,
                                     uint16_t *const destination) noexcept {
  destination[0] =
      sllm_phase78_fp8_prefill_decode(static_cast<uint8_t>(packed), lds_lut);
  destination[1] = sllm_phase78_fp8_prefill_decode(
      static_cast<uint8_t>(packed >> 8U), lds_lut);
  destination[2] = sllm_phase78_fp8_prefill_decode(
      static_cast<uint8_t>(packed >> 16U), lds_lut);
  destination[3] = sllm_phase78_fp8_prefill_decode(
      static_cast<uint8_t>(packed >> 24U), lds_lut);
}

// One 256-thread workgroup owns a 64x64 output tile.  The source tensors use
// row-major [M,K] and [N,K] storage, matching the ID71 kernel contract.
__device__ __forceinline__ void sllm_phase78_fp8_prefill_64x64_k32(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, uint16_t (&activation_tile)[kTileM][kLdsStride],
    uint16_t (&weight_tile)[kTileN][kLdsStride],
    uint16_t *const lds_lut) noexcept {
  constexpr uint32_t rows_per_thread = kTileM / kThreadRows;
  constexpr uint32_t columns_per_thread = kTileN / kThreadColumns;
  constexpr uint32_t loads_per_row = kTileK / kValuesPerPackedLoad;

  if (threadIdx.x < kLutEntries) {
    const uint32_t code = threadIdx.x;
    lds_lut[sllm_phase78_fp8_prefill_lut_slot(code)] =
        sllm_phase78_fp8_prefill_lds_lut_table[code];
  }
  __syncthreads();

  // m/n are validated by the launcher in normal use.  The n != 0 guard keeps
  // the tile mapping defined for direct callers as well; k == 0 naturally
  // emits a scaled zero tile through the epilogue below.
  if (m == 0U || n == 0U) {
    return;
  }
  const uint64_t column_tiles = (n - 1U) / kTileN + 1U;
  const uint64_t tile_index = static_cast<uint64_t>(blockIdx.x);
  const uint64_t row_base = (tile_index / column_tiles) * kTileM;
  const uint64_t column_base = (tile_index % column_tiles) * kTileN;
  const uint32_t thread = threadIdx.x;
  const uint32_t local_row = thread >> 4U;
  const uint32_t local_column = thread & UINT32_C(15);
  float accumulators[rows_per_thread][columns_per_thread] = {};
  const uint16_t *const table = lds_lut;

  for (uint64_t base = 0U; base < k; base += kTileK) {
    for (uint32_t index = thread; index < kTileM * loads_per_row;
         index += kWorkgroupSize) {
      const uint32_t row = index / loads_per_row;
      const uint32_t inner = (index % loads_per_row) * kValuesPerPackedLoad;
      const uint64_t source_row = row_base + row;
      const uint64_t source_inner = base + inner;
      if ((k % kValuesPerPackedLoad) == 0U && source_row < m &&
          source_inner + kValuesPerPackedLoad <= k &&
          (reinterpret_cast<uintptr_t>(activation + source_row * k +
                                       source_inner) &
           static_cast<uintptr_t>(3U)) == 0U) {
        const uint32_t packed =
            __builtin_nontemporal_load(reinterpret_cast<const uint32_t *>(
                activation + source_row * k + source_inner));
        sllm_phase78_fp8_prefill_decode_four(packed, table,
                                             &activation_tile[row][inner]);
      } else {
#pragma unroll
        for (uint32_t lane = 0U; lane < kValuesPerPackedLoad; ++lane) {
          activation_tile[row][inner + lane] =
              source_row < m && source_inner + lane < k
                  ? sllm_phase78_fp8_prefill_decode(
                        __builtin_nontemporal_load(activation + source_row * k +
                                                   source_inner + lane),
                        table)
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
          source_inner + kValuesPerPackedLoad <= k &&
          (reinterpret_cast<uintptr_t>(weight + source_column * k +
                                       source_inner) &
           static_cast<uintptr_t>(3U)) == 0U) {
        const uint32_t packed =
            __builtin_nontemporal_load(reinterpret_cast<const uint32_t *>(
                weight + source_column * k + source_inner));
        sllm_phase78_fp8_prefill_decode_four(packed, table,
                                             &weight_tile[column][inner]);
      } else {
#pragma unroll
        for (uint32_t lane = 0U; lane < kValuesPerPackedLoad; ++lane) {
          weight_tile[column][inner + lane] =
              source_column < n && source_inner + lane < k
                  ? sllm_phase78_fp8_prefill_decode(
                        __builtin_nontemporal_load(weight + source_column * k +
                                                   source_inner + lane),
                        table)
                  : UINT16_C(0);
        }
      }
    }
    __syncthreads();

    if (row_base + local_row < m && column_base + local_column < n) {
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
            // Keep ID71's half2 dot order and non-saturating FP32
            // accumulation.  Outer scales and BF16 RNE stay in the epilogue.
            accumulators[row][column] =
                amd_mixed_dot(activation_pairs[row], weight_pairs[column],
                              accumulators[row][column], false);
          }
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
        output[output_row * n + output_column] =
            sllm_phase78_fp8_prefill_bf16_rne(accumulators[row][column] *
                                              activation_scales[output_row] *
                                              weight_scales[output_column]);
      }
    }
  }
}

extern "C" __global__ __launch_bounds__(
    kWorkgroupSize,
    1) void sllm_matmul_fp8_outer_prefill_gfx1030_lds_lut_v1(const uint8_t
                                                                 *const
                                                                     activation,
                                                             const float *const
                                                                 activation_scales,
                                                             const uint8_t
                                                                 *const weight,
                                                             const float *const
                                                                 weight_scales,
                                                             uint16_t
                                                                 *const output,
                                                             const uint64_t m,
                                                             const uint64_t k,
                                                             const uint64_t n) {
  __shared__ __align__(16) uint16_t activation_tile[kTileM][kLdsStride];
  __shared__ __align__(16) uint16_t weight_tile[kTileN][kLdsStride];
  __shared__ __align__(16) uint16_t lds_lut[kLutStorageEntries];
  sllm_phase78_fp8_prefill_64x64_k32(activation, activation_scales, weight,
                                     weight_scales, output, m, k, n,
                                     activation_tile, weight_tile, lds_lut);
}

} // namespace sllm_phase78_fp8_prefill_lds_lut
