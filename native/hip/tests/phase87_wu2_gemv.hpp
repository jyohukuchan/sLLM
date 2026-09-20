#ifndef SLLM_PHASE87_WU2_GEMV_HPP
#define SLLM_PHASE87_WU2_GEMV_HPP

// Phase 87 WU2 C1 scratch candidate.
//
// This is a standalone M=1..3 outer-scale FP8 GEMV candidate for the exact
// gfx1201 native-FP8 lane.  It deliberately does not reuse the gfx1030 ID92
// body: the workgroup has four wave32s (rather than ID92's eight), each wave
// owns four output columns, and the K loop issues four packed dword groups per
// lane before decoding and accumulating them.  The same source is useful for
// a gfx1030 comparison build; low_precision_block_codec.hpp selects its
// software E4M3FN decoder there, while gfx1201 uses two packed-float2 native
// conversions per dword.
//
// Layout contract (matching the current FP8 outer provider):
//   activation: M*K E4M3FN bytes, row-major [M,K]
//   activation_scales: M FP32 values, one dynamic/token scale per row
//   weight: N*K E4M3FN bytes, output-major [N,K]
//   weight_scales: N FP32 values, one outer/channel scale per output column
//   output: M*N BF16 values, row-major [M,N]
//
// The launch geometry for the exported wrappers is grid=((N+15)/16,1,1),
// block=(128,1,1).  K and N need not be aligned: aligned four-byte spans use
// a dword load, while row or tail addresses that are not four-byte aligned use
// bounded byte assembly.  No byte outside the logical K span is read.

#include "../../lowp/include/lowp/detail/low_precision_block_codec.hpp"

#include <hip/hip_runtime.h>
#if defined(__gfx1201__)
#include <hip/hip_ext_ocp.h>
#endif

#include <cstdint>

namespace sllm_phase87_wu2 {

constexpr uint32_t kWaveSize = 32U;
constexpr uint32_t kColumnsPerWave = 4U;
constexpr uint32_t kWavesPerBlock = 4U;
constexpr uint32_t kThreadsPerBlock = kWaveSize * kWavesPerBlock;
constexpr uint32_t kDwordsPerIteration = 4U;
constexpr uint32_t kColumnsPerBlock = kColumnsPerWave * kWavesPerBlock;

// The dword load is guarded by both the logical span and the actual pointer
// alignment.  A non-aligned K makes the next row's base potentially unaligned,
// even when the allocation itself is aligned.
__device__ __forceinline__ uint32_t
load_dword_or_bytes(const uint8_t *const source, const uint64_t byte_offset,
                    const uint64_t logical_bytes) noexcept {
  if (byte_offset >= logical_bytes) {
    return 0U;
  }
  const uint64_t remaining = logical_bytes - byte_offset;
  const uint8_t *const address = source + byte_offset;
  if (remaining >= UINT64_C(4) && (reinterpret_cast<uintptr_t>(address) &
                                   static_cast<uintptr_t>(3U)) == 0U) {
    return __builtin_nontemporal_load(
        reinterpret_cast<const uint32_t *>(address));
  }

  uint32_t packed = 0U;
#pragma unroll
  for (uint32_t byte = 0U; byte < 4U; ++byte) {
    if (byte_offset + byte < logical_bytes) {
      packed |= static_cast<uint32_t>(source[byte_offset + byte])
                << (byte * 8U);
    }
  }
  return packed;
}

// The scalar entrypoint is retained for the gfx1030 comparison build.  On
// gfx1201 the packed helper below uses the vendor OCP wrapper's underlying
// __builtin_amdgcn_cvt_pk_f32_fp8 instead, converting two adjacent FP8 bytes
// to two FP32 values in one instruction.
__device__ __forceinline__ float decode_e4m3fn(const uint8_t code) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode(code);
}

__device__ __forceinline__ void decode_e4m3fnx4(const uint32_t packed,
                                                float output[4U]) noexcept {
#if defined(__gfx1201__)
  // The low and high uint16 lanes each contain two adjacent E4M3FN bytes.
  // The native gfx1201 packed conversion returns an ext_vector_type(2)
  // (`__amd_floatx2_storage_t`), so this emits two packed conversions per
  // dword while keeping the software path entirely separate.
  const __amd_floatx2_storage_t low = __builtin_amdgcn_cvt_pk_f32_fp8(
      static_cast<__amd_fp8x2_storage_t>(packed & UINT32_C(0xffff)), false);
  const __amd_floatx2_storage_t high = __builtin_amdgcn_cvt_pk_f32_fp8(
      static_cast<__amd_fp8x2_storage_t>(packed >> 16U), false);
  output[0] = low[0];
  output[1] = low[1];
  output[2] = high[0];
  output[3] = high[1];
#else
#pragma unroll
  for (uint32_t byte = 0U; byte < 4U; ++byte) {
    output[byte] = decode_e4m3fn(static_cast<uint8_t>(packed >> (byte * 8U)));
  }
#endif
}

__device__ __forceinline__ uint16_t
f32_to_bf16_rne(const float value) noexcept {
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

template <uint32_t Rows, uint32_t DwordsPerIteration = kDwordsPerIteration,
          uint32_t ColumnsPerWave = kColumnsPerWave,
          uint32_t WavesPerBlock = kWavesPerBlock>
__device__ __forceinline__ void
fp8_gemv_c1_body(const uint8_t *const activation,
                 const float *const activation_scales,
                 const uint8_t *const weight, const float *const weight_scales,
                 uint16_t *const output, const uint64_t m, const uint64_t k,
                 const uint64_t n) noexcept {
  static_assert(Rows >= 1U && Rows <= 3U, "WU2 C1 supports M=1..3 only");
  static_assert(DwordsPerIteration == 4U,
                "WU2 C1 keeps one measured four-dword configuration");
  static_assert(ColumnsPerWave == 4U && WavesPerBlock == 4U,
                "WU2 C1 keeps one measured wave/block configuration");
  static_assert(kWaveSize * WavesPerBlock == kThreadsPerBlock,
                "wave/block geometry must match the launcher contract");

  if (m != Rows || n == 0U) {
    return;
  }

  const uint32_t lane = threadIdx.x & (kWaveSize - 1U);
  const uint32_t wave = threadIdx.x / kWaveSize;
  const uint64_t column_base =
      (static_cast<uint64_t>(blockIdx.x) * WavesPerBlock + wave) *
      ColumnsPerWave;
  if (column_base >= n) {
    return;
  }

  float accumulators[Rows][ColumnsPerWave] = {};
  const uint64_t dword_count = k / UINT64_C(4) + (k % UINT64_C(4) != 0U);

  // Each lane owns four independent K dwords per loop turn.  Activation
  // dwords are decoded once per row and reused across the four output columns;
  // weights remain output-major so each column's dword stream is contiguous.
  for (uint64_t dword_base = lane; dword_base < dword_count;
       dword_base += static_cast<uint64_t>(kWaveSize) * DwordsPerIteration) {
    uint32_t activation_packs[Rows][DwordsPerIteration];
    uint32_t weight_packs[ColumnsPerWave][DwordsPerIteration];

#pragma unroll
    for (uint32_t dword = 0U; dword < DwordsPerIteration; ++dword) {
      const uint64_t dword_index =
          dword_base + static_cast<uint64_t>(dword) * kWaveSize;
      const uint64_t byte_offset = dword_index * UINT64_C(4);
#pragma unroll
      for (uint32_t row = 0U; row < Rows; ++row) {
        const uint8_t *const activation_row =
            activation + static_cast<uint64_t>(row) * k;
        activation_packs[row][dword] =
            load_dword_or_bytes(activation_row, byte_offset, k);
      }
#pragma unroll
      for (uint32_t column_offset = 0U; column_offset < ColumnsPerWave;
           ++column_offset) {
        const uint64_t column = column_base + column_offset;
        weight_packs[column_offset][dword] =
            column < n
                ? load_dword_or_bytes(weight + column * k, byte_offset, k)
                : 0U;
      }
    }

    float activation_values[Rows][DwordsPerIteration][4U];
#pragma unroll
    for (uint32_t row = 0U; row < Rows; ++row) {
#pragma unroll
      for (uint32_t dword = 0U; dword < DwordsPerIteration; ++dword) {
        decode_e4m3fnx4(activation_packs[row][dword],
                        activation_values[row][dword]);
      }
    }

#pragma unroll
    for (uint32_t dword = 0U; dword < DwordsPerIteration; ++dword) {
#pragma unroll
      for (uint32_t column_offset = 0U; column_offset < ColumnsPerWave;
           ++column_offset) {
        if (column_base + column_offset >= n) {
          continue;
        }
        float weight_values[4U];
        decode_e4m3fnx4(weight_packs[column_offset][dword], weight_values);
#pragma unroll
        for (uint32_t byte = 0U; byte < 4U; ++byte) {
#pragma unroll
          for (uint32_t row = 0U; row < Rows; ++row) {
            accumulators[row][column_offset] =
                fmaf(activation_values[row][dword][byte], weight_values[byte],
                     accumulators[row][column_offset]);
          }
        }
      }
    }
  }

  // One wave owns each four-column group.  Its lanes partition K, then the
  // fixed wave32 tree produces one FP32 sum per row and output column.
#pragma unroll
  for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
#pragma unroll
    for (uint32_t row = 0U; row < Rows; ++row) {
#pragma unroll
      for (uint32_t column_offset = 0U; column_offset < ColumnsPerWave;
           ++column_offset) {
        accumulators[row][column_offset] +=
            __shfl_down(accumulators[row][column_offset], offset, kWaveSize);
      }
    }
  }

  if (lane == 0U) {
#pragma unroll
    for (uint32_t row = 0U; row < Rows; ++row) {
      const float activation_scale = activation_scales[row];
#pragma unroll
      for (uint32_t column_offset = 0U; column_offset < ColumnsPerWave;
           ++column_offset) {
        const uint64_t column = column_base + column_offset;
        if (column < n) {
          const float scaled = accumulators[row][column_offset] *
                               activation_scale * weight_scales[column];
          output[static_cast<uint64_t>(row) * n + column] =
              f32_to_bf16_rne(scaled);
        }
      }
    }
  }
}

template <uint32_t Rows, uint32_t DwordsPerIteration = kDwordsPerIteration,
          uint32_t ColumnsPerWave = kColumnsPerWave,
          uint32_t WavesPerBlock = kWavesPerBlock>
__global__ __launch_bounds__(kThreadsPerBlock, 1) void fp8_gemv_c1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  fp8_gemv_c1_body<Rows, DwordsPerIteration, ColumnsPerWave, WavesPerBlock>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

// The wrappers keep driver-side launch code independent of template syntax.
// Compile this header for gfx1201 to use native FP8 conversion; compiling the
// same wrappers for gfx1030 supplies the software comparison implementation.
#define SLLM_PHASE87_WU2_DEFINE_GEMV_WRAPPER(NAME, ROWS)                       \
  extern "C" __global__ __launch_bounds__(kThreadsPerBlock, 1) void NAME(      \
      const uint8_t *const activation, const float *const activation_scales,   \
      const uint8_t *const weight, const float *const weight_scales,           \
      uint16_t *const output, const uint64_t m, const uint64_t k,              \
      const uint64_t n) {                                                      \
    fp8_gemv_c1_body<ROWS>(activation, activation_scales, weight,              \
                           weight_scales, output, m, k, n);                    \
  }

SLLM_PHASE87_WU2_DEFINE_GEMV_WRAPPER(sllm_phase87_wu2_fp8_gemv_c1_m1_v1, 1U)
SLLM_PHASE87_WU2_DEFINE_GEMV_WRAPPER(sllm_phase87_wu2_fp8_gemv_c1_m2_v1, 2U)
SLLM_PHASE87_WU2_DEFINE_GEMV_WRAPPER(sllm_phase87_wu2_fp8_gemv_c1_m3_v1, 3U)

#undef SLLM_PHASE87_WU2_DEFINE_GEMV_WRAPPER

} // namespace sllm_phase87_wu2

#endif // SLLM_PHASE87_WU2_GEMV_HPP
