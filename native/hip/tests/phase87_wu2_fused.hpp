// Phase 87 WU2 C2 scratch kernel.
//
// This is a standalone fused BF16 -> FP8 activation quantization and M=1..3
// FP8 GEMV candidate for the gfx1201 W8A8 projection comparison.  It is
// intentionally kept in the phase87 test directory; the WU2 driver includes
// this header directly and production dispatch is owned by the parent task.
//
// Input layout follows the existing outer-scale FP8 contract:
//   activation: [M,K] row-major BF16
//   weight:     [N,K] row-major E4M3FN bytes
//   weight_scale: [N] FP32 outer scale
//   output:     [M,N] row-major BF16
//
// A workgroup owns one activation row and eight adjacent output columns.  It
// computes the same row amax, scale, and value encoding as
// sllm_matmul_bf16_to_fp8_outer_v2, keeps the encoded row in LDS, and then
// performs the GEMV.  The diagnostic pointer is optional.  When non-null,
// block x=0 publishes one row as K encoded bytes followed by the four little-
// endian bytes of the FP32 scale.  `diagnostic_stride` is the byte stride
// between rows and must be at least K+4.  Benchmark callers pass nullptr.

#ifndef PHASE87_WU2_FUSED_HPP
#define PHASE87_WU2_FUSED_HPP

#include "../../lowp/include/lowp/detail/low_precision_block_codec.hpp"

#include <cstdint>

namespace phase87_wu2 {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kWaveWidth = 32U;
constexpr uint32_t kWaveCount = kThreads / kWaveWidth;
constexpr uint32_t kColumnsPerBlock = kWaveCount;

// Keep the value conversion in one helper so the finite pair path below is
// visibly the same conversion as the production v2 quantizer.  The scalar
// path is needed for odd K and for non-finite values, matching v2's branch.
__device__ __forceinline__ uint8_t
phase87_wu2_fp8_scalar(const float value) noexcept {
  return sllm_lowp::float_to_fp8_native(value, false);
}

__device__ __forceinline__ uint16_t
phase87_wu2_bf16_rne(const float value) noexcept {
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

__device__ __forceinline__ void phase87_wu2_publish_diagnostic(
    uint8_t *const diagnostic, const uint64_t diagnostic_stride,
    const uint64_t row, const uint64_t k, const uint8_t *const encoded,
    const float scale) noexcept {
  if (diagnostic == nullptr || diagnostic_stride < k + 4U) {
    return;
  }
  uint8_t *const row_diagnostic = diagnostic + row * diagnostic_stride;
  for (uint64_t index = threadIdx.x; index < k; index += blockDim.x) {
    row_diagnostic[index] = encoded[index];
  }
  const uint32_t scale_bits = __float_as_uint(scale);
  if (threadIdx.x < 4U) {
    row_diagnostic[k + threadIdx.x] = static_cast<uint8_t>(
        scale_bits >> (static_cast<uint32_t>(threadIdx.x) * 8U));
  }
}

template <bool NativeFp8Dot>
__device__ __forceinline__ float phase87_wu2_dot4(const uint32_t lhs,
                                                  const uint32_t rhs,
                                                  float accumulator) noexcept {
#if defined(__gfx1201__) && __has_builtin(__builtin_amdgcn_dot4_f32_fp8_fp8)
  if constexpr (NativeFp8Dot) {
    return __builtin_amdgcn_dot4_f32_fp8_fp8(lhs, rhs, accumulator);
  }
#else
  (void)NativeFp8Dot;
#endif
  // gfx1030 has no FP8 dot4 builtin.  Keep this fallback scalar and exact in
  // FP32 so it can be used for comparative correctness runs on gfx1030.
  for (uint32_t index = 0U; index < 4U; ++index) {
    const uint8_t lhs_code = static_cast<uint8_t>(lhs >> (index * 8U));
    const uint8_t rhs_code = static_cast<uint8_t>(rhs >> (index * 8U));
    accumulator =
        fmaf(sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode(lhs_code),
             sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode(rhs_code),
             accumulator);
  }
  return accumulator;
}

template <bool NativeFp8Dot>
__global__ __launch_bounds__(kThreads, 1) void phase87_wu2_fused_gemv(
    const uint16_t *const activation, const uint8_t *const weight,
    const float *const weight_scales, uint16_t *const output, const uint64_t m,
    const uint64_t k, const uint64_t n, uint8_t *const diagnostic,
    const uint64_t diagnostic_stride) {
  // One block owns one row and eight adjacent output columns.  The dynamic
  // LDS allocation is exactly K bytes; callers bound K to the WU2 contract
  // (K <= 17408), which is comfortably below the per-block LDS limit.
  extern __shared__ uint8_t activation_fp8[];
  const uint64_t row = static_cast<uint64_t>(blockIdx.y);
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * kColumnsPerBlock;
  if (row >= m || column_base >= n || k == 0U || k > UINT64_C(17408)) {
    return;
  }

  // This is the same reduction topology and operation order as v2.  Every
  // output block repeats the scan because the candidate deliberately removes
  // the independent activation-quantization launch.
  const uint32_t lane = threadIdx.x & (kWaveWidth - 1U);
  const uint32_t wave = threadIdx.x / kWaveWidth;
  float maximum = 0.0F;
  const uint64_t activation_offset = row * k;
  for (uint64_t column = threadIdx.x; column < k; column += blockDim.x) {
    const float value = __uint_as_float(
        static_cast<uint32_t>(activation[activation_offset + column]) << 16U);
    maximum = fmaxf(maximum, fabsf(value));
  }
#pragma unroll
  for (uint32_t offset = kWaveWidth / 2U; offset != 0U; offset >>= 1U) {
    maximum = fmaxf(maximum, __shfl_down(maximum, offset, kWaveWidth));
  }
  __shared__ float wave_maxima[kWaveCount];
  __shared__ float shared_scale;
  if (lane == 0U) {
    wave_maxima[wave] = maximum;
  }
  __syncthreads();
  if (wave == 0U) {
    maximum = lane < kWaveCount ? wave_maxima[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = kWaveWidth / 2U; offset != 0U; offset >>= 1U) {
      maximum = fmaxf(maximum, __shfl_down(maximum, offset, kWaveWidth));
    }
    if (lane == 0U) {
      shared_scale = maximum == 0.0F ? 1.0F : maximum / 448.0F;
    }
  }
  __syncthreads();

  // v2 quantizes after the scale is known.  For even K, finite adjacent
  // values use the packed HIP converter; odd K uses its scalar branch for the
  // entire row.  Dividing by shared_scale before the conversion is deliberate.
  const uint64_t row_offset = row * k;
  if ((k & UINT64_C(1)) == 0U) {
    const uint64_t pairs = k / UINT64_C(2);
    for (uint64_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
      const uint64_t offset = pair * UINT64_C(2);
      const float first =
          __uint_as_float(static_cast<uint32_t>(activation[row_offset + offset])
                          << 16U) /
          shared_scale;
      const float second =
          __uint_as_float(
              static_cast<uint32_t>(activation[row_offset + offset + 1U])
              << 16U) /
          shared_scale;
      uint8_t *const destination = activation_fp8 + offset;
      if (isfinite(first) && isfinite(second)) {
        const uint16_t packed = __hip_cvt_float2_to_fp8x2(
            make_float2(first, second), __HIP_SATFINITE, __HIP_E4M3);
        *reinterpret_cast<uint16_t *>(destination) = packed;
      } else {
        destination[0] = phase87_wu2_fp8_scalar(first);
        destination[1] = phase87_wu2_fp8_scalar(second);
      }
    }
  } else {
    for (uint64_t column = threadIdx.x; column < k; column += blockDim.x) {
      const float value =
          __uint_as_float(static_cast<uint32_t>(activation[row_offset + column])
                          << 16U) /
          shared_scale;
      activation_fp8[column] = phase87_wu2_fp8_scalar(value);
    }
  }
  __syncthreads();

  if (blockIdx.x == 0U && diagnostic != nullptr) {
    phase87_wu2_publish_diagnostic(diagnostic, diagnostic_stride, row, k,
                                   activation_fp8, shared_scale);
  }
  __syncthreads();

  const uint64_t column = column_base + wave;
  float accumulator = 0.0F;
  for (uint64_t base = static_cast<uint64_t>(lane) * 4U; base < k;
       base += static_cast<uint64_t>(kWaveWidth) * 4U) {
    uint32_t activation_pack = 0U;
    uint32_t valid = 0U;
    for (; valid < 4U && base + valid < k; ++valid) {
      activation_pack |= static_cast<uint32_t>(activation_fp8[base + valid])
                         << (valid * 8U);
    }
    if (column < n) {
      const uint64_t weight_offset = column * k + base;
      uint32_t weight_pack = 0U;
      for (uint32_t index = 0U; index < valid; ++index) {
        weight_pack |= static_cast<uint32_t>(weight[weight_offset + index])
                       << (index * 8U);
      }
      accumulator = phase87_wu2_dot4<NativeFp8Dot>(activation_pack, weight_pack,
                                                   accumulator);
    }
  }

  float reduced = accumulator;
#pragma unroll
  for (uint32_t offset = kWaveWidth / 2U; offset != 0U; offset >>= 1U) {
    reduced += __shfl_down(reduced, offset, kWaveWidth);
  }
  if (lane == 0U && column < n) {
    output[row * n + column] =
        phase87_wu2_bf16_rne(reduced * shared_scale * weight_scales[column]);
  }
}

} // namespace phase87_wu2

#endif // PHASE87_WU2_FUSED_HPP
