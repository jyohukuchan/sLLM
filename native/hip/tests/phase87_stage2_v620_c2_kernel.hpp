#ifndef SLLM_PHASE87_STAGE2_V620_C2_KERNEL_HPP
#define SLLM_PHASE87_STAGE2_V620_C2_KERNEL_HPP

// WU-2V C2 test-only kernel.  The production control remains ID82 in the
// linked lowp archive.  This candidate stages the M=1 activation row into
// shared FP16 once per workgroup, then retains ID82's four-column weight LUT,
// two-chunk prefetch, FP32 dot order, reduction, and BF16 epilogue.

#include <lowp/detail/bf16_helpers.inc>
#include <lowp/detail/low_precision_block_codec.hpp>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace phase87_stage2_v620_c2 {

constexpr uint64_t kK = UINT64_C(6144);
constexpr uint64_t kN = UINT64_C(5120);
constexpr uint32_t kThreads = 256U;
constexpr uint32_t kWave = 32U;
constexpr uint32_t kColumnsPerWave = 4U;
constexpr uint32_t kWaves = kThreads / kWave;
constexpr uint32_t kGroups = 12U;
constexpr uint32_t kPrefetchChunks = 2U;
constexpr uint32_t kStagedBytes = static_cast<uint32_t>(kK * sizeof(uint16_t));

// Exact ID82 FP8 E4M3FN -> FP16 bit table.  The candidate uses it during the
// single activation staging pass and again for every weight dword.
__device__ __constant__ uint16_t kLut[256U] = {
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

__device__ __forceinline__ uint32_t lut_slot(const uint32_t code) noexcept {
  return code + (code >> 5U);
}

__device__ __forceinline__ uint32_t
lut_pair(const uint32_t packed, const uint16_t *const lut) noexcept {
  const uint8_t first = static_cast<uint8_t>(packed & UINT32_C(0xff));
  const uint8_t second = static_cast<uint8_t>((packed >> 8U) & UINT32_C(0xff));
  return static_cast<uint32_t>(lut[lut_slot(first)]) |
         (static_cast<uint32_t>(lut[lut_slot(second)]) << 16U);
}

__device__ __forceinline__ __half2 packed_half2(const uint32_t bits) noexcept {
  return *reinterpret_cast<const __half2 *>(&bits);
}

__global__ __launch_bounds__(kThreads, 1) void candidate_kernel(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  __shared__ __align__(16) uint16_t activation_fp16[kK];
  __shared__ __align__(16) uint16_t lut[272U];
  lut[lut_slot(threadIdx.x)] = kLut[threadIdx.x];
  __syncthreads();
  if (m != 1U || k != kK || n != kN)
    return;

  // Decode the complete activation row once for the whole CTA.  The stored
  // bits are exactly the ID82 LUT result, so the later dot order is unchanged.
  for (uint64_t index = static_cast<uint64_t>(threadIdx.x) * 4U; index < kK;
       index += static_cast<uint64_t>(kThreads) * 4U) {
    const uint32_t packed = __builtin_nontemporal_load(
        reinterpret_cast<const uint32_t *>(activation + index));
    auto *const destination =
        reinterpret_cast<uint32_t *>(activation_fp16 + index);
    destination[0] = lut_pair(packed, lut);
    destination[1] = lut_pair(packed >> 16U, lut);
  }
  __syncthreads();

  const uint32_t lane = threadIdx.x & (kWave - 1U);
  const uint32_t wave = threadIdx.x >> 5U;
  const uint64_t column_base =
      (static_cast<uint64_t>(blockIdx.x) * kWaves + wave) * kColumnsPerWave;
  float accumulators[kColumnsPerWave] = {};
  const auto *const staged = reinterpret_cast<const __half2 *>(activation_fp16);

#pragma unroll 1
  for (uint32_t group = 0U; group < kGroups; ++group) {
    const uint32_t base = lane + group * kWave * kPrefetchChunks;
    __half2 activation_first_low[kPrefetchChunks];
    __half2 activation_first_high[kPrefetchChunks];
    __half2 activation_second_low[kPrefetchChunks];
    __half2 activation_second_high[kPrefetchChunks];
    uint32_t weight_first[kPrefetchChunks][kColumnsPerWave];
    uint32_t weight_second[kPrefetchChunks][kColumnsPerWave];
#pragma unroll
    for (uint32_t chunk = 0U; chunk < kPrefetchChunks; ++chunk) {
      const uint32_t iteration = base + chunk * kWave;
      const uint64_t pair_index = static_cast<uint64_t>(iteration) * 4U;
      activation_first_low[chunk] = staged[pair_index];
      activation_first_high[chunk] = staged[pair_index + 1U];
      activation_second_low[chunk] = staged[pair_index + 2U];
      activation_second_high[chunk] = staged[pair_index + 3U];
#pragma unroll
      for (uint32_t local_column = 0U; local_column < kColumnsPerWave;
           ++local_column) {
        const uint64_t column = column_base + local_column;
        const auto *const column_dwords =
            reinterpret_cast<const uint32_t *>(weight + column * kK);
        weight_first[chunk][local_column] =
            __builtin_nontemporal_load(column_dwords + iteration * 2U);
        weight_second[chunk][local_column] =
            __builtin_nontemporal_load(column_dwords + iteration * 2U + 1U);
      }
    }
#pragma unroll
    for (uint32_t chunk = 0U; chunk < kPrefetchChunks; ++chunk) {
#pragma unroll
      for (uint32_t local_column = 0U; local_column < kColumnsPerWave;
           ++local_column) {
        const __half2 first_low =
            packed_half2(lut_pair(weight_first[chunk][local_column], lut));
        const __half2 first_high = packed_half2(
            lut_pair(weight_first[chunk][local_column] >> 16U, lut));
        const __half2 second_low =
            packed_half2(lut_pair(weight_second[chunk][local_column], lut));
        const __half2 second_high = packed_half2(
            lut_pair(weight_second[chunk][local_column] >> 16U, lut));
        accumulators[local_column] =
            amd_mixed_dot(activation_first_low[chunk], first_low,
                          accumulators[local_column], false);
        accumulators[local_column] =
            amd_mixed_dot(activation_first_high[chunk], first_high,
                          accumulators[local_column], false);
        accumulators[local_column] =
            amd_mixed_dot(activation_second_low[chunk], second_low,
                          accumulators[local_column], false);
        accumulators[local_column] =
            amd_mixed_dot(activation_second_high[chunk], second_high,
                          accumulators[local_column], false);
      }
    }
  }

#pragma unroll
  for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U) {
#pragma unroll
    for (uint32_t local_column = 0U; local_column < kColumnsPerWave;
         ++local_column) {
      accumulators[local_column] +=
          __shfl_down(accumulators[local_column], offset, kWave);
    }
  }
  if (lane == 0U) {
    const float activation_scale = activation_scales[0];
#pragma unroll
    for (uint32_t local_column = 0U; local_column < kColumnsPerWave;
         ++local_column) {
      const uint64_t column = column_base + local_column;
      output[column] =
          float_to_bf16_rne_bits(accumulators[local_column] * activation_scale *
                                 weight_scales[column]);
    }
  }
}

} // namespace phase87_stage2_v620_c2

#endif // SLLM_PHASE87_STAGE2_V620_C2_KERNEL_HPP
