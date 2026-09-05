// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase9-mmvf-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// f5919bf458ef190468b5c329bb293f8a54a1e69c,
// ggml/src/ggml-cuda/mmvf.cu
// The NVFP4 signed-byte lookup additionally adapts llama.cpp's AMD byte
// permutation primitive.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase78-nvfp4-byte-permute-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// 3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70,
// ggml/src/ggml-cuda/vecdotq.cuh
// SPDX-License-Identifier: MIT

#include "low_precision_block_codec.hpp"
#include "matmul_kernel_internal.hpp"

#include <hip/hip_fp8.h>

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1201__)
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>
#define SLLM_MATMUL_HAS_GFX12_ROCWMMA 1
#endif

#include <cstdint>

namespace {

__device__ __forceinline__ float bf16_to_float(const uint16_t value) noexcept {
  return __uint_as_float(static_cast<uint32_t>(value) << 16U);
}

__device__ __forceinline__ uint16_t
float_to_bf16_rne_bits(const float value) noexcept {
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

__device__ __forceinline__ float e4m3fn_to_float(const uint8_t bits) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode(bits);
}

__device__ __forceinline__ uint8_t float_to_e4m3fn(float value) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::encode(value);
}

__device__ __forceinline__ uint8_t
float_to_fp8_native(const float value, const bool fnuz) noexcept {
  return sllm_lowp::float_to_fp8_native(value, fnuz);
}

__device__ __forceinline__ uint8_t float_to_e4m3fnuz(float value) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E4M3FnuZ>::encode(value);
}

__device__ __forceinline__ float e2m1_to_float(const uint8_t bits) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E2M1>::decode(bits);
}

struct Fp8HalfPacks final {
  __half2 first;
  __half2 second;
};

__device__ __forceinline__ Fp8HalfPacks
e4m3fnx4_to_half2_packs(const uint32_t packed) noexcept {
  Fp8HalfPacks result{};
  sllm_lowp::e4m3fnx4_to_half2x2(packed, &result.first, &result.second);
  return result;
}

__device__ __forceinline__ __half2
packed_half2_from_bits(const uint32_t bits) noexcept {
  return *reinterpret_cast<const __half2 *>(&bits);
}

// E2M1 values are exact multiples of 0.5.  Mapping one nibble to value*2
// therefore preserves the NVFP4 inner product exactly while allowing four
// products to use one signed byte dot instruction.  The per-block result is
// multiplied by 0.25 before the E4M3 block scales are applied.
struct E2M1Scaled2Packs final {
  int32_t even;
  int32_t odd;
};

// Adapted from llama.cpp's AMD get_int_from_table_16 byte-permute lookup.
// One packed dword contains eight E2M1 nibbles.  The result splits even and
// odd nibbles into two signed int8x4 packs; their two dot4 results can be added
// because all eight values belong to the same block16 scale domain.
__device__ __forceinline__ E2M1Scaled2Packs
e2m1x8_scaled2_to_i8x4_pair(const uint32_t packed) noexcept {
  constexpr uint32_t table_0_3 = UINT32_C(0x03020100);
  constexpr uint32_t table_4_7 = UINT32_C(0x0c080604);
  constexpr uint32_t table_8_11 = UINT32_C(0xfdfeff00);
  constexpr uint32_t table_12_15 = UINT32_C(0xf4f8fafc);
  constexpr uint32_t low_index_mask = UINT32_C(0x07070707);
  constexpr uint32_t byte_identity = UINT32_C(0x03020100);
  constexpr uint32_t sign_index_mask = UINT32_C(0x08080808);

  const uint32_t even_indices = packed;
  const uint32_t odd_indices = packed >> 4U;
  const uint32_t even_low = __builtin_amdgcn_perm(
      table_4_7, table_0_3, even_indices & low_index_mask);
  const uint32_t odd_low =
      __builtin_amdgcn_perm(table_4_7, table_0_3, odd_indices & low_index_mask);
  const uint32_t even_high = __builtin_amdgcn_perm(
      table_12_15, table_8_11, even_indices & low_index_mask);
  const uint32_t odd_high = __builtin_amdgcn_perm(table_12_15, table_8_11,
                                                  odd_indices & low_index_mask);
  const uint32_t even_select =
      byte_identity | ((even_indices & sign_index_mask) >> 1U);
  const uint32_t odd_select =
      byte_identity | ((odd_indices & sign_index_mask) >> 1U);
  return E2M1Scaled2Packs{
      static_cast<int32_t>(
          __builtin_amdgcn_perm(even_high, even_low, even_select)),
      static_cast<int32_t>(
          __builtin_amdgcn_perm(odd_high, odd_low, odd_select)),
  };
}

// Four packed E2M1 values map exactly to OCP E4M3FN.  Keep the transform in
// integer byte lanes so gfx1201 can feed the resident NVFP4 tiles directly to
// FP8 rocWMMA without materializing a whole-tensor expansion.
#if defined(__gfx1201__)
__device__ __forceinline__ uint32_t
e2m1x4_to_e4m3fn_exact_bits(const uint16_t packed) noexcept {
  const uint32_t lanes =
      (static_cast<uint32_t>(packed) & UINT32_C(0x000f)) |
      ((static_cast<uint32_t>(packed) & UINT32_C(0x00f0)) << 4U) |
      ((static_cast<uint32_t>(packed) & UINT32_C(0x0f00)) << 8U) |
      ((static_cast<uint32_t>(packed) & UINT32_C(0xf000)) << 12U);
  constexpr uint32_t positive_0_3 = UINT32_C(0x3c383000);
  constexpr uint32_t positive_4_7 = UINT32_C(0x4c484440);
  constexpr uint32_t low_index_mask = UINT32_C(0x07070707);
  const uint32_t positive =
      __builtin_amdgcn_perm(positive_4_7, positive_0_3, lanes & low_index_mask);
  return positive | ((lanes & UINT32_C(0x08080808)) << 4U);
}
#endif

__device__ __forceinline__ int32_t signed_dot4(
    const int32_t lhs, const int32_t rhs, const int32_t accumulator) noexcept {
#if __has_builtin(__builtin_amdgcn_sdot4)
  return __builtin_amdgcn_sdot4(lhs, rhs, accumulator, false);
#else
  int32_t result = accumulator;
#pragma unroll
  for (uint32_t index = 0U; index < 4U; ++index) {
    const int32_t left =
        static_cast<int8_t>(static_cast<uint32_t>(lhs) >> (index * 8U));
    const int32_t right =
        static_cast<int8_t>(static_cast<uint32_t>(rhs) >> (index * 8U));
    result += left * right;
  }
  return result;
#endif
}

__device__ __forceinline__ uint8_t float_to_e2m1(float value) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E2M1>::encode(value);
}

__device__ __forceinline__ float e3m2_to_float(const uint8_t raw) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E3M2>::decode(raw);
}

__device__ __forceinline__ uint8_t float_to_e3m2(float value) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E3M2>::encode(value);
}

__device__ __forceinline__ float e8m0_to_float(const uint8_t bits) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E8M0>::decode(bits);
}

__device__ __forceinline__ uint8_t
mxfp4_even_scale_code(const float maximum) noexcept {
  return sllm_lowp::mxfp4_even_scale_code(maximum);
}

__device__ __forceinline__ uint8_t
packed_e3m2_at(const uint8_t *const row, const uint64_t index) noexcept {
  return sllm_lowp::packed_e3m2_at(row, index);
}

} // namespace

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_to_fp8_outer_v1(
    const uint16_t *const activation, uint8_t *const quantized,
    float *const scales, const uint64_t m, const uint64_t k,
    const uint32_t fnuz) {
  const uint64_t row = blockIdx.x;
  if (row >= m) {
    return;
  }
  float maximum = 0.0F;
  for (uint64_t column = threadIdx.x; column < k; column += blockDim.x) {
    maximum =
        fmaxf(maximum, fabsf(bf16_to_float(activation[row * k + column])));
  }
  __shared__ float reductions[256];
  reductions[threadIdx.x] = maximum;
  __syncthreads();
  for (uint32_t offset = 128U; offset != 0U; offset >>= 1U) {
    if (threadIdx.x < offset) {
      reductions[threadIdx.x] =
          fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  const float scale = reductions[0] == 0.0F
                          ? 1.0F
                          : reductions[0] / (fnuz != 0U ? 240.0F : 448.0F);
  if (threadIdx.x == 0U) {
    scales[row] = scale;
  }
  for (uint64_t column = threadIdx.x; column < k; column += blockDim.x) {
    const float value = bf16_to_float(activation[row * k + column]) / scale;
    quantized[row * k + column] =
        fnuz != 0U ? float_to_e4m3fnuz(value) : float_to_e4m3fn(value);
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_to_fp8_outer_v2(
    const uint16_t *const activation, uint8_t *const quantized,
    float *const scales, const uint64_t m, const uint64_t k,
    const uint32_t fnuz) {
  const uint64_t row = blockIdx.x;
  if (row >= m) {
    return;
  }
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t wave_count = 8U;
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  float maximum = 0.0F;
  for (uint64_t column = threadIdx.x; column < k; column += blockDim.x) {
    maximum =
        fmaxf(maximum, fabsf(bf16_to_float(activation[row * k + column])));
  }
#pragma unroll
  for (uint32_t offset = wave_width / 2U; offset != 0U; offset >>= 1U) {
    maximum = fmaxf(maximum, __shfl_down(maximum, offset, wave_width));
  }
  __shared__ float wave_maxima[wave_count];
  __shared__ float shared_scale;
  if (lane == 0U) {
    wave_maxima[wave] = maximum;
  }
  __syncthreads();
  if (wave == 0U) {
    maximum = lane < wave_count ? wave_maxima[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = wave_width / 2U; offset != 0U; offset >>= 1U) {
      maximum = fmaxf(maximum, __shfl_down(maximum, offset, wave_width));
    }
    if (lane == 0U) {
      shared_scale =
          maximum == 0.0F ? 1.0F : maximum / (fnuz != 0U ? 240.0F : 448.0F);
      scales[row] = shared_scale;
    }
  }
  __syncthreads();
  const uint64_t row_offset = row * k;
  if ((k & UINT64_C(1)) == 0U) {
    const uint64_t pairs = k / UINT64_C(2);
    for (uint64_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
      const float first =
          bf16_to_float(activation[row_offset + pair * UINT64_C(2)]) /
          shared_scale;
      const float second =
          bf16_to_float(activation[row_offset + pair * UINT64_C(2) + 1U]) /
          shared_scale;
      uint16_t packed;
      if (isfinite(first) && isfinite(second)) {
        packed = __hip_cvt_float2_to_fp8x2(
            make_float2(first, second), __HIP_SATFINITE,
            fnuz != 0U ? __HIP_E4M3_FNUZ : __HIP_E4M3);
      } else {
        packed = static_cast<uint16_t>(
            static_cast<uint16_t>(float_to_fp8_native(first, fnuz != 0U)) |
            (static_cast<uint16_t>(float_to_fp8_native(second, fnuz != 0U))
             << 8U));
      }
      reinterpret_cast<uint16_t *>(quantized + row_offset)[pair] = packed;
    }
  } else {
    for (uint64_t column = threadIdx.x; column < k; column += blockDim.x) {
      const float value =
          bf16_to_float(activation[row_offset + column]) / shared_scale;
      quantized[row_offset + column] = float_to_fp8_native(value, fnuz != 0U);
    }
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_fp8_outer_emulation_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index >= m * n) {
    return;
  }
  const uint64_t row = index / n;
  const uint64_t column = index - row * n;
  float accumulator = 0.0F;
  for (uint64_t inner = 0U; inner < k; ++inner) {
    accumulator =
        fmaf(e4m3fn_to_float(activation[row * k + inner]),
             e4m3fn_to_float(weight[column * k + inner]), accumulator);
  }
  output[index] = float_to_bf16_rne_bits(accumulator * activation_scales[row] *
                                         weight_scales[column]);
}

// Phase 78 ID70 transient staging primitives. E4M3FN finite values are exactly
// representable as FP16, so this conversion does not add a rounding step. The
// only numerical-order change relative to the software control is the rocBLAS
// FP32 reduction tree between these kernels.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_fp8_e4m3fn_to_fp16_staging_v1(
    const uint8_t *const input, uint16_t *const output,
    const uint64_t element_count) {
  constexpr uint64_t elements_per_thread = 4U;
  const uint64_t index = (static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                          static_cast<uint64_t>(threadIdx.x)) *
                         elements_per_thread;
  if (index >= element_count) {
    return;
  }
  const bool packed_access =
      (reinterpret_cast<uintptr_t>(input) & UINT64_C(3)) == 0U &&
      (reinterpret_cast<uintptr_t>(output) & UINT64_C(3)) == 0U;
  if (packed_access && element_count - index >= elements_per_thread) {
    const uint32_t packed = __builtin_nontemporal_load(
        reinterpret_cast<const uint32_t *>(input + index));
    const sllm_lowp::E4M3FnFp16x4Bits expanded =
        sllm_lowp::e4m3fnx4_to_fp16x2_bits(packed);
    auto *const packed_output = reinterpret_cast<uint32_t *>(output + index);
    packed_output[0] = expanded.low;
    packed_output[1] = expanded.high;
    return;
  }
  for (uint64_t lane = 0U;
       lane < elements_per_thread && index + lane < element_count; ++lane) {
    output[index + lane] = sllm_lowp::e4m3fn_to_fp16_bits_no_table(
        __builtin_nontemporal_load(input + index + lane));
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_fp8_outer_f16scale_epilogue_v1(
    const float *const input, const float *const activation_scales,
    const float *const weight_scales, uint16_t *const output, const uint64_t m,
    const uint64_t n) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < m * n) {
    const uint64_t row = index / n;
    const uint64_t column = index - row * n;
    output[index] = float_to_bf16_rne_bits(
        input[index] * activation_scales[row] * weight_scales[column]);
  }
}

// gfx1030 prefill specialization for the Unsloth outer-vector FP8 recipe.
// Each workgroup computes a 16x16 output tile and reuses the activation and
// weight bytes through a 32-wide K tile.  Scales remain outer vectors: one
// dynamic scale per activation row and one resident scale per weight row.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_fp8_outer_prefill_tiled16_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t tile_m = 16U;
  constexpr uint32_t tile_n = 16U;
  constexpr uint32_t tile_k = 32U;
  __shared__ uint8_t activation_tile[tile_m * tile_k];
  __shared__ uint8_t weight_tile[tile_n * tile_k];

  const uint64_t tile_column = static_cast<uint64_t>(blockIdx.x) * tile_n;
  const uint64_t tile_row = static_cast<uint64_t>(blockIdx.y) * tile_m;
  const uint32_t local_column = threadIdx.x & UINT32_C(15);
  const uint32_t local_row = threadIdx.x >> 4U;
  const uint64_t row = tile_row + local_row;
  const uint64_t column = tile_column + local_column;
  float accumulator = 0.0F;

  for (uint64_t base = 0U; base < k; base += tile_k) {
    const uint32_t local = threadIdx.x;
#pragma unroll
    for (uint32_t offset = 0U; offset < tile_m * tile_k; offset += 256U) {
      const uint32_t tile_index = local + offset;
      if (tile_index < tile_m * tile_k) {
        const uint32_t source_row = tile_index / tile_k;
        const uint32_t source_inner = tile_index & UINT32_C(31);
        const uint64_t global_row = tile_row + source_row;
        const uint64_t global_inner = base + source_inner;
        activation_tile[tile_index] =
            global_row < m && global_inner < k
                ? __builtin_nontemporal_load(activation + global_row * k +
                                             global_inner)
                : UINT8_C(0);
        const uint32_t source_column = tile_index / tile_k;
        const uint64_t global_column = tile_column + source_column;
        weight_tile[tile_index] =
            global_column < n && global_inner < k
                ? __builtin_nontemporal_load(weight + global_column * k +
                                             global_inner)
                : UINT8_C(0);
      }
    }
    __syncthreads();
    if (row < m && column < n) {
#pragma unroll
      for (uint32_t inner = 0U; inner < tile_k; ++inner) {
        const uint64_t global_inner = base + inner;
        if (global_inner < k) {
          accumulator =
              fmaf(e4m3fn_to_float(activation_tile[local_row * tile_k + inner]),
                   e4m3fn_to_float(weight_tile[local_column * tile_k + inner]),
                   accumulator);
        }
      }
    }
    __syncthreads();
  }
  if (row < m && column < n) {
    output[row * n + column] = float_to_bf16_rne_bits(
        accumulator * activation_scales[row] * weight_scales[column]);
  }
}

// Phase 78 ID72 transient NVFP4 staging primitives.  A thread owns one
// block16, issues one naturally aligned packed-value load and one E4M3 scale
// load, then materializes sixteen exactly representable FP16 operands.  The
// global tensor scales stay device-resident and are applied after the shared
// FP16/F32 GEMM so no host synchronization enters the execution path.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_block16_to_fp16_staging_v1(
    const uint8_t *const packed, const uint8_t *const block_scales,
    uint16_t *const output, const uint64_t rows, const uint64_t k) {
  const uint64_t blocks_per_row = k / UINT64_C(16);
  const uint64_t block_index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t block_count = rows * blocks_per_row;
  if (block_index >= block_count) {
    return;
  }
  const uint64_t row = block_index / blocks_per_row;
  const uint64_t block = block_index - row * blocks_per_row;
  const uint64_t packed_offset = row * (k / UINT64_C(2)) + block * UINT64_C(8);
  const uint64_t packed_values = __builtin_nontemporal_load(
      reinterpret_cast<const uint64_t *>(packed + packed_offset));
  const float scale =
      e4m3fn_to_float(__builtin_nontemporal_load(block_scales + block_index));
  const uint64_t output_offset = row * k + block * UINT64_C(16);
#pragma unroll
  for (uint32_t index = 0U; index < 16U; ++index) {
    const uint8_t pair =
        static_cast<uint8_t>(packed_values >> ((index / 2U) * 8U));
    const uint8_t code = (index & 1U) == 0U ? pair & UINT8_C(0x0f) : pair >> 4U;
    const __half half_value = __float2half_rn(e2m1_to_float(code) * scale);
    output[output_offset + index] = static_cast<__half_raw>(half_value).x;
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_tensor_scale_epilogue_v1(
    const float *const input, const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t element_count) {
  __shared__ float tensor_scale;
  if (threadIdx.x == 0U) {
    tensor_scale = weight_tensor_scale[0] * input_tensor_scale[0];
  }
  __syncthreads();
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < element_count) {
    output[index] = float_to_bf16_rne_bits(input[index] * tensor_scale);
  }
}

// Phase 78 ID83: expand one NVFP4 block16 into an OCP E4M3FN byte plane.
// The workspace is context-owned and reused across dispatches; every source
// value is decoded and encoded exactly once.  Clamp before encoding so the
// native FP8 conversion never sees the architecture-specific overflow edge.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_block16_to_fp8_staging_v1(
    const uint8_t *const packed, const uint8_t *const block_scales,
    uint8_t *const output, const uint64_t rows, const uint64_t k) {
  const uint64_t blocks_per_row = k / UINT64_C(16);
  const uint64_t block_index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t block_count = rows * blocks_per_row;
  if (block_index >= block_count) {
    return;
  }
  const uint64_t row = block_index / blocks_per_row;
  const uint64_t block = block_index - row * blocks_per_row;
  const uint64_t packed_offset = row * (k / UINT64_C(2)) + block * UINT64_C(8);
  const uint64_t packed_values = __builtin_nontemporal_load(
      reinterpret_cast<const uint64_t *>(packed + packed_offset));
  const float scale =
      e4m3fn_to_float(__builtin_nontemporal_load(block_scales + block_index));
  const uint64_t output_offset = row * k + block * UINT64_C(16);
  auto *const output_words =
      reinterpret_cast<uint32_t *>(output + output_offset);
#pragma unroll
  for (uint32_t word = 0U; word < 4U; ++word) {
    uint32_t encoded = 0U;
#pragma unroll
    for (uint32_t lane = 0U; lane < 4U; ++lane) {
      const uint32_t index = word * 4U + lane;
      const uint8_t pair =
          static_cast<uint8_t>(packed_values >> ((index / 2U) * 8U));
      const uint8_t code =
          (index & 1U) == 0U ? pair & UINT8_C(0x0f) : pair >> 4U;
      float value = e2m1_to_float(code) * scale;
      value = fmaxf(-448.0F, fminf(448.0F, value));
      encoded |= static_cast<uint32_t>(float_to_e4m3fn(value)) << (lane * 8U);
    }
    output_words[word] = encoded;
  }
}

extern "C" __global__ void sllm_matmul_nvfp4_tensor_scale_product_v1(
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, float *const output) {
  if (threadIdx.x == 0U && blockIdx.x == 0U) {
    output[0] = weight_tensor_scale[0] * input_tensor_scale[0];
  }
}

// gfx1030 matrix-shaped FP8 outer-scale provider.  Each 16x16 logical thread
// tile computes 128x64 outputs while a K32 stage converts resident E4M3 bytes
// to exact FP16 values in LDS.  RDNA2 then uses packed FP16 dot2 instructions;
// the row/channel F32 outer scales are applied once in the epilogue.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_fp8_outer_prefill_gfx1030_half2_128x64_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t tile_m = 128U;
  constexpr uint32_t tile_n = 64U;
  constexpr uint32_t tile_k = 32U;
  constexpr uint32_t lds_stride = tile_k + 2U;
  constexpr uint32_t thread_rows = 16U;
  constexpr uint32_t thread_columns = 16U;
  constexpr uint32_t rows_per_thread = tile_m / thread_rows;
  constexpr uint32_t columns_per_thread = tile_n / thread_columns;
  __shared__ uint16_t activation_tile[tile_m][lds_stride];
  __shared__ uint16_t weight_tile[tile_n][lds_stride];

  const uint64_t column_tiles = (n + tile_n - 1U) / tile_n;
  const uint64_t tile_index = blockIdx.x;
  const uint64_t row_base = (tile_index / column_tiles) * tile_m;
  const uint64_t column_base = (tile_index % column_tiles) * tile_n;
  const uint32_t thread = threadIdx.x;
  const uint32_t local_row = thread >> 4U;
  const uint32_t local_column = thread & UINT32_C(15);
  float accumulators[rows_per_thread][columns_per_thread] = {};

  for (uint64_t base = 0U; base < k; base += tile_k) {
    for (uint32_t index = thread; index < tile_m * tile_k; index += 256U) {
      const uint32_t row = index / tile_k;
      const uint32_t inner = index % tile_k;
      const uint64_t source_row = row_base + row;
      const uint64_t source_inner = base + inner;
      activation_tile[row][inner] =
          source_row < m && source_inner < k
              ? sllm_lowp::e4m3fn_to_fp16_bits(__builtin_nontemporal_load(
                    activation + source_row * k + source_inner))
              : UINT16_C(0);
    }
    for (uint32_t index = thread; index < tile_n * tile_k; index += 256U) {
      const uint32_t column = index / tile_k;
      const uint32_t inner = index % tile_k;
      const uint64_t source_column = column_base + column;
      const uint64_t source_inner = base + inner;
      weight_tile[column][inner] =
          source_column < n && source_inner < k
              ? sllm_lowp::e4m3fn_to_fp16_bits(__builtin_nontemporal_load(
                    weight + source_column * k + source_inner))
              : UINT16_C(0);
    }
    __syncthreads();

#pragma unroll
    for (uint32_t inner = 0U; inner < tile_k; inner += 2U) {
      __half2 activation_pairs[rows_per_thread];
      __half2 weight_pairs[columns_per_thread];
#pragma unroll
      for (uint32_t row = 0U; row < rows_per_thread; ++row) {
        activation_pairs[row] = *reinterpret_cast<const __half2 *>(
            &activation_tile[local_row + row * thread_rows][inner]);
      }
#pragma unroll
      for (uint32_t column = 0U; column < columns_per_thread; ++column) {
        weight_pairs[column] = *reinterpret_cast<const __half2 *>(
            &weight_tile[local_column + column * thread_columns][inner]);
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
    const uint64_t output_row = row_base + local_row + row * thread_rows;
#pragma unroll
    for (uint32_t column = 0U; column < columns_per_thread; ++column) {
      const uint64_t output_column =
          column_base + local_column + column * thread_columns;
      if (output_row < m && output_column < n) {
        output[output_row * n + output_column] = float_to_bf16_rne_bits(
            accumulators[row][column] * activation_scales[output_row] *
            weight_scales[output_column]);
      }
    }
  }
}

// Phase 78 bounded full-tile specialization for ID71. The caller only
// enters this body for M=1024 and the two measured (K,N) pairs, so every
// load, K stage, accumulator, and store is in-bounds. The tile, half2
// accumulation order, scales, and BF16 epilogue remain ID71-identical.
__device__ __forceinline__ void
sllm_matmul_fp8_outer_prefill_gfx1030_half2_64x64_full_tile_body_legacy(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, uint16_t (*const activation_tile)[34U],
    uint16_t (*const weight_tile)[34U]) {
  (void)m;
  constexpr uint32_t tile_m = 64U;
  constexpr uint32_t tile_n = 64U;
  constexpr uint32_t tile_k = 32U;
  constexpr uint32_t thread_rows = 16U;
  constexpr uint32_t thread_columns = 16U;
  constexpr uint32_t rows_per_thread = tile_m / thread_rows;
  constexpr uint32_t columns_per_thread = tile_n / thread_columns;

  const uint64_t column_tiles = n / tile_n;
  const uint64_t tile_index = blockIdx.x;
  const uint64_t row_base = (tile_index / column_tiles) * tile_m;
  const uint64_t column_base = (tile_index % column_tiles) * tile_n;
  const uint32_t thread = threadIdx.x;
  const uint32_t local_row = thread >> 4U;
  const uint32_t local_column = thread & UINT32_C(15);
  float accumulators[rows_per_thread][columns_per_thread] = {};

  for (uint64_t base = 0U; base < k; base += tile_k) {
    constexpr uint32_t values_per_load = 4U;
    constexpr uint32_t loads_per_row = tile_k / values_per_load;
    for (uint32_t index = thread; index < tile_m * loads_per_row;
         index += 256U) {
      const uint32_t row = index / loads_per_row;
      const uint32_t inner = (index % loads_per_row) * values_per_load;
      const uint64_t source_row = row_base + row;
      const uint64_t source_inner = base + inner;
      const uint32_t packed = *reinterpret_cast<const uint32_t *>(
          activation + source_row * k + source_inner);
      const sllm_lowp::E4M3FnFp16x4Bits expanded =
          sllm_lowp::e4m3fnx4_to_fp16x2_bits(packed);
      auto *const packed_output =
          reinterpret_cast<uint32_t *>(&activation_tile[row][inner]);
      packed_output[0] = expanded.low;
      packed_output[1] = expanded.high;
    }
    for (uint32_t index = thread; index < tile_n * loads_per_row;
         index += 256U) {
      const uint32_t column = index / loads_per_row;
      const uint32_t inner = (index % loads_per_row) * values_per_load;
      const uint64_t source_column = column_base + column;
      const uint64_t source_inner = base + inner;
      const uint32_t packed = *reinterpret_cast<const uint32_t *>(
          weight + source_column * k + source_inner);
      const sllm_lowp::E4M3FnFp16x4Bits expanded =
          sllm_lowp::e4m3fnx4_to_fp16x2_bits(packed);
      auto *const packed_output =
          reinterpret_cast<uint32_t *>(&weight_tile[column][inner]);
      packed_output[0] = expanded.low;
      packed_output[1] = expanded.high;
    }
    __syncthreads();

#pragma unroll
    for (uint32_t inner = 0U; inner < tile_k; inner += 2U) {
      __half2 activation_pairs[rows_per_thread];
      __half2 weight_pairs[columns_per_thread];
#pragma unroll
      for (uint32_t row = 0U; row < rows_per_thread; ++row) {
        activation_pairs[row] = *reinterpret_cast<const __half2 *>(
            &activation_tile[local_row + row * thread_rows][inner]);
      }
#pragma unroll
      for (uint32_t column = 0U; column < columns_per_thread; ++column) {
        weight_pairs[column] = *reinterpret_cast<const __half2 *>(
            &weight_tile[local_column + column * thread_columns][inner]);
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
    const uint64_t output_row = row_base + local_row + row * thread_rows;
#pragma unroll
    for (uint32_t column = 0U; column < columns_per_thread; ++column) {
      const uint64_t output_column =
          column_base + local_column + column * thread_columns;
      output[output_row * n + output_column] = float_to_bf16_rne_bits(
          accumulators[row][column] * activation_scales[output_row] *
          weight_scales[output_column]);
    }
  }
}

// Phase 78 load64 path for the exact full-tile shapes.  Each thread
// owns two adjacent four-byte FP8 chunks in one row, so one aligned 64-bit
// ingress load replaces the two 32-bit loads used by the legacy body.  The
// existing four-byte conversion helper is applied independently to the low
// and high words, preserving the FP8-to-FP16 bits and all dot/epilogue order.
__device__ __forceinline__ void
sllm_matmul_fp8_outer_prefill_gfx1030_half2_64x64_full_tile_body_load64(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, uint16_t (*const activation_tile)[34U],
    uint16_t (*const weight_tile)[34U]) {
  (void)m;
  constexpr uint32_t tile_m = 64U;
  constexpr uint32_t tile_n = 64U;
  constexpr uint32_t tile_k = 32U;
  constexpr uint32_t values_per_load = 4U;
  constexpr uint32_t loads_per_row = tile_k / values_per_load;
  constexpr uint32_t thread_rows = 16U;
  constexpr uint32_t thread_columns = 16U;
  constexpr uint32_t rows_per_thread = tile_m / thread_rows;
  constexpr uint32_t columns_per_thread = tile_n / thread_columns;

  const uint64_t column_tiles = n / tile_n;
  const uint64_t tile_index = blockIdx.x;
  const uint64_t row_base = (tile_index / column_tiles) * tile_m;
  const uint64_t column_base = (tile_index % column_tiles) * tile_n;
  const uint32_t thread = threadIdx.x;
  const uint32_t load_index = thread * 2U;
  const uint32_t local_load_row = load_index / loads_per_row;
  const uint32_t local_load_inner =
      (load_index % loads_per_row) * values_per_load;
  const uint32_t local_row = thread >> 4U;
  const uint32_t local_column = thread & UINT32_C(15);
  float accumulators[rows_per_thread][columns_per_thread] = {};

  for (uint64_t base = 0U; base < k; base += tile_k) {
    const uint8_t *const activation_source =
        activation + (row_base + local_load_row) * k + base + local_load_inner;
    const uint8_t *const weight_source =
        weight + (column_base + local_load_row) * k + base + local_load_inner;
    const uint64_t packed_activation =
        *reinterpret_cast<const uint64_t *>(activation_source);
    const uint64_t packed_weight =
        *reinterpret_cast<const uint64_t *>(weight_source);
    const sllm_lowp::E4M3FnFp16x4Bits activation_low =
        sllm_lowp::e4m3fnx4_to_fp16x2_bits(
            static_cast<uint32_t>(packed_activation));
    const sllm_lowp::E4M3FnFp16x4Bits activation_high =
        sllm_lowp::e4m3fnx4_to_fp16x2_bits(
            static_cast<uint32_t>(packed_activation >> 32U));
    const sllm_lowp::E4M3FnFp16x4Bits weight_low =
        sllm_lowp::e4m3fnx4_to_fp16x2_bits(
            static_cast<uint32_t>(packed_weight));
    const sllm_lowp::E4M3FnFp16x4Bits weight_high =
        sllm_lowp::e4m3fnx4_to_fp16x2_bits(
            static_cast<uint32_t>(packed_weight >> 32U));
    auto *const activation_output = reinterpret_cast<uint32_t *>(
        &activation_tile[local_load_row][local_load_inner]);
    activation_output[0] = activation_low.low;
    activation_output[1] = activation_low.high;
    activation_output[2] = activation_high.low;
    activation_output[3] = activation_high.high;
    auto *const weight_output = reinterpret_cast<uint32_t *>(
        &weight_tile[local_load_row][local_load_inner]);
    weight_output[0] = weight_low.low;
    weight_output[1] = weight_low.high;
    weight_output[2] = weight_high.low;
    weight_output[3] = weight_high.high;
    __syncthreads();

#pragma unroll
    for (uint32_t inner = 0U; inner < tile_k; inner += 2U) {
      __half2 activation_pairs[rows_per_thread];
      __half2 weight_pairs[columns_per_thread];
#pragma unroll
      for (uint32_t row = 0U; row < rows_per_thread; ++row) {
        activation_pairs[row] = *reinterpret_cast<const __half2 *>(
            &activation_tile[local_row + row * thread_rows][inner]);
      }
#pragma unroll
      for (uint32_t column = 0U; column < columns_per_thread; ++column) {
        weight_pairs[column] = *reinterpret_cast<const __half2 *>(
            &weight_tile[local_column + column * thread_columns][inner]);
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
    const uint64_t output_row = row_base + local_row + row * thread_rows;
#pragma unroll
    for (uint32_t column = 0U; column < columns_per_thread; ++column) {
      const uint64_t output_column =
          column_base + local_column + column * thread_columns;
      output[output_row * n + output_column] = float_to_bf16_rne_bits(
          accumulators[row][column] * activation_scales[output_row] *
          weight_scales[output_column]);
    }
  }
}

// Phase 78 ID71 keeps the exact ID63 arithmetic but halves the row tile. The
// resulting four-by-four accumulator footprint reduces VGPR pressure while
// retaining K32 FP8-to-FP16 LDS staging and FP32 dot2 accumulation. Bounds on
// every stage and store preserve the outer-vector contract for M/K/N tails.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_fp8_outer_prefill_gfx1030_half2_64x64_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t tile_m = 64U;
  constexpr uint32_t tile_n = 64U;
  constexpr uint32_t tile_k = 32U;
  constexpr uint32_t lds_stride = tile_k + 2U;
  constexpr uint32_t thread_rows = 16U;
  constexpr uint32_t thread_columns = 16U;
  constexpr uint32_t rows_per_thread = tile_m / thread_rows;
  constexpr uint32_t columns_per_thread = tile_n / thread_columns;
  __shared__ __align__(4) uint16_t activation_tile[tile_m][lds_stride];
  __shared__ __align__(4) uint16_t weight_tile[tile_n][lds_stride];

  const bool use_full_tile_fast_path =
      m == UINT64_C(1024) && ((k == UINT64_C(6144) && n == UINT64_C(5120)) ||
                              (k == UINT64_C(5120) && n == UINT64_C(10240)));
  const bool aligned_load64 =
      (reinterpret_cast<uintptr_t>(activation) & static_cast<uintptr_t>(7U)) ==
          0U &&
      (reinterpret_cast<uintptr_t>(weight) & static_cast<uintptr_t>(7U)) == 0U;
  if (use_full_tile_fast_path && aligned_load64) {
    sllm_matmul_fp8_outer_prefill_gfx1030_half2_64x64_full_tile_body_load64(
        activation, activation_scales, weight, weight_scales, output, m, k, n,
        activation_tile, weight_tile);
    return;
  }
  if (use_full_tile_fast_path) {
    sllm_matmul_fp8_outer_prefill_gfx1030_half2_64x64_full_tile_body_legacy(
        activation, activation_scales, weight, weight_scales, output, m, k, n,
        activation_tile, weight_tile);
    return;
  }

  const uint64_t column_tiles = (n + tile_n - 1U) / tile_n;
  const uint64_t tile_index = blockIdx.x;
  const uint64_t row_base = (tile_index / column_tiles) * tile_m;
  const uint64_t column_base = (tile_index % column_tiles) * tile_n;
  const uint32_t thread = threadIdx.x;
  const uint32_t local_row = thread >> 4U;
  const uint32_t local_column = thread & UINT32_C(15);
  float accumulators[rows_per_thread][columns_per_thread] = {};

  for (uint64_t base = 0U; base < k; base += tile_k) {
    constexpr uint32_t values_per_load = 4U;
    constexpr uint32_t loads_per_row = tile_k / values_per_load;
    for (uint32_t index = thread; index < tile_m * loads_per_row;
         index += 256U) {
      const uint32_t row = index / loads_per_row;
      const uint32_t inner = (index % loads_per_row) * values_per_load;
      const uint64_t source_row = row_base + row;
      const uint64_t source_inner = base + inner;
      if ((k % values_per_load) == 0U && source_row < m &&
          source_inner + values_per_load <= k) {
        const uint32_t packed = *reinterpret_cast<const uint32_t *>(
            activation + source_row * k + source_inner);
        const sllm_lowp::E4M3FnFp16x4Bits expanded =
            sllm_lowp::e4m3fnx4_to_fp16x2_bits(packed);
        auto *const packed_output =
            reinterpret_cast<uint32_t *>(&activation_tile[row][inner]);
        packed_output[0] = expanded.low;
        packed_output[1] = expanded.high;
      } else {
#pragma unroll
        for (uint32_t lane = 0U; lane < values_per_load; ++lane) {
          activation_tile[row][inner + lane] =
              source_row < m && source_inner + lane < k
                  ? sllm_lowp::e4m3fn_to_fp16_bits(
                        *(activation + source_row * k + source_inner + lane))
                  : UINT16_C(0);
        }
      }
    }
    for (uint32_t index = thread; index < tile_n * loads_per_row;
         index += 256U) {
      const uint32_t column = index / loads_per_row;
      const uint32_t inner = (index % loads_per_row) * values_per_load;
      const uint64_t source_column = column_base + column;
      const uint64_t source_inner = base + inner;
      if ((k % values_per_load) == 0U && source_column < n &&
          source_inner + values_per_load <= k) {
        const uint32_t packed = *reinterpret_cast<const uint32_t *>(
            weight + source_column * k + source_inner);
        const sllm_lowp::E4M3FnFp16x4Bits expanded =
            sllm_lowp::e4m3fnx4_to_fp16x2_bits(packed);
        auto *const packed_output =
            reinterpret_cast<uint32_t *>(&weight_tile[column][inner]);
        packed_output[0] = expanded.low;
        packed_output[1] = expanded.high;
      } else {
#pragma unroll
        for (uint32_t lane = 0U; lane < values_per_load; ++lane) {
          weight_tile[column][inner + lane] =
              source_column < n && source_inner + lane < k
                  ? sllm_lowp::e4m3fn_to_fp16_bits(
                        *(weight + source_column * k + source_inner + lane))
                  : UINT16_C(0);
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (uint32_t inner = 0U; inner < tile_k; inner += 2U) {
      __half2 activation_pairs[rows_per_thread];
      __half2 weight_pairs[columns_per_thread];
#pragma unroll
      for (uint32_t row = 0U; row < rows_per_thread; ++row) {
        activation_pairs[row] = *reinterpret_cast<const __half2 *>(
            &activation_tile[local_row + row * thread_rows][inner]);
      }
#pragma unroll
      for (uint32_t column = 0U; column < columns_per_thread; ++column) {
        weight_pairs[column] = *reinterpret_cast<const __half2 *>(
            &weight_tile[local_column + column * thread_columns][inner]);
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
    const uint64_t output_row = row_base + local_row + row * thread_rows;
#pragma unroll
    for (uint32_t column = 0U; column < columns_per_thread; ++column) {
      const uint64_t output_column =
          column_base + local_column + column * thread_columns;
      if (output_row < m && output_column < n) {
        output[output_row * n + output_column] = float_to_bf16_rne_bits(
            accumulators[row][column] * activation_scales[output_row] *
            weight_scales[output_column]);
      }
    }
  }
}

// gfx1030 M=1 FP8 outer-scale GEMV candidate. A wave owns four adjacent
// output columns. Each lane decodes one activation pair exactly to FP16 once,
// reuses it for all four resident column accumulators, and streams coalesced
// weight pairs from the four row-major weight rows. Eight wave32s therefore
// cover 32 columns per workgroup without LDS or inter-wave synchronization.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_fp8_outer_decode_gfx1030_half2_wave4col32_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t wave_size = 32U;
  constexpr uint32_t columns_per_wave = 4U;
  constexpr uint32_t waves_per_workgroup = 8U;
  static_assert(wave_size * waves_per_workgroup == 256U);

  if (m != 1U) {
    return;
  }
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  const uint64_t column_base =
      (static_cast<uint64_t>(blockIdx.x) * waves_per_workgroup + wave) *
      columns_per_wave;
  if (column_base >= n) {
    return;
  }

  float accumulators[columns_per_wave] = {};
  const uint64_t pair_count = k / UINT64_C(2);
  const auto *const activation_pairs =
      reinterpret_cast<const uint16_t *>(activation);
  for (uint64_t pair = lane; pair < pair_count; pair += wave_size) {
    const uint16_t activation_bits =
        __builtin_nontemporal_load(activation_pairs + pair);
    const __half2 activation_pair = sllm_lowp::e4m3fnx2_to_half2(
        static_cast<uint8_t>(activation_bits),
        static_cast<uint8_t>(activation_bits >> 8U));
#pragma unroll
    for (uint32_t local_column = 0U; local_column < columns_per_wave;
         ++local_column) {
      const uint64_t column = column_base + local_column;
      if (column < n) {
        const auto *const weight_pairs =
            reinterpret_cast<const uint16_t *>(weight + column * k);
        const uint16_t weight_bits =
            __builtin_nontemporal_load(weight_pairs + pair);
        const __half2 weight_pair = sllm_lowp::e4m3fnx2_to_half2(
            static_cast<uint8_t>(weight_bits),
            static_cast<uint8_t>(weight_bits >> 8U));
        accumulators[local_column] = amd_mixed_dot(
            activation_pair, weight_pair, accumulators[local_column], false);
      }
    }
  }

#pragma unroll
  for (uint32_t offset = wave_size / 2U; offset != 0U; offset >>= 1U) {
#pragma unroll
    for (uint32_t local_column = 0U; local_column < columns_per_wave;
         ++local_column) {
      accumulators[local_column] +=
          __shfl_down(accumulators[local_column], offset, wave_size);
    }
  }
  if (lane == 0U) {
    const float activation_scale = activation_scales[0];
#pragma unroll
    for (uint32_t local_column = 0U; local_column < columns_per_wave;
         ++local_column) {
      const uint64_t column = column_base + local_column;
      if (column < n) {
        output[column] =
            float_to_bf16_rne_bits(accumulators[local_column] *
                                   activation_scale * weight_scales[column]);
      }
    }
  }
}

// Phase 78 ID68 gfx1030 FP8 outer-scale GEMV candidate.  The ID66 layout is
// retained, but each lane consumes an eight-value dword chunk: two activation
// dwords and two dwords for each of four adjacent output columns are issued
// before the packed E4M3FN x4 -> two-half2 conversion.  This removes the
// per-pair byte loads while preserving the exact E4M3FN-to-FP16 ingress.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_fp8_outer_decode_gfx1030_dword8_wave4col32_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t wave_size = 32U;
  constexpr uint32_t values_per_dword = 4U;
  constexpr uint32_t dwords_per_iteration = 2U;
  constexpr uint32_t values_per_iteration =
      values_per_dword * dwords_per_iteration;
  constexpr uint32_t columns_per_wave = 4U;
  constexpr uint32_t waves_per_workgroup = 8U;
  static_assert(wave_size * waves_per_workgroup == 256U);

  if (m != 1U || n == 0U || k == 0U || (k % 64U) != 0U) {
    return;
  }
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  const uint64_t column_base =
      (static_cast<uint64_t>(blockIdx.x) * waves_per_workgroup + wave) *
      columns_per_wave;
  if (column_base >= n) {
    return;
  }

  float accumulators[columns_per_wave] = {};
  const uint64_t iteration_count = k / values_per_iteration;
  const auto *const activation_dwords =
      reinterpret_cast<const uint32_t *>(activation);
  for (uint64_t iteration = lane; iteration < iteration_count;
       iteration += wave_size) {
    // Issue all dword loads before doing any conversion or dot product.  K is
    // a multiple of 64, so every source address is naturally dword-aligned.
    const uint32_t activation_first = __builtin_nontemporal_load(
        activation_dwords + iteration * dwords_per_iteration);
    const uint32_t activation_second = __builtin_nontemporal_load(
        activation_dwords + iteration * dwords_per_iteration + 1U);
    uint32_t weight_first[columns_per_wave];
    uint32_t weight_second[columns_per_wave];
#pragma unroll
    for (uint32_t local_column = 0U; local_column < columns_per_wave;
         ++local_column) {
      const uint64_t column = column_base + local_column;
      if (column < n) {
        const auto *const column_dwords =
            reinterpret_cast<const uint32_t *>(weight + column * k);
        weight_first[local_column] = __builtin_nontemporal_load(
            column_dwords + iteration * dwords_per_iteration);
        weight_second[local_column] = __builtin_nontemporal_load(
            column_dwords + iteration * dwords_per_iteration + 1U);
      } else {
        weight_first[local_column] = 0U;
        weight_second[local_column] = 0U;
      }
    }

    __half2 activation_pairs[2];
    __half2 activation_second_pairs[2];
    sllm_lowp::e4m3fnx4_to_half2x2(activation_first, &activation_pairs[0],
                                   &activation_pairs[1]);
    sllm_lowp::e4m3fnx4_to_half2x2(activation_second,
                                   &activation_second_pairs[0],
                                   &activation_second_pairs[1]);
#pragma unroll
    for (uint32_t local_column = 0U; local_column < columns_per_wave;
         ++local_column) {
      if (column_base + local_column < n) {
        __half2 weight_pairs[2];
        sllm_lowp::e4m3fnx4_to_half2x2(weight_first[local_column],
                                       &weight_pairs[0], &weight_pairs[1]);
        accumulators[local_column] =
            amd_mixed_dot(activation_pairs[0], weight_pairs[0],
                          accumulators[local_column], false);
        accumulators[local_column] =
            amd_mixed_dot(activation_pairs[1], weight_pairs[1],
                          accumulators[local_column], false);
        sllm_lowp::e4m3fnx4_to_half2x2(weight_second[local_column],
                                       &weight_pairs[0], &weight_pairs[1]);
        accumulators[local_column] =
            amd_mixed_dot(activation_second_pairs[0], weight_pairs[0],
                          accumulators[local_column], false);
        accumulators[local_column] =
            amd_mixed_dot(activation_second_pairs[1], weight_pairs[1],
                          accumulators[local_column], false);
      }
    }
  }

#pragma unroll
  for (uint32_t offset = wave_size / 2U; offset != 0U; offset >>= 1U) {
#pragma unroll
    for (uint32_t local_column = 0U; local_column < columns_per_wave;
         ++local_column) {
      accumulators[local_column] +=
          __shfl_down(accumulators[local_column], offset, wave_size);
    }
  }
  if (lane == 0U) {
    const float activation_scale = activation_scales[0];
#pragma unroll
    for (uint32_t local_column = 0U; local_column < columns_per_wave;
         ++local_column) {
      const uint64_t column = column_base + local_column;
      if (column < n) {
        output[column] =
            float_to_bf16_rne_bits(accumulators[local_column] *
                                   activation_scale * weight_scales[column]);
      }
    }
  }
}

// Phase 78 ID75/76 gfx1030 FP8 outer-scale GEMV candidates. The activation
// row is expanded once per workgroup into dynamic LDS and reused by all eight
// wave32s. The arithmetic is intentionally identical to ID68: exact E4M3FN
// to FP16 ingress, FP32 dot2 accumulation, outer scales, and BF16 RNE store.
template <uint32_t ColumnsPerWave>
__device__ __forceinline__ void fp8_outer_decode_activation_shared_body(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, uint16_t *const activation_fp16) {
  static_assert(ColumnsPerWave == 4U || ColumnsPerWave == 8U);
  constexpr uint32_t wave_size = 32U;
  constexpr uint32_t waves_per_workgroup = 8U;
  constexpr uint32_t columns_per_workgroup =
      waves_per_workgroup * ColumnsPerWave;
  if (m != 1U || n == 0U || k == 0U || (k % 64U) != 0U) {
    return;
  }
  const uint32_t lane = threadIdx.x & (wave_size - 1U);
  const uint32_t wave = threadIdx.x / wave_size;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * columns_per_workgroup +
      static_cast<uint64_t>(wave) * ColumnsPerWave;

  // K is a multiple of 64 for the exact Qwen3.8 cache. Four uint16 values are
  // written per thread iteration, covering the row without overlap.
  for (uint64_t index = static_cast<uint64_t>(threadIdx.x) * 4U; index < k;
       index += static_cast<uint64_t>(blockDim.x) * 4U) {
    const uint32_t packed = __builtin_nontemporal_load(
        reinterpret_cast<const uint32_t *>(activation + index));
    const sllm_lowp::E4M3FnFp16x4Bits expanded =
        sllm_lowp::e4m3fnx4_to_fp16x2_bits(packed);
    auto *const destination =
        reinterpret_cast<uint32_t *>(activation_fp16 + index);
    destination[0] = expanded.low;
    destination[1] = expanded.high;
  }
  __syncthreads();
  if (column_base >= n) {
    return;
  }

  const uint64_t iteration_count = k / 8U;
  float accumulators[ColumnsPerWave] = {};
  for (uint64_t iteration = lane; iteration < iteration_count;
       iteration += wave_size) {
    const uint64_t index = iteration * 8U;
    const auto *const activation_pairs =
        reinterpret_cast<const __half2 *>(activation_fp16 + index);
    const __half2 activation_first = activation_pairs[0];
    const __half2 activation_second = activation_pairs[1];
    const __half2 activation_third = activation_pairs[2];
    const __half2 activation_fourth = activation_pairs[3];
#pragma unroll
    for (uint32_t local_column = 0U; local_column < ColumnsPerWave;
         ++local_column) {
      const uint64_t column = column_base + local_column;
      if (column < n) {
        const auto *const column_dwords =
            reinterpret_cast<const uint32_t *>(weight + column * k + index);
        const Fp8HalfPacks weight_first =
            e4m3fnx4_to_half2_packs(__builtin_nontemporal_load(column_dwords));
        const Fp8HalfPacks weight_second = e4m3fnx4_to_half2_packs(
            __builtin_nontemporal_load(column_dwords + 1U));
        accumulators[local_column] =
            amd_mixed_dot(activation_first, weight_first.first,
                          accumulators[local_column], false);
        accumulators[local_column] =
            amd_mixed_dot(activation_second, weight_first.second,
                          accumulators[local_column], false);
        accumulators[local_column] =
            amd_mixed_dot(activation_third, weight_second.first,
                          accumulators[local_column], false);
        accumulators[local_column] =
            amd_mixed_dot(activation_fourth, weight_second.second,
                          accumulators[local_column], false);
      }
    }
  }

#pragma unroll
  for (uint32_t offset = wave_size / 2U; offset != 0U; offset >>= 1U) {
#pragma unroll
    for (uint32_t local_column = 0U; local_column < ColumnsPerWave;
         ++local_column) {
      accumulators[local_column] +=
          __shfl_down(accumulators[local_column], offset, wave_size);
    }
  }
  if (lane == 0U) {
    const float activation_scale = activation_scales[0];
#pragma unroll
    for (uint32_t local_column = 0U; local_column < ColumnsPerWave;
         ++local_column) {
      const uint64_t column = column_base + local_column;
      if (column < n) {
        output[column] =
            float_to_bf16_rne_bits(accumulators[local_column] *
                                   activation_scale * weight_scales[column]);
      }
    }
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_fp8_outer_decode_gfx1030_actshared_wave4col32_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  extern __shared__ __align__(16) uint16_t activation_fp16[];
  fp8_outer_decode_activation_shared_body<4U>(activation, activation_scales,
                                              weight, weight_scales, output, m,
                                              k, n, activation_fp16);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_fp8_outer_decode_gfx1030_actshared_wave8col64_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  extern __shared__ __align__(16) uint16_t activation_fp16[];
  fp8_outer_decode_activation_shared_body<8U>(activation, activation_scales,
                                              weight, weight_scales, output, m,
                                              k, n, activation_fp16);
}

// ID82: exact E4M3FN ingress through a resident FP16-bit LUT. The table is
// initialized in the code object so each workgroup can copy it into LDS
// without a per-launch host transfer.
__device__ __constant__ uint16_t
    sllm_fp8_outer_decode_gfx1030_lds_lut_fp16_v1[256U] = {
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

__device__ __forceinline__ uint32_t
fp8_outer_decode_lut_slot(const uint32_t code) noexcept {
  return code + (code >> 5U);
}

__device__ __forceinline__ uint32_t fp8_outer_decode_lut_pair(
    const uint32_t packed, const uint16_t *const lut) noexcept {
  const uint8_t first = static_cast<uint8_t>(packed & UINT32_C(0xff));
  const uint8_t second = static_cast<uint8_t>((packed >> 8U) & UINT32_C(0xff));
  return static_cast<uint32_t>(lut[fp8_outer_decode_lut_slot(first)]) |
         (static_cast<uint32_t>(lut[fp8_outer_decode_lut_slot(second)]) << 16U);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_wave4col32_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  // Eight padding halfwords separate groups of 32 entries. The maximum
  // mapped slot is 262; 272 entries provide aligned trailing slack.
  __shared__ __align__(16) uint16_t lut[272U];
  if (threadIdx.x < 256U) {
    lut[fp8_outer_decode_lut_slot(threadIdx.x)] =
        sllm_fp8_outer_decode_gfx1030_lds_lut_fp16_v1[threadIdx.x];
  }
  __syncthreads();
  constexpr uint32_t wave_size = 32U;
  constexpr uint32_t columns_per_wave = 4U;
  constexpr uint32_t waves_per_workgroup = 8U;
  constexpr uint32_t values_per_iteration = 8U;
  constexpr uint32_t prefetch_chunks = 2U;
  static_assert(wave_size * waves_per_workgroup == 256U);
  if (m != 1U || n == 0U || k == 0U || (k % 64U) != 0U)
    return;
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  const uint64_t column_base =
      (static_cast<uint64_t>(blockIdx.x) * waves_per_workgroup + wave) *
      columns_per_wave;
  if (column_base >= n)
    return;

  float accumulators[columns_per_wave] = {};
  const uint64_t iteration_count = k / values_per_iteration;
  const auto *const activation_dwords =
      reinterpret_cast<const uint32_t *>(activation);
  // Prefetch two independent K chunks before decoding either chunk.  The
  // second pass below remains in ascending chunk order and keeps the same
  // four dot operations as the single-chunk ID82 body.
  for (uint64_t base = lane; base < iteration_count;
       base += static_cast<uint64_t>(wave_size) * prefetch_chunks) {
    uint32_t activation_first[prefetch_chunks];
    uint32_t activation_second[prefetch_chunks];
    uint32_t weight_first[prefetch_chunks][columns_per_wave];
    uint32_t weight_second[prefetch_chunks][columns_per_wave];
#pragma unroll
    for (uint32_t chunk = 0U; chunk < prefetch_chunks; ++chunk) {
      const uint64_t iteration =
          base + static_cast<uint64_t>(chunk) * wave_size;
      if (iteration < iteration_count) {
        activation_first[chunk] =
            __builtin_nontemporal_load(activation_dwords + iteration * 2U);
        activation_second[chunk] =
            __builtin_nontemporal_load(activation_dwords + iteration * 2U + 1U);
#pragma unroll
        for (uint32_t local_column = 0U; local_column < columns_per_wave;
             ++local_column) {
          const uint64_t column = column_base + local_column;
          if (column < n) {
            const auto *const column_dwords =
                reinterpret_cast<const uint32_t *>(weight + column * k);
            weight_first[chunk][local_column] =
                __builtin_nontemporal_load(column_dwords + iteration * 2U);
            weight_second[chunk][local_column] =
                __builtin_nontemporal_load(column_dwords + iteration * 2U + 1U);
          } else {
            weight_first[chunk][local_column] = 0U;
            weight_second[chunk][local_column] = 0U;
          }
        }
      } else {
        activation_first[chunk] = 0U;
        activation_second[chunk] = 0U;
#pragma unroll
        for (uint32_t local_column = 0U; local_column < columns_per_wave;
             ++local_column) {
          weight_first[chunk][local_column] = 0U;
          weight_second[chunk][local_column] = 0U;
        }
      }
    }
#pragma unroll
    for (uint32_t chunk = 0U; chunk < prefetch_chunks; ++chunk) {
      const uint64_t iteration =
          base + static_cast<uint64_t>(chunk) * wave_size;
      if (iteration >= iteration_count)
        continue;
      const __half2 activation_first_low = packed_half2_from_bits(
          fp8_outer_decode_lut_pair(activation_first[chunk], lut));
      const __half2 activation_first_high = packed_half2_from_bits(
          fp8_outer_decode_lut_pair(activation_first[chunk] >> 16U, lut));
      const __half2 activation_second_low = packed_half2_from_bits(
          fp8_outer_decode_lut_pair(activation_second[chunk], lut));
      const __half2 activation_second_high = packed_half2_from_bits(
          fp8_outer_decode_lut_pair(activation_second[chunk] >> 16U, lut));
#pragma unroll
      for (uint32_t local_column = 0U; local_column < columns_per_wave;
           ++local_column) {
        const uint64_t column = column_base + local_column;
        if (column >= n)
          continue;
        const __half2 weight_first_low = packed_half2_from_bits(
            fp8_outer_decode_lut_pair(weight_first[chunk][local_column], lut));
        const __half2 weight_first_high =
            packed_half2_from_bits(fp8_outer_decode_lut_pair(
                weight_first[chunk][local_column] >> 16U, lut));
        const __half2 weight_second_low = packed_half2_from_bits(
            fp8_outer_decode_lut_pair(weight_second[chunk][local_column], lut));
        const __half2 weight_second_high =
            packed_half2_from_bits(fp8_outer_decode_lut_pair(
                weight_second[chunk][local_column] >> 16U, lut));
        accumulators[local_column] =
            amd_mixed_dot(activation_first_low, weight_first_low,
                          accumulators[local_column], false);
        accumulators[local_column] =
            amd_mixed_dot(activation_first_high, weight_first_high,
                          accumulators[local_column], false);
        accumulators[local_column] =
            amd_mixed_dot(activation_second_low, weight_second_low,
                          accumulators[local_column], false);
        accumulators[local_column] =
            amd_mixed_dot(activation_second_high, weight_second_high,
                          accumulators[local_column], false);
      }
    }
  }

#pragma unroll
  for (uint32_t offset = wave_size / 2U; offset != 0U; offset >>= 1U) {
#pragma unroll
    for (uint32_t local_column = 0U; local_column < columns_per_wave;
         ++local_column) {
      accumulators[local_column] +=
          __shfl_down(accumulators[local_column], offset, wave_size);
    }
  }
  if (lane == 0U) {
    const float activation_scale = activation_scales[0];
#pragma unroll
    for (uint32_t local_column = 0U; local_column < columns_per_wave;
         ++local_column) {
      const uint64_t column = column_base + local_column;
      if (column < n) {
        output[column] =
            float_to_bf16_rne_bits(accumulators[local_column] *
                                   activation_scale * weight_scales[column]);
      }
    }
  }
}

// ID82 exact tuple specializations share this rolled-loop body. Each wrapper
// remains a separate C-linkage code object because its high VGPR usage must not
// affect the broad ID82 body or the other exact tuples.
template <uint64_t TupleK, uint64_t TupleN, uint32_t TupleGroups>
__device__ __forceinline__ void fp8_outer_decode_lds_lut_tuple_body(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const uint32_t group_count, uint16_t *const lut) {
  constexpr uint32_t wave_size = 32U;
  constexpr uint32_t columns_per_wave = 4U;
  constexpr uint32_t waves_per_workgroup = 8U;
  constexpr uint32_t values_per_iteration = 8U;
  constexpr uint32_t prefetch_chunks = 2U;
  static_assert(TupleK / values_per_iteration ==
                wave_size * prefetch_chunks * TupleGroups);
  static_assert(TupleN % (waves_per_workgroup * columns_per_wave) == 0U);

  if (m != 1U || k != TupleK || n != TupleN || group_count != TupleGroups) {
    return;
  }

  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  const uint64_t column_base =
      (static_cast<uint64_t>(blockIdx.x) * waves_per_workgroup + wave) *
      columns_per_wave;
  float accumulators[columns_per_wave] = {};
  const auto *const activation_dwords =
      reinterpret_cast<const uint32_t *>(activation);

#pragma unroll 1
  for (uint32_t group = 0U; group < group_count; ++group) {
    const uint32_t base = lane + group * wave_size * prefetch_chunks;
    uint32_t activation_first[prefetch_chunks];
    uint32_t activation_second[prefetch_chunks];
    uint32_t weight_first[prefetch_chunks][columns_per_wave];
    uint32_t weight_second[prefetch_chunks][columns_per_wave];
#pragma unroll
    for (uint32_t chunk = 0U; chunk < prefetch_chunks; ++chunk) {
      const uint32_t iteration = base + chunk * wave_size;
      activation_first[chunk] =
          __builtin_nontemporal_load(activation_dwords + iteration * 2U);
      activation_second[chunk] =
          __builtin_nontemporal_load(activation_dwords + iteration * 2U + 1U);
#pragma unroll
      for (uint32_t local_column = 0U; local_column < columns_per_wave;
           ++local_column) {
        const uint64_t column = column_base + local_column;
        const auto *const column_dwords =
            reinterpret_cast<const uint32_t *>(weight + column * TupleK);
        weight_first[chunk][local_column] =
            __builtin_nontemporal_load(column_dwords + iteration * 2U);
        weight_second[chunk][local_column] =
            __builtin_nontemporal_load(column_dwords + iteration * 2U + 1U);
      }
    }
#pragma unroll
    for (uint32_t chunk = 0U; chunk < prefetch_chunks; ++chunk) {
      const __half2 activation_first_low = packed_half2_from_bits(
          fp8_outer_decode_lut_pair(activation_first[chunk], lut));
      const __half2 activation_first_high = packed_half2_from_bits(
          fp8_outer_decode_lut_pair(activation_first[chunk] >> 16U, lut));
      const __half2 activation_second_low = packed_half2_from_bits(
          fp8_outer_decode_lut_pair(activation_second[chunk], lut));
      const __half2 activation_second_high = packed_half2_from_bits(
          fp8_outer_decode_lut_pair(activation_second[chunk] >> 16U, lut));
#pragma unroll
      for (uint32_t local_column = 0U; local_column < columns_per_wave;
           ++local_column) {
        const __half2 weight_first_low = packed_half2_from_bits(
            fp8_outer_decode_lut_pair(weight_first[chunk][local_column], lut));
        const __half2 weight_first_high =
            packed_half2_from_bits(fp8_outer_decode_lut_pair(
                weight_first[chunk][local_column] >> 16U, lut));
        const __half2 weight_second_low = packed_half2_from_bits(
            fp8_outer_decode_lut_pair(weight_second[chunk][local_column], lut));
        const __half2 weight_second_high =
            packed_half2_from_bits(fp8_outer_decode_lut_pair(
                weight_second[chunk][local_column] >> 16U, lut));
        accumulators[local_column] =
            amd_mixed_dot(activation_first_low, weight_first_low,
                          accumulators[local_column], false);
        accumulators[local_column] =
            amd_mixed_dot(activation_first_high, weight_first_high,
                          accumulators[local_column], false);
        accumulators[local_column] =
            amd_mixed_dot(activation_second_low, weight_second_low,
                          accumulators[local_column], false);
        accumulators[local_column] =
            amd_mixed_dot(activation_second_high, weight_second_high,
                          accumulators[local_column], false);
      }
    }
  }

#pragma unroll
  for (uint32_t offset = wave_size / 2U; offset != 0U; offset >>= 1U) {
#pragma unroll
    for (uint32_t local_column = 0U; local_column < columns_per_wave;
         ++local_column) {
      accumulators[local_column] +=
          __shfl_down(accumulators[local_column], offset, wave_size);
    }
  }
  if (lane == 0U) {
    const float activation_scale = activation_scales[0];
#pragma unroll
    for (uint32_t local_column = 0U; local_column < columns_per_wave;
         ++local_column) {
      const uint64_t column = column_base + local_column;
      output[column] =
          float_to_bf16_rne_bits(accumulators[local_column] * activation_scale *
                                 weight_scales[column]);
    }
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_k5120n17408_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const uint32_t group_count) {
  __shared__ __align__(16) uint16_t lut[272U];
  lut[fp8_outer_decode_lut_slot(threadIdx.x)] =
      sllm_fp8_outer_decode_gfx1030_lds_lut_fp16_v1[threadIdx.x];
  __syncthreads();
  fp8_outer_decode_lds_lut_tuple_body<UINT64_C(5120), UINT64_C(17408), 10U>(
      activation, activation_scales, weight, weight_scales, output, m, k, n,
      group_count, lut);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_k6144n5120_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const uint32_t group_count) {
  __shared__ __align__(16) uint16_t lut[272U];
  lut[fp8_outer_decode_lut_slot(threadIdx.x)] =
      sllm_fp8_outer_decode_gfx1030_lds_lut_fp16_v1[threadIdx.x];
  __syncthreads();
  fp8_outer_decode_lds_lut_tuple_body<UINT64_C(6144), UINT64_C(5120), 12U>(
      activation, activation_scales, weight, weight_scales, output, m, k, n,
      group_count, lut);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_k5120n10240_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const uint32_t group_count) {
  __shared__ __align__(16) uint16_t lut[272U];
  lut[fp8_outer_decode_lut_slot(threadIdx.x)] =
      sllm_fp8_outer_decode_gfx1030_lds_lut_fp16_v1[threadIdx.x];
  __syncthreads();
  fp8_outer_decode_lds_lut_tuple_body<UINT64_C(5120), UINT64_C(10240), 10U>(
      activation, activation_scales, weight, weight_scales, output, m, k, n,
      group_count, lut);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_k5120n6144_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const uint32_t group_count) {
  __shared__ __align__(16) uint16_t lut[272U];
  lut[fp8_outer_decode_lut_slot(threadIdx.x)] =
      sllm_fp8_outer_decode_gfx1030_lds_lut_fp16_v1[threadIdx.x];
  __syncthreads();
  fp8_outer_decode_lds_lut_tuple_body<UINT64_C(5120), UINT64_C(6144), 10U>(
      activation, activation_scales, weight, weight_scales, output, m, k, n,
      group_count, lut);
}
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_to_nvfp4_block16_v1(
    const uint16_t *const activation, uint8_t *const packed_activation,
    uint8_t *const block_scales, const float *const input_tensor_scale,
    const uint64_t m, const uint64_t k) {
  const uint64_t blocks_per_row = (k + UINT64_C(15)) / UINT64_C(16);
  const uint64_t block_index = blockIdx.x;
  if (block_index >= m * blocks_per_row) {
    return;
  }
  const uint64_t row = block_index / blocks_per_row;
  const uint64_t block = block_index - row * blocks_per_row;
  const uint64_t base = block * UINT64_C(16);
  const uint64_t packed_row_bytes = (k + UINT64_C(1)) / UINT64_C(2);
  __shared__ float values[16];
  __shared__ float decoded_scale;
  if (threadIdx.x < 16U) {
    const uint64_t column = base + threadIdx.x;
    values[threadIdx.x] =
        column < k ? bf16_to_float(activation[row * k + column]) : 0.0F;
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    float maximum = 0.0F;
    for (uint32_t index = 0U; index != 16U; ++index) {
      maximum = fmaxf(maximum, fabsf(values[index]));
    }
    const float global = input_tensor_scale[0];
    const float raw_scale =
        maximum == 0.0F || !(global > 0.0F) ? 0.0F : maximum / (6.0F * global);
    const uint8_t encoded_scale = float_to_e4m3fn(raw_scale);
    block_scales[block_index] = encoded_scale;
    decoded_scale = e4m3fn_to_float(encoded_scale) * global;
  }
  __syncthreads();
  if (threadIdx.x < 8U) {
    const uint32_t first = threadIdx.x * 2U;
    const uint64_t first_column = base + first;
    const uint64_t second_column = first_column + 1U;
    if (first_column < k) {
      const uint8_t low = decoded_scale > 0.0F
                              ? float_to_e2m1(values[first] / decoded_scale)
                              : 0U;
      const uint8_t high =
          second_column < k && decoded_scale > 0.0F
              ? float_to_e2m1(values[first + 1U] / decoded_scale)
              : 0U;
      packed_activation[row * packed_row_bytes + first_column / UINT64_C(2)] =
          static_cast<uint8_t>(low | static_cast<uint8_t>(high << 4U));
    }
  }
}

// Opt-in activation quantizer candidate for Phase 78. Eight wave32s process
// eight independent K=16 blocks per workgroup. Unlike the control kernel,
// every wave keeps its block-local values in registers: lane<16> loads BF16,
// the wave reduces its maximum, lane 0 encodes and broadcasts the E4M3 scale,
// and lane<8> writes packed pairs. There is no LDS or barrier in this path.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_to_nvfp4_block16_wave8_v1(
    const uint16_t *const activation, uint8_t *const packed_activation,
    uint8_t *const block_scales, const float *const input_tensor_scale,
    const uint64_t m, const uint64_t k) {
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t blocks_per_workgroup = 8U;
  const uint32_t lane = threadIdx.x & (wave_width - 1U);
  const uint32_t wave = threadIdx.x / wave_width;
  const uint64_t blocks_per_row = (k + UINT64_C(15)) / UINT64_C(16);
  const uint64_t block_index =
      static_cast<uint64_t>(blockIdx.x) * blocks_per_workgroup + wave;
  if (block_index >= m * blocks_per_row) {
    return;
  }
  const uint64_t row = block_index / blocks_per_row;
  const uint64_t block = block_index - row * blocks_per_row;
  const uint64_t base = block * UINT64_C(16);
  const uint64_t packed_row_bytes = (k + UINT64_C(1)) / UINT64_C(2);
  const uint32_t column = lane;
  const uint64_t source_column = base + column;
  const float value = lane < 16U && source_column < k
                          ? bf16_to_float(activation[row * k + source_column])
                          : 0.0F;
  float maximum = lane < 16U ? fabsf(value) : 0.0F;
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    maximum = fmaxf(maximum, __shfl_down(maximum, offset, wave_width));
  }
  const float global = input_tensor_scale[0];
  // Shuffle a full dword. HIP's byte-sized shuffle overload can leave the
  // upper lanes implementation-defined on wave32 targets and produced an
  // incorrect decoded scale in the tail cases.
  uint32_t encoded_scale_bits = 0U;
  if (lane == 0U) {
    const float raw_scale =
        maximum == 0.0F || !(global > 0.0F) ? 0.0F : maximum / (6.0F * global);
    encoded_scale_bits = float_to_e4m3fn(raw_scale);
    block_scales[block_index] = static_cast<uint8_t>(encoded_scale_bits);
  }
  encoded_scale_bits = __shfl(encoded_scale_bits, 0, wave_width);
  const uint8_t encoded_scale = static_cast<uint8_t>(encoded_scale_bits);
  const float decoded_scale = e4m3fn_to_float(encoded_scale) * global;
  // Cross-lane reads must execute under the full wave mask. Masking lanes
  // 8..15 before they serve as shuffle sources returns undefined data.
  const uint32_t pair_lane = lane & 7U;
  const uint32_t shuffled_first = pair_lane * 2U;
  const float first_value =
      __shfl(value, static_cast<int>(shuffled_first), wave_width);
  const float second_value =
      __shfl(value, static_cast<int>(shuffled_first + 1U), wave_width);
  if (lane < 8U) {
    const uint32_t first = lane * 2U;
    const uint64_t first_column = base + first;
    const uint64_t second_column = first_column + 1U;
    const uint8_t low = first_column < k && decoded_scale > 0.0F
                            ? float_to_e2m1(first_value / decoded_scale)
                            : 0U;
    const uint8_t high = second_column < k && decoded_scale > 0.0F
                             ? float_to_e2m1(second_value / decoded_scale)
                             : 0U;
    if (first_column < k) {
      packed_activation[row * packed_row_bytes + first_column / UINT64_C(2)] =
          static_cast<uint8_t>(low | static_cast<uint8_t>(high << 4U));
    }
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_block16_packed_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  const uint64_t output_index = blockIdx.x;
  if (output_index >= m * n) {
    return;
  }
  const uint64_t row = output_index / n;
  const uint64_t column = output_index - row * n;
  const uint64_t blocks_per_row = (k + UINT64_C(15)) / UINT64_C(16);
  const uint64_t packed_activation_row = (k + UINT64_C(1)) / UINT64_C(2);
  float partial = 0.0F;
  for (uint64_t inner = threadIdx.x; inner < k; inner += blockDim.x) {
    const uint8_t activation_pair = __builtin_nontemporal_load(
        packed_activation + row * packed_activation_row + inner / 2U);
    const uint8_t activation_code = (inner & UINT64_C(1)) == 0U
                                        ? activation_pair & UINT8_C(0x0f)
                                        : activation_pair >> 4U;
    const uint64_t weight_index = column * k + inner;
    const uint8_t weight_pair =
        __builtin_nontemporal_load(packed_weight + weight_index / UINT64_C(2));
    const uint8_t weight_code = (weight_index & UINT64_C(1)) == 0U
                                    ? weight_pair & UINT8_C(0x0f)
                                    : weight_pair >> 4U;
    const float activation_scale = e4m3fn_to_float(
        activation_block_scales[row * blocks_per_row + inner / 16U]);
    const float weight_scale = e4m3fn_to_float(
        weight_block_scales[column * blocks_per_row + inner / 16U]);
    partial += e2m1_to_float(activation_code) * activation_scale *
               e2m1_to_float(weight_code) * weight_scale;
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    partial += __shfl_down(partial, offset, 32U);
  }
  __shared__ float wave_sums[8];
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  if (lane == 0U) {
    wave_sums[wave] = partial;
  }
  __syncthreads();
  if (wave == 0U) {
    partial = lane < 8U ? wave_sums[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
      partial += __shfl_down(partial, offset, 32U);
    }
    if (lane == 0U) {
      output[output_index] = float_to_bf16_rne_bits(
          partial * weight_tensor_scale[0] * input_tensor_scale[0]);
    }
  }
}

// Phase 78 ID69: gfx1201 NVFP4 W4A4 FP16-WMMA candidate.  Unlike ID64, each
// packed E2M1 value is decoded and multiplied by its E4M3 block-16 scale
// before it enters the matrix operand tile.  The FP32 WMMA accumulator stays
// resident for every K stage, so no per-stage contribution fill, layout
// transform, or post-MMA scale multiply is required.  Only the tensor scale
// remains in the final BF16 RNE epilogue.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_nvfp4_w4a4_prefill_gfx1201_wmma_f16scale128x64_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
#if defined(__gfx1201__)
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t waves_per_workgroup = 8U;
  constexpr uint32_t tile_m = 16U;
  constexpr uint32_t tile_n = 16U;
  constexpr uint32_t column_tiles = 4U;
  constexpr uint32_t fragment_k = 16U;
  constexpr uint32_t stage_k = 32U;
  constexpr uint32_t scale_blocks_per_stage = stage_k / fragment_k;
  constexpr uint32_t rows_per_workgroup = waves_per_workgroup * tile_m;
  constexpr uint32_t columns_per_workgroup = column_tiles * tile_n;
  constexpr uint32_t tile_values = tile_m * stage_k;
  constexpr uint32_t output_values = tile_m * tile_n;
  constexpr uint32_t activation_tile_values = waves_per_workgroup * tile_values;
  constexpr uint32_t weight_tile_values = column_tiles * tile_values;

  // 12 KiB of FP16 matrix operands plus 1.25 KiB of FP16 block scales.
  // The scale tiles are FP16 because the candidate intentionally absorbs the
  // block scale into the matrix operands at ingress.
  __shared__ __align__(4)
      rocwmma::float16_t activation_tile[activation_tile_values];
  __shared__ __align__(4) rocwmma::float16_t weight_tile[weight_tile_values];
  __shared__ rocwmma::float16_t activation_scale_tile[waves_per_workgroup]
                                                     [tile_m]
                                                     [scale_blocks_per_stage];
  __shared__ rocwmma::float16_t weight_scale_tile[column_tiles * tile_n]
                                                 [scale_blocks_per_stage];
  __shared__ float tensor_scale;

  using AFragment =
      rocwmma::fragment<rocwmma::matrix_a, tile_m, tile_n, fragment_k,
                        rocwmma::float16_t, rocwmma::row_major>;
  using BFragment =
      rocwmma::fragment<rocwmma::matrix_b, tile_m, tile_n, fragment_k,
                        rocwmma::float16_t, rocwmma::col_major>;
  using AccumulatorFragment = rocwmma::fragment<rocwmma::accumulator, tile_m,
                                                tile_n, fragment_k, float>;

  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (wave_width - 1U);
  const uint32_t wave = thread / wave_width;
  const uint64_t row_group_base =
      static_cast<uint64_t>(blockIdx.y) * rows_per_workgroup;
  const uint64_t row_tile_base =
      row_group_base + static_cast<uint64_t>(wave) * tile_m;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * columns_per_workgroup;
  const uint64_t blocks_per_row = k / fragment_k;
  const uint64_t stages =
      (blocks_per_row + scale_blocks_per_stage - 1U) / scale_blocks_per_stage;
  const uint64_t packed_row_bytes = k / 2U;

  if (thread == 0U) {
    tensor_scale = weight_tensor_scale[0] * input_tensor_scale[0];
  }

  AccumulatorFragment contributions[column_tiles];
#pragma unroll
  for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
    rocwmma::fill_fragment(contributions[column_tile], 0.0F);
  }

  for (uint64_t stage = 0U; stage < stages; ++stage) {
    const uint64_t inner_base = stage * stage_k;

    // Load the two block scales needed by this stage once into LDS.  Invalid
    // K blocks and M/N tails receive zero scales and zero matrix operands.
    for (uint32_t index = thread;
         index < waves_per_workgroup * tile_m * scale_blocks_per_stage;
         index += blockDim.x) {
      const uint32_t scale_block = index % scale_blocks_per_stage;
      const uint32_t row_index = index / scale_blocks_per_stage;
      const uint32_t source_wave = row_index / tile_m;
      const uint32_t local_row = row_index - source_wave * tile_m;
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      activation_scale_tile[source_wave][local_row][scale_block] =
          row < m && block < blocks_per_row
              ? static_cast<rocwmma::float16_t>(e4m3fn_to_float(
                    activation_block_scales[row * blocks_per_row + block]))
              : static_cast<rocwmma::float16_t>(0.0F);
    }
    for (uint32_t index = thread;
         index < column_tiles * tile_n * scale_blocks_per_stage;
         index += blockDim.x) {
      const uint32_t scale_block = index % scale_blocks_per_stage;
      const uint32_t local_column = index / scale_blocks_per_stage;
      const uint64_t column = column_base + local_column;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      weight_scale_tile[local_column][scale_block] =
          column < n && block < blocks_per_row
              ? static_cast<rocwmma::float16_t>(e4m3fn_to_float(
                    weight_block_scales[column * blocks_per_row + block]))
              : static_cast<rocwmma::float16_t>(0.0F);
    }
    __syncthreads();

    // Decode directly into the FP16 matrix operand tiles.  The source
    // encoding is two E2M1 nibbles per byte; the block scale is common to all
    // sixteen values in a block, so its FP16 product is exact with respect to
    // the requested ingress contract (up to the deliberate FP16 operand
    // rounding before WMMA).
    for (uint32_t index = thread; index < activation_tile_values;
         index += blockDim.x) {
      const uint32_t source_wave = index / tile_values;
      const uint32_t tile_index = index - source_wave * tile_values;
      const uint32_t local_row = tile_index / stage_k;
      const uint32_t local_inner = tile_index - local_row * stage_k;
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      const uint64_t inner = inner_base + local_inner;
      if (row < m && inner < k) {
        const uint8_t packed = __builtin_nontemporal_load(
            packed_activation + row * packed_row_bytes + inner / 2U);
        const uint8_t code =
            (inner & 1U) == 0U ? packed & UINT8_C(0x0f) : packed >> 4U;
        const uint32_t scale_block = local_inner / fragment_k;
        const float scale = static_cast<float>(
            activation_scale_tile[source_wave][local_row][scale_block]);
        activation_tile[index] =
            static_cast<rocwmma::float16_t>(e2m1_to_float(code) * scale);
      } else {
        activation_tile[index] = static_cast<rocwmma::float16_t>(0.0F);
      }
    }
    for (uint32_t index = thread; index < weight_tile_values;
         index += blockDim.x) {
      const uint32_t column_tile = index / tile_values;
      const uint32_t tile_index = index - column_tile * tile_values;
      const uint32_t local_column = tile_index / stage_k;
      const uint32_t local_inner = tile_index - local_column * stage_k;
      const uint64_t column = column_base +
                              static_cast<uint64_t>(column_tile) * tile_n +
                              local_column;
      const uint64_t inner = inner_base + local_inner;
      if (column < n && inner < k) {
        const uint8_t packed = __builtin_nontemporal_load(
            packed_weight + column * packed_row_bytes + inner / 2U);
        const uint8_t code =
            (inner & 1U) == 0U ? packed & UINT8_C(0x0f) : packed >> 4U;
        const uint32_t scale_block = local_inner / fragment_k;
        const float scale =
            static_cast<float>(weight_scale_tile[column_tile * tile_n +
                                                 local_column][scale_block]);
        weight_tile[index] =
            static_cast<rocwmma::float16_t>(e2m1_to_float(code) * scale);
      } else {
        weight_tile[index] = static_cast<rocwmma::float16_t>(0.0F);
      }
    }
    __syncthreads();

#pragma unroll
    for (uint32_t scale_block = 0U; scale_block < scale_blocks_per_stage;
         ++scale_block) {
      AFragment activation_fragment;
      rocwmma::load_matrix_sync(activation_fragment,
                                activation_tile + wave * tile_values +
                                    scale_block * fragment_k,
                                stage_k);
#pragma unroll
      for (uint32_t column_tile = 0U; column_tile < column_tiles;
           ++column_tile) {
        BFragment weight_fragment;
        rocwmma::load_matrix_sync(weight_fragment,
                                  weight_tile + column_tile * tile_values +
                                      scale_block * fragment_k,
                                  stage_k);
        rocwmma::mma_sync(contributions[column_tile], activation_fragment,
                          weight_fragment, contributions[column_tile]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
    const auto contribution_row_major =
        rocwmma::apply_data_layout<rocwmma::row_major>(
            contributions[column_tile]);
#pragma unroll
    for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
      const uint32_t local_row =
          (lane / tile_n) * (output_values / wave_width) + slot;
      const uint32_t local_column = lane % tile_n;
      const uint64_t row = row_tile_base + local_row;
      const uint64_t column = column_base + column_tile * tile_n + local_column;
      if (row < m && column < n) {
        output[row * n + column] =
            float_to_bf16_rne_bits(contribution_row_major[slot] * tensor_scale);
      }
    }
  }
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

// M=1 decode specialization.  The generic W4A4 kernel launches one workgroup
// per (row,column), which is correct but adds a row-index division and keeps
// the activation stream in every workgroup.  Decode has exactly one row, so a
// workgroup owns one output column and reuses the activation block scales
// while reducing K.  The arithmetic contract is identical to the generic
// kernel; only the launch geometry and address calculation change.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_block16_decode_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  if (m != 1U) {
    return;
  }
  const uint64_t column = blockIdx.x;
  if (column >= n) {
    return;
  }
  const uint64_t blocks_per_row = (k + UINT64_C(15)) / UINT64_C(16);
  float partial = 0.0F;
  for (uint64_t inner = threadIdx.x; inner < k; inner += blockDim.x) {
    const uint8_t activation_pair =
        __builtin_nontemporal_load(packed_activation + inner / 2U);
    const uint8_t activation_code = (inner & UINT64_C(1)) == 0U
                                        ? activation_pair & UINT8_C(0x0f)
                                        : activation_pair >> 4U;
    const uint64_t weight_index = column * k + inner;
    const uint8_t weight_pair =
        __builtin_nontemporal_load(packed_weight + weight_index / UINT64_C(2));
    const uint8_t weight_code = (weight_index & UINT64_C(1)) == 0U
                                    ? weight_pair & UINT8_C(0x0f)
                                    : weight_pair >> 4U;
    const float activation_scale =
        e4m3fn_to_float(activation_block_scales[inner / UINT64_C(16)]);
    const float weight_scale = e4m3fn_to_float(
        weight_block_scales[column * blocks_per_row + inner / UINT64_C(16)]);
    partial += e2m1_to_float(activation_code) * activation_scale *
               e2m1_to_float(weight_code) * weight_scale;
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    partial += __shfl_down(partial, offset, 32U);
  }
  __shared__ float wave_sums[8];
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  if (lane == 0U) {
    wave_sums[wave] = partial;
  }
  __syncthreads();
  if (wave == 0U) {
    partial = lane < 8U ? wave_sums[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
      partial += __shfl_down(partial, offset, 32U);
    }
    if (lane == 0U) {
      output[column] = float_to_bf16_rne_bits(partial * weight_tensor_scale[0] *
                                              input_tensor_scale[0]);
    }
  }
}

// Phase 78 opt-in M=1 decode candidate. A 128-thread workgroup owns one
// adjacent output-column tile, so each thread computes one independent
// output. The packed activation row and its decoded block scales are loaded
// once into dynamic LDS; the following DP4A loop has no per-column reduction or
// barrier. K is bounded by the selector to the largest Qwen3.8-27B Phase 78
// projection (17,408), keeping this LDS allocation at 13,056 bytes.
extern "C" __global__
__launch_bounds__(128, 1) void sllm_matmul_nvfp4_w4a4_decode_columns128_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  if (m != 1U || k == 0U || (k % UINT64_C(16)) != 0U ||
      k > sllm_matmul_kernel::kNvfp4W4A4DecodeColumns128MaxK) {
    return;
  }

  const uint64_t packed_activation_bytes = k / UINT64_C(2);
  const uint64_t blocks_per_row = k / UINT64_C(16);
  extern __shared__ uint8_t decode_lds[];
  uint8_t *const activation_tile = decode_lds;
  // K is a multiple of 16, hence packed_activation_bytes is 8-byte aligned.
  float *const activation_scales =
      reinterpret_cast<float *>(decode_lds + packed_activation_bytes);

  for (uint64_t index = threadIdx.x; index < packed_activation_bytes;
       index += blockDim.x) {
    activation_tile[index] =
        __builtin_nontemporal_load(packed_activation + index);
  }
  for (uint64_t block = threadIdx.x; block < blocks_per_row;
       block += blockDim.x) {
    activation_scales[block] = e4m3fn_to_float(
        __builtin_nontemporal_load(activation_block_scales + block));
  }
  __syncthreads();

  const uint64_t column = static_cast<uint64_t>(blockIdx.x) * UINT64_C(128) +
                          static_cast<uint64_t>(threadIdx.x);
  if (column >= n) {
    return;
  }
  const uint64_t packed_weight_row_bytes = k / UINT64_C(2);
  const uint8_t *const weight_row =
      packed_weight + column * packed_weight_row_bytes;
  const uint8_t *const weight_scale_row =
      weight_block_scales + column * blocks_per_row;
  float accumulator = 0.0F;

  // Each block16 consists of two packed uint32 words. Each word contributes
  // two signed dot4 operations (even and odd nibbles), for four exact DP4A
  // operations per block. E2M1 values were scaled by two in the byte packs,
  // so divide the integer sum by four before applying the two E4M3 scales.
  for (uint64_t block = 0U; block < blocks_per_row; ++block) {
    const uint64_t packed_offset = block * UINT64_C(8);
    const auto *const activation_words =
        reinterpret_cast<const uint32_t *>(activation_tile + packed_offset);
    const auto *const weight_words =
        reinterpret_cast<const uint32_t *>(weight_row + packed_offset);
    const uint32_t activation0 = activation_words[0];
    const uint32_t activation1 = activation_words[1];
    const uint32_t weight0 =
        __builtin_nontemporal_load(weight_words + UINT32_C(0));
    const uint32_t weight1 =
        __builtin_nontemporal_load(weight_words + UINT32_C(1));
    const E2M1Scaled2Packs activation_pack0 =
        e2m1x8_scaled2_to_i8x4_pair(activation0);
    const E2M1Scaled2Packs activation_pack1 =
        e2m1x8_scaled2_to_i8x4_pair(activation1);
    const E2M1Scaled2Packs weight_pack0 = e2m1x8_scaled2_to_i8x4_pair(weight0);
    const E2M1Scaled2Packs weight_pack1 = e2m1x8_scaled2_to_i8x4_pair(weight1);
    int32_t block_sum = 0;
    block_sum =
        signed_dot4(activation_pack0.even, weight_pack0.even, block_sum);
    block_sum = signed_dot4(activation_pack0.odd, weight_pack0.odd, block_sum);
    block_sum =
        signed_dot4(activation_pack1.even, weight_pack1.even, block_sum);
    block_sum = signed_dot4(activation_pack1.odd, weight_pack1.odd, block_sum);
    const float weight_scale =
        e4m3fn_to_float(__builtin_nontemporal_load(weight_scale_row + block));
    accumulator += static_cast<float>(block_sum) * 0.25F *
                   activation_scales[block] * weight_scale;
  }

  output[column] = float_to_bf16_rne_bits(accumulator * weight_tensor_scale[0] *
                                          input_tensor_scale[0]);
}

// Phase 78 ID67 decode candidate. Eight wave32s cover a 32-column output
// tile; each wave owns four adjacent columns and each lane carries a block16
// index strided by 32. The activation's two packed dwords and decoded scale
// are fetched once per lane and reused across the four column accumulators.
// There is no LDS and no barrier: only the final four independent wave
// reductions remain before lane zero stores BF16-RNE results.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_decode_dp4a_wave4col32_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  if (m != 1U || k == 0U || (k % UINT64_C(16)) != 0U ||
      k > sllm_matmul_kernel::kNvfp4W4A4DecodeColumns128MaxK) {
    return;
  }
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t columns_per_wave = 4U;
  constexpr uint32_t columns_per_workgroup = 32U;
  const uint32_t lane = threadIdx.x & (wave_width - 1U);
  const uint32_t wave = threadIdx.x / wave_width;
  const uint64_t blocks_per_row = k / UINT64_C(16);
  const uint64_t block_base = static_cast<uint64_t>(blockIdx.x) *
                              static_cast<uint64_t>(columns_per_workgroup);
  const uint64_t column_base =
      block_base + static_cast<uint64_t>(wave) * columns_per_wave;
  float accumulators[columns_per_wave] = {};
  const uint64_t packed_row_bytes = k / UINT64_C(2);

  for (uint64_t block = lane; block < blocks_per_row;
       block += static_cast<uint64_t>(wave_width)) {
    const uint64_t packed_offset = block * UINT64_C(8);
    const auto *const activation_words =
        reinterpret_cast<const uint32_t *>(packed_activation + packed_offset);
    const E2M1Scaled2Packs activation_pack0 =
        e2m1x8_scaled2_to_i8x4_pair(activation_words[0]);
    const E2M1Scaled2Packs activation_pack1 =
        e2m1x8_scaled2_to_i8x4_pair(activation_words[1]);
    const float activation_scale = e4m3fn_to_float(
        __builtin_nontemporal_load(activation_block_scales + block));

#pragma unroll
    for (uint32_t column_offset = 0U; column_offset < columns_per_wave;
         ++column_offset) {
      const uint64_t column = column_base + column_offset;
      if (column >= n) {
        continue;
      }
      const auto *const weight_words = reinterpret_cast<const uint32_t *>(
          packed_weight + column * packed_row_bytes + packed_offset);
      const E2M1Scaled2Packs weight_pack0 = e2m1x8_scaled2_to_i8x4_pair(
          __builtin_nontemporal_load(weight_words + UINT32_C(0)));
      const E2M1Scaled2Packs weight_pack1 = e2m1x8_scaled2_to_i8x4_pair(
          __builtin_nontemporal_load(weight_words + UINT32_C(1)));
      int32_t block_sum = 0;
      block_sum =
          signed_dot4(activation_pack0.even, weight_pack0.even, block_sum);
      block_sum =
          signed_dot4(activation_pack0.odd, weight_pack0.odd, block_sum);
      block_sum =
          signed_dot4(activation_pack1.even, weight_pack1.even, block_sum);
      block_sum =
          signed_dot4(activation_pack1.odd, weight_pack1.odd, block_sum);
      const float weight_scale = e4m3fn_to_float(__builtin_nontemporal_load(
          weight_block_scales + column * blocks_per_row + block));
      accumulators[column_offset] += static_cast<float>(block_sum) * 0.25F *
                                     activation_scale * weight_scale;
    }
  }

#pragma unroll
  for (uint32_t column_offset = 0U; column_offset < columns_per_wave;
       ++column_offset) {
#pragma unroll
    for (uint32_t offset = wave_width / 2U; offset != 0U; offset >>= 1U) {
      accumulators[column_offset] +=
          __shfl_down(accumulators[column_offset], offset, wave_width);
    }
    const uint64_t column = column_base + column_offset;
    if (lane == 0U && column < n) {
      output[column] = float_to_bf16_rne_bits(accumulators[column_offset] *
                                              weight_tensor_scale[0] *
                                              input_tensor_scale[0]);
    }
  }
}

// Phase 78 ID73, exact gfx1030 Qwen3.8 decode.  The eight waves keep ID67's
// four-column mapping, but expand the single packed activation row and its
// block scales into dynamic LDS once per workgroup.  Every wave then reuses
// those values while independently streaming its four weight rows.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_decode_dp4a_activation_shared_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  if (!sllm_matmul_kernel::phase78_nvfp4_w4a4_decode_activation_shared_shape(
          m, k, n)) {
    return;
  }
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t waves_per_workgroup = 8U;
  constexpr uint32_t columns_per_wave = 4U;
  constexpr uint32_t columns_per_workgroup =
      waves_per_workgroup * columns_per_wave;
  const uint32_t lane = threadIdx.x & (wave_width - 1U);
  const uint32_t wave = threadIdx.x / wave_width;
  const uint64_t blocks_per_row = k / UINT64_C(16);
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * columns_per_workgroup +
      static_cast<uint64_t>(wave) * columns_per_wave;
  const uint64_t packed_row_bytes = k / UINT64_C(2);

  // Four int8x4 packs (16 bytes) plus one FP32 decoded scale (4 bytes) are
  // shared per block16: dynamic LDS is therefore exactly K*5/4 bytes.
  extern __shared__ uint32_t shared[];
  int32_t *const activation_packs = reinterpret_cast<int32_t *>(shared);
  float *const activation_scale_values =
      reinterpret_cast<float *>(shared + blocks_per_row * UINT64_C(4));
  for (uint64_t block = threadIdx.x; block < blocks_per_row;
       block += blockDim.x) {
    const auto *const activation_words = reinterpret_cast<const uint32_t *>(
        packed_activation + block * UINT64_C(8));
    const E2M1Scaled2Packs first = e2m1x8_scaled2_to_i8x4_pair(
        __builtin_nontemporal_load(activation_words + UINT32_C(0)));
    const E2M1Scaled2Packs second = e2m1x8_scaled2_to_i8x4_pair(
        __builtin_nontemporal_load(activation_words + UINT32_C(1)));
    activation_packs[block * UINT64_C(4) + UINT64_C(0)] = first.even;
    activation_packs[block * UINT64_C(4) + UINT64_C(1)] = first.odd;
    activation_packs[block * UINT64_C(4) + UINT64_C(2)] = second.even;
    activation_packs[block * UINT64_C(4) + UINT64_C(3)] = second.odd;
    activation_scale_values[block] = e4m3fn_to_float(
        __builtin_nontemporal_load(activation_block_scales + block));
  }
  __syncthreads();

  float accumulators[columns_per_wave] = {};
  for (uint64_t block = lane; block < blocks_per_row;
       block += static_cast<uint64_t>(wave_width)) {
    const E2M1Scaled2Packs activation_pack0 = {
        activation_packs[block * UINT64_C(4) + UINT64_C(0)],
        activation_packs[block * UINT64_C(4) + UINT64_C(1)]};
    const E2M1Scaled2Packs activation_pack1 = {
        activation_packs[block * UINT64_C(4) + UINT64_C(2)],
        activation_packs[block * UINT64_C(4) + UINT64_C(3)]};
    const float activation_scale = activation_scale_values[block];
#pragma unroll
    for (uint32_t column_offset = 0U; column_offset < columns_per_wave;
         ++column_offset) {
      const uint64_t column = column_base + column_offset;
      if (column >= n) {
        continue;
      }
      const auto *const weight_words = reinterpret_cast<const uint32_t *>(
          packed_weight + column * packed_row_bytes + block * UINT64_C(8));
      const E2M1Scaled2Packs weight_pack0 = e2m1x8_scaled2_to_i8x4_pair(
          __builtin_nontemporal_load(weight_words + UINT32_C(0)));
      const E2M1Scaled2Packs weight_pack1 = e2m1x8_scaled2_to_i8x4_pair(
          __builtin_nontemporal_load(weight_words + UINT32_C(1)));
      int32_t block_sum = 0;
      block_sum =
          signed_dot4(activation_pack0.even, weight_pack0.even, block_sum);
      block_sum =
          signed_dot4(activation_pack0.odd, weight_pack0.odd, block_sum);
      block_sum =
          signed_dot4(activation_pack1.even, weight_pack1.even, block_sum);
      block_sum =
          signed_dot4(activation_pack1.odd, weight_pack1.odd, block_sum);
      const float weight_scale = e4m3fn_to_float(__builtin_nontemporal_load(
          weight_block_scales + column * blocks_per_row + block));
      accumulators[column_offset] += static_cast<float>(block_sum) * 0.25F *
                                     activation_scale * weight_scale;
    }
  }

#pragma unroll
  for (uint32_t column_offset = 0U; column_offset < columns_per_wave;
       ++column_offset) {
#pragma unroll
    for (uint32_t offset = wave_width / 2U; offset != 0U; offset >>= 1U) {
      accumulators[column_offset] +=
          __shfl_down(accumulators[column_offset], offset, wave_width);
    }
    const uint64_t column = column_base + column_offset;
    if (lane == 0U && column < n) {
      output[column] = float_to_bf16_rne_bits(accumulators[column_offset] *
                                              weight_tensor_scale[0] *
                                              input_tensor_scale[0]);
    }
  }
}

// Prefill specialization for NVFP4 W4A4.  A workgroup owns one output column
// and eight consecutive rows.  The packed weight values and block scales are
// loaded once per 256-value K tile, then consumed by eight wave32 reductions;
// this removes the M-fold weight traffic of the elementwise kernel while
// preserving the per-row activation block scales and the two global scales.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_block16_prefill_row8_tiled256_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t rows_per_workgroup = 8U;
  constexpr uint32_t tile_k = 256U;
  __shared__ float weight_tile[tile_k];
  __shared__ float weight_scale_tile[tile_k / 16U];
  __shared__ float shared_weight_tensor_scale;
  __shared__ float shared_input_tensor_scale;

  const uint64_t column = static_cast<uint64_t>(blockIdx.x) % n;
  const uint64_t row_base =
      (static_cast<uint64_t>(blockIdx.x) / n) * rows_per_workgroup;
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  const uint64_t row = row_base + wave;
  const uint64_t blocks_per_weight_row = (k + UINT64_C(15)) / UINT64_C(16);
  const uint64_t blocks_per_activation_row = (k + UINT64_C(15)) / UINT64_C(16);
  const uint64_t packed_activation_row = (k + UINT64_C(1)) / UINT64_C(2);

  if (threadIdx.x == 0U) {
    shared_weight_tensor_scale = weight_tensor_scale[0];
    shared_input_tensor_scale = input_tensor_scale[0];
  }
  float accumulator = 0.0F;
  for (uint64_t base = 0U; base < k; base += tile_k) {
    if (threadIdx.x < tile_k / 16U) {
      const uint64_t scale_inner = base + threadIdx.x * UINT64_C(16);
      weight_scale_tile[threadIdx.x] =
          scale_inner < k
              ? e4m3fn_to_float(
                    weight_block_scales[column * blocks_per_weight_row +
                                        scale_inner / 16U])
              : 0.0F;
    }
    __syncthreads();
    const uint64_t global_inner = base + threadIdx.x;
    if (global_inner < k) {
      const uint64_t weight_index = column * k + global_inner;
      const uint8_t packed = __builtin_nontemporal_load(
          packed_weight + weight_index / UINT64_C(2));
      const uint8_t code = (weight_index & UINT64_C(1)) == 0U
                               ? packed & UINT8_C(0x0f)
                               : packed >> 4U;
      weight_tile[threadIdx.x] =
          e2m1_to_float(code) * weight_scale_tile[threadIdx.x / 16U];
    } else {
      weight_tile[threadIdx.x] = 0.0F;
    }
    __syncthreads();
    if (row < m) {
      const uint32_t valid = static_cast<uint32_t>(
          k - base < tile_k ? k - base : static_cast<uint64_t>(tile_k));
      for (uint32_t offset = lane; offset < valid; offset += wave_width) {
        const uint64_t inner = base + offset;
        const uint8_t activation_pair = __builtin_nontemporal_load(
            packed_activation + row * packed_activation_row + inner / 2U);
        const uint8_t activation_code = (inner & UINT64_C(1)) == 0U
                                            ? activation_pair & UINT8_C(0x0f)
                                            : activation_pair >> 4U;
        const float activation_scale = e4m3fn_to_float(
            activation_block_scales[row * blocks_per_activation_row +
                                    inner / UINT64_C(16)]);
        accumulator += e2m1_to_float(activation_code) * activation_scale *
                       weight_tile[offset];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    accumulator += __shfl_down(accumulator, offset, 32U);
  }
  if (lane == 0U && row < m) {
    output[row * n + column] = float_to_bf16_rne_bits(
        accumulator * shared_weight_tensor_scale * shared_input_tensor_scale);
  }
}

// Prefill specialization that tiles both output dimensions.  Each workgroup
// owns an 8x8 output tile and has eight wave32s, one wave per output row.  The
// four lanes in each group of four reduce one output column, so the same
// activation tile is reused across eight columns and the same weight tile is
// reused across eight rows.  Raw nibbles stay in LDS to avoid expanding the
// 16x16 block values to floats before they are consumed.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_block16_prefill_row8_col8_tiled256_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  constexpr uint32_t rows_per_workgroup = 8U;
  constexpr uint32_t columns_per_workgroup = 8U;
  constexpr uint32_t tile_k = 256U;
  constexpr uint32_t tile_packed_bytes = tile_k / 2U;
  constexpr uint32_t weight_tile_packed_bytes = tile_packed_bytes + 1U;
  constexpr uint32_t blocks_per_tile = tile_k / 16U;
  __shared__ uint8_t activation_tile[rows_per_workgroup * tile_packed_bytes];
  __shared__ uint8_t
      weight_tile[columns_per_workgroup * weight_tile_packed_bytes];
  __shared__ float activation_scale_tile[rows_per_workgroup * blocks_per_tile];
  __shared__ float weight_scale_tile[columns_per_workgroup * blocks_per_tile];
  __shared__ float shared_weight_tensor_scale;
  __shared__ float shared_input_tensor_scale;

  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  const uint32_t local_column = lane >> 2U;
  const uint32_t reduction_lane = lane & UINT32_C(3);
  const uint64_t tile_columns =
      (n + columns_per_workgroup - 1U) / columns_per_workgroup;
  const uint64_t tile_index = blockIdx.x;
  const uint64_t tile_row = (tile_index / tile_columns) * rows_per_workgroup;
  const uint64_t tile_column =
      (tile_index - (tile_index / tile_columns) * tile_columns) *
      columns_per_workgroup;
  const uint64_t row = tile_row + wave;
  const uint64_t column = tile_column + local_column;
  const uint64_t blocks_per_row = (k + UINT64_C(15)) / UINT64_C(16);
  const uint64_t packed_activation_row = (k + UINT64_C(1)) / UINT64_C(2);
  const uint64_t weight_row_start = column * k;
  const uint32_t weight_start_parity =
      static_cast<uint32_t>(weight_row_start & UINT64_C(1));

  if (threadIdx.x == 0U) {
    shared_weight_tensor_scale = weight_tensor_scale[0];
    shared_input_tensor_scale = input_tensor_scale[0];
  }

  float accumulator = 0.0F;
  for (uint64_t base = 0U; base < k; base += tile_k) {
    const uint64_t remaining = k - base;
    const uint32_t valid = static_cast<uint32_t>(
        remaining < tile_k ? remaining : static_cast<uint64_t>(tile_k));

    // Each thread loads eight bytes from each value plane.  Weight rows are
    // flattened before nibble packing, hence the extra byte for odd row starts.
    for (uint32_t index = threadIdx.x;
         index < rows_per_workgroup * tile_packed_bytes; index += 256U) {
      const uint32_t source_row = index / tile_packed_bytes;
      const uint32_t byte = index & (tile_packed_bytes - 1U);
      const uint64_t global_row = tile_row + source_row;
      const uint64_t global_inner = base + static_cast<uint64_t>(byte) * 2U;
      activation_tile[index] =
          global_row < m && global_inner < k
              ? __builtin_nontemporal_load(packed_activation +
                                           global_row * packed_activation_row +
                                           base / 2U + byte)
              : UINT8_C(0);
    }
    for (uint32_t index = threadIdx.x;
         index < columns_per_workgroup * weight_tile_packed_bytes;
         index += 256U) {
      const uint32_t source_column = index / weight_tile_packed_bytes;
      const uint32_t byte = index % weight_tile_packed_bytes;
      const uint64_t global_column = tile_column + source_column;
      const uint64_t global_inner = base + static_cast<uint64_t>(byte) * 2U;
      const uint64_t start = global_column * k + base;
      const uint32_t needed_bytes =
          (valid + static_cast<uint32_t>(start & UINT64_C(1)) + 1U) / 2U;
      weight_tile[index] =
          global_column < n && byte < needed_bytes
              ? __builtin_nontemporal_load(packed_weight + start / 2U + byte)
              : UINT8_C(0);
      (void)global_inner;
    }
    for (uint32_t index = threadIdx.x;
         index < rows_per_workgroup * blocks_per_tile; index += 256U) {
      const uint32_t source_row = index / blocks_per_tile;
      const uint32_t block = index & (blocks_per_tile - 1U);
      const uint64_t global_row = tile_row + source_row;
      const uint64_t global_block = base / 16U + block;
      activation_scale_tile[index] =
          global_row < m && global_block < blocks_per_row
              ? e4m3fn_to_float(
                    activation_block_scales[global_row * blocks_per_row +
                                            global_block])
              : 0.0F;
    }
    for (uint32_t index = threadIdx.x;
         index < columns_per_workgroup * blocks_per_tile; index += 256U) {
      const uint32_t source_column = index / blocks_per_tile;
      const uint32_t block = index & (blocks_per_tile - 1U);
      const uint64_t global_column = tile_column + source_column;
      const uint64_t global_block = base / 16U + block;
      weight_scale_tile[index] =
          global_column < n && global_block < blocks_per_row
              ? e4m3fn_to_float(
                    weight_block_scales[global_column * blocks_per_row +
                                        global_block])
              : 0.0F;
    }
    __syncthreads();

    if (row < m && column < n) {
      const uint32_t packed_weight_parity = weight_start_parity;
      for (uint32_t offset = reduction_lane; offset < valid; offset += 4U) {
        const uint32_t activation_byte = offset / 2U;
        const uint8_t activation_pair =
            activation_tile[wave * tile_packed_bytes + activation_byte];
        const uint8_t activation_code = (offset & UINT32_C(1)) == 0U
                                            ? activation_pair & UINT8_C(0x0f)
                                            : activation_pair >> 4U;
        const uint32_t weight_byte = (offset + packed_weight_parity) / 2U;
        const uint8_t weight_pair =
            weight_tile[local_column * weight_tile_packed_bytes + weight_byte];
        const uint8_t weight_code =
            ((packed_weight_parity + offset) & UINT32_C(1)) == 0U
                ? weight_pair & UINT8_C(0x0f)
                : weight_pair >> 4U;
        const float activation_scale =
            activation_scale_tile[wave * blocks_per_tile + offset / 16U];
        const float weight_scale =
            weight_scale_tile[local_column * blocks_per_tile + offset / 16U];
        accumulator += e2m1_to_float(activation_code) * activation_scale *
                       e2m1_to_float(weight_code) * weight_scale;
      }
    }
    __syncthreads();
  }

  accumulator += __shfl_down(accumulator, 2U, 4U);
  accumulator += __shfl_down(accumulator, 1U, 4U);
  if (row < m && column < n && reduction_lane == 0U) {
    output[row * n + column] = float_to_bf16_rne_bits(
        accumulator * shared_weight_tensor_scale * shared_input_tensor_scale);
  }
}

// Matrix-shaped NVFP4 prefill provider.  A 16x16 logical thread tile computes
// a 64x64 output tile (4x4 outputs per thread).  Packed E2M1 values are
// expanded into signed value*2 bytes in LDS, then accumulated four products at
// a time with v_dot4_i32_i8.  Scaling remains block16-exact: integer sums are
// converted after every 16-value block and multiplied by the corresponding
// activation and weight E4M3 scales before FP32 accumulation.
template <typename Index>
__device__ __forceinline__ uint32_t
nvfp4_prefill_load_packed32(const uint8_t *const address) noexcept {
  if constexpr (sizeof(Index) == sizeof(uint32_t)) {
    return *reinterpret_cast<const uint32_t *>(address);
  } else {
    return __builtin_nontemporal_load(
        reinterpret_cast<const uint32_t *>(address));
  }
}

template <uint32_t TileK, uint32_t TileM = 64U, typename Index = uint64_t>
__device__ __forceinline__ void
sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_body(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  static_assert(sizeof(Index) == sizeof(uint32_t) ||
                sizeof(Index) == sizeof(uint64_t));
  constexpr uint32_t tile_m = TileM;
  static_assert(tile_m == 32U || tile_m == 64U);
  constexpr uint32_t tile_n = 64U;
  constexpr uint32_t tile_k = TileK;
  static_assert(tile_k == 32U || tile_k == 128U);
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

  using index_t = Index;
  const index_t m_index = static_cast<index_t>(m);
  const index_t k_index = static_cast<index_t>(k);
  const index_t n_index = static_cast<index_t>(n);
  const index_t column_tiles = (n_index + static_cast<index_t>(tile_n - 1U)) /
                               static_cast<index_t>(tile_n);
  const index_t tile_index = static_cast<index_t>(blockIdx.x);
  const index_t row_base =
      (tile_index / column_tiles) * static_cast<index_t>(tile_m);
  const index_t column_base =
      (tile_index % column_tiles) * static_cast<index_t>(tile_n);
  const uint32_t thread = threadIdx.x;
  const uint32_t local_row = thread >> 4U;
  const uint32_t local_column = thread & UINT32_C(15);
  const index_t packed_row_bytes = k_index / static_cast<index_t>(2U);
  const index_t blocks_per_row = k_index / static_cast<index_t>(block_k);
  float accumulators[rows_per_thread][columns_per_thread] = {};

  for (index_t base = 0U; base < k_index;
       base += static_cast<index_t>(tile_k)) {
    for (uint32_t index = thread; index < tile_m * packed_chunks_per_stage;
         index += 256U) {
      const uint32_t row = index / packed_chunks_per_stage;
      const uint32_t chunk = index % packed_chunks_per_stage;
      const index_t source_row = row_base + static_cast<index_t>(row);
      const index_t inner = base + static_cast<index_t>(chunk) * 8U;
      const E2M1Scaled2Packs values =
          source_row < m_index && inner + 8U <= k_index
              ? e2m1x8_scaled2_to_i8x4_pair(nvfp4_prefill_load_packed32<Index>(
                    packed_activation + source_row * packed_row_bytes +
                    inner / static_cast<index_t>(2U)))
              : E2M1Scaled2Packs{0, 0};
      activation_tile[row][chunk * 2U] = values.even;
      activation_tile[row][chunk * 2U + 1U] = values.odd;
    }
    for (uint32_t index = thread; index < tile_n * packed_chunks_per_stage;
         index += 256U) {
      const uint32_t column = index / packed_chunks_per_stage;
      const uint32_t chunk = index % packed_chunks_per_stage;
      const index_t source_column = column_base + static_cast<index_t>(column);
      const index_t inner = base + static_cast<index_t>(chunk) * 8U;
      const E2M1Scaled2Packs values =
          source_column < n_index && inner + 8U <= k_index
              ? e2m1x8_scaled2_to_i8x4_pair(nvfp4_prefill_load_packed32<Index>(
                    packed_weight + source_column * packed_row_bytes +
                    inner / static_cast<index_t>(2U)))
              : E2M1Scaled2Packs{0, 0};
      weight_tile[column][chunk * 2U] = values.even;
      weight_tile[column][chunk * 2U + 1U] = values.odd;
    }
    for (uint32_t index = thread; index < tile_m * blocks_per_stage;
         index += 256U) {
      const uint32_t row = index / blocks_per_stage;
      const uint32_t block = index % blocks_per_stage;
      const index_t source_row = row_base + static_cast<index_t>(row);
      const index_t source_block =
          base / static_cast<index_t>(block_k) + static_cast<index_t>(block);
      // E4M3 * 2^-2 is exact in FP32. Moving this binary shift into
      // staging removes one multiply per output/block without changing
      // (integer_dot * 0.25F) * activation_scale or its rounding.
      activation_scale_tile[row][block] =
          source_row < m_index && source_block < blocks_per_row
              ? e4m3fn_to_float(
                    activation_block_scales[source_row * blocks_per_row +
                                            source_block]) *
                    0.25F
              : 0.0F;
    }
    for (uint32_t index = thread; index < tile_n * blocks_per_stage;
         index += 256U) {
      const uint32_t column = index / blocks_per_stage;
      const uint32_t block = index % blocks_per_stage;
      const index_t source_column = column_base + static_cast<index_t>(column);
      const index_t source_block =
          base / static_cast<index_t>(block_k) + static_cast<index_t>(block);
      weight_scale_tile[column][block] =
          source_column < n_index && source_block < blocks_per_row
              ? e4m3fn_to_float(
                    weight_block_scales[source_column * blocks_per_row +
                                        source_block])
              : 0.0F;
    }
    __syncthreads();

#pragma unroll
    for (uint32_t block = 0U; block < blocks_per_stage; ++block) {
      if (base + static_cast<index_t>(block) * block_k >= k_index) {
        continue;
      }
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
                signed_dot4(activation_packs[row], weight_packs[column],
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
          accumulators[row][column] +=
              static_cast<float>(block_sums[row][column]) * activation_scale *
              weight_scale;
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
      const index_t output_row = row_base + static_cast<index_t>(local_row) +
                                 static_cast<index_t>(row) * thread_rows;
      const index_t output_column =
          column_base + static_cast<index_t>(local_column) +
          static_cast<index_t>(column) * thread_columns;
      if (output_row < m_index && output_column < n_index) {
        output[output_row * n_index + output_column] =
            float_to_bf16_rne_bits(accumulators[row][column] * tensor_scale);
      }
    }
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_body<32U>(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n);
}

// ID62 index32 variant. Keep this as a separate entry point so the compiler
// can allocate the reduced index arithmetic independently of the existing
// uint64 body; no runtime branch is mixed into the original kernel.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_index32_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_body<32U, 64U, uint32_t>(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n);
}

// The exact M=1024, K=5120, N=17408 gfx1030 projection has enough K-stage
// work to hide the next stage's four raw global loads behind the current
// stage's dot4 arithmetic.  Keep this in a separate entry point: the ordinary
// Index32 provider retains its established instruction stream and remains the
// route for every other shape.
struct Nvfp4W4A4PrefillRawStage64x64K32 final {
  uint32_t activation;
  uint32_t weight;
  uint32_t activation_scale;
  uint32_t weight_scale;
};

__device__ __forceinline__ Nvfp4W4A4PrefillRawStage64x64K32
nvfp4_w4a4_prefill_dp4a_64x64_prefetch_raw_stage(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales, const uint32_t m,
    const uint32_t k, const uint32_t n, const uint32_t row_base,
    const uint32_t column_base, const uint32_t packed_row_bytes,
    const uint32_t blocks_per_row, const uint32_t thread,
    const uint32_t base) noexcept {
  Nvfp4W4A4PrefillRawStage64x64K32 raw{};
  const uint32_t packed_row = thread >> 2U;
  const uint32_t packed_chunk = thread & UINT32_C(3);
  const uint32_t packed_inner = base + packed_chunk * 8U;
  const uint32_t source_row = row_base + packed_row;
  const uint32_t source_column = column_base + packed_row;
  if (source_row < m && packed_inner + 8U <= k) {
    raw.activation = *reinterpret_cast<const uint32_t *>(
        packed_activation + source_row * packed_row_bytes + packed_inner / 2U);
  }
  if (source_column < n && packed_inner + 8U <= k) {
    raw.weight = *reinterpret_cast<const uint32_t *>(
        packed_weight + source_column * packed_row_bytes + packed_inner / 2U);
  }

  if (thread < 128U) {
    const uint32_t scale_row = thread >> 1U;
    const uint32_t scale_block = thread & UINT32_C(1);
    const uint32_t source_scale_row = row_base + scale_row;
    const uint32_t source_scale_column = column_base + scale_row;
    const uint32_t source_block = base / 16U + scale_block;
    if (source_scale_row < m && source_block < blocks_per_row) {
      raw.activation_scale =
          activation_block_scales[source_scale_row * blocks_per_row +
                                  source_block];
    }
    if (source_scale_column < n && source_block < blocks_per_row) {
      raw.weight_scale =
          weight_block_scales[source_scale_column * blocks_per_row +
                              source_block];
    }
  }
  return raw;
}

__device__ __forceinline__ void
sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_index32_pipeline_body(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m64, const uint64_t k64, const uint64_t n64) {
  constexpr uint32_t tile_m = 64U;
  constexpr uint32_t tile_n = 64U;
  constexpr uint32_t tile_k = 32U;
  constexpr uint32_t block_k = 16U;
  constexpr uint32_t blocks_per_stage = tile_k / block_k;
  constexpr uint32_t packed_groups_per_stage = tile_k / 4U;
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

  const uint32_t m = static_cast<uint32_t>(m64);
  const uint32_t k = static_cast<uint32_t>(k64);
  const uint32_t n = static_cast<uint32_t>(n64);
  const uint32_t column_tiles = (n + tile_n - 1U) / tile_n;
  const uint32_t tile_index = blockIdx.x;
  const uint32_t row_base = (tile_index / column_tiles) * tile_m;
  const uint32_t column_base = (tile_index % column_tiles) * tile_n;
  const uint32_t thread = threadIdx.x;
  const uint32_t local_row = thread >> 4U;
  const uint32_t local_column = thread & UINT32_C(15);
  const uint32_t packed_row_bytes = k / 2U;
  const uint32_t blocks_per_row = k / block_k;
  float accumulators[rows_per_thread][columns_per_thread] = {};

  Nvfp4W4A4PrefillRawStage64x64K32 current{};
  if (k != 0U) {
    current = nvfp4_w4a4_prefill_dp4a_64x64_prefetch_raw_stage(
        packed_activation, activation_block_scales, packed_weight,
        weight_block_scales, m, k, n, row_base, column_base, packed_row_bytes,
        blocks_per_row, thread, 0U);
  }

  for (uint32_t base = 0U; base < k; base += tile_k) {
    const uint32_t packed_row = thread >> 2U;
    const uint32_t packed_chunk = thread & UINT32_C(3);
    const E2M1Scaled2Packs activation_values =
        e2m1x8_scaled2_to_i8x4_pair(current.activation);
    const E2M1Scaled2Packs weight_values =
        e2m1x8_scaled2_to_i8x4_pair(current.weight);
    activation_tile[packed_row][packed_chunk * 2U] = activation_values.even;
    activation_tile[packed_row][packed_chunk * 2U + 1U] = activation_values.odd;
    weight_tile[packed_row][packed_chunk * 2U] = weight_values.even;
    weight_tile[packed_row][packed_chunk * 2U + 1U] = weight_values.odd;

    if (thread < 128U) {
      const uint32_t scale_row = thread >> 1U;
      const uint32_t scale_block = thread & UINT32_C(1);
      activation_scale_tile[scale_row][scale_block] =
          e4m3fn_to_float(static_cast<uint8_t>(current.activation_scale)) *
          0.25F;
      weight_scale_tile[scale_row][scale_block] =
          e4m3fn_to_float(static_cast<uint8_t>(current.weight_scale));
    }
    __syncthreads();

    Nvfp4W4A4PrefillRawStage64x64K32 next{};
    const uint32_t next_base = base + tile_k;
    if (next_base < k) {
      next = nvfp4_w4a4_prefill_dp4a_64x64_prefetch_raw_stage(
          packed_activation, activation_block_scales, packed_weight,
          weight_block_scales, m, k, n, row_base, column_base, packed_row_bytes,
          blocks_per_row, thread, next_base);
    }

#pragma unroll
    for (uint32_t block = 0U; block < blocks_per_stage; ++block) {
      if (base + block * block_k >= k)
        continue;
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
                signed_dot4(activation_packs[row], weight_packs[column],
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
          accumulators[row][column] +=
              static_cast<float>(block_sums[row][column]) * activation_scale *
              weight_scale;
        }
      }
    }
    __syncthreads();
    current = next;
  }

  const float tensor_scale = weight_tensor_scale[0] * input_tensor_scale[0];
#pragma unroll
  for (uint32_t row = 0U; row < rows_per_thread; ++row) {
#pragma unroll
    for (uint32_t column = 0U; column < columns_per_thread; ++column) {
      const uint32_t output_row = row_base + local_row + row * thread_rows;
      const uint32_t output_column =
          column_base + local_column + column * thread_columns;
      if (output_row < m && output_column < n) {
        output[output_row * n + output_column] =
            float_to_bf16_rne_bits(accumulators[row][column] * tensor_scale);
      }
    }
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_nvfp4_w4a4_prefill_dp4a64x64_index32_pipeline_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_index32_pipeline_body(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n);
}

// Short prompts retain ID62's per-output block16 arithmetic and ordering.
// Only inactive output rows are removed; for M <= 32 the audited grid is
// unchanged because both the 32-row and 64-row providers use one row tile.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_32x64_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_body<32U, 32U>(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n);
}

// ID62 short-prompt split-K4 research candidate. Each producer owns a
// contiguous floor-partitioned group of TileK=32 stages and writes FP32
// partials. The fixed-order reducer applies tensor scales and BF16-RNE once.
// This path is deliberately separate from the ordinary ID62 and Index32
// kernels; its launcher is guarded to the two measured gfx1030 shapes.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_short_split4_produce_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales, float *const partial_workspace,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  constexpr uint32_t tile_m = 32U;
  constexpr uint32_t tile_n = 64U;
  constexpr uint32_t tile_k = 32U;
  constexpr uint32_t split_count = 4U;
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
  static_assert(rows_per_thread == 2U && columns_per_thread == 4U);

  __shared__ int32_t activation_tile[tile_m][lds_group_stride];
  __shared__ int32_t weight_tile[tile_n][lds_group_stride];
  __shared__ float activation_scale_tile[tile_m][lds_scale_stride];
  __shared__ float weight_scale_tile[tile_n][lds_scale_stride];

  if (m == 0U || k == 0U || n == 0U || (k % block_k) != 0U)
    return;
  const uint64_t column_tiles = (n + tile_n - 1U) / tile_n;
  const uint64_t tile_index = blockIdx.x;
  const uint64_t row_base = (tile_index / column_tiles) * tile_m;
  const uint64_t column_base = (tile_index % column_tiles) * tile_n;
  const uint32_t split_index = blockIdx.z;
  const uint32_t thread = threadIdx.x;
  const uint32_t local_row = thread >> 4U;
  const uint32_t local_column = thread & UINT32_C(15);
  const uint64_t packed_row_bytes = k / 2U;
  const uint64_t blocks_per_row = k / block_k;
  const uint64_t stage_count = (k + tile_k - 1U) / tile_k;
  const uint64_t stages_per_split = stage_count / split_count;
  const uint64_t remainder_stages = stage_count % split_count;
  const uint64_t stage_begin = stages_per_split * split_index +
                               (remainder_stages * split_index) / split_count;
  const uint64_t stage_end =
      stages_per_split * (split_index + 1U) +
      (remainder_stages * (split_index + 1U)) / split_count;
  float accumulators[rows_per_thread][columns_per_thread] = {};

  for (uint64_t stage = stage_begin; stage < stage_end; ++stage) {
    const uint64_t base = stage * tile_k;
    for (uint32_t index = thread; index < tile_m * packed_chunks_per_stage;
         index += 256U) {
      const uint32_t row = index / packed_chunks_per_stage;
      const uint32_t chunk = index % packed_chunks_per_stage;
      const uint64_t source_row = row_base + row;
      const uint64_t inner = base + static_cast<uint64_t>(chunk) * 8U;
      const E2M1Scaled2Packs values =
          source_row < m && inner + 8U <= k
              ? e2m1x8_scaled2_to_i8x4_pair(
                    nvfp4_prefill_load_packed32<uint32_t>(
                        packed_activation + source_row * packed_row_bytes +
                        inner / 2U))
              : E2M1Scaled2Packs{0, 0};
      activation_tile[row][chunk * 2U] = values.even;
      activation_tile[row][chunk * 2U + 1U] = values.odd;
    }
    for (uint32_t index = thread; index < tile_n * packed_chunks_per_stage;
         index += 256U) {
      const uint32_t column = index / packed_chunks_per_stage;
      const uint32_t chunk = index % packed_chunks_per_stage;
      const uint64_t source_column = column_base + column;
      const uint64_t inner = base + static_cast<uint64_t>(chunk) * 8U;
      const E2M1Scaled2Packs values =
          source_column < n && inner + 8U <= k
              ? e2m1x8_scaled2_to_i8x4_pair(
                    nvfp4_prefill_load_packed32<uint32_t>(
                        packed_weight + source_column * packed_row_bytes +
                        inner / 2U))
              : E2M1Scaled2Packs{0, 0};
      weight_tile[column][chunk * 2U] = values.even;
      weight_tile[column][chunk * 2U + 1U] = values.odd;
    }
    for (uint32_t index = thread; index < tile_m * blocks_per_stage;
         index += 256U) {
      const uint32_t row = index / blocks_per_stage;
      const uint32_t block = index % blocks_per_stage;
      const uint64_t source_row = row_base + row;
      const uint64_t source_block = base / block_k + block;
      activation_scale_tile[row][block] =
          source_row < m && source_block < blocks_per_row
              ? e4m3fn_to_float(__builtin_nontemporal_load(
                    activation_block_scales + source_row * blocks_per_row +
                    source_block))
              : 0.0F;
    }
    for (uint32_t index = thread; index < tile_n * blocks_per_stage;
         index += 256U) {
      const uint32_t column = index / blocks_per_stage;
      const uint32_t block = index % blocks_per_stage;
      const uint64_t source_column = column_base + column;
      const uint64_t source_block = base / block_k + block;
      weight_scale_tile[column][block] =
          source_column < n && source_block < blocks_per_row
              ? e4m3fn_to_float(__builtin_nontemporal_load(
                    weight_block_scales + source_column * blocks_per_row +
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
      for (uint32_t group = 0U; group < 4U; ++group) {
        int32_t activation_packs[rows_per_thread];
        int32_t weight_packs[columns_per_thread];
#pragma unroll
        for (uint32_t row = 0U; row < rows_per_thread; ++row) {
          activation_packs[row] = activation_tile[local_row + row * thread_rows]
                                                 [block * 4U + group];
        }
#pragma unroll
        for (uint32_t column = 0U; column < columns_per_thread; ++column) {
          weight_packs[column] =
              weight_tile[local_column + column * thread_columns]
                         [block * 4U + group];
        }
#pragma unroll
        for (uint32_t row = 0U; row < rows_per_thread; ++row) {
#pragma unroll
          for (uint32_t column = 0U; column < columns_per_thread; ++column) {
            block_sums[row][column] =
                signed_dot4(activation_packs[row], weight_packs[column],
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
          accumulators[row][column] +=
              static_cast<float>(block_sums[row][column]) * 0.25F *
              activation_scale * weight_scale;
        }
      }
    }
    __syncthreads();
  }

  const uint64_t elements = m * n;
#pragma unroll
  for (uint32_t row = 0U; row < rows_per_thread; ++row) {
#pragma unroll
    for (uint32_t column = 0U; column < columns_per_thread; ++column) {
      const uint64_t output_row = row_base + local_row + row * thread_rows;
      const uint64_t output_column =
          column_base + local_column + column * thread_columns;
      if (output_row < m && output_column < n) {
        partial_workspace[static_cast<uint64_t>(split_index) * elements +
                          output_row * n + output_column] =
            accumulators[row][column];
      }
    }
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_short_split4_reduce_v1(
    const float *const partial_workspace,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t n) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * 256U + threadIdx.x;
  const uint64_t elements = m * n;
  if (index >= elements)
    return;
  float accumulator = partial_workspace[index];
  accumulator += partial_workspace[elements + index];
  accumulator += partial_workspace[2U * elements + index];
  accumulator += partial_workspace[3U * elements + index];
  output[index] = float_to_bf16_rne_bits(accumulator * weight_tensor_scale[0] *
                                         input_tensor_scale[0]);
}

// Phase 78 ID80 retains ID62's exact block16 arithmetic and 64x64 output
// tile, but stages K=128 between workgroup barriers.  The larger LDS tile
// lowers synchronization frequency on gfx1030 while preserving a separate
// opt-in rollback to ID62.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_k128_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_body<128U>(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n);
}

// gfx1201 matrix-core candidate for NVFP4 W4A4.  Each rocWMMA operation is
// limited to one K=16 NVFP4 scale domain; the contribution is transformed to
// row-major lane order and scaled before the next block is accumulated.  The
// E2M1 -> E4M3FN ingress is exact, so this changes only the FP32 summation
// order relative to the scalar/DP4A providers.
template <bool OrdinaryLoads>
__device__ __forceinline__ void
nvfp4_w4a4_gfx1201_wmma128x64_body(const uint8_t *const packed_activation,
                                   const uint8_t *const activation_block_scales,
                                   const uint8_t *const packed_weight,
                                   const uint8_t *const weight_block_scales,
                                   const float *const weight_tensor_scale,
                                   const float *const input_tensor_scale,
                                   uint16_t *const output, const uint64_t m,
                                   const uint64_t k, const uint64_t n) {
#if defined(__gfx1201__)
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t waves_per_workgroup = 8U;
  constexpr uint32_t tile_m = 16U;
  constexpr uint32_t tile_n = 16U;
  constexpr uint32_t column_tiles = 4U;
  constexpr uint32_t fragment_k = 16U;
  constexpr uint32_t stage_k = 32U;
  constexpr uint32_t scale_blocks_per_stage = stage_k / fragment_k;
  constexpr uint32_t rows_per_workgroup = waves_per_workgroup * tile_m;
  constexpr uint32_t columns_per_workgroup = column_tiles * tile_n;
  constexpr uint32_t tile_values = tile_m * stage_k;
  constexpr uint32_t values_per_group = 4U;
  constexpr uint32_t groups_per_tile = tile_values / values_per_group;
  constexpr uint32_t output_values = tile_m * tile_n;

  __shared__ __align__(4)
      rocwmma::float8_t activation_tile[waves_per_workgroup][tile_values];
  __shared__ __align__(4)
      rocwmma::float8_t weight_tile[column_tiles][tile_values];
  __shared__ float activation_scale_tile[waves_per_workgroup][tile_m]
                                        [scale_blocks_per_stage];
  __shared__ float weight_scale_tile[column_tiles * tile_n]
                                    [scale_blocks_per_stage];
  __shared__ float tensor_scale;

  using AFragment =
      rocwmma::fragment<rocwmma::matrix_a, tile_m, tile_n, fragment_k,
                        rocwmma::float8_t, rocwmma::row_major>;
  using BFragment =
      rocwmma::fragment<rocwmma::matrix_b, tile_m, tile_n, fragment_k,
                        rocwmma::float8_t, rocwmma::col_major>;
  using AccumulatorFragment = rocwmma::fragment<rocwmma::accumulator, tile_m,
                                                tile_n, fragment_k, float>;

  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (wave_width - 1U);
  const uint32_t wave = thread / wave_width;
  const uint64_t row_group_base =
      static_cast<uint64_t>(blockIdx.y) * rows_per_workgroup;
  const uint64_t row_tile_base =
      row_group_base + static_cast<uint64_t>(wave) * tile_m;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * columns_per_workgroup;
  const uint64_t blocks_per_row = k / fragment_k;
  const uint64_t stages =
      (blocks_per_row + scale_blocks_per_stage - 1U) / scale_blocks_per_stage;
  const uint64_t packed_row_bytes = k / 2U;
  float accumulators[column_tiles][output_values / wave_width] = {};

  if (thread == 0U) {
    tensor_scale = weight_tensor_scale[0] * input_tensor_scale[0];
  }

  for (uint64_t stage = 0U; stage < stages; ++stage) {
    const uint64_t inner_base = stage * stage_k;
    auto *const activation_groups =
        reinterpret_cast<uint32_t *>(activation_tile);
    auto *const weight_groups = reinterpret_cast<uint32_t *>(weight_tile);

    for (uint32_t group = thread; group < waves_per_workgroup * groups_per_tile;
         group += blockDim.x) {
      const uint32_t source_wave = group / groups_per_tile;
      const uint32_t wave_group = group - source_wave * groups_per_tile;
      const uint32_t local_row = wave_group / (stage_k / values_per_group);
      const uint32_t local_group =
          wave_group - local_row * (stage_k / values_per_group);
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      uint16_t packed = 0U;
      if (row < m && inner_base + local_group * values_per_group < k) {
        const auto *const source = reinterpret_cast<const uint16_t *>(
            packed_activation + row * packed_row_bytes + inner_base / 2U +
            local_group * 2U);
        if constexpr (OrdinaryLoads) {
          packed = *source;
        } else {
          packed = __builtin_nontemporal_load(source);
        }
      }
      activation_groups[group] = e2m1x4_to_e4m3fn_exact_bits(packed);
    }
    for (uint32_t group = thread; group < column_tiles * groups_per_tile;
         group += blockDim.x) {
      const uint32_t column_tile = group / groups_per_tile;
      const uint32_t tile_group = group - column_tile * groups_per_tile;
      const uint32_t local_column = tile_group / (stage_k / values_per_group);
      const uint32_t local_group =
          tile_group - local_column * (stage_k / values_per_group);
      const uint64_t column = column_base + column_tile * tile_n + local_column;
      uint16_t packed = 0U;
      if (column < n && inner_base + local_group * values_per_group < k) {
        const auto *const source = reinterpret_cast<const uint16_t *>(
            packed_weight + column * packed_row_bytes + inner_base / 2U +
            local_group * 2U);
        if constexpr (OrdinaryLoads) {
          packed = *source;
        } else {
          packed = __builtin_nontemporal_load(source);
        }
      }
      weight_groups[group] = e2m1x4_to_e4m3fn_exact_bits(packed);
    }
    if (thread < waves_per_workgroup * tile_m * scale_blocks_per_stage) {
      const uint32_t scale_block = thread % scale_blocks_per_stage;
      const uint32_t row_index = thread / scale_blocks_per_stage;
      const uint32_t source_wave = row_index / tile_m;
      const uint32_t local_row = row_index - source_wave * tile_m;
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      activation_scale_tile[source_wave][local_row][scale_block] =
          row < m && block < blocks_per_row
              ? e4m3fn_to_float(
                    activation_block_scales[row * blocks_per_row + block])
              : 0.0F;
    }
    if (thread < column_tiles * tile_n * scale_blocks_per_stage) {
      const uint32_t scale_block = thread % scale_blocks_per_stage;
      const uint32_t local_column = thread / scale_blocks_per_stage;
      const uint64_t column = column_base + local_column;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      weight_scale_tile[local_column][scale_block] =
          column < n && block < blocks_per_row
              ? e4m3fn_to_float(
                    weight_block_scales[column * blocks_per_row + block])
              : 0.0F;
    }
    __syncthreads();

    for (uint32_t scale_block = 0U; scale_block < scale_blocks_per_stage;
         ++scale_block) {
      AFragment activation_fragment;
      rocwmma::load_matrix_sync(
          activation_fragment, activation_tile[wave] + scale_block * fragment_k,
          stage_k);
#pragma unroll
      for (uint32_t column_tile = 0U; column_tile < column_tiles;
           ++column_tile) {
        BFragment weight_fragment;
        AccumulatorFragment contribution;
        rocwmma::fill_fragment(contribution, 0.0F);
        rocwmma::load_matrix_sync(
            weight_fragment,
            weight_tile[column_tile] + scale_block * fragment_k, stage_k);
        rocwmma::mma_sync(contribution, activation_fragment, weight_fragment,
                          contribution);
        const auto contribution_row_major =
            rocwmma::apply_data_layout<rocwmma::row_major>(contribution);
#pragma unroll
        for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
          const uint32_t local_row =
              (lane / tile_n) * (output_values / wave_width) + slot;
          const uint32_t local_column = lane % tile_n;
          float term = contribution_row_major[slot] *
                       activation_scale_tile[wave][local_row][scale_block];
          term *= weight_scale_tile[column_tile * tile_n + local_column]
                                   [scale_block];
          accumulators[column_tile][slot] += term;
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
#pragma unroll
    for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
      const uint32_t local_row =
          (lane / tile_n) * (output_values / wave_width) + slot;
      const uint32_t local_column = lane % tile_n;
      const uint64_t row = row_tile_base + local_row;
      const uint64_t column = column_base + column_tile * tile_n + local_column;
      if (row < m && column < n) {
        output[row * n + column] = float_to_bf16_rne_bits(
            accumulators[column_tile][slot] * tensor_scale);
      }
    }
  }
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

extern "C" __global__
__launch_bounds__(256, 1) void sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  nvfp4_w4a4_gfx1201_wmma128x64_body<false>(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_nvfp4_w4a4_prefill_gfx1201_wmma_ordinary_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  nvfp4_w4a4_gfx1201_wmma128x64_body<true>(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n);
}

// R9700 ID64 short split-K4 candidate. This is the ID64 128x64 WMMA body
// with only the contiguous K-stage range and final partial-plane store changed.
// Tensor scaling and BF16 conversion remain in the fixed-order reducer.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_split4_partial_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales, float *const partial_workspace,
    const uint64_t m, const uint64_t k, const uint64_t n) {
#if defined(__gfx1201__)
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t waves_per_workgroup = 8U;
  constexpr uint32_t tile_m = 16U;
  constexpr uint32_t tile_n = 16U;
  constexpr uint32_t column_tiles = 4U;
  constexpr uint32_t fragment_k = 16U;
  constexpr uint32_t stage_k = 32U;
  constexpr uint32_t scale_blocks_per_stage = stage_k / fragment_k;
  constexpr uint32_t rows_per_workgroup = waves_per_workgroup * tile_m;
  constexpr uint32_t columns_per_workgroup = column_tiles * tile_n;
  constexpr uint32_t tile_values = tile_m * stage_k;
  constexpr uint32_t values_per_group = 4U;
  constexpr uint32_t groups_per_tile = tile_values / values_per_group;
  constexpr uint32_t output_values = tile_m * tile_n;
  constexpr uint32_t split_count = 4U;

  __shared__ __align__(4)
      rocwmma::float8_t activation_tile[waves_per_workgroup][tile_values];
  __shared__ __align__(4)
      rocwmma::float8_t weight_tile[column_tiles][tile_values];
  __shared__ float activation_scale_tile[waves_per_workgroup][tile_m]
                                        [scale_blocks_per_stage];
  __shared__ float weight_scale_tile[column_tiles * tile_n]
                                    [scale_blocks_per_stage];

  using AFragment =
      rocwmma::fragment<rocwmma::matrix_a, tile_m, tile_n, fragment_k,
                        rocwmma::float8_t, rocwmma::row_major>;
  using BFragment =
      rocwmma::fragment<rocwmma::matrix_b, tile_m, tile_n, fragment_k,
                        rocwmma::float8_t, rocwmma::col_major>;
  using AccumulatorFragment = rocwmma::fragment<rocwmma::accumulator, tile_m,
                                                tile_n, fragment_k, float>;

  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (wave_width - 1U);
  const uint32_t wave = thread / wave_width;
  const uint64_t row_group_base =
      static_cast<uint64_t>(blockIdx.y) * rows_per_workgroup;
  const uint64_t row_tile_base =
      row_group_base + static_cast<uint64_t>(wave) * tile_m;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * columns_per_workgroup;
  const uint64_t blocks_per_row = k / fragment_k;
  const uint64_t stages =
      (blocks_per_row + scale_blocks_per_stage - 1U) / scale_blocks_per_stage;
  const uint32_t split = blockIdx.z;
  const uint64_t stage_begin =
      (stages * static_cast<uint64_t>(split)) / split_count;
  const uint64_t stage_end =
      (stages * static_cast<uint64_t>(split + 1U)) / split_count;
  const uint64_t packed_row_bytes = k / 2U;
  const uint64_t output_elements = m * n;
  const uint64_t partial_base = static_cast<uint64_t>(split) * output_elements;
  float accumulators[column_tiles][output_values / wave_width] = {};

  for (uint64_t stage = stage_begin; stage < stage_end; ++stage) {
    const uint64_t inner_base = stage * stage_k;
    auto *const activation_groups =
        reinterpret_cast<uint32_t *>(activation_tile);
    auto *const weight_groups = reinterpret_cast<uint32_t *>(weight_tile);

    for (uint32_t group = thread; group < waves_per_workgroup * groups_per_tile;
         group += blockDim.x) {
      const uint32_t source_wave = group / groups_per_tile;
      const uint32_t wave_group = group - source_wave * groups_per_tile;
      const uint32_t local_row = wave_group / (stage_k / values_per_group);
      const uint32_t local_group =
          wave_group - local_row * (stage_k / values_per_group);
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      uint16_t packed = 0U;
      if (row < m && inner_base + local_group * values_per_group < k) {
        // The gfx1201 split4 launcher admits only M17/K17408/N5120.  Keep the
        // exact tile and arithmetic while using the measured ordinary-load
        // ingress for this route.
        packed = *reinterpret_cast<const uint16_t *>(
            packed_activation + row * packed_row_bytes + inner_base / 2U +
            local_group * 2U);
      }
      activation_groups[group] = e2m1x4_to_e4m3fn_exact_bits(packed);
    }
    for (uint32_t group = thread; group < column_tiles * groups_per_tile;
         group += blockDim.x) {
      const uint32_t column_tile = group / groups_per_tile;
      const uint32_t tile_group = group - column_tile * groups_per_tile;
      const uint32_t local_column = tile_group / (stage_k / values_per_group);
      const uint32_t local_group =
          tile_group - local_column * (stage_k / values_per_group);
      const uint64_t column = column_base + column_tile * tile_n + local_column;
      uint16_t packed = 0U;
      if (column < n && inner_base + local_group * values_per_group < k) {
        packed = *reinterpret_cast<const uint16_t *>(
            packed_weight + column * packed_row_bytes + inner_base / 2U +
            local_group * 2U);
      }
      weight_groups[group] = e2m1x4_to_e4m3fn_exact_bits(packed);
    }
    if (thread < waves_per_workgroup * tile_m * scale_blocks_per_stage) {
      const uint32_t scale_block = thread % scale_blocks_per_stage;
      const uint32_t row_index = thread / scale_blocks_per_stage;
      const uint32_t source_wave = row_index / tile_m;
      const uint32_t local_row = row_index - source_wave * tile_m;
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      activation_scale_tile[source_wave][local_row][scale_block] =
          row < m && block < blocks_per_row
              ? e4m3fn_to_float(
                    activation_block_scales[row * blocks_per_row + block])
              : 0.0F;
    }
    if (thread < column_tiles * tile_n * scale_blocks_per_stage) {
      const uint32_t scale_block = thread % scale_blocks_per_stage;
      const uint32_t local_column = thread / scale_blocks_per_stage;
      const uint64_t column = column_base + local_column;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      weight_scale_tile[local_column][scale_block] =
          column < n && block < blocks_per_row
              ? e4m3fn_to_float(
                    weight_block_scales[column * blocks_per_row + block])
              : 0.0F;
    }
    __syncthreads();

    for (uint32_t scale_block = 0U; scale_block < scale_blocks_per_stage;
         ++scale_block) {
      AFragment activation_fragment;
      rocwmma::load_matrix_sync(
          activation_fragment, activation_tile[wave] + scale_block * fragment_k,
          stage_k);
#pragma unroll
      for (uint32_t column_tile = 0U; column_tile < column_tiles;
           ++column_tile) {
        BFragment weight_fragment;
        AccumulatorFragment contribution;
        rocwmma::fill_fragment(contribution, 0.0F);
        rocwmma::load_matrix_sync(
            weight_fragment,
            weight_tile[column_tile] + scale_block * fragment_k, stage_k);
        rocwmma::mma_sync(contribution, activation_fragment, weight_fragment,
                          contribution);
        const auto contribution_row_major =
            rocwmma::apply_data_layout<rocwmma::row_major>(contribution);
#pragma unroll
        for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
          const uint32_t local_row =
              (lane / tile_n) * (output_values / wave_width) + slot;
          const uint32_t local_column = lane % tile_n;
          float term = contribution_row_major[slot] *
                       activation_scale_tile[wave][local_row][scale_block];
          term *= weight_scale_tile[column_tile * tile_n + local_column]
                                   [scale_block];
          accumulators[column_tile][slot] += term;
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
#pragma unroll
    for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
      const uint32_t local_row =
          (lane / tile_n) * (output_values / wave_width) + slot;
      const uint32_t local_column = lane % tile_n;
      const uint64_t row = row_tile_base + local_row;
      const uint64_t column = column_base + column_tile * tile_n + local_column;
      if (row < m && column < n) {
        partial_workspace[partial_base + row * n + column] =
            accumulators[column_tile][slot];
      }
    }
  }
#else
  (void)packed_activation;
  (void)activation_block_scales;
  (void)packed_weight;
  (void)weight_block_scales;
  (void)partial_workspace;
  (void)m;
  (void)k;
  (void)n;
#endif
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_split4_reduce_v1(
    const float *const partial_workspace,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t n) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * 256U + threadIdx.x;
  const uint64_t elements = m * n;
  if (index >= elements)
    return;
  const float p0 = partial_workspace[index];
  const float p1 = partial_workspace[elements + index];
  const float p2 = partial_workspace[2U * elements + index];
  const float p3 = partial_workspace[3U * elements + index];
  const float reduced = ((p0 + p1) + p2) + p3;
  output[index] = float_to_bf16_rne_bits(reduced * weight_tensor_scale[0] *
                                         input_tensor_scale[0]);
}

// Phase 78 ID81 preserves ID64's per-K16 contribution/scaling order while
// reducing the output tile to 128x32.  The smaller accumulator footprint was
// bitwise-equivalent in the standalone gfx1201 tile sweep and is force-only.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x32_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
#if defined(__gfx1201__)
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t waves_per_workgroup = 8U;
  constexpr uint32_t tile_m = 16U;
  constexpr uint32_t tile_n = 16U;
  constexpr uint32_t column_tiles = 2U;
  constexpr uint32_t fragment_k = 16U;
  constexpr uint32_t stage_k = 32U;
  constexpr uint32_t scale_blocks_per_stage = stage_k / fragment_k;
  constexpr uint32_t rows_per_workgroup = waves_per_workgroup * tile_m;
  constexpr uint32_t columns_per_workgroup = column_tiles * tile_n;
  constexpr uint32_t tile_values = tile_m * stage_k;
  constexpr uint32_t values_per_group = 4U;
  constexpr uint32_t groups_per_tile = tile_values / values_per_group;
  constexpr uint32_t output_values = tile_m * tile_n;

  __shared__ __align__(4)
      rocwmma::float8_t activation_tile[waves_per_workgroup][tile_values];
  __shared__ __align__(4)
      rocwmma::float8_t weight_tile[column_tiles][tile_values];
  __shared__ float activation_scale_tile[waves_per_workgroup][tile_m]
                                        [scale_blocks_per_stage];
  __shared__ float weight_scale_tile[column_tiles * tile_n]
                                    [scale_blocks_per_stage];
  __shared__ float tensor_scale;

  using AFragment =
      rocwmma::fragment<rocwmma::matrix_a, tile_m, tile_n, fragment_k,
                        rocwmma::float8_t, rocwmma::row_major>;
  using BFragment =
      rocwmma::fragment<rocwmma::matrix_b, tile_m, tile_n, fragment_k,
                        rocwmma::float8_t, rocwmma::col_major>;
  using AccumulatorFragment = rocwmma::fragment<rocwmma::accumulator, tile_m,
                                                tile_n, fragment_k, float>;

  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (wave_width - 1U);
  const uint32_t wave = thread / wave_width;
  const uint64_t row_group_base =
      static_cast<uint64_t>(blockIdx.y) * rows_per_workgroup;
  const uint64_t row_tile_base =
      row_group_base + static_cast<uint64_t>(wave) * tile_m;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * columns_per_workgroup;
  const uint64_t blocks_per_row = k / fragment_k;
  const uint64_t stages =
      (blocks_per_row + scale_blocks_per_stage - 1U) / scale_blocks_per_stage;
  const uint64_t packed_row_bytes = k / 2U;
  float accumulators[column_tiles][output_values / wave_width] = {};

  if (thread == 0U) {
    tensor_scale = weight_tensor_scale[0] * input_tensor_scale[0];
  }

  for (uint64_t stage = 0U; stage < stages; ++stage) {
    const uint64_t inner_base = stage * stage_k;
    auto *const activation_groups =
        reinterpret_cast<uint32_t *>(activation_tile);
    auto *const weight_groups = reinterpret_cast<uint32_t *>(weight_tile);

    for (uint32_t group = thread; group < waves_per_workgroup * groups_per_tile;
         group += blockDim.x) {
      const uint32_t source_wave = group / groups_per_tile;
      const uint32_t wave_group = group - source_wave * groups_per_tile;
      const uint32_t local_row = wave_group / (stage_k / values_per_group);
      const uint32_t local_group =
          wave_group - local_row * (stage_k / values_per_group);
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      uint16_t packed = 0U;
      if (row < m && inner_base + local_group * values_per_group < k) {
        packed = __builtin_nontemporal_load(reinterpret_cast<const uint16_t *>(
            packed_activation + row * packed_row_bytes + inner_base / 2U +
            local_group * 2U));
      }
      activation_groups[group] = e2m1x4_to_e4m3fn_exact_bits(packed);
    }
    for (uint32_t group = thread; group < column_tiles * groups_per_tile;
         group += blockDim.x) {
      const uint32_t column_tile = group / groups_per_tile;
      const uint32_t tile_group = group - column_tile * groups_per_tile;
      const uint32_t local_column = tile_group / (stage_k / values_per_group);
      const uint32_t local_group =
          tile_group - local_column * (stage_k / values_per_group);
      const uint64_t column = column_base + column_tile * tile_n + local_column;
      uint16_t packed = 0U;
      if (column < n && inner_base + local_group * values_per_group < k) {
        packed = __builtin_nontemporal_load(reinterpret_cast<const uint16_t *>(
            packed_weight + column * packed_row_bytes + inner_base / 2U +
            local_group * 2U));
      }
      weight_groups[group] = e2m1x4_to_e4m3fn_exact_bits(packed);
    }
    if (thread < waves_per_workgroup * tile_m * scale_blocks_per_stage) {
      const uint32_t scale_block = thread % scale_blocks_per_stage;
      const uint32_t row_index = thread / scale_blocks_per_stage;
      const uint32_t source_wave = row_index / tile_m;
      const uint32_t local_row = row_index - source_wave * tile_m;
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      activation_scale_tile[source_wave][local_row][scale_block] =
          row < m && block < blocks_per_row
              ? e4m3fn_to_float(
                    activation_block_scales[row * blocks_per_row + block])
              : 0.0F;
    }
    if (thread < column_tiles * tile_n * scale_blocks_per_stage) {
      const uint32_t scale_block = thread % scale_blocks_per_stage;
      const uint32_t local_column = thread / scale_blocks_per_stage;
      const uint64_t column = column_base + local_column;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      weight_scale_tile[local_column][scale_block] =
          column < n && block < blocks_per_row
              ? e4m3fn_to_float(
                    weight_block_scales[column * blocks_per_row + block])
              : 0.0F;
    }
    __syncthreads();

    for (uint32_t scale_block = 0U; scale_block < scale_blocks_per_stage;
         ++scale_block) {
      AFragment activation_fragment;
      rocwmma::load_matrix_sync(
          activation_fragment, activation_tile[wave] + scale_block * fragment_k,
          stage_k);
#pragma unroll
      for (uint32_t column_tile = 0U; column_tile < column_tiles;
           ++column_tile) {
        BFragment weight_fragment;
        AccumulatorFragment contribution;
        rocwmma::fill_fragment(contribution, 0.0F);
        rocwmma::load_matrix_sync(
            weight_fragment,
            weight_tile[column_tile] + scale_block * fragment_k, stage_k);
        rocwmma::mma_sync(contribution, activation_fragment, weight_fragment,
                          contribution);
        const auto contribution_row_major =
            rocwmma::apply_data_layout<rocwmma::row_major>(contribution);
#pragma unroll
        for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
          const uint32_t local_row =
              (lane / tile_n) * (output_values / wave_width) + slot;
          const uint32_t local_column = lane % tile_n;
          float term = contribution_row_major[slot] *
                       activation_scale_tile[wave][local_row][scale_block];
          term *= weight_scale_tile[column_tile * tile_n + local_column]
                                   [scale_block];
          accumulators[column_tile][slot] += term;
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
#pragma unroll
    for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
      const uint32_t local_row =
          (lane / tile_n) * (output_values / wave_width) + slot;
      const uint32_t local_column = lane % tile_n;
      const uint64_t row = row_tile_base + local_row;
      const uint64_t column = column_base + column_tile * tile_n + local_column;
      if (row < m && column < n) {
        output[row * n + column] = float_to_bf16_rne_bits(
            accumulators[column_tile][slot] * tensor_scale);
      }
    }
  }
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

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_to_mxfp4_block32_even_v1(
    const uint16_t *const activation, uint8_t *const packed_activation,
    uint8_t *const block_scales, const uint64_t m, const uint64_t k) {
  const uint64_t blocks_per_row = (k + UINT64_C(31)) / UINT64_C(32);
  const uint64_t block_index = blockIdx.x;
  if (block_index >= m * blocks_per_row) {
    return;
  }
  const uint64_t row = block_index / blocks_per_row;
  const uint64_t block = block_index - row * blocks_per_row;
  const uint64_t base = block * UINT64_C(32);
  const uint64_t packed_row_bytes = (k + UINT64_C(1)) / UINT64_C(2);
  __shared__ float values[32];
  __shared__ float decoded_scale;
  if (threadIdx.x < 32U) {
    const uint64_t column = base + threadIdx.x;
    values[threadIdx.x] =
        column < k ? bf16_to_float(activation[row * k + column]) : 0.0F;
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    float maximum = 0.0F;
    for (uint32_t index = 0U; index != 32U; ++index) {
      maximum = fmaxf(maximum, fabsf(values[index]));
    }
    const uint8_t encoded_scale = mxfp4_even_scale_code(maximum);
    block_scales[block_index] = encoded_scale;
    decoded_scale = e8m0_to_float(encoded_scale);
  }
  __syncthreads();
  if (threadIdx.x < 16U) {
    const uint32_t first = threadIdx.x * 2U;
    const uint64_t first_column = base + first;
    const uint64_t second_column = first_column + 1U;
    if (first_column < k) {
      const uint8_t low = isfinite(decoded_scale) && decoded_scale > 0.0F
                              ? float_to_e2m1(values[first] / decoded_scale)
                              : 0U;
      const uint8_t high =
          second_column < k && isfinite(decoded_scale) && decoded_scale > 0.0F
              ? float_to_e2m1(values[first + 1U] / decoded_scale)
              : 0U;
      packed_activation[row * packed_row_bytes + first_column / UINT64_C(2)] =
          static_cast<uint8_t>(low | static_cast<uint8_t>(high << 4U));
    }
  }
}

__device__ __forceinline__ void sllm_matmul_mxfp4_w4a4_block32_body(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  const uint64_t output_index = blockIdx.x;
  if (output_index >= m * n) {
    return;
  }
  const uint64_t row = output_index / n;
  const uint64_t column = output_index - row * n;
  const uint64_t blocks_per_row = (k + UINT64_C(31)) / UINT64_C(32);
  const uint64_t packed_row_bytes = (k + UINT64_C(1)) / UINT64_C(2);
  float partial = 0.0F;
  for (uint64_t inner = threadIdx.x; inner < k; inner += blockDim.x) {
    const uint8_t activation_pair = __builtin_nontemporal_load(
        packed_activation + row * packed_row_bytes + inner / UINT64_C(2));
    const uint8_t activation_code = (inner & UINT64_C(1)) == 0U
                                        ? activation_pair & UINT8_C(0x0f)
                                        : activation_pair >> 4U;
    const uint8_t weight_pair = __builtin_nontemporal_load(
        packed_weight + column * packed_row_bytes + inner / UINT64_C(2));
    const uint8_t weight_code = (inner & UINT64_C(1)) == 0U
                                    ? weight_pair & UINT8_C(0x0f)
                                    : weight_pair >> 4U;
    const float activation_scale = e8m0_to_float(
        activation_block_scales[row * blocks_per_row + inner / UINT64_C(32)]);
    const float weight_scale = e8m0_to_float(
        weight_block_scales[column * blocks_per_row + inner / UINT64_C(32)]);
    partial += e2m1_to_float(activation_code) * activation_scale *
               e2m1_to_float(weight_code) * weight_scale;
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    partial += __shfl_down(partial, offset, 32U);
  }
  __shared__ float wave_sums[8];
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  if (lane == 0U) {
    wave_sums[wave] = partial;
  }
  __syncthreads();
  if (wave == 0U) {
    partial = lane < 8U ? wave_sums[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
      partial += __shfl_down(partial, offset, 32U);
    }
    if (lane == 0U) {
      output[output_index] = float_to_bf16_rne_bits(partial);
    }
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_mxfp4_w4a4_block32_decode_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  sllm_matmul_mxfp4_w4a4_block32_body(packed_activation,
                                      activation_block_scales, packed_weight,
                                      weight_block_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_mxfp4_w4a4_block32_prefill_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  sllm_matmul_mxfp4_w4a4_block32_body(packed_activation,
                                      activation_block_scales, packed_weight,
                                      weight_block_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(32, 1) void sllm_matmul_bf16_to_mxfp8_e4m3_block32_v1(
    const uint16_t *const activation, uint8_t *const quantized,
    uint8_t *const block_scales, const uint64_t m, const uint64_t k) {
  const uint64_t blocks_per_row = k / UINT64_C(32);
  const uint64_t block_index = blockIdx.x;
  if (block_index >= m * blocks_per_row) {
    return;
  }
  const uint64_t row = block_index / blocks_per_row;
  const uint64_t block = block_index - row * blocks_per_row;
  const uint64_t base = block * UINT64_C(32);
  const uint32_t lane = threadIdx.x;
  const float value = bf16_to_float(activation[row * k + base + lane]);
  uint32_t has_nan = static_cast<uint32_t>(isnan(value));
  float maximum = has_nan != 0U ? 0.0F : fabsf(value);
  maximum = sllm_lowp::wave_amax(maximum);
  has_nan = sllm_lowp::wave_or(has_nan);
  uint32_t scale = 0U;
  if (lane == 0U) {
    scale = sllm_lowp::BlockCodec<sllm_lowp::Mxfp8E4Block32>::scale_code(
        maximum, has_nan != 0U);
    block_scales[block_index] = static_cast<uint8_t>(scale);
  }
  scale = __shfl(scale, 0U, 32U);
  const float decoded_scale = e8m0_to_float(static_cast<uint8_t>(scale));
  quantized[row * k + base + lane] =
      isfinite(decoded_scale) && decoded_scale > 0.0F
          ? float_to_e4m3fn(value / decoded_scale)
          : 0U;
}

extern "C" __global__
__launch_bounds__(32, 1) void sllm_matmul_bf16_to_mxfp6_e3m2_block32_v1(
    const uint16_t *const activation, uint8_t *const packed_activation,
    uint8_t *const block_scales, const uint64_t m, const uint64_t k) {
  const uint64_t blocks_per_row = k / UINT64_C(32);
  const uint64_t block_index = blockIdx.x;
  if (block_index >= m * blocks_per_row) {
    return;
  }
  const uint64_t row = block_index / blocks_per_row;
  const uint64_t block = block_index - row * blocks_per_row;
  const uint64_t base = block * UINT64_C(32);
  const uint32_t lane = threadIdx.x;
  const float value = bf16_to_float(activation[row * k + base + lane]);
  uint32_t has_nan = static_cast<uint32_t>(isnan(value));
  float maximum = has_nan != 0U ? 0.0F : fabsf(value);
  maximum = sllm_lowp::wave_amax(maximum);
  has_nan = sllm_lowp::wave_or(has_nan);
  uint32_t scale = 0U;
  if (lane == 0U) {
    scale = sllm_lowp::BlockCodec<sllm_lowp::Mxfp6E3Block32>::scale_code(
        maximum, has_nan != 0U);
    block_scales[block_index] = static_cast<uint8_t>(scale);
  }
  scale = __shfl(scale, 0U, 32U);
  const float decoded_scale = e8m0_to_float(static_cast<uint8_t>(scale));
  const uint32_t code =
      isfinite(decoded_scale) && decoded_scale > 0.0F
          ? static_cast<uint32_t>(float_to_e3m2(value / decoded_scale))
          : 0U;
  const uint32_t group = lane & ~UINT32_C(3);
  const uint32_t packed =
      __shfl(code, static_cast<int>(group), 32U) |
      (__shfl(code, static_cast<int>(group + 1U), 32U) << 6U) |
      (__shfl(code, static_cast<int>(group + 2U), 32U) << 12U) |
      (__shfl(code, static_cast<int>(group + 3U), 32U) << 18U);
  if ((lane & UINT32_C(3)) == 0U) {
    const uint64_t row_bytes = k * UINT64_C(3) / UINT64_C(4);
    const uint64_t destination = row * row_bytes + block * UINT64_C(24) +
                                 (lane / UINT32_C(4)) * UINT64_C(3);
    packed_activation[destination] = static_cast<uint8_t>(packed);
    packed_activation[destination + 1U] = static_cast<uint8_t>(packed >> 8U);
    packed_activation[destination + 2U] = static_cast<uint8_t>(packed >> 16U);
  }
}

__device__ __forceinline__ void sllm_matmul_mxfp8_w8a8_block32_body(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  const uint64_t output_index = blockIdx.x;
  if (output_index >= m * n) {
    return;
  }
  const uint64_t row = output_index / n;
  const uint64_t column = output_index - row * n;
  const uint64_t blocks_per_row = k / UINT64_C(32);
  const sllm_lowp::BlockScaledView<sllm_lowp::Mxfp8E4Block32> activation_view{
      activation, activation_scales, nullptr, k, k, blocks_per_row};
  const sllm_lowp::BlockScaledView<sllm_lowp::Mxfp8E4Block32> weight_view{
      weight, weight_scales, nullptr, k, k, blocks_per_row};
  float partial = 0.0F;
  for (uint64_t inner = threadIdx.x; inner < k; inner += blockDim.x) {
    partial += sllm_lowp::BlockCodec<sllm_lowp::Mxfp8E4Block32>::load(
                   activation_view, row, static_cast<uint32_t>(inner)) *
               sllm_lowp::BlockCodec<sllm_lowp::Mxfp8E4Block32>::load(
                   weight_view, column, static_cast<uint32_t>(inner));
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    partial += __shfl_down(partial, offset, 32U);
  }
  __shared__ float wave_sums[8];
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  if (lane == 0U) {
    wave_sums[wave] = partial;
  }
  __syncthreads();
  if (wave == 0U) {
    partial = lane < 8U ? wave_sums[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
      partial += __shfl_down(partial, offset, 32U);
    }
    if (lane == 0U) {
      output[output_index] = float_to_bf16_rne_bits(partial);
    }
  }
}

__device__ __forceinline__ void sllm_matmul_mxfp6_w6a6_block32_body(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  const uint64_t output_index = blockIdx.x;
  if (output_index >= m * n) {
    return;
  }
  const uint64_t row = output_index / n;
  const uint64_t column = output_index - row * n;
  const uint64_t blocks_per_row = k / UINT64_C(32);
  const uint64_t row_bytes = k * UINT64_C(3) / UINT64_C(4);
  const sllm_lowp::BlockScaledView<sllm_lowp::Mxfp6E3Block32> activation_view{
      activation, activation_scales, nullptr, k, row_bytes, blocks_per_row};
  const sllm_lowp::BlockScaledView<sllm_lowp::Mxfp6E3Block32> weight_view{
      weight, weight_scales, nullptr, k, row_bytes, blocks_per_row};
  float partial = 0.0F;
  for (uint64_t inner = threadIdx.x; inner < k; inner += blockDim.x) {
    partial += sllm_lowp::BlockCodec<sllm_lowp::Mxfp6E3Block32>::load(
                   activation_view, row, static_cast<uint32_t>(inner)) *
               sllm_lowp::BlockCodec<sllm_lowp::Mxfp6E3Block32>::load(
                   weight_view, column, static_cast<uint32_t>(inner));
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    partial += __shfl_down(partial, offset, 32U);
  }
  __shared__ float wave_sums[8];
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  if (lane == 0U) {
    wave_sums[wave] = partial;
  }
  __syncthreads();
  if (wave == 0U) {
    partial = lane < 8U ? wave_sums[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
      partial += __shfl_down(partial, offset, 32U);
    }
    if (lane == 0U) {
      output[output_index] = float_to_bf16_rne_bits(partial);
    }
  }
}

#define SLLM_DEFINE_MX_WA_KERNEL(symbol, body)                                 \
  extern "C" __global__ __launch_bounds__(256, 1) void symbol(                 \
      const uint8_t *const activation, const uint8_t *const activation_scales, \
      const uint8_t *const weight, const uint8_t *const weight_scales,         \
      uint16_t *const output, const uint64_t m, const uint64_t k,              \
      const uint64_t n) {                                                      \
    body(activation, activation_scales, weight, weight_scales, output, m, k,   \
         n);                                                                   \
  }

SLLM_DEFINE_MX_WA_KERNEL(sllm_matmul_mxfp8_w8a8_e4m3_block32_decode_v1,
                         sllm_matmul_mxfp8_w8a8_block32_body)
SLLM_DEFINE_MX_WA_KERNEL(sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_v1,
                         sllm_matmul_mxfp8_w8a8_block32_body)
SLLM_DEFINE_MX_WA_KERNEL(sllm_matmul_mxfp6_w6a6_e3m2_block32_decode_v1,
                         sllm_matmul_mxfp6_w6a6_block32_body)
SLLM_DEFINE_MX_WA_KERNEL(sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_v1,
                         sllm_matmul_mxfp6_w6a6_block32_body)

#undef SLLM_DEFINE_MX_WA_KERNEL

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_row8_v2(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t rows_per_workgroup = 8U;
  constexpr uint32_t tile_k = 256U;
  constexpr uint32_t blocks_per_tile = tile_k / 32U;
  __shared__ float weight_tile[tile_k];
  __shared__ float weight_scale_tile[blocks_per_tile];
  __shared__ float activation_scale_tile[rows_per_workgroup][blocks_per_tile];
  const uint64_t column = static_cast<uint64_t>(blockIdx.x) % n;
  const uint64_t row_base =
      (static_cast<uint64_t>(blockIdx.x) / n) * rows_per_workgroup;
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  const uint64_t row = row_base + wave;
  const uint64_t blocks_per_row = k / UINT64_C(32);
  float accumulator = 0.0F;
  for (uint64_t base = 0U; base < k; base += tile_k) {
    if (threadIdx.x < blocks_per_tile) {
      const uint64_t block = base / UINT64_C(32) + threadIdx.x;
      weight_scale_tile[threadIdx.x] =
          block < blocks_per_row
              ? e8m0_to_float(weight_scales[column * blocks_per_row + block])
              : 0.0F;
    }
    if (threadIdx.x < rows_per_workgroup * blocks_per_tile) {
      const uint32_t scale_row = threadIdx.x / blocks_per_tile;
      const uint32_t scale_block = threadIdx.x % blocks_per_tile;
      const uint64_t source_row = row_base + scale_row;
      const uint64_t block = base / UINT64_C(32) + scale_block;
      activation_scale_tile[scale_row][scale_block] =
          source_row < m && block < blocks_per_row
              ? e8m0_to_float(
                    activation_scales[source_row * blocks_per_row + block])
              : 0.0F;
    }
    const uint64_t global_inner = base + threadIdx.x;
    weight_tile[threadIdx.x] = global_inner < k
                                   ? e4m3fn_to_float(__builtin_nontemporal_load(
                                         weight + column * k + global_inner))
                                   : 0.0F;
    __syncthreads();
    if (row < m) {
      const uint32_t valid = static_cast<uint32_t>(
          k - base < tile_k ? k - base : static_cast<uint64_t>(tile_k));
      for (uint32_t offset = lane; offset < valid; offset += wave_width) {
        float term = e4m3fn_to_float(activation[row * k + base + offset]) *
                     activation_scale_tile[wave][offset / 32U];
        term *= weight_tile[offset];
        term *= weight_scale_tile[offset / 32U];
        accumulator += term;
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    accumulator += __shfl_down(accumulator, offset, 32U);
  }
  if (lane == 0U && row < m) {
    output[row * n + column] = float_to_bf16_rne_bits(accumulator);
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_row8_v2(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t rows_per_workgroup = 8U;
  constexpr uint32_t tile_k = 256U;
  constexpr uint32_t blocks_per_tile = tile_k / 32U;
  __shared__ float weight_tile[tile_k];
  __shared__ float weight_scale_tile[blocks_per_tile];
  __shared__ float activation_scale_tile[rows_per_workgroup][blocks_per_tile];
  const uint64_t column = static_cast<uint64_t>(blockIdx.x) % n;
  const uint64_t row_base =
      (static_cast<uint64_t>(blockIdx.x) / n) * rows_per_workgroup;
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  const uint64_t row = row_base + wave;
  const uint64_t blocks_per_row = k / UINT64_C(32);
  const uint64_t row_bytes = k * UINT64_C(3) / UINT64_C(4);
  const uint8_t *const weight_row = weight + column * row_bytes;
  const uint8_t *const activation_row =
      activation + (row < m ? row : 0U) * row_bytes;
  float accumulator = 0.0F;
  for (uint64_t base = 0U; base < k; base += tile_k) {
    if (threadIdx.x < blocks_per_tile) {
      const uint64_t block = base / UINT64_C(32) + threadIdx.x;
      weight_scale_tile[threadIdx.x] =
          block < blocks_per_row
              ? e8m0_to_float(weight_scales[column * blocks_per_row + block])
              : 0.0F;
    }
    if (threadIdx.x < rows_per_workgroup * blocks_per_tile) {
      const uint32_t scale_row = threadIdx.x / blocks_per_tile;
      const uint32_t scale_block = threadIdx.x % blocks_per_tile;
      const uint64_t source_row = row_base + scale_row;
      const uint64_t block = base / UINT64_C(32) + scale_block;
      activation_scale_tile[scale_row][scale_block] =
          source_row < m && block < blocks_per_row
              ? e8m0_to_float(
                    activation_scales[source_row * blocks_per_row + block])
              : 0.0F;
    }
    const uint64_t global_inner = base + threadIdx.x;
    weight_tile[threadIdx.x] =
        global_inner < k
            ? e3m2_to_float(packed_e3m2_at(weight_row, global_inner))
            : 0.0F;
    __syncthreads();
    if (row < m) {
      const uint32_t valid = static_cast<uint32_t>(
          k - base < tile_k ? k - base : static_cast<uint64_t>(tile_k));
      for (uint32_t offset = lane; offset < valid; offset += wave_width) {
        float term =
            e3m2_to_float(packed_e3m2_at(activation_row, base + offset)) *
            activation_scale_tile[wave][offset / 32U];
        term *= weight_tile[offset];
        term *= weight_scale_tile[offset / 32U];
        accumulator += term;
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    accumulator += __shfl_down(accumulator, offset, 32U);
  }
  if (lane == 0U && row < m) {
    output[row * n + column] = float_to_bf16_rne_bits(accumulator);
  }
}

struct Mxfp8MmqFormat {
  __device__ __forceinline__ static uint64_t row_bytes(const uint64_t k) {
    return k;
  }

  __device__ __forceinline__ static float
  load_activation(const uint8_t *const row, const uint64_t index) {
    return sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode_mx_value_plane(
        row[index]);
  }

  __device__ __forceinline__ static float load_weight(const uint8_t *const row,
                                                      const uint64_t index) {
    return decode_weight_byte(__builtin_nontemporal_load(row + index));
  }

  __device__ __forceinline__ static float
  decode_weight_byte(const uint8_t bits) {
    return sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode_mx_value_plane(
        bits);
  }
};

struct Mxfp6MmqFormat {
  __device__ __forceinline__ static uint64_t row_bytes(const uint64_t k) {
    return k * UINT64_C(3) / UINT64_C(4);
  }

  __device__ __forceinline__ static float
  load_activation(const uint8_t *const row, const uint64_t index) {
    return e3m2_to_float(packed_e3m2_at(row, index));
  }

  __device__ __forceinline__ static float load_weight(const uint8_t *const row,
                                                      const uint64_t index) {
    return e3m2_to_float(packed_e3m2_at(row, index));
  }
};

// Phase 70 keeps the resident 6-bit stream unchanged and converts only the
// value consumed by the established MXFP8-style MMQ tile.  Both activation
// and weight ingress use the same exact code transform.
struct Mxfp6ViaE4MmqFormat {
  __device__ __forceinline__ static uint64_t row_bytes(const uint64_t k) {
    return k * UINT64_C(3) / UINT64_C(4);
  }

  __device__ __forceinline__ static float decode_e3m2_code(const uint8_t code) {
    return sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode_mx_value_plane(
        sllm_lowp::e3m2_to_e4m3fn_exact_bits(code));
  }

  __device__ __forceinline__ static float
  load_activation(const uint8_t *const row, const uint64_t index) {
    return decode_e3m2_code(packed_e3m2_at(row, index));
  }

  __device__ __forceinline__ static float load_weight(const uint8_t *const row,
                                                      const uint64_t index) {
    return decode_e3m2_code(packed_e3m2_at(row, index));
  }
};

// Keep packed-value ingress independent from the MMQ arithmetic schedule.
// The scalar policy remains the format-generic default used by MXFP8 and
// MXFP6.  The vector policy is instantiated only for byte-addressed MXFP8;
// future packed formats can provide their own ingress policy without cloning
// the row/column/K decomposition.
struct MmqScalarWeightIngress {
  static constexpr uint32_t values_per_load = 1U;

  template <typename Format>
  __device__ __forceinline__ static void
  stage(const uint8_t *const row, const uint64_t index, float *const output) {
    output[0] = Format::load_weight(row, index);
  }
};

struct Mxfp8MmqVector32WeightIngress {
  static constexpr uint32_t values_per_load = 4U;

  template <typename Format>
  __device__ __forceinline__ static void
  stage(const uint8_t *const row, const uint64_t index, float *const output) {
    const uint32_t packed = __builtin_nontemporal_load(
        reinterpret_cast<const uint32_t *>(row + index));
#pragma unroll
    for (uint32_t byte = 0U; byte < values_per_load; ++byte) {
      output[byte] = Format::decode_weight_byte(
          static_cast<uint8_t>(packed >> (byte * 8U)));
    }
  }
};

struct Mxfp6MmqPacked24ViaE4WeightIngress {
  static constexpr uint32_t values_per_load = 4U;

  template <typename Format>
  __device__ __forceinline__ static void
  stage(const uint8_t *const row, const uint64_t index, float *const output) {
    const uint64_t byte_index = (index >> 2U) * UINT64_C(3);
    const uint32_t packed =
        static_cast<uint32_t>(__builtin_nontemporal_load(row + byte_index)) |
        (static_cast<uint32_t>(
             __builtin_nontemporal_load(row + byte_index + UINT64_C(1)))
         << 8U) |
        (static_cast<uint32_t>(
             __builtin_nontemporal_load(row + byte_index + UINT64_C(2)))
         << 16U);
#pragma unroll
    for (uint32_t value = 0U; value < values_per_load; ++value) {
      output[value] = Format::decode_e3m2_code(
          static_cast<uint8_t>(packed >> (value * 6U)) & UINT8_C(0x3f));
    }
  }
};

// This candidate borrows llama.cpp MMQ's multi-row/multi-column/K-tile
// decomposition, but intentionally retains sLLM's packed MX values, E8M0
// scales, FP32 accumulation, and row8 reduction order.  In particular, it
// does not introduce the llama.cpp Q8_1 activation or integer dot path.
template <typename Format, uint32_t Columns,
          typename WeightIngress = MmqScalarWeightIngress,
          bool RegisterBlockScales = false>
__device__ __forceinline__ void sllm_matmul_mx_wa_mmq_columns_body(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t rows_per_workgroup = 8U;
  constexpr uint32_t tile_k = 256U;
  constexpr uint32_t blocks_per_tile = tile_k / 32U;
  constexpr uint32_t ingress_values = WeightIngress::values_per_load;
  static_assert(tile_k % ingress_values == 0U);
  __shared__ float weight_tile[Columns][tile_k];
  __shared__ float weight_scale_tile[Columns][blocks_per_tile];
  __shared__ float activation_scale_tile[rows_per_workgroup][blocks_per_tile];
  const uint64_t column_tiles =
      (n + static_cast<uint64_t>(Columns) - 1U) / Columns;
  const uint64_t tile_index = static_cast<uint64_t>(blockIdx.x);
  const uint64_t column_base = (tile_index % column_tiles) * Columns;
  const uint64_t row_base = (tile_index / column_tiles) * rows_per_workgroup;
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  const uint64_t row = row_base + wave;
  const uint64_t blocks_per_row = k / UINT64_C(32);
  const uint64_t row_bytes = Format::row_bytes(k);
  const uint8_t *const activation_row =
      activation + (row < m ? row : 0U) * row_bytes;
  float accumulators[Columns] = {};
  for (uint64_t base = 0U; base < k; base += tile_k) {
    for (uint32_t index = threadIdx.x; index < Columns * blocks_per_tile;
         index += blockDim.x) {
      const uint32_t local_column = index / blocks_per_tile;
      const uint32_t scale_block = index % blocks_per_tile;
      const uint64_t column = column_base + local_column;
      const uint64_t block = base / UINT64_C(32) + scale_block;
      weight_scale_tile[local_column][scale_block] =
          column < n && block < blocks_per_row
              ? e8m0_to_float(weight_scales[column * blocks_per_row + block])
              : 0.0F;
    }
    if (threadIdx.x < rows_per_workgroup * blocks_per_tile) {
      const uint32_t scale_row = threadIdx.x / blocks_per_tile;
      const uint32_t scale_block = threadIdx.x % blocks_per_tile;
      const uint64_t source_row = row_base + scale_row;
      const uint64_t block = base / UINT64_C(32) + scale_block;
      activation_scale_tile[scale_row][scale_block] =
          source_row < m && block < blocks_per_row
              ? e8m0_to_float(
                    activation_scales[source_row * blocks_per_row + block])
              : 0.0F;
    }
    constexpr uint32_t ingress_groups_per_column = tile_k / ingress_values;
    for (uint32_t index = threadIdx.x;
         index < Columns * ingress_groups_per_column; index += blockDim.x) {
      const uint32_t local_column = index / ingress_groups_per_column;
      const uint32_t group = index % ingress_groups_per_column;
      const uint32_t offset = group * ingress_values;
      const uint64_t column = column_base + local_column;
      const uint64_t global_inner = base + offset;
      if (column < n && global_inner + ingress_values <= k) {
        WeightIngress::template stage<Format>(
            weight + column * row_bytes, global_inner,
            &weight_tile[local_column][offset]);
      } else {
#pragma unroll
        for (uint32_t value = 0U; value < ingress_values; ++value) {
          weight_tile[local_column][offset + value] = 0.0F;
        }
      }
    }
    __syncthreads();
    if (row < m) {
      const uint32_t valid = static_cast<uint32_t>(
          k - base < tile_k ? k - base : static_cast<uint64_t>(tile_k));
      if constexpr (RegisterBlockScales) {
#pragma unroll
        for (uint32_t scale_block = 0U; scale_block < blocks_per_tile;
             ++scale_block) {
          const uint32_t offset = scale_block * wave_width + lane;
          if (offset >= valid) {
            continue;
          }
          const float activation_scale =
              activation_scale_tile[wave][scale_block];
          const float activation_value =
              Format::load_activation(activation_row, base + offset) *
              activation_scale;
#pragma unroll
          for (uint32_t local_column = 0U; local_column < Columns;
               ++local_column) {
            const float weight_scale =
                weight_scale_tile[local_column][scale_block];
            float term = activation_value * weight_tile[local_column][offset];
            term *= weight_scale;
            accumulators[local_column] += term;
          }
        }
      } else {
        for (uint32_t offset = lane; offset < valid; offset += wave_width) {
          const float activation_value =
              Format::load_activation(activation_row, base + offset) *
              activation_scale_tile[wave][offset / 32U];
#pragma unroll
          for (uint32_t local_column = 0U; local_column < Columns;
               ++local_column) {
            float term = activation_value * weight_tile[local_column][offset];
            term *= weight_scale_tile[local_column][offset / 32U];
            accumulators[local_column] += term;
          }
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
#pragma unroll
    for (uint32_t local_column = 0U; local_column < Columns; ++local_column) {
      accumulators[local_column] +=
          __shfl_down(accumulators[local_column], offset, 32U);
    }
  }
  if (lane == 0U && row < m) {
#pragma unroll
    for (uint32_t local_column = 0U; local_column < Columns; ++local_column) {
      const uint64_t column = column_base + local_column;
      if (column < n) {
        output[row * n + column] =
            float_to_bf16_rne_bits(accumulators[local_column]);
      }
    }
  }
}

#define SLLM_DEFINE_MX_WA_MMQ_COLUMNS_KERNEL(symbol, format, columns)          \
  extern "C" __global__ __launch_bounds__(256, 1) void symbol(                 \
      const uint8_t *const activation, const uint8_t *const activation_scales, \
      const uint8_t *const weight, const uint8_t *const weight_scales,         \
      uint16_t *const output, const uint64_t m, const uint64_t k,              \
      const uint64_t n) {                                                      \
    sllm_matmul_mx_wa_mmq_columns_body<format, columns>(                       \
        activation, activation_scales, weight, weight_scales, output, m, k,    \
        n);                                                                    \
  }

SLLM_DEFINE_MX_WA_MMQ_COLUMNS_KERNEL(
    sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_mmq_col4_v4, Mxfp8MmqFormat, 4U)
SLLM_DEFINE_MX_WA_MMQ_COLUMNS_KERNEL(
    sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_mmq_col8_v4, Mxfp8MmqFormat, 8U)
SLLM_DEFINE_MX_WA_MMQ_COLUMNS_KERNEL(sllm_mxfp8_w8a8_gfx1030_mmq_col16_v1,
                                     Mxfp8MmqFormat, 16U)
SLLM_DEFINE_MX_WA_MMQ_COLUMNS_KERNEL(sllm_mxfp8_w8a8_gfx1030_mmq_col32_v1,
                                     Mxfp8MmqFormat, 32U)
SLLM_DEFINE_MX_WA_MMQ_COLUMNS_KERNEL(
    sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_mmq_col4_v4, Mxfp6MmqFormat, 4U)
SLLM_DEFINE_MX_WA_MMQ_COLUMNS_KERNEL(
    sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_mmq_col8_v4, Mxfp6MmqFormat, 8U)

#undef SLLM_DEFINE_MX_WA_MMQ_COLUMNS_KERNEL

#define SLLM_DEFINE_MXFP8_GFX1030_MMQ_PHASE69_KERNEL(symbol, ingress,          \
                                                     register_scales)          \
  extern "C" __global__ __launch_bounds__(256, 1) void symbol(                 \
      const uint8_t *const activation, const uint8_t *const activation_scales, \
      const uint8_t *const weight, const uint8_t *const weight_scales,         \
      uint16_t *const output, const uint64_t m, const uint64_t k,              \
      const uint64_t n) {                                                      \
    sllm_matmul_mx_wa_mmq_columns_body<Mxfp8MmqFormat, 8U, ingress,            \
                                       register_scales>(                       \
        activation, activation_scales, weight, weight_scales, output, m, k,    \
        n);                                                                    \
  }

SLLM_DEFINE_MXFP8_GFX1030_MMQ_PHASE69_KERNEL(
    sllm_mxfp8_w8a8_gfx1030_mmq_col8_regscale_v1, MmqScalarWeightIngress, true)
SLLM_DEFINE_MXFP8_GFX1030_MMQ_PHASE69_KERNEL(
    sllm_mxfp8_w8a8_gfx1030_mmq_col8_vector32_v1, Mxfp8MmqVector32WeightIngress,
    false)
SLLM_DEFINE_MXFP8_GFX1030_MMQ_PHASE69_KERNEL(
    sllm_mxfp8_w8a8_gfx1030_mmq_col8_regscale_vector32_v1,
    Mxfp8MmqVector32WeightIngress, true)

#undef SLLM_DEFINE_MXFP8_GFX1030_MMQ_PHASE69_KERNEL

extern "C" __global__
__launch_bounds__(256, 1) void sllm_mxfp6_w6a6_gfx1030_mmq_col8_via_e4m3_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mx_wa_mmq_columns_body<Mxfp6ViaE4MmqFormat, 8U,
                                     Mxfp6MmqPacked24ViaE4WeightIngress, false>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

// Phase 75 separates the block-format ingress from the gfx1030 half2 tile.
// Both MXFP8 and MXFP6 therefore share the same 16x16-thread output mapping,
// K-stage policy, scale handling, dot2 tree, FP32 accumulation, and BF16 RNE
// store.  Only the resident value-plane load/expansion differs by format.
struct Mxfp8E4M3Half2Ingress {
  static constexpr uint32_t values_per_load = 4U;

  __device__ __forceinline__ static uint64_t row_bytes(const uint64_t k) {
    return k;
  }

  __device__ __forceinline__ static void stage(const uint8_t *const row,
                                               const uint64_t index,
                                               uint16_t *const output) {
    const uint32_t packed = __builtin_nontemporal_load(
        reinterpret_cast<const uint32_t *>(row + index));
#pragma unroll
    for (uint32_t lane = 0U; lane < values_per_load; ++lane) {
      output[lane] = sllm_lowp::e4m3fn_to_fp16_bits(
          static_cast<uint8_t>(packed >> (lane * 8U)));
    }
  }
};

struct Mxfp6E3M2ScalarHalf2Ingress {
  static constexpr uint32_t values_per_load = 1U;

  __device__ __forceinline__ static uint64_t row_bytes(const uint64_t k) {
    return k * UINT64_C(3) / UINT64_C(4);
  }

  __device__ __forceinline__ static void stage(const uint8_t *const row,
                                               const uint64_t index,
                                               uint16_t *const output) {
    output[0] = sllm_lowp::e3m2_to_fp16_bits(packed_e3m2_at(row, index));
  }
};

struct Mxfp6E3M2Packed4Half2Ingress {
  static constexpr uint32_t values_per_load = 4U;

  __device__ __forceinline__ static uint64_t row_bytes(const uint64_t k) {
    return k * UINT64_C(3) / UINT64_C(4);
  }

  __device__ __forceinline__ static void stage(const uint8_t *const row,
                                               const uint64_t index,
                                               uint16_t *const output) {
    const uint32_t packed = sllm_lowp::packed_e3m2x4_at(row, index);
#pragma unroll
    for (uint32_t lane = 0U; lane < values_per_load; ++lane) {
      output[lane] = sllm_lowp::e3m2_to_fp16_bits(
          static_cast<uint8_t>(packed >> (lane * 6U)));
    }
  }
};

template <typename FormatIngress, uint32_t RowsPerWorkgroup,
          uint32_t ColumnsPerWorkgroup, uint32_t BlocksPerStage>
__device__ __forceinline__ void sllm_matmul_gfx1030_half2_dot2_body(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t thread_rows = 16U;
  constexpr uint32_t thread_columns = 16U;
  constexpr uint32_t block_k = 32U;
  constexpr uint32_t tile_k = block_k * BlocksPerStage;
  constexpr uint32_t rows_per_thread = RowsPerWorkgroup / thread_rows;
  constexpr uint32_t columns_per_thread = ColumnsPerWorkgroup / thread_columns;
  constexpr uint32_t ingress_values = FormatIngress::values_per_load;
  constexpr uint32_t ingress_groups_per_row = tile_k / ingress_values;
  static_assert(RowsPerWorkgroup >= thread_rows &&
                RowsPerWorkgroup % thread_rows == 0U);
  static_assert(ColumnsPerWorkgroup >= thread_columns &&
                ColumnsPerWorkgroup % thread_columns == 0U);
  static_assert(BlocksPerStage > 0U && tile_k % ingress_values == 0U);
  __shared__ uint16_t activation_tile[RowsPerWorkgroup][tile_k];
  __shared__ uint16_t weight_tile[ColumnsPerWorkgroup][tile_k];
  __shared__ float activation_scale_tile[RowsPerWorkgroup][BlocksPerStage];
  __shared__ float weight_scale_tile[ColumnsPerWorkgroup][BlocksPerStage];

  const uint64_t column_tiles =
      (n + static_cast<uint64_t>(ColumnsPerWorkgroup) - 1U) /
      ColumnsPerWorkgroup;
  const uint64_t tile_index = static_cast<uint64_t>(blockIdx.x);
  const uint64_t column_base =
      (tile_index % column_tiles) * ColumnsPerWorkgroup;
  const uint64_t row_base = (tile_index / column_tiles) * RowsPerWorkgroup;
  const uint32_t thread = threadIdx.x;
  const uint32_t local_column = thread & UINT32_C(15);
  const uint32_t local_row = thread >> UINT32_C(4);
  const uint64_t blocks_per_row = k / UINT64_C(32);
  const uint64_t row_bytes = FormatIngress::row_bytes(k);
  float accumulators[rows_per_thread][columns_per_thread] = {};

  for (uint64_t base = 0U; base < k; base += tile_k) {
    const uint64_t stage_block = base / UINT64_C(32);
    for (uint32_t index = thread; index < RowsPerWorkgroup * BlocksPerStage;
         index += sllm_matmul_kernel::kWorkgroupSize) {
      const uint32_t row = index / BlocksPerStage;
      const uint32_t local_block = index % BlocksPerStage;
      const uint64_t source_row = row_base + row;
      const uint64_t block = stage_block + local_block;
      activation_scale_tile[row][local_block] =
          source_row < m && block < blocks_per_row
              ? e8m0_to_float(
                    activation_scales[source_row * blocks_per_row + block])
              : 0.0F;
    }
    for (uint32_t index = thread; index < ColumnsPerWorkgroup * BlocksPerStage;
         index += sllm_matmul_kernel::kWorkgroupSize) {
      const uint32_t column = index / BlocksPerStage;
      const uint32_t local_block = index % BlocksPerStage;
      const uint64_t source_column = column_base + column;
      const uint64_t block = stage_block + local_block;
      weight_scale_tile[column][local_block] =
          source_column < n && block < blocks_per_row
              ? e8m0_to_float(
                    weight_scales[source_column * blocks_per_row + block])
              : 0.0F;
    }
    for (uint32_t index = thread;
         index < RowsPerWorkgroup * ingress_groups_per_row;
         index += sllm_matmul_kernel::kWorkgroupSize) {
      const uint32_t row = index / ingress_groups_per_row;
      const uint32_t offset = (index % ingress_groups_per_row) * ingress_values;
      const uint64_t source_row = row_base + row;
      const uint64_t global_inner = base + offset;
      if (source_row < m && global_inner + ingress_values <= k) {
        FormatIngress::stage(activation + source_row * row_bytes, global_inner,
                             &activation_tile[row][offset]);
      } else {
#pragma unroll
        for (uint32_t lane = 0U; lane < ingress_values; ++lane) {
          activation_tile[row][offset + lane] = UINT16_C(0);
        }
      }
    }
    for (uint32_t index = thread;
         index < ColumnsPerWorkgroup * ingress_groups_per_row;
         index += sllm_matmul_kernel::kWorkgroupSize) {
      const uint32_t column = index / ingress_groups_per_row;
      const uint32_t offset = (index % ingress_groups_per_row) * ingress_values;
      const uint64_t source_column = column_base + column;
      const uint64_t global_inner = base + offset;
      if (source_column < n && global_inner + ingress_values <= k) {
        FormatIngress::stage(weight + source_column * row_bytes, global_inner,
                             &weight_tile[column][offset]);
      } else {
#pragma unroll
        for (uint32_t lane = 0U; lane < ingress_values; ++lane) {
          weight_tile[column][offset + lane] = UINT16_C(0);
        }
      }
    }
    __syncthreads();

    const uint32_t valid_blocks =
        static_cast<uint32_t>(min(static_cast<uint64_t>(BlocksPerStage),
                                  (k - base) / static_cast<uint64_t>(block_k)));
#pragma unroll
    for (uint32_t local_block = 0U; local_block < BlocksPerStage;
         ++local_block) {
      if (local_block >= valid_blocks) {
        continue;
      }
      float block_sums[rows_per_thread][columns_per_thread] = {};
#pragma unroll
      for (uint32_t inner = local_block * block_k;
           inner < (local_block + 1U) * block_k; inner += 2U) {
        __half2 activation_pairs[rows_per_thread];
        __half2 weight_pairs[columns_per_thread];
#pragma unroll
        for (uint32_t row = 0U; row < rows_per_thread; ++row) {
          activation_pairs[row] = *reinterpret_cast<const __half2 *>(
              &activation_tile[local_row + row * thread_rows][inner]);
        }
#pragma unroll
        for (uint32_t column = 0U; column < columns_per_thread; ++column) {
          weight_pairs[column] = *reinterpret_cast<const __half2 *>(
              &weight_tile[local_column + column * thread_columns][inner]);
        }
#pragma unroll
        for (uint32_t row = 0U; row < rows_per_thread; ++row) {
#pragma unroll
          for (uint32_t column = 0U; column < columns_per_thread; ++column) {
            block_sums[row][column] =
                amd_mixed_dot(activation_pairs[row], weight_pairs[column],
                              block_sums[row][column], false);
          }
        }
      }
#pragma unroll
      for (uint32_t row = 0U; row < rows_per_thread; ++row) {
        const float activation_scale =
            activation_scale_tile[local_row + row * thread_rows][local_block];
#pragma unroll
        for (uint32_t column = 0U; column < columns_per_thread; ++column) {
          const float weight_scale =
              weight_scale_tile[local_column + column * thread_columns]
                               [local_block];
          accumulators[row][column] +=
              block_sums[row][column] * activation_scale * weight_scale;
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t row = 0U; row < rows_per_thread; ++row) {
#pragma unroll
    for (uint32_t column = 0U; column < columns_per_thread; ++column) {
      const uint64_t output_row = row_base + local_row + row * thread_rows;
      const uint64_t output_column =
          column_base + local_column + column * thread_columns;
      if (output_row < m && output_column < n) {
        output[output_row * n + output_column] =
            float_to_bf16_rne_bits(accumulators[row][column]);
      }
    }
  }
}

template <typename FormatIngress, uint32_t RowsPerWorkgroup,
          uint32_t ColumnsPerWorkgroup>
__device__ __forceinline__ void sllm_stage_gfx1030_half2_k32(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    const uint64_t m, const uint64_t k, const uint64_t n,
    const uint64_t row_base, const uint64_t column_base, const uint64_t base,
    uint16_t *const activation_tile, uint16_t *const weight_tile,
    float *const activation_scale_tile, float *const weight_scale_tile) {
  constexpr uint32_t block_k = 32U;
  constexpr uint32_t ingress_values = FormatIngress::values_per_load;
  constexpr uint32_t ingress_groups_per_row = block_k / ingress_values;
  static_assert(block_k % ingress_values == 0U);
  const uint32_t thread = threadIdx.x;
  const uint64_t block = base / UINT64_C(32);
  const uint64_t blocks_per_row = k / UINT64_C(32);
  const uint64_t row_bytes = FormatIngress::row_bytes(k);
  for (uint32_t row = thread; row < RowsPerWorkgroup;
       row += sllm_matmul_kernel::kWorkgroupSize) {
    const uint64_t source_row = row_base + row;
    activation_scale_tile[row] =
        source_row < m && block < blocks_per_row
            ? e8m0_to_float(
                  activation_scales[source_row * blocks_per_row + block])
            : 0.0F;
  }
  for (uint32_t column = thread; column < ColumnsPerWorkgroup;
       column += sllm_matmul_kernel::kWorkgroupSize) {
    const uint64_t source_column = column_base + column;
    weight_scale_tile[column] =
        source_column < n && block < blocks_per_row
            ? e8m0_to_float(
                  weight_scales[source_column * blocks_per_row + block])
            : 0.0F;
  }
  for (uint32_t index = thread;
       index < RowsPerWorkgroup * ingress_groups_per_row;
       index += sllm_matmul_kernel::kWorkgroupSize) {
    const uint32_t row = index / ingress_groups_per_row;
    const uint32_t offset = (index % ingress_groups_per_row) * ingress_values;
    const uint64_t source_row = row_base + row;
    if (source_row < m && base + offset + ingress_values <= k) {
      FormatIngress::stage(activation + source_row * row_bytes, base + offset,
                           activation_tile + row * block_k + offset);
    } else {
#pragma unroll
      for (uint32_t lane = 0U; lane < ingress_values; ++lane) {
        activation_tile[row * block_k + offset + lane] = UINT16_C(0);
      }
    }
  }
  for (uint32_t index = thread;
       index < ColumnsPerWorkgroup * ingress_groups_per_row;
       index += sllm_matmul_kernel::kWorkgroupSize) {
    const uint32_t column = index / ingress_groups_per_row;
    const uint32_t offset = (index % ingress_groups_per_row) * ingress_values;
    const uint64_t source_column = column_base + column;
    if (source_column < n && base + offset + ingress_values <= k) {
      FormatIngress::stage(weight + source_column * row_bytes, base + offset,
                           weight_tile + column * block_k + offset);
    } else {
#pragma unroll
      for (uint32_t lane = 0U; lane < ingress_values; ++lane) {
        weight_tile[column * block_k + offset + lane] = UINT16_C(0);
      }
    }
  }
}

template <typename FormatIngress, uint32_t RowsPerWorkgroup,
          uint32_t ColumnsPerWorkgroup>
__device__ __forceinline__ void sllm_matmul_gfx1030_half2_dot2_k32_double_body(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t thread_rows = 16U;
  constexpr uint32_t thread_columns = 16U;
  constexpr uint32_t block_k = 32U;
  constexpr uint32_t rows_per_thread = RowsPerWorkgroup / thread_rows;
  constexpr uint32_t columns_per_thread = ColumnsPerWorkgroup / thread_columns;
  static_assert(RowsPerWorkgroup % thread_rows == 0U);
  static_assert(ColumnsPerWorkgroup % thread_columns == 0U);
  __shared__ uint16_t activation_tiles[2][RowsPerWorkgroup][block_k];
  __shared__ uint16_t weight_tiles[2][ColumnsPerWorkgroup][block_k];
  __shared__ float activation_scale_tiles[2][RowsPerWorkgroup];
  __shared__ float weight_scale_tiles[2][ColumnsPerWorkgroup];

  const uint64_t column_tiles =
      (n + static_cast<uint64_t>(ColumnsPerWorkgroup) - 1U) /
      ColumnsPerWorkgroup;
  const uint64_t tile_index = static_cast<uint64_t>(blockIdx.x);
  const uint64_t column_base =
      (tile_index % column_tiles) * ColumnsPerWorkgroup;
  const uint64_t row_base = (tile_index / column_tiles) * RowsPerWorkgroup;
  const uint32_t local_column = threadIdx.x & UINT32_C(15);
  const uint32_t local_row = threadIdx.x >> UINT32_C(4);
  float accumulators[rows_per_thread][columns_per_thread] = {};

  sllm_stage_gfx1030_half2_k32<FormatIngress, RowsPerWorkgroup,
                               ColumnsPerWorkgroup>(
      activation, activation_scales, weight, weight_scales, m, k, n, row_base,
      column_base, 0U, &activation_tiles[0][0][0], &weight_tiles[0][0][0],
      &activation_scale_tiles[0][0], &weight_scale_tiles[0][0]);
  __syncthreads();

  uint32_t current_buffer = 0U;
  for (uint64_t base = 0U; base < k; base += block_k) {
    const uint64_t next_base = base + block_k;
    const uint32_t next_buffer = current_buffer ^ 1U;
    if (next_base < k) {
      sllm_stage_gfx1030_half2_k32<FormatIngress, RowsPerWorkgroup,
                                   ColumnsPerWorkgroup>(
          activation, activation_scales, weight, weight_scales, m, k, n,
          row_base, column_base, next_base,
          &activation_tiles[next_buffer][0][0],
          &weight_tiles[next_buffer][0][0],
          &activation_scale_tiles[next_buffer][0],
          &weight_scale_tiles[next_buffer][0]);
    }
    float block_sums[rows_per_thread][columns_per_thread] = {};
#pragma unroll
    for (uint32_t inner = 0U; inner < block_k; inner += 2U) {
      __half2 activation_pairs[rows_per_thread];
      __half2 weight_pairs[columns_per_thread];
#pragma unroll
      for (uint32_t row = 0U; row < rows_per_thread; ++row) {
        activation_pairs[row] = *reinterpret_cast<const __half2 *>(
            &activation_tiles[current_buffer][local_row + row * thread_rows]
                             [inner]);
      }
#pragma unroll
      for (uint32_t column = 0U; column < columns_per_thread; ++column) {
        weight_pairs[column] = *reinterpret_cast<const __half2 *>(
            &weight_tiles[current_buffer]
                         [local_column + column * thread_columns][inner]);
      }
#pragma unroll
      for (uint32_t row = 0U; row < rows_per_thread; ++row) {
#pragma unroll
        for (uint32_t column = 0U; column < columns_per_thread; ++column) {
          block_sums[row][column] =
              amd_mixed_dot(activation_pairs[row], weight_pairs[column],
                            block_sums[row][column], false);
        }
      }
    }
#pragma unroll
    for (uint32_t row = 0U; row < rows_per_thread; ++row) {
      const float activation_scale =
          activation_scale_tiles[current_buffer][local_row + row * thread_rows];
#pragma unroll
      for (uint32_t column = 0U; column < columns_per_thread; ++column) {
        const float weight_scale =
            weight_scale_tiles[current_buffer]
                              [local_column + column * thread_columns];
        accumulators[row][column] +=
            block_sums[row][column] * activation_scale * weight_scale;
      }
    }
    __syncthreads();
    current_buffer = next_buffer;
  }

#pragma unroll
  for (uint32_t row = 0U; row < rows_per_thread; ++row) {
#pragma unroll
    for (uint32_t column = 0U; column < columns_per_thread; ++column) {
      const uint64_t output_row = row_base + local_row + row * thread_rows;
      const uint64_t output_column =
          column_base + local_column + column * thread_columns;
      if (output_row < m && output_column < n) {
        output[output_row * n + output_column] =
            float_to_bf16_rne_bits(accumulators[row][column]);
      }
    }
  }
}

#define SLLM_DEFINE_GFX1030_HALF2_KERNEL(symbol, ingress, rows, columns,       \
                                         blocks)                               \
  extern "C" __global__ __launch_bounds__(256, 1) void symbol(                 \
      const uint8_t *const activation, const uint8_t *const activation_scales, \
      const uint8_t *const weight, const uint8_t *const weight_scales,         \
      uint16_t *const output, const uint64_t m, const uint64_t k,              \
      const uint64_t n) {                                                      \
    sllm_matmul_gfx1030_half2_dot2_body<ingress, rows, columns, blocks>(       \
        activation, activation_scales, weight, weight_scales, output, m, k,    \
        n);                                                                    \
  }

SLLM_DEFINE_GFX1030_HALF2_KERNEL(sllm_mxfp6_w6a6_gfx1030_half2_32x32_v1,
                                 Mxfp6E3M2ScalarHalf2Ingress, 32U, 32U, 1U)
SLLM_DEFINE_GFX1030_HALF2_KERNEL(sllm_mxfp8_w8a8_gfx1030_half2_32x32_k32_v1,
                                 Mxfp8E4M3Half2Ingress, 32U, 32U, 1U)
SLLM_DEFINE_GFX1030_HALF2_KERNEL(sllm_mxfp8_w8a8_gfx1030_half2_64x64_k32_v1,
                                 Mxfp8E4M3Half2Ingress, 64U, 64U, 1U)
SLLM_DEFINE_GFX1030_HALF2_KERNEL(sllm_mxfp8_w8a8_gfx1030_half2_128x32_k32_v1,
                                 Mxfp8E4M3Half2Ingress, 128U, 32U, 1U)
SLLM_DEFINE_GFX1030_HALF2_KERNEL(sllm_mxfp8_w8a8_gfx1030_half2_128x64_k32_v1,
                                 Mxfp8E4M3Half2Ingress, 128U, 64U, 1U)
SLLM_DEFINE_GFX1030_HALF2_KERNEL(sllm_mxfp8_w8a8_gfx1030_half2_128x64_k64_v1,
                                 Mxfp8E4M3Half2Ingress, 128U, 64U, 2U)
SLLM_DEFINE_GFX1030_HALF2_KERNEL(sllm_mxfp8_w8a8_gfx1030_half2_128x64_k128_v1,
                                 Mxfp8E4M3Half2Ingress, 128U, 64U, 4U)

extern "C" __global__
__launch_bounds__(256, 1) void sllm_mxfp8_w8a8_gfx1030_half2_128x64_k32_double_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_gfx1030_half2_dot2_k32_double_body<Mxfp8E4M3Half2Ingress, 128U,
                                                 64U>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_mxfp6_w6a6_gfx1030_half2_128x64_k32d_scalar_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_gfx1030_half2_dot2_k32_double_body<Mxfp6E3M2ScalarHalf2Ingress,
                                                 128U, 64U>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_mxfp6_w6a6_gfx1030_half2_128x64_k32d_pack4_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_gfx1030_half2_dot2_k32_double_body<Mxfp6E3M2Packed4Half2Ingress,
                                                 128U, 64U>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

#undef SLLM_DEFINE_GFX1030_HALF2_KERNEL

#if defined(SLLM_MATMUL_HAS_GFX12_ROCWMMA)
static_assert(sizeof(rocwmma::float8_t) == sizeof(uint8_t));
#endif

// Eight independent waves cover a 128-row output tile. The raw OCP E4M3 value
// planes are staged as bytes and consumed directly by gfx12 FP8 WMMA. E8M0
// block-32 scales remain separate: each wave keeps its unscaled 16x16 WMMA
// contributions in registers, transforms them to row-major lane layout once
// per K block, and applies the row/column scale pair while accumulating. No
// whole-tensor BF16/FP32 expansion or contribution scratch tile is created;
// zero-padded LDS tiles make both M and N tails fail-safe.
struct Mxfp8WmmaTileIngress {
  __device__ __forceinline__ static uint64_t row_bytes(const uint64_t k) {
    return k;
  }

  __device__ __forceinline__ static uint8_t
  load_activation(const uint8_t *const row, const uint64_t index) {
    return row[index];
  }

  __device__ __forceinline__ static uint8_t
  load_weight(const uint8_t *const row, const uint64_t index) {
    return __builtin_nontemporal_load(row + index);
  }
};

struct Mxfp6ViaE4WmmaTileIngress {
  __device__ __forceinline__ static uint64_t row_bytes(const uint64_t k) {
    return k * UINT64_C(3) / UINT64_C(4);
  }

  __device__ __forceinline__ static uint8_t
  load_activation(const uint8_t *const row, const uint64_t index) {
    return sllm_lowp::e3m2_to_e4m3fn_exact_bits(packed_e3m2_at(row, index));
  }

  __device__ __forceinline__ static uint8_t
  load_weight(const uint8_t *const row, const uint64_t index) {
    return sllm_lowp::e3m2_to_e4m3fn_exact_bits(packed_e3m2_at(row, index));
  }
};

struct Mxfp6ViaE4WmmaPacked4Ingress {
  __device__ __forceinline__ static uint64_t row_bytes(const uint64_t k) {
    return k * UINT64_C(3) / UINT64_C(4);
  }

  __device__ __forceinline__ static uint32_t
  load_activation4(const uint8_t *const row, const uint64_t first_index) {
    return sllm_lowp::e3m2x4_to_e4m3fn_exact_bits(
        sllm_lowp::packed_e3m2x4_at(row, first_index));
  }

  __device__ __forceinline__ static uint32_t
  load_weight4(const uint8_t *const row, const uint64_t first_index) {
    return sllm_lowp::e3m2x4_to_e4m3fn_exact_bits(
        sllm_lowp::packed_e3m2x4_at(row, first_index));
  }
};

// Phase 74 gfx1201 candidate: preserve the ID45 WMMA/LDS/scale path while
// replacing only the packed E3M2x4 -> E4M3x4 ingress conversion with a 32-bit
// byte-lane SWAR transform.
struct Mxfp6ViaE4WmmaPacked4SwarIngress {
  __device__ __forceinline__ static uint64_t row_bytes(const uint64_t k) {
    return k * UINT64_C(3) / UINT64_C(4);
  }

  __device__ __forceinline__ static uint32_t
  load_activation4(const uint8_t *const row, const uint64_t first_index) {
    return sllm_lowp::e3m2x4_to_e4m3fn_exact_bits_swar(
        sllm_lowp::packed_e3m2x4_at(row, first_index));
  }

  __device__ __forceinline__ static uint32_t
  load_weight4(const uint8_t *const row, const uint64_t first_index) {
    return sllm_lowp::e3m2x4_to_e4m3fn_exact_bits_swar(
        sllm_lowp::packed_e3m2x4_at(row, first_index));
  }
};

template <typename ValueIngress, uint32_t ColumnTiles,
          bool PackedGroupIngress = false>
__device__ __forceinline__ void sllm_matmul_mx_via_e4_gfx1201_wmma_body(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
#if defined(__gfx1201__)
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t waves_per_workgroup = 8U;
  constexpr uint32_t tile_m = 16U;
  constexpr uint32_t tile_n = 16U;
  constexpr uint32_t column_tiles = ColumnTiles;
  constexpr uint32_t block_k = 32U;
  constexpr uint32_t tile_values = tile_m * block_k;
  constexpr uint32_t output_values = tile_m * tile_n;
  __shared__ __align__(4)
      rocwmma::float8_t activation_tile[waves_per_workgroup][tile_values];
  __shared__ __align__(4)
      rocwmma::float8_t weight_tile[column_tiles][tile_values];
  __shared__ float activation_scale_tile[waves_per_workgroup][tile_m];
  __shared__ float weight_scale_tile[column_tiles * tile_n];

  using AFragment = rocwmma::fragment<rocwmma::matrix_a, tile_m, tile_n, tile_m,
                                      rocwmma::float8_t, rocwmma::row_major>;
  using BFragment = rocwmma::fragment<rocwmma::matrix_b, tile_m, tile_n, tile_m,
                                      rocwmma::float8_t, rocwmma::col_major>;
  using AccumulatorFragment =
      rocwmma::fragment<rocwmma::accumulator, tile_m, tile_n, tile_m, float>;

  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (wave_width - 1U);
  const uint32_t wave = thread / wave_width;
  const uint64_t row_group_base =
      static_cast<uint64_t>(blockIdx.y) *
      sllm_matmul_kernel::kMxfp8W8A8PrefillWmmaRowsPerWorkgroup;
  const uint64_t row_tile_base =
      row_group_base + static_cast<uint64_t>(wave) * tile_m;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * column_tiles * tile_n;
  const uint64_t blocks_per_row = k / block_k;
  const uint64_t value_row_bytes = ValueIngress::row_bytes(k);
  float accumulators[column_tiles][output_values / wave_width] = {};

  for (uint64_t block = 0U; block < blocks_per_row; ++block) {
    const uint64_t inner_base = block * block_k;
    auto *const activation_raw = reinterpret_cast<uint8_t *>(activation_tile);
    auto *const weight_raw = reinterpret_cast<uint8_t *>(weight_tile);

    if constexpr (PackedGroupIngress) {
      constexpr uint32_t values_per_group = 4U;
      constexpr uint32_t groups_per_tile = tile_values / values_per_group;
      auto *const activation_groups =
          reinterpret_cast<uint32_t *>(activation_raw);
      auto *const weight_groups = reinterpret_cast<uint32_t *>(weight_raw);
      for (uint32_t group = thread;
           group < waves_per_workgroup * groups_per_tile; group += blockDim.x) {
        const uint32_t source_wave = group / groups_per_tile;
        const uint32_t wave_group = group - source_wave * groups_per_tile;
        const uint32_t local_row = wave_group / (block_k / values_per_group);
        const uint32_t local_group =
            wave_group - local_row * (block_k / values_per_group);
        const uint64_t row = row_group_base +
                             static_cast<uint64_t>(source_wave) * tile_m +
                             local_row;
        activation_groups[group] =
            row < m ? ValueIngress::load_activation4(
                          activation + row * value_row_bytes,
                          inner_base + local_group * values_per_group)
                    : 0U;
      }
      for (uint32_t group = thread; group < column_tiles * groups_per_tile;
           group += blockDim.x) {
        const uint32_t column_tile = group / groups_per_tile;
        const uint32_t tile_group = group - column_tile * groups_per_tile;
        const uint32_t local_column = tile_group / (block_k / values_per_group);
        const uint32_t local_group =
            tile_group - local_column * (block_k / values_per_group);
        const uint64_t column =
            column_base + column_tile * tile_n + local_column;
        weight_groups[group] =
            column < n ? ValueIngress::load_weight4(
                             weight + column * value_row_bytes,
                             inner_base + local_group * values_per_group)
                       : 0U;
      }
    } else {
      for (uint32_t index = thread; index < waves_per_workgroup * tile_values;
           index += blockDim.x) {
        const uint32_t source_wave = index / tile_values;
        const uint32_t wave_index = index - source_wave * tile_values;
        const uint32_t local_row = wave_index / block_k;
        const uint32_t local_inner = wave_index - local_row * block_k;
        const uint64_t row = row_group_base +
                             static_cast<uint64_t>(source_wave) * tile_m +
                             local_row;
        activation_raw[index] = row < m
                                    ? ValueIngress::load_activation(
                                          activation + row * value_row_bytes,
                                          inner_base + local_inner)
                                    : 0U;
      }
      for (uint32_t index = thread; index < column_tiles * tile_values;
           index += blockDim.x) {
        const uint32_t column_tile = index / tile_values;
        const uint32_t tile_index = index - column_tile * tile_values;
        const uint32_t local_column = tile_index / block_k;
        const uint32_t local_inner = tile_index - local_column * block_k;
        const uint64_t column =
            column_base + column_tile * tile_n + local_column;
        weight_raw[index] =
            column < n
                ? ValueIngress::load_weight(weight + column * value_row_bytes,
                                            inner_base + local_inner)
                : 0U;
      }
    }
    if (thread < waves_per_workgroup * tile_m) {
      const uint32_t source_wave = thread / tile_m;
      const uint32_t local_row = thread - source_wave * tile_m;
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      activation_scale_tile[source_wave][local_row] =
          row < m
              ? e8m0_to_float(activation_scales[row * blocks_per_row + block])
              : 0.0F;
    }
    if (thread < column_tiles * tile_n) {
      const uint64_t column = column_base + thread;
      weight_scale_tile[thread] =
          column < n
              ? e8m0_to_float(weight_scales[column * blocks_per_row + block])
              : 0.0F;
    }
    __syncthreads();

    AFragment activation_fragment;
    AccumulatorFragment contributions[column_tiles];
#pragma unroll
    for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
      rocwmma::fill_fragment(contributions[column_tile], 0.0F);
    }
    rocwmma::load_matrix_sync(activation_fragment, activation_tile[wave],
                              block_k);
#pragma unroll
    for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
      BFragment weight_fragment;
      rocwmma::load_matrix_sync(weight_fragment, weight_tile[column_tile],
                                block_k);
      rocwmma::mma_sync(contributions[column_tile], activation_fragment,
                        weight_fragment, contributions[column_tile]);
    }
    rocwmma::load_matrix_sync(activation_fragment,
                              activation_tile[wave] + tile_m, block_k);
#pragma unroll
    for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
      BFragment weight_fragment;
      rocwmma::load_matrix_sync(weight_fragment,
                                weight_tile[column_tile] + tile_m, block_k);
      rocwmma::mma_sync(contributions[column_tile], activation_fragment,
                        weight_fragment, contributions[column_tile]);
    }
#pragma unroll
    for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
      const auto contribution_row_major =
          rocwmma::apply_data_layout<rocwmma::row_major>(
              contributions[column_tile]);
#pragma unroll
      for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
        const uint32_t local_row =
            (lane / tile_n) * (output_values / wave_width) + slot;
        const uint32_t local_column = lane % tile_n;
        float term = contribution_row_major[slot] *
                     activation_scale_tile[wave][local_row];
        term *= weight_scale_tile[column_tile * tile_n + local_column];
        accumulators[column_tile][slot] += term;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
#pragma unroll
    for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
      const uint32_t local_row =
          (lane / tile_n) * (output_values / wave_width) + slot;
      const uint32_t local_column = lane % tile_n;
      const uint64_t row = row_tile_base + local_row;
      const uint64_t column = column_base + column_tile * tile_n + local_column;
      if (row < m && column < n) {
        output[row * n + column] =
            float_to_bf16_rne_bits(accumulators[column_tile][slot]);
      }
    }
  }
#else
  (void)activation;
  (void)activation_scales;
  (void)weight;
  (void)weight_scales;
  (void)output;
  (void)m;
  (void)k;
  (void)n;
#endif
}

// Phase 64 candidates keep the Phase 63 arithmetic order and output mapping
// fixed while varying only workgroup height, the physical LDS row stride, or
// the fragment load source. LdsStride=33 is a public-rocWMMA-compatible
// bank-conflict probe: rocWMMA does not expose a custom XOR-addressed LDS
// accessor, so padding is used to perturb the same bank mapping without
// depending on its private fragment layout. DirectActivation and DirectWeight
// bypass their respective value tiles; the small E8M0 scale tiles remain
// shared. DirectActivation is dispatched only for complete 128-row groups.
template <uint32_t WavesPerWorkgroup, uint32_t ColumnTiles, uint32_t LdsStride,
          bool DirectActivation, bool DirectWeight>
__device__ __forceinline__ void
sllm_matmul_mxfp8_w8a8_gfx1201_wmma_phase64_body(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
#if defined(__gfx1201__)
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t waves_per_workgroup = WavesPerWorkgroup;
  constexpr uint32_t tile_m = 16U;
  constexpr uint32_t tile_n = 16U;
  constexpr uint32_t column_tiles = ColumnTiles;
  constexpr uint32_t block_k = 32U;
  constexpr uint32_t lds_stride = LdsStride;
  constexpr uint32_t rows_per_workgroup = waves_per_workgroup * tile_m;
  constexpr uint32_t activation_lds_values =
      DirectActivation ? 1U : waves_per_workgroup * tile_m * lds_stride;
  constexpr uint32_t weight_lds_values =
      DirectWeight ? 1U : column_tiles * tile_m * lds_stride;
  constexpr uint32_t output_values = tile_m * tile_n;
  static_assert(waves_per_workgroup == 4U || waves_per_workgroup == 8U);
  static_assert(lds_stride >= block_k);

  __shared__ rocwmma::float8_t activation_tile[activation_lds_values];
  __shared__ rocwmma::float8_t weight_tile[weight_lds_values];
  __shared__ float activation_scale_tile[waves_per_workgroup][tile_m];
  __shared__ float weight_scale_tile[column_tiles * tile_n];

  using AFragment = rocwmma::fragment<rocwmma::matrix_a, tile_m, tile_n, tile_m,
                                      rocwmma::float8_t, rocwmma::row_major>;
  using BFragment = rocwmma::fragment<rocwmma::matrix_b, tile_m, tile_n, tile_m,
                                      rocwmma::float8_t, rocwmma::col_major>;
  using AccumulatorFragment =
      rocwmma::fragment<rocwmma::accumulator, tile_m, tile_n, tile_m, float>;

  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (wave_width - 1U);
  const uint32_t wave = thread / wave_width;
  const uint64_t row_group_base =
      static_cast<uint64_t>(blockIdx.y) * rows_per_workgroup;
  const uint64_t row_tile_base =
      row_group_base + static_cast<uint64_t>(wave) * tile_m;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * column_tiles * tile_n;
  const uint64_t blocks_per_row = k / block_k;
  float accumulators[column_tiles][output_values / wave_width] = {};

  for (uint64_t block = 0U; block < blocks_per_row; ++block) {
    const uint64_t inner_base = block * block_k;
    if constexpr (!DirectActivation) {
      auto *const activation_raw = reinterpret_cast<uint8_t *>(activation_tile);
      constexpr uint32_t activation_logical_values =
          waves_per_workgroup * tile_m * block_k;
      for (uint32_t index = thread; index < activation_logical_values;
           index += blockDim.x) {
        const uint32_t source_wave = index / (tile_m * block_k);
        const uint32_t wave_index = index - source_wave * tile_m * block_k;
        const uint32_t local_row = wave_index / block_k;
        const uint32_t local_inner = wave_index - local_row * block_k;
        const uint64_t row = row_group_base +
                             static_cast<uint64_t>(source_wave) * tile_m +
                             local_row;
        activation_raw[(source_wave * tile_m + local_row) * lds_stride +
                       local_inner] =
            row < m ? activation[row * k + inner_base + local_inner] : 0U;
      }
    }
    if constexpr (!DirectWeight) {
      auto *const weight_raw = reinterpret_cast<uint8_t *>(weight_tile);
      constexpr uint32_t weight_logical_values =
          column_tiles * tile_m * block_k;
      for (uint32_t index = thread; index < weight_logical_values;
           index += blockDim.x) {
        const uint32_t column_tile = index / (tile_m * block_k);
        const uint32_t tile_index = index - column_tile * tile_m * block_k;
        const uint32_t local_column = tile_index / block_k;
        const uint32_t local_inner = tile_index - local_column * block_k;
        const uint64_t column =
            column_base + column_tile * tile_n + local_column;
        weight_raw[(column_tile * tile_m + local_column) * lds_stride +
                   local_inner] =
            column < n ? __builtin_nontemporal_load(weight + column * k +
                                                    inner_base + local_inner)
                       : 0U;
      }
    }
    if (thread < waves_per_workgroup * tile_m) {
      const uint32_t source_wave = thread / tile_m;
      const uint32_t local_row = thread - source_wave * tile_m;
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      activation_scale_tile[source_wave][local_row] =
          row < m
              ? e8m0_to_float(activation_scales[row * blocks_per_row + block])
              : 0.0F;
    }
    if (thread < column_tiles * tile_n) {
      const uint64_t column = column_base + thread;
      weight_scale_tile[thread] =
          column < n
              ? e8m0_to_float(weight_scales[column * blocks_per_row + block])
              : 0.0F;
    }
    __syncthreads();

    AFragment activation_fragment;
    AccumulatorFragment contributions[column_tiles];
#pragma unroll
    for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
      rocwmma::fill_fragment(contributions[column_tile], 0.0F);
    }
    if constexpr (DirectActivation) {
      const auto *const activation_matrix =
          reinterpret_cast<const rocwmma::float8_t *>(
              activation + row_tile_base * k + inner_base);
      rocwmma::load_matrix_sync(activation_fragment, activation_matrix,
                                static_cast<uint32_t>(k));
    } else {
      const auto *const activation_wave_tile =
          activation_tile + wave * tile_m * lds_stride;
      rocwmma::load_matrix_sync(activation_fragment, activation_wave_tile,
                                lds_stride);
    }
#pragma unroll
    for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
      BFragment weight_fragment;
      if constexpr (DirectWeight) {
        const uint64_t column = column_base + column_tile * tile_n;
        const auto *const weight_matrix =
            reinterpret_cast<const rocwmma::float8_t *>(weight + column * k +
                                                        inner_base);
        rocwmma::load_matrix_sync(weight_fragment, weight_matrix,
                                  static_cast<uint32_t>(k));
      } else {
        rocwmma::load_matrix_sync(
            weight_fragment, weight_tile + column_tile * tile_m * lds_stride,
            lds_stride);
      }
      rocwmma::mma_sync(contributions[column_tile], activation_fragment,
                        weight_fragment, contributions[column_tile]);
    }
    if constexpr (DirectActivation) {
      const auto *const activation_matrix =
          reinterpret_cast<const rocwmma::float8_t *>(
              activation + row_tile_base * k + inner_base + tile_m);
      rocwmma::load_matrix_sync(activation_fragment, activation_matrix,
                                static_cast<uint32_t>(k));
    } else {
      const auto *const activation_wave_tile =
          activation_tile + wave * tile_m * lds_stride;
      rocwmma::load_matrix_sync(activation_fragment,
                                activation_wave_tile + tile_m, lds_stride);
    }
#pragma unroll
    for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
      BFragment weight_fragment;
      if constexpr (DirectWeight) {
        const uint64_t column = column_base + column_tile * tile_n;
        const auto *const weight_matrix =
            reinterpret_cast<const rocwmma::float8_t *>(weight + column * k +
                                                        inner_base + tile_m);
        rocwmma::load_matrix_sync(weight_fragment, weight_matrix,
                                  static_cast<uint32_t>(k));
      } else {
        rocwmma::load_matrix_sync(
            weight_fragment,
            weight_tile + column_tile * tile_m * lds_stride + tile_m,
            lds_stride);
      }
      rocwmma::mma_sync(contributions[column_tile], activation_fragment,
                        weight_fragment, contributions[column_tile]);
    }
#pragma unroll
    for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
      const auto contribution_row_major =
          rocwmma::apply_data_layout<rocwmma::row_major>(
              contributions[column_tile]);
#pragma unroll
      for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
        const uint32_t local_row =
            (lane / tile_n) * (output_values / wave_width) + slot;
        const uint32_t local_column = lane % tile_n;
        float term = contribution_row_major[slot] *
                     activation_scale_tile[wave][local_row];
        term *= weight_scale_tile[column_tile * tile_n + local_column];
        accumulators[column_tile][slot] += term;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
#pragma unroll
    for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
      const uint32_t local_row =
          (lane / tile_n) * (output_values / wave_width) + slot;
      const uint32_t local_column = lane % tile_n;
      const uint64_t row = row_tile_base + local_row;
      const uint64_t column = column_base + column_tile * tile_n + local_column;
      if (row < m && column < n) {
        output[row * n + column] =
            float_to_bf16_rne_bits(accumulators[column_tile][slot]);
      }
    }
  }
#else
  (void)activation;
  (void)activation_scales;
  (void)weight;
  (void)weight_scales;
  (void)output;
  (void)m;
  (void)k;
  (void)n;
#endif
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_wmma128x16x32_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mx_via_e4_gfx1201_wmma_body<Mxfp8WmmaTileIngress, 1U>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_wmma128x64x32_v2(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mx_via_e4_gfx1201_wmma_body<Mxfp8WmmaTileIngress, 4U>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_mxfp6_w6a6_gfx1201_wmma128x64_via_e4m3_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mx_via_e4_gfx1201_wmma_body<Mxfp6ViaE4WmmaTileIngress, 4U>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_mxfp6_w6a6_gfx1201_wmma128x64_pack4_v2(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mx_via_e4_gfx1201_wmma_body<Mxfp6ViaE4WmmaPacked4Ingress, 4U,
                                          true>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_mxfp6_w6a6_gfx1201_wmma128x64_pack4_swar_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mx_via_e4_gfx1201_wmma_body<Mxfp6ViaE4WmmaPacked4SwarIngress, 4U,
                                          true>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_mxfp6_w6a6_gfx1201_wmma128x128_pack4_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mx_via_e4_gfx1201_wmma_body<Mxfp6ViaE4WmmaPacked4Ingress, 8U,
                                          true>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(128, 1) void sllm_mxfp8_w8a8_gfx1201_wmma64x64_4w_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mxfp8_w8a8_gfx1201_wmma_phase64_body<4U, 4U, 32U, false, false>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_mxfp8_w8a8_gfx1201_wmma128x64_pad33_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mxfp8_w8a8_gfx1201_wmma_phase64_body<8U, 4U, 33U, false, false>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_mxfp8_w8a8_gfx1201_wmma128x64_direct_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mxfp8_w8a8_gfx1201_wmma_phase64_body<8U, 4U, 32U, false, true>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_mxfp8_w8a8_gfx1201_wmma128x64_adirect_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mxfp8_w8a8_gfx1201_wmma_phase64_body<8U, 4U, 32U, true, false>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_mxfp8_w8a8_gfx1201_wmma128x64_bdirect_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mxfp8_w8a8_gfx1201_wmma_phase64_body<8U, 4U, 32U, true, true>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_mxfp8_w8a8_gfx1201_wmma128x128_bdirect_v1(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  sllm_matmul_mxfp8_w8a8_gfx1201_wmma_phase64_body<8U, 8U, 32U, true, true>(
      activation, activation_scales, weight, weight_scales, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_tiled16_v3(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t tile = 16U;
  __shared__ float activation_tile[tile][tile];
  __shared__ float weight_tile[tile][tile];
  const uint32_t local_row = threadIdx.y;
  const uint32_t local_column = threadIdx.x;
  const uint64_t row = static_cast<uint64_t>(blockIdx.y) * tile + local_row;
  const uint64_t column =
      static_cast<uint64_t>(blockIdx.x) * tile + local_column;
  const uint64_t blocks_per_row = k / UINT64_C(32);
  float accumulator = 0.0F;
  for (uint64_t base = 0U; base < k; base += tile) {
    const uint64_t activation_inner = base + local_column;
    const uint64_t weight_inner = base + local_row;
    if (row < m && activation_inner < k) {
      activation_tile[local_row][local_column] =
          e4m3fn_to_float(activation[row * k + activation_inner]) *
          e8m0_to_float(activation_scales[row * blocks_per_row +
                                          activation_inner / UINT64_C(32)]);
    } else {
      activation_tile[local_row][local_column] = 0.0F;
    }
    if (column < n && weight_inner < k) {
      weight_tile[local_row][local_column] =
          e4m3fn_to_float(
              __builtin_nontemporal_load(weight + column * k + weight_inner)) *
          e8m0_to_float(weight_scales[column * blocks_per_row +
                                      weight_inner / UINT64_C(32)]);
    } else {
      weight_tile[local_row][local_column] = 0.0F;
    }
    __syncthreads();
#pragma unroll
    for (uint32_t inner = 0U; inner != tile; ++inner) {
      accumulator +=
          activation_tile[local_row][inner] * weight_tile[inner][local_column];
    }
    __syncthreads();
  }
  if (row < m && column < n) {
    output[row * n + column] = float_to_bf16_rne_bits(accumulator);
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_tiled16_v3(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t tile = 16U;
  __shared__ float activation_tile[tile][tile];
  __shared__ float weight_tile[tile][tile];
  const uint32_t local_row = threadIdx.y;
  const uint32_t local_column = threadIdx.x;
  const uint64_t row = static_cast<uint64_t>(blockIdx.y) * tile + local_row;
  const uint64_t column =
      static_cast<uint64_t>(blockIdx.x) * tile + local_column;
  const uint64_t blocks_per_row = k / UINT64_C(32);
  const uint64_t row_bytes = k * UINT64_C(3) / UINT64_C(4);
  float accumulator = 0.0F;
  for (uint64_t base = 0U; base < k; base += tile) {
    const uint64_t activation_inner = base + local_column;
    const uint64_t weight_inner = base + local_row;
    if (row < m && activation_inner < k) {
      activation_tile[local_row][local_column] =
          e3m2_to_float(
              packed_e3m2_at(activation + row * row_bytes, activation_inner)) *
          e8m0_to_float(activation_scales[row * blocks_per_row +
                                          activation_inner / UINT64_C(32)]);
    } else {
      activation_tile[local_row][local_column] = 0.0F;
    }
    if (column < n && weight_inner < k) {
      weight_tile[local_row][local_column] =
          e3m2_to_float(
              packed_e3m2_at(weight + column * row_bytes, weight_inner)) *
          e8m0_to_float(weight_scales[column * blocks_per_row +
                                      weight_inner / UINT64_C(32)]);
    } else {
      weight_tile[local_row][local_column] = 0.0F;
    }
    __syncthreads();
#pragma unroll
    for (uint32_t inner = 0U; inner != tile; ++inner) {
      accumulator +=
          activation_tile[local_row][inner] * weight_tile[inner][local_column];
    }
    __syncthreads();
  }
  if (row < m && column < n) {
    output[row * n + column] = float_to_bf16_rne_bits(accumulator);
  }
}

#pragma clang fp contract(off)
extern "C" __global__ __launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  const uint64_t output_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                                static_cast<uint64_t>(threadIdx.x);
  const uint64_t output_elements = m * n;
  if (output_index < output_elements) {
    const uint64_t row = output_index / n;
    const uint64_t column = output_index - row * n;
    float accumulator = 0.0F;
    for (uint64_t reduction = 0U; reduction != k; ++reduction) {
      const float activation_value =
          bf16_to_float(activation[row * k + reduction]);
      const float weight_value = bf16_to_float(weight[column * k + reduction]);
      accumulator += activation_value * weight_value;
    }
    output[output_index] = float_to_bf16_rne_bits(accumulator);
  }
}

// Row-major [M,K] x transposed row-major [N,K].  A 16x16 output tile shares
// both input tiles, eliminating the baseline kernel's redundant global loads.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_tiled16_v2(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t tile = 16U;
  __shared__ uint16_t activation_tile[tile][tile];
  __shared__ uint16_t weight_tile[tile][tile];
  const uint32_t local_row = threadIdx.y;
  const uint32_t local_column = threadIdx.x;
  const uint64_t row = static_cast<uint64_t>(blockIdx.y) * tile + local_row;
  const uint64_t column =
      static_cast<uint64_t>(blockIdx.x) * tile + local_column;
  float accumulator = 0.0F;
  for (uint64_t base = 0U; base < k; base += tile) {
    const uint64_t activation_k = base + local_column;
    const uint64_t weight_k = base + local_row;
    activation_tile[local_row][local_column] =
        row < m && activation_k < k ? activation[row * k + activation_k] : 0U;
    weight_tile[local_row][local_column] =
        column < n && weight_k < k ? weight[column * k + weight_k] : 0U;
    __syncthreads();
#pragma unroll
    for (uint32_t inner = 0U; inner != tile; ++inner) {
      accumulator += bf16_to_float(activation_tile[local_row][inner]) *
                     bf16_to_float(weight_tile[inner][local_column]);
    }
    __syncthreads();
  }
  if (row < m && column < n) {
    output[row * n + column] = float_to_bf16_rne_bits(accumulator);
  }
}

// Decode is a matrix-vector product. One workgroup owns one output column and
// reduces K cooperatively; this avoids launching mostly idle 16x16 tiles.
//
// The paired BF16 loads and two-level wave reduction are adapted from the
// floating MMVF organization in llama.cpp mmvf.cu at fixed commit
// f5919bf458ef190468b5c329bb293f8a54a1e69c. The ggml tensor/runtime and
// fusion machinery are deliberately not imported; this kernel retains sLLM's
// BF16 input/output and FP32 accumulation contract.
template <uint32_t WaveWidth, uint32_t WaveCount>
__device__ __forceinline__ void
matmul_bf16_decode_body(const uint16_t *const activation,
                        const uint16_t *const weight, uint16_t *const output,
                        const uint64_t k, const uint64_t n,
                        const uint64_t column) {
  if (column >= n) {
    return;
  }
  float partial = 0.0F;
  const uint16_t *const weight_row = weight + column * k;
  const bool paired =
      (k & UINT64_C(1)) == 0U && ((reinterpret_cast<uintptr_t>(activation) |
                                   reinterpret_cast<uintptr_t>(weight_row)) &
                                  static_cast<uintptr_t>(3U)) == 0U;
  if (paired) {
    const auto *const activation_pairs =
        reinterpret_cast<const uint32_t *>(activation);
    const auto *const weight_pairs =
        reinterpret_cast<const uint32_t *>(weight_row);
    const uint64_t pair_count = k / 2U;
    for (uint64_t pair = threadIdx.x; pair < pair_count; pair += blockDim.x) {
      const uint32_t activation_pair = activation_pairs[pair];
      const uint32_t weight_pair =
          __builtin_nontemporal_load(weight_pairs + pair);
      partial += bf16_to_float(static_cast<uint16_t>(activation_pair)) *
                 bf16_to_float(static_cast<uint16_t>(weight_pair));
      partial += bf16_to_float(static_cast<uint16_t>(activation_pair >> 16U)) *
                 bf16_to_float(static_cast<uint16_t>(weight_pair >> 16U));
    }
  } else {
    for (uint64_t reduction = threadIdx.x; reduction < k;
         reduction += blockDim.x) {
      partial += bf16_to_float(activation[reduction]) *
                 bf16_to_float(weight_row[reduction]);
    }
  }

#pragma unroll
  for (uint32_t offset = WaveWidth / 2U; offset != 0U; offset >>= 1U) {
    partial += __shfl_down(partial, offset, WaveWidth);
  }
  __shared__ float wave_sums[WaveCount];
  const uint32_t lane = threadIdx.x % WaveWidth;
  const uint32_t wave = threadIdx.x / WaveWidth;
  if (lane == 0U) {
    wave_sums[wave] = partial;
  }
  __syncthreads();
  if (wave == 0U) {
    partial = lane < WaveCount ? wave_sums[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = WaveWidth / 2U; offset != 0U; offset >>= 1U) {
      partial += __shfl_down(partial, offset, WaveWidth);
    }
    if (lane == 0U) {
      output[column] = float_to_bf16_rne_bits(partial);
    }
  }
}

// Short GDN projection provider for the exact Qwen shape M=17, K=5120,
// N=48.  Unlike the tiled16 prefill body, each block owns one output element
// and reuses the established decode reduction body for its row.  The paired
// BF16 loads, FP32 products, and two-level wave-sum tree therefore remain the
// same as matmul_bf16_decode_body<32U, 8U>; only the M dimension is exposed to
// the grid.  This is selected by the launcher only for the exact supported
// targets and shape below; all other prefill shapes retain tiled16.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_prefill_gdn_thin_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  const uint64_t row = static_cast<uint64_t>(blockIdx.y);
  const uint64_t column = static_cast<uint64_t>(blockIdx.x);
  if (row < m && column < n) {
    matmul_bf16_decode_body<32U, 8U>(activation + row * k, weight,
                                     output + row * n, k, n, column);
  }
}

template <uint32_t WaveWidth, uint32_t WaveCount>
__device__ __forceinline__ void matmul_bf16_serial_rows_body(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const uint64_t row_start = 0U,
    const uint64_t column_override = UINT64_MAX) {
  constexpr uint32_t max_rows = 8U;
  const uint64_t column = column_override == UINT64_MAX
                              ? static_cast<uint64_t>(blockIdx.x)
                              : column_override;
  if (column >= n || m == 0U || m > max_rows) {
    return;
  }
  float partial[max_rows] = {};
  const uint16_t *const weight_row = weight + column * k;
  const uint16_t *const activation_start = activation + row_start * k;
  const bool paired = (k & UINT64_C(1)) == 0U &&
                      ((reinterpret_cast<uintptr_t>(activation_start) |
                        reinterpret_cast<uintptr_t>(weight_row)) &
                       static_cast<uintptr_t>(3U)) == 0U;
  if (paired) {
    const auto *const weight_pairs =
        reinterpret_cast<const uint32_t *>(weight_row);
    const uint64_t pair_count = k / 2U;
    for (uint64_t pair = threadIdx.x; pair < pair_count; pair += blockDim.x) {
      const uint32_t weight_pair =
          __builtin_nontemporal_load(weight_pairs + pair);
      const float weight0 = bf16_to_float(static_cast<uint16_t>(weight_pair));
      const float weight1 =
          bf16_to_float(static_cast<uint16_t>(weight_pair >> 16U));
      for (uint32_t row = 0U; row < m; ++row) {
        const auto *const activation_pairs = reinterpret_cast<const uint32_t *>(
            activation_start + static_cast<uint64_t>(row) * k);
        const uint32_t activation_pair = activation_pairs[pair];
        partial[row] +=
            bf16_to_float(static_cast<uint16_t>(activation_pair)) * weight0;
        partial[row] +=
            bf16_to_float(static_cast<uint16_t>(activation_pair >> 16U)) *
            weight1;
      }
    }
  } else {
    for (uint64_t reduction = threadIdx.x; reduction < k;
         reduction += blockDim.x) {
      const float weight_value = bf16_to_float(weight_row[reduction]);
      for (uint32_t row = 0U; row < m; ++row) {
        partial[row] +=
            bf16_to_float(
                activation_start[static_cast<uint64_t>(row) * k + reduction]) *
            weight_value;
      }
    }
  }

  for (uint32_t offset = WaveWidth / 2U; offset != 0U; offset >>= 1U) {
    for (uint32_t row = 0U; row < m; ++row) {
      partial[row] += __shfl_down(partial[row], offset, WaveWidth);
    }
  }
  __shared__ float wave_sums[max_rows][WaveCount];
  const uint32_t lane = threadIdx.x % WaveWidth;
  const uint32_t wave = threadIdx.x / WaveWidth;
  if (lane == 0U) {
    for (uint32_t row = 0U; row < m; ++row) {
      wave_sums[row][wave] = partial[row];
    }
  }
  __syncthreads();
  if (wave == 0U) {
    for (uint32_t row = 0U; row < m; ++row) {
      partial[row] = lane < WaveCount ? wave_sums[row][lane] : 0.0F;
    }
    for (uint32_t offset = WaveWidth / 2U; offset != 0U; offset >>= 1U) {
      for (uint32_t row = 0U; row < m; ++row) {
        partial[row] += __shfl_down(partial[row], offset, WaveWidth);
      }
    }
    if (lane == 0U) {
      for (uint32_t row = 0U; row < m; ++row) {
        output[(row_start + static_cast<uint64_t>(row)) * n + column] =
            float_to_bf16_rne_bits(partial[row]);
      }
    }
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_decode_v4(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t k, const uint64_t n) {
  matmul_bf16_decode_body<32U, 8U>(activation, weight, output, k, n,
                                   blockIdx.x);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_decode_wave64_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t k, const uint64_t n) {
  matmul_bf16_decode_body<64U, 4U>(activation, weight, output, k, n,
                                   blockIdx.x);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_decode_serial_rows_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  matmul_bf16_serial_rows_body<32U, 8U>(activation, weight, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_decode_serial_rows_wave64_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  matmul_bf16_serial_rows_body<64U, 4U>(activation, weight, output, m, k, n);
}

// Short prefill provider for the exact gfx1030 Qwen projection shapes.  Each
// block owns one output column and one consecutive group of up to eight rows;
// the existing serial-reduction body is reused unchanged for each group.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_prefill_short_serial_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  if (n == 0U || m == 0U) {
    return;
  }
  const uint64_t column = blockIdx.x % n;
  const uint64_t row_group = blockIdx.x / n;
  const uint64_t row_start = row_group * UINT64_C(8);
  if (row_start >= m) {
    return;
  }
  const uint64_t remaining_rows = m - row_start;
  const uint64_t rows =
      remaining_rows < UINT64_C(8) ? remaining_rows : UINT64_C(8);
  // gfx1030 uses wave32, matching the established M=2..8 provider.
  matmul_bf16_serial_rows_body<32U, 8U>(activation, weight, output, rows, k, n,
                                        row_start, column);
}

extern "C" __global__ void
sllm_matmul_fp32_to_bf16_short_mixed_v1(const float *const input,
                                        uint16_t *const output,
                                        const uint64_t element_count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    output[index] = float_to_bf16_rne_bits(input[index]);
  }
}

// The Phase 15 provider remains the decode path and is also the within-binary
// prefill performance control when SLLM_NVFP4_FORCE_BASELINE=1 is explicit.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_block16_packed_dequant_v1(
    const uint16_t *const activation, const uint8_t *const packed_weight,
    const uint8_t *const block_scales, const float *const tensor_scale,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  const uint64_t output_index = blockIdx.x;
  if (output_index >= m * n) {
    return;
  }
  const uint64_t row = output_index / n;
  const uint64_t column = output_index - row * n;
  const uint64_t blocks_per_weight_row = (k + UINT64_C(15)) / UINT64_C(16);
  float partial = 0.0F;
  for (uint64_t inner = threadIdx.x; inner < k; inner += blockDim.x) {
    const uint64_t weight_index = column * k + inner;
    const uint8_t packed =
        __builtin_nontemporal_load(packed_weight + weight_index / UINT64_C(2));
    const uint8_t code = (weight_index & UINT64_C(1)) == 0U
                             ? packed & UINT8_C(0x0f)
                             : packed >> 4U;
    const float scale = e4m3fn_to_float(
        block_scales[column * blocks_per_weight_row + inner / UINT64_C(16)]);
    partial += bf16_to_float(activation[row * k + inner]) *
               e2m1_to_float(code) * scale * tensor_scale[0];
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    partial += __shfl_down(partial, offset, 32U);
  }
  __shared__ float wave_sums[8];
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  if (lane == 0U) {
    wave_sums[wave] = partial;
  }
  __syncthreads();
  if (wave == 0U) {
    partial = lane < 8U ? wave_sums[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
      partial += __shfl_down(partial, offset, 32U);
    }
    if (lane == 0U) {
      output[output_index] = float_to_bf16_rne_bits(partial);
    }
  }
}

// Prefill maps one wave to one M row. Eight rows share the packed weight
// decode for each output column and keep the expansion bounded to one K tile.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_block16_prefill_row8_tiled256_v2(
    const uint16_t *const activation, const uint8_t *const packed_weight,
    const uint8_t *const block_scales, const float *const tensor_scale,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t rows_per_workgroup = 8U;
  constexpr uint32_t tile_k = 256U;
  __shared__ float weight_tile[tile_k];
  __shared__ float scale_tile[tile_k / 16U];
  __shared__ float shared_tensor_scale;
  const uint64_t column = static_cast<uint64_t>(blockIdx.x) % n;
  const uint64_t row_base =
      (static_cast<uint64_t>(blockIdx.x) / n) * rows_per_workgroup;
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  const uint64_t row = row_base + wave;
  const uint64_t blocks_per_weight_row = (k + UINT64_C(15)) / UINT64_C(16);
  float accumulator = 0.0F;
  if (threadIdx.x == 0U) {
    shared_tensor_scale = tensor_scale[0];
  }
  for (uint64_t base = 0U; base < k; base += tile_k) {
    if (threadIdx.x < tile_k / 16U) {
      const uint64_t scale_inner = base + threadIdx.x * UINT64_C(16);
      scale_tile[threadIdx.x] =
          scale_inner < k
              ? e4m3fn_to_float(block_scales[column * blocks_per_weight_row +
                                             scale_inner / UINT64_C(16)])
              : 0.0F;
    }
    __syncthreads();
    const uint64_t global_inner = base + threadIdx.x;
    if (global_inner < k) {
      const uint64_t weight_index = column * k + global_inner;
      const uint8_t packed = __builtin_nontemporal_load(
          packed_weight + weight_index / UINT64_C(2));
      const uint8_t code = (weight_index & UINT64_C(1)) == 0U
                               ? packed & UINT8_C(0x0f)
                               : packed >> 4U;
      weight_tile[threadIdx.x] =
          e2m1_to_float(code) * scale_tile[threadIdx.x / 16U];
    } else {
      weight_tile[threadIdx.x] = 0.0F;
    }
    __syncthreads();
    if (row < m) {
      const uint32_t valid = static_cast<uint32_t>(
          k - base < tile_k ? k - base : static_cast<uint64_t>(tile_k));
      for (uint32_t offset = lane; offset < valid; offset += wave_width) {
        accumulator += bf16_to_float(activation[row * k + base + offset]) *
                       weight_tile[offset];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    accumulator += __shfl_down(accumulator, offset, 32U);
  }
  if (lane == 0U && row < m) {
    output[row * n + column] =
        float_to_bf16_rne_bits(accumulator * shared_tensor_scale);
  }
}

#include "fp8_prefill_f16_tile_staging.inc"
#include "fp8_prefill_lds_lut.inc"
#include "fp8_prefill_short_m32.inc"
#include "nvfp4_decode_scale_lut.inc"

// Keep the production registry symbol within the public 64-byte ABI field.
// The target-specific kernels in nvfp4_decode_scale_lut.inc remain available
// for focused probes, while this wrapper selects the exact compiled-target
// body used by the single ID84 candidate.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_decode_scale_lut_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
#if defined(__gfx1201__)
  __shared__ float scale_lut[sllm_id84_nvfp4_scale_lut_detail::kScaleLutSlots];
  sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_populate_constant_lut(
      scale_lut);
  sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_id67_body(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n, scale_lut);
#elif defined(__gfx1030__)
  __shared__ float scale_lut[sllm_id84_nvfp4_scale_lut_detail::kScaleLutSlots];
  sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_populate_constant_lut(
      scale_lut);
  sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_id73_body(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n, scale_lut);
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

namespace sllm_matmul_kernel {
hipError_t launch_fp8_quantize(const uint16_t *const activation,
                               uint8_t *const quantized, float *const scales,
                               const uint64_t m, const uint64_t k,
                               const bool fnuz,
                               const hipStream_t stream) noexcept {
  const char *const force_baseline =
      std::getenv("SLLM_FP8_QUANT_FORCE_BASELINE");
  // Phase 15O did not have a current MI300X tuple. Keep the verified FNUZ
  // provider on v1 until the OCP candidate is independently measured there.
  if (fnuz ||
      (force_baseline != nullptr && std::strcmp(force_baseline, "1") == 0)) {
    hipLaunchKernelGGL(sllm_matmul_bf16_to_fp8_outer_v1,
                       dim3(static_cast<uint32_t>(m)), dim3(kWorkgroupSize), 0U,
                       stream, activation, quantized, scales, m, k,
                       fnuz ? UINT32_C(1) : UINT32_C(0));
  } else {
    hipLaunchKernelGGL(sllm_matmul_bf16_to_fp8_outer_v2,
                       dim3(static_cast<uint32_t>(m)), dim3(kWorkgroupSize), 0U,
                       stream, activation, quantized, scales, m, k,
                       fnuz ? UINT32_C(1) : UINT32_C(0));
  }
  return hipGetLastError();
}

hipError_t launch_fp8_emulation(const uint8_t *const activation,
                                const float *const activation_scales,
                                const uint8_t *const weight,
                                const float *const weight_scales,
                                uint16_t *const output, const uint64_t m,
                                const uint64_t k, const uint64_t n,
                                const hipStream_t stream) noexcept {
  const uint64_t elements = m * n;
  const uint32_t blocks =
      static_cast<uint32_t>((elements + kWorkgroupSize - 1U) / kWorkgroupSize);
  hipLaunchKernelGGL(sllm_matmul_fp8_outer_emulation_v1, dim3(blocks),
                     dim3(kWorkgroupSize), 0U, stream, activation,
                     activation_scales, weight, weight_scales, output, m, k, n);
  return hipGetLastError();
}

hipError_t launch_fp8_e4m3fn_to_fp16_staging(
    const uint8_t *const input, uint16_t *const output,
    const uint64_t element_count, const hipStream_t stream) noexcept {
  if (input == nullptr || output == nullptr || element_count == 0U) {
    return hipErrorInvalidValue;
  }
  constexpr uint64_t elements_per_thread = 4U;
  if (element_count > UINT64_MAX - (elements_per_thread - UINT64_C(1))) {
    return hipErrorInvalidValue;
  }
  const uint64_t work_items =
      (element_count + elements_per_thread - UINT64_C(1)) / elements_per_thread;
  if (work_items > UINT64_MAX - (kWorkgroupSize - UINT64_C(1))) {
    return hipErrorInvalidValue;
  }
  const uint64_t blocks =
      (work_items + kWorkgroupSize - UINT64_C(1)) / kWorkgroupSize;
  if (blocks == 0U || blocks > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_matmul_fp8_e4m3fn_to_fp16_staging_v1,
                     dim3(static_cast<uint32_t>(blocks)), dim3(kWorkgroupSize),
                     0U, stream, input, output, element_count);
  return hipGetLastError();
}

hipError_t launch_fp8_outer_f16scale_epilogue(
    const float *const input, const float *const activation_scales,
    const float *const weight_scales, uint16_t *const output, const uint64_t m,
    const uint64_t n, const hipStream_t stream) noexcept {
  if (input == nullptr || activation_scales == nullptr ||
      weight_scales == nullptr || output == nullptr || m == 0U || n == 0U ||
      m > UINT64_MAX / n) {
    return hipErrorInvalidValue;
  }
  const uint64_t elements = m * n;
  const uint64_t blocks =
      (elements + kWorkgroupSize - UINT64_C(1)) / kWorkgroupSize;
  if (blocks == 0U || blocks > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_matmul_fp8_outer_f16scale_epilogue_v1,
                     dim3(static_cast<uint32_t>(blocks)), dim3(kWorkgroupSize),
                     0U, stream, input, activation_scales, weight_scales,
                     output, m, n);
  return hipGetLastError();
}

hipError_t launch_nvfp4_block16_to_fp16_staging(
    const uint8_t *const packed, const uint8_t *const block_scales,
    uint16_t *const output, const uint64_t rows, const uint64_t k,
    const hipStream_t stream) noexcept {
  if (packed == nullptr || block_scales == nullptr || output == nullptr ||
      rows == 0U || k == 0U || (k % UINT64_C(16)) != 0U ||
      rows > UINT64_MAX / (k / UINT64_C(16))) {
    return hipErrorInvalidValue;
  }
  const uint64_t block_count = rows * (k / UINT64_C(16));
  const uint64_t grid =
      (block_count + kWorkgroupSize - UINT64_C(1)) / kWorkgroupSize;
  if (grid == 0U || grid > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_matmul_nvfp4_block16_to_fp16_staging_v1,
                     dim3(static_cast<uint32_t>(grid)), dim3(kWorkgroupSize),
                     0U, stream, packed, block_scales, output, rows, k);
  return hipGetLastError();
}

hipError_t launch_nvfp4_block16_to_fp8_staging(
    const uint8_t *const packed, const uint8_t *const block_scales,
    uint8_t *const output, const uint64_t rows, const uint64_t k,
    const hipStream_t stream) noexcept {
  if (packed == nullptr || block_scales == nullptr || output == nullptr ||
      rows == 0U || k == 0U || (k % UINT64_C(16)) != 0U ||
      rows > UINT64_MAX / (k / UINT64_C(16))) {
    return hipErrorInvalidValue;
  }
  const uint64_t block_count = rows * (k / UINT64_C(16));
  const uint64_t grid =
      (block_count + kWorkgroupSize - UINT64_C(1)) / kWorkgroupSize;
  if (grid == 0U || grid > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_matmul_nvfp4_block16_to_fp8_staging_v1,
                     dim3(static_cast<uint32_t>(grid)), dim3(kWorkgroupSize),
                     0U, stream, packed, block_scales, output, rows, k);
  return hipGetLastError();
}

hipError_t
launch_nvfp4_tensor_scale_product(const float *const weight_tensor_scale,
                                  const float *const input_tensor_scale,
                                  float *const output,
                                  const hipStream_t stream) noexcept {
  if (weight_tensor_scale == nullptr || input_tensor_scale == nullptr ||
      output == nullptr) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_matmul_nvfp4_tensor_scale_product_v1, dim3(1U),
                     dim3(1U), 0U, stream, weight_tensor_scale,
                     input_tensor_scale, output);
  return hipGetLastError();
}

hipError_t launch_nvfp4_tensor_scale_epilogue(
    const float *const input, const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t n, const hipStream_t stream) noexcept {
  if (input == nullptr || weight_tensor_scale == nullptr ||
      input_tensor_scale == nullptr || output == nullptr || m == 0U ||
      n == 0U || m > UINT64_MAX / n) {
    return hipErrorInvalidValue;
  }
  const uint64_t elements = m * n;
  const uint64_t grid =
      (elements + kWorkgroupSize - UINT64_C(1)) / kWorkgroupSize;
  if (grid == 0U || grid > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_matmul_nvfp4_tensor_scale_epilogue_v1,
                     dim3(static_cast<uint32_t>(grid)), dim3(kWorkgroupSize),
                     0U, stream, input, weight_tensor_scale, input_tensor_scale,
                     output, elements);
  return hipGetLastError();
}

hipError_t launch_fp8_outer_prefill_tiled16(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const hipStream_t stream) noexcept {
  if (m <= 1U || k == 0U || n == 0U) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_matmul_fp8_outer_prefill_tiled16_v1,
                     dim3(static_cast<uint32_t>((n + 15U) / 16U),
                          static_cast<uint32_t>((m + 15U) / 16U)),
                     dim3(kWorkgroupSize), 0U, stream, activation,
                     activation_scales, weight, weight_scales, output, m, k, n);
  return hipGetLastError();
}

hipError_t launch_fp8_outer_prefill_gfx1030_half2_128x64(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const hipStream_t stream) noexcept {
  if (m <= 1U || k == 0U || (k % 2U) != 0U || n == 0U) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(
      sllm_matmul_fp8_outer_prefill_gfx1030_half2_128x64_v1,
      dim3(static_cast<uint32_t>(((m + 127U) / 128U) * ((n + 63U) / 64U))),
      dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
      weight_scales, output, m, k, n);
  return hipGetLastError();
}

hipError_t launch_fp8_outer_prefill_gfx1030_half2_64x64(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const hipStream_t stream) noexcept {
  if (m <= 1U || k == 0U || n == 0U) {
    return hipErrorInvalidValue;
  }
  if (fp8_outer_prefill_gfx1030_half2_short_m32_n64_shape(m, k, n)) {
    hipLaunchKernelGGL(
        sllm_phase78_fp8_short_m32::
            sllm_matmul_fp8_outer_prefill_gfx1030_half2_32x64_v1,
        dim3(grid_size_x(KernelVariant::Fp8OuterPrefillGfx1030Half2_64x64, m, n,
                         k)),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (fp8_outer_prefill_gfx1030_half2_short_m32_n32_shape(m, k, n)) {
    hipLaunchKernelGGL(
        sllm_phase78_fp8_short_m32::
            sllm_matmul_fp8_outer_prefill_gfx1030_half2_32x32_v1,
        dim3(grid_size_x(KernelVariant::Fp8OuterPrefillGfx1030Half2_64x64, m, n,
                         k)),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else {
    hipLaunchKernelGGL(
        sllm_matmul_fp8_outer_prefill_gfx1030_half2_64x64_v1,
        dim3(static_cast<uint32_t>(((m + 63U) / 64U) * ((n + 63U) / 64U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  }
  return hipGetLastError();
}

hipError_t launch_fp8_outer_prefill_gfx1030_lds_lut(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const hipStream_t stream) noexcept {
  if (!fp8_outer_prefill_gfx1030_lds_lut_shape(m, k, n)) {
    return hipErrorInvalidValue;
  }
  const uint64_t row_tiles = (m + 63U) / 64U;
  const uint64_t column_tiles = (n + 63U) / 64U;
  if (row_tiles == 0U || column_tiles == 0U ||
      row_tiles > UINT64_MAX / column_tiles ||
      row_tiles * column_tiles > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_phase78_fp8_prefill_lds_lut::
                         sllm_matmul_fp8_outer_prefill_gfx1030_lds_lut_v1,
                     dim3(static_cast<uint32_t>(row_tiles * column_tiles)),
                     dim3(kFp8OuterPrefillGfx1030LdsLutWorkgroupSize), 0U,
                     stream, activation, activation_scales, weight,
                     weight_scales, output, m, k, n);
  return hipGetLastError();
}

hipError_t launch_fp8_outer_prefill_gfx1030_f16_tile_staging(
    const uint16_t *const activation, const float *const activation_scales,
    const uint16_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const hipStream_t stream) noexcept {
  if (!fp8_outer_prefill_gfx1030_f16_tile_staging_shape(m, k, n)) {
    return hipErrorInvalidValue;
  }
  const uint64_t row_tiles = (m + 63U) / 64U;
  const uint64_t column_tiles = (n + 63U) / 64U;
  if (row_tiles == 0U || column_tiles == 0U ||
      row_tiles > UINT64_MAX / column_tiles ||
      row_tiles * column_tiles > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(
      sllm_phase78_fp8_f16_tile_staging::
          sllm_matmul_fp8_outer_prefill_gfx1030_f16_tile_staging_v1,
      dim3(static_cast<uint32_t>(row_tiles * column_tiles)),
      dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
      weight_scales, output, m, k, n);
  return hipGetLastError();
}

hipError_t launch_fp8_outer_decode_gfx1030_half2_wave4col32(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const hipStream_t stream) noexcept {
  if (!fp8_outer_decode_gfx1030_half2_shape(m, k, n)) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(
      sllm_matmul_fp8_outer_decode_gfx1030_half2_wave4col32_v1,
      dim3(static_cast<uint32_t>(
          (n + kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup - 1U) /
          kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup)),
      dim3(kFp8OuterDecodeGfx1030Half2WorkgroupSize), 0U, stream, activation,
      activation_scales, weight, weight_scales, output, m, k, n);
  return hipGetLastError();
}

hipError_t launch_fp8_outer_decode_gfx1030_dword8_wave4col32(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const hipStream_t stream) noexcept {
  if (!fp8_outer_decode_gfx1030_half2_shape(m, k, n)) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(
      sllm_matmul_fp8_outer_decode_gfx1030_dword8_wave4col32_v1,
      dim3(static_cast<uint32_t>(
          (n + kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup - 1U) /
          kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup)),
      dim3(kFp8OuterDecodeGfx1030Half2WorkgroupSize), 0U, stream, activation,
      activation_scales, weight, weight_scales, output, m, k, n);
  return hipGetLastError();
}

hipError_t launch_fp8_outer_decode_gfx1030_lds_lut_wave4col32(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const hipStream_t stream) noexcept {
  if (!fp8_outer_decode_gfx1030_half2_shape(m, k, n)) {
    return hipErrorInvalidValue;
  }
  if (fp8_outer_decode_gfx1030_lds_lut_tuple_shape(m, k, n)) {
    if (k == UINT64_C(5120) && n == UINT64_C(17408)) {
      return launch_fp8_outer_decode_gfx1030_lds_lut_k5120n17408(
          activation, activation_scales, weight, weight_scales, output, m, k, n,
          stream);
    }
    if (k == UINT64_C(6144) && n == UINT64_C(5120)) {
      return launch_fp8_outer_decode_gfx1030_lds_lut_k6144n5120(
          activation, activation_scales, weight, weight_scales, output, m, k, n,
          stream);
    }
    if (k == UINT64_C(5120) && n == UINT64_C(6144)) {
      return launch_fp8_outer_decode_gfx1030_lds_lut_k5120n6144(
          activation, activation_scales, weight, weight_scales, output, m, k, n,
          stream);
    }
    return launch_fp8_outer_decode_gfx1030_lds_lut_k5120n10240(
        activation, activation_scales, weight, weight_scales, output, m, k, n,
        stream);
  }
  hipLaunchKernelGGL(
      sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_wave4col32_v1,
      dim3(static_cast<uint32_t>(
          (n + kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup - 1U) /
          kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup)),
      dim3(kFp8OuterDecodeGfx1030LdsLutWorkgroupSize), 0U, stream, activation,
      activation_scales, weight, weight_scales, output, m, k, n);
  return hipGetLastError();
}

hipError_t launch_fp8_outer_decode_gfx1030_lds_lut_k5120n17408(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const hipStream_t stream) noexcept {
  if (m != 1U || k != UINT64_C(5120) || n != UINT64_C(17408)) {
    return hipErrorInvalidValue;
  }
  constexpr uint32_t tuple_groups = 10U;
  hipLaunchKernelGGL(
      sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_k5120n17408_v1,
      dim3(static_cast<uint32_t>(
          (n + kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup - 1U) /
          kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup)),
      dim3(kFp8OuterDecodeGfx1030LdsLutWorkgroupSize), 0U, stream, activation,
      activation_scales, weight, weight_scales, output, m, k, n, tuple_groups);
  return hipGetLastError();
}

hipError_t launch_fp8_outer_decode_gfx1030_lds_lut_k6144n5120(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const hipStream_t stream) noexcept {
  if (m != 1U || k != UINT64_C(6144) || n != UINT64_C(5120)) {
    return hipErrorInvalidValue;
  }
  constexpr uint32_t tuple_groups = 12U;
  hipLaunchKernelGGL(
      sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_k6144n5120_v1,
      dim3(static_cast<uint32_t>(
          (n + kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup - 1U) /
          kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup)),
      dim3(kFp8OuterDecodeGfx1030LdsLutWorkgroupSize), 0U, stream, activation,
      activation_scales, weight, weight_scales, output, m, k, n, tuple_groups);
  return hipGetLastError();
}

hipError_t launch_fp8_outer_decode_gfx1030_lds_lut_k5120n10240(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const hipStream_t stream) noexcept {
  if (m != 1U || k != UINT64_C(5120) || n != UINT64_C(10240)) {
    return hipErrorInvalidValue;
  }
  constexpr uint32_t tuple_groups = 10U;
  hipLaunchKernelGGL(
      sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_k5120n10240_v1,
      dim3(static_cast<uint32_t>(
          (n + kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup - 1U) /
          kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup)),
      dim3(kFp8OuterDecodeGfx1030LdsLutWorkgroupSize), 0U, stream, activation,
      activation_scales, weight, weight_scales, output, m, k, n, tuple_groups);
  return hipGetLastError();
}

hipError_t launch_fp8_outer_decode_gfx1030_lds_lut_k5120n6144(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const hipStream_t stream) noexcept {
  if (m != 1U || k != UINT64_C(5120) || n != UINT64_C(6144)) {
    return hipErrorInvalidValue;
  }
  constexpr uint32_t tuple_groups = 10U;
  hipLaunchKernelGGL(
      sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_k5120n6144_v1,
      dim3(static_cast<uint32_t>(
          (n + kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup - 1U) /
          kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup)),
      dim3(kFp8OuterDecodeGfx1030LdsLutWorkgroupSize), 0U, stream, activation,
      activation_scales, weight, weight_scales, output, m, k, n, tuple_groups);
  return hipGetLastError();
}

hipError_t launch_fp8_outer_decode_gfx1030_activation_shared_wave4col32(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const hipStream_t stream) noexcept {
  if (!fp8_outer_decode_gfx1030_activation_shared_wave4_shape(m, k, n)) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(
      sllm_matmul_fp8_outer_decode_gfx1030_actshared_wave4col32_v1,
      dim3(static_cast<uint32_t>(
          (n + kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup - 1U) /
          kFp8OuterDecodeGfx1030Half2ColumnsPerWorkgroup)),
      dim3(kFp8OuterDecodeGfx1030Half2WorkgroupSize),
      static_cast<size_t>(
          fp8_outer_decode_gfx1030_activation_shared_lds_bytes(k)),
      stream, activation, activation_scales, weight, weight_scales, output, m,
      k, n);
  return hipGetLastError();
}

hipError_t launch_fp8_outer_decode_gfx1030_activation_shared_wave8col64(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const hipStream_t stream) noexcept {
  if (!fp8_outer_decode_gfx1030_activation_shared_wave8_shape(m, k, n)) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(
      sllm_matmul_fp8_outer_decode_gfx1030_actshared_wave8col64_v1,
      dim3(static_cast<uint32_t>((n + 63U) / 64U)),
      dim3(kFp8OuterDecodeGfx1030Half2WorkgroupSize),
      static_cast<size_t>(
          fp8_outer_decode_gfx1030_activation_shared_lds_bytes(k)),
      stream, activation, activation_scales, weight, weight_scales, output, m,
      k, n);
  return hipGetLastError();
}

hipError_t launch_nvfp4(const uint16_t *const activation,
                        const uint8_t *const packed_weight,
                        const uint8_t *const block_scales,
                        const float *const tensor_scale, uint16_t *const output,
                        const uint64_t m, const uint64_t k, const uint64_t n,
                        const KernelVariant variant,
                        const hipStream_t stream) noexcept {
  if (variant == KernelVariant::Nvfp4BaselinePackedDequant ||
      variant == KernelVariant::Nvfp4DecodePackedDequant) {
    hipLaunchKernelGGL(sllm_matmul_nvfp4_block16_packed_dequant_v1,
                       dim3(static_cast<uint32_t>(m * n)), dim3(kWorkgroupSize),
                       0U, stream, activation, packed_weight, block_scales,
                       tensor_scale, output, m, k, n);
  } else if (variant == KernelVariant::Nvfp4PrefillRow8Tiled256) {
    hipLaunchKernelGGL(sllm_matmul_nvfp4_block16_prefill_row8_tiled256_v2,
                       dim3(static_cast<uint32_t>(((m + 7U) / 8U) * n)),
                       dim3(kWorkgroupSize), 0U, stream, activation,
                       packed_weight, block_scales, tensor_scale, output, m, k,
                       n);
  } else {
    return hipErrorInvalidValue;
  }
  return hipGetLastError();
}

hipError_t launch_nvfp4_quantize(const uint16_t *const activation,
                                 uint8_t *const packed_activation,
                                 uint8_t *const block_scales,
                                 const float *const input_tensor_scale,
                                 const uint64_t m, const uint64_t k,
                                 const hipStream_t stream) noexcept {
  const uint64_t blocks_per_row = (k + UINT64_C(15)) / UINT64_C(16);
  const char *const force_wave8 =
      std::getenv(sllm_matmul_kernel::kNvfp4ActivationQuantizeWave8Environment);
  const char *const force_baseline = std::getenv("SLLM_NVFP4_FORCE_BASELINE");
  const char *const force_w4a4_baseline =
      std::getenv("SLLM_NVFP4_W4A4_FORCE_BASELINE");
  if (force_wave8 != nullptr && std::strcmp(force_wave8, "1") == 0 &&
      !(force_baseline != nullptr && std::strcmp(force_baseline, "1") == 0) &&
      !(force_w4a4_baseline != nullptr &&
        std::strcmp(force_w4a4_baseline, "1") == 0)) {
    hipLaunchKernelGGL(
        sllm_matmul_bf16_to_nvfp4_block16_wave8_v1,
        dim3(static_cast<uint32_t>((m * blocks_per_row + 7U) / 8U)),
        dim3(kWorkgroupSize), 0U, stream, activation, packed_activation,
        block_scales, input_tensor_scale, m, k);
    return hipGetLastError();
  }
  hipLaunchKernelGGL(sllm_matmul_bf16_to_nvfp4_block16_v1,
                     dim3(static_cast<uint32_t>(m * blocks_per_row)),
                     dim3(kWorkgroupSize), 0U, stream, activation,
                     packed_activation, block_scales, input_tensor_scale, m, k);
  return hipGetLastError();
}

hipError_t launch_nvfp4_w4a4(const uint8_t *const packed_activation,
                             const uint8_t *const activation_block_scales,
                             const uint8_t *const packed_weight,
                             const uint8_t *const weight_block_scales,
                             const float *const weight_tensor_scale,
                             const float *const input_tensor_scale,
                             uint16_t *const output, const uint64_t m,
                             const uint64_t k, const uint64_t n,
                             const KernelVariant variant,
                             const hipStream_t stream) noexcept {
  if (variant != KernelVariant::Nvfp4W4A4Packed &&
      variant != KernelVariant::Nvfp4W4A4Decode &&
      variant != KernelVariant::Nvfp4W4A4PrefillRow8Tiled256 &&
      variant != KernelVariant::Nvfp4W4A4PrefillRow8Col8Tiled256 &&
      variant != KernelVariant::Nvfp4W4A4PrefillDp4a64x64 &&
      variant != KernelVariant::Nvfp4W4A4PrefillDp4a64x64K128 &&
      variant != KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x64 &&
      variant != KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x32 &&
      variant != KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaF16Scale128x64 &&
      variant != KernelVariant::Nvfp4W4A4PrefillGfx1201Fp8Staging &&
      variant != KernelVariant::Nvfp4W4A4DecodeColumns128 &&
      variant != KernelVariant::Nvfp4W4A4DecodeWave4Column32 &&
      variant != KernelVariant::Nvfp4W4A4DecodeActivationShared &&
      variant != KernelVariant::Nvfp4W4A4DecodeScaleLut) {
    return hipErrorInvalidValue;
  }
  if (variant == KernelVariant::Nvfp4W4A4Decode) {
    hipLaunchKernelGGL(sllm_matmul_nvfp4_w4a4_block16_decode_v1,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, packed_activation, activation_block_scales,
                       packed_weight, weight_block_scales, weight_tensor_scale,
                       input_tensor_scale, output, m, k, n);
  } else if (variant == KernelVariant::Nvfp4W4A4DecodeActivationShared) {
    if (!sllm_matmul_kernel::phase78_nvfp4_w4a4_decode_activation_shared_shape(
            m, k, n)) {
      return hipErrorInvalidValue;
    }
    const uint64_t dynamic_shared_bytes =
        sllm_matmul_kernel::nvfp4_w4a4_decode_activation_shared_lds_bytes(k);
    hipLaunchKernelGGL(
        sllm_matmul_nvfp4_w4a4_decode_dp4a_activation_shared_v1,
        dim3(static_cast<uint32_t>((n + UINT64_C(31)) / UINT64_C(32))),
        dim3(sllm_matmul_kernel::kNvfp4W4A4DecodeActivationSharedWorkgroupSize),
        static_cast<size_t>(dynamic_shared_bytes), stream, packed_activation,
        activation_block_scales, packed_weight, weight_block_scales,
        weight_tensor_scale, input_tensor_scale, output, m, k, n);
  } else if (variant == KernelVariant::Nvfp4W4A4DecodeScaleLut) {
    // This launcher is host code: __gfx*__ exists only in the device pass.
    // Use the build's exact target for the matching device body and LDS size.
#if defined(SLLM_HIP_COMPILE_TARGET)
    const bool gfx1030 = std::strcmp(SLLM_HIP_COMPILE_TARGET, "gfx1030") == 0;
    const bool gfx1201 = std::strcmp(SLLM_HIP_COMPILE_TARGET, "gfx1201") == 0;
    if (!(gfx1030 || gfx1201)) {
      return hipErrorNotSupported;
    }
    if (!(gfx1030 ? phase78_nvfp4_w4a4_decode_activation_shared_shape(m, k, n)
                  : phase78_nvfp4_w4a4_decode_wave4col32_shape(m, k, n))) {
      return hipErrorInvalidValue;
    }
    const bool gfx1201_activation_shared =
        gfx1201 &&
        phase78_nvfp4_w4a4_decode_scale_lut_gfx1201_activation_shared_shape(
            m, k, n);
    const size_t dynamic_shared_bytes =
        gfx1030 ? static_cast<size_t>(
                      nvfp4_w4a4_decode_activation_shared_lds_bytes(k))
        : gfx1201_activation_shared
            ? static_cast<size_t>(
                  nvfp4_w4a4_decode_activation_shared_lds_bytes(k))
            : 0U;
    if (gfx1201_activation_shared) {
      hipLaunchKernelGGL(sllm_nvfp4_w4a4_decode_scale_lut_gfx1201_actshared_v1,
                         dim3(static_cast<uint32_t>((n + 31U) / 32U)),
                         dim3(kNvfp4W4A4DecodeScaleLutWorkgroupSize),
                         dynamic_shared_bytes, stream, packed_activation,
                         activation_block_scales, packed_weight,
                         weight_block_scales, weight_tensor_scale,
                         input_tensor_scale, output, m, k, n);
    } else {
      hipLaunchKernelGGL(sllm_matmul_nvfp4_w4a4_decode_scale_lut_v1,
                         dim3(static_cast<uint32_t>((n + 31U) / 32U)),
                         dim3(kNvfp4W4A4DecodeScaleLutWorkgroupSize),
                         dynamic_shared_bytes, stream, packed_activation,
                         activation_block_scales, packed_weight,
                         weight_block_scales, weight_tensor_scale,
                         input_tensor_scale, output, m, k, n);
    }
#else
    return hipErrorNotSupported;
#endif
  } else if (variant == KernelVariant::Nvfp4W4A4DecodeWave4Column32) {
    if (!sllm_matmul_kernel::phase78_nvfp4_w4a4_decode_wave4col32_shape(m, k,
                                                                        n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_matmul_nvfp4_w4a4_decode_dp4a_wave4col32_v1,
        dim3(static_cast<uint32_t>((n + 31U) / 32U)),
        dim3(sllm_matmul_kernel::kNvfp4W4A4DecodeWave4Column32WorkgroupSize),
        0U, stream, packed_activation, activation_block_scales, packed_weight,
        weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
        k, n);
  } else if (variant == KernelVariant::Nvfp4W4A4DecodeColumns128) {
    if (!sllm_matmul_kernel::phase78_nvfp4_w4a4_decode_columns128_shape(m, k,
                                                                        n)) {
      return hipErrorInvalidValue;
    }
    const uint64_t packed_activation_bytes = k / UINT64_C(2);
    const uint64_t blocks_per_row = k / UINT64_C(16);
    const uint64_t dynamic_shared_bytes =
        packed_activation_bytes + blocks_per_row * sizeof(float);
    hipLaunchKernelGGL(
        sllm_matmul_nvfp4_w4a4_decode_columns128_v1,
        dim3(static_cast<uint32_t>((n + 127U) / 128U)),
        dim3(sllm_matmul_kernel::kNvfp4W4A4DecodeColumns128WorkgroupSize),
        static_cast<size_t>(dynamic_shared_bytes), stream, packed_activation,
        activation_block_scales, packed_weight, weight_block_scales,
        weight_tensor_scale, input_tensor_scale, output, m, k, n);
  } else if (variant == KernelVariant::Nvfp4W4A4PrefillRow8Tiled256) {
    hipLaunchKernelGGL(sllm_matmul_nvfp4_w4a4_block16_prefill_row8_tiled256_v1,
                       dim3(static_cast<uint32_t>(((m + 7U) / 8U) * n)),
                       dim3(kWorkgroupSize), 0U, stream, packed_activation,
                       activation_block_scales, packed_weight,
                       weight_block_scales, weight_tensor_scale,
                       input_tensor_scale, output, m, k, n);
  } else if (variant == KernelVariant::Nvfp4W4A4PrefillRow8Col8Tiled256) {
    hipLaunchKernelGGL(
        sllm_matmul_nvfp4_w4a4_block16_prefill_row8_col8_tiled256_v1,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 7U) / 8U))),
        dim3(kWorkgroupSize), 0U, stream, packed_activation,
        activation_block_scales, packed_weight, weight_block_scales,
        weight_tensor_scale, input_tensor_scale, output, m, k, n);
  } else if (variant == KernelVariant::Nvfp4W4A4PrefillDp4a64x64) {
    if (m <= 1U || k == 0U || (k % 16U) != 0U || n == 0U) {
      return hipErrorInvalidValue;
    }
    if (m <= 32U && std::strcmp(SLLM_HIP_COMPILE_TARGET, "gfx1030") == 0) {
      hipLaunchKernelGGL(sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_32x64_v1,
                         dim3(static_cast<uint32_t>((n + 63U) / 64U)),
                         dim3(kWorkgroupSize), 0U, stream, packed_activation,
                         activation_block_scales, packed_weight,
                         weight_block_scales, weight_tensor_scale,
                         input_tensor_scale, output, m, k, n);
      return hipGetLastError();
    }
    if (std::strcmp(SLLM_HIP_COMPILE_TARGET, "gfx1030") == 0 &&
        phase78_nvfp4_w4a4_dp4a_index32_pipeline_shape(m, k, n)) {
      hipLaunchKernelGGL(
          sllm_nvfp4_w4a4_prefill_dp4a64x64_index32_pipeline_v1,
          dim3(static_cast<uint32_t>(((m + 63U) / 64U) * ((n + 63U) / 64U))),
          dim3(kWorkgroupSize), 0U, stream, packed_activation,
          activation_block_scales, packed_weight, weight_block_scales,
          weight_tensor_scale, input_tensor_scale, output, m, k, n);
      return hipGetLastError();
    }
    if (std::strcmp(SLLM_HIP_COMPILE_TARGET, "gfx1030") == 0 &&
        phase78_nvfp4_w4a4_dp4a_index32_shape(m, k, n)) {
      hipLaunchKernelGGL(
          sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_index32_v1,
          dim3(static_cast<uint32_t>(((m + 63U) / 64U) * ((n + 63U) / 64U))),
          dim3(kWorkgroupSize), 0U, stream, packed_activation,
          activation_block_scales, packed_weight, weight_block_scales,
          weight_tensor_scale, input_tensor_scale, output, m, k, n);
      return hipGetLastError();
    }
    hipLaunchKernelGGL(
        sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_v1,
        dim3(static_cast<uint32_t>(((m + 63U) / 64U) * ((n + 63U) / 64U))),
        dim3(kWorkgroupSize), 0U, stream, packed_activation,
        activation_block_scales, packed_weight, weight_block_scales,
        weight_tensor_scale, input_tensor_scale, output, m, k, n);
  } else if (variant == KernelVariant::Nvfp4W4A4PrefillDp4a64x64K128) {
    if (m <= 1U || k == 0U || (k % 16U) != 0U || n == 0U) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_k128_v1,
        dim3(static_cast<uint32_t>(((m + 63U) / 64U) * ((n + 63U) / 64U))),
        dim3(kWorkgroupSize), 0U, stream, packed_activation,
        activation_block_scales, packed_weight, weight_block_scales,
        weight_tensor_scale, input_tensor_scale, output, m, k, n);
  } else if (variant == KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x64) {
    if (m <= 1U || k == 0U || (k % 16U) != 0U || n == 0U) {
      return hipErrorInvalidValue;
    }
    if (phase78_gfx1201_nvfp4_wmma_ordinary_shape(m, k, n)) {
      hipLaunchKernelGGL(sllm_nvfp4_w4a4_prefill_gfx1201_wmma_ordinary_v1,
                         dim3(static_cast<uint32_t>((n + 63U) / 64U),
                              static_cast<uint32_t>((m + 127U) / 128U)),
                         dim3(kWorkgroupSize), 0U, stream, packed_activation,
                         activation_block_scales, packed_weight,
                         weight_block_scales, weight_tensor_scale,
                         input_tensor_scale, output, m, k, n);
    } else {
      hipLaunchKernelGGL(sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_v1,
                         dim3(static_cast<uint32_t>((n + 63U) / 64U),
                              static_cast<uint32_t>((m + 127U) / 128U)),
                         dim3(kWorkgroupSize), 0U, stream, packed_activation,
                         activation_block_scales, packed_weight,
                         weight_block_scales, weight_tensor_scale,
                         input_tensor_scale, output, m, k, n);
    }
  } else if (variant == KernelVariant::Nvfp4W4A4PrefillGfx1201Wmma128x32) {
    if (!sllm_matmul_kernel::phase78_gfx1201_nvfp4_w4a4_wmma128x32_shape(m, k,
                                                                         n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x32_v1,
                       dim3(static_cast<uint32_t>((n + 31U) / 32U),
                            static_cast<uint32_t>((m + 127U) / 128U)),
                       dim3(kWorkgroupSize), 0U, stream, packed_activation,
                       activation_block_scales, packed_weight,
                       weight_block_scales, weight_tensor_scale,
                       input_tensor_scale, output, m, k, n);
  } else if (variant ==
             KernelVariant::Nvfp4W4A4PrefillGfx1201WmmaF16Scale128x64) {
    if (!sllm_matmul_kernel::phase78_gfx1201_nvfp4_w4a4_wmma128x64_shape(m, k,
                                                                         n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_nvfp4_w4a4_prefill_gfx1201_wmma_f16scale128x64_v1,
        dim3(static_cast<uint32_t>((n + 63U) / 64U),
             static_cast<uint32_t>((m + 127U) / 128U)),
        dim3(sllm_matmul_kernel::
                 kNvfp4W4A4PrefillGfx1201WmmaF16ScaleWorkgroupSize),
        0U, stream, packed_activation, activation_block_scales, packed_weight,
        weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
        k, n);
  } else if (variant == KernelVariant::Nvfp4W4A4PrefillGfx1201Fp8Staging) {
    return hipErrorInvalidValue;
  } else {
    hipLaunchKernelGGL(sllm_matmul_nvfp4_w4a4_block16_packed_v1,
                       dim3(static_cast<uint32_t>(m * n)), dim3(kWorkgroupSize),
                       0U, stream, packed_activation, activation_block_scales,
                       packed_weight, weight_block_scales, weight_tensor_scale,
                       input_tensor_scale, output, m, k, n);
  }
  return hipGetLastError();
}

// ID62 short split-K4 launcher. The caller reserves
// 4 * m * n * sizeof(float) bytes for partial_workspace. Only the two
// measured gfx1030 M=17 projections are accepted; all other shapes remain on
// their existing ID62 paths in the runtime layer.
hipError_t launch_nvfp4_w4a4_prefill_dp4a_short_split4(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    float *const partial_workspace, const uint64_t m, const uint64_t k,
    const uint64_t n, const hipStream_t stream) noexcept {
#if defined(SLLM_HIP_COMPILE_TARGET)
  if (std::strcmp(SLLM_HIP_COMPILE_TARGET, "gfx1030") != 0) {
    return hipErrorNotSupported;
  }
#else
  return hipErrorNotSupported;
#endif
  const bool exact_wide =
      m == UINT64_C(17) && k == UINT64_C(5120) && n == UINT64_C(17408);
  const bool exact_down =
      m == UINT64_C(17) && k == UINT64_C(17408) && n == UINT64_C(5120);
  if ((!exact_wide && !exact_down) || partial_workspace == nullptr) {
    return hipErrorInvalidValue;
  }
  const uint64_t tile_columns = (n + UINT64_C(63)) / UINT64_C(64);
  const uint64_t tile_rows = (m + UINT64_C(31)) / UINT64_C(32);
  const uint64_t producer_blocks = tile_rows * tile_columns;
  hipLaunchKernelGGL(
      sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_short_split4_produce_v1,
      dim3(static_cast<uint32_t>(producer_blocks), 1U, 4U),
      dim3(kWorkgroupSize), 0U, stream, packed_activation,
      activation_block_scales, packed_weight, weight_block_scales,
      partial_workspace, m, k, n);
  const hipError_t producer_status = hipGetLastError();
  if (producer_status != hipSuccess)
    return producer_status;
  const uint64_t elements = m * n;
  const uint64_t reduce_blocks =
      (elements + kWorkgroupSize - UINT64_C(1)) / kWorkgroupSize;
  hipLaunchKernelGGL(
      sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_short_split4_reduce_v1,
      dim3(static_cast<uint32_t>(reduce_blocks)), dim3(kWorkgroupSize), 0U,
      stream, partial_workspace, weight_tensor_scale, input_tensor_scale,
      output, m, n);
  return hipGetLastError();
}

// ID64 gfx1201 short split-K4 launcher. Only the measured M=17 down shape is
// accepted; all other ID64 shapes retain the original single-kernel route.
hipError_t launch_nvfp4_w4a4_prefill_gfx1201_wmma128x64_split4(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    float *const partial_workspace, const uint64_t m, const uint64_t k,
    const uint64_t n, const hipStream_t stream) noexcept {
#if defined(SLLM_HIP_COMPILE_TARGET)
  if (std::strcmp(SLLM_HIP_COMPILE_TARGET, "gfx1201") != 0) {
    return hipErrorNotSupported;
  }
#else
  return hipErrorNotSupported;
#endif
  if (!phase78_gfx1201_nvfp4_w4a4_split4_shape(m, k, n) ||
      partial_workspace == nullptr) {
    return hipErrorInvalidValue;
  }
  const uint64_t tile_columns = (n + UINT64_C(63)) / UINT64_C(64);
  const uint64_t tile_rows = (m + UINT64_C(127)) / UINT64_C(128);
  hipLaunchKernelGGL(
      sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_split4_partial_v1,
      dim3(static_cast<uint32_t>(tile_columns),
           static_cast<uint32_t>(tile_rows), 4U),
      dim3(kWorkgroupSize), 0U, stream, packed_activation,
      activation_block_scales, packed_weight, weight_block_scales,
      partial_workspace, m, k, n);
  const hipError_t partial_status = hipGetLastError();
  if (partial_status != hipSuccess)
    return partial_status;
  const uint64_t elements = m * n;
  const uint64_t reduce_blocks =
      (elements + kWorkgroupSize - UINT64_C(1)) / kWorkgroupSize;
  hipLaunchKernelGGL(
      sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_split4_reduce_v1,
      dim3(static_cast<uint32_t>(reduce_blocks)), dim3(kWorkgroupSize), 0U,
      stream, partial_workspace, weight_tensor_scale, input_tensor_scale,
      output, m, n);
  return hipGetLastError();
}

hipError_t launch_mxfp4_quantize(const uint16_t *const activation,
                                 uint8_t *const packed_activation,
                                 uint8_t *const block_scales, const uint64_t m,
                                 const uint64_t k,
                                 const hipStream_t stream) noexcept {
  const uint64_t blocks_per_row = (k + UINT64_C(31)) / UINT64_C(32);
  hipLaunchKernelGGL(sllm_matmul_bf16_to_mxfp4_block32_even_v1,
                     dim3(static_cast<uint32_t>(m * blocks_per_row)),
                     dim3(kWorkgroupSize), 0U, stream, activation,
                     packed_activation, block_scales, m, k);
  return hipGetLastError();
}

hipError_t launch_mxfp4_w4a4(const uint8_t *const packed_activation,
                             const uint8_t *const activation_block_scales,
                             const uint8_t *const packed_weight,
                             const uint8_t *const weight_block_scales,
                             uint16_t *const output, const uint64_t m,
                             const uint64_t k, const uint64_t n,
                             const KernelVariant variant,
                             const hipStream_t stream) noexcept {
  if (variant == KernelVariant::Mxfp4W4A4Decode) {
    hipLaunchKernelGGL(sllm_matmul_mxfp4_w4a4_block32_decode_v1,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, packed_activation, activation_block_scales,
                       packed_weight, weight_block_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp4W4A4Prefill) {
    hipLaunchKernelGGL(sllm_matmul_mxfp4_w4a4_block32_prefill_v1,
                       dim3(static_cast<uint32_t>(m * n)), dim3(kWorkgroupSize),
                       0U, stream, packed_activation, activation_block_scales,
                       packed_weight, weight_block_scales, output, m, k, n);
  } else {
    return hipErrorInvalidValue;
  }
  return hipGetLastError();
}

hipError_t launch_mxfp8_quantize(const uint16_t *const activation,
                                 uint8_t *const quantized,
                                 uint8_t *const block_scales, const uint64_t m,
                                 const uint64_t k,
                                 const hipStream_t stream) noexcept {
  const uint64_t blocks_per_row = k / UINT64_C(32);
  hipLaunchKernelGGL(sllm_matmul_bf16_to_mxfp8_e4m3_block32_v1,
                     dim3(static_cast<uint32_t>(m * blocks_per_row)), dim3(32U),
                     0U, stream, activation, quantized, block_scales, m, k);
  return hipGetLastError();
}

hipError_t launch_mxfp8_w8a8(const uint8_t *const activation,
                             const uint8_t *const activation_scales,
                             const uint8_t *const weight,
                             const uint8_t *const weight_scales,
                             uint16_t *const output, const uint64_t m,
                             const uint64_t k, const uint64_t n,
                             const hipStream_t stream) noexcept {
  const KernelVariant variant = select_mxfp8_variant(m);
  return launch_mxfp8_w8a8(activation, activation_scales, weight, weight_scales,
                           output, m, k, n, variant, stream);
}

hipError_t launch_mxfp8_w8a8(const uint8_t *const activation,
                             const uint8_t *const activation_scales,
                             const uint8_t *const weight,
                             const uint8_t *const weight_scales,
                             uint16_t *const output, const uint64_t m,
                             const uint64_t k, const uint64_t n,
                             const KernelVariant variant,
                             const hipStream_t stream) noexcept {
  if (variant == KernelVariant::Mxfp8W8A8Decode) {
    hipLaunchKernelGGL(sllm_matmul_mxfp8_w8a8_e4m3_block32_decode_v1,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, activation, activation_scales, weight,
                       weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8Prefill) {
    hipLaunchKernelGGL(sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_v1,
                       dim3(static_cast<uint32_t>(m * n)), dim3(kWorkgroupSize),
                       0U, stream, activation, activation_scales, weight,
                       weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillRow8) {
    hipLaunchKernelGGL(sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_row8_v2,
                       dim3(static_cast<uint32_t>(((m + 7U) / 8U) * n)),
                       dim3(kWorkgroupSize), 0U, stream, activation,
                       activation_scales, weight, weight_scales, output, m, k,
                       n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillMmqCol4) {
    hipLaunchKernelGGL(
        sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_mmq_col4_v4,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 3U) / 4U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillMmqCol8) {
    hipLaunchKernelGGL(
        sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_mmq_col8_v4,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 7U) / 8U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col16) {
    if (!phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1030_mmq_col16_v1,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 15U) / 16U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col32) {
    if (!phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1030_mmq_col32_v1,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 31U) / 32U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Regscale) {
    if (!phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1030_mmq_col8_regscale_v1,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 7U) / 8U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Vector32) {
    if (!phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1030_mmq_col8_vector32_v1,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 7U) / 8U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant ==
             KernelVariant::Mxfp8W8A8PrefillMmqGfx1030RegscaleVector32) {
    if (!phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1030_mmq_col8_regscale_vector32_v1,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 7U) / 8U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_32x32K32) {
    if (!phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1030_half2_32x32_k32_v1,
        dim3(static_cast<uint32_t>(((m + 31U) / 32U) * ((n + 31U) / 32U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_64x64K32) {
    if (!phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1030_half2_64x64_k32_v1,
        dim3(static_cast<uint32_t>(((m + 63U) / 64U) * ((n + 63U) / 64U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x32K32) {
    if (!phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1030_half2_128x32_k32_v1,
        dim3(static_cast<uint32_t>(((m + 127U) / 128U) * ((n + 31U) / 32U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K32) {
    if (!phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1030_half2_128x64_k32_v1,
        dim3(static_cast<uint32_t>(((m + 127U) / 128U) * ((n + 63U) / 64U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K64) {
    if (!phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1030_half2_128x64_k64_v1,
        dim3(static_cast<uint32_t>(((m + 127U) / 128U) * ((n + 63U) / 64U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant ==
             KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K128) {
    if (!phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1030_half2_128x64_k128_v1,
        dim3(static_cast<uint32_t>(((m + 127U) / 128U) * ((n + 63U) / 64U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant ==
             KernelVariant::Mxfp8W8A8PrefillGfx1030Half2_128x64K32Double) {
    if (!phase67_mxfp8_mmq_gfx1030_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1030_half2_128x64_k32_double_v1,
        dim3(static_cast<uint32_t>(((m + 127U) / 128U) * ((n + 63U) / 64U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillWmmaN16) {
    if (!phase63_mxfp8_wmma_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_wmma128x16x32_v1,
        dim3(static_cast<uint32_t>(
                 (n + kMxfp8W8A8PrefillWmmaN16ColumnsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaN16ColumnsPerWorkgroup),
             static_cast<uint32_t>(
                 (m + kMxfp8W8A8PrefillWmmaRowsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmmaWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillWmmaN64) {
    if (!phase63_mxfp8_wmma_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_wmma128x64x32_v2,
        dim3(static_cast<uint32_t>(
                 (n + kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup),
             static_cast<uint32_t>(
                 (m + kMxfp8W8A8PrefillWmmaRowsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmmaWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillWmma4Wave) {
    if (!phase64_mxfp8_wmma_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1201_wmma64x64_4w_v1,
        dim3(static_cast<uint32_t>(n /
                                   kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup),
             static_cast<uint32_t>(
                 (m + kMxfp8W8A8PrefillWmma4WaveRowsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmma4WaveRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmma4WaveWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillWmmaLdsPad) {
    if (!phase64_mxfp8_wmma_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1201_wmma128x64_pad33_v1,
        dim3(static_cast<uint32_t>(n /
                                   kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup),
             static_cast<uint32_t>(
                 (m + kMxfp8W8A8PrefillWmmaRowsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmmaWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillWmmaDirectWeight) {
    if (!phase64_mxfp8_wmma_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1201_wmma128x64_direct_v1,
        dim3(static_cast<uint32_t>(n /
                                   kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup),
             static_cast<uint32_t>(
                 (m + kMxfp8W8A8PrefillWmmaRowsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmmaWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillWmmaDirectActivation) {
    if (!phase65_mxfp8_wmma_direct_activation_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1201_wmma128x64_adirect_v1,
        dim3(static_cast<uint32_t>(n /
                                   kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup),
             static_cast<uint32_t>(m / kMxfp8W8A8PrefillWmmaRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmmaWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillWmmaDirectBoth) {
    if (!phase65_mxfp8_wmma_direct_activation_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1201_wmma128x64_bdirect_v1,
        dim3(static_cast<uint32_t>(n /
                                   kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup),
             static_cast<uint32_t>(m / kMxfp8W8A8PrefillWmmaRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmmaWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillWmmaN128DirectBoth) {
    if (!phase66_mxfp8_wmma_n128_direct_both_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp8_w8a8_gfx1201_wmma128x128_bdirect_v1,
        dim3(static_cast<uint32_t>(
                 n / kMxfp8W8A8PrefillWmmaN128ColumnsPerWorkgroup),
             static_cast<uint32_t>(m / kMxfp8W8A8PrefillWmmaRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmmaWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp8W8A8PrefillTiled16) {
    hipLaunchKernelGGL(sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_tiled16_v3,
                       dim3(static_cast<uint32_t>((n + 15U) / 16U),
                            static_cast<uint32_t>((m + 15U) / 16U)),
                       dim3(16U, 16U), 0U, stream, activation,
                       activation_scales, weight, weight_scales, output, m, k,
                       n);
  } else {
    return hipErrorInvalidValue;
  }
  return hipGetLastError();
}

hipError_t launch_mxfp6_quantize(const uint16_t *const activation,
                                 uint8_t *const packed,
                                 uint8_t *const block_scales, const uint64_t m,
                                 const uint64_t k,
                                 const hipStream_t stream) noexcept {
  const uint64_t blocks_per_row = k / UINT64_C(32);
  hipLaunchKernelGGL(sllm_matmul_bf16_to_mxfp6_e3m2_block32_v1,
                     dim3(static_cast<uint32_t>(m * blocks_per_row)), dim3(32U),
                     0U, stream, activation, packed, block_scales, m, k);
  return hipGetLastError();
}

hipError_t launch_mxfp6_w6a6(const uint8_t *const activation,
                             const uint8_t *const activation_scales,
                             const uint8_t *const weight,
                             const uint8_t *const weight_scales,
                             uint16_t *const output, const uint64_t m,
                             const uint64_t k, const uint64_t n,
                             const KernelVariant variant,
                             const hipStream_t stream) noexcept {
  if (variant == KernelVariant::Mxfp6W6A6Decode) {
    hipLaunchKernelGGL(sllm_matmul_mxfp6_w6a6_e3m2_block32_decode_v1,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, activation, activation_scales, weight,
                       weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp6W6A6Prefill) {
    hipLaunchKernelGGL(sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_v1,
                       dim3(static_cast<uint32_t>(m * n)), dim3(kWorkgroupSize),
                       0U, stream, activation, activation_scales, weight,
                       weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp6W6A6PrefillRow8) {
    hipLaunchKernelGGL(sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_row8_v2,
                       dim3(static_cast<uint32_t>(((m + 7U) / 8U) * n)),
                       dim3(kWorkgroupSize), 0U, stream, activation,
                       activation_scales, weight, weight_scales, output, m, k,
                       n);
  } else if (variant == KernelVariant::Mxfp6W6A6PrefillMmqCol4) {
    hipLaunchKernelGGL(
        sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_mmq_col4_v4,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 3U) / 4U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp6W6A6PrefillMmqCol8) {
    hipLaunchKernelGGL(
        sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_mmq_col8_v4,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 7U) / 8U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp6W6A6PrefillMmqGfx1030ViaE4M3) {
    if (!phase70_mxfp6_via_e4m3_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp6_w6a6_gfx1030_mmq_col8_via_e4m3_v1,
        dim3(static_cast<uint32_t>(((m + 7U) / 8U) * ((n + 7U) / 8U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp6W6A6PrefillGfx1030Half2Dot2) {
    if (!phase74_gfx1030_mxfp6_half2_dot2_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp6_w6a6_gfx1030_half2_32x32_v1,
        dim3(static_cast<uint32_t>(((m + 31U) / 32U) * ((n + 31U) / 32U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant ==
             KernelVariant::
                 Mxfp6W6A6PrefillGfx1030Half2_128x64K32DoubleScalar) {
    if (!phase74_gfx1030_mxfp6_half2_dot2_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp6_w6a6_gfx1030_half2_128x64_k32d_scalar_v1,
        dim3(static_cast<uint32_t>(((m + 127U) / 128U) * ((n + 63U) / 64U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant ==
             KernelVariant::Mxfp6W6A6PrefillGfx1030Half2_128x64K32DoublePack4) {
    if (!phase74_gfx1030_mxfp6_half2_dot2_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp6_w6a6_gfx1030_half2_128x64_k32d_pack4_v1,
        dim3(static_cast<uint32_t>(((m + 127U) / 128U) * ((n + 63U) / 64U))),
        dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
        weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201ViaE4M3N64) {
    if (!phase70_mxfp6_via_e4m3_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp6_w6a6_gfx1201_wmma128x64_via_e4m3_v1,
        dim3(static_cast<uint32_t>(
                 (n + kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup),
             static_cast<uint32_t>(
                 (m + kMxfp8W8A8PrefillWmmaRowsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmmaWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4N64) {
    if (!phase70_mxfp6_via_e4m3_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp6_w6a6_gfx1201_wmma128x64_pack4_v2,
        dim3(static_cast<uint32_t>(
                 (n + kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup),
             static_cast<uint32_t>(
                 (m + kMxfp8W8A8PrefillWmmaRowsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmmaWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4Swar) {
    if (!phase70_mxfp6_via_e4m3_supported_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp6_w6a6_gfx1201_wmma128x64_pack4_swar_v1,
        dim3(static_cast<uint32_t>(
                 (n + kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaN64ColumnsPerWorkgroup),
             static_cast<uint32_t>(
                 (m + kMxfp8W8A8PrefillWmmaRowsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmmaWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp6W6A6PrefillWmmaGfx1201Pack4N128) {
    if (!phase70_gfx1201_mxfp6_wmma_pack4_n128_shape(m, k, n)) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        sllm_mxfp6_w6a6_gfx1201_wmma128x128_pack4_v1,
        dim3(static_cast<uint32_t>(
                 (n + kMxfp8W8A8PrefillWmmaN128ColumnsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaN128ColumnsPerWorkgroup),
             static_cast<uint32_t>(
                 (m + kMxfp8W8A8PrefillWmmaRowsPerWorkgroup - 1U) /
                 kMxfp8W8A8PrefillWmmaRowsPerWorkgroup)),
        dim3(kMxfp8W8A8PrefillWmmaWorkgroupSize), 0U, stream, activation,
        activation_scales, weight, weight_scales, output, m, k, n);
  } else if (variant == KernelVariant::Mxfp6W6A6PrefillTiled16) {
    hipLaunchKernelGGL(sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_tiled16_v3,
                       dim3(static_cast<uint32_t>((n + 15U) / 16U),
                            static_cast<uint32_t>((m + 15U) / 16U)),
                       dim3(16U, 16U), 0U, stream, activation,
                       activation_scales, weight, weight_scales, output, m, k,
                       n);
  } else {
    return hipErrorInvalidValue;
  }
  return hipGetLastError();
}

hipError_t launch(const uint16_t *const activation,
                  const uint16_t *const weight, uint16_t *const output,
                  const uint64_t m, const uint64_t k, const uint64_t n,
                  const KernelVariant variant,
                  const hipStream_t stream) noexcept {
  if (variant == KernelVariant::HipBlas) {
    return hipErrorInvalidValue;
  }
  if (variant == KernelVariant::DecodeReductionWave64) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_decode_wave64_v1,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, activation, weight, output, k, n);
  } else if (variant == KernelVariant::DecodeReduction) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_decode_v4,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, activation, weight, output, k, n);
  } else if (variant == KernelVariant::SerialRowsReductionWave64) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_decode_serial_rows_wave64_v1,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, activation, weight, output, m, k, n);
  } else if (variant == KernelVariant::SerialRowsReduction) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_decode_serial_rows_v1,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, activation, weight, output, m, k, n);
  } else if (variant == KernelVariant::PrefillShortSerial) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_prefill_short_serial_v1,
                       dim3(grid_size_x(variant, m, n)), dim3(kWorkgroupSize),
                       0U, stream, activation, weight, output, m, k, n);
  } else if (variant == KernelVariant::PrefillTiled16) {
#if defined(SLLM_HIP_COMPILE_TARGET)
    const bool gdn_thin_shape =
        (std::strcmp(SLLM_HIP_COMPILE_TARGET, "gfx1030") == 0 ||
         std::strcmp(SLLM_HIP_COMPILE_TARGET, "gfx1201") == 0) &&
        m == UINT64_C(17) && k == UINT64_C(5120) && n == UINT64_C(48);
#else
    constexpr bool gdn_thin_shape = false;
#endif
    if (gdn_thin_shape) {
      hipLaunchKernelGGL(
          sllm_matmul_bf16_fp32_prefill_gdn_thin_v1,
          dim3(static_cast<uint32_t>(n), static_cast<uint32_t>(m)),
          dim3(kWorkgroupSize), 0U, stream, activation, weight, output, m, k,
          n);
    } else {
      hipLaunchKernelGGL(sllm_matmul_bf16_fp32_tiled16_v2,
                         dim3(static_cast<uint32_t>((n + 15U) / 16U),
                              static_cast<uint32_t>((m + 15U) / 16U)),
                         dim3(16U, 16U), 0U, stream, activation, weight, output,
                         m, k, n);
    }
  } else {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_v1,
                       dim3(grid_size_x(variant, m, n)), dim3(kWorkgroupSize),
                       0U, stream, activation, weight, output, m, k, n);
  }
  return hipGetLastError();
}

hipError_t launch_short_mixed_f32_to_bf16(const float *const output_f32,
                                          uint16_t *const output,
                                          const uint64_t element_count,
                                          const hipStream_t stream) noexcept {
  if (element_count == 0U || element_count > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_matmul_fp32_to_bf16_short_mixed_v1,
                     dim3(static_cast<uint32_t>((element_count + 255U) / 256U)),
                     dim3(kWorkgroupSize), 0U, stream, output_f32, output,
                     element_count);
  return hipGetLastError();
}

} // namespace sllm_matmul_kernel
