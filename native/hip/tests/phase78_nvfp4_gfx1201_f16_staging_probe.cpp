// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase78-nvfp4-byte-permute-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// 3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70, ggml/src/ggml-cuda/vecdotq.cuh
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

// Phase 78 standalone R9700 NVFP4 W4A4 transient-FP16 staging probe.
//
// The ID64 control is the existing gfx1201 FP8-WMMA provider shape: packed
// NVFP4 values are expanded to exact E4M3 bytes in a 128x64 tile and each
// block16 scale is applied around the FP32 WMMA contribution.  The staging
// candidate expands the resident NVFP4 matrices into FP16, executes one
// rocBLAS F16/F16->F32 GEMM, then applies the tensor scale and BF16-RNE output
// epilogue.  This is developer evidence only; no runtime selector is changed.

#include "low_precision_block_codec.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <utility>
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
constexpr uint32_t kWarmups = 3U;
constexpr uint32_t kMeasured = 10U;
constexpr std::array<uint32_t, 3> kNumericalSeeds = {
    UINT32_C(0x243f6a88), UINT32_C(0x85a308d3), UINT32_C(0x13198a2e)};

__device__ __forceinline__ float e2m1_to_float(const uint8_t code) {
  constexpr float positive[8] = {0.0F, 0.5F, 1.0F, 1.5F,
                                 2.0F, 3.0F, 4.0F, 6.0F};
  const float value = positive[code & 7U];
  return (code & 8U) == 0U ? value : -value;
}

__device__ __forceinline__ float e4m3fn_to_float(const uint8_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x80U) << 24U;
  const uint32_t magnitude = bits & 0x7fU;
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  if (exponent == 0U) {
    if (mantissa == 0U) {
      return __uint_as_float(sign);
    }
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
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

__device__ __forceinline__ uint32_t
e2m1x4_to_e4m3fn_exact(const uint16_t packed) {
  const uint32_t lanes = (static_cast<uint32_t>(packed) & 0x000fU) |
                         ((static_cast<uint32_t>(packed) & 0x00f0U) << 4U) |
                         ((static_cast<uint32_t>(packed) & 0x0f00U) << 8U) |
                         ((static_cast<uint32_t>(packed) & 0xf000U) << 12U);
  constexpr uint32_t positive_0_3 = UINT32_C(0x3c383000);
  constexpr uint32_t positive_4_7 = UINT32_C(0x4c484440);
  constexpr uint32_t low_index_mask = UINT32_C(0x07070707);
  const uint32_t positive =
      __builtin_amdgcn_perm(positive_4_7, positive_0_3, lanes & low_index_mask);
  return positive | ((lanes & UINT32_C(0x08080808)) << 4U);
}

// ID64 control copied as a standalone provider so its 128x64 WMMA and scale
// order can be compared with the library staging pipeline.
__global__ __launch_bounds__(kThreads, 1) void id64_nvfp4_wmma128x64_kernel(
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
      activation_groups[group] = e2m1x4_to_e4m3fn_exact(packed);
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
      weight_groups[group] = e2m1x4_to_e4m3fn_exact(packed);
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
        const auto row_major =
            rocwmma::apply_data_layout<rocwmma::row_major>(contribution);
#pragma unroll
        for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
          const uint32_t local_row =
              (lane / tile_n) * (output_values / wave_width) + slot;
          const uint32_t local_column = lane % tile_n;
          float term = row_major[slot] *
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
        output[row * n + column] =
            bf16_rne(accumulators[column_tile][slot] * tensor_scale);
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

__global__ __launch_bounds__(256, 1) void nvfp4_to_fp16_kernel(
    const uint8_t *const packed, const uint8_t *const scales,
    uint16_t *const output, const uint64_t rows, const uint64_t k) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t elements = rows * k;
  if (index >= elements) {
    return;
  }
  const uint64_t row = index / k;
  const uint64_t inner = index - row * k;
  const uint8_t pair = __builtin_nontemporal_load(packed + index / 2U);
  const uint8_t code = (index & 1U) == 0U ? pair & 0x0fU : pair >> 4U;
  const float value = e2m1_to_float(code) *
                      e4m3fn_to_float(scales[row * (k / 16U) + inner / 16U]);
  const __half half_value = __float2half_rn(value);
  output[index] = static_cast<__half_raw>(half_value).x;
}

// One thread owns one block16: a single 64-bit packed load and one E4M3 scale
// load produce all sixteen FP16 operands.  This removes the scalar ingress
// kernel's repeated packed-byte and scale loads from the staging critical path.
__global__ __launch_bounds__(256, 1) void nvfp4_to_fp16_block16_kernel(
    const uint8_t *const packed, const uint8_t *const scales,
    uint16_t *const output, const uint64_t rows, const uint64_t k) {
  const uint64_t blocks_per_row = k / 16U;
  const uint64_t block_index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t blocks = rows * blocks_per_row;
  if (block_index >= blocks) {
    return;
  }
  const uint64_t row = block_index / blocks_per_row;
  const uint64_t block = block_index - row * blocks_per_row;
  const uint64_t packed_offset = row * (k / 2U) + block * 8U;
  const uint64_t packed_values = __builtin_nontemporal_load(
      reinterpret_cast<const uint64_t *>(packed + packed_offset));
  const float scale =
      e4m3fn_to_float(__builtin_nontemporal_load(scales + block_index));
  const uint64_t output_offset = row * k + block * 16U;
#pragma unroll
  for (uint32_t index = 0U; index < 16U; ++index) {
    const uint8_t pair =
        static_cast<uint8_t>(packed_values >> ((index / 2U) * 8U));
    const uint8_t code = (index & 1U) == 0U ? pair & 0x0fU : pair >> 4U;
    const __half half_value = __float2half_rn(e2m1_to_float(code) * scale);
    output[output_offset + index] = static_cast<__half_raw>(half_value).x;
  }
}

__global__ __launch_bounds__(256, 1) void bf16_epilogue_kernel(
    const float *const input, uint16_t *const output, const uint64_t elements,
    const float tensor_scale) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = bf16_rne(input[index] * tensor_scale);
  }
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

bool exact_gfx1201(const char *const arch) {
  if (arch == nullptr)
    return false;
  const std::string_view value(arch);
  constexpr std::string_view prefix = "gfx1201";
  return value == prefix || (value.size() > prefix.size() &&
                             value.compare(0U, prefix.size(), prefix) == 0 &&
                             value[prefix.size()] == ':');
}

float host_e2m1(const uint8_t code) {
  constexpr std::array<float, 8> positive = {0.0F, 0.5F, 1.0F, 1.5F,
                                             2.0F, 3.0F, 4.0F, 6.0F};
  const float value = positive[code & 7U];
  return (code & 8U) == 0U ? value : -value;
}

float host_e4m3(const uint8_t bits) {
  const uint32_t magnitude = bits & 0x7fU;
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  if (exponent == 0U)
    return static_cast<float>(mantissa) * 0x1p-9F;
  if (magnitude == 0x7fU)
    return std::numeric_limits<float>::quiet_NaN();
  return std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                    static_cast<int>(exponent) - 7);
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

struct Buffers final {
  uint8_t *activation = nullptr;
  uint8_t *activation_scales = nullptr;
  uint8_t *weight = nullptr;
  uint8_t *weight_scales = nullptr;
  uint16_t *activation_fp16 = nullptr;
  uint16_t *weight_fp16 = nullptr;
  float *gemm_output = nullptr;
  uint16_t *output = nullptr;
  float *weight_tensor_scale = nullptr;
  float *input_tensor_scale = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  rocblas_handle rocblas = nullptr;
};

void cleanup(Buffers *const b) {
  if (b == nullptr)
    return;
  if (b->rocblas != nullptr)
    (void)rocblas_destroy_handle(b->rocblas);
  if (b->stop != nullptr)
    (void)hipEventDestroy(b->stop);
  if (b->start != nullptr)
    (void)hipEventDestroy(b->start);
  if (b->stream != nullptr)
    (void)hipStreamDestroy(b->stream);
  if (b->output != nullptr)
    (void)hipFree(b->output);
  if (b->gemm_output != nullptr)
    (void)hipFree(b->gemm_output);
  if (b->weight_fp16 != nullptr)
    (void)hipFree(b->weight_fp16);
  if (b->activation_fp16 != nullptr)
    (void)hipFree(b->activation_fp16);
  if (b->input_tensor_scale != nullptr)
    (void)hipFree(b->input_tensor_scale);
  if (b->weight_tensor_scale != nullptr)
    (void)hipFree(b->weight_tensor_scale);
  if (b->weight_scales != nullptr)
    (void)hipFree(b->weight_scales);
  if (b->weight != nullptr)
    (void)hipFree(b->weight);
  if (b->activation_scales != nullptr)
    (void)hipFree(b->activation_scales);
  if (b->activation != nullptr)
    (void)hipFree(b->activation);
  *b = {};
}

bool make_buffers(const uint64_t m, const uint64_t k, const uint64_t n,
                  Buffers *const b) {
  if (b == nullptr || m == 0U || k == 0U || n == 0U || (k % 16U) != 0U ||
      m > SIZE_MAX / k || n > SIZE_MAX / k)
    return false;
  const uint64_t blocks = k / 16U;
  const size_t packed_a = static_cast<size_t>(m * k / 2U);
  const size_t packed_w = static_cast<size_t>(n * k / 2U);
  const size_t scales_a = static_cast<size_t>(m * blocks);
  const size_t scales_w = static_cast<size_t>(n * blocks);
  const size_t fp16_a = static_cast<size_t>(m * k * sizeof(uint16_t));
  const size_t fp16_w = static_cast<size_t>(n * k * sizeof(uint16_t));
  const size_t output_f32 = static_cast<size_t>(m * n * sizeof(float));
  const size_t output_bf16 = static_cast<size_t>(m * n * sizeof(uint16_t));
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
             hipMalloc(reinterpret_cast<void **>(&b->activation_fp16), fp16_a),
             "hipMalloc activation fp16") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->weight_fp16), fp16_w),
                "hipMalloc weight fp16") &&
         hip_ok(
             hipMalloc(reinterpret_cast<void **>(&b->gemm_output), output_f32),
             "hipMalloc gemm output") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->output), output_bf16),
                "hipMalloc output") &&
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
         rocblas_ok(
             rocblas_set_atomics_mode(b->rocblas, rocblas_atomics_not_allowed),
             "rocblas_set_atomics_mode") &&
         rocblas_ok(rocblas_set_stream(b->rocblas, b->stream),
                    "rocblas_set_stream");
}

uint32_t mix32(uint32_t value) {
  value ^= value >> 16U;
  value *= UINT32_C(0x7feb352d);
  value ^= value >> 15U;
  value *= UINT32_C(0x846ca68b);
  return value ^ (value >> 16U);
}

uint8_t positive_finite_e4m3(const uint64_t index, const uint32_t seed) {
  // 0x00..0x7e are every non-negative finite E4M3FN code.  73 is
  // coprime with 127, so each consecutive 127-element window is a complete
  // permutation; the seed only rotates the corpus.
  return static_cast<uint8_t>((index * UINT64_C(73) + seed) % UINT64_C(127));
}

void fill_inputs(const uint64_t m, const uint64_t k, const uint64_t n,
                 const uint32_t seed, std::vector<uint8_t> *const activation,
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
      const uint32_t ordinal = static_cast<uint32_t>(row * k + inner);
      const uint8_t code =
          static_cast<uint8_t>(mix32(ordinal ^ seed) & UINT32_C(0x0f));
      const size_t index = static_cast<size_t>(row * k / 2U + inner / 2U);
      if ((inner & 1U) == 0U)
        (*activation)[index] = code;
      else
        (*activation)[index] |= static_cast<uint8_t>(code << 4U);
    }
    for (uint64_t block = 0U; block < blocks; ++block) {
      (*activation_scales)[static_cast<size_t>(row * blocks + block)] =
          positive_finite_e4m3(row * blocks + block, seed);
    }
  }
  for (uint64_t column = 0U; column < n; ++column) {
    for (uint64_t inner = 0U; inner < k; ++inner) {
      const uint32_t ordinal = static_cast<uint32_t>(column * k + inner);
      const uint8_t code = static_cast<uint8_t>(
          mix32(ordinal ^ seed ^ UINT32_C(0x9e3779b9)) & UINT32_C(0x0f));
      const size_t index = static_cast<size_t>(column * k / 2U + inner / 2U);
      if ((inner & 1U) == 0U)
        (*weight)[index] = code;
      else
        (*weight)[index] |= static_cast<uint8_t>(code << 4U);
    }
    for (uint64_t block = 0U; block < blocks; ++block) {
      (*weight_scales)[static_cast<size_t>(column * blocks + block)] =
          positive_finite_e4m3(column * blocks + block,
                               seed ^ UINT32_C(0xa5a5a5a5));
    }
  }
}

bool upload(const uint64_t m, const uint64_t k, const uint64_t n,
            const std::vector<uint8_t> &activation,
            const std::vector<uint8_t> &activation_scales,
            const std::vector<uint8_t> &weight,
            const std::vector<uint8_t> &weight_scales, Buffers *const b) {
  const float weight_tensor_scale = 0.75F;
  const float input_tensor_scale = 1.125F;
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
  hipLaunchKernelGGL(id64_nvfp4_wmma128x64_kernel,
                     dim3(static_cast<uint32_t>((n + 63U) / 64U),
                          static_cast<uint32_t>((m + 127U) / 128U)),
                     dim3(kThreads), 0U, b->stream, b->activation,
                     b->activation_scales, b->weight, b->weight_scales,
                     b->weight_tensor_scale, b->input_tensor_scale, b->output,
                     m, k, n);
  return hipGetLastError() == hipSuccess;
}

bool launch_staging(const uint64_t m, const uint64_t k, const uint64_t n,
                    const int32_t solution, const bool block16_ingress,
                    Buffers *const b) {
  const uint64_t a_elements = m * k;
  const uint64_t w_elements = n * k;
  if (block16_ingress) {
    const uint64_t blocks_a = m * (k / 16U);
    const uint64_t blocks_w = n * (k / 16U);
    hipLaunchKernelGGL(
        nvfp4_to_fp16_block16_kernel,
        dim3(static_cast<uint32_t>((blocks_a + kThreads - 1U) / kThreads)),
        dim3(kThreads), 0U, b->stream, b->activation, b->activation_scales,
        b->activation_fp16, m, k);
    hipLaunchKernelGGL(
        nvfp4_to_fp16_block16_kernel,
        dim3(static_cast<uint32_t>((blocks_w + kThreads - 1U) / kThreads)),
        dim3(kThreads), 0U, b->stream, b->weight, b->weight_scales,
        b->weight_fp16, n, k);
  } else {
    hipLaunchKernelGGL(
        nvfp4_to_fp16_kernel,
        dim3(static_cast<uint32_t>((a_elements + kThreads - 1U) / kThreads)),
        dim3(kThreads), 0U, b->stream, b->activation, b->activation_scales,
        b->activation_fp16, m, k);
    hipLaunchKernelGGL(
        nvfp4_to_fp16_kernel,
        dim3(static_cast<uint32_t>((w_elements + kThreads - 1U) / kThreads)),
        dim3(kThreads), 0U, b->stream, b->weight, b->weight_scales,
        b->weight_fp16, n, k);
  }
  const float alpha = 1.0F;
  const float beta = 0.0F;
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
          "rocblas_gemm_ex")) {
    return false;
  }
  const uint64_t elements = m * n;
  hipLaunchKernelGGL(
      bf16_epilogue_kernel,
      dim3(static_cast<uint32_t>((elements + kThreads - 1U) / kThreads)),
      dim3(kThreads), 0U, b->stream, b->gemm_output, b->output, elements,
      0.75F * 1.125F);
  return hipGetLastError() == hipSuccess;
}

bool measure_control(const uint64_t m, const uint64_t k, const uint64_t n,
                     Buffers *const b, float *const median_us) {
  for (uint32_t i = 0U; i < kWarmups; ++i) {
    if (!launch_control(m, k, n, b) ||
        !hip_ok(hipStreamSynchronize(b->stream), "control warmup"))
      return false;
  }
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
                     const int32_t solution, const bool block16_ingress,
                     Buffers *const b, float *const median_us) {
  for (uint32_t i = 0U; i < kWarmups; ++i) {
    if (!launch_staging(m, k, n, solution, block16_ingress, b) ||
        !hip_ok(hipStreamSynchronize(b->stream), "staging warmup"))
      return false;
  }
  std::array<float, kMeasured> samples{};
  for (uint32_t i = 0U; i < kMeasured; ++i) {
    if (!hip_ok(hipEventRecord(b->start, b->stream), "staging event start") ||
        !launch_staging(m, k, n, solution, block16_ingress, b) ||
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

uint32_t ordered_bf16(const uint16_t bits) {
  if ((bits & UINT16_C(0x7fff)) == 0U) {
    return UINT32_C(0x8000);
  }
  return (bits & UINT16_C(0x8000)) != 0U
             ? static_cast<uint16_t>(~bits)
             : static_cast<uint16_t>(bits | UINT16_C(0x8000));
}

bool compare_outputs(const std::vector<uint16_t> &control,
                     const std::vector<uint16_t> &candidate,
                     const char *const name, const uint64_t m,
                     const uint64_t n) {
  std::array<size_t, 6> ulp_histogram{};
  uint32_t max_ulp = 0U;
  double max_abs = 0.0;
  double max_rel = 0.0;
  for (size_t i = 0U; i < control.size(); ++i) {
    const float lhs = host_bf16_to_float(control[i]);
    const float rhs = host_bf16_to_float(candidate[i]);
    const double abs = std::abs(static_cast<double>(lhs) - rhs);
    const double rel =
        abs / std::max(1.0e-6, std::abs(static_cast<double>(lhs)));
    max_abs = std::max(max_abs, abs);
    max_rel = std::max(max_rel, rel);
    const uint32_t lhs_ordered = ordered_bf16(control[i]);
    const uint32_t rhs_ordered = ordered_bf16(candidate[i]);
    const uint32_t ulp = lhs_ordered > rhs_ordered ? lhs_ordered - rhs_ordered
                                                   : rhs_ordered - lhs_ordered;
    max_ulp = std::max(max_ulp, ulp);
    const size_t bucket = ulp == 0U   ? 0U
                          : ulp == 1U ? 1U
                          : ulp == 2U ? 2U
                          : ulp <= 4U ? 3U
                          : ulp <= 8U ? 4U
                                      : 5U;
    ++ulp_histogram[bucket];
  }
  std::printf("compare candidate=%s m=%llu n=%llu max_abs=%.8g max_rel=%.8g "
              "max_bf16_ulp=%u "
              "ulp_histogram=0:%zu,1:%zu,2:%zu,3-4:%zu,5-8:%zu,>8:%zu\n",
              name, static_cast<unsigned long long>(m),
              static_cast<unsigned long long>(n), max_abs, max_rel, max_ulp,
              ulp_histogram[0], ulp_histogram[1], ulp_histogram[2],
              ulp_histogram[3], ulp_histogram[4], ulp_histogram[5]);
  return true;
}

bool run_numerical_oracle(const uint64_t m, const uint64_t k, const uint64_t n,
                          const uint32_t seed) {
  const uint64_t blocks_per_row = k / UINT64_C(16);
  std::vector<uint8_t> activation;
  std::vector<uint8_t> activation_scales;
  std::vector<uint8_t> weight;
  std::vector<uint8_t> weight_scales;
  fill_inputs(m, k, n, seed, &activation, &activation_scales, &weight,
              &weight_scales);

  std::array<bool, 127> scale_codes_seen{};
  for (const uint8_t scale : activation_scales) {
    scale_codes_seen[scale] = true;
  }
  for (const uint8_t scale : weight_scales) {
    scale_codes_seen[scale] = true;
  }
  const bool complete_scale_corpus =
      std::all_of(scale_codes_seen.begin(), scale_codes_seen.end(),
                  [](const bool seen) { return seen; });

  Buffers buffers;
  if (!make_buffers(m, k, n, &buffers) ||
      !upload(m, k, n, activation, activation_scales, weight, weight_scales,
              &buffers)) {
    cleanup(&buffers);
    return false;
  }
  const size_t output_elements = static_cast<size_t>(m * n);
  std::vector<uint16_t> control(output_elements);
  std::vector<uint16_t> staging_scalar(output_elements);
  std::vector<uint16_t> staging(output_elements);
  std::vector<uint16_t> staging_repeat(output_elements);
  if (!launch_control(m, k, n, &buffers) ||
      !hip_ok(hipStreamSynchronize(buffers.stream), "oracle control sync") ||
      !hip_ok(hipMemcpy(control.data(), buffers.output,
                        control.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              "oracle control copy") ||
      !launch_staging(m, k, n, 0, false, &buffers) ||
      !hip_ok(hipStreamSynchronize(buffers.stream),
              "oracle scalar staging sync") ||
      !hip_ok(hipMemcpy(staging_scalar.data(), buffers.output,
                        staging_scalar.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              "oracle scalar staging copy") ||
      !launch_staging(m, k, n, 0, true, &buffers) ||
      !hip_ok(hipStreamSynchronize(buffers.stream),
              "oracle block16 staging sync") ||
      !hip_ok(hipMemcpy(staging.data(), buffers.output,
                        staging.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              "oracle block16 staging copy")) {
    cleanup(&buffers);
    return false;
  }

  bool deterministic = true;
  for (uint32_t repetition = 1U; repetition < 8U && deterministic;
       ++repetition) {
    deterministic = launch_staging(m, k, n, 0, true, &buffers) &&
                    hip_ok(hipStreamSynchronize(buffers.stream),
                           "oracle repeat staging sync") &&
                    hip_ok(hipMemcpy(staging_repeat.data(), buffers.output,
                                     staging_repeat.size() * sizeof(uint16_t),
                                     hipMemcpyDeviceToHost),
                           "oracle repeat staging copy") &&
                    staging_repeat == staging;
  }
  const bool ingress_exact = staging_scalar == staging;

  std::vector<uint16_t> semantic(output_elements);
  double max_normalized_error = 0.0;
  for (uint64_t row = 0U; row < m; ++row) {
    for (uint64_t column = 0U; column < n; ++column) {
      double expected = 0.0;
      double absolute_sum = 0.0;
      for (uint64_t inner = 0U; inner < k; ++inner) {
        const uint8_t activation_pair = activation[static_cast<size_t>(
            row * k / UINT64_C(2) + inner / UINT64_C(2))];
        const uint8_t weight_pair = weight[static_cast<size_t>(
            column * k / UINT64_C(2) + inner / UINT64_C(2))];
        const uint8_t activation_code = (inner & UINT64_C(1)) == 0U
                                            ? activation_pair & UINT8_C(0x0f)
                                            : activation_pair >> 4U;
        const uint8_t weight_code = (inner & UINT64_C(1)) == 0U
                                        ? weight_pair & UINT8_C(0x0f)
                                        : weight_pair >> 4U;
        const double term =
            static_cast<double>(host_e2m1(activation_code)) *
            host_e4m3(activation_scales[static_cast<size_t>(
                row * blocks_per_row + inner / UINT64_C(16))]) *
            static_cast<double>(host_e2m1(weight_code)) *
            host_e4m3(weight_scales[static_cast<size_t>(
                column * blocks_per_row + inner / UINT64_C(16))]);
        expected += term;
        absolute_sum += std::abs(term);
      }
      constexpr double tensor_scale = 0.75 * 1.125;
      expected *= tensor_scale;
      absolute_sum *= tensor_scale;
      const size_t output_index = static_cast<size_t>(row * n + column);
      semantic[output_index] = host_bf16_rne(static_cast<float>(expected));
      const double observed =
          static_cast<double>(host_bf16_to_float(staging[output_index]));
      const double normalized_error =
          std::abs(observed - expected) / std::max(absolute_sum, 0x1p-100);
      max_normalized_error = std::max(max_normalized_error, normalized_error);
    }
  }

  char comparison_name[96]{};
  std::snprintf(comparison_name, sizeof(comparison_name),
                "staging-vs-real-semantic-seed-%08x", seed);
  compare_outputs(semantic, staging, comparison_name, m, n);
  std::snprintf(comparison_name, sizeof(comparison_name),
                "staging-vs-ID64-seed-%08x", seed);
  compare_outputs(control, staging, comparison_name, m, n);
  std::printf("oracle-diverse m=%llu k=%llu n=%llu seed=%08x "
              "positive_finite_scale_corpus=%s scalar_block16_bitwise=%s "
              "repeat8_bitwise=%s max_normalized_error=%.8g status=%s\n",
              static_cast<unsigned long long>(m),
              static_cast<unsigned long long>(k),
              static_cast<unsigned long long>(n), seed,
              complete_scale_corpus ? "complete" : "incomplete",
              ingress_exact ? "PASS" : "FAIL", deterministic ? "PASS" : "FAIL",
              max_normalized_error,
              complete_scale_corpus && ingress_exact && deterministic &&
                      max_normalized_error <= 0.01
                  ? "PASS"
                  : "FAIL");
  cleanup(&buffers);
  return complete_scale_corpus && ingress_exact && deterministic &&
         max_normalized_error <= 0.01;
}

bool print_solution_list(rocblas_handle handle, const uint64_t m,
                         const uint64_t k, const uint64_t n, const Buffers &b) {
  const float alpha = 1.0F;
  const float beta = 0.0F;
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
  const size_t shown = std::min<size_t>(solutions.size(), 16U);
  for (size_t i = 0U; i < shown; ++i)
    std::printf("%s%d", i == 0U ? "" : ",", solutions[i]);
  std::printf("\n");
  return true;
}

void print_resources(const char *const name, const void *const function,
                     const size_t dynamic_shared_bytes = 0U) {
  hipFuncAttributes attributes{};
  const hipError_t attr_status = hipFuncGetAttributes(&attributes, function);
  int active_blocks = 0;
  const hipError_t occupancy_status =
      hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks, function, kThreads, dynamic_shared_bytes);
  std::printf("resources candidate=%s vgpr=%d sgpr=runtime lds_static=%zu "
              "lds_dynamic=%zu scratch=%zu max_threads=%d active_blocks=%d "
              "attr=%s occupancy=%s\n",
              name, attributes.numRegs, attributes.sharedSizeBytes,
              dynamic_shared_bytes, attributes.localSizeBytes,
              attributes.maxThreadsPerBlock, active_blocks,
              hipGetErrorString(attr_status),
              hipGetErrorString(occupancy_status));
}

} // namespace

int main() {
  constexpr int device = 0;
  if (!hip_ok(hipSetDevice(device), "hipSetDevice"))
    return EXIT_FAILURE;
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device),
              "hipGetDeviceProperties") ||
      !exact_gfx1201(properties.gcnArchName)) {
    std::fprintf(stderr, "exact gfx1201 required; got %s\n",
                 properties.gcnArchName);
    return EXIT_FAILURE;
  }
  std::printf("target=%s device=%d pci=%04x:%02x:%02x name=%s\n",
              properties.gcnArchName, device, properties.pciDomainID,
              properties.pciBusID, properties.pciDeviceID, properties.name);
  print_resources("id64-wmma128x64",
                  reinterpret_cast<const void *>(id64_nvfp4_wmma128x64_kernel));
  print_resources("nvfp4-ingress-scalar",
                  reinterpret_cast<const void *>(nvfp4_to_fp16_kernel));
  print_resources("nvfp4-ingress-block16",
                  reinterpret_cast<const void *>(nvfp4_to_fp16_block16_kernel));
  print_resources("bf16-epilogue",
                  reinterpret_cast<const void *>(bf16_epilogue_kernel));

  // The selector boundary is M=128.  B-1/B/B+1 also retain a non-aligned N
  // tail, while K=48 spans three independent block16 scale domains.  Across
  // each case the activation corpus alone covers every positive finite
  // E4M3FN scale code; three fixed seeds vary both scale ordering and nibbles.
  constexpr std::array<uint64_t, 3> oracle_rows = {127U, 128U, 129U};
  bool all_ok = true;
  for (size_t index = 0U; index < oracle_rows.size(); ++index) {
    all_ok = run_numerical_oracle(oracle_rows[index], 48U, 65U,
                                  kNumericalSeeds[index]) &&
             all_ok;
  }

  const int32_t solution = [] {
    const char *const text = std::getenv("SLLM_PHASE78_F16_STAGING_SOLUTION");
    return text == nullptr
               ? 0
               : static_cast<int32_t>(std::strtol(text, nullptr, 10));
  }();
  const std::array<std::array<uint64_t, 3>, 4> shapes = {
      std::array<uint64_t, 3>{128U, 5120U, 17408U},
      std::array<uint64_t, 3>{128U, 17408U, 5120U},
      std::array<uint64_t, 3>{512U, 5120U, 17408U},
      std::array<uint64_t, 3>{512U, 17408U, 5120U}};
  // M=1024/2048 are enabled with SLLM_PHASE78_F16_STAGING_LONG=1 because
  // they allocate large F32 output scratch and are measured after the short
  // operator cases have identified a usable rocBLAS solution.
  const bool long_run = std::getenv("SLLM_PHASE78_F16_STAGING_LONG") != nullptr;
  const bool long_compare =
      std::getenv("SLLM_PHASE78_F16_STAGING_LONG_COMPARE") != nullptr;
  for (const auto &shape : shapes) {
    const uint64_t m = shape[0], k = shape[1], n = shape[2];
    std::vector<uint8_t> activation;
    std::vector<uint8_t> activation_scales;
    std::vector<uint8_t> weight;
    std::vector<uint8_t> weight_scales;
    fill_inputs(m, k, n, kNumericalSeeds[0], &activation, &activation_scales,
                &weight, &weight_scales);
    Buffers buffers;
    if (!make_buffers(m, k, n, &buffers) ||
        !upload(m, k, n, activation, activation_scales, weight, weight_scales,
                &buffers)) {
      cleanup(&buffers);
      return EXIT_FAILURE;
    }
    if (!launch_staging(m, k, n, solution, true, &buffers) ||
        !hip_ok(hipStreamSynchronize(buffers.stream), "solution preparation")) {
      cleanup(&buffers);
      return EXIT_FAILURE;
    }
    (void)print_solution_list(buffers.rocblas, m, k, n, buffers);
    float control_us = 0.0F;
    float staging_us = 0.0F;
    float staging_scalar_us = 0.0F;
    if (!measure_control(m, k, n, &buffers, &control_us) ||
        !measure_staging(m, k, n, solution, false, &buffers,
                         &staging_scalar_us) ||
        !measure_staging(m, k, n, solution, true, &buffers, &staging_us)) {
      cleanup(&buffers);
      return EXIT_FAILURE;
    }
    const double weight_bytes = static_cast<double>(n) * k / 2.0;
    const double pipeline_bytes =
        weight_bytes + static_cast<double>(m) * k / 2.0;
    std::printf(
        "result m=%llu k=%llu n=%llu control_us=%.3f staging_scalar_us=%.3f "
        "staging_block16_us=%.3f staging_gbps=%.6f workspace_bytes=%llu "
        "solution=%d\n",
        static_cast<unsigned long long>(m), static_cast<unsigned long long>(k),
        static_cast<unsigned long long>(n), control_us, staging_scalar_us,
        staging_us, pipeline_bytes / staging_us / 1000.0,
        static_cast<unsigned long long>(m * k * sizeof(uint16_t) +
                                        n * k * sizeof(uint16_t) +
                                        m * n * sizeof(float)),
        solution);
    std::vector<uint16_t> control_output(static_cast<size_t>(m * n));
    std::vector<uint16_t> staging_output(static_cast<size_t>(m * n));
    if (!launch_control(m, k, n, &buffers) ||
        !hip_ok(hipDeviceSynchronize(), "control compare sync") ||
        !hip_ok(hipMemcpy(control_output.data(), buffers.output,
                          control_output.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "control compare copy") ||
        !launch_staging(m, k, n, solution, true, &buffers) ||
        !hip_ok(hipDeviceSynchronize(), "staging compare sync") ||
        !hip_ok(hipMemcpy(staging_output.data(), buffers.output,
                          staging_output.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "staging compare copy")) {
      cleanup(&buffers);
      return EXIT_FAILURE;
    }
    compare_outputs(control_output, staging_output, "staging-vs-ID64", m, n);
    cleanup(&buffers);
  }
  if (long_run) {
    const std::array<std::array<uint64_t, 3>, 4> long_shapes = {
        std::array<uint64_t, 3>{1024U, 5120U, 17408U},
        std::array<uint64_t, 3>{1024U, 17408U, 5120U},
        std::array<uint64_t, 3>{2048U, 5120U, 17408U},
        std::array<uint64_t, 3>{2048U, 17408U, 5120U}};
    for (const auto &shape : long_shapes) {
      const uint64_t m = shape[0], k = shape[1], n = shape[2];
      std::vector<uint8_t> activation;
      std::vector<uint8_t> activation_scales;
      std::vector<uint8_t> weight;
      std::vector<uint8_t> weight_scales;
      fill_inputs(m, k, n, kNumericalSeeds[1], &activation, &activation_scales,
                  &weight, &weight_scales);
      Buffers buffers;
      if (!make_buffers(m, k, n, &buffers) ||
          !upload(m, k, n, activation, activation_scales, weight, weight_scales,
                  &buffers)) {
        cleanup(&buffers);
        return EXIT_FAILURE;
      }
      float control_us = 0.0F, staging_us = 0.0F;
      all_ok =
          measure_control(m, k, n, &buffers, &control_us) &&
          measure_staging(m, k, n, solution, true, &buffers, &staging_us) &&
          all_ok;
      std::printf(
          "result-long m=%llu k=%llu n=%llu control_us=%.3f staging_us=%.3f "
          "workspace_bytes=%llu solution=%d\n",
          static_cast<unsigned long long>(m),
          static_cast<unsigned long long>(k),
          static_cast<unsigned long long>(n), control_us, staging_us,
          static_cast<unsigned long long>(m * k * sizeof(uint16_t) +
                                          n * k * sizeof(uint16_t) +
                                          m * n * sizeof(float)),
          solution);
      if (long_compare) {
        std::vector<uint16_t> control_output(static_cast<size_t>(m * n));
        std::vector<uint16_t> staging_output(static_cast<size_t>(m * n));
        if (!launch_control(m, k, n, &buffers) ||
            !hip_ok(hipDeviceSynchronize(), "long control compare sync") ||
            !hip_ok(hipMemcpy(control_output.data(), buffers.output,
                              control_output.size() * sizeof(uint16_t),
                              hipMemcpyDeviceToHost),
                    "long control compare copy") ||
            !launch_staging(m, k, n, solution, true, &buffers) ||
            !hip_ok(hipDeviceSynchronize(), "long staging compare sync") ||
            !hip_ok(hipMemcpy(staging_output.data(), buffers.output,
                              staging_output.size() * sizeof(uint16_t),
                              hipMemcpyDeviceToHost),
                    "long staging compare copy")) {
          cleanup(&buffers);
          return EXIT_FAILURE;
        }
        compare_outputs(control_output, staging_output, "staging-vs-ID64-long",
                        m, n);
      }
      cleanup(&buffers);
    }
  }
  std::printf("summary status=%s warmups=%u measured=%u\n",
              all_ok ? "PASS" : "FAIL", kWarmups, kMeasured);
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
