// Phase 78 standalone gfx1030 short-prefill FP8 TileM32 probe.
//
// The control is the current production ID71 64x64/K32 kernel loaded from the
// archive linked below.  The candidate uses one 32x64 output tile with the
// same K32 staging, half2 accumulation order, scales, and BF16-RNE epilogue.
// It is intended for short M prefill shapes and remains a standalone probe;
// it does not change production dispatch.
//
// This file is intentionally a standalone probe and is not linked into the
// public runtime.

#include "low_precision_block_codec.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

// Direct archive linkage keeps the control tied to the current production
// ID71 object instead of silently comparing two standalone source copies.
extern "C" __global__ void sllm_matmul_fp8_outer_prefill_gfx1030_half2_64x64_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n);

namespace {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kThreadRows = 16U;
constexpr uint32_t kThreadColumns = 16U;
constexpr uint32_t kTileK = 32U;
constexpr int kWarmups = 3;
constexpr int kMeasured = 10;

__host__ __device__ constexpr uint8_t activation_code(const uint64_t row,
                                                      const uint64_t inner) {
  const uint32_t hash =
      static_cast<uint32_t>(row * UINT64_C(37) + inner * UINT64_C(11) + 5U);
  const uint8_t magnitude =
      static_cast<uint8_t>(UINT32_C(0x18) + (hash & UINT32_C(0x1f)));
  return static_cast<uint8_t>(magnitude | ((hash & UINT32_C(0x20)) << 2U));
}

__host__ __device__ constexpr uint8_t weight_code(const uint64_t column,
                                                  const uint64_t inner) {
  const uint32_t hash =
      static_cast<uint32_t>(column * UINT64_C(19) + inner * UINT64_C(7) + 13U);
  const uint8_t magnitude =
      static_cast<uint8_t>(UINT32_C(0x18) + (hash & UINT32_C(0x1f)));
  return static_cast<uint8_t>(magnitude | ((hash & UINT32_C(0x20)) << 2U));
}

__host__ __device__ constexpr float activation_scale(const uint64_t row) {
  return (row % 3U) == 0U ? 0.5F : (row % 3U) == 1U ? 1.0F : 2.0F;
}

__host__ __device__ constexpr float weight_scale(const uint64_t column) {
  return (column % 3U) == 0U ? 2.0F : (column % 3U) == 1U ? 1.0F : 0.5F;
}

__device__ __forceinline__ uint16_t to_bf16_rne(const float value) noexcept {
  const uint32_t bits = __float_as_uint(value);
  constexpr uint32_t exponent_mask = UINT32_C(0x7f800000);
  constexpr uint32_t fraction_mask = UINT32_C(0x007fffff);
  if ((bits & exponent_mask) == exponent_mask) {
    if ((bits & fraction_mask) != 0U) {
      const uint16_t sign = static_cast<uint16_t>((bits >> 16U) & 0x8000U);
      const uint16_t payload = static_cast<uint16_t>((bits >> 16U) & 0x003fU);
      return static_cast<uint16_t>(sign | UINT16_C(0x7fc0) | payload);
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

template <uint32_t TileM, uint32_t TileN, uint32_t LdsStride>
__device__ __forceinline__ void
stage_outer_k32(const uint8_t *const activation, const uint8_t *const weight,
                const uint64_t m, const uint64_t k, const uint64_t n,
                const uint64_t row_base, const uint64_t column_base,
                const uint64_t inner_base, uint16_t *const activation_tile,
                uint16_t *const weight_tile) {
  constexpr uint32_t values_per_load = 4U;
  constexpr uint32_t loads_per_row = kTileK / values_per_load;
  static_assert(kTileK % values_per_load == 0U);
  const uint32_t thread = threadIdx.x;

  for (uint32_t index = thread; index < TileM * loads_per_row;
       index += kThreads) {
    const uint32_t row = index / loads_per_row;
    const uint32_t inner = (index % loads_per_row) * values_per_load;
    const uint64_t source_row = row_base + row;
    const uint64_t source_inner = inner_base + inner;
    uint16_t *const destination =
        activation_tile + static_cast<size_t>(row) * LdsStride + inner;
    if ((k % values_per_load) == 0U && source_row < m &&
        source_inner + values_per_load <= k) {
      const uint32_t packed =
          __builtin_nontemporal_load(reinterpret_cast<const uint32_t *>(
              activation + source_row * k + source_inner));
      const sllm_lowp::E4M3FnFp16x4Bits expanded =
          sllm_lowp::e4m3fnx4_to_fp16x2_bits(packed);
      auto *const packed_destination =
          reinterpret_cast<uint32_t *>(destination);
      packed_destination[0] = expanded.low;
      packed_destination[1] = expanded.high;
    } else {
#pragma unroll
      for (uint32_t lane = 0U; lane < values_per_load; ++lane) {
        destination[lane] =
            source_row < m && source_inner + lane < k
                ? sllm_lowp::e4m3fn_to_fp16_bits(__builtin_nontemporal_load(
                      activation + source_row * k + source_inner + lane))
                : UINT16_C(0);
      }
    }
  }

  for (uint32_t index = thread; index < TileN * loads_per_row;
       index += kThreads) {
    const uint32_t column = index / loads_per_row;
    const uint32_t inner = (index % loads_per_row) * values_per_load;
    const uint64_t source_column = column_base + column;
    const uint64_t source_inner = inner_base + inner;
    uint16_t *const destination =
        weight_tile + static_cast<size_t>(column) * LdsStride + inner;
    if ((k % values_per_load) == 0U && source_column < n &&
        source_inner + values_per_load <= k) {
      const uint32_t packed =
          __builtin_nontemporal_load(reinterpret_cast<const uint32_t *>(
              weight + source_column * k + source_inner));
      const sllm_lowp::E4M3FnFp16x4Bits expanded =
          sllm_lowp::e4m3fnx4_to_fp16x2_bits(packed);
      auto *const packed_destination =
          reinterpret_cast<uint32_t *>(destination);
      packed_destination[0] = expanded.low;
      packed_destination[1] = expanded.high;
    } else {
#pragma unroll
      for (uint32_t lane = 0U; lane < values_per_load; ++lane) {
        destination[lane] =
            source_column < n && source_inner + lane < k
                ? sllm_lowp::e4m3fn_to_fp16_bits(__builtin_nontemporal_load(
                      weight + source_column * k + source_inner + lane))
                : UINT16_C(0);
      }
    }
  }
}

// ID71-equivalent control: one padded K32 buffer and a 64x64 output tile.
template <uint32_t TileM, uint32_t TileN>
__global__ __launch_bounds__(kThreads, 1) void fp8_outer_single_kernel(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  static_assert(TileM % kThreadRows == 0U);
  static_assert(TileN % kThreadColumns == 0U);
  constexpr uint32_t rows_per_thread = TileM / kThreadRows;
  constexpr uint32_t columns_per_thread = TileN / kThreadColumns;
  constexpr uint32_t lds_stride = kTileK + 2U;
  __shared__ __align__(4) uint16_t activation_tile[TileM][lds_stride];
  __shared__ __align__(4) uint16_t weight_tile[TileN][lds_stride];

  const uint64_t column_tiles = (n + TileN - 1U) / TileN;
  const uint64_t tile_index = static_cast<uint64_t>(blockIdx.x);
  const uint64_t row_base = (tile_index / column_tiles) * TileM;
  const uint64_t column_base = (tile_index % column_tiles) * TileN;
  const uint32_t local_row = threadIdx.x >> 4U;
  const uint32_t local_column = threadIdx.x & UINT32_C(15);
  float accumulators[rows_per_thread][columns_per_thread] = {};

  for (uint64_t base = 0U; base < k; base += kTileK) {
    stage_outer_k32<TileM, TileN, lds_stride>(
        activation, weight, m, k, n, row_base, column_base, base,
        &activation_tile[0][0], &weight_tile[0][0]);
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
        output[output_row * n + output_column] = to_bf16_rne(
            accumulators[row][column] * activation_scales[output_row] *
            weight_scales[output_column]);
      }
    }
  }
}

// ID55-style schedule: preload buffer zero, stage the next K32 tile into the
// disjoint buffer while computing the current tile, synchronize once, toggle.
template <uint32_t TileM, uint32_t TileN>
__global__ __launch_bounds__(kThreads, 1) void fp8_outer_double_kernel(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  static_assert(TileM % kThreadRows == 0U);
  static_assert(TileN % kThreadColumns == 0U);
  constexpr uint32_t rows_per_thread = TileM / kThreadRows;
  constexpr uint32_t columns_per_thread = TileN / kThreadColumns;
  constexpr uint32_t lds_stride = kTileK;
  __shared__ __align__(4) uint16_t activation_tiles[2][TileM][lds_stride];
  __shared__ __align__(4) uint16_t weight_tiles[2][TileN][lds_stride];

  const uint64_t column_tiles = (n + TileN - 1U) / TileN;
  const uint64_t tile_index = static_cast<uint64_t>(blockIdx.x);
  const uint64_t row_base = (tile_index / column_tiles) * TileM;
  const uint64_t column_base = (tile_index % column_tiles) * TileN;
  const uint32_t local_row = threadIdx.x >> 4U;
  const uint32_t local_column = threadIdx.x & UINT32_C(15);
  float accumulators[rows_per_thread][columns_per_thread] = {};

  stage_outer_k32<TileM, TileN, lds_stride>(
      activation, weight, m, k, n, row_base, column_base, 0U,
      &activation_tiles[0][0][0], &weight_tiles[0][0][0]);
  __syncthreads();

  uint32_t current_buffer = 0U;
  for (uint64_t base = 0U; base < k; base += kTileK) {
    const uint64_t next_base = base + kTileK;
    const uint32_t next_buffer = current_buffer ^ 1U;
    if (next_base < k) {
      stage_outer_k32<TileM, TileN, lds_stride>(
          activation, weight, m, k, n, row_base, column_base, next_base,
          &activation_tiles[next_buffer][0][0],
          &weight_tiles[next_buffer][0][0]);
    }

#pragma unroll
    for (uint32_t inner = 0U; inner < kTileK; inner += 2U) {
      __half2 activation_pairs[rows_per_thread];
      __half2 weight_pairs[columns_per_thread];
#pragma unroll
      for (uint32_t row = 0U; row < rows_per_thread; ++row) {
        activation_pairs[row] = *reinterpret_cast<const __half2 *>(
            &activation_tiles[current_buffer][local_row + row * kThreadRows]
                             [inner]);
      }
#pragma unroll
      for (uint32_t column = 0U; column < columns_per_thread; ++column) {
        weight_pairs[column] = *reinterpret_cast<const __half2 *>(
            &weight_tiles[current_buffer]
                         [local_column + column * kThreadColumns][inner]);
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
    current_buffer = next_buffer;
  }

#pragma unroll
  for (uint32_t row = 0U; row < rows_per_thread; ++row) {
    const uint64_t output_row = row_base + local_row + row * kThreadRows;
#pragma unroll
    for (uint32_t column = 0U; column < columns_per_thread; ++column) {
      const uint64_t output_column =
          column_base + local_column + column * kThreadColumns;
      if (output_row < m && output_column < n) {
        output[output_row * n + output_column] = to_bf16_rne(
            accumulators[row][column] * activation_scales[output_row] *
            weight_scales[output_column]);
      }
    }
  }
}

__global__ void fill_activation_kernel(uint8_t *const activation,
                                       const uint64_t m, const uint64_t k) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < m * k) {
    activation[index] = activation_code(index / k, index % k);
  }
}

__global__ void fill_weight_kernel(uint8_t *const weight, const uint64_t k,
                                   const uint64_t n) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < n * k) {
    weight[index] = weight_code(index / k, index % k);
  }
}

__global__ void fill_scale_kernel(float *const activation_scales,
                                  float *const weight_scales, const uint64_t m,
                                  const uint64_t n) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < m) {
    activation_scales[index] = activation_scale(index);
  }
  if (index < n) {
    weight_scales[index] = weight_scale(index);
  }
}

__global__ void e4m3x4_probe_kernel(const uint32_t *const packed_input,
                                    uint16_t *const output) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 256U) {
    const sllm_lowp::E4M3FnFp16x4Bits expanded =
        sllm_lowp::e4m3fnx4_to_fp16x2_bits(packed_input[index]);
    auto *const packed_output =
        reinterpret_cast<uint32_t *>(output + index * 4U);
    packed_output[0] = expanded.low;
    packed_output[1] = expanded.high;
  }
}

enum class CandidateId : uint32_t { ArchiveId71, TileM32N64, TileM32N32 };

struct Candidate final {
  CandidateId id;
  const char *name;
  uint32_t tile_m;
  uint32_t tile_n;
  const void *function;
};

Candidate candidate(const CandidateId id) {
  switch (id) {
  case CandidateId::ArchiveId71:
    return {id, "id71-archive-control-64x64-k32-single", 64U, 64U,
            reinterpret_cast<const void *>(
                sllm_matmul_fp8_outer_prefill_gfx1030_half2_64x64_v1)};
  case CandidateId::TileM32N64:
    return {id, "candidate-tilem32x64-k32-single", 32U, 64U,
            reinterpret_cast<const void *>(fp8_outer_single_kernel<32U, 64U>)};
  case CandidateId::TileM32N32:
    return {id, "candidate-tilem32x32-k32-single", 32U, 32U,
            reinterpret_cast<const void *>(fp8_outer_single_kernel<32U, 32U>)};
  }
  return {id, "invalid", 0U, 0U, nullptr};
}

struct Shape final {
  uint64_t m;
  uint64_t k;
  uint64_t n;
  const char *name;
};

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::fprintf(stderr, "hip error operation=%s status=%s\n", operation,
               hipGetErrorString(status));
  return false;
}

bool exact_gfx1030(const char *const arch) {
  if (arch == nullptr) {
    return false;
  }
  const std::string_view value(arch);
  constexpr std::string_view prefix = "gfx1030";
  return value == prefix || (value.size() > prefix.size() &&
                             value.compare(0U, prefix.size(), prefix) == 0 &&
                             value[prefix.size()] == ':');
}

bool checked_product(const uint64_t first, const uint64_t second,
                     size_t *const result) {
  if (result == nullptr || first > SIZE_MAX || second > SIZE_MAX ||
      (first != 0U && second > SIZE_MAX / first)) {
    return false;
  }
  *result = static_cast<size_t>(first * second);
  return true;
}

uint16_t host_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t exponent = bits & UINT32_C(0x7f800000);
  const uint32_t fraction = bits & UINT32_C(0x007fffff);
  if (exponent == UINT32_C(0x7f800000)) {
    if (fraction != 0U) {
      return static_cast<uint16_t>((bits >> 16U) | UINT32_C(0x0040));
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

uint16_t host_e4m3_to_fp16_bits_reference(const uint8_t bits) {
  const uint16_t sign = static_cast<uint16_t>(bits & UINT8_C(0x80)) << 8U;
  const uint8_t magnitude = bits & UINT8_C(0x7f);
  const uint8_t exponent = magnitude >> 3U;
  const uint8_t mantissa = magnitude & UINT8_C(0x07);
  if (exponent == 0U) {
    constexpr std::array<uint16_t, 8> subnormals = {
        UINT16_C(0x0000), UINT16_C(0x1800), UINT16_C(0x1c00), UINT16_C(0x1e00),
        UINT16_C(0x2000), UINT16_C(0x2100), UINT16_C(0x2200), UINT16_C(0x2300)};
    return static_cast<uint16_t>(sign | subnormals[mantissa]);
  }
  if (magnitude == UINT8_C(0x7f)) {
    return static_cast<uint16_t>(sign | UINT16_C(0x7e00));
  }
  return static_cast<uint16_t>(
      sign | static_cast<uint16_t>(exponent + UINT8_C(8)) << 10U |
      static_cast<uint16_t>(mantissa) << 7U);
}

float host_e4m3_to_float(const uint8_t bits) {
  const bool negative = (bits & UINT8_C(0x80)) != 0U;
  const uint8_t magnitude = bits & UINT8_C(0x7f);
  const uint8_t exponent = magnitude >> 3U;
  const uint8_t mantissa = magnitude & UINT8_C(0x07);
  float value = 0.0F;
  if (exponent == 0U) {
    value = std::ldexp(static_cast<float>(mantissa), -9);
  } else if (magnitude == UINT8_C(0x7f)) {
    return std::numeric_limits<float>::quiet_NaN();
  } else {
    value = std::ldexp(1.0F + static_cast<float>(mantissa) * 0.125F,
                       static_cast<int>(exponent) - 7);
  }
  return negative ? -value : value;
}

float bf16_to_float(const uint16_t bits) {
  const uint32_t expanded = static_cast<uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &expanded, sizeof(value));
  return value;
}

struct DeviceBuffers final {
  uint8_t *activation = nullptr;
  float *activation_scales = nullptr;
  uint8_t *weight = nullptr;
  float *weight_scales = nullptr;
  uint16_t *output = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  size_t output_elements = 0U;
};

void free_buffers(DeviceBuffers *const buffers) {
  if (buffers == nullptr) {
    return;
  }
  if (buffers->stop != nullptr) {
    (void)hipEventDestroy(buffers->stop);
  }
  if (buffers->start != nullptr) {
    (void)hipEventDestroy(buffers->start);
  }
  if (buffers->stream != nullptr) {
    (void)hipStreamDestroy(buffers->stream);
  }
  if (buffers->output != nullptr) {
    (void)hipFree(buffers->output);
  }
  if (buffers->weight_scales != nullptr) {
    (void)hipFree(buffers->weight_scales);
  }
  if (buffers->weight != nullptr) {
    (void)hipFree(buffers->weight);
  }
  if (buffers->activation_scales != nullptr) {
    (void)hipFree(buffers->activation_scales);
  }
  if (buffers->activation != nullptr) {
    (void)hipFree(buffers->activation);
  }
  *buffers = {};
}

bool make_buffers(const Shape shape, DeviceBuffers *const buffers) {
  if (buffers == nullptr || shape.m == 0U || shape.k == 0U || shape.n == 0U) {
    return false;
  }
  size_t activation_elements = 0U;
  size_t weight_elements = 0U;
  size_t output_elements = 0U;
  if (!checked_product(shape.m, shape.k, &activation_elements) ||
      !checked_product(shape.n, shape.k, &weight_elements) ||
      !checked_product(shape.m, shape.n, &output_elements) ||
      output_elements > SIZE_MAX / sizeof(uint16_t)) {
    return false;
  }
  buffers->output_elements = output_elements;
  if (!hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->activation),
                        activation_elements),
              "hipMalloc activation") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->activation_scales),
                        static_cast<size_t>(shape.m) * sizeof(float)),
              "hipMalloc activation scales") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->weight),
                        weight_elements),
              "hipMalloc weight") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->weight_scales),
                        static_cast<size_t>(shape.n) * sizeof(float)),
              "hipMalloc weight scales") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->output),
                        output_elements * sizeof(uint16_t)),
              "hipMalloc output") ||
      !hip_ok(hipStreamCreate(&buffers->stream), "hipStreamCreate") ||
      !hip_ok(hipEventCreate(&buffers->start), "hipEventCreate start") ||
      !hip_ok(hipEventCreate(&buffers->stop), "hipEventCreate stop")) {
    free_buffers(buffers);
    return false;
  }
  return true;
}

bool initialize_buffers(const Shape shape, DeviceBuffers *const buffers) {
  size_t activation_elements = 0U;
  size_t weight_elements = 0U;
  if (!checked_product(shape.m, shape.k, &activation_elements) ||
      !checked_product(shape.n, shape.k, &weight_elements)) {
    return false;
  }
  const auto blocks = [](const size_t count) {
    return static_cast<uint32_t>((count + kThreads - 1U) / kThreads);
  };
  hipLaunchKernelGGL(fill_activation_kernel, dim3(blocks(activation_elements)),
                     dim3(kThreads), 0U, buffers->stream, buffers->activation,
                     shape.m, shape.k);
  if (!hip_ok(hipGetLastError(), "launch fill activation")) {
    return false;
  }
  hipLaunchKernelGGL(fill_weight_kernel, dim3(blocks(weight_elements)),
                     dim3(kThreads), 0U, buffers->stream, buffers->weight,
                     shape.k, shape.n);
  if (!hip_ok(hipGetLastError(), "launch fill weight")) {
    return false;
  }
  hipLaunchKernelGGL(
      fill_scale_kernel, dim3(blocks(std::max<size_t>(shape.m, shape.n))),
      dim3(kThreads), 0U, buffers->stream, buffers->activation_scales,
      buffers->weight_scales, shape.m, shape.n);
  return hip_ok(hipGetLastError(), "launch fill scales") &&
         hip_ok(hipStreamSynchronize(buffers->stream),
                "initialize synchronize");
}

bool launch(const Candidate current, const Shape shape,
            DeviceBuffers *const buffers) {
  const uint64_t row_tiles = (shape.m + current.tile_m - 1U) / current.tile_m;
  const uint64_t column_tiles =
      (shape.n + current.tile_n - 1U) / current.tile_n;
  const uint64_t block_count = row_tiles * column_tiles;
  if (block_count == 0U || block_count > UINT32_MAX) {
    return false;
  }
  switch (current.id) {
  case CandidateId::ArchiveId71:
    hipLaunchKernelGGL(sllm_matmul_fp8_outer_prefill_gfx1030_half2_64x64_v1,
                       dim3(static_cast<uint32_t>(block_count)), dim3(kThreads),
                       0U, buffers->stream, buffers->activation,
                       buffers->activation_scales, buffers->weight,
                       buffers->weight_scales, buffers->output, shape.m,
                       shape.k, shape.n);
    break;
  case CandidateId::TileM32N64:
    hipLaunchKernelGGL((fp8_outer_single_kernel<32U, 64U>),
                       dim3(static_cast<uint32_t>(block_count)), dim3(kThreads),
                       0U, buffers->stream, buffers->activation,
                       buffers->activation_scales, buffers->weight,
                       buffers->weight_scales, buffers->output, shape.m,
                       shape.k, shape.n);
    break;
  case CandidateId::TileM32N32:
    hipLaunchKernelGGL((fp8_outer_single_kernel<32U, 32U>),
                       dim3(static_cast<uint32_t>(block_count)), dim3(kThreads),
                       0U, buffers->stream, buffers->activation,
                       buffers->activation_scales, buffers->weight,
                       buffers->weight_scales, buffers->output, shape.m,
                       shape.k, shape.n);
    break;
  }
  return hip_ok(hipGetLastError(), "launch candidate");
}

bool capture(const Candidate current, const Shape shape,
             DeviceBuffers *const buffers, std::vector<uint16_t> *const host) {
  host->resize(buffers->output_elements);
  return launch(current, shape, buffers) &&
         hip_ok(hipStreamSynchronize(buffers->stream), "capture synchronize") &&
         hip_ok(hipMemcpy(host->data(), buffers->output,
                          buffers->output_elements * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "capture output");
}

struct TimingStats final {
  float minimum_us = 0.0F;
  float median_us = 0.0F;
  float maximum_us = 0.0F;
};

bool measure_one(const Candidate current, const Shape shape,
                 DeviceBuffers *const buffers, float *const elapsed_us) {
  if (!hip_ok(hipEventRecord(buffers->start, buffers->stream),
              "event record start") ||
      !launch(current, shape, buffers) ||
      !hip_ok(hipEventRecord(buffers->stop, buffers->stream),
              "event record stop") ||
      !hip_ok(hipEventSynchronize(buffers->stop), "event synchronize") ||
      !hip_ok(hipEventElapsedTime(elapsed_us, buffers->start, buffers->stop),
              "event elapsed")) {
    return false;
  }
  *elapsed_us *= 1000.0F;
  return true;
}

TimingStats summarize_samples(std::array<float, kMeasured> samples) {
  std::sort(samples.begin(), samples.end());
  return {samples.front(), samples[kMeasured / 2], samples.back()};
}

// All paths receive the same three warmups.  Measured iterations rotate the
// first path to reduce stream/clock order bias while retaining one kernel per
// timed event.
bool measure_interleaved(const std::array<Candidate, 3> &candidates,
                         const Shape shape, DeviceBuffers *const buffers,
                         std::array<TimingStats, 3> *const stats) {
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    for (size_t offset = 0U; offset < candidates.size(); ++offset) {
      const size_t index =
          (static_cast<size_t>(warmup) + offset) % candidates.size();
      if (!launch(candidates[index], shape, buffers)) {
        return false;
      }
    }
  }
  if (!hip_ok(hipStreamSynchronize(buffers->stream),
              "interleaved warmup synchronize")) {
    return false;
  }

  std::array<std::array<float, kMeasured>, 3> samples{};
  for (int iteration = 0; iteration < kMeasured; ++iteration) {
    const size_t first = static_cast<size_t>(iteration) % candidates.size();
    for (size_t offset = 0U; offset < candidates.size(); ++offset) {
      const size_t index = (first + offset) % candidates.size();
      if (!measure_one(candidates[index], shape, buffers,
                       &samples[index][iteration])) {
        return false;
      }
    }
  }
  for (size_t index = 0U; index < candidates.size(); ++index) {
    (*stats)[index] = summarize_samples(samples[index]);
  }
  return true;
}

bool compare_outputs(const char *const label, const Shape shape,
                     const std::vector<uint16_t> &expected,
                     const std::vector<uint16_t> &actual) {
  if (expected.size() != actual.size()) {
    std::printf("compare label=%s shape=%s size_mismatch=1 status=FAIL\n",
                label, shape.name);
    return false;
  }
  size_t mismatches = 0U;
  size_t first = 0U;
  for (size_t index = 0U; index < expected.size(); ++index) {
    if (expected[index] != actual[index]) {
      if (mismatches == 0U) {
        first = index;
      }
      ++mismatches;
    }
  }
  std::printf(
      "compare label=%s shape=%s elements=%zu mismatches=%zu first_row=%zu "
      "first_column=%zu status=%s\n",
      label, shape.name, expected.size(), mismatches,
      mismatches == 0U ? 0U : first / static_cast<size_t>(shape.n),
      mismatches == 0U ? 0U : first % static_cast<size_t>(shape.n),
      mismatches == 0U ? "PASS" : "FAIL");
  return mismatches == 0U;
}

bool check_oracle(const Candidate current, const Shape shape,
                  const std::vector<uint16_t> &actual, const bool exhaustive) {
  std::vector<uint64_t> rows;
  std::vector<uint64_t> columns;
  if (exhaustive) {
    rows.reserve(static_cast<size_t>(shape.m));
    columns.reserve(static_cast<size_t>(shape.n));
    for (uint64_t row = 0U; row < shape.m; ++row) {
      rows.push_back(row);
    }
    for (uint64_t column = 0U; column < shape.n; ++column) {
      columns.push_back(column);
    }
  } else {
    rows = {0U, shape.m / 2U, shape.m - 1U};
    columns = {0U, shape.n / 2U, shape.n - 1U};
  }

  size_t mismatches = 0U;
  uint32_t max_bf16_ulp = 0U;
  double max_abs = 0.0;
  double max_rel = 0.0;
  for (const uint64_t row : rows) {
    for (const uint64_t column : columns) {
      float accumulator = 0.0F;
      for (uint64_t inner = 0U; inner < shape.k; ++inner) {
        accumulator = std::fmaf(host_e4m3_to_float(activation_code(row, inner)),
                                host_e4m3_to_float(weight_code(column, inner)),
                                accumulator);
      }
      const uint16_t expected = host_bf16_rne(
          accumulator * activation_scale(row) * weight_scale(column));
      const uint16_t observed =
          actual[static_cast<size_t>(row * shape.n + column)];
      const float expected_value = bf16_to_float(expected);
      const float observed_value = bf16_to_float(observed);
      const double abs_error =
          std::abs(static_cast<double>(observed_value) - expected_value);
      const double rel_error =
          abs_error /
          std::max(1.0e-30, std::abs(static_cast<double>(expected_value)));
      max_abs = std::max(max_abs, abs_error);
      max_rel = std::max(max_rel, rel_error);
      const uint32_t ulp = static_cast<uint32_t>(
          expected > observed ? expected - observed : observed - expected);
      max_bf16_ulp = std::max(max_bf16_ulp, ulp);
      if (expected != observed) {
        ++mismatches;
      }
    }
  }
  std::printf(
      "oracle candidate=%s shape=%s sampled=%zux%zu exhaustive=%d "
      "max_abs=%.9g max_rel=%.9g max_bf16_ulp=%u mismatches=%zu status=%s\n",
      current.name, shape.name, rows.size(), columns.size(), exhaustive ? 1 : 0,
      max_abs, max_rel, max_bf16_ulp, mismatches,
      mismatches == 0U ? "PASS" : "FAIL");
  return mismatches == 0U;
}

bool check_all_codes() {
  std::array<uint32_t, 256> host_input{};
  for (uint32_t code = 0U; code < 256U; ++code) {
    host_input[code] = code * UINT32_C(0x01010101);
  }
  uint32_t *device_input = nullptr;
  uint16_t *device_output = nullptr;
  bool ok = hip_ok(hipMalloc(reinterpret_cast<void **>(&device_input),
                             host_input.size() * sizeof(uint32_t)),
                   "hipMalloc all256 input") &&
            hip_ok(hipMalloc(reinterpret_cast<void **>(&device_output),
                             host_input.size() * 4U * sizeof(uint16_t)),
                   "hipMalloc all256 output");
  if (ok) {
    ok = hip_ok(hipMemcpy(device_input, host_input.data(),
                          host_input.size() * sizeof(uint32_t),
                          hipMemcpyHostToDevice),
                "hipMemcpy all256 input");
  }
  if (ok) {
    hipLaunchKernelGGL(e4m3x4_probe_kernel, dim3(1), dim3(256), 0U, nullptr,
                       device_input, device_output);
    ok = hip_ok(hipGetLastError(), "launch all256") &&
         hip_ok(hipDeviceSynchronize(), "synchronize all256");
  }
  std::array<uint16_t, 1024> host_output{};
  if (ok) {
    ok = hip_ok(hipMemcpy(host_output.data(), device_output,
                          host_output.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "hipMemcpy all256 output");
  }
  std::array<size_t, 4> lane_mismatches{};
  if (ok) {
    for (uint32_t code = 0U; code < 256U; ++code) {
      const uint16_t expected =
          host_e4m3_to_fp16_bits_reference(static_cast<uint8_t>(code));
      for (uint32_t lane = 0U; lane < 4U; ++lane) {
        if (host_output[code * 4U + lane] != expected) {
          ++lane_mismatches[lane];
        }
      }
    }
  }
  const size_t total = lane_mismatches[0] + lane_mismatches[1] +
                       lane_mismatches[2] + lane_mismatches[3];
  std::printf("oracle ingress=E4M3FNx4 codes=256 lane0=%zu lane1=%zu lane2=%zu "
              "lane3=%zu mismatches=%zu status=%s\n",
              lane_mismatches[0], lane_mismatches[1], lane_mismatches[2],
              lane_mismatches[3], total, ok && total == 0U ? "PASS" : "FAIL");
  if (device_output != nullptr) {
    (void)hipFree(device_output);
  }
  if (device_input != nullptr) {
    (void)hipFree(device_input);
  }
  return ok && total == 0U;
}

void print_resources(const Candidate current) {
  hipFuncAttributes attributes{};
  const hipError_t attribute_status =
      hipFuncGetAttributes(&attributes, current.function);
  int active_blocks = 0;
  const hipError_t occupancy_status =
      hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks, current.function, kThreads, 0U);
  std::printf("resources candidate=%s vgpr=%d lds=%zu scratch=%zu spill=%s "
              "max_threads=%d active_blocks_per_cu=%d active_waves_per_cu=%d "
              "attributes=%s occupancy=%s\n",
              current.name, attributes.numRegs, attributes.sharedSizeBytes,
              attributes.localSizeBytes,
              attributes.localSizeBytes == 0U ? "none" : "present",
              attributes.maxThreadsPerBlock, active_blocks, active_blocks * 8,
              hipGetErrorString(attribute_status),
              hipGetErrorString(occupancy_status));
}

bool parse_device(const char *const text, int *const device) {
  if (text == nullptr || device == nullptr) {
    return false;
  }
  char *end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed < 0L || parsed > INT_MAX) {
    return false;
  }
  *device = static_cast<int>(parsed);
  return true;
}

} // namespace

int main(int argc, char **argv) {
  int device = 0;
  if (argc > 2 || (argc == 2 && !parse_device(argv[1], &device))) {
    std::fprintf(stderr,
                 "usage: phase78_fp8_gfx1030_short_m32_probe [DEVICE]\n");
    return EXIT_FAILURE;
  }
  if (!hip_ok(hipSetDevice(device), "hipSetDevice")) {
    return EXIT_FAILURE;
  }
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device),
              "hipGetDeviceProperties") ||
      !exact_gfx1030(properties.gcnArchName)) {
    std::fprintf(stderr, "exact gfx1030 required; got %s\n",
                 properties.gcnArchName);
    return EXIT_FAILURE;
  }
  std::printf("target=%s logical_device=%d pci=%04x:%02x:%02x name=%s "
              "order=alternating common_warmups=%d measured=%d\n",
              properties.gcnArchName, device, properties.pciDomainID,
              properties.pciBusID, properties.pciDeviceID, properties.name,
              kWarmups, kMeasured);

  bool all_ok = check_all_codes();
  const Candidate control = candidate(CandidateId::ArchiveId71);
  const Candidate tile_m32_n64 = candidate(CandidateId::TileM32N64);
  const Candidate tile_m32_n32 = candidate(CandidateId::TileM32N32);
  print_resources(control);
  print_resources(tile_m32_n64);
  print_resources(tile_m32_n32);

  constexpr std::array<Shape, 5> shapes = {
      Shape{17U, 6144U, 5120U, "m17-k6144-n5120"},
      Shape{31U, 6144U, 5120U, "m31-k6144-n5120"},
      Shape{32U, 6144U, 5120U, "m32-k6144-n5120"},
      Shape{33U, 6144U, 5120U, "m33-k6144-n5120"},
      Shape{17U, 70U, 31U, "tiny-m17-k70-n31"}};

  for (const Shape shape : shapes) {
    DeviceBuffers buffers;
    if (!make_buffers(shape, &buffers) ||
        !initialize_buffers(shape, &buffers)) {
      free_buffers(&buffers);
      return EXIT_FAILURE;
    }

    std::vector<uint16_t> control_output;
    std::vector<uint16_t> control_repeat;
    std::vector<uint16_t> tile_m32_n64_output;
    std::vector<uint16_t> tile_m32_n64_repeat;
    std::vector<uint16_t> tile_m32_n32_output;
    std::vector<uint16_t> tile_m32_n32_repeat;
    if (!capture(control, shape, &buffers, &control_output) ||
        !capture(control, shape, &buffers, &control_repeat) ||
        !capture(tile_m32_n64, shape, &buffers, &tile_m32_n64_output) ||
        !capture(tile_m32_n64, shape, &buffers, &tile_m32_n64_repeat) ||
        !capture(tile_m32_n32, shape, &buffers, &tile_m32_n32_output) ||
        !capture(tile_m32_n32, shape, &buffers, &tile_m32_n32_repeat)) {
      free_buffers(&buffers);
      return EXIT_FAILURE;
    }
    const bool exhaustive = shape.name[0] == 't';
    all_ok = compare_outputs("control-determinism", shape, control_output,
                             control_repeat) &&
             all_ok;
    all_ok = compare_outputs("tilem32n64-determinism", shape,
                             tile_m32_n64_output, tile_m32_n64_repeat) &&
             all_ok;
    all_ok = compare_outputs("tilem32n32-determinism", shape,
                             tile_m32_n32_output, tile_m32_n32_repeat) &&
             all_ok;
    all_ok = compare_outputs("tilem32n64-bitwise-vs-archive-id71", shape,
                             control_output, tile_m32_n64_output) &&
             all_ok;
    all_ok = compare_outputs("tilem32n32-bitwise-vs-archive-id71", shape,
                             control_output, tile_m32_n32_output) &&
             all_ok;
    all_ok = check_oracle(control, shape, control_output, exhaustive) && all_ok;
    all_ok =
        check_oracle(tile_m32_n64, shape, tile_m32_n64_output, exhaustive) &&
        all_ok;
    all_ok =
        check_oracle(tile_m32_n32, shape, tile_m32_n32_output, exhaustive) &&
        all_ok;

    const std::array<Candidate, 3> candidates = {control, tile_m32_n64,
                                                 tile_m32_n32};
    std::array<TimingStats, 3> timing{};
    if (!measure_interleaved(candidates, shape, &buffers, &timing)) {
      free_buffers(&buffers);
      return EXIT_FAILURE;
    }
    const double tflops =
        (2.0 * static_cast<double>(shape.m) * static_cast<double>(shape.k) *
         static_cast<double>(shape.n)) /
        (static_cast<double>(timing[0].median_us) * 1.0e6);
    std::printf(
        "result shape=%s m=%llu k=%llu n=%llu control=%s "
        "control_min_us=%.3f control_median_us=%.3f control_max_us=%.3f "
        "candidate_n64=%s candidate_n64_min_us=%.3f "
        "candidate_n64_median_us=%.3f candidate_n64_max_us=%.3f "
        "speedup_control_over_n64=%.6f candidate_n32=%s "
        "candidate_n32_min_us=%.3f candidate_n32_median_us=%.3f "
        "candidate_n32_max_us=%.3f speedup_control_over_n32=%.6f "
        "control_tflops=%.6f\n",
        shape.name, static_cast<unsigned long long>(shape.m),
        static_cast<unsigned long long>(shape.k),
        static_cast<unsigned long long>(shape.n), control.name,
        timing[0].minimum_us, timing[0].median_us, timing[0].maximum_us,
        tile_m32_n64.name, timing[1].minimum_us, timing[1].median_us,
        timing[1].maximum_us, timing[0].median_us / timing[1].median_us,
        tile_m32_n32.name, timing[2].minimum_us, timing[2].median_us,
        timing[2].maximum_us, timing[0].median_us / timing[2].median_us,
        tflops);
    free_buffers(&buffers);
  }

  size_t free_bytes = 0U;
  size_t total_bytes = 0U;
  const bool memory_ok =
      hip_ok(hipMemGetInfo(&free_bytes, &total_bytes), "hipMemGetInfo");
  all_ok = memory_ok && all_ok;
  std::printf(
      "cleanup free_bytes=%zu total_bytes=%zu status=%s\n"
      "summary status=%s control=archive-id71 candidates=tilem32x64,tilem32x32 "
      "shapes=%zu warmups=%d measured=%d\n",
      free_bytes, total_bytes, memory_ok ? "PASS" : "FAIL",
      all_ok ? "PASS" : "FAIL", shapes.size(), kWarmups, kMeasured);
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
