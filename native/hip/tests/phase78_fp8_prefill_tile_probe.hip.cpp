// Phase 78 standalone gfx1030 FP8 outer-prefill tile probe.
//
// This file is intentionally not part of the public build.  It reproduces the
// current ID63 128x64x32 half2 tile and compares smaller LDS/register tiles
// while keeping the numerical contract unchanged: OCP E4M3FN ingress is
// converted exactly to FP16, products are accumulated in FP32, outer F32
// scales are applied in the epilogue, and the result is BF16 RNE.

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

namespace {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kThreadRows = 16U;
constexpr uint32_t kThreadColumns = 16U;
constexpr int kWarmups = 3;
constexpr int kMeasured = 10;

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

template <uint32_t TileM, uint32_t TileN, uint32_t TileK>
__global__ __launch_bounds__(kThreads, 1) void fp8_outer_tile_kernel(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  static_assert(TileM % kThreadRows == 0U);
  static_assert(TileN % kThreadColumns == 0U);
  static_assert(TileK % 4U == 0U);
  constexpr uint32_t rows_per_thread = TileM / kThreadRows;
  constexpr uint32_t columns_per_thread = TileN / kThreadColumns;
  constexpr uint32_t lds_stride = TileK + 2U;
  __shared__ uint16_t activation_tile[TileM][lds_stride];
  __shared__ uint16_t weight_tile[TileN][lds_stride];

  const uint64_t column_tiles = (n + TileN - 1U) / TileN;
  const uint64_t tile_index = static_cast<uint64_t>(blockIdx.x);
  const uint64_t row_base = (tile_index / column_tiles) * TileM;
  const uint64_t column_base = (tile_index % column_tiles) * TileN;
  const uint32_t thread = threadIdx.x;
  const uint32_t local_row = thread >> 4U;
  const uint32_t local_column = thread & 15U;
  float accumulators[rows_per_thread][columns_per_thread] = {};

  for (uint64_t base = 0U; base < k; base += TileK) {
    constexpr uint32_t values_per_load = 4U;
    constexpr uint32_t loads_per_row = TileK / values_per_load;
    for (uint32_t index = thread; index < TileM * loads_per_row;
         index += kThreads) {
      const uint32_t row = index / loads_per_row;
      const uint32_t inner = (index % loads_per_row) * values_per_load;
      const uint64_t source_row = row_base + row;
      const uint64_t source_inner = base + inner;
      if ((k % values_per_load) == 0U && source_row < m &&
          source_inner + values_per_load <= k) {
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
        for (uint32_t lane = 0U; lane < values_per_load; ++lane) {
          activation_tile[row][inner + lane] =
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
      const uint64_t source_inner = base + inner;
      if ((k % values_per_load) == 0U && source_column < n &&
          source_inner + values_per_load <= k) {
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
        for (uint32_t lane = 0U; lane < values_per_load; ++lane) {
          weight_tile[column][inner + lane] =
              source_column < n && source_inner + lane < k
                  ? sllm_lowp::e4m3fn_to_fp16_bits(__builtin_nontemporal_load(
                        weight + source_column * k + source_inner + lane))
                  : UINT16_C(0);
        }
      }
    }
    __syncthreads();

    if (row_base + local_row < m && column_base + local_column < n) {
#pragma unroll
      for (uint32_t inner = 0U; inner < TileK; inner += 2U) {
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

__global__ void e4m3_to_fp16_probe_kernel(const uint8_t *const input,
                                          uint16_t *const output) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 256U) {
    output[index] = sllm_lowp::e4m3fn_to_fp16_bits(input[index]);
  }
}

struct Candidate final {
  const char *name;
  uint32_t tile_m;
  uint32_t tile_n;
  uint32_t tile_k;
  const void *function;
};

enum class CandidateId {
  Id63,
  M64N64K32,
  M64N64K64,
  M64N64K128,
  M64N32K32,
  M64N32K64
};

Candidate candidate(const CandidateId id) {
  switch (id) {
  case CandidateId::Id63:
    return {
        "id63-128x64-k32", 128U, 64U, 32U,
        reinterpret_cast<const void *>(fp8_outer_tile_kernel<128U, 64U, 32U>)};
  case CandidateId::M64N64K32:
    return {
        "candidate-64x64-k32", 64U, 64U, 32U,
        reinterpret_cast<const void *>(fp8_outer_tile_kernel<64U, 64U, 32U>)};
  case CandidateId::M64N64K64:
    return {
        "candidate-64x64-k64", 64U, 64U, 64U,
        reinterpret_cast<const void *>(fp8_outer_tile_kernel<64U, 64U, 64U>)};
  case CandidateId::M64N64K128:
    return {
        "candidate-64x64-k128", 64U, 64U, 128U,
        reinterpret_cast<const void *>(fp8_outer_tile_kernel<64U, 64U, 128U>)};
  case CandidateId::M64N32K32:
    return {
        "candidate-64x32-k32", 64U, 32U, 32U,
        reinterpret_cast<const void *>(fp8_outer_tile_kernel<64U, 32U, 32U>)};
  case CandidateId::M64N32K64:
    return {
        "candidate-64x32-k64", 64U, 32U, 64U,
        reinterpret_cast<const void *>(fp8_outer_tile_kernel<64U, 32U, 64U>)};
  }
  return {"invalid", 0U, 0U, 0U, nullptr};
}

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

float fp16_to_float(const uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16U;
  const uint32_t exponent = (bits >> 10U) & 0x1fU;
  const uint32_t mantissa = bits & 0x3ffU;
  uint32_t result = sign;
  if (exponent == 0U) {
    if (mantissa != 0U) {
      float value = static_cast<float>(mantissa) * 0x1p-24F;
      std::memcpy(&result, &value, sizeof(result));
      result = (result & UINT32_C(0x7fffffff)) | sign;
    }
  } else if (exponent == 31U) {
    result |= UINT32_C(0x7f800000) | (mantissa << 13U);
  } else {
    result |= ((exponent + 112U) << 23U) | (mantissa << 13U);
  }
  float value = 0.0F;
  std::memcpy(&value, &result, sizeof(value));
  return value;
}

float bf16_to_float_host(const uint16_t bits) {
  const uint32_t expanded = static_cast<uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &expanded, sizeof(value));
  return value;
}

float e4m3_to_float_host(const uint8_t code) {
  return fp16_to_float(sllm_lowp::e4m3fn_to_fp16_bits(code));
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

bool make_buffers(const uint64_t m, const uint64_t k, const uint64_t n,
                  DeviceBuffers *const buffers) {
  if (buffers == nullptr || m == 0U || k == 0U || n == 0U || m > SIZE_MAX / k ||
      n > SIZE_MAX / k) {
    return false;
  }
  const size_t activation_bytes = static_cast<size_t>(m * k);
  const size_t weight_bytes = static_cast<size_t>(n * k);
  const size_t output_bytes = static_cast<size_t>(m * n * sizeof(uint16_t));
  if (!hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->activation),
                        activation_bytes),
              "hipMalloc activation") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->activation_scales),
                        m * sizeof(float)),
              "hipMalloc activation scales") ||
      !hip_ok(
          hipMalloc(reinterpret_cast<void **>(&buffers->weight), weight_bytes),
          "hipMalloc weight") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->weight_scales),
                        n * sizeof(float)),
              "hipMalloc weight scales") ||
      !hip_ok(
          hipMalloc(reinterpret_cast<void **>(&buffers->output), output_bytes),
          "hipMalloc output") ||
      !hip_ok(hipStreamCreate(&buffers->stream), "hipStreamCreate") ||
      !hip_ok(hipEventCreate(&buffers->start), "hipEventCreate start") ||
      !hip_ok(hipEventCreate(&buffers->stop), "hipEventCreate stop")) {
    free_buffers(buffers);
    return false;
  }
  return true;
}

void fill_inputs(const uint64_t m, const uint64_t k, const uint64_t n,
                 std::vector<uint8_t> *const activation,
                 std::vector<uint8_t> *const weight,
                 std::vector<float> *const activation_scales,
                 std::vector<float> *const weight_scales) {
  activation->resize(static_cast<size_t>(m * k));
  weight->resize(static_cast<size_t>(n * k));
  activation_scales->resize(static_cast<size_t>(m));
  weight_scales->resize(static_cast<size_t>(n));
  for (uint64_t row = 0U; row < m; ++row) {
    (*activation_scales)[row] =
        0.75F + static_cast<float>(row % 13U) * 0.03125F;
    for (uint64_t inner = 0U; inner < k; ++inner) {
      uint32_t code =
          static_cast<uint32_t>((row * 37U + inner * 11U + 5U) & 255U);
      if (code == 0x7fU || code == 0xffU) {
        code = 0x7eU;
      }
      (*activation)[static_cast<size_t>(row * k + inner)] =
          static_cast<uint8_t>(code);
    }
  }
  for (uint64_t column = 0U; column < n; ++column) {
    (*weight_scales)[column] =
        0.625F + static_cast<float>(column % 17U) * 0.0234375F;
    for (uint64_t inner = 0U; inner < k; ++inner) {
      uint32_t code =
          static_cast<uint32_t>((column * 19U + inner * 7U + 13U) & 255U);
      if (code == 0x7fU || code == 0xffU) {
        code = 0x7eU;
      }
      (*weight)[static_cast<size_t>(column * k + inner)] =
          static_cast<uint8_t>(code);
    }
  }
}

bool upload_inputs(const uint64_t m, const uint64_t k, const uint64_t n,
                   const std::vector<uint8_t> &activation,
                   const std::vector<uint8_t> &weight,
                   const std::vector<float> &activation_scales,
                   const std::vector<float> &weight_scales,
                   DeviceBuffers *const buffers) {
  return hip_ok(hipMemcpy(buffers->activation, activation.data(), m * k,
                          hipMemcpyHostToDevice),
                "hipMemcpy activation") &&
         hip_ok(hipMemcpy(buffers->activation_scales, activation_scales.data(),
                          m * sizeof(float), hipMemcpyHostToDevice),
                "hipMemcpy activation scales") &&
         hip_ok(hipMemcpy(buffers->weight, weight.data(), n * k,
                          hipMemcpyHostToDevice),
                "hipMemcpy weight") &&
         hip_ok(hipMemcpy(buffers->weight_scales, weight_scales.data(),
                          n * sizeof(float), hipMemcpyHostToDevice),
                "hipMemcpy weight scales") &&
         hip_ok(hipMemset(buffers->output, 0, m * n * sizeof(uint16_t)),
                "hipMemset output");
}

bool launch(const Candidate &candidate, const uint64_t m, const uint64_t k,
            const uint64_t n, DeviceBuffers *const buffers) {
  const uint64_t row_tiles = (m + candidate.tile_m - 1U) / candidate.tile_m;
  const uint64_t column_tiles = (n + candidate.tile_n - 1U) / candidate.tile_n;
  const uint64_t block_count = row_tiles * column_tiles;
  if (block_count == 0U || block_count > UINT32_MAX) {
    return false;
  }
  if (candidate.tile_m == 128U && candidate.tile_n == 64U &&
      candidate.tile_k == 32U) {
    hipLaunchKernelGGL((fp8_outer_tile_kernel<128U, 64U, 32U>),
                       dim3(static_cast<uint32_t>(block_count)), dim3(kThreads),
                       0U, buffers->stream, buffers->activation,
                       buffers->activation_scales, buffers->weight,
                       buffers->weight_scales, buffers->output, m, k, n);
  } else if (candidate.tile_m == 64U && candidate.tile_n == 64U &&
             candidate.tile_k == 32U) {
    hipLaunchKernelGGL((fp8_outer_tile_kernel<64U, 64U, 32U>),
                       dim3(static_cast<uint32_t>(block_count)), dim3(kThreads),
                       0U, buffers->stream, buffers->activation,
                       buffers->activation_scales, buffers->weight,
                       buffers->weight_scales, buffers->output, m, k, n);
  } else if (candidate.tile_m == 64U && candidate.tile_n == 64U &&
             candidate.tile_k == 64U) {
    hipLaunchKernelGGL((fp8_outer_tile_kernel<64U, 64U, 64U>),
                       dim3(static_cast<uint32_t>(block_count)), dim3(kThreads),
                       0U, buffers->stream, buffers->activation,
                       buffers->activation_scales, buffers->weight,
                       buffers->weight_scales, buffers->output, m, k, n);
  } else if (candidate.tile_m == 64U && candidate.tile_n == 64U &&
             candidate.tile_k == 128U) {
    hipLaunchKernelGGL((fp8_outer_tile_kernel<64U, 64U, 128U>),
                       dim3(static_cast<uint32_t>(block_count)), dim3(kThreads),
                       0U, buffers->stream, buffers->activation,
                       buffers->activation_scales, buffers->weight,
                       buffers->weight_scales, buffers->output, m, k, n);
  } else if (candidate.tile_m == 64U && candidate.tile_n == 32U &&
             candidate.tile_k == 32U) {
    hipLaunchKernelGGL((fp8_outer_tile_kernel<64U, 32U, 32U>),
                       dim3(static_cast<uint32_t>(block_count)), dim3(kThreads),
                       0U, buffers->stream, buffers->activation,
                       buffers->activation_scales, buffers->weight,
                       buffers->weight_scales, buffers->output, m, k, n);
  } else {
    hipLaunchKernelGGL((fp8_outer_tile_kernel<64U, 32U, 64U>),
                       dim3(static_cast<uint32_t>(block_count)), dim3(kThreads),
                       0U, buffers->stream, buffers->activation,
                       buffers->activation_scales, buffers->weight,
                       buffers->weight_scales, buffers->output, m, k, n);
  }
  return hipGetLastError() == hipSuccess;
}

bool measure(const Candidate &candidate, const uint64_t m, const uint64_t k,
             const uint64_t n, DeviceBuffers *const buffers, float *const us) {
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    if (!launch(candidate, m, k, n, buffers) ||
        !hip_ok(hipStreamSynchronize(buffers->stream), "warmup synchronize")) {
      return false;
    }
  }
  std::array<float, kMeasured> samples{};
  for (int iteration = 0; iteration < kMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(buffers->start, buffers->stream),
                "event start") ||
        !launch(candidate, m, k, n, buffers) ||
        !hip_ok(hipEventRecord(buffers->stop, buffers->stream), "event stop") ||
        !hip_ok(hipEventSynchronize(buffers->stop), "event synchronize") ||
        !hip_ok(hipEventElapsedTime(&samples[iteration], buffers->start,
                                    buffers->stop),
                "event elapsed")) {
      return false;
    }
    samples[iteration] *= 1000.0F;
  }
  std::sort(samples.begin(), samples.end());
  *us = samples[kMeasured / 2];
  return true;
}

bool check_output(const Candidate &candidate, const uint64_t m,
                  const uint64_t k, const uint64_t n,
                  const std::vector<uint8_t> &activation,
                  const std::vector<uint8_t> &weight,
                  const std::vector<float> &activation_scales,
                  const std::vector<float> &weight_scales,
                  const std::vector<uint16_t> &actual) {
  double max_abs = 0.0;
  double max_rel = 0.0;
  size_t mismatch = 0U;
  // Full matrices are intentionally not sent through a host oracle here.
  // The representative tensors contain billions of products; a 4x4 output
  // sample still exercises every K value while keeping this probe focused on
  // tile correctness and performance.
  const uint64_t oracle_rows = std::min<uint64_t>(m, 4U);
  const uint64_t oracle_columns = std::min<uint64_t>(n, 4U);
  for (uint64_t row = 0U; row < oracle_rows; ++row) {
    for (uint64_t column = 0U; column < oracle_columns; ++column) {
      float accumulator = 0.0F;
      for (uint64_t inner = 0U; inner < k; ++inner) {
        accumulator = std::fmaf(
            e4m3_to_float_host(
                activation[static_cast<size_t>(row * k + inner)]),
            e4m3_to_float_host(weight[static_cast<size_t>(column * k + inner)]),
            accumulator);
      }
      const float expected_value = accumulator *
                                   activation_scales[static_cast<size_t>(row)] *
                                   weight_scales[static_cast<size_t>(column)];
      const uint16_t expected = host_bf16_rne(expected_value);
      const float expected_float = bf16_to_float_host(expected);
      const float actual_float =
          bf16_to_float_host(actual[static_cast<size_t>(row * n + column)]);
      const double abs_error =
          std::abs(static_cast<double>(actual_float) - expected_float);
      const double relative =
          abs_error /
          std::max(1.0e-6, std::abs(static_cast<double>(expected_float)));
      max_abs = std::max(max_abs, abs_error);
      max_rel = std::max(max_rel, relative);
      if ((std::isnan(expected_float) != std::isnan(actual_float)) ||
          (!std::isnan(expected_float) && abs_error > 0.125)) {
        ++mismatch;
      }
    }
  }
  std::printf("oracle candidate=%s m=%llu k=%llu n=%llu sampled=%llux%llu "
              "max_abs=%.8g max_rel=%.8g mismatches=%zu status=%s\n",
              candidate.name, static_cast<unsigned long long>(m),
              static_cast<unsigned long long>(k),
              static_cast<unsigned long long>(n),
              static_cast<unsigned long long>(oracle_rows),
              static_cast<unsigned long long>(oracle_columns), max_abs, max_rel,
              mismatch, mismatch == 0U ? "PASS" : "FAIL");
  return mismatch == 0U;
}

bool check_all_codes() {
  uint8_t *input = nullptr;
  uint16_t *output = nullptr;
  if (!hip_ok(hipMalloc(reinterpret_cast<void **>(&input), 256U),
              "hipMalloc codes input") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&output),
                        256U * sizeof(uint16_t)),
              "hipMalloc codes output")) {
    if (input != nullptr) {
      (void)hipFree(input);
    }
    if (output != nullptr) {
      (void)hipFree(output);
    }
    return false;
  }
  std::array<uint8_t, 256> codes{};
  for (uint32_t index = 0U; index < 256U; ++index) {
    codes[index] = static_cast<uint8_t>(index);
  }
  bool ok = hip_ok(
      hipMemcpy(input, codes.data(), codes.size(), hipMemcpyHostToDevice),
      "hipMemcpy codes");
  if (ok) {
    hipLaunchKernelGGL((e4m3_to_fp16_probe_kernel), dim3(1), dim3(256), 0U,
                       nullptr, input, output);
    // hipLaunchKernelGGL is a statement macro, so validate the launch
    // separately.
    ok = hip_ok(hipGetLastError(), "launch code converter") &&
         hip_ok(hipDeviceSynchronize(), "synchronize code converter");
  }
  std::array<uint16_t, 256> converted{};
  ok = ok && hip_ok(hipMemcpy(converted.data(), output,
                              converted.size() * sizeof(uint16_t),
                              hipMemcpyDeviceToHost),
                    "hipMemcpy converted codes");
  size_t mismatches = 0U;
  for (uint32_t index = 0U; index < 256U; ++index) {
    if (converted[index] != sllm_lowp::e4m3fn_to_fp16_bits(codes[index])) {
      ++mismatches;
    }
  }
  std::printf("oracle ingress all256 mismatches=%zu status=%s\n", mismatches,
              ok && mismatches == 0U ? "PASS" : "FAIL");
  (void)hipFree(output);
  (void)hipFree(input);
  return ok && mismatches == 0U;
}

void print_resources(const Candidate &candidate) {
  hipFuncAttributes attributes{};
  const hipError_t attribute_status =
      hipFuncGetAttributes(&attributes, candidate.function);
  int active_blocks = 0;
  const hipError_t occupancy_status =
      hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks, candidate.function, kThreads, 0U);
  std::printf(
      "resources candidate=%s registers=%d sgpr=unavailable lds=%zu "
      "scratch=%zu max_threads=%d active_blocks=%d attr=%s occupancy=%s\n",
      candidate.name, attributes.numRegs, attributes.sharedSizeBytes,
      attributes.localSizeBytes, attributes.maxThreadsPerBlock, active_blocks,
      hipGetErrorString(attribute_status), hipGetErrorString(occupancy_status));
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
    std::fprintf(stderr, "usage: phase78_fp8_prefill_tile_probe [DEVICE]\n");
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
  std::printf("target=%s device=%d pci=%04x:%02x:%02x name=%s\n",
              properties.gcnArchName, device, properties.pciDomainID,
              properties.pciBusID, properties.pciDeviceID, properties.name);
  if (!check_all_codes()) {
    return EXIT_FAILURE;
  }

  const std::array<CandidateId, 6> candidate_ids = {
      CandidateId::Id63,       CandidateId::M64N64K32, CandidateId::M64N64K64,
      CandidateId::M64N64K128, CandidateId::M64N32K32, CandidateId::M64N32K64};
  for (const CandidateId id : candidate_ids) {
    print_resources(candidate(id));
  }

  struct Shape final {
    uint64_t m;
    uint64_t k;
    uint64_t n;
  };
  const std::array<Shape, 6> shapes = {
      Shape{128U, 5120U, 17408U},  Shape{128U, 17408U, 5120U},
      Shape{512U, 5120U, 17408U},  Shape{512U, 17408U, 5120U},
      Shape{1024U, 5120U, 17408U}, Shape{1024U, 17408U, 5120U}};
  bool all_ok = true;
  for (const Shape shape : shapes) {
    std::vector<uint8_t> activation;
    std::vector<uint8_t> weight;
    std::vector<float> activation_scales;
    std::vector<float> weight_scales;
    fill_inputs(shape.m, shape.k, shape.n, &activation, &weight,
                &activation_scales, &weight_scales);
    DeviceBuffers buffers;
    if (!make_buffers(shape.m, shape.k, shape.n, &buffers) ||
        !upload_inputs(shape.m, shape.k, shape.n, activation, weight,
                       activation_scales, weight_scales, &buffers)) {
      free_buffers(&buffers);
      return EXIT_FAILURE;
    }
    for (const CandidateId id : candidate_ids) {
      const Candidate current = candidate(id);
      float median_us = 0.0F;
      if (!measure(current, shape.m, shape.k, shape.n, &buffers, &median_us)) {
        free_buffers(&buffers);
        return EXIT_FAILURE;
      }
      std::printf("result candidate=%s m=%llu k=%llu n=%llu median_us=%.3f "
                  "tokens_per_s=%.6f\n",
                  current.name, static_cast<unsigned long long>(shape.m),
                  static_cast<unsigned long long>(shape.k),
                  static_cast<unsigned long long>(shape.n), median_us,
                  1000000.0F / median_us);
      if ((shape.m == 128U && shape.k == 5120U && shape.n == 17408U) ||
          (shape.m == 512U && shape.k == 17408U && shape.n == 5120U)) {
        std::vector<uint16_t> actual(static_cast<size_t>(shape.m * shape.n));
        if (!hip_ok(hipMemcpy(actual.data(), buffers.output,
                              actual.size() * sizeof(uint16_t),
                              hipMemcpyDeviceToHost),
                    "hipMemcpy oracle output")) {
          free_buffers(&buffers);
          return EXIT_FAILURE;
        }
        all_ok =
            check_output(current, shape.m, shape.k, shape.n, activation, weight,
                         activation_scales, weight_scales, actual) &&
            all_ok;
      }
    }
    free_buffers(&buffers);
  }

  // Boundary oracle: non-aligned M/N and a K that is not a TileK multiple.
  // The same all-finite code construction keeps this check focused on tile
  // bounds and exact ingress rather than NaN propagation.
  constexpr Shape boundary{17U, 70U, 31U};
  std::vector<uint8_t> activation;
  std::vector<uint8_t> weight;
  std::vector<float> activation_scales;
  std::vector<float> weight_scales;
  fill_inputs(boundary.m, boundary.k, boundary.n, &activation, &weight,
              &activation_scales, &weight_scales);
  DeviceBuffers buffers;
  if (!make_buffers(boundary.m, boundary.k, boundary.n, &buffers) ||
      !upload_inputs(boundary.m, boundary.k, boundary.n, activation, weight,
                     activation_scales, weight_scales, &buffers)) {
    free_buffers(&buffers);
    return EXIT_FAILURE;
  }
  std::vector<uint16_t> actual(static_cast<size_t>(boundary.m * boundary.n));
  for (const CandidateId id : candidate_ids) {
    const Candidate current = candidate(id);
    if (!launch(current, boundary.m, boundary.k, boundary.n, &buffers) ||
        !hip_ok(hipDeviceSynchronize(), "boundary synchronize") ||
        !hip_ok(hipMemcpy(actual.data(), buffers.output,
                          actual.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "boundary output")) {
      free_buffers(&buffers);
      return EXIT_FAILURE;
    }
    all_ok =
        check_output(current, boundary.m, boundary.k, boundary.n, activation,
                     weight, activation_scales, weight_scales, actual) &&
        all_ok;
  }
  free_buffers(&buffers);
  std::printf("summary status=%s candidates=%zu warmups=%d measured=%d\n",
              all_ok ? "PASS" : "FAIL", candidate_ids.size(), kWarmups,
              kMeasured);
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
