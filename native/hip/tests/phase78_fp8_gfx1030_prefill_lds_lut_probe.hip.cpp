// Phase 78 standalone gfx1030 FP8 prefill ingress LUT probe.
//
// The control is the production ID71 64x64/K32 half2 tile.  The candidate
// keeps its geometry, dot order, outer scales, and BF16 RNE epilogue, but
// converts both activation and weight E4M3FN bytes through a padded 256-entry
// FP16-bit LUT staged once in LDS per workgroup.  This file is standalone and
// intentionally does not modify or link production sources.

#include "low_precision_block_codec.hpp"

#include <hip/hip_fp16.h>
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
constexpr uint32_t kThreadRows = 16U;
constexpr uint32_t kThreadColumns = 16U;
constexpr uint32_t kTileM = 64U;
constexpr uint32_t kTileN = 64U;
constexpr uint32_t kTileK = 32U;
constexpr int kWarmups = 3;
constexpr int kMeasured = 10;
constexpr uint32_t kGuardWords = 32U;

__device__ __constant__ uint16_t kFp8Lut[256U];

__device__ __forceinline__ uint16_t bf16_rne(const float value) noexcept {
  const uint32_t bits = __float_as_uint(value);
  if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
    if ((bits & UINT32_C(0x007fffff)) != 0U) {
      return static_cast<uint16_t>(((bits >> 16U) & 0x8000U) | 0x7fc0U |
                                   ((bits >> 16U) & 0x003fU));
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

__device__ __forceinline__ uint32_t lut_slot(const uint32_t code) noexcept {
  return code + (code >> 5U);
}

template <bool Lut>
__device__ __forceinline__ uint16_t decode_fp8(const uint8_t code,
                                               const uint16_t *const table) {
  if constexpr (Lut) {
    return table[lut_slot(static_cast<uint32_t>(code))];
  }
  (void)table;
  return sllm_lowp::e4m3fn_to_fp16_bits(code);
}

template <bool Lut>
__device__ __forceinline__ void decode_four(const uint32_t packed,
                                            const uint16_t *const table,
                                            uint16_t *const destination) {
  destination[0] = decode_fp8<Lut>(static_cast<uint8_t>(packed), table);
  destination[1] = decode_fp8<Lut>(static_cast<uint8_t>(packed >> 8U), table);
  destination[2] = decode_fp8<Lut>(static_cast<uint8_t>(packed >> 16U), table);
  destination[3] = decode_fp8<Lut>(static_cast<uint8_t>(packed >> 24U), table);
}

template <bool Lut>
__device__ __forceinline__ void fp8_prefill_64x64_k32(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, uint16_t (&activation_tile)[kTileM][kTileK + 2U],
    uint16_t (&weight_tile)[kTileN][kTileK + 2U], uint16_t *const lut) {
  constexpr uint32_t rows_per_thread = kTileM / kThreadRows;
  constexpr uint32_t columns_per_thread = kTileN / kThreadColumns;

  if constexpr (Lut) {
    if (threadIdx.x < 256U) {
      const uint32_t code = threadIdx.x;
      lut[lut_slot(code)] = kFp8Lut[code];
    }
    __syncthreads();
  }

  const uint64_t column_tiles = (n + kTileN - 1U) / kTileN;
  const uint64_t tile_index = static_cast<uint64_t>(blockIdx.x);
  const uint64_t row_base = (tile_index / column_tiles) * kTileM;
  const uint64_t column_base = (tile_index % column_tiles) * kTileN;
  const uint32_t thread = threadIdx.x;
  const uint32_t local_row = thread >> 4U;
  const uint32_t local_column = thread & 15U;
  float accumulators[rows_per_thread][columns_per_thread] = {};
  const uint16_t *const table = Lut ? lut : nullptr;

  for (uint64_t base = 0U; base < k; base += kTileK) {
    constexpr uint32_t values_per_load = 4U;
    constexpr uint32_t loads_per_row = kTileK / values_per_load;
    for (uint32_t index = thread; index < kTileM * loads_per_row;
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
        decode_four<Lut>(packed, table, &activation_tile[row][inner]);
      } else {
#pragma unroll
        for (uint32_t lane = 0U; lane < values_per_load; ++lane) {
          activation_tile[row][inner + lane] =
              source_row < m && source_inner + lane < k
                  ? decode_fp8<Lut>(
                        __builtin_nontemporal_load(activation + source_row * k +
                                                   source_inner + lane),
                        table)
                  : UINT16_C(0);
        }
      }
    }
    for (uint32_t index = thread; index < kTileN * loads_per_row;
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
        decode_four<Lut>(packed, table, &weight_tile[column][inner]);
      } else {
#pragma unroll
        for (uint32_t lane = 0U; lane < values_per_load; ++lane) {
          weight_tile[column][inner + lane] =
              source_column < n && source_inner + lane < k
                  ? decode_fp8<Lut>(
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
            bf16_rne(accumulators[row][column] * activation_scales[output_row] *
                     weight_scales[output_column]);
      }
    }
  }
}

__global__ __launch_bounds__(kThreads, 1) void fp8_prefill_control(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  __shared__ __align__(16) uint16_t activation_tile[kTileM][kTileK + 2U];
  __shared__ __align__(16) uint16_t weight_tile[kTileN][kTileK + 2U];
  fp8_prefill_64x64_k32<false>(activation, activation_scales, weight,
                               weight_scales, output, m, k, n, activation_tile,
                               weight_tile, nullptr);
}

__global__ __launch_bounds__(kThreads, 1) void fp8_prefill_lut(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  __shared__ __align__(16) uint16_t activation_tile[kTileM][kTileK + 2U];
  __shared__ __align__(16) uint16_t weight_tile[kTileN][kTileK + 2U];
  __shared__ __align__(16) uint16_t lut[272U];
  fp8_prefill_64x64_k32<true>(activation, activation_scales, weight,
                              weight_scales, output, m, k, n, activation_tile,
                              weight_tile, lut);
}

template <bool Lut>
__global__ void code_conversion(const uint8_t *const input,
                                uint16_t *const output) {
  __shared__ uint16_t table[272U];
  if constexpr (Lut) {
    if (threadIdx.x < 256U)
      table[lut_slot(threadIdx.x)] = kFp8Lut[threadIdx.x];
    __syncthreads();
  }
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 256U) {
    output[index] = decode_fp8<Lut>(input[index], Lut ? table : nullptr);
  }
}

struct Candidate final {
  const char *name;
  const void *function;
  bool lut;
};

Candidate control() {
  return {"id71-control-64x64-k32",
          reinterpret_cast<const void *>(fp8_prefill_control), false};
}

Candidate candidate_lut() {
  return {"lds-lut-64x64-k32-v1",
          reinterpret_cast<const void *>(fp8_prefill_lut), true};
}

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "hip error operation=%s status=%s\n", operation,
               hipGetErrorString(status));
  return false;
}

uint16_t host_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
    if ((bits & UINT32_C(0x007fffff)) != 0U)
      return static_cast<uint16_t>(((bits >> 16U) & 0x8000U) | 0x7fc0U |
                                   ((bits >> 16U) & 0x003fU));
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U))
    ++upper;
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

float e4m3_to_float(const uint8_t code) {
  return fp16_to_float(sllm_lowp::e4m3fn_to_fp16_bits(code));
}

struct Buffers final {
  uint8_t *activation = nullptr;
  float *activation_scales = nullptr;
  uint8_t *weight = nullptr;
  float *weight_scales = nullptr;
  uint16_t *output = nullptr;
  uint64_t output_words = 0U;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
};

void free_buffers(Buffers *const b) {
  if (b == nullptr)
    return;
  if (b->stop != nullptr)
    (void)hipEventDestroy(b->stop);
  if (b->start != nullptr)
    (void)hipEventDestroy(b->start);
  if (b->stream != nullptr)
    (void)hipStreamDestroy(b->stream);
  if (b->output != nullptr)
    (void)hipFree(b->output);
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
  if (b == nullptr || m == 0U || k == 0U || n == 0U || m > SIZE_MAX / k ||
      n > SIZE_MAX / k || m > SIZE_MAX / n)
    return false;
  const uint64_t output_words = m * n + kGuardWords;
  b->output_words = output_words;
  return hip_ok(hipMalloc(reinterpret_cast<void **>(&b->activation),
                          static_cast<size_t>(m * k)),
                "malloc activation") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->activation_scales),
                          static_cast<size_t>(m * sizeof(float))),
                "malloc activation scales") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->weight),
                          static_cast<size_t>(n * k)),
                "malloc weight") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->weight_scales),
                          static_cast<size_t>(n * sizeof(float))),
                "malloc weight scales") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->output),
                          static_cast<size_t>(output_words * sizeof(uint16_t))),
                "malloc output") &&
         hip_ok(hipStreamCreate(&b->stream), "create stream") &&
         hip_ok(hipEventCreate(&b->start), "create start") &&
         hip_ok(hipEventCreate(&b->stop), "create stop");
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
    (*activation_scales)[static_cast<size_t>(row)] =
        0.75F + static_cast<float>(row % 13U) * 0.03125F;
    for (uint64_t inner = 0U; inner < k; ++inner) {
      uint32_t code =
          static_cast<uint32_t>((row * 37U + inner * 11U + 5U) & 255U);
      if (code == 0x7fU || code == 0xffU)
        code = 0x7eU;
      (*activation)[static_cast<size_t>(row * k + inner)] =
          static_cast<uint8_t>(code);
    }
  }
  for (uint64_t column = 0U; column < n; ++column) {
    (*weight_scales)[static_cast<size_t>(column)] =
        0.625F + static_cast<float>(column % 17U) * 0.0234375F;
    for (uint64_t inner = 0U; inner < k; ++inner) {
      uint32_t code =
          static_cast<uint32_t>((column * 19U + inner * 7U + 13U) & 255U);
      if (code == 0x7fU || code == 0xffU)
        code = 0x7eU;
      (*weight)[static_cast<size_t>(column * k + inner)] =
          static_cast<uint8_t>(code);
    }
  }
}

bool upload(const uint64_t m, const uint64_t k, const uint64_t n,
            const std::vector<uint8_t> &activation,
            const std::vector<uint8_t> &weight,
            const std::vector<float> &activation_scales,
            const std::vector<float> &weight_scales, Buffers *const b) {
  return hip_ok(hipMemcpy(b->activation, activation.data(), m * k,
                          hipMemcpyHostToDevice),
                "copy activation") &&
         hip_ok(hipMemcpy(b->activation_scales, activation_scales.data(),
                          m * sizeof(float), hipMemcpyHostToDevice),
                "copy activation scales") &&
         hip_ok(
             hipMemcpy(b->weight, weight.data(), n * k, hipMemcpyHostToDevice),
             "copy weight") &&
         hip_ok(hipMemcpy(b->weight_scales, weight_scales.data(),
                          n * sizeof(float), hipMemcpyHostToDevice),
                "copy weight scales") &&
         hip_ok(hipMemset(b->output, 0xa5, b->output_words * sizeof(uint16_t)),
                "clear output guard");
}

bool launch(const Candidate &c, const uint64_t m, const uint64_t k,
            const uint64_t n, Buffers *const b) {
  const uint64_t rows = (m + kTileM - 1U) / kTileM;
  const uint64_t columns = (n + kTileN - 1U) / kTileN;
  const uint64_t blocks = rows * columns;
  if (blocks == 0U || blocks > UINT32_MAX)
    return false;
  if (c.lut) {
    hipLaunchKernelGGL((fp8_prefill_lut), dim3(static_cast<uint32_t>(blocks)),
                       dim3(kThreads), 0U, b->stream, b->activation,
                       b->activation_scales, b->weight, b->weight_scales,
                       b->output, m, k, n);
  } else {
    hipLaunchKernelGGL((fp8_prefill_control),
                       dim3(static_cast<uint32_t>(blocks)), dim3(kThreads), 0U,
                       b->stream, b->activation, b->activation_scales,
                       b->weight, b->weight_scales, b->output, m, k, n);
  }
  return hip_ok(hipGetLastError(), "launch prefill");
}

bool measure(const Candidate &c, const uint64_t m, const uint64_t k,
             const uint64_t n, Buffers *const b, float *const median_us) {
  for (int i = 0; i < kWarmups; ++i) {
    if (!launch(c, m, k, n, b) ||
        !hip_ok(hipStreamSynchronize(b->stream), "warmup sync"))
      return false;
  }
  std::array<float, kMeasured> samples{};
  for (int i = 0; i < kMeasured; ++i) {
    if (!hip_ok(hipEventRecord(b->start, b->stream), "event start") ||
        !launch(c, m, k, n, b) ||
        !hip_ok(hipEventRecord(b->stop, b->stream), "event stop") ||
        !hip_ok(hipEventSynchronize(b->stop), "event sync") ||
        !hip_ok(hipEventElapsedTime(&samples[static_cast<size_t>(i)], b->start,
                                    b->stop),
                "event elapsed"))
      return false;
    samples[static_cast<size_t>(i)] *= 1000.0F;
  }
  std::sort(samples.begin(), samples.end());
  *median_us = samples[kMeasured / 2];
  return true;
}

bool copy_rows(const uint64_t m, const uint64_t n, const Buffers &b,
               std::vector<uint16_t> *const rows) {
  const uint64_t count = std::min<uint64_t>(m, 4U) * n;
  rows->resize(static_cast<size_t>(count));
  return hip_ok(hipMemcpy(rows->data(), b.output,
                          static_cast<size_t>(count * sizeof(uint16_t)),
                          hipMemcpyDeviceToHost),
                "copy oracle rows");
}

bool oracle_compare(const uint64_t m, const uint64_t k, const uint64_t n,
                    const std::vector<uint8_t> &activation,
                    const std::vector<uint8_t> &weight,
                    const std::vector<float> &activation_scales,
                    const std::vector<float> &weight_scales,
                    const std::vector<uint16_t> &control,
                    const std::vector<uint16_t> &lut) {
  size_t oracle_mismatch = 0U;
  size_t candidate_mismatch = 0U;
  size_t ulp_max = 0U;
  const uint64_t rows = std::min<uint64_t>(m, 4U);
  for (uint64_t row = 0U; row < rows; ++row) {
    for (uint64_t column = 0U; column < n; ++column) {
      const size_t index = static_cast<size_t>(row * n + column);
      if (control[index] != lut[index])
        ++candidate_mismatch;
      if (column >= 4U)
        continue;
      float accumulator = 0.0F;
      for (uint64_t inner = 0U; inner < k; ++inner) {
        accumulator = std::fmaf(
            e4m3_to_float(activation[static_cast<size_t>(row * k + inner)]),
            e4m3_to_float(weight[static_cast<size_t>(column * k + inner)]),
            accumulator);
      }
      const uint16_t expected = host_bf16_rne(
          accumulator * activation_scales[static_cast<size_t>(row)] *
          weight_scales[static_cast<size_t>(column)]);
      const uint16_t observed = lut[index];
      const size_t ulp =
          expected > observed ? expected - observed : observed - expected;
      ulp_max = std::max(ulp_max, ulp);
      if (expected != observed)
        ++oracle_mismatch;
    }
  }
  std::printf(
      "oracle m=%llu k=%llu n=%llu sampled_rows=%llu sampled_cols=4 "
      "lut_vs_control_mismatches=%zu oracle_mismatches=%zu max_bf16_ulp=%zu "
      "status=%s\n",
      static_cast<unsigned long long>(m), static_cast<unsigned long long>(k),
      static_cast<unsigned long long>(n), static_cast<unsigned long long>(rows),
      candidate_mismatch, oracle_mismatch, ulp_max,
      candidate_mismatch == 0U && oracle_mismatch == 0U ? "PASS" : "FAIL");
  return candidate_mismatch == 0U && oracle_mismatch == 0U;
}

bool check_codes() {
  uint8_t *input = nullptr;
  uint16_t *control = nullptr;
  uint16_t *lut = nullptr;
  std::array<uint8_t, 256> host_codes{};
  for (uint32_t i = 0U; i < 256U; ++i)
    host_codes[i] = static_cast<uint8_t>(i);
  const bool allocated =
      hip_ok(hipMalloc(reinterpret_cast<void **>(&input), 256U),
             "codes input") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&control), 512U),
             "codes control") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&lut), 512U), "codes lut");
  if (!allocated) {
    if (input != nullptr)
      (void)hipFree(input);
    if (control != nullptr)
      (void)hipFree(control);
    if (lut != nullptr)
      (void)hipFree(lut);
    return false;
  }
  bool ok =
      hip_ok(hipMemcpy(input, host_codes.data(), 256U, hipMemcpyHostToDevice),
             "copy codes");
  if (ok) {
    hipLaunchKernelGGL((code_conversion<false>), dim3(1U), dim3(256U), 0U,
                       nullptr, input, control);
    ok = hip_ok(hipGetLastError(), "launch control codes") &&
         hip_ok(hipDeviceSynchronize(), "sync control codes");
  }
  if (ok) {
    hipLaunchKernelGGL((code_conversion<true>), dim3(1U), dim3(256U), 0U,
                       nullptr, input, lut);
    ok = hip_ok(hipGetLastError(), "launch lut codes") &&
         hip_ok(hipDeviceSynchronize(), "sync lut codes");
  }
  std::array<uint16_t, 256> control_host{};
  std::array<uint16_t, 256> lut_host{};
  ok = ok &&
       hip_ok(
           hipMemcpy(control_host.data(), control, 512U, hipMemcpyDeviceToHost),
           "copy control codes") &&
       hip_ok(hipMemcpy(lut_host.data(), lut, 512U, hipMemcpyDeviceToHost),
              "copy lut codes");
  size_t control_mismatch = 0U;
  size_t lut_mismatch = 0U;
  size_t lut_control_mismatch = 0U;
  for (uint32_t i = 0U; i < 256U; ++i) {
    if (control_host[i] != sllm_lowp::e4m3fn_to_fp16_bits(host_codes[i]))
      ++control_mismatch;
    if (lut_host[i] != sllm_lowp::e4m3fn_to_fp16_bits(host_codes[i]))
      ++lut_mismatch;
    if (control_host[i] != lut_host[i])
      ++lut_control_mismatch;
  }
  std::printf("oracle ingress codes=256 control_mismatches=%zu "
              "lut_mismatches=%zu control_lut_mismatches=%zu status=%s\n",
              control_mismatch, lut_mismatch, lut_control_mismatch,
              ok && control_mismatch == 0U && lut_mismatch == 0U &&
                      lut_control_mismatch == 0U
                  ? "PASS"
                  : "FAIL");
  (void)hipFree(lut);
  (void)hipFree(control);
  (void)hipFree(input);
  return ok && control_mismatch == 0U && lut_mismatch == 0U &&
         lut_control_mismatch == 0U;
}

void print_resources(const Candidate &c) {
  hipFuncAttributes attr{};
  const hipError_t attr_status = hipFuncGetAttributes(&attr, c.function);
  int active_blocks = 0;
  const hipError_t occupancy_status =
      hipOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, c.function,
                                                   kThreads, 0U);
  std::printf(
      "resources candidate=%s vgpr=%d lds=%zu scratch=%zu max_threads=%d "
      "active_blocks=%d attr=%s occupancy=%s\n",
      c.name, attr.numRegs, attr.sharedSizeBytes, attr.localSizeBytes,
      attr.maxThreadsPerBlock, active_blocks, hipGetErrorString(attr_status),
      hipGetErrorString(occupancy_status));
}

bool exact_gfx1030(const char *const arch) {
  if (arch == nullptr)
    return false;
  const std::string_view value(arch);
  return value == "gfx1030" ||
         (value.size() > 7U && value.compare(0U, 7U, "gfx1030") == 0 &&
          value[7U] == ':');
}

struct Shape final {
  uint64_t m;
  uint64_t k;
  uint64_t n;
  const char *name;
  uint64_t calls;
};

bool run_shape(const Shape shape, const Candidate &control_candidate,
               const Candidate &lut_candidate, double *const weighted_control,
               double *const weighted_lut, double *const model_control,
               double *const model_lut) {
  std::vector<uint8_t> activation;
  std::vector<uint8_t> weight;
  std::vector<float> activation_scales;
  std::vector<float> weight_scales;
  fill_inputs(shape.m, shape.k, shape.n, &activation, &weight,
              &activation_scales, &weight_scales);
  Buffers buffers;
  if (!make_buffers(shape.m, shape.k, shape.n, &buffers) ||
      !upload(shape.m, shape.k, shape.n, activation, weight, activation_scales,
              weight_scales, &buffers)) {
    free_buffers(&buffers);
    return false;
  }
  float control_us = 0.0F;
  float lut_us = 0.0F;
  if (!measure(control_candidate, shape.m, shape.k, shape.n, &buffers,
               &control_us)) {
    free_buffers(&buffers);
    return false;
  }
  std::vector<uint16_t> control_output;
  if (!copy_rows(shape.m, shape.n, buffers, &control_output) ||
      !measure(lut_candidate, shape.m, shape.k, shape.n, &buffers, &lut_us)) {
    free_buffers(&buffers);
    return false;
  }
  std::vector<uint16_t> lut_output;
  if (!copy_rows(shape.m, shape.n, buffers, &lut_output)) {
    free_buffers(&buffers);
    return false;
  }
  bool ok = true;
  const size_t guard_start = static_cast<size_t>(shape.m * shape.n);
  std::vector<uint16_t> guard(kGuardWords, 0U);
  ok = ok &&
       hip_ok(hipMemcpy(guard.data(), buffers.output + guard_start,
                        kGuardWords * sizeof(uint16_t), hipMemcpyDeviceToHost),
              "copy output guard");
  bool guard_ok = true;
  for (const uint16_t word : guard)
    guard_ok = guard_ok && word == 0xa5a5U;
  ok = ok && guard_ok;
  ok = ok && oracle_compare(shape.m, shape.k, shape.n, activation, weight,
                            activation_scales, weight_scales, control_output,
                            lut_output);
  // A second candidate submission verifies deterministic output independent of
  // the event timing path and catches accidental uninitialized LDS use.
  std::vector<uint16_t> repeat_output;
  ok = ok && launch(lut_candidate, shape.m, shape.k, shape.n, &buffers) &&
       hip_ok(hipStreamSynchronize(buffers.stream), "determinism sync") &&
       copy_rows(shape.m, shape.n, buffers, &repeat_output) &&
       repeat_output == lut_output;
  const double bytes =
      static_cast<double>(shape.m * shape.k + shape.n * shape.k);
  const double speedup = static_cast<double>(control_us) /
                         std::max(1.0, static_cast<double>(lut_us));
  const double control_gbps = bytes / static_cast<double>(control_us) / 1000.0;
  const double lut_gbps = bytes / static_cast<double>(lut_us) / 1000.0;
  std::printf(
      "result shape=%s m=%llu k=%llu n=%llu calls=%llu control_us=%.3f "
      "lut_us=%.3f speedup=%.6f control_gbps=%.3f lut_gbps=%.3f guard=%s "
      "determinism=%s status=%s\n",
      shape.name, static_cast<unsigned long long>(shape.m),
      static_cast<unsigned long long>(shape.k),
      static_cast<unsigned long long>(shape.n),
      static_cast<unsigned long long>(shape.calls), control_us, lut_us, speedup,
      control_gbps, lut_gbps, guard_ok ? "PASS" : "FAIL",
      repeat_output == lut_output ? "PASS" : "FAIL", ok ? "PASS" : "FAIL");
  *weighted_control +=
      static_cast<double>(control_us) * static_cast<double>(shape.calls);
  *weighted_lut +=
      static_cast<double>(lut_us) * static_cast<double>(shape.calls);
  if (shape.m == 1024U) {
    *model_control +=
        static_cast<double>(control_us) * static_cast<double>(shape.calls);
    *model_lut +=
        static_cast<double>(lut_us) * static_cast<double>(shape.calls);
  }
  free_buffers(&buffers);
  return ok;
}

} // namespace

int main() {
  int device = 0;
  if (!hip_ok(hipSetDevice(device), "set device"))
    return EXIT_FAILURE;
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device),
              "device properties") ||
      !exact_gfx1030(properties.gcnArchName)) {
    std::fprintf(stderr, "exact gfx1030 required; got %s\n",
                 properties.gcnArchName);
    return EXIT_FAILURE;
  }
  std::printf(
      "identity target=%s logical_device=%d pci=%04x:%02x:%02x name=%s\n",
      properties.gcnArchName, device, properties.pciDomainID,
      properties.pciBusID, properties.pciDeviceID, properties.name);

  std::array<uint16_t, 256> host_lut{};
  for (uint32_t i = 0U; i < 256U; ++i)
    host_lut[i] = sllm_lowp::e4m3fn_to_fp16_bits(static_cast<uint8_t>(i));
  if (!hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(kFp8Lut), host_lut.data(),
                                sizeof(host_lut)),
              "copy FP8 LUT") ||
      !check_codes())
    return EXIT_FAILURE;

  const Candidate control_candidate = control();
  const Candidate lut_candidate = candidate_lut();
  print_resources(control_candidate);
  print_resources(lut_candidate);

  constexpr std::array<Shape, 9> shapes = {
      Shape{128U, 5120U, 17408U, "wide-m128", 112U},
      Shape{512U, 5120U, 17408U, "wide-m512", 112U},
      Shape{1024U, 5120U, 17408U, "wide-m1024", 112U},
      Shape{128U, 17408U, 5120U, "down-m128", 56U},
      Shape{512U, 17408U, 5120U, "down-m512", 56U},
      Shape{1024U, 17408U, 5120U, "down-m1024", 56U},
      Shape{128U, 5120U, 248320U, "lm-head-m128", 1U},
      Shape{512U, 5120U, 248320U, "lm-head-m512", 1U},
      Shape{1024U, 5120U, 248320U, "lm-head-m1024", 1U}};
  bool ok = true;
  double weighted_control = 0.0;
  double weighted_lut = 0.0;
  double model_control = 0.0;
  double model_lut = 0.0;
  for (const Shape shape : shapes) {
    ok = run_shape(shape, control_candidate, lut_candidate, &weighted_control,
                   &weighted_lut, &model_control, &model_lut) &&
         ok;
  }

  // Non-aligned M/K/N exercises the scalar ingress and all output guards.
  const Shape boundary{17U, 70U, 31U, "boundary-m17-k70-n31", 1U};
  ok = run_shape(boundary, control_candidate, lut_candidate, &weighted_control,
                 &weighted_lut, &model_control, &model_lut) &&
       ok;
  const double weighted_speedup =
      weighted_control / std::max(1.0, weighted_lut);
  const double model_speedup = model_control / std::max(1.0, model_lut);
  std::printf(
      "summary weighted_control_us=%.3f weighted_lut_us=%.3f "
      "weighted_speedup=%.6f model_m1024_control_us=%.3f "
      "model_m1024_lut_us=%.3f model_m1024_speedup=%.6f warmups=%d measured=%d "
      "status=%s cleanup=0\n",
      weighted_control, weighted_lut, weighted_speedup, model_control,
      model_lut, model_speedup, kWarmups, kMeasured, ok ? "PASS" : "FAIL");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
