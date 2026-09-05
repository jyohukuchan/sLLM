// Phase 78 gfx1030 transient FP16 ingress probe.
//
// The control is the current production ID71 64x64/K32 half2 kernel.  The
// candidate converts each resident FP8 matrix to a transient FP16 workspace
// once, then consumes that workspace with the same tile, K order, half2 dot
// operation, and BF16-RNE epilogue.  Candidate timing includes both staging
// launches and the consumer; the workspace size is printed with every case.
// This file is a standalone evidence probe and does not change production
// dispatch or allocate a resident workspace.

#include "low_precision_block_codec.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

// The probe links this symbol against the current production ID71 archive.
// Keeping it external prevents this experiment from silently comparing two
// copies compiled from different production sources.
extern "C" __global__ void sllm_matmul_fp8_outer_prefill_gfx1030_half2_64x64_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n);

namespace {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kTileM = 64U;
constexpr uint32_t kTileN = 64U;
constexpr uint32_t kTileK = 32U;
constexpr uint32_t kLdsStride = kTileK + 2U;
constexpr uint32_t kThreadRows = 16U;
constexpr uint32_t kThreadColumns = 16U;
constexpr uint32_t kValuesPerPackedLoad = 4U;
constexpr uint32_t kGuardWords = 64U;
constexpr uint16_t kGuardWord = UINT16_C(0xa5a5);
constexpr int kWarmups = 3;
constexpr int kMeasured = 10;

__device__ __forceinline__ uint16_t
staged_bf16_rne(const float value) noexcept {
  const uint32_t bits = __float_as_uint(value);
  constexpr uint32_t exponent_mask = UINT32_C(0x7f800000);
  constexpr uint32_t fraction_mask = UINT32_C(0x007fffff);
  if ((bits & exponent_mask) == exponent_mask) {
    if ((bits & fraction_mask) != 0U) {
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

// Each thread owns one four-byte chunk where possible.  The scalar tail is
// required for odd K and for the final partial row, and uses the same exact
// FP8-to-FP16 conversion as ID71.  This kernel is intentionally independent
// of the consumer so its cost can be included explicitly in total timing.
__global__ __launch_bounds__(kThreads, 1) void fp8_to_f16_transient_kernel(
    const uint8_t *const input, uint16_t *const output,
    const uint64_t element_count) {
  const uint64_t chunk =
      (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kValuesPerPackedLoad;
  if (chunk >= element_count) {
    return;
  }
  if (chunk + kValuesPerPackedLoad <= element_count &&
      (reinterpret_cast<uintptr_t>(input + chunk) &
       static_cast<uintptr_t>(3U)) == 0U) {
    const uint32_t packed = __builtin_nontemporal_load(
        reinterpret_cast<const uint32_t *>(input + chunk));
    const sllm_lowp::E4M3FnFp16x4Bits expanded =
        sllm_lowp::e4m3fnx4_to_fp16x2_bits(packed);
    auto *const destination = reinterpret_cast<uint32_t *>(output + chunk);
    destination[0] = expanded.low;
    destination[1] = expanded.high;
    return;
  }
#pragma unroll
  for (uint32_t lane = 0U; lane < kValuesPerPackedLoad; ++lane) {
    const uint64_t index = chunk + lane;
    if (index < element_count) {
      output[index] = sllm_lowp::e4m3fn_to_fp16_bits(
          __builtin_nontemporal_load(input + index));
    }
  }
}

// This is the ID71 body after ingress.  Its only input difference is that
// tile values are copied directly from the transient FP16 matrices rather
// than decoded from FP8 bytes.  The 64x64/K32 LDS layout, half2 load order,
// amd_mixed_dot order, barriers, scales, and epilogue intentionally match
// the current production kernel.
template <bool PackedK>
__global__ __launch_bounds__(
    kThreads,
    1) void sllm_phase78_fp8_f16_tile_staging_consumer_v1(const void
                                                              *const activation,
                                                          const float *const
                                                              activation_scales,
                                                          const void
                                                              *const weight,
                                                          const float *const
                                                              weight_scales,
                                                          uint16_t
                                                              *const output,
                                                          const uint64_t m,
                                                          const uint64_t k,
                                                          const uint64_t n) {
  constexpr uint32_t rows_per_thread = kTileM / kThreadRows;
  constexpr uint32_t columns_per_thread = kTileN / kThreadColumns;
  __shared__ __align__(4) uint16_t activation_tile[kTileM][kLdsStride];
  __shared__ __align__(4) uint16_t weight_tile[kTileN][kLdsStride];

  const uint64_t column_tiles = (n + kTileN - 1U) / kTileN;
  const uint64_t tile_index = static_cast<uint64_t>(blockIdx.x);
  const uint64_t row_base = (tile_index / column_tiles) * kTileM;
  const uint64_t column_base = (tile_index % column_tiles) * kTileN;
  const uint32_t thread = threadIdx.x;
  const uint32_t local_row = thread >> 4U;
  const uint32_t local_column = thread & UINT32_C(15);
  float accumulators[rows_per_thread][columns_per_thread] = {};

  for (uint64_t base = 0U; base < k; base += kTileK) {
    constexpr uint32_t loads_per_row = kTileK / kValuesPerPackedLoad;
    for (uint32_t index = thread; index < kTileM * loads_per_row;
         index += kThreads) {
      const uint32_t row = index / loads_per_row;
      const uint32_t inner = (index % loads_per_row) * kValuesPerPackedLoad;
      const uint64_t source_row = row_base + row;
      const uint64_t source_inner = base + inner;
      uint16_t *const destination = &activation_tile[row][inner];
      if (source_row < m && source_inner < k) {
        const uint16_t *const source =
            reinterpret_cast<const uint16_t *>(activation) + source_row * k +
            source_inner;
        const uint32_t *const packed_source =
            reinterpret_cast<const uint32_t *>(activation) +
            source_row * (k / 2U) + source_inner / 2U;
        if constexpr (PackedK) {
          for (uint32_t lane = 0U; lane < kValuesPerPackedLoad; ++lane) {
            const uint64_t source_index = source_inner + lane;
            if (source_index < k) {
              const uint32_t packed = packed_source[lane / 2U];
              destination[lane] =
                  static_cast<uint16_t>(packed >> ((lane & 1U) * 16U));
            } else {
              destination[lane] = UINT16_C(0);
            }
          }
        } else {
#pragma unroll
          for (uint32_t lane = 0U; lane < kValuesPerPackedLoad; ++lane) {
            const uint64_t source_index = source_inner + lane;
            destination[lane] = source_index < k
                                    ? __builtin_nontemporal_load(source + lane)
                                    : UINT16_C(0);
          }
        }
      } else {
#pragma unroll
        for (uint32_t lane = 0U; lane < kValuesPerPackedLoad; ++lane)
          destination[lane] = UINT16_C(0);
      }
    }
    for (uint32_t index = thread; index < kTileN * loads_per_row;
         index += kThreads) {
      const uint32_t column = index / loads_per_row;
      const uint32_t inner = (index % loads_per_row) * kValuesPerPackedLoad;
      const uint64_t source_column = column_base + column;
      const uint64_t source_inner = base + inner;
      uint16_t *const destination = &weight_tile[column][inner];
      if (source_column < n && source_inner < k) {
        const uint16_t *const source =
            reinterpret_cast<const uint16_t *>(weight) + source_column * k +
            source_inner;
        const uint32_t *const packed_source =
            reinterpret_cast<const uint32_t *>(weight) +
            source_column * (k / 2U) + source_inner / 2U;
        if constexpr (PackedK) {
          for (uint32_t lane = 0U; lane < kValuesPerPackedLoad; ++lane) {
            const uint64_t source_index = source_inner + lane;
            if (source_index < k) {
              const uint32_t packed = packed_source[lane / 2U];
              destination[lane] =
                  static_cast<uint16_t>(packed >> ((lane & 1U) * 16U));
            } else {
              destination[lane] = UINT16_C(0);
            }
          }
        } else {
#pragma unroll
          for (uint32_t lane = 0U; lane < kValuesPerPackedLoad; ++lane) {
            const uint64_t source_index = source_inner + lane;
            destination[lane] = source_index < k
                                    ? __builtin_nontemporal_load(source + lane)
                                    : UINT16_C(0);
          }
        }
      } else {
#pragma unroll
        for (uint32_t lane = 0U; lane < kValuesPerPackedLoad; ++lane)
          destination[lane] = UINT16_C(0);
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
        output[output_row * n + output_column] = staged_bf16_rne(
            accumulators[row][column] * activation_scales[output_row] *
            weight_scales[output_column]);
      }
    }
  }
}

struct HostInputs final {
  std::vector<uint8_t> activation;
  std::vector<uint8_t> weight;
  std::vector<float> activation_scales;
  std::vector<float> weight_scales;
};

struct DeviceBuffers final {
  uint8_t *activation = nullptr;
  uint8_t *weight = nullptr;
  float *activation_scales = nullptr;
  float *weight_scales = nullptr;
  uint16_t *activation_f16 = nullptr;
  uint16_t *weight_f16 = nullptr;
  uint16_t *production_output = nullptr;
  uint16_t *staged_output = nullptr;
  uint64_t output_words = 0U;
  uint64_t transient_bytes = 0U;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
};

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "hip error operation=%s status=%s\n", operation,
               hipGetErrorString(status));
  return false;
}

void release(DeviceBuffers *const buffers) {
  if (buffers == nullptr)
    return;
  if (buffers->stop != nullptr)
    (void)hipEventDestroy(buffers->stop);
  if (buffers->start != nullptr)
    (void)hipEventDestroy(buffers->start);
  if (buffers->stream != nullptr)
    (void)hipStreamDestroy(buffers->stream);
  if (buffers->staged_output != nullptr)
    (void)hipFree(buffers->staged_output);
  if (buffers->production_output != nullptr)
    (void)hipFree(buffers->production_output);
  if (buffers->weight_f16 != nullptr)
    (void)hipFree(buffers->weight_f16);
  if (buffers->activation_f16 != nullptr)
    (void)hipFree(buffers->activation_f16);
  if (buffers->weight_scales != nullptr)
    (void)hipFree(buffers->weight_scales);
  if (buffers->activation_scales != nullptr)
    (void)hipFree(buffers->activation_scales);
  if (buffers->weight != nullptr)
    (void)hipFree(buffers->weight);
  if (buffers->activation != nullptr)
    (void)hipFree(buffers->activation);
  *buffers = {};
}

bool allocate(const uint64_t m, const uint64_t k, const uint64_t n,
              DeviceBuffers *const buffers) {
  if (buffers == nullptr || m == 0U || k == 0U || n == 0U ||
      m > UINT64_MAX / k || n > UINT64_MAX / k || m > UINT64_MAX / n)
    return false;
  const uint64_t activation_elements = m * k;
  const uint64_t weight_elements = n * k;
  const uint64_t output_elements = m * n;
  if (activation_elements > SIZE_MAX / sizeof(uint16_t) ||
      weight_elements > SIZE_MAX / sizeof(uint16_t) ||
      output_elements > UINT64_MAX - kGuardWords ||
      (output_elements + kGuardWords) > SIZE_MAX / sizeof(uint16_t))
    return false;
  buffers->output_words = output_elements + kGuardWords;
  buffers->transient_bytes =
      (activation_elements + weight_elements) * sizeof(uint16_t);
  const bool ok =
      hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->activation),
                       static_cast<size_t>(activation_elements)),
             "malloc activation") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->weight),
                       static_cast<size_t>(weight_elements)),
             "malloc weight") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->activation_scales),
                       static_cast<size_t>(m * sizeof(float))),
             "malloc activation scales") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->weight_scales),
                       static_cast<size_t>(n * sizeof(float))),
             "malloc weight scales") &&
      hip_ok(hipMalloc(
                 reinterpret_cast<void **>(&buffers->activation_f16),
                 static_cast<size_t>(activation_elements * sizeof(uint16_t))),
             "malloc activation transient") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->weight_f16),
                       static_cast<size_t>(weight_elements * sizeof(uint16_t))),
             "malloc weight transient") &&
      hip_ok(hipMalloc(
                 reinterpret_cast<void **>(&buffers->production_output),
                 static_cast<size_t>(buffers->output_words * sizeof(uint16_t))),
             "malloc production output") &&
      hip_ok(hipMalloc(
                 reinterpret_cast<void **>(&buffers->staged_output),
                 static_cast<size_t>(buffers->output_words * sizeof(uint16_t))),
             "malloc staged output") &&
      hip_ok(hipStreamCreate(&buffers->stream), "create stream") &&
      hip_ok(hipEventCreate(&buffers->start), "create start") &&
      hip_ok(hipEventCreate(&buffers->stop), "create stop");
  if (!ok)
    release(buffers);
  return ok;
}

uint8_t finite_code(const uint64_t value) {
  const uint8_t code = static_cast<uint8_t>(value & UINT64_C(0xff));
  return code == UINT8_C(0x7f) || code == UINT8_C(0xff) ? UINT8_C(0x7e) : code;
}

bool fill_inputs(const uint64_t m, const uint64_t k, const uint64_t n,
                 const uint32_t seed, HostInputs *const inputs) {
  if (inputs == nullptr || m > SIZE_MAX / k || n > SIZE_MAX / k)
    return false;
  inputs->activation.resize(static_cast<size_t>(m * k));
  inputs->weight.resize(static_cast<size_t>(n * k));
  inputs->activation_scales.resize(static_cast<size_t>(m));
  inputs->weight_scales.resize(static_cast<size_t>(n));
  for (uint64_t row = 0U; row < m; ++row) {
    inputs->activation_scales[row] =
        0.5F + static_cast<float>((row + seed) % 7U) * 0.125F;
    for (uint64_t inner = 0U; inner < k; ++inner) {
      inputs->activation[static_cast<size_t>(row * k + inner)] = finite_code(
          row * UINT64_C(37) + inner * UINT64_C(11) + seed * UINT64_C(17) + 5U);
    }
  }
  for (uint64_t column = 0U; column < n; ++column) {
    inputs->weight_scales[column] =
        0.625F + static_cast<float>((column + seed) % 9U) * 0.09375F;
    for (uint64_t inner = 0U; inner < k; ++inner) {
      inputs->weight[static_cast<size_t>(column * k + inner)] =
          finite_code(column * UINT64_C(19) + inner * UINT64_C(7) +
                      seed * UINT64_C(29) + 13U);
    }
  }
  return true;
}

bool upload(const uint64_t m, const uint64_t k, const uint64_t n,
            const HostInputs &inputs, DeviceBuffers *const buffers) {
  return hip_ok(hipMemcpy(buffers->activation, inputs.activation.data(),
                          static_cast<size_t>(m * k), hipMemcpyHostToDevice),
                "copy activation") &&
         hip_ok(hipMemcpy(buffers->weight, inputs.weight.data(),
                          static_cast<size_t>(n * k), hipMemcpyHostToDevice),
                "copy weight") &&
         hip_ok(hipMemcpy(buffers->activation_scales,
                          inputs.activation_scales.data(),
                          static_cast<size_t>(m * sizeof(float)),
                          hipMemcpyHostToDevice),
                "copy activation scales") &&
         hip_ok(hipMemcpy(buffers->weight_scales, inputs.weight_scales.data(),
                          static_cast<size_t>(n * sizeof(float)),
                          hipMemcpyHostToDevice),
                "copy weight scales");
}

uint32_t grid_for_elements(const uint64_t elements) {
  const uint64_t chunks =
      (elements + kValuesPerPackedLoad - 1U) / kValuesPerPackedLoad;
  return static_cast<uint32_t>((chunks + kThreads - 1U) / kThreads);
}

bool launch_stage(const uint8_t *const input, uint16_t *const output,
                  const uint64_t elements, hipStream_t stream) {
  const uint64_t chunks =
      (elements + kValuesPerPackedLoad - 1U) / kValuesPerPackedLoad;
  if (chunks == 0U || chunks > static_cast<uint64_t>(UINT32_MAX) * kThreads)
    return false;
  hipLaunchKernelGGL(fp8_to_f16_transient_kernel,
                     dim3(grid_for_elements(elements)), dim3(kThreads), 0U,
                     stream, input, output, elements);
  return hip_ok(hipGetLastError(), "launch transient FP16 stage");
}

uint32_t tile_count(const uint64_t m, const uint64_t n) {
  const uint64_t rows = (m + kTileM - 1U) / kTileM;
  const uint64_t columns = (n + kTileN - 1U) / kTileN;
  if (rows == 0U || columns == 0U || rows > UINT64_MAX / columns ||
      rows * columns > UINT32_MAX)
    return 0U;
  return static_cast<uint32_t>(rows * columns);
}

bool launch_production(const uint64_t m, const uint64_t k, const uint64_t n,
                       DeviceBuffers *const buffers) {
  const uint32_t tiles = tile_count(m, n);
  if (tiles == 0U)
    return false;
  hipLaunchKernelGGL(sllm_matmul_fp8_outer_prefill_gfx1030_half2_64x64_v1,
                     dim3(tiles), dim3(kThreads), 0U, buffers->stream,
                     buffers->activation, buffers->activation_scales,
                     buffers->weight, buffers->weight_scales,
                     buffers->production_output, m, k, n);
  return hip_ok(hipGetLastError(), "launch current ID71");
}

bool launch_staged(const uint64_t m, const uint64_t k, const uint64_t n,
                   DeviceBuffers *const buffers) {
  const uint64_t activation_elements = m * k;
  const uint64_t weight_elements = n * k;
  if (!launch_stage(buffers->activation, buffers->activation_f16,
                    activation_elements, buffers->stream) ||
      !launch_stage(buffers->weight, buffers->weight_f16, weight_elements,
                    buffers->stream))
    return false;
  const uint32_t tiles = tile_count(m, n);
  if (tiles == 0U)
    return false;
  if ((k % kValuesPerPackedLoad) == 0U) {
    hipLaunchKernelGGL((sllm_phase78_fp8_f16_tile_staging_consumer_v1<true>),
                       dim3(tiles), dim3(kThreads), 0U, buffers->stream,
                       static_cast<const void *>(buffers->activation_f16),
                       buffers->activation_scales,
                       static_cast<const void *>(buffers->weight_f16),
                       buffers->weight_scales, buffers->staged_output, m, k, n);
  } else {
    hipLaunchKernelGGL((sllm_phase78_fp8_f16_tile_staging_consumer_v1<false>),
                       dim3(tiles), dim3(kThreads), 0U, buffers->stream,
                       static_cast<const void *>(buffers->activation_f16),
                       buffers->activation_scales,
                       static_cast<const void *>(buffers->weight_f16),
                       buffers->weight_scales, buffers->staged_output, m, k, n);
  }
  return hip_ok(hipGetLastError(), "launch staged consumer");
}

bool synchronize(DeviceBuffers *const buffers, const char *const operation) {
  return hip_ok(hipStreamSynchronize(buffers->stream), operation);
}

bool clear_outputs(DeviceBuffers *const buffers) {
  return hip_ok(hipMemset(buffers->production_output, 0xa5,
                          static_cast<size_t>(buffers->output_words *
                                              sizeof(uint16_t))),
                "clear production output") &&
         hip_ok(hipMemset(buffers->staged_output, 0xa5,
                          static_cast<size_t>(buffers->output_words *
                                              sizeof(uint16_t))),
                "clear staged output");
}

bool measure(const uint64_t m, const uint64_t k, const uint64_t n,
             DeviceBuffers *const buffers, float *const production_us,
             float *const staged_us) {
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    if (!launch_production(m, k, n, buffers) ||
        !synchronize(buffers, "ID71 warmup") ||
        !launch_staged(m, k, n, buffers) ||
        !synchronize(buffers, "staged warmup"))
      return false;
  }
  std::array<float, kMeasured> production_samples{};
  std::array<float, kMeasured> staged_samples{};
  for (int iteration = 0; iteration < kMeasured; ++iteration) {
    // Alternate launch order so a steadily changing clock or thermal state
    // does not make one candidate permanently first or second.
    const bool production_first = (iteration & 1) == 0;
    auto measure_production = [&]() {
      return hip_ok(hipEventRecord(buffers->start, buffers->stream),
                    "production event start") &&
             launch_production(m, k, n, buffers) &&
             hip_ok(hipEventRecord(buffers->stop, buffers->stream),
                    "production event stop") &&
             hip_ok(hipEventSynchronize(buffers->stop),
                    "production event synchronize") &&
             hip_ok(hipEventElapsedTime(&production_samples[iteration],
                                        buffers->start, buffers->stop),
                    "production elapsed");
    };
    auto measure_staged = [&]() {
      return hip_ok(hipEventRecord(buffers->start, buffers->stream),
                    "staged event start") &&
             launch_staged(m, k, n, buffers) &&
             hip_ok(hipEventRecord(buffers->stop, buffers->stream),
                    "staged event stop") &&
             hip_ok(hipEventSynchronize(buffers->stop),
                    "staged event synchronize") &&
             hip_ok(hipEventElapsedTime(&staged_samples[iteration],
                                        buffers->start, buffers->stop),
                    "staged elapsed");
    };
    if ((production_first && (!measure_production() || !measure_staged())) ||
        (!production_first && (!measure_staged() || !measure_production())))
      return false;
  }
  std::sort(production_samples.begin(), production_samples.end());
  std::sort(staged_samples.begin(), staged_samples.end());
  *production_us = production_samples[kMeasured / 2] * 1000.0F;
  *staged_us = staged_samples[kMeasured / 2] * 1000.0F;
  return true;
}

bool copy_outputs(const uint64_t elements, DeviceBuffers *const buffers,
                  std::vector<uint16_t> *const production,
                  std::vector<uint16_t> *const staged) {
  production->resize(static_cast<size_t>(elements));
  staged->resize(static_cast<size_t>(elements));
  return hip_ok(hipMemcpy(production->data(), buffers->production_output,
                          static_cast<size_t>(elements * sizeof(uint16_t)),
                          hipMemcpyDeviceToHost),
                "copy production output") &&
         hip_ok(hipMemcpy(staged->data(), buffers->staged_output,
                          static_cast<size_t>(elements * sizeof(uint16_t)),
                          hipMemcpyDeviceToHost),
                "copy staged output");
}

bool check_guards(const DeviceBuffers &buffers) {
  std::array<uint16_t, kGuardWords> production{};
  std::array<uint16_t, kGuardWords> staged{};
  const size_t offset = static_cast<size_t>(buffers.output_words - kGuardWords);
  if (!hip_ok(hipMemcpy(production.data(), buffers.production_output + offset,
                        sizeof(production), hipMemcpyDeviceToHost),
              "copy production guard") ||
      !hip_ok(hipMemcpy(staged.data(), buffers.staged_output + offset,
                        sizeof(staged), hipMemcpyDeviceToHost),
              "copy staged guard"))
    return false;
  for (uint32_t index = 0U; index < kGuardWords; ++index) {
    if (production[index] != kGuardWord || staged[index] != kGuardWord) {
      std::printf(
          "guard status=FAIL index=%u production=0x%04x staged=0x%04x\n", index,
          production[index], staged[index]);
      return false;
    }
  }
  return true;
}

// Independent host FP8 decoder used only by the tiny oracle.  It does not
// call the device codec or consume the candidate's conversion table.
uint16_t host_float_to_half_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint16_t sign = static_cast<uint16_t>((bits >> 16U) & 0x8000U);
  const uint32_t exponent = (bits >> 23U) & 0xffU;
  uint32_t mantissa = bits & 0x7fffffU;
  if (exponent == 0xffU) {
    return static_cast<uint16_t>(sign | (mantissa == 0U ? 0x7c00U : 0x7e00U));
  }
  int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
  if (half_exponent >= 31)
    return static_cast<uint16_t>(sign | 0x7c00U);
  if (half_exponent <= 0) {
    if (half_exponent < -10)
      return sign;
    mantissa |= 0x800000U;
    const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);
    uint32_t result = mantissa >> shift;
    const uint32_t remainder = mantissa & ((UINT32_C(1) << shift) - 1U);
    const uint32_t halfway = UINT32_C(1) << (shift - 1U);
    if (remainder > halfway ||
        (remainder == halfway && (result & UINT32_C(1)) != 0U))
      ++result;
    return static_cast<uint16_t>(sign | result);
  }
  uint32_t result =
      (static_cast<uint32_t>(half_exponent) << 10U) | (mantissa >> 13U);
  const uint32_t remainder = mantissa & UINT32_C(0x1fff);
  if (remainder > UINT32_C(0x1000) ||
      (remainder == UINT32_C(0x1000) && (result & UINT32_C(1)) != 0U))
    ++result;
  return static_cast<uint16_t>(sign | result);
}

uint16_t host_fp8_to_half(const uint8_t code) {
  const uint32_t sign = static_cast<uint32_t>(code & 0x80U);
  const uint32_t magnitude = static_cast<uint32_t>(code & 0x7fU);
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  if (magnitude == 0x7fU)
    return static_cast<uint16_t>((sign << 8U) | 0x7e00U);
  const float value =
      exponent == 0U ? static_cast<float>(mantissa) * 0x1p-9F
                     : std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                                  static_cast<int>(exponent) - 7);
  const uint16_t result = host_float_to_half_rne(value);
  return static_cast<uint16_t>(result | (sign << 8U));
}

float host_half_to_float(const uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16U;
  const uint32_t exponent = (bits >> 10U) & 0x1fU;
  const uint32_t mantissa = bits & 0x3ffU;
  uint32_t result = sign;
  if (exponent == 0U) {
    if (mantissa == 0U)
      return [&] {
        float value = 0.0F;
        uint32_t raw = sign;
        std::memcpy(&value, &raw, sizeof(value));
        return value;
      }();
    float value = std::ldexp(static_cast<float>(mantissa), -24);
    if (sign != 0U)
      value = -value;
    return value;
  }
  result |= ((exponent + 112U) << 23U) | (mantissa << 13U);
  float value = 0.0F;
  std::memcpy(&value, &result, sizeof(value));
  return value;
}

uint16_t host_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000U) == 0x7f800000U) {
    if ((bits & 0x007fffffU) != 0U)
      return static_cast<uint16_t>((bits >> 16U & 0x8000U) | 0x7fc0U |
                                   (bits >> 16U & 0x003fU));
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & 0xffffU;
  if (lower > 0x8000U || (lower == 0x8000U && (upper & UINT32_C(1)) != 0U))
    ++upper;
  return static_cast<uint16_t>(upper);
}

bool tiny_oracle(const uint64_t m, const uint64_t k, const uint64_t n,
                 const HostInputs &inputs,
                 const std::vector<uint16_t> &production,
                 const std::vector<uint16_t> &staged) {
  size_t production_mismatches = 0U;
  size_t staged_mismatches = 0U;
  for (uint64_t row = 0U; row < m; ++row) {
    for (uint64_t column = 0U; column < n; ++column) {
      float accumulator = 0.0F;
      for (uint64_t inner = 0U; inner < k; ++inner) {
        const float activation = host_half_to_float(host_fp8_to_half(
            inputs.activation[static_cast<size_t>(row * k + inner)]));
        const float weight = host_half_to_float(host_fp8_to_half(
            inputs.weight[static_cast<size_t>(column * k + inner)]));
        accumulator = std::fmaf(activation, weight, accumulator);
      }
      const uint16_t expected =
          host_bf16_rne(accumulator * inputs.activation_scales[row] *
                        inputs.weight_scales[column]);
      const size_t index = static_cast<size_t>(row * n + column);
      production_mismatches += production[index] != expected ? 1U : 0U;
      staged_mismatches += staged[index] != expected ? 1U : 0U;
    }
  }
  std::printf(
      "tiny-oracle m=%llu k=%llu n=%llu production_bf16_mismatches=%zu "
      "staged_bf16_mismatches=%zu status=%s\n",
      static_cast<unsigned long long>(m), static_cast<unsigned long long>(k),
      static_cast<unsigned long long>(n), production_mismatches,
      staged_mismatches,
      production_mismatches == 0U && staged_mismatches == 0U ? "PASS" : "FAIL");
  return production_mismatches == 0U && staged_mismatches == 0U;
}

bool compare_bits(const uint64_t m, const uint64_t n,
                  const std::vector<uint16_t> &production,
                  const std::vector<uint16_t> &staged) {
  const size_t elements = static_cast<size_t>(m * n);
  size_t mismatches = 0U;
  size_t first = elements;
  for (size_t index = 0U; index < elements; ++index) {
    if (production[index] != staged[index]) {
      if (first == elements)
        first = index;
      ++mismatches;
    }
  }
  std::printf("bitwise m=%llu n=%llu mismatches=%zu first=%s status=%s\n",
              static_cast<unsigned long long>(m),
              static_cast<unsigned long long>(n), mismatches,
              first == elements ? "none" : std::to_string(first).c_str(),
              mismatches == 0U ? "PASS" : "FAIL");
  return mismatches == 0U;
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
  uint32_t seed;
  const char *name;
};

bool run_case(const Shape shape, bool *const all_ok) {
  HostInputs inputs;
  if (!fill_inputs(shape.m, shape.k, shape.n, shape.seed, &inputs))
    return false;
  DeviceBuffers buffers;
  if (!allocate(shape.m, shape.k, shape.n, &buffers) ||
      !upload(shape.m, shape.k, shape.n, inputs, &buffers) ||
      !clear_outputs(&buffers)) {
    release(&buffers);
    return false;
  }
  float production_us = 0.0F;
  float staged_us = 0.0F;
  const bool measured =
      measure(shape.m, shape.k, shape.n, &buffers, &production_us, &staged_us);
  if (!measured || !synchronize(&buffers, "case synchronize")) {
    release(&buffers);
    return false;
  }
  std::vector<uint16_t> production;
  std::vector<uint16_t> staged;
  const bool copied =
      copy_outputs(shape.m * shape.n, &buffers, &production, &staged);
  const bool guards = copied && check_guards(buffers);
  bool bitwise = false;
  if (copied)
    bitwise = compare_bits(shape.m, shape.n, production, staged);
  bool oracle = true;
  if (shape.m <= 32U && shape.n <= 32U && shape.k <= 128U)
    oracle = copied &&
             tiny_oracle(shape.m, shape.k, shape.n, inputs, production, staged);
  const bool case_ok = measured && guards && bitwise && oracle;
  const double ratio = production_us > 0.0F
                           ? static_cast<double>(staged_us) / production_us
                           : 0.0;
  std::printf("result case=%s m=%llu k=%llu n=%llu production_us=%.3f "
              "staged_total_us=%.3f ratio=%.6f transient_workspace_bytes=%llu "
              "status=%s\n",
              shape.name, static_cast<unsigned long long>(shape.m),
              static_cast<unsigned long long>(shape.k),
              static_cast<unsigned long long>(shape.n), production_us,
              staged_us, ratio,
              static_cast<unsigned long long>(buffers.transient_bytes),
              case_ok ? "PASS" : "FAIL");
  *all_ok = *all_ok && case_ok;
  release(&buffers);
  return true;
}

bool parse_device(const char *const text, int *const device) {
  if (text == nullptr || device == nullptr)
    return false;
  char *end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed < 0L || parsed > INT32_MAX)
    return false;
  *device = static_cast<int>(parsed);
  return true;
}

} // namespace

int main(int argc, char **argv) {
  int device = 0;
  if (argc > 2 || (argc == 2 && !parse_device(argv[1], &device))) {
    std::fprintf(
        stderr, "usage: phase78_fp8_gfx1030_f16_tile_staging_probe [DEVICE]\n");
    return EXIT_FAILURE;
  }
  if (!hip_ok(hipSetDevice(device), "set device"))
    return EXIT_FAILURE;
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device),
              "get device properties") ||
      !exact_gfx1030(properties.gcnArchName)) {
    std::fprintf(stderr, "exact gfx1030 required; got %s\n",
                 properties.gcnArchName);
    return EXIT_FAILURE;
  }
  std::printf("target=%s device=%d pci=%04x:%02x:%02x name=%s\n",
              properties.gcnArchName, device, properties.pciDomainID,
              properties.pciBusID, properties.pciDeviceID, properties.name);

  // The first two cases provide the requested representative K6144/N5120
  // measurements.  The odd M cases exercise the same 64x64 tails, while the
  // final case makes both M and N non-aligned and forces scalar K-tail ingress.
  const std::array<Shape, 5> shapes = {
      Shape{128U, 6144U, 5120U, 11U, "representative-m128"},
      Shape{1024U, 6144U, 5120U, 13U, "representative-m1024"},
      Shape{17U, 6144U, 5120U, 17U, "odd-m17"},
      Shape{219U, 6144U, 5120U, 19U, "odd-m219"},
      Shape{17U, 70U, 31U, 23U, "nonaligned-tiny"}};
  bool all_ok = true;
  for (const Shape shape : shapes) {
    if (!run_case(shape, &all_ok))
      return EXIT_FAILURE;
  }
  std::printf("summary status=%s cases=%zu warmups=%d measured=%d "
              "candidate_timing=stage_activation+stage_weight+consumer "
              "transient_workspace=counted\n",
              all_ok ? "PASS" : "FAIL", shapes.size(), kWarmups, kMeasured);
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
