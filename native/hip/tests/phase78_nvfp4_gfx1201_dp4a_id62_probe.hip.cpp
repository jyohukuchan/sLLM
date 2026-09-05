// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase78-nvfp4-byte-permute-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// 3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70, ggml/src/ggml-cuda/vecdotq.cuh
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

// Phase 78 standalone gfx1201 NVFP4 ID62-DP4A versus ID64-WMMA probe.
//
// This developer-only binary asks one bounded question: can the existing
// 64x64/K32 block16-exact DP4A recipe execute usefully on gfx1201?  The
// production-equivalent ID64 WMMA kernel is embedded as the control.  DP4A
// keeps one integer dot sum per K16 scale domain, then uses the same
// activation-scale, weight-scale, FP32 accumulation, tensor-scale, and BF16
// RNE order.  No production selector or source is changed.

#define main phase78_scaled_ingress_embedded_main
#include "phase78_nvfp4_gfx1201_wmma_scaled_ingress_probe.hip.cpp"
#undef main

namespace {

constexpr uint32_t kDp4aWarmups = 3U;
constexpr uint32_t kDp4aMeasured = 10U;
constexpr uint64_t kDp4aMaximumLds = UINT64_C(64) * 1024U;

static_assert(kDp4aWarmups == kWarmups);
static_assert(kDp4aMeasured == kMeasured);

constexpr std::array<Shape, 3> kDp4aBoundaryShapes = {{
    {17U, 64U, 17U, 0U, "small-m17-n17"},
    {63U, 64U, 65U, 0U, "boundary-m63-n65"},
    {65U, 64U, 63U, 0U, "boundary-m65-n63"},
}};

struct Dp4aScaledPacks final {
  int32_t even;
  int32_t odd;
};

__device__ __forceinline__ Dp4aScaledPacks
dp4a_e2m1x8_scaled2_to_i8x4_pair(const uint32_t packed) {
  // Exact signed bytes for E2M1 value * 2.  Even and odd nibbles are split
  // because one v_dot4_i32_i8 consumes four byte lanes.
  constexpr uint32_t table_0_3 = UINT32_C(0x03020100);
  constexpr uint32_t table_4_7 = UINT32_C(0x0c080604);
  constexpr uint32_t table_8_11 = UINT32_C(0xfdfeff00);
  constexpr uint32_t table_12_15 = UINT32_C(0xf4f8fafc);
  constexpr uint32_t low_mask = UINT32_C(0x07070707);
  constexpr uint32_t identity = UINT32_C(0x03020100);
  constexpr uint32_t sign_mask = UINT32_C(0x08080808);
  const uint32_t even_indices = packed;
  const uint32_t odd_indices = packed >> 4U;
  const uint32_t even_low =
      __builtin_amdgcn_perm(table_4_7, table_0_3, even_indices & low_mask);
  const uint32_t odd_low =
      __builtin_amdgcn_perm(table_4_7, table_0_3, odd_indices & low_mask);
  const uint32_t even_high =
      __builtin_amdgcn_perm(table_12_15, table_8_11, even_indices & low_mask);
  const uint32_t odd_high =
      __builtin_amdgcn_perm(table_12_15, table_8_11, odd_indices & low_mask);
  const uint32_t even_select = identity | ((even_indices & sign_mask) >> 1U);
  const uint32_t odd_select = identity | ((odd_indices & sign_mask) >> 1U);
  return {
      static_cast<int32_t>(
          __builtin_amdgcn_perm(even_high, even_low, even_select)),
      static_cast<int32_t>(
          __builtin_amdgcn_perm(odd_high, odd_low, odd_select)),
  };
}

__device__ __forceinline__ int32_t dp4a_signed_dot4(const int32_t lhs,
                                                    const int32_t rhs,
                                                    const int32_t accumulator) {
#if __has_builtin(__builtin_amdgcn_sdot4)
  return __builtin_amdgcn_sdot4(lhs, rhs, accumulator, false);
#else
  int32_t result = accumulator;
#pragma unroll
  for (uint32_t lane = 0U; lane < 4U; ++lane) {
    result += static_cast<int8_t>(static_cast<uint32_t>(lhs) >> (lane * 8U)) *
              static_cast<int8_t>(static_cast<uint32_t>(rhs) >> (lane * 8U));
  }
  return result;
#endif
}

// Exact copy of ID62's 64x64/K32 geometry and arithmetic contract, with a
// probe-local symbol so the production binary remains untouched.
__global__ __launch_bounds__(256, 1) void id62_dp4a64x64_kernel(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  constexpr uint32_t tile_m = 64U;
  constexpr uint32_t tile_n = 64U;
  constexpr uint32_t tile_k = 32U;
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

  const uint64_t column_tiles = (n + tile_n - 1U) / tile_n;
  const uint64_t tile_index = blockIdx.x;
  const uint64_t row_base = (tile_index / column_tiles) * tile_m;
  const uint64_t column_base = (tile_index % column_tiles) * tile_n;
  const uint32_t thread = threadIdx.x;
  const uint32_t local_row = thread >> 4U;
  const uint32_t local_column = thread & UINT32_C(15);
  const uint64_t packed_row_bytes = k / UINT64_C(2);
  const uint64_t blocks_per_row = k / block_k;
  float accumulators[rows_per_thread][columns_per_thread] = {};

  for (uint64_t base = 0U; base < k; base += tile_k) {
    for (uint32_t index = thread; index < tile_m * packed_chunks_per_stage;
         index += blockDim.x) {
      const uint32_t row = index / packed_chunks_per_stage;
      const uint32_t chunk = index % packed_chunks_per_stage;
      const uint64_t source_row = row_base + row;
      const uint64_t inner = base + static_cast<uint64_t>(chunk) * 8U;
      const Dp4aScaledPacks values =
          source_row < m && inner + 8U <= k
              ? dp4a_e2m1x8_scaled2_to_i8x4_pair(__builtin_nontemporal_load(
                    reinterpret_cast<const uint32_t *>(
                        packed_activation + source_row * packed_row_bytes +
                        inner / UINT64_C(2))))
              : Dp4aScaledPacks{0, 0};
      activation_tile[row][chunk * 2U] = values.even;
      activation_tile[row][chunk * 2U + 1U] = values.odd;
    }
    for (uint32_t index = thread; index < tile_n * packed_chunks_per_stage;
         index += blockDim.x) {
      const uint32_t column = index / packed_chunks_per_stage;
      const uint32_t chunk = index % packed_chunks_per_stage;
      const uint64_t source_column = column_base + column;
      const uint64_t inner = base + static_cast<uint64_t>(chunk) * 8U;
      const Dp4aScaledPacks values =
          source_column < n && inner + 8U <= k
              ? dp4a_e2m1x8_scaled2_to_i8x4_pair(__builtin_nontemporal_load(
                    reinterpret_cast<const uint32_t *>(
                        packed_weight + source_column * packed_row_bytes +
                        inner / UINT64_C(2))))
              : Dp4aScaledPacks{0, 0};
      weight_tile[column][chunk * 2U] = values.even;
      weight_tile[column][chunk * 2U + 1U] = values.odd;
    }
    for (uint32_t index = thread; index < tile_m * blocks_per_stage;
         index += blockDim.x) {
      const uint32_t row = index / blocks_per_stage;
      const uint32_t block = index % blocks_per_stage;
      const uint64_t source_row = row_base + row;
      const uint64_t source_block = base / block_k + block;
      activation_scale_tile[row][block] =
          source_row < m && source_block < blocks_per_row
              ? e4m3fn_to_float(
                    activation_block_scales[source_row * blocks_per_row +
                                            source_block])
              : 0.0F;
    }
    for (uint32_t index = thread; index < tile_n * blocks_per_stage;
         index += blockDim.x) {
      const uint32_t column = index / blocks_per_stage;
      const uint32_t block = index % blocks_per_stage;
      const uint64_t source_column = column_base + column;
      const uint64_t source_block = base / block_k + block;
      weight_scale_tile[column][block] =
          source_column < n && source_block < blocks_per_row
              ? e4m3fn_to_float(
                    weight_block_scales[source_column * blocks_per_row +
                                        source_block])
              : 0.0F;
    }
    __syncthreads();

#pragma unroll
    for (uint32_t block = 0U; block < blocks_per_stage; ++block) {
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
                dp4a_signed_dot4(activation_packs[row], weight_packs[column],
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

  const float tensor_scale = weight_tensor_scale[0] * input_tensor_scale[0];
#pragma unroll
  for (uint32_t row = 0U; row < rows_per_thread; ++row) {
#pragma unroll
    for (uint32_t column = 0U; column < columns_per_thread; ++column) {
      const uint64_t output_row = row_base + local_row + row * thread_rows;
      const uint64_t output_column =
          column_base + local_column + column * thread_columns;
      if (output_row < m && output_column < n) {
        output[output_row * n + output_column] =
            bf16_rne(accumulators[row][column] * tensor_scale);
      }
    }
  }
}

enum class Dp4aVariant : uint32_t { Id64 = 0U, Id62Dp4a = 1U };

constexpr std::array<Dp4aVariant, 2> kDp4aVariants = {Dp4aVariant::Id64,
                                                      Dp4aVariant::Id62Dp4a};

const char *dp4a_variant_name(const Dp4aVariant variant) {
  switch (variant) {
  case Dp4aVariant::Id64:
    return "id64-wmma128x64-control";
  case Dp4aVariant::Id62Dp4a:
    return "id62-dp4a64x64-candidate";
  }
  return "unknown";
}

const void *dp4a_kernel_pointer(const Dp4aVariant variant) {
  switch (variant) {
  case Dp4aVariant::Id64:
    return reinterpret_cast<const void *>(id64_control_kernel);
  case Dp4aVariant::Id62Dp4a:
    return reinterpret_cast<const void *>(id62_dp4a64x64_kernel);
  }
  return nullptr;
}

struct Dp4aResource final {
  bool available = false;
  int vgpr = 0;
  std::size_t lds = 0U;
  std::size_t scratch = 0U;
  int active_blocks = 0;
  int active_waves = 0;
  double occupancy = 0.0;
};

Dp4aResource dp4a_resource(const Dp4aVariant variant,
                           const hipDeviceProp_t &properties) {
  Dp4aResource resource;
  const void *const kernel = dp4a_kernel_pointer(variant);
  hipFuncAttributes attributes{};
  int active_blocks = 0;
  if (kernel == nullptr ||
      !hip_ok(hipFuncGetAttributes(&attributes, kernel),
              "DP4A hipFuncGetAttributes") ||
      !hip_ok(hipOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
                                                           kernel, 256, 0U),
              "DP4A occupancy")) {
    return resource;
  }
  resource.available = true;
  resource.vgpr = attributes.numRegs;
  resource.lds = attributes.sharedSizeBytes;
  resource.scratch = attributes.localSizeBytes;
  resource.active_blocks = active_blocks;
  resource.active_waves = active_blocks * 8;
  resource.occupancy = properties.maxThreadsPerMultiProcessor == 0
                           ? 0.0
                           : static_cast<double>(active_blocks * 256) /
                                 properties.maxThreadsPerMultiProcessor;
  const bool pass = resource.lds <= kDp4aMaximumLds && resource.scratch == 0U &&
                    active_blocks > 0;
  std::cout << "resources variant=" << dp4a_variant_name(variant)
            << " threads=256 vgpr=" << resource.vgpr << " lds=" << resource.lds
            << " scratch_per_thread=" << resource.scratch
            << " max_threads=" << attributes.maxThreadsPerBlock
            << " active_blocks_per_cu=" << resource.active_blocks
            << " active_waves_per_cu=" << resource.active_waves
            << " occupancy=" << std::fixed << std::setprecision(6)
            << resource.occupancy << " status=" << (pass ? "PASS" : "FAIL")
            << "\n";
  return resource;
}

bool launch_dp4a_variant(const Dp4aVariant variant, const Shape &shape,
                         const DeviceBuffers &buffers) {
  switch (variant) {
  case Dp4aVariant::Id64: {
    const dim3 grid(static_cast<uint32_t>((shape.n + 63U) / 64U),
                    static_cast<uint32_t>((shape.m + 127U) / 128U));
    hipLaunchKernelGGL(
        id64_control_kernel, grid, dim3(256U), 0U, buffers.stream,
        buffers.activation, buffers.activation_scales, buffers.weight,
        buffers.weight_scales, buffers.weight_tensor_scale,
        buffers.input_tensor_scale, buffers.output, shape.m, shape.k, shape.n);
    break;
  }
  case Dp4aVariant::Id62Dp4a: {
    const uint64_t column_tiles = (shape.n + UINT64_C(63)) / UINT64_C(64);
    const uint64_t row_tiles = (shape.m + UINT64_C(63)) / UINT64_C(64);
    const dim3 grid(static_cast<uint32_t>(column_tiles * row_tiles));
    hipLaunchKernelGGL(
        id62_dp4a64x64_kernel, grid, dim3(256U), 0U, buffers.stream,
        buffers.activation, buffers.activation_scales, buffers.weight,
        buffers.weight_scales, buffers.weight_tensor_scale,
        buffers.input_tensor_scale, buffers.output, shape.m, shape.k, shape.n);
    break;
  }
  }
  return hip_ok(hipGetLastError(), "DP4A comparison kernel launch");
}

struct Dp4aMeasurement final {
  std::array<float, kDp4aMeasured> samples_us{};
  float median_us = 0.0F;
  float mad_us = 0.0F;
  float minimum_us = 0.0F;
  float maximum_us = 0.0F;
  std::size_t repeat_mismatches = 0U;
  std::vector<uint16_t> output;
  bool ran = false;
  bool deterministic = false;
};

float dp4a_upper_median(std::array<float, kDp4aMeasured> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2U];
}

bool measure_dp4a_variant(const Dp4aVariant variant, const Shape &shape,
                          const DeviceBuffers &buffers,
                          Dp4aMeasurement *const measurement) {
  const std::size_t elements = static_cast<std::size_t>(shape.m * shape.n);
  const std::size_t bytes = elements * sizeof(uint16_t);
  for (uint32_t warmup = 0U; warmup < kDp4aWarmups; ++warmup) {
    if (!launch_dp4a_variant(variant, shape, buffers)) {
      return false;
    }
  }
  if (!hip_ok(hipStreamSynchronize(buffers.stream),
              "DP4A warmup synchronize")) {
    return false;
  }
  measurement->output.resize(elements);
  std::vector<uint16_t> current(elements);
  for (uint32_t iteration = 0U; iteration < kDp4aMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(buffers.start, buffers.stream),
                "DP4A timing start") ||
        !launch_dp4a_variant(variant, shape, buffers) ||
        !hip_ok(hipEventRecord(buffers.stop, buffers.stream),
                "DP4A timing stop") ||
        !hip_ok(hipEventSynchronize(buffers.stop), "DP4A timing synchronize")) {
      return false;
    }
    float milliseconds = 0.0F;
    if (!hip_ok(hipEventElapsedTime(&milliseconds, buffers.start, buffers.stop),
                "DP4A timing elapsed") ||
        !hip_ok(hipMemcpy(current.data(), buffers.output, bytes,
                          hipMemcpyDeviceToHost),
                "DP4A output copy")) {
      return false;
    }
    measurement->samples_us[iteration] = milliseconds * 1000.0F;
    if (iteration == 0U) {
      measurement->output = current;
    } else {
      for (std::size_t index = 0U; index < elements; ++index) {
        measurement->repeat_mismatches += static_cast<std::size_t>(
            current[index] != measurement->output[index]);
      }
    }
  }
  measurement->median_us = dp4a_upper_median(measurement->samples_us);
  std::array<float, kDp4aMeasured> deviations{};
  for (std::size_t index = 0U; index < deviations.size(); ++index) {
    deviations[index] =
        std::abs(measurement->samples_us[index] - measurement->median_us);
  }
  measurement->mad_us = dp4a_upper_median(deviations);
  measurement->minimum_us = *std::min_element(measurement->samples_us.begin(),
                                              measurement->samples_us.end());
  measurement->maximum_us = *std::max_element(measurement->samples_us.begin(),
                                              measurement->samples_us.end());
  measurement->ran = true;
  measurement->deterministic = measurement->repeat_mismatches == 0U;
  std::cout << "timing shape=" << shape.name
            << " variant=" << dp4a_variant_name(variant)
            << " warmups=" << kDp4aWarmups << " measured=" << kDp4aMeasured
            << " samples_us=";
  for (std::size_t index = 0U; index < measurement->samples_us.size();
       ++index) {
    if (index != 0U) {
      std::cout << ',';
    }
    std::cout << std::fixed << std::setprecision(3)
              << measurement->samples_us[index];
  }
  std::cout << " median_us=" << measurement->median_us
            << " mad_us=" << measurement->mad_us
            << " min_us=" << measurement->minimum_us
            << " max_us=" << measurement->maximum_us
            << " repeat_bf16_mismatches=" << measurement->repeat_mismatches
            << " deterministic="
            << (measurement->deterministic ? "PASS" : "FAIL") << "\n";
  return measurement->deterministic;
}

std::size_t dp4a_nonfinite(const std::vector<uint16_t> &values) {
  return static_cast<std::size_t>(
      std::count_if(values.begin(), values.end(), [](const uint16_t bits) {
        return (bits & UINT16_C(0x7f80)) == UINT16_C(0x7f80);
      }));
}

bool check_named_host_oracle(
    const Shape &shape, const char *const variant,
    const std::array<OraclePoint, kOracleSamples> &oracle,
    const std::vector<uint16_t> &output) {
  std::size_t bf16_mismatches = 0U;
  uint32_t max_ulp = 0U;
  double max_abs = 0.0;
  double max_normalized = 0.0;
  for (const OraclePoint &point : oracle) {
    const uint16_t observed_bits = output[point.index];
    const double observed = host_bf16_to_float(observed_bits);
    const double absolute_error = std::abs(observed - point.expected);
    max_abs = std::max(max_abs, absolute_error);
    max_normalized =
        std::max(max_normalized,
                 absolute_error / std::max(point.absolute_sum,
                                           std::numeric_limits<double>::min()));
    bf16_mismatches +=
        static_cast<std::size_t>(observed_bits != point.expected_bf16);
    const uint32_t lhs = ordered_bf16(observed_bits);
    const uint32_t rhs = ordered_bf16(point.expected_bf16);
    max_ulp = std::max(max_ulp, lhs > rhs ? lhs - rhs : rhs - lhs);
  }
  const bool pass = max_normalized <= 0.01;
  std::cout << "host_oracle shape=" << shape.name << " variant=" << variant
            << " samples=" << kOracleSamples
            << " bf16_mismatches=" << bf16_mismatches
            << " max_bf16_ulp=" << max_ulp
            << " max_abs=" << std::setprecision(10) << max_abs
            << " max_normalized_error=" << max_normalized
            << " tolerance=0.01 status=" << (pass ? "PASS" : "FAIL") << "\n";
  return pass;
}

struct Dp4aShapeResult final {
  Shape shape{};
  std::array<Dp4aMeasurement, kDp4aVariants.size()> measurements;
  bool ok = false;
};

bool run_dp4a_shape(const Shape &shape, Dp4aShapeResult *const result,
                    CleanupTotals *const cleanup_totals) {
  std::cout << "shape_begin name=" << shape.name << " m=" << shape.m
            << " k=" << shape.k << " n=" << shape.n
            << " occurrences=" << shape.occurrences << "\n";
  const HostInputs inputs = make_inputs(shape);
  DeviceBuffers buffers;
  if (!allocate_and_upload(shape, inputs, &buffers)) {
    cleanup(&buffers, cleanup_totals);
    return false;
  }
  result->shape = shape;
  bool ok = true;
  for (const Dp4aVariant variant : kDp4aVariants) {
    ok = measure_dp4a_variant(
             variant, shape, buffers,
             &result->measurements[static_cast<std::size_t>(variant)]) &&
         ok;
  }
  if (ok) {
    const auto oracle = host_oracle(shape, inputs);
    for (const Dp4aVariant variant : kDp4aVariants) {
      const Dp4aMeasurement &measurement =
          result->measurements[static_cast<std::size_t>(variant)];
      ok = check_named_host_oracle(shape, dp4a_variant_name(variant), oracle,
                                   measurement.output) &&
           ok;
      const std::size_t nonfinite = dp4a_nonfinite(measurement.output);
      std::cout << "finite shape=" << shape.name
                << " variant=" << dp4a_variant_name(variant)
                << " nonfinite=" << nonfinite
                << " status=" << (nonfinite == 0U ? "PASS" : "FAIL") << "\n";
      ok = nonfinite == 0U && ok;
    }
    const auto &id64 = result->measurements[0];
    const auto &id62 = result->measurements[1];
    const Comparison comparison = compare(id64.output, id62.output);
    const bool bitwise = comparison.mismatches == 0U;
    std::cout << "bit_compare shape=" << shape.name << " pair=id64-vs-id62-dp4a"
              << " bf16_mismatches=" << comparison.mismatches
              << " max_bf16_ulp=" << comparison.max_ulp
              << " max_abs=" << std::setprecision(10) << comparison.max_abs
              << " max_rel=" << comparison.max_rel << " id64_fnv64=0x"
              << std::hex << hash_bf16(id64.output) << " id62_fnv64=0x"
              << hash_bf16(id62.output) << std::dec
              << " status=" << (bitwise ? "PASS" : "FAIL") << "\n";
    ok = bitwise && ok;
  }
  cleanup(&buffers, cleanup_totals);
  result->ok = ok;
  std::cout << "shape_end name=" << shape.name
            << " status=" << (ok ? "PASS" : "FAIL") << "\n";
  return ok;
}

double
dp4a_weighted_total(const std::array<Dp4aShapeResult, kShapes.size()> &results,
                    const Dp4aVariant variant) {
  double total = 0.0;
  for (const Dp4aShapeResult &result : results) {
    total += result.measurements[static_cast<std::size_t>(variant)].median_us *
             result.shape.occurrences;
  }
  return total;
}

bool print_dp4a_weighted(
    const std::array<Dp4aShapeResult, kShapes.size()> &results) {
  uint64_t total_weight = 0U;
  for (const Dp4aShapeResult &result : results) {
    total_weight += result.shape.occurrences;
  }
  const double id64_total = dp4a_weighted_total(results, Dp4aVariant::Id64);
  const double id62_total = dp4a_weighted_total(results, Dp4aVariant::Id62Dp4a);
  for (const Dp4aVariant variant : kDp4aVariants) {
    const double total = dp4a_weighted_total(results, variant);
    std::cout << "weighted variant=" << dp4a_variant_name(variant)
              << " shapes=6 qwen_projection_weight=" << total_weight
              << " weighted_total_us=" << std::fixed << std::setprecision(3)
              << total << " weighted_mean_us=" << total / total_weight
              << " speedup_vs_id64=" << (total > 0.0 ? id64_total / total : 0.0)
              << "\n";
  }
  for (const Dp4aShapeResult &result : results) {
    const float id64 = result.measurements[0].median_us;
    const float id62 = result.measurements[1].median_us;
    std::cout << "shape_speedup shape=" << result.shape.name
              << " id64_us=" << id64 << " id62_dp4a_us=" << id62
              << " speedup_vs_id64=" << (id62 > 0.0F ? id64 / id62 : 0.0F)
              << "\n";
  }
  const bool two_x = id62_total <= id64_total / 2.0;
  std::cout << "two_x_gate scope=weighted-six-shape id64_total_us="
            << id64_total << " id62_dp4a_total_us=" << id62_total
            << " half_id64_us=" << id64_total / 2.0
            << " status=" << (two_x ? "PASS" : "FAIL") << "\n";
  return two_x;
}

} // namespace

int main(int argc, char **argv) {
  int device = 0;
  if (argc > 2 || (argc == 2 && !parse_device(argv[1], &device))) {
    std::cerr << "usage: phase78_nvfp4_gfx1201_dp4a_id62_probe [DEVICE]\n";
    return EXIT_FAILURE;
  }
  if (!hip_ok(hipSetDevice(device), "DP4A hipSetDevice")) {
    return EXIT_FAILURE;
  }
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device),
              "DP4A hipGetDeviceProperties")) {
    return EXIT_FAILURE;
  }
  int runtime_version = 0;
  if (!hip_ok(hipRuntimeGetVersion(&runtime_version),
              "DP4A hipRuntimeGetVersion")) {
    return EXIT_FAILURE;
  }
  std::cout << "identity device=" << device << " name=" << properties.name
            << " pci=" << std::hex << std::setw(4) << std::setfill('0')
            << properties.pciDomainID << ':' << std::setw(2)
            << properties.pciBusID << ':' << std::setw(2)
            << properties.pciDeviceID << std::dec << std::setfill(' ')
            << " arch=" << properties.gcnArchName
            << " hip_header=" << HIP_VERSION_MAJOR << '.' << HIP_VERSION_MINOR
            << '.' << HIP_VERSION_PATCH << " hip_runtime=" << runtime_version
            << " total_global_mem=" << properties.totalGlobalMem << "\n";
  if (!exact_gfx1201(properties.gcnArchName)) {
    std::cerr << "unsupported exact target: expected gfx1201, observed "
              << properties.gcnArchName << "\n";
    return EXIT_FAILURE;
  }

  bool resources_ok = true;
  for (const Dp4aVariant variant : kDp4aVariants) {
    const Dp4aResource resource = dp4a_resource(variant, properties);
    resources_ok = resource.available && resource.lds <= kDp4aMaximumLds &&
                   resource.scratch == 0U && resource.active_blocks > 0 &&
                   resources_ok;
  }

  CleanupTotals cleanup_totals;
  bool correctness_ok = resources_ok;
  for (const Shape &shape : kDp4aBoundaryShapes) {
    Dp4aShapeResult ignored;
    correctness_ok =
        run_dp4a_shape(shape, &ignored, &cleanup_totals) && correctness_ok;
  }
  const bool executable = correctness_ok;
  std::cout << "PHASE78_NVFP4_GFX1201_DP4A_EXECUTABLE="
            << (executable ? "PASS" : "FAIL") << "\n";

  std::array<Dp4aShapeResult, kShapes.size()> results{};
  if (correctness_ok) {
    for (std::size_t index = 0U; index < kShapes.size(); ++index) {
      correctness_ok =
          run_dp4a_shape(kShapes[index], &results[index], &cleanup_totals) &&
          correctness_ok;
    }
  }
  const bool two_x = correctness_ok && print_dp4a_weighted(results);
  const bool cleanup_ok =
      cleanup_totals.ok && cleanup_totals.allocations == cleanup_totals.frees;
  std::cout << "cleanup allocations=" << cleanup_totals.allocations
            << " frees=" << cleanup_totals.frees
            << " status=" << (cleanup_ok ? "PASS" : "FAIL") << "\n";
  const bool evidence_ok = correctness_ok && resources_ok && cleanup_ok;
  std::cout << "PHASE78_NVFP4_GFX1201_DP4A_ID62_EVIDENCE="
            << (evidence_ok ? "PASS" : "FAIL") << "\n";
  std::cout << "PHASE78_NVFP4_GFX1201_DP4A_ID62_DECISION="
            << (evidence_ok && two_x ? "GO" : "N0") << "\n";
  return evidence_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
