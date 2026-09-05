// Phase 78 standalone gfx1201 NVFP4 ID64 tile-geometry sweep.
//
// This probe holds ID64's E2M1->E4M3 ingress, K=16 scale-domain WMMA,
// activation-scale/weight-scale/FP32-add order, and BF16-RNE epilogue fixed.
// It changes only the number of 16-row waves and 16-column accumulator tiles
// owned by a workgroup.  The production 128x64 ID64 kernel is the control;
// 128x32, 128x16, and 64x32 are independent candidates.

#define main phase78_scaled_ingress_embedded_main
#include "phase78_nvfp4_gfx1201_wmma_scaled_ingress_probe.hip.cpp"
#undef main

#include <string>

namespace {

constexpr uint32_t kSweepWarmups = 3U;
constexpr uint32_t kSweepMeasured = 10U;
constexpr uint64_t kSweepMaximumLds = UINT64_C(64) * 1024U;
constexpr int kSweepMaximumVgpr = 128;

static_assert(kSweepWarmups == kWarmups);
static_assert(kSweepMeasured == kMeasured);

constexpr std::array<Shape, 4> kSweepBoundaryShapes = {{
    {127U, 64U, 31U, 0U, "boundary-m127-n31"},
    {129U, 64U, 33U, 0U, "boundary-m129-n33"},
    {127U, 64U, 63U, 0U, "boundary-m127-n63"},
    {129U, 64U, 65U, 0U, "boundary-m129-n65"},
}};

enum class TileVariant : uint32_t {
  Id64_128x64 = 0U,
  Tile128x32 = 1U,
  Tile128x16 = 2U,
  Tile64x32 = 3U,
};

constexpr std::array<TileVariant, 4> kTileVariants = {
    TileVariant::Id64_128x64, TileVariant::Tile128x32, TileVariant::Tile128x16,
    TileVariant::Tile64x32};

const char *tile_variant_name(const TileVariant variant) {
  switch (variant) {
  case TileVariant::Id64_128x64:
    return "id64-128x64-control";
  case TileVariant::Tile128x32:
    return "id64-order-128x32";
  case TileVariant::Tile128x16:
    return "id64-order-128x16";
  case TileVariant::Tile64x32:
    return "id64-order-64x32";
  }
  return "unknown-tile";
}

constexpr uint32_t tile_rows(const TileVariant variant) {
  switch (variant) {
  case TileVariant::Id64_128x64:
  case TileVariant::Tile128x32:
  case TileVariant::Tile128x16:
    return 128U;
  case TileVariant::Tile64x32:
    return 64U;
  }
  return 0U;
}

constexpr uint32_t tile_columns(const TileVariant variant) {
  switch (variant) {
  case TileVariant::Id64_128x64:
    return 64U;
  case TileVariant::Tile128x32:
  case TileVariant::Tile64x32:
    return 32U;
  case TileVariant::Tile128x16:
    return 16U;
  }
  return 0U;
}

constexpr uint32_t tile_threads(const TileVariant variant) {
  return tile_rows(variant) == 64U ? 128U : 256U;
}

template <uint32_t WavesPerWorkgroup, uint32_t ColumnTiles>
__global__
__launch_bounds__(WavesPerWorkgroup * 32U, 1) void id64_geometry_kernel(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
#if defined(__gfx1201__)
  static_assert(WavesPerWorkgroup == 4U || WavesPerWorkgroup == 8U);
  static_assert(ColumnTiles == 1U || ColumnTiles == 2U);
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t tile_m = 16U;
  constexpr uint32_t tile_n = 16U;
  constexpr uint32_t fragment_k = 16U;
  constexpr uint32_t stage_k = 32U;
  constexpr uint32_t scale_blocks_per_stage = stage_k / fragment_k;
  constexpr uint32_t rows_per_workgroup = WavesPerWorkgroup * tile_m;
  constexpr uint32_t columns_per_workgroup = ColumnTiles * tile_n;
  constexpr uint32_t tile_values = tile_m * stage_k;
  constexpr uint32_t values_per_group = 4U;
  constexpr uint32_t groups_per_tile = tile_values / values_per_group;
  constexpr uint32_t output_values = tile_m * tile_n;

  __shared__ __align__(4)
      rocwmma::float8_t activation_tile[WavesPerWorkgroup][tile_values];
  __shared__ __align__(4)
      rocwmma::float8_t weight_tile[ColumnTiles][tile_values];
  __shared__ float activation_scale_tile[WavesPerWorkgroup][tile_m]
                                        [scale_blocks_per_stage];
  __shared__ float weight_scale_tile[ColumnTiles * tile_n]
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
  const uint64_t packed_row_bytes = k / UINT64_C(2);
  float accumulators[ColumnTiles][output_values / wave_width] = {};

  if (thread == 0U) {
    tensor_scale = weight_tensor_scale[0] * input_tensor_scale[0];
  }

  for (uint64_t stage = 0U; stage < stages; ++stage) {
    const uint64_t inner_base = stage * stage_k;
    auto *const activation_groups =
        reinterpret_cast<uint32_t *>(activation_tile);
    auto *const weight_groups = reinterpret_cast<uint32_t *>(weight_tile);

    for (uint32_t group = thread; group < WavesPerWorkgroup * groups_per_tile;
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
        packed = __builtin_nontemporal_load(
            reinterpret_cast<const uint16_t *>(packed_activation +
                                               row * packed_row_bytes +
                                               inner_base / UINT64_C(2)) +
            local_group);
      }
      activation_groups[group] = e2m1x4_to_e4m3fn_exact(packed);
    }

    for (uint32_t group = thread; group < ColumnTiles * groups_per_tile;
         group += blockDim.x) {
      const uint32_t column_tile = group / groups_per_tile;
      const uint32_t tile_group = group - column_tile * groups_per_tile;
      const uint32_t local_column = tile_group / (stage_k / values_per_group);
      const uint32_t local_group =
          tile_group - local_column * (stage_k / values_per_group);
      const uint64_t column = column_base +
                              static_cast<uint64_t>(column_tile) * tile_n +
                              local_column;
      uint16_t packed = 0U;
      if (column < n && inner_base + local_group * values_per_group < k) {
        packed = __builtin_nontemporal_load(
            reinterpret_cast<const uint16_t *>(packed_weight +
                                               column * packed_row_bytes +
                                               inner_base / UINT64_C(2)) +
            local_group);
      }
      weight_groups[group] = e2m1x4_to_e4m3fn_exact(packed);
    }

    if (thread < WavesPerWorkgroup * tile_m * scale_blocks_per_stage) {
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
    if (thread < ColumnTiles * tile_n * scale_blocks_per_stage) {
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
      for (uint32_t column_tile = 0U; column_tile < ColumnTiles;
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
  for (uint32_t column_tile = 0U; column_tile < ColumnTiles; ++column_tile) {
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

struct SweepResource final {
  bool available = false;
  int vgpr = 0;
  std::size_t lds = 0U;
  std::size_t scratch = 0U;
  int active_blocks = 0;
  int active_waves = 0;
  double occupancy = 0.0;
};

const void *tile_kernel_pointer(const TileVariant variant) {
  switch (variant) {
  case TileVariant::Id64_128x64:
    return reinterpret_cast<const void *>(id64_control_kernel);
  case TileVariant::Tile128x32:
    return reinterpret_cast<const void *>(id64_geometry_kernel<8U, 2U>);
  case TileVariant::Tile128x16:
    return reinterpret_cast<const void *>(id64_geometry_kernel<8U, 1U>);
  case TileVariant::Tile64x32:
    return reinterpret_cast<const void *>(id64_geometry_kernel<4U, 2U>);
  }
  return nullptr;
}

SweepResource sweep_resource(const TileVariant variant,
                             const hipDeviceProp_t &properties) {
  SweepResource resource;
  hipFuncAttributes attributes{};
  const void *const kernel = tile_kernel_pointer(variant);
  const int threads = static_cast<int>(tile_threads(variant));
  int active_blocks = 0;
  if (!hip_ok(hipFuncGetAttributes(&attributes, kernel),
              "tile sweep hipFuncGetAttributes") ||
      !hip_ok(hipOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
                                                           kernel, threads, 0U),
              "tile sweep occupancy")) {
    return resource;
  }
  resource.available = true;
  resource.vgpr = attributes.numRegs;
  resource.lds = attributes.sharedSizeBytes;
  resource.scratch = attributes.localSizeBytes;
  resource.active_blocks = active_blocks;
  resource.active_waves =
      active_blocks * static_cast<int>(tile_threads(variant) / 32U);
  resource.occupancy = properties.maxThreadsPerMultiProcessor == 0
                           ? 0.0
                           : static_cast<double>(active_blocks * threads) /
                                 properties.maxThreadsPerMultiProcessor;
  const bool pass = resource.scratch == 0U &&
                    resource.lds <= kSweepMaximumLds &&
                    resource.vgpr <= kSweepMaximumVgpr && active_blocks > 0;
  std::cout << "resources variant=" << tile_variant_name(variant)
            << " tile_m=" << tile_rows(variant)
            << " tile_n=" << tile_columns(variant) << " threads=" << threads
            << " vgpr=" << resource.vgpr << " lds=" << resource.lds
            << " scratch_per_thread=" << resource.scratch
            << " max_threads=" << attributes.maxThreadsPerBlock
            << " active_blocks_per_cu=" << active_blocks
            << " active_waves_per_cu=" << resource.active_waves
            << " occupancy=" << std::fixed << std::setprecision(6)
            << resource.occupancy << " status=" << (pass ? "PASS" : "FAIL")
            << "\n";
  return resource;
}

bool launch_tile(const TileVariant variant, const Shape &shape,
                 const DeviceBuffers &buffers) {
  const uint32_t rows = tile_rows(variant);
  const uint32_t columns = tile_columns(variant);
  if (rows == 0U || columns == 0U) {
    return false;
  }
  const dim3 grid(static_cast<uint32_t>((shape.n + columns - 1U) / columns),
                  static_cast<uint32_t>((shape.m + rows - 1U) / rows));
  const dim3 block(tile_threads(variant));
  switch (variant) {
  case TileVariant::Id64_128x64:
    hipLaunchKernelGGL(id64_control_kernel, grid, block, 0U, buffers.stream,
                       buffers.activation, buffers.activation_scales,
                       buffers.weight, buffers.weight_scales,
                       buffers.weight_tensor_scale, buffers.input_tensor_scale,
                       buffers.output, shape.m, shape.k, shape.n);
    break;
  case TileVariant::Tile128x32:
    hipLaunchKernelGGL(
        (id64_geometry_kernel<8U, 2U>), grid, block, 0U, buffers.stream,
        buffers.activation, buffers.activation_scales, buffers.weight,
        buffers.weight_scales, buffers.weight_tensor_scale,
        buffers.input_tensor_scale, buffers.output, shape.m, shape.k, shape.n);
    break;
  case TileVariant::Tile128x16:
    hipLaunchKernelGGL(
        (id64_geometry_kernel<8U, 1U>), grid, block, 0U, buffers.stream,
        buffers.activation, buffers.activation_scales, buffers.weight,
        buffers.weight_scales, buffers.weight_tensor_scale,
        buffers.input_tensor_scale, buffers.output, shape.m, shape.k, shape.n);
    break;
  case TileVariant::Tile64x32:
    hipLaunchKernelGGL(
        (id64_geometry_kernel<4U, 2U>), grid, block, 0U, buffers.stream,
        buffers.activation, buffers.activation_scales, buffers.weight,
        buffers.weight_scales, buffers.weight_tensor_scale,
        buffers.input_tensor_scale, buffers.output, shape.m, shape.k, shape.n);
    break;
  }
  return hip_ok(hipGetLastError(), "tile sweep kernel launch");
}

struct SweepMeasurement final {
  std::array<float, kSweepMeasured> samples_us{};
  float median_us = 0.0F;
  float mad_us = 0.0F;
  float minimum_us = 0.0F;
  float maximum_us = 0.0F;
  std::size_t repeat_mismatches = 0U;
  std::vector<uint16_t> output;
  bool ran = false;
  bool deterministic = false;
};

float sweep_upper_median(std::array<float, kSweepMeasured> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2U];
}

bool measure_tile(const TileVariant variant, const Shape &shape,
                  const DeviceBuffers &buffers,
                  SweepMeasurement *const measurement) {
  const std::size_t elements = static_cast<std::size_t>(shape.m * shape.n);
  const std::size_t bytes = elements * sizeof(uint16_t);
  for (uint32_t warmup = 0U; warmup < kSweepWarmups; ++warmup) {
    if (!launch_tile(variant, shape, buffers)) {
      return false;
    }
  }
  if (!hip_ok(hipStreamSynchronize(buffers.stream),
              "tile sweep warmup synchronize")) {
    return false;
  }
  measurement->output.resize(elements);
  std::vector<uint16_t> current(elements);
  for (uint32_t iteration = 0U; iteration < kSweepMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(buffers.start, buffers.stream),
                "tile sweep timing start") ||
        !launch_tile(variant, shape, buffers) ||
        !hip_ok(hipEventRecord(buffers.stop, buffers.stream),
                "tile sweep timing stop") ||
        !hip_ok(hipEventSynchronize(buffers.stop),
                "tile sweep timing synchronize")) {
      return false;
    }
    float milliseconds = 0.0F;
    if (!hip_ok(hipEventElapsedTime(&milliseconds, buffers.start, buffers.stop),
                "tile sweep elapsed") ||
        !hip_ok(hipMemcpy(current.data(), buffers.output, bytes,
                          hipMemcpyDeviceToHost),
                "tile sweep output copy")) {
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
  measurement->median_us = sweep_upper_median(measurement->samples_us);
  std::array<float, kSweepMeasured> deviations{};
  for (std::size_t index = 0U; index < deviations.size(); ++index) {
    deviations[index] =
        std::abs(measurement->samples_us[index] - measurement->median_us);
  }
  measurement->mad_us = sweep_upper_median(deviations);
  measurement->minimum_us = *std::min_element(measurement->samples_us.begin(),
                                              measurement->samples_us.end());
  measurement->maximum_us = *std::max_element(measurement->samples_us.begin(),
                                              measurement->samples_us.end());
  measurement->ran = true;
  measurement->deterministic = measurement->repeat_mismatches == 0U;
  std::cout << "timing shape=" << shape.name
            << " variant=" << tile_variant_name(variant)
            << " tile_m=" << tile_rows(variant)
            << " tile_n=" << tile_columns(variant)
            << " warmups=" << kSweepWarmups << " measured=" << kSweepMeasured
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

std::size_t sweep_nonfinite(const std::vector<uint16_t> &values) {
  return static_cast<std::size_t>(
      std::count_if(values.begin(), values.end(), [](const uint16_t bits) {
        return (bits & UINT16_C(0x7f80)) == UINT16_C(0x7f80);
      }));
}

struct SweepShapeResult final {
  Shape shape{};
  std::array<SweepMeasurement, kTileVariants.size()> measurements;
  bool ok = false;
};

bool run_sweep_shape(const Shape &shape, SweepShapeResult *const result,
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
  for (const TileVariant variant : kTileVariants) {
    ok = measure_tile(
             variant, shape, buffers,
             &result->measurements[static_cast<std::size_t>(variant)]) &&
         ok;
  }
  const auto &control = result->measurements[0];
  if (control.ran) {
    const auto oracle = host_oracle(shape, inputs);
    ok = check_host_oracle(shape, Variant::Id64, oracle, control.output) && ok;
  }
  for (const TileVariant variant : kTileVariants) {
    const auto &measurement =
        result->measurements[static_cast<std::size_t>(variant)];
    if (!control.ran || !measurement.ran) {
      ok = false;
      continue;
    }
    const Comparison comparison = compare(control.output, measurement.output);
    const std::size_t nonfinite = sweep_nonfinite(measurement.output);
    std::cout << "bit_compare shape=" << shape.name << " pair=id64-vs-"
              << tile_variant_name(variant)
              << " bf16_mismatches=" << comparison.mismatches
              << " max_bf16_ulp=" << comparison.max_ulp
              << " nonfinite=" << nonfinite << " control_fnv64=0x" << std::hex
              << hash_bf16(control.output) << " candidate_fnv64=0x"
              << hash_bf16(measurement.output) << std::dec << " status="
              << (comparison.mismatches == 0U && nonfinite == 0U ? "PASS"
                                                                 : "FAIL")
              << "\n";
    ok = comparison.mismatches == 0U && nonfinite == 0U && ok;
  }
  cleanup(&buffers, cleanup_totals);
  result->ok = ok;
  std::cout << "shape_end name=" << shape.name
            << " status=" << (ok ? "PASS" : "FAIL") << "\n";
  return ok;
}

void print_sweep_weighted(
    const std::array<SweepShapeResult, kShapes.size()> &results) {
  double control_total = 0.0;
  uint64_t total_weight = 0U;
  for (const SweepShapeResult &result : results) {
    control_total +=
        result.measurements[0].median_us * result.shape.occurrences;
    total_weight += result.shape.occurrences;
  }
  for (const TileVariant variant : kTileVariants) {
    double total = 0.0;
    for (const SweepShapeResult &result : results) {
      total +=
          result.measurements[static_cast<std::size_t>(variant)].median_us *
          result.shape.occurrences;
    }
    std::cout << "weighted variant=" << tile_variant_name(variant)
              << " shapes=6 qwen_projection_weight=" << total_weight
              << " weighted_total_us=" << std::fixed << std::setprecision(3)
              << total << " weighted_mean_us=" << total / total_weight
              << " speedup_vs_id64="
              << (total > 0.0 ? control_total / total : 0.0) << "\n";
  }
  for (const uint64_t m : {UINT64_C(128), UINT64_C(512), UINT64_C(1024)}) {
    for (const TileVariant variant : kTileVariants) {
      double total = 0.0;
      double control = 0.0;
      uint64_t weight = 0U;
      for (const SweepShapeResult &result : results) {
        if (result.shape.m == m) {
          total +=
              result.measurements[static_cast<std::size_t>(variant)].median_us *
              result.shape.occurrences;
          control +=
              result.measurements[0].median_us * result.shape.occurrences;
          weight += result.shape.occurrences;
        }
      }
      std::cout << "weighted_by_m m=" << m
                << " variant=" << tile_variant_name(variant)
                << " qwen_projection_weight=" << weight
                << " weighted_mean_us=" << std::fixed << std::setprecision(3)
                << total / weight
                << " speedup_vs_id64=" << (total > 0.0 ? control / total : 0.0)
                << "\n";
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  int device = 0;
  if (argc > 2 || (argc == 2 && !parse_device(argv[1], &device))) {
    std::cerr << "usage: phase78_nvfp4_gfx1201_wmma_tile_sweep_probe "
                 "[DEVICE]\n";
    return EXIT_FAILURE;
  }
  if (!hip_ok(hipSetDevice(device), "tile sweep hipSetDevice")) {
    return EXIT_FAILURE;
  }
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device),
              "tile sweep hipGetDeviceProperties")) {
    return EXIT_FAILURE;
  }
  std::cout << "identity device=" << device << " name=" << properties.name
            << " pci_domain=" << properties.pciDomainID
            << " pci_bus=" << properties.pciBusID
            << " pci_device=" << properties.pciDeviceID
            << " arch=" << properties.gcnArchName
            << " total_global_mem=" << properties.totalGlobalMem << "\n";
  if (!exact_gfx1201(properties.gcnArchName)) {
    std::cerr << "unsupported exact target: expected gfx1201, observed "
              << properties.gcnArchName << "\n";
    return EXIT_FAILURE;
  }

  bool resources_ok = true;
  for (const TileVariant variant : kTileVariants) {
    const SweepResource resource = sweep_resource(variant, properties);
    resources_ok = resource.available && resource.vgpr <= kSweepMaximumVgpr &&
                   resource.lds <= kSweepMaximumLds && resource.scratch == 0U &&
                   resource.active_blocks > 0 && resources_ok;
  }

  CleanupTotals cleanup_totals;
  bool correctness_ok = true;
  for (const Shape &shape : kSweepBoundaryShapes) {
    SweepShapeResult ignored;
    correctness_ok =
        run_sweep_shape(shape, &ignored, &cleanup_totals) && correctness_ok;
  }
  std::array<SweepShapeResult, kShapes.size()> results{};
  for (std::size_t index = 0U; index < kShapes.size(); ++index) {
    correctness_ok =
        run_sweep_shape(kShapes[index], &results[index], &cleanup_totals) &&
        correctness_ok;
  }
  print_sweep_weighted(results);

  bool two_x = true;
  for (const SweepShapeResult &result : results) {
    if (result.shape.m != 1024U) {
      continue;
    }
    const float control = result.measurements[0].median_us;
    float best = std::numeric_limits<float>::infinity();
    TileVariant best_variant = TileVariant::Id64_128x64;
    for (const TileVariant variant :
         {TileVariant::Tile128x32, TileVariant::Tile128x16,
          TileVariant::Tile64x32}) {
      const float candidate =
          result.measurements[static_cast<std::size_t>(variant)].median_us;
      if (candidate < best) {
        best = candidate;
        best_variant = variant;
      }
    }
    const float target = result.shape.k == 5120U ? 4230.0F : 4183.0F;
    const bool relative = best <= control / 2.0F;
    const bool absolute = best <= target;
    std::cout << "two_x_gate shape=" << result.shape.name
              << " control_us=" << control
              << " best_variant=" << tile_variant_name(best_variant)
              << " best_us=" << best << " half_control_us=" << control / 2.0F
              << " absolute_target_us=" << target
              << " relative=" << (relative ? "PASS" : "FAIL")
              << " absolute=" << (absolute ? "PASS" : "FAIL") << "\n";
    two_x = relative && absolute && two_x;
  }

  const bool cleanup_ok =
      cleanup_totals.ok && cleanup_totals.allocations == cleanup_totals.frees;
  std::cout << "cleanup allocations=" << cleanup_totals.allocations
            << " frees=" << cleanup_totals.frees
            << " status=" << (cleanup_ok ? "PASS" : "FAIL") << "\n";
  const bool evidence_ok = correctness_ok && resources_ok && cleanup_ok;
  std::cout << "PHASE78_NVFP4_WMMA_TILE_SWEEP_EVIDENCE="
            << (evidence_ok ? "PASS" : "FAIL") << "\n";
  std::cout << "PHASE78_NVFP4_WMMA_TILE_SWEEP_DECISION="
            << (evidence_ok && two_x ? "GO" : "N0") << "\n";
  return evidence_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
