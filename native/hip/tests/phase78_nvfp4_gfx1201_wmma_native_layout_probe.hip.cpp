// Phase 78 standalone gfx1201 NVFP4 native accumulator-layout probe.
//
// The byte-for-byte ID64 control calls apply_data_layout<row_major>() after
// every K16 WMMA.  ROCm 7.14's gfx12 rocWMMA traits say that this particular
// conversion is a register-layout NOP.  The candidate pins those traits with
// compile-time assertions and consumes the native accumulator slots directly.
// Two follow-up schedules retain two or four independent WMMA contributions
// before applying the per-K16 activation/weight scales, testing whether that
// separation exposes useful WMMA/VALU overlap without changing sum order.
// No production source or selector is changed.

#define main phase78_scaled_ingress_embedded_main
#include "phase78_nvfp4_gfx1201_wmma_scaled_ingress_probe.hip.cpp"
#undef main

namespace {

constexpr uint32_t kNativeWarmups = 3U;
constexpr uint32_t kNativeMeasured = 10U;
constexpr uint64_t kNativeMaximumLds = UINT64_C(64) * 1024U;

static_assert(kNativeWarmups == kWarmups);
static_assert(kNativeMeasured == kMeasured);

constexpr std::array<Shape, 3> kNativeBoundaryShapes = {{
    {17U, 64U, 17U, 0U, "small-m17-n17"},
    {127U, 64U, 63U, 0U, "boundary-m127-n63"},
    {129U, 64U, 65U, 0U, "boundary-m129-n65"},
}};

// ID64 with only the syntactic row-major wrapper removed.  The static
// assertions document the exact installed-header contract that makes raw
// slot S in lane L represent row 8*(L/16)+S, column L%16 for a 16x16 tile.
template <uint32_t ColumnBatch>
__global__ __launch_bounds__(256, 1) void id64_native_layout_kernel(
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

  static_assert(ColumnBatch == 1U || ColumnBatch == 2U || ColumnBatch == 4U);
  static_assert(column_tiles % ColumnBatch == 0U);

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
  using RowMajorAccumulator =
      rocwmma::apply_data_layout_t<AccumulatorFragment, rocwmma::row_major>;
  using NativeIO = rocwmma::GetIOConfig_t<AccumulatorFragment>;
  using RowMajorIO = rocwmma::GetIOConfig_t<RowMajorAccumulator>;
  using NativeRegisterLayout = typename NativeIO::IOLayout::FragmentLayout;
  using RowMajorRegisterLayout = typename RowMajorIO::IOLayout::FragmentLayout;
  using RowMajorMatrixLayout = typename RowMajorIO::IOLayout::MatrixLayout;

  static_assert(
      rocwmma::is_layout_same_v<NativeRegisterLayout, RowMajorRegisterLayout>);
  static_assert(rocwmma::layout_traits<NativeRegisterLayout>::Format ==
                rocwmma::RegisterLayout::Format::ACC_INT_A_MAJOR);
  static_assert(rocwmma::layout_traits<RowMajorRegisterLayout>::Format ==
                rocwmma::RegisterLayout::Format::ACC_INT_A_MAJOR);
  static_assert(AccumulatorFragment::size() == 8U);
  static_assert(RowMajorIO::IOLayout::MmaDim == 16U);
  // RowInlineInt's storage-side vector width is one; the architecture-fixed
  // accumulator MaxVW used by its ACC_INT_A -> AOS_INT transform is eight.
  static_assert(RowMajorIO::IOLayout::MaxVW == 1U);
  static_assert(rocwmma::layout_traits<RowMajorMatrixLayout>::DimPerThread ==
                1U);
  static_assert(rocwmma::layout_traits<RowMajorMatrixLayout>::KPerThread == 8U);
  static_assert(rocwmma::interleave_idx_traits<
                rocwmma::interleave_idx<1U, 8U, 8U>>::IsNop);

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
            packed_activation + row * packed_row_bytes +
            inner_base / UINT64_C(2) + local_group * 2U));
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
      const uint64_t column = column_base +
                              static_cast<uint64_t>(column_tile) * tile_n +
                              local_column;
      uint16_t packed = 0U;
      if (column < n && inner_base + local_group * values_per_group < k) {
        packed = __builtin_nontemporal_load(reinterpret_cast<const uint16_t *>(
            packed_weight + column * packed_row_bytes +
            inner_base / UINT64_C(2) + local_group * 2U));
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
      for (uint32_t column_batch = 0U; column_batch < column_tiles;
           column_batch += ColumnBatch) {
        BFragment weight_fragments[ColumnBatch];
        AccumulatorFragment contributions[ColumnBatch];
#pragma unroll
        for (uint32_t batch_index = 0U; batch_index < ColumnBatch;
             ++batch_index) {
          rocwmma::fill_fragment(contributions[batch_index], 0.0F);
          rocwmma::load_matrix_sync(weight_fragments[batch_index],
                                    weight_tile[column_batch + batch_index] +
                                        scale_block * fragment_k,
                                    stage_k);
        }
#pragma unroll
        for (uint32_t batch_index = 0U; batch_index < ColumnBatch;
             ++batch_index) {
          rocwmma::mma_sync(contributions[batch_index], activation_fragment,
                            weight_fragments[batch_index],
                            contributions[batch_index]);
        }
#pragma unroll
        for (uint32_t batch_index = 0U; batch_index < ColumnBatch;
             ++batch_index) {
          const uint32_t column_tile = column_batch + batch_index;
#pragma unroll
          for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
            const uint32_t local_row =
                (lane / tile_n) * (output_values / wave_width) + slot;
            const uint32_t local_column = lane % tile_n;
            float term = contributions[batch_index][slot] *
                         activation_scale_tile[wave][local_row][scale_block];
            term *= weight_scale_tile[column_tile * tile_n + local_column]
                                     [scale_block];
            accumulators[column_tile][slot] += term;
          }
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

enum class NativeVariant : uint32_t {
  Id64 = 0U,
  NativeSlots = 1U,
  Batch2 = 2U,
  Batch4 = 3U,
};

constexpr std::array<NativeVariant, 4> kNativeVariants = {
    NativeVariant::Id64, NativeVariant::NativeSlots, NativeVariant::Batch2,
    NativeVariant::Batch4};

const char *native_variant_name(const NativeVariant variant) {
  switch (variant) {
  case NativeVariant::Id64:
    return "id64-apply-row-major-control";
  case NativeVariant::NativeSlots:
    return "native-accumulator-slots";
  case NativeVariant::Batch2:
    return "native-slots-batch2";
  case NativeVariant::Batch4:
    return "native-slots-batch4";
  }
  return "unknown";
}

const void *native_kernel_pointer(const NativeVariant variant) {
  switch (variant) {
  case NativeVariant::Id64:
    return reinterpret_cast<const void *>(id64_control_kernel);
  case NativeVariant::NativeSlots:
    return reinterpret_cast<const void *>(id64_native_layout_kernel<1U>);
  case NativeVariant::Batch2:
    return reinterpret_cast<const void *>(id64_native_layout_kernel<2U>);
  case NativeVariant::Batch4:
    return reinterpret_cast<const void *>(id64_native_layout_kernel<4U>);
  }
  return nullptr;
}

struct NativeResource final {
  bool available = false;
  int vgpr = 0;
  std::size_t lds = 0U;
  std::size_t scratch = 0U;
  int active_blocks = 0;
  int active_waves = 0;
  double occupancy = 0.0;
};

NativeResource native_resource(const NativeVariant variant,
                               const hipDeviceProp_t &properties) {
  NativeResource resource;
  const void *const kernel = native_kernel_pointer(variant);
  hipFuncAttributes attributes{};
  int active_blocks = 0;
  if (kernel == nullptr ||
      !hip_ok(hipFuncGetAttributes(&attributes, kernel),
              "native-layout hipFuncGetAttributes") ||
      !hip_ok(hipOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
                                                           kernel, 256, 0U),
              "native-layout occupancy")) {
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
  const bool pass = resource.lds <= kNativeMaximumLds &&
                    resource.scratch == 0U && active_blocks > 0;
  std::cout << "resources variant=" << native_variant_name(variant)
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

bool launch_native_variant(const NativeVariant variant, const Shape &shape,
                           const DeviceBuffers &buffers) {
  const dim3 grid(static_cast<uint32_t>((shape.n + 63U) / 64U),
                  static_cast<uint32_t>((shape.m + 127U) / 128U));
  switch (variant) {
  case NativeVariant::Id64:
    hipLaunchKernelGGL(
        id64_control_kernel, grid, dim3(256U), 0U, buffers.stream,
        buffers.activation, buffers.activation_scales, buffers.weight,
        buffers.weight_scales, buffers.weight_tensor_scale,
        buffers.input_tensor_scale, buffers.output, shape.m, shape.k, shape.n);
    break;
  case NativeVariant::NativeSlots:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(id64_native_layout_kernel<1U>), grid, dim3(256U), 0U,
        buffers.stream, buffers.activation, buffers.activation_scales,
        buffers.weight, buffers.weight_scales, buffers.weight_tensor_scale,
        buffers.input_tensor_scale, buffers.output, shape.m, shape.k, shape.n);
    break;
  case NativeVariant::Batch2:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(id64_native_layout_kernel<2U>), grid, dim3(256U), 0U,
        buffers.stream, buffers.activation, buffers.activation_scales,
        buffers.weight, buffers.weight_scales, buffers.weight_tensor_scale,
        buffers.input_tensor_scale, buffers.output, shape.m, shape.k, shape.n);
    break;
  case NativeVariant::Batch4:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(id64_native_layout_kernel<4U>), grid, dim3(256U), 0U,
        buffers.stream, buffers.activation, buffers.activation_scales,
        buffers.weight, buffers.weight_scales, buffers.weight_tensor_scale,
        buffers.input_tensor_scale, buffers.output, shape.m, shape.k, shape.n);
    break;
  }
  return hip_ok(hipGetLastError(), "native-layout kernel launch");
}

struct NativeMeasurement final {
  std::array<float, kNativeMeasured> samples_us{};
  float median_us = 0.0F;
  float mad_us = 0.0F;
  float minimum_us = 0.0F;
  float maximum_us = 0.0F;
  std::size_t repeat_mismatches = 0U;
  std::vector<uint16_t> output;
  bool ran = false;
  bool deterministic = false;
};

float native_upper_median(std::array<float, kNativeMeasured> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2U];
}

bool measure_native_variant(const NativeVariant variant, const Shape &shape,
                            const DeviceBuffers &buffers,
                            NativeMeasurement *const measurement) {
  const std::size_t elements = static_cast<std::size_t>(shape.m * shape.n);
  const std::size_t bytes = elements * sizeof(uint16_t);
  for (uint32_t warmup = 0U; warmup < kNativeWarmups; ++warmup) {
    if (!launch_native_variant(variant, shape, buffers)) {
      return false;
    }
  }
  if (!hip_ok(hipStreamSynchronize(buffers.stream),
              "native-layout warmup synchronize")) {
    return false;
  }
  measurement->output.resize(elements);
  std::vector<uint16_t> current(elements);
  for (uint32_t iteration = 0U; iteration < kNativeMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(buffers.start, buffers.stream),
                "native-layout timing start") ||
        !launch_native_variant(variant, shape, buffers) ||
        !hip_ok(hipEventRecord(buffers.stop, buffers.stream),
                "native-layout timing stop") ||
        !hip_ok(hipEventSynchronize(buffers.stop),
                "native-layout timing synchronize")) {
      return false;
    }
    float milliseconds = 0.0F;
    if (!hip_ok(hipEventElapsedTime(&milliseconds, buffers.start, buffers.stop),
                "native-layout timing elapsed") ||
        !hip_ok(hipMemcpy(current.data(), buffers.output, bytes,
                          hipMemcpyDeviceToHost),
                "native-layout output copy")) {
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
  measurement->median_us = native_upper_median(measurement->samples_us);
  std::array<float, kNativeMeasured> deviations{};
  for (std::size_t index = 0U; index < deviations.size(); ++index) {
    deviations[index] =
        std::abs(measurement->samples_us[index] - measurement->median_us);
  }
  measurement->mad_us = native_upper_median(deviations);
  measurement->minimum_us = *std::min_element(measurement->samples_us.begin(),
                                              measurement->samples_us.end());
  measurement->maximum_us = *std::max_element(measurement->samples_us.begin(),
                                              measurement->samples_us.end());
  measurement->ran = true;
  measurement->deterministic = measurement->repeat_mismatches == 0U;
  std::cout << "timing shape=" << shape.name
            << " variant=" << native_variant_name(variant)
            << " warmups=" << kNativeWarmups << " measured=" << kNativeMeasured
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

std::size_t native_nonfinite(const std::vector<uint16_t> &values) {
  return static_cast<std::size_t>(
      std::count_if(values.begin(), values.end(), [](const uint16_t bits) {
        return (bits & UINT16_C(0x7f80)) == UINT16_C(0x7f80);
      }));
}

bool native_host_oracle(const Shape &shape, const char *const variant,
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

struct NativeShapeResult final {
  Shape shape{};
  std::array<NativeMeasurement, kNativeVariants.size()> measurements;
  bool ok = false;
};

bool run_native_shape(const Shape &shape, NativeShapeResult *const result,
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
  for (const NativeVariant variant : kNativeVariants) {
    ok = measure_native_variant(
             variant, shape, buffers,
             &result->measurements[static_cast<std::size_t>(variant)]) &&
         ok;
  }
  if (ok) {
    const auto oracle = host_oracle(shape, inputs);
    for (const NativeVariant variant : kNativeVariants) {
      const NativeMeasurement &measurement =
          result->measurements[static_cast<std::size_t>(variant)];
      ok = native_host_oracle(shape, native_variant_name(variant), oracle,
                              measurement.output) &&
           ok;
      const std::size_t nonfinite = native_nonfinite(measurement.output);
      std::cout << "finite shape=" << shape.name
                << " variant=" << native_variant_name(variant)
                << " nonfinite=" << nonfinite
                << " status=" << (nonfinite == 0U ? "PASS" : "FAIL") << "\n";
      ok = nonfinite == 0U && ok;
    }
    const auto &control = result->measurements[0];
    for (std::size_t index = 1U; index < kNativeVariants.size(); ++index) {
      const NativeVariant variant = kNativeVariants[index];
      const auto &candidate = result->measurements[index];
      const Comparison comparison = compare(control.output, candidate.output);
      const bool bitwise = comparison.mismatches == 0U;
      std::cout << "bit_compare shape=" << shape.name << " pair=id64-vs-"
                << native_variant_name(variant)
                << " bf16_mismatches=" << comparison.mismatches
                << " max_bf16_ulp=" << comparison.max_ulp
                << " max_abs=" << std::setprecision(10) << comparison.max_abs
                << " max_rel=" << comparison.max_rel << " id64_fnv64=0x"
                << std::hex << hash_bf16(control.output)
                << " candidate_fnv64=0x" << hash_bf16(candidate.output)
                << std::dec << " status=" << (bitwise ? "PASS" : "FAIL")
                << "\n";
      ok = bitwise && ok;
    }
  }
  cleanup(&buffers, cleanup_totals);
  result->ok = ok;
  std::cout << "shape_end name=" << shape.name
            << " status=" << (ok ? "PASS" : "FAIL") << "\n";
  return ok;
}

double native_weighted_total(
    const std::array<NativeShapeResult, kShapes.size()> &results,
    const NativeVariant variant) {
  double total = 0.0;
  for (const NativeShapeResult &result : results) {
    total += result.measurements[static_cast<std::size_t>(variant)].median_us *
             result.shape.occurrences;
  }
  return total;
}

bool print_native_weighted(
    const std::array<NativeShapeResult, kShapes.size()> &results) {
  uint64_t total_weight = 0U;
  for (const NativeShapeResult &result : results) {
    total_weight += result.shape.occurrences;
  }
  const double control_total =
      native_weighted_total(results, NativeVariant::Id64);
  for (const NativeVariant variant : kNativeVariants) {
    const double total = native_weighted_total(results, variant);
    std::cout << "weighted variant=" << native_variant_name(variant)
              << " shapes=6 qwen_projection_weight=" << total_weight
              << " weighted_total_us=" << std::fixed << std::setprecision(3)
              << total << " weighted_mean_us=" << total / total_weight
              << " speedup_vs_id64="
              << (total > 0.0 ? control_total / total : 0.0) << "\n";
  }
  for (const NativeShapeResult &result : results) {
    const float control = result.measurements[0].median_us;
    for (std::size_t index = 1U; index < kNativeVariants.size(); ++index) {
      const NativeVariant variant = kNativeVariants[index];
      const float candidate = result.measurements[index].median_us;
      std::cout << "shape_speedup shape=" << result.shape.name
                << " variant=" << native_variant_name(variant)
                << " id64_us=" << control << " candidate_us=" << candidate
                << " speedup_vs_id64="
                << (candidate > 0.0F ? control / candidate : 0.0F) << "\n";
    }
  }
  bool any_two_x = false;
  for (std::size_t index = 1U; index < kNativeVariants.size(); ++index) {
    const NativeVariant variant = kNativeVariants[index];
    const double candidate_total = native_weighted_total(results, variant);
    const bool two_x = candidate_total <= control_total / 2.0;
    std::cout << "two_x_gate scope=weighted-six-shape variant="
              << native_variant_name(variant)
              << " id64_total_us=" << control_total
              << " candidate_total_us=" << candidate_total
              << " half_id64_us=" << control_total / 2.0
              << " status=" << (two_x ? "PASS" : "FAIL") << "\n";
    any_two_x = any_two_x || two_x;
  }
  return any_two_x;
}

} // namespace

int main(int argc, char **argv) {
  int device = 0;
  if (argc > 2 || (argc == 2 && !parse_device(argv[1], &device))) {
    std::cerr
        << "usage: phase78_nvfp4_gfx1201_wmma_native_layout_probe [DEVICE]\n";
    return EXIT_FAILURE;
  }
  if (!hip_ok(hipSetDevice(device), "native-layout hipSetDevice")) {
    return EXIT_FAILURE;
  }
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device),
              "native-layout hipGetDeviceProperties")) {
    return EXIT_FAILURE;
  }
  int runtime_version = 0;
  if (!hip_ok(hipRuntimeGetVersion(&runtime_version),
              "native-layout hipRuntimeGetVersion")) {
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
  std::cout << "layout_contract rocwmma=ROCm-7.14 gfx=gfx1201"
               " native_format=ACC_INT_A_MAJOR"
               " row_major_fragment_format=ACC_INT_A_MAJOR"
               " fragment_slots=8 storage_interleave=interleave<1,8,8>:NOP"
               " mapping=row:8*(lane/16)+slot,col:lane%16"
               " apply_data_layout_runtime_work=none\n";

  bool resources_ok = true;
  for (const NativeVariant variant : kNativeVariants) {
    const NativeResource resource = native_resource(variant, properties);
    resources_ok = resource.available && resource.lds <= kNativeMaximumLds &&
                   resource.scratch == 0U && resource.active_blocks > 0 &&
                   resources_ok;
  }

  CleanupTotals cleanup_totals;
  bool correctness_ok = resources_ok;
  for (const Shape &shape : kNativeBoundaryShapes) {
    NativeShapeResult ignored;
    correctness_ok =
        run_native_shape(shape, &ignored, &cleanup_totals) && correctness_ok;
  }

  std::array<NativeShapeResult, kShapes.size()> results{};
  if (correctness_ok) {
    for (std::size_t index = 0U; index < kShapes.size(); ++index) {
      correctness_ok =
          run_native_shape(kShapes[index], &results[index], &cleanup_totals) &&
          correctness_ok;
    }
  }
  const bool two_x = correctness_ok && print_native_weighted(results);
  const bool cleanup_ok =
      cleanup_totals.ok && cleanup_totals.allocations == cleanup_totals.frees;
  std::cout << "cleanup allocations=" << cleanup_totals.allocations
            << " frees=" << cleanup_totals.frees
            << " status=" << (cleanup_ok ? "PASS" : "FAIL") << "\n";
  const bool evidence_ok = correctness_ok && resources_ok && cleanup_ok;
  std::cout << "PHASE78_NVFP4_GFX1201_WMMA_NATIVE_LAYOUT_EVIDENCE="
            << (evidence_ok ? "PASS" : "FAIL") << "\n";
  std::cout << "PHASE78_NVFP4_GFX1201_WMMA_NATIVE_LAYOUT_DECISION="
            << (evidence_ok && two_x ? "GO" : "N0") << "\n";
  return evidence_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
