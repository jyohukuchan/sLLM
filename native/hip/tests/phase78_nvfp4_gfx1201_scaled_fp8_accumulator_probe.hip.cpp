// Phase 78 standalone gfx1201 NVFP4 scaled-FP8 accumulator probe.
//
// ID64 preserves each E2M1 block exactly in E4M3FN, performs one K16 WMMA,
// and applies the activation and weight block scales to every FP32 output.
// These deliberately approximate candidates instead multiply each E2M1 value
// by its E4M3FN block scale at LDS ingress and round it back to E4M3FN.  The
// stage candidate accumulates both K16 products in one K32 fragment before a
// scalar output add.  The persistent candidate retains four FP32 accumulator
// fragments across all K and has no scalar accumulation in the hot loop.
// Tensor-global scales remain in the BF16 epilogue.
//
// This is a developer-only probe.  It does not alter production dispatch.

#define main phase78_scaled_ingress_embedded_main
#include "phase78_nvfp4_gfx1201_wmma_scaled_ingress_probe.hip.cpp"
#undef main

namespace {

constexpr uint64_t kScaledFp8MaximumLds = UINT64_C(64) * 1024U;
constexpr double kScaledFp8AccuracyTolerance = 0.01;

constexpr std::array<Shape, 3> kScaledFp8BoundaryShapes = {{
    {17U, 64U, 17U, 0U, "small-m17-n17"},
    {127U, 64U, 63U, 0U, "boundary-m127-n63"},
    {129U, 64U, 65U, 0U, "boundary-m129-n65"},
}};

enum class ScaledFp8Variant : uint32_t {
  Id64 = 0U,
  StageK32 = 1U,
  PersistentK = 2U,
};

constexpr std::array<ScaledFp8Variant, 3> kScaledFp8Variants = {
    ScaledFp8Variant::Id64, ScaledFp8Variant::StageK32,
    ScaledFp8Variant::PersistentK};

const char *scaled_fp8_variant_name(const ScaledFp8Variant variant) {
  switch (variant) {
  case ScaledFp8Variant::Id64:
    return "id64-exact-block-scale";
  case ScaledFp8Variant::StageK32:
    return "scaled-fp8-stagek32-pair";
  case ScaledFp8Variant::PersistentK:
    return "scaled-fp8-persistent-k";
  }
  return "unknown";
}

enum class ScaleProfile : uint32_t { Realistic = 0U, Stress = 1U };

constexpr std::array<ScaleProfile, 2> kScaleProfiles = {ScaleProfile::Realistic,
                                                        ScaleProfile::Stress};

const char *scale_profile_name(const ScaleProfile profile) {
  switch (profile) {
  case ScaleProfile::Realistic:
    return "realistic-nonsaturating";
  case ScaleProfile::Stress:
    return "stress-extremes";
  }
  return "unknown";
}

[[maybe_unused]] __device__ __forceinline__ uint32_t
scaled_e2m1x4_to_e4m3fn(const uint16_t packed, const float scale) {
  // HIP 7.14's gfx12 float2 helper computes a saturated temporary but feeds
  // the original float2 to cvt_pk_fp8.  Clamp explicitly so stress inputs
  // cannot round to the OCP E4M3FN NaN code 0x7f.
  const auto clamp = [](const float value) {
    return __builtin_amdgcn_fmed3f(value, 448.0F, -448.0F);
  };
  const float first = clamp(sllm_lowp::e2m1_to_float(static_cast<uint8_t>(
                                packed & UINT16_C(0x000f))) *
                            scale);
  const float second = clamp(sllm_lowp::e2m1_to_float(static_cast<uint8_t>(
                                 (packed >> 4U) & UINT16_C(0x000f))) *
                             scale);
  const float third = clamp(sllm_lowp::e2m1_to_float(static_cast<uint8_t>(
                                (packed >> 8U) & UINT16_C(0x000f))) *
                            scale);
  const float fourth = clamp(sllm_lowp::e2m1_to_float(static_cast<uint8_t>(
                                 (packed >> 12U) & UINT16_C(0x000f))) *
                             scale);
  const uint16_t low = __hip_cvt_float2_to_fp8x2(make_float2(first, second),
                                                 __HIP_SATFINITE, __HIP_E4M3);
  const uint16_t high = __hip_cvt_float2_to_fp8x2(make_float2(third, fourth),
                                                  __HIP_SATFINITE, __HIP_E4M3);
  return static_cast<uint32_t>(low) | (static_cast<uint32_t>(high) << 16U);
}

template <bool Persistent>
__global__ __launch_bounds__(256, 1) void scaled_fp8_accumulator_kernel(
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
  constexpr uint32_t groups_per_row = stage_k / values_per_group;
  constexpr uint32_t groups_per_tile = tile_values / values_per_group;
  constexpr uint32_t output_values = tile_m * tile_n;
  constexpr uint32_t output_values_per_lane = output_values / wave_width;

  __shared__ __align__(4)
      rocwmma::float8_t activation_tile[waves_per_workgroup][tile_values];
  __shared__ __align__(4)
      rocwmma::float8_t weight_tile[column_tiles][tile_values];
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

  float stage_accumulators[column_tiles][output_values_per_lane] = {};
  AccumulatorFragment persistent_accumulators[column_tiles];
  if constexpr (Persistent) {
#pragma unroll
    for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
      rocwmma::fill_fragment(persistent_accumulators[column_tile], 0.0F);
    }
  }

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
      const uint32_t local_row = wave_group / groups_per_row;
      const uint32_t local_group = wave_group - local_row * groups_per_row;
      const uint32_t scale_block =
          local_group / (fragment_k / values_per_group);
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      uint16_t packed = 0U;
      float scale = 0.0F;
      if (row < m && block < blocks_per_row) {
        packed = __builtin_nontemporal_load(reinterpret_cast<const uint16_t *>(
            packed_activation + row * packed_row_bytes +
            inner_base / UINT64_C(2) + local_group * 2U));
        scale = sllm_lowp::e4m3fn_to_float(__builtin_nontemporal_load(
            activation_block_scales + row * blocks_per_row + block));
      }
      activation_groups[group] = scaled_e2m1x4_to_e4m3fn(packed, scale);
    }

    for (uint32_t group = thread; group < column_tiles * groups_per_tile;
         group += blockDim.x) {
      const uint32_t column_tile = group / groups_per_tile;
      const uint32_t tile_group = group - column_tile * groups_per_tile;
      const uint32_t local_column = tile_group / groups_per_row;
      const uint32_t local_group = tile_group - local_column * groups_per_row;
      const uint32_t scale_block =
          local_group / (fragment_k / values_per_group);
      const uint64_t column = column_base +
                              static_cast<uint64_t>(column_tile) * tile_n +
                              local_column;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      uint16_t packed = 0U;
      float scale = 0.0F;
      if (column < n && block < blocks_per_row) {
        packed = __builtin_nontemporal_load(reinterpret_cast<const uint16_t *>(
            packed_weight + column * packed_row_bytes +
            inner_base / UINT64_C(2) + local_group * 2U));
        scale = sllm_lowp::e4m3fn_to_float(__builtin_nontemporal_load(
            weight_block_scales + column * blocks_per_row + block));
      }
      weight_groups[group] = scaled_e2m1x4_to_e4m3fn(packed, scale);
    }
    __syncthreads();

    AFragment activation_fragments[scale_blocks_per_stage];
#pragma unroll
    for (uint32_t scale_block = 0U; scale_block < scale_blocks_per_stage;
         ++scale_block) {
      rocwmma::load_matrix_sync(
          activation_fragments[scale_block],
          activation_tile[wave] + scale_block * fragment_k, stage_k);
    }

    if constexpr (Persistent) {
#pragma unroll
      for (uint32_t column_tile = 0U; column_tile < column_tiles;
           ++column_tile) {
#pragma unroll
        for (uint32_t scale_block = 0U; scale_block < scale_blocks_per_stage;
             ++scale_block) {
          BFragment weight_fragment;
          rocwmma::load_matrix_sync(
              weight_fragment,
              weight_tile[column_tile] + scale_block * fragment_k, stage_k);
          rocwmma::mma_sync(persistent_accumulators[column_tile],
                            activation_fragments[scale_block], weight_fragment,
                            persistent_accumulators[column_tile]);
        }
      }
    } else {
#pragma unroll
      for (uint32_t column_tile = 0U; column_tile < column_tiles;
           ++column_tile) {
        AccumulatorFragment contribution;
        rocwmma::fill_fragment(contribution, 0.0F);
#pragma unroll
        for (uint32_t scale_block = 0U; scale_block < scale_blocks_per_stage;
             ++scale_block) {
          BFragment weight_fragment;
          rocwmma::load_matrix_sync(
              weight_fragment,
              weight_tile[column_tile] + scale_block * fragment_k, stage_k);
          rocwmma::mma_sync(contribution, activation_fragments[scale_block],
                            weight_fragment, contribution);
        }
        const auto row_major =
            rocwmma::apply_data_layout<rocwmma::row_major>(contribution);
#pragma unroll
        for (uint32_t slot = 0U; slot < output_values_per_lane; ++slot) {
          stage_accumulators[column_tile][slot] += row_major[slot];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
    if constexpr (Persistent) {
      const auto row_major = rocwmma::apply_data_layout<rocwmma::row_major>(
          persistent_accumulators[column_tile]);
#pragma unroll
      for (uint32_t slot = 0U; slot < output_values_per_lane; ++slot) {
        const uint32_t local_row =
            (lane / tile_n) * output_values_per_lane + slot;
        const uint32_t local_column = lane % tile_n;
        const uint64_t row = row_tile_base + local_row;
        const uint64_t column =
            column_base + column_tile * tile_n + local_column;
        if (row < m && column < n) {
          output[row * n + column] = bf16_rne(row_major[slot] * tensor_scale);
        }
      }
    } else {
#pragma unroll
      for (uint32_t slot = 0U; slot < output_values_per_lane; ++slot) {
        const uint32_t local_row =
            (lane / tile_n) * output_values_per_lane + slot;
        const uint32_t local_column = lane % tile_n;
        const uint64_t row = row_tile_base + local_row;
        const uint64_t column =
            column_base + column_tile * tile_n + local_column;
        if (row < m && column < n) {
          output[row * n + column] =
              bf16_rne(stage_accumulators[column_tile][slot] * tensor_scale);
        }
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

const void *scaled_fp8_kernel_pointer(const ScaledFp8Variant variant) {
  switch (variant) {
  case ScaledFp8Variant::Id64:
    return reinterpret_cast<const void *>(id64_control_kernel);
  case ScaledFp8Variant::StageK32:
    return reinterpret_cast<const void *>(scaled_fp8_accumulator_kernel<false>);
  case ScaledFp8Variant::PersistentK:
    return reinterpret_cast<const void *>(scaled_fp8_accumulator_kernel<true>);
  }
  return nullptr;
}

ResourceInfo scaled_fp8_resource_info(const ScaledFp8Variant variant,
                                      const hipDeviceProp_t &properties) {
  ResourceInfo result;
  hipFuncAttributes attributes{};
  int active_blocks = 0;
  const void *const kernel = scaled_fp8_kernel_pointer(variant);
  if (kernel == nullptr ||
      !hip_ok(hipFuncGetAttributes(&attributes, kernel),
              "scaled-fp8 hipFuncGetAttributes") ||
      !hip_ok(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                  &active_blocks, kernel, static_cast<int>(kThreads), 0U),
              "scaled-fp8 occupancy")) {
    return result;
  }
  result.available = true;
  result.vgpr = attributes.numRegs;
  result.lds = attributes.sharedSizeBytes;
  result.scratch = attributes.localSizeBytes;
  result.max_threads = attributes.maxThreadsPerBlock;
  result.active_blocks_per_cu = active_blocks;
  result.occupancy = properties.maxThreadsPerMultiProcessor == 0
                         ? 0.0
                         : static_cast<double>(active_blocks * kThreads) /
                               properties.maxThreadsPerMultiProcessor;
  const bool pass = result.lds <= kScaledFp8MaximumLds &&
                    result.scratch == 0U && active_blocks > 0;
  std::cout << "resources variant=" << scaled_fp8_variant_name(variant)
            << " vgpr=" << result.vgpr << " lds=" << result.lds
            << " scratch_per_thread=" << result.scratch
            << " max_threads=" << result.max_threads
            << " active_blocks_per_cu=" << result.active_blocks_per_cu
            << " active_waves_per_cu=" << result.active_blocks_per_cu * 8
            << " occupancy=" << std::fixed << std::setprecision(6)
            << result.occupancy << " status=" << (pass ? "PASS" : "FAIL")
            << "\n";
  return result;
}

bool launch_scaled_fp8(const ScaledFp8Variant variant, const Shape &shape,
                       const DeviceBuffers &buffers) {
  const dim3 grid(static_cast<uint32_t>((shape.n + 63U) / 64U),
                  static_cast<uint32_t>((shape.m + 127U) / 128U));
  const dim3 block(kThreads);
  switch (variant) {
  case ScaledFp8Variant::Id64:
    hipLaunchKernelGGL(id64_control_kernel, grid, block, 0U, buffers.stream,
                       buffers.activation, buffers.activation_scales,
                       buffers.weight, buffers.weight_scales,
                       buffers.weight_tensor_scale, buffers.input_tensor_scale,
                       buffers.output, shape.m, shape.k, shape.n);
    break;
  case ScaledFp8Variant::StageK32:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(scaled_fp8_accumulator_kernel<false>), grid, block, 0U,
        buffers.stream, buffers.activation, buffers.activation_scales,
        buffers.weight, buffers.weight_scales, buffers.weight_tensor_scale,
        buffers.input_tensor_scale, buffers.output, shape.m, shape.k, shape.n);
    break;
  case ScaledFp8Variant::PersistentK:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(scaled_fp8_accumulator_kernel<true>), grid, block, 0U,
        buffers.stream, buffers.activation, buffers.activation_scales,
        buffers.weight, buffers.weight_scales, buffers.weight_tensor_scale,
        buffers.input_tensor_scale, buffers.output, shape.m, shape.k, shape.n);
    break;
  }
  return hip_ok(hipGetLastError(), "scaled-fp8 kernel launch");
}

struct ScaledFp8Measurement final {
  std::array<float, kMeasured> samples_us{};
  float median_us = 0.0F;
  float mad_us = 0.0F;
  float minimum_us = 0.0F;
  float maximum_us = 0.0F;
  std::size_t repeat_mismatches = 0U;
  std::vector<uint16_t> output;
  bool ran = false;
  bool deterministic = false;
};

float scaled_fp8_upper_median(std::array<float, kMeasured> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2U];
}

bool measure_scaled_fp8(const ScaledFp8Variant variant,
                        const ScaleProfile profile, const Shape &shape,
                        const DeviceBuffers &buffers,
                        ScaledFp8Measurement *const result) {
  const std::size_t elements = static_cast<std::size_t>(shape.m * shape.n);
  const std::size_t bytes = elements * sizeof(uint16_t);
  for (uint32_t warmup = 0U; warmup < kWarmups; ++warmup) {
    if (!launch_scaled_fp8(variant, shape, buffers)) {
      return false;
    }
  }
  if (!hip_ok(hipStreamSynchronize(buffers.stream),
              "scaled-fp8 warmup synchronize")) {
    return false;
  }
  result->output.resize(elements);
  std::vector<uint16_t> current(elements);
  for (uint32_t iteration = 0U; iteration < kMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(buffers.start, buffers.stream),
                "scaled-fp8 timing start") ||
        !launch_scaled_fp8(variant, shape, buffers) ||
        !hip_ok(hipEventRecord(buffers.stop, buffers.stream),
                "scaled-fp8 timing stop") ||
        !hip_ok(hipEventSynchronize(buffers.stop),
                "scaled-fp8 timing synchronize")) {
      return false;
    }
    float milliseconds = 0.0F;
    if (!hip_ok(hipEventElapsedTime(&milliseconds, buffers.start, buffers.stop),
                "scaled-fp8 timing elapsed") ||
        !hip_ok(hipMemcpy(current.data(), buffers.output, bytes,
                          hipMemcpyDeviceToHost),
                "scaled-fp8 copy output")) {
      return false;
    }
    result->samples_us[iteration] = milliseconds * 1000.0F;
    if (iteration == 0U) {
      result->output = current;
    } else {
      for (std::size_t index = 0U; index < elements; ++index) {
        result->repeat_mismatches +=
            static_cast<std::size_t>(current[index] != result->output[index]);
      }
    }
  }
  result->median_us = scaled_fp8_upper_median(result->samples_us);
  std::array<float, kMeasured> deviations{};
  for (std::size_t index = 0U; index < deviations.size(); ++index) {
    deviations[index] = std::abs(result->samples_us[index] - result->median_us);
  }
  result->mad_us = scaled_fp8_upper_median(deviations);
  result->minimum_us =
      *std::min_element(result->samples_us.begin(), result->samples_us.end());
  result->maximum_us =
      *std::max_element(result->samples_us.begin(), result->samples_us.end());
  result->ran = true;
  result->deterministic = result->repeat_mismatches == 0U;
  std::cout << "timing profile=" << scale_profile_name(profile)
            << " shape=" << shape.name
            << " variant=" << scaled_fp8_variant_name(variant)
            << " warmups=" << kWarmups << " measured=" << kMeasured
            << " samples_us=";
  for (std::size_t index = 0U; index < result->samples_us.size(); ++index) {
    if (index != 0U) {
      std::cout << ',';
    }
    std::cout << std::fixed << std::setprecision(3)
              << result->samples_us[index];
  }
  std::cout << " median_us=" << result->median_us
            << " mad_us=" << result->mad_us << " min_us=" << result->minimum_us
            << " max_us=" << result->maximum_us
            << " repeat_bf16_mismatches=" << result->repeat_mismatches
            << " deterministic=" << (result->deterministic ? "PASS" : "FAIL")
            << "\n";
  return result->deterministic;
}

uint8_t profile_scale_code(const ScaleProfile profile, const uint64_t index,
                           const uint32_t seed) {
  const uint32_t mixed = mix32(static_cast<uint32_t>(index) ^ seed);
  if (profile == ScaleProfile::Realistic) {
    // Synthetic, non-saturating block scales from 0.125 through 4.0.  The
    // interval includes every E4M3 mantissa, so this is not a power-of-two
    // exactness shortcut; max |E2M1*scale| is 24, well below E4M3FN's 448.
    return static_cast<uint8_t>(UINT8_C(0x20) + mixed % UINT32_C(41));
  }
  // Positive finite subnormal, ordinary, and near-maximum E4M3FN scales.
  // Values above 448/6 deliberately exercise saturation for large E2M1 codes.
  constexpr std::array<uint8_t, 16> stress = {
      UINT8_C(0x01), UINT8_C(0x02), UINT8_C(0x04), UINT8_C(0x07),
      UINT8_C(0x08), UINT8_C(0x10), UINT8_C(0x18), UINT8_C(0x20),
      UINT8_C(0x60), UINT8_C(0x68), UINT8_C(0x70), UINT8_C(0x74),
      UINT8_C(0x78), UINT8_C(0x7a), UINT8_C(0x7c), UINT8_C(0x7e)};
  return stress[mixed & UINT32_C(15)];
}

HostInputs make_profile_inputs(const Shape &shape, const ScaleProfile profile) {
  HostInputs inputs = make_inputs(shape);
  for (std::size_t index = 0U; index < inputs.activation_scales.size();
       ++index) {
    inputs.activation_scales[index] =
        profile_scale_code(profile, index, kSeed ^ UINT32_C(0x6a09e667));
  }
  for (std::size_t index = 0U; index < inputs.weight_scales.size(); ++index) {
    inputs.weight_scales[index] =
        profile_scale_code(profile, index, kSeed ^ UINT32_C(0xbb67ae85));
  }
  return inputs;
}

void print_scale_profile_stats(const Shape &shape, const ScaleProfile profile,
                               const HostInputs &inputs) {
  double minimum = std::numeric_limits<double>::infinity();
  double maximum = 0.0;
  std::size_t subnormal = 0U;
  std::size_t saturation_risk = 0U;
  const auto consume = [&](const std::vector<uint8_t> &scales) {
    for (const uint8_t code : scales) {
      const double value = e4m3fn_to_float(code);
      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
      subnormal += static_cast<std::size_t>((code & UINT8_C(0x78)) == 0U &&
                                            (code & UINT8_C(0x07)) != 0U);
      saturation_risk += static_cast<std::size_t>(value * 6.0 > 448.0);
    }
  };
  consume(inputs.activation_scales);
  consume(inputs.weight_scales);
  const std::size_t total =
      inputs.activation_scales.size() + inputs.weight_scales.size();
  std::cout << "scale_profile profile=" << scale_profile_name(profile)
            << " shape=" << shape.name << " scales=" << total
            << " min=" << minimum << " max=" << maximum
            << " subnormal=" << subnormal
            << " saturation_risk_blocks=" << saturation_risk << "\n";
}

struct ScaledFp8OracleStats final {
  std::size_t bf16_mismatches = 0U;
  uint32_t max_ulp = 0U;
  double max_abs = 0.0;
  double max_normalized = 0.0;
  bool pass = false;
};

ScaledFp8OracleStats
check_scaled_fp8_oracle(const Shape &shape, const ScaleProfile profile,
                        const ScaledFp8Variant variant,
                        const std::array<OraclePoint, kOracleSamples> &oracle,
                        const std::vector<uint16_t> &output) {
  ScaledFp8OracleStats result;
  for (const OraclePoint &point : oracle) {
    const uint16_t observed_bits = output[point.index];
    const double observed = host_bf16_to_float(observed_bits);
    if (!std::isfinite(observed)) {
      result.max_abs = std::numeric_limits<double>::infinity();
      result.max_normalized = std::numeric_limits<double>::infinity();
      ++result.bf16_mismatches;
      result.max_ulp = UINT32_MAX;
      continue;
    }
    const double absolute = std::abs(observed - point.expected);
    result.max_abs = std::max(result.max_abs, absolute);
    result.max_normalized =
        std::max(result.max_normalized,
                 absolute / std::max(point.absolute_sum,
                                     std::numeric_limits<double>::min()));
    result.bf16_mismatches +=
        static_cast<std::size_t>(observed_bits != point.expected_bf16);
    const uint32_t lhs = ordered_bf16(observed_bits);
    const uint32_t rhs = ordered_bf16(point.expected_bf16);
    result.max_ulp =
        std::max(result.max_ulp, lhs > rhs ? lhs - rhs : rhs - lhs);
  }
  result.pass = std::isfinite(result.max_normalized) &&
                result.max_normalized <= kScaledFp8AccuracyTolerance;
  std::cout << "host_double_nvfp4_oracle profile="
            << scale_profile_name(profile) << " shape=" << shape.name
            << " variant=" << scaled_fp8_variant_name(variant)
            << " samples=" << kOracleSamples
            << " bf16_mismatches=" << result.bf16_mismatches
            << " max_bf16_ulp=" << result.max_ulp
            << " max_abs=" << std::setprecision(10) << result.max_abs
            << " max_normalized_error=" << result.max_normalized
            << " tolerance=" << kScaledFp8AccuracyTolerance
            << " status=" << (result.pass ? "PASS" : "FAIL") << "\n";
  return result;
}

std::size_t scaled_fp8_nonfinite(const std::vector<uint16_t> &values) {
  return static_cast<std::size_t>(
      std::count_if(values.begin(), values.end(), [](const uint16_t bits) {
        return (bits & UINT16_C(0x7f80)) == UINT16_C(0x7f80);
      }));
}

struct ScaledFp8ShapeResult final {
  Shape shape{};
  ScaleProfile profile = ScaleProfile::Realistic;
  std::array<ScaledFp8Measurement, kScaledFp8Variants.size()> measurements;
  std::array<ScaledFp8OracleStats, kScaledFp8Variants.size()> oracle_stats;
  bool evidence_ok = false;
};

bool run_scaled_fp8_shape(const Shape &shape, const ScaleProfile profile,
                          ScaledFp8ShapeResult *const result,
                          CleanupTotals *const cleanup_totals) {
  std::cout << "shape_begin profile=" << scale_profile_name(profile)
            << " name=" << shape.name << " m=" << shape.m << " k=" << shape.k
            << " n=" << shape.n << " occurrences=" << shape.occurrences << "\n";
  const HostInputs inputs = make_profile_inputs(shape, profile);
  print_scale_profile_stats(shape, profile, inputs);
  DeviceBuffers buffers;
  if (!allocate_and_upload(shape, inputs, &buffers)) {
    cleanup(&buffers, cleanup_totals);
    return false;
  }
  result->shape = shape;
  result->profile = profile;
  bool evidence_ok = true;
  for (const ScaledFp8Variant variant : kScaledFp8Variants) {
    const std::size_t index = static_cast<std::size_t>(variant);
    evidence_ok = measure_scaled_fp8(variant, profile, shape, buffers,
                                     &result->measurements[index]) &&
                  evidence_ok;
  }
  if (evidence_ok) {
    const auto oracle = host_oracle(shape, inputs);
    for (const ScaledFp8Variant variant : kScaledFp8Variants) {
      const std::size_t index = static_cast<std::size_t>(variant);
      const auto &output = result->measurements[index].output;
      result->oracle_stats[index] =
          check_scaled_fp8_oracle(shape, profile, variant, oracle, output);
      const std::size_t nonfinite = scaled_fp8_nonfinite(output);
      std::cout << "finite profile=" << scale_profile_name(profile)
                << " shape=" << shape.name
                << " variant=" << scaled_fp8_variant_name(variant)
                << " nonfinite=" << nonfinite
                << " status=" << (nonfinite == 0U ? "PASS" : "FAIL") << "\n";
      evidence_ok = nonfinite == 0U && evidence_ok;
      if (variant == ScaledFp8Variant::Id64) {
        evidence_ok = result->oracle_stats[index].pass && evidence_ok;
      }
    }
    const auto &control = result->measurements[0].output;
    for (std::size_t index = 1U; index < kScaledFp8Variants.size(); ++index) {
      const Comparison comparison =
          compare(control, result->measurements[index].output);
      std::cout << "bf16_compare profile=" << scale_profile_name(profile)
                << " shape=" << shape.name << " pair=id64-vs-"
                << scaled_fp8_variant_name(kScaledFp8Variants[index])
                << " mismatches=" << comparison.mismatches
                << " max_bf16_ulp=" << comparison.max_ulp
                << " max_abs=" << std::setprecision(10) << comparison.max_abs
                << " max_rel=" << comparison.max_rel << " id64_fnv64=0x"
                << std::hex << hash_bf16(control) << " candidate_fnv64=0x"
                << hash_bf16(result->measurements[index].output) << std::dec
                << "\n";
    }
  }
  cleanup(&buffers, cleanup_totals);
  result->evidence_ok = evidence_ok;
  std::cout << "shape_end profile=" << scale_profile_name(profile)
            << " name=" << shape.name
            << " evidence=" << (evidence_ok ? "PASS" : "FAIL") << "\n";
  return evidence_ok;
}

template <std::size_t ShapeCount>
double scaled_fp8_weighted_total(
    const std::array<ScaledFp8ShapeResult, ShapeCount> &results,
    const ScaledFp8Variant variant) {
  double total = 0.0;
  for (const ScaledFp8ShapeResult &result : results) {
    total += result.measurements[static_cast<std::size_t>(variant)].median_us *
             result.shape.occurrences;
  }
  return total;
}

template <std::size_t BoundaryCount, std::size_t ShapeCount>
std::array<bool, kScaledFp8Variants.size()> print_scaled_fp8_summary(
    const ScaleProfile profile,
    const std::array<ScaledFp8ShapeResult, BoundaryCount> &boundaries,
    const std::array<ScaledFp8ShapeResult, ShapeCount> &results) {
  std::array<bool, kScaledFp8Variants.size()> go{};
  std::array<bool, kScaledFp8Variants.size()> accuracy{};
  accuracy.fill(true);
  std::array<double, kScaledFp8Variants.size()> maximum_normalized{};
  const auto consume_oracles = [&](const auto &set) {
    for (const ScaledFp8ShapeResult &result : set) {
      for (std::size_t index = 0U; index < kScaledFp8Variants.size(); ++index) {
        accuracy[index] = accuracy[index] && result.oracle_stats[index].pass;
        maximum_normalized[index] =
            std::max(maximum_normalized[index],
                     result.oracle_stats[index].max_normalized);
      }
    }
  };
  consume_oracles(boundaries);
  consume_oracles(results);

  uint64_t total_weight = 0U;
  for (const ScaledFp8ShapeResult &result : results) {
    total_weight += result.shape.occurrences;
  }
  const double control_total =
      scaled_fp8_weighted_total(results, ScaledFp8Variant::Id64);
  for (std::size_t index = 0U; index < kScaledFp8Variants.size(); ++index) {
    const ScaledFp8Variant variant = kScaledFp8Variants[index];
    const double total = scaled_fp8_weighted_total(results, variant);
    const bool two_x = index != 0U && total <= control_total / 2.0;
    go[index] = accuracy[index] && two_x;
    std::cout << "summary profile=" << scale_profile_name(profile)
              << " variant=" << scaled_fp8_variant_name(variant)
              << " accuracy_max_normalized=" << std::setprecision(10)
              << maximum_normalized[index]
              << " accuracy_tolerance=" << kScaledFp8AccuracyTolerance
              << " accuracy_status=" << (accuracy[index] ? "PASS" : "FAIL")
              << " qwen_projection_weight=" << total_weight
              << " weighted_total_us=" << std::fixed << std::setprecision(3)
              << total << " weighted_mean_us=" << total / total_weight
              << " speedup_vs_id64="
              << (total > 0.0 ? control_total / total : 0.0)
              << " two_x_status=" << (two_x ? "PASS" : "FAIL") << "\n";
  }
  for (const ScaledFp8ShapeResult &result : results) {
    const float control = result.measurements[0].median_us;
    for (std::size_t index = 1U; index < kScaledFp8Variants.size(); ++index) {
      const float candidate = result.measurements[index].median_us;
      std::cout << "shape_speedup profile=" << scale_profile_name(profile)
                << " shape=" << result.shape.name << " variant="
                << scaled_fp8_variant_name(kScaledFp8Variants[index])
                << " id64_us=" << control << " candidate_us=" << candidate
                << " speedup_vs_id64="
                << (candidate > 0.0F ? control / candidate : 0.0F) << "\n";
    }
  }
  return go;
}

} // namespace

int main(int argc, char **argv) {
  int device = 0;
  if (argc > 2 || (argc == 2 && !parse_device(argv[1], &device))) {
    std::cerr << "usage: phase78_nvfp4_gfx1201_scaled_fp8_accumulator_probe "
                 "[DEVICE]\n";
    return EXIT_FAILURE;
  }
  if (!hip_ok(hipSetDevice(device), "scaled-fp8 hipSetDevice")) {
    return EXIT_FAILURE;
  }
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device),
              "scaled-fp8 hipGetDeviceProperties")) {
    return EXIT_FAILURE;
  }
  int runtime_version = 0;
  if (!hip_ok(hipRuntimeGetVersion(&runtime_version),
              "scaled-fp8 hipRuntimeGetVersion")) {
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
  for (const ScaledFp8Variant variant : kScaledFp8Variants) {
    const ResourceInfo resource = scaled_fp8_resource_info(variant, properties);
    resources_ok = resource.available && resource.lds <= kScaledFp8MaximumLds &&
                   resource.scratch == 0U &&
                   resource.active_blocks_per_cu > 0 && resources_ok;
  }
  if (!resources_ok) {
    std::cout << "PHASE78_NVFP4_GFX1201_SCALED_FP8_EVIDENCE=FAIL\n";
    std::cout << "PHASE78_NVFP4_GFX1201_SCALED_FP8_DECISION=N0\n";
    return EXIT_FAILURE;
  }

  CleanupTotals cleanup_totals;
  bool evidence_ok = true;
  std::array<std::array<ScaledFp8ShapeResult, kScaledFp8BoundaryShapes.size()>,
             kScaleProfiles.size()>
      boundaries{};
  std::array<std::array<ScaledFp8ShapeResult, kShapes.size()>,
             kScaleProfiles.size()>
      results{};

  for (std::size_t profile_index = 0U; profile_index < kScaleProfiles.size();
       ++profile_index) {
    const ScaleProfile profile = kScaleProfiles[profile_index];
    for (std::size_t index = 0U; index < kScaledFp8BoundaryShapes.size();
         ++index) {
      evidence_ok = run_scaled_fp8_shape(
                        kScaledFp8BoundaryShapes[index], profile,
                        &boundaries[profile_index][index], &cleanup_totals) &&
                    evidence_ok;
    }
    for (std::size_t index = 0U; index < kShapes.size(); ++index) {
      evidence_ok = run_scaled_fp8_shape(kShapes[index], profile,
                                         &results[profile_index][index],
                                         &cleanup_totals) &&
                    evidence_ok;
    }
  }

  std::array<bool, kScaledFp8Variants.size()> all_profiles_go{};
  all_profiles_go.fill(true);
  for (std::size_t profile_index = 0U; profile_index < kScaleProfiles.size();
       ++profile_index) {
    const auto profile_go = print_scaled_fp8_summary(
        kScaleProfiles[profile_index], boundaries[profile_index],
        results[profile_index]);
    for (std::size_t index = 1U; index < kScaledFp8Variants.size(); ++index) {
      all_profiles_go[index] = all_profiles_go[index] && profile_go[index];
    }
  }
  const bool any_go = all_profiles_go[1] || all_profiles_go[2];
  const bool cleanup_ok =
      cleanup_totals.ok && cleanup_totals.allocations == cleanup_totals.frees;
  std::cout << "cleanup allocations=" << cleanup_totals.allocations
            << " frees=" << cleanup_totals.frees
            << " status=" << (cleanup_ok ? "PASS" : "FAIL") << "\n";
  evidence_ok = evidence_ok && resources_ok && cleanup_ok;
  std::cout << "PHASE78_NVFP4_GFX1201_SCALED_FP8_EVIDENCE="
            << (evidence_ok ? "PASS" : "FAIL") << "\n";
  std::cout << "PHASE78_NVFP4_GFX1201_SCALED_FP8_DECISION="
            << (evidence_ok && any_go ? "GO" : "N0") << "\n";
  return evidence_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
