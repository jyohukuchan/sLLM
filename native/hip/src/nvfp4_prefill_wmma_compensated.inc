// Phase 83 ID89 candidate: gfx1201 FP8-WMMA K=16 NVFP4 terms with a
// Kahan-compensated outer FP32 fragment accumulation.  The E2M1 -> E4M3FN
// ingress and all WMMA tile geometry are intentionally identical to ID64.
// This include is inserted into matmul_kernel.hip.cpp after the existing ID64
// WMMA body/wrappers, where the codec helpers and rocWMMA headers are visible.
// It remains an opt-in candidate until the runtime selector and GPU evidence
// establish register pressure, numerical behavior, and performance.

template <bool OrdinaryLoads>
__device__ __forceinline__ void
nvfp4_w4a4_gfx1201_wmma128x64_kahan_body(const uint8_t *const packed_activation,
                                   const uint8_t *const activation_block_scales,
                                   const uint8_t *const packed_weight,
                                   const uint8_t *const weight_block_scales,
                                   const float *const weight_tensor_scale,
                                   const float *const input_tensor_scale,
                                   uint16_t *const output, const uint64_t m,
                                   const uint64_t k, const uint64_t n) {
#pragma clang fp contract(off)
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
  float corrections[column_tiles][output_values / wave_width] = {};

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
          // Kahan-compensated outer accumulation keeps the exact FP8-WMMA
          // K=16 term order while reducing the long K reduction error.  The
          // surrounding function is compiled with FP contraction disabled.
          const float adjusted = term - corrections[column_tile][slot];
          const float total = accumulators[column_tile][slot] + adjusted;
          corrections[column_tile][slot] =
              (total - accumulators[column_tile][slot]) - adjusted;
          accumulators[column_tile][slot] = total;
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
            (accumulators[column_tile][slot] * weight_tensor_scale[0]) *
            input_tensor_scale[0]);
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
__launch_bounds__(256, 1) void
sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_kahan_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  nvfp4_w4a4_gfx1201_wmma128x64_kahan_body<false>(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n);
}
