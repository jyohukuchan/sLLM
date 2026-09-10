// Phase 83 ID89 candidate: gfx1201 FP8-WMMA K=16 NVFP4 terms with a
// Kahan-compensated outer FP32 fragment accumulation.  The E2M1 -> E4M3FN
// ingress and all WMMA tile geometry are intentionally identical to ID64.
// This include is inserted into matmul_kernel.hip.cpp after the existing ID64
// WMMA body/wrappers, where the codec helpers and rocWMMA headers are visible.
// The scoped runtime policy selects ID89 for the exact tuples at M >= 64. The
// original ID89 symbol keeps its logical name and N64 launch contract and
// remains StageK32. Host dispatch selects the measured lookahead symbol for
// the exact StageK64 tuples at M >= 256; the StageK64 symbol remains available
// as the control entry point.

template <bool OrdinaryLoads, uint32_t StageK>
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
  static_assert(StageK == 32U || StageK == 64U,
                "NVFP4 WMMA Kahan supports only StageK32 or StageK64");
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t waves_per_workgroup = 8U;
  constexpr uint32_t tile_m = 16U;
  constexpr uint32_t tile_n = 16U;
  constexpr uint32_t column_tiles = 4U;
  constexpr uint32_t fragment_k = 16U;
  constexpr uint32_t stage_k = StageK;
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
    for (uint32_t scale_index = thread;
         scale_index < waves_per_workgroup * tile_m * scale_blocks_per_stage;
         scale_index += blockDim.x) {
      const uint32_t scale_block = scale_index % scale_blocks_per_stage;
      const uint32_t row_index = scale_index / scale_blocks_per_stage;
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
    for (uint32_t scale_index = thread;
         scale_index < column_tiles * tile_n * scale_blocks_per_stage;
         scale_index += blockDim.x) {
      const uint32_t scale_block = scale_index % scale_blocks_per_stage;
      const uint32_t local_column = scale_index / scale_blocks_per_stage;
      const uint64_t column = column_base + local_column;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      weight_scale_tile[local_column][scale_block] =
          column < n && block < blocks_per_row
              ? e4m3fn_to_float(
                    weight_block_scales[column * blocks_per_row + block])
              : 0.0F;
    }
    __syncthreads();

    if constexpr (StageK == 64U) {
#pragma unroll 1
      for (uint32_t scale_block = 0U; scale_block < scale_blocks_per_stage;
           ++scale_block) {
        AFragment activation_fragment;
        rocwmma::load_matrix_sync(
            activation_fragment,
            activation_tile[wave] + scale_block * fragment_k, stage_k);
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
    } else {
#pragma unroll 2
      for (uint32_t scale_block = 0U; scale_block < scale_blocks_per_stage;
           ++scale_block) {
        AFragment activation_fragment;
        rocwmma::load_matrix_sync(
            activation_fragment,
            activation_tile[wave] + scale_block * fragment_k, stage_k);
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
  nvfp4_w4a4_gfx1201_wmma128x64_kahan_body<false, 32U>(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void
sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_kahan_stage64_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  nvfp4_w4a4_gfx1201_wmma128x64_kahan_body<false, 64U>(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n);
}

// Phase 83.5 measured candidate: gfx1201 FP8-WMMA K=16 NVFP4 terms with a
// Kahan-compensated outer FP32 fragment accumulation.  The E2M1 -> E4M3FN
// ingress and all WMMA tile geometry are intentionally identical to ID64.
// This include is inserted into matmul_kernel.hip.cpp after the existing ID64
// WMMA body/wrappers, where the codec helpers and rocWMMA headers are visible.
// Phase 83.5 ID89 measured StageK64 lookahead. The existing StageK32 and
// StageK64 wrappers above remain the numerical controls; this wrapper uses the
// bounded gfx1201 parity-LDS/raw-VGPR body validated on the exact target.

template <bool OrdinaryLoads, uint64_t StaticK = 0U, uint64_t StaticN = 0U,
          bool PaddedOperandRows = false>
__device__ __forceinline__ void
nvfp4_w4a4_gfx1201_wmma128x64_kahan_lookahead_body(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
#pragma clang fp contract(off)
#if defined(__gfx1201__)
  constexpr bool static_interior = StaticK != 0U && StaticN != 0U;
  static_assert(!static_interior ||
                    ((StaticK % 64U) == 0U && (StaticN % 64U) == 0U),
                "ID89 aligned specialization requires K/N StageK64 tiles");
  static_assert(!static_interior ||
                    ((StaticK == 5120U && StaticN == 17408U) ||
                     (StaticK == 17408U && StaticN == 5120U)),
                "ID89 aligned specialization requires a measured tuple");
  static_assert(!PaddedOperandRows || static_interior,
                "ID89 padded operands require StaticK/StaticN specialization");
  if constexpr (static_interior) {
    (void)m;
    (void)k;
    (void)n;
  }
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t waves_per_workgroup = 8U;
  constexpr uint32_t tile_m = 16U;
  constexpr uint32_t tile_n = 16U;
  constexpr uint32_t column_tiles = 4U;
  constexpr uint32_t fragment_k = 16U;
  // Measured candidate: stage four K=16 terms per synchronization tile.
  // The per-term Kahan update and term order are unchanged.
  constexpr uint32_t stage_k = 64U;
  constexpr uint32_t scale_blocks_per_stage = stage_k / fragment_k;
  constexpr uint32_t rows_per_workgroup = waves_per_workgroup * tile_m;
  constexpr uint32_t columns_per_workgroup = column_tiles * tile_n;
  constexpr uint32_t tile_values = tile_m * stage_k;
  constexpr uint32_t values_per_group = 4U;
  constexpr uint32_t groups_per_tile = tile_values / values_per_group;
  constexpr uint32_t operand_row_bytes = PaddedOperandRows ? 68U : stage_k;
  constexpr uint32_t operand_row_words = operand_row_bytes / values_per_group;
  constexpr uint32_t output_values = tile_m * tile_n;
  constexpr uint32_t stage_buffers = 2U;

  __shared__ __align__(4)
      rocwmma::float8_t
          activation_tile[stage_buffers][waves_per_workgroup]
                        [tile_m * operand_row_bytes];
  __shared__ __align__(4)
      rocwmma::float8_t
          weight_tile[stage_buffers][column_tiles][tile_m * operand_row_bytes];
  __shared__ float activation_scale_tile[stage_buffers][waves_per_workgroup]
                                        [tile_m][scale_blocks_per_stage];
  __shared__ float weight_scale_tile[stage_buffers][column_tiles * tile_n]
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
  const uint64_t blocks_per_row =
      static_interior ? StaticK / fragment_k : k / fragment_k;
  const uint64_t stages = static_interior
                              ? StaticK / stage_k
                              : (blocks_per_row + scale_blocks_per_stage - 1U) /
                                    scale_blocks_per_stage;
  const uint64_t packed_row_bytes = static_interior ? StaticK / 2U : k / 2U;
  float accumulators[column_tiles][output_values / wave_width] = {};
  float corrections[column_tiles][output_values / wave_width] = {};
  // Twelve packed E2M1 groups plus three raw E4M3 scale bytes are held while
  // the current stage's remaining K=16 WMMA terms execute. They are decoded
  // only after compute and written to the alternate LDS parity.
  uint16_t next_activation_packed[8] = {};
  uint16_t next_weight_packed[4] = {};
  uint8_t next_activation_scale[2] = {};
  uint8_t next_weight_scale[1] = {};

  for (uint64_t stage = 0U; stage < stages; ++stage) {
    const uint64_t inner_base = stage * stage_k;
    const uint32_t buffer = static_cast<uint32_t>(stage & 1U);
    const uint32_t next_buffer = buffer ^ 1U;
    auto *const activation_groups =
        reinterpret_cast<uint32_t *>(activation_tile[buffer]);
    auto *const weight_groups =
        reinterpret_cast<uint32_t *>(weight_tile[buffer]);

    if (stage == 0U) {
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
      if (static_interior ||
          (row < m && inner_base + local_group * values_per_group < k)) {
        const auto *const source = reinterpret_cast<const uint16_t *>(
            packed_activation + row * packed_row_bytes + inner_base / 2U +
            local_group * 2U);
        if constexpr (OrdinaryLoads) {
          packed = *source;
        } else {
          packed = __builtin_nontemporal_load(source);
        }
      }
      const uint32_t row_word =
          source_wave * tile_m * operand_row_words +
          local_row * operand_row_words + local_group;
      activation_groups[row_word] = e2m1x4_to_e4m3fn_exact_bits(packed);
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
      if (static_interior ||
          (column < n && inner_base + local_group * values_per_group < k)) {
        const auto *const source = reinterpret_cast<const uint16_t *>(
            packed_weight + column * packed_row_bytes + inner_base / 2U +
            local_group * 2U);
        if constexpr (OrdinaryLoads) {
          packed = *source;
        } else {
          packed = __builtin_nontemporal_load(source);
        }
      }
      const uint32_t row_word =
          column_tile * tile_m * operand_row_words +
          local_column * operand_row_words + local_group;
      weight_groups[row_word] = e2m1x4_to_e4m3fn_exact_bits(packed);
    }
    for (uint32_t scale_index = thread;
         scale_index < waves_per_workgroup * tile_m * scale_blocks_per_stage;
         scale_index += blockDim.x) {
      const uint32_t scale_block = scale_index % scale_blocks_per_stage;
      const uint32_t row_index = scale_index / scale_blocks_per_stage;
      const uint32_t source_wave = row_index / tile_m;
      const uint32_t local_row = row_index - source_wave * tile_m;
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      activation_scale_tile[buffer][source_wave][local_row][scale_block] =
          static_interior
              ? e4m3fn_to_float(
                    activation_block_scales[row * blocks_per_row + block])
              : (row < m && block < blocks_per_row
                     ? e4m3fn_to_float(
                           activation_block_scales[row * blocks_per_row +
                                                   block])
                     : 0.0F);
    }
    for (uint32_t scale_index = thread;
         scale_index < column_tiles * tile_n * scale_blocks_per_stage;
         scale_index += blockDim.x) {
      const uint32_t scale_block = scale_index % scale_blocks_per_stage;
      const uint32_t local_column = scale_index / scale_blocks_per_stage;
      const uint64_t column = column_base + local_column;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      weight_scale_tile[buffer][local_column][scale_block] =
          static_interior
              ? e4m3fn_to_float(
                    weight_block_scales[column * blocks_per_row + block])
              : (column < n && block < blocks_per_row
                     ? e4m3fn_to_float(
                           weight_block_scales[column * blocks_per_row +
                                                block])
                     : 0.0F);
    }
      __syncthreads();
    }

#pragma unroll 1
    for (uint32_t scale_block = 0U; scale_block < scale_blocks_per_stage;
         ++scale_block) {
      AFragment activation_fragment;
      rocwmma::load_matrix_sync(
          activation_fragment,
          activation_tile[buffer][wave] + scale_block * fragment_k,
          operand_row_bytes);
#pragma unroll
      for (uint32_t column_tile = 0U; column_tile < column_tiles;
           ++column_tile) {
        BFragment weight_fragment;
        AccumulatorFragment contribution;
        rocwmma::fill_fragment(contribution, 0.0F);
        rocwmma::load_matrix_sync(
            weight_fragment,
            weight_tile[buffer][column_tile] + scale_block * fragment_k,
            operand_row_bytes);
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
                       activation_scale_tile[buffer][wave][local_row]
                                            [scale_block];
          term *= weight_scale_tile[buffer][column_tile * tile_n + local_column]
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

      if (scale_block == 0U && stage + 1U < stages) {
        const uint64_t next_inner_base = (stage + 1U) * stage_k;
#pragma unroll
        for (uint32_t local = 0U; local < 8U; ++local) {
          const uint32_t group = thread + local * blockDim.x;
          const uint32_t source_wave = group / groups_per_tile;
          const uint32_t wave_group = group - source_wave * groups_per_tile;
          const uint32_t local_row = wave_group / (stage_k / values_per_group);
          const uint32_t local_group =
              wave_group - local_row * (stage_k / values_per_group);
          const uint64_t row =
              row_group_base + static_cast<uint64_t>(source_wave) * tile_m +
              local_row;
          uint16_t packed = 0U;
          if (static_interior ||
              (row < m &&
               next_inner_base + local_group * values_per_group < k)) {
            const auto *const source = reinterpret_cast<const uint16_t *>(
                packed_activation + row * packed_row_bytes +
                next_inner_base / 2U + local_group * 2U);
            if constexpr (OrdinaryLoads) {
              packed = *source;
            } else {
              packed = __builtin_nontemporal_load(source);
            }
          }
          next_activation_packed[local] = packed;
        }
#pragma unroll
        for (uint32_t local = 0U; local < 4U; ++local) {
          const uint32_t group = thread + local * blockDim.x;
          const uint32_t column_tile = group / groups_per_tile;
          const uint32_t tile_group = group - column_tile * groups_per_tile;
          const uint32_t local_column =
              tile_group / (stage_k / values_per_group);
          const uint32_t local_group =
              tile_group - local_column * (stage_k / values_per_group);
          const uint64_t column =
              column_base + column_tile * tile_n + local_column;
          uint16_t packed = 0U;
          if (static_interior ||
              (column < n &&
               next_inner_base + local_group * values_per_group < k)) {
            const auto *const source = reinterpret_cast<const uint16_t *>(
                packed_weight + column * packed_row_bytes +
                next_inner_base / 2U + local_group * 2U);
            if constexpr (OrdinaryLoads) {
              packed = *source;
            } else {
              packed = __builtin_nontemporal_load(source);
            }
          }
          next_weight_packed[local] = packed;
        }
        for (uint32_t local = 0U; local < 2U; ++local) {
          const uint32_t scale_index = thread + local * blockDim.x;
          const uint32_t next_scale_block =
              scale_index % scale_blocks_per_stage;
          const uint32_t row_index = scale_index / scale_blocks_per_stage;
          const uint32_t source_wave = row_index / tile_m;
          const uint32_t local_row = row_index - source_wave * tile_m;
          const uint64_t row = row_group_base +
                               static_cast<uint64_t>(source_wave) * tile_m +
                               local_row;
          const uint64_t block =
              (stage + 1U) * scale_blocks_per_stage + next_scale_block;
          next_activation_scale[local] =
              static_interior
                  ? activation_block_scales[row * blocks_per_row + block]
                  : (row < m && block < blocks_per_row
                         ? activation_block_scales[row * blocks_per_row + block]
                         : 0U);
        }
        const uint32_t scale_index = thread;
        const uint32_t next_scale_block =
            scale_index % scale_blocks_per_stage;
        const uint32_t local_column = scale_index / scale_blocks_per_stage;
        const uint64_t column = column_base + local_column;
        const uint64_t block =
            (stage + 1U) * scale_blocks_per_stage + next_scale_block;
        next_weight_scale[0] =
            static_interior
                ? weight_block_scales[column * blocks_per_row + block]
                : (column < n && block < blocks_per_row
                       ? weight_block_scales[column * blocks_per_row + block]
                       : 0U);
      }
    }

    if (stage + 1U < stages) {
      auto *const next_activation_groups =
          reinterpret_cast<uint32_t *>(activation_tile[next_buffer]);
      auto *const next_weight_groups =
          reinterpret_cast<uint32_t *>(weight_tile[next_buffer]);
#pragma unroll
      for (uint32_t local = 0U; local < 8U; ++local) {
        const uint32_t group = thread + local * blockDim.x;
        const uint32_t source_wave = group / groups_per_tile;
        const uint32_t wave_group = group - source_wave * groups_per_tile;
        const uint32_t local_row =
            wave_group / (stage_k / values_per_group);
        const uint32_t local_group =
            wave_group - local_row * (stage_k / values_per_group);
        const uint32_t row_word =
            (source_wave * tile_m + local_row) * operand_row_words +
            local_group;
        next_activation_groups[row_word] =
            e2m1x4_to_e4m3fn_exact_bits(next_activation_packed[local]);
      }
#pragma unroll
      for (uint32_t local = 0U; local < 4U; ++local) {
        const uint32_t group = thread + local * blockDim.x;
        const uint32_t column_tile = group / groups_per_tile;
        const uint32_t tile_group = group - column_tile * groups_per_tile;
        const uint32_t local_column =
            tile_group / (stage_k / values_per_group);
        const uint32_t local_group =
            tile_group - local_column * (stage_k / values_per_group);
        const uint32_t row_word =
            (column_tile * tile_m + local_column) * operand_row_words +
            local_group;
        next_weight_groups[row_word] =
            e2m1x4_to_e4m3fn_exact_bits(next_weight_packed[local]);
      }
      for (uint32_t local = 0U; local < 2U; ++local) {
        const uint32_t scale_index = thread + local * blockDim.x;
        const uint32_t scale_block = scale_index % scale_blocks_per_stage;
        const uint32_t row_index = scale_index / scale_blocks_per_stage;
        const uint32_t source_wave = row_index / tile_m;
        const uint32_t local_row = row_index - source_wave * tile_m;
        activation_scale_tile[next_buffer][source_wave][local_row]
                             [scale_block] =
            e4m3fn_to_float(next_activation_scale[local]);
      }
      const uint32_t scale_index = thread;
      const uint32_t scale_block = scale_index % scale_blocks_per_stage;
      const uint32_t local_column = scale_index / scale_blocks_per_stage;
      weight_scale_tile[next_buffer][local_column][scale_block] =
          e4m3fn_to_float(next_weight_scale[0]);
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
      if (static_interior || (row < m && column < n)) {
        const uint64_t output_stride = static_interior ? StaticN : n;
        output[row * output_stride + column] = float_to_bf16_rne_bits(
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
sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_kahan_lookahead_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  nvfp4_w4a4_gfx1201_wmma128x64_kahan_lookahead_body<false, 0U, 0U>(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void
sllm_nvfp4_gfx1201_wmma128x64_aligned_k5120n17408_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  nvfp4_w4a4_gfx1201_wmma128x64_kahan_lookahead_body<false, 5120U,
                                                       17408U>(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void
sllm_nvfp4_gfx1201_wmma128x64_aligned_k17408n5120_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  nvfp4_w4a4_gfx1201_wmma128x64_kahan_lookahead_body<false, 17408U,
                                                       5120U>(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void
sllm_nvfp4_gfx1201_wmma128x64_pad68_k5120n17408_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  nvfp4_w4a4_gfx1201_wmma128x64_kahan_lookahead_body<false, 5120U, 17408U,
                                                     true>(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void
sllm_nvfp4_gfx1201_wmma128x64_pad68_k17408n5120_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
  nvfp4_w4a4_gfx1201_wmma128x64_kahan_lookahead_body<false, 17408U, 5120U,
                                                     true>(
      packed_activation, activation_block_scales, packed_weight,
      weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
      k, n);
}
