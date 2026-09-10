// Phase83.5 ID94 exact gfx1030/gfx1201 NVFP4 small-M body.
//
// The original path keeps ID73's eight-wave/four-column mapping and per-row
// dot4/FMA/reduction order. Each lane decodes one weight block once and reuses
// the decoded words and scale across Rows activation accumulators in VGPRs.
// The gfx1201 M=3 down-projection tuple below uses the same arithmetic with
// decoded activations staged in LDS. On gfx1030, fold the exact quarter
// factor into each activation scale once and reuse it across four columns.
template <uint32_t Rows>
__device__ __forceinline__ void sllm_nvfp4_w4a4_small_m_vgpr_reuse_body(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n,
    const float *const scale_lut) {
  static_assert(Rows >= 2U && Rows <= 4U);
  constexpr uint32_t wave_size = 32U;
  constexpr uint32_t columns_per_wave = 4U;
  constexpr uint32_t waves_per_workgroup = 8U;
  constexpr uint32_t prefetch_chunks = 2U;
  static_assert(wave_size * waves_per_workgroup == 256U);
  if (m != Rows ||
      !((k == UINT64_C(5120) && n == UINT64_C(17408)) ||
        (k == UINT64_C(17408) && n == UINT64_C(5120)))) {
    return;
  }

  const uint32_t lane = threadIdx.x & (wave_size - 1U);
  const uint32_t wave = threadIdx.x / wave_size;
  const uint64_t column_base =
      (static_cast<uint64_t>(blockIdx.x) * waves_per_workgroup + wave) *
      columns_per_wave;
  const uint64_t blocks_per_row = k / UINT64_C(16);
  const uint64_t packed_row_bytes = k / UINT64_C(2);
  float accumulators[Rows][columns_per_wave] = {};

#pragma unroll 1
  for (uint64_t group = lane; group < blocks_per_row;
       group += static_cast<uint64_t>(wave_size) * prefetch_chunks) {
    sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_prefetch2_weight_block prefetched[prefetch_chunks];
#pragma unroll
    for (uint32_t chunk = 0U; chunk < prefetch_chunks; ++chunk) {
      sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_prefetch2_weight_load(
          packed_weight, weight_block_scales, blocks_per_row, column_base, n,
          group + static_cast<uint64_t>(chunk) * wave_size,
          &prefetched[chunk]);
    }

#pragma unroll
    for (uint32_t chunk = 0U; chunk < prefetch_chunks; ++chunk) {
      const uint64_t block =
          group + static_cast<uint64_t>(chunk) * wave_size;
      if (block >= blocks_per_row) {
        continue;
      }

      sllm_id84_nvfp4_scale_lut_detail::ScaledPacks weight_pack0[columns_per_wave];
      sllm_id84_nvfp4_scale_lut_detail::ScaledPacks weight_pack1[columns_per_wave];
      float weight_scales[columns_per_wave];
#pragma unroll
      for (uint32_t local_column = 0U; local_column < columns_per_wave;
           ++local_column) {
        weight_pack0[local_column] = sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_scaled_packs(
            prefetched[chunk].weight_word0[local_column]);
        weight_pack1[local_column] = sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_scaled_packs(
            prefetched[chunk].weight_word1[local_column]);
        weight_scales[local_column] = sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_load_scale(
            prefetched[chunk].weight_scale[local_column], scale_lut);
      }

#pragma unroll
      for (uint32_t row = 0U; row < Rows; ++row) {
        const uint8_t *const row_activation =
            packed_activation + static_cast<uint64_t>(row) * packed_row_bytes;
        const auto *const activation_words =
            reinterpret_cast<const uint32_t *>(
                row_activation + block * UINT64_C(8));
        const sllm_id84_nvfp4_scale_lut_detail::ScaledPacks activation_pack0 =
            sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_scaled_packs(
                __builtin_nontemporal_load(activation_words + 0U));
        const sllm_id84_nvfp4_scale_lut_detail::ScaledPacks activation_pack1 =
            sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_scaled_packs(
                __builtin_nontemporal_load(activation_words + 1U));
        float activation_scale = sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_load_scale(
            __builtin_nontemporal_load(
                activation_block_scales +
                static_cast<uint64_t>(row) * blocks_per_row + block),
            scale_lut);
#if defined(__gfx1030__)
        // E2M1 dot coefficients and E4M3 scales make this regrouping exact.
        // Computing the quarter scale once removes three repeated multiplies.
        activation_scale *= 0.25F;
#endif
#pragma unroll
        for (uint32_t local_column = 0U; local_column < columns_per_wave;
             ++local_column) {
          int32_t block_sum = 0;
          block_sum = sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_dot4(
              activation_pack0.even, weight_pack0[local_column].even,
              block_sum);
          block_sum = sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_dot4(
              activation_pack0.odd, weight_pack0[local_column].odd, block_sum);
          block_sum = sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_dot4(
              activation_pack1.even, weight_pack1[local_column].even,
              block_sum);
          block_sum = sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_dot4(
              activation_pack1.odd, weight_pack1[local_column].odd, block_sum);
          float scaled_dot = static_cast<float>(block_sum);
#if !defined(__gfx1030__)
          scaled_dot *= 0.25F;
#endif
          accumulators[row][local_column] = fmaf(
              scaled_dot * activation_scale, weight_scales[local_column],
              accumulators[row][local_column]);
        }
      }
    }
  }

#pragma unroll
  for (uint32_t offset = wave_size / 2U; offset != 0U; offset >>= 1U) {
#pragma unroll
    for (uint32_t row = 0U; row < Rows; ++row) {
#pragma unroll
      for (uint32_t local_column = 0U; local_column < columns_per_wave;
           ++local_column) {
        accumulators[row][local_column] +=
            __shfl_down(accumulators[row][local_column], offset, wave_size);
      }
    }
  }
  if (lane == 0U) {
#pragma unroll
    for (uint32_t row = 0U; row < Rows; ++row) {
#pragma unroll
      for (uint32_t local_column = 0U; local_column < columns_per_wave;
           ++local_column) {
        const uint64_t column = column_base + local_column;
        if (column < n) {
          output[static_cast<uint64_t>(row) * n + column] =
              sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_bf16_rne(
                  accumulators[row][local_column] *
                  weight_tensor_scale[0] * input_tensor_scale[0]);
        }
      }
    }
  }
}

// The measured gfx1201 M=3 down-projection tuple stages decoded activation
// packs/scales in LDS. Keep this helper inline and feed it the wrapper's LUT;
// all other targets, rows, and tuples stay on the body above.
template <uint32_t Rows>
__device__ __forceinline__ void
sllm_nvfp4_w4a4_small_m_vgpr_reuse_activation_lds_body(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n,
    const float *const scale_lut) {
  static_assert(Rows == 3U);
  constexpr uint32_t threads_per_workgroup = 256U;
  constexpr uint32_t wave_size = 32U;
  constexpr uint32_t columns_per_wave = 4U;
  constexpr uint32_t waves_per_workgroup = 8U;
  constexpr uint32_t stage_blocks = 64U;
  constexpr uint32_t prefetch_chunks = 2U;
  static_assert(wave_size * waves_per_workgroup == threads_per_workgroup);
  if (m != Rows || k != UINT64_C(17408) || n != UINT64_C(5120)) {
    return;
  }

  const uint32_t lane = threadIdx.x & (wave_size - 1U);
  const uint32_t wave = threadIdx.x / wave_size;
  const uint64_t column_base =
      (static_cast<uint64_t>(blockIdx.x) * waves_per_workgroup + wave) *
      columns_per_wave;
  const uint64_t blocks_per_row = k / UINT64_C(16);
  const uint64_t packed_row_bytes = k / UINT64_C(2);

  // Four decoded activation packs and one LUT-decoded scale per staged block.
  __shared__ int32_t activation_packs[Rows][stage_blocks][4];
  __shared__ float activation_scales[Rows][stage_blocks];
  float accumulators[Rows][columns_per_wave] = {};

  const uint64_t stage_count =
      (blocks_per_row + stage_blocks - 1U) / stage_blocks;
  for (uint64_t stage = 0U; stage < stage_count; ++stage) {
    const uint64_t block_base = stage * stage_blocks;

    // Every thread participates in the tile stores. The trailing barrier is
    // the before-overwrite barrier for the next tile.
    for (uint32_t index = threadIdx.x;
         index < Rows * stage_blocks * 4U; index += threads_per_workgroup) {
      const uint32_t row = index / (stage_blocks * 4U);
      const uint32_t remainder = index % (stage_blocks * 4U);
      const uint32_t block = remainder / 4U;
      const uint32_t pack = remainder % 4U;
      const uint64_t source_block = block_base + block;
      uint32_t value = 0U;
      if (source_block < blocks_per_row) {
        const auto *const words = reinterpret_cast<const uint32_t *>(
            packed_activation + static_cast<uint64_t>(row) * packed_row_bytes +
            source_block * UINT64_C(8));
        const uint32_t raw = __builtin_nontemporal_load(words + (pack / 2U));
        const auto decoded =
            sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_scaled_packs(raw);
        value = pack == 0U ? decoded.even
                           : pack == 1U ? decoded.odd
                                        : pack == 2U ? decoded.even
                                                     : decoded.odd;
      }
      activation_packs[row][block][pack] = static_cast<int32_t>(value);
    }
    for (uint32_t index = threadIdx.x; index < Rows * stage_blocks;
         index += threads_per_workgroup) {
      const uint32_t row = index / stage_blocks;
      const uint32_t block = index % stage_blocks;
      const uint64_t source_block = block_base + block;
      activation_scales[row][block] =
          source_block < blocks_per_row
              ? sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_load_scale(
                    __builtin_nontemporal_load(
                        activation_block_scales +
                        static_cast<uint64_t>(row) * blocks_per_row +
                        source_block),
                    scale_lut)
              : 0.0F;
    }
    __syncthreads();

#pragma unroll 1
    for (uint64_t group = lane; group < stage_blocks;
         group += static_cast<uint64_t>(wave_size) * prefetch_chunks) {
      sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_prefetch2_weight_block
          prefetched[prefetch_chunks];
#pragma unroll
      for (uint32_t chunk = 0U; chunk < prefetch_chunks; ++chunk) {
        sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_prefetch2_weight_load(
            packed_weight, weight_block_scales, blocks_per_row, column_base, n,
            block_base + group + static_cast<uint64_t>(chunk) * wave_size,
            &prefetched[chunk]);
      }

#pragma unroll
      for (uint32_t chunk = 0U; chunk < prefetch_chunks; ++chunk) {
        const uint64_t block =
            group + static_cast<uint64_t>(chunk) * wave_size;
        const uint64_t source_block = block_base + block;
        if (source_block >= blocks_per_row) {
          continue;
        }
        sllm_id84_nvfp4_scale_lut_detail::ScaledPacks
            weight_pack0[columns_per_wave];
        sllm_id84_nvfp4_scale_lut_detail::ScaledPacks
            weight_pack1[columns_per_wave];
        float weight_scales[columns_per_wave];
#pragma unroll
        for (uint32_t local_column = 0U;
             local_column < columns_per_wave; ++local_column) {
          weight_pack0[local_column] =
              sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_scaled_packs(
                  prefetched[chunk].weight_word0[local_column]);
          weight_pack1[local_column] =
              sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_scaled_packs(
                  prefetched[chunk].weight_word1[local_column]);
          weight_scales[local_column] =
              sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_load_scale(
                  prefetched[chunk].weight_scale[local_column], scale_lut);
        }
#pragma unroll
        for (uint32_t row = 0U; row < Rows; ++row) {
          const auto activation_pack0 =
              sllm_id84_nvfp4_scale_lut_detail::ScaledPacks{
                  static_cast<uint32_t>(activation_packs[row][block][0]),
                  static_cast<uint32_t>(activation_packs[row][block][1])};
          const auto activation_pack1 =
              sllm_id84_nvfp4_scale_lut_detail::ScaledPacks{
                  static_cast<uint32_t>(activation_packs[row][block][2]),
                  static_cast<uint32_t>(activation_packs[row][block][3])};
          const float activation_scale = activation_scales[row][block];
#pragma unroll
          for (uint32_t local_column = 0U;
               local_column < columns_per_wave; ++local_column) {
            int32_t block_sum = 0;
            block_sum = sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_dot4(
                activation_pack0.even, weight_pack0[local_column].even,
                block_sum);
            block_sum = sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_dot4(
                activation_pack0.odd, weight_pack0[local_column].odd,
                block_sum);
            block_sum = sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_dot4(
                activation_pack1.even, weight_pack1[local_column].even,
                block_sum);
            block_sum = sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_dot4(
                activation_pack1.odd, weight_pack1[local_column].odd,
                block_sum);
            accumulators[row][local_column] = fmaf(
                (static_cast<float>(block_sum) * 0.25F) * activation_scale,
                weight_scales[local_column], accumulators[row][local_column]);
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t offset = wave_size / 2U; offset != 0U; offset >>= 1U) {
#pragma unroll
    for (uint32_t row = 0U; row < Rows; ++row) {
#pragma unroll
      for (uint32_t local_column = 0U; local_column < columns_per_wave;
           ++local_column) {
        accumulators[row][local_column] +=
            __shfl_down(accumulators[row][local_column], offset, wave_size);
      }
    }
  }
  if (lane == 0U) {
#pragma unroll
    for (uint32_t row = 0U; row < Rows; ++row) {
#pragma unroll
      for (uint32_t local_column = 0U; local_column < columns_per_wave;
           ++local_column) {
        const uint64_t column = column_base + local_column;
        if (column < n) {
          output[static_cast<uint64_t>(row) * n + column] =
              sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_bf16_rne(
                  accumulators[row][local_column] * weight_tensor_scale[0] *
                  input_tensor_scale[0]);
        }
      }
    }
  }
}


extern "C" __global__ __launch_bounds__(256, 1)
void sllm_nvfp4_w4a4_small_m_vgpr_reuse_v1(
    const uint8_t *const packed_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
#if defined(__gfx1030__) || defined(__gfx1201__)
  __shared__ float scale_lut[sllm_id84_nvfp4_scale_lut_detail::kScaleLutSlots];
  sllm_id84_nvfp4_scale_lut_detail::sllm_id84_nvfp4_populate_constant_lut(
      scale_lut);
#if defined(__gfx1201__)
  if (m == 3U && k == UINT64_C(17408) && n == UINT64_C(5120)) {
    sllm_nvfp4_w4a4_small_m_vgpr_reuse_activation_lds_body<3U>(
        packed_activation, activation_block_scales, packed_weight,
        weight_block_scales, weight_tensor_scale, input_tensor_scale, output,
        m, k, n, scale_lut);
    return;
  }
#endif
  if (m == 2U) {
    sllm_nvfp4_w4a4_small_m_vgpr_reuse_body<2U>(
        packed_activation, activation_block_scales, packed_weight,
        weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
        k, n, scale_lut);
  } else if (m == 3U) {
    sllm_nvfp4_w4a4_small_m_vgpr_reuse_body<3U>(
        packed_activation, activation_block_scales, packed_weight,
        weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
        k, n, scale_lut);
  } else if (m == 4U) {
    sllm_nvfp4_w4a4_small_m_vgpr_reuse_body<4U>(
        packed_activation, activation_block_scales, packed_weight,
        weight_block_scales, weight_tensor_scale, input_tensor_scale, output, m,
        k, n, scale_lut);
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
