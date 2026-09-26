// Phase 87 WU-P1 paged-prefill probe.
//
// This header is deliberately probe-only.  It is included after
// causal_attention_kernel.hip.cpp, from inside namespace
// sllm_causal_attention_kernel, so the MXFP8 qtile4 loader, BF16 helpers, and
// HIP constants used below are already available.  The kernel is the
// qtile8/w16 Qwen3.8 GQA6 prefill provider with only its KV row addressing
// changed: logical 128-token pages are translated through block_table to
// physical pages.  The block table is read once by lane zero at the start of
// each page-aligned interval and broadcast through shared memory.
//
// Signature of the probe launcher:
//
//   hipError_t launch_phase87_wup1_paged_prefill_gqa6_qtile8_w16(
//       const uint16_t *query, const void *key, const void *value,
//       const void *key_scales, const void *value_scales,
//       const float *key_outer_scales, const float *value_outer_scales,
//       uint16_t *output, uint32_t query_count, uint64_t start_position,
//       uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
//       uint32_t encoding, const uint32_t *block_table,
//       uint32_t block_count, float static_key_scale,
//       float static_value_scale, bool wave_local_kv, hipStream_t stream);
//
// The harness-facing fixed-shape convenience launcher is:
//
//   hipError_t launch_wup1_paged_prefill_qtile8(
//       const uint16_t *query, const void *key, const void *value,
//       const void *key_scales, const void *value_scales,
//       const uint32_t *block_table, uint16_t *output,
//       uint32_t query_count, uint64_t start_position, hipStream_t stream);
//
// For the short-prefill provider, the analogous fixed-shape launcher is:
//
//   hipError_t launch_wup1_paged_prefill_qtile4(
//       const uint16_t *query, const void *key, const void *value,
//       const void *key_scales, const void *value_scales,
//       const uint32_t *block_table, uint16_t *output,
//       uint32_t query_count, uint64_t start_position, hipStream_t stream);
//
// block_table[i] is the physical page number for logical token positions
// [128*i, 128*i+128).  Pages are assumed to contain 128 logical tokens and
// to be laid out with kv_heads rows per token, using the same row-major value
// and E8M0 scale layout as the contiguous provider.  block_count must cover
// every logical page through start_position + query_count - 1.  The probe
// intentionally does not validate physical page numbers against an
// allocation size because that size is not part of the production ABI.
//
// This faithful qtile8/w16 provider has the same representative-shape
// preconditions as production: query_count >= 128, q_heads == 24,
// kv_heads == 4, head_dim == 256, and MXFP8 E4 KV.  The production selector
// additionally uses start_position >= 1024; the launcher accepts any
// non-overflowing start position so a probe can exercise a longer prefix.
// The qtile4 launcher below covers the short GQA6 MXFP8 E4 provider.  GQA4,
// FP16, FP8 formats other than MXFP8 E4, NVFP4, and other attention providers
// remain intentionally unimplemented in this probe.

namespace phase87_wup1_paged_detail {

constexpr uint32_t kPageTokens = 128U;

// This fixed Qwen3.8 geometry has eight E8M0 scale bytes per KV row. The
// existing scalar codecs still perform the value and scale arithmetic; only
// the scale fetch is grouped so the wave-local path avoids eight shuffles.
__device__ __forceinline__ uint64_t prefill_packed_row_scales(
    const void *const scales, const uint64_t row) noexcept {
  return *reinterpret_cast<const uint64_t *>(
      static_cast<const uint8_t *>(scales) + row * 8U);
}

__device__ __forceinline__ float prefill_e4_with_packed_scale(
    const void *const values, const uint64_t row, const uint32_t dimension,
    const uint64_t packed_scales, const uint32_t scale_index) noexcept {
  const uint8_t code =
      static_cast<const uint8_t *>(values)[row * 256U + dimension];
  const uint8_t scale_code =
      static_cast<uint8_t>(packed_scales >> (scale_index * 8U));
  return e4m3fn_to_f32(code) * e8m0_to_f32(scale_code);
}

template <bool WaveLocalKv = false>
__global__
__launch_bounds__(512, 1) void phase87_wup1_paged_prefill_gqa6_qtile8_w16_kernel(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const float *const key_outer_scales, const float *const value_outer_scales,
    uint16_t *const output, const uint32_t query_count,
    const uint64_t start_position, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim,
    const float static_key_scale, const float static_value_scale,
    const uint32_t *const block_table) {
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kWaveCount = 16U;
  constexpr uint32_t kGqaRatio = 6U;
  constexpr uint32_t kQueryTile = 8U;
  constexpr uint32_t kLogicalQueries = kGqaRatio * kQueryTile;
  constexpr uint32_t kQueriesPerWave = kLogicalQueries / kWaveCount;
  constexpr uint32_t kHeadDim = SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM;
  constexpr uint32_t kDimensionsPerLane = kHeadDim / kWaveSize;
  static_assert(kGqaRatio == 6U);
  static_assert(kLogicalQueries % kWaveCount == 0U);
  static_assert(kQueriesPerWave == 3U);

  const uint64_t flat = blockIdx.x;
  const uint64_t tile = flat / kv_heads;
  const uint32_t kv_head = static_cast<uint32_t>(flat % kv_heads);
  const uint64_t first_row = tile * kQueryTile;
  if (first_row >= query_count) {
    return;
  }

  const uint32_t dimension = threadIdx.x;
  const uint32_t lane = dimension & (kWaveSize - 1U);
  const uint32_t wave = dimension / kWaveSize;
  const uint32_t first_query_head = kv_head * kGqaRatio;
  // All three logical items in a wave share one row because
  // kQueriesPerWave=3 and kGqaRatio=6. Preserve the original short-circuit
  // behavior for invalid rows by guarding the causal-limit addition.
  const uint64_t wave_row =
      first_row + (static_cast<uint64_t>(wave) * kQueriesPerWave) / kGqaRatio;
  const bool wave_row_valid = wave_row < query_count;
  const uint64_t wave_causal_limit =
      wave_row_valid ? start_position + wave_row : 0U;

#if defined(__gfx1030__)
  // Scratch qtile-kv4-r14 mapping: four complete K/V rows are staged by the
  // same 512 threads, then consumed in ascending key order.  This is the
  // measured V620 path; gfx1201 retains the original one-row staging below.
  constexpr uint32_t kKeyTile = 4U;
  __shared__ float key_tile[kKeyTile][kHeadDim];
  __shared__ float value_tile[kKeyTile][kHeadDim];
#endif
  // One shared entry avoids a block-table read for every scalar or key.  Only
  // lane zero performs the global read; all threads synchronize before using
  // the physical page number.
  __shared__ uint32_t physical_page;

  float query_values[kQueriesPerWave][kDimensionsPerLane];
  float accumulations[kQueriesPerWave][kDimensionsPerLane] = {};
  // Lane 0/1/2 owns one query each.  Keep the online-softmax state only in
  // that owner lane; value updates and final normalization consume it via
  // wave shuffles.
  float own_running_maximum = -std::numeric_limits<float>::infinity();
  float own_running_denominator = 0.0F;
#pragma unroll
  for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
    const uint32_t logical_query = wave * kQueriesPerWave + item;
    const uint64_t row = first_row + logical_query / kGqaRatio;
    const uint64_t safe_row =
        row < query_count ? row : static_cast<uint64_t>(query_count - 1U);
    const uint32_t query_head = first_query_head + logical_query % kGqaRatio;
    const uint16_t *const query_row =
        query +
        (safe_row * q_heads + query_head) * static_cast<uint64_t>(head_dim);
#pragma unroll
    for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
      const uint32_t current = lane + index * kWaveSize;
      query_values[item][index] = row < query_count && current < head_dim
                                      ? bf16_to_f32(query_row[current])
                                      : 0.0F;
    }
  }

  const uint64_t tile_end = first_row + kQueryTile;
  const uint64_t last_row =
      (tile_end < query_count ? tile_end : query_count) - 1U;
  const uint64_t last_query_position = start_position + last_row;

  // The outer loop is page aligned.  The inner loops are byte-for-byte the
  // contiguous provider's key order and arithmetic; only kv_row uses the
  // physical page selected above.
  for (uint64_t page_begin = 0U; page_begin <= last_query_position;
       page_begin += kPageTokens) {
    if (threadIdx.x == 0U) {
      physical_page =
          block_table[page_begin / static_cast<uint64_t>(kPageTokens)];
    }
    __syncthreads();
    const uint64_t physical_page_base =
        static_cast<uint64_t>(physical_page) * kPageTokens;
    const uint64_t page_end =
        (page_begin > UINT64_MAX - kPageTokens ||
         page_begin + kPageTokens > last_query_position + 1U)
            ? last_query_position + 1U
            : page_begin + kPageTokens;
#if defined(__gfx1030__)
    for (uint64_t key_begin = page_begin; key_begin < page_end;
         key_begin += kKeyTile) {
      const uint64_t remaining = page_end - key_begin;
      const uint32_t key_count =
          remaining < kKeyTile ? static_cast<uint32_t>(remaining) : kKeyTile;
      for (uint32_t element = dimension; element < kKeyTile * 2U * kHeadDim;
           element += 512U) {
        const uint32_t key_index = element / (2U * kHeadDim);
        const uint32_t plane_dimension = element % (2U * kHeadDim);
        if (key_index < key_count) {
          const uint64_t logical_position = key_begin + key_index;
          const uint64_t physical_row =
              physical_page_base + (logical_position - page_begin);
          const uint64_t kv_row = physical_row * kv_heads + kv_head;
          if (plane_dimension < head_dim) {
            key_tile[key_index][plane_dimension] =
                load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, false>(
                    key, key_scales, key_outer_scales, kv_row, plane_dimension,
                    head_dim, static_key_scale);
          } else {
            const uint32_t value_dimension = plane_dimension - head_dim;
            value_tile[key_index][value_dimension] =
                load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, false>(
                    value, value_scales, value_outer_scales, kv_row,
                    value_dimension, head_dim, static_value_scale);
          }
        }
      }
      __syncthreads();

      for (uint32_t key_index = 0U; key_index < key_count; ++key_index) {
        const uint64_t key_position = key_begin + key_index;
        float own_score = 0.0F;
        uint32_t own_active = 0U;
#pragma unroll
        for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
          const bool item_active =
              wave_row_valid && key_position <= wave_causal_limit;
          float products[kDimensionsPerLane];
#pragma unroll
          for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
            const uint32_t current = lane + index * kWaveSize;
            products[index] =
                item_active && current < head_dim
                    ? query_values[item][index] * key_tile[key_index][current]
                    : 0.0F;
          }
          const float pair0 = products[0] + products[1];
          const float pair1 = products[2] + products[3];
          const float pair2 = products[4] + products[5];
          const float pair3 = products[6] + products[7];
          float partial = (pair0 + pair1) + (pair2 + pair3);
          for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
            partial += __shfl_down(partial, offset, kWaveSize);
          }
          const float reduced_score = __shfl(partial, 0U, kWaveSize);
          if (lane == item) {
            own_score = reduced_score;
            own_active = item_active ? 1U : 0U;
          }
        }

        if (lane < kQueriesPerWave && own_active != 0U) {
          own_score *= rsqrtf(static_cast<float>(head_dim));
        }

        float own_rescale = 1.0F;
        float own_contribution = 0.0F;
        if (lane < kQueriesPerWave && own_active != 0U) {
          const float own_next_maximum = fmaxf(own_running_maximum, own_score);
          own_rescale = expf(own_running_maximum - own_next_maximum);
          own_contribution = expf(own_score - own_next_maximum);
          own_running_denominator =
              own_running_denominator * own_rescale + own_contribution;
          own_running_maximum = own_next_maximum;
        }
#pragma unroll
        for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
          const float rescale = __shfl(own_rescale, static_cast<int>(item),
                                       static_cast<int>(kWaveSize));
          const float contribution =
              __shfl(own_contribution, static_cast<int>(item),
                     static_cast<int>(kWaveSize));
          const bool item_active =
              wave_row_valid && key_position <= wave_causal_limit;
#pragma unroll
          for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
            const uint32_t current = lane + index * kWaveSize;
            if (item_active && current < head_dim) {
              accumulations[item][index] =
                  accumulations[item][index] * rescale +
                  contribution * value_tile[key_index][current];
            }
          }
        }
      }
      __syncthreads();
    }
#else
    if constexpr (WaveLocalKv) {
      // Each wave loads its own K/V lane slice, removing per-key LDS staging
      // and barriers. WU-P1 uses this path at its reviewed M=128..219 shapes
      // on gfx1201; it preserves the QK/softmax/value arithmetic and key order.
      for (uint64_t key_position = page_begin; key_position < page_end;
           ++key_position) {
        const uint64_t physical_row =
            physical_page_base + (key_position - page_begin);
        const uint64_t kv_row = physical_row * kv_heads + kv_head;
        float key_register[kDimensionsPerLane];
        const uint64_t packed_key_scales =
            prefill_packed_row_scales(key_scales, kv_row);
#pragma unroll
        for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
          const uint32_t current = lane + index * kWaveSize;
          key_register[index] = prefill_e4_with_packed_scale(
              key, kv_row, current, packed_key_scales, index);
        }

        float own_score = 0.0F;
        uint32_t own_active = 0U;
#pragma unroll
        for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
          const bool item_active =
              wave_row_valid && key_position <= wave_causal_limit;
          float products[kDimensionsPerLane];
#pragma unroll
          for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
            const uint32_t current = lane + index * kWaveSize;
            products[index] =
                item_active && current < head_dim
                    ? query_values[item][index] * key_register[index]
                    : 0.0F;
          }
          const float pair0 = products[0] + products[1];
          const float pair1 = products[2] + products[3];
          const float pair2 = products[4] + products[5];
          const float pair3 = products[6] + products[7];
          float partial = (pair0 + pair1) + (pair2 + pair3);
          for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
            partial += __shfl_down(partial, offset, kWaveSize);
          }
          const float reduced_score = __shfl(partial, 0U, kWaveSize);
          if (lane == item) {
            own_score = reduced_score;
            own_active = item_active ? 1U : 0U;
          }
        }

        if (lane < kQueriesPerWave && own_active != 0U) {
          own_score *= rsqrtf(static_cast<float>(head_dim));
        }

        float own_rescale = 1.0F;
        float own_contribution = 0.0F;
        if (lane < kQueriesPerWave && own_active != 0U) {
          const float own_next_maximum = fmaxf(own_running_maximum, own_score);
          own_rescale = expf(own_running_maximum - own_next_maximum);
          own_contribution = expf(own_score - own_next_maximum);
          own_running_denominator =
              own_running_denominator * own_rescale + own_contribution;
          own_running_maximum = own_next_maximum;
        }
        float value_register[kDimensionsPerLane];
        const uint64_t packed_value_scales =
            prefill_packed_row_scales(value_scales, kv_row);
#pragma unroll
        for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
          const uint32_t current = lane + index * kWaveSize;
          value_register[index] = prefill_e4_with_packed_scale(
              value, kv_row, current, packed_value_scales, index);
        }
#pragma unroll
        for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
          const float rescale = __shfl(own_rescale, static_cast<int>(item),
                                       static_cast<int>(kWaveSize));
          const float contribution =
              __shfl(own_contribution, static_cast<int>(item),
                     static_cast<int>(kWaveSize));
          const bool item_active =
              wave_row_valid && key_position <= wave_causal_limit;
#pragma unroll
          for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
            const uint32_t current = lane + index * kWaveSize;
            if (item_active && current < head_dim) {
              accumulations[item][index] =
                  accumulations[item][index] * rescale +
                  contribution * value_register[index];
            }
          }
        }
      }
    } else {
      __shared__ float key_tile[kHeadDim];
      __shared__ float value_tile[kHeadDim];
      for (uint64_t key_position = page_begin; key_position < page_end;
           ++key_position) {
        const uint64_t physical_row =
            physical_page_base + (key_position - page_begin);
        const uint64_t kv_row = physical_row * kv_heads + kv_head;
        if (dimension < head_dim) {
          key_tile[dimension] =
              load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, false>(
                  key, key_scales, key_outer_scales, kv_row, dimension,
                  head_dim, static_key_scale);
        } else if (dimension < 2U * head_dim) {
          const uint32_t value_dimension = dimension - head_dim;
          value_tile[value_dimension] =
              load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, false>(
                  value, value_scales, value_outer_scales, kv_row,
                  value_dimension, head_dim, static_value_scale);
        }
        __syncthreads();

        float own_score = 0.0F;
        uint32_t own_active = 0U;
#pragma unroll
        for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
          const bool item_active =
              wave_row_valid && key_position <= wave_causal_limit;
          float products[kDimensionsPerLane];
#pragma unroll
          for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
            const uint32_t current = lane + index * kWaveSize;
            products[index] =
                item_active && current < head_dim
                    ? query_values[item][index] * key_tile[current]
                    : 0.0F;
          }
          const float pair0 = products[0] + products[1];
          const float pair1 = products[2] + products[3];
          const float pair2 = products[4] + products[5];
          const float pair3 = products[6] + products[7];
          float partial = (pair0 + pair1) + (pair2 + pair3);
          for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
            partial += __shfl_down(partial, offset, kWaveSize);
          }
          const float reduced_score = __shfl(partial, 0U, kWaveSize);
          if (lane == item) {
            own_score = reduced_score;
            own_active = item_active ? 1U : 0U;
          }
        }

        if (lane < kQueriesPerWave && own_active != 0U) {
          own_score *= rsqrtf(static_cast<float>(head_dim));
        }

        float own_rescale = 1.0F;
        float own_contribution = 0.0F;
        if (lane < kQueriesPerWave && own_active != 0U) {
          const float own_next_maximum = fmaxf(own_running_maximum, own_score);
          own_rescale = expf(own_running_maximum - own_next_maximum);
          own_contribution = expf(own_score - own_next_maximum);
          own_running_denominator =
              own_running_denominator * own_rescale + own_contribution;
          own_running_maximum = own_next_maximum;
        }
#pragma unroll
        for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
          const float rescale = __shfl(own_rescale, static_cast<int>(item),
                                       static_cast<int>(kWaveSize));
          const float contribution =
              __shfl(own_contribution, static_cast<int>(item),
                     static_cast<int>(kWaveSize));
          const bool item_active =
              wave_row_valid && key_position <= wave_causal_limit;
#pragma unroll
          for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
            const uint32_t current = lane + index * kWaveSize;
            if (item_active && current < head_dim) {
              accumulations[item][index] =
                  accumulations[item][index] * rescale +
                  contribution * value_tile[current];
            }
          }
        }
        __syncthreads();
      }
    }
#endif
  }

#pragma unroll
  for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
    const uint32_t logical_query = wave * kQueriesPerWave + item;
    const uint64_t row = first_row + logical_query / kGqaRatio;
    const uint32_t query_head = first_query_head + logical_query % kGqaRatio;
    if (row < query_count) {
      uint16_t *const output_row = output + (row * q_heads + query_head) *
                                                static_cast<uint64_t>(head_dim);
      const float denominator =
          __shfl(own_running_denominator, static_cast<int>(item),
                 static_cast<int>(kWaveSize));
#pragma unroll
      for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
        const uint32_t current = lane + index * kWaveSize;
        if (current < head_dim) {
          output_row[current] =
              f32_to_bf16_rne(accumulations[item][index] / denominator);
        }
      }
    }
  }
}

// The short-prefill counterpart used by the production selector for
// qtile4/GQA6.  It is kept here because WU-P1 probes the 1023/1024/1025
// context boundaries, where qtile8/w16 is not the selected provider.  The
// only functional difference from the contiguous provider is the
// page-table-based physical row calculation.
__global__
__launch_bounds__(256, 1) void phase87_wup1_paged_prefill_gqa6_qtile4_kernel(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const float *const key_outer_scales, const float *const value_outer_scales,
    uint16_t *const output, const uint32_t query_count,
    const uint64_t start_position, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim,
    const float static_key_scale, const float static_value_scale,
    const uint32_t *const block_table) {
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kWaveCount = 8U;
  constexpr uint32_t kGqaRatio = 6U;
  constexpr uint32_t kQueryTile = 4U;
  constexpr uint32_t kLogicalQueries = kGqaRatio * kQueryTile;
  constexpr uint32_t kQueriesPerWave = kLogicalQueries / kWaveCount;
  constexpr uint32_t kHeadDim = SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM;
  constexpr uint32_t kDimensionsPerLane = kHeadDim / kWaveSize;
  static_assert(kGqaRatio == 6U);
  static_assert(kLogicalQueries % kWaveCount == 0U);
  static_assert(kQueriesPerWave == 3U);

  const uint64_t flat = blockIdx.x;
  const uint64_t tile = flat / kv_heads;
  const uint32_t kv_head = static_cast<uint32_t>(flat % kv_heads);
  const uint64_t first_row = tile * kQueryTile;
  if (first_row >= query_count) {
    return;
  }
  const uint32_t dimension = threadIdx.x;
  const uint32_t lane = dimension & (kWaveSize - 1U);
  const uint32_t wave = dimension / kWaveSize;
  const uint32_t first_query_head = kv_head * kGqaRatio;
  __shared__ float key_tile[kHeadDim];
  __shared__ float value_tile[kHeadDim];
  __shared__ uint32_t physical_page;
  float query_values[kQueriesPerWave][kDimensionsPerLane];
  float accumulations[kQueriesPerWave][kDimensionsPerLane] = {};
  float running_maximum[kQueriesPerWave];
  float running_denominator[kQueriesPerWave];
#pragma unroll
  for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
    running_maximum[item] = -std::numeric_limits<float>::infinity();
    running_denominator[item] = 0.0F;
    const uint32_t logical_query = wave * kQueriesPerWave + item;
    const uint64_t row = first_row + logical_query / kGqaRatio;
    const uint64_t safe_row =
        row < query_count ? row : static_cast<uint64_t>(query_count - 1U);
    const uint32_t query_head = first_query_head + logical_query % kGqaRatio;
    const uint16_t *const query_row =
        query +
        (safe_row * q_heads + query_head) * static_cast<uint64_t>(head_dim);
#pragma unroll
    for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
      const uint32_t current = lane + index * kWaveSize;
      query_values[item][index] = row < query_count && current < head_dim
                                      ? bf16_to_f32(query_row[current])
                                      : 0.0F;
    }
  }

  const uint64_t tile_end = first_row + kQueryTile;
  const uint64_t last_row =
      (tile_end < query_count ? tile_end : query_count) - 1U;
  const uint64_t last_query_position = start_position + last_row;
  for (uint64_t page_begin = 0U; page_begin <= last_query_position;
       page_begin += kPageTokens) {
    if (threadIdx.x == 0U) {
      physical_page =
          block_table[page_begin / static_cast<uint64_t>(kPageTokens)];
    }
    __syncthreads();
    const uint64_t page_end =
        (page_begin > UINT64_MAX - kPageTokens ||
         page_begin + kPageTokens > last_query_position + 1U)
            ? last_query_position + 1U
            : page_begin + kPageTokens;
    for (uint64_t key_position = page_begin; key_position < page_end;
         ++key_position) {
      const uint64_t physical_row =
          static_cast<uint64_t>(physical_page) * kPageTokens +
          (key_position - page_begin);
      const uint64_t kv_row = physical_row * kv_heads + kv_head;
      if (dimension < head_dim) {
        key_tile[dimension] =
            load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, false>(
                key, key_scales, key_outer_scales, kv_row, dimension, head_dim,
                static_key_scale);
        value_tile[dimension] =
            load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, false>(
                value, value_scales, value_outer_scales, kv_row, dimension,
                head_dim, static_value_scale);
      }
      __syncthreads();

#pragma unroll
      for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
        const uint32_t logical_query = wave * kQueriesPerWave + item;
        const uint64_t row = first_row + logical_query / kGqaRatio;
        const bool active =
            row < query_count && key_position <= start_position + row;
        float products[kDimensionsPerLane];
#pragma unroll
        for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
          const uint32_t current = lane + index * kWaveSize;
          products[index] = active && current < head_dim
                                ? query_values[item][index] * key_tile[current]
                                : 0.0F;
        }
        const float pair0 = products[0] + products[1];
        const float pair1 = products[2] + products[3];
        const float pair2 = products[4] + products[5];
        const float pair3 = products[6] + products[7];
        float partial = (pair0 + pair1) + (pair2 + pair3);
        for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
          partial += __shfl_down(partial, offset, kWaveSize);
        }
        float rescale = 1.0F;
        float contribution = 0.0F;
        float next_maximum = running_maximum[item];
        if (lane == 0U) {
          if (active) {
            const float current_score =
                partial * rsqrtf(static_cast<float>(head_dim));
            next_maximum = fmaxf(running_maximum[item], current_score);
            rescale = expf(running_maximum[item] - next_maximum);
            contribution = expf(current_score - next_maximum);
          }
        }
        rescale = __shfl(rescale, 0U, kWaveSize);
        contribution = __shfl(contribution, 0U, kWaveSize);
        next_maximum = __shfl(next_maximum, 0U, kWaveSize);
        running_denominator[item] =
            running_denominator[item] * rescale + contribution;
        running_maximum[item] = next_maximum;
#pragma unroll
        for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
          const uint32_t current = lane + index * kWaveSize;
          if (active && current < head_dim) {
            accumulations[item][index] = accumulations[item][index] * rescale +
                                         contribution * value_tile[current];
          }
        }
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
    const uint32_t logical_query = wave * kQueriesPerWave + item;
    const uint64_t row = first_row + logical_query / kGqaRatio;
    const uint32_t query_head = first_query_head + logical_query % kGqaRatio;
    if (row < query_count) {
      uint16_t *const output_row = output + (row * q_heads + query_head) *
                                                static_cast<uint64_t>(head_dim);
#pragma unroll
      for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
        const uint32_t current = lane + index * kWaveSize;
        if (current < head_dim) {
          output_row[current] = f32_to_bf16_rne(accumulations[item][index] /
                                                running_denominator[item]);
        }
      }
    }
  }
}

} // namespace phase87_wup1_paged_detail

inline hipError_t launch_phase87_wup1_paged_prefill_gqa6_qtile8_w16(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const float *const key_outer_scales, const float *const value_outer_scales,
    uint16_t *const output, const uint32_t query_count,
    const uint64_t start_position, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim, const uint32_t encoding,
    const uint32_t *const block_table, const uint32_t block_count,
    const float static_key_scale, const float static_value_scale,
    const bool wave_local_kv, const hipStream_t stream) noexcept {
  if (query == nullptr || key == nullptr || value == nullptr ||
      key_scales == nullptr || value_scales == nullptr || output == nullptr ||
      block_table == nullptr || query_count < 128U ||
      query_count > SLLM_HIP_CAUSAL_ATTENTION_MAX_M || q_heads != 24U ||
      kv_heads != 4U || head_dim != SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM ||
      encoding != SLLM_HIP_KV_ENCODING_MXFP8_E4_V1 ||
      start_position > UINT64_MAX - static_cast<uint64_t>(query_count)) {
    return hipErrorInvalidValue;
  }
  const uint64_t last_query_position =
      start_position + static_cast<uint64_t>(query_count) - 1U;
  const uint64_t required_pages =
      last_query_position / phase87_wup1_paged_detail::kPageTokens + 1U;
  if (required_pages > block_count || required_pages > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  const uint64_t gqa_block_count =
      (static_cast<uint64_t>(query_count) + 7U) / 8U * kv_heads;
  if (gqa_block_count > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidValue;
  }
  if (wave_local_kv) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            phase87_wup1_paged_detail::
                phase87_wup1_paged_prefill_gqa6_qtile8_w16_kernel<true>),
        dim3(static_cast<uint32_t>(gqa_block_count)), dim3(512U), 0U, stream,
        query, key, value, key_scales, value_scales, key_outer_scales,
        value_outer_scales, output, query_count, start_position, q_heads,
        kv_heads, head_dim, static_key_scale, static_value_scale, block_table);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            phase87_wup1_paged_detail::
                phase87_wup1_paged_prefill_gqa6_qtile8_w16_kernel<false>),
        dim3(static_cast<uint32_t>(gqa_block_count)), dim3(512U), 0U, stream,
        query, key, value, key_scales, value_scales, key_outer_scales,
        value_outer_scales, output, query_count, start_position, q_heads,
        kv_heads, head_dim, static_key_scale, static_value_scale, block_table);
  }
  return hipGetLastError();
}

// Fixed Qwen3.8 harness entry point.  The block count is derived from the
// requested logical range because this intentionally small probe API does not
// carry an allocation size.  The caller remains responsible for providing at
// least that many valid physical-page entries. gfx1030 uses the staged qtile8
// implementation; gfx1201 uses its wave-local form for the reviewed shapes.
inline hipError_t launch_wup1_paged_prefill_qtile8(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const uint32_t *const block_table, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const hipStream_t stream) noexcept {
  if (query_count == 0U ||
      start_position > UINT64_MAX - static_cast<uint64_t>(query_count)) {
    return hipErrorInvalidValue;
  }
  const uint64_t last_query_position =
      start_position + static_cast<uint64_t>(query_count) - 1U;
  const uint64_t required_pages =
      last_query_position / phase87_wup1_paged_detail::kPageTokens + 1U;
  if (required_pages > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  return launch_phase87_wup1_paged_prefill_gqa6_qtile8_w16(
      query, key, value, key_scales, value_scales, nullptr, nullptr, output,
      query_count, start_position, 24U, 4U, SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM,
      SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, block_table,
      static_cast<uint32_t>(required_pages), 1.0F, 1.0F, true, stream);
}

// Fixed short-prefill entry point used for the qtile4 boundary cases.  It
// follows the production GQA6 qtile4 admission rule (at least 64 query rows)
// while the WU-P1 harness normally supplies M=128.
inline hipError_t launch_wup1_paged_prefill_qtile4(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const uint32_t *const block_table, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const hipStream_t stream) noexcept {
  if (query == nullptr || key == nullptr || value == nullptr ||
      key_scales == nullptr || value_scales == nullptr ||
      block_table == nullptr || output == nullptr || query_count < 64U ||
      query_count > SLLM_HIP_CAUSAL_ATTENTION_MAX_M ||
      start_position > UINT64_MAX - static_cast<uint64_t>(query_count)) {
    return hipErrorInvalidValue;
  }
  const uint64_t last_query_position =
      start_position + static_cast<uint64_t>(query_count) - 1U;
  const uint64_t required_pages =
      last_query_position / phase87_wup1_paged_detail::kPageTokens + 1U;
  if (required_pages > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  const uint64_t gqa_block_count =
      (static_cast<uint64_t>(query_count) + 3U) / 4U * 4U;
  if (gqa_block_count > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(phase87_wup1_paged_detail::
                          phase87_wup1_paged_prefill_gqa6_qtile4_kernel),
      dim3(static_cast<uint32_t>(gqa_block_count)), dim3(256U), 0U, stream,
      query, key, value, key_scales, value_scales, nullptr, nullptr, output,
      query_count, start_position, 24U, 4U, SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM,
      1.0F, 1.0F, block_table);
  return hipGetLastError();
}
