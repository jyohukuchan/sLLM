// Phase 87 WU1 C2 scratch kernel.
//
// This header is included from causal_attention_kernel.hip.cpp after the
// staged stage-2 kernel, while still inside its existing anonymous namespace.
// It intentionally adds no includes or namespace: the parent translation unit
// supplies the BF16/MXFP8 helpers and the low-precision encoding constants.
//
// C2 groups four adjacent keys per wave.  The six waves in a block are the
// six query heads belonging to one KV head, so an eight-row decoded tile is
// loaded once and shared by all six GQA queries.  The grouped dot product has
// four independent eight-term accumulators before the width-eight reduction.
// That changes the control accumulation order (N1), but keeps the longest
// serial FP32 sum at eight products plus the balanced combine/reduction.

__global__ __launch_bounds__(192, 1) void phase87_wu1_c2_stage1(
    const uint16_t *const query, const uint8_t *const key,
    const uint8_t *const value, const uint8_t *const key_scales,
    const uint8_t *const value_scales, float *const workspace,
    const uint32_t query_count, const uint64_t start_position) {
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kWaveCount = 6U;
  constexpr uint32_t kKvHeads = 4U;
  constexpr uint32_t kQHeads = 24U;
  constexpr uint32_t kGqaRatio = 6U;
  constexpr uint32_t kSplits = 32U;
  constexpr uint32_t kHeadDim = 256U;
  constexpr uint32_t kKeyTile = 8U;
  constexpr uint32_t kKeysPerGroup = 4U;
  constexpr uint32_t kLanesPerKey = 8U;
  constexpr uint32_t kDimensionsPerLane = kHeadDim / kLanesPerKey;
  constexpr uint32_t kOutputDimensionsPerLane = kHeadDim / kWaveSize;
  constexpr uint32_t kWorkspaceStride = kHeadDim + 2U;
#if defined(__gfx1030__)
  constexpr bool kUseFusedE4Decode = true;
#else
  constexpr bool kUseFusedE4Decode = false;
#endif

  const uint32_t block = static_cast<uint32_t>(blockIdx.x);
  const uint32_t split = block % kSplits;
  const uint32_t kv_head = (block / kSplits) % kKvHeads;
  const uint32_t query_index = block / (kKvHeads * kSplits);
  const uint32_t wave = threadIdx.x / kWaveSize;
  const uint32_t lane = threadIdx.x & (kWaveSize - 1U);
  const uint32_t key_group = lane / kLanesPerKey;
  const uint32_t dimension_lane = lane % kLanesPerKey;
  const uint32_t query_head = kv_head * kGqaRatio + wave;

  if (query_index >= query_count || query_head >= kQHeads ||
      wave >= kWaveCount) {
    return;
  }

  const uint64_t committed_kv_length =
      start_position + static_cast<uint64_t>(query_index) + 1U;
  const uint64_t split_begin =
      committed_kv_length * static_cast<uint64_t>(split) / kSplits;
  const uint64_t split_end =
      committed_kv_length * static_cast<uint64_t>(split + 1U) / kSplits;
  const uint64_t workspace_base =
      ((static_cast<uint64_t>(query_index) * kQHeads + query_head) * kSplits +
       split) *
      kWorkspaceStride;
  float *const partial = workspace + workspace_base;

  // Empty split intervals are common at context lengths below the split
  // count.  Publish a neutral partial and return before entering the shared
  // tile loop; every thread in this block takes the same branch.
  if (split_begin >= split_end) {
    for (uint32_t index = lane; index < kHeadDim; index += kWaveSize) {
      partial[2U + index] = 0.0F;
    }
    if (lane == 0U) {
      partial[0] = -std::numeric_limits<float>::infinity();
      partial[1] = 0.0F;
    }
    return;
  }

  const uint16_t *const query_row =
      query +
      (static_cast<uint64_t>(query_index) * kQHeads + query_head) * kHeadDim;

  // The query is retained in the dot-product layout: each lane owns 32
  // dimensions (dimension_lane + 8*i), split into four independent groups of
  // eight products.  Output accumulation below uses the normal eight
  // dimensions per lane (lane + 32*i).
  float query_values[kDimensionsPerLane];
#pragma unroll
  for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
    query_values[index] =
        bf16_to_f32(query_row[dimension_lane + kLanesPerKey * index]);
  }
  float accumulations[kOutputDimensionsPerLane] = {};

  // Eight decoded K/V rows are 8 * 256 * 2 * sizeof(float) = 16 KiB.  All
  // six waves participate in the load so each encoded row is decoded once per
  // KV head, then consumed by the six GQA query heads.
  __shared__ float key_tile[kKeyTile][kHeadDim];
  __shared__ float value_tile[kKeyTile][kHeadDim];
  float local_maximum = -std::numeric_limits<float>::infinity();
  float local_denominator = 0.0F;

  for (uint64_t tile_begin = split_begin; tile_begin < split_end;
       tile_begin += kKeyTile) {
    const uint64_t remaining = split_end - tile_begin;
    const uint32_t tile_count =
        remaining < kKeyTile ? static_cast<uint32_t>(remaining) : kKeyTile;
    for (uint32_t element = threadIdx.x; element < kKeyTile * 2U * kHeadDim;
         element += blockDim.x) {
      const uint32_t key_index = element / (2U * kHeadDim);
      const uint32_t plane_dimension = element % (2U * kHeadDim);
      if (key_index < tile_count) {
        const uint64_t kv_row =
            (tile_begin + static_cast<uint64_t>(key_index)) * kKvHeads +
            kv_head;
        if (plane_dimension < kHeadDim) {
          key_tile[key_index][plane_dimension] =
              load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1,
                             kUseFusedE4Decode>(key, key_scales, nullptr,
                                                kv_row, plane_dimension,
                                                kHeadDim, 1.0F);
        } else {
          const uint32_t dimension = plane_dimension - kHeadDim;
          value_tile[key_index][dimension] =
              load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1,
                             kUseFusedE4Decode>(value, value_scales, nullptr,
                                                kv_row, dimension, kHeadDim,
                                                1.0F);
        }
      }
    }
    __syncthreads();

    // Process four keys at once.  The key group owns one key and its eight
    // lanes cover all 256 dot dimensions.  The second four-key group reuses
    // the same shared tile after the online-softmax update.
    for (uint32_t key_offset = 0U; key_offset < tile_count;
         key_offset += kKeysPerGroup) {
      const uint32_t active_keys = (tile_count - key_offset) < kKeysPerGroup
                                       ? tile_count - key_offset
                                       : kKeysPerGroup;
      const bool key_active = key_group < active_keys;
      const uint32_t key_index =
          key_active ? key_offset + key_group : key_offset;

      float dot_accumulations[4] = {};
#pragma unroll
      for (uint32_t group = 0U; group < 4U; ++group) {
#pragma unroll
        for (uint32_t term = 0U; term < 8U; ++term) {
          const uint32_t query_index_in_lane = group * 8U + term;
          const uint32_t dimension =
              dimension_lane + kLanesPerKey * query_index_in_lane;
          dot_accumulations[group] += query_values[query_index_in_lane] *
                                      key_tile[key_index][dimension];
        }
      }
      const float pair0 = dot_accumulations[0] + dot_accumulations[1];
      const float pair1 = dot_accumulations[2] + dot_accumulations[3];
      float dot_partial = pair0 + pair1;
      for (uint32_t offset = kLanesPerKey / 2U; offset != 0U; offset >>= 1U) {
        dot_partial += __shfl_down(dot_partial, offset, kLanesPerKey);
      }
      // The first lane of each eight-lane key group now owns the complete
      // score. Broadcast it within the group before collecting four scores.
      float own_score = __shfl(dot_partial, 0, kLanesPerKey);
      own_score = key_active ? own_score * rsqrtf(static_cast<float>(kHeadDim))
                             : -std::numeric_limits<float>::infinity();

      const float score0 = __shfl(own_score, 0, kWaveSize);
      const float score1 = __shfl(own_score, 8, kWaveSize);
      const float score2 = __shfl(own_score, 16, kWaveSize);
      const float score3 = __shfl(own_score, 24, kWaveSize);
      const float tile_maximum =
          fmaxf(fmaxf(score0, score1), fmaxf(score2, score3));
      const float next_maximum = fmaxf(local_maximum, tile_maximum);

      // One owner lane per key evaluates its probability; the value is
      // broadcast over the eight lanes that own that key.  Lane zero handles
      // the prior-state rescale, avoiding six copies of that exponential.
      float rescale = 0.0F;
      if (lane == 0U) {
        rescale = expf(local_maximum - next_maximum);
      }
      rescale = __shfl(rescale, 0, kWaveSize);
      float own_probability = 0.0F;
      if (dimension_lane == 0U && key_active) {
        own_probability = expf(own_score - next_maximum);
      }
      const float probability0 = __shfl(own_probability, 0, kWaveSize);
      const float probability1 = __shfl(own_probability, 8, kWaveSize);
      const float probability2 = __shfl(own_probability, 16, kWaveSize);
      const float probability3 = __shfl(own_probability, 24, kWaveSize);
      const float probability_sum =
          (probability0 + probability1) + (probability2 + probability3);
      // The same balanced pair tree is used for all output dimensions.  This
      // keeps the grouped update's added arithmetic depth bounded while
      // preserving the online-softmax recurrence.
#pragma unroll
      for (uint32_t index = 0U; index < kOutputDimensionsPerLane; ++index) {
        const uint32_t dimension = lane + kWaveSize * index;
        const float value0 =
            key_offset + 0U < tile_count
                ? probability0 * value_tile[key_offset + 0U][dimension]
                : 0.0F;
        const float value1 =
            key_offset + 1U < tile_count
                ? probability1 * value_tile[key_offset + 1U][dimension]
                : 0.0F;
        const float value2 =
            key_offset + 2U < tile_count
                ? probability2 * value_tile[key_offset + 2U][dimension]
                : 0.0F;
        const float value3 =
            key_offset + 3U < tile_count
                ? probability3 * value_tile[key_offset + 3U][dimension]
                : 0.0F;
        const float value_pair0 = value0 + value1;
        const float value_pair1 = value2 + value3;
        const float grouped_value = value_pair0 + value_pair1;
        accumulations[index] = accumulations[index] * rescale + grouped_value;
      }
      local_denominator = local_denominator * rescale + probability_sum;
      local_maximum = next_maximum;
    }
    // Do not overwrite the shared tile until every GQA wave has consumed it.
    __syncthreads();
  }

  if (lane == 0U) {
    partial[0] = local_maximum;
    partial[1] = local_denominator;
  }
#pragma unroll
  for (uint32_t index = 0U; index < kOutputDimensionsPerLane; ++index) {
    const uint32_t dimension = lane + kWaveSize * index;
    partial[2U + dimension] = accumulations[index];
  }
}
