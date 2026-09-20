// Phase 87 WU1.1 scratch candidates.
//
// This fragment is included by phase87_wu1_1_probe.hip.cpp inside the
// existing sllm_causal_attention_kernel anonymous namespace.  It deliberately
// does not touch the production dispatcher.  Both candidates preserve the
// existing [M,24,Splits,258] partial workspace and use the already reviewed
// MXFP8 qtile4 loader.

// C1/GQA-shared stage1 with a configurable split count.  A block owns one KV
// head and its six query heads; the eight-row K/V tile is decoded once for all
// six waves.  The online-softmax and per-wave arithmetic order is the WU1 C1
// order, so only the partition count changes between the WU1.1 candidates.
template <uint32_t kSplits>
__global__ __launch_bounds__(192, 1) void phase87_wu1_1_gqa_stage1(
    const uint16_t *const query, const uint8_t *const key,
    const uint8_t *const value, const uint8_t *const key_scales,
    const uint8_t *const value_scales, float *const workspace,
    const uint32_t query_count, const uint64_t start_position) {
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kWaveCount = 6U;
  constexpr uint32_t kKvHeads = 4U;
  constexpr uint32_t kQHeads = 24U;
  constexpr uint32_t kGqaRatio = 6U;
  constexpr uint32_t kHeadDim = 256U;
  constexpr uint32_t kKeyTile = 8U;
  constexpr uint32_t kWorkspaceStride = kHeadDim + 2U;
  constexpr uint32_t kElementsPerLane = kHeadDim / kWaveSize;
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
  const uint16_t *const query_row =
      query +
      (static_cast<uint64_t>(query_index) * kQHeads + query_head) * kHeadDim;
  const uint64_t workspace_base =
      ((static_cast<uint64_t>(query_index) * kQHeads + query_head) * kSplits +
       split) *
      kWorkspaceStride;
  float *const partial = workspace + workspace_base;

  float query_values[kElementsPerLane];
  float accumulations[kElementsPerLane] = {};
#pragma unroll
  for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
    query_values[index] = bf16_to_f32(query_row[lane + index * kWaveSize]);
  }
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
    for (uint32_t key_index = 0U; key_index < tile_count; ++key_index) {
      float partial_score = 0.0F;
#pragma unroll
      for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
        const uint32_t current = lane + index * kWaveSize;
        partial_score += query_values[index] * key_tile[key_index][current];
      }
      for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
        partial_score += __shfl_down(partial_score, offset, kWaveSize);
      }
      float rescale = 0.0F;
      float contribution = 0.0F;
      if (lane == 0U) {
        const float current_score =
            partial_score * rsqrtf(static_cast<float>(kHeadDim));
        const float next_maximum = fmaxf(local_maximum, current_score);
        rescale = expf(local_maximum - next_maximum);
        contribution = expf(current_score - next_maximum);
        local_denominator = local_denominator * rescale + contribution;
        local_maximum = next_maximum;
      }
      rescale = __shfl(rescale, 0U, kWaveSize);
      contribution = __shfl(contribution, 0U, kWaveSize);
#pragma unroll
      for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
        const uint32_t current = lane + index * kWaveSize;
        accumulations[index] = accumulations[index] * rescale +
                               contribution * value_tile[key_index][current];
      }
    }
    __syncthreads();
  }

  if (lane == 0U) {
    partial[0] = local_maximum;
    partial[1] = local_denominator;
  }
#pragma unroll
  for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
    const uint32_t current = lane + index * kWaveSize;
    partial[2U + current] = accumulations[index];
  }
}

// Conventional staged-wave stage1 with 64/128/256 partitions.  This is the
// same arithmetic as WU1 C3 (query preload, qtile4 decode, and one wave per
// query-head/partition), kept as a single split-count template for the WU1.1
// driver instead of adding three copies of the kernel body.
template <uint32_t kSplits>
__global__ __launch_bounds__(32, 1) void phase87_wu1_1_wave_stage1(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const float *const key_outer_scales, const float *const value_outer_scales,
    float *const workspace, const uint32_t query_count,
    const uint64_t start_position, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim,
    const float static_key_scale, const float static_value_scale) {
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kHeadDim = SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM;
  constexpr uint32_t kElementsPerLane = kHeadDim / kWaveSize;
  const uint64_t flat = static_cast<uint64_t>(blockIdx.x);
  const uint64_t blocks_per_query = static_cast<uint64_t>(q_heads) * kSplits;
  const uint32_t query_index = static_cast<uint32_t>(flat / blocks_per_query);
  const uint32_t query_head =
      static_cast<uint32_t>((flat % blocks_per_query) / kSplits);
  const uint32_t split = static_cast<uint32_t>(flat % kSplits);
  if (query_index >= query_count || query_head >= q_heads ||
      head_dim != kHeadDim || kv_heads == 0U) {
    return;
  }
  const uint32_t lane = threadIdx.x & (kWaveSize - 1U);
  const uint32_t kv_head = query_head / (q_heads / kv_heads);
  const uint16_t *const query_row =
      query +
      (static_cast<uint64_t>(query_index) * q_heads + query_head) * head_dim;
  const uint64_t committed_kv_length = start_position + query_index + 1U;
  float accumulations[kElementsPerLane];
  float query_values[kElementsPerLane];
#if defined(__gfx1030__)
  constexpr bool kUseFusedE4Decode = true;
#else
  constexpr bool kUseFusedE4Decode = false;
#endif
#pragma unroll
  for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
    accumulations[index] = 0.0F;
    const uint32_t current = lane + index * kWaveSize;
    query_values[index] =
        current < head_dim ? bf16_to_f32(query_row[current]) : 0.0F;
  }
  float local_maximum = -std::numeric_limits<float>::infinity();
  float local_denominator = 0.0F;
  const uint64_t split_begin =
      committed_kv_length * static_cast<uint64_t>(split) / kSplits;
  const uint64_t split_end =
      committed_kv_length * static_cast<uint64_t>(split + 1U) / kSplits;
  for (uint64_t key_position = split_begin; key_position < split_end;
       ++key_position) {
    const uint64_t kv_row = key_position * kv_heads + kv_head;
    float partial = 0.0F;
#pragma unroll
    for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
      const uint32_t current = lane + index * kWaveSize;
      if (current < head_dim) {
        partial +=
            query_values[index] *
            load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, kUseFusedE4Decode>(
                key, key_scales, key_outer_scales, kv_row, current, head_dim,
                static_key_scale);
      }
    }
    for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
      partial += __shfl_down(partial, offset, kWaveSize);
    }
    float rescale = 0.0F;
    float contribution = 0.0F;
    if (lane == 0U) {
      const float current_score =
          partial * rsqrtf(static_cast<float>(head_dim));
      const float next_maximum = fmaxf(local_maximum, current_score);
      rescale = expf(local_maximum - next_maximum);
      contribution = expf(current_score - next_maximum);
      local_denominator = local_denominator * rescale + contribution;
      local_maximum = next_maximum;
    }
    rescale = __shfl(rescale, 0U, kWaveSize);
    contribution = __shfl(contribution, 0U, kWaveSize);
#pragma unroll
    for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
      const uint32_t current = lane + index * kWaveSize;
      if (current < head_dim) {
        accumulations[index] =
            accumulations[index] * rescale +
            contribution * load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1,
                                          kUseFusedE4Decode>(
                               value, value_scales, value_outer_scales, kv_row,
                               current, head_dim, static_value_scale);
      }
    }
  }
  const uint64_t stride = static_cast<uint64_t>(kHeadDim) + 2U;
  const uint64_t base =
      ((static_cast<uint64_t>(query_index) * q_heads + query_head) * kSplits +
       split) *
      stride;
  if (lane == 0U) {
    workspace[base] = local_maximum;
    workspace[base + 1U] = local_denominator;
  }
#pragma unroll
  for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
    const uint32_t current = lane + index * kWaveSize;
    if (current < head_dim) {
      workspace[base + 2U + current] = accumulations[index];
    }
  }
}
