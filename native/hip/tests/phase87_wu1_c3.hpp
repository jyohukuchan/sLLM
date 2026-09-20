// C3: project ID93 arithmetic with configurable splits; retain fused E4 decode
// on gfx1030.
template <bool UseQueryPreload, uint32_t Encoding, uint32_t kWaveCount>
__global__ __launch_bounds__(32, 1) void phase87_wu1_c3_stage1(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const float *const key_outer_scales, const float *const value_outer_scales,
    float *const workspace, const uint32_t query_count,
    const uint64_t start_position, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim,
    const float static_key_scale, const float static_value_scale) {
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kHeadDim = SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM;
  constexpr uint32_t kDimensionsPerLane = kHeadDim / kWaveSize;
  const uint64_t flat = static_cast<uint64_t>(blockIdx.x);
  const uint64_t blocks_per_query = static_cast<uint64_t>(q_heads) * kWaveCount;
  const uint32_t query_index = static_cast<uint32_t>(flat / blocks_per_query);
  const uint32_t query_head =
      static_cast<uint32_t>((flat % blocks_per_query) / kWaveCount);
  const uint32_t wave = static_cast<uint32_t>(flat % kWaveCount);
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
  float accumulations[kDimensionsPerLane];
  float query_values[kDimensionsPerLane];
#if defined(__gfx1030__)
  constexpr bool kUseFusedE4Decode =
      Encoding == SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;
#else
  constexpr bool kUseFusedE4Decode = false;
#endif
#pragma unroll
  for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
    accumulations[index] = 0.0F;
    if constexpr (UseQueryPreload) {
      const uint32_t current = lane + index * kWaveSize;
      query_values[index] =
          current < head_dim ? bf16_to_f32(query_row[current]) : 0.0F;
    }
  }
  float local_maximum = -std::numeric_limits<float>::infinity();
  float local_denominator = 0.0F;
  const uint64_t split_begin = committed_kv_length * wave / kWaveCount;
  const uint64_t split_end =
      committed_kv_length * (static_cast<uint64_t>(wave) + 1U) / kWaveCount;
  for (uint64_t key_position = split_begin; key_position < split_end;
       ++key_position) {
    const uint64_t kv_row = key_position * kv_heads + kv_head;
    float partial = 0.0F;
#pragma unroll
    for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
      const uint32_t current = lane + index * kWaveSize;
      if (current < head_dim) {
        if constexpr (UseQueryPreload) {
          partial += query_values[index] *
                     load_kv_qtile4<Encoding, kUseFusedE4Decode>(
                         key, key_scales, key_outer_scales, kv_row, current,
                         head_dim, static_key_scale);
        } else {
          partial += bf16_to_f32(query_row[current]) *
                     load_kv_qtile4<Encoding, kUseFusedE4Decode>(
                         key, key_scales, key_outer_scales, kv_row, current,
                         head_dim, static_key_scale);
        }
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
    for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
      const uint32_t current = lane + index * kWaveSize;
      if (current < head_dim) {
        accumulations[index] =
            accumulations[index] * rescale +
            contribution * load_kv_qtile4<Encoding, kUseFusedE4Decode>(
                               value, value_scales, value_outer_scales, kv_row,
                               current, head_dim, static_value_scale);
      }
    }
  }
  const uint64_t stride = static_cast<uint64_t>(kHeadDim) + 2U;
  const uint64_t base =
      ((static_cast<uint64_t>(query_index) * q_heads + query_head) *
           kWaveCount +
       wave) *
      stride;
  if (lane == 0U) {
    workspace[base] = local_maximum;
    workspace[base + 1U] = local_denominator;
  }
#pragma unroll
  for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
    const uint32_t current = lane + index * kWaveSize;
    if (current < head_dim) {
      workspace[base + 2U + current] = accumulations[index];
    }
  }
}
