// Phase 87 WU-D3 candidate hooks.
//
// The D3 probe intentionally starts with only the production control path.
// Candidate bodies are selected after WU-D2 establishes the persistence
// window.  Keeping the enum and dispatch contract here lets the probe report
// the exact requested candidate while returning a fail-closed status until a
// bounded candidate is added.

enum class Phase87Wud3Candidate : unsigned {
  Control = 0U,
  C1Tile32 = 1U,
  C1Tile64 = 2U,
  C2BlockRemap = 3U,
  C3Prefetch = 4U,
};

inline const char *
phase87_wud3_candidate_name(const Phase87Wud3Candidate candidate) noexcept {
  switch (candidate) {
  case Phase87Wud3Candidate::Control:
    return "control";
  case Phase87Wud3Candidate::C1Tile32:
    return "c1-tile32";
  case Phase87Wud3Candidate::C1Tile64:
    return "c1-tile64";
  case Phase87Wud3Candidate::C2BlockRemap:
    return "c2-block-remap";
  case Phase87Wud3Candidate::C3Prefetch:
    return "c3-prefetch";
  }
  return "unknown";
}

// Candidate implementations are deliberately disabled until WU-D2 results
// select the persistence-relevant variant.  The probe must not silently
// measure control under a candidate label.
inline bool phase87_wud3_candidate_implemented(
    const Phase87Wud3Candidate candidate) noexcept {
  return candidate != Phase87Wud3Candidate::C1Tile64;
}

// A tile32 candidate would consume exactly 64 KiB of LDS for decoded K/V.
// Tile64 is rejected at probe preflight so an eventual implementation cannot
// accidentally launch a 128 KiB-LDS kernel on a target with a smaller limit.
inline bool phase87_wud3_candidate_preflight_safe(
    const Phase87Wud3Candidate candidate) noexcept {
  return candidate != Phase87Wud3Candidate::C1Tile64;
}

namespace sllm_causal_attention_kernel {
namespace phase87_wud3_detail {

constexpr uint32_t kWaveSize = 32U;
constexpr uint32_t kWaveCount = 6U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kQHeads = 24U;
constexpr uint32_t kGqaRatio = 6U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kWorkspaceStride = kHeadDim + 2U;
constexpr uint32_t kElementsPerLane = kHeadDim / kWaveSize;

// C1, C2 and C3 share this body.  The accumulator and online-softmax loops
// are deliberately kept byte-for-byte in the same order as the production
// GQA stage1.  Only the tile capacity, block mapping, or when the next tile
// is filled changes.
template <uint32_t kSplits, uint32_t kKeyTile, bool kRemapBlocks,
          bool kLookahead>
__global__ __launch_bounds__(192, 1) void stage1(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    float *const workspace, const uint32_t query_count,
    const uint64_t start_position) {
#if defined(__gfx1030__)
  constexpr bool kUseFusedE4Decode = true;
#else
  constexpr bool kUseFusedE4Decode = false;
#endif
  const uint32_t block = static_cast<uint32_t>(blockIdx.x);
  uint32_t split = 0U;
  uint32_t kv_head = 0U;
  uint32_t query_index = 0U;
  if constexpr (kRemapBlocks) {
    // Production order is (query, kv_head, split).  This candidate uses
    // (query, split, kv_head), leaving logical workspace indexing unchanged.
    split = (block / kKvHeads) % kSplits;
    kv_head = block % kKvHeads;
    query_index = block / (kKvHeads * kSplits);
  } else {
    split = block % kSplits;
    kv_head = (block / kSplits) % kKvHeads;
    query_index = block / (kKvHeads * kSplits);
  }
  const uint32_t wave = threadIdx.x / kWaveSize;
  const uint32_t lane = threadIdx.x & (kWaveSize - 1U);
  const uint32_t query_head = kv_head * kGqaRatio + wave;
  if (query_index >= query_count || query_head >= kQHeads || wave >= kWaveCount)
    return;

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
  for (uint32_t index = 0U; index < kElementsPerLane; ++index)
    query_values[index] = bf16_to_f32(query_row[lane + index * kWaveSize]);
  if (split_begin >= split_end) {
    for (uint32_t index = lane; index < kHeadDim; index += kWaveSize)
      partial[2U + index] = 0.0F;
    if (lane == 0U) {
      partial[0] = -std::numeric_limits<float>::infinity();
      partial[1] = 0.0F;
    }
    return;
  }

  // C1 uses tile32 (64 KiB decoded K/V LDS).  C2 retains tile8 and only
  // changes block order.  C3 retains tile8 and fills the alternate buffer
  // before computing the current tile, which changes request timing without
  // changing key order or the softmax recurrence.
  __shared__ float key_tile[kLookahead ? 2U : 1U][kKeyTile][kHeadDim];
  __shared__ float value_tile[kLookahead ? 2U : 1U][kKeyTile][kHeadDim];
  float local_maximum = -std::numeric_limits<float>::infinity();
  float local_denominator = 0.0F;

  const auto fill_tile = [&](const uint64_t tile_begin, const uint32_t buffer) {
    const uint64_t remaining = split_end - tile_begin;
    const uint32_t tile_count =
        remaining < kKeyTile ? static_cast<uint32_t>(remaining) : kKeyTile;
    for (uint32_t element = threadIdx.x; element < kKeyTile * 2U * kHeadDim;
         element += blockDim.x) {
      const uint32_t key_index = element / (2U * kHeadDim);
      const uint32_t plane_dimension = element % (2U * kHeadDim);
      if (key_index >= tile_count)
        continue;
      const uint64_t kv_row =
          (tile_begin + static_cast<uint64_t>(key_index)) * kKvHeads + kv_head;
      if (plane_dimension < kHeadDim) {
        key_tile[buffer][key_index][plane_dimension] =
            load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, kUseFusedE4Decode>(
                key, key_scales, nullptr, kv_row, plane_dimension, kHeadDim,
                1.0F);
      } else {
        const uint32_t dimension = plane_dimension - kHeadDim;
        value_tile[buffer][key_index][dimension] =
            load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, kUseFusedE4Decode>(
                value, value_scales, nullptr, kv_row, dimension, kHeadDim,
                1.0F);
      }
    }
  };

  uint32_t current_buffer = 0U;
  fill_tile(split_begin, current_buffer);
  __syncthreads();
  for (uint64_t tile_begin = split_begin; tile_begin < split_end;
       tile_begin += kKeyTile) {
    const uint64_t next_begin = tile_begin + kKeyTile;
    if constexpr (kLookahead) {
      if (next_begin < split_end)
        fill_tile(next_begin, current_buffer ^ 1U);
    } else if (tile_begin != split_begin) {
      fill_tile(tile_begin, current_buffer);
    }
    __syncthreads();
    const uint64_t remaining = split_end - tile_begin;
    const uint32_t tile_count =
        remaining < kKeyTile ? static_cast<uint32_t>(remaining) : kKeyTile;
    for (uint32_t key_index = 0U; key_index < tile_count; ++key_index) {
      float partial_score = 0.0F;
#pragma unroll
      for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
        const uint32_t current = lane + index * kWaveSize;
        partial_score +=
            query_values[index] * key_tile[current_buffer][key_index][current];
      }
      for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U)
        partial_score += __shfl_down(partial_score, offset, kWaveSize);
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
        accumulations[index] =
            accumulations[index] * rescale +
            contribution * value_tile[current_buffer][key_index][current];
      }
    }
    __syncthreads();
    if constexpr (kLookahead)
      current_buffer ^= 1U;
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

} // namespace phase87_wud3_detail
} // namespace sllm_causal_attention_kernel
