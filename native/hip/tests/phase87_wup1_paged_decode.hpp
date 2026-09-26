// Phase 87 WU-P1 probe-only paged decode stage 1.
//
// This file is intentionally a test header rather than a production kernel.
// The WU-P1 harness includes causal_attention_kernel.hip.cpp first, while it
// is still in sllm_causal_attention_kernel's anonymous namespace.  The
// production BF16, MXFP8 and online-softmax helpers are therefore available
// here.  The two kernels below keep the production stage-1 arithmetic and
// workspace order, changing only the KV row lookup:
//
//   logical token -> 128-token logical page -> physical page in block_table
//
// A page is stored token-major, with all four KV heads in each token.  The
// key/value byte pools and their E8M0 scale pools use the same physical page
// numbering.  A block-table entry is read once for each page-aligned portion
// of a split; tokens inside that portion are then visited in ascending order
// using the existing eight-token tile order.
//
// The exported launcher is probe-only.  It launches stage 1 and writes the
// same [M,24,S,258] FP32 workspace consumed by the existing stage-2 merge
// kernels.  It does not alter a public ABI or the default runtime path.

constexpr uint32_t kPhase87Wup1PagedTokensPerPage = 128U;
constexpr uint32_t kPhase87Wup1PagedKvHeads = 4U;
constexpr uint32_t kPhase87Wup1PagedQHeads = 24U;
constexpr uint32_t kPhase87Wup1PagedHeadDim = 256U;
constexpr uint32_t kPhase87Wup1PagedScaleBlocks =
    kPhase87Wup1PagedHeadDim / 32U;
constexpr uint32_t kPhase87Wup1PagedWorkspaceStride =
    kPhase87Wup1PagedHeadDim + 2U;
constexpr uint64_t kPhase87Wup1PagedValueBytesPerPage =
    static_cast<uint64_t>(kPhase87Wup1PagedTokensPerPage) *
    kPhase87Wup1PagedKvHeads * kPhase87Wup1PagedHeadDim;
constexpr uint64_t kPhase87Wup1PagedScaleBytesPerPage =
    static_cast<uint64_t>(kPhase87Wup1PagedTokensPerPage) *
    kPhase87Wup1PagedKvHeads * kPhase87Wup1PagedScaleBlocks;

namespace phase87_wup1_paged_detail {

__device__ __forceinline__ uint64_t
page_end_token(const uint64_t logical_page) noexcept {
  return (logical_page + 1U) * kPhase87Wup1PagedTokensPerPage;
}

// The returned row is local to the physical page.  Keeping the physical page
// base separate from the row makes the block-table load visibly one-per-page
// and avoids accidentally treating a non-identity table as contiguous.
__device__ __forceinline__ uint64_t
page_local_row(const uint64_t logical_token, const uint32_t kv_head) noexcept {
  return (logical_token % kPhase87Wup1PagedTokensPerPage) *
             kPhase87Wup1PagedKvHeads +
         kv_head;
}

__device__ __forceinline__ const uint8_t *
paged_values_page(const void *const values,
                  const uint32_t physical_page) noexcept {
  return static_cast<const uint8_t *>(values) +
         static_cast<uint64_t>(physical_page) *
             kPhase87Wup1PagedValueBytesPerPage;
}

__device__ __forceinline__ const uint8_t *
paged_scales_page(const void *const scales,
                  const uint32_t physical_page) noexcept {
  return static_cast<const uint8_t *>(scales) +
         static_cast<uint64_t>(physical_page) *
             kPhase87Wup1PagedScaleBytesPerPage;
}

template <bool UseQueryPreload, uint32_t kSplits>
__global__ __launch_bounds__(192, 1) void gqa6_stage1(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const uint32_t *const block_table, float *const workspace,
    const uint32_t query_count, const uint64_t start_position) {
  static_assert(UseQueryPreload, "GQA-shared probe preloads query values");
  static_assert(kSplits == 32U || kSplits == 128U);
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kWaveCount = 6U;
  constexpr uint32_t kGqaRatio = 6U;
  constexpr uint32_t kKeyTile = 8U;
  constexpr uint32_t kElementsPerLane = kPhase87Wup1PagedHeadDim / kWaveSize;
#if defined(__gfx1030__)
  constexpr bool kUseFusedE4Decode = true;
#else
  constexpr bool kUseFusedE4Decode = false;
#endif

  const uint32_t block = static_cast<uint32_t>(blockIdx.x);
  const uint32_t split = block % kSplits;
  const uint32_t kv_head = (block / kSplits) % kPhase87Wup1PagedKvHeads;
  const uint32_t query_index = block / (kPhase87Wup1PagedKvHeads * kSplits);
  const uint32_t wave = threadIdx.x / kWaveSize;
  const uint32_t lane = threadIdx.x & (kWaveSize - 1U);
  const uint32_t query_head = kv_head * kGqaRatio + wave;

  if (query_index >= query_count || query_head >= kPhase87Wup1PagedQHeads ||
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
      query + (static_cast<uint64_t>(query_index) * kPhase87Wup1PagedQHeads +
               query_head) *
                  kPhase87Wup1PagedHeadDim;

  float query_values[kElementsPerLane];
  float accumulations[kElementsPerLane] = {};
#pragma unroll
  for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
    query_values[index] = bf16_to_f32(query_row[lane + index * kWaveSize]);
  }

  const uint64_t workspace_base =
      ((static_cast<uint64_t>(query_index) * kPhase87Wup1PagedQHeads +
        query_head) *
           kSplits +
       split) *
      kPhase87Wup1PagedWorkspaceStride;
  float *const partial = workspace + workspace_base;
  if (split_begin >= split_end) {
    for (uint32_t index = lane; index < kPhase87Wup1PagedHeadDim;
         index += kWaveSize) {
      partial[2U + index] = 0.0F;
    }
    if (lane == 0U) {
      partial[0] = -std::numeric_limits<float>::infinity();
      partial[1] = 0.0F;
    }
    return;
  }

  __shared__ float key_tile[kKeyTile][kPhase87Wup1PagedHeadDim];
  __shared__ float value_tile[kKeyTile][kPhase87Wup1PagedHeadDim];
  float local_maximum = -std::numeric_limits<float>::infinity();
  float local_denominator = 0.0F;

  const uint64_t first_page = split_begin / kPhase87Wup1PagedTokensPerPage;
  const uint64_t page_count =
      split_end == 0U ? 0U
                      : (split_end - 1U) / kPhase87Wup1PagedTokensPerPage + 1U;
  for (uint64_t logical_page = first_page; logical_page < page_count;
       ++logical_page) {
    // This is deliberately the only block-table access in this page segment.
    const uint32_t physical_page = block_table[logical_page];
    const uint8_t *const key_page = paged_values_page(key, physical_page);
    const uint8_t *const value_page = paged_values_page(value, physical_page);
    const uint8_t *const key_scale_page =
        paged_scales_page(key_scales, physical_page);
    const uint8_t *const value_scale_page =
        paged_scales_page(value_scales, physical_page);
    const uint64_t page_begin = logical_page * kPhase87Wup1PagedTokensPerPage;
    const uint64_t segment_begin =
        split_begin > page_begin ? split_begin : page_begin;
    const uint64_t page_end = page_end_token(logical_page);
    const uint64_t segment_end = split_end < page_end ? split_end : page_end;

    for (uint64_t tile_begin = segment_begin; tile_begin < segment_end;
         tile_begin += kKeyTile) {
      const uint64_t remaining = segment_end - tile_begin;
      const uint32_t tile_count =
          remaining < kKeyTile ? static_cast<uint32_t>(remaining) : kKeyTile;
      for (uint32_t element = threadIdx.x;
           element < kKeyTile * 2U * kPhase87Wup1PagedHeadDim;
           element += blockDim.x) {
        const uint32_t key_index = element / (2U * kPhase87Wup1PagedHeadDim);
        const uint32_t plane_dimension =
            element % (2U * kPhase87Wup1PagedHeadDim);
        if (key_index < tile_count) {
          const uint64_t kv_row = page_local_row(
              tile_begin + static_cast<uint64_t>(key_index), kv_head);
          if (plane_dimension < kPhase87Wup1PagedHeadDim) {
            key_tile[key_index][plane_dimension] =
                load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1,
                               kUseFusedE4Decode>(
                    key_page, key_scale_page, nullptr, kv_row, plane_dimension,
                    kPhase87Wup1PagedHeadDim, 1.0F);
          } else {
            const uint32_t dimension =
                plane_dimension - kPhase87Wup1PagedHeadDim;
            value_tile[key_index][dimension] =
                load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1,
                               kUseFusedE4Decode>(
                    value_page, value_scale_page, nullptr, kv_row, dimension,
                    kPhase87Wup1PagedHeadDim, 1.0F);
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
              partial_score *
              rsqrtf(static_cast<float>(kPhase87Wup1PagedHeadDim));
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

__device__ __forceinline__ uint32_t load_wup1_wave_uniform_page(
    const uint32_t *const table, const uint64_t logical_page) noexcept {
#if defined(__gfx1201__)
  const auto pointer = reinterpret_cast<uintptr_t>(table + logical_page);
  const uint32_t low =
      __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(pointer));
  const uint32_t high =
      __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(pointer >> 32U));
  const uint64_t uniform = (static_cast<uint64_t>(high) << 32U) | low;
  uint32_t page = 0U;
  asm volatile("s_load_dword %0, %1, 0\n\ts_waitcnt lgkmcnt(0)"
               : "=s"(page)
               : "s"(uniform)
               : "memory");
  return page;
#else
  return table[logical_page];
#endif
}

__device__ __forceinline__ uint64_t
load_wup1_packed_row_scales(const void *const scales, const uint64_t row,
                            const uint32_t lane) noexcept {
  (void)lane;
  return *reinterpret_cast<const uint64_t *>(
      static_cast<const uint8_t *>(scales) +
      row * kPhase87Wup1PagedScaleBlocks);
}

__device__ __forceinline__ float load_wup1_e4_with_packed_scale(
    const void *const values, const uint64_t row, const uint32_t dimension,
    const uint64_t packed_scales, const uint32_t scale_index) noexcept {
  const uint8_t code = static_cast<const uint8_t *>(
      values)[row * kPhase87Wup1PagedHeadDim + dimension];
  const uint8_t scale_code =
      static_cast<uint8_t>(packed_scales >> (scale_index * 8U));
  return e4m3fn_to_f32(code) * e8m0_to_f32(scale_code);
}

template <uint32_t kSplits, bool UseQueryPreload = true>
__global__ __launch_bounds__(32, 1) void wave_split_stage1(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const uint32_t *const block_table, float *const workspace,
    const uint32_t query_count, const uint64_t start_position,
    const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim) {
  static_assert(kSplits == 32U || kSplits == 128U);
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kElementsPerLane = kPhase87Wup1PagedHeadDim / kWaveSize;

  const uint64_t flat = static_cast<uint64_t>(blockIdx.x);
  const uint64_t blocks_per_query = static_cast<uint64_t>(q_heads) * kSplits;
  const uint32_t query_index = static_cast<uint32_t>(flat / blocks_per_query);
  const uint32_t query_head =
      static_cast<uint32_t>((flat % blocks_per_query) / kSplits);
  const uint32_t wave = static_cast<uint32_t>(flat % kSplits);
  if (query_index >= query_count || query_head >= q_heads ||
      q_heads != kPhase87Wup1PagedQHeads ||
      kv_heads != kPhase87Wup1PagedKvHeads ||
      head_dim != kPhase87Wup1PagedHeadDim || wave >= kSplits) {
    return;
  }
  const uint32_t lane = threadIdx.x & (kWaveSize - 1U);
  const uint32_t kv_head = query_head / (q_heads / kv_heads);
  const uint16_t *const query_row =
      query +
      (static_cast<uint64_t>(query_index) * q_heads + query_head) * head_dim;
  const uint64_t committed_kv_length =
      start_position + static_cast<uint64_t>(query_index) + 1U;
  const uint64_t split_begin =
      committed_kv_length * static_cast<uint64_t>(wave) / kSplits;
  const uint64_t split_end =
      committed_kv_length * static_cast<uint64_t>(wave + 1U) / kSplits;

  float accumulations[kElementsPerLane];
  float query_values[kElementsPerLane];
#pragma unroll
  for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
    accumulations[index] = 0.0F;
    if constexpr (UseQueryPreload) {
      const uint32_t current = lane + index * kWaveSize;
      query_values[index] = bf16_to_f32(query_row[current]);
    }
  }

  const uint64_t base =
      ((static_cast<uint64_t>(query_index) * q_heads + query_head) * kSplits +
       wave) *
      kPhase87Wup1PagedWorkspaceStride;
  if (split_begin >= split_end) {
    for (uint32_t index = lane; index < head_dim; index += kWaveSize) {
      workspace[base + 2U + index] = 0.0F;
    }
    if (lane == 0U) {
      workspace[base] = -std::numeric_limits<float>::infinity();
      workspace[base + 1U] = 0.0F;
    }
    return;
  }

  float local_maximum = -std::numeric_limits<float>::infinity();
  float local_denominator = 0.0F;
  const uint64_t first_page = split_begin / kPhase87Wup1PagedTokensPerPage;
  const uint64_t page_count =
      split_end == 0U ? 0U
                      : (split_end - 1U) / kPhase87Wup1PagedTokensPerPage + 1U;
  for (uint64_t logical_page = first_page; logical_page < page_count;
       ++logical_page) {
    // As in the GQA-shared kernel, this table load is outside the token/tile
    // loops and therefore occurs once per page-aligned split subinterval.
    const uint32_t physical_page =
        load_wup1_wave_uniform_page(block_table, logical_page);
    const uint64_t page_begin = logical_page * kPhase87Wup1PagedTokensPerPage;
    const uint64_t segment_begin =
        split_begin > page_begin ? split_begin : page_begin;
    const uint64_t page_end = page_end_token(logical_page);
    const uint64_t segment_end = split_end < page_end ? split_end : page_end;

    const uint64_t local_begin = segment_begin - page_begin;
    const uint64_t local_end = segment_end - page_begin;
    const uint64_t row_base = static_cast<uint64_t>(physical_page) *
                                  kPhase87Wup1PagedTokensPerPage *
                                  kPhase87Wup1PagedKvHeads +
                              kv_head;
    for (uint64_t local_token = local_begin; local_token < local_end;
         ++local_token) {
      const uint64_t kv_row = row_base + local_token * kPhase87Wup1PagedKvHeads;
      const uint64_t packed_key_scales =
          load_wup1_packed_row_scales(key_scales, kv_row, lane);
      const uint64_t packed_value_scales =
          load_wup1_packed_row_scales(value_scales, kv_row, lane);
      float partial = 0.0F;
#pragma unroll
      for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
        const uint32_t current = lane + index * kWaveSize;
        if constexpr (UseQueryPreload) {
          partial += query_values[index] *
                     load_wup1_e4_with_packed_scale(key, kv_row, current,
                                                    packed_key_scales, index);
        } else {
          partial += bf16_to_f32(query_row[current]) *
                     load_wup1_e4_with_packed_scale(key, kv_row, current,
                                                    packed_key_scales, index);
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
        accumulations[index] = accumulations[index] * rescale +
                               contribution * load_wup1_e4_with_packed_scale(
                                                  value, kv_row, current,
                                                  packed_value_scales, index);
      }
    }
  }

  if (lane == 0U) {
    workspace[base] = local_maximum;
    workspace[base + 1U] = local_denominator;
  }
#pragma unroll
  for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
    const uint32_t current = lane + index * kWaveSize;
    workspace[base + 2U + current] = accumulations[index];
  }
}

} // namespace phase87_wup1_paged_detail

// Launch stage 1 of the WU-P1 paged decode probe.
//
// `key`/`value` are pools of physical 128-token pages in token-major
// [token,kv_head,head_dim] layout.  `key_scales`/`value_scales` use
// [token,kv_head,head_dim/32] E8M0 rows.  `block_table[logical_page]` gives
// the physical page number for that logical page.  The caller must allocate
// `workspace_bytes` for [query_count,24,split_count,258] FP32 values and use
// the existing stage-2 merge with that workspace.  `split_count` must be 32
// for committed lengths below 8192 and 128 for lengths at least 8192.
//
// Set `use_gqa_shared=true` for the gfx1030-style 192-thread GQA-shared
// kernel, or false for the gfx1201-style 32-thread wave-split kernel.  This
// function is deliberately a stage-1-only probe entry point; it is not part
// of the public C ABI and is not used by production dispatch.
hipError_t launch_wup1_paged_decode_stage1(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const uint32_t *const block_table, float *const workspace,
    const uint32_t query_count, const uint64_t start_position,
    const uint32_t split_count, const bool use_gqa_shared,
    const hipStream_t stream) noexcept {
  if (query == nullptr || key == nullptr || value == nullptr ||
      key_scales == nullptr || value_scales == nullptr ||
      block_table == nullptr || workspace == nullptr || query_count == 0U ||
      query_count > 3U || start_position > UINT64_MAX - query_count ||
      start_position + query_count == 0U ||
      (split_count != 32U && split_count != 128U)) {
    return hipErrorInvalidValue;
  }
  const uint64_t committed_kv_length = start_position + query_count;
  if ((split_count == 32U && committed_kv_length >= 8192U) ||
      (split_count == 128U && committed_kv_length < 8192U)) {
    return hipErrorInvalidValue;
  }
  if (use_gqa_shared) {
    if (split_count == 128U) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(phase87_wup1_paged_detail::gqa6_stage1<true, 128U>),
          dim3(query_count * kPhase87Wup1PagedKvHeads * 128U), dim3(192U), 0U,
          stream, query, key, value, key_scales, value_scales, block_table,
          workspace, query_count, start_position);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(phase87_wup1_paged_detail::gqa6_stage1<true, 32U>),
          dim3(query_count * kPhase87Wup1PagedKvHeads * 32U), dim3(192U), 0U,
          stream, query, key, value, key_scales, value_scales, block_table,
          workspace, query_count, start_position);
    }
  } else if (split_count == 128U) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            phase87_wup1_paged_detail::wave_split_stage1<128U, true>),
        dim3(query_count * kPhase87Wup1PagedQHeads * 128U), dim3(32U), 0U,
        stream, query, key, value, key_scales, value_scales, block_table,
        workspace, query_count, start_position, kPhase87Wup1PagedQHeads,
        kPhase87Wup1PagedKvHeads, kPhase87Wup1PagedHeadDim);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            phase87_wup1_paged_detail::wave_split_stage1<32U, true>),
        dim3(query_count * kPhase87Wup1PagedQHeads * 32U), dim3(32U), 0U,
        stream, query, key, value, key_scales, value_scales, block_table,
        workspace, query_count, start_position, kPhase87Wup1PagedQHeads,
        kPhase87Wup1PagedKvHeads, kPhase87Wup1PagedHeadDim);
  }
  return hipGetLastError();
}
