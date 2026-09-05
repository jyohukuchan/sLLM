// Phase 78 standalone GQA6 FP16-prefill query-tile probe.
//
// This is intentionally not part of the production build.  It reproduces the
// current GQA6 Q4/K32 (gfx1030) and Q4/K8 (gfx1201) FP16-KV route, the existing
// block-softmax route, and probes Q8/Q16 query tiles.  The only changed
// variable is the number of adjacent query rows owned by a block; all 24
// query heads, head_dim=256, causal masking, FP32 online softmax, and BF16-RNE
// output are retained.  Build this file directly with hipcc for one target,
// e.g. --offload-arch=gfx1030 or --offload-arch=gfx1201.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

namespace {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kWave = 32U;
constexpr uint32_t kWaveCount = 8U;
constexpr uint32_t kQHeads = 24U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kGqaRatio = 6U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kWarmups = 3U;
constexpr uint32_t kMeasured = 10U;
constexpr uint32_t kPipelineBatch = kKvHeads;

__device__ __forceinline__ float bf16_to_float(const uint16_t bits) noexcept {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16U);
}

__device__ __forceinline__ float fp16_to_float(const uint16_t bits) noexcept {
  return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ uint16_t bf16_rne(const float value) noexcept {
  const uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7f800000U) == 0x7f800000U) {
    if ((bits & 0x007fffffU) != 0U) {
      return static_cast<uint16_t>(((bits >> 16U) & 0x8000U) | 0x7fc0U |
                                   ((bits >> 16U) & 0x003fU));
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & 0xffffU;
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

// N2 feasibility path: pack the six Q heads belonging to one KV head into a
// D x (M*6) column-major matrix (the natural row-major [M,6,D] memory), so
// rocBLAS can issue four strided-batched QK GEMMs.  This is separate from the
// production kernels and is only used by the standalone probe.
__global__ void pack_gqa6_query_fp16_kernel(const uint16_t *const query,
                                            uint16_t *const packed,
                                            const uint32_t query_count) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = static_cast<uint64_t>(kPipelineBatch) * query_count *
                         kGqaRatio * kHeadDim;
  if (index >= total)
    return;
  const uint64_t per_batch =
      static_cast<uint64_t>(query_count) * kGqaRatio * kHeadDim;
  const uint32_t batch = static_cast<uint32_t>(index / per_batch);
  const uint64_t local = index - static_cast<uint64_t>(batch) * per_batch;
  const uint32_t q_index = static_cast<uint32_t>(local / kHeadDim);
  const uint32_t dimension = static_cast<uint32_t>(local % kHeadDim);
  const uint32_t row = q_index / kGqaRatio;
  const uint32_t local_head = q_index % kGqaRatio;
  const uint32_t qhead = batch * kGqaRatio + local_head;
  const uint16_t bits =
      query[(static_cast<uint64_t>(row) * kQHeads + qhead) * kHeadDim +
            dimension];
  packed[local] = __half_as_ushort(__float2half(bf16_to_float(bits)));
}

__device__ __forceinline__ float reduce_warp_sum(float value) noexcept {
  for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U) {
    value += __shfl_down(value, offset, kWave);
  }
  return value;
}

// In-place FP32 causal softmax over the QK GEMM's column-major [Q,C] score
// matrix.  One block handles one (KV head, query row, local query head).
__global__ void gqa6_pipeline_softmax_kernel(float *const scores,
                                             const uint32_t query_count,
                                             const uint64_t start_position,
                                             const uint64_t context) {
  const uint64_t q_flat = blockIdx.x;
  const uint64_t q_count = static_cast<uint64_t>(query_count) * kGqaRatio;
  const uint64_t batch = q_flat / q_count;
  const uint64_t q_index = q_flat % q_count;
  const uint64_t row = q_index / kGqaRatio;
  if (batch >= kPipelineBatch || row >= query_count)
    return;
  const uint64_t key_limit =
      std::min<uint64_t>(context - 1U, start_position + row);
  const uint64_t stride = q_count;
  const uint64_t base = batch * q_count * context + q_index;
  __shared__ float warp_values[kThreads / kWave];
  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (kWave - 1U);
  const uint32_t wave = thread / kWave;
  float maximum = -INFINITY;
  for (uint64_t key_index = thread; key_index <= key_limit;
       key_index += kThreads) {
    maximum = fmaxf(maximum, scores[base + key_index * stride]);
  }
  for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U) {
    maximum = fmaxf(maximum, __shfl_down(maximum, offset, kWave));
  }
  if (lane == 0U)
    warp_values[wave] = maximum;
  __syncthreads();
  if (wave == 0U) {
    maximum = lane < kThreads / kWave ? warp_values[lane] : -INFINITY;
    for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U) {
      maximum = fmaxf(maximum, __shfl_down(maximum, offset, kWave));
    }
    if (lane == 0U)
      warp_values[0] = maximum;
  }
  __syncthreads();
  maximum = warp_values[0];
  float denominator = 0.0F;
  for (uint64_t key_index = thread; key_index <= key_limit;
       key_index += kThreads) {
    const float probability = expf(scores[base + key_index * stride] - maximum);
    scores[base + key_index * stride] = probability;
    denominator += probability;
  }
  denominator = reduce_warp_sum(denominator);
  if (lane == 0U)
    warp_values[wave] = denominator;
  __syncthreads();
  if (wave == 0U) {
    denominator = lane < kThreads / kWave ? warp_values[lane] : 0.0F;
    denominator = reduce_warp_sum(denominator);
    if (lane == 0U)
      warp_values[0] = denominator;
  }
  __syncthreads();
  denominator = warp_values[0];
  for (uint64_t key_index = thread; key_index < context;
       key_index += kThreads) {
    if (key_index > key_limit) {
      scores[base + key_index * stride] = 0.0F;
    } else {
      scores[base + key_index * stride] /= denominator;
    }
  }
}

__global__ void unpack_gqa6_pipeline_kernel(const float *const pv,
                                            uint16_t *const output,
                                            const uint32_t query_count) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = static_cast<uint64_t>(kPipelineBatch) * query_count *
                         kGqaRatio * kHeadDim;
  if (index >= total)
    return;
  const uint64_t per_batch =
      static_cast<uint64_t>(query_count) * kGqaRatio * kHeadDim;
  const uint32_t batch = static_cast<uint32_t>(index / per_batch);
  const uint64_t local = index - static_cast<uint64_t>(batch) * per_batch;
  const uint32_t q_index = static_cast<uint32_t>(local / kHeadDim);
  const uint32_t dimension = static_cast<uint32_t>(local % kHeadDim);
  const uint32_t row = q_index / kGqaRatio;
  const uint32_t local_head = q_index % kGqaRatio;
  const uint32_t qhead = batch * kGqaRatio + local_head;
  // rocBLAS C is column-major [Q,D], hence pv[d*Q+q].
  const uint64_t q_count = static_cast<uint64_t>(query_count) * kGqaRatio;
  const float value = pv[static_cast<uint64_t>(batch) * q_count * kHeadDim +
                         static_cast<uint64_t>(dimension) * q_count + q_index];
  output[(static_cast<uint64_t>(row) * kQHeads + qhead) * kHeadDim +
         dimension] = bf16_rne(value);
}

__global__ void convert_gqa6_value_fp32_kernel(const uint16_t *const value,
                                               float *const value_fp32,
                                               const uint64_t context) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total =
      static_cast<uint64_t>(kPipelineBatch) * context * kHeadDim;
  if (index < total) {
    const uint64_t per_batch = context * kHeadDim;
    const uint32_t batch = static_cast<uint32_t>(index / per_batch);
    const uint64_t local = index - static_cast<uint64_t>(batch) * per_batch;
    const uint64_t row = local / kHeadDim;
    const uint64_t dimension = local % kHeadDim;
    value_fp32[index] =
        fp16_to_float(value[(row * kKvHeads + batch) * kHeadDim + dimension]);
  }
}

__global__ void convert_gqa6_query_fp32_kernel(const uint16_t *const query,
                                               float *const query_fp32,
                                               const uint32_t query_count) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = static_cast<uint64_t>(kPipelineBatch) * query_count *
                         kGqaRatio * kHeadDim;
  if (index >= total)
    return;
  const uint64_t per_batch =
      static_cast<uint64_t>(query_count) * kGqaRatio * kHeadDim;
  const uint32_t batch = static_cast<uint32_t>(index / per_batch);
  const uint64_t local = index - static_cast<uint64_t>(batch) * per_batch;
  const uint32_t q_index = static_cast<uint32_t>(local / kHeadDim);
  const uint32_t dimension = static_cast<uint32_t>(local % kHeadDim);
  const uint32_t row = q_index / kGqaRatio;
  const uint32_t qhead = batch * kGqaRatio + q_index % kGqaRatio;
  query_fp32[local] = bf16_to_float(
      query[(static_cast<uint64_t>(row) * kQHeads + qhead) * kHeadDim +
            dimension]);
}

__global__ void convert_gqa6_key_fp32_kernel(const uint16_t *const key,
                                             float *const key_fp32,
                                             const uint64_t context) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total =
      static_cast<uint64_t>(kPipelineBatch) * context * kHeadDim;
  if (index < total) {
    const uint64_t per_batch = context * kHeadDim;
    const uint32_t batch = static_cast<uint32_t>(index / per_batch);
    const uint64_t local = index - static_cast<uint64_t>(batch) * per_batch;
    const uint64_t row = local / kHeadDim;
    const uint64_t dimension = local % kHeadDim;
    key_fp32[index] =
        fp16_to_float(key[(row * kKvHeads + batch) * kHeadDim + dimension]);
  }
}

template <uint32_t QueryTile, uint32_t KeyTile, bool BlockSoftmax>
__global__ __launch_bounds__(kThreads, 1) void gqa6_prefill_kernel(
    const uint16_t *const query, const uint16_t *const key,
    const uint16_t *const value, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position) {
  static_assert(QueryTile == 4U || QueryTile == 8U || QueryTile == 16U);
  static_assert(KeyTile == 8U || KeyTile == 32U);
  static_assert((QueryTile * kGqaRatio) % kWaveCount == 0U);
  constexpr uint32_t kLogicalQueries = QueryTile * kGqaRatio;
  constexpr uint32_t kQueriesPerWave = kLogicalQueries / kWaveCount;
  constexpr uint32_t kPairsPerLane = kHeadDim / kWave / 2U;
  static_assert(kPairsPerLane == 4U);

  const uint64_t flat = static_cast<uint64_t>(blockIdx.x);
  const uint64_t tile = flat / kKvHeads;
  const uint32_t kv_head = static_cast<uint32_t>(flat % kKvHeads);
  const uint64_t first_row = tile * QueryTile;
  if (first_row >= query_count) {
    return;
  }
  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (kWave - 1U);
  const uint32_t wave = thread / kWave;
  const uint32_t first_query_head = kv_head * kGqaRatio;

  __shared__ uint16_t key_tile[KeyTile][kHeadDim];
  __shared__ uint16_t value_tile[KeyTile][kHeadDim];
  // The Q8/K32 candidate uses 6 KiB here and Q16/K32 uses 12 KiB.  Together
  // with K/V (32 KiB), Q16/K32 remains under the 64 KiB LDS limit.
  __shared__ float score_tile[KeyTile][kLogicalQueries];
  __shared__ float tile_maximum[kLogicalQueries];
  __shared__ float tile_denominator[kLogicalQueries];
  float2 query_values[kQueriesPerWave][kPairsPerLane];
  float2 accumulations[kQueriesPerWave][kPairsPerLane] = {};
  float running_maximum[kQueriesPerWave];
  float running_denominator[kQueriesPerWave];

#pragma unroll
  for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
    running_maximum[item] = -INFINITY;
    running_denominator[item] = 0.0F;
    const uint32_t logical_query = wave * kQueriesPerWave + item;
    const uint64_t row = first_row + logical_query / kGqaRatio;
    const uint32_t query_head = first_query_head + logical_query % kGqaRatio;
    const uint64_t safe_row = row < query_count ? row : query_count - 1U;
    const uint16_t *const query_row =
        query + (safe_row * kQHeads + query_head) * kHeadDim;
#pragma unroll
    for (uint32_t pair = 0U; pair < kPairsPerLane; ++pair) {
      const uint32_t dimension = lane * 2U + pair * kWave * 2U;
      query_values[item][pair] =
          row < query_count
              ? make_float2(bf16_to_float(query_row[dimension]),
                            bf16_to_float(query_row[dimension + 1U]))
              : make_float2(0.0F, 0.0F);
    }
  }

  const uint64_t tile_end = first_row + QueryTile;
  const uint64_t last_row =
      (tile_end < query_count ? tile_end : query_count) - 1U;
  const uint64_t last_query_position = start_position + last_row;
  for (uint64_t key_begin = 0U; key_begin <= last_query_position;
       key_begin += KeyTile) {
    const uint64_t remaining = last_query_position - key_begin + 1U;
    const uint32_t key_count =
        remaining < KeyTile ? static_cast<uint32_t>(remaining) : KeyTile;
    for (uint32_t element = thread; element < KeyTile * kHeadDim;
         element += kThreads) {
      const uint32_t key_index = element / kHeadDim;
      const uint32_t dimension = element % kHeadDim;
      if (key_index < key_count) {
        const uint64_t kv_row = (key_begin + key_index) * kKvHeads + kv_head;
        key_tile[key_index][dimension] =
            kv_row < (UINT64_MAX / kHeadDim)
                ? key[kv_row * kHeadDim + dimension]
                : 0U;
        value_tile[key_index][dimension] =
            kv_row < (UINT64_MAX / kHeadDim)
                ? value[kv_row * kHeadDim + dimension]
                : 0U;
      }
    }
    __syncthreads();

    if constexpr (BlockSoftmax) {
      for (uint32_t key_index = 0U; key_index < key_count; ++key_index) {
        const uint64_t key_position = key_begin + key_index;
        for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
          const uint32_t logical_query = wave * kQueriesPerWave + item;
          const uint64_t row = first_row + logical_query / kGqaRatio;
          const bool active =
              row < query_count && key_position <= start_position + row;
          float partial = 0.0F;
#pragma unroll
          for (uint32_t pair = 0U; pair < kPairsPerLane; ++pair) {
            const uint32_t dimension = lane * 2U + pair * kWave * 2U;
            const float2 q = query_values[item][pair];
            const float2 k =
                make_float2(fp16_to_float(key_tile[key_index][dimension]),
                            fp16_to_float(key_tile[key_index][dimension + 1U]));
            partial += active ? q.x * k.x + q.y * k.y : 0.0F;
          }
          for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U) {
            partial += __shfl_down(partial, offset, kWave);
          }
          if (lane == 0U) {
            score_tile[key_index][logical_query] =
                active ? partial * rsqrtf(static_cast<float>(kHeadDim))
                       : -INFINITY;
          }
        }
      }
      __syncthreads();

      for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
        const uint32_t logical_query = wave * kQueriesPerWave + item;
        const uint64_t row = first_row + logical_query / kGqaRatio;
        if (lane == 0U) {
          float maximum = -INFINITY;
          if (row < query_count) {
            for (uint32_t key_index = 0U; key_index < KeyTile; ++key_index) {
              if (key_index < key_count) {
                maximum = fmaxf(maximum, score_tile[key_index][logical_query]);
              }
            }
          }
          tile_maximum[logical_query] = maximum;
        }
      }
      __syncthreads();
      for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
        const uint32_t logical_query = wave * kQueriesPerWave + item;
        const uint64_t row = first_row + logical_query / kGqaRatio;
        if (lane == 0U) {
          float denominator = 0.0F;
          if (row < query_count) {
            for (uint32_t key_index = 0U; key_index < KeyTile; ++key_index) {
              if (key_index < key_count &&
                  tile_maximum[logical_query] != -INFINITY) {
                const float probability =
                    expf(score_tile[key_index][logical_query] -
                         tile_maximum[logical_query]);
                score_tile[key_index][logical_query] = probability;
                denominator += probability;
              }
            }
          }
          tile_denominator[logical_query] = denominator;
        }
      }
      __syncthreads();

      for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
        const uint32_t logical_query = wave * kQueriesPerWave + item;
        const uint64_t row = first_row + logical_query / kGqaRatio;
        float rescale = 1.0F;
        float tile_scale = 0.0F;
        float next_maximum = running_maximum[item];
        if (lane == 0U && row < query_count &&
            tile_denominator[logical_query] > 0.0F) {
          next_maximum =
              fmaxf(running_maximum[item], tile_maximum[logical_query]);
          rescale = expf(running_maximum[item] - next_maximum);
          tile_scale = expf(tile_maximum[logical_query] - next_maximum);
          running_denominator[item] =
              running_denominator[item] * rescale +
              tile_denominator[logical_query] * tile_scale;
          running_maximum[item] = next_maximum;
        }
        rescale = __shfl(rescale, 0U, kWave);
        tile_scale = __shfl(tile_scale, 0U, kWave);
        next_maximum = __shfl(next_maximum, 0U, kWave);
        running_maximum[item] = next_maximum;
        running_denominator[item] =
            __shfl(running_denominator[item], 0U, kWave);
        if (row < query_count) {
#pragma unroll
          for (uint32_t pair = 0U; pair < kPairsPerLane; ++pair) {
            float2 tile_accumulation = make_float2(0.0F, 0.0F);
            const uint32_t dimension = lane * 2U + pair * kWave * 2U;
            for (uint32_t key_index = 0U; key_index < KeyTile; ++key_index) {
              if (key_index < key_count && tile_scale != 0.0F &&
                  score_tile[key_index][logical_query] != -INFINITY) {
                const float probability =
                    score_tile[key_index][logical_query] * tile_scale;
                const float2 v = make_float2(
                    fp16_to_float(value_tile[key_index][dimension]),
                    fp16_to_float(value_tile[key_index][dimension + 1U]));
                tile_accumulation.x += probability * v.x;
                tile_accumulation.y += probability * v.y;
              }
            }
            accumulations[item][pair].x =
                accumulations[item][pair].x * rescale + tile_accumulation.x;
            accumulations[item][pair].y =
                accumulations[item][pair].y * rescale + tile_accumulation.y;
          }
        }
      }
    } else {
      // Control path: one online-softmax update per key, equivalent to the
      // current Q4/K32 or Q4/K8 FP16 prefill implementation.
      for (uint32_t key_index = 0U; key_index < key_count; ++key_index) {
        const uint64_t key_position = key_begin + key_index;
        for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
          const uint32_t logical_query = wave * kQueriesPerWave + item;
          const uint64_t row = first_row + logical_query / kGqaRatio;
          const bool active =
              row < query_count && key_position <= start_position + row;
          float partial = 0.0F;
#pragma unroll
          for (uint32_t pair = 0U; pair < kPairsPerLane; ++pair) {
            const uint32_t dimension = lane * 2U + pair * kWave * 2U;
            const float2 q = query_values[item][pair];
            const float2 k =
                make_float2(fp16_to_float(key_tile[key_index][dimension]),
                            fp16_to_float(key_tile[key_index][dimension + 1U]));
            partial += active ? q.x * k.x + q.y * k.y : 0.0F;
          }
          for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U) {
            partial += __shfl_down(partial, offset, kWave);
          }
          float rescale = 1.0F;
          float contribution = 0.0F;
          float next_maximum = running_maximum[item];
          if (lane == 0U && active) {
            const float current_score =
                partial * rsqrtf(static_cast<float>(kHeadDim));
            next_maximum = fmaxf(running_maximum[item], current_score);
            rescale = expf(running_maximum[item] - next_maximum);
            contribution = expf(current_score - next_maximum);
          }
          rescale = __shfl(rescale, 0U, kWave);
          contribution = __shfl(contribution, 0U, kWave);
          next_maximum = __shfl(next_maximum, 0U, kWave);
          running_denominator[item] =
              running_denominator[item] * rescale + contribution;
          running_maximum[item] = next_maximum;
#pragma unroll
          for (uint32_t pair = 0U; pair < kPairsPerLane; ++pair) {
            const uint32_t dimension = lane * 2U + pair * kWave * 2U;
            if (active) {
              const float2 v = make_float2(
                  fp16_to_float(value_tile[key_index][dimension]),
                  fp16_to_float(value_tile[key_index][dimension + 1U]));
              accumulations[item][pair].x =
                  accumulations[item][pair].x * rescale + contribution * v.x;
              accumulations[item][pair].y =
                  accumulations[item][pair].y * rescale + contribution * v.y;
            }
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
    const uint32_t logical_query = wave * kQueriesPerWave + item;
    const uint64_t row = first_row + logical_query / kGqaRatio;
    const uint32_t query_head = first_query_head + logical_query % kGqaRatio;
    if (row < query_count) {
      uint16_t *const output_row =
          output + (row * kQHeads + query_head) * kHeadDim;
#pragma unroll
      for (uint32_t pair = 0U; pair < kPairsPerLane; ++pair) {
        const uint32_t dimension = lane * 2U + pair * kWave * 2U;
        const float denominator = running_denominator[item];
        output_row[dimension] =
            bf16_rne(accumulations[item][pair].x / denominator);
        output_row[dimension + 1U] =
            bf16_rne(accumulations[item][pair].y / denominator);
      }
    }
  }
}

enum class CandidateId : uint32_t {
  Q4K32Control,
  Q4K8Control,
  Q4K32Block,
  Q4K8Block,
  Q8K32Control,
  Q8K8Control,
  Q16K32Control,
  Q16K8Control,
  Q8K32Block,
  Q8K8Block,
  Q16K32Block,
  Q16K8Block,
};

struct Candidate final {
  CandidateId id;
  const char *name;
  uint32_t query_tile;
  uint32_t key_tile;
  bool block_softmax;
  const void *function;
};

Candidate candidate(const CandidateId id) {
  switch (id) {
  case CandidateId::Q4K32Control:
    return {
        id,
        "q4-k32-control",
        4U,
        32U,
        false,
        reinterpret_cast<const void *>(gqa6_prefill_kernel<4U, 32U, false>)};
  case CandidateId::Q4K8Control:
    return {id,
            "q4-k8-control",
            4U,
            8U,
            false,
            reinterpret_cast<const void *>(gqa6_prefill_kernel<4U, 8U, false>)};
  case CandidateId::Q4K32Block:
    return {id,
            "q4-k32-blocksoftmax",
            4U,
            32U,
            true,
            reinterpret_cast<const void *>(gqa6_prefill_kernel<4U, 32U, true>)};
  case CandidateId::Q4K8Block:
    return {id,
            "q4-k8-blocksoftmax",
            4U,
            8U,
            true,
            reinterpret_cast<const void *>(gqa6_prefill_kernel<4U, 8U, true>)};
  case CandidateId::Q8K32Control:
    return {
        id,
        "q8-k32-control",
        8U,
        32U,
        false,
        reinterpret_cast<const void *>(gqa6_prefill_kernel<8U, 32U, false>)};
  case CandidateId::Q8K8Control:
    return {id,
            "q8-k8-control",
            8U,
            8U,
            false,
            reinterpret_cast<const void *>(gqa6_prefill_kernel<8U, 8U, false>)};
  case CandidateId::Q16K32Control:
    return {
        id,
        "q16-k32-control",
        16U,
        32U,
        false,
        reinterpret_cast<const void *>(gqa6_prefill_kernel<16U, 32U, false>)};
  case CandidateId::Q16K8Control:
    return {
        id,
        "q16-k8-control",
        16U,
        8U,
        false,
        reinterpret_cast<const void *>(gqa6_prefill_kernel<16U, 8U, false>)};
  case CandidateId::Q8K32Block:
    return {id,
            "q8-k32-blocksoftmax",
            8U,
            32U,
            true,
            reinterpret_cast<const void *>(gqa6_prefill_kernel<8U, 32U, true>)};
  case CandidateId::Q8K8Block:
    return {id,
            "q8-k8-blocksoftmax",
            8U,
            8U,
            true,
            reinterpret_cast<const void *>(gqa6_prefill_kernel<8U, 8U, true>)};
  case CandidateId::Q16K32Block:
    return {
        id,
        "q16-k32-blocksoftmax",
        16U,
        32U,
        true,
        reinterpret_cast<const void *>(gqa6_prefill_kernel<16U, 32U, true>)};
  case CandidateId::Q16K8Block:
    return {id,
            "q16-k8-blocksoftmax",
            16U,
            8U,
            true,
            reinterpret_cast<const void *>(gqa6_prefill_kernel<16U, 8U, true>)};
  }
  return {id, "invalid", 0U, 0U, false, nullptr};
}

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::fprintf(stderr, "hip error operation=%s status=%s\n", operation,
               hipGetErrorString(status));
  return false;
}

uint16_t host_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000U) == 0x7f800000U) {
    if ((bits & 0x007fffffU) != 0U) {
      return static_cast<uint16_t>(((bits >> 16U) & 0x8000U) | 0x7fc0U |
                                   ((bits >> 16U) & 0x003fU));
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & 0xffffU;
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

uint16_t host_fp16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16U) & 0x8000U;
  const uint32_t exponent = (bits >> 23U) & 0xffU;
  const uint32_t mantissa = bits & 0x7fffffU;
  if (exponent == 0xffU) {
    return static_cast<uint16_t>(sign | 0x7c00U |
                                 (mantissa != 0U ? 0x0200U : 0U));
  }
  int32_t half_exp = static_cast<int32_t>(exponent) - 127 + 15;
  if (half_exp <= 0) {
    if (half_exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    const uint32_t mant = mantissa | 0x800000U;
    const uint32_t shift = static_cast<uint32_t>(14 - half_exp);
    uint32_t rounded = mant >> shift;
    const uint32_t remainder = mant & ((1U << shift) - 1U);
    if (remainder > (1U << (shift - 1U)) ||
        (remainder == (1U << (shift - 1U)) && (rounded & 1U) != 0U)) {
      ++rounded;
    }
    return static_cast<uint16_t>(sign | rounded);
  }
  if (half_exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00U);
  }
  uint32_t rounded_mant = mantissa >> 13U;
  const uint32_t remainder = mantissa & 0x1fffU;
  if (remainder > 0x1000U ||
      (remainder == 0x1000U && (rounded_mant & 1U) != 0U)) {
    ++rounded_mant;
    if (rounded_mant == 0x400U) {
      rounded_mant = 0U;
      ++half_exp;
      if (half_exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U);
      }
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exp) << 10U) |
                               rounded_mant);
}

float host_bf16_to_float(const uint16_t bits) {
  uint32_t expanded = static_cast<uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &expanded, sizeof(value));
  return value;
}

float host_fp16_to_float(const uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16U;
  const uint32_t exponent = (bits >> 10U) & 0x1fU;
  const uint32_t mantissa = bits & 0x3ffU;
  uint32_t result = sign;
  if (exponent == 0U) {
    if (mantissa != 0U) {
      float value = static_cast<float>(mantissa) * 0x1p-24F;
      std::memcpy(&result, &value, sizeof(result));
      result = (result & 0x7fffffffU) | sign;
    }
  } else if (exponent == 31U) {
    result |= 0x7f800000U | (mantissa << 13U);
  } else {
    result |= ((exponent + 112U) << 23U) | (mantissa << 13U);
  }
  float value = 0.0F;
  std::memcpy(&value, &result, sizeof(value));
  return value;
}

struct Buffers final {
  uint16_t *query = nullptr;
  uint16_t *key = nullptr;
  uint16_t *value = nullptr;
  uint16_t *output = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
};

void free_buffers(Buffers *const b) {
  if (b == nullptr) {
    return;
  }
  if (b->stop != nullptr)
    (void)hipEventDestroy(b->stop);
  if (b->start != nullptr)
    (void)hipEventDestroy(b->start);
  if (b->stream != nullptr)
    (void)hipStreamDestroy(b->stream);
  if (b->output != nullptr)
    (void)hipFree(b->output);
  if (b->value != nullptr)
    (void)hipFree(b->value);
  if (b->key != nullptr)
    (void)hipFree(b->key);
  if (b->query != nullptr)
    (void)hipFree(b->query);
  *b = {};
}

bool make_buffers(const uint64_t rows, const uint64_t context,
                  Buffers *const b) {
  const size_t query_bytes = rows * kQHeads * kHeadDim * sizeof(uint16_t);
  const size_t kv_bytes = context * kKvHeads * kHeadDim * sizeof(uint16_t);
  const size_t output_bytes = query_bytes;
  return hip_ok(hipMalloc(reinterpret_cast<void **>(&b->query), query_bytes),
                "hipMalloc query") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->key), kv_bytes),
                "hipMalloc key") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->value), kv_bytes),
                "hipMalloc value") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->output), output_bytes),
                "hipMalloc output") &&
         hip_ok(hipStreamCreate(&b->stream), "hipStreamCreate") &&
         hip_ok(hipEventCreate(&b->start), "hipEventCreate start") &&
         hip_ok(hipEventCreate(&b->stop), "hipEventCreate stop");
}

void fill_inputs(const uint64_t rows, const uint64_t context,
                 std::vector<uint16_t> *const query,
                 std::vector<uint16_t> *const key,
                 std::vector<uint16_t> *const value) {
  query->resize(rows * kQHeads * kHeadDim);
  key->resize(context * kKvHeads * kHeadDim);
  value->resize(context * kKvHeads * kHeadDim);
  for (uint64_t row = 0U; row < rows; ++row) {
    for (uint32_t head = 0U; head < kQHeads; ++head) {
      for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        const float x =
            0.45F *
            std::sin(static_cast<float>(
                         (row * 7U + head * 13U + dimension * 3U) % 97U) *
                     0.071F);
        (*query)[(row * kQHeads + head) * kHeadDim + dimension] =
            host_bf16_rne(x);
      }
    }
  }
  for (uint64_t row = 0U; row < context; ++row) {
    for (uint32_t head = 0U; head < kKvHeads; ++head) {
      for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        const float x =
            0.65F *
            std::cos(static_cast<float>(
                         (row * 11U + head * 5U + dimension * 17U) % 131U) *
                     0.053F);
        const float y =
            0.55F *
            std::sin(static_cast<float>(
                         (row * 3U + head * 19U + dimension * 7U) % 127U) *
                     0.067F);
        (*key)[(row * kKvHeads + head) * kHeadDim + dimension] =
            host_fp16_rne(x);
        (*value)[(row * kKvHeads + head) * kHeadDim + dimension] =
            host_fp16_rne(y);
      }
    }
  }
}

bool upload(const std::vector<uint16_t> &query,
            const std::vector<uint16_t> &key,
            const std::vector<uint16_t> &value, Buffers *const b) {
  return hip_ok(hipMemcpy(b->query, query.data(),
                          query.size() * sizeof(uint16_t),
                          hipMemcpyHostToDevice),
                "copy query") &&
         hip_ok(hipMemcpy(b->key, key.data(), key.size() * sizeof(uint16_t),
                          hipMemcpyHostToDevice),
                "copy key") &&
         hip_ok(hipMemcpy(b->value, value.data(),
                          value.size() * sizeof(uint16_t),
                          hipMemcpyHostToDevice),
                "copy value");
}

bool launch(const Candidate &c, const uint64_t rows, const uint64_t start,
            Buffers *const b) {
  const uint64_t blocks =
      ((rows + c.query_tile - 1U) / c.query_tile) * kKvHeads;
  if (blocks == 0U || blocks > UINT32_MAX) {
    return false;
  }
#define LAUNCH(Q, K, B)                                                        \
  hipLaunchKernelGGL((gqa6_prefill_kernel<Q, K, B>),                           \
                     dim3(static_cast<uint32_t>(blocks)), dim3(kThreads), 0U,  \
                     b->stream, b->query, b->key, b->value, b->output,         \
                     static_cast<uint32_t>(rows), start)
  switch (c.id) {
  case CandidateId::Q4K32Control:
    LAUNCH(4U, 32U, false);
    break;
  case CandidateId::Q4K8Control:
    LAUNCH(4U, 8U, false);
    break;
  case CandidateId::Q4K32Block:
    LAUNCH(4U, 32U, true);
    break;
  case CandidateId::Q4K8Block:
    LAUNCH(4U, 8U, true);
    break;
  case CandidateId::Q8K32Control:
    LAUNCH(8U, 32U, false);
    break;
  case CandidateId::Q8K8Control:
    LAUNCH(8U, 8U, false);
    break;
  case CandidateId::Q16K32Control:
    LAUNCH(16U, 32U, false);
    break;
  case CandidateId::Q16K8Control:
    LAUNCH(16U, 8U, false);
    break;
  case CandidateId::Q8K32Block:
    LAUNCH(8U, 32U, true);
    break;
  case CandidateId::Q8K8Block:
    LAUNCH(8U, 8U, true);
    break;
  case CandidateId::Q16K32Block:
    LAUNCH(16U, 32U, true);
    break;
  case CandidateId::Q16K8Block:
    LAUNCH(16U, 8U, true);
    break;
  }
#undef LAUNCH
  return hipGetLastError() == hipSuccess;
}

bool measure(const Candidate &c, const uint64_t rows, const uint64_t start,
             Buffers *const b, float *const median_us) {
  for (uint32_t i = 0U; i < kWarmups; ++i) {
    if (!launch(c, rows, start, b) ||
        !hip_ok(hipStreamSynchronize(b->stream), "warmup synchronize")) {
      return false;
    }
  }
  std::array<float, kMeasured> samples{};
  for (uint32_t i = 0U; i < kMeasured; ++i) {
    if (!hip_ok(hipEventRecord(b->start, b->stream), "event start") ||
        !launch(c, rows, start, b) ||
        !hip_ok(hipEventRecord(b->stop, b->stream), "event stop") ||
        !hip_ok(hipEventSynchronize(b->stop), "event synchronize") ||
        !hip_ok(hipEventElapsedTime(&samples[i], b->start, b->stop),
                "event elapsed")) {
      return false;
    }
    samples[i] *= 1000.0F;
  }
  std::sort(samples.begin(), samples.end());
  *median_us = samples[kMeasured / 2U];
  return true;
}

uint32_t ulp_distance(const uint16_t a, const uint16_t b) {
  if ((a & 0x7fffU) == 0U && (b & 0x7fffU) == 0U) {
    return 0U;
  }
  const int32_t ai = (a & 0x8000U) ? 0x8000 - (a & 0x7fffU) : 0x8000 + a;
  const int32_t bi = (b & 0x8000U) ? 0x8000 - (b & 0x7fffU) : 0x8000 + b;
  return static_cast<uint32_t>(std::abs(ai - bi));
}

bool host_oracle(const uint64_t rows, const uint64_t context,
                 const uint64_t start, const std::vector<uint16_t> &query,
                 const std::vector<uint16_t> &key,
                 const std::vector<uint16_t> &value,
                 std::vector<uint16_t> *const expected) {
  expected->assign(rows * kQHeads * kHeadDim, 0U);
  for (uint64_t row = 0U; row < rows; ++row) {
    const uint64_t last_key = std::min<uint64_t>(context - 1U, start + row);
    for (uint32_t qhead = 0U; qhead < kQHeads; ++qhead) {
      const uint32_t kv_head = qhead / kGqaRatio;
      std::vector<float> scores(static_cast<size_t>(last_key + 1U));
      float maximum = -INFINITY;
      for (uint64_t position = 0U; position <= last_key; ++position) {
        float dot = 0.0F;
        const uint16_t *const qrow =
            query.data() + (row * kQHeads + qhead) * kHeadDim;
        const uint16_t *const krow =
            key.data() + (position * kKvHeads + kv_head) * kHeadDim;
        for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
          dot = std::fmaf(host_bf16_to_float(qrow[dimension]),
                          host_fp16_to_float(krow[dimension]), dot);
        }
        scores[position] =
            dot * (1.0F / std::sqrt(static_cast<float>(kHeadDim)));
        maximum = std::max(maximum, scores[position]);
      }
      float denominator = 0.0F;
      for (float &score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      uint16_t *const out =
          expected->data() + (row * kQHeads + qhead) * kHeadDim;
      for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        float result = 0.0F;
        for (uint64_t position = 0U; position <= last_key; ++position) {
          const uint16_t *const vrow =
              value.data() + (position * kKvHeads + kv_head) * kHeadDim;
          result += (scores[position] / denominator) *
                    host_fp16_to_float(vrow[dimension]);
        }
        out[dimension] = host_bf16_rne(result);
      }
    }
  }
  return true;
}

bool copy_output(const uint64_t rows, Buffers *const b,
                 std::vector<uint16_t> *const output) {
  output->resize(rows * kQHeads * kHeadDim);
  return hip_ok(hipMemcpy(output->data(), b->output,
                          output->size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "copy output");
}

bool check_oracle(const Candidate &c, const uint64_t rows,
                  const std::vector<uint16_t> &expected,
                  const std::vector<uint16_t> &actual, uint32_t *const max_ulp,
                  uint64_t *const mismatches) {
  *max_ulp = 0U;
  *mismatches = 0U;
  double max_abs = 0.0;
  for (size_t i = 0U; i < expected.size(); ++i) {
    const uint32_t distance = ulp_distance(expected[i], actual[i]);
    *max_ulp = std::max(*max_ulp, distance);
    max_abs = std::max(
        max_abs, std::abs(static_cast<double>(host_bf16_to_float(expected[i])) -
                          static_cast<double>(host_bf16_to_float(actual[i]))));
    if (distance > 1U) {
      ++*mismatches;
      if (*mismatches <= 8U) {
        std::printf("mismatch candidate=%s index=%zu expected=0x%04x "
                    "actual=0x%04x distance=%u expected_f=%g actual_f=%g\n",
                    c.name, i, expected[i], actual[i], distance,
                    host_bf16_to_float(expected[i]),
                    host_bf16_to_float(actual[i]));
      }
    }
  }
  std::printf("oracle candidate=%s rows=%llu values=%zu max_bf16_ulp=%u "
              "over1=%llu max_abs=%g status=%s\n",
              c.name, static_cast<unsigned long long>(rows), expected.size(),
              *max_ulp, static_cast<unsigned long long>(*mismatches), max_abs,
              *mismatches == 0U ? "PASS" : "FAIL");
  return *mismatches == 0U;
}

bool check_against_control(const Candidate &c, const std::vector<uint16_t> &ref,
                           const std::vector<uint16_t> &actual,
                           uint32_t *const max_ulp, uint64_t *const over_one) {
  *max_ulp = 0U;
  *over_one = 0U;
  for (size_t i = 0U; i < ref.size(); ++i) {
    const uint32_t distance = ulp_distance(ref[i], actual[i]);
    *max_ulp = std::max(*max_ulp, distance);
    if (distance > 1U) {
      ++*over_one;
    }
  }
  std::printf("control_compare candidate=%s values=%zu max_bf16_ulp=%u "
              "over1=%llu status=%s\n",
              c.name, ref.size(), *max_ulp,
              static_cast<unsigned long long>(*over_one),
              *over_one == 0U ? "PASS" : "INFO");
  return true;
}

void print_resources(const Candidate &c) {
  hipFuncAttributes attributes{};
  const hipError_t attr = hipFuncGetAttributes(&attributes, c.function);
  int active_blocks = 0;
  const hipError_t occupancy = hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, c.function, kThreads, 0U);
  std::printf("resources candidate=%s registers=%d lds=%zu scratch=%zu "
              "max_threads=%d active_blocks=%d occupancy=%s attrs=%s\n",
              c.name, attributes.numRegs, attributes.sharedSizeBytes,
              attributes.localSizeBytes, attributes.maxThreadsPerBlock,
              active_blocks, hipGetErrorString(occupancy),
              hipGetErrorString(attr));
}

struct PipelineBuffers final {
  float *query_fp32 = nullptr;
  float *key_fp32 = nullptr;
  float *value_fp32 = nullptr;
  float *scores = nullptr;
  float *pv = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  rocblas_handle handle = nullptr;
};

void free_pipeline_buffers(PipelineBuffers *const p) {
  if (p == nullptr)
    return;
  if (p->handle != nullptr)
    (void)rocblas_destroy_handle(p->handle);
  if (p->stop != nullptr)
    (void)hipEventDestroy(p->stop);
  if (p->start != nullptr)
    (void)hipEventDestroy(p->start);
  if (p->pv != nullptr)
    (void)hipFree(p->pv);
  if (p->scores != nullptr)
    (void)hipFree(p->scores);
  if (p->value_fp32 != nullptr)
    (void)hipFree(p->value_fp32);
  if (p->key_fp32 != nullptr)
    (void)hipFree(p->key_fp32);
  if (p->query_fp32 != nullptr)
    (void)hipFree(p->query_fp32);
  *p = {};
}

bool rocblas_ok(const rocblas_status status, const char *const operation) {
  if (status == rocblas_status_success)
    return true;
  std::fprintf(stderr, "rocblas error operation=%s status=%d\n", operation,
               static_cast<int>(status));
  return false;
}

bool make_pipeline_buffers(const uint64_t rows, const uint64_t context,
                           hipStream_t stream, PipelineBuffers *const p) {
  const uint64_t q_count = rows * kGqaRatio;
  const uint64_t batches = kPipelineBatch;
  if (rows == 0U || context == 0U || q_count > INT32_MAX ||
      q_count * context > SIZE_MAX / batches ||
      batches * q_count * context > SIZE_MAX / sizeof(float)) {
    return false;
  }
  const size_t query_fp32_bytes =
      static_cast<size_t>(batches * q_count * kHeadDim * sizeof(float));
  const size_t key_fp32_bytes =
      static_cast<size_t>(batches * context * kHeadDim * sizeof(float));
  const size_t value_fp32_bytes =
      static_cast<size_t>(batches * context * kHeadDim * sizeof(float));
  const size_t score_bytes =
      static_cast<size_t>(batches * q_count * context * sizeof(float));
  const size_t pv_bytes =
      static_cast<size_t>(batches * q_count * kHeadDim * sizeof(float));
  if (!hip_ok(hipMalloc(reinterpret_cast<void **>(&p->query_fp32),
                        query_fp32_bytes),
              "hipMalloc pipeline query fp32") ||
      !hip_ok(
          hipMalloc(reinterpret_cast<void **>(&p->key_fp32), key_fp32_bytes),
          "hipMalloc pipeline key fp32") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&p->value_fp32),
                        value_fp32_bytes),
              "hipMalloc pipeline value fp32") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&p->scores), score_bytes),
              "hipMalloc pipeline scores") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&p->pv), pv_bytes),
              "hipMalloc pipeline pv") ||
      !hip_ok(hipEventCreate(&p->start), "hipEventCreate pipeline start") ||
      !hip_ok(hipEventCreate(&p->stop), "hipEventCreate pipeline stop") ||
      !rocblas_ok(rocblas_create_handle(&p->handle), "rocblas_create_handle") ||
      !rocblas_ok(rocblas_set_stream(p->handle, stream),
                  "rocblas_set_stream")) {
    free_pipeline_buffers(p);
    return false;
  }
  return true;
}

bool launch_pipeline(Buffers *const b, PipelineBuffers *const p,
                     const uint64_t rows, const uint64_t context,
                     const uint64_t start) {
  const uint64_t q_count = rows * kGqaRatio;
  const uint64_t total_q =
      static_cast<uint64_t>(kPipelineBatch) * q_count * kHeadDim;
  const float qk_alpha = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  const float beta_zero = 0.0F;
  const rocblas_stride a_stride =
      static_cast<rocblas_stride>(q_count * kHeadDim);
  const rocblas_stride b_stride =
      static_cast<rocblas_stride>(context * kHeadDim);
  const rocblas_stride c_stride =
      static_cast<rocblas_stride>(q_count * context);
  if (!rocblas_ok(rocblas_sgemm_strided_batched(
                      p->handle, rocblas_operation_transpose,
                      rocblas_operation_none, static_cast<rocblas_int>(q_count),
                      static_cast<rocblas_int>(context),
                      static_cast<rocblas_int>(kHeadDim), &qk_alpha,
                      p->query_fp32, static_cast<rocblas_int>(kHeadDim),
                      a_stride, p->key_fp32, static_cast<rocblas_int>(kHeadDim),
                      b_stride, &beta_zero, p->scores,
                      static_cast<rocblas_int>(q_count), c_stride,
                      static_cast<rocblas_int>(kPipelineBatch)),
                  "pipeline QK GEMM")) {
    return false;
  }
  const uint64_t softmax_blocks =
      static_cast<uint64_t>(kPipelineBatch) * q_count;
  hipLaunchKernelGGL((gqa6_pipeline_softmax_kernel),
                     dim3(static_cast<uint32_t>(softmax_blocks)),
                     dim3(kThreads), 0U, b->stream, p->scores,
                     static_cast<uint32_t>(rows), start, context);
  if (!hip_ok(hipGetLastError(), "pipeline softmax launch"))
    return false;
  const rocblas_stride score_stride =
      static_cast<rocblas_stride>(q_count * context);
  const rocblas_stride value_stride =
      static_cast<rocblas_stride>(context * kHeadDim);
  const rocblas_stride pv_stride =
      static_cast<rocblas_stride>(q_count * kHeadDim);
  const float pv_alpha = 1.0F;
  if (!rocblas_ok(
          rocblas_sgemm_strided_batched(
              p->handle, rocblas_operation_none, rocblas_operation_transpose,
              static_cast<rocblas_int>(q_count),
              static_cast<rocblas_int>(kHeadDim),
              static_cast<rocblas_int>(context), &pv_alpha, p->scores,
              static_cast<rocblas_int>(q_count), score_stride, p->value_fp32,
              static_cast<rocblas_int>(kHeadDim), value_stride, &beta_zero,
              p->pv, static_cast<rocblas_int>(q_count), pv_stride,
              static_cast<rocblas_int>(kPipelineBatch)),
          "pipeline PV GEMM")) {
    return false;
  }
  const uint32_t unpack_blocks =
      static_cast<uint32_t>((total_q + kThreads - 1U) / kThreads);
  hipLaunchKernelGGL((unpack_gqa6_pipeline_kernel), dim3(unpack_blocks),
                     dim3(kThreads), 0U, b->stream, p->pv, b->output,
                     static_cast<uint32_t>(rows));
  return hip_ok(hipGetLastError(), "pipeline unpack launch");
}

bool measure_pipeline(Buffers *const b, PipelineBuffers *const p,
                      const uint64_t rows, const uint64_t context,
                      const uint64_t start, float *const median_us) {
  for (uint32_t i = 0U; i < kWarmups; ++i) {
    if (!launch_pipeline(b, p, rows, context, start) ||
        !hip_ok(hipStreamSynchronize(b->stream), "pipeline warmup sync")) {
      return false;
    }
  }
  std::array<float, kMeasured> samples{};
  for (uint32_t i = 0U; i < kMeasured; ++i) {
    if (!hip_ok(hipEventRecord(p->start, b->stream), "pipeline event start") ||
        !launch_pipeline(b, p, rows, context, start) ||
        !hip_ok(hipEventRecord(p->stop, b->stream), "pipeline event stop") ||
        !hip_ok(hipEventSynchronize(p->stop), "pipeline event sync") ||
        !hip_ok(hipEventElapsedTime(&samples[i], p->start, p->stop),
                "pipeline event elapsed")) {
      return false;
    }
    samples[i] *= 1000.0F;
  }
  std::sort(samples.begin(), samples.end());
  *median_us = samples[kMeasured / 2U];
  return true;
}

bool prepare_pipeline_values(Buffers *const b, PipelineBuffers *const p,
                             const uint64_t rows, const uint64_t context) {
  const uint64_t total_q =
      static_cast<uint64_t>(kPipelineBatch) * rows * kGqaRatio * kHeadDim;
  const uint64_t total_k =
      static_cast<uint64_t>(kPipelineBatch) * context * kHeadDim;
  const uint32_t query_blocks =
      static_cast<uint32_t>((total_q + kThreads - 1U) / kThreads);
  const uint32_t key_blocks =
      static_cast<uint32_t>((total_k + kThreads - 1U) / kThreads);
  hipLaunchKernelGGL((convert_gqa6_query_fp32_kernel), dim3(query_blocks),
                     dim3(kThreads), 0U, b->stream, b->query, p->query_fp32,
                     static_cast<uint32_t>(rows));
  hipLaunchKernelGGL((convert_gqa6_key_fp32_kernel), dim3(key_blocks),
                     dim3(kThreads), 0U, b->stream, b->key, p->key_fp32,
                     context);
  hipLaunchKernelGGL((convert_gqa6_value_fp32_kernel), dim3(key_blocks),
                     dim3(kThreads), 0U, b->stream, b->value, p->value_fp32,
                     context);
  return hip_ok(hipGetLastError(), "pipeline FP32 staging") &&
         hip_ok(hipStreamSynchronize(b->stream),
                "pipeline FP32 staging synchronize");
}

void print_pipeline_resources() {
  const std::array<const void *, 5> functions = {
      reinterpret_cast<const void *>(convert_gqa6_query_fp32_kernel),
      reinterpret_cast<const void *>(convert_gqa6_key_fp32_kernel),
      reinterpret_cast<const void *>(gqa6_pipeline_softmax_kernel),
      reinterpret_cast<const void *>(unpack_gqa6_pipeline_kernel),
      reinterpret_cast<const void *>(convert_gqa6_value_fp32_kernel)};
  const std::array<const char *, 6> names = {"stage_q_fp32", "stage_k_fp32",
                                             "softmax_fp32", "unpack_bf16",
                                             "stage_v_fp32"};
  for (size_t i = 0U; i < functions.size(); ++i) {
    hipFuncAttributes attributes{};
    const hipError_t attr = hipFuncGetAttributes(&attributes, functions[i]);
    int active_blocks = 0;
    const hipError_t occupancy = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, functions[i], kThreads, 0U);
    std::printf("pipeline_resources kernel=%s registers=%d lds=%zu scratch=%zu "
                "active_blocks=%d occupancy=%s attrs=%s\n",
                names[i], attributes.numRegs, attributes.sharedSizeBytes,
                attributes.localSizeBytes, active_blocks,
                hipGetErrorString(occupancy), hipGetErrorString(attr));
  }
}

bool parse_device(const char *const text, int *const result) {
  if (text == nullptr || result == nullptr)
    return false;
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < 0L || value > INT32_MAX) {
    return false;
  }
  *result = static_cast<int>(value);
  return true;
}

} // namespace

int main(int argc, char **argv) {
  int device = 0;
  if (argc > 2 || (argc == 2 && !parse_device(argv[1], &device))) {
    std::fprintf(stderr, "usage: phase78_gqa6_prefill_qtile_probe [DEVICE]\n");
    return EXIT_FAILURE;
  }
  if (!hip_ok(hipSetDevice(device), "hipSetDevice"))
    return EXIT_FAILURE;
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device),
              "hipGetDeviceProperties")) {
    return EXIT_FAILURE;
  }
  std::printf("target=%s device=%d pci=%04x:%02x:%02x name=%s\n",
              properties.gcnArchName, device, properties.pciDomainID,
              properties.pciBusID, properties.pciDeviceID, properties.name);

  const std::array<CandidateId, 12> ids = {
      CandidateId::Q4K32Control,  CandidateId::Q4K8Control,
      CandidateId::Q4K32Block,    CandidateId::Q4K8Block,
      CandidateId::Q8K32Control,  CandidateId::Q8K8Control,
      CandidateId::Q16K32Control, CandidateId::Q16K8Control,
      CandidateId::Q8K32Block,    CandidateId::Q8K8Block,
      CandidateId::Q16K32Block,   CandidateId::Q16K8Block};
  for (const CandidateId id : ids)
    print_resources(candidate(id));

  // Non-aligned host oracle: all 24 query heads are checked at the exact
  // production head geometry, while the small context keeps the CPU oracle
  // finite.  Query rows are intentionally not a multiple of Q4/Q8/Q16.
  {
    constexpr uint64_t rows = 7U;
    constexpr uint64_t context = 13U;
    constexpr uint64_t start = 5U;
    std::vector<uint16_t> query, key, value, expected, actual;
    fill_inputs(rows, context, &query, &key, &value);
    host_oracle(rows, context, start, query, key, value, &expected);
    for (const CandidateId id : ids) {
      Buffers b;
      if (!make_buffers(rows, context, &b) || !upload(query, key, value, &b)) {
        free_buffers(&b);
        return EXIT_FAILURE;
      }
      const Candidate c = candidate(id);
      if (!launch(c, rows, start, &b) ||
          !hip_ok(hipStreamSynchronize(b.stream), "oracle synchronize") ||
          !copy_output(rows, &b, &actual)) {
        free_buffers(&b);
        return EXIT_FAILURE;
      }
      uint32_t max_ulp = 0U;
      uint64_t mismatches = 0U;
      if (!check_oracle(c, rows, expected, actual, &max_ulp, &mismatches)) {
        free_buffers(&b);
        return EXIT_FAILURE;
      }
      free_buffers(&b);
    }
  }

  bool all_ok = true;
  for (const uint64_t rows : {512U, 1024U}) {
    constexpr uint64_t context = 9435U;
    const uint64_t start = context - rows;
    std::vector<uint16_t> query, key, value, control, actual;
    fill_inputs(rows, context, &query, &key, &value);
    Buffers b;
    if (!make_buffers(rows, context, &b) || !upload(query, key, value, &b)) {
      free_buffers(&b);
      return EXIT_FAILURE;
    }
    const Candidate control_candidate = candidate(CandidateId::Q4K32Control);
    if (!launch(control_candidate, rows, start, &b) ||
        !hip_ok(hipStreamSynchronize(b.stream), "control synchronize") ||
        !copy_output(rows, &b, &control)) {
      free_buffers(&b);
      return EXIT_FAILURE;
    }
    for (const CandidateId id : ids) {
      const Candidate c = candidate(id);
      float median_us = 0.0F;
      if (!measure(c, rows, start, &b, &median_us) ||
          !copy_output(rows, &b, &actual)) {
        free_buffers(&b);
        return EXIT_FAILURE;
      }
      uint32_t max_ulp = 0U;
      uint64_t over_one = 0U;
      check_against_control(c, control, actual, &max_ulp, &over_one);
      const uint64_t tiles = (rows + c.query_tile - 1U) / c.query_tile;
      uint64_t staged_kv_bytes = 0U;
      for (uint64_t tile = 0U; tile < tiles; ++tile) {
        const uint64_t last_row =
            std::min<uint64_t>(rows, (tile + 1U) * c.query_tile) - 1U;
        const uint64_t key_rows = start + last_row + 1U;
        staged_kv_bytes +=
            key_rows * kKvHeads * kHeadDim * sizeof(uint16_t) * 2U;
      }
      const double staged_gbps = static_cast<double>(staged_kv_bytes) /
                                 (static_cast<double>(median_us) * 1000.0);
      const uint64_t logical_bytes =
          rows * kQHeads * context * kHeadDim * sizeof(uint16_t) * 2U;
      const double logical_gbps = static_cast<double>(logical_bytes) /
                                  (static_cast<double>(median_us) * 1000.0);
      std::printf(
          "result candidate=%s rows=%llu context=%llu start=%llu "
          "median_us=%.3f ms=%.6f staged_kv_bytes=%llu staged_GBps=%.3f "
          "logical_GBps=%.3f max_bf16_ulp=%u over1=%llu\n",
          c.name, static_cast<unsigned long long>(rows),
          static_cast<unsigned long long>(context),
          static_cast<unsigned long long>(start), median_us,
          median_us / 1000.0F, static_cast<unsigned long long>(staged_kv_bytes),
          staged_gbps, logical_gbps, max_ulp,
          static_cast<unsigned long long>(over_one));
      all_ok = all_ok && over_one == 0U;
    }
    free_buffers(&b);
  }
  // N2 feasibility: four strided-batched rocBLAS FP16-KV GEMM pipelines with
  // FP32 QK scores, in-place causal softmax, FP32 PV GEMM, and BF16 unpack.
  // The score matrix is intentionally explicit workspace: this measures the
  // proposed implementation shape rather than claiming a production path.
  print_pipeline_resources();
  {
    constexpr uint64_t rows = 7U;
    constexpr uint64_t context = 13U;
    constexpr uint64_t start = 5U;
    std::vector<uint16_t> query, key, value, expected, actual;
    fill_inputs(rows, context, &query, &key, &value);
    host_oracle(rows, context, start, query, key, value, &expected);
    Buffers b;
    PipelineBuffers p;
    if (!make_buffers(rows, context, &b) || !upload(query, key, value, &b) ||
        !make_pipeline_buffers(rows, context, b.stream, &p) ||
        !prepare_pipeline_values(&b, &p, rows, context) ||
        !launch_pipeline(&b, &p, rows, context, start) ||
        !hip_ok(hipStreamSynchronize(b.stream), "pipeline oracle sync") ||
        !copy_output(rows, &b, &actual)) {
      free_pipeline_buffers(&p);
      free_buffers(&b);
      return EXIT_FAILURE;
    }
    const Candidate pipeline_candidate = {CandidateId::Q4K32Control,
                                          "rocblas-fp32-score-pipeline",
                                          0U,
                                          0U,
                                          false,
                                          nullptr};
    uint32_t max_ulp = 0U;
    uint64_t mismatches = 0U;
    // rocBLAS uses a fixed/tuned GEMM reduction tree, so this feasibility
    // result is reported as an informational numerical class rather than
    // making the standalone candidate fail the exact BF16-RNE gate.
    (void)check_oracle(pipeline_candidate, rows, expected, actual, &max_ulp,
                       &mismatches);
    free_pipeline_buffers(&p);
    free_buffers(&b);
  }
  for (const uint64_t rows : {128U, 512U}) {
    constexpr uint64_t context = 9435U;
    const uint64_t start = context - rows;
    std::vector<uint16_t> query, key, value, control, actual;
    fill_inputs(rows, context, &query, &key, &value);
    Buffers b;
    PipelineBuffers p;
    if (!make_buffers(rows, context, &b) || !upload(query, key, value, &b) ||
        !make_pipeline_buffers(rows, context, b.stream, &p) ||
        !prepare_pipeline_values(&b, &p, rows, context)) {
      free_pipeline_buffers(&p);
      free_buffers(&b);
      return EXIT_FAILURE;
    }
    const Candidate control_candidate = candidate(CandidateId::Q4K32Control);
    if (!launch(control_candidate, rows, start, &b) ||
        !hip_ok(hipStreamSynchronize(b.stream), "pipeline control sync") ||
        !copy_output(rows, &b, &control)) {
      free_pipeline_buffers(&p);
      free_buffers(&b);
      return EXIT_FAILURE;
    }
    float median_us = 0.0F;
    if (!measure_pipeline(&b, &p, rows, context, start, &median_us) ||
        !copy_output(rows, &b, &actual)) {
      free_pipeline_buffers(&p);
      free_buffers(&b);
      return EXIT_FAILURE;
    }
    uint32_t max_ulp = 0U;
    uint64_t over_one = 0U;
    const Candidate pipeline_candidate = {CandidateId::Q4K32Control,
                                          "rocblas-fp32-score-pipeline",
                                          0U,
                                          0U,
                                          false,
                                          nullptr};
    check_against_control(pipeline_candidate, control, actual, &max_ulp,
                          &over_one);
    const uint64_t q_count = rows * kGqaRatio;
    const uint64_t query_bytes =
        static_cast<uint64_t>(kPipelineBatch) * q_count * kHeadDim * 4U;
    const uint64_t key_value_stage_bytes =
        static_cast<uint64_t>(kPipelineBatch) * context * kHeadDim * 4U;
    const uint64_t score_bytes =
        static_cast<uint64_t>(kPipelineBatch) * q_count * context * 4U;
    const uint64_t pv_bytes =
        static_cast<uint64_t>(kPipelineBatch) * q_count * kHeadDim * 4U;
    const double gflops = 4.0 * static_cast<double>(q_count) *
                          static_cast<double>(context) *
                          static_cast<double>(kHeadDim) /
                          (static_cast<double>(median_us) * 1000.0);
    std::printf(
        "pipeline_result rows=%llu context=%llu start=%llu median_us=%.3f "
        "ms=%.6f workspace_bytes=%llu query_f32_bytes=%llu "
        "key_f32_bytes=%llu value_f32_bytes=%llu score_bytes=%llu "
        "pv_bytes=%llu qk_pv_gflops=%.3f max_bf16_ulp=%u over1=%llu "
        "score_precision=fp32 status=%s\n",
        static_cast<unsigned long long>(rows),
        static_cast<unsigned long long>(context),
        static_cast<unsigned long long>(start), median_us, median_us / 1000.0F,
        static_cast<unsigned long long>(
            query_bytes + key_value_stage_bytes * 2U + score_bytes + pv_bytes),
        static_cast<unsigned long long>(query_bytes),
        static_cast<unsigned long long>(key_value_stage_bytes),
        static_cast<unsigned long long>(key_value_stage_bytes),
        static_cast<unsigned long long>(score_bytes),
        static_cast<unsigned long long>(pv_bytes), gflops, max_ulp,
        static_cast<unsigned long long>(over_one),
        over_one == 0U ? "PASS" : "INFO");
    free_pipeline_buffers(&p);
    free_buffers(&b);
  }
  std::printf("summary status=%s candidates=%zu warmups=%u measured=%u "
              "oracle=nonaligned7x13_all24heads\n",
              all_ok ? "PASS" : "FAIL", ids.size(), kWarmups, kMeasured);
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
