// Phase 78 standalone gfx1201 GQA6 fused-attention feasibility probe.
//
// This file is deliberately outside the production build.  The control is
// the current scalar Q4/K8 FP16-KV route.  The candidate owns eight rows (48
// Q heads) per KV head, computes QK with rocWMMA FP16xFP16->FP32, performs an
// in-LDS blockwise online softmax, and computes PV with rocWMMA.  Only the
// running BF16 output is global; no full QK score workspace is allocated.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

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
constexpr uint32_t kWaves = 8U;
constexpr uint32_t kQHeads = 24U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kGqa = 6U;
constexpr uint32_t kD = 256U;
constexpr uint32_t kWarmups = 3U;
constexpr uint32_t kMeasured = 10U;

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
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U))
    ++upper;
  return static_cast<uint16_t>(upper);
}

// Existing scalar Q4/K8 control. One block owns four query rows and one KV
// head; each wave processes three of the 24 logical query/head pairs.
__global__ __launch_bounds__(kThreads, 1) void gqa6_q4k8_control_kernel(
    const uint16_t *const query, const uint16_t *const key,
    const uint16_t *const value, uint16_t *const output, const uint32_t rows,
    const uint64_t start_position) {
  constexpr uint32_t qtile = 4U;
  constexpr uint32_t ktile = 8U;
  constexpr uint32_t pairs_per_lane = kD / kWave / 2U;
  constexpr uint32_t q_per_wave = qtile * kGqa / kWaves;
  const uint64_t flat = static_cast<uint64_t>(blockIdx.x);
  const uint64_t tile = flat / kKvHeads;
  const uint32_t kv_head = static_cast<uint32_t>(flat % kKvHeads);
  const uint64_t first_row = tile * qtile;
  if (first_row >= rows)
    return;
  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (kWave - 1U);
  const uint32_t wave = thread / kWave;
  const uint32_t first_head = kv_head * kGqa;
  __shared__ uint16_t key_tile[ktile][kD];
  __shared__ uint16_t value_tile[ktile][kD];
  float2 qv[q_per_wave][pairs_per_lane];
  float2 acc[q_per_wave][pairs_per_lane] = {};
  float running_max[q_per_wave];
  float running_den[q_per_wave];
#pragma unroll
  for (uint32_t item = 0U; item < q_per_wave; ++item) {
    running_max[item] = -INFINITY;
    running_den[item] = 0.0F;
    const uint32_t logical = wave * q_per_wave + item;
    const uint64_t row = first_row + logical / kGqa;
    const uint32_t head = first_head + logical % kGqa;
    const uint64_t safe_row = row < rows ? row : rows - 1U;
    const uint16_t *const qr = query + (safe_row * kQHeads + head) * kD;
#pragma unroll
    for (uint32_t pair = 0U; pair < pairs_per_lane; ++pair) {
      const uint32_t d = lane * 2U + pair * kWave * 2U;
      qv[item][pair] = row < rows ? make_float2(bf16_to_float(qr[d]),
                                                bf16_to_float(qr[d + 1U]))
                                  : make_float2(0.0F, 0.0F);
    }
  }
  const uint64_t row_count =
      rows - first_row < qtile ? rows - first_row : qtile;
  const uint64_t last_row = first_row + row_count - 1U;
  const uint64_t last_position = start_position + last_row;
  for (uint64_t key_begin = 0U; key_begin <= last_position;
       key_begin += ktile) {
    const uint32_t count = static_cast<uint32_t>(
        last_position - key_begin + 1U < ktile ? last_position - key_begin + 1U
                                               : ktile);
    for (uint32_t index = thread; index < ktile * kD; index += kThreads) {
      const uint32_t ki = index / kD;
      const uint32_t d = index % kD;
      if (ki < count) {
        const uint64_t kv = (key_begin + ki) * kKvHeads + kv_head;
        key_tile[ki][d] = key[kv * kD + d];
        value_tile[ki][d] = value[kv * kD + d];
      }
    }
    __syncthreads();
    for (uint32_t ki = 0U; ki < count; ++ki) {
      const uint64_t position = key_begin + ki;
      for (uint32_t item = 0U; item < q_per_wave; ++item) {
        const uint32_t logical = wave * q_per_wave + item;
        const uint64_t row = first_row + logical / kGqa;
        const bool active = row < rows && position <= start_position + row;
        float partial = 0.0F;
#pragma unroll
        for (uint32_t pair = 0U; pair < pairs_per_lane; ++pair) {
          const uint32_t d = lane * 2U + pair * kWave * 2U;
          const float2 q = qv[item][pair];
          partial += active ? q.x * fp16_to_float(key_tile[ki][d]) +
                                  q.y * fp16_to_float(key_tile[ki][d + 1U])
                            : 0.0F;
        }
        for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U)
          partial += __shfl_down(partial, offset, kWave);
        float rescale = 1.0F;
        float contribution = 0.0F;
        float next = running_max[item];
        if (lane == 0U && active) {
          const float score = partial * rsqrtf(static_cast<float>(kD));
          next = fmaxf(running_max[item], score);
          rescale = expf(running_max[item] - next);
          contribution = expf(score - next);
        }
        rescale = __shfl(rescale, 0U, kWave);
        contribution = __shfl(contribution, 0U, kWave);
        next = __shfl(next, 0U, kWave);
        running_den[item] = running_den[item] * rescale + contribution;
        running_max[item] = next;
#pragma unroll
        for (uint32_t pair = 0U; pair < pairs_per_lane; ++pair) {
          const uint32_t d = lane * 2U + pair * kWave * 2U;
          if (active) {
            acc[item][pair].x = acc[item][pair].x * rescale +
                                contribution * fp16_to_float(value_tile[ki][d]);
            acc[item][pair].y =
                acc[item][pair].y * rescale +
                contribution * fp16_to_float(value_tile[ki][d + 1U]);
          }
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t item = 0U; item < q_per_wave; ++item) {
    const uint32_t logical = wave * q_per_wave + item;
    const uint64_t row = first_row + logical / kGqa;
    const uint32_t head = first_head + logical % kGqa;
    if (row < rows) {
      uint16_t *const out = output + (row * kQHeads + head) * kD;
#pragma unroll
      for (uint32_t pair = 0U; pair < pairs_per_lane; ++pair) {
        const uint32_t d = lane * 2U + pair * kWave * 2U;
        out[d] = bf16_rne(acc[item][pair].x / running_den[item]);
        out[d + 1U] = bf16_rne(acc[item][pair].y / running_den[item]);
      }
    }
  }
}

// Fused candidate. QTile=8 gives 48 logical query/head rows per block. Three
// waves each own one 16x16 WMMA score tile; the remaining waves participate in
// cooperative Q/K/V staging. K/V are transposed in LDS for matrix-B loads.
__global__ __launch_bounds__(kThreads, 1) void gqa6_fused_wmma_kernel(
    const uint16_t *const query, const uint16_t *const key,
    const uint16_t *const value, float *const output_f32, const uint32_t rows,
    const uint64_t context, const uint64_t start_position) {
#if defined(__gfx1201__)
  constexpr uint32_t qtile = 8U;
  constexpr uint32_t ktile = 16U;
  constexpr uint32_t q_rows = qtile * kGqa;
  constexpr uint32_t score_m = 16U;
  constexpr uint32_t score_n = 16U;
  constexpr uint32_t frag_k = 16U;
  constexpr uint32_t q_waves = q_rows / score_m;
  static_assert(q_waves == 3U);
  using AFragment =
      rocwmma::fragment<rocwmma::matrix_a, score_m, score_n, frag_k,
                        rocwmma::float16_t, rocwmma::row_major>;
  using QkBFragment =
      rocwmma::fragment<rocwmma::matrix_b, score_m, score_n, frag_k,
                        rocwmma::float16_t, rocwmma::col_major>;
  // PV has B[k=token,n=dimension] in token-major LDS, so its natural
  // row-major layout is the transpose of the QK operand's column-major view.
  using PvBFragment =
      rocwmma::fragment<rocwmma::matrix_b, score_m, score_n, frag_k,
                        rocwmma::float16_t, rocwmma::row_major>;
  using AccFragment =
      rocwmma::fragment<rocwmma::accumulator, score_m, score_n, frag_k, float>;
  const uint64_t flat = static_cast<uint64_t>(blockIdx.x);
  const uint64_t tile = flat / kKvHeads;
  const uint32_t kv_head = static_cast<uint32_t>(flat % kKvHeads);
  const uint64_t first_row = tile * qtile;
  if (first_row >= rows)
    return;
  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (kWave - 1U);
  const uint32_t wave = thread / kWave;
  const uint32_t first_head = kv_head * kGqa;
  __shared__ __align__(4) rocwmma::float16_t q_tile[q_rows][kD];
  // Matrix-B is column-major: one KV token (one matrix column) is contiguous
  // in D, exactly as in the production WMMA ingress tiles.
  __shared__ __align__(4) rocwmma::float16_t key_t[ktile][kD];
  __shared__ __align__(4) rocwmma::float16_t value_t[ktile][kD];
  __shared__ float score_tile[q_rows][ktile];
  __shared__ __align__(4) rocwmma::float16_t probability_tile[q_rows][ktile];
  __shared__ float tile_rescale[q_rows];

  // Load all 48 query/head rows once. Q is already BF16 in the model-facing
  // contract and is converted to FP16 only for the WMMA operand.
  for (uint32_t index = thread; index < q_rows * kD; index += kThreads) {
    const uint32_t qindex = index / kD;
    const uint32_t d = index % kD;
    const uint64_t row = first_row + qindex / kGqa;
    const uint32_t head = first_head + qindex % kGqa;
    q_tile[qindex][d] =
        row < rows ? static_cast<rocwmma::float16_t>(
                         bf16_to_float(query[(row * kQHeads + head) * kD + d]))
                   : static_cast<rocwmma::float16_t>(0.0F);
  }
  __syncthreads();

  const uint64_t row_count =
      rows - first_row < qtile ? rows - first_row : qtile;
  const uint64_t last_row = first_row + row_count - 1U;
  const uint64_t raw_last_position = start_position + last_row;
  const uint64_t last_position =
      raw_last_position < context ? raw_last_position : context - 1U;
  float running_max[score_m];
  float running_den[score_m];
#pragma unroll
  for (uint32_t row = 0U; row < score_m; ++row) {
    running_max[row] = -INFINITY;
    running_den[row] = 0.0F;
  }

  for (uint64_t key_begin = 0U; key_begin <= last_position;
       key_begin += ktile) {
    const uint32_t key_count = static_cast<uint32_t>(
        last_position - key_begin + 1U < ktile ? last_position - key_begin + 1U
                                               : ktile);
    for (uint32_t index = thread; index < ktile * kD; index += kThreads) {
      const uint32_t key_index = index / kD;
      const uint32_t d = index % kD;
      if (key_index < key_count) {
        const uint64_t kv = (key_begin + key_index) * kKvHeads + kv_head;
        key_t[key_index][d] =
            static_cast<rocwmma::float16_t>(fp16_to_float(key[kv * kD + d]));
        value_t[key_index][d] =
            static_cast<rocwmma::float16_t>(fp16_to_float(value[kv * kD + d]));
      } else {
        key_t[key_index][d] = static_cast<rocwmma::float16_t>(0.0F);
        value_t[key_index][d] = static_cast<rocwmma::float16_t>(0.0F);
      }
    }
    __syncthreads();

    // QK: each of the first three waves computes one 16x16 score tile. The
    // accumulator is FP32 and is stored only in block-local score LDS.
    if (wave < q_waves) {
      AccFragment scores;
      rocwmma::fill_fragment(scores, 0.0F);
      const uint32_t mbase = wave * score_m;
      for (uint32_t kbase = 0U; kbase < kD; kbase += frag_k) {
        AFragment af;
        QkBFragment bf;
        rocwmma::load_matrix_sync(af, q_tile[mbase] + kbase, kD);
        rocwmma::load_matrix_sync(bf, key_t[0] + kbase, kD);
        rocwmma::mma_sync(scores, af, bf, scores);
      }
      rocwmma::store_matrix_sync(
          score_tile[mbase],
          rocwmma::apply_data_layout<rocwmma::row_major>(scores), ktile);
    }
    __syncthreads();

    // Blockwise online softmax. Each active wave handles its 16 score rows;
    // lane<16 owns one key in the current tile.  The probability written to
    // LDS is already multiplied by the online tile scale, so PV needs no
    // second softmax pass.
    if (wave < q_waves) {
      const uint32_t mbase = wave * score_m;
      for (uint32_t row = 0U; row < score_m; ++row) {
        const uint32_t qindex = mbase + row;
        const uint64_t global_row = first_row + qindex / kGqa;
        const bool row_active = global_row < rows;
        float tile_max = -INFINITY;
        if (lane < ktile) {
          const uint64_t position = key_begin + lane;
          const bool active = row_active && lane < key_count &&
                              position <= start_position + global_row;
          tile_max =
              active ? score_tile[qindex][lane] * rsqrtf(static_cast<float>(kD))
                     : -INFINITY;
        }
        for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U)
          tile_max = fmaxf(tile_max, __shfl_down(tile_max, offset, kWave));
        tile_max = __shfl(tile_max, 0U, kWave);
        float probability = 0.0F;
        if (lane < ktile && tile_max != -INFINITY) {
          const uint64_t position = key_begin + lane;
          const bool active = row_active && lane < key_count &&
                              position <= start_position + global_row;
          if (active) {
            probability =
                expf(score_tile[qindex][lane] * rsqrtf(static_cast<float>(kD)) -
                     tile_max);
          }
        }
        float tile_den = probability;
        for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U)
          tile_den += __shfl_down(tile_den, offset, kWave);
        tile_den = __shfl(tile_den, 0U, kWave);
        float rescale = 1.0F;
        float tile_scale = 0.0F;
        float next_max = running_max[row];
        if (lane == 0U && row_active && tile_den > 0.0F) {
          next_max = fmaxf(running_max[row], tile_max);
          rescale = expf(running_max[row] - next_max);
          tile_scale = expf(tile_max - next_max);
          running_den[row] = running_den[row] * rescale + tile_den * tile_scale;
          running_max[row] = next_max;
        }
        rescale = __shfl(rescale, 0U, kWave);
        tile_scale = __shfl(tile_scale, 0U, kWave);
        next_max = __shfl(next_max, 0U, kWave);
        running_max[row] = next_max;
        running_den[row] = __shfl(running_den[row], 0U, kWave);
        if (lane < ktile) {
          probability_tile[qindex][lane] =
              static_cast<rocwmma::float16_t>(probability * tile_scale);
        }
        if (lane == 0U)
          tile_rescale[qindex] = rescale;
      }
    }
    __syncthreads();

    // PV: all three active waves compute 16x16 output fragments for each D
    // tile. The previous running output is global only to keep LDS below the
    // 64 KiB limit; no score matrix leaves the block.
    if (wave < q_waves) {
      const uint32_t mbase = wave * score_m;
      const bool final_key_tile = key_begin + ktile > last_position;
      for (uint32_t dbase = 0U; dbase < kD; dbase += score_n) {
        AFragment af;
        PvBFragment bf;
        AccFragment contribution;
        rocwmma::load_matrix_sync(af, probability_tile[mbase], ktile);
        rocwmma::load_matrix_sync(bf, value_t[0] + dbase, kD);
        rocwmma::fill_fragment(contribution, 0.0F);
        rocwmma::mma_sync(contribution, af, bf, contribution);
        const auto contribution_rm =
            rocwmma::apply_data_layout<rocwmma::row_major>(contribution);
#pragma unroll
        for (uint32_t slot = 0U; slot < (score_m * score_n) / kWave; ++slot) {
          const uint32_t local_row =
              (lane / score_n) * ((score_m * score_n) / kWave) + slot;
          const uint32_t local_col = lane % score_n;
          const uint32_t qindex = mbase + local_row;
          const uint64_t global_row = first_row + qindex / kGqa;
          const uint32_t head = first_head + qindex % kGqa;
          if (global_row < rows) {
            float prior = key_begin == 0U
                              ? 0.0F
                              : output_f32[(global_row * kQHeads + head) * kD +
                                           dbase + local_col];
            const float numerator =
                prior * tile_rescale[qindex] + contribution_rm[slot];
            output_f32[(global_row * kQHeads + head) * kD + dbase + local_col] =
                final_key_tile ? numerator / running_den[local_row] : numerator;
          }
        }
      }
    }
    __syncthreads();
  }
#else
  (void)query;
  (void)key;
  (void)value;
  (void)output_f32;
  (void)rows;
  (void)context;
  (void)start_position;
#endif
}

// Persistent FlashAttention-shaped candidate.  A block owns two query rows
// for one KV head (12 logical Q/head rows, padded to 16).  Wave 0 computes
// the QK tile and online softmax; all eight waves split the 256-D PV result
// into two 16-D WMMA tiles each.  The 16 FP32 output values per lane remain in
// registers across the complete context, so no output accumulator traffic is
// emitted for intermediate K/V tiles.
__global__ __launch_bounds__(kThreads, 1) void gqa6_persistent_wmma_kernel(
    const uint16_t *const query, const uint16_t *const key,
    const uint16_t *const value, float *const output_f32, const uint32_t rows,
    const uint64_t context, const uint64_t start_position) {
#if defined(__gfx1201__)
  constexpr uint32_t qtile_rows = 2U;
  constexpr uint32_t q_rows = qtile_rows * kGqa;
  constexpr uint32_t q_rows_padded = 16U;
  constexpr uint32_t ktile = 16U;
  constexpr uint32_t tile_m = 16U;
  constexpr uint32_t tile_n = 16U;
  constexpr uint32_t frag_k = 16U;
  constexpr uint32_t d_tiles_per_wave = 2U;
  static_assert(q_rows == 12U && q_rows_padded == tile_m);
  using AFragment = rocwmma::fragment<rocwmma::matrix_a, tile_m, tile_n, frag_k,
                                      rocwmma::float16_t, rocwmma::row_major>;
  using QkBFragment =
      rocwmma::fragment<rocwmma::matrix_b, tile_m, tile_n, frag_k,
                        rocwmma::float16_t, rocwmma::col_major>;
  using PvBFragment =
      rocwmma::fragment<rocwmma::matrix_b, tile_m, tile_n, frag_k,
                        rocwmma::float16_t, rocwmma::row_major>;
  using AccFragment =
      rocwmma::fragment<rocwmma::accumulator, tile_m, tile_n, frag_k, float>;
  const uint64_t flat = static_cast<uint64_t>(blockIdx.x);
  const uint64_t tile = flat / kKvHeads;
  const uint32_t kv_head = static_cast<uint32_t>(flat % kKvHeads);
  const uint64_t first_row = tile * qtile_rows;
  if (first_row >= rows)
    return;
  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (kWave - 1U);
  const uint32_t wave = thread / kWave;
  const uint32_t first_head = kv_head * kGqa;

  // 8 KiB each for Q, K, and V, plus 1 KiB FP32 score and 512 B FP16
  // probability.  The running state is shared because only wave 0 owns QK.
  __shared__ __align__(4) rocwmma::float16_t q_tile[q_rows_padded][kD];
  __shared__ __align__(4) rocwmma::float16_t key_t[ktile][kD];
  __shared__ __align__(4) rocwmma::float16_t value_t[ktile][kD];
  __shared__ float score_tile[q_rows_padded][ktile];
  __shared__ __align__(4)
      rocwmma::float16_t probability_tile[q_rows_padded][ktile];
  __shared__ float running_max[q_rows_padded];
  __shared__ float running_den[q_rows_padded];
  __shared__ float tile_rescale[q_rows_padded];

  for (uint32_t index = thread; index < q_rows_padded * kD; index += kThreads) {
    const uint32_t qindex = index / kD;
    const uint32_t d = index % kD;
    const uint64_t row = first_row + qindex / kGqa;
    const uint32_t head = first_head + qindex % kGqa;
    q_tile[qindex][d] = qindex < q_rows && row < rows
                            ? static_cast<rocwmma::float16_t>(bf16_to_float(
                                  query[(row * kQHeads + head) * kD + d]))
                            : static_cast<rocwmma::float16_t>(0.0F);
  }
  if (thread < q_rows_padded) {
    running_max[thread] = -INFINITY;
    running_den[thread] = 0.0F;
    tile_rescale[thread] = 1.0F;
  }
  __syncthreads();

  const uint64_t row_count =
      rows - first_row < qtile_rows ? rows - first_row : qtile_rows;
  const uint64_t last_row = first_row + row_count - 1U;
  const uint64_t raw_last_position = start_position + last_row;
  const uint64_t last_position =
      raw_last_position < context ? raw_last_position : context - 1U;
  float accum[d_tiles_per_wave][(tile_m * tile_n) / kWave] = {};

  for (uint64_t key_begin = 0U; key_begin <= last_position;
       key_begin += ktile) {
    const uint32_t key_count = static_cast<uint32_t>(
        last_position - key_begin + 1U < ktile ? last_position - key_begin + 1U
                                               : ktile);
    for (uint32_t index = thread; index < ktile * kD; index += kThreads) {
      const uint32_t key_index = index / kD;
      const uint32_t d = index % kD;
      if (key_index < key_count) {
        const uint64_t kv = (key_begin + key_index) * kKvHeads + kv_head;
        key_t[key_index][d] =
            static_cast<rocwmma::float16_t>(fp16_to_float(key[kv * kD + d]));
        value_t[key_index][d] =
            static_cast<rocwmma::float16_t>(fp16_to_float(value[kv * kD + d]));
      } else {
        key_t[key_index][d] = static_cast<rocwmma::float16_t>(0.0F);
        value_t[key_index][d] = static_cast<rocwmma::float16_t>(0.0F);
      }
    }
    __syncthreads();

    // One wave owns the complete 16x16 QK tile.  This is the key difference
    // from the previous candidate: PV waves do not redundantly recompute QK.
    if (wave == 0U) {
      AccFragment scores;
      rocwmma::fill_fragment(scores, 0.0F);
      for (uint32_t kbase = 0U; kbase < kD; kbase += frag_k) {
        AFragment af;
        QkBFragment bf;
        rocwmma::load_matrix_sync(af, q_tile[0] + kbase, kD);
        rocwmma::load_matrix_sync(bf, key_t[0] + kbase, kD);
        rocwmma::mma_sync(scores, af, bf, scores);
      }
      rocwmma::store_matrix_sync(
          score_tile[0], rocwmma::apply_data_layout<rocwmma::row_major>(scores),
          ktile);

      for (uint32_t qindex = 0U; qindex < q_rows_padded; ++qindex) {
        const uint64_t global_row = first_row + qindex / kGqa;
        const bool row_active = qindex < q_rows && global_row < rows;
        float tile_max = -INFINITY;
        if (lane < ktile) {
          const uint64_t position = key_begin + lane;
          const bool active = row_active && lane < key_count &&
                              position <= start_position + global_row;
          tile_max =
              active ? score_tile[qindex][lane] * rsqrtf(static_cast<float>(kD))
                     : -INFINITY;
        }
        for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U)
          tile_max = fmaxf(tile_max, __shfl_down(tile_max, offset, kWave));
        tile_max = __shfl(tile_max, 0U, kWave);
        float probability = 0.0F;
        if (lane < ktile && tile_max != -INFINITY) {
          const uint64_t position = key_begin + lane;
          const bool active = row_active && lane < key_count &&
                              position <= start_position + global_row;
          if (active) {
            probability =
                expf(score_tile[qindex][lane] * rsqrtf(static_cast<float>(kD)) -
                     tile_max);
          }
        }
        float tile_den = probability;
        for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U)
          tile_den += __shfl_down(tile_den, offset, kWave);
        tile_den = __shfl(tile_den, 0U, kWave);
        float rescale = 1.0F;
        float tile_scale = 0.0F;
        float next_max = running_max[qindex];
        if (lane == 0U && row_active && tile_den > 0.0F) {
          next_max = fmaxf(running_max[qindex], tile_max);
          rescale = expf(running_max[qindex] - next_max);
          tile_scale = expf(tile_max - next_max);
          running_den[qindex] =
              running_den[qindex] * rescale + tile_den * tile_scale;
          running_max[qindex] = next_max;
        }
        rescale = __shfl(rescale, 0U, kWave);
        tile_scale = __shfl(tile_scale, 0U, kWave);
        if (lane < ktile) {
          probability_tile[qindex][lane] =
              static_cast<rocwmma::float16_t>(probability * tile_scale);
        }
        if (lane == 0U)
          tile_rescale[qindex] = rescale;
      }
    }
    __syncthreads();

    // Eight waves cover 8*2*16 = 256 output dimensions.  Each wave keeps its
    // two 16-D WMMA accumulator tiles in registers over all K/V tiles.
    for (uint32_t d_tile = 0U; d_tile < d_tiles_per_wave; ++d_tile) {
      const uint32_t dbase = wave * d_tiles_per_wave * tile_n + d_tile * tile_n;
      AFragment af;
      PvBFragment bf;
      AccFragment contribution;
      rocwmma::load_matrix_sync(af, probability_tile[0], ktile);
      rocwmma::load_matrix_sync(bf, value_t[0] + dbase, kD);
      rocwmma::fill_fragment(contribution, 0.0F);
      rocwmma::mma_sync(contribution, af, bf, contribution);
      const auto contribution_rm =
          rocwmma::apply_data_layout<rocwmma::row_major>(contribution);
#pragma unroll
      for (uint32_t slot = 0U; slot < (tile_m * tile_n) / kWave; ++slot)
        accum[d_tile][slot] += contribution_rm[slot];
    }
    __syncthreads();
  }

  for (uint32_t d_tile = 0U; d_tile < d_tiles_per_wave; ++d_tile) {
    const uint32_t dbase = wave * d_tiles_per_wave * tile_n + d_tile * tile_n;
#pragma unroll
    for (uint32_t slot = 0U; slot < (tile_m * tile_n) / kWave; ++slot) {
      const uint32_t local_row =
          (lane / tile_n) * ((tile_m * tile_n) / kWave) + slot;
      const uint32_t local_col = lane % tile_n;
      const uint32_t qindex = local_row;
      const uint64_t global_row = first_row + qindex / kGqa;
      const uint32_t head = first_head + qindex % kGqa;
      if (qindex < q_rows && global_row < rows) {
        const uint64_t output_index =
            (global_row * kQHeads + head) * kD + dbase + local_col;
        output_f32[output_index] = accum[d_tile][slot] / running_den[qindex];
      }
    }
  }
#else
  (void)query;
  (void)key;
  (void)value;
  (void)output_f32;
  (void)rows;
  (void)context;
  (void)start_position;
#endif
}

__global__ void unpack_bf16_kernel(const float *const input,
                                   uint16_t *const output,
                                   const uint64_t values) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < values)
    output[index] = bf16_rne(input[index]);
}

struct Buffers final {
  uint16_t *query = nullptr;
  uint16_t *key = nullptr;
  uint16_t *value = nullptr;
  uint16_t *output = nullptr;
  float *output_f32 = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
};

void free_buffers(Buffers *const b) {
  if (b == nullptr)
    return;
  if (b->stop != nullptr)
    (void)hipEventDestroy(b->stop);
  if (b->start != nullptr)
    (void)hipEventDestroy(b->start);
  if (b->stream != nullptr)
    (void)hipStreamDestroy(b->stream);
  if (b->output_f32 != nullptr)
    (void)hipFree(b->output_f32);
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

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "hip error operation=%s status=%s\n", operation,
               hipGetErrorString(status));
  return false;
}

bool make_buffers(const uint64_t rows, const uint64_t context,
                  Buffers *const b) {
  const size_t qbytes = rows * kQHeads * kD * sizeof(uint16_t);
  const size_t kvbytes = context * kKvHeads * kD * sizeof(uint16_t);
  const size_t fbytes = rows * kQHeads * kD * sizeof(float);
  return hip_ok(hipMalloc(reinterpret_cast<void **>(&b->query), qbytes),
                "hipMalloc query") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->key), kvbytes),
                "hipMalloc key") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->value), kvbytes),
                "hipMalloc value") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->output), qbytes),
                "hipMalloc output") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->output_f32), fbytes),
                "hipMalloc output f32") &&
         hip_ok(hipStreamCreate(&b->stream), "hipStreamCreate") &&
         hip_ok(hipEventCreate(&b->start), "hipEventCreate start") &&
         hip_ok(hipEventCreate(&b->stop), "hipEventCreate stop");
}

uint16_t host_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000U) == 0x7f800000U) {
    if ((bits & 0x007fffffU) != 0U)
      return static_cast<uint16_t>(((bits >> 16U) & 0x8000U) | 0x7fc0U |
                                   ((bits >> 16U) & 0x003fU));
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & 0xffffU;
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U))
    ++upper;
  return static_cast<uint16_t>(upper);
}

uint16_t host_fp16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16U) & 0x8000U;
  const uint32_t exponent = (bits >> 23U) & 0xffU;
  const uint32_t mantissa = bits & 0x7fffffU;
  if (exponent == 0xffU)
    return static_cast<uint16_t>(sign | 0x7c00U |
                                 (mantissa != 0U ? 0x0200U : 0U));
  int32_t half_exp = static_cast<int32_t>(exponent) - 127 + 15;
  if (half_exp <= 0) {
    if (half_exp < -10)
      return static_cast<uint16_t>(sign);
    const uint32_t mant = mantissa | 0x800000U;
    const uint32_t shift = static_cast<uint32_t>(14 - half_exp);
    uint32_t rounded = mant >> shift;
    const uint32_t remainder = mant & ((1U << shift) - 1U);
    if (remainder > (1U << (shift - 1U)) ||
        (remainder == (1U << (shift - 1U)) && (rounded & 1U) != 0U))
      ++rounded;
    return static_cast<uint16_t>(sign | rounded);
  }
  if (half_exp >= 31)
    return static_cast<uint16_t>(sign | 0x7c00U);
  uint32_t rounded = mantissa >> 13U;
  const uint32_t remainder = mantissa & 0x1fffU;
  if (remainder > 0x1000U || (remainder == 0x1000U && (rounded & 1U) != 0U)) {
    ++rounded;
    if (rounded == 0x400U) {
      rounded = 0U;
      ++half_exp;
      if (half_exp >= 31)
        return static_cast<uint16_t>(sign | 0x7c00U);
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exp) << 10U) |
                               rounded);
}

float host_bf16(const uint16_t bits) {
  uint32_t expanded = static_cast<uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &expanded, sizeof(value));
  return value;
}

float host_fp16(const uint16_t bits) {
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

void fill_inputs(const uint64_t rows, const uint64_t context,
                 std::vector<uint16_t> *const query,
                 std::vector<uint16_t> *const key,
                 std::vector<uint16_t> *const value) {
  query->resize(rows * kQHeads * kD);
  key->resize(context * kKvHeads * kD);
  value->resize(context * kKvHeads * kD);
  for (uint64_t row = 0U; row < rows; ++row) {
    for (uint32_t head = 0U; head < kQHeads; ++head) {
      for (uint32_t d = 0U; d < kD; ++d) {
        const float x =
            0.45F * std::sin(static_cast<float>(
                                 (row * 7U + head * 13U + d * 3U) % 97U) *
                             0.071F);
        (*query)[(row * kQHeads + head) * kD + d] = host_bf16_rne(x);
      }
    }
  }
  for (uint64_t row = 0U; row < context; ++row) {
    for (uint32_t head = 0U; head < kKvHeads; ++head) {
      for (uint32_t d = 0U; d < kD; ++d) {
        const float k =
            0.65F * std::cos(static_cast<float>(
                                 (row * 11U + head * 5U + d * 17U) % 131U) *
                             0.053F);
        const float v =
            0.55F * std::sin(static_cast<float>(
                                 (row * 3U + head * 19U + d * 7U) % 127U) *
                             0.067F);
        (*key)[(row * kKvHeads + head) * kD + d] = host_fp16_rne(k);
        (*value)[(row * kKvHeads + head) * kD + d] = host_fp16_rne(v);
      }
    }
  }
}

bool upload(const std::vector<uint16_t> &query,
            const std::vector<uint16_t> &key,
            const std::vector<uint16_t> &value, Buffers *const b) {
  return hip_ok(hipMemcpy(b->query, query.data(), query.size() * 2U,
                          hipMemcpyHostToDevice),
                "copy query") &&
         hip_ok(hipMemcpy(b->key, key.data(), key.size() * 2U,
                          hipMemcpyHostToDevice),
                "copy key") &&
         hip_ok(hipMemcpy(b->value, value.data(), value.size() * 2U,
                          hipMemcpyHostToDevice),
                "copy value");
}

uint32_t ulp_distance(const uint16_t a, const uint16_t b) {
  if ((a & 0x7fffU) == 0U && (b & 0x7fffU) == 0U)
    return 0U;
  const int32_t ai = (a & 0x8000U) ? 0x8000 - (a & 0x7fffU) : 0x8000 + a;
  const int32_t bi = (b & 0x8000U) ? 0x8000 - (b & 0x7fffU) : 0x8000 + b;
  return static_cast<uint32_t>(std::abs(ai - bi));
}

bool host_oracle(const uint64_t rows, const uint64_t context,
                 const uint64_t start, const std::vector<uint16_t> &query,
                 const std::vector<uint16_t> &key,
                 const std::vector<uint16_t> &value,
                 std::vector<uint16_t> *const expected) {
  expected->assign(rows * kQHeads * kD, 0U);
  for (uint64_t row = 0U; row < rows; ++row) {
    const uint64_t limit = std::min<uint64_t>(context - 1U, start + row);
    for (uint32_t head = 0U; head < kQHeads; ++head) {
      const uint32_t kv = head / kGqa;
      std::vector<float> score(limit + 1U);
      float maximum = -INFINITY;
      const uint16_t *const qr = query.data() + (row * kQHeads + head) * kD;
      for (uint64_t p = 0U; p <= limit; ++p) {
        const uint16_t *const kr = key.data() + (p * kKvHeads + kv) * kD;
        float dot = 0.0F;
        for (uint32_t d = 0U; d < kD; ++d)
          dot = std::fmaf(host_bf16(qr[d]), host_fp16(kr[d]), dot);
        score[p] = dot * (1.0F / std::sqrt(static_cast<float>(kD)));
        maximum = std::max(maximum, score[p]);
      }
      float denom = 0.0F;
      for (float &s : score) {
        s = std::exp(s - maximum);
        denom += s;
      }
      uint16_t *const out = expected->data() + (row * kQHeads + head) * kD;
      for (uint32_t d = 0U; d < kD; ++d) {
        float result = 0.0F;
        for (uint64_t p = 0U; p <= limit; ++p) {
          result += (score[p] / denom) *
                    host_fp16(value[(p * kKvHeads + kv) * kD + d]);
        }
        out[d] = host_bf16_rne(result);
      }
    }
  }
  return true;
}

bool launch_control(Buffers *const b, const uint64_t rows,
                    const uint64_t start) {
  const uint64_t blocks = ((rows + 3U) / 4U) * kKvHeads;
  hipLaunchKernelGGL((gqa6_q4k8_control_kernel),
                     dim3(static_cast<uint32_t>(blocks)), dim3(kThreads), 0U,
                     b->stream, b->query, b->key, b->value, b->output,
                     static_cast<uint32_t>(rows), start);
  return hip_ok(hipGetLastError(), "control launch");
}

bool launch_fused(Buffers *const b, const uint64_t rows, const uint64_t context,
                  const uint64_t start) {
  const uint64_t blocks = ((rows + 7U) / 8U) * kKvHeads;
  hipLaunchKernelGGL((gqa6_fused_wmma_kernel),
                     dim3(static_cast<uint32_t>(blocks)), dim3(kThreads), 0U,
                     b->stream, b->query, b->key, b->value, b->output_f32,
                     static_cast<uint32_t>(rows), context, start);
  if (!hip_ok(hipGetLastError(), "fused launch"))
    return false;
  const uint64_t output_values = rows * kQHeads * kD;
  const uint32_t unpack_blocks =
      static_cast<uint32_t>((output_values + kThreads - 1U) / kThreads);
  hipLaunchKernelGGL((unpack_bf16_kernel), dim3(unpack_blocks), dim3(kThreads),
                     0U, b->stream, b->output_f32, b->output, output_values);
  return hip_ok(hipGetLastError(), "unpack launch");
}

bool launch_persistent(Buffers *const b, const uint64_t rows,
                       const uint64_t context, const uint64_t start) {
  const uint64_t blocks = ((rows + 1U) / 2U) * kKvHeads;
  hipLaunchKernelGGL((gqa6_persistent_wmma_kernel),
                     dim3(static_cast<uint32_t>(blocks)), dim3(kThreads), 0U,
                     b->stream, b->query, b->key, b->value, b->output_f32,
                     static_cast<uint32_t>(rows), context, start);
  if (!hip_ok(hipGetLastError(), "persistent launch"))
    return false;
  const uint64_t output_values = rows * kQHeads * kD;
  const uint32_t unpack_blocks =
      static_cast<uint32_t>((output_values + kThreads - 1U) / kThreads);
  hipLaunchKernelGGL((unpack_bf16_kernel), dim3(unpack_blocks), dim3(kThreads),
                     0U, b->stream, b->output_f32, b->output, output_values);
  return hip_ok(hipGetLastError(), "persistent unpack launch");
}

bool measure_control(Buffers *const b, const uint64_t rows,
                     const uint64_t start, float *const median_us) {
  for (uint32_t i = 0U; i < kWarmups; ++i)
    if (!launch_control(b, rows, start) ||
        !hip_ok(hipStreamSynchronize(b->stream), "control warmup"))
      return false;
  std::array<float, kMeasured> samples{};
  for (uint32_t i = 0U; i < kMeasured; ++i) {
    if (!hip_ok(hipEventRecord(b->start, b->stream), "control event start") ||
        !launch_control(b, rows, start) ||
        !hip_ok(hipEventRecord(b->stop, b->stream), "control event stop") ||
        !hip_ok(hipEventSynchronize(b->stop), "control event sync") ||
        !hip_ok(hipEventElapsedTime(&samples[i], b->start, b->stop),
                "control elapsed"))
      return false;
    samples[i] *= 1000.0F;
  }
  std::sort(samples.begin(), samples.end());
  *median_us = samples[kMeasured / 2U];
  return true;
}

bool measure_fused(Buffers *const b, const uint64_t rows,
                   const uint64_t context, const uint64_t start,
                   float *const median_us) {
  for (uint32_t i = 0U; i < kWarmups; ++i)
    if (!launch_fused(b, rows, context, start) ||
        !hip_ok(hipStreamSynchronize(b->stream), "fused warmup"))
      return false;
  std::array<float, kMeasured> samples{};
  for (uint32_t i = 0U; i < kMeasured; ++i) {
    if (!hip_ok(hipEventRecord(b->start, b->stream), "fused event start") ||
        !launch_fused(b, rows, context, start) ||
        !hip_ok(hipEventRecord(b->stop, b->stream), "fused event stop") ||
        !hip_ok(hipEventSynchronize(b->stop), "fused event sync") ||
        !hip_ok(hipEventElapsedTime(&samples[i], b->start, b->stop),
                "fused elapsed"))
      return false;
    samples[i] *= 1000.0F;
  }
  std::sort(samples.begin(), samples.end());
  *median_us = samples[kMeasured / 2U];
  return true;
}

bool measure_persistent(Buffers *const b, const uint64_t rows,
                        const uint64_t context, const uint64_t start,
                        float *const median_us) {
  for (uint32_t i = 0U; i < kWarmups; ++i)
    if (!launch_persistent(b, rows, context, start) ||
        !hip_ok(hipStreamSynchronize(b->stream), "persistent warmup"))
      return false;
  std::array<float, kMeasured> samples{};
  for (uint32_t i = 0U; i < kMeasured; ++i) {
    if (!hip_ok(hipEventRecord(b->start, b->stream),
                "persistent event start") ||
        !launch_persistent(b, rows, context, start) ||
        !hip_ok(hipEventRecord(b->stop, b->stream), "persistent event stop") ||
        !hip_ok(hipEventSynchronize(b->stop), "persistent event sync") ||
        !hip_ok(hipEventElapsedTime(&samples[i], b->start, b->stop),
                "persistent elapsed"))
      return false;
    samples[i] *= 1000.0F;
  }
  std::sort(samples.begin(), samples.end());
  *median_us = samples[kMeasured / 2U];
  return true;
}

bool copy_output(const uint64_t rows, Buffers *const b,
                 std::vector<uint16_t> *const output) {
  output->resize(rows * kQHeads * kD);
  return hip_ok(hipMemcpy(output->data(), b->output,
                          output->size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "copy output");
}

void compare(const char *const name, const std::vector<uint16_t> &ref,
             const std::vector<uint16_t> &actual) {
  uint32_t max_ulp = 0U;
  uint64_t over_one = 0U;
  double max_abs = 0.0;
  for (size_t i = 0U; i < ref.size(); ++i) {
    max_ulp = std::max(max_ulp, ulp_distance(ref[i], actual[i]));
    if (ulp_distance(ref[i], actual[i]) > 1U)
      ++over_one;
    max_abs =
        std::max(max_abs, std::abs(static_cast<double>(host_bf16(ref[i])) -
                                   static_cast<double>(host_bf16(actual[i]))));
  }
  std::printf("compare candidate=%s values=%zu max_bf16_ulp=%u over1=%llu "
              "max_abs=%g status=%s\n",
              name, ref.size(), max_ulp,
              static_cast<unsigned long long>(over_one), max_abs,
              over_one == 0U ? "PASS" : "INFO");
}

void print_resources() {
  const std::array<const char *, 3> names = {"q4k8_control", "fused_wmma",
                                             "persistent_wmma"};
  const std::array<const void *, 3> functions = {
      reinterpret_cast<const void *>(gqa6_q4k8_control_kernel),
      reinterpret_cast<const void *>(gqa6_fused_wmma_kernel),
      reinterpret_cast<const void *>(gqa6_persistent_wmma_kernel)};
  for (size_t i = 0U; i < functions.size(); ++i) {
    hipFuncAttributes attr{};
    const hipError_t attr_status = hipFuncGetAttributes(&attr, functions[i]);
    int active = 0;
    const hipError_t occ = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, functions[i], kThreads, 0U);
    std::printf("resources candidate=%s registers=%d lds=%zu scratch=%zu "
                "active_blocks=%d occupancy=%s attrs=%s\n",
                names[i], attr.numRegs, attr.sharedSizeBytes,
                attr.localSizeBytes, active, hipGetErrorString(occ),
                hipGetErrorString(attr_status));
  }
}

} // namespace

int main() {
  if (!hip_ok(hipSetDevice(0), "hipSetDevice"))
    return EXIT_FAILURE;
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties"))
    return EXIT_FAILURE;
  std::printf("target=%s pci=%04x:%02x:%02x name=%s\n", properties.gcnArchName,
              properties.pciDomainID, properties.pciBusID,
              properties.pciDeviceID, properties.name);
  if (std::string_view(properties.gcnArchName).find("gfx1201") != 0U) {
    std::fprintf(stderr, "gfx1201 required\n");
    return EXIT_FAILURE;
  }
  print_resources();

  // Non-aligned exact-geometry oracle: every 24 query head is checked.
  {
    constexpr uint64_t rows = 7U;
    constexpr uint64_t context = 13U;
    constexpr uint64_t start = 5U;
    std::vector<uint16_t> query, key, value, expected, actual;
    fill_inputs(rows, context, &query, &key, &value);
    host_oracle(rows, context, start, query, key, value, &expected);
    Buffers b;
    if (!make_buffers(rows, context, &b) || !upload(query, key, value, &b) ||
        !launch_control(&b, rows, start) ||
        !hip_ok(hipStreamSynchronize(b.stream), "small control sync") ||
        !copy_output(rows, &b, &actual)) {
      free_buffers(&b);
      return EXIT_FAILURE;
    }
    compare("q4k8_control_small", expected, actual);
    if (!launch_persistent(&b, rows, context, start) ||
        !hip_ok(hipStreamSynchronize(b.stream), "small persistent sync") ||
        !copy_output(rows, &b, &actual)) {
      free_buffers(&b);
      return EXIT_FAILURE;
    }
    compare("persistent_wmma_small", expected, actual);
    free_buffers(&b);
  }

  bool all_ok = true;
  for (const uint64_t rows : {128U, 512U}) {
    constexpr uint64_t context = 9435U;
    const uint64_t start = context - rows;
    std::vector<uint16_t> query, key, value, control, actual;
    fill_inputs(rows, context, &query, &key, &value);
    Buffers b;
    if (!make_buffers(rows, context, &b) || !upload(query, key, value, &b)) {
      free_buffers(&b);
      return EXIT_FAILURE;
    }
    float control_us = 0.0F;
    float fused_us = 0.0F;
    if (!measure_control(&b, rows, start, &control_us) ||
        !copy_output(rows, &b, &control) ||
        !measure_persistent(&b, rows, context, start, &fused_us) ||
        !copy_output(rows, &b, &actual)) {
      free_buffers(&b);
      return EXIT_FAILURE;
    }
    compare("persistent_wmma_vs_control", control, actual);
    const uint64_t workspace = rows * kQHeads * kD * sizeof(float);
    std::printf("result rows=%llu context=%llu start=%llu control_ms=%.6f "
                "persistent_ms=%.6f speedup=%.3f workspace_bytes=%llu "
                "score_workspace_bytes=0 score_precision=fp32 "
                "probability_precision=fp16\n",
                static_cast<unsigned long long>(rows),
                static_cast<unsigned long long>(context),
                static_cast<unsigned long long>(start), control_us / 1000.0F,
                fused_us / 1000.0F, control_us / fused_us,
                static_cast<unsigned long long>(workspace));
    all_ok = all_ok && fused_us > 0.0F;
    free_buffers(&b);
  }
  std::printf("summary status=%s warmups=%u measured=%u target=gfx1201\n",
              all_ok ? "PASS" : "FAIL", kWarmups, kMeasured);
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
