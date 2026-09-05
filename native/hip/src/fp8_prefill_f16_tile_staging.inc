// Phase 78 ID86: transient FP16 ingress followed by the gfx1030 64x64/K32
// half2 tile.  The staging kernels are the existing ID70 primitives in
// matmul_kernel.hip.cpp; this body only consumes their row-major FP16 output.
// FP8 values remain resident in their original byte representation.

namespace sllm_phase78_fp8_f16_tile_staging {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kTileM = 64U;
constexpr uint32_t kTileN = 64U;
constexpr uint32_t kTileK = 32U;
constexpr uint32_t kLdsStride = kTileK + 2U;
constexpr uint32_t kThreadRows = 16U;
constexpr uint32_t kThreadColumns = 16U;
constexpr uint32_t kValuesPerPackedLoad = 4U;

// K is restricted by the host selector to a multiple of four.  Each source
// row is therefore dword aligned when the staging buffer itself is aligned,
// including the final partial K tile.  Bounds checks still cover K tails.
extern "C" __global__
__launch_bounds__(kThreads, 1) void sllm_matmul_fp8_outer_prefill_gfx1030_f16_tile_staging_v1(
    const uint16_t *const activation, const float *const activation_scales,
    const uint16_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t rows_per_thread = kTileM / kThreadRows;
  constexpr uint32_t columns_per_thread = kTileN / kThreadColumns;
  __shared__ __align__(4) uint16_t activation_tile[kTileM][kLdsStride];
  __shared__ __align__(4) uint16_t weight_tile[kTileN][kLdsStride];

  const uint64_t column_tiles = (n + kTileN - 1U) / kTileN;
  const uint64_t tile_index = static_cast<uint64_t>(blockIdx.x);
  const uint64_t row_base = (tile_index / column_tiles) * kTileM;
  const uint64_t column_base = (tile_index % column_tiles) * kTileN;
  const uint32_t thread = threadIdx.x;
  const uint32_t local_row = thread >> 4U;
  const uint32_t local_column = thread & UINT32_C(15);
  float accumulators[rows_per_thread][columns_per_thread] = {};

  for (uint64_t base = 0U; base < k; base += kTileK) {
    constexpr uint32_t loads_per_row = kTileK / kValuesPerPackedLoad;
    for (uint32_t index = thread; index < kTileM * loads_per_row;
         index += kThreads) {
      const uint32_t row = index / loads_per_row;
      const uint32_t inner = (index % loads_per_row) * kValuesPerPackedLoad;
      const uint64_t source_row = row_base + row;
      const uint64_t source_inner = base + inner;
      uint16_t *const destination = &activation_tile[row][inner];
      if (source_row < m && source_inner < k) {
        const uint32_t *const packed_source =
            reinterpret_cast<const uint32_t *>(activation) +
            source_row * (k / 2U) + source_inner / 2U;
#pragma unroll
        for (uint32_t lane = 0U; lane < kValuesPerPackedLoad; ++lane) {
          const uint64_t source_index = source_inner + lane;
          if (source_index < k) {
            const uint32_t packed = packed_source[lane / 2U];
            destination[lane] =
                static_cast<uint16_t>(packed >> ((lane & 1U) * 16U));
          } else {
            destination[lane] = UINT16_C(0);
          }
        }
      } else {
#pragma unroll
        for (uint32_t lane = 0U; lane < kValuesPerPackedLoad; ++lane) {
          destination[lane] = UINT16_C(0);
        }
      }
    }
    for (uint32_t index = thread; index < kTileN * loads_per_row;
         index += kThreads) {
      const uint32_t column = index / loads_per_row;
      const uint32_t inner = (index % loads_per_row) * kValuesPerPackedLoad;
      const uint64_t source_column = column_base + column;
      const uint64_t source_inner = base + inner;
      uint16_t *const destination = &weight_tile[column][inner];
      if (source_column < n && source_inner < k) {
        const uint32_t *const packed_source =
            reinterpret_cast<const uint32_t *>(weight) +
            source_column * (k / 2U) + source_inner / 2U;
#pragma unroll
        for (uint32_t lane = 0U; lane < kValuesPerPackedLoad; ++lane) {
          const uint64_t source_index = source_inner + lane;
          if (source_index < k) {
            const uint32_t packed = packed_source[lane / 2U];
            destination[lane] =
                static_cast<uint16_t>(packed >> ((lane & 1U) * 16U));
          } else {
            destination[lane] = UINT16_C(0);
          }
        }
      } else {
#pragma unroll
        for (uint32_t lane = 0U; lane < kValuesPerPackedLoad; ++lane) {
          destination[lane] = UINT16_C(0);
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (uint32_t inner = 0U; inner < kTileK; inner += 2U) {
      __half2 activation_pairs[rows_per_thread];
      __half2 weight_pairs[columns_per_thread];
#pragma unroll
      for (uint32_t row = 0U; row < rows_per_thread; ++row) {
        activation_pairs[row] = *reinterpret_cast<const __half2 *>(
            &activation_tile[local_row + row * kThreadRows][inner]);
      }
#pragma unroll
      for (uint32_t column = 0U; column < columns_per_thread; ++column) {
        weight_pairs[column] = *reinterpret_cast<const __half2 *>(
            &weight_tile[local_column + column * kThreadColumns][inner]);
      }
#pragma unroll
      for (uint32_t row = 0U; row < rows_per_thread; ++row) {
#pragma unroll
        for (uint32_t column = 0U; column < columns_per_thread; ++column) {
          accumulators[row][column] =
              amd_mixed_dot(activation_pairs[row], weight_pairs[column],
                            accumulators[row][column], false);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t row = 0U; row < rows_per_thread; ++row) {
    const uint64_t output_row = row_base + local_row + row * kThreadRows;
#pragma unroll
    for (uint32_t column = 0U; column < columns_per_thread; ++column) {
      const uint64_t output_column =
          column_base + local_column + column * kThreadColumns;
      if (output_row < m && output_column < n) {
        output[output_row * n + output_column] = float_to_bf16_rne_bits(
            accumulators[row][column] * activation_scales[output_row] *
            weight_scales[output_column]);
      }
    }
  }
}

} // namespace sllm_phase78_fp8_f16_tile_staging
