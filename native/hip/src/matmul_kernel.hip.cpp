// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase9-mmvf-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// f5919bf458ef190468b5c329bb293f8a54a1e69c,
// ggml/src/ggml-cuda/mmvf.cu
// SPDX-License-Identifier: MIT

#include "matmul_kernel_internal.hpp"
#include <lowp/detail/low_precision_block_codec.hpp>

#include <hip/hip_fp8.h>

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1201__)
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>
#define SLLM_MATMUL_HAS_GFX12_ROCWMMA 1
#endif

#include <cstdint>
#include <cstring>

namespace {

#include <lowp/detail/bf16_helpers.inc>

// Phase 87 WU2 C3: gfx1201 native packed-FP8 dot4 GEMV.  The helper keeps
// the bounded-byte tail path used by the scratch probe so numerical probes
// can exercise non-aligned K/N values even though the production launcher
// below admits only the reviewed M=1 shapes.
__device__ __forceinline__ uint32_t phase87_wu2_load_fp8_dword_or_bytes(
    const uint8_t *const source, const uint64_t byte_offset,
    const uint64_t logical_bytes) noexcept {
  if (byte_offset >= logical_bytes) {
    return 0U;
  }
  const uint64_t remaining = logical_bytes - byte_offset;
  const uint8_t *const address = source + byte_offset;
  if (remaining >= UINT64_C(4) && (reinterpret_cast<uintptr_t>(address) &
                                   static_cast<uintptr_t>(3U)) == 0U) {
    return __builtin_nontemporal_load(
        reinterpret_cast<const uint32_t *>(address));
  }
  uint32_t packed = 0U;
#pragma unroll
  for (uint32_t byte = 0U; byte < 4U; ++byte) {
    if (byte_offset + byte < logical_bytes) {
      packed |= static_cast<uint32_t>(source[byte_offset + byte])
                << (byte * 8U);
    }
  }
  return packed;
}

__device__ __forceinline__ float
phase87_wu2_dot4_fp8(const uint32_t lhs, const uint32_t rhs,
                     const float accumulator) noexcept {
#if defined(__gfx1201__) && __has_builtin(__builtin_amdgcn_dot4_f32_fp8_fp8)
  return __builtin_amdgcn_dot4_f32_fp8_fp8(lhs, rhs, accumulator);
#else
  // The production selector never chooses this symbol on gfx1030.  Keep a
  // software implementation for comparative compile/run probes and ensure
  // the gfx1201 intrinsic is not emitted into another target's code object.
  float result = accumulator;
#pragma unroll
  for (uint32_t byte = 0U; byte < 4U; ++byte) {
    const uint8_t lhs_code = static_cast<uint8_t>(lhs >> (byte * 8U));
    const uint8_t rhs_code = static_cast<uint8_t>(rhs >> (byte * 8U));
    result = fmaf(sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode(lhs_code),
                  sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode(rhs_code),
                  result);
  }
  return result;
#endif
}

} // namespace

#pragma clang fp contract(off)
extern "C" __global__ __launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  const uint64_t output_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                                static_cast<uint64_t>(threadIdx.x);
  const uint64_t output_elements = m * n;
  if (output_index < output_elements) {
    const uint64_t row = output_index / n;
    const uint64_t column = output_index - row * n;
    float accumulator = 0.0F;
    for (uint64_t reduction = 0U; reduction != k; ++reduction) {
      const float activation_value =
          bf16_to_float(activation[row * k + reduction]);
      const float weight_value = bf16_to_float(weight[column * k + reduction]);
      accumulator += activation_value * weight_value;
    }
    output[output_index] = float_to_bf16_rne_bits(accumulator);
  }
}

// Row-major [M,K] x transposed row-major [N,K].  A 16x16 output tile shares
// both input tiles, eliminating the baseline kernel's redundant global loads.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_tiled16_v2(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  constexpr uint32_t tile = 16U;
  __shared__ uint16_t activation_tile[tile][tile];
  __shared__ uint16_t weight_tile[tile][tile];
  const uint32_t local_row = threadIdx.y;
  const uint32_t local_column = threadIdx.x;
  const uint64_t row = static_cast<uint64_t>(blockIdx.y) * tile + local_row;
  const uint64_t column =
      static_cast<uint64_t>(blockIdx.x) * tile + local_column;
  float accumulator = 0.0F;
  for (uint64_t base = 0U; base < k; base += tile) {
    const uint64_t activation_k = base + local_column;
    const uint64_t weight_k = base + local_row;
    activation_tile[local_row][local_column] =
        row < m && activation_k < k ? activation[row * k + activation_k] : 0U;
    weight_tile[local_row][local_column] =
        column < n && weight_k < k ? weight[column * k + weight_k] : 0U;
    __syncthreads();
#pragma unroll
    for (uint32_t inner = 0U; inner != tile; ++inner) {
      accumulator += bf16_to_float(activation_tile[local_row][inner]) *
                     bf16_to_float(weight_tile[inner][local_column]);
    }
    __syncthreads();
  }
  if (row < m && column < n) {
    output[row * n + column] = float_to_bf16_rne_bits(accumulator);
  }
}

// Phase 83 gfx1030 large-prefill provider. The B tile is stored transposed in
// LDS so adjacent column threads read adjacent BF16 values. This preserves
// the production BF16 contract: direct BF16 ingress, ascending K FP32
// accumulation with contraction disabled, and BF16 round-to-nearest-even
// output. Guarded M/N/K tails intentionally share the same zero-fill rule as
// the tiled16 provider.
constexpr uint32_t kBf16Prefill64Tile = 64U;
constexpr uint32_t kBf16Prefill64TileK = 32U;
constexpr uint32_t kBf16Prefill64ThreadsPerRow = 16U;
constexpr uint32_t kBf16Prefill64RowsPerThread = 4U;
constexpr uint32_t kBf16Prefill64ColumnsPerThread = 4U;

#pragma clang fp contract(off)
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_prefill_gfx1030_64x64_k32_transposed_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  __shared__ uint16_t activation_tile[kBf16Prefill64Tile][kBf16Prefill64TileK];
  __shared__ uint16_t weight_tile[kBf16Prefill64TileK][kBf16Prefill64Tile];

  const uint32_t thread = threadIdx.x;
  const uint32_t thread_row = thread / kBf16Prefill64ThreadsPerRow;
  const uint32_t thread_column = thread % kBf16Prefill64ThreadsPerRow;
  // The public launcher intentionally retains the historical one-dimensional
  // grid metadata. Decode that flattened M/N tile index here instead of
  // reading blockIdx.y (which is always zero for a 1-D launch).
  const uint64_t column_tiles =
      (n + kBf16Prefill64Tile - 1U) / kBf16Prefill64Tile;
  const uint64_t flat_tile = static_cast<uint64_t>(blockIdx.x);
  const uint64_t row_tile = flat_tile / column_tiles;
  const uint64_t column_tile = flat_tile - row_tile * column_tiles;
  const uint64_t row_base =
      row_tile * kBf16Prefill64Tile +
      static_cast<uint64_t>(thread_row) * kBf16Prefill64RowsPerThread;
  const uint64_t column_base =
      column_tile * kBf16Prefill64Tile +
      static_cast<uint64_t>(thread_column) * kBf16Prefill64ColumnsPerThread;
  float accumulators[kBf16Prefill64RowsPerThread]
                    [kBf16Prefill64ColumnsPerThread] = {};

  for (uint64_t base = 0U; base < k; base += kBf16Prefill64TileK) {
    for (uint32_t index = thread;
         index < kBf16Prefill64Tile * kBf16Prefill64TileK; index += 256U) {
      const uint32_t tile_row = index / kBf16Prefill64TileK;
      const uint32_t tile_inner = index % kBf16Prefill64TileK;
      const uint64_t row = row_tile * kBf16Prefill64Tile + tile_row;
      const uint64_t column = column_tile * kBf16Prefill64Tile + tile_row;
      const uint64_t inner = base + tile_inner;
      activation_tile[tile_row][tile_inner] =
          row < m && inner < k ? activation[row * k + inner] : 0U;
      // Transposed shared layout makes the per-output-column load contiguous.
      weight_tile[tile_inner][tile_row] =
          column < n && inner < k ? weight[column * k + inner] : 0U;
    }
    __syncthreads();

#pragma unroll
    for (uint32_t inner = 0U; inner < kBf16Prefill64TileK; ++inner) {
#pragma unroll
      for (uint32_t local_row = 0U; local_row < kBf16Prefill64RowsPerThread;
           ++local_row) {
        const float activation_value = bf16_to_float(
            activation_tile[thread_row * kBf16Prefill64RowsPerThread +
                            local_row][inner]);
#pragma unroll
        for (uint32_t local_column = 0U;
             local_column < kBf16Prefill64ColumnsPerThread; ++local_column) {
          const float weight_value = bf16_to_float(
              weight_tile[inner]
                         [thread_column * kBf16Prefill64ColumnsPerThread +
                          local_column]);
          accumulators[local_row][local_column] +=
              activation_value * weight_value;
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t local_row = 0U; local_row < kBf16Prefill64RowsPerThread;
       ++local_row) {
#pragma unroll
    for (uint32_t local_column = 0U;
         local_column < kBf16Prefill64ColumnsPerThread; ++local_column) {
      const uint64_t row = row_base + local_row;
      const uint64_t column = column_base + local_column;
      if (row < m && column < n) {
        output[row * n + column] =
            float_to_bf16_rne_bits(accumulators[local_row][local_column]);
      }
    }
  }
}

// Decode is a matrix-vector product. One workgroup owns one output column and
// reduces K cooperatively; this avoids launching mostly idle 16x16 tiles.
//
// The paired BF16 loads and two-level wave reduction are adapted from the
// floating MMVF organization in llama.cpp mmvf.cu at fixed commit
// f5919bf458ef190468b5c329bb293f8a54a1e69c. The ggml tensor/runtime and
// fusion machinery are deliberately not imported; this kernel retains sLLM's
// BF16 input/output and FP32 accumulation contract.
template <uint32_t WaveWidth, uint32_t WaveCount>
__device__ __forceinline__ void
matmul_bf16_decode_body(const uint16_t *const activation,
                        const uint16_t *const weight, uint16_t *const output,
                        const uint64_t k, const uint64_t n,
                        const uint64_t column) {
  if (column >= n) {
    return;
  }
  float partial = 0.0F;
  const uint16_t *const weight_row = weight + column * k;
  const bool paired =
      (k & UINT64_C(1)) == 0U && ((reinterpret_cast<uintptr_t>(activation) |
                                   reinterpret_cast<uintptr_t>(weight_row)) &
                                  static_cast<uintptr_t>(3U)) == 0U;
  if (paired) {
    const auto *const activation_pairs =
        reinterpret_cast<const uint32_t *>(activation);
    const auto *const weight_pairs =
        reinterpret_cast<const uint32_t *>(weight_row);
    const uint64_t pair_count = k / 2U;
    for (uint64_t pair = threadIdx.x; pair < pair_count; pair += blockDim.x) {
      const uint32_t activation_pair = activation_pairs[pair];
      const uint32_t weight_pair =
          __builtin_nontemporal_load(weight_pairs + pair);
      partial += bf16_to_float(static_cast<uint16_t>(activation_pair)) *
                 bf16_to_float(static_cast<uint16_t>(weight_pair));
      partial += bf16_to_float(static_cast<uint16_t>(activation_pair >> 16U)) *
                 bf16_to_float(static_cast<uint16_t>(weight_pair >> 16U));
    }
  } else {
    for (uint64_t reduction = threadIdx.x; reduction < k;
         reduction += blockDim.x) {
      partial += bf16_to_float(activation[reduction]) *
                 bf16_to_float(weight_row[reduction]);
    }
  }

#pragma unroll
  for (uint32_t offset = WaveWidth / 2U; offset != 0U; offset >>= 1U) {
    partial += __shfl_down(partial, offset, WaveWidth);
  }
  __shared__ float wave_sums[WaveCount];
  const uint32_t lane = threadIdx.x % WaveWidth;
  const uint32_t wave = threadIdx.x / WaveWidth;
  if (lane == 0U) {
    wave_sums[wave] = partial;
  }
  __syncthreads();
  if (wave == 0U) {
    partial = lane < WaveCount ? wave_sums[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = WaveWidth / 2U; offset != 0U; offset >>= 1U) {
      partial += __shfl_down(partial, offset, WaveWidth);
    }
    if (lane == 0U) {
      output[column] = float_to_bf16_rne_bits(partial);
    }
  }
}

// Short GDN projection provider for the exact Qwen shape M=17, K=5120,
// N=48.  Unlike the tiled16 prefill body, each block owns one output element
// and reuses the established decode reduction body for its row.  The paired
// BF16 loads, FP32 products, and two-level wave-sum tree therefore remain the
// same as matmul_bf16_decode_body<32U, 8U>; only the M dimension is exposed to
// the grid.  This is selected by the launcher only for the exact supported
// targets and shape below; all other prefill shapes retain tiled16.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_prefill_gdn_thin_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  const uint64_t row = static_cast<uint64_t>(blockIdx.y);
  const uint64_t column = static_cast<uint64_t>(blockIdx.x);
  if (row < m && column < n) {
    matmul_bf16_decode_body<32U, 8U>(activation + row * k, weight,
                                     output + row * n, k, n, column);
  }
}

template <uint32_t WaveWidth, uint32_t WaveCount>
__device__ __forceinline__ void matmul_bf16_serial_rows_body(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const uint64_t row_start = 0U,
    const uint64_t column_override = UINT64_MAX) {
  constexpr uint32_t max_rows = 8U;
  const uint64_t column = column_override == UINT64_MAX
                              ? static_cast<uint64_t>(blockIdx.x)
                              : column_override;
  if (column >= n || m == 0U || m > max_rows) {
    return;
  }
  float partial[max_rows] = {};
  const uint16_t *const weight_row = weight + column * k;
  const uint16_t *const activation_start = activation + row_start * k;
  const bool paired = (k & UINT64_C(1)) == 0U &&
                      ((reinterpret_cast<uintptr_t>(activation_start) |
                        reinterpret_cast<uintptr_t>(weight_row)) &
                       static_cast<uintptr_t>(3U)) == 0U;
  if (paired) {
    const auto *const weight_pairs =
        reinterpret_cast<const uint32_t *>(weight_row);
    const uint64_t pair_count = k / 2U;
    for (uint64_t pair = threadIdx.x; pair < pair_count; pair += blockDim.x) {
      const uint32_t weight_pair =
          __builtin_nontemporal_load(weight_pairs + pair);
      const float weight0 = bf16_to_float(static_cast<uint16_t>(weight_pair));
      const float weight1 =
          bf16_to_float(static_cast<uint16_t>(weight_pair >> 16U));
      for (uint32_t row = 0U; row < m; ++row) {
        const auto *const activation_pairs = reinterpret_cast<const uint32_t *>(
            activation_start + static_cast<uint64_t>(row) * k);
        const uint32_t activation_pair = activation_pairs[pair];
        partial[row] +=
            bf16_to_float(static_cast<uint16_t>(activation_pair)) * weight0;
        partial[row] +=
            bf16_to_float(static_cast<uint16_t>(activation_pair >> 16U)) *
            weight1;
      }
    }
  } else {
    for (uint64_t reduction = threadIdx.x; reduction < k;
         reduction += blockDim.x) {
      const float weight_value = bf16_to_float(weight_row[reduction]);
      for (uint32_t row = 0U; row < m; ++row) {
        partial[row] +=
            bf16_to_float(
                activation_start[static_cast<uint64_t>(row) * k + reduction]) *
            weight_value;
      }
    }
  }

  for (uint32_t offset = WaveWidth / 2U; offset != 0U; offset >>= 1U) {
    for (uint32_t row = 0U; row < m; ++row) {
      partial[row] += __shfl_down(partial[row], offset, WaveWidth);
    }
  }
  __shared__ float wave_sums[max_rows][WaveCount];
  const uint32_t lane = threadIdx.x % WaveWidth;
  const uint32_t wave = threadIdx.x / WaveWidth;
  if (lane == 0U) {
    for (uint32_t row = 0U; row < m; ++row) {
      wave_sums[row][wave] = partial[row];
    }
  }
  __syncthreads();
  if (wave == 0U) {
    for (uint32_t row = 0U; row < m; ++row) {
      partial[row] = lane < WaveCount ? wave_sums[row][lane] : 0.0F;
    }
    for (uint32_t offset = WaveWidth / 2U; offset != 0U; offset >>= 1U) {
      for (uint32_t row = 0U; row < m; ++row) {
        partial[row] += __shfl_down(partial[row], offset, WaveWidth);
      }
    }
    if (lane == 0U) {
      for (uint32_t row = 0U; row < m; ++row) {
        output[(row_start + static_cast<uint64_t>(row)) * n + column] =
            float_to_bf16_rne_bits(partial[row]);
      }
    }
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_decode_v4(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t k, const uint64_t n) {
  matmul_bf16_decode_body<32U, 8U>(activation, weight, output, k, n,
                                   blockIdx.x);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_decode_wave64_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t k, const uint64_t n) {
  matmul_bf16_decode_body<64U, 4U>(activation, weight, output, k, n,
                                   blockIdx.x);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_decode_serial_rows_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  matmul_bf16_serial_rows_body<32U, 8U>(activation, weight, output, m, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_decode_serial_rows_wave64_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  matmul_bf16_serial_rows_body<64U, 4U>(activation, weight, output, m, k, n);
}

// Short prefill provider for the exact gfx1030 Qwen projection shapes.  Each
// block owns one output column and one consecutive group of up to eight rows;
// the existing serial-reduction body is reused unchanged for each group.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_prefill_short_serial_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  if (n == 0U || m == 0U) {
    return;
  }
  const uint64_t column = blockIdx.x % n;
  const uint64_t row_group = blockIdx.x / n;
  const uint64_t row_start = row_group * UINT64_C(8);
  if (row_start >= m) {
    return;
  }
  const uint64_t remaining_rows = m - row_start;
  const uint64_t rows =
      remaining_rows < UINT64_C(8) ? remaining_rows : UINT64_C(8);
  // gfx1030 uses wave32, matching the established M=2..8 provider.
  matmul_bf16_serial_rows_body<32U, 8U>(activation, weight, output, rows, k, n,
                                        row_start, column);
}

extern "C" __global__ void
sllm_matmul_fp32_to_bf16_short_mixed_v1(const float *const input,
                                        uint16_t *const output,
                                        const uint64_t element_count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    output[index] = float_to_bf16_rne_bits(input[index]);
  }
}

// One wave owns one output column.  A block has eight waves and therefore
// handles eight adjacent columns.  Four packed dwords per lane preserve the
// C3 scratch arithmetic and load schedule; the final four subtotal values are
// combined in the same fixed order before the wave reduction.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_fp8_outer_gfx1201_dot4_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  const uint32_t lane = threadIdx.x % 32U;
  const uint32_t wave = threadIdx.x / 32U;
  const uint64_t row = static_cast<uint64_t>(blockIdx.y);
  const uint64_t column = static_cast<uint64_t>(blockIdx.x) * 8U + wave;
  if (row >= m || column >= n) {
    return;
  }

  float sums[4] = {};
  for (uint64_t base = static_cast<uint64_t>(lane) * 16U; base < k;
       base += 512U) {
    uint32_t activation_values[4];
    uint32_t weight_values[4];
    const uint8_t *const activation_row = activation + row * k;
    const uint8_t *const weight_row = weight + column * k;
    if (base + 16U <= k &&
        ((reinterpret_cast<uintptr_t>(activation_row + base) |
          reinterpret_cast<uintptr_t>(weight_row + base)) &
         static_cast<uintptr_t>(15U)) == 0U) {
      const uint4 activation_vector =
          *reinterpret_cast<const uint4 *>(activation_row + base);
      using Packed4 = uint32_t __attribute__((ext_vector_type(4)));
      const Packed4 weight_vector = __builtin_nontemporal_load(
          reinterpret_cast<const Packed4 *>(weight_row + base));
      activation_values[0] = activation_vector.x;
      activation_values[1] = activation_vector.y;
      activation_values[2] = activation_vector.z;
      activation_values[3] = activation_vector.w;
      weight_values[0] = weight_vector[0];
      weight_values[1] = weight_vector[1];
      weight_values[2] = weight_vector[2];
      weight_values[3] = weight_vector[3];
    } else {
#pragma unroll
      for (uint32_t index = 0U; index < 4U; ++index) {
        activation_values[index] = phase87_wu2_load_fp8_dword_or_bytes(
            activation_row, base + static_cast<uint64_t>(index) * 4U, k);
        weight_values[index] = phase87_wu2_load_fp8_dword_or_bytes(
            weight_row, base + static_cast<uint64_t>(index) * 4U, k);
      }
    }
#pragma unroll
    for (uint32_t index = 0U; index < 4U; ++index) {
      sums[index] = phase87_wu2_dot4_fp8(activation_values[index],
                                         weight_values[index], sums[index]);
    }
  }

  float accumulator = (sums[0] + sums[1]) + (sums[2] + sums[3]);
#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    accumulator += __shfl_down(accumulator, offset, 32U);
  }
  if (lane == 0U) {
    output[row * n + column] = float_to_bf16_rne_bits(
        accumulator * activation_scales[row] * weight_scales[column]);
  }
}

namespace sllm_matmul_kernel {

hipError_t launch(const uint16_t *const activation,
                  const uint16_t *const weight, uint16_t *const output,
                  const uint64_t m, const uint64_t k, const uint64_t n,
                  const KernelVariant variant,
                  const hipStream_t stream) noexcept {
  if (variant == HostKernelVariant::HipBlas) {
    return hipErrorInvalidValue;
  }
  if (variant == HostKernelVariant::DecodeReductionWave64) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_decode_wave64_v1,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, activation, weight, output, k, n);
  } else if (variant == HostKernelVariant::DecodeReduction) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_decode_v4,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, activation, weight, output, k, n);
  } else if (variant == HostKernelVariant::SerialRowsReductionWave64) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_decode_serial_rows_wave64_v1,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, activation, weight, output, m, k, n);
  } else if (variant == HostKernelVariant::SerialRowsReduction) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_decode_serial_rows_v1,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, activation, weight, output, m, k, n);
  } else if (variant == HostKernelVariant::PrefillShortSerial) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_prefill_short_serial_v1,
                       dim3(grid_size_x(variant, m, n)), dim3(kWorkgroupSize),
                       0U, stream, activation, weight, output, m, k, n);
  } else if (variant == HostKernelVariant::Bf16PrefillGfx1030_64x64) {
    hipLaunchKernelGGL(
        sllm_matmul_bf16_fp32_prefill_gfx1030_64x64_k32_transposed_v1,
        dim3(grid_size_x(variant, m, n)), dim3(kWorkgroupSize), 0U, stream,
        activation, weight, output, m, k, n);
  } else if (variant == HostKernelVariant::PrefillTiled16) {
#if defined(SLLM_HIP_COMPILE_TARGET)
    const bool gdn_thin_shape =
        (std::strcmp(SLLM_HIP_COMPILE_TARGET, "gfx1030") == 0 ||
         std::strcmp(SLLM_HIP_COMPILE_TARGET, "gfx1201") == 0) &&
        m == UINT64_C(17) && k == UINT64_C(5120) && n == UINT64_C(48);
#else
    constexpr bool gdn_thin_shape = false;
#endif
    if (gdn_thin_shape) {
      hipLaunchKernelGGL(
          sllm_matmul_bf16_fp32_prefill_gdn_thin_v1,
          dim3(static_cast<uint32_t>(n), static_cast<uint32_t>(m)),
          dim3(kWorkgroupSize), 0U, stream, activation, weight, output, m, k,
          n);
    } else {
      hipLaunchKernelGGL(sllm_matmul_bf16_fp32_tiled16_v2,
                         dim3(static_cast<uint32_t>((n + 15U) / 16U),
                              static_cast<uint32_t>((m + 15U) / 16U)),
                         dim3(16U, 16U), 0U, stream, activation, weight, output,
                         m, k, n);
    }
  } else {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_v1,
                       dim3(grid_size_x(variant, m, n)), dim3(kWorkgroupSize),
                       0U, stream, activation, weight, output, m, k, n);
  }
  return hipGetLastError();
}

hipError_t launch_short_mixed_f32_to_bf16(const float *const output_f32,
                                          uint16_t *const output,
                                          const uint64_t element_count,
                                          const hipStream_t stream) noexcept {
  if (element_count == 0U || element_count > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_matmul_fp32_to_bf16_short_mixed_v1,
                     dim3(static_cast<uint32_t>((element_count + 255U) / 256U)),
                     dim3(kWorkgroupSize), 0U, stream, output_f32, output,
                     element_count);
  return hipGetLastError();
}

hipError_t launch_fp8_outer_gfx1201_dot4(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const hipStream_t stream) noexcept {
  if (activation == nullptr || activation_scales == nullptr ||
      weight == nullptr || weight_scales == nullptr || output == nullptr ||
      !fp8_outer_gfx1201_dot4_shape(m, k, n)) {
    return hipErrorInvalidValue;
  }
#if defined(SLLM_HIP_COMPILE_TARGET)
  if (std::strcmp(SLLM_HIP_COMPILE_TARGET, "gfx1201") != 0) {
    return hipErrorInvalidValue;
  }
#endif
  hipLaunchKernelGGL(
      sllm_matmul_fp8_outer_gfx1201_dot4_v1,
      dim3(static_cast<uint32_t>((n + 7U) / 8U), static_cast<uint32_t>(m)),
      dim3(kWorkgroupSize), 0U, stream, activation, activation_scales, weight,
      weight_scales, output, m, k, n);
  return hipGetLastError();
}

} // namespace sllm_matmul_kernel
