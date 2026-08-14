// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase9-mmvf-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// f5919bf458ef190468b5c329bb293f8a54a1e69c,
// ggml/src/ggml-cuda/mmvf.cu
// SPDX-License-Identifier: MIT

#include "matmul_kernel_internal.hpp"

#include <cstdint>

namespace {

__device__ __forceinline__ float bf16_to_float(const uint16_t value) noexcept {
  return __uint_as_float(static_cast<uint32_t>(value) << 16U);
}

__device__ __forceinline__ uint16_t
float_to_bf16_rne_bits(const float value) noexcept {
  const uint32_t bits = __float_as_uint(value);
  constexpr uint32_t exponent_mask = UINT32_C(0x7f800000);
  constexpr uint32_t fraction_mask = UINT32_C(0x007fffff);
  if ((bits & exponent_mask) == exponent_mask) {
    if ((bits & fraction_mask) != 0U) {
      const uint16_t sign =
          static_cast<uint16_t>((bits >> 16U) & UINT32_C(0x8000));
      const uint16_t payload =
          static_cast<uint16_t>((bits >> 16U) & UINT32_C(0x003f));
      return static_cast<uint16_t>(sign | UINT16_C(0x7fc0) | payload);
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & UINT32_C(1)) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

__device__ __forceinline__ float e4m3fn_to_float(const uint8_t bits) noexcept {
  const float sign = (bits & UINT8_C(0x80)) == 0U ? 1.0F : -1.0F;
  const uint8_t exponent = static_cast<uint8_t>((bits >> 3U) & UINT8_C(0x0f));
  const uint8_t mantissa = static_cast<uint8_t>(bits & UINT8_C(0x07));
  if (exponent == 0U) {
    return mantissa == 0U ? copysignf(0.0F, sign)
                          : sign * static_cast<float>(mantissa) *
                                ldexpf(1.0F, -9);
  }
  if (exponent == UINT8_C(0x0f) && mantissa == UINT8_C(0x07)) {
    return NAN;
  }
  return sign * (1.0F + static_cast<float>(mantissa) / 8.0F) *
         ldexpf(1.0F, static_cast<int>(exponent) - 7);
}

__device__ __forceinline__ uint8_t float_to_e4m3fn(float value) noexcept {
  const uint8_t sign = signbit(value) ? UINT8_C(0x80) : 0U;
  value = fabsf(value);
  if (value == 0.0F) {
    return sign;
  }
  if (!isfinite(value) || value >= 448.0F) {
    return static_cast<uint8_t>(sign | UINT8_C(0x7e));
  }
  uint32_t low = 0U;
  uint32_t high = UINT32_C(0x7e);
  while (low < high) {
    const uint32_t middle = (low + high) >> 1U;
    if (e4m3fn_to_float(static_cast<uint8_t>(middle)) < value) {
      low = middle + 1U;
    } else {
      high = middle;
    }
  }
  const uint8_t upper = static_cast<uint8_t>(low);
  const uint8_t lower = upper == 0U ? 0U : static_cast<uint8_t>(upper - 1U);
  const float lower_error = value - e4m3fn_to_float(lower);
  const float upper_error = e4m3fn_to_float(upper) - value;
  const bool select_upper = upper_error < lower_error ||
                            (upper_error == lower_error &&
                             (upper & UINT8_C(1)) == 0U &&
                             (lower & UINT8_C(1)) != 0U);
  return static_cast<uint8_t>(sign | (select_upper ? upper : lower));
}

__device__ __forceinline__ float e4m3fnuz_to_float(const uint8_t bits) noexcept {
  if (bits == UINT8_C(0x80)) {
    return NAN;
  }
  const float sign = (bits & UINT8_C(0x80)) == 0U ? 1.0F : -1.0F;
  const uint8_t exponent = static_cast<uint8_t>((bits >> 3U) & UINT8_C(0x0f));
  const uint8_t mantissa = static_cast<uint8_t>(bits & UINT8_C(0x07));
  return exponent == 0U
             ? sign * static_cast<float>(mantissa) * ldexpf(1.0F, -10)
             : sign * (1.0F + static_cast<float>(mantissa) / 8.0F) *
                   ldexpf(1.0F, static_cast<int>(exponent) - 8);
}

__device__ __forceinline__ uint8_t float_to_e4m3fnuz(float value) noexcept {
  if (isnan(value)) {
    return UINT8_C(0x80);
  }
  const bool negative = signbit(value);
  value = fabsf(value);
  if (value == 0.0F) {
    return 0U;
  }
  if (!isfinite(value) || value >= 240.0F) {
    return negative ? UINT8_C(0xff) : UINT8_C(0x7f);
  }
  uint8_t low = 0U;
  uint8_t high = UINT8_C(0x7f);
  while (low < high) {
    const uint8_t middle =
        static_cast<uint8_t>(low + static_cast<uint8_t>((high - low) / 2U));
    if (e4m3fnuz_to_float(middle) < value) {
      low = static_cast<uint8_t>(middle + 1U);
    } else {
      high = middle;
    }
  }
  const uint8_t upper = low;
  const uint8_t lower = upper == 0U ? 0U : static_cast<uint8_t>(upper - 1U);
  const float lower_error = value - e4m3fnuz_to_float(lower);
  const float upper_error = e4m3fnuz_to_float(upper) - value;
  const bool select_upper = upper_error < lower_error ||
                            (upper_error == lower_error &&
                             (upper & UINT8_C(1)) == 0U &&
                             (lower & UINT8_C(1)) != 0U);
  const uint8_t selected = select_upper ? upper : lower;
  return negative && selected != 0U
             ? static_cast<uint8_t>(selected | UINT8_C(0x80))
             : selected;
}

} // namespace

extern "C" __global__ __launch_bounds__(256, 1) void
sllm_matmul_bf16_to_fp8_outer_v1(const uint16_t *const activation,
                                 uint8_t *const quantized, float *const scales,
                                 const uint64_t m, const uint64_t k,
                                 const uint32_t fnuz) {
  const uint64_t row = blockIdx.x;
  if (row >= m) {
    return;
  }
  float maximum = 0.0F;
  for (uint64_t column = threadIdx.x; column < k; column += blockDim.x) {
    maximum = fmaxf(maximum, fabsf(bf16_to_float(activation[row * k + column])));
  }
  __shared__ float reductions[256];
  reductions[threadIdx.x] = maximum;
  __syncthreads();
  for (uint32_t offset = 128U; offset != 0U; offset >>= 1U) {
    if (threadIdx.x < offset) {
      reductions[threadIdx.x] =
          fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  const float scale = reductions[0] == 0.0F
                          ? 1.0F
                          : reductions[0] / (fnuz != 0U ? 240.0F : 448.0F);
  if (threadIdx.x == 0U) {
    scales[row] = scale;
  }
  for (uint64_t column = threadIdx.x; column < k; column += blockDim.x) {
    const float value = bf16_to_float(activation[row * k + column]) / scale;
    quantized[row * k + column] =
        fnuz != 0U ? float_to_e4m3fnuz(value) : float_to_e4m3fn(value);
  }
}

extern "C" __global__ __launch_bounds__(256, 1) void
sllm_matmul_fp8_outer_emulation_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index >= m * n) {
    return;
  }
  const uint64_t row = index / n;
  const uint64_t column = index - row * n;
  float accumulator = 0.0F;
  for (uint64_t inner = 0U; inner < k; ++inner) {
    accumulator = fmaf(e4m3fn_to_float(activation[row * k + inner]),
                       e4m3fn_to_float(weight[column * k + inner]),
                       accumulator);
  }
  output[index] = float_to_bf16_rne_bits(
      accumulator * activation_scales[row] * weight_scales[column]);
}

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

// Decode is a matrix-vector product. One workgroup owns one output column and
// reduces K cooperatively; this avoids launching mostly idle 16x16 tiles.
//
// The paired BF16 loads and two-level wave reduction are adapted from the
// floating MMVF organization in llama.cpp mmvf.cu at fixed commit
// f5919bf458ef190468b5c329bb293f8a54a1e69c. The ggml tensor/runtime and
// fusion machinery are deliberately not imported; this kernel retains sLLM's
// BF16 input/output and FP32 accumulation contract.
template <uint32_t WaveWidth, uint32_t WaveCount>
__device__ __forceinline__ void matmul_bf16_decode_body(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t k, const uint64_t n) {
  const uint64_t column = blockIdx.x;
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
      const uint32_t weight_pair = weight_pairs[pair];
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

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_decode_v3(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t k, const uint64_t n) {
  matmul_bf16_decode_body<32U, 8U>(activation, weight, output, k, n);
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_matmul_bf16_fp32_decode_wave64_v1(
    const uint16_t *const activation, const uint16_t *const weight,
    uint16_t *const output, const uint64_t k, const uint64_t n) {
  matmul_bf16_decode_body<64U, 4U>(activation, weight, output, k, n);
}

namespace sllm_matmul_kernel {
hipError_t launch_fp8_quantize(
    const uint16_t *const activation, uint8_t *const quantized,
    float *const scales, const uint64_t m, const uint64_t k,
    const bool fnuz,
    const hipStream_t stream) noexcept {
  hipLaunchKernelGGL(sllm_matmul_bf16_to_fp8_outer_v1,
                     dim3(static_cast<uint32_t>(m)), dim3(kWorkgroupSize), 0U,
                     stream, activation, quantized, scales, m, k,
                     fnuz ? UINT32_C(1) : UINT32_C(0));
  return hipGetLastError();
}

hipError_t launch_fp8_emulation(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n, const hipStream_t stream) noexcept {
  const uint64_t elements = m * n;
  const uint32_t blocks = static_cast<uint32_t>(
      (elements + kWorkgroupSize - 1U) / kWorkgroupSize);
  hipLaunchKernelGGL(sllm_matmul_fp8_outer_emulation_v1, dim3(blocks),
                     dim3(kWorkgroupSize), 0U, stream, activation,
                     activation_scales, weight, weight_scales, output, m, k,
                     n);
  return hipGetLastError();
}

hipError_t launch(const uint16_t *const activation,
                  const uint16_t *const weight, uint16_t *const output,
                  const uint64_t m, const uint64_t k, const uint64_t n,
                  const KernelVariant variant,
                  const hipStream_t stream) noexcept {
  if (variant == KernelVariant::HipBlas) {
    return hipErrorInvalidValue;
  }
  if (variant == KernelVariant::DecodeReductionWave64) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_decode_wave64_v1,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, activation, weight, output, k, n);
  } else if (variant == KernelVariant::DecodeReduction) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_decode_v3,
                       dim3(static_cast<uint32_t>(n)), dim3(kWorkgroupSize), 0U,
                       stream, activation, weight, output, k, n);
  } else if (variant == KernelVariant::PrefillTiled16) {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_tiled16_v2,
                       dim3(static_cast<uint32_t>((n + 15U) / 16U),
                            static_cast<uint32_t>((m + 15U) / 16U)),
                       dim3(16U, 16U), 0U, stream, activation, weight, output,
                       m, k, n);
  } else {
    hipLaunchKernelGGL(sllm_matmul_bf16_fp32_v1,
                       dim3(grid_size_x(variant, m, n)), dim3(kWorkgroupSize),
                       0U, stream, activation, weight, output, m, k, n);
  }
  return hipGetLastError();
}

} // namespace sllm_matmul_kernel
