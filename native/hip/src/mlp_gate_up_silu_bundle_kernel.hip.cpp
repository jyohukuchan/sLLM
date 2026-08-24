#include "mlp_gate_up_silu_bundle_kernel_internal.hpp"

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

} // namespace

/* One block owns one N column and computes both projections in the same
 * reduction.  The two partial sums deliberately mirror
 * matmul_bf16_decode_body<32,8>: paired BF16 loads, FP32 accumulation, and a
 * two-level wave reduction.  SiLU consumes the rounded gate BF16 and rounded
 * up BF16 exactly as the existing elementwise kernel does. */
extern "C" __global__
__launch_bounds__(256, 1) void sllm_mlp_gate_up_silu_bundle_bf16_fp32_decode_v1(
    const uint16_t *const activation, const uint16_t *const gate_weight,
    const uint16_t *const up_weight, uint16_t *const gate_output,
    uint16_t *const up_output, uint16_t *const silu_output, const uint64_t k,
    const uint64_t n) {
  const uint64_t column = static_cast<uint64_t>(blockIdx.x);
  if (column >= n) {
    return;
  }
  float gate_partial = 0.0F;
  float up_partial = 0.0F;
  const uint16_t *const gate_row = gate_weight + column * k;
  const uint16_t *const up_row = up_weight + column * k;
  const bool paired =
      (k & UINT64_C(1)) == 0U && ((reinterpret_cast<uintptr_t>(activation) |
                                   reinterpret_cast<uintptr_t>(gate_row) |
                                   reinterpret_cast<uintptr_t>(up_row)) &
                                  static_cast<uintptr_t>(3U)) == 0U;
  if (paired) {
    const auto *const activation_pairs =
        reinterpret_cast<const uint32_t *>(activation);
    const auto *const gate_pairs = reinterpret_cast<const uint32_t *>(gate_row);
    const auto *const up_pairs = reinterpret_cast<const uint32_t *>(up_row);
    const uint64_t pair_count = k / 2U;
    for (uint64_t pair = threadIdx.x; pair < pair_count; pair += blockDim.x) {
      const uint32_t activation_pair = activation_pairs[pair];
      const uint32_t gate_pair = __builtin_nontemporal_load(gate_pairs + pair);
      const uint32_t up_pair = __builtin_nontemporal_load(up_pairs + pair);
      gate_partial += bf16_to_float(static_cast<uint16_t>(activation_pair)) *
                      bf16_to_float(static_cast<uint16_t>(gate_pair));
      gate_partial +=
          bf16_to_float(static_cast<uint16_t>(activation_pair >> 16U)) *
          bf16_to_float(static_cast<uint16_t>(gate_pair >> 16U));
      up_partial += bf16_to_float(static_cast<uint16_t>(activation_pair)) *
                    bf16_to_float(static_cast<uint16_t>(up_pair));
      up_partial +=
          bf16_to_float(static_cast<uint16_t>(activation_pair >> 16U)) *
          bf16_to_float(static_cast<uint16_t>(up_pair >> 16U));
    }
  } else {
    for (uint64_t reduction = threadIdx.x; reduction < k;
         reduction += blockDim.x) {
      const float input = bf16_to_float(activation[reduction]);
      gate_partial += input * bf16_to_float(gate_row[reduction]);
      up_partial += input * bf16_to_float(up_row[reduction]);
    }
  }

#pragma unroll
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    gate_partial += __shfl_down(gate_partial, offset, 32U);
    up_partial += __shfl_down(up_partial, offset, 32U);
  }
  __shared__ float gate_wave_sums[8];
  __shared__ float up_wave_sums[8];
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  if (lane == 0U) {
    gate_wave_sums[wave] = gate_partial;
    up_wave_sums[wave] = up_partial;
  }
  __syncthreads();
  if (wave == 0U) {
    gate_partial = lane < 8U ? gate_wave_sums[lane] : 0.0F;
    up_partial = lane < 8U ? up_wave_sums[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
      gate_partial += __shfl_down(gate_partial, offset, 32U);
      up_partial += __shfl_down(up_partial, offset, 32U);
    }
    if (lane == 0U) {
      const uint16_t gate_bf16 = float_to_bf16_rne_bits(gate_partial);
      const uint16_t up_bf16 = float_to_bf16_rne_bits(up_partial);
      gate_output[column] = gate_bf16;
      up_output[column] = up_bf16;
      const float gate_value = bf16_to_float(gate_bf16);
      const float silu = gate_value / (1.0F + ::expf(-gate_value));
      const uint16_t silu_bf16 = float_to_bf16_rne_bits(silu);
      silu_output[column] = float_to_bf16_rne_bits(bf16_to_float(silu_bf16) *
                                                   bf16_to_float(up_bf16));
    }
  }
}

namespace sllm_mlp_gate_up_silu_bundle_kernel {

hipError_t launch(const uint16_t *const activation,
                  const uint16_t *const gate_weight,
                  const uint16_t *const up_weight, uint16_t *const gate_output,
                  uint16_t *const up_output, uint16_t *const silu_output,
                  const uint64_t k, const hipStream_t stream) noexcept {
  hipLaunchKernelGGL(sllm_mlp_gate_up_silu_bundle_bf16_fp32_decode_v1,
                     dim3(9216U), dim3(256U), 0U, stream, activation,
                     gate_weight, up_weight, gate_output, up_output,
                     silu_output, k, 9216U);
  return hipGetLastError();
}

} // namespace sllm_mlp_gate_up_silu_bundle_kernel
