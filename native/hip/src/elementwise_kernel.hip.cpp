#include "elementwise_kernel_internal.hpp"

#include <lowp/detail/low_precision_block_codec.hpp>

#include <cmath>
#include <cstdint>
#include <limits>

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

// The rounded BF16 activation the decomposed producer writes and the Phase 87
// prequant quantizer then reads. The standalone kernel and the fused variants
// compose this one helper, so both BF16 roundings stay bit-identical by
// construction rather than by duplicated arithmetic.
__device__ __forceinline__ uint16_t
silu_mul_activation(const uint16_t gate, const uint16_t up) noexcept {
  const float gate_value = bf16_to_float(gate);
  const float silu = gate_value / (1.0F + ::expf(-gate_value));
  const uint16_t silu_bf16 = float_to_bf16_rne_bits(silu);
  return float_to_bf16_rne_bits(bf16_to_float(silu_bf16) * bf16_to_float(up));
}

__device__ __forceinline__ uint16_t sigmoid_mul_activation(
    const uint16_t gate, const uint16_t attention_value) noexcept {
  const float gate_value = bf16_to_float(gate);
  const float sigmoid = 1.0F / (1.0F + ::expf(-gate_value));
  const uint16_t sigmoid_bf16 = float_to_bf16_rne_bits(sigmoid);
  return float_to_bf16_rne_bits(bf16_to_float(sigmoid_bf16) *
                                bf16_to_float(attention_value));
}

} // namespace

extern "C" __global__ __launch_bounds__(
    256, 1) void sllm_elementwise_copy_bf16_v1(const uint16_t *const input,
                                               uint16_t *const output,
                                               const uint64_t element_count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    output[index] = input[index];
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_elementwise_add_bf16_fp32_v1(
    const uint16_t *const input0, const uint16_t *const input1,
    uint16_t *const output, const uint64_t element_count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    output[index] = float_to_bf16_rne_bits(bf16_to_float(input0[index]) +
                                           bf16_to_float(input1[index]));
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_elementwise_silu_mul_bf16_fp32_v1(
    const uint16_t *const gate, const uint16_t *const up,
    uint16_t *const output, const uint64_t element_count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    output[index] = silu_mul_activation(gate[index], up[index]);
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_elementwise_sigmoid_mul_bf16_fp32_v1(
    const uint16_t *const gate, const uint16_t *const attention_value,
    uint16_t *const output, const uint64_t element_count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    output[index] = sigmoid_mul_activation(gate[index], attention_value[index]);
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_elementwise_scalar_mul_bf16_fp32_v1(
    const uint16_t *const input, const uint16_t *const scalar,
    uint16_t *const output, const uint64_t element_count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    output[index] = float_to_bf16_rne_bits(bf16_to_float(input[index]) *
                                           bf16_to_float(scalar[0]));
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_elementwise_gelu_tanh_mul_bf16_fp32_v1(
    const uint16_t *const gate, const uint16_t *const up,
    uint16_t *const output, const uint64_t element_count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    constexpr float sqrt_two_over_pi = 0.7978845608028654F;
    constexpr float cubic_coefficient = 0.044715F;
    const float value = bf16_to_float(gate[index]);
    const float inner =
        sqrt_two_over_pi * (value + cubic_coefficient * value * value * value);
    const float gelu = 0.5F * value * (1.0F + ::tanhf(inner));
    const uint16_t gelu_bf16 = float_to_bf16_rne_bits(gelu);
    output[index] = float_to_bf16_rne_bits(bf16_to_float(gelu_bf16) *
                                           bf16_to_float(up[index]));
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_elementwise_tanh_softcap_bf16_fp32_v1(
    const uint16_t *const input, const uint16_t *const cap,
    uint16_t *const output, const uint64_t element_count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    const float cap_value = bf16_to_float(cap[0]);
    output[index] = float_to_bf16_rne_bits(
        ::tanhf(bf16_to_float(input[index]) / cap_value) * cap_value);
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_elementwise_broadcast_add_bf16_fp32_v1(
    const uint16_t *const input, const uint16_t *const vector,
    uint16_t *const output, const uint64_t element_count,
    const uint64_t width) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    output[index] = float_to_bf16_rne_bits(
        bf16_to_float(input[index]) + bf16_to_float(vector[index % width]));
  }
}

extern "C" __global__
__launch_bounds__(256, 1) void sllm_elementwise_broadcast_mul_bf16_fp32_v1(
    const uint16_t *const input, const uint16_t *const vector,
    uint16_t *const output, const uint64_t element_count,
    const uint64_t width) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    output[index] = float_to_bf16_rne_bits(
        bf16_to_float(input[index]) * bf16_to_float(vector[index % width]));
  }
}

namespace {

// Phase 87 stage 7: producer-side activation quantization fusion.
//
// These variants fold the activation quantizer into the producer so the
// consumer matmul can skip its own quantize launch (one HIP graph node and
// one inter-kernel gap per removed launch). All variants reproduce the
// decomposed chain bit-for-bit: the standalone elementwise producer followed
// by `sllm_matmul_bf16_to_fp8_outer_v2` / `..._nvfp4_block16_wave8_v1`.
//
// The producer arithmetic comes from the `silu_mul_activation` /
// `sigmoid_mul_activation` helpers above and the FP8/E2M1 codecs from
// `lowp/detail/low_precision_block_codec.hpp`, so the math is shared with the
// standalone kernels rather than copied.
//
// `Produce` is a compile-time function pointer, so the SiLU-multiply and
// Sigmoid-multiply FP8 kernels instantiate the identical quantizer
// orchestration. The grid is one 1024-thread workgroup per row: the FP8
// quantizer is per-ROW, so the whole row must reduce before any encoding.
// That collapses decode m=1 to a single workgroup where the decomposed
// elementwise kernel used 68 — a possible performance loss that stage 7
// measures rather than assumes. The larger block keeps 32 waves resident to
// hide memory latency.
//
// Phase 2 recomputes the rounded BF16 activation instead of holding it in
// registers across the barrier: at k up to 17408 each of the 1024 threads
// would have to pin up to 17 BF16 values for an unbounded k, and the fused
// signature deliberately has no BF16 output buffer to stage into. This
// mirrors pass 3 of `sllm_rmsnorm_residual_prequant_fp8_v1`, which also
// recomputes. The fmaxf-based amax is order independent (NaN operands drop
// out immediately), so the 1024-thread reduction tree matches the 256-thread
// tree of `sllm_matmul_bf16_to_fp8_outer_v2` exactly.
template <uint16_t (*Produce)(const uint16_t, const uint16_t)>
__device__ __forceinline__ void elementwise_prequant_fp8_body(
    const uint16_t *const gate, const uint16_t *const partner,
    uint8_t *const quantized, float *const activation_scales, const uint32_t m,
    const uint32_t k) noexcept {
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t wave_count = 32U;
  __shared__ float wave_maxima[wave_count];
  __shared__ float shared_scale;
  const uint32_t lane = threadIdx.x & UINT32_C(31);
  const uint32_t wave = threadIdx.x >> 5U;
  const uint64_t row = static_cast<uint64_t>(blockIdx.x);
  if (row >= m) {
    return;
  }
  const uint64_t row_offset = row * static_cast<uint64_t>(k);

  // Phase 1: byte-identical to the standalone producer per element, folded
  // into the row amax that sllm_matmul_bf16_to_fp8_outer_v2 derives from the
  // BF16 activation buffer. The seed stays at 0.0F so an all-NaN row reduces
  // to 0 exactly as the decomposed quantizer does.
  float maximum = 0.0F;
  for (uint32_t column = threadIdx.x; column < k; column += blockDim.x) {
    const float value = bf16_to_float(
        Produce(gate[row_offset + column], partner[row_offset + column]));
    maximum = fmaxf(maximum, fabsf(value));
  }
  for (uint32_t delta = wave_width / 2U; delta != 0U; delta >>= 1U) {
    maximum = fmaxf(maximum, __shfl_down(maximum, delta, wave_width));
  }
  if (lane == 0U) {
    wave_maxima[wave] = maximum;
  }
  __syncthreads();
  if (wave == 0U) {
    // 32 waves each feed one lane of wave 0, so no lane needs a zero pad.
    maximum = wave_maxima[lane];
    for (uint32_t delta = wave_width / 2U; delta != 0U; delta >>= 1U) {
      maximum = fmaxf(maximum, __shfl_down(maximum, delta, wave_width));
    }
    if (lane == 0U) {
      shared_scale = maximum == 0.0F ? 1.0F : maximum / 448.0F;
      activation_scales[row] = shared_scale;
    }
  }
  __syncthreads();

  // Phase 2: recompute the rounded BF16 activation and encode it exactly as
  // sllm_matmul_bf16_to_fp8_outer_v2 does on the standalone activation
  // buffer. OCP E4M3FN only; FNUZ targets route to the v1 quantizer and are
  // out of scope, so fnuz is a constant false here.
  if ((k & UINT32_C(1)) == 0U) {
    const uint64_t pairs = static_cast<uint64_t>(k) / UINT64_C(2);
    for (uint64_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
      const uint64_t first_column = pair * UINT64_C(2);
      const uint64_t second_column = first_column + UINT64_C(1);
      const float first =
          bf16_to_float(Produce(gate[row_offset + first_column],
                                partner[row_offset + first_column])) /
          shared_scale;
      const float second =
          bf16_to_float(Produce(gate[row_offset + second_column],
                                partner[row_offset + second_column])) /
          shared_scale;
      uint16_t packed;
      if (isfinite(first) && isfinite(second)) {
        packed = __hip_cvt_float2_to_fp8x2(make_float2(first, second),
                                           __HIP_SATFINITE, __HIP_E4M3);
      } else {
        packed = static_cast<uint16_t>(
            static_cast<uint16_t>(
                sllm_lowp::float_to_fp8_native(first, false)) |
            (static_cast<uint16_t>(
                 sllm_lowp::float_to_fp8_native(second, false))
             << 8U));
      }
      reinterpret_cast<uint16_t *>(quantized + row_offset)[pair] = packed;
    }
  } else {
    for (uint32_t column = threadIdx.x; column < k; column += blockDim.x) {
      const float value = bf16_to_float(Produce(gate[row_offset + column],
                                                partner[row_offset + column])) /
                          shared_scale;
      quantized[row_offset + column] =
          sllm_lowp::float_to_fp8_native(value, false);
    }
  }
}

} // namespace

extern "C" __global__
__launch_bounds__(1024, 1) void sllm_elementwise_silu_mul_prequant_fp8_v1(
    const uint16_t *const gate, const uint16_t *const up,
    uint8_t *const quantized, float *const activation_scales, const uint32_t m,
    const uint32_t k) {
  elementwise_prequant_fp8_body<silu_mul_activation>(gate, up, quantized,
                                                     activation_scales, m, k);
}

extern "C" __global__
__launch_bounds__(1024, 1) void sllm_elementwise_sigmoid_mul_prequant_fp8_v1(
    const uint16_t *const gate, const uint16_t *const attention_value,
    uint8_t *const quantized, float *const activation_scales, const uint32_t m,
    const uint32_t k) {
  elementwise_prequant_fp8_body<sigmoid_mul_activation>(
      gate, attention_value, quantized, activation_scales, m, k);
}

// NVFP4 quantization is per-16-element block, so this variant stays
// element-parallel: 16 lane groups of 16 lanes each cover 16 blocks per
// 256-thread workgroup, and grid = ceil(m * ceil(k/16) / 16). For m=1,
// k=17408 that is 68 workgroups — exactly the workgroup count of the
// decomposed elementwise kernel, so no parallelism is lost. Each group
// mirrors sllm_matmul_bf16_to_nvfp4_block16_wave8_v1 lane-for-lane through a
// width-16 reduce: the decomposed quantizer pads lanes 16..31 with zero
// seeds inside its wave32, which the fmaxf(0, ·) seed below reproduces, so
// an all-NaN group reduces to 0 and stays in sync with the packed codes.
extern "C" __global__
__launch_bounds__(256, 1) void sllm_elementwise_silu_mul_prequant_nvfp4_v1(
    const uint16_t *const gate, const uint16_t *const up,
    uint8_t *const packed_activation, uint8_t *const activation_block_scales,
    const float *const input_tensor_scale, const uint32_t m, const uint32_t k) {
  constexpr uint32_t group_width = 16U;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_in_group = tid % group_width;
  const uint32_t group = tid / group_width;
  const uint32_t group_count = blockDim.x / group_width;
  const uint64_t blocks_per_row =
      (static_cast<uint64_t>(k) + UINT64_C(15)) / UINT64_C(16);
  const uint64_t total_blocks = static_cast<uint64_t>(m) * blocks_per_row;
  const uint64_t packed_row_bytes =
      (static_cast<uint64_t>(k) + UINT64_C(1)) / UINT64_C(2);
  const uint64_t global_block =
      static_cast<uint64_t>(blockIdx.x) * group_count + group;
  if (global_block >= total_blocks) {
    return;
  }
  const uint64_t row = global_block / blocks_per_row;
  const uint64_t block = global_block - row * blocks_per_row;
  const uint64_t base = block * UINT64_C(16);
  const uint64_t row_offset = row * static_cast<uint64_t>(k);
  const uint64_t column = base + lane_in_group;
  const float value =
      column < k ? bf16_to_float(silu_mul_activation(gate[row_offset + column],
                                                     up[row_offset + column]))
                 : 0.0F;
  // Seed through fmaxf(0, ·) so a NaN element contributes nothing, exactly
  // as the decomposed quantizer's unused lanes 16..31 do.
  float maximum = fmaxf(0.0F, fabsf(value));
  for (uint32_t delta = group_width / 2U; delta != 0U; delta >>= 1U) {
    maximum = fmaxf(maximum, __shfl_down(maximum, delta, group_width));
  }
  const float global = input_tensor_scale[0];
  // Shuffle a full dword. HIP's byte-sized shuffle overload can leave the
  // upper lanes implementation-defined on wave32 targets and produced an
  // incorrect decoded scale in the tail cases.
  uint32_t encoded_scale_bits = 0U;
  if (lane_in_group == 0U) {
    const float raw_scale_value =
        maximum == 0.0F || !(global > 0.0F) ? 0.0F : maximum / (6.0F * global);
    encoded_scale_bits = sllm_lowp::float_to_e4m3fn(raw_scale_value);
    activation_block_scales[row * blocks_per_row + block] =
        static_cast<uint8_t>(encoded_scale_bits);
  }
  encoded_scale_bits = __shfl(encoded_scale_bits, 0, group_width);
  const uint8_t encoded_scale = static_cast<uint8_t>(encoded_scale_bits);
  const float decoded_scale =
      sllm_lowp::e4m3fn_to_float(encoded_scale) * global;
  // Cross-lane reads must execute under the full group mask. Masking lanes
  // 8..15 before they serve as shuffle sources returns undefined data.
  const uint32_t pair_lane = lane_in_group & 7U;
  const uint32_t shuffled_first = pair_lane * 2U;
  const float first_value =
      __shfl(value, static_cast<int>(shuffled_first), group_width);
  const float second_value =
      __shfl(value, static_cast<int>(shuffled_first + 1U), group_width);
  if (lane_in_group < 8U) {
    const uint32_t first = lane_in_group * 2U;
    const uint64_t first_column = base + first;
    const uint64_t second_column = first_column + UINT64_C(1);
    const uint8_t low =
        first_column < k && decoded_scale > 0.0F
            ? sllm_lowp::float_to_e2m1(first_value / decoded_scale)
            : 0U;
    const uint8_t high =
        second_column < k && decoded_scale > 0.0F
            ? sllm_lowp::float_to_e2m1(second_value / decoded_scale)
            : 0U;
    if (first_column < k) {
      packed_activation[row * packed_row_bytes + first_column / UINT64_C(2)] =
          static_cast<uint8_t>(low | static_cast<uint8_t>(high << 4U));
    }
  }
}

namespace sllm_elementwise_kernel {
namespace {

bool grid_for(const uint64_t element_count, dim3 *const grid) noexcept {
  if (grid == nullptr || element_count == 0U) {
    return false;
  }
  const uint64_t workgroup = static_cast<uint64_t>(kWorkgroupSize);
  const uint64_t blocks =
      element_count / workgroup +
      static_cast<uint64_t>(element_count % workgroup != 0U);
  if (blocks > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  *grid = dim3(static_cast<uint32_t>(blocks), 1U, 1U);
  return true;
}

} // namespace

hipError_t launch_copy(const uint16_t *const input, uint16_t *const output,
                       const uint64_t element_count,
                       const hipStream_t stream) noexcept {
  dim3 grid;
  if (!grid_for(element_count, &grid)) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_elementwise_copy_bf16_v1, grid, dim3(kWorkgroupSize),
                     0U, stream, input, output, element_count);
  return hipGetLastError();
}

hipError_t launch_add(const uint16_t *const input0,
                      const uint16_t *const input1, uint16_t *const output,
                      const uint64_t element_count,
                      const hipStream_t stream) noexcept {
  dim3 grid;
  if (!grid_for(element_count, &grid)) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_elementwise_add_bf16_fp32_v1, grid,
                     dim3(kWorkgroupSize), 0U, stream, input0, input1, output,
                     element_count);
  return hipGetLastError();
}

hipError_t launch_silu_mul(const uint16_t *const gate, const uint16_t *const up,
                           uint16_t *const output, const uint64_t element_count,
                           const hipStream_t stream) noexcept {
  dim3 grid;
  if (!grid_for(element_count, &grid)) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_elementwise_silu_mul_bf16_fp32_v1, grid,
                     dim3(kWorkgroupSize), 0U, stream, gate, up, output,
                     element_count);
  return hipGetLastError();
}

hipError_t launch_sigmoid_mul(const uint16_t *const gate,
                              const uint16_t *const attention_value,
                              uint16_t *const output,
                              const uint64_t element_count,
                              const hipStream_t stream) noexcept {
  dim3 grid;
  if (!grid_for(element_count, &grid)) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_elementwise_sigmoid_mul_bf16_fp32_v1, grid,
                     dim3(kWorkgroupSize), 0U, stream, gate, attention_value,
                     output, element_count);
  return hipGetLastError();
}

hipError_t launch_scalar_mul(const uint16_t *const input,
                             const uint16_t *const scalar,
                             uint16_t *const output,
                             const uint64_t element_count,
                             const hipStream_t stream) noexcept {
  dim3 grid;
  if (!grid_for(element_count, &grid)) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_elementwise_scalar_mul_bf16_fp32_v1, grid,
                     dim3(kWorkgroupSize), 0U, stream, input, scalar, output,
                     element_count);
  return hipGetLastError();
}

hipError_t launch_gelu_tanh_mul(const uint16_t *const gate,
                                const uint16_t *const up,
                                uint16_t *const output,
                                const uint64_t element_count,
                                const hipStream_t stream) noexcept {
  dim3 grid;
  if (!grid_for(element_count, &grid)) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_elementwise_gelu_tanh_mul_bf16_fp32_v1, grid,
                     dim3(kWorkgroupSize), 0U, stream, gate, up, output,
                     element_count);
  return hipGetLastError();
}

hipError_t launch_tanh_softcap(const uint16_t *const input,
                               const uint16_t *const cap,
                               uint16_t *const output,
                               const uint64_t element_count,
                               const hipStream_t stream) noexcept {
  dim3 grid;
  if (!grid_for(element_count, &grid)) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_elementwise_tanh_softcap_bf16_fp32_v1, grid,
                     dim3(kWorkgroupSize), 0U, stream, input, cap, output,
                     element_count);
  return hipGetLastError();
}

hipError_t
launch_broadcast_add(const uint16_t *const input, const uint16_t *const vector,
                     uint16_t *const output, const uint64_t element_count,
                     const uint64_t width, const hipStream_t stream) noexcept {
  dim3 grid;
  if (!grid_for(element_count, &grid) || width == 0U) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_elementwise_broadcast_add_bf16_fp32_v1, grid,
                     dim3(kWorkgroupSize), 0U, stream, input, vector, output,
                     element_count, width);
  return hipGetLastError();
}

hipError_t
launch_broadcast_mul(const uint16_t *const input, const uint16_t *const vector,
                     uint16_t *const output, const uint64_t element_count,
                     const uint64_t width, const hipStream_t stream) noexcept {
  dim3 grid;
  if (!grid_for(element_count, &grid) || width == 0U) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_elementwise_broadcast_mul_bf16_fp32_v1, grid,
                     dim3(kWorkgroupSize), 0U, stream, input, vector, output,
                     element_count, width);
  return hipGetLastError();
}

// Phase 87 stage 7: producer-side activation quantization launchers. The FP8
// variants use one 1024-thread workgroup per row because the row amax must
// reduce before encoding; the NVFP4 variant keeps the element-parallel
// 256-thread workgroup of the decomposed elementwise kernel.
hipError_t launch_silu_mul_prequant_fp8(const uint16_t *const gate,
                                        const uint16_t *const up,
                                        uint8_t *const quantized,
                                        float *const activation_scales,
                                        const uint32_t m, const uint32_t k,
                                        const hipStream_t stream) noexcept {
  if (m == 0U || k == 0U) {
    return hipErrorInvalidValue;
  }
  const dim3 grid(m, 1U, 1U);
  const dim3 block(1024U, 1U, 1U);
  hipLaunchKernelGGL(sllm_elementwise_silu_mul_prequant_fp8_v1, grid, block, 0U,
                     stream, gate, up, quantized, activation_scales, m, k);
  return hipGetLastError();
}

hipError_t launch_silu_mul_prequant_nvfp4(
    const uint16_t *const gate, const uint16_t *const up,
    uint8_t *const packed_activation, uint8_t *const activation_block_scales,
    const float *const input_tensor_scale, const uint32_t m, const uint32_t k,
    const hipStream_t stream) noexcept {
  if (m == 0U || k == 0U) {
    return hipErrorInvalidValue;
  }
  const uint64_t blocks_per_row =
      (static_cast<uint64_t>(k) + UINT64_C(15)) / UINT64_C(16);
  const uint64_t total_blocks = static_cast<uint64_t>(m) * blocks_per_row;
  // 16 lane groups of 16 lanes cover 16 blocks per workgroup; round up so
  // the tail group idles on its bounds check instead of losing a block.
  const uint64_t workgroups = (total_blocks + UINT64_C(15)) / UINT64_C(16);
  if (workgroups > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidValue;
  }
  const dim3 grid(static_cast<uint32_t>(workgroups), 1U, 1U);
  const dim3 block(256U, 1U, 1U);
  hipLaunchKernelGGL(sllm_elementwise_silu_mul_prequant_nvfp4_v1, grid, block,
                     0U, stream, gate, up, packed_activation,
                     activation_block_scales, input_tensor_scale, m, k);
  return hipGetLastError();
}

hipError_t launch_sigmoid_mul_prequant_fp8(
    const uint16_t *const gate, const uint16_t *const attention_value,
    uint8_t *const quantized, float *const activation_scales, const uint32_t m,
    const uint32_t k, const hipStream_t stream) noexcept {
  if (m == 0U || k == 0U) {
    return hipErrorInvalidValue;
  }
  const dim3 grid(m, 1U, 1U);
  const dim3 block(1024U, 1U, 1U);
  hipLaunchKernelGGL(sllm_elementwise_sigmoid_mul_prequant_fp8_v1, grid, block,
                     0U, stream, gate, attention_value, quantized,
                     activation_scales, m, k);
  return hipGetLastError();
}

} // namespace sllm_elementwise_kernel
