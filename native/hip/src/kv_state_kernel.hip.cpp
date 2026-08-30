#include "kv_state_kernel_internal.hpp"

#include <cmath>
#include <cstdint>

namespace {

using Bf16Input = const uint16_t *;

__device__ __forceinline__ uint16_t float_bits_to_f16(const uint32_t bits) {
  const uint32_t sign = (bits >> 16U) & UINT32_C(0x8000);
  const uint32_t exponent = (bits >> 23U) & UINT32_C(0xff);
  const uint32_t fraction = bits & UINT32_C(0x7fffff);
  if (exponent == UINT32_C(0xff)) {
    if (fraction == 0U) {
      return static_cast<uint16_t>(sign | UINT32_C(0x7c00));
    }
    return static_cast<uint16_t>(sign | UINT32_C(0x7e00));
  }
  const int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<uint16_t>(sign | UINT32_C(0x7c00));
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    const uint32_t mantissa = fraction | UINT32_C(0x800000);
    const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((UINT32_C(1) << shift) - 1U);
    const uint32_t halfway = UINT32_C(1) << (shift - 1U);
    if (remainder > halfway ||
        (remainder == halfway && (rounded & UINT32_C(1)) != 0U)) {
      ++rounded;
    }
    return static_cast<uint16_t>(sign | rounded);
  }
  uint32_t rounded_fraction = fraction >> 13U;
  const uint32_t remainder = fraction & UINT32_C(0x1fff);
  if (remainder > UINT32_C(0x1000) ||
      (remainder == UINT32_C(0x1000) &&
       (rounded_fraction & UINT32_C(1)) != 0U)) {
    ++rounded_fraction;
    if (rounded_fraction == UINT32_C(0x400)) {
      rounded_fraction = 0U;
      if (half_exponent + 1 >= 31) {
        return static_cast<uint16_t>(sign | UINT32_C(0x7c00));
      }
      return static_cast<uint16_t>(
          sign | (static_cast<uint32_t>(half_exponent + 1) << 10U));
    }
  }
  return static_cast<uint16_t>(
      sign | (static_cast<uint32_t>(half_exponent) << 10U) | rounded_fraction);
}

__device__ __forceinline__ uint16_t bf16_to_f16(const uint16_t value) {
  return float_bits_to_f16(static_cast<uint32_t>(value) << 16U);
}

__device__ __forceinline__ float bf16_to_float(const uint16_t value) {
  return __uint_as_float(static_cast<uint32_t>(value) << 16U);
}

__device__ __forceinline__ float e4m3fn_to_float(const uint8_t bits) {
  const float sign = (bits & UINT8_C(0x80)) == 0U ? 1.0F : -1.0F;
  const uint8_t exponent = static_cast<uint8_t>((bits >> 3U) & 0x0fU);
  const uint8_t mantissa = static_cast<uint8_t>(bits & 0x07U);
  if (exponent == 0U) {
    return mantissa == 0U
               ? copysignf(0.0F, sign)
               : sign * static_cast<float>(mantissa) * ldexpf(1.0F, -9);
  }
  if (exponent == 0x0fU && mantissa == 0x07U) {
    return NAN;
  }
  return sign * (1.0F + static_cast<float>(mantissa) / 8.0F) *
         ldexpf(1.0F, static_cast<int>(exponent) - 7);
}

__device__ __forceinline__ uint8_t float_to_e4m3fn(float value) {
  const uint8_t sign = signbit(value) ? UINT8_C(0x80) : 0U;
  if (isnan(value)) {
    return UINT8_C(0x7f);
  }
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
  const bool upper_selected =
      upper_error < lower_error ||
      (upper_error == lower_error && (upper & 1U) == 0U && (lower & 1U) != 0U);
  return static_cast<uint8_t>(sign | (upper_selected ? upper : lower));
}

__device__ __forceinline__ uint8_t
float_to_e4m3fn_fp8_append(const float value) {
#if defined(__gfx1201__)
  if (isnan(value)) {
    return UINT8_C(0x7f);
  }
  const uint8_t sign = signbit(value) ? UINT8_C(0x80) : 0U;
  const float magnitude = fabsf(value);
  if (magnitude == 0.0F) {
    return sign;
  }
  if (!isfinite(magnitude) || magnitude >= 448.0F) {
    return static_cast<uint8_t>(sign | UINT8_C(0x7e));
  }
  const uint32_t packed = static_cast<uint32_t>(
      __builtin_amdgcn_cvt_pk_fp8_f32(value, value, 0, false));
  return static_cast<uint8_t>(packed & UINT32_C(0xff));
#else
  return float_to_e4m3fn(value);
#endif
}

__device__ __forceinline__ float e4m3fnuz_to_float(const uint8_t bits) {
  if (bits == UINT8_C(0x80)) {
    return NAN;
  }
  const float sign = (bits & UINT8_C(0x80)) == 0U ? 1.0F : -1.0F;
  const uint8_t exponent = static_cast<uint8_t>((bits >> 3U) & 0x0fU);
  const uint8_t mantissa = static_cast<uint8_t>(bits & 0x07U);
  if (exponent == 0U) {
    return sign * static_cast<float>(mantissa) * ldexpf(1.0F, -10);
  }
  return sign * (1.0F + static_cast<float>(mantissa) / 8.0F) *
         ldexpf(1.0F, static_cast<int>(exponent) - 8);
}

__device__ __forceinline__ uint8_t float_to_e4m3fnuz(float value) {
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
  uint32_t low = 0U;
  uint32_t high = UINT32_C(0x7f);
  while (low < high) {
    const uint32_t middle = (low + high) >> 1U;
    if (e4m3fnuz_to_float(static_cast<uint8_t>(middle)) < value) {
      low = middle + 1U;
    } else {
      high = middle;
    }
  }
  const uint8_t upper = static_cast<uint8_t>(low);
  const uint8_t lower = upper == 0U ? 0U : static_cast<uint8_t>(upper - 1U);
  const float lower_error = value - e4m3fnuz_to_float(lower);
  const float upper_error = e4m3fnuz_to_float(upper) - value;
  const uint8_t selected =
      upper_error < lower_error || (upper_error == lower_error &&
                                    (upper & 1U) == 0U && (lower & 1U) != 0U)
          ? upper
          : lower;
  return negative && selected != 0U ? static_cast<uint8_t>(selected | 0x80U)
                                    : selected;
}

__device__ __forceinline__ float e5m2_to_float(const uint8_t bits) {
  const float sign = (bits & UINT8_C(0x80)) == 0U ? 1.0F : -1.0F;
  const uint8_t exponent = static_cast<uint8_t>((bits >> 2U) & 0x1fU);
  const uint8_t mantissa = static_cast<uint8_t>(bits & 0x03U);
  if (exponent == 0U) {
    return mantissa == 0U
               ? copysignf(0.0F, sign)
               : sign * static_cast<float>(mantissa) * ldexpf(1.0F, -16);
  }
  if (exponent == UINT8_C(0x1f)) {
    return mantissa == 0U ? copysignf(INFINITY, sign) : NAN;
  }
  return sign * (1.0F + static_cast<float>(mantissa) / 4.0F) *
         ldexpf(1.0F, static_cast<int>(exponent) - 15);
}

__device__ __forceinline__ uint8_t float_to_e5m2(float value) {
  if (isnan(value)) {
    return UINT8_C(0x7f);
  }
  const uint8_t sign = signbit(value) ? UINT8_C(0x80) : 0U;
  value = fabsf(value);
  if (value == 0.0F) {
    return sign;
  }
  if (!isfinite(value) || value >= 57344.0F) {
    return static_cast<uint8_t>(sign | UINT8_C(0x7b));
  }
  uint32_t low = 0U;
  uint32_t high = UINT32_C(0x7b);
  while (low < high) {
    const uint32_t middle = (low + high) >> 1U;
    if (e5m2_to_float(static_cast<uint8_t>(middle)) < value) {
      low = middle + 1U;
    } else {
      high = middle;
    }
  }
  const uint8_t upper = static_cast<uint8_t>(low);
  const uint8_t lower = upper == 0U ? 0U : static_cast<uint8_t>(upper - 1U);
  const float lower_error = value - e5m2_to_float(lower);
  const float upper_error = e5m2_to_float(upper) - value;
  const bool upper_selected =
      upper_error < lower_error ||
      (upper_error == lower_error && (upper & 1U) == 0U && (lower & 1U) != 0U);
  return static_cast<uint8_t>(sign | (upper_selected ? upper : lower));
}

__device__ __forceinline__ float e8m0_to_float(const uint8_t bits) {
  if (bits == UINT8_C(0xff)) {
    return NAN;
  }
  return bits == 0U ? __uint_as_float(UINT32_C(0x00400000))
                    : __uint_as_float(static_cast<uint32_t>(bits) << 23U);
}

// OCP MX uses one E8M0 power-of-two scale for each block.  The exponent is
// the block amax's floor power-of-two divided by the largest power-of-two
// representable by the element format (2^8 for E4M3, 2^15 for E5M2), then
// clamped to the finite E8M0 exponent range.  This rule is shared by the
// block16 KV format and standard MXFP8; physical E4 FNUZ encoding remains
// independent of the scale calculation.
__device__ __forceinline__ uint8_t ocp_mxfp8_e8m0_scale(const float maximum,
                                                        const bool e5) {
  if (!(maximum > 0.0F) || !isfinite(maximum)) {
    return UINT8_C(127);
  }
  const int32_t floor_exponent = ilogbf(maximum);
  const int32_t denominator_exponent = e5 ? 15 : 8;
  const int32_t scale_exponent = floor_exponent - denominator_exponent;
  const int32_t clamped = max(-127, min(127, scale_exponent));
  return static_cast<uint8_t>(clamped + 127);
}

#if SLLM_ENABLE_PHASE54_KV_RESEARCH
__device__ __forceinline__ uint8_t phase54_block16_e8m0_scale(
    const float maximum, const bool e5, const uint32_t recipe) {
  if (!(maximum > 0.0F) || !isfinite(maximum)) {
    return UINT8_C(127);
  }
  const int32_t denominator_exponent = e5 ? 15 : 8;
  const int32_t floor_maximum_exponent = ilogbf(maximum);
  int32_t selected_maximum_exponent = floor_maximum_exponent;
  if (recipe == sllm_kv_state_kernel::kPhase54KvRecipeCeilExponent) {
    if (maximum != ldexpf(1.0F, floor_maximum_exponent)) {
      ++selected_maximum_exponent;
    }
  } else if (recipe ==
             sllm_kv_state_kernel::kPhase54KvRecipeNearestEvenExponent) {
    // Compare against the geometric midpoint between adjacent powers of two.
    // At the exact midpoint, round the resulting scale exponent to even.
    constexpr float kSqrtTwo = 0x1.6a09e6p+0F;
    const float normalized = ldexpf(maximum, -floor_maximum_exponent);
    const int32_t floor_scale_exponent =
        floor_maximum_exponent - denominator_exponent;
    if (normalized > kSqrtTwo ||
        (normalized == kSqrtTwo && (floor_scale_exponent & 1) != 0)) {
      ++selected_maximum_exponent;
    }
  }
  const int32_t scale_exponent =
      selected_maximum_exponent - denominator_exponent;
  const int32_t clamped = max(-127, min(127, scale_exponent));
  return static_cast<uint8_t>(clamped + 127);
}
#endif

__device__ __forceinline__ uint8_t float_to_e2m1(float value) {
  constexpr float positive[8] = {0.0F, 0.5F, 1.0F, 1.5F,
                                 2.0F, 3.0F, 4.0F, 6.0F};
  const uint8_t sign = signbit(value) ? UINT8_C(0x08) : 0U;
  if (isnan(value)) {
    return sign;
  }
  const float magnitude = fminf(fabsf(value), 6.0F);
  uint8_t best = 0U;
  float best_error = INFINITY;
  for (uint8_t candidate = 0U; candidate != 8U; ++candidate) {
    const float error = fabsf(positive[candidate] - magnitude);
    if (error < best_error ||
        (error == best_error && (candidate & 1U) == 0U && (best & 1U) != 0U)) {
      best = candidate;
      best_error = error;
    }
  }
  return static_cast<uint8_t>(sign | best);
}

} // namespace

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_KV_WORKGROUP_SIZE,
    1) void sllm_kv_state_bf16_to_fp8_token_major_v1(Bf16Input key_input,
                                                     Bf16Input value_input,
                                                     uint8_t *const key_output,
                                                     uint8_t
                                                         *const value_output,
                                                     float *const key_scales,
                                                     float *const value_scales,
                                                     const uint32_t token_count,
                                                     const uint64_t
                                                         start_position,
                                                     const uint32_t head_count,
                                                     const uint32_t head_dim,
                                                     const float
                                                         fixed_key_scale,
                                                     const float
                                                         fixed_value_scale,
                                                     const uint32_t
                                                         static_mode) {
  const uint64_t row = blockIdx.x;
  if (row >= static_cast<uint64_t>(token_count) * head_count) {
    return;
  }
  const uint64_t input_base = row * head_dim;
  const uint64_t token = row / head_count;
  const uint64_t head = row % head_count;
  const uint64_t output_row = (start_position + token) * head_count + head;
  const uint64_t output_base = output_row * head_dim;
  __shared__ float key_maxima[SLLM_HIP_KV_WORKGROUP_SIZE];
  __shared__ float value_maxima[SLLM_HIP_KV_WORKGROUP_SIZE];
  const uint32_t dimension = threadIdx.x;
  float key_maximum = 0.0F;
  float value_maximum = 0.0F;
  for (uint32_t current = dimension; current < head_dim;
       current += blockDim.x) {
    const float key_value = bf16_to_float(key_input[input_base + current]);
    const float value_value = bf16_to_float(value_input[input_base + current]);
    key_maximum =
        fmaxf(key_maximum, isfinite(key_value) ? fabsf(key_value) : 0.0F);
    value_maximum =
        fmaxf(value_maximum, isfinite(value_value) ? fabsf(value_value) : 0.0F);
  }
  key_maxima[dimension] = key_maximum;
  value_maxima[dimension] = value_maximum;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2U; stride != 0U; stride >>= 1U) {
    if (dimension < stride) {
      key_maxima[dimension] =
          fmaxf(key_maxima[dimension], key_maxima[dimension + stride]);
      value_maxima[dimension] =
          fmaxf(value_maxima[dimension], value_maxima[dimension + stride]);
    }
    __syncthreads();
  }
  const float key_scale =
      static_mode != 0U
          ? fixed_key_scale
          : (key_maxima[0] == 0.0F ? 1.0F : key_maxima[0] / 448.0F);
  const float value_scale =
      static_mode != 0U
          ? fixed_value_scale
          : (value_maxima[0] == 0.0F ? 1.0F : value_maxima[0] / 448.0F);
  // Static FP8 keeps one checked scale pair in the state descriptor and has
  // no per-row scale planes. Dynamic FP8 alone publishes row scales here.
  if (dimension == 0U && static_mode == 0U) {
    key_scales[output_row] = key_scale;
    value_scales[output_row] = value_scale;
  }
  for (uint32_t current = dimension; current < head_dim;
       current += blockDim.x) {
    key_output[output_base + current] = float_to_e4m3fn_fp8_append(
        bf16_to_float(key_input[input_base + current]) / key_scale);
    value_output[output_base + current] = float_to_e4m3fn_fp8_append(
        bf16_to_float(value_input[input_base + current]) / value_scale);
  }
}

template <bool Key>
__device__ void
quantize_fp8_block16_row(const uint16_t *const input, uint8_t *const output,
                         uint8_t *const scales, const uint64_t input_base,
                         const uint64_t output_row, const uint32_t head_dim,
                         const uint32_t encoding) {
  if (threadIdx.x != 0U) {
    return;
  }
  const uint64_t blocks_per_row = (static_cast<uint64_t>(head_dim) + 15U) / 16U;
  const uint64_t padded_head_dim = blocks_per_row * 16U;
  const bool e5 = encoding == SLLM_HIP_KV_ENCODING_FP8_E5_BLOCK16_V1 ||
                  encoding == SLLM_HIP_KV_ENCODING_FP8_E5_BLOCK16_V2;
#if defined(__gfx942__)
  constexpr bool e4_fnuz = true;
#else
  constexpr bool e4_fnuz = false;
#endif
  for (uint64_t block = 0U; block != blocks_per_row; ++block) {
    const uint32_t begin = static_cast<uint32_t>(block * 16U);
    const uint32_t end = min(begin + 16U, head_dim);
    float maximum = 0.0F;
    bool all_zero = true;
    for (uint32_t dimension = begin; dimension != end; ++dimension) {
      const float value = bf16_to_float(input[input_base + dimension]);
      all_zero = all_zero && value == 0.0F;
      if (isfinite(value)) {
        maximum = fmaxf(maximum, fabsf(value));
      }
    }
    const uint8_t scale_bits = ocp_mxfp8_e8m0_scale(maximum, e5);
    const float scale = e8m0_to_float(scale_bits);
    scales[output_row * blocks_per_row + block] = scale_bits;
    for (uint32_t dimension = begin; dimension != end; ++dimension) {
      if (all_zero) {
        output[output_row * padded_head_dim + dimension] = 0U;
        continue;
      }
      const float normalized =
          bf16_to_float(input[input_base + dimension]) / scale;
      output[output_row * padded_head_dim + dimension] =
          e5 ? float_to_e5m2(normalized)
             : (e4_fnuz ? float_to_e4m3fnuz(normalized)
                        : float_to_e4m3fn_fp8_append(normalized));
    }
    // Physical rows are block padded.  Clear the invalid tail lanes on every
    // append so stale bytes from a prior resident row cannot leak into the
    // canonical state image.
    for (uint32_t dimension = end;
         dimension != static_cast<uint32_t>((block + 1U) * 16U); ++dimension) {
      output[output_row * padded_head_dim + dimension] = 0U;
    }
  }
  (void)Key;
}

template <bool E5>
__device__ void
quantize_mxfp8_pair(const uint16_t *const key_input,
                    const uint16_t *const value_input,
                    uint8_t *const key_output, uint8_t *const value_output,
                    uint8_t *const key_scales, uint8_t *const value_scales,
                    const uint64_t input_base, const uint64_t output_row,
                    const uint32_t head_dim) {
  constexpr uint32_t kBlockSize = 32U;
  constexpr uint32_t kWaveSize = 32U;
  const uint64_t blocks_per_row =
      (static_cast<uint64_t>(head_dim) + kBlockSize - 1U) / kBlockSize;
  const uint64_t padded_head_dim = blocks_per_row * kBlockSize;
  const uint32_t lane = threadIdx.x & (kWaveSize - 1U);
  const uint32_t wave = threadIdx.x / kWaveSize;
  const uint32_t wave_count = blockDim.x / kWaveSize;
  for (uint64_t block = wave; block < blocks_per_row; block += wave_count) {
    const uint32_t dimension = static_cast<uint32_t>(block * kBlockSize) + lane;
    const bool active = dimension < head_dim;
    const float key_value =
        active ? bf16_to_float(key_input[input_base + dimension]) : 0.0F;
    const float value_value =
        active ? bf16_to_float(value_input[input_base + dimension]) : 0.0F;
    float key_maximum = isfinite(key_value) ? fabsf(key_value) : 0.0F;
    float value_maximum = isfinite(value_value) ? fabsf(value_value) : 0.0F;
    uint32_t key_all_zero = !active || key_value == 0.0F ? 1U : 0U;
    uint32_t value_all_zero = !active || value_value == 0.0F ? 1U : 0U;
    for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
      key_maximum =
          fmaxf(key_maximum, __shfl_down(key_maximum, offset, kWaveSize));
      value_maximum =
          fmaxf(value_maximum, __shfl_down(value_maximum, offset, kWaveSize));
      key_all_zero &= __shfl_down(key_all_zero, offset, kWaveSize);
      value_all_zero &= __shfl_down(value_all_zero, offset, kWaveSize);
    }
    uint32_t key_scale_bits = 0U;
    uint32_t value_scale_bits = 0U;
    float key_scale = 1.0F;
    float value_scale = 1.0F;
    if (lane == 0U) {
      key_scale_bits = ocp_mxfp8_e8m0_scale(key_maximum, E5);
      value_scale_bits = ocp_mxfp8_e8m0_scale(value_maximum, E5);
      key_scales[output_row * blocks_per_row + block] =
          static_cast<uint8_t>(key_scale_bits);
      value_scales[output_row * blocks_per_row + block] =
          static_cast<uint8_t>(value_scale_bits);
      key_scale = e8m0_to_float(static_cast<uint8_t>(key_scale_bits));
      value_scale = e8m0_to_float(static_cast<uint8_t>(value_scale_bits));
    }
    key_all_zero = __shfl(key_all_zero, 0U, kWaveSize);
    value_all_zero = __shfl(value_all_zero, 0U, kWaveSize);
    key_scale = __shfl(key_scale, 0U, kWaveSize);
    value_scale = __shfl(value_scale, 0U, kWaveSize);
    if (dimension < padded_head_dim) {
      key_output[output_row * padded_head_dim + dimension] =
          !active || key_all_zero != 0U
              ? 0U
              : (E5 ? float_to_e5m2(key_value / key_scale)
                    : float_to_e4m3fn_fp8_append(key_value / key_scale));
      value_output[output_row * padded_head_dim + dimension] =
          !active || value_all_zero != 0U
              ? 0U
              : (E5 ? float_to_e5m2(value_value / value_scale)
                    : float_to_e4m3fn_fp8_append(value_value / value_scale));
    }
  }
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_KV_WORKGROUP_SIZE,
    1) void sllm_kv_state_bf16_to_mxfp8_e4_token_major_v1(Bf16Input key_input,
                                                          Bf16Input value_input,
                                                          uint8_t
                                                              *const key_output,
                                                          uint8_t *const
                                                              value_output,
                                                          uint8_t
                                                              *const key_scales,
                                                          uint8_t *const
                                                              value_scales,
                                                          const uint32_t
                                                              token_count,
                                                          const uint64_t
                                                              start_position,
                                                          const uint32_t
                                                              head_count,
                                                          const uint32_t
                                                              head_dim,
                                                          const uint32_t /*encoding*/) {
  const uint64_t row = blockIdx.x;
  if (row >= static_cast<uint64_t>(token_count) * head_count) {
    return;
  }
  const uint64_t token = row / head_count;
  const uint64_t head = row % head_count;
  const uint64_t output_row = (start_position + token) * head_count + head;
  const uint64_t input_base = row * head_dim;
  quantize_mxfp8_pair<false>(key_input, value_input, key_output, value_output,
                             key_scales, value_scales, input_base, output_row,
                             head_dim);
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_KV_WORKGROUP_SIZE,
    1) void sllm_kv_state_bf16_to_mxfp8_e5_token_major_v1(Bf16Input key_input,
                                                          Bf16Input value_input,
                                                          uint8_t
                                                              *const key_output,
                                                          uint8_t *const
                                                              value_output,
                                                          uint8_t
                                                              *const key_scales,
                                                          uint8_t *const
                                                              value_scales,
                                                          const uint32_t
                                                              token_count,
                                                          const uint64_t
                                                              start_position,
                                                          const uint32_t
                                                              head_count,
                                                          const uint32_t
                                                              head_dim,
                                                          const uint32_t /*encoding*/) {
  const uint64_t row = blockIdx.x;
  if (row >= static_cast<uint64_t>(token_count) * head_count) {
    return;
  }
  const uint64_t token = row / head_count;
  const uint64_t head = row % head_count;
  const uint64_t output_row = (start_position + token) * head_count + head;
  const uint64_t input_base = row * head_dim;
  quantize_mxfp8_pair<true>(key_input, value_input, key_output, value_output,
                            key_scales, value_scales, input_base, output_row,
                            head_dim);
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_KV_WORKGROUP_SIZE,
    1) void sllm_kv_state_bf16_to_fp8_block16_token_major_v1(Bf16Input
                                                                 key_input,
                                                             Bf16Input
                                                                 value_input,
                                                             uint8_t *const
                                                                 key_output,
                                                             uint8_t *const
                                                                 value_output,
                                                             uint8_t *const
                                                                 key_scales,
                                                             uint8_t *const
                                                                 value_scales,
                                                             const uint32_t
                                                                 token_count,
                                                             const uint64_t
                                                                 start_position,
                                                             const uint32_t
                                                                 head_count,
                                                             const uint32_t
                                                                 head_dim,
                                                             const uint32_t
                                                                 encoding) {
  const uint64_t row = blockIdx.x;
  if (row >= static_cast<uint64_t>(token_count) * head_count) {
    return;
  }
  const uint64_t token = row / head_count;
  const uint64_t head = row % head_count;
  const uint64_t output_row = (start_position + token) * head_count + head;
  const uint64_t input_base = row * head_dim;
  quantize_fp8_block16_row<true>(key_input, key_output, key_scales, input_base,
                                 output_row, head_dim, encoding);
  quantize_fp8_block16_row<false>(value_input, value_output, value_scales,
                                  input_base, output_row, head_dim, encoding);
}

template <bool E5>
__device__ void quantize_fp8_block16_v2_pair(
    Bf16Input key_input, Bf16Input value_input, uint8_t *const key_output,
    uint8_t *const value_output, uint8_t *const key_scales,
    uint8_t *const value_scales, const uint64_t row, const uint32_t token_count,
    const uint64_t start_position, const uint32_t head_count,
    const uint32_t head_dim) {
  if (row >= static_cast<uint64_t>(token_count) * head_count) {
    return;
  }
  const uint64_t token = row / head_count;
  const uint64_t head = row % head_count;
  const uint64_t output_row = (start_position + token) * head_count + head;
  const uint64_t input_base = row * head_dim;
  constexpr uint32_t kEncoding = E5 ? SLLM_HIP_KV_ENCODING_FP8_E5_BLOCK16_V2
                                    : SLLM_HIP_KV_ENCODING_FP8_E4_BLOCK16_V2;
  quantize_fp8_block16_row<true>(key_input, key_output, key_scales, input_base,
                                 output_row, head_dim, kEncoding);
  quantize_fp8_block16_row<false>(value_input, value_output, value_scales,
                                  input_base, output_row, head_dim, kEncoding);
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_KV_WORKGROUP_SIZE,
    1) void sllm_kv_state_bf16_to_fp8_e4_block16_token_major_v2(Bf16Input
                                                                    key_input,
                                                                Bf16Input
                                                                    value_input,
                                                                uint8_t *const
                                                                    key_output,
                                                                uint8_t *const
                                                                    value_output,
                                                                uint8_t *const
                                                                    key_scales,
                                                                uint8_t *const
                                                                    value_scales,
                                                                const uint32_t
                                                                    token_count,
                                                                const uint64_t
                                                                    start_position,
                                                                const uint32_t
                                                                    head_count,
                                                                const uint32_t
                                                                    head_dim,
                                                                const uint32_t /*encoding*/) {
  quantize_fp8_block16_v2_pair<false>(key_input, value_input, key_output,
                                      value_output, key_scales, value_scales,
                                      blockIdx.x, token_count, start_position,
                                      head_count, head_dim);
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_KV_WORKGROUP_SIZE,
    1) void sllm_kv_state_bf16_to_fp8_e5_block16_token_major_v2(Bf16Input
                                                                    key_input,
                                                                Bf16Input
                                                                    value_input,
                                                                uint8_t *const
                                                                    key_output,
                                                                uint8_t *const
                                                                    value_output,
                                                                uint8_t *const
                                                                    key_scales,
                                                                uint8_t *const
                                                                    value_scales,
                                                                const uint32_t
                                                                    token_count,
                                                                const uint64_t
                                                                    start_position,
                                                                const uint32_t
                                                                    head_count,
                                                                const uint32_t
                                                                    head_dim,
                                                                const uint32_t /*encoding*/) {
  quantize_fp8_block16_v2_pair<true>(key_input, value_input, key_output,
                                     value_output, key_scales, value_scales,
                                     blockIdx.x, token_count, start_position,
                                     head_count, head_dim);
}

#if SLLM_ENABLE_PHASE54_KV_RESEARCH
template <bool E5>
__device__ void quantize_fp8_block16_phase54_research_row(
    const uint16_t *const input, uint8_t *const output, uint8_t *const scales,
    const uint64_t input_base, const uint64_t output_row,
    const uint32_t head_dim, const uint32_t recipe) {
  if (threadIdx.x != 0U) {
    return;
  }
  const uint64_t blocks_per_row = (static_cast<uint64_t>(head_dim) + 15U) / 16U;
  const uint64_t padded_head_dim = blocks_per_row * 16U;
#if defined(__gfx942__)
  constexpr bool e4_fnuz = true;
#else
  constexpr bool e4_fnuz = false;
#endif
  for (uint64_t block = 0U; block != blocks_per_row; ++block) {
    const uint32_t begin = static_cast<uint32_t>(block * 16U);
    const uint32_t end = min(begin + 16U, head_dim);
    const bool duplicate_parent32 =
        recipe == sllm_kv_state_kernel::kPhase54KvRecipeParent32Duplicate;
    const uint32_t scale_begin =
        duplicate_parent32 ? static_cast<uint32_t>((block / 2U) * 32U) : begin;
    const uint32_t scale_end =
        duplicate_parent32 ? min(scale_begin + 32U, head_dim) : end;
    float maximum = 0.0F;
    bool all_zero = true;
    for (uint32_t dimension = scale_begin; dimension != scale_end;
         ++dimension) {
      const float value = bf16_to_float(input[input_base + dimension]);
      all_zero = all_zero && value == 0.0F;
      if (isfinite(value)) {
        maximum = fmaxf(maximum, fabsf(value));
      }
    }
    const uint8_t scale_bits = phase54_block16_e8m0_scale(maximum, E5, recipe);
    const float scale = e8m0_to_float(scale_bits);
    scales[output_row * blocks_per_row + block] = scale_bits;
    for (uint32_t dimension = begin; dimension != end; ++dimension) {
      if (all_zero) {
        output[output_row * padded_head_dim + dimension] = 0U;
        continue;
      }
      const float normalized =
          bf16_to_float(input[input_base + dimension]) / scale;
      output[output_row * padded_head_dim + dimension] =
          E5 ? float_to_e5m2(normalized)
             : (e4_fnuz ? float_to_e4m3fnuz(normalized)
                        : float_to_e4m3fn_fp8_append(normalized));
    }
    for (uint32_t dimension = end;
         dimension != static_cast<uint32_t>((block + 1U) * 16U); ++dimension) {
      output[output_row * padded_head_dim + dimension] = 0U;
    }
  }
}

template <bool E5>
__device__ void quantize_fp8_block16_phase54_research_pair(
    Bf16Input key_input, Bf16Input value_input, uint8_t *const key_output,
    uint8_t *const value_output, uint8_t *const key_scales,
    uint8_t *const value_scales, const uint64_t row, const uint32_t token_count,
    const uint64_t start_position, const uint32_t head_count,
    const uint32_t head_dim, const uint32_t key_recipe,
    const uint32_t value_recipe) {
  if (row >= static_cast<uint64_t>(token_count) * head_count) {
    return;
  }
  const uint64_t token = row / head_count;
  const uint64_t head = row % head_count;
  const uint64_t output_row = (start_position + token) * head_count + head;
  const uint64_t input_base = row * head_dim;
  quantize_fp8_block16_phase54_research_row<E5>(
      key_input, key_output, key_scales, input_base, output_row, head_dim,
      key_recipe);
  quantize_fp8_block16_phase54_research_row<E5>(
      value_input, value_output, value_scales, input_base, output_row, head_dim,
      value_recipe);
}

extern "C" __global__
__launch_bounds__(SLLM_HIP_KV_WORKGROUP_SIZE, 1) void sllm_kv_state_bf16_to_fp8_e4_block16_phase54_research_v1(
    Bf16Input key_input, Bf16Input value_input, uint8_t *const key_output,
    uint8_t *const value_output, uint8_t *const key_scales,
    uint8_t *const value_scales, const uint32_t token_count,
    const uint64_t start_position, const uint32_t head_count,
    const uint32_t head_dim, const uint32_t key_recipe,
    const uint32_t value_recipe) {
  quantize_fp8_block16_phase54_research_pair<false>(
      key_input, value_input, key_output, value_output, key_scales,
      value_scales, blockIdx.x, token_count, start_position, head_count,
      head_dim, key_recipe, value_recipe);
}

extern "C" __global__
__launch_bounds__(SLLM_HIP_KV_WORKGROUP_SIZE, 1) void sllm_kv_state_bf16_to_fp8_e5_block16_phase54_research_v1(
    Bf16Input key_input, Bf16Input value_input, uint8_t *const key_output,
    uint8_t *const value_output, uint8_t *const key_scales,
    uint8_t *const value_scales, const uint32_t token_count,
    const uint64_t start_position, const uint32_t head_count,
    const uint32_t head_dim, const uint32_t key_recipe,
    const uint32_t value_recipe) {
  quantize_fp8_block16_phase54_research_pair<true>(
      key_input, value_input, key_output, value_output, key_scales,
      value_scales, blockIdx.x, token_count, start_position, head_count,
      head_dim, key_recipe, value_recipe);
}
#endif

template <bool Key>
__device__ void
quantize_nvfp4_row(const uint16_t *const input, uint8_t *const packed,
                   uint8_t *const block_scales, float *const outer_scales,
                   const uint64_t input_base, const uint64_t output_row,
                   const uint32_t head_dim) {
  if (threadIdx.x != 0U) {
    return;
  }
  float maximum = 0.0F;
  for (uint32_t dimension = 0U; dimension != head_dim; ++dimension) {
    const float value = bf16_to_float(input[input_base + dimension]);
    maximum = fmaxf(maximum, isfinite(value) ? fabsf(value) : 0.0F);
  }
  const float outer = maximum == 0.0F ? 1.0F : maximum / (448.0F * 6.0F);
  outer_scales[output_row] = outer;
  const uint64_t packed_per_row = (static_cast<uint64_t>(head_dim) + 1U) / 2U;
  const uint64_t blocks_per_row = (static_cast<uint64_t>(head_dim) + 15U) / 16U;
  for (uint64_t block = 0U; block != blocks_per_row; ++block) {
    const uint32_t begin = static_cast<uint32_t>(block * 16U);
    const uint32_t end = min(begin + 16U, head_dim);
    float block_maximum = 0.0F;
    bool block_has_infinity = false;
    for (uint32_t dimension = begin; dimension != end; ++dimension) {
      const float value = bf16_to_float(input[input_base + dimension]);
      block_maximum =
          fmaxf(block_maximum, isfinite(value) ? fabsf(value) : 0.0F);
      block_has_infinity = block_has_infinity || isinf(value);
    }
    // E2M1 has no non-finite encodings. Preserve NaN as canonical zero, and
    // saturate either infinity to the largest value representable by the
    // row's finite-derived outer scale. For an all-infinite row outer is one.
    if (block_has_infinity) {
      block_maximum = 448.0F * 6.0F * outer;
    }
    const uint8_t scale_bits = float_to_e4m3fn((block_maximum / 6.0F) / outer);
    const float decoded_scale = e4m3fn_to_float(scale_bits);
    block_scales[output_row * blocks_per_row + block] = scale_bits;
    for (uint32_t dimension = begin; dimension < end; dimension += 2U) {
      const float first = bf16_to_float(input[input_base + dimension]);
      const uint8_t low = decoded_scale == 0.0F
                              ? 0U
                              : float_to_e2m1(first / (decoded_scale * outer));
      uint8_t high = 0U;
      if (dimension + 1U < end) {
        const float second = bf16_to_float(input[input_base + dimension + 1U]);
        high = decoded_scale == 0.0F
                   ? 0U
                   : float_to_e2m1(second / (decoded_scale * outer));
      }
      packed[output_row * packed_per_row + dimension / 2U] =
          static_cast<uint8_t>(low | (high << 4U));
    }
  }
  (void)Key;
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_KV_WORKGROUP_SIZE,
    1) void sllm_kv_state_bf16_to_nvfp4_token_major_v1(Bf16Input key_input,
                                                       Bf16Input value_input,
                                                       uint8_t
                                                           *const key_output,
                                                       uint8_t
                                                           *const value_output,
                                                       uint8_t *const
                                                           key_block_scales,
                                                       uint8_t *const
                                                           value_block_scales,
                                                       float *const
                                                           key_outer_scales,
                                                       float *const
                                                           value_outer_scales,
                                                       const uint32_t
                                                           token_count,
                                                       const uint64_t
                                                           start_position,
                                                       const uint32_t
                                                           head_count,
                                                       const uint32_t
                                                           head_dim) {
  const uint64_t row = blockIdx.x;
  if (row >= static_cast<uint64_t>(token_count) * head_count) {
    return;
  }
  const uint64_t token = row / head_count;
  const uint64_t head = row % head_count;
  const uint64_t output_row = (start_position + token) * head_count + head;
  const uint64_t input_base = row * head_dim;
  quantize_nvfp4_row<true>(key_input, key_output, key_block_scales,
                           key_outer_scales, input_base, output_row, head_dim);
  quantize_nvfp4_row<false>(value_input, value_output, value_block_scales,
                            value_outer_scales, input_base, output_row,
                            head_dim);
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_KV_WORKGROUP_SIZE,
    1) void sllm_kv_state_bf16_to_f16_token_major_v2(Bf16Input key_input,
                                                     Bf16Input value_input,
                                                     uint16_t *const key_output,
                                                     uint16_t
                                                         *const value_output,
                                                     const uint32_t token_count,
                                                     const uint64_t
                                                     /*capacity_tokens*/,
                                                     const uint64_t
                                                         start_position,
                                                     const uint32_t head_count,
                                                     const uint32_t head_dim) {
  const uint64_t element = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                           static_cast<uint64_t>(threadIdx.x);
  const uint64_t row_width = static_cast<uint64_t>(head_count) * head_dim;
  const uint64_t total = static_cast<uint64_t>(token_count) * row_width;
  if (element >= total) {
    return;
  }
  const uint64_t row = element / row_width;
  const uint64_t within_row = element % row_width;
  const uint64_t output_offset =
      (start_position + row) * row_width + within_row;
  key_output[output_offset] = bf16_to_f16(key_input[element]);
  value_output[output_offset] = bf16_to_f16(value_input[element]);
}

namespace sllm_kv_state_kernel {

hipError_t launch(const uint16_t *const key_input,
                  const uint16_t *const value_input, void *const key_output,
                  void *const value_output, void *const key_scales,
                  void *const value_scales, float *const key_outer_scales,
                  float *const value_outer_scales, const uint32_t token_count,
                  const uint64_t capacity_tokens, const uint64_t start_position,
                  const uint32_t head_count, const uint32_t head_dim,
                  const uint32_t encoding, const float static_key_scale,
                  const float static_value_scale,
                  const hipStream_t stream) noexcept {
  const uint64_t total = static_cast<uint64_t>(token_count) * head_count;
  const uint32_t grid_count =
      encoding == SLLM_HIP_KV_ENCODING_FP16_V1
          ? static_cast<uint32_t>(
                (total * head_dim + SLLM_HIP_KV_WORKGROUP_SIZE - 1U) /
                SLLM_HIP_KV_WORKGROUP_SIZE)
          : static_cast<uint32_t>(total);
  const dim3 grid(grid_count, 1U, 1U);
  const dim3 block(SLLM_HIP_KV_WORKGROUP_SIZE, 1U, 1U);
  if (encoding == SLLM_HIP_KV_ENCODING_FP16_V1) {
    hipLaunchKernelGGL(sllm_kv_state_bf16_to_f16_token_major_v2, grid, block,
                       0U, stream, key_input, value_input,
                       static_cast<uint16_t *>(key_output),
                       static_cast<uint16_t *>(value_output), token_count,
                       capacity_tokens, start_position, head_count, head_dim);
  } else if (encoding == SLLM_HIP_KV_ENCODING_FP8_V1 ||
             encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1) {
    hipLaunchKernelGGL(
        sllm_kv_state_bf16_to_fp8_token_major_v1, grid, block, 0U, stream,
        key_input, value_input, static_cast<uint8_t *>(key_output),
        static_cast<uint8_t *>(value_output), static_cast<float *>(key_scales),
        static_cast<float *>(value_scales), token_count, start_position,
        head_count, head_dim, static_key_scale, static_value_scale,
        encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1 ? UINT32_C(1)
                                                       : UINT32_C(0));
  } else if (encoding == SLLM_HIP_KV_ENCODING_NVFP4_V1) {
    hipLaunchKernelGGL(
        sllm_kv_state_bf16_to_nvfp4_token_major_v1, grid, block, 0U, stream,
        key_input, value_input, static_cast<uint8_t *>(key_output),
        static_cast<uint8_t *>(value_output),
        static_cast<uint8_t *>(key_scales),
        static_cast<uint8_t *>(value_scales), key_outer_scales,
        value_outer_scales, token_count, start_position, head_count, head_dim);
  } else if (encoding == SLLM_HIP_KV_ENCODING_FP8_E4_BLOCK16_V2) {
#if SLLM_ENABLE_PHASE54_KV_RESEARCH
    uint32_t key_recipe = sllm_kv_state_kernel::kPhase54KvRecipeFloor;
    uint32_t value_recipe = sllm_kv_state_kernel::kPhase54KvRecipeFloor;
    (void)sllm_phase54_kv_research_get_recipe_pair_v1(&key_recipe,
                                                      &value_recipe);
    if (key_recipe != sllm_kv_state_kernel::kPhase54KvRecipeFloor ||
        value_recipe != sllm_kv_state_kernel::kPhase54KvRecipeFloor) {
      hipLaunchKernelGGL(
          sllm_kv_state_bf16_to_fp8_e4_block16_phase54_research_v1, grid, block,
          0U, stream, key_input, value_input,
          static_cast<uint8_t *>(key_output),
          static_cast<uint8_t *>(value_output),
          static_cast<uint8_t *>(key_scales),
          static_cast<uint8_t *>(value_scales), token_count, start_position,
          head_count, head_dim, key_recipe, value_recipe);
    } else {
#endif
      hipLaunchKernelGGL(sllm_kv_state_bf16_to_fp8_e4_block16_token_major_v2,
                         grid, block, 0U, stream, key_input, value_input,
                         static_cast<uint8_t *>(key_output),
                         static_cast<uint8_t *>(value_output),
                         static_cast<uint8_t *>(key_scales),
                         static_cast<uint8_t *>(value_scales), token_count,
                         start_position, head_count, head_dim, encoding);
#if SLLM_ENABLE_PHASE54_KV_RESEARCH
    }
#endif
  } else if (encoding == SLLM_HIP_KV_ENCODING_FP8_E5_BLOCK16_V2) {
#if SLLM_ENABLE_PHASE54_KV_RESEARCH
    uint32_t key_recipe = sllm_kv_state_kernel::kPhase54KvRecipeFloor;
    uint32_t value_recipe = sllm_kv_state_kernel::kPhase54KvRecipeFloor;
    (void)sllm_phase54_kv_research_get_recipe_pair_v1(&key_recipe,
                                                      &value_recipe);
    if (key_recipe != sllm_kv_state_kernel::kPhase54KvRecipeFloor ||
        value_recipe != sllm_kv_state_kernel::kPhase54KvRecipeFloor) {
      hipLaunchKernelGGL(
          sllm_kv_state_bf16_to_fp8_e5_block16_phase54_research_v1, grid, block,
          0U, stream, key_input, value_input,
          static_cast<uint8_t *>(key_output),
          static_cast<uint8_t *>(value_output),
          static_cast<uint8_t *>(key_scales),
          static_cast<uint8_t *>(value_scales), token_count, start_position,
          head_count, head_dim, key_recipe, value_recipe);
    } else {
#endif
      hipLaunchKernelGGL(sllm_kv_state_bf16_to_fp8_e5_block16_token_major_v2,
                         grid, block, 0U, stream, key_input, value_input,
                         static_cast<uint8_t *>(key_output),
                         static_cast<uint8_t *>(value_output),
                         static_cast<uint8_t *>(key_scales),
                         static_cast<uint8_t *>(value_scales), token_count,
                         start_position, head_count, head_dim, encoding);
#if SLLM_ENABLE_PHASE54_KV_RESEARCH
    }
#endif
  } else if (encoding == SLLM_HIP_KV_ENCODING_MXFP8_E4_V1) {
    hipLaunchKernelGGL(sllm_kv_state_bf16_to_mxfp8_e4_token_major_v1, grid,
                       block, 0U, stream, key_input, value_input,
                       static_cast<uint8_t *>(key_output),
                       static_cast<uint8_t *>(value_output),
                       static_cast<uint8_t *>(key_scales),
                       static_cast<uint8_t *>(value_scales), token_count,
                       start_position, head_count, head_dim, encoding);
  } else if (encoding == SLLM_HIP_KV_ENCODING_MXFP8_E5_V1) {
    hipLaunchKernelGGL(sllm_kv_state_bf16_to_mxfp8_e5_token_major_v1, grid,
                       block, 0U, stream, key_input, value_input,
                       static_cast<uint8_t *>(key_output),
                       static_cast<uint8_t *>(value_output),
                       static_cast<uint8_t *>(key_scales),
                       static_cast<uint8_t *>(value_scales), token_count,
                       start_position, head_count, head_dim, encoding);
  } else {
    return hipErrorInvalidValue;
  }
  return hipGetLastError();
}

} // namespace sllm_kv_state_kernel
