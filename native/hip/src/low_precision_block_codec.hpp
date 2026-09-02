#ifndef SLLM_LOW_PRECISION_BLOCK_CODEC_HPP
#define SLLM_LOW_PRECISION_BLOCK_CODEC_HPP

#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace sllm_lowp {

struct E4M3Fn {};
struct E4M3FnuZ {};
struct E5M2 {};
struct E3M2 {};
struct E2M1 {};
struct E8M0 {};

// Every finite OCP E3M2 value has an exact OCP E4M3FN representation.  Keep
// this as a code-to-code transform so packed MXFP6 tensors remain resident and
// callers can expand only the tile currently consumed by an MXFP8 arithmetic
// path.  The two formats use sign bits 5/7 and exponent biases 3/7;
// E3M2's three non-zero subnormals need the explicit normal E4M3 encodings
// below, while all normal values are a bias adjustment plus one mantissa bit.
__host__ __device__ constexpr uint8_t
e3m2_to_e4m3fn_exact_bits(const uint8_t raw) noexcept {
  const uint8_t bits = raw & UINT8_C(0x3f);
  const uint8_t sign = static_cast<uint8_t>((bits & UINT8_C(0x20)) << 2U);
  const uint8_t magnitude = bits & UINT8_C(0x1f);
  const uint8_t exponent = magnitude >> 2U;
  const uint8_t mantissa = magnitude & UINT8_C(0x03);
  if (exponent == 0U) {
    constexpr uint8_t subnormal_map[4] = {UINT8_C(0x00), UINT8_C(0x18),
                                          UINT8_C(0x20), UINT8_C(0x24)};
    return static_cast<uint8_t>(sign | subnormal_map[mantissa]);
  }
  return static_cast<uint8_t>(
      sign | static_cast<uint8_t>((exponent + UINT8_C(4)) << 3U) |
      static_cast<uint8_t>(mantissa << 1U));
}

// Convert one native MXFP6 packing group (four E3M2 values in the low 24
// bits) into four byte-addressable E4M3FN values.  Consumers that materialize
// a tile for FP8 arithmetic can load the three source bytes once instead of
// repeating the same 24-bit load for every scalar lane.
__host__ __device__ constexpr uint32_t
e3m2x4_to_e4m3fn_exact_bits(const uint32_t packed) noexcept {
  uint32_t converted = 0U;
  for (uint32_t lane = 0U; lane < 4U; ++lane) {
    const uint8_t code =
        static_cast<uint8_t>((packed >> (lane * 6U)) & UINT32_C(0x3f));
    converted |= static_cast<uint32_t>(e3m2_to_e4m3fn_exact_bits(code))
                 << (lane * 8U);
  }
  return converted;
}

// Expand the four six-bit E3M2 lanes into byte lanes, then apply the exact
// E4M3 mapping with lane-wise integer operations. The exponent-zero path is
// selected with a full-byte mask so signed zero, subnormal, and normal lanes
// may be mixed in one packed group without scalar branches or a lookup table.
__host__ __device__ constexpr uint32_t
e3m2x4_to_e4m3fn_exact_bits_swar(const uint32_t packed) noexcept {
  const uint32_t lanes = (packed & UINT32_C(0x0000003f)) |
                         ((packed & UINT32_C(0x00000fc0)) << 2U) |
                         ((packed & UINT32_C(0x0003f000)) << 4U) |
                         ((packed & UINT32_C(0x00fc0000)) << 6U);
  const uint32_t sign = (lanes & UINT32_C(0x20202020)) << 2U;
  const uint32_t exponent = (lanes >> 2U) & UINT32_C(0x07070707);
  const uint32_t mantissa = lanes & UINT32_C(0x03030303);
  const uint32_t normal =
      sign | ((exponent + UINT32_C(0x04040404)) << 3U) | (mantissa << 1U);

  const uint32_t mantissa_low = mantissa & UINT32_C(0x01010101);
  const uint32_t mantissa_high = (mantissa & UINT32_C(0x02020202)) >> 1U;
  const uint32_t both_mantissa_bits = mantissa_low & mantissa_high;
  const uint32_t high_only = mantissa_high & ~mantissa_low;
  const uint32_t subnormal = sign + mantissa_low * UINT32_C(0x18) +
                             high_only * UINT32_C(0x20) +
                             both_mantissa_bits * UINT32_C(0x0c);

  // exponent is restricted to three bits per byte, so collapsing those bits
  // into each lane's low bit does not allow neighboring lanes to interact.
  const uint32_t exponent_any = (exponent & UINT32_C(0x01010101)) |
                                ((exponent >> 1U) & UINT32_C(0x01010101)) |
                                ((exponent >> 2U) & UINT32_C(0x01010101));
  const uint32_t exponent_zero_mask =
      ((~exponent_any) & UINT32_C(0x01010101)) * UINT32_C(0xff);
  return (subnormal & exponent_zero_mask) | (normal & ~exponent_zero_mask);
}

template <typename Format> struct ScalarCodec;

template <> struct ScalarCodec<E4M3Fn> {
  // Decode an E4M3 value plane produced by the internal MX quantizer.  The
  // caller must retain the paired E8M0 scale: NaN-containing blocks use scale
  // 255 and a zero value plane, while all emitted value magnitudes are <=126.
  // Signed zero and every E4M3 subnormal remain exact.
  __device__ __forceinline__ static float
  decode_mx_value_plane(const uint8_t bits) noexcept {
    const uint32_t sign = static_cast<uint32_t>(bits & UINT8_C(0x80)) << 24U;
    const uint32_t magnitude = static_cast<uint32_t>(bits) & UINT32_C(0x7f);
    const uint32_t exponent = magnitude >> 3U;
    const uint32_t mantissa = magnitude & UINT32_C(0x07);
    if (__builtin_expect(exponent == 0U, 0)) {
      if (mantissa == 0U) {
        return __uint_as_float(sign);
      }
      const float value = static_cast<float>(mantissa) * 0x1p-9F;
      return __uint_as_float(__float_as_uint(value) | sign);
    }
    return __uint_as_float(sign | ((exponent + 120U) << 23U) |
                           (mantissa << 20U));
  }

  __device__ __forceinline__ static float decode(const uint8_t bits) noexcept {
#if defined(__gfx1201__)
    return __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(bits), 0);
#else
    const uint32_t sign = static_cast<uint32_t>(bits & UINT8_C(0x80)) << 24U;
    const uint32_t magnitude = static_cast<uint32_t>(bits) & UINT32_C(0x7f);
    const uint32_t exponent = magnitude >> 3U;
    const uint32_t mantissa = magnitude & UINT32_C(0x07);
    if (exponent == 0U) {
      if (mantissa == 0U) {
        return __uint_as_float(sign);
      }
      const float value = static_cast<float>(mantissa) * 0x1p-9F;
      return __uint_as_float(__float_as_uint(value) | sign);
    }
    if (magnitude == UINT32_C(0x7f)) {
      return __uint_as_float(sign | UINT32_C(0x7fc00000));
    }
    return __uint_as_float(sign | ((exponent + 120U) << 23U) |
                           (mantissa << 20U));
#endif
  }

  __device__ __forceinline__ static uint8_t encode(float value) noexcept {
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
#if defined(__gfx1201__)
    const uint32_t packed = static_cast<uint32_t>(
        __builtin_amdgcn_cvt_pk_fp8_f32(value, value, 0, false));
    return static_cast<uint8_t>(packed & UINT32_C(0xff));
#else
    if (magnitude < 0.015625F) {
      const float scaled = magnitude * 512.0F;
      const uint32_t floor = static_cast<uint32_t>(scaled);
      const float fraction = scaled - static_cast<float>(floor);
      const uint32_t rounded =
          floor +
          static_cast<uint32_t>(fraction > 0.5F ||
                                (fraction == 0.5F && (floor & 1U) != 0U));
      return static_cast<uint8_t>(sign | static_cast<uint8_t>(rounded));
    }
    const uint32_t bits = __float_as_uint(magnitude);
    const uint32_t rounded =
        bits + UINT32_C(0x0007ffff) + ((bits >> 20U) & UINT32_C(1));
    const uint32_t exponent = ((rounded >> 23U) & UINT32_C(0xff)) - 120U;
    const uint32_t code =
        (exponent << 3U) | ((rounded >> 20U) & UINT32_C(0x07));
    return static_cast<uint8_t>(
        sign | static_cast<uint8_t>(min(code, UINT32_C(0x7e))));
#endif
  }

  // Decode the internal block-scaled representation without materializing an
  // E8M0 float and multiplying it afterwards.  Normal E4M3 values and scales
  // whose combined exponent remains a normal FP32 value can be represented by
  // adjusting the FP32 exponent directly.  Zero, subnormal, NaN, underflow,
  // and overflow cases retain the scalar reference behavior below.
  __device__ __forceinline__ static float
  decode_scaled(const uint8_t bits, const uint8_t scale_bits) noexcept {
    if (scale_bits == UINT8_C(0xff)) {
      return NAN;
    }
    const uint32_t sign = static_cast<uint32_t>(bits & UINT8_C(0x80)) << 24U;
    const uint32_t magnitude = static_cast<uint32_t>(bits) & UINT32_C(0x7f);
    const uint32_t exponent = magnitude >> 3U;
    const uint32_t mantissa = magnitude & UINT32_C(0x07);
    const int32_t scaled_exponent =
        static_cast<int32_t>(exponent) + static_cast<int32_t>(scale_bits) - 7;
    if (exponent != 0U && magnitude != UINT32_C(0x7f) && scaled_exponent > 0 &&
        scaled_exponent < 255) {
      return __uint_as_float(sign |
                             (static_cast<uint32_t>(scaled_exponent) << 23U) |
                             (mantissa << 20U));
    }
    const float scale = __uint_as_float(
        scale_bits == 0U ? UINT32_C(0x00400000)
                         : static_cast<uint32_t>(scale_bits) << 23U);
    return decode(bits) * scale;
  }
};

template <> struct ScalarCodec<E4M3FnuZ> {
  __device__ __forceinline__ static float decode(const uint8_t bits) noexcept {
    if (bits == UINT8_C(0x80)) {
      return NAN;
    }
    const uint32_t sign = static_cast<uint32_t>(bits & UINT8_C(0x80)) << 24U;
    const uint32_t magnitude = static_cast<uint32_t>(bits) & UINT32_C(0x7f);
    const uint32_t exponent = magnitude >> 3U;
    const uint32_t mantissa = magnitude & UINT32_C(0x07);
    if (exponent == 0U) {
      if (mantissa == 0U) {
        return 0.0F;
      }
      const float value = static_cast<float>(mantissa) * 0x1p-10F;
      return __uint_as_float(__float_as_uint(value) | sign);
    }
    return __uint_as_float(sign | ((exponent + 119U) << 23U) |
                           (mantissa << 20U));
  }

  __device__ __forceinline__ static uint8_t encode(float value) noexcept {
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
      if (decode(middle) < value) {
        low = static_cast<uint8_t>(middle + 1U);
      } else {
        high = middle;
      }
    }
    const uint8_t upper = low;
    const uint8_t lower = upper == 0U ? 0U : static_cast<uint8_t>(upper - 1U);
    const float lower_error = value - decode(lower);
    const float upper_error = decode(upper) - value;
    const bool select_upper =
        upper_error < lower_error ||
        (upper_error == lower_error && (upper & UINT8_C(1)) == 0U &&
         (lower & UINT8_C(1)) != 0U);
    const uint8_t selected = select_upper ? upper : lower;
    return negative && selected != 0U
               ? static_cast<uint8_t>(selected | UINT8_C(0x80))
               : selected;
  }
};

template <> struct ScalarCodec<E5M2> {
  __device__ __forceinline__ static float decode(const uint8_t bits) noexcept {
#if defined(__gfx1030__)
    __half_raw half_bits{};
    half_bits.x = static_cast<uint16_t>(static_cast<uint16_t>(bits) << 8U);
    return __half2float(__half{half_bits});
#else
    const uint32_t sign = static_cast<uint32_t>(bits & UINT8_C(0x80)) << 24U;
    const uint32_t exponent =
        (static_cast<uint32_t>(bits) >> 2U) & UINT32_C(0x1f);
    const uint32_t mantissa = static_cast<uint32_t>(bits) & UINT32_C(0x03);
    if (exponent == 0U) {
      if (mantissa == 0U) {
        return __uint_as_float(sign);
      }
      const float value = static_cast<float>(mantissa) * 0x1p-16F;
      return __uint_as_float(__float_as_uint(value) | sign);
    }
    if (exponent == UINT32_C(0x1f)) {
      return mantissa == 0U ? __uint_as_float(sign | UINT32_C(0x7f800000))
                            : __uint_as_float(sign | UINT32_C(0x7fc00000) |
                                              (mantissa << 21U));
    }
    return __uint_as_float(sign | ((exponent + 112U) << 23U) |
                           (mantissa << 21U));
#endif
  }

  __device__ __forceinline__ static uint8_t encode(float value) noexcept {
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
      if (decode(static_cast<uint8_t>(middle)) < value) {
        low = middle + 1U;
      } else {
        high = middle;
      }
    }
    const uint8_t upper = static_cast<uint8_t>(low);
    const uint8_t lower = upper == 0U ? 0U : static_cast<uint8_t>(upper - 1U);
    const float lower_error = value - decode(lower);
    const float upper_error = decode(upper) - value;
    const bool upper_selected =
        upper_error < lower_error || (upper_error == lower_error &&
                                      (upper & 1U) == 0U && (lower & 1U) != 0U);
    return static_cast<uint8_t>(sign | (upper_selected ? upper : lower));
  }
};

template <> struct ScalarCodec<E3M2> {
  __device__ __forceinline__ static float decode(const uint8_t raw) noexcept {
    const uint8_t bits = raw & UINT8_C(0x3f);
    const uint32_t sign = static_cast<uint32_t>(bits & UINT8_C(0x20)) << 26U;
    const uint32_t exponent =
        (static_cast<uint32_t>(bits) >> 2U) & UINT32_C(0x07);
    const uint32_t mantissa = static_cast<uint32_t>(bits) & UINT32_C(0x03);
    if (exponent == 0U) {
      if (mantissa == 0U) {
        return __uint_as_float(sign);
      }
      const float value = static_cast<float>(mantissa) * 0.0625F;
      return __uint_as_float(__float_as_uint(value) | sign);
    }
    return __uint_as_float(sign | ((exponent + 124U) << 23U) |
                           (mantissa << 21U));
  }

  __device__ __forceinline__ static uint8_t encode(float value) noexcept {
    const uint8_t sign = signbit(value) ? UINT8_C(0x20) : 0U;
    value = fabsf(value);
    if (value == 0.0F) {
      return sign;
    }
    if (!isfinite(value) || value >= 28.0F) {
      return static_cast<uint8_t>(sign | UINT8_C(0x1f));
    }
    if (value < 0.25F) {
      const float scaled = value * 16.0F;
      const uint32_t floor = static_cast<uint32_t>(scaled);
      const float fraction = scaled - static_cast<float>(floor);
      const uint32_t rounded =
          floor +
          static_cast<uint32_t>(fraction > 0.5F ||
                                (fraction == 0.5F && (floor & 1U) != 0U));
      return static_cast<uint8_t>(sign | static_cast<uint8_t>(rounded));
    }
    const uint32_t bits = __float_as_uint(value);
    const uint32_t rounded =
        bits + UINT32_C(0x000fffff) + ((bits >> 21U) & UINT32_C(1));
    const uint32_t exponent = ((rounded >> 23U) & UINT32_C(0xff)) - 124U;
    const uint32_t code =
        (exponent << 2U) | ((rounded >> 21U) & UINT32_C(0x03));
    return static_cast<uint8_t>(
        sign | static_cast<uint8_t>(min(code, UINT32_C(0x1f))));
  }
};

// All finite E3M2 values are exactly representable by IEEE FP16.  Returning
// the half encoding directly lets packed MXFP6 kernels form half2 operands
// without routing each value through an FP32 temporary.
__host__ __device__ constexpr uint16_t
e3m2_to_fp16_bits(const uint8_t raw) noexcept {
  const uint8_t bits = raw & UINT8_C(0x3f);
  const uint16_t sign =
      static_cast<uint16_t>(static_cast<uint16_t>(bits & UINT8_C(0x20)) << 10U);
  const uint8_t magnitude = bits & UINT8_C(0x1f);
  const uint8_t exponent = magnitude >> 2U;
  const uint8_t mantissa = magnitude & UINT8_C(0x03);
  if (exponent == 0U) {
    constexpr uint16_t subnormal_map[4] = {UINT16_C(0x0000), UINT16_C(0x2c00),
                                           UINT16_C(0x3000), UINT16_C(0x3200)};
    return static_cast<uint16_t>(sign | subnormal_map[mantissa]);
  }
  return static_cast<uint16_t>(
      sign | static_cast<uint16_t>(exponent + UINT8_C(12)) << 10U |
      static_cast<uint16_t>(mantissa) << 8U);
}

__device__ __forceinline__ __half e3m2_to_half(const uint8_t raw) noexcept {
  __half_raw bits{};
  bits.x = e3m2_to_fp16_bits(raw);
  return __half{bits};
}

__device__ __forceinline__ __half2
e3m2x2_to_half2(const uint8_t first, const uint8_t second) noexcept {
  return __halves2half2(e3m2_to_half(first), e3m2_to_half(second));
}

template <> struct ScalarCodec<E2M1> {
  __device__ __forceinline__ static float decode(const uint8_t bits) noexcept {
    constexpr float positive[8] = {0.0F, 0.5F, 1.0F, 1.5F,
                                   2.0F, 3.0F, 4.0F, 6.0F};
    const float value = positive[bits & UINT8_C(0x07)];
    return (bits & UINT8_C(0x08)) == 0U ? value : -value;
  }

  __device__ __forceinline__ static uint8_t encode(float value) noexcept {
    const uint8_t sign = signbit(value) ? UINT8_C(0x08) : 0U;
    if (isnan(value)) {
      return sign;
    }
    value = fminf(fabsf(value), 6.0F);
    uint8_t selected = 0U;
    float selected_error = value;
    for (uint8_t code = 1U; code != 8U; ++code) {
      const float error = fabsf(value - decode(code));
      if (error < selected_error ||
          (error == selected_error && (code & UINT8_C(1)) == 0U &&
           (selected & UINT8_C(1)) != 0U)) {
        selected = code;
        selected_error = error;
      }
    }
    return static_cast<uint8_t>(sign | selected);
  }
};

template <> struct ScalarCodec<E8M0> {
  __device__ __forceinline__ static float decode(const uint8_t bits) noexcept {
    if (bits == UINT8_C(0xff)) {
      return NAN;
    }
    return __uint_as_float(bits == 0U ? UINT32_C(0x00400000)
                                      : static_cast<uint32_t>(bits) << 23U);
  }
};

__device__ __forceinline__ float e4m3fn_to_float(const uint8_t bits) noexcept {
  return ScalarCodec<E4M3Fn>::decode(bits);
}
__device__ __forceinline__ uint8_t float_to_e4m3fn(float value) noexcept {
  return ScalarCodec<E4M3Fn>::encode(value);
}
__device__ __forceinline__ float
e4m3fnuz_to_float(const uint8_t bits) noexcept {
  return ScalarCodec<E4M3FnuZ>::decode(bits);
}
__device__ __forceinline__ uint8_t float_to_e4m3fnuz(float value) noexcept {
  return ScalarCodec<E4M3FnuZ>::encode(value);
}
__device__ __forceinline__ float e5m2_to_float(const uint8_t bits) noexcept {
  return ScalarCodec<E5M2>::decode(bits);
}
__device__ __forceinline__ uint8_t float_to_e5m2(float value) noexcept {
  return ScalarCodec<E5M2>::encode(value);
}
__device__ __forceinline__ float e3m2_to_float(const uint8_t bits) noexcept {
  return ScalarCodec<E3M2>::decode(bits);
}
__device__ __forceinline__ uint8_t float_to_e3m2(float value) noexcept {
  return ScalarCodec<E3M2>::encode(value);
}
__device__ __forceinline__ float e2m1_to_float(const uint8_t bits) noexcept {
  return ScalarCodec<E2M1>::decode(bits);
}
__device__ __forceinline__ uint8_t float_to_e2m1(float value) noexcept {
  return ScalarCodec<E2M1>::encode(value);
}
__device__ __forceinline__ float e8m0_to_float(const uint8_t bits) noexcept {
  return ScalarCodec<E8M0>::decode(bits);
}

__device__ __forceinline__ uint8_t
float_to_fp8_native(const float value, const bool fnuz) noexcept {
  if (isnan(value)) {
    return fnuz ? UINT8_C(0x80) : UINT8_C(0x7e);
  }
  if (isinf(value)) {
    if (fnuz) {
      return signbit(value) ? UINT8_C(0xff) : UINT8_C(0x7f);
    }
    return signbit(value) ? UINT8_C(0xfe) : UINT8_C(0x7e);
  }
  return __hip_cvt_float_to_fp8(value, __HIP_SATFINITE,
                                fnuz ? __HIP_E4M3_FNUZ : __HIP_E4M3);
}

__device__ __forceinline__ uint8_t
ocp_mx_scale_code(const float maximum, const int32_t element_power) noexcept {
  if (isnan(maximum)) {
    return UINT8_C(0xff);
  }
  if (maximum == 0.0F || isinf(maximum)) {
    return UINT8_C(127);
  }
  const uint32_t bits = __float_as_uint(maximum) & UINT32_C(0x7fffffff);
  const uint32_t biased = (bits >> 23U) & UINT32_C(0xff);
  int32_t floor_exponent = 0;
  if (biased != 0U) {
    floor_exponent = static_cast<int32_t>(biased) - 127;
  } else {
    const uint32_t mantissa = bits & UINT32_C(0x007fffff);
    floor_exponent = static_cast<int32_t>(31 - __builtin_clz(mantissa)) - 149;
  }
  int32_t exponent = floor_exponent - element_power;
  exponent = exponent < -127 ? -127 : (exponent > 127 ? 127 : exponent);
  return static_cast<uint8_t>(exponent + 127);
}

__device__ __forceinline__ uint8_t
ocp_mxfp8_e8m0_scale(const float maximum, const bool e5) noexcept {
  if (!(maximum > 0.0F) || !isfinite(maximum)) {
    return UINT8_C(127);
  }
  return ocp_mx_scale_code(maximum, e5 ? 15 : 8);
}

__device__ __forceinline__ uint8_t
mxfp4_even_scale_code(const float maximum) noexcept {
  if (!isfinite(maximum)) {
    return UINT8_C(0xff);
  }
  if (maximum == 0.0F) {
    return 0U;
  }
  const uint32_t rounded_exponent =
      (__float_as_uint(maximum) + UINT32_C(0x00200000)) & UINT32_C(0x7f800000);
  int32_t code = static_cast<int32_t>(rounded_exponent >> 23U) - 2;
  code = code < 0 ? 0 : (code > 254 ? 254 : code);
  return static_cast<uint8_t>(code);
}

__device__ __forceinline__ uint8_t
packed_e3m2_at(const uint8_t *const row, const uint64_t index) noexcept {
  const uint64_t byte = (index / UINT64_C(4)) * UINT64_C(3);
  const uint32_t packed = static_cast<uint32_t>(row[byte]) |
                          (static_cast<uint32_t>(row[byte + 1U]) << 8U) |
                          (static_cast<uint32_t>(row[byte + 2U]) << 16U);
  return static_cast<uint8_t>(
      (packed >> static_cast<uint32_t>((index & UINT64_C(3)) * UINT64_C(6))) &
      UINT32_C(0x3f));
}

__device__ __forceinline__ uint32_t packed_e3m2x4_at(
    const uint8_t *const row, const uint64_t first_index) noexcept {
  const uint64_t byte = (first_index / UINT64_C(4)) * UINT64_C(3);
  return static_cast<uint32_t>(row[byte]) |
         (static_cast<uint32_t>(row[byte + 1U]) << 8U) |
         (static_cast<uint32_t>(row[byte + 2U]) << 16U);
}

__device__ __forceinline__ uint8_t
packed_nibble_at(const uint8_t *const row, const uint64_t index) noexcept {
  const uint8_t packed = row[index / UINT64_C(2)];
  return (index & UINT64_C(1)) == 0U
             ? static_cast<uint8_t>(packed & UINT8_C(0x0f))
             : static_cast<uint8_t>(packed >> 4U);
}

template <uint32_t Width = 32U>
__device__ __forceinline__ float wave_amax(float value) noexcept {
#pragma unroll
  for (uint32_t offset = Width / 2U; offset != 0U; offset >>= 1U) {
    value = fmaxf(value, __shfl_down(value, offset, Width));
  }
  return value;
}

template <uint32_t Width = 32U>
__device__ __forceinline__ uint32_t wave_or(uint32_t value) noexcept {
#pragma unroll
  for (uint32_t offset = Width / 2U; offset != 0U; offset >>= 1U) {
    value |= __shfl_down(value, offset, Width);
  }
  return value;
}

template <uint32_t Width = 32U>
__device__ __forceinline__ uint32_t wave_and(uint32_t value) noexcept {
#pragma unroll
  for (uint32_t offset = Width / 2U; offset != 0U; offset >>= 1U) {
    value &= __shfl_down(value, offset, Width);
  }
  return value;
}

struct Mxfp8E4Block32 {
  using Element = E4M3Fn;
  static constexpr uint32_t kBlockSize = 32U;
  static constexpr uint32_t kBitsPerElement = 8U;
  static constexpr int32_t kElementPower = 8;
  static constexpr bool kPacked = false;
  static constexpr bool kHasOuterScale = false;
};

struct Mxfp8E5Block32 {
  using Element = E5M2;
  static constexpr uint32_t kBlockSize = 32U;
  static constexpr uint32_t kBitsPerElement = 8U;
  static constexpr int32_t kElementPower = 15;
  static constexpr bool kPacked = false;
  static constexpr bool kHasOuterScale = false;
};

struct Mxfp6E3Block32 {
  using Element = E3M2;
  static constexpr uint32_t kBlockSize = 32U;
  static constexpr uint32_t kBitsPerElement = 6U;
  static constexpr int32_t kElementPower = 4;
  static constexpr bool kPacked = true;
  static constexpr bool kHasOuterScale = false;
};

struct Nvfp4Block16 {
  using Element = E2M1;
  static constexpr uint32_t kBlockSize = 16U;
  static constexpr uint32_t kBitsPerElement = 4U;
  static constexpr bool kPacked = true;
  static constexpr bool kHasOuterScale = true;
};

struct Mxfp4E2Block32 {
  using Element = E2M1;
  static constexpr uint32_t kBlockSize = 32U;
  static constexpr uint32_t kBitsPerElement = 4U;
  static constexpr bool kPacked = true;
  static constexpr bool kHasOuterScale = false;
};

template <typename BlockFormat> struct BlockScaledView {
  const uint8_t *values;
  const uint8_t *block_scales;
  const float *outer_scales;
  uint64_t logical_columns;
  uint64_t value_stride;
  uint64_t scale_stride;
};

template <typename BlockFormat> struct MutableBlockScaledView {
  uint8_t *values;
  uint8_t *block_scales;
  float *outer_scales;
  uint64_t logical_columns;
  uint64_t value_stride;
  uint64_t scale_stride;
};

template <typename BlockFormat> struct BlockCodec;

template <> struct BlockCodec<Mxfp8E4Block32> {
  using Format = Mxfp8E4Block32;
  __device__ __forceinline__ static uint8_t
  scale_code(const float maximum, const bool has_nan = false) noexcept {
    return ocp_mx_scale_code(has_nan ? NAN : maximum, Format::kElementPower);
  }
  __device__ __forceinline__ static float
  load(const BlockScaledView<Format> &view, const uint64_t row,
       const uint32_t element) noexcept {
    const uint8_t value = view.values[row * view.value_stride + element];
    const uint8_t scale = view.block_scales[row * view.scale_stride +
                                            element / Format::kBlockSize];
    return ScalarCodec<E4M3Fn>::decode_scaled(value, scale);
  }
};

template <> struct BlockCodec<Mxfp8E5Block32> {
  using Format = Mxfp8E5Block32;
  __device__ __forceinline__ static uint8_t
  scale_code(const float maximum, const bool has_nan = false) noexcept {
    return ocp_mx_scale_code(has_nan ? NAN : maximum, Format::kElementPower);
  }
  __device__ __forceinline__ static float
  load(const BlockScaledView<Format> &view, const uint64_t row,
       const uint32_t element) noexcept {
    const uint8_t value = view.values[row * view.value_stride + element];
    const uint8_t scale = view.block_scales[row * view.scale_stride +
                                            element / Format::kBlockSize];
    return ScalarCodec<E5M2>::decode(value) * ScalarCodec<E8M0>::decode(scale);
  }
};

template <> struct BlockCodec<Mxfp6E3Block32> {
  using Format = Mxfp6E3Block32;
  __device__ __forceinline__ static uint8_t
  scale_code(const float maximum, const bool has_nan = false) noexcept {
    return ocp_mx_scale_code(has_nan ? NAN : maximum, Format::kElementPower);
  }
  __device__ __forceinline__ static float
  load(const BlockScaledView<Format> &view, const uint64_t row,
       const uint32_t element) noexcept {
    const uint8_t *const values = view.values + row * view.value_stride;
    const uint8_t value = packed_e3m2_at(values, element);
    const uint8_t scale = view.block_scales[row * view.scale_stride +
                                            element / Format::kBlockSize];
    return ScalarCodec<E3M2>::decode(value) * ScalarCodec<E8M0>::decode(scale);
  }
};

template <> struct BlockCodec<Nvfp4Block16> {
  using Format = Nvfp4Block16;
  __device__ __forceinline__ static float
  load(const BlockScaledView<Format> &view, const uint64_t row,
       const uint32_t element) noexcept {
    const uint8_t *const values = view.values + row * view.value_stride;
    const uint8_t value = packed_nibble_at(values, element);
    const uint8_t scale = view.block_scales[row * view.scale_stride +
                                            element / Format::kBlockSize];
    return ScalarCodec<E2M1>::decode(value) *
           ScalarCodec<E4M3Fn>::decode(scale) * view.outer_scales[row];
  }
};

template <> struct BlockCodec<Mxfp4E2Block32> {
  using Format = Mxfp4E2Block32;
  __device__ __forceinline__ static uint8_t
  scale_code(const float maximum, const bool has_nan = false) noexcept {
    return mxfp4_even_scale_code(has_nan ? NAN : maximum);
  }
  __device__ __forceinline__ static float
  load(const BlockScaledView<Format> &view, const uint64_t row,
       const uint32_t element) noexcept {
    const uint8_t *const values = view.values + row * view.value_stride;
    const uint8_t value = packed_nibble_at(values, element);
    const uint8_t scale = view.block_scales[row * view.scale_stride +
                                            element / Format::kBlockSize];
    return ScalarCodec<E2M1>::decode(value) * ScalarCodec<E8M0>::decode(scale);
  }
};

template <typename BlockFormat>
__device__ __forceinline__ BlockScaledView<BlockFormat>
make_block_scaled_view(const void *const values, const void *const scales,
                       const float *const outer_scales,
                       const uint64_t logical_columns) noexcept {
  constexpr uint64_t block = BlockFormat::kBlockSize;
  const uint64_t blocks_per_row = (logical_columns + block - 1U) / block;
  uint64_t value_stride = logical_columns;
  if constexpr (BlockFormat::kPacked) {
    if constexpr (BlockFormat::kBitsPerElement == 4U) {
      value_stride = (logical_columns + 1U) / 2U;
    } else if constexpr (BlockFormat::kBitsPerElement == 6U) {
      value_stride = ((logical_columns + 3U) / 4U) * 3U;
    }
  } else {
    value_stride = blocks_per_row * block;
  }
  return {static_cast<const uint8_t *>(values),
          static_cast<const uint8_t *>(scales),
          outer_scales,
          logical_columns,
          value_stride,
          blocks_per_row};
}

} // namespace sllm_lowp

#endif // SLLM_LOW_PRECISION_BLOCK_CODEC_HPP
