#include <hip/hip_runtime.h>
// SQ8_1 is an independent I8/F16 artifact ABI.  The payload pointer and each
// row start are validated as 16-byte aligned by the host launcher.  A complete
// K=32 block therefore has exactly two aligned 16-byte loads.
typedef unsigned int ullm_sq8_1_uint4 __attribute__((ext_vector_type(4)));
static_assert(sizeof(ullm_sq8_1_uint4) == 16u, "SQ8_1 uint4 must be 16 bytes");
static_assert(alignof(ullm_sq8_1_uint4) == 16u, "SQ8_1 uint4 must be aligned");

__device__ unsigned int ullm_sq8_1_word_byte(ullm_sq8_1_uint4 words, unsigned int index) {
    const unsigned int word_index = index >> 2u;
    const unsigned int word = word_index == 0u ? words.x :
        (word_index == 1u ? words.y : (word_index == 2u ? words.z : words.w));
    return (word >> ((index & 3u) * 8u)) & 0xffu;
}

__device__ int ullm_sq8_1_signed_byte(unsigned int value) {
    return value >= 128u ? static_cast<int>(value) - 256 : static_cast<int>(value);
}

__device__ float ullm_sq8_1_f16_to_f32(unsigned short bits) {
    const unsigned int sign = static_cast<unsigned int>(bits >> 15u);
    const unsigned int exponent = (static_cast<unsigned int>(bits) >> 10u) & 0x1fu;
    const unsigned int mantissa = static_cast<unsigned int>(bits) & 0x03ffu;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            return sign == 0u ? 0.0f : -0.0f;
        }
        const float value = static_cast<float>(mantissa) * 5.960464477539063e-8f;
        return sign == 0u ? value : -value;
    }
    if (exponent == 31u) {
        return __uint_as_float((sign << 31u) | 0x7f800000u | (mantissa << 13u));
    }
    return __uint_as_float(
        (sign << 31u) | ((exponent + 112u) << 23u) | (mantissa << 13u));
}

__device__ unsigned int ullm_sq8_1_round_shift_ties_even(unsigned int value, unsigned int shift) {
    if (shift == 0u) {
        return value;
    }
    const unsigned int quotient = value >> shift;
    const unsigned int remainder = value & ((1u << shift) - 1u);
    const unsigned int halfway = 1u << (shift - 1u);
    return remainder > halfway || (remainder == halfway && (quotient & 1u) != 0u)
        ? quotient + 1u
        : quotient;
}

__device__ unsigned short ullm_sq8_1_f32_to_f16_rne_bits(float value) {
    const unsigned int raw = __float_as_uint(value);
    const unsigned short sign = static_cast<unsigned short>((raw >> 16u) & 0x8000u);
    const unsigned int exponent_bits = (raw >> 23u) & 0xffu;
    const unsigned int mantissa = raw & 0x7fffffu;
    if (exponent_bits == 0xffu) {
        return static_cast<unsigned short>(sign | (mantissa == 0u ? 0x7c00u : 0x7e00u));
    }
    const int exponent = static_cast<int>(exponent_bits) - 127;
    if (exponent > 15) {
        return static_cast<unsigned short>(sign | 0x7c00u);
    }
    if (exponent >= -14) {
        unsigned int rounded = ullm_sq8_1_round_shift_ties_even(mantissa, 13u);
        unsigned int half_exponent = static_cast<unsigned int>(exponent + 15);
        if (rounded == 0x400u) {
            rounded = 0u;
            ++half_exponent;
            if (half_exponent >= 0x1fu) {
                return static_cast<unsigned short>(sign | 0x7c00u);
            }
        }
        return static_cast<unsigned short>(sign | (half_exponent << 10u) | rounded);
    }
    if (exponent < -25) {
        return sign;
    }
    const unsigned int significand = mantissa | 0x800000u;
    const unsigned int subnormal = ullm_sq8_1_round_shift_ties_even(
        significand,
        static_cast<unsigned int>(-exponent - 1));
    return static_cast<unsigned short>(sign | subnormal);
}

__device__ float ullm_sq8_1_ceil_f16(float value) {
    unsigned short bits = ullm_sq8_1_f32_to_f16_rne_bits(value);
    if (bits == 0u) {
        return ullm_sq8_1_f16_to_f32(1u);
    }
    float stored = ullm_sq8_1_f16_to_f32(bits);
    if (stored < value) {
        ++bits;
        stored = ullm_sq8_1_f16_to_f32(bits);
    }
    return stored;
}

// Semantic baseline: signed I8 x signed I8 accumulation.  RDNA3/RDNA4 use
// VOP3P `v_dot4_i32_iu8` with both signed controls, while gfx1030/CDNA use the
// legacy cumulative spelling.  Both forms implement v_dot4_i32_i8 semantics.
__device__ __forceinline__ int ullm_sq8_1_dot4_i32_i8(int lhs, int rhs, int accum) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_sudot4(true, lhs, true, rhs, accum, false);
#elif defined(__gfx1030__) || defined(__gfx942__) || defined(__gfx950__)
    return __builtin_amdgcn_sdot4(lhs, rhs, accum, false);
#else
    return accum;
#endif
}

__device__ float ullm_sq8_1_reduce_sum(float value, float *partial) {
    partial[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1u; offset > 0u; offset >>= 1u) {
        if (threadIdx.x < offset) {
            partial[threadIdx.x] += partial[threadIdx.x + offset];
        }
        __syncthreads();
    }
    return partial[0];
}

extern "C" __global__ void ullm_sq8_1_matvec_w8a16_f32_kernel(
    const unsigned char *payload,
    const unsigned short *weight_scales,
    const float *input,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long payload_row_stride,
    float *output) {
    const unsigned long long row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum = 0.0f;
    if (row < rows) {
        const unsigned long long groups = (cols + 31ull) / 32ull;
        const unsigned char *row_payload = payload + row * payload_row_stride;
        for (unsigned long long group = tid; group < groups; group += blockDim.x) {
            const unsigned long long start = group * 32ull;
            const unsigned int count = static_cast<unsigned int>(
                (cols - start) < 32ull ? (cols - start) : 32ull);
            float dot = 0.0f;
            if (count == 32u) {
                // Two uint4 loads, each covering sixteen of this aligned K=32 block.
                const ullm_sq8_1_uint4 first =
                    *reinterpret_cast<const ullm_sq8_1_uint4 *>(row_payload + start);
                const ullm_sq8_1_uint4 second =
                    *reinterpret_cast<const ullm_sq8_1_uint4 *>(row_payload + start + 16ull);
                for (unsigned int index = 0u; index < 16u; ++index) {
                    dot += static_cast<float>(ullm_sq8_1_signed_byte(ullm_sq8_1_word_byte(first, index))) *
                        input[start + index];
                }
                for (unsigned int index = 0u; index < 16u; ++index) {
                    dot += static_cast<float>(ullm_sq8_1_signed_byte(ullm_sq8_1_word_byte(second, index))) *
                        input[start + 16ull + index];
                }
            } else {
                for (unsigned int index = 0u; index < count; ++index) {
                    dot += static_cast<float>(ullm_sq8_1_signed_byte(row_payload[start + index])) *
                        input[start + index];
                }
            }
            sum += dot * ullm_sq8_1_f16_to_f32(weight_scales[row * groups + group]);
        }
    }
    const float reduced = ullm_sq8_1_reduce_sum(sum, partial);
    if (tid == 0u && row < rows) {
        output[row] = reduced;
    }
}

extern "C" __global__ void ullm_sq8_1_matvec_w8a8_explicit_f32_kernel(
    const unsigned char *payload,
    const unsigned short *weight_scales,
    const float *input,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long payload_row_stride,
    float *output) {
    const unsigned long long row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum = 0.0f;
    if (row < rows) {
        const unsigned long long groups = (cols + 31ull) / 32ull;
        const unsigned char *row_payload = payload + row * payload_row_stride;
        for (unsigned long long group = tid; group < groups; group += blockDim.x) {
            const unsigned long long start = group * 32ull;
            const unsigned int count = static_cast<unsigned int>(
                (cols - start) < 32ull ? (cols - start) : 32ull);
            float maximum = 0.0f;
            for (unsigned int index = 0u; index < count; ++index) {
                maximum = fmaxf(maximum, fabsf(input[start + index]));
            }
            const float activation_scale = maximum == 0.0f
                ? 1.0f
                : ullm_sq8_1_ceil_f16(maximum / 127.0f);
            int activation_codes[32];
            for (unsigned int index = 0u; index < 32u; ++index) {
                if (index >= count) {
                    activation_codes[index] = 0;
                } else {
                    const int rounded = static_cast<int>(rintf(input[start + index] / activation_scale));
                    activation_codes[index] = rounded < -127 ? -127 : (rounded > 127 ? 127 : rounded);
                }
            }
            int dot = 0;
            if (count == 32u) {
                // The full logical block is two aligned uint4 payload loads.
                const ullm_sq8_1_uint4 first =
                    *reinterpret_cast<const ullm_sq8_1_uint4 *>(row_payload + start);
                const ullm_sq8_1_uint4 second =
                    *reinterpret_cast<const ullm_sq8_1_uint4 *>(row_payload + start + 16ull);
                const unsigned int data_words[8] = {
                    first.x, first.y, first.z, first.w, second.x, second.y, second.z, second.w};
                for (unsigned int word = 0u; word < 8u; ++word) {
                    unsigned int activation_word = 0u;
                    for (unsigned int byte = 0u; byte < 4u; ++byte) {
                        activation_word |= (static_cast<unsigned int>(activation_codes[word * 4u + byte]) & 0xffu)
                            << (byte * 8u);
                    }
                    dot = ullm_sq8_1_dot4_i32_i8(
                        static_cast<int>(data_words[word]), static_cast<int>(activation_word), dot);
                }
            } else {
                for (unsigned int word = 0u; word < 8u; ++word) {
                    unsigned int data_word = 0u;
                    unsigned int activation_word = 0u;
                    for (unsigned int byte = 0u; byte < 4u; ++byte) {
                        const unsigned int index = word * 4u + byte;
                        if (index < count) {
                            data_word |= static_cast<unsigned int>(row_payload[start + index]) << (byte * 8u);
                        }
                        activation_word |= (static_cast<unsigned int>(activation_codes[index]) & 0xffu)
                            << (byte * 8u);
                    }
                    dot = ullm_sq8_1_dot4_i32_i8(
                        static_cast<int>(data_word), static_cast<int>(activation_word), dot);
                }
            }
            const float weight_scale = ullm_sq8_1_f16_to_f32(weight_scales[row * groups + group]);
            sum += static_cast<float>(dot) * weight_scale * activation_scale;
        }
    }
    const float reduced = ullm_sq8_1_reduce_sum(sum, partial);
    if (tid == 0u && row < rows) {
        output[row] = reduced;
    }
}
