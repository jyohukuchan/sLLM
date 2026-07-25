// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

    static const char *sq8_1_matvec_kernel_source() {
        return R"(
// SQ8_1 is an independent I8/F16 artifact ABI.  The payload pointer and each
// row start are validated as 16-byte aligned by the host launcher.  A complete
// K=32 block therefore has exactly two aligned 16-byte loads.
typedef unsigned int ullm_sq8_1_uint4 __attribute__((ext_vector_type(4)));
static_assert(sizeof(ullm_sq8_1_uint4) == 16u, "SQ8_1 uint4 must be 16 bytes");
static_assert(alignof(ullm_sq8_1_uint4) == 16u, "SQ8_1 uint4 must be aligned");

constexpr unsigned int kUllmSq8_1WaveSize = 32u;
constexpr unsigned int kUllmSq8_1RowsPerBlock = 8u;

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

// The runtime launches 256 threads as eight logical wave32 rows.  Supplying a
// fixed width keeps two logical rows independent when a CDNA target exposes a
// wave64.  The caller uses this only for reductions within one logical row.
__device__ __forceinline__ float ullm_sq8_1_wave32_sum(float value) {
#pragma unroll
    for (unsigned int offset = kUllmSq8_1WaveSize >> 1u; offset > 0u; offset >>= 1u) {
        value += __shfl_down(value, offset, kUllmSq8_1WaveSize);
    }
    return value;
}

__device__ __forceinline__ float ullm_sq8_1_wave32_broadcast_lane0(float value) {
    // readlane indexes the physical wave.  Offset a logical second half-wave
    // on wave64 targets so it does not accidentally read physical lane zero.
    const unsigned int hardware_lane = threadIdx.x & (static_cast<unsigned int>(warpSize) - 1u);
    return __uint_as_float(__builtin_amdgcn_readlane(
        __float_as_uint(value), static_cast<int>(hardware_lane & ~31u)));
}

__device__ __forceinline__ float ullm_sq8_1_dot4_w8a16(
    unsigned int packed_weights,
    float4 activation,
    float accum) {
    accum = fmaf(
        static_cast<float>(static_cast<signed char>(packed_weights & 0xffu)), activation.x, accum);
    accum = fmaf(
        static_cast<float>(static_cast<signed char>((packed_weights >> 8u) & 0xffu)), activation.y, accum);
    accum = fmaf(
        static_cast<float>(static_cast<signed char>((packed_weights >> 16u) & 0xffu)), activation.z, accum);
    return fmaf(
        static_cast<float>(static_cast<signed char>((packed_weights >> 24u) & 0xffu)), activation.w, accum);
}

__device__ __forceinline__ float ullm_sq8_1_dot32_w8a16(
    ullm_sq8_1_uint4 first,
    ullm_sq8_1_uint4 second,
    const float4 *activation) {
    float dot = 0.0f;
    dot = ullm_sq8_1_dot4_w8a16(first.x, activation[0], dot);
    dot = ullm_sq8_1_dot4_w8a16(first.y, activation[1], dot);
    dot = ullm_sq8_1_dot4_w8a16(first.z, activation[2], dot);
    dot = ullm_sq8_1_dot4_w8a16(first.w, activation[3], dot);
    dot = ullm_sq8_1_dot4_w8a16(second.x, activation[4], dot);
    dot = ullm_sq8_1_dot4_w8a16(second.y, activation[5], dot);
    dot = ullm_sq8_1_dot4_w8a16(second.z, activation[6], dot);
    return ullm_sq8_1_dot4_w8a16(second.w, activation[7], dot);
}

__device__ __forceinline__ int ullm_sq8_1_dot32_w8a8(
    ullm_sq8_1_uint4 weights_first,
    ullm_sq8_1_uint4 weights_second,
    ullm_sq8_1_uint4 activation_first,
    ullm_sq8_1_uint4 activation_second) {
    int dot = 0;
    dot = ullm_sq8_1_dot4_i32_i8(
        static_cast<int>(weights_first.x), static_cast<int>(activation_first.x), dot);
    dot = ullm_sq8_1_dot4_i32_i8(
        static_cast<int>(weights_first.y), static_cast<int>(activation_first.y), dot);
    dot = ullm_sq8_1_dot4_i32_i8(
        static_cast<int>(weights_first.z), static_cast<int>(activation_first.z), dot);
    dot = ullm_sq8_1_dot4_i32_i8(
        static_cast<int>(weights_first.w), static_cast<int>(activation_first.w), dot);
    dot = ullm_sq8_1_dot4_i32_i8(
        static_cast<int>(weights_second.x), static_cast<int>(activation_second.x), dot);
    dot = ullm_sq8_1_dot4_i32_i8(
        static_cast<int>(weights_second.y), static_cast<int>(activation_second.y), dot);
    dot = ullm_sq8_1_dot4_i32_i8(
        static_cast<int>(weights_second.z), static_cast<int>(activation_second.z), dot);
    return ullm_sq8_1_dot4_i32_i8(
        static_cast<int>(weights_second.w), static_cast<int>(activation_second.w), dot);
}

// Eight independent rows share one 256-thread workgroup.  Each logical
// wave32 owns one row and walks five K=32 blocks for the common 5120-wide
// decode shape.  This removes the reference LDS tree entirely while retaining
// two aligned uint4 loads per complete K=32 block.
extern "C" __global__ __launch_bounds__(256) void ullm_sq8_1_matvec_w8a16_f32_kernel(
    const unsigned char *payload,
    const unsigned short *weight_scales,
    const float *input,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long payload_row_stride,
    float *output) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (kUllmSq8_1WaveSize - 1u);
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x) *
        static_cast<unsigned long long>(kUllmSq8_1RowsPerBlock) + (tid >> 5u);
    if (row >= rows) {
        return;
    }
    float sum = 0.0f;
    const unsigned long long groups = (cols + 31ull) / 32ull;
    const unsigned char *row_payload = payload + row * payload_row_stride;
    for (unsigned long long group = lane; group < groups; group += kUllmSq8_1WaveSize) {
        const unsigned long long start = group * 32ull;
        const unsigned int count = static_cast<unsigned int>(
            (cols - start) < 32ull ? (cols - start) : 32ull);
        float dot = 0.0f;
        if (count == 32u) {
            // Exactly two aligned 128-bit payload loads; input is also K=32 aligned.
            const ullm_sq8_1_uint4 first =
                *reinterpret_cast<const ullm_sq8_1_uint4 *>(row_payload + start);
            const ullm_sq8_1_uint4 second =
                *reinterpret_cast<const ullm_sq8_1_uint4 *>(row_payload + start + 16ull);
            dot = ullm_sq8_1_dot32_w8a16(
                first, second, reinterpret_cast<const float4 *>(input + start));
        } else {
            for (unsigned int index = 0u; index < count; ++index) {
                dot = fmaf(
                    static_cast<float>(ullm_sq8_1_signed_byte(row_payload[start + index])),
                    input[start + index],
                    dot);
            }
        }
        sum += dot * ullm_sq8_1_f16_to_f32(weight_scales[row * groups + group]);
    }
    const float reduced = ullm_sq8_1_wave32_sum(sum);
    if (lane == 0u) {
        output[row] = reduced;
    }
}

// W8A8 reuses one dynamically quantized K=32 activation plane for eight rows.
// It turns the reference implementation's per-row activation quantization
// into per-row-tile work while retaining its explicit-only public ABI.  The
// host only selects this path when its dynamic LDS request is within the
// conservative 48 KiB cap; otherwise the fallback below keeps all shapes valid.
extern "C" __global__ __launch_bounds__(256) void ullm_sq8_1_matvec_w8a8_explicit_f32_kernel(
    const unsigned char *payload,
    const unsigned short *weight_scales,
    const float *input,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long payload_row_stride,
    float *output) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (kUllmSq8_1WaveSize - 1u);
    const unsigned int logical_wave = tid >> 5u;
    const unsigned long long groups = (cols + 31ull) / 32ull;
    const unsigned long long activation_code_bytes = ((cols + 15ull) / 16ull) * 16ull;
    extern __shared__ __attribute__((aligned(16))) unsigned char activation_storage[];
    unsigned char *activation_codes = activation_storage;
    float *activation_scales = reinterpret_cast<float *>(activation_storage + activation_code_bytes);

    // Eight logical waves quantize eight independent blocks at a time.  Each
    // lane owns one activation element, so scale reduction and packing require
    // no LDS until the completed codes are consumed by all eight output rows.
    for (unsigned long long group = logical_wave; group < groups;
         group += kUllmSq8_1RowsPerBlock) {
        const unsigned long long start = group * 32ull;
        const unsigned int count = static_cast<unsigned int>(
            (cols - start) < 32ull ? (cols - start) : 32ull);
        const float value = lane < count ? input[start + lane] : 0.0f;
        float maximum = fabsf(value);
#pragma unroll
        for (unsigned int offset = kUllmSq8_1WaveSize >> 1u; offset > 0u; offset >>= 1u) {
            maximum = fmaxf(maximum, __shfl_down(maximum, offset, kUllmSq8_1WaveSize));
        }
        float activation_scale = lane == 0u
            ? (maximum == 0.0f ? 1.0f : ullm_sq8_1_ceil_f16(maximum / 127.0f))
            : 0.0f;
        activation_scale = ullm_sq8_1_wave32_broadcast_lane0(activation_scale);
        if (lane == 0u) {
            activation_scales[group] = activation_scale;
        }
        if (lane < count) {
            const int rounded = static_cast<int>(rintf(value / activation_scale));
            const int clamped = rounded < -127 ? -127 : (rounded > 127 ? 127 : rounded);
            activation_codes[start + lane] = static_cast<unsigned char>(static_cast<signed char>(clamped));
        }
    }
    __syncthreads();

    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x) *
        static_cast<unsigned long long>(kUllmSq8_1RowsPerBlock) + logical_wave;
    float sum = 0.0f;
    if (row < rows) {
        const unsigned char *row_payload = payload + row * payload_row_stride;
        for (unsigned long long group = lane; group < groups; group += kUllmSq8_1WaveSize) {
            const unsigned long long start = group * 32ull;
            const unsigned int count = static_cast<unsigned int>(
                (cols - start) < 32ull ? (cols - start) : 32ull);
            int dot = 0;
            if (count == 32u) {
                // Both planes are K=32/16-byte aligned: two uint4 loads each.
                const ullm_sq8_1_uint4 weights_first =
                    *reinterpret_cast<const ullm_sq8_1_uint4 *>(row_payload + start);
                const ullm_sq8_1_uint4 weights_second =
                    *reinterpret_cast<const ullm_sq8_1_uint4 *>(row_payload + start + 16ull);
                const ullm_sq8_1_uint4 activation_first =
                    *reinterpret_cast<const ullm_sq8_1_uint4 *>(activation_codes + start);
                const ullm_sq8_1_uint4 activation_second =
                    *reinterpret_cast<const ullm_sq8_1_uint4 *>(activation_codes + start + 16ull);
                dot = ullm_sq8_1_dot32_w8a8(
                    weights_first, weights_second, activation_first, activation_second);
            } else {
                for (unsigned int index = 0u; index < count; ++index) {
                    dot += ullm_sq8_1_signed_byte(row_payload[start + index]) *
                        static_cast<int>(static_cast<signed char>(activation_codes[start + index]));
                }
            }
            const float weight_scale = ullm_sq8_1_f16_to_f32(weight_scales[row * groups + group]);
            sum += static_cast<float>(dot) * weight_scale * activation_scales[group];
        }
    }
    const float reduced = ullm_sq8_1_wave32_sum(sum);
    if (lane == 0u && row < rows) {
        output[row] = reduced;
    }
}

// Shape-preserving fallback for W8A8 calls whose activation plane would exceed
// the bounded dynamic LDS reservation.  It still uses wave reduction and
// aligned payload loads, but intentionally retains per-row quantization.
extern "C" __global__ __launch_bounds__(256) void ullm_sq8_1_matvec_w8a8_explicit_fallback_f32_kernel(
    const unsigned char *payload,
    const unsigned short *weight_scales,
    const float *input,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long payload_row_stride,
    float *output) {
    const unsigned long long row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (kUllmSq8_1WaveSize - 1u);
    const unsigned int logical_wave = tid >> 5u;
    __shared__ float wave_sums[kUllmSq8_1RowsPerBlock];
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
    const float reduced = ullm_sq8_1_wave32_sum(sum);
    if (lane == 0u) {
        wave_sums[logical_wave] = reduced;
    }
    __syncthreads();
    if (tid < kUllmSq8_1WaveSize) {
        const float block_sum = tid < kUllmSq8_1RowsPerBlock ? wave_sums[tid] : 0.0f;
        const float final_sum = ullm_sq8_1_wave32_sum(block_sum);
        if (tid == 0u && row < rows) {
            output[row] = final_sum;
        }
    }
}
)";
    }
