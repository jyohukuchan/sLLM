// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
    static const char *paged_kv_write_kernel_source() {
        return R"(
extern "C" __global__ void ullm_paged_kv_write_f32_kernel(
    const float *k,
    const float *v,
    const unsigned int *block_table,
    unsigned long long cache_position,
    unsigned long long block_size,
    unsigned long long cache_blocks,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float *k_cache,
    float *v_cache) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long k_elements = kv_heads * head_dim;
    const unsigned long long v_elements = kv_heads * value_dim;
    const unsigned long long total_elements = k_elements + v_elements;
    if (index >= total_elements) {
        return;
    }
    const unsigned long long block_index = cache_position / block_size;
    const unsigned long long block_offset = cache_position - block_index * block_size;
    const unsigned long long block_id = static_cast<unsigned long long>(block_table[block_index]);
    if (block_id >= cache_blocks) {
        return;
    }
    const unsigned long long physical_timestep = block_id * block_size + block_offset;
    if (index < k_elements) {
        const unsigned long long kv_head = index / head_dim;
        const unsigned long long dim = index - kv_head * head_dim;
        k_cache[(physical_timestep * kv_heads + kv_head) * head_dim + dim] = k[index];
        return;
    }
    const unsigned long long v_index = index - k_elements;
    const unsigned long long kv_head = v_index / value_dim;
    const unsigned long long dim = v_index - kv_head * value_dim;
    v_cache[(physical_timestep * kv_heads + kv_head) * value_dim + dim] = v[v_index];
}

// The typed kernels deliberately live alongside (rather than replacing) the
// F32 writer above.  F32/F32 therefore retains its old symbol, launch shape,
// and generated body.  Every non-F32 ordered K/V pair below has a separately
// instantiated body: there is no dtype branch in a payload loop.
enum {
    ULLM_TYPED_KV_F32 = 0,
    ULLM_TYPED_KV_F16 = 1,
    ULLM_TYPED_KV_FP8_E4M3FN = 2,
};

__device__ __forceinline__ unsigned int ullm_typed_kv_round_shift_ties_even(
    unsigned int value,
    unsigned int shift) {
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

__device__ __forceinline__ unsigned short ullm_typed_kv_f32_to_f16_rne(float value) {
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
        unsigned int rounded = ullm_typed_kv_round_shift_ties_even(mantissa, 13u);
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
    return static_cast<unsigned short>(
        sign | ullm_typed_kv_round_shift_ties_even(
                   significand, static_cast<unsigned int>(-exponent - 1)));
}

__device__ __forceinline__ float ullm_typed_kv_f16_to_f32(unsigned short bits) {
    const float sign = (bits & 0x8000u) == 0u ? 1.0f : -1.0f;
    const unsigned int exponent = (static_cast<unsigned int>(bits) >> 10u) & 0x1fu;
    const unsigned int mantissa = static_cast<unsigned int>(bits) & 0x03ffu;
    if (exponent == 0u) {
        return mantissa == 0u ? sign * 0.0f : sign * ldexpf(static_cast<float>(mantissa), -24);
    }
    if (exponent == 0x1fu) {
        return mantissa == 0u ? sign * __int_as_float(0x7f800000) : __int_as_float(0x7fc00000);
    }
    return sign * ldexpf(1.0f + static_cast<float>(mantissa) * 0.0009765625f,
                         static_cast<int>(exponent) - 15);
}

__device__ __forceinline__ bool ullm_typed_kv_positive_f16_ceil(
    float value,
    unsigned short *bits_out) {
    if (bits_out == nullptr || !isfinite(value) || value <= 0.0f) {
        return false;
    }
    unsigned short bits = ullm_typed_kv_f32_to_f16_rne(value);
    if (bits == 0u) {
        bits = 1u;
    }
    if (bits >= 0x7c00u) {
        return false;
    }
    if (ullm_typed_kv_f16_to_f32(bits) < value) {
        ++bits;
        if (bits >= 0x7c00u) {
            return false;
        }
    }
    const float stored = ullm_typed_kv_f16_to_f32(bits);
    if (!isfinite(stored) || stored <= 0.0f || stored < value) {
        return false;
    }
    *bits_out = bits;
    return true;
}

__device__ __forceinline__ float ullm_typed_kv_fp8_e4m3fn_to_f32(unsigned char value) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    // OCP E4M3FN, format selector 0.  This is the native RDNA4 conversion
    // used by the SQ8_0 HIPRTC path as well.
    return __builtin_amdgcn_cvt_f32_fp8(static_cast<unsigned int>(value), 0);
#else
    const unsigned int raw = static_cast<unsigned int>(value);
    const unsigned int sign = raw >> 7u;
    const unsigned int exponent = (raw >> 3u) & 0x0fu;
    const unsigned int mantissa = raw & 0x07u;
    if (exponent == 0x0fu && mantissa == 0x07u) {
        return __uint_as_float(0x7fc00000u);
    }
    const float magnitude = exponent == 0u
        ? static_cast<float>(mantissa) * 0.001953125f
        : ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f,
                 static_cast<int>(exponent) - 7);
    return sign == 0u ? magnitude : -magnitude;
#endif
}

__device__ __forceinline__ unsigned char ullm_typed_kv_f32_to_fp8_e4m3fn_rne(
    float value,
    float scale) {
    const unsigned int sign = (__float_as_uint(value) >> 24u) & 0x80u;
    const float magnitude = fabsf(value / scale);
    if (magnitude == 0.0f) {
        return static_cast<unsigned char>(sign);
    }
    if (magnitude >= 448.0f) {
        return static_cast<unsigned char>(sign | 0x7eu);
    }
    unsigned int lower_bound = 0u;
    unsigned int upper_bound = 126u;
    while (lower_bound < upper_bound) {
        const unsigned int middle = lower_bound + (upper_bound - lower_bound) / 2u;
        if (fabsf(ullm_typed_kv_fp8_e4m3fn_to_f32(static_cast<unsigned char>(middle))) < magnitude) {
            lower_bound = middle + 1u;
        } else {
            upper_bound = middle;
        }
    }
    const unsigned int upper = lower_bound;
    const float upper_value = ullm_typed_kv_fp8_e4m3fn_to_f32(static_cast<unsigned char>(upper));
    if (upper_value == magnitude) {
        return static_cast<unsigned char>(sign | upper);
    }
    const unsigned int lower = upper - 1u;
    const double lower_distance = static_cast<double>(magnitude) -
        static_cast<double>(ullm_typed_kv_fp8_e4m3fn_to_f32(static_cast<unsigned char>(lower)));
    const double upper_distance = static_cast<double>(upper_value) - static_cast<double>(magnitude);
    const unsigned int encoded = lower_distance < upper_distance
        ? lower
        : (upper_distance < lower_distance ? upper : ((lower & 1u) == 0u ? lower : upper));
    return static_cast<unsigned char>(sign | encoded);
}

template <int DTYPE>
__device__ __forceinline__ void ullm_typed_kv_store_row(
    const float *source,
    unsigned long long elements,
    unsigned long long physical_head,
    unsigned long long row_dim,
    void *payload,
    unsigned short *scales,
    float *reduction) {
    const unsigned int tid = threadIdx.x;
    float local_maximum = 0.0f;
    bool finite = true;
    if constexpr (DTYPE != ULLM_TYPED_KV_F32) {
        for (unsigned long long index = tid; index < elements; index += blockDim.x) {
            const float value = source[index];
            finite = finite && isfinite(value);
            local_maximum = fmaxf(local_maximum, fabsf(value));
        }
    }
    reduction[tid] = finite ? local_maximum : __int_as_float(0x7f800000);
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1u; offset > 0u; offset >>= 1u) {
        if (tid < offset) {
            reduction[tid] = fmaxf(reduction[tid], reduction[tid + offset]);
        }
        __syncthreads();
    }

    unsigned short scale_bits = 0x3c00u;
    bool valid = true;
    if constexpr (DTYPE == ULLM_TYPED_KV_F16) {
        valid = isfinite(reduction[0]) && reduction[0] <= 65504.0f;
    } else if constexpr (DTYPE == ULLM_TYPED_KV_FP8_E4M3FN) {
        if (!isfinite(reduction[0])) {
            valid = false;
        } else if (reduction[0] != 0.0f) {
            valid = ullm_typed_kv_positive_f16_ceil(reduction[0] / 448.0f, &scale_bits);
        }
    }
    if (tid == 0u) {
        reduction[0] = valid ? static_cast<float>(scale_bits) : -1.0f;
    }
    __syncthreads();
    if (reduction[0] < 0.0f) {
        return;
    }
    scale_bits = static_cast<unsigned short>(reduction[0]);
    if constexpr (DTYPE == ULLM_TYPED_KV_FP8_E4M3FN) {
        if (tid == 0u) {
            scales[physical_head] = scale_bits;
        }
    }
    const float scale = ullm_typed_kv_f16_to_f32(scale_bits);
    const unsigned long long destination = physical_head * row_dim;
    for (unsigned long long index = tid; index < elements; index += blockDim.x) {
        if constexpr (DTYPE == ULLM_TYPED_KV_F32) {
            static_cast<float *>(payload)[destination + index] = source[index];
        } else if constexpr (DTYPE == ULLM_TYPED_KV_F16) {
            static_cast<unsigned short *>(payload)[destination + index] =
                ullm_typed_kv_f32_to_f16_rne(source[index]);
        } else {
            static_cast<unsigned char *>(payload)[destination + index] =
                ullm_typed_kv_f32_to_fp8_e4m3fn_rne(source[index], scale);
        }
    }
}

template <int K_DTYPE, int V_DTYPE>
__device__ __forceinline__ void ullm_paged_kv_write_typed_kernel_impl(
    const float *k,
    const float *v,
    const unsigned int *block_table,
    unsigned long long cache_position,
    unsigned long long block_size,
    unsigned long long cache_blocks,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    void *k_cache,
    void *v_cache,
    unsigned short *k_scales,
    unsigned short *v_scales,
    float *reduction) {
    const unsigned long long plane_head = static_cast<unsigned long long>(blockIdx.x);
    const bool is_k = plane_head < kv_heads;
    const unsigned long long kv_head = is_k ? plane_head : plane_head - kv_heads;
    if (kv_head >= kv_heads) {
        return;
    }
    const unsigned long long block_index = cache_position / block_size;
    const unsigned long long block_offset = cache_position - block_index * block_size;
    const unsigned long long block_id = static_cast<unsigned long long>(block_table[block_index]);
    if (block_id >= cache_blocks) {
        return;
    }
    const unsigned long long physical = block_id * block_size + block_offset;
    const unsigned long long physical_head = physical * kv_heads + kv_head;
    if (is_k) {
        ullm_typed_kv_store_row<K_DTYPE>(
            k + kv_head * head_dim,
            head_dim,
            physical_head,
            head_dim,
            k_cache,
            k_scales,
            reduction);
    } else {
        ullm_typed_kv_store_row<V_DTYPE>(
            v + kv_head * value_dim,
            value_dim,
            physical_head,
            value_dim,
            v_cache,
            v_scales,
            reduction);
    }
}

#define ULLM_DEFINE_TYPED_PAGED_KV_WRITE(K_DTYPE, V_DTYPE, K_NAME, V_NAME) \
extern "C" __global__ void ullm_paged_kv_write_typed_k##K_NAME##_v##V_NAME##_kernel( \
    const float *k, const float *v, const unsigned int *block_table, \
    unsigned long long cache_position, unsigned long long block_size, \
    unsigned long long cache_blocks, unsigned long long kv_heads, \
    unsigned long long head_dim, unsigned long long value_dim, \
    void *k_cache, void *v_cache, unsigned short *k_scales, unsigned short *v_scales) { \
    __shared__ float reduction[256]; \
    ullm_paged_kv_write_typed_kernel_impl<K_DTYPE, V_DTYPE>( \
        k, v, block_table, cache_position, block_size, cache_blocks, kv_heads, head_dim, value_dim, \
        k_cache, v_cache, k_scales, v_scales, reduction); \
}

ULLM_DEFINE_TYPED_PAGED_KV_WRITE(ULLM_TYPED_KV_F32, ULLM_TYPED_KV_F16, f32, f16)
ULLM_DEFINE_TYPED_PAGED_KV_WRITE(ULLM_TYPED_KV_F32, ULLM_TYPED_KV_FP8_E4M3FN, f32, fp8)
ULLM_DEFINE_TYPED_PAGED_KV_WRITE(ULLM_TYPED_KV_F16, ULLM_TYPED_KV_F32, f16, f32)
ULLM_DEFINE_TYPED_PAGED_KV_WRITE(ULLM_TYPED_KV_F16, ULLM_TYPED_KV_F16, f16, f16)
ULLM_DEFINE_TYPED_PAGED_KV_WRITE(ULLM_TYPED_KV_F16, ULLM_TYPED_KV_FP8_E4M3FN, f16, fp8)
ULLM_DEFINE_TYPED_PAGED_KV_WRITE(ULLM_TYPED_KV_FP8_E4M3FN, ULLM_TYPED_KV_F32, fp8, f32)
ULLM_DEFINE_TYPED_PAGED_KV_WRITE(ULLM_TYPED_KV_FP8_E4M3FN, ULLM_TYPED_KV_F16, fp8, f16)
ULLM_DEFINE_TYPED_PAGED_KV_WRITE(ULLM_TYPED_KV_FP8_E4M3FN, ULLM_TYPED_KV_FP8_E4M3FN, fp8, fp8)
#undef ULLM_DEFINE_TYPED_PAGED_KV_WRITE
)";
    }

    static const char *paged_chunk_kernel_source() {
        return R"(
extern "C" __global__ void ullm_paged_kv_write_chunk_f32_kernel(
    const float *k,
    const float *v,
    const unsigned int *block_table,
    unsigned long long cache_start,
    unsigned long long m,
    unsigned long long block_size,
    unsigned long long cache_blocks,
    unsigned long long table_entries,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float *k_cache,
    float *v_cache) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long k_row = kv_heads * head_dim;
    const unsigned long long v_row = kv_heads * value_dim;
    const unsigned long long k_elements = m * k_row;
    const unsigned long long v_elements = m * v_row;
    if (index >= k_elements + v_elements || k_row == 0ull || v_row == 0ull) {
        return;
    }
    const bool is_k = index < k_elements;
    const unsigned long long row = is_k ? index / k_row : (index - k_elements) / v_row;
    const unsigned long long row_offset = is_k ? index - row * k_row : index - k_elements - row * v_row;
    const unsigned long long logical = cache_start + row;
    const unsigned long long block_index = logical / block_size;
    if (block_index >= table_entries) {
        return;
    }
    const unsigned long long block_offset = logical - block_index * block_size;
    const unsigned long long block_id = static_cast<unsigned long long>(block_table[block_index]);
    if (block_id >= cache_blocks) {
        return;
    }
    const unsigned long long physical = block_id * block_size + block_offset;
    if (is_k) {
        k_cache[(physical * kv_heads * head_dim) + row_offset] = k[row * k_row + row_offset];
    } else {
        v_cache[(physical * kv_heads * value_dim) + row_offset] = v[row * v_row + row_offset];
    }
}

extern "C" __global__ void ullm_paged_causal_gqa_chunk_f32_kernel(
    const float *q,
    const float *gate,
    const float *k_cache,
    const float *v_cache,
    const unsigned int *block_table,
    unsigned long long cached_prefix_len,
    unsigned long long m,
    unsigned long long block_size,
    unsigned long long cache_blocks,
    unsigned long long table_entries,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float *output) {
    const unsigned long long q_head_index = static_cast<unsigned long long>(blockIdx.x);
    const unsigned long long query_heads = m * q_heads;
    if (q_head_index >= query_heads || q_heads == 0ull || kv_heads == 0ull ||
        head_dim == 0ull || value_dim == 0ull || blockDim.x != 256u ||
        head_dim > 256ull || value_dim > 256ull) {
        return;
    }
    const unsigned int tid = threadIdx.x;
    const unsigned long long row = q_head_index / q_heads;
    const unsigned long long q_head = q_head_index - row * q_heads;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_base = (row * q_heads + q_head) * head_dim;
    const unsigned long long cache_len = cached_prefix_len + row + 1ull;
    __shared__ float partial[256];
    __shared__ float shared_prev_scale;
    __shared__ float shared_weight;
    __shared__ float shared_max_score;
    float weighted = 0.0f;
    float denominator = 0.0f;
    float max_score = -3.4028234663852886e38f;
    for (unsigned long long source = 0ull; source < cache_len; ++source) {
        const unsigned long long block_index = source / block_size;
        if (block_index >= table_entries) {
            if (tid < value_dim) {
                output[q_head_index * value_dim + tid] = 0.0f;
            }
            return;
        }
        const unsigned long long block_offset = source - block_index * block_size;
        const unsigned long long block_id = static_cast<unsigned long long>(block_table[block_index]);
        if (block_id >= cache_blocks) {
            if (tid < value_dim) {
                output[q_head_index * value_dim + tid] = 0.0f;
            }
            return;
        }
        const unsigned long long physical = block_id * block_size + block_offset;
        const unsigned long long k_base = (physical * kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            dot += q[q_base + dim] * k_cache[k_base + dim];
        }
        partial[tid] = dot;
        __syncthreads();
        for (unsigned int stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            const float score = partial[0] * softmax_scale;
            const float new_max_score = fmaxf(max_score, score);
            shared_prev_scale =
                max_score <= -3.4028234663852886e38f ? 0.0f : expf(max_score - new_max_score);
            shared_weight = expf(score - new_max_score);
            shared_max_score = new_max_score;
        }
        __syncthreads();
        const unsigned long long v_index =
            (physical * kv_heads + kv_head) * value_dim + tid;
        if (tid < value_dim) {
            weighted = weighted * shared_prev_scale + shared_weight * v_cache[v_index];
        }
        denominator = denominator * shared_prev_scale + shared_weight;
        max_score = shared_max_score;
        __syncthreads();
    }
    if (tid < value_dim) {
        const unsigned long long output_index = q_head_index * value_dim + tid;
        float result = weighted / denominator;
        if (gate != nullptr) {
            const float gate_value = gate[output_index];
            result *= 1.0f / (1.0f + expf(-gate_value));
        }
        output[output_index] = result;
    }
}

// Typed chunk write/read specializations.  F32 remains the two legacy symbols
// above; non-F32 pairs have compile-time payload paths and FP8 E4M3FN uses an
// independent FP16 scale for every physical-token/KV-head/plane row.
enum {
    ULLM_TYPED_CHUNK_F32 = 0,
    ULLM_TYPED_CHUNK_F16 = 1,
    ULLM_TYPED_CHUNK_FP8_E4M3FN = 2,
};

__device__ __forceinline__ unsigned int ullm_typed_chunk_round_shift_ties_even(
    unsigned int value, unsigned int shift) {
    if (shift == 0u) return value;
    const unsigned int quotient = value >> shift;
    const unsigned int remainder = value & ((1u << shift) - 1u);
    const unsigned int halfway = 1u << (shift - 1u);
    return remainder > halfway || (remainder == halfway && (quotient & 1u) != 0u)
        ? quotient + 1u : quotient;
}

__device__ __forceinline__ unsigned short ullm_typed_chunk_f32_to_f16_rne(float value) {
    const unsigned int raw = __float_as_uint(value);
    const unsigned short sign = static_cast<unsigned short>((raw >> 16u) & 0x8000u);
    const unsigned int exponent_bits = (raw >> 23u) & 0xffu;
    const unsigned int mantissa = raw & 0x7fffffu;
    if (exponent_bits == 0xffu) return static_cast<unsigned short>(sign | (mantissa == 0u ? 0x7c00u : 0x7e00u));
    const int exponent = static_cast<int>(exponent_bits) - 127;
    if (exponent > 15) return static_cast<unsigned short>(sign | 0x7c00u);
    if (exponent >= -14) {
        unsigned int rounded = ullm_typed_chunk_round_shift_ties_even(mantissa, 13u);
        unsigned int half_exponent = static_cast<unsigned int>(exponent + 15);
        if (rounded == 0x400u) {
            rounded = 0u;
            ++half_exponent;
            if (half_exponent >= 0x1fu) return static_cast<unsigned short>(sign | 0x7c00u);
        }
        return static_cast<unsigned short>(sign | (half_exponent << 10u) | rounded);
    }
    if (exponent < -25) return sign;
    return static_cast<unsigned short>(sign | ullm_typed_chunk_round_shift_ties_even(
        mantissa | 0x800000u, static_cast<unsigned int>(-exponent - 1)));
}

__device__ __forceinline__ float ullm_typed_chunk_f16_to_f32(unsigned short bits) {
    const float sign = (bits & 0x8000u) == 0u ? 1.0f : -1.0f;
    const unsigned int exponent = (static_cast<unsigned int>(bits) >> 10u) & 0x1fu;
    const unsigned int mantissa = static_cast<unsigned int>(bits) & 0x03ffu;
    if (exponent == 0u) return mantissa == 0u ? sign * 0.0f : sign * ldexpf(static_cast<float>(mantissa), -24);
    if (exponent == 0x1fu) return mantissa == 0u ? sign * __int_as_float(0x7f800000) : __int_as_float(0x7fc00000);
    return sign * ldexpf(1.0f + static_cast<float>(mantissa) * 0.0009765625f,
                         static_cast<int>(exponent) - 15);
}

__device__ __forceinline__ float ullm_typed_chunk_fp8_to_f32(unsigned char value) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_cvt_f32_fp8(static_cast<unsigned int>(value), 0);
#else
    const unsigned int raw = static_cast<unsigned int>(value);
    const unsigned int sign = raw >> 7u;
    const unsigned int exponent = (raw >> 3u) & 0x0fu;
    const unsigned int mantissa = raw & 0x07u;
    if (exponent == 0x0fu && mantissa == 0x07u) return __uint_as_float(0x7fc00000u);
    const float magnitude = exponent == 0u ? static_cast<float>(mantissa) * 0.001953125f
        : ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f, static_cast<int>(exponent) - 7);
    return sign == 0u ? magnitude : -magnitude;
#endif
}

__device__ __forceinline__ bool ullm_typed_chunk_positive_f16_ceil(
    float value, unsigned short *bits_out) {
    if (bits_out == nullptr || !isfinite(value) || value <= 0.0f) return false;
    unsigned short bits = ullm_typed_chunk_f32_to_f16_rne(value);
    if (bits == 0u) bits = 1u;
    if (bits >= 0x7c00u) return false;
    if (ullm_typed_chunk_f16_to_f32(bits) < value) {
        ++bits;
        if (bits >= 0x7c00u) return false;
    }
    if (!isfinite(ullm_typed_chunk_f16_to_f32(bits)) ||
        ullm_typed_chunk_f16_to_f32(bits) < value) return false;
    *bits_out = bits;
    return true;
}

__device__ __forceinline__ unsigned char ullm_typed_chunk_f32_to_fp8_rne(
    float value, float scale) {
    const unsigned int sign = (__float_as_uint(value) >> 24u) & 0x80u;
    const float magnitude = fabsf(value / scale);
    if (magnitude == 0.0f) return static_cast<unsigned char>(sign);
    if (magnitude >= 448.0f) return static_cast<unsigned char>(sign | 0x7eu);
    unsigned int low = 0u;
    unsigned int high = 126u;
    while (low < high) {
        const unsigned int middle = low + (high - low) / 2u;
        if (fabsf(ullm_typed_chunk_fp8_to_f32(static_cast<unsigned char>(middle))) < magnitude)
            low = middle + 1u;
        else high = middle;
    }
    const unsigned int upper = low;
    const float upper_value = ullm_typed_chunk_fp8_to_f32(static_cast<unsigned char>(upper));
    if (upper_value == magnitude) return static_cast<unsigned char>(sign | upper);
    const unsigned int lower = upper - 1u;
    const double lower_distance = static_cast<double>(magnitude) -
        static_cast<double>(ullm_typed_chunk_fp8_to_f32(static_cast<unsigned char>(lower)));
    const double upper_distance = static_cast<double>(upper_value) - static_cast<double>(magnitude);
    const unsigned int encoded = lower_distance < upper_distance ? lower :
        (upper_distance < lower_distance ? upper : ((lower & 1u) == 0u ? lower : upper));
    return static_cast<unsigned char>(sign | encoded);
}

template <int DTYPE>
__device__ __forceinline__ float ullm_typed_chunk_row_scale(
    const unsigned short *scales, unsigned long long physical_head) {
    if constexpr (DTYPE == ULLM_TYPED_CHUNK_FP8_E4M3FN)
        return ullm_typed_chunk_f16_to_f32(scales[physical_head]);
    return 1.0f;
}

template <int DTYPE>
__device__ __forceinline__ float ullm_typed_chunk_load(
    const void *payload, unsigned long long index, float scale) {
    if constexpr (DTYPE == ULLM_TYPED_CHUNK_F32)
        return static_cast<const float *>(payload)[index];
    else if constexpr (DTYPE == ULLM_TYPED_CHUNK_F16)
        return ullm_typed_chunk_f16_to_f32(static_cast<const unsigned short *>(payload)[index]);
    else
        return ullm_typed_chunk_fp8_to_f32(static_cast<const unsigned char *>(payload)[index]) * scale;
}

template <int DTYPE>
__device__ __forceinline__ void ullm_typed_chunk_store_row(
    const float *source, unsigned long long elements, unsigned long long physical_head,
    unsigned long long row_dim, void *payload, unsigned short *scales, float *partial) {
    const unsigned int tid = threadIdx.x;
    float local_maximum = 0.0f;
    bool finite = true;
    if constexpr (DTYPE != ULLM_TYPED_CHUNK_F32) {
        for (unsigned long long index = tid; index < elements; index += blockDim.x) {
            const float value = source[index];
            finite = finite && isfinite(value);
            local_maximum = fmaxf(local_maximum, fabsf(value));
        }
    }
    partial[tid] = finite ? local_maximum : __int_as_float(0x7f800000);
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }
    unsigned short scale_bits = 0x3c00u;
    bool valid = true;
    if constexpr (DTYPE == ULLM_TYPED_CHUNK_F16) valid = isfinite(partial[0]) && partial[0] <= 65504.0f;
    else if constexpr (DTYPE == ULLM_TYPED_CHUNK_FP8_E4M3FN) {
        valid = isfinite(partial[0]) &&
            (partial[0] == 0.0f || ullm_typed_chunk_positive_f16_ceil(partial[0] / 448.0f, &scale_bits));
    }
    if (tid == 0u) partial[0] = valid ? static_cast<float>(scale_bits) : -1.0f;
    __syncthreads();
    if (partial[0] < 0.0f) return;
    scale_bits = static_cast<unsigned short>(partial[0]);
    if constexpr (DTYPE == ULLM_TYPED_CHUNK_FP8_E4M3FN) if (tid == 0u) scales[physical_head] = scale_bits;
    const float scale = ullm_typed_chunk_f16_to_f32(scale_bits);
    const unsigned long long destination = physical_head * row_dim;
    for (unsigned long long index = tid; index < elements; index += blockDim.x) {
        if constexpr (DTYPE == ULLM_TYPED_CHUNK_F32)
            static_cast<float *>(payload)[destination + index] = source[index];
        else if constexpr (DTYPE == ULLM_TYPED_CHUNK_F16)
            static_cast<unsigned short *>(payload)[destination + index] = ullm_typed_chunk_f32_to_f16_rne(source[index]);
        else
            static_cast<unsigned char *>(payload)[destination + index] = ullm_typed_chunk_f32_to_fp8_rne(source[index], scale);
    }
}

template <int K_DTYPE, int V_DTYPE>
__device__ __forceinline__ void ullm_paged_kv_write_chunk_typed_kernel_impl(
    const float *k, const float *v, const unsigned int *block_table,
    unsigned long long cache_start, unsigned long long m, unsigned long long block_size,
    unsigned long long cache_blocks, unsigned long long table_entries,
    unsigned long long kv_heads, unsigned long long head_dim, unsigned long long value_dim,
    void *k_cache, void *v_cache, unsigned short *k_scales, unsigned short *v_scales,
    float *partial) {
    const unsigned long long row_plane_head = blockIdx.x;
    const unsigned long long heads_per_plane = m * kv_heads;
    const bool is_k = row_plane_head < heads_per_plane;
    const unsigned long long packed = is_k ? row_plane_head : row_plane_head - heads_per_plane;
    const unsigned long long row = packed / kv_heads;
    const unsigned long long kv_head = packed - row * kv_heads;
    if (row >= m || kv_head >= kv_heads) return;
    const unsigned long long logical = cache_start + row;
    const unsigned long long block_index = logical / block_size;
    if (block_index >= table_entries) return;
    const unsigned long long block_id = static_cast<unsigned long long>(block_table[block_index]);
    if (block_id >= cache_blocks) return;
    const unsigned long long physical = block_id * block_size + (logical - block_index * block_size);
    const unsigned long long physical_head = physical * kv_heads + kv_head;
    if (is_k) {
        ullm_typed_chunk_store_row<K_DTYPE>(
            k + (row * kv_heads + kv_head) * head_dim, head_dim, physical_head,
            head_dim, k_cache, k_scales, partial);
    } else {
        ullm_typed_chunk_store_row<V_DTYPE>(
            v + (row * kv_heads + kv_head) * value_dim, value_dim, physical_head,
            value_dim, v_cache, v_scales, partial);
    }
}

template <int K_DTYPE, int V_DTYPE>
__device__ __forceinline__ void ullm_paged_causal_gqa_chunk_typed_kernel_impl(
    const float *q, const float *gate, const void *k_cache, const void *v_cache,
    const unsigned int *block_table, const unsigned short *k_scales,
    const unsigned short *v_scales, unsigned long long cached_prefix_len,
    unsigned long long m, unsigned long long block_size, unsigned long long cache_blocks,
    unsigned long long table_entries, unsigned long long q_heads,
    unsigned long long kv_heads, unsigned long long head_dim,
    unsigned long long value_dim, float softmax_scale, float *output,
    float *partial, float *shared_scales, float *state) {
    const unsigned long long q_head_index = blockIdx.x;
    if (q_head_index >= m * q_heads || q_heads == 0ull || kv_heads == 0ull ||
        blockDim.x != 256u || head_dim > 256ull || value_dim > 256ull) return;
    const unsigned int tid = threadIdx.x;
    const unsigned long long row = q_head_index / q_heads;
    const unsigned long long q_head = q_head_index - row * q_heads;
    const unsigned long long kv_head = q_head / (q_heads / kv_heads);
    const unsigned long long q_base = (row * q_heads + q_head) * head_dim;
    const unsigned long long cache_len = cached_prefix_len + row + 1ull;
    float weighted = 0.0f;
    float denominator = 0.0f;
    float max_score = -3.4028234663852886e38f;
    for (unsigned long long source = 0ull; source < cache_len; ++source) {
        const unsigned long long block_index = source / block_size;
        if (block_index >= table_entries) {
            if (tid < value_dim) output[q_head_index * value_dim + tid] = 0.0f;
            return;
        }
        const unsigned long long block_id = static_cast<unsigned long long>(block_table[block_index]);
        if (block_id >= cache_blocks) {
            if (tid < value_dim) output[q_head_index * value_dim + tid] = 0.0f;
            return;
        }
        const unsigned long long physical = block_id * block_size + (source - block_index * block_size);
        const unsigned long long physical_head = physical * kv_heads + kv_head;
        if (tid == 0u) {
            shared_scales[0] = ullm_typed_chunk_row_scale<K_DTYPE>(k_scales, physical_head);
            shared_scales[1] = ullm_typed_chunk_row_scale<V_DTYPE>(v_scales, physical_head);
        }
        __syncthreads();
        float dot = 0.0f;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x)
            dot += q[q_base + dim] * ullm_typed_chunk_load<K_DTYPE>(
                k_cache, physical_head * head_dim + dim, shared_scales[0]);
        partial[tid] = dot;
        __syncthreads();
        for (unsigned int stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            __syncthreads();
        }
        if (tid == 0u) {
            const float score = partial[0] * softmax_scale;
            const float new_max = fmaxf(max_score, score);
            state[0] = max_score <= -3.0e38f ? 0.0f : expf(max_score - new_max);
            state[1] = expf(score - new_max);
            state[2] = new_max;
        }
        __syncthreads();
        if (tid < value_dim) {
            weighted = weighted * state[0] + state[1] * ullm_typed_chunk_load<V_DTYPE>(
                v_cache, physical_head * value_dim + tid, shared_scales[1]);
        }
        denominator = denominator * state[0] + state[1];
        max_score = state[2];
        __syncthreads();
    }
    if (tid < value_dim) {
        const unsigned long long output_index = q_head_index * value_dim + tid;
        float result = weighted / denominator;
        if (gate != nullptr) result *= 1.0f / (1.0f + expf(-gate[output_index]));
        output[output_index] = result;
    }
}

#define ULLM_DEFINE_TYPED_PAGED_CHUNK(K_DTYPE, V_DTYPE, K_NAME, V_NAME) \
extern "C" __global__ void ullm_paged_kv_write_chunk_typed_k##K_NAME##_v##V_NAME##_kernel( \
    const float *k, const float *v, const unsigned int *block_table, \
    unsigned long long cache_start, unsigned long long m, unsigned long long block_size, \
    unsigned long long cache_blocks, unsigned long long table_entries, \
    unsigned long long kv_heads, unsigned long long head_dim, unsigned long long value_dim, \
    void *k_cache, void *v_cache, unsigned short *k_scales, unsigned short *v_scales) { \
    __shared__ float partial[256]; \
    ullm_paged_kv_write_chunk_typed_kernel_impl<K_DTYPE, V_DTYPE>( \
        k, v, block_table, cache_start, m, block_size, cache_blocks, table_entries, kv_heads, \
        head_dim, value_dim, k_cache, v_cache, k_scales, v_scales, partial); \
} \
extern "C" __global__ void ullm_paged_causal_gqa_chunk_typed_k##K_NAME##_v##V_NAME##_kernel( \
    const float *q, const float *gate, const void *k_cache, const void *v_cache, \
    const unsigned int *block_table, const unsigned short *k_scales, const unsigned short *v_scales, \
    unsigned long long cached_prefix_len, unsigned long long m, unsigned long long block_size, \
    unsigned long long cache_blocks, unsigned long long table_entries, unsigned long long q_heads, \
    unsigned long long kv_heads, unsigned long long head_dim, unsigned long long value_dim, \
    float softmax_scale, float *output) { \
    __shared__ float partial[256]; \
    __shared__ float shared_scales[2]; \
    __shared__ float state[3]; \
    ullm_paged_causal_gqa_chunk_typed_kernel_impl<K_DTYPE, V_DTYPE>( \
        q, gate, k_cache, v_cache, block_table, k_scales, v_scales, cached_prefix_len, m, \
        block_size, cache_blocks, table_entries, q_heads, kv_heads, head_dim, value_dim, \
        softmax_scale, output, partial, shared_scales, state); \
}

ULLM_DEFINE_TYPED_PAGED_CHUNK(ULLM_TYPED_CHUNK_F32, ULLM_TYPED_CHUNK_F16, f32, f16)
ULLM_DEFINE_TYPED_PAGED_CHUNK(ULLM_TYPED_CHUNK_F32, ULLM_TYPED_CHUNK_FP8_E4M3FN, f32, fp8)
ULLM_DEFINE_TYPED_PAGED_CHUNK(ULLM_TYPED_CHUNK_F16, ULLM_TYPED_CHUNK_F32, f16, f32)
ULLM_DEFINE_TYPED_PAGED_CHUNK(ULLM_TYPED_CHUNK_F16, ULLM_TYPED_CHUNK_F16, f16, f16)
ULLM_DEFINE_TYPED_PAGED_CHUNK(ULLM_TYPED_CHUNK_F16, ULLM_TYPED_CHUNK_FP8_E4M3FN, f16, fp8)
ULLM_DEFINE_TYPED_PAGED_CHUNK(ULLM_TYPED_CHUNK_FP8_E4M3FN, ULLM_TYPED_CHUNK_F32, fp8, f32)
ULLM_DEFINE_TYPED_PAGED_CHUNK(ULLM_TYPED_CHUNK_FP8_E4M3FN, ULLM_TYPED_CHUNK_F16, fp8, f16)
ULLM_DEFINE_TYPED_PAGED_CHUNK(ULLM_TYPED_CHUNK_FP8_E4M3FN, ULLM_TYPED_CHUNK_FP8_E4M3FN, fp8, fp8)
#undef ULLM_DEFINE_TYPED_PAGED_CHUNK
)";
    }

    // Production gfx1201 Qwen3.5 cold-prefill WMMA-QK path. It intentionally lives in a
    // separate rocWMMA HIPRTC module from paged_chunk_kernel_source(), so adding rocWMMA
    // headers/options cannot perturb the scalar paged reader's compilation or dispatch.
    static const char *paged_causal_gqa_chunk_wmma_kernel_source() {
        return R"(
#include <rocwmma/rocwmma.hpp>

extern "C" __global__ void ullm_paged_causal_gqa_chunk_wmma_f32_kernel(
    const float *q,
    const float *gate,
    const float *k_cache,
    const float *v_cache,
    const unsigned int *block_table,
    unsigned long long cached_prefix_len,
    unsigned long long m,
    unsigned long long block_size,
    unsigned long long cache_blocks,
    unsigned long long table_entries,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float *output) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    using namespace rocwmma;
    constexpr unsigned int kQueryTile = 16u;
    constexpr unsigned int kSourceTile = 16u;
    constexpr unsigned int kWmmaK = 16u;
    constexpr unsigned int kQHeads = 16u;
    constexpr unsigned int kKvHeads = 4u;
    constexpr unsigned int kGqaRatio = kQHeads / kKvHeads;
    constexpr unsigned int kHeadDim = 256u;
    constexpr unsigned int kValueDim = 256u;
    constexpr unsigned int kPageTokens = 256u;
    constexpr unsigned int kThreads = 256u;
    constexpr unsigned int kValuesPerThread =
        (kQueryTile * kValueDim) / kThreads;
    static_assert(kQueryTile == kSourceTile && kQueryTile == kWmmaK,
                  "rocWMMA QK tile must be 16x16x16");
    static_assert(kThreads % 32u == 0u, "production kernel requires whole wave32 groups");
    static_assert(kValuesPerThread == 16u,
                  "production kernel assigns one 16-value vector to each thread");

    using FragA = fragment<matrix_a, kQueryTile, kSourceTile, kWmmaK, float16_t, row_major>;
    using FragB = fragment<matrix_b, kQueryTile, kSourceTile, kWmmaK, float16_t, col_major>;
    using FragC =
        fragment<accumulator, kQueryTile, kSourceTile, kWmmaK, float32_t, row_major>;

    constexpr unsigned int kQTileLdsBytes =
        kQueryTile * kHeadDim * sizeof(float16_t);
    constexpr unsigned int kKTileLdsBytes =
        kSourceTile * kHeadDim * sizeof(float16_t);
    constexpr unsigned int kVTileLdsBytes = kSourceTile * kValueDim * sizeof(float);
    constexpr unsigned int kScoresLdsBytes = kQueryTile * kSourceTile * sizeof(float);
    constexpr unsigned int kRowStateLdsBytes = 3u * kQueryTile * sizeof(float);
    constexpr unsigned int kStaticLdsBytes =
        kQTileLdsBytes + kKTileLdsBytes + kVTileLdsBytes + kScoresLdsBytes +
        kRowStateLdsBytes;
    static_assert(kStaticLdsBytes == 33984u, "production LDS accounting drifted");
    static_assert(kStaticLdsBytes <= 64u * 1024u,
                  "production kernel must fit the gfx1201 64 KiB LDS budget");

    const unsigned int tid = threadIdx.x;
    const unsigned long long q_head = static_cast<unsigned long long>(blockIdx.y);
    const unsigned long long row_base =
        static_cast<unsigned long long>(blockIdx.x) * kQueryTile;
    if (q_head >= q_heads || row_base >= m || q_heads != kQHeads || kv_heads != kKvHeads ||
        head_dim != kHeadDim || value_dim != kValueDim || block_size != kPageTokens ||
        cache_blocks == 0ull || table_entries == 0ull || blockDim.x != kThreads) {
        return;
    }

    // This is deliberately one Q head per CTA: q_head / 4 implements Qwen3.5's GQA=4 mapping.
    const unsigned long long kv_head = q_head / kGqaRatio;
    const unsigned long long remaining_rows = m - row_base;
    const unsigned int rows_in_tile = remaining_rows < kQueryTile ?
        static_cast<unsigned int>(remaining_rows) : kQueryTile;
    // The longest causal row in this CTA is its final valid query row, not merely the tile width.
    const unsigned long long max_cache_len = cached_prefix_len + row_base + rows_in_tile;
    // A wave spans two 16-value groups across all 16 rows. At each AV inner-loop step, the
    // 16 lanes for a value group therefore broadcast the same V element from LDS instead of
    // striding through a pair of shared-memory banks.
    const unsigned int output_row = tid % kQueryTile;
    const unsigned int value_base = (tid / kQueryTile) * kValuesPerThread;

    __shared__ __attribute__((aligned(16))) float16_t q_tile[kQueryTile * kHeadDim];
    __shared__ __attribute__((aligned(16))) float16_t k_tile[kSourceTile * kHeadDim];
    __shared__ __attribute__((aligned(16))) float v_tile[kSourceTile * kValueDim];
    // Holds raw QK scores until the row leaders perform online softmax, then holds weights.
    __shared__ float scores[kQueryTile * kSourceTile];
    __shared__ float row_max[kQueryTile];
    __shared__ float row_sum[kQueryTile];
    __shared__ float row_old_scale[kQueryTile];

    // Keeping the 16 output values per thread in registers saves 16 KiB of LDS versus a shared
    // accumulator. AV stays FP32/scalar in v1; see the source comment below for the WMMA follow-up.
    float output_acc[kValuesPerThread] = {};

    for (unsigned int index = tid; index < kQueryTile * kHeadDim; index += kThreads) {
        const unsigned int row_in_tile = index / kHeadDim;
        const unsigned int dim = index - row_in_tile * kHeadDim;
        const unsigned long long row = row_base + row_in_tile;
        q_tile[index] = row < m ?
            static_cast<float16_t>(q[(row * q_heads + q_head) * kHeadDim + dim]) :
            static_cast<float16_t>(0.0f);
    }
    if (tid < kQueryTile) {
        row_max[tid] = -3.4028234663852886e38f;
        row_sum[tid] = 0.0f;
        row_old_scale[tid] = 0.0f;
    }
    __syncthreads();

    // Each source tile begins at a multiple of 16. With the production path's fixed 256-token page,
    // no tile crosses a page boundary, so one block-table lookup indexes all staged K/V values.
    for (unsigned long long source_tile_start = 0ull;
         source_tile_start < max_cache_len;
         source_tile_start += kSourceTile) {
        const unsigned long long logical_block = source_tile_start / kPageTokens;
        const unsigned long long physical_tile_base =
            static_cast<unsigned long long>(block_table[logical_block]) * kPageTokens +
            (source_tile_start & (kPageTokens - 1u));

        for (unsigned int index = tid; index < kSourceTile * kHeadDim; index += kThreads) {
            const unsigned int source_in_tile = index / kHeadDim;
            const unsigned int dim = index - source_in_tile * kHeadDim;
            const unsigned long long source = source_tile_start + source_in_tile;
            if (source < max_cache_len) {
                const unsigned long long physical = physical_tile_base + source_in_tile;
                k_tile[index] = static_cast<float16_t>(
                    k_cache[(physical * kKvHeads + kv_head) * kHeadDim + dim]);
                v_tile[source_in_tile * kValueDim + dim] =
                    v_cache[(physical * kKvHeads + kv_head) * kValueDim + dim];
            } else {
                k_tile[index] = static_cast<float16_t>(0.0f);
                v_tile[source_in_tile * kValueDim + dim] = 0.0f;
            }
        }
        __syncthreads();

        // Wave 0 computes Q[16,256] * K[16,256]^T. q_tile is row-major [M,K], while the
        // source-major k_tile is exactly the column-major [K,N] storage requested by FragB.
        if (tid < 32u) {
            FragA a;
            FragB b;
            FragC c;
            fill_fragment(c, 0.0f);
#pragma unroll
            for (unsigned int dim_base = 0u; dim_base < kHeadDim; dim_base += kWmmaK) {
                load_matrix_sync(a, q_tile + dim_base, kHeadDim);
                load_matrix_sync(b, k_tile + dim_base, kHeadDim);
                mma_sync(c, a, b, c);
            }
            store_matrix_sync(scores, c, kSourceTile);
        }
        __syncthreads();

        // FlashAttention online state is per query row. Causal positions are represented as
        // negative infinity / zero weight before the max/sum update, including tail padding.
        if (tid < kQueryTile) {
            const unsigned int row_in_tile = tid;
            if (row_in_tile < rows_in_tile) {
                const unsigned long long causal_limit =
                    cached_prefix_len + row_base + row_in_tile + 1ull;
                float tile_max = -3.4028234663852886e38f;
#pragma unroll
                for (unsigned int source_in_tile = 0u;
                     source_in_tile < kSourceTile;
                     ++source_in_tile) {
                    const unsigned long long source = source_tile_start + source_in_tile;
                    const float score = source < causal_limit ?
                        scores[row_in_tile * kSourceTile + source_in_tile] * softmax_scale :
                        -3.4028234663852886e38f;
                    tile_max = score > tile_max ? score : tile_max;
                }
                const float old_max = row_max[row_in_tile];
                const float new_max = old_max > tile_max ? old_max : tile_max;
                const float old_scale = old_max <= -3.0e38f ?
                    0.0f : expf(old_max - new_max);
                float tile_sum = 0.0f;
#pragma unroll
                for (unsigned int source_in_tile = 0u;
                     source_in_tile < kSourceTile;
                     ++source_in_tile) {
                    const unsigned long long source = source_tile_start + source_in_tile;
                    const float weight = source < causal_limit ?
                        expf(scores[row_in_tile * kSourceTile + source_in_tile] *
                                 softmax_scale - new_max) :
                        0.0f;
                    scores[row_in_tile * kSourceTile + source_in_tile] = weight;
                    tile_sum += weight;
                }
                row_old_scale[row_in_tile] = old_scale;
                row_sum[row_in_tile] = row_sum[row_in_tile] * old_scale + tile_sum;
                row_max[row_in_tile] = new_max;
            } else {
                row_old_scale[row_in_tile] = 0.0f;
#pragma unroll
                for (unsigned int source_in_tile = 0u;
                     source_in_tile < kSourceTile;
                     ++source_in_tile) {
                    scores[row_in_tile * kSourceTile + source_in_tile] = 0.0f;
                }
            }
        }
        __syncthreads();

        const float old_scale = row_old_scale[output_row];
#pragma unroll
        for (unsigned int value = 0u; value < kValuesPerThread; ++value) {
            output_acc[value] *= old_scale;
        }
#pragma unroll
        for (unsigned int source_in_tile = 0u;
             source_in_tile < kSourceTile;
             ++source_in_tile) {
            const float weight = scores[output_row * kSourceTile + source_in_tile];
#pragma unroll
            for (unsigned int value = 0u; value < kValuesPerThread; ++value) {
                output_acc[value] += weight *
                    v_tile[source_in_tile * kValueDim + value_base + value];
            }
        }
        // AV intentionally remains scalar FP32 for v1. A WMMA AV follow-up needs an explicit
        // softmax-weight FP16 conversion and a safe [K,N] V transpose/layout, which would add a
        // second precision change and is not appropriate before this QK-only path is validated.
        __syncthreads();
    }

    if (output_row < rows_in_tile) {
        const unsigned long long row = row_base + output_row;
        const unsigned long long output_base =
            (row * q_heads + q_head) * kValueDim + value_base;
        const float denominator = row_sum[output_row];
#pragma unroll
        for (unsigned int value = 0u; value < kValuesPerThread; ++value) {
            float result = output_acc[value] / denominator;
            if (gate != nullptr) {
                const float gate_value = gate[output_base + value];
                result *= 1.0f / (1.0f + expf(-gate_value));
            }
            output[output_base + value] = result;
        }
    }
#else
    (void)q;
    (void)gate;
    (void)k_cache;
    (void)v_cache;
    (void)block_table;
    (void)cached_prefix_len;
    (void)m;
    (void)block_size;
    (void)cache_blocks;
    (void)table_entries;
    (void)q_heads;
    (void)kv_heads;
    (void)head_dim;
    (void)value_dim;
    (void)softmax_scale;
    (void)output;
#endif
}
)";
    }

