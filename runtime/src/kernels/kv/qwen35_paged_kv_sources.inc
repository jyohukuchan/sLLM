// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
    static const char *qwen35_qk_norm_rope_paged_kv_write_kernel_source() {
        return R"(
extern "C" __global__ void ullm_qwen35_qk_norm_rope_paged_kv_write_f32_kernel(
    const float *q_projected,
    const float *k_projected,
    const float *v_projected,
    const float *q_weight,
    const float *k_weight,
    const unsigned int *block_table,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    unsigned long long rotary_dim,
    unsigned long long position_offset,
    float rope_base,
    float epsilon,
    unsigned long long cache_position,
    unsigned long long block_size,
    unsigned long long cache_blocks,
    float *q_gate_output,
    float *q_rope_output,
    float *k_cache,
    float *v_cache) {
    const unsigned long long segment = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned long long k_segment_start = q_heads;
    const unsigned long long v_segment_start = q_heads + kv_heads;
    const unsigned long long total_segments = q_heads + kv_heads + kv_heads;
    if (segment >= total_segments) {
        return;
    }
    const unsigned long long block_index = cache_position / block_size;
    const unsigned long long block_offset = cache_position - block_index * block_size;
    const unsigned long long block_id = static_cast<unsigned long long>(block_table[block_index]);
    if (block_id >= cache_blocks) {
        return;
    }
    const unsigned long long physical_timestep = block_id * block_size + block_offset;
    if (segment >= v_segment_start) {
        const unsigned long long kv_head = segment - v_segment_start;
        const unsigned long long source_base = kv_head * value_dim;
        const unsigned long long cache_base =
            (physical_timestep * kv_heads + kv_head) * value_dim;
        for (unsigned long long dim = tid; dim < value_dim; dim += blockDim.x) {
            v_cache[cache_base + dim] = v_projected[source_base + dim];
        }
        return;
    }

    __shared__ float partial[256];
    float sum_squares = 0.0f;
    if (segment < q_heads) {
        const unsigned long long source_base = segment * 2ull * head_dim;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            const float value = q_projected[source_base + dim];
            q_gate_output[segment * head_dim + dim] = q_projected[source_base + head_dim + dim];
            sum_squares += value * value;
        }
    } else {
        const unsigned long long kv_head = segment - k_segment_start;
        const unsigned long long source_base = kv_head * head_dim;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            const float value = k_projected[source_base + dim];
            sum_squares += value * value;
        }
    }
    partial[tid] = sum_squares;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + epsilon);
    const unsigned long long half = rotary_dim >> 1;
    const float position = static_cast<float>(position_offset);
    if (segment < q_heads) {
        const unsigned long long source_base = segment * 2ull * head_dim;
        const unsigned long long output_base = segment * head_dim;
        for (unsigned long long pair_dim = tid; pair_dim < half; pair_dim += blockDim.x) {
            const float exponent =
                (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
            const float theta = position / powf(rope_base, exponent);
            const float c = cosf(theta);
            const float s = sinf(theta);
            const float first = q_projected[source_base + pair_dim] * inv_rms * q_weight[pair_dim];
            const unsigned long long second_dim = half + pair_dim;
            const float second =
                q_projected[source_base + second_dim] * inv_rms * q_weight[second_dim];
            q_rope_output[output_base + pair_dim] = first * c - second * s;
            q_rope_output[output_base + second_dim] = second * c + first * s;
        }
        for (unsigned long long dim = rotary_dim + tid; dim < head_dim; dim += blockDim.x) {
            q_rope_output[output_base + dim] =
                q_projected[source_base + dim] * inv_rms * q_weight[dim];
        }
    } else {
        const unsigned long long kv_head = segment - k_segment_start;
        const unsigned long long source_base = kv_head * head_dim;
        const unsigned long long cache_base =
            (physical_timestep * kv_heads + kv_head) * head_dim;
        for (unsigned long long pair_dim = tid; pair_dim < half; pair_dim += blockDim.x) {
            const float exponent =
                (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
            const float theta = position / powf(rope_base, exponent);
            const float c = cosf(theta);
            const float s = sinf(theta);
            const float first = k_projected[source_base + pair_dim] * inv_rms * k_weight[pair_dim];
            const unsigned long long second_dim = half + pair_dim;
            const float second =
                k_projected[source_base + second_dim] * inv_rms * k_weight[second_dim];
            k_cache[cache_base + pair_dim] = first * c - second * s;
            k_cache[cache_base + second_dim] = second * c + first * s;
        }
        for (unsigned long long dim = rotary_dim + tid; dim < head_dim; dim += blockDim.x) {
            k_cache[cache_base + dim] =
                k_projected[source_base + dim] * inv_rms * k_weight[dim];
        }
    }
}

// Typed Qwen3.5 writer specializations.  The F32 symbol above is deliberately
// retained verbatim; typed cache selections enter only one of the eight
// wrappers below.  FP8 scales are independent per physical token, KV head,
// and K/V plane.
enum {
    ULLM_FUSED_TYPED_KV_F32 = 0,
    ULLM_FUSED_TYPED_KV_F16 = 1,
    ULLM_FUSED_TYPED_KV_FP8_E4M3FN = 2,
};

__device__ __forceinline__ unsigned int ullm_fused_typed_kv_round_shift_ties_even(
    unsigned int value, unsigned int shift) {
    if (shift == 0u) return value;
    const unsigned int quotient = value >> shift;
    const unsigned int remainder = value & ((1u << shift) - 1u);
    const unsigned int halfway = 1u << (shift - 1u);
    return remainder > halfway || (remainder == halfway && (quotient & 1u) != 0u)
        ? quotient + 1u : quotient;
}

__device__ __forceinline__ unsigned short ullm_fused_typed_kv_f32_to_f16_rne(float value) {
    const unsigned int raw = __float_as_uint(value);
    const unsigned short sign = static_cast<unsigned short>((raw >> 16u) & 0x8000u);
    const unsigned int exponent_bits = (raw >> 23u) & 0xffu;
    const unsigned int mantissa = raw & 0x7fffffu;
    if (exponent_bits == 0xffu) {
        return static_cast<unsigned short>(sign | (mantissa == 0u ? 0x7c00u : 0x7e00u));
    }
    const int exponent = static_cast<int>(exponent_bits) - 127;
    if (exponent > 15) return static_cast<unsigned short>(sign | 0x7c00u);
    if (exponent >= -14) {
        unsigned int rounded = ullm_fused_typed_kv_round_shift_ties_even(mantissa, 13u);
        unsigned int half_exponent = static_cast<unsigned int>(exponent + 15);
        if (rounded == 0x400u) {
            rounded = 0u;
            ++half_exponent;
            if (half_exponent >= 0x1fu) return static_cast<unsigned short>(sign | 0x7c00u);
        }
        return static_cast<unsigned short>(sign | (half_exponent << 10u) | rounded);
    }
    if (exponent < -25) return sign;
    return static_cast<unsigned short>(sign | ullm_fused_typed_kv_round_shift_ties_even(
        mantissa | 0x800000u, static_cast<unsigned int>(-exponent - 1)));
}

__device__ __forceinline__ float ullm_fused_typed_kv_f16_to_f32(unsigned short bits) {
    const float sign = (bits & 0x8000u) == 0u ? 1.0f : -1.0f;
    const unsigned int exponent = (static_cast<unsigned int>(bits) >> 10u) & 0x1fu;
    const unsigned int mantissa = static_cast<unsigned int>(bits) & 0x03ffu;
    if (exponent == 0u) return mantissa == 0u ? sign * 0.0f : sign * ldexpf(static_cast<float>(mantissa), -24);
    if (exponent == 0x1fu) return mantissa == 0u ? sign * __int_as_float(0x7f800000) : __int_as_float(0x7fc00000);
    return sign * ldexpf(1.0f + static_cast<float>(mantissa) * 0.0009765625f,
                         static_cast<int>(exponent) - 15);
}

__device__ __forceinline__ bool ullm_fused_typed_kv_positive_f16_ceil(
    float value, unsigned short *bits_out) {
    if (bits_out == nullptr || !isfinite(value) || value <= 0.0f) return false;
    unsigned short bits = ullm_fused_typed_kv_f32_to_f16_rne(value);
    if (bits == 0u) bits = 1u;
    if (bits >= 0x7c00u) return false;
    if (ullm_fused_typed_kv_f16_to_f32(bits) < value) {
        ++bits;
        if (bits >= 0x7c00u) return false;
    }
    const float stored = ullm_fused_typed_kv_f16_to_f32(bits);
    if (!isfinite(stored) || stored <= 0.0f || stored < value) return false;
    *bits_out = bits;
    return true;
}

__device__ __forceinline__ float ullm_fused_typed_kv_fp8_to_f32(unsigned char value) {
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

__device__ __forceinline__ unsigned char ullm_fused_typed_kv_f32_to_fp8_rne(
    float value, float scale) {
    const unsigned int sign = (__float_as_uint(value) >> 24u) & 0x80u;
    const float magnitude = fabsf(value / scale);
    if (magnitude == 0.0f) return static_cast<unsigned char>(sign);
    if (magnitude >= 448.0f) return static_cast<unsigned char>(sign | 0x7eu);
    unsigned int low = 0u;
    unsigned int high = 126u;
    while (low < high) {
        const unsigned int middle = low + (high - low) / 2u;
        if (fabsf(ullm_fused_typed_kv_fp8_to_f32(static_cast<unsigned char>(middle))) < magnitude)
            low = middle + 1u;
        else high = middle;
    }
    const unsigned int upper = low;
    const float upper_value = ullm_fused_typed_kv_fp8_to_f32(static_cast<unsigned char>(upper));
    if (upper_value == magnitude) return static_cast<unsigned char>(sign | upper);
    const unsigned int lower = upper - 1u;
    const double low_distance = static_cast<double>(magnitude) -
        static_cast<double>(ullm_fused_typed_kv_fp8_to_f32(static_cast<unsigned char>(lower)));
    const double high_distance = static_cast<double>(upper_value) - static_cast<double>(magnitude);
    const unsigned int encoded = low_distance < high_distance ? lower
        : (high_distance < low_distance ? upper : ((lower & 1u) == 0u ? lower : upper));
    return static_cast<unsigned char>(sign | encoded);
}

template <int DTYPE>
__device__ __forceinline__ void ullm_fused_typed_kv_store_row(
    const float *source, unsigned long long elements, unsigned long long physical_head,
    unsigned long long row_dim, void *payload, unsigned short *scales, float *partial) {
    const unsigned int tid = threadIdx.x;
    float local_maximum = 0.0f;
    bool finite = true;
    if constexpr (DTYPE != ULLM_FUSED_TYPED_KV_F32) {
        for (unsigned long long index = tid; index < elements; index += blockDim.x) {
            const float value = source[index];
            finite = finite && isfinite(value);
            local_maximum = fmaxf(local_maximum, fabsf(value));
        }
    }
    partial[tid] = finite ? local_maximum : __int_as_float(0x7f800000);
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1u; offset > 0u; offset >>= 1u) {
        if (tid < offset) partial[tid] = fmaxf(partial[tid], partial[tid + offset]);
        __syncthreads();
    }
    unsigned short scale_bits = 0x3c00u;
    bool valid = true;
    if constexpr (DTYPE == ULLM_FUSED_TYPED_KV_F16) {
        valid = isfinite(partial[0]) && partial[0] <= 65504.0f;
    } else if constexpr (DTYPE == ULLM_FUSED_TYPED_KV_FP8_E4M3FN) {
        if (!isfinite(partial[0])) valid = false;
        else if (partial[0] != 0.0f)
            valid = ullm_fused_typed_kv_positive_f16_ceil(partial[0] / 448.0f, &scale_bits);
    }
    if (tid == 0u) partial[0] = valid ? static_cast<float>(scale_bits) : -1.0f;
    __syncthreads();
    if (partial[0] < 0.0f) return;
    scale_bits = static_cast<unsigned short>(partial[0]);
    if constexpr (DTYPE == ULLM_FUSED_TYPED_KV_FP8_E4M3FN) {
        if (tid == 0u) scales[physical_head] = scale_bits;
    }
    const float scale = ullm_fused_typed_kv_f16_to_f32(scale_bits);
    const unsigned long long base = physical_head * row_dim;
    for (unsigned long long index = tid; index < elements; index += blockDim.x) {
        if constexpr (DTYPE == ULLM_FUSED_TYPED_KV_F32)
            static_cast<float *>(payload)[base + index] = source[index];
        else if constexpr (DTYPE == ULLM_FUSED_TYPED_KV_F16)
            static_cast<unsigned short *>(payload)[base + index] = ullm_fused_typed_kv_f32_to_f16_rne(source[index]);
        else
            static_cast<unsigned char *>(payload)[base + index] = ullm_fused_typed_kv_f32_to_fp8_rne(source[index], scale);
    }
}

template <int DTYPE>
__device__ __forceinline__ void ullm_fused_typed_kv_store_rope_k_row(
    const float *k_projected, const float *k_weight, unsigned long long source_base,
    unsigned long long head_dim, unsigned long long rotary_dim, float position, float rope_base,
    float inv_rms, unsigned long long physical_head, void *payload, unsigned short *scales,
    float *partial) {
    const unsigned int tid = threadIdx.x;
    const unsigned long long half = rotary_dim >> 1u;
    float local_maximum = 0.0f;
    bool finite = true;
    if constexpr (DTYPE != ULLM_FUSED_TYPED_KV_F32) {
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            float value;
            if (dim < half) {
                const float exponent = 2.0f * static_cast<float>(dim) / static_cast<float>(rotary_dim);
                const float theta = position / powf(rope_base, exponent);
                const float first = k_projected[source_base + dim] * inv_rms * k_weight[dim];
                const float second = k_projected[source_base + half + dim] * inv_rms * k_weight[half + dim];
                value = first * cosf(theta) - second * sinf(theta);
            } else if (dim < rotary_dim) {
                const unsigned long long pair = dim - half;
                const float exponent = 2.0f * static_cast<float>(pair) / static_cast<float>(rotary_dim);
                const float theta = position / powf(rope_base, exponent);
                const float first = k_projected[source_base + pair] * inv_rms * k_weight[pair];
                const float second = k_projected[source_base + dim] * inv_rms * k_weight[dim];
                value = second * cosf(theta) + first * sinf(theta);
            } else {
                value = k_projected[source_base + dim] * inv_rms * k_weight[dim];
            }
            finite = finite && isfinite(value);
            local_maximum = fmaxf(local_maximum, fabsf(value));
        }
    }
    partial[tid] = finite ? local_maximum : __int_as_float(0x7f800000);
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1u; offset > 0u; offset >>= 1u) {
        if (tid < offset) partial[tid] = fmaxf(partial[tid], partial[tid + offset]);
        __syncthreads();
    }
    unsigned short scale_bits = 0x3c00u;
    bool valid = true;
    if constexpr (DTYPE == ULLM_FUSED_TYPED_KV_F16) valid = isfinite(partial[0]) && partial[0] <= 65504.0f;
    else if constexpr (DTYPE == ULLM_FUSED_TYPED_KV_FP8_E4M3FN) {
        if (!isfinite(partial[0])) valid = false;
        else if (partial[0] != 0.0f)
            valid = ullm_fused_typed_kv_positive_f16_ceil(partial[0] / 448.0f, &scale_bits);
    }
    if (tid == 0u) partial[0] = valid ? static_cast<float>(scale_bits) : -1.0f;
    __syncthreads();
    if (partial[0] < 0.0f) return;
    scale_bits = static_cast<unsigned short>(partial[0]);
    if constexpr (DTYPE == ULLM_FUSED_TYPED_KV_FP8_E4M3FN) if (tid == 0u) scales[physical_head] = scale_bits;
    const float scale = ullm_fused_typed_kv_f16_to_f32(scale_bits);
    const unsigned long long base = physical_head * head_dim;
    for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
        float value;
        if (dim < half) {
            const float exponent = 2.0f * static_cast<float>(dim) / static_cast<float>(rotary_dim);
            const float theta = position / powf(rope_base, exponent);
            const float first = k_projected[source_base + dim] * inv_rms * k_weight[dim];
            const float second = k_projected[source_base + half + dim] * inv_rms * k_weight[half + dim];
            value = first * cosf(theta) - second * sinf(theta);
        } else if (dim < rotary_dim) {
            const unsigned long long pair = dim - half;
            const float exponent = 2.0f * static_cast<float>(pair) / static_cast<float>(rotary_dim);
            const float theta = position / powf(rope_base, exponent);
            const float first = k_projected[source_base + pair] * inv_rms * k_weight[pair];
            const float second = k_projected[source_base + dim] * inv_rms * k_weight[dim];
            value = second * cosf(theta) + first * sinf(theta);
        } else value = k_projected[source_base + dim] * inv_rms * k_weight[dim];
        if constexpr (DTYPE == ULLM_FUSED_TYPED_KV_F32) static_cast<float *>(payload)[base + dim] = value;
        else if constexpr (DTYPE == ULLM_FUSED_TYPED_KV_F16) static_cast<unsigned short *>(payload)[base + dim] = ullm_fused_typed_kv_f32_to_f16_rne(value);
        else static_cast<unsigned char *>(payload)[base + dim] = ullm_fused_typed_kv_f32_to_fp8_rne(value, scale);
    }
}

template <int K_DTYPE, int V_DTYPE>
__device__ __forceinline__ void ullm_qwen35_qk_norm_rope_paged_kv_write_typed_impl(
    const float *q_projected, const float *k_projected, const float *v_projected,
    const float *q_weight, const float *k_weight, const unsigned int *block_table,
    unsigned long long q_heads, unsigned long long kv_heads, unsigned long long head_dim,
    unsigned long long value_dim, unsigned long long rotary_dim, unsigned long long position_offset,
    float rope_base, float epsilon, unsigned long long cache_position, unsigned long long block_size,
    unsigned long long cache_blocks, float *q_gate_output, float *q_rope_output,
    void *k_cache, void *v_cache, unsigned short *k_scales, unsigned short *v_scales,
    float *partial) {
    const unsigned long long segment = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned long long k_segment_start = q_heads;
    const unsigned long long v_segment_start = q_heads + kv_heads;
    if (segment >= q_heads + kv_heads + kv_heads) return;
    const unsigned long long block_index = cache_position / block_size;
    const unsigned long long block_offset = cache_position - block_index * block_size;
    const unsigned long long block_id = static_cast<unsigned long long>(block_table[block_index]);
    if (block_id >= cache_blocks) return;
    const unsigned long long physical = block_id * block_size + block_offset;
    if (segment >= v_segment_start) {
        const unsigned long long kv_head = segment - v_segment_start;
        ullm_fused_typed_kv_store_row<V_DTYPE>(v_projected + kv_head * value_dim, value_dim,
            physical * kv_heads + kv_head, value_dim, v_cache, v_scales, partial);
        return;
    }
    float sum_squares = 0.0f;
    if (segment < q_heads) {
        const unsigned long long source_base = segment * 2ull * head_dim;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            const float value = q_projected[source_base + dim];
            q_gate_output[segment * head_dim + dim] = q_projected[source_base + head_dim + dim];
            sum_squares += value * value;
        }
    } else {
        const unsigned long long kv_head = segment - k_segment_start;
        const unsigned long long source_base = kv_head * head_dim;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            const float value = k_projected[source_base + dim];
            sum_squares += value * value;
        }
    }
    partial[tid] = sum_squares;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1u; offset > 0u; offset >>= 1u) {
        if (tid < offset) partial[tid] += partial[tid + offset];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + epsilon);
    const unsigned long long half = rotary_dim >> 1u;
    const float position = static_cast<float>(position_offset);
    if (segment < q_heads) {
        const unsigned long long source_base = segment * 2ull * head_dim;
        const unsigned long long output_base = segment * head_dim;
        for (unsigned long long pair_dim = tid; pair_dim < half; pair_dim += blockDim.x) {
            const float exponent = 2.0f * static_cast<float>(pair_dim) / static_cast<float>(rotary_dim);
            const float theta = position / powf(rope_base, exponent);
            const float first = q_projected[source_base + pair_dim] * inv_rms * q_weight[pair_dim];
            const unsigned long long second_dim = half + pair_dim;
            const float second = q_projected[source_base + second_dim] * inv_rms * q_weight[second_dim];
            q_rope_output[output_base + pair_dim] = first * cosf(theta) - second * sinf(theta);
            q_rope_output[output_base + second_dim] = second * cosf(theta) + first * sinf(theta);
        }
        for (unsigned long long dim = rotary_dim + tid; dim < head_dim; dim += blockDim.x)
            q_rope_output[output_base + dim] = q_projected[source_base + dim] * inv_rms * q_weight[dim];
    } else {
        const unsigned long long kv_head = segment - k_segment_start;
        ullm_fused_typed_kv_store_rope_k_row<K_DTYPE>(
            k_projected, k_weight, kv_head * head_dim, head_dim, rotary_dim, position, rope_base,
            inv_rms, physical * kv_heads + kv_head, k_cache, k_scales, partial);
    }
}

#define ULLM_DEFINE_TYPED_QWEN35_PAGED_KV_WRITE(K_DTYPE, V_DTYPE, K_NAME, V_NAME) \
extern "C" __global__ void ullm_qwen35_qk_norm_rope_paged_kv_write_typed_k##K_NAME##_v##V_NAME##_kernel( \
    const float *q_projected, const float *k_projected, const float *v_projected, \
    const float *q_weight, const float *k_weight, const unsigned int *block_table, \
    unsigned long long q_heads, unsigned long long kv_heads, unsigned long long head_dim, \
    unsigned long long value_dim, unsigned long long rotary_dim, unsigned long long position_offset, \
    float rope_base, float epsilon, unsigned long long cache_position, unsigned long long block_size, \
    unsigned long long cache_blocks, float *q_gate_output, float *q_rope_output, \
    void *k_cache, void *v_cache, unsigned short *k_scales, unsigned short *v_scales) { \
    __shared__ float partial[256]; \
    ullm_qwen35_qk_norm_rope_paged_kv_write_typed_impl<K_DTYPE, V_DTYPE>( \
        q_projected, k_projected, v_projected, q_weight, k_weight, block_table, q_heads, kv_heads, \
        head_dim, value_dim, rotary_dim, position_offset, rope_base, epsilon, cache_position, \
        block_size, cache_blocks, q_gate_output, q_rope_output, k_cache, v_cache, k_scales, v_scales, partial); \
}

ULLM_DEFINE_TYPED_QWEN35_PAGED_KV_WRITE(ULLM_FUSED_TYPED_KV_F32, ULLM_FUSED_TYPED_KV_F16, f32, f16)
ULLM_DEFINE_TYPED_QWEN35_PAGED_KV_WRITE(ULLM_FUSED_TYPED_KV_F32, ULLM_FUSED_TYPED_KV_FP8_E4M3FN, f32, fp8)
ULLM_DEFINE_TYPED_QWEN35_PAGED_KV_WRITE(ULLM_FUSED_TYPED_KV_F16, ULLM_FUSED_TYPED_KV_F32, f16, f32)
ULLM_DEFINE_TYPED_QWEN35_PAGED_KV_WRITE(ULLM_FUSED_TYPED_KV_F16, ULLM_FUSED_TYPED_KV_F16, f16, f16)
ULLM_DEFINE_TYPED_QWEN35_PAGED_KV_WRITE(ULLM_FUSED_TYPED_KV_F16, ULLM_FUSED_TYPED_KV_FP8_E4M3FN, f16, fp8)
ULLM_DEFINE_TYPED_QWEN35_PAGED_KV_WRITE(ULLM_FUSED_TYPED_KV_FP8_E4M3FN, ULLM_FUSED_TYPED_KV_F32, fp8, f32)
ULLM_DEFINE_TYPED_QWEN35_PAGED_KV_WRITE(ULLM_FUSED_TYPED_KV_FP8_E4M3FN, ULLM_FUSED_TYPED_KV_F16, fp8, f16)
ULLM_DEFINE_TYPED_QWEN35_PAGED_KV_WRITE(ULLM_FUSED_TYPED_KV_FP8_E4M3FN, ULLM_FUSED_TYPED_KV_FP8_E4M3FN, fp8, fp8)
#undef ULLM_DEFINE_TYPED_QWEN35_PAGED_KV_WRITE
)";
    }

