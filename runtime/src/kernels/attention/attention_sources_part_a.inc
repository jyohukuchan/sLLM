// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
    static const char *cached_prefix_rocwmma_fp8_e4m3_kernel_source() {
        return R"(
#include <rocwmma/rocwmma.hpp>

__device__ float ullm_cached_prefix_rocwmma_fp8_e4m3_to_f32(unsigned char value) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_cvt_f32_fp8(static_cast<unsigned int>(value), 0);
#else
    const unsigned int sign = static_cast<unsigned int>(value) >> 7;
    const unsigned int exponent = (static_cast<unsigned int>(value) >> 3) & 0x0fu;
    const unsigned int mantissa = static_cast<unsigned int>(value) & 0x07u;
    if (exponent == 0u) {
        const float magnitude = static_cast<float>(mantissa) * 0.001953125f;
        return sign == 0u ? magnitude : -magnitude;
    }
    const unsigned int fp32_bits = (sign << 31) | ((exponent + 120u) << 23) | (mantissa << 20);
    return __uint_as_float(fp32_bits);
#endif
}

extern "C" __global__ void ullm_cached_prefix_attn_fp8_e4m3_rocwmma_kernel(
    const unsigned char *q,
    const unsigned char *k_cache,
    const unsigned char *v_cache,
    unsigned long long cached_prefix_len,
    unsigned long long new_tokens,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long q_groups_per_kv,
    unsigned long long head_dim,
    unsigned long long value_dim,
    unsigned long long value_group_width,
    float softmax_scale,
    float q_scale,
    float k_scale,
    float v_scale,
    float *output) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    using namespace rocwmma;
    using FragA = fragment<matrix_a, 16, 16, 16, float8_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, float8_t, col_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float32_t, row_major>;

    const unsigned long long value_groups =
        (value_dim + value_group_width - 1ull) / value_group_width;
    const unsigned long long groups_per_token = kv_heads * q_groups_per_kv * value_groups;
    const unsigned long long block = static_cast<unsigned long long>(blockIdx.x);
    if (groups_per_token == 0ull || block >= new_tokens * groups_per_token) {
        return;
    }
    const unsigned long long token_index = block / groups_per_token;
    const unsigned long long group_within_token = block - token_index * groups_per_token;
    const unsigned long long value_group = group_within_token % value_groups;
    const unsigned long long head_group = group_within_token / value_groups;
    const unsigned long long kv_head = head_group / q_groups_per_kv;
    const unsigned long long q_group_in_kv = head_group - kv_head * q_groups_per_kv;
    const unsigned long long value_base = value_group * value_group_width;
    const unsigned long long value_count =
        value_dim - value_base < value_group_width ? value_dim - value_base : value_group_width;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long q_head_base = kv_head * q_per_kv + q_group_in_kv * 16ull;
    const unsigned long long cache_len = cached_prefix_len + token_index + 1ull;
    const unsigned int thread = threadIdx.x;
    const unsigned int lane = thread & 31u;

    __shared__ unsigned char k_tile[16 * 16];
    __shared__ float scores[16 * 16];
    __shared__ float row_max[16];
    __shared__ float row_sum[16];
    __shared__ float row_new_max[16];
    __shared__ float row_old_scale[16];
    __shared__ float row_tile_sum[16];
    extern __shared__ float row_acc[];

    for (unsigned int index = thread; index < 16u; index += blockDim.x) {
        row_max[index] = -3.4028234663852886e38f;
        row_sum[index] = 0.0f;
    }
    for (unsigned long long index = static_cast<unsigned long long>(thread);
         index < 16ull * value_count;
         index += static_cast<unsigned long long>(blockDim.x)) {
        row_acc[index] = 0.0f;
    }
    __syncthreads();

    FragA a;
    FragB b;
    FragC c;
    const unsigned long long q_base = (token_index * q_heads + q_head_base) * head_dim;
    const float score_scale = softmax_scale * q_scale * k_scale;

    for (unsigned long long tile_start = 0ull; tile_start < cache_len; tile_start += 16ull) {
        const unsigned long long remaining = cache_len - tile_start;
        const unsigned int tile_count =
            remaining < 16ull ? static_cast<unsigned int>(remaining) : 16u;

        if (thread < 32u) {
            fill_fragment(c, 0.0f);
        }
        for (unsigned long long dim_start = 0ull; dim_start < head_dim; dim_start += 16ull) {
            if (thread < 32u) {
                for (unsigned int index = lane; index < 16u * 16u; index += 32u) {
                    const unsigned int row = index / 16u;
                    const unsigned int col = index - row * 16u;
                    const unsigned long long source_timestep =
                        tile_start + static_cast<unsigned long long>(row);
                    if (row < tile_count) {
                        const unsigned long long k_index =
                            (source_timestep * kv_heads + kv_head) * head_dim +
                            dim_start + static_cast<unsigned long long>(col);
                        k_tile[index] = k_cache[k_index];
                    } else {
                        k_tile[index] = 0u;
                    }
                }
            }
            __syncthreads();

            if (thread < 32u) {
                load_matrix_sync(
                    a,
                    reinterpret_cast<const float8_t *>(q + q_base + dim_start),
                    static_cast<unsigned int>(head_dim));
                load_matrix_sync(b, reinterpret_cast<const float8_t *>(k_tile), 16);
                mma_sync(c, a, b, c);
            }
            __syncthreads();
        }
        if (thread < 32u) {
            store_matrix_sync(scores, c, 16);
        }
        __syncthreads();

        if (thread < 16u) {
            const unsigned int row = thread;
            float tile_max = -3.4028234663852886e38f;
            for (unsigned int col = 0u; col < tile_count; ++col) {
                const float score = scores[row * 16u + col] * score_scale;
                tile_max = score > tile_max ? score : tile_max;
            }
            const float old_max = row_max[row];
            const float new_max = old_max > tile_max ? old_max : tile_max;
            const float old_scale =
                old_max <= -3.0e38f ? 0.0f : expf(old_max - new_max);
            float tile_sum = 0.0f;
            for (unsigned int col = 0u; col < tile_count; ++col) {
                const float score = scores[row * 16u + col] * score_scale;
                tile_sum += expf(score - new_max);
            }
            row_new_max[row] = new_max;
            row_old_scale[row] = old_scale;
            row_tile_sum[row] = tile_sum;
        }
        __syncthreads();

        for (unsigned long long index = static_cast<unsigned long long>(thread);
             index < 16ull * value_count;
             index += static_cast<unsigned long long>(blockDim.x)) {
            const unsigned int row = static_cast<unsigned int>(index / value_count);
            const unsigned long long value =
                index - static_cast<unsigned long long>(row) * value_count;
            row_acc[index] *= row_old_scale[row];
            const float new_max = row_new_max[row];
            for (unsigned int col = 0u; col < tile_count; ++col) {
                const float score = scores[row * 16u + col] * score_scale;
                const float weight = expf(score - new_max);
                const unsigned long long source_timestep =
                    tile_start + static_cast<unsigned long long>(col);
                const unsigned long long v_index =
                    (source_timestep * kv_heads + kv_head) * value_dim + value_base + value;
                row_acc[index] +=
                    weight *
                    (ullm_cached_prefix_rocwmma_fp8_e4m3_to_f32(v_cache[v_index]) * v_scale);
            }
        }
        __syncthreads();

        if (thread < 16u) {
            const unsigned int row = thread;
            row_sum[row] = row_sum[row] * row_old_scale[row] + row_tile_sum[row];
            row_max[row] = row_new_max[row];
        }
        __syncthreads();
    }

    for (unsigned long long index = static_cast<unsigned long long>(thread);
         index < 16ull * value_count;
         index += static_cast<unsigned long long>(blockDim.x)) {
        const unsigned int row = static_cast<unsigned int>(index / value_count);
        const unsigned long long value =
            index - static_cast<unsigned long long>(row) * value_count;
        const unsigned long long output_base =
            (token_index * q_heads + q_head_base) * value_dim + value_base;
        const float denominator = row_sum[row];
            output[output_base + static_cast<unsigned long long>(row) * value_dim +
               value] = row_acc[index] / denominator;
    }
#else
    if (threadIdx.x == 0u) {
        const unsigned long long output_elements = new_tokens * q_heads * value_dim;
        for (unsigned long long i = 0ull; i < output_elements; ++i) {
            output[i] = 0.0f;
        }
    }
#endif
}
)";
    }

