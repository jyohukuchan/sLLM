// Included inside HipRtcRuntime's private section by ullm_runtime.cpp.
// Keep these HIPRTC source-string builders in this file; runtime loading and
// compile helpers stay in ullm_runtime.cpp.

    static const char *wmma_fp8_probe_kernel_source() {
        return R"(
extern "C" __global__ void ullm_wmma_fp8_probe_kernel(unsigned int *output) {
    if (blockIdx.x != 0u || threadIdx.x >= 32u) {
        return;
    }
#if defined(__gfx1200__) || defined(__gfx1201__)
    typedef int int2 __attribute__((ext_vector_type(2)));
    typedef float float8 __attribute__((ext_vector_type(8)));
    const int lane = static_cast<int>(threadIdx.x);
    int2 a = {0x01010101 ^ lane, 0x02020202 ^ (lane << 1)};
    int2 b = {0x03030303 ^ (lane << 2), 0x04040404 ^ (lane << 3)};
    float8 c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8 d = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
    union FloatBits {
        float f;
        unsigned int u;
    };
    FloatBits bits;
    bits.f = d[0];
    if (threadIdx.x == 0u) {
        output[0] = 0x574d4d41u ^ bits.u;
    }
#else
    if (threadIdx.x == 0u) {
        output[0] = 0u;
    }
#endif
}
)";
    }

    // Production shuffle-reduction source for the exact gfx1201/Qwen3.5 recurrent geometry. It
    // preserves the token-serial gated-delta recurrence and changes only the two reductions.
    static const char *linear_attn_recurrent_kernel_source() {
        return R"(
__device__ float ullm_linear_attn_recurrent_shuffle_reduce_to_lane0(
    float value,
    float *wave_partials) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % warpSize;
    const unsigned int wave = tid / warpSize;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, warpSize);
    }
    if (lane == 0u) {
        wave_partials[wave] = value;
    }
    __syncthreads();

    if (wave == 0u) {
        const unsigned int wave_count = (blockDim.x + warpSize - 1u) / warpSize;
        value = lane < wave_count ? wave_partials[lane] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            value += __shfl_down(value, offset, warpSize);
        }
    }
    return value;
}

extern "C" __global__ void ullm_linear_attn_recurrent_f32_kernel(
    const float *q,
    const float *k,
    const float *v,
    const float *gate,
    const float *beta,
    unsigned long long key_heads,
    unsigned long long value_heads,
    unsigned long long sequence_len,
    unsigned long long key_dim,
    unsigned long long value_dim,
    float *state,
    float *output) {
    const unsigned long long block = blockIdx.x;
    const unsigned long long value_head = block / value_dim;
    const unsigned long long value = block - value_head * value_dim;
    const unsigned int tid = threadIdx.x;
    // The exact gfx1201 production launch uses 128 threads, which is four wave32s. Eight slots
    // keep the scratch independent of a literal wave width while remaining above that target's
    // four waves.
    __shared__ float current_wave_partials[8];
    __shared__ float output_wave_partials[8];
    __shared__ float v_prime_shared;

    if (blockDim.x != 128u || value_head >= value_heads || value >= value_dim) {
        return;
    }
    const unsigned long long key_head_group = value_heads / key_heads;
    const unsigned long long key_head = value_head / key_head_group;
    const unsigned long long state_head_offset = value_head * key_dim * value_dim;
    const unsigned long long state_index =
        state_head_offset + static_cast<unsigned long long>(tid) * value_dim + value;
    float state_value = state[state_index];

    for (unsigned long long timestep = 0; timestep < sequence_len; ++timestep) {
        const unsigned long long value_head_index = timestep * value_heads + value_head;
        const unsigned long long key_head_index = timestep * key_heads + key_head;
        const unsigned long long qk_base = key_head_index * key_dim;
        const unsigned long long v_base = value_head_index * value_dim;
        const float decay = expf(gate[value_head_index]);
        const float beta_value = beta[value_head_index];
        const unsigned long long key = tid;
        const float key_value = k[qk_base + key];
        const float query_value = q[qk_base + key];
        const float decayed = state_value * decay;
        const float current = decayed * key_value;
        const float current_sum = ullm_linear_attn_recurrent_shuffle_reduce_to_lane0(
            current,
            current_wave_partials);
        if (tid == 0u) {
            v_prime_shared = (v[v_base + value] - current_sum) * beta_value;
        }
        __syncthreads();

        const float updated = decayed + key_value * v_prime_shared;
        state_value = updated;
        const float output_sum = ullm_linear_attn_recurrent_shuffle_reduce_to_lane0(
            updated * query_value,
            output_wave_partials);
        if (tid == 0u) {
            output[v_base + value] = output_sum;
        }

        // No post-output CTA barrier is needed because the two reductions use distinct scratch
        // arrays. Nonzero waves may start the next timestep while wave 0 completes the output
        // reduction, but that next timestep first writes only current_wave_partials. Before any
        // later reuse of output_wave_partials, the next v-prime handoff barrier requires wave 0
        // to have completed this reduction.
    }
    state[state_index] = state_value;
}
)";
    }

    static const char *wmma_fp8_qk_probe_kernel_source() {
        return R"(
__device__ int ullm_pack_fp8x4(const unsigned char *values) {
    const unsigned int packed = static_cast<unsigned int>(values[0]) |
        (static_cast<unsigned int>(values[1]) << 8) |
        (static_cast<unsigned int>(values[2]) << 16) |
        (static_cast<unsigned int>(values[3]) << 24);
    return static_cast<int>(packed);
}

extern "C" __global__ void ullm_wmma_fp8_qk_probe_kernel(
    const unsigned char *q,
    const unsigned char *k,
    float *output) {
    if (blockIdx.x != 0u || threadIdx.x >= 32u) {
        return;
    }
#if defined(__gfx1200__) || defined(__gfx1201__)
    typedef int int2 __attribute__((ext_vector_type(2)));
    typedef float float8 __attribute__((ext_vector_type(8)));
    const unsigned int lane = threadIdx.x;
    const unsigned int base = lane * 8u;
    int2 a = {ullm_pack_fp8x4(q + base), ullm_pack_fp8x4(q + base + 4u)};
    int2 b = {ullm_pack_fp8x4(k + base), ullm_pack_fp8x4(k + base + 4u)};
    float8 c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8 d = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
    output[base + 0u] = d[0];
    output[base + 1u] = d[1];
    output[base + 2u] = d[2];
    output[base + 3u] = d[3];
    output[base + 4u] = d[4];
    output[base + 5u] = d[5];
    output[base + 6u] = d[6];
    output[base + 7u] = d[7];
#else
    const unsigned int base = threadIdx.x * 8u;
    output[base + 0u] = 0.0f;
    output[base + 1u] = 0.0f;
    output[base + 2u] = 0.0f;
    output[base + 3u] = 0.0f;
    output[base + 4u] = 0.0f;
    output[base + 5u] = 0.0f;
    output[base + 6u] = 0.0f;
    output[base + 7u] = 0.0f;
#endif
}
)";
    }

    static const char *rocwmma_fp8_qk_probe_kernel_source() {
        return R"(
#include <rocwmma/rocwmma.hpp>

extern "C" __global__ void ullm_rocwmma_fp8_qk_probe_kernel(
    const unsigned char *q,
    const unsigned char *k,
    float *output) {
    if (blockIdx.x != 0u || threadIdx.x >= 32u) {
        return;
    }
#if defined(__gfx1200__) || defined(__gfx1201__)
    using namespace rocwmma;
    using FragA = fragment<matrix_a, 16, 16, 16, float8_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, float8_t, col_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float32_t, row_major>;

    FragA a;
    FragB b;
    FragC c;
    fill_fragment(c, 0.0f);
    load_matrix_sync(a, reinterpret_cast<const float8_t *>(q), 16);
    load_matrix_sync(b, reinterpret_cast<const float8_t *>(k), 16);
    mma_sync(c, a, b, c);
    store_matrix_sync(output, c, 16);
#else
    if (threadIdx.x == 0u) {
        for (unsigned int i = 0u; i < 256u; ++i) {
            output[i] = 0.0f;
        }
    }
#endif
}
)";
    }

    static const char *rocwmma_fp8_attn_probe_kernel_source() {
        return R"(
#include <rocwmma/rocwmma.hpp>

extern "C" __global__ void ullm_rocwmma_fp8_attn_probe_kernel(
    const unsigned char *q,
    const unsigned char *k,
    const float *v,
    float *output) {
    if (blockIdx.x != 0u || threadIdx.x >= 32u) {
        return;
    }
#if defined(__gfx1200__) || defined(__gfx1201__)
    using namespace rocwmma;
    using FragA = fragment<matrix_a, 16, 16, 16, float8_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, float8_t, col_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float32_t, row_major>;

    __shared__ float scores[16 * 16];
    __shared__ float row_max[16];
    __shared__ float row_sum[16];
    __shared__ float row_acc[16 * 16];

    const unsigned int lane = threadIdx.x;
    for (unsigned int index = lane; index < 16u; index += 32u) {
        row_max[index] = -3.4028234663852886e38f;
        row_sum[index] = 0.0f;
    }
    for (unsigned int index = lane; index < 16u * 16u; index += 32u) {
        row_acc[index] = 0.0f;
    }
    __syncthreads();

    FragA a;
    FragB b;
    FragC c;
    load_matrix_sync(a, reinterpret_cast<const float8_t *>(q), 16);
    for (unsigned int tile = 0u; tile < 2u; ++tile) {
        fill_fragment(c, 0.0f);
        load_matrix_sync(b, reinterpret_cast<const float8_t *>(k + tile * 16u * 16u), 16);
        mma_sync(c, a, b, c);
        store_matrix_sync(scores, c, 16);
        __syncthreads();

        if (lane < 16u) {
            const unsigned int row = lane;
            float tile_max = -3.4028234663852886e38f;
            for (unsigned int col = 0u; col < 16u; ++col) {
                const float score = scores[row * 16u + col];
                tile_max = score > tile_max ? score : tile_max;
            }
            const float old_max = row_max[row];
            const float new_max = old_max > tile_max ? old_max : tile_max;
            const float old_scale =
                old_max <= -3.0e38f ? 0.0f : expf(old_max - new_max);
            float tile_sum = 0.0f;
            for (unsigned int value = 0u; value < 16u; ++value) {
                row_acc[row * 16u + value] *= old_scale;
            }
            for (unsigned int col = 0u; col < 16u; ++col) {
                const float weight = expf(scores[row * 16u + col] - new_max);
                tile_sum += weight;
                const unsigned int source_token = tile * 16u + col;
                for (unsigned int value = 0u; value < 16u; ++value) {
                    row_acc[row * 16u + value] += weight * v[source_token * 16u + value];
                }
            }
            row_sum[row] = row_sum[row] * old_scale + tile_sum;
            row_max[row] = new_max;
        }
        __syncthreads();
    }

    if (lane < 16u) {
        const unsigned int row = lane;
        const float denominator = row_sum[row];
        for (unsigned int value = 0u; value < 16u; ++value) {
            output[row * 16u + value] = row_acc[row * 16u + value] / denominator;
        }
    }
#else
    if (threadIdx.x == 0u) {
        for (unsigned int i = 0u; i < 256u; ++i) {
            output[i] = 0.0f;
        }
    }
#endif
}
)";
    }

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

    static const char *aq4_kernel_source() {
        return R"(
extern "C" __global__ void ullm_aq4_dequant_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long elements,
    float *output,
    unsigned int *error_out) {
    const unsigned long long element =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element >= elements) {
        return;
    }
    const unsigned char packed = indices[element >> 1];
    const unsigned char codebook_index =
        (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
    const unsigned long long group = element / group_size;
    const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
    if (scale_index >= scale_count) {
        atomicOr(error_out, 1u);
        output[element] = 0.0f;
        return;
    }
    output[element] = codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
}
)";
    }

    static const char *aq4_row_kernel_source() {
        return R"(
extern "C" __global__ void ullm_aq4_row_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long row_index,
    float *output,
    unsigned int *error_out) {
    const unsigned long long col =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= cols || row_index >= rows) {
        return;
    }
    const unsigned long long element = row_index * cols + col;
    const unsigned char packed = indices[element >> 1];
    const unsigned char codebook_index =
        (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
    const unsigned long long group = element / group_size;
    const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
    if (scale_index >= scale_count) {
        atomicOr(error_out, 1u);
        output[col] = 0.0f;
        return;
    }
    float value = codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
    if (row_scales != nullptr && row_index < row_scale_count) {
        value *= row_scales[row_index];
    }
    output[col] = value;
}
)";
    }

    static unsigned int aq4_rows_per_block_for_arch(
        const std::string &arch,
        const char *env_name = nullptr) {
        const unsigned int fallback = arch.rfind("gfx12", 0) == 0 ? 32u : 1u;
        return aq4_rows_per_block_from_env(env_name, nullptr, fallback);
    }

    static std::string aq4_rows_per_block_preamble_for_rows(unsigned int rows_per_block) {
        return "#define ULLM_AQ4_ROWS_PER_BLOCK " + std::to_string(rows_per_block) + "\n";
    }

    static std::string aq4_rows_per_block_preamble(
        const std::string &arch,
        const char *env_name = nullptr) {
        return aq4_rows_per_block_preamble_for_rows(
            aq4_rows_per_block_for_arch(arch, env_name));
    }

    static unsigned int aq4_fused_rows_per_block_for_arch(
        const std::string &arch,
        unsigned int rdna4_rows_per_block,
        unsigned int rdna2_rows_per_block = 2u,
        const char *env_name = nullptr) {
        // RDNA2 loses too much per-row parallelism above 2 rows/block on fused AQ4 kernels.
        const unsigned int fallback =
            arch.rfind("gfx12", 0) == 0 ? rdna4_rows_per_block : rdna2_rows_per_block;
        return aq4_rows_per_block_from_env(env_name, "ULLM_AQ4_FUSED_RPB", fallback);
    }

    static std::string aq4_fused_rows_per_block_preamble(
        const std::string &arch,
        unsigned int rdna4_rows_per_block,
        unsigned int rdna2_rows_per_block = 2u,
        const char *env_name = nullptr) {
        return aq4_rows_per_block_preamble_for_rows(
            aq4_fused_rows_per_block_for_arch(
                arch,
                rdna4_rows_per_block,
                rdna2_rows_per_block,
                env_name));
    }

    static std::string aq4_matvec_kernel_source_for_arch(const std::string &arch) {
        const unsigned int rows_per_block =
            aq4_rows_per_block_for_arch(arch, "ULLM_AQ4_MATVEC_RPB");
        const std::string preamble = aq4_rows_per_block_preamble_for_rows(rows_per_block);
        if (arch == "gfx1201" && rows_per_block == 32u) {
            return preamble + aq4_matvec_kernel_source();
        }
        return preamble + aq4_matvec_wide_load_reference_kernel_source();
    }

    // Direct-only M=1 comparison. It compiles the retained wide-load rollback body under an
    // isolated entry-point name.
    static std::string aq4_matvec_wide_load_prototype_kernel_source_for_arch(
        const std::string &arch) {
        return aq4_rows_per_block_preamble(arch, "ULLM_AQ4_MATVEC_RPB") +
               aq4_matvec_wide_load_prototype_kernel_source();
    }

    // Direct-only gfx1201 experiment. The production M=1 body uses RPB=32, so one wave32
    // contains four independent 8-thread rows. Fix this candidate to that geometry and use
    // width=8 shuffles so a row's reduction never reads a neighbouring row's partials.
    static std::string aq4_matvec_shuffle_prototype_kernel_source_for_arch(
        const std::string &arch) {
        (void)arch;
        return aq4_rows_per_block_preamble_for_rows(32u) +
               aq4_matvec_shuffle_prototype_kernel_source();
    }

    static std::string aq4_matvec_batch_kernel_source_for_arch(const std::string &arch) {
        return aq4_rows_per_block_preamble(arch, "ULLM_AQ4_MATVEC_BATCH_RPB") +
               aq4_matvec_batch_kernel_source();
    }

    static std::string aq4_matvec_tiled_kernel_source_for_arch(const std::string &arch) {
        // The first tiled production slice is intentionally gfx1201-only at dispatch time.
        // Keep the source architecture-neutral so a future exact-arch rule can opt in without
        // changing the cache or kernel name.
        (void)arch;
        return aq4_matvec_tiled_kernel_source();
    }

    static std::string aq4_matvec_register_kernel_source_for_arch(const std::string &arch) {
        (void)arch;
        return aq4_matvec_register_kernel_source();
    }

    static std::string aq4_matvec_top1_kernel_source_for_arch(const std::string &arch) {
        std::string preamble =
            aq4_rows_per_block_preamble_for_rows(aq4_matvec_top1_rows_per_block_from_env());
        if (arch.rfind("gfx12", 0) != 0 ||
            std::getenv("ULLM_DISABLE_AQ4_MATVEC_TOP1_WARP_REDUCE") != nullptr) {
            preamble += "#define ULLM_AQ4_MATVEC_TOP1_USE_SHARED_REDUCE 1\n";
        }
        return preamble + aq4_matvec_top1_kernel_source();
    }

    static std::string aq4_matvec_add_kernel_source_for_arch(const std::string &arch) {
        const unsigned int rows_per_block = aq4_fused_rows_per_block_for_arch(
            arch,
            8u,
            2u,
            "ULLM_AQ4_MATVEC_ADD_RPB");
        const std::string preamble = aq4_rows_per_block_preamble_for_rows(rows_per_block);
        if (arch == "gfx1201" && rows_per_block == 8u) {
            // Test-only rollback selector for an in-process-identical full-model A/B.  The
            // HIPRTC module cache is process-local, so a clean profile process can select the
            // retained pre-specialization shuffle body without changing the served default.
            // No manifest sets this; unset always means the production specialized body.
            if (const char *reference = std::getenv("ULLM_AQ4_MATVEC_ADD_USE_SHUFFLE_REFERENCE");
                reference != nullptr && reference[0] == '1' && reference[1] == '\0') {
                return preamble + aq4_matvec_add_shuffle_reference_kernel_source();
            }
            return preamble + aq4_matvec_add_kernel_source();
        }
        return preamble + aq4_matvec_add_wide_load_reference_kernel_source();
    }

    // Direct-only experiment. Keep add's RPB=8 geometry; unlike the multi-stream fused
    // experiments, this body owns exactly one packed AQ4 stream and one accumulator.
    static std::string aq4_matvec_add_wide_load_prototype_kernel_source_for_arch(
        const std::string &arch) {
        return aq4_fused_rows_per_block_preamble(
                   arch,
                   8u,
                   2u,
                   "ULLM_AQ4_MATVEC_ADD_RPB") +
               aq4_matvec_add_wide_load_prototype_kernel_source();
    }

    // Direct-only gfx1201 experiment. It retains the promoted uint4 weight traversal and
    // residual-add epilogue, changing only the RPB=8 per-row LDS tree to a wave32 shuffle.
    static std::string aq4_matvec_add_shuffle_prototype_kernel_source_for_arch(
        const std::string &arch) {
        return aq4_fused_rows_per_block_preamble(
                   arch,
                   8u,
                   2u,
                   "ULLM_AQ4_MATVEC_ADD_RPB") +
               aq4_matvec_add_shuffle_prototype_kernel_source();
    }

    static std::string aq4_matvec_pair_kernel_source_for_arch(const std::string &arch) {
        return aq4_fused_rows_per_block_preamble(
                   arch,
                   16u,
                   4u,
                   "ULLM_AQ4_MATVEC_PAIR_RPB") +
               aq4_matvec_pair_kernel_source();
    }

    static std::string aq4_matvec_triple_kernel_source_for_arch(const std::string &arch) {
        const unsigned int rows_per_block = aq4_fused_rows_per_block_for_arch(
            arch,
            8u,
            4u,
            "ULLM_AQ4_MATVEC_TRIPLE_RPB");
        const bool use_gfx1201_shuffle = arch == "gfx1201" && rows_per_block == 8u;
        return aq4_rows_per_block_preamble_for_rows(rows_per_block) +
               (use_gfx1201_shuffle ? aq4_matvec_triple_kernel_source()
                                    : aq4_matvec_triple_scalar_reference_kernel_source());
    }

    // Direct-only gfx1201 experiment. It retains triple's scalar packed-byte traversal and
    // three independent accumulators, but replaces the RPB=8 per-row LDS trees with wave32
    // shuffles. The direct ABI rejects any RPB override other than 8 before compiling it.
    static std::string aq4_matvec_triple_shuffle_prototype_kernel_source_for_arch(
        const std::string &arch) {
        return aq4_fused_rows_per_block_preamble(
                   arch,
                   8u,
                   4u,
                   "ULLM_AQ4_MATVEC_TRIPLE_RPB") +
               aq4_matvec_triple_shuffle_prototype_kernel_source();
    }

    static std::string aq4_matvec_qkv_z_gate_beta_kernel_source_for_arch(
        const std::string &arch) {
        const unsigned int rows_per_block = aq4_fused_rows_per_block_for_arch(
            arch,
            4u,
            2u,
            "ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB");
        const bool use_gfx1201_shuffle = arch == "gfx1201" && rows_per_block == 4u;
        return aq4_rows_per_block_preamble_for_rows(rows_per_block) +
               (use_gfx1201_shuffle ? aq4_matvec_qkv_z_gate_beta_kernel_source()
                                    : aq4_matvec_qkv_z_gate_beta_scalar_reference_kernel_source());
    }

    // Direct-only M=1 experiment. Keep the production fused body intact: this source replaces
    // only its scalar packed-byte helpers and gives the resulting module a unique entry point.
    static std::string aq4_matvec_qkv_z_gate_beta_wide_load_prototype_kernel_source_for_arch(
        const std::string &arch) {
        return aq4_fused_rows_per_block_preamble(
                   arch,
                   4u,
                   2u,
                   "ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB") +
               aq4_matvec_qkv_z_gate_beta_wide_load_prototype_kernel_source();
    }

    // Direct-only gfx1201 experiment. It retains the production scalar packed-byte traversal and
    // paired streams, but changes the 64-thread-per-row LDS trees into wave32 shuffles followed
    // by the minimal two-wave-per-row LDS combine.
    static std::string aq4_matvec_qkv_z_gate_beta_shuffle_prototype_kernel_source_for_arch(
        const std::string &arch) {
        return aq4_fused_rows_per_block_preamble(
                   arch,
                   4u,
                   2u,
                   "ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB") +
               aq4_matvec_qkv_z_gate_beta_shuffle_prototype_kernel_source();
    }

    static std::string aq4_matvec_silu_mul_kernel_source_for_arch(const std::string &arch) {
        const unsigned int rows_per_block = aq4_fused_rows_per_block_for_arch(
            arch,
            8u,
            2u,
            "ULLM_AQ4_MATVEC_SILU_MUL_RPB");
        const bool use_gfx1201_shuffle = arch == "gfx1201" && rows_per_block == 8u;
        return aq4_rows_per_block_preamble_for_rows(rows_per_block) +
               (use_gfx1201_shuffle ? aq4_matvec_silu_mul_kernel_source()
                                    : aq4_matvec_silu_mul_scalar_reference_kernel_source());
    }

    // Direct-only M=1 experiment; see the qkv/z/gate/beta counterpart above.
    static std::string aq4_matvec_silu_mul_wide_load_prototype_kernel_source_for_arch(
        const std::string &arch) {
        return aq4_fused_rows_per_block_preamble(
                   arch,
                   8u,
                   2u,
                   "ULLM_AQ4_MATVEC_SILU_MUL_RPB") +
               aq4_matvec_silu_mul_wide_load_prototype_kernel_source();
    }

    // Direct-only gfx1201 experiment.  It retains the production scalar packed-byte traversal and
    // changes only the two independent per-row reductions from LDS trees to wave32 shuffles.
    static std::string aq4_matvec_silu_mul_shuffle_prototype_kernel_source_for_arch(
        const std::string &arch) {
        return aq4_fused_rows_per_block_preamble(
                   arch,
                   8u,
                   2u,
                   "ULLM_AQ4_MATVEC_SILU_MUL_RPB") +
               aq4_matvec_silu_mul_shuffle_prototype_kernel_source();
    }

    static std::string aq4_matvec_gate_beta_kernel_source_for_arch(const std::string &arch) {
        return aq4_rows_per_block_preamble(arch, "ULLM_AQ4_MATVEC_RPB") +
               aq4_matvec_gate_beta_kernel_source();
    }

    // Superseded scalar M=1 AQ4 matvec reference. It is deliberately not passed to HIPRTC:
    // production keeps the same entry-point ABI but uses the validated wide-load body below.
    // Retain this source for controlled rollback comparisons.
    [[maybe_unused]] static const char *aq4_matvec_scalar_reference_kernel_source() {
        return R"(
extern "C" __global__ void ullm_aq4_matvec_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    float *output) {
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float partial[256];
    float sum = 0.0f;
    if (row < rows) {
        const unsigned long long row_offset = static_cast<unsigned long long>(row) * cols;
        if ((cols % group_size) == 0ull) {
            const unsigned long long groups_per_row = cols / group_size;
            for (unsigned long long group_in_row = lane; group_in_row < groups_per_row;
                 group_in_row += threads_per_row) {
                const unsigned long long group = static_cast<unsigned long long>(row) * groups_per_row +
                    group_in_row;
                const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
                if (scale_index >= scale_count) {
                    continue;
                }
                float raw_sum = 0.0f;
                const unsigned long long col_start = group_in_row * group_size;
                if (group_size == 16ull) {
#pragma unroll
                    for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                        const unsigned long long col = col_start + offset;
                        const unsigned char packed = indices[(row_offset + col) >> 1];
                        raw_sum += codebook[packed & 0x0f] * input[col];
                        raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                    }
                } else if (group_size == 8ull) {
#pragma unroll
                    for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                        const unsigned long long col = col_start + offset;
                        const unsigned char packed = indices[(row_offset + col) >> 1];
                        raw_sum += codebook[packed & 0x0f] * input[col];
                        raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                    }
                } else if ((group_size & 1ull) == 0ull) {
                    for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                        const unsigned long long col = col_start + offset;
                        const unsigned char packed = indices[(row_offset + col) >> 1];
                        raw_sum += codebook[packed & 0x0f] * input[col];
                        raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                    }
                } else {
                    for (unsigned long long offset = 0; offset < group_size; ++offset) {
                        const unsigned long long col = col_start + offset;
                        const unsigned long long element = row_offset + col;
                        const unsigned char packed = indices[element >> 1];
                        const unsigned char codebook_index =
                            (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                        raw_sum += codebook[codebook_index] * input[col];
                    }
                }
                sum += raw_sum * scale_values[scale_index] * tensor_scale;
            }
        } else {
            for (unsigned long long col = lane; col < cols; col += threads_per_row) {
                const unsigned long long element = row_offset + col;
                const unsigned char packed = indices[element >> 1];
                const unsigned char codebook_index =
                    (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                const unsigned long long group = element / group_size;
                const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
                if (scale_index < scale_count) {
                    const float value =
                        codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                    sum += value * input[col];
                }
            }
        }
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            partial[partial_offset + lane] += partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && row < rows) {
        float value = partial[partial_offset];
        if (row_scales != nullptr && row < row_scale_count) {
            value *= row_scales[row];
        }
        output[row] = value;
    }
}
)";
    }

    // Superseded LDS-reduction AQ4 matvec wide-load reference. It remains the rollback body and
    // is used for every architecture or RPB outside the validated gfx1201 RPB=32 geometry.
    // Retain this source for controlled rollback comparisons.
    [[maybe_unused]] static const char *aq4_matvec_wide_load_reference_kernel_source() {
        return R"(
// Validated Qwen3.5 AQ4 M=1 geometries have group8/group16 and complete 32-element chunks.
// `uint4` is the same explicit 128-bit packed-AQ4 load type used by the Wide-K WMMA kernels.
extern "C" __global__ void ullm_aq4_matvec_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    float *output) {
    static_assert(sizeof(uint4) == 16u, "AQ4 wide load must remain 128 bits");
    static_assert(alignof(uint4) == 16u, "AQ4 wide load must remain 16-byte aligned");
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float partial[256];
    float sum = 0.0f;
    if (row < rows) {
        const unsigned long long row_offset = row * cols;
        const unsigned long long chunks_per_row = cols / 32ull;
        const unsigned long long groups_per_row = cols / group_size;
        // `cols` is a multiple of 32, so row_offset/2 and every chunk address are 16-byte aligned.
        const unsigned long long row_byte_offset = row_offset >> 1;
        for (unsigned long long chunk = lane; chunk < chunks_per_row;
             chunk += threads_per_row) {
            const uint4 packed = *reinterpret_cast<const uint4 *>(
                indices + row_byte_offset + chunk * 16ull);
            const unsigned long long col_base = chunk * 32ull;
            const unsigned long long group_base =
                row * groups_per_row + (col_base / group_size);
            // A uint4 has four 32-bit words, hence sixteen packed AQ4 bytes total. A group8
            // consumes four bytes and a group16 consumes eight bytes, possibly spanning words.
            const unsigned int groups_per_chunk = group_size == 8ull ? 4u : 2u;
#pragma unroll
            for (unsigned int group_in_chunk = 0u; group_in_chunk < 4u; ++group_in_chunk) {
                if (group_in_chunk >= groups_per_chunk) {
                    continue;
                }
                const unsigned long long group = group_base + group_in_chunk;
                const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
                if (scale_index >= scale_count) {
                    continue;
                }
                const unsigned int byte_start =
                    group_in_chunk * static_cast<unsigned int>(group_size >> 1);
                const unsigned int byte_count = static_cast<unsigned int>(group_size >> 1);
                float raw_sum = 0.0f;
#pragma unroll
                for (unsigned int byte_offset = 0u; byte_offset < 8u; ++byte_offset) {
                    if (byte_offset >= byte_count) {
                        continue;
                    }
                    const unsigned int packed_byte_index = byte_start + byte_offset;
                    const unsigned int word_index = packed_byte_index >> 2;
                    const unsigned int word = word_index == 0u ? packed.x :
                        (word_index == 1u ? packed.y :
                            (word_index == 2u ? packed.z : packed.w));
                    const unsigned int packed_byte =
                        (word >> (8u * (packed_byte_index & 3u))) & 0xffu;
                    const unsigned long long col = col_base +
                        static_cast<unsigned long long>(packed_byte_index * 2u);
                    raw_sum += codebook[packed_byte & 0x0fu] * input[col];
                    raw_sum += codebook[(packed_byte >> 4) & 0x0fu] * input[col + 1ull];
                }
                sum += raw_sum * scale_values[scale_index] * tensor_scale;
            }
        }
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            partial[partial_offset + lane] += partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && row < rows) {
        float value = partial[partial_offset];
        if (row_scales != nullptr && row < row_scale_count) {
            value *= row_scales[row];
        }
        output[row] = value;
    }
}
)";
    }

    // Production M=1 AQ4 matvec body. The host ABI and HIPRTC entry-point name remain unchanged,
    // so resident decode continues through its existing direct aq4_matvec_f32 dispatch. It
    // retains the wide-load traversal and changes only the RPB=32 per-row LDS tree to the
    // validated gfx1201 width-8 sub-wave shuffle reduction.
    static std::string aq4_matvec_kernel_source() {
        std::string production = aq4_matvec_wide_load_reference_kernel_source();
        const std::string lds_setup = R"(    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float partial[256];
    float sum = 0.0f;)";
        const std::string shuffle_setup = R"(    float sum = 0.0f;)";
        const size_t setup_offset = production.find(lds_setup);
        if (setup_offset == std::string::npos) {
            return "";
        }
        production.replace(setup_offset, lds_setup.size(), shuffle_setup);

        const std::string lds_tree = R"(    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            partial[partial_offset + lane] += partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && row < rows) {
        float value = partial[partial_offset];)";
        const std::string shuffle_reduction = R"(    // RPB=32 gives four rows per wave32. width=8 keeps each row's lanes isolated.
    for (unsigned int offset = 4u; offset > 0u; offset >>= 1u) {
        sum += __shfl_down(sum, offset, 8u);
    }
    if (lane == 0 && row < rows) {
        float value = sum;)";
        const size_t tree_offset = production.find(lds_tree);
        if (tree_offset == std::string::npos) {
            return "";
        }
        production.replace(tree_offset, lds_tree.size(), shuffle_reduction);
        return production;
    }

    // Keep the direct-only prototype ABI available for historical differential/timing comparison.
    // It compiles the wide-load rollback body under its original, isolated entry-point name.
    static std::string aq4_matvec_wide_load_prototype_kernel_source() {
        std::string source = aq4_matvec_wide_load_reference_kernel_source();
        constexpr const char production_name[] = "ullm_aq4_matvec_f32_kernel";
        const size_t name_offset = source.find(production_name);
        if (name_offset == std::string::npos) {
            return "";
        }
        source.replace(
            name_offset,
            sizeof(production_name) - 1u,
            "ullm_aq4_matvec_wide_load_prototype_f32_kernel");
        return source;
    }

    // Keep the direct-only prototype ABI available for historical differential/timing comparison.
    // It compiles the same validated body under its original, isolated entry-point name.
    static std::string aq4_matvec_shuffle_prototype_kernel_source() {
        std::string source = aq4_matvec_kernel_source();
        constexpr const char production_name[] = "ullm_aq4_matvec_f32_kernel";
        const size_t name_offset = source.find(production_name);
        if (name_offset == std::string::npos) {
            return "";
        }
        source.replace(
            name_offset,
            sizeof(production_name) - 1u,
            "ullm_aq4_matvec_shuffle_prototype_f32_kernel");
        return source;
    }

    static const char *aq4_matvec_batch_kernel_source() {
        return R"(
extern "C" __global__ void ullm_aq4_matvec_batch_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long batch_count,
    float *output) {
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned long long batch = static_cast<unsigned long long>(blockIdx.y);
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float partial[256];
    float sum = 0.0f;
    if (row < rows && batch < batch_count) {
        const unsigned long long row_offset = static_cast<unsigned long long>(row) * cols;
        const float *batch_input = input + batch * cols;
        if ((cols % group_size) == 0ull) {
            const unsigned long long groups_per_row = cols / group_size;
            for (unsigned long long group_in_row = lane; group_in_row < groups_per_row;
                 group_in_row += threads_per_row) {
                const unsigned long long group = static_cast<unsigned long long>(row) * groups_per_row +
                    group_in_row;
                const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
                if (scale_index >= scale_count) {
                    continue;
                }
                float raw_sum = 0.0f;
                const unsigned long long col_start = group_in_row * group_size;
                if (group_size == 16ull) {
#pragma unroll
                    for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                        const unsigned long long col = col_start + offset;
                        const unsigned char packed = indices[(row_offset + col) >> 1];
                        raw_sum += codebook[packed & 0x0f] * batch_input[col];
                        raw_sum += codebook[(packed >> 4) & 0x0f] * batch_input[col + 1ull];
                    }
                } else if (group_size == 8ull) {
#pragma unroll
                    for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                        const unsigned long long col = col_start + offset;
                        const unsigned char packed = indices[(row_offset + col) >> 1];
                        raw_sum += codebook[packed & 0x0f] * batch_input[col];
                        raw_sum += codebook[(packed >> 4) & 0x0f] * batch_input[col + 1ull];
                    }
                } else if ((group_size & 1ull) == 0ull) {
                    for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                        const unsigned long long col = col_start + offset;
                        const unsigned char packed = indices[(row_offset + col) >> 1];
                        raw_sum += codebook[packed & 0x0f] * batch_input[col];
                        raw_sum += codebook[(packed >> 4) & 0x0f] * batch_input[col + 1ull];
                    }
                } else {
                    for (unsigned long long offset = 0; offset < group_size; ++offset) {
                        const unsigned long long col = col_start + offset;
                        const unsigned long long element = row_offset + col;
                        const unsigned char packed = indices[element >> 1];
                        const unsigned char codebook_index =
                            (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                        raw_sum += codebook[codebook_index] * batch_input[col];
                    }
                }
                sum += raw_sum * scale_values[scale_index] * tensor_scale;
            }
        } else {
            for (unsigned long long col = lane; col < cols; col += threads_per_row) {
                const unsigned long long element = row_offset + col;
                const unsigned char packed = indices[element >> 1];
                const unsigned char codebook_index =
                    (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                const unsigned long long group = element / group_size;
                const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
                if (scale_index < scale_count) {
                    const float value =
                        codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                    sum += value * batch_input[col];
                }
            }
        }
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            partial[partial_offset + lane] += partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && row < rows && batch < batch_count) {
        float value = partial[partial_offset];
        if (row_scales != nullptr && row < row_scale_count) {
            value *= row_scales[row];
        }
        output[batch * rows + row] = value;
    }
}
)";
    }

    static const char *aq4_matvec_tiled_kernel_source() {
        return R"(
#define ULLM_AQ4_TILED_BM 8u
#define ULLM_AQ4_TILED_BN 32u
#define ULLM_AQ4_TILED_BK 128u
#define ULLM_AQ4_TILED_THREADS 256u
static_assert(ULLM_AQ4_TILED_BM * ULLM_AQ4_TILED_BN == ULLM_AQ4_TILED_THREADS,
              "AQ4 tiled kernel requires one thread per output");
static_assert(ULLM_AQ4_TILED_BK % 16u == 0u,
              "AQ4 tiled kernel requires group16-compatible K tile");

extern "C" __global__ void ullm_aq4_gemm_tiled_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long batch_count,
    float *output) {
    constexpr unsigned int BM = ULLM_AQ4_TILED_BM;
    constexpr unsigned int BN = ULLM_AQ4_TILED_BN;
    constexpr unsigned int BK = ULLM_AQ4_TILED_BK;
    constexpr unsigned int tid_count = ULLM_AQ4_TILED_THREADS;
    const unsigned int tid = threadIdx.x;
    const unsigned int tile_m = tid / BN;
    const unsigned int tile_n = tid - tile_m * BN;
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * BN + tile_n;
    const unsigned long long batch =
        static_cast<unsigned long long>(blockIdx.y) * BM + tile_m;
    __shared__ float weight_tile[BN * BK];
    __shared__ float input_tile[BM * BK];
    float sum = 0.0f;
    const unsigned long long groups_per_row = cols / group_size;

    // The dispatch classifier admits only group_size == 16 and cols divisible by BK.
    // Keep an explicit guard so an accidental direct launch cannot read malformed metadata.
    const bool valid_geometry = group_size == 16ull && groups_per_row * group_size == cols;
    for (unsigned long long k_base = 0ull; k_base < cols; k_base += BK) {
        for (unsigned int flat = tid; flat < BN * BK; flat += tid_count) {
            const unsigned int local_n = flat / BK;
            const unsigned int local_k = flat - local_n * BK;
            const unsigned long long load_row =
                static_cast<unsigned long long>(blockIdx.x) * BN + local_n;
            const unsigned long long load_col = k_base + local_k;
            float value = 0.0f;
            if (valid_geometry && load_row < rows && load_col < cols) {
                const unsigned long long element = load_row * cols + load_col;
                const unsigned char packed = indices[element >> 1];
                const unsigned char codebook_index =
                    (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                const unsigned long long group = load_row * groups_per_row + load_col / group_size;
                const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
                // Host validation rejects this case before launch. Keep the load fail-closed for
                // safety if a caller bypasses that validation.
                if (scale_index < scale_count) {
                    value = codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                }
            }
            weight_tile[flat] = value;
        }
        for (unsigned int flat = tid; flat < BM * BK; flat += tid_count) {
            const unsigned int local_m = flat / BK;
            const unsigned int local_k = flat - local_m * BK;
            const unsigned long long load_batch =
                static_cast<unsigned long long>(blockIdx.y) * BM + local_m;
            const unsigned long long load_col = k_base + local_k;
            input_tile[flat] =
                load_batch < batch_count && load_col < cols ? input[load_batch * cols + load_col] : 0.0f;
        }
        __syncthreads();
        if (row < rows && batch < batch_count && valid_geometry) {
            const unsigned int weight_offset = tile_n * BK;
            const unsigned int input_offset = tile_m * BK;
            // One thread owns each output, so accumulation is deterministic in ascending K
            // order. This differs from the legacy lane-partial reduction order by design.
            for (unsigned int k = 0; k < BK; ++k) {
                sum += weight_tile[weight_offset + k] * input_tile[input_offset + k];
            }
        }
        __syncthreads();
    }
    if (row < rows && batch < batch_count && valid_geometry) {
        if (row_scales != nullptr && row < row_scale_count) {
            sum *= row_scales[row];
        }
        output[batch * rows + row] = sum;
    }
}
)";
    }

    static const char *aq4_matvec_register_kernel_source() {
        return R"(
#define ULLM_AQ4_REGISTER_ROWS_PER_BLOCK 32u
#define ULLM_AQ4_REGISTER_LANES_PER_ROW 8u
#define ULLM_AQ4_REGISTER_THREADS 256u
static_assert(
    ULLM_AQ4_REGISTER_ROWS_PER_BLOCK * ULLM_AQ4_REGISTER_LANES_PER_ROW ==
        ULLM_AQ4_REGISTER_THREADS,
    "AQ4 register kernel requires eight lanes per output row");

template <unsigned int BM, unsigned int GROUP_SIZE>
__device__ __forceinline__ void ullm_aq4_gemm_register_f32_impl(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long batch_count,
    float *output) {
    static_assert(BM == 4u || BM == 8u, "AQ4 register kernel supports BM4 or BM8");
    static_assert(
        GROUP_SIZE == 8u || GROUP_SIZE == 16u,
        "AQ4 register kernel supports group8 or group16");
    constexpr unsigned int rows_per_block = ULLM_AQ4_REGISTER_ROWS_PER_BLOCK;
    constexpr unsigned int lanes_per_row = ULLM_AQ4_REGISTER_LANES_PER_ROW;
    const unsigned int tid = threadIdx.x;
    const unsigned int row_in_block = tid / lanes_per_row;
    const unsigned int lane = tid & (lanes_per_row - 1u);
    // gfx1201 uses wave32, so each wave contains four independent width-8 row subgroups.
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned long long batch_base = static_cast<unsigned long long>(blockIdx.y) * BM;
    float acc[BM] = {};

    if (row < rows && group_size == static_cast<unsigned long long>(GROUP_SIZE) &&
        (cols % static_cast<unsigned long long>(GROUP_SIZE)) == 0ull) {
        const unsigned long long groups_per_row =
            cols / static_cast<unsigned long long>(GROUP_SIZE);
        const unsigned long long row_offset = row * cols;
        for (unsigned long long group_in_row = lane; group_in_row < groups_per_row;
             group_in_row += lanes_per_row) {
            const unsigned long long group = row * groups_per_row + group_in_row;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index >= scale_count) {
                continue;
            }
            const float group_scale = scale_values[scale_index] * tensor_scale;
            const unsigned long long col_start =
                group_in_row * static_cast<unsigned long long>(GROUP_SIZE);
#pragma unroll
            for (unsigned int packed_offset = 0; packed_offset < GROUP_SIZE / 2u;
                 ++packed_offset) {
                const unsigned long long col = col_start + packed_offset * 2u;
                const unsigned char packed = indices[(row_offset + col) >> 1];
                const float weight0 = codebook[packed & 0x0f] * group_scale;
                const float weight1 = codebook[(packed >> 4) & 0x0f] * group_scale;
#pragma unroll
                for (unsigned int batch_offset = 0; batch_offset < BM; ++batch_offset) {
                    const unsigned long long batch = batch_base + batch_offset;
                    if (batch < batch_count) {
                        const float *batch_input = input + batch * cols;
                        acc[batch_offset] +=
                            weight0 * batch_input[col] + weight1 * batch_input[col + 1ull];
                    }
                }
            }
        }
    }

#pragma unroll
    for (unsigned int batch_offset = 0; batch_offset < BM; ++batch_offset) {
        acc[batch_offset] += __shfl_down(acc[batch_offset], 4u, lanes_per_row);
        acc[batch_offset] += __shfl_down(acc[batch_offset], 2u, lanes_per_row);
        acc[batch_offset] += __shfl_down(acc[batch_offset], 1u, lanes_per_row);
    }
    if (lane == 0u && row < rows) {
        const float row_scale =
            row_scales != nullptr && row < row_scale_count ? row_scales[row] : 1.0f;
#pragma unroll
        for (unsigned int batch_offset = 0; batch_offset < BM; ++batch_offset) {
            const unsigned long long batch = batch_base + batch_offset;
            if (batch < batch_count) {
                output[batch * rows + row] = acc[batch_offset] * row_scale;
            }
        }
    }
}

extern "C" __global__ void ullm_aq4_gemm_register_bm4_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long batch_count,
    float *output) {
    ullm_aq4_gemm_register_f32_impl<4u, 16u>(
        indices,
        scale_indices,
        codebook,
        scale_values,
        input,
        row_scales,
        scale_count,
        group_size,
        tensor_scale,
        row_scale_count,
        rows,
        cols,
        batch_count,
        output);
}

extern "C" __global__ void ullm_aq4_gemm_register_bm8_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long batch_count,
    float *output) {
    ullm_aq4_gemm_register_f32_impl<8u, 16u>(
        indices,
        scale_indices,
        codebook,
        scale_values,
        input,
        row_scales,
        scale_count,
        group_size,
        tensor_scale,
        row_scale_count,
        rows,
        cols,
        batch_count,
        output);
}

extern "C" __global__ void ullm_aq4_gemm_register_bm8_group8_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long batch_count,
    float *output) {
    ullm_aq4_gemm_register_f32_impl<8u, 8u>(
        indices,
        scale_indices,
        codebook,
        scale_values,
        input,
        row_scales,
        scale_count,
        group_size,
        tensor_scale,
        row_scale_count,
        rows,
        cols,
        batch_count,
        output);
}
)";
    }

    // Superseded v1 reference for the gfx1201 AQ4 WMMA group16 M=128 kernel. It is deliberately
    // not passed to HIPRTC: production uses the double-buffered source below under this same
    // entry-point ABI. Keep this source to make a controlled future rollback comparison possible.
    [[maybe_unused]] static const char *aq4_matvec_wmma_prototype_v1_reference_kernel_source() {
        return R"(
#include <rocwmma/rocwmma.hpp>

extern "C" __global__ void ullm_aq4_gemm_wmma_prototype_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long batch_count,
    float *output) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    using namespace rocwmma;
    constexpr unsigned int kTileRows = 16u;
    constexpr unsigned int kTileCols = 16u;
    constexpr unsigned int kWmmaK = 16u;
    constexpr unsigned int kWideK = 32u;
    constexpr unsigned int kBatchTile = 128u;
    constexpr unsigned int kWavesPerBlock = kBatchTile / kTileCols;
    constexpr unsigned int kThreads = kWavesPerBlock * 32u;
    static_assert(kThreads == 256u, "AQ4 WMMA prototype expects eight wave32s");
    static_assert(kWideK == 2u * kWmmaK, "AQ4 Wide-K path requires two WMMA K tiles");

    // `uint4` makes the paired group16 (32 AQ4 elements = 16 bytes) source load explicit.
    // Target geometry guarantees 16-byte alignment: the allocation base, every row stride,
    // and every k_base are all multiples of 16 bytes.
    typedef unsigned int ullm_aq4_wmma_uint4 __attribute__((ext_vector_type(4)));
    typedef float ullm_aq4_wmma_float4 __attribute__((ext_vector_type(4)));
    using FragA = fragment<matrix_a, kTileRows, kTileCols, kWmmaK, float16_t, row_major>;
    using FragB = fragment<matrix_b, kTileRows, kTileCols, kWmmaK, float16_t, col_major>;
    using FragC = fragment<accumulator, kTileRows, kTileCols, kWmmaK, float32_t, row_major>;

    __shared__ __attribute__((aligned(16))) float16_t weight_tile[kTileRows * kWideK];
    // B is [K, M] column-major. Each of the eight waves owns one adjacent 16-column slice.
    __shared__ __attribute__((aligned(16))) float16_t input_tile[kBatchTile * kWideK];
    // Store through rocWMMA's row-major fragment API, then transpose to the ABI's [batch][row]
    // order explicitly. This avoids relying on the documented raw-gfx12 WMMA lane mapping.
    __shared__ __attribute__((aligned(16))) float output_tile[
        kWavesPerBlock * kTileRows * kTileCols];

    const unsigned int tid = threadIdx.x;
    const unsigned int wave = tid >> 5;
    const unsigned long long row_base = static_cast<unsigned long long>(blockIdx.x) * kTileRows;
    const unsigned long long batch_base = static_cast<unsigned long long>(blockIdx.y) * kBatchTile;

    // Keep a uniform device-side guard as a fail-closed backstop if a caller bypasses host
    // validation. `cols % 32 == 0` preserves paired group16 Wide-K loads and 16-byte source
    // alignment; `rows % 16 == 0` gives every workgroup a complete output row tile.
    if (group_size != 16ull || batch_count != kBatchTile || rows == 0ull || cols == 0ull ||
        (rows % kTileRows) != 0ull || (cols % kWideK) != 0ull) {
        return;
    }

    FragC c;
    fill_fragment(c, 0.0f);
    const unsigned long long groups_per_row = cols / 16ull;

    // Wide-K: every iteration covers two adjacent group16s. One loader thread per output row
    // performs a 128-bit packed AQ4 load, decodes its two independent group scales to FP16 LDS,
    // then every wave consumes the same A tile with two K=16 WMMA instructions.
    for (unsigned long long k_base = 0ull; k_base < cols; k_base += kWideK) {
        if (tid < kTileRows) {
            const unsigned long long row = row_base + tid;
            const unsigned long long group_base = row * groups_per_row + k_base / 16ull;
            const unsigned int scale_index0 = static_cast<unsigned int>(scale_indices[group_base]);
            const unsigned int scale_index1 =
                static_cast<unsigned int>(scale_indices[group_base + 1ull]);
            // HIP-side metadata validation deliberately avoids a D2H barrier. Match the existing
            // register kernel's fail-closed handling of malformed scale indices.
            const float group_scale0 =
                scale_index0 < scale_count ? scale_values[scale_index0] : 0.0f;
            const float group_scale1 =
                scale_index1 < scale_count ? scale_values[scale_index1] : 0.0f;
            const unsigned long long byte_offset = (row * cols + k_base) >> 1;
            const ullm_aq4_wmma_uint4 packed_words =
                *reinterpret_cast<const ullm_aq4_wmma_uint4 *>(indices + byte_offset);
#pragma unroll
            for (unsigned int word_index = 0u; word_index < 4u; ++word_index) {
                const unsigned int word = static_cast<unsigned int>(packed_words[word_index]);
#pragma unroll
                for (unsigned int byte_index = 0u; byte_index < 4u; ++byte_index) {
                    const unsigned int packed_byte = (word >> (byte_index * 8u)) & 0xffu;
                    const unsigned int local_k = word_index * 8u + byte_index * 2u;
                    const float group_scale = local_k < 16u ? group_scale0 : group_scale1;
                    weight_tile[tid * kWideK + local_k] = static_cast<float16_t>(
                        codebook[packed_byte & 0x0fu] * group_scale);
                    weight_tile[tid * kWideK + local_k + 1u] = static_cast<float16_t>(
                        codebook[(packed_byte >> 4u) & 0x0fu] * group_scale);
                }
            }
        }

        // Convert the batch-major FP32 input to the logical B=[K, M] col-major FP16 tile.
        // Four adjacent K values are loaded at a time, so the activation path also uses 128-bit
        // global transactions under the exact M=128/K-multiple-of-32 prototype contract.
        for (unsigned int input_vector = tid;
             input_vector < (kBatchTile * kWideK) / 4u;
             input_vector += kThreads) {
            const unsigned int batch_in_tile = input_vector / (kWideK / 4u);
            const unsigned int k_in_tile = (input_vector % (kWideK / 4u)) * 4u;
            const unsigned long long input_offset =
                (batch_base + batch_in_tile) * cols + k_base + k_in_tile;
            const ullm_aq4_wmma_float4 values =
                *reinterpret_cast<const ullm_aq4_wmma_float4 *>(input + input_offset);
            const unsigned int shared_offset = batch_in_tile * kWideK + k_in_tile;
            input_tile[shared_offset + 0u] = static_cast<float16_t>(values[0]);
            input_tile[shared_offset + 1u] = static_cast<float16_t>(values[1]);
            input_tile[shared_offset + 2u] = static_cast<float16_t>(values[2]);
            input_tile[shared_offset + 3u] = static_cast<float16_t>(values[3]);
        }
        __syncthreads();

        FragA a0;
        FragA a1;
        FragB b0;
        FragB b1;
        load_matrix_sync(a0, weight_tile, kWideK);
        load_matrix_sync(a1, weight_tile + kWmmaK, kWideK);
        const unsigned int wave_input_offset = wave * kTileCols * kWideK;
        load_matrix_sync(b0, input_tile + wave_input_offset, kWideK);
        load_matrix_sync(b1, input_tile + wave_input_offset + kWmmaK, kWideK);
        mma_sync(c, a0, b0, c);
        mma_sync(c, a1, b1, c);
        __syncthreads();
    }

    store_matrix_sync(output_tile + wave * kTileRows * kTileCols, c, kTileCols);
    __syncthreads();

    for (unsigned int output_index = tid;
         output_index < kWavesPerBlock * kTileRows * kTileCols;
         output_index += kThreads) {
        const unsigned int output_wave = output_index / (kTileRows * kTileCols);
        const unsigned int output_in_wave = output_index % (kTileRows * kTileCols);
        const unsigned int local_row = output_in_wave / kTileCols;
        const unsigned int local_batch = output_in_wave % kTileCols;
        const unsigned long long row = row_base + local_row;
        const unsigned long long batch = batch_base + output_wave * kTileCols + local_batch;
        const float row_scale =
            row_scales != nullptr && row < row_scale_count ? row_scales[row] : 1.0f;
        output[batch * rows + row] = output_tile[output_index] * tensor_scale * row_scale;
    }
#else
    (void)indices;
    (void)scale_indices;
    (void)codebook;
    (void)scale_values;
    (void)input;
    (void)row_scales;
    (void)scale_count;
    (void)group_size;
    (void)tensor_scale;
    (void)row_scale_count;
    (void)rows;
    (void)cols;
    (void)batch_count;
    (void)output;
#endif
}
)";
    }

    // Superseded v2 reference for the gfx1201 AQ4 WMMA group16 M=128 kernel. It is deliberately
    // not passed to HIPRTC: production uses the direct-output source below under this same
    // entry-point ABI. Keep this source to make a controlled future rollback comparison possible.
    [[maybe_unused]] static const char *aq4_matvec_wmma_prototype_v2_reference_kernel_source() {
        return R"(
#include <rocwmma/rocwmma.hpp>

extern "C" __global__ void ullm_aq4_gemm_wmma_prototype_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long batch_count,
    float *output) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    using namespace rocwmma;
    constexpr unsigned int kTileRows = 16u;
    constexpr unsigned int kTileCols = 16u;
    constexpr unsigned int kWmmaK = 16u;
    constexpr unsigned int kWideK = 32u;
    constexpr unsigned int kBatchTile = 128u;
    constexpr unsigned int kStages = 2u;
    constexpr unsigned int kWavesPerBlock = kBatchTile / kTileCols;
    constexpr unsigned int kThreads = kWavesPerBlock * 32u;
    static_assert(kThreads == 256u, "AQ4 WMMA production kernel expects eight wave32s");
    static_assert(kWideK == 2u * kWmmaK, "AQ4 Wide-K path requires two WMMA K tiles");
    static_assert(kStages == 2u, "AQ4 WMMA production kernel is a ping-pong LDS pipeline");

    typedef unsigned int ullm_aq4_wmma_uint4 __attribute__((ext_vector_type(4)));
    typedef float ullm_aq4_wmma_float4 __attribute__((ext_vector_type(4)));
    using FragA = fragment<matrix_a, kTileRows, kTileCols, kWmmaK, float16_t, row_major>;
    using FragB = fragment<matrix_b, kTileRows, kTileCols, kWmmaK, float16_t, col_major>;
    using FragC = fragment<accumulator, kTileRows, kTileCols, kWmmaK, float32_t, row_major>;

    // This allocation is per CTA, so it is unchanged for the 32x4096 linear_attn_a/b
    // geometry: that launch has exactly two row CTAs, each with the same ping-pong stages.
    constexpr unsigned int kWeightTilesLdsBytes =
        kStages * kTileRows * kWideK * sizeof(float16_t);
    constexpr unsigned int kInputTilesLdsBytes =
        kStages * kBatchTile * kWideK * sizeof(float16_t);
    constexpr unsigned int kOutputTileLdsBytes =
        kWavesPerBlock * kTileRows * kTileCols * sizeof(float);
    constexpr unsigned int kStaticLdsBytes =
        kWeightTilesLdsBytes + kInputTilesLdsBytes + kOutputTileLdsBytes;
    static_assert(kStaticLdsBytes == 26624u, "AQ4 WMMA production LDS accounting drifted");
    static_assert(kStaticLdsBytes <= 64u * 1024u, "AQ4 WMMA production kernel must fit gfx1201 LDS");

    // The production kernel has two ping-pong LDS stages. A wave first copies the current stage
    // into fragments, then prefetches its next input vectors to registers while its current WMMA
    // work executes.
    // The next LDS stage is only published at the transition barrier, so no stage is overwritten
    // while another wave can still be reading it.
    __shared__ __attribute__((aligned(16))) float16_t weight_tiles[
        kStages][kTileRows * kWideK];
    __shared__ __attribute__((aligned(16))) float16_t input_tiles[
        kStages][kBatchTile * kWideK];
    __shared__ __attribute__((aligned(16))) float output_tile[
        kWavesPerBlock * kTileRows * kTileCols];

    const unsigned int tid = threadIdx.x;
    const unsigned int wave = tid >> 5;
    const unsigned long long row_base = static_cast<unsigned long long>(blockIdx.x) * kTileRows;
    const unsigned long long batch_base = static_cast<unsigned long long>(blockIdx.y) * kBatchTile;

    if (group_size != 16ull || batch_count != kBatchTile || rows == 0ull || cols == 0ull ||
        (rows % kTileRows) != 0ull || (cols % kWideK) != 0ull) {
        return;
    }

    FragC c;
    fill_fragment(c, 0.0f);
    const unsigned long long groups_per_row = cols / 16ull;

    // Prime stage 0. From this point on, every nonfinal K iteration prepares
    // the other stage before the next iteration consumes it.
    if (tid < kTileRows) {
        const unsigned long long row = row_base + tid;
        const unsigned long long group_base = row * groups_per_row;
        const unsigned int scale_index0 = static_cast<unsigned int>(scale_indices[group_base]);
        const unsigned int scale_index1 = static_cast<unsigned int>(scale_indices[group_base + 1ull]);
        const float group_scale0 =
            scale_index0 < scale_count ? scale_values[scale_index0] : 0.0f;
        const float group_scale1 =
            scale_index1 < scale_count ? scale_values[scale_index1] : 0.0f;
        const unsigned long long byte_offset = (row * cols) >> 1;
        const ullm_aq4_wmma_uint4 packed_words =
            *reinterpret_cast<const ullm_aq4_wmma_uint4 *>(indices + byte_offset);
#pragma unroll
        for (unsigned int word_index = 0u; word_index < 4u; ++word_index) {
            const unsigned int word = static_cast<unsigned int>(packed_words[word_index]);
#pragma unroll
            for (unsigned int byte_index = 0u; byte_index < 4u; ++byte_index) {
                const unsigned int packed_byte = (word >> (byte_index * 8u)) & 0xffu;
                const unsigned int local_k = word_index * 8u + byte_index * 2u;
                const float group_scale = local_k < 16u ? group_scale0 : group_scale1;
                weight_tiles[0][tid * kWideK + local_k] = static_cast<float16_t>(
                    codebook[packed_byte & 0x0fu] * group_scale);
                weight_tiles[0][tid * kWideK + local_k + 1u] = static_cast<float16_t>(
                    codebook[(packed_byte >> 4u) & 0x0fu] * group_scale);
            }
        }
    }
    for (unsigned int input_vector = tid;
         input_vector < (kBatchTile * kWideK) / 4u;
         input_vector += kThreads) {
        const unsigned int batch_in_tile = input_vector / (kWideK / 4u);
        const unsigned int k_in_tile = (input_vector % (kWideK / 4u)) * 4u;
        const unsigned long long input_offset =
            (batch_base + batch_in_tile) * cols + k_in_tile;
        const ullm_aq4_wmma_float4 values =
            *reinterpret_cast<const ullm_aq4_wmma_float4 *>(input + input_offset);
        const unsigned int shared_offset = batch_in_tile * kWideK + k_in_tile;
        input_tiles[0][shared_offset + 0u] = static_cast<float16_t>(values[0]);
        input_tiles[0][shared_offset + 1u] = static_cast<float16_t>(values[1]);
        input_tiles[0][shared_offset + 2u] = static_cast<float16_t>(values[2]);
        input_tiles[0][shared_offset + 3u] = static_cast<float16_t>(values[3]);
    }
    __syncthreads();

    for (unsigned long long k_base = 0ull; k_base < cols; k_base += kWideK) {
        const unsigned int current_stage =
            static_cast<unsigned int>((k_base / kWideK) & 1ull);
        const unsigned int next_stage = current_stage ^ 1u;
        const unsigned int wave_input_offset = wave * kTileCols * kWideK;

        // Pull the first current A/B pair into registers, then start the next global reads before
        // the current two matrix operations. The second current pair is still read from the
        // untouched current stage below; prefetched values are not consumed until their stores,
        // letting the scheduler cover part of their latency with current-stage WMMA work.
        FragA a;
        FragB b;
        load_matrix_sync(a, weight_tiles[current_stage], kWideK);
        load_matrix_sync(b, input_tiles[current_stage] + wave_input_offset, kWideK);

        const bool has_next = k_base < cols - kWideK;
        ullm_aq4_wmma_float4 prefetched_input[4];
        ullm_aq4_wmma_uint4 prefetched_weights;
        float prefetched_group_scale0 = 0.0f;
        float prefetched_group_scale1 = 0.0f;
        if (has_next && tid < kTileRows) {
            const unsigned long long next_k_base = k_base + kWideK;
            const unsigned long long row = row_base + tid;
            const unsigned long long group_base =
                row * groups_per_row + next_k_base / 16ull;
            const unsigned int scale_index0 = static_cast<unsigned int>(scale_indices[group_base]);
            const unsigned int scale_index1 =
                static_cast<unsigned int>(scale_indices[group_base + 1ull]);
            prefetched_group_scale0 =
                scale_index0 < scale_count ? scale_values[scale_index0] : 0.0f;
            prefetched_group_scale1 =
                scale_index1 < scale_count ? scale_values[scale_index1] : 0.0f;
            const unsigned long long byte_offset = (row * cols + next_k_base) >> 1;
            prefetched_weights =
                *reinterpret_cast<const ullm_aq4_wmma_uint4 *>(indices + byte_offset);
        }
        if (has_next) {
#pragma unroll
            for (unsigned int prefetch_index = 0u; prefetch_index < 4u; ++prefetch_index) {
                const unsigned int input_vector = tid + prefetch_index * kThreads;
                const unsigned int batch_in_tile = input_vector / (kWideK / 4u);
                const unsigned int k_in_tile = (input_vector % (kWideK / 4u)) * 4u;
                const unsigned long long input_offset =
                    (batch_base + batch_in_tile) * cols + k_base + kWideK + k_in_tile;
                prefetched_input[prefetch_index] =
                    *reinterpret_cast<const ullm_aq4_wmma_float4 *>(input + input_offset);
            }
        }

        mma_sync(c, a, b, c);
        load_matrix_sync(a, weight_tiles[current_stage] + kWmmaK, kWideK);
        load_matrix_sync(b, input_tiles[current_stage] + wave_input_offset + kWmmaK, kWideK);
        mma_sync(c, a, b, c);

        if (has_next) {
            if (tid < kTileRows) {
#pragma unroll
                for (unsigned int word_index = 0u; word_index < 4u; ++word_index) {
                    const unsigned int word = static_cast<unsigned int>(prefetched_weights[word_index]);
#pragma unroll
                    for (unsigned int byte_index = 0u; byte_index < 4u; ++byte_index) {
                        const unsigned int packed_byte = (word >> (byte_index * 8u)) & 0xffu;
                        const unsigned int local_k = word_index * 8u + byte_index * 2u;
                        const float group_scale =
                            local_k < 16u ? prefetched_group_scale0 : prefetched_group_scale1;
                        weight_tiles[next_stage][tid * kWideK + local_k] =
                            static_cast<float16_t>(codebook[packed_byte & 0x0fu] * group_scale);
                        weight_tiles[next_stage][tid * kWideK + local_k + 1u] =
                            static_cast<float16_t>(
                                codebook[(packed_byte >> 4u) & 0x0fu] * group_scale);
                    }
                }
            }
#pragma unroll
            for (unsigned int prefetch_index = 0u; prefetch_index < 4u; ++prefetch_index) {
                const unsigned int input_vector = tid + prefetch_index * kThreads;
                const unsigned int batch_in_tile = input_vector / (kWideK / 4u);
                const unsigned int k_in_tile = (input_vector % (kWideK / 4u)) * 4u;
                const unsigned int shared_offset = batch_in_tile * kWideK + k_in_tile;
                const ullm_aq4_wmma_float4 values = prefetched_input[prefetch_index];
                input_tiles[next_stage][shared_offset + 0u] = static_cast<float16_t>(values[0]);
                input_tiles[next_stage][shared_offset + 1u] = static_cast<float16_t>(values[1]);
                input_tiles[next_stage][shared_offset + 2u] = static_cast<float16_t>(values[2]);
                input_tiles[next_stage][shared_offset + 3u] = static_cast<float16_t>(values[3]);
            }
            // This is both the producer/consumer hand-off for the next stage and proof that all
            // waves have finished using the stage that will be overwritten on the next pass.
            __syncthreads();
        }
    }

    store_matrix_sync(output_tile + wave * kTileRows * kTileCols, c, kTileCols);
    __syncthreads();

    for (unsigned int output_index = tid;
         output_index < kWavesPerBlock * kTileRows * kTileCols;
         output_index += kThreads) {
        const unsigned int output_wave = output_index / (kTileRows * kTileCols);
        const unsigned int output_in_wave = output_index % (kTileRows * kTileCols);
        const unsigned int local_row = output_in_wave / kTileCols;
        const unsigned int local_batch = output_in_wave % kTileCols;
        const unsigned long long row = row_base + local_row;
        const unsigned long long batch = batch_base + output_wave * kTileCols + local_batch;
        const float row_scale =
            row_scales != nullptr && row < row_scale_count ? row_scales[row] : 1.0f;
        output[batch * rows + row] = output_tile[output_index] * tensor_scale * row_scale;
    }
#else
    (void)indices;
    (void)scale_indices;
    (void)codebook;
    (void)scale_values;
    (void)input;
    (void)row_scales;
    (void)scale_count;
    (void)group_size;
    (void)tensor_scale;
    (void)row_scale_count;
    (void)rows;
    (void)cols;
    (void)batch_count;
    (void)output;
#endif
}
)";
    }

    // Production direct-output gfx1201 AQ4 WMMA path for group16 M=128 projections. It retains
    // v2's double-buffered Wide-K pipeline, but rocWMMA stores C directly in the column-major
    // final-output layout rather than through an 8 KiB LDS staging tile. This lowers the CTA LDS
    // footprint from 26,624 B to 18,432 B, allowing three 8-wave CTAs per gfx1201 CU.
    static std::string aq4_matvec_wmma_prototype_kernel_source() {
        std::string source(aq4_matvec_wmma_prototype_v2_reference_kernel_source());
        const auto replace_once = [&source](const char *from, const char *to) {
            const size_t position = source.find(from);
            if (position == std::string::npos) {
                return false;
            }
            source.replace(position, std::strlen(from), to);
            return true;
        };
        const bool replaced =
            replace_once(
                "using FragC = fragment<accumulator, kTileRows, kTileCols, kWmmaK, float32_t, row_major>;",
                "using FragC = fragment<accumulator, kTileRows, kTileCols, kWmmaK, float32_t, col_major>;") &&
            replace_once(
                "constexpr unsigned int kOutputTileLdsBytes =\n"
                "        kWavesPerBlock * kTileRows * kTileCols * sizeof(float);",
                "constexpr unsigned int kOutputTileLdsBytes = 0u;") &&
            replace_once("kStaticLdsBytes == 26624u", "kStaticLdsBytes == 18432u") &&
            replace_once(
                "    __shared__ __attribute__((aligned(16))) float output_tile[\n"
                "        kWavesPerBlock * kTileRows * kTileCols];\n",
                "") &&
            replace_once(
                "store_matrix_sync(output_tile + wave * kTileRows * kTileCols, c, kTileCols);",
                "store_matrix_sync(output + (batch_base + wave * kTileCols) * rows + row_base, c, rows);") &&
            replace_once(
                "output[batch * rows + row] = output_tile[output_index] * tensor_scale * row_scale;",
                "output[batch * rows + row] = output[batch * rows + row] * tensor_scale * row_scale;");
        if (!replaced) {
            return "";
        }
        return source;
    }

    // Experimental ragged-M variant of the group16 WMMA kernel.  This deliberately starts from
    // the v2 source rather than the production direct-output v4 transformation: rocWMMA's direct
    // store_matrix_sync cannot be guarded per batch row.  Keeping v2's 8 KiB output LDS tile lets
    // the final scalar store skip invalid rows, while preserving the proven two-stage Wide-K
    // input/weight pipeline unchanged.  It is a separate source/module from production.
    static std::string aq4_matvec_wmma_ragged_m_prototype_kernel_source() {
        std::string source(aq4_matvec_wmma_prototype_v2_reference_kernel_source());
        const auto replace_once = [&source](const char *from, const char *to) {
            const size_t position = source.find(from);
            if (position == std::string::npos) {
                return false;
            }
            source.replace(position, std::strlen(from), to);
            return true;
        };
        const bool replaced =
            replace_once(
                "ullm_aq4_gemm_wmma_prototype_f32_kernel",
                "ullm_aq4_gemm_wmma_ragged_m_prototype_f32_kernel") &&
            replace_once(
                "unsigned long long batch_count,\n"
                "    float *output)",
                "unsigned long long m_actual,\n"
                "    float *output)") &&
            replace_once(
                "if (group_size != 16ull || batch_count != kBatchTile || rows == 0ull || cols == 0ull ||\n"
                "        (rows % kTileRows) != 0ull || (cols % kWideK) != 0ull) {",
                "if (group_size != 16ull || m_actual == 0ull || m_actual > kBatchTile ||\n"
                "        rows == 0ull || cols == 0ull || (rows % kTileRows) != 0ull ||\n"
                "        (cols % kWideK) != 0ull) {") &&
            replace_once(
                "        const unsigned long long input_offset =\n"
                "            (batch_base + batch_in_tile) * cols + k_in_tile;\n"
                "        const ullm_aq4_wmma_float4 values =\n"
                "            *reinterpret_cast<const ullm_aq4_wmma_float4 *>(input + input_offset);\n"
                "        const unsigned int shared_offset = batch_in_tile * kWideK + k_in_tile;\n"
                "        input_tiles[0][shared_offset + 0u] = static_cast<float16_t>(values[0]);\n"
                "        input_tiles[0][shared_offset + 1u] = static_cast<float16_t>(values[1]);\n"
                "        input_tiles[0][shared_offset + 2u] = static_cast<float16_t>(values[2]);\n"
                "        input_tiles[0][shared_offset + 3u] = static_cast<float16_t>(values[3]);",
                "        const unsigned int shared_offset = batch_in_tile * kWideK + k_in_tile;\n"
                "        // Do not form an input address for an invalid physical row.\n"
                "        if (batch_base + batch_in_tile < m_actual) {\n"
                "            const unsigned long long input_offset =\n"
                "                (batch_base + batch_in_tile) * cols + k_in_tile;\n"
                "            const ullm_aq4_wmma_float4 values =\n"
                "                *reinterpret_cast<const ullm_aq4_wmma_float4 *>(input + input_offset);\n"
                "            input_tiles[0][shared_offset + 0u] = static_cast<float16_t>(values[0]);\n"
                "            input_tiles[0][shared_offset + 1u] = static_cast<float16_t>(values[1]);\n"
                "            input_tiles[0][shared_offset + 2u] = static_cast<float16_t>(values[2]);\n"
                "            input_tiles[0][shared_offset + 3u] = static_cast<float16_t>(values[3]);\n"
                "        } else {\n"
                "            input_tiles[0][shared_offset + 0u] = static_cast<float16_t>(0.0f);\n"
                "            input_tiles[0][shared_offset + 1u] = static_cast<float16_t>(0.0f);\n"
                "            input_tiles[0][shared_offset + 2u] = static_cast<float16_t>(0.0f);\n"
                "            input_tiles[0][shared_offset + 3u] = static_cast<float16_t>(0.0f);\n"
                "        }") &&
            replace_once(
                "                const unsigned long long input_offset =\n"
                "                    (batch_base + batch_in_tile) * cols + k_base + kWideK + k_in_tile;\n"
                "                prefetched_input[prefetch_index] =\n"
                "                    *reinterpret_cast<const ullm_aq4_wmma_float4 *>(input + input_offset);",
                "                // The steady-state prefetch follows the same rule as stage 0:\n"
                "                // no invalid-row pointer arithmetic and no invalid global load.\n"
                "                if (batch_base + batch_in_tile < m_actual) {\n"
                "                    const unsigned long long input_offset =\n"
                "                        (batch_base + batch_in_tile) * cols + k_base + kWideK + k_in_tile;\n"
                "                    prefetched_input[prefetch_index] =\n"
                "                        *reinterpret_cast<const ullm_aq4_wmma_float4 *>(input + input_offset);\n"
                "                }") &&
            replace_once(
                "                const unsigned int shared_offset = batch_in_tile * kWideK + k_in_tile;\n"
                "                const ullm_aq4_wmma_float4 values = prefetched_input[prefetch_index];\n"
                "                input_tiles[next_stage][shared_offset + 0u] = static_cast<float16_t>(values[0]);\n"
                "                input_tiles[next_stage][shared_offset + 1u] = static_cast<float16_t>(values[1]);\n"
                "                input_tiles[next_stage][shared_offset + 2u] = static_cast<float16_t>(values[2]);\n"
                "                input_tiles[next_stage][shared_offset + 3u] = static_cast<float16_t>(values[3]);",
                "                const unsigned int shared_offset = batch_in_tile * kWideK + k_in_tile;\n"
                "                if (batch_base + batch_in_tile < m_actual) {\n"
                "                    const ullm_aq4_wmma_float4 values = prefetched_input[prefetch_index];\n"
                "                    input_tiles[next_stage][shared_offset + 0u] = static_cast<float16_t>(values[0]);\n"
                "                    input_tiles[next_stage][shared_offset + 1u] = static_cast<float16_t>(values[1]);\n"
                "                    input_tiles[next_stage][shared_offset + 2u] = static_cast<float16_t>(values[2]);\n"
                "                    input_tiles[next_stage][shared_offset + 3u] = static_cast<float16_t>(values[3]);\n"
                "                } else {\n"
                "                    input_tiles[next_stage][shared_offset + 0u] = static_cast<float16_t>(0.0f);\n"
                "                    input_tiles[next_stage][shared_offset + 1u] = static_cast<float16_t>(0.0f);\n"
                "                    input_tiles[next_stage][shared_offset + 2u] = static_cast<float16_t>(0.0f);\n"
                "                    input_tiles[next_stage][shared_offset + 3u] = static_cast<float16_t>(0.0f);\n"
                "                }") &&
            replace_once(
                "        const float row_scale =\n"
                "            row_scales != nullptr && row < row_scale_count ? row_scales[row] : 1.0f;\n"
                "        output[batch * rows + row] = output_tile[output_index] * tensor_scale * row_scale;",
                "        if (batch < m_actual) {\n"
                "            const float row_scale =\n"
                "                row_scales != nullptr && row < row_scale_count ? row_scales[row] : 1.0f;\n"
                "            output[batch * rows + row] = output_tile[output_index] * tensor_scale * row_scale;\n"
                "        }") &&
            replace_once("    (void)batch_count;", "    (void)m_actual;");
        if (!replaced) {
            return "";
        }
        return source;
    }

    // Experimental v3 of the gfx1201 AQ4 WMMA path. It keeps v2's two-stage asynchronous
    // prefetch discipline, but each stage owns two physically independent Wide-K microtiles.
    // That permits four WMMA operations between producer/consumer barriers without changing
    // the v2 per-wave prefetch cohort or the proven 32-half LDS leading dimensions.
    static const char *aq4_matvec_wmma_prototype_v3_kernel_source() {
        return R"(
#include <rocwmma/rocwmma.hpp>

extern "C" __global__ void ullm_aq4_gemm_wmma_prototype_v3_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long batch_count,
    float *output) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    using namespace rocwmma;
    constexpr unsigned int kTileRows = 16u;
    constexpr unsigned int kTileCols = 16u;
    constexpr unsigned int kWmmaK = 16u;
    constexpr unsigned int kWideK = 32u;
    constexpr unsigned int kWideKsPerStage = 2u;
    constexpr unsigned int kStageK = kWideKsPerStage * kWideK;
    constexpr unsigned int kBatchTile = 128u;
    constexpr unsigned int kStages = 2u;
    constexpr unsigned int kWavesPerBlock = kBatchTile / kTileCols;
    constexpr unsigned int kThreads = kWavesPerBlock * 32u;
    static_assert(kThreads == 256u, "AQ4 WMMA prototype v3 expects eight wave32s");
    static_assert(kWideK == 2u * kWmmaK, "AQ4 Wide-K path requires two WMMA K tiles");
    static_assert(kWideKsPerStage == 2u, "AQ4 WMMA prototype v3 groups two Wide-K microtiles");
    static_assert(kStages == 2u, "AQ4 WMMA prototype v3 is a ping-pong LDS pipeline");

    typedef unsigned int ullm_aq4_wmma_v3_uint4 __attribute__((ext_vector_type(4)));
    typedef float ullm_aq4_wmma_v3_float4 __attribute__((ext_vector_type(4)));
    using FragA = fragment<matrix_a, kTileRows, kTileCols, kWmmaK, float16_t, row_major>;
    using FragB = fragment<matrix_b, kTileRows, kTileCols, kWmmaK, float16_t, col_major>;
    using FragC = fragment<accumulator, kTileRows, kTileCols, kWmmaK, float32_t, row_major>;

    // A v3 stage is two separate 32-K slabs rather than a 64-K matrix with a changed leading
    // dimension. Keeping each slab's ldm at 32 preserves v2's rocWMMA/LDS access layout while
    // doubling the useful WMMA work protected by one transition barrier.
    constexpr unsigned int kWeightTilesLdsBytes =
        kStages * kWideKsPerStage * kTileRows * kWideK * sizeof(float16_t);
    constexpr unsigned int kInputTilesLdsBytes =
        kStages * kWideKsPerStage * kBatchTile * kWideK * sizeof(float16_t);
    constexpr unsigned int kOutputTileLdsBytes =
        kWavesPerBlock * kTileRows * kTileCols * sizeof(float);
    constexpr unsigned int kStaticLdsBytes =
        kWeightTilesLdsBytes + kInputTilesLdsBytes + kOutputTileLdsBytes;
    static_assert(kStaticLdsBytes == 45056u, "AQ4 WMMA prototype v3 LDS accounting drifted");
    static_assert(
        kStaticLdsBytes <= 64u * 1024u,
        "AQ4 WMMA prototype v3 kernel must fit gfx1201 LDS");

    // The two physical Wide-K slabs in each stage let the next 64 K values be published with a
    // single barrier. Each pass below only holds one 32-K prefetch cohort in VGPRs, matching v2.
    __shared__ __attribute__((aligned(16))) float16_t weight_tiles[
        kStages][kWideKsPerStage][kTileRows * kWideK];
    __shared__ __attribute__((aligned(16))) float16_t input_tiles[
        kStages][kWideKsPerStage][kBatchTile * kWideK];
    __shared__ __attribute__((aligned(16))) float output_tile[
        kWavesPerBlock * kTileRows * kTileCols];

    const unsigned int tid = threadIdx.x;
    const unsigned int wave = tid >> 5;
    const unsigned long long row_base = static_cast<unsigned long long>(blockIdx.x) * kTileRows;
    const unsigned long long batch_base =
        static_cast<unsigned long long>(blockIdx.y) * kBatchTile;

    // V3 deliberately narrows its direct-only contract from v2's K%32 set to K%64. All five
    // production group16 M=128 projections meet this condition, while a partial stage would
    // otherwise leave one of the four WMMA calls reading unpublished LDS.
    if (group_size != 16ull || batch_count != kBatchTile || rows == 0ull || cols == 0ull ||
        (rows % kTileRows) != 0ull || (cols % kStageK) != 0ull) {
        return;
    }

    FragC c;
    fill_fragment(c, 0.0f);
    const unsigned long long groups_per_row = cols / 16ull;

    // Prime both physical slabs of stage 0. This start-up work is intentionally direct; every
    // subsequent stage is register-prefetched while the previous slab's two WMMA calls execute.
    for (unsigned int stage_wide_k = 0u; stage_wide_k < kWideKsPerStage; ++stage_wide_k) {
        const unsigned long long stage_k_base =
            static_cast<unsigned long long>(stage_wide_k) * kWideK;
        if (tid < kTileRows) {
            const unsigned long long row = row_base + tid;
            const unsigned long long group_base =
                row * groups_per_row + stage_k_base / 16ull;
            const unsigned int scale_index0 = static_cast<unsigned int>(scale_indices[group_base]);
            const unsigned int scale_index1 =
                static_cast<unsigned int>(scale_indices[group_base + 1ull]);
            const float group_scale0 =
                scale_index0 < scale_count ? scale_values[scale_index0] : 0.0f;
            const float group_scale1 =
                scale_index1 < scale_count ? scale_values[scale_index1] : 0.0f;
            const unsigned long long byte_offset = (row * cols + stage_k_base) >> 1;
            const ullm_aq4_wmma_v3_uint4 packed_words =
                *reinterpret_cast<const ullm_aq4_wmma_v3_uint4 *>(indices + byte_offset);
#pragma unroll
            for (unsigned int word_index = 0u; word_index < 4u; ++word_index) {
                const unsigned int word = static_cast<unsigned int>(packed_words[word_index]);
#pragma unroll
                for (unsigned int byte_index = 0u; byte_index < 4u; ++byte_index) {
                    const unsigned int packed_byte = (word >> (byte_index * 8u)) & 0xffu;
                    const unsigned int local_k = word_index * 8u + byte_index * 2u;
                    const float group_scale = local_k < 16u ? group_scale0 : group_scale1;
                    weight_tiles[0][stage_wide_k][tid * kWideK + local_k] =
                        static_cast<float16_t>(codebook[packed_byte & 0x0fu] * group_scale);
                    weight_tiles[0][stage_wide_k][tid * kWideK + local_k + 1u] =
                        static_cast<float16_t>(
                            codebook[(packed_byte >> 4u) & 0x0fu] * group_scale);
                }
            }
        }
        for (unsigned int input_vector = tid;
             input_vector < (kBatchTile * kWideK) / 4u;
             input_vector += kThreads) {
            const unsigned int batch_in_tile = input_vector / (kWideK / 4u);
            const unsigned int k_in_tile = (input_vector % (kWideK / 4u)) * 4u;
            const unsigned long long input_offset =
                (batch_base + batch_in_tile) * cols + stage_k_base + k_in_tile;
            const ullm_aq4_wmma_v3_float4 values =
                *reinterpret_cast<const ullm_aq4_wmma_v3_float4 *>(input + input_offset);
            const unsigned int shared_offset = batch_in_tile * kWideK + k_in_tile;
            input_tiles[0][stage_wide_k][shared_offset + 0u] =
                static_cast<float16_t>(values[0]);
            input_tiles[0][stage_wide_k][shared_offset + 1u] =
                static_cast<float16_t>(values[1]);
            input_tiles[0][stage_wide_k][shared_offset + 2u] =
                static_cast<float16_t>(values[2]);
            input_tiles[0][stage_wide_k][shared_offset + 3u] =
                static_cast<float16_t>(values[3]);
        }
    }
    __syncthreads();

    // One v3 iteration consumes two Wide-K slabs = four K=16 WMMA operations per wave. It
    // prefetches and stores one next-stage slab at a time, so the array of four float4 values has
    // the same lifetime and size as v2 rather than retaining a K+2 cohort in registers.
    for (unsigned long long k_base = 0ull; k_base < cols; k_base += kStageK) {
        const unsigned int current_stage =
            static_cast<unsigned int>((k_base / kStageK) & 1ull);
        const unsigned int next_stage = current_stage ^ 1u;
        const unsigned int wave_input_offset = wave * kTileCols * kWideK;
        const bool has_next = k_base < cols - kStageK;

#pragma unroll 1
        for (unsigned int stage_wide_k = 0u; stage_wide_k < kWideKsPerStage; ++stage_wide_k) {
            const unsigned long long stage_k_offset =
                static_cast<unsigned long long>(stage_wide_k) * kWideK;
            FragA a;
            FragB b;
            load_matrix_sync(a, weight_tiles[current_stage][stage_wide_k], kWideK);
            load_matrix_sync(
                b,
                input_tiles[current_stage][stage_wide_k] + wave_input_offset,
                kWideK);

            ullm_aq4_wmma_v3_float4 prefetched_input[4];
            ullm_aq4_wmma_v3_uint4 prefetched_weights;
            float prefetched_group_scale0 = 0.0f;
            float prefetched_group_scale1 = 0.0f;
            if (has_next && tid < kTileRows) {
                const unsigned long long next_k_base = k_base + kStageK + stage_k_offset;
                const unsigned long long row = row_base + tid;
                const unsigned long long group_base =
                    row * groups_per_row + next_k_base / 16ull;
                const unsigned int scale_index0 =
                    static_cast<unsigned int>(scale_indices[group_base]);
                const unsigned int scale_index1 =
                    static_cast<unsigned int>(scale_indices[group_base + 1ull]);
                prefetched_group_scale0 =
                    scale_index0 < scale_count ? scale_values[scale_index0] : 0.0f;
                prefetched_group_scale1 =
                    scale_index1 < scale_count ? scale_values[scale_index1] : 0.0f;
                const unsigned long long byte_offset = (row * cols + next_k_base) >> 1;
                prefetched_weights =
                    *reinterpret_cast<const ullm_aq4_wmma_v3_uint4 *>(indices + byte_offset);
            }
            if (has_next) {
#pragma unroll
                for (unsigned int prefetch_index = 0u; prefetch_index < 4u; ++prefetch_index) {
                    const unsigned int input_vector = tid + prefetch_index * kThreads;
                    const unsigned int batch_in_tile = input_vector / (kWideK / 4u);
                    const unsigned int k_in_tile = (input_vector % (kWideK / 4u)) * 4u;
                    const unsigned long long input_offset =
                        (batch_base + batch_in_tile) * cols + k_base + kStageK +
                        stage_k_offset + k_in_tile;
                    prefetched_input[prefetch_index] =
                        *reinterpret_cast<const ullm_aq4_wmma_v3_float4 *>(input + input_offset);
                }
            }

            mma_sync(c, a, b, c);
            load_matrix_sync(
                a,
                weight_tiles[current_stage][stage_wide_k] + kWmmaK,
                kWideK);
            load_matrix_sync(
                b,
                input_tiles[current_stage][stage_wide_k] + wave_input_offset + kWmmaK,
                kWideK);
            mma_sync(c, a, b, c);

            if (has_next) {
                if (tid < kTileRows) {
#pragma unroll
                    for (unsigned int word_index = 0u; word_index < 4u; ++word_index) {
                        const unsigned int word =
                            static_cast<unsigned int>(prefetched_weights[word_index]);
#pragma unroll
                        for (unsigned int byte_index = 0u; byte_index < 4u; ++byte_index) {
                            const unsigned int packed_byte =
                                (word >> (byte_index * 8u)) & 0xffu;
                            const unsigned int local_k = word_index * 8u + byte_index * 2u;
                            const float group_scale =
                                local_k < 16u ? prefetched_group_scale0 : prefetched_group_scale1;
                            weight_tiles[next_stage][stage_wide_k][tid * kWideK + local_k] =
                                static_cast<float16_t>(
                                    codebook[packed_byte & 0x0fu] * group_scale);
                            weight_tiles[next_stage][stage_wide_k][tid * kWideK + local_k + 1u] =
                                static_cast<float16_t>(
                                    codebook[(packed_byte >> 4u) & 0x0fu] * group_scale);
                        }
                    }
                }
#pragma unroll
                for (unsigned int prefetch_index = 0u; prefetch_index < 4u; ++prefetch_index) {
                    const unsigned int input_vector = tid + prefetch_index * kThreads;
                    const unsigned int batch_in_tile = input_vector / (kWideK / 4u);
                    const unsigned int k_in_tile = (input_vector % (kWideK / 4u)) * 4u;
                    const unsigned int shared_offset = batch_in_tile * kWideK + k_in_tile;
                    const ullm_aq4_wmma_v3_float4 values = prefetched_input[prefetch_index];
                    input_tiles[next_stage][stage_wide_k][shared_offset + 0u] =
                        static_cast<float16_t>(values[0]);
                    input_tiles[next_stage][stage_wide_k][shared_offset + 1u] =
                        static_cast<float16_t>(values[1]);
                    input_tiles[next_stage][stage_wide_k][shared_offset + 2u] =
                        static_cast<float16_t>(values[2]);
                    input_tiles[next_stage][stage_wide_k][shared_offset + 3u] =
                        static_cast<float16_t>(values[3]);
                }
            }
        }

        if (has_next) {
            // One hand-off covers both next-stage Wide-K slabs. It both publishes their LDS
            // stores and proves that all waves are done with the stage that will be overwritten
            // after the next 64-K phase.
            __syncthreads();
        }
    }

    store_matrix_sync(output_tile + wave * kTileRows * kTileCols, c, kTileCols);
    __syncthreads();

    for (unsigned int output_index = tid;
         output_index < kWavesPerBlock * kTileRows * kTileCols;
         output_index += kThreads) {
        const unsigned int output_wave = output_index / (kTileRows * kTileCols);
        const unsigned int output_in_wave = output_index % (kTileRows * kTileCols);
        const unsigned int local_row = output_in_wave / kTileCols;
        const unsigned int local_batch = output_in_wave % kTileCols;
        const unsigned long long row = row_base + local_row;
        const unsigned long long batch = output_wave * kTileCols + local_batch;
        const float row_scale =
            row_scales != nullptr && row < row_scale_count ? row_scales[row] : 1.0f;
        output[batch * rows + row] = output_tile[output_index] * tensor_scale * row_scale;
    }
#else
    (void)indices;
    (void)scale_indices;
    (void)codebook;
    (void)scale_values;
    (void)input;
    (void)row_scales;
    (void)scale_count;
    (void)group_size;
    (void)tensor_scale;
    (void)row_scale_count;
    (void)rows;
    (void)cols;
    (void)batch_count;
    (void)output;
#endif
}
)";
    }

    // Production gfx1201 AQ4 group8 WMMA path. It deliberately owns a separate HIPRTC
    // source/module from the group16 production kernel, so its rocWMMA compilation cannot
    // perturb the served group16 code object.
    static const char *aq4_matvec_wmma_group8_prototype_kernel_source() {
        return R"(
#include <rocwmma/rocwmma.hpp>

extern "C" __global__ void ullm_aq4_gemm_wmma_group8_prototype_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long batch_count,
    float *output) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    using namespace rocwmma;
    constexpr unsigned int kTileRows = 16u;
    constexpr unsigned int kTileCols = 16u;
    constexpr unsigned int kWmmaK = 16u;
    constexpr unsigned int kWideK = 32u;
    constexpr unsigned int kBatchTile = 128u;
    constexpr unsigned int kStages = 2u;
    constexpr unsigned int kWavesPerBlock = kBatchTile / kTileCols;
    constexpr unsigned int kThreads = kWavesPerBlock * 32u;
    static_assert(kThreads == 256u, "AQ4 group8 WMMA prototype expects eight wave32s");
    static_assert(kWideK == 2u * kWmmaK, "AQ4 group8 Wide-K path requires two WMMA K tiles");
    static_assert(kStages == 2u, "AQ4 group8 WMMA prototype is a ping-pong LDS pipeline");

    typedef unsigned int ullm_aq4_wmma_group8_uint4 __attribute__((ext_vector_type(4)));
    typedef float ullm_aq4_wmma_group8_float4 __attribute__((ext_vector_type(4)));
    using FragA = fragment<matrix_a, kTileRows, kTileCols, kWmmaK, float16_t, row_major>;
    using FragB = fragment<matrix_b, kTileRows, kTileCols, kWmmaK, float16_t, col_major>;
    using FragC = fragment<accumulator, kTileRows, kTileCols, kWmmaK, float32_t, row_major>;

    constexpr unsigned int kWeightTilesLdsBytes =
        kStages * kTileRows * kWideK * sizeof(float16_t);
    constexpr unsigned int kInputTilesLdsBytes =
        kStages * kBatchTile * kWideK * sizeof(float16_t);
    constexpr unsigned int kOutputTileLdsBytes =
        kWavesPerBlock * kTileRows * kTileCols * sizeof(float);
    constexpr unsigned int kStaticLdsBytes =
        kWeightTilesLdsBytes + kInputTilesLdsBytes + kOutputTileLdsBytes;
    static_assert(kStaticLdsBytes == 26624u, "AQ4 group8 WMMA LDS accounting drifted");
    static_assert(
        kStaticLdsBytes <= 64u * 1024u,
        "AQ4 group8 WMMA prototype kernel must fit gfx1201 LDS");

    __shared__ __attribute__((aligned(16))) float16_t weight_tiles[
        kStages][kTileRows * kWideK];
    __shared__ __attribute__((aligned(16))) float16_t input_tiles[
        kStages][kBatchTile * kWideK];
    __shared__ __attribute__((aligned(16))) float output_tile[
        kWavesPerBlock * kTileRows * kTileCols];

    const unsigned int tid = threadIdx.x;
    const unsigned int wave = tid >> 5;
    const unsigned long long row_base = static_cast<unsigned long long>(blockIdx.x) * kTileRows;
    const unsigned long long batch_base =
        static_cast<unsigned long long>(blockIdx.y) * kBatchTile;

    if (group_size != 8ull || batch_count != kBatchTile || rows == 0ull || cols == 0ull ||
        (rows % kTileRows) != 0ull || (cols % kWideK) != 0ull) {
        return;
    }

    FragC c;
    fill_fragment(c, 0.0f);
    const unsigned long long groups_per_row = cols / 8ull;

    // A 16-byte packed-nibble load is 32 AQ4 elements. For group8 that is four consecutive
    // scale groups, and each uint32 word below contains exactly one eight-element group.
    if (tid < kTileRows) {
        const unsigned long long row = row_base + tid;
        const unsigned long long group_base = row * groups_per_row;
        const unsigned int scale_index0 = static_cast<unsigned int>(scale_indices[group_base]);
        const unsigned int scale_index1 =
            static_cast<unsigned int>(scale_indices[group_base + 1ull]);
        const unsigned int scale_index2 =
            static_cast<unsigned int>(scale_indices[group_base + 2ull]);
        const unsigned int scale_index3 =
            static_cast<unsigned int>(scale_indices[group_base + 3ull]);
        const float group_scale0 =
            scale_index0 < scale_count ? scale_values[scale_index0] : 0.0f;
        const float group_scale1 =
            scale_index1 < scale_count ? scale_values[scale_index1] : 0.0f;
        const float group_scale2 =
            scale_index2 < scale_count ? scale_values[scale_index2] : 0.0f;
        const float group_scale3 =
            scale_index3 < scale_count ? scale_values[scale_index3] : 0.0f;
        const unsigned long long byte_offset = (row * cols) >> 1;
        const ullm_aq4_wmma_group8_uint4 packed_words =
            *reinterpret_cast<const ullm_aq4_wmma_group8_uint4 *>(indices + byte_offset);
#pragma unroll
        for (unsigned int word_index = 0u; word_index < 4u; ++word_index) {
            const unsigned int word = static_cast<unsigned int>(packed_words[word_index]);
            const float group_scale =
                word_index == 0u ? group_scale0 :
                word_index == 1u ? group_scale1 :
                word_index == 2u ? group_scale2 : group_scale3;
#pragma unroll
            for (unsigned int byte_index = 0u; byte_index < 4u; ++byte_index) {
                const unsigned int packed_byte = (word >> (byte_index * 8u)) & 0xffu;
                const unsigned int local_k = word_index * 8u + byte_index * 2u;
                weight_tiles[0][tid * kWideK + local_k] = static_cast<float16_t>(
                    codebook[packed_byte & 0x0fu] * group_scale);
                weight_tiles[0][tid * kWideK + local_k + 1u] = static_cast<float16_t>(
                    codebook[(packed_byte >> 4u) & 0x0fu] * group_scale);
            }
        }
    }
    for (unsigned int input_vector = tid;
         input_vector < (kBatchTile * kWideK) / 4u;
         input_vector += kThreads) {
        const unsigned int batch_in_tile = input_vector / (kWideK / 4u);
        const unsigned int k_in_tile = (input_vector % (kWideK / 4u)) * 4u;
        const unsigned long long input_offset =
            (batch_base + batch_in_tile) * cols + k_in_tile;
        const ullm_aq4_wmma_group8_float4 values =
            *reinterpret_cast<const ullm_aq4_wmma_group8_float4 *>(input + input_offset);
        const unsigned int shared_offset = batch_in_tile * kWideK + k_in_tile;
        input_tiles[0][shared_offset + 0u] = static_cast<float16_t>(values[0]);
        input_tiles[0][shared_offset + 1u] = static_cast<float16_t>(values[1]);
        input_tiles[0][shared_offset + 2u] = static_cast<float16_t>(values[2]);
        input_tiles[0][shared_offset + 3u] = static_cast<float16_t>(values[3]);
    }
    __syncthreads();

    for (unsigned long long k_base = 0ull; k_base < cols; k_base += kWideK) {
        const unsigned int current_stage =
            static_cast<unsigned int>((k_base / kWideK) & 1ull);
        const unsigned int next_stage = current_stage ^ 1u;
        const unsigned int wave_input_offset = wave * kTileCols * kWideK;

        FragA a;
        FragB b;
        load_matrix_sync(a, weight_tiles[current_stage], kWideK);
        load_matrix_sync(b, input_tiles[current_stage] + wave_input_offset, kWideK);

        const bool has_next = k_base < cols - kWideK;
        ullm_aq4_wmma_group8_float4 prefetched_input[4];
        ullm_aq4_wmma_group8_uint4 prefetched_weights;
        float prefetched_group_scale0 = 0.0f;
        float prefetched_group_scale1 = 0.0f;
        float prefetched_group_scale2 = 0.0f;
        float prefetched_group_scale3 = 0.0f;
        if (has_next && tid < kTileRows) {
            const unsigned long long next_k_base = k_base + kWideK;
            const unsigned long long row = row_base + tid;
            const unsigned long long group_base =
                row * groups_per_row + next_k_base / 8ull;
            const unsigned int scale_index0 =
                static_cast<unsigned int>(scale_indices[group_base]);
            const unsigned int scale_index1 =
                static_cast<unsigned int>(scale_indices[group_base + 1ull]);
            const unsigned int scale_index2 =
                static_cast<unsigned int>(scale_indices[group_base + 2ull]);
            const unsigned int scale_index3 =
                static_cast<unsigned int>(scale_indices[group_base + 3ull]);
            prefetched_group_scale0 =
                scale_index0 < scale_count ? scale_values[scale_index0] : 0.0f;
            prefetched_group_scale1 =
                scale_index1 < scale_count ? scale_values[scale_index1] : 0.0f;
            prefetched_group_scale2 =
                scale_index2 < scale_count ? scale_values[scale_index2] : 0.0f;
            prefetched_group_scale3 =
                scale_index3 < scale_count ? scale_values[scale_index3] : 0.0f;
            const unsigned long long byte_offset = (row * cols + next_k_base) >> 1;
            prefetched_weights =
                *reinterpret_cast<const ullm_aq4_wmma_group8_uint4 *>(indices + byte_offset);
        }
        if (has_next) {
#pragma unroll
            for (unsigned int prefetch_index = 0u; prefetch_index < 4u; ++prefetch_index) {
                const unsigned int input_vector = tid + prefetch_index * kThreads;
                const unsigned int batch_in_tile = input_vector / (kWideK / 4u);
                const unsigned int k_in_tile = (input_vector % (kWideK / 4u)) * 4u;
                const unsigned long long input_offset =
                    (batch_base + batch_in_tile) * cols + k_base + kWideK + k_in_tile;
                prefetched_input[prefetch_index] =
                    *reinterpret_cast<const ullm_aq4_wmma_group8_float4 *>(input + input_offset);
            }
        }

        mma_sync(c, a, b, c);
        load_matrix_sync(a, weight_tiles[current_stage] + kWmmaK, kWideK);
        load_matrix_sync(b, input_tiles[current_stage] + wave_input_offset + kWmmaK, kWideK);
        mma_sync(c, a, b, c);

        if (has_next) {
            if (tid < kTileRows) {
#pragma unroll
                for (unsigned int word_index = 0u; word_index < 4u; ++word_index) {
                    const unsigned int word =
                        static_cast<unsigned int>(prefetched_weights[word_index]);
                    const float group_scale =
                        word_index == 0u ? prefetched_group_scale0 :
                        word_index == 1u ? prefetched_group_scale1 :
                        word_index == 2u ? prefetched_group_scale2 : prefetched_group_scale3;
#pragma unroll
                    for (unsigned int byte_index = 0u; byte_index < 4u; ++byte_index) {
                        const unsigned int packed_byte = (word >> (byte_index * 8u)) & 0xffu;
                        const unsigned int local_k = word_index * 8u + byte_index * 2u;
                        weight_tiles[next_stage][tid * kWideK + local_k] =
                            static_cast<float16_t>(codebook[packed_byte & 0x0fu] * group_scale);
                        weight_tiles[next_stage][tid * kWideK + local_k + 1u] =
                            static_cast<float16_t>(
                                codebook[(packed_byte >> 4u) & 0x0fu] * group_scale);
                    }
                }
            }
#pragma unroll
            for (unsigned int prefetch_index = 0u; prefetch_index < 4u; ++prefetch_index) {
                const unsigned int input_vector = tid + prefetch_index * kThreads;
                const unsigned int batch_in_tile = input_vector / (kWideK / 4u);
                const unsigned int k_in_tile = (input_vector % (kWideK / 4u)) * 4u;
                const unsigned int shared_offset = batch_in_tile * kWideK + k_in_tile;
                const ullm_aq4_wmma_group8_float4 values = prefetched_input[prefetch_index];
                input_tiles[next_stage][shared_offset + 0u] = static_cast<float16_t>(values[0]);
                input_tiles[next_stage][shared_offset + 1u] = static_cast<float16_t>(values[1]);
                input_tiles[next_stage][shared_offset + 2u] = static_cast<float16_t>(values[2]);
                input_tiles[next_stage][shared_offset + 3u] = static_cast<float16_t>(values[3]);
            }
            __syncthreads();
        }
    }

    store_matrix_sync(output_tile + wave * kTileRows * kTileCols, c, kTileCols);
    __syncthreads();

    for (unsigned int output_index = tid;
         output_index < kWavesPerBlock * kTileRows * kTileCols;
         output_index += kThreads) {
        const unsigned int output_wave = output_index / (kTileRows * kTileCols);
        const unsigned int output_in_wave = output_index % (kTileRows * kTileCols);
        const unsigned int local_row = output_in_wave / kTileCols;
        const unsigned int local_batch = output_in_wave % kTileCols;
        const unsigned long long row = row_base + local_row;
        const unsigned long long batch = batch_base + output_wave * kTileCols + local_batch;
        const float row_scale =
            row_scales != nullptr && row < row_scale_count ? row_scales[row] : 1.0f;
        output[batch * rows + row] = output_tile[output_index] * tensor_scale * row_scale;
    }
#else
    (void)indices;
    (void)scale_indices;
    (void)codebook;
    (void)scale_values;
    (void)input;
    (void)row_scales;
    (void)scale_count;
    (void)group_size;
    (void)tensor_scale;
    (void)row_scale_count;
    (void)rows;
    (void)cols;
    (void)batch_count;
    (void)output;
#endif
}
)";
    }

    // Ragged-M group8 variant. Group8 has the same two-stage Wide-K ping-pong pipeline as the
    // M=128 source above, so both the initial publication and steady-state prefetch must avoid
    // forming addresses for physical rows outside m_actual. The group8 four-scale decode stays
    // entirely in the shared source and is not changed by this additive experiment.
    static std::string aq4_matvec_wmma_group8_ragged_m_prototype_kernel_source() {
        std::string source(aq4_matvec_wmma_group8_prototype_kernel_source());
        const auto replace_once = [&source](const char *from, const char *to) {
            const size_t position = source.find(from);
            if (position == std::string::npos) {
                return false;
            }
            source.replace(position, std::strlen(from), to);
            return true;
        };
        const bool replaced =
            replace_once(
                "ullm_aq4_gemm_wmma_group8_prototype_f32_kernel",
                "ullm_aq4_gemm_wmma_group8_ragged_m_prototype_f32_kernel") &&
            replace_once(
                "unsigned long long batch_count,\n"
                "    float *output)",
                "unsigned long long m_actual,\n"
                "    float *output)") &&
            replace_once(
                "if (group_size != 8ull || batch_count != kBatchTile || rows == 0ull || cols == 0ull ||\n"
                "        (rows % kTileRows) != 0ull || (cols % kWideK) != 0ull) {",
                "if (group_size != 8ull || m_actual == 0ull || m_actual > kBatchTile ||\n"
                "        rows == 0ull || cols == 0ull || (rows % kTileRows) != 0ull ||\n"
                "        (cols % kWideK) != 0ull) {") &&
            replace_once(
                "        const unsigned long long input_offset =\n"
                "            (batch_base + batch_in_tile) * cols + k_in_tile;\n"
                "        const ullm_aq4_wmma_group8_float4 values =\n"
                "            *reinterpret_cast<const ullm_aq4_wmma_group8_float4 *>(input + input_offset);\n"
                "        const unsigned int shared_offset = batch_in_tile * kWideK + k_in_tile;\n"
                "        input_tiles[0][shared_offset + 0u] = static_cast<float16_t>(values[0]);\n"
                "        input_tiles[0][shared_offset + 1u] = static_cast<float16_t>(values[1]);\n"
                "        input_tiles[0][shared_offset + 2u] = static_cast<float16_t>(values[2]);\n"
                "        input_tiles[0][shared_offset + 3u] = static_cast<float16_t>(values[3]);",
                "        const unsigned int shared_offset = batch_in_tile * kWideK + k_in_tile;\n"
                "        // Check before calculating an address: exact-sized ragged inputs have no tail.\n"
                "        if (batch_base + batch_in_tile < m_actual) {\n"
                "            const unsigned long long input_offset =\n"
                "                (batch_base + batch_in_tile) * cols + k_in_tile;\n"
                "            const ullm_aq4_wmma_group8_float4 values =\n"
                "                *reinterpret_cast<const ullm_aq4_wmma_group8_float4 *>(input + input_offset);\n"
                "            input_tiles[0][shared_offset + 0u] = static_cast<float16_t>(values[0]);\n"
                "            input_tiles[0][shared_offset + 1u] = static_cast<float16_t>(values[1]);\n"
                "            input_tiles[0][shared_offset + 2u] = static_cast<float16_t>(values[2]);\n"
                "            input_tiles[0][shared_offset + 3u] = static_cast<float16_t>(values[3]);\n"
                "        } else {\n"
                "            input_tiles[0][shared_offset + 0u] = static_cast<float16_t>(0.0f);\n"
                "            input_tiles[0][shared_offset + 1u] = static_cast<float16_t>(0.0f);\n"
                "            input_tiles[0][shared_offset + 2u] = static_cast<float16_t>(0.0f);\n"
                "            input_tiles[0][shared_offset + 3u] = static_cast<float16_t>(0.0f);\n"
                "        }") &&
            replace_once(
                "                const unsigned long long input_offset =\n"
                "                    (batch_base + batch_in_tile) * cols + k_base + kWideK + k_in_tile;\n"
                "                prefetched_input[prefetch_index] =\n"
                "                    *reinterpret_cast<const ullm_aq4_wmma_group8_float4 *>(input + input_offset);",
                "                if (batch_base + batch_in_tile < m_actual) {\n"
                "                    const unsigned long long input_offset =\n"
                "                        (batch_base + batch_in_tile) * cols + k_base + kWideK + k_in_tile;\n"
                "                    prefetched_input[prefetch_index] =\n"
                "                        *reinterpret_cast<const ullm_aq4_wmma_group8_float4 *>(input + input_offset);\n"
                "                }") &&
            replace_once(
                "                const unsigned int shared_offset = batch_in_tile * kWideK + k_in_tile;\n"
                "                const ullm_aq4_wmma_group8_float4 values = prefetched_input[prefetch_index];\n"
                "                input_tiles[next_stage][shared_offset + 0u] = static_cast<float16_t>(values[0]);\n"
                "                input_tiles[next_stage][shared_offset + 1u] = static_cast<float16_t>(values[1]);\n"
                "                input_tiles[next_stage][shared_offset + 2u] = static_cast<float16_t>(values[2]);\n"
                "                input_tiles[next_stage][shared_offset + 3u] = static_cast<float16_t>(values[3]);",
                "                const unsigned int shared_offset = batch_in_tile * kWideK + k_in_tile;\n"
                "                if (batch_base + batch_in_tile < m_actual) {\n"
                "                    const ullm_aq4_wmma_group8_float4 values = prefetched_input[prefetch_index];\n"
                "                    input_tiles[next_stage][shared_offset + 0u] = static_cast<float16_t>(values[0]);\n"
                "                    input_tiles[next_stage][shared_offset + 1u] = static_cast<float16_t>(values[1]);\n"
                "                    input_tiles[next_stage][shared_offset + 2u] = static_cast<float16_t>(values[2]);\n"
                "                    input_tiles[next_stage][shared_offset + 3u] = static_cast<float16_t>(values[3]);\n"
                "                } else {\n"
                "                    input_tiles[next_stage][shared_offset + 0u] = static_cast<float16_t>(0.0f);\n"
                "                    input_tiles[next_stage][shared_offset + 1u] = static_cast<float16_t>(0.0f);\n"
                "                    input_tiles[next_stage][shared_offset + 2u] = static_cast<float16_t>(0.0f);\n"
                "                    input_tiles[next_stage][shared_offset + 3u] = static_cast<float16_t>(0.0f);\n"
                "                }") &&
            replace_once(
                "        const float row_scale =\n"
                "            row_scales != nullptr && row < row_scale_count ? row_scales[row] : 1.0f;\n"
                "        output[batch * rows + row] = output_tile[output_index] * tensor_scale * row_scale;",
                "        if (batch < m_actual) {\n"
                "            const float row_scale =\n"
                "                row_scales != nullptr && row < row_scale_count ? row_scales[row] : 1.0f;\n"
                "            output[batch * rows + row] = output_tile[output_index] * tensor_scale * row_scale;\n"
                "        }") &&
            replace_once("    (void)batch_count;", "    (void)m_actual;");
        if (!replaced) {
            return "";
        }
        return source;
    }

    static const char *aq4_matvec_top1_kernel_source() {
        return R"(
extern "C" __global__ void ullm_aq4_matvec_top1_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    float *partial_values,
    unsigned int *partial_indices) {
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float partial[256];
    __shared__ float row_values[32];
    __shared__ unsigned int row_indices[32];
    float sum = 0.0f;
    if (row < rows) {
        const unsigned long long row_offset = static_cast<unsigned long long>(row) * cols;
        if ((cols % group_size) == 0ull) {
            const unsigned long long groups_per_row = cols / group_size;
            for (unsigned long long group_in_row = lane; group_in_row < groups_per_row;
                 group_in_row += threads_per_row) {
                const unsigned long long group =
                    static_cast<unsigned long long>(row) * groups_per_row + group_in_row;
                const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
                if (scale_index >= scale_count) {
                    continue;
                }
                float raw_sum = 0.0f;
                const unsigned long long col_start = group_in_row * group_size;
                if (group_size == 16ull) {
#pragma unroll
                    for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                        const unsigned long long col = col_start + offset;
                        const unsigned char packed = indices[(row_offset + col) >> 1];
                        raw_sum += codebook[packed & 0x0f] * input[col];
                        raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                    }
                } else if (group_size == 8ull) {
#pragma unroll
                    for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                        const unsigned long long col = col_start + offset;
                        const unsigned char packed = indices[(row_offset + col) >> 1];
                        raw_sum += codebook[packed & 0x0f] * input[col];
                        raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                    }
                } else if ((group_size & 1ull) == 0ull) {
                    for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                        const unsigned long long col = col_start + offset;
                        const unsigned char packed = indices[(row_offset + col) >> 1];
                        raw_sum += codebook[packed & 0x0f] * input[col];
                        raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                    }
                } else {
                    for (unsigned long long offset = 0; offset < group_size; ++offset) {
                        const unsigned long long col = col_start + offset;
                        const unsigned long long element = row_offset + col;
                        const unsigned char packed = indices[element >> 1];
                        const unsigned char codebook_index =
                            (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                        raw_sum += codebook[codebook_index] * input[col];
                    }
                }
                sum += raw_sum * scale_values[scale_index] * tensor_scale;
            }
        } else {
            for (unsigned long long col = lane; col < cols; col += threads_per_row) {
                const unsigned long long element = row_offset + col;
                const unsigned char packed = indices[element >> 1];
                const unsigned char codebook_index =
                    (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                const unsigned long long group = element / group_size;
                const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
                if (scale_index < scale_count) {
                    const float value =
                        codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                    sum += value * input[col];
                }
            }
        }
    }
    float row_sum = sum;
#if defined(ULLM_AQ4_MATVEC_TOP1_USE_SHARED_REDUCE)
    partial[tid] = row_sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            partial[partial_offset + lane] += partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    row_sum = partial[partial_offset];
#else
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        row_sum += __shfl_down(row_sum, offset, threads_per_row);
    }
#endif
    if (lane == 0) {
        float value = -3.4028234663852886e38f;
        unsigned int token_index = 0xffffffffu;
        if (row < rows) {
            value = row_sum;
            if (row_scales != nullptr && row < row_scale_count) {
                value *= row_scales[row];
            }
            if (!(value == value)) {
                value = -3.4028234663852886e38f;
            }
            token_index = static_cast<unsigned int>(row);
        }
        row_values[row_in_block] = value;
        row_indices[row_in_block] = token_index;
    }
    __syncthreads();
    if (tid < rows_per_block) {
        for (unsigned int offset = rows_per_block >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const float right_value = row_values[tid + offset];
                const unsigned int right_index = row_indices[tid + offset];
                const float left_value = row_values[tid];
                const unsigned int left_index = row_indices[tid];
                if (right_value > left_value ||
                    (right_value == left_value && right_index < left_index)) {
                    row_values[tid] = right_value;
                    row_indices[tid] = right_index;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            partial_values[blockIdx.x] = row_values[0];
            partial_indices[blockIdx.x] = row_indices[0];
        }
    }
}
)";
    }

    // Superseded scalar AQ4 matvec-add reference. It is deliberately not passed to HIPRTC:
    // production keeps the same entry-point ABI but uses the validated wide-load body below.
    // Retain this source for controlled rollback comparisons.
    [[maybe_unused]] static const char *aq4_matvec_add_scalar_reference_kernel_source() {
        return R"(
__device__ float ullm_aq4_matvec_add_thread_sum(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size) {
    const unsigned long long row_offset = row * cols;
    float sum = 0.0f;
    if ((cols % group_size) == 0ull) {
        const unsigned long long groups_per_row = cols / group_size;
        for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
             group_in_row += block_size) {
            const unsigned long long group = row * groups_per_row + group_in_row;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index >= scale_count) {
                continue;
            }
            float raw_sum = 0.0f;
            const unsigned long long col_start = group_in_row * group_size;
            if (group_size == 16ull) {
#pragma unroll
                for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else if (group_size == 8ull) {
#pragma unroll
                for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else if ((group_size & 1ull) == 0ull) {
                for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else {
                for (unsigned long long offset = 0; offset < group_size; ++offset) {
                    const unsigned long long col = col_start + offset;
                    const unsigned long long element = row_offset + col;
                    const unsigned char packed = indices[element >> 1];
                    const unsigned char codebook_index =
                        (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                    raw_sum += codebook[codebook_index] * input[col];
                }
            }
            sum += raw_sum * scale_values[scale_index] * tensor_scale;
        }
    } else {
        for (unsigned long long col = tid; col < cols; col += block_size) {
            const unsigned long long element = row_offset + col;
            const unsigned char packed = indices[element >> 1];
            const unsigned char codebook_index =
                (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
            const unsigned long long group = element / group_size;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index < scale_count) {
                const float value =
                    codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                sum += value * input[col];
            }
        }
    }
    return sum;
}

extern "C" __global__ void ullm_aq4_matvec_add_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *residual,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    float *output) {
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float partial[256];
    float sum = 0.0f;
    if (row < rows) {
        sum = ullm_aq4_matvec_add_thread_sum(
            indices,
            scale_indices,
            codebook,
            scale_values,
            input,
            scale_count,
            group_size,
            tensor_scale,
            row,
            cols,
            lane,
            threads_per_row);
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            partial[partial_offset + lane] += partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && row < rows) {
        float value = partial[partial_offset];
        if (row_scales != nullptr && row < row_scale_count) {
            value *= row_scales[row];
        }
        output[row] = residual[row] + value;
    }
}
)";
    }

    // Superseded LDS-reduction AQ4 matvec-add wide-load reference. It remains the rollback body
    // and is used for every architecture or RPB outside the validated gfx1201 RPB=8 geometry.
    // Retain this source for controlled rollback comparisons.
    [[maybe_unused]] static const char *aq4_matvec_add_wide_load_reference_kernel_source() {
        return R"(
extern "C" __global__ void ullm_aq4_matvec_add_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *residual,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    float *output) {
    static_assert(sizeof(uint4) == 16u, "AQ4 add wide load must remain 128 bits");
    static_assert(alignof(uint4) == 16u, "AQ4 add wide load must remain 16-byte aligned");
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float partial[256];
    float sum = 0.0f;
    if (row < rows) {
        const unsigned long long row_offset = row * cols;
        const unsigned long long row_byte_offset = row_offset >> 1;
        const unsigned long long chunks_per_row = cols / 32ull;
        const unsigned long long groups_per_row = cols / group_size;
        // Validated Qwen3.5 AQ4 matvec-add geometries have group8/group16 and complete,
        // aligned 32-element chunks. Each scoped uint4 supplies 16 packed bytes: four group8s
        // or two group16s for this one stream.
        for (unsigned long long chunk = lane; chunk < chunks_per_row;
             chunk += threads_per_row) {
            const uint4 packed = *reinterpret_cast<const uint4 *>(
                indices + row_byte_offset + chunk * 16ull);
            const unsigned long long col_base = chunk * 32ull;
            const unsigned long long group_base =
                row * groups_per_row + col_base / group_size;
            const unsigned int groups_per_chunk = group_size == 8ull ? 4u : 2u;
#pragma unroll
            for (unsigned int group_in_chunk = 0u; group_in_chunk < 4u; ++group_in_chunk) {
                if (group_in_chunk >= groups_per_chunk) {
                    continue;
                }
                const unsigned long long group = group_base + group_in_chunk;
                const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
                if (scale_index >= scale_count) {
                    continue;
                }
                const unsigned int byte_start =
                    group_in_chunk * static_cast<unsigned int>(group_size >> 1);
                const unsigned int byte_count = static_cast<unsigned int>(group_size >> 1);
                float raw_sum = 0.0f;
#pragma unroll
                for (unsigned int byte_offset = 0u; byte_offset < 8u; ++byte_offset) {
                    if (byte_offset >= byte_count) {
                        continue;
                    }
                    const unsigned int packed_byte_index = byte_start + byte_offset;
                    const unsigned int word_index = packed_byte_index >> 2;
                    const unsigned int word = word_index == 0u ? packed.x :
                        (word_index == 1u ? packed.y :
                            (word_index == 2u ? packed.z : packed.w));
                    const unsigned int packed_byte =
                        (word >> (8u * (packed_byte_index & 3u))) & 0xffu;
                    const unsigned long long col =
                        col_base + static_cast<unsigned long long>(packed_byte_index * 2u);
                    raw_sum += codebook[packed_byte & 0x0fu] * input[col];
                    raw_sum += codebook[(packed_byte >> 4) & 0x0fu] * input[col + 1ull];
                }
                sum += raw_sum * scale_values[scale_index] * tensor_scale;
            }
        }
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            partial[partial_offset + lane] += partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    // Preserve the production residual-add epilogue: it is not another AQ4 stream.
    if (lane == 0 && row < rows) {
        float value = partial[partial_offset];
        if (row_scales != nullptr && row < row_scale_count) {
            value *= row_scales[row];
        }
        output[row] = residual[row] + value;
    }
}
)";
    }

    // This is the retained gfx1201/RPB=8 baseline: the promoted uint4 traversal with the
    // wave32 reduction, but before group-size-specialized traversal.  Keep it available through
    // the direct-only shuffle prototype so a candidate remains directly comparable to the exact
    // production predecessor rather than to the older LDS body.
    static std::string aq4_matvec_add_shuffle_reference_kernel_source() {
        std::string production = aq4_matvec_add_wide_load_reference_kernel_source();

        const std::string lds_setup = R"(    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float partial[256];
    float sum = 0.0f;)";
        const std::string shuffle_setup = R"(    // RPB=8 maps each row's contiguous 32 threads to one gfx1201 wave32.
    float sum = 0.0f;)";
        const size_t setup_offset = production.find(lds_setup);
        if (setup_offset == std::string::npos) {
            return "";
        }
        production.replace(setup_offset, lds_setup.size(), shuffle_setup);

        const std::string lds_tree = R"(    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            partial[partial_offset + lane] += partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    // Preserve the production residual-add epilogue: it is not another AQ4 stream.
    if (lane == 0 && row < rows) {
        float value = partial[partial_offset];)";
        const std::string shuffle_reduction = R"(    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, threads_per_row);
    }
    // Preserve the production residual-add epilogue: it is not another AQ4 stream.
    if (lane == 0 && row < rows) {
        float value = sum;)";
        const size_t tree_offset = production.find(lds_tree);
        if (tree_offset == std::string::npos) {
            return "";
        }
        production.replace(tree_offset, lds_tree.size(), shuffle_reduction);
        return production;
    }

    // Specialize the only two shapes accepted by aq4_matvec_add_f32_hip_kernel: g8 and g16,
    // both with complete 32-element uint4 chunks.  The retained generic body walks four group
    // slots and eight byte slots dynamically.  In this body the uniform group-size dispatch
    // selects g8/g16 once per row, then each word and byte position is statically bounded.
    //
    // The arithmetic order is intentionally preserved: each byte contributes low nibble first,
    // then high nibble; g16 consumes x,y before z,w; a group is scaled only after all of its
    // elements are accumulated.  The residual/row-scale output epilogue is inherited unchanged.
    static std::string aq4_matvec_add_group_specialized_kernel_source() {
        std::string production = aq4_matvec_add_shuffle_reference_kernel_source();
        const std::string traversal_start = R"(    if (row < rows) {
        const unsigned long long row_offset = row * cols;
)";
        const std::string reduction_start =
            "    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {";
        const size_t start = production.find(traversal_start);
        if (start == std::string::npos) {
            return "";
        }
        const size_t end = production.find(reduction_start, start);
        if (end == std::string::npos || end < start || end < 6u ||
            production.compare(end - 6u, 6u, "    }\n") != 0) {
            return "";
        }
        const std::string specialized_traversal = R"(    if (row < rows) {
        const unsigned long long row_offset = row * cols;
        const unsigned long long row_byte_offset = row_offset >> 1;
        const unsigned long long chunks_per_row = cols / 32ull;
        if (group_size == 8ull) {
            const unsigned long long groups_per_row = cols >> 3;
            for (unsigned long long chunk = lane; chunk < chunks_per_row;
                 chunk += threads_per_row) {
                const uint4 packed = *reinterpret_cast<const uint4 *>(
                    indices + row_byte_offset + chunk * 16ull);
                const unsigned long long col_base = chunk * 32ull;
                const unsigned long long group_base = row * groups_per_row + chunk * 4ull;
#pragma unroll
                for (unsigned int group_in_chunk = 0u; group_in_chunk < 4u; ++group_in_chunk) {
                    const unsigned int scale_index =
                        static_cast<unsigned int>(scale_indices[group_base + group_in_chunk]);
                    if (scale_index >= scale_count) {
                        continue;
                    }
                    const unsigned int word = group_in_chunk == 0u ? packed.x :
                        (group_in_chunk == 1u ? packed.y :
                            (group_in_chunk == 2u ? packed.z : packed.w));
                    float raw_sum = 0.0f;
#pragma unroll
                    for (unsigned int byte_offset = 0u; byte_offset < 4u; ++byte_offset) {
                        const unsigned int packed_byte =
                            (word >> (8u * byte_offset)) & 0xffu;
                        const unsigned long long col = col_base +
                            static_cast<unsigned long long>(group_in_chunk * 8u + byte_offset * 2u);
                        raw_sum += codebook[packed_byte & 0x0fu] * input[col];
                        raw_sum += codebook[(packed_byte >> 4) & 0x0fu] * input[col + 1ull];
                    }
                    sum += raw_sum * scale_values[scale_index] * tensor_scale;
                }
            }
        } else if (group_size == 16ull) {
            const unsigned long long groups_per_row = cols >> 4;
            for (unsigned long long chunk = lane; chunk < chunks_per_row;
                 chunk += threads_per_row) {
                const uint4 packed = *reinterpret_cast<const uint4 *>(
                    indices + row_byte_offset + chunk * 16ull);
                const unsigned long long col_base = chunk * 32ull;
                const unsigned long long group_base = row * groups_per_row + chunk * 2ull;
#pragma unroll
                for (unsigned int group_in_chunk = 0u; group_in_chunk < 2u; ++group_in_chunk) {
                    const unsigned int scale_index =
                        static_cast<unsigned int>(scale_indices[group_base + group_in_chunk]);
                    if (scale_index >= scale_count) {
                        continue;
                    }
                    const unsigned int first_word =
                        group_in_chunk == 0u ? packed.x : packed.z;
                    const unsigned int second_word =
                        group_in_chunk == 0u ? packed.y : packed.w;
                    float raw_sum = 0.0f;
#pragma unroll
                    for (unsigned int byte_offset = 0u; byte_offset < 4u; ++byte_offset) {
                        const unsigned int packed_byte =
                            (first_word >> (8u * byte_offset)) & 0xffu;
                        const unsigned long long col = col_base +
                            static_cast<unsigned long long>(group_in_chunk * 16u + byte_offset * 2u);
                        raw_sum += codebook[packed_byte & 0x0fu] * input[col];
                        raw_sum += codebook[(packed_byte >> 4) & 0x0fu] * input[col + 1ull];
                    }
#pragma unroll
                    for (unsigned int byte_offset = 0u; byte_offset < 4u; ++byte_offset) {
                        const unsigned int packed_byte =
                            (second_word >> (8u * byte_offset)) & 0xffu;
                        const unsigned long long col = col_base +
                            static_cast<unsigned long long>(
                                group_in_chunk * 16u + 8u + byte_offset * 2u);
                        raw_sum += codebook[packed_byte & 0x0fu] * input[col];
                        raw_sum += codebook[(packed_byte >> 4) & 0x0fu] * input[col + 1ull];
                    }
                    sum += raw_sum * scale_values[scale_index] * tensor_scale;
                }
            }
        }
    }
)";
        production.replace(start, end - start, specialized_traversal);
        return production;
    }

    // Production AQ4 matvec-add body. The host ABI and HIPRTC entry-point name remain unchanged,
    // so package resident decode continues through its existing direct aq4_matvec_add_f32 dispatch.
    static std::string aq4_matvec_add_kernel_source() {
        return aq4_matvec_add_group_specialized_kernel_source();
    }

    // Keep the direct-only prototype ABI available for historical differential/timing comparison.
    // It compiles the wide-load rollback body under its original, isolated entry-point name.
    static std::string aq4_matvec_add_wide_load_prototype_kernel_source() {
        std::string source = aq4_matvec_add_wide_load_reference_kernel_source();
        constexpr const char production_name[] = "ullm_aq4_matvec_add_f32_kernel";
        const size_t name_offset = source.find(production_name);
        if (name_offset == std::string::npos) {
            return "";
        }
        source.replace(
            name_offset,
            sizeof(production_name) - 1u,
            "ullm_aq4_matvec_add_wide_load_prototype_f32_kernel");
        return source;
    }

    // Keep the direct-only prototype ABI available as the exact pre-specialization gfx1201/RPB=8
    // baseline for differential and timing comparisons.
    static std::string aq4_matvec_add_shuffle_prototype_kernel_source() {
        std::string source = aq4_matvec_add_shuffle_reference_kernel_source();
        constexpr const char production_name[] = "ullm_aq4_matvec_add_f32_kernel";
        const size_t name_offset = source.find(production_name);
        if (name_offset == std::string::npos) {
            return "";
        }
        source.replace(
            name_offset,
            sizeof(production_name) - 1u,
            "ullm_aq4_matvec_add_shuffle_prototype_f32_kernel");
        return source;
    }

    static const char *aq4_matvec_pair_kernel_source() {
        return R"(
__device__ float ullm_aq4_matvec_pair_thread_sum(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size) {
    const unsigned long long row_offset = row * cols;
    float sum = 0.0f;
    if ((cols % group_size) == 0ull) {
        const unsigned long long groups_per_row = cols / group_size;
        for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
             group_in_row += block_size) {
            const unsigned long long group = row * groups_per_row + group_in_row;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index >= scale_count) {
                continue;
            }
            float raw_sum = 0.0f;
            const unsigned long long col_start = group_in_row * group_size;
            if (group_size == 16ull) {
#pragma unroll
                for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else if (group_size == 8ull) {
#pragma unroll
                for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else if ((group_size & 1ull) == 0ull) {
                for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else {
                for (unsigned long long offset = 0; offset < group_size; ++offset) {
                    const unsigned long long col = col_start + offset;
                    const unsigned long long element = row_offset + col;
                    const unsigned char packed = indices[element >> 1];
                    const unsigned char codebook_index =
                        (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                    raw_sum += codebook[codebook_index] * input[col];
                }
            }
            sum += raw_sum * scale_values[scale_index] * tensor_scale;
        }
    } else {
        for (unsigned long long col = tid; col < cols; col += block_size) {
            const unsigned long long element = row_offset + col;
            const unsigned char packed = indices[element >> 1];
            const unsigned char codebook_index =
                (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
            const unsigned long long group = element / group_size;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index < scale_count) {
                const float value =
                    codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                sum += value * input[col];
            }
        }
    }
    return sum;
}

__device__ void ullm_aq4_matvec_pair_thread_sums(
    const unsigned char *left_indices,
    const unsigned char *left_scale_indices,
    const float *left_codebook,
    const float *left_scale_values,
    unsigned long long left_scale_count,
    float left_tensor_scale,
    const unsigned char *right_indices,
    const unsigned char *right_scale_indices,
    const float *right_codebook,
    const float *right_scale_values,
    unsigned long long right_scale_count,
    float right_tensor_scale,
    const float *input,
    unsigned long long group_size,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size,
    float *left_sum_output,
    float *right_sum_output) {
    const unsigned long long row_offset = row * cols;
    const unsigned long long groups_per_row = cols / group_size;
    float left_sum = 0.0f;
    float right_sum = 0.0f;
    for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
         group_in_row += block_size) {
        const unsigned long long group = row * groups_per_row + group_in_row;
        const unsigned int left_scale_index =
            static_cast<unsigned int>(left_scale_indices[group]);
        const unsigned int right_scale_index =
            static_cast<unsigned int>(right_scale_indices[group]);
        float left_raw_sum = 0.0f;
        float right_raw_sum = 0.0f;
        const unsigned long long col_start = group_in_row * group_size;
        if (group_size == 16ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char left_packed = left_indices[packed_index];
                const unsigned char right_packed = right_indices[packed_index];
                left_raw_sum += left_codebook[left_packed & 0x0f] * input_low;
                right_raw_sum += right_codebook[right_packed & 0x0f] * input_low;
                left_raw_sum += left_codebook[(left_packed >> 4) & 0x0f] * input_high;
                right_raw_sum += right_codebook[(right_packed >> 4) & 0x0f] * input_high;
            }
        } else if (group_size == 8ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char left_packed = left_indices[packed_index];
                const unsigned char right_packed = right_indices[packed_index];
                left_raw_sum += left_codebook[left_packed & 0x0f] * input_low;
                right_raw_sum += right_codebook[right_packed & 0x0f] * input_low;
                left_raw_sum += left_codebook[(left_packed >> 4) & 0x0f] * input_high;
                right_raw_sum += right_codebook[(right_packed >> 4) & 0x0f] * input_high;
            }
        } else if ((group_size & 1ull) == 0ull) {
            for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char left_packed = left_indices[packed_index];
                const unsigned char right_packed = right_indices[packed_index];
                left_raw_sum += left_codebook[left_packed & 0x0f] * input_low;
                right_raw_sum += right_codebook[right_packed & 0x0f] * input_low;
                left_raw_sum += left_codebook[(left_packed >> 4) & 0x0f] * input_high;
                right_raw_sum += right_codebook[(right_packed >> 4) & 0x0f] * input_high;
            }
        } else {
            for (unsigned long long offset = 0; offset < group_size; ++offset) {
                const unsigned long long col = col_start + offset;
                const unsigned long long element = row_offset + col;
                const float input_value = input[col];
                const unsigned char left_packed = left_indices[element >> 1];
                const unsigned char left_codebook_index =
                    (element & 1ull) == 0ull ? (left_packed & 0x0f) :
                                               ((left_packed >> 4) & 0x0f);
                const unsigned char right_packed = right_indices[element >> 1];
                const unsigned char right_codebook_index =
                    (element & 1ull) == 0ull ? (right_packed & 0x0f) :
                                               ((right_packed >> 4) & 0x0f);
                left_raw_sum += left_codebook[left_codebook_index] * input_value;
                right_raw_sum += right_codebook[right_codebook_index] * input_value;
            }
        }
        if (left_scale_index < left_scale_count) {
            left_sum += left_raw_sum * left_scale_values[left_scale_index] * left_tensor_scale;
        }
        if (right_scale_index < right_scale_count) {
            right_sum +=
                right_raw_sum * right_scale_values[right_scale_index] * right_tensor_scale;
        }
    }
    *left_sum_output = left_sum;
    *right_sum_output = right_sum;
}

extern "C" __global__ void ullm_aq4_matvec_pair_f32_kernel(
    const unsigned char *left_indices,
    const unsigned char *left_scale_indices,
    const float *left_codebook,
    const float *left_scale_values,
    const float *left_row_scales,
    unsigned long long left_scale_count,
    unsigned long long left_group_size,
    float left_tensor_scale,
    unsigned long long left_row_scale_count,
    const unsigned char *right_indices,
    const unsigned char *right_scale_indices,
    const float *right_codebook,
    const float *right_scale_values,
    const float *right_row_scales,
    unsigned long long right_scale_count,
    unsigned long long right_group_size,
    float right_tensor_scale,
    unsigned long long right_row_scale_count,
    const float *input,
    unsigned long long left_rows,
    unsigned long long right_rows,
    unsigned long long cols,
    float *left_output,
    float *right_output) {
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned long long work_rows = left_rows > right_rows ? left_rows : right_rows;
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float left_partial[256];
    __shared__ float right_partial[256];
    float left_sum = 0.0f;
    float right_sum = 0.0f;
    if (row < work_rows) {
        if (row < left_rows && row < right_rows && left_group_size == right_group_size &&
            (cols % left_group_size) == 0ull) {
            ullm_aq4_matvec_pair_thread_sums(
                left_indices,
                left_scale_indices,
                left_codebook,
                left_scale_values,
                left_scale_count,
                left_tensor_scale,
                right_indices,
                right_scale_indices,
                right_codebook,
                right_scale_values,
                right_scale_count,
                right_tensor_scale,
                input,
                left_group_size,
                row,
                cols,
                lane,
                threads_per_row,
                &left_sum,
                &right_sum);
        } else {
            if (row < left_rows) {
                left_sum = ullm_aq4_matvec_pair_thread_sum(
                    left_indices,
                    left_scale_indices,
                    left_codebook,
                    left_scale_values,
                    input,
                    left_scale_count,
                    left_group_size,
                    left_tensor_scale,
                    row,
                    cols,
                    lane,
                    threads_per_row);
            }
            if (row < right_rows) {
                right_sum = ullm_aq4_matvec_pair_thread_sum(
                    right_indices,
                    right_scale_indices,
                    right_codebook,
                    right_scale_values,
                    input,
                    right_scale_count,
                    right_group_size,
                    right_tensor_scale,
                    row,
                    cols,
                    lane,
                    threads_per_row);
            }
        }
    }
    left_partial[tid] = left_sum;
    right_partial[tid] = right_sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            left_partial[partial_offset + lane] += left_partial[partial_offset + lane + offset];
            right_partial[partial_offset + lane] +=
                right_partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0) {
        if (row < left_rows) {
            float left_value = left_partial[partial_offset];
            if (left_row_scales != nullptr && row < left_row_scale_count) {
                left_value *= left_row_scales[row];
            }
            left_output[row] = left_value;
        }
        if (row < right_rows) {
            float right_value = right_partial[partial_offset];
            if (right_row_scales != nullptr && row < right_row_scale_count) {
                right_value *= right_row_scales[row];
            }
            right_output[row] = right_value;
        }
    }
}
)";
    }

    // Superseded scalar-reduction AQ4 matvec triple reference. It is deliberately not passed
    // to HIPRTC for the validated gfx1201 RPB=8 geometry: production keeps the same entry-point
    // ABI but uses the wave32 shuffle reduction below. Retain this source for controlled rollback
    // comparisons and for generic HIPRTC compilations outside that validated geometry.
    [[maybe_unused]] static const char *aq4_matvec_triple_scalar_reference_kernel_source() {
        return R"(
__device__ float ullm_aq4_matvec_triple_thread_sum(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size) {
    const unsigned long long row_offset = row * cols;
    float sum = 0.0f;
    if ((cols % group_size) == 0ull) {
        const unsigned long long groups_per_row = cols / group_size;
        for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
             group_in_row += block_size) {
            const unsigned long long group = row * groups_per_row + group_in_row;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index >= scale_count) {
                continue;
            }
            float raw_sum = 0.0f;
            const unsigned long long col_start = group_in_row * group_size;
            if (group_size == 16ull) {
#pragma unroll
                for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else if (group_size == 8ull) {
#pragma unroll
                for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else if ((group_size & 1ull) == 0ull) {
                for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else {
                for (unsigned long long offset = 0; offset < group_size; ++offset) {
                    const unsigned long long col = col_start + offset;
                    const unsigned long long element = row_offset + col;
                    const unsigned char packed = indices[element >> 1];
                    const unsigned char codebook_index =
                        (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                    raw_sum += codebook[codebook_index] * input[col];
                }
            }
            sum += raw_sum * scale_values[scale_index] * tensor_scale;
        }
    } else {
        for (unsigned long long col = tid; col < cols; col += block_size) {
            const unsigned long long element = row_offset + col;
            const unsigned char packed = indices[element >> 1];
            const unsigned char codebook_index =
                (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
            const unsigned long long group = element / group_size;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index < scale_count) {
                const float value =
                    codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                sum += value * input[col];
            }
        }
    }
    return sum;
}

__device__ void ullm_aq4_matvec_triple_thread_sums(
    const unsigned char *first_indices,
    const unsigned char *first_scale_indices,
    const float *first_codebook,
    const float *first_scale_values,
    unsigned long long first_scale_count,
    float first_tensor_scale,
    const unsigned char *second_indices,
    const unsigned char *second_scale_indices,
    const float *second_codebook,
    const float *second_scale_values,
    unsigned long long second_scale_count,
    float second_tensor_scale,
    const unsigned char *third_indices,
    const unsigned char *third_scale_indices,
    const float *third_codebook,
    const float *third_scale_values,
    unsigned long long third_scale_count,
    float third_tensor_scale,
    const float *input,
    unsigned long long group_size,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size,
    float *first_sum_output,
    float *second_sum_output,
    float *third_sum_output) {
    const unsigned long long row_offset = row * cols;
    const unsigned long long groups_per_row = cols / group_size;
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    float third_sum = 0.0f;
    for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
         group_in_row += block_size) {
        const unsigned long long group = row * groups_per_row + group_in_row;
        const unsigned int first_scale_index =
            static_cast<unsigned int>(first_scale_indices[group]);
        const unsigned int second_scale_index =
            static_cast<unsigned int>(second_scale_indices[group]);
        const unsigned int third_scale_index =
            static_cast<unsigned int>(third_scale_indices[group]);
        float first_raw_sum = 0.0f;
        float second_raw_sum = 0.0f;
        float third_raw_sum = 0.0f;
        const unsigned long long col_start = group_in_row * group_size;
        if (group_size == 16ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char first_packed = first_indices[packed_index];
                const unsigned char second_packed = second_indices[packed_index];
                const unsigned char third_packed = third_indices[packed_index];
                first_raw_sum += first_codebook[first_packed & 0x0f] * input_low;
                second_raw_sum += second_codebook[second_packed & 0x0f] * input_low;
                third_raw_sum += third_codebook[third_packed & 0x0f] * input_low;
                first_raw_sum += first_codebook[(first_packed >> 4) & 0x0f] * input_high;
                second_raw_sum += second_codebook[(second_packed >> 4) & 0x0f] * input_high;
                third_raw_sum += third_codebook[(third_packed >> 4) & 0x0f] * input_high;
            }
        } else if (group_size == 8ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char first_packed = first_indices[packed_index];
                const unsigned char second_packed = second_indices[packed_index];
                const unsigned char third_packed = third_indices[packed_index];
                first_raw_sum += first_codebook[first_packed & 0x0f] * input_low;
                second_raw_sum += second_codebook[second_packed & 0x0f] * input_low;
                third_raw_sum += third_codebook[third_packed & 0x0f] * input_low;
                first_raw_sum += first_codebook[(first_packed >> 4) & 0x0f] * input_high;
                second_raw_sum += second_codebook[(second_packed >> 4) & 0x0f] * input_high;
                third_raw_sum += third_codebook[(third_packed >> 4) & 0x0f] * input_high;
            }
        } else if ((group_size & 1ull) == 0ull) {
            for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char first_packed = first_indices[packed_index];
                const unsigned char second_packed = second_indices[packed_index];
                const unsigned char third_packed = third_indices[packed_index];
                first_raw_sum += first_codebook[first_packed & 0x0f] * input_low;
                second_raw_sum += second_codebook[second_packed & 0x0f] * input_low;
                third_raw_sum += third_codebook[third_packed & 0x0f] * input_low;
                first_raw_sum += first_codebook[(first_packed >> 4) & 0x0f] * input_high;
                second_raw_sum += second_codebook[(second_packed >> 4) & 0x0f] * input_high;
                third_raw_sum += third_codebook[(third_packed >> 4) & 0x0f] * input_high;
            }
        } else {
            for (unsigned long long offset = 0; offset < group_size; ++offset) {
                const unsigned long long col = col_start + offset;
                const unsigned long long element = row_offset + col;
                const float input_value = input[col];
                const unsigned char first_packed = first_indices[element >> 1];
                const unsigned char first_codebook_index =
                    (element & 1ull) == 0ull ? (first_packed & 0x0f) :
                                               ((first_packed >> 4) & 0x0f);
                const unsigned char second_packed = second_indices[element >> 1];
                const unsigned char second_codebook_index =
                    (element & 1ull) == 0ull ? (second_packed & 0x0f) :
                                               ((second_packed >> 4) & 0x0f);
                const unsigned char third_packed = third_indices[element >> 1];
                const unsigned char third_codebook_index =
                    (element & 1ull) == 0ull ? (third_packed & 0x0f) :
                                               ((third_packed >> 4) & 0x0f);
                first_raw_sum += first_codebook[first_codebook_index] * input_value;
                second_raw_sum += second_codebook[second_codebook_index] * input_value;
                third_raw_sum += third_codebook[third_codebook_index] * input_value;
            }
        }
        if (first_scale_index < first_scale_count) {
            first_sum += first_raw_sum * first_scale_values[first_scale_index] *
                         first_tensor_scale;
        }
        if (second_scale_index < second_scale_count) {
            second_sum += second_raw_sum * second_scale_values[second_scale_index] *
                          second_tensor_scale;
        }
        if (third_scale_index < third_scale_count) {
            third_sum += third_raw_sum * third_scale_values[third_scale_index] *
                         third_tensor_scale;
        }
    }
    *first_sum_output = first_sum;
    *second_sum_output = second_sum;
    *third_sum_output = third_sum;
}

extern "C" __global__ void ullm_aq4_matvec_triple_f32_kernel(
    const unsigned char *first_indices,
    const unsigned char *first_scale_indices,
    const float *first_codebook,
    const float *first_scale_values,
    const float *first_row_scales,
    unsigned long long first_scale_count,
    unsigned long long first_group_size,
    float first_tensor_scale,
    unsigned long long first_row_scale_count,
    const unsigned char *second_indices,
    const unsigned char *second_scale_indices,
    const float *second_codebook,
    const float *second_scale_values,
    const float *second_row_scales,
    unsigned long long second_scale_count,
    unsigned long long second_group_size,
    float second_tensor_scale,
    unsigned long long second_row_scale_count,
    const unsigned char *third_indices,
    const unsigned char *third_scale_indices,
    const float *third_codebook,
    const float *third_scale_values,
    const float *third_row_scales,
    unsigned long long third_scale_count,
    unsigned long long third_group_size,
    float third_tensor_scale,
    unsigned long long third_row_scale_count,
    const float *input,
    unsigned long long first_rows,
    unsigned long long second_rows,
    unsigned long long third_rows,
    unsigned long long cols,
    float *first_output,
    float *second_output,
    float *third_output) {
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    unsigned long long work_rows = first_rows > second_rows ? first_rows : second_rows;
    work_rows = work_rows > third_rows ? work_rows : third_rows;
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float first_partial[256];
    __shared__ float second_partial[256];
    __shared__ float third_partial[256];
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    float third_sum = 0.0f;
    if (row < work_rows) {
        if (row < first_rows && row < second_rows && row < third_rows &&
            first_group_size == second_group_size && first_group_size == third_group_size &&
            (cols % first_group_size) == 0ull) {
            ullm_aq4_matvec_triple_thread_sums(
                first_indices,
                first_scale_indices,
                first_codebook,
                first_scale_values,
                first_scale_count,
                first_tensor_scale,
                second_indices,
                second_scale_indices,
                second_codebook,
                second_scale_values,
                second_scale_count,
                second_tensor_scale,
                third_indices,
                third_scale_indices,
                third_codebook,
                third_scale_values,
                third_scale_count,
                third_tensor_scale,
                input,
                first_group_size,
                row,
                cols,
                lane,
                threads_per_row,
                &first_sum,
                &second_sum,
                &third_sum);
        } else {
            if (row < first_rows) {
                first_sum = ullm_aq4_matvec_triple_thread_sum(
                    first_indices,
                    first_scale_indices,
                    first_codebook,
                    first_scale_values,
                    input,
                    first_scale_count,
                    first_group_size,
                    first_tensor_scale,
                    row,
                    cols,
                    lane,
                    threads_per_row);
            }
            if (row < second_rows) {
                second_sum = ullm_aq4_matvec_triple_thread_sum(
                    second_indices,
                    second_scale_indices,
                    second_codebook,
                    second_scale_values,
                    input,
                    second_scale_count,
                    second_group_size,
                    second_tensor_scale,
                    row,
                    cols,
                    lane,
                    threads_per_row);
            }
            if (row < third_rows) {
                third_sum = ullm_aq4_matvec_triple_thread_sum(
                    third_indices,
                    third_scale_indices,
                    third_codebook,
                    third_scale_values,
                    input,
                    third_scale_count,
                    third_group_size,
                    third_tensor_scale,
                    row,
                    cols,
                    lane,
                    threads_per_row);
            }
        }
    }
    first_partial[tid] = first_sum;
    second_partial[tid] = second_sum;
    third_partial[tid] = third_sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            first_partial[partial_offset + lane] +=
                first_partial[partial_offset + lane + offset];
            second_partial[partial_offset + lane] +=
                second_partial[partial_offset + lane + offset];
            third_partial[partial_offset + lane] +=
                third_partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0) {
        if (row < first_rows) {
            float first_value = first_partial[partial_offset];
            if (first_row_scales != nullptr && row < first_row_scale_count) {
                first_value *= first_row_scales[row];
            }
            first_output[row] = first_value;
        }
        if (row < second_rows) {
            float second_value = second_partial[partial_offset];
            if (second_row_scales != nullptr && row < second_row_scale_count) {
                second_value *= second_row_scales[row];
            }
            second_output[row] = second_value;
        }
        if (row < third_rows) {
            float third_value = third_partial[partial_offset];
            if (third_row_scales != nullptr && row < third_row_scale_count) {
                third_value *= third_row_scales[row];
            }
            third_output[row] = third_value;
        }
    }
}
)";
    }

    // Production AQ4 matvec triple body. The host ABI and HIPRTC entry-point name remain
    // unchanged, so Qwen3.5 self-attention continues through its existing fused dispatch.
    static std::string aq4_matvec_triple_kernel_source() {
        std::string prototype = aq4_matvec_triple_scalar_reference_kernel_source();

        const std::string lds_setup = R"(    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float first_partial[256];
    __shared__ float second_partial[256];
    __shared__ float third_partial[256];
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    float third_sum = 0.0f;)";
        const std::string shuffle_setup = R"(    // RPB=8 maps each row's contiguous 32 threads to one gfx1201 wave32.
    // Retain the three independent stream accumulators, but no cross-wave/LDS combine is needed.
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    float third_sum = 0.0f;)";
        const size_t setup_offset = prototype.find(lds_setup);
        if (setup_offset == std::string::npos) return "";
        prototype.replace(setup_offset, lds_setup.size(), shuffle_setup);

        const std::string lds_tree = R"(    first_partial[tid] = first_sum;
    second_partial[tid] = second_sum;
    third_partial[tid] = third_sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            first_partial[partial_offset + lane] +=
                first_partial[partial_offset + lane + offset];
            second_partial[partial_offset + lane] +=
                second_partial[partial_offset + lane + offset];
            third_partial[partial_offset + lane] +=
                third_partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0) {
        if (row < first_rows) {
            float first_value = first_partial[partial_offset];)";
        const std::string shuffle_reduction = R"(    for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        first_sum += __shfl_down(first_sum, offset, warpSize);
        second_sum += __shfl_down(second_sum, offset, warpSize);
        third_sum += __shfl_down(third_sum, offset, warpSize);
    }
    if (lane == 0) {
        if (row < first_rows) {
            float first_value = first_sum;)";
        const size_t tree_offset = prototype.find(lds_tree);
        if (tree_offset == std::string::npos) return "";
        prototype.replace(tree_offset, lds_tree.size(), shuffle_reduction);

        const std::string second_value = "float second_value = second_partial[partial_offset];";
        const size_t second_value_offset = prototype.find(second_value);
        if (second_value_offset == std::string::npos) return "";
        prototype.replace(second_value_offset, second_value.size(), "float second_value = second_sum;");

        const std::string third_value = "float third_value = third_partial[partial_offset];";
        const size_t third_value_offset = prototype.find(third_value);
        if (third_value_offset == std::string::npos) return "";
        prototype.replace(third_value_offset, third_value.size(), "float third_value = third_sum;");
        return prototype;
    }

    // Keep the direct-only prototype ABI available for historical differential/timing comparison.
    // It compiles the same validated body under its original, isolated entry-point name.
    static std::string aq4_matvec_triple_shuffle_prototype_kernel_source() {
        std::string prototype = aq4_matvec_triple_kernel_source();
        const std::string production_name = "ullm_aq4_matvec_triple_f32_kernel";
        const size_t name_offset = prototype.find(production_name);
        if (name_offset == std::string::npos) return "";
        prototype.replace(
            name_offset,
            production_name.size(),
            "ullm_aq4_matvec_triple_shuffle_prototype_f32_kernel");
        return prototype;
    }

    // Superseded scalar-reduction AQ4 qkv/z gate/beta reference. It is deliberately not passed
    // to HIPRTC for the validated gfx1201 RPB=4 geometry: production keeps the same entry-point
    // ABI but uses the wave32 shuffle reduction below. Retain this source for controlled rollback
    // comparisons and for generic HIPRTC compilations outside that validated geometry.
    [[maybe_unused]] static const char *aq4_matvec_qkv_z_gate_beta_scalar_reference_kernel_source() {
        return R"(
__device__ float ullm_aq4_qkv_z_gate_beta_thread_sum(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size) {
    const unsigned long long row_offset = row * cols;
    float sum = 0.0f;
    if ((cols % group_size) == 0ull) {
        const unsigned long long groups_per_row = cols / group_size;
        for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
             group_in_row += block_size) {
            const unsigned long long group = row * groups_per_row + group_in_row;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index >= scale_count) {
                continue;
            }
            float raw_sum = 0.0f;
            const unsigned long long col_start = group_in_row * group_size;
            if (group_size == 16ull) {
#pragma unroll
                for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else if (group_size == 8ull) {
#pragma unroll
                for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else {
                for (unsigned long long offset = 0; offset < group_size; ++offset) {
                    const unsigned long long col = col_start + offset;
                    const unsigned long long element = row_offset + col;
                    const unsigned char packed = indices[element >> 1];
                    const unsigned char codebook_index =
                        (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                    raw_sum += codebook[codebook_index] * input[col];
                }
            }
            sum += raw_sum * scale_values[scale_index] * tensor_scale;
        }
    } else {
        for (unsigned long long col = tid; col < cols; col += block_size) {
            const unsigned long long element = row_offset + col;
            const unsigned char packed = indices[element >> 1];
            const unsigned char codebook_index =
                (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
            const unsigned long long group = element / group_size;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index < scale_count) {
                const float value =
                    codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                sum += value * input[col];
            }
        }
    }
    return sum;
}

__device__ void ullm_aq4_qkv_z_gate_beta_pair_thread_sums(
    const unsigned char *a_indices,
    const unsigned char *a_scale_indices,
    const float *a_codebook,
    const float *a_scale_values,
    unsigned long long a_scale_count,
    float a_tensor_scale,
    const unsigned char *b_indices,
    const unsigned char *b_scale_indices,
    const float *b_codebook,
    const float *b_scale_values,
    unsigned long long b_scale_count,
    float b_tensor_scale,
    const float *input,
    unsigned long long group_size,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size,
    float *a_sum_output,
    float *b_sum_output) {
    const unsigned long long row_offset = row * cols;
    const unsigned long long groups_per_row = cols / group_size;
    float a_sum = 0.0f;
    float b_sum = 0.0f;
    for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
         group_in_row += block_size) {
        const unsigned long long group = row * groups_per_row + group_in_row;
        const unsigned int a_scale_index = static_cast<unsigned int>(a_scale_indices[group]);
        const unsigned int b_scale_index = static_cast<unsigned int>(b_scale_indices[group]);
        float a_raw_sum = 0.0f;
        float b_raw_sum = 0.0f;
        const unsigned long long col_start = group_in_row * group_size;
        if (group_size == 16ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char a_packed = a_indices[packed_index];
                const unsigned char b_packed = b_indices[packed_index];
                a_raw_sum += a_codebook[a_packed & 0x0f] * input_low;
                b_raw_sum += b_codebook[b_packed & 0x0f] * input_low;
                a_raw_sum += a_codebook[(a_packed >> 4) & 0x0f] * input_high;
                b_raw_sum += b_codebook[(b_packed >> 4) & 0x0f] * input_high;
            }
        } else if (group_size == 8ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char a_packed = a_indices[packed_index];
                const unsigned char b_packed = b_indices[packed_index];
                a_raw_sum += a_codebook[a_packed & 0x0f] * input_low;
                b_raw_sum += b_codebook[b_packed & 0x0f] * input_low;
                a_raw_sum += a_codebook[(a_packed >> 4) & 0x0f] * input_high;
                b_raw_sum += b_codebook[(b_packed >> 4) & 0x0f] * input_high;
            }
        } else if ((group_size & 1ull) == 0ull) {
            for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char a_packed = a_indices[packed_index];
                const unsigned char b_packed = b_indices[packed_index];
                a_raw_sum += a_codebook[a_packed & 0x0f] * input_low;
                b_raw_sum += b_codebook[b_packed & 0x0f] * input_low;
                a_raw_sum += a_codebook[(a_packed >> 4) & 0x0f] * input_high;
                b_raw_sum += b_codebook[(b_packed >> 4) & 0x0f] * input_high;
            }
        } else {
            for (unsigned long long offset = 0; offset < group_size; ++offset) {
                const unsigned long long col = col_start + offset;
                const unsigned long long element = row_offset + col;
                const float input_value = input[col];
                const unsigned char a_packed = a_indices[element >> 1];
                const unsigned char a_codebook_index =
                    (element & 1ull) == 0ull ? (a_packed & 0x0f) :
                                               ((a_packed >> 4) & 0x0f);
                const unsigned char b_packed = b_indices[element >> 1];
                const unsigned char b_codebook_index =
                    (element & 1ull) == 0ull ? (b_packed & 0x0f) :
                                               ((b_packed >> 4) & 0x0f);
                a_raw_sum += a_codebook[a_codebook_index] * input_value;
                b_raw_sum += b_codebook[b_codebook_index] * input_value;
            }
        }
        if (a_scale_index < a_scale_count) {
            a_sum += a_raw_sum * a_scale_values[a_scale_index] * a_tensor_scale;
        }
        if (b_scale_index < b_scale_count) {
            b_sum += b_raw_sum * b_scale_values[b_scale_index] * b_tensor_scale;
        }
    }
    *a_sum_output = a_sum;
    *b_sum_output = b_sum;
}

extern "C" __global__ void ullm_aq4_matvec_qkv_z_gate_beta_f32_kernel(
    const unsigned char *qkv_indices,
    const unsigned char *qkv_scale_indices,
    const float *qkv_codebook,
    const float *qkv_scale_values,
    const float *qkv_row_scales,
    unsigned long long qkv_scale_count,
    unsigned long long qkv_group_size,
    float qkv_tensor_scale,
    unsigned long long qkv_row_scale_count,
    const unsigned char *z_indices,
    const unsigned char *z_scale_indices,
    const float *z_codebook,
    const float *z_scale_values,
    const float *z_row_scales,
    unsigned long long z_scale_count,
    unsigned long long z_group_size,
    float z_tensor_scale,
    unsigned long long z_row_scale_count,
    const unsigned char *a_indices,
    const unsigned char *a_scale_indices,
    const float *a_codebook,
    const float *a_scale_values,
    const float *a_row_scales,
    unsigned long long a_scale_count,
    unsigned long long a_group_size,
    float a_tensor_scale,
    unsigned long long a_row_scale_count,
    const unsigned char *b_indices,
    const unsigned char *b_scale_indices,
    const float *b_codebook,
    const float *b_scale_values,
    const float *b_row_scales,
    unsigned long long b_scale_count,
    unsigned long long b_group_size,
    float b_tensor_scale,
    unsigned long long b_row_scale_count,
    const float *input,
    const float *a_log,
    const float *dt_bias,
    unsigned long long qkv_rows,
    unsigned long long z_rows,
    unsigned long long heads,
    unsigned long long cols,
    float *qkv_output,
    float *z_output,
    float *gate_output,
    float *beta_output) {
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long logical_row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned long long projection_rows = qkv_rows > z_rows ? qkv_rows : z_rows;
    const unsigned long long total_rows = projection_rows + heads;
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float primary_partial[256];
    __shared__ float secondary_partial[256];
    float primary_sum = 0.0f;
    float secondary_sum = 0.0f;
    if (logical_row < projection_rows) {
        if (logical_row < qkv_rows && logical_row < z_rows &&
            qkv_group_size == z_group_size && (cols % qkv_group_size) == 0ull) {
            ullm_aq4_qkv_z_gate_beta_pair_thread_sums(
                qkv_indices,
                qkv_scale_indices,
                qkv_codebook,
                qkv_scale_values,
                qkv_scale_count,
                qkv_tensor_scale,
                z_indices,
                z_scale_indices,
                z_codebook,
                z_scale_values,
                z_scale_count,
                z_tensor_scale,
                input,
                qkv_group_size,
                logical_row,
                cols,
                lane,
                threads_per_row,
                &primary_sum,
                &secondary_sum);
        } else {
            if (logical_row < qkv_rows) {
                primary_sum = ullm_aq4_qkv_z_gate_beta_thread_sum(
                    qkv_indices,
                    qkv_scale_indices,
                    qkv_codebook,
                    qkv_scale_values,
                    input,
                    qkv_scale_count,
                    qkv_group_size,
                    qkv_tensor_scale,
                    logical_row,
                    cols,
                    lane,
                    threads_per_row);
            }
            if (logical_row < z_rows) {
                secondary_sum = ullm_aq4_qkv_z_gate_beta_thread_sum(
                    z_indices,
                    z_scale_indices,
                    z_codebook,
                    z_scale_values,
                    input,
                    z_scale_count,
                    z_group_size,
                    z_tensor_scale,
                    logical_row,
                    cols,
                    lane,
                    threads_per_row);
            }
        }
    } else if (logical_row < total_rows) {
        const unsigned long long head = logical_row - projection_rows;
        if (a_group_size == b_group_size && (cols % a_group_size) == 0ull) {
            ullm_aq4_qkv_z_gate_beta_pair_thread_sums(
                a_indices,
                a_scale_indices,
                a_codebook,
                a_scale_values,
                a_scale_count,
                a_tensor_scale,
                b_indices,
                b_scale_indices,
                b_codebook,
                b_scale_values,
                b_scale_count,
                b_tensor_scale,
                input,
                a_group_size,
                head,
                cols,
                lane,
                threads_per_row,
                &primary_sum,
                &secondary_sum);
        } else {
            primary_sum = ullm_aq4_qkv_z_gate_beta_thread_sum(
                a_indices,
                a_scale_indices,
                a_codebook,
                a_scale_values,
                input,
                a_scale_count,
                a_group_size,
                a_tensor_scale,
                head,
                cols,
                lane,
                threads_per_row);
            secondary_sum = ullm_aq4_qkv_z_gate_beta_thread_sum(
                b_indices,
                b_scale_indices,
                b_codebook,
                b_scale_values,
                input,
                b_scale_count,
                b_group_size,
                b_tensor_scale,
                head,
                cols,
                lane,
                threads_per_row);
        }
    }
    primary_partial[tid] = primary_sum;
    secondary_partial[tid] = secondary_sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            primary_partial[partial_offset + lane] += primary_partial[partial_offset + lane + offset];
            secondary_partial[partial_offset + lane] +=
                secondary_partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && logical_row < total_rows) {
        float value = primary_partial[partial_offset];
        if (logical_row < projection_rows) {
            if (logical_row < qkv_rows) {
                if (qkv_row_scales != nullptr && logical_row < qkv_row_scale_count) {
                    value *= qkv_row_scales[logical_row];
                }
                qkv_output[logical_row] = value;
            }
            if (logical_row < z_rows) {
                float z_value = secondary_partial[partial_offset];
                if (z_row_scales != nullptr && logical_row < z_row_scale_count) {
                    z_value *= z_row_scales[logical_row];
                }
                z_output[logical_row] = z_value;
            }
        } else {
            const unsigned long long head = logical_row - projection_rows;
            float b_value = secondary_partial[partial_offset];
            if (a_row_scales != nullptr && head < a_row_scale_count) {
                value *= a_row_scales[head];
            }
            if (b_row_scales != nullptr && head < b_row_scale_count) {
                b_value *= b_row_scales[head];
            }
            const float x = value + dt_bias[head];
            const float softplus = x <= 20.0f ? log1pf(expf(x)) : x;
            gate_output[head] = -expf(a_log[head]) * softplus;
            beta_output[head] = 1.0f / (1.0f + expf(-b_value));
        }
    }
}
)";
    }

    static std::string aq4_matvec_qkv_z_gate_beta_wide_load_prototype_kernel_source() {
        std::string production = aq4_matvec_qkv_z_gate_beta_scalar_reference_kernel_source();
        const size_t kernel_offset = production.find("extern \"C\" __global__ void");
        if (kernel_offset == std::string::npos) {
            return "";
        }
        production.erase(0u, kernel_offset);
        const std::string production_name = "ullm_aq4_matvec_qkv_z_gate_beta_f32_kernel";
        const size_t name_offset = production.find(production_name);
        if (name_offset == std::string::npos) {
            return "";
        }
        production.replace(
            name_offset,
            production_name.size(),
            "ullm_aq4_matvec_qkv_z_gate_beta_wide_load_prototype_f32_kernel");
        return R"(
// Each uint4 is four 32-bit words = exactly sixteen packed bytes, never 32 bytes.
static_assert(sizeof(uint4) == 16u, "AQ4 wide load must remain 128 bits");
static_assert(alignof(uint4) == 16u, "AQ4 wide load must remain 16-byte aligned");

__device__ unsigned int ullm_aq4_wide_load_packed_byte(uint4 words, unsigned int byte_index) {
    const unsigned int word = byte_index < 4u ? words.x :
        (byte_index < 8u ? words.y : (byte_index < 12u ? words.z : words.w));
    return (word >> (8u * (byte_index & 3u))) & 0xffu;
}

__device__ float ullm_aq4_qkv_z_gate_beta_thread_sum(
    const unsigned char *indices, const unsigned char *scale_indices,
    const float *codebook, const float *scale_values, const float *input,
    unsigned long long scale_count, unsigned long long group_size, float tensor_scale,
    unsigned long long row, unsigned long long cols, unsigned int tid, unsigned int block_size) {
    const unsigned long long groups_per_row = cols / group_size;
    const unsigned long long row_byte_offset = (row * cols) >> 1;
    const unsigned int groups_per_chunk = group_size == 8ull ? 4u : 2u;
    float sum = 0.0f;
    for (unsigned long long chunk = tid; chunk < cols / 32ull; chunk += block_size) {
        const uint4 packed = *reinterpret_cast<const uint4 *>(indices + row_byte_offset + chunk * 16ull);
        const unsigned long long col_base = chunk * 32ull;
        const unsigned long long group_base = row * groups_per_row + col_base / group_size;
#pragma unroll
        for (unsigned int group_in_chunk = 0u; group_in_chunk < 4u; ++group_in_chunk) {
            if (group_in_chunk >= groups_per_chunk) continue;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group_base + group_in_chunk]);
            if (scale_index >= scale_count) continue;
            const unsigned int byte_start = group_in_chunk * static_cast<unsigned int>(group_size >> 1);
            const unsigned int byte_count = static_cast<unsigned int>(group_size >> 1);
            float raw_sum = 0.0f;
#pragma unroll
            for (unsigned int byte_offset = 0u; byte_offset < 8u; ++byte_offset) {
                if (byte_offset >= byte_count) continue;
                const unsigned int packed_byte_index = byte_start + byte_offset;
                const unsigned int packed_byte = ullm_aq4_wide_load_packed_byte(packed, packed_byte_index);
                const unsigned long long col = col_base + static_cast<unsigned long long>(packed_byte_index) * 2ull;
                raw_sum += codebook[packed_byte & 0x0fu] * input[col];
                raw_sum += codebook[(packed_byte >> 4) & 0x0fu] * input[col + 1ull];
            }
            sum += raw_sum * scale_values[scale_index] * tensor_scale;
        }
    }
    return sum;
}

__device__ void ullm_aq4_qkv_z_gate_beta_pair_thread_sums(
    const unsigned char *a_indices, const unsigned char *a_scale_indices,
    const float *a_codebook, const float *a_scale_values, unsigned long long a_scale_count,
    float a_tensor_scale, const unsigned char *b_indices, const unsigned char *b_scale_indices,
    const float *b_codebook, const float *b_scale_values, unsigned long long b_scale_count,
    float b_tensor_scale, const float *input, unsigned long long group_size,
    unsigned long long row, unsigned long long cols, unsigned int tid, unsigned int block_size,
    float *a_sum_output, float *b_sum_output) {
    const unsigned long long groups_per_row = cols / group_size;
    const unsigned long long row_byte_offset = (row * cols) >> 1;
    const unsigned int groups_per_chunk = group_size == 8ull ? 4u : 2u;
    float a_sum = 0.0f, b_sum = 0.0f;
    for (unsigned long long chunk = tid; chunk < cols / 32ull; chunk += block_size) {
        // These are intentionally independent wide traversals: the matrices do not share bytes.
        const uint4 a_packed = *reinterpret_cast<const uint4 *>(a_indices + row_byte_offset + chunk * 16ull);
        const uint4 b_packed = *reinterpret_cast<const uint4 *>(b_indices + row_byte_offset + chunk * 16ull);
        const unsigned long long col_base = chunk * 32ull;
        const unsigned long long group_base = row * groups_per_row + col_base / group_size;
#pragma unroll
        for (unsigned int group_in_chunk = 0u; group_in_chunk < 4u; ++group_in_chunk) {
            if (group_in_chunk >= groups_per_chunk) continue;
            const unsigned int a_scale_index = static_cast<unsigned int>(a_scale_indices[group_base + group_in_chunk]);
            const unsigned int b_scale_index = static_cast<unsigned int>(b_scale_indices[group_base + group_in_chunk]);
            const unsigned int byte_start = group_in_chunk * static_cast<unsigned int>(group_size >> 1);
            const unsigned int byte_count = static_cast<unsigned int>(group_size >> 1);
            float a_raw_sum = 0.0f, b_raw_sum = 0.0f;
#pragma unroll
            for (unsigned int byte_offset = 0u; byte_offset < 8u; ++byte_offset) {
                if (byte_offset >= byte_count) continue;
                const unsigned int packed_byte_index = byte_start + byte_offset;
                const unsigned int a_byte = ullm_aq4_wide_load_packed_byte(a_packed, packed_byte_index);
                const unsigned int b_byte = ullm_aq4_wide_load_packed_byte(b_packed, packed_byte_index);
                const unsigned long long col = col_base + static_cast<unsigned long long>(packed_byte_index) * 2ull;
                a_raw_sum += a_codebook[a_byte & 0x0fu] * input[col] + a_codebook[(a_byte >> 4) & 0x0fu] * input[col + 1ull];
                b_raw_sum += b_codebook[b_byte & 0x0fu] * input[col] + b_codebook[(b_byte >> 4) & 0x0fu] * input[col + 1ull];
            }
            if (a_scale_index < a_scale_count) a_sum += a_raw_sum * a_scale_values[a_scale_index] * a_tensor_scale;
            if (b_scale_index < b_scale_count) b_sum += b_raw_sum * b_scale_values[b_scale_index] * b_tensor_scale;
        }
    }
    *a_sum_output = a_sum; *b_sum_output = b_sum;
}
)" + production;
    }

    // Production AQ4 qkv/z gate/beta body. The host ABI and HIPRTC entry-point name remain
    // unchanged, so package resident decode continues through its existing fused dispatch.
    static std::string aq4_matvec_qkv_z_gate_beta_kernel_source() {
        std::string production = aq4_matvec_qkv_z_gate_beta_scalar_reference_kernel_source();

        const std::string lds_setup = R"(    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float primary_partial[256];
    __shared__ float secondary_partial[256];
    float primary_sum = 0.0f;
    float secondary_sum = 0.0f;)";
        const std::string shuffle_setup = R"(    // RPB=4 yields 64 contiguous threads per row: exactly two gfx1201 wave32s.
    // Preserve the two independent pair-stream accumulators while retaining only one value per
    // stream/wave for the cross-wave combine (4 rows * 2 waves = 8 values each).
    const unsigned int wave_in_row = lane / warpSize;
    const unsigned int lane_in_wave = lane % warpSize;
    const unsigned int wave_partial_offset = row_in_block * 2u + wave_in_row;
    __shared__ float primary_wave_partial[8];
    __shared__ float secondary_wave_partial[8];
    float primary_sum = 0.0f;
    float secondary_sum = 0.0f;)";
        const size_t setup_offset = production.find(lds_setup);
        if (setup_offset == std::string::npos) return "";
        production.replace(setup_offset, lds_setup.size(), shuffle_setup);

        const std::string lds_tree = R"(    primary_partial[tid] = primary_sum;
    secondary_partial[tid] = secondary_sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            primary_partial[partial_offset + lane] += primary_partial[partial_offset + lane + offset];
            secondary_partial[partial_offset + lane] +=
                secondary_partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && logical_row < total_rows) {
        float value = primary_partial[partial_offset];)";
        const std::string shuffle_reduction = R"(    // Shuffle within each wave first. The source has exactly 64 threads per row, so
    // no shuffle crosses row boundaries; the remaining cross-wave sum is two LDS reads.
    for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        primary_sum += __shfl_down(primary_sum, offset, warpSize);
        secondary_sum += __shfl_down(secondary_sum, offset, warpSize);
    }
    if (lane_in_wave == 0u) {
        primary_wave_partial[wave_partial_offset] = primary_sum;
        secondary_wave_partial[wave_partial_offset] = secondary_sum;
    }
    __syncthreads();
    if (lane == 0 && logical_row < total_rows) {
        float value = primary_wave_partial[row_in_block * 2u] +
            primary_wave_partial[row_in_block * 2u + 1u];)";
        const size_t tree_offset = production.find(lds_tree);
        if (tree_offset == std::string::npos) return "";
        production.replace(tree_offset, lds_tree.size(), shuffle_reduction);

        const std::string secondary_value = "float z_value = secondary_partial[partial_offset];";
        const std::string secondary_wave_value =
            "float z_value = secondary_wave_partial[row_in_block * 2u] + "
            "secondary_wave_partial[row_in_block * 2u + 1u];";
        const size_t z_value_offset = production.find(secondary_value);
        if (z_value_offset == std::string::npos) return "";
        production.replace(z_value_offset, secondary_value.size(), secondary_wave_value);

        const std::string b_value = "float b_value = secondary_partial[partial_offset];";
        const std::string secondary_b_value =
            "float b_value = secondary_wave_partial[row_in_block * 2u] + "
            "secondary_wave_partial[row_in_block * 2u + 1u];";
        const size_t b_value_offset = production.find(b_value);
        if (b_value_offset == std::string::npos) return "";
        production.replace(b_value_offset, b_value.size(), secondary_b_value);
        return production;
    }

    // Keep the direct-only prototype ABI available for historical differential/timing comparison.
    // It compiles the same validated body under its original, isolated entry-point name.
    static std::string aq4_matvec_qkv_z_gate_beta_shuffle_prototype_kernel_source() {
        std::string prototype = aq4_matvec_qkv_z_gate_beta_kernel_source();
        const std::string production_name = "ullm_aq4_matvec_qkv_z_gate_beta_f32_kernel";
        const size_t name_offset = prototype.find(production_name);
        if (name_offset == std::string::npos) return "";
        prototype.replace(name_offset, production_name.size(),
            "ullm_aq4_matvec_qkv_z_gate_beta_shuffle_prototype_f32_kernel");
        return prototype;
    }

    // Superseded scalar-reduction AQ4 matvec SiLU-mul reference. It is deliberately not passed
    // to HIPRTC: production keeps the same entry-point ABI but uses the validated wave32
    // shuffle reduction below. Retain this source for controlled rollback comparisons.
    [[maybe_unused]] static const char *aq4_matvec_silu_mul_scalar_reference_kernel_source() {
        return R"(
__device__ float ullm_aq4_matvec_thread_sum(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size) {
    const unsigned long long row_offset = row * cols;
    float sum = 0.0f;
    if ((cols % group_size) == 0ull) {
        const unsigned long long groups_per_row = cols / group_size;
        for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
             group_in_row += block_size) {
            const unsigned long long group = row * groups_per_row + group_in_row;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index >= scale_count) {
                continue;
            }
            float raw_sum = 0.0f;
            const unsigned long long col_start = group_in_row * group_size;
            for (unsigned long long offset = 0; offset < group_size; ++offset) {
                const unsigned long long col = col_start + offset;
                const unsigned long long element = row_offset + col;
                const unsigned char packed = indices[element >> 1];
                const unsigned char codebook_index =
                    (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                raw_sum += codebook[codebook_index] * input[col];
            }
            sum += raw_sum * scale_values[scale_index] * tensor_scale;
        }
    } else {
        for (unsigned long long col = tid; col < cols; col += block_size) {
            const unsigned long long element = row_offset + col;
            const unsigned char packed = indices[element >> 1];
            const unsigned char codebook_index =
                (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
            const unsigned long long group = element / group_size;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index < scale_count) {
                const float value =
                    codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                sum += value * input[col];
            }
        }
    }
    return sum;
}

__device__ void ullm_aq4_matvec_silu_mul_thread_sums(
    const unsigned char *gate_indices,
    const unsigned char *gate_scale_indices,
    const float *gate_codebook,
    const float *gate_scale_values,
    unsigned long long gate_scale_count,
    float gate_tensor_scale,
    const unsigned char *up_indices,
    const unsigned char *up_scale_indices,
    const float *up_codebook,
    const float *up_scale_values,
    unsigned long long up_scale_count,
    float up_tensor_scale,
    const float *input,
    unsigned long long group_size,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size,
    float *gate_sum_output,
    float *up_sum_output) {
    const unsigned long long row_offset = row * cols;
    const unsigned long long groups_per_row = cols / group_size;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
         group_in_row += block_size) {
        const unsigned long long group = row * groups_per_row + group_in_row;
        const unsigned int gate_scale_index =
            static_cast<unsigned int>(gate_scale_indices[group]);
        const unsigned int up_scale_index = static_cast<unsigned int>(up_scale_indices[group]);
        float gate_raw_sum = 0.0f;
        float up_raw_sum = 0.0f;
        const unsigned long long col_start = group_in_row * group_size;
        if (group_size == 16ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char gate_packed = gate_indices[packed_index];
                const unsigned char up_packed = up_indices[packed_index];
                gate_raw_sum += gate_codebook[gate_packed & 0x0f] * input_low;
                up_raw_sum += up_codebook[up_packed & 0x0f] * input_low;
                gate_raw_sum += gate_codebook[(gate_packed >> 4) & 0x0f] * input_high;
                up_raw_sum += up_codebook[(up_packed >> 4) & 0x0f] * input_high;
            }
        } else if (group_size == 8ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char gate_packed = gate_indices[packed_index];
                const unsigned char up_packed = up_indices[packed_index];
                gate_raw_sum += gate_codebook[gate_packed & 0x0f] * input_low;
                up_raw_sum += up_codebook[up_packed & 0x0f] * input_low;
                gate_raw_sum += gate_codebook[(gate_packed >> 4) & 0x0f] * input_high;
                up_raw_sum += up_codebook[(up_packed >> 4) & 0x0f] * input_high;
            }
        } else if ((group_size & 1ull) == 0ull) {
            for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char gate_packed = gate_indices[packed_index];
                const unsigned char up_packed = up_indices[packed_index];
                gate_raw_sum += gate_codebook[gate_packed & 0x0f] * input_low;
                up_raw_sum += up_codebook[up_packed & 0x0f] * input_low;
                gate_raw_sum += gate_codebook[(gate_packed >> 4) & 0x0f] * input_high;
                up_raw_sum += up_codebook[(up_packed >> 4) & 0x0f] * input_high;
            }
        } else {
            for (unsigned long long offset = 0; offset < group_size; ++offset) {
                const unsigned long long col = col_start + offset;
                const unsigned long long element = row_offset + col;
                const float input_value = input[col];
                const unsigned char gate_packed = gate_indices[element >> 1];
                const unsigned char gate_codebook_index =
                    (element & 1ull) == 0ull ? (gate_packed & 0x0f) :
                                               ((gate_packed >> 4) & 0x0f);
                const unsigned char up_packed = up_indices[element >> 1];
                const unsigned char up_codebook_index =
                    (element & 1ull) == 0ull ? (up_packed & 0x0f) : ((up_packed >> 4) & 0x0f);
                gate_raw_sum += gate_codebook[gate_codebook_index] * input_value;
                up_raw_sum += up_codebook[up_codebook_index] * input_value;
            }
        }
        if (gate_scale_index < gate_scale_count) {
            gate_sum += gate_raw_sum * gate_scale_values[gate_scale_index] * gate_tensor_scale;
        }
        if (up_scale_index < up_scale_count) {
            up_sum += up_raw_sum * up_scale_values[up_scale_index] * up_tensor_scale;
        }
    }
    *gate_sum_output = gate_sum;
    *up_sum_output = up_sum;
}

extern "C" __global__ void ullm_aq4_matvec_silu_mul_f32_kernel(
    const unsigned char *gate_indices,
    const unsigned char *gate_scale_indices,
    const float *gate_codebook,
    const float *gate_scale_values,
    const float *gate_row_scales,
    unsigned long long gate_scale_count,
    unsigned long long gate_group_size,
    float gate_tensor_scale,
    unsigned long long gate_row_scale_count,
    const unsigned char *up_indices,
    const unsigned char *up_scale_indices,
    const float *up_codebook,
    const float *up_scale_values,
    const float *up_row_scales,
    unsigned long long up_scale_count,
    unsigned long long up_group_size,
    float up_tensor_scale,
    unsigned long long up_row_scale_count,
    const float *input,
    unsigned long long rows,
    unsigned long long cols,
    float *output) {
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float gate_partial[256];
    __shared__ float up_partial[256];
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    if (row < rows) {
        if (gate_group_size == up_group_size && (cols % gate_group_size) == 0ull) {
            ullm_aq4_matvec_silu_mul_thread_sums(
                gate_indices,
                gate_scale_indices,
                gate_codebook,
                gate_scale_values,
                gate_scale_count,
                gate_tensor_scale,
                up_indices,
                up_scale_indices,
                up_codebook,
                up_scale_values,
                up_scale_count,
                up_tensor_scale,
                input,
                gate_group_size,
                row,
                cols,
                lane,
                threads_per_row,
                &gate_sum,
                &up_sum);
        } else {
            gate_sum = ullm_aq4_matvec_thread_sum(
                gate_indices,
                gate_scale_indices,
                gate_codebook,
                gate_scale_values,
                input,
                gate_scale_count,
                gate_group_size,
                gate_tensor_scale,
                row,
                cols,
                lane,
                threads_per_row);
            up_sum = ullm_aq4_matvec_thread_sum(
                up_indices,
                up_scale_indices,
                up_codebook,
                up_scale_values,
                input,
                up_scale_count,
                up_group_size,
                up_tensor_scale,
                row,
                cols,
                lane,
                threads_per_row);
        }
    }
    gate_partial[tid] = gate_sum;
    up_partial[tid] = up_sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            gate_partial[partial_offset + lane] += gate_partial[partial_offset + lane + offset];
            up_partial[partial_offset + lane] += up_partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && row < rows) {
        float gate_value = gate_partial[partial_offset];
        float up_value = up_partial[partial_offset];
        if (gate_row_scales != nullptr && row < gate_row_scale_count) {
            gate_value *= gate_row_scales[row];
        }
        if (up_row_scales != nullptr && row < up_row_scale_count) {
            up_value *= up_row_scales[row];
        }
        const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
        output[row] = gate_value * sigmoid * up_value;
    }
}
)";
    }

    static std::string aq4_matvec_silu_mul_wide_load_prototype_kernel_source() {
        std::string production = aq4_matvec_silu_mul_scalar_reference_kernel_source();
        const size_t kernel_offset = production.find("extern \"C\" __global__ void");
        if (kernel_offset == std::string::npos) return "";
        production.erase(0u, kernel_offset);
        const std::string production_name = "ullm_aq4_matvec_silu_mul_f32_kernel";
        const size_t name_offset = production.find(production_name);
        if (name_offset == std::string::npos) return "";
        production.replace(
            name_offset,
            production_name.size(),
            "ullm_aq4_matvec_silu_mul_wide_load_prototype_f32_kernel");
        return R"(
static_assert(sizeof(uint4) == 16u, "AQ4 wide load must remain 128 bits");
static_assert(alignof(uint4) == 16u, "AQ4 wide load must remain 16-byte aligned");
__device__ unsigned int ullm_aq4_wide_load_packed_byte(uint4 words, unsigned int byte_index) {
    const unsigned int word = byte_index < 4u ? words.x :
        (byte_index < 8u ? words.y : (byte_index < 12u ? words.z : words.w));
    return (word >> (8u * (byte_index & 3u))) & 0xffu;
}
__device__ float ullm_aq4_matvec_thread_sum(
    const unsigned char *indices, const unsigned char *scale_indices,
    const float *codebook, const float *scale_values, const float *input,
    unsigned long long scale_count, unsigned long long group_size, float tensor_scale,
    unsigned long long row, unsigned long long cols, unsigned int tid, unsigned int block_size) {
    const unsigned long long groups_per_row = cols / group_size;
    const unsigned long long row_byte_offset = (row * cols) >> 1;
    const unsigned int groups_per_chunk = group_size == 8ull ? 4u : 2u;
    float sum = 0.0f;
    for (unsigned long long chunk = tid; chunk < cols / 32ull; chunk += block_size) {
        const uint4 packed = *reinterpret_cast<const uint4 *>(indices + row_byte_offset + chunk * 16ull);
        const unsigned long long col_base = chunk * 32ull;
        const unsigned long long group_base = row * groups_per_row + col_base / group_size;
#pragma unroll
        for (unsigned int group_in_chunk = 0u; group_in_chunk < 4u; ++group_in_chunk) {
            if (group_in_chunk >= groups_per_chunk) continue;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group_base + group_in_chunk]);
            if (scale_index >= scale_count) continue;
            const unsigned int byte_start = group_in_chunk * static_cast<unsigned int>(group_size >> 1);
            const unsigned int byte_count = static_cast<unsigned int>(group_size >> 1);
            float raw_sum = 0.0f;
#pragma unroll
            for (unsigned int byte_offset = 0u; byte_offset < 8u; ++byte_offset) {
                if (byte_offset >= byte_count) continue;
                const unsigned int packed_byte_index = byte_start + byte_offset;
                const unsigned int packed_byte = ullm_aq4_wide_load_packed_byte(packed, packed_byte_index);
                const unsigned long long col = col_base + static_cast<unsigned long long>(packed_byte_index) * 2ull;
                raw_sum += codebook[packed_byte & 0x0fu] * input[col];
                raw_sum += codebook[(packed_byte >> 4) & 0x0fu] * input[col + 1ull];
            }
            sum += raw_sum * scale_values[scale_index] * tensor_scale;
        }
    }

    return sum;
}
__device__ void ullm_aq4_matvec_silu_mul_thread_sums(
    const unsigned char *gate_indices, const unsigned char *gate_scale_indices,
    const float *gate_codebook, const float *gate_scale_values, unsigned long long gate_scale_count,
    float gate_tensor_scale, const unsigned char *up_indices, const unsigned char *up_scale_indices,
    const float *up_codebook, const float *up_scale_values, unsigned long long up_scale_count,
    float up_tensor_scale, const float *input, unsigned long long group_size,
    unsigned long long row, unsigned long long cols, unsigned int tid, unsigned int block_size,
    float *gate_sum_output, float *up_sum_output) {
    const unsigned long long groups_per_row = cols / group_size;
    const unsigned long long row_byte_offset = (row * cols) >> 1;
    const unsigned int groups_per_chunk = group_size == 8ull ? 4u : 2u;
    float gate_sum = 0.0f, up_sum = 0.0f;
    for (unsigned long long chunk = tid; chunk < cols / 32ull; chunk += block_size) {
        const uint4 gate_packed = *reinterpret_cast<const uint4 *>(gate_indices + row_byte_offset + chunk * 16ull);
        const uint4 up_packed = *reinterpret_cast<const uint4 *>(up_indices + row_byte_offset + chunk * 16ull);
        const unsigned long long col_base = chunk * 32ull;
        const unsigned long long group_base = row * groups_per_row + col_base / group_size;
#pragma unroll
        for (unsigned int group_in_chunk = 0u; group_in_chunk < 4u; ++group_in_chunk) {
            if (group_in_chunk >= groups_per_chunk) continue;
            const unsigned int gate_scale_index = static_cast<unsigned int>(gate_scale_indices[group_base + group_in_chunk]);
            const unsigned int up_scale_index = static_cast<unsigned int>(up_scale_indices[group_base + group_in_chunk]);
            const unsigned int byte_start = group_in_chunk * static_cast<unsigned int>(group_size >> 1);
            const unsigned int byte_count = static_cast<unsigned int>(group_size >> 1);
            float gate_raw_sum = 0.0f, up_raw_sum = 0.0f;
#pragma unroll
            for (unsigned int byte_offset = 0u; byte_offset < 8u; ++byte_offset) {
                if (byte_offset >= byte_count) continue;
                const unsigned int packed_byte_index = byte_start + byte_offset;
                const unsigned int gate_byte = ullm_aq4_wide_load_packed_byte(gate_packed, packed_byte_index);
                const unsigned int up_byte = ullm_aq4_wide_load_packed_byte(up_packed, packed_byte_index);
                const unsigned long long col = col_base + static_cast<unsigned long long>(packed_byte_index) * 2ull;
                gate_raw_sum += gate_codebook[gate_byte & 0x0fu] * input[col] + gate_codebook[(gate_byte >> 4) & 0x0fu] * input[col + 1ull];
                up_raw_sum += up_codebook[up_byte & 0x0fu] * input[col] + up_codebook[(up_byte >> 4) & 0x0fu] * input[col + 1ull];
            }
            if (gate_scale_index < gate_scale_count) gate_sum += gate_raw_sum * gate_scale_values[gate_scale_index] * gate_tensor_scale;
            if (up_scale_index < up_scale_count) up_sum += up_raw_sum * up_scale_values[up_scale_index] * up_tensor_scale;
        }
    }
    *gate_sum_output = gate_sum; *up_sum_output = up_sum;
}
)" + production;
    }

    // Production AQ4 matvec SiLU-mul body. The host ABI and HIPRTC entry-point name remain
    // unchanged, so package resident decode continues through its existing direct
    // aq4_matvec_silu_mul_f32 dispatch.
    static std::string aq4_matvec_silu_mul_kernel_source() {
        std::string production = aq4_matvec_silu_mul_scalar_reference_kernel_source();
        const std::string lds_setup = R"(    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float gate_partial[256];
    __shared__ float up_partial[256];
    float gate_sum = 0.0f;
    float up_sum = 0.0f;)";
        const std::string shuffle_setup = R"(    // RPB=8 maps each row's contiguous 32 threads to one gfx1201 wave32.
    // Gate and up keep independent accumulators, but neither needs cross-wave/LDS reduction.
    float gate_sum = 0.0f;
    float up_sum = 0.0f;)";
        const size_t setup_offset = production.find(lds_setup);
        if (setup_offset == std::string::npos) return "";
        production.replace(setup_offset, lds_setup.size(), shuffle_setup);
        const std::string lds_tree = R"(    gate_partial[tid] = gate_sum;
    up_partial[tid] = up_sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            gate_partial[partial_offset + lane] += gate_partial[partial_offset + lane + offset];
            up_partial[partial_offset + lane] += up_partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && row < rows) {
        float gate_value = gate_partial[partial_offset];
        float up_value = up_partial[partial_offset];)";
        const std::string shuffle_reduction = R"(    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down(gate_sum, offset, threads_per_row);
        up_sum += __shfl_down(up_sum, offset, threads_per_row);
    }
    if (lane == 0 && row < rows) {
        float gate_value = gate_sum;
        float up_value = up_sum;)";
        const size_t tree_offset = production.find(lds_tree);
        if (tree_offset == std::string::npos) return "";
        production.replace(tree_offset, lds_tree.size(), shuffle_reduction);
        return production;
    }

    // Keep the direct-only prototype ABI available for historical differential/timing comparison.
    // It compiles the same validated body under its original, isolated entry-point name.
    static std::string aq4_matvec_silu_mul_shuffle_prototype_kernel_source() {
        std::string prototype = aq4_matvec_silu_mul_kernel_source();
        const std::string production_name = "ullm_aq4_matvec_silu_mul_f32_kernel";
        const size_t name_offset = prototype.find(production_name);
        if (name_offset == std::string::npos) return "";
        prototype.replace(name_offset, production_name.size(),
            "ullm_aq4_matvec_silu_mul_shuffle_prototype_f32_kernel");
        return prototype;
    }

    static const char *aq4_matvec_gate_beta_kernel_source() {
        return R"(
__device__ float ullm_aq4_gate_beta_thread_sum(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size) {
    const unsigned long long row_offset = row * cols;
    float sum = 0.0f;
    if ((cols % group_size) == 0ull) {
        const unsigned long long groups_per_row = cols / group_size;
        for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
             group_in_row += block_size) {
            const unsigned long long group = row * groups_per_row + group_in_row;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index >= scale_count) {
                continue;
            }
            float raw_sum = 0.0f;
            const unsigned long long col_start = group_in_row * group_size;
            for (unsigned long long offset = 0; offset < group_size; ++offset) {
                const unsigned long long col = col_start + offset;
                const unsigned long long element = row_offset + col;
                const unsigned char packed = indices[element >> 1];
                const unsigned char codebook_index =
                    (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                raw_sum += codebook[codebook_index] * input[col];
            }
            sum += raw_sum * scale_values[scale_index] * tensor_scale;
        }
    } else {
        for (unsigned long long col = tid; col < cols; col += block_size) {
            const unsigned long long element = row_offset + col;
            const unsigned char packed = indices[element >> 1];
            const unsigned char codebook_index =
                (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
            const unsigned long long group = element / group_size;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index < scale_count) {
                const float value =
                    codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                sum += value * input[col];
            }
        }
    }
    return sum;
}

__device__ void ullm_aq4_gate_beta_thread_sums(
    const unsigned char *a_indices,
    const unsigned char *a_scale_indices,
    const float *a_codebook,
    const float *a_scale_values,
    unsigned long long a_scale_count,
    float a_tensor_scale,
    const unsigned char *b_indices,
    const unsigned char *b_scale_indices,
    const float *b_codebook,
    const float *b_scale_values,
    unsigned long long b_scale_count,
    float b_tensor_scale,
    const float *input,
    unsigned long long group_size,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size,
    float *a_sum_output,
    float *b_sum_output) {
    const unsigned long long row_offset = row * cols;
    const unsigned long long groups_per_row = cols / group_size;
    float a_sum = 0.0f;
    float b_sum = 0.0f;
    for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
         group_in_row += block_size) {
        const unsigned long long group = row * groups_per_row + group_in_row;
        const unsigned int a_scale_index = static_cast<unsigned int>(a_scale_indices[group]);
        const unsigned int b_scale_index = static_cast<unsigned int>(b_scale_indices[group]);
        float a_raw_sum = 0.0f;
        float b_raw_sum = 0.0f;
        const unsigned long long col_start = group_in_row * group_size;
        if (group_size == 16ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char a_packed = a_indices[packed_index];
                const unsigned char b_packed = b_indices[packed_index];
                a_raw_sum += a_codebook[a_packed & 0x0f] * input_low;
                b_raw_sum += b_codebook[b_packed & 0x0f] * input_low;
                a_raw_sum += a_codebook[(a_packed >> 4) & 0x0f] * input_high;
                b_raw_sum += b_codebook[(b_packed >> 4) & 0x0f] * input_high;
            }
        } else if (group_size == 8ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char a_packed = a_indices[packed_index];
                const unsigned char b_packed = b_indices[packed_index];
                a_raw_sum += a_codebook[a_packed & 0x0f] * input_low;
                b_raw_sum += b_codebook[b_packed & 0x0f] * input_low;
                a_raw_sum += a_codebook[(a_packed >> 4) & 0x0f] * input_high;
                b_raw_sum += b_codebook[(b_packed >> 4) & 0x0f] * input_high;
            }
        } else if ((group_size & 1ull) == 0ull) {
            for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char a_packed = a_indices[packed_index];
                const unsigned char b_packed = b_indices[packed_index];
                a_raw_sum += a_codebook[a_packed & 0x0f] * input_low;
                b_raw_sum += b_codebook[b_packed & 0x0f] * input_low;
                a_raw_sum += a_codebook[(a_packed >> 4) & 0x0f] * input_high;
                b_raw_sum += b_codebook[(b_packed >> 4) & 0x0f] * input_high;
            }
        } else {
            for (unsigned long long offset = 0; offset < group_size; ++offset) {
                const unsigned long long col = col_start + offset;
                const unsigned long long element = row_offset + col;
                const float input_value = input[col];
                const unsigned char a_packed = a_indices[element >> 1];
                const unsigned char a_codebook_index =
                    (element & 1ull) == 0ull ? (a_packed & 0x0f) :
                                               ((a_packed >> 4) & 0x0f);
                const unsigned char b_packed = b_indices[element >> 1];
                const unsigned char b_codebook_index =
                    (element & 1ull) == 0ull ? (b_packed & 0x0f) :
                                               ((b_packed >> 4) & 0x0f);
                a_raw_sum += a_codebook[a_codebook_index] * input_value;
                b_raw_sum += b_codebook[b_codebook_index] * input_value;
            }
        }
        if (a_scale_index < a_scale_count) {
            a_sum += a_raw_sum * a_scale_values[a_scale_index] * a_tensor_scale;
        }
        if (b_scale_index < b_scale_count) {
            b_sum += b_raw_sum * b_scale_values[b_scale_index] * b_tensor_scale;
        }
    }
    *a_sum_output = a_sum;
    *b_sum_output = b_sum;
}

extern "C" __global__ void ullm_aq4_matvec_gate_beta_f32_kernel(
    const unsigned char *a_indices,
    const unsigned char *a_scale_indices,
    const float *a_codebook,
    const float *a_scale_values,
    const float *a_row_scales,
    unsigned long long a_scale_count,
    unsigned long long a_group_size,
    float a_tensor_scale,
    unsigned long long a_row_scale_count,
    const unsigned char *b_indices,
    const unsigned char *b_scale_indices,
    const float *b_codebook,
    const float *b_scale_values,
    const float *b_row_scales,
    unsigned long long b_scale_count,
    unsigned long long b_group_size,
    float b_tensor_scale,
    unsigned long long b_row_scale_count,
    const float *input,
    const float *a_log,
    const float *dt_bias,
    unsigned long long heads,
    unsigned long long cols,
    float *gate_output,
    float *beta_output) {
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long head =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float a_partial[256];
    __shared__ float b_partial[256];
    float a_sum = 0.0f;
    float b_sum = 0.0f;
    if (head < heads) {
        if (a_group_size == b_group_size && (cols % a_group_size) == 0ull) {
            ullm_aq4_gate_beta_thread_sums(
                a_indices,
                a_scale_indices,
                a_codebook,
                a_scale_values,
                a_scale_count,
                a_tensor_scale,
                b_indices,
                b_scale_indices,
                b_codebook,
                b_scale_values,
                b_scale_count,
                b_tensor_scale,
                input,
                a_group_size,
                head,
                cols,
                lane,
                threads_per_row,
                &a_sum,
                &b_sum);
        } else {
            a_sum = ullm_aq4_gate_beta_thread_sum(
                a_indices,
                a_scale_indices,
                a_codebook,
                a_scale_values,
                input,
                a_scale_count,
                a_group_size,
                a_tensor_scale,
                head,
                cols,
                lane,
                threads_per_row);
            b_sum = ullm_aq4_gate_beta_thread_sum(
                b_indices,
                b_scale_indices,
                b_codebook,
                b_scale_values,
                input,
                b_scale_count,
                b_group_size,
                b_tensor_scale,
                head,
                cols,
                lane,
                threads_per_row);
        }
    }
    a_partial[tid] = a_sum;
    b_partial[tid] = b_sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            a_partial[partial_offset + lane] += a_partial[partial_offset + lane + offset];
            b_partial[partial_offset + lane] += b_partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && head < heads) {
        float a_value = a_partial[partial_offset];
        float b_value = b_partial[partial_offset];
        if (a_row_scales != nullptr && head < a_row_scale_count) {
            a_value *= a_row_scales[head];
        }
        if (b_row_scales != nullptr && head < b_row_scale_count) {
            b_value *= b_row_scales[head];
        }
        const float x = a_value + dt_bias[head];
        const float softplus = x <= 20.0f ? logf(1.0f + expf(x)) : x;
        gate_output[head] = -expf(a_log[head]) * softplus;
        beta_output[head] = 1.0f / (1.0f + expf(-b_value));
    }
}
)";
    }

#include "kernels/primitives/primitives_sources_part_0.inc"
#include "kernels/sq8_0/sq8_0_matvec_hiprtc.inc"
#include "kernels/sq8_1/sq8_1_matvec_hiprtc.inc"

#include "kernels/primitives/primitives_sources_part_a.inc"
#include "kernels/rope/rope_sources_part_a.inc"
#include "kernels/kv/qwen35_paged_kv_sources.inc"
#include "kernels/primitives/primitives_sources_part_b.inc"
#include "kernels/rope/rope_sources_part_b.inc"
#include "kernels/attention/attention_sources.inc"
#include "kernels/kv/kv_sources.inc"
#include "kernels/linear_attn/linear_attn_sources_part_b.inc"
