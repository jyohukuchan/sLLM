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

    static const char *matvec_kernel_source() {
        return R"(
extern "C" __global__ void ullm_matvec_f32_kernel(
    const float *matrix,
    const float *input,
    unsigned long long rows,
    unsigned long long cols,
    float *output) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum = 0.0f;
    if (row < rows) {
        const unsigned long long row_offset = static_cast<unsigned long long>(row) * cols;
        for (unsigned long long col = tid; col < cols; col += blockDim.x) {
            sum += matrix[row_offset + col] * input[col];
        }
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0 && row < rows) {
        output[row] = partial[0];
    }
}
)";
    }

#include "kernels/sq8_0/sq8_0_matvec_hiprtc.inc"
#include "kernels/sq8_1/sq8_1_matvec_hiprtc.inc"

    static const char *matvec_bf16_kernel_source() {
        return R"(
__device__ float ullm_bf16_to_f32(unsigned short value) {
    return __uint_as_float(static_cast<unsigned int>(value) << 16);
}

__device__ float ullm_bf16_pair_low_to_f32(unsigned int value) {
    return __uint_as_float((value & 0xffffu) << 16);
}

__device__ float ullm_bf16_pair_high_to_f32(unsigned int value) {
    return __uint_as_float((value & 0xffff0000u));
}

extern "C" __global__ void ullm_matvec_bf16_f32_kernel(
    const unsigned short *matrix,
    const float *input,
    unsigned long long rows,
    unsigned long long cols,
    float *output) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum = 0.0f;
    if (row < rows) {
        const unsigned long long row_offset = static_cast<unsigned long long>(row) * cols;
        if ((cols & 1ull) == 0ull) {
            const unsigned int *matrix_pairs =
                reinterpret_cast<const unsigned int *>(matrix + row_offset);
            const unsigned long long pair_cols = cols >> 1;
            for (unsigned long long pair_col = tid; pair_col < pair_cols;
                 pair_col += blockDim.x) {
                const unsigned int packed = matrix_pairs[pair_col];
                const unsigned long long col = pair_col << 1;
                sum += ullm_bf16_pair_low_to_f32(packed) * input[col];
                sum += ullm_bf16_pair_high_to_f32(packed) * input[col + 1ull];
            }
        } else {
            for (unsigned long long col = tid; col < cols; col += blockDim.x) {
                sum += ullm_bf16_to_f32(matrix[row_offset + col]) * input[col];
            }
        }
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0 && row < rows) {
        output[row] = partial[0];
    }
}
)";
    }

    static const char *bf16_row_kernel_source() {
        return R"(
__device__ float ullm_bf16_to_f32(unsigned short value) {
    return __uint_as_float(static_cast<unsigned int>(value) << 16);
}

extern "C" __global__ void ullm_bf16_row_f32_kernel(
    const unsigned short *matrix,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long row_index,
    float *output) {
    const unsigned int tid = threadIdx.x;
    if (row_index >= rows) {
        return;
    }
    const unsigned long long row_offset = row_index * cols;
    for (unsigned long long col = tid; col < cols; col += blockDim.x) {
        output[col] = ullm_bf16_to_f32(matrix[row_offset + col]);
    }
}
)";
    }

    // Superseded scalar-reduction RMSNorm reference. It is deliberately not passed to HIPRTC
    // for gfx1201: production keeps the same entry-point ABI but uses the validated wave32
    // shuffle reduction below. Retain this source for controlled rollback comparisons and for
    // generic HIPRTC compilations outside that exact architecture.
    [[maybe_unused]] static const char *rmsnorm_scalar_reference_kernel_source() {
        return R"(
extern "C" __global__ void ullm_rmsnorm_f32_kernel(
    const float *input,
    const float *weight,
    unsigned long long elements,
    float epsilon,
    float *output) {
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum = 0.0f;
    for (unsigned long long index = tid; index < elements; index += blockDim.x) {
        const float value = input[index];
        sum += value * value;
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(elements) + epsilon);
    for (unsigned long long index = tid; index < elements; index += blockDim.x) {
        output[index] = input[index] * inv_rms * weight[index];
    }
}
)";
    }

    // Production RMSNorm body. The host ABI and HIPRTC entry-point name remain unchanged, so
    // Qwen3.5 decode continues through its existing RMSNorm dispatch.
    static std::string rmsnorm_kernel_source() {
        std::string production = rmsnorm_scalar_reference_kernel_source();
        const std::string lds_setup = R"(    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum = 0.0f;)";
        const std::string shuffle_setup = R"(    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % warpSize;
    const unsigned int wave = tid / warpSize;
    __shared__ float wave_partials[8];
    float sum = 0.0f;)";
        const size_t setup_offset = production.find(lds_setup);
        if (setup_offset == std::string::npos) return "";
        production.replace(setup_offset, lds_setup.size(), shuffle_setup);

        const std::string lds_reduction = R"(    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(elements) + epsilon);)";
        const std::string shuffle_reduction = R"(    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, warpSize);
    }
    if (lane == 0u) {
        wave_partials[wave] = sum;
    }
    __syncthreads();

    if (wave == 0u) {
        const unsigned int wave_count = (blockDim.x + warpSize - 1u) / warpSize;
        sum = lane < wave_count ? wave_partials[lane] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset, warpSize);
        }
        if (lane == 0u) {
            wave_partials[0] = sum;
        }
    }
    __syncthreads();
    const float inv_rms = rsqrtf(wave_partials[0] / static_cast<float>(elements) + epsilon);)";
        const size_t reduction_offset = production.find(lds_reduction);
        if (reduction_offset == std::string::npos) return "";
        production.replace(reduction_offset, lds_reduction.size(), shuffle_reduction);
        return production;
    }

    // Keep the direct-only prototype ABI available for historical differential/timing comparison.
    // It compiles the same validated body under its original, isolated entry-point name.
    static std::string rmsnorm_shuffle_prototype_kernel_source() {
        std::string prototype = rmsnorm_kernel_source();
        const std::string production_name = "ullm_rmsnorm_f32_kernel";
        const size_t name_offset = prototype.find(production_name);
        if (name_offset == std::string::npos) return "";
        prototype.replace(
            name_offset,
            production_name.size(),
            "ullm_rmsnorm_shuffle_prototype_f32_kernel");
        return prototype;
    }

    static std::string rmsnorm_kernel_source_for_arch(const std::string &arch) {
        return arch == "gfx1201" ? rmsnorm_kernel_source()
                                 : rmsnorm_scalar_reference_kernel_source();
    }

    static const char *top1_kernel_source() {
        return R"(
extern "C" __global__ void ullm_top1_f32_kernel(
    const float *input,
    unsigned long long elements,
    float *partial_values,
    unsigned int *partial_indices) {
    const unsigned int tid = threadIdx.x;
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + tid;
    __shared__ float values[256];
    __shared__ unsigned int indices[256];
    float value = -3.4028234663852886e38f;
    unsigned int token_index = 0xffffffffu;
    if (index < elements) {
        value = input[index];
        if (!(value == value)) {
            value = -3.4028234663852886e38f;
        }
        token_index = static_cast<unsigned int>(index);
    }
    values[tid] = value;
    indices[tid] = token_index;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            const float right_value = values[tid + offset];
            const unsigned int right_index = indices[tid + offset];
            const float left_value = values[tid];
            const unsigned int left_index = indices[tid];
            if (right_value > left_value ||
                (right_value == left_value && right_index < left_index)) {
                values[tid] = right_value;
                indices[tid] = right_index;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_values[blockIdx.x] = values[0];
        partial_indices[blockIdx.x] = indices[0];
    }
}
)";
    }

    static const char *top1_pairs_kernel_source() {
        return R"(
extern "C" __global__ void ullm_top1_pairs_f32_kernel(
    const float *input_values,
    const unsigned int *input_indices,
    unsigned long long elements,
    float *partial_values,
    unsigned int *partial_indices) {
    const unsigned int tid = threadIdx.x;
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + tid;
    __shared__ float values[256];
    __shared__ unsigned int indices[256];
    float value = -3.4028234663852886e38f;
    unsigned int token_index = 0xffffffffu;
    if (index < elements) {
        value = input_values[index];
        if (!(value == value)) {
            value = -3.4028234663852886e38f;
        }
        token_index = input_indices[index];
    }
    values[tid] = value;
    indices[tid] = token_index;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            const float right_value = values[tid + offset];
            const unsigned int right_index = indices[tid + offset];
            const float left_value = values[tid];
            const unsigned int left_index = indices[tid];
            if (right_value > left_value ||
                (right_value == left_value && right_index < left_index)) {
                values[tid] = right_value;
                indices[tid] = right_index;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_values[blockIdx.x] = values[0];
        partial_indices[blockIdx.x] = indices[0];
    }
}
)";
    }

    static const char *segmented_rmsnorm_kernel_source() {
        return R"(
extern "C" __global__ void ullm_segmented_rmsnorm_f32_kernel(
    const float *input,
    const float *weight,
    unsigned long long segments,
    unsigned long long segment_size,
    float epsilon,
    float *output) {
    const unsigned long long segment = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum_squares = 0.0f;
    if (segment < segments) {
        const unsigned long long base = segment * segment_size;
        for (unsigned long long dim = tid; dim < segment_size; dim += blockDim.x) {
            const float value = input[base + dim];
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
    if (segment >= segments) {
        return;
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(segment_size) + epsilon);
    const unsigned long long base = segment * segment_size;
    for (unsigned long long dim = tid; dim < segment_size; dim += blockDim.x) {
        output[base + dim] = input[base + dim] * inv_rms * weight[dim];
    }
}
)";
    }

    // Superseded scalar-reduction segmented RMSNorm SiLU-mul reference. It is deliberately not
    // passed to HIPRTC for gfx1201: production keeps the same entry-point ABI but uses the
    // validated wave32 shuffle reduction below. Retain this source for controlled rollback
    // comparisons and for generic HIPRTC compilations outside that exact architecture.
    [[maybe_unused]] static const char *segmented_rmsnorm_silu_mul_scalar_reference_kernel_source() {
        return R"(
extern "C" __global__ void ullm_segmented_rmsnorm_silu_mul_f32_kernel(
    const float *input,
    const float *weight,
    const float *gate,
    unsigned long long segments,
    unsigned long long segment_size,
    float epsilon,
    float *output) {
    const unsigned long long segment = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum_squares = 0.0f;
    if (segment < segments) {
        const unsigned long long base = segment * segment_size;
        for (unsigned long long dim = tid; dim < segment_size; dim += blockDim.x) {
            const float value = input[base + dim];
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
    if (segment >= segments) {
        return;
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(segment_size) + epsilon);
    const unsigned long long base = segment * segment_size;
    for (unsigned long long dim = tid; dim < segment_size; dim += blockDim.x) {
        const unsigned long long index = base + dim;
        const float gate_value = gate[index];
        const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
        const float normed = input[index] * inv_rms * weight[dim];
        output[index] = gate_value * sigmoid * normed;
    }
}
)";
    }

    // Production segmented RMSNorm SiLU-mul body. The host ABI and HIPRTC entry-point name
    // remain unchanged, so Qwen3.5 decode continues through its existing linear-attention post
    // dispatch.
    static std::string segmented_rmsnorm_silu_mul_kernel_source() {
        std::string production = segmented_rmsnorm_silu_mul_scalar_reference_kernel_source();
        const std::string lds_setup = R"(    const unsigned long long segment = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum_squares = 0.0f;)";
        const std::string shuffle_setup = R"(    const unsigned long long segment = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % warpSize;
    const unsigned int wave = tid / warpSize;
    // Eight wave32 leaders replace the 256-thread LDS tree.
    __shared__ float wave_partials[8];
    float sum_squares = 0.0f;)";
        const size_t setup_offset = production.find(lds_setup);
        if (setup_offset == std::string::npos) return "";
        production.replace(setup_offset, lds_setup.size(), shuffle_setup);

        const std::string lds_reduction = R"(    partial[tid] = sum_squares;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    if (segment >= segments) {
        return;
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(segment_size) + epsilon);)";
        const std::string shuffle_reduction = R"(    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum_squares += __shfl_down(sum_squares, offset, warpSize);
    }
    if (lane == 0u) {
        wave_partials[wave] = sum_squares;
    }
    __syncthreads();

    if (wave == 0u) {
        const unsigned int wave_count = (blockDim.x + warpSize - 1u) / warpSize;
        sum_squares = lane < wave_count ? wave_partials[lane] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum_squares += __shfl_down(sum_squares, offset, warpSize);
        }
        if (lane == 0u) {
            wave_partials[0] = sum_squares;
        }
    }
    __syncthreads();
    if (segment >= segments) {
        return;
    }
    const float inv_rms =
        rsqrtf(wave_partials[0] / static_cast<float>(segment_size) + epsilon);)";
        const size_t reduction_offset = production.find(lds_reduction);
        if (reduction_offset == std::string::npos) return "";
        production.replace(reduction_offset, lds_reduction.size(), shuffle_reduction);
        return production;
    }

    // Keep the direct-only prototype ABI available for historical differential/timing comparison.
    // It compiles the same validated body under its original, isolated entry-point name.
    static std::string segmented_rmsnorm_silu_mul_shuffle_prototype_kernel_source() {
        std::string prototype = segmented_rmsnorm_silu_mul_kernel_source();
        const std::string production_name = "ullm_segmented_rmsnorm_silu_mul_f32_kernel";
        const size_t name_offset = prototype.find(production_name);
        if (name_offset == std::string::npos) return "";
        prototype.replace(
            name_offset,
            production_name.size(),
            "ullm_segmented_rmsnorm_silu_mul_shuffle_prototype_f32_kernel");
        return prototype;
    }

    static std::string segmented_rmsnorm_silu_mul_kernel_source_for_arch(const std::string &arch) {
        return arch == "gfx1201" ? segmented_rmsnorm_silu_mul_kernel_source()
                                 : segmented_rmsnorm_silu_mul_scalar_reference_kernel_source();
    }

    static const char *silu_mul_kernel_source() {
        return R"(
extern "C" __global__ void ullm_silu_mul_f32_kernel(
    const float *gate,
    const float *up,
    unsigned long long elements,
    float *output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const float gate_value = gate[index];
    const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
    output[index] = gate_value * sigmoid * up[index];
}
)";
    }

    static const char *sigmoid_mul_kernel_source() {
        return R"(
extern "C" __global__ void ullm_sigmoid_mul_f32_kernel(
    const float *gate,
    const float *input,
    unsigned long long elements,
    float *output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const float gate_value = gate[index];
    const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
    output[index] = sigmoid * input[index];
}
)";
    }

    static const char *qwen35_split_q_gate_kernel_source() {
        return R"(
extern "C" __global__ void ullm_qwen35_split_q_gate_f32_kernel(
    const float *projected,
    unsigned long long q_heads,
    unsigned long long head_dim,
    float *query_output,
    float *gate_output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long elements = q_heads * head_dim;
    if (index >= elements) {
        return;
    }
    const unsigned long long dim = index % head_dim;
    const unsigned long long head = index / head_dim;
    const unsigned long long source_base = head * 2ULL * head_dim;
    query_output[index] = projected[source_base + dim];
    gate_output[index] = projected[source_base + head_dim + dim];
}
)";
    }

    static const char *qwen35_qk_norm_rope_kernel_source() {
        return R"(
extern "C" __global__ void ullm_qwen35_qk_norm_rope_f32_kernel(
    const float *q_projected,
    const float *k_projected,
    const float *q_weight,
    const float *k_weight,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long rotary_dim,
    unsigned long long position_offset,
    float rope_base,
    float epsilon,
    float *q_gate_output,
    float *q_rope_output,
    float *k_rope_output) {
    const unsigned long long segment = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned long long total_segments = q_heads + kv_heads;
    __shared__ float partial[256];
    float sum_squares = 0.0f;
    if (segment < q_heads) {
        const unsigned long long source_base = segment * 2ull * head_dim;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            const float value = q_projected[source_base + dim];
            q_gate_output[segment * head_dim + dim] = q_projected[source_base + head_dim + dim];
            sum_squares += value * value;
        }
    } else if (segment < total_segments) {
        const unsigned long long k_segment = segment - q_heads;
        const unsigned long long source_base = k_segment * head_dim;
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
    if (segment >= total_segments) {
        return;
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
        const unsigned long long k_segment = segment - q_heads;
        const unsigned long long source_base = k_segment * head_dim;
        const unsigned long long output_base = k_segment * head_dim;
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
            k_rope_output[output_base + pair_dim] = first * c - second * s;
            k_rope_output[output_base + second_dim] = second * c + first * s;
        }
        for (unsigned long long dim = rotary_dim + tid; dim < head_dim; dim += blockDim.x) {
            k_rope_output[output_base + dim] =
                k_projected[source_base + dim] * inv_rms * k_weight[dim];
        }
    }
}
)";
    }

    static const char *qwen35_qk_norm_rope_batch_kernel_source() {
        return R"(
extern "C" __global__ void ullm_qwen35_qk_norm_rope_batch_f32_kernel(
    const float *q_projected,
    const float *k_projected,
    const float *q_weight,
    const float *k_weight,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long sequence_len,
    unsigned long long head_dim,
    unsigned long long rotary_dim,
    unsigned long long position_offset,
    float rope_base,
    float epsilon,
    float *q_gate_output,
    float *q_rope_output,
    float *k_rope_output) {
    const unsigned long long total_segments_per_token = q_heads + kv_heads;
    const unsigned long long global_segment = blockIdx.x;
    const unsigned long long token = global_segment / total_segments_per_token;
    const unsigned long long segment = global_segment - token * total_segments_per_token;
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum_squares = 0.0f;
    if (token < sequence_len && segment < q_heads) {
        const unsigned long long q_projected_stride = q_heads * 2ull * head_dim;
        const unsigned long long source_base =
            token * q_projected_stride + segment * 2ull * head_dim;
        const unsigned long long q_output_stride = q_heads * head_dim;
        const unsigned long long output_base = token * q_output_stride + segment * head_dim;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            const float value = q_projected[source_base + dim];
            q_gate_output[output_base + dim] = q_projected[source_base + head_dim + dim];
            sum_squares += value * value;
        }
    } else if (token < sequence_len && segment < total_segments_per_token) {
        const unsigned long long k_segment = segment - q_heads;
        const unsigned long long k_projected_stride = kv_heads * head_dim;
        const unsigned long long source_base =
            token * k_projected_stride + k_segment * head_dim;
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
    if (token >= sequence_len || segment >= total_segments_per_token) {
        return;
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + epsilon);
    const unsigned long long half = rotary_dim >> 1;
    const float position = static_cast<float>(position_offset + token);
    if (segment < q_heads) {
        const unsigned long long q_projected_stride = q_heads * 2ull * head_dim;
        const unsigned long long source_base =
            token * q_projected_stride + segment * 2ull * head_dim;
        const unsigned long long q_output_stride = q_heads * head_dim;
        const unsigned long long output_base = token * q_output_stride + segment * head_dim;
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
        const unsigned long long k_segment = segment - q_heads;
        const unsigned long long k_projected_stride = kv_heads * head_dim;
        const unsigned long long source_base =
            token * k_projected_stride + k_segment * head_dim;
        const unsigned long long k_output_stride = kv_heads * head_dim;
        const unsigned long long output_base = token * k_output_stride + k_segment * head_dim;
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
            k_rope_output[output_base + pair_dim] = first * c - second * s;
            k_rope_output[output_base + second_dim] = second * c + first * s;
        }
        for (unsigned long long dim = rotary_dim + tid; dim < head_dim; dim += blockDim.x) {
            k_rope_output[output_base + dim] =
                k_projected[source_base + dim] * inv_rms * k_weight[dim];
        }
    }
}
)";
    }

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

    static const char *add_kernel_source() {
        return R"(
extern "C" __global__ void ullm_add_f32_kernel(
    const float *lhs,
    const float *rhs,
    unsigned long long elements,
    float *output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    output[index] = lhs[index] + rhs[index];
}
)";
    }

    static const char *rope_kernel_source() {
        return R"(
extern "C" __global__ void ullm_rope_f32_kernel(
    const float *input,
    unsigned long long sequence_len,
    unsigned long long heads,
    unsigned long long head_dim,
    unsigned long long rotary_dim,
    unsigned long long position_offset,
    float rope_base,
    float *output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long half = rotary_dim >> 1;
    const unsigned long long work_dim = half + (head_dim - rotary_dim);
    const unsigned long long work_items = sequence_len * heads * work_dim;
    if (index >= work_items) {
        return;
    }
    const unsigned long long local_dim = index % work_dim;
    const unsigned long long head_index = index / work_dim;
    const unsigned long long base = head_index * head_dim;
    if (local_dim >= half) {
        const unsigned long long dim = rotary_dim + (local_dim - half);
        output[base + dim] = input[base + dim];
        return;
    }
    const unsigned long long pair_dim = local_dim;
    const unsigned long long timestep = head_index / heads;
    const float position = static_cast<float>(position_offset + timestep);
    const float exponent =
        (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
    const float theta = position / powf(rope_base, exponent);
    const float c = cosf(theta);
    const float s = sinf(theta);
    const float first = input[base + pair_dim];
    const float second = input[base + half + pair_dim];
    output[base + pair_dim] = first * c - second * s;
    output[base + half + pair_dim] = second * c + first * s;
}
)";
    }

    static const char *causal_attn_kernel_source() {
        return R"(
extern "C" __global__ void ullm_causal_attn_f32_kernel(
    const float *q,
    const float *k,
    const float *v,
    unsigned long long sequence_len,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float *output) {
    const unsigned long long q_head_index = static_cast<unsigned long long>(blockIdx.x);
    const unsigned long long q_head_elements = sequence_len * q_heads;
    if (q_head_index >= q_head_elements) {
        return;
    }
    const unsigned long long q_head = q_head_index % q_heads;
    const unsigned long long timestep = q_head_index / q_heads;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_base = (timestep * q_heads + q_head) * head_dim;
    const unsigned int tid = threadIdx.x;

    __shared__ float reduce[256];
    __shared__ float shared_score;
    __shared__ float shared_weight;
    __shared__ float shared_max_score;
    __shared__ float shared_denominator;
    __shared__ float shared_prev_scale;

    if (value_dim <= static_cast<unsigned long long>(blockDim.x)) {
        const unsigned long long value = tid;
        const bool value_active = value < value_dim;
        float max_score = -3.4028234663852886e38f;
        float denominator = 0.0f;
        float weighted = 0.0f;
        for (unsigned long long source_timestep = 0; source_timestep <= timestep; ++source_timestep) {
            const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
                partial += q[q_base + dim] * k[k_base + dim];
            }
            reduce[tid] = partial;
            __syncthreads();
            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const float score = reduce[0] * softmax_scale;
                const float new_max_score = score > max_score ? score : max_score;
                shared_prev_scale =
                    max_score <= -3.4028234663852886e38f ? 0.0f : expf(max_score - new_max_score);
                shared_weight = expf(score - new_max_score);
                shared_score = new_max_score;
            }
            __syncthreads();
            if (value_active) {
                const unsigned long long v_index =
                    (source_timestep * kv_heads + kv_head) * value_dim + value;
                weighted = weighted * shared_prev_scale + shared_weight * v[v_index];
            }
            denominator = denominator * shared_prev_scale + shared_weight;
            max_score = shared_score;
            __syncthreads();
        }
        if (value_active) {
            const unsigned long long output_base = q_head_index * value_dim;
            output[output_base + value] = weighted / denominator;
        }
        return;
    }

    float max_score = -3.4028234663852886e38f;
    for (unsigned long long source_timestep = 0; source_timestep <= timestep; ++source_timestep) {
        const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
        float partial = 0.0f;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            partial += q[q_base + dim] * k[k_base + dim];
        }
        reduce[tid] = partial;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            shared_score = reduce[0] * softmax_scale;
        }
        __syncthreads();
        const float score = shared_score;
        max_score = score > max_score ? score : max_score;
    }
    if (tid == 0) {
        shared_max_score = max_score;
    }
    __syncthreads();
    max_score = shared_max_score;

    float denominator = 0.0f;
    for (unsigned long long source_timestep = 0; source_timestep <= timestep; ++source_timestep) {
        const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
        float partial = 0.0f;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            partial += q[q_base + dim] * k[k_base + dim];
        }
        reduce[tid] = partial;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            shared_weight = expf(reduce[0] * softmax_scale - max_score);
        }
        __syncthreads();
        denominator += shared_weight;
    }
    if (tid == 0) {
        shared_denominator = denominator;
    }
    __syncthreads();
    denominator = shared_denominator;

    const unsigned long long output_base = q_head_index * value_dim;
    for (unsigned long long value_base = 0; value_base < value_dim; value_base += blockDim.x) {
        const unsigned long long value = value_base + tid;
        const bool value_active = value < value_dim;
        float weighted = 0.0f;
        for (unsigned long long source_timestep = 0; source_timestep <= timestep; ++source_timestep) {
            const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
                partial += q[q_base + dim] * k[k_base + dim];
            }
            reduce[tid] = partial;
            __syncthreads();
            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                shared_weight = expf(reduce[0] * softmax_scale - max_score);
            }
            __syncthreads();
            const float weight = shared_weight;
            if (value_active) {
                const unsigned long long v_index =
                    (source_timestep * kv_heads + kv_head) * value_dim + value;
                weighted += weight * v[v_index];
            }
        }
        if (value_active) {
            output[output_base + value] = weighted / denominator;
        }
    }
}
)";
    }

    static const char *causal_attn_flash2_f32_kernel_source() {
        return R"(
extern "C" __global__ void ullm_causal_attn_f32_flash2_kernel(
    const float *q,
    const float *k,
    const float *v,
    unsigned long long sequence_len,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float *output) {
    constexpr unsigned int kTileTokens = 64u;
    const unsigned long long q_head_index = static_cast<unsigned long long>(blockIdx.x);
    const unsigned long long q_head_elements = sequence_len * q_heads;
    if (q_head_index >= q_head_elements) {
        return;
    }
    const unsigned int tid = threadIdx.x;
    if (value_dim > static_cast<unsigned long long>(blockDim.x)) {
        return;
    }

    const unsigned long long q_head = q_head_index % q_heads;
    const unsigned long long timestep = q_head_index / q_heads;
    const unsigned long long cache_len = timestep + 1ull;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_base = (timestep * q_heads + q_head) * head_dim;
    const unsigned long long value = static_cast<unsigned long long>(tid);
    const bool value_active = value < value_dim;

    __shared__ float reduce[256];
    __shared__ float scores[kTileTokens];
    __shared__ float shared_tile_max;
    __shared__ float shared_tile_sum;
    __shared__ float shared_m_new;
    __shared__ float shared_alpha;

    float max_score = -3.4028234663852886e38f;
    float denominator = 0.0f;
    float weighted = 0.0f;

    for (unsigned long long tile_start = 0; tile_start < cache_len; tile_start += kTileTokens) {
        const unsigned long long remaining = cache_len - tile_start;
        const unsigned int tile_count =
            remaining < static_cast<unsigned long long>(kTileTokens) ?
                static_cast<unsigned int>(remaining) :
                kTileTokens;

        for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
            const unsigned long long source_timestep =
                tile_start + static_cast<unsigned long long>(tile_offset);
            const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
                partial += q[q_base + dim] * k[k_base + dim];
            }
            reduce[tid] = partial;
            __syncthreads();
            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0u) {
                scores[tile_offset] = reduce[0] * softmax_scale;
            }
            __syncthreads();
        }

        float local_max = -3.4028234663852886e38f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float score = scores[tile_offset];
            local_max = score > local_max ? score : local_max;
        }
        reduce[tid] = local_max;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] = reduce[tid] > reduce[tid + stride] ? reduce[tid] : reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            shared_tile_max = reduce[0];
            const float m_new = max_score > shared_tile_max ? max_score : shared_tile_max;
            shared_m_new = m_new;
            shared_alpha = max_score <= -3.0e38f ? 0.0f : expf(max_score - m_new);
        }
        __syncthreads();

        const float m_new = shared_m_new;
        float local_sum = 0.0f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float weight = expf(scores[tile_offset] - m_new);
            scores[tile_offset] = weight;
            local_sum += weight;
        }
        reduce[tid] = local_sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            shared_tile_sum = reduce[0];
        }
        __syncthreads();

        if (value_active) {
            float tile_weighted = 0.0f;
            for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
                const unsigned long long source_timestep =
                    tile_start + static_cast<unsigned long long>(tile_offset);
                const unsigned long long v_index =
                    (source_timestep * kv_heads + kv_head) * value_dim + value;
                tile_weighted += scores[tile_offset] * v[v_index];
            }
            weighted = weighted * shared_alpha + tile_weighted;
        }
        denominator = denominator * shared_alpha + shared_tile_sum;
        max_score = m_new;
        __syncthreads();
    }

    if (value_active) {
        const unsigned long long output_base = q_head_index * value_dim;
        output[output_base + value] = weighted / denominator;
    }
}
)";
    }

    static const char *causal_attn_batch_kernel_source() {
        return R"(
extern "C" __global__ void ullm_causal_attn_batch_f32_kernel(
    const float *q,
    const float *k,
    const float *v,
    unsigned long long batch_count,
    unsigned long long sequence_len,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float *output) {
    const unsigned long long global_q_head_index = static_cast<unsigned long long>(blockIdx.x);
    const unsigned long long q_head_elements_per_batch = sequence_len * q_heads;
    const unsigned long long q_head_elements = batch_count * q_head_elements_per_batch;
    if (global_q_head_index >= q_head_elements) {
        return;
    }
    const unsigned long long batch = global_q_head_index / q_head_elements_per_batch;
    const unsigned long long q_head_index =
        global_q_head_index - batch * q_head_elements_per_batch;
    const unsigned long long q_head = q_head_index % q_heads;
    const unsigned long long timestep = q_head_index / q_heads;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_batch_base = batch * q_head_elements_per_batch * head_dim;
    const unsigned long long k_batch_base = batch * sequence_len * kv_heads * head_dim;
    const unsigned long long v_batch_base = batch * sequence_len * kv_heads * value_dim;
    const unsigned long long output_batch_base = batch * q_head_elements_per_batch * value_dim;
    const unsigned long long q_base = q_batch_base + (timestep * q_heads + q_head) * head_dim;
    const unsigned int tid = threadIdx.x;

    __shared__ float reduce[256];
    __shared__ float shared_score;
    __shared__ float shared_weight;
    __shared__ float shared_max_score;
    __shared__ float shared_denominator;
    __shared__ float shared_prev_scale;

    if (value_dim <= static_cast<unsigned long long>(blockDim.x)) {
        const unsigned long long value = tid;
        const bool value_active = value < value_dim;
        float max_score = -3.4028234663852886e38f;
        float denominator = 0.0f;
        float weighted = 0.0f;
        for (unsigned long long source_timestep = 0; source_timestep <= timestep; ++source_timestep) {
            const unsigned long long k_base =
                k_batch_base + (source_timestep * kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
                partial += q[q_base + dim] * k[k_base + dim];
            }
            reduce[tid] = partial;
            __syncthreads();
            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const float score = reduce[0] * softmax_scale;
                const float new_max_score = score > max_score ? score : max_score;
                shared_prev_scale =
                    max_score <= -3.4028234663852886e38f ? 0.0f : expf(max_score - new_max_score);
                shared_weight = expf(score - new_max_score);
                shared_score = new_max_score;
            }
            __syncthreads();
            if (value_active) {
                const unsigned long long v_index =
                    v_batch_base + (source_timestep * kv_heads + kv_head) * value_dim + value;
                weighted = weighted * shared_prev_scale + shared_weight * v[v_index];
            }
            denominator = denominator * shared_prev_scale + shared_weight;
            max_score = shared_score;
            __syncthreads();
        }
        if (value_active) {
            const unsigned long long output_base = output_batch_base + q_head_index * value_dim;
            output[output_base + value] = weighted / denominator;
        }
        return;
    }

    float max_score = -3.4028234663852886e38f;
    for (unsigned long long source_timestep = 0; source_timestep <= timestep; ++source_timestep) {
        const unsigned long long k_base =
            k_batch_base + (source_timestep * kv_heads + kv_head) * head_dim;
        float partial = 0.0f;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            partial += q[q_base + dim] * k[k_base + dim];
        }
        reduce[tid] = partial;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            shared_score = reduce[0] * softmax_scale;
        }
        __syncthreads();
        const float score = shared_score;
        max_score = score > max_score ? score : max_score;
    }
    if (tid == 0) {
        shared_max_score = max_score;
    }
    __syncthreads();
    max_score = shared_max_score;

    float denominator = 0.0f;
    for (unsigned long long source_timestep = 0; source_timestep <= timestep; ++source_timestep) {
        const unsigned long long k_base =
            k_batch_base + (source_timestep * kv_heads + kv_head) * head_dim;
        float partial = 0.0f;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            partial += q[q_base + dim] * k[k_base + dim];
        }
        reduce[tid] = partial;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            shared_weight = expf(reduce[0] * softmax_scale - max_score);
        }
        __syncthreads();
        denominator += shared_weight;
    }
    if (tid == 0) {
        shared_denominator = denominator;
    }
    __syncthreads();
    denominator = shared_denominator;

    const unsigned long long output_base = output_batch_base + q_head_index * value_dim;
    for (unsigned long long value_base = 0; value_base < value_dim; value_base += blockDim.x) {
        const unsigned long long value = value_base + tid;
        const bool value_active = value < value_dim;
        float weighted = 0.0f;
        for (unsigned long long source_timestep = 0; source_timestep <= timestep; ++source_timestep) {
            const unsigned long long k_base =
                k_batch_base + (source_timestep * kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
                partial += q[q_base + dim] * k[k_base + dim];
            }
            reduce[tid] = partial;
            __syncthreads();
            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                shared_weight = expf(reduce[0] * softmax_scale - max_score);
            }
            __syncthreads();
            const float weight = shared_weight;
            if (value_active) {
                const unsigned long long v_index =
                    v_batch_base + (source_timestep * kv_heads + kv_head) * value_dim + value;
                weighted += weight * v[v_index];
            }
        }
        if (value_active) {
            output[output_base + value] = weighted / denominator;
        }
    }
}
)";
    }

    static const char *causal_attn_batch_flash2_f32_kernel_source() {
        return R"(
extern "C" __global__ void ullm_causal_attn_batch_f32_flash2_kernel(
    const float *q,
    const float *k,
    const float *v,
    unsigned long long batch_count,
    unsigned long long sequence_len,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float *output) {
    constexpr unsigned int kTileTokens = 64u;
    const unsigned long long global_q_head_index = static_cast<unsigned long long>(blockIdx.x);
    const unsigned long long q_head_elements_per_batch = sequence_len * q_heads;
    const unsigned long long q_head_elements = batch_count * q_head_elements_per_batch;
    if (global_q_head_index >= q_head_elements) {
        return;
    }
    const unsigned int tid = threadIdx.x;
    if (value_dim > static_cast<unsigned long long>(blockDim.x)) {
        return;
    }

    const unsigned long long batch = global_q_head_index / q_head_elements_per_batch;
    const unsigned long long q_head_index =
        global_q_head_index - batch * q_head_elements_per_batch;
    const unsigned long long q_head = q_head_index % q_heads;
    const unsigned long long timestep = q_head_index / q_heads;
    const unsigned long long cache_len = timestep + 1ull;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_batch_base = batch * q_head_elements_per_batch * head_dim;
    const unsigned long long k_batch_base = batch * sequence_len * kv_heads * head_dim;
    const unsigned long long v_batch_base = batch * sequence_len * kv_heads * value_dim;
    const unsigned long long output_batch_base = batch * q_head_elements_per_batch * value_dim;
    const unsigned long long q_base = q_batch_base + (timestep * q_heads + q_head) * head_dim;
    const unsigned long long value = static_cast<unsigned long long>(tid);
    const bool value_active = value < value_dim;

    __shared__ float reduce[256];
    __shared__ float scores[kTileTokens];
    __shared__ float shared_tile_max;
    __shared__ float shared_tile_sum;
    __shared__ float shared_m_new;
    __shared__ float shared_alpha;

    float max_score = -3.4028234663852886e38f;
    float denominator = 0.0f;
    float weighted = 0.0f;

    for (unsigned long long tile_start = 0; tile_start < cache_len; tile_start += kTileTokens) {
        const unsigned long long remaining = cache_len - tile_start;
        const unsigned int tile_count =
            remaining < static_cast<unsigned long long>(kTileTokens) ?
                static_cast<unsigned int>(remaining) :
                kTileTokens;

        for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
            const unsigned long long source_timestep =
                tile_start + static_cast<unsigned long long>(tile_offset);
            const unsigned long long k_base =
                k_batch_base + (source_timestep * kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
                partial += q[q_base + dim] * k[k_base + dim];
            }
            reduce[tid] = partial;
            __syncthreads();
            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0u) {
                scores[tile_offset] = reduce[0] * softmax_scale;
            }
            __syncthreads();
        }

        float local_max = -3.4028234663852886e38f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float score = scores[tile_offset];
            local_max = score > local_max ? score : local_max;
        }
        reduce[tid] = local_max;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] = reduce[tid] > reduce[tid + stride] ? reduce[tid] : reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            shared_tile_max = reduce[0];
            const float m_new = max_score > shared_tile_max ? max_score : shared_tile_max;
            shared_m_new = m_new;
            shared_alpha = max_score <= -3.0e38f ? 0.0f : expf(max_score - m_new);
        }
        __syncthreads();

        const float m_new = shared_m_new;
        float local_sum = 0.0f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float weight = expf(scores[tile_offset] - m_new);
            scores[tile_offset] = weight;
            local_sum += weight;
        }
        reduce[tid] = local_sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            shared_tile_sum = reduce[0];
        }
        __syncthreads();

        if (value_active) {
            float tile_weighted = 0.0f;
            for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
                const unsigned long long source_timestep =
                    tile_start + static_cast<unsigned long long>(tile_offset);
                const unsigned long long v_index =
                    v_batch_base + (source_timestep * kv_heads + kv_head) * value_dim + value;
                tile_weighted += scores[tile_offset] * v[v_index];
            }
            weighted = weighted * shared_alpha + tile_weighted;
        }
        denominator = denominator * shared_alpha + shared_tile_sum;
        max_score = m_new;
        __syncthreads();
    }

    if (value_active) {
        const unsigned long long output_base = output_batch_base + q_head_index * value_dim;
        output[output_base + value] = weighted / denominator;
    }
}
)";
    }

    static const char *decode_attn_kernel_source() {
        return R"(
extern "C" __global__ void ullm_decode_attn_f32_kernel(
    const float *q,
    const float *k_cache,
    const float *v_cache,
    unsigned long long cache_len,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float *output) {
    __shared__ float reduce[256];
    __shared__ float shared_score;
    __shared__ float shared_weight;
    __shared__ float shared_prev_scale;

    if (gridDim.x == q_heads && blockDim.x == 256u && head_dim <= 256ull &&
        value_dim <= 256ull) {
        const unsigned long long q_head = static_cast<unsigned long long>(blockIdx.x);
        const unsigned int tid = threadIdx.x;
        const unsigned long long q_per_kv = q_heads / kv_heads;
        const unsigned long long kv_head = q_head / q_per_kv;
        const unsigned long long q_base = q_head * head_dim;
        const unsigned long long value = tid;
        const bool value_active = value < value_dim;

        float max_score = -3.4028234663852886e38f;
        float denominator = 0.0f;
        float weighted = 0.0f;
        for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
            const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
            const float local = tid < head_dim ? q[q_base + tid] * k_cache[k_base + tid] : 0.0f;
            reduce[tid] = local;
            __syncthreads();
            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const float score = reduce[0] * softmax_scale;
                const float new_max_score = score > max_score ? score : max_score;
                shared_prev_scale =
                    max_score <= -3.4028234663852886e38f ? 0.0f : expf(max_score - new_max_score);
                shared_weight = expf(score - new_max_score);
                shared_score = new_max_score;
            }
            __syncthreads();
            if (value_active) {
                const unsigned long long v_index =
                    (source_timestep * kv_heads + kv_head) * value_dim + value;
                weighted = weighted * shared_prev_scale + shared_weight * v_cache[v_index];
            }
            denominator = denominator * shared_prev_scale + shared_weight;
            max_score = shared_score;
            __syncthreads();
        }
        if (value_active) {
            output[q_head * value_dim + value] = weighted / denominator;
        }
        return;
    }

    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long elements = q_heads * value_dim;
    if (index >= elements) {
        return;
    }
    const unsigned long long value = index % value_dim;
    const unsigned long long q_head = index / value_dim;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_base = q_head * head_dim;

    float max_score = -3.4028234663852886e38f;
    for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
        const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        for (unsigned long long dim = 0; dim < head_dim; ++dim) {
            score += q[q_base + dim] * k_cache[k_base + dim];
        }
        score *= softmax_scale;
        max_score = score > max_score ? score : max_score;
    }

    float denominator = 0.0f;
    float weighted = 0.0f;
    for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
        const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        for (unsigned long long dim = 0; dim < head_dim; ++dim) {
            score += q[q_base + dim] * k_cache[k_base + dim];
        }
        const float weight = expf(score * softmax_scale - max_score);
        denominator += weight;
        const unsigned long long v_index =
            (source_timestep * kv_heads + kv_head) * value_dim + value;
        weighted += weight * v_cache[v_index];
    }
    output[index] = weighted / denominator;
}
)";
    }

    static const char *cached_prefix_attn_kernel_source() {
        return R"(
extern "C" __global__ void ullm_cached_prefix_attn_f32_kernel(
    const float *q,
    const float *k_cache,
    const float *v_cache,
    unsigned long long cached_prefix_len,
    unsigned long long new_tokens,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float *output) {
    const unsigned long long q_head_index = static_cast<unsigned long long>(blockIdx.x);
    const unsigned long long q_head_elements = new_tokens * q_heads;
    if (q_head_index >= q_head_elements) {
        return;
    }
    const unsigned long long q_head = q_head_index % q_heads;
    const unsigned long long token_index = q_head_index / q_heads;
    const unsigned long long cache_len = cached_prefix_len + token_index + 1;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_base = (token_index * q_heads + q_head) * head_dim;
    const unsigned int tid = threadIdx.x;

    __shared__ float reduce[256];
    __shared__ float shared_score;
    __shared__ float shared_weight;
    __shared__ float shared_max_score;
    __shared__ float shared_denominator;
    __shared__ float shared_prev_scale;

    if (value_dim <= static_cast<unsigned long long>(blockDim.x)) {
        const unsigned long long value = tid;
        const bool value_active = value < value_dim;
        float max_score = -3.4028234663852886e38f;
        float denominator = 0.0f;
        float weighted = 0.0f;
        for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
            const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
                partial += q[q_base + dim] * k_cache[k_base + dim];
            }
            reduce[tid] = partial;
            __syncthreads();
            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const float score = reduce[0] * softmax_scale;
                const float new_max_score = score > max_score ? score : max_score;
                shared_prev_scale =
                    max_score <= -3.4028234663852886e38f ? 0.0f : expf(max_score - new_max_score);
                shared_weight = expf(score - new_max_score);
                shared_score = new_max_score;
            }
            __syncthreads();
            if (value_active) {
                const unsigned long long v_index =
                    (source_timestep * kv_heads + kv_head) * value_dim + value;
                weighted = weighted * shared_prev_scale + shared_weight * v_cache[v_index];
            }
            denominator = denominator * shared_prev_scale + shared_weight;
            max_score = shared_score;
            __syncthreads();
        }
        if (value_active) {
            const unsigned long long output_base = q_head_index * value_dim;
            output[output_base + value] = weighted / denominator;
        }
        return;
    }

    float max_score = -3.4028234663852886e38f;
    for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
        const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
        float partial = 0.0f;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            partial += q[q_base + dim] * k_cache[k_base + dim];
        }
        reduce[tid] = partial;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            shared_score = reduce[0] * softmax_scale;
        }
        __syncthreads();
        const float score = shared_score;
        max_score = score > max_score ? score : max_score;
    }
    if (tid == 0) {
        shared_max_score = max_score;
    }
    __syncthreads();
    max_score = shared_max_score;

    float denominator = 0.0f;
    for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
        const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
        float partial = 0.0f;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            partial += q[q_base + dim] * k_cache[k_base + dim];
        }
        reduce[tid] = partial;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            shared_weight = expf(reduce[0] * softmax_scale - max_score);
        }
        __syncthreads();
        denominator += shared_weight;
    }
    if (tid == 0) {
        shared_denominator = denominator;
    }
    __syncthreads();
    denominator = shared_denominator;

    const unsigned long long output_base = q_head_index * value_dim;
    for (unsigned long long value_base = 0; value_base < value_dim; value_base += blockDim.x) {
        const unsigned long long value = value_base + tid;
        const bool value_active = value < value_dim;
        float weighted = 0.0f;
        for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
            const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
                partial += q[q_base + dim] * k_cache[k_base + dim];
            }
            reduce[tid] = partial;
            __syncthreads();
            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                shared_weight = expf(reduce[0] * softmax_scale - max_score);
            }
            __syncthreads();
            const float weight = shared_weight;
            if (value_active) {
                const unsigned long long v_index =
                    (source_timestep * kv_heads + kv_head) * value_dim + value;
                weighted += weight * v_cache[v_index];
            }
        }
        if (value_active) {
            output[output_base + value] = weighted / denominator;
        }
    }
}
)";
    }

    static void replace_all_source_text(
        std::string *source,
        const std::string &from,
        const std::string &to) {
        if (source == nullptr || from.empty()) {
            return;
        }
        size_t pos = 0;
        while ((pos = source->find(from, pos)) != std::string::npos) {
            source->replace(pos, from.size(), to);
            pos += to.size();
        }
    }

    static std::string cached_prefix_attn_fp8_e4m3_kernel_source() {
        std::string source = R"(
__device__ float ullm_fp8_e4m3_to_f32(unsigned char value) {
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
)";
        source += cached_prefix_attn_kernel_source();
        replace_all_source_text(
            &source,
            "ullm_cached_prefix_attn_f32_kernel",
            "ullm_cached_prefix_attn_fp8_e4m3_kernel");
        replace_all_source_text(
            &source,
            "    const float *k_cache,\n"
            "    const float *v_cache,",
            "    const unsigned char *k_cache,\n"
            "    const unsigned char *v_cache,");
        replace_all_source_text(
            &source,
            "    float softmax_scale,\n"
            "    float *output)",
            "    float softmax_scale,\n"
            "    float k_scale,\n"
            "    float v_scale,\n"
            "    float *output)");
        replace_all_source_text(
            &source,
            "k_cache[k_base + dim]",
            "(ullm_fp8_e4m3_to_f32(k_cache[k_base + dim]) * k_scale)");
        replace_all_source_text(
            &source,
            "v_cache[v_index]",
            "(ullm_fp8_e4m3_to_f32(v_cache[v_index]) * v_scale)");
        return source;
    }

    static const char *cached_prefix_flash2_f32_kernel_source() {
        return R"(
extern "C" __global__ void ullm_cached_prefix_attn_f32_flash2_kernel(
    const float *q,
    const float *k_cache,
    const float *v_cache,
    unsigned long long cached_prefix_len,
    unsigned long long new_tokens,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float *output) {
    constexpr unsigned int kTileTokens = 64u;
    const unsigned long long q_head_index = static_cast<unsigned long long>(blockIdx.x);
    const unsigned long long q_head_elements = new_tokens * q_heads;
    if (q_head_index >= q_head_elements) {
        return;
    }
    const unsigned int tid = threadIdx.x;
    if (value_dim > static_cast<unsigned long long>(blockDim.x)) {
        return;
    }

    const unsigned long long q_head = q_head_index % q_heads;
    const unsigned long long token_index = q_head_index / q_heads;
    const unsigned long long cache_len = cached_prefix_len + token_index + 1;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_base = (token_index * q_heads + q_head) * head_dim;
    const unsigned long long value = static_cast<unsigned long long>(tid);
    const bool value_active = value < value_dim;

    __shared__ float reduce[256];
    __shared__ float scores[kTileTokens];
    __shared__ float shared_tile_max;
    __shared__ float shared_tile_sum;
    __shared__ float shared_m_new;
    __shared__ float shared_alpha;

    float max_score = -3.4028234663852886e38f;
    float denominator = 0.0f;
    float weighted = 0.0f;

    for (unsigned long long tile_start = 0; tile_start < cache_len; tile_start += kTileTokens) {
        const unsigned long long remaining = cache_len - tile_start;
        const unsigned int tile_count =
            remaining < static_cast<unsigned long long>(kTileTokens) ?
                static_cast<unsigned int>(remaining) :
                kTileTokens;

        for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
            const unsigned long long source_timestep =
                tile_start + static_cast<unsigned long long>(tile_offset);
            const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
                partial += q[q_base + dim] * k_cache[k_base + dim];
            }
            reduce[tid] = partial;
            __syncthreads();
            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0u) {
                scores[tile_offset] = reduce[0] * softmax_scale;
            }
            __syncthreads();
        }

        float local_max = -3.4028234663852886e38f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float score = scores[tile_offset];
            local_max = score > local_max ? score : local_max;
        }
        reduce[tid] = local_max;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] = reduce[tid] > reduce[tid + stride] ? reduce[tid] : reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            shared_tile_max = reduce[0];
            const float m_new = max_score > shared_tile_max ? max_score : shared_tile_max;
            shared_m_new = m_new;
            shared_alpha = max_score <= -3.0e38f ? 0.0f : expf(max_score - m_new);
        }
        __syncthreads();

        const float m_new = shared_m_new;
        float local_sum = 0.0f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float weight = expf(scores[tile_offset] - m_new);
            scores[tile_offset] = weight;
            local_sum += weight;
        }
        reduce[tid] = local_sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            shared_tile_sum = reduce[0];
        }
        __syncthreads();

        if (value_active) {
            float tile_weighted = 0.0f;
            for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
                const unsigned long long source_timestep =
                    tile_start + static_cast<unsigned long long>(tile_offset);
                const unsigned long long v_index =
                    (source_timestep * kv_heads + kv_head) * value_dim + value;
                tile_weighted += scores[tile_offset] * v_cache[v_index];
            }
            weighted = weighted * shared_alpha + tile_weighted;
        }
        denominator = denominator * shared_alpha + shared_tile_sum;
        max_score = m_new;
        __syncthreads();
    }

    if (value_active) {
        const unsigned long long output_base = q_head_index * value_dim;
        output[output_base + value] = weighted / denominator;
    }
}
)";
    }

    // This variant keeps the generic Flash2 arithmetic schedule for every
    // individual Q head.  One CTA owns a (token, KV-head) group and stages a
    // 20-token K or V segment only once, then visits the five Q heads in a
    // deterministic sequence.  The five heads therefore reuse the same K/V
    // staging, while score, max, sum, and weighted-value reductions retain the
    // generic 256-thread tree and 64-token online-softmax boundary.
    static const char *cached_prefix_flash2_f32_gqa_grouped_serial_kernel_source() {
        return R"(
extern "C" __global__ void ullm_cached_prefix_attn_f32_flash2_gqa_grouped_kernel(
    const float *q,
    const float *k_cache,
    const float *v_cache,
    unsigned long long cached_prefix_len,
    unsigned long long new_tokens,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float *output) {
    constexpr unsigned int kStageTokens = 20u;
    constexpr unsigned int kFlashTileTokens = 64u;
    constexpr unsigned int kQHeadsPerKv = 5u;
    constexpr unsigned int kHeadDim = 128u;

    if (blockDim.x != 256u || kv_heads == 0u || head_dim != kHeadDim ||
        value_dim != kHeadDim || q_heads != kv_heads * kQHeadsPerKv) {
        return;
    }
    const unsigned long long grouped_index = static_cast<unsigned long long>(blockIdx.x);
    const unsigned long long grouped_count = new_tokens * kv_heads;
    if (grouped_index >= grouped_count) {
        return;
    }

    const unsigned long long token_index = grouped_index / kv_heads;
    const unsigned long long kv_head = grouped_index % kv_heads;
    const unsigned long long cache_len = cached_prefix_len + token_index + 1u;
    const unsigned int tid = threadIdx.x;
    const unsigned long long value = static_cast<unsigned long long>(tid);
    const bool value_active = value < value_dim;

    __shared__ float staged_kv[kStageTokens * kHeadDim];
    __shared__ float reduce[256];
    __shared__ float scores[kQHeadsPerKv * kFlashTileTokens];
    __shared__ float shared_tile_max[kQHeadsPerKv];
    __shared__ float shared_tile_sum[kQHeadsPerKv];
    __shared__ float shared_m_new[kQHeadsPerKv];
    __shared__ float shared_alpha[kQHeadsPerKv];

    float max_score[kQHeadsPerKv] = {
        -3.4028234663852886e38f,
        -3.4028234663852886e38f,
        -3.4028234663852886e38f,
        -3.4028234663852886e38f,
        -3.4028234663852886e38f,
    };
    float denominator[kQHeadsPerKv] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float weighted[kQHeadsPerKv] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (unsigned long long tile_start = 0u; tile_start < cache_len; tile_start += kFlashTileTokens) {
        const unsigned long long remaining = cache_len - tile_start;
        const unsigned int tile_count =
            remaining < static_cast<unsigned long long>(kFlashTileTokens) ?
                static_cast<unsigned int>(remaining) : kFlashTileTokens;

        // Stage K once per KV head.  For each Q head, the score calculation
        // below is deliberately the generic Flash2 body with only the source
        // load redirected to the exact staged float.
        for (unsigned int stage_start = 0u; stage_start < tile_count; stage_start += kStageTokens) {
            const unsigned int stage_count =
                tile_count - stage_start < kStageTokens ? tile_count - stage_start : kStageTokens;
            const unsigned int stage_elements = stage_count * kHeadDim;
            for (unsigned int index = tid; index < stage_elements; index += blockDim.x) {
                const unsigned int token_offset = index / kHeadDim;
                const unsigned int dim = index % kHeadDim;
                const unsigned long long source_timestep =
                    tile_start + static_cast<unsigned long long>(stage_start + token_offset);
                staged_kv[index] =
                    k_cache[(source_timestep * kv_heads + kv_head) * head_dim + dim];
            }
            __syncthreads();

            for (unsigned int q_slot = 0u; q_slot < kQHeadsPerKv; ++q_slot) {
                const unsigned long long q_head =
                    kv_head * kQHeadsPerKv + static_cast<unsigned long long>(q_slot);
                const unsigned long long q_base = (token_index * q_heads + q_head) * head_dim;
                const unsigned int score_base = q_slot * kFlashTileTokens;
                for (unsigned int token_offset = 0u; token_offset < stage_count; ++token_offset) {
                    float partial = 0.0f;
                    for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
                        partial += q[q_base + dim] * staged_kv[token_offset * kHeadDim + dim];
                    }
                    reduce[tid] = partial;
                    __syncthreads();
                    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                        if (tid < stride) {
                            reduce[tid] += reduce[tid + stride];
                        }
                        __syncthreads();
                    }
                    if (tid == 0u) {
                        scores[score_base + stage_start + token_offset] = reduce[0] * softmax_scale;
                    }
                    __syncthreads();
                }
            }
            __syncthreads();
        }

        // Preserve generic Flash2's max and sum trees separately for every
        // Q head.  Only the inter-head schedule changed; a head's own score
        // order and online state update are unchanged.
        for (unsigned int q_slot = 0u; q_slot < kQHeadsPerKv; ++q_slot) {
            const unsigned int score_base = q_slot * kFlashTileTokens;
            float local_max = -3.4028234663852886e38f;
            for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
                const float score = scores[score_base + tile_offset];
                local_max = score > local_max ? score : local_max;
            }
            reduce[tid] = local_max;
            __syncthreads();
            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] = reduce[tid] > reduce[tid + stride] ? reduce[tid] : reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0u) {
                shared_tile_max[q_slot] = reduce[0];
                const float m_new = max_score[q_slot] > shared_tile_max[q_slot] ?
                    max_score[q_slot] : shared_tile_max[q_slot];
                shared_m_new[q_slot] = m_new;
                shared_alpha[q_slot] = max_score[q_slot] <= -3.0e38f ?
                    0.0f : expf(max_score[q_slot] - m_new);
            }
            __syncthreads();

            const float m_new = shared_m_new[q_slot];
            float local_sum = 0.0f;
            for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
                const float weight = expf(scores[score_base + tile_offset] - m_new);
                scores[score_base + tile_offset] = weight;
                local_sum += weight;
            }
            reduce[tid] = local_sum;
            __syncthreads();
            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0u) {
                shared_tile_sum[q_slot] = reduce[0];
            }
            __syncthreads();
        }

        float tile_weighted[kQHeadsPerKv] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        // Reuse the K staging allocation for V.  The V loop has the generic
        // token order for each Q head even though q slots are interleaved.
        for (unsigned int stage_start = 0u; stage_start < tile_count; stage_start += kStageTokens) {
            const unsigned int stage_count =
                tile_count - stage_start < kStageTokens ? tile_count - stage_start : kStageTokens;
            const unsigned int stage_elements = stage_count * kHeadDim;
            for (unsigned int index = tid; index < stage_elements; index += blockDim.x) {
                const unsigned int token_offset = index / kHeadDim;
                const unsigned int dim = index % kHeadDim;
                const unsigned long long source_timestep =
                    tile_start + static_cast<unsigned long long>(stage_start + token_offset);
                staged_kv[index] =
                    v_cache[(source_timestep * kv_heads + kv_head) * value_dim + dim];
            }
            __syncthreads();
            if (value_active) {
                for (unsigned int q_slot = 0u; q_slot < kQHeadsPerKv; ++q_slot) {
                    const unsigned int score_base = q_slot * kFlashTileTokens + stage_start;
                    float current = tile_weighted[q_slot];
                    for (unsigned int token_offset = 0u; token_offset < stage_count; ++token_offset) {
                        current += scores[score_base + token_offset] *
                            staged_kv[token_offset * kHeadDim + value];
                    }
                    tile_weighted[q_slot] = current;
                }
            }
            __syncthreads();
        }

        for (unsigned int q_slot = 0u; q_slot < kQHeadsPerKv; ++q_slot) {
            if (value_active) {
                weighted[q_slot] = weighted[q_slot] * shared_alpha[q_slot] + tile_weighted[q_slot];
            }
            denominator[q_slot] = denominator[q_slot] * shared_alpha[q_slot] + shared_tile_sum[q_slot];
            max_score[q_slot] = shared_m_new[q_slot];
        }
        __syncthreads();
    }

    if (value_active) {
        for (unsigned int q_slot = 0u; q_slot < kQHeadsPerKv; ++q_slot) {
            const unsigned long long q_head =
                kv_head * kQHeadsPerKv + static_cast<unsigned long long>(q_slot);
            const unsigned long long output_base =
                (token_index * q_heads + q_head) * value_dim + value;
            output[output_base] = weighted[q_slot] / denominator[q_slot];
        }
    }
}
)";
    }

    // This isolated R9700-only source is intentionally a separate symbol.  It
    // keeps the public Flash2 launch contract and F32 KV layout unchanged while
    // replacing the three 256-lane LDS trees with wave32 reductions plus the
    // necessary LDS handoff.  The normal source above remains the
    // default until its numerical/resource/timing gates have passed.
    static std::string cached_prefix_flash2_f32_staged_wave32_prototype_kernel_source() {
        std::string source = R"(
__device__ float ullm_sq8_0_flash2_reduce_sum_wave32(float value, float *partial) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int wave = tid >> 5u;
    const unsigned int wave_count = (blockDim.x + 31u) >> 5u;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, 32);
    }
    if (lane == 0u) {
        partial[wave] = value;
    }
    __syncthreads();
    if (wave == 0u) {
        value = lane < wave_count ? partial[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down(value, offset, 32);
        }
        if (lane == 0u) {
            partial[0] = value;
        }
    }
    __syncthreads();
    return partial[0];
}

__device__ float ullm_sq8_0_flash2_reduce_max_wave32(float value, float *partial) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int wave = tid >> 5u;
    const unsigned int wave_count = (blockDim.x + 31u) >> 5u;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down(value, offset, 32);
        value = value > other ? value : other;
    }
    if (lane == 0u) {
        partial[wave] = value;
    }
    __syncthreads();
    if (wave == 0u) {
        value = lane < wave_count ? partial[lane] : -3.4028234663852886e38f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_down(value, offset, 32);
            value = value > other ? value : other;
        }
        if (lane == 0u) {
            partial[0] = value;
        }
    }
    __syncthreads();
    return partial[0];
}
)";
        source += cached_prefix_flash2_f32_kernel_source();
        replace_all_source_text(
            &source,
            "ullm_cached_prefix_attn_f32_flash2_kernel",
            "ullm_sq8_0_cached_prefix_attn_f32_flash2_staged_wave32_prototype_kernel");
        replace_all_source_text(
            &source,
            "            reduce[tid] = partial;\n"
            "            __syncthreads();\n"
            "            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {\n"
            "                if (tid < stride) {\n"
            "                    reduce[tid] += reduce[tid + stride];\n"
            "                }\n"
            "                __syncthreads();\n"
            "            }\n"
            "            if (tid == 0u) {\n"
            "                scores[tile_offset] = reduce[0] * softmax_scale;\n"
            "            }\n"
            "            __syncthreads();",
            "            const float score =\n"
            "                ullm_sq8_0_flash2_reduce_sum_wave32(partial, reduce) * softmax_scale;\n"
            "            if (tid == 0u) {\n"
            "                scores[tile_offset] = score;\n"
            "            }\n"
            "            __syncthreads();");
        replace_all_source_text(
            &source,
            "        reduce[tid] = local_max;\n"
            "        __syncthreads();\n"
            "        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {\n"
            "            if (tid < stride) {\n"
            "                reduce[tid] = reduce[tid] > reduce[tid + stride] ? reduce[tid] : reduce[tid + stride];\n"
            "            }\n"
            "            __syncthreads();\n"
            "        }",
            "        const float tile_max = ullm_sq8_0_flash2_reduce_max_wave32(local_max, reduce);");
        replace_all_source_text(
            &source,
            "            shared_tile_max = reduce[0];",
            "            shared_tile_max = tile_max;");
        replace_all_source_text(
            &source,
            "        reduce[tid] = local_sum;\n"
            "        __syncthreads();\n"
            "        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {\n"
            "            if (tid < stride) {\n"
            "                reduce[tid] += reduce[tid + stride];\n"
            "            }\n"
            "            __syncthreads();\n"
            "        }",
            "        const float tile_sum = ullm_sq8_0_flash2_reduce_sum_wave32(local_sum, reduce);");
        replace_all_source_text(
            &source,
            "            shared_tile_sum = reduce[0];",
            "            shared_tile_sum = tile_sum;");
        return source;
    }

    static const char *cached_prefix_flash2_fp8_e4m3_kernel_source() {
        return R"(
__device__ float ullm_flash2_fp8_e4m3_to_f32(unsigned char value) {
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

extern "C" __global__ void ullm_cached_prefix_attn_fp8_e4m3_flash2_kernel(
    const void *q_raw,
    const unsigned char *k_cache,
    const unsigned char *v_cache,
    unsigned long long cached_prefix_len,
    unsigned long long new_tokens,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float q_scale,
    float k_scale,
    float v_scale,
    unsigned int q_is_fp8,
    float *output) {
    constexpr unsigned int kTileTokens = 64u;
    const unsigned long long q_head_index = static_cast<unsigned long long>(blockIdx.x);
    const unsigned long long q_head_elements = new_tokens * q_heads;
    if (q_head_index >= q_head_elements) {
        return;
    }
    const unsigned int tid = threadIdx.x;
    if (value_dim > static_cast<unsigned long long>(blockDim.x)) {
        return;
    }

    const unsigned long long q_head = q_head_index % q_heads;
    const unsigned long long token_index = q_head_index / q_heads;
    const unsigned long long cache_len = cached_prefix_len + token_index + 1;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_base = (token_index * q_heads + q_head) * head_dim;
    const float *q_f32 = reinterpret_cast<const float *>(q_raw);
    const unsigned char *q_fp8 = reinterpret_cast<const unsigned char *>(q_raw);
    const unsigned long long value = static_cast<unsigned long long>(tid);
    const bool value_active = value < value_dim;

    __shared__ float reduce[256];
    __shared__ float scores[kTileTokens];
    __shared__ float shared_tile_max;
    __shared__ float shared_tile_sum;
    __shared__ float shared_m_new;
    __shared__ float shared_alpha;

    float max_score = -3.4028234663852886e38f;
    float denominator = 0.0f;
    float weighted = 0.0f;

    for (unsigned long long tile_start = 0; tile_start < cache_len; tile_start += kTileTokens) {
        const unsigned long long remaining = cache_len - tile_start;
        const unsigned int tile_count =
            remaining < static_cast<unsigned long long>(kTileTokens) ?
                static_cast<unsigned int>(remaining) :
                kTileTokens;

        for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
            const unsigned long long source_timestep =
                tile_start + static_cast<unsigned long long>(tile_offset);
            const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
                const float q_value = q_is_fp8 == 0u ?
                    q_f32[q_base + dim] :
                    (ullm_flash2_fp8_e4m3_to_f32(q_fp8[q_base + dim]) * q_scale);
                partial += q_value *
                    (ullm_flash2_fp8_e4m3_to_f32(k_cache[k_base + dim]) * k_scale);
            }
            reduce[tid] = partial;
            __syncthreads();
            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0u) {
                scores[tile_offset] = reduce[0] * softmax_scale;
            }
            __syncthreads();
        }

        float local_max = -3.4028234663852886e38f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float score = scores[tile_offset];
            local_max = score > local_max ? score : local_max;
        }
        reduce[tid] = local_max;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] = reduce[tid] > reduce[tid + stride] ? reduce[tid] : reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            shared_tile_max = reduce[0];
            const float m_new = max_score > shared_tile_max ? max_score : shared_tile_max;
            shared_m_new = m_new;
            shared_alpha = max_score <= -3.0e38f ? 0.0f : expf(max_score - m_new);
        }
        __syncthreads();

        const float m_new = shared_m_new;
        float local_sum = 0.0f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float weight = expf(scores[tile_offset] - m_new);
            scores[tile_offset] = weight;
            local_sum += weight;
        }
        reduce[tid] = local_sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            shared_tile_sum = reduce[0];
        }
        __syncthreads();

        if (value_active) {
            float tile_weighted = 0.0f;
            for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
                const unsigned long long source_timestep =
                    tile_start + static_cast<unsigned long long>(tile_offset);
                const unsigned long long v_index =
                    (source_timestep * kv_heads + kv_head) * value_dim + value;
                tile_weighted +=
                    scores[tile_offset] *
                    (ullm_flash2_fp8_e4m3_to_f32(v_cache[v_index]) * v_scale);
            }
            weighted = weighted * shared_alpha + tile_weighted;
        }
        denominator = denominator * shared_alpha + shared_tile_sum;
        max_score = m_new;
        __syncthreads();
    }

    if (value_active) {
        const unsigned long long output_base = q_head_index * value_dim;
        output[output_base + value] = weighted / denominator;
    }
}
)";
    }

    static std::string paged_decode_attn_kernel_source_for_arch(const std::string &arch) {
        (void)arch;
        std::string preamble;
        const auto experimental_flag_is_one = [](const char *name) {
            const char *value = std::getenv(name);
            return value != nullptr && value[0] == '1' && value[1] == '\0';
        };
        if (std::getenv("ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE") != nullptr) {
            preamble += "#define ULLM_PAGED_DECODE_USE_SHARED_REDUCE 1\n";
        }
        if (std::getenv("ULLM_DISABLE_PAGED_DECODE_ONLINE_SOFTMAX") != nullptr) {
            preamble += "#define ULLM_PAGED_DECODE_USE_TWO_PASS_SOFTMAX 1\n";
        }
        if (std::getenv("ULLM_EXPERIMENTAL_PAGED_DECODE_WAVE_SCALAR_SOFTMAX") != nullptr) {
            preamble += "#define ULLM_PAGED_DECODE_USE_WAVE_SCALAR_SOFTMAX 1\n";
        }
        // These two selectors are intentionally test-only.  The grouped body is
        // shape-guarded in the device source and the host launcher, so an
        // unsupported geometry continues to use the canonical split body.
        const bool gqa_grouped =
            experimental_flag_is_one("ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT");
        const bool gqa_pipelined =
            experimental_flag_is_one("ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_PIPELINED_SPLIT");
        if (gqa_grouped || gqa_pipelined) {
            preamble += "#define ULLM_PAGED_DECODE_USE_GQA_GROUPED_SPLIT 1\n";
        }
        if (gqa_pipelined) {
            preamble += "#define ULLM_PAGED_DECODE_USE_GQA_PIPELINED_SPLIT 1\n";
        }
        return preamble + paged_decode_attn_kernel_source();
    }

    static const char *paged_decode_attn_kernel_source() {
        return R"(
__device__ float ullm_paged_decode_reduce_sum_256(float value, float *partial) {
#if defined(ULLM_PAGED_DECODE_USE_SHARED_REDUCE)
    const unsigned int tid = threadIdx.x;
    partial[tid] = value;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    return partial[0];
#else
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % warpSize;
    const unsigned int wave = tid / warpSize;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, warpSize);
    }
    if (lane == 0) {
        partial[wave] = value;
    }
    __syncthreads();

    float block_sum = 0.0f;
    const unsigned int wave_count = (blockDim.x + warpSize - 1u) / warpSize;
    if (wave == 0) {
        block_sum = lane < wave_count ? partial[lane] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            block_sum += __shfl_down(block_sum, offset, warpSize);
        }
        if (lane == 0) {
            partial[0] = block_sum;
        }
    }
    __syncthreads();
    return partial[0];
#endif
}

extern "C" __global__ void ullm_paged_decode_attn_f32_kernel(
    const float *q,
    const float *gate,
    const float *k_cache,
    const float *v_cache,
    const unsigned int *block_table,
    unsigned long long cache_len,
    unsigned long long block_size,
    unsigned long long cache_blocks,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float *output) {
    __shared__ float partial[256];
    if (gridDim.x == q_heads && blockDim.x == 256u && head_dim <= 256ull &&
        value_dim <= 256ull) {
        const unsigned long long q_head = blockIdx.x;
        const unsigned int tid = threadIdx.x;
#if defined(ULLM_PAGED_DECODE_USE_WAVE_SCALAR_SOFTMAX) && !defined(ULLM_PAGED_DECODE_USE_TWO_PASS_SOFTMAX)
        const unsigned int lane = tid % warpSize;
        const unsigned int wave = tid / warpSize;
#endif
        const unsigned long long q_per_kv = q_heads / kv_heads;
        const unsigned long long kv_head = q_head / q_per_kv;
        const unsigned long long q_base = q_head * head_dim;

#if defined(ULLM_PAGED_DECODE_USE_TWO_PASS_SOFTMAX)
        float max_score = -3.4028234663852886e38f;
        for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
            const unsigned long long block_index = source_timestep / block_size;
            const unsigned long long block_offset = source_timestep - block_index * block_size;
            const unsigned long long block_id =
                static_cast<unsigned long long>(block_table[block_index]);
            if (block_id >= cache_blocks) {
                if (tid < value_dim) {
                    output[q_head * value_dim + tid] = 0.0f;
                }
                return;
            }
            const unsigned long long physical_timestep = block_id * block_size + block_offset;
            const unsigned long long k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
            const float local =
                tid < head_dim ? q[q_base + tid] * k_cache[k_base + tid] : 0.0f;
            const float score =
                ullm_paged_decode_reduce_sum_256(local, partial) * softmax_scale;
            max_score = score > max_score ? score : max_score;
        }

        float denominator = 0.0f;
        float weighted = 0.0f;
        for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
            const unsigned long long block_index = source_timestep / block_size;
            const unsigned long long block_offset = source_timestep - block_index * block_size;
            const unsigned long long block_id =
                static_cast<unsigned long long>(block_table[block_index]);
            if (block_id >= cache_blocks) {
                if (tid < value_dim) {
                    output[q_head * value_dim + tid] = 0.0f;
                }
                return;
            }
            const unsigned long long physical_timestep = block_id * block_size + block_offset;
            const unsigned long long k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
            const float local =
                tid < head_dim ? q[q_base + tid] * k_cache[k_base + tid] : 0.0f;
            const float score =
                ullm_paged_decode_reduce_sum_256(local, partial) * softmax_scale;
            const float weight = expf(score - max_score);
            denominator += weight;
            if (tid < value_dim) {
                const unsigned long long v_index =
                    (physical_timestep * kv_heads + kv_head) * value_dim + tid;
                weighted += weight * v_cache[v_index];
            }
        }
#else
        float max_score = -3.4028234663852886e38f;
        float denominator = 0.0f;
        float weighted = 0.0f;
        unsigned long long block_index = 0ull;
        unsigned long long block_end = block_size;
        unsigned long long block_id = static_cast<unsigned long long>(block_table[0]);
        if (block_id >= cache_blocks) {
            if (tid < value_dim) {
                output[q_head * value_dim + tid] = 0.0f;
            }
            return;
        }
        unsigned long long physical_timestep = block_id * block_size;
        for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
            if (source_timestep == block_end) {
                ++block_index;
                block_end += block_size;
                block_id = static_cast<unsigned long long>(block_table[block_index]);
                if (block_id >= cache_blocks) {
                    if (tid < value_dim) {
                        output[q_head * value_dim + tid] = 0.0f;
                    }
                    return;
                }
                physical_timestep = block_id * block_size;
            }
            const unsigned long long k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
            const float local =
                tid < head_dim ? q[q_base + tid] * k_cache[k_base + tid] : 0.0f;
            const float score =
                ullm_paged_decode_reduce_sum_256(local, partial) * softmax_scale;
#if defined(ULLM_PAGED_DECODE_USE_WAVE_SCALAR_SOFTMAX) && !defined(ULLM_PAGED_DECODE_USE_TWO_PASS_SOFTMAX)
            const bool value_active = tid < value_dim;
            float v = 0.0f;
            if (value_active) {
                const unsigned long long v_index =
                    (physical_timestep * kv_heads + kv_head) * value_dim + tid;
                v = v_cache[v_index];
            }
            const bool wave_has_value =
                static_cast<unsigned long long>(wave) * static_cast<unsigned long long>(warpSize) < value_dim;
            if (wave_has_value) {
                float wave_scale = 0.0f;
                float wave_replaces_max = 0.0f;
                if (lane == 0u) {
                    if (score > max_score) {
                        wave_scale = expf(max_score - score);
                        denominator = denominator * wave_scale + 1.0f;
                        max_score = score;
                        wave_replaces_max = 1.0f;
                    } else {
                        wave_scale = expf(score - max_score);
                        denominator += wave_scale;
                    }
                }
                wave_scale = __shfl(wave_scale, 0, warpSize);
                wave_replaces_max = __shfl(wave_replaces_max, 0, warpSize);
                if (value_active) {
                    if (wave_replaces_max != 0.0f) {
                        weighted = weighted * wave_scale + v;
                    } else {
                        weighted += wave_scale * v;
                    }
                }
            }
#else
            if (tid < value_dim) {
                const unsigned long long v_index =
                    (physical_timestep * kv_heads + kv_head) * value_dim + tid;
                const float v = v_cache[v_index];
                if (score > max_score) {
                    const float scale = expf(max_score - score);
                    weighted = weighted * scale + v;
                    denominator = denominator * scale + 1.0f;
                    max_score = score;
                } else {
                    const float weight = expf(score - max_score);
                    weighted += weight * v;
                    denominator += weight;
                }
            }
#endif
            ++physical_timestep;
        }
#endif
#if defined(ULLM_PAGED_DECODE_USE_WAVE_SCALAR_SOFTMAX) && !defined(ULLM_PAGED_DECODE_USE_TWO_PASS_SOFTMAX)
        const float final_denominator = __shfl(denominator, 0, warpSize);
        if (tid < value_dim) {
            const unsigned long long output_index = q_head * value_dim + tid;
            const float decoded = weighted / final_denominator;
            if (gate != nullptr) {
                const float gate_value = gate[output_index];
                const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
                output[output_index] = sigmoid * decoded;
            } else {
                output[output_index] = decoded;
            }
        }
#else
        if (tid < value_dim) {
            const unsigned long long output_index = q_head * value_dim + tid;
            const float decoded = weighted / denominator;
            if (gate != nullptr) {
                const float gate_value = gate[output_index];
                const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
                output[output_index] = sigmoid * decoded;
            } else {
                output[output_index] = decoded;
            }
        }
#endif
        return;
    }

    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long elements = q_heads * value_dim;
    if (index >= elements) {
        return;
    }
    const unsigned long long value = index % value_dim;
    const unsigned long long q_head = index / value_dim;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_base = q_head * head_dim;

    float max_score = -3.4028234663852886e38f;
    for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
        const unsigned long long block_index = source_timestep / block_size;
        const unsigned long long block_offset = source_timestep - block_index * block_size;
        const unsigned long long block_id = static_cast<unsigned long long>(block_table[block_index]);
        if (block_id >= cache_blocks) {
            output[index] = 0.0f;
            return;
        }
        const unsigned long long physical_timestep = block_id * block_size + block_offset;
        const unsigned long long k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        for (unsigned long long dim = 0; dim < head_dim; ++dim) {
            score += q[q_base + dim] * k_cache[k_base + dim];
        }
        score *= softmax_scale;
        max_score = score > max_score ? score : max_score;
    }

    float denominator = 0.0f;
    float weighted = 0.0f;
    for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
        const unsigned long long block_index = source_timestep / block_size;
        const unsigned long long block_offset = source_timestep - block_index * block_size;
        const unsigned long long block_id = static_cast<unsigned long long>(block_table[block_index]);
        if (block_id >= cache_blocks) {
            output[index] = 0.0f;
            return;
        }
        const unsigned long long physical_timestep = block_id * block_size + block_offset;
        const unsigned long long k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        for (unsigned long long dim = 0; dim < head_dim; ++dim) {
            score += q[q_base + dim] * k_cache[k_base + dim];
        }
        const float weight = expf(score * softmax_scale - max_score);
        denominator += weight;
        const unsigned long long v_index =
            (physical_timestep * kv_heads + kv_head) * value_dim + value;
        weighted += weight * v_cache[v_index];
    }
    const float decoded = weighted / denominator;
    if (gate != nullptr) {
        const float gate_value = gate[index];
        const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
        output[index] = sigmoid * decoded;
    } else {
        output[index] = decoded;
    }
}

extern "C" __global__ void ullm_paged_decode_attn_split_partial_f32_kernel(
    const float *q,
    const float *k_cache,
    const float *v_cache,
    const unsigned int *block_table,
    unsigned long long cache_len,
    unsigned long long block_size,
    unsigned long long cache_blocks,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    unsigned long long source_tile,
    unsigned long long split_count,
    float *workspace) {
    __shared__ float partial[256];
#if defined(ULLM_PAGED_DECODE_USE_GQA_GROUPED_SPLIT)
    __shared__ float gqa_stage[512];
    __shared__ int gqa_stage_valid[2];
    // R9700 SQ8_0 decode shape: five query heads share one KV head.  One
    // workgroup owns those five heads for one source tile.  The first five
    // waves hold one Q head each (four dimensions/value elements per lane);
    // the remaining three waves stage a single K+V row for all five.  Thus
    // K/V have one semantic global load per KV head rather than five.  This
    // says nothing about physical HBM traffic or cache hit rate.
    const unsigned long long gqa_q_per_kv =
        kv_heads == 0ull ? 0ull : q_heads / kv_heads;
    if (blockDim.x == 256u && warpSize == 32u && gqa_q_per_kv == 5ull &&
        head_dim == 128ull && value_dim == 128ull &&
        static_cast<unsigned long long>(gridDim.x) == kv_heads * split_count) {
        const unsigned int tid = threadIdx.x;
        const unsigned int lane = tid % 32u;
        const bool consumer = tid < 160u;
        const unsigned long long grouped_index = blockIdx.x;
        const unsigned long long kv_head = grouped_index / split_count;
        const unsigned long long split_id = grouped_index - kv_head * split_count;
        if (kv_head >= kv_heads) {
            return;
        }
        const unsigned long long source_start = split_id * source_tile;
        const unsigned long long source_remaining = cache_len - source_start;
        const unsigned long long source_count =
            source_tile < source_remaining ? source_tile : source_remaining;
        const unsigned long long stride = value_dim + 2ull;
        const unsigned long long q_head =
            kv_head * gqa_q_per_kv + static_cast<unsigned long long>(tid / 32u);

        float q0 = 0.0f;
        float q1 = 0.0f;
        float q2 = 0.0f;
        float q3 = 0.0f;
        float weighted0 = 0.0f;
        float weighted1 = 0.0f;
        float weighted2 = 0.0f;
        float weighted3 = 0.0f;
        float max_score = -3.4028234663852886e38f;
        float denominator = 0.0f;
        bool invalid = false;
        if (consumer) {
            const unsigned long long q_base = q_head * 128ull;
            q0 = q[q_base + lane];
            q1 = q[q_base + 32ull + lane];
            q2 = q[q_base + 64ull + lane];
            q3 = q[q_base + 96ull + lane];
        }

        // Two K/V rows are needed only by the pipelined body.  Keeping the
        // same allocation in the non-pipelined body makes the resource delta
        // explicit and keeps the two experiments ABI-identical.
#if defined(ULLM_PAGED_DECODE_USE_GQA_PIPELINED_SPLIT)
        // Prime stage zero with all eight waves.  Subsequent rows are loaded
        // by waves 5--7 while waves 0--4 consume the other buffer; the sole
        // bottom barrier both protects the old buffer and publishes the next.
        const unsigned long long initial_block_index = source_start / block_size;
        const unsigned long long initial_block_offset =
            source_start - initial_block_index * block_size;
        const unsigned long long initial_block_id =
            static_cast<unsigned long long>(block_table[initial_block_index]);
        const bool initial_valid = initial_block_id < cache_blocks;
        if (tid == 0u) {
            gqa_stage_valid[0] = initial_valid ? 1 : 0;
        }
        if (initial_valid) {
            const unsigned long long physical_timestep =
                initial_block_id * block_size + initial_block_offset;
            if (tid < 128u) {
                gqa_stage[tid] =
                    k_cache[(physical_timestep * kv_heads + kv_head) * 128ull + tid];
            } else {
                gqa_stage[tid] =
                    v_cache[(physical_timestep * kv_heads + kv_head) * 128ull + (tid - 128u)];
            }
        } else {
            gqa_stage[tid] = 0.0f;
        }
        __syncthreads();

        for (unsigned long long source_offset = 0ull;
             source_offset < source_count;
             ++source_offset) {
            const unsigned int current_stage = static_cast<unsigned int>(source_offset & 1ull);
            const unsigned int next_stage = current_stage ^ 1u;
            if (source_offset + 1ull < source_count && tid >= 160u) {
                const unsigned long long next_source = source_start + source_offset + 1ull;
                const unsigned long long next_block_index = next_source / block_size;
                const unsigned long long next_block_offset =
                    next_source - next_block_index * block_size;
                const unsigned long long next_block_id =
                    static_cast<unsigned long long>(block_table[next_block_index]);
                const bool next_valid = next_block_id < cache_blocks;
                if (tid == 160u) {
                    gqa_stage_valid[next_stage] = next_valid ? 1 : 0;
                }
                const unsigned long long next_physical_timestep =
                    next_block_id * block_size + next_block_offset;
                for (unsigned int copy_index = tid - 160u;
                     copy_index < 256u;
                     copy_index += 96u) {
                    const unsigned int destination = next_stage * 256u + copy_index;
                    if (!next_valid) {
                        gqa_stage[destination] = 0.0f;
                    } else if (copy_index < 128u) {
                        gqa_stage[destination] = k_cache[
                            (next_physical_timestep * kv_heads + kv_head) * 128ull + copy_index];
                    } else {
                        gqa_stage[destination] = v_cache[
                            (next_physical_timestep * kv_heads + kv_head) * 128ull +
                            (copy_index - 128u)];
                    }
                }
            }

            if (consumer) {
                const unsigned int stage_base = current_stage * 256u;
                if (gqa_stage_valid[current_stage] == 0) {
                    invalid = true;
                } else {
                    float dot = q0 * gqa_stage[stage_base + lane] +
                        q1 * gqa_stage[stage_base + 32u + lane] +
                        q2 * gqa_stage[stage_base + 64u + lane] +
                        q3 * gqa_stage[stage_base + 96u + lane];
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        dot += __shfl_down(dot, offset, warpSize);
                    }
                    const float score = __shfl(dot, 0, warpSize) * softmax_scale;
                    float scale = 0.0f;
                    float replaces_max = 0.0f;
                    if (lane == 0u) {
                        if (score > max_score) {
                            scale = expf(max_score - score);
                            denominator = denominator * scale + 1.0f;
                            max_score = score;
                            replaces_max = 1.0f;
                        } else {
                            scale = expf(score - max_score);
                            denominator += scale;
                        }
                    }
                    scale = __shfl(scale, 0, warpSize);
                    replaces_max = __shfl(replaces_max, 0, warpSize);
                    const float v0 = gqa_stage[stage_base + 128u + lane];
                    const float v1 = gqa_stage[stage_base + 160u + lane];
                    const float v2 = gqa_stage[stage_base + 192u + lane];
                    const float v3 = gqa_stage[stage_base + 224u + lane];
                    if (replaces_max != 0.0f) {
                        weighted0 = weighted0 * scale + v0;
                        weighted1 = weighted1 * scale + v1;
                        weighted2 = weighted2 * scale + v2;
                        weighted3 = weighted3 * scale + v3;
                    } else {
                        weighted0 += scale * v0;
                        weighted1 += scale * v1;
                        weighted2 += scale * v2;
                        weighted3 += scale * v3;
                    }
                }
            }
            __syncthreads();
        }
#else
        // Baseline grouped form: shared K/V reuse without changing the
        // original two CTA barriers per source row.  This isolates the GQA
        // reuse effect from the pipelined-barrier experiment above.
        for (unsigned long long source_offset = 0ull;
             source_offset < source_count;
             ++source_offset) {
            const unsigned long long source_timestep = source_start + source_offset;
            const unsigned long long block_index = source_timestep / block_size;
            const unsigned long long block_offset = source_timestep - block_index * block_size;
            const unsigned long long block_id =
                static_cast<unsigned long long>(block_table[block_index]);
            const bool row_valid = block_id < cache_blocks;
            if (tid == 0u) {
                gqa_stage_valid[0] = row_valid ? 1 : 0;
            }
            if (row_valid) {
                const unsigned long long physical_timestep =
                    block_id * block_size + block_offset;
                if (tid < 128u) {
                    gqa_stage[tid] =
                        k_cache[(physical_timestep * kv_heads + kv_head) * 128ull + tid];
                } else {
                    gqa_stage[tid] =
                        v_cache[(physical_timestep * kv_heads + kv_head) * 128ull + (tid - 128u)];
                }
            } else {
                gqa_stage[tid] = 0.0f;
            }
            __syncthreads();

            if (consumer) {
                if (gqa_stage_valid[0] == 0) {
                    invalid = true;
                } else {
                    float dot = q0 * gqa_stage[lane] + q1 * gqa_stage[32u + lane] +
                        q2 * gqa_stage[64u + lane] + q3 * gqa_stage[96u + lane];
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        dot += __shfl_down(dot, offset, warpSize);
                    }
                    const float score = __shfl(dot, 0, warpSize) * softmax_scale;
                    float scale = 0.0f;
                    float replaces_max = 0.0f;
                    if (lane == 0u) {
                        if (score > max_score) {
                            scale = expf(max_score - score);
                            denominator = denominator * scale + 1.0f;
                            max_score = score;
                            replaces_max = 1.0f;
                        } else {
                            scale = expf(score - max_score);
                            denominator += scale;
                        }
                    }
                    scale = __shfl(scale, 0, warpSize);
                    replaces_max = __shfl(replaces_max, 0, warpSize);
                    const float v0 = gqa_stage[128u + lane];
                    const float v1 = gqa_stage[160u + lane];
                    const float v2 = gqa_stage[192u + lane];
                    const float v3 = gqa_stage[224u + lane];
                    if (replaces_max != 0.0f) {
                        weighted0 = weighted0 * scale + v0;
                        weighted1 = weighted1 * scale + v1;
                        weighted2 = weighted2 * scale + v2;
                        weighted3 = weighted3 * scale + v3;
                    } else {
                        weighted0 += scale * v0;
                        weighted1 += scale * v1;
                        weighted2 += scale * v2;
                        weighted3 += scale * v3;
                    }
                }
            }
            __syncthreads();
        }
#endif

        if (consumer) {
            const unsigned long long workspace_base =
                (q_head * split_count + split_id) * stride;
            if (lane == 0u) {
                workspace[workspace_base] = invalid ? 0.0f : max_score;
                workspace[workspace_base + 1ull] = invalid ? -1.0f : denominator;
            }
            workspace[workspace_base + 2ull + lane] = invalid ? 0.0f : weighted0;
            workspace[workspace_base + 2ull + 32ull + lane] = invalid ? 0.0f : weighted1;
            workspace[workspace_base + 2ull + 64ull + lane] = invalid ? 0.0f : weighted2;
            workspace[workspace_base + 2ull + 96ull + lane] = invalid ? 0.0f : weighted3;
        }
        return;
    }
#endif

    if (blockDim.x != 256u || head_dim > 256ull || value_dim > 256ull) {
        return;
    }

    const unsigned long long partial_index = blockIdx.x;
    const unsigned long long q_head = partial_index / split_count;
    const unsigned long long split_id = partial_index - q_head * split_count;
    if (q_head >= q_heads) {
        return;
    }
    const unsigned int tid = threadIdx.x;
    const unsigned long long stride = value_dim + 2ull;
    const unsigned long long workspace_base = partial_index * stride;
    const unsigned long long source_start = split_id * source_tile;
    const unsigned long long source_remaining =
        source_start < cache_len ? cache_len - source_start : 0ull;
    const unsigned long long source_count =
        source_tile < source_remaining ? source_tile : source_remaining;
    const unsigned long long source_end = source_start + source_count;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_base = q_head * head_dim;

    float max_score = -3.4028234663852886e38f;
    float denominator = 0.0f;
    float weighted = 0.0f;
    for (unsigned long long source_timestep = source_start;
         source_timestep < source_end;
         ++source_timestep) {
        const unsigned long long block_index = source_timestep / block_size;
        const unsigned long long block_offset = source_timestep - block_index * block_size;
        const unsigned long long block_id =
            static_cast<unsigned long long>(block_table[block_index]);
        if (block_id >= cache_blocks) {
            if (tid == 0u) {
                workspace[workspace_base] = 0.0f;
                workspace[workspace_base + 1ull] = -1.0f;
            }
            if (tid < value_dim) {
                workspace[workspace_base + 2ull + tid] = 0.0f;
            }
            return;
        }
        const unsigned long long physical_timestep = block_id * block_size + block_offset;
        const unsigned long long k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
        const float local =
            tid < head_dim ? q[q_base + tid] * k_cache[k_base + tid] : 0.0f;
        const float score =
            ullm_paged_decode_reduce_sum_256(local, partial) * softmax_scale;
        if (tid < value_dim) {
            const unsigned long long v_index =
                (physical_timestep * kv_heads + kv_head) * value_dim + tid;
            const float v = v_cache[v_index];
            if (score > max_score) {
                const float scale = expf(max_score - score);
                weighted = weighted * scale + v;
                denominator = denominator * scale + 1.0f;
                max_score = score;
            } else {
                const float weight = expf(score - max_score);
                weighted += weight * v;
                denominator += weight;
            }
        }
    }
    if (tid == 0u) {
        workspace[workspace_base] = max_score;
        workspace[workspace_base + 1ull] = denominator;
    }
    if (tid < value_dim) {
        workspace[workspace_base + 2ull + tid] = weighted;
    }
}

extern "C" __global__ void ullm_paged_decode_attn_split_merge_f32_kernel(
    const float *gate,
    unsigned long long q_heads,
    unsigned long long value_dim,
    unsigned long long split_count,
    const float *workspace,
    float *output) {
    if (blockDim.x != 256u || value_dim > 256ull) {
        return;
    }
    const unsigned long long q_head = blockIdx.x;
    if (q_head >= q_heads) {
        return;
    }
    const unsigned int tid = threadIdx.x;
    const unsigned long long stride = value_dim + 2ull;
    float merged_max = -3.4028234663852886e38f;
    float merged_denominator = 0.0f;
    float merged_numerator = 0.0f;
    bool invalid = false;
    for (unsigned long long split_id = 0ull; split_id < split_count; ++split_id) {
        const unsigned long long partial_base =
            (q_head * split_count + split_id) * stride;
        const float partial_max = workspace[partial_base];
        const float partial_denominator = workspace[partial_base + 1ull];
        if (partial_denominator < 0.0f) {
            invalid = true;
            break;
        }
        const float partial_numerator =
            tid < value_dim ? workspace[partial_base + 2ull + tid] : 0.0f;
        if (split_id == 0ull) {
            merged_max = partial_max;
            merged_denominator = partial_denominator;
            merged_numerator = partial_numerator;
        } else {
            const float new_max = partial_max > merged_max ? partial_max : merged_max;
            const float alpha = expf(merged_max - new_max);
            const float beta = expf(partial_max - new_max);
            merged_denominator =
                merged_denominator * alpha + partial_denominator * beta;
            merged_numerator = merged_numerator * alpha + partial_numerator * beta;
            merged_max = new_max;
        }
    }

    if (tid < value_dim) {
        const unsigned long long output_index = q_head * value_dim + tid;
        if (invalid) {
            output[output_index] = 0.0f;
            return;
        }
        const float decoded = merged_numerator / merged_denominator;
        if (gate != nullptr) {
            const float gate_value = gate[output_index];
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            output[output_index] = sigmoid * decoded;
        } else {
            output[output_index] = decoded;
        }
    }
}

// Typed direct decode is compiled as one body per ordered K/V pair.  The
// legacy F32 kernel above remains the F32/F32 implementation and is not
// textually changed by this addition.
enum {
    ULLM_TYPED_DECODE_F32 = 0,
    ULLM_TYPED_DECODE_F16 = 1,
    ULLM_TYPED_DECODE_FP8_E4M3FN = 2,
};

__device__ __forceinline__ float ullm_typed_decode_f16_to_f32(unsigned short bits) {
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

__device__ __forceinline__ float ullm_typed_decode_fp8_e4m3fn_to_f32(unsigned char value) {
#if defined(__gfx1200__) || defined(__gfx1201__)
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

template <int DTYPE>
__device__ __forceinline__ float ullm_typed_decode_row_scale(
    const unsigned short *scales,
    unsigned long long physical_head) {
    if constexpr (DTYPE == ULLM_TYPED_DECODE_FP8_E4M3FN) {
        const unsigned short bits = scales[physical_head];
        // A raw +0 is the reset marker.  Normal writers never emit a
        // negative/non-finite scale, but avoiding a non-finite accumulator
        // here keeps an unwritten row semantically zero on device as on CPU.
        const float scale = ullm_typed_decode_f16_to_f32(bits);
        return bits == 0u || !isfinite(scale) || scale <= 0.0f ? 0.0f : scale;
    }
    (void)scales;
    (void)physical_head;
    return 1.0f;
}

template <int DTYPE>
__device__ __forceinline__ float ullm_typed_decode_load(
    const void *payload,
    unsigned long long index,
    float row_scale) {
    if constexpr (DTYPE == ULLM_TYPED_DECODE_F32) {
        return static_cast<const float *>(payload)[index];
    } else if constexpr (DTYPE == ULLM_TYPED_DECODE_F16) {
        return ullm_typed_decode_f16_to_f32(static_cast<const unsigned short *>(payload)[index]);
    } else {
        return ullm_typed_decode_fp8_e4m3fn_to_f32(
                   static_cast<const unsigned char *>(payload)[index]) * row_scale;
    }
}

template <int K_DTYPE, int V_DTYPE>
__device__ __forceinline__ void ullm_paged_decode_attn_typed_kernel_impl(
    const float *q,
    const void *k_cache,
    const void *v_cache,
    const unsigned int *block_table,
    const unsigned short *k_scales,
    const unsigned short *v_scales,
    unsigned long long cache_len,
    unsigned long long block_size,
    unsigned long long cache_blocks,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float *output,
    float *partial,
    float *shared_scales) {
    if (gridDim.x == q_heads && blockDim.x == 256u && head_dim <= 256ull &&
        value_dim <= 256ull) {
        const unsigned long long q_head = blockIdx.x;
        const unsigned int tid = threadIdx.x;
        const unsigned long long q_per_kv = q_heads / kv_heads;
        const unsigned long long kv_head = q_head / q_per_kv;
        const unsigned long long q_base = q_head * head_dim;
        float max_score = -3.4028234663852886e38f;
        float denominator = 0.0f;
        float weighted = 0.0f;
        unsigned long long block_index = 0ull;
        unsigned long long block_end = block_size;
        unsigned long long block_id = static_cast<unsigned long long>(block_table[0]);
        if (block_id >= cache_blocks) {
            if (tid < value_dim) {
                output[q_head * value_dim + tid] = 0.0f;
            }
            return;
        }
        unsigned long long physical = block_id * block_size;
        for (unsigned long long source = 0ull; source < cache_len; ++source) {
            if (source == block_end) {
                ++block_index;
                block_end += block_size;
                block_id = static_cast<unsigned long long>(block_table[block_index]);
                if (block_id >= cache_blocks) {
                    if (tid < value_dim) {
                        output[q_head * value_dim + tid] = 0.0f;
                    }
                    return;
                }
                physical = block_id * block_size;
            }
            const unsigned long long physical_head = physical * kv_heads + kv_head;
            // One lane loads each FP8 row scale; every thread in the CTA
            // shares it for QK/AV.  This is intentionally outside the
            // payload loops and avoids one scale global load per element.
            if (tid == 0u) {
                shared_scales[0] = ullm_typed_decode_row_scale<K_DTYPE>(k_scales, physical_head);
                shared_scales[1] = ullm_typed_decode_row_scale<V_DTYPE>(v_scales, physical_head);
            }
            __syncthreads();
            const unsigned long long k_base = physical_head * head_dim;
            const float local = tid < head_dim
                ? q[q_base + tid] * ullm_typed_decode_load<K_DTYPE>(
                    k_cache, k_base + tid, shared_scales[0])
                : 0.0f;
            const float score = ullm_paged_decode_reduce_sum_256(local, partial) * softmax_scale;
            if (tid < value_dim) {
                const unsigned long long v_index = physical_head * value_dim + tid;
                const float value = ullm_typed_decode_load<V_DTYPE>(
                    v_cache, v_index, shared_scales[1]);
                if (score > max_score) {
                    const float scale = expf(max_score - score);
                    weighted = weighted * scale + value;
                    denominator = denominator * scale + 1.0f;
                    max_score = score;
                } else {
                    const float weight = expf(score - max_score);
                    weighted += weight * value;
                    denominator += weight;
                }
            }
            // Do not let lane zero overwrite the two row scales while a
            // sibling lane is still consuming the V value from this row.
            __syncthreads();
            ++physical;
        }
        if (tid < value_dim) {
            output[q_head * value_dim + tid] = weighted / denominator;
        }
        return;
    }

    // Shape-generic fallback.  Each output element owns its scalar online
    // softmax state; dtype selection is still compile-time specialization.
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long output_elements = q_heads * value_dim;
    if (index >= output_elements) {
        return;
    }
    const unsigned long long value_dim_index = index % value_dim;
    const unsigned long long q_head = index / value_dim;
    const unsigned long long kv_head = q_head / (q_heads / kv_heads);
    const unsigned long long q_base = q_head * head_dim;
    float max_score = -3.4028234663852886e38f;
    for (unsigned long long source = 0ull; source < cache_len; ++source) {
        const unsigned long long block_index = source / block_size;
        const unsigned long long block_offset = source - block_index * block_size;
        const unsigned long long block_id = static_cast<unsigned long long>(block_table[block_index]);
        if (block_id >= cache_blocks) {
            output[index] = 0.0f;
            return;
        }
        const unsigned long long physical = block_id * block_size + block_offset;
        const unsigned long long physical_head = physical * kv_heads + kv_head;
        const float k_scale = ullm_typed_decode_row_scale<K_DTYPE>(k_scales, physical_head);
        float score = 0.0f;
        const unsigned long long k_base = physical_head * head_dim;
        for (unsigned long long dim = 0ull; dim < head_dim; ++dim) {
            score += q[q_base + dim] * ullm_typed_decode_load<K_DTYPE>(
                k_cache, k_base + dim, k_scale);
        }
        max_score = fmaxf(max_score, score * softmax_scale);
    }
    float denominator = 0.0f;
    float weighted = 0.0f;
    for (unsigned long long source = 0ull; source < cache_len; ++source) {
        const unsigned long long block_index = source / block_size;
        const unsigned long long block_offset = source - block_index * block_size;
        const unsigned long long block_id = static_cast<unsigned long long>(block_table[block_index]);
        if (block_id >= cache_blocks) {
            output[index] = 0.0f;
            return;
        }
        const unsigned long long physical = block_id * block_size + block_offset;
        const unsigned long long physical_head = physical * kv_heads + kv_head;
        const float k_scale = ullm_typed_decode_row_scale<K_DTYPE>(k_scales, physical_head);
        const float v_scale = ullm_typed_decode_row_scale<V_DTYPE>(v_scales, physical_head);
        float score = 0.0f;
        const unsigned long long k_base = physical_head * head_dim;
        for (unsigned long long dim = 0ull; dim < head_dim; ++dim) {
            score += q[q_base + dim] * ullm_typed_decode_load<K_DTYPE>(
                k_cache, k_base + dim, k_scale);
        }
        const float weight = expf(score * softmax_scale - max_score);
        denominator += weight;
        weighted += weight * ullm_typed_decode_load<V_DTYPE>(
            v_cache, physical_head * value_dim + value_dim_index, v_scale);
    }
    output[index] = weighted / denominator;
}

#define ULLM_DEFINE_TYPED_PAGED_DECODE(K_DTYPE, V_DTYPE, K_NAME, V_NAME) \
extern "C" __global__ void ullm_paged_decode_attn_typed_k##K_NAME##_v##V_NAME##_kernel( \
    const float *q, const void *k_cache, const void *v_cache, const unsigned int *block_table, \
    const unsigned short *k_scales, const unsigned short *v_scales, \
    unsigned long long cache_len, unsigned long long block_size, unsigned long long cache_blocks, \
    unsigned long long q_heads, unsigned long long kv_heads, unsigned long long head_dim, \
    unsigned long long value_dim, float softmax_scale, float *output) { \
    __shared__ float partial[256]; \
    __shared__ float shared_scales[2]; \
    ullm_paged_decode_attn_typed_kernel_impl<K_DTYPE, V_DTYPE>( \
        q, k_cache, v_cache, block_table, k_scales, v_scales, cache_len, block_size, cache_blocks, \
        q_heads, kv_heads, head_dim, value_dim, softmax_scale, output, partial, shared_scales); \
}

ULLM_DEFINE_TYPED_PAGED_DECODE(ULLM_TYPED_DECODE_F32, ULLM_TYPED_DECODE_F16, f32, f16)
ULLM_DEFINE_TYPED_PAGED_DECODE(ULLM_TYPED_DECODE_F32, ULLM_TYPED_DECODE_FP8_E4M3FN, f32, fp8)
ULLM_DEFINE_TYPED_PAGED_DECODE(ULLM_TYPED_DECODE_F16, ULLM_TYPED_DECODE_F32, f16, f32)
ULLM_DEFINE_TYPED_PAGED_DECODE(ULLM_TYPED_DECODE_F16, ULLM_TYPED_DECODE_F16, f16, f16)
ULLM_DEFINE_TYPED_PAGED_DECODE(ULLM_TYPED_DECODE_F16, ULLM_TYPED_DECODE_FP8_E4M3FN, f16, fp8)
ULLM_DEFINE_TYPED_PAGED_DECODE(ULLM_TYPED_DECODE_FP8_E4M3FN, ULLM_TYPED_DECODE_F32, fp8, f32)
ULLM_DEFINE_TYPED_PAGED_DECODE(ULLM_TYPED_DECODE_FP8_E4M3FN, ULLM_TYPED_DECODE_F16, fp8, f16)
ULLM_DEFINE_TYPED_PAGED_DECODE(ULLM_TYPED_DECODE_FP8_E4M3FN, ULLM_TYPED_DECODE_FP8_E4M3FN, fp8, fp8)
#undef ULLM_DEFINE_TYPED_PAGED_DECODE

// Typed split partials retain the reviewed F32 merge kernel.  The 16Q/4KV,
// 256-wide Qwen3.5 path is deliberately grouped: a CTA stages one dequantized
// K/V row and its two FP8 scales once for all four query heads sharing that KV
// head.  Other valid geometries use the scalar one-query-head body below.
template <int K_DTYPE, int V_DTYPE>
__device__ __forceinline__ void ullm_paged_decode_attn_split_typed_kernel_impl(
    const float *q, const void *k_cache, const void *v_cache,
    const unsigned int *block_table, const unsigned short *k_scales,
    const unsigned short *v_scales, unsigned long long cache_len,
    unsigned long long block_size, unsigned long long cache_blocks,
    unsigned long long q_heads, unsigned long long kv_heads,
    unsigned long long head_dim, unsigned long long value_dim,
    float softmax_scale, unsigned long long source_tile,
    unsigned long long split_count, float *workspace, float *partial,
    float *shared_scales, float *gqa_stage, float *gqa_state) {
    const unsigned int tid = threadIdx.x;
    const unsigned long long q_per_kv = kv_heads == 0ull ? 0ull : q_heads / kv_heads;

    // Qwen3.5 AQ4_0 decode: four query heads share a KV head.  The host only
    // selects this grid form for this exact geometry, keeping the legacy F32
    // grouped (5x128) experiment completely isolated.
    if (blockDim.x == 256u && q_per_kv == 4ull && head_dim == 256ull &&
        value_dim == 256ull && static_cast<unsigned long long>(gridDim.x) ==
            kv_heads * split_count) {
        const unsigned long long grouped_index = blockIdx.x;
        const unsigned long long kv_head = grouped_index / split_count;
        const unsigned long long split_id = grouped_index - kv_head * split_count;
        if (kv_head >= kv_heads) {
            return;
        }
        const unsigned long long source_start = split_id * source_tile;
        const unsigned long long source_remaining =
            source_start < cache_len ? cache_len - source_start : 0ull;
        const unsigned long long source_count =
            source_tile < source_remaining ? source_tile : source_remaining;
        const unsigned int group = tid >> 6u;
        const unsigned int lane = tid & 63u;
        const unsigned long long q_head = kv_head * 4ull + static_cast<unsigned long long>(group);
        const unsigned long long q_base = q_head * 256ull;
        float q0 = q[q_base + lane];
        float q1 = q[q_base + 64ull + lane];
        float q2 = q[q_base + 128ull + lane];
        float q3 = q[q_base + 192ull + lane];
        float weighted0 = 0.0f;
        float weighted1 = 0.0f;
        float weighted2 = 0.0f;
        float weighted3 = 0.0f;
        float max_score = -3.4028234663852886e38f;
        float denominator = 0.0f;
        bool invalid = false;
        for (unsigned long long offset = 0ull; offset < source_count; ++offset) {
            const unsigned long long source = source_start + offset;
            const unsigned long long logical_block = source / block_size;
            const unsigned long long block_offset = source - logical_block * block_size;
            const unsigned long long block_id =
                static_cast<unsigned long long>(block_table[logical_block]);
            const bool row_valid = block_id < cache_blocks;
            if (tid == 0u) {
                gqa_state[0] = row_valid ? 1.0f : 0.0f;
                if (row_valid) {
                    const unsigned long long physical = block_id * block_size + block_offset;
                    const unsigned long long physical_head = physical * kv_heads + kv_head;
                    shared_scales[0] = ullm_typed_decode_row_scale<K_DTYPE>(
                        k_scales, physical_head);
                    shared_scales[1] = ullm_typed_decode_row_scale<V_DTYPE>(
                        v_scales, physical_head);
                }
            }
            __syncthreads();
            if (gqa_state[0] == 0.0f) {
                invalid = true;
                break;
            }
            const unsigned long long physical = block_id * block_size + block_offset;
            const unsigned long long physical_head = physical * kv_heads + kv_head;
            for (unsigned int element = tid; element < 512u; element += blockDim.x) {
                if (element < 256u) {
                    gqa_stage[element] = ullm_typed_decode_load<K_DTYPE>(
                        k_cache, physical_head * 256ull + element, shared_scales[0]);
                } else {
                    gqa_stage[element] = ullm_typed_decode_load<V_DTYPE>(
                        v_cache, physical_head * 256ull + (element - 256u), shared_scales[1]);
                }
            }
            __syncthreads();
            partial[tid] = q0 * gqa_stage[lane] + q1 * gqa_stage[64u + lane] +
                q2 * gqa_stage[128u + lane] + q3 * gqa_stage[192u + lane];
            for (unsigned int reduce = 32u; reduce > 0u; reduce >>= 1u) {
                if (lane < reduce) {
                    partial[tid] += partial[tid + reduce];
                }
                __syncthreads();
            }
            if (lane == 0u) {
                const float score = partial[tid] * softmax_scale;
                const float new_max = fmaxf(max_score, score);
                gqa_state[4u + group] =
                    max_score <= -3.0e38f ? 0.0f : expf(max_score - new_max);
                gqa_state[8u + group] = expf(score - new_max);
                gqa_state[12u + group] = new_max;
                denominator = denominator * gqa_state[4u + group] + gqa_state[8u + group];
                gqa_state[16u + group] = denominator;
                max_score = new_max;
            }
            __syncthreads();
            const float previous = gqa_state[4u + group];
            const float weight = gqa_state[8u + group];
            weighted0 = weighted0 * previous + weight * gqa_stage[256u + lane];
            weighted1 = weighted1 * previous + weight * gqa_stage[320u + lane];
            weighted2 = weighted2 * previous + weight * gqa_stage[384u + lane];
            weighted3 = weighted3 * previous + weight * gqa_stage[448u + lane];
            __syncthreads();
        }
        const unsigned long long stride = value_dim + 2ull;
        const unsigned long long workspace_base = (q_head * split_count + split_id) * stride;
        if (lane == 0u) {
            workspace[workspace_base] = invalid ? 0.0f : max_score;
            workspace[workspace_base + 1ull] = invalid ? -1.0f : denominator;
        }
        workspace[workspace_base + 2ull + lane] = invalid ? 0.0f : weighted0;
        workspace[workspace_base + 66ull + lane] = invalid ? 0.0f : weighted1;
        workspace[workspace_base + 130ull + lane] = invalid ? 0.0f : weighted2;
        workspace[workspace_base + 194ull + lane] = invalid ? 0.0f : weighted3;
        return;
    }

    if (blockDim.x != 256u || head_dim > 256ull || value_dim > 256ull) {
        return;
    }
    const unsigned long long partial_index = blockIdx.x;
    const unsigned long long q_head = partial_index / split_count;
    const unsigned long long split_id = partial_index - q_head * split_count;
    if (q_head >= q_heads) {
        return;
    }
    const unsigned long long stride = value_dim + 2ull;
    const unsigned long long workspace_base = partial_index * stride;
    const unsigned long long source_start = split_id * source_tile;
    const unsigned long long source_remaining =
        source_start < cache_len ? cache_len - source_start : 0ull;
    const unsigned long long source_count =
        source_tile < source_remaining ? source_tile : source_remaining;
    const unsigned long long q_head_kv = q_head / q_per_kv;
    const unsigned long long q_base = q_head * head_dim;
    float max_score = -3.4028234663852886e38f;
    float denominator = 0.0f;
    float weighted = 0.0f;
    for (unsigned long long offset = 0ull; offset < source_count; ++offset) {
        const unsigned long long source = source_start + offset;
        const unsigned long long logical_block = source / block_size;
        const unsigned long long block_offset = source - logical_block * block_size;
        const unsigned long long block_id =
            static_cast<unsigned long long>(block_table[logical_block]);
        if (block_id >= cache_blocks) {
            if (tid == 0u) {
                workspace[workspace_base] = 0.0f;
                workspace[workspace_base + 1ull] = -1.0f;
            }
            if (tid < value_dim) {
                workspace[workspace_base + 2ull + tid] = 0.0f;
            }
            return;
        }
        const unsigned long long physical = block_id * block_size + block_offset;
        const unsigned long long physical_head = physical * kv_heads + q_head_kv;
        if (tid == 0u) {
            shared_scales[0] = ullm_typed_decode_row_scale<K_DTYPE>(k_scales, physical_head);
            shared_scales[1] = ullm_typed_decode_row_scale<V_DTYPE>(v_scales, physical_head);
        }
        __syncthreads();
        float dot = 0.0f;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            dot += q[q_base + dim] * ullm_typed_decode_load<K_DTYPE>(
                k_cache, physical_head * head_dim + dim, shared_scales[0]);
        }
        const float score = ullm_paged_decode_reduce_sum_256(dot, partial) * softmax_scale;
        if (tid < value_dim) {
            const float value = ullm_typed_decode_load<V_DTYPE>(
                v_cache, physical_head * value_dim + tid, shared_scales[1]);
            if (score > max_score) {
                const float rescale = expf(max_score - score);
                weighted = weighted * rescale + value;
                denominator = denominator * rescale + 1.0f;
                max_score = score;
            } else {
                const float weight = expf(score - max_score);
                weighted += weight * value;
                denominator += weight;
            }
        }
        __syncthreads();
    }
    if (tid == 0u) {
        workspace[workspace_base] = max_score;
        workspace[workspace_base + 1ull] = denominator;
    }
    if (tid < value_dim) {
        workspace[workspace_base + 2ull + tid] = weighted;
    }
}

#define ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT(K_DTYPE, V_DTYPE, K_NAME, V_NAME) \
extern "C" __global__ void ullm_paged_decode_attn_split_typed_k##K_NAME##_v##V_NAME##_partial_kernel( \
    const float *q, const void *k_cache, const void *v_cache, const unsigned int *block_table, \
    const unsigned short *k_scales, const unsigned short *v_scales, \
    unsigned long long cache_len, unsigned long long block_size, unsigned long long cache_blocks, \
    unsigned long long q_heads, unsigned long long kv_heads, unsigned long long head_dim, \
    unsigned long long value_dim, float softmax_scale, unsigned long long source_tile, \
    unsigned long long split_count, float *workspace) { \
    __shared__ float partial[256]; \
    __shared__ float shared_scales[2]; \
    __shared__ float gqa_stage[512]; \
    __shared__ float gqa_state[20]; \
    ullm_paged_decode_attn_split_typed_kernel_impl<K_DTYPE, V_DTYPE>( \
        q, k_cache, v_cache, block_table, k_scales, v_scales, cache_len, block_size, cache_blocks, \
        q_heads, kv_heads, head_dim, value_dim, softmax_scale, source_tile, split_count, workspace, \
        partial, shared_scales, gqa_stage, gqa_state); \
}

ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT(ULLM_TYPED_DECODE_F32, ULLM_TYPED_DECODE_F16, f32, f16)
ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT(ULLM_TYPED_DECODE_F32, ULLM_TYPED_DECODE_FP8_E4M3FN, f32, fp8)
ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT(ULLM_TYPED_DECODE_F16, ULLM_TYPED_DECODE_F32, f16, f32)
ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT(ULLM_TYPED_DECODE_F16, ULLM_TYPED_DECODE_F16, f16, f16)
ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT(ULLM_TYPED_DECODE_F16, ULLM_TYPED_DECODE_FP8_E4M3FN, f16, fp8)
ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT(ULLM_TYPED_DECODE_FP8_E4M3FN, ULLM_TYPED_DECODE_F32, fp8, f32)
ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT(ULLM_TYPED_DECODE_FP8_E4M3FN, ULLM_TYPED_DECODE_F16, fp8, f16)
ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT(ULLM_TYPED_DECODE_FP8_E4M3FN, ULLM_TYPED_DECODE_FP8_E4M3FN, fp8, fp8)
#undef ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT
)";
    }

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

    static const char *depthwise_conv1d_kernel_source() {
        return R"(
extern "C" __global__ void ullm_depthwise_conv1d_f32_kernel(
    const float *input,
    const float *weight,
    unsigned long long channels,
    unsigned long long sequence_len,
    unsigned long long kernel_size,
    float *output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long elements = channels * sequence_len;
    if (index >= elements) {
        return;
    }
    const unsigned long long timestep = index / channels;
    const unsigned long long channel = index - timestep * channels;
    float sum = 0.0f;
    for (unsigned long long kernel = 0; kernel < kernel_size; ++kernel) {
        const unsigned long long left_padding = kernel_size - 1 - kernel;
        if (timestep < left_padding) {
            continue;
        }
        const unsigned long long source_timestep = timestep - left_padding;
        sum += input[source_timestep * channels + channel] *
               weight[channel * kernel_size + kernel];
    }
    output[index] = sum;
}
)";
    }

    // Superseded scalar-reduction linear-attention QKV prepare reference. It remains the
    // rollback source and is still used for every architecture or geometry outside the validated
    // gfx1201 Qwen3.5 M=1 shape.
    [[maybe_unused]] static const char *linear_attn_qkv_prepare_scalar_reference_kernel_source() {
        return R"(
__device__ float ullm_linear_attn_qkv_conv_step_value(
    const float *qkv,
    const float *conv_weight,
    float *conv_history,
    unsigned long long channels,
    unsigned long long kernel_size,
    unsigned long long channel) {
    for (unsigned long long kernel = 0; kernel + 1 < kernel_size; ++kernel) {
        conv_history[kernel * channels + channel] =
            conv_history[(kernel + 1ull) * channels + channel];
    }
    conv_history[(kernel_size - 1ull) * channels + channel] = qkv[channel];

    float sum = 0.0f;
    for (unsigned long long kernel = 0; kernel < kernel_size; ++kernel) {
        sum += conv_history[kernel * channels + channel] *
               conv_weight[channel * kernel_size + kernel];
    }
    const float sigmoid = 1.0f / (1.0f + expf(-sum));
    return sum * sigmoid;
}

extern "C" __global__ void ullm_linear_attn_qkv_prepare_f32_kernel(
    const float *qkv,
    const float *conv_weight,
    float *conv_history,
    unsigned long long key_heads,
    unsigned long long value_heads,
    unsigned long long key_dim,
    unsigned long long value_dim,
    unsigned long long kernel_size,
    float q_scale,
    int qk_l2_norm,
    float *conv_output,
    float *q_output,
    float *k_output,
    float *v_output) {
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    const unsigned long long q_elements = key_heads * key_dim;
    const unsigned long long k_base = q_elements;
    const unsigned long long v_base = q_elements * 2ull;
    const unsigned long long v_elements = value_heads * value_dim;
    const unsigned long long channels = q_elements + q_elements + v_elements;
    const unsigned long long block = blockIdx.x;

    if (block < key_heads) {
        const unsigned long long head = block;
        const unsigned long long source_base = head * key_dim;
        float sum_squares = 0.0f;
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const unsigned long long channel = source_base + dim;
            const float value = ullm_linear_attn_qkv_conv_step_value(
                qkv,
                conv_weight,
                conv_history,
                channels,
                kernel_size,
                channel);
            conv_output[channel] = value;
            sum_squares += value * value;
        }
        partial[tid] = sum_squares;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            __syncthreads();
        }
        const float norm = sqrtf(partial[0] + 1.0e-6f);
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const unsigned long long channel = source_base + dim;
            const float value = conv_output[channel];
            q_output[head * key_dim + dim] =
                qk_l2_norm != 0 ? (value / norm) * q_scale : value * q_scale;
        }
        return;
    }

    if (block < key_heads * 2ull) {
        const unsigned long long head = block - key_heads;
        const unsigned long long source_base = k_base + head * key_dim;
        float sum_squares = 0.0f;
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const unsigned long long channel = source_base + dim;
            const float value = ullm_linear_attn_qkv_conv_step_value(
                qkv,
                conv_weight,
                conv_history,
                channels,
                kernel_size,
                channel);
            conv_output[channel] = value;
            sum_squares += value * value;
        }
        partial[tid] = sum_squares;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            __syncthreads();
        }
        const float norm = sqrtf(partial[0] + 1.0e-6f);
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const unsigned long long channel = source_base + dim;
            const float value = conv_output[channel];
            k_output[head * key_dim + dim] = qk_l2_norm != 0 ? value / norm : value;
        }
        return;
    }

    const unsigned long long v_block = block - key_heads * 2ull;
    const unsigned long long v_index = v_block * blockDim.x + tid;
    if (v_index < v_elements) {
        const unsigned long long channel = v_base + v_index;
        const float value = ullm_linear_attn_qkv_conv_step_value(
            qkv,
            conv_weight,
            conv_history,
            channels,
            kernel_size,
            channel);
        conv_output[channel] = value;
        v_output[v_index] = value;
    }
}

extern "C" __global__ void ullm_linear_attn_qkv_conv_step_silu_f32_kernel(
    const float *qkv,
    const float *conv_weight,
    float *conv_history,
    unsigned long long channels,
    unsigned long long kernel_size,
    float *conv_output) {
    const unsigned long long channel =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= channels) {
        return;
    }
    for (unsigned long long kernel = 0; kernel + 1 < kernel_size; ++kernel) {
        conv_history[kernel * channels + channel] =
            conv_history[(kernel + 1ull) * channels + channel];
    }
    conv_history[(kernel_size - 1ull) * channels + channel] = qkv[channel];

    float sum = 0.0f;
    for (unsigned long long kernel = 0; kernel < kernel_size; ++kernel) {
        sum += conv_history[kernel * channels + channel] *
               conv_weight[channel * kernel_size + kernel];
    }
    const float sigmoid = 1.0f / (1.0f + expf(-sum));
    conv_output[channel] = sum * sigmoid;
}

extern "C" __global__ void ullm_linear_attn_qkv_split_l2norm_f32_kernel(
    const float *conv_output,
    unsigned long long key_heads,
    unsigned long long value_heads,
    unsigned long long key_dim,
    unsigned long long value_dim,
    float q_scale,
    int qk_l2_norm,
    float *q_output,
    float *k_output,
    float *v_output) {
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    const unsigned long long q_elements = key_heads * key_dim;
    const unsigned long long k_base = q_elements;
    const unsigned long long v_base = q_elements * 2ull;
    const unsigned long long v_elements = value_heads * value_dim;
    const unsigned long long block = blockIdx.x;

    if (block < key_heads) {
        const unsigned long long head = block;
        const unsigned long long source_base = head * key_dim;
        float sum_squares = 0.0f;
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const float value = conv_output[source_base + dim];
            sum_squares += value * value;
        }
        partial[tid] = sum_squares;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            __syncthreads();
        }
        const float norm = sqrtf(partial[0] + 1.0e-6f);
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const float value = conv_output[source_base + dim];
            q_output[head * key_dim + dim] =
                qk_l2_norm != 0 ? (value / norm) * q_scale : value * q_scale;
        }
        return;
    }

    if (block < key_heads * 2ull) {
        const unsigned long long head = block - key_heads;
        const unsigned long long source_base = k_base + head * key_dim;
        float sum_squares = 0.0f;
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const float value = conv_output[source_base + dim];
            sum_squares += value * value;
        }
        partial[tid] = sum_squares;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            __syncthreads();
        }
        const float norm = sqrtf(partial[0] + 1.0e-6f);
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const float value = conv_output[source_base + dim];
            k_output[head * key_dim + dim] = qk_l2_norm != 0 ? value / norm : value;
        }
        return;
    }

    const unsigned long long v_block = block - key_heads * 2ull;
    const unsigned long long v_index = v_block * blockDim.x + tid;
    if (v_index < v_elements) {
        v_output[v_index] = conv_output[v_base + v_index];
    }
}
)";
    }

    // Production linear-attention QKV prepare body. The host ABI and HIPRTC entry-point name
    // remain unchanged. On gfx1201, only the validated Qwen3.5 Q/K CTAs replace their scalar
    // tree with the wave32 reduction; V CTAs retain their reduction-free path exactly as-is.
    static std::string linear_attn_qkv_prepare_kernel_source() {
        std::string production = linear_attn_qkv_prepare_scalar_reference_kernel_source();
        const std::string lds_setup = R"(    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];)";
        const std::string shuffle_setup = R"(    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    // Q/K CTAs use these eight wave32 leaders only for the exact Qwen3.5 geometry below.
    // V CTAs never access either reduction scratch because their path has no reduction.
    __shared__ float wave_partials[8];)";
        const size_t setup_offset = production.find(lds_setup);
        if (setup_offset == std::string::npos) {
            return "";
        }
        production.replace(setup_offset, lds_setup.size(), shuffle_setup);

        const std::string shape_setup = R"(    const unsigned long long channels = q_elements + q_elements + v_elements;
    const unsigned long long block = blockIdx.x;)";
        const std::string shuffle_shape_setup = R"(    const unsigned long long channels = q_elements + q_elements + v_elements;
    const unsigned long long block = blockIdx.x;
    const bool use_qwen35_shuffle =
        key_heads == 16ull &&
        value_heads == 32ull &&
        key_dim == 128ull &&
        value_dim == 128ull &&
        kernel_size == 4ull;)";
        const size_t shape_offset = production.find(shape_setup);
        if (shape_offset == std::string::npos) {
            return "";
        }
        production.replace(shape_offset, shape_setup.size(), shuffle_shape_setup);

        const std::string lds_reduction = R"(        partial[tid] = sum_squares;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            __syncthreads();
        }
        const float norm = sqrtf(partial[0] + 1.0e-6f);)";
        const std::string shuffle_reduction = R"(        float reduced_sum_squares = 0.0f;
        if (use_qwen35_shuffle) {
            reduced_sum_squares = ullm_linear_attn_qkv_prepare_shuffle_reduce_to_all(
                sum_squares,
                wave_partials);
        } else {
            partial[tid] = sum_squares;
            __syncthreads();
            for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    partial[tid] += partial[tid + offset];
                }
                __syncthreads();
            }
            reduced_sum_squares = partial[0];
        }
        const float norm = sqrtf(reduced_sum_squares + 1.0e-6f);)";
        const size_t first_reduction = production.find(lds_reduction);
        if (first_reduction == std::string::npos) {
            return "";
        }
        production.replace(first_reduction, lds_reduction.size(), shuffle_reduction);
        const size_t second_reduction = production.find(lds_reduction);
        if (second_reduction == std::string::npos) {
            return "";
        }
        production.replace(second_reduction, lds_reduction.size(), shuffle_reduction);

        return R"(
__device__ __forceinline__ float ullm_linear_attn_qkv_prepare_shuffle_reduce_to_all(
    float sum,
    float *wave_partials) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % warpSize;
    const unsigned int wave = tid / warpSize;
    for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, warpSize);
    }
    if (lane == 0u) {
        wave_partials[wave] = sum;
    }
    __syncthreads();

    if (wave == 0u) {
        sum = lane < 8u ? wave_partials[lane] : 0.0f;
        for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset, warpSize);
        }
        if (lane == 0u) {
            wave_partials[0] = sum;
        }
    }
    __syncthreads();
    return wave_partials[0];
}
)" + production;
    }

    static std::string linear_attn_qkv_prepare_kernel_source_for_arch(const std::string &arch) {
        return arch == "gfx1201" ? linear_attn_qkv_prepare_kernel_source()
                                 : linear_attn_qkv_prepare_scalar_reference_kernel_source();
    }

    // Direct-only gfx1201 experiment for Qwen3.5's M=1 linear-attention prepare launch. It
    // keeps the scalar served reference intact and gives the candidate a distinct entry point.
    // Only Q/K CTAs use the helper below; V CTAs retain their reduction-free path unchanged.
    static std::string linear_attn_qkv_prepare_shuffle_prototype_kernel_source() {
        std::string prototype = linear_attn_qkv_prepare_scalar_reference_kernel_source();
        const std::string next_kernel =
            "\nextern \"C\" __global__ void ullm_linear_attn_qkv_conv_step_silu_f32_kernel";
        const size_t next_kernel_offset = prototype.find(next_kernel);
        if (next_kernel_offset == std::string::npos) {
            return "";
        }
        prototype.erase(next_kernel_offset);

        const std::string production_name = "ullm_linear_attn_qkv_prepare_f32_kernel";
        const size_t name_offset = prototype.find(production_name);
        if (name_offset == std::string::npos) {
            return "";
        }
        prototype.replace(
            name_offset,
            production_name.size(),
            "ullm_linear_attn_qkv_prepare_shuffle_prototype_f32_kernel");

        const std::string lds_setup = R"(    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];)";
        const std::string shuffle_setup = R"(    const unsigned int tid = threadIdx.x;
    // Eight wave32 leaders are enough for the fixed 256-thread direct launch. V blocks never
    // access this scratch because their path has no Q/K normalization reduction.
    __shared__ float wave_partials[8];)";
        const size_t setup_offset = prototype.find(lds_setup);
        if (setup_offset == std::string::npos) {
            return "";
        }
        prototype.replace(setup_offset, lds_setup.size(), shuffle_setup);

        const std::string lds_reduction = R"(        partial[tid] = sum_squares;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            __syncthreads();
        }
        const float norm = sqrtf(partial[0] + 1.0e-6f);)";
        const std::string shuffle_reduction = R"(        const float norm = sqrtf(
            ullm_linear_attn_qkv_prepare_shuffle_reduce_to_all(
                sum_squares,
                wave_partials) + 1.0e-6f);)";
        size_t first_reduction = prototype.find(lds_reduction);
        if (first_reduction == std::string::npos) {
            return "";
        }
        prototype.replace(first_reduction, lds_reduction.size(), shuffle_reduction);
        const size_t second_reduction = prototype.find(lds_reduction);
        if (second_reduction == std::string::npos) {
            return "";
        }
        prototype.replace(second_reduction, lds_reduction.size(), shuffle_reduction);
        if (prototype.find(lds_reduction) != std::string::npos) {
            return "";
        }

        return R"(
__device__ __forceinline__ float ullm_linear_attn_qkv_prepare_shuffle_reduce_to_all(
    float sum,
    float *wave_partials) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % warpSize;
    const unsigned int wave = tid / warpSize;
    for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, warpSize);
    }
    if (lane == 0u) {
        wave_partials[wave] = sum;
    }
    __syncthreads();

    if (wave == 0u) {
        sum = lane < 8u ? wave_partials[lane] : 0.0f;
        for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset, warpSize);
        }
        if (lane == 0u) {
            wave_partials[0] = sum;
        }
    }
    __syncthreads();
    return wave_partials[0];
}
)" + prototype;
    }

    static const char *linear_attn_qkv_prepare_batch_kernel_source() {
        return R"(
__device__ float ullm_linear_attn_qkv_conv_batch_value(
    const float *qkv,
    const float *conv_weight,
    const float *conv_history,
    unsigned long long sequence_len,
    unsigned long long channels,
    unsigned long long kernel_size,
    unsigned long long token,
    unsigned long long channel) {
    float sum = 0.0f;
    for (unsigned long long kernel = 0; kernel < kernel_size; ++kernel) {
        const long long source_token = static_cast<long long>(token) -
            static_cast<long long>(kernel_size - 1ull - kernel);
        float source = 0.0f;
        if (source_token >= 0) {
            source = qkv[static_cast<unsigned long long>(source_token) * channels + channel];
        } else {
            const unsigned long long history_index = kernel + token + 1ull;
            source = history_index < kernel_size ? conv_history[history_index * channels + channel] : 0.0f;
        }
        sum += source * conv_weight[channel * kernel_size + kernel];
    }
    const float sigmoid = 1.0f / (1.0f + expf(-sum));
    return sum * sigmoid;
}

__device__ void ullm_linear_attn_qkv_conv_batch_update_history(
    const float *qkv,
    float *conv_history,
    unsigned long long sequence_len,
    unsigned long long channels,
    unsigned long long kernel_size,
    unsigned long long channel) {
    if (sequence_len >= kernel_size) {
        const unsigned long long first_token = sequence_len - kernel_size;
        for (unsigned long long kernel = 0; kernel < kernel_size; ++kernel) {
            conv_history[kernel * channels + channel] =
                qkv[(first_token + kernel) * channels + channel];
        }
        return;
    }
    const unsigned long long preserved = kernel_size - sequence_len;
    for (unsigned long long kernel = 0; kernel < preserved; ++kernel) {
        conv_history[kernel * channels + channel] =
            conv_history[(kernel + sequence_len) * channels + channel];
    }
    for (unsigned long long token = 0; token < sequence_len; ++token) {
        conv_history[(preserved + token) * channels + channel] =
            qkv[token * channels + channel];
    }
}

extern "C" __global__ void ullm_linear_attn_qkv_prepare_batch_update_history_f32_kernel(
    const float *qkv,
    float *conv_history,
    unsigned long long sequence_len,
    unsigned long long channels,
    unsigned long long kernel_size) {
    const unsigned long long channel =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= channels) {
        return;
    }
    ullm_linear_attn_qkv_conv_batch_update_history(
        qkv,
        conv_history,
        sequence_len,
        channels,
        kernel_size,
        channel);
}

extern "C" __global__ void ullm_linear_attn_qkv_prepare_batch_f32_kernel(
    const float *qkv,
    const float *conv_weight,
    float *conv_history,
    unsigned long long key_heads,
    unsigned long long value_heads,
    unsigned long long key_dim,
    unsigned long long value_dim,
    unsigned long long kernel_size,
    unsigned long long sequence_len,
    float q_scale,
    int qk_l2_norm,
    float *conv_output,
    float *q_output,
    float *k_output,
    float *v_output) {
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    const unsigned long long q_elements = key_heads * key_dim;
    const unsigned long long k_base = q_elements;
    const unsigned long long v_base = q_elements * 2ull;
    const unsigned long long v_elements = value_heads * value_dim;
    const unsigned long long channels = q_elements + q_elements + v_elements;
    const unsigned long long block = blockIdx.x;
    const unsigned long long token = blockIdx.y;
    if (token >= sequence_len) {
        return;
    }

    if (block < key_heads) {
        const unsigned long long head = block;
        const unsigned long long source_base = head * key_dim;
        float sum_squares = 0.0f;
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const unsigned long long channel = source_base + dim;
            const float value = ullm_linear_attn_qkv_conv_batch_value(
                qkv,
                conv_weight,
                conv_history,
                sequence_len,
                channels,
                kernel_size,
                token,
                channel);
            conv_output[token * channels + channel] = value;
            sum_squares += value * value;
        }
        partial[tid] = sum_squares;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            __syncthreads();
        }
        const float norm = sqrtf(partial[0] + 1.0e-6f);
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const unsigned long long channel = source_base + dim;
            const float value = conv_output[token * channels + channel];
            q_output[token * q_elements + head * key_dim + dim] =
                qk_l2_norm != 0 ? (value / norm) * q_scale : value * q_scale;
        }
        return;
    }

    if (block < key_heads * 2ull) {
        const unsigned long long head = block - key_heads;
        const unsigned long long source_base = k_base + head * key_dim;
        float sum_squares = 0.0f;
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const unsigned long long channel = source_base + dim;
            const float value = ullm_linear_attn_qkv_conv_batch_value(
                qkv,
                conv_weight,
                conv_history,
                sequence_len,
                channels,
                kernel_size,
                token,
                channel);
            conv_output[token * channels + channel] = value;
            sum_squares += value * value;
        }
        partial[tid] = sum_squares;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            __syncthreads();
        }
        const float norm = sqrtf(partial[0] + 1.0e-6f);
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const unsigned long long channel = source_base + dim;
            const float value = conv_output[token * channels + channel];
            k_output[token * q_elements + head * key_dim + dim] =
                qk_l2_norm != 0 ? value / norm : value;
        }
        return;
    }

    const unsigned long long v_block = block - key_heads * 2ull;
    const unsigned long long v_index = v_block * blockDim.x + tid;
    if (v_index < v_elements) {
        const unsigned long long channel = v_base + v_index;
        const float value = ullm_linear_attn_qkv_conv_batch_value(
            qkv,
            conv_weight,
            conv_history,
            sequence_len,
            channels,
            kernel_size,
            token,
            channel);
        conv_output[token * channels + channel] = value;
        v_output[token * v_elements + v_index] = value;
    }
}
)";
    }

    static const char *linear_attn_gate_beta_kernel_source() {
        return R"(
extern "C" __global__ void ullm_linear_attn_gate_beta_f32_kernel(
    const float *a,
    const float *b,
    const float *a_log,
    const float *dt_bias,
    unsigned long long heads,
    unsigned long long sequence_len,
    float *gate_output,
    float *beta_output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long elements = heads * sequence_len;
    if (index >= elements) {
        return;
    }
    const unsigned long long head = index - (index / heads) * heads;
    const float x = a[index] + dt_bias[head];
    const float softplus = x <= 20.0f ? logf(1.0f + expf(x)) : x;
    gate_output[index] = -expf(a_log[head]) * softplus;
    const float b_value = b[index];
    beta_output[index] = 1.0f / (1.0f + expf(-b_value));
}
)";
    }

    // Generic stable-ABI fallback for recurrent shapes outside the Qwen3.5 shuffle
    // specialization, or if its target-specific code object is unavailable.
    static const char *linear_attn_recurrent_generic_kernel_source() {
        return R"(
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
    if (blockDim.x > 1u) {
        const unsigned long long block = blockIdx.x;
        const unsigned long long value_head = block / value_dim;
        const unsigned long long value = block - value_head * value_dim;
        const unsigned int tid = threadIdx.x;
        __shared__ float partial[256];
        __shared__ float v_prime_shared;
        if (value_head >= value_heads || value >= value_dim) {
            return;
        }
        const unsigned long long key_head_group = value_heads / key_heads;
        const unsigned long long key_head = value_head / key_head_group;
        const unsigned long long state_head_offset = value_head * key_dim * value_dim;

        const bool state_register_resident =
            key_dim <= static_cast<unsigned long long>(blockDim.x);
        const bool active = static_cast<unsigned long long>(tid) < key_dim;
        unsigned long long state_index = 0ull;
        float state_value = 0.0f;
        if (state_register_resident && active) {
            const unsigned long long key = tid;
            state_index = state_head_offset + key * value_dim + value;
            state_value = state[state_index];
        }

        for (unsigned long long timestep = 0; timestep < sequence_len; ++timestep) {
            const unsigned long long value_head_index = timestep * value_heads + value_head;
            const unsigned long long key_head_index = timestep * key_heads + key_head;
            const unsigned long long qk_base = key_head_index * key_dim;
            const unsigned long long v_base = value_head_index * value_dim;
            const float decay = expf(gate[value_head_index]);
            const float beta_value = beta[value_head_index];

            if (state_register_resident) {
                float current = 0.0f;
                float decayed = 0.0f;
                float key_value = 0.0f;
                float query_value = 0.0f;
                if (active) {
                    const unsigned long long key = tid;
                    key_value = k[qk_base + key];
                    query_value = q[qk_base + key];
                    decayed = state_value * decay;
                    current = decayed * key_value;
                }
                partial[tid] = current;
                __syncthreads();
                for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
                    if (tid < offset) {
                        partial[tid] += partial[tid + offset];
                    }
                    __syncthreads();
                }
                if (tid == 0) {
                    v_prime_shared = (v[v_base + value] - partial[0]) * beta_value;
                }
                __syncthreads();

                float sum = 0.0f;
                if (active) {
                    const float updated = decayed + key_value * v_prime_shared;
                    state_value = updated;
                    sum = updated * query_value;
                }
                partial[tid] = sum;
                __syncthreads();
                for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
                    if (tid < offset) {
                        partial[tid] += partial[tid + offset];
                    }
                    __syncthreads();
                }
                if (tid == 0) {
                    output[v_base + value] = partial[0];
                }
                __syncthreads();
                continue;
            }

            float current = 0.0f;
            for (unsigned long long key = tid; key < key_dim; key += blockDim.x) {
                const unsigned long long state_index = state_head_offset + key * value_dim + value;
                const float decayed = state[state_index] * decay;
                state[state_index] = decayed;
                current += decayed * k[qk_base + key];
            }
            partial[tid] = current;
            __syncthreads();
            for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    partial[tid] += partial[tid + offset];
                }
                __syncthreads();
            }
            if (tid == 0) {
                v_prime_shared = (v[v_base + value] - partial[0]) * beta_value;
            }
            __syncthreads();

            float sum = 0.0f;
            for (unsigned long long key = tid; key < key_dim; key += blockDim.x) {
                const unsigned long long state_index = state_head_offset + key * value_dim + value;
                const float updated = state[state_index] + k[qk_base + key] * v_prime_shared;
                state[state_index] = updated;
                sum += updated * q[qk_base + key];
            }
            partial[tid] = sum;
            __syncthreads();
            for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    partial[tid] += partial[tid + offset];
                }
                __syncthreads();
            }
            if (tid == 0) {
                output[v_base + value] = partial[0];
            }
            __syncthreads();
        }
        if (state_register_resident && active) {
            state[state_index] = state_value;
        }
        return;
    }

    const unsigned long long value_head = blockIdx.x;
    if (value_head >= value_heads || threadIdx.x != 0) {
        return;
    }
    const unsigned long long key_head_group = value_heads / key_heads;
    const unsigned long long key_head = value_head / key_head_group;
    const unsigned long long state_head_offset = value_head * key_dim * value_dim;
    for (unsigned long long timestep = 0; timestep < sequence_len; ++timestep) {
        const unsigned long long value_head_index = timestep * value_heads + value_head;
        const unsigned long long key_head_index = timestep * key_heads + key_head;
        const unsigned long long qk_base = key_head_index * key_dim;
        const unsigned long long v_base = value_head_index * value_dim;
        const float decay = expf(gate[value_head_index]);
        const float beta_value = beta[value_head_index];

        for (unsigned long long key = 0; key < key_dim; ++key) {
            const unsigned long long state_key_offset = state_head_offset + key * value_dim;
            for (unsigned long long value = 0; value < value_dim; ++value) {
                state[state_key_offset + value] *= decay;
            }
        }

        for (unsigned long long value = 0; value < value_dim; ++value) {
            float current = 0.0f;
            for (unsigned long long key = 0; key < key_dim; ++key) {
                current += state[state_head_offset + key * value_dim + value] *
                           k[qk_base + key];
            }
            const float v_prime = (v[v_base + value] - current) * beta_value;
            for (unsigned long long key = 0; key < key_dim; ++key) {
                state[state_head_offset + key * value_dim + value] +=
                    k[qk_base + key] * v_prime;
            }
        }

        for (unsigned long long value = 0; value < value_dim; ++value) {
            float sum = 0.0f;
            for (unsigned long long key = 0; key < key_dim; ++key) {
                sum += state[state_head_offset + key * value_dim + value] *
                       q[qk_base + key];
            }
            output[v_base + value] = sum;
        }
    }
}
)";
    }
