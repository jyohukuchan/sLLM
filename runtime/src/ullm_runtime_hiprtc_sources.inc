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

#include "kernels/aq4/aq4_sources.inc"
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
