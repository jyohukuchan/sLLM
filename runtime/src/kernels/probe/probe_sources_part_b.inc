// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
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

