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

#include "kernels/probe/probe_sources_part_b.inc"
#include "kernels/attention/attention_sources_part_a.inc"
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
