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
#include "kernels/linear_attn/linear_attn_sources_part_a.inc"
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
