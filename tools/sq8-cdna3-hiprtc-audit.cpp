// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0
//
// Compile the generic SQ8_0 runtime kernels for a named AMDGPU target without
// enumerating, opening, or launching a HIP device.  This intentionally reuses
// HipRtcRuntime rather than copying its source strings or compiler options so
// that the result covers the exact HIPRTC programs used by the runtime.

#include "../runtime/src/ullm_runtime.cpp"

#include <array>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using CompileMethod = bool (HipRtcRuntime::*)(
    const std::string&,
    std::vector<char>*,
    std::string*);

struct AuditCase {
    const char* name;
    CompileMethod compile;
};

constexpr std::array<AuditCase, 27> kRequiredSq8_0Cases = {{
    {"matvec_bf16_f32", &HipRtcRuntime::compile_matvec_bf16_kernel},
    {"top1_f32", &HipRtcRuntime::compile_top1_kernel},
    {"rmsnorm_f32", &HipRtcRuntime::compile_rmsnorm_kernel},
    {"segmented_rmsnorm_f32", &HipRtcRuntime::compile_segmented_rmsnorm_kernel},
    {"segmented_rmsnorm_silu_mul_f32", &HipRtcRuntime::compile_segmented_rmsnorm_silu_mul_kernel},
    {"silu_mul_f32", &HipRtcRuntime::compile_silu_mul_kernel},
    {"sigmoid_mul_f32", &HipRtcRuntime::compile_sigmoid_mul_kernel},
    {"add_f32", &HipRtcRuntime::compile_add_kernel},
    {"rope_f32", &HipRtcRuntime::compile_rope_kernel},
    {"causal_attn_f32", &HipRtcRuntime::compile_causal_attn_kernel},
    {"causal_attn_f32_flash2", &HipRtcRuntime::compile_causal_attn_f32_flash2_kernel},
    {"causal_attn_batch_f32", &HipRtcRuntime::compile_causal_attn_batch_kernel},
    {"causal_attn_batch_f32_flash2", &HipRtcRuntime::compile_causal_attn_batch_f32_flash2_kernel},
    {"cached_prefix_attn_f32", &HipRtcRuntime::compile_cached_prefix_attn_kernel},
    {"cached_prefix_attn_f32_flash2", &HipRtcRuntime::compile_cached_prefix_attn_f32_flash2_kernel},
    {"sq_fp8_matvec_f32", &HipRtcRuntime::compile_sq_fp8_matvec_kernel},
    {"sq_fp8_matvec_batch_f32", &HipRtcRuntime::compile_sq_fp8_matvec_batch_kernel},
    {"sq_fp8_matvec_pair_f32", &HipRtcRuntime::compile_sq_fp8_matvec_pair_kernel},
    {"sq_fp8_matvec_triple_f32", &HipRtcRuntime::compile_sq_fp8_matvec_triple_kernel},
    {"paged_decode_attn_f32", &HipRtcRuntime::compile_paged_decode_attn_kernel},
    {"paged_kv_write_f32", &HipRtcRuntime::compile_paged_kv_write_kernel},
    {"paged_chunk_f32", &HipRtcRuntime::compile_paged_chunk_kernel},
    {"qwen35_split_q_gate_f32", &HipRtcRuntime::compile_qwen35_split_q_gate_kernel},
    {"qwen35_qk_norm_rope_f32", &HipRtcRuntime::compile_qwen35_qk_norm_rope_kernel},
    {"qwen35_qk_norm_rope_batch_f32", &HipRtcRuntime::compile_qwen35_qk_norm_rope_batch_kernel},
    {"qwen35_qk_norm_rope_paged_kv_write_f32", &HipRtcRuntime::compile_qwen35_qk_norm_rope_paged_kv_write_kernel},
    {"depthwise_conv1d_f32", &HipRtcRuntime::compile_depthwise_conv1d_kernel},
}};

void usage(const char* program) {
    std::fprintf(stderr, "usage: %s [--arch gfx942]\n", program);
}

} // namespace

int main(int argc, char** argv) {
    std::string arch = "gfx942";
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            return 0;
        }
        if (argument == "--arch" && index + 1 < argc) {
            arch = argv[++index];
            continue;
        }
        usage(argv[0]);
        return 2;
    }
    if (arch != "gfx942") {
        std::fprintf(stderr, "this SQ8_0 CDNA3 audit accepts only --arch gfx942, got %s\n", arch.c_str());
        return 2;
    }

    HipRtcRuntime runtime;
    bool passed = true;
    for (const AuditCase& audit_case : kRequiredSq8_0Cases) {
        std::vector<char> code;
        std::string error;
        if ((runtime.*audit_case.compile)(arch, &code, &error)) {
            std::printf("PASS %s code_bytes=%zu\n", audit_case.name, code.size());
            continue;
        }
        passed = false;
        std::fprintf(
            stderr,
            "FAIL %s\n%s\n",
            audit_case.name,
            error.empty() ? "HIPRTC failed without a diagnostic" : error.c_str());
    }
    return passed ? 0 : 1;
}
