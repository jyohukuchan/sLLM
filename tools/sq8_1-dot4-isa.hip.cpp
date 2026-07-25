// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0
//
// Offline-only ISA probe for the SQ8_1 signed-I8 K=32 dot primitive.  This is
// deliberately independent of the runtime's HIPRTC string so the compiler's
// architecture lowering can be audited directly.

#include <hip/hip_runtime.h>

__device__ __forceinline__ int sq8_1_dot4_i32_i8(int lhs, int rhs, int accum) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
    // RDNA3/RDNA4 expose the VOP3P instruction through sudot4.  Both signed
    // controls are true, so this is the v_dot4_i32_i8 semantic baseline even
    // though disassembly spells the opcode v_dot4_i32_iu8.
    return __builtin_amdgcn_sudot4(true, lhs, true, rhs, accum, false);
#elif defined(__gfx1030__) || defined(__gfx942__) || defined(__gfx950__)
    return __builtin_amdgcn_sdot4(lhs, rhs, accum, false);
#else
#error "SQ8_1 dot4 ISA probe requires a supported signed-dot AMDGPU target"
#endif
}

extern "C" __global__ void ullm_sq8_1_dot4_i32_i8_probe(
    const int *lhs,
    const int *rhs,
    int *output) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    int dot = 0;
#pragma unroll
    for (int iteration = 0; iteration < 8; ++iteration) {
        dot = sq8_1_dot4_i32_i8(lhs[index + iteration * 256u], rhs[index + iteration * 256u], dot);
    }
    output[index] = dot;
}
