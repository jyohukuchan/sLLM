// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0
//
// Offline-only ISA probes for the SQ9_0 versus Q8_0 decision.  These are not
// runtime kernels: no host launcher is linked, no GPU is selected, and the
// fixed K=128 inner loops make static instruction accounting reproducible.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr unsigned kProbeK = 128;
constexpr unsigned kQ8Group = 32;
constexpr unsigned kQ8GroupsPerProbe = kProbeK / kQ8Group;
constexpr unsigned kDot4PerGroup = kQ8Group / 4;
constexpr unsigned kSq9HighBytesPerRow = kProbeK / 8;

static_assert(kProbeK % kQ8Group == 0, "Q8_0 probe must have whole scale blocks");
static_assert(kQ8Group % 4 == 0, "sdot4 probe must have whole packed words");

__device__ __forceinline__ int sdot4(const int packed_weight, const int packed_activation, int acc) {
    return __builtin_amdgcn_sdot4(packed_weight, packed_activation, acc, false);
}

__device__ __forceinline__ __half sq9_0_to_half(
    const std::uint8_t low_byte, const std::uint8_t high_byte, const unsigned bit_index) {
    const std::uint16_t code = static_cast<std::uint16_t>(low_byte) |
                               (static_cast<std::uint16_t>((high_byte >> bit_index) & 1u) << 8);
    // E5M3 and binary16 share the exponent bias.  This is the exact SQ9_0
    // conversion contract: assemble the nine-bit code, then shift it by seven.
    return __ushort_as_half(static_cast<std::uint16_t>(code << 7));
}

__device__ __forceinline__ __half2 pack_half2(const __half low, const __half high) {
    return __halves2half2(low, high);
}

}  // namespace

// Q8_0 W8A8: four 32-value blocks, each eight sdot4 operations followed by
// one conversion and the block's weight/activation scale product.  It is the
// direct-int8-dot baseline used for the comparison.
extern "C" __global__ __launch_bounds__(64) void ullm_q8_0_w8a8_g32_gemv_isa(
    const std::int8_t* __restrict__ weights,
    const std::int8_t* __restrict__ activations,
    const __half* __restrict__ weight_scales,
    const __half* __restrict__ activation_scales,
    float* __restrict__ output) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned weight_base = row * kProbeK;
    const auto* const packed_weights = reinterpret_cast<const int*>(weights + weight_base);
    const auto* const packed_activations = reinterpret_cast<const int*>(activations);
    float sum = 0.0f;

#pragma unroll
    for (unsigned group = 0; group < kQ8GroupsPerProbe; ++group) {
        int dot = 0;
#pragma unroll
        for (unsigned packed = 0; packed < kDot4PerGroup; ++packed) {
            const unsigned index = group * kDot4PerGroup + packed;
            dot = sdot4(packed_weights[index], packed_activations[index], dot);
        }
        const float scale = __half2float(weight_scales[row * kQ8GroupsPerProbe + group]) *
                            __half2float(activation_scales[group]);
        sum = __builtin_fmaf(static_cast<float>(dot), scale, sum);
    }
    output[row] = sum;
}

// SQ9_0 W8A8: values cannot enter the integer dot instruction.  It preserves
// blockwise activation scaling by applying that scale once to each 32-value
// FP32 partial sum; the per-value work remains E5M3 decode + int8 conversion
// + floating FMA.
extern "C" __global__ __launch_bounds__(64) void ullm_sq9_0_w8a8_g32_gemv_isa(
    const std::uint8_t* __restrict__ low_plane,
    const std::uint8_t* __restrict__ high_plane,
    const std::int8_t* __restrict__ activations,
    const __half* __restrict__ activation_scales,
    float* __restrict__ output) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned low_base = row * kProbeK;
    const unsigned high_base = row * kSq9HighBytesPerRow;
    float sum = 0.0f;

#pragma unroll
    for (unsigned group = 0; group < kQ8GroupsPerProbe; ++group) {
        float partial = 0.0f;
#pragma unroll
        for (unsigned lane = 0; lane < kQ8Group; ++lane) {
            const unsigned index = group * kQ8Group + lane;
            const __half weight = sq9_0_to_half(
                low_plane[low_base + index], high_plane[high_base + (index >> 3)], index & 7u);
            partial = __builtin_fmaf(
                __half2float(weight), static_cast<float>(activations[index]), partial);
        }
        sum = __builtin_fmaf(partial, __half2float(activation_scales[group]), sum);
    }
    output[row] = sum;
}

// Q8_0 W8A16 reference-style path.  The necessary signed-int8 conversion and
// per-value multiplication by the weight scale are intentionally explicit.
extern "C" __global__ __launch_bounds__(64) void ullm_q8_0_w8a16_g32_gemv_isa(
    const std::int8_t* __restrict__ weights,
    const __half* __restrict__ activations,
    const __half* __restrict__ weight_scales,
    float* __restrict__ output) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned weight_base = row * kProbeK;
    float sum = 0.0f;

#pragma unroll
    for (unsigned group = 0; group < kQ8GroupsPerProbe; ++group) {
        const float weight_scale = __half2float(weight_scales[row * kQ8GroupsPerProbe + group]);
#pragma unroll
        for (unsigned lane = 0; lane < kQ8Group; ++lane) {
            const unsigned index = group * kQ8Group + lane;
            const float weight = static_cast<float>(weights[weight_base + index]) * weight_scale;
            sum = __builtin_fmaf(weight, __half2float(activations[index]), sum);
        }
    }
    output[row] = sum;
}

// SQ9_0 W8A16 reference-style path.  It has no reconstruction-scale multiply,
// but each weight still requires plane assembly, a shift/bitcast, half-to-f32
// conversion, and a floating FMA.
extern "C" __global__ __launch_bounds__(64) void ullm_sq9_0_w8a16_gemv_isa(
    const std::uint8_t* __restrict__ low_plane,
    const std::uint8_t* __restrict__ high_plane,
    const __half* __restrict__ activations,
    float* __restrict__ output) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned low_base = row * kProbeK;
    const unsigned high_base = row * kSq9HighBytesPerRow;
    float sum = 0.0f;

#pragma unroll
    for (unsigned index = 0; index < kProbeK; ++index) {
        const __half weight = sq9_0_to_half(
            low_plane[low_base + index], high_plane[high_base + (index >> 3)], index & 7u);
        sum = __builtin_fmaf(__half2float(weight), __half2float(activations[index]), sum);
    }
    output[row] = sum;
}

// Prefill-rate companion probes: packed half2 FMA is the favorable FP16
// arithmetic form for SQ9_0.  These use FP16 accumulation only to expose the
// packed instruction mix; they are not numerical-reference kernels.
extern "C" __global__ __launch_bounds__(64) void ullm_q8_0_w8a16_pk_f16_g32_gemv_isa(
    const std::int8_t* __restrict__ weights,
    const __half* __restrict__ activations,
    const __half* __restrict__ weight_scales,
    float* __restrict__ output) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned weight_base = row * kProbeK;
    __half2 sum = __float2half2_rn(0.0f);

#pragma unroll
    for (unsigned group = 0; group < kQ8GroupsPerProbe; ++group) {
        const float weight_scale = __half2float(weight_scales[row * kQ8GroupsPerProbe + group]);
#pragma unroll
        for (unsigned pair = 0; pair < kQ8Group / 2; ++pair) {
            const unsigned index = group * kQ8Group + pair * 2;
            const __half left = __float2half_rn(
                static_cast<float>(weights[weight_base + index]) * weight_scale);
            const __half right = __float2half_rn(
                static_cast<float>(weights[weight_base + index + 1]) * weight_scale);
            const __half2 activation = *reinterpret_cast<const __half2*>(activations + index);
            sum = __hfma2(pack_half2(left, right), activation, sum);
        }
    }
    output[row] = __low2float(sum) + __high2float(sum);
}

extern "C" __global__ __launch_bounds__(64) void ullm_sq9_0_w8a16_pk_f16_gemv_isa(
    const std::uint8_t* __restrict__ low_plane,
    const std::uint8_t* __restrict__ high_plane,
    const __half* __restrict__ activations,
    float* __restrict__ output) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned low_base = row * kProbeK;
    const unsigned high_base = row * kSq9HighBytesPerRow;
    __half2 sum = __float2half2_rn(0.0f);

#pragma unroll
    for (unsigned pair = 0; pair < kProbeK / 2; ++pair) {
        const unsigned index = pair * 2;
        const __half left = sq9_0_to_half(
            low_plane[low_base + index], high_plane[high_base + (index >> 3)], index & 7u);
        const __half right = sq9_0_to_half(
            low_plane[low_base + index + 1],
            high_plane[high_base + ((index + 1) >> 3)],
            (index + 1) & 7u);
        const __half2 activation = *reinterpret_cast<const __half2*>(activations + index);
        sum = __hfma2(pack_half2(left, right), activation, sum);
    }
    output[row] = __low2float(sum) + __high2float(sum);
}
