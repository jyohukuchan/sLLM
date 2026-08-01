// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0
//
// Isolated, non-production SQ8_0 gfx1201 Phase 2 candidate. It deliberately
// changes reduction grouping and is not included by the runtime HIPRTC source.

#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "SQ8_0 Phase 2 candidate is gfx1201-only"
#endif

typedef unsigned int ullm_sq8_phase2_uint4 __attribute__((ext_vector_type(4)));

static_assert(sizeof(ullm_sq8_phase2_uint4) == 16u, "uint4 payload load must be 16 bytes");
static_assert(alignof(ullm_sq8_phase2_uint4) == 16u, "uint4 payload load must be aligned");

constexpr unsigned int kUllmSq8Phase2WaveSize = 32u;

template <unsigned int Selector>
__device__ __forceinline__ float ullm_sq8_phase2_fp8(unsigned int packed) {
    return __builtin_amdgcn_cvt_f32_fp8(packed, Selector);
}

__device__ __forceinline__ float ullm_sq8_phase2_accumulate_uint4(
    ullm_sq8_phase2_uint4 packed,
    const float* input,
    float scale,
    float sum) {
    sum = fmaf(ullm_sq8_phase2_fp8<0u>(packed.x) * scale, input[0], sum);
    sum = fmaf(ullm_sq8_phase2_fp8<1u>(packed.x) * scale, input[1], sum);
    sum = fmaf(ullm_sq8_phase2_fp8<2u>(packed.x) * scale, input[2], sum);
    sum = fmaf(ullm_sq8_phase2_fp8<3u>(packed.x) * scale, input[3], sum);
    sum = fmaf(ullm_sq8_phase2_fp8<0u>(packed.y) * scale, input[4], sum);
    sum = fmaf(ullm_sq8_phase2_fp8<1u>(packed.y) * scale, input[5], sum);
    sum = fmaf(ullm_sq8_phase2_fp8<2u>(packed.y) * scale, input[6], sum);
    sum = fmaf(ullm_sq8_phase2_fp8<3u>(packed.y) * scale, input[7], sum);
    sum = fmaf(ullm_sq8_phase2_fp8<0u>(packed.z) * scale, input[8], sum);
    sum = fmaf(ullm_sq8_phase2_fp8<1u>(packed.z) * scale, input[9], sum);
    sum = fmaf(ullm_sq8_phase2_fp8<2u>(packed.z) * scale, input[10], sum);
    sum = fmaf(ullm_sq8_phase2_fp8<3u>(packed.z) * scale, input[11], sum);
    sum = fmaf(ullm_sq8_phase2_fp8<0u>(packed.w) * scale, input[12], sum);
    sum = fmaf(ullm_sq8_phase2_fp8<1u>(packed.w) * scale, input[13], sum);
    sum = fmaf(ullm_sq8_phase2_fp8<2u>(packed.w) * scale, input[14], sum);
    sum = fmaf(ullm_sq8_phase2_fp8<3u>(packed.w) * scale, input[15], sum);
    return sum;
}

__device__ __forceinline__ float ullm_sq8_phase2_scale_at(
    const float* scales,
    unsigned long long row,
    unsigned long long col,
    unsigned int scale_kind,
    unsigned long long scale_block_rows,
    unsigned long long scale_block_cols,
    unsigned long long blocks_per_row,
    bool per_row_scale_grid) {
    if (scale_kind == 0u) return scales[0];
    if (scale_kind == 1u) return scales[row];
    const unsigned long long scale_row = per_row_scale_grid ? row : row / scale_block_rows;
    return scales[scale_row * blocks_per_row + col / scale_block_cols];
}

__device__ __forceinline__ float ullm_sq8_phase2_accumulate_row(
    const unsigned char* row_payload,
    const float* input,
    const float* scales,
    unsigned long long row,
    unsigned long long cols,
    unsigned int scale_kind,
    unsigned long long scale_block_rows,
    unsigned long long scale_block_cols,
    unsigned long long blocks_per_row,
    bool per_row_scale_grid,
    unsigned int tid) {
    float sum = 0.0f;
    const unsigned long long segments = (cols + 15ull) / 16ull;
    const bool segment_scale = scale_kind != 2u ||
        (scale_block_cols >= 16ull && (scale_block_cols & 15ull) == 0ull);
    for (unsigned long long segment = tid; segment < segments; segment += blockDim.x) {
        const unsigned long long start = segment * 16ull;
        const unsigned int count = static_cast<unsigned int>(
            (cols - start) < 16ull ? (cols - start) : 16ull);
        const bool wide_aligned = count == 16u &&
            ((reinterpret_cast<unsigned long long>(row_payload + start) & 15ull) == 0ull);
        if (segment_scale && wide_aligned) {
            const float scale = ullm_sq8_phase2_scale_at(
                scales, row, start, scale_kind, scale_block_rows, scale_block_cols,
                blocks_per_row, per_row_scale_grid);
            const ullm_sq8_phase2_uint4 packed =
                *reinterpret_cast<const ullm_sq8_phase2_uint4*>(row_payload + start);
            sum = ullm_sq8_phase2_accumulate_uint4(packed, input + start, scale, sum);
        } else {
            for (unsigned int index = 0u; index < count; ++index) {
                const unsigned long long col = start + index;
                const float scale = ullm_sq8_phase2_scale_at(
                    scales, row, col, scale_kind, scale_block_rows, scale_block_cols,
                    blocks_per_row, per_row_scale_grid);
                sum = fmaf(
                    ullm_sq8_phase2_fp8<0u>(static_cast<unsigned int>(row_payload[col])) * scale,
                    input[col], sum);
            }
        }
    }
    return sum;
}

__device__ __forceinline__ float ullm_sq8_phase2_wave32_sum(float value) {
#pragma unroll
    for (unsigned int offset = kUllmSq8Phase2WaveSize >> 1u; offset > 0u; offset >>= 1u) {
        value += __shfl_down(value, offset, kUllmSq8Phase2WaveSize);
    }
    return value;
}

__device__ __forceinline__ float ullm_sq8_phase2_reduce(float sum, float* wave_partial, unsigned int tid) {
    const unsigned int lane = tid & (kUllmSq8Phase2WaveSize - 1u);
    const unsigned int wave = tid >> 5u;
    const float reduced = ullm_sq8_phase2_wave32_sum(sum);
    if (lane == 0u) wave_partial[wave] = reduced;
    __syncthreads();
    if (tid != 0u) return 0.0f;
    return wave_partial[0] + wave_partial[1] + wave_partial[2] + wave_partial[3] +
        wave_partial[4] + wave_partial[5] + wave_partial[6] + wave_partial[7];
}

extern "C" __global__ __launch_bounds__(256) void ullm_sq_fp8_matvec_f32_kernel(
    const unsigned char* payload,
    const float* scales,
    const float* input,
    unsigned long long rows,
    unsigned long long cols,
    unsigned int scale_kind,
    unsigned long long scale_block_rows,
    unsigned long long scale_block_cols,
    float* output) {
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x);
    const unsigned int tid = threadIdx.x;
    __shared__ float wave_partial[8];
    float sum = 0.0f;
    if (row < rows) {
        const unsigned long long blocks_per_row = scale_kind == 2u
            ? 1ull + (cols - 1ull) / scale_block_cols
            : 1ull;
        sum = ullm_sq8_phase2_accumulate_row(
            payload + row * cols, input, scales, row, cols, scale_kind, scale_block_rows,
            scale_block_cols, blocks_per_row, false, tid);
    }
    const float reduced = ullm_sq8_phase2_reduce(sum, wave_partial, tid);
    if (tid == 0u && row < rows) output[row] = reduced;
}

extern "C" __global__ __launch_bounds__(256) void ullm_sq_fp8_matvec_batch_f32_kernel(
    const unsigned char* payload,
    const float* scales,
    const float* input,
    unsigned long long rows,
    unsigned long long cols,
    unsigned int scale_kind,
    unsigned long long scale_block_rows,
    unsigned long long scale_block_cols,
    unsigned long long batch_count,
    float* output) {
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x);
    const unsigned long long batch = static_cast<unsigned long long>(blockIdx.y);
    const unsigned int tid = threadIdx.x;
    __shared__ float wave_partial[8];
    float sum = 0.0f;
    if (row < rows && batch < batch_count) {
        const unsigned long long blocks_per_row = scale_kind == 2u
            ? 1ull + (cols - 1ull) / scale_block_cols
            : 1ull;
        sum = ullm_sq8_phase2_accumulate_row(
            payload + row * cols, input + batch * cols, scales, row, cols, scale_kind,
            scale_block_rows, scale_block_cols, blocks_per_row, false, tid);
    }
    const float reduced = ullm_sq8_phase2_reduce(sum, wave_partial, tid);
    if (tid == 0u && row < rows && batch < batch_count) output[batch * rows + row] = reduced;
}

extern "C" __global__ __launch_bounds__(256) void ullm_sq_fp8_matvec_pair_f32_kernel(
    const unsigned char* left_payload,
    const float* left_scales,
    unsigned long long left_rows,
    unsigned int left_scale_kind,
    unsigned long long left_scale_block_cols,
    const unsigned char* right_payload,
    const float* right_scales,
    unsigned long long right_rows,
    unsigned int right_scale_kind,
    unsigned long long right_scale_block_cols,
    const float* input,
    unsigned long long cols,
    float* left_output,
    float* right_output) {
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x);
    const unsigned int matrix_index = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const bool is_left = matrix_index == 0u;
    const unsigned char* payload = is_left ? left_payload : right_payload;
    const float* scales = is_left ? left_scales : right_scales;
    const unsigned long long rows = is_left ? left_rows : right_rows;
    const unsigned int scale_kind = is_left ? left_scale_kind : right_scale_kind;
    const unsigned long long scale_block_cols = is_left ? left_scale_block_cols : right_scale_block_cols;
    __shared__ float wave_partial[8];
    float sum = 0.0f;
    if (row < rows) {
        const unsigned long long blocks_per_row = scale_kind == 2u
            ? (cols + scale_block_cols - 1ull) / scale_block_cols
            : 1ull;
        sum = ullm_sq8_phase2_accumulate_row(
            payload + row * cols, input, scales, row, cols, scale_kind, 1ull,
            scale_block_cols, blocks_per_row, true, tid);
    }
    const float reduced = ullm_sq8_phase2_reduce(sum, wave_partial, tid);
    if (tid == 0u && row < rows) {
        if (is_left) left_output[row] = reduced;
        else right_output[row] = reduced;
    }
}

extern "C" __global__ __launch_bounds__(256) void ullm_sq_fp8_matvec_triple_f32_kernel(
    const unsigned char* first_payload,
    const float* first_scales,
    unsigned long long first_rows,
    unsigned int first_scale_kind,
    unsigned long long first_scale_block_cols,
    const unsigned char* second_payload,
    const float* second_scales,
    unsigned long long second_rows,
    unsigned int second_scale_kind,
    unsigned long long second_scale_block_cols,
    const unsigned char* third_payload,
    const float* third_scales,
    unsigned long long third_rows,
    unsigned int third_scale_kind,
    unsigned long long third_scale_block_cols,
    const float* input,
    unsigned long long cols,
    float* first_output,
    float* second_output,
    float* third_output) {
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x);
    const unsigned int matrix_index = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned char* payload = matrix_index == 0u ? first_payload :
        (matrix_index == 1u ? second_payload : third_payload);
    const float* scales = matrix_index == 0u ? first_scales :
        (matrix_index == 1u ? second_scales : third_scales);
    const unsigned long long rows = matrix_index == 0u ? first_rows :
        (matrix_index == 1u ? second_rows : third_rows);
    const unsigned int scale_kind = matrix_index == 0u ? first_scale_kind :
        (matrix_index == 1u ? second_scale_kind : third_scale_kind);
    const unsigned long long scale_block_cols = matrix_index == 0u ? first_scale_block_cols :
        (matrix_index == 1u ? second_scale_block_cols : third_scale_block_cols);
    __shared__ float wave_partial[8];
    float sum = 0.0f;
    if (row < rows) {
        const unsigned long long blocks_per_row = scale_kind == 2u
            ? (cols + scale_block_cols - 1ull) / scale_block_cols
            : 1ull;
        sum = ullm_sq8_phase2_accumulate_row(
            payload + row * cols, input, scales, row, cols, scale_kind, 1ull,
            scale_block_cols, blocks_per_row, true, tid);
    }
    const float reduced = ullm_sq8_phase2_reduce(sum, wave_partial, tid);
    if (tid == 0u && row < rows) {
        if (matrix_index == 0u) first_output[row] = reduced;
        else if (matrix_index == 1u) second_output[row] = reduced;
        else third_output[row] = reduced;
    }
}
