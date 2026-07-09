// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

    static const char *sq_fp8_matvec_kernel_source() {
        return R"(
__device__ float ullm_sq_fp8_e4m3fn_to_f32(unsigned char value) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_cvt_f32_fp8(static_cast<unsigned int>(value), 0);
#else
    const unsigned int raw = static_cast<unsigned int>(value);
    const unsigned int sign = raw >> 7;
    const unsigned int exponent = (raw >> 3) & 0x0fu;
    const unsigned int mantissa = raw & 0x07u;
    if (exponent == 0x0fu && mantissa == 0x07u) {
        return __uint_as_float(0x7fc00000u);
    }
    if (exponent == 0u) {
        const float magnitude = static_cast<float>(mantissa) * 0.001953125f;
        return sign == 0u ? magnitude : -magnitude;
    }
    const unsigned int fp32_bits = (sign << 31) | ((exponent + 120u) << 23) | (mantissa << 20);
    return __uint_as_float(fp32_bits);
#endif
}

extern "C" __global__ void ullm_sq_fp8_matvec_f32_kernel(
    const unsigned char *payload,
    const float *scales,
    const float *input,
    unsigned long long rows,
    unsigned long long cols,
    unsigned int scale_kind,
    unsigned long long scale_block_rows,
    unsigned long long scale_block_cols,
    float *output) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum = 0.0f;
    if (row < rows) {
        const unsigned long long row_offset = static_cast<unsigned long long>(row) * cols;
        unsigned long long blocks_per_row = 1ull;
        if (scale_kind == 2u) {
            blocks_per_row = 1ull + (cols - 1ull) / scale_block_cols;
        }
        for (unsigned long long col = tid; col < cols; col += blockDim.x) {
            float scale = scales[0];
            if (scale_kind == 1u) {
                scale = scales[row];
            } else if (scale_kind == 2u) {
                scale = scales[(static_cast<unsigned long long>(row) / scale_block_rows) *
                                   blocks_per_row +
                               (col / scale_block_cols)];
            }
            const float weight = ullm_sq_fp8_e4m3fn_to_f32(payload[row_offset + col]) * scale;
            sum += weight * input[col];
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

extern "C" __global__ void ullm_sq_fp8_matvec_batch_f32_kernel(
    const unsigned char *payload,
    const float *scales,
    const float *input,
    unsigned long long rows,
    unsigned long long cols,
    unsigned int scale_kind,
    unsigned long long scale_block_rows,
    unsigned long long scale_block_cols,
    unsigned long long batch_count,
    float *output) {
    const unsigned int row = blockIdx.x;
    const unsigned int batch = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum = 0.0f;
    if (row < rows && batch < batch_count) {
        const unsigned long long row_offset = static_cast<unsigned long long>(row) * cols;
        const unsigned long long input_offset = static_cast<unsigned long long>(batch) * cols;
        unsigned long long blocks_per_row = 1ull;
        if (scale_kind == 2u) {
            blocks_per_row = 1ull + (cols - 1ull) / scale_block_cols;
        }
        for (unsigned long long col = tid; col < cols; col += blockDim.x) {
            float scale = scales[0];
            if (scale_kind == 1u) {
                scale = scales[row];
            } else if (scale_kind == 2u) {
                scale = scales[(static_cast<unsigned long long>(row) / scale_block_rows) *
                                   blocks_per_row +
                               (col / scale_block_cols)];
            }
            const float weight = ullm_sq_fp8_e4m3fn_to_f32(payload[row_offset + col]) * scale;
            sum += weight * input[input_offset + col];
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
    if (tid == 0 && row < rows && batch < batch_count) {
        output[static_cast<unsigned long long>(batch) * rows + row] = partial[0];
    }
}

extern "C" __global__ void ullm_sq_fp8_matvec_pair_f32_kernel(
    const unsigned char *left_payload,
    const float *left_scales,
    unsigned long long left_rows,
    unsigned int left_scale_kind,
    unsigned long long left_scale_block_cols,
    const unsigned char *right_payload,
    const float *right_scales,
    unsigned long long right_rows,
    unsigned int right_scale_kind,
    unsigned long long right_scale_block_cols,
    const float *input,
    unsigned long long cols,
    float *left_output,
    float *right_output) {
    const unsigned int row = blockIdx.x;
    const unsigned int matrix_index = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    const bool is_left = matrix_index == 0u;
    const unsigned char *payload = is_left ? left_payload : right_payload;
    const float *scales = is_left ? left_scales : right_scales;
    const unsigned long long rows = is_left ? left_rows : right_rows;
    const unsigned int scale_kind = is_left ? left_scale_kind : right_scale_kind;
    const unsigned long long scale_block_cols =
        is_left ? left_scale_block_cols : right_scale_block_cols;
    float sum = 0.0f;
    if (row < rows) {
        const unsigned long long row_offset = static_cast<unsigned long long>(row) * cols;
        unsigned long long blocks_per_row = 1ull;
        if (scale_kind == 2u) {
            blocks_per_row = (cols + scale_block_cols - 1ull) / scale_block_cols;
        }
        for (unsigned long long col = tid; col < cols; col += blockDim.x) {
            float scale = scales[0];
            if (scale_kind == 1u) {
                scale = scales[row];
            } else if (scale_kind == 2u) {
                scale = scales[static_cast<unsigned long long>(row) * blocks_per_row +
                               (col / scale_block_cols)];
            }
            const float weight = ullm_sq_fp8_e4m3fn_to_f32(payload[row_offset + col]) * scale;
            sum += weight * input[col];
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
        if (is_left) {
            left_output[row] = partial[0];
        } else {
            right_output[row] = partial[0];
        }
    }
}

extern "C" __global__ void ullm_sq_fp8_matvec_triple_f32_kernel(
    const unsigned char *first_payload,
    const float *first_scales,
    unsigned long long first_rows,
    unsigned int first_scale_kind,
    unsigned long long first_scale_block_cols,
    const unsigned char *second_payload,
    const float *second_scales,
    unsigned long long second_rows,
    unsigned int second_scale_kind,
    unsigned long long second_scale_block_cols,
    const unsigned char *third_payload,
    const float *third_scales,
    unsigned long long third_rows,
    unsigned int third_scale_kind,
    unsigned long long third_scale_block_cols,
    const float *input,
    unsigned long long cols,
    float *first_output,
    float *second_output,
    float *third_output) {
    const unsigned int row = blockIdx.x;
    const unsigned int matrix_index = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    const unsigned char *payload =
        matrix_index == 0u ? first_payload : (matrix_index == 1u ? second_payload : third_payload);
    const float *scales =
        matrix_index == 0u ? first_scales : (matrix_index == 1u ? second_scales : third_scales);
    const unsigned long long rows =
        matrix_index == 0u ? first_rows : (matrix_index == 1u ? second_rows : third_rows);
    const unsigned int scale_kind =
        matrix_index == 0u ? first_scale_kind :
        (matrix_index == 1u ? second_scale_kind : third_scale_kind);
    const unsigned long long scale_block_cols =
        matrix_index == 0u ? first_scale_block_cols :
        (matrix_index == 1u ? second_scale_block_cols : third_scale_block_cols);
    float sum = 0.0f;
    if (row < rows) {
        const unsigned long long row_offset = static_cast<unsigned long long>(row) * cols;
        unsigned long long blocks_per_row = 1ull;
        if (scale_kind == 2u) {
            blocks_per_row = (cols + scale_block_cols - 1ull) / scale_block_cols;
        }
        for (unsigned long long col = tid; col < cols; col += blockDim.x) {
            float scale = scales[0];
            if (scale_kind == 1u) {
                scale = scales[row];
            } else if (scale_kind == 2u) {
                scale = scales[static_cast<unsigned long long>(row) * blocks_per_row +
                               (col / scale_block_cols)];
            }
            const float weight = ullm_sq_fp8_e4m3fn_to_f32(payload[row_offset + col]) * scale;
            sum += weight * input[col];
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
        if (matrix_index == 0u) {
            first_output[row] = partial[0];
        } else if (matrix_index == 1u) {
            second_output[row] = partial[0];
        } else {
            third_output[row] = partial[0];
        }
    }
}
)";
    }
