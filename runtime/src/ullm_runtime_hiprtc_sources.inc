// Included inside HipRtcRuntime's private section by ullm_runtime.cpp.
// Keep these HIPRTC source-string builders in this file; runtime loading and
// compile helpers stay in ullm_runtime.cpp.

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
        return aq4_rows_per_block_preamble(arch, "ULLM_AQ4_MATVEC_RPB") +
               aq4_matvec_kernel_source();
    }

    static std::string aq4_matvec_batch_kernel_source_for_arch(const std::string &arch) {
        return aq4_rows_per_block_preamble(arch, "ULLM_AQ4_MATVEC_BATCH_RPB") +
               aq4_matvec_batch_kernel_source();
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
        return aq4_fused_rows_per_block_preamble(
                   arch,
                   8u,
                   2u,
                   "ULLM_AQ4_MATVEC_ADD_RPB") +
               aq4_matvec_add_kernel_source();
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
        return aq4_fused_rows_per_block_preamble(
                   arch,
                   8u,
                   4u,
                   "ULLM_AQ4_MATVEC_TRIPLE_RPB") +
               aq4_matvec_triple_kernel_source();
    }

    static std::string aq4_matvec_qkv_z_gate_beta_kernel_source_for_arch(
        const std::string &arch) {
        return aq4_fused_rows_per_block_preamble(
                   arch,
                   4u,
                   2u,
                   "ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB") +
               aq4_matvec_qkv_z_gate_beta_kernel_source();
    }

    static std::string aq4_matvec_silu_mul_kernel_source_for_arch(const std::string &arch) {
        return aq4_fused_rows_per_block_preamble(
                   arch,
                   8u,
                   2u,
                   "ULLM_AQ4_MATVEC_SILU_MUL_RPB") +
               aq4_matvec_silu_mul_kernel_source();
    }

    static std::string aq4_matvec_gate_beta_kernel_source_for_arch(const std::string &arch) {
        return aq4_rows_per_block_preamble(arch, "ULLM_AQ4_MATVEC_RPB") +
               aq4_matvec_gate_beta_kernel_source();
    }

    static const char *aq4_matvec_kernel_source() {
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

    static const char *aq4_matvec_add_kernel_source() {
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

    static const char *aq4_matvec_triple_kernel_source() {
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

    static const char *aq4_matvec_qkv_z_gate_beta_kernel_source() {
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

    static const char *aq4_matvec_silu_mul_kernel_source() {
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

    static const char *rmsnorm_kernel_source() {
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

    static const char *segmented_rmsnorm_silu_mul_kernel_source() {
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
    const unsigned int sign = static_cast<unsigned int>(value) >> 7;
    const unsigned int exponent = (static_cast<unsigned int>(value) >> 3) & 0x0fu;
    const unsigned int mantissa = static_cast<unsigned int>(value) & 0x07u;
    if (exponent == 0u) {
        const float magnitude = static_cast<float>(mantissa) * 0.001953125f;
        return sign == 0u ? magnitude : -magnitude;
    }
    const unsigned int fp32_bits = (sign << 31) | ((exponent + 120u) << 23) | (mantissa << 20);
    return __uint_as_float(fp32_bits);
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

    static std::string paged_decode_attn_kernel_source_for_arch(const std::string &arch) {
        (void)arch;
        std::string preamble;
        if (std::getenv("ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE") != nullptr) {
            preamble += "#define ULLM_PAGED_DECODE_USE_SHARED_REDUCE 1\n";
        }
        if (std::getenv("ULLM_DISABLE_PAGED_DECODE_ONLINE_SOFTMAX") != nullptr) {
            preamble += "#define ULLM_PAGED_DECODE_USE_TWO_PASS_SOFTMAX 1\n";
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
            ++physical_timestep;
        }
#endif
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

    static const char *linear_attn_qkv_prepare_kernel_source() {
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

    static const char *linear_attn_recurrent_kernel_source() {
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

        for (unsigned long long timestep = 0; timestep < sequence_len; ++timestep) {
            const unsigned long long value_head_index = timestep * value_heads + value_head;
            const unsigned long long key_head_index = timestep * key_heads + key_head;
            const unsigned long long qk_base = key_head_index * key_dim;
            const unsigned long long v_base = value_head_index * value_dim;
            const float decay = expf(gate[value_head_index]);
            const float beta_value = beta[value_head_index];

            if (key_dim <= static_cast<unsigned long long>(blockDim.x)) {
                float current = 0.0f;
                float decayed = 0.0f;
                float key_value = 0.0f;
                float query_value = 0.0f;
                unsigned long long state_index = 0ull;
                const bool active = static_cast<unsigned long long>(tid) < key_dim;
                if (active) {
                    const unsigned long long key = tid;
                    state_index = state_head_offset + key * value_dim + value;
                    key_value = k[qk_base + key];
                    query_value = q[qk_base + key];
                    decayed = state[state_index] * decay;
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
                    state[state_index] = updated;
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
