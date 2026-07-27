// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
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

    // Superseded scalar-reduction RMSNorm reference. It is deliberately not passed to HIPRTC
    // for gfx1201: production keeps the same entry-point ABI but uses the validated wave32
    // shuffle reduction below. Retain this source for controlled rollback comparisons and for
    // generic HIPRTC compilations outside that exact architecture.
    [[maybe_unused]] static const char *rmsnorm_scalar_reference_kernel_source() {
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

    // Production RMSNorm body. The host ABI and HIPRTC entry-point name remain unchanged, so
    // Qwen3.5 decode continues through its existing RMSNorm dispatch.
    static std::string rmsnorm_kernel_source() {
        std::string production = rmsnorm_scalar_reference_kernel_source();
        const std::string lds_setup = R"(    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum = 0.0f;)";
        const std::string shuffle_setup = R"(    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % warpSize;
    const unsigned int wave = tid / warpSize;
    __shared__ float wave_partials[8];
    float sum = 0.0f;)";
        const size_t setup_offset = production.find(lds_setup);
        if (setup_offset == std::string::npos) return "";
        production.replace(setup_offset, lds_setup.size(), shuffle_setup);

        const std::string lds_reduction = R"(    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(elements) + epsilon);)";
        const std::string shuffle_reduction = R"(    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, warpSize);
    }
    if (lane == 0u) {
        wave_partials[wave] = sum;
    }
    __syncthreads();

    if (wave == 0u) {
        const unsigned int wave_count = (blockDim.x + warpSize - 1u) / warpSize;
        sum = lane < wave_count ? wave_partials[lane] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset, warpSize);
        }
        if (lane == 0u) {
            wave_partials[0] = sum;
        }
    }
    __syncthreads();
    const float inv_rms = rsqrtf(wave_partials[0] / static_cast<float>(elements) + epsilon);)";
        const size_t reduction_offset = production.find(lds_reduction);
        if (reduction_offset == std::string::npos) return "";
        production.replace(reduction_offset, lds_reduction.size(), shuffle_reduction);
        return production;
    }

    // Keep the direct-only prototype ABI available for historical differential/timing comparison.
    // It compiles the same validated body under its original, isolated entry-point name.
    static std::string rmsnorm_shuffle_prototype_kernel_source() {
        std::string prototype = rmsnorm_kernel_source();
        const std::string production_name = "ullm_rmsnorm_f32_kernel";
        const size_t name_offset = prototype.find(production_name);
        if (name_offset == std::string::npos) return "";
        prototype.replace(
            name_offset,
            production_name.size(),
            "ullm_rmsnorm_shuffle_prototype_f32_kernel");
        return prototype;
    }

    static std::string rmsnorm_kernel_source_for_arch(const std::string &arch) {
        return arch == "gfx1201" ? rmsnorm_kernel_source()
                                 : rmsnorm_scalar_reference_kernel_source();
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

    // Superseded scalar-reduction segmented RMSNorm SiLU-mul reference. It is deliberately not
    // passed to HIPRTC for gfx1201: production keeps the same entry-point ABI but uses the
    // validated wave32 shuffle reduction below. Retain this source for controlled rollback
    // comparisons and for generic HIPRTC compilations outside that exact architecture.
    [[maybe_unused]] static const char *segmented_rmsnorm_silu_mul_scalar_reference_kernel_source() {
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

    // Production segmented RMSNorm SiLU-mul body. The host ABI and HIPRTC entry-point name
    // remain unchanged, so Qwen3.5 decode continues through its existing linear-attention post
    // dispatch.
    static std::string segmented_rmsnorm_silu_mul_kernel_source() {
        std::string production = segmented_rmsnorm_silu_mul_scalar_reference_kernel_source();
        const std::string lds_setup = R"(    const unsigned long long segment = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum_squares = 0.0f;)";
        const std::string shuffle_setup = R"(    const unsigned long long segment = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % warpSize;
    const unsigned int wave = tid / warpSize;
    // Eight wave32 leaders replace the 256-thread LDS tree.
    __shared__ float wave_partials[8];
    float sum_squares = 0.0f;)";
        const size_t setup_offset = production.find(lds_setup);
        if (setup_offset == std::string::npos) return "";
        production.replace(setup_offset, lds_setup.size(), shuffle_setup);

        const std::string lds_reduction = R"(    partial[tid] = sum_squares;
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
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(segment_size) + epsilon);)";
        const std::string shuffle_reduction = R"(    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum_squares += __shfl_down(sum_squares, offset, warpSize);
    }
    if (lane == 0u) {
        wave_partials[wave] = sum_squares;
    }
    __syncthreads();

    if (wave == 0u) {
        const unsigned int wave_count = (blockDim.x + warpSize - 1u) / warpSize;
        sum_squares = lane < wave_count ? wave_partials[lane] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum_squares += __shfl_down(sum_squares, offset, warpSize);
        }
        if (lane == 0u) {
            wave_partials[0] = sum_squares;
        }
    }
    __syncthreads();
    if (segment >= segments) {
        return;
    }
    const float inv_rms =
        rsqrtf(wave_partials[0] / static_cast<float>(segment_size) + epsilon);)";
        const size_t reduction_offset = production.find(lds_reduction);
        if (reduction_offset == std::string::npos) return "";
        production.replace(reduction_offset, lds_reduction.size(), shuffle_reduction);
        return production;
    }

    // Keep the direct-only prototype ABI available for historical differential/timing comparison.
    // It compiles the same validated body under its original, isolated entry-point name.
    static std::string segmented_rmsnorm_silu_mul_shuffle_prototype_kernel_source() {
        std::string prototype = segmented_rmsnorm_silu_mul_kernel_source();
        const std::string production_name = "ullm_segmented_rmsnorm_silu_mul_f32_kernel";
        const size_t name_offset = prototype.find(production_name);
        if (name_offset == std::string::npos) return "";
        prototype.replace(
            name_offset,
            production_name.size(),
            "ullm_segmented_rmsnorm_silu_mul_shuffle_prototype_f32_kernel");
        return prototype;
    }

    static std::string segmented_rmsnorm_silu_mul_kernel_source_for_arch(const std::string &arch) {
        return arch == "gfx1201" ? segmented_rmsnorm_silu_mul_kernel_source()
                                 : segmented_rmsnorm_silu_mul_scalar_reference_kernel_source();
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

    static const char *gelu_tanh_mul_kernel_source() {
        return R"(
extern "C" __global__ void ullm_gelu_tanh_mul_f32_kernel(
    const float *gate,
    const float *up,
    unsigned long long elements,
    float *output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float value = gate[index];
    // Match Transformers GELUTanh: literal and F32 operation order are part of the contract.
    const float activated = 0.5f * value *
        (1.0f + tanhf(value * 0.7978845608f * (1.0f + 0.044715f * value * value)));
    output[index] = activated * up[index];
}
)";
    }
