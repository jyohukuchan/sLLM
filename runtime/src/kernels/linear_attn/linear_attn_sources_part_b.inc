// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
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

    // Superseded scalar-reduction linear-attention QKV prepare reference. It remains the
    // rollback source and is still used for every architecture or geometry outside the validated
    // gfx1201 Qwen3.5 M=1 shape.
    [[maybe_unused]] static const char *linear_attn_qkv_prepare_scalar_reference_kernel_source() {
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

    // Production linear-attention QKV prepare body. The host ABI and HIPRTC entry-point name
    // remain unchanged. On gfx1201, only the validated Qwen3.5 Q/K CTAs replace their scalar
    // tree with the wave32 reduction; V CTAs retain their reduction-free path exactly as-is.
    static std::string linear_attn_qkv_prepare_kernel_source() {
        std::string production = linear_attn_qkv_prepare_scalar_reference_kernel_source();
        const std::string lds_setup = R"(    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];)";
        const std::string shuffle_setup = R"(    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    // Q/K CTAs use these eight wave32 leaders only for the exact Qwen3.5 geometry below.
    // V CTAs never access either reduction scratch because their path has no reduction.
    __shared__ float wave_partials[8];)";
        const size_t setup_offset = production.find(lds_setup);
        if (setup_offset == std::string::npos) {
            return "";
        }
        production.replace(setup_offset, lds_setup.size(), shuffle_setup);

        const std::string shape_setup = R"(    const unsigned long long channels = q_elements + q_elements + v_elements;
    const unsigned long long block = blockIdx.x;)";
        const std::string shuffle_shape_setup = R"(    const unsigned long long channels = q_elements + q_elements + v_elements;
    const unsigned long long block = blockIdx.x;
    const bool use_qwen35_shuffle =
        key_heads == 16ull &&
        value_heads == 32ull &&
        key_dim == 128ull &&
        value_dim == 128ull &&
        kernel_size == 4ull;)";
        const size_t shape_offset = production.find(shape_setup);
        if (shape_offset == std::string::npos) {
            return "";
        }
        production.replace(shape_offset, shape_setup.size(), shuffle_shape_setup);

        const std::string lds_reduction = R"(        partial[tid] = sum_squares;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            __syncthreads();
        }
        const float norm = sqrtf(partial[0] + 1.0e-6f);)";
        const std::string shuffle_reduction = R"(        float reduced_sum_squares = 0.0f;
        if (use_qwen35_shuffle) {
            reduced_sum_squares = ullm_linear_attn_qkv_prepare_shuffle_reduce_to_all(
                sum_squares,
                wave_partials);
        } else {
            partial[tid] = sum_squares;
            __syncthreads();
            for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    partial[tid] += partial[tid + offset];
                }
                __syncthreads();
            }
            reduced_sum_squares = partial[0];
        }
        const float norm = sqrtf(reduced_sum_squares + 1.0e-6f);)";
        const size_t first_reduction = production.find(lds_reduction);
        if (first_reduction == std::string::npos) {
            return "";
        }
        production.replace(first_reduction, lds_reduction.size(), shuffle_reduction);
        const size_t second_reduction = production.find(lds_reduction);
        if (second_reduction == std::string::npos) {
            return "";
        }
        production.replace(second_reduction, lds_reduction.size(), shuffle_reduction);

        return R"(
__device__ __forceinline__ float ullm_linear_attn_qkv_prepare_shuffle_reduce_to_all(
    float sum,
    float *wave_partials) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % warpSize;
    const unsigned int wave = tid / warpSize;
    for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, warpSize);
    }
    if (lane == 0u) {
        wave_partials[wave] = sum;
    }
    __syncthreads();

    if (wave == 0u) {
        sum = lane < 8u ? wave_partials[lane] : 0.0f;
        for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset, warpSize);
        }
        if (lane == 0u) {
            wave_partials[0] = sum;
        }
    }
    __syncthreads();
    return wave_partials[0];
}
)" + production;
    }

    static std::string linear_attn_qkv_prepare_kernel_source_for_arch(const std::string &arch) {
        return arch == "gfx1201" ? linear_attn_qkv_prepare_kernel_source()
                                 : linear_attn_qkv_prepare_scalar_reference_kernel_source();
    }

    // Direct-only gfx1201 experiment for Qwen3.5's M=1 linear-attention prepare launch. It
    // keeps the scalar served reference intact and gives the candidate a distinct entry point.
    // Only Q/K CTAs use the helper below; V CTAs retain their reduction-free path unchanged.
    static std::string linear_attn_qkv_prepare_shuffle_prototype_kernel_source() {
        std::string prototype = linear_attn_qkv_prepare_scalar_reference_kernel_source();
        const std::string next_kernel =
            "\nextern \"C\" __global__ void ullm_linear_attn_qkv_conv_step_silu_f32_kernel";
        const size_t next_kernel_offset = prototype.find(next_kernel);
        if (next_kernel_offset == std::string::npos) {
            return "";
        }
        prototype.erase(next_kernel_offset);

        const std::string production_name = "ullm_linear_attn_qkv_prepare_f32_kernel";
        const size_t name_offset = prototype.find(production_name);
        if (name_offset == std::string::npos) {
            return "";
        }
        prototype.replace(
            name_offset,
            production_name.size(),
            "ullm_linear_attn_qkv_prepare_shuffle_prototype_f32_kernel");

        const std::string lds_setup = R"(    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];)";
        const std::string shuffle_setup = R"(    const unsigned int tid = threadIdx.x;
    // Eight wave32 leaders are enough for the fixed 256-thread direct launch. V blocks never
    // access this scratch because their path has no Q/K normalization reduction.
    __shared__ float wave_partials[8];)";
        const size_t setup_offset = prototype.find(lds_setup);
        if (setup_offset == std::string::npos) {
            return "";
        }
        prototype.replace(setup_offset, lds_setup.size(), shuffle_setup);

        const std::string lds_reduction = R"(        partial[tid] = sum_squares;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            __syncthreads();
        }
        const float norm = sqrtf(partial[0] + 1.0e-6f);)";
        const std::string shuffle_reduction = R"(        const float norm = sqrtf(
            ullm_linear_attn_qkv_prepare_shuffle_reduce_to_all(
                sum_squares,
                wave_partials) + 1.0e-6f);)";
        size_t first_reduction = prototype.find(lds_reduction);
        if (first_reduction == std::string::npos) {
            return "";
        }
        prototype.replace(first_reduction, lds_reduction.size(), shuffle_reduction);
        const size_t second_reduction = prototype.find(lds_reduction);
        if (second_reduction == std::string::npos) {
            return "";
        }
        prototype.replace(second_reduction, lds_reduction.size(), shuffle_reduction);
        if (prototype.find(lds_reduction) != std::string::npos) {
            return "";
        }

        return R"(
__device__ __forceinline__ float ullm_linear_attn_qkv_prepare_shuffle_reduce_to_all(
    float sum,
    float *wave_partials) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % warpSize;
    const unsigned int wave = tid / warpSize;
    for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, warpSize);
    }
    if (lane == 0u) {
        wave_partials[wave] = sum;
    }
    __syncthreads();

    if (wave == 0u) {
        sum = lane < 8u ? wave_partials[lane] : 0.0f;
        for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset, warpSize);
        }
        if (lane == 0u) {
            wave_partials[0] = sum;
        }
    }
    __syncthreads();
    return wave_partials[0];
}
)" + prototype;
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

    // Generic stable-ABI fallback for recurrent shapes outside the Qwen3.5 shuffle
    // specialization, or if its target-specific code object is unavailable.
    static const char *linear_attn_recurrent_generic_kernel_source() {
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

        const bool state_register_resident =
            key_dim <= static_cast<unsigned long long>(blockDim.x);
        const bool active = static_cast<unsigned long long>(tid) < key_dim;
        unsigned long long state_index = 0ull;
        float state_value = 0.0f;
        if (state_register_resident && active) {
            const unsigned long long key = tid;
            state_index = state_head_offset + key * value_dim + value;
            state_value = state[state_index];
        }

        for (unsigned long long timestep = 0; timestep < sequence_len; ++timestep) {
            const unsigned long long value_head_index = timestep * value_heads + value_head;
            const unsigned long long key_head_index = timestep * key_heads + key_head;
            const unsigned long long qk_base = key_head_index * key_dim;
            const unsigned long long v_base = value_head_index * value_dim;
            const float decay = expf(gate[value_head_index]);
            const float beta_value = beta[value_head_index];

            if (state_register_resident) {
                float current = 0.0f;
                float decayed = 0.0f;
                float key_value = 0.0f;
                float query_value = 0.0f;
                if (active) {
                    const unsigned long long key = tid;
                    key_value = k[qk_base + key];
                    query_value = q[qk_base + key];
                    decayed = state_value * decay;
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
                    state_value = updated;
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
        if (state_register_resident && active) {
            state[state_index] = state_value;
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
