// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
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

