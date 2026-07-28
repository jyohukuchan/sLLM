// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
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

    static const char *causal_attn_flash2_f32_kernel_source() {
        return R"(
extern "C" __global__ void ullm_causal_attn_f32_flash2_kernel(
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
    constexpr unsigned int kTileTokens = 64u;
    const unsigned long long q_head_index = static_cast<unsigned long long>(blockIdx.x);
    const unsigned long long q_head_elements = sequence_len * q_heads;
    if (q_head_index >= q_head_elements) {
        return;
    }
    const unsigned int tid = threadIdx.x;
    if (value_dim > static_cast<unsigned long long>(blockDim.x)) {
        return;
    }

    const unsigned long long q_head = q_head_index % q_heads;
    const unsigned long long timestep = q_head_index / q_heads;
    const unsigned long long cache_len = timestep + 1ull;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_base = (timestep * q_heads + q_head) * head_dim;
    const unsigned long long value = static_cast<unsigned long long>(tid);
    const bool value_active = value < value_dim;

    __shared__ float reduce[256];
    __shared__ float scores[kTileTokens];
    __shared__ float shared_tile_max;
    __shared__ float shared_tile_sum;
    __shared__ float shared_m_new;
    __shared__ float shared_alpha;

    float max_score = -3.4028234663852886e38f;
    float denominator = 0.0f;
    float weighted = 0.0f;

    for (unsigned long long tile_start = 0; tile_start < cache_len; tile_start += kTileTokens) {
        const unsigned long long remaining = cache_len - tile_start;
        const unsigned int tile_count =
            remaining < static_cast<unsigned long long>(kTileTokens) ?
                static_cast<unsigned int>(remaining) :
                kTileTokens;

        for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
            const unsigned long long source_timestep =
                tile_start + static_cast<unsigned long long>(tile_offset);
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
            if (tid == 0u) {
                scores[tile_offset] = reduce[0] * softmax_scale;
            }
            __syncthreads();
        }

        float local_max = -3.4028234663852886e38f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float score = scores[tile_offset];
            local_max = score > local_max ? score : local_max;
        }
        reduce[tid] = local_max;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] = reduce[tid] > reduce[tid + stride] ? reduce[tid] : reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            shared_tile_max = reduce[0];
            const float m_new = max_score > shared_tile_max ? max_score : shared_tile_max;
            shared_m_new = m_new;
            shared_alpha = max_score <= -3.0e38f ? 0.0f : expf(max_score - m_new);
        }
        __syncthreads();

        const float m_new = shared_m_new;
        float local_sum = 0.0f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float weight = expf(scores[tile_offset] - m_new);
            scores[tile_offset] = weight;
            local_sum += weight;
        }
        reduce[tid] = local_sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            shared_tile_sum = reduce[0];
        }
        __syncthreads();

        if (value_active) {
            float tile_weighted = 0.0f;
            for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
                const unsigned long long source_timestep =
                    tile_start + static_cast<unsigned long long>(tile_offset);
                const unsigned long long v_index =
                    (source_timestep * kv_heads + kv_head) * value_dim + value;
                tile_weighted += scores[tile_offset] * v[v_index];
            }
            weighted = weighted * shared_alpha + tile_weighted;
        }
        denominator = denominator * shared_alpha + shared_tile_sum;
        max_score = m_new;
        __syncthreads();
    }

    if (value_active) {
        const unsigned long long output_base = q_head_index * value_dim;
        output[output_base + value] = weighted / denominator;
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

    static const char *causal_attn_batch_flash2_f32_kernel_source() {
        return R"(
extern "C" __global__ void ullm_causal_attn_batch_f32_flash2_kernel(
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
    constexpr unsigned int kTileTokens = 64u;
    const unsigned long long global_q_head_index = static_cast<unsigned long long>(blockIdx.x);
    const unsigned long long q_head_elements_per_batch = sequence_len * q_heads;
    const unsigned long long q_head_elements = batch_count * q_head_elements_per_batch;
    if (global_q_head_index >= q_head_elements) {
        return;
    }
    const unsigned int tid = threadIdx.x;
    if (value_dim > static_cast<unsigned long long>(blockDim.x)) {
        return;
    }

    const unsigned long long batch = global_q_head_index / q_head_elements_per_batch;
    const unsigned long long q_head_index =
        global_q_head_index - batch * q_head_elements_per_batch;
    const unsigned long long q_head = q_head_index % q_heads;
    const unsigned long long timestep = q_head_index / q_heads;
    const unsigned long long cache_len = timestep + 1ull;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_batch_base = batch * q_head_elements_per_batch * head_dim;
    const unsigned long long k_batch_base = batch * sequence_len * kv_heads * head_dim;
    const unsigned long long v_batch_base = batch * sequence_len * kv_heads * value_dim;
    const unsigned long long output_batch_base = batch * q_head_elements_per_batch * value_dim;
    const unsigned long long q_base = q_batch_base + (timestep * q_heads + q_head) * head_dim;
    const unsigned long long value = static_cast<unsigned long long>(tid);
    const bool value_active = value < value_dim;

    __shared__ float reduce[256];
    __shared__ float scores[kTileTokens];
    __shared__ float shared_tile_max;
    __shared__ float shared_tile_sum;
    __shared__ float shared_m_new;
    __shared__ float shared_alpha;

    float max_score = -3.4028234663852886e38f;
    float denominator = 0.0f;
    float weighted = 0.0f;

    for (unsigned long long tile_start = 0; tile_start < cache_len; tile_start += kTileTokens) {
        const unsigned long long remaining = cache_len - tile_start;
        const unsigned int tile_count =
            remaining < static_cast<unsigned long long>(kTileTokens) ?
                static_cast<unsigned int>(remaining) :
                kTileTokens;

        for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
            const unsigned long long source_timestep =
                tile_start + static_cast<unsigned long long>(tile_offset);
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
            if (tid == 0u) {
                scores[tile_offset] = reduce[0] * softmax_scale;
            }
            __syncthreads();
        }

        float local_max = -3.4028234663852886e38f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float score = scores[tile_offset];
            local_max = score > local_max ? score : local_max;
        }
        reduce[tid] = local_max;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] = reduce[tid] > reduce[tid + stride] ? reduce[tid] : reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            shared_tile_max = reduce[0];
            const float m_new = max_score > shared_tile_max ? max_score : shared_tile_max;
            shared_m_new = m_new;
            shared_alpha = max_score <= -3.0e38f ? 0.0f : expf(max_score - m_new);
        }
        __syncthreads();

        const float m_new = shared_m_new;
        float local_sum = 0.0f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float weight = expf(scores[tile_offset] - m_new);
            scores[tile_offset] = weight;
            local_sum += weight;
        }
        reduce[tid] = local_sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            shared_tile_sum = reduce[0];
        }
        __syncthreads();

        if (value_active) {
            float tile_weighted = 0.0f;
            for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
                const unsigned long long source_timestep =
                    tile_start + static_cast<unsigned long long>(tile_offset);
                const unsigned long long v_index =
                    v_batch_base + (source_timestep * kv_heads + kv_head) * value_dim + value;
                tile_weighted += scores[tile_offset] * v[v_index];
            }
            weighted = weighted * shared_alpha + tile_weighted;
        }
        denominator = denominator * shared_alpha + shared_tile_sum;
        max_score = m_new;
        __syncthreads();
    }

    if (value_active) {
        const unsigned long long output_base = output_batch_base + q_head_index * value_dim;
        output[output_base + value] = weighted / denominator;
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
#if defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_cvt_f32_fp8(static_cast<unsigned int>(value), 0);
#else
    const unsigned int sign = static_cast<unsigned int>(value) >> 7;
    const unsigned int exponent = (static_cast<unsigned int>(value) >> 3) & 0x0fu;
    const unsigned int mantissa = static_cast<unsigned int>(value) & 0x07u;
    if (exponent == 0u) {
        const float magnitude = static_cast<float>(mantissa) * 0.001953125f;
        return sign == 0u ? magnitude : -magnitude;
    }
    const unsigned int fp32_bits = (sign << 31) | ((exponent + 120u) << 23) | (mantissa << 20);
    return __uint_as_float(fp32_bits);
#endif
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

    static const char *cached_prefix_flash2_f32_kernel_source() {
        return R"(
extern "C" __global__ void ullm_cached_prefix_attn_f32_flash2_kernel(
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
    constexpr unsigned int kTileTokens = 64u;
    const unsigned long long q_head_index = static_cast<unsigned long long>(blockIdx.x);
    const unsigned long long q_head_elements = new_tokens * q_heads;
    if (q_head_index >= q_head_elements) {
        return;
    }
    const unsigned int tid = threadIdx.x;
    if (value_dim > static_cast<unsigned long long>(blockDim.x)) {
        return;
    }

    const unsigned long long q_head = q_head_index % q_heads;
    const unsigned long long token_index = q_head_index / q_heads;
    const unsigned long long cache_len = cached_prefix_len + token_index + 1;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_base = (token_index * q_heads + q_head) * head_dim;
    const unsigned long long value = static_cast<unsigned long long>(tid);
    const bool value_active = value < value_dim;

    __shared__ float reduce[256];
    __shared__ float scores[kTileTokens];
    __shared__ float shared_tile_max;
    __shared__ float shared_tile_sum;
    __shared__ float shared_m_new;
    __shared__ float shared_alpha;

    float max_score = -3.4028234663852886e38f;
    float denominator = 0.0f;
    float weighted = 0.0f;

    for (unsigned long long tile_start = 0; tile_start < cache_len; tile_start += kTileTokens) {
        const unsigned long long remaining = cache_len - tile_start;
        const unsigned int tile_count =
            remaining < static_cast<unsigned long long>(kTileTokens) ?
                static_cast<unsigned int>(remaining) :
                kTileTokens;

        for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
            const unsigned long long source_timestep =
                tile_start + static_cast<unsigned long long>(tile_offset);
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
            if (tid == 0u) {
                scores[tile_offset] = reduce[0] * softmax_scale;
            }
            __syncthreads();
        }

        float local_max = -3.4028234663852886e38f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float score = scores[tile_offset];
            local_max = score > local_max ? score : local_max;
        }
        reduce[tid] = local_max;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] = reduce[tid] > reduce[tid + stride] ? reduce[tid] : reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            shared_tile_max = reduce[0];
            const float m_new = max_score > shared_tile_max ? max_score : shared_tile_max;
            shared_m_new = m_new;
            shared_alpha = max_score <= -3.0e38f ? 0.0f : expf(max_score - m_new);
        }
        __syncthreads();

        const float m_new = shared_m_new;
        float local_sum = 0.0f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float weight = expf(scores[tile_offset] - m_new);
            scores[tile_offset] = weight;
            local_sum += weight;
        }
        reduce[tid] = local_sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            shared_tile_sum = reduce[0];
        }
        __syncthreads();

        if (value_active) {
            float tile_weighted = 0.0f;
            for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
                const unsigned long long source_timestep =
                    tile_start + static_cast<unsigned long long>(tile_offset);
                const unsigned long long v_index =
                    (source_timestep * kv_heads + kv_head) * value_dim + value;
                tile_weighted += scores[tile_offset] * v_cache[v_index];
            }
            weighted = weighted * shared_alpha + tile_weighted;
        }
        denominator = denominator * shared_alpha + shared_tile_sum;
        max_score = m_new;
        __syncthreads();
    }

    if (value_active) {
        const unsigned long long output_base = q_head_index * value_dim;
        output[output_base + value] = weighted / denominator;
    }
}
)";
    }

    // This variant keeps the generic Flash2 arithmetic schedule for every
    // individual Q head.  One CTA owns a (token, KV-head) group and stages a
    // 20-token K or V segment only once, then visits the five Q heads in a
    // deterministic sequence.  The five heads therefore reuse the same K/V
    // staging, while score, max, sum, and weighted-value reductions retain the
    // generic 256-thread tree and 64-token online-softmax boundary.
    static const char *cached_prefix_flash2_f32_gqa_grouped_serial_kernel_source() {
        return R"(
extern "C" __global__ void ullm_cached_prefix_attn_f32_flash2_gqa_grouped_kernel(
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
    constexpr unsigned int kStageTokens = 20u;
    constexpr unsigned int kFlashTileTokens = 64u;
    constexpr unsigned int kQHeadsPerKv = 5u;
    constexpr unsigned int kHeadDim = 128u;

    if (blockDim.x != 256u || kv_heads == 0u || head_dim != kHeadDim ||
        value_dim != kHeadDim || q_heads != kv_heads * kQHeadsPerKv) {
        return;
    }
    const unsigned long long grouped_index = static_cast<unsigned long long>(blockIdx.x);
    const unsigned long long grouped_count = new_tokens * kv_heads;
    if (grouped_index >= grouped_count) {
        return;
    }

    const unsigned long long token_index = grouped_index / kv_heads;
    const unsigned long long kv_head = grouped_index % kv_heads;
    const unsigned long long cache_len = cached_prefix_len + token_index + 1u;
    const unsigned int tid = threadIdx.x;
    const unsigned long long value = static_cast<unsigned long long>(tid);
    const bool value_active = value < value_dim;

    __shared__ float staged_kv[kStageTokens * kHeadDim];
    __shared__ float reduce[256];
    __shared__ float scores[kQHeadsPerKv * kFlashTileTokens];
    __shared__ float shared_tile_max[kQHeadsPerKv];
    __shared__ float shared_tile_sum[kQHeadsPerKv];
    __shared__ float shared_m_new[kQHeadsPerKv];
    __shared__ float shared_alpha[kQHeadsPerKv];

    float max_score[kQHeadsPerKv] = {
        -3.4028234663852886e38f,
        -3.4028234663852886e38f,
        -3.4028234663852886e38f,
        -3.4028234663852886e38f,
        -3.4028234663852886e38f,
    };
    float denominator[kQHeadsPerKv] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float weighted[kQHeadsPerKv] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (unsigned long long tile_start = 0u; tile_start < cache_len; tile_start += kFlashTileTokens) {
        const unsigned long long remaining = cache_len - tile_start;
        const unsigned int tile_count =
            remaining < static_cast<unsigned long long>(kFlashTileTokens) ?
                static_cast<unsigned int>(remaining) : kFlashTileTokens;

        // Stage K once per KV head.  For each Q head, the score calculation
        // below is deliberately the generic Flash2 body with only the source
        // load redirected to the exact staged float.
        for (unsigned int stage_start = 0u; stage_start < tile_count; stage_start += kStageTokens) {
            const unsigned int stage_count =
                tile_count - stage_start < kStageTokens ? tile_count - stage_start : kStageTokens;
            const unsigned int stage_elements = stage_count * kHeadDim;
            for (unsigned int index = tid; index < stage_elements; index += blockDim.x) {
                const unsigned int token_offset = index / kHeadDim;
                const unsigned int dim = index % kHeadDim;
                const unsigned long long source_timestep =
                    tile_start + static_cast<unsigned long long>(stage_start + token_offset);
                staged_kv[index] =
                    k_cache[(source_timestep * kv_heads + kv_head) * head_dim + dim];
            }
            __syncthreads();

            for (unsigned int q_slot = 0u; q_slot < kQHeadsPerKv; ++q_slot) {
                const unsigned long long q_head =
                    kv_head * kQHeadsPerKv + static_cast<unsigned long long>(q_slot);
                const unsigned long long q_base = (token_index * q_heads + q_head) * head_dim;
                const unsigned int score_base = q_slot * kFlashTileTokens;
                for (unsigned int token_offset = 0u; token_offset < stage_count; ++token_offset) {
                    float partial = 0.0f;
                    for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
                        partial += q[q_base + dim] * staged_kv[token_offset * kHeadDim + dim];
                    }
                    reduce[tid] = partial;
                    __syncthreads();
                    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                        if (tid < stride) {
                            reduce[tid] += reduce[tid + stride];
                        }
                        __syncthreads();
                    }
                    if (tid == 0u) {
                        scores[score_base + stage_start + token_offset] = reduce[0] * softmax_scale;
                    }
                    __syncthreads();
                }
            }
            __syncthreads();
        }

        // Preserve generic Flash2's max and sum trees separately for every
        // Q head.  Only the inter-head schedule changed; a head's own score
        // order and online state update are unchanged.
        for (unsigned int q_slot = 0u; q_slot < kQHeadsPerKv; ++q_slot) {
            const unsigned int score_base = q_slot * kFlashTileTokens;
            float local_max = -3.4028234663852886e38f;
            for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
                const float score = scores[score_base + tile_offset];
                local_max = score > local_max ? score : local_max;
            }
            reduce[tid] = local_max;
            __syncthreads();
            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] = reduce[tid] > reduce[tid + stride] ? reduce[tid] : reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0u) {
                shared_tile_max[q_slot] = reduce[0];
                const float m_new = max_score[q_slot] > shared_tile_max[q_slot] ?
                    max_score[q_slot] : shared_tile_max[q_slot];
                shared_m_new[q_slot] = m_new;
                shared_alpha[q_slot] = max_score[q_slot] <= -3.0e38f ?
                    0.0f : expf(max_score[q_slot] - m_new);
            }
            __syncthreads();

            const float m_new = shared_m_new[q_slot];
            float local_sum = 0.0f;
            for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
                const float weight = expf(scores[score_base + tile_offset] - m_new);
                scores[score_base + tile_offset] = weight;
                local_sum += weight;
            }
            reduce[tid] = local_sum;
            __syncthreads();
            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0u) {
                shared_tile_sum[q_slot] = reduce[0];
            }
            __syncthreads();
        }

        float tile_weighted[kQHeadsPerKv] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        // Reuse the K staging allocation for V.  The V loop has the generic
        // token order for each Q head even though q slots are interleaved.
        for (unsigned int stage_start = 0u; stage_start < tile_count; stage_start += kStageTokens) {
            const unsigned int stage_count =
                tile_count - stage_start < kStageTokens ? tile_count - stage_start : kStageTokens;
            const unsigned int stage_elements = stage_count * kHeadDim;
            for (unsigned int index = tid; index < stage_elements; index += blockDim.x) {
                const unsigned int token_offset = index / kHeadDim;
                const unsigned int dim = index % kHeadDim;
                const unsigned long long source_timestep =
                    tile_start + static_cast<unsigned long long>(stage_start + token_offset);
                staged_kv[index] =
                    v_cache[(source_timestep * kv_heads + kv_head) * value_dim + dim];
            }
            __syncthreads();
            if (value_active) {
                for (unsigned int q_slot = 0u; q_slot < kQHeadsPerKv; ++q_slot) {
                    const unsigned int score_base = q_slot * kFlashTileTokens + stage_start;
                    float current = tile_weighted[q_slot];
                    for (unsigned int token_offset = 0u; token_offset < stage_count; ++token_offset) {
                        current += scores[score_base + token_offset] *
                            staged_kv[token_offset * kHeadDim + value];
                    }
                    tile_weighted[q_slot] = current;
                }
            }
            __syncthreads();
        }

        for (unsigned int q_slot = 0u; q_slot < kQHeadsPerKv; ++q_slot) {
            if (value_active) {
                weighted[q_slot] = weighted[q_slot] * shared_alpha[q_slot] + tile_weighted[q_slot];
            }
            denominator[q_slot] = denominator[q_slot] * shared_alpha[q_slot] + shared_tile_sum[q_slot];
            max_score[q_slot] = shared_m_new[q_slot];
        }
        __syncthreads();
    }

    if (value_active) {
        for (unsigned int q_slot = 0u; q_slot < kQHeadsPerKv; ++q_slot) {
            const unsigned long long q_head =
                kv_head * kQHeadsPerKv + static_cast<unsigned long long>(q_slot);
            const unsigned long long output_base =
                (token_index * q_heads + q_head) * value_dim + value;
            output[output_base] = weighted[q_slot] / denominator[q_slot];
        }
    }
}
)";
    }

    // This isolated R9700-only source is intentionally a separate symbol.  It
    // keeps the public Flash2 launch contract and F32 KV layout unchanged while
    // replacing the three 256-lane LDS trees with wave32 reductions plus the
    // necessary LDS handoff.  The normal source above remains the
    // default until its numerical/resource/timing gates have passed.
    static std::string cached_prefix_flash2_f32_staged_wave32_prototype_kernel_source() {
        std::string source = R"(
__device__ float ullm_sq8_0_flash2_reduce_sum_wave32(float value, float *partial) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int wave = tid >> 5u;
    const unsigned int wave_count = (blockDim.x + 31u) >> 5u;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, 32);
    }
    if (lane == 0u) {
        partial[wave] = value;
    }
    __syncthreads();
    if (wave == 0u) {
        value = lane < wave_count ? partial[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down(value, offset, 32);
        }
        if (lane == 0u) {
            partial[0] = value;
        }
    }
    __syncthreads();
    return partial[0];
}

__device__ float ullm_sq8_0_flash2_reduce_max_wave32(float value, float *partial) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int wave = tid >> 5u;
    const unsigned int wave_count = (blockDim.x + 31u) >> 5u;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down(value, offset, 32);
        value = value > other ? value : other;
    }
    if (lane == 0u) {
        partial[wave] = value;
    }
    __syncthreads();
    if (wave == 0u) {
        value = lane < wave_count ? partial[lane] : -3.4028234663852886e38f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_down(value, offset, 32);
            value = value > other ? value : other;
        }
        if (lane == 0u) {
            partial[0] = value;
        }
    }
    __syncthreads();
    return partial[0];
}
)";
        source += cached_prefix_flash2_f32_kernel_source();
        replace_all_source_text(
            &source,
            "ullm_cached_prefix_attn_f32_flash2_kernel",
            "ullm_sq8_0_cached_prefix_attn_f32_flash2_staged_wave32_prototype_kernel");
        replace_all_source_text(
            &source,
            "            reduce[tid] = partial;\n"
            "            __syncthreads();\n"
            "            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {\n"
            "                if (tid < stride) {\n"
            "                    reduce[tid] += reduce[tid + stride];\n"
            "                }\n"
            "                __syncthreads();\n"
            "            }\n"
            "            if (tid == 0u) {\n"
            "                scores[tile_offset] = reduce[0] * softmax_scale;\n"
            "            }\n"
            "            __syncthreads();",
            "            const float score =\n"
            "                ullm_sq8_0_flash2_reduce_sum_wave32(partial, reduce) * softmax_scale;\n"
            "            if (tid == 0u) {\n"
            "                scores[tile_offset] = score;\n"
            "            }\n"
            "            __syncthreads();");
        replace_all_source_text(
            &source,
            "        reduce[tid] = local_max;\n"
            "        __syncthreads();\n"
            "        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {\n"
            "            if (tid < stride) {\n"
            "                reduce[tid] = reduce[tid] > reduce[tid + stride] ? reduce[tid] : reduce[tid + stride];\n"
            "            }\n"
            "            __syncthreads();\n"
            "        }",
            "        const float tile_max = ullm_sq8_0_flash2_reduce_max_wave32(local_max, reduce);");
        replace_all_source_text(
            &source,
            "            shared_tile_max = reduce[0];",
            "            shared_tile_max = tile_max;");
        replace_all_source_text(
            &source,
            "        reduce[tid] = local_sum;\n"
            "        __syncthreads();\n"
            "        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {\n"
            "            if (tid < stride) {\n"
            "                reduce[tid] += reduce[tid + stride];\n"
            "            }\n"
            "            __syncthreads();\n"
            "        }",
            "        const float tile_sum = ullm_sq8_0_flash2_reduce_sum_wave32(local_sum, reduce);");
        replace_all_source_text(
            &source,
            "            shared_tile_sum = reduce[0];",
            "            shared_tile_sum = tile_sum;");
        return source;
    }

    static const char *cached_prefix_flash2_fp8_e4m3_kernel_source() {
        return R"(
__device__ float ullm_flash2_fp8_e4m3_to_f32(unsigned char value) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_cvt_f32_fp8(static_cast<unsigned int>(value), 0);
#else
    const unsigned int sign = static_cast<unsigned int>(value) >> 7;
    const unsigned int exponent = (static_cast<unsigned int>(value) >> 3) & 0x0fu;
    const unsigned int mantissa = static_cast<unsigned int>(value) & 0x07u;
    if (exponent == 0u) {
        const float magnitude = static_cast<float>(mantissa) * 0.001953125f;
        return sign == 0u ? magnitude : -magnitude;
    }
    const unsigned int fp32_bits = (sign << 31) | ((exponent + 120u) << 23) | (mantissa << 20);
    return __uint_as_float(fp32_bits);
#endif
}

extern "C" __global__ void ullm_cached_prefix_attn_fp8_e4m3_flash2_kernel(
    const void *q_raw,
    const unsigned char *k_cache,
    const unsigned char *v_cache,
    unsigned long long cached_prefix_len,
    unsigned long long new_tokens,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float q_scale,
    float k_scale,
    float v_scale,
    unsigned int q_is_fp8,
    float *output) {
    constexpr unsigned int kTileTokens = 64u;
    const unsigned long long q_head_index = static_cast<unsigned long long>(blockIdx.x);
    const unsigned long long q_head_elements = new_tokens * q_heads;
    if (q_head_index >= q_head_elements) {
        return;
    }
    const unsigned int tid = threadIdx.x;
    if (value_dim > static_cast<unsigned long long>(blockDim.x)) {
        return;
    }

    const unsigned long long q_head = q_head_index % q_heads;
    const unsigned long long token_index = q_head_index / q_heads;
    const unsigned long long cache_len = cached_prefix_len + token_index + 1;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_base = (token_index * q_heads + q_head) * head_dim;
    const float *q_f32 = reinterpret_cast<const float *>(q_raw);
    const unsigned char *q_fp8 = reinterpret_cast<const unsigned char *>(q_raw);
    const unsigned long long value = static_cast<unsigned long long>(tid);
    const bool value_active = value < value_dim;

    __shared__ float reduce[256];
    __shared__ float scores[kTileTokens];
    __shared__ float shared_tile_max;
    __shared__ float shared_tile_sum;
    __shared__ float shared_m_new;
    __shared__ float shared_alpha;

    float max_score = -3.4028234663852886e38f;
    float denominator = 0.0f;
    float weighted = 0.0f;

    for (unsigned long long tile_start = 0; tile_start < cache_len; tile_start += kTileTokens) {
        const unsigned long long remaining = cache_len - tile_start;
        const unsigned int tile_count =
            remaining < static_cast<unsigned long long>(kTileTokens) ?
                static_cast<unsigned int>(remaining) :
                kTileTokens;

        for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
            const unsigned long long source_timestep =
                tile_start + static_cast<unsigned long long>(tile_offset);
            const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
                const float q_value = q_is_fp8 == 0u ?
                    q_f32[q_base + dim] :
                    (ullm_flash2_fp8_e4m3_to_f32(q_fp8[q_base + dim]) * q_scale);
                partial += q_value *
                    (ullm_flash2_fp8_e4m3_to_f32(k_cache[k_base + dim]) * k_scale);
            }
            reduce[tid] = partial;
            __syncthreads();
            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0u) {
                scores[tile_offset] = reduce[0] * softmax_scale;
            }
            __syncthreads();
        }

        float local_max = -3.4028234663852886e38f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float score = scores[tile_offset];
            local_max = score > local_max ? score : local_max;
        }
        reduce[tid] = local_max;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] = reduce[tid] > reduce[tid + stride] ? reduce[tid] : reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            shared_tile_max = reduce[0];
            const float m_new = max_score > shared_tile_max ? max_score : shared_tile_max;
            shared_m_new = m_new;
            shared_alpha = max_score <= -3.0e38f ? 0.0f : expf(max_score - m_new);
        }
        __syncthreads();

        const float m_new = shared_m_new;
        float local_sum = 0.0f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float weight = expf(scores[tile_offset] - m_new);
            scores[tile_offset] = weight;
            local_sum += weight;
        }
        reduce[tid] = local_sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            shared_tile_sum = reduce[0];
        }
        __syncthreads();

        if (value_active) {
            float tile_weighted = 0.0f;
            for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
                const unsigned long long source_timestep =
                    tile_start + static_cast<unsigned long long>(tile_offset);
                const unsigned long long v_index =
                    (source_timestep * kv_heads + kv_head) * value_dim + value;
                tile_weighted +=
                    scores[tile_offset] *
                    (ullm_flash2_fp8_e4m3_to_f32(v_cache[v_index]) * v_scale);
            }
            weighted = weighted * shared_alpha + tile_weighted;
        }
        denominator = denominator * shared_alpha + shared_tile_sum;
        max_score = m_new;
        __syncthreads();
    }

    if (value_active) {
        const unsigned long long output_base = q_head_index * value_dim;
        output[output_base + value] = weighted / denominator;
    }
}
)";
    }

    static std::string paged_decode_attn_kernel_source_for_arch(const std::string &arch) {
        (void)arch;
        std::string preamble;
        const auto experimental_flag_is_one = [](const char *name) {
            const char *value = std::getenv(name);
            return value != nullptr && value[0] == '1' && value[1] == '\0';
        };
        if (std::getenv("ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE") != nullptr) {
            preamble += "#define ULLM_PAGED_DECODE_USE_SHARED_REDUCE 1\n";
        }
        if (std::getenv("ULLM_DISABLE_PAGED_DECODE_ONLINE_SOFTMAX") != nullptr) {
            preamble += "#define ULLM_PAGED_DECODE_USE_TWO_PASS_SOFTMAX 1\n";
        }
        if (std::getenv("ULLM_EXPERIMENTAL_PAGED_DECODE_WAVE_SCALAR_SOFTMAX") != nullptr) {
            preamble += "#define ULLM_PAGED_DECODE_USE_WAVE_SCALAR_SOFTMAX 1\n";
        }
        // These two selectors are intentionally test-only.  The grouped body is
        // shape-guarded in the device source and the host launcher, so an
        // unsupported geometry continues to use the canonical split body.
        const bool gqa_grouped =
            experimental_flag_is_one("ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT");
        const bool gqa_pipelined =
            experimental_flag_is_one("ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_PIPELINED_SPLIT");
        if (gqa_grouped || gqa_pipelined) {
            preamble += "#define ULLM_PAGED_DECODE_USE_GQA_GROUPED_SPLIT 1\n";
        }
        if (gqa_pipelined) {
            preamble += "#define ULLM_PAGED_DECODE_USE_GQA_PIPELINED_SPLIT 1\n";
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
    // Gemma4 full-attention shape: 8Q/1KV/512/512.  The generic fallback
    // below gives each output element a thread, which makes every one of the
    // 512 output threads recompute the same 512-term Q dot K.  Keep a
    // 256-thread CTA per Q head instead: each thread owns a pair of Q/K and V
    // dimensions.  This deliberately does not alter the canonical <=256
    // branch below (including Qwen3.5's production 256-wide path).
    if (gridDim.x == q_heads && blockDim.x == 256u && head_dim == 512ull &&
        value_dim == 512ull) {
        const unsigned long long q_head = blockIdx.x;
        const unsigned int tid = threadIdx.x;
        const unsigned long long q_per_kv = q_heads / kv_heads;
        const unsigned long long kv_head = q_head / q_per_kv;
        const unsigned long long q_base = q_head * 512ull;
        const unsigned long long paired_dim = static_cast<unsigned long long>(tid) + 256ull;

        float max_score = -3.4028234663852886e38f;
        float denominator = 0.0f;
        float weighted_lo = 0.0f;
        float weighted_hi = 0.0f;
        unsigned long long block_index = 0ull;
        unsigned long long block_end = block_size;
        unsigned long long block_id = static_cast<unsigned long long>(block_table[0]);
        if (block_id >= cache_blocks) {
            output[q_head * 512ull + tid] = 0.0f;
            output[q_head * 512ull + paired_dim] = 0.0f;
            return;
        }
        unsigned long long physical_timestep = block_id * block_size;
        for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
            if (source_timestep == block_end) {
                ++block_index;
                block_end += block_size;
                block_id = static_cast<unsigned long long>(block_table[block_index]);
                if (block_id >= cache_blocks) {
                    output[q_head * 512ull + tid] = 0.0f;
                    output[q_head * 512ull + paired_dim] = 0.0f;
                    return;
                }
                physical_timestep = block_id * block_size;
            }
            const unsigned long long k_base =
                (physical_timestep * kv_heads + kv_head) * 512ull;
            const float local =
                q[q_base + tid] * k_cache[k_base + tid] +
                q[q_base + paired_dim] * k_cache[k_base + paired_dim];
            const float score =
                ullm_paged_decode_reduce_sum_256(local, partial) * softmax_scale;
            const unsigned long long v_base =
                (physical_timestep * kv_heads + kv_head) * 512ull;
            const float value_lo = v_cache[v_base + tid];
            const float value_hi = v_cache[v_base + paired_dim];
            if (score > max_score) {
                const float scale = expf(max_score - score);
                weighted_lo = weighted_lo * scale + value_lo;
                weighted_hi = weighted_hi * scale + value_hi;
                denominator = denominator * scale + 1.0f;
                max_score = score;
            } else {
                const float weight = expf(score - max_score);
                weighted_lo += weight * value_lo;
                weighted_hi += weight * value_hi;
                denominator += weight;
            }
            ++physical_timestep;
        }
        const unsigned long long output_base = q_head * 512ull;
        const float decoded_lo = weighted_lo / denominator;
        const float decoded_hi = weighted_hi / denominator;
        if (gate != nullptr) {
            const float gate_lo = gate[output_base + tid];
            const float gate_hi = gate[output_base + paired_dim];
            output[output_base + tid] =
                (1.0f / (1.0f + expf(-gate_lo))) * decoded_lo;
            output[output_base + paired_dim] =
                (1.0f / (1.0f + expf(-gate_hi))) * decoded_hi;
        } else {
            output[output_base + tid] = decoded_lo;
            output[output_base + paired_dim] = decoded_hi;
        }
        return;
    }
    if (gridDim.x == q_heads && blockDim.x == 256u && head_dim <= 256ull &&
        value_dim <= 256ull) {
        const unsigned long long q_head = blockIdx.x;
        const unsigned int tid = threadIdx.x;
#if defined(ULLM_PAGED_DECODE_USE_WAVE_SCALAR_SOFTMAX) && !defined(ULLM_PAGED_DECODE_USE_TWO_PASS_SOFTMAX)
        const unsigned int lane = tid % warpSize;
        const unsigned int wave = tid / warpSize;
#endif
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
#if defined(ULLM_PAGED_DECODE_USE_WAVE_SCALAR_SOFTMAX) && !defined(ULLM_PAGED_DECODE_USE_TWO_PASS_SOFTMAX)
            const bool value_active = tid < value_dim;
            float v = 0.0f;
            if (value_active) {
                const unsigned long long v_index =
                    (physical_timestep * kv_heads + kv_head) * value_dim + tid;
                v = v_cache[v_index];
            }
            const bool wave_has_value =
                static_cast<unsigned long long>(wave) * static_cast<unsigned long long>(warpSize) < value_dim;
            if (wave_has_value) {
                float wave_scale = 0.0f;
                float wave_replaces_max = 0.0f;
                if (lane == 0u) {
                    if (score > max_score) {
                        wave_scale = expf(max_score - score);
                        denominator = denominator * wave_scale + 1.0f;
                        max_score = score;
                        wave_replaces_max = 1.0f;
                    } else {
                        wave_scale = expf(score - max_score);
                        denominator += wave_scale;
                    }
                }
                wave_scale = __shfl(wave_scale, 0, warpSize);
                wave_replaces_max = __shfl(wave_replaces_max, 0, warpSize);
                if (value_active) {
                    if (wave_replaces_max != 0.0f) {
                        weighted = weighted * wave_scale + v;
                    } else {
                        weighted += wave_scale * v;
                    }
                }
            }
#else
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
#endif
            ++physical_timestep;
        }
#endif
#if defined(ULLM_PAGED_DECODE_USE_WAVE_SCALAR_SOFTMAX) && !defined(ULLM_PAGED_DECODE_USE_TWO_PASS_SOFTMAX)
        const float final_denominator = __shfl(denominator, 0, warpSize);
        if (tid < value_dim) {
            const unsigned long long output_index = q_head * value_dim + tid;
            const float decoded = weighted / final_denominator;
            if (gate != nullptr) {
                const float gate_value = gate[output_index];
                const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
                output[output_index] = sigmoid * decoded;
            } else {
                output[output_index] = decoded;
            }
        }
#else
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
#endif
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

extern "C" __global__ void ullm_paged_decode_attn_split_partial_f32_kernel(
    const float *q,
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
    unsigned long long source_tile,
    unsigned long long split_count,
    float *workspace) {
    __shared__ float partial[256];
#if defined(ULLM_PAGED_DECODE_USE_GQA_GROUPED_SPLIT)
    __shared__ float gqa_stage[512];
    __shared__ int gqa_stage_valid[2];
    // R9700 SQ8_0 decode shape: five query heads share one KV head.  One
    // workgroup owns those five heads for one source tile.  The first five
    // waves hold one Q head each (four dimensions/value elements per lane);
    // the remaining three waves stage a single K+V row for all five.  Thus
    // K/V have one semantic global load per KV head rather than five.  This
    // says nothing about physical HBM traffic or cache hit rate.
    const unsigned long long gqa_q_per_kv =
        kv_heads == 0ull ? 0ull : q_heads / kv_heads;
    if (blockDim.x == 256u && warpSize == 32u && gqa_q_per_kv == 5ull &&
        head_dim == 128ull && value_dim == 128ull &&
        static_cast<unsigned long long>(gridDim.x) == kv_heads * split_count) {
        const unsigned int tid = threadIdx.x;
        const unsigned int lane = tid % 32u;
        const bool consumer = tid < 160u;
        const unsigned long long grouped_index = blockIdx.x;
        const unsigned long long kv_head = grouped_index / split_count;
        const unsigned long long split_id = grouped_index - kv_head * split_count;
        if (kv_head >= kv_heads) {
            return;
        }
        const unsigned long long source_start = split_id * source_tile;
        const unsigned long long source_remaining = cache_len - source_start;
        const unsigned long long source_count =
            source_tile < source_remaining ? source_tile : source_remaining;
        const unsigned long long stride = value_dim + 2ull;
        const unsigned long long q_head =
            kv_head * gqa_q_per_kv + static_cast<unsigned long long>(tid / 32u);

        float q0 = 0.0f;
        float q1 = 0.0f;
        float q2 = 0.0f;
        float q3 = 0.0f;
        float weighted0 = 0.0f;
        float weighted1 = 0.0f;
        float weighted2 = 0.0f;
        float weighted3 = 0.0f;
        float max_score = -3.4028234663852886e38f;
        float denominator = 0.0f;
        bool invalid = false;
        if (consumer) {
            const unsigned long long q_base = q_head * 128ull;
            q0 = q[q_base + lane];
            q1 = q[q_base + 32ull + lane];
            q2 = q[q_base + 64ull + lane];
            q3 = q[q_base + 96ull + lane];
        }

        // Two K/V rows are needed only by the pipelined body.  Keeping the
        // same allocation in the non-pipelined body makes the resource delta
        // explicit and keeps the two experiments ABI-identical.
#if defined(ULLM_PAGED_DECODE_USE_GQA_PIPELINED_SPLIT)
        // Prime stage zero with all eight waves.  Subsequent rows are loaded
        // by waves 5--7 while waves 0--4 consume the other buffer; the sole
        // bottom barrier both protects the old buffer and publishes the next.
        const unsigned long long initial_block_index = source_start / block_size;
        const unsigned long long initial_block_offset =
            source_start - initial_block_index * block_size;
        const unsigned long long initial_block_id =
            static_cast<unsigned long long>(block_table[initial_block_index]);
        const bool initial_valid = initial_block_id < cache_blocks;
        if (tid == 0u) {
            gqa_stage_valid[0] = initial_valid ? 1 : 0;
        }
        if (initial_valid) {
            const unsigned long long physical_timestep =
                initial_block_id * block_size + initial_block_offset;
            if (tid < 128u) {
                gqa_stage[tid] =
                    k_cache[(physical_timestep * kv_heads + kv_head) * 128ull + tid];
            } else {
                gqa_stage[tid] =
                    v_cache[(physical_timestep * kv_heads + kv_head) * 128ull + (tid - 128u)];
            }
        } else {
            gqa_stage[tid] = 0.0f;
        }
        __syncthreads();

        for (unsigned long long source_offset = 0ull;
             source_offset < source_count;
             ++source_offset) {
            const unsigned int current_stage = static_cast<unsigned int>(source_offset & 1ull);
            const unsigned int next_stage = current_stage ^ 1u;
            if (source_offset + 1ull < source_count && tid >= 160u) {
                const unsigned long long next_source = source_start + source_offset + 1ull;
                const unsigned long long next_block_index = next_source / block_size;
                const unsigned long long next_block_offset =
                    next_source - next_block_index * block_size;
                const unsigned long long next_block_id =
                    static_cast<unsigned long long>(block_table[next_block_index]);
                const bool next_valid = next_block_id < cache_blocks;
                if (tid == 160u) {
                    gqa_stage_valid[next_stage] = next_valid ? 1 : 0;
                }
                const unsigned long long next_physical_timestep =
                    next_block_id * block_size + next_block_offset;
                for (unsigned int copy_index = tid - 160u;
                     copy_index < 256u;
                     copy_index += 96u) {
                    const unsigned int destination = next_stage * 256u + copy_index;
                    if (!next_valid) {
                        gqa_stage[destination] = 0.0f;
                    } else if (copy_index < 128u) {
                        gqa_stage[destination] = k_cache[
                            (next_physical_timestep * kv_heads + kv_head) * 128ull + copy_index];
                    } else {
                        gqa_stage[destination] = v_cache[
                            (next_physical_timestep * kv_heads + kv_head) * 128ull +
                            (copy_index - 128u)];
                    }
                }
            }

            if (consumer) {
                const unsigned int stage_base = current_stage * 256u;
                if (gqa_stage_valid[current_stage] == 0) {
                    invalid = true;
                } else {
                    float dot = q0 * gqa_stage[stage_base + lane] +
                        q1 * gqa_stage[stage_base + 32u + lane] +
                        q2 * gqa_stage[stage_base + 64u + lane] +
                        q3 * gqa_stage[stage_base + 96u + lane];
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        dot += __shfl_down(dot, offset, warpSize);
                    }
                    const float score = __shfl(dot, 0, warpSize) * softmax_scale;
                    float scale = 0.0f;
                    float replaces_max = 0.0f;
                    if (lane == 0u) {
                        if (score > max_score) {
                            scale = expf(max_score - score);
                            denominator = denominator * scale + 1.0f;
                            max_score = score;
                            replaces_max = 1.0f;
                        } else {
                            scale = expf(score - max_score);
                            denominator += scale;
                        }
                    }
                    scale = __shfl(scale, 0, warpSize);
                    replaces_max = __shfl(replaces_max, 0, warpSize);
                    const float v0 = gqa_stage[stage_base + 128u + lane];
                    const float v1 = gqa_stage[stage_base + 160u + lane];
                    const float v2 = gqa_stage[stage_base + 192u + lane];
                    const float v3 = gqa_stage[stage_base + 224u + lane];
                    if (replaces_max != 0.0f) {
                        weighted0 = weighted0 * scale + v0;
                        weighted1 = weighted1 * scale + v1;
                        weighted2 = weighted2 * scale + v2;
                        weighted3 = weighted3 * scale + v3;
                    } else {
                        weighted0 += scale * v0;
                        weighted1 += scale * v1;
                        weighted2 += scale * v2;
                        weighted3 += scale * v3;
                    }
                }
            }
            __syncthreads();
        }
#else
        // Baseline grouped form: shared K/V reuse without changing the
        // original two CTA barriers per source row.  This isolates the GQA
        // reuse effect from the pipelined-barrier experiment above.
        for (unsigned long long source_offset = 0ull;
             source_offset < source_count;
             ++source_offset) {
            const unsigned long long source_timestep = source_start + source_offset;
            const unsigned long long block_index = source_timestep / block_size;
            const unsigned long long block_offset = source_timestep - block_index * block_size;
            const unsigned long long block_id =
                static_cast<unsigned long long>(block_table[block_index]);
            const bool row_valid = block_id < cache_blocks;
            if (tid == 0u) {
                gqa_stage_valid[0] = row_valid ? 1 : 0;
            }
            if (row_valid) {
                const unsigned long long physical_timestep =
                    block_id * block_size + block_offset;
                if (tid < 128u) {
                    gqa_stage[tid] =
                        k_cache[(physical_timestep * kv_heads + kv_head) * 128ull + tid];
                } else {
                    gqa_stage[tid] =
                        v_cache[(physical_timestep * kv_heads + kv_head) * 128ull + (tid - 128u)];
                }
            } else {
                gqa_stage[tid] = 0.0f;
            }
            __syncthreads();

            if (consumer) {
                if (gqa_stage_valid[0] == 0) {
                    invalid = true;
                } else {
                    float dot = q0 * gqa_stage[lane] + q1 * gqa_stage[32u + lane] +
                        q2 * gqa_stage[64u + lane] + q3 * gqa_stage[96u + lane];
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        dot += __shfl_down(dot, offset, warpSize);
                    }
                    const float score = __shfl(dot, 0, warpSize) * softmax_scale;
                    float scale = 0.0f;
                    float replaces_max = 0.0f;
                    if (lane == 0u) {
                        if (score > max_score) {
                            scale = expf(max_score - score);
                            denominator = denominator * scale + 1.0f;
                            max_score = score;
                            replaces_max = 1.0f;
                        } else {
                            scale = expf(score - max_score);
                            denominator += scale;
                        }
                    }
                    scale = __shfl(scale, 0, warpSize);
                    replaces_max = __shfl(replaces_max, 0, warpSize);
                    const float v0 = gqa_stage[128u + lane];
                    const float v1 = gqa_stage[160u + lane];
                    const float v2 = gqa_stage[192u + lane];
                    const float v3 = gqa_stage[224u + lane];
                    if (replaces_max != 0.0f) {
                        weighted0 = weighted0 * scale + v0;
                        weighted1 = weighted1 * scale + v1;
                        weighted2 = weighted2 * scale + v2;
                        weighted3 = weighted3 * scale + v3;
                    } else {
                        weighted0 += scale * v0;
                        weighted1 += scale * v1;
                        weighted2 += scale * v2;
                        weighted3 += scale * v3;
                    }
                }
            }
            __syncthreads();
        }
#endif

        if (consumer) {
            const unsigned long long workspace_base =
                (q_head * split_count + split_id) * stride;
            if (lane == 0u) {
                workspace[workspace_base] = invalid ? 0.0f : max_score;
                workspace[workspace_base + 1ull] = invalid ? -1.0f : denominator;
            }
            workspace[workspace_base + 2ull + lane] = invalid ? 0.0f : weighted0;
            workspace[workspace_base + 2ull + 32ull + lane] = invalid ? 0.0f : weighted1;
            workspace[workspace_base + 2ull + 64ull + lane] = invalid ? 0.0f : weighted2;
            workspace[workspace_base + 2ull + 96ull + lane] = invalid ? 0.0f : weighted3;
        }
        return;
    }
#endif

    if (blockDim.x != 256u || head_dim > 256ull || value_dim > 256ull) {
        return;
    }

    const unsigned long long partial_index = blockIdx.x;
    const unsigned long long q_head = partial_index / split_count;
    const unsigned long long split_id = partial_index - q_head * split_count;
    if (q_head >= q_heads) {
        return;
    }
    const unsigned int tid = threadIdx.x;
    const unsigned long long stride = value_dim + 2ull;
    const unsigned long long workspace_base = partial_index * stride;
    const unsigned long long source_start = split_id * source_tile;
    const unsigned long long source_remaining =
        source_start < cache_len ? cache_len - source_start : 0ull;
    const unsigned long long source_count =
        source_tile < source_remaining ? source_tile : source_remaining;
    const unsigned long long source_end = source_start + source_count;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_base = q_head * head_dim;

    float max_score = -3.4028234663852886e38f;
    float denominator = 0.0f;
    float weighted = 0.0f;
    for (unsigned long long source_timestep = source_start;
         source_timestep < source_end;
         ++source_timestep) {
        const unsigned long long block_index = source_timestep / block_size;
        const unsigned long long block_offset = source_timestep - block_index * block_size;
        const unsigned long long block_id =
            static_cast<unsigned long long>(block_table[block_index]);
        if (block_id >= cache_blocks) {
            if (tid == 0u) {
                workspace[workspace_base] = 0.0f;
                workspace[workspace_base + 1ull] = -1.0f;
            }
            if (tid < value_dim) {
                workspace[workspace_base + 2ull + tid] = 0.0f;
            }
            return;
        }
        const unsigned long long physical_timestep = block_id * block_size + block_offset;
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
    }
    if (tid == 0u) {
        workspace[workspace_base] = max_score;
        workspace[workspace_base + 1ull] = denominator;
    }
    if (tid < value_dim) {
        workspace[workspace_base + 2ull + tid] = weighted;
    }
}

extern "C" __global__ void ullm_paged_decode_attn_split_merge_f32_kernel(
    const float *gate,
    unsigned long long q_heads,
    unsigned long long value_dim,
    unsigned long long split_count,
    const float *workspace,
    float *output) {
    if (blockDim.x != 256u || value_dim > 256ull) {
        return;
    }
    const unsigned long long q_head = blockIdx.x;
    if (q_head >= q_heads) {
        return;
    }
    const unsigned int tid = threadIdx.x;
    const unsigned long long stride = value_dim + 2ull;
    float merged_max = -3.4028234663852886e38f;
    float merged_denominator = 0.0f;
    float merged_numerator = 0.0f;
    bool invalid = false;
    for (unsigned long long split_id = 0ull; split_id < split_count; ++split_id) {
        const unsigned long long partial_base =
            (q_head * split_count + split_id) * stride;
        const float partial_max = workspace[partial_base];
        const float partial_denominator = workspace[partial_base + 1ull];
        if (partial_denominator < 0.0f) {
            invalid = true;
            break;
        }
        const float partial_numerator =
            tid < value_dim ? workspace[partial_base + 2ull + tid] : 0.0f;
        if (split_id == 0ull) {
            merged_max = partial_max;
            merged_denominator = partial_denominator;
            merged_numerator = partial_numerator;
        } else {
            const float new_max = partial_max > merged_max ? partial_max : merged_max;
            const float alpha = expf(merged_max - new_max);
            const float beta = expf(partial_max - new_max);
            merged_denominator =
                merged_denominator * alpha + partial_denominator * beta;
            merged_numerator = merged_numerator * alpha + partial_numerator * beta;
            merged_max = new_max;
        }
    }

    if (tid < value_dim) {
        const unsigned long long output_index = q_head * value_dim + tid;
        if (invalid) {
            output[output_index] = 0.0f;
            return;
        }
        const float decoded = merged_numerator / merged_denominator;
        if (gate != nullptr) {
            const float gate_value = gate[output_index];
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            output[output_index] = sigmoid * decoded;
        } else {
            output[output_index] = decoded;
        }
    }
}

// Typed direct decode is compiled as one body per ordered K/V pair.  The
// legacy F32 kernel above remains the F32/F32 implementation and is not
// textually changed by this addition.
enum {
    ULLM_TYPED_DECODE_F32 = 0,
    ULLM_TYPED_DECODE_F16 = 1,
    ULLM_TYPED_DECODE_FP8_E4M3FN = 2,
};

__device__ __forceinline__ float ullm_typed_decode_f16_to_f32(unsigned short bits) {
    const float sign = (bits & 0x8000u) == 0u ? 1.0f : -1.0f;
    const unsigned int exponent = (static_cast<unsigned int>(bits) >> 10u) & 0x1fu;
    const unsigned int mantissa = static_cast<unsigned int>(bits) & 0x03ffu;
    if (exponent == 0u) {
        return mantissa == 0u ? sign * 0.0f : sign * ldexpf(static_cast<float>(mantissa), -24);
    }
    if (exponent == 0x1fu) {
        return mantissa == 0u ? sign * __int_as_float(0x7f800000) : __int_as_float(0x7fc00000);
    }
    return sign * ldexpf(1.0f + static_cast<float>(mantissa) * 0.0009765625f,
                         static_cast<int>(exponent) - 15);
}

__device__ __forceinline__ float ullm_typed_decode_fp8_e4m3fn_to_f32(unsigned char value) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_cvt_f32_fp8(static_cast<unsigned int>(value), 0);
#else
    const unsigned int raw = static_cast<unsigned int>(value);
    const unsigned int sign = raw >> 7u;
    const unsigned int exponent = (raw >> 3u) & 0x0fu;
    const unsigned int mantissa = raw & 0x07u;
    if (exponent == 0x0fu && mantissa == 0x07u) {
        return __uint_as_float(0x7fc00000u);
    }
    const float magnitude = exponent == 0u
        ? static_cast<float>(mantissa) * 0.001953125f
        : ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f,
                 static_cast<int>(exponent) - 7);
    return sign == 0u ? magnitude : -magnitude;
#endif
}

template <int DTYPE>
__device__ __forceinline__ float ullm_typed_decode_row_scale(
    const unsigned short *scales,
    unsigned long long physical_head) {
    if constexpr (DTYPE == ULLM_TYPED_DECODE_FP8_E4M3FN) {
        const unsigned short bits = scales[physical_head];
        // A raw +0 is the reset marker.  Normal writers never emit a
        // negative/non-finite scale, but avoiding a non-finite accumulator
        // here keeps an unwritten row semantically zero on device as on CPU.
        const float scale = ullm_typed_decode_f16_to_f32(bits);
        return bits == 0u || !isfinite(scale) || scale <= 0.0f ? 0.0f : scale;
    }
    (void)scales;
    (void)physical_head;
    return 1.0f;
}

template <int DTYPE>
__device__ __forceinline__ float ullm_typed_decode_load(
    const void *payload,
    unsigned long long index,
    float row_scale) {
    if constexpr (DTYPE == ULLM_TYPED_DECODE_F32) {
        return static_cast<const float *>(payload)[index];
    } else if constexpr (DTYPE == ULLM_TYPED_DECODE_F16) {
        return ullm_typed_decode_f16_to_f32(static_cast<const unsigned short *>(payload)[index]);
    } else {
        return ullm_typed_decode_fp8_e4m3fn_to_f32(
                   static_cast<const unsigned char *>(payload)[index]) * row_scale;
    }
}

template <int K_DTYPE, int V_DTYPE>
__device__ __forceinline__ void ullm_paged_decode_attn_typed_kernel_impl(
    const float *q,
    const void *k_cache,
    const void *v_cache,
    const unsigned int *block_table,
    const unsigned short *k_scales,
    const unsigned short *v_scales,
    unsigned long long cache_len,
    unsigned long long block_size,
    unsigned long long cache_blocks,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float *output,
    float *partial,
    float *shared_scales) {
    if (gridDim.x == q_heads && blockDim.x == 256u && head_dim <= 256ull &&
        value_dim <= 256ull) {
        const unsigned long long q_head = blockIdx.x;
        const unsigned int tid = threadIdx.x;
        const unsigned long long q_per_kv = q_heads / kv_heads;
        const unsigned long long kv_head = q_head / q_per_kv;
        const unsigned long long q_base = q_head * head_dim;
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
        unsigned long long physical = block_id * block_size;
        for (unsigned long long source = 0ull; source < cache_len; ++source) {
            if (source == block_end) {
                ++block_index;
                block_end += block_size;
                block_id = static_cast<unsigned long long>(block_table[block_index]);
                if (block_id >= cache_blocks) {
                    if (tid < value_dim) {
                        output[q_head * value_dim + tid] = 0.0f;
                    }
                    return;
                }
                physical = block_id * block_size;
            }
            const unsigned long long physical_head = physical * kv_heads + kv_head;
            // One lane loads each FP8 row scale; every thread in the CTA
            // shares it for QK/AV.  This is intentionally outside the
            // payload loops and avoids one scale global load per element.
            if (tid == 0u) {
                shared_scales[0] = ullm_typed_decode_row_scale<K_DTYPE>(k_scales, physical_head);
                shared_scales[1] = ullm_typed_decode_row_scale<V_DTYPE>(v_scales, physical_head);
            }
            __syncthreads();
            const unsigned long long k_base = physical_head * head_dim;
            const float local = tid < head_dim
                ? q[q_base + tid] * ullm_typed_decode_load<K_DTYPE>(
                    k_cache, k_base + tid, shared_scales[0])
                : 0.0f;
            const float score = ullm_paged_decode_reduce_sum_256(local, partial) * softmax_scale;
            if (tid < value_dim) {
                const unsigned long long v_index = physical_head * value_dim + tid;
                const float value = ullm_typed_decode_load<V_DTYPE>(
                    v_cache, v_index, shared_scales[1]);
                if (score > max_score) {
                    const float scale = expf(max_score - score);
                    weighted = weighted * scale + value;
                    denominator = denominator * scale + 1.0f;
                    max_score = score;
                } else {
                    const float weight = expf(score - max_score);
                    weighted += weight * value;
                    denominator += weight;
                }
            }
            // Do not let lane zero overwrite the two row scales while a
            // sibling lane is still consuming the V value from this row.
            __syncthreads();
            ++physical;
        }
        if (tid < value_dim) {
            output[q_head * value_dim + tid] = weighted / denominator;
        }
        return;
    }

    // Shape-generic fallback.  Each output element owns its scalar online
    // softmax state; dtype selection is still compile-time specialization.
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long output_elements = q_heads * value_dim;
    if (index >= output_elements) {
        return;
    }
    const unsigned long long value_dim_index = index % value_dim;
    const unsigned long long q_head = index / value_dim;
    const unsigned long long kv_head = q_head / (q_heads / kv_heads);
    const unsigned long long q_base = q_head * head_dim;
    float max_score = -3.4028234663852886e38f;
    for (unsigned long long source = 0ull; source < cache_len; ++source) {
        const unsigned long long block_index = source / block_size;
        const unsigned long long block_offset = source - block_index * block_size;
        const unsigned long long block_id = static_cast<unsigned long long>(block_table[block_index]);
        if (block_id >= cache_blocks) {
            output[index] = 0.0f;
            return;
        }
        const unsigned long long physical = block_id * block_size + block_offset;
        const unsigned long long physical_head = physical * kv_heads + kv_head;
        const float k_scale = ullm_typed_decode_row_scale<K_DTYPE>(k_scales, physical_head);
        float score = 0.0f;
        const unsigned long long k_base = physical_head * head_dim;
        for (unsigned long long dim = 0ull; dim < head_dim; ++dim) {
            score += q[q_base + dim] * ullm_typed_decode_load<K_DTYPE>(
                k_cache, k_base + dim, k_scale);
        }
        max_score = fmaxf(max_score, score * softmax_scale);
    }
    float denominator = 0.0f;
    float weighted = 0.0f;
    for (unsigned long long source = 0ull; source < cache_len; ++source) {
        const unsigned long long block_index = source / block_size;
        const unsigned long long block_offset = source - block_index * block_size;
        const unsigned long long block_id = static_cast<unsigned long long>(block_table[block_index]);
        if (block_id >= cache_blocks) {
            output[index] = 0.0f;
            return;
        }
        const unsigned long long physical = block_id * block_size + block_offset;
        const unsigned long long physical_head = physical * kv_heads + kv_head;
        const float k_scale = ullm_typed_decode_row_scale<K_DTYPE>(k_scales, physical_head);
        const float v_scale = ullm_typed_decode_row_scale<V_DTYPE>(v_scales, physical_head);
        float score = 0.0f;
        const unsigned long long k_base = physical_head * head_dim;
        for (unsigned long long dim = 0ull; dim < head_dim; ++dim) {
            score += q[q_base + dim] * ullm_typed_decode_load<K_DTYPE>(
                k_cache, k_base + dim, k_scale);
        }
        const float weight = expf(score * softmax_scale - max_score);
        denominator += weight;
        weighted += weight * ullm_typed_decode_load<V_DTYPE>(
            v_cache, physical_head * value_dim + value_dim_index, v_scale);
    }
    output[index] = weighted / denominator;
}

#define ULLM_DEFINE_TYPED_PAGED_DECODE(K_DTYPE, V_DTYPE, K_NAME, V_NAME) \
extern "C" __global__ void ullm_paged_decode_attn_typed_k##K_NAME##_v##V_NAME##_kernel( \
    const float *q, const void *k_cache, const void *v_cache, const unsigned int *block_table, \
    const unsigned short *k_scales, const unsigned short *v_scales, \
    unsigned long long cache_len, unsigned long long block_size, unsigned long long cache_blocks, \
    unsigned long long q_heads, unsigned long long kv_heads, unsigned long long head_dim, \
    unsigned long long value_dim, float softmax_scale, float *output) { \
    __shared__ float partial[256]; \
    __shared__ float shared_scales[2]; \
    ullm_paged_decode_attn_typed_kernel_impl<K_DTYPE, V_DTYPE>( \
        q, k_cache, v_cache, block_table, k_scales, v_scales, cache_len, block_size, cache_blocks, \
        q_heads, kv_heads, head_dim, value_dim, softmax_scale, output, partial, shared_scales); \
}

ULLM_DEFINE_TYPED_PAGED_DECODE(ULLM_TYPED_DECODE_F32, ULLM_TYPED_DECODE_F16, f32, f16)
ULLM_DEFINE_TYPED_PAGED_DECODE(ULLM_TYPED_DECODE_F32, ULLM_TYPED_DECODE_FP8_E4M3FN, f32, fp8)
ULLM_DEFINE_TYPED_PAGED_DECODE(ULLM_TYPED_DECODE_F16, ULLM_TYPED_DECODE_F32, f16, f32)
ULLM_DEFINE_TYPED_PAGED_DECODE(ULLM_TYPED_DECODE_F16, ULLM_TYPED_DECODE_F16, f16, f16)
ULLM_DEFINE_TYPED_PAGED_DECODE(ULLM_TYPED_DECODE_F16, ULLM_TYPED_DECODE_FP8_E4M3FN, f16, fp8)
ULLM_DEFINE_TYPED_PAGED_DECODE(ULLM_TYPED_DECODE_FP8_E4M3FN, ULLM_TYPED_DECODE_F32, fp8, f32)
ULLM_DEFINE_TYPED_PAGED_DECODE(ULLM_TYPED_DECODE_FP8_E4M3FN, ULLM_TYPED_DECODE_F16, fp8, f16)
ULLM_DEFINE_TYPED_PAGED_DECODE(ULLM_TYPED_DECODE_FP8_E4M3FN, ULLM_TYPED_DECODE_FP8_E4M3FN, fp8, fp8)
#undef ULLM_DEFINE_TYPED_PAGED_DECODE

// Typed split partials retain the reviewed F32 merge kernel.  The 16Q/4KV,
// 256-wide Qwen3.5 path is deliberately grouped: a CTA stages one dequantized
// K/V row and its two FP8 scales once for all four query heads sharing that KV
// head.  Other valid geometries use the scalar one-query-head body below.
template <int K_DTYPE, int V_DTYPE>
__device__ __forceinline__ void ullm_paged_decode_attn_split_typed_kernel_impl(
    const float *q, const void *k_cache, const void *v_cache,
    const unsigned int *block_table, const unsigned short *k_scales,
    const unsigned short *v_scales, unsigned long long cache_len,
    unsigned long long block_size, unsigned long long cache_blocks,
    unsigned long long q_heads, unsigned long long kv_heads,
    unsigned long long head_dim, unsigned long long value_dim,
    float softmax_scale, unsigned long long source_tile,
    unsigned long long split_count, float *workspace, float *partial,
    float *shared_scales, float *gqa_stage, float *gqa_state) {
    const unsigned int tid = threadIdx.x;
    const unsigned long long q_per_kv = kv_heads == 0ull ? 0ull : q_heads / kv_heads;

    // Qwen3.5 AQ4_0 decode: four query heads share a KV head.  The host only
    // selects this grid form for this exact geometry, keeping the legacy F32
    // grouped (5x128) experiment completely isolated.
    if (blockDim.x == 256u && q_per_kv == 4ull && head_dim == 256ull &&
        value_dim == 256ull && static_cast<unsigned long long>(gridDim.x) ==
            kv_heads * split_count) {
        const unsigned long long grouped_index = blockIdx.x;
        const unsigned long long kv_head = grouped_index / split_count;
        const unsigned long long split_id = grouped_index - kv_head * split_count;
        if (kv_head >= kv_heads) {
            return;
        }
        const unsigned long long source_start = split_id * source_tile;
        const unsigned long long source_remaining =
            source_start < cache_len ? cache_len - source_start : 0ull;
        const unsigned long long source_count =
            source_tile < source_remaining ? source_tile : source_remaining;
        const unsigned int group = tid >> 6u;
        const unsigned int lane = tid & 63u;
        const unsigned long long q_head = kv_head * 4ull + static_cast<unsigned long long>(group);
        const unsigned long long q_base = q_head * 256ull;
        float q0 = q[q_base + lane];
        float q1 = q[q_base + 64ull + lane];
        float q2 = q[q_base + 128ull + lane];
        float q3 = q[q_base + 192ull + lane];
        float weighted0 = 0.0f;
        float weighted1 = 0.0f;
        float weighted2 = 0.0f;
        float weighted3 = 0.0f;
        float max_score = -3.4028234663852886e38f;
        float denominator = 0.0f;
        bool invalid = false;
        for (unsigned long long offset = 0ull; offset < source_count; ++offset) {
            const unsigned long long source = source_start + offset;
            const unsigned long long logical_block = source / block_size;
            const unsigned long long block_offset = source - logical_block * block_size;
            const unsigned long long block_id =
                static_cast<unsigned long long>(block_table[logical_block]);
            const bool row_valid = block_id < cache_blocks;
            if (tid == 0u) {
                gqa_state[0] = row_valid ? 1.0f : 0.0f;
                if (row_valid) {
                    const unsigned long long physical = block_id * block_size + block_offset;
                    const unsigned long long physical_head = physical * kv_heads + kv_head;
                    shared_scales[0] = ullm_typed_decode_row_scale<K_DTYPE>(
                        k_scales, physical_head);
                    shared_scales[1] = ullm_typed_decode_row_scale<V_DTYPE>(
                        v_scales, physical_head);
                }
            }
            __syncthreads();
            if (gqa_state[0] == 0.0f) {
                invalid = true;
                break;
            }
            const unsigned long long physical = block_id * block_size + block_offset;
            const unsigned long long physical_head = physical * kv_heads + kv_head;
            for (unsigned int element = tid; element < 512u; element += blockDim.x) {
                if (element < 256u) {
                    gqa_stage[element] = ullm_typed_decode_load<K_DTYPE>(
                        k_cache, physical_head * 256ull + element, shared_scales[0]);
                } else {
                    gqa_stage[element] = ullm_typed_decode_load<V_DTYPE>(
                        v_cache, physical_head * 256ull + (element - 256u), shared_scales[1]);
                }
            }
            __syncthreads();
            partial[tid] = q0 * gqa_stage[lane] + q1 * gqa_stage[64u + lane] +
                q2 * gqa_stage[128u + lane] + q3 * gqa_stage[192u + lane];
            for (unsigned int reduce = 32u; reduce > 0u; reduce >>= 1u) {
                if (lane < reduce) {
                    partial[tid] += partial[tid + reduce];
                }
                __syncthreads();
            }
            if (lane == 0u) {
                const float score = partial[tid] * softmax_scale;
                const float new_max = fmaxf(max_score, score);
                gqa_state[4u + group] =
                    max_score <= -3.0e38f ? 0.0f : expf(max_score - new_max);
                gqa_state[8u + group] = expf(score - new_max);
                gqa_state[12u + group] = new_max;
                denominator = denominator * gqa_state[4u + group] + gqa_state[8u + group];
                gqa_state[16u + group] = denominator;
                max_score = new_max;
            }
            __syncthreads();
            const float previous = gqa_state[4u + group];
            const float weight = gqa_state[8u + group];
            weighted0 = weighted0 * previous + weight * gqa_stage[256u + lane];
            weighted1 = weighted1 * previous + weight * gqa_stage[320u + lane];
            weighted2 = weighted2 * previous + weight * gqa_stage[384u + lane];
            weighted3 = weighted3 * previous + weight * gqa_stage[448u + lane];
            __syncthreads();
        }
        const unsigned long long stride = value_dim + 2ull;
        const unsigned long long workspace_base = (q_head * split_count + split_id) * stride;
        if (lane == 0u) {
            workspace[workspace_base] = invalid ? 0.0f : max_score;
            workspace[workspace_base + 1ull] = invalid ? -1.0f : denominator;
        }
        workspace[workspace_base + 2ull + lane] = invalid ? 0.0f : weighted0;
        workspace[workspace_base + 66ull + lane] = invalid ? 0.0f : weighted1;
        workspace[workspace_base + 130ull + lane] = invalid ? 0.0f : weighted2;
        workspace[workspace_base + 194ull + lane] = invalid ? 0.0f : weighted3;
        return;
    }

    if (blockDim.x != 256u || head_dim > 256ull || value_dim > 256ull) {
        return;
    }
    const unsigned long long partial_index = blockIdx.x;
    const unsigned long long q_head = partial_index / split_count;
    const unsigned long long split_id = partial_index - q_head * split_count;
    if (q_head >= q_heads) {
        return;
    }
    const unsigned long long stride = value_dim + 2ull;
    const unsigned long long workspace_base = partial_index * stride;
    const unsigned long long source_start = split_id * source_tile;
    const unsigned long long source_remaining =
        source_start < cache_len ? cache_len - source_start : 0ull;
    const unsigned long long source_count =
        source_tile < source_remaining ? source_tile : source_remaining;
    const unsigned long long q_head_kv = q_head / q_per_kv;
    const unsigned long long q_base = q_head * head_dim;
    float max_score = -3.4028234663852886e38f;
    float denominator = 0.0f;
    float weighted = 0.0f;
    for (unsigned long long offset = 0ull; offset < source_count; ++offset) {
        const unsigned long long source = source_start + offset;
        const unsigned long long logical_block = source / block_size;
        const unsigned long long block_offset = source - logical_block * block_size;
        const unsigned long long block_id =
            static_cast<unsigned long long>(block_table[logical_block]);
        if (block_id >= cache_blocks) {
            if (tid == 0u) {
                workspace[workspace_base] = 0.0f;
                workspace[workspace_base + 1ull] = -1.0f;
            }
            if (tid < value_dim) {
                workspace[workspace_base + 2ull + tid] = 0.0f;
            }
            return;
        }
        const unsigned long long physical = block_id * block_size + block_offset;
        const unsigned long long physical_head = physical * kv_heads + q_head_kv;
        if (tid == 0u) {
            shared_scales[0] = ullm_typed_decode_row_scale<K_DTYPE>(k_scales, physical_head);
            shared_scales[1] = ullm_typed_decode_row_scale<V_DTYPE>(v_scales, physical_head);
        }
        __syncthreads();
        float dot = 0.0f;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            dot += q[q_base + dim] * ullm_typed_decode_load<K_DTYPE>(
                k_cache, physical_head * head_dim + dim, shared_scales[0]);
        }
        const float score = ullm_paged_decode_reduce_sum_256(dot, partial) * softmax_scale;
        if (tid < value_dim) {
            const float value = ullm_typed_decode_load<V_DTYPE>(
                v_cache, physical_head * value_dim + tid, shared_scales[1]);
            if (score > max_score) {
                const float rescale = expf(max_score - score);
                weighted = weighted * rescale + value;
                denominator = denominator * rescale + 1.0f;
                max_score = score;
            } else {
                const float weight = expf(score - max_score);
                weighted += weight * value;
                denominator += weight;
            }
        }
        __syncthreads();
    }
    if (tid == 0u) {
        workspace[workspace_base] = max_score;
        workspace[workspace_base + 1ull] = denominator;
    }
    if (tid < value_dim) {
        workspace[workspace_base + 2ull + tid] = weighted;
    }
}

#define ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT(K_DTYPE, V_DTYPE, K_NAME, V_NAME) \
extern "C" __global__ void ullm_paged_decode_attn_split_typed_k##K_NAME##_v##V_NAME##_partial_kernel( \
    const float *q, const void *k_cache, const void *v_cache, const unsigned int *block_table, \
    const unsigned short *k_scales, const unsigned short *v_scales, \
    unsigned long long cache_len, unsigned long long block_size, unsigned long long cache_blocks, \
    unsigned long long q_heads, unsigned long long kv_heads, unsigned long long head_dim, \
    unsigned long long value_dim, float softmax_scale, unsigned long long source_tile, \
    unsigned long long split_count, float *workspace) { \
    __shared__ float partial[256]; \
    __shared__ float shared_scales[2]; \
    __shared__ float gqa_stage[512]; \
    __shared__ float gqa_state[20]; \
    ullm_paged_decode_attn_split_typed_kernel_impl<K_DTYPE, V_DTYPE>( \
        q, k_cache, v_cache, block_table, k_scales, v_scales, cache_len, block_size, cache_blocks, \
        q_heads, kv_heads, head_dim, value_dim, softmax_scale, source_tile, split_count, workspace, \
        partial, shared_scales, gqa_stage, gqa_state); \
}

ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT(ULLM_TYPED_DECODE_F32, ULLM_TYPED_DECODE_F16, f32, f16)
ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT(ULLM_TYPED_DECODE_F32, ULLM_TYPED_DECODE_FP8_E4M3FN, f32, fp8)
ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT(ULLM_TYPED_DECODE_F16, ULLM_TYPED_DECODE_F32, f16, f32)
ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT(ULLM_TYPED_DECODE_F16, ULLM_TYPED_DECODE_F16, f16, f16)
ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT(ULLM_TYPED_DECODE_F16, ULLM_TYPED_DECODE_FP8_E4M3FN, f16, fp8)
ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT(ULLM_TYPED_DECODE_FP8_E4M3FN, ULLM_TYPED_DECODE_F32, fp8, f32)
ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT(ULLM_TYPED_DECODE_FP8_E4M3FN, ULLM_TYPED_DECODE_F16, fp8, f16)
ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT(ULLM_TYPED_DECODE_FP8_E4M3FN, ULLM_TYPED_DECODE_FP8_E4M3FN, fp8, fp8)
#undef ULLM_DEFINE_TYPED_PAGED_DECODE_SPLIT

// Exact Gemma4 E2B full-attention prefill reader.  One CTA owns one of the
// eight Q heads and retains two V accumulators per query row in registers.
// K/V for each source token is loaded once into LDS then reused for the whole
// query chunk.  The `source <= prefix + row` check is deliberately inside the
// row loop: moving it outside makes early queries see later chunk tokens.
extern "C" __global__ void ullm_gemma_full_attn_batched_512_f32_kernel(
    const float *q, const float *k_cache, const float *v_cache,
    const unsigned int *block_table, unsigned long long prefix_len,
    unsigned long long query_rows, unsigned long long cache_len,
    unsigned long long block_size, unsigned long long cache_blocks,
    float softmax_scale, float *output) {
    const unsigned int tid = threadIdx.x;
    const unsigned long long q_head = static_cast<unsigned long long>(blockIdx.x);
    if (q_head >= 8ull || blockDim.x != 256u || query_rows == 0ull || query_rows > 128ull ||
        prefix_len + query_rows > cache_len) {
        return;
    }
    __shared__ float cached_k[512];
    __shared__ float cached_v[512];
    __shared__ float reduce[256];
    __shared__ float row_max[128];
    __shared__ float row_denom[128];
    __shared__ float update_scale;
    __shared__ float update_weight;

    float weighted_lo[128];
    float weighted_hi[128];
    #pragma unroll
    for (unsigned int row = 0; row < 128u; ++row) {
        weighted_lo[row] = 0.0f;
        weighted_hi[row] = 0.0f;
        if (tid == 0u) {
            row_max[row] = -3.4028234663852886e38f;
            row_denom[row] = 0.0f;
        }
    }
    __syncthreads();

    for (unsigned long long source = 0; source < prefix_len + query_rows; ++source) {
        const unsigned long long logical_block = source / block_size;
        const unsigned long long block_offset = source - logical_block * block_size;
        if (logical_block >= (cache_len - 1ull) / block_size + 1ull) {
            return;
        }
        const unsigned long long block_id = static_cast<unsigned long long>(block_table[logical_block]);
        if (block_id >= cache_blocks) {
            return;
        }
        const unsigned long long physical = block_id * block_size + block_offset;
        for (unsigned int value = tid; value < 512u; value += 256u) {
            cached_k[value] = k_cache[physical * 512ull + value];
            cached_v[value] = v_cache[physical * 512ull + value];
        }
        __syncthreads();
        for (unsigned long long row = 0; row < query_rows; ++row) {
            if (source > prefix_len + row) {
                continue;
            }
            const unsigned long long q_base = (row * 8ull + q_head) * 512ull;
            float dot = q[q_base + tid] * cached_k[tid] +
                        q[q_base + static_cast<unsigned long long>(tid) + 256ull] *
                            cached_k[tid + 256u];
            reduce[tid] = dot;
            __syncthreads();
            for (unsigned int stride = 128u; stride > 0u; stride >>= 1u) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0u) {
                const float score = reduce[0] * softmax_scale;
                const float old_max = row_max[row];
                const float new_max = score > old_max ? score : old_max;
                update_scale = old_max <= -3.4028234663852886e38f ? 0.0f : expf(old_max - new_max);
                update_weight = expf(score - new_max);
                row_max[row] = new_max;
                row_denom[row] = row_denom[row] * update_scale + update_weight;
            }
            __syncthreads();
            weighted_lo[row] = weighted_lo[row] * update_scale + update_weight * cached_v[tid];
            weighted_hi[row] = weighted_hi[row] * update_scale + update_weight * cached_v[tid + 256u];
            __syncthreads();
        }
    }
    for (unsigned long long row = 0; row < query_rows; ++row) {
        const unsigned long long output_base = (row * 8ull + q_head) * 512ull;
        weighted_lo[row] /= row_denom[row];
        weighted_hi[row] /= row_denom[row];
        output[output_base + tid] = weighted_lo[row];
        output[output_base + static_cast<unsigned long long>(tid) + 256ull] = weighted_hi[row];
    }
}

// Gemma4's local-prefill reader deliberately separates the pre-write ring
// history from this chunk's freshly projected K/V.  A full ring cannot be
// overwritten before row 0 has consumed its oldest entry: direct logical to
// physical mapping preserves that pre-write history without a snapshot copy.
extern "C" __global__ void ullm_gemma_sliding_attn_ring_batched_256_f32_kernel(
    const float *q, const float *k_ring, const float *v_ring,
    const float *fresh_k, const float *fresh_v,
    unsigned long long prefix_len, unsigned long long history_rows,
    unsigned long long query_rows, unsigned long long ring_capacity,
    unsigned long long q_heads, float softmax_scale, float *output) {
    const unsigned int tid = threadIdx.x;
    const unsigned long long q_head = static_cast<unsigned long long>(blockIdx.x);
    if (q_head >= q_heads || blockDim.x != 256u || q_heads == 0ull ||
        query_rows == 0ull || query_rows > 128ull || history_rows > 511ull ||
        history_rows > prefix_len || ring_capacity == 0ull) {
        return;
    }
    __shared__ float cached_k[256];
    __shared__ float cached_v[256];
    __shared__ float reduce[256];
    __shared__ float row_max[128];
    __shared__ float row_denom[128];
    __shared__ float update_scale;
    __shared__ float update_weight;
    float weighted[128];
    #pragma unroll
    for (unsigned int row = 0; row < 128u; ++row) {
        weighted[row] = 0.0f;
        if (tid == 0u) {
            row_max[row] = -3.4028234663852886e38f;
            row_denom[row] = 0.0f;
        }
    }
    __syncthreads();

    const unsigned long long source_rows = history_rows + query_rows;
    for (unsigned long long source = 0; source < source_rows; ++source) {
        if (source < history_rows) {
            const unsigned long long logical = prefix_len - history_rows + source;
            const unsigned long long physical = logical % ring_capacity;
            cached_k[tid] = k_ring[physical * 256ull + tid];
            cached_v[tid] = v_ring[physical * 256ull + tid];
        } else {
            const unsigned long long fresh = source - history_rows;
            cached_k[tid] = fresh_k[fresh * 256ull + tid];
            cached_v[tid] = fresh_v[fresh * 256ull + tid];
        }
        __syncthreads();
        for (unsigned long long row = 0; row < query_rows; ++row) {
            // History is wholly visible. Fresh K/V is visible only through
            // this query's row, enforcing the upper causal bound.
            if (source >= history_rows && source - history_rows > row) continue;
            const unsigned long long q_base = (row * q_heads + q_head) * 256ull;
            reduce[tid] = q[q_base + tid] * cached_k[tid];
            __syncthreads();
            for (unsigned int stride = 128u; stride > 0u; stride >>= 1u) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0u) {
                const float score = reduce[0] * softmax_scale;
                const float old_max = row_max[row];
                const float new_max = score > old_max ? score : old_max;
                update_scale = old_max <= -3.4028234663852886e38f ? 0.0f : expf(old_max - new_max);
                update_weight = expf(score - new_max);
                row_max[row] = new_max;
                row_denom[row] = row_denom[row] * update_scale + update_weight;
            }
            __syncthreads();
            weighted[row] = weighted[row] * update_scale + update_weight * cached_v[tid];
            __syncthreads();
        }
    }
    for (unsigned long long row = 0; row < query_rows; ++row) {
        output[(row * q_heads + q_head) * 256ull + tid] = weighted[row] / row_denom[row];
    }
}

// Gemma split-KV prefill readers.  These are intentionally separate from the
// production paged-decode split ABI: Gemma retains every row of a prefill
// chunk in a CTA, so each partial writes one online-softmax state per row.
// Workspace layout is [row][q_head][split][max, denominator, numerator...].
extern "C" __global__ void ullm_gemma_full_attn_batched_512_split_partial_f32_kernel(
    const float *q, const float *k_cache, const float *v_cache,
    const unsigned int *block_table, unsigned long long prefix_len,
    unsigned long long query_rows, unsigned long long cache_len,
    unsigned long long block_size, unsigned long long cache_blocks,
    float softmax_scale, unsigned long long split_count, float *workspace) {
    const unsigned int tid = threadIdx.x;
    const unsigned long long partial_index = blockIdx.x;
    const unsigned long long q_head = partial_index / split_count;
    const unsigned long long split_id = partial_index - q_head * split_count;
    if (q_head >= 8ull || blockDim.x != 256u || query_rows == 0ull || query_rows > 128ull ||
        prefix_len + query_rows > cache_len || split_count == 0ull) return;
    const unsigned long long visible = prefix_len + query_rows;
    const unsigned long long tile = (visible + split_count - 1ull) / split_count;
    const unsigned long long source_start = split_id * tile;
    const unsigned long long source_end = source_start < visible
        ? min(visible, source_start + tile) : source_start;
    __shared__ float cached_k[512];
    __shared__ float cached_v[512];
    __shared__ float reduce[256];
    __shared__ float row_max[128];
    __shared__ float row_denom[128];
    __shared__ float update_scale;
    __shared__ float update_weight;
    float weighted_lo[128];
    float weighted_hi[128];
    #pragma unroll
    for (unsigned int row = 0; row < 128u; ++row) {
        weighted_lo[row] = 0.0f;
        weighted_hi[row] = 0.0f;
        if (tid == 0u) { row_max[row] = -3.4028234663852886e38f; row_denom[row] = 0.0f; }
    }
    __syncthreads();
    for (unsigned long long source = source_start; source < source_end; ++source) {
        const unsigned long long logical_block = source / block_size;
        const unsigned long long block_offset = source - logical_block * block_size;
        const unsigned long long block_id = static_cast<unsigned long long>(block_table[logical_block]);
        if (block_id >= cache_blocks) return;
        const unsigned long long physical = block_id * block_size + block_offset;
        for (unsigned int value = tid; value < 512u; value += 256u) {
            cached_k[value] = k_cache[physical * 512ull + value];
            cached_v[value] = v_cache[physical * 512ull + value];
        }
        __syncthreads();
        for (unsigned long long row = 0; row < query_rows; ++row) {
            if (source > prefix_len + row) continue;
            const unsigned long long q_base = (row * 8ull + q_head) * 512ull;
            reduce[tid] = q[q_base + tid] * cached_k[tid] +
                q[q_base + tid + 256ull] * cached_k[tid + 256u];
            __syncthreads();
            for (unsigned int stride = 128u; stride > 0u; stride >>= 1u) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0u) {
                const float score = reduce[0] * softmax_scale;
                const float old_max = row_max[row];
                const float new_max = score > old_max ? score : old_max;
                update_scale = old_max <= -3.0e38f ? 0.0f : expf(old_max - new_max);
                update_weight = expf(score - new_max);
                row_max[row] = new_max;
                row_denom[row] = row_denom[row] * update_scale + update_weight;
            }
            __syncthreads();
            weighted_lo[row] = weighted_lo[row] * update_scale + update_weight * cached_v[tid];
            weighted_hi[row] = weighted_hi[row] * update_scale + update_weight * cached_v[tid + 256u];
            __syncthreads();
        }
    }
    const unsigned long long stride = 514ull;
    for (unsigned long long row = 0; row < query_rows; ++row) {
        const unsigned long long base = ((row * 8ull + q_head) * split_count + split_id) * stride;
        if (tid == 0u) { workspace[base] = row_max[row]; workspace[base + 1ull] = row_denom[row]; }
        workspace[base + 2ull + tid] = weighted_lo[row];
        workspace[base + 258ull + tid] = weighted_hi[row];
    }
}

extern "C" __global__ void ullm_gemma_full_attn_batched_512_split_merge_f32_kernel(
    unsigned long long query_rows, unsigned long long split_count,
    const float *workspace, float *output) {
    const unsigned int tid = threadIdx.x;
    const unsigned long long q_head = blockIdx.x;
    if (q_head >= 8ull || blockDim.x != 256u || query_rows == 0ull || query_rows > 128ull) return;
    const unsigned long long stride = 514ull;
    for (unsigned long long row = 0; row < query_rows; ++row) {
        float merged_max = -3.4028234663852886e38f;
        float merged_denom = 0.0f;
        float merged_lo = 0.0f;
        float merged_hi = 0.0f;
        for (unsigned long long split = 0; split < split_count; ++split) {
            const unsigned long long base = ((row * 8ull + q_head) * split_count + split) * stride;
            const float denom = workspace[base + 1ull];
            if (denom == 0.0f) continue;
            const float partial_max = workspace[base];
            const float new_max = partial_max > merged_max ? partial_max : merged_max;
            const float alpha = merged_denom == 0.0f ? 0.0f : expf(merged_max - new_max);
            const float beta = expf(partial_max - new_max);
            merged_lo = merged_lo * alpha + workspace[base + 2ull + tid] * beta;
            merged_hi = merged_hi * alpha + workspace[base + 258ull + tid] * beta;
            merged_denom = merged_denom * alpha + denom * beta;
            merged_max = new_max;
        }
        const unsigned long long out = (row * 8ull + q_head) * 512ull;
        output[out + tid] = merged_lo / merged_denom;
        output[out + tid + 256ull] = merged_hi / merged_denom;
    }
}

extern "C" __global__ void ullm_gemma_sliding_attn_ring_batched_256_split_partial_f32_kernel(
    const float *q, const float *k_ring, const float *v_ring,
    const float *fresh_k, const float *fresh_v, unsigned long long prefix_len,
    unsigned long long history_rows, unsigned long long query_rows,
    unsigned long long ring_capacity, unsigned long long q_heads,
    float softmax_scale, unsigned long long split_count, float *workspace) {
    const unsigned int tid = threadIdx.x;
    const unsigned long long partial_index = blockIdx.x;
    const unsigned long long q_head = partial_index / split_count;
    const unsigned long long split_id = partial_index - q_head * split_count;
    if (q_head >= q_heads || blockDim.x != 256u || query_rows == 0ull || query_rows > 128ull ||
        history_rows > 511ull || history_rows > prefix_len || ring_capacity == 0ull || split_count == 0ull) return;
    const unsigned long long sources = history_rows + query_rows;
    const unsigned long long tile = (sources + split_count - 1ull) / split_count;
    const unsigned long long source_start = split_id * tile;
    const unsigned long long source_end = source_start < sources ? min(sources, source_start + tile) : source_start;
    __shared__ float cached_k[256]; __shared__ float cached_v[256]; __shared__ float reduce[256];
    __shared__ float row_max[128]; __shared__ float row_denom[128];
    __shared__ float update_scale; __shared__ float update_weight;
    float weighted[128];
    #pragma unroll
    for (unsigned int row = 0; row < 128u; ++row) {
        weighted[row] = 0.0f;
        if (tid == 0u) { row_max[row] = -3.4028234663852886e38f; row_denom[row] = 0.0f; }
    }
    __syncthreads();
    for (unsigned long long source = source_start; source < source_end; ++source) {
        if (source < history_rows) {
            const unsigned long long logical = prefix_len - history_rows + source;
            const unsigned long long physical = logical % ring_capacity;
            cached_k[tid] = k_ring[physical * 256ull + tid]; cached_v[tid] = v_ring[physical * 256ull + tid];
        } else {
            const unsigned long long fresh = source - history_rows;
            cached_k[tid] = fresh_k[fresh * 256ull + tid]; cached_v[tid] = fresh_v[fresh * 256ull + tid];
        }
        __syncthreads();
        for (unsigned long long row = 0; row < query_rows; ++row) {
            if (source >= history_rows && source - history_rows > row) continue;
            const unsigned long long q_base = (row * q_heads + q_head) * 256ull;
            reduce[tid] = q[q_base + tid] * cached_k[tid];
            __syncthreads();
            for (unsigned int stride = 128u; stride > 0u; stride >>= 1u) {
                if (tid < stride) reduce[tid] += reduce[tid + stride]; __syncthreads();
            }
            if (tid == 0u) {
                const float score = reduce[0] * softmax_scale; const float old_max = row_max[row];
                const float new_max = score > old_max ? score : old_max;
                update_scale = old_max <= -3.0e38f ? 0.0f : expf(old_max - new_max);
                update_weight = expf(score - new_max); row_max[row] = new_max;
                row_denom[row] = row_denom[row] * update_scale + update_weight;
            }
            __syncthreads(); weighted[row] = weighted[row] * update_scale + update_weight * cached_v[tid]; __syncthreads();
        }
    }
    const unsigned long long stride = 258ull;
    for (unsigned long long row = 0; row < query_rows; ++row) {
        const unsigned long long base = ((row * q_heads + q_head) * split_count + split_id) * stride;
        if (tid == 0u) { workspace[base] = row_max[row]; workspace[base + 1ull] = row_denom[row]; }
        workspace[base + 2ull + tid] = weighted[row];
    }
}

extern "C" __global__ void ullm_gemma_sliding_attn_ring_batched_256_split_merge_f32_kernel(
    unsigned long long query_rows, unsigned long long q_heads, unsigned long long split_count,
    const float *workspace, float *output) {
    const unsigned int tid = threadIdx.x; const unsigned long long q_head = blockIdx.x;
    if (q_head >= q_heads || blockDim.x != 256u || query_rows == 0ull || query_rows > 128ull) return;
    const unsigned long long stride = 258ull;
    for (unsigned long long row = 0; row < query_rows; ++row) {
        float merged_max = -3.4028234663852886e38f, merged_denom = 0.0f, numerator = 0.0f;
        for (unsigned long long split = 0; split < split_count; ++split) {
            const unsigned long long base = ((row * q_heads + q_head) * split_count + split) * stride;
            const float denom = workspace[base + 1ull]; if (denom == 0.0f) continue;
            const float partial_max = workspace[base]; const float new_max = partial_max > merged_max ? partial_max : merged_max;
            const float alpha = merged_denom == 0.0f ? 0.0f : expf(merged_max - new_max);
            const float beta = expf(partial_max - new_max);
            numerator = numerator * alpha + workspace[base + 2ull + tid] * beta;
            merged_denom = merged_denom * alpha + denom * beta; merged_max = new_max;
        }
        output[(row * q_heads + q_head) * 256ull + tid] = numerator / merged_denom;
    }
}
)";
    }
