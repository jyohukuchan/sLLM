
__device__ __forceinline__ float ullm_sq8_0_reduce_sum_tree(float value, float* partial) {
    const unsigned int tid = threadIdx.x;
    partial[tid] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    return partial[0];
}

__device__ __forceinline__ float ullm_sq8_0_reduce_max_tree(float value, float* partial) {
    const unsigned int tid = threadIdx.x;
    partial[tid] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] = partial[tid] > partial[tid + stride] ? partial[tid] : partial[tid + stride];
        }
        __syncthreads();
    }
    return partial[0];
}

__device__ __forceinline__ float ullm_sq8_0_reduce_sum_wave32(float value, float* partial) {
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

__device__ __forceinline__ float ullm_sq8_0_reduce_max_wave32(float value, float* partial) {
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

template <int Stage>
__device__ __forceinline__ float ullm_sq8_0_reduce_sum(float value, float* partial) {
    if constexpr (Stage == 3) {
        return ullm_sq8_0_reduce_sum_wave32(value, partial);
    }
    return ullm_sq8_0_reduce_sum_tree(value, partial);
}

template <int Stage>
__device__ __forceinline__ float ullm_sq8_0_reduce_qk(float value, float* partial) {
    if constexpr (Stage >= 1) {
        return ullm_sq8_0_reduce_sum_wave32(value, partial);
    }
    return ullm_sq8_0_reduce_sum_tree(value, partial);
}

template <int Stage>
__device__ __forceinline__ float ullm_sq8_0_reduce_max(float value, float* partial) {
    if constexpr (Stage >= 2) {
        return ullm_sq8_0_reduce_max_wave32(value, partial);
    }
    return ullm_sq8_0_reduce_max_tree(value, partial);
}

template <int Stage>
__device__ void ullm_sq8_0_flash2_body(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    unsigned long long cached_prefix_len,
    unsigned long long new_tokens,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float* output) {
    constexpr unsigned int kTileTokens = 64u;
    const unsigned long long q_head_index = (unsigned long long) blockIdx.x;
    const unsigned long long q_head_elements = new_tokens * q_heads;
    if (q_head_index >= q_head_elements) return;
    const unsigned int tid = threadIdx.x;
    if (value_dim > (unsigned long long) blockDim.x) return;
    const unsigned long long q_head = q_head_index % q_heads;
    const unsigned long long token_index = q_head_index / q_heads;
    const unsigned long long cache_len = cached_prefix_len + token_index + 1ull;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_base = (token_index * q_heads + q_head) * head_dim;
    const unsigned long long value = (unsigned long long) tid;
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
        const unsigned int tile_count = remaining < (unsigned long long) kTileTokens
            ? (unsigned int) remaining : kTileTokens;
        for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
            const unsigned long long source_timestep = tile_start + (unsigned long long) tile_offset;
            const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
                partial += q[q_base + dim] * k_cache[k_base + dim];
            }
            const float score = ullm_sq8_0_reduce_qk<Stage>(partial, reduce) * softmax_scale;
            if (tid == 0u) scores[tile_offset] = score;
            __syncthreads();
        }

        float local_max = -3.4028234663852886e38f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float score = scores[tile_offset];
            local_max = score > local_max ? score : local_max;
        }
        const float tile_max = ullm_sq8_0_reduce_max<Stage>(local_max, reduce);
        if (tid == 0u) {
            shared_tile_max = tile_max;
            const float m_new = max_score > tile_max ? max_score : tile_max;
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
        const float tile_sum = ullm_sq8_0_reduce_sum<Stage>(local_sum, reduce);
        if (tid == 0u) shared_tile_sum = tile_sum;
        __syncthreads();

        if (value_active) {
            float tile_weighted = 0.0f;
            for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
                const unsigned long long source_timestep = tile_start + (unsigned long long) tile_offset;
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
    if (value_active) output[q_head_index * value_dim + value] = weighted / denominator;
}

extern "C" __global__ void ullm_sq8_0_flash2_legacy_reference_kernel(
    const float* q, const float* k, const float* v, unsigned long long cached_prefix_len,
    unsigned long long new_tokens, unsigned long long q_heads, unsigned long long kv_heads,
    unsigned long long head_dim, unsigned long long value_dim, float softmax_scale, float* output) {
    ullm_sq8_0_flash2_body<0>(q, k, v, cached_prefix_len, new_tokens, q_heads, kv_heads,
                               head_dim, value_dim, softmax_scale, output);
}

extern "C" __global__ void ullm_sq8_0_flash2_qk_wave32_prototype_kernel(
    const float* q, const float* k, const float* v, unsigned long long cached_prefix_len,
    unsigned long long new_tokens, unsigned long long q_heads, unsigned long long kv_heads,
    unsigned long long head_dim, unsigned long long value_dim, float softmax_scale, float* output) {
    ullm_sq8_0_flash2_body<1>(q, k, v, cached_prefix_len, new_tokens, q_heads, kv_heads,
                               head_dim, value_dim, softmax_scale, output);
}

extern "C" __global__ void ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel(
    const float* q, const float* k, const float* v, unsigned long long cached_prefix_len,
    unsigned long long new_tokens, unsigned long long q_heads, unsigned long long kv_heads,
    unsigned long long head_dim, unsigned long long value_dim, float softmax_scale, float* output) {
    ullm_sq8_0_flash2_body<2>(q, k, v, cached_prefix_len, new_tokens, q_heads, kv_heads,
                               head_dim, value_dim, softmax_scale, output);
}

extern "C" __global__ void ullm_sq8_0_flash2_staged_wave32_prototype_kernel(
    const float* q, const float* k, const float* v, unsigned long long cached_prefix_len,
    unsigned long long new_tokens, unsigned long long q_heads, unsigned long long kv_heads,
    unsigned long long head_dim, unsigned long long value_dim, float softmax_scale, float* output) {
    ullm_sq8_0_flash2_body<3>(q, k, v, cached_prefix_len, new_tokens, q_heads, kv_heads,
                               head_dim, value_dim, softmax_scale, output);
}

extern "C" __global__ void ullm_sq8_0_pmc_probe_kernel(
    const float* input, float* output, unsigned long long count) {
    const unsigned long long tid = (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long) gridDim.x * blockDim.x;
    float value = 0.0f;
    for (unsigned long long index = tid; index < count; index += stride) {
        const float loaded = ((volatile const float*) input)[index];
        value = fmaf(value, 1.000123f, loaded);
        value = fmaf(value, 0.999877f, loaded * 0.5f);
        value = fmaf(value, 1.000031f, loaded * 0.25f);
        value = fmaf(value, 0.999969f, loaded * 0.125f);
    }
    if (tid < count) output[tid] = value;
}
