// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
class HipPagedCausalGqaChunkWmmaKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second->function;
        }

        // The production guard is exact gfx1201. Do not silently fall back to another
        // gfx12 code object if this one fails to compile or load on the target R9700.
        const std::string arch = "gfx1201";
        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_paged_causal_gqa_chunk_wmma_kernel(
                arch,
                &code,
                &compile_error)) {
            append_error(
                error,
                compile_error.empty() ?
                    "failed to build paged causal GQA WMMA HIP kernel" : compile_error);
            return nullptr;
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            append_error(error, "hipModuleLoadData failed for " + arch);
            return nullptr;
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function,
                module,
                "ullm_paged_causal_gqa_chunk_wmma_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            append_error(error, "hipModuleGetFunction failed for " + arch);
            return nullptr;
        }
        auto loaded = std::make_unique<LoadedModule>();
        loaded->module = module;
        loaded->function = function;
        loaded->arch = arch;
        void *result = loaded->function;
        modules_.emplace(device_id, std::move(loaded));
        return result;
    }

private:
    struct LoadedModule {
        void *module = nullptr;
        void *function = nullptr;
        std::string arch;
    };

    static void append_error(std::string *error, const std::string &message) {
        if (error == nullptr || message.empty()) {
            return;
        }
        if (!error->empty()) {
            error->append("\n");
        }
        error->append(message);
    }

    std::mutex mutex_;
    std::unordered_map<int, std::unique_ptr<LoadedModule>> modules_;
};

HipPagedCausalGqaChunkWmmaKernelCache &hip_paged_causal_gqa_chunk_wmma_kernel_cache() {
    static HipPagedCausalGqaChunkWmmaKernelCache cache;
    return cache;
}

bool paged_causal_gqa_chunk_wmma_shape_is_supported(
    const std::string &arch,
    size_t block_size,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim) {
    return arch == "gfx1201" && block_size == 256u && q_heads == 16u && kv_heads == 4u &&
        head_dim == 256u && value_dim == 256u;
}

bool paged_kv_write_chunk_f32_hip_kernel(
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_start,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t block_table_entries,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = k_buffer->hip_device_id;
    void *function = hip_paged_chunk_kernel_cache().writer_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }
    const size_t k_row = kv_heads * head_dim;
    const size_t v_row = kv_heads * value_dim;
    const size_t elements = m * (k_row + v_row);
    constexpr unsigned int launch_block_size = 256;
    const size_t grid_size = (elements + launch_block_size - 1) / launch_block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "paged KV write chunk element count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_cache_start = static_cast<unsigned long long>(cache_start);
    unsigned long long kernel_m = static_cast<unsigned long long>(m);
    unsigned long long kernel_block_size = static_cast<unsigned long long>(block_size);
    unsigned long long kernel_cache_blocks = static_cast<unsigned long long>(cache_blocks);
    unsigned long long kernel_table_entries = static_cast<unsigned long long>(block_table_entries);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *k_ptr = k_buffer->ptr;
    void *v_ptr = v_buffer->ptr;
    void *table_ptr = block_table_buffer->ptr;
    void *k_cache_ptr = k_cache_buffer->ptr;
    void *v_cache_ptr = v_cache_buffer->ptr;
    void *kernel_params[] = {
        &k_ptr,
        &v_ptr,
        &table_ptr,
        &kernel_cache_start,
        &kernel_m,
        &kernel_block_size,
        &kernel_cache_blocks,
        &kernel_table_entries,
        &kernel_kv_heads,
        &kernel_head_dim,
        &kernel_value_dim,
        &k_cache_ptr,
        &v_cache_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 paged KV write chunk";
        }
        return false;
    }
    return true;
}

bool paged_kv_write_chunk_typed_f32_hip_kernel(
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_start,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t block_table_entries,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    ullm_kv_cache_dtype k_dtype,
    ullm_kv_cache_dtype v_dtype,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_buffer *k_scale_buffer,
    ullm_runtime_buffer *v_scale_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    if (k_dtype == ULLM_KV_CACHE_DTYPE_F32 && v_dtype == ULLM_KV_CACHE_DTYPE_F32) {
        return paged_kv_write_chunk_f32_hip_kernel(
            k_buffer, v_buffer, block_table_buffer, cache_start, m, block_size, cache_blocks,
            block_table_entries, kv_heads, head_dim, value_dim, k_cache_buffer, v_cache_buffer,
            stream, error);
    }
    const int device_id = k_buffer->hip_device_id;
    void *function = hip_paged_chunk_kernel_cache().typed_writer_for_device(
        device_id, k_dtype, v_dtype, error);
    if (function == nullptr) return false;
    constexpr unsigned int launch_block_size = 256;
    const size_t rows = m * kv_heads * 2u;
    if (rows > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) *error = "typed paged KV write chunk grid exceeds HIP grid limit";
        return false;
    }
    unsigned long long kernel_cache_start = static_cast<unsigned long long>(cache_start);
    unsigned long long kernel_m = static_cast<unsigned long long>(m);
    unsigned long long kernel_block_size = static_cast<unsigned long long>(block_size);
    unsigned long long kernel_cache_blocks = static_cast<unsigned long long>(cache_blocks);
    unsigned long long kernel_table_entries = static_cast<unsigned long long>(block_table_entries);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *k_ptr = k_buffer->ptr;
    void *v_ptr = v_buffer->ptr;
    void *table_ptr = block_table_buffer->ptr;
    void *k_cache_ptr = k_cache_buffer->ptr;
    void *v_cache_ptr = v_cache_buffer->ptr;
    void *k_scale_ptr = k_scale_buffer == nullptr ? nullptr : k_scale_buffer->ptr;
    void *v_scale_ptr = v_scale_buffer == nullptr ? nullptr : v_scale_buffer->ptr;
    void *kernel_params[] = {
        &k_ptr, &v_ptr, &table_ptr, &kernel_cache_start, &kernel_m,
        &kernel_block_size, &kernel_cache_blocks, &kernel_table_entries,
        &kernel_kv_heads, &kernel_head_dim, &kernel_value_dim, &k_cache_ptr,
        &v_cache_ptr, &k_scale_ptr, &v_scale_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function, static_cast<unsigned int>(rows), launch_block_size,
            kernel_params, hip_stream, device_id)) {
        if (error != nullptr) *error = "hipModuleLaunchKernel failed for typed paged KV write chunk";
        return false;
    }
    return true;
}

bool paged_causal_gqa_chunk_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t block_table_entries,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    void *function = hip_paged_chunk_kernel_cache().reader_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }
    constexpr unsigned int launch_block_size = 256;
    const size_t grid_size = m * q_heads;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "paged causal GQA chunk element count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_cached_prefix_len = static_cast<unsigned long long>(cached_prefix_len);
    unsigned long long kernel_m = static_cast<unsigned long long>(m);
    unsigned long long kernel_block_size = static_cast<unsigned long long>(block_size);
    unsigned long long kernel_cache_blocks = static_cast<unsigned long long>(cache_blocks);
    unsigned long long kernel_table_entries = static_cast<unsigned long long>(block_table_entries);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *q_ptr = q_buffer->ptr;
    void *gate_ptr = gate_buffer == nullptr ? nullptr : gate_buffer->ptr;
    void *k_ptr = k_cache_buffer->ptr;
    void *v_ptr = v_cache_buffer->ptr;
    void *table_ptr = block_table_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr,
        &gate_ptr,
        &k_ptr,
        &v_ptr,
        &table_ptr,
        &kernel_cached_prefix_len,
        &kernel_m,
        &kernel_block_size,
        &kernel_cache_blocks,
        &kernel_table_entries,
        &kernel_q_heads,
        &kernel_kv_heads,
        &kernel_head_dim,
        &kernel_value_dim,
        &softmax_scale,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 paged causal GQA chunk";
        }
        return false;
    }
    return true;
}

bool paged_causal_gqa_chunk_typed_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    const ullm_runtime_buffer *k_scale_buffer,
    const ullm_runtime_buffer *v_scale_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t block_table_entries,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_kv_cache_dtype k_dtype,
    ullm_kv_cache_dtype v_dtype,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    if (k_dtype == ULLM_KV_CACHE_DTYPE_F32 && v_dtype == ULLM_KV_CACHE_DTYPE_F32) {
        return paged_causal_gqa_chunk_f32_hip_kernel(
            q_buffer, gate_buffer, k_cache_buffer, v_cache_buffer, block_table_buffer,
            cached_prefix_len, m, block_size, cache_blocks, block_table_entries, q_heads,
            kv_heads, head_dim, value_dim, softmax_scale, output_buffer, stream, error);
    }
    const int device_id = q_buffer->hip_device_id;
    void *function = hip_paged_chunk_kernel_cache().typed_reader_for_device(
        device_id, k_dtype, v_dtype, error);
    if (function == nullptr) return false;
    constexpr unsigned int launch_block_size = 256;
    const size_t grid_size = m * q_heads;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) *error = "typed paged causal GQA chunk grid exceeds HIP grid limit";
        return false;
    }
    unsigned long long kernel_cached_prefix_len = static_cast<unsigned long long>(cached_prefix_len);
    unsigned long long kernel_m = static_cast<unsigned long long>(m);
    unsigned long long kernel_block_size = static_cast<unsigned long long>(block_size);
    unsigned long long kernel_cache_blocks = static_cast<unsigned long long>(cache_blocks);
    unsigned long long kernel_table_entries = static_cast<unsigned long long>(block_table_entries);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *q_ptr = q_buffer->ptr;
    void *gate_ptr = gate_buffer == nullptr ? nullptr : gate_buffer->ptr;
    void *k_ptr = k_cache_buffer->ptr;
    void *v_ptr = v_cache_buffer->ptr;
    void *table_ptr = block_table_buffer->ptr;
    void *k_scale_ptr = k_scale_buffer == nullptr ? nullptr : k_scale_buffer->ptr;
    void *v_scale_ptr = v_scale_buffer == nullptr ? nullptr : v_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr, &gate_ptr, &k_ptr, &v_ptr, &table_ptr, &k_scale_ptr, &v_scale_ptr,
        &kernel_cached_prefix_len, &kernel_m, &kernel_block_size, &kernel_cache_blocks,
        &kernel_table_entries, &kernel_q_heads, &kernel_kv_heads, &kernel_head_dim,
        &kernel_value_dim, &softmax_scale, &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function, static_cast<unsigned int>(grid_size), launch_block_size,
            kernel_params, hip_stream, device_id)) {
        if (error != nullptr) *error = "hipModuleLaunchKernel failed for typed paged causal GQA chunk";
        return false;
    }
    return true;
}

bool paged_causal_gqa_chunk_wmma_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t block_table_entries,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    if (!paged_causal_gqa_chunk_wmma_shape_is_supported(
            hip_runtime().device_arch_name(device_id),
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            value_dim)) {
        if (error != nullptr) {
            *error =
                "paged causal GQA WMMA requires gfx1201, block=256, Q=16, KV=4, and head/value dim=256";
        }
        return false;
    }
    if (m == 0u || m > 128u) {
        if (error != nullptr) {
            *error = "paged causal GQA WMMA requires M in 1..=128";
        }
        return false;
    }
    const size_t grid_x = (m + 15u) / 16u;
    const size_t grid_y = q_heads;
    if (grid_x == 0u || grid_y == 0u ||
        grid_x > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        grid_y > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "paged causal GQA WMMA dimensions exceed HIP grid limit";
        }
        return false;
    }
    void *function = hip_paged_causal_gqa_chunk_wmma_kernel_cache().function_for_device(
        device_id,
        error);
    if (function == nullptr) {
        return false;
    }

    unsigned long long kernel_cached_prefix_len = static_cast<unsigned long long>(cached_prefix_len);
    unsigned long long kernel_m = static_cast<unsigned long long>(m);
    unsigned long long kernel_block_size = static_cast<unsigned long long>(block_size);
    unsigned long long kernel_cache_blocks = static_cast<unsigned long long>(cache_blocks);
    unsigned long long kernel_table_entries = static_cast<unsigned long long>(block_table_entries);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *q_ptr = q_buffer->ptr;
    void *gate_ptr = gate_buffer == nullptr ? nullptr : gate_buffer->ptr;
    void *k_ptr = k_cache_buffer->ptr;
    void *v_ptr = v_cache_buffer->ptr;
    void *table_ptr = block_table_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr,
        &gate_ptr,
        &k_ptr,
        &v_ptr,
        &table_ptr,
        &kernel_cached_prefix_len,
        &kernel_m,
        &kernel_block_size,
        &kernel_cache_blocks,
        &kernel_table_entries,
        &kernel_q_heads,
        &kernel_kv_heads,
        &kernel_head_dim,
        &kernel_value_dim,
        &softmax_scale,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel_2d(
            function,
            static_cast<unsigned int>(grid_x),
            static_cast<unsigned int>(grid_y),
            256u,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for paged causal GQA WMMA";
        }
        return false;
    }
    return true;
}

ullm_status validate_paged_chunk_block_table_hip(
    const ullm_runtime_buffer *block_table_buffer,
    size_t table_entries,
    size_t cache_blocks,
    ullm_runtime_stream *stream,
    const char *operation) {
    if (block_table_buffer == nullptr || table_entries == 0 || cache_blocks == 0) {
        set_error("f32 paged chunk HIP block table validation received invalid geometry");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (table_entries > std::numeric_limits<size_t>::max() / sizeof(std::uint32_t)) {
        set_error("f32 paged chunk HIP block table validation byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    std::vector<std::uint32_t> host_table;
    try {
        host_table.resize(table_entries);
    } catch (const std::bad_alloc &) {
        set_error("f32 paged chunk HIP block table validation ran out of host memory");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = block_table_buffer->hip_device_id;
    const size_t table_bytes = table_entries * sizeof(std::uint32_t);
    if (!hip_runtime().copy_async(
            host_table.data(),
            block_table_buffer->ptr,
            table_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("f32 paged chunk HIP block table D2H validation copy failed");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("f32 paged chunk HIP block table validation synchronization failed");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    for (size_t index = 0; index < host_table.size(); ++index) {
        if (static_cast<size_t>(host_table[index]) >= cache_blocks) {
            std::string message = operation == nullptr ? "f32 paged chunk" : operation;
            message += " block table contains an out-of-range block id";
            set_error(message.c_str());
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
    }
    return ULLM_STATUS_OK;
}

void depthwise_conv1d_f32_host(
    const float *input,
    const float *weight,
    size_t channels,
    size_t sequence_len,
    size_t kernel_size,
    float *output) {
    for (size_t timestep = 0; timestep < sequence_len; ++timestep) {
        for (size_t channel = 0; channel < channels; ++channel) {
            float sum = 0.0f;
            for (size_t kernel = 0; kernel < kernel_size; ++kernel) {
                const size_t left_padding = kernel_size - 1 - kernel;
                if (timestep < left_padding) {
                    continue;
                }
                const size_t source_timestep = timestep - left_padding;
                sum += input[source_timestep * channels + channel] *
                       weight[channel * kernel_size + kernel];
            }
            output[timestep * channels + channel] = sum;
        }
    }
}

