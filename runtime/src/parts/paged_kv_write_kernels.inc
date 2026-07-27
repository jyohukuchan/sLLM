// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
class HipPagedKvWriteKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        LoadedModule *loaded = ensure_device_locked(device_id, error);
        return loaded == nullptr ? nullptr : loaded->function;
    }

    void *typed_function_for_device(
        int device_id,
        ullm_kv_cache_dtype k_dtype,
        ullm_kv_cache_dtype v_dtype,
        std::string *error) {
        const int k_index = static_cast<int>(k_dtype);
        const int v_index = static_cast<int>(v_dtype);
        if (k_index < 0 || k_index > 2 || v_index < 0 || v_index > 2) {
            append_error(error, "typed paged KV write received an unsupported cache dtype");
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        LoadedModule *loaded = ensure_device_locked(device_id, error);
        if (loaded == nullptr) {
            return nullptr;
        }
        // F32/F32 is intentionally the existing legacy symbol.  The typed
        // ABI may ask for it, but it must never replace the F32 fast path.
        return k_index == 0 && v_index == 0 ? loaded->function : loaded->typed[k_index][v_index];
    }

private:
    struct LoadedModule {
        void *module = nullptr;
        void *function = nullptr;
        void *typed[3][3] = {};
        std::string arch;
    };

    static const char *typed_symbol_name(int k_dtype, int v_dtype) {
        static const char *const names[3][3] = {
            {nullptr,
             "ullm_paged_kv_write_typed_kf32_vf16_kernel",
             "ullm_paged_kv_write_typed_kf32_vfp8_kernel"},
            {"ullm_paged_kv_write_typed_kf16_vf32_kernel",
             "ullm_paged_kv_write_typed_kf16_vf16_kernel",
             "ullm_paged_kv_write_typed_kf16_vfp8_kernel"},
            {"ullm_paged_kv_write_typed_kfp8_vf32_kernel",
             "ullm_paged_kv_write_typed_kfp8_vf16_kernel",
             "ullm_paged_kv_write_typed_kfp8_vfp8_kernel"},
        };
        return names[k_dtype][v_dtype];
    }

    LoadedModule *ensure_device_locked(int device_id, std::string *error) {
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second.get();
        }

        const std::vector<std::string> candidates = hip_arch_candidates(device_id);
        if (candidates.empty()) {
            append_error(error, "unable to infer HIP offload architecture for device");
            return nullptr;
        }

        std::string compile_errors;
        for (const std::string &arch : candidates) {
            std::vector<char> code;
            std::string compile_error;
            if (!hiprtc_runtime().compile_paged_kv_write_kernel(arch, &code, &compile_error)) {
                append_error(&compile_errors, compile_error);
                continue;
            }

            void *module = nullptr;
            if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
                append_error(&compile_errors, "hipModuleLoadData failed for " + arch);
                continue;
            }
            void *function = nullptr;
            if (!hip_runtime().module_get_function(
                    &function,
                    module,
                    "ullm_paged_kv_write_f32_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction failed for " + arch);
                continue;
            }

            auto loaded = std::make_unique<LoadedModule>();
            loaded->module = module;
            loaded->function = function;
            bool typed_loaded = true;
            for (int k_dtype = 0; k_dtype < 3 && typed_loaded; ++k_dtype) {
                for (int v_dtype = 0; v_dtype < 3; ++v_dtype) {
                    if (k_dtype == 0 && v_dtype == 0) {
                        continue;
                    }
                    const char *symbol = typed_symbol_name(k_dtype, v_dtype);
                    if (!hip_runtime().module_get_function(
                            &loaded->typed[k_dtype][v_dtype], module, symbol, device_id)) {
                        typed_loaded = false;
                        break;
                    }
                }
            }
            if (!typed_loaded) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction failed for typed paged KV write kernels");
                continue;
            }
            loaded->arch = arch;
            LoadedModule *result = loaded.get();
            modules_.emplace(device_id, std::move(loaded));
            return result;
        }
        append_error(
            error,
            compile_errors.empty() ? "failed to build paged KV write HIP kernel" : compile_errors);
        return nullptr;
    }

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

HipPagedKvWriteKernelCache &hip_paged_kv_write_kernel_cache() {
    static HipPagedKvWriteKernelCache cache;
    return cache;
}

bool paged_kv_write_f32_hip_kernel(
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_position,
    size_t block_size,
    size_t cache_blocks,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = k_buffer->hip_device_id;
    void *function = hip_paged_kv_write_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int launch_block_size = 256;
    const size_t elements = kv_heads * head_dim + kv_heads * value_dim;
    const size_t grid_size = (elements + launch_block_size - 1) / launch_block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "paged KV write element count exceeds HIP grid limit";
        }
        return false;
    }

    unsigned long long kernel_cache_position = static_cast<unsigned long long>(cache_position);
    unsigned long long kernel_block_size = static_cast<unsigned long long>(block_size);
    unsigned long long kernel_cache_blocks = static_cast<unsigned long long>(cache_blocks);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *k_ptr = k_buffer->ptr;
    void *v_ptr = v_buffer->ptr;
    void *block_table_ptr = block_table_buffer->ptr;
    void *k_cache_ptr = k_cache_buffer->ptr;
    void *v_cache_ptr = v_cache_buffer->ptr;
    void *kernel_params[] = {
        &k_ptr,
        &v_ptr,
        &block_table_ptr,
        &kernel_cache_position,
        &kernel_block_size,
        &kernel_cache_blocks,
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
            *error = "hipModuleLaunchKernel failed for f32 paged KV write";
        }
        return false;
    }
    return true;
}

bool paged_kv_write_typed_f32_hip_kernel(
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_position,
    size_t block_size,
    size_t cache_blocks,
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
        return paged_kv_write_f32_hip_kernel(
            k_buffer,
            v_buffer,
            block_table_buffer,
            cache_position,
            block_size,
            cache_blocks,
            kv_heads,
            head_dim,
            value_dim,
            k_cache_buffer,
            v_cache_buffer,
            stream,
            error);
    }
    const int device_id = k_buffer->hip_device_id;
    void *function = hip_paged_kv_write_kernel_cache().typed_function_for_device(
        device_id, k_dtype, v_dtype, error);
    if (function == nullptr) {
        return false;
    }
    constexpr unsigned int launch_block_size = 256;
    if (kv_heads > static_cast<size_t>(std::numeric_limits<unsigned int>::max() / 2u)) {
        if (error != nullptr) {
            *error = "typed paged KV write head count exceeds HIP grid limit";
        }
        return false;
    }
    const unsigned int grid_size = static_cast<unsigned int>(kv_heads * 2u);
    unsigned long long kernel_cache_position = static_cast<unsigned long long>(cache_position);
    unsigned long long kernel_block_size = static_cast<unsigned long long>(block_size);
    unsigned long long kernel_cache_blocks = static_cast<unsigned long long>(cache_blocks);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *k_ptr = k_buffer->ptr;
    void *v_ptr = v_buffer->ptr;
    void *block_table_ptr = block_table_buffer->ptr;
    void *k_cache_ptr = k_cache_buffer->ptr;
    void *v_cache_ptr = v_cache_buffer->ptr;
    void *k_scale_ptr = k_scale_buffer == nullptr ? nullptr : k_scale_buffer->ptr;
    void *v_scale_ptr = v_scale_buffer == nullptr ? nullptr : v_scale_buffer->ptr;
    void *kernel_params[] = {
        &k_ptr,
        &v_ptr,
        &block_table_ptr,
        &kernel_cache_position,
        &kernel_block_size,
        &kernel_cache_blocks,
        &kernel_kv_heads,
        &kernel_head_dim,
        &kernel_value_dim,
        &k_cache_ptr,
        &v_cache_ptr,
        &k_scale_ptr,
        &v_scale_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            grid_size,
            launch_block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for typed paged KV write";
        }
        return false;
    }
    return true;
}

ullm_status paged_kv_write_f32_hip_staging(
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_position,
    size_t block_size,
    size_t cache_blocks,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    size_t k_bytes,
    size_t v_bytes,
    size_t block_table_bytes,
    size_t k_cache_bytes,
    size_t v_cache_bytes,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_k(k_bytes / sizeof(float));
    std::vector<float> host_v(v_bytes / sizeof(float));
    std::vector<std::uint32_t> host_block_table(block_table_bytes / sizeof(std::uint32_t));
    std::vector<float> host_k_cache(k_cache_bytes / sizeof(float));
    std::vector<float> host_v_cache(v_cache_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = k_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_k.data(),
            k_buffer->ptr,
            k_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_v.data(),
            v_buffer->ptr,
            v_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_block_table.data(),
            block_table_buffer->ptr,
            block_table_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_k_cache.data(),
            k_cache_buffer->ptr,
            k_cache_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_v_cache.data(),
            v_cache_buffer->ptr,
            v_cache_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 paged KV write HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 paged KV write HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    const size_t block_index = cache_position / block_size;
    const std::uint32_t block_id = host_block_table[block_index];
    if (static_cast<size_t>(block_id) >= cache_blocks) {
        set_error("f32 paged KV write block table contains an out-of-range block id");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    paged_kv_write_f32_host(
        host_k.data(),
        host_v.data(),
        host_block_table.data(),
        cache_position,
        block_size,
        kv_heads,
        head_dim,
        value_dim,
        host_k_cache.data(),
        host_v_cache.data());

    if (!hip_runtime().copy_async(
            k_cache_buffer->ptr,
            host_k_cache.data(),
            k_cache_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            v_cache_buffer->ptr,
            host_v_cache.data(),
            v_cache_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 paged KV write caches back to HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 paged KV write HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

