// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
class HipPagedDecodeAttnKernelCache {
public:
    struct Functions {
        void *legacy = nullptr;
        void *gemma_full_batched_512 = nullptr;
        void *gemma_sliding_ring_batched_256 = nullptr;
        void *gemma_full_batched_512_split_partial = nullptr;
        void *gemma_full_batched_512_split_merge = nullptr;
        void *gemma_sliding_ring_batched_256_split_partial = nullptr;
        void *gemma_sliding_ring_batched_256_split_merge = nullptr;
        void *split_partial = nullptr;
        void *split_merge = nullptr;
    };

    void *function_for_device(int device_id, std::string *error) {
        Functions functions;
        if (!functions_for_device(device_id, &functions, error)) {
            return nullptr;
        }
        return functions.legacy;
    }

    void *gemma_full_batched_512_function_for_device(int device_id, std::string *error) {
        Functions functions;
        if (!functions_for_device(device_id, &functions, error)) {
            return nullptr;
        }
        return functions.gemma_full_batched_512;
    }

    void *gemma_sliding_ring_batched_256_function_for_device(int device_id, std::string *error) {
        Functions functions;
        if (!functions_for_device(device_id, &functions, error)) return nullptr;
        return functions.gemma_sliding_ring_batched_256;
    }

    bool gemma_split_functions_for_device(int device_id, Functions *functions, std::string *error) {
        return functions_for_device(device_id, functions, error);
    }

    void *typed_function_for_device(
        int device_id,
        ullm_kv_cache_dtype k_dtype,
        ullm_kv_cache_dtype v_dtype,
        std::string *error) {
        const int k_index = static_cast<int>(k_dtype);
        const int v_index = static_cast<int>(v_dtype);
        if (k_index < 0 || k_index > 2 || v_index < 0 || v_index > 2) {
            append_error(error, "typed paged decode received an unsupported cache dtype");
            return nullptr;
        }
        Functions ignored;
        if (!functions_for_device(device_id, &ignored, error)) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = modules_.find(device_id);
        if (found == modules_.end()) {
            append_error(error, "typed paged decode HIP module disappeared after load");
            return nullptr;
        }
        // Keep F32/F32 bound to the unmodified legacy body.
        return k_index == 0 && v_index == 0 ? found->second->legacy_function
                                             : found->second->typed[k_index][v_index];
    }

    void *typed_split_partial_function_for_device(
        int device_id,
        ullm_kv_cache_dtype k_dtype,
        ullm_kv_cache_dtype v_dtype,
        std::string *error) {
        const int k_index = static_cast<int>(k_dtype);
        const int v_index = static_cast<int>(v_dtype);
        if (k_index < 0 || k_index > 2 || v_index < 0 || v_index > 2) {
            append_error(error, "typed paged decode split received an unsupported cache dtype");
            return nullptr;
        }
        Functions ignored;
        if (!functions_for_device(device_id, &ignored, error)) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = modules_.find(device_id);
        if (found == modules_.end()) {
            append_error(error, "typed paged decode split HIP module disappeared after load");
            return nullptr;
        }
        return k_index == 0 && v_index == 0 ? found->second->split_partial_function
                                             : found->second->typed_split[k_index][v_index];
    }

    bool functions_for_device(int device_id, Functions *functions, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            assign_functions(*found->second, functions);
            return true;
        }

        const std::vector<std::string> candidates = hip_arch_candidates(device_id);
        if (candidates.empty()) {
            append_error(error, "unable to infer HIP offload architecture for device");
            return false;
        }

        std::string compile_errors;
        for (const std::string &arch : candidates) {
            std::vector<char> code;
            std::string compile_error;
            if (!hiprtc_runtime().compile_paged_decode_attn_kernel(arch, &code, &compile_error)) {
                append_error(&compile_errors, compile_error);
                continue;
            }

            void *module = nullptr;
            if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
                append_error(&compile_errors, "hipModuleLoadData failed for " + arch);
                continue;
            }
            void *legacy_function = nullptr;
            if (!hip_runtime().module_get_function(
                    &legacy_function,
                    module,
                    "ullm_paged_decode_attn_f32_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction failed for " + arch);
                continue;
            }
            void *split_partial_function = nullptr;
            void *gemma_full_batched_512_function = nullptr;
            if (!hip_runtime().module_get_function(
                    &gemma_full_batched_512_function,
                    module,
                    "ullm_gemma_full_attn_batched_512_f32_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction Gemma full batched attention failed for " + arch);
                continue;
            }
            void *gemma_sliding_ring_batched_256_function = nullptr;
            if (!hip_runtime().module_get_function(
                    &gemma_sliding_ring_batched_256_function,
                    module,
                    "ullm_gemma_sliding_attn_ring_batched_256_f32_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction Gemma sliding ring batched attention failed for " + arch);
                continue;
            }
            void *gemma_full_batched_512_split_partial_function = nullptr;
            void *gemma_full_batched_512_split_merge_function = nullptr;
            void *gemma_sliding_ring_batched_256_split_partial_function = nullptr;
            void *gemma_sliding_ring_batched_256_split_merge_function = nullptr;
            if (!hip_runtime().module_get_function(&gemma_full_batched_512_split_partial_function, module,
                    "ullm_gemma_full_attn_batched_512_split_partial_f32_kernel", device_id) ||
                !hip_runtime().module_get_function(&gemma_full_batched_512_split_merge_function, module,
                    "ullm_gemma_full_attn_batched_512_split_merge_f32_kernel", device_id) ||
                !hip_runtime().module_get_function(&gemma_sliding_ring_batched_256_split_partial_function, module,
                    "ullm_gemma_sliding_attn_ring_batched_256_split_partial_f32_kernel", device_id) ||
                !hip_runtime().module_get_function(&gemma_sliding_ring_batched_256_split_merge_function, module,
                    "ullm_gemma_sliding_attn_ring_batched_256_split_merge_f32_kernel", device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction Gemma split attention failed for " + arch);
                continue;
            }
            if (!hip_runtime().module_get_function(
                    &split_partial_function,
                    module,
                    "ullm_paged_decode_attn_split_partial_f32_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(
                    &compile_errors,
                    "hipModuleGetFunction split partial failed for " + arch);
                continue;
            }
            void *split_merge_function = nullptr;
            if (!hip_runtime().module_get_function(
                    &split_merge_function,
                    module,
                    "ullm_paged_decode_attn_split_merge_f32_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(
                    &compile_errors,
                    "hipModuleGetFunction split merge failed for " + arch);
                continue;
            }

            auto loaded = std::make_unique<LoadedModule>();
            loaded->module = module;
            loaded->legacy_function = legacy_function;
            loaded->gemma_full_batched_512_function = gemma_full_batched_512_function;
            loaded->gemma_sliding_ring_batched_256_function = gemma_sliding_ring_batched_256_function;
            loaded->gemma_full_batched_512_split_partial_function = gemma_full_batched_512_split_partial_function;
            loaded->gemma_full_batched_512_split_merge_function = gemma_full_batched_512_split_merge_function;
            loaded->gemma_sliding_ring_batched_256_split_partial_function = gemma_sliding_ring_batched_256_split_partial_function;
            loaded->gemma_sliding_ring_batched_256_split_merge_function = gemma_sliding_ring_batched_256_split_merge_function;
            loaded->split_partial_function = split_partial_function;
            loaded->split_merge_function = split_merge_function;
            bool typed_loaded = true;
            for (int k_dtype = 0; k_dtype < 3 && typed_loaded; ++k_dtype) {
                for (int v_dtype = 0; v_dtype < 3; ++v_dtype) {
                    if (k_dtype == 0 && v_dtype == 0) {
                        continue;
                    }
                    if (!hip_runtime().module_get_function(
                            &loaded->typed[k_dtype][v_dtype],
                            module,
                            typed_symbol_name(k_dtype, v_dtype),
                            device_id)) {
                        typed_loaded = false;
                        break;
                    }
                    if (!hip_runtime().module_get_function(
                            &loaded->typed_split[k_dtype][v_dtype],
                            module,
                            typed_split_symbol_name(k_dtype, v_dtype),
                            device_id)) {
                        typed_loaded = false;
                        break;
                    }
                }
            }
            if (!typed_loaded) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction typed paged decode failed for " + arch);
                continue;
            }
            loaded->arch = arch;
            assign_functions(*loaded, functions);
            modules_.emplace(device_id, std::move(loaded));
            return true;
        }
        append_error(
            error,
            compile_errors.empty() ? "failed to build paged decode attention HIP kernel" : compile_errors);
        return false;
    }

private:
    struct LoadedModule {
        void *module = nullptr;
        void *legacy_function = nullptr;
        void *gemma_full_batched_512_function = nullptr;
        void *gemma_sliding_ring_batched_256_function = nullptr;
        void *gemma_full_batched_512_split_partial_function = nullptr;
        void *gemma_full_batched_512_split_merge_function = nullptr;
        void *gemma_sliding_ring_batched_256_split_partial_function = nullptr;
        void *gemma_sliding_ring_batched_256_split_merge_function = nullptr;
        void *split_partial_function = nullptr;
        void *split_merge_function = nullptr;
        void *typed[3][3] = {};
        void *typed_split[3][3] = {};
        std::string arch;
    };

    static const char *typed_symbol_name(int k_dtype, int v_dtype) {
        static const char *const names[3][3] = {
            {nullptr,
             "ullm_paged_decode_attn_typed_kf32_vf16_kernel",
             "ullm_paged_decode_attn_typed_kf32_vfp8_kernel"},
            {"ullm_paged_decode_attn_typed_kf16_vf32_kernel",
             "ullm_paged_decode_attn_typed_kf16_vf16_kernel",
             "ullm_paged_decode_attn_typed_kf16_vfp8_kernel"},
            {"ullm_paged_decode_attn_typed_kfp8_vf32_kernel",
             "ullm_paged_decode_attn_typed_kfp8_vf16_kernel",
             "ullm_paged_decode_attn_typed_kfp8_vfp8_kernel"},
        };
        return names[k_dtype][v_dtype];
    }

    static const char *typed_split_symbol_name(int k_dtype, int v_dtype) {
        static const char *const names[3][3] = {
            {nullptr,
             "ullm_paged_decode_attn_split_typed_kf32_vf16_partial_kernel",
             "ullm_paged_decode_attn_split_typed_kf32_vfp8_partial_kernel"},
            {"ullm_paged_decode_attn_split_typed_kf16_vf32_partial_kernel",
             "ullm_paged_decode_attn_split_typed_kf16_vf16_partial_kernel",
             "ullm_paged_decode_attn_split_typed_kf16_vfp8_partial_kernel"},
            {"ullm_paged_decode_attn_split_typed_kfp8_vf32_partial_kernel",
             "ullm_paged_decode_attn_split_typed_kfp8_vf16_partial_kernel",
             "ullm_paged_decode_attn_split_typed_kfp8_vfp8_partial_kernel"},
        };
        return names[k_dtype][v_dtype];
    }

    static void assign_functions(const LoadedModule &loaded, Functions *functions) {
        if (functions == nullptr) {
            return;
        }
        functions->legacy = loaded.legacy_function;
        functions->gemma_full_batched_512 = loaded.gemma_full_batched_512_function;
        functions->gemma_sliding_ring_batched_256 = loaded.gemma_sliding_ring_batched_256_function;
        functions->gemma_full_batched_512_split_partial = loaded.gemma_full_batched_512_split_partial_function;
        functions->gemma_full_batched_512_split_merge = loaded.gemma_full_batched_512_split_merge_function;
        functions->gemma_sliding_ring_batched_256_split_partial = loaded.gemma_sliding_ring_batched_256_split_partial_function;
        functions->gemma_sliding_ring_batched_256_split_merge = loaded.gemma_sliding_ring_batched_256_split_merge_function;
        functions->split_partial = loaded.split_partial_function;
        functions->split_merge = loaded.split_merge_function;
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

HipPagedDecodeAttnKernelCache &hip_paged_decode_attn_kernel_cache() {
    static HipPagedDecodeAttnKernelCache cache;
    return cache;
}

bool paged_decode_attn_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    void *function = hip_paged_decode_attn_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int launch_block_size = 256;
    const size_t output_elements = q_heads * value_dim;
    // The exact 512-wide body below keeps the same one-CTA-per-Q-head
    // decomposition as the canonical <=256 path: every lane owns two Q/K and
    // V elements.  Do not send it to the scalar output-element fallback.
    const bool use_head_parallel_kernel =
        (head_dim <= launch_block_size && value_dim <= launch_block_size) ||
        (head_dim == 512u && value_dim == 512u);
    const size_t grid_size = use_head_parallel_kernel
                                 ? q_heads
                                 : (output_elements + launch_block_size - 1) / launch_block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "paged decode attention launch grid exceeds HIP grid limit";
        }
        return false;
    }

    unsigned long long kernel_cache_len = static_cast<unsigned long long>(cache_len);
    unsigned long long kernel_block_size = static_cast<unsigned long long>(block_size);
    unsigned long long kernel_cache_blocks = static_cast<unsigned long long>(cache_blocks);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *q_ptr = q_buffer->ptr;
    void *gate_ptr = gate_buffer == nullptr ? nullptr : gate_buffer->ptr;
    void *k_ptr = k_cache_buffer->ptr;
    void *v_ptr = v_cache_buffer->ptr;
    void *block_table_ptr = block_table_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr,
        &gate_ptr,
        &k_ptr,
        &v_ptr,
        &block_table_ptr,
        &kernel_cache_len,
        &kernel_block_size,
        &kernel_cache_blocks,
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
            *error = "hipModuleLaunchKernel failed for f32 paged decode attention";
        }
        return false;
    }
    return true;
}

bool gemma_full_attn_batched_512_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t prefix_len,
    size_t query_rows,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    void *function = hip_paged_decode_attn_kernel_cache()
        .gemma_full_batched_512_function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }
    unsigned long long kernel_prefix_len = static_cast<unsigned long long>(prefix_len);
    unsigned long long kernel_query_rows = static_cast<unsigned long long>(query_rows);
    unsigned long long kernel_cache_len = static_cast<unsigned long long>(cache_len);
    unsigned long long kernel_block_size = static_cast<unsigned long long>(block_size);
    unsigned long long kernel_cache_blocks = static_cast<unsigned long long>(cache_blocks);
    void *q_ptr = q_buffer->ptr;
    void *k_ptr = k_cache_buffer->ptr;
    void *v_ptr = v_cache_buffer->ptr;
    void *block_table_ptr = block_table_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr, &k_ptr, &v_ptr, &block_table_ptr, &kernel_prefix_len,
        &kernel_query_rows, &kernel_cache_len, &kernel_block_size, &kernel_cache_blocks,
        &softmax_scale, &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(function, 8u, 256u, kernel_params, hip_stream, device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for Gemma full batched attention";
        }
        return false;
    }
    return true;
}

bool gemma_sliding_attn_ring_batched_256_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_ring_buffer,
    const ullm_runtime_buffer *v_ring_buffer,
    const ullm_runtime_buffer *fresh_k_buffer,
    const ullm_runtime_buffer *fresh_v_buffer,
    size_t prefix_len, size_t history_rows, size_t query_rows,
    size_t ring_capacity, size_t q_heads, float softmax_scale,
    ullm_runtime_buffer *output_buffer, ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    void *function = hip_paged_decode_attn_kernel_cache()
        .gemma_sliding_ring_batched_256_function_for_device(device_id, error);
    if (function == nullptr) return false;
    unsigned long long kernel_prefix_len = static_cast<unsigned long long>(prefix_len);
    unsigned long long kernel_history_rows = static_cast<unsigned long long>(history_rows);
    unsigned long long kernel_query_rows = static_cast<unsigned long long>(query_rows);
    unsigned long long kernel_ring_capacity = static_cast<unsigned long long>(ring_capacity);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    void *q_ptr = q_buffer->ptr;
    void *k_ring_ptr = k_ring_buffer->ptr;
    void *v_ring_ptr = v_ring_buffer->ptr;
    void *fresh_k_ptr = fresh_k_buffer->ptr;
    void *fresh_v_ptr = fresh_v_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *params[] = {&q_ptr, &k_ring_ptr, &v_ring_ptr, &fresh_k_ptr, &fresh_v_ptr,
        &kernel_prefix_len, &kernel_history_rows, &kernel_query_rows, &kernel_ring_capacity,
        &kernel_q_heads, &softmax_scale, &output_ptr};
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(function, static_cast<unsigned int>(q_heads), 256u,
                                             params, hip_stream, device_id)) {
        if (error != nullptr) *error = "hipModuleLaunchKernel failed for Gemma sliding ring batched attention";
        return false;
    }
    return true;
}

bool gemma_full_attn_batched_512_split_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer, const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer, const ullm_runtime_buffer *block_table_buffer,
    size_t prefix_len, size_t query_rows, size_t cache_len, size_t block_size, size_t cache_blocks,
    float softmax_scale, size_t split_count, ullm_runtime_buffer *workspace_buffer,
    ullm_runtime_buffer *output_buffer, ullm_runtime_stream *stream, std::string *error) {
    HipPagedDecodeAttnKernelCache::Functions functions;
    const int device_id = q_buffer->hip_device_id;
    if (!hip_paged_decode_attn_kernel_cache().gemma_split_functions_for_device(device_id, &functions, error) ||
        functions.gemma_full_batched_512_split_partial == nullptr || functions.gemma_full_batched_512_split_merge == nullptr) return false;
    unsigned long long prefix = prefix_len, rows = query_rows, length = cache_len, block = block_size,
        blocks = cache_blocks, splits = split_count;
    void *q = q_buffer->ptr; void *k = k_cache_buffer->ptr; void *v = v_cache_buffer->ptr;
    void *table = block_table_buffer->ptr; void *workspace = workspace_buffer->ptr; void *output = output_buffer->ptr;
    void *partial_params[] = {&q, &k, &v, &table, &prefix, &rows, &length, &block, &blocks,
        &softmax_scale, &splits, &workspace};
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(functions.gemma_full_batched_512_split_partial,
            static_cast<unsigned int>(8u * split_count), 256u, partial_params, hip_stream, device_id)) {
        if (error != nullptr) *error = "hipModuleLaunchKernel failed for Gemma full split partial attention";
        return false;
    }
    void *merge_params[] = {&rows, &splits, &workspace, &output};
    if (!hip_runtime().module_launch_kernel(functions.gemma_full_batched_512_split_merge, 8u, 256u,
            merge_params, hip_stream, device_id)) {
        if (error != nullptr) *error = "hipModuleLaunchKernel failed for Gemma full split merge attention";
        return false;
    }
    return true;
}

bool gemma_sliding_attn_ring_batched_256_split_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer, const ullm_runtime_buffer *k_ring_buffer,
    const ullm_runtime_buffer *v_ring_buffer, const ullm_runtime_buffer *fresh_k_buffer,
    const ullm_runtime_buffer *fresh_v_buffer, size_t prefix_len, size_t history_rows,
    size_t query_rows, size_t ring_capacity, size_t q_heads, float softmax_scale,
    size_t split_count, ullm_runtime_buffer *workspace_buffer, ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream, std::string *error) {
    HipPagedDecodeAttnKernelCache::Functions functions;
    const int device_id = q_buffer->hip_device_id;
    if (!hip_paged_decode_attn_kernel_cache().gemma_split_functions_for_device(device_id, &functions, error) ||
        functions.gemma_sliding_ring_batched_256_split_partial == nullptr || functions.gemma_sliding_ring_batched_256_split_merge == nullptr) return false;
    unsigned long long prefix = prefix_len, history = history_rows, rows = query_rows, ring = ring_capacity,
        heads = q_heads, splits = split_count;
    void *q = q_buffer->ptr; void *kr = k_ring_buffer->ptr; void *vr = v_ring_buffer->ptr;
    void *fk = fresh_k_buffer->ptr; void *fv = fresh_v_buffer->ptr; void *workspace = workspace_buffer->ptr; void *output = output_buffer->ptr;
    void *partial_params[] = {&q, &kr, &vr, &fk, &fv, &prefix, &history, &rows, &ring, &heads,
        &softmax_scale, &splits, &workspace};
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(functions.gemma_sliding_ring_batched_256_split_partial,
            static_cast<unsigned int>(q_heads * split_count), 256u, partial_params, hip_stream, device_id)) {
        if (error != nullptr) *error = "hipModuleLaunchKernel failed for Gemma sliding split partial attention";
        return false;
    }
    void *merge_params[] = {&rows, &heads, &splits, &workspace, &output};
    if (!hip_runtime().module_launch_kernel(functions.gemma_sliding_ring_batched_256_split_merge,
            static_cast<unsigned int>(q_heads), 256u, merge_params, hip_stream, device_id)) {
        if (error != nullptr) *error = "hipModuleLaunchKernel failed for Gemma sliding split merge attention";
        return false;
    }
    return true;
}

bool paged_decode_attn_typed_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    const ullm_runtime_buffer *k_scale_buffer,
    const ullm_runtime_buffer *v_scale_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
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
        return paged_decode_attn_f32_hip_kernel(
            q_buffer,
            nullptr,
            k_cache_buffer,
            v_cache_buffer,
            block_table_buffer,
            cache_len,
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output_buffer,
            stream,
            error);
    }
    const int device_id = q_buffer->hip_device_id;
    void *function = hip_paged_decode_attn_kernel_cache().typed_function_for_device(
        device_id, k_dtype, v_dtype, error);
    if (function == nullptr) {
        return false;
    }
    constexpr unsigned int launch_block_size = 256;
    const size_t output_elements = q_heads * value_dim;
    const bool use_head_parallel_kernel =
        head_dim <= launch_block_size && value_dim <= launch_block_size;
    const size_t grid_size = use_head_parallel_kernel
                                 ? q_heads
                                 : (output_elements + launch_block_size - 1u) / launch_block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "typed paged decode attention launch grid exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_cache_len = static_cast<unsigned long long>(cache_len);
    unsigned long long kernel_block_size = static_cast<unsigned long long>(block_size);
    unsigned long long kernel_cache_blocks = static_cast<unsigned long long>(cache_blocks);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *q_ptr = q_buffer->ptr;
    void *k_ptr = k_cache_buffer->ptr;
    void *v_ptr = v_cache_buffer->ptr;
    void *block_table_ptr = block_table_buffer->ptr;
    void *k_scale_ptr = k_scale_buffer == nullptr ? nullptr : k_scale_buffer->ptr;
    void *v_scale_ptr = v_scale_buffer == nullptr ? nullptr : v_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr,
        &k_ptr,
        &v_ptr,
        &block_table_ptr,
        &k_scale_ptr,
        &v_scale_ptr,
        &kernel_cache_len,
        &kernel_block_size,
        &kernel_cache_blocks,
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
            *error = "hipModuleLaunchKernel failed for typed paged decode attention";
        }
        return false;
    }
    return true;
}

bool paged_decode_attn_split_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    size_t source_tile,
    size_t split_count,
    ullm_runtime_buffer *workspace_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    HipPagedDecodeAttnKernelCache::Functions functions;
    if (!hip_paged_decode_attn_kernel_cache().functions_for_device(
            device_id,
            &functions,
            error)) {
        return false;
    }
    if (functions.split_partial == nullptr || functions.split_merge == nullptr) {
        if (error != nullptr) {
            *error = "paged decode split attention HIP functions are unavailable";
        }
        return false;
    }

    constexpr unsigned int launch_block_size = 256;
    const char *grouped_env = std::getenv("ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT");
    const char *pipelined_env = std::getenv("ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_PIPELINED_SPLIT");
    const bool grouped_requested =
        (grouped_env != nullptr && grouped_env[0] == '1' && grouped_env[1] == '\0') ||
        (pipelined_env != nullptr && pipelined_env[0] == '1' && pipelined_env[1] == '\0');
    const std::string device_arch = hip_runtime().device_arch_name(device_id);
    const bool is_gfx1201 =
        device_arch == "gfx1201" || device_arch.rfind("gfx1201:", 0) == 0;
    const bool grouped_shape =
        kv_heads != 0 && is_gfx1201 && q_heads % kv_heads == 0 &&
        q_heads / kv_heads == 5u && head_dim == 128u &&
        value_dim == 128u;
    const size_t partial_groups = grouped_requested && grouped_shape ? kv_heads : q_heads;
    if (split_count != 0 && partial_groups > std::numeric_limits<size_t>::max() / split_count) {
        if (error != nullptr) {
            *error = "paged decode split attention partial grid overflows";
        }
        return false;
    }
    const size_t partial_grid_size = partial_groups * split_count;
    if (partial_grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        q_heads > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "paged decode split attention launch grid exceeds HIP grid limit";
        }
        return false;
    }

    unsigned long long kernel_cache_len = static_cast<unsigned long long>(cache_len);
    unsigned long long kernel_block_size = static_cast<unsigned long long>(block_size);
    unsigned long long kernel_cache_blocks = static_cast<unsigned long long>(cache_blocks);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    unsigned long long kernel_source_tile = static_cast<unsigned long long>(source_tile);
    unsigned long long kernel_split_count = static_cast<unsigned long long>(split_count);
    void *q_ptr = q_buffer->ptr;
    void *gate_ptr = gate_buffer == nullptr ? nullptr : gate_buffer->ptr;
    void *k_ptr = k_cache_buffer->ptr;
    void *v_ptr = v_cache_buffer->ptr;
    void *block_table_ptr = block_table_buffer->ptr;
    void *workspace_ptr = workspace_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *partial_params[] = {
        &q_ptr,
        &k_ptr,
        &v_ptr,
        &block_table_ptr,
        &kernel_cache_len,
        &kernel_block_size,
        &kernel_cache_blocks,
        &kernel_q_heads,
        &kernel_kv_heads,
        &kernel_head_dim,
        &kernel_value_dim,
        &softmax_scale,
        &kernel_source_tile,
        &kernel_split_count,
        &workspace_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            functions.split_partial,
            static_cast<unsigned int>(partial_grid_size),
            launch_block_size,
            partial_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 paged decode split partial attention";
        }
        return false;
    }

    void *merge_params[] = {
        &gate_ptr,
        &kernel_q_heads,
        &kernel_value_dim,
        &kernel_split_count,
        &workspace_ptr,
        &output_ptr,
    };
    if (!hip_runtime().module_launch_kernel(
            functions.split_merge,
            static_cast<unsigned int>(q_heads),
            launch_block_size,
            merge_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 paged decode split merge attention";
        }
        return false;
    }
    return true;
}

bool paged_decode_attn_split_typed_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    const ullm_runtime_buffer *k_scale_buffer,
    const ullm_runtime_buffer *v_scale_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    size_t source_tile,
    size_t split_count,
    ullm_kv_cache_dtype k_dtype,
    ullm_kv_cache_dtype v_dtype,
    ullm_runtime_buffer *workspace_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    if (k_dtype == ULLM_KV_CACHE_DTYPE_F32 && v_dtype == ULLM_KV_CACHE_DTYPE_F32) {
        return paged_decode_attn_split_f32_hip_kernel(
            q_buffer,
            gate_buffer,
            k_cache_buffer,
            v_cache_buffer,
            block_table_buffer,
            cache_len,
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            source_tile,
            split_count,
            workspace_buffer,
            output_buffer,
            stream,
            error);
    }
    const int device_id = q_buffer->hip_device_id;
    void *partial_function =
        hip_paged_decode_attn_kernel_cache().typed_split_partial_function_for_device(
            device_id, k_dtype, v_dtype, error);
    HipPagedDecodeAttnKernelCache::Functions functions;
    if (partial_function == nullptr ||
        !hip_paged_decode_attn_kernel_cache().functions_for_device(device_id, &functions, error) ||
        functions.split_merge == nullptr) {
        if (error != nullptr && error->empty()) {
            *error = "typed paged decode split HIP functions are unavailable";
        }
        return false;
    }
    constexpr unsigned int launch_block_size = 256;
    const bool grouped_qwen35_shape =
        kv_heads != 0u && q_heads % kv_heads == 0u && q_heads / kv_heads == 4u &&
        head_dim == 256u && value_dim == 256u;
    const size_t partial_groups = grouped_qwen35_shape ? kv_heads : q_heads;
    if (split_count != 0u && partial_groups > std::numeric_limits<size_t>::max() / split_count) {
        if (error != nullptr) {
            *error = "typed paged decode split partial grid overflows";
        }
        return false;
    }
    const size_t partial_grid_size = partial_groups * split_count;
    if (partial_grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        q_heads > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "typed paged decode split launch grid exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_cache_len = static_cast<unsigned long long>(cache_len);
    unsigned long long kernel_block_size = static_cast<unsigned long long>(block_size);
    unsigned long long kernel_cache_blocks = static_cast<unsigned long long>(cache_blocks);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    unsigned long long kernel_source_tile = static_cast<unsigned long long>(source_tile);
    unsigned long long kernel_split_count = static_cast<unsigned long long>(split_count);
    void *q_ptr = q_buffer->ptr;
    void *k_ptr = k_cache_buffer->ptr;
    void *v_ptr = v_cache_buffer->ptr;
    void *table_ptr = block_table_buffer->ptr;
    void *k_scale_ptr = k_scale_buffer == nullptr ? nullptr : k_scale_buffer->ptr;
    void *v_scale_ptr = v_scale_buffer == nullptr ? nullptr : v_scale_buffer->ptr;
    void *workspace_ptr = workspace_buffer->ptr;
    void *gate_ptr = gate_buffer == nullptr ? nullptr : gate_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *partial_params[] = {
        &q_ptr,
        &k_ptr,
        &v_ptr,
        &table_ptr,
        &k_scale_ptr,
        &v_scale_ptr,
        &kernel_cache_len,
        &kernel_block_size,
        &kernel_cache_blocks,
        &kernel_q_heads,
        &kernel_kv_heads,
        &kernel_head_dim,
        &kernel_value_dim,
        &softmax_scale,
        &kernel_source_tile,
        &kernel_split_count,
        &workspace_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            partial_function,
            static_cast<unsigned int>(partial_grid_size),
            launch_block_size,
            partial_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for typed paged decode split partial attention";
        }
        return false;
    }
    void *merge_params[] = {
        &gate_ptr,
        &kernel_q_heads,
        &kernel_value_dim,
        &kernel_split_count,
        &workspace_ptr,
        &output_ptr,
    };
    if (!hip_runtime().module_launch_kernel(
            functions.split_merge,
            static_cast<unsigned int>(q_heads),
            launch_block_size,
            merge_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for typed paged decode split merge attention";
        }
        return false;
    }
    return true;
}

ullm_status paged_decode_attn_f32_hip_staging(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    size_t q_bytes,
    size_t gate_bytes,
    size_t k_bytes,
    size_t v_bytes,
    size_t block_table_bytes,
    size_t output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_q(q_bytes / sizeof(float));
    std::vector<float> host_gate(gate_bytes / sizeof(float));
    std::vector<float> host_k(k_bytes / sizeof(float));
    std::vector<float> host_v(v_bytes / sizeof(float));
    std::vector<std::uint32_t> host_block_table(block_table_bytes / sizeof(std::uint32_t));
    std::vector<float> host_output(output_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = q_buffer->hip_device_id;

    bool copied = hip_runtime().copy_async(
            host_q.data(),
            q_buffer->ptr,
            q_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id);
    if (copied && gate_buffer != nullptr) {
        copied = hip_runtime().copy_async(
            host_gate.data(),
            gate_buffer->ptr,
            gate_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id);
    }
    if (!copied ||
        !hip_runtime().copy_async(
            host_k.data(),
            k_cache_buffer->ptr,
            k_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_v.data(),
            v_cache_buffer->ptr,
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
            device_id)) {
        set_error("failed to copy f32 paged decode attention HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 paged decode attention HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    for (std::uint32_t block_id : host_block_table) {
        if (static_cast<size_t>(block_id) >= cache_blocks) {
            set_error("f32 paged decode attention block table contains an out-of-range block id");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
    }

    paged_decode_attn_f32_host(
        host_q.data(),
        host_k.data(),
        host_v.data(),
        host_block_table.data(),
        cache_len,
        block_size,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        host_output.data());
    if (gate_buffer != nullptr) {
        for (size_t i = 0; i < host_output.size(); ++i) {
            const float gate_value = host_gate[i];
            const float sigmoid = 1.0f / (1.0f + std::exp(-gate_value));
            host_output[i] *= sigmoid;
        }
    }
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 paged decode attention output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 paged decode attention HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}
