// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
class HipPagedChunkKernelCache {
public:
    void *writer_for_device(int device_id, std::string *error) {
        ensure_device(device_id, error);
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        return found == modules_.end() ? nullptr : found->second->writer;
    }

    void *reader_for_device(int device_id, std::string *error) {
        ensure_device(device_id, error);
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        return found == modules_.end() ? nullptr : found->second->reader;
    }

    void *typed_writer_for_device(
        int device_id,
        ullm_kv_cache_dtype k_dtype,
        ullm_kv_cache_dtype v_dtype,
        std::string *error) {
        const int k_index = static_cast<int>(k_dtype);
        const int v_index = static_cast<int>(v_dtype);
        if (k_index < 0 || k_index > 2 || v_index < 0 || v_index > 2) {
            append_error(error, "typed paged chunk writer received an unsupported cache dtype");
            return nullptr;
        }
        ensure_device(device_id, error);
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = modules_.find(device_id);
        if (found == modules_.end()) {
            return nullptr;
        }
        return k_index == 0 && v_index == 0 ? found->second->writer
                                             : found->second->typed_writer[k_index][v_index];
    }

    void *typed_reader_for_device(
        int device_id,
        ullm_kv_cache_dtype k_dtype,
        ullm_kv_cache_dtype v_dtype,
        std::string *error) {
        const int k_index = static_cast<int>(k_dtype);
        const int v_index = static_cast<int>(v_dtype);
        if (k_index < 0 || k_index > 2 || v_index < 0 || v_index > 2) {
            append_error(error, "typed paged chunk reader received an unsupported cache dtype");
            return nullptr;
        }
        ensure_device(device_id, error);
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = modules_.find(device_id);
        if (found == modules_.end()) {
            return nullptr;
        }
        return k_index == 0 && v_index == 0 ? found->second->reader
                                             : found->second->typed_reader[k_index][v_index];
    }

private:
    struct LoadedModule {
        void *module = nullptr;
        void *writer = nullptr;
        void *reader = nullptr;
        void *typed_writer[3][3] = {};
        void *typed_reader[3][3] = {};
        std::string arch;
    };

    static const char *typed_writer_symbol_name(int k_dtype, int v_dtype) {
        static const char *const names[3][3] = {
            {nullptr,
             "ullm_paged_kv_write_chunk_typed_kf32_vf16_kernel",
             "ullm_paged_kv_write_chunk_typed_kf32_vfp8_kernel"},
            {"ullm_paged_kv_write_chunk_typed_kf16_vf32_kernel",
             "ullm_paged_kv_write_chunk_typed_kf16_vf16_kernel",
             "ullm_paged_kv_write_chunk_typed_kf16_vfp8_kernel"},
            {"ullm_paged_kv_write_chunk_typed_kfp8_vf32_kernel",
             "ullm_paged_kv_write_chunk_typed_kfp8_vf16_kernel",
             "ullm_paged_kv_write_chunk_typed_kfp8_vfp8_kernel"},
        };
        return names[k_dtype][v_dtype];
    }

    static const char *typed_reader_symbol_name(int k_dtype, int v_dtype) {
        static const char *const names[3][3] = {
            {nullptr,
             "ullm_paged_causal_gqa_chunk_typed_kf32_vf16_kernel",
             "ullm_paged_causal_gqa_chunk_typed_kf32_vfp8_kernel"},
            {"ullm_paged_causal_gqa_chunk_typed_kf16_vf32_kernel",
             "ullm_paged_causal_gqa_chunk_typed_kf16_vf16_kernel",
             "ullm_paged_causal_gqa_chunk_typed_kf16_vfp8_kernel"},
            {"ullm_paged_causal_gqa_chunk_typed_kfp8_vf32_kernel",
             "ullm_paged_causal_gqa_chunk_typed_kfp8_vf16_kernel",
             "ullm_paged_causal_gqa_chunk_typed_kfp8_vfp8_kernel"},
        };
        return names[k_dtype][v_dtype];
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

    void ensure_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (modules_.find(device_id) != modules_.end()) {
            return;
        }
        const std::vector<std::string> candidates = hip_arch_candidates(device_id);
        if (candidates.empty()) {
            append_error(error, "unable to infer HIP offload architecture for device");
            return;
        }
        std::string compile_errors;
        for (const std::string &arch : candidates) {
            std::vector<char> code;
            std::string compile_error;
            if (!hiprtc_runtime().compile_paged_chunk_kernel(arch, &code, &compile_error)) {
                append_error(&compile_errors, compile_error);
                continue;
            }
            void *module = nullptr;
            if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
                append_error(&compile_errors, "hipModuleLoadData failed for " + arch);
                continue;
            }
            void *writer = nullptr;
            void *reader = nullptr;
            if (!hip_runtime().module_get_function(
                    &writer, module, "ullm_paged_kv_write_chunk_f32_kernel", device_id) ||
                !hip_runtime().module_get_function(
                    &reader, module, "ullm_paged_causal_gqa_chunk_f32_kernel", device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction failed for paged chunk kernels");
                continue;
            }
            auto loaded = std::make_unique<LoadedModule>();
            loaded->module = module;
            loaded->writer = writer;
            loaded->reader = reader;
            bool typed_loaded = true;
            for (int k_dtype = 0; k_dtype < 3 && typed_loaded; ++k_dtype) {
                for (int v_dtype = 0; v_dtype < 3; ++v_dtype) {
                    if (k_dtype == 0 && v_dtype == 0) {
                        continue;
                    }
                    if (!hip_runtime().module_get_function(
                            &loaded->typed_writer[k_dtype][v_dtype],
                            module,
                            typed_writer_symbol_name(k_dtype, v_dtype),
                            device_id) ||
                        !hip_runtime().module_get_function(
                            &loaded->typed_reader[k_dtype][v_dtype],
                            module,
                            typed_reader_symbol_name(k_dtype, v_dtype),
                            device_id)) {
                        typed_loaded = false;
                        break;
                    }
                }
            }
            if (!typed_loaded) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction typed paged chunk kernel failed for " + arch);
                continue;
            }
            loaded->arch = arch;
            modules_.emplace(device_id, std::move(loaded));
            return;
        }
        append_error(
            error,
            compile_errors.empty() ? "failed to build paged chunk HIP kernels" : compile_errors);
    }

    std::mutex mutex_;
    std::unordered_map<int, std::unique_ptr<LoadedModule>> modules_;
};

HipPagedChunkKernelCache &hip_paged_chunk_kernel_cache() {
    static HipPagedChunkKernelCache cache;
    return cache;
}

// Kept separate from HipPagedChunkKernelCache so the scalar paged reader keeps its original
// HIPRTC source, compile options, loaded module, and dispatch behavior.
