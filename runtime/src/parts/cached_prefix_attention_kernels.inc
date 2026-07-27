// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
class HipCachedPrefixAttnKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second->function;
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
            if (!hiprtc_runtime().compile_cached_prefix_attn_kernel(arch, &code, &compile_error)) {
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
                    "ullm_cached_prefix_attn_f32_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction failed for " + arch);
                continue;
            }

            auto loaded = std::make_unique<LoadedModule>();
            loaded->module = module;
            loaded->function = function;
            loaded->arch = arch;
            void *result = loaded->function;
            modules_.emplace(device_id, std::move(loaded));
            return result;
        }
        append_error(
            error,
            compile_errors.empty() ? "failed to build cached prefix attention HIP kernel" :
                                     compile_errors);
        return nullptr;
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

HipCachedPrefixAttnKernelCache &hip_cached_prefix_attn_kernel_cache() {
    static HipCachedPrefixAttnKernelCache cache;
    return cache;
}

bool cached_prefix_attn_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    void *function = hip_cached_prefix_attn_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t grid_size = new_tokens * q_heads;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "cached prefix attention q head-sequence count exceeds HIP grid limit";
        }
        return false;
    }

    unsigned long long kernel_cached_prefix_len =
        static_cast<unsigned long long>(cached_prefix_len);
    unsigned long long kernel_new_tokens = static_cast<unsigned long long>(new_tokens);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *q_ptr = q_buffer->ptr;
    void *k_ptr = k_cache_buffer->ptr;
    void *v_ptr = v_cache_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr,
        &k_ptr,
        &v_ptr,
        &kernel_cached_prefix_len,
        &kernel_new_tokens,
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
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 cached prefix attention";
        }
        return false;
    }
    return true;
}

ullm_status cached_prefix_attn_f32_hip_staging(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    size_t q_bytes,
    size_t k_bytes,
    size_t v_bytes,
    size_t output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_q(q_bytes / sizeof(float));
    std::vector<float> host_k(k_bytes / sizeof(float));
    std::vector<float> host_v(v_bytes / sizeof(float));
    std::vector<float> host_output(output_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = q_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_q.data(),
            q_buffer->ptr,
            q_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
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
            device_id)) {
        set_error("failed to copy f32 cached prefix attention HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 cached prefix attention HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    cached_prefix_attn_f32_host(
        host_q.data(),
        host_k.data(),
        host_v.data(),
        cached_prefix_len,
        new_tokens,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 cached prefix attention output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 cached prefix attention HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipCachedPrefixAttnFp8E4m3KernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second->function;
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
            if (!hiprtc_runtime().compile_cached_prefix_attn_fp8_e4m3_kernel(
                    arch,
                    &code,
                    &compile_error)) {
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
                    "ullm_cached_prefix_attn_fp8_e4m3_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction failed for " + arch);
                continue;
            }

            auto loaded = std::make_unique<LoadedModule>();
            loaded->module = module;
            loaded->function = function;
            loaded->arch = arch;
            void *result = loaded->function;
            modules_.emplace(device_id, std::move(loaded));
            return result;
        }
        append_error(
            error,
            compile_errors.empty() ? "failed to build fp8 e4m3 cached prefix attention HIP kernel" :
                                     compile_errors);
        return nullptr;
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

HipCachedPrefixAttnFp8E4m3KernelCache &hip_cached_prefix_attn_fp8_e4m3_kernel_cache() {
    static HipCachedPrefixAttnFp8E4m3KernelCache cache;
    return cache;
}

enum class CachedPrefixAttnF32Flash2KernelVariant {
    Generic,
    StagedWave32Prototype,
    GqaGrouped,
};

class HipCachedPrefixAttnF32Flash2KernelCache {
public:
    void *function_for_device(
        int device_id,
        CachedPrefixAttnF32Flash2KernelVariant variant,
        std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto *modules = &modules_;
        if (variant == CachedPrefixAttnF32Flash2KernelVariant::StagedWave32Prototype) {
            modules = &staged_wave32_prototype_modules_;
        } else if (variant == CachedPrefixAttnF32Flash2KernelVariant::GqaGrouped) {
            modules = &gqa_grouped_modules_;
        }
        auto found = modules->find(device_id);
        if (found != modules->end()) {
            return found->second->function;
        }

        const std::vector<std::string> candidates = hip_arch_candidates(device_id);
        if (candidates.empty()) {
            append_error(error, "unable to infer HIP offload architecture for device");
            return nullptr;
        }

        std::string compile_errors;
        for (const std::string &arch : candidates) {
            if (variant != CachedPrefixAttnF32Flash2KernelVariant::Generic &&
                arch.rfind("gfx1201", 0) != 0) {
                append_error(
                    &compile_errors,
                    "f32 cached prefix Flash2 specialized kernel is restricted to gfx1201");
                continue;
            }
            std::vector<char> code;
            std::string compile_error;
            bool compiled = false;
            if (variant == CachedPrefixAttnF32Flash2KernelVariant::StagedWave32Prototype) {
                compiled =
                    hiprtc_runtime().compile_cached_prefix_attn_f32_flash2_staged_wave32_prototype_kernel(
                        arch,
                        &code,
                        &compile_error);
            } else if (variant == CachedPrefixAttnF32Flash2KernelVariant::GqaGrouped) {
                compiled = hiprtc_runtime().compile_cached_prefix_attn_f32_flash2_gqa_grouped_kernel(
                    arch,
                    &code,
                    &compile_error);
            } else {
                compiled = hiprtc_runtime().compile_cached_prefix_attn_f32_flash2_kernel(
                    arch,
                    &code,
                    &compile_error);
            }
            if (!compiled) {
                append_error(&compile_errors, compile_error);
                continue;
            }

            void *module = nullptr;
            if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
                append_error(&compile_errors, "hipModuleLoadData failed for " + arch);
                continue;
            }
            void *function = nullptr;
            const char *function_name = "ullm_cached_prefix_attn_f32_flash2_kernel";
            if (variant == CachedPrefixAttnF32Flash2KernelVariant::StagedWave32Prototype) {
                function_name =
                    "ullm_sq8_0_cached_prefix_attn_f32_flash2_staged_wave32_prototype_kernel";
            } else if (variant == CachedPrefixAttnF32Flash2KernelVariant::GqaGrouped) {
                function_name = "ullm_cached_prefix_attn_f32_flash2_gqa_grouped_kernel";
            }
            if (!hip_runtime().module_get_function(
                    &function,
                    module,
                    function_name,
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction failed for " + arch);
                continue;
            }

            auto loaded = std::make_unique<LoadedModule>();
            loaded->module = module;
            loaded->function = function;
            loaded->arch = arch;
            void *result = loaded->function;
            modules->emplace(device_id, std::move(loaded));
            return result;
        }
        const char *build_failure = "failed to build f32 cached prefix flash2 attention HIP kernel";
        if (variant == CachedPrefixAttnF32Flash2KernelVariant::StagedWave32Prototype) {
            build_failure = "failed to build f32 cached prefix Flash2 staged wave32 prototype HIP kernel";
        } else if (variant == CachedPrefixAttnF32Flash2KernelVariant::GqaGrouped) {
            build_failure = "failed to build f32 cached prefix Flash2 GQA grouped HIP kernel";
        }
        append_error(
            error,
            compile_errors.empty() ? build_failure :
                compile_errors);
        return nullptr;
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
    std::unordered_map<int, std::unique_ptr<LoadedModule>> staged_wave32_prototype_modules_;
    std::unordered_map<int, std::unique_ptr<LoadedModule>> gqa_grouped_modules_;
};

HipCachedPrefixAttnF32Flash2KernelCache &hip_cached_prefix_attn_f32_flash2_kernel_cache() {
    static HipCachedPrefixAttnF32Flash2KernelCache cache;
    return cache;
}

class HipCachedPrefixAttnFp8E4m3Flash2KernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second->function;
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
            if (!hiprtc_runtime().compile_cached_prefix_attn_fp8_e4m3_flash2_kernel(
                    arch,
                    &code,
                    &compile_error)) {
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
                    "ullm_cached_prefix_attn_fp8_e4m3_flash2_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction failed for " + arch);
                continue;
            }

            auto loaded = std::make_unique<LoadedModule>();
            loaded->module = module;
            loaded->function = function;
            loaded->arch = arch;
            void *result = loaded->function;
            modules_.emplace(device_id, std::move(loaded));
            return result;
        }
        append_error(
            error,
            compile_errors.empty() ?
                "failed to build fp8 e4m3 cached prefix flash2 attention HIP kernel" :
                compile_errors);
        return nullptr;
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

HipCachedPrefixAttnFp8E4m3Flash2KernelCache &hip_cached_prefix_attn_fp8_e4m3_flash2_kernel_cache() {
    static HipCachedPrefixAttnFp8E4m3Flash2KernelCache cache;
    return cache;
}

class HipCachedPrefixAttnFp8E4m3RocwmmaKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second->function;
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
            if (!hiprtc_runtime().compile_cached_prefix_attn_fp8_e4m3_rocwmma_kernel(
                    arch,
                    &code,
                    &compile_error)) {
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
                    "ullm_cached_prefix_attn_fp8_e4m3_rocwmma_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction failed for " + arch);
                continue;
            }

            auto loaded = std::make_unique<LoadedModule>();
            loaded->module = module;
            loaded->function = function;
            loaded->arch = arch;
            void *result = loaded->function;
            modules_.emplace(device_id, std::move(loaded));
            return result;
        }
        append_error(
            error,
            compile_errors.empty() ?
                "failed to build fp8 e4m3 cached prefix rocWMMA attention HIP kernel" :
                compile_errors);
        return nullptr;
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

HipCachedPrefixAttnFp8E4m3RocwmmaKernelCache &
hip_cached_prefix_attn_fp8_e4m3_rocwmma_kernel_cache() {
    static HipCachedPrefixAttnFp8E4m3RocwmmaKernelCache cache;
    return cache;
}

bool cached_prefix_attn_fp8_e4m3_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float k_scale,
    float v_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    void *function =
        hip_cached_prefix_attn_fp8_e4m3_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t grid_size = new_tokens * q_heads;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "fp8 e4m3 cached prefix attention q head-sequence count exceeds HIP grid limit";
        }
        return false;
    }

    unsigned long long kernel_cached_prefix_len =
        static_cast<unsigned long long>(cached_prefix_len);
    unsigned long long kernel_new_tokens = static_cast<unsigned long long>(new_tokens);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *q_ptr = q_buffer->ptr;
    void *k_ptr = k_cache_buffer->ptr;
    void *v_ptr = v_cache_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr,
        &k_ptr,
        &v_ptr,
        &kernel_cached_prefix_len,
        &kernel_new_tokens,
        &kernel_q_heads,
        &kernel_kv_heads,
        &kernel_head_dim,
        &kernel_value_dim,
        &softmax_scale,
        &k_scale,
        &v_scale,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for fp8 e4m3 cached prefix attention";
        }
        return false;
    }
    return true;
}

bool cached_prefix_attn_f32_flash2_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    // Keep the existing isolated staged-wave32 experiment opt-in.  The GQA
    // mapping below is the normal fast path only for the exact validated
    // Qwen3-14B F32-KV shape on gfx1201; every other layout retains generic
    // Flash2.  An explicit disable remains available for A/B diagnostics.
    const bool use_staged_wave32_prototype =
        std::getenv("ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE") != nullptr;
    const bool disable_gqa_grouped =
        std::getenv("ULLM_DISABLE_SQ8_0_FLASH2_GQA_GROUPED") != nullptr;
    const std::string device_arch = hip_runtime().device_arch_name(device_id);
    const bool is_gfx1201 =
        device_arch == "gfx1201" || device_arch.rfind("gfx1201:", 0) == 0;
    const bool use_gqa_grouped = !use_staged_wave32_prototype && !disable_gqa_grouped &&
        is_gfx1201 && kv_heads != 0u && q_heads % kv_heads == 0u &&
        q_heads / kv_heads == 5u && head_dim == 128u && value_dim == 128u;
    CachedPrefixAttnF32Flash2KernelVariant variant =
        CachedPrefixAttnF32Flash2KernelVariant::Generic;
    if (use_staged_wave32_prototype) {
        variant = CachedPrefixAttnF32Flash2KernelVariant::StagedWave32Prototype;
    } else if (use_gqa_grouped) {
        variant = CachedPrefixAttnF32Flash2KernelVariant::GqaGrouped;
    }
    void *function = hip_cached_prefix_attn_f32_flash2_kernel_cache().function_for_device(
        device_id,
        variant,
        error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    if (value_dim > block_size) {
        if (error != nullptr) {
            *error = "f32 cached prefix flash2 attention value_dim exceeds 256";
        }
        return false;
    }
    const size_t launch_heads = use_gqa_grouped ? kv_heads : q_heads;
    const size_t grid_size = new_tokens * launch_heads;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error =
                "f32 cached prefix flash2 attention q head-sequence count exceeds HIP grid limit";
        }
        return false;
    }

    unsigned long long kernel_cached_prefix_len =
        static_cast<unsigned long long>(cached_prefix_len);
    unsigned long long kernel_new_tokens = static_cast<unsigned long long>(new_tokens);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *q_ptr = q_buffer->ptr;
    void *k_ptr = k_cache_buffer->ptr;
    void *v_ptr = v_cache_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr,
        &k_ptr,
        &v_ptr,
        &kernel_cached_prefix_len,
        &kernel_new_tokens,
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
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 cached prefix flash2 attention";
        }
        return false;
    }
    return true;
}

bool cached_prefix_attn_fp8_e4m3_flash2_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float q_scale,
    float k_scale,
    float v_scale,
    bool q_is_fp8,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    void *function =
        hip_cached_prefix_attn_fp8_e4m3_flash2_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    if (value_dim > block_size) {
        if (error != nullptr) {
            *error = "fp8 e4m3 cached prefix flash2 attention value_dim exceeds 256";
        }
        return false;
    }
    const size_t grid_size = new_tokens * q_heads;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error =
                "fp8 e4m3 cached prefix flash2 attention q head-sequence count exceeds HIP grid limit";
        }
        return false;
    }

    unsigned long long kernel_cached_prefix_len =
        static_cast<unsigned long long>(cached_prefix_len);
    unsigned long long kernel_new_tokens = static_cast<unsigned long long>(new_tokens);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *q_ptr = q_buffer->ptr;
    void *k_ptr = k_cache_buffer->ptr;
    void *v_ptr = v_cache_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    unsigned int kernel_q_is_fp8 = q_is_fp8 ? 1u : 0u;
    void *kernel_params[] = {
        &q_ptr,
        &k_ptr,
        &v_ptr,
        &kernel_cached_prefix_len,
        &kernel_new_tokens,
        &kernel_q_heads,
        &kernel_kv_heads,
        &kernel_head_dim,
        &kernel_value_dim,
        &softmax_scale,
        &q_scale,
        &k_scale,
        &v_scale,
        &kernel_q_is_fp8,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for fp8 e4m3 cached prefix flash2 attention";
        }
        return false;
    }
    return true;
}

bool cached_prefix_attn_fp8_e4m3_rocwmma_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float q_scale,
    float k_scale,
    float v_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    int major = 0;
    int minor = 0;
    hip_runtime().device_compute_capability(device_id, &major, &minor);
    if (major != 12) {
        if (error != nullptr) {
            *error = "fp8 e4m3 cached prefix rocWMMA attention requires RDNA4/gfx12 HIP device";
        }
        return false;
    }
    const size_t q_per_kv = q_heads / kv_heads;
    if ((q_per_kv % 16u) != 0u) {
        if (error != nullptr) {
            *error = "fp8 e4m3 cached prefix rocWMMA attention requires q_heads/kv_heads to be a multiple of 16";
        }
        return false;
    }
    if ((head_dim % 16u) != 0u || (value_dim % 16u) != 0u) {
        if (error != nullptr) {
            *error = "fp8 e4m3 cached prefix rocWMMA attention requires head_dim and value_dim to be multiples of 16";
        }
        return false;
    }
    const size_t q_groups_per_kv = q_per_kv / 16u;
    const size_t value_group_width = rocwmma_cached_prefix_value_group_width(new_tokens);
    const size_t value_groups = (value_dim + value_group_width - 1u) / value_group_width;
    if (kv_heads > std::numeric_limits<size_t>::max() / q_groups_per_kv ||
        kv_heads * q_groups_per_kv > std::numeric_limits<size_t>::max() / value_groups) {
        if (error != nullptr) {
            *error = "fp8 e4m3 cached prefix rocWMMA attention groups-per-token overflows";
        }
        return false;
    }
    const size_t groups_per_token = kv_heads * q_groups_per_kv * value_groups;
    if (new_tokens > std::numeric_limits<size_t>::max() / groups_per_token) {
        if (error != nullptr) {
            *error =
                "fp8 e4m3 cached prefix rocWMMA attention grid size overflows";
        }
        return false;
    }
    const size_t grid_size = new_tokens * groups_per_token;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error =
                "fp8 e4m3 cached prefix rocWMMA attention grid size exceeds HIP grid limit";
        }
        return false;
    }

    void *function =
        hip_cached_prefix_attn_fp8_e4m3_rocwmma_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 128;
    unsigned long long kernel_cached_prefix_len =
        static_cast<unsigned long long>(cached_prefix_len);
    unsigned long long kernel_new_tokens = static_cast<unsigned long long>(new_tokens);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_q_groups_per_kv =
        static_cast<unsigned long long>(q_groups_per_kv);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    unsigned long long kernel_value_group_width =
        static_cast<unsigned long long>(value_group_width);
    void *q_ptr = q_buffer->ptr;
    void *k_ptr = k_cache_buffer->ptr;
    void *v_ptr = v_cache_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr,
        &k_ptr,
        &v_ptr,
        &kernel_cached_prefix_len,
        &kernel_new_tokens,
        &kernel_q_heads,
        &kernel_kv_heads,
        &kernel_q_groups_per_kv,
        &kernel_head_dim,
        &kernel_value_dim,
        &kernel_value_group_width,
        &softmax_scale,
        &q_scale,
        &k_scale,
        &v_scale,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const size_t dynamic_shared_bytes = 16u * value_group_width * sizeof(float);
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            block_size,
            kernel_params,
            hip_stream,
            device_id,
            static_cast<unsigned int>(dynamic_shared_bytes))) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for fp8 e4m3 cached prefix rocWMMA attention";
        }
        return false;
    }
    return true;
}

ullm_status cached_prefix_attn_fp8_e4m3_hip_staging(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float k_scale,
    float v_scale,
    size_t q_bytes,
    size_t k_bytes,
    size_t v_bytes,
    size_t output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_q(q_bytes / sizeof(float));
    std::vector<uint8_t> host_k(k_bytes);
    std::vector<uint8_t> host_v(v_bytes);
    std::vector<float> host_output(output_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = q_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_q.data(),
            q_buffer->ptr,
            q_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
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
            device_id)) {
        set_error("failed to copy fp8 e4m3 cached prefix attention HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize fp8 e4m3 cached prefix attention HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    cached_prefix_attn_fp8_e4m3_host(
        host_q.data(),
        host_k.data(),
        host_v.data(),
        cached_prefix_len,
        new_tokens,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        k_scale,
        v_scale,
        host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy fp8 e4m3 cached prefix attention output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize fp8 e4m3 cached prefix attention HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

