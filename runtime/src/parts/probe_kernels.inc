// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
namespace {

bool checked_range(size_t offset, size_t bytes, size_t total) {
    return offset <= total && bytes <= total - offset;
}

bool stream_matches_buffer(const ullm_runtime_buffer *buffer, const ullm_runtime_stream *stream) {
    if (stream == nullptr) {
        return true;
    }
    return buffer->backend == stream->backend && buffer->hip_device_id == stream->hip_device_id;
}

bool buffers_share_backend(
    const ullm_runtime_buffer *lhs,
    const ullm_runtime_buffer *rhs) {
    return lhs->backend == rhs->backend && lhs->hip_device_id == rhs->hip_device_id;
}

bool synchronize_hip_staging(const ullm_runtime_stream *stream, int device_id) {
    if (stream != nullptr) {
        return hip_runtime().synchronize_stream(stream->stream, device_id);
    }
    return hip_runtime().synchronize_device(device_id);
}

bool aq4_dequant_host(
    const std::uint8_t *indices,
    const std::uint8_t *scale_indices,
    const float *codebook,
    const float *scale_values,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t elements,
    float *output) {
    for (size_t element = 0; element < elements; ++element) {
        const std::uint8_t packed = indices[element / 2];
        const std::uint8_t codebook_index =
            (element % 2 == 0) ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
        const size_t group = element / group_size;
        const size_t scale_index = static_cast<size_t>(scale_indices[group]);
        if (scale_index >= scale_count) {
            set_error("AQ4 dequant scale index is out of range");
            return false;
        }
        output[element] = codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
    }
    return true;
}

bool aq4_row_f32_host(
    const std::uint8_t *indices,
    const std::uint8_t *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *row_scales,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t row_index,
    float *output) {
    if (row_index >= rows) {
        set_error("AQ4 row index is out of range");
        return false;
    }
    const size_t row_offset = row_index * cols;
    for (size_t col = 0; col < cols; ++col) {
        const size_t element = row_offset + col;
        const std::uint8_t packed = indices[element / 2];
        const std::uint8_t codebook_index =
            (element % 2 == 0) ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
        const size_t group = element / group_size;
        const size_t scale_index = static_cast<size_t>(scale_indices[group]);
        if (scale_index >= scale_count) {
            set_error("AQ4 row scale index is out of range");
            return false;
        }
        float value = codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
        if (row_scales != nullptr && row_index < row_scale_count) {
            value *= row_scales[row_index];
        }
        output[col] = value;
    }
    return true;
}

bool aq4_scale_indices_host_are_valid(
    const std::uint8_t *scale_indices,
    size_t groups,
    size_t scale_count) {
    for (size_t group = 0; group < groups; ++group) {
        if (static_cast<size_t>(scale_indices[group]) >= scale_count) {
            set_error("AQ4 matvec batch scale index is out of range");
            return false;
        }
    }
    return true;
}

std::vector<std::string> hip_arch_candidates(int device_id) {
    int major = 0;
    int minor = 0;
    hip_runtime().device_compute_capability(device_id, &major, &minor);
    std::vector<std::string> candidates;
    if (major == 12 && minor == 0) {
        candidates.emplace_back("gfx1201");
        candidates.emplace_back("gfx1200");
    } else if (major == 10 && minor == 3) {
        candidates.emplace_back("gfx1030");
    } else if (major == 9) {
        candidates.emplace_back("gfx9" + std::to_string(minor) + "0");
    } else if (major > 0) {
        candidates.emplace_back("gfx" + std::to_string(major) + std::to_string(minor) + "0");
    }
    return candidates;
}

struct Aq4MatvecLaunchConfig {
    unsigned int block_size = 256;
    unsigned int rows_per_block = 1;
};

Aq4MatvecLaunchConfig aq4_matvec_launch_config_for_device(int device_id) {
    int major = 0;
    int minor = 0;
    hip_runtime().device_compute_capability(device_id, &major, &minor);
    const unsigned int fallback = major >= 12 ? 32u : 1u;
    return Aq4MatvecLaunchConfig{
        256u,
        aq4_rows_per_block_from_env("ULLM_AQ4_MATVEC_RPB", nullptr, fallback)};
}

Aq4MatvecLaunchConfig aq4_matvec_launch_config_for_fused_kernel(
    int device_id,
    unsigned int rdna4_rows_per_block,
    unsigned int rdna2_rows_per_block = 2u,
    const char *env_name = nullptr) {
    int major = 0;
    int minor = 0;
    hip_runtime().device_compute_capability(device_id, &major, &minor);
    Aq4MatvecLaunchConfig launch_config = aq4_matvec_launch_config_for_device(device_id);
    const unsigned int fallback = major >= 12 ? rdna4_rows_per_block : rdna2_rows_per_block;
    launch_config.rows_per_block =
        aq4_rows_per_block_from_env(env_name, "ULLM_AQ4_FUSED_RPB", fallback);
    return launch_config;
}

class HipWmmaFp8ProbeKernelCache {
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
            if (!hiprtc_runtime().compile_wmma_fp8_probe_kernel(arch, &code, &compile_error)) {
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
                    "ullm_wmma_fp8_probe_kernel",
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
            compile_errors.empty() ? "failed to build WMMA FP8 probe HIP kernel" :
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

HipWmmaFp8ProbeKernelCache &hip_wmma_fp8_probe_kernel_cache() {
    static HipWmmaFp8ProbeKernelCache cache;
    return cache;
}

bool wmma_fp8_probe_hip_kernel(
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = output_buffer->hip_device_id;
    void *function = hip_wmma_fp8_probe_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {&output_ptr};
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            1u,
            32u,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for WMMA FP8 probe";
        }
        return false;
    }
    return true;
}

class HipWmmaFp8QkProbeKernelCache {
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
            if (!hiprtc_runtime().compile_wmma_fp8_qk_probe_kernel(arch, &code, &compile_error)) {
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
                    "ullm_wmma_fp8_qk_probe_kernel",
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
            compile_errors.empty() ? "failed to build WMMA FP8 QK probe HIP kernel" :
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

HipWmmaFp8QkProbeKernelCache &hip_wmma_fp8_qk_probe_kernel_cache() {
    static HipWmmaFp8QkProbeKernelCache cache;
    return cache;
}

bool wmma_fp8_qk_probe_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = output_buffer->hip_device_id;
    int major = 0;
    int minor = 0;
    hip_runtime().device_compute_capability(device_id, &major, &minor);
    if (major != 12) {
        if (error != nullptr) {
            *error = "WMMA FP8 QK probe requires RDNA4/gfx12 HIP device";
        }
        return false;
    }

    void *function = hip_wmma_fp8_qk_probe_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const void *q_ptr = q_buffer->ptr;
    const void *k_ptr = k_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {&q_ptr, &k_ptr, &output_ptr};
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            1u,
            32u,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for WMMA FP8 QK probe";
        }
        return false;
    }
    return true;
}

class HipRocwmmaFp8QkProbeKernelCache {
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
            if (!hiprtc_runtime().compile_rocwmma_fp8_qk_probe_kernel(
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
                    "ullm_rocwmma_fp8_qk_probe_kernel",
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
            compile_errors.empty() ? "failed to build rocWMMA FP8 QK probe HIP kernel" :
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

HipRocwmmaFp8QkProbeKernelCache &hip_rocwmma_fp8_qk_probe_kernel_cache() {
    static HipRocwmmaFp8QkProbeKernelCache cache;
    return cache;
}

bool rocwmma_fp8_qk_probe_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = output_buffer->hip_device_id;
    int major = 0;
    int minor = 0;
    hip_runtime().device_compute_capability(device_id, &major, &minor);
    if (major != 12) {
        if (error != nullptr) {
            *error = "rocWMMA FP8 QK probe requires RDNA4/gfx12 HIP device";
        }
        return false;
    }

    void *function = hip_rocwmma_fp8_qk_probe_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const void *q_ptr = q_buffer->ptr;
    const void *k_ptr = k_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {&q_ptr, &k_ptr, &output_ptr};
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            1u,
            32u,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for rocWMMA FP8 QK probe";
        }
        return false;
    }
    return true;
}

class HipRocwmmaFp8AttnProbeKernelCache {
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
            if (!hiprtc_runtime().compile_rocwmma_fp8_attn_probe_kernel(
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
                    "ullm_rocwmma_fp8_attn_probe_kernel",
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
            compile_errors.empty() ? "failed to build rocWMMA FP8 attention probe HIP kernel" :
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

HipRocwmmaFp8AttnProbeKernelCache &hip_rocwmma_fp8_attn_probe_kernel_cache() {
    static HipRocwmmaFp8AttnProbeKernelCache cache;
    return cache;
}

bool rocwmma_fp8_attn_probe_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = output_buffer->hip_device_id;
    int major = 0;
    int minor = 0;
    hip_runtime().device_compute_capability(device_id, &major, &minor);
    if (major != 12) {
        if (error != nullptr) {
            *error = "rocWMMA FP8 attention probe requires RDNA4/gfx12 HIP device";
        }
        return false;
    }

    void *function = hip_rocwmma_fp8_attn_probe_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const void *q_ptr = q_buffer->ptr;
    const void *k_ptr = k_buffer->ptr;
    const void *v_ptr = v_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {&q_ptr, &k_ptr, &v_ptr, &output_ptr};
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            1u,
            32u,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for rocWMMA FP8 attention probe";
        }
        return false;
    }
    return true;
}

