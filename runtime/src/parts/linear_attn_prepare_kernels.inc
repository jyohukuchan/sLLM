// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
class HipLinearAttnQkvPrepareKernelCache {
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
            if (!hiprtc_runtime().compile_linear_attn_qkv_prepare_kernel(arch, &code, &compile_error)) {
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
                    "ullm_linear_attn_qkv_prepare_f32_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction qkv prepare failed for " + arch);
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
            compile_errors.empty() ? "failed to build linear attention qkv prepare HIP kernel"
                                   : compile_errors);
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

HipLinearAttnQkvPrepareKernelCache &hip_linear_attn_qkv_prepare_kernel_cache() {
    static HipLinearAttnQkvPrepareKernelCache cache;
    return cache;
}

constexpr size_t kLinearAttnQkvPrepareShufflePrototypeKeyHeads = 16u;
constexpr size_t kLinearAttnQkvPrepareShufflePrototypeValueHeads = 32u;
constexpr size_t kLinearAttnQkvPrepareShufflePrototypeKeyDim = 128u;
constexpr size_t kLinearAttnQkvPrepareShufflePrototypeValueDim = 128u;
constexpr size_t kLinearAttnQkvPrepareShufflePrototypeKernelSize = 4u;
constexpr unsigned int kLinearAttnQkvPrepareShufflePrototypeThreads = 256u;
constexpr unsigned int kLinearAttnQkvPrepareShufflePrototypeGrid = 48u;

bool linear_attn_qkv_prepare_shuffle_prototype_shape_is_supported(
    const std::string &arch,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size) {
    return arch == "gfx1201" &&
        key_heads == kLinearAttnQkvPrepareShufflePrototypeKeyHeads &&
        value_heads == kLinearAttnQkvPrepareShufflePrototypeValueHeads &&
        key_dim == kLinearAttnQkvPrepareShufflePrototypeKeyDim &&
        value_dim == kLinearAttnQkvPrepareShufflePrototypeValueDim &&
        kernel_size == kLinearAttnQkvPrepareShufflePrototypeKernelSize;
}

// This cache is intentionally independent from the served qkv-prepare module so the direct
// experiment cannot affect production HIPRTC source selection, code objects, or dispatch.
class HipLinearAttnQkvPrepareShufflePrototypeKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second->function;
        }
        auto failed = failures_.find(device_id);
        if (failed != failures_.end()) {
            append_error(error, failed->second);
            return nullptr;
        }
        if (hip_runtime().device_arch_name(device_id) != "gfx1201") {
            append_error(
                error,
                "linear attention qkv prepare shuffle prototype requires exact gfx1201 architecture");
            return nullptr;
        }

        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_linear_attn_qkv_prepare_shuffle_prototype_kernel(
                "gfx1201",
                &code,
                &compile_error)) {
            return fail(
                device_id,
                compile_error.empty() ?
                    "failed to build linear attention qkv prepare shuffle prototype HIP kernel" :
                    compile_error,
                error);
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            return fail(
                device_id,
                "hipModuleLoadData failed for linear attention qkv prepare shuffle prototype",
                error);
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function,
                module,
                "ullm_linear_attn_qkv_prepare_shuffle_prototype_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            return fail(
                device_id,
                "hipModuleGetFunction failed for linear attention qkv prepare shuffle prototype",
                error);
        }

        auto loaded = std::make_unique<LoadedModule>();
        loaded->module = module;
        loaded->function = function;
        void *result = loaded->function;
        modules_.emplace(device_id, std::move(loaded));
        return result;
    }

private:
    struct LoadedModule {
        void *module = nullptr;
        void *function = nullptr;
    };

    void *fail(int device_id, const std::string &message, std::string *error) {
        failures_.emplace(device_id, message);
        append_error(error, message);
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
    std::unordered_map<int, std::string> failures_;
};

HipLinearAttnQkvPrepareShufflePrototypeKernelCache &
hip_linear_attn_qkv_prepare_shuffle_prototype_kernel_cache() {
    static HipLinearAttnQkvPrepareShufflePrototypeKernelCache cache;
    return cache;
}

bool linear_attn_qkv_prepare_shuffle_prototype_f32_hip_kernel(
    const ullm_runtime_buffer *qkv_buffer,
    const ullm_runtime_buffer *conv_weight_buffer,
    ullm_runtime_buffer *conv_history_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    float q_scale,
    bool qk_l2_norm,
    ullm_runtime_buffer *conv_output_buffer,
    ullm_runtime_buffer *q_output_buffer,
    ullm_runtime_buffer *k_output_buffer,
    ullm_runtime_buffer *v_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = qkv_buffer->hip_device_id;
    if (!linear_attn_qkv_prepare_shuffle_prototype_shape_is_supported(
            hip_runtime().device_arch_name(device_id),
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size)) {
        if (error != nullptr) {
            *error =
                "linear attention qkv prepare shuffle prototype requires gfx1201 and Qwen3.5 16/32 heads, 128/128 dims, kernel size 4";
        }
        return false;
    }
    void *function = hip_linear_attn_qkv_prepare_shuffle_prototype_kernel_cache()
                         .function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    void *qkv_ptr = qkv_buffer->ptr;
    void *conv_weight_ptr = conv_weight_buffer->ptr;
    void *conv_history_ptr = conv_history_buffer->ptr;
    unsigned long long kernel_key_heads = static_cast<unsigned long long>(key_heads);
    unsigned long long kernel_value_heads = static_cast<unsigned long long>(value_heads);
    unsigned long long kernel_key_dim = static_cast<unsigned long long>(key_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    unsigned long long kernel_size_arg = static_cast<unsigned long long>(kernel_size);
    int kernel_qk_l2_norm = qk_l2_norm ? 1 : 0;
    void *conv_output_ptr = conv_output_buffer->ptr;
    void *q_output_ptr = q_output_buffer->ptr;
    void *k_output_ptr = k_output_buffer->ptr;
    void *v_output_ptr = v_output_buffer->ptr;
    void *kernel_params[] = {
        &qkv_ptr,
        &conv_weight_ptr,
        &conv_history_ptr,
        &kernel_key_heads,
        &kernel_value_heads,
        &kernel_key_dim,
        &kernel_value_dim,
        &kernel_size_arg,
        &q_scale,
        &kernel_qk_l2_norm,
        &conv_output_ptr,
        &q_output_ptr,
        &k_output_ptr,
        &v_output_ptr,
    };
    // 32 Q/K-normalizing CTAs plus ceil(32 * 128 / 256) == 16 reduction-free V CTAs.
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            kLinearAttnQkvPrepareShufflePrototypeGrid,
            kLinearAttnQkvPrepareShufflePrototypeThreads,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for linear attention qkv prepare shuffle prototype";
        }
        return false;
    }
    return true;
}

class HipLinearAttnQkvPrepareBatchKernelCache {
public:
    bool functions_for_device(
        int device_id,
        void **prepare_function,
        void **update_history_function,
        std::string *error) {
        if (prepare_function == nullptr || update_history_function == nullptr) {
            append_error(error, "linear attention qkv prepare batch received null function output");
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            *prepare_function = found->second->prepare_function;
            *update_history_function = found->second->update_history_function;
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
            if (!hiprtc_runtime().compile_linear_attn_qkv_prepare_batch_kernel(arch, &code, &compile_error)) {
                append_error(&compile_errors, compile_error);
                continue;
            }

            void *module = nullptr;
            if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
                append_error(&compile_errors, "hipModuleLoadData failed for " + arch);
                continue;
            }
            void *prepare = nullptr;
            if (!hip_runtime().module_get_function(
                    &prepare,
                    module,
                    "ullm_linear_attn_qkv_prepare_batch_f32_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(
                    &compile_errors,
                    "hipModuleGetFunction qkv prepare batch failed for " + arch);
                continue;
            }
            void *update_history = nullptr;
            if (!hip_runtime().module_get_function(
                    &update_history,
                    module,
                    "ullm_linear_attn_qkv_prepare_batch_update_history_f32_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(
                    &compile_errors,
                    "hipModuleGetFunction qkv prepare batch history update failed for " + arch);
                continue;
            }

            auto loaded = std::make_unique<LoadedModule>();
            loaded->module = module;
            loaded->prepare_function = prepare;
            loaded->update_history_function = update_history;
            loaded->arch = arch;
            *prepare_function = loaded->prepare_function;
            *update_history_function = loaded->update_history_function;
            modules_.emplace(device_id, std::move(loaded));
            return true;
        }
        append_error(
            error,
            compile_errors.empty() ? "failed to build linear attention qkv prepare batch HIP kernel"
                                   : compile_errors);
        return false;
    }

private:
    struct LoadedModule {
        void *module = nullptr;
        void *prepare_function = nullptr;
        void *update_history_function = nullptr;
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

HipLinearAttnQkvPrepareBatchKernelCache &hip_linear_attn_qkv_prepare_batch_kernel_cache() {
    static HipLinearAttnQkvPrepareBatchKernelCache cache;
    return cache;
}

bool linear_attn_qkv_prepare_f32_hip_kernel(
    const ullm_runtime_buffer *qkv_buffer,
    const ullm_runtime_buffer *conv_weight_buffer,
    ullm_runtime_buffer *conv_history_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    float q_scale,
    bool qk_l2_norm,
    ullm_runtime_buffer *conv_output_buffer,
    ullm_runtime_buffer *q_output_buffer,
    ullm_runtime_buffer *k_output_buffer,
    ullm_runtime_buffer *v_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = qkv_buffer->hip_device_id;
    void *function = hip_linear_attn_qkv_prepare_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t v_elements = value_heads * value_dim;
    const size_t v_grid_size = (v_elements + block_size - 1) / block_size;
    const size_t grid_size = key_heads * 2 + v_grid_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "linear attention qkv prepare grid size exceeds HIP limit";
        }
        return false;
    }

    void *qkv_ptr = qkv_buffer->ptr;
    void *conv_weight_ptr = conv_weight_buffer->ptr;
    void *conv_history_ptr = conv_history_buffer->ptr;
    unsigned long long kernel_key_heads = static_cast<unsigned long long>(key_heads);
    unsigned long long kernel_value_heads = static_cast<unsigned long long>(value_heads);
    unsigned long long kernel_key_dim = static_cast<unsigned long long>(key_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    unsigned long long kernel_size_arg = static_cast<unsigned long long>(kernel_size);
    int kernel_qk_l2_norm = qk_l2_norm ? 1 : 0;
    void *conv_output_ptr = conv_output_buffer->ptr;
    void *q_output_ptr = q_output_buffer->ptr;
    void *k_output_ptr = k_output_buffer->ptr;
    void *v_output_ptr = v_output_buffer->ptr;
    void *kernel_params[] = {
        &qkv_ptr,
        &conv_weight_ptr,
        &conv_history_ptr,
        &kernel_key_heads,
        &kernel_value_heads,
        &kernel_key_dim,
        &kernel_value_dim,
        &kernel_size_arg,
        &q_scale,
        &kernel_qk_l2_norm,
        &conv_output_ptr,
        &q_output_ptr,
        &k_output_ptr,
        &v_output_ptr,
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
            *error = "hipModuleLaunchKernel failed for linear attention qkv prepare";
        }
        return false;
    }
    return true;
}

bool linear_attn_qkv_prepare_batch_f32_hip_kernel(
    const ullm_runtime_buffer *qkv_buffer,
    const ullm_runtime_buffer *conv_weight_buffer,
    ullm_runtime_buffer *conv_history_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    size_t sequence_len,
    float q_scale,
    bool qk_l2_norm,
    ullm_runtime_buffer *conv_output_buffer,
    ullm_runtime_buffer *q_output_buffer,
    ullm_runtime_buffer *k_output_buffer,
    ullm_runtime_buffer *v_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = qkv_buffer->hip_device_id;
    void *prepare_function = nullptr;
    void *update_history_function = nullptr;
    if (!hip_linear_attn_qkv_prepare_batch_kernel_cache().functions_for_device(
            device_id,
            &prepare_function,
            &update_history_function,
            error)) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t q_elements = key_heads * key_dim;
    const size_t v_elements = value_heads * value_dim;
    const size_t channels = q_elements * 2 + v_elements;
    const size_t v_grid_size = (v_elements + block_size - 1) / block_size;
    const size_t prepare_grid_x = key_heads * 2 + v_grid_size;
    const size_t update_grid_x = (channels + block_size - 1) / block_size;
    if (prepare_grid_x > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        update_grid_x > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        sequence_len > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "linear attention qkv prepare batch grid size exceeds HIP limit";
        }
        return false;
    }

    void *qkv_ptr = qkv_buffer->ptr;
    void *conv_weight_ptr = conv_weight_buffer->ptr;
    void *conv_history_ptr = conv_history_buffer->ptr;
    unsigned long long kernel_key_heads = static_cast<unsigned long long>(key_heads);
    unsigned long long kernel_value_heads = static_cast<unsigned long long>(value_heads);
    unsigned long long kernel_key_dim = static_cast<unsigned long long>(key_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    unsigned long long kernel_size_arg = static_cast<unsigned long long>(kernel_size);
    unsigned long long kernel_sequence_len = static_cast<unsigned long long>(sequence_len);
    int kernel_qk_l2_norm = qk_l2_norm ? 1 : 0;
    void *conv_output_ptr = conv_output_buffer->ptr;
    void *q_output_ptr = q_output_buffer->ptr;
    void *k_output_ptr = k_output_buffer->ptr;
    void *v_output_ptr = v_output_buffer->ptr;
    void *prepare_params[] = {
        &qkv_ptr,
        &conv_weight_ptr,
        &conv_history_ptr,
        &kernel_key_heads,
        &kernel_value_heads,
        &kernel_key_dim,
        &kernel_value_dim,
        &kernel_size_arg,
        &kernel_sequence_len,
        &q_scale,
        &kernel_qk_l2_norm,
        &conv_output_ptr,
        &q_output_ptr,
        &k_output_ptr,
        &v_output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel_2d(
            prepare_function,
            static_cast<unsigned int>(prepare_grid_x),
            static_cast<unsigned int>(sequence_len),
            block_size,
            prepare_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for linear attention qkv prepare batch";
        }
        return false;
    }

    unsigned long long kernel_channels = static_cast<unsigned long long>(channels);
    void *update_params[] = {
        &qkv_ptr,
        &conv_history_ptr,
        &kernel_sequence_len,
        &kernel_channels,
        &kernel_size_arg,
    };
    if (!hip_runtime().module_launch_kernel(
            update_history_function,
            static_cast<unsigned int>(update_grid_x),
            block_size,
            update_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for linear attention qkv prepare batch history update";
        }
        return false;
    }
    return true;
}

ullm_status linear_attn_qkv_prepare_f32_hip_staging(
    const ullm_runtime_buffer *qkv_buffer,
    const ullm_runtime_buffer *conv_weight_buffer,
    ullm_runtime_buffer *conv_history_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    float q_scale,
    bool qk_l2_norm,
    size_t qkv_bytes,
    size_t conv_history_bytes,
    size_t q_bytes,
    size_t k_bytes,
    size_t v_bytes,
    ullm_runtime_buffer *conv_output_buffer,
    ullm_runtime_buffer *q_output_buffer,
    ullm_runtime_buffer *k_output_buffer,
    ullm_runtime_buffer *v_output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_qkv(qkv_bytes / sizeof(float));
    std::vector<float> host_conv_weight(conv_history_bytes / sizeof(float));
    std::vector<float> host_conv_history(conv_history_bytes / sizeof(float));
    std::vector<float> host_conv_output(qkv_bytes / sizeof(float));
    std::vector<float> host_q(q_bytes / sizeof(float));
    std::vector<float> host_k(k_bytes / sizeof(float));
    std::vector<float> host_v(v_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = qkv_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_qkv.data(),
            qkv_buffer->ptr,
            qkv_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_conv_weight.data(),
            conv_weight_buffer->ptr,
            conv_history_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_conv_history.data(),
            conv_history_buffer->ptr,
            conv_history_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy linear attention qkv prepare HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize linear attention qkv prepare HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    linear_attn_qkv_prepare_f32_host(
        host_qkv.data(),
        host_conv_weight.data(),
        host_conv_history.data(),
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        kernel_size,
        q_scale,
        qk_l2_norm,
        host_conv_output.data(),
        host_q.data(),
        host_k.data(),
        host_v.data());

    if (!hip_runtime().copy_async(
            conv_history_buffer->ptr,
            host_conv_history.data(),
            conv_history_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            conv_output_buffer->ptr,
            host_conv_output.data(),
            qkv_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            q_output_buffer->ptr,
            host_q.data(),
            q_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            k_output_buffer->ptr,
            host_k.data(),
            k_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            v_output_buffer->ptr,
            host_v.data(),
            v_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy linear attention qkv prepare outputs to HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize linear attention qkv prepare HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status linear_attn_qkv_prepare_batch_f32_hip_staging(
    const ullm_runtime_buffer *qkv_buffer,
    const ullm_runtime_buffer *conv_weight_buffer,
    ullm_runtime_buffer *conv_history_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    size_t sequence_len,
    float q_scale,
    bool qk_l2_norm,
    size_t qkv_bytes,
    size_t conv_history_bytes,
    size_t q_bytes,
    size_t k_bytes,
    size_t v_bytes,
    ullm_runtime_buffer *conv_output_buffer,
    ullm_runtime_buffer *q_output_buffer,
    ullm_runtime_buffer *k_output_buffer,
    ullm_runtime_buffer *v_output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_qkv(qkv_bytes / sizeof(float));
    std::vector<float> host_conv_weight(conv_history_bytes / sizeof(float));
    std::vector<float> host_conv_history(conv_history_bytes / sizeof(float));
    std::vector<float> host_conv_output(qkv_bytes / sizeof(float));
    std::vector<float> host_q(q_bytes / sizeof(float));
    std::vector<float> host_k(k_bytes / sizeof(float));
    std::vector<float> host_v(v_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = qkv_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_qkv.data(),
            qkv_buffer->ptr,
            qkv_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_conv_weight.data(),
            conv_weight_buffer->ptr,
            conv_history_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_conv_history.data(),
            conv_history_buffer->ptr,
            conv_history_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy linear attention qkv prepare batch HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize linear attention qkv prepare batch HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    linear_attn_qkv_prepare_batch_f32_host(
        host_qkv.data(),
        host_conv_weight.data(),
        host_conv_history.data(),
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        kernel_size,
        sequence_len,
        q_scale,
        qk_l2_norm,
        host_conv_output.data(),
        host_q.data(),
        host_k.data(),
        host_v.data());

    if (!hip_runtime().copy_async(
            conv_history_buffer->ptr,
            host_conv_history.data(),
            conv_history_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            conv_output_buffer->ptr,
            host_conv_output.data(),
            qkv_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            q_output_buffer->ptr,
            host_q.data(),
            q_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            k_output_buffer->ptr,
            host_k.data(),
            k_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            v_output_buffer->ptr,
            host_v.data(),
            v_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy linear attention qkv prepare batch outputs to HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize linear attention qkv prepare batch HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

void linear_attn_gate_beta_f32_host(
    const float *a,
    const float *b,
    const float *a_log,
    const float *dt_bias,
    size_t heads,
    size_t sequence_len,
    float *gate_output,
    float *beta_output) {
    for (size_t timestep = 0; timestep < sequence_len; ++timestep) {
        for (size_t head = 0; head < heads; ++head) {
            const size_t index = timestep * heads + head;
            const float x = a[index] + dt_bias[head];
            const float softplus = x <= 20.0f ? std::log1p(std::exp(x)) : x;
            gate_output[index] = -std::exp(a_log[head]) * softplus;
            const float b_value = b[index];
            beta_output[index] = 1.0f / (1.0f + std::exp(-b_value));
        }
    }
}

class HipLinearAttnGateBetaKernelCache {
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
            if (!hiprtc_runtime().compile_linear_attn_gate_beta_kernel(arch, &code, &compile_error)) {
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
                    "ullm_linear_attn_gate_beta_f32_kernel",
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
            compile_errors.empty() ? "failed to build linear attention gate beta HIP kernel" :
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

HipLinearAttnGateBetaKernelCache &hip_linear_attn_gate_beta_kernel_cache() {
    static HipLinearAttnGateBetaKernelCache cache;
    return cache;
}

bool linear_attn_gate_beta_f32_hip_kernel(
    const ullm_runtime_buffer *a_buffer,
    const ullm_runtime_buffer *b_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t heads,
    size_t sequence_len,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = a_buffer->hip_device_id;
    void *function = hip_linear_attn_gate_beta_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t elements = heads * sequence_len;
    const size_t grid_size = (elements + block_size - 1) / block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "linear attention gate beta element count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_heads = static_cast<unsigned long long>(heads);
    unsigned long long kernel_sequence_len = static_cast<unsigned long long>(sequence_len);
    void *a_ptr = a_buffer->ptr;
    void *b_ptr = b_buffer->ptr;
    void *a_log_ptr = a_log_buffer->ptr;
    void *dt_bias_ptr = dt_bias_buffer->ptr;
    void *gate_output_ptr = gate_output_buffer->ptr;
    void *beta_output_ptr = beta_output_buffer->ptr;
    void *kernel_params[] = {
        &a_ptr,
        &b_ptr,
        &a_log_ptr,
        &dt_bias_ptr,
        &kernel_heads,
        &kernel_sequence_len,
        &gate_output_ptr,
        &beta_output_ptr,
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
            *error = "hipModuleLaunchKernel failed for f32 linear attention gate beta";
        }
        return false;
    }
    return true;
}

ullm_status linear_attn_gate_beta_f32_hip_staging(
    const ullm_runtime_buffer *a_buffer,
    const ullm_runtime_buffer *b_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t heads,
    size_t sequence_len,
    size_t sequence_bytes,
    size_t head_bytes,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_a(sequence_bytes / sizeof(float));
    std::vector<float> host_b(sequence_bytes / sizeof(float));
    std::vector<float> host_a_log(head_bytes / sizeof(float));
    std::vector<float> host_dt_bias(head_bytes / sizeof(float));
    std::vector<float> host_gate(sequence_bytes / sizeof(float));
    std::vector<float> host_beta(sequence_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = a_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_a.data(),
            a_buffer->ptr,
            sequence_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_b.data(),
            b_buffer->ptr,
            sequence_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_a_log.data(),
            a_log_buffer->ptr,
            head_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_dt_bias.data(),
            dt_bias_buffer->ptr,
            head_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 linear attention gate beta HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 linear attention gate beta HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    linear_attn_gate_beta_f32_host(
        host_a.data(),
        host_b.data(),
        host_a_log.data(),
        host_dt_bias.data(),
        heads,
        sequence_len,
        host_gate.data(),
        host_beta.data());
    if (!hip_runtime().copy_async(
            gate_output_buffer->ptr,
            host_gate.data(),
            sequence_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            beta_output_buffer->ptr,
            host_beta.data(),
            sequence_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 linear attention gate beta outputs to HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 linear attention gate beta HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

void linear_attn_recurrent_f32_host(
    const float *q,
    const float *k,
    const float *v,
    const float *gate,
    const float *beta,
    size_t key_heads,
    size_t value_heads,
    size_t sequence_len,
    size_t key_dim,
    size_t value_dim,
    float *state,
    float *output) {
    const size_t key_head_group = value_heads / key_heads;
    for (size_t timestep = 0; timestep < sequence_len; ++timestep) {
        for (size_t value_head = 0; value_head < value_heads; ++value_head) {
            const size_t key_head = value_head / key_head_group;
            const size_t value_head_index = timestep * value_heads + value_head;
            const size_t key_head_index = timestep * key_heads + key_head;
            const size_t qk_base = key_head_index * key_dim;
            const size_t v_base = value_head_index * value_dim;
            const size_t state_head_offset = value_head * key_dim * value_dim;
            const float decay = std::exp(gate[value_head_index]);
            const float beta_value = beta[value_head_index];

            for (size_t key = 0; key < key_dim; ++key) {
                const size_t state_key_offset = state_head_offset + key * value_dim;
                for (size_t value = 0; value < value_dim; ++value) {
                    state[state_key_offset + value] *= decay;
                }
            }

            for (size_t value = 0; value < value_dim; ++value) {
                float current = 0.0f;
                for (size_t key = 0; key < key_dim; ++key) {
                    current += state[state_head_offset + key * value_dim + value] *
                               k[qk_base + key];
                }
                const float v_prime = (v[v_base + value] - current) * beta_value;
                for (size_t key = 0; key < key_dim; ++key) {
                    state[state_head_offset + key * value_dim + value] +=
                        k[qk_base + key] * v_prime;
                }
            }

            for (size_t value = 0; value < value_dim; ++value) {
                float sum = 0.0f;
                for (size_t key = 0; key < key_dim; ++key) {
                    sum += state[state_head_offset + key * value_dim + value] *
                           q[qk_base + key];
                }
                output[v_base + value] = sum;
            }
        }
    }
}

