// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
class HipLinearAttnRecurrentGenericKernelCache {
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
            if (!hiprtc_runtime().compile_linear_attn_recurrent_generic_kernel(
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
                    "ullm_linear_attn_recurrent_f32_kernel",
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
            compile_errors.empty() ? "failed to build linear attention recurrent HIP kernel" :
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

HipLinearAttnRecurrentGenericKernelCache &hip_linear_attn_recurrent_generic_kernel_cache() {
    static HipLinearAttnRecurrentGenericKernelCache cache;
    return cache;
}

constexpr size_t kLinearAttnRecurrentShuffleKeyHeads = 16u;
constexpr size_t kLinearAttnRecurrentShuffleValueHeads = 32u;
constexpr size_t kLinearAttnRecurrentShuffleKeyDim = 128u;
constexpr size_t kLinearAttnRecurrentShuffleValueDim = 128u;
constexpr unsigned int kLinearAttnRecurrentShuffleThreads = 128u;

bool linear_attn_recurrent_shuffle_shape_is_supported(
    const std::string &arch,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim) {
    return arch == "gfx1201" && key_heads == kLinearAttnRecurrentShuffleKeyHeads &&
        value_heads == kLinearAttnRecurrentShuffleValueHeads &&
        key_dim == kLinearAttnRecurrentShuffleKeyDim &&
        value_dim == kLinearAttnRecurrentShuffleValueDim;
}

// Keep the exact-shape shuffle implementation in a separate HIPRTC module. Mixing the generic
// fallback into this source changes its register/LDS allocation, whereas both modules retain the
// stable ullm_linear_attn_recurrent_f32_kernel entry point and stable public ABI.
class HipLinearAttnRecurrentShuffleKernelCache {
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
            return fail(
                device_id,
                "linear attention recurrent shuffle requires exact gfx1201 architecture",
                error);
        }

        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_linear_attn_recurrent_kernel(
                "gfx1201",
                &code,
                &compile_error)) {
            return fail(
                device_id,
                compile_error.empty() ?
                    "failed to build linear attention recurrent shuffle HIP kernel" :
                    compile_error,
                error);
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            return fail(
                device_id,
                "hipModuleLoadData failed for linear attention recurrent shuffle",
                error);
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function,
                module,
                "ullm_linear_attn_recurrent_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            return fail(
                device_id,
                "hipModuleGetFunction failed for linear attention recurrent shuffle",
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

HipLinearAttnRecurrentShuffleKernelCache &hip_linear_attn_recurrent_shuffle_kernel_cache() {
    static HipLinearAttnRecurrentShuffleKernelCache cache;
    return cache;
}

bool linear_attn_recurrent_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *beta_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t sequence_len,
    size_t key_dim,
    size_t value_dim,
    ullm_runtime_buffer *state_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    if (value_heads > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "linear attention recurrent value head count exceeds HIP grid limit";
        }
        return false;
    }

    unsigned long long kernel_key_heads = static_cast<unsigned long long>(key_heads);
    unsigned long long kernel_value_heads = static_cast<unsigned long long>(value_heads);
    unsigned long long kernel_sequence_len = static_cast<unsigned long long>(sequence_len);
    unsigned long long kernel_key_dim = static_cast<unsigned long long>(key_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *q_ptr = q_buffer->ptr;
    void *k_ptr = k_buffer->ptr;
    void *v_ptr = v_buffer->ptr;
    void *gate_ptr = gate_buffer->ptr;
    void *beta_ptr = beta_buffer->ptr;
    void *state_ptr = state_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr,
        &k_ptr,
        &v_ptr,
        &gate_ptr,
        &beta_ptr,
        &kernel_key_heads,
        &kernel_value_heads,
        &kernel_sequence_len,
        &kernel_key_dim,
        &kernel_value_dim,
        &state_ptr,
        &output_ptr,
    };
    if (value_dim != 0 &&
        value_heads > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) / value_dim) {
        if (error != nullptr) {
            *error = "linear attention recurrent grid exceeds HIP grid limit";
        }
        return false;
    }
    unsigned int grid_size = static_cast<unsigned int>(value_heads * value_dim);
    const unsigned int default_block = key_dim <= 128 ? 128u : 256u;
    unsigned int block_size = block_size_from_env(
        "ULLM_LINEAR_ATTN_RECURRENT_BLOCK",
        block_size_from_env("ULLM_LINEAR_ATTN_RECURRENT_DECODE_BLOCK", default_block));

    void *function = nullptr;
    if (block_size == kLinearAttnRecurrentShuffleThreads &&
        linear_attn_recurrent_shuffle_shape_is_supported(
            hip_runtime().device_arch_name(device_id),
            key_heads,
            value_heads,
            key_dim,
            value_dim)) {
        // Keep the generic module as a safe stable-ABI fallback if the target-specific code
        // object cannot be built or loaded on a particular HIP installation.
        std::string shuffle_error;
        function = hip_linear_attn_recurrent_shuffle_kernel_cache().function_for_device(
            device_id,
            &shuffle_error);
        if (function == nullptr) {
            function = hip_linear_attn_recurrent_generic_kernel_cache().function_for_device(
                device_id,
                error);
            if (function == nullptr && !shuffle_error.empty()) {
                if (error != nullptr && !error->empty()) {
                    error->append("\n");
                }
                if (error != nullptr) {
                    error->append(shuffle_error);
                }
            }
        }
    } else {
        function = hip_linear_attn_recurrent_generic_kernel_cache().function_for_device(
            device_id,
            error);
    }
    if (function == nullptr) {
        return false;
    }

    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            grid_size,
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 linear attention recurrent";
        }
        return false;
    }
    return true;
}

ullm_status linear_attn_recurrent_f32_hip_staging(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *beta_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t sequence_len,
    size_t key_dim,
    size_t value_dim,
    size_t qk_bytes,
    size_t v_bytes,
    size_t gate_beta_bytes,
    size_t state_bytes,
    ullm_runtime_buffer *state_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_q(qk_bytes / sizeof(float));
    std::vector<float> host_k(qk_bytes / sizeof(float));
    std::vector<float> host_v(v_bytes / sizeof(float));
    std::vector<float> host_gate(gate_beta_bytes / sizeof(float));
    std::vector<float> host_beta(gate_beta_bytes / sizeof(float));
    std::vector<float> host_state(state_bytes / sizeof(float));
    std::vector<float> host_output(v_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = q_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_q.data(),
            q_buffer->ptr,
            qk_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_k.data(),
            k_buffer->ptr,
            qk_bytes,
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
            host_gate.data(),
            gate_buffer->ptr,
            gate_beta_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_beta.data(),
            beta_buffer->ptr,
            gate_beta_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_state.data(),
            state_buffer->ptr,
            state_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 linear attention recurrent HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 linear attention recurrent HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    linear_attn_recurrent_f32_host(
        host_q.data(),
        host_k.data(),
        host_v.data(),
        host_gate.data(),
        host_beta.data(),
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        host_state.data(),
        host_output.data());
    if (!hip_runtime().copy_async(
            state_buffer->ptr,
            host_state.data(),
            state_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            v_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 linear attention recurrent outputs to HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 linear attention recurrent HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

