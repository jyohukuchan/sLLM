
ullm_status sigmoid_mul_f32_hip_staging(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t elements,
    size_t required_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_gate(elements);
    std::vector<float> host_input(elements);
    std::vector<float> host_output(elements);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = gate_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_gate.data(),
            gate_buffer->ptr,
            required_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            required_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 sigmoid-mul HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 sigmoid-mul HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    sigmoid_mul_f32_host(host_gate.data(), host_input.data(), elements, host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 sigmoid-mul output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 sigmoid-mul HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipQwen35SplitQGateKernelCache {
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
            if (!hiprtc_runtime().compile_qwen35_split_q_gate_kernel(arch, &code, &compile_error)) {
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
                    "ullm_qwen35_split_q_gate_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build Qwen3.5 q/gate split HIP kernel"
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

HipQwen35SplitQGateKernelCache &hip_qwen35_split_q_gate_kernel_cache() {
    static HipQwen35SplitQGateKernelCache cache;
    return cache;
}

bool qwen35_split_q_gate_f32_hip_kernel(
    const ullm_runtime_buffer *projected_buffer,
    size_t q_heads,
    size_t head_dim,
    ullm_runtime_buffer *query_output_buffer,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = projected_buffer->hip_device_id;
    void *function = hip_qwen35_split_q_gate_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const size_t elements = q_heads * head_dim;
    constexpr unsigned int block_size = 256;
    const size_t grid_size = (elements + block_size - 1) / block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "Qwen3.5 q/gate split element count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    void *projected_ptr = projected_buffer->ptr;
    void *query_output_ptr = query_output_buffer->ptr;
    void *gate_output_ptr = gate_output_buffer->ptr;
    void *kernel_params[] = {
        &projected_ptr,
        &kernel_q_heads,
        &kernel_head_dim,
        &query_output_ptr,
        &gate_output_ptr,
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
            *error = "hipModuleLaunchKernel failed for Qwen3.5 q/gate split";
        }
        return false;
    }
    return true;
}

ullm_status qwen35_split_q_gate_f32_hip_staging(
    const ullm_runtime_buffer *projected_buffer,
    size_t q_heads,
    size_t head_dim,
    size_t projected_bytes,
    size_t output_bytes,
    ullm_runtime_buffer *query_output_buffer,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_stream *stream) {
    const size_t projected_elements = q_heads * head_dim * 2;
    const size_t output_elements = q_heads * head_dim;
    std::vector<float> host_projected(projected_elements);
    std::vector<float> host_query(output_elements);
    std::vector<float> host_gate(output_elements);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = projected_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_projected.data(),
            projected_buffer->ptr,
            projected_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy Qwen3.5 q/gate split HIP input to host staging buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize Qwen3.5 q/gate split HIP input staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    qwen35_split_q_gate_f32_host(
        host_projected.data(),
        q_heads,
        head_dim,
        host_query.data(),
        host_gate.data());
    if (!hip_runtime().copy_async(
            query_output_buffer->ptr,
            host_query.data(),
            output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            gate_output_buffer->ptr,
            host_gate.data(),
            output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy Qwen3.5 q/gate split HIP outputs");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize Qwen3.5 q/gate split HIP output staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipQwen35QkNormRopeKernelCache {
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
            if (!hiprtc_runtime().compile_qwen35_qk_norm_rope_kernel(
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
                    "ullm_qwen35_qk_norm_rope_f32_kernel",
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
        append_error(error, compile_errors.empty()
                                ? "failed to build Qwen3.5 q/k norm RoPE HIP kernel"
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

HipQwen35QkNormRopeKernelCache &hip_qwen35_qk_norm_rope_kernel_cache() {
    static HipQwen35QkNormRopeKernelCache cache;
    return cache;
}

bool qwen35_qk_norm_rope_f32_hip_kernel(
    const ullm_runtime_buffer *q_projected_buffer,
    const ullm_runtime_buffer *k_projected_buffer,
    const ullm_runtime_buffer *q_weight_buffer,
    const ullm_runtime_buffer *k_weight_buffer,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    ullm_runtime_buffer *q_gate_output_buffer,
    ullm_runtime_buffer *q_rope_output_buffer,
    ullm_runtime_buffer *k_rope_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_projected_buffer->hip_device_id;
    void *function = hip_qwen35_qk_norm_rope_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const size_t segments = q_heads + kv_heads;
    if (segments > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "Qwen3.5 q/k norm RoPE segment count exceeds HIP grid limit";
        }
        return false;
    }
    constexpr unsigned int block_size = 256;
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_rotary_dim = static_cast<unsigned long long>(rotary_dim);
    unsigned long long kernel_position_offset = static_cast<unsigned long long>(position_offset);
    void *q_projected_ptr = q_projected_buffer->ptr;
    void *k_projected_ptr = k_projected_buffer->ptr;
    void *q_weight_ptr = q_weight_buffer->ptr;
    void *k_weight_ptr = k_weight_buffer->ptr;
    void *q_gate_output_ptr = q_gate_output_buffer->ptr;
    void *q_rope_output_ptr = q_rope_output_buffer->ptr;
    void *k_rope_output_ptr = k_rope_output_buffer->ptr;
    void *kernel_params[] = {
        &q_projected_ptr,
        &k_projected_ptr,
        &q_weight_ptr,
        &k_weight_ptr,
        &kernel_q_heads,
        &kernel_kv_heads,
        &kernel_head_dim,
        &kernel_rotary_dim,
        &kernel_position_offset,
        &rope_base,
        &epsilon,
        &q_gate_output_ptr,
        &q_rope_output_ptr,
        &k_rope_output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(segments),
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for Qwen3.5 q/k norm RoPE";
        }
        return false;
    }
    return true;
}

ullm_status qwen35_qk_norm_rope_f32_hip_staging(
    const ullm_runtime_buffer *q_projected_buffer,
    const ullm_runtime_buffer *k_projected_buffer,
    const ullm_runtime_buffer *q_weight_buffer,
    const ullm_runtime_buffer *k_weight_buffer,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    size_t q_projected_bytes,
    size_t k_projected_bytes,
    size_t weight_bytes,
    size_t q_output_bytes,
    size_t k_output_bytes,
    ullm_runtime_buffer *q_gate_output_buffer,
    ullm_runtime_buffer *q_rope_output_buffer,
    ullm_runtime_buffer *k_rope_output_buffer,
    ullm_runtime_stream *stream) {
    const size_t q_projected_elements = q_heads * head_dim * 2;
    const size_t q_output_elements = q_heads * head_dim;
    const size_t k_output_elements = kv_heads * head_dim;
    std::vector<float> host_q_projected(q_projected_elements);
    std::vector<float> host_k_projected(k_output_elements);
    std::vector<float> host_q_weight(head_dim);
    std::vector<float> host_k_weight(head_dim);
    std::vector<float> host_q_gate(q_output_elements);
    std::vector<float> host_q_rope(q_output_elements);
    std::vector<float> host_k_rope(k_output_elements);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = q_projected_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_q_projected.data(),
            q_projected_buffer->ptr,
            q_projected_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_k_projected.data(),
            k_projected_buffer->ptr,
            k_projected_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_q_weight.data(),
            q_weight_buffer->ptr,
            weight_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_k_weight.data(),
            k_weight_buffer->ptr,
            weight_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy Qwen3.5 q/k norm RoPE HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize Qwen3.5 q/k norm RoPE HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    qwen35_qk_norm_rope_f32_host(
        host_q_projected.data(),
        host_k_projected.data(),
        host_q_weight.data(),
        host_k_weight.data(),
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        epsilon,
        host_q_gate.data(),
        host_q_rope.data(),
        host_k_rope.data());
    if (!hip_runtime().copy_async(
            q_gate_output_buffer->ptr,
            host_q_gate.data(),
            q_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            q_rope_output_buffer->ptr,
            host_q_rope.data(),
            q_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            k_rope_output_buffer->ptr,
            host_k_rope.data(),
            k_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy Qwen3.5 q/k norm RoPE HIP outputs");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize Qwen3.5 q/k norm RoPE HIP output staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipQwen35QkNormRopeBatchKernelCache {
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
            if (!hiprtc_runtime().compile_qwen35_qk_norm_rope_batch_kernel(
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
                    "ullm_qwen35_qk_norm_rope_batch_f32_kernel",
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
        append_error(error, compile_errors.empty()
                                ? "failed to build Qwen3.5 q/k norm RoPE batch HIP kernel"
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

HipQwen35QkNormRopeBatchKernelCache &hip_qwen35_qk_norm_rope_batch_kernel_cache() {
    static HipQwen35QkNormRopeBatchKernelCache cache;
    return cache;
}

bool qwen35_qk_norm_rope_batch_f32_hip_kernel(
    const ullm_runtime_buffer *q_projected_buffer,
    const ullm_runtime_buffer *k_projected_buffer,
    const ullm_runtime_buffer *q_weight_buffer,
    const ullm_runtime_buffer *k_weight_buffer,
    size_t q_heads,
    size_t kv_heads,
    size_t sequence_len,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    ullm_runtime_buffer *q_gate_output_buffer,
    ullm_runtime_buffer *q_rope_output_buffer,
    ullm_runtime_buffer *k_rope_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_projected_buffer->hip_device_id;
    void *function =
        hip_qwen35_qk_norm_rope_batch_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const size_t segments_per_token = q_heads + kv_heads;
    if (sequence_len > std::numeric_limits<size_t>::max() / segments_per_token) {
        if (error != nullptr) {
            *error = "Qwen3.5 q/k norm RoPE batch segment count overflows";
        }
        return false;
    }
    const size_t segments = sequence_len * segments_per_token;
    if (segments > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "Qwen3.5 q/k norm RoPE batch segment count exceeds HIP grid limit";
        }
        return false;
    }
    constexpr unsigned int block_size = 256;
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_sequence_len = static_cast<unsigned long long>(sequence_len);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_rotary_dim = static_cast<unsigned long long>(rotary_dim);
    unsigned long long kernel_position_offset = static_cast<unsigned long long>(position_offset);
    void *q_projected_ptr = q_projected_buffer->ptr;
    void *k_projected_ptr = k_projected_buffer->ptr;
    void *q_weight_ptr = q_weight_buffer->ptr;
    void *k_weight_ptr = k_weight_buffer->ptr;
    void *q_gate_output_ptr = q_gate_output_buffer->ptr;
    void *q_rope_output_ptr = q_rope_output_buffer->ptr;
    void *k_rope_output_ptr = k_rope_output_buffer->ptr;
    void *kernel_params[] = {
        &q_projected_ptr,
        &k_projected_ptr,
        &q_weight_ptr,
        &k_weight_ptr,
        &kernel_q_heads,
        &kernel_kv_heads,
        &kernel_sequence_len,
        &kernel_head_dim,
        &kernel_rotary_dim,
        &kernel_position_offset,
        &rope_base,
        &epsilon,
        &q_gate_output_ptr,
        &q_rope_output_ptr,
        &k_rope_output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(segments),
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for Qwen3.5 q/k norm RoPE batch";
        }
        return false;
    }
    return true;
}

ullm_status qwen35_qk_norm_rope_batch_f32_hip_staging(
    const ullm_runtime_buffer *q_projected_buffer,
    const ullm_runtime_buffer *k_projected_buffer,
    const ullm_runtime_buffer *q_weight_buffer,
    const ullm_runtime_buffer *k_weight_buffer,
    size_t q_heads,
    size_t kv_heads,
    size_t sequence_len,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    size_t q_projected_bytes,
    size_t k_projected_bytes,
    size_t weight_bytes,
    size_t q_output_bytes,
    size_t k_output_bytes,
    ullm_runtime_buffer *q_gate_output_buffer,
    ullm_runtime_buffer *q_rope_output_buffer,
    ullm_runtime_buffer *k_rope_output_buffer,
    ullm_runtime_stream *stream) {
    const size_t q_projected_elements = sequence_len * q_heads * head_dim * 2;
    const size_t q_output_elements = sequence_len * q_heads * head_dim;
    const size_t k_output_elements = sequence_len * kv_heads * head_dim;
    std::vector<float> host_q_projected(q_projected_elements);
    std::vector<float> host_k_projected(k_output_elements);
    std::vector<float> host_q_weight(head_dim);
    std::vector<float> host_k_weight(head_dim);
    std::vector<float> host_q_gate(q_output_elements);
    std::vector<float> host_q_rope(q_output_elements);
    std::vector<float> host_k_rope(k_output_elements);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = q_projected_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_q_projected.data(),
            q_projected_buffer->ptr,
            q_projected_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_k_projected.data(),
            k_projected_buffer->ptr,
            k_projected_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_q_weight.data(),
            q_weight_buffer->ptr,
            weight_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_k_weight.data(),
            k_weight_buffer->ptr,
            weight_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy Qwen3.5 q/k norm RoPE batch HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize Qwen3.5 q/k norm RoPE batch HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    qwen35_qk_norm_rope_batch_f32_host(
        host_q_projected.data(),
        host_k_projected.data(),
        host_q_weight.data(),
        host_k_weight.data(),
        q_heads,
        kv_heads,
        sequence_len,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        epsilon,
        host_q_gate.data(),
        host_q_rope.data(),
        host_k_rope.data());
    if (!hip_runtime().copy_async(
            q_gate_output_buffer->ptr,
            host_q_gate.data(),
            q_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            q_rope_output_buffer->ptr,
            host_q_rope.data(),
            q_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            k_rope_output_buffer->ptr,
            host_k_rope.data(),
            k_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy Qwen3.5 q/k norm RoPE batch HIP outputs");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize Qwen3.5 q/k norm RoPE batch HIP output staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipQwen35QkNormRopePagedKvWriteKernelCache {
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
            if (!hiprtc_runtime().compile_qwen35_qk_norm_rope_paged_kv_write_kernel(
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
                    "ullm_qwen35_qk_norm_rope_paged_kv_write_f32_kernel",
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
        append_error(error, compile_errors.empty()
                                ? "failed to build Qwen3.5 q/k norm RoPE paged KV write HIP kernel"
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

HipQwen35QkNormRopePagedKvWriteKernelCache
    &hip_qwen35_qk_norm_rope_paged_kv_write_kernel_cache() {
    static HipQwen35QkNormRopePagedKvWriteKernelCache cache;
    return cache;
}

bool qwen35_qk_norm_rope_paged_kv_write_f32_hip_kernel(
    const ullm_runtime_buffer *q_projected_buffer,
    const ullm_runtime_buffer *k_projected_buffer,
    const ullm_runtime_buffer *v_projected_buffer,
    const ullm_runtime_buffer *q_weight_buffer,
    const ullm_runtime_buffer *k_weight_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    size_t cache_position,
    size_t block_size,
    size_t cache_blocks,
    ullm_runtime_buffer *q_gate_output_buffer,
    ullm_runtime_buffer *q_rope_output_buffer,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_projected_buffer->hip_device_id;
    void *function =
        hip_qwen35_qk_norm_rope_paged_kv_write_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const size_t segments = q_heads + kv_heads + kv_heads;
    if (segments > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "Qwen3.5 q/k norm RoPE paged KV write segment count exceeds HIP grid limit";
        }
        return false;
    }
    constexpr unsigned int launch_block_size = 256;
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    unsigned long long kernel_rotary_dim = static_cast<unsigned long long>(rotary_dim);
    unsigned long long kernel_position_offset = static_cast<unsigned long long>(position_offset);
    unsigned long long kernel_cache_position = static_cast<unsigned long long>(cache_position);
    unsigned long long kernel_block_size = static_cast<unsigned long long>(block_size);
    unsigned long long kernel_cache_blocks = static_cast<unsigned long long>(cache_blocks);
    void *q_projected_ptr = q_projected_buffer->ptr;
    void *k_projected_ptr = k_projected_buffer->ptr;
    void *v_projected_ptr = v_projected_buffer->ptr;
    void *q_weight_ptr = q_weight_buffer->ptr;
    void *k_weight_ptr = k_weight_buffer->ptr;
    void *block_table_ptr = block_table_buffer->ptr;
    void *q_gate_output_ptr = q_gate_output_buffer->ptr;
    void *q_rope_output_ptr = q_rope_output_buffer->ptr;
    void *k_cache_ptr = k_cache_buffer->ptr;
    void *v_cache_ptr = v_cache_buffer->ptr;
    void *kernel_params[] = {
        &q_projected_ptr,
        &k_projected_ptr,
        &v_projected_ptr,
        &q_weight_ptr,
        &k_weight_ptr,
        &block_table_ptr,
        &kernel_q_heads,
        &kernel_kv_heads,
        &kernel_head_dim,
        &kernel_value_dim,
        &kernel_rotary_dim,
        &kernel_position_offset,
        &rope_base,
        &epsilon,
        &kernel_cache_position,
        &kernel_block_size,
        &kernel_cache_blocks,
        &q_gate_output_ptr,
        &q_rope_output_ptr,
        &k_cache_ptr,
        &v_cache_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(segments),
            launch_block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for Qwen3.5 q/k norm RoPE paged KV write";
        }
        return false;
    }
    return true;
}

ullm_status qwen35_qk_norm_rope_paged_kv_write_f32_hip_staging(
    const ullm_runtime_buffer *q_projected_buffer,
    const ullm_runtime_buffer *k_projected_buffer,
    const ullm_runtime_buffer *v_projected_buffer,
    const ullm_runtime_buffer *q_weight_buffer,
    const ullm_runtime_buffer *k_weight_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    size_t cache_position,
    size_t block_size,
    size_t q_projected_bytes,
    size_t k_projected_bytes,
    size_t v_projected_bytes,
    size_t weight_bytes,
    size_t block_table_bytes,
    size_t q_output_bytes,
    size_t k_cache_bytes,
    size_t v_cache_bytes,
    ullm_runtime_buffer *q_gate_output_buffer,
    ullm_runtime_buffer *q_rope_output_buffer,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_stream *stream) {
    const size_t q_projected_elements = q_heads * head_dim * 2;
    const size_t k_projected_elements = kv_heads * head_dim;
    const size_t v_projected_elements = kv_heads * value_dim;
    const size_t q_output_elements = q_heads * head_dim;
    const size_t weight_elements = head_dim;
    const size_t block_table_entries = block_table_bytes / sizeof(std::uint32_t);
    const size_t k_cache_elements = k_cache_bytes / sizeof(float);
    const size_t v_cache_elements = v_cache_bytes / sizeof(float);
    std::vector<float> host_q_projected(q_projected_elements);
    std::vector<float> host_k_projected(k_projected_elements);
    std::vector<float> host_v_projected(v_projected_elements);
    std::vector<float> host_q_weight(weight_elements);
    std::vector<float> host_k_weight(weight_elements);
    std::vector<std::uint32_t> host_block_table(block_table_entries);
    std::vector<float> host_q_gate(q_output_elements);
    std::vector<float> host_q_rope(q_output_elements);
    std::vector<float> host_k_cache(k_cache_elements);
    std::vector<float> host_v_cache(v_cache_elements);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = q_projected_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_q_projected.data(),
            q_projected_buffer->ptr,
            q_projected_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_k_projected.data(),
            k_projected_buffer->ptr,
            k_projected_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_v_projected.data(),
            v_projected_buffer->ptr,
            v_projected_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_q_weight.data(),
            q_weight_buffer->ptr,
            weight_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_k_weight.data(),
            k_weight_buffer->ptr,
            weight_bytes,
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
        set_error("failed to copy Qwen3.5 q/k norm RoPE paged KV write HIP inputs");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize Qwen3.5 q/k norm RoPE paged KV write input copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    qwen35_qk_norm_rope_paged_kv_write_f32_host(
        host_q_projected.data(),
        host_k_projected.data(),
        host_v_projected.data(),
        host_q_weight.data(),
        host_k_weight.data(),
        host_block_table.data(),
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        rotary_dim,
        position_offset,
        rope_base,
        epsilon,
        cache_position,
        block_size,
        host_q_gate.data(),
        host_q_rope.data(),
        host_k_cache.data(),
        host_v_cache.data());
    if (!hip_runtime().copy_async(
            q_gate_output_buffer->ptr,
            host_q_gate.data(),
            q_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            q_rope_output_buffer->ptr,
            host_q_rope.data(),
            q_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
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
        set_error("failed to copy Qwen3.5 q/k norm RoPE paged KV write HIP outputs");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize Qwen3.5 q/k norm RoPE paged KV write output copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipAddKernelCache {
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
            if (!hiprtc_runtime().compile_add_kernel(arch, &code, &compile_error)) {
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
                    "ullm_add_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build f32 add HIP kernel" : compile_errors);
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

HipAddKernelCache &hip_add_kernel_cache() {
    static HipAddKernelCache cache;
    return cache;
}

bool add_f32_hip_kernel(
    const ullm_runtime_buffer *lhs_buffer,
    const ullm_runtime_buffer *rhs_buffer,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = lhs_buffer->hip_device_id;
    void *function = hip_add_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t grid_size = (elements + block_size - 1) / block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "f32 add element count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_elements = static_cast<unsigned long long>(elements);
    void *lhs_ptr = lhs_buffer->ptr;
    void *rhs_ptr = rhs_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &lhs_ptr,
        &rhs_ptr,
        &kernel_elements,
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
            *error = "hipModuleLaunchKernel failed for f32 add";
        }
        return false;
    }
    return true;
}

ullm_status add_f32_hip_staging(
    const ullm_runtime_buffer *lhs_buffer,
    const ullm_runtime_buffer *rhs_buffer,
    size_t elements,
    size_t required_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_lhs(elements);
    std::vector<float> host_rhs(elements);
    std::vector<float> host_output(elements);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = lhs_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_lhs.data(),
            lhs_buffer->ptr,
            required_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_rhs.data(),
            rhs_buffer->ptr,
            required_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 add HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 add HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    add_f32_host(host_lhs.data(), host_rhs.data(), elements, host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 add output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 add HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipRopeKernelCache {
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
            if (!hiprtc_runtime().compile_rope_kernel(arch, &code, &compile_error)) {
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
                    "ullm_rope_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build RoPE HIP kernel" : compile_errors);
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

HipRopeKernelCache &hip_rope_kernel_cache() {
    static HipRopeKernelCache cache;
    return cache;
}

bool rope_f32_hip_kernel(
    const ullm_runtime_buffer *input_buffer,
    size_t sequence_len,
    size_t heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = input_buffer->hip_device_id;
    void *function = hip_rope_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t work_dim = (rotary_dim / 2) + (head_dim - rotary_dim);
    const size_t work_items = sequence_len * heads * work_dim;
    const size_t grid_size = (work_items + block_size - 1) / block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "RoPE element count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_sequence_len = static_cast<unsigned long long>(sequence_len);
    unsigned long long kernel_heads = static_cast<unsigned long long>(heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_rotary_dim = static_cast<unsigned long long>(rotary_dim);
    unsigned long long kernel_position_offset = static_cast<unsigned long long>(position_offset);
    void *input_ptr = input_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &input_ptr,
        &kernel_sequence_len,
        &kernel_heads,
        &kernel_head_dim,
        &kernel_rotary_dim,
        &kernel_position_offset,
        &rope_base,
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
            *error = "hipModuleLaunchKernel failed for f32 RoPE";
        }
        return false;
    }
    return true;
}

ullm_status rope_f32_hip_staging(
    const ullm_runtime_buffer *input_buffer,
    size_t sequence_len,
    size_t heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    size_t required_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_input(required_bytes / sizeof(float));
    std::vector<float> host_output(required_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = input_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            required_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 RoPE HIP input to host staging buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 RoPE HIP input staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    rope_f32_host(
        host_input.data(),
        sequence_len,
        heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 RoPE output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 RoPE HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipCausalAttnKernelCache {
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
            if (!hiprtc_runtime().compile_causal_attn_kernel(arch, &code, &compile_error)) {
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
                    "ullm_causal_attn_f32_kernel",
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
            compile_errors.empty() ? "failed to build causal attention HIP kernel" : compile_errors);
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

HipCausalAttnKernelCache &hip_causal_attn_kernel_cache() {
    static HipCausalAttnKernelCache cache;
    return cache;
}

class HipCausalAttnF32Flash2KernelCache {
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
            if (!hiprtc_runtime().compile_causal_attn_f32_flash2_kernel(
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
                    "ullm_causal_attn_f32_flash2_kernel",
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
            compile_errors.empty() ? "failed to build f32 causal flash2 attention HIP kernel" :
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

HipCausalAttnF32Flash2KernelCache &hip_causal_attn_f32_flash2_kernel_cache() {
    static HipCausalAttnF32Flash2KernelCache cache;
    return cache;
}

class HipCausalAttnBatchKernelCache {
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
            if (!hiprtc_runtime().compile_causal_attn_batch_kernel(arch, &code, &compile_error)) {
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
                    "ullm_causal_attn_batch_f32_kernel",
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
            compile_errors.empty() ? "failed to build batched causal attention HIP kernel" : compile_errors);
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

HipCausalAttnBatchKernelCache &hip_causal_attn_batch_kernel_cache() {
    static HipCausalAttnBatchKernelCache cache;
    return cache;
}

class HipCausalAttnBatchF32Flash2KernelCache {
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
            if (!hiprtc_runtime().compile_causal_attn_batch_f32_flash2_kernel(
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
                    "ullm_causal_attn_batch_f32_flash2_kernel",
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
                "failed to build f32 batched causal flash2 attention HIP kernel" :
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

HipCausalAttnBatchF32Flash2KernelCache &
hip_causal_attn_batch_f32_flash2_kernel_cache() {
    static HipCausalAttnBatchF32Flash2KernelCache cache;
    return cache;
}

bool causal_attn_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    size_t sequence_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    void *function = hip_causal_attn_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t grid_size = sequence_len * q_heads;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "causal attention q head-sequence count exceeds HIP grid limit";
        }
        return false;
    }

    unsigned long long kernel_sequence_len = static_cast<unsigned long long>(sequence_len);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *q_ptr = q_buffer->ptr;
    void *k_ptr = k_buffer->ptr;
    void *v_ptr = v_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr,
        &k_ptr,
        &v_ptr,
        &kernel_sequence_len,
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
            *error = "hipModuleLaunchKernel failed for f32 causal attention";
        }
        return false;
    }
    return true;
}

bool causal_attn_f32_flash2_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    size_t sequence_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    void *function = hip_causal_attn_f32_flash2_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    if (value_dim > block_size) {
        if (error != nullptr) {
            *error = "f32 causal flash2 attention value_dim exceeds 256";
        }
        return false;
    }
    const size_t grid_size = sequence_len * q_heads;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "f32 causal flash2 attention q head-sequence count exceeds HIP grid limit";
        }
        return false;
    }

    unsigned long long kernel_sequence_len = static_cast<unsigned long long>(sequence_len);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *q_ptr = q_buffer->ptr;
    void *k_ptr = k_buffer->ptr;
    void *v_ptr = v_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr,
        &k_ptr,
        &v_ptr,
        &kernel_sequence_len,
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
            *error = "hipModuleLaunchKernel failed for f32 causal flash2 attention";
        }
        return false;
    }
    return true;
}

bool causal_attn_batch_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    size_t batch_count,
    size_t sequence_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    void *function = hip_causal_attn_batch_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    if (sequence_len > std::numeric_limits<size_t>::max() / q_heads ||
        batch_count > std::numeric_limits<size_t>::max() / (sequence_len * q_heads)) {
        if (error != nullptr) {
            *error = "batched causal attention q head-sequence count overflows";
        }
        return false;
    }
    const size_t grid_size = batch_count * sequence_len * q_heads;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "batched causal attention q head-sequence count exceeds HIP grid limit";
        }
        return false;
    }

    unsigned long long kernel_batch_count = static_cast<unsigned long long>(batch_count);
    unsigned long long kernel_sequence_len = static_cast<unsigned long long>(sequence_len);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *q_ptr = q_buffer->ptr;
    void *k_ptr = k_buffer->ptr;
    void *v_ptr = v_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr,
        &k_ptr,
        &v_ptr,
        &kernel_batch_count,
        &kernel_sequence_len,
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
            *error = "hipModuleLaunchKernel failed for f32 batched causal attention";
        }
        return false;
    }
    return true;
}

bool causal_attn_batch_f32_flash2_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    size_t batch_count,
    size_t sequence_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    void *function =
        hip_causal_attn_batch_f32_flash2_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    if (value_dim > block_size) {
        if (error != nullptr) {
            *error = "f32 batched causal flash2 attention value_dim exceeds 256";
        }
        return false;
    }
    if (sequence_len > std::numeric_limits<size_t>::max() / q_heads ||
        batch_count > std::numeric_limits<size_t>::max() / (sequence_len * q_heads)) {
        if (error != nullptr) {
            *error = "f32 batched causal flash2 attention q head-sequence count overflows";
        }
        return false;
    }
    const size_t grid_size = batch_count * sequence_len * q_heads;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error =
                "f32 batched causal flash2 attention q head-sequence count exceeds HIP grid limit";
        }
        return false;
    }

    unsigned long long kernel_batch_count = static_cast<unsigned long long>(batch_count);
    unsigned long long kernel_sequence_len = static_cast<unsigned long long>(sequence_len);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *q_ptr = q_buffer->ptr;
    void *k_ptr = k_buffer->ptr;
    void *v_ptr = v_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr,
        &k_ptr,
        &v_ptr,
        &kernel_batch_count,
        &kernel_sequence_len,
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
            *error = "hipModuleLaunchKernel failed for f32 batched causal flash2 attention";
        }
        return false;
    }
    return true;
}

ullm_status causal_attn_f32_hip_staging(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    size_t sequence_len,
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
            device_id)) {
        set_error("failed to copy f32 causal attention HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 causal attention HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    causal_attn_f32_host(
        host_q.data(),
        host_k.data(),
        host_v.data(),
        sequence_len,
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
        set_error("failed to copy f32 causal attention output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 causal attention HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status causal_attn_batch_f32_hip_staging(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    size_t batch_count,
    size_t sequence_len,
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
            device_id)) {
        set_error("failed to copy f32 batched causal attention HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 batched causal attention HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    const size_t q_elements_per_batch = sequence_len * q_heads * head_dim;
    const size_t k_elements_per_batch = sequence_len * kv_heads * head_dim;
    const size_t v_elements_per_batch = sequence_len * kv_heads * value_dim;
    const size_t output_elements_per_batch = sequence_len * q_heads * value_dim;
    for (size_t batch = 0; batch < batch_count; ++batch) {
        causal_attn_f32_host(
            host_q.data() + batch * q_elements_per_batch,
            host_k.data() + batch * k_elements_per_batch,
            host_v.data() + batch * v_elements_per_batch,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            host_output.data() + batch * output_elements_per_batch);
    }
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 batched causal attention output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 batched causal attention HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipDecodeAttnKernelCache {
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
            if (!hiprtc_runtime().compile_decode_attn_kernel(arch, &code, &compile_error)) {
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
                    "ullm_decode_attn_f32_kernel",
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
            compile_errors.empty() ? "failed to build decode attention HIP kernel" : compile_errors);
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

HipDecodeAttnKernelCache &hip_decode_attn_kernel_cache() {
    static HipDecodeAttnKernelCache cache;
    return cache;
}

bool decode_attn_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cache_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    void *function = hip_decode_attn_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t output_elements = q_heads * value_dim;
    const bool disable_head_parallel =
        std::getenv("ULLM_DISABLE_DECODE_ATTN_HEAD_PARALLEL") != nullptr;
    const bool use_head_parallel_kernel =
        !disable_head_parallel && head_dim <= block_size && value_dim <= block_size;
    size_t grid_size =
        use_head_parallel_kernel ? q_heads : (output_elements + block_size - 1) / block_size;
    if (disable_head_parallel && grid_size == q_heads) {
        if (grid_size == std::numeric_limits<size_t>::max()) {
            if (error != nullptr) {
                *error = "decode attention diagnostic launch grid overflows";
            }
            return false;
        }
        grid_size += 1;
    }
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "decode attention output element count exceeds HIP grid limit";
        }
        return false;
    }

    unsigned long long kernel_cache_len = static_cast<unsigned long long>(cache_len);
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
        &kernel_cache_len,
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
            *error = "hipModuleLaunchKernel failed for f32 decode attention";
        }
        return false;
    }
    return true;
}

ullm_status decode_attn_f32_hip_staging(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cache_len,
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
        set_error("failed to copy f32 decode attention HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 decode attention HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    decode_attn_f32_host(
        host_q.data(),
        host_k.data(),
        host_v.data(),
        cache_len,
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
        set_error("failed to copy f32 decode attention output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 decode attention HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

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

class HipCachedPrefixAttnF32Flash2KernelCache {
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
            if (!hiprtc_runtime().compile_cached_prefix_attn_f32_flash2_kernel(
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
                    "ullm_cached_prefix_attn_f32_flash2_kernel",
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
                "failed to build f32 cached prefix flash2 attention HIP kernel" :
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
    void *function =
        hip_cached_prefix_attn_f32_flash2_kernel_cache().function_for_device(device_id, error);
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
    const size_t grid_size = new_tokens * q_heads;
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

class HipPagedDecodeAttnKernelCache {
public:
    struct Functions {
        void *legacy = nullptr;
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
            loaded->split_partial_function = split_partial_function;
            loaded->split_merge_function = split_merge_function;
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
        void *split_partial_function = nullptr;
        void *split_merge_function = nullptr;
        std::string arch;
    };

    static void assign_functions(const LoadedModule &loaded, Functions *functions) {
        if (functions == nullptr) {
            return;
        }
        functions->legacy = loaded.legacy_function;
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
    const bool use_head_parallel_kernel =
        head_dim <= launch_block_size && value_dim <= launch_block_size;
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
    if (split_count != 0 && q_heads > std::numeric_limits<size_t>::max() / split_count) {
        if (error != nullptr) {
            *error = "paged decode split attention partial grid overflows";
        }
        return false;
    }
    const size_t partial_grid_size = q_heads * split_count;
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

class HipPagedKvWriteKernelCache {
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
            loaded->arch = arch;
            void *result = loaded->function;
            modules_.emplace(device_id, std::move(loaded));
            return result;
        }
        append_error(
            error,
            compile_errors.empty() ? "failed to build paged KV write HIP kernel" : compile_errors);
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

private:
    struct LoadedModule {
        void *module = nullptr;
        void *writer = nullptr;
        void *reader = nullptr;
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

class HipDepthwiseConv1dKernelCache {
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
            if (!hiprtc_runtime().compile_depthwise_conv1d_kernel(arch, &code, &compile_error)) {
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
                    "ullm_depthwise_conv1d_f32_kernel",
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
            compile_errors.empty() ? "failed to build depthwise conv1d HIP kernel" : compile_errors);
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

HipDepthwiseConv1dKernelCache &hip_depthwise_conv1d_kernel_cache() {
    static HipDepthwiseConv1dKernelCache cache;
    return cache;
}

bool depthwise_conv1d_f32_hip_kernel(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t channels,
    size_t sequence_len,
    size_t kernel_size,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = input_buffer->hip_device_id;
    void *function = hip_depthwise_conv1d_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t elements = channels * sequence_len;
    const size_t grid_size = (elements + block_size - 1) / block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "depthwise conv1d element count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_channels = static_cast<unsigned long long>(channels);
    unsigned long long kernel_sequence_len = static_cast<unsigned long long>(sequence_len);
    unsigned long long kernel_size_arg = static_cast<unsigned long long>(kernel_size);
    void *input_ptr = input_buffer->ptr;
    void *weight_ptr = weight_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &input_ptr,
        &weight_ptr,
        &kernel_channels,
        &kernel_sequence_len,
        &kernel_size_arg,
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
            *error = "hipModuleLaunchKernel failed for f32 depthwise conv1d";
        }
        return false;
    }
    return true;
}

ullm_status depthwise_conv1d_f32_hip_staging(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t channels,
    size_t sequence_len,
    size_t kernel_size,
    size_t required_input_bytes,
    size_t required_weight_bytes,
    size_t required_output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_input(required_input_bytes / sizeof(float));
    std::vector<float> host_weight(required_weight_bytes / sizeof(float));
    std::vector<float> host_output(required_output_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = input_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            required_input_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_weight.data(),
            weight_buffer->ptr,
            required_weight_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 depthwise conv1d HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 depthwise conv1d HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    depthwise_conv1d_f32_host(
        host_input.data(),
        host_weight.data(),
        channels,
        sequence_len,
        kernel_size,
        host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 depthwise conv1d output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 depthwise conv1d HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

void linear_attn_qkv_prepare_f32_host(
    const float *qkv,
    const float *conv_weight,
    float *conv_history,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    float q_scale,
    bool qk_l2_norm,
    float *conv_output,
    float *q_output,
    float *k_output,
    float *v_output) {
    const size_t q_elements = key_heads * key_dim;
    const size_t k_elements = q_elements;
    const size_t v_elements = value_heads * value_dim;
    const size_t channels = q_elements + k_elements + v_elements;
    for (size_t channel = 0; channel < channels; ++channel) {
        for (size_t kernel = 0; kernel + 1 < kernel_size; ++kernel) {
            conv_history[kernel * channels + channel] =
                conv_history[(kernel + 1) * channels + channel];
        }
        conv_history[(kernel_size - 1) * channels + channel] = qkv[channel];

        float sum = 0.0f;
        for (size_t kernel = 0; kernel < kernel_size; ++kernel) {
            sum += conv_history[kernel * channels + channel] *
                   conv_weight[channel * kernel_size + kernel];
        }
        const float sigmoid = 1.0f / (1.0f + std::exp(-sum));
        conv_output[channel] = sum * sigmoid;
    }

    for (size_t head = 0; head < key_heads; ++head) {
        const size_t q_source = head * key_dim;
        const size_t k_source = q_elements + head * key_dim;
        const size_t target = head * key_dim;
        float q_square_sum = 0.0f;
        float k_square_sum = 0.0f;
        for (size_t dim = 0; dim < key_dim; ++dim) {
            const float q_value = conv_output[q_source + dim];
            const float k_value = conv_output[k_source + dim];
            q_square_sum += q_value * q_value;
            k_square_sum += k_value * k_value;
        }
        const float q_norm = std::sqrt(q_square_sum + 1.0e-6f);
        const float k_norm = std::sqrt(k_square_sum + 1.0e-6f);
        for (size_t dim = 0; dim < key_dim; ++dim) {
            const float q_value = conv_output[q_source + dim];
            const float k_value = conv_output[k_source + dim];
            q_output[target + dim] = qk_l2_norm ? (q_value / q_norm) * q_scale : q_value * q_scale;
            k_output[target + dim] = qk_l2_norm ? k_value / k_norm : k_value;
        }
    }

    const size_t v_source = q_elements + k_elements;
    std::memcpy(v_output, conv_output + v_source, v_elements * sizeof(float));
}

void linear_attn_qkv_prepare_batch_f32_host(
    const float *qkv,
    const float *conv_weight,
    float *conv_history,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    size_t sequence_len,
    float q_scale,
    bool qk_l2_norm,
    float *conv_output,
    float *q_output,
    float *k_output,
    float *v_output) {
    const size_t q_elements = key_heads * key_dim;
    const size_t v_elements = value_heads * value_dim;
    const size_t channels = q_elements * 2 + v_elements;
    for (size_t token = 0; token < sequence_len; ++token) {
        linear_attn_qkv_prepare_f32_host(
            qkv + token * channels,
            conv_weight,
            conv_history,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            q_scale,
            qk_l2_norm,
            conv_output + token * channels,
            q_output + token * q_elements,
            k_output + token * q_elements,
            v_output + token * v_elements);
    }
}

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

class HipLinearAttnRecurrentKernelCache {
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
            if (!hiprtc_runtime().compile_linear_attn_recurrent_kernel(arch, &code, &compile_error)) {
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

HipLinearAttnRecurrentKernelCache &hip_linear_attn_recurrent_kernel_cache() {
    static HipLinearAttnRecurrentKernelCache cache;
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
    void *function = hip_linear_attn_recurrent_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }
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

} // namespace

#include "../ullm_runtime_api.inc"
