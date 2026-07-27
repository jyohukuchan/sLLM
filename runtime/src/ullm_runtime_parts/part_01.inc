
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
                }
            }
            if (!typed_loaded) {
                hip_runtime().module_unload(module, device_id);
                append_error(
                    &compile_errors,
                    "hipModuleGetFunction typed Qwen3.5 paged KV write failed for " + arch);
                continue;
            }
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

    void *typed_function_for_device(
        int device_id,
        ullm_kv_cache_dtype k_dtype,
        ullm_kv_cache_dtype v_dtype,
        std::string *error) {
        const int k_index = static_cast<int>(k_dtype);
        const int v_index = static_cast<int>(v_dtype);
        if (k_index < 0 || k_index > 2 || v_index < 0 || v_index > 2) {
            append_error(error, "typed Qwen3.5 paged KV writer received an unsupported cache dtype");
            return nullptr;
        }
        void *legacy = function_for_device(device_id, error);
        if (legacy == nullptr) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = modules_.find(device_id);
        if (found == modules_.end()) {
            append_error(error, "typed Qwen3.5 paged KV HIP module disappeared after load");
            return nullptr;
        }
        return k_index == 0 && v_index == 0 ? found->second->function
                                             : found->second->typed[k_index][v_index];
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
             "ullm_qwen35_qk_norm_rope_paged_kv_write_typed_kf32_vf16_kernel",
             "ullm_qwen35_qk_norm_rope_paged_kv_write_typed_kf32_vfp8_kernel"},
            {"ullm_qwen35_qk_norm_rope_paged_kv_write_typed_kf16_vf32_kernel",
             "ullm_qwen35_qk_norm_rope_paged_kv_write_typed_kf16_vf16_kernel",
             "ullm_qwen35_qk_norm_rope_paged_kv_write_typed_kf16_vfp8_kernel"},
            {"ullm_qwen35_qk_norm_rope_paged_kv_write_typed_kfp8_vf32_kernel",
             "ullm_qwen35_qk_norm_rope_paged_kv_write_typed_kfp8_vf16_kernel",
             "ullm_qwen35_qk_norm_rope_paged_kv_write_typed_kfp8_vfp8_kernel"},
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

bool qwen35_qk_norm_rope_paged_kv_write_typed_f32_hip_kernel(
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
    ullm_kv_cache_dtype k_dtype,
    ullm_kv_cache_dtype v_dtype,
    ullm_runtime_buffer *q_gate_output_buffer,
    ullm_runtime_buffer *q_rope_output_buffer,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_buffer *k_scale_buffer,
    ullm_runtime_buffer *v_scale_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    if (k_dtype == ULLM_KV_CACHE_DTYPE_F32 && v_dtype == ULLM_KV_CACHE_DTYPE_F32) {
        return qwen35_qk_norm_rope_paged_kv_write_f32_hip_kernel(
            q_projected_buffer,
            k_projected_buffer,
            v_projected_buffer,
            q_weight_buffer,
            k_weight_buffer,
            block_table_buffer,
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
            cache_blocks,
            q_gate_output_buffer,
            q_rope_output_buffer,
            k_cache_buffer,
            v_cache_buffer,
            stream,
            error);
    }
    const int device_id = q_projected_buffer->hip_device_id;
    void *function = hip_qwen35_qk_norm_rope_paged_kv_write_kernel_cache()
        .typed_function_for_device(device_id, k_dtype, v_dtype, error);
    if (function == nullptr) {
        return false;
    }
    const size_t segments = q_heads + kv_heads + kv_heads;
    if (segments > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "typed Qwen3.5 q/k norm RoPE paged KV write segment count exceeds HIP grid limit";
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
    void *k_scale_ptr = k_scale_buffer == nullptr ? nullptr : k_scale_buffer->ptr;
    void *v_scale_ptr = v_scale_buffer == nullptr ? nullptr : v_scale_buffer->ptr;
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
        &k_scale_ptr,
        &v_scale_ptr,
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
            *error = "hipModuleLaunchKernel failed for typed Qwen3.5 q/k norm RoPE paged KV write";
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

#include "../parts/cached_prefix_attention_kernels.inc"
#include "../parts/paged_decode_attention_kernels.inc"
#include "../parts/paged_kv_write_kernels.inc"
#include "../parts/paged_chunk_kernels.inc"
#include "../parts/paged_wmma_chunk_kernels.inc"
#include "../parts/depthwise_conv1d_kernels.inc"
#include "../parts/linear_attn_prepare_kernels.inc"
#include "../parts/linear_attn_recurrent_kernels.inc"
#include "../parts/runtime_epilogue.inc"
