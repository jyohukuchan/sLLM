// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
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

