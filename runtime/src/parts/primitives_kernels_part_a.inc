// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
class HipMatvecKernelCache {
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
            if (!hiprtc_runtime().compile_matvec_kernel(arch, &code, &compile_error)) {
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
                    "ullm_matvec_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build matvec HIP kernel" : compile_errors);
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

HipMatvecKernelCache &hip_matvec_kernel_cache() {
    static HipMatvecKernelCache cache;
    return cache;
}

class HipBf16MatvecKernelCache {
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
            if (!hiprtc_runtime().compile_matvec_bf16_kernel(arch, &code, &compile_error)) {
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
                    "ullm_matvec_bf16_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build BF16 matvec HIP kernel" : compile_errors);
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

HipBf16MatvecKernelCache &hip_bf16_matvec_kernel_cache() {
    static HipBf16MatvecKernelCache cache;
    return cache;
}

class HipGemmaBf16MatmulKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) return found->second->function;
        const std::vector<std::string> candidates = hip_arch_candidates(device_id);
        if (candidates.empty()) {
            append_error(error, "unable to infer HIP offload architecture for Gemma BF16 matmul");
            return nullptr;
        }
        std::string compile_errors;
        for (const std::string &arch : candidates) {
            std::vector<char> code;
            std::string compile_error;
            if (!hiprtc_runtime().compile_gemma_bf16_matmul_kernel(arch, &code, &compile_error)) {
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
                    &function, module, "ullm_gemma_bf16_matmul_f32_kernel", device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction failed for " + arch);
                continue;
            }
            auto loaded = std::make_unique<LoadedModule>();
            loaded->module = module;
            loaded->function = function;
            void *result = loaded->function;
            modules_.emplace(device_id, std::move(loaded));
            return result;
        }
        append_error(error, compile_errors.empty() ? "failed to build Gemma BF16 matmul HIP kernel" : compile_errors);
        return nullptr;
    }

private:
    struct LoadedModule { void *module = nullptr; void *function = nullptr; };
    static void append_error(std::string *error, const std::string &message) {
        if (error == nullptr || message.empty()) return;
        if (!error->empty()) error->append("\n");
        error->append(message);
    }
    std::mutex mutex_;
    std::unordered_map<int, std::unique_ptr<LoadedModule>> modules_;
};

HipGemmaBf16MatmulKernelCache &hip_gemma_bf16_matmul_kernel_cache() {
    static HipGemmaBf16MatmulKernelCache cache;
    return cache;
}

class HipBf16RowKernelCache {
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
            if (!hiprtc_runtime().compile_bf16_row_kernel(arch, &code, &compile_error)) {
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
                    "ullm_bf16_row_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build BF16 row HIP kernel" : compile_errors);
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

HipBf16RowKernelCache &hip_bf16_row_kernel_cache() {
    static HipBf16RowKernelCache cache;
    return cache;
}

bool matvec_f32_hip_kernel(
    const ullm_runtime_buffer *matrix_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = matrix_buffer->hip_device_id;
    void *function = hip_matvec_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    if (rows > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "matvec row count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *matrix_ptr = matrix_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &matrix_ptr,
        &input_ptr,
        &kernel_rows,
        &kernel_cols,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(rows),
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 matvec";
        }
        return false;
    }
    return true;
}

bool matvec_bf16_f32_hip_kernel(
    const ullm_runtime_buffer *matrix_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = matrix_buffer->hip_device_id;
    void *function = hip_bf16_matvec_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 64;
    if (rows > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "BF16 matvec row count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *matrix_ptr = matrix_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &matrix_ptr,
        &input_ptr,
        &kernel_rows,
        &kernel_cols,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(rows),
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for BF16 matvec";
        }
        return false;
    }
    return true;
}

bool gemma_bf16_matmul_f32_hip_kernel(
    const ullm_runtime_buffer *matrix_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    size_t batch_count,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = matrix_buffer->hip_device_id;
    void *function = hip_gemma_bf16_matmul_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) return false;
    if (rows > std::numeric_limits<unsigned int>::max() ||
        batch_count > std::numeric_limits<unsigned int>::max()) {
        if (error != nullptr) *error = "Gemma BF16 matmul dimensions exceed HIP grid limit";
        return false;
    }
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_batch = static_cast<unsigned long long>(batch_count);
    void *matrix_ptr = matrix_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &matrix_ptr, &input_ptr, &kernel_rows, &kernel_cols, &kernel_batch, &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel_2d(
            function,
            static_cast<unsigned int>(rows),
            static_cast<unsigned int>(batch_count),
            64,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) *error = "hipModuleLaunchKernel failed for Gemma BF16 matmul";
        return false;
    }
    return true;
}

bool bf16_row_f32_hip_kernel(
    const ullm_runtime_buffer *matrix_buffer,
    size_t rows,
    size_t cols,
    size_t row_index,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = matrix_buffer->hip_device_id;
    void *function = hip_bf16_row_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_row_index = static_cast<unsigned long long>(row_index);
    void *matrix_ptr = matrix_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &matrix_ptr,
        &kernel_rows,
        &kernel_cols,
        &kernel_row_index,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            1,
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for BF16 row";
        }
        return false;
    }
    return true;
}

ullm_status matvec_f32_hip_staging(
    const ullm_runtime_buffer *matrix_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    size_t required_matrix_bytes,
    size_t required_input_bytes,
    size_t required_output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_matrix(required_matrix_bytes / sizeof(float));
    std::vector<float> host_input(cols);
    std::vector<float> host_output(rows);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = matrix_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_matrix.data(),
            matrix_buffer->ptr,
            required_matrix_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            required_input_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 matvec HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 matvec HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    matvec_f32_host(host_matrix.data(), host_input.data(), rows, cols, host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 matvec output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 matvec HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status matvec_bf16_f32_hip_staging(
    const ullm_runtime_buffer *matrix_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    size_t required_matrix_bytes,
    size_t required_input_bytes,
    size_t required_output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<uint16_t> host_matrix(required_matrix_bytes / sizeof(uint16_t));
    std::vector<float> host_input(cols);
    std::vector<float> host_output(rows);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = matrix_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_matrix.data(),
            matrix_buffer->ptr,
            required_matrix_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            required_input_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy BF16 matvec HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize BF16 matvec HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    matvec_bf16_f32_host(host_matrix.data(), host_input.data(), rows, cols, host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy BF16 matvec output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize BF16 matvec HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status bf16_row_f32_hip_staging(
    const ullm_runtime_buffer *matrix_buffer,
    size_t rows,
    size_t cols,
    size_t row_index,
    size_t required_matrix_bytes,
    size_t required_output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<uint16_t> host_row(cols);
    std::vector<float> host_output(cols);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = matrix_buffer->hip_device_id;
    const size_t row_bytes = cols * sizeof(uint16_t);
    if (row_index >= rows) {
        set_error("BF16 row staging index is out of range");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t row_offset = row_index * row_bytes;
    if (row_offset > required_matrix_bytes || row_bytes > required_matrix_bytes - row_offset) {
        set_error("BF16 row byte offset overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (!hip_runtime().copy_async(
            host_row.data(),
            static_cast<const std::uint8_t *>(matrix_buffer->ptr) + row_offset,
            row_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy BF16 row HIP input to host staging buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize BF16 row HIP input staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    bf16_row_f32_host(host_row.data(), 1, cols, 0, host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy BF16 row output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize BF16 row HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

void top1_f32_partials_host(
    const float *input,
    size_t elements,
    size_t partial_count,
    float *partial_values,
    uint32_t *partial_indices) {
    constexpr size_t block_size = 256;
    for (size_t block = 0; block < partial_count; ++block) {
        const size_t start = block * block_size;
        const size_t end = std::min(start + block_size, elements);
        float best_value = -std::numeric_limits<float>::max();
        uint32_t best_index = std::numeric_limits<uint32_t>::max();
        for (size_t index = start; index < end; ++index) {
            float value = input[index];
            if (value != value) {
                value = -std::numeric_limits<float>::max();
            }
            const auto token_index = static_cast<uint32_t>(index);
            if (value > best_value || (value == best_value && token_index < best_index)) {
                best_value = value;
                best_index = token_index;
            }
        }
        partial_values[block] = best_value;
        partial_indices[block] = best_index;
    }
}

void top1_pairs_f32_partials_host(
    const float *input_values,
    const uint32_t *input_indices,
    size_t elements,
    size_t partial_count,
    float *partial_values,
    uint32_t *partial_indices) {
    constexpr size_t block_size = 256;
    for (size_t block = 0; block < partial_count; ++block) {
        const size_t start = block * block_size;
        const size_t end = std::min(start + block_size, elements);
        float best_value = -std::numeric_limits<float>::max();
        uint32_t best_index = std::numeric_limits<uint32_t>::max();
        for (size_t index = start; index < end; ++index) {
            float value = input_values[index];
            if (value != value) {
                value = -std::numeric_limits<float>::max();
            }
            const uint32_t token_index = input_indices[index];
            if (value > best_value || (value == best_value && token_index < best_index)) {
                best_value = value;
                best_index = token_index;
            }
        }
        partial_values[block] = best_value;
        partial_indices[block] = best_index;
    }
}

class HipTop1KernelCache {
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
            if (!hiprtc_runtime().compile_top1_kernel(arch, &code, &compile_error)) {
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
                    "ullm_top1_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build top1 HIP kernel" : compile_errors);
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

HipTop1KernelCache &hip_top1_kernel_cache() {
    static HipTop1KernelCache cache;
    return cache;
}

class HipTop1PairsKernelCache {
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
            if (!hiprtc_runtime().compile_top1_pairs_kernel(arch, &code, &compile_error)) {
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
                    "ullm_top1_pairs_f32_kernel",
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
            compile_errors.empty() ? "failed to build top1 pairs HIP kernel" : compile_errors);
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

HipTop1PairsKernelCache &hip_top1_pairs_kernel_cache() {
    static HipTop1PairsKernelCache cache;
    return cache;
}

bool top1_f32_hip_kernel(
    const ullm_runtime_buffer *input_buffer,
    size_t elements,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = input_buffer->hip_device_id;
    void *function = hip_top1_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    if (partial_count > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "top1 partial count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_elements = static_cast<unsigned long long>(elements);
    void *input_ptr = input_buffer->ptr;
    void *partial_values_ptr = partial_values_buffer->ptr;
    void *partial_indices_ptr = partial_indices_buffer->ptr;
    void *kernel_params[] = {
        &input_ptr,
        &kernel_elements,
        &partial_values_ptr,
        &partial_indices_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(partial_count),
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 top1";
        }
        return false;
    }
    return true;
}

bool top1_pairs_f32_hip_kernel(
    const ullm_runtime_buffer *values_buffer,
    const ullm_runtime_buffer *indices_buffer,
    size_t elements,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = values_buffer->hip_device_id;
    void *function = hip_top1_pairs_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    if (partial_count > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "top1 pairs partial count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_elements = static_cast<unsigned long long>(elements);
    void *values_ptr = values_buffer->ptr;
    void *indices_ptr = indices_buffer->ptr;
    void *partial_values_ptr = partial_values_buffer->ptr;
    void *partial_indices_ptr = partial_indices_buffer->ptr;
    void *kernel_params[] = {
        &values_ptr,
        &indices_ptr,
        &kernel_elements,
        &partial_values_ptr,
        &partial_indices_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(partial_count),
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 top1 pairs";
        }
        return false;
    }
    return true;
}

ullm_status top1_f32_hip_staging(
    const ullm_runtime_buffer *input_buffer,
    size_t elements,
    size_t input_bytes,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    size_t partial_values_bytes,
    size_t partial_indices_bytes,
    ullm_runtime_stream *stream) {
    std::vector<float> host_input(elements);
    std::vector<float> host_partial_values(partial_count);
    std::vector<uint32_t> host_partial_indices(partial_count);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = input_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            input_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 top1 HIP input to host staging buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 top1 HIP input staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    top1_f32_partials_host(
        host_input.data(),
        elements,
        partial_count,
        host_partial_values.data(),
        host_partial_indices.data());
    if (!hip_runtime().copy_async(
            partial_values_buffer->ptr,
            host_partial_values.data(),
            partial_values_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            partial_indices_buffer->ptr,
            host_partial_indices.data(),
            partial_indices_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 top1 partial outputs to HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 top1 HIP output staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status top1_pairs_f32_hip_staging(
    const ullm_runtime_buffer *values_buffer,
    const ullm_runtime_buffer *indices_buffer,
    size_t elements,
    size_t values_bytes,
    size_t indices_bytes,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    size_t partial_values_bytes,
    size_t partial_indices_bytes,
    ullm_runtime_stream *stream) {
    std::vector<float> host_values(elements);
    std::vector<uint32_t> host_indices(elements);
    std::vector<float> host_partial_values(partial_count);
    std::vector<uint32_t> host_partial_indices(partial_count);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = values_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_values.data(),
            values_buffer->ptr,
            values_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_indices.data(),
            indices_buffer->ptr,
            indices_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 top1 pairs HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 top1 pairs HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    top1_pairs_f32_partials_host(
        host_values.data(),
        host_indices.data(),
        elements,
        partial_count,
        host_partial_values.data(),
        host_partial_indices.data());
    if (!hip_runtime().copy_async(
            partial_values_buffer->ptr,
            host_partial_values.data(),
            partial_values_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            partial_indices_buffer->ptr,
            host_partial_indices.data(),
            partial_indices_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 top1 pairs partial outputs to HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 top1 pairs HIP output staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

void rmsnorm_f32_host(
    const float *input,
    const float *weight,
    size_t elements,
    float epsilon,
    float *output) {
    float sum_squares = 0.0f;
    for (size_t index = 0; index < elements; ++index) {
        sum_squares += input[index] * input[index];
    }
    const float inv_rms = 1.0f / std::sqrt(sum_squares / static_cast<float>(elements) + epsilon);
    for (size_t index = 0; index < elements; ++index) {
        output[index] = input[index] * inv_rms * weight[index];
    }
}

class HipRmsNormKernelCache {
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
            if (!hiprtc_runtime().compile_rmsnorm_kernel(arch, &code, &compile_error)) {
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
                    "ullm_rmsnorm_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build RMSNorm HIP kernel" : compile_errors);
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

HipRmsNormKernelCache &hip_rmsnorm_kernel_cache() {
    static HipRmsNormKernelCache cache;
    return cache;
}

// This cache is deliberately separate from production: it compiles only the fixed gfx1201
// wave32-reduction module and is reachable only through the direct prototype ABI.
class HipRmsNormShufflePrototypeKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second->function;
        }
        if (hip_runtime().device_arch_name(device_id) != "gfx1201") {
            append_error(error, "RMSNorm shuffle prototype requires exact gfx1201 architecture");
            return nullptr;
        }

        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_rmsnorm_shuffle_prototype_kernel(
                "gfx1201", &code, &compile_error)) {
            append_error(
                error,
                compile_error.empty() ?
                    "failed to build RMSNorm shuffle prototype HIP kernel" :
                    compile_error);
            return nullptr;
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            append_error(error, "hipModuleLoadData failed for RMSNorm shuffle prototype");
            return nullptr;
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function,
                module,
                "ullm_rmsnorm_shuffle_prototype_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            append_error(error, "hipModuleGetFunction failed for RMSNorm shuffle prototype");
            return nullptr;
        }
        auto loaded = std::make_unique<LoadedModule>();
        loaded->module = module;
        loaded->function = function;
        modules_.emplace(device_id, std::move(loaded));
        return function;
    }

private:
    struct LoadedModule {
        void *module = nullptr;
        void *function = nullptr;
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

HipRmsNormShufflePrototypeKernelCache &hip_rmsnorm_shuffle_prototype_kernel_cache() {
    static HipRmsNormShufflePrototypeKernelCache cache;
    return cache;
}

bool rmsnorm_f32_hip_kernel(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t elements,
    float epsilon,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = input_buffer->hip_device_id;
    void *function = hip_rmsnorm_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    unsigned long long kernel_elements = static_cast<unsigned long long>(elements);
    void *input_ptr = input_buffer->ptr;
    void *weight_ptr = weight_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &input_ptr,
        &weight_ptr,
        &kernel_elements,
        &epsilon,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            1,
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 RMSNorm";
        }
        return false;
    }
    return true;
}

bool rmsnorm_shuffle_prototype_f32_hip_kernel(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t elements,
    float epsilon,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = input_buffer->hip_device_id;
    void *function = hip_rmsnorm_shuffle_prototype_kernel_cache().function_for_device(
        device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    unsigned long long kernel_elements = static_cast<unsigned long long>(elements);
    void *input_ptr = input_buffer->ptr;
    void *weight_ptr = weight_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &input_ptr,
        &weight_ptr,
        &kernel_elements,
        &epsilon,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            1,
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for RMSNorm shuffle prototype";
        }
        return false;
    }
    return true;
}

ullm_status rmsnorm_f32_hip_staging(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t elements,
    float epsilon,
    size_t required_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_input(elements);
    std::vector<float> host_weight(elements);
    std::vector<float> host_output(elements);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = input_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            required_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_weight.data(),
            weight_buffer->ptr,
            required_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 RMSNorm HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 RMSNorm HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    rmsnorm_f32_host(host_input.data(), host_weight.data(), elements, epsilon, host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 RMSNorm output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 RMSNorm HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

void segmented_rmsnorm_f32_host(
    const float *input,
    const float *weight,
    size_t segments,
    size_t segment_size,
    float epsilon,
    float *output) {
    for (size_t segment = 0; segment < segments; ++segment) {
        const size_t base = segment * segment_size;
        float sum_squares = 0.0f;
        for (size_t dim = 0; dim < segment_size; ++dim) {
            const float value = input[base + dim];
            sum_squares += value * value;
        }
        const float inv_rms =
            1.0f / std::sqrt(sum_squares / static_cast<float>(segment_size) + epsilon);
        for (size_t dim = 0; dim < segment_size; ++dim) {
            output[base + dim] = input[base + dim] * inv_rms * weight[dim];
        }
    }
}

void segmented_rmsnorm_silu_mul_f32_host(
    const float *input,
    const float *weight,
    const float *gate,
    size_t segments,
    size_t segment_size,
    float epsilon,
    float *output) {
    for (size_t segment = 0; segment < segments; ++segment) {
        const size_t base = segment * segment_size;
        float sum_squares = 0.0f;
        for (size_t dim = 0; dim < segment_size; ++dim) {
            const float value = input[base + dim];
            sum_squares += value * value;
        }
        const float inv_rms =
            1.0f / std::sqrt(sum_squares / static_cast<float>(segment_size) + epsilon);
        for (size_t dim = 0; dim < segment_size; ++dim) {
            const size_t index = base + dim;
            const float gate_value = gate[index];
            const float sigmoid = 1.0f / (1.0f + std::exp(-gate_value));
            const float normed = input[index] * inv_rms * weight[dim];
            output[index] = gate_value * sigmoid * normed;
        }
    }
}

class HipSegmentedRmsNormKernelCache {
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
            if (!hiprtc_runtime().compile_segmented_rmsnorm_kernel(arch, &code, &compile_error)) {
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
                    "ullm_segmented_rmsnorm_f32_kernel",
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
            compile_errors.empty() ? "failed to build segmented RMSNorm HIP kernel" : compile_errors);
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

HipSegmentedRmsNormKernelCache &hip_segmented_rmsnorm_kernel_cache() {
    static HipSegmentedRmsNormKernelCache cache;
    return cache;
}

bool segmented_rmsnorm_f32_hip_kernel(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t segments,
    size_t segment_size,
    float epsilon,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = input_buffer->hip_device_id;
    void *function = hip_segmented_rmsnorm_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    if (segments > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "segmented RMSNorm segment count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_segments = static_cast<unsigned long long>(segments);
    unsigned long long kernel_segment_size = static_cast<unsigned long long>(segment_size);
    void *input_ptr = input_buffer->ptr;
    void *weight_ptr = weight_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &input_ptr,
        &weight_ptr,
        &kernel_segments,
        &kernel_segment_size,
        &epsilon,
        &output_ptr,
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
            *error = "hipModuleLaunchKernel failed for f32 segmented RMSNorm";
        }
        return false;
    }
    return true;
}

ullm_status segmented_rmsnorm_f32_hip_staging(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t segments,
    size_t segment_size,
    float epsilon,
    size_t input_output_bytes,
    size_t weight_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_input(input_output_bytes / sizeof(float));
    std::vector<float> host_weight(weight_bytes / sizeof(float));
    std::vector<float> host_output(input_output_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = input_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            input_output_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_weight.data(),
            weight_buffer->ptr,
            weight_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 segmented RMSNorm HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 segmented RMSNorm HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    segmented_rmsnorm_f32_host(
        host_input.data(),
        host_weight.data(),
        segments,
        segment_size,
        epsilon,
        host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            input_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 segmented RMSNorm output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 segmented RMSNorm HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipSegmentedRmsNormSiluMulKernelCache {
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
            if (!hiprtc_runtime().compile_segmented_rmsnorm_silu_mul_kernel(arch, &code, &compile_error)) {
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
                    "ullm_segmented_rmsnorm_silu_mul_f32_kernel",
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
            compile_errors.empty() ? "failed to build segmented RMSNorm SiLU-mul HIP kernel"
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

HipSegmentedRmsNormSiluMulKernelCache &hip_segmented_rmsnorm_silu_mul_kernel_cache() {
    static HipSegmentedRmsNormSiluMulKernelCache cache;
    return cache;
}

constexpr size_t kQwen35SegmentedRmsNormSiluMulShufflePrototypeSegments = 32u;
constexpr size_t kQwen35SegmentedRmsNormSiluMulShufflePrototypeSegmentSize = 128u;
constexpr unsigned int kQwen35SegmentedRmsNormSiluMulShufflePrototypeThreads = 256u;

bool segmented_rmsnorm_silu_mul_shuffle_prototype_shape_is_supported(
    const std::string &arch,
    size_t segments,
    size_t segment_size) {
    return arch == "gfx1201" &&
        segments == kQwen35SegmentedRmsNormSiluMulShufflePrototypeSegments &&
        segment_size == kQwen35SegmentedRmsNormSiluMulShufflePrototypeSegmentSize;
}

// This cache is intentionally independent from the served segmented RMSNorm SiLU-mul module.
// The direct experiment therefore cannot affect production HIPRTC source selection, code
// objects, or dispatch.
class HipSegmentedRmsNormSiluMulShufflePrototypeKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second->function;
        }
        if (hip_runtime().device_arch_name(device_id) != "gfx1201") {
            append_error(
                error,
                "segmented RMSNorm SiLU-mul shuffle prototype requires exact gfx1201 architecture");
            return nullptr;
        }

        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_segmented_rmsnorm_silu_mul_shuffle_prototype_kernel(
                "gfx1201",
                &code,
                &compile_error)) {
            append_error(
                error,
                compile_error.empty() ?
                    "failed to build segmented RMSNorm SiLU-mul shuffle prototype HIP kernel" :
                    compile_error);
            return nullptr;
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            append_error(error, "hipModuleLoadData failed for segmented RMSNorm SiLU-mul shuffle prototype");
            return nullptr;
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function,
                module,
                "ullm_segmented_rmsnorm_silu_mul_shuffle_prototype_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            append_error(
                error,
                "hipModuleGetFunction failed for segmented RMSNorm SiLU-mul shuffle prototype");
            return nullptr;
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

HipSegmentedRmsNormSiluMulShufflePrototypeKernelCache &
hip_segmented_rmsnorm_silu_mul_shuffle_prototype_kernel_cache() {
    static HipSegmentedRmsNormSiluMulShufflePrototypeKernelCache cache;
    return cache;
}

bool segmented_rmsnorm_silu_mul_f32_hip_kernel(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    const ullm_runtime_buffer *gate_buffer,
    size_t segments,
    size_t segment_size,
    float epsilon,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = input_buffer->hip_device_id;
    void *function = hip_segmented_rmsnorm_silu_mul_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    if (segments > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "segmented RMSNorm SiLU-mul segment count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_segments = static_cast<unsigned long long>(segments);
    unsigned long long kernel_segment_size = static_cast<unsigned long long>(segment_size);
    void *input_ptr = input_buffer->ptr;
    void *weight_ptr = weight_buffer->ptr;
    void *gate_ptr = gate_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &input_ptr,
        &weight_ptr,
        &gate_ptr,
        &kernel_segments,
        &kernel_segment_size,
        &epsilon,
        &output_ptr,
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
            *error = "hipModuleLaunchKernel failed for f32 segmented RMSNorm SiLU-mul";
        }
        return false;
    }
    return true;
}

bool segmented_rmsnorm_silu_mul_shuffle_prototype_f32_hip_kernel(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    const ullm_runtime_buffer *gate_buffer,
    size_t segments,
    size_t segment_size,
    float epsilon,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = input_buffer->hip_device_id;
    if (!segmented_rmsnorm_silu_mul_shuffle_prototype_shape_is_supported(
            hip_runtime().device_arch_name(device_id),
            segments,
            segment_size)) {
        if (error != nullptr) {
            *error =
                "segmented RMSNorm SiLU-mul shuffle prototype requires gfx1201 and Qwen3.5 segments=32, segment size=128";
        }
        return false;
    }
    void *function = hip_segmented_rmsnorm_silu_mul_shuffle_prototype_kernel_cache()
                         .function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    unsigned long long kernel_segments = static_cast<unsigned long long>(segments);
    unsigned long long kernel_segment_size = static_cast<unsigned long long>(segment_size);
    void *input_ptr = input_buffer->ptr;
    void *weight_ptr = weight_buffer->ptr;
    void *gate_ptr = gate_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &input_ptr,
        &weight_ptr,
        &gate_ptr,
        &kernel_segments,
        &kernel_segment_size,
        &epsilon,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(kQwen35SegmentedRmsNormSiluMulShufflePrototypeSegments),
            kQwen35SegmentedRmsNormSiluMulShufflePrototypeThreads,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for segmented RMSNorm SiLU-mul shuffle prototype";
        }
        return false;
    }
    return true;
}

ullm_status segmented_rmsnorm_silu_mul_f32_hip_staging(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    const ullm_runtime_buffer *gate_buffer,
    size_t segments,
    size_t segment_size,
    float epsilon,
    size_t input_output_bytes,
    size_t weight_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_input(input_output_bytes / sizeof(float));
    std::vector<float> host_weight(weight_bytes / sizeof(float));
    std::vector<float> host_gate(input_output_bytes / sizeof(float));
    std::vector<float> host_output(input_output_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = input_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            input_output_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_weight.data(),
            weight_buffer->ptr,
            weight_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_gate.data(),
            gate_buffer->ptr,
            input_output_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 segmented RMSNorm SiLU-mul HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 segmented RMSNorm SiLU-mul HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    segmented_rmsnorm_silu_mul_f32_host(
        host_input.data(),
        host_weight.data(),
        host_gate.data(),
        segments,
        segment_size,
        epsilon,
        host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            input_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 segmented RMSNorm SiLU-mul output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 segmented RMSNorm SiLU-mul HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

void silu_mul_f32_host(
    const float *gate,
    const float *up,
    size_t elements,
    float *output) {
    for (size_t index = 0; index < elements; ++index) {
        const float gate_value = gate[index];
        const float sigmoid = 1.0f / (1.0f + std::exp(-gate_value));
        output[index] = gate_value * sigmoid * up[index];
    }
}

void sigmoid_mul_f32_host(
    const float *gate,
    const float *input,
    size_t elements,
    float *output) {
    for (size_t index = 0; index < elements; ++index) {
        const float gate_value = gate[index];
        const float sigmoid = 1.0f / (1.0f + std::exp(-gate_value));
        output[index] = sigmoid * input[index];
    }
}

void qwen35_split_q_gate_f32_host(
    const float *projected,
    size_t q_heads,
    size_t head_dim,
    float *query_output,
    float *gate_output) {
    for (size_t head = 0; head < q_heads; ++head) {
        const size_t source_base = head * 2 * head_dim;
        const size_t output_base = head * head_dim;
        for (size_t dim = 0; dim < head_dim; ++dim) {
            query_output[output_base + dim] = projected[source_base + dim];
            gate_output[output_base + dim] = projected[source_base + head_dim + dim];
        }
    }
}

void qwen35_qk_norm_rope_f32_host(
    const float *q_projected,
    const float *k_projected,
    const float *q_weight,
    const float *k_weight,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    float *q_gate_output,
    float *q_rope_output,
    float *k_rope_output) {
    const size_t half = rotary_dim / 2;
    const float position = static_cast<float>(position_offset);
    for (size_t head = 0; head < q_heads; ++head) {
        const size_t source_base = head * 2 * head_dim;
        const size_t output_base = head * head_dim;
        float sum_squares = 0.0f;
        for (size_t dim = 0; dim < head_dim; ++dim) {
            const float value = q_projected[source_base + dim];
            q_gate_output[output_base + dim] = q_projected[source_base + head_dim + dim];
            sum_squares += value * value;
        }
        const float inv_rms =
            1.0f / std::sqrt(sum_squares / static_cast<float>(head_dim) + epsilon);
        for (size_t pair_dim = 0; pair_dim < half; ++pair_dim) {
            const float exponent =
                (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
            const float theta = position / std::pow(rope_base, exponent);
            const float c = std::cos(theta);
            const float s = std::sin(theta);
            const float first =
                q_projected[source_base + pair_dim] * inv_rms * q_weight[pair_dim];
            const size_t second_dim = half + pair_dim;
            const float second =
                q_projected[source_base + second_dim] * inv_rms * q_weight[second_dim];
            q_rope_output[output_base + pair_dim] = first * c - second * s;
            q_rope_output[output_base + second_dim] = second * c + first * s;
        }
        for (size_t dim = rotary_dim; dim < head_dim; ++dim) {
            q_rope_output[output_base + dim] =
                q_projected[source_base + dim] * inv_rms * q_weight[dim];
        }
    }
    for (size_t head = 0; head < kv_heads; ++head) {
        const size_t base = head * head_dim;
        float sum_squares = 0.0f;
        for (size_t dim = 0; dim < head_dim; ++dim) {
            const float value = k_projected[base + dim];
            sum_squares += value * value;
        }
        const float inv_rms =
            1.0f / std::sqrt(sum_squares / static_cast<float>(head_dim) + epsilon);
        for (size_t pair_dim = 0; pair_dim < half; ++pair_dim) {
            const float exponent =
                (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
            const float theta = position / std::pow(rope_base, exponent);
            const float c = std::cos(theta);
            const float s = std::sin(theta);
            const float first = k_projected[base + pair_dim] * inv_rms * k_weight[pair_dim];
            const size_t second_dim = half + pair_dim;
            const float second = k_projected[base + second_dim] * inv_rms * k_weight[second_dim];
            k_rope_output[base + pair_dim] = first * c - second * s;
            k_rope_output[base + second_dim] = second * c + first * s;
        }
        for (size_t dim = rotary_dim; dim < head_dim; ++dim) {
            k_rope_output[base + dim] = k_projected[base + dim] * inv_rms * k_weight[dim];
        }
    }
}

void qwen35_qk_norm_rope_batch_f32_host(
    const float *q_projected,
    const float *k_projected,
    const float *q_weight,
    const float *k_weight,
    size_t q_heads,
    size_t kv_heads,
    size_t sequence_len,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    float *q_gate_output,
    float *q_rope_output,
    float *k_rope_output) {
    const size_t q_projected_stride = q_heads * head_dim * 2;
    const size_t q_output_stride = q_heads * head_dim;
    const size_t k_stride = kv_heads * head_dim;
    for (size_t timestep = 0; timestep < sequence_len; ++timestep) {
        qwen35_qk_norm_rope_f32_host(
            q_projected + timestep * q_projected_stride,
            k_projected + timestep * k_stride,
            q_weight,
            k_weight,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset + timestep,
            rope_base,
            epsilon,
            q_gate_output + timestep * q_output_stride,
            q_rope_output + timestep * q_output_stride,
            k_rope_output + timestep * k_stride);
    }
}

void qwen35_qk_norm_rope_paged_kv_write_f32_host(
    const float *q_projected,
    const float *k_projected,
    const float *v_projected,
    const float *q_weight,
    const float *k_weight,
    const std::uint32_t *block_table,
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
    float *q_gate_output,
    float *q_rope_output,
    float *k_cache,
    float *v_cache) {
    const size_t half = rotary_dim / 2;
    const float position = static_cast<float>(position_offset);
    const size_t block_index = cache_position / block_size;
    const size_t block_offset = cache_position - block_index * block_size;
    const size_t physical_timestep =
        static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
    for (size_t head = 0; head < q_heads; ++head) {
        const size_t source_base = head * 2 * head_dim;
        const size_t output_base = head * head_dim;
        float sum_squares = 0.0f;
        for (size_t dim = 0; dim < head_dim; ++dim) {
            const float value = q_projected[source_base + dim];
            q_gate_output[output_base + dim] = q_projected[source_base + head_dim + dim];
            sum_squares += value * value;
        }
        const float inv_rms =
            1.0f / std::sqrt(sum_squares / static_cast<float>(head_dim) + epsilon);
        for (size_t pair_dim = 0; pair_dim < half; ++pair_dim) {
            const float exponent =
                (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
            const float theta = position / std::pow(rope_base, exponent);
            const float c = std::cos(theta);
            const float s = std::sin(theta);
            const float first =
                q_projected[source_base + pair_dim] * inv_rms * q_weight[pair_dim];
            const size_t second_dim = half + pair_dim;
            const float second =
                q_projected[source_base + second_dim] * inv_rms * q_weight[second_dim];
            q_rope_output[output_base + pair_dim] = first * c - second * s;
            q_rope_output[output_base + second_dim] = second * c + first * s;
        }
        for (size_t dim = rotary_dim; dim < head_dim; ++dim) {
            q_rope_output[output_base + dim] =
                q_projected[source_base + dim] * inv_rms * q_weight[dim];
        }
    }
    for (size_t head = 0; head < kv_heads; ++head) {
        const size_t k_source_base = head * head_dim;
        const size_t k_cache_base = (physical_timestep * kv_heads + head) * head_dim;
        float sum_squares = 0.0f;
        for (size_t dim = 0; dim < head_dim; ++dim) {
            const float value = k_projected[k_source_base + dim];
            sum_squares += value * value;
        }
        const float inv_rms =
            1.0f / std::sqrt(sum_squares / static_cast<float>(head_dim) + epsilon);
        for (size_t pair_dim = 0; pair_dim < half; ++pair_dim) {
            const float exponent =
                (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
            const float theta = position / std::pow(rope_base, exponent);
            const float c = std::cos(theta);
            const float s = std::sin(theta);
            const float first =
                k_projected[k_source_base + pair_dim] * inv_rms * k_weight[pair_dim];
            const size_t second_dim = half + pair_dim;
            const float second =
                k_projected[k_source_base + second_dim] * inv_rms * k_weight[second_dim];
            k_cache[k_cache_base + pair_dim] = first * c - second * s;
            k_cache[k_cache_base + second_dim] = second * c + first * s;
        }
        for (size_t dim = rotary_dim; dim < head_dim; ++dim) {
            k_cache[k_cache_base + dim] =
                k_projected[k_source_base + dim] * inv_rms * k_weight[dim];
        }

        const size_t v_source_base = head * value_dim;
        const size_t v_cache_base = (physical_timestep * kv_heads + head) * value_dim;
        std::copy(
            v_projected + v_source_base,
            v_projected + v_source_base + value_dim,
            v_cache + v_cache_base);
    }
}

void add_f32_host(
    const float *lhs,
    const float *rhs,
    size_t elements,
    float *output) {
    for (size_t index = 0; index < elements; ++index) {
        output[index] = lhs[index] + rhs[index];
    }
}

void rope_f32_host(
    const float *input,
    size_t sequence_len,
    size_t heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float *output) {
    const size_t half = rotary_dim / 2;
    for (size_t timestep = 0; timestep < sequence_len; ++timestep) {
        const float position = static_cast<float>(position_offset + timestep);
        for (size_t head = 0; head < heads; ++head) {
            const size_t base = (timestep * heads + head) * head_dim;
            for (size_t pair_dim = 0; pair_dim < half; ++pair_dim) {
                const float exponent =
                    (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
                const float theta = position / std::pow(rope_base, exponent);
                const float c = std::cos(theta);
                const float s = std::sin(theta);
                const float first = input[base + pair_dim];
                const float second = input[base + half + pair_dim];
                output[base + pair_dim] = first * c - second * s;
                output[base + half + pair_dim] = second * c + first * s;
            }
            for (size_t dim = rotary_dim; dim < head_dim; ++dim) {
                output[base + dim] = input[base + dim];
            }
        }
    }
}

void gemma_proportional_rope_f32_host(
    const float *input,
    size_t sequence_len,
    size_t heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float *output) {
    const size_t half = head_dim / 2;
    const size_t active_pairs = rotary_dim / 2;
    for (size_t timestep = 0; timestep < sequence_len; ++timestep) {
        const float position = static_cast<float>(position_offset + timestep);
        for (size_t head = 0; head < heads; ++head) {
            const size_t base = (timestep * heads + head) * head_dim;
            for (size_t pair_dim = 0; pair_dim < active_pairs; ++pair_dim) {
                const float exponent =
                    (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(head_dim);
                const float theta = position / std::pow(rope_base, exponent);
                const float c = std::cos(theta);
                const float s = std::sin(theta);
                const float first = input[base + pair_dim];
                const float second = input[base + half + pair_dim];
                output[base + pair_dim] = first * c - second * s;
                output[base + half + pair_dim] = second * c + first * s;
            }
            for (size_t dim = active_pairs; dim < half; ++dim) {
                output[base + dim] = input[base + dim];
            }
            for (size_t dim = half + active_pairs; dim < head_dim; ++dim) {
                output[base + dim] = input[base + dim];
            }
        }
    }
}

void causal_attn_f32_host(
    const float *q,
    const float *k,
    const float *v,
    size_t sequence_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float *output) {
    const size_t q_per_kv = q_heads / kv_heads;
    for (size_t timestep = 0; timestep < sequence_len; ++timestep) {
        for (size_t q_head = 0; q_head < q_heads; ++q_head) {
            const size_t kv_head = q_head / q_per_kv;
            const size_t q_base = (timestep * q_heads + q_head) * head_dim;
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t source_timestep = 0; source_timestep <= timestep; ++source_timestep) {
                const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += q[q_base + dim] * k[k_base + dim];
                }
                score *= softmax_scale;
                max_score = std::max(max_score, score);
            }

            float denominator = 0.0f;
            for (size_t source_timestep = 0; source_timestep <= timestep; ++source_timestep) {
                const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += q[q_base + dim] * k[k_base + dim];
                }
                denominator += std::exp(score * softmax_scale - max_score);
            }

            const size_t output_base = (timestep * q_heads + q_head) * value_dim;
            for (size_t value = 0; value < value_dim; ++value) {
                float weighted = 0.0f;
                for (size_t source_timestep = 0; source_timestep <= timestep; ++source_timestep) {
                    const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                    float score = 0.0f;
                    for (size_t dim = 0; dim < head_dim; ++dim) {
                        score += q[q_base + dim] * k[k_base + dim];
                    }
                    const float weight = std::exp(score * softmax_scale - max_score);
                    const size_t v_index =
                        (source_timestep * kv_heads + kv_head) * value_dim + value;
                    weighted += weight * v[v_index];
                }
                output[output_base + value] = weighted / denominator;
            }
        }
    }
}

void decode_attn_f32_host(
    const float *q,
    const float *k_cache,
    const float *v_cache,
    size_t cache_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float *output) {
    const size_t q_per_kv = q_heads / kv_heads;
    for (size_t q_head = 0; q_head < q_heads; ++q_head) {
        const size_t kv_head = q_head / q_per_kv;
        const size_t q_base = q_head * head_dim;
        float max_score = -std::numeric_limits<float>::infinity();
        for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
            const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
            float score = 0.0f;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                score += q[q_base + dim] * k_cache[k_base + dim];
            }
            score *= softmax_scale;
            max_score = std::max(max_score, score);
        }

        float denominator = 0.0f;
        for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
            const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
            float score = 0.0f;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                score += q[q_base + dim] * k_cache[k_base + dim];
            }
            denominator += std::exp(score * softmax_scale - max_score);
        }

        const size_t output_base = q_head * value_dim;
        for (size_t value = 0; value < value_dim; ++value) {
            float weighted = 0.0f;
            for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += q[q_base + dim] * k_cache[k_base + dim];
                }
                const float weight = std::exp(score * softmax_scale - max_score);
                const size_t v_index =
                    (source_timestep * kv_heads + kv_head) * value_dim + value;
                weighted += weight * v_cache[v_index];
            }
            output[output_base + value] = weighted / denominator;
        }
    }
}

void cached_prefix_attn_f32_host(
    const float *q,
    const float *k_cache,
    const float *v_cache,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float *output) {
    const size_t q_per_kv = q_heads / kv_heads;
    for (size_t token_index = 0; token_index < new_tokens; ++token_index) {
        const size_t cache_len = cached_prefix_len + token_index + 1;
        for (size_t q_head = 0; q_head < q_heads; ++q_head) {
            const size_t kv_head = q_head / q_per_kv;
            const size_t q_base = (token_index * q_heads + q_head) * head_dim;
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += q[q_base + dim] * k_cache[k_base + dim];
                }
                score *= softmax_scale;
                max_score = std::max(max_score, score);
            }

            float denominator = 0.0f;
            for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += q[q_base + dim] * k_cache[k_base + dim];
                }
                denominator += std::exp(score * softmax_scale - max_score);
            }

            const size_t output_base = (token_index * q_heads + q_head) * value_dim;
            for (size_t value = 0; value < value_dim; ++value) {
                float weighted = 0.0f;
                for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                    const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                    float score = 0.0f;
                    for (size_t dim = 0; dim < head_dim; ++dim) {
                        score += q[q_base + dim] * k_cache[k_base + dim];
                    }
                    const float weight = std::exp(score * softmax_scale - max_score);
                    const size_t v_index =
                        (source_timestep * kv_heads + kv_head) * value_dim + value;
                    weighted += weight * v_cache[v_index];
                }
                output[output_base + value] = weighted / denominator;
            }
        }
    }
}

float fp8_e4m3_to_f32_unscaled(uint8_t value) {
    const unsigned int raw = static_cast<unsigned int>(value);
    const unsigned int sign = raw >> 7;
    const unsigned int exponent = (raw >> 3) & 0x0fu;
    const unsigned int mantissa = raw & 0x07u;
    float magnitude = 0.0f;
    if (exponent == 0u) {
        magnitude = static_cast<float>(mantissa) * 0.001953125f;
    } else {
        magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) * 0.125f,
                               static_cast<int>(exponent) - 7);
    }
    return sign == 0u ? magnitude : -magnitude;
}

void wmma_fp8_qk_probe_host(
    const uint8_t *q,
    const uint8_t *k,
    float *output) {
    constexpr size_t tile = 16;
    for (size_t row = 0; row < tile; ++row) {
        for (size_t col = 0; col < tile; ++col) {
            float sum = 0.0f;
            for (size_t dim = 0; dim < tile; ++dim) {
                const float q_value = fp8_e4m3_to_f32_unscaled(q[row * tile + dim]);
                const float k_value = fp8_e4m3_to_f32_unscaled(k[col * tile + dim]);
                sum += q_value * k_value;
            }
            output[row * tile + col] = sum;
        }
    }
}

void rocwmma_fp8_attn_probe_host(
    const uint8_t *q,
    const uint8_t *k,
    const float *v,
    float *output) {
    constexpr size_t q_tokens = 16;
    constexpr size_t kv_tokens = 32;
    constexpr size_t head_dim = 16;
    for (size_t row = 0; row < q_tokens; ++row) {
        float scores[kv_tokens];
        float max_score = -std::numeric_limits<float>::infinity();
        for (size_t token = 0; token < kv_tokens; ++token) {
            float score = 0.0f;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                const float q_value = fp8_e4m3_to_f32_unscaled(q[row * head_dim + dim]);
                const float k_value = fp8_e4m3_to_f32_unscaled(k[token * head_dim + dim]);
                score += q_value * k_value;
            }
            scores[token] = score;
            max_score = std::max(max_score, score);
        }

        float denominator = 0.0f;
        for (size_t value = 0; value < head_dim; ++value) {
            output[row * head_dim + value] = 0.0f;
        }
        for (size_t token = 0; token < kv_tokens; ++token) {
            const float weight = std::exp(scores[token] - max_score);
            denominator += weight;
            for (size_t value = 0; value < head_dim; ++value) {
                output[row * head_dim + value] += weight * v[token * head_dim + value];
            }
        }
        for (size_t value = 0; value < head_dim; ++value) {
            output[row * head_dim + value] /= denominator;
        }
    }
}

void cached_prefix_attn_fp8_e4m3_host(
    const float *q,
    const uint8_t *k_cache,
    const uint8_t *v_cache,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float k_scale,
    float v_scale,
    float *output) {
    const size_t q_per_kv = q_heads / kv_heads;
    for (size_t token_index = 0; token_index < new_tokens; ++token_index) {
        const size_t cache_len = cached_prefix_len + token_index + 1;
        for (size_t q_head = 0; q_head < q_heads; ++q_head) {
            const size_t kv_head = q_head / q_per_kv;
            const size_t q_base = (token_index * q_heads + q_head) * head_dim;
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += q[q_base + dim] *
                             (fp8_e4m3_to_f32_unscaled(k_cache[k_base + dim]) * k_scale);
                }
                score *= softmax_scale;
                max_score = std::max(max_score, score);
            }

            float denominator = 0.0f;
            for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += q[q_base + dim] *
                             (fp8_e4m3_to_f32_unscaled(k_cache[k_base + dim]) * k_scale);
                }
                denominator += std::exp(score * softmax_scale - max_score);
            }

            const size_t output_base = (token_index * q_heads + q_head) * value_dim;
            for (size_t value = 0; value < value_dim; ++value) {
                float weighted = 0.0f;
                for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                    const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                    float score = 0.0f;
                    for (size_t dim = 0; dim < head_dim; ++dim) {
                        score += q[q_base + dim] *
                                 (fp8_e4m3_to_f32_unscaled(k_cache[k_base + dim]) * k_scale);
                    }
                    const float weight = std::exp(score * softmax_scale - max_score);
                    const size_t v_index =
                        (source_timestep * kv_heads + kv_head) * value_dim + value;
                    weighted +=
                        weight * (fp8_e4m3_to_f32_unscaled(v_cache[v_index]) * v_scale);
                }
                output[output_base + value] = weighted / denominator;
            }
        }
    }
}

void cached_prefix_attn_fp8_e4m3_rocwmma_host(
    const uint8_t *q,
    const uint8_t *k_cache,
    const uint8_t *v_cache,
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
    float *output) {
    const size_t q_per_kv = q_heads / kv_heads;
    for (size_t token_index = 0; token_index < new_tokens; ++token_index) {
        const size_t cache_len = cached_prefix_len + token_index + 1;
        for (size_t q_head = 0; q_head < q_heads; ++q_head) {
            const size_t kv_head = q_head / q_per_kv;
            const size_t q_base = (token_index * q_heads + q_head) * head_dim;
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += (fp8_e4m3_to_f32_unscaled(q[q_base + dim]) * q_scale) *
                             (fp8_e4m3_to_f32_unscaled(k_cache[k_base + dim]) * k_scale);
                }
                score *= softmax_scale;
                max_score = std::max(max_score, score);
            }

            float denominator = 0.0f;
            for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += (fp8_e4m3_to_f32_unscaled(q[q_base + dim]) * q_scale) *
                             (fp8_e4m3_to_f32_unscaled(k_cache[k_base + dim]) * k_scale);
                }
                denominator += std::exp(score * softmax_scale - max_score);
            }

            const size_t output_base = (token_index * q_heads + q_head) * value_dim;
            for (size_t value = 0; value < value_dim; ++value) {
                float weighted = 0.0f;
                for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                    const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                    float score = 0.0f;
                    for (size_t dim = 0; dim < head_dim; ++dim) {
                        score += (fp8_e4m3_to_f32_unscaled(q[q_base + dim]) * q_scale) *
                                 (fp8_e4m3_to_f32_unscaled(k_cache[k_base + dim]) * k_scale);
                    }
                    const float weight = std::exp(score * softmax_scale - max_score);
                    const size_t v_index =
                        (source_timestep * kv_heads + kv_head) * value_dim + value;
                    weighted +=
                        weight * (fp8_e4m3_to_f32_unscaled(v_cache[v_index]) * v_scale);
                }
                output[output_base + value] = weighted / denominator;
            }
        }
    }
}

void paged_decode_attn_f32_host(
    const float *q,
    const float *k_cache,
    const float *v_cache,
    const std::uint32_t *block_table,
    size_t cache_len,
    size_t block_size,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float *output) {
    const size_t q_per_kv = q_heads / kv_heads;
    for (size_t q_head = 0; q_head < q_heads; ++q_head) {
        const size_t kv_head = q_head / q_per_kv;
        const size_t q_base = q_head * head_dim;
        float max_score = -std::numeric_limits<float>::infinity();
        for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
            const size_t block_index = source_timestep / block_size;
            const size_t block_offset = source_timestep - block_index * block_size;
            const size_t physical_timestep =
                static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
            const size_t k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
            float score = 0.0f;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                score += q[q_base + dim] * k_cache[k_base + dim];
            }
            score *= softmax_scale;
            max_score = std::max(max_score, score);
        }

        float denominator = 0.0f;
        for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
            const size_t block_index = source_timestep / block_size;
            const size_t block_offset = source_timestep - block_index * block_size;
            const size_t physical_timestep =
                static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
            const size_t k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
            float score = 0.0f;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                score += q[q_base + dim] * k_cache[k_base + dim];
            }
            denominator += std::exp(score * softmax_scale - max_score);
        }

        const size_t output_base = q_head * value_dim;
        for (size_t value = 0; value < value_dim; ++value) {
            float weighted = 0.0f;
            for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                const size_t block_index = source_timestep / block_size;
                const size_t block_offset = source_timestep - block_index * block_size;
                const size_t physical_timestep =
                    static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
                const size_t k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += q[q_base + dim] * k_cache[k_base + dim];
                }
                const float weight = std::exp(score * softmax_scale - max_score);
                const size_t v_index =
                    (physical_timestep * kv_heads + kv_head) * value_dim + value;
                weighted += weight * v_cache[v_index];
            }
            output[output_base + value] = weighted / denominator;
        }
    }
}

void paged_kv_write_f32_host(
    const float *k,
    const float *v,
    const std::uint32_t *block_table,
    size_t cache_position,
    size_t block_size,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float *k_cache,
    float *v_cache) {
    const size_t block_index = cache_position / block_size;
    const size_t block_offset = cache_position - block_index * block_size;
    const size_t physical_timestep =
        static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
    for (size_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
        const size_t k_src = kv_head * head_dim;
        const size_t k_dst = (physical_timestep * kv_heads + kv_head) * head_dim;
        std::copy(k + k_src, k + k_src + head_dim, k_cache + k_dst);

        const size_t v_src = kv_head * value_dim;
        const size_t v_dst = (physical_timestep * kv_heads + kv_head) * value_dim;
        std::copy(v + v_src, v + v_src + value_dim, v_cache + v_dst);
    }
}

// Persistent K/V cache helpers.  These intentionally live beside the F32
// host reference so CPU execution and HIP staging share one exact storage
// contract.  They are not SQ8_0 helpers: KV is dynamically appended and its
// E4M3FN scale is independently stored for each (token, KV-head, plane).
bool kv_cache_dtype_is_supported(ullm_kv_cache_dtype dtype) {
    return dtype == ULLM_KV_CACHE_DTYPE_F32 || dtype == ULLM_KV_CACHE_DTYPE_F16 ||
        dtype == ULLM_KV_CACHE_DTYPE_FP8_E4M3FN;
}

size_t kv_cache_dtype_payload_bytes(ullm_kv_cache_dtype dtype) {
    switch (dtype) {
        case ULLM_KV_CACHE_DTYPE_F32:
            return sizeof(float);
        case ULLM_KV_CACHE_DTYPE_F16:
            return sizeof(uint16_t);
        case ULLM_KV_CACHE_DTYPE_FP8_E4M3FN:
            return sizeof(uint8_t);
        default:
            return 0u;
    }
}

bool kv_cache_dtype_requires_scale(ullm_kv_cache_dtype dtype) {
    return dtype == ULLM_KV_CACHE_DTYPE_FP8_E4M3FN;
}

float kv_cache_f16_to_f32(uint16_t bits) {
    const float sign = (bits & 0x8000u) == 0u ? 1.0f : -1.0f;
    const unsigned int exponent = (static_cast<unsigned int>(bits) >> 10u) & 0x1fu;
    const unsigned int mantissa = static_cast<unsigned int>(bits) & 0x03ffu;
    if (exponent == 0u) {
        return mantissa == 0u ? sign * 0.0f : sign * std::ldexp(static_cast<float>(mantissa), -24);
    }
    if (exponent == 0x1fu) {
        return mantissa == 0u ? sign * std::numeric_limits<float>::infinity()
                              : std::numeric_limits<float>::quiet_NaN();
    }
    return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                             static_cast<int>(exponent) - 15);
}

uint32_t kv_cache_round_shift_ties_even(uint32_t value, unsigned int shift) {
    if (shift == 0u) {
        return value;
    }
    const uint32_t quotient = value >> shift;
    const uint32_t remainder = value & ((uint32_t{1} << shift) - 1u);
    const uint32_t halfway = uint32_t{1} << (shift - 1u);
    return remainder > halfway || (remainder == halfway && (quotient & 1u) != 0u)
        ? quotient + 1u
        : quotient;
}

uint16_t kv_cache_f32_to_f16_rne_bits(float value) {
    uint32_t raw = 0u;
    std::memcpy(&raw, &value, sizeof(raw));
    const uint16_t sign = static_cast<uint16_t>((raw >> 16u) & 0x8000u);
    const uint32_t exponent_bits = (raw >> 23u) & 0xffu;
    const uint32_t mantissa = raw & 0x7fffffu;
    if (exponent_bits == 0xffu) {
        return static_cast<uint16_t>(sign | (mantissa == 0u ? 0x7c00u : 0x7e00u));
    }
    const int exponent = static_cast<int>(exponent_bits) - 127;
    if (exponent > 15) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    if (exponent >= -14) {
        uint32_t rounded = kv_cache_round_shift_ties_even(mantissa, 13u);
        uint32_t half_exponent = static_cast<uint32_t>(exponent + 15);
        if (rounded == 0x400u) {
            rounded = 0u;
            ++half_exponent;
            if (half_exponent >= 0x1fu) {
                return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }
        return static_cast<uint16_t>(sign | (half_exponent << 10u) | rounded);
    }
    if (exponent < -25) {
        return sign;
    }
    const uint32_t significand = mantissa | 0x800000u;
    return static_cast<uint16_t>(
        sign | kv_cache_round_shift_ties_even(significand, static_cast<unsigned int>(-exponent - 1)));
}

bool kv_cache_f32_to_positive_f16_ceil(float value, uint16_t *bits_out) {
    if (bits_out == nullptr || !std::isfinite(value) || value <= 0.0f) {
        return false;
    }
    uint16_t bits = kv_cache_f32_to_f16_rne_bits(value);
    // Preserve a positive scale even below the smallest representable F16
    // subnormal.  That is preferable to silently mapping a nonzero head to
    // a zero-scale cache row.
    if (bits == 0u) {
        bits = 1u;
    }
    if (bits >= 0x7c00u) {
        return false;
    }
    if (kv_cache_f16_to_f32(bits) < value) {
        ++bits;
        if (bits >= 0x7c00u) {
            return false;
        }
    }
    const float stored = kv_cache_f16_to_f32(bits);
    if (!std::isfinite(stored) || stored <= 0.0f || stored < value) {
        return false;
    }
    *bits_out = bits;
    return true;
}

float kv_cache_fp8_e4m3fn_to_f32(uint8_t value) {
    const unsigned int raw = static_cast<unsigned int>(value);
    const unsigned int sign = raw >> 7u;
    const unsigned int exponent = (raw >> 3u) & 0x0fu;
    const unsigned int mantissa = raw & 0x07u;
    // OCP E4M3FN reserves +/-0x7f as NaN; the dynamically-written encoder
    // below never emits it for finite values.
    if (exponent == 0x0fu && mantissa == 0x07u) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float magnitude = exponent == 0u
        ? static_cast<float>(mantissa) * 0.001953125f
        : std::ldexp(1.0f + static_cast<float>(mantissa) * 0.125f,
                     static_cast<int>(exponent) - 7);
    return sign == 0u ? magnitude : -magnitude;
}

uint8_t kv_cache_f32_to_fp8_e4m3fn_rne(float value, float scale) {
    uint32_t raw = 0u;
    std::memcpy(&raw, &value, sizeof(raw));
    const unsigned int sign = (raw >> 24u) & 0x80u;
    const float magnitude = std::fabs(value / scale);
    if (magnitude == 0.0f) {
        return static_cast<uint8_t>(sign);
    }
    if (magnitude >= 448.0f) {
        return static_cast<uint8_t>(sign | 0x7eu);
    }
    unsigned int lower_bound = 0u;
    unsigned int upper_bound = 126u;
    while (lower_bound < upper_bound) {
        const unsigned int middle = lower_bound + (upper_bound - lower_bound) / 2u;
        if (std::fabs(kv_cache_fp8_e4m3fn_to_f32(static_cast<uint8_t>(middle))) < magnitude) {
            lower_bound = middle + 1u;
        } else {
            upper_bound = middle;
        }
    }
    const unsigned int upper = lower_bound;
    const float upper_value = kv_cache_fp8_e4m3fn_to_f32(static_cast<uint8_t>(upper));
    if (upper_value == magnitude) {
        return static_cast<uint8_t>(sign | upper);
    }
    const unsigned int lower = upper - 1u;
    const double lower_distance = static_cast<double>(magnitude) -
        static_cast<double>(kv_cache_fp8_e4m3fn_to_f32(static_cast<uint8_t>(lower)));
    const double upper_distance = static_cast<double>(upper_value) - static_cast<double>(magnitude);
    const unsigned int encoded = lower_distance < upper_distance
        ? lower
        : (upper_distance < lower_distance ? upper : ((lower & 1u) == 0u ? lower : upper));
    return static_cast<uint8_t>(sign | encoded);
}

bool kv_cache_write_typed_plane_f32(
    const float *source,
    size_t elements,
    ullm_kv_cache_dtype dtype,
    void *payload,
    uint16_t *scale_slot) {
    if (source == nullptr || payload == nullptr || !kv_cache_dtype_is_supported(dtype)) {
        return false;
    }
    if (dtype == ULLM_KV_CACHE_DTYPE_F32) {
        std::copy(source, source + elements, static_cast<float *>(payload));
        return true;
    }
    if (dtype == ULLM_KV_CACHE_DTYPE_F16) {
        auto *destination = static_cast<uint16_t *>(payload);
        for (size_t index = 0; index < elements; ++index) {
            if (!std::isfinite(source[index])) {
                return false;
            }
            const uint16_t bits = kv_cache_f32_to_f16_rne_bits(source[index]);
            if ((bits & 0x7c00u) == 0x7c00u) {
                return false;
            }
            destination[index] = bits;
        }
        return true;
    }
    if (scale_slot == nullptr) {
        return false;
    }
    float maximum = 0.0f;
    for (size_t index = 0; index < elements; ++index) {
        if (!std::isfinite(source[index])) {
            return false;
        }
        maximum = std::max(maximum, std::fabs(source[index]));
    }
    uint16_t scale_bits = 0x3c00u; // exact 1.0 for an all-zero head.
    if (maximum != 0.0f &&
        !kv_cache_f32_to_positive_f16_ceil(maximum / 448.0f, &scale_bits)) {
        return false;
    }
    const float scale = kv_cache_f16_to_f32(scale_bits);
    auto *destination = static_cast<uint8_t *>(payload);
    for (size_t index = 0; index < elements; ++index) {
        destination[index] = kv_cache_f32_to_fp8_e4m3fn_rne(source[index], scale);
    }
    *scale_slot = scale_bits;
    return true;
}

// Validate before mutating either plane.  A K/V cache row is one logical
// transaction: if V cannot be represented after K has been accepted, callers
// must not observe a half-written token.  F32 intentionally retains its
// legacy pass-through behavior; the non-F32 formats reject non-finite values
// and F16 overflow before any payload or scale store occurs.
bool kv_cache_validate_typed_plane_f32(
    const float *source,
    size_t elements,
    ullm_kv_cache_dtype dtype) {
    if (source == nullptr || !kv_cache_dtype_is_supported(dtype)) {
        return false;
    }
    if (dtype == ULLM_KV_CACHE_DTYPE_F32) {
        return true;
    }
    float maximum = 0.0f;
    for (size_t index = 0; index < elements; ++index) {
        if (!std::isfinite(source[index])) {
            return false;
        }
        if (dtype == ULLM_KV_CACHE_DTYPE_F16) {
            const uint16_t bits = kv_cache_f32_to_f16_rne_bits(source[index]);
            if ((bits & 0x7c00u) == 0x7c00u) {
                return false;
            }
        } else {
            maximum = std::max(maximum, std::fabs(source[index]));
        }
    }
    if (dtype != ULLM_KV_CACHE_DTYPE_FP8_E4M3FN || maximum == 0.0f) {
        return true;
    }
    uint16_t scale_bits = 0u;
    return kv_cache_f32_to_positive_f16_ceil(maximum / 448.0f, &scale_bits);
}

bool kv_cache_read_typed_value(
    const void *payload,
    ullm_kv_cache_dtype dtype,
    const uint16_t *scale_slot,
    size_t index,
    float *value_out) {
    if (payload == nullptr || value_out == nullptr || !kv_cache_dtype_is_supported(dtype)) {
        return false;
    }
    float value = 0.0f;
    if (dtype == ULLM_KV_CACHE_DTYPE_F32) {
        value = static_cast<const float *>(payload)[index];
    } else if (dtype == ULLM_KV_CACHE_DTYPE_F16) {
        value = kv_cache_f16_to_f32(static_cast<const uint16_t *>(payload)[index]);
    } else {
        if (scale_slot == nullptr) {
            return false;
        }
        const uint16_t scale_bits = *scale_slot;
        const float scale = kv_cache_f16_to_f32(scale_bits);
        // A zero scale denotes a reset/unwritten row.  Its zero payload is
        // semantically zero and must not turn diagnostic readback into NaN.
        if (scale_bits == 0u) {
            *value_out = 0.0f;
            return true;
        }
        // Writer-produced FP8 scales are finite and strictly positive.  Do
        // not silently accept a corrupted negative, infinite, or NaN FP16
        // scale: negative scales would flip values without an observable
        // failure, which violates the cache's encoding contract.
        if (!std::isfinite(scale) || scale <= 0.0f) {
            return false;
        }
        value = kv_cache_fp8_e4m3fn_to_f32(static_cast<const uint8_t *>(payload)[index]) * scale;
    }
    if (!std::isfinite(value)) {
        return false;
    }
    *value_out = value;
    return true;
}

bool paged_kv_write_typed_f32_host(
    const float *k,
    const float *v,
    const std::uint32_t *block_table,
    size_t cache_position,
    size_t block_size,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    ullm_kv_cache_dtype k_dtype,
    ullm_kv_cache_dtype v_dtype,
    void *k_cache,
    void *v_cache,
    uint16_t *k_scales,
    uint16_t *v_scales) {
    const size_t block_index = cache_position / block_size;
    const size_t block_offset = cache_position - block_index * block_size;
    const size_t physical_timestep =
        static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
    const size_t k_payload_bytes = kv_cache_dtype_payload_bytes(k_dtype);
    const size_t v_payload_bytes = kv_cache_dtype_payload_bytes(v_dtype);
    for (size_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
        if (!kv_cache_validate_typed_plane_f32(
                k + kv_head * head_dim, head_dim, k_dtype) ||
            !kv_cache_validate_typed_plane_f32(
                v + kv_head * value_dim, value_dim, v_dtype)) {
            return false;
        }
    }
    for (size_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
        const size_t physical_head = physical_timestep * kv_heads + kv_head;
        const size_t k_source_offset = kv_head * head_dim;
        const size_t v_source_offset = kv_head * value_dim;
        auto *k_destination = static_cast<uint8_t *>(k_cache) +
            physical_head * head_dim * k_payload_bytes;
        auto *v_destination = static_cast<uint8_t *>(v_cache) +
            physical_head * value_dim * v_payload_bytes;
        uint16_t *k_scale = k_dtype == ULLM_KV_CACHE_DTYPE_FP8_E4M3FN
            ? k_scales + physical_head
            : nullptr;
        uint16_t *v_scale = v_dtype == ULLM_KV_CACHE_DTYPE_FP8_E4M3FN
            ? v_scales + physical_head
            : nullptr;
        if (!kv_cache_write_typed_plane_f32(
                k + k_source_offset, head_dim, k_dtype, k_destination, k_scale) ||
            !kv_cache_write_typed_plane_f32(
                v + v_source_offset, value_dim, v_dtype, v_destination, v_scale)) {
            return false;
        }
    }
    return true;
}

bool paged_decode_attn_typed_f32_host(
    const float *q,
    const void *k_cache,
    const void *v_cache,
    const std::uint32_t *block_table,
    const uint16_t *k_scales,
    const uint16_t *v_scales,
    size_t cache_len,
    size_t block_size,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_kv_cache_dtype k_dtype,
    ullm_kv_cache_dtype v_dtype,
    float *output) {
    const size_t q_per_kv = q_heads / kv_heads;
    const size_t k_payload_bytes = kv_cache_dtype_payload_bytes(k_dtype);
    const size_t v_payload_bytes = kv_cache_dtype_payload_bytes(v_dtype);
    for (size_t q_head = 0; q_head < q_heads; ++q_head) {
        const size_t kv_head = q_head / q_per_kv;
        const size_t q_base = q_head * head_dim;
        float max_score = -std::numeric_limits<float>::infinity();
        for (size_t source = 0; source < cache_len; ++source) {
            const size_t block_index = source / block_size;
            const size_t block_offset = source - block_index * block_size;
            const size_t physical =
                static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
            const size_t physical_head = physical * kv_heads + kv_head;
            const auto *k_plane = static_cast<const uint8_t *>(k_cache) +
                physical_head * head_dim * k_payload_bytes;
            const uint16_t *k_scale = k_dtype == ULLM_KV_CACHE_DTYPE_FP8_E4M3FN
                ? k_scales + physical_head
                : nullptr;
            float score = 0.0f;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                float key = 0.0f;
                if (!kv_cache_read_typed_value(k_plane, k_dtype, k_scale, dim, &key)) {
                    return false;
                }
                score += q[q_base + dim] * key;
            }
            max_score = std::max(max_score, score * softmax_scale);
        }
        float denominator = 0.0f;
        std::vector<float> weighted(value_dim, 0.0f);
        for (size_t source = 0; source < cache_len; ++source) {
            const size_t block_index = source / block_size;
            const size_t block_offset = source - block_index * block_size;
            const size_t physical =
                static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
            const size_t physical_head = physical * kv_heads + kv_head;
            const auto *k_plane = static_cast<const uint8_t *>(k_cache) +
                physical_head * head_dim * k_payload_bytes;
            const auto *v_plane = static_cast<const uint8_t *>(v_cache) +
                physical_head * value_dim * v_payload_bytes;
            const uint16_t *k_scale = k_dtype == ULLM_KV_CACHE_DTYPE_FP8_E4M3FN
                ? k_scales + physical_head
                : nullptr;
            const uint16_t *v_scale = v_dtype == ULLM_KV_CACHE_DTYPE_FP8_E4M3FN
                ? v_scales + physical_head
                : nullptr;
            float score = 0.0f;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                float key = 0.0f;
                if (!kv_cache_read_typed_value(k_plane, k_dtype, k_scale, dim, &key)) {
                    return false;
                }
                score += q[q_base + dim] * key;
            }
            const float weight = std::exp(score * softmax_scale - max_score);
            denominator += weight;
            for (size_t value = 0; value < value_dim; ++value) {
                float cached_value = 0.0f;
                if (!kv_cache_read_typed_value(v_plane, v_dtype, v_scale, value, &cached_value)) {
                    return false;
                }
                weighted[value] += weight * cached_value;
            }
        }
        if (!std::isfinite(denominator) || denominator <= 0.0f) {
            return false;
        }
        const size_t output_base = q_head * value_dim;
        for (size_t value = 0; value < value_dim; ++value) {
            output[output_base + value] = weighted[value] / denominator;
        }
    }
    return true;
}

void paged_kv_write_chunk_f32_host(
    const float *k,
    const float *v,
    const std::uint32_t *block_table,
    size_t cache_start,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float *k_cache,
    float *v_cache) {
    (void)cache_blocks;
    for (size_t row = 0; row < m; ++row) {
        const size_t logical_position = cache_start + row;
        const size_t block_index = logical_position / block_size;
        const size_t block_offset = logical_position - block_index * block_size;
        const size_t physical_timestep =
            static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
        for (size_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
            const size_t k_src = (row * kv_heads + kv_head) * head_dim;
            const size_t k_dst = (physical_timestep * kv_heads + kv_head) * head_dim;
            std::copy(k + k_src, k + k_src + head_dim, k_cache + k_dst);
            const size_t v_src = (row * kv_heads + kv_head) * value_dim;
            const size_t v_dst = (physical_timestep * kv_heads + kv_head) * value_dim;
            std::copy(v + v_src, v + v_src + value_dim, v_cache + v_dst);
        }
    }
}

void paged_causal_gqa_chunk_f32_host(
    const float *q,
    const float *k_cache,
    const float *v_cache,
    const std::uint32_t *block_table,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float *output) {
    const size_t q_per_kv = q_heads / kv_heads;
    for (size_t row = 0; row < m; ++row) {
        const size_t cache_len = cached_prefix_len + row + 1;
        for (size_t q_head = 0; q_head < q_heads; ++q_head) {
            const size_t kv_head = q_head / q_per_kv;
            const size_t q_base = (row * q_heads + q_head) * head_dim;
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t source = 0; source < cache_len; ++source) {
                const size_t block_index = source / block_size;
                const size_t block_offset = source - block_index * block_size;
                const size_t physical =
                    static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
                const size_t k_base = (physical * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += q[q_base + dim] * k_cache[k_base + dim];
                }
                max_score = std::max(max_score, score * softmax_scale);
            }
            float denominator = 0.0f;
            const size_t output_base = (row * q_heads + q_head) * value_dim;
            std::vector<float> weighted(value_dim, 0.0f);
            for (size_t source = 0; source < cache_len; ++source) {
                const size_t block_index = source / block_size;
                const size_t block_offset = source - block_index * block_size;
                const size_t physical =
                    static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
                const size_t k_base = (physical * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += q[q_base + dim] * k_cache[k_base + dim];
                }
                const float weight = std::exp(score * softmax_scale - max_score);
                denominator += weight;
                const size_t v_base = (physical * kv_heads + kv_head) * value_dim;
                for (size_t value = 0; value < value_dim; ++value) {
                    weighted[value] += weight * v_cache[v_base + value];
                }
            }
            for (size_t value = 0; value < value_dim; ++value) {
                output[output_base + value] = weighted[value] / denominator;
            }
        }
    }
}

class HipSiluMulKernelCache {
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
            if (!hiprtc_runtime().compile_silu_mul_kernel(arch, &code, &compile_error)) {
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
                    "ullm_silu_mul_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build SiLU-mul HIP kernel" : compile_errors);
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

HipSiluMulKernelCache &hip_silu_mul_kernel_cache() {
    static HipSiluMulKernelCache cache;
    return cache;
}

bool silu_mul_f32_hip_kernel(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *up_buffer,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = gate_buffer->hip_device_id;
    void *function = hip_silu_mul_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t grid_size = (elements + block_size - 1) / block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "SiLU-mul element count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_elements = static_cast<unsigned long long>(elements);
    void *gate_ptr = gate_buffer->ptr;
    void *up_ptr = up_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &gate_ptr,
        &up_ptr,
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
            *error = "hipModuleLaunchKernel failed for f32 SiLU-mul";
        }
        return false;
    }
    return true;
}

ullm_status silu_mul_f32_hip_staging(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *up_buffer,
    size_t elements,
    size_t required_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_gate(elements);
    std::vector<float> host_up(elements);
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
            host_up.data(),
            up_buffer->ptr,
            required_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 SiLU-mul HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 SiLU-mul HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    silu_mul_f32_host(host_gate.data(), host_up.data(), elements, host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 SiLU-mul output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 SiLU-mul HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

void gelu_tanh_mul_f32_host(
    const float *gate,
    const float *up,
    size_t elements,
    float *output) {
    constexpr float coefficient = 0.7978845608f;
    for (size_t index = 0; index < elements; ++index) {
        const float value = gate[index];
        // Keep this exact operation order aligned with Transformers GELUTanh.
        const float activated = 0.5f * value *
            (1.0f + std::tanh(value * coefficient * (1.0f + 0.044715f * value * value)));
        output[index] = activated * up[index];
    }
}

class HipGeluTanhMulKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) return found->second->function;
        std::string failures;
        for (const std::string &arch : hip_arch_candidates(device_id)) {
            std::vector<char> code;
            std::string compile_error;
            if (!hiprtc_runtime().compile_gelu_tanh_mul_kernel(arch, &code, &compile_error)) {
                failures += compile_error + "\n";
                continue;
            }
            void *module = nullptr;
            void *function = nullptr;
            if (!hip_runtime().module_load_data(&module, code.data(), device_id) ||
                !hip_runtime().module_get_function(
                    &function, module, "ullm_gelu_tanh_mul_f32_kernel", device_id)) {
                if (module != nullptr) hip_runtime().module_unload(module, device_id);
                failures += "failed to load GELUTanh-mul HIP module for " + arch + "\n";
                continue;
            }
            auto loaded = std::make_unique<LoadedModule>();
            loaded->module = module;
            loaded->function = function;
            void *result = function;
            modules_.emplace(device_id, std::move(loaded));
            return result;
        }
        if (error != nullptr) *error = failures.empty() ? "failed to build GELUTanh-mul HIP kernel" : failures;
        return nullptr;
    }
private:
    struct LoadedModule { void *module = nullptr; void *function = nullptr; };
    std::mutex mutex_;
    std::unordered_map<int, std::unique_ptr<LoadedModule>> modules_;
};

HipGeluTanhMulKernelCache &hip_gelu_tanh_mul_kernel_cache() {
    static HipGeluTanhMulKernelCache cache;
    return cache;
}

bool gelu_tanh_mul_f32_hip_kernel(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *up_buffer,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    void *function = hip_gelu_tanh_mul_kernel_cache().function_for_device(gate_buffer->hip_device_id, error);
    if (function == nullptr) return false;
    constexpr unsigned int block_size = 256;
    const size_t grid_size = (elements + block_size - 1) / block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) *error = "GELUTanh-mul element count exceeds HIP grid limit";
        return false;
    }
    unsigned long long kernel_elements = static_cast<unsigned long long>(elements);
    void *gate_ptr = gate_buffer->ptr;
    void *up_ptr = up_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *params[] = {&gate_ptr, &up_ptr, &kernel_elements, &output_ptr};
    if (!hip_runtime().module_launch_kernel(
            function, static_cast<unsigned int>(grid_size), block_size, params,
            stream == nullptr ? nullptr : stream->stream, gate_buffer->hip_device_id)) {
        if (error != nullptr) *error = "hipModuleLaunchKernel failed for f32 GELUTanh-mul";
        return false;
    }
    return true;
}

ullm_status gelu_tanh_mul_f32_hip_staging(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *up_buffer,
    size_t elements,
    size_t required_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> gate(elements), up(elements), output(elements);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = gate_buffer->hip_device_id;
    if (!hip_runtime().copy_async(gate.data(), gate_buffer->ptr, required_bytes, HIP_MEMCPY_DEVICE_TO_HOST, hip_stream, device_id) ||
        !hip_runtime().copy_async(up.data(), up_buffer->ptr, required_bytes, HIP_MEMCPY_DEVICE_TO_HOST, hip_stream, device_id) ||
        !synchronize_hip_staging(stream, device_id)) {
        set_error("failed to copy f32 GELUTanh-mul inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    gelu_tanh_mul_f32_host(gate.data(), up.data(), elements, output.data());
    if (!hip_runtime().copy_async(output_buffer->ptr, output.data(), required_bytes, HIP_MEMCPY_HOST_TO_DEVICE, hip_stream, device_id) ||
        !synchronize_hip_staging(stream, device_id)) {
        set_error("failed to copy f32 GELUTanh-mul output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipSigmoidMulKernelCache {
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
            if (!hiprtc_runtime().compile_sigmoid_mul_kernel(arch, &code, &compile_error)) {
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
                    "ullm_sigmoid_mul_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build sigmoid-mul HIP kernel" : compile_errors);
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

HipSigmoidMulKernelCache &hip_sigmoid_mul_kernel_cache() {
    static HipSigmoidMulKernelCache cache;
    return cache;
}

bool sigmoid_mul_f32_hip_kernel(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = gate_buffer->hip_device_id;
    void *function = hip_sigmoid_mul_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t grid_size = (elements + block_size - 1) / block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "sigmoid-mul element count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_elements = static_cast<unsigned long long>(elements);
    void *gate_ptr = gate_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &gate_ptr,
        &input_ptr,
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
            *error = "hipModuleLaunchKernel failed for f32 sigmoid-mul";
        }
        return false;
    }
    return true;
}
