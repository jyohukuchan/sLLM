// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
class HipAq4KernelCache {
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
            if (!hiprtc_runtime().compile_aq4_kernel(arch, &code, &compile_error)) {
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
                    "ullm_aq4_dequant_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build AQ4 HIP kernel" : compile_errors);
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

HipAq4KernelCache &hip_aq4_kernel_cache() {
    static HipAq4KernelCache cache;
    return cache;
}

class HipAq4RowKernelCache {
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
            if (!hiprtc_runtime().compile_aq4_row_kernel(arch, &code, &compile_error)) {
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
                    "ullm_aq4_row_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build AQ4 row HIP kernel" : compile_errors);
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

HipAq4RowKernelCache &hip_aq4_row_kernel_cache() {
    static HipAq4RowKernelCache cache;
    return cache;
}

enum class HipAq4LaunchResult {
    Ok,
    InvalidArgument,
    RuntimeError,
};

HipAq4LaunchResult aq4_dequant_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const float *scale_values,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t elements,
    size_t required_output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    void *function = hip_aq4_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return HipAq4LaunchResult::RuntimeError;
    }

    const size_t scale_value_bytes = scale_count * sizeof(float);
    void *device_scale_values = hip_runtime().malloc_device(scale_value_bytes, device_id);
    void *device_error = hip_runtime().malloc_device(sizeof(std::uint32_t), device_id);
    if (device_scale_values == nullptr || device_error == nullptr) {
        if (device_scale_values != nullptr) {
            hip_runtime().free_device(device_scale_values, device_id);
        }
        if (device_error != nullptr) {
            hip_runtime().free_device(device_error, device_id);
        }
        if (error != nullptr) {
            *error = "failed to allocate AQ4 HIP kernel temporary buffers";
        }
        return HipAq4LaunchResult::RuntimeError;
    }

    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const std::uint32_t zero = 0;
    if (!hip_runtime().copy_async(
            device_scale_values,
            scale_values,
            scale_value_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            device_error,
            &zero,
            sizeof(zero),
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        hip_runtime().free_device(device_scale_values, device_id);
        hip_runtime().free_device(device_error, device_id);
        if (error != nullptr) {
            *error = "failed to upload AQ4 HIP kernel temporary buffers";
        }
        return HipAq4LaunchResult::RuntimeError;
    }

    constexpr unsigned int block_size = 256;
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_elements = static_cast<unsigned long long>(elements);
    const unsigned int grid_x =
        static_cast<unsigned int>((elements + block_size - 1) / block_size);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &device_scale_values,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_elements,
        &output_ptr,
        &device_error,
    };
    if (!hip_runtime().module_launch_kernel(
            function,
            grid_x,
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        hip_runtime().free_device(device_scale_values, device_id);
        hip_runtime().free_device(device_error, device_id);
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 dequant";
        }
        return HipAq4LaunchResult::RuntimeError;
    }

    std::uint32_t host_error = 0;
    if (!hip_runtime().copy_async(
            &host_error,
            device_error,
            sizeof(host_error),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !synchronize_hip_staging(stream, device_id)) {
        hip_runtime().free_device(device_scale_values, device_id);
        hip_runtime().free_device(device_error, device_id);
        if (error != nullptr) {
            *error = "failed to read AQ4 HIP kernel status";
        }
        return HipAq4LaunchResult::RuntimeError;
    }

    hip_runtime().free_device(device_scale_values, device_id);
    hip_runtime().free_device(device_error, device_id);
    if (host_error != 0) {
        if (error != nullptr) {
            *error = "AQ4 dequant scale index is out of range";
        }
        return HipAq4LaunchResult::InvalidArgument;
    }
    (void)required_output_bytes;
    return HipAq4LaunchResult::Ok;
}

HipAq4LaunchResult aq4_row_f32_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t row_index,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    void *function = hip_aq4_row_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return HipAq4LaunchResult::RuntimeError;
    }

    void *device_error = hip_runtime().malloc_device(sizeof(std::uint32_t), device_id);
    if (device_error == nullptr) {
        if (error != nullptr) {
            *error = "failed to allocate AQ4 row HIP kernel status buffer";
        }
        return HipAq4LaunchResult::RuntimeError;
    }

    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const std::uint32_t zero = 0;
    if (!hip_runtime().copy_async(
            device_error,
            &zero,
            sizeof(zero),
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        hip_runtime().free_device(device_error, device_id);
        if (error != nullptr) {
            *error = "failed to upload AQ4 row HIP kernel status buffer";
        }
        return HipAq4LaunchResult::RuntimeError;
    }

    constexpr unsigned int block_size = 256;
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_row_index = static_cast<unsigned long long>(row_index);
    const unsigned int grid_x =
        static_cast<unsigned int>((cols + block_size - 1) / block_size);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *row_scales_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &scale_values_ptr,
        &row_scales_ptr,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_row_scale_count,
        &kernel_rows,
        &kernel_cols,
        &kernel_row_index,
        &output_ptr,
        &device_error,
    };
    if (!hip_runtime().module_launch_kernel(
            function,
            grid_x,
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        hip_runtime().free_device(device_error, device_id);
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 row";
        }
        return HipAq4LaunchResult::RuntimeError;
    }

    std::uint32_t host_error = 0;
    if (!hip_runtime().copy_async(
            &host_error,
            device_error,
            sizeof(host_error),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !synchronize_hip_staging(stream, device_id)) {
        hip_runtime().free_device(device_error, device_id);
        if (error != nullptr) {
            *error = "failed to read AQ4 row HIP kernel status";
        }
        return HipAq4LaunchResult::RuntimeError;
    }

    hip_runtime().free_device(device_error, device_id);
    if (host_error != 0) {
        if (error != nullptr) {
            *error = "AQ4 row scale index is out of range";
        }
        return HipAq4LaunchResult::InvalidArgument;
    }
    return HipAq4LaunchResult::Ok;
}

ullm_status aq4_dequant_hip_staging(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const float *scale_values,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t elements,
    size_t required_index_bytes,
    size_t groups,
    size_t required_output_bytes,
    size_t codebook_entries,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<std::uint8_t> host_indices(required_index_bytes);
    std::vector<std::uint8_t> host_scale_indices(groups);
    std::vector<float> host_codebook(codebook_entries);
    std::vector<float> host_output(elements);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = index_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_indices.data(),
            index_buffer->ptr,
            required_index_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_scale_indices.data(),
            scale_buffer->ptr,
            groups,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_codebook.data(),
            codebook_buffer->ptr,
            codebook_buffer->bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!aq4_dequant_host(
            host_indices.data(),
            host_scale_indices.data(),
            host_codebook.data(),
            scale_values,
            scale_count,
            group_size,
            tensor_scale,
            elements,
            host_output.data())) {
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 materialized output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

bool aq4_matvec_f32_host(
    const std::uint8_t *indices,
    const std::uint8_t *scale_indices,
    const float *codebook,
    const float *scale_values,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    const float *input,
    const float *row_scales,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    float *output) {
    for (size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        const size_t row_offset = row * cols;
        for (size_t col = 0; col < cols; ++col) {
            const size_t element = row_offset + col;
            const std::uint8_t packed = indices[element / 2];
            const std::uint8_t codebook_index =
                (element % 2 == 0) ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
            const size_t group = element / group_size;
            const size_t scale_index = static_cast<size_t>(scale_indices[group]);
            if (scale_index >= scale_count) {
                set_error("AQ4 matvec scale index is out of range");
                return false;
            }
            const float value = codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
            sum += value * input[col];
        }
        output[row] = row_scales != nullptr && row < row_scale_count ? sum * row_scales[row] : sum;
    }
    return true;
}

bool aq4_matvec_batch_f32_host(
    const std::uint8_t *indices,
    const std::uint8_t *scale_indices,
    const float *codebook,
    const float *scale_values,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    const float *input,
    const float *row_scales,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t batch_count,
    float *output) {
    for (size_t batch = 0; batch < batch_count; ++batch) {
        if (!aq4_matvec_f32_host(
                indices,
                scale_indices,
                codebook,
                scale_values,
                scale_count,
                group_size,
                tensor_scale,
                input + batch * cols,
                row_scales,
                row_scale_count,
                rows,
                cols,
                output + batch * rows)) {
            return false;
        }
    }
    return true;
}

size_t aq4_matvec_top1_partial_count(size_t rows) {
    const unsigned int rows_per_block = aq4_matvec_top1_rows_per_block_from_env();
    return (rows + rows_per_block - 1) / rows_per_block;
}

bool aq4_matvec_top1_f32_host(
    const std::uint8_t *indices,
    const std::uint8_t *scale_indices,
    const float *codebook,
    const float *scale_values,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    const float *input,
    const float *row_scales,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t partial_count,
    float *partial_values,
    uint32_t *partial_indices) {
    std::vector<float> output(rows);
    if (!aq4_matvec_f32_host(
            indices,
            scale_indices,
            codebook,
            scale_values,
            scale_count,
            group_size,
            tensor_scale,
            input,
            row_scales,
            row_scale_count,
            rows,
            cols,
            output.data())) {
        return false;
    }
    for (size_t partial = 0; partial < partial_count; ++partial) {
        const size_t start = partial * kAq4MatvecTop1RowsPerBlock;
        const size_t end = std::min(start + kAq4MatvecTop1RowsPerBlock, rows);
        float best_value = -std::numeric_limits<float>::max();
        uint32_t best_index = std::numeric_limits<uint32_t>::max();
        for (size_t row = start; row < end; ++row) {
            float value = output[row];
            if (value != value) {
                value = -std::numeric_limits<float>::max();
            }
            const auto token_index = static_cast<uint32_t>(row);
            if (value > best_value || (value == best_value && token_index < best_index)) {
                best_value = value;
                best_index = token_index;
            }
        }
        partial_values[partial] = best_value;
        partial_indices[partial] = best_index;
    }
    return true;
}

bool aq4_matvec_add_f32_host(
    const std::uint8_t *indices,
    const std::uint8_t *scale_indices,
    const float *codebook,
    const float *scale_values,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    const float *input,
    const float *residual,
    const float *row_scales,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    float *output) {
    std::vector<float> matvec_output(rows);
    if (!aq4_matvec_f32_host(
            indices,
            scale_indices,
            codebook,
            scale_values,
            scale_count,
            group_size,
            tensor_scale,
            input,
            row_scales,
            row_scale_count,
            rows,
            cols,
            matvec_output.data())) {
        return false;
    }
    for (size_t row = 0; row < rows; ++row) {
        output[row] = residual[row] + matvec_output[row];
    }
    return true;
}

bool aq4_matvec_silu_mul_f32_host(
    const std::uint8_t *gate_indices,
    const std::uint8_t *gate_scale_indices,
    const float *gate_codebook,
    const float *gate_scale_values,
    size_t gate_scale_count,
    size_t gate_group_size,
    float gate_tensor_scale,
    const float *gate_row_scales,
    size_t gate_row_scale_count,
    const std::uint8_t *up_indices,
    const std::uint8_t *up_scale_indices,
    const float *up_codebook,
    const float *up_scale_values,
    size_t up_scale_count,
    size_t up_group_size,
    float up_tensor_scale,
    const float *up_row_scales,
    size_t up_row_scale_count,
    const float *input,
    size_t rows,
    size_t cols,
    float *output) {
    std::vector<float> gate(rows);
    std::vector<float> up(rows);
    if (!aq4_matvec_f32_host(
            gate_indices,
            gate_scale_indices,
            gate_codebook,
            gate_scale_values,
            gate_scale_count,
            gate_group_size,
            gate_tensor_scale,
            input,
            gate_row_scales,
            gate_row_scale_count,
            rows,
            cols,
            gate.data())) {
        return false;
    }
    if (!aq4_matvec_f32_host(
            up_indices,
            up_scale_indices,
            up_codebook,
            up_scale_values,
            up_scale_count,
            up_group_size,
            up_tensor_scale,
            input,
            up_row_scales,
            up_row_scale_count,
            rows,
            cols,
            up.data())) {
        return false;
    }
    for (size_t row = 0; row < rows; ++row) {
        const float gate_value = gate[row];
        const float sigmoid = 1.0f / (1.0f + std::exp(-gate_value));
        output[row] = gate_value * sigmoid * up[row];
    }
    return true;
}

bool aq4_matvec_gate_beta_f32_host(
    const std::uint8_t *a_indices,
    const std::uint8_t *a_scale_indices,
    const float *a_codebook,
    const float *a_scale_values,
    size_t a_scale_count,
    size_t a_group_size,
    float a_tensor_scale,
    const float *a_row_scales,
    size_t a_row_scale_count,
    const std::uint8_t *b_indices,
    const std::uint8_t *b_scale_indices,
    const float *b_codebook,
    const float *b_scale_values,
    size_t b_scale_count,
    size_t b_group_size,
    float b_tensor_scale,
    const float *b_row_scales,
    size_t b_row_scale_count,
    const float *input,
    const float *a_log,
    const float *dt_bias,
    size_t heads,
    size_t cols,
    float *gate_output,
    float *beta_output) {
    std::vector<float> a(heads);
    std::vector<float> b(heads);
    if (!aq4_matvec_f32_host(
            a_indices,
            a_scale_indices,
            a_codebook,
            a_scale_values,
            a_scale_count,
            a_group_size,
            a_tensor_scale,
            input,
            a_row_scales,
            a_row_scale_count,
            heads,
            cols,
            a.data())) {
        return false;
    }
    if (!aq4_matvec_f32_host(
            b_indices,
            b_scale_indices,
            b_codebook,
            b_scale_values,
            b_scale_count,
            b_group_size,
            b_tensor_scale,
            input,
            b_row_scales,
            b_row_scale_count,
            heads,
            cols,
            b.data())) {
        return false;
    }
    for (size_t head = 0; head < heads; ++head) {
        const float x = a[head] + dt_bias[head];
        const float softplus = x <= 20.0f ? std::log1p(std::exp(x)) : x;
        gate_output[head] = -std::exp(a_log[head]) * softplus;
        beta_output[head] = 1.0f / (1.0f + std::exp(-b[head]));
    }
    return true;
}

class HipAq4MatvecKernelCache {
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
            if (!hiprtc_runtime().compile_aq4_matvec_kernel(arch, &code, &compile_error)) {
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
                    "ullm_aq4_matvec_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build AQ4 matvec HIP kernel" : compile_errors);
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

HipAq4MatvecKernelCache &hip_aq4_matvec_kernel_cache() {
    static HipAq4MatvecKernelCache cache;
    return cache;
}

// The wide-load decode experiment must never share the production module cache: it has a
// different HIPRTC source and is exposed only through its direct prototype ABI.
class HipAq4MatvecWideLoadPrototypeKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second->function;
        }
        if (hip_runtime().device_arch_name(device_id) != "gfx1201") {
            append_error(error, "AQ4 wide-load matvec prototype requires exact gfx1201 architecture");
            return nullptr;
        }
        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_aq4_matvec_wide_load_prototype_kernel(
                "gfx1201", &code, &compile_error)) {
            append_error(
                error,
                compile_error.empty() ?
                    "failed to build AQ4 wide-load matvec prototype HIP kernel" : compile_error);
            return nullptr;
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            append_error(error, "hipModuleLoadData failed for gfx1201 AQ4 wide-load matvec prototype");
            return nullptr;
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function,
                module,
                "ullm_aq4_matvec_wide_load_prototype_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            append_error(error, "hipModuleGetFunction failed for AQ4 wide-load matvec prototype");
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

HipAq4MatvecWideLoadPrototypeKernelCache &hip_aq4_matvec_wide_load_prototype_kernel_cache() {
    static HipAq4MatvecWideLoadPrototypeKernelCache cache;
    return cache;
}

// This direct-only reduction experiment has a distinct HIPRTC body and must not share either
// the production or wide-load comparison module caches.
class HipAq4MatvecShufflePrototypeKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second->function;
        }
        if (hip_runtime().device_arch_name(device_id) != "gfx1201") {
            append_error(error, "AQ4 matvec shuffle prototype requires exact gfx1201 architecture");
            return nullptr;
        }
        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_aq4_matvec_shuffle_prototype_kernel(
                "gfx1201", &code, &compile_error)) {
            append_error(
                error,
                compile_error.empty() ?
                    "failed to build AQ4 matvec shuffle prototype HIP kernel" : compile_error);
            return nullptr;
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            append_error(error, "hipModuleLoadData failed for gfx1201 AQ4 matvec shuffle prototype");
            return nullptr;
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function,
                module,
                "ullm_aq4_matvec_shuffle_prototype_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            append_error(error, "hipModuleGetFunction failed for AQ4 matvec shuffle prototype");
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

HipAq4MatvecShufflePrototypeKernelCache &hip_aq4_matvec_shuffle_prototype_kernel_cache() {
    static HipAq4MatvecShufflePrototypeKernelCache cache;
    return cache;
}

class HipAq4MatvecBatchKernelCache {
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
            if (!hiprtc_runtime().compile_aq4_matvec_batch_kernel(
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
                    "ullm_aq4_matvec_batch_f32_kernel",
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
            compile_errors.empty() ? "failed to build AQ4 matvec batch HIP kernel" :
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

HipAq4MatvecBatchKernelCache &hip_aq4_matvec_batch_kernel_cache() {
    static HipAq4MatvecBatchKernelCache cache;
    return cache;
}

class HipAq4MatvecTiledKernelCache {
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

        // This cache is deliberately exact-architecture: gfx1200 and other gfx12 variants
        // must stay on the legacy kernel until their own measured tile is admitted.
        if (hip_runtime().device_arch_name(device_id) != "gfx1201") {
            append_error(error, "AQ4 tiled GEMM requires exact gfx1201 architecture");
            return nullptr;
        }

        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_aq4_matvec_tiled_kernel(
                "gfx1201",
                &code,
                &compile_error)) {
            const std::string message = compile_error.empty() ?
                "failed to build AQ4 tiled GEMM HIP kernel" : compile_error;
            failures_.emplace(device_id, message);
            append_error(error, message);
            return nullptr;
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            const std::string message = "hipModuleLoadData failed for gfx1201 AQ4 tiled GEMM";
            failures_.emplace(device_id, message);
            append_error(error, message);
            return nullptr;
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function,
                module,
                "ullm_aq4_gemm_tiled_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            const std::string message = "hipModuleGetFunction failed for AQ4 tiled GEMM";
            failures_.emplace(device_id, message);
            append_error(error, message);
            return nullptr;
        }

        auto loaded = std::make_unique<LoadedModule>();
        loaded->module = module;
        loaded->function = function;
        loaded->arch = "gfx1201";
        void *result = loaded->function;
        modules_.emplace(device_id, std::move(loaded));
        return result;
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
    std::unordered_map<int, std::string> failures_;
};

HipAq4MatvecTiledKernelCache &hip_aq4_matvec_tiled_kernel_cache() {
    static HipAq4MatvecTiledKernelCache cache;
    return cache;
}

class HipAq4MatvecRegisterKernelCache {
public:
    void *function_for_device(
        int device_id,
        unsigned int batch_tile,
        size_t group_size,
        std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool supported_variant =
            (batch_tile == kAq4RegisterBm4 && group_size == kAq4RegisterGroup16) ||
            (batch_tile == kAq4RegisterBm8 &&
             (group_size == kAq4RegisterGroup8 || group_size == kAq4RegisterGroup16));
        if (!supported_variant) {
            append_error(error, "AQ4 register GEMM supports BM4/group16 or BM8/group8/group16");
            return nullptr;
        }
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return batch_tile == kAq4RegisterBm4 ? found->second->function_bm4 :
                   group_size == kAq4RegisterGroup8 ? found->second->function_bm8_group8 :
                                                     found->second->function_bm8;
        }
        auto failed = failures_.find(device_id);
        if (failed != failures_.end()) {
            append_error(error, failed->second);
            return nullptr;
        }
        if (hip_runtime().device_arch_name(device_id) != "gfx1201") {
            append_error(error, "AQ4 register GEMM requires exact gfx1201 architecture");
            return nullptr;
        }

        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_aq4_matvec_register_kernel(
                "gfx1201",
                &code,
                &compile_error)) {
            return fail(
                device_id,
                compile_error.empty() ? "failed to build AQ4 register GEMM HIP kernels" :
                                        compile_error,
                error);
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            return fail(
                device_id,
                "hipModuleLoadData failed for gfx1201 AQ4 register GEMM",
                error);
        }
        void *function_bm4 = nullptr;
        void *function_bm8 = nullptr;
        void *function_bm8_group8 = nullptr;
        if (!hip_runtime().module_get_function(
                &function_bm4,
                module,
                "ullm_aq4_gemm_register_bm4_f32_kernel",
                device_id) ||
            !hip_runtime().module_get_function(
                &function_bm8,
                module,
                "ullm_aq4_gemm_register_bm8_f32_kernel",
                device_id) ||
            !hip_runtime().module_get_function(
                &function_bm8_group8,
                module,
                "ullm_aq4_gemm_register_bm8_group8_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            return fail(
                device_id,
                "hipModuleGetFunction failed for AQ4 register GEMM",
                error);
        }

        auto loaded = std::make_unique<LoadedModule>();
        loaded->module = module;
        loaded->function_bm4 = function_bm4;
        loaded->function_bm8 = function_bm8;
        loaded->function_bm8_group8 = function_bm8_group8;
        void *result = batch_tile == kAq4RegisterBm4 ? function_bm4 :
                       group_size == kAq4RegisterGroup8 ? function_bm8_group8 : function_bm8;
        modules_.emplace(device_id, std::move(loaded));
        return result;
    }

private:
    struct LoadedModule {
        void *module = nullptr;
        void *function_bm4 = nullptr;
        void *function_bm8 = nullptr;
        void *function_bm8_group8 = nullptr;
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

HipAq4MatvecRegisterKernelCache &hip_aq4_matvec_register_kernel_cache() {
    static HipAq4MatvecRegisterKernelCache cache;
    return cache;
}

// Keep the experimental WMMA module separate from the production register module. Besides
// avoiding changes to the established BM4/BM8 cache, this supplies HIPRTC with rocWMMA include
// paths only for the source that needs them.
class HipAq4MatvecWmmaPrototypeKernelCache {
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
            append_error(error, "AQ4 WMMA prototype requires exact gfx1201 architecture");
            return nullptr;
        }

        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_aq4_matvec_wmma_prototype_kernel(
                "gfx1201",
                &code,
                &compile_error)) {
            return fail(
                device_id,
                compile_error.empty() ? "failed to build AQ4 WMMA prototype HIP kernel" :
                                        compile_error,
                error);
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            return fail(
                device_id,
                "hipModuleLoadData failed for gfx1201 AQ4 WMMA prototype",
                error);
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function,
                module,
                "ullm_aq4_gemm_wmma_prototype_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            return fail(
                device_id,
                "hipModuleGetFunction failed for AQ4 WMMA prototype",
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

HipAq4MatvecWmmaPrototypeKernelCache &hip_aq4_matvec_wmma_prototype_kernel_cache() {
    static HipAq4MatvecWmmaPrototypeKernelCache cache;
    return cache;
}

// The ragged-M experiment needs a distinct code object: unlike production v4, it retains the
// v2 output LDS tile so invalid physical rows can be prevented from reaching global output.
class HipAq4MatvecWmmaRaggedMPrototypeKernelCache {
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
            append_error(error, "AQ4 WMMA ragged-M prototype requires exact gfx1201 architecture");
            return nullptr;
        }

        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_aq4_matvec_wmma_ragged_m_prototype_kernel(
                "gfx1201",
                &code,
                &compile_error)) {
            return fail(
                device_id,
                compile_error.empty() ? "failed to build AQ4 WMMA ragged-M prototype HIP kernel" :
                                        compile_error,
                error);
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            return fail(
                device_id,
                "hipModuleLoadData failed for gfx1201 AQ4 WMMA ragged-M prototype",
                error);
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function,
                module,
                "ullm_aq4_gemm_wmma_ragged_m_prototype_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            return fail(
                device_id,
                "hipModuleGetFunction failed for AQ4 WMMA ragged-M prototype",
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

HipAq4MatvecWmmaRaggedMPrototypeKernelCache &
hip_aq4_matvec_wmma_ragged_m_prototype_kernel_cache() {
    static HipAq4MatvecWmmaRaggedMPrototypeKernelCache cache;
    return cache;
}

// V3 is intentionally a distinct HIPRTC module/cache. Its larger LDS stage changes code
// generation and resource allocation, so sharing the production v2 cache would invalidate the
// A/B comparison and could perturb the served code object.
class HipAq4MatvecWmmaPrototypeV3KernelCache {
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
            append_error(error, "AQ4 WMMA prototype v3 requires exact gfx1201 architecture");
            return nullptr;
        }

        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_aq4_matvec_wmma_prototype_v3_kernel(
                "gfx1201",
                &code,
                &compile_error)) {
            return fail(
                device_id,
                compile_error.empty() ? "failed to build AQ4 WMMA prototype v3 HIP kernel" :
                                        compile_error,
                error);
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            return fail(
                device_id,
                "hipModuleLoadData failed for gfx1201 AQ4 WMMA prototype v3",
                error);
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function,
                module,
                "ullm_aq4_gemm_wmma_prototype_v3_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            return fail(
                device_id,
                "hipModuleGetFunction failed for AQ4 WMMA prototype v3",
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

HipAq4MatvecWmmaPrototypeV3KernelCache &hip_aq4_matvec_wmma_prototype_v3_kernel_cache() {
    static HipAq4MatvecWmmaPrototypeV3KernelCache cache;
    return cache;
}

// The group8 production path owns a second rocWMMA module/cache. The group16 module and all
// scalar/register dispatch paths remain isolated from its compiler flags and source.
class HipAq4MatvecWmmaGroup8PrototypeKernelCache {
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
            append_error(error, "AQ4 group8 WMMA prototype requires exact gfx1201 architecture");
            return nullptr;
        }

        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_aq4_matvec_wmma_group8_prototype_kernel(
                "gfx1201",
                &code,
                &compile_error)) {
            return fail(
                device_id,
                compile_error.empty() ?
                    "failed to build AQ4 group8 WMMA prototype HIP kernel" :
                    compile_error,
                error);
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            return fail(
                device_id,
                "hipModuleLoadData failed for gfx1201 AQ4 group8 WMMA prototype",
                error);
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function,
                module,
                "ullm_aq4_gemm_wmma_group8_prototype_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            return fail(
                device_id,
                "hipModuleGetFunction failed for AQ4 group8 WMMA prototype",
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

HipAq4MatvecWmmaGroup8PrototypeKernelCache &
hip_aq4_matvec_wmma_group8_prototype_kernel_cache() {
    static HipAq4MatvecWmmaGroup8PrototypeKernelCache cache;
    return cache;
}

// Ragged group8 retains an output LDS tile for its guarded scalar final stores, so it requires a
// separate code object from the direct M=128 group8 module.
class HipAq4MatvecWmmaGroup8RaggedMPrototypeKernelCache {
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
            append_error(error, "AQ4 group8 ragged-M WMMA prototype requires exact gfx1201 architecture");
            return nullptr;
        }
        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_aq4_matvec_wmma_group8_ragged_m_prototype_kernel(
                "gfx1201", &code, &compile_error)) {
            return fail(
                device_id,
                compile_error.empty() ?
                    "failed to build AQ4 group8 ragged-M WMMA prototype HIP kernel" :
                    compile_error,
                error);
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            return fail(device_id, "hipModuleLoadData failed for gfx1201 AQ4 group8 ragged-M WMMA prototype", error);
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function, module, "ullm_aq4_gemm_wmma_group8_ragged_m_prototype_f32_kernel", device_id)) {
            hip_runtime().module_unload(module, device_id);
            return fail(device_id, "hipModuleGetFunction failed for AQ4 group8 ragged-M WMMA prototype", error);
        }
        auto loaded = std::make_unique<LoadedModule>();
        loaded->module = module;
        loaded->function = function;
        void *result = loaded->function;
        modules_.emplace(device_id, std::move(loaded));
        return result;
    }

private:
    struct LoadedModule { void *module = nullptr; void *function = nullptr; };
    void *fail(int device_id, const std::string &message, std::string *error) {
        failures_.emplace(device_id, message);
        append_error(error, message);
        return nullptr;
    }
    static void append_error(std::string *error, const std::string &message) {
        if (error == nullptr || message.empty()) return;
        if (!error->empty()) error->append("\n");
        error->append(message);
    }
    std::mutex mutex_;
    std::unordered_map<int, std::unique_ptr<LoadedModule>> modules_;
    std::unordered_map<int, std::string> failures_;
};

HipAq4MatvecWmmaGroup8RaggedMPrototypeKernelCache &
hip_aq4_matvec_wmma_group8_ragged_m_prototype_kernel_cache() {
    static HipAq4MatvecWmmaGroup8RaggedMPrototypeKernelCache cache;
    return cache;
}

class HipAq4MatvecTop1KernelCache {
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
            if (!hiprtc_runtime().compile_aq4_matvec_top1_kernel(arch, &code, &compile_error)) {
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
                    "ullm_aq4_matvec_top1_f32_kernel",
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
            compile_errors.empty() ? "failed to build AQ4 matvec top1 HIP kernel"
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

HipAq4MatvecTop1KernelCache &hip_aq4_matvec_top1_kernel_cache() {
    static HipAq4MatvecTop1KernelCache cache;
    return cache;
}

class HipAq4MatvecAddKernelCache {
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
            if (!hiprtc_runtime().compile_aq4_matvec_add_kernel(arch, &code, &compile_error)) {
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
                    "ullm_aq4_matvec_add_f32_kernel",
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
            compile_errors.empty() ? "failed to build AQ4 matvec add HIP kernel" :
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

HipAq4MatvecAddKernelCache &hip_aq4_matvec_add_kernel_cache() {
    static HipAq4MatvecAddKernelCache cache;
    return cache;
}

// Direct-only cache: this experimental module must never replace production matvec-add.
class HipAq4MatvecAddWideLoadPrototypeKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second->function;
        }
        if (hip_runtime().device_arch_name(device_id) != "gfx1201") {
            append_error(error, "AQ4 matvec add wide-load prototype requires exact gfx1201 architecture");
            return nullptr;
        }
        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_aq4_matvec_add_wide_load_prototype_kernel(
                "gfx1201", &code, &compile_error)) {
            append_error(
                error,
                compile_error.empty() ?
                    "failed to build AQ4 matvec add wide-load prototype HIP kernel" :
                    compile_error);
            return nullptr;
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            append_error(error, "hipModuleLoadData failed for AQ4 matvec add wide-load prototype");
            return nullptr;
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function,
                module,
                "ullm_aq4_matvec_add_wide_load_prototype_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            append_error(error, "hipModuleGetFunction failed for AQ4 matvec add wide-load prototype");
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

HipAq4MatvecAddWideLoadPrototypeKernelCache &hip_aq4_matvec_add_wide_load_prototype_kernel_cache() {
    static HipAq4MatvecAddWideLoadPrototypeKernelCache cache;
    return cache;
}

// This remains independent from production and the wide-load comparison module: it changes only
// matvec-add's reduction, which keeps differential and timing results attributable to shuffles.
class HipAq4MatvecAddShufflePrototypeKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second->function;
        }
        if (hip_runtime().device_arch_name(device_id) != "gfx1201") {
            append_error(error, "AQ4 matvec add shuffle prototype requires exact gfx1201 architecture");
            return nullptr;
        }
        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_aq4_matvec_add_shuffle_prototype_kernel(
                "gfx1201", &code, &compile_error)) {
            append_error(
                error,
                compile_error.empty() ?
                    "failed to build AQ4 matvec add shuffle prototype HIP kernel" :
                    compile_error);
            return nullptr;
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            append_error(error, "hipModuleLoadData failed for AQ4 matvec add shuffle prototype");
            return nullptr;
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function,
                module,
                "ullm_aq4_matvec_add_shuffle_prototype_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            append_error(error, "hipModuleGetFunction failed for AQ4 matvec add shuffle prototype");
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

HipAq4MatvecAddShufflePrototypeKernelCache &hip_aq4_matvec_add_shuffle_prototype_kernel_cache() {
    static HipAq4MatvecAddShufflePrototypeKernelCache cache;
    return cache;
}

class HipAq4MatvecPairKernelCache {
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
            if (!hiprtc_runtime().compile_aq4_matvec_pair_kernel(arch, &code, &compile_error)) {
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
                    "ullm_aq4_matvec_pair_f32_kernel",
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
            compile_errors.empty() ? "failed to build AQ4 matvec pair HIP kernel" :
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

HipAq4MatvecPairKernelCache &hip_aq4_matvec_pair_kernel_cache() {
    static HipAq4MatvecPairKernelCache cache;
    return cache;
}

class HipAq4MatvecTripleKernelCache {
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
            if (!hiprtc_runtime().compile_aq4_matvec_triple_kernel(
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
                    "ullm_aq4_matvec_triple_f32_kernel",
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
            compile_errors.empty() ? "failed to build AQ4 matvec triple HIP kernel" :
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

HipAq4MatvecTripleKernelCache &hip_aq4_matvec_triple_kernel_cache() {
    static HipAq4MatvecTripleKernelCache cache;
    return cache;
}

// This stays independent from the production triple cache: it changes only the RPB=8
// reduction and remains directly callable until a device differential/timing window validates it.
class HipAq4MatvecTripleShufflePrototypeKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) return found->second->function;
        if (hip_runtime().device_arch_name(device_id) != "gfx1201") {
            append_error(error, "AQ4 matvec triple shuffle prototype requires exact gfx1201 architecture");
            return nullptr;
        }
        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_aq4_matvec_triple_shuffle_prototype_kernel(
                "gfx1201", &code, &compile_error)) {
            append_error(error, compile_error.empty() ?
                "failed to build AQ4 matvec triple shuffle prototype HIP kernel" : compile_error);
            return nullptr;
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            append_error(error, "hipModuleLoadData failed for AQ4 matvec triple shuffle prototype");
            return nullptr;
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function,
                module,
                "ullm_aq4_matvec_triple_shuffle_prototype_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            append_error(error, "hipModuleGetFunction failed for AQ4 matvec triple shuffle prototype");
            return nullptr;
        }
        auto loaded = std::make_unique<LoadedModule>();
        loaded->module = module;
        loaded->function = function;
        modules_.emplace(device_id, std::move(loaded));
        return function;
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

HipAq4MatvecTripleShufflePrototypeKernelCache &
hip_aq4_matvec_triple_shuffle_prototype_kernel_cache() {
    static HipAq4MatvecTripleShufflePrototypeKernelCache cache;
    return cache;
}

class HipAq4MatvecQkvZGateBetaKernelCache {
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
            if (!hiprtc_runtime().compile_aq4_matvec_qkv_z_gate_beta_kernel(
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
                    "ullm_aq4_matvec_qkv_z_gate_beta_f32_kernel",
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
            compile_errors.empty() ? "failed to build AQ4 qkv/z gate/beta HIP kernel" :
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

// These two direct-only modules are deliberately kept out of the production fused caches.
class HipAq4FusedWideLoadPrototypeKernelCache {
public:
    enum class Kind { QkvZGateBeta, SiluMul };

    explicit HipAq4FusedWideLoadPrototypeKernelCache(Kind kind) : kind_(kind) {}

    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) return found->second->function;
        if (hip_runtime().device_arch_name(device_id) != "gfx1201") {
            append_error(error, "AQ4 fused wide-load prototype requires exact gfx1201 architecture");
            return nullptr;
        }
        std::vector<char> code;
        std::string compile_error;
        const bool compiled = kind_ == Kind::QkvZGateBeta
            ? hiprtc_runtime().compile_aq4_matvec_qkv_z_gate_beta_wide_load_prototype_kernel(
                  "gfx1201", &code, &compile_error)
            : hiprtc_runtime().compile_aq4_matvec_silu_mul_wide_load_prototype_kernel(
                  "gfx1201", &code, &compile_error);
        if (!compiled) {
            append_error(error, compile_error.empty() ?
                "failed to build AQ4 fused wide-load prototype HIP kernel" : compile_error);
            return nullptr;
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            append_error(error, "hipModuleLoadData failed for AQ4 fused wide-load prototype");
            return nullptr;
        }
        const char *name = kind_ == Kind::QkvZGateBeta
            ? "ullm_aq4_matvec_qkv_z_gate_beta_wide_load_prototype_f32_kernel"
            : "ullm_aq4_matvec_silu_mul_wide_load_prototype_f32_kernel";
        void *function = nullptr;
        if (!hip_runtime().module_get_function(&function, module, name, device_id)) {
            hip_runtime().module_unload(module, device_id);
            append_error(error, "hipModuleGetFunction failed for AQ4 fused wide-load prototype");
            return nullptr;
        }
        auto loaded = std::make_unique<LoadedModule>();
        loaded->module = module;
        loaded->function = function;
        modules_.emplace(device_id, std::move(loaded));
        return function;
    }

private:
    struct LoadedModule { void *module = nullptr; void *function = nullptr; };
    static void append_error(std::string *error, const std::string &message) {
        if (error == nullptr || message.empty()) return;
        if (!error->empty()) error->append("\n");
        error->append(message);
    }
    Kind kind_;
    std::mutex mutex_;
    std::unordered_map<int, std::unique_ptr<LoadedModule>> modules_;
};

HipAq4FusedWideLoadPrototypeKernelCache &hip_aq4_matvec_qkv_z_gate_beta_wide_load_prototype_kernel_cache() {
    static HipAq4FusedWideLoadPrototypeKernelCache cache(
        HipAq4FusedWideLoadPrototypeKernelCache::Kind::QkvZGateBeta);
    return cache;
}

HipAq4FusedWideLoadPrototypeKernelCache &hip_aq4_matvec_silu_mul_wide_load_prototype_kernel_cache() {
    static HipAq4FusedWideLoadPrototypeKernelCache cache(
        HipAq4FusedWideLoadPrototypeKernelCache::Kind::SiluMul);
    return cache;
}

// This source stays independent from production and the failed wide-load experiment: it changes
// only the reduction, so its code object remains a direct scalar-load comparison point.
class HipAq4MatvecQkvZGateBetaShufflePrototypeKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) return found->second->function;
        if (hip_runtime().device_arch_name(device_id) != "gfx1201") {
            append_error(error, "AQ4 qkv/z gate/beta shuffle prototype requires exact gfx1201 architecture");
            return nullptr;
        }
        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_aq4_matvec_qkv_z_gate_beta_shuffle_prototype_kernel(
                "gfx1201", &code, &compile_error)) {
            append_error(error, compile_error.empty() ?
                "failed to build AQ4 qkv/z gate/beta shuffle prototype HIP kernel" : compile_error);
            return nullptr;
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            append_error(error, "hipModuleLoadData failed for AQ4 qkv/z gate/beta shuffle prototype");
            return nullptr;
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function,
                module,
                "ullm_aq4_matvec_qkv_z_gate_beta_shuffle_prototype_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            append_error(error, "hipModuleGetFunction failed for AQ4 qkv/z gate/beta shuffle prototype");
            return nullptr;
        }
        auto loaded = std::make_unique<LoadedModule>();
        loaded->module = module;
        loaded->function = function;
        modules_.emplace(device_id, std::move(loaded));
        return function;
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

HipAq4MatvecQkvZGateBetaShufflePrototypeKernelCache &
hip_aq4_matvec_qkv_z_gate_beta_shuffle_prototype_kernel_cache() {
    static HipAq4MatvecQkvZGateBetaShufflePrototypeKernelCache cache;
    return cache;
}

// This stays independent from both production and the wide-load experiment: its source changes
// only the reduction and must not perturb either comparison code object.
class HipAq4MatvecSiluMulShufflePrototypeKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) return found->second->function;
        if (hip_runtime().device_arch_name(device_id) != "gfx1201") {
            append_error(error, "AQ4 SiLU-mul shuffle prototype requires exact gfx1201 architecture");
            return nullptr;
        }
        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_aq4_matvec_silu_mul_shuffle_prototype_kernel(
                "gfx1201", &code, &compile_error)) {
            append_error(error, compile_error.empty() ?
                "failed to build AQ4 SiLU-mul shuffle prototype HIP kernel" : compile_error);
            return nullptr;
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            append_error(error, "hipModuleLoadData failed for AQ4 SiLU-mul shuffle prototype");
            return nullptr;
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function,
                module,
                "ullm_aq4_matvec_silu_mul_shuffle_prototype_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            append_error(error, "hipModuleGetFunction failed for AQ4 SiLU-mul shuffle prototype");
            return nullptr;
        }
        auto loaded = std::make_unique<LoadedModule>();
        loaded->module = module;
        loaded->function = function;
        modules_.emplace(device_id, std::move(loaded));
        return function;
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

HipAq4MatvecSiluMulShufflePrototypeKernelCache &
hip_aq4_matvec_silu_mul_shuffle_prototype_kernel_cache() {
    static HipAq4MatvecSiluMulShufflePrototypeKernelCache cache;
    return cache;
}

HipAq4MatvecQkvZGateBetaKernelCache &hip_aq4_matvec_qkv_z_gate_beta_kernel_cache() {
    static HipAq4MatvecQkvZGateBetaKernelCache cache;
    return cache;
}

enum class Aq4MatvecBatchDispatchKind : uint32_t {
    Legacy = 0u,
    TiledLds = 1u,
    RegisterBm4 = 2u,
    RegisterBm8 = 3u,
    RegisterBm8Group8 = 4u,
};

bool aq4_tiled_experimental_enabled() {
    const char *raw = std::getenv("ULLM_EXPERIMENTAL_HIP_AQ4_TILED_GEMM");
    return raw != nullptr && std::strcmp(raw, "1") == 0;
}

enum class Aq4RegisterExperimentalSetting {
    Absent,
    Invalid,
    Bm4,
    Bm8,
};

Aq4RegisterExperimentalSetting aq4_register_experimental_setting() {
    const char *raw = std::getenv("ULLM_EXPERIMENTAL_HIP_AQ4_REGISTER_BM");
    if (raw == nullptr) {
        return Aq4RegisterExperimentalSetting::Absent;
    }
    if (std::strcmp(raw, "4") == 0) {
        return Aq4RegisterExperimentalSetting::Bm4;
    }
    if (std::strcmp(raw, "8") == 0) {
        return Aq4RegisterExperimentalSetting::Bm8;
    }
    return Aq4RegisterExperimentalSetting::Invalid;
}

bool aq4_matvec_batch_tiled_shape_is_supported(
    const std::string &arch,
    size_t group_size,
    size_t rows,
    size_t cols,
    size_t batch_count) {
    // Keep dispatch fail-closed. The initial tile handles row/K multiples exactly while
    // guarding batch tails in-kernel; ragged rows or K remain on the established kernel.
    if (arch != "gfx1201" || group_size != 16u || batch_count < kAq4TiledBm || rows == 0u ||
        cols == 0u || rows % kAq4TiledBn != 0u || cols % kAq4TiledBk != 0u) {
        return false;
    }
    if (rows > std::numeric_limits<size_t>::max() - (kAq4TiledBn - 1u)) {
        return false;
    }
    if (cols > std::numeric_limits<size_t>::max() - (kAq4TiledBk - 1u)) {
        return false;
    }
    if (batch_count > std::numeric_limits<size_t>::max() - (kAq4TiledBm - 1u)) {
        return false;
    }
    return true;
}

bool aq4_matvec_batch_register_shape_is_supported(
    const std::string &arch,
    size_t group_size,
    size_t rows,
    size_t cols,
    size_t batch_count,
    unsigned int batch_tile) {
    const bool supported_variant =
        (batch_tile == kAq4RegisterBm4 && group_size == kAq4RegisterGroup16) ||
        (batch_tile == kAq4RegisterBm8 &&
         (group_size == kAq4RegisterGroup8 || group_size == kAq4RegisterGroup16));
    if (!supported_variant || arch != "gfx1201" || batch_count < batch_tile || rows == 0u ||
        cols == 0u || rows % kAq4RegisterRowsPerBlock != 0u || cols % kAq4TiledBk != 0u) {
        return false;
    }
    if (rows > std::numeric_limits<size_t>::max() - (kAq4RegisterRowsPerBlock - 1u) ||
        cols > std::numeric_limits<size_t>::max() - (kAq4TiledBk - 1u) ||
        batch_count > std::numeric_limits<size_t>::max() - (batch_tile - 1u)) {
        return false;
    }
    return true;
}

bool aq4_matvec_batch_wmma_prototype_shape_is_supported(
    const std::string &arch,
    size_t group_size,
    size_t rows,
    size_t cols,
    size_t batch_count) {
    // The paired group16 Wide-K source load requires K divisible by 32. That also makes every
    // packed-index row stride and every FP32 input row stride 16-byte aligned. Complete 16-row
    // tiles let the launch use an exact row grid, including rows=32 -> grid_x=2.
    return arch == "gfx1201" && group_size == kAq4RegisterGroup16 &&
        batch_count == kAq4WmmaPrototypeBm && rows != 0u && cols != 0u &&
        rows % kAq4WmmaPrototypeBn == 0u && cols % kAq4WmmaPrototypeBk == 0u;
}

bool aq4_matvec_batch_wmma_ragged_m_prototype_shape_is_supported(
    const std::string &arch,
    size_t group_size,
    size_t rows,
    size_t cols,
    size_t m_actual) {
    // The physical CTA always computes 128 rows; m_actual only limits global activation loads
    // and output stores.  Wide-K alignment is unchanged from the production group16 kernel.
    return arch == "gfx1201" && group_size == kAq4RegisterGroup16 && m_actual != 0u &&
        m_actual <= kAq4WmmaPrototypeBm && rows != 0u && cols != 0u &&
        rows % kAq4WmmaPrototypeBn == 0u && cols % kAq4WmmaPrototypeBk == 0u;
}

bool aq4_matvec_batch_wmma_prototype_v3_shape_is_supported(
    const std::string &arch,
    size_t group_size,
    size_t rows,
    size_t cols,
    size_t batch_count) {
    // V3 groups two existing 32-K Wide-K microtiles into one LDS hand-off, so K must be
    // divisible by 64. This direct-only experiment intentionally leaves v2's broader K%32
    // production eligibility and dispatch unchanged.
    return arch == "gfx1201" && group_size == kAq4RegisterGroup16 &&
        batch_count == kAq4WmmaPrototypeBm && rows != 0u && cols != 0u &&
        rows % kAq4WmmaPrototypeBn == 0u && cols % kAq4WmmaPrototypeV3Bk == 0u;
}

bool aq4_matvec_batch_wmma_group8_prototype_shape_is_supported(
    const std::string &arch,
    size_t group_size,
    size_t rows,
    size_t cols,
    size_t batch_count) {
    // K divisible by 32 makes every packed-index row stride and FP32 input row stride 16-byte
    // aligned. Complete 16-row tiles let the launch use an exact row grid. These are the full
    // production group8 WMMA constraints; the served 4096x4096 and 1024x4096 projections are
    // both members of this set.
    return arch == "gfx1201" && group_size == kAq4RegisterGroup8 &&
        batch_count == kAq4WmmaPrototypeBm && rows != 0u && cols != 0u &&
        rows % kAq4WmmaPrototypeBn == 0u && cols % kAq4WmmaPrototypeBk == 0u;
}

bool aq4_matvec_batch_wmma_group8_ragged_m_prototype_shape_is_supported(
    const std::string &arch,
    size_t group_size,
    size_t rows,
    size_t cols,
    size_t m_actual) {
    return arch == "gfx1201" && group_size == kAq4RegisterGroup8 && m_actual != 0u &&
        m_actual <= kAq4WmmaPrototypeBm && rows != 0u && cols != 0u &&
        rows % kAq4WmmaPrototypeBn == 0u && cols % kAq4WmmaPrototypeBk == 0u;
}

Aq4MatvecBatchDispatchKind aq4_matvec_batch_dispatch_kind_for_device(
    int device_id,
    size_t group_size,
    size_t rows,
    size_t cols,
    size_t batch_count) {
    if (device_id < 0) {
        return Aq4MatvecBatchDispatchKind::Legacy;
    }
    const std::string arch = hip_runtime().device_arch_name(device_id);
    const Aq4RegisterExperimentalSetting register_setting =
        aq4_register_experimental_setting();
    if (register_setting == Aq4RegisterExperimentalSetting::Bm4) {
        return aq4_matvec_batch_register_shape_is_supported(
                   arch,
                   group_size,
                   rows,
                   cols,
                   batch_count,
                   kAq4RegisterBm4) ?
            Aq4MatvecBatchDispatchKind::RegisterBm4 :
            Aq4MatvecBatchDispatchKind::Legacy;
    }
    if (register_setting == Aq4RegisterExperimentalSetting::Bm8) {
        if (!aq4_matvec_batch_register_shape_is_supported(
                arch,
                group_size,
                rows,
                cols,
                batch_count,
                kAq4RegisterBm8)) {
            return Aq4MatvecBatchDispatchKind::Legacy;
        }
        return group_size == kAq4RegisterGroup8 ?
            Aq4MatvecBatchDispatchKind::RegisterBm8Group8 :
            Aq4MatvecBatchDispatchKind::RegisterBm8;
    }
    // A present but invalid register setting is fail-closed and suppresses the LDS experiment.
    // With a valid setting, the register experiment always has precedence over the LDS flag.
    if (register_setting == Aq4RegisterExperimentalSetting::Invalid) {
        return Aq4MatvecBatchDispatchKind::Legacy;
    }
    return aq4_tiled_experimental_enabled() &&
            aq4_matvec_batch_tiled_shape_is_supported(
                arch,
                group_size,
                rows,
                cols,
                batch_count) ?
        Aq4MatvecBatchDispatchKind::TiledLds : Aq4MatvecBatchDispatchKind::Legacy;
}

bool aq4_matvec_batch_f32_tiled_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t batch_count,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    void *function = hip_aq4_matvec_tiled_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }
    const size_t grid_x = (rows + kAq4TiledBn - 1u) / kAq4TiledBn;
    const size_t grid_y = (batch_count + kAq4TiledBm - 1u) / kAq4TiledBm;
    if (grid_x == 0u || grid_y == 0u ||
        grid_x > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        grid_y > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 tiled GEMM dimensions exceed HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_batch_count = static_cast<unsigned long long>(batch_count);
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &scale_values_ptr,
        &input_ptr,
        &row_scale_ptr,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_row_scale_count,
        &kernel_rows,
        &kernel_cols,
        &kernel_batch_count,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel_2d(
            function,
            static_cast<unsigned int>(grid_x),
            static_cast<unsigned int>(grid_y),
            kAq4TiledThreads,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 tiled GEMM";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_batch_f32_register_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t batch_count,
    unsigned int batch_tile,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const bool supported_variant =
        (batch_tile == kAq4RegisterBm4 && group_size == kAq4RegisterGroup16) ||
        (batch_tile == kAq4RegisterBm8 &&
         (group_size == kAq4RegisterGroup8 || group_size == kAq4RegisterGroup16));
    if (!supported_variant) {
        if (error != nullptr) {
            *error = "AQ4 register GEMM supports BM4/group16 or BM8/group8/group16";
        }
        return false;
    }
    const int device_id = index_buffer->hip_device_id;
    void *function = hip_aq4_matvec_register_kernel_cache().function_for_device(
        device_id,
        batch_tile,
        group_size,
        error);
    if (function == nullptr) {
        return false;
    }
    const size_t grid_x = (rows + kAq4RegisterRowsPerBlock - 1u) /
        kAq4RegisterRowsPerBlock;
    const size_t grid_y = (batch_count + batch_tile - 1u) / batch_tile;
    if (grid_x == 0u || grid_y == 0u ||
        grid_x > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        grid_y > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 register GEMM dimensions exceed HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_batch_count = static_cast<unsigned long long>(batch_count);
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &scale_values_ptr,
        &input_ptr,
        &row_scale_ptr,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_row_scale_count,
        &kernel_rows,
        &kernel_cols,
        &kernel_batch_count,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel_2d(
            function,
            static_cast<unsigned int>(grid_x),
            static_cast<unsigned int>(grid_y),
            kAq4RegisterThreads,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 register GEMM";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_batch_f32_wmma_prototype_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t batch_count,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    if (!aq4_matvec_batch_wmma_prototype_shape_is_supported(
            hip_runtime().device_arch_name(device_id),
            group_size,
            rows,
            cols,
            batch_count)) {
        if (error != nullptr) {
            *error = "AQ4 WMMA prototype requires gfx1201, group16, M=128, rows divisible by 16, and cols divisible by 32";
        }
        return false;
    }
    // The HIPRTC kernel uses explicit 128-bit vector loads for packed AQ4 and FP32 input tiles.
    // HIP allocations satisfy this in normal use, but reject an arbitrary misaligned sub-buffer
    // rather than issue a potentially undefined vector load.
    if ((reinterpret_cast<std::uintptr_t>(index_buffer->ptr) & 15u) != 0u ||
        (reinterpret_cast<std::uintptr_t>(input_buffer->ptr) & 15u) != 0u) {
        if (error != nullptr) {
            *error = "AQ4 WMMA prototype requires 16-byte-aligned index and input buffers";
        }
        return false;
    }
    void *function = hip_aq4_matvec_wmma_prototype_kernel_cache().function_for_device(
        device_id,
        error);
    if (function == nullptr) {
        return false;
    }
    const size_t grid_x = rows / kAq4WmmaPrototypeBn;
    const size_t grid_y = batch_count / kAq4WmmaPrototypeBm;
    if (grid_x == 0u || grid_y == 0u ||
        grid_x > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        grid_y > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 WMMA prototype dimensions exceed HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_batch_count = static_cast<unsigned long long>(batch_count);
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &scale_values_ptr,
        &input_ptr,
        &row_scale_ptr,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_row_scale_count,
        &kernel_rows,
        &kernel_cols,
        &kernel_batch_count,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel_2d(
            function,
            static_cast<unsigned int>(grid_x),
            static_cast<unsigned int>(grid_y),
            kAq4WmmaPrototypeThreads,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 WMMA prototype";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_batch_f32_wmma_ragged_m_prototype_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t m_actual,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    if (!aq4_matvec_batch_wmma_ragged_m_prototype_shape_is_supported(
            hip_runtime().device_arch_name(device_id), group_size, rows, cols, m_actual)) {
        if (error != nullptr) {
            *error =
                "AQ4 WMMA ragged-M prototype requires gfx1201, group16, 1<=M<=128, rows divisible by 16, and cols divisible by 32";
        }
        return false;
    }
    if ((reinterpret_cast<std::uintptr_t>(index_buffer->ptr) & 15u) != 0u ||
        (reinterpret_cast<std::uintptr_t>(input_buffer->ptr) & 15u) != 0u) {
        if (error != nullptr) {
            *error = "AQ4 WMMA ragged-M prototype requires 16-byte-aligned index and input buffers";
        }
        return false;
    }
    void *function = hip_aq4_matvec_wmma_ragged_m_prototype_kernel_cache().function_for_device(
        device_id,
        error);
    if (function == nullptr) {
        return false;
    }
    const size_t grid_x = rows / kAq4WmmaPrototypeBn;
    // One CTA always supplies the 128 physical M rows; m_actual is the logical row count.
    const size_t grid_y = 1u;
    if (grid_x == 0u || grid_x > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 WMMA ragged-M prototype dimensions exceed HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_m_actual = static_cast<unsigned long long>(m_actual);
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &scale_values_ptr,
        &input_ptr,
        &row_scale_ptr,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_row_scale_count,
        &kernel_rows,
        &kernel_cols,
        &kernel_m_actual,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel_2d(
            function,
            static_cast<unsigned int>(grid_x),
            static_cast<unsigned int>(grid_y),
            kAq4WmmaPrototypeThreads,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 WMMA ragged-M prototype";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_batch_f32_wmma_prototype_v3_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t batch_count,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    if (!aq4_matvec_batch_wmma_prototype_v3_shape_is_supported(
            hip_runtime().device_arch_name(device_id),
            group_size,
            rows,
            cols,
            batch_count)) {
        if (error != nullptr) {
            *error =
                "AQ4 WMMA prototype v3 requires gfx1201, group16, M=128, rows divisible by 16, and cols divisible by 64";
        }
        return false;
    }
    // The source keeps v2's 128-bit vector loads. Retain its fail-closed alignment check rather
    // than allowing an arbitrary sub-buffer to reach the experimental module.
    if ((reinterpret_cast<std::uintptr_t>(index_buffer->ptr) & 15u) != 0u ||
        (reinterpret_cast<std::uintptr_t>(input_buffer->ptr) & 15u) != 0u) {
        if (error != nullptr) {
            *error =
                "AQ4 WMMA prototype v3 requires 16-byte-aligned index and input buffers";
        }
        return false;
    }
    void *function = hip_aq4_matvec_wmma_prototype_v3_kernel_cache().function_for_device(
        device_id,
        error);
    if (function == nullptr) {
        return false;
    }
    const size_t grid_x = rows / kAq4WmmaPrototypeBn;
    const size_t grid_y = batch_count / kAq4WmmaPrototypeBm;
    if (grid_x == 0u || grid_y == 0u ||
        grid_x > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        grid_y > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 WMMA prototype v3 dimensions exceed HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_batch_count = static_cast<unsigned long long>(batch_count);
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &scale_values_ptr,
        &input_ptr,
        &row_scale_ptr,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_row_scale_count,
        &kernel_rows,
        &kernel_cols,
        &kernel_batch_count,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel_2d(
            function,
            static_cast<unsigned int>(grid_x),
            static_cast<unsigned int>(grid_y),
            kAq4WmmaPrototypeThreads,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 WMMA prototype v3";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_batch_f32_wmma_group8_ragged_m_prototype_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t m_actual,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    if (!aq4_matvec_batch_wmma_group8_ragged_m_prototype_shape_is_supported(
            hip_runtime().device_arch_name(device_id), group_size, rows, cols, m_actual)) {
        if (error != nullptr) {
            *error = "AQ4 group8 ragged-M WMMA prototype requires gfx1201, group8, 1<=M<=128, rows divisible by 16, and cols divisible by 32";
        }
        return false;
    }
    if ((reinterpret_cast<std::uintptr_t>(index_buffer->ptr) & 15u) != 0u ||
        (reinterpret_cast<std::uintptr_t>(input_buffer->ptr) & 15u) != 0u) {
        if (error != nullptr) {
            *error = "AQ4 group8 ragged-M WMMA prototype requires 16-byte-aligned index and input buffers";
        }
        return false;
    }
    void *function = hip_aq4_matvec_wmma_group8_ragged_m_prototype_kernel_cache()
                         .function_for_device(device_id, error);
    if (function == nullptr) return false;
    const size_t grid_x = rows / kAq4WmmaPrototypeBn;
    if (grid_x == 0u || grid_x > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) *error = "AQ4 group8 ragged-M WMMA prototype dimensions exceed HIP grid limit";
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_m_actual = static_cast<unsigned long long>(m_actual);
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &index_ptr, &scale_ptr, &codebook_ptr, &scale_values_ptr, &input_ptr, &row_scale_ptr,
        &kernel_scale_count, &kernel_group_size, &tensor_scale, &kernel_row_scale_count,
        &kernel_rows, &kernel_cols, &kernel_m_actual, &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel_2d(
            function, static_cast<unsigned int>(grid_x), 1u, kAq4WmmaPrototypeThreads,
            kernel_params, hip_stream, device_id)) {
        if (error != nullptr) *error = "hipModuleLaunchKernel failed for AQ4 group8 ragged-M WMMA prototype";
        return false;
    }
    return true;
}

bool aq4_matvec_batch_f32_wmma_group8_prototype_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t batch_count,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    if (!aq4_matvec_batch_wmma_group8_prototype_shape_is_supported(
            hip_runtime().device_arch_name(device_id),
            group_size,
            rows,
            cols,
            batch_count)) {
        if (error != nullptr) {
            *error =
                "AQ4 group8 WMMA prototype requires gfx1201, group8, M=128, nonzero rows divisible by 16, and cols divisible by 32";
        }
        return false;
    }
    // The HIPRTC kernel uses explicit 128-bit vector loads for packed AQ4 and FP32 input tiles.
    // Reject an arbitrary misaligned sub-buffer rather than issue a potentially undefined load.
    if ((reinterpret_cast<std::uintptr_t>(index_buffer->ptr) & 15u) != 0u ||
        (reinterpret_cast<std::uintptr_t>(input_buffer->ptr) & 15u) != 0u) {
        if (error != nullptr) {
            *error =
                "AQ4 group8 WMMA prototype requires 16-byte-aligned index and input buffers";
        }
        return false;
    }
    void *function =
        hip_aq4_matvec_wmma_group8_prototype_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }
    const size_t grid_x = rows / kAq4WmmaPrototypeBn;
    const size_t grid_y = batch_count / kAq4WmmaPrototypeBm;
    if (grid_x == 0u || grid_y == 0u ||
        grid_x > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        grid_y > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 group8 WMMA prototype dimensions exceed HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_batch_count = static_cast<unsigned long long>(batch_count);
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &scale_values_ptr,
        &input_ptr,
        &row_scale_ptr,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_row_scale_count,
        &kernel_rows,
        &kernel_cols,
        &kernel_batch_count,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel_2d(
            function,
            static_cast<unsigned int>(grid_x),
            static_cast<unsigned int>(grid_y),
            kAq4WmmaPrototypeThreads,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 group8 WMMA prototype";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_f32_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    // The promoted body is validated for the Qwen3.5 decode contract only. Preserve the
    // established public ABI for unrelated AQ4 geometries by letting the caller use its existing
    // host-staging fallback rather than launching a wide kernel with incomplete chunks.
    if (hip_runtime().device_arch_name(device_id) != "gfx1201" ||
        (group_size != 8u && group_size != 16u) || cols % 32u != 0u ||
        cols % group_size != 0u ||
        (reinterpret_cast<std::uintptr_t>(index_buffer->ptr) & 15u) != 0u) {
        if (error != nullptr) {
            *error = "AQ4 production wide-load matvec requires gfx1201, group8/group16, "
                "cols divisible by 32, and a 16-byte-aligned index buffer";
        }
        return false;
    }
    void *function = hip_aq4_matvec_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const Aq4MatvecLaunchConfig launch_config = aq4_matvec_launch_config_for_device(device_id);
    const size_t grid_size =
        (rows + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec row count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &scale_values_ptr,
        &input_ptr,
        &row_scale_ptr,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_row_scale_count,
        &kernel_rows,
        &kernel_cols,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 matvec";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_wide_load_prototype_f32_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    void *function = hip_aq4_matvec_wide_load_prototype_kernel_cache().function_for_device(
        device_id, error);
    if (function == nullptr) {
        return false;
    }
    const Aq4MatvecLaunchConfig launch_config = aq4_matvec_launch_config_for_device(device_id);
    const size_t grid_size =
        (rows + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 wide-load matvec prototype row count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &index_ptr, &scale_ptr, &codebook_ptr, &scale_values_ptr, &input_ptr, &row_scale_ptr,
        &kernel_scale_count, &kernel_group_size, &tensor_scale, &kernel_row_scale_count,
        &kernel_rows, &kernel_cols, &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 wide-load matvec prototype";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_shuffle_prototype_f32_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    void *function = hip_aq4_matvec_shuffle_prototype_kernel_cache().function_for_device(
        device_id, error);
    if (function == nullptr) {
        return false;
    }
    // The candidate source fixes RPB=32. Eight contiguous threads own each row, so every
    // wave32 contains exactly four width-8 sub-wave reductions.
    constexpr unsigned int block_size = 256u;
    constexpr unsigned int rows_per_block = 32u;
    const size_t grid_size = (rows + rows_per_block - 1u) / rows_per_block;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec shuffle prototype row count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &index_ptr, &scale_ptr, &codebook_ptr, &scale_values_ptr, &input_ptr, &row_scale_ptr,
        &kernel_scale_count, &kernel_group_size, &tensor_scale, &kernel_row_scale_count,
        &kernel_rows, &kernel_cols, &output_ptr,
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
            *error = "hipModuleLaunchKernel failed for AQ4 matvec shuffle prototype";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_batch_f32_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t batch_count,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    void *function = hip_aq4_matvec_batch_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const Aq4MatvecLaunchConfig launch_config = aq4_matvec_launch_config_for_device(device_id);
    const size_t grid_x =
        (rows + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_x > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        batch_count > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec batch dimensions exceed HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_batch_count = static_cast<unsigned long long>(batch_count);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &scale_values_ptr,
        &input_ptr,
        &row_scale_ptr,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_row_scale_count,
        &kernel_rows,
        &kernel_cols,
        &kernel_batch_count,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel_2d(
            function,
            static_cast<unsigned int>(grid_x),
            static_cast<unsigned int>(batch_count),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 matvec batch";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_top1_f32_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    void *function = hip_aq4_matvec_top1_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const unsigned int rows_per_block = aq4_matvec_top1_rows_per_block_from_env();
    const size_t grid_size = (rows + rows_per_block - 1) / rows_per_block;
    if (grid_size != partial_count) {
        if (error != nullptr) {
            *error = "AQ4 matvec top1 partial count does not match row block count";
        }
        return false;
    }
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec top1 row count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *partial_values_ptr = partial_values_buffer->ptr;
    void *partial_indices_ptr = partial_indices_buffer->ptr;
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &scale_values_ptr,
        &input_ptr,
        &row_scale_ptr,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_row_scale_count,
        &kernel_rows,
        &kernel_cols,
        &partial_values_ptr,
        &partial_indices_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            256u,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 matvec top1";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_add_f32_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *residual_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    // The promoted body consumes one aligned uint4 (32 AQ4 elements) at a time. Preserve the
    // established generic-shape API contract through the caller's staging fallback; production
    // Qwen3.5 residual projections are all group8/group16 with complete 32-element chunks.
    if ((group_size != 8u && group_size != 16u) || cols % 32u != 0u) {
        if (error != nullptr) {
            *error =
                "AQ4 matvec add wide-load HIP kernel requires group8/group16 and cols divisible by 32";
        }
        return false;
    }
    const int device_id = index_buffer->hip_device_id;
    void *function = hip_aq4_matvec_add_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const Aq4MatvecLaunchConfig launch_config =
        aq4_matvec_launch_config_for_fused_kernel(
            device_id,
            8u,
            2u,
            "ULLM_AQ4_MATVEC_ADD_RPB");
    const size_t grid_size =
        (rows + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec add row count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *residual_ptr = residual_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &scale_values_ptr,
        &input_ptr,
        &residual_ptr,
        &row_scale_ptr,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_row_scale_count,
        &kernel_rows,
        &kernel_cols,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 matvec add";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_add_wide_load_prototype_f32_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *residual_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    void *function = hip_aq4_matvec_add_wide_load_prototype_kernel_cache().function_for_device(
        device_id, error);
    if (function == nullptr) {
        return false;
    }
    const Aq4MatvecLaunchConfig launch_config =
        aq4_matvec_launch_config_for_fused_kernel(
            device_id,
            8u,
            2u,
            "ULLM_AQ4_MATVEC_ADD_RPB");
    const size_t grid_size =
        (rows + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec add wide-load prototype row count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *residual_ptr = residual_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &scale_values_ptr,
        &input_ptr,
        &residual_ptr,
        &row_scale_ptr,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_row_scale_count,
        &kernel_rows,
        &kernel_cols,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 matvec add wide-load prototype";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_add_shuffle_prototype_f32_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *residual_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    const Aq4MatvecLaunchConfig launch_config =
        aq4_matvec_launch_config_for_fused_kernel(
            device_id,
            8u,
            2u,
            "ULLM_AQ4_MATVEC_ADD_RPB");
    if (launch_config.rows_per_block != 8u) {
        if (error != nullptr) {
            *error = "AQ4 matvec add shuffle prototype requires ULLM_AQ4_MATVEC_ADD_RPB=8";
        }
        return false;
    }
    void *function = hip_aq4_matvec_add_shuffle_prototype_kernel_cache().function_for_device(
        device_id, error);
    if (function == nullptr) {
        return false;
    }
    const size_t grid_size =
        (rows + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec add shuffle prototype row count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *residual_ptr = residual_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &scale_values_ptr,
        &input_ptr,
        &residual_ptr,
        &row_scale_ptr,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_row_scale_count,
        &kernel_rows,
        &kernel_cols,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 matvec add shuffle prototype";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_pair_f32_hip_kernel(
    const ullm_runtime_buffer *left_index_buffer,
    const ullm_runtime_buffer *left_scale_buffer,
    const ullm_runtime_buffer *left_codebook_buffer,
    const ullm_runtime_buffer *left_scale_values_buffer,
    const ullm_runtime_buffer *left_row_scale_buffer,
    size_t left_scale_count,
    size_t left_group_size,
    float left_tensor_scale,
    size_t left_row_scale_count,
    const ullm_runtime_buffer *right_index_buffer,
    const ullm_runtime_buffer *right_scale_buffer,
    const ullm_runtime_buffer *right_codebook_buffer,
    const ullm_runtime_buffer *right_scale_values_buffer,
    const ullm_runtime_buffer *right_row_scale_buffer,
    size_t right_scale_count,
    size_t right_group_size,
    float right_tensor_scale,
    size_t right_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    size_t left_rows,
    size_t right_rows,
    size_t cols,
    ullm_runtime_buffer *left_output_buffer,
    ullm_runtime_buffer *right_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = left_index_buffer->hip_device_id;
    void *function = hip_aq4_matvec_pair_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const Aq4MatvecLaunchConfig launch_config =
        aq4_matvec_launch_config_for_fused_kernel(
            device_id,
            16u,
            4u,
            "ULLM_AQ4_MATVEC_PAIR_RPB");
    const size_t work_rows = std::max(left_rows, right_rows);
    const size_t grid_size =
        (work_rows + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec pair row count exceeds HIP grid limit";
        }
        return false;
    }
    void *left_index_ptr = left_index_buffer->ptr;
    void *left_scale_ptr = left_scale_buffer->ptr;
    void *left_codebook_ptr = left_codebook_buffer->ptr;
    void *left_scale_values_ptr = left_scale_values_buffer->ptr;
    void *left_row_scale_ptr =
        left_row_scale_buffer == nullptr ? nullptr : left_row_scale_buffer->ptr;
    unsigned long long kernel_left_scale_count =
        static_cast<unsigned long long>(left_scale_count);
    unsigned long long kernel_left_group_size =
        static_cast<unsigned long long>(left_group_size);
    unsigned long long kernel_left_row_scale_count =
        static_cast<unsigned long long>(left_row_scale_count);
    void *right_index_ptr = right_index_buffer->ptr;
    void *right_scale_ptr = right_scale_buffer->ptr;
    void *right_codebook_ptr = right_codebook_buffer->ptr;
    void *right_scale_values_ptr = right_scale_values_buffer->ptr;
    void *right_row_scale_ptr =
        right_row_scale_buffer == nullptr ? nullptr : right_row_scale_buffer->ptr;
    unsigned long long kernel_right_scale_count =
        static_cast<unsigned long long>(right_scale_count);
    unsigned long long kernel_right_group_size =
        static_cast<unsigned long long>(right_group_size);
    unsigned long long kernel_right_row_scale_count =
        static_cast<unsigned long long>(right_row_scale_count);
    void *input_ptr = input_buffer->ptr;
    unsigned long long kernel_left_rows = static_cast<unsigned long long>(left_rows);
    unsigned long long kernel_right_rows = static_cast<unsigned long long>(right_rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *left_output_ptr = left_output_buffer->ptr;
    void *right_output_ptr = right_output_buffer->ptr;
    void *kernel_params[] = {
        &left_index_ptr,
        &left_scale_ptr,
        &left_codebook_ptr,
        &left_scale_values_ptr,
        &left_row_scale_ptr,
        &kernel_left_scale_count,
        &kernel_left_group_size,
        &left_tensor_scale,
        &kernel_left_row_scale_count,
        &right_index_ptr,
        &right_scale_ptr,
        &right_codebook_ptr,
        &right_scale_values_ptr,
        &right_row_scale_ptr,
        &kernel_right_scale_count,
        &kernel_right_group_size,
        &right_tensor_scale,
        &kernel_right_row_scale_count,
        &input_ptr,
        &kernel_left_rows,
        &kernel_right_rows,
        &kernel_cols,
        &left_output_ptr,
        &right_output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 matvec pair";
        }
        return false;
    }
    return true;
}

enum class Aq4MatvecTripleKernelKind : uint32_t {
    Production,
    ShufflePrototype,
};

bool aq4_matvec_triple_f32_hip_kernel(
    const ullm_runtime_buffer *first_index_buffer,
    const ullm_runtime_buffer *first_scale_buffer,
    const ullm_runtime_buffer *first_codebook_buffer,
    const ullm_runtime_buffer *first_scale_values_buffer,
    const ullm_runtime_buffer *first_row_scale_buffer,
    size_t first_scale_count,
    size_t first_group_size,
    float first_tensor_scale,
    size_t first_row_scale_count,
    const ullm_runtime_buffer *second_index_buffer,
    const ullm_runtime_buffer *second_scale_buffer,
    const ullm_runtime_buffer *second_codebook_buffer,
    const ullm_runtime_buffer *second_scale_values_buffer,
    const ullm_runtime_buffer *second_row_scale_buffer,
    size_t second_scale_count,
    size_t second_group_size,
    float second_tensor_scale,
    size_t second_row_scale_count,
    const ullm_runtime_buffer *third_index_buffer,
    const ullm_runtime_buffer *third_scale_buffer,
    const ullm_runtime_buffer *third_codebook_buffer,
    const ullm_runtime_buffer *third_scale_values_buffer,
    const ullm_runtime_buffer *third_row_scale_buffer,
    size_t third_scale_count,
    size_t third_group_size,
    float third_tensor_scale,
    size_t third_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    size_t first_rows,
    size_t second_rows,
    size_t third_rows,
    size_t cols,
    ullm_runtime_buffer *first_output_buffer,
    ullm_runtime_buffer *second_output_buffer,
    ullm_runtime_buffer *third_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error,
    Aq4MatvecTripleKernelKind kind = Aq4MatvecTripleKernelKind::Production) {
    const int device_id = first_index_buffer->hip_device_id;
    const Aq4MatvecLaunchConfig launch_config =
        aq4_matvec_launch_config_for_fused_kernel(
            device_id,
            8u,
            4u,
            "ULLM_AQ4_MATVEC_TRIPLE_RPB");
    if (kind == Aq4MatvecTripleKernelKind::ShufflePrototype &&
        launch_config.rows_per_block != 8u) {
        if (error != nullptr) {
            *error = "AQ4 matvec triple shuffle prototype requires ULLM_AQ4_MATVEC_TRIPLE_RPB=8";
        }
        return false;
    }
    void *function = kind == Aq4MatvecTripleKernelKind::ShufflePrototype
        ? hip_aq4_matvec_triple_shuffle_prototype_kernel_cache().function_for_device(device_id, error)
        : hip_aq4_matvec_triple_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }
    const size_t work_rows = std::max(first_rows, std::max(second_rows, third_rows));
    const size_t grid_size =
        (work_rows + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec triple row count exceeds HIP grid limit";
        }
        return false;
    }
    void *first_index_ptr = first_index_buffer->ptr;
    void *first_scale_ptr = first_scale_buffer->ptr;
    void *first_codebook_ptr = first_codebook_buffer->ptr;
    void *first_scale_values_ptr = first_scale_values_buffer->ptr;
    void *first_row_scale_ptr =
        first_row_scale_buffer == nullptr ? nullptr : first_row_scale_buffer->ptr;
    unsigned long long kernel_first_scale_count =
        static_cast<unsigned long long>(first_scale_count);
    unsigned long long kernel_first_group_size =
        static_cast<unsigned long long>(first_group_size);
    unsigned long long kernel_first_row_scale_count =
        static_cast<unsigned long long>(first_row_scale_count);
    void *second_index_ptr = second_index_buffer->ptr;
    void *second_scale_ptr = second_scale_buffer->ptr;
    void *second_codebook_ptr = second_codebook_buffer->ptr;
    void *second_scale_values_ptr = second_scale_values_buffer->ptr;
    void *second_row_scale_ptr =
        second_row_scale_buffer == nullptr ? nullptr : second_row_scale_buffer->ptr;
    unsigned long long kernel_second_scale_count =
        static_cast<unsigned long long>(second_scale_count);
    unsigned long long kernel_second_group_size =
        static_cast<unsigned long long>(second_group_size);
    unsigned long long kernel_second_row_scale_count =
        static_cast<unsigned long long>(second_row_scale_count);
    void *third_index_ptr = third_index_buffer->ptr;
    void *third_scale_ptr = third_scale_buffer->ptr;
    void *third_codebook_ptr = third_codebook_buffer->ptr;
    void *third_scale_values_ptr = third_scale_values_buffer->ptr;
    void *third_row_scale_ptr =
        third_row_scale_buffer == nullptr ? nullptr : third_row_scale_buffer->ptr;
    unsigned long long kernel_third_scale_count =
        static_cast<unsigned long long>(third_scale_count);
    unsigned long long kernel_third_group_size =
        static_cast<unsigned long long>(third_group_size);
    unsigned long long kernel_third_row_scale_count =
        static_cast<unsigned long long>(third_row_scale_count);
    void *input_ptr = input_buffer->ptr;
    unsigned long long kernel_first_rows = static_cast<unsigned long long>(first_rows);
    unsigned long long kernel_second_rows = static_cast<unsigned long long>(second_rows);
    unsigned long long kernel_third_rows = static_cast<unsigned long long>(third_rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *first_output_ptr = first_output_buffer->ptr;
    void *second_output_ptr = second_output_buffer->ptr;
    void *third_output_ptr = third_output_buffer->ptr;
    void *kernel_params[] = {
        &first_index_ptr,
        &first_scale_ptr,
        &first_codebook_ptr,
        &first_scale_values_ptr,
        &first_row_scale_ptr,
        &kernel_first_scale_count,
        &kernel_first_group_size,
        &first_tensor_scale,
        &kernel_first_row_scale_count,
        &second_index_ptr,
        &second_scale_ptr,
        &second_codebook_ptr,
        &second_scale_values_ptr,
        &second_row_scale_ptr,
        &kernel_second_scale_count,
        &kernel_second_group_size,
        &second_tensor_scale,
        &kernel_second_row_scale_count,
        &third_index_ptr,
        &third_scale_ptr,
        &third_codebook_ptr,
        &third_scale_values_ptr,
        &third_row_scale_ptr,
        &kernel_third_scale_count,
        &kernel_third_group_size,
        &third_tensor_scale,
        &kernel_third_row_scale_count,
        &input_ptr,
        &kernel_first_rows,
        &kernel_second_rows,
        &kernel_third_rows,
        &kernel_cols,
        &first_output_ptr,
        &second_output_ptr,
        &third_output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 matvec triple";
        }
        return false;
    }
    return true;
}

enum class Aq4MatvecQkvZGateBetaKernelKind : uint32_t {
    Production,
    WideLoadPrototype,
    ShufflePrototype,
};

bool aq4_matvec_qkv_z_gate_beta_f32_hip_kernel(
    const ullm_runtime_buffer *qkv_index_buffer,
    const ullm_runtime_buffer *qkv_scale_buffer,
    const ullm_runtime_buffer *qkv_codebook_buffer,
    const ullm_runtime_buffer *qkv_scale_values_buffer,
    const ullm_runtime_buffer *qkv_row_scale_buffer,
    size_t qkv_scale_count,
    size_t qkv_group_size,
    float qkv_tensor_scale,
    size_t qkv_row_scale_count,
    const ullm_runtime_buffer *z_index_buffer,
    const ullm_runtime_buffer *z_scale_buffer,
    const ullm_runtime_buffer *z_codebook_buffer,
    const ullm_runtime_buffer *z_scale_values_buffer,
    const ullm_runtime_buffer *z_row_scale_buffer,
    size_t z_scale_count,
    size_t z_group_size,
    float z_tensor_scale,
    size_t z_row_scale_count,
    const ullm_runtime_buffer *a_index_buffer,
    const ullm_runtime_buffer *a_scale_buffer,
    const ullm_runtime_buffer *a_codebook_buffer,
    const ullm_runtime_buffer *a_scale_values_buffer,
    const ullm_runtime_buffer *a_row_scale_buffer,
    size_t a_scale_count,
    size_t a_group_size,
    float a_tensor_scale,
    size_t a_row_scale_count,
    const ullm_runtime_buffer *b_index_buffer,
    const ullm_runtime_buffer *b_scale_buffer,
    const ullm_runtime_buffer *b_codebook_buffer,
    const ullm_runtime_buffer *b_scale_values_buffer,
    const ullm_runtime_buffer *b_row_scale_buffer,
    size_t b_scale_count,
    size_t b_group_size,
    float b_tensor_scale,
    size_t b_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t qkv_rows,
    size_t z_rows,
    size_t heads,
    size_t cols,
    ullm_runtime_buffer *qkv_output_buffer,
    ullm_runtime_buffer *z_output_buffer,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error,
    Aq4MatvecQkvZGateBetaKernelKind kind = Aq4MatvecQkvZGateBetaKernelKind::Production) {
    const int device_id = qkv_index_buffer->hip_device_id;
    if (kind == Aq4MatvecQkvZGateBetaKernelKind::WideLoadPrototype &&
        ((reinterpret_cast<uintptr_t>(qkv_index_buffer->ptr) & 15u) != 0u ||
         (reinterpret_cast<uintptr_t>(z_index_buffer->ptr) & 15u) != 0u ||
         (reinterpret_cast<uintptr_t>(a_index_buffer->ptr) & 15u) != 0u ||
         (reinterpret_cast<uintptr_t>(b_index_buffer->ptr) & 15u) != 0u)) {
        if (error != nullptr) *error = "AQ4 qkv/z gate/beta wide-load prototype requires 16-byte aligned index buffers";
        return false;
    }
    void *function = kind == Aq4MatvecQkvZGateBetaKernelKind::WideLoadPrototype
        ? hip_aq4_matvec_qkv_z_gate_beta_wide_load_prototype_kernel_cache().function_for_device(device_id, error)
        : kind == Aq4MatvecQkvZGateBetaKernelKind::ShufflePrototype
            ? hip_aq4_matvec_qkv_z_gate_beta_shuffle_prototype_kernel_cache().function_for_device(device_id, error)
            : hip_aq4_matvec_qkv_z_gate_beta_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const Aq4MatvecLaunchConfig launch_config =
        aq4_matvec_launch_config_for_fused_kernel(
            device_id,
            4u,
            2u,
            "ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB");
    const size_t max_size = std::numeric_limits<size_t>::max();
    const size_t projection_rows = std::max(qkv_rows, z_rows);
    if (projection_rows > max_size - heads) {
        if (error != nullptr) {
            *error = "AQ4 qkv/z gate/beta row count overflows";
        }
        return false;
    }
    const size_t total_rows = projection_rows + heads;
    const size_t grid_size =
        (total_rows + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 qkv/z gate/beta row count exceeds HIP grid limit";
        }
        return false;
    }

    void *qkv_index_ptr = qkv_index_buffer->ptr;
    void *qkv_scale_ptr = qkv_scale_buffer->ptr;
    void *qkv_codebook_ptr = qkv_codebook_buffer->ptr;
    void *qkv_scale_values_ptr = qkv_scale_values_buffer->ptr;
    void *qkv_row_scale_ptr =
        qkv_row_scale_buffer == nullptr ? nullptr : qkv_row_scale_buffer->ptr;
    unsigned long long kernel_qkv_scale_count =
        static_cast<unsigned long long>(qkv_scale_count);
    unsigned long long kernel_qkv_group_size =
        static_cast<unsigned long long>(qkv_group_size);
    unsigned long long kernel_qkv_row_scale_count =
        static_cast<unsigned long long>(qkv_row_scale_count);
    void *z_index_ptr = z_index_buffer->ptr;
    void *z_scale_ptr = z_scale_buffer->ptr;
    void *z_codebook_ptr = z_codebook_buffer->ptr;
    void *z_scale_values_ptr = z_scale_values_buffer->ptr;
    void *z_row_scale_ptr = z_row_scale_buffer == nullptr ? nullptr : z_row_scale_buffer->ptr;
    unsigned long long kernel_z_scale_count = static_cast<unsigned long long>(z_scale_count);
    unsigned long long kernel_z_group_size = static_cast<unsigned long long>(z_group_size);
    unsigned long long kernel_z_row_scale_count =
        static_cast<unsigned long long>(z_row_scale_count);
    void *a_index_ptr = a_index_buffer->ptr;
    void *a_scale_ptr = a_scale_buffer->ptr;
    void *a_codebook_ptr = a_codebook_buffer->ptr;
    void *a_scale_values_ptr = a_scale_values_buffer->ptr;
    void *a_row_scale_ptr = a_row_scale_buffer == nullptr ? nullptr : a_row_scale_buffer->ptr;
    unsigned long long kernel_a_scale_count = static_cast<unsigned long long>(a_scale_count);
    unsigned long long kernel_a_group_size = static_cast<unsigned long long>(a_group_size);
    unsigned long long kernel_a_row_scale_count =
        static_cast<unsigned long long>(a_row_scale_count);
    void *b_index_ptr = b_index_buffer->ptr;
    void *b_scale_ptr = b_scale_buffer->ptr;
    void *b_codebook_ptr = b_codebook_buffer->ptr;
    void *b_scale_values_ptr = b_scale_values_buffer->ptr;
    void *b_row_scale_ptr = b_row_scale_buffer == nullptr ? nullptr : b_row_scale_buffer->ptr;
    unsigned long long kernel_b_scale_count = static_cast<unsigned long long>(b_scale_count);
    unsigned long long kernel_b_group_size = static_cast<unsigned long long>(b_group_size);
    unsigned long long kernel_b_row_scale_count =
        static_cast<unsigned long long>(b_row_scale_count);
    void *input_ptr = input_buffer->ptr;
    void *a_log_ptr = a_log_buffer->ptr;
    void *dt_bias_ptr = dt_bias_buffer->ptr;
    unsigned long long kernel_qkv_rows = static_cast<unsigned long long>(qkv_rows);
    unsigned long long kernel_z_rows = static_cast<unsigned long long>(z_rows);
    unsigned long long kernel_heads = static_cast<unsigned long long>(heads);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *qkv_output_ptr = qkv_output_buffer->ptr;
    void *z_output_ptr = z_output_buffer->ptr;
    void *gate_output_ptr = gate_output_buffer->ptr;
    void *beta_output_ptr = beta_output_buffer->ptr;
    void *kernel_params[] = {
        &qkv_index_ptr,
        &qkv_scale_ptr,
        &qkv_codebook_ptr,
        &qkv_scale_values_ptr,
        &qkv_row_scale_ptr,
        &kernel_qkv_scale_count,
        &kernel_qkv_group_size,
        &qkv_tensor_scale,
        &kernel_qkv_row_scale_count,
        &z_index_ptr,
        &z_scale_ptr,
        &z_codebook_ptr,
        &z_scale_values_ptr,
        &z_row_scale_ptr,
        &kernel_z_scale_count,
        &kernel_z_group_size,
        &z_tensor_scale,
        &kernel_z_row_scale_count,
        &a_index_ptr,
        &a_scale_ptr,
        &a_codebook_ptr,
        &a_scale_values_ptr,
        &a_row_scale_ptr,
        &kernel_a_scale_count,
        &kernel_a_group_size,
        &a_tensor_scale,
        &kernel_a_row_scale_count,
        &b_index_ptr,
        &b_scale_ptr,
        &b_codebook_ptr,
        &b_scale_values_ptr,
        &b_row_scale_ptr,
        &kernel_b_scale_count,
        &kernel_b_group_size,
        &b_tensor_scale,
        &kernel_b_row_scale_count,
        &input_ptr,
        &a_log_ptr,
        &dt_bias_ptr,
        &kernel_qkv_rows,
        &kernel_z_rows,
        &kernel_heads,
        &kernel_cols,
        &qkv_output_ptr,
        &z_output_ptr,
        &gate_output_ptr,
        &beta_output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 qkv/z gate/beta";
        }
        return false;
    }
    return true;
}

ullm_status aq4_matvec_f32_hip_staging(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t required_index_bytes,
    size_t groups,
    size_t codebook_entries,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<std::uint8_t> host_indices(required_index_bytes);
    std::vector<std::uint8_t> host_scale_indices(groups);
    std::vector<float> host_codebook(codebook_entries);
    std::vector<float> host_scale_values(scale_count);
    std::vector<float> host_input(cols);
    std::vector<float> host_row_scales(row_scale_count);
    std::vector<float> host_output(rows);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = index_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_indices.data(),
            index_buffer->ptr,
            required_index_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_scale_indices.data(),
            scale_buffer->ptr,
            groups,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_codebook.data(),
            codebook_buffer->ptr,
            codebook_buffer->bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_scale_values.data(),
            scale_values_buffer->ptr,
            scale_count * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            cols * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        (row_scale_buffer != nullptr && row_scale_count > 0 &&
         !hip_runtime().copy_async(
             host_row_scales.data(),
             row_scale_buffer->ptr,
             row_scale_count * sizeof(float),
             HIP_MEMCPY_DEVICE_TO_HOST,
             hip_stream,
             device_id))) {
        set_error("failed to copy AQ4 matvec HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!aq4_matvec_f32_host(
            host_indices.data(),
            host_scale_indices.data(),
            host_codebook.data(),
            host_scale_values.data(),
            scale_count,
            group_size,
            tensor_scale,
            host_input.data(),
            row_scale_buffer == nullptr ? nullptr : host_row_scales.data(),
            row_scale_count,
            rows,
            cols,
            host_output.data())) {
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            rows * sizeof(float),
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 matvec output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status aq4_matvec_batch_f32_hip_staging(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t batch_count,
    size_t required_index_bytes,
    size_t groups,
    size_t codebook_entries,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<std::uint8_t> host_indices(required_index_bytes);
    std::vector<std::uint8_t> host_scale_indices(groups);
    std::vector<float> host_codebook(codebook_entries);
    std::vector<float> host_scale_values(scale_count);
    std::vector<float> host_input(batch_count * cols);
    std::vector<float> host_row_scales(row_scale_count);
    std::vector<float> host_output(batch_count * rows);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = index_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_indices.data(),
            index_buffer->ptr,
            required_index_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_scale_indices.data(),
            scale_buffer->ptr,
            groups,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_codebook.data(),
            codebook_buffer->ptr,
            codebook_buffer->bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_scale_values.data(),
            scale_values_buffer->ptr,
            scale_count * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            batch_count * cols * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        (row_scale_buffer != nullptr && row_scale_count > 0 &&
         !hip_runtime().copy_async(
             host_row_scales.data(),
             row_scale_buffer->ptr,
             row_scale_count * sizeof(float),
             HIP_MEMCPY_DEVICE_TO_HOST,
             hip_stream,
             device_id))) {
        set_error("failed to copy AQ4 matvec batch HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec batch HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!aq4_matvec_batch_f32_host(
            host_indices.data(),
            host_scale_indices.data(),
            host_codebook.data(),
            host_scale_values.data(),
            scale_count,
            group_size,
            tensor_scale,
            host_input.data(),
            row_scale_buffer == nullptr ? nullptr : host_row_scales.data(),
            row_scale_count,
            rows,
            cols,
            batch_count,
            host_output.data())) {
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            batch_count * rows * sizeof(float),
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 matvec batch output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec batch HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status aq4_matvec_add_f32_hip_staging(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *residual_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t required_index_bytes,
    size_t groups,
    size_t codebook_entries,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<std::uint8_t> host_indices(required_index_bytes);
    std::vector<std::uint8_t> host_scale_indices(groups);
    std::vector<float> host_codebook(codebook_entries);
    std::vector<float> host_scale_values(scale_count);
    std::vector<float> host_input(cols);
    std::vector<float> host_residual(rows);
    std::vector<float> host_row_scales(row_scale_count);
    std::vector<float> host_output(rows);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = index_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_indices.data(),
            index_buffer->ptr,
            required_index_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_scale_indices.data(),
            scale_buffer->ptr,
            groups,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_codebook.data(),
            codebook_buffer->ptr,
            codebook_buffer->bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_scale_values.data(),
            scale_values_buffer->ptr,
            scale_count * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            cols * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_residual.data(),
            residual_buffer->ptr,
            rows * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        (row_scale_buffer != nullptr && row_scale_count > 0 &&
         !hip_runtime().copy_async(
             host_row_scales.data(),
             row_scale_buffer->ptr,
             row_scale_count * sizeof(float),
             HIP_MEMCPY_DEVICE_TO_HOST,
             hip_stream,
             device_id))) {
        set_error("failed to copy AQ4 matvec add HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec add HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!aq4_matvec_add_f32_host(
            host_indices.data(),
            host_scale_indices.data(),
            host_codebook.data(),
            host_scale_values.data(),
            scale_count,
            group_size,
            tensor_scale,
            host_input.data(),
            host_residual.data(),
            row_scale_buffer == nullptr ? nullptr : host_row_scales.data(),
            row_scale_count,
            rows,
            cols,
            host_output.data())) {
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            rows * sizeof(float),
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 matvec add output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec add HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipAq4MatvecSiluMulKernelCache {
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
            if (!hiprtc_runtime().compile_aq4_matvec_silu_mul_kernel(arch, &code, &compile_error)) {
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
                    "ullm_aq4_matvec_silu_mul_f32_kernel",
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
            compile_errors.empty() ? "failed to build AQ4 matvec SiLU-mul HIP kernel" : compile_errors);
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

HipAq4MatvecSiluMulKernelCache &hip_aq4_matvec_silu_mul_kernel_cache() {
    static HipAq4MatvecSiluMulKernelCache cache;
    return cache;
}

enum class Aq4MatvecSiluMulKernelKind : uint32_t {
    Production,
    WideLoadPrototype,
    ShufflePrototype,
};

bool aq4_matvec_silu_mul_f32_hip_kernel(
    const ullm_runtime_buffer *gate_index_buffer,
    const ullm_runtime_buffer *gate_scale_buffer,
    const ullm_runtime_buffer *gate_codebook_buffer,
    const ullm_runtime_buffer *gate_scale_values_buffer,
    const ullm_runtime_buffer *gate_row_scale_buffer,
    size_t gate_scale_count,
    size_t gate_group_size,
    float gate_tensor_scale,
    size_t gate_row_scale_count,
    const ullm_runtime_buffer *up_index_buffer,
    const ullm_runtime_buffer *up_scale_buffer,
    const ullm_runtime_buffer *up_codebook_buffer,
    const ullm_runtime_buffer *up_scale_values_buffer,
    const ullm_runtime_buffer *up_row_scale_buffer,
    size_t up_scale_count,
    size_t up_group_size,
    float up_tensor_scale,
    size_t up_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error,
    Aq4MatvecSiluMulKernelKind kind = Aq4MatvecSiluMulKernelKind::Production) {
    const int device_id = gate_index_buffer->hip_device_id;
    if (kind == Aq4MatvecSiluMulKernelKind::WideLoadPrototype &&
        ((reinterpret_cast<uintptr_t>(gate_index_buffer->ptr) & 15u) != 0u ||
         (reinterpret_cast<uintptr_t>(up_index_buffer->ptr) & 15u) != 0u)) {
        if (error != nullptr) *error = "AQ4 SiLU-mul wide-load prototype requires 16-byte aligned index buffers";
        return false;
    }
    void *function = kind == Aq4MatvecSiluMulKernelKind::WideLoadPrototype
        ? hip_aq4_matvec_silu_mul_wide_load_prototype_kernel_cache().function_for_device(device_id, error)
        : kind == Aq4MatvecSiluMulKernelKind::ShufflePrototype
            ? hip_aq4_matvec_silu_mul_shuffle_prototype_kernel_cache().function_for_device(device_id, error)
            : hip_aq4_matvec_silu_mul_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const Aq4MatvecLaunchConfig launch_config =
        aq4_matvec_launch_config_for_fused_kernel(
            device_id,
            8u,
            2u,
            "ULLM_AQ4_MATVEC_SILU_MUL_RPB");
    const size_t grid_size =
        (rows + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec SiLU-mul row count exceeds HIP grid limit";
        }
        return false;
    }
    void *gate_index_ptr = gate_index_buffer->ptr;
    void *gate_scale_ptr = gate_scale_buffer->ptr;
    void *gate_codebook_ptr = gate_codebook_buffer->ptr;
    void *gate_scale_values_ptr = gate_scale_values_buffer->ptr;
    void *gate_row_scale_ptr =
        gate_row_scale_buffer == nullptr ? nullptr : gate_row_scale_buffer->ptr;
    unsigned long long kernel_gate_scale_count =
        static_cast<unsigned long long>(gate_scale_count);
    unsigned long long kernel_gate_group_size = static_cast<unsigned long long>(gate_group_size);
    unsigned long long kernel_gate_row_scale_count =
        static_cast<unsigned long long>(gate_row_scale_count);
    void *up_index_ptr = up_index_buffer->ptr;
    void *up_scale_ptr = up_scale_buffer->ptr;
    void *up_codebook_ptr = up_codebook_buffer->ptr;
    void *up_scale_values_ptr = up_scale_values_buffer->ptr;
    void *up_row_scale_ptr = up_row_scale_buffer == nullptr ? nullptr : up_row_scale_buffer->ptr;
    unsigned long long kernel_up_scale_count = static_cast<unsigned long long>(up_scale_count);
    unsigned long long kernel_up_group_size = static_cast<unsigned long long>(up_group_size);
    unsigned long long kernel_up_row_scale_count =
        static_cast<unsigned long long>(up_row_scale_count);
    void *input_ptr = input_buffer->ptr;
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &gate_index_ptr,
        &gate_scale_ptr,
        &gate_codebook_ptr,
        &gate_scale_values_ptr,
        &gate_row_scale_ptr,
        &kernel_gate_scale_count,
        &kernel_gate_group_size,
        &gate_tensor_scale,
        &kernel_gate_row_scale_count,
        &up_index_ptr,
        &up_scale_ptr,
        &up_codebook_ptr,
        &up_scale_values_ptr,
        &up_row_scale_ptr,
        &kernel_up_scale_count,
        &kernel_up_group_size,
        &up_tensor_scale,
        &kernel_up_row_scale_count,
        &input_ptr,
        &kernel_rows,
        &kernel_cols,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 matvec SiLU-mul";
        }
        return false;
    }
    return true;
}

ullm_status aq4_matvec_silu_mul_f32_hip_staging(
    const ullm_runtime_buffer *gate_index_buffer,
    const ullm_runtime_buffer *gate_scale_buffer,
    const ullm_runtime_buffer *gate_codebook_buffer,
    const ullm_runtime_buffer *gate_scale_values_buffer,
    const ullm_runtime_buffer *gate_row_scale_buffer,
    size_t gate_scale_count,
    size_t gate_group_size,
    float gate_tensor_scale,
    size_t gate_row_scale_count,
    size_t gate_required_index_bytes,
    size_t gate_groups,
    size_t gate_codebook_entries,
    const ullm_runtime_buffer *up_index_buffer,
    const ullm_runtime_buffer *up_scale_buffer,
    const ullm_runtime_buffer *up_codebook_buffer,
    const ullm_runtime_buffer *up_scale_values_buffer,
    const ullm_runtime_buffer *up_row_scale_buffer,
    size_t up_scale_count,
    size_t up_group_size,
    float up_tensor_scale,
    size_t up_row_scale_count,
    size_t up_required_index_bytes,
    size_t up_groups,
    size_t up_codebook_entries,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<std::uint8_t> host_gate_indices(gate_required_index_bytes);
    std::vector<std::uint8_t> host_gate_scale_indices(gate_groups);
    std::vector<float> host_gate_codebook(gate_codebook_entries);
    std::vector<float> host_gate_scale_values(gate_scale_count);
    std::vector<float> host_gate_row_scales(gate_row_scale_count);
    std::vector<std::uint8_t> host_up_indices(up_required_index_bytes);
    std::vector<std::uint8_t> host_up_scale_indices(up_groups);
    std::vector<float> host_up_codebook(up_codebook_entries);
    std::vector<float> host_up_scale_values(up_scale_count);
    std::vector<float> host_up_row_scales(up_row_scale_count);
    std::vector<float> host_input(cols);
    std::vector<float> host_output(rows);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = gate_index_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_gate_indices.data(),
            gate_index_buffer->ptr,
            gate_required_index_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_gate_scale_indices.data(),
            gate_scale_buffer->ptr,
            gate_groups,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_gate_codebook.data(),
            gate_codebook_buffer->ptr,
            gate_codebook_buffer->bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_gate_scale_values.data(),
            gate_scale_values_buffer->ptr,
            gate_scale_count * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        (gate_row_scale_buffer != nullptr && gate_row_scale_count > 0 &&
         !hip_runtime().copy_async(
             host_gate_row_scales.data(),
             gate_row_scale_buffer->ptr,
             gate_row_scale_count * sizeof(float),
             HIP_MEMCPY_DEVICE_TO_HOST,
             hip_stream,
             device_id)) ||
        !hip_runtime().copy_async(
            host_up_indices.data(),
            up_index_buffer->ptr,
            up_required_index_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_up_scale_indices.data(),
            up_scale_buffer->ptr,
            up_groups,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_up_codebook.data(),
            up_codebook_buffer->ptr,
            up_codebook_buffer->bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_up_scale_values.data(),
            up_scale_values_buffer->ptr,
            up_scale_count * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        (up_row_scale_buffer != nullptr && up_row_scale_count > 0 &&
         !hip_runtime().copy_async(
             host_up_row_scales.data(),
             up_row_scale_buffer->ptr,
             up_row_scale_count * sizeof(float),
             HIP_MEMCPY_DEVICE_TO_HOST,
             hip_stream,
             device_id)) ||
        !hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            cols * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 matvec SiLU-mul HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec SiLU-mul HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!aq4_matvec_silu_mul_f32_host(
            host_gate_indices.data(),
            host_gate_scale_indices.data(),
            host_gate_codebook.data(),
            host_gate_scale_values.data(),
            gate_scale_count,
            gate_group_size,
            gate_tensor_scale,
            gate_row_scale_buffer == nullptr ? nullptr : host_gate_row_scales.data(),
            gate_row_scale_count,
            host_up_indices.data(),
            host_up_scale_indices.data(),
            host_up_codebook.data(),
            host_up_scale_values.data(),
            up_scale_count,
            up_group_size,
            up_tensor_scale,
            up_row_scale_buffer == nullptr ? nullptr : host_up_row_scales.data(),
            up_row_scale_count,
            host_input.data(),
            rows,
            cols,
            host_output.data())) {
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            rows * sizeof(float),
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 matvec SiLU-mul output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec SiLU-mul HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipAq4MatvecGateBetaKernelCache {
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
            if (!hiprtc_runtime().compile_aq4_matvec_gate_beta_kernel(arch, &code, &compile_error)) {
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
                    "ullm_aq4_matvec_gate_beta_f32_kernel",
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
            compile_errors.empty() ? "failed to build AQ4 matvec gate/beta HIP kernel" : compile_errors);
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

HipAq4MatvecGateBetaKernelCache &hip_aq4_matvec_gate_beta_kernel_cache() {
    static HipAq4MatvecGateBetaKernelCache cache;
    return cache;
}

bool aq4_matvec_gate_beta_f32_hip_kernel(
    const ullm_runtime_buffer *a_index_buffer,
    const ullm_runtime_buffer *a_scale_buffer,
    const ullm_runtime_buffer *a_codebook_buffer,
    const ullm_runtime_buffer *a_scale_values_buffer,
    const ullm_runtime_buffer *a_row_scale_buffer,
    size_t a_scale_count,
    size_t a_group_size,
    float a_tensor_scale,
    size_t a_row_scale_count,
    const ullm_runtime_buffer *b_index_buffer,
    const ullm_runtime_buffer *b_scale_buffer,
    const ullm_runtime_buffer *b_codebook_buffer,
    const ullm_runtime_buffer *b_scale_values_buffer,
    const ullm_runtime_buffer *b_row_scale_buffer,
    size_t b_scale_count,
    size_t b_group_size,
    float b_tensor_scale,
    size_t b_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t heads,
    size_t cols,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = a_index_buffer->hip_device_id;
    void *function = hip_aq4_matvec_gate_beta_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const Aq4MatvecLaunchConfig launch_config = aq4_matvec_launch_config_for_device(device_id);
    const size_t grid_size =
        (heads + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec gate/beta head count exceeds HIP grid limit";
        }
        return false;
    }
    void *a_index_ptr = a_index_buffer->ptr;
    void *a_scale_ptr = a_scale_buffer->ptr;
    void *a_codebook_ptr = a_codebook_buffer->ptr;
    void *a_scale_values_ptr = a_scale_values_buffer->ptr;
    void *a_row_scale_ptr = a_row_scale_buffer == nullptr ? nullptr : a_row_scale_buffer->ptr;
    unsigned long long kernel_a_scale_count = static_cast<unsigned long long>(a_scale_count);
    unsigned long long kernel_a_group_size = static_cast<unsigned long long>(a_group_size);
    unsigned long long kernel_a_row_scale_count =
        static_cast<unsigned long long>(a_row_scale_count);
    void *b_index_ptr = b_index_buffer->ptr;
    void *b_scale_ptr = b_scale_buffer->ptr;
    void *b_codebook_ptr = b_codebook_buffer->ptr;
    void *b_scale_values_ptr = b_scale_values_buffer->ptr;
    void *b_row_scale_ptr = b_row_scale_buffer == nullptr ? nullptr : b_row_scale_buffer->ptr;
    unsigned long long kernel_b_scale_count = static_cast<unsigned long long>(b_scale_count);
    unsigned long long kernel_b_group_size = static_cast<unsigned long long>(b_group_size);
    unsigned long long kernel_b_row_scale_count =
        static_cast<unsigned long long>(b_row_scale_count);
    void *input_ptr = input_buffer->ptr;
    void *a_log_ptr = a_log_buffer->ptr;
    void *dt_bias_ptr = dt_bias_buffer->ptr;
    unsigned long long kernel_heads = static_cast<unsigned long long>(heads);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *gate_output_ptr = gate_output_buffer->ptr;
    void *beta_output_ptr = beta_output_buffer->ptr;
    void *kernel_params[] = {
        &a_index_ptr,
        &a_scale_ptr,
        &a_codebook_ptr,
        &a_scale_values_ptr,
        &a_row_scale_ptr,
        &kernel_a_scale_count,
        &kernel_a_group_size,
        &a_tensor_scale,
        &kernel_a_row_scale_count,
        &b_index_ptr,
        &b_scale_ptr,
        &b_codebook_ptr,
        &b_scale_values_ptr,
        &b_row_scale_ptr,
        &kernel_b_scale_count,
        &kernel_b_group_size,
        &b_tensor_scale,
        &kernel_b_row_scale_count,
        &input_ptr,
        &a_log_ptr,
        &dt_bias_ptr,
        &kernel_heads,
        &kernel_cols,
        &gate_output_ptr,
        &beta_output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 matvec gate/beta";
        }
        return false;
    }
    return true;
}

ullm_status aq4_matvec_gate_beta_f32_hip_staging(
    const ullm_runtime_buffer *a_index_buffer,
    const ullm_runtime_buffer *a_scale_buffer,
    const ullm_runtime_buffer *a_codebook_buffer,
    const ullm_runtime_buffer *a_scale_values_buffer,
    const ullm_runtime_buffer *a_row_scale_buffer,
    size_t a_scale_count,
    size_t a_group_size,
    float a_tensor_scale,
    size_t a_row_scale_count,
    size_t a_required_index_bytes,
    size_t a_groups,
    size_t a_codebook_entries,
    const ullm_runtime_buffer *b_index_buffer,
    const ullm_runtime_buffer *b_scale_buffer,
    const ullm_runtime_buffer *b_codebook_buffer,
    const ullm_runtime_buffer *b_scale_values_buffer,
    const ullm_runtime_buffer *b_row_scale_buffer,
    size_t b_scale_count,
    size_t b_group_size,
    float b_tensor_scale,
    size_t b_row_scale_count,
    size_t b_required_index_bytes,
    size_t b_groups,
    size_t b_codebook_entries,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t heads,
    size_t cols,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<std::uint8_t> host_a_indices(a_required_index_bytes);
    std::vector<std::uint8_t> host_a_scale_indices(a_groups);
    std::vector<float> host_a_codebook(a_codebook_entries);
    std::vector<float> host_a_scale_values(a_scale_count);
    std::vector<float> host_a_row_scales(a_row_scale_count);
    std::vector<std::uint8_t> host_b_indices(b_required_index_bytes);
    std::vector<std::uint8_t> host_b_scale_indices(b_groups);
    std::vector<float> host_b_codebook(b_codebook_entries);
    std::vector<float> host_b_scale_values(b_scale_count);
    std::vector<float> host_b_row_scales(b_row_scale_count);
    std::vector<float> host_input(cols);
    std::vector<float> host_a_log(heads);
    std::vector<float> host_dt_bias(heads);
    std::vector<float> host_gate(heads);
    std::vector<float> host_beta(heads);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = a_index_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_a_indices.data(),
            a_index_buffer->ptr,
            a_required_index_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_a_scale_indices.data(),
            a_scale_buffer->ptr,
            a_groups,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_a_codebook.data(),
            a_codebook_buffer->ptr,
            a_codebook_buffer->bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_a_scale_values.data(),
            a_scale_values_buffer->ptr,
            a_scale_count * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        (a_row_scale_buffer != nullptr && a_row_scale_count > 0 &&
         !hip_runtime().copy_async(
             host_a_row_scales.data(),
             a_row_scale_buffer->ptr,
             a_row_scale_count * sizeof(float),
             HIP_MEMCPY_DEVICE_TO_HOST,
             hip_stream,
             device_id)) ||
        !hip_runtime().copy_async(
            host_b_indices.data(),
            b_index_buffer->ptr,
            b_required_index_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_b_scale_indices.data(),
            b_scale_buffer->ptr,
            b_groups,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_b_codebook.data(),
            b_codebook_buffer->ptr,
            b_codebook_buffer->bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_b_scale_values.data(),
            b_scale_values_buffer->ptr,
            b_scale_count * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        (b_row_scale_buffer != nullptr && b_row_scale_count > 0 &&
         !hip_runtime().copy_async(
             host_b_row_scales.data(),
             b_row_scale_buffer->ptr,
             b_row_scale_count * sizeof(float),
             HIP_MEMCPY_DEVICE_TO_HOST,
             hip_stream,
             device_id)) ||
        !hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            cols * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_a_log.data(),
            a_log_buffer->ptr,
            heads * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_dt_bias.data(),
            dt_bias_buffer->ptr,
            heads * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 matvec gate/beta HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec gate/beta HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!aq4_matvec_gate_beta_f32_host(
            host_a_indices.data(),
            host_a_scale_indices.data(),
            host_a_codebook.data(),
            host_a_scale_values.data(),
            a_scale_count,
            a_group_size,
            a_tensor_scale,
            a_row_scale_buffer == nullptr ? nullptr : host_a_row_scales.data(),
            a_row_scale_count,
            host_b_indices.data(),
            host_b_scale_indices.data(),
            host_b_codebook.data(),
            host_b_scale_values.data(),
            b_scale_count,
            b_group_size,
            b_tensor_scale,
            b_row_scale_buffer == nullptr ? nullptr : host_b_row_scales.data(),
            b_row_scale_count,
            host_input.data(),
            host_a_log.data(),
            host_dt_bias.data(),
            heads,
            cols,
            host_gate.data(),
            host_beta.data())) {
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!hip_runtime().copy_async(
            gate_output_buffer->ptr,
            host_gate.data(),
            heads * sizeof(float),
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            beta_output_buffer->ptr,
            host_beta.data(),
            heads * sizeof(float),
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 matvec gate/beta output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec gate/beta HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

void matvec_f32_host(
    const float *matrix,
    const float *input,
    size_t rows,
    size_t cols,
    float *output) {
    for (size_t row = 0; row < rows; ++row) {
        const float *row_values = matrix + row * cols;
        float sum = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            sum += row_values[col] * input[col];
        }
        output[row] = sum;
    }
}

#include "../kernels/sq8_0/sq8_0_matvec_runtime.inc"
#include "../kernels/sq8_1/sq8_1_matvec_runtime.inc"

float bf16_to_f32_host(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void matvec_bf16_f32_host(
    const uint16_t *matrix,
    const float *input,
    size_t rows,
    size_t cols,
    float *output) {
    for (size_t row = 0; row < rows; ++row) {
        const uint16_t *row_values = matrix + row * cols;
        float sum = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            sum += bf16_to_f32_host(row_values[col]) * input[col];
        }
        output[row] = sum;
    }
}

void bf16_row_f32_host(
    const uint16_t *matrix,
    size_t rows,
    size_t cols,
    size_t row_index,
    float *output) {
    if (row_index >= rows) {
        return;
    }
    const uint16_t *row_values = matrix + row_index * cols;
    for (size_t col = 0; col < cols; ++col) {
        output[col] = bf16_to_f32_host(row_values[col]);
    }
}

