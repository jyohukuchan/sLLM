// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

float sq_fp8_e4m3fn_to_f32(uint8_t value) {
    const unsigned int raw = static_cast<unsigned int>(value);
    const unsigned int sign = raw >> 7;
    const unsigned int exponent = (raw >> 3) & 0x0fu;
    const unsigned int mantissa = raw & 0x07u;
    if (exponent == 0x0fu && mantissa == 0x07u) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    float magnitude = 0.0f;
    if (exponent == 0u) {
        magnitude = static_cast<float>(mantissa) * 0.001953125f;
    } else {
        magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) * 0.125f,
                               static_cast<int>(exponent) - 7);
    }
    return sign == 0u ? magnitude : -magnitude;
}

bool sq_fp8_matvec_f32_host(
    const uint8_t *payload,
    const float *scales,
    const float *input,
    size_t rows,
    size_t cols,
    uint32_t scale_kind,
    size_t scale_block_rows,
    size_t scale_block_cols,
    float *output) {
    if (payload == nullptr || scales == nullptr || input == nullptr || output == nullptr) {
        return false;
    }
    size_t blocks_per_row = 1;
    if (scale_kind == 2u) {
        if (scale_block_cols == 0) {
            return false;
        }
        if (scale_block_rows == 0) {
            return false;
        }
        blocks_per_row = 1 + (cols - 1) / scale_block_cols;
    }
    for (size_t row = 0; row < rows; ++row) {
        const size_t row_offset = row * cols;
        float sum = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            float scale = scales[0];
            if (scale_kind == 1u) {
                scale = scales[row];
            } else if (scale_kind == 2u) {
                scale = scales[(row / scale_block_rows) * blocks_per_row +
                               col / scale_block_cols];
            }
            const float value = sq_fp8_e4m3fn_to_f32(payload[row_offset + col]);
            if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 0.0f) {
                return false;
            }
            sum += value * scale * input[col];
        }
        output[row] = sum;
    }
    return true;
}

bool sq_fp8_matvec_batch_f32_host(
    const uint8_t *payload,
    const float *scales,
    const float *input,
    size_t rows,
    size_t cols,
    uint32_t scale_kind,
    size_t scale_block_rows,
    size_t scale_block_cols,
    size_t batch_count,
    float *output) {
    if (payload == nullptr || scales == nullptr || input == nullptr || output == nullptr) {
        return false;
    }
    for (size_t batch = 0; batch < batch_count; ++batch) {
        if (!sq_fp8_matvec_f32_host(
                payload,
                scales,
                input + batch * cols,
                rows,
                cols,
                scale_kind,
                scale_block_rows,
                scale_block_cols,
                output + batch * rows)) {
            return false;
        }
    }
    return true;
}

bool sq_fp8_matvec_pair_f32_host(
    const uint8_t *left_payload,
    const float *left_scales,
    uint32_t left_scale_kind,
    size_t left_scale_block_cols,
    const uint8_t *right_payload,
    const float *right_scales,
    uint32_t right_scale_kind,
    size_t right_scale_block_cols,
    const float *input,
    size_t left_rows,
    size_t right_rows,
    size_t cols,
    float *left_output,
    float *right_output) {
    return sq_fp8_matvec_f32_host(
               left_payload,
               left_scales,
               input,
               left_rows,
               cols,
               left_scale_kind,
               1,
               left_scale_block_cols,
               left_output) &&
           sq_fp8_matvec_f32_host(
               right_payload,
               right_scales,
               input,
               right_rows,
               cols,
               right_scale_kind,
               1,
               right_scale_block_cols,
               right_output);
}

bool sq_fp8_matvec_triple_f32_host(
    const uint8_t *first_payload,
    const float *first_scales,
    uint32_t first_scale_kind,
    size_t first_scale_block_cols,
    const uint8_t *second_payload,
    const float *second_scales,
    uint32_t second_scale_kind,
    size_t second_scale_block_cols,
    const uint8_t *third_payload,
    const float *third_scales,
    uint32_t third_scale_kind,
    size_t third_scale_block_cols,
    const float *input,
    size_t first_rows,
    size_t second_rows,
    size_t third_rows,
    size_t cols,
    float *first_output,
    float *second_output,
    float *third_output) {
    return sq_fp8_matvec_f32_host(
               first_payload,
               first_scales,
               input,
               first_rows,
               cols,
               first_scale_kind,
               1,
               first_scale_block_cols,
               first_output) &&
           sq_fp8_matvec_f32_host(
               second_payload,
               second_scales,
               input,
               second_rows,
               cols,
               second_scale_kind,
               1,
               second_scale_block_cols,
               second_output) &&
           sq_fp8_matvec_f32_host(
               third_payload,
               third_scales,
               input,
               third_rows,
               cols,
               third_scale_kind,
               1,
               third_scale_block_cols,
               third_output);
}

class HipSqFp8MatvecKernelCache {
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
            if (!hiprtc_runtime().compile_sq_fp8_matvec_kernel(arch, &code, &compile_error)) {
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
                    "ullm_sq_fp8_matvec_f32_kernel",
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
            compile_errors.empty() ? "failed to build SQ FP8 matvec HIP kernel" :
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

HipSqFp8MatvecKernelCache &hip_sq_fp8_matvec_kernel_cache() {
    static HipSqFp8MatvecKernelCache cache;
    return cache;
}

class HipSqFp8MatvecBatchKernelCache {
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
            if (!hiprtc_runtime().compile_sq_fp8_matvec_batch_kernel(arch, &code, &compile_error)) {
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
                    "ullm_sq_fp8_matvec_batch_f32_kernel",
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
            compile_errors.empty() ? "failed to build SQ FP8 matvec batch HIP kernel" :
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

HipSqFp8MatvecBatchKernelCache &hip_sq_fp8_matvec_batch_kernel_cache() {
    static HipSqFp8MatvecBatchKernelCache cache;
    return cache;
}

class HipSqFp8MatvecPairKernelCache {
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
            if (!hiprtc_runtime().compile_sq_fp8_matvec_pair_kernel(arch, &code, &compile_error)) {
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
                    "ullm_sq_fp8_matvec_pair_f32_kernel",
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
            compile_errors.empty() ? "failed to build SQ FP8 matvec pair HIP kernel" :
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

HipSqFp8MatvecPairKernelCache &hip_sq_fp8_matvec_pair_kernel_cache() {
    static HipSqFp8MatvecPairKernelCache cache;
    return cache;
}

class HipSqFp8MatvecTripleKernelCache {
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
            if (!hiprtc_runtime().compile_sq_fp8_matvec_triple_kernel(arch, &code, &compile_error)) {
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
                    "ullm_sq_fp8_matvec_triple_f32_kernel",
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
            compile_errors.empty() ? "failed to build SQ FP8 matvec triple HIP kernel" :
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

HipSqFp8MatvecTripleKernelCache &hip_sq_fp8_matvec_triple_kernel_cache() {
    static HipSqFp8MatvecTripleKernelCache cache;
    return cache;
}

bool sq_fp8_matvec_f32_hip_kernel(
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    uint32_t scale_kind,
    size_t scale_block_rows,
    size_t scale_block_cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = payload_buffer->hip_device_id;
    void *function = hip_sq_fp8_matvec_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    if (rows > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "SQ FP8 matvec row count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_scale_block_cols =
        static_cast<unsigned long long>(scale_block_cols);
    unsigned long long kernel_scale_block_rows =
        static_cast<unsigned long long>(scale_block_rows);
    unsigned int kernel_scale_kind = scale_kind;
    void *payload_ptr = payload_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &payload_ptr,
        &scale_ptr,
        &input_ptr,
        &kernel_rows,
        &kernel_cols,
        &kernel_scale_kind,
        &kernel_scale_block_rows,
        &kernel_scale_block_cols,
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
            *error = "hipModuleLaunchKernel failed for SQ FP8 matvec";
        }
        return false;
    }
    return true;
}

bool sq_fp8_matvec_batch_f32_hip_kernel(
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    uint32_t scale_kind,
    size_t scale_block_rows,
    size_t scale_block_cols,
    size_t batch_count,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = payload_buffer->hip_device_id;
    void *function = hip_sq_fp8_matvec_batch_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    if (rows > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        batch_count > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "SQ FP8 matvec batch row or batch count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_scale_block_cols =
        static_cast<unsigned long long>(scale_block_cols);
    unsigned long long kernel_scale_block_rows =
        static_cast<unsigned long long>(scale_block_rows);
    unsigned long long kernel_batch_count = static_cast<unsigned long long>(batch_count);
    unsigned int kernel_scale_kind = scale_kind;
    void *payload_ptr = payload_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &payload_ptr,
        &scale_ptr,
        &input_ptr,
        &kernel_rows,
        &kernel_cols,
        &kernel_scale_kind,
        &kernel_scale_block_rows,
        &kernel_scale_block_cols,
        &kernel_batch_count,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel_2d(
            function,
            static_cast<unsigned int>(rows),
            static_cast<unsigned int>(batch_count),
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for SQ FP8 matvec batch";
        }
        return false;
    }
    return true;
}

bool sq_fp8_matvec_pair_f32_hip_kernel(
    const ullm_runtime_buffer *left_payload_buffer,
    const ullm_runtime_buffer *left_scale_buffer,
    uint32_t left_scale_kind,
    size_t left_scale_block_cols,
    const ullm_runtime_buffer *right_payload_buffer,
    const ullm_runtime_buffer *right_scale_buffer,
    uint32_t right_scale_kind,
    size_t right_scale_block_cols,
    const ullm_runtime_buffer *input_buffer,
    size_t left_rows,
    size_t right_rows,
    size_t cols,
    ullm_runtime_buffer *left_output_buffer,
    ullm_runtime_buffer *right_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = left_payload_buffer->hip_device_id;
    void *function = hip_sq_fp8_matvec_pair_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t max_rows = std::max(left_rows, right_rows);
    if (max_rows > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "SQ FP8 matvec pair row count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_left_rows = static_cast<unsigned long long>(left_rows);
    unsigned long long kernel_right_rows = static_cast<unsigned long long>(right_rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_left_scale_block_cols =
        static_cast<unsigned long long>(left_scale_block_cols);
    unsigned long long kernel_right_scale_block_cols =
        static_cast<unsigned long long>(right_scale_block_cols);
    unsigned int kernel_left_scale_kind = left_scale_kind;
    unsigned int kernel_right_scale_kind = right_scale_kind;
    void *left_payload_ptr = left_payload_buffer->ptr;
    void *left_scale_ptr = left_scale_buffer->ptr;
    void *right_payload_ptr = right_payload_buffer->ptr;
    void *right_scale_ptr = right_scale_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *left_output_ptr = left_output_buffer->ptr;
    void *right_output_ptr = right_output_buffer->ptr;
    void *kernel_params[] = {
        &left_payload_ptr,
        &left_scale_ptr,
        &kernel_left_rows,
        &kernel_left_scale_kind,
        &kernel_left_scale_block_cols,
        &right_payload_ptr,
        &right_scale_ptr,
        &kernel_right_rows,
        &kernel_right_scale_kind,
        &kernel_right_scale_block_cols,
        &input_ptr,
        &kernel_cols,
        &left_output_ptr,
        &right_output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel_2d(
            function,
            static_cast<unsigned int>(max_rows),
            2,
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for SQ FP8 matvec pair";
        }
        return false;
    }
    return true;
}

bool sq_fp8_matvec_triple_f32_hip_kernel(
    const ullm_runtime_buffer *first_payload_buffer,
    const ullm_runtime_buffer *first_scale_buffer,
    uint32_t first_scale_kind,
    size_t first_scale_block_cols,
    const ullm_runtime_buffer *second_payload_buffer,
    const ullm_runtime_buffer *second_scale_buffer,
    uint32_t second_scale_kind,
    size_t second_scale_block_cols,
    const ullm_runtime_buffer *third_payload_buffer,
    const ullm_runtime_buffer *third_scale_buffer,
    uint32_t third_scale_kind,
    size_t third_scale_block_cols,
    const ullm_runtime_buffer *input_buffer,
    size_t first_rows,
    size_t second_rows,
    size_t third_rows,
    size_t cols,
    ullm_runtime_buffer *first_output_buffer,
    ullm_runtime_buffer *second_output_buffer,
    ullm_runtime_buffer *third_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = first_payload_buffer->hip_device_id;
    void *function = hip_sq_fp8_matvec_triple_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t max_rows = std::max(first_rows, std::max(second_rows, third_rows));
    if (max_rows > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "SQ FP8 matvec triple row count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_first_rows = static_cast<unsigned long long>(first_rows);
    unsigned long long kernel_second_rows = static_cast<unsigned long long>(second_rows);
    unsigned long long kernel_third_rows = static_cast<unsigned long long>(third_rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_first_scale_block_cols =
        static_cast<unsigned long long>(first_scale_block_cols);
    unsigned long long kernel_second_scale_block_cols =
        static_cast<unsigned long long>(second_scale_block_cols);
    unsigned long long kernel_third_scale_block_cols =
        static_cast<unsigned long long>(third_scale_block_cols);
    unsigned int kernel_first_scale_kind = first_scale_kind;
    unsigned int kernel_second_scale_kind = second_scale_kind;
    unsigned int kernel_third_scale_kind = third_scale_kind;
    void *first_payload_ptr = first_payload_buffer->ptr;
    void *first_scale_ptr = first_scale_buffer->ptr;
    void *second_payload_ptr = second_payload_buffer->ptr;
    void *second_scale_ptr = second_scale_buffer->ptr;
    void *third_payload_ptr = third_payload_buffer->ptr;
    void *third_scale_ptr = third_scale_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *first_output_ptr = first_output_buffer->ptr;
    void *second_output_ptr = second_output_buffer->ptr;
    void *third_output_ptr = third_output_buffer->ptr;
    void *kernel_params[] = {
        &first_payload_ptr,
        &first_scale_ptr,
        &kernel_first_rows,
        &kernel_first_scale_kind,
        &kernel_first_scale_block_cols,
        &second_payload_ptr,
        &second_scale_ptr,
        &kernel_second_rows,
        &kernel_second_scale_kind,
        &kernel_second_scale_block_cols,
        &third_payload_ptr,
        &third_scale_ptr,
        &kernel_third_rows,
        &kernel_third_scale_kind,
        &kernel_third_scale_block_cols,
        &input_ptr,
        &kernel_cols,
        &first_output_ptr,
        &second_output_ptr,
        &third_output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel_2d(
            function,
            static_cast<unsigned int>(max_rows),
            3,
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for SQ FP8 matvec triple";
        }
        return false;
    }
    return true;
}

ullm_status sq_fp8_matvec_f32_hip_staging(
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    uint32_t scale_kind,
    size_t scale_block_cols,
    size_t required_payload_bytes,
    size_t required_scale_bytes,
    size_t required_input_bytes,
    size_t required_output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<uint8_t> host_payload(required_payload_bytes);
    std::vector<float> host_scales(required_scale_bytes / sizeof(float));
    std::vector<float> host_input(cols);
    std::vector<float> host_output(rows);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = payload_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_payload.data(),
            payload_buffer->ptr,
            required_payload_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_scales.data(),
            scale_buffer->ptr,
            required_scale_bytes,
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
        set_error("failed to copy SQ FP8 matvec HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize SQ FP8 matvec HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!sq_fp8_matvec_f32_host(
            host_payload.data(),
            host_scales.data(),
            host_input.data(),
            rows,
            cols,
            scale_kind,
            1,
            scale_block_cols,
            host_output.data())) {
        set_error("failed to run SQ FP8 matvec host staging path");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy SQ FP8 matvec host staging output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize SQ FP8 matvec HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status sq_fp8_matvec_batch_f32_hip_staging(
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    uint32_t scale_kind,
    size_t scale_block_cols,
    size_t batch_count,
    size_t required_payload_bytes,
    size_t required_scale_bytes,
    size_t required_input_bytes,
    size_t required_output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<uint8_t> host_payload(required_payload_bytes);
    std::vector<float> host_scales(required_scale_bytes / sizeof(float));
    std::vector<float> host_input(required_input_bytes / sizeof(float));
    std::vector<float> host_output(required_output_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = payload_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_payload.data(),
            payload_buffer->ptr,
            required_payload_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_scales.data(),
            scale_buffer->ptr,
            required_scale_bytes,
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
        set_error("failed to copy SQ FP8 matvec batch HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize SQ FP8 matvec batch HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!sq_fp8_matvec_batch_f32_host(
            host_payload.data(),
            host_scales.data(),
            host_input.data(),
            rows,
            cols,
            scale_kind,
            1,
            scale_block_cols,
            batch_count,
            host_output.data())) {
        set_error("failed to run SQ FP8 matvec batch host staging path");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy SQ FP8 matvec batch host staging output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize SQ FP8 matvec batch HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}
