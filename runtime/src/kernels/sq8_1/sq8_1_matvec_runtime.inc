// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

constexpr size_t kSq8_1GroupSize = 32u;
constexpr size_t kSq8_1PayloadAlignmentBytes = 16u;
constexpr unsigned int kSq8_1MatvecThreads = 256u;
constexpr unsigned int kSq8_1MatvecRowsPerBlock = 8u;
// Keep the tiled W8A8 path below half the conservative per-workgroup LDS
// budget.  The fallback kernel remains available for arbitrary valid shapes.
constexpr size_t kSq8_1W8A8TiledSharedLimitBytes = 48u * 1024u;

size_t sq8_1_groups_per_row(size_t cols) {
    return cols == 0u ? 0u : 1u + (cols - 1u) / kSq8_1GroupSize;
}

bool sq8_1_payload_row_stride(size_t cols, size_t *stride) {
    if (cols == 0u || stride == nullptr ||
        cols > std::numeric_limits<size_t>::max() - (kSq8_1PayloadAlignmentBytes - 1u)) {
        return false;
    }
    *stride = ((cols + kSq8_1PayloadAlignmentBytes - 1u) / kSq8_1PayloadAlignmentBytes) *
        kSq8_1PayloadAlignmentBytes;
    return true;
}

bool sq8_1_w8a8_tiled_shared_bytes(size_t cols, size_t *bytes) {
    if (bytes == nullptr) {
        return false;
    }
    size_t activation_code_bytes = 0u;
    if (!sq8_1_payload_row_stride(cols, &activation_code_bytes)) {
        return false;
    }
    const size_t groups = sq8_1_groups_per_row(cols);
    if (groups == 0u || groups >
            (std::numeric_limits<size_t>::max() - activation_code_bytes) / sizeof(float)) {
        return false;
    }
    *bytes = activation_code_bytes + groups * sizeof(float);
    return true;
}

float sq8_1_f16_to_f32_host(uint16_t bits) {
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

uint32_t sq8_1_round_shift_ties_even(uint32_t value, unsigned int shift) {
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

uint16_t sq8_1_f32_to_f16_rne_bits(float value) {
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
        uint32_t rounded = sq8_1_round_shift_ties_even(mantissa, 13u);
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
        sign | sq8_1_round_shift_ties_even(significand, static_cast<unsigned int>(-exponent - 1)));
}

bool sq8_1_ceil_f16_host(float value, float *stored) {
    if (stored == nullptr || !std::isfinite(value) || value <= 0.0f) {
        return false;
    }
    uint16_t bits = sq8_1_f32_to_f16_rne_bits(value);
    if (bits == 0u) {
        *stored = sq8_1_f16_to_f32_host(1u);
        return true;
    }
    if (bits >= 0x7c00u) {
        return false;
    }
    float scale = sq8_1_f16_to_f32_host(bits);
    if (scale < value) {
        ++bits;
        if (bits >= 0x7c00u) {
            return false;
        }
        scale = sq8_1_f16_to_f32_host(bits);
    }
    if (!std::isfinite(scale) || scale <= 0.0f || scale < value) {
        return false;
    }
    *stored = scale;
    return true;
}

int sq8_1_round_ties_even_to_i32(float value) {
    const float lower = std::floor(value);
    const float fraction = value - lower;
    int result = static_cast<int>(lower);
    if (fraction > 0.5f || (fraction == 0.5f && (result & 1) != 0)) {
        ++result;
    }
    return result;
}

bool sq8_1_quantize_activation_block_host(
    const float *input,
    size_t count,
    int8_t *codes,
    float *scale) {
    if (input == nullptr || codes == nullptr || scale == nullptr || count == 0u || count > kSq8_1GroupSize) {
        return false;
    }
    float maximum = 0.0f;
    for (size_t index = 0u; index < count; ++index) {
        if (!std::isfinite(input[index])) {
            return false;
        }
        maximum = std::max(maximum, std::fabs(input[index]));
    }
    if (maximum == 0.0f) {
        *scale = 1.0f;
        std::fill(codes, codes + count, int8_t{0});
        return true;
    }
    if (!sq8_1_ceil_f16_host(maximum / 127.0f, scale)) {
        return false;
    }
    for (size_t index = 0u; index < count; ++index) {
        const int rounded = sq8_1_round_ties_even_to_i32(input[index] / *scale);
        const int clamped = std::max(-127, std::min(127, rounded));
        codes[index] = static_cast<int8_t>(clamped);
    }
    return true;
}

bool sq8_1_matvec_w8a16_f32_host(
    const uint8_t *payload,
    const uint16_t *weight_scales,
    const float *input,
    size_t rows,
    size_t cols,
    size_t payload_row_stride,
    float *output) {
    if (payload == nullptr || weight_scales == nullptr || input == nullptr || output == nullptr ||
        rows == 0u || cols == 0u) {
        return false;
    }
    size_t expected_stride = 0u;
    if (!sq8_1_payload_row_stride(cols, &expected_stride) || payload_row_stride != expected_stride) {
        return false;
    }
    const size_t groups = sq8_1_groups_per_row(cols);
    for (size_t row = 0u; row < rows; ++row) {
        const uint8_t *row_payload = payload + row * payload_row_stride;
        if (std::any_of(row_payload + cols, row_payload + payload_row_stride, [](uint8_t byte) {
                return byte != 0u;
            })) {
            return false;
        }
        float total = 0.0f;
        for (size_t group = 0u; group < groups; ++group) {
            const size_t start = group * kSq8_1GroupSize;
            const size_t stop = std::min(start + kSq8_1GroupSize, cols);
            const float scale = sq8_1_f16_to_f32_host(weight_scales[row * groups + group]);
            if (!std::isfinite(scale) || scale <= 0.0f) {
                return false;
            }
            float partial = 0.0f;
            for (size_t col = start; col < stop; ++col) {
                if (row_payload[col] == 0x80u || !std::isfinite(input[col])) {
                    return false;
                }
                partial += static_cast<float>(static_cast<int8_t>(row_payload[col])) * input[col];
            }
            total += partial * scale;
        }
        output[row] = total;
    }
    return true;
}

bool sq8_1_matvec_w8a8_explicit_f32_host(
    const uint8_t *payload,
    const uint16_t *weight_scales,
    const float *input,
    size_t rows,
    size_t cols,
    size_t payload_row_stride,
    float *output) {
    if (payload == nullptr || weight_scales == nullptr || input == nullptr || output == nullptr ||
        rows == 0u || cols == 0u) {
        return false;
    }
    size_t expected_stride = 0u;
    if (!sq8_1_payload_row_stride(cols, &expected_stride) || payload_row_stride != expected_stride) {
        return false;
    }
    const size_t groups = sq8_1_groups_per_row(cols);
    std::vector<int8_t> activation_codes(cols);
    std::vector<float> activation_scales(groups);
    for (size_t group = 0u; group < groups; ++group) {
        const size_t start = group * kSq8_1GroupSize;
        const size_t count = std::min(kSq8_1GroupSize, cols - start);
        if (!sq8_1_quantize_activation_block_host(
                input + start,
                count,
                activation_codes.data() + start,
                activation_scales.data() + group)) {
            return false;
        }
    }
    for (size_t row = 0u; row < rows; ++row) {
        const uint8_t *row_payload = payload + row * payload_row_stride;
        if (std::any_of(row_payload + cols, row_payload + payload_row_stride, [](uint8_t byte) {
                return byte != 0u;
            })) {
            return false;
        }
        float total = 0.0f;
        for (size_t group = 0u; group < groups; ++group) {
            const size_t start = group * kSq8_1GroupSize;
            const size_t stop = std::min(start + kSq8_1GroupSize, cols);
            const float weight_scale = sq8_1_f16_to_f32_host(weight_scales[row * groups + group]);
            if (!std::isfinite(weight_scale) || weight_scale <= 0.0f) {
                return false;
            }
            int32_t dot = 0;
            for (size_t col = start; col < stop; ++col) {
                if (row_payload[col] == 0x80u) {
                    return false;
                }
                dot += static_cast<int32_t>(static_cast<int8_t>(row_payload[col])) *
                    static_cast<int32_t>(activation_codes[col]);
            }
            total += static_cast<float>(dot) * weight_scale * activation_scales[group];
        }
        output[row] = total;
    }
    return true;
}

std::string sq8_1_exact_dot_arch_for_device(int device_id) {
    std::string arch = hip_runtime().device_arch_name(device_id);
    const size_t feature_separator = arch.find(':');
    if (feature_separator != std::string::npos) {
        arch.resize(feature_separator);
    }
    if (arch == "gfx1030" || arch == "gfx942" || arch == "gfx950" || arch == "gfx1100" ||
        arch == "gfx1201") {
        return arch;
    }
    return "";
}

class HipSq8_1MatvecKernelCache {
public:
    void *function_for_device(int device_id, bool w8a8, bool w8a8_tiled, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            if (!w8a8) {
                return found->second->w8a16_function;
            }
            return w8a8_tiled ? found->second->w8a8_tiled_function
                              : found->second->w8a8_fallback_function;
        }
        const std::string arch = sq8_1_exact_dot_arch_for_device(device_id);
        if (arch.empty()) {
            append_error(error, "SQ8_1 signed-dot HIP kernel has no exact supported architecture");
            return nullptr;
        }
        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_sq8_1_matvec_kernel(arch, &code, &compile_error)) {
            append_error(error, compile_error.empty() ? "SQ8_1 HIPRTC compilation failed" : compile_error);
            return nullptr;
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            append_error(error, "hipModuleLoadData failed for SQ8_1 " + arch);
            return nullptr;
        }
        void *w8a16_function = nullptr;
        void *w8a8_tiled_function = nullptr;
        void *w8a8_fallback_function = nullptr;
        if (!hip_runtime().module_get_function(
                &w8a16_function, module, "ullm_sq8_1_matvec_w8a16_f32_kernel", device_id) ||
            !hip_runtime().module_get_function(
                &w8a8_tiled_function, module, "ullm_sq8_1_matvec_w8a8_explicit_f32_kernel", device_id) ||
            !hip_runtime().module_get_function(
                &w8a8_fallback_function,
                module,
                "ullm_sq8_1_matvec_w8a8_explicit_fallback_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            append_error(error, "hipModuleGetFunction failed for SQ8_1 " + arch);
            return nullptr;
        }
        auto loaded = std::make_unique<LoadedModule>();
        loaded->module = module;
        loaded->w8a16_function = w8a16_function;
        loaded->w8a8_tiled_function = w8a8_tiled_function;
        loaded->w8a8_fallback_function = w8a8_fallback_function;
        void *result = !w8a8 ? loaded->w8a16_function
                              : (w8a8_tiled ? loaded->w8a8_tiled_function
                                             : loaded->w8a8_fallback_function);
        modules_.emplace(device_id, std::move(loaded));
        return result;
    }

private:
    struct LoadedModule {
        void *module = nullptr;
        void *w8a16_function = nullptr;
        void *w8a8_tiled_function = nullptr;
        void *w8a8_fallback_function = nullptr;
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

HipSq8_1MatvecKernelCache &hip_sq8_1_matvec_kernel_cache() {
    static HipSq8_1MatvecKernelCache cache;
    return cache;
}

bool sq8_1_matvec_f32_hip_kernel(
    bool w8a8,
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    size_t payload_row_stride,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    if ((reinterpret_cast<uintptr_t>(payload_buffer->ptr) & (kSq8_1PayloadAlignmentBytes - 1u)) != 0u) {
        if (error != nullptr) {
            *error = "SQ8_1 HIP payload address is not 16-byte aligned";
        }
        return false;
    }
    if (rows > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "SQ8_1 row count exceeds HIP grid limit";
        }
        return false;
    }
    const int device_id = payload_buffer->hip_device_id;
    size_t w8a8_shared_bytes = 0u;
    const bool use_w8a8_tiled = w8a8 && sq8_1_w8a8_tiled_shared_bytes(cols, &w8a8_shared_bytes) &&
        w8a8_shared_bytes <= kSq8_1W8A8TiledSharedLimitBytes;
    void *function = hip_sq8_1_matvec_kernel_cache().function_for_device(
        device_id, w8a8, use_w8a8_tiled, error);
    if (function == nullptr) {
        return false;
    }
    void *payload_ptr = payload_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_stride = static_cast<unsigned long long>(payload_row_stride);
    void *kernel_params[] = {
        &payload_ptr,
        &scale_ptr,
        &input_ptr,
        &kernel_rows,
        &kernel_cols,
        &kernel_stride,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const size_t rows_per_block = (w8a8 && !use_w8a8_tiled)
        ? 1u
        : static_cast<size_t>(kSq8_1MatvecRowsPerBlock);
    const unsigned int grid_x = static_cast<unsigned int>(
        1u + (rows - 1u) / rows_per_block);
    if (!hip_runtime().module_launch_kernel(
        function,
        grid_x,
        kSq8_1MatvecThreads,
        kernel_params,
        hip_stream,
        device_id,
        use_w8a8_tiled ? static_cast<unsigned int>(w8a8_shared_bytes) : 0u)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for SQ8_1 matvec";
        }
        return false;
    }
    return true;
}
