#include "aq_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

extern "C" {

ullm_aq_kernel_version ullm_aq_get_kernel_version() {
    return ullm_aq_kernel_version{0, 1, 0};
}

std::size_t ullm_aq_pack_nibbles(
    const std::uint8_t * low,
    const std::uint8_t * high,
    std::uint8_t * output,
    std::size_t len) {
    if (low == nullptr || high == nullptr || output == nullptr) {
        return 0;
    }

    for (std::size_t i = 0; i < len; ++i) {
        const std::uint8_t lo = low[i] & 0x0f;
        const std::uint8_t hi = (high[i] & 0x0f) << 4;
        output[i] = static_cast<std::uint8_t>(lo | hi);
    }

    return len;
}

namespace {

float decode_bf16(const std::uint8_t * input) {
    const std::uint16_t raw =
        static_cast<std::uint16_t>(input[0]) |
        static_cast<std::uint16_t>(static_cast<std::uint16_t>(input[1]) << 8);
    const std::uint32_t bits = static_cast<std::uint32_t>(raw) << 16;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float decode_f16(const std::uint8_t * input) {
    const std::uint16_t raw =
        static_cast<std::uint16_t>(input[0]) |
        static_cast<std::uint16_t>(static_cast<std::uint16_t>(input[1]) << 8);
    const std::uint32_t sign = static_cast<std::uint32_t>(raw & 0x8000U) << 16;
    const std::uint16_t exp = static_cast<std::uint16_t>((raw >> 10) & 0x1fU);
    const std::uint16_t frac = static_cast<std::uint16_t>(raw & 0x03ffU);
    std::uint32_t bits = 0;
    if (exp == 0) {
        if (frac == 0) {
            bits = sign;
        } else {
            std::uint16_t frac_norm = frac;
            int exp_unbiased = -14;
            while ((frac_norm & 0x0400U) == 0) {
                frac_norm = static_cast<std::uint16_t>(frac_norm << 1);
                exp_unbiased -= 1;
            }
            frac_norm = static_cast<std::uint16_t>(frac_norm & 0x03ffU);
            bits = sign |
                (static_cast<std::uint32_t>(exp_unbiased + 127) << 23) |
                (static_cast<std::uint32_t>(frac_norm) << 13);
        }
    } else if (exp == 0x1fU) {
        bits = sign | 0x7f800000U | (static_cast<std::uint32_t>(frac) << 13);
    } else {
        const int exp_f32 = static_cast<int>(exp) - 15 + 127;
        bits = sign |
            (static_cast<std::uint32_t>(exp_f32) << 23) |
            (static_cast<std::uint32_t>(frac) << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float decode_16bit(const std::uint8_t * input, std::uint32_t dtype) {
    if (dtype == ULLM_AQ_DTYPE_F16) {
        return decode_f16(input);
    }
    return decode_bf16(input);
}

std::size_t nearest_scale_index(float target, const float * scales, std::size_t scale_count) {
    if (target <= scales[0]) {
        return 0;
    }
    const std::size_t last = scale_count - 1;
    if (target >= scales[last]) {
        return last;
    }
    const float * end = scales + scale_count;
    const float * it = std::lower_bound(scales, end, target);
    const std::size_t idx = static_cast<std::size_t>(it - scales);
    const std::size_t prev = idx - 1;
    if (std::fabs(target - scales[prev]) < std::fabs(target - scales[idx])) {
        return prev;
    }
    return idx;
}

std::size_t nearest_codebook_index(float value, const float * codebook, std::size_t codebook_count) {
    std::size_t best_index = 0;
    float best_error = std::numeric_limits<float>::infinity();
    for (std::size_t index = 0; index < codebook_count; ++index) {
        const float error = std::fabs(value - codebook[index]);
        if (error < best_error) {
            best_error = error;
            best_index = index;
        }
    }
    return best_index;
}

float max_codebook_abs(const float * codebook, std::size_t codebook_count) {
    float max_code = 0.0f;
    for (std::size_t index = 0; index < codebook_count; ++index) {
        max_code = std::max(max_code, std::fabs(codebook[index]));
    }
    return std::max(max_code, 1.0e-12f);
}

void reset_metrics(ullm_aq_quant_metrics * metrics, std::size_t scale_count) {
    metrics->elements = 0;
    metrics->groups = 0;
    metrics->sse = 0.0;
    metrics->ref_sse = 0.0;
    metrics->max_abs_error = 0.0f;
    for (std::uint64_t & count : metrics->index_counts) {
        count = 0;
    }
    metrics->scale_index_min = static_cast<std::uint32_t>(scale_count);
    metrics->scale_index_max = 0;
    metrics->scale_window_improved_groups = 0;
}

int quantize_16bit_chunk_impl(
    std::uint32_t dtype,
    const std::uint8_t * input,
    std::size_t input_bytes,
    std::size_t group_size,
    const float * scale_values,
    std::size_t scale_count,
    const float * codebook,
    std::size_t codebook_count,
    float tensor_scale,
    std::size_t scale_window,
    std::uint8_t * packed_indices,
    std::size_t packed_indices_bytes,
    std::uint8_t * scale_indices,
    std::size_t scale_indices_bytes,
    ullm_aq_quant_metrics * metrics) {
    if (input == nullptr || scale_values == nullptr || codebook == nullptr ||
        packed_indices == nullptr || scale_indices == nullptr || metrics == nullptr) {
        return -1;
    }
    if (group_size == 0 || scale_count == 0 || scale_count > 256 ||
        codebook_count != 16 ||
        !std::isfinite(tensor_scale) || tensor_scale <= 0.0f) {
        return -2;
    }
    constexpr std::size_t element_size = 2;
    const std::size_t group_bytes = group_size * element_size;
    if (group_bytes == 0 || input_bytes % group_bytes != 0) {
        return -3;
    }
    const std::size_t elements = input_bytes / element_size;
    const std::size_t groups = input_bytes / group_bytes;
    const std::size_t required_index_bytes = (elements + 1) / 2;
    if (packed_indices_bytes < required_index_bytes || scale_indices_bytes < groups) {
        return -4;
    }

    reset_metrics(metrics, scale_count);
    const float max_code = max_codebook_abs(codebook, codebook_count);

    for (std::size_t group_index = 0; group_index < groups; ++group_index) {
        const std::uint8_t * group = input + group_index * group_bytes;
        float absmax = 0.0f;
        for (std::size_t item = 0; item < group_size; ++item) {
            const float value = decode_16bit(group + item * element_size, dtype);
            if (!std::isnan(value)) {
                absmax = std::max(absmax, std::fabs(value));
            }
        }

        const float scale_target = absmax / tensor_scale / max_code;
        const std::size_t center = nearest_scale_index(scale_target, scale_values, scale_count);
        const std::size_t start = center > scale_window ? center - scale_window : 0;
        const std::size_t max_right = scale_count - 1 - center;
        const std::size_t end = center + std::min(scale_window, max_right);
        std::size_t best_scale_index = center;
        double best_group_sse = std::numeric_limits<double>::infinity();

        for (std::size_t scale_index = start; scale_index <= end; ++scale_index) {
            const float combined_scale = scale_values[scale_index] * tensor_scale;
            double group_sse = 0.0;
            for (std::size_t item = 0; item < group_size; ++item) {
                const float value = decode_16bit(group + item * element_size, dtype);
                if (std::isnan(value)) {
                    continue;
                }
                const float normalized = value / combined_scale;
                const std::size_t codebook_index =
                    nearest_codebook_index(normalized, codebook, codebook_count);
                const float recon = codebook[codebook_index] * combined_scale;
                const float error = value - recon;
                group_sse += static_cast<double>(error * error);
            }
            if (group_sse < best_group_sse) {
                best_group_sse = group_sse;
                best_scale_index = scale_index;
            }
        }

        if (best_scale_index != center) {
            metrics->scale_window_improved_groups += 1;
        }
        metrics->scale_index_min =
            std::min(metrics->scale_index_min, static_cast<std::uint32_t>(best_scale_index));
        metrics->scale_index_max =
            std::max(metrics->scale_index_max, static_cast<std::uint32_t>(best_scale_index));
        scale_indices[group_index] = static_cast<std::uint8_t>(best_scale_index);

        const float combined_scale = scale_values[best_scale_index] * tensor_scale;
        for (std::size_t item = 0; item < group_size; ++item) {
            const std::size_t element_index = group_index * group_size + item;
            const float value = decode_16bit(group + item * element_size, dtype);
            std::size_t codebook_index = 0;
            if (!std::isnan(value)) {
                const float normalized = value / combined_scale;
                codebook_index = nearest_codebook_index(normalized, codebook, codebook_count);
                const float recon = codebook[codebook_index] * combined_scale;
                const float error = value - recon;
                metrics->elements += 1;
                metrics->sse += static_cast<double>(error * error);
                metrics->ref_sse += static_cast<double>(value * value);
                metrics->max_abs_error = std::max(metrics->max_abs_error, std::fabs(error));
                metrics->index_counts[codebook_index] += 1;
            }
            const std::uint8_t nibble = static_cast<std::uint8_t>(codebook_index & 0x0f);
            std::uint8_t & output = packed_indices[element_index / 2];
            if ((element_index & 1U) == 0) {
                output = nibble;
            } else {
                output = static_cast<std::uint8_t>(output | static_cast<std::uint8_t>(nibble << 4));
            }
        }
        metrics->groups += 1;
    }

    return 0;
}

} // namespace

int ullm_aq_quantize_chunk_v1(
    const ullm_aq_quantize_chunk_request_v1 * request,
    ullm_aq_quant_metrics * metrics,
    std::size_t metrics_size) {
    if (request == nullptr || metrics == nullptr) {
        return -1;
    }
    if (request->struct_size < sizeof(ullm_aq_quantize_chunk_request_v1) ||
        metrics_size < sizeof(ullm_aq_quant_metrics)) {
        return -2;
    }
    if (request->dtype != ULLM_AQ_DTYPE_BF16 && request->dtype != ULLM_AQ_DTYPE_F16) {
        return -5;
    }
    return quantize_16bit_chunk_impl(
        request->dtype,
        request->input,
        request->input_bytes,
        request->group_size,
        request->scale_values,
        request->scale_count,
        request->codebook,
        request->codebook_count,
        request->tensor_scale,
        request->scale_window,
        request->packed_indices,
        request->packed_indices_bytes,
        request->scale_indices,
        request->scale_indices_bytes,
        metrics);
}

int ullm_aq_quantize_bf16_chunk(
    const std::uint8_t * input,
    std::size_t input_bytes,
    std::size_t group_size,
    const float * scale_values,
    std::size_t scale_count,
    const float * codebook,
    std::size_t codebook_count,
    float tensor_scale,
    std::size_t scale_window,
    std::uint8_t * packed_indices,
    std::size_t packed_indices_bytes,
    std::uint8_t * scale_indices,
    std::size_t scale_indices_bytes,
    ullm_aq_quant_metrics * metrics) {
    return quantize_16bit_chunk_impl(
        ULLM_AQ_DTYPE_BF16,
        input,
        input_bytes,
        group_size,
        scale_values,
        scale_count,
        codebook,
        codebook_count,
        tensor_scale,
        scale_window,
        packed_indices,
        packed_indices_bytes,
        scale_indices,
        scale_indices_bytes,
        metrics);
}

}
