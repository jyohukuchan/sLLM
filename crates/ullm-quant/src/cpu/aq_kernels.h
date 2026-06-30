#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

struct ullm_aq_kernel_version {
    std::uint32_t major;
    std::uint32_t minor;
    std::uint32_t patch;
};

struct ullm_aq_quant_metrics {
    std::uint64_t elements;
    std::uint64_t groups;
    double sse;
    double ref_sse;
    float max_abs_error;
    std::uint64_t index_counts[16];
    std::uint32_t scale_index_min;
    std::uint32_t scale_index_max;
    std::uint64_t scale_window_improved_groups;
};

ullm_aq_kernel_version ullm_aq_get_kernel_version();

std::size_t ullm_aq_pack_nibbles(
    const std::uint8_t * low,
    const std::uint8_t * high,
    std::uint8_t * output,
    std::size_t len);

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
    ullm_aq_quant_metrics * metrics);

}
