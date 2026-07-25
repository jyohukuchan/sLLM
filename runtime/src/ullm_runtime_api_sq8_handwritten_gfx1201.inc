// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Internal-only C ABI wrappers for the SQ8_0 gfx1201 handwritten WMMA
// feasibility prototype.  This file deliberately does not alter the public
// header, the legacy dispatcher, or the existing CK implementation.

#include "sq8_handwritten_gfx1201.h"

namespace {

#if defined(ULLM_RUNTIME_ROCM_HANDWRITTEN_PROJECTION_GFX1201)

bool sq8_handwritten_checked_mul(size_t lhs, size_t rhs, size_t* out) {
    if (out == nullptr || (lhs != 0u && rhs > std::numeric_limits<size_t>::max() / lhs)) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool sq8_handwritten_exact_m1_shape(size_t m, size_t n, size_t k) {
    return m == 1u && ((n == 5120u && k == 5120u) || (n == 1024u && k == 5120u) ||
                       (n == 17408u && k == 5120u) || (n == 5120u && k == 17408u));
}

template <size_t Count>
bool sq8_handwritten_buffers_share_backend(
    const std::array<const ullm_runtime_buffer*, Count>& buffers) {
    const ullm_runtime_buffer* first = buffers.front();
    for (const ullm_runtime_buffer* buffer : buffers) {
        if (!buffers_share_backend(first, buffer)) {
            return false;
        }
    }
    return true;
}

template <size_t Count>
bool sq8_handwritten_buffers_have_unique_storage(
    const std::array<const ullm_runtime_buffer*, Count>& buffers) {
    for (size_t left = 0u; left < Count; ++left) {
        for (size_t right = left + 1u; right < Count; ++right) {
            if (buffers[left]->ptr == buffers[right]->ptr) {
                return false;
            }
        }
    }
    return true;
}

bool sq8_handwritten_projection_sizes(
    size_t n,
    size_t k,
    size_t* activation_bytes,
    size_t* activation_scale_bytes,
    size_t* weight_bytes,
    size_t* weight_scale_bytes,
    size_t* output_bytes) {
    size_t activation_scale_elements = 0u;
    size_t weight_elements = 0u;
    size_t weight_scale_elements = 0u;
    return sq8_handwritten_checked_mul(k / 128u, sizeof(float), activation_scale_bytes) &&
           sq8_handwritten_checked_mul(n, k, &weight_elements) &&
           sq8_handwritten_checked_mul(weight_elements, sizeof(unsigned char), weight_bytes) &&
           sq8_handwritten_checked_mul(n / 128u, k / 128u, &weight_scale_elements) &&
           sq8_handwritten_checked_mul(
               weight_scale_elements, sizeof(float), weight_scale_bytes) &&
           sq8_handwritten_checked_mul(n, sizeof(float), output_bytes) &&
           sq8_handwritten_checked_mul(k, sizeof(unsigned char), activation_bytes) &&
           sq8_handwritten_checked_mul(k / 128u, 1u, &activation_scale_elements) &&
           activation_scale_elements == k / 128u;
}

#endif

} // namespace

extern "C" ullm_status ullm_runtime_sq8_handwritten_gfx1201_m1_projection_f32(
    const ullm_runtime_buffer* quantized_activation_buffer,
    const ullm_runtime_buffer* activation_scale_buffer,
    const ullm_runtime_buffer* weight_buffer,
    const ullm_runtime_buffer* weight_scale_buffer,
    size_t m,
    size_t n,
    size_t k,
    ullm_runtime_buffer* output_buffer,
    ullm_runtime_stream* stream) {
    if (quantized_activation_buffer == nullptr || activation_scale_buffer == nullptr ||
        weight_buffer == nullptr || weight_scale_buffer == nullptr || output_buffer == nullptr) {
        set_error("SQ8 handwritten WMMA projection received a null buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
#if !defined(ULLM_RUNTIME_ROCM_HANDWRITTEN_PROJECTION_GFX1201)
    (void)m;
    (void)n;
    (void)k;
    (void)stream;
    set_error(
        "SQ8 handwritten WMMA projection requires Cargo feature rocm-handwritten-projection-gfx1201");
    return ULLM_STATUS_RUNTIME_ERROR;
#else
    if (!sq8_handwritten_exact_m1_shape(m, n, k)) {
        set_error("SQ8 handwritten WMMA projection requires an exact Qwen3-14B SQ8_0 M=1 shape");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const std::array<const ullm_runtime_buffer*, 5> buffers = {
        quantized_activation_buffer,
        activation_scale_buffer,
        weight_buffer,
        weight_scale_buffer,
        output_buffer,
    };
    if (!sq8_handwritten_buffers_share_backend(buffers)) {
        set_error("SQ8 handwritten WMMA projection buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!sq8_handwritten_buffers_have_unique_storage(buffers)) {
        set_error("SQ8 handwritten WMMA projection buffers must not alias");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("SQ8 handwritten WMMA projection stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->backend != BackendKind::Hip) {
        set_error("SQ8 handwritten WMMA projection requires HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    size_t activation_bytes = 0u;
    size_t activation_scale_bytes = 0u;
    size_t weight_bytes = 0u;
    size_t weight_scale_bytes = 0u;
    size_t output_bytes = 0u;
    if (!sq8_handwritten_projection_sizes(n,
                                          k,
                                          &activation_bytes,
                                          &activation_scale_bytes,
                                          &weight_bytes,
                                          &weight_scale_bytes,
                                          &output_bytes)) {
        set_error("SQ8 handwritten WMMA projection byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (quantized_activation_buffer->bytes < activation_bytes ||
        activation_scale_buffer->bytes < activation_scale_bytes ||
        weight_buffer->bytes < weight_bytes || weight_scale_buffer->bytes < weight_scale_bytes ||
        output_buffer->bytes < output_bytes) {
        set_error("SQ8 handwritten WMMA projection buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    std::array<char, 512> helper_error{};
    if (!ullm_sq8_handwritten_gfx1201_m1_wmma_projection(
            quantized_activation_buffer->ptr,
            activation_scale_buffer->ptr,
            weight_buffer->ptr,
            weight_scale_buffer->ptr,
            n,
            k,
            output_buffer->ptr,
            stream == nullptr ? nullptr : stream->stream,
            output_buffer->hip_device_id,
            helper_error.data(),
            helper_error.size())) {
        set_error(helper_error[0] == '\0' ? "SQ8 handwritten WMMA projection failed"
                                           : helper_error.data());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
#endif
}

extern "C" ullm_status ullm_runtime_sq8_handwritten_gfx1201_m1_resources(
    const ullm_runtime_buffer* device_anchor,
    uint32_t* vgpr_per_thread,
    size_t* static_lds_bytes,
    size_t* local_bytes_per_thread,
    int* threads_per_block,
    int* active_blocks_per_cu) {
    if (device_anchor == nullptr || vgpr_per_thread == nullptr || static_lds_bytes == nullptr ||
        local_bytes_per_thread == nullptr || threads_per_block == nullptr ||
        active_blocks_per_cu == nullptr) {
        set_error("SQ8 handwritten WMMA resource query received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
#if !defined(ULLM_RUNTIME_ROCM_HANDWRITTEN_PROJECTION_GFX1201)
    set_error(
        "SQ8 handwritten WMMA resource query requires Cargo feature rocm-handwritten-projection-gfx1201");
    return ULLM_STATUS_RUNTIME_ERROR;
#else
    if (device_anchor->backend != BackendKind::Hip) {
        set_error("SQ8 handwritten WMMA resource query requires a HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    std::array<char, 512> helper_error{};
    if (!ullm_sq8_handwritten_gfx1201_m1_wmma_resources(device_anchor->hip_device_id,
                                                         vgpr_per_thread,
                                                         static_lds_bytes,
                                                         local_bytes_per_thread,
                                                         threads_per_block,
                                                         active_blocks_per_cu,
                                                         helper_error.data(),
                                                         helper_error.size())) {
        set_error(helper_error[0] == '\0' ? "SQ8 handwritten WMMA resource query failed"
                                           : helper_error.data());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
#endif
}
