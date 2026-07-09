// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "sq8_ck_gfx1201.h"

namespace {

#if defined(ULLM_RUNTIME_ROCM_CK_GFX1201)

bool sq8_ck_m_is_measured(size_t m) {
    return m == 1u || m == 2u || m == 4u || m == 8u || m == 16u || m == 32u || m == 128u;
}

bool sq8_ck_checked_mul(size_t lhs, size_t rhs, size_t *out) {
    if (out == nullptr || (lhs != 0u && rhs > std::numeric_limits<size_t>::max() / lhs)) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool sq8_ck_projection_implementation(
    size_t m,
    size_t n,
    size_t k,
    ullm_sq8_ck_implementation *implementation) {
    if (implementation == nullptr) {
        return false;
    }
    if ((n == 5120u && k == 5120u) || (n == 1024u && k == 5120u)) {
        *implementation = ULLM_SQ8_CK_MEM_V1_DEFAULT_TILE_16X128X128;
        return true;
    }
    if (n == 17408u && k == 5120u) {
        *implementation = m == 128u
                              ? ULLM_SQ8_CK_MEM_V1_DEFAULT_TILE_16X256X128
                              : ULLM_SQ8_CK_MEM_V1_KPADDING_TILE_16X128X256;
        return true;
    }
    if (n == 5120u && k == 17408u) {
        *implementation = m == 128u
                              ? ULLM_SQ8_CK_MEM_V1_DEFAULT_TILE_16X128X128
                              : ULLM_SQ8_CK_MEM_V1_DEFAULT_TILE_16X128X256;
        return true;
    }
    return false;
}

bool sq8_ck_buffers_share_backend(
    const std::array<const ullm_runtime_buffer *, 6> &buffers) {
    const ullm_runtime_buffer *first = buffers.front();
    for (const ullm_runtime_buffer *buffer : buffers) {
        if (!buffers_share_backend(first, buffer)) {
            return false;
        }
    }
    return true;
}

template <size_t Count>
bool sq8_ck_buffers_have_unique_storage(
    const std::array<const ullm_runtime_buffer *, Count> &buffers) {
    for (size_t left = 0u; left < Count; ++left) {
        for (size_t right = left + 1u; right < Count; ++right) {
            if (buffers[left]->ptr == buffers[right]->ptr) {
                return false;
            }
        }
    }
    return true;
}

#endif

} // namespace

ullm_status ullm_runtime_sq8_ck_quantize_activation_f32(
    const ullm_runtime_buffer *input_buffer,
    size_t m,
    size_t k,
    ullm_runtime_buffer *quantized_buffer,
    ullm_runtime_buffer *scale_buffer,
    ullm_runtime_stream *stream) {
    if (input_buffer == nullptr || quantized_buffer == nullptr || scale_buffer == nullptr) {
        set_error("SQ8 CK activation quantization received a null buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
#if !defined(ULLM_RUNTIME_ROCM_CK_GFX1201)
    (void)m;
    (void)k;
    (void)stream;
    set_error("SQ8 CK activation quantization requires Cargo feature rocm-ck-gfx1201");
    return ULLM_STATUS_RUNTIME_ERROR;
#else
    if (!sq8_ck_m_is_measured(m) || k == 0u || k % 128u != 0u) {
        set_error("SQ8 CK activation quantization requires measured M and K divisible by 128");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(input_buffer, quantized_buffer) ||
        !buffers_share_backend(input_buffer, scale_buffer)) {
        set_error("SQ8 CK activation quantization buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!sq8_ck_buffers_have_unique_storage(std::array<const ullm_runtime_buffer *, 3>{
            input_buffer,
            quantized_buffer,
            scale_buffer,
        })) {
        set_error("SQ8 CK activation quantization buffers must not alias");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(input_buffer, stream)) {
        set_error("SQ8 CK activation quantization stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->backend != BackendKind::Hip) {
        set_error("SQ8 CK activation quantization requires HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    size_t elements = 0u;
    size_t input_bytes = 0u;
    size_t scale_elements = 0u;
    size_t scale_bytes = 0u;
    if (!sq8_ck_checked_mul(m, k, &elements) ||
        !sq8_ck_checked_mul(elements, sizeof(float), &input_bytes) ||
        !sq8_ck_checked_mul(m, k / 128u, &scale_elements) ||
        !sq8_ck_checked_mul(scale_elements, sizeof(float), &scale_bytes)) {
        set_error("SQ8 CK activation quantization byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < input_bytes || quantized_buffer->bytes < elements ||
        scale_buffer->bytes < scale_bytes) {
        set_error("SQ8 CK activation quantization buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    std::array<char, 512> helper_error{};
    if (!ullm_sq8_ck_gfx1201_quantize_activation(
            input_buffer->ptr,
            quantized_buffer->ptr,
            scale_buffer->ptr,
            m,
            k,
            stream == nullptr ? nullptr : stream->stream,
            input_buffer->hip_device_id,
            helper_error.data(),
            helper_error.size())) {
        set_error(helper_error[0] == '\0' ? "SQ8 CK activation quantization failed"
                                           : helper_error.data());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
#endif
}

ullm_status ullm_runtime_sq8_ck_projection_f32(
    const ullm_runtime_buffer *quantized_activation_buffer,
    const ullm_runtime_buffer *activation_scale_buffer,
    const ullm_runtime_buffer *weight_buffer,
    const ullm_runtime_buffer *weight_scale_buffer,
    size_t m,
    size_t n,
    size_t k,
    ullm_runtime_buffer *workspace_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    ullm_sq8_ck_implementation *implementation) {
    if (quantized_activation_buffer == nullptr || activation_scale_buffer == nullptr ||
        weight_buffer == nullptr || weight_scale_buffer == nullptr || workspace_buffer == nullptr ||
        output_buffer == nullptr || implementation == nullptr) {
        set_error("SQ8 CK projection received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    *implementation = ULLM_SQ8_CK_IMPLEMENTATION_UNAVAILABLE;
#if !defined(ULLM_RUNTIME_ROCM_CK_GFX1201)
    (void)m;
    (void)n;
    (void)k;
    (void)stream;
    set_error("SQ8 CK projection requires Cargo feature rocm-ck-gfx1201");
    return ULLM_STATUS_RUNTIME_ERROR;
#else
    if (!sq8_ck_m_is_measured(m)) {
        set_error("SQ8 CK projection M is outside the measured dispatch set");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!sq8_ck_projection_implementation(m, n, k, implementation)) {
        set_error("SQ8 CK projection shape is outside the measured dispatch table");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (n % 128u != 0u || k % 128u != 0u) {
        set_error("SQ8 CK projection N and K must be divisible by 128");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const std::array<const ullm_runtime_buffer *, 6> buffers = {
        quantized_activation_buffer,
        activation_scale_buffer,
        weight_buffer,
        weight_scale_buffer,
        workspace_buffer,
        output_buffer,
    };
    if (!sq8_ck_buffers_share_backend(buffers)) {
        set_error("SQ8 CK projection buffers belong to different backends or devices");
        *implementation = ULLM_SQ8_CK_IMPLEMENTATION_UNAVAILABLE;
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!sq8_ck_buffers_have_unique_storage(buffers)) {
        set_error("SQ8 CK projection buffers must not alias");
        *implementation = ULLM_SQ8_CK_IMPLEMENTATION_UNAVAILABLE;
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("SQ8 CK projection stream belongs to a different backend or device");
        *implementation = ULLM_SQ8_CK_IMPLEMENTATION_UNAVAILABLE;
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->backend != BackendKind::Hip) {
        set_error("SQ8 CK projection requires HIP buffers");
        *implementation = ULLM_SQ8_CK_IMPLEMENTATION_UNAVAILABLE;
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    size_t activation_elements = 0u;
    size_t activation_scale_elements = 0u;
    size_t weight_elements = 0u;
    size_t weight_scale_elements = 0u;
    size_t output_elements = 0u;
    size_t activation_scale_bytes = 0u;
    size_t weight_scale_bytes = 0u;
    size_t workspace_bytes = 0u;
    size_t output_bytes = 0u;
    if (!sq8_ck_checked_mul(m, k, &activation_elements) ||
        !sq8_ck_checked_mul(m, k / 128u, &activation_scale_elements) ||
        !sq8_ck_checked_mul(n, k, &weight_elements) ||
        !sq8_ck_checked_mul(n / 128u, k / 128u, &weight_scale_elements) ||
        !sq8_ck_checked_mul(m, n, &output_elements) ||
        !sq8_ck_checked_mul(activation_scale_elements, sizeof(float), &activation_scale_bytes) ||
        !sq8_ck_checked_mul(weight_scale_elements, sizeof(float), &weight_scale_bytes) ||
        !sq8_ck_checked_mul(output_elements, sizeof(uint16_t), &workspace_bytes) ||
        !sq8_ck_checked_mul(output_elements, sizeof(float), &output_bytes)) {
        set_error("SQ8 CK projection byte size overflows");
        *implementation = ULLM_SQ8_CK_IMPLEMENTATION_UNAVAILABLE;
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (quantized_activation_buffer->bytes < activation_elements ||
        activation_scale_buffer->bytes < activation_scale_bytes ||
        weight_buffer->bytes < weight_elements || weight_scale_buffer->bytes < weight_scale_bytes ||
        workspace_buffer->bytes < workspace_bytes || output_buffer->bytes < output_bytes) {
        set_error("SQ8 CK projection buffer is too small");
        *implementation = ULLM_SQ8_CK_IMPLEMENTATION_UNAVAILABLE;
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    std::array<char, 512> helper_error{};
    if (!ullm_sq8_ck_gfx1201_projection(
            quantized_activation_buffer->ptr,
            activation_scale_buffer->ptr,
            weight_buffer->ptr,
            weight_scale_buffer->ptr,
            m,
            n,
            k,
            workspace_buffer->ptr,
            output_buffer->ptr,
            stream == nullptr ? nullptr : stream->stream,
            output_buffer->hip_device_id,
            static_cast<uint32_t>(*implementation),
            helper_error.data(),
            helper_error.size())) {
        set_error(helper_error[0] == '\0' ? "SQ8 CK projection failed" : helper_error.data());
        *implementation = ULLM_SQ8_CK_IMPLEMENTATION_UNAVAILABLE;
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
#endif
}
