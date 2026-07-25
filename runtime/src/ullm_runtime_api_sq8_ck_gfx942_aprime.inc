// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Internal-only C ABI wrappers for the isolated SQ8_0 gfx942 prototype.  They
// are deliberately absent from runtime/include/ullm_runtime.h so the existing
// gfx1201 public ABI and dispatch remain unchanged.

#include "sq8_ck_gfx942_aprime.h"
#include "sq8_ck_gfx942_arch.h"
#include "sq8_ck_gfx942_control.h"

namespace {

#if defined(ULLM_RUNTIME_ROCM_CK_GFX942_APRIME)

bool sq8_gfx942_aprime_m_is_measured(size_t m) {
    return m == 1u || m == 2u || m == 4u || m == 8u || m == 16u || m == 32u || m == 128u;
}

bool sq8_gfx942_aprime_checked_mul(size_t lhs, size_t rhs, size_t* out) {
    if (out == nullptr || (lhs != 0u && rhs > std::numeric_limits<size_t>::max() / lhs)) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

// This is an A′-private copy of the measured model table.  It must not call or
// modify the gfx1201 dispatcher, whose buffers have OCP semantics.
bool sq8_gfx942_aprime_projection_implementation(
    size_t m,
    size_t n,
    size_t k,
    uint32_t* implementation) {
    if (implementation == nullptr) {
        return false;
    }
    if ((n == 5120u && k == 5120u) || (n == 1024u && k == 5120u)) {
        *implementation = 1u; // 16x128x128 Default
        return true;
    }
    if (n == 17408u && k == 5120u) {
        *implementation = m == 128u ? 3u : 2u; // 16x256x128 / 16x128x256 KPadding
        return true;
    }
    if (n == 5120u && k == 17408u) {
        *implementation = m == 128u ? 1u : 4u; // 16x128x128 / 16x128x256 Default
        return true;
    }
    return false;
}

template <size_t Count>
bool sq8_gfx942_aprime_buffers_share_backend(
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
bool sq8_gfx942_aprime_buffers_have_unique_storage(
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

bool sq8_gfx942_aprime_projection_sizes(
    size_t m,
    size_t n,
    size_t k,
    size_t* activation_elements,
    size_t* activation_scale_bytes,
    size_t* weight_elements,
    size_t* weight_scale_bytes,
    size_t* workspace_bytes,
    size_t* output_bytes) {
    size_t activation_scale_elements = 0u;
    size_t weight_scale_elements = 0u;
    size_t output_elements = 0u;
    return sq8_gfx942_aprime_checked_mul(m, k, activation_elements) &&
           sq8_gfx942_aprime_checked_mul(m, k / 128u, &activation_scale_elements) &&
           sq8_gfx942_aprime_checked_mul(n, k, weight_elements) &&
           sq8_gfx942_aprime_checked_mul(n / 128u, k / 128u, &weight_scale_elements) &&
           sq8_gfx942_aprime_checked_mul(m, n, &output_elements) &&
           sq8_gfx942_aprime_checked_mul(
               activation_scale_elements, sizeof(float), activation_scale_bytes) &&
           sq8_gfx942_aprime_checked_mul(
               weight_scale_elements, sizeof(float), weight_scale_bytes) &&
           sq8_gfx942_aprime_checked_mul(output_elements, sizeof(uint16_t), workspace_bytes) &&
           sq8_gfx942_aprime_checked_mul(output_elements, sizeof(float), output_bytes);
}

#endif

} // namespace

extern "C" int ullm_runtime_sq8_ck_gfx942_aprime_arch_name_is_exact(const char* gcn_arch_name) {
    return ullm::sq8_ck_gfx942::is_exact_gfx942_gcn_arch_name(gcn_arch_name) ? 1 : 0;
}

extern "C" ullm_status ullm_runtime_sq8_ck_gfx942_aprime_projection_fnuz_prepacked_f32(
    const ullm_runtime_buffer* activation_fnuz_prepacked_buffer,
    const ullm_runtime_buffer* activation_scale_f32_x2_buffer,
    const ullm_runtime_buffer* weight_fnuz_prepacked_buffer,
    const ullm_runtime_buffer* weight_scale_f32_x2_buffer,
    size_t m,
    size_t n,
    size_t k,
    ullm_runtime_buffer* workspace_bf16_buffer,
    ullm_runtime_buffer* output_f32_buffer,
    ullm_runtime_stream* stream,
    uint32_t* implementation) {
    if (activation_fnuz_prepacked_buffer == nullptr || activation_scale_f32_x2_buffer == nullptr ||
        weight_fnuz_prepacked_buffer == nullptr || weight_scale_f32_x2_buffer == nullptr ||
        workspace_bf16_buffer == nullptr || output_f32_buffer == nullptr || implementation == nullptr) {
        set_error("SQ8_0 gfx942 A′ projection received a null buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    *implementation = 0u;
#if !defined(ULLM_RUNTIME_ROCM_CK_GFX942_APRIME)
    (void)m;
    (void)n;
    (void)k;
    (void)stream;
    set_error("SQ8_0 gfx942 A′ projection requires Cargo feature rocm-ck-gfx942-aprime");
    return ULLM_STATUS_RUNTIME_ERROR;
#else
    if (!sq8_gfx942_aprime_m_is_measured(m) || n == 0u || k == 0u || n % 128u != 0u ||
        k % 128u != 0u) {
        set_error("SQ8_0 gfx942 A′ projection requires measured M and N/K divisible by 128");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!sq8_gfx942_aprime_projection_implementation(m, n, k, implementation)) {
        set_error("SQ8_0 gfx942 A′ projection shape is outside its isolated measured dispatch table");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const std::array<const ullm_runtime_buffer*, 6> buffers = {
        activation_fnuz_prepacked_buffer,
        activation_scale_f32_x2_buffer,
        weight_fnuz_prepacked_buffer,
        weight_scale_f32_x2_buffer,
        workspace_bf16_buffer,
        output_f32_buffer,
    };
    if (!sq8_gfx942_aprime_buffers_share_backend(buffers)) {
        set_error("SQ8_0 gfx942 A′ projection buffers belong to different backends or devices");
        *implementation = 0u;
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!sq8_gfx942_aprime_buffers_have_unique_storage(buffers)) {
        set_error("SQ8_0 gfx942 A′ projection buffers must not alias");
        *implementation = 0u;
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_f32_buffer, stream)) {
        set_error("SQ8_0 gfx942 A′ projection stream belongs to a different backend or device");
        *implementation = 0u;
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_f32_buffer->backend != BackendKind::Hip) {
        set_error("SQ8_0 gfx942 A′ projection requires HIP buffers");
        *implementation = 0u;
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    size_t activation_elements = 0u;
    size_t activation_scale_bytes = 0u;
    size_t weight_elements = 0u;
    size_t weight_scale_bytes = 0u;
    size_t workspace_bytes = 0u;
    size_t output_bytes = 0u;
    if (!sq8_gfx942_aprime_projection_sizes(m,
                                             n,
                                             k,
                                             &activation_elements,
                                             &activation_scale_bytes,
                                             &weight_elements,
                                             &weight_scale_bytes,
                                             &workspace_bytes,
                                             &output_bytes)) {
        set_error("SQ8_0 gfx942 A′ projection byte size overflows");
        *implementation = 0u;
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (activation_fnuz_prepacked_buffer->bytes < activation_elements ||
        activation_scale_f32_x2_buffer->bytes < activation_scale_bytes ||
        weight_fnuz_prepacked_buffer->bytes < weight_elements ||
        weight_scale_f32_x2_buffer->bytes < weight_scale_bytes ||
        workspace_bf16_buffer->bytes < workspace_bytes || output_f32_buffer->bytes < output_bytes) {
        set_error("SQ8_0 gfx942 A′ projection buffer is too small");
        *implementation = 0u;
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    std::array<char, 512> helper_error{};
    if (!ullm_sq8_ck_gfx942_aprime_projection_fnuz_prepacked(
            activation_fnuz_prepacked_buffer->ptr,
            activation_scale_f32_x2_buffer->ptr,
            weight_fnuz_prepacked_buffer->ptr,
            weight_scale_f32_x2_buffer->ptr,
            m,
            n,
            k,
            workspace_bf16_buffer->ptr,
            output_f32_buffer->ptr,
            stream == nullptr ? nullptr : stream->stream,
            output_f32_buffer->hip_device_id,
            *implementation,
            helper_error.data(),
            helper_error.size())) {
        set_error(helper_error[0] == '\0' ? "SQ8_0 gfx942 A′ projection failed"
                                           : helper_error.data());
        *implementation = 0u;
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
#endif
}

extern "C" ullm_status ullm_runtime_sq8_ck_gfx942_control_dequant_ocp_bf16_projection_f32(
    const ullm_runtime_buffer* activation_ocp_buffer,
    const ullm_runtime_buffer* activation_scale_f32_buffer,
    const ullm_runtime_buffer* weight_ocp_buffer,
    const ullm_runtime_buffer* weight_scale_f32_buffer,
    size_t m,
    size_t n,
    size_t k,
    ullm_runtime_buffer* activation_bf16_buffer,
    ullm_runtime_buffer* weight_bf16_buffer,
    ullm_runtime_buffer* output_f32_buffer,
    ullm_runtime_stream* stream) {
    if (activation_ocp_buffer == nullptr || activation_scale_f32_buffer == nullptr ||
        weight_ocp_buffer == nullptr || weight_scale_f32_buffer == nullptr ||
        activation_bf16_buffer == nullptr || weight_bf16_buffer == nullptr ||
        output_f32_buffer == nullptr) {
        set_error("SQ8_0 gfx942 B control received a null buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
#if !defined(ULLM_RUNTIME_ROCM_CK_GFX942_APRIME)
    (void)m;
    (void)n;
    (void)k;
    (void)stream;
    set_error("SQ8_0 gfx942 B control requires Cargo feature rocm-ck-gfx942-aprime");
    return ULLM_STATUS_RUNTIME_ERROR;
#else
    if (!sq8_gfx942_aprime_m_is_measured(m) || n == 0u || k == 0u || n % 128u != 0u ||
        k % 128u != 0u) {
        set_error("SQ8_0 gfx942 B control requires measured M and N/K divisible by 128");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const std::array<const ullm_runtime_buffer*, 7> buffers = {
        activation_ocp_buffer,
        activation_scale_f32_buffer,
        weight_ocp_buffer,
        weight_scale_f32_buffer,
        activation_bf16_buffer,
        weight_bf16_buffer,
        output_f32_buffer,
    };
    if (!sq8_gfx942_aprime_buffers_share_backend(buffers)) {
        set_error("SQ8_0 gfx942 B control buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!sq8_gfx942_aprime_buffers_have_unique_storage(buffers)) {
        set_error("SQ8_0 gfx942 B control buffers must not alias");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_f32_buffer, stream)) {
        set_error("SQ8_0 gfx942 B control stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_f32_buffer->backend != BackendKind::Hip) {
        set_error("SQ8_0 gfx942 B control requires HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    size_t activation_elements = 0u;
    size_t activation_scale_bytes = 0u;
    size_t weight_elements = 0u;
    size_t weight_scale_bytes = 0u;
    size_t ignored_workspace_bytes = 0u;
    size_t output_bytes = 0u;
    size_t activation_bf16_bytes = 0u;
    size_t weight_bf16_bytes = 0u;
    if (!sq8_gfx942_aprime_projection_sizes(m,
                                             n,
                                             k,
                                             &activation_elements,
                                             &activation_scale_bytes,
                                             &weight_elements,
                                             &weight_scale_bytes,
                                             &ignored_workspace_bytes,
                                             &output_bytes) ||
        !sq8_gfx942_aprime_checked_mul(
            activation_elements, sizeof(uint16_t), &activation_bf16_bytes) ||
        !sq8_gfx942_aprime_checked_mul(weight_elements, sizeof(uint16_t), &weight_bf16_bytes)) {
        set_error("SQ8_0 gfx942 B control byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (activation_ocp_buffer->bytes < activation_elements ||
        activation_scale_f32_buffer->bytes < activation_scale_bytes ||
        weight_ocp_buffer->bytes < weight_elements ||
        weight_scale_f32_buffer->bytes < weight_scale_bytes ||
        activation_bf16_buffer->bytes < activation_bf16_bytes ||
        weight_bf16_buffer->bytes < weight_bf16_bytes || output_f32_buffer->bytes < output_bytes) {
        set_error("SQ8_0 gfx942 B control buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    std::array<char, 512> helper_error{};
    if (!ullm_sq8_ck_gfx942_control_dequant_ocp_bf16_projection(
            activation_ocp_buffer->ptr,
            activation_scale_f32_buffer->ptr,
            weight_ocp_buffer->ptr,
            weight_scale_f32_buffer->ptr,
            m,
            n,
            k,
            activation_bf16_buffer->ptr,
            weight_bf16_buffer->ptr,
            output_f32_buffer->ptr,
            stream == nullptr ? nullptr : stream->stream,
            output_f32_buffer->hip_device_id,
            helper_error.data(),
            helper_error.size())) {
        set_error(helper_error[0] == '\0' ? "SQ8_0 gfx942 B control projection failed"
                                           : helper_error.data());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
#endif
}

extern "C" ullm_status ullm_runtime_sq8_ck_gfx942_aprime_fragment_probe_fnuz(
    const ullm_runtime_buffer* a_fnuz_16x32_row_major_buffer,
    const ullm_runtime_buffer* b_fnuz_32x16_column_major_buffer,
    ullm_runtime_buffer* matrix_f32_16x16_buffer,
    ullm_runtime_buffer* fragment_f32_lane64x4_buffer,
    ullm_runtime_stream* stream) {
    if (a_fnuz_16x32_row_major_buffer == nullptr || b_fnuz_32x16_column_major_buffer == nullptr ||
        matrix_f32_16x16_buffer == nullptr || fragment_f32_lane64x4_buffer == nullptr) {
        set_error("SQ8_0 gfx942 A′ fragment probe received a null buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
#if !defined(ULLM_RUNTIME_ROCM_CK_GFX942_APRIME)
    (void)stream;
    set_error("SQ8_0 gfx942 A′ fragment probe requires Cargo feature rocm-ck-gfx942-aprime");
    return ULLM_STATUS_RUNTIME_ERROR;
#else
    const std::array<const ullm_runtime_buffer*, 4> buffers = {
        a_fnuz_16x32_row_major_buffer,
        b_fnuz_32x16_column_major_buffer,
        matrix_f32_16x16_buffer,
        fragment_f32_lane64x4_buffer,
    };
    if (!sq8_gfx942_aprime_buffers_share_backend(buffers) ||
        !sq8_gfx942_aprime_buffers_have_unique_storage(buffers)) {
        set_error("SQ8_0 gfx942 A′ fragment probe buffers must be non-aliasing HIP storage on one device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(matrix_f32_16x16_buffer, stream) ||
        matrix_f32_16x16_buffer->backend != BackendKind::Hip) {
        set_error("SQ8_0 gfx942 A′ fragment probe requires a matching HIP stream and buffers");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    constexpr size_t kABytes = 16u * 32u;
    constexpr size_t kBBytes = 32u * 16u;
    constexpr size_t kMatrixBytes = 16u * 16u * sizeof(float);
    constexpr size_t kFragmentBytes = 64u * 4u * sizeof(float);
    if (a_fnuz_16x32_row_major_buffer->bytes < kABytes ||
        b_fnuz_32x16_column_major_buffer->bytes < kBBytes ||
        matrix_f32_16x16_buffer->bytes < kMatrixBytes ||
        fragment_f32_lane64x4_buffer->bytes < kFragmentBytes) {
        set_error("SQ8_0 gfx942 A′ fragment probe buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    std::array<char, 512> helper_error{};
    if (!ullm_sq8_ck_gfx942_aprime_fragment_probe_fnuz(
            a_fnuz_16x32_row_major_buffer->ptr,
            b_fnuz_32x16_column_major_buffer->ptr,
            matrix_f32_16x16_buffer->ptr,
            fragment_f32_lane64x4_buffer->ptr,
            stream == nullptr ? nullptr : stream->stream,
            matrix_f32_16x16_buffer->hip_device_id,
            helper_error.data(),
            helper_error.size())) {
        set_error(helper_error[0] == '\0' ? "SQ8_0 gfx942 A′ fragment probe failed"
                                           : helper_error.data());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
#endif
}
