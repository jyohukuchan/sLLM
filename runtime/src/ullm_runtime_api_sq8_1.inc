// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

// SQ8_1 has a distinct payload/scale ABI.  Any row compensation remains
// external metadata and must be applied by the caller after this matvec; it is
// never encoded into the signed-I8 payload or the per-K32 F16 scale plane.
static ullm_status ullm_runtime_sq8_1_matvec_f32_impl(
    bool w8a8,
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    size_t payload_row_stride,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (payload_buffer == nullptr || scale_buffer == nullptr || input_buffer == nullptr ||
        output_buffer == nullptr) {
        set_error("SQ8_1 matvec received a null buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0u || cols == 0u) {
        set_error("SQ8_1 matvec rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(payload_buffer, scale_buffer) ||
        !buffers_share_backend(payload_buffer, input_buffer) ||
        !buffers_share_backend(payload_buffer, output_buffer)) {
        set_error("SQ8_1 matvec buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("SQ8_1 matvec stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    size_t expected_stride = 0u;
    if (!sq8_1_payload_row_stride(cols, &expected_stride) || payload_row_stride != expected_stride) {
        set_error("SQ8_1 payload row stride must equal round_up(cols, 16)");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t groups = sq8_1_groups_per_row(cols);
    if (groups == 0u || rows > max_size / expected_stride || rows > max_size / groups ||
        cols > max_size / sizeof(float) || rows > max_size / sizeof(float)) {
        set_error("SQ8_1 matvec shape byte count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_payload_bytes = rows * expected_stride;
    const size_t scale_values = rows * groups;
    if (scale_values > max_size / sizeof(uint16_t)) {
        set_error("SQ8_1 scale byte count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_scale_bytes = scale_values * sizeof(uint16_t);
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t required_output_bytes = rows * sizeof(float);
    if (payload_buffer->bytes < required_payload_bytes || scale_buffer->bytes < required_scale_bytes ||
        input_buffer->bytes < required_input_bytes || output_buffer->bytes < required_output_bytes) {
        set_error("SQ8_1 matvec buffer is too small for its declared shape");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (payload_buffer->backend == BackendKind::Cpu) {
        const bool ok = w8a8
            ? sq8_1_matvec_w8a8_explicit_f32_host(
                  static_cast<const uint8_t *>(payload_buffer->ptr),
                  static_cast<const uint16_t *>(scale_buffer->ptr),
                  static_cast<const float *>(input_buffer->ptr),
                  rows,
                  cols,
                  payload_row_stride,
                  static_cast<float *>(output_buffer->ptr))
            : sq8_1_matvec_w8a16_f32_host(
                  static_cast<const uint8_t *>(payload_buffer->ptr),
                  static_cast<const uint16_t *>(scale_buffer->ptr),
                  static_cast<const float *>(input_buffer->ptr),
                  rows,
                  cols,
                  payload_row_stride,
                  static_cast<float *>(output_buffer->ptr));
        if (!ok) {
            set_error("SQ8_1 CPU reference rejected malformed payload, scale, or activation");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_error;
    if (!sq8_1_matvec_f32_hip_kernel(
            w8a8,
            payload_buffer,
            scale_buffer,
            input_buffer,
            rows,
            cols,
            payload_row_stride,
            output_buffer,
            stream,
            &hip_error)) {
        set_error(
            hip_error.empty() ? (w8a8 ? "SQ8_1 explicit W8A8 HIP kernel is unavailable"
                                      : "SQ8_1 W8A16 HIP kernel is unavailable")
                              : hip_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_sq8_1_matvec_w8a16_f32(
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    size_t payload_row_stride,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    return ullm_runtime_sq8_1_matvec_f32_impl(
        false,
        payload_buffer,
        scale_buffer,
        input_buffer,
        rows,
        cols,
        payload_row_stride,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_sq8_1_matvec_w8a8_explicit_f32(
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    size_t payload_row_stride,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    return ullm_runtime_sq8_1_matvec_f32_impl(
        true,
        payload_buffer,
        scale_buffer,
        input_buffer,
        rows,
        cols,
        payload_row_stride,
        output_buffer,
        stream);
}
