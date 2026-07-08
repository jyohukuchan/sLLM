ullm_status ullm_runtime_smoke_add_f32(
    const float *lhs,
    const float *rhs,
    float *out,
    size_t count) {
    if (count > 0 && (lhs == nullptr || rhs == nullptr || out == nullptr)) {
        set_error("smoke add received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < count; ++i) {
        out[i] = lhs[i] + rhs[i];
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_wmma_fp8_probe(
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (output_buffer == nullptr) {
        set_error("WMMA FP8 probe received a null output buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < sizeof(uint32_t)) {
        set_error("WMMA FP8 probe output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("WMMA FP8 probe stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (output_buffer->backend == BackendKind::Cpu) {
        auto *output = static_cast<uint32_t *>(output_buffer->ptr);
        output[0] = 0x574d4d43u;
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (wmma_fp8_probe_hip_kernel(output_buffer, stream, &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    set_error(
        hip_kernel_error.empty() ? "WMMA FP8 probe HIP kernel is unavailable" :
                                   hip_kernel_error.c_str());
    return ULLM_STATUS_RUNTIME_ERROR;
}

ullm_status ullm_runtime_wmma_fp8_qk_probe(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    constexpr size_t kTileElements = 16 * 16;
    constexpr size_t kOutputBytes = kTileElements * sizeof(float);
    if (q_buffer == nullptr || k_buffer == nullptr || output_buffer == nullptr) {
        set_error("WMMA FP8 QK probe received a null buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_buffer->bytes < kTileElements || k_buffer->bytes < kTileElements) {
        set_error("WMMA FP8 QK probe input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < kOutputBytes) {
        set_error("WMMA FP8 QK probe output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_buffer, k_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error("WMMA FP8 QK probe buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(q_buffer, stream) ||
        !stream_matches_buffer(k_buffer, stream) ||
        !stream_matches_buffer(output_buffer, stream)) {
        set_error("WMMA FP8 QK probe stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (output_buffer->backend == BackendKind::Cpu) {
        wmma_fp8_qk_probe_host(
            static_cast<const uint8_t *>(q_buffer->ptr),
            static_cast<const uint8_t *>(k_buffer->ptr),
            static_cast<float *>(output_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (wmma_fp8_qk_probe_hip_kernel(q_buffer, k_buffer, output_buffer, stream, &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    set_error(
        hip_kernel_error.empty() ? "WMMA FP8 QK probe HIP kernel is unavailable" :
                                   hip_kernel_error.c_str());
    return ULLM_STATUS_RUNTIME_ERROR;
}

ullm_status ullm_runtime_rocwmma_fp8_qk_probe(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    constexpr size_t kTileElements = 16 * 16;
    constexpr size_t kOutputBytes = kTileElements * sizeof(float);
    if (q_buffer == nullptr || k_buffer == nullptr || output_buffer == nullptr) {
        set_error("rocWMMA FP8 QK probe received a null buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_buffer->bytes < kTileElements || k_buffer->bytes < kTileElements) {
        set_error("rocWMMA FP8 QK probe input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < kOutputBytes) {
        set_error("rocWMMA FP8 QK probe output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_buffer, k_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error("rocWMMA FP8 QK probe buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(q_buffer, stream) ||
        !stream_matches_buffer(k_buffer, stream) ||
        !stream_matches_buffer(output_buffer, stream)) {
        set_error("rocWMMA FP8 QK probe stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (output_buffer->backend == BackendKind::Cpu) {
        wmma_fp8_qk_probe_host(
            static_cast<const uint8_t *>(q_buffer->ptr),
            static_cast<const uint8_t *>(k_buffer->ptr),
            static_cast<float *>(output_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (rocwmma_fp8_qk_probe_hip_kernel(
            q_buffer,
            k_buffer,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    set_error(
        hip_kernel_error.empty() ? "rocWMMA FP8 QK probe HIP kernel is unavailable" :
                                   hip_kernel_error.c_str());
    return ULLM_STATUS_RUNTIME_ERROR;
}
