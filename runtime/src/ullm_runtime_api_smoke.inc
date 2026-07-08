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
