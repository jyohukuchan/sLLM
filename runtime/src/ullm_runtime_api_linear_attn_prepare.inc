ullm_status ullm_runtime_linear_attn_qkv_prepare_f32(
    const ullm_runtime_buffer *qkv_buffer,
    const ullm_runtime_buffer *conv_weight_buffer,
    ullm_runtime_buffer *conv_history_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    float q_scale,
    int qk_l2_norm,
    ullm_runtime_buffer *conv_output_buffer,
    ullm_runtime_buffer *q_output_buffer,
    ullm_runtime_buffer *k_output_buffer,
    ullm_runtime_buffer *v_output_buffer,
    ullm_runtime_stream *stream) {
    if (qkv_buffer == nullptr || conv_weight_buffer == nullptr ||
        conv_history_buffer == nullptr || conv_output_buffer == nullptr ||
        q_output_buffer == nullptr || k_output_buffer == nullptr || v_output_buffer == nullptr) {
        set_error("linear attention qkv prepare received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (key_heads == 0 || value_heads == 0 || key_dim == 0 || value_dim == 0 ||
        kernel_size == 0) {
        set_error("linear attention qkv prepare dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(q_scale)) {
        set_error("linear attention qkv prepare q_scale must be finite");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(qkv_buffer, conv_weight_buffer) ||
        !buffers_share_backend(qkv_buffer, conv_history_buffer) ||
        !buffers_share_backend(qkv_buffer, conv_output_buffer) ||
        !buffers_share_backend(qkv_buffer, q_output_buffer) ||
        !buffers_share_backend(qkv_buffer, k_output_buffer) ||
        !buffers_share_backend(qkv_buffer, v_output_buffer)) {
        set_error("linear attention qkv prepare buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(conv_output_buffer, stream) ||
        !stream_matches_buffer(q_output_buffer, stream) ||
        !stream_matches_buffer(k_output_buffer, stream) ||
        !stream_matches_buffer(v_output_buffer, stream)) {
        set_error("linear attention qkv prepare stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (key_heads > max_size / key_dim || value_heads > max_size / value_dim) {
        set_error("linear attention qkv prepare element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = key_heads * key_dim;
    const size_t k_elements = q_elements;
    const size_t v_elements = value_heads * value_dim;
    if (q_elements > max_size - k_elements || q_elements + k_elements > max_size - v_elements) {
        set_error("linear attention qkv prepare channel count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t channels = q_elements + k_elements + v_elements;
    if (channels > max_size / kernel_size ||
        channels > max_size / sizeof(float) ||
        q_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float)) {
        set_error("linear attention qkv prepare byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t qkv_bytes = channels * sizeof(float);
    const size_t conv_history_bytes = channels * kernel_size * sizeof(float);
    const size_t q_bytes = q_elements * sizeof(float);
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    if (qkv_buffer->bytes < qkv_bytes) {
        set_error("linear attention qkv prepare qkv buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (conv_weight_buffer->bytes < conv_history_bytes) {
        set_error("linear attention qkv prepare conv weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (conv_history_buffer->bytes < conv_history_bytes) {
        set_error("linear attention qkv prepare conv history buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (conv_output_buffer->bytes < qkv_bytes) {
        set_error("linear attention qkv prepare conv output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_output_buffer->bytes < q_bytes || k_output_buffer->bytes < k_bytes ||
        v_output_buffer->bytes < v_bytes) {
        set_error("linear attention qkv prepare output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (qkv_buffer->backend == BackendKind::Cpu) {
        linear_attn_qkv_prepare_f32_host(
            static_cast<const float *>(qkv_buffer->ptr),
            static_cast<const float *>(conv_weight_buffer->ptr),
            static_cast<float *>(conv_history_buffer->ptr),
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            q_scale,
            qk_l2_norm != 0,
            static_cast<float *>(conv_output_buffer->ptr),
            static_cast<float *>(q_output_buffer->ptr),
            static_cast<float *>(k_output_buffer->ptr),
            static_cast<float *>(v_output_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (linear_attn_qkv_prepare_f32_hip_kernel(
            qkv_buffer,
            conv_weight_buffer,
            conv_history_buffer,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            q_scale,
            qk_l2_norm != 0,
            conv_output_buffer,
            q_output_buffer,
            k_output_buffer,
            v_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "linear attention qkv prepare HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return linear_attn_qkv_prepare_f32_hip_staging(
        qkv_buffer,
        conv_weight_buffer,
        conv_history_buffer,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        kernel_size,
        q_scale,
        qk_l2_norm != 0,
        qkv_bytes,
        conv_history_bytes,
        q_bytes,
        k_bytes,
        v_bytes,
        conv_output_buffer,
        q_output_buffer,
        k_output_buffer,
        v_output_buffer,
        stream);
}

ullm_status ullm_runtime_linear_attn_qkv_prepare_batch_f32(
    const ullm_runtime_buffer *qkv_buffer,
    const ullm_runtime_buffer *conv_weight_buffer,
    ullm_runtime_buffer *conv_history_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    size_t sequence_len,
    float q_scale,
    int qk_l2_norm,
    ullm_runtime_buffer *conv_output_buffer,
    ullm_runtime_buffer *q_output_buffer,
    ullm_runtime_buffer *k_output_buffer,
    ullm_runtime_buffer *v_output_buffer,
    ullm_runtime_stream *stream) {
    if (qkv_buffer == nullptr || conv_weight_buffer == nullptr ||
        conv_history_buffer == nullptr || conv_output_buffer == nullptr ||
        q_output_buffer == nullptr || k_output_buffer == nullptr || v_output_buffer == nullptr) {
        set_error("linear attention qkv prepare batch received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (key_heads == 0 || value_heads == 0 || key_dim == 0 || value_dim == 0 ||
        kernel_size == 0 || sequence_len == 0) {
        set_error("linear attention qkv prepare batch dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(q_scale)) {
        set_error("linear attention qkv prepare batch q_scale must be finite");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(qkv_buffer, conv_weight_buffer) ||
        !buffers_share_backend(qkv_buffer, conv_history_buffer) ||
        !buffers_share_backend(qkv_buffer, conv_output_buffer) ||
        !buffers_share_backend(qkv_buffer, q_output_buffer) ||
        !buffers_share_backend(qkv_buffer, k_output_buffer) ||
        !buffers_share_backend(qkv_buffer, v_output_buffer)) {
        set_error("linear attention qkv prepare batch buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(conv_output_buffer, stream) ||
        !stream_matches_buffer(q_output_buffer, stream) ||
        !stream_matches_buffer(k_output_buffer, stream) ||
        !stream_matches_buffer(v_output_buffer, stream)) {
        set_error("linear attention qkv prepare batch stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (key_heads > max_size / key_dim || value_heads > max_size / value_dim) {
        set_error("linear attention qkv prepare batch element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = key_heads * key_dim;
    const size_t k_elements = q_elements;
    const size_t v_elements = value_heads * value_dim;
    if (q_elements > max_size - k_elements || q_elements + k_elements > max_size - v_elements) {
        set_error("linear attention qkv prepare batch channel count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t channels = q_elements + k_elements + v_elements;
    if (channels > max_size / kernel_size ||
        channels > max_size / sequence_len ||
        q_elements > max_size / sequence_len ||
        k_elements > max_size / sequence_len ||
        v_elements > max_size / sequence_len) {
        set_error("linear attention qkv prepare batch element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t qkv_elements = channels * sequence_len;
    const size_t q_output_elements = q_elements * sequence_len;
    const size_t k_output_elements = k_elements * sequence_len;
    const size_t v_output_elements = v_elements * sequence_len;
    if (qkv_elements > max_size / sizeof(float) ||
        channels > max_size / kernel_size ||
        channels * kernel_size > max_size / sizeof(float) ||
        q_output_elements > max_size / sizeof(float) ||
        k_output_elements > max_size / sizeof(float) ||
        v_output_elements > max_size / sizeof(float)) {
        set_error("linear attention qkv prepare batch byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t qkv_bytes = qkv_elements * sizeof(float);
    const size_t conv_history_bytes = channels * kernel_size * sizeof(float);
    const size_t q_bytes = q_output_elements * sizeof(float);
    const size_t k_bytes = k_output_elements * sizeof(float);
    const size_t v_bytes = v_output_elements * sizeof(float);
    if (qkv_buffer->bytes < qkv_bytes) {
        set_error("linear attention qkv prepare batch qkv buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (conv_weight_buffer->bytes < conv_history_bytes) {
        set_error("linear attention qkv prepare batch conv weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (conv_history_buffer->bytes < conv_history_bytes) {
        set_error("linear attention qkv prepare batch conv history buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (conv_output_buffer->bytes < qkv_bytes) {
        set_error("linear attention qkv prepare batch conv output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_output_buffer->bytes < q_bytes || k_output_buffer->bytes < k_bytes ||
        v_output_buffer->bytes < v_bytes) {
        set_error("linear attention qkv prepare batch output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (qkv_buffer->backend == BackendKind::Cpu) {
        linear_attn_qkv_prepare_batch_f32_host(
            static_cast<const float *>(qkv_buffer->ptr),
            static_cast<const float *>(conv_weight_buffer->ptr),
            static_cast<float *>(conv_history_buffer->ptr),
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            sequence_len,
            q_scale,
            qk_l2_norm != 0,
            static_cast<float *>(conv_output_buffer->ptr),
            static_cast<float *>(q_output_buffer->ptr),
            static_cast<float *>(k_output_buffer->ptr),
            static_cast<float *>(v_output_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (linear_attn_qkv_prepare_batch_f32_hip_kernel(
            qkv_buffer,
            conv_weight_buffer,
            conv_history_buffer,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            sequence_len,
            q_scale,
            qk_l2_norm != 0,
            conv_output_buffer,
            q_output_buffer,
            k_output_buffer,
            v_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL") != nullptr ||
        std::getenv("ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "linear attention qkv prepare batch HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return linear_attn_qkv_prepare_batch_f32_hip_staging(
        qkv_buffer,
        conv_weight_buffer,
        conv_history_buffer,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        kernel_size,
        sequence_len,
        q_scale,
        qk_l2_norm != 0,
        qkv_bytes,
        conv_history_bytes,
        q_bytes,
        k_bytes,
        v_bytes,
        conv_output_buffer,
        q_output_buffer,
        k_output_buffer,
        v_output_buffer,
        stream);
}
